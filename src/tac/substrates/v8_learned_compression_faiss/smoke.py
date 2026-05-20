# SPDX-License-Identifier: MIT
"""Deterministic local CPU smoke helpers for V8 learned-compression Faiss.

This module is intentionally a non-promotional training/export smoke. It
materializes the four previously dead surfaces together:

* categorical posterior logits/codewords,
* a positive scale hyperprior,
* byte-closed export payload bytes plus a parseable manifest,
* an eval-roundtrip-aware proxy hook that does not load scorers.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from tac.differentiable_eval_roundtrip import apply_eval_roundtrip_during_training

V8_EXPORT_MAGIC = b"V8FEXP01"
V8_EXPORT_SCHEMA = "v8_learned_compression_faiss_export_smoke_v1"
RATE_DENOMINATOR_BYTES = 37_545_489.0


@dataclass(frozen=True)
class V8LocalSmokeConfig:
    """Configuration for the deterministic local V8 export smoke."""

    num_pairs: int = 8
    latent_dim: int = 28
    hidden_dim: int = 32
    categorical_groups: int = 8
    codebook_size: int = 16
    beta: float = 0.001
    temperature_start: float = 1.0
    temperature_end: float = 0.25
    seed: int = 20260520
    eval_roundtrip_resize: bool = False

    def validate(self) -> None:
        if self.num_pairs <= 0:
            raise ValueError("num_pairs must be positive")
        if self.latent_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("latent_dim and hidden_dim must be positive")
        if self.categorical_groups <= 0:
            raise ValueError("categorical_groups must be positive")
        if self.codebook_size <= 1:
            raise ValueError("codebook_size must be greater than 1")
        if self.temperature_start <= 0 or self.temperature_end <= 0:
            raise ValueError("temperatures must be positive")
        if self.beta < 0:
            raise ValueError("beta must be nonnegative")

    def as_dict(self) -> dict[str, Any]:
        return {
            "num_pairs": self.num_pairs,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "categorical_groups": self.categorical_groups,
            "codebook_size": self.codebook_size,
            "beta": self.beta,
            "temperature_start": self.temperature_start,
            "temperature_end": self.temperature_end,
            "seed": self.seed,
            "eval_roundtrip_resize": self.eval_roundtrip_resize,
        }


class TinyV8LearnedCompressionModel(torch.nn.Module):
    """Small deterministic model for local smoke export only."""

    def __init__(self, config: V8LocalSmokeConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(config.latent_dim, config.hidden_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(
                config.hidden_dim,
                config.categorical_groups * config.codebook_size,
            ),
        )
        self.scale_hyperprior = torch.nn.Linear(config.latent_dim, config.categorical_groups * 2)
        self.decoder_codebook = torch.nn.Parameter(
            torch.empty(config.categorical_groups, config.codebook_size, config.latent_dim)
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.config.seed)
        for module in self.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight, generator=generator)
                torch.nn.init.uniform_(module.bias, -0.02, 0.02, generator=generator)
        torch.nn.init.normal_(self.decoder_codebook, mean=0.0, std=0.08, generator=generator)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        cfg = self.config
        logits = self.encoder(features).reshape(
            features.shape[0], cfg.categorical_groups, cfg.codebook_size
        )
        temperature = cfg.temperature_end
        probs = torch.softmax(logits / temperature, dim=-1)
        codewords = probs.argmax(dim=-1)
        hard = F.one_hot(codewords, num_classes=cfg.codebook_size).to(dtype=probs.dtype)
        straight_through = hard + probs - probs.detach()
        decoded_groups = torch.einsum("ngk,gkd->ngd", straight_through, self.decoder_codebook)
        reconstruction = decoded_groups.mean(dim=1)
        hyper = self.scale_hyperprior(features).reshape(features.shape[0], cfg.categorical_groups, 2)
        hyper_means = hyper[..., 0]
        hyper_scales = F.softplus(hyper[..., 1]) + 1.0e-3
        return {
            "logits": logits,
            "probs": probs,
            "codewords": codewords,
            "hyper_means": hyper_means,
            "hyper_scales": hyper_scales,
            "reconstruction": reconstruction,
        }


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _tensor_f32_bytes(tensor: torch.Tensor) -> bytes:
    return (
        tensor.detach()
        .cpu()
        .contiguous()
        .numpy()
        .astype("<f4", copy=False)
        .tobytes()
    )


def _codeword_bytes(codewords: torch.Tensor, codebook_size: int) -> tuple[bytes, str]:
    arr = codewords.detach().cpu().contiguous().numpy()
    if codebook_size <= 256:
        return arr.astype("u1", copy=False).tobytes(), "uint8"
    return arr.astype("<u2", copy=False).tobytes(), "uint16_le"


def _quantized_i8_blob(tensor: torch.Tensor) -> tuple[bytes, dict[str, Any]]:
    t = tensor.detach().cpu().contiguous()
    max_abs = float(t.abs().max().item())
    scale = max(max_abs / 127.0, 1.0e-8)
    q = torch.clamp(torch.round(t / scale), -127, 127).to(torch.int8)
    blob = q.numpy().astype("i1", copy=False).tobytes()
    return blob, {
        "dtype": "int8",
        "shape": list(t.shape),
        "symmetric_scale": scale,
        "raw_f32_sha256": _sha256_bytes(_tensor_f32_bytes(t)),
    }


def generate_synthetic_features(config: V8LocalSmokeConfig) -> torch.Tensor:
    """Generate deterministic local features without loading contest scorers."""

    config.validate()
    pair = torch.arange(config.num_pairs, dtype=torch.float32).unsqueeze(1)
    dim = torch.arange(config.latent_dim, dtype=torch.float32).unsqueeze(0)
    base = torch.sin((pair + 1.0) * (dim + 1.0) / 11.0)
    trend = torch.cos((pair + 3.0) * (dim + 5.0) / 17.0) * 0.25
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config.seed + 17)
    noise = torch.randn(config.num_pairs, config.latent_dim, generator=generator) * 0.01
    return (base + trend + noise).to(torch.float32)


def load_feature_json(path: str | Path, config: V8LocalSmokeConfig) -> torch.Tensor:
    """Load supplied local V8 features from JSON and fail closed on mismatch."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"V8 feature JSON not found: {source}")
    raw = json.loads(source.read_text(encoding="utf-8"))
    features = raw.get("features") if isinstance(raw, dict) else raw
    tensor = torch.tensor(features, dtype=torch.float32)
    expected = (config.num_pairs, config.latent_dim)
    if tuple(tensor.shape) != expected:
        raise ValueError(f"feature JSON shape {tuple(tensor.shape)} does not match {expected}")
    if not torch.isfinite(tensor).all():
        raise ValueError("feature JSON contains non-finite values")
    return tensor


def _empirical_code_entropy_bits(codewords: torch.Tensor, codebook_size: int) -> float:
    total = 0.0
    groups = codewords.shape[1]
    for group_idx in range(groups):
        counts = torch.bincount(codewords[:, group_idx], minlength=codebook_size).to(torch.float64)
        probs = counts / counts.sum().clamp_min(1.0)
        nz = probs[probs > 0]
        total += float(-(nz * torch.log2(nz)).sum().item())
    return total / max(groups, 1)


def _eval_roundtrip_proxy(features: torch.Tensor, config: V8LocalSmokeConfig) -> dict[str, Any]:
    rgb_vec = torch.sigmoid(features[:, :3]) * 255.0
    rgb = rgb_vec.reshape(features.shape[0], 3, 1, 1).expand(features.shape[0], 3, 4, 4)
    rounded = apply_eval_roundtrip_during_training(
        rgb,
        simulate_uint8=True,
        simulate_resize=config.eval_roundtrip_resize,
        ste_round=True,
    )
    return {
        "helper": "tac.differentiable_eval_roundtrip.apply_eval_roundtrip_during_training",
        "enabled": True,
        "simulate_uint8": True,
        "simulate_resize": config.eval_roundtrip_resize,
        "scorer_loads": False,
        "l1_delta": float((rounded - rgb).abs().mean().item()),
    }


def _build_payload(
    *,
    config: V8LocalSmokeConfig,
    model: TinyV8LearnedCompressionModel,
    result: dict[str, torch.Tensor],
    feature_source: dict[str, Any],
    metrics: dict[str, Any],
) -> tuple[bytes, dict[str, Any]]:
    blobs: list[tuple[str, bytes, dict[str, Any]]] = []

    code_blob, code_dtype = _codeword_bytes(result["codewords"], config.codebook_size)
    blobs.append(
        (
            "categorical_codewords",
            code_blob,
            {
                "dtype": code_dtype,
                "shape": list(result["codewords"].shape),
                "role": "categorical_posterior_codeword_stream",
            },
        )
    )
    for name, tensor in (
        ("posterior_logits", result["logits"]),
        ("scale_hyperprior_means", result["hyper_means"]),
        ("scale_hyperprior_scales", result["hyper_scales"]),
    ):
        blob = _tensor_f32_bytes(tensor)
        blobs.append((name, blob, {"dtype": "float32_le", "shape": list(tensor.shape)}))

    for name, tensor in model.state_dict().items():
        blob, meta = _quantized_i8_blob(tensor)
        blobs.append((f"model_i8_{name}", blob, meta))

    offset = 0
    blob_entries: list[dict[str, Any]] = []
    blob_payload_parts: list[bytes] = []
    for name, blob, meta in blobs:
        blob_entries.append(
            {
                "name": name,
                "offset": offset,
                "length": len(blob),
                "sha256": _sha256_bytes(blob),
                **meta,
            }
        )
        blob_payload_parts.append(blob)
        offset += len(blob)

    header = {
        "schema": V8_EXPORT_SCHEMA,
        "config": config.as_dict(),
        "feature_source": feature_source,
        "blob_payload_bytes": offset,
        "blobs": blob_entries,
        "metrics": metrics,
        "custody": {
            "research_only": True,
            "dispatch_enabled": False,
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "axis_label": "[diagnostic-CPU local export smoke; no scorer load]",
            "no_network": True,
            "inflate_time_scorer_loads": False,
        },
    }
    header_bytes = _json_bytes(header)
    payload = V8_EXPORT_MAGIC + struct.pack("<I", len(header_bytes)) + header_bytes + b"".join(
        blob_payload_parts
    )
    header["payload_sha256"] = _sha256_bytes(payload)
    header["payload_bytes"] = len(payload)
    return payload, header


def parse_v8_export_payload(raw: bytes) -> dict[str, Any]:
    """Parse and validate the byte-closed local V8 export payload."""

    if len(raw) < len(V8_EXPORT_MAGIC) + 4:
        raise ValueError("V8 export payload too short")
    if raw[: len(V8_EXPORT_MAGIC)] != V8_EXPORT_MAGIC:
        raise ValueError("V8 export payload magic mismatch")
    header_len = struct.unpack("<I", raw[len(V8_EXPORT_MAGIC) : len(V8_EXPORT_MAGIC) + 4])[0]
    header_start = len(V8_EXPORT_MAGIC) + 4
    header_end = header_start + header_len
    if header_end > len(raw):
        raise ValueError("V8 export header length exceeds payload")
    header = json.loads(raw[header_start:header_end].decode("utf-8"))
    blob_payload = raw[header_end:]
    if len(blob_payload) != header.get("blob_payload_bytes"):
        raise ValueError("V8 export blob payload length mismatch")
    for blob in header.get("blobs", []):
        start = int(blob["offset"])
        end = start + int(blob["length"])
        if start < 0 or end > len(blob_payload):
            raise ValueError(f"V8 export blob {blob.get('name')} is out of range")
        if _sha256_bytes(blob_payload[start:end]) != blob["sha256"]:
            raise ValueError(f"V8 export blob {blob.get('name')} sha256 mismatch")
    return header


def run_local_cpu_export_smoke(
    config: V8LocalSmokeConfig,
    *,
    features: torch.Tensor | None = None,
    feature_source: dict[str, Any] | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Run deterministic CPU-only V8 smoke and return payload plus manifest."""

    config.validate()
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    feature_source = feature_source or {"kind": "synthetic", "score_claim": False}
    features = generate_synthetic_features(config) if features is None else features.to(torch.float32)
    expected = (config.num_pairs, config.latent_dim)
    if tuple(features.shape) != expected:
        raise ValueError(f"features shape {tuple(features.shape)} does not match {expected}")
    if not torch.isfinite(features).all():
        raise ValueError("features contain non-finite values")

    model = TinyV8LearnedCompressionModel(config).cpu().eval()
    with torch.no_grad():
        result = model(features.cpu())

    probs = result["probs"].clamp_min(1.0e-12)
    posterior_entropy_bits = float((-(probs * torch.log2(probs)).sum(dim=-1)).mean().item())
    empirical_entropy_bits = _empirical_code_entropy_bits(result["codewords"], config.codebook_size)
    mse = float(F.mse_loss(result["reconstruction"], features.cpu()).item())
    code_bits = (
        config.num_pairs
        * config.categorical_groups
        * max(1, math.ceil(math.log2(config.codebook_size)))
    )
    scale_bytes = int(result["hyper_means"].numel() * 4 + result["hyper_scales"].numel() * 4)
    eval_hook = _eval_roundtrip_proxy(result["reconstruction"], config)
    metrics = {
        "proxy_reconstruction_mse": mse,
        "proxy_rate_distortion_loss": mse + config.beta * (code_bits / max(config.num_pairs, 1)),
        "posterior_entropy_bits_per_group": posterior_entropy_bits,
        "empirical_code_entropy_bits_per_group": empirical_entropy_bits,
        "categorical_code_bits_uncompressed": code_bits,
        "scale_hyperprior_bytes_float32": scale_bytes,
        "scale_hyperprior_scale_min": float(result["hyper_scales"].min().item()),
        "scale_hyperprior_scale_max": float(result["hyper_scales"].max().item()),
        "eval_roundtrip_hook": eval_hook,
        "score_claim": False,
        "promotion_eligible": False,
    }
    payload, header = _build_payload(
        config=config,
        model=model,
        result=result,
        feature_source=feature_source,
        metrics=metrics,
    )
    parse_v8_export_payload(payload)
    return payload, header


def write_v8_export_files(
    output_dir: str | Path,
    *,
    payload: bytes,
    header: dict[str, Any],
    payload_name: str = "v8_byte_closed_export.bin",
) -> dict[str, Any]:
    """Write payload and manifest, returning the file-level custody record."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    payload_path = out / payload_name
    payload_path.write_bytes(payload)
    payload_sha = _sha256_bytes(payload)
    manifest = {
        "schema": "v8_learned_compression_faiss_local_export_manifest_v1",
        "payload_path": str(payload_path),
        "payload_bytes": len(payload),
        "payload_sha256": payload_sha,
        "parsed_header": header,
        "research_only": True,
        "dispatch_enabled": False,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "axis_label": "[diagnostic-CPU local export smoke; no scorer load]",
        "required_next_handoff": "Worker A submission runtime may consume this payload grammar; this smoke does not edit submissions/v8.",
    }
    manifest_path = out / "v8_byte_closed_export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "payload_path": str(payload_path),
        "manifest_path": str(manifest_path),
        "payload_bytes": len(payload),
        "payload_sha256": payload_sha,
    }
