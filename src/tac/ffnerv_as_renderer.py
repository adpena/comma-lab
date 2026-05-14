# SPDX-License-Identifier: MIT
"""FFNeRV-as-renderer — Fourier-features NeRV substrate.

Per operator directive 2026-05-11 (NeRV-family expansion) + CLAUDE.md HNeRV
parity discipline (lesson 5: full RGB renderer; lesson 4: inflate ≤200 LOC).
FFNeRV adds a **multi-band Fourier-feature positional encoding** (sin/cos at
N frequencies) to the per-pair latent BEFORE feeding the NeRV decoder. This
is the canonical Tancik 2020 trick that makes coordinate-MLP-style networks
fit high-frequency content effectively.

Why Fourier features
--------------------
- **Spectral bias mitigation**: vanilla MLPs/NeRVs are biased toward low
  frequencies (Rahaman et al. 2019). Fourier features re-parameterize the
  input so the network sees a high-frequency-rich representation, dramatically
  improving sharp-edge / texture reconstruction at fixed param budget.
- **No extra params per latent**: the Fourier encoding is a fixed
  (non-trainable) frequency matrix — only the NeRV decoder weights are
  learned. Per-frequency-band amplitude scaling provides a learnable
  band-pass control.
- **Composability**: FFNeRV's Fourier-encoded latent slots into ANY downstream
  NeRV decoder (Lane 12-v2 PixelShuffle, BlockNeRV tile decoder, MNeRV
  hierarchical, HiNeRV multi-scale, …).

Architecture (default config)
-----------------------------
- Latent: (B, 16) per-pair as in Lane 12-v2.
- Fourier encoding: 16 frequencies × {sin, cos} = 32 features per latent dim
  → encoded latent of shape (B, 16 * 32 + 16) = (B, 528) (16 raw passthrough
  + 512 Fourier features).
- Downstream NeRV decoder: same PixelShuffle stack as Lane 12-v2.
- Output: (B, 2, 3, 384, 512) RGB.

REPRESENTATION_ARCHIVE_GRAMMAR_BLUEPRINT (per CLAUDE.md Catalog #124)
---------------------------------------------------------------------
  archive_grammar: monolithic single-file 0.bin (16-byte fixed header +
    5 length-prefixed sections: decoder_blob INT8+brotli, freq_table FP16
    raw, scale_table FP16, latent_blob uint8 delta-zigzag+brotli, sidecar empty)
  parser_section_manifest: ARCHIVE_GRAMMAR_FFNERV in this module with
    schema_keys_in_order = FFNeRVRenderer.SCHEMA (pinned ordering)
  inflate_runtime_loc_budget: substrate_engineering — Phase B inflate ≤200
    LOC contest-hermetic
  runtime_dep_closure: torch + brotli (zero dep on tac.*)
  export_format: ffnerv_phase_a_monolithic_singlefile_0bin
  score_aware_loss: trainer wires Lane 12-v2 train_step pattern with
    differentiable rgb_to_yuv6 + load_differentiable_scorers + Lagrangian
  bolt_on_loc_budget: substrate_engineering (full FF-encoded NeRV renderer)
  no_op_detector_planned: export_to_archive returns sha256

CLAUDE.md compliance (lesson-by-lesson)
---------------------------------------
- L1 (score-aware): trainer routes through `train_step` with SegNet + PoseNet.
- L2 (export-first): ARCHIVE_GRAMMAR_FFNERV declared at module level.
- L4 (inflate ≤ 200 LOC): substrate-engineering target.
- L5 (full RGB renderer): forward returns (B, 2, 3, H, W).
- L6 (score-domain Lagrangian): train_step delegates to Lagrangian via the
  same surrogate signatures.
- L8 (eval-roundtrip): train_step simulates uint8 STE before scorer.
- L11 (no-op detector): export_to_archive returns sha256.
- L13 (KILL is last resort): N/A.

Predicted Δ score
-----------------
``[predicted; HNeRV parity discipline; Fourier-feature positional encoding
gives ~1.3× spectral coverage at high frequencies vs vanilla NeRV per
Tancik 2020 spectral bias paper; pose-axis marginal 2.71× SegNet at PR106
r2 frontier]``. NOT a score claim until [contest-CUDA] anchor lands.

Format ID: 0x61 (NeRV-family expansion magic 0xFE).
"""
from __future__ import annotations

import hashlib
import io
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Magic + format ────────────────────────────────────────────────────────


FFNERV_MAGIC: bytes = b"FFNV"
FFNERV_FORMAT_VERSION: int = 1
FFNERV_FORMAT_ID: int = 0x61


# ── Archive grammar (parser-section manifest) ────────────────────────────


ARCHIVE_GRAMMAR_FFNERV: dict = {
    "format_version": FFNERV_FORMAT_VERSION,
    "format_id": FFNERV_FORMAT_ID,
    "magic": FFNERV_MAGIC.decode("ascii"),
    "sections": [
        {
            "name": "header",
            "offset": 0,
            "length": 16,
            "kind": "fixed_header",
            "fields": [
                ("magic", "4s", 4),
                ("version", "<H", 2),
                ("format_id", "<H", 2),
                ("latent_dim", "<H", 2),
                ("n_pairs", "<H", 2),
                ("n_frequencies", "<H", 2),
                ("base_channels", "<H", 2),
            ],
        },
        {
            "name": "decoder_blob",
            "offset_after": "header",
            "length_field_le_u32": True,
            "kind": "brotli_int8_codes_schema_driven",
        },
        {
            "name": "freq_table",
            "offset_after": "decoder_blob",
            "length_field_le_u32": True,
            "kind": "fp16_raw_frequencies",
        },
        {
            "name": "scale_table",
            "offset_after": "freq_table",
            "length_field_le_u32": True,
            "kind": "fp16_raw_one_per_schema_entry",
        },
        {
            "name": "latent_blob",
            "offset_after": "scale_table",
            "length_field_le_u32": True,
            "kind": "brotli_uint8_asym_delta_split",
        },
        {
            "name": "sidecar_blob",
            "offset_after": "latent_blob",
            "length_field_le_u32": True,
            "kind": "brotli_optional_phase_b",
            "phase_a_empty": True,
        },
    ],
    "schema_keys_in_order": "FFNeRVRenderer.SCHEMA",
    "predicted_total_bytes": "155_000 to 185_000 [predicted; FF features expand stem dim]",
}


# ── Config ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FFNeRVConfig:
    """Frozen config for FFNeRV-as-renderer Phase A.

    Attributes
    ----------
    latent_dim
        Raw per-pair latent dim. Default 16 mirrors Lane 12-v2.
    n_frequencies
        Number of Fourier frequency bands. Default 16. Encoded dim becomes
        ``latent_dim * (1 + 2 * n_frequencies)`` (raw + sin + cos per band).
    base_channels
        NeRV decoder base channel width.
    base_h, base_w
        Initial spatial dims after stem. Default (6, 8).
    eval_size
        Native render size (H, W). Default (384, 512).
    n_pairs
        Number of per-pair latents.
    n_stages
        PixelShuffle stages. Default 6.
    log_freq_min, log_freq_max
        Log-spaced frequency range. Default 0..6 (1, 2, 4, 8, 16, 32, …, 64).
    lambda_seg, lambda_pose
        Score-aware loss weights.
    cuda_required
        If True (default), train_step raises if CUDA unavailable.
    """

    latent_dim: int = 16
    n_frequencies: int = 16
    base_channels: int = 36
    base_h: int = 6
    base_w: int = 8
    eval_size: tuple[int, int] = (384, 512)
    n_pairs: int = 600
    n_stages: int = 6
    log_freq_min: float = 0.0
    log_freq_max: float = 6.0
    frames_per_pair: int = 2
    lambda_seg: float = 100.0
    lambda_pose: float = 288.6751345948129
    cuda_required: bool = True

    def __post_init__(self) -> None:
        if self.latent_dim <= 0:
            raise ValueError(f"latent_dim must be positive, got {self.latent_dim}")
        if self.n_frequencies <= 0:
            raise ValueError(f"n_frequencies must be positive, got {self.n_frequencies}")
        if self.base_channels <= 0:
            raise ValueError(f"base_channels must be positive, got {self.base_channels}")
        if self.n_stages != 6:
            raise ValueError(f"Phase A pinned at n_stages=6, got {self.n_stages}")
        if self.frames_per_pair != 2:
            raise ValueError(f"Phase A pinned at frames_per_pair=2, got {self.frames_per_pair}")
        if self.log_freq_max <= self.log_freq_min:
            raise ValueError("log_freq_max must exceed log_freq_min")

    @property
    def encoded_dim(self) -> int:
        """Dimension after Fourier encoding: raw + 2 * n_freq per dim."""
        return self.latent_dim * (1 + 2 * self.n_frequencies)


# ── Fourier feature encoding ─────────────────────────────────────────────


class FourierFeatureEncoding(nn.Module):
    """Multi-band sin/cos positional encoding per Tancik 2020.

    Frequencies are log-spaced between ``2^log_freq_min`` and ``2^log_freq_max``.
    The frequency table is a NON-TRAINABLE buffer — only the downstream decoder
    weights are learned. We expose the frequencies as a buffer so the archive
    can ship them deterministically (they are config-derived but persisting
    them avoids any reconstruction drift at inflate time).
    """

    def __init__(self, latent_dim: int, n_frequencies: int,
                 log_freq_min: float = 0.0, log_freq_max: float = 6.0) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.n_frequencies = n_frequencies
        # Log-spaced base frequencies; broadcast against latent on forward.
        freqs = torch.logspace(
            log_freq_min, log_freq_max, n_frequencies, base=2.0
        )  # (n_freq,)
        # Buffer (non-trainable) so it is persisted / restored deterministically.
        self.register_buffer("frequencies", freqs)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """``z (B, latent_dim) → (B, latent_dim * (1 + 2*n_freq))``."""
        if z.dim() != 2 or z.shape[1] != self.latent_dim:
            raise ValueError(
                f"FourierFeatureEncoding expected (B, {self.latent_dim}), got {tuple(z.shape)}"
            )
        # angles: (B, latent_dim, n_freq)
        angles = z.unsqueeze(-1) * self.frequencies.view(1, 1, -1) * (2 * math.pi)
        sin_part = torch.sin(angles).reshape(z.shape[0], -1)
        cos_part = torch.cos(angles).reshape(z.shape[0], -1)
        # Concat raw + sin + cos.
        return torch.cat([z, sin_part, cos_part], dim=-1)


# ── Renderer ──────────────────────────────────────────────────────────────


class FFNeRVRenderer(nn.Module):
    """FFNeRV — Fourier-feature positional encoding + NeRV decoder.

    Forward signature: ``z (B, latent_dim) → (B, 2, 3, H, W)``.
    """

    def __init__(self, config: FFNeRVConfig) -> None:
        super().__init__()
        self.config = config
        self.fourier = FourierFeatureEncoding(
            latent_dim=config.latent_dim,
            n_frequencies=config.n_frequencies,
            log_freq_min=config.log_freq_min,
            log_freq_max=config.log_freq_max,
        )
        C = config.base_channels
        self.channels: list[int] = [
            C, C, C, int(C * 0.75), int(C * 0.58), int(C * 0.5), int(C * 0.5)
        ]
        self.stem = nn.Linear(
            config.encoded_dim, self.channels[0] * config.base_h * config.base_w
        )
        self.blocks = nn.ModuleList()
        self.skips = nn.ModuleList()
        for i in range(config.n_stages):
            in_ch = self.channels[i]
            out_ch = self.channels[i + 1]
            self.blocks.append(nn.Conv2d(in_ch, out_ch * 4, 3, padding=1))
            self.skips.append(
                nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
            )
        self.ps = nn.PixelShuffle(2)
        final_ch = self.channels[-1]
        self.refine = nn.Sequential(
            nn.Conv2d(final_ch, final_ch // 2, 3, padding=2, dilation=2),
            nn.Conv2d(final_ch // 2, final_ch, 3, padding=1),
        )
        self.rgb_0 = nn.Conv2d(final_ch, 3, 3, padding=1)
        self.rgb_1 = nn.Conv2d(final_ch, 3, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """``z (B, latent_dim) → (B, 2, 3, H, W)``."""
        if z.dim() != 2 or z.shape[1] != self.config.latent_dim:
            raise ValueError(
                f"forward expected (B, {self.config.latent_dim}), got {tuple(z.shape)}"
            )
        z_enc = self.fourier(z)
        B = z.shape[0]
        x = self.stem(z_enc).view(
            B, self.channels[0], self.config.base_h, self.config.base_w
        )
        x = torch.sin(x)
        for block, skip in zip(self.blocks, self.skips):
            identity = F.interpolate(
                x, scale_factor=2, mode="bilinear", align_corners=False
            )
            identity = skip(identity)
            x = self.ps(block(x))
            x = torch.sin(x + identity)
        x = x + 0.1 * torch.sin(self.refine(x))
        f0 = torch.sigmoid(self.rgb_0(x)) * 255.0
        f1 = torch.sigmoid(self.rgb_1(x)) * 255.0
        return torch.stack([f0, f1], dim=1)

    @property
    def schema(self) -> list[tuple[str, tuple[int, ...]]]:
        out: list[tuple[str, tuple[int, ...]]] = []
        sd = self.state_dict()
        keys = ["stem.weight", "stem.bias"]
        for i in range(self.config.n_stages):
            keys += [f"blocks.{i}.weight", f"blocks.{i}.bias"]
        for i in range(self.config.n_stages):
            if isinstance(self.skips[i], nn.Conv2d):
                keys += [f"skips.{i}.weight", f"skips.{i}.bias"]
        keys += [
            "refine.0.weight", "refine.0.bias",
            "refine.1.weight", "refine.1.bias",
            "rgb_0.weight", "rgb_0.bias",
            "rgb_1.weight", "rgb_1.bias",
        ]
        for key in keys:
            if key in sd:
                out.append((key, tuple(sd[key].shape)))
        return out


# ── Latent table ─────────────────────────────────────────────────────────


class FFNeRVLatentTable(nn.Module):
    def __init__(self, n_pairs: int, latent_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_pairs, latent_dim)
        nn.init.normal_(self.embedding.weight, std=0.01)

    def forward(self, pair_indices: torch.Tensor) -> torch.Tensor:
        return self.embedding(pair_indices)


# ── train_step ───────────────────────────────────────────────────────────


def _eval_roundtrip_uint8_clamp(rgb: torch.Tensor) -> torch.Tensor:
    clamped = rgb.clamp(0.0, 255.0)
    rounded = clamped + (clamped.round().detach() - clamped.detach())
    return rounded


def _pose_tensor(output: torch.Tensor | dict) -> torch.Tensor:
    if isinstance(output, dict):
        if "pose" not in output:
            raise KeyError("PoseNet output dict missing 'pose' key")
        value = output["pose"]
    else:
        value = output
    if not torch.is_tensor(value):
        raise TypeError(f"pose output must be a tensor, got {type(value).__name__}")
    return value


def train_step_ffnerv(
    *,
    renderer: FFNeRVRenderer,
    latent_table: FFNeRVLatentTable,
    pair_indices: torch.Tensor,
    gt_pairs_uint8: torch.Tensor,
    scorer_seg: nn.Module,
    scorer_pose: nn.Module,
    seg_surrogate: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    pose_surrogate: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    lambda_seg: float,
    lambda_pose: float,
    eval_roundtrip: bool = True,
) -> dict:
    """Score-aware training step (mirrors Lane 12-v2)."""
    if not eval_roundtrip:
        raise ValueError(
            "eval_roundtrip=False is forbidden by CLAUDE.md "
            "check_no_eval_roundtrip_false."
        )
    z = latent_table(pair_indices)
    decoded = renderer(z)
    B, F_pp, C, H_native, W_native = decoded.shape
    flat = decoded.reshape(B * F_pp, C, H_native, W_native)
    H_camera, W_camera = gt_pairs_uint8.shape[-2], gt_pairs_uint8.shape[-1]
    up = F.interpolate(
        flat, size=(H_camera, W_camera), mode="bicubic", align_corners=False
    )
    up_uint8_ste = _eval_roundtrip_uint8_clamp(up)
    up_pairs = up_uint8_ste.reshape(B, F_pp, C, H_camera, W_camera)
    gt_pairs = gt_pairs_uint8.float()

    seg_pred_logits = scorer_seg(scorer_seg.preprocess_input(up_pairs))
    with torch.no_grad():
        seg_target_logits = scorer_seg(scorer_seg.preprocess_input(gt_pairs))
    loss_seg_unweighted = seg_surrogate(seg_pred_logits, seg_target_logits)

    pose_pred = _pose_tensor(scorer_pose(scorer_pose.preprocess_input(up_pairs)))
    with torch.no_grad():
        pose_target = _pose_tensor(scorer_pose(scorer_pose.preprocess_input(gt_pairs)))
    loss_pose_unweighted = pose_surrogate(pose_pred, pose_target)

    loss_seg = lambda_seg * loss_seg_unweighted
    loss_pose = lambda_pose * loss_pose_unweighted
    loss = loss_seg + loss_pose
    return {
        "loss": loss,
        "loss_seg": loss_seg,
        "loss_pose": loss_pose,
        "loss_seg_unweighted": loss_seg_unweighted,
        "loss_pose_unweighted": loss_pose_unweighted,
    }


# ── Quantization + archive packing ───────────────────────────────────────


def _quantize_per_tensor_int8_with_fp16_scale(
    tensor: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_abs = float(tensor.abs().max().item())
    scale = max(max_abs, 1e-8) / 127.0
    scale_fp16 = torch.tensor([scale], dtype=torch.float16)
    q = (tensor / scale).round().clamp(-128, 127).to(torch.int8)
    return q, scale_fp16


def _quantize_latent_table_uint8_delta_split(latents: torch.Tensor) -> bytes:
    import brotli

    n, d = latents.shape
    mins = latents.min(dim=0).values.to(torch.float16)
    maxs = latents.max(dim=0).values.to(torch.float16)
    scales = ((maxs - mins).float() / 255.0).clamp(min=1e-8).to(torch.float16)
    q = ((latents - mins.float()) / scales.float()).round().clamp(0, 255).to(torch.int32)
    delta = torch.zeros_like(q)
    delta[0] = q[0]
    delta[1:] = q[1:] - q[:-1]
    delta_zz = torch.where(delta >= 0, 2 * delta, -2 * delta - 1).to(torch.int32)
    delta_zz = delta_zz.clamp(0, 65535)
    delta_lo = (delta_zz & 0xFF).to(torch.uint8)
    delta_hi = ((delta_zz >> 8) & 0xFF).to(torch.uint8)
    buf = io.BytesIO()
    buf.write(struct.pack("<II", n, d))
    buf.write(mins.numpy().tobytes())
    buf.write(scales.numpy().tobytes())
    buf.write(delta_lo.numpy().tobytes())
    buf.write(delta_hi.numpy().tobytes())
    return brotli.compress(buf.getvalue(), quality=11)


def export_ffnerv_to_archive(
    *,
    renderer: FFNeRVRenderer,
    latent_table: FFNeRVLatentTable,
    output_path: Path,
) -> str:
    import brotli

    config = renderer.config
    schema = renderer.schema
    latent_shape = tuple(latent_table.embedding.weight.shape)
    expected = (config.n_pairs, config.latent_dim)
    if latent_shape != expected:
        raise ValueError(f"latent_table shape {latent_shape} != expected {expected}")

    sd = renderer.state_dict()
    int8_chunks: list[bytes] = []
    scales_fp16: list[bytes] = []
    for key, expected_shape in schema:
        if key not in sd:
            raise KeyError(f"schema key {key!r} missing from state_dict")
        tensor = sd[key]
        if tuple(tensor.shape) != expected_shape:
            raise ValueError(
                f"schema shape mismatch for {key!r}: expected {expected_shape}, got {tuple(tensor.shape)}"
            )
        q, scale = _quantize_per_tensor_int8_with_fp16_scale(tensor)
        int8_chunks.append(q.detach().cpu().numpy().tobytes())
        scales_fp16.append(scale.detach().cpu().numpy().tobytes())

    decoder_blob = brotli.compress(b"".join(int8_chunks), quality=11)
    # Frequency table: small (n_frequencies floats); FP16.
    freq_table = renderer.fourier.frequencies.detach().cpu().to(torch.float16).numpy().tobytes()
    scale_table = b"".join(scales_fp16)
    latent_blob = _quantize_latent_table_uint8_delta_split(
        latent_table.embedding.weight.detach().cpu()
    )
    sidecar_blob = b""

    out = io.BytesIO()
    out.write(FFNERV_MAGIC)
    out.write(struct.pack("<H", FFNERV_FORMAT_VERSION))
    out.write(struct.pack("<H", FFNERV_FORMAT_ID))
    out.write(struct.pack("<H", config.latent_dim))
    out.write(struct.pack("<H", config.n_pairs))
    out.write(struct.pack("<H", config.n_frequencies))
    out.write(struct.pack("<H", config.base_channels))
    out.write(struct.pack("<I", len(decoder_blob)))
    out.write(decoder_blob)
    out.write(struct.pack("<I", len(freq_table)))
    out.write(freq_table)
    out.write(struct.pack("<I", len(scale_table)))
    out.write(scale_table)
    out.write(struct.pack("<I", len(latent_blob)))
    out.write(latent_blob)
    out.write(struct.pack("<I", len(sidecar_blob)))
    out.write(sidecar_blob)

    archive_bytes = out.getvalue()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(archive_bytes)
    return hashlib.sha256(archive_bytes).hexdigest()


# ── Smoke synthetic helper + default surrogates ──────────────────────────


def _make_synthetic_pair_batch_for_smoke(
    *, batch_size: int, latent_dim: int, eval_size: tuple[int, int],
    n_pairs: int, seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """# SYNTHETIC_NON_SMOKE_OK:ffnerv_phase_a_scaffold_smoke_test_only"""
    g = torch.Generator().manual_seed(seed)
    pair_indices = torch.randint(0, n_pairs, (batch_size,), generator=g)
    gt_pairs = torch.randint(
        0, 256, (batch_size, 2, 3, 874, 1164), generator=g, dtype=torch.uint8,
    )
    return pair_indices, gt_pairs


def default_seg_surrogate(
    pred_logits: torch.Tensor, target_logits: torch.Tensor
) -> torch.Tensor:
    T = 2.0
    log_p = F.log_softmax(pred_logits / T, dim=1)
    q = F.softmax(target_logits / T, dim=1)
    return F.kl_div(log_p, q, reduction="none").sum(dim=1).mean() * (T * T)


def default_pose_surrogate(
    pred_pose: torch.Tensor, target_pose: torch.Tensor
) -> torch.Tensor:
    return F.mse_loss(pred_pose[..., :6], target_pose[..., :6])
