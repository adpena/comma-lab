#!/usr/bin/env python3
"""PR101 200-param tiny_nn implementation - capacity-faithful, PMF-partial.

The 2026-05-08 implementation-vs-model audit found:
- Prior tool tools/pr101_tiny_nn_predict_pmf.py uses rank-K factorized
  softmax with 5K (rank=8) to 7.5K (rank=32) params — NOT the catalog's
  predicted "200-param MLP".
- The catalog row predicted: "200-param MLP predicting per-tensor PMF;
  ~400B model + AAC" but no implementation matched that capacity.

This tool builds a TRUE 200-param MLP (28->H=6->2) = 188 params total,
predicting (mean, log_scale) per tensor for a Gaussian conditional
arithmetic codec. ~400B model size after fp16 quant. This is faithful to
the catalog row's capacity and overhead constraint, but only a parametric
Gaussian PMF proxy for the phrase "per-tensor PMF"; it is not a full
255-bin per-tensor PMF predictor.

Architecture:
  Input:  28-dim one-hot (tensor_id)
  Hidden: 6 units + ReLU
  Output: 2 floats (mean, log_scale)
  Total params: 28*6 + 6 + 6*2 + 2 = 188 (within "200" budget)

Why this 1:1 fidelity matters:
- Prior larger NN attempts (5K, 7.5K params, 1.1KB MLP) all FAILED to
  beat brotli because model_bytes grew faster than AC payload shrunk.
- The catalog's 167,000 B prediction assumed 200-param MLP would have
  ~400 B model overhead leaving ~166,600 B for AC payload —
  achievable IF the empirical entropy (5.58 bits/elem * 228K elem / 8
  = 159 KB) plus model could fit.
- This test resolves whether the SMALLER model class can hit that band.

CLAUDE.md compliance: pure CPU/MPS for training (NEVER scorer-load);
evidence tagged `[CPU-prep faithful 200-param test]`; output is byte-
anchor only; no score claim. Per the canonical fix, ready_for_exact_
eval_dispatch is ALWAYS False from CPU/MPS evidence — only
[contest-CUDA] can flip it.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import struct
import sys
from pathlib import Path

import brotli
import constriction
import numpy as np
import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.pr101_split_brotli_codec import (  # noqa: E402
    FIXED_STATE_SCHEMA,
    N_QUANT,
    _quantize_tensor,
)

TOOL_NAME = "tools/pr101_tiny_nn_200_param_faithful.py"
SCHEMA_VERSION = "pr101_tiny_nn_200_param_faithful.v1"
ARCHIVE_OVERHEAD_BYTES = 16_094
N_TENSORS = len(FIXED_STATE_SCHEMA)
N_SYMBOLS = 2 * N_QUANT + 1
EVIDENCE_GRADE = "[CPU-prep 200-param gaussian-pmf proxy]"
EVIDENCE_SEMANTICS = "cpu_200_param_gaussian_pmf_proxy_byte_anchor_no_score"
DISPATCH_BLOCKERS = (
    "no_runtime_decoder_packet_built",
    "missing_exact_cuda_auth_eval",
    "requires_exact_cuda_auth_eval_before_any_score_use",
    "cpu_proxy_byte_anchor_not_score_evidence",
)


def proxy_evidence_contract() -> dict:
    return {
        "evidence_grade": EVIDENCE_GRADE,
        "evidence_marker": EVIDENCE_GRADE,
        "evidence_semantics": EVIDENCE_SEMANTICS,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "proxy_row": True,
        "dispatch_blockers": list(DISPATCH_BLOCKERS),
    }


class TinyNN200(nn.Module):
    """Faithful 200-param MLP — exactly matches catalog spec.

    28-dim one-hot -> Linear(28, 6) -> ReLU -> Linear(6, 2)
    Param count: 28*6 + 6 + 6*2 + 2 = 188 (<= 200)
    """

    def __init__(self, n_tensors: int = N_TENSORS, hidden: int = 6):
        super().__init__()
        self.n_tensors = n_tensors
        self.fc1 = nn.Linear(n_tensors, hidden)
        self.fc2 = nn.Linear(hidden, 2)
        self.act = nn.ReLU()

    def forward(self, tensor_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = tensor_idx.shape[0]
        oh = torch.zeros(b, self.n_tensors, device=tensor_idx.device, dtype=torch.float32)
        oh.scatter_(1, tensor_idx.unsqueeze(1), 1.0)
        h = self.act(self.fc1(oh))
        out = self.fc2(h)
        mean = out[:, 0]
        log_scale = out[:, 1].clamp(-3.0, 6.0)
        return mean, log_scale


def gaussian_log_pmf(symbol: torch.Tensor, mean: torch.Tensor, log_scale: torch.Tensor) -> torch.Tensor:
    scale = torch.exp(log_scale)
    upper = (symbol.float() + 0.5 - mean) / (scale * 1.4142135)
    lower = (symbol.float() - 0.5 - mean) / (scale * 1.4142135)
    pmf = 0.5 * (torch.erf(upper) - torch.erf(lower))
    return torch.log(pmf.clamp_min(1e-12))


def collect_symbols(state_dict_path: Path) -> tuple[list[np.ndarray], list[float]]:
    sd = torch.load(state_dict_path, map_location="cpu", weights_only=False)  # WEIGHTS_ONLY_FALSE_OK:trusted-PR101-substrate-state-dict-local-artifact
    sym_per_tensor: list[np.ndarray] = []
    scales: list[float] = []
    for name, _shape in FIXED_STATE_SCHEMA:
        qt = _quantize_tensor(name, sd[name], n_quant=N_QUANT)
        sym_per_tensor.append(qt.q_i8.astype(np.int32))
        scales.append(float(qt.scale))
    return sym_per_tensor, scales


def train_tiny(sym_per_tensor: list[np.ndarray], *, epochs: int = 100, lr: float = 5e-3, seed: int = 0) -> TinyNN200:
    torch.manual_seed(seed)
    model = TinyNN200()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[tiny-200] total params: {n_params}")
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    all_t_idx: list[np.ndarray] = []
    all_sym: list[np.ndarray] = []
    for t_idx, syms in enumerate(sym_per_tensor):
        n = syms.size
        if n == 0:
            continue
        all_t_idx.append(np.full(n, t_idx, dtype=np.int64))
        all_sym.append(syms.flatten().astype(np.float32))
    t_idx_full = torch.from_numpy(np.concatenate(all_t_idx))
    sym_full = torch.from_numpy(np.concatenate(all_sym))
    n_total = t_idx_full.numel()
    print(f"[tiny-200] training on {n_total:,} symbols")

    g = torch.Generator().manual_seed(seed)
    batch_size = 8192
    for ep in range(epochs):
        perm = torch.randperm(n_total, generator=g)
        ep_loss = 0.0
        n_batches = 0
        for s in range(0, n_total, batch_size):
            idx = perm[s : s + batch_size]
            mean, log_scale = model(t_idx_full[idx])
            log_pmf = gaussian_log_pmf(sym_full[idx], mean, log_scale)
            loss = -log_pmf.mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
            ep_loss += float(loss.item())
            n_batches += 1
        if (ep + 1) % 25 == 0 or ep == 0:
            avg_nats = ep_loss / max(n_batches, 1)
            avg_bits = avg_nats / np.log(2)
            print(f"  ep {ep+1:3d}/{epochs}: avg_bits={avg_bits:.4f}/symbol")
    return model


def encode_with_tiny(sym_per_tensor: list[np.ndarray], model: TinyNN200) -> tuple[bytes, dict]:
    model.eval()
    all_payloads: list[bytes] = []
    total_n = 0
    total_bits = 0.0
    with torch.no_grad():
        for t_idx, syms in enumerate(sym_per_tensor):
            n = syms.size
            if n == 0:
                all_payloads.append(b"")
                continue
            mean, log_scale = model(torch.tensor([t_idx], dtype=torch.long))
            mean_v = float(mean.item())
            scale_v = float(np.exp(log_scale.item()))
            mean_arr = np.full(n, mean_v, dtype=np.float64)
            scale_arr = np.full(n, max(scale_v, 0.05), dtype=np.float64)
            symbols_zero = syms.flatten().astype(np.int32)
            encoder = constriction.stream.queue.RangeEncoder()
            qg = constriction.stream.model.QuantizedGaussian(-N_QUANT, N_QUANT)
            encoder.encode(symbols_zero, qg, mean_arr, scale_arr)
            payload = bytes(encoder.get_compressed())
            all_payloads.append(payload)
            log_pmf = gaussian_log_pmf(
                torch.from_numpy(syms.flatten().astype(np.float32)),
                torch.from_numpy(mean_arr.astype(np.float32)),
                torch.log(torch.from_numpy(scale_arr.astype(np.float32))),
            )
            total_bits += float(-log_pmf.sum() / np.log(2))
            total_n += n
    payload_concat = b"".join(struct.pack("<I", len(p)) + p for p in all_payloads)
    return payload_concat, {
        "ac_payload_concat_bytes": len(payload_concat),
        "n_total_symbols": total_n,
        "weighted_theoretical_bits_per_element": total_bits / max(total_n, 1),
    }


def serialize_model(model: TinyNN200, scales: list[float]) -> bytes:
    buf = bytearray()
    buf += b"TN20"
    buf += struct.pack("<I", N_TENSORS)
    for s in scales:
        buf += np.float16(s).tobytes()
    for p in model.parameters():
        flat = p.detach().cpu().numpy().astype(np.float16).flatten()
        buf += struct.pack("<I", flat.size)
        buf += flat.tobytes()
    return bytes(buf)


def model_spec_fidelity(n_params: int, model_brotli_bytes: int) -> dict:
    """Return the precise implementation-vs-catalog fidelity classification."""
    capacity_match = n_params <= 200
    overhead_match = model_brotli_bytes <= 800
    distribution_contract_match = False
    return {
        "predicted": "200-param MLP predicting per-tensor PMF; ~400B model + AAC",
        "actual": "188-param MLP predicting Gaussian(mean, log_scale) per tensor",
        "actual_params": n_params,
        "actual_model_brotli_bytes": model_brotli_bytes,
        "capacity_constraint_match": capacity_match,
        "model_overhead_match": overhead_match,
        "distribution_contract_match": distribution_contract_match,
        "1:1_fidelity": capacity_match and overhead_match and distribution_contract_match,
        "fidelity_scope": "capacity_and_model_overhead_only",
        "model_spec_drift": [
            "parametric_gaussian_two_output_not_full_255_bin_per_tensor_pmf",
        ],
    }


def run_codec(state_dict_path: Path, *, epochs: int, seed: int) -> dict:
    sym_per_tensor, scales = collect_symbols(state_dict_path)
    n_total = sum(s.size for s in sym_per_tensor)
    print(f"[tiny-200] PR101 substrate: {n_total:,} symbols, {len(sym_per_tensor)} tensors")

    model = train_tiny(sym_per_tensor, epochs=epochs, seed=seed)
    n_params = sum(p.numel() for p in model.parameters())

    payload_concat, ac_stats = encode_with_tiny(sym_per_tensor, model)
    payload_brotli = brotli.compress(payload_concat, quality=11, lgwin=16, lgblock=19)

    model_blob = serialize_model(model, scales)
    model_brotli = brotli.compress(model_blob, quality=11, lgwin=16, lgblock=19)

    decoder_blob_bytes = len(model_brotli) + len(payload_brotli)
    archive_bytes = decoder_blob_bytes + ARCHIVE_OVERHEAD_BYTES
    return {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        **proxy_evidence_contract(),
        "input_state_dict": str(state_dict_path),
        "epochs": epochs,
        "seed": seed,
        "model_total_params": n_params,
        "model_blob_raw_bytes": len(model_blob),
        "model_blob_brotli_bytes": len(model_brotli),
        "ac_payload_concat_bytes": len(payload_concat),
        "ac_payload_brotli_bytes": len(payload_brotli),
        "decoder_blob_bytes": decoder_blob_bytes,
        "archive_overhead_bytes": ARCHIVE_OVERHEAD_BYTES,
        "archive_bytes": archive_bytes,
        "n_symbols": n_total,
        "weighted_theoretical_bits_per_element": ac_stats["weighted_theoretical_bits_per_element"],
        "model_spec_match": model_spec_fidelity(n_params, len(model_brotli)),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--state-dict",
        type=Path,
        default=REPO_ROOT / "experiments/results/pr101_codecop_sweep_20260507_codex/pr101_decoder_state_dict.pt",
    )
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--output-evidence", type=Path, default=None)
    args = p.parse_args(argv)

    if not args.state_dict.is_file():
        raise SystemExit(f"state_dict not found: {args.state_dict}")

    manifest = run_codec(args.state_dict, epochs=args.epochs, seed=args.seed)

    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    if args.output_json is None:
        out_dir = REPO_ROOT / f"reports/raw/pr101_tiny_nn_200param_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output_json = out_dir / "manifest.json"
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nmanifest: {args.output_json}")
    print(f"\nmodel_total_params: {manifest['model_total_params']}")
    print(f"model brotli: {manifest['model_blob_brotli_bytes']} B (raw {manifest['model_blob_raw_bytes']} B)")
    print(f"ac payload brotli: {manifest['ac_payload_brotli_bytes']:,} B")
    print(f"\narchive_bytes: {manifest['archive_bytes']:,} B")
    print("  vs catalog prediction: 167,000 B")
    delta_pred = manifest["archive_bytes"] - 167_000
    print(f"  delta vs prediction: {delta_pred:+,} B")
    print("  vs brotli baseline: 178,144 B")
    delta_brotli = manifest["archive_bytes"] - 178_144
    verdict = "BEAT" if delta_brotli < 0 else "TIES" if abs(delta_brotli) < 100 else "LOSES"
    print(f"  delta vs brotli: {delta_brotli:+,} B ({verdict} brotli)")
    print(f"\n1:1 fidelity vs catalog model_spec: {manifest['model_spec_match']['1:1_fidelity']}")
    print(f"fidelity scope: {manifest['model_spec_match']['fidelity_scope']}")

    if args.output_evidence:
        evidence_row = {
            "technique": "tiny_nn_pmf_predictor_200param_gaussian",
            "empirical_archive_bytes": manifest["archive_bytes"],
            "model_total_params": manifest["model_total_params"],
            **proxy_evidence_contract(),
            "source": (
                f"{EVIDENCE_GRADE} {args.output_json} "
                f"(capacity-faithful 200-param Gaussian PMF proxy for catalog "
                f"spec '200-param MLP'; "
                f"actual params={manifest['model_total_params']}, "
                f"actual model brotli={manifest['model_blob_brotli_bytes']} B; "
                "not a full 255-bin per-tensor PMF)"
            ),
            "timestamp": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "model_spec_fidelity_audit": manifest["model_spec_match"],
            "supersedes_prior_5K_param_test": True,
        }
        args.output_evidence.parent.mkdir(parents=True, exist_ok=True)
        with args.output_evidence.open("a", encoding="utf-8") as f:
            f.write(json.dumps(evidence_row) + "\n")
        print(f"\nevidence row appended: {args.output_evidence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
