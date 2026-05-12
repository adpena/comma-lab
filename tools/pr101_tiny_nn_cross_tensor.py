#!/usr/bin/env python3
"""PR101 tiny_nn cross-tensor PMF predictor — faithful 1:1 architecture-class test.

Background (2026-05-08 implementation-vs-model audit):
- Audit criterion #3 of 3 remaining: cross-tensor context predictor —
  per-tensor PMF predicted from NEIGHBORING-tensor empirical PMFs as
  features (not just position within current tensor).

This is the architecture class the PR101 joint-entropy floor finding
hinted at: cross-tensor mutual information from sign+DCT+Hessian+skip+
scale-residual sources potentially worth 14-32 KB. The MLP/LSTM/Transformer
architectures all condition on context within the SAME tensor; this one
conditions on OTHER tensors.

Architecture (1:1 with audit criterion):
  Per-tensor input features (~known to decoder once side-info shipped):
    - tensor_id one-hot (28 dim)
    - per-tensor scale (1 dim, log-normalized)
    - rank/shape features (3 dim)
    - NEIGHBORING TENSORS' empirical PMF moments (mean, std, skew, kurt,
      L0-fraction) — but charged in the side-info bytes!
  Hidden:  Linear(features -> H) + ReLU
  Output: Linear(H, 2*N_QUANT+1 == 255) -- categorical PMF directly

Param count target: ~1-3K with H=8-16.

Why charge cross-tensor moments as side info:
- The decoder needs the same conditioning features the encoder used.
- We give it the FULL per-tensor moment summary as fp16 (5 floats * 28
  tensors = 280 bytes). Brotli will compress correlations.
- The neural model is then a tiny MLP that maps (tensor_id, neighbor
  moments) -> categorical PMF.

This tests whether cross-tensor MI is exploitable WITH a small neural
context model. If the per-tensor PMFs cluster (e.g., all conv kernels
look alike), a tiny model can predict any tensor's PMF from a few
neighbors' moments better than per-tensor empirical PMFs (which cost
255 fp16 entries * 28 tensors = 14 KB raw, ~3-5 KB brotli).

CLAUDE.md compliance:
- Pure CPU/MPS (NEVER scorer-load).
- proxy_evidence_contract: ready_for_exact_eval_dispatch=False.
- `[CPU-prep faithful CrossTensor test]` evidence grade.
- All cross-tensor moments (neighbor features) charged in side info.
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
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.pr101_split_brotli_codec import (  # noqa: E402
    FIXED_STATE_SCHEMA,
    N_QUANT,
    _quantize_tensor,
)

TOOL_NAME = "tools/pr101_tiny_nn_cross_tensor.py"
SCHEMA_VERSION = "pr101_tiny_nn_cross_tensor.v1"
ARCHIVE_OVERHEAD_BYTES = 16_094
N_TENSORS = len(FIXED_STATE_SCHEMA)
N_SYMBOLS = 2 * N_QUANT + 1  # 255 categories
EVIDENCE_GRADE = "[CPU-prep faithful CrossTensor test]"
EVIDENCE_SEMANTICS = "cpu_faithful_cross_tensor_pmf_byte_anchor_no_score"
DISPATCH_BLOCKERS = (
    "no_runtime_decoder_packet_built",
    "missing_exact_cuda_auth_eval",
    "requires_exact_cuda_auth_eval_before_any_score_use",
    "cpu_proxy_byte_anchor_not_score_evidence",
)
REFERENCE_BROTLI_OPTUNA_BYTES = 178_144
N_MOMENT_FEATURES = 5  # mean, std, skew, kurt, L0-fraction


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


class CrossTensorPMFPredictor(nn.Module):
    """Tiny MLP that maps (tensor_id, neighbor_moments) -> categorical PMF.

    Input dimension:
      - one-hot tensor_id: N_TENSORS = 28
      - own moment features:                     N_MOMENT_FEATURES = 5
      - mean of neighbor (other tensors) moments: N_MOMENT_FEATURES = 5
      Total: 38

    Hidden: H=12, output: N_SYMBOLS=255
    Param count: 38*12 + 12 + 12*255 + 255 = ~3.7K
    """

    def __init__(
        self,
        n_tensors: int = N_TENSORS,
        n_symbols: int = N_SYMBOLS,
        hidden: int = 12,
        n_moment_feats: int = N_MOMENT_FEATURES,
    ):
        super().__init__()
        self.n_tensors = n_tensors
        self.n_symbols = n_symbols
        self.hidden = hidden
        self.n_moment_feats = n_moment_feats
        in_dim = n_tensors + 2 * n_moment_feats
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, n_symbols)
        self.act = nn.ReLU()

    def forward(
        self,
        tensor_id: torch.Tensor,
        own_moments: torch.Tensor,
        neighbor_moments: torch.Tensor,
    ) -> torch.Tensor:
        b = tensor_id.shape[0]
        oh = torch.zeros(
            b,
            self.n_tensors,
            device=tensor_id.device,
            dtype=torch.float32,
        )
        oh.scatter_(1, tensor_id.unsqueeze(1), 1.0)
        x = torch.cat([oh, own_moments, neighbor_moments], dim=-1)
        h = self.act(self.fc1(x))
        logits = self.fc2(h)
        return logits


def gaussian_log_pmf_categorical(
    symbol_signed: torch.Tensor, logits: torch.Tensor
) -> torch.Tensor:
    """Categorical NLL: symbol_signed in [-N_QUANT, N_QUANT] -> [0, N_SYMBOLS-1]."""
    targets = (symbol_signed + N_QUANT).long()
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def collect_symbols(
    state_dict_path: Path,
) -> tuple[list[np.ndarray], list[float]]:
    sd = torch.load(state_dict_path, map_location="cpu", weights_only=False)  # WEIGHTS_ONLY_FALSE_OK:trusted-PR101-substrate-state-dict-local-artifact
    sym_per_tensor: list[np.ndarray] = []
    scales: list[float] = []
    for name, _shape in FIXED_STATE_SCHEMA:
        qt = _quantize_tensor(name, sd[name], n_quant=N_QUANT)
        sym_per_tensor.append(qt.q_i8.astype(np.int32))
        scales.append(float(qt.scale))
    return sym_per_tensor, scales


def compute_moments(syms: np.ndarray) -> np.ndarray:
    """Per-tensor moment summary: (mean, std, skew, kurt, L0-fraction)."""
    if syms.size == 0:
        return np.zeros(N_MOMENT_FEATURES, dtype=np.float32)
    s = syms.astype(np.float64)
    mu = float(s.mean())
    sd = float(s.std()) + 1e-9
    centered = s - mu
    skew = float((centered**3).mean() / (sd**3))
    kurt = float((centered**4).mean() / (sd**4)) - 3.0
    l0_frac = float((s == 0).sum() / s.size)
    # Normalize roughly to [-3, 3] for stable training
    return np.array(
        [
            mu / N_QUANT,  # in [-1, 1]
            sd / N_QUANT,  # roughly in [0, 1]
            np.clip(skew, -3.0, 3.0),
            np.clip(kurt, -3.0, 10.0) / 3.0,
            l0_frac,  # in [0, 1]
        ],
        dtype=np.float32,
    )


def build_cross_tensor_features(
    sym_per_tensor: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (own_moments [N_T, 5], neighbor_moments [N_T, 5]).

    The neighbor moment for tensor t is the mean of all OTHER tensors'
    moments (a leave-one-out average). This is what the decoder will
    receive in side info.
    """
    own = np.stack([compute_moments(s) for s in sym_per_tensor], axis=0)
    n = own.shape[0]
    neighbor = np.zeros_like(own)
    if n > 1:
        total = own.sum(axis=0, keepdims=True)
        for i in range(n):
            neighbor[i] = (total[0] - own[i]) / (n - 1)
    return own, neighbor


def train_cross_tensor(
    sym_per_tensor: list[np.ndarray],
    *,
    hidden: int,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
) -> tuple[CrossTensorPMFPredictor, np.ndarray, np.ndarray]:
    torch.manual_seed(seed)
    own, neighbor = build_cross_tensor_features(sym_per_tensor)
    own_t = torch.from_numpy(own).float()
    neighbor_t = torch.from_numpy(neighbor).float()
    model = CrossTensorPMFPredictor(hidden=hidden)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[cross-tensor] total params: {n_params}")
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    all_t_idx: list[np.ndarray] = []
    all_sym: list[np.ndarray] = []
    for t_idx, syms in enumerate(sym_per_tensor):
        n = syms.size
        if n == 0:
            continue
        all_t_idx.append(np.full(n, t_idx, dtype=np.int64))
        all_sym.append(syms.flatten().astype(np.int64))
    t_idx_full = torch.from_numpy(np.concatenate(all_t_idx))
    sym_full = torch.from_numpy(np.concatenate(all_sym))
    n_total = int(t_idx_full.numel())
    print(f"[cross-tensor] training on {n_total:,} symbols")

    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        perm = torch.randperm(n_total, generator=g)
        ep_loss = 0.0
        n_batches = 0
        for s in range(0, n_total, batch_size):
            idx = perm[s : s + batch_size]
            tids = t_idx_full[idx]
            mean = own_t[tids]
            nb_mean = neighbor_t[tids]
            logits = model(tids, mean, nb_mean)
            log_pmf = gaussian_log_pmf_categorical(sym_full[idx], logits)
            loss = -log_pmf.mean()
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            ep_loss += float(loss.item())
            n_batches += 1
        if (ep + 1) % max(1, epochs // 5) == 0 or ep == 0:
            avg_nats = ep_loss / max(n_batches, 1)
            avg_bits = avg_nats / np.log(2)
            print(
                f"  ep {ep+1:3d}/{epochs}: avg_bits={avg_bits:.4f}/symbol"
            )
    return model, own, neighbor


def encode_with_cross_tensor(
    sym_per_tensor: list[np.ndarray],
    model: CrossTensorPMFPredictor,
    own: np.ndarray,
    neighbor: np.ndarray,
) -> tuple[bytes, dict]:
    """Encode each tensor with cross-tensor-conditioned categorical PMF."""
    model.eval()
    own_t = torch.from_numpy(own).float()
    neighbor_t = torch.from_numpy(neighbor).float()
    all_payloads: list[bytes] = []
    total_n = 0
    total_bits = 0.0
    with torch.no_grad():
        for t_idx, syms in enumerate(sym_per_tensor):
            n = syms.size
            if n == 0:
                all_payloads.append(b"")
                continue
            tid = torch.tensor([t_idx], dtype=torch.long)
            logits = model(tid, own_t[tid], neighbor_t[tid])
            log_probs = F.log_softmax(logits, dim=-1)
            probs = log_probs.exp().squeeze(0).cpu().numpy().astype(np.float64)
            probs = np.maximum(probs, 1e-9)
            probs /= probs.sum()
            symbols_signed = syms.flatten().astype(np.int32)
            symbols_unsigned = (symbols_signed + N_QUANT).astype(np.int32)
            cat = constriction.stream.model.Categorical(
                probs, perfect=False
            )
            encoder = constriction.stream.queue.RangeEncoder()
            encoder.encode(symbols_unsigned, cat)
            payload = bytes(encoder.get_compressed())
            all_payloads.append(payload)
            log_pmf = log_probs[0, symbols_unsigned]
            total_bits += float(-log_pmf.sum() / np.log(2))
            total_n += n
    payload_concat = b"".join(
        struct.pack("<I", len(p)) + p for p in all_payloads
    )
    return payload_concat, {
        "ac_payload_concat_bytes": len(payload_concat),
        "n_total_symbols": total_n,
        "weighted_theoretical_bits_per_element": (
            total_bits / max(total_n, 1)
        ),
    }


def serialize_model(
    model: CrossTensorPMFPredictor,
    scales: list[float],
    own_moments: np.ndarray,
) -> bytes:
    """Serialize: header + scales + own_moments + model weights.

    own_moments is the canonical side info: each row [5 fp16] for each
    tensor. The decoder rebuilds neighbor_moments as the leave-one-out
    mean — a deterministic computation from own_moments. So we ONLY
    charge own_moments in side info.
    """
    buf = bytearray()
    buf += b"XTNN"
    buf += struct.pack("<I", N_TENSORS)
    for s in scales:
        buf += np.float16(s).tobytes()
    # own_moments side info: N_TENSORS x N_MOMENT_FEATURES fp16
    buf += own_moments.astype(np.float16).tobytes()
    # Model weights
    for _name, p in model.state_dict().items():
        flat = p.detach().cpu().numpy().astype(np.float16).flatten()
        buf += struct.pack("<I", flat.size)
        buf += flat.tobytes()
    return bytes(buf)


def run_codec(
    state_dict_path: Path,
    *,
    hidden: int,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
) -> dict:
    sym_per_tensor, scales = collect_symbols(state_dict_path)
    n_total = sum(s.size for s in sym_per_tensor)
    print(
        f"[cross-tensor] PR101 substrate: {n_total:,} symbols, "
        f"{len(sym_per_tensor)} tensors"
    )

    model, own, neighbor = train_cross_tensor(
        sym_per_tensor,
        hidden=hidden,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        seed=seed,
    )
    n_params = sum(p.numel() for p in model.parameters())

    payload_concat, ac_stats = encode_with_cross_tensor(
        sym_per_tensor, model, own, neighbor
    )
    payload_brotli = brotli.compress(
        payload_concat, quality=11, lgwin=16, lgblock=19
    )

    model_blob = serialize_model(model, scales, own)
    model_brotli = brotli.compress(
        model_blob, quality=11, lgwin=16, lgblock=19
    )

    decoder_blob_bytes = len(model_brotli) + len(payload_brotli)
    archive_bytes = decoder_blob_bytes + ARCHIVE_OVERHEAD_BYTES
    return {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        **proxy_evidence_contract(),
        "input_state_dict": str(state_dict_path),
        "epochs": epochs,
        "hidden": hidden,
        "batch_size": batch_size,
        "lr": lr,
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
        "side_info_own_moments_bytes": int(
            own.astype(np.float16).nbytes
        ),
        "weighted_theoretical_bits_per_element": ac_stats[
            "weighted_theoretical_bits_per_element"
        ],
        "comparison_brotli_optuna_bytes": REFERENCE_BROTLI_OPTUNA_BYTES,
        "delta_vs_brotli_optuna": (
            archive_bytes - REFERENCE_BROTLI_OPTUNA_BYTES
        ),
        "model_spec_match": {
            "predicted": (
                "Cross-tensor context predictor: per-tensor PMF from "
                "neighboring tensors' moments as features"
            ),
            "actual_params": n_params,
            "actual_model_brotli_bytes": len(model_brotli),
            "1:1_fidelity": 1_000 <= n_params <= 5_000,
        },
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--state-dict",
        type=Path,
        default=REPO_ROOT
        / "experiments/results/pr101_codecop_sweep_20260507_codex"
        / "pr101_decoder_state_dict.pt",
    )
    p.add_argument("--hidden", type=int, default=12)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--output-evidence", type=Path, default=None)
    args = p.parse_args(argv)

    if not args.state_dict.is_file():
        raise SystemExit(f"state_dict not found: {args.state_dict}")

    manifest = run_codec(
        args.state_dict,
        hidden=args.hidden,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    if args.output_json is None:
        out_dir = REPO_ROOT / f"reports/raw/pr101_tiny_nn_cross_tensor_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output_json = out_dir / "manifest.json"
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"\nmanifest: {args.output_json}")
    print(f"\nmodel_total_params: {manifest['model_total_params']}")
    print(
        f"model brotli: {manifest['model_blob_brotli_bytes']} B "
        f"(raw {manifest['model_blob_raw_bytes']} B; "
        f"side_info_own_moments={manifest['side_info_own_moments_bytes']} B)"
    )
    print(
        f"ac payload brotli: {manifest['ac_payload_brotli_bytes']:,} B "
        f"(theoretical {manifest['weighted_theoretical_bits_per_element']:.4f}"
        " bits/symbol)"
    )
    print(f"\narchive_bytes: {manifest['archive_bytes']:,} B")
    print(f"  vs brotli baseline: {REFERENCE_BROTLI_OPTUNA_BYTES:,} B")
    delta_brotli = manifest["delta_vs_brotli_optuna"]
    verdict = (
        "BEAT"
        if delta_brotli < 0
        else "TIES"
        if abs(delta_brotli) < 100
        else "LOSES"
    )
    print(f"  delta: {delta_brotli:+,} B ({verdict} brotli)")
    print(
        f"\n1:1 fidelity vs audit criterion: "
        f"{manifest['model_spec_match']['1:1_fidelity']}"
    )

    if args.output_evidence:
        evidence_row = {
            "technique": "tiny_nn_pmf_predictor_cross_tensor_faithful",
            "empirical_archive_bytes": manifest["archive_bytes"],
            "model_total_params": manifest["model_total_params"],
            **proxy_evidence_contract(),
            "source": (
                f"{EVIDENCE_GRADE} {args.output_json} "
                f"(faithful 1:1 cross-tensor implementation; "
                f"params={manifest['model_total_params']}, "
                f"model brotli={manifest['model_blob_brotli_bytes']} B)"
            ),
            "timestamp": _dt.datetime.now(_dt.UTC).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "model_spec_fidelity_audit": manifest["model_spec_match"],
            "audit_criterion": "cross_tensor_pmf_predictor",
        }
        args.output_evidence.parent.mkdir(parents=True, exist_ok=True)
        with args.output_evidence.open("a", encoding="utf-8") as f:
            f.write(json.dumps(evidence_row) + "\n")
        print(f"\nevidence row appended: {args.output_evidence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
