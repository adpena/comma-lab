#!/usr/bin/env python3
"""PR101 Balle-class hyperprior codec — anchors the
``compressai_balle_hyperprior`` row in cathedral_autopilot's catalog.

Architecture (lightweight Balle-style):

  Hyperprior encoder: (tensor_id one-hot[28] + pos_normalized[3]) -> hidden[16] -> 2
    Output: (mean, log_scale) per-symbol Gaussian conditional prior
  Decoder runs same NN; no side-info needed beyond the NN weights themselves.

NN params: 31*16 + 16 + 16*2 + 2 = 546 weights -> ~1.1 KB fp16.
Per-symbol AC encoded under predicted (mean, scale) via constriction's
QuantizedGaussian model.

CLAUDE.md compliance: pure CPU/MPS for training (NEVER scorer-load); evidence
tagged [MPS-research-signal] - signal-only, never promotion. Output evidence
row is byte-anchor only; no score claim.
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

TOOL_NAME = "tools/pr101_compressai_balle_hyperprior.py"
SCHEMA_VERSION = "pr101_compressai_balle_hyperprior.v1"
ARCHIVE_OVERHEAD_BYTES = 16_094
N_TENSORS = len(FIXED_STATE_SCHEMA)
N_SYMBOLS = 2 * N_QUANT + 1  # 255
EVIDENCE_GRADE = "[MPS-research-signal]"
EVIDENCE_SEMANTICS = "mps_proxy_curve_shape_only_no_score"
DISPATCH_BLOCKERS = (
    "mps_proxy_signal_not_score_evidence",
    "not_exact_cuda_auth_eval",
    "no_exact_archive_adjudication",
    "missing_exact_cuda_auth_eval",
    "requires_exact_cuda_auth_eval_before_any_score_use",
)


def proxy_evidence_contract() -> dict[str, object]:
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


class HyperpriorMLP(nn.Module):
    """Tiny MLP: (tensor_id_onehot, pos_features) -> (mean, log_scale).

    Designed to fit in ~1 KB compressed. Decoder runs same MLP.
    """

    def __init__(self, n_tensors: int = N_TENSORS, hidden: int = 16):
        super().__init__()
        self.n_tensors = n_tensors
        in_dim = n_tensors + 3  # one-hot + (pos_norm, pos_log, pos_inv)
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, 2)
        self.act = nn.ReLU()

    def forward(self, tensor_idx: torch.Tensor, pos_norm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # tensor_idx: [B] long, pos_norm: [B] float in [0,1]
        b = tensor_idx.shape[0]
        oh = torch.zeros(b, self.n_tensors, device=tensor_idx.device, dtype=torch.float32)
        oh.scatter_(1, tensor_idx.unsqueeze(1), 1.0)
        pos_log = torch.log1p(pos_norm)
        pos_inv = 1.0 / (1.0 + pos_norm)
        feats = torch.cat([oh, pos_norm.unsqueeze(1), pos_log.unsqueeze(1), pos_inv.unsqueeze(1)], dim=1)
        h = self.act(self.fc1(feats))
        out = self.fc2(h)
        mean = out[:, 0]
        log_scale = out[:, 1].clamp(-3.0, 6.0)
        return mean, log_scale


def gaussian_log_pmf(symbol: torch.Tensor, mean: torch.Tensor, log_scale: torch.Tensor) -> torch.Tensor:
    """log P(symbol) under quantized Gaussian (no constriction; pure torch for training).

    Uses Balle-style: P(s) = Phi(s+0.5; mean, scale) - Phi(s-0.5; mean, scale).
    """
    scale = torch.exp(log_scale)
    upper = (symbol.float() + 0.5 - mean) / (scale * 1.4142135)
    lower = (symbol.float() - 0.5 - mean) / (scale * 1.4142135)
    pmf = 0.5 * (torch.erf(upper) - torch.erf(lower))
    pmf = pmf.clamp_min(1e-12)
    return torch.log(pmf)


def collect_symbols(state_dict_path: Path) -> tuple[list[np.ndarray], list[float]]:
    """Quantize each tensor; return list of symbol arrays + per-tensor scales."""
    sd = torch.load(state_dict_path, map_location="cpu", weights_only=False)
    if not isinstance(sd, dict):
        raise SystemExit(f"state_dict at {state_dict_path} is not a dict")
    sym_per_tensor: list[np.ndarray] = []
    scales: list[float] = []
    for name, _shape in FIXED_STATE_SCHEMA:
        qt = _quantize_tensor(name, sd[name], n_quant=N_QUANT)
        sym_per_tensor.append(qt.q_i8.astype(np.int32))
        scales.append(float(qt.scale))
    return sym_per_tensor, scales


def train_hyperprior(
    sym_per_tensor: list[np.ndarray],
    *,
    device: str = "mps",
    epochs: int = 60,
    batch_size: int = 8192,
    lr: float = 5e-3,
    seed: int = 0,
) -> HyperpriorMLP:
    """Train the hyperprior MLP on all symbols. Returns trained model on CPU."""
    torch.manual_seed(seed)
    dev = torch.device(device)
    model = HyperpriorMLP().to(dev)
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Build the global symbol corpus with (tensor_idx, pos_norm, symbol)
    all_t_idx: list[np.ndarray] = []
    all_pos: list[np.ndarray] = []
    all_sym: list[np.ndarray] = []
    for t_idx, syms in enumerate(sym_per_tensor):
        n = syms.size
        if n == 0:
            continue
        all_t_idx.append(np.full(n, t_idx, dtype=np.int64))
        all_pos.append((np.arange(n, dtype=np.float32) / max(n - 1, 1)).astype(np.float32))
        all_sym.append(syms.flatten().astype(np.float32))
    t_idx_full = np.concatenate(all_t_idx)
    pos_full = np.concatenate(all_pos)
    sym_full = np.concatenate(all_sym)
    n_total = t_idx_full.size

    t_idx_t = torch.from_numpy(t_idx_full).to(dev)
    pos_t = torch.from_numpy(pos_full).to(dev)
    sym_t = torch.from_numpy(sym_full).to(dev)

    print(f"[balle-hyperprior] training on {n_total:,} symbols, device={device}")
    g = torch.Generator(device=dev).manual_seed(seed)
    for ep in range(epochs):
        perm = torch.randperm(n_total, generator=g, device=dev)
        ep_loss = 0.0
        n_batches = 0
        for s in range(0, n_total, batch_size):
            idx = perm[s : s + batch_size]
            mean, log_scale = model(t_idx_t[idx], pos_t[idx])
            log_pmf = gaussian_log_pmf(sym_t[idx], mean, log_scale)
            loss = -log_pmf.mean()
            optim.zero_grad()
            loss.backward()
            optim.step()
            ep_loss += float(loss.item())
            n_batches += 1
        if (ep + 1) % 10 == 0 or ep == 0:
            avg_nats = ep_loss / max(n_batches, 1)
            avg_bits = avg_nats / np.log(2)
            print(f"  ep {ep+1:3d}/{epochs}: avg_nats={avg_nats:.4f} ({avg_bits:.4f} bits/symbol)")

    model.cpu()
    return model


def encode_with_hyperprior(
    sym_per_tensor: list[np.ndarray],
    model: HyperpriorMLP,
) -> tuple[bytes, dict]:
    """Encode all symbols using the trained model + constriction. Returns payload + stats."""
    model.eval()
    all_payloads: list[bytes] = []
    total_bits = 0.0
    total_n = 0
    per_tensor: list[dict] = []
    with torch.no_grad():
        for t_idx, syms in enumerate(sym_per_tensor):
            n = syms.size
            if n == 0:
                all_payloads.append(b"")
                per_tensor.append({"tensor_idx": t_idx, "n": 0, "bytes": 0})
                continue
            t_idx_t = torch.full((n,), t_idx, dtype=torch.long)
            pos_t = torch.from_numpy(np.arange(n, dtype=np.float32) / max(n - 1, 1))
            mean, log_scale = model(t_idx_t, pos_t)
            mean_np = mean.cpu().numpy().astype(np.float64)
            scale_np = np.exp(log_scale.cpu().numpy().astype(np.float64))
            scale_np = np.clip(scale_np, 0.05, 200.0)
            symbols_zero_centered = syms.flatten().astype(np.int32)
            encoder = constriction.stream.queue.RangeEncoder()
            # Family model: don't bake mean/std at construction; supply per-symbol arrays at encode.
            qg = constriction.stream.model.QuantizedGaussian(-N_QUANT, N_QUANT)
            encoder.encode(symbols_zero_centered, qg, mean_np, scale_np)
            payload = bytes(encoder.get_compressed())
            all_payloads.append(payload)
            # Theoretical bits via gaussian_log_pmf
            log_pmf = gaussian_log_pmf(
                torch.from_numpy(syms.flatten().astype(np.float32)),
                torch.from_numpy(mean_np.astype(np.float32)),
                torch.log(torch.from_numpy(scale_np.astype(np.float32))),
            )
            tensor_bits = float(-log_pmf.sum() / np.log(2))
            total_bits += tensor_bits
            total_n += n
            per_tensor.append({
                "tensor_idx": t_idx,
                "n": int(n),
                "ac_bytes": len(payload),
                "theoretical_bits": tensor_bits,
                "bits_per_element_theoretical": tensor_bits / max(n, 1),
            })
    payload_concat = b"".join(
        struct.pack("<I", len(p)) + p for p in all_payloads
    )
    stats = {
        "ac_payload_concat_bytes": len(payload_concat),
        "weighted_theoretical_bits_per_element": total_bits / max(total_n, 1),
        "n_total_symbols": total_n,
        "per_tensor": per_tensor,
    }
    return payload_concat, stats


def serialize_model(model: HyperpriorMLP, scales: list[float]) -> bytes:
    """Pack model weights as fp16 + per-tensor scales as fp16."""
    buf = bytearray()
    buf += b"BLLE"
    buf += struct.pack("<I", N_TENSORS)
    for s in scales:
        buf += np.float16(s).tobytes()
    for p in model.parameters():
        flat = p.detach().cpu().numpy().astype(np.float16).flatten()
        buf += struct.pack("<I", flat.size)
        buf += flat.tobytes()
    return bytes(buf)


def run_codec(state_dict_path: Path, *, device: str, epochs: int, batch_size: int, seed: int) -> dict:
    sym_per_tensor, scales = collect_symbols(state_dict_path)
    n_total = sum(s.size for s in sym_per_tensor)
    print(f"[balle-hyperprior] PR101 substrate: {n_total:,} symbols across {len(sym_per_tensor)} tensors")

    model = train_hyperprior(sym_per_tensor, device=device, epochs=epochs, batch_size=batch_size, seed=seed)

    payload_concat, ac_stats = encode_with_hyperprior(sym_per_tensor, model)
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
        "device": device,
        "epochs": epochs,
        "batch_size": batch_size,
        "seed": seed,
        "n_symbols": n_total,
        "ac_payload_concat_bytes": len(payload_concat),
        "ac_payload_brotli_bytes": len(payload_brotli),
        "model_blob_raw_bytes": len(model_blob),
        "model_blob_brotli_bytes": len(model_brotli),
        "decoder_blob_bytes": decoder_blob_bytes,
        "archive_overhead_bytes": ARCHIVE_OVERHEAD_BYTES,
        "archive_bytes": archive_bytes,
        "weighted_theoretical_bits_per_element": ac_stats["weighted_theoretical_bits_per_element"],
        "per_tensor": ac_stats["per_tensor"],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--state-dict",
        type=Path,
        default=REPO_ROOT
        / "experiments/results/pr101_codecop_sweep_20260507_codex/pr101_decoder_state_dict.pt",
    )
    p.add_argument("--device", choices=["mps", "cpu"], default="mps")
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--output-evidence", type=Path, default=None)
    args = p.parse_args(argv)

    if not args.state_dict.is_file():
        raise SystemExit(f"state_dict not found: {args.state_dict}")

    manifest = run_codec(
        args.state_dict, device=args.device, epochs=args.epochs,
        batch_size=args.batch_size, seed=args.seed,
    )

    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    if args.output_json is None:
        out_dir = REPO_ROOT / f"reports/raw/pr101_balle_hyperprior_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output_json = out_dir / "manifest.json"
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nmanifest: {args.output_json}")
    print(f"\narchive_bytes: {manifest['archive_bytes']:,} B  [MPS-research-signal]")
    print("  vs cathedral_autopilot prediction: 158,000 B")
    delta = manifest["archive_bytes"] - 158_000
    verdict = "BEAT" if delta < -1000 else "TIED" if abs(delta) <= 1000 else "MISSED"
    print(f"  delta: {delta:+,} B ({verdict} prediction)")
    print("\n  vs brotli baseline: 178,144 B")
    delta_brotli = manifest["archive_bytes"] - 178_144
    print(f"  delta: {delta_brotli:+,} B "
          f"({'BEAT' if delta_brotli < 0 else 'LOSES'} brotli)")
    print(f"\n  model: {manifest['model_blob_raw_bytes']} B raw -> {manifest['model_blob_brotli_bytes']} B brotli")
    print(f"  ac payload: {manifest['ac_payload_concat_bytes']:,} -> {manifest['ac_payload_brotli_bytes']:,} B brotli")
    print(f"  weighted theoretical bits/element: {manifest['weighted_theoretical_bits_per_element']:.4f}")

    if args.output_evidence:
        evidence_row = {
            "technique": "compressai_balle_hyperprior",
            "empirical_archive_bytes": manifest["archive_bytes"],
            **proxy_evidence_contract(),
            "source": f"[MPS-research-signal] {args.output_json}",
            "timestamp": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        args.output_evidence.parent.mkdir(parents=True, exist_ok=True)
        with args.output_evidence.open("a", encoding="utf-8") as f:
            f.write(json.dumps(evidence_row) + "\n")
        print(f"\nevidence row appended: {args.output_evidence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
