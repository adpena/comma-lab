#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""PR101 tiny_nn Transformer PMF predictor — faithful 1:1 architecture-class test.

Background (2026-05-08 implementation-vs-model audit):
- Audit criterion #2 of 3 remaining: tiny transformer (1-2 layers,
  ~1-5K params) with self-attention over per-tensor symbol context.

This tool builds a small transformer encoder block that scans each
tensor's symbol stream with masked self-attention (causal) and emits
a Gaussian PMF (mean, log_scale) per position.

Architecture (1:1 with audit criterion):
  Input:  prev_symbol embedding (N_SYMBOLS+1 -> d_model) +
          tensor_id  embedding (N_TENSORS  -> d_model) +
          sinusoidal positional encoding
  Block:  1 layer of MultiheadAttention(d_model, n_heads=2, dropout=0)
          + FFN(d_model -> d_ff -> d_model)
          + LayerNorms (post-norm)
  Output: Linear(d_model, 2) -> (mean, log_scale)
  Param count target: ~1-5K (with d_model=8, d_ff=16, n_heads=2)

Why this matters:
- Self-attention captures long-range patterns inside a tensor that
  LSTMs may struggle with at tiny capacity. If PR101 weight tensors
  have repeating patterns (e.g., kernel-row periodicity in 3x3 conv
  weights flattened to int8), a transformer can attend to positions
  matching the period.
- This is qualitatively different from BOTH the 188-param MLP
  (stateless) AND the LSTM (sequential without attention).

CLAUDE.md compliance:
- Pure CPU/MPS (NEVER scorer-load).
- proxy_evidence_contract: ready_for_exact_eval_dispatch=False.
- `[CPU-prep faithful Transformer test]` evidence grade.
- No score claim. Byte-anchor only.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
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

TOOL_NAME = "tools/pr101_tiny_nn_transformer.py"
SCHEMA_VERSION = "pr101_tiny_nn_transformer.v1"
ARCHIVE_OVERHEAD_BYTES = 16_094
N_TENSORS = len(FIXED_STATE_SCHEMA)
N_SYMBOLS = 2 * N_QUANT + 1
START_SYMBOL = N_SYMBOLS
EVIDENCE_GRADE = "[CPU-prep faithful Transformer test]"
EVIDENCE_SEMANTICS = "cpu_faithful_transformer_pmf_byte_anchor_no_score"
DISPATCH_BLOCKERS = (
    "no_runtime_decoder_packet_built",
    "missing_exact_cuda_auth_eval",
    "requires_exact_cuda_auth_eval_before_any_score_use",
    "cpu_proxy_byte_anchor_not_score_evidence",
)
REFERENCE_BROTLI_OPTUNA_BYTES = 178_144


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


def _sinusoidal_pe(n: int, d: int) -> torch.Tensor:
    """Standard fixed sinusoidal positional encoding (no params)."""
    pe = torch.zeros(n, d)
    if n == 0:
        return pe
    position = torch.arange(0, n, dtype=torch.float).unsqueeze(1)
    if d == 1:
        pe[:, 0] = torch.sin(position[:, 0])
        return pe
    div_term = torch.exp(
        torch.arange(0, d, 2, dtype=torch.float)
        * (-math.log(10000.0) / d)
    )
    pe[:, 0::2] = torch.sin(position * div_term[: pe[:, 0::2].shape[1]])
    pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
    return pe


class TinyTransformerPredictor(nn.Module):
    """Tiny transformer block with causal self-attention.

    d_model=8, n_heads=2, d_ff=16, n_tensors=28, n_symbols=255:
    embeddings ~2300 + attention ~280 + FFN ~280 + head ~18 ~ 2900 params.
    """

    def __init__(
        self,
        n_tensors: int = N_TENSORS,
        n_symbols: int = N_SYMBOLS,
        d_model: int = 8,
        n_heads: int = 2,
        d_ff: int = 16,
        max_len: int = 4096,
    ):
        super().__init__()
        self.n_tensors = n_tensors
        self.n_symbols = n_symbols
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_len = max_len
        self.sym_emb = nn.Embedding(n_symbols + 1, d_model)
        self.tensor_emb = nn.Embedding(n_tensors, d_model)
        # Fixed sinusoidal PE (no learned params, zero byte cost beyond code).
        self.register_buffer(
            "pe", _sinusoidal_pe(max_len, d_model), persistent=False
        )
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 2)

    def forward(
        self,
        prev_symbol: torch.Tensor,
        tensor_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # prev_symbol, tensor_id: [B, T] long
        b, t = prev_symbol.shape
        if t > self.max_len:
            raise ValueError(
                f"sequence {t} exceeds max_len {self.max_len}; "
                f"reduce --bptt-window or rebuild PE table"
            )
        s = self.sym_emb(prev_symbol)
        ti = self.tensor_emb(tensor_id)
        x = s + ti + self.pe[:t].unsqueeze(0)  # [B, T, D]
        # Causal mask
        mask = torch.full((t, t), float("-inf"), device=x.device)
        mask = torch.triu(mask, diagonal=1)
        attn_out, _ = self.attn(x, x, x, attn_mask=mask, need_weights=False)
        x = self.ln1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)
        out = self.head(x)
        mean = out[..., 0]
        log_scale = out[..., 1].clamp(-3.0, 6.0)
        return mean, log_scale


def gaussian_log_pmf(
    symbol: torch.Tensor, mean: torch.Tensor, log_scale: torch.Tensor
) -> torch.Tensor:
    scale = torch.exp(log_scale)
    upper = (symbol.float() + 0.5 - mean) / (scale * 1.4142135)
    lower = (symbol.float() - 0.5 - mean) / (scale * 1.4142135)
    pmf = 0.5 * (torch.erf(upper) - torch.erf(lower))
    return torch.log(pmf.clamp_min(1e-12))


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


def _build_sequences(
    sym_per_tensor: list[np.ndarray],
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
    prev_seqs: list[torch.Tensor] = []
    tid_seqs: list[torch.Tensor] = []
    tgt_seqs: list[torch.Tensor] = []
    for t_idx, syms in enumerate(sym_per_tensor):
        n = syms.size
        if n == 0:
            continue
        unsigned = (syms.flatten().astype(np.int64) + N_QUANT)
        prev = np.empty(n, dtype=np.int64)
        prev[0] = START_SYMBOL
        prev[1:] = unsigned[:-1]
        tid = np.full(n, t_idx, dtype=np.int64)
        prev_seqs.append(torch.from_numpy(prev))
        tid_seqs.append(torch.from_numpy(tid))
        tgt_seqs.append(torch.from_numpy(syms.flatten().astype(np.int64)))
    return prev_seqs, tid_seqs, tgt_seqs


def train_transformer(
    sym_per_tensor: list[np.ndarray],
    *,
    d_model: int,
    n_heads: int,
    d_ff: int,
    epochs: int,
    lr: float,
    bptt_window: int,
    seed: int,
) -> TinyTransformerPredictor:
    torch.manual_seed(seed)
    model = TinyTransformerPredictor(
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        max_len=max(bptt_window, 1024),
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[transformer] total params: {n_params}")
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    prev_seqs, tid_seqs, tgt_seqs = _build_sequences(sym_per_tensor)
    n_total = sum(int(s.numel()) for s in prev_seqs)
    print(
        f"[transformer] training on {n_total:,} symbols across "
        f"{len(prev_seqs)} tensors"
    )

    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        ep_loss = 0.0
        n_batches = 0
        windows: list[tuple[int, int, int]] = []
        for t_idx, prev in enumerate(prev_seqs):
            n = int(prev.numel())
            if n <= bptt_window:
                windows.append((t_idx, 0, n))
            else:
                for s in range(0, n, bptt_window):
                    windows.append((t_idx, s, min(s + bptt_window, n)))
        perm = torch.randperm(len(windows), generator=g).tolist()
        for w_idx in perm:
            t_idx, s, e = windows[w_idx]
            prev = prev_seqs[t_idx][s:e].unsqueeze(0)
            tid = tid_seqs[t_idx][s:e].unsqueeze(0)
            tgt = tgt_seqs[t_idx][s:e].unsqueeze(0)
            mean, log_scale = model(prev, tid)
            log_pmf = gaussian_log_pmf(tgt, mean, log_scale)
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
    return model


def encode_with_transformer(
    sym_per_tensor: list[np.ndarray],
    model: TinyTransformerPredictor,
    *,
    bptt_window: int,
) -> tuple[bytes, dict]:
    """Encode each tensor with transformer-predicted Gaussian PMF.

    For long tensors, we emit one stream PER WINDOW so encoding stays
    within max_len. The receiver can reconstruct by concatenation.
    """
    model.eval()
    all_payloads: list[bytes] = []
    total_n = 0
    total_bits = 0.0
    qg = constriction.stream.model.QuantizedGaussian(-N_QUANT, N_QUANT)
    with torch.no_grad():
        prev_seqs, tid_seqs, _ = _build_sequences(sym_per_tensor)
        idx_offset = 0
        for t_idx, syms in enumerate(sym_per_tensor):
            n = syms.size
            if n == 0:
                all_payloads.append(b"")
                continue
            prev_full = prev_seqs[idx_offset]
            tid_full = tid_seqs[idx_offset]
            idx_offset += 1
            tensor_payloads: list[bytes] = []
            for s in range(0, n, bptt_window):
                e = min(s + bptt_window, n)
                prev = prev_full[s:e].unsqueeze(0)
                tid = tid_full[s:e].unsqueeze(0)
                mean, log_scale = model(prev, tid)
                mean_arr = (
                    mean.squeeze(0).cpu().numpy().astype(np.float64)
                )
                scale_arr = (
                    torch.exp(log_scale.squeeze(0))
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )
                scale_arr = np.maximum(scale_arr, 0.05)
                seg = syms.flatten()[s:e].astype(np.int32)
                encoder = constriction.stream.queue.RangeEncoder()
                encoder.encode(seg, qg, mean_arr, scale_arr)
                tensor_payloads.append(bytes(encoder.get_compressed()))
                log_pmf = gaussian_log_pmf(
                    torch.from_numpy(seg.astype(np.float32)),
                    torch.from_numpy(mean_arr.astype(np.float32)),
                    torch.log(
                        torch.from_numpy(scale_arr.astype(np.float32))
                    ),
                )
                total_bits += float(-log_pmf.sum() / np.log(2))
            # Concatenate window payloads under one length-prefixed stream
            tensor_blob = b"".join(
                struct.pack("<I", len(p)) + p for p in tensor_payloads
            )
            all_payloads.append(tensor_blob)
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
    model: TinyTransformerPredictor, scales: list[float]
) -> bytes:
    buf = bytearray()
    buf += b"TFMR"
    buf += struct.pack("<I", N_TENSORS)
    for s in scales:
        buf += np.float16(s).tobytes()
    for _name, p in model.state_dict().items():
        flat = p.detach().cpu().numpy().astype(np.float16).flatten()
        buf += struct.pack("<I", flat.size)
        buf += flat.tobytes()
    return bytes(buf)


def run_codec(
    state_dict_path: Path,
    *,
    d_model: int,
    n_heads: int,
    d_ff: int,
    epochs: int,
    lr: float,
    bptt_window: int,
    seed: int,
) -> dict:
    sym_per_tensor, scales = collect_symbols(state_dict_path)
    n_total = sum(s.size for s in sym_per_tensor)
    print(
        f"[transformer] PR101 substrate: {n_total:,} symbols, "
        f"{len(sym_per_tensor)} tensors"
    )

    model = train_transformer(
        sym_per_tensor,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        epochs=epochs,
        lr=lr,
        bptt_window=bptt_window,
        seed=seed,
    )
    n_params = sum(p.numel() for p in model.parameters())

    payload_concat, ac_stats = encode_with_transformer(
        sym_per_tensor, model, bptt_window=bptt_window
    )
    payload_brotli = brotli.compress(
        payload_concat, quality=11, lgwin=16, lgblock=19
    )
    model_blob = serialize_model(model, scales)
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
        "d_model": d_model,
        "n_heads": n_heads,
        "d_ff": d_ff,
        "bptt_window": bptt_window,
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
        "weighted_theoretical_bits_per_element": ac_stats[
            "weighted_theoretical_bits_per_element"
        ],
        "comparison_brotli_optuna_bytes": REFERENCE_BROTLI_OPTUNA_BYTES,
        "delta_vs_brotli_optuna": (
            archive_bytes - REFERENCE_BROTLI_OPTUNA_BYTES
        ),
        "model_spec_match": {
            "predicted": (
                "Tiny transformer (1-2 layers, ~1-5K params) "
                "with self-attention"
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
    p.add_argument("--d-model", type=int, default=8)
    p.add_argument("--n-heads", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=16)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--bptt-window", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--output-evidence", type=Path, default=None)
    args = p.parse_args(argv)

    if not args.state_dict.is_file():
        raise SystemExit(f"state_dict not found: {args.state_dict}")

    manifest = run_codec(
        args.state_dict,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_ff=args.d_ff,
        epochs=args.epochs,
        lr=args.lr,
        bptt_window=args.bptt_window,
        seed=args.seed,
    )

    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    if args.output_json is None:
        out_dir = REPO_ROOT / f"reports/raw/pr101_tiny_nn_transformer_{ts}"
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
        f"(raw {manifest['model_blob_raw_bytes']} B)"
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
            "technique": "tiny_nn_pmf_predictor_transformer_faithful",
            "empirical_archive_bytes": manifest["archive_bytes"],
            "model_total_params": manifest["model_total_params"],
            **proxy_evidence_contract(),
            "source": (
                f"{EVIDENCE_GRADE} {args.output_json} "
                f"(faithful 1:1 transformer implementation; "
                f"params={manifest['model_total_params']}, "
                f"model brotli={manifest['model_blob_brotli_bytes']} B)"
            ),
            "timestamp": _dt.datetime.now(_dt.UTC).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "model_spec_fidelity_audit": manifest["model_spec_match"],
            "audit_criterion": "transformer_pmf_predictor",
        }
        args.output_evidence.parent.mkdir(parents=True, exist_ok=True)
        with args.output_evidence.open("a", encoding="utf-8") as f:
            f.write(json.dumps(evidence_row) + "\n")
        print(f"\nevidence row appended: {args.output_evidence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
