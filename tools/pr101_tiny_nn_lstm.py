#!/usr/bin/env python3
"""PR101 tiny_nn LSTM PMF predictor — faithful 1:1 architecture-class test.

Background (2026-05-08 implementation-vs-model audit):
- The cathedral_autopilot tiny_nn row predicted "200-param MLP". Three
  alternative architecture classes were enumerated as REMAINING audit
  criteria but never implemented at faithful capacity:
    * LSTM-based PMF predictor (sequential, prev_symbol -> next-symbol PMF)
    * Transformer-based PMF predictor (self-attention)
    * Cross-tensor context predictor (neighbor-tensor stats as features)

This tool is the LSTM variant. It uses a tiny LSTM (~500-2K params)
that scans each tensor's symbol stream sequentially and predicts the
next symbol's PMF as a Gaussian (mean, log_scale).

Architecture (1:1 with audit criterion):
  Input:  prev_symbol embedding (N_SYMBOLS+1 -> embed_dim) +
          tensor_id embedding  (N_TENSORS  -> embed_dim)
  LSTM:   nn.LSTM(input_size=2*embed_dim, hidden_size=H, num_layers=1)
  Output: Linear(H, 2) -> (mean, log_scale)
  Param count target: ~500-2K

Why this matters:
- The 188-param MLP just falsified at predicted capacity.
- LSTMs capture temporal context (Markov order > 1) that a stateless
  MLP cannot. If PR101 symbol streams have any sequential structure
  beyond first-order Markov (the prior MLP+prev_symbol variant was
  rank=8 factorized, not LSTM), this tool is the architecture class
  that detects it.
- An LSTM at ~1K params is qualitatively different from the 188-param
  MLP — different inductive bias entirely.

CLAUDE.md compliance:
- Pure CPU/MPS (NEVER scorer-load).
- proxy_evidence_contract: ready_for_exact_eval_dispatch=False from
  any CPU/MPS evidence — only [contest-CUDA] auth eval can flip it.
- Every reported byte count is `[CPU-prep faithful LSTM test]`-tagged.
- No score claim. No promotion eligibility. Byte-anchor only.
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

TOOL_NAME = "tools/pr101_tiny_nn_lstm.py"
SCHEMA_VERSION = "pr101_tiny_nn_lstm.v1"
ARCHIVE_OVERHEAD_BYTES = 16_094
N_TENSORS = len(FIXED_STATE_SCHEMA)
N_SYMBOLS = 2 * N_QUANT + 1  # symmetric int8 range [-127,127] = 255
START_SYMBOL = N_SYMBOLS  # sentinel for sequence start
EVIDENCE_GRADE = "[CPU-prep faithful LSTM test]"
EVIDENCE_SEMANTICS = "cpu_faithful_lstm_pmf_byte_anchor_no_score"
DISPATCH_BLOCKERS = (
    "no_runtime_decoder_packet_built",
    "missing_exact_cuda_auth_eval",
    "requires_exact_cuda_auth_eval_before_any_score_use",
    "cpu_proxy_byte_anchor_not_score_evidence",
)
REFERENCE_BROTLI_OPTUNA_BYTES = 178_144  # canonical PR101 brotli baseline


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


class TinyLSTMPredictor(nn.Module):
    """Tiny LSTM that emits per-position (mean, log_scale) for a Gaussian PMF.

    Architecture:
      prev_symbol (LongTensor[B,T]) ---> Embedding(N_SYMBOLS+1, E)
      tensor_id   (LongTensor[B,T]) ---> Embedding(N_TENSORS,   E)
      concat -> LSTM(2E -> H) -> Linear(H, 2) -> (mean, log_scale)

    With E=4, H=8, n_tensors=28, n_symbols=255: ~1240 params.
    """

    def __init__(
        self,
        n_tensors: int = N_TENSORS,
        n_symbols: int = N_SYMBOLS,
        embed_dim: int = 4,
        hidden: int = 8,
    ):
        super().__init__()
        self.n_tensors = n_tensors
        self.n_symbols = n_symbols
        self.embed_dim = embed_dim
        self.hidden = hidden
        self.sym_emb = nn.Embedding(n_symbols + 1, embed_dim)  # +1 for START
        self.tensor_emb = nn.Embedding(n_tensors, embed_dim)
        self.lstm = nn.LSTM(
            input_size=2 * embed_dim,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
        )
        self.head = nn.Linear(hidden, 2)

    def forward(
        self,
        prev_symbol: torch.Tensor,
        tensor_id: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # prev_symbol, tensor_id: [B, T] long
        s = self.sym_emb(prev_symbol)  # [B, T, E]
        t = self.tensor_emb(tensor_id)  # [B, T, E]
        x = torch.cat([s, t], dim=-1)  # [B, T, 2E]
        h, _ = self.lstm(x)  # [B, T, H]
        out = self.head(h)  # [B, T, 2]
        mean = out[..., 0]
        log_scale = out[..., 1].clamp(-3.0, 6.0)
        return mean, log_scale


def gaussian_log_pmf(
    symbol: torch.Tensor, mean: torch.Tensor, log_scale: torch.Tensor
) -> torch.Tensor:
    """Discrete Gaussian PMF on symmetric int8 grid [-N_QUANT, N_QUANT]."""
    scale = torch.exp(log_scale)
    upper = (symbol.float() + 0.5 - mean) / (scale * 1.4142135)
    lower = (symbol.float() - 0.5 - mean) / (scale * 1.4142135)
    pmf = 0.5 * (torch.erf(upper) - torch.erf(lower))
    return torch.log(pmf.clamp_min(1e-12))


def collect_symbols(
    state_dict_path: Path,
) -> tuple[list[np.ndarray], list[float]]:
    """Quantize each tensor in FIXED_STATE_SCHEMA -> int8 symbols + scales."""
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
    """One sequence per tensor: (prev_sym [N], tensor_id [N], target [N])."""
    prev_seqs: list[torch.Tensor] = []
    tid_seqs: list[torch.Tensor] = []
    tgt_seqs: list[torch.Tensor] = []
    for t_idx, syms in enumerate(sym_per_tensor):
        n = syms.size
        if n == 0:
            continue
        # Map signed symbols [-N_QUANT, N_QUANT] -> unsigned [0, 2N_QUANT]
        unsigned = (syms.flatten().astype(np.int64) + N_QUANT)
        prev = np.empty(n, dtype=np.int64)
        prev[0] = START_SYMBOL
        prev[1:] = unsigned[:-1]
        tid = np.full(n, t_idx, dtype=np.int64)
        prev_seqs.append(torch.from_numpy(prev))
        tid_seqs.append(torch.from_numpy(tid))
        # target stays in signed coords for Gaussian PMF training
        tgt_seqs.append(torch.from_numpy(syms.flatten().astype(np.int64)))
    return prev_seqs, tid_seqs, tgt_seqs


def train_lstm(
    sym_per_tensor: list[np.ndarray],
    *,
    embed_dim: int,
    hidden: int,
    epochs: int,
    lr: float,
    bptt_window: int,
    seed: int,
) -> TinyLSTMPredictor:
    torch.manual_seed(seed)
    model = TinyLSTMPredictor(embed_dim=embed_dim, hidden=hidden)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[lstm] total params: {n_params}")
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    prev_seqs, tid_seqs, tgt_seqs = _build_sequences(sym_per_tensor)
    n_total = sum(int(s.numel()) for s in prev_seqs)
    print(f"[lstm] training on {n_total:,} symbols across {len(prev_seqs)} tensors")

    # We train with BPTT over fixed windows. Long tensors get many windows;
    # short tensors get one full pass.
    g = torch.Generator().manual_seed(seed)
    for ep in range(epochs):
        ep_loss = 0.0
        n_batches = 0
        # Build all (tensor_idx, start, end) windows for this epoch
        windows: list[tuple[int, int, int]] = []
        for t_idx, prev in enumerate(prev_seqs):
            n = int(prev.numel())
            if n <= bptt_window:
                windows.append((t_idx, 0, n))
            else:
                # Slide non-overlapping windows
                for s in range(0, n, bptt_window):
                    windows.append((t_idx, s, min(s + bptt_window, n)))
        # Shuffle window order
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
            print(f"  ep {ep+1:3d}/{epochs}: avg_bits={avg_bits:.4f}/symbol")
    return model


def encode_with_lstm(
    sym_per_tensor: list[np.ndarray],
    model: TinyLSTMPredictor,
) -> tuple[bytes, dict]:
    """Encode each tensor's symbol stream with LSTM-predicted Gaussian PMF.

    Returns concatenated AC payloads (with length prefix per stream) and
    summary stats. Per CLAUDE.md, this builds a REAL constriction range
    bitstream (not just a theoretical NLL estimate).
    """
    model.eval()
    prev_seqs, tid_seqs, _tgt_seqs = _build_sequences(sym_per_tensor)
    all_payloads: list[bytes] = []
    total_n = 0
    total_bits = 0.0
    qg = constriction.stream.model.QuantizedGaussian(-N_QUANT, N_QUANT)
    with torch.no_grad():
        idx_offset = 0
        for t_idx, syms in enumerate(sym_per_tensor):
            n = syms.size
            if n == 0:
                all_payloads.append(b"")
                continue
            prev = prev_seqs[idx_offset].unsqueeze(0)
            tid = tid_seqs[idx_offset].unsqueeze(0)
            idx_offset += 1
            mean, log_scale = model(prev, tid)
            mean_arr = mean.squeeze(0).cpu().numpy().astype(np.float64)
            scale_arr = (
                torch.exp(log_scale.squeeze(0)).cpu().numpy().astype(np.float64)
            )
            scale_arr = np.maximum(scale_arr, 0.05)
            symbols_signed = syms.flatten().astype(np.int32)
            encoder = constriction.stream.queue.RangeEncoder()
            encoder.encode(symbols_signed, qg, mean_arr, scale_arr)
            payload = bytes(encoder.get_compressed())
            all_payloads.append(payload)
            # Theoretical bits (advisory; the AC payload is what gets shipped).
            log_pmf = gaussian_log_pmf(
                torch.from_numpy(syms.flatten().astype(np.float32)),
                torch.from_numpy(mean_arr.astype(np.float32)),
                torch.log(torch.from_numpy(scale_arr.astype(np.float32))),
            )
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
    model: TinyLSTMPredictor, scales: list[float]
) -> bytes:
    """Serialize LSTM weights as fp16 + per-tensor scales as fp16."""
    buf = bytearray()
    buf += b"LSTM"  # magic
    buf += struct.pack("<I", N_TENSORS)
    for s in scales:
        buf += np.float16(s).tobytes()
    # Order: state_dict ordering is deterministic in PyTorch
    for _name, p in model.state_dict().items():
        flat = p.detach().cpu().numpy().astype(np.float16).flatten()
        buf += struct.pack("<I", flat.size)
        buf += flat.tobytes()
    return bytes(buf)


def run_codec(
    state_dict_path: Path,
    *,
    embed_dim: int,
    hidden: int,
    epochs: int,
    lr: float,
    bptt_window: int,
    seed: int,
) -> dict:
    sym_per_tensor, scales = collect_symbols(state_dict_path)
    n_total = sum(s.size for s in sym_per_tensor)
    print(
        f"[lstm] PR101 substrate: {n_total:,} symbols, "
        f"{len(sym_per_tensor)} tensors"
    )

    model = train_lstm(
        sym_per_tensor,
        embed_dim=embed_dim,
        hidden=hidden,
        epochs=epochs,
        lr=lr,
        bptt_window=bptt_window,
        seed=seed,
    )
    n_params = sum(p.numel() for p in model.parameters())

    payload_concat, ac_stats = encode_with_lstm(sym_per_tensor, model)
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
        "embed_dim": embed_dim,
        "hidden": hidden,
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
        "delta_vs_brotli_optuna": archive_bytes - REFERENCE_BROTLI_OPTUNA_BYTES,
        "model_spec_match": {
            "predicted": (
                "LSTM-based PMF predictor (~500-2K params); per-tensor "
                "sequential prediction"
            ),
            "actual_params": n_params,
            "actual_model_brotli_bytes": len(model_brotli),
            "1:1_fidelity": 500 <= n_params <= 2_000,
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
    p.add_argument("--embed-dim", type=int, default=4)
    p.add_argument("--hidden", type=int, default=8)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--bptt-window", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--output-evidence", type=Path, default=None)
    args = p.parse_args(argv)

    if not args.state_dict.is_file():
        raise SystemExit(f"state_dict not found: {args.state_dict}")

    manifest = run_codec(
        args.state_dict,
        embed_dim=args.embed_dim,
        hidden=args.hidden,
        epochs=args.epochs,
        lr=args.lr,
        bptt_window=args.bptt_window,
        seed=args.seed,
    )

    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    if args.output_json is None:
        out_dir = REPO_ROOT / f"reports/raw/pr101_tiny_nn_lstm_{ts}"
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
            "technique": "tiny_nn_pmf_predictor_lstm_faithful",
            "empirical_archive_bytes": manifest["archive_bytes"],
            "model_total_params": manifest["model_total_params"],
            **proxy_evidence_contract(),
            "source": (
                f"{EVIDENCE_GRADE} {args.output_json} "
                f"(faithful 1:1 LSTM implementation; "
                f"params={manifest['model_total_params']}, "
                f"model brotli={manifest['model_blob_brotli_bytes']} B)"
            ),
            "timestamp": _dt.datetime.now(_dt.UTC).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
            "model_spec_fidelity_audit": manifest["model_spec_match"],
            "audit_criterion": "lstm_pmf_predictor",
        }
        args.output_evidence.parent.mkdir(parents=True, exist_ok=True)
        with args.output_evidence.open("a", encoding="utf-8") as f:
            f.write(json.dumps(evidence_row) + "\n")
        print(f"\nevidence row appended: {args.output_evidence}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
