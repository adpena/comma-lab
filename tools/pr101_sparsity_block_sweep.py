#!/usr/bin/env python3
"""PR101 post-hoc sparsity sweep — anchors the
``sparsity_alpha_0.7_imp_retrain`` row in cathedral_autopilot's catalog.

Method (post-hoc, no retraining):
  For each sparsity level alpha in {0.5, 0.6, 0.7, 0.8, 0.9}:
    Compute per-tensor magnitude-based threshold so alpha fraction of
    weights have |w| < threshold; zero those weights.
    Pack as CSR: nz_indices (uint16) + nz_values (int8 quantized) + brotli.
    Measure archive bytes.

This is a BYTE-ANCHOR for the catalog row (predicted_archive_bytes 65,000).
The SCORE impact of sparsity requires retraining + contest-CUDA eval —
explicitly NOT measured here. Tag: [CPU-prep empirical byte-anchor only].

Council mandate (2026-05-07 grand council on path forward): the encoder
lane is SATURATED at 178KB. Architecture lane (sparsity, arch_shrink) is
where the 5-10x headroom lives. This tool is the byte-anchor on the
architecture lane.

CLAUDE.md compliance: pure CPU + numpy + brotli; no scorer load; no
contest score claims; output tagged [CPU-prep empirical byte-anchor only].
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import struct
import sys
from pathlib import Path

import brotli
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.pr101_split_brotli_codec import (  # noqa: E402
    FIXED_STATE_SCHEMA,
    N_QUANT,
    _quantize_tensor,
)

TOOL_NAME = "tools/pr101_sparsity_block_sweep.py"
SCHEMA_VERSION = "pr101_sparsity_block_sweep.v1"
ARCHIVE_OVERHEAD_BYTES = 16_094
EVIDENCE_GRADE = "[CPU-prep empirical byte-anchor only]"
DISPATCH_BLOCKERS = [
    "post_hoc_sparsity_not_retrained",
    "score_impact_unknown_without_contest_cuda",
    "no_archive_substitution_performed",
    "missing_exact_cuda_auth_eval",
    "no_decoder_packet_built",
]


def sparsify_tensor_int8(symbols_i8: np.ndarray, alpha: float) -> tuple[np.ndarray, np.ndarray, int]:
    """Zero out smallest-magnitude alpha fraction. Return (nz_indices, nz_values, n_total).

    Indices are uint32 (we have up to ~50K elements per tensor; uint32 safe).
    Values are int8.
    """
    n = symbols_i8.size
    if alpha <= 0 or n == 0:
        return np.arange(n, dtype=np.uint32), symbols_i8.flatten().astype(np.int8), n
    abs_vals = np.abs(symbols_i8.flatten().astype(np.int32))
    # Threshold: alpha fraction get zeroed
    n_keep = max(1, int(round((1.0 - alpha) * n)))
    if n_keep >= n:
        return np.arange(n, dtype=np.uint32), symbols_i8.flatten().astype(np.int8), n
    # Pick top n_keep by absolute magnitude
    top_idx = np.argpartition(abs_vals, n - n_keep)[n - n_keep:]
    top_idx_sorted = np.sort(top_idx)
    nz_values = symbols_i8.flatten()[top_idx_sorted].astype(np.int8)
    return top_idx_sorted.astype(np.uint32), nz_values, n


def encode_csr(nz_indices: np.ndarray, nz_values: np.ndarray, n_total: int) -> bytes:
    """Encode (indices_u32, values_i8, n_total) as a delta-coded payload.

    Use delta encoding on indices then varbyte (or just keep u32 — brotli will
    delta-encode internally).
    """
    if nz_values.size == 0:
        return struct.pack("<II", n_total, 0)
    # Delta-encode indices to give brotli more redundancy
    deltas = np.diff(np.concatenate([np.array([0], dtype=np.uint32), nz_indices])).astype(np.uint32)
    return (
        struct.pack("<II", n_total, nz_values.size)
        + deltas.tobytes()
        + nz_values.tobytes()
    )


def sweep_sparsity(state_dict_path: Path, alphas: list[float]) -> dict:
    """For each alpha, build a sparse CSR-style payload + brotli, measure."""
    import torch
    sd = torch.load(state_dict_path, map_location="cpu", weights_only=False)  # WEIGHTS_ONLY_FALSE_OK:trusted-PR101-substrate-state-dict-local-artifact
    if not isinstance(sd, dict):
        raise SystemExit(f"state_dict at {state_dict_path} is not a dict")

    # Quantize once
    quantized: list[tuple[str, np.ndarray, float]] = []
    for name, _shape in FIXED_STATE_SCHEMA:
        qt = _quantize_tensor(name, sd[name], n_quant=N_QUANT)
        quantized.append((name, qt.q_i8.flatten(), float(qt.scale)))
    total_elements = sum(q.size for _, q, _ in quantized)

    rows: list[dict] = []
    for alpha in alphas:
        all_payloads: list[bytes] = []
        n_zeroed_total = 0
        scales: list[float] = []
        for name, syms, scale in quantized:
            scales.append(scale)
            nz_idx, nz_val, n = sparsify_tensor_int8(syms, alpha)
            payload = encode_csr(nz_idx, nz_val, n)
            all_payloads.append(payload)
            n_zeroed_total += (n - nz_val.size)
        # scales blob (per-tensor fp16)
        scales_blob = np.array(scales, dtype=np.float16).tobytes()
        # Concat all
        full_blob = scales_blob + b"".join(
            struct.pack("<I", len(p)) + p for p in all_payloads
        )
        compressed = brotli.compress(full_blob, quality=11, lgwin=16, lgblock=19)
        archive_bytes = len(compressed) + ARCHIVE_OVERHEAD_BYTES
        rows.append({
            "alpha": alpha,
            "n_zeroed": int(n_zeroed_total),
            "fraction_zeroed": n_zeroed_total / max(total_elements, 1),
            "raw_payload_bytes": len(full_blob),
            "brotli_bytes": len(compressed),
            "archive_bytes": archive_bytes,
        })
    rows.sort(key=lambda r: r["archive_bytes"])
    return {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "evidence_grade": EVIDENCE_GRADE,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "score_affecting_payload_changed": True,  # zeroing weights changes payload meaning
        "dispatch_blockers": DISPATCH_BLOCKERS,
        "input_state_dict": str(state_dict_path),
        "alphas_swept": alphas,
        "n_total_elements": total_elements,
        "best_archive_bytes": rows[0]["archive_bytes"],
        "best_alpha": rows[0]["alpha"],
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--state-dict",
        type=Path,
        default=REPO_ROOT
        / "experiments/results/pr101_codecop_sweep_20260507_codex/pr101_decoder_state_dict.pt",
    )
    p.add_argument("--alphas", type=float, nargs="+",
                   default=[0.0, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9])
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--output-evidence", type=Path, default=None)
    args = p.parse_args(argv)

    if not args.state_dict.is_file():
        raise SystemExit(f"state_dict not found: {args.state_dict}")

    manifest = sweep_sparsity(args.state_dict, args.alphas)

    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    if args.output_json is None:
        out_dir = REPO_ROOT / f"reports/raw/pr101_sparsity_sweep_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output_json = out_dir / "manifest.json"
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nmanifest: {args.output_json}")
    print("\n  alpha | fraction_zeroed | archive_bytes")
    for r in sorted(manifest["rows"], key=lambda r: r["alpha"]):
        print(f"  {r['alpha']:>5.2f} | {r['fraction_zeroed']:>14.4f} | {r['archive_bytes']:>14,}")
    print(f"\nbest_alpha: {manifest['best_alpha']}, archive_bytes: {manifest['best_archive_bytes']:,} B")

    # Anchor the alpha=0.7 row for cathedral_autopilot
    alpha_07_row = next((r for r in manifest["rows"] if abs(r["alpha"] - 0.7) < 1e-6), None)
    if args.output_evidence and alpha_07_row is not None:
        evidence_row = {
            "technique": "sparsity_alpha_0.7_imp_retrain",
            "empirical_archive_bytes": alpha_07_row["archive_bytes"],
            "source": (
                f"[CPU-prep empirical byte-anchor only] {args.output_json} "
                f"(alpha=0.7, post-hoc no retrain)"
            ),
            "timestamp": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        args.output_evidence.parent.mkdir(parents=True, exist_ok=True)
        with args.output_evidence.open("a", encoding="utf-8") as f:
            f.write(json.dumps(evidence_row) + "\n")
        print(f"\nevidence row appended: {args.output_evidence}")

    print("\nNOTE: byte-anchor only. SCORE impact of post-hoc sparsity is unknown")
    print("and requires retraining + [contest-CUDA] auth eval. Catalog row")
    print("sparsity_alpha_0.7_imp_retrain remains DEFERRED for score promotion.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
