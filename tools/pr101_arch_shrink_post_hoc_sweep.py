#!/usr/bin/env python3
"""PR101 post-hoc arch_shrink byte sweep — anchors the
``arch_shrink_x0.4_quantizr_class`` row in cathedral_autopilot's catalog.

Method (post-hoc, no retraining):
  For each shrink ratio r in {0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0}:
    For each tensor in PR101's state_dict:
      Truncate the weight tensor to keep only floor(d * r) channels
      (channel pruning by L2 magnitude, last-axis primarily).
    Pack the truncated state_dict via the canonical PR101 brotli encoder.
    Measure archive bytes.

This is a BYTE-ANCHOR for the catalog row (predicted_archive_bytes 80,000
at r=0.4 = 88K-element Quantizr-class). The SCORE impact requires
retraining on the smaller arch — explicitly NOT measured here. Tag:
``[CPU-prep empirical byte-anchor only]``.

Per the council mandate, architecture is where the 5-10x byte headroom
lives. This tool gives the byte-anchor; A3.1 (full MPS training) gives
the score anchor.

CLAUDE.md compliance: pure CPU + numpy + brotli; no scorer load; no
contest score claims; output tagged ``[CPU-prep empirical byte-anchor only]``.
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

TOOL_NAME = "tools/pr101_arch_shrink_post_hoc_sweep.py"
SCHEMA_VERSION = "pr101_arch_shrink_post_hoc_sweep.v1"
ARCHIVE_OVERHEAD_BYTES = 16_094
EVIDENCE_GRADE = "[CPU-prep empirical byte-anchor only]"
DISPATCH_BLOCKERS = [
    "post_hoc_arch_truncate_not_retrained",
    "score_impact_unknown_without_contest_cuda",
    "no_archive_substitution_performed",
    "missing_exact_cuda_auth_eval",
    "channel_truncation_breaks_inference_until_retrained",
]


def channel_l2_magnitude(tensor: np.ndarray) -> np.ndarray:
    """L2 norm per output channel (axis 0 by convention for HNeRV decoder weights)."""
    if tensor.ndim == 1:
        return np.abs(tensor)
    flat = tensor.reshape(tensor.shape[0], -1)
    return np.sqrt((flat ** 2).sum(axis=1))


def truncate_to_top_channels(tensor: np.ndarray, n_keep: int) -> np.ndarray:
    """Keep top-n_keep channels by L2 magnitude. Preserves axis 0 ordering."""
    if tensor.ndim == 0 or tensor.size == 0:
        return tensor
    n_total = tensor.shape[0]
    if n_keep >= n_total:
        return tensor
    if n_keep <= 0:
        # Edge case: keep at least 1 channel
        n_keep = 1
    mags = channel_l2_magnitude(tensor)
    top_idx = np.argsort(mags)[-n_keep:]
    top_idx_sorted = np.sort(top_idx)
    return tensor[top_idx_sorted]


def sweep_arch_shrink(state_dict_path: Path, ratios: list[float]) -> dict:
    """Truncate each tensor's channels by ratio, brotli, measure."""
    import torch
    sd = torch.load(state_dict_path, map_location="cpu", weights_only=False)
    if not isinstance(sd, dict):
        raise SystemExit(f"state_dict at {state_dict_path} is not a dict")

    # Collect original tensors as numpy
    originals: list[tuple[str, np.ndarray]] = []
    n_total_elements = 0
    for name, _shape in FIXED_STATE_SCHEMA:
        t = sd[name].detach().cpu().to(torch.float32).numpy()
        originals.append((name, t))
        n_total_elements += t.size

    rows: list[dict] = []
    for r in ratios:
        per_tensor_stats: list[dict] = []
        truncated_payloads: list[bytes] = []
        scales: list[float] = []
        truncated_total_elements = 0

        for name, t_orig in originals:
            n_channels = t_orig.shape[0] if t_orig.ndim >= 1 else 1
            n_keep = max(1, int(round(n_channels * r)))
            t_trunc = truncate_to_top_channels(t_orig, n_keep)
            truncated_total_elements += t_trunc.size

            # Quantize the truncated tensor with the same int8 path PR101 uses
            qt_orig = _quantize_tensor(name, sd[name], n_quant=N_QUANT)  # for scale comparison
            scale_f = float(np.abs(t_trunc).max() / N_QUANT) if t_trunc.size and float(np.abs(t_trunc).max()) > 0 else float(qt_orig.scale)
            scales.append(scale_f)
            symbols_i8 = np.clip(
                np.round(t_trunc.flatten() / max(scale_f, 1e-12)).astype(np.int32),
                -N_QUANT, +N_QUANT,
            ).astype(np.int8)
            payload = struct.pack("<I", symbols_i8.size) + symbols_i8.tobytes()
            truncated_payloads.append(payload)
            per_tensor_stats.append({
                "name": name,
                "n_channels_orig": int(n_channels),
                "n_channels_kept": int(n_keep),
                "n_elements_orig": int(t_orig.size),
                "n_elements_kept": int(t_trunc.size),
            })

        scales_blob = np.array(scales, dtype=np.float16).tobytes()
        full_blob = scales_blob + b"".join(truncated_payloads)
        compressed = brotli.compress(full_blob, quality=11, lgwin=16, lgblock=19)
        archive_bytes = len(compressed) + ARCHIVE_OVERHEAD_BYTES
        rows.append({
            "shrink_ratio": r,
            "n_elements_orig": n_total_elements,
            "n_elements_kept": truncated_total_elements,
            "fraction_kept": truncated_total_elements / max(n_total_elements, 1),
            "raw_payload_bytes": len(full_blob),
            "brotli_bytes": len(compressed),
            "archive_bytes": archive_bytes,
            "per_tensor": per_tensor_stats,
        })

    rows.sort(key=lambda r: r["archive_bytes"])
    return {
        "schema": SCHEMA_VERSION,
        "tool": TOOL_NAME,
        "evidence_grade": EVIDENCE_GRADE,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "score_affecting_payload_changed": True,
        "charged_bits_changed": True,
        "dispatch_blockers": DISPATCH_BLOCKERS,
        "input_state_dict": str(state_dict_path),
        "ratios_swept": ratios,
        "n_total_elements_orig": n_total_elements,
        "best_archive_bytes": rows[0]["archive_bytes"],
        "best_shrink_ratio": rows[0]["shrink_ratio"],
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
    p.add_argument("--ratios", type=float, nargs="+",
                   default=[0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    p.add_argument("--output-json", type=Path, default=None)
    p.add_argument("--output-evidence", type=Path, default=None)
    args = p.parse_args(argv)

    if not args.state_dict.is_file():
        raise SystemExit(f"state_dict not found: {args.state_dict}")

    manifest = sweep_arch_shrink(args.state_dict, args.ratios)

    ts = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    if args.output_json is None:
        out_dir = REPO_ROOT / f"reports/raw/pr101_arch_shrink_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output_json = out_dir / "manifest.json"
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"\nmanifest: {args.output_json}\n")
    print(f"  ratio | fraction_kept | archive_bytes")
    for r in sorted(manifest["rows"], key=lambda r: r["shrink_ratio"]):
        print(f"  {r['shrink_ratio']:>5.2f} | {r['fraction_kept']:>13.4f} | {r['archive_bytes']:>14,}")
    print(f"\nbest_ratio: {manifest['best_shrink_ratio']}, archive_bytes: {manifest['best_archive_bytes']:,} B")

    # Anchor the r=0.4 row for cathedral_autopilot
    r_04_row = next((r for r in manifest["rows"] if abs(r["shrink_ratio"] - 0.4) < 1e-6), None)
    if args.output_evidence and r_04_row is not None:
        evidence_row = {
            "technique": "arch_shrink_x0.4_quantizr_class",
            "empirical_archive_bytes": r_04_row["archive_bytes"],
            "source": (
                f"[CPU-prep empirical byte-anchor only] {args.output_json} "
                f"(ratio=0.4, post-hoc no retrain; channel L2-truncate)"
            ),
            "timestamp": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "contest_dispatch_verdict": "byte_anchor_only_score_unknown",
        }
        args.output_evidence.parent.mkdir(parents=True, exist_ok=True)
        with args.output_evidence.open("a", encoding="utf-8") as f:
            f.write(json.dumps(evidence_row) + "\n")
        print(f"\nevidence row appended: {args.output_evidence}")

    print(f"\nNOTE: byte-anchor only. SCORE impact of post-hoc arch truncation is")
    print(f"unknown without retraining. Catalog row arch_shrink_x0.4_quantizr_class")
    print(f"remains DEFERRED for score promotion until [contest-CUDA] eval lands.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
