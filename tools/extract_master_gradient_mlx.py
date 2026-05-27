#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""$0-MLX-local per-pair + per-axis master-gradient CLI (sister of extract_master_gradient.py).

Runs the MLX SegNet+PoseNet scorer port (M5 Max GPU) as a fast forward oracle +
per-decoder-tensor finite differences to emit the canonical
``(N_archive_bytes, N_pairs, 3_axes)`` per-pair master gradient on the canonical
FRONTIER fec6/PR101 archive -- the authoritative per-pair signal the 5D canvas /
Dykstra Pareto solver / DROP-MANY beam search / bit_allocator need (they have
been running on archive-AGGREGATE or selector-INHERITED data).

NON-PROMOTABLE per CLAUDE.md "MLX portable-local-substrate authority" +
Catalog #192/#127/#323: macOS-MLX master gradient is RESEARCH-SIGNAL for the
closed-form PREDICTION sweep ONLY. NOT a contest score; NOT promotable. The
contest-CUDA/CPU exact-eval per Catalog #246 remains required before any
score/frontier/PR claim.

Canonical schema matches the 2026-05-18 8pair PyTorch layout:
    .omx/state/master_gradient_a1_headered_diagnostic_8pair_per_pair_20260518.npy
    shape (N_archive_bytes, N_pairs, 3) float64, axes = (seg, pose, rate).

Example::

    .venv/bin/python tools/extract_master_gradient_mlx.py \
        --archive experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip \
        --n-pairs 64 \
        --out .omx/state/master_gradient_fec6_frontier_mlx_per_pair_20260527.npy
"""

from __future__ import annotations

import argparse
import datetime as _dt
import fcntl
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.master_gradient import append_anchor_locked  # noqa: E402
from tac.master_gradient_mlx_extractor import (  # noqa: E402
    AXIS_ORDER,
    DEFAULT_FD_REL_EPS,
    EVIDENCE_GRADE_MLX,
    EVIDENCE_TAG_MLX,
    HARDWARE_SUBSTRATE_MLX,
    HEURISTIC_GRADIENT_BYTE_DOMAIN,
    HEURISTIC_GRADIENT_TENSOR_KIND,
    SCHEMA_VERSION,
    MLXMasterGradientError,
    build_mlx_master_gradient_anchor,
    extract_mlx_per_pair_master_gradient,
    mlx_master_gradient_anchor_blockers,
)


def _utc_now() -> str:
    return _dt.datetime.now(_dt.UTC).isoformat()


def _parse_axes(axes_arg: str) -> tuple[str, ...]:
    requested = tuple(a.strip() for a in axes_arg.split(",") if a.strip())
    for a in requested:
        if a not in AXIS_ORDER:
            raise SystemExit(f"unknown axis {a!r}; valid axes: {AXIS_ORDER}")
    return requested


def _locked_save_npy(out_path: Path, arr: np.ndarray) -> None:
    """fcntl-locked atomic .npy write per Catalog #131/#138."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = out_path.with_name(f"{out_path.name}.lock")
    # np.save appends ".npy" if the path lacks it; use an explicit .npy tmp so the
    # post-save replace finds the actual written file.
    tmp_path = out_path.with_name(
        f"{out_path.name}.tmp.{_dt.datetime.now().strftime('%H%M%S%f')}.npy"
    )
    with lock_path.open("a", encoding="utf-8") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        try:
            np.save(tmp_path, arr)
            tmp_path.replace(out_path)
        finally:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)


def _append_jsonl_locked(row: dict[str, object], *, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = output_path.with_name(f"{output_path.name}.lock")
    line = json.dumps(row, sort_keys=True)
    with lock_path.open("a", encoding="utf-8") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        try:
            with output_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        finally:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="$0-MLX-local per-pair per-axis master-gradient extractor "
        "(NON-PROMOTABLE macOS-MLX research-signal per Catalog #192)."
    )
    parser.add_argument(
        "--archive",
        required=True,
        type=Path,
        help="Path to the fec6/PR101 frontier archive.zip (or raw payload).",
    )
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=64,
        help="Number of pairs to extract per-pair gradient for (default 64; full=589-600).",
    )
    parser.add_argument(
        "--n-pairs-total",
        type=int,
        default=600,
        help="Total contest pair count for authority metadata (default 600).",
    )
    parser.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output .npy path (canonical (N_bytes, N_pairs, 3) schema).",
    )
    parser.add_argument(
        "--axes",
        default="seg,pose,rate",
        help="Comma-separated axes to retain in metadata (default seg,pose,rate). "
        "The .npy always carries all 3 columns; this only annotates intent.",
    )
    parser.add_argument(
        "--fd-rel-eps",
        type=float,
        default=DEFAULT_FD_REL_EPS,
        help=f"Relative FD epsilon (multiple of per-tensor RMS; default {DEFAULT_FD_REL_EPS}).",
    )
    parser.add_argument(
        "--pair-batch-size",
        type=int,
        default=16,
        help="Number of frame pairs to decode/score per MLX batch (default 16).",
    )
    parser.add_argument(
        "--upstream-dir",
        type=Path,
        default=REPO_ROOT / "upstream",
        help="Upstream dir for scorer weights (default REPO_ROOT/upstream).",
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        default=REPO_ROOT / "upstream" / "videos" / "0.mkv",
        help="Ground-truth contest video (default upstream/videos/0.mkv).",
    )
    parser.add_argument(
        "--manifest-jsonl",
        type=Path,
        default=REPO_ROOT / ".omx" / "state" / "mlx_research_signal_manifest.jsonl",
        help="Canonical MLX research-signal manifest JSONL (NON-PROMOTABLE rows).",
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Skip the manifest JSONL append (smoke/dry-run mode).",
    )
    parser.add_argument(
        "--anchor-jsonl",
        type=Path,
        default=REPO_ROOT / ".omx" / "state" / "master_gradient_anchors.jsonl",
        help="Canonical master-gradient anchor JSONL consumed by planners.",
    )
    parser.add_argument(
        "--write-anchor",
        action="store_true",
        help=(
            "Append a planner-consumable master-gradient anchor only if the "
            "result proves canonical byte-domain/source-runtime eligibility."
        ),
    )
    parser.add_argument(
        "--call-id",
        default=None,
        help="Optional extraction call id / run label for the master-gradient anchor.",
    )
    parser.add_argument("--verbose", action="store_true", help="Per-tensor FD progress logging.")
    args = parser.parse_args(argv)

    axes = _parse_axes(args.axes)

    if args.n_pairs <= 0:
        raise SystemExit("--n-pairs must be positive")
    if args.pair_batch_size <= 0:
        raise SystemExit("--pair-batch-size must be positive")

    try:
        result = extract_mlx_per_pair_master_gradient(
            args.archive,
            upstream_dir=args.upstream_dir,
            video_path=args.video_path,
            n_pairs_used=args.n_pairs,
            n_pairs_total=args.n_pairs_total,
            fd_rel_eps=args.fd_rel_eps,
            pair_batch_size=args.pair_batch_size,
            verbose=args.verbose,
        )
    except MLXMasterGradientError as exc:
        print(f"[mlx-master-gradient] FATAL: {exc}", file=sys.stderr)
        return 1

    _locked_save_npy(args.out, result.per_pair_per_byte)

    captured_at_utc = _utc_now()
    anchor_written = False
    anchor_blockers = mlx_master_gradient_anchor_blockers(result)
    if args.write_anchor:
        if anchor_blockers:
            print(
                "[mlx-master-gradient] refusing master-gradient anchor: "
                + ", ".join(anchor_blockers),
                file=sys.stderr,
            )
            return 1
        anchor = build_mlx_master_gradient_anchor(
            result,
            gradient_array_path=args.out,
            measurement_call_id=args.call_id,
            measurement_utc=captured_at_utc,
        )
        append_anchor_locked(anchor, path=args.anchor_jsonl)
        anchor_written = True

    # Sidecar metadata JSON (canonical custody).
    sidecar = {
        "schema_version": SCHEMA_VERSION,
        "npy_path": str(args.out),
        "npy_shape": list(result.per_pair_per_byte.shape),
        "axis_order": list(AXIS_ORDER),
        "axes_requested": list(axes),
        "archive_path": str(args.archive),
        "archive_sha256": result.archive_sha256,
        "archive_bytes_count": result.archive_bytes_count,
        "n_pairs_used": result.n_pairs_used,
        "n_pairs_total": result.n_pairs_total,
        "fd_rel_eps": result.fd_rel_eps,
        "pair_batch_size": args.pair_batch_size,
        "n_decoder_tensors": result.n_decoder_tensors,
        "decompressed_decoder_len": result.decompressed_decoder_len,
        "decoder_blob_offset": result.decoder_blob_offset,
        "operating_point": result.operating_point,
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        "hardware_substrate": HARDWARE_SUBSTRATE_MLX,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "promotable": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "measurement_method": result.metadata["measurement_method"],
        "gradient_tensor_kind": result.metadata.get(
            "gradient_tensor_kind",
            HEURISTIC_GRADIENT_TENSOR_KIND,
        ),
        "gradient_byte_domain": result.metadata.get(
            "gradient_byte_domain",
            HEURISTIC_GRADIENT_BYTE_DOMAIN,
        ),
        "codec_grammar": result.metadata["codec_grammar"],
        "captured_at_utc": captured_at_utc,
        "master_gradient_anchor_path": str(args.anchor_jsonl),
        "master_gradient_anchor_written": anchor_written,
        "master_gradient_anchor_blockers": anchor_blockers,
        "tool": "tools/extract_master_gradient_mlx.py",
        "canonical_helper": "tac.master_gradient_mlx_extractor.extract_mlx_per_pair_master_gradient",
        "non_promotable_note": (
            "macOS-MLX research-signal per CLAUDE.md 'MLX portable-local-substrate "
            "authority' + Catalog #192/#127/#323. Gates the closed-form PREDICTION "
            "sweep ONLY; contest-CUDA/CPU exact-eval per Catalog #246 required before "
            "any score/frontier/PR claim."
        ),
    }
    sidecar_path = args.out.with_suffix(args.out.suffix + ".meta.json")
    sidecar_lock = sidecar_path.with_name(f"{sidecar_path.name}.lock")
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    with sidecar_lock.open("a", encoding="utf-8") as lock_fh:
        fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
        try:
            sidecar_path.write_text(json.dumps(sidecar, indent=2, sort_keys=True) + "\n")
        finally:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)

    # Append the NON-PROMOTABLE manifest row to the canonical MLX research-signal posterior.
    if not args.no_manifest:
        manifest_row = {
            "kind": "mlx_per_pair_master_gradient",
            "archive_sha256": result.archive_sha256,
            "archive_path": str(args.archive),
            "npy_path": str(args.out),
            "npy_shape": list(result.per_pair_per_byte.shape),
            "n_pairs_used": result.n_pairs_used,
            "n_pairs_total": result.n_pairs_total,
            "axis_order": list(AXIS_ORDER),
            "operating_point": result.operating_point,
            "fd_rel_eps": result.fd_rel_eps,
            "pair_batch_size": args.pair_batch_size,
            "evidence_grade": EVIDENCE_GRADE_MLX,
            "evidence_tag": EVIDENCE_TAG_MLX,
            "hardware_substrate": HARDWARE_SUBSTRATE_MLX,
            "measurement_method": result.metadata["measurement_method"],
            "gradient_tensor_kind": sidecar["gradient_tensor_kind"],
            "gradient_byte_domain": sidecar["gradient_byte_domain"],
            "score_claim": False,
            "score_claim_valid": False,
            "promotion_eligible": False,
            "promotable": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "master_gradient_anchor_path": str(args.anchor_jsonl),
            "master_gradient_anchor_written": anchor_written,
            "master_gradient_anchor_blockers": anchor_blockers,
            "captured_at_utc": sidecar["captured_at_utc"],
        }
        _append_jsonl_locked(manifest_row, output_path=args.manifest_jsonl)

    print(
        json.dumps(
            {
                "out": str(args.out),
                "sidecar": str(sidecar_path),
                "shape": list(result.per_pair_per_byte.shape),
                "archive_sha256": result.archive_sha256[:24],
                "n_pairs_used": result.n_pairs_used,
                "operating_point": result.operating_point,
                "n_decoder_tensors": result.n_decoder_tensors,
                "evidence_grade": EVIDENCE_GRADE_MLX,
                "master_gradient_anchor": str(args.anchor_jsonl) if anchor_written else None,
                "master_gradient_anchor_blockers": anchor_blockers,
                "nonzero_byte_rows": int(np.count_nonzero(np.abs(result.per_pair_per_byte).sum(axis=(1, 2)))),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
