#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure the live V10 M2 unrounded-target realization arm.

The old M2 receipt held an already-rounded uint8 scorer plane fixed.  This arm
instead preserves the *exact integer numerators* of the source camera frames'
factor-2 resize.  The source block is therefore a constructive witness for the
bounded four-variable Diophantine problem in every scorer cell.  Only certified
integer-null camera pixels are changed, and the fill is selected by measured
Brotli section size per resumable chunk.

The output is a real counted ``archive.zip`` containing the selected camera
preimage sections.  The generated scorer-free inflate program reconstructs
``inflated/0.raw`` solely from those bytes.  Running upstream/evaluate.py on
that output is the hard CPU-Torch oracle.  This deliberately charges the
source-dependent target description; it does not assume that an unrounded
target can be recovered from the rounded-Y spine at zero payload bytes.

Authority: [macOS-CPU advisory], score_claim=false, pointer unmoved.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import itertools
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import brotli
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tac.optimization.evaluator_invisibility_basis import (  # noqa: E402
    CAMERA_H,
    CAMERA_W,
    SCORER_INPUT_H,
    SCORER_INPUT_W,
    derive_tier1_resize_null_space,
)
from tac.optimization.uint8_lattice_feasibility import (  # noqa: E402
    DisjointResizeOperator,
)

SCHEMA = "m2_live_target_selection_receipt.v1"
LANE_ID = "lane_v10_m2_live_target_selection_20260720"
BASELINE_ARCHIVE_BYTES = 409_526_925
BASELINE_D_SEG = 0.00015196
BASELINE_D_POSE = 0.00010184
SOURCE_NORMALIZER_BYTES = 37_545_489
MIN_FREE_BYTES = 16 * 1024**3
STRATEGIES = ("constant", "horizontal_predictor", "vertical_predictor", "neighbor_mean")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


def _source_indices(visible: np.ndarray) -> np.ndarray:
    """Map each axis coordinate to itself or the preceding visible coordinate.

    Leading invisible coordinates use the first visible coordinate, matching
    ``resize_null_preimage._apply_horizontal_predictor_fill``.
    """
    visible = np.asarray(visible, dtype=bool)
    hits = np.flatnonzero(visible)
    if not len(hits):
        return np.zeros(len(visible), dtype=np.int64)
    out = np.empty(len(visible), dtype=np.int64)
    last = int(hits[0])
    for i in range(len(visible)):
        if visible[i]:
            last = i
        out[i] = last
    return out


def _fill_candidate(
    frames: np.ndarray,
    mask: np.ndarray,
    strategy: str,
    *,
    row_source: np.ndarray | None = None,
    col_source: np.ndarray | None = None,
) -> np.ndarray:
    """Vectorized implementation of the canonical #49 tier-1 fill policies."""
    x = np.asarray(frames, dtype=np.uint8)
    if x.ndim != 4 or x.shape[1:3] != mask.shape or x.shape[-1] != 3:
        raise ValueError("frames/mask geometry mismatch")
    out = x.copy()
    if strategy == "constant":
        out[:, mask, :] = 0
        return out
    if strategy == "neighbor_mean":
        keep = ~mask
        vals = x[:, keep, :].astype(np.float64)
        means = np.rint(vals.mean(axis=1)).clip(0, 255).astype(np.uint8)
        for i in range(len(out)):
            out[i, mask, :] = means[i]
        return out

    visible_rows = ~np.all(mask, axis=1)
    visible_cols = ~np.all(mask, axis=0)
    if strategy == "horizontal_predictor":
        if col_source is None:
            col_source = _source_indices(visible_cols)
        out[:, ~visible_rows, :, :] = 0
        rows = np.flatnonzero(visible_rows)
        out[:, rows, :, :] = x[:, rows[:, None], col_source[None, :], :]
        return out
    if strategy == "vertical_predictor":
        if row_source is None:
            row_source = _source_indices(visible_rows)
        out[:, :, ~visible_cols, :] = 0
        cols = np.flatnonzero(visible_cols)
        out[:, :, cols, :] = x[:, row_source[:, None], cols[None, :], :]
        return out
    raise ValueError(f"unknown strategy: {strategy}")


def _interleave(f0: np.ndarray, f1: np.ndarray) -> np.ndarray:
    if f0.shape != f1.shape:
        raise ValueError("frame planes differ")
    out = np.empty((2 * len(f0), *f0.shape[1:]), dtype=np.uint8)
    out[0::2] = f0
    out[1::2] = f1
    return out


def _compress(raw: np.ndarray, quality: int) -> bytes:
    return brotli.compress(np.ascontiguousarray(raw).tobytes(), quality=quality)


def _exact_numerator_proof(
    op: DisjointResizeOperator, original: np.ndarray, selected: np.ndarray
) -> tuple[int, int]:
    equal = total = 0
    for before, after in zip(original, selected, strict=True):
        nb, db = op.apply_numerators(before)
        na, da = op.apply_numerators(after)
        if db != da:
            raise AssertionError("resize denominators changed")
        equal += int(np.count_nonzero(nb == na))
        total += int(nb.size)
    if equal != total:
        raise AssertionError(f"integer-null fill changed {total - equal} resize numerators")
    return equal, total


def _build_chunk(
    *,
    chunk_id: int,
    pair_start: int,
    pair_stop: int,
    gt_f0: np.ndarray,
    gt_f1: np.ndarray,
    mask: np.ndarray,
    op: DisjointResizeOperator,
    quality: int,
    chunks_dir: Path,
) -> dict:
    frames = _interleave(np.asarray(gt_f0[pair_start:pair_stop]), np.asarray(gt_f1[pair_start:pair_stop]))
    before = _compress(frames, quality)
    visible_rows = ~np.all(mask, axis=1)
    visible_cols = ~np.all(mask, axis=0)
    row_source = _source_indices(visible_rows)
    col_source = _source_indices(visible_cols)

    winner_name = ""
    winner_raw: np.ndarray | None = None
    winner_bytes: bytes | None = None
    candidate_sizes: dict[str, int] = {}
    for strategy in STRATEGIES:
        candidate = _fill_candidate(
            frames,
            mask,
            strategy,
            row_source=row_source,
            col_source=col_source,
        )
        encoded = _compress(candidate, quality)
        candidate_sizes[strategy] = len(encoded)
        if winner_bytes is None or len(encoded) < len(winner_bytes):
            winner_name, winner_raw, winner_bytes = strategy, candidate, encoded
    assert winner_raw is not None and winner_bytes is not None
    equal, total = _exact_numerator_proof(op, frames, winner_raw)

    section = chunks_dir / f"chunk-{chunk_id:04d}.selected.br"
    tmp = section.with_suffix(section.suffix + ".tmp")
    tmp.write_bytes(winner_bytes)
    os.replace(tmp, section)
    row = {
        "schema": "m2_live_target_selection_chunk.v1",
        "chunk_id": chunk_id,
        "pair_start": pair_start,
        "pair_stop": pair_stop,
        "frame_count": 2 * (pair_stop - pair_start),
        "strategy": winner_name,
        "candidate_section_bytes": candidate_sizes,
        "section_bytes_before": len(before),
        "section_bytes_after": len(winner_bytes),
        "section_bytes_delta": len(winner_bytes) - len(before),
        "selected_section": section.name,
        "selected_section_sha256": hashlib.sha256(winner_bytes).hexdigest(),
        "changed_values": int(np.count_nonzero(frames != winner_raw)),
        "numerator_equal_values": equal,
        "numerator_total_values": total,
        "numerator_exact": equal == total,
        "brotli_quality": quality,
    }
    _atomic_json(chunks_dir / f"chunk-{chunk_id:04d}.manifest.json", row)
    return row


def _inflate_py() -> str:
    return '''#!/usr/bin/env python3
import brotli, json, os, sys, zipfile
from pathlib import Path
archive_dir, output_dir, names_file = map(Path, sys.argv[1:4])
names = [x.strip() for x in names_file.read_text().splitlines() if x.strip()]
if names != ["0"]:
    raise SystemExit(f"expected one video named 0, got {names}")
output_dir.mkdir(parents=True, exist_ok=True)
target = output_dir / "0.raw"
tmp = target.with_suffix(".raw.tmp")
with zipfile.ZipFile(archive_dir / "archive.zip") as zf, tmp.open("wb") as out:
    manifest = json.loads(zf.read("manifest.json"))
    for row in manifest["chunks"]:
        blob = zf.read("chunks/" + row["selected_section"])
        out.write(brotli.decompress(blob))
os.replace(tmp, target)
'''


def _write_runtime(out_dir: Path) -> None:
    inflate_py = out_dir / "inflate.py"
    inflate_sh = out_dir / "inflate.sh"
    inflate_py.write_text(_inflate_py())
    inflate_sh.write_text(
        '#!/bin/sh\nexec "${M2_PYTHON:-python3}" "$(dirname "$0")/inflate.py" "$@"\n'
    )
    inflate_py.chmod(0o755)
    inflate_sh.chmod(0o755)


def _build_archive(out_dir: Path, chunks: list[dict], metadata: dict) -> Path:
    archive = out_dir / "archive.zip"
    tmp = archive.with_suffix(".zip.tmp")
    packet_manifest = dict(metadata)
    packet_manifest["chunks"] = chunks
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        zf.writestr("manifest.json", json.dumps(packet_manifest, sort_keys=True, separators=(",", ":")))
        for row in chunks:
            path = out_dir / "chunks" / row["selected_section"]
            zf.write(path, "chunks/" + path.name)
    os.replace(tmp, archive)
    return archive


def _parse_report(path: Path) -> tuple[float, float]:
    d_seg = d_pose = None
    for line in path.read_text().splitlines():
        if "Average PoseNet Distortion:" in line:
            d_pose = float(line.rsplit(":", 1)[1])
        elif "Average SegNet Distortion:" in line:
            d_seg = float(line.rsplit(":", 1)[1])
    if d_seg is None or d_pose is None:
        raise RuntimeError(f"could not parse oracle report: {path}")
    return d_seg, d_pose


def _structural_decomposition(z: np.lib.npyio.NpzFile, chunks: list[dict]) -> dict:
    labels = np.asarray(z["lstars"])
    margins = np.asarray(z["margins"], dtype=np.float64)
    n_classes = int(labels.max()) + 1
    class_counts = np.bincount(labels.reshape(-1), minlength=n_classes)
    edges = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, float("inf"))
    margin_counts: dict[str, int] = {}
    abs_margin = np.abs(margins)
    for lo, hi in itertools.pairwise(edges):
        label = f"[{lo:.0e},{hi:.0e})" if np.isfinite(hi) else f"[{lo:.0e},inf)"
        margin_counts[label] = int(np.count_nonzero((abs_margin >= lo) & (abs_margin < hi)))

    by_strategy: dict[str, dict[str, int]] = {}
    for strategy, rows in collections.defaultdict(list, {
        name: [r for r in chunks if r["strategy"] == name] for name in STRATEGIES
    }).items():
        if not rows:
            continue
        by_strategy[strategy] = {
            "chunks_won": len(rows),
            "pairs": sum(int(r["pair_stop"]) - int(r["pair_start"]) for r in rows),
            "section_bytes_before": sum(int(r["section_bytes_before"]) for r in rows),
            "section_bytes_after": sum(int(r["section_bytes_after"]) for r in rows),
            "section_bytes_freed": sum(
                int(r["section_bytes_before"]) - int(r["section_bytes_after"]) for r in rows
            ),
        }
    ranked_sections = sorted(
        (
            {
                "chunk_id": int(r["chunk_id"]),
                "strategy": r["strategy"],
                "section_bytes_after": int(r["section_bytes_after"]),
                "section_bytes_freed": int(r["section_bytes_before"])
                - int(r["section_bytes_after"]),
            }
            for r in chunks
        ),
        key=lambda row: row["section_bytes_after"],
        reverse=True,
    )
    return {
        "bytes_by_winning_fill": by_strategy,
        "largest_selected_sections": ranked_sections[:8],
        "seg_reference_pixels_by_class": {
            str(i): int(count) for i, count in enumerate(class_counts)
        },
        "seg_reference_pixels_by_abs_margin_stratum": margin_counts,
        "integer_null_changed_camera_values": sum(int(r["changed_values"]) for r in chunks),
        "resize_numerator_values": sum(int(r["numerator_total_values"]) for r in chunks),
        "resize_numerator_mismatches": sum(
            int(r["numerator_total_values"]) - int(r["numerator_equal_values"]) for r in chunks
        ),
    }


def _oracle_decomposition(structural: dict, d_seg: float | None, d_pose: float | None) -> dict:
    exact_zero = d_seg == 0.0 and d_pose == 0.0
    if not exact_zero:
        return {
            "status": "BLOCKED_EXACT_ZERO_REQUIRED_FOR_NONNEGATIVE_DECOMPOSITION",
            "reason": "official aggregates are nonzero; a transition-ledger scorer is required",
        }
    return {
        "status": "EXACT_ZERO_FROM_NONNEGATIVE_OFFICIAL_TERMS",
        "encode_vs_realize": {"d_A_exact_numerator_mismatch": 0.0, "d_B_hard_oracle": 0.0},
        "seg_flips_by_source_class": dict.fromkeys(
            structural["seg_reference_pixels_by_class"], 0
        ),
        "seg_flips_by_abs_margin_stratum": dict.fromkeys(
            structural["seg_reference_pixels_by_abs_margin_stratum"], 0
        ),
        "pose_mse_by_output_dimension": {str(i): 0.0 for i in range(6)},
        "derivation": (
            "official d_seg and d_pose are sums/means of nonnegative per-cell and per-dimension "
            "terms; exact aggregate zero implies every decomposed term is zero"
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-cache", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--chunk-pairs", type=int, default=12)
    ap.add_argument("--brotli-quality", type=int, default=5)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--evaluate-py", type=Path)
    ap.add_argument("--uncompressed-dir", type=Path)
    ap.add_argument("--video-names-file", type=Path)
    ap.add_argument("--python", type=Path, default=Path(sys.executable))
    args = ap.parse_args()

    if not args.gt_cache.is_file():
        raise SystemExit(f"missing GT cache: {args.gt_cache}")
    if args.chunk_pairs < 1 or not 0 <= args.brotli_quality <= 11:
        raise SystemExit("invalid chunk-pairs or Brotli quality")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(args.out_dir)
    if usage.free < MIN_FREE_BYTES:
        raise SystemExit(f"storage preflight refused: only {usage.free} bytes free")
    chunks_dir = args.out_dir / "chunks"
    chunks_dir.mkdir(exist_ok=True)

    started = time.time()
    z = np.load(args.gt_cache, mmap_mode="r")
    gt_f0, gt_f1 = z["gt_f0"], z["gt_f1"]
    n_pairs = int(z["n_pairs"])
    if n_pairs != 600 or gt_f0.shape != (600, CAMERA_H, CAMERA_W, 3) or gt_f1.shape != gt_f0.shape:
        raise SystemExit(f"n600 custody mismatch: n={n_pairs}, f0={gt_f0.shape}, f1={gt_f1.shape}")
    basis = derive_tier1_resize_null_space()
    mask = basis.zero_weight_pixel_mask()
    op = DisjointResizeOperator.build(
        camera_h=CAMERA_H,
        camera_w=CAMERA_W,
        scorer_h=SCORER_INPUT_H,
        scorer_w=SCORER_INPUT_W,
    )

    chunks: list[dict] = []
    for chunk_id, start in enumerate(range(0, n_pairs, args.chunk_pairs)):
        stop = min(n_pairs, start + args.chunk_pairs)
        manifest = chunks_dir / f"chunk-{chunk_id:04d}.manifest.json"
        section = chunks_dir / f"chunk-{chunk_id:04d}.selected.br"
        if args.resume and manifest.is_file() and section.is_file():
            row = json.loads(manifest.read_text())
            if row.get("selected_section_sha256") != _sha256(section):
                raise SystemExit(f"resume hash mismatch: {section}")
        else:
            row = _build_chunk(
                chunk_id=chunk_id,
                pair_start=start,
                pair_stop=stop,
                gt_f0=gt_f0,
                gt_f1=gt_f1,
                mask=mask,
                op=op,
                quality=args.brotli_quality,
                chunks_dir=chunks_dir,
            )
        chunks.append(row)
        print(
            f"chunk {chunk_id:04d} pairs {start}:{stop} {row['strategy']} "
            f"{row['section_bytes_before']}->{row['section_bytes_after']}",
            flush=True,
        )

    before = sum(int(x["section_bytes_before"]) for x in chunks)
    after = sum(int(x["section_bytes_after"]) for x in chunks)
    structural = _structural_decomposition(z, chunks)
    metadata = {
        "schema": SCHEMA,
        "lane_id": LANE_ID,
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotion_eligible": False,
        "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "target": "unrounded exact source scorer reference (integer resize numerators)",
        "pair_count": n_pairs,
        "frame_count": 2 * n_pairs,
        "gt_cache": str(args.gt_cache),
        "gt_cache_bytes": args.gt_cache.stat().st_size,
        "gt_cache_sha256": _sha256(args.gt_cache),
        "integer_null_mask_pixels": int(np.count_nonzero(mask)),
        "integer_null_mask_fraction": float(np.mean(mask)),
        "full_linear_nullity_fraction": 1.0 - (SCORER_INPUT_H * SCORER_INPUT_W) / (CAMERA_H * CAMERA_W),
        "section_bytes_before": before,
        "section_bytes_after": after,
        "section_bytes_delta": after - before,
        "brotli_quality": args.brotli_quality,
        "structural_decomposition": structural,
    }
    archive = _build_archive(args.out_dir, chunks, metadata)
    _write_runtime(args.out_dir)
    names = args.out_dir / "video_names.txt"
    names.write_text("0\n")
    inflated = args.out_dir / "inflated"
    subprocess.run(
        [str(args.out_dir / "inflate.sh"), str(args.out_dir), str(inflated), str(names)],
        check=True,
        env={
            **os.environ,
            "PYTHONDONTWRITEBYTECODE": "1",
            "M2_PYTHON": str(args.python),
        },
    )
    raw = inflated / "0.raw"
    expected_raw = 2 * n_pairs * CAMERA_H * CAMERA_W * 3
    if raw.stat().st_size != expected_raw:
        raise SystemExit(f"inflate raw bytes {raw.stat().st_size} != {expected_raw}")

    report = args.out_dir / "official_cpu_report.txt"
    if args.evaluate_py:
        if not args.uncompressed_dir or not args.video_names_file:
            raise SystemExit("evaluation requires uncompressed-dir and video-names-file")
        subprocess.run(
            [
                str(args.python),
                str(args.evaluate_py),
                "--submission-dir",
                str(args.out_dir),
                "--uncompressed-dir",
                str(args.uncompressed_dir),
                "--video-names-file",
                str(args.video_names_file),
                "--device",
                "cpu",
                "--seed",
                "1234",
                "--report",
                str(report),
            ],
            check=True,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        d_seg, d_pose = _parse_report(report)
    else:
        d_seg = d_pose = None

    archive_bytes = archive.stat().st_size
    receipt = dict(metadata)
    oracle_decomposition = _oracle_decomposition(structural, d_seg, d_pose)
    baseline_nonrate = 100.0 * BASELINE_D_SEG + float(np.sqrt(10.0 * BASELINE_D_POSE))
    candidate_nonrate = (
        None if d_seg is None or d_pose is None else 100.0 * d_seg + float(np.sqrt(10.0 * d_pose))
    )
    receipt.update(
        {
            "archive_path": str(archive),
            "archive_bytes": archive_bytes,
            "archive_sha256": _sha256(archive),
            "inflated_raw_path": str(raw),
            "inflated_raw_bytes": raw.stat().st_size,
            "inflated_raw_sha256": _sha256(raw),
            "runtime_inflate_py_sha256": _sha256(args.out_dir / "inflate.py"),
            "runtime_inflate_sh_sha256": _sha256(args.out_dir / "inflate.sh"),
            "official_cpu_report": str(report) if report.is_file() else None,
            "d_seg": d_seg,
            "d_pose": d_pose,
            "delta_d_seg_vs_capstone": None if d_seg is None else d_seg - BASELINE_D_SEG,
            "delta_d_pose_vs_capstone": None if d_pose is None else d_pose - BASELINE_D_POSE,
            "delta_bytes_vs_capstone": archive_bytes - BASELINE_ARCHIVE_BYTES,
            "rate_term": 25.0 * archive_bytes / SOURCE_NORMALIZER_BYTES,
            "baseline_nonrate_score": baseline_nonrate,
            "candidate_nonrate_score": candidate_nonrate,
            "nonrate_score_recovery": (
                None if candidate_nonrate is None else baseline_nonrate - candidate_nonrate
            ),
            "oracle_decomposition": oracle_decomposition,
            "elapsed_seconds": time.time() - started,
            "verdict_scope": "n600 direct exact-numerator target realization; no claim that the rounded-Y spine can derive source fractions without charged payload",
        }
    )
    _atomic_json(args.out_dir / "receipt.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
