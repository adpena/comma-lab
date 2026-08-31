#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Scorer-free n600 preprocessing parity on the retained AFR1 raw witness.

This probe executes the exact upstream ``SegNet.preprocess_input`` and
``PoseNet.preprocess_input`` methods without constructing either scorer or
loading scorer weights.  It compares their tensors against the two live local
instrument spellings: the MLX cache bridge and the explicit
``align_corners=False`` spelling used by local scorer helpers.  A deliberately
wrong ``align_corners=True`` control proves the comparison is sensitive.

The run is streaming and stage-resumable.  Every pair batch writes an atomic
checkpoint containing source/tensor hashes and complete mismatch statistics;
no full-n600 scorer-input tensor is materialized.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from tac.local_acceleration.mlx_preprocess import (
    SEGNET_INPUT_HW,
    _rgb_to_yuv6_torch,
    load_raw_video_memmap,
    non_overlapping_pair_indices,
    preprocess_scorer_inputs_from_pairs,
)

SCHEMA = "ddm_ux1_afr1_operator_parity.v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(contiguous.dtype.str.encode("ascii"))
    digest.update(json.dumps(list(contiguous.shape), separators=(",", ":")).encode())
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _comparison(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        raise ValueError(
            "comparison tensors disagree in shape/dtype: "
            f"{reference.shape}/{reference.dtype} vs {candidate.shape}/{candidate.dtype}"
        )
    difference = candidate - reference
    absolute = difference.abs()
    return {
        "elements": int(reference.numel()),
        "differing_elements": int(torch.count_nonzero(difference).item()),
        "sum_abs": float(absolute.sum(dtype=torch.float64).item()),
        "sum_sq": float((difference * difference).sum(dtype=torch.float64).item()),
        "max_abs": float(absolute.max().item()),
    }


def _combine(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    fields = [row["comparisons"][key] for row in rows]
    elements = sum(int(field["elements"]) for field in fields)
    differing = sum(int(field["differing_elements"]) for field in fields)
    return {
        "elements": elements,
        "differing_elements": differing,
        "differing_fraction": differing / elements if elements else None,
        "sum_abs": sum(float(field["sum_abs"]) for field in fields),
        "sum_sq": sum(float(field["sum_sq"]) for field in fields),
        "max_abs": max(float(field["max_abs"]) for field in fields),
    }


def _upstream_modules(upstream_dir: Path) -> Any:
    sys.path.insert(0, str(upstream_dir))
    try:
        return importlib.import_module("modules")
    finally:
        sys.path.pop(0)


def _batch_receipt(
    *,
    modules: Any,
    pairs: np.ndarray,
    pair_start: int,
    pair_end: int,
    raw_sha256: str,
) -> dict[str, Any]:
    pair_indices = non_overlapping_pair_indices(pair_end * 2)[pair_start:pair_end]
    pair_copy = np.array(pairs, dtype=np.uint8, copy=True, order="C")
    frame_tensor = (
        torch.from_numpy(pair_copy)
        .float()
        .permute(0, 1, 4, 2, 3)
        .contiguous()
    )

    exact_seg = modules.SegNet.preprocess_input(None, frame_tensor).contiguous()
    exact_pose = modules.PoseNet.preprocess_input(None, frame_tensor).contiguous()

    cache = preprocess_scorer_inputs_from_pairs(
        pair_copy,
        pair_indices=pair_indices,
        source="retained_afr1_raw",
    )
    cache_seg = torch.from_numpy(cache.segnet_last_rgb)
    cache_pose = torch.from_numpy(cache.posenet_yuv6_pair)

    flat = frame_tensor.reshape(-1, *frame_tensor.shape[2:])
    explicit_resized = F.interpolate(
        flat,
        size=SEGNET_INPUT_HW,
        mode="bilinear",
        align_corners=False,
    )
    explicit_pair = explicit_resized.reshape(
        frame_tensor.shape[0],
        frame_tensor.shape[1],
        3,
        *SEGNET_INPUT_HW,
    )
    explicit_seg = explicit_pair[:, -1].contiguous()
    explicit_pose = _rgb_to_yuv6_torch(explicit_resized).reshape(
        frame_tensor.shape[0],
        12,
        SEGNET_INPUT_HW[0] // 2,
        SEGNET_INPUT_HW[1] // 2,
    ).contiguous()

    wrong_resized = F.interpolate(
        flat,
        size=SEGNET_INPUT_HW,
        mode="bilinear",
        align_corners=True,
    )
    wrong_pair = wrong_resized.reshape(
        frame_tensor.shape[0],
        frame_tensor.shape[1],
        3,
        *SEGNET_INPUT_HW,
    )
    wrong_seg = wrong_pair[:, -1].contiguous()
    wrong_pose = _rgb_to_yuv6_torch(wrong_resized).reshape(
        frame_tensor.shape[0],
        12,
        SEGNET_INPUT_HW[0] // 2,
        SEGNET_INPUT_HW[1] // 2,
    ).contiguous()

    arrays = {
        "exact_seg": exact_seg.detach().cpu().numpy(),
        "exact_pose": exact_pose.detach().cpu().numpy(),
        "cache_seg": cache_seg.numpy(),
        "cache_pose": cache_pose.numpy(),
        "explicit_false_seg": explicit_seg.detach().cpu().numpy(),
        "explicit_false_pose": explicit_pose.detach().cpu().numpy(),
        "negative_align_true_seg": wrong_seg.detach().cpu().numpy(),
        "negative_align_true_pose": wrong_pose.detach().cpu().numpy(),
    }
    return {
        "schema": f"{SCHEMA}.batch",
        "pair_start": pair_start,
        "pair_end_exclusive": pair_end,
        "pair_count": pair_end - pair_start,
        "source_raw_sha256": raw_sha256,
        "source_pair_bytes_sha256": _array_sha256(pair_copy),
        "array_sha256": {name: _array_sha256(value) for name, value in arrays.items()},
        "comparisons": {
            "mlx_cache_seg_vs_exact": _comparison(exact_seg, cache_seg),
            "mlx_cache_pose_vs_exact": _comparison(exact_pose, cache_pose),
            "explicit_false_seg_vs_exact": _comparison(exact_seg, explicit_seg),
            "explicit_false_pose_vs_exact": _comparison(exact_pose, explicit_pose),
            "negative_align_true_seg_vs_exact": _comparison(exact_seg, wrong_seg),
            "negative_align_true_pose_vs_exact": _comparison(exact_pose, wrong_pose),
        },
        "score_claim": False,
        "payload_policy": (
            "streaming hash-and-compare; no full-n600 tensor payload materialized; "
            "retained source raw remains the byte authority"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-raw-sha256", required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--batch-pairs", type=int, default=2)
    parser.add_argument("--torch-threads", type=int, default=4)
    args = parser.parse_args()
    if args.batch_pairs < 1 or args.torch_threads < 1:
        raise SystemExit("--batch-pairs and --torch-threads must be positive")

    raw_path = args.raw.resolve()
    archive_path = args.archive.resolve()
    upstream_dir = args.upstream_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_sha256 = _sha256_file(raw_path)
    if raw_sha256 != args.expected_raw_sha256:
        raise SystemExit(
            f"raw SHA mismatch: expected={args.expected_raw_sha256} observed={raw_sha256}"
        )

    raw = load_raw_video_memmap(raw_path)
    pair_indices = non_overlapping_pair_indices(raw.shape[0])
    modules = _upstream_modules(upstream_dir)
    torch.set_num_threads(args.torch_threads)
    started = time.time()
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for pair_start in range(0, len(pair_indices), args.batch_pairs):
        pair_end = min(len(pair_indices), pair_start + args.batch_pairs)
        checkpoint = checkpoint_dir / f"pairs_{pair_start:04d}_{pair_end:04d}.json"
        if checkpoint.is_file():
            row = json.loads(checkpoint.read_text(encoding="utf-8"))
            if (
                row.get("schema") != f"{SCHEMA}.batch"
                or row.get("source_raw_sha256") != raw_sha256
                or int(row.get("pair_start", -1)) != pair_start
                or int(row.get("pair_end_exclusive", -1)) != pair_end
            ):
                raise SystemExit(f"incompatible checkpoint: {checkpoint}")
        else:
            frame_indices = pair_indices[pair_start:pair_end].reshape(-1)
            pairs = np.asarray(raw[frame_indices]).reshape(
                pair_end - pair_start,
                2,
                *raw.shape[1:],
            )
            row = _batch_receipt(
                modules=modules,
                pairs=pairs,
                pair_start=pair_start,
                pair_end=pair_end,
                raw_sha256=raw_sha256,
            )
            _atomic_json(checkpoint, row)
        rows.append(row)
        _atomic_json(
            output_dir / "checkpoint.json",
            {
                "schema": f"{SCHEMA}.checkpoint",
                "source_raw_sha256": raw_sha256,
                "pairs_complete": pair_end,
                "pairs_total": len(pair_indices),
                "last_stage": checkpoint.name,
                "resume_from": str(output_dir / "checkpoint.json"),
            },
        )

    comparison_keys = sorted(rows[0]["comparisons"])
    combined = {key: _combine(rows, key) for key in comparison_keys}
    live_keys = [
        "mlx_cache_seg_vs_exact",
        "mlx_cache_pose_vs_exact",
        "explicit_false_seg_vs_exact",
        "explicit_false_pose_vs_exact",
    ]
    live_bit_identical = all(
        int(combined[key]["differing_elements"]) == 0 for key in live_keys
    )
    negative_control_fired = all(
        int(combined[key]["differing_elements"]) > 0
        for key in (
            "negative_align_true_seg_vs_exact",
            "negative_align_true_pose_vs_exact",
        )
    )
    receipt = {
        "schema": SCHEMA,
        "axis": "[macOS-CPU scorer-free n600 operator parity]",
        "source_raw": {
            "path": str(raw_path),
            "bytes": raw_path.stat().st_size,
            "sha256": raw_sha256,
            "frames": int(raw.shape[0]),
            "pairs": len(pair_indices),
        },
        "archive": {
            "path": str(archive_path),
            "bytes": archive_path.stat().st_size,
            "sha256": _sha256_file(archive_path),
        },
        "upstream": {
            "path": str(upstream_dir),
            "evaluate_py_sha256": _sha256_file(upstream_dir / "evaluate.py"),
            "modules_py_sha256": _sha256_file(upstream_dir / "modules.py"),
            "frame_utils_py_sha256": _sha256_file(upstream_dir / "frame_utils.py"),
        },
        "batch_pairs": args.batch_pairs,
        "checkpoint_count": len(rows),
        "comparisons": combined,
        "live_instruments_bit_identical": live_bit_identical,
        "negative_control_fired": negative_control_fired,
        "bounded_dseg_mismatch_from_preprocessing": 0.0 if live_bit_identical else None,
        "closure_threshold_dseg": 0.01,
        "verdict": (
            "BOUNDED-CLOSED"
            if live_bit_identical and negative_control_fired
            else "OPEN-executed-probe"
        ),
        "elapsed_seconds_this_invocation": time.time() - started,
        "score_claim": False,
        "promotable": False,
        "scorer_loaded": False,
        "modal_dispatched": False,
        "payload_policy": (
            "all per-stage hashes/statistics retained; source raw retained byte-exact; "
            "no full-n600 derived tensor payload materialized"
        ),
    }
    _atomic_json(output_dir / "RESULT.json", receipt)
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
