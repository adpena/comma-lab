#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure the v7.5.2 fp32-EMA to parsed-int8 witness gap on real n600 states.

The control renders the preserved fp32 EMA.  The treatment is reconstructed
from an actual LVLS1 payload after the canonical byte-close parser.  Both arms
then use the same NumPy witness, real R, and frozen CPU-torch SegNet argmax.
No optimizer update, training launch, upstream evaluator, or run-dir mutation
occurs.  A per-chunk partial receipt makes the n600 local probe resumable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for path in (REPO, REPO / "src", REPO / "upstream", REPO / "experiments", REPO / "tools"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

EXPECTED_CHECKPOINT_SHA256 = "ef2c097f98f74dbd16e77c6f7b60f05e0a630b6bd65ee55bf334336c4549c965"
DEFAULT_RUN_DIR = REPO / "experiments/results/levelset_v752_baseline_20260710T185913Z"
DEFAULT_GT_CACHE = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
DEFAULT_CHECKPOINT = "levelset_witness_ema_mlx.npz"
AXIS = "[macOS-CPU/numpy-fp32 advisory; receiver-realized; NON-PROMOTABLE]"


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _load_lstars(cache_path: Path, n_pairs: int) -> tuple[np.ndarray, int]:
    cache = np.load(cache_path, allow_pickle=False)
    cached_pairs = int(cache["n_pairs"])
    if cached_pairs < n_pairs:
        raise ValueError(f"GT cache n{cached_pairs} is smaller than requested n{n_pairs}")
    # Only the authoritative SegNet argmax member is needed.  Do not materialize
    # 4.7 GB of frame/Pose members for this Seg-only measurement.
    return np.asarray(cache["lstars"][:n_pairs], dtype=np.int64), cached_pairs


def _byteclose_arms(
    checkpoint_dir: Path, checkpoint_name: str, packet_dir: Path
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any], dict[str, Any]]:
    import levelset_byte_close_and_eval as B

    from tac.boundary_math.lever_b_levelset_generator import int8_dequant_params

    fp32, cfg = B._load_levelset_ckpt(checkpoint_dir, checkpoint_name)
    so = B.detect_self_orient(
        cfg,
        {"freq_across": 32.0, "freq_along": 4.0, "tau": 4.0, "iters": 4},
    )
    blob, build_report = B.build_levelset_blob(fp32, cfg, so, None)
    archive_path, archive_bytes = B.assemble_packet(blob, packet_dir)
    manifest, parsed, parsed_code, lane_pairs, pose_carrier, _chart = B._dequant_blob(blob)  # (#497) 7th chart block: mechanical unpack update
    if lane_pairs is not None or pose_carrier is not None:
        raise ValueError("plain v7.5.2 gap probe unexpectedly decoded a lane/pose carrier")
    parsed["code"] = parsed_code

    direct = int8_dequant_params({key: value for key, value in fp32.items() if not key.startswith("pose_carrier.")})
    mismatches: list[str] = []
    for key, parsed_value in parsed.items():
        if key not in direct or not np.array_equal(parsed_value, direct[key]):
            mismatches.append(key)
    missing = sorted(set(direct) - set(parsed))
    if mismatches or missing:
        raise ValueError(f"parser/direct-int8 mismatch paths={mismatches}, missing={missing} (NO-FAKE)")
    packet = {
        "lvls1_blob_bytes": len(blob),
        "lvls1_blob_sha256": hashlib.sha256(blob).hexdigest(),
        "archive_path": str(archive_path.relative_to(REPO)),
        "archive_bytes": archive_bytes,
        "archive_sha256": _sha256(archive_path),
        "manifest": manifest,
        "build_report": build_report,
        "parse_back_equals_direct_int8_dequant": True,
    }
    return fp32, parsed, cfg, packet


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    import train_witness_realized_through_R_mlx as T
    import witness_per_stage_annulus_attribution as W

    from tac.boundary_math.seg_core import load_real_segnet

    source_path = Path(__file__).resolve()
    source_sha_at_start = _sha256(source_path)
    checkpoint = (args.run_dir / args.checkpoint).resolve()
    checkpoint_sha = _sha256(checkpoint)
    if checkpoint_sha != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("checkpoint SHA does not match the preregistered v7.5.2 EMA")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    free_bytes = int(shutil.disk_usage(args.out.parent).free)
    required_free_bytes = 100 * 1024 * 1024
    if free_bytes < required_free_bytes:
        raise RuntimeError(f"storage preflight failed: {free_bytes} B free < {required_free_bytes} B required")
    packet_dir = args.out.parent / "byteclose_packet"
    fp32, parsed_int8, _byteclose_cfg, packet = _byteclose_arms(args.run_dir.resolve(), args.checkpoint, packet_dir)
    witness_params, raw_cfg = W.load_ckpt(checkpoint)
    if set(witness_params) != set(fp32) or any(not np.array_equal(witness_params[key], fp32[key]) for key in fp32):
        raise ValueError("canonical witness loader and byte-close loader disagree")
    fp32 = witness_params
    scalars = W.cfg_scalars(raw_cfg, fp32)
    if scalars["self_orient"]:
        raise ValueError("the preregistered v7.5.2 checkpoint was expected to be self_orient=OFF")
    coords, curvelet = W.build_render_context(scalars)

    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = str(args.torch_threads)
    torch.set_num_threads(args.torch_threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    lstars, cache_pairs = _load_lstars(args.gt_cache, args.num_pairs)
    segnet = load_real_segnet("cpu")

    partial_path = args.out.with_name(args.out.stem + ".stage.json")
    rows: list[dict[str, Any]] = []
    if partial_path.is_file():
        prior = json.loads(partial_path.read_text())
        if (
            prior.get("checkpoint_sha256") != checkpoint_sha
            or prior.get("required_pairs") != args.num_pairs
            or prior.get("probe_source_sha256") != source_sha_at_start
        ):
            raise ValueError("partial receipt belongs to another source/checkpoint/pair contract")
        rows = list(prior.get("rows", []))
    complete = {int(row["pair_index"]) for row in rows}
    started = time.perf_counter()
    for start in range(0, args.num_pairs, args.chunk):
        indices = [index for index in range(start, min(start + args.chunk, args.num_pairs)) if index not in complete]
        if not indices:
            continue
        fp_frames: list[np.ndarray] = []
        int8_frames: list[np.ndarray] = []
        render_rows: list[tuple[int, bool, int, bool]] = []
        for pair_index in indices:
            fp_frame, fp_iters, fp_converged = W.render_frame1_argmax(
                fp32,
                scalars,
                coords,
                curvelet,
                fp32["code"][2 * pair_index + 1],
                args.so_iters,
            )
            int8_frame, int8_iters, int8_converged = W.render_frame1_argmax(
                parsed_int8,
                scalars,
                coords,
                curvelet,
                parsed_int8["code"][2 * pair_index + 1],
                args.so_iters,
            )
            fp_frames.append(fp_frame)
            int8_frames.append(int8_frame)
            render_rows.append((fp_iters, fp_converged, int8_iters, int8_converged))
        arm_frames = fp_frames + int8_frames
        arm_gt = [lstars[index] for index in indices] * 2
        dseg, argmax = T.cpu_verdict_d_seg_argmax_batch(segnet, arm_frames, arm_gt)
        count = len(indices)
        for local, pair_index in enumerate(indices):
            fp_am = argmax[local]
            int8_am = argmax[count + local]
            fp_iters, fp_converged, int8_iters, int8_converged = render_rows[local]
            rows.append(
                {
                    "pair_index": pair_index,
                    "d_seg_fp32_ema": float(dseg[local]),
                    "d_seg_parsed_int8": float(dseg[count + local]),
                    "d_seg_gap_int8_minus_fp32": float(dseg[count + local] - dseg[local]),
                    "segnet_argmax_flips_int8_vs_fp32": int(np.count_nonzero(fp_am != int8_am)),
                    "segnet_argmax_pixels": int(fp_am.size),
                    "fp32_receiver_iters": fp_iters,
                    "fp32_receiver_converged": fp_converged,
                    "int8_receiver_iters": int8_iters,
                    "int8_receiver_converged": int8_converged,
                }
            )
        rows.sort(key=lambda row: int(row["pair_index"]))
        complete.update(indices)
        _atomic_json(
            partial_path,
            {
                "schema": "int8_witness_byteclose_gap_partial.v1",
                "checkpoint_sha256": checkpoint_sha,
                "probe_source_sha256": source_sha_at_start,
                "required_pairs": args.num_pairs,
                "completed_pairs": len(rows),
                "last_completed_at_utc": _utc(),
                "rows": rows,
            },
        )
        print(
            json.dumps(
                {
                    "completed_pairs": len(rows),
                    "required_pairs": args.num_pairs,
                    "latest_pair": indices[-1],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if len(rows) != args.num_pairs:
        raise RuntimeError(f"partial completion {len(rows)} != n{args.num_pairs}")
    if _sha256(source_path) != source_sha_at_start:
        raise RuntimeError("probe source changed during execution; terminal evidence refused")
    fp_values = np.asarray([row["d_seg_fp32_ema"] for row in rows], dtype=np.float64)
    int8_values = np.asarray([row["d_seg_parsed_int8"] for row in rows], dtype=np.float64)
    gap = float(np.mean(int8_values) - np.mean(fp_values))
    flip_count = sum(int(row["segnet_argmax_flips_int8_vs_fp32"]) for row in rows)
    flip_pixels = sum(int(row["segnet_argmax_pixels"]) for row in rows)
    receipt = {
        "schema": "int8_witness_byteclose_gap_n600.v1",
        "completed_at_utc": _utc(),
        "status": "MEASURED",
        "lane_id": "lane_int8_training_rungs_20260713",
        "axis": AXIS,
        "research_only": True,
        "training_launched": False,
        "verdict_scope": (
            "post-hoc per-tensor symmetric int8 byte-close gap for the exact v7.5.2 EMA on the "
            "first 600 real cached pairs, canonical LVLS1 parse-back, NumPy receiver, real R, and "
            "frozen macOS CPU-torch SegNet only; no d_pose, archive score, contest-CPU/CUDA, QAT "
            "outcome, promotion, or transfer to another checkpoint"
        ),
        "labels": {
            "d_seg_rows_and_means": "MEASURED",
            "gap_and_score_unit_prize_ceiling": "DERIVED_FROM_MEASURED",
            "QAT_recovery": "UNMEASURED_TICKET",
        },
        "authority": {
            "score_claim": False,
            "pointer_moved": False,
            "promotion_eligible": False,
        },
        "provenance": {
            "git_sha": _git_sha(),
            "probe_source": str(source_path.relative_to(REPO)),
            "probe_source_sha256": source_sha_at_start,
            "checkpoint": str(checkpoint.relative_to(REPO)),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": checkpoint_sha,
            "checkpoint_mutated": False,
            "gt_cache": str(args.gt_cache.resolve().relative_to(REPO)),
            "gt_cache_bytes": args.gt_cache.stat().st_size,
            "gt_cache_cached_pairs": cache_pairs,
            "gt_member_consumed": "lstars[0:600] only",
            "torch_threads": args.torch_threads,
            "torch_interop_threads": 1,
            "storage_preflight": {
                "path": str(args.out.parent.relative_to(REPO)),
                "free_bytes": free_bytes,
                "required_free_bytes": required_free_bytes,
                "ok": True,
                "large_artifact_created": False,
            },
            "partial_stage_checkpoint": str(partial_path.relative_to(REPO)),
            "packet": packet,
        },
        "measurement": {
            "n_pairs": args.num_pairs,
            "n600_evidence": args.num_pairs == 600,
            "d_seg_fp32_ema": float(np.mean(fp_values)),
            "d_seg_parsed_int8": float(np.mean(int8_values)),
            "d_seg_gap_int8_minus_fp32": gap,
            "seg_score_unit_gap_100x": 100.0 * gap,
            "segnet_argmax_flips_int8_vs_fp32": flip_count,
            "segnet_argmax_pixels": flip_pixels,
            "segnet_argmax_flip_fraction": float(flip_count / flip_pixels),
            "elapsed_seconds_this_invocation": time.perf_counter() - started,
            "rows": rows,
        },
        "qat_ticket": {
            "ticketed": True,
            "default_off": True,
            "target_grid": "the same LVLS1 per-tensor symmetric absmax/127 grid",
            "finishing_stage_only": True,
            "ab_control": "current fp32 training then post-hoc LVLS1 int8",
            "ab_treatment": "same training state plus finishing-stage fake-quant STE at LVLS1 grid",
            "admission": "receiver-closed n600 d_seg and exact archive bytes; no proxy-only admission",
        },
        "pointer_delta": "ZERO",
    }
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--gt-cache", type=Path, default=DEFAULT_GT_CACHE)
    parser.add_argument("--num-pairs", type=int, default=600)
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--so-iters", type=int, default=4)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out = args.out.resolve()
    if not (1 <= args.num_pairs <= 600):
        raise SystemExit("--num-pairs must be in 1..600")
    if args.chunk < 1 or args.torch_threads < 1:
        raise SystemExit("--chunk and --torch-threads must be positive")
    if str(args.out).startswith(("/tmp/", "/private/tmp/")):
        raise SystemExit("refusing a temporary durable evidence path")
    payload = run(args)
    _atomic_json(args.out, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
