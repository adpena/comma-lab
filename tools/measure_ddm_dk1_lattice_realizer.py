#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Measure dk1 lattice-native pose-null realization on phase-selected blocks.

This is a bounded advisory receipt tool.  It reads the et2/et1 phase-field
offsets from the SSD tier, selects real nonzero block16 locations, derives a
local shift target from the parent scorer-plane RGB field, projects that target
to the frame-1 yuv6 null space, and races:

  naive uniform scorer-round -> Dykstra round/project -> bounded CVP/Babai.

No n600 score is computed.  Real PoseNet leakage is measured against the parent
pair output on the selected blocks only, with score_claim=false.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO / "src", REPO / "experiments"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from ddm_sq1_eta_seg_realization import CAM_H, CAM_W, N_PAIRS_TOTAL, Scorer, seq_len  # noqa: E402
from tac.optimization.lattice_native_pose_null_realizer import (  # noqa: E402
    add_private_delta_to_frame,
    build_default_operator,
    extract_private_camera_block,
    pose_constraint_matrix,
    private_block_geometry,
    project_scorer_delta_to_pose_null,
    realize_lattice_native_block,
    results_to_receipt,
)

DEFAULT_RAW = Path(
    "/Volumes/VertigoDataTier/pact/ddm_et2_20260806/"
    "parent_tq1c_decode/submission/inflated/0.raw"
)
DEFAULT_OFFSETS = Path(
    "/Volumes/VertigoDataTier/pact/ddm_et2_20260806/"
    "phase_field/tq1c_block16_offsets.npy"
)
DEFAULT_OUT = REPO / ".omx/research/ddm_dk1_20260806/lattice_realizer_measurement.json"


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as fh:
        json.dump(payload, fh, indent=1, default=jsonable)
        fh.write("\n")
    tmp.replace(path)


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def scorer_plane_from_camera(operator, frame: np.ndarray) -> np.ndarray:
    return operator.apply(frame.astype(np.float64)).astype(np.float64)


def _all_phase_blocks(offsets: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Return all (pair, block_index, scorer_row, scorer_col) nonzero phase blocks."""

    if offsets.ndim != 3 or offsets.shape[2] != 2:
        raise RuntimeError(f"unexpected offsets shape {offsets.shape}")
    blocks_w = 512 // 16
    found: list[tuple[int, int, int, int]] = []
    for pair in range(offsets.shape[0]):
        nz = np.flatnonzero(np.any(offsets[pair] != 0, axis=1))
        for block in nz:
            br = int(block // blocks_w)
            bc = int(block % blocks_w)
            scorer_row = (br * 16 + 8) & ~1
            scorer_col = (bc * 16 + 8) & ~1
            if scorer_row <= 382 and scorer_col <= 510:
                found.append((int(pair), int(block), int(scorer_row), int(scorer_col)))
    return found


def select_phase_blocks(
    offsets: np.ndarray,
    *,
    n: int,
    mode: str,
    seed: int,
    strata: int,
) -> tuple[list[tuple[int, int, int, int]], dict[str, Any]]:
    """Select phase blocks with explicit denominator and selection provenance."""

    candidates = _all_phase_blocks(offsets)
    if n < 1:
        raise RuntimeError("n must be positive")
    if len(candidates) < n:
        raise RuntimeError(f"found only {len(candidates)} nonzero phase blocks")
    if mode == "first_nonzero":
        selected = candidates[:n]
        return selected, {
            "mode": "first_nonzero_phase_blocks",
            "candidate_population_blocks": len(candidates),
            "selection_scope": "phase-field nonzero block16 offsets with valid even 2x2 scorer centers",
            "note": "video-order first-n; instance-scope only, not population authority",
        }
    if mode != "stratified_nonzero":
        raise RuntimeError(f"unknown selection mode: {mode}")
    if strata < 1:
        raise RuntimeError("strata must be positive")

    n_pairs = int(offsets.shape[0])
    buckets: list[list[tuple[int, int, int, int]]] = [[] for _ in range(strata)]
    for item in candidates:
        pair = item[0]
        bucket = min(strata - 1, int(pair * strata // max(1, n_pairs)))
        buckets[bucket].append(item)

    rng = np.random.default_rng(int(seed))
    shuffled: list[list[tuple[int, int, int, int]]] = []
    for bucket_items in buckets:
        if not bucket_items:
            shuffled.append([])
            continue
        order = rng.permutation(len(bucket_items))
        shuffled.append([bucket_items[int(i)] for i in order])

    selected = []
    cursor = [0] * strata
    while len(selected) < n:
        progressed = False
        for idx, bucket_items in enumerate(shuffled):
            if cursor[idx] >= len(bucket_items):
                continue
            selected.append(bucket_items[cursor[idx]])
            cursor[idx] += 1
            progressed = True
            if len(selected) >= n:
                break
        if not progressed:
            break
    if len(selected) < n:
        raise RuntimeError(f"stratified selector produced only {len(selected)} blocks")

    return selected, {
        "mode": "stratified_nonzero_phase_blocks",
        "seed": int(seed),
        "strata": int(strata),
        "candidate_population_blocks": len(candidates),
        "candidate_blocks_per_stratum": [len(bucket) for bucket in buckets],
        "selection_scope": "phase-field nonzero block16 offsets with valid even 2x2 scorer centers",
        "selected_blocks_per_stratum": [
            sum(1 for item in selected if min(strata - 1, int(item[0] * strata // max(1, n_pairs))) == idx)
            for idx in range(strata)
        ],
    }


def shifted_target_delta(
    scorer_frame: np.ndarray,
    scorer_row: int,
    scorer_col: int,
    offset: np.ndarray,
    *,
    scale: float,
) -> np.ndarray:
    dy, dx = int(offset[0]), int(offset[1])
    src = scorer_frame[scorer_row : scorer_row + 2, scorer_col : scorer_col + 2]
    rr = int(np.clip(scorer_row + dy, 0, scorer_frame.shape[0] - 2))
    cc = int(np.clip(scorer_col + dx, 0, scorer_frame.shape[1] - 2))
    shifted = scorer_frame[rr : rr + 2, cc : cc + 2]
    return project_scorer_delta_to_pose_null((shifted - src) * float(scale))


def pose_delta(sc: Scorer, base_pose: Any, pair: np.ndarray) -> float:
    return sc.d_pose(base_pose, sc.pose_out(pair.astype(np.uint8)))


def execute(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    if args.n_blocks > 8 and not args.skip_real_posenet:
        raise RuntimeError(
            "dk1 charter allows only small-n frozen-scorer use; use --skip-real-posenet "
            "for scorer-free n_blocks > 8"
        )
    operator = build_default_operator()
    offsets = np.load(args.offsets, mmap_mode="r")
    selected, selection_provenance = select_phase_blocks(
        offsets,
        n=args.n_blocks,
        mode=args.selection_mode,
        seed=args.selection_seed,
        strata=args.selection_strata,
    )

    raw = np.memmap(
        args.raw,
        dtype=np.uint8,
        mode="r",
        shape=(N_PAIRS_TOTAL * seq_len, CAM_H, CAM_W, 3),
    )
    sc = None if args.skip_real_posenet else Scorer(args.threads)
    rows = []
    for pair, block, scorer_row, scorer_col in selected:
        frame0 = np.asarray(raw[seq_len * pair], dtype=np.uint8)
        frame1 = np.asarray(raw[seq_len * pair + 1], dtype=np.uint8)
        scorer_frame1 = scorer_plane_from_camera(operator, frame1)
        target = shifted_target_delta(
            scorer_frame1,
            scorer_row,
            scorer_col,
            offsets[pair, block],
            scale=args.target_scale,
        )
        geom = private_block_geometry(operator, scorer_row, scorer_col)
        base_block = extract_private_camera_block(frame1, geom)
        results = realize_lattice_native_block(
            target,
            geom,
            base_block=base_block,
            dykstra_iterations=args.dykstra_iterations,
            cvp_tap_radius=args.cvp_tap_radius,
        )
        base_pair = np.stack([frame0, frame1])
        base_pose = sc.pose_out(base_pair) if sc is not None else None
        method_rows = {}
        for name, result in results.items():
            edited_f1 = add_private_delta_to_frame(frame1, geom, result.camera_delta)
            edited_pair = np.stack([frame0, edited_f1])
            method_rows[name] = {
                **result.to_dict(),
                "real_posenet_dpose_vs_parent_pair": (
                    pose_delta(sc, base_pose, edited_pair)
                    if sc is not None and base_pose is not None
                    else None
                ),
                "realized_scorer_delta_l2": float(np.linalg.norm(result.scorer_delta)),
                "target_scorer_delta_l2": float(np.linalg.norm(target)),
                "local_delta_retention_l2": (
                    float(np.linalg.norm(result.scorer_delta) / np.linalg.norm(target))
                    if np.linalg.norm(target) > 0
                    else None
                ),
            }
        rows.append(
            {
                "pair": pair,
                "phase_block": block,
                "phase_offset_dy_dx": offsets[pair, block].astype(int).tolist(),
                "scorer_row": scorer_row,
                "scorer_col": scorer_col,
                "geometry": geom.to_dict(),
                "target_pose_leakage_sq": float(
                    np.dot(
                        pose_constraint_matrix() @ target.reshape(12),
                        pose_constraint_matrix() @ target.reshape(12),
                    )
                ),
                "target_scorer_delta_l2": float(np.linalg.norm(target)),
                "methods": method_rows,
                "best_local": results_to_receipt(results)["best_by_pose_then_seg"],
            }
        )

    aggregate = {}
    for name in ("naive", "dykstra", "cvp"):
        vals_pose = [row["methods"][name]["pose_leakage_sq"] for row in rows]
        vals_real = [
            row["methods"][name]["real_posenet_dpose_vs_parent_pair"]
            for row in rows
            if row["methods"][name]["real_posenet_dpose_vs_parent_pair"] is not None
        ]
        vals_seg = [row["methods"][name]["seg_discrepancy"] for row in rows]
        aggregate[name] = {
            "pose_leakage_sq_mean": float(np.mean(vals_pose)),
            "pose_leakage_sq_median": float(np.median(vals_pose)),
            "real_posenet_dpose_vs_parent_pair_mean": (
                float(np.mean(vals_real)) if vals_real else None
            ),
            "real_posenet_dpose_vs_parent_pair_median": (
                float(np.median(vals_real)) if vals_real else None
            ),
            "seg_discrepancy_mean": float(np.mean(vals_seg)),
        }

    return {
        "schema": "ddm_dk1_lattice_realizer_measurement.v1",
        "axis": (
            "[scorer-free local A(Dx)/D-private advisory]"
            if args.skip_real_posenet
            else "[macOS-CPU frozen-PoseNet advisory] small-n"
        ),
        "score_claim": False,
        "promotion_eligible": False,
        "n600_run": False,
        "frozen_posenet_used": not bool(args.skip_real_posenet),
        "n_blocks": len(rows),
        "inputs": {
            "raw": str(args.raw),
            "raw_sha256": sha256_file(args.raw),
            "offsets": str(args.offsets),
            "offsets_sha256": sha256_file(args.offsets),
        },
        "selection": {
            **selection_provenance,
            "blocks": [[p, b, r, c] for p, b, r, c in selected],
            "target_scale": args.target_scale,
        },
        "solver": {
            "dykstra_iterations": args.dykstra_iterations,
            "cvp_tap_radius": args.cvp_tap_radius,
            "exact_d_weights": True,
            "uniform_025_assumption": False,
        },
        "aggregate": aggregate,
        "rows": rows,
        "elapsed_s": time.time() - started,
        "boundaries": [
            "block-local advisory only",
            "target is local shift delta from phase-field offset, projected to ker(A)",
            (
                "real PoseNet leakage skipped by request; receipt is scorer-free"
                if args.skip_real_posenet
                else "real PoseNet leakage measured versus parent pair output, not GT authority"
            ),
            "no SegNet score, no archive build, no n600 scorer slot",
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--offsets", type=Path, default=DEFAULT_OFFSETS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--n-blocks", type=int, default=4)
    ap.add_argument(
        "--selection-mode",
        choices=("first_nonzero", "stratified_nonzero"),
        default="first_nonzero",
    )
    ap.add_argument("--selection-seed", type=int, default=20260806)
    ap.add_argument("--selection-strata", type=int, default=10)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--skip-real-posenet", action="store_true")
    ap.add_argument("--target-scale", type=float, default=0.25)
    ap.add_argument("--dykstra-iterations", type=int, default=8)
    ap.add_argument("--cvp-tap-radius", type=int, default=1)
    args = ap.parse_args()
    payload = execute(args)
    write_json_atomic(args.out, payload)
    print(json.dumps({"out": str(args.out), "aggregate": payload["aggregate"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
