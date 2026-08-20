#!/usr/bin/env python3
"""Retain the PyAV-vs-DALI pair rankings and prove the live ps1u selector lineage.

This is a scorer-free, non-promotable lineage/selection measurement.  It reads
the already-retained n600 pose vectors, persists every derived mass and ranking
array, and proves that the live ``top_mass_pairs`` call consumes the
content-addressed DALI authority table.  Re-running with the same output is an
idempotent resume; stage receipts and payloads are atomically replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Final

import numpy as np

REPO: Final = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments import ddm_ps1u_uncapped_pose_solve as ps1u

OUTPUT: Final = Path("/Volumes/VertigoDataTier/pact/ddm_gt2/retained")
BASE_POSE: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_pz4_joint_target_conditioned_receiver/"
    "direct_v6/full_n600_eval/retained/pose_vectors/cp135_base_first6_n600.npy"
)
DALI_GT: Final = BASE_POSE.with_name("gt_first6_dali_n600.npy")
PYAV_GT: Final = BASE_POSE.with_name("gt_first6_n600.npy")  # GT_LINEAGE_OK: deliberate PYAV_YUV420_TO_RGB historical-control table sha256 82ed61ce6a11, compared against DALI but never used by the live selector
EXPECTED_BASE_SHA256: Final = "e64e8bd36c1a603da30c15fa581cdaeda409e8939cefe61c3d01d09ac0850386"
EXPECTED_DALI_SHA256: Final = "8d5cfa83df55b89493ba43b1e5386d792c836c32791666192499a089068e7eff"
EXPECTED_PYAV_SHA256: Final = "82ed61ce6a11a6612502527fbb6864a22fe6c6099312e637d971214ab660fb27"
AXIS: Final = "[macOS-CPU advisory, scorer-free retained GT-vector analysis]"


class GT2Error(RuntimeError):
    """An input pin, selector binding, or retained-payload invariant differed."""


def sha256_file(path: Path, *, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise GT2Error(f"required retained file is absent: {path}")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def assert_file(path: Path, expected_sha256: str) -> dict[str, Any]:
    record = file_record(path)
    if record["sha256"] != expected_sha256:
        raise GT2Error(
            f"content pin mismatch for {path}: expected {expected_sha256}, "
            f"measured {record['sha256']}"
        )
    return record


def atomic_save_npy(path: Path, value: np.ndarray) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    return file_record(path)


def atomic_write_json(path: Path, value: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    with tmp.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(tmp, path)
    return file_record(path)


def _mass(base_pose: np.ndarray, gt_pose: np.ndarray) -> np.ndarray:
    return np.mean(
        np.square(base_pose.astype(np.float64) - gt_pose.astype(np.float64)), axis=1
    )


def _stable_descending_order(values: np.ndarray) -> np.ndarray:
    return np.argsort(-values, kind="stable").astype(np.int32)


def _ordinal_ranks(order: np.ndarray) -> np.ndarray:
    ranks = np.empty(order.shape, dtype=np.int32)
    ranks[order] = np.arange(order.size, dtype=np.int32)
    return ranks


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    if args.resume_from.resolve() != output:
        raise GT2Error("--resume-from must resolve to --output")
    if not 1 <= args.top_k <= ps1u.PAIR_COUNT:
        raise GT2Error(f"--top-k must lie in [1, {ps1u.PAIR_COUNT}]")
    output.mkdir(parents=True, exist_ok=True)

    inputs = {
        "base_pose": assert_file(args.base_pose, EXPECTED_BASE_SHA256),
        "dali_gt": assert_file(args.dali_gt, EXPECTED_DALI_SHA256),
        "pyav_gt": assert_file(args.pyav_gt, EXPECTED_PYAV_SHA256),
    }
    atomic_write_json(
        output / "checkpoints/stage_10_inputs.json",
        {
            "schema": "ddm_gt2_selector_inputs.v1",
            "axis": AXIS,
            "inputs": inputs,
            "score_claim": False,
        },
    )

    base_pose = np.load(args.base_pose, allow_pickle=False)
    dali_gt = np.load(args.dali_gt, allow_pickle=False)
    pyav_gt = np.load(args.pyav_gt, allow_pickle=False)
    expected_shape = (ps1u.PAIR_COUNT, ps1u.POSE_DIMENSIONS)
    for name, value in (("base_pose", base_pose), ("dali_gt", dali_gt), ("pyav_gt", pyav_gt)):
        if value.shape != expected_shape:
            raise GT2Error(f"{name} shape {value.shape} != {expected_shape}")

    dali_mass = _mass(base_pose, dali_gt)
    pyav_mass = _mass(base_pose, pyav_gt)
    dali_order = _stable_descending_order(dali_mass)
    pyav_order = _stable_descending_order(pyav_mass)
    dali_ranks = _ordinal_ranks(dali_order)
    pyav_ranks = _ordinal_ranks(pyav_order)
    spearman = float(np.corrcoef(dali_ranks.astype(np.float64), pyav_ranks.astype(np.float64))[0, 1])

    k = args.top_k
    dali_top = dali_order[:k]
    pyav_top = pyav_order[:k]
    overlap = np.intersect1d(dali_top, pyav_top, assume_unique=True).astype(np.int32)
    outputs = {
        "dali_per_pair_mass": atomic_save_npy(output / "dali_per_pair_mass.f64.npy", dali_mass),
        "pyav_per_pair_mass": atomic_save_npy(output / "pyav_per_pair_mass.f64.npy", pyav_mass),
        "dali_rank_order": atomic_save_npy(output / "dali_rank_order.i32.npy", dali_order),
        "pyav_rank_order": atomic_save_npy(output / "pyav_rank_order.i32.npy", pyav_order),
        "dali_ordinal_ranks": atomic_save_npy(output / "dali_ordinal_ranks.i32.npy", dali_ranks),
        "pyav_ordinal_ranks": atomic_save_npy(output / "pyav_ordinal_ranks.i32.npy", pyav_ranks),
        "dali_top_k_ranked": atomic_save_npy(output / "dali_top_k_ranked.i32.npy", dali_top),
        "pyav_top_k_ranked": atomic_save_npy(output / "pyav_top_k_ranked.i32.npy", pyav_top),
        "top_k_overlap": atomic_save_npy(output / "top_k_overlap.i32.npy", overlap),
    }

    live_paths = ps1u._qs1_paths()
    live_base = Path(live_paths["base_pose"]).resolve()
    live_gt = Path(live_paths["gt_pose"]).resolve()
    if live_base != args.base_pose.resolve() or live_gt != args.dali_gt.resolve():
        raise GT2Error(
            "live ps1u selector paths differ from the measured inputs: "
            f"base={live_base}, gt={live_gt}"
        )
    live_selected = ps1u.top_mass_pairs(k, base_pose, dali_gt)
    expected_selected = np.sort(dali_top).astype(np.int32)
    if not np.array_equal(live_selected, expected_selected):
        raise GT2Error("live top_mass_pairs selection does not match the retained DALI ranking")

    result = {
        "schema": "ddm_gt2_dali_gt_selector_repoint.v1",
        "axis": AXIS,
        "selection_mode": "complete n600 stable descending residual-mass rank",
        "pair_denominator": ps1u.PAIR_COUNT,
        "pose_dimensions": ps1u.POSE_DIMENSIONS,
        "top_k": k,
        "inputs": inputs,
        "outputs": outputs,
        "dali_top_k_ranked": dali_top.astype(int).tolist(),
        "pyav_top_k_ranked": pyav_top.astype(int).tolist(),
        "overlap_pair_ids": overlap.astype(int).tolist(),
        "overlap_count": int(overlap.size),
        "overlap_fraction": float(overlap.size / k),
        "changed_membership_count": int(k - overlap.size),
        "spearman_ordinal_rank_correlation": spearman,
        "live_selector": {
            "producer": "experiments.ddm_ps1u_uncapped_pose_solve.top_mass_pairs",
            "consumer": "experiments.ddm_ps1u_uncapped_pose_solve.run --top-mass",
            "base_pose_path": str(live_base),
            "gt_pose_path": str(live_gt),
            "gt_pose_sha256": inputs["dali_gt"]["sha256"],
            "declared_lineage": "DALI_NVDEC",
            "selection_matches_retained_dali_rank": True,
        },
        "historical_pyav_result_re_ranked": False,
        "command_argv": [sys.executable, *sys.argv],
        "all_materialized_rankings_retained": True,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
    }
    atomic_write_json(output / "checkpoints/stage_90_complete.json", result)
    atomic_write_json(output / "DDM_GT2_SELECTOR_RESULT.json", result)
    return result


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--resume-from", type=Path, default=OUTPUT)
    parser.add_argument("--base-pose", type=Path, default=BASE_POSE)
    parser.add_argument("--dali-gt", type=Path, default=DALI_GT)
    parser.add_argument("--pyav-gt", type=Path, default=PYAV_GT)
    parser.add_argument("--top-k", type=int, default=30)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
