#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fail-closed exact tile-halo feasibility receipt on a real n600 witness anchor.

This is a structural/coverage probe, not a scorer run.  It validates the actual
frozen upstream SegNet module topology, consumes a file-backed n600 boundary
coverage measurement, derives the phase-aware local halo, and records whether
an exact sparse-forward benchmark is admissible.  Global squeeze/excitation or
a full-frame local halo REFUSES the benchmark before a tautological "full frame
called a tile" can mint a speedup.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO, REPO / "src", REPO / "upstream"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.local_acceleration.tile_halo_exactness import (  # noqa: E402
    cadence_from_temporal_iou,
    derive_exact_tile_halo_contract,
    derive_receptive_field_rows,
    inspect_torch_segnet_architecture,
)

DEFAULT_RUN_DIR = REPO / "experiments/results/levelset_v752_baseline_20260710T185913Z"
DEFAULT_COVERAGE = (
    REPO
    / "experiments/results/levelset_n600_witness_mod32cap_20260706T115554Z"
    / "lane_share_probe_ep225_n600.json"
)
DEFAULT_CHECKPOINT = "levelset_witness_ema_mlx.npz"


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
    ).strip()


def _load_n600_coverage(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    pairs = payload.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != 600:
        raise ValueError(f"{path} is not n600: pairs length={len(pairs or [])}")
    if sorted(int(value) for value in pairs) != list(range(600)):
        raise ValueError(f"{path} does not cover exact pair ids 0..599")
    regions = payload.get("d_seg_regions") or {}
    boundary = regions.get("bulk_boundary") or {}
    coverage = float(boundary["px_share"])
    if not (0.0 < coverage < 1.0):
        raise ValueError(f"invalid n600 boundary px_share={coverage}")
    return {
        "n_pairs": len(pairs),
        "pair_ids_sha256": hashlib.sha256(
            json.dumps(pairs, separators=(",", ":")).encode()
        ).hexdigest(),
        "boundary_area_fraction": coverage,
        "boundary_flip_mass_share": float(boundary["share_of_flips"]),
        "axis": payload.get("axis"),
        "gradient_surface": payload.get("gradient_surface"),
        "source_epoch": payload.get("epoch"),
        "source_checkpoint": payload.get("ckpt"),
        "source_seconds": payload.get("secs"),
    }


def _class_cadences() -> dict[str, Any]:
    # File-backed values are operator-routed settled anchors.  The derivation is
    # pre-registered here: largest K with IoU**K >= 0.90.
    ious = {
        "Road": 0.955,
        "Lane": 0.263,
        "Undrivable": 0.995,
        "Movable": 0.903,
        "MyCar_nonstatic_remainder": 0.994,
    }
    cadences = {name: cadence_from_temporal_iou(iou) for name, iou in ious.items()}
    pair_cadences: dict[str, int] = {}
    names = ["Road", "Lane", "Undrivable", "Movable", "MyCar_nonstatic_remainder"]
    for left_index, left in enumerate(names):
        for right in names[left_index:]:
            pair_cadences[f"{left}<->{right}"] = min(cadences[left], cadences[right])
    return {
        "status": "DERIVED_TRAINING_PATH_PROPOSAL_ONLY",
        "freshness_survival_bar": 0.90,
        "law": "max integer K>=1 such that temporal_iou**K >= 0.90",
        "measured_temporal_ious": ious,
        "class_refresh_cadence_steps": cadences,
        "class_pair_refresh_cadence_steps": pair_cadences,
        "MyCar_static_core": "NEVER_COMPUTE_PROPOSAL; measured IoU 0.994 and 25.6% area, overlap with blind coordinates unresolved",
    }


def build_receipt(run_dir: Path, checkpoint_name: str, coverage_path: Path) -> dict[str, Any]:
    run_dir = run_dir.resolve()
    checkpoint = run_dir / checkpoint_name
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if not coverage_path.is_file():
        raise FileNotFoundError(coverage_path)

    # Import after upstream was put on sys.path.  Instantiation is weights-free,
    # CPU metadata inspection only; no forward and no run-dir mutation.
    from modules import SegNet

    segnet = SegNet().eval()
    architecture = inspect_torch_segnet_architecture(segnet)
    rows = [row.to_dict() for row in derive_receptive_field_rows()]
    contract = derive_exact_tile_halo_contract()
    coverage = _load_n600_coverage(coverage_path)

    go_exact = contract.exact_dependency != "FULL_FRAME_GLOBAL"
    speedup = contract.ideal_exact_speedup_upper_bound
    go_speed = speedup >= 2.0
    return {
        "schema": "cheapen_real95_tile_halo_exactness.v1",
        "written_at_utc": _utc(),
        "lane_id": "lane_cheapen_real95_tilehalo_fp16_20260713",
        "axis": "[architecture proof + macOS-CPU advisory n600 coverage; NON-PROMOTABLE]",
        "evidence_grade": {
            "architecture_and_halo": "DERIVED_FROM_ACTUAL_MODULE_TOPOLOGY",
            "coverage": "MEASURED_N600_REAL_WITNESS_STATE",
            "speedup": "DERIVED_IDEAL_UPPER_BOUND; NOT_A_GPU_TIMING",
        },
        "provenance": {
            "git_sha": _git_sha(),
            "probe_source": "tools/probe_tile_halo_exactness_n600.py",
            "probe_source_sha256": _sha256(Path(__file__).resolve()),
            "contract_source": "src/tac/local_acceleration/tile_halo_exactness.py",
            "contract_source_sha256": _sha256(
                REPO / "src/tac/local_acceleration/tile_halo_exactness.py"
            ),
            "upstream_scorer_source": "upstream/modules.py",
            "upstream_scorer_source_sha256": _sha256(REPO / "upstream/modules.py"),
            "run_dir": str(run_dir.relative_to(REPO)),
            "run_dir_mutated": False,
            "checkpoint": str(checkpoint.relative_to(REPO)),
            "checkpoint_bytes": checkpoint.stat().st_size,
            "checkpoint_sha256": _sha256(checkpoint),
            "coverage_artifact": str(coverage_path.resolve().relative_to(REPO)),
            "coverage_artifact_bytes": coverage_path.stat().st_size,
            "coverage_artifact_sha256": _sha256(coverage_path),
        },
        "architecture": architecture,
        "phase_aware_receptive_field_rows": rows,
        "exact_tile_contract": contract.to_dict(),
        "n600_real_coverage": coverage,
        "sensitivity_waterfill_inputs": {
            "tier0_blind_coordinate_fraction": 0.227,
            "tier0_MyCar_static_core_fraction": 0.256,
            "tier0_union_fraction_bounds_due_to_unknown_overlap": [0.256, 0.483],
            "tier2_boundary_area_fraction_from_n600": coverage["boundary_area_fraction"],
            "tier2_after_exact_halo_source_area_fraction": 1.0,
            "class_flip_density_per_area_anchors": {
                "Road": 2.2,
                "Lane": 32.0,
                "Undrivable": 0.26,
                "MyCar_static_core": 0.0,
                "Movable": "NOT_NUMERIC_IN_DIRECTIVE; REFUSE_TO_GUESS",
            },
        },
        "tier1_refresh_policy": _class_cadences(),
        "exactness_check": {
            "exact_on_tiles_verified": False,
            "n600_logit_bitcompare": "STRUCTURALLY_REFUSED_BEFORE_EXECUTION",
            "reason": (
                "there is no smaller-than-frame exact tile operator to compare: 23 spatial "
                "SqueezeExcite means globally couple the frame, and the no-SE safe local halo "
                "is 685 px; calling the full frame a tile would be tautological"
            ),
            "lost_by_approximate_sparse_formulation": [
                "global SqueezeExcite mean and its VJP outside selected tiles",
                "interior CE/bulk SegNet cotangent",
                "class-interior margin and calibration forces",
                "current-step Tier-1 changes between stale refreshes",
            ],
        },
        "lever_a_gate": {
            "go_bar": "exact-on-tiles AND measured speedup >=2.0x",
            "exactness_pass": go_exact,
            "speed_pass": go_speed,
            "measured_speedup_at_coverage_x": None,
            "exact_speedup_upper_bound_at_measured_coverage_x": speedup,
            "verdict": "NO_GO",
            "verdict_scope": contract.verdict_scope,
            "reformulation_queue": list(contract.reformulation_queue),
        },
        "pointer_delta": "ZERO; means-only structural/coverage probe",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    parser.add_argument("--coverage-artifact", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if str(out).startswith("/tmp/") or str(out).startswith("/private/tmp/"):
        raise SystemExit("refusing /tmp durable evidence path")
    receipt = build_receipt(args.run_dir, args.checkpoint, args.coverage_artifact.resolve())
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
