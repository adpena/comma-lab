#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Fail-closed composition of the cheapen-real-95 measurement receipts.

The two proposed levers overlap on scorer forward.  This probe therefore never
multiplies isolated factors.  It emits a numeric composed speedup only when the
wall split, each individual gate, and one joint A+B before/after receipt are all
MEASURED.  Otherwise it records the missing custody as a durable blocker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from tac.canonical_equations.amdahl_measured_wall_split_20260713 import (  # noqa: E402
    EQUATION_ID,
    AmdahlWallSplit,
    MeasuredLever,
    MeasuredSeconds,
)

RESULT_DIR = REPO / "experiments/results/cheapen_real95_tilehalo_fp16_20260713"
DEFAULT_WALL = RESULT_DIR / "current_wall_receipt.json"
DEFAULT_TILE = RESULT_DIR / "tile_halo_receipt.json"
DEFAULT_PRECISION = RESULT_DIR / "mlx_precision_receipt.json"


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load(path: Path, schema: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("schema") != schema:
        raise ValueError(f"{path}: schema {payload.get('schema')!r} != {schema!r}")
    return payload


def _source(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve().relative_to(REPO)),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def build_receipt(
    *,
    wall_path: Path,
    tile_path: Path,
    precision_path: Path,
    joint_ab_path: Path | None,
) -> dict[str, Any]:
    wall = _load(wall_path, "cheapen_real95_current_wall.v1")
    tile = _load(tile_path, "cheapen_real95_tile_halo_exactness.v1")
    precision = _load(precision_path, "cheapen_real95_mlx_precision_n600.v1")
    wall_row = wall["canonical_equation_ready_row"]
    tile_gate = tile["lever_a_gate"]
    precision_gate = precision["lever_b_verdict"]

    blockers: list[str] = []
    if not bool(wall_row.get("composition_admissible")):
        blockers.append("current scorer/render/R/loss wall components are not all MEASURED")
    if not (
        tile_gate.get("verdict") == "GO"
        and tile["evidence_grade"].get("speedup") == "MEASURED"
    ):
        blockers.append("Lever A lacks exact-on-tiles plus a MEASURED >=2x timing")
    if precision.get("status") != "MEASURED" or precision_gate.get("verdict") != "GO":
        blockers.append("Lever B lacks an n600 MEASURED cosine/throughput GO")

    joint: dict[str, Any] | None = None
    if joint_ab_path is None:
        blockers.append("overlapping A+B scorer work lacks a joint MEASURED before/after receipt")
    else:
        joint = _load(joint_ab_path, "cheapen_real95_joint_ab.v1")
        if joint.get("status") != "MEASURED":
            blockers.append("joint A+B receipt is not status=MEASURED")

    composition: dict[str, Any]
    if blockers:
        composition = {
            "status": "REFUSED_INCOMPLETE_MEASUREMENTS",
            "composed_speedup_x": None,
            "composed_seconds_per_epoch": None,
            "blockers": blockers,
            "forbidden_shortcut": "do not multiply isolated or derived A and B factors",
        }
    else:
        assert joint is not None
        baseline = float(wall_row["total_training_critical_path_s_per_epoch"])
        before = float(joint["joint_scorer_seconds_before"])
        after = float(joint["joint_scorer_seconds_after"])
        equation = AmdahlWallSplit(
            baseline_seconds=MeasuredSeconds(baseline, str(wall_path)),
            levers=(
                MeasuredLever(
                    "tile_halo_x_mixed_precision_joint",
                    "scorer_forward_backward_joint",
                    MeasuredSeconds(before, str(joint_ab_path)),
                    MeasuredSeconds(after, str(joint_ab_path)),
                ),
            ),
            async_cpu_verdict_service_seconds=MeasuredSeconds(
                float(
                    wall_row["components"][
                        "cpu_torch_verdict_service_amortized_s_per_epoch"
                    ]
                ),
                str(wall_path),
            ),
            async_cpu_verdict_critical_path_seconds=MeasuredSeconds(
                float(
                    wall_row["components"][
                        "cpu_torch_verdict_critical_path_s_per_epoch"
                    ]
                ),
                str(wall_path),
            ),
        ).compose()
        composition = {"status": "MEASURED_COMPOSITION", **equation}

    sources = {
        "wall": _source(wall_path),
        "tile_halo": _source(tile_path),
        "mlx_precision": _source(precision_path),
        "joint_ab": _source(joint_ab_path) if joint_ab_path is not None else None,
    }
    return {
        "schema": "cheapen_real95_composition.v1",
        "written_at_utc": _utc(),
        "lane_id": "lane_cheapen_real95_tilehalo_fp16_20260713",
        "axis": "[macOS-MLX training research-signal; NON-PROMOTABLE]",
        "canonical_equation_id": EQUATION_ID,
        "implementation": {
            "probe": _source(Path(__file__).resolve()),
            "equation": _source(
                REPO
                / "src/tac/canonical_equations/amdahl_measured_wall_split_20260713.py"
            ),
        },
        "sources": sources,
        "current_wall_s_per_epoch": wall["measured_wall"]["median_training_epoch_s"],
        "lever_a": {
            "verdict": tile_gate.get("verdict"),
            "exact_on_tiles": tile["exactness_check"]["exact_on_tiles_verified"],
            "measured_speedup_x": tile_gate.get("measured_speedup_at_coverage_x"),
            "exact_speedup_upper_bound_x": tile_gate.get(
                "exact_speedup_upper_bound_at_measured_coverage_x"
            ),
            "speedup_evidence": tile["evidence_grade"].get("speedup"),
        },
        "lever_b": {
            "verdict": precision_gate.get("verdict"),
            "cosine": precision_gate.get("cosine"),
            "speedup_x": precision_gate.get("speedup_x"),
            "status": precision.get("status"),
        },
        "composition": composition,
        "pointer_delta": "ZERO; composition receipt only",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wall", type=Path, default=DEFAULT_WALL)
    parser.add_argument("--tile-halo", type=Path, default=DEFAULT_TILE)
    parser.add_argument("--precision", type=Path, default=DEFAULT_PRECISION)
    parser.add_argument("--joint-ab", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    out = args.out.resolve()
    if str(out).startswith(("/tmp/", "/private/tmp/")):
        raise SystemExit("refusing /tmp durable evidence path")
    payload = build_receipt(
        wall_path=args.wall.resolve(),
        tile_path=args.tile_halo.resolve(),
        precision_path=args.precision.resolve(),
        joint_ab_path=args.joint_ab.resolve() if args.joint_ab else None,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(out)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
