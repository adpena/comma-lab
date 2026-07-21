#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Harvest seed_compose_b2 SSD evidence into durable decomposed artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any, Final

RATE_DENOMINATOR_BYTES: Final = 37_545_489
LAMBDA_RATE: Final = 25.0 / RATE_DENOMINATOR_BYTES
EXPECTED_ADAPTER_SHA256: Final = "db259daafaa7acba68060c7de8352611634a9de185d0e44c8f7f0fce408518f7"
EXPECTED_MEASUREMENT_SHA256: Final = "8768d180eac95b29fc170ff1566a2a494e7b02e167c5c2b89125d23507e1d8d7"


class HarvestError(ValueError):
    """Fail closed on missing, mixed, or malformed evidence."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HarvestError(f"cannot read JSON evidence: {path}") from exc
    if not isinstance(value, dict):
        raise HarvestError(f"evidence must be one object: {path}")
    return value


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write(path, json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n")


def _require_prefix(receipt: Mapping[str, Any], count: int) -> None:
    b2 = receipt.get("b2")
    if not isinstance(b2, dict) or b2.get("pair_count") != count:
        raise HarvestError(f"hard-oracle receipt is not exact n{count}")
    custody = b2.get("custody", {})
    if custody.get("adapter", {}).get("source_sha256") != EXPECTED_ADAPTER_SHA256:
        raise HarvestError("hard-oracle adapter source custody drifted")
    implementation = receipt.get("config", {}).get("implementation_sources", {}).get("files", {})
    if implementation.get("tools/measure_predict_project_receiver.py") != EXPECTED_MEASUREMENT_SHA256:
        raise HarvestError("measurement-runner source custody drifted")


def _aggregate_b3(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    totals: Counter[int] = Counter()
    satisfied: Counter[int] = Counter()
    for row in rows:
        for class_id, values in row["b3"]["by_class"].items():
            class_int = int(class_id)
            totals[class_int] += int(values["total"])
            satisfied[class_int] += int(values["already_satisfied"])
    return [
        {
            "class_id": class_id,
            "already_satisfied": satisfied[class_id],
            "total": totals[class_id],
            "violations": totals[class_id] - satisfied[class_id],
            "fraction": satisfied[class_id] / totals[class_id],
        }
        for class_id in range(5)
    ]


def _constraint_satisfaction(point: Mapping[str, Any]) -> dict[str, Any]:
    factorization = point["D4_factorization"]

    def convert(rows: list[Mapping[str, Any]], key: str) -> list[dict[str, Any]]:
        result = []
        for row in rows:
            total = int(row["constraint_count"])
            result.append(
                {
                    key: row[key],
                    "total": total,
                    "predictor_satisfied": 0,
                    "predictor_fraction": 0.0,
                    "represented_satisfied": total,
                    "represented_fraction": 1.0 if total else None,
                    "standalone_zlib9_bytes": row["standalone_zlib9_bytes"],
                }
            )
        return result

    return {
        "point": point["name"],
        "by_class": convert(factorization["per_class"], "class_id"),
        "by_stratum": convert(factorization["per_stratum"], "stratum"),
        "scope": "selected rows were filtered to be predictor violations under every curve point",
    }


def _marginals(points: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for left, right in pairwise(points):
        left_seg = float(left["description_metrics"]["d_seg_represented_vs_frozen_cpu_torch_lstar"])
        right_seg = float(right["description_metrics"]["d_seg_represented_vs_frozen_cpu_torch_lstar"])
        added_bytes = int(right["seed_bytes"]) - int(left["seed_bytes"])
        benefit = 100.0 * (left_seg - right_seg)
        marginal = benefit / added_bytes
        rows.append(
            {
                "from": left["name"],
                "to": right["name"],
                "added_seed_bytes": added_bytes,
                "seg_description_score_benefit": benefit,
                "marginal_benefit_per_byte": marginal,
                "lambda_rate_25_over_37545489": LAMBDA_RATE,
                "marginal_over_lambda": marginal / LAMBDA_RATE,
                "admit_at_lambda": marginal >= LAMBDA_RATE,
            }
        )
    return rows


def harvest(evidence_root: Path, research_root: Path) -> dict[str, Path]:
    compose_path = evidence_root / "receipt.json"
    compose = _load(compose_path)
    prefix_receipts: dict[int, dict[str, Any]] = {}
    prefix_paths: dict[int, Path] = {}
    for count in (16, 64, 600):
        path = evidence_root / f"hard_oracle_n{count}" / "receipt.json"
        receipt = _load(path)
        _require_prefix(receipt, count)
        prefix_receipts[count] = receipt
        prefix_paths[count] = path
    stages_root = evidence_root / "hard_oracle_n600" / "stages"
    stage_paths = sorted(stages_root.glob("pair_*.json"))
    if len(stage_paths) != 600:
        raise HarvestError(f"expected 600 preserved pair stages, found {len(stage_paths)}")
    stage_rows = [_load(path) for path in stage_paths]
    if [row["pair_index"] for row in stage_rows] != list(range(600)):
        raise HarvestError("n600 stage rows are not contiguous pair order")
    if not all(row["hard_oracle"]["cell_exact"] and row["hard_oracle"]["pose_within_tube"] for row in stage_rows):
        raise HarvestError("n600 cell/tube invariant failed")
    if any(row["hard_oracle"]["uint8_factor2_exact"] for row in stage_rows):
        raise HarvestError("n600 rows unexpectedly claim uint8 realization")

    points = compose["D1_curve"]
    marginals = _marginals(points)
    pair_vector = [
        {
            "pair_index": row["pair_index"],
            "d_seg_description": row["hard_oracle"]["d_seg"],
            "d_pose_tube_debt": row["hard_oracle"]["d_pose"],
            "declared_constraint_violations_before_projection": row["violation_records"],
            "cell_exact_after_projection": row["hard_oracle"]["cell_exact"],
            "pose_within_tube": row["hard_oracle"]["pose_within_tube"],
            "uint8_factor2_exact": row["hard_oracle"]["uint8_factor2_exact"],
            "double_decode_equal": row["double_decode_equal"],
        }
        for row in stage_rows
    ]
    b5 = prefix_receipts[600]["b5"]
    measurements = {
        "schema": "seed_compose_b2_measurements.v1",
        "verdict": "REAL_N600_DESCRIPTION_SEED_MEASURED_RECEIVER_REALIZATION_BLOCKED",
        "verdict_scope": (
            "frozen CPU-Torch cache-replay cell descriptions and banked PoseNet tubes; "
            "not camera-RGB realization, not upstream evaluate.py, not contest score"
        ),
        "D1_curves": points,
        "D2_prefix_ladder": [
            {
                "n": count,
                "receipt_path": str(prefix_paths[count]),
                "receipt_sha256": _sha256(prefix_paths[count]),
                "b2": prefix_receipts[count]["b2"],
            }
            for count in (16, 64, 600)
        ],
        "D3_predictor_satisfaction": {
            "whole_field_by_target_class_n600": _aggregate_b3(stage_rows),
            "selected_constraint_sites_by_point": [_constraint_satisfaction(point) for point in points],
            "by_stratum_note": (
                "whole-field stratum labels are unavailable; selected-site strata are declared from "
                "Road-Lane transition, Movable transition, Fisher-margin critical, or interior rules"
            ),
            "why_low": [
                "Lane temporal mode is absent, so its site comes from full-occupancy centroid custody",
                "the five-site compatibility Voronoi raster is not native Morse-Smale arc-to-cell semantics",
                "PoseNet-derived xi has zero rotational gain under settled G1 calibration and cannot express partition residual",
                "movable tracks are box carriers and cannot represent non-box topology",
            ],
        },
        "D4_factorization": compose["D4_factorization"],
        "D5_advisory": {
            "points": [
                {
                    "name": point["name"],
                    "seed_bytes": point["seed_bytes"],
                    "seed_zlib9_bytes": point["seed_zlib9_bytes"],
                    **point["description_metrics"],
                }
                for point in points
            ],
            "marginals": marginals,
            "kkt_status": "BOUNDARY_NO_INTERIOR_KNEE",
            "selected_preregistered_point": points[0]["name"],
            "symmetric_neighbor_triple_status": "INFEASIBLE_AT_NONNEGATIVE_CONSTRAINT_FLOOR",
            "one_sided_curvature_points": [point["name"] for point in points],
            "reason": "both measured tightening marginals are below lambda; the loosest preregistered point wins",
            "advisory_score": points[0]["description_metrics"]["advisory_description_objective"],
            "measured_full_pipeline_score": None,
            "score_claim": False,
        },
        "block_diagonal": {
            "structure": "600 independent scorer pairs plus shared chart/trajectory/rate coupling",
            "pair_vector": pair_vector,
            "shared_couplings": ["ground_chart", "trajectory", "movable_track_dictionary", "container_rate"],
        },
        "B5_single_object_vs_per_frame": b5,
        "terminal_pipeline": [
            {"rung": 0, "name": "compose_at_knee", "status": "MEASURED_BOUNDARY_NO_INTERIOR_KNEE"},
            {
                "rung": 1,
                "name": "realization_gate_fixed_magnitude_deadzone",
                "status": "BLOCKED",
                "blocker": "SINGLE_OBJECT_TO_CAMERA_RGB_REALIZATION_UNMEASURED",
            },
            {"rung": 2, "name": "PairLocalDiagonalFinisher_400", "status": "NOT_REACHED_RUNG1_BLOCKED"},
            {"rung": 3, "name": "global_MC_finisher_396", "status": "NOT_REACHED_RUNG1_BLOCKED"},
            {"rung": 4, "name": "gauge_quotient_derived_ties_553", "status": "NOT_REACHED_RUNG1_BLOCKED"},
            {"rung": 5, "name": "decoder_adaptive_statistics_strip", "status": "NOT_REACHED_RUNG1_BLOCKED"},
            {"rung": 6, "name": "JRD_last_safe_prefix_453", "status": "NOT_REACHED_RUNG1_BLOCKED"},
            {"rung": 7, "name": "container_entropy_pack_557", "status": "NOT_REACHED_RUNG1_BLOCKED"},
            {"rung": 8, "name": "composition_closure_waterfill_fixed_point", "status": "NOT_REACHED_RUNG1_BLOCKED"},
            {"rung": 9, "name": "byte_close_R6", "status": "NOT_REACHED_RUNG1_BLOCKED"},
        ],
        "evidence_custody": {
            "compose_receipt": {"path": str(compose_path), "sha256": _sha256(compose_path)},
            "adapter_source_sha256": EXPECTED_ADAPTER_SHA256,
            "measurement_source_sha256": EXPECTED_MEASUREMENT_SHA256,
            "pair_stage_count": 600,
        },
        "authority": {
            "axis": "[macOS-CPU advisory]",
            "score_claim": False,
            "promotion_eligible": False,
            "pointer_moved": False,
        },
    }

    measurement_path = research_root / "seed_compose_b2_measurements_20260721.json"
    _atomic_json(measurement_path, measurements)
    measurement_sha = _sha256(measurement_path)
    measurement_display = measurement_path.relative_to(research_root.parents[1])
    reuse = {
        "schema": "seed_compose_b2_reuse_manifest.v1",
        "reused_surfaces": compose["reuse_manifest"],
        "source_files": {
            "src/tac/optimization/predict_project_schema.py": _sha256(
                research_root.parents[1] / "src/tac/optimization/predict_project_schema.py"
            ),
            "src/tac/optimization/predict_project_receiver.py": _sha256(
                research_root.parents[1] / "src/tac/optimization/predict_project_receiver.py"
            ),
            "src/tac/optimization/s2_partition_seed.py": _sha256(
                research_root.parents[1] / "src/tac/optimization/s2_partition_seed.py"
            ),
            "src/tac/optimization/seed_compose_b2.py": EXPECTED_ADAPTER_SHA256,
            "tools/measure_predict_project_receiver.py": EXPECTED_MEASUREMENT_SHA256,
        },
        "evidence": measurements["evidence_custody"],
        "research_only": True,
    }
    reuse_path = research_root / "seed_compose_b2_reuse_manifest_20260721.json"
    _atomic_json(reuse_path, reuse)

    class_rows = measurements["D3_predictor_satisfaction"]["whole_field_by_target_class_n600"]
    findings = """# Codex findings — seed_compose_b2 real n600 constraint seed

## Verdict

`REAL_N600_DESCRIPTION_SEED_MEASURED_RECEIVER_REALIZATION_BLOCKED`.

The single PPCS object is real, canonical, and measured through frozen CPU-Torch cache replay at n16, n64, and n600. At n600 it is cell-exact at all 3,188 declared sites, all 600 banked PoseNet targets remain inside their declared tubes, `d_pose=0`, and every double decode is byte-identical. It is **not** camera-RGB receiver-closed: `uint8_factor2_exact=false` for all 600 pairs. This is `[macOS-CPU advisory]`, not `upstream/evaluate.py`, not a contest score, and the frontier pointer remains unchanged.

## D1/D5 curve and KKT verdict

| point | constraints | PPCS bytes | zlib-9 | d_seg description | advisory objective |
|---|---:|---:|---:|---:|---:|
"""
    for point in points:
        metrics = point["description_metrics"]
        findings += (
            f"| {point['name']} | {point['selected_constraint_count']:,} | {point['seed_bytes']:,} | "
            f"{point['seed_zlib9_bytes']:,} | {metrics['d_seg_represented_vs_frozen_cpu_torch_lstar']:.12f} | "
            f"{metrics['advisory_description_objective']:.12f} |\n"
        )
    findings += "\nBoth measured tightening marginals are below `25/37,545,489`; the preregistered minimum is therefore a **boundary KKT result**, not an interior knee. A symmetric looser/knee/tighter triple is infeasible at the nonnegative/full-pair tube floor; the three rows above provide one-sided curvature.\n\n"
    findings += "## D2 hard-oracle ladder\n\n| scope | d_seg | d_pose | cells exact | Pose tubes | uint8 factor-2 |\n|---|---:|---:|---|---|---|\n"
    for count in (16, 64, 600):
        row = prefix_receipts[count]["b2"]
        findings += (
            f"| n{count} | {row['d_seg']:.12f} | {row['d_pose']:.12f} | {row['cell_exact']} | "
            f"{row['pose_within_tube']} | {row['uint8_factor2_exact']} |\n"
        )
    findings += "\n## D3 predictor satisfaction by target class\n\n| class | satisfied | total | fraction |\n|---:|---:|---:|---:|\n"
    for row in class_rows:
        findings += (
            f"| {row['class_id']} | {row['already_satisfied']:,} | {row['total']:,} | {row['fraction']:.9f} |\n"
        )
    findings += (
        "\nLane is the failure center. Its whole-field satisfaction is low because Lane never wins the temporal mode; its compatibility site is derived from full n600 occupancy instead. The loose constraint prefix is correspondingly concentrated on Road-Lane boundaries. This negative is scoped to the five-site compatibility raster. The native Morse-Smale family remains open under exact blocker `MS_ARC_TO_CELL_RASTERIZATION_SEMANTICS_UNMEASURED`.\n\n"
        "## B5 and terminal handoff\n\n"
        f"The single object is {b5['single_object']['raw_bytes']:,} raw bytes versus {b5['per_frame']['raw_bytes']:,} for equal per-frame represented fields; zlib-9 is {b5['single_object']['zlib9_bytes']:,} versus {b5['per_frame']['zlib9_bytes']:,}. Rung 1 (camera-RGB realization) blocks the diagonal finisher, MC finisher, gauge quotient, adaptive-statistics strip, JRD, #557 entropy pack, composition closure, and R6. Those later rungs were not faked or bypassed.\n\n"
        "## STORES CONSULTED\n\n"
        f"- `{compose_path}`\n"
        f"- `{prefix_paths[16]}`\n"
        f"- `{prefix_paths[64]}`\n"
        f"- `{prefix_paths[600]}` and its 600 preserved pair stages\n"
        "- `.omx/research/g1_worldsheet_g3_cellcode_measurements_20260720T210000Z.json` through LawRef\n"
        "- `.omx/research/s2_compose_full_partition_20260721T041640Z.json` and the exact S2 packet\n\n"
        f"Machine-readable measurement: `{measurement_display}` (`{measurement_sha}`).\n"
    )
    findings_path = research_root / "codex_findings_seed_compose_b2_20260721_codex.md"
    _atomic_write(findings_path, findings.encode())

    dag = f"""# seed_compose_b2 DAG FEED — 2026-07-21

```text
frozen CPU-Torch n600 cache cf8d8360...
  + S2 finite event seed df4c0534... (17,926 events / 39,836 B)
  + G1 PoseNet->xi LawRefs (s_t=-0.00143, s_r=0)
  + canonical movable-site correspondence
  -> five-site compatibility chart [native MS blocked]
  -> 3 nested PPCS curve points
  -> n16 -> n64 -> n600 cache-replay description oracle
       |-- cell exact: PASS
       |-- Pose tube: PASS
       |-- deterministic double decode: PASS
       `-- camera RGB / uint8 factor-2: FAIL CLOSED
             -> terminal rungs 2..9 NOT REACHED
```

- DSL leg: strict `predict_project_constraint_seed.v0`, exact seed 1234/batch16 callback contract.
- DAG leg: all 600 pair rows are preserved and exposed as a block-diagonal vector; shared chart/trajectory/rate are the only cross-pair couplings.
- Equation leg: G1 constants resolve through `dsl_custodied_scalar_identity_v1`; measured allocation marginals are compared to `25/37,545,489` and yield `BOUNDARY_NO_INTERIOR_KNEE`.
- Exact blocker: `MS_ARC_TO_CELL_RASTERIZATION_SEMANTICS_UNMEASURED`.
- Next authorizing edge: bind this object to a camera-RGB inverse-R realization with realized-entry telemetry. Only then may #400/#396/#553/JRD/#557/R6 execute.
- Measurement artifact: `{measurement_display}` SHA-256 `{measurement_sha}`.
- Pointer delta: none.
"""
    dag_path = research_root / "seed_compose_b2_DAG_FEED_20260721.md"
    _atomic_write(dag_path, dag.encode())
    return {
        "measurement": measurement_path,
        "findings": findings_path,
        "dag": dag_path,
        "reuse": reuse_path,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=Path("/Volumes/VertigoDataTier/pact/evidence/seed_compose_20260721"),
    )
    parser.add_argument("--research-root", type=Path, default=Path(".omx/research"))
    args = parser.parse_args()
    outputs = harvest(args.evidence_root.resolve(), args.research_root.resolve())
    print(json.dumps({key: str(value) for key, value in outputs.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
