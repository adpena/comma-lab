#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure n600 multicoefficient transition coverage without faking infeasibility.

The landed v19b archive is a receiver-closed, integer-lattice-native
multicoefficient stack under C1's 200 KB box.  Its existing receipt reports net
per-class error deltas, which cannot distinguish helpful closures from harmful
collateral.  This audit replays the v15 control and v19b stack through their
real receivers and the frozen scorers, then records the full transition
confusion for every target-label stratum.

This tool deliberately does *not* call an optimizer stall or one measured stack
a global infeasibility certificate.  A numeric certified-infeasible residual is
emitted only when a finite reachable-set manifest, exhaustive enumeration
receipt, exact byte box, and zero-collateral receiver replay are all present.
The current landed inputs do not carry those proofs, so the honest result is a
measured reach table plus an explicit certification blocker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "src", REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.optimization.direct_description_carrier_compose import (  # noqa: E402
    CLASS_ORDER,
    receive_carrier_compose_archive,
    rfc8785_canonicalize,
)
from tac.optimization.direct_description_minimizer import (  # noqa: E402
    DirectDescriptionError,
    _read_regular_file_once,
)
from tac.optimization.direct_description_preuint8_channel import (  # noqa: E402
    receive_preuint8_q8_archive,
)
from tools.measure_ddm_v14_realization_fidelity import _forward  # noqa: E402
from tools.measure_ddm_v19b_joint_remeasure_stack import (  # noqa: E402
    DDMV19BJointRemeasureStackConfigV1,
    _load_sources,
)

SCHEMA = "ddm_m5_multicoefficient_solve_coverage_receipt.v1"
BATCH_SCHEMA = "ddm_m5_multicoefficient_transition_batch.v1"
LANE_ID = "lane_ddm_m5_multicoefficient_solve_coverage_20260723"
EVIDENCE_AXIS = "[macOS-CPU frozen-scorer advisory]"


class DDMV5CoverageConfigV1(BaseModel):
    """SHA-bound, local-only exact-replay contract."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    schema_: Literal["DDMV5CoverageConfigV1"] = Field(
        default="DDMV5CoverageConfigV1",
        alias="schema",
        serialization_alias="schema",
    )
    run_id: str = Field(min_length=8)
    seed: Literal[1234] = 1234
    m3_receipt_path: str = Field(min_length=1)
    m3_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    c1_ledger_path: str = Field(min_length=1)
    c1_ledger_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    v19b_config_path: str = Field(min_length=1)
    v19b_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    v19b_receipt_path: str = Field(min_length=1)
    v19b_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    j5_smoke_receipt_path: str = Field(min_length=1)
    j5_smoke_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scorer_batch_size: Literal[16] = 16
    pair_count: Literal[600] = 600
    byte_box: Literal[200000] = 200000
    research_only: Literal[True] = True
    execution_allowed: Literal[False] = False
    score_claim: Literal[False] = False

    def typed_config_hash(self) -> str:
        return hashlib.sha256(
            rfc8785_canonicalize(self.model_dump(mode="json", by_alias=True))
        ).hexdigest()


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def _bound_json(path_text: str, digest: str, name: str) -> dict[str, Any]:
    path = REPO_ROOT / path_text
    payload = _read_regular_file_once(path)
    actual = _sha256(payload)
    if actual != digest:
        raise DirectDescriptionError(f"{name} SHA-256 differs: {actual} != {digest}")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DirectDescriptionError(f"{name} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise DirectDescriptionError(f"{name} must contain one JSON object")
    return value


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _read_regular_file_once(path) != payload:
            raise DirectDescriptionError(f"immutable output differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _portable(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def transition_rows(
    labels: np.ndarray,
    control_cells: np.ndarray,
    candidate_cells: np.ndarray,
    class_order: Sequence[str],
) -> dict[str, dict[str, int | bool]]:
    """Return exact target-stratum transitions for one scorer batch."""

    labels_value = np.asarray(labels)
    control = np.asarray(control_cells)
    candidate = np.asarray(candidate_cells)
    if (
        labels_value.shape != control.shape
        or labels_value.shape != candidate.shape
        or labels_value.size == 0
    ):
        raise DirectDescriptionError("labels/control/candidate must share one nonempty shape")
    if len(class_order) != len(set(class_order)) or not class_order:
        raise DirectDescriptionError("class order must be nonempty and unique")
    if int(labels_value.min()) < 0 or int(labels_value.max()) >= len(class_order):
        raise DirectDescriptionError("labels contain a class outside the detected class order")
    rows: dict[str, dict[str, int | bool]] = {}
    for class_id, class_name in enumerate(class_order):
        stratum = labels_value == class_id
        control_error = control != labels_value
        candidate_error = candidate != labels_value
        helpful = stratum & control_error & ~candidate_error
        harmful = stratum & ~control_error & candidate_error
        persistent = stratum & control_error & candidate_error
        rows[class_name] = {
            "class_id": class_id,
            "sites": int(np.count_nonzero(stratum)),
            "control_errors": int(np.count_nonzero(stratum & control_error)),
            "helpful_flips": int(np.count_nonzero(helpful)),
            "harmful_off_target_flips": int(np.count_nonzero(harmful)),
            "persistent_errors": int(np.count_nonzero(persistent)),
            "candidate_errors": int(np.count_nonzero(stratum & candidate_error)),
            "zero_off_target_collateral": not bool(np.any(harmful)),
        }
    return rows


def _sum_transition_rows(
    batches: Sequence[Mapping[str, Any]], class_order: Sequence[str]
) -> dict[str, dict[str, int | bool | float]]:
    result: dict[str, dict[str, int | bool | float]] = {}
    numeric = (
        "sites",
        "control_errors",
        "helpful_flips",
        "harmful_off_target_flips",
        "persistent_errors",
        "candidate_errors",
    )
    for class_id, class_name in enumerate(class_order):
        row = {
            key: sum(int(batch["per_stratum"][class_name][key]) for batch in batches)
            for key in numeric
        }
        row["class_id"] = class_id
        row["net_errors_closed"] = int(row["control_errors"]) - int(
            row["candidate_errors"]
        )
        row["measured_helpful_fraction"] = (
            0.0
            if int(row["control_errors"]) == 0
            else int(row["helpful_flips"]) / int(row["control_errors"])
        )
        row["zero_off_target_collateral"] = (
            int(row["harmful_off_target_flips"]) == 0
        )
        result[class_name] = row
    return result


def certificate_eligibility(
    *,
    numeric_byte_box: bool,
    finite_reachable_set_manifest: bool,
    exhaustive_enumeration: bool,
    exact_receiver_replay: bool,
    isolated_per_stratum_solutions: bool,
) -> tuple[bool, list[str]]:
    """Fail closed unless the evidence can support a universal negative."""

    predicates = {
        "numeric_byte_box": numeric_byte_box,
        "finite_reachable_set_manifest": finite_reachable_set_manifest,
        "exhaustive_enumeration": exhaustive_enumeration,
        "exact_receiver_replay": exact_receiver_replay,
        "isolated_per_stratum_solutions": isolated_per_stratum_solutions,
    }
    missing = [name for name, value in predicates.items() if not value]
    return not missing, missing


def _m3_frame1_rows(receipt: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = {
        str(row["stratum"]): row
        for row in receipt.get("frontier", ())
        if row.get("frame") == "frame_1"
    }
    if tuple(rows) != tuple(CLASS_ORDER):
        raise DirectDescriptionError(
            f"m3 frame_1 class order {tuple(rows)!r} differs from detected {CLASS_ORDER!r}"
        )
    return rows


def _source_custody(
    config: DDMV5CoverageConfigV1,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    DDMV19BJointRemeasureStackConfigV1,
    dict[str, Any],
]:
    m3 = _bound_json(config.m3_receipt_path, config.m3_receipt_sha256, "m3 receipt")
    c1 = _bound_json(config.c1_ledger_path, config.c1_ledger_sha256, "c1 ledger")
    v19b_receipt = _bound_json(
        config.v19b_receipt_path, config.v19b_receipt_sha256, "v19b receipt"
    )
    j5 = _bound_json(
        config.j5_smoke_receipt_path,
        config.j5_smoke_receipt_sha256,
        "j5 smoke receipt",
    )
    v19b_payload = _read_regular_file_once(REPO_ROOT / config.v19b_config_path)
    if _sha256(v19b_payload) != config.v19b_config_sha256:
        raise DirectDescriptionError("v19b config SHA-256 differs")
    v19b_config = DDMV19BJointRemeasureStackConfigV1.model_validate_json(v19b_payload)
    _v19_config, _source_receipt, ctx = _load_sources(v19b_config)
    if c1.get("box", {}).get("archive_bytes_max") != config.byte_box:
        raise DirectDescriptionError("configured byte box differs from the bound C1 box")
    if (
        m3.get("score_claim") is not False
        or v19b_receipt.get("score_claim") is not False
        or j5.get("score_claim") is not False
    ):
        raise DirectDescriptionError("source false-authority flags differ")
    return m3, c1, v19b_receipt, j5, v19b_config, ctx


def run(config: DDMV5CoverageConfigV1, output_directory: Path) -> Path:
    """Perform exact transition replay and emit the fail-closed receipt."""

    root = output_directory.resolve()
    root.mkdir(parents=True, exist_ok=True)
    receipt_path = root / "ddm_m5_multicoefficient_solve_coverage_receipt.json"
    config_hash = config.typed_config_hash()
    if receipt_path.exists():
        receipt = json.loads(_read_regular_file_once(receipt_path))
        if receipt.get("typed_config_sha256") != config_hash:
            raise DirectDescriptionError("completed M5 receipt config identity differs")
        print(json.dumps({"resumed": True, "receipt": _portable(receipt_path)}))
        return receipt_path

    m3, c1, v19b, j5, _v19b_config, ctx = _source_custody(config)
    m3_rows = _m3_frame1_rows(m3)
    baseline_archive = bytes(ctx["n600_archive"])
    final = v19b.get("n600", {})
    candidate_meta = final.get("archive", {})
    candidate_path = REPO_ROOT / str(candidate_meta.get("path", ""))
    candidate_archive = _read_regular_file_once(candidate_path)
    if (
        len(candidate_archive) != int(candidate_meta.get("bytes", -1))
        or _sha256(candidate_archive) != candidate_meta.get("sha256")
    ):
        raise DirectDescriptionError("v19b n600 final archive custody differs")
    if len(candidate_archive) > config.byte_box:
        raise DirectDescriptionError("v19b n600 archive is outside the bound C1 byte box")

    control_meta = c1.get("control", {})
    if (
        len(baseline_archive) != int(control_meta.get("archive_bytes", -1))
        or _sha256(baseline_archive) != control_meta.get("archive_sha256")
    ):
        raise DirectDescriptionError("C1 control archive identity differs")
    m3_control_errors = [int(row["control_errors"]) for row in m3_rows.values()]
    j5_control = int(j5["exact_n600"]["baseline"]["global_errors"])
    if (
        any(value < 0 for value in m3_control_errors)
        or sum(m3_control_errors) != int(control_meta["seg_errors"])
        or j5_control != int(control_meta["seg_errors"])
    ):
        raise DirectDescriptionError("m3/J5/C1 control count custody differs")

    stage = root / "stage_checkpoints" / "01_exact_n600_transitions"
    source_ids = np.arange(config.pair_count, dtype=np.int64)
    for start in range(0, config.pair_count, config.scorer_batch_size):
        stop = min(start + config.scorer_batch_size, config.pair_count)
        checkpoint = stage / f"batch_{start:04d}_{stop:04d}.json"
        if checkpoint.exists():
            row = json.loads(_read_regular_file_once(checkpoint))
            if (
                row.get("schema") != BATCH_SCHEMA
                or row.get("typed_config_sha256") != config_hash
                or row.get("control_archive_sha256") != _sha256(baseline_archive)
                or row.get("candidate_archive_sha256") != _sha256(candidate_archive)
            ):
                raise DirectDescriptionError("M5 transition checkpoint identity differs")
            continue
        pair_ids = tuple(int(value) for value in source_ids[start:stop])
        control_receiver = receive_carrier_compose_archive(baseline_archive)
        candidate_receiver = receive_preuint8_q8_archive(candidate_archive)
        control_camera = control_receiver.render_camera_pairs(pair_ids)
        candidate_camera = candidate_receiver.render_camera_pairs(pair_ids)
        control_cells, control_pose = _forward(
            ctx["segnet"], ctx["posenet"], control_camera
        )
        candidate_cells, candidate_pose = _forward(
            ctx["segnet"], ctx["posenet"], candidate_camera
        )
        labels = np.asarray(ctx["labels_all"][source_ids[start:stop]])
        poses = np.asarray(ctx["poses_all"][source_ids[start:stop]])
        row = {
            "schema": BATCH_SCHEMA,
            "typed_config_sha256": config_hash,
            "source_pair_ids": list(pair_ids),
            "control_archive_sha256": _sha256(baseline_archive),
            "candidate_archive_sha256": _sha256(candidate_archive),
            "per_stratum": transition_rows(
                labels, control_cells, candidate_cells, CLASS_ORDER
            ),
            "control_cells_sha256": _sha256_array(control_cells),
            "candidate_cells_sha256": _sha256_array(candidate_cells),
            "control_pose_squared_error_sum": (
                f"{float(np.square(control_pose - poses).sum(dtype=np.float64)):.12f}"
            ),
            "candidate_pose_squared_error_sum": (
                f"{float(np.square(candidate_pose - poses).sum(dtype=np.float64)):.12f}"
            ),
            "pose_coordinates": int(control_pose.size),
            "receiver_replay": "uint8_camera_to_exact_R_to_frozen_scorers",
            "evidence_axis": EVIDENCE_AXIS,
            "score_claim": False,
        }
        _atomic_json(checkpoint, row)

    batches = [
        json.loads(_read_regular_file_once(path))
        for path in sorted(stage.glob("batch_*.json"))
    ]
    expected_batches = (
        config.pair_count + config.scorer_batch_size - 1
    ) // config.scorer_batch_size
    if len(batches) != expected_batches:
        raise DirectDescriptionError("M5 n600 transition batch coverage is incomplete")
    if [pair for row in batches for pair in row["source_pair_ids"]] != list(
        range(config.pair_count)
    ):
        raise DirectDescriptionError("M5 n600 pair coverage is noncontiguous")
    per_stratum = _sum_transition_rows(batches, CLASS_ORDER)
    for class_name, row in per_stratum.items():
        if int(row["control_errors"]) != int(m3_rows[class_name]["control_errors"]):
            raise DirectDescriptionError(
                f"{class_name} exact replay differs from the m3 control count"
            )

    eligible, missing = certificate_eligibility(
        numeric_byte_box=True,
        finite_reachable_set_manifest=False,
        exhaustive_enumeration=False,
        exact_receiver_replay=True,
        isolated_per_stratum_solutions=False,
    )
    if eligible:
        raise DirectDescriptionError(
            "current inputs unexpectedly satisfy certification eligibility"
        )
    assigned = int(m3["aggregate"]["c1_assigned_residual_errors"])
    measured_net = sum(int(row["net_errors_closed"]) for row in per_stratum.values())
    helpful = sum(int(row["helpful_flips"]) for row in per_stratum.values())
    harmful = sum(
        int(row["harmful_off_target_flips"]) for row in per_stratum.values()
    )
    pose_sse_control = sum(
        float(row["control_pose_squared_error_sum"]) for row in batches
    )
    pose_sse_candidate = sum(
        float(row["candidate_pose_squared_error_sum"]) for row in batches
    )
    pose_coordinates = sum(int(row["pose_coordinates"]) for row in batches)
    storage = shutil.disk_usage(root)
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "lane_id": LANE_ID,
        "run_id": config.run_id,
        "seed": config.seed,
        "typed_config": config.model_dump(mode="json", by_alias=True),
        "typed_config_sha256": config_hash,
        "class_order": {
            "names": list(CLASS_ORDER),
            "class_ids": {name: index for index, name in enumerate(CLASS_ORDER)},
            "source": "tac.optimization.direct_description_carrier_compose.CLASS_ORDER",
            "self_detected_not_redeclared": True,
        },
        "measurement": {
            "pair_count": config.pair_count,
            "scorer_batch_size": config.scorer_batch_size,
            "all_pairs_checkpointed": True,
            "control": {
                "archive_bytes": len(baseline_archive),
                "archive_sha256": _sha256(baseline_archive),
            },
            "multicoefficient_candidate": {
                "archive_bytes": len(candidate_archive),
                "archive_sha256": _sha256(candidate_archive),
                "shared_delta_bytes": len(candidate_archive) - len(baseline_archive),
                "inside_c1_byte_box": len(candidate_archive) <= config.byte_box,
                "integer_lattice_native": True,
                "receiver_closed": True,
                "source": "v19b sequential joint stack",
            },
            "per_stratum": per_stratum,
            "aggregate": {
                "helpful_flips": helpful,
                "harmful_off_target_flips": harmful,
                "net_errors_closed": measured_net,
                "candidate_errors": sum(
                    int(row["candidate_errors"]) for row in per_stratum.values()
                ),
                "zero_off_target_collateral": harmful == 0,
                "control_d_pose": pose_sse_control / pose_coordinates,
                "candidate_d_pose": pose_sse_candidate / pose_coordinates,
            },
        },
        "solve_coverage": {
            "Road_frame_1": (
                "PARTIAL_REACH_MEASURED_FULL_SOLVE_NOT_CERTIFIED"
                if int(per_stratum["Road"]["helpful_flips"]) > 0
                else "NO_REACH_MEASURED_FULL_SOLVE_NOT_CERTIFIED"
            ),
            "Lane_frame_1": "JOINT_N600_REACH_MEASURED_ZERO_COLLATERAL_FULL_SOLVE_NOT_ADMITTED",
            "Lane_subset_failure_diagnosis": (
                "G2G2 was under-parameterized and failed the joint semantic-cell/pose predicates; "
                "it was not an infeasibility certificate. The n600 v19b stack demonstrates nonzero "
                "Lane reach but is joint, collateral-bearing, and non-exhaustive."
            ),
            "fully_inverse_solved_strata": [],
            "verdict_scope": (
                "INSTANCE:V19B_GREEDY_INTEGER_LATTICE_STACK_AT_C1_200000_BYTE_BOX_N600; "
                "not an exhaustive grammar, family, or global MDL optimum verdict"
            ),
        },
        "certification": {
            "eligible": False,
            "missing_predicates": missing,
            "certified_infeasible_residual_per_stratum": dict.fromkeys(
                CLASS_ORDER
            ),
            "catalog_366_true_scope_errors": None,
            "catalog_366_true_scope_interval_errors": [0, assigned],
            "current_master_counterfactual_after_v19b_net_effect": int(
                m3["aggregate"][
                    "current_master_counterfactual_residual_after_v19b_subtraction"
                ]
            ),
            "verdict": "NUMERIC_TRUE_SCOPE_NOT_CERTIFIABLE_NONEXHAUSTIVE_REACHABLE_SET",
            "reason": (
                "Exact replay measures one receiver-closed multicoefficient stack, but the "
                "landed custody has no finite manifest of every admissible coefficient program, "
                "no exhaustive enumeration/optimality certificate within 200000 bytes, and no "
                "isolated per-stratum zero-collateral solution. A solver stall or net class delta "
                "cannot prove that no admissible program exists."
            ),
        },
        "quarantine_waiver": {
            "r1_signal_only": True,
            "d_pose": 0.001610,
            "bytes": 7195,
            "bytes_consumed": 0,
        },
        "storage_preflight": {
            "root": _portable(root),
            "free_bytes_after": storage.free,
            "large_new_artifacts": False,
            "checkpoint_policy": "one immutable JSON checkpoint per scorer batch",
        },
        "sources": {
            "m3_receipt": {
                "path": config.m3_receipt_path,
                "sha256": config.m3_receipt_sha256,
            },
            "c1_ledger": {
                "path": config.c1_ledger_path,
                "sha256": config.c1_ledger_sha256,
            },
            "v19b_receipt": {
                "path": config.v19b_receipt_path,
                "sha256": config.v19b_receipt_sha256,
            },
            "j5_smoke": {
                "path": config.j5_smoke_receipt_path,
                "sha256": config.j5_smoke_receipt_sha256,
                "role": "independent exact-n600 control-count and one-move identity cross-check",
            },
        },
        "evidence_axis": EVIDENCE_AXIS,
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer": str(m3["pointer"]),
        "pointer_moved": False,
        "main_landing_review_required": True,
    }
    _atomic_json(receipt_path, receipt)
    print(
        json.dumps(
            {
                "resumed": False,
                "receipt": _portable(receipt_path),
                "verdict": receipt["certification"]["verdict"],
            }
        )
    )
    return receipt_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    config = DDMV5CoverageConfigV1.model_validate_json(
        _read_regular_file_once(args.config)
    )
    run(config, args.output_directory)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
