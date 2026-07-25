# SPDX-License-Identifier: MIT
"""Fail-closed custody compiler for the DDM WS4 pose-null warm-start arm.

WS4 is a composition audit before it is an optimizer.  A projection is only
lawful when a measured pose-coupling row can be joined to an actual W_seg
correction through an explicit source-decision key.  Pair/bucket coincidence
is not a join key.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = "ddm_ws4_pose_null_projected_start.v1"
ARBITRATION_SCHEMA = "ddm_ws3_warm_start_slope_arbitration.v1"
EVIDENCE_AXIS = "[macOS-CPU frozen-scorer advisory]"
POINTER = "0.1910828242 [contest-CPU]"
CRITICAL_RATIO = 4.1215446777965665
DM2_POSE_HARM_ROWS = (5, 10, 11, 12, 23)
DM4_EXACT_CURE_ROWS = (5, 23)
DM4_UNCURED_ROWS = (10, 11, 12)
WSEG_N600 = {
    "archive_bytes": 138031,
    "archive_sha256": "264a09abb8f614eca104eb4ab1d0a12005ba65ec6a4fbc6620ff92f1c73281a9",
    "d_pose": 146.36493245487773,
    "d_seg": 0.024124510023328993,
    "errors": 2845843,
    "num_pairs": 600,
}


class WS4PoseNullError(ValueError):
    """Raised when a custody or projection premise fails closed."""


@dataclass(frozen=True, slots=True)
class BoundArtifact:
    path: Path
    sha256: str
    bytes: int

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any], *, repo_root: Path) -> BoundArtifact:
        raw_path = Path(str(payload["path"]))
        path = raw_path if raw_path.is_absolute() else repo_root / raw_path
        return cls(path=path, sha256=str(payload["sha256"]), bytes=int(payload["bytes"]))

    def read(self) -> bytes:
        payload = self.path.read_bytes()
        observed = hashlib.sha256(payload).hexdigest()
        if len(payload) != self.bytes or observed != self.sha256:
            raise WS4PoseNullError(
                f"custody mismatch for {self.path}: "
                f"expected {self.bytes}/{self.sha256}, observed {len(payload)}/{observed}"
            )
        return payload

    def read_json(self) -> dict[str, Any]:
        value = json.loads(self.read())
        if not isinstance(value, dict):
            raise WS4PoseNullError(f"JSON artifact is not an object: {self.path}")
        return value

    def receipt(self) -> dict[str, Any]:
        return {"path": str(self.path), "sha256": self.sha256, "bytes": self.bytes}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _require_schema(payload: Mapping[str, Any], schema: str, label: str) -> None:
    if payload.get("schema") != schema:
        raise WS4PoseNullError(f"{label} schema differs: {payload.get('schema')!r}")


def _recursive_values(payload: Any, key: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(payload, Mapping):
        for name, value in payload.items():
            if name == key:
                found.append(value)
            found.extend(_recursive_values(value, key))
    elif isinstance(payload, list):
        for value in payload:
            found.extend(_recursive_values(value, key))
    return found


def classify_projection_components(
    *,
    ws1: Mapping[str, Any],
    ws2: Mapping[str, Any],
    dm2: Mapping[str, Any],
    dm4: Mapping[str, Any],
    ws3_arbitration: Mapping[str, Any],
    cc3: Mapping[str, Any],
    j9_ticket: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify only custody-joinable W_seg corrections by pose coupling.

    The expected result for the sealed WS4 inputs is an empty projection set.
    This is not an optimizer failure: it is the only lawful outcome when DM2's
    G3 demand rows have no foreign key to the W_seg decision chain.
    """

    _require_schema(ws1, "ddm_ws1_seglex96_filtered_warmstart_measurement.v1", "WS1")
    _require_schema(ws2, "ddm_ws2_warm_start_custody_producer.v1", "WS2")
    _require_schema(dm2, "ddm_dm2_l3_realization_race.v1", "DM2")
    _require_schema(dm4, "ddm_dm4_targeted_realization_cures.v1", "DM4")
    _require_schema(ws3_arbitration, ARBITRATION_SCHEMA, "WS3 arbitration")
    _require_schema(cc3, "ddm_cc3_mixed_coder_receiver_integration_mirror.v1", "CC3")

    controls = ws1.get("controls")
    if not isinstance(controls, Mapping):
        raise WS4PoseNullError("WS1 controls missing")
    base = controls.get("seglex96_base")
    temporal = controls.get("temporal_hood_masked")
    if not isinstance(base, Mapping) or not isinstance(temporal, Mapping):
        raise WS4PoseNullError("WS1 W_seg control rows missing")
    suffix_pose_delta = float(temporal["d_pose"]) - float(base["d_pose"])
    suffix_seg_delta = float(temporal["d_seg"]) - float(base["d_seg"])
    if suffix_pose_delta >= 0.0:
        raise WS4PoseNullError("sealed W_seg temporal suffix is no longer pose-beneficial")

    ws2_wseg = ws2.get("archive_custody", {}).get("W_seg")
    ws2_metrics = ws2.get("fresh_batch32_endpoints", {}).get("W_seg")
    if not isinstance(ws2_wseg, Mapping) or not isinstance(ws2_metrics, Mapping):
        raise WS4PoseNullError("WS2 W_seg custody or n600 endpoint missing")
    for field, expected in WSEG_N600.items():
        if field == "num_pairs":
            continue
        observed = ws2_wseg.get(field) if field in {"archive_bytes", "archive_sha256"} else ws2_metrics.get(field)
        if observed != expected:
            raise WS4PoseNullError(f"WS2 W_seg {field} differs: {observed!r} != {expected!r}")
    if ws2_wseg.get("j5_stage00_lift_recompile_byte_identical") is not True:
        raise WS4PoseNullError("WS2 J5 W_seg byte-identity receipt is not true")

    dm2_rows = dm2.get("rows")
    dm4_rows = dm4.get("rows")
    if not isinstance(dm2_rows, list) or not isinstance(dm4_rows, list):
        raise WS4PoseNullError("DM2/DM4 row arrays missing")
    pose_harm = tuple(
        int(row["row_index"])
        for row in dm2_rows
        if isinstance(row, Mapping) and row.get("pose", {}).get("pose_nonharm") is False
    )
    if pose_harm != DM2_POSE_HARM_ROWS:
        raise WS4PoseNullError(f"DM2 pose-harm rows differ: {pose_harm!r}")
    dm4_by_index = {int(row["row_index"]): row for row in dm4_rows if isinstance(row, Mapping)}
    cured = tuple(
        index
        for index in DM2_POSE_HARM_ROWS
        if dm4_by_index[index].get("cure_disposition") == "POSE_HARM_CURED_EXACT_L4"
    )
    uncured = tuple(
        index
        for index in DM2_POSE_HARM_ROWS
        if dm4_by_index[index].get("cure_disposition") == "POSE_HARM_PRICED_NOT_CURED_WITHIN_FIXED_MENU"
    )
    if cured != DM4_EXACT_CURE_ROWS or uncured != DM4_UNCURED_ROWS:
        raise WS4PoseNullError(f"DM4 cure partition differs: cured={cured}, uncured={uncured}")

    required_join_keys = {
        "source_decision_path",
        "source_decision_sha256",
        "wseg_correction_id",
    }
    joinable_rows = [index for index in DM2_POSE_HARM_ROWS if required_join_keys.issubset(dm4_by_index[index].keys())]
    if joinable_rows:
        raise WS4PoseNullError("DM4 unexpectedly acquired W_seg join keys; explicit reviewed projection is required")

    rerank = ws1.get("seg_lexicographic_rerank")
    if not isinstance(rerank, Mapping):
        raise WS4PoseNullError("WS1 correction decision chain missing")
    strict = rerank.get("strict_accepted")
    prefix_count = rerank.get("receiver_recompile_status", {}).get("settled_v19b_prefix_move_count")
    if not isinstance(strict, list) or len(strict) != 96 or prefix_count != 10:
        raise WS4PoseNullError("WS1 accepted correction inventory differs from 96+10")
    pose_coupling_fields = {
        "d_pose",
        "delta_d_pose",
        "pose_nonharm",
        "delta_pair_pose_mse",
    }
    strict_with_pose = [
        row.get("candidate_id") for row in strict if isinstance(row, Mapping) and pose_coupling_fields.intersection(row)
    ]
    if strict_with_pose:
        raise WS4PoseNullError("WS1 strict decisions unexpectedly acquired per-move pose coupling")

    cc3_endpoint = cc3.get("endpoint")
    if not isinstance(cc3_endpoint, Mapping):
        raise WS4PoseNullError("CC3 endpoint missing")
    target_values = [
        float(value)
        for value in _recursive_values(j9_ticket, "target_d_pose")
        if value is not None and math.isfinite(float(value))
    ]
    sealed_target = 163.06116431842463
    if not any(math.isclose(value, sealed_target, rel_tol=0.0, abs_tol=1e-12) for value in target_values):
        raise WS4PoseNullError("J9 sealed stage-3 target_d_pose missing")
    cc3_pose_headroom = sealed_target - float(cc3_endpoint["d_pose"])
    if cc3_pose_headroom < 0.0:
        raise WS4PoseNullError("CC3 composition exceeds the sealed stage-3 pose target")

    ws3_registered = ws3_arbitration.get("registered_slope_verdict")
    if (
        not isinstance(ws3_registered, Mapping)
        or ws3_registered.get("reason") != "SEG_REGRESSION"
        or ws3_registered.get("decision") != "KEEP_WJOINT"
    ):
        raise WS4PoseNullError("WS3 registered falsifier verdict differs")

    return {
        "schema": "ddm_ws4_pose_coupling_classification.v1",
        "wseg_correction_inventory": {
            "settled_v19b_prefix_moves": 10,
            "strict_v19c_moves": 96,
            "total_base_moves": 106,
            "per_move_pose_coupling_available": False,
            "projection_disposition": "REFUSE_UNMEASURED_PER_MOVE_POSE_CLASSIFICATION",
        },
        "temporal_suffix": {
            "payload_bytes": int(temporal["archive_bytes"]) - int(base["archive_bytes"]),
            "delta_d_pose": suffix_pose_delta,
            "delta_d_seg": suffix_seg_delta,
            "pose_coupling": "POSE_BENEFICIAL",
            "projection_disposition": "PRESERVE",
        },
        "dm2_dm4_rows": {
            "pose_harm_row_indices": list(pose_harm),
            "dm4_exact_cure_row_indices": list(cured),
            "dm4_uncured_row_indices": list(uncured),
            "wseg_foreign_key_joinable_row_indices": joinable_rows,
            "source_instance": "G3_SEMANTIC_DEMAND_25_ROW_INSTANCE",
            "projection_disposition": "REFUSE_CROSS_INSTANCE_OVERLAY",
        },
        "cc3_composition": {
            "measured_d_pose": float(cc3_endpoint["d_pose"]),
            "sealed_stage3_target_d_pose": sealed_target,
            "pose_headroom": cc3_pose_headroom,
            "required_pose_projection": False,
        },
        "projection": {
            "projected_component_ids": [],
            "projected_component_count": 0,
            "materialization": "IDENTITY_WSEG_NO_LAWFUL_POSE_PUNISHED_COMPONENT",
            "dm4_projector_invoked": False,
        },
        "verdict": "EMPTY_LAWFUL_PROJECTION_SET_IDENTITY_WSEG_PERP",
        "verdict_scope": (
            "INSTANCE: sealed WS2 W_seg composition against the DM2/DM4 G3 row set; "
            "no negative on future projection with explicit per-correction PoseNet custody"
        ),
    }


def build_arbitration_receipt(
    *,
    ws3_arbitration: Mapping[str, Any],
    wseg_perp_custody: Mapping[str, Any],
    wjoint_step50_custody: Mapping[str, Any],
    inputs: Mapping[str, Any],
) -> dict[str, Any]:
    """Build a WS3-schema fail-fast arbitration receipt for identity W_seg⊥."""

    _require_schema(ws3_arbitration, ARBITRATION_SCHEMA, "WS3 arbitration")
    registered = ws3_arbitration.get("registered_slope_verdict")
    if not isinstance(registered, Mapping):
        raise WS4PoseNullError("WS3 registered slope verdict missing")
    if (
        ws3_arbitration.get("critical_ratio") != CRITICAL_RATIO
        or registered.get("decision") != "KEEP_WJOINT"
        or registered.get("reason") != "SEG_REGRESSION"
        or float(registered.get("seg_regression_per_step", 0.0)) <= 0.0
    ):
        raise WS4PoseNullError("WS3 fail-fast arbitration premise differs")
    if (
        wseg_perp_custody.get("sha256") != WSEG_N600["archive_sha256"]
        or wseg_perp_custody.get("bytes") != WSEG_N600["archive_bytes"]
    ):
        raise WS4PoseNullError("W_seg_perp is not byte-identical to settled W_seg")

    return {
        "schema": ARBITRATION_SCHEMA,
        "equation_id": "ddm_ws1_warm_start_slope_falsifier_v1",
        "critical_ratio": CRITICAL_RATIO,
        "evidence_axis": EVIDENCE_AXIS,
        "inputs": dict(inputs),
        "registered_slope_verdict": dict(registered),
        "window_deltas": {
            "W_seg_perp_terminal_proposal": dict(ws3_arbitration["window_deltas"]["W_seg_terminal_proposal"]),
            "W_joint_preserved_reference": dict(ws3_arbitration["window_deltas"]["W_joint"]),
        },
        "window_status": {
            "W_seg_perp": {
                "exact_steps": [0, 1],
                "accepted_steps": [0],
                "status": "PREREGISTERED_FAIL_FAST_FORMULATION_STOP_AT_EXACT_TERMINAL_PROPOSAL",
                "reuse_basis": "BYTE_IDENTICAL_START_SHA_AND_UNCHANGED_WS3_HARNESS",
            },
            "W_joint_step50_live": {
                "materialized": True,
                "parameter_shadow": "live_resume_state",
                "status": "PRESERVED_FALLBACK_PENDING_J10_SHADOW_CONSISTENT_N600_RESEAL",
            },
        },
        "candidate_custody": {
            "W_seg_perp": dict(wseg_perp_custody),
            "W_joint_step50_live": dict(wjoint_step50_custody),
        },
        "selected_warm_start": "W_joint_step50_live",
        "verdict": "ARBITRATED_FORMULATION_STOP_KEEP_WJOINT_STEP50_LIVE",
        "verdict_scope": (
            "FORMULATION: post-hoc pose-null projection of this sealed W_seg composition; "
            "identity start inherits WS3 first-proposal SEG_REGRESSION. "
            "W_joint step50 n600 reseal remains owned by J10."
        ),
        "score_claim": False,
        "promotion_eligible": False,
        "execution_allowed": False,
        "research_only": True,
        "pointer": POINTER,
        "pointer_moved": False,
        "main_review_required": True,
    }
