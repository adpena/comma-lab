# SPDX-License-Identifier: MIT
"""Fail-closed eligibility check for carrying Task #602 member outputs.

This module does not synthesize a replacement member or archive.  It audits the
preserved Task #602 interface and emits a scoped blocker when the output lacks
the bytes and numeric-domain transition required by the Task #603/#613
same-artifact receiver.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

from tac.optimization.direct_description_minimizer import (
    POINTER_SCORE_TEXT,
    DirectDescriptionError,
    _publish_new_bytes,
    _read_regular_file_once,
    _require_sha256,
    _sha256,
    rfc8785_canonicalize,
)

CONFIG_SCHEMA: Final = "MdlMemberCarrierPreflightConfigV1"
RESULT_SCHEMA: Final = "mdl_member_carrier_preflight.v1"
RUN_ID: Final = "ddm_mdl_member_carrier_n600_20260722"
REQUIRED_PAIRS: Final = 600
BOX_BAR_BYTES_PER_PAIR: Final = 440
KNEE_BYTES: Final = 216 * 1024
STAGE_SCHEMA: Final = "mdl_polytope_member_pair_stage.v2"
SOURCE_RECEIPT_SCHEMA: Final = "mdl_polytope_member_measurement.v1"
TARGET_RECEIPT_SCHEMA: Final = "direct_description_full_precision_target_planes.v1"


def _json_object(payload: bytes, label: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DirectDescriptionError(f"{label} is not valid JSON") from exc
    if not isinstance(value, dict):
        raise DirectDescriptionError(f"{label} must be a JSON object")
    return value


def _read_sha_bound(path: Path, expected_sha256: str, label: str) -> tuple[bytes, dict[str, Any]]:
    payload = _read_regular_file_once(path)
    if _sha256(payload) != _require_sha256(expected_sha256, f"{label}_sha256"):
        raise DirectDescriptionError(f"{label} SHA-256 drift")
    return payload, _json_object(payload, label)


class MdlMemberCarrierPreflightConfigV1(BaseModel):
    """Typed, read-only custody for one Task #602 carrier audit."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, populate_by_name=True)

    schema_: Literal["MdlMemberCarrierPreflightConfigV1"] = Field(
        default=CONFIG_SCHEMA, alias="schema", serialization_alias="schema"
    )
    run_id: Literal["ddm_mdl_member_carrier_n600_20260722"] = RUN_ID
    required_pairs: Literal[600] = REQUIRED_PAIRS
    compact_receipt_path: StrictStr
    compact_receipt_sha256: StrictStr
    full_receipt_path: StrictStr
    full_receipt_sha256: StrictStr
    stage_root: StrictStr
    target_receipt_path: StrictStr
    target_receipt_sha256: StrictStr
    solver_source_path: StrictStr
    solver_source_sha256: StrictStr
    producer_tool_path: StrictStr
    producer_tool_sha256: StrictStr
    scorer_batch_size: Literal[16] = 16
    max_runtime_seconds: StrictInt = Field(default=600, ge=1, le=600)
    research_only: Literal[True] = True
    execution_allowed: Literal[False] = False
    score_claim: Literal[False] = False

    @model_validator(mode="after")
    def _valid(self) -> MdlMemberCarrierPreflightConfigV1:
        for field in (
            "compact_receipt_sha256",
            "full_receipt_sha256",
            "target_receipt_sha256",
            "solver_source_sha256",
            "producer_tool_sha256",
        ):
            _require_sha256(getattr(self, field), field)
        if not Path(self.full_receipt_path).is_absolute() or not Path(self.stage_root).is_absolute():
            raise ValueError("Task #602 full receipt and stage root require absolute custody paths")
        return self

    def typed_config_hash(self) -> str:
        return _sha256(rfc8785_canonicalize(self.model_dump(mode="json", by_alias=True)))

    def dsl_compile_hash(self) -> str:
        return _sha256(
            rfc8785_canonicalize(
                {
                    "compile_target": RESULT_SCHEMA,
                    "typed_config": self.model_dump(mode="json", by_alias=True),
                }
            )
        )


class MdlMemberCarrierPreflightProgramV1(BaseModel):
    """Only governed argv accepted by the thin consumer."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    config_path: StrictStr
    output_path: StrictStr

    def compile_consumer_argv(self) -> tuple[str, ...]:
        return (
            "/usr/bin/env",
            "python3",
            "tools/check_mdl_member_carrier_preflight.py",
            "--config",
            self.config_path,
            "--output",
            self.output_path,
            "--execution-allowed",
            "false",
        )


def _load_stages(stage_root: Path, config_sha256: str) -> list[dict[str, Any]]:
    paths = sorted(stage_root.glob("pair_*.json"))
    if not paths:
        raise DirectDescriptionError("Task #602 has no preserved pair stages")
    rows: list[dict[str, Any]] = []
    for expected_pair, path in enumerate(paths):
        row = _json_object(_read_regular_file_once(path), f"Task #602 stage {path.name}")
        if (
            row.get("schema") != STAGE_SCHEMA
            or row.get("pair_index") != expected_pair
            or row.get("config_sha256") != config_sha256
        ):
            raise DirectDescriptionError(f"Task #602 stage identity drift: {path}")
        if not isinstance(row.get("selected_equals_canonical"), bool):
            raise DirectDescriptionError(f"Task #602 stage lacks selection disposition: {path}")
        if not isinstance(row.get("changed_values"), int) or row["changed_values"] < 0:
            raise DirectDescriptionError(f"Task #602 stage changed-value count is invalid: {path}")
        payload = row.get("selected_frame_payload")
        if payload is not None and not isinstance(payload, dict):
            raise DirectDescriptionError(f"Task #602 stage payload custody is malformed: {path}")
        rows.append(row)
    return rows


def _payload_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected_canonical = sum(bool(row["selected_equals_canonical"]) for row in rows)
    payload_rows = sum(isinstance(row.get("selected_frame_payload"), dict) for row in rows)
    changed_values = sum(int(row["changed_values"]) for row in rows)
    return {
        "preserved_pair_stages": len(rows),
        "selected_equals_canonical_pairs": selected_canonical,
        "noncanonical_selected_pairs": len(rows) - selected_canonical,
        "selected_frame_payload_rows": payload_rows,
        "changed_values": changed_values,
        "payload_semantics": (
            "selected_frame_payload is written only for a noncanonical selection; canonical selections "
            "refer back to the read-only 3.66 GB source raw and do not preserve a coded member description"
        ),
    }


def audit_mdl_member_carrier(config: MdlMemberCarrierPreflightConfigV1) -> dict[str, Any]:
    """Audit exact Task #602 bytes without constructing a substitute carrier."""

    started = time.monotonic()
    repo_root = Path(__file__).resolve().parents[3]
    preflight_module_path = repo_root / "src/tac/optimization/mdl_member_carrier_preflight.py"
    preflight_cli_path = repo_root / "tools/check_mdl_member_carrier_preflight.py"
    preflight_module_bytes = _read_regular_file_once(preflight_module_path)
    preflight_cli_bytes = _read_regular_file_once(preflight_cli_path)
    compact_bytes, compact = _read_sha_bound(
        Path(config.compact_receipt_path), config.compact_receipt_sha256, "compact_receipt"
    )
    full_bytes, full = _read_sha_bound(Path(config.full_receipt_path), config.full_receipt_sha256, "full_receipt")
    target_bytes, target = _read_sha_bound(
        Path(config.target_receipt_path), config.target_receipt_sha256, "target_receipt"
    )
    solver_bytes = _read_regular_file_once(Path(config.solver_source_path))
    producer_bytes = _read_regular_file_once(Path(config.producer_tool_path))
    if _sha256(solver_bytes) != config.solver_source_sha256:
        raise DirectDescriptionError("Task #602 solver source SHA-256 drift")
    if _sha256(producer_bytes) != config.producer_tool_sha256:
        raise DirectDescriptionError("Task #602 producer source SHA-256 drift")
    if full.get("schema") != SOURCE_RECEIPT_SCHEMA or target.get("schema") != TARGET_RECEIPT_SCHEMA:
        raise DirectDescriptionError("Task #602 or target receipt schema drift")
    if compact.get("schema") != "mdl_polytope_member_solve_compact_receipt.v1":
        raise DirectDescriptionError("Task #602 compact receipt schema drift")
    stage_root = Path(config.stage_root)
    if str(stage_root) != full.get("runtime", {}).get("stage_root"):
        raise DirectDescriptionError("Task #602 stage root differs from full receipt")
    rows = _load_stages(stage_root, str(full.get("config_sha256", "")))
    completed_prefix = int(full.get("completed_prefix", -1))
    payload = _payload_summary(rows)
    if time.monotonic() - started > config.max_runtime_seconds:
        raise DirectDescriptionError("Task #602 carrier preflight exceeded its typed runtime bound")
    d2 = full.get("D2_exact_member_selection", {})
    d3 = full.get("D3_same_coder_comparison", {})
    d4 = full.get("D4_n600_estimate_and_rate_feed", {})
    if (
        completed_prefix != len(rows)
        or int(d2.get("selected_equals_canonical_pairs", -1)) != payload["selected_equals_canonical_pairs"]
        or int(d2.get("integer_resize_exact_pairs", -1)) != len(rows)
    ):
        raise DirectDescriptionError("Task #602 stage/receipt coverage disagreement")
    selected_zlib = int(d3.get("selected_member_zlib9_bytes", -1))
    canonical_zlib = int(d3.get("canonical_member_zlib9_bytes", -1))
    if selected_zlib <= 0 or selected_zlib != canonical_zlib:
        raise DirectDescriptionError("Task #602 diagnostic coder disposition drift")
    if d4.get("n600_estimate") is not None or d4.get("activated") is not False:
        raise DirectDescriptionError("Task #602 n600 gate disposition drift")
    seed_scope = str(d3.get("seed_coder", {}).get("scope", ""))
    if "raw-member zlib is diagnostic" not in seed_scope:
        raise DirectDescriptionError("Task #602 diagnostic-versus-counted coder scope is missing")
    if target.get("plane_dtype") != "uint8":
        raise DirectDescriptionError("target receipt numeric-domain custody drift")

    bytes_per_pair = selected_zlib / completed_prefix
    gates = {
        "n600_member_solve_coverage": {
            "passed": completed_prefix == config.required_pairs,
            "observed_pairs": completed_prefix,
            "required_pairs": config.required_pairs,
        },
        "receiver_carriable_coded_member_payload": {
            "passed": False,
            "preserved_selected_frame_payload_rows": payload["selected_frame_payload_rows"],
            "reason": (
                "Task #602 declares no receiver member codec or archive section; canonical selections "
                "have no independent payload, and a changed selection would preserve diagnostic NPZ frames"
            ),
        },
        "counted_archive_mdl_inside_solve": {
            "passed": False,
            "reason": seed_scope,
        },
        "pre_uint8_member_state": {
            "passed": False,
            "reason": (
                "Task #602 canonical/selected arrays and the target receipt are already uint8; no preserved "
                "pre-quantization member exists from which to measure #532 realization loss"
            ),
            "target_plane_dtype": target["plane_dtype"],
        },
        "pose_stream_in_member_payload": {
            "passed": False,
            "reason": "Task #602 stages contain scorer comparisons, not a receiver-consumed Pose payload section",
        },
    }
    failed = [name for name, gate in gates.items() if not bool(gate["passed"])]
    if not failed:
        raise DirectDescriptionError("preflight unexpectedly found a complete Task #602 carrier")
    return {
        "schema": RESULT_SCHEMA,
        "run_id": config.run_id,
        "task": 603,
        "feeds_task": 613,
        "master_task": 578,
        "source_task": 602,
        "verdict": "BLOCKED_602_OUTPUT_IS_NOT_A_RECEIVER_CARRIER",
        "verdict_scope": (
            "FORMULATION_OUTPUT_INTERFACE: preserved Task #602 n64 bounded uint8 member search; "
            "does not close the MDL-member family or the direct-description family"
        ),
        "research_only": True,
        "execution_allowed": False,
        "candidate_archive": False,
        "score_claim": False,
        "pointer": f"{POINTER_SCORE_TEXT} [contest-CPU]",
        "pointer_moved": False,
        "typed_config": config.model_dump(mode="json", by_alias=True),
        "typed_config_sha256": config.typed_config_hash(),
        "dsl_compile_hash": config.dsl_compile_hash(),
        "preflight_producer": {
            "module": {
                "path": "src/tac/optimization/mdl_member_carrier_preflight.py",
                "bytes": len(preflight_module_bytes),
                "sha256": _sha256(preflight_module_bytes),
            },
            "cli": {
                "path": "tools/check_mdl_member_carrier_preflight.py",
                "bytes": len(preflight_cli_bytes),
                "sha256": _sha256(preflight_cli_bytes),
            },
        },
        "source_custody": {
            "compact_receipt": {
                "path": config.compact_receipt_path,
                "bytes": len(compact_bytes),
                "sha256": config.compact_receipt_sha256,
            },
            "full_receipt": {
                "path": config.full_receipt_path,
                "bytes": len(full_bytes),
                "sha256": config.full_receipt_sha256,
            },
            "target_receipt": {
                "path": config.target_receipt_path,
                "bytes": len(target_bytes),
                "sha256": config.target_receipt_sha256,
            },
            "solver_source": {
                "path": config.solver_source_path,
                "bytes": len(solver_bytes),
                "sha256": config.solver_source_sha256,
            },
            "producer_tool": {
                "path": config.producer_tool_path,
                "bytes": len(producer_bytes),
                "sha256": config.producer_tool_sha256,
            },
            "stage_root": config.stage_root,
            "source_bulk_mutated": False,
        },
        "source_output": payload,
        "bounded_preflight": {
            "max_runtime_seconds": config.max_runtime_seconds,
            "scorer_invoked": False,
            "scorer_batch_size_if_eligible": config.scorer_batch_size,
        },
        "eligibility_gates": gates,
        "failed_gates": failed,
        "curve": [],
        "curve_disposition": (
            "No (archive_bytes, membership) row exists: Task #602 produced no receiver-valid archive "
            "member payload, and substituting source raw or the Task #603 smooth chart would change the formulation."
        ),
        "non_curve_diagnostic": {
            "pairs": completed_prefix,
            "selected_member_zlib9_bytes": selected_zlib,
            "bytes_per_pair": f"{bytes_per_pair:.6f}",
            "box_bar_bytes_per_pair": BOX_BAR_BYTES_PER_PAIR,
            "multiple_of_box_bar": f"{bytes_per_pair / BOX_BAR_BYTES_PER_PAIR:.6f}",
            "knee_bytes": KNEE_BYTES,
            "multiple_of_216kib_knee": f"{selected_zlib / KNEE_BYTES:.6f}",
            "membership": None,
            "pose_stream_bytes": None,
            "pose_stream_completeness": None,
            "registerable_curve_row": False,
            "reason": "zlib-9 over decoded uint8 camera members is explicitly diagnostic, not len(A(z))",
        },
        "no_toy_substitution": {
            "source_raw_as_payload": "forbidden: source input is not a Task #602 coded-member description",
            "task603_smooth_chart_as_payload": "forbidden: decisive constant-classifier-equivalent formulation",
            "project_n64_to_n600": "forbidden: Task #602 receipt explicitly records n600_estimate=null",
        },
        "blocker_delta": {
            "member_carrier_point": "UNMEASURED -> BLOCKED_WITH_EXACT_INTERFACE_CLASSIFICATION",
            "n600_same_artifact_apparatus": "GREEN_UNCHANGED",
            "primary_register_green_rows": "8/19 UNCHANGED",
        },
        "required_successor": (
            "A solver must persist a decoder-consumed coded member payload and an independent pre-uint8 "
            "state, price exact final archive bytes inside admission, carry Pose in the same artifact, and "
            "then rerun frozen-SegNet batch16 membership through uint8/R."
        ),
        "main_landing_review_required": True,
    }


def write_mdl_member_carrier_preflight(
    config: MdlMemberCarrierPreflightConfigV1,
    output_path: Path,
) -> tuple[dict[str, Any], Path]:
    result = audit_mdl_member_carrier(config)
    payload = rfc8785_canonicalize(result) + b"\n"
    return result, _publish_new_bytes(Path(output_path), payload)


__all__ = [
    "BOX_BAR_BYTES_PER_PAIR",
    "CONFIG_SCHEMA",
    "KNEE_BYTES",
    "RESULT_SCHEMA",
    "MdlMemberCarrierPreflightConfigV1",
    "MdlMemberCarrierPreflightProgramV1",
    "audit_mdl_member_carrier",
    "write_mdl_member_carrier_preflight",
]
