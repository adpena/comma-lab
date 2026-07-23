#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Probe B source closure and generated-column restricted-master entrypoint.

The pre-registered question requires all column families to share one exact-R
operating point.  This runner fails closed before pricing when the bound sources
cannot form that master.  A blocked receipt is evidence: it prevents the v12
legacy scorer-grid row, the G1 mask-only row, and the v15/v16 camera-R rows from
being silently mixed into a fictional equal-byte curve.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tac.optimization.direct_description_carrier_compose import (  # noqa: E402
    WORLDSHEET_G1_MEMBER,
    compile_carrier_compose_archive,
    parse_carrier_compose_archive,
    receive_carrier_compose_archive,
    rfc8785_canonicalize,
)
from tac.optimization.direct_description_minimizer import (  # noqa: E402
    DirectDescriptionError,
    _read_regular_file_once,
)

SCHEMA = "ddm_a1_column_generated_correction_receipt.v1"
LANE_ID = "ddm_v18_column_generation_vocabulary"
EVIDENCE_AXIS = "[macOS-CPU frozen-scorer advisory]"
POINTER = "0.1910828242 [contest-CPU]"
V12_DSEG = "0.034003668891"
V12_DPOSE = "163.034719422881"
V12_FINAL_BYTES = 106_106
FIXED_BUDGETS = (16_384, 49_152, 98_304, 147_456)
FIXED_FAMILIES = (
    "v12_control",
    "g1_grammar_coordinate",
    "realized_residual_vjp",
    "curvelet_boundary",
)
FIXED_SELECTORS = (
    "v12_sequential_greedy_control",
    "beam_width_32",
    "conflict_miqp",
)
FIXED_CODER_ENTRANTS = (
    "unstructured_explicit_indices",
    "structured_nm_2_of_4_coding_order",
    "mx_block_shared_scale_int4",
)
VERDICT = "BLOCKED_PRECONDITION_NO_COMMON_EXACT_R_MASTER"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _portable(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _bound_bytes(path: Path, digest: str, name: str) -> bytes:
    payload = _read_regular_file_once(path)
    if _sha256(payload) != digest:
        raise DirectDescriptionError(f"{name} SHA-256 differs")
    return payload


def _bound_json(path: Path, digest: str, name: str) -> dict[str, Any]:
    try:
        return json.loads(_bound_bytes(path, digest, name))
    except json.JSONDecodeError as exc:
        raise DirectDescriptionError(f"{name} is not valid JSON") from exc


class DDMA1ColumnGeneratedCorrectionConfigV1(BaseModel):
    """Fixed Probe B preregistration plus explicit v15/v16 custody."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    schema_: Literal["DDMA1ColumnGeneratedCorrectionConfigV1"] = Field(
        default="DDMA1ColumnGeneratedCorrectionConfigV1",
        alias="schema",
        serialization_alias="schema",
    )
    run_id: StrictStr = Field(min_length=8)
    seed: Literal[1234] = 1234
    source_v12_receipt: StrictStr
    source_v12_receipt_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    source_v12_archive: StrictStr
    source_v12_archive_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    grammar_receipt: StrictStr
    grammar_receipt_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    source_v15_receipt: StrictStr
    source_v15_receipt_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    source_v15_archive: StrictStr
    source_v15_archive_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    source_v16_receipt: StrictStr
    source_v16_receipt_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    pair_start: Literal[448] = 448
    pair_count: Literal[64] = 64
    column_families: tuple[
        Literal[
            "v12_control",
            "g1_grammar_coordinate",
            "realized_residual_vjp",
            "curvelet_boundary",
        ],
        ...,
    ] = FIXED_FAMILIES
    pricing_metric: Literal["realized_joint_objective_reduced_cost"] = (
        "realized_joint_objective_reduced_cost"
    )
    selection_modes: tuple[
        Literal["v12_sequential_greedy_control", "beam_width_32", "conflict_miqp"],
        ...,
    ] = FIXED_SELECTORS
    coder_entrants: tuple[
        Literal[
            "unstructured_explicit_indices",
            "structured_nm_2_of_4_coding_order",
            "mx_block_shared_scale_int4",
        ],
        ...,
    ] = FIXED_CODER_ENTRANTS
    coder_comparison_rule: Literal["matched_realized_d_seg_minimum_exact_bytes"] = (
        "matched_realized_d_seg_minimum_exact_bytes"
    )
    maximum_pricing_rounds: Literal[3] = 3
    maximum_new_columns_per_round: Literal[64] = 64
    added_byte_budgets: tuple[StrictInt, ...] = FIXED_BUDGETS
    exact_replay_after_each_selected_set: Literal[True] = True
    memory_ceiling_gib: Literal[116] = 116
    research_only: Literal[True] = True
    execution_allowed: Literal[False] = False
    score_claim: Literal[False] = False

    @model_validator(mode="after")
    def _fixed_preregistration(self) -> DDMA1ColumnGeneratedCorrectionConfigV1:
        if self.column_families != FIXED_FAMILIES:
            raise ValueError("Probe B column families are fixed by preregistration")
        if self.selection_modes != FIXED_SELECTORS:
            raise ValueError("Probe B selector controls are fixed by preregistration")
        if self.coder_entrants != FIXED_CODER_ENTRANTS:
            raise ValueError("Probe B coder entrants are fixed by operator amendment")
        if self.added_byte_budgets != FIXED_BUDGETS:
            raise ValueError("Probe B equal-byte rungs are fixed by preregistration")
        return self

    def typed_config_hash(self) -> str:
        return _sha256(rfc8785_canonicalize(self.model_dump(mode="json", by_alias=True)))


def _producer_paths() -> tuple[Path, ...]:
    return (
        Path(__file__),
        REPO_ROOT / "src/tac/optimization/ddm_column_generation.py",
        REPO_ROOT / "src/tac/canonical_equations/ddm_v18_column_pricing_law_20260723.py",
        REPO_ROOT / "src/tac/optimization/direct_description_carrier_compose.py",
    )


def _producer_custody() -> list[dict[str, Any]]:
    return [
        {
            "path": _portable(path),
            "bytes": path.stat().st_size,
            "sha256": _sha256(_read_regular_file_once(path)),
        }
        for path in _producer_paths()
    ]


def _revalidate_existing_receipt(
    receipt: Mapping[str, Any],
    config: DDMA1ColumnGeneratedCorrectionConfigV1,
) -> None:
    if receipt.get("schema") != SCHEMA or receipt.get("verdict") != VERDICT:
        raise DirectDescriptionError("existing Probe B receipt schema/verdict differs")
    if receipt.get("typed_config_sha256") != config.typed_config_hash():
        raise DirectDescriptionError("existing Probe B receipt typed config differs")
    bound_sources = (
        (config.source_v12_receipt, config.source_v12_receipt_sha256, "v12 receipt"),
        (config.source_v12_archive, config.source_v12_archive_sha256, "v12 archive"),
        (config.grammar_receipt, config.grammar_receipt_sha256, "G1 receipt"),
        (config.source_v15_receipt, config.source_v15_receipt_sha256, "v15 receipt"),
        (config.source_v15_archive, config.source_v15_archive_sha256, "v15 archive"),
        (config.source_v16_receipt, config.source_v16_receipt_sha256, "v16 receipt"),
    )
    for path, digest, name in bound_sources:
        _bound_bytes(Path(path), digest, name)
    if receipt.get("producer_custody") != _producer_custody():
        raise DirectDescriptionError(
            "existing Probe B receipt producer custody differs; choose a fresh output directory"
        )


def evaluate_source_closure(
    *,
    v12: Mapping[str, Any],
    grammar: Mapping[str, Any],
    v15: Mapping[str, Any],
    v16: Mapping[str, Any],
    v12_has_realization_profile: bool,
    hybrid_compile_error: str | None,
) -> tuple[dict[str, Any], ...]:
    """Return exact blockers; an empty tuple is the only pricing authorization."""

    blockers: list[dict[str, Any]] = []
    v12_axis = v12.get("evidence_axis")
    if (
        v12.get("schema") != "direct_description_v12_obligation_drain_receipt.v1"
        or v12.get("score_claim") is not False
        or v12.get("pointer_moved") is not False
    ):
        blockers.append(
            {
                "code": "V12_CONTROL_CUSTODY_MISMATCH",
                "evidence": "bound v12 receipt schema/authority fields differ",
            }
        )
    if v12_axis != EVIDENCE_AXIS or v12_has_realization_profile:
        blockers.append(
            {
                "code": "V12_CONTROL_PATH_UNEXPECTED",
                "evidence": {
                    "evidence_axis": v12_axis,
                    "realization_profile_present": v12_has_realization_profile,
                },
            }
        )
    if not v12_has_realization_profile:
        blockers.append(
            {
                "code": "V12_CONTROL_NOT_CAMERA_UINT8_R",
                "evidence": (
                    "CarrierComposeReceiverV1.render_pairs is the V9-V13 legacy scorer-grid path; "
                    "the bound v12 archive has no realization_profile and render_camera_pairs refuses it"
                ),
            }
        )
    coverage = grammar.get("coverage_projection", {})
    if (
        grammar.get("schema") != "ddm_g1_grammar_induction_compact_receipt.v1"
        or grammar.get("candidate_archive") is not False
        or coverage.get("receiver_closed") is not False
        or coverage.get("pose_measured") is not False
    ):
        blockers.append(
            {
                "code": "G1_RECEIPT_CUSTODY_MISMATCH",
                "evidence": "G1 receipt no longer matches preregistered mask-only source",
            }
        )
    else:
        blockers.append(
            {
                "code": "G1_COORDINATES_NOT_RECEIVER_CLOSED",
                "evidence": {
                    "candidate_archive": False,
                    "receiver_closed": False,
                    "pose_measured": False,
                    "verdict_scope": grammar.get("verdict_scope"),
                },
            }
        )
    v15_control = v15.get("inherited_control", {})
    if (
        v15.get("schema") != "ddm_v15_scorer_solved_template_receipt.v1"
        or v15.get("score_claim") is not False
    ):
        blockers.append(
            {
                "code": "V15_TEMPLATE_CUSTODY_MISMATCH",
                "evidence": "bound v15 receipt schema/authority fields differ",
            }
        )
    elif str(v15_control.get("d_seg")) != "0.027470296224":
        blockers.append(
            {
                "code": "V15_OPERATING_POINT_DRIFT",
                "evidence": {"inherited_control_d_seg": v15_control.get("d_seg")},
            }
        )
    else:
        blockers.append(
            {
                "code": "V15_TEMPLATE_DOF_NOT_MEASURED_AT_V12_OPERATING_POINT",
                "evidence": {
                    "v15_inherited_control_d_seg": v15_control.get("d_seg"),
                    "v12_control_d_seg": V12_DSEG,
                },
            }
        )
    conditionals = v16.get("conditionals", {})
    if v16.get("schema") != "ddm_v16_coupled_joint_solve_receipt.v1":
        blockers.append(
            {
                "code": "V16_OPERATOR_CUSTODY_MISMATCH",
                "evidence": "bound v16 receipt schema differs",
            }
        )
    elif conditionals.get("linearization_invalid") is not True:
        blockers.append(
            {
                "code": "V16_EXPECTED_INVALIDATION_MISSING",
                "evidence": {"linearization_invalid": conditionals.get("linearization_invalid")},
            }
        )
    else:
        blockers.append(
            {
                "code": "V16_LINEARIZATION_INVALID_AT_SOURCE",
                "evidence": {
                    "linearization_invalid": True,
                    "fork": v16.get("fork"),
                },
            }
        )
    if hybrid_compile_error is not None:
        blockers.append(
            {
                "code": "NO_COMMON_HYBRID_ARCHIVE_SCHEMA",
                "evidence": hybrid_compile_error,
            }
        )
    return tuple(blockers)


def _hybrid_compile_probe(v12_archive: bytes, v15_archive: bytes) -> dict[str, Any]:
    """Attempt the exact family composition without weakening compiler guards."""

    v12_members, _ = parse_carrier_compose_archive(v12_archive)
    v15_members, _ = parse_carrier_compose_archive(v15_archive)
    v12_receiver = receive_carrier_compose_archive(v12_archive)
    v15_receiver = receive_carrier_compose_archive(v15_archive)
    g1 = v15_members.get(WORLDSHEET_G1_MEMBER, b"")
    if not g1 or v15_receiver.realization_profile is None or v15_receiver.scorer_solved_templates is None:
        return {
            "attempted": False,
            "error": "v15 source lacks G1/realization/template prerequisites",
        }
    try:
        archive, _ = compile_carrier_compose_archive(
            v12_members["predictor.zip"],
            v12_receiver.symbols,
            boundary_shearlets=v12_receiver.boundary_shearlets,
            island_shapes=v12_receiver.island_shapes,
            obligation_vocabulary=True,
            worldsheet_g1_payload=g1,
            realization_profile=v15_receiver.realization_profile,
            scorer_solved_templates=v15_receiver.scorer_solved_templates,
        )
    except DirectDescriptionError as exc:
        return {"attempted": True, "error": str(exc)}
    return {
        "attempted": True,
        "error": None,
        "archive_bytes": len(archive),
        "archive_sha256": _sha256(archive),
    }


def _backend_preflight(seed: int) -> dict[str, Any]:
    """Exercise the mandated MLX Metal fused-R and grouped-backward surfaces."""

    try:
        import mlx.core as mx

        from tac.local_acceleration.metal_fused_r_operator import (
            _fused_r_metal_vjp,
            fused_r_vjp_numpy,
        )
        from tac.local_acceleration.mlx_scorer_adapters import (
            _custom_metal_backward_status,
        )

        mx.set_default_device(mx.gpu)
        rng = np.random.default_rng(seed)
        x = rng.uniform(-10.0, 265.0, size=(1, 12, 16, 3)).astype(np.float32)
        gradient = rng.standard_normal((1, 12, 16, 3)).astype(np.float32)
        numpy_vjp = fused_r_vjp_numpy(
            x,
            gradient,
            camera_hw=(27, 35),
            output_hw=(12, 16),
            ste_round=True,
        )
        metal = _fused_r_metal_vjp(
            mx.array(x),
            mx.array(gradient),
            camera_hw=(27, 35),
            output_hw=(12, 16),
            ste_round=True,
        )
        mx.eval(metal)
        metal_vjp = np.asarray(metal, dtype=np.float32)
        grouped_active, grouped_reason = _custom_metal_backward_status()
        identical = bool(np.array_equal(numpy_vjp, metal_vjp))
        return {
            "mlx_gpu_available": True,
            "device": str(mx.default_device()),
            "fused_r_vjp_bit_identical_numpy_fp32": identical,
            "fused_r_vjp_max_abs_delta": float(np.max(np.abs(numpy_vjp - metal_vjp))),
            "custom_grouped_backward_active": grouped_active,
            "custom_grouped_backward_reason": grouped_reason,
            "pricing_authorized": identical and grouped_active,
            "score_authority": False,
        }
    except Exception as exc:
        return {
            "mlx_gpu_available": False,
            "pricing_authorized": False,
            "blocker": f"{type(exc).__name__}: {exc}",
            "score_authority": False,
        }


def _blocked_equal_byte_rows() -> list[dict[str, Any]]:
    return [
        {
            "added_byte_budget": budget,
            "v12_control": {
                "d_seg": V12_DSEG,
                "d_pose": V12_DPOSE,
                "archive_bytes": V12_FINAL_BYTES,
                "measurement_path": "legacy scorer-grid; reference-only, not exact-R Probe B evidence",
            },
            "generated_vocabulary": {
                "d_seg": None,
                "d_pose": None,
                "archive_bytes": None,
                "status": "NOT_MEASURED_PRECONDITION_BLOCKED",
            },
            "exact_replay_complete": False,
            "global_selector": None,
            "beats_v12": None,
        }
        for budget in FIXED_BUDGETS
    ]


def _pricing_history() -> list[dict[str, Any]]:
    return [
        {
            "round": index,
            "complete": False,
            "exact_pricing": False,
            "generated_column_count": 0,
            "negative_reduced_cost_count": None,
            "status": "NOT_RUN_PRECONDITION_BLOCKED",
        }
        for index in range(1, 4)
    ]


def _storage_preflight(root: Path, memory_ceiling_gib: int) -> dict[str, Any]:
    usage = shutil.disk_usage(root.parent if root.parent.exists() else REPO_ROOT)
    return {
        "output_path": _portable(root),
        "output_tier": "local_small_manifest_only",
        "free_bytes": usage.free,
        "memory_ceiling_gib": memory_ceiling_gib,
        "large_artifacts_created": False,
        "ssd_preferred_for_future_scorer_caches": "/Volumes/VertigoDataTier/pact",
        "certify_or_block": True,
    }


def _publish_receipt(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = rfc8785_canonicalize(dict(payload))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def run(
    config: DDMA1ColumnGeneratedCorrectionConfigV1,
    output_directory: Path,
    semantic_argv: Sequence[str],
) -> Path:
    root = output_directory.resolve()
    receipt_path = root / "ddm_a1_column_generated_correction_receipt.json"
    if receipt_path.exists():
        receipt = json.loads(_read_regular_file_once(receipt_path))
        _revalidate_existing_receipt(receipt, config)
        print(
            json.dumps(
                {
                    "receipt": str(receipt_path),
                    "verdict": receipt["verdict"],
                    "resumed": True,
                }
            )
        )
        return receipt_path

    storage = _storage_preflight(root, config.memory_ceiling_gib)
    v12 = _bound_json(Path(config.source_v12_receipt), config.source_v12_receipt_sha256, "v12 receipt")
    v12_archive = _bound_bytes(
        Path(config.source_v12_archive), config.source_v12_archive_sha256, "v12 archive"
    )
    grammar = _bound_json(Path(config.grammar_receipt), config.grammar_receipt_sha256, "G1 receipt")
    v15 = _bound_json(Path(config.source_v15_receipt), config.source_v15_receipt_sha256, "v15 receipt")
    v15_archive = _bound_bytes(
        Path(config.source_v15_archive), config.source_v15_archive_sha256, "v15 archive"
    )
    v16 = _bound_json(Path(config.source_v16_receipt), config.source_v16_receipt_sha256, "v16 receipt")
    v12_receiver = receive_carrier_compose_archive(v12_archive)
    hybrid = _hybrid_compile_probe(v12_archive, v15_archive)
    blockers = evaluate_source_closure(
        v12=v12,
        grammar=grammar,
        v15=v15,
        v16=v16,
        v12_has_realization_profile=v12_receiver.realization_profile is not None,
        hybrid_compile_error=hybrid.get("error"),
    )
    backend = _backend_preflight(config.seed)
    if not backend.get("pricing_authorized"):
        blockers = (
            *blockers,
            {
                "code": "COMPUTE_MANDATE_PREFLIGHT_FAILED",
                "evidence": backend,
            },
        )
    if not blockers:
        raise DirectDescriptionError(
            "Probe B common exact-R source closure is now available; this blocker-only "
            "revision must be superseded by the measured three-round executor"
        )
    pricing = _pricing_history()
    equal_byte = _blocked_equal_byte_rows()
    receipt = {
        "schema": SCHEMA,
        "lane_id": LANE_ID,
        "run_id": config.run_id,
        "typed_config": config.model_dump(mode="json", by_alias=True),
        "typed_config_sha256": config.typed_config_hash(),
        "semantic_argv": list(semantic_argv),
        "producer_custody": _producer_custody(),
        "source_custody": {
            "v12_receipt": {
                "path": config.source_v12_receipt,
                "sha256": config.source_v12_receipt_sha256,
            },
            "v12_archive": {
                "path": config.source_v12_archive,
                "bytes": len(v12_archive),
                "sha256": config.source_v12_archive_sha256,
            },
            "g1_receipt": {
                "path": config.grammar_receipt,
                "sha256": config.grammar_receipt_sha256,
            },
            "v15_receipt": {
                "path": config.source_v15_receipt,
                "sha256": config.source_v15_receipt_sha256,
            },
            "v15_archive": {
                "path": config.source_v15_archive,
                "bytes": len(v15_archive),
                "sha256": config.source_v15_archive_sha256,
            },
            "v16_receipt": {
                "path": config.source_v16_receipt,
                "sha256": config.source_v16_receipt_sha256,
            },
        },
        "source_closure": {
            "authorized": False,
            "blockers": list(blockers),
            "hybrid_compile_probe": hybrid,
            "required_common_path": (
                "one receiver schema that composes V11 corrections + G1 natural productions + "
                "V15/V16 template DOF, then measures both v12 control and selected sets through "
                "camera paint -> uint8 -> evaluator R -> frozen scorers"
            ),
            "reformulation_queue": [
                {
                    "priority": 1,
                    "action": (
                        "land a reviewed hybrid archive schema for V11 corrections, G1 natural "
                        "productions, and V15/V16 template DOF without weakening parse-back"
                    ),
                },
                {
                    "priority": 2,
                    "action": (
                        "remeasure the v12 control and every proposed column set from the same "
                        "camera-resolution realization through uint8, evaluator R, and scorers"
                    ),
                },
                {
                    "priority": 3,
                    "action": (
                        "race unstructured indices against 2-of-4 coding-order support metadata "
                        "at matched realized d_seg; include shared-scale int4 MX blocks whenever "
                        "a floating-point payload exists"
                    ),
                    "coder_entrants": list(config.coder_entrants),
                    "status": "QUEUED_NOT_MEASURED_PRECONDITION_BLOCKED",
                },
            ],
        },
        "compute_preflight": backend,
        "pricing_round_history": pricing,
        "equal_byte_rows": equal_byte,
        "falsifier": {
            "condition": (
                "three complete exact pricing rounds have no negative reduced-cost column AND "
                "global exact replay has no equal-byte v12 beat"
            ),
            "eligible": False,
            "triggered": False,
            "reason": "source closure failed before round 1; incomplete rounds cannot close the formulation",
            "verdict_scope": "FORMULATION:COLUMN_FAMILIES_AND_THREE_ROUND_GLOBAL_SELECTION remains open",
        },
        "verdict": VERDICT,
        "verdict_scope": (
            "PRECONDITION only: current sources do not share one exact-R master; no correction, "
            "grammar, direct-description, or generated-vocabulary family verdict"
        ),
        "triality": {
            "dsl": "DDMA1ColumnGeneratedCorrectionConfigV1",
            "dag": ".omx/research/ddm_v18_column_generation_vocabulary_DAG_FEED_20260723.md",
            "equations": "tac.canonical_equations.ddm_v18_column_pricing_law_20260723",
        },
        "storage_preflight": storage,
        "resume": {
            "source_closure_checkpoint": True,
            "pricing_rounds_preserved": 0,
            "safe_resume_boundary": "round_00_source_closure",
            "all_preserved": True,
        },
        "stores_consulted": [
            "CLAUDE.md",
            "AGENTS.md",
            "docs/operating_manual_craft_handoff.md",
            ".omx/research/ddm_a1_naive_verdict_audit_20260723_codex.md",
            config.source_v12_receipt,
            config.source_v12_archive,
            config.grammar_receipt,
            config.source_v15_receipt,
            config.source_v15_archive,
            config.source_v16_receipt,
            "src/tac/through_r/flip_inverse.py",
            "src/tac/local_acceleration/metal_fused_r_operator.py",
            ".omx/state/lane_registry.json",
        ],
        "host": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
        },
        "pointer": POINTER,
        "pointer_moved": False,
        "evidence_axis": EVIDENCE_AXIS,
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
        "d_seg_claim": False,
        "d_pose_claim": False,
        "promotion_eligible": False,
        "main_landing_review_required": True,
    }
    _publish_receipt(receipt_path, receipt)
    print(json.dumps({"receipt": str(receipt_path), "verdict": VERDICT, "resumed": False}))
    return receipt_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    config = DDMA1ColumnGeneratedCorrectionConfigV1.model_validate_json(
        _read_regular_file_once(args.config)
    )
    semantic_argv = [
        "tools/probe_ddm_a1_column_generated_correction.py",
        "--config",
        _portable(args.config),
        "--output-directory",
        _portable(args.output_directory),
    ]
    run(config, args.output_directory, semantic_argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
