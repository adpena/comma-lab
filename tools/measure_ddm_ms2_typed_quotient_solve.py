#!/usr/bin/env python3
"""Emit the fail-closed DDM MS2 custody receipt.

The current repository snapshot lacks the measured scorer geometry and landed
PF2 atlas needed to construct an admissible n600 candidate.  This command
therefore performs a source/custody preflight and writes an exact blocker
receipt.  It never substitutes identity geometry, never calls a scorer, and
never treats a MAIN handoff absent from this isolated base as locally rehashed
solver input.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Final

REPO: Final = Path(__file__).resolve().parents[1]
SRC: Final = REPO / "src"
for local_path in (str(SRC), str(REPO)):
    if local_path not in sys.path:
        sys.path.insert(0, local_path)

from tac.canonical_equations.ddm_ms2_typed_quotient_solve_20260724 import (  # noqa: E402
    EQUATION_IDS,
)
from tac.optimization.ddm_min_description_contract import (  # noqa: E402
    build_minimum_description_headline,
)
from tac.optimization.ddm_typed_quotient_solve import (  # noqa: E402
    CLASS_PAIRS,
    STREAM_TYPE_CONTRACT_MODULE,
    EvaluationRecursionLevel,
)

SCHEMA: Final = "ddm_ms2_typed_quotient_solve_repo_receipt.v1"
RUN_ID: Final = "ddm_ms2_typed_quotient_solve_20260724T022221Z"
LANE_ID: Final = "lane_ddm_ms2_typed_quotient_solve_20260724"
AXIS: Final = "[macOS-CPU frozen-scorer advisory]"
POINTER: Final = "0.1910828242 [contest-CPU]"
DEFAULT_OUTPUT: Final = REPO / ".omx/research/ddm_ms2_typed_quotient_solve_20260724_receipt.json"
MS1_RECEIPT: Final = ".omx/research/ddm_ms1_min_description_lattice_solve_20260724_receipt.json"
G3_RECEIPT: Final = ".omx/research/ddm_g3_score_atlas_n600_20260722T204000Z/ddm_g3_score_atlas_receipt.json"
G4_RECEIPT: Final = (
    ".omx/research/ddm_g4_spatial_stationarity_n600_20260722T212138Z/ddm_g4_spatial_stationarity_receipt.json"
)
DR2B_RECEIPT: Final = ".omx/research/ddm_dr2b_tolerance_ladder_and_costate_rows_n600_20260723_v4/receipt.json"
INNER_JACOBIAN_STATUS: Final = ".omx/research/realization_g2d_predict_base_projection_blocker_20260721.json"
V16_EQUATIONS: Final = ".omx/research/ddm_v16_coupled_joint_solve_equations_20260723.json"
V16_RECEIPT: Final = (
    ".omx/research/ddm_v16_coupled_joint_solve_20260723T002500Z/ddm_v16_coupled_joint_solve_receipt.json"
)
V16_POSE_TUBE_RECEIPT: Final = (
    ".omx/research/ddm_v16_coupled_joint_solve_lane_fix_20260723T013500Z/ddm_v16_coupled_joint_solve_receipt.json"
)
AT1X_RECEIPT: Final = ".omx/research/ddm_at1x_atlas_materialize_20260723/atlas_receipt.json"
AUTHORITY_SHA256: Final = "bfda63623b2c87141fb05b5a59f5c27d4411c8d0260fd85e024436a82d08eba8"
PF2_MAIN_RECEIPT_SHA256: Final = "85084f7bd3a03dbd1b9f04fe6a9b84df4948a6caf64620beef42da8924345f73"
PF2_INVALIDATED_IDENTITY_SHA256: Final = "e48468898a64935168ae863d55efd4e38e0cb27ea18278998a1901bc858229fd"
PF2_MAIN_COMMIT_SHORT: Final = "b8c81edec2"
PF2_SCHEMA: Final = "ddm_pf2_dimension_conditioned_two_type_measurement.v1"
PF2_BLOCKER_ID: Final = "PF2_METRIC_ACTIVE_THREE_FORMULATION_ADJUDICATION_INCOMPLETE"


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
            total += len(block)
    return total, digest.hexdigest()


def _json_source(relative_path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    path = REPO / relative_path
    size, sha256 = _sha256_file(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    schema = payload.get("schema")
    if schema is None and isinstance(payload.get("body"), dict):
        schema = payload["body"].get("schema")
    return payload, {
        "path": relative_path,
        "bytes": size,
        "sha256": sha256,
        "schema": schema,
    }


def _ssd_preflight() -> dict[str, Any]:
    tiers = (
        Path("/Volumes/VertigoDataTier/pact"),
        Path("/Volumes/APDataStore/pact"),
    )
    rows: list[dict[str, Any]] = []
    selected: str | None = None
    for tier in tiers:
        exists = tier.is_dir()
        free_bytes = shutil.disk_usage(tier).free if exists else None
        rows.append(
            {
                "path": str(tier),
                "exists": exists,
                "free_bytes": free_bytes,
            }
        )
        if selected is None and exists:
            selected = str(tier)
    return {
        "waterfall": rows,
        "selected_tier_for_future_bulk": selected,
        "bulk_materialized_this_run": False,
        "cleanup_action": "NONE_NO_BULK_CREATED",
        "status": "PREFLIGHT_ONLY_BLOCKED_BEFORE_LARGE_ARTIFACT",
    }


def _unavailable_dual_slots() -> list[dict[str, Any]]:
    return [
        {
            "slot_id": f"class_pair_{left}_{right}",
            "block_instantiated": False,
            "block_key": {
                "stratum": None,
                "scorer_visibility": None,
                "g4_temporal_class": None,
                "class_pair": [left, right],
            },
            "measured_flip_mass": None,
            "kkt_dual": None,
            "score_gain_per_byte": None,
            "exact_delta_bytes": None,
            "units": {
                "kkt_dual": "score_units_per_constraint",
                "score_gain_per_byte": "score_units_per_byte",
                "exact_delta_bytes": "bytes",
            },
            "availability": "UNAVAILABLE",
            "pooling": "FORBIDDEN",
            "reason": "PF2_MAIN_ATLAS_NOT_LOCALLY_COMPOSED_AND_METRIC_ACTIVE_SOLVE_NOT_RUN",
        }
        for left, right in CLASS_PAIRS
    ]


def _representation_rows() -> list[dict[str, Any]]:
    return [
        {
            "type": "SKELETON",
            "stream_type_contract_module": STREAM_TYPE_CONTRACT_MODULE,
            "evaluate_recursion_level": EvaluationRecursionLevel.LEVEL0_SCORE_SIGNATURE.value,
            "derivation": "discrete quotient/event topology/reuse",
            "byte_partition": "COUNTED_REAL_CODER",
            "measurement_status": "BLOCKED_PENDING_LANDED_PF2_PER_STRATUM_RACE",
        },
        {
            "type": "CONNECTION",
            "stream_type_contract_module": STREAM_TYPE_CONTRACT_MODULE,
            "evaluate_recursion_level": EvaluationRecursionLevel.LEVEL2_PAIR_TRAJECTORY.value,
            "derivation": "xi/se3/homography/g4 temporal transport",
            "byte_partition": {
                "operator_code": "FREE_RULE118",
                "parameters": "COUNTED",
                "exceptions": "COUNTED",
            },
            "physical_bev_claim": False,
            "measurement_status": "G4_XI_PROXY_ONLY_NO_PHYSICAL_BEV_CUSTODY",
        },
        {
            "type": "FIBER",
            "stream_type_contract_module": STREAM_TYPE_CONTRACT_MODULE,
            "evaluate_recursion_level": EvaluationRecursionLevel.LEVEL1_SCORER_INTERNALS.value,
            "derivation": "margin-priced continuous scorer/Pose6 coordinates",
            "byte_partition": "COUNTED_NATIVE_ANALOG_CODER",
            "measurement_status": "BLOCKED_PENDING_LANDED_PF2_PER_STRATUM_RACE",
        },
        {
            "type": "GAUGE",
            "stream_type_contract_module": STREAM_TYPE_CONTRACT_MODULE,
            "evaluate_recursion_level": EvaluationRecursionLevel.LEVEL1_SCORER_INTERNALS.value,
            "derivation": "R-null/scorer-invisible coordinates dropped before solve",
            "byte_partition": "ZERO_BYTES_BY_CONSTRUCTION",
            "counted_bytes": 0,
            "measurement_status": "STRICT_CODE_LAW_IMPLEMENTED_NO_N600_CANDIDATE",
        },
        {
            "type": "RESIDUAL",
            "stream_type_contract_module": STREAM_TYPE_CONTRACT_MODULE,
            "evaluate_recursion_level": EvaluationRecursionLevel.LEVEL2_PAIR_TRAJECTORY.value,
            "derivation": "exceptions left after typed skeleton/connection/fiber generation",
            "byte_partition": "COUNTED_EXACT_RUNTIME_PACKET",
            "measurement_status": "BLOCKED_NO_ADMISSIBLE_N600_CANDIDATE",
        },
    ]


def build_receipt(*, finished_at_utc: str) -> dict[str, Any]:
    """Re-derive the current fail-closed receipt from landed sources."""

    ms1, ms1_source = _json_source(MS1_RECEIPT)
    _, g3_source = _json_source(G3_RECEIPT)
    g4, g4_source = _json_source(G4_RECEIPT)
    dr2b, dr2b_source = _json_source(DR2B_RECEIPT)
    inner, inner_source = _json_source(INNER_JACOBIAN_STATUS)
    v16_equations, v16_equations_source = _json_source(V16_EQUATIONS)
    v16, v16_source = _json_source(V16_RECEIPT)
    _, v16_pose_tube_source = _json_source(V16_POSE_TUBE_RECEIPT)
    _, at1x_source = _json_source(AT1X_RECEIPT)
    if ms1.get("schema") != "ddm_ms1_min_description_lattice_solve_repo_receipt.v1":
        raise ValueError("MS1 predecessor receipt schema drifted")
    if ms1["authority"].get("score_claim") is not False or ms1["authority"].get("promotion_eligible") is not False:
        raise ValueError("MS1 authority firewall drifted")

    diagnostic = ms1["realized_frozen_scorer_diagnostic"]
    coder = ms1["diagnostic_conditioning_coder"]
    headline = build_minimum_description_headline(
        stored_problem_bytes=None,
        stored_problem_sha256=None,
        exception_bytes=int(coder["previous_frame_exception_bytes"]),
        exception_sha256=None,
        realized_d_seg=float(diagnostic["realized_d_seg"]),
        realized_d_pose=float(diagnostic["realized_d_pose"]),
        stored_problem_own_lineage=False,
        donor_conditioned=False,
        expansion_receiver_closed=False,
        pose_tube_active=False,
        realized_uint8_r_frozen_scorers=True,
        quotient_coordinates_only=False,
        scorer_metric_active=False,
        alternating_typed_subproblems=False,
        typed_blocks_active=False,
        per_dimension_quanta_active=False,
    )
    # Preserve the historical role label while reusing the exact firewall.
    headline["solve_mandated_exceptions"]["conditional_coding_role"] = (
        "quoted MS1 diagnostic previous-frame conditioning only; not an MS2 counted own-lineage exception stream"
    )

    missing = inner["missing_closure"]
    g4_summary = g4["summary"]
    blockers = [
        "PF2_MAIN_LANDED_ATLAS_REQUIRES_MERGE_COMPOSITION",
        "MEASURED_SCORER_GRAM_NOT_CUSTODIED",
        "EXACT_COMPOSITE_R_HESSIAN_NOT_CUSTODIED",
        "REALIZED_INNER_JACOBIAN_SECANTS_NOT_CUSTODIED",
        "N600_BATCH32_POSE_QUADRATIC_TUBE_NOT_CUSTODIED",
        "PER_DIMENSION_SCORER_QUANTA_NOT_CUSTODIED",
        "N600_METRIC_ACTIVE_ALTERNATION_NOT_RUN",
        "G3_HARD_BLOCK_BOUNDED_EXACT_SIEVE_NOT_RUN",
        "N600_GENERALIZED_DICTIONARY_ALTERNATION_NOT_RUN",
        "N600_TYPED_CANDIDATE_NOT_MATERIALIZED",
    ]
    return {
        "schema": SCHEMA,
        "run_id": RUN_ID,
        "lane_id": LANE_ID,
        "schema_authority_note": {
            "prompt_phrase": "extend the ms1 v2 schema",
            "landed_predecessor_reality": ("MS1 repo receipt and executable headline/typing helpers are v1"),
            "resolution": (
                "additive MS2 repo_receipt.v1 SHA-binds the landed MS1 v1 receipt "
                "and reuses build_minimum_description_headline unchanged"
            ),
        },
        "authority": {
            "evidence_axis": AXIS,
            "research_only": True,
            "execution_allowed": False,
            "score_claim": False,
            "promotion_eligible": False,
            "pointer": POINTER,
            "pointer_moved": False,
            "torch_threads_required": 4,
            "main_landing_review_required": True,
        },
        "score_claim": False,
        "promotion_eligible": False,
        "pointer": POINTER,
        "pointer_moved": False,
        "predecessor_receipt": ms1_source,
        "settled_ms1_facts_quoted_not_rerun": {
            "best_conditional_coder": {
                "bytes": int(coder["previous_frame_exception_bytes"]),
                "fraction_saved": float(coder["previous_frame_fraction_saved"]),
                "times_target_archive": float(coder["best_exception_bytes_over_target"]),
                "target_archive_bytes": int(coder["target_archive_bytes"]),
            },
            "identity_euclidean_full_kernel_cvp_control": {
                "proposal_wins": int(ms1["local_member_selection"]["previous_frame"]["proposal_wins"]),
                "proposal_count": 1200,
                "gauge_fraction_percent": 80.67423152232891,
                "authority": "INSTANCE_SCOPED_NAIVE_CONTROL_REQUIRES_METRIC_ACTIVE_RERUN",
            },
            "unchanged_member_oracle": {
                "pair_count": int(diagnostic["pair_count"]),
                "argmax_identical_pairs": int(diagnostic["argmax_identical_pairs"]),
                "pose6_bit_identical_pairs": int(diagnostic["pose6_bit_identical_pairs"]),
                "d_seg": float(diagnostic["realized_d_seg"]),
                "d_pose": float(diagnostic["realized_d_pose"]),
                "batch_geometry": int(diagnostic["batch_geometry"]),
                "status": "QUOTED_MS1_DIAGNOSTIC_NOT_NEW_MS2_MEASUREMENT",
            },
        },
        "measurement": {
            "finished_at_utc": finished_at_utc,
            "required_pair_count": 600,
            "measured_pair_count": 0,
            "torch_threads_required": 4,
            "torch_invoked": False,
            "receiver_invoked": False,
            "r_operator_invoked": False,
            "frozen_scorer_invoked": False,
            "status": "NOT_RUN_FAIL_CLOSED_BEFORE_CANDIDATE",
            "reason": blockers,
        },
        "storage_preflight": _ssd_preflight(),
        "input_custody": {
            "landed": {
                "ms1": ms1_source,
                "g3_hard_pair_atlas": g3_source,
                "g4_stationarity": g4_source,
                "dr2b_tolerance_rows": dr2b_source,
                "inner_jacobian_status": inner_source,
                "v16_equations": v16_equations_source,
                "v16_receipt": v16_source,
                "v16_pose_tube_receipt": v16_pose_tube_source,
                "at1x_metric_atlas_receipt": at1x_source,
            },
            "pf2_main_landing_handoff": {
                "status": "MAIN_LANDED_CITED_NOT_LOCALLY_REHASHED",
                "main_commit_short": PF2_MAIN_COMMIT_SHORT,
                "receipt_sha256": PF2_MAIN_RECEIPT_SHA256,
                "receipt_bytes": 1_696_256,
                "schema": PF2_SCHEMA,
                "reported_bucket_count": 1200,
                "class_pair_count": 10,
                "measured_event_mass": 4_011_236,
                "largest_pair_bucket": {
                    "class_pair": "Road-Undrivable",
                    "event_mass": 1_280_501,
                },
                "main_landed": True,
                "self_declared_verdict": (
                    "PF2_METRIC_ACTIVE_THREE_FORMULATION_ADJUDICATION_INCOMPLETE_"
                    "F1_EXACT_CONTROL_SURVIVES_F3_NEGATIVE_F2_OWED"
                ),
                "metric_rerun_blocker_id": PF2_BLOCKER_ID,
                "pf2r_requires_from_ms2": [
                    "bucket-complete rank4 winner/rival margin-Fisher field with lambda",
                    "pairwise PoseNet6 quadratic for the F2 hood basis",
                    "exact composite-R readback",
                    "Euclid-vs-Fisher cosine and relative-norm diagnostics",
                ],
                "invalidated_identity_metric_receipt_sha256": (PF2_INVALIDATED_IDENTITY_SHA256),
                "safe_use": (
                    "CITE_TYPED_ATLAS_FOR_MAIN_COMPOSITION; DO_NOT_TREAT_AS_"
                    "METRIC_ACTIVE_ADJUDICATION"
                ),
            },
            "scorer_geometry_audit": {
                "seg_rank4_head": "EXACT_HEAD_GEOMETRY_EXISTS",
                "seg_source_vjp": "MEASURED_REAL_N600_SOURCE_ARRANGEMENT_ONLY",
                "seg_metric_gram": "ABSENT",
                "exact_composite_r_hessian": "ABSENT",
                "realized_backbone_secants": missing["realized_backbone_secants"],
                "qp_receiver_closure": missing["qp_receiver_closure"],
                "inner_jacobian_record_sha256": missing["record_sha256"],
                "v16_gauss_newton_authority": v16_equations["reductions"]["Gauss_Newton_JtJ"],
                "v16_all_kkt_attempts_converged": bool(v16_equations["measurement"]["all_four_kkt_attempts_converged"]),
                "at1x_pose6_vjp_gram": ("FULL_N600_TWO_6X6_ROW_GRAMS_PER_PAIR_BUT_PAIR_LOCAL_BATCH1"),
                "at1x_seg_metric": ("ONE_CONTRACTED_ROW_ENERGY_PER_DEPTH_NOT_PER_BLOCK_RANK4_FISHER_GRAM"),
                "v16_pose_tube": (
                    "EIGHT_PAIRS_BATCH16_THREADS4_96_LINEAR_BOX_ROWS_ONLY; "
                    "FIRST_ORDER_AND_GN_MAX_ITERATIONS_RESIDUAL_NOT_CLEAN"
                ),
                "dual_metric_readback": "NOT_AVAILABLE_FOR_MS2_BLOCKS",
            },
            "g4_scope": {
                "classes": [
                    "STATIC_IN_IMAGE",
                    "STATIC_IN_XI_PROXY",
                    "TRANSIENT",
                ],
                "static_in_image_fraction": float(
                    g4_summary["stationarity_decomposition"]["all"]["classes"]["STATIC_IN_IMAGE"]["fraction"]
                ),
                "physical_bev_fraction": g4_summary["independent_physical_bev_fraction"],
                "bev_blocker": g4_summary["bev_blocker"],
                "rule": "STATIC_IN_XI_PROXY_MUST_NOT_BE_LAUNDERED_AS_PHYSICAL_BEV",
            },
            "dr2b_scope": {
                "verdict": dr2b["verdict"],
                "verdict_scope": dr2b["verdict_scope"],
                "use": "INPUT_PRICE_ROWS_ONLY_NOT_FULL_PER_VISIBLE_DIMENSION_QUANTA",
            },
        },
        "five_typings": {
            "quotient_coordinates_only": {
                "declaration": False,
                "apparatus": "STRICT_VISIBLE_QUOTIENT_AND_GAUGE_ZERO_LAW_IMPLEMENTED",
                "visible_fraction": 0.1932576847767109,
                "visible_fraction_epistemic_status": "DERIVED_FROM_EXACT_580_NULLITY",
                "projector": "separable_resize_full_kernel_direct_sum_v1",
                "preimage": "tac.optimization.uint8_lattice_feasibility.minimum_norm_real_preimage",
                "free_fill": "tac.through_r.blind_coordinate",
                "n600_realization": "NOT_RUN",
            },
            "scorer_metric_active": {
                "declaration": False,
                "required": (
                    "rank4 winner-rival hyperplanes + margin-Fisher + <=6D Pose "
                    "quadratic + exact composite-R Hessian/adjoint + realized inner Jacobian"
                ),
                "coordinate_system": "seg_rank4_winner_rival_hyperplanes_plus_pose6",
                "identity_euclidean": "CONTROL_ONLY_NOT_EXECUTED",
                "geometry_ladder": [
                    "MEASURED_SCORER_SECOND_ORDER",
                    "SIMPLER_FORMS_CONTROL_ONLY",
                    "IDENTITY_EUCLIDEAN_CONTROL",
                ],
                "apparatus": "STRICT_NONIDENTITY_GATE_IMPLEMENTED",
            },
            "alternating_typed_subproblems": {
                "declaration": False,
                "required_order": [
                    "ARGMAX_CELL_SELECTION",
                    "WITHIN_CELL_CONTINUOUS_LATTICE",
                    "REAL_CODER_PRICE",
                ],
                "pose_tube_active_each_iteration": False,
                "real_e3_packet_price_api": ("tac.optimization.ddm_runtime_exporter:price_exact_runtime_marginal"),
                "completed_cycles": 0,
            },
            "typed_blocks_active": {
                "declaration": False,
                "canonical_contract_module": STREAM_TYPE_CONTRACT_MODULE,
                "canonical_types_expected": [
                    "StreamType",
                    "TypedStreamTag",
                ],
                "parallel_enum_defined": False,
                "key": [
                    "stratum",
                    "scorer_visibility",
                    "g4_temporal_class",
                ],
                "class_count_required": 5,
                "class_pair_count_required": 10,
                "quoted_pf2_class_pair_coverage": 10,
                "locally_composed_class_pair_coverage": 0,
                "pf2_reconciliation": (
                    "MAIN_LANDED_COMMIT_B8C81EDEC2_NOT_PRESENT_IN_ISOLATED_BASE"
                ),
                "physical_bev_amortization_active": False,
            },
            "per_dimension_quanta_active": {
                "declaration": False,
                "law": "effective_quantum = uint8_step * measured_scorer_sensitivity",
                "tolerance_knees_swept": False,
                "apparatus": "STRICT_EFFECTIVE_QUANTUM_TYPE_IMPLEMENTED",
            },
        },
        "ws1_pose_serving_fiber_prior": {
            "source": (
                ".omx/research/"
                "codex_findings_ddm_ws1_seg_lexicographic_warmstart_20260724_codex.md"
            ),
            "source_status": "INBOX_HANDOFF_ONLY_NOT_PRESENT_IN_ISOLATED_BASE",
            "receiver_sha256_prefix": "4fbba057b10c",
            "joint_accepted_moves": 104,
            "seg_lexicographic_retained": 96,
            "pose_serving_fiber_reclassified": 8,
            "rule": (
                "pose-serving content is FIBER with POSE_VISIBLE or "
                "SEG_POSE_VISIBLE pricing against the exact <=6D Pose quadratic"
            ),
            "strict_validator": (
                "tac.optimization.ddm_typed_quotient_solve.TypedBlock.pose_serving"
            ),
            "used_as_measured_block_row": False,
            "main_composition_review_required": True,
        },
        "representation_rows": _representation_rows(),
        "per_block_dual_exchange_rows": [],
        "unavailable_pair_slots": _unavailable_dual_slots(),
        "dual_policy": {
            "per_block_rows_required_for_solve_feed": True,
            "pooled_lambda_forbidden": True,
            "fisher_margin_as_dual_forbidden": True,
            "imputation_forbidden": True,
            "status": "UNAVAILABLE_NO_INSTANTIATED_METRIC_ACTIVE_BLOCKS",
        },
        "bounded_exact_hard_block_sieve": {
            "method_available": ("tac.optimization.ddm_typed_quotient_solve:bounded_exact_metric_sieve"),
            "g3_hard_pair_registry": (".omx/research/ddm_g3_score_atlas_n600_20260722T204000Z/hard_pair_registry.json"),
            "representative_blocks_run": 0,
            "search_complete": False,
            "status": "NOT_RUN_NO_MEASURED_SCORER_GEOMETRY_OR_LANDED_TYPED_BLOCK",
            "lattice_family_negative_authorized": False,
        },
        "alternating_dictionary": {
            "method_available": ("tac.optimization.ddm_typed_quotient_solve:generalized_metric_dictionary_update"),
            "plain_svd_authority": "FORBIDDEN_CONTROL_ONLY",
            "n600_basis_updates": 0,
            "tolerance_homotopy_completed": False,
            "status": "NOT_RUN_NO_MEASURED_SCORER_GEOMETRY",
        },
        "skeleton_fiber_distillation": {
            "historical_ms1_numerical_factors": int(
                ms1["sense_custody"]["historical_factorization"]["numerical_factors_above_one_byte_floor"]
            ),
            "distilled_factors": 0,
            "measured_per_stratum_race_consumed": False,
            "status": "BLOCKED_METRIC_GEOMETRY_AND_LOCAL_PF2_COMPOSITION",
        },
        "resumability": {
            "run_started": False,
            "per_block_atomic_checkpoint_contract": {
                "path_template": (
                    "/Volumes/VertigoDataTier/pact/ddm_ms2_typed_quotient_solve/"
                    "blocks/{block_id}/stage_{stage_index:04d}.json"
                ),
                "write": "temporary_plus_fsync_plus_os_replace",
                "preserve_all_stages": True,
                "resume_key": [
                    "block_id",
                    "stage_index",
                    "geometry_receipt_sha256",
                    "pf2_atlas_receipt_sha256",
                    "typed_config_sha256",
                ],
            },
            "complete_block_stages": 0,
            "stage_loss_bound": "NO_RUN; future run at most one intra-block stage",
        },
        "frame_custody": {
            "ms2_candidate_exists": False,
            "frame_0_sha256": None,
            "frame_0_byte_identical_claim": None,
            "pose_tube_claim": False,
            "reason": "NO_ADMISSIBLE_CANDIDATE_MATERIALIZED",
        },
        "campaign_headline": headline,
        "blockers": blockers,
        "verdict": "BLOCKED_NO_ADMISSIBLE_METRIC_ACTIVE_N600_CANDIDATE",
        "verdict_scope": (
            "INSTANCE x current landed-custody preflight. This is not a "
            "quotient, lattice, typed-block, dictionary, stored-problem, "
            "receiver, family, or paradigm negative."
        ),
        "first_rung_measurements": [
            (
                "At MAIN merge, compose PF2 commit b8c81edec2, rehash its receipt, "
                "and validate all 1,200 typed rows without rebuilding the atlas."
            ),
            "Materialize SHA-bound scorer-native Gram plus exact composite-R adjoint/Hessian over visible quotient coordinates.",
            "Measure candidate-arrangement realized inner-Jacobian secants and close the receiver QP.",
            "Build the n600 batch32 <=6D Pose quadratic and keep its tube active inside each alternating cycle.",
            "Run bounded exact sieve on preregistered G3 hard blocks before any lattice-family negative.",
            "Run per-block generalized metric dictionary updates along the tolerance homotopy.",
            "Emit exact unpooled block dual/exchange rows and only then rerun the headline firewall.",
        ],
        "train_decision_table_feed": {
            "column": "SOLVE",
            "status": "BLOCKED_CUSTODY_NO_DUAL_ROWS",
            "pooling": "FORBIDDEN",
            "pointer": POINTER,
            "pointer_moved": False,
        },
        "named_downstream_consumers": [
            {
                "consumer": "train-decision-table SOLVE column",
                "status": "BLOCKED_CUSTODY_NO_DUAL_ROWS",
            },
            {
                "consumer": "pf2r metric-active three-formulation rerun",
                "blocker_id": PF2_BLOCKER_ID,
                "status": "BLOCKED_MISSING_MS2_METRIC_CUSTODY_ROWS",
            },
        ],
        "canonical_equations": {
            "equation_ids": list(EQUATION_IDS),
            "callable_module": ("tac.canonical_equations.ddm_ms2_typed_quotient_solve_20260724"),
            "registration_contract": (
                "registered exactly once against the preserved pre-PF2 receipt; "
                "later PF2 landing custody is linked by locked append-only "
                "domain_refined events; live latest-event state verified"
            ),
            "historical_registration_receipt": {
                "path": (
                    ".omx/research/"
                    "ddm_ms2_typed_quotient_solve_pre_pf2_20260724T031200Z_receipt.json"
                ),
                "sha256": (
                    "9b17c5108e4b8d5a517ecb66276fc0e78162e54b53a9f4d819a48286989b98b6"
                ),
            },
        },
        "triality": {
            "dsl": [
                "src/tac/optimization/ddm_typed_quotient_solve.py",
                "src/tac/optimization/ddm_runtime_exporter.py",
            ],
            "dag": (".omx/research/ddm_ms2_typed_quotient_solve_DAG_FEED_20260724.md"),
            "equations": list(EQUATION_IDS),
            "consumed_by": "train-decision-table SOLVE column after MAIN review",
        },
        "directive_consumption": [
            {
                "source": "authority_file",
                "source_sha256": AUTHORITY_SHA256,
                "status": "CONSUMED",
                "durable_effect": "MS2 five typings, local-only authority, receipt and review contract",
            },
            {
                "source": "ddm_scorer_native_doctrine_and_synthesis_20260723 points 1-9b",
                "status": "CONSUMED",
                "durable_effect": "scorer-native/R/real-coder and no-fake firewall",
            },
            {
                "source": MS1_RECEIPT,
                "source_sha256": ms1_source["sha256"],
                "status": "CONSUMED_QUOTED_NOT_RERUN",
                "durable_effect": "predecessor facts and exact headline helper inputs",
            },
            {
                "source": "inbox:2026-07-24T02:27:12Z",
                "status": "CONSUMED_P0",
                "durable_effect": (
                    "nonidentity measured scorer metric binds KKT/CVP/trust/ranking/"
                    "generalized dictionary; Euclidean control never verdict-bearing"
                ),
            },
            {
                "source": "inbox:2026-07-24T02:28:21Z",
                "status": "CONSUMED_P0",
                "durable_effect": (
                    "exact composite-R second order, scorer hyperplane coordinates, "
                    "and most-optimal-first ladder enforced"
                ),
            },
            {
                "source": "inbox:2026-07-24T03:02:01Z",
                "status": "CONSUMED_AS_TYPED_PRIOR_SOURCE_CUSTODY_OWED",
                "durable_effect": (
                    "WS1 8/104 pose-serving moves are FIBER priced in Pose-visible "
                    "coordinates; stream fields cite ddm_min_description_contract "
                    "and no parallel enum is defined"
                ),
            },
            {
                "source": "inbox:2026-07-24T03:13:37Z",
                "status": "CONSUMED_MAIN_LANDING_CITATION",
                "durable_effect": (
                    "PF2 commit b8c81edec2 supplies the 1,200-row typed atlas; "
                    "PF2R is a second named consumer; metric adjudication remains "
                    "blocked on MS2 scorer-geometry custody"
                ),
            },
            {
                "source": "PF2 MAIN landing",
                "source_sha256": PF2_MAIN_RECEIPT_SHA256,
                "status": "CITED_FOR_MAIN_COMPOSITION_NOT_LOCALLY_REHASHED",
                "durable_effect": (
                    "typed atlas must be reused rather than rebuilt; current "
                    "isolated-base declaration remains false until composition"
                ),
            },
            {
                "source": "RD1 dual feed",
                "status": "NOT_PRESENT_NOT_IMPUTED",
                "durable_effect": "per-block dual rows remain null and SOLVE feed stays blocked",
            },
        ],
        "actuation": {
            "paid_dispatch": False,
            "exact_eval": False,
            "frontier_mutation": False,
            "training_launch": False,
            "local_n600_scorer_run": False,
        },
        "main_landing_review_required": True,
    }


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    temporary = path.with_name(f"{path.name}.partial.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--finished-at-utc",
        default=None,
        help="ISO UTC override for deterministic replay tests",
    )
    parser.add_argument("--register-equations", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    finished = args.finished_at_utc or dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    receipt = build_receipt(finished_at_utc=finished)
    _atomic_write_json(args.output, receipt)
    result: dict[str, Any] = {
        "output": str(args.output),
        "bytes": _sha256_file(args.output)[0],
        "sha256": _sha256_file(args.output)[1],
        "verdict": receipt["verdict"],
        "score_claim": False,
    }
    if args.register_equations:
        from tac.canonical_equations.ddm_ms2_typed_quotient_solve_20260724 import (
            populate_ddm_ms2_typed_quotient_equations,
        )

        equations = populate_ddm_ms2_typed_quotient_equations(
            receipt_path=args.output,
            agent="codex",
            subagent_id="ddm_ms2_typed_quotient_solve_20260724T022221Z",
        )
        result["registered_equation_ids"] = [row.equation_id for row in equations]
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
