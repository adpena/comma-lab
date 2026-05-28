# SPDX-License-Identifier: MIT
"""Mathematical contract for scorer-region cascade materializers."""

from __future__ import annotations

from typing import Any

from tac.optimization.dqs1_materializer_feedback_bridge import FALSE_AUTHORITY

SCORER_REGION_OPERATOR_CONTRACT_SCHEMA = "scorer_region_operator_contract.v1"


def build_scorer_region_operator_contract(
    *,
    chain_label: str,
    receiver_patch_enabled: bool = False,
) -> dict[str, Any]:
    """Return the reusable operator/Lagrangian contract for Cascade C."""

    return {
        "schema": SCORER_REGION_OPERATOR_CONTRACT_SCHEMA,
        "chain_label": chain_label,
        "chain_position_order": ["P19", "P18", "P11", "P15"],
        "optimization_functional": {
            "schema": "contest_score_lagrangian_proxy.v1",
            "expression": "DeltaS_hat = DeltaD_seg + DeltaD_pose + 25*DeltaBytes/37545489",
            "rate_denominator_bytes": 37_545_489,
            "exact_auth_eval_is_final_authority": True,
            "local_terms_are_acquisition_signal_only": True,
        },
        "entropy_position_model": [
            {
                "position": "P19",
                "surface": "PoseNet entropy",
                "operator": "pose_null_subset_detection",
                "phase": "scorer",
            },
            {
                "position": "P18",
                "surface": "SegNet entropy",
                "operator": "class_region_waterfill",
                "phase": "scorer",
            },
            {
                "position": "P11",
                "surface": "selector stream entropy",
                "operator": "context_recode",
                "phase": "archive",
            },
            {
                "position": "P15",
                "surface": "zip wrapper entropy",
                "operator": "deterministic_repack",
                "phase": "codec",
            },
        ],
        "composition_law": {
            "position_disjoint": "expected_linear_until_empirically_falsified",
            "position_shared": "subadditive_select_best_survivor_only",
            "cascade": (
                "upstream_scorer_region_repairs_may_enable_downstream_selector_or_"
                "wrapper_codecs"
            ),
            "selected_survivor_rule": (
                "use_P15_archive_zip_repack_only_when_rate_positive_and_receiver_"
                "contract_closed_else_use_P11_selector_context_recode"
            ),
        },
        "receiver_contract": {
            "enabled": bool(receiver_patch_enabled),
            "operator": "R_T(frame_pairs)=frame1_region_waterfill_after_selector_decode",
            "candidate_archive_binding": (
                "receiver patch must copy the selected local survivor archive, not the "
                "original source archive"
            ),
            "proof_required": (
                "shape_preserving_full_frame_shell_inflate_output_change"
                if receiver_patch_enabled
                else "receiver_patch_not_materialized"
            ),
        },
        "proof_obligations": [
            "full_frame_inflate_parity_before_budget_spend",
            "selected_survivor_archive_custody",
            "receiver_runtime_content_tree_custody",
            "shape_preserving_full_frame_output_change_if_receiver_patch_enabled",
            "contest_cpu_or_cuda_auth_eval_before_score_or_promotion_claim",
        ],
        "packetir_generalization_targets": [
            "packetir.selector_context_stream",
            "packetir.receiver_region_operator",
            "hnerv.selector_sidecar",
            "boostnerv.residual_region_budget",
            "nerv_family.frame_pair_region_operator",
            "non_nerv.archive_bound_region_transform",
        ],
        "allowed_use": "mathematical_planning_contract_for_queue_owned_local_search",
        "forbidden_use": "score_claim_or_dispatch_authority",
        **FALSE_AUTHORITY,
    }


__all__ = [
    "SCORER_REGION_OPERATOR_CONTRACT_SCHEMA",
    "build_scorer_region_operator_contract",
]
