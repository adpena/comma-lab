# SPDX-License-Identifier: MIT
from __future__ import annotations

from tac.geometry_feedback_readiness import (
    EXACT_CUDA_BLOCKER,
    UNCHARGED_GEOMETRY_FEEDBACK_BLOCKER,
    build_geometry_feedback_runtime_contract,
    geometry_feedback_contract_failures,
)


def test_geometry_feedback_contract_blocks_uncharged_lapose_raft_openpilot_feedback() -> None:
    contract = build_geometry_feedback_runtime_contract(
        lane_key="raft_radial_openpilot_pose",
        paradigms=("pose", "openpilot_priors", "la_pose"),
        role="proposal_or_pose_sidecar_replacement",
    )

    assert contract["score_claim"] is False
    assert contract["dispatch_attempted"] is False
    assert contract["ready_for_exact_eval_dispatch"] is False
    assert contract["charged_runtime_consumed"] is False
    assert UNCHARGED_GEOMETRY_FEEDBACK_BLOCKER in contract["dispatch_blockers"]
    assert "geometry_feedback_no_charged_artifact_proof" in contract["dispatch_blockers"]
    assert "geometry_feedback_no_runtime_consumer_proof" in contract["dispatch_blockers"]
    assert geometry_feedback_contract_failures(contract) == ()


def test_geometry_feedback_contract_narrows_only_after_charged_runtime_match() -> None:
    contract = build_geometry_feedback_runtime_contract(
        lane_key="telescopic_foveation_field",
        paradigms=("foveation", "openpilot_priors", "pose"),
        role="scorer_weighted_proposal_or_replacement",
        charged_artifacts=(
            {
                "path": "archive.zip",
                "member": "foveation_params.bin",
                "bytes": 32,
                "sha256": "a" * 64,
                "charged": True,
            },
        ),
        runtime_consumers=(
            {
                "path": "submissions/robust_current/inflate_renderer.py",
                "consumes_charged_artifact": True,
                "consumed_artifacts": ["foveation_params.bin"],
            },
        ),
    )

    assert contract["charged_runtime_consumed"] is True
    assert contract["charged_artifact_count"] == 1
    assert contract["runtime_consumer_count"] == 1
    assert UNCHARGED_GEOMETRY_FEEDBACK_BLOCKER not in contract["dispatch_blockers"]
    assert EXACT_CUDA_BLOCKER in contract["dispatch_blockers"]
    assert contract["ready_for_exact_eval_dispatch"] is False
    assert geometry_feedback_contract_failures(contract) == ()


def test_geometry_feedback_contract_validator_rejects_dispatchable_payload() -> None:
    contract = build_geometry_feedback_runtime_contract(
        lane_key="lapose_motion_atom_allocator",
        paradigms=("la_pose", "openpilot_priors"),
        role="proposal_allocator",
    )
    contract["ready_for_exact_eval_dispatch"] = True
    contract["dispatch_blockers"] = [
        item
        for item in contract["dispatch_blockers"]
        if item != UNCHARGED_GEOMETRY_FEEDBACK_BLOCKER
    ]

    failures = geometry_feedback_contract_failures(contract)

    assert "geometry_feedback_contract_ready_for_exact_eval_dispatch_false" in failures
    assert "geometry_feedback_contract_uncharged_blocker_present" in failures
