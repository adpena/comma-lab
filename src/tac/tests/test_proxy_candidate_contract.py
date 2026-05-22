# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest

from tac.optimization.proxy_candidate_contract import (
    apply_proxy_evidence_boundary,
    auth_bridge_score_rankable,
    require_no_truthy_authority_fields,
    truthy_authority_field_violations,
    validate_proxy_candidate,
)


def test_proxy_boundary_forces_promotable_false() -> None:
    row = apply_proxy_evidence_boundary(
        {
            "candidate_id": "unsafe",
            "promotable": True,
            "score_claim": True,
        }
    )

    assert row["promotable"] is False
    assert row["score_claim"] is False
    assert validate_proxy_candidate(row) == []


def test_consumer_payload_authority_validator_rejects_truthy_fields() -> None:
    payload = {
        "score_claim_valid": True,
        "dispatch_packet_ready": True,
        "domain_payload": {"ok": True},
    }

    violations = truthy_authority_field_violations(payload)

    assert violations == [
        "score_claim_valid=truthy",
        "dispatch_packet_ready=truthy",
    ]
    with pytest.raises(ValueError, match="score_claim_valid=truthy"):
        require_no_truthy_authority_fields(payload, context="fixture")


def test_consumer_payload_authority_validator_recurses_into_nested_payloads() -> None:
    payload = {
        "optimizer_recipe": {
            "id": "unsafe",
            "ready_for_exact_eval_dispatch": True,
        },
        "candidate_payloads": [
            {"promotable": True},
            {"domain_payload": {"ok": True}},
        ],
    }

    violations = truthy_authority_field_violations(payload)

    assert violations == [
        "optimizer_recipe.ready_for_exact_eval_dispatch=truthy",
        "candidate_payloads[0].promotable=truthy",
    ]
    with pytest.raises(
        ValueError,
        match=r"optimizer_recipe\.ready_for_exact_eval_dispatch=truthy",
    ):
        require_no_truthy_authority_fields(payload, context="fixture")


def test_consumer_payload_authority_validator_rejects_string_and_numeric_truthy() -> None:
    payload = {
        "score_claim_valid": "true",
        "dispatch_packet_ready": 1,
        "promotion_eligible": "yes",
        "rank_or_kill_eligible": "0",
        "domain_payload": {"ok": True},
    }

    violations = truthy_authority_field_violations(payload)

    assert violations == [
        "score_claim_valid=truthy",
        "dispatch_packet_ready=truthy",
        "promotion_eligible=truthy",
    ]
    with pytest.raises(ValueError, match="promotion_eligible=truthy"):
        require_no_truthy_authority_fields(payload, context="fixture")


def test_consumer_payload_authority_validator_allows_false_or_missing() -> None:
    require_no_truthy_authority_fields(
        {
            "score_claim_valid": False,
            "dispatch_packet_ready": False,
            "domain_payload": {"ok": True},
        },
        context="fixture",
    )


def test_auth_bridge_score_rankable_accepts_contest_cpu_gha_axis_variants() -> None:
    assert auth_bridge_score_rankable(
        {"score_comparable": True, "score_axis": "contest_cpu_gha"}
    )
    assert auth_bridge_score_rankable(
        {"score_comparable": True, "score_axis": "[contest-CPU GHA Linux x86_64]"}
    )
    assert auth_bridge_score_rankable(
        {"score_comparable": True, "score_axis": "contest-cpu-linux-x86-64-modal"}
    )
    assert not auth_bridge_score_rankable(
        {"score_comparable": False, "score_axis": "[contest-CPU GHA Linux x86_64]"}
    )
    assert not auth_bridge_score_rankable(
        {"score_comparable": True, "score_axis": "macOS-CPU advisory"}
    )
