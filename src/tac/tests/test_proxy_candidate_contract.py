# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest

from tac.optimization.proxy_candidate_contract import (
    apply_proxy_evidence_boundary,
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
        "score_claim_valid=true",
        "dispatch_packet_ready=true",
    ]
    with pytest.raises(ValueError, match="score_claim_valid=true"):
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
