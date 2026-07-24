# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from tac.optimization.ddm_dm4_j5_adapter import (
    DM4J5AdapterError,
    adapt_dm4_proposals,
)
from tac.optimization.direct_description_joint_descent import (
    DirectDescriptionJointDescentTypedConfigV1,
)

REPO = Path(__file__).resolve().parents[4]
RECEIPT = (
    REPO
    / ".omx/research/ddm_dm4_targeted_realization_cures_20260724T142722Z"
    / "ddm_dm4_targeted_realization_cures_receipt.json"
)


def _receipt_sha() -> str:
    return hashlib.sha256(RECEIPT.read_bytes()).hexdigest()


def test_disabled_dm4_source_is_byte_identical() -> None:
    archive = b"receiver-closed-archive"
    output, proposals, receipt = adapt_dm4_proposals(
        receipt_path=RECEIPT,
        receipt_sha256=_receipt_sha(),
        base_archive=archive,
        enabled=False,
    )
    assert output is archive
    assert proposals == ()
    assert receipt["byte_identical_disabled"] is True
    assert receipt["base_archive_sha256"] == receipt["output_archive_sha256"]


def test_enabled_dm4_source_emits_only_typed_scorer_recursive_proposals() -> None:
    archive = b"receiver-closed-archive"
    output, proposals, receipt = adapt_dm4_proposals(
        receipt_path=RECEIPT,
        receipt_sha256=_receipt_sha(),
        base_archive=archive,
        enabled=True,
    )
    assert output == archive
    assert proposals
    assert receipt["proposal_count"] == len(proposals)
    assert [proposal.proposal_id for proposal in proposals] == sorted(
        proposal.proposal_id for proposal in proposals
    )
    for proposal in proposals:
        payload = proposal.to_payload()
        assert payload["type"] == "joint"
        assert payload["aimed_cell"]["bucket_id"]
        assert payload["corrected_J_row"]["metric"].startswith("rank4 target-vs-runner")
        assert payload["support_footprint"]["support_rule"].startswith("scorer-recursive")
        assert payload["candidate"]["candidate_id"].startswith("scorer_recursive_")


def test_dm4_source_refuses_receipt_hash_drift() -> None:
    with pytest.raises(DM4J5AdapterError, match="receipt SHA differs"):
        adapt_dm4_proposals(
            receipt_path=RECEIPT,
            receipt_sha256="0" * 64,
            base_archive=b"archive",
            enabled=True,
        )


def test_j5_consumer_exposes_the_hash_bound_dm4_proposal_source() -> None:
    consumer = SimpleNamespace(
        semantic_program={
            "proposal_sources": {
                "dm4_scorer_recursive": {
                    "path": str(RECEIPT),
                    "sha256": _receipt_sha(),
                    "adapter": (
                        "tac.optimization.ddm_dm4_j5_adapter.adapt_dm4_proposals"
                    ),
                    "application_authority": (
                        "fail_closed_until_counted_J5_application_operator_exists"
                    ),
                }
            }
        },
        ticket_path=str(REPO / ".omx/research/configs/unused.json"),
    )
    archive = b"receiver-closed-archive"
    output, proposals, receipt = (
        DirectDescriptionJointDescentTypedConfigV1.dm4_j5_proposal_source(
            consumer,
            base_archive=archive,
            enabled=True,
        )
    )
    assert output == archive
    assert len(proposals) == receipt["proposal_count"] == 6
