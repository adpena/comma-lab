# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.witness_control.taskspace_g23_g29_g33_endpoint_adapter import (
    G36BlockerCodeV1,
    G36EndpointAdapterError,
    G36EndpointAdapterReadinessReceiptV1,
    G36EvidenceRoleV1,
    G36G29PreflightEvidenceBytesV1,
    audit_g36_endpoint_adapter_readiness,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
RECEIPT = (
    REPO_ROOT
    / ".omx/research/original_taskspace_inverse_witness_codec_20260725"
    / "g36_endpoint_adapter_readiness_g29_preflight_20260726.json"
)
G29_ROOT = Path("/Volumes/VertigoDataTier/pact/g29_public_auth_closure_20260726_codex.pc0fLk/receipts")


def _real_evidence() -> G36G29PreflightEvidenceBytesV1:
    required = {
        "compile": G29_ROOT / "compile.json",
        "placement": G29_ROOT / "placement.json",
        "abi": G29_ROOT / "decoder_abi.json",
        "audit_0": G29_ROOT / "source_audit_000.json",
        "audit_1": G29_ROOT / "source_audit_001.json",
        "discovery": G29_ROOT / "001_dependency_discovery.json",
        "readiness": G29_ROOT / "002_execution_readiness.json",
    }
    if not all(path.is_file() for path in required.values()):
        pytest.skip("certified real G29 preflight receipts are not mounted")
    return G36G29PreflightEvidenceBytesV1(
        compile_receipt_bytes=required["compile"].read_bytes(),
        placement_receipt_bytes=required["placement"].read_bytes(),
        decoder_abi_receipt_bytes=required["abi"].read_bytes(),
        generic_source_audit_receipt_bytes=(
            required["audit_0"].read_bytes(),
            required["audit_1"].read_bytes(),
        ),
        dependency_discovery_receipt_bytes=required["discovery"].read_bytes(),
        execution_readiness_receipt_bytes=required["readiness"].read_bytes(),
    )


def test_checked_in_readiness_receipt_roundtrips_without_authority() -> None:
    receipt = G36EndpointAdapterReadinessReceiptV1.from_receipt_bytes(RECEIPT.read_bytes())
    assert receipt.to_receipt_bytes() == RECEIPT.read_bytes()
    assert not receipt.g33_base_constructible
    assert not receipt.g33_endpoint_constructible
    assert not receipt.score_authority
    assert not receipt.candidate_archive
    assert not receipt.frontier_pointer_moved
    assert receipt.research_only
    assert G36EvidenceRoleV1.G17_WHOLE_OBJECT_STATE.value in receipt.missing_roles
    assert G36EvidenceRoleV1.G17_G29_RECEIVER_RASTER_BRIDGE.value in receipt.missing_roles


def test_real_g29_preflight_emits_exact_fail_closed_partition() -> None:
    receipt = audit_g36_endpoint_adapter_readiness(
        _real_evidence(),
        dynamic_pointer_bytes=(REPO_ROOT / ".omx/state/canonical_frontier_pointer.json").read_bytes(),
    )
    assert G36EndpointAdapterReadinessReceiptV1.from_receipt_bytes(receipt.to_receipt_bytes()) == receipt
    assert receipt.archive_nbytes == 80_238
    assert receipt.exact_rate_term + receipt.residual_distortion_score_budget_at_current_rate == (
        receipt.dynamic_target_score
    )
    assert G36BlockerCodeV1.G29_PUBLIC_EVALUATOR_EXECUTION_RECEIPT_OWED.value in receipt.blockers
    assert G36EvidenceRoleV1.G29_PUBLIC_RUNTIME_COMPILE.value in receipt.satisfied_roles


def test_receipt_refuses_noncanonical_reemit() -> None:
    value = json.loads(RECEIPT.read_bytes())
    pretty = json.dumps(value, indent=2, sort_keys=True).encode("ascii")
    with pytest.raises(G36EndpointAdapterError, match="canonical parse/re-emit"):
        G36EndpointAdapterReadinessReceiptV1.from_receipt_bytes(pretty)


def test_receipt_refuses_authority_bit_flip_even_when_resealed() -> None:
    value = json.loads(RECEIPT.read_bytes())
    value["score_authority"] = True
    resealed = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii")
    with pytest.raises(G36EndpointAdapterError, match="mint authority"):
        G36EndpointAdapterReadinessReceiptV1.from_receipt_bytes(resealed)
