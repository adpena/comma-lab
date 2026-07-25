# SPDX-License-Identifier: MIT
"""Real-artifact and strict-custody tests for applied-action adapters."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from tac.analysis.applied_action_adapters import (
    ADAPTER_MANIFEST_SCHEMA,
    PF3_PHYSICAL_EDGE,
    AppliedActionAdapterError,
    adapt_j8f_checkpoints,
    adapt_j12_receipt,
    adapt_pf3_checkpoints,
    build_adapter_manifest,
    canonical_json_sha256,
    validate_adapter_manifest,
    verify_source_artifact,
)
from tools.build_applied_action_receipts import (
    DEFAULT_J8F_CONFIG,
    DEFAULT_J8F_ROOT,
    DEFAULT_J12_RECEIPT,
    DEFAULT_PF3_RECEIPT,
    _j8f_inputs,
    _j12_inputs,
    _pf3_inputs,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="session")
def real_inputs() -> dict[str, Any]:
    if not (DEFAULT_J8F_ROOT / "ddm_j8f_counted_application_receipt.json").exists():
        pytest.skip("custodied SSD artifacts are not mounted")
    return {
        "j8": _j8f_inputs(DEFAULT_J8F_ROOT, DEFAULT_J8F_CONFIG),
        "pf3": _pf3_inputs(DEFAULT_PF3_RECEIPT),
        "j12": _j12_inputs(DEFAULT_J12_RECEIPT),
    }


def _results(real_inputs: dict[str, Any]) -> tuple[Any, ...]:
    smoke, checkpoints, config, smoke_id, checkpoint_ids, config_id, custody = real_inputs["j8"]
    pf_receipt, pf_checkpoints, pf_id, pf_checkpoint_ids, pf_custody = real_inputs["pf3"]
    j12_receipt, pricing, j12_id, pricing_id, j12_custody = real_inputs["j12"]
    return (
        *adapt_j8f_checkpoints(
            smoke,
            checkpoints,
            config,
            smoke_artifact=smoke_id,
            checkpoint_artifacts=checkpoint_ids,
            config_artifact=config_id,
            custody_artifacts=custody,
        ),
        adapt_pf3_checkpoints(
            pf_receipt,
            pf_checkpoints,
            receipt_artifact=pf_id,
            checkpoint_artifacts=pf_checkpoint_ids,
            custody_artifacts=pf_custody,
        ),
        adapt_j12_receipt(
            j12_receipt,
            pricing,
            receipt_artifact=j12_id,
            pricing_artifact=pricing_id,
            custody_artifacts=j12_custody,
        ),
    )


def _rehash_manifest(manifest: dict[str, Any]) -> None:
    manifest.pop("content_sha256", None)
    manifest["content_sha256"] = canonical_json_sha256(manifest)


def test_j8f_binds_all_sources_and_preserves_honest_debt(real_inputs: dict[str, Any]) -> None:
    result = _results(real_inputs)[0]

    assert result.ok is False
    assert [blocker.code for blocker in result.blockers] == [
        "J8F_APPLICATION_TO_CUMULATIVE_ARCHIVE_EDGE_ABSENT",
        "J8F_FINAL_CHANGED_UINT8_IDENTITY_ABSENT",
        "J8F_SINGLE_LAWFUL_BYTE_HOME_ABSENT",
        "J8F_PER_STEP_SCORE_TRANSITIONS_ABSENT",
        "J8F_REFERENCE_ARCHIVE_BYTES_ABSENT",
    ]
    assert dict(result.source_counts) == {
        "application_count": 12,
        "application_state_byte_mismatch_count": 1,
        "application_state_sha_mismatch_count": 11,
        "checkpoint_artifact_count": 12,
    }
    assert sum("application_step_" in artifact.path for artifact in result.source_artifacts) == 12


def test_j8f_cumulative_prefix_and_scorer_foreign_keys_fail_closed(
    real_inputs: dict[str, Any],
) -> None:
    smoke, checkpoints, config, smoke_id, checkpoint_ids, config_id, custody = real_inputs["j8"]
    changed_checkpoints = copy.deepcopy(checkpoints)
    changed_checkpoints[1]["application_receipts"][0]["score_claim"] = True
    changed_ids = list(checkpoint_ids)
    changed_ids[1] = replace(changed_ids[1], json_content_sha256=canonical_json_sha256(changed_checkpoints[1]))
    with pytest.raises(AppliedActionAdapterError, match="cumulative application content prefix"):
        adapt_j8f_checkpoints(
            smoke,
            changed_checkpoints,
            config,
            smoke_artifact=smoke_id,
            checkpoint_artifacts=changed_ids,
            config_artifact=config_id,
            custody_artifacts=custody,
        )

    changed_smoke = copy.deepcopy(smoke)
    changed_smoke["range_gauge_projected_arm"]["verdict"]["archive_bytes"] += 1
    changed_smoke_id = replace(smoke_id, json_content_sha256=canonical_json_sha256(changed_smoke))
    with pytest.raises(AppliedActionAdapterError, match="scorer-to-archive foreign key"):
        adapt_j8f_checkpoints(
            changed_smoke,
            checkpoints,
            config,
            smoke_artifact=changed_smoke_id,
            checkpoint_artifacts=checkpoint_ids,
            config_artifact=config_id,
            custody_artifacts=custody,
        )


def test_pf3_covers_all_16_artifacts_and_three_families(real_inputs: dict[str, Any]) -> None:
    result = _results(real_inputs)[1]

    assert result.ok is False
    assert dict(result.source_counts) == {
        "candidate_artifact_count": 16,
        "coordinate_family_count": 3,
        "uphill_edge_count": 16,
    }
    assert [blocker.code for blocker in result.blockers] == [
        "PF3_APPLICATION_OPERATOR_VERSION_ABSENT",
        "PF3_RECEIVER_SHA256_ABSENT",
        "PF3_CHANGED_UINT8_SHA256_ABSENT",
    ]
    assert PF3_PHYSICAL_EDGE == "V19C_BASE_TO_ONE_EXACT_RG3_COORDINATE"
    assert sum("stage_checkpoints/02_candidates" in item.path for item in result.source_artifacts) == 16
    assert "RD1" not in json.dumps(result.as_dict(), sort_keys=True)


def test_pf3_digest_chain_drift_fails_closed(real_inputs: dict[str, Any]) -> None:
    receipt, checkpoints, receipt_id, checkpoint_ids, custody = real_inputs["pf3"]
    changed = copy.deepcopy(receipt)
    changed["inventory"]["candidate_checkpoint_custody"]["digest_chain_sha256"] = "0" * 64
    changed_id = replace(receipt_id, json_content_sha256=canonical_json_sha256(changed))
    with pytest.raises(AppliedActionAdapterError, match="digest chain"):
        adapt_pf3_checkpoints(
            changed,
            checkpoints,
            receipt_artifact=changed_id,
            checkpoint_artifacts=checkpoint_ids,
            custody_artifacts=custody,
        )


def test_j12_validates_endpoints_and_preserves_reseal_warning(real_inputs: dict[str, Any]) -> None:
    result = _results(real_inputs)[2]

    assert result.ok is False
    assert dict(result.source_counts) == {
        "composite_count": 8,
        "sealed_proposal_count": 4,
        "single_count": 16,
    }
    assert [blocker.code for blocker in result.blockers] == [
        "J12_APPLICATION_OPERATOR_VERSION_ABSENT",
        "J12_RECEIVER_SHA256_ABSENT",
        "J12_CHANGED_UINT8_IDENTITY_ABSENT",
        "J12_SINGLE_LAWFUL_BYTE_HOME_ABSENT",
        "J12_MAIN_RESEAL_REVIEW_REQUIRED",
        "J12_PC1_NUMERICAL_WARNING_UNRESOLVED",
    ]


def test_j12_nonlinear_delta_drift_fails_closed(real_inputs: dict[str, Any]) -> None:
    receipt, pricing, receipt_id, pricing_id, custody = real_inputs["j12"]
    changed = copy.deepcopy(receipt)
    changed["pc1_adapter"]["step16"]["joint_delta"] += 0.25
    changed_id = replace(receipt_id, json_content_sha256=canonical_json_sha256(changed))
    with pytest.raises(AppliedActionAdapterError, match="compact step16 delta differs"):
        adapt_j12_receipt(
            changed,
            pricing,
            receipt_artifact=changed_id,
            pricing_artifact=pricing_id,
            custody_artifacts=custody,
        )


def test_manifest_validator_rejects_hash_count_authority_and_digest_drift(
    real_inputs: dict[str, Any],
) -> None:
    manifest = build_adapter_manifest(_results(real_inputs))
    assert manifest["schema"] == ADAPTER_MANIFEST_SCHEMA
    assert manifest["receipt_count"] == 0
    assert manifest["blocked_source_count"] == 3
    assert manifest["source_artifact_count"] > 100

    bad_hash = copy.deepcopy(manifest)
    bad_hash["content_sha256"] = "0" * 64
    with pytest.raises(AppliedActionAdapterError, match="content hash"):
        validate_adapter_manifest(bad_hash)

    bad_count = copy.deepcopy(manifest)
    bad_count["source_artifact_count"] -= 1
    _rehash_manifest(bad_count)
    with pytest.raises(AppliedActionAdapterError, match="artifact count"):
        validate_adapter_manifest(bad_count)

    bad_semantic_count = copy.deepcopy(manifest)
    bad_semantic_count["results"][1]["source_counts"]["candidate_artifact_count"] = 15
    _rehash_manifest(bad_semantic_count)
    with pytest.raises(AppliedActionAdapterError, match="semantic counts"):
        validate_adapter_manifest(bad_semantic_count)

    bad_authority = copy.deepcopy(manifest)
    bad_authority["promotion_eligible"] = True
    _rehash_manifest(bad_authority)
    with pytest.raises(AppliedActionAdapterError, match="false-authority"):
        validate_adapter_manifest(bad_authority)

    bad_digest = copy.deepcopy(manifest)
    bad_digest["results"][0]["source_artifact_digest_sha256"] = "0" * 64
    _rehash_manifest(bad_digest)
    with pytest.raises(AppliedActionAdapterError, match="artifact digest"):
        validate_adapter_manifest(bad_digest)


def test_file_identity_verifier_rejects_declared_hash_drift(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"real bytes")
    with pytest.raises(AppliedActionAdapterError, match="SHA-256 differs"):
        verify_source_artifact(source, role="test", expected_sha256="0" * 64)


def test_cli_materializes_the_same_blocker_only_shape(tmp_path: Path) -> None:
    if not (DEFAULT_J8F_ROOT / "ddm_j8f_counted_application_receipt.json").exists():
        pytest.skip("custodied SSD artifacts are not mounted")
    output = tmp_path / "manifest.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools/build_applied_action_receipts.py"),
            "--output",
            str(output),
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    summary = json.loads(completed.stdout)
    manifest = json.loads(output.read_text(encoding="utf-8"))
    validate_adapter_manifest(manifest)
    assert summary["content_sha256"] == manifest["content_sha256"]
    assert manifest["receipt_count"] == 0
    assert manifest["blocked_source_count"] == 3
    assert manifest["source_artifact_count"] == 145
