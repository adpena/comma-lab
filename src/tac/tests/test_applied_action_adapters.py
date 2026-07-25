# SPDX-License-Identifier: MIT
"""Real-artifact tests for the fail-closed applied-action adapters."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.analysis.applied_action_adapters import (
    ADAPTER_MANIFEST_SCHEMA,
    PF3_PHYSICAL_EDGE,
    AppliedActionAdapterError,
    adapt_j8f_checkpoints,
    adapt_j12_receipt,
    adapt_pf3_checkpoint,
    build_adapter_manifest,
    load_json,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
J8F_ROOT = Path(
    "/Volumes/VertigoDataTier/pact/experiments/results/"
    "ddm_j8f_counted_application_20260724T181414Z"
)
J8F_RECEIPT = J8F_ROOT / "ddm_j8f_counted_application_receipt.json"
J8F_CONFIG = REPO_ROOT / ".omx/research/configs/ddm_j8f_counted_application_20260724.json"
PF3_RECEIPT = (
    REPO_ROOT / ".omx/research/ddm_pf3_finite_price_materialization_20260725T193409Z/receipt.json"
)
J12_RECEIPT = (
    REPO_ROOT / ".omx/research/ddm_j12_366_receiver_coordinate_custody_receipt_20260725.json"
)


def _j8f_inputs() -> tuple[dict[str, object], list[dict[str, object]], dict[str, object]]:
    if not J8F_RECEIPT.exists():
        pytest.skip("custodied J8F SSD artifacts are not mounted")
    paths = sorted((J8F_ROOT / "checkpoints").glob("application_step_*.json"))
    return (
        dict(load_json(J8F_RECEIPT)),
        [dict(load_json(path)) for path in paths],
        dict(load_json(J8F_CONFIG)),
    )


def _pf3_inputs() -> tuple[dict[str, object], dict[str, object], Path]:
    receipt = dict(load_json(PF3_RECEIPT))
    inventory = receipt["inventory"]
    assert isinstance(inventory, dict)
    custody = inventory["candidate_checkpoint_custody"]
    assert isinstance(custody, dict)
    artifacts = custody["artifacts"]
    assert isinstance(artifacts, list)
    checkpoint_path = Path(artifacts[0]["path"])
    if not checkpoint_path.exists():
        pytest.skip("custodied PF3 SSD artifacts are not mounted")
    return receipt, dict(load_json(checkpoint_path)), checkpoint_path


def test_j8f_real_chain_blocks_inference_from_ordered_incidences() -> None:
    smoke, checkpoints, config = _j8f_inputs()
    results = adapt_j8f_checkpoints(smoke, checkpoints, config)

    assert len(results) == 1
    assert results[0].ok is False
    assert [blocker.code for blocker in results[0].blockers] == [
        "J8F_FINAL_CHANGED_UINT8_IDENTITY_ABSENT",
        "J8F_SINGLE_LAWFUL_BYTE_HOME_ABSENT",
        "J8F_PER_STEP_SCORE_TRANSITIONS_ABSENT",
    ]
    assert results[0].receipt is None


def test_j8f_adapter_is_deterministic_and_schema_drift_fails_closed() -> None:
    smoke, checkpoints, config = _j8f_inputs()
    first = adapt_j8f_checkpoints(smoke, checkpoints, config)[0].as_dict()
    second = adapt_j8f_checkpoints(smoke, checkpoints, config)[0].as_dict()
    assert first == second

    checkpoints[0]["schema"] = "foreign.schema"
    with pytest.raises(AppliedActionAdapterError, match="checkpoint schema differs"):
        adapt_j8f_checkpoints(smoke, checkpoints, config)


def test_pf3_real_edge_stays_physical_and_blocks_missing_change_identity() -> None:
    receipt, checkpoint, checkpoint_path = _pf3_inputs()
    result = adapt_pf3_checkpoint(receipt, checkpoint, source_id=str(checkpoint_path))

    assert result.ok is False
    assert [blocker.code for blocker in result.blockers] == [
        "PF3_APPLICATION_OPERATOR_VERSION_ABSENT",
        "PF3_RECEIVER_SHA256_ABSENT",
        "PF3_CHANGED_UINT8_SHA256_ABSENT",
    ]
    assert result.blockers[-1].owed_field == "changed_uint8_sha256"
    assert PF3_PHYSICAL_EDGE == "V19C_BASE_TO_ONE_EXACT_RG3_COORDINATE"
    assert "RD1" not in json.dumps(result.as_dict(), sort_keys=True)


def test_j12_real_receipt_reports_each_missing_foreign_key() -> None:
    result = adapt_j12_receipt(load_json(J12_RECEIPT), source_id=str(J12_RECEIPT))

    assert result.ok is False
    assert [blocker.code for blocker in result.blockers] == [
        "J12_APPLICATION_OPERATOR_VERSION_ABSENT",
        "J12_RECEIVER_SHA256_ABSENT",
        "J12_CHANGED_UINT8_IDENTITY_ABSENT",
        "J12_SINGLE_LAWFUL_BYTE_HOME_ABSENT",
    ]


def test_manifest_is_deterministic_and_never_claims_authority() -> None:
    smoke, checkpoints, config = _j8f_inputs()
    pf3_receipt, pf3_checkpoint, checkpoint_path = _pf3_inputs()
    results = (
        *adapt_j8f_checkpoints(smoke, checkpoints, config),
        adapt_pf3_checkpoint(pf3_receipt, pf3_checkpoint, source_id=str(checkpoint_path)),
        adapt_j12_receipt(load_json(J12_RECEIPT), source_id=str(J12_RECEIPT)),
    )

    first = build_adapter_manifest(results)
    second = build_adapter_manifest(results)
    assert first == second
    assert first["schema"] == ADAPTER_MANIFEST_SCHEMA
    assert first["receipt_count"] == 0
    assert first["blocked_source_count"] == 3
    assert first["research_only"] is True
    assert first["promotion_eligible"] is False
    assert first["score_claim"] is False
    assert len(first["content_sha256"]) == 64


def test_cli_materializes_the_same_real_manifest(tmp_path: Path) -> None:
    if not J8F_RECEIPT.exists():
        pytest.skip("custodied J8F SSD artifacts are not mounted")
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
    assert summary["content_sha256"] == manifest["content_sha256"]
    assert manifest["receipt_count"] == 0
    assert manifest["blocked_source_count"] == 3
