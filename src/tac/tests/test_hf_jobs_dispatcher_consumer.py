# SPDX-License-Identifier: MIT
"""Tests for the HF Jobs cathedral routing consumer."""
from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

from tac.cathedral.consumer_contract import (
    CathedralConsumerContract,
    HookNumber,
    validate_consumer_module,
)

PKG_PATH = "tac.cathedral_consumers.hf_jobs_dispatcher_consumer"


@pytest.fixture
def consumer_module():
    return importlib.import_module(PKG_PATH)


def _valid_minimal_candidate() -> dict[str, object]:
    return {
        "candidate_id": "hf_jobs_route_candidate",
        "axis_tag": "[predicted]",
        "hf_jobs": {
            "script": "experiments/hf_jobs_segnet_surrogate_distillation.py",
            "hub_dataset_repo": "adpena/comma-video-segnet-image-level-600pairs",
            "hub_model_repo": "adpena/segnet-image-level-surrogate-mobilenet-v3-small-200ep",
            "flavor": "t4-small",
            "lane_id": "lane_hf_jobs_segnet_surrogate_distillation_20260519",
        },
    }


def test_consumer_contract_constants(consumer_module) -> None:
    """Module-level contract constants match Catalog #335 expectations."""
    assert consumer_module.CONSUMER_NAME == "hf_jobs_dispatcher_consumer"
    assert consumer_module.CONSUMER_VERSION == "0.1.0"
    assert consumer_module.ROUTE_HF_JOBS == "hf_jobs"
    assert consumer_module.ROUTE_NONE == "none"
    assert (
        consumer_module.HF_JOBS_HELPER_MODULE
        == "tac.deploy.hf_jobs.job_id_ledger"
    )
    assert (
        consumer_module.HF_JOBS_DISPATCHER_CLI
        == "tools/dispatch_hf_jobs_vision_training.py"
    )


def test_consumer_satisfies_cathedral_contract(consumer_module) -> None:
    """The package is auto-discovery compatible."""
    assert isinstance(consumer_module, CathedralConsumerContract)
    reg = validate_consumer_module(consumer_module, module_path=PKG_PATH)
    assert reg.contract_compliant is True
    assert reg.validation_errors == ()


def test_consumer_declares_hook_4_active(consumer_module) -> None:
    """HF Jobs routing is a Catalog #125 hook #4 consumer."""
    assert consumer_module.CONSUMER_HOOK_NUMBERS == (
        HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH,
    )


def test_missing_hf_jobs_helper_fails_closed_no_dispatch_no_score(
    consumer_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Absent canonical HF Jobs helper yields no route and no score claim."""
    monkeypatch.setattr(
        consumer_module,
        "HF_JOBS_HELPER_MODULE",
        "tac.deploy.hf_jobs.missing_job_id_ledger",
    )

    result = consumer_module.consume_candidate(_valid_minimal_candidate())

    assert result["recommended_route"] == consumer_module.ROUTE_NONE
    assert result["predicted_delta_adjustment"] == 0.0
    assert result["axis_tag"] == "[predicted]"
    assert result["promotable"] is False
    assert result["score_claim"] is False
    assert result["promotion_eligible"] is False
    assert result["dispatch_ready"] is False
    assert result["ready_for_exact_eval_dispatch"] is False

    notes = result["notes"]["hf_jobs_dispatcher_consumer"]
    assert notes["accepted"] is False
    assert notes["failure_class"] == "missing_hf_jobs_helper"
    assert "module:tac.deploy.hf_jobs.missing_job_id_ledger" in notes[
        "helper_status"
    ]["missing_helpers"]


def test_valid_minimal_candidate_emits_hf_jobs_route_notes_without_claims(
    consumer_module,
) -> None:
    """A minimal HF Jobs plan surfaces routing notes, not promotion authority."""
    result = consumer_module.consume_candidate(_valid_minimal_candidate())

    assert result["recommended_route"] == consumer_module.ROUTE_HF_JOBS
    assert result["routing_target"] == consumer_module.ROUTE_HF_JOBS
    assert result["predicted_delta_adjustment"] == 0.0
    assert result["axis_tag"] == "[predicted]"
    assert result["promotable"] is False
    assert result["score_claim"] is False
    assert result["promotion_eligible"] is False
    assert result["dispatch_ready"] is False
    assert result["ready_for_exact_eval_dispatch"] is False

    route = result["routing_target_candidate"]
    assert route["route"] == consumer_module.ROUTE_HF_JOBS
    assert route["consumer_dispatches_jobs"] is False
    assert route["operator_authorize_required"] is True
    assert "modal" in route["route_competes_with"]
    assert "lightning" in route["route_competes_with"]
    assert "vastai" in route["route_competes_with"]
    assert "local_mps" in route["route_competes_with"]
    assert "local_cpu" in route["route_competes_with"]
    assert route["plan"]["platform"] == "hf_jobs"
    assert route["plan"]["flavor"] == "t4-small"
    assert (
        route["plan"]["hub_dataset_repo"]
        == "adpena/comma-video-segnet-image-level-600pairs"
    )
    assert (
        route["plan"]["hub_model_repo"]
        == "adpena/segnet-image-level-surrogate-mobilenet-v3-small-200ep"
    )
    assert route["plan"]["defaulted_fields"] == []


def test_hf_jobs_marker_candidate_uses_canonical_defaults_without_dispatch(
    consumer_module,
) -> None:
    """Autopilot's reduced CandidateRow payload can still surface HF Jobs."""
    result = consumer_module.consume_candidate(
        {
            "candidate_id": "lane_hf_jobs_segnet_surrogate_distillation_20260519",
            "axis_tag": "[predicted]",
        }
    )

    assert result["recommended_route"] == consumer_module.ROUTE_HF_JOBS
    assert result["score_claim"] is False
    assert result["promotion_eligible"] is False
    assert result["dispatch_ready"] is False
    assert result["ready_for_exact_eval_dispatch"] is False

    route = result["routing_target_candidate"]
    assert route["consumer_dispatches_jobs"] is False
    assert route["operator_authorize_required"] is True
    assert route["plan"]["script"] == consumer_module.DEFAULT_HF_JOBS_SCRIPT
    assert (
        route["plan"]["hub_dataset_repo"]
        == consumer_module.DEFAULT_HF_JOBS_HUB_DATASET_REPO
    )
    assert (
        route["plan"]["hub_model_repo"]
        == consumer_module.DEFAULT_HF_JOBS_HUB_MODEL_REPO
    )
    assert route["plan"]["flavor"] == consumer_module.DEFAULT_HF_JOBS_FLAVOR
    assert route["plan"]["defaulted_fields"] == [
        "script",
        "hub_dataset_repo",
        "hub_model_repo",
    ]


def test_non_hf_jobs_candidate_without_config_fails_closed(
    consumer_module,
) -> None:
    """Canonical defaults are only applied to explicit HF Jobs route markers."""
    result = consumer_module.consume_candidate(
        {
            "candidate_id": "ordinary_modal_candidate",
            "axis_tag": "[predicted]",
        }
    )

    assert result["recommended_route"] == consumer_module.ROUTE_NONE
    notes = result["notes"]["hf_jobs_dispatcher_consumer"]
    assert notes["failure_class"] == "missing_required_hf_jobs_fields"
    assert notes["route_candidate"]["missing_fields"] == [
        "script",
        "hub_dataset_repo",
        "hub_model_repo",
    ]


def test_live_autopilot_invoker_surfaces_hf_jobs_consumer_route() -> None:
    """Regression: auto-discovery path passes only reduced CandidateRow fields."""
    from tools.cathedral_autopilot_autonomous_loop import (
        CandidateRow,
        invoke_cathedral_consumers_on_candidates,
    )

    candidate = CandidateRow(
        candidate_id="lane_hf_jobs_segnet_surrogate_distillation_20260519",
        family="hf_jobs",
        predicted_score_delta=0.0,
        expected_information_gain=0.0,
        estimated_dispatch_cost_usd=0.40,
    )

    payload = invoke_cathedral_consumers_on_candidates(
        [candidate], top_n=1, repo_root="."
    )
    rows = [
        row
        for row in payload["invocations"]
        if row.get("consumer_name") == "hf_jobs_dispatcher_consumer"
    ]

    assert len(rows) == 1
    row = rows[0]
    assert row["axis_tag"] == "[predicted]"
    assert row["promotable"] is False
    assert row["predicted_delta_adjustment"] == 0.0
    assert "HF Jobs route candidate available" in row["rationale"]
    assert "missing required field" not in row["rationale"]


def test_missing_required_hf_jobs_fields_fail_closed(consumer_module) -> None:
    """Incomplete candidate metadata does not produce an HF Jobs route."""
    result = consumer_module.consume_candidate(
        {
            "axis_tag": "[predicted]",
            "hf_jobs": {"script": "experiments/hf_jobs_segnet_surrogate_distillation.py"},
        }
    )

    assert result["recommended_route"] == consumer_module.ROUTE_NONE
    notes = result["notes"]["hf_jobs_dispatcher_consumer"]
    assert notes["failure_class"] == "missing_required_hf_jobs_fields"
    assert notes["route_candidate"]["missing_fields"] == [
        "hub_dataset_repo",
        "hub_model_repo",
    ]
    assert result["score_claim"] is False


def test_consumer_never_calls_dispatch_or_ledger_mutation_helpers(
    consumer_module, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Helper presence is inspected, but spend/ledger mutation helpers are not called."""

    def forbidden_call(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise AssertionError("consumer must not call dispatch or ledger mutation helpers")

    fake_helper = SimpleNamespace(
        HF_JOBS_CALL_ID_LEDGER_PATH=".omx/state/hf_jobs_call_id_ledger.jsonl",
        SCHEMA_VERSION="fake_schema_for_test",
        register_dispatched_hf_jobs_id_fail_closed=forbidden_call,
        poll_ledger_for_hf_jobs_id=forbidden_call,
        query_by_lane=forbidden_call,
    )

    original_import = consumer_module.importlib.import_module

    def fake_import_module(name: str):
        if name == consumer_module.HF_JOBS_HELPER_MODULE:
            return fake_helper
        return original_import(name)

    monkeypatch.setattr(consumer_module.importlib, "import_module", fake_import_module)

    result = consumer_module.consume_candidate(_valid_minimal_candidate())

    assert result["recommended_route"] == consumer_module.ROUTE_HF_JOBS
    assert result["routing_target_candidate"]["ledger_schema_version"] == (
        "fake_schema_for_test"
    )
    assert result["dispatch_ready"] is False
