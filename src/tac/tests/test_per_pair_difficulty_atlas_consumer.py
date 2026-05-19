# SPDX-License-Identifier: MIT
"""Focused tests for per_pair_difficulty_atlas_consumer."""
from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from tac.cathedral.consumer_contract import HookNumber, validate_consumer_module
from tac.cathedral_consumers import per_pair_difficulty_atlas_consumer as consumer

if TYPE_CHECKING:
    import pytest


ARCHIVE_SHA = "a" * 64


def _write_per_pair_anchor(
    tmp_path: Path,
    *,
    archive_sha: str = ARCHIVE_SHA,
    pair_scores: list[float] | None = None,
) -> tuple[Path, Path, np.ndarray, dict[str, object]]:
    arr = np.zeros((2, 3, 3), dtype=np.float32)
    arr[:, 0, 0] = [0.5, 0.5]  # norm ~= 0.707, weighted by score in tests
    arr[:, 1, 1] = [4.0, 4.0]  # norm ~= 5.657
    arr[:, 2, 2] = [1.0, 1.0]  # norm ~= 1.414
    gradient_path = tmp_path / "per_pair_gradient.npy"
    np.save(gradient_path, arr)
    anchor: dict[str, object] = {
        "archive_sha256": archive_sha,
        "gradient_array_path": str(gradient_path),
        "gradient_tensor_kind": "per_pair_per_byte_v1",
        "measurement_axis": "[macOS-CPU advisory]",
        "measurement_hardware": "darwin_arm64_m5_max_macos_cpu_advisory",
        "measurement_method": "autograd_per_pair_subset_axis_corrected",
        "measurement_call_id": "local-call-123",
        "measurement_utc": "2026-05-19T14:00:00Z",
        "n_bytes": 2,
        "n_pairs": 3,
        "n_pairs_used": 3,
        "n_pairs_total": 600,
        "operating_point": {
            "d_pose": 0.1,
            "d_seg": 0.1,
            "rate": 0.1,
            "score": 0.1,
        },
        "schema_version": "master_gradient_anchor_v1",
    }
    if pair_scores is not None:
        anchor["per_pair_scores"] = pair_scores
    ledger = tmp_path / "master_gradient_anchors.jsonl"
    ledger.write_text(json.dumps(anchor) + "\n", encoding="utf-8")
    return ledger, gradient_path, arr, anchor


def test_consumer_module_exposes_canonical_contract() -> None:
    registration = validate_consumer_module(
        consumer,
        module_path="tac.cathedral_consumers.per_pair_difficulty_atlas_consumer",
    )

    assert registration.contract_compliant is True
    assert registration.validation_errors == ()


def test_contract_constants_mark_hook_5_active() -> None:
    assert consumer.CONSUMER_NAME == "per_pair_difficulty_atlas_consumer"
    assert consumer.CONSUMER_VERSION == "0.1.0"
    assert HookNumber.SENSITIVITY_MAP in consumer.CONSUMER_HOOK_NUMBERS
    assert HookNumber.CATHEDRAL_AUTOPILOT_DISPATCH in consumer.CONSUMER_HOOK_NUMBERS
    assert HookNumber.CONTINUAL_LEARNING_POSTERIOR in consumer.CONSUMER_HOOK_NUMBERS
    assert HookNumber.PARETO_CONSTRAINT not in consumer.CONSUMER_HOOK_NUMBERS


def test_posterior_wire_status_fails_closed_for_predicted_per_pair_schema() -> None:
    status = consumer.posterior_wire_status()

    assert status["helper"] == "tac.continual_learning.posterior_update_locked"
    assert status["helper_callable"] is True
    assert status["predicted_per_pair_anchor_supported"] is False
    assert status["direct_posterior_jsonl_mutation"] is False
    assert "ContestResult" in status["reason"]


def test_posterior_wire_status_missing_helper_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_import_module = consumer.importlib.import_module

    def missing_module(name: str):
        if name == "tac.continual_learning":
            raise ImportError("simulated missing continual-learning helper")
        return original_import_module(name)

    monkeypatch.setattr(consumer.importlib, "import_module", missing_module)

    status = consumer.posterior_wire_status()

    assert status["helper_available"] is False
    assert status["predicted_per_pair_anchor_supported"] is False
    assert status["status"] == "fail_closed_helper_missing"
    assert status["direct_posterior_jsonl_mutation"] is False


def test_missing_anchor_returns_no_signal_and_does_not_mutate_jsonl(
    tmp_path: Path,
) -> None:
    ledger = tmp_path / "empty_master_gradient_anchors.jsonl"
    ledger.write_text("", encoding="utf-8")
    forbidden_jsonl = tmp_path / "continual_learning_posterior.jsonl"

    verdict = consumer.consume_candidate(
        {
            "archive_sha256": ARCHIVE_SHA,
            "master_gradient_anchor_path": str(ledger),
        }
    )

    assert verdict["predicted_delta_adjustment"] == 0.0
    assert verdict["promotable"] is False
    assert verdict["score_claim"] is False
    assert verdict["axis_tag"] == "[predicted]"
    assert "no usable per-pair master-gradient anchor" in verdict["rationale"]
    assert not forbidden_jsonl.exists()


def test_non_mapping_candidate_returns_no_signal() -> None:
    verdict = consumer.consume_candidate("not-a-candidate")  # type: ignore[arg-type]

    assert verdict["predicted_delta_adjustment"] == 0.0
    assert verdict["promotable"] is False
    assert verdict["axis_tag"] == "[predicted]"
    assert verdict["score_claim"] is False


def test_consume_candidate_computes_deterministic_pair_score_weighted_order(
    tmp_path: Path,
) -> None:
    ledger, _gradient_path, _arr, _anchor = _write_per_pair_anchor(tmp_path)

    verdict = consumer.consume_candidate(
        {
            "archive_sha256": ARCHIVE_SHA,
            "master_gradient_anchor_path": str(ledger),
            "pair_scores": [10.0, 1.0, 2.0],
            "top_k": 3,
        }
    )

    payload = verdict["notes"]["per_pair_difficulty_atlas"]
    assert verdict["predicted_delta_adjustment"] == 0.0
    assert verdict["promotable"] is False
    assert verdict["axis_tag"] == "[predicted]"
    assert payload["posterior_mutation_performed"] is False
    assert payload["posterior_update_status"] == "payload_only_canonical_helper_mismatch"
    assert payload["pair_score_source"] == "candidate.pair_scores"
    assert [row["pair_index"] for row in payload["all_pairs"]] == [0, 1, 2]
    assert payload["all_pairs"][0]["difficulty_score"] > payload["all_pairs"][1][
        "difficulty_score"
    ]
    assert payload["all_pairs"][1]["difficulty_score"] > payload["all_pairs"][2][
        "difficulty_score"
    ]


def test_difficulty_ties_break_by_pair_index() -> None:
    arr = np.ones((2, 3, 3), dtype=np.float32)

    payload = consumer.build_predicted_difficulty_payload(
        arr,
        archive_sha256=ARCHIVE_SHA,
        source_anchor={
            "gradient_tensor_kind": "per_pair_per_byte_v1",
            "measurement_axis": "[predicted]",
        },
        candidate={"pair_scores": [1.0, 1.0, 1.0], "top_k": 3},
    )

    assert [row["pair_index"] for row in payload["all_pairs"]] == [0, 1, 2]
    assert [row["difficulty_rank"] for row in payload["all_pairs"]] == [0, 1, 2]


def test_anchor_pair_scores_are_used_when_candidate_scores_absent(
    tmp_path: Path,
) -> None:
    ledger, _gradient_path, _arr, _anchor = _write_per_pair_anchor(
        tmp_path,
        pair_scores=[1.0, 0.2, 5.0],
    )

    verdict = consumer.consume_candidate(
        {
            "archive_sha256": ARCHIVE_SHA,
            "master_gradient_anchor_path": str(ledger),
            "top_k": 3,
        }
    )

    payload = verdict["notes"]["per_pair_difficulty_atlas"]
    assert payload["pair_score_source"] == "anchor.per_pair_scores"
    assert [row["pair_index"] for row in payload["all_pairs"]] == [2, 1, 0]


def test_missing_pair_scores_use_unit_weights_with_explicit_source(
    tmp_path: Path,
) -> None:
    ledger, _gradient_path, _arr, _anchor = _write_per_pair_anchor(tmp_path)

    verdict = consumer.consume_candidate(
        {
            "archive_sha256": ARCHIVE_SHA,
            "master_gradient_anchor_path": str(ledger),
            "top_k": 3,
        }
    )

    payload = verdict["notes"]["per_pair_difficulty_atlas"]
    assert payload["pair_score_source"] == "unit_default_no_pair_score_field"
    assert [row["pair_index"] for row in payload["all_pairs"]] == [1, 2, 0]


def test_update_from_anchor_builds_predicted_payload_without_empirical_authority(
    tmp_path: Path,
) -> None:
    _ledger, _gradient_path, _arr, anchor = _write_per_pair_anchor(
        tmp_path,
        pair_scores=[10.0, 1.0, 2.0],
    )

    result = consumer.update_from_anchor(anchor)

    assert result["accepted"] is False
    assert result["status"] == "payload_only_canonical_helper_mismatch"
    assert result["axis_tag"] == "[predicted]"
    assert result["score_claim"] is False
    assert result["promotion_eligible"] is False
    payload = result["predicted_anchor_payload"]
    assert payload["empirical_anchor"] is False
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False


def test_update_from_anchor_rejects_non_per_pair_anchor(tmp_path: Path) -> None:
    _ledger, _gradient_path, _arr, anchor = _write_per_pair_anchor(tmp_path)
    anchor["gradient_tensor_kind"] = "aggregate_per_byte_v1"

    result = consumer.update_from_anchor(anchor)

    assert result["accepted"] is False
    assert result["status"] == "fail_closed"
    assert "per_pair_per_byte_v1" in result["reason"]
    assert result["score_claim"] is False


def test_update_from_anchor_missing_gradient_file_fails_closed(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "missing.npy"

    result = consumer.update_from_anchor(
        {
            "archive_sha256": ARCHIVE_SHA,
            "gradient_array_path": str(missing),
            "gradient_tensor_kind": "per_pair_per_byte_v1",
        }
    )

    assert result["accepted"] is False
    assert result["status"] == "fail_closed"
    assert result["posterior_mutation_performed"] is False
    assert "not found" in result["reason"]


def test_invalid_pair_score_vector_fails_closed(tmp_path: Path) -> None:
    ledger, _gradient_path, _arr, _anchor = _write_per_pair_anchor(tmp_path)

    verdict = consumer.consume_candidate(
        {
            "archive_sha256": ARCHIVE_SHA,
            "master_gradient_anchor_path": str(ledger),
            "pair_scores": [1.0, 2.0],
        }
    )

    assert verdict["predicted_delta_adjustment"] == 0.0
    assert verdict["promotable"] is False
    assert "pair score vector length" in verdict["rationale"]


def test_posterior_update_locked_is_not_called_for_predicted_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import tac.continual_learning as continual_learning

    ledger, _gradient_path, _arr, _anchor = _write_per_pair_anchor(tmp_path)

    def forbidden_call(*_args, **_kwargs):  # pragma: no cover - should not run
        raise AssertionError("posterior_update_locked must not be called")

    monkeypatch.setattr(
        continual_learning,
        "posterior_update_locked",
        forbidden_call,
    )

    verdict = consumer.consume_candidate(
        {
            "archive_sha256": ARCHIVE_SHA,
            "master_gradient_anchor_path": str(ledger),
            "pair_scores": [10.0, 1.0, 2.0],
        }
    )

    payload = verdict["notes"]["per_pair_difficulty_atlas"]
    assert payload["posterior_wire_status"]["helper_callable"] is True
    assert payload["posterior_wire_status"][
        "predicted_per_pair_anchor_supported"
    ] is False
