from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.cathedral_autopilot_autonomous_loop import (
    CandidateRow,
    _coerce_consumer_payload,
    _invoke_consumer_safely,
    load_candidates_from_jsonl,
    load_candidates_from_substrate_composition_ranking,
)


class _PayloadEchoConsumer:
    __name__ = "payload_echo_consumer"
    CONSUMER_NAME = "payload_echo_consumer"
    CONSUMER_VERSION = "0.1.0"

    @staticmethod
    def consume_candidate(payload: dict) -> dict:
        return {
            "predicted_delta_adjustment": 0.0,
            "rationale": (
                f"procedural={isinstance(payload.get('procedural_codebook_savings_candidate'), dict)} "
                f"per_frame={isinstance(payload.get('per_frame_decomposition'), dict)}"
            ),
            "axis_tag": "[predicted]",
            "promotable": False,
            "confidence": 0.0,
        }


class _UnsafeTierAConsumer:
    __name__ = "unsafe_tier_a_consumer"
    CONSUMER_NAME = "unsafe_tier_a_consumer"
    CONSUMER_VERSION = "0.1.0"

    @staticmethod
    def consume_candidate(payload: dict) -> dict:
        return {
            "predicted_delta_adjustment": -0.001,
            "rationale": "unsafe",
            "axis_tag": "[predicted]",
            "promotable": True,
            "confidence": 1.0,
        }


def _consumer_payload() -> dict:
    return {
        "procedural_codebook_savings_candidate": {
            "substrate_id": "test",
            "n_codebook_bytes": 4096,
            "k_seed_bytes": 16,
            "affected_frame_indices": [1],
        },
        "per_frame_decomposition": {
            "topology": "non_overlapping",
            "n_pairs": 1,
            "n_frames": 2,
            "top_frames": [{"rank": 1, "frame_index": 1, "total_l1": 1.0}],
        },
    }


def test_candidate_row_consumer_payload_reaches_consumer() -> None:
    candidate = CandidateRow(
        candidate_id="c",
        family="test",
        predicted_score_delta=0.0,
        expected_information_gain=0.0,
        estimated_dispatch_cost_usd=0.0,
        consumer_payload=_consumer_payload(),
    )

    out = _invoke_consumer_safely(_PayloadEchoConsumer, candidate)

    assert "error" not in out
    assert out["rationale"] == "procedural=True per_frame=True"


def test_tier_a_consumer_output_contract_is_enforced() -> None:
    candidate = CandidateRow(
        candidate_id="c",
        family="test",
        predicted_score_delta=0.0,
        expected_information_gain=0.0,
        estimated_dispatch_cost_usd=0.0,
    )

    out = _invoke_consumer_safely(_UnsafeTierAConsumer, candidate)

    assert out["error"].startswith("consumer_contract_violation:")
    assert "Tier A predicted_delta_adjustment must be 0.0" in out["error"]
    assert "Tier A promotable must be False" in out["error"]


def test_load_candidates_from_jsonl_preserves_consumer_payload(tmp_path: Path) -> None:
    path = tmp_path / "candidates.jsonl"
    row = {
        "candidate_id": "c",
        "family": "test",
        "predicted_score_delta": 0.0,
        "expected_information_gain": 0.0,
        "estimated_dispatch_cost_usd": 0.01,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "consumer_payload": _consumer_payload(),
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    candidates = load_candidates_from_jsonl(path)

    assert candidates[0].consumer_payload["procedural_codebook_savings_candidate"][
        "affected_frame_indices"
    ] == [1]


def test_load_candidates_from_jsonl_rejects_root_authority_flags(tmp_path: Path) -> None:
    path = tmp_path / "candidates.jsonl"
    row = {
        "candidate_id": "c",
        "family": "test",
        "predicted_score_delta": 0.0,
        "expected_information_gain": 0.0,
        "estimated_dispatch_cost_usd": 0.01,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": True,
    }
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="rank_or_kill_eligible=truthy"):
        load_candidates_from_jsonl(path)


def test_substrate_composition_ranking_rejects_root_authority_flags(
    tmp_path: Path,
) -> None:
    path = tmp_path / "substrate_ranking.json"
    payload = {
        "schema": "tac_autopilot_dispatch_ranking_v1",
        "score_claim": False,
        "ranked_dispatches": [
            {
                "candidate_id": "c",
                "family": "test",
                "predicted_score_delta": -0.001,
                "expected_information_gain": 0.0,
                "estimated_dispatch_cost_usd": 0.01,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "gpu_launched": True,
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="gpu_launched=truthy"):
        load_candidates_from_substrate_composition_ranking(path)


def test_cathedral_consumer_payload_rejects_rank_or_promote_authority() -> None:
    for key in (
        "rank_or_kill_eligible",
        "promotable",
        "score_claim_valid",
        "dispatch_packet_ready",
    ):
        with pytest.raises(ValueError, match=key):
            _coerce_consumer_payload(
                {"consumer_payload": {key: True}},
                context="fixture",
            )


def test_cathedral_consumer_payload_rejects_nested_authority() -> None:
    with pytest.raises(
        ValueError,
        match=r"optimizer_recipe\.ready_for_exact_eval_dispatch=truthy",
    ):
        _coerce_consumer_payload(
            {
                "consumer_payload": {
                    "optimizer_recipe": {
                        "id": "unsafe",
                        "ready_for_exact_eval_dispatch": True,
                    }
                }
            },
            context="fixture",
        )
