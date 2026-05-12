from __future__ import annotations

import math

from tac.auth_eval_result import (
    parse_finite_auth_eval_score,
    recompute_contest_score_from_payload,
)


def test_parse_finite_auth_eval_score_refuses_missing_score() -> None:
    assert parse_finite_auth_eval_score({"seg_dist": 0.1}) is None


def test_parse_finite_auth_eval_score_refuses_nan_score() -> None:
    assert parse_finite_auth_eval_score({"score": "nan"}) is None


def test_parse_finite_auth_eval_score_accepts_finite_score_without_components() -> None:
    parsed = parse_finite_auth_eval_score({"final_score": 0.19284758})
    assert parsed is not None
    assert parsed.score == 0.19284758
    assert parsed.source_key == "final_score"
    assert parsed.recomputed_score is None


def test_recompute_contest_score_from_components() -> None:
    payload = {
        "seg_dist": 0.001,
        "pose_dist": 0.0004,
        "archive_bytes": 150_000,
    }
    expected = 100.0 * 0.001 + math.sqrt(10.0 * 0.0004) + 25.0 * 150_000 / 37_545_489.0
    assert recompute_contest_score_from_payload(payload) == expected


def test_parse_finite_auth_eval_score_refuses_component_mismatch() -> None:
    payload = {
        "score": 0.5,
        "seg_dist": 0.001,
        "pose_dist": 0.0004,
        "archive_bytes": 150_000,
    }
    assert parse_finite_auth_eval_score(payload) is None


def test_parse_finite_auth_eval_score_accepts_component_match() -> None:
    payload = {
        "seg_dist": 0.001,
        "pose_dist": 0.0004,
        "archive_bytes": 150_000,
    }
    payload["score"] = recompute_contest_score_from_payload(payload)
    parsed = parse_finite_auth_eval_score(payload)
    assert parsed is not None
    assert parsed.recomputed_matches is True


def test_parse_prefers_canonical_score_and_canonical_component_aliases() -> None:
    payload = {
        "final_score": 0.19,
        "avg_segnet_dist": 0.001,
        "avg_posenet_dist": 0.0004,
        "rate_unscaled": 0.002,
    }
    payload["canonical_score"] = recompute_contest_score_from_payload(payload)

    parsed = parse_finite_auth_eval_score(payload, require_component_recompute=True)

    assert parsed is not None
    assert parsed.source_key == "canonical_score"
    assert parsed.score == payload["canonical_score"]


def test_parse_requires_component_recompute_when_requested() -> None:
    assert (
        parse_finite_auth_eval_score(
            {"final_score": 0.19284758},
            require_component_recompute=True,
        )
        is None
    )
