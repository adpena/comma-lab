from __future__ import annotations

import pytest

from tac.score_target_filter import (
    DEFAULT_SCORE_LOWERING_TARGET,
    decide_score_target_routing,
    parse_predicted_band,
)


def test_parse_predicted_band_accepts_sequence_and_string() -> None:
    assert parse_predicted_band((0.205, 0.208)) == (0.205, 0.208)
    assert parse_predicted_band("[0.208, 0.205]") == (0.205, 0.208)


def test_parse_predicted_band_rejects_malformed_string() -> None:
    with pytest.raises(ValueError):
        parse_predicted_band("0.205")


def test_decision_keeps_only_bands_that_can_beat_target() -> None:
    above = decide_score_target_routing((0.205, 0.208))
    plausible = decide_score_target_routing((0.16, 0.20))

    assert above.target_score == DEFAULT_SCORE_LOWERING_TARGET
    assert above.active is False
    assert above.status == "above_target"
    assert plausible.active is True
    assert plausible.status == "target_plausible"


def test_decision_fails_closed_on_unknown_band_when_requested() -> None:
    decision = decide_score_target_routing(None, keep_unknown=False)

    assert decision.active is False
    assert decision.status == "unknown_band"
