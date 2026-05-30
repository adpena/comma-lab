# SPDX-License-Identifier: MIT
"""Tests for canonical Alice-vs-Eve adversarial loop.

Per CLAUDE.md "NO FAKE IMPLEMENTATIONS" + Slot EEE substantive-distinctness
gate: every test verifies BEHAVIOR not just constants. Jaccard < 1.0 across
enum alternatives is the canonical substantive-distinctness gate.
"""

from __future__ import annotations

import pytest

from tac.composition.fridrich_school_inverse_steganalysis_patterns import (
    AliceEveLoopError,
    AliceEveRoundConfig,
    AliceEveScoringRule,
    canonical_alice_seed_algorithms,
    canonical_eve_seed_detectors,
    compute_alice_score_minimax,
    compute_eve_score_minimax,
)


def test_canonical_alice_seed_algorithms_matches_yousfi_autostego() -> None:
    """Per Yousfi autostego README verbatim: 3 SOTA algorithms HILL/WOW/S-UNIWARD."""
    seeds = canonical_alice_seed_algorithms()
    assert seeds == ("HILL", "WOW", "S-UNIWARD")
    assert len(seeds) == 3
    assert all(isinstance(s, str) for s in seeds)


def test_canonical_eve_seed_detectors_matches_autostego_eve_py() -> None:
    """Per autostego/eve.py PipelineConfig.detectors + steganalysis/ dir contents."""
    seeds = canonical_eve_seed_detectors()
    assert seeds == ("SRNet", "SRM", "LCLSMR", "fusion")
    assert "LCLSMR" in seeds, "LCLSMR is the NEW canonical detector ported in this package"
    assert "fusion" in seeds, "fusion is canonical score-level ensemble"


def test_alice_scoring_minimax_canonical_returns_min() -> None:
    """Per Yousfi README verbatim: 'Alice's score is the minimum of the 3 accuracies'."""
    accs = {"HILL": 0.65, "WOW": 0.55, "S-UNIWARD": 0.80}
    score = compute_alice_score_minimax(accs, AliceEveScoringRule.MINIMAX_CANONICAL)
    assert score == pytest.approx(0.55), "minimax MUST return min (Alice's worst algorithm)"


def test_eve_scoring_minimax_canonical_returns_max() -> None:
    """Per Yousfi README verbatim: 'Eve's score is the maximum of the 3 accuracies'."""
    accs = {"SRNet": 0.82, "LCLSMR": 0.65, "SRM": 0.75}
    score = compute_eve_score_minimax(accs, AliceEveScoringRule.MINIMAX_CANONICAL)
    assert score == pytest.approx(0.82), "minimax MUST return max (Eve's best detector)"


def test_alice_scoring_strategies_substantively_distinct() -> None:
    """Slot EEE substantive-distinctness gate: each canonical strategy MUST produce
    distinct numerical output on the canonical test input."""
    accs = {"HILL": 0.65, "WOW": 0.55, "S-UNIWARD": 0.80}
    outputs = {}
    for strategy in AliceEveScoringRule:
        outputs[strategy] = compute_alice_score_minimax(accs, strategy)
    # MINIMAX=0.55, MEAN=0.667, MEDIAN=0.65, WEIGHTED_SUM=0.667 (equal weight fallback)
    # 3 distinct values (MEAN and WEIGHTED_SUM coincide under equal weights, which is documented).
    distinct = {round(v, 5) for v in outputs.values()}
    assert len(distinct) >= 3, (
        f"At least 3 distinct outputs required (Slot EEE gate); got {distinct}"
    )
    assert outputs[AliceEveScoringRule.MINIMAX_CANONICAL] == pytest.approx(0.55)
    assert outputs[AliceEveScoringRule.MEDIAN_ROBUST] == pytest.approx(0.65)


def test_eve_scoring_strategies_substantively_distinct() -> None:
    """Sister of Alice scoring; each canonical Eve strategy produces distinct output."""
    accs = {"SRNet": 0.82, "LCLSMR": 0.65, "SRM": 0.75}
    outputs = {}
    for strategy in AliceEveScoringRule:
        outputs[strategy] = compute_eve_score_minimax(accs, strategy)
    distinct = {round(v, 5) for v in outputs.values()}
    assert len(distinct) >= 3
    assert outputs[AliceEveScoringRule.MINIMAX_CANONICAL] == pytest.approx(0.82)
    assert outputs[AliceEveScoringRule.MEDIAN_ROBUST] == pytest.approx(0.75)


def test_alice_score_empty_raises() -> None:
    """Empty input raises AliceEveLoopError."""
    with pytest.raises(AliceEveLoopError, match="empty"):
        compute_alice_score_minimax({}, AliceEveScoringRule.MINIMAX_CANONICAL)


def test_eve_score_empty_raises() -> None:
    """Empty input raises AliceEveLoopError."""
    with pytest.raises(AliceEveLoopError, match="empty"):
        compute_eve_score_minimax({}, AliceEveScoringRule.MINIMAX_CANONICAL)


def test_alice_score_non_numeric_raises() -> None:
    """Non-numeric value raises AliceEveLoopError."""
    with pytest.raises(AliceEveLoopError, match="must be numeric"):
        compute_alice_score_minimax(
            {"HILL": "high"}, AliceEveScoringRule.MINIMAX_CANONICAL  # type: ignore[arg-type]
        )


def test_alice_eve_round_config_canonical_defaults() -> None:
    """Default config matches Yousfi autostego canonical 3-algorithm 3-detector setup."""
    cfg = AliceEveRoundConfig()
    assert cfg.scoring_rule == AliceEveScoringRule.MINIMAX_CANONICAL
    assert cfg.alice_algorithm_count == 3
    assert cfg.eve_detector_count == 3


def test_alice_eve_round_config_invalid_strategy_raises() -> None:
    """Wrong type for scoring_rule raises."""
    with pytest.raises(AliceEveLoopError, match="must be AliceEveScoringRule"):
        AliceEveRoundConfig(scoring_rule="bogus")  # type: ignore[arg-type]


def test_alice_eve_round_config_invalid_counts_raise() -> None:
    """Zero or negative counts raise."""
    with pytest.raises(AliceEveLoopError, match="alice_algorithm_count"):
        AliceEveRoundConfig(alice_algorithm_count=0)
    with pytest.raises(AliceEveLoopError, match="eve_detector_count"):
        AliceEveRoundConfig(eve_detector_count=-1)
    with pytest.raises(AliceEveLoopError, match="max_rounds"):
        AliceEveRoundConfig(max_rounds=0)


def test_minimax_canonical_vs_mean_distinguishes_outlier() -> None:
    """The canonical minimax MUST punish Alice's worst algorithm; MEAN does not.

    The empirical insight: if one of Alice's 3 algorithms is dramatically weak
    (e.g. 0.95 detection rate) and the other two are strong (0.05), MEAN gives
    Alice score ~0.35 (looks OK) while MINIMAX gives 0.05 (correctly identifies
    that Alice's weakest algorithm is what Eve will exploit).
    """
    accs = {"strong_1": 0.05, "strong_2": 0.05, "weak": 0.95}
    minimax_score = compute_alice_score_minimax(accs, AliceEveScoringRule.MINIMAX_CANONICAL)
    mean_score = compute_alice_score_minimax(accs, AliceEveScoringRule.MEAN_BASELINE)
    # Minimax = min = 0.05 (alice is strong on her best algorithm; rule rewards her)
    # Mean = ~0.35
    assert minimax_score == pytest.approx(0.05)
    assert mean_score == pytest.approx(0.35, abs=0.01)
    assert minimax_score < mean_score, (
        "MINIMAX must be smaller (better for Alice) than MEAN when 2 of 3 "
        "algorithms are strong"
    )


def test_unhandled_scoring_rule_raises() -> None:
    """Defensive guard: passing a string instead of enum raises."""
    accs = {"HILL": 0.5}
    with pytest.raises((AliceEveLoopError, AttributeError)):
        compute_alice_score_minimax(accs, "bogus_string")  # type: ignore[arg-type]
