# SPDX-License-Identifier: MIT
"""Isolated tests for lane_gain_chain_composed_20260716 (locked-registry pattern)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.canonical_equations.equation import CanonicalEquation
from tac.canonical_equations.lane_gain_chain_composed_20260716 import (
    COMPOSED_GAIN_MED,
    EQUATION_ID,
    LANE_ISLAND_FRAC_BELOW_05,
    LANE_ISLAND_PERSISTENCE_MEDIAN,
    SKIP_GAIN_RATIO_MED,
    build_lane_gain_chain_composed_v1,
    perpair_lambda_prior,
)

REPO_ROOT = Path(__file__).resolve().parents[3].parent
S1 = REPO_ROOT / "experiments/results/lane_channel_refactor_20260716/s1_gain_chain.json"
S3 = REPO_ROOT / "experiments/results/lane_channel_refactor_20260716/s3_dash_geometry.json"


def test_lambda_prior_basic() -> None:
    assert perpair_lambda_prior(0.0212, 0.0471) == pytest.approx(0.0471 / 0.0212)
    assert perpair_lambda_prior(2.0, 2.0) == pytest.approx(1.0)


def test_lambda_prior_rejects_nonpositive() -> None:
    with pytest.raises(ValueError):
        perpair_lambda_prior(0.0, 1.0)
    with pytest.raises(ValueError):
        perpair_lambda_prior(1.0, -1.0)


def test_gain_inversion_law_constants() -> None:
    """Road-Lane composed gain is the minimum of the measured majors (the law's sign)."""
    assert COMPOSED_GAIN_MED["Road-Lane"] == min(COMPOSED_GAIN_MED.values())
    ratios = [v / COMPOSED_GAIN_MED["Road-Lane"] for k, v in COMPOSED_GAIN_MED.items()
              if k != "Road-Lane"]
    assert min(ratios) > 1.5
    assert SKIP_GAIN_RATIO_MED["Road-Lane"] == max(SKIP_GAIN_RATIO_MED.values())


def test_builds_canonical_equation() -> None:
    eq = build_lane_gain_chain_composed_v1()
    assert isinstance(eq, CanonicalEquation)
    assert eq.equation_id == EQUATION_ID
    assert len(eq.empirical_anchors) == 3
    dom = eq.domain_of_validity
    assert dom["research_only"] is True
    assert dom["score_claim"] is False


@pytest.mark.skipif(not S1.exists(), reason="s1 artifact not present")
def test_constants_match_s1_artifact() -> None:
    d = json.loads(S1.read_text())
    tbl = {r["pair"]: r for r in d["per_pair_gain_table"]}
    for pair, val in COMPOSED_GAIN_MED.items():
        assert tbl[pair]["gain_med"] == pytest.approx(val, abs=5e-4)
    for pair, val in SKIP_GAIN_RATIO_MED.items():
        assert tbl[pair]["skip_gain_ratio_med"] == pytest.approx(val, abs=5e-3)


@pytest.mark.skipif(not S3.exists(), reason="s3 artifact not present")
def test_constants_match_s3_artifact() -> None:
    d = json.loads(S3.read_text())
    lane = d["persistence_by_class"]["Lane"]
    assert lane["median"] == pytest.approx(LANE_ISLAND_PERSISTENCE_MEDIAN, abs=2e-3)
    assert lane["frac_below_0.5"] == pytest.approx(LANE_ISLAND_FRAC_BELOW_05, abs=2e-3)
