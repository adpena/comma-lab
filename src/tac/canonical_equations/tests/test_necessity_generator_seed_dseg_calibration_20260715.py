# SPDX-License-Identifier: MIT
"""Isolated tests for necessity_generator_seed_dseg_calibration_20260715."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.canonical_equations.necessity_generator_seed_dseg_calibration_20260715 import (
    DSEG_PALETTE_EPS0,
    DSEG_PALETTE_HOOD_EPS0,
    EQUATION_ID,
    HOOD_SEED_BYTES,
    N_RATE,
    POINTER,
    build_necessity_generator_seed_dseg_calibration_v1,
    hood_tex_delta_s_per_byte,
    min_s_operating_point,
    predicted_S,
)

REPO_ROOT = Path(__file__).resolve().parents[3].parent
ART_DIR = REPO_ROOT / "experiments" / "results" / "necessity_dseg_calibration_20260715"


def test_predicted_S_matches_manual() -> None:
    got = predicted_S(0.01, 1000, pose_contribution=0.1)
    assert got == pytest.approx(100 * 0.01 + 0.1 + 25 * 1000 / N_RATE)


def test_predicted_S_rejects_negative() -> None:
    with pytest.raises(ValueError):
        predicted_S(-0.1, 100)
    with pytest.raises(ValueError):
        predicted_S(0.1, -100)


def test_hood_tex_is_a_buy() -> None:
    # negative ΔS/byte == the buy lowers S
    assert hood_tex_delta_s_per_byte() < 0.0


def test_min_s_operating_point_not_sub_pointer() -> None:
    mp = min_s_operating_point()
    assert mp.dseg_real == pytest.approx(DSEG_PALETTE_HOOD_EPS0)
    assert mp.predicted_S > POINTER  # the honest verdict: 8.4x ABOVE
    assert mp.sub_pointer is False
    assert mp.ratio_over_pointer == pytest.approx(mp.predicted_S / POINTER)


def test_hood_reduces_dseg_71pct() -> None:
    assert DSEG_PALETTE_HOOD_EPS0 < DSEG_PALETTE_EPS0
    frac = (DSEG_PALETTE_EPS0 - DSEG_PALETTE_HOOD_EPS0) / DSEG_PALETTE_EPS0
    assert frac == pytest.approx(0.7075, abs=1e-3)


def test_equation_builds_and_validates() -> None:
    eq = build_necessity_generator_seed_dseg_calibration_v1()
    assert eq.equation_id == EQUATION_ID
    assert eq.domain_of_validity["research_only"] is True
    assert eq.domain_of_validity["score_claim"] is False
    assert eq.domain_of_validity["verdict_scope"].startswith("FORMULATION")
    assert len(eq.empirical_anchors) == 3


def test_constants_match_measured_summary() -> None:
    """Module constants must equal the on-disk measured summary (no drift)."""
    summary_path = ART_DIR / "summary.json"
    if not summary_path.exists():
        pytest.skip("calibration artifacts not present on this checkout")
    s = json.loads(summary_path.read_text())
    assert s["eps"]["0.0"]["dseg_real"] == pytest.approx(DSEG_PALETTE_EPS0, rel=1e-5)
    assert s["eps"]["0.0_hoodtex"]["dseg_real"] == pytest.approx(DSEG_PALETTE_HOOD_EPS0, rel=1e-5)
    assert s["hood_seed_bytes_ds16"] == HOOD_SEED_BYTES
    # min-S is the hoodtex arm
    arms = {k: v["seed"]["predicted_S_adjusted"] for k, v in s["eps"].items()}
    assert min(arms, key=lambda k: arms[k]) == "0.0_hoodtex"
