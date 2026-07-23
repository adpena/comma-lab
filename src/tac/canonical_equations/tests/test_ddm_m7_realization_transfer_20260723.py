from __future__ import annotations

import math

import pytest

from tac.canonical_equations.ddm_m7_realization_transfer_20260723 import (
    DDM_M7_REALIZATION_TRANSFER_EQUATION_ID,
    DOMAIN_DECLARATION,
    ScoreGapDecomposition,
    contest_score_terms,
    realization_transfer_ratios,
    require_score_gap_closure,
    score_gap_closes,
    score_gap_decomposition,
)


def test_equation_identity_and_instance_level_domain() -> None:
    assert (
        DDM_M7_REALIZATION_TRANSFER_EQUATION_ID
        == "ddm_m7_solve_to_realized_transfer_receiver_closed_v1"
    )
    assert DOMAIN_DECLARATION["scope"] == "instance_level_diagnostic"
    assert DOMAIN_DECLARATION["not_a_universal_transfer_coefficient"] is True


def test_contest_score_arithmetic() -> None:
    terms = contest_score_terms(
        d_seg=0.00015196,
        d_pose=0.00010184,
        archive_bytes=177169,
        reference_bytes=37545489,
    )
    assert terms.seg == pytest.approx(0.015196)
    assert terms.pose == pytest.approx(math.sqrt(0.0010184))
    assert terms.rate == pytest.approx(25 * 177169 / 37545489)
    assert terms.total == pytest.approx(terms.seg + terms.pose + terms.rate)


def test_transfer_ratios_are_realized_over_counterfactual() -> None:
    ratios = realization_transfer_ratios(
        counterfactual_d_seg=0.25,
        counterfactual_d_pose=0.5,
        realized_d_seg=0.5,
        realized_d_pose=0.125,
    )
    assert ratios.d_seg == 2.0
    assert ratios.d_pose == 0.25


def test_gap_decomposition_closes_with_identical_rate() -> None:
    counterfactual = contest_score_terms(
        d_seg=0.00015196,
        d_pose=0.00010184,
        archive_bytes=177169,
        reference_bytes=37545489,
    )
    realized = contest_score_terms(
        d_seg=0.00054530,
        d_pose=0.00002931,
        archive_bytes=177169,
        reference_bytes=37545489,
    )
    gap = score_gap_decomposition(
        counterfactual=counterfactual,
        realized=realized,
    )
    assert gap.rate == 0.0
    assert score_gap_closes(gap)
    require_score_gap_closure(gap)


def test_gap_closure_refuses_nonadditive_total() -> None:
    gap = ScoreGapDecomposition(seg=1.0, pose=2.0, rate=3.0, total=7.0)
    assert not score_gap_closes(gap)
    with pytest.raises(ValueError, match="do not sum"):
        require_score_gap_closure(gap)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (
            {
                "d_seg": -1.0,
                "d_pose": 0.0,
                "archive_bytes": 1,
                "reference_bytes": 2,
            },
            "d_seg",
        ),
        (
            {
                "d_seg": 0.0,
                "d_pose": 0.0,
                "archive_bytes": 1,
                "reference_bytes": 0,
            },
            "reference_bytes",
        ),
    ],
)
def test_score_terms_refuse_invalid_domain(
    kwargs: dict[str, float | int],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        contest_score_terms(**kwargs)
