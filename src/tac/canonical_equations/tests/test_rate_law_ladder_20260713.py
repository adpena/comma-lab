"""The rate-law ladder registration keeps its anchors + constants + owed measurables intact."""
from pathlib import Path

from tac.canonical_equations import rate_law_ladder_20260713 as L


def test_equation_id_and_axis():
    assert L.EQUATION_ID == "rate_law_ladder_v1"
    assert "score_claim=false" in L._AXIS


def test_rung_memo_anchors_exist_on_disk():
    for memo in (L.RUNG1_MEMO, L.RUNG2_MEMO, L.RUNG3_MEMO, L.RUNG4_MEMO):
        assert Path(memo).is_file(), f"missing rung anchor: {memo}"


def test_burnside_constants():
    assert L.SCORE_IMAGE_LOG2_BOUND_BITS == 64
    assert L.QUOTIENT_LABEL_IDEAL_BITS_MAX == 64


def test_named_terms_and_owed_measurables():
    assert "q_G" in L.GAP_TERM and "U(W)" in L.GAP_TERM
    assert "Theta" in L.TWIST_TERM
    assert "H(E|X,C)" in L.TEMPORAL_CHAIN  # the marked-event term is present
    assert len(L.OWED_MEASURABLES) == 4
    assert "fiber_completeness_gap_n600" in L.OWED_MEASURABLES


def test_composed_statement_mentions_section_engineering():
    s = L.composed_statement()
    assert "section engineering" in s and "64" in s
