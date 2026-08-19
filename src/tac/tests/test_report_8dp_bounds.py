"""rv13 F3 + F9 (and round-12 F1 / rv13 F2) — the report-8dp bound law.

Each test names the defect it prevents. The controls are executed: a test that
asserts the exact WRONG numbers the campaign published is included, so the
module is proved to distinguish them rather than merely to produce something.
"""

from __future__ import annotations

import json
import math

import pytest

from tac.report_8dp_bounds import (
    DEFAULT_COMPONENT_ROUNDING_ABS_BOUND,
    BoundContractError,
    DeltaBound,
    RowBound,
    delta_bound,
    derive_pose_score_bound,
    derive_seg_score_bound,
    extract_auth_eval_components,
    row_bound_from_result,
)

# Measured on the live to1 / ck2 T4 receipts, 2026-08-19.
LIVE_D_POSE = 7.77e-06
LIVE_PUBLISHED_POSE_BOUND = 2.836608391523776e-06
LIVE_PUBLISHED_SEG_BOUND = 5e-07
LIVE_PUBLISHED_ROW_TOTAL = 3.336608391523776e-06
LIVE_PUBLISHED_TWO_ROW_TOTAL = 6.673217e-06

# What rv13 F9 measured as the HAND-RECOMPUTED (linearized) value, and F3 as the
# hand-typed two-row total built from it. These must NOT be what we produce.
HAND_TYPED_POSE_BOUND = 2.836152e-06
HAND_TYPED_TWO_ROW_TOTAL = 6.672304e-06


def _inner(**over) -> dict:
    row = {
        "avg_posenet_dist": LIVE_D_POSE,
        "avg_segnet_dist": 0.00030309,
        "report_component_rounding_abs_bound": DEFAULT_COMPONENT_ROUNDING_ABS_BOUND,
        "report_8dp_pose_score_worst_case_abs_error_bound": LIVE_PUBLISHED_POSE_BOUND,
        "report_8dp_seg_score_worst_case_abs_error_bound": LIVE_PUBLISHED_SEG_BOUND,
        "report_8dp_score_worst_case_abs_error_bound": LIVE_PUBLISHED_ROW_TOTAL,
    }
    row.update(over)
    return row


# ---------------------------------------------------------------------------
# F9 — the exact endpoint form, not the linearization
# ---------------------------------------------------------------------------


def test_f9_derivation_reproduces_the_published_bound_exactly():
    """The whole point of F9: derived must EQUAL published, not approximate it."""
    derived = derive_pose_score_bound(LIVE_D_POSE)
    assert derived == pytest.approx(LIVE_PUBLISHED_POSE_BOUND, rel=1e-12)


def test_f9_derivation_is_NOT_the_linearized_value():
    """POSITIVE CONTROL: the module must distinguish the two forms.

    The linearization is what the to1 seal used; it disagrees in the 4th
    significant figure. A test that only checked "close to published" would
    pass on the wrong form.
    """
    derived = derive_pose_score_bound(LIVE_D_POSE)
    linearized = 5.0 / math.sqrt(10.0 * LIVE_D_POSE) * DEFAULT_COMPONENT_ROUNDING_ABS_BOUND
    assert linearized == pytest.approx(HAND_TYPED_POSE_BOUND, rel=1e-6)
    assert derived != pytest.approx(linearized, rel=1e-9), (
        "the module reproduced the linearized form rv13 F9 named as the defect"
    )
    assert derived > linearized  # sqrt is concave; the lower endpoint moves further


def test_f9_prefers_the_published_field_over_deriving():
    row = row_bound_from_result(_inner())
    assert row.source == "published"
    assert row.pose == LIVE_PUBLISHED_POSE_BOUND


def test_f9_derives_only_when_the_receipt_publishes_nothing():
    row = row_bound_from_result(
        _inner(
            report_8dp_pose_score_worst_case_abs_error_bound=None,
            report_8dp_seg_score_worst_case_abs_error_bound=None,
        )
    )
    assert row.source == "derived"
    assert row.pose == pytest.approx(LIVE_PUBLISHED_POSE_BOUND, rel=1e-12)


def test_pose_bound_GROWS_as_d_pose_falls():
    """The counter-intuitive direction, asserted so nobody 'fixes' it."""
    assert derive_pose_score_bound(1e-6) > derive_pose_score_bound(1e-4)


def test_seg_bound_is_linear_and_d_pose_independent():
    assert derive_seg_score_bound() == pytest.approx(100.0 * DEFAULT_COMPONENT_ROUNDING_ABS_BOUND)


def test_derivation_refuses_nonsense_d_pose():
    with pytest.raises(BoundContractError):
        derive_pose_score_bound(-1.0)
    with pytest.raises(BoundContractError):
        derive_pose_score_bound("7.77e-6")


# ---------------------------------------------------------------------------
# F3 / F2 — bounds ADD for a delta, and the addends must sum to the total
# ---------------------------------------------------------------------------


def test_f2_delta_bound_is_the_SUM_of_two_rows():
    bound = delta_bound(_inner(), _inner())
    assert bound.total == pytest.approx(LIVE_PUBLISHED_TWO_ROW_TOTAL, rel=1e-6)
    assert bound.total == pytest.approx(2 * LIVE_PUBLISHED_ROW_TOTAL, rel=1e-12)


def test_f2_the_one_row_division_is_exactly_2x_the_correct_multiple():
    """POSITIVE CONTROL reproducing the measured ck2 error.

    rv13 F2: the memo said 131.1x, the seal said 65.6x, and the overstatement
    was exactly 2.00x. Both numbers are reproduced here so the module is shown
    to compute the right one rather than merely a plausible one.
    """
    net_ds = -4.374693322012624e-04
    bound = delta_bound(_inner(), _inner())
    correct = bound.multiple_of(net_ds)
    wrong = abs(net_ds) / bound.base.total
    assert correct == pytest.approx(65.6, abs=0.05)
    assert wrong == pytest.approx(131.1, abs=0.05)
    assert wrong / correct == pytest.approx(2.00, rel=1e-9)


def test_f3_rendered_addends_sum_to_the_rendered_total():
    """The exact requirement commit 6e976eeafd asked for and to1's seal missed."""
    bound = delta_bound(_inner(), _inner())
    text = bound.describe(net_ds=-6.991519007781832e-05)
    assert f"{bound.total:.6e}" in text
    assert f"{bound.base.total:.6e}" in text
    assert bound.base.total + bound.candidate.total == pytest.approx(bound.total, rel=1e-12)


def test_f3_does_not_reproduce_the_hand_typed_total():
    """POSITIVE CONTROL: the wrong two-row total must not be producible."""
    bound = delta_bound(_inner(), _inner())
    assert bound.total != pytest.approx(HAND_TYPED_TWO_ROW_TOTAL, rel=1e-9)


def test_f3_says_EQUAL_when_d_pose_matches():
    """to1's seal called two identical bounds 'unequal per row'. They were equal."""
    bound = delta_bound(_inner(), _inner())
    assert bound.rows_are_equal
    assert "are equal here" in bound.describe()


def test_f3_says_UNEQUAL_when_d_pose_differs():
    """The other half of the control — the phrase must track the arithmetic."""
    bound = delta_bound(_inner(), _inner(
        avg_posenet_dist=7.65e-06,
        report_8dp_pose_score_worst_case_abs_error_bound=2.858e-06,
    ))
    assert not bound.rows_are_equal
    assert "are unequal here" in bound.describe()


def test_f3_self_check_FIRES_on_an_inconsistent_total():
    """POSITIVE CONTROL for the self-check itself.

    Built by hand-forcing a mismatch, because the public constructor cannot
    produce one — which is the design goal.
    """
    class _Lying(DeltaBound):
        @property
        def total(self) -> float:  # type: ignore[override]
            return 1.0

    bad = _Lying(base=RowBound(seg=5e-7, pose=2.8e-6, d_pose=7.77e-6, source="published"),
                 candidate=RowBound(seg=5e-7, pose=2.8e-6, d_pose=7.77e-6, source="published"))
    with pytest.raises(BoundContractError):
        bad.self_check()
    with pytest.raises(BoundContractError):
        bad.describe()


def test_delta_bound_cannot_be_built_from_one_row():
    """Structural: there is no single-row constructor. F2 is unrepresentable."""
    with pytest.raises(TypeError):
        DeltaBound(base=RowBound(seg=5e-7, pose=2.8e-6, d_pose=1e-5, source="derived"))


# ---------------------------------------------------------------------------
# Extraction — the nested, bytes-repr'd artifact
# ---------------------------------------------------------------------------


def test_extracts_from_the_modal_wrapper_with_bytes_repr_artifact():
    """The real shape: the bounds are NOT top-level on MODAL_REMOTE_RESULT.json.

    They sit in artifacts["contest_auth_eval.json"], stored as the repr of a
    bytes object. Encapsulated so the next seal writer does not retype the
    number rather than hand-decode this.
    """
    inner_bytes = json.dumps(_inner()).encode()
    wrapper = {
        "score_recomputed_from_components": 0.1565,
        "artifacts": {"contest_auth_eval.json": repr(inner_bytes)},
    }
    block = extract_auth_eval_components(wrapper)
    assert block["report_8dp_pose_score_worst_case_abs_error_bound"] == LIVE_PUBLISHED_POSE_BOUND
    assert row_bound_from_result(wrapper).source == "published"


def test_extraction_returns_empty_on_junk():
    assert extract_auth_eval_components(None) == {}
    assert extract_auth_eval_components("not json") == {}
    assert extract_auth_eval_components({"unrelated": 1}) == {}


def test_row_bound_REFUSES_when_nothing_is_computable():
    """POSITIVE CONTROL: refuse rather than invent."""
    with pytest.raises(BoundContractError):
        row_bound_from_result({"unrelated": 1})
    with pytest.raises(BoundContractError):
        row_bound_from_result({"avg_posenet_dist": "not-a-number"})


def test_no_api_accepts_a_hand_typed_bound():
    """The module's central refusal, asserted as a contract."""
    import inspect

    from tac import report_8dp_bounds

    for name, obj in vars(report_8dp_bounds).items():
        if name.startswith("_") or not callable(obj):
            continue
        try:
            params = inspect.signature(obj).parameters
        except (TypeError, ValueError):
            continue
        assert "bound" not in params, f"{name} accepts a caller-supplied bound"


# ---------------------------------------------------------------------------
# The live receipts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "base,cand",
    [(
        "/Volumes/APDataStore/pact/ddm_ck2/t4_row_r2/MODAL_REMOTE_RESULT.json",
        "/Volumes/APDataStore/pact/ddm_to1/t4_row_r1/MODAL_REMOTE_RESULT.json",
    )],
)
def test_live_receipts_reproduce_the_published_two_row_total(base, cand):
    from pathlib import Path

    if not (Path(base).is_file() and Path(cand).is_file()):
        pytest.skip("SSD custody tier not mounted")
    bound = delta_bound(json.loads(Path(base).read_text()), json.loads(Path(cand).read_text()))
    assert bound.base.source == "published"
    assert bound.total == pytest.approx(LIVE_PUBLISHED_TWO_ROW_TOTAL, rel=1e-6)
    assert bound.multiple_of(-6.991519007781832e-05) == pytest.approx(10.48, abs=0.005)


def test_a_stray_bool_is_not_a_published_bound():
    """POSITIVE CONTROL for the bool guard (bool subclasses int).

    ``True`` would otherwise be read as a published bound of 1.0 and dominate
    every margin it touched, silently.
    """
    row = row_bound_from_result(
        _inner(
            report_8dp_pose_score_worst_case_abs_error_bound=True,
            report_8dp_seg_score_worst_case_abs_error_bound=True,
        )
    )
    assert row.source == "derived"
    assert row.pose == pytest.approx(LIVE_PUBLISHED_POSE_BOUND, rel=1e-12)


def test_a_bool_d_pose_is_refused():
    with pytest.raises(BoundContractError):
        row_bound_from_result({"avg_posenet_dist": True})
