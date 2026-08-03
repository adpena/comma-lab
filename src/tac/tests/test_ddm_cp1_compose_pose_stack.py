"""Tests for ``tools/cp1_compose_pose_stack.py`` (ddm_cp1 composition arm).

The tool answers whether the measured pose-line wins COMPOSE.  These tests
cover the score arithmetic (which must be recomputed from components, never
read off a rounded headline), the fail-closed guards that stop a wrong-scale
fold or a subset verdict, and the structural section-disjointness fact the
whole matrix rests on.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[3]
_TOOL = REPO / "tools" / "cp1_compose_pose_stack.py"


def _load():
    spec = importlib.util.spec_from_file_location("cp1_compose_pose_stack", _TOOL)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["cp1_compose_pose_stack"] = mod
    spec.loader.exec_module(mod)
    return mod


cp1 = _load()


# --------------------------------------------------------------------------- #
# score arithmetic -- recomputed from components
# --------------------------------------------------------------------------- #
def test_contribution_is_the_scores_own_sqrt_form():
    assert cp1.contribution(0.1) == pytest.approx(1.0)
    assert cp1.contribution(0.0025512504590816916) == pytest.approx(
        0.15972634282051573, abs=1e-12)


def test_contribution_is_concave_so_the_tail_carries_the_term():
    """Halving a small mean buys more than halving a large one, in S."""
    big = cp1.contribution(0.02) - cp1.contribution(0.01)
    small = cp1.contribution(0.002) - cp1.contribution(0.001)
    assert big > small > 0.0


def test_contribution_of_zero_is_zero():
    assert cp1.contribution(0.0) == 0.0


def test_rate_term_uses_the_contest_denominator():
    assert cp1.rate_term(37_545_489) == pytest.approx(25.0)
    assert cp1.rate_term(360_339) == pytest.approx(0.2399349493090901, abs=1e-15)


def test_rate_term_is_linear_in_bytes():
    assert cp1.rate_term(2000) - cp1.rate_term(1000) == pytest.approx(
        cp1.rate_term(1000), abs=1e-15)


def test_composed_S_reproduces_the_live_base_from_components():
    """dc1_fold: the live own-vehicle base, recomputed rather than quoted."""
    out = cp1.composed_S(cp1.DC1_FOLD_SEG_TERM, cp1.DC1_FOLD_DPOSE,
                         cp1.DC1_FOLD_BYTES)
    assert out["S"] == pytest.approx(0.8983766169605341, abs=1e-9)
    assert out["seg"] + out["pose"] + out["rate"] == pytest.approx(out["S"])


def test_composed_S_reproduces_the_measured_cp1_fold_row():
    out = cp1.composed_S(0.431179, 0.002553784570419757, 360_339)
    assert out["S"] == pytest.approx(0.8309195990971923, abs=1e-9)


def test_composed_S_gap_is_measured_against_the_pr130_floor():
    out = cp1.composed_S(0.431179, 0.00516574, 360_309)
    assert out["gap_to_bar"] == pytest.approx(out["S"] - cp1.PR130_BAR_S)
    assert cp1.PR130_BAR_S == 0.172141


def test_the_measured_fold_is_a_net_loss_against_pj2_alone():
    """The unit's headline, as arithmetic: the rate refund does NOT cover the
    pose cost of re-rounding a tighter solution onto a different lattice."""
    pj2 = cp1.composed_S(0.431179, 0.0025512504590816916, 360_406)
    fold = cp1.composed_S(0.431179, 0.002553784570419757, 360_339)
    assert fold["rate"] < pj2["rate"]          # the refund is real
    assert fold["pose"] > pj2["pose"]          # and it is over-spent
    assert fold["S"] > pj2["S"]                # net LOSS
    assert fold["S"] - pj2["S"] == pytest.approx(3.469e-05, rel=0.02)


# --------------------------------------------------------------------------- #
# fail-closed guards
# --------------------------------------------------------------------------- #
def test_report_refuses_a_subset_verdict(tmp_path):
    """n600 IS the evidence bar; a partial run may not carry a verdict."""
    (tmp_path / "cp1_score_shard0.jsonl").write_text(
        json.dumps({"pair": 0, "d_source_reported": 1.0,
                    "d_folded_quantized_at_incumbent_st": 1.0,
                    "canary_abs_err": 0.0}) + "\n")
    ns = _ns(out_dir=tmp_path, out=tmp_path / "r.json", allow_partial=False,
             archive_bytes=0, source_bytes=360_406)
    with pytest.raises(SystemExit, match="n600 IS the evidence bar"):
        cp1.run_report(ns)


def test_report_refuses_a_canary_above_the_measured_instrument_floor(tmp_path):
    rows = [{"pair": i, "d_source_reported": 1.0,
             "d_folded_quantized_at_incumbent_st": 1.0,
             "canary_abs_err": 1.0} for i in range(cp1.N_PAIRS)]
    (tmp_path / "cp1_score_shard0.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    ns = _ns(out_dir=tmp_path, out=tmp_path / "r.json", allow_partial=False,
             archive_bytes=0, source_bytes=360_406)
    with pytest.raises(SystemExit, match="CANARY"):
        cp1.run_report(ns)


def test_canary_floor_matches_the_ms8_measured_value():
    """A drifted floor would silently admit a broken harness."""
    assert cp1.CANARY_MAX_ABS_ERR == 1.2e-05


def test_report_accepts_a_complete_run_and_recomputes_S(tmp_path):
    rows = [{"pair": i, "d_source_reported": 0.001,
             "d_folded_quantized_at_incumbent_st": 0.002,
             "canary_abs_err": 1e-9} for i in range(cp1.N_PAIRS)]
    (tmp_path / "cp1_score_shard0.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    ns = _ns(out_dir=tmp_path, out=tmp_path / "r.json", allow_partial=False,
             archive_bytes=360_339, source_bytes=360_406)
    cp1.run_report(ns)
    doc = json.loads((tmp_path / "r.json").read_text())
    assert doc["n_scored"] == cp1.N_PAIRS
    assert doc["pairs_worse_after_fold"] == cp1.N_PAIRS
    assert doc["composed_folded"]["pose"] == pytest.approx(
        math.sqrt(10 * 0.002))


def test_report_marks_itself_non_promotable(tmp_path):
    rows = [{"pair": i, "d_source_reported": 0.001,
             "d_folded_quantized_at_incumbent_st": 0.001,
             "canary_abs_err": 0.0} for i in range(cp1.N_PAIRS)]
    (tmp_path / "cp1_score_shard0.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows))
    ns = _ns(out_dir=tmp_path, out=tmp_path / "r.json", allow_partial=False,
             archive_bytes=360_339, source_bytes=360_406)
    cp1.run_report(ns)
    doc = json.loads((tmp_path / "r.json").read_text())
    assert doc["score_claim"] is False
    assert doc["promotion_eligible"] is False
    assert doc["pointer_moved"] is False
    assert "advisory" in doc["axis"]


# --------------------------------------------------------------------------- #
# the structural fact the matrix rests on
# --------------------------------------------------------------------------- #
def test_the_fold_is_a_pure_scale_on_the_translation_triple():
    """``t = s_t * [p2,p1,p0]`` -- only p[0:3] may move, and by exactly k."""
    poses = np.arange(12, dtype=np.float64).reshape(2, 6) + 1.0
    k = np.array([2.0, 0.5])
    resc = poses.copy()
    resc[:, 0:3] = poses[:, 0:3] * k[:, None]
    assert np.array_equal(resc[:, 3:], poses[:, 3:])
    assert np.allclose(resc[:, 0:3] * (1.0 / k)[:, None], poses[:, 0:3])


def test_scale_degeneracy_leaves_the_effective_translation_invariant():
    p = np.array([33.5, 0.138, -0.189])
    s = 0.08
    for k in (0.5, 1.0, 1.75, 2.0, 4.0):
        assert np.allclose(s * p, (s / k) * (p * k), rtol=0, atol=1e-13)


def test_module_constants_match_the_live_base_archive():
    assert cp1.DC1_FOLD_BYTES == 360_309
    assert cp1.DC1_FOLD_SEG_TERM == 0.431179
    assert cp1.N_PAIRS == 600
    assert cp1.ARCHIVE_DENOM == 37_545_489.0


def _ns(**kw):
    import argparse

    ns = argparse.Namespace()
    for k, v in kw.items():
        setattr(ns, k, v)
    return ns
