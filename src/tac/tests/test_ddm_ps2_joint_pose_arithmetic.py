"""Targeted tests for the ddm_ps2 joint pose/seg/rate arithmetic.

These functions decide whether a post-hoc channel supplies, so the properties that matter are
(a) the sqrt pose law is inverted correctly by the break-even solver, (b) the aggregation
conventions match `upstream/evaluate.py` (ratio of means, not mean of ratios), and (c) the seg
term is driven by the channel's CLIP-WIDE described flips rather than whatever sample eta was
estimated on -- the conflation that inflated this arm's own first draft 4.5x.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


def _load(stem: str):
    path = REPO / "experiments" / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[stem] = mod
    spec.loader.exec_module(mod)
    return mod


ps2 = _load("ddm_ps2_f2_joint_adjudicate")
gate = _load("ddm_ps2_joint_gate_survivor")


def test_pose_break_even_inverts_the_sqrt_law() -> None:
    """dS_pose(break_even(x)) + x == 0 for any seg+rate term that supplies."""
    for seg_rate in (-1e-3, -3.3568e-4, -1e-5):
        r = ps2.pose_ratio_break_even(seg_rate)
        assert ps2.dS_pose(r) + seg_rate == pytest.approx(0.0, abs=1e-15)


def test_break_even_ratio_exceeds_one_only_when_seg_rate_supplies() -> None:
    """A channel that already loses on seg+rate cannot tolerate ANY pose cost."""
    assert ps2.pose_ratio_break_even(-1e-4) > 1.0
    assert ps2.pose_ratio_break_even(+1e-4) < 1.0
    assert ps2.pose_ratio_break_even(0.0) == pytest.approx(1.0)


def test_dS_pose_is_zero_at_unit_ratio_and_signed_correctly() -> None:
    assert ps2.dS_pose(1.0) == pytest.approx(0.0, abs=1e-18)
    assert ps2.dS_pose(1.5) > 0.0      # pose got worse -> a cost
    assert ps2.dS_pose(0.5) < 0.0      # pose got better -> a credit


def test_pose_agg_ratio_is_ratio_of_means_not_mean_of_ratios() -> None:
    """The two disagree whenever pairs carry unequal pose mass; evaluate.py uses ratio-of-means.

    Constructed so a mean-of-ratios reading would give 1.05 while the correct answer is 1.0099.
    """
    rows = [{"d_pose_before": 100.0, "d_pose_after": 100.0},
            {"d_pose_before": 1.0, "d_pose_after": 2.0}]
    assert ps2.pose_agg_ratio(rows) == pytest.approx(102.0 / 101.0)
    assert ps2.pose_agg_ratio(rows) != pytest.approx(1.5)


def test_pooled_eta_is_ratio_of_sums_not_mean_of_per_pair_eta() -> None:
    """A pair describing 200 flips must not weigh the same as one describing 5."""
    rows = [{"flips_before": 200, "flips_after": 200, "n_described_ring0": 200},
            {"flips_before": 5, "flips_after": 0, "n_described_ring0": 5}]
    assert ps2.pooled_eta(rows) == pytest.approx(5.0 / 205.0)


def test_seg_term_uses_clipwide_described_flips_not_sample_flips() -> None:
    """The regression guard for this arm's own first-draft bug.

    fo2h's published seg+rate figure is reproduced only with the channel's clip-wide 6,512
    described flips. Passing a 48-pair sample's flip count instead inflates the seg gain and
    flips the sign of the verdict.
    """
    got = ps2.net_dS_seg_rate(ps2.FO2H_PUBLISHED["pooled_eta_n48"],
                              ps2.FO1_DESCRIBED_FLIPS, ps2.FO1_TOTAL_B)
    assert got == pytest.approx(ps2.FO2H_PUBLISHED["net_dS_n48"], abs=1e-15)
    wrong = ps2.net_dS_seg_rate(ps2.FO2H_PUBLISHED["pooled_eta_n48"], 1795.0, ps2.FO1_TOTAL_B)
    assert wrong > 0.0 > got, "sample-flip conflation must be caught, not silently rescored"


def test_subset_index_price_is_combinatorial_and_degenerate_at_the_ends() -> None:
    """Telling the decoder WHICH pairs were edited is priced as an exact rank, not a bitmap."""
    assert gate.subset_index_bytes(600, 0) == 0.0
    assert gate.subset_index_bytes(600, 600) == 0.0
    mid = gate.subset_index_bytes(600, 300)
    assert mid == pytest.approx(74.4, abs=0.5)          # ~C(600,300) -> 595 bits
    assert gate.subset_index_bytes(600, 30) < mid       # sparse subsets are cheaper than a bitmap
    assert gate.subset_index_bytes(600, 30) < 600 / 8.0


def test_joint_dS_of_an_empty_gate_is_pure_rate_cost() -> None:
    """Editing nothing must cost exactly the payload and change neither seg nor pose."""
    rows = [{"flips_before": 10, "flips_after": 3, "n_described_ring0": 12,
             "d_pose_before": 1e-5, "d_pose_after": 2e-5}]
    import numpy as np
    out = gate.joint_dS(rows, np.zeros(1), 0.0)
    assert out["dS_seg"] == pytest.approx(0.0)
    assert out["pose_ratio"] == pytest.approx(1.0)
    assert out["dS_pose"] == pytest.approx(0.0, abs=1e-18)
    assert out["dS_joint"] == pytest.approx(gate.FO1_TOTAL_B * gate.RATE_DS_PER_BYTE)


def test_ungated_joint_reproduces_the_published_three_term_composition() -> None:
    """End-to-end: seg+rate -0.000336 and pose +0.001424 compose to a NET LOSS."""
    seg_rate = ps2.FO2H_PUBLISHED["net_dS_n48"]
    pose = ps2.FO2H_PUBLISHED["delta_S_pose_n48"]
    assert seg_rate < 0.0 < pose
    assert seg_rate + pose == pytest.approx(0.001087878201153595, abs=1e-15)
    assert abs(pose / seg_rate) == pytest.approx(4.2408, abs=1e-3)
