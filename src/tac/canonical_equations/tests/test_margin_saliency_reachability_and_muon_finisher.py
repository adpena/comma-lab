# SPDX-License-Identifier: MIT
"""Focused tests for the triality-sync canonical equations (FEED-03n..03q):
LEVER-4 S_R reachability replaces the inert texture proxy + Muon finishing-stage schedule.
"""
from __future__ import annotations

import math

import pytest

from tac.canonical_equations.margin_saliency_reachability_and_muon_finisher_20260703 import (
    MUON_COLD_START_DSEG_SPIKE,
    MUON_EQUATION_ID,
    SALIENCY_EQUATION_ID,
    muon_cold_start_transition_spike,
    through_r_saliency,
)
from tac.canonical_equations.margin_saliency_reachability_and_muon_finisher_20260703 import (
    build_margin_saliency_reachability_replaces_texture_proxy_v1 as build_saliency,
)
from tac.canonical_equations.margin_saliency_reachability_and_muon_finisher_20260703 import (
    build_muon_finisher_schedule_warmstart_and_lr_anneal_v1 as build_muon,
)


def test_saliency_equation_builds_and_validates() -> None:
    eq = build_saliency()
    assert eq.equation_id == SALIENCY_EQUATION_ID == eq.equation_id.lower()
    assert len(eq.empirical_anchors) == 2
    # non-orphan contract: producers AND consumers present (CanonicalEquation.__post_init__ enforces)
    assert eq.canonical_producers and eq.canonical_consumers
    assert "tools/precompute_sR_reachability.py" in eq.canonical_producers
    assert "tac.witness_dsl.gauge" in eq.canonical_consumers


def test_muon_equation_builds_and_validates() -> None:
    eq = build_muon()
    assert eq.equation_id == MUON_EQUATION_ID == eq.equation_id.lower()
    assert len(eq.empirical_anchors) == 2
    assert eq.canonical_producers and eq.canonical_consumers
    assert "experiments/train_witness_realized_through_R_mlx.py" in eq.canonical_producers
    assert "tac.witness_dsl.gauge" in eq.canonical_consumers


def test_through_r_saliency_formula() -> None:
    # sal = exp(-margin/tau) * S_R_norm ; at margin=0 -> exp(0)=1 -> sal == s_r_norm.
    assert through_r_saliency(margin=0.0, s_r_norm=0.7, tau=0.5) == pytest.approx(0.7)
    # a positive margin down-weights (exp(-m/tau) < 1); matches the fragility factor.
    assert through_r_saliency(margin=0.5, s_r_norm=1.0, tau=0.5) == pytest.approx(math.exp(-1.0))


def test_texture_proxy_inert_anchor_records_chance_alignment() -> None:
    eq = build_saliency()
    inert = eq.empirical_anchors[0]
    emp = inert.empirical_output
    # texprox-vs-S_R alignment ~= statistical chance -> texture is INERT.
    assert emp["texprox_vs_sR_top5pct_jaccard"] == pytest.approx(0.024)
    assert emp["chance_jaccard"] == pytest.approx(0.026)
    assert emp["texprox_vs_sR_pearson"] == pytest.approx(-0.033)
    # the whole S_R-alignment is the w factor, not texture (marginal gain ~ +0.009).
    assert emp["full_lever_sR_alignment"] > emp["w_only_sR_alignment"]
    assert emp["texture_marginal_alignment_gain"] == pytest.approx(0.009)
    assert inert.empirical_verification_status == "VERIFIED_VIA_EMPIRICAL_ANCHOR"


def test_saliency_net_dseg_anchor_is_205_gated() -> None:
    eq = build_saliency()
    net = eq.empirical_anchors[1]
    # honest: no borrowed number -> the net n600 d_seg is #205-gated.
    assert "205-gated" in net.empirical_output["net_n600_dseg"]
    assert net.empirical_verification_status == "ASSUMED_AWAITING_VERIFICATION"


def test_muon_cold_start_spike_constant_and_callable() -> None:
    assert pytest.approx(0.000357) == MUON_COLD_START_DSEG_SPIKE
    assert muon_cold_start_transition_spike() == pytest.approx(0.000357)
    eq = build_muon()
    spike = eq.empirical_anchors[0]
    assert spike.empirical_output["dseg_spike"] == pytest.approx(0.000357)
    assert spike.empirical_verification_status == "VERIFIED_VIA_EMPIRICAL_ANCHOR"
    # the warm-start/anneal NET d_seg is #205-gated (arm #270).
    sched = eq.empirical_anchors[1]
    assert "205-gated" in sched.empirical_output["net_n600_dseg"]
    assert sched.empirical_verification_status == "ASSUMED_AWAITING_VERIFICATION"
