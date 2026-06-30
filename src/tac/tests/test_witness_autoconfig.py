"""Tests for tac.witness_autoconfig — the clip -> witness_config actuator.

These tests verify BEHAVIOR (the generators actually compute / route), not just
constants: the flag-validation test parses the real trainer argparse so a config
that emitted an invented flag would FAIL; the intrinsic-dim test recovers a known
manifold dimension from real data. Pure CPU / numpy; no GPU, no heavy I/O.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from tac import witness_autoconfig as wac

_REPO = Path(__file__).resolve().parents[3]
_TRAINER = _REPO / "experiments/train_levelset_witness_realized_through_R_mlx.py"
_GT_N600 = "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"


def _real_trainer_flags() -> frozenset[str]:
    text = _TRAINER.read_text()
    return frozenset(re.findall(r'add_argument\(\s*"(--[a-z0-9-]+)"', text))


# --------------------------------------------------------------------------
# Whitney embedding generator
# --------------------------------------------------------------------------
def test_whitney_clamp_m9_to_19():
    assert wac.whitney_mod_dim(9) == 19  # 2*9+1 = 19, in band


def test_whitney_clamp_m12_to_25():
    assert wac.whitney_mod_dim(12) == 25  # 2*12+1 = 25


def test_whitney_clamp_m20_ceiling_26():
    assert wac.whitney_mod_dim(20) == 26  # 2*20+1 = 41 -> clamp to 26


def test_whitney_clamp_low_floor_19():
    assert wac.whitney_mod_dim(5) == 19  # 2*5+1 = 11 -> clamp up to 19


# --------------------------------------------------------------------------
# intrinsic-dim generator: MEASURES from data, FALLS BACK when absent (NO-FAKE)
# --------------------------------------------------------------------------
def test_intrinsic_dim_fallback_flagged_when_absent():
    pv = wac.intrinsic_dim(None)
    assert pv.is_fallback
    assert pv.source == wac.SRC_FALLBACK
    assert pv.value == pytest.approx(9.0)


def test_intrinsic_dim_measures_known_low_dim_manifold():
    # a 2-D plane linearly embedded in 12-D -> intrinsic dim ~2, NOT fallback.
    rng = np.random.default_rng(0)
    latent = rng.standard_normal((300, 2))
    basis = rng.standard_normal((2, 12))
    with np.errstate(all="ignore"):  # spurious Accelerate matmul FP-flag on macOS
        X = latent @ basis + 1e-3 * rng.standard_normal((300, 12))
    pv = wac.intrinsic_dim(X)
    assert pv.source == wac.SRC_MEASURED
    assert not pv.is_fallback
    assert 1.3 < float(pv.value) < 3.5  # ~2


# --------------------------------------------------------------------------
# mod / hidden / muon-lr / verdict-pairs exact values (the dogfood revisions)
# --------------------------------------------------------------------------
def test_mod_dim_overfit_ships_26():
    assert wac.mod_dim_generator(None, overfit=True).value == 26


def test_mod_dim_aggressive_uses_whitney_floor():
    # overfit=False with fallback m=9 -> Whitney floor 19 (aggressive theta*).
    assert wac.mod_dim_generator(None, overfit=False).value == 19


def test_hidden_dim_is_96_not_120():
    assert wac.hidden_dim_generator(None).value == 96


def test_hidden_dim_picks_rd_min_when_sweep_supplied():
    pv = wac.hidden_dim_generator({96: 90621, 120: 111902, 128: 161000})
    assert pv.value == 96
    assert pv.source == wac.SRC_MEASURED


def test_muon_lr_is_proven_0p002():
    assert wac.muon_lr_generator().value == pytest.approx(0.002)


def test_verdict_pairs_is_96():
    assert wac.verdict_pairs_generator(600).value == 96


# --------------------------------------------------------------------------
# curriculum schedule generator
# --------------------------------------------------------------------------
def test_curriculum_schedule_proven_1000():
    s = wac.curriculum_schedule(1000)
    assert s["tau_softplus_start_epoch"].value == 300
    assert s["l7_start_epoch"].value == 600
    assert s["muon_start_epoch"].value == 726


def test_curriculum_schedule_scales_proportionally():
    s = wac.curriculum_schedule(2000)
    assert s["tau_softplus_start_epoch"].value == 600
    assert s["l7_start_epoch"].value == 1200
    assert s["muon_start_epoch"].value == 1452


# --------------------------------------------------------------------------
# lever priors (attribution-clean) + portability split
# --------------------------------------------------------------------------
def test_lever_priors_attribution_clean_first():
    lp = wac.lever_priors()
    assert lp["surgical_levers_enabled"] is False
    assert lp["dm1_enabled"] is False
    assert "margin_saliency" in lp["deferred_levers"]
    assert "lane_prior_phi1" in lp["active_geometric_priors"]


def test_portability_split_has_all_three_classes():
    p = wac.portability_split()
    vals = set(p.values())
    assert wac.Portability.SCORER_FIXED in vals
    assert wac.Portability.DOMAIN in vals
    assert wac.Portability.INSTANCE in vals
    # mod/hidden are instance-conditioned; muon-lr is scorer-fixed.
    assert p["mod_dim"] == wac.Portability.INSTANCE
    assert p["muon_lr"] == wac.Portability.SCORER_FIXED


def test_warp_priors_per_class_design_level():
    w = wac.warp_priors()
    assert w["per_class"]["Road"]["warp"] == "ground_homography"
    assert w["per_class"]["hood"]["warp"] == "identity"
    assert w["per_class"]["sky"]["warp"] == "rotation_only"
    assert "design" in w["status"]


# --------------------------------------------------------------------------
# derive_config: valid ranges, exact values, determinism
# --------------------------------------------------------------------------
def test_derive_config_valid_ranges_and_stage_order():
    cfg = wac.derive_config(_GT_N600, num_pairs=600)
    assert 19 <= cfg.mod_dim <= 26
    assert cfg.hidden_dim > 0
    assert cfg.muon_lr > 0
    assert cfg.epochs > 0
    # curriculum monotone: 0 < tau < l7 <= epochs and muon >= l7.
    assert 0 < cfg.tau_softplus_start_epoch < cfg.l7_start_epoch <= cfg.epochs
    assert cfg.muon_start_epoch >= cfg.l7_start_epoch


def test_derive_config_exact_dogfood_values():
    cfg = wac.derive_config(_GT_N600, num_pairs=600)
    assert cfg.mod_dim == 26
    assert cfg.hidden_dim == 96
    assert cfg.muon_lr == pytest.approx(0.002)
    assert cfg.verdict_pairs == 96
    assert cfg.surgical_levers_enabled is False
    assert cfg.dm1_enabled is False


def test_derive_config_deterministic():
    a = wac.derive_config(_GT_N600, num_pairs=600)
    b = wac.derive_config(_GT_N600, num_pairs=600)
    assert a == b


def test_derive_config_provenance_present_for_derived_fields():
    cfg = wac.derive_config(_GT_N600, num_pairs=600)
    for k in ("mod_dim", "hidden_dim", "muon_lr", "verdict_pairs",
              "tau_softplus_start_epoch", "l7_start_epoch", "muon_start_epoch"):
        assert k in cfg.provenance
        assert isinstance(cfg.provenance[k], wac.ProvenancedValue)
        assert cfg.provenance[k].provenance  # non-empty rationale string


# --------------------------------------------------------------------------
# the flag-validation contract: EVERY emitted flag is a REAL trainer flag
# (behavior test — would fail if any flag were invented)
# --------------------------------------------------------------------------
def test_emitted_flags_all_exist_in_trainer_argparse():
    cfg = wac.derive_config(_GT_N600, num_pairs=600)
    real = _real_trainer_flags()
    emitted = [flag for flag, _ in cfg.to_trainer_flags("out/dir")]
    missing = [f for f in emitted if f not in real]
    assert missing == [], f"invented flags not in trainer argparse: {missing}"


def test_command_has_critical_revisions_and_is_attribution_clean():
    cfg = wac.derive_config(_GT_N600, num_pairs=600)
    cmd = cfg.to_command("experiments/results/levelset_n600_v2_x")
    # the 4 binding revisions
    assert "--muon-lr 0.002" in cmd
    assert "--mod-dim 26" in cmd
    assert "--hidden-dim 96" in cmd
    assert "--verdict-pairs 96" in cmd
    # attribution-clean: NO surgical levers / DM1
    for off in ("--margin-saliency", "--lane-thin", "--hardness",
                "--film-stiefel", "--code-spectral-entropy", "--dm1-telemetry"):
        assert off not in cmd, f"{off} should be OFF in attribution-clean launch"
    # from-scratch: NO resume
    assert "--resume-from" not in cmd
    # perf-env prefix present
    assert "TAC_MLX_CUSTOM_GROUPED_BACKWARD=1" in cmd
