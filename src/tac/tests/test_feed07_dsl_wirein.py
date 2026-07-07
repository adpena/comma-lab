"""FEED-07a/07b DSL wire-in (#335): council-ready composition + ledger visibility.

The artifact the derived-optimal-next-run symposium picks up: the FEED-07a PRIMARY-arm
composition (rebalanced anisotropic basis + rule-118 analytic lane band + LADDER eased island
birth + Muon warm-start) COMPOSES end-to-end — validate() clean AND the compiled argv parses
through the trainer's REAL argparse. #220 UNBLOCK (2026-07-07): compose-after-downsample landed
in aa_sdf_observation_render, so AACoverageRender now ALSO composes with the base-grid composers
(seed/band/residual) — the trainer's pure guard is asserted to ACCEPT the combinations it used
to refuse. The FEED-07b BUILD levers (FinerBiasInit / LogitAdjust) ride the same class tests.

Triality: DSL leg (the levers) + equations leg (anisotropic_basis_two_regime_allocation_v1 /
step_native_activation_edge_optimality_v1) of DAG FEED-07a/07b. means != ends: composition
plumbing, NOT a score; pointer moves only via a byte-closed exact row.
"""
from __future__ import annotations

import dataclasses as dc
import importlib.util
import sys
from pathlib import Path

import pytest

from tac import witness_autoconfig as wac
from tac.witness_dsl import curriculum_dsl as cd
from tac.witness_dsl import lever_registry as LR

_REPO = Path(__file__).resolve().parents[3]
_TRAINER_PATH = _REPO / "experiments/train_levelset_witness_realized_through_R_mlx.py"

_GT = "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"

_FEED07A_COMPOSITION = (
    "DirectionalBasisRebalance", "AnalyticLaneRenderBand", "SeedIslandEased", "MuonWarmStart",
)
_AA_ARM = ("AACoverageRender", "DirectionalBasisRebalance", "MuonWarmStart")

_NEW_LEVERS = (
    "DirectionalBasisRebalance", "AACoverageRender", "StepNativeActivation", "MuonWarmStart",
    "PersistenceTopology", "MarginFieldHead",
    # FEED-07b BUILD halves (2026-07-07): #310 FINER++ bias-init + #218 loss-time logit adjust.
    "FinerBiasInit", "LogitAdjust",
)


def _render_argv(pairs) -> list[str]:
    argv: list[str] = []
    for flag, val in pairs:
        argv.append(flag)
        if val is not None:
            argv.append(str(val))
    return argv


def test_feed07a_primary_arm_composition_parses_end_to_end():
    """(C) base sealed config + the FEED-07a arm composition -> real-argparse parse. Both
    triality legs: the autoconfig leg (dsl_levers merge) AND the DSL-program leg (validate)."""
    ap = cd.build_real_trainer_parser()
    cfg = wac.derive_sealed_205_config(_GT, num_pairs=600, epochs=1000)
    argv = _render_argv(dc.replace(cfg, dsl_levers=_FEED07A_COMPOSITION).to_trainer_flags("OUT"))
    try:
        ap.parse_args(argv)
    except SystemExit as exc:  # pragma: no cover - diagnostic path
        raise AssertionError(
            f"FEED-07a composition argv rejected by the REAL trainer argparse (rc={exc.code})"
        ) from exc
    # DSL-program leg: the same composition over BASELINE validates clean (never-invent-flags,
    # store_true-C2, type-compat all satisfied by construction).
    prog = cd.BASELINE
    for name in _FEED07A_COMPOSITION:
        prog = prog.with_lever(LR.resolve_composable_lever(name))
    assert prog.validate() == [], "FEED-07a composition must reference only real, type-compatible flags"
    fd = prog.flag_dict()
    # the derived two-regime allocation actually landed in the composed flags (regime-1 default)
    assert fd["--freq-across"] == 32.0 and fd["--freq-along"] == 6.0 and fd["--n-dir-freqs"] == 4
    assert fd["--self-orient"] is True
    assert fd["--lane-render-band"] is True                # rule-118 band ON
    assert fd["--seed-island-eased"] is True               # LADDER eased birth
    assert fd["--muon-warm-start-momentum"] is True        # #270 warm-start
    assert fd["--muon-lr-final-frac"] == 0.1


def test_feed07b_aa_arm_parses_and_now_composes_with_base_grid_composers():
    """(C) #220 UNBLOCK: the AA arm parses through the REAL argparse AND the trainer's pure
    compose guard now ACCEPTS AA x seed/band/residual (compose_fn runs AFTER box_downsample at
    the base grid — the combinations it used to refuse compose by construction). The guard is
    kept (same signature) as the fail-closed home for future fine-grid-only composers."""
    ap = cd.build_real_trainer_parser()
    cfg = wac.derive_sealed_205_config(_GT, num_pairs=600, epochs=1000)
    argv = _render_argv(dc.replace(cfg, dsl_levers=_AA_ARM).to_trainer_flags("OUT"))
    try:
        ap.parse_args(argv)
    except SystemExit as exc:  # pragma: no cover - diagnostic path
        raise AssertionError(f"AA arm argv rejected by the real trainer argparse (rc={exc.code})") from exc
    spec = importlib.util.spec_from_file_location("tl_feed07_guard", str(_TRAINER_PATH))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    # formerly-refused combinations must now be ACCEPTED (no raise)
    mod._validate_aa_compose_compat(True, False, False, True)   # AA x island seed
    mod._validate_aa_compose_compat(True, True, False, False)   # AA x lane band
    mod._validate_aa_compose_compat(True, True, True, True)     # AA x all three
    # and the AA arm composes IN THE SAME PROGRAM as the base-grid composers now
    prog = cd.BASELINE
    for name in ("AACoverageRender", "SeedIslandEased", "AnalyticLaneRenderBand", "MuonWarmStart"):
        prog = prog.with_lever(LR.resolve_composable_lever(name))
    assert prog.validate() == [], "post-#220 the AA arm must compose with seed/band cleanly"


def test_new_levers_composable_and_regime_law_agrees_with_equation():
    """The 6 FEED-07a/07b levers are zero-required-arg single-Lever factories (the sibling
    class-fix predicate) AND the lever's derived allocation agrees with the registered
    canonical equation (triality: DSL leg == equations leg)."""
    comp = LR.name_composable_levers()
    for name in _NEW_LEVERS:
        assert name in comp, f"{name} must be composable via --dsl-lever"
        assert isinstance(LR.resolve_composable_lever(name), cd.Lever)
    from tac.canonical_equations.anisotropic_basis_two_regime_allocation_20260707 import (
        freq_along_for_regime,
    )
    assert freq_along_for_regime(32, "lane_offloaded") == 6
    assert freq_along_for_regime(32, "lane_carried") == 26
    lv = cd.DirectionalBasisRebalance(regime="lane_carried")
    assert lv.overrides["--freq-along"] == 26.0, "DSL leg must emit the equations-leg law"
    with pytest.raises(ValueError, match="unknown regime"):
        cd.DirectionalBasisRebalance(regime="typo_regime")
    # StepNativeActivation guards: anneal-toward-step only (fixed-high-beta diverges, measured)
    with pytest.raises(ValueError, match="beta_start <= beta_end"):
        cd.StepNativeActivation(beta_start=8.0, beta_end=4.0)
    with pytest.raises(ValueError, match="lr_final_frac"):
        cd.MuonWarmStart(lr_final_frac=0.0)


def test_new_levers_land_in_activation_ledger_duty_to_measure(tmp_path):
    """(D) ledger visibility: known_levers() auto-derives from lever_factories(), so the new
    levers appear and — never having fired — sit in never_fired()/duty_to_measure() for the
    #247 costate SENSE layer. No wiring needed; assert the auto-derivation holds (against an
    isolated empty ledger so a live event file cannot mask the derivation)."""
    from tac.witness_dsl.activation_ledger import duty_to_measure, known_levers, never_fired
    known = known_levers()
    for name in _NEW_LEVERS:
        assert name in known, f"{name} missing from the activation ledger's known set"
    empty = tmp_path / "activation_ledger_empty.jsonl"  # nonexistent => zero events
    nf = never_fired(path=empty)
    duty = duty_to_measure(path=empty)
    for name in _NEW_LEVERS:
        assert name in nf, f"{name} must surface as never-fired on an empty ledger"
        assert name in duty, f"{name} must land in the SENSE duty_to_measure queue"
