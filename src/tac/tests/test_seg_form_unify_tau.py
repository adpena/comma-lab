"""--seg-form-unify-tau: the ONE continuous L_τ seg loss + its DSL/mutual-exclusion wiring.

Proves (witness-native schedule derivation `.omx/research/witness_native_schedule_derivation_20260709.md`
§1.2, and the build memo `.omx/research/seg_form_unify_tau_build_20260708.md`):

  1. NUMERICAL EQUIVALENCE. The unified per-pixel kernel L_τ = τ·logsumexp(φ/τ) − φ_y is EXACTLY the
     `ce` branch's per-pixel base at τ=1, → ReLU(−margin) as τ→0, and in the 2-class limit is EXACTLY
     the incumbent `tau_softplus` branch (τ·softplus(−m/τ)) at ANY τ — while the full-multiclass form
     ≥ the top-2 `tau_softplus` at moderate τ (the documented mapping, not a fudge).
  2. DEFAULT-OFF byte-identity: the flag absent => `_seg_form_for_epoch` is the unchanged discrete
     dispatch (the loss form the trainer picks is untouched).
  3. MUTUAL-EXCLUSION: unify + an explicit --tau-softplus-start-epoch is refused (fail-closed).
  4. DSL triality: the SegFormUnifyTau Lever factory holds the flag and lever_registry maps it.
  5. --tau-anneal-shape geometric (the derivation's element-2 0-LOC flip) is a valid choice and the
     render anneal produces a log-spaced (geometric) τ(ep).
"""
from __future__ import annotations

import importlib.util
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

_REPO = Path(__file__).resolve().parents[3]
_LEVELSET = _REPO / "experiments" / "train_levelset_witness_realized_through_R_mlx.py"


def _load_levelset():
    """Load the levelset trainer (it prepends `experiments/` to sys.path AND imports the base)."""
    spec = importlib.util.spec_from_file_location("_unify_levelset_under_test", _LEVELSET)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def _base_module():
    _load_levelset()  # ensures experiments/ is on sys.path
    import train_witness_realized_through_R_mlx as base  # noqa: E402
    return base


# ---------------------------------------------------------------------------- reference per-pixel math
def _rand_logits(seed: int = 0, h: int = 6, w: int = 5, k: int = 5):
    rng = np.random.default_rng(seed)
    logits = rng.standard_normal((1, h, w, k)).astype(np.float32) * 2.0
    y = rng.integers(0, k, size=(1, h, w))
    oh = np.eye(k, dtype=np.float32)[y]  # (1,h,w,k)
    return mx.array(logits), mx.array(oh)


def _ce_perpixel(logits, oh):
    return mx.logsumexp(logits, axis=-1) - mx.sum(logits * oh, axis=-1)


def _tau_softplus_perpixel(logits, oh, tau):
    gt = mx.sum(logits * oh, axis=-1)
    runner = mx.max(logits + oh * (-1e9), axis=-1)
    signed = gt - runner
    z = -signed / tau
    return tau * mx.logaddexp(mx.zeros_like(z), z)


def _relu_neg_margin(logits, oh):
    gt = mx.sum(logits * oh, axis=-1)
    runner = mx.max(logits + oh * (-1e9), axis=-1)
    return mx.maximum(runner - gt, 0.0)  # ReLU(−m), m = gt − runner


# =========================================================================== 1. NUMERICAL EQUIVALENCE
def test_unify_tau_equals_ce_at_tau_1():
    base = _base_module()
    logits, oh = _rand_logits(seed=1)
    lt = np.asarray(base._seg_unify_tau_perpixel(logits, oh, 1.0))
    ce = np.asarray(_ce_perpixel(logits, oh))
    # τ=1: τ·logsumexp(φ/τ) − φ_y == logsumexp(φ) − φ_y (x/1.0==x, 1.0*y==y) — fp tolerance.
    assert np.allclose(lt, ce, rtol=0, atol=1e-5), f"max|Δ|={np.max(np.abs(lt - ce))}"


def test_unify_tau_bit_close_ce_across_seeds():
    base = _base_module()
    for s in range(4):
        logits, oh = _rand_logits(seed=10 + s)
        lt = np.asarray(base._seg_unify_tau_perpixel(logits, oh, 1.0))
        ce = np.asarray(_ce_perpixel(logits, oh))
        assert np.max(np.abs(lt - ce)) < 1e-5


def test_unify_tau_approaches_relu_margin_as_tau_small():
    base = _base_module()
    logits, oh = _rand_logits(seed=2)
    lt = np.asarray(base._seg_unify_tau_perpixel(logits, oh, 1e-3))
    relu = np.asarray(_relu_neg_margin(logits, oh))
    # τ→0: L_τ → max_k φ_k − φ_y = ReLU(−m). At τ=1e-3 the softmax→argmax gap is O(τ·ln K).
    assert np.max(np.abs(lt - relu)) < 5e-2, f"max|Δ|={np.max(np.abs(lt - relu))}"


def test_unify_tau_equals_tau_softplus_in_two_class_limit():
    """The documented mapping: tau_softplus IS the top-2 reduction of L_τ. In a 2-class scenario
    (3 of 5 logits negligible), the full logsumexp == the top-2 logaddexp, so L_τ == τ·softplus(−m/τ)
    EXACTLY at ANY τ."""
    base = _base_module()
    rng = np.random.default_rng(7)
    h, w = 5, 4
    logits = np.full((1, h, w, 5), -1e4, dtype=np.float32)
    # two active classes 0 (=gt) and 1 (=competitor) with O(1) logits; 2,3,4 negligible.
    logits[..., 0] = rng.standard_normal((1, h, w)).astype(np.float32)
    logits[..., 1] = rng.standard_normal((1, h, w)).astype(np.float32)
    oh = np.zeros((1, h, w, 5), dtype=np.float32)
    oh[..., 0] = 1.0
    L, OH = mx.array(logits), mx.array(oh)
    for tau in (1.0, 0.5, 0.31, 0.1):
        lt = np.asarray(base._seg_unify_tau_perpixel(L, OH, tau))
        tsp = np.asarray(_tau_softplus_perpixel(L, OH, tau))
        assert np.max(np.abs(lt - tsp)) < 1e-3, f"tau={tau} max|Δ|={np.max(np.abs(lt - tsp))}"


def test_unify_tau_ge_top2_tau_softplus_at_moderate_tau():
    """Full-multiclass L_τ ≥ the top-2 tau_softplus (the gap = the sub-runner-up logsumexp mass) —
    so L_τ is the PARENT, not a rescale, of tau_softplus at general τ."""
    base = _base_module()
    logits, oh = _rand_logits(seed=3)
    for tau in (0.31, 0.5):
        lt = np.asarray(base._seg_unify_tau_perpixel(logits, oh, tau))
        tsp = np.asarray(_tau_softplus_perpixel(logits, oh, tau))
        assert np.all(lt >= tsp - 1e-5), "L_τ must dominate the top-2 tau_softplus"
        assert float(np.mean(lt)) >= float(np.mean(tsp)) - 1e-6


# =========================================================================== 2. DEFAULT-OFF dispatch
def test_seg_form_for_epoch_default_off_is_discrete():
    m = _load_levelset()
    args = types.SimpleNamespace(seg_form_unify_tau=False, curriculum=True, seg_loss="ce",
                                 tau_softplus_start_epoch=300, l7_start_epoch=800)
    assert m._seg_form_for_epoch(1, args) == "ce"
    assert m._seg_form_for_epoch(300, args) == "tau_softplus"
    assert m._seg_form_for_epoch(800, args) == "l7_softplus"


def test_seg_form_for_epoch_unify_is_constant():
    m = _load_levelset()
    args = types.SimpleNamespace(seg_form_unify_tau=True, curriculum=True, seg_loss="ce",
                                 tau_softplus_start_epoch=300, l7_start_epoch=800)
    for ep in (1, 100, 300, 726, 3000):
        assert m._seg_form_for_epoch(ep, args) == "unify_tau"


def test_unify_tau_argparse_flag_is_store_true_default_off():
    src = _LEVELSET.read_text(encoding="utf-8")
    assert 'ap.add_argument("--seg-form-unify-tau", action="store_true", default=False,' in src
    # the base loss carries the reachable branch + the pure kernel it composes
    bsrc = (_REPO / "experiments" / "train_witness_realized_through_R_mlx.py").read_text(encoding="utf-8")
    assert 'elif form == "unify_tau":' in bsrc
    assert "def _seg_unify_tau_perpixel(" in bsrc


def test_l7_argparse_default_is_disabled_sentinel():
    src = _LEVELSET.read_text(encoding="utf-8")
    assert (
        '"--l7-start-epoch",\n'
        "        type=int,\n"
        "        default=-1,"
    ) in src


def test_l7_default_sentinel_resolves_to_positive_never_runs_form():
    m = _load_levelset()
    resolved, warning = m.resolve_l7_start_epoch(-1, 1000)
    assert resolved == 1001
    assert warning is None


def test_l7_explicit_historical_value_is_preserved_and_warned():
    m = _load_levelset()
    resolved, warning = m.resolve_l7_start_epoch(800, 1000)
    assert resolved == 800
    assert warning is not None
    assert warning["stage"] == "l7_explicit_opt_in_WARN"
    assert warning["classification"] == "MEASURED_DEFECT"
    assert warning["explicit_opt_in"] is True


@pytest.mark.parametrize("requested", [0, -1, 1001, 5000])
def test_l7_explicit_disabled_or_parked_values_do_not_warn(requested):
    m = _load_levelset()
    resolved, warning = m.resolve_l7_start_epoch(requested, 1000)
    assert resolved > 1000
    assert warning is None


def test_trainer_couples_loss_tau_to_render_temp_floored():
    """The coupling cell is set to max(render softmax-temp, --softmax-temp-end) and passed as
    tau_override ONLY for the unify form (byte-identical when off)."""
    src = _LEVELSET.read_text(encoding="utf-8")
    assert '_unify_tau_state["tau"] = max(float(model.softmax_temp), float(args.softmax_temp_end))' in src
    assert 'seg_form == "unify_tau"' in src
    assert "tau_override=_uni_tau" in src


# =========================================================================== 3. MUTUAL-EXCLUSION
def test_mutual_exclusion_refuses_explicit_tau_softplus_start_epoch():
    m = _load_levelset()
    with pytest.raises(ValueError, match="tau-softplus-start-epoch"):
        m.validate_seg_form_unify_tau_config(
            seg_form_unify_tau=True,
            argv=["prog", "--seg-form-unify-tau", "--tau-softplus-start-epoch", "300"])


def test_mutual_exclusion_refuses_equals_form():
    m = _load_levelset()
    with pytest.raises(ValueError):
        m.validate_seg_form_unify_tau_config(
            seg_form_unify_tau=True, argv=["prog", "--tau-softplus-start-epoch=300"])


def test_mutual_exclusion_allows_unify_without_tau_softplus():
    m = _load_levelset()
    # no raise
    m.validate_seg_form_unify_tau_config(seg_form_unify_tau=True, argv=["prog", "--seg-form-unify-tau"])


def test_validate_is_noop_when_unify_off():
    m = _load_levelset()
    # off => the guard never inspects argv (tau-softplus-start-epoch is a legal discrete-run flag)
    m.validate_seg_form_unify_tau_config(
        seg_form_unify_tau=False, argv=["prog", "--tau-softplus-start-epoch", "300"])


# =========================================================================== 4. DSL triality
def test_dsl_segformunifytau_factory():
    from tac.witness_dsl.curriculum_dsl import SegFormUnifyTau
    lev = SegFormUnifyTau()
    assert lev.name == "seg_form_unify_tau"
    assert lev.overrides == {"--seg-form-unify-tau": True}
    assert lev.epochs_delta == 0  # full-run schedule reshape, not a warm-start arm


def test_dsl_factory_compiles_to_bare_flag():
    from tac.witness_dsl.curriculum_dsl import SegFormUnifyTau
    lev = SegFormUnifyTau()
    # a True store_true value compiles to the bare flag (compile_trainer_argv True-branch).
    val = lev.overrides["--seg-form-unify-tau"]
    assert val is True


def test_lever_registry_maps_seg_form_unify_tau_flag():
    from tac.witness_dsl.lever_registry import completeness
    c = completeness()
    assert "--seg-form-unify-tau" not in set(c.unmapped), "flag must be DSL-held (mapped)"


# =========================================================================== 5. geometric anneal shape
def test_tau_anneal_shape_geometric_is_a_valid_choice():
    src = _LEVELSET.read_text(encoding="utf-8")
    assert 'ap.add_argument("--tau-anneal-shape", choices=["cosine", "geometric", "cosine_hold"]' in src


def test_geometric_render_anneal_is_log_spaced():
    m = _load_levelset()
    args = types.SimpleNamespace(anneal_epochs=1000, epochs=1000, tau_anneal_shape="geometric",
                                 softmax_temp_start=1.0, softmax_temp_end=0.31, tau_hold_frac=1.0)
    t1 = m._softmax_temp_for_epoch(1, args)
    tend = m._softmax_temp_for_epoch(1000, args)
    tmid = m._softmax_temp_for_epoch(500, args)  # prog≈0.4995
    assert t1 == pytest.approx(1.0, abs=1e-6)
    assert tend == pytest.approx(0.31, abs=1e-6)
    # geometric: τ(prog) = start·(end/start)^prog => log-linear in prog (NOT the cosine midpoint).
    prog = (500 - 1) / (1000 - 1)
    expected_mid = 1.0 * (0.31 / 1.0) ** prog
    assert tmid == pytest.approx(expected_mid, rel=1e-6)
    # geometric spends MORE epochs at small τ than cosine: its midpoint is BELOW the cosine midpoint.
    cos_mid = 0.31 + 0.5 * (1.0 - 0.31) * (1 + math.cos(math.pi * prog))
    assert tmid < cos_mid
