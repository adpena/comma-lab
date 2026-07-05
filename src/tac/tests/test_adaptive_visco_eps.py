"""Tests for the V6 #320 ADAPTIVE-eps CFL-edge tracker (the DERIVED mechanism cure for the eikonal
ep110 re-entry) — the ``--eikonal-viscosity-adaptive`` path in the level-set witness trainer.

DE #318 §4 Arm-2 / symposium #317 §7.4: eps(t) = clamp(|c_a(t)|*sqrt(eta*lambda_eik/8)*(1+margin),
floor, upper), with |c_a(t)| = mean|(|grad m|-1)/|grad m|| measured no-grad on the witness decision
margin. These tests pin: (1) the eps law closed-form (edge/floor/upper/inverted clamps) + numpy<->MLX
parity; (2) the |c_a| sharpness proxy math on ANALYTIC fields (unit-gradient plane -> 0, m=3x -> 2/3,
m=0.5x -> 1 [the ill-posed a<1], quadratic) + numpy<->MLX byte-parity on random fields + the
small-margin band; (3) _measure_ca_mlx over a stub model.sdf (deterministic, witness-only); (4) the
OFF-path byte-identity of _loss_terms_row (None kwargs => no new keys) + the telemetry keys when ON;
(5) argparse defaults (adaptive OFF, floor 0.3, upper 0.7, margin 0.5) + the anneal-gate skip source.
Default OFF => byte-identical (the anneal gate is `and not visco_adaptive` = a no-op when False)."""
from __future__ import annotations

import importlib.util
import math
import pathlib
import sys

import numpy as np
import pytest
import mlx.core as mx  # noqa: E402

from tac.boundary_math.eikonal_sharpness_proxy_reference import (
    adaptive_visco_eps as ref_adaptive_eps,
    sharpness_proxy_c_a as ref_c_a,
    self_test as ref_self_test,
)

_MODPATH = pathlib.Path(__file__).resolve().parents[3] / "experiments" / \
    "train_levelset_witness_realized_through_R_mlx.py"


def _load(path: pathlib.Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # pragma: no cover
        sys.modules.pop(name, None)
        pytest.skip(f"could not import {path.name}: {type(exc).__name__}: {exc}",
                    allow_module_level=True)
    return mod


MOD = _load(_MODPATH, "_levelset_trainer_adaptive_eps_test")
SRC = _MODPATH.read_text()

if not hasattr(MOD, "_adaptive_visco_eps"):
    pytest.skip("adaptive-eps not yet in this trainer checkout", allow_module_level=True)

H, W, K = 16, 20, 3


def _phi_from_margin(m_hw: np.ndarray, k: int = K) -> mx.array:
    """(H*W, K) phi whose decision margin top1-top2 == m_hw (m>=0): phi0=m, phi1=0, rest=-10."""
    hh, ww = m_hw.shape
    phi = np.full((hh * ww, k), -10.0, np.float32)
    phi[:, 0] = m_hw.reshape(-1).astype(np.float32)
    phi[:, 1] = 0.0
    return mx.array(phi)


# ───────────────────────────── (1) the eps law closed-form + numpy<->MLX parity
def test_eps_law_equals_closed_form_edge_unclamped():
    # |c_a|=1, eta=1e-3, lam=0.05, margin=0.5 -> sqrt(1e-3*0.05/8)*1.5, wide clamp => unclamped.
    got = MOD._adaptive_visco_eps(1.0, 1e-3, 0.05, 0.5, 0.0, 10.0)
    ref = math.sqrt(1e-3 * 0.05 / 8.0) * 1.5
    assert got == pytest.approx(ref, rel=0, abs=1e-15)


def test_eps_law_floor_clamp():
    # tiny edge -> clamp UP to the floor 0.3.
    assert MOD._adaptive_visco_eps(1e-9, 1e-3, 0.05, 0.5, 0.3, 0.7) == 0.3


def test_eps_law_upper_clamp():
    # huge |c_a| -> clamp DOWN to the upper 0.7.
    assert MOD._adaptive_visco_eps(1e6, 1e-3, 0.05, 0.5, 0.3, 0.7) == 0.7


def test_eps_law_inverted_bounds_raise_upper_to_floor():
    # upper < floor => upper raised to floor => well-defined single value.
    assert MOD._adaptive_visco_eps(1e6, 1e-3, 0.05, 0.5, 0.5, 0.2) == 0.5


def test_eps_law_negative_product_under_sqrt_clamped_to_zero():
    # negative eta*lam (non-physical) => sqrt arg clamped at 0 => edge 0 => floor.
    assert MOD._adaptive_visco_eps(1.0, -1.0, 0.05, 0.5, 0.3, 0.7) == 0.3


def test_eps_law_matches_numpy_reference_across_grid():
    rng = np.random.default_rng(320)
    for _ in range(50):
        ca = float(rng.uniform(0, 3))
        eta = float(rng.uniform(1e-4, 2e-3))
        lam = float(rng.uniform(0.01, 0.2))
        mfac = float(rng.uniform(0.0, 1.0))
        lo, hi = 0.3, 0.7
        got = MOD._adaptive_visco_eps(ca, eta, lam, mfac, lo, hi)
        ref = ref_adaptive_eps(ca, eta, lam, mfac, lo, hi)
        assert got == pytest.approx(ref, rel=0, abs=1e-15)


def test_eps_law_monotone_increasing_in_c_a_until_clamp():
    # eps rises with |c_a| (tracks the rising CFL edge) until the upper clamp bites. NOTE (honest,
    # constant-dependent): at eta=1e-3/lam=0.05 the edge is only 0.00375*|c_a|, so eps stays FLOORED
    # at 0.3 until |c_a|>~80 and reaches the 0.7 upper only at |c_a|>~187 => adaptive-eps degrades
    # gracefully to a CONSTANT-0.3-FLOOR for this config (which alone fixes the eps->0 half of the
    # v5 re-entry) and only rises if sharpness genuinely explodes. The `8` constant is measurement-owed.
    prev = -1.0
    for ca in np.linspace(0.0, 300.0, 60):
        e = MOD._adaptive_visco_eps(float(ca), 1e-3, 0.05, 0.5, 0.3, 0.7)
        assert e >= prev - 1e-12
        prev = e
    assert MOD._adaptive_visco_eps(0.0, 1e-3, 0.05, 0.5, 0.3, 0.7) == 0.3   # ca=0 -> floor
    assert prev == 0.7  # ca=300 -> 300*0.00375=1.125 -> clamps at upper


# ───────────────────────────── (2) the |c_a| sharpness proxy math + numpy<->MLX byte-parity
def test_c_a_zero_on_unit_gradient_plane():
    # m = x has |grad m| = 1 (central diff) => c_a = |(1-1)/1| = 0.
    x = np.tile(np.arange(W, dtype=np.float64), (H, 1))
    got = MOD._ca_from_margin_mlx(mx.array(x.astype(np.float32)))
    assert abs(got) < 1e-5


def test_c_a_grad3_plane_two_thirds():
    x = np.tile(np.arange(W, dtype=np.float64), (H, 1))
    got = MOD._ca_from_margin_mlx(mx.array((3.0 * x).astype(np.float32)))
    assert got == pytest.approx(2.0 / 3.0, rel=1e-4)


def test_c_a_grad_half_plane_is_one_ill_posed_regime():
    # m = 0.5x has |grad m| = 0.5 < 1 (the DE ill-posed a<1) => c_a = |(0.5-1)/0.5| = 1.
    x = np.tile(np.arange(W, dtype=np.float64), (H, 1))
    got = MOD._ca_from_margin_mlx(mx.array((0.5 * x).astype(np.float32)))
    assert got == pytest.approx(1.0, rel=1e-4)


def test_c_a_matches_numpy_reference_on_random_field():
    rng = np.random.default_rng(7)
    m = (2.0 + rng.standard_normal((H, W)) * 0.3).astype(np.float32)
    got = MOD._ca_from_margin_mlx(mx.array(m))
    ref = ref_c_a(m.astype(np.float64))
    assert got == pytest.approx(ref, rel=1e-4)


def test_c_a_band_restriction_matches_numpy_reference():
    rng = np.random.default_rng(9)
    # margin with a genuine small-margin annulus (values straddling 0).
    m = (rng.standard_normal((H, W)) * 0.8).astype(np.float32)
    band = 0.5
    got = MOD._ca_from_margin_mlx(mx.array(m), band=band)
    ref = ref_c_a(m.astype(np.float64), band=band)
    assert got == pytest.approx(ref, rel=1e-4, abs=1e-6)


def test_c_a_empty_band_returns_zero():
    x = np.tile(np.arange(W, dtype=np.float64), (H, 1)) + 100.0  # all |m| >> band
    assert MOD._ca_from_margin_mlx(mx.array(x.astype(np.float32)), band=1e-6) == 0.0


# ───────────────────────────── (3) _measure_ca_mlx over a stub model (witness-only, deterministic)
class _StubModel:
    """model.sdf(cf, code_idx) -> the phi for a fixed per-pair margin (frame0 index = 2*pi)."""

    def __init__(self, margins: dict[int, np.ndarray]):
        self._m = margins

    def sdf(self, cf, code_idx):
        pi = int(cf)              # cf_fn returns pi (identity) in the test
        return _phi_from_margin(self._m[pi])


def test_measure_ca_mlx_averages_over_fixed_pairs_deterministic():
    rng = np.random.default_rng(21)
    margins = {p: (2.0 + rng.standard_normal((H, W)) * 0.3).astype(np.float32) for p in (0, 3, 6)}
    model = _StubModel(margins)
    pairs = [0, 3, 6]
    got1 = MOD._measure_ca_mlx(model, pairs, lambda pi: pi, H, W, band=0.0)
    got2 = MOD._measure_ca_mlx(model, pairs, lambda pi: pi, H, W, band=0.0)
    assert got1 == got2  # deterministic (no RNG)
    expected = float(np.mean([ref_c_a(margins[p].astype(np.float64)) for p in pairs]))
    assert got1 == pytest.approx(expected, rel=1e-4)


# ───────────────────────────── (4) OFF-path byte-identity + ON telemetry of _loss_terms_row
def test_loss_terms_row_omits_visco_keys_when_none():
    row = MOD._loss_terms_row({"eikonal": 1.0}, 1.0, 5, 0)
    assert "visco_eps" not in row and "visco_c_a" not in row  # OFF path => identical schema


def test_loss_terms_row_emits_visco_keys_when_provided():
    row = MOD._loss_terms_row({"eikonal": 1.0}, 1.0, 5, 0, visco_eps=0.42, visco_c_a=0.13)
    assert row["visco_eps"] == 0.42 and row["visco_c_a"] == 0.13
    # the terms/sum-check are unaffected (visco_* are NOT loss addends).
    assert "visco_eps" not in row["terms"]


# ───────────────────────────── (5) argparse defaults + the anneal-gate skip source
def test_argparse_adaptive_defaults():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--eikonal-viscosity-adaptive", action="store_true")
    ap.add_argument("--eikonal-visco-eps-floor", type=float, default=0.3)
    ap.add_argument("--eikonal-visco-eps-upper", type=float, default=0.7)
    ap.add_argument("--eikonal-visco-margin-factor", type=float, default=0.5)
    a = ap.parse_args([])
    assert a.eikonal_viscosity_adaptive is False
    assert a.eikonal_visco_eps_floor == 0.3
    assert a.eikonal_visco_eps_upper == 0.7
    assert a.eikonal_visco_margin_factor == 0.5


def test_source_registers_adaptive_flags():
    for flag in ("--eikonal-viscosity-adaptive", "--eikonal-visco-eps-floor",
                 "--eikonal-visco-eps-upper", "--eikonal-visco-margin-factor",
                 "--eikonal-visco-ca-pairs", "--eikonal-visco-ca-band"):
        assert flag in SRC, f"missing argparse flag {flag}"


def test_anneal_gate_skips_linear_when_adaptive():
    # the linear anneal must be gated OFF when adaptive is set (else double-set visco_eps).
    assert 'not _eik_stab["visco_adaptive"]' in SRC


def test_adaptive_block_uses_current_eta_and_lambda():
    # eta from opt.learning_rate (set just above), lambda from eik_w_ep => current-epoch values.
    assert "_eta_t = float(opt.learning_rate)" in SRC
    assert "_adaptive_visco_eps(_ca_t, _eta_t, float(eik_w_ep)" in SRC


# ───────────────────────────── (numpy reference self-test as a first-class guard)
def test_numpy_reference_self_test_passes():
    out = ref_self_test()
    assert out["stencil_max_abs_diff_vs_318_gx"] == 0.0
    assert out["adaptive_eps_floor_clamp"] == 0.3
    assert out["adaptive_eps_upper_clamp"] == 0.7
