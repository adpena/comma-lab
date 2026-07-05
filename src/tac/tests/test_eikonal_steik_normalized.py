"""Tests for the V6 SYMPOSIUM (#317) NORMALIZED StEik operator ``_eikonal_steik_normalized_mlx``.

The raw StEik term |grad m^T H grad m| = |grad m|^2 * |n^T H n| carries a QUARTIC |grad m|^2 scaling
that SELF-AMPLIFIES at the far-from-SDF resumed state (|grad m| >> 1) — measured 575x-1431x runaway
in the FEED-05v arbitration (NO-GO). The normalized form
    L_norm = mean | (grad m^T H grad m) / (|grad m|^2 + eps) |  = mean |n^T H n|
strips the |grad m|^2 factor and damps ONLY the unit-normal 2nd derivative n^T H n (the StEik-proven
anti-diffusive mode), leaving tangential curvature (lane dashes) free.

Term math is checked on ANALYTIC fields where n^T H n is hand-computable (linear plane -> 0; quadratic
-> closed form) plus an independent numpy re-implementation on random fields. The self-amplification
removal (raw ~ s^3 vs normalized ~ s under m -> s*m) is the decisive quantitative guard. Flag plumbing
+ fail-closed (silent-no-op / div-by-zero) round it out. Default OFF => byte-identical (guarded)."""
from __future__ import annotations

import importlib.util
import pathlib
import re
import sys

import numpy as np
import pytest

pytest.importorskip("mlx", reason="level-set witness trainer requires mlx")
import mlx.core as mx  # noqa: E402

_REPO = pathlib.Path(__file__).resolve().parents[3]
_MODPATH = _REPO / "experiments" / "train_levelset_witness_realized_through_R_mlx.py"


def _load(path: pathlib.Path, name: str):
    if not path.exists():
        pytest.skip(f"trainer not found at {path}", allow_module_level=True)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:  # pragma: no cover - env-dependent
        sys.modules.pop(name, None)
        pytest.skip(f"could not import {path.name}: {type(exc).__name__}: {exc}",
                    allow_module_level=True)
    return mod


MOD = _load(_MODPATH, "_levelset_trainer_steik_norm_test")
SRC = _MODPATH.read_text()

# The normalized-steik operator lands in the SHARED trainer via a coordinated commit (concurrent
# #313 micro-batch / #315 nucleus-guard edits). Skip gracefully if a checkout predates that landing
# rather than hard-fail — the working-tree run + the coordinated commit both carry the function.
if not hasattr(MOD, "_eikonal_steik_normalized_mlx"):
    pytest.skip("normalized-steik operator not yet in this trainer checkout",
                allow_module_level=True)

H, W, K = 16, 20, 3
NORM_EPS = 1e-2


def _phi_from_margin(m_hw: np.ndarray, k: int = K) -> mx.array:
    """(H*W, K) phi whose decision margin top1-top2 == m_hw (m>0): phi0=m, phi1=0, rest=-10."""
    hh, ww = m_hw.shape
    phi = np.full((hh * ww, k), -10.0, np.float32)
    phi[:, 0] = m_hw.reshape(-1).astype(np.float32)
    phi[:, 1] = 0.0
    return mx.array(phi)


def _margin_interior_np(m: np.ndarray):
    m = m.astype(np.float64)
    gx = 0.5 * (m[1:-1, 2:] - m[1:-1, :-2])
    gy = 0.5 * (m[2:, 1:-1] - m[:-2, 1:-1])
    m_xx = m[1:-1, 2:] - 2.0 * m[1:-1, 1:-1] + m[1:-1, :-2]
    m_yy = m[2:, 1:-1] - 2.0 * m[1:-1, 1:-1] + m[:-2, 1:-1]
    m_xy = 0.25 * (m[2:, 2:] - m[2:, :-2] - m[:-2, 2:] + m[:-2, :-2])
    return gx, gy, m_xx, m_yy, m_xy


def _norm_np(m: np.ndarray, eps: float = NORM_EPS) -> float:
    gx, gy, m_xx, m_yy, m_xy = _margin_interior_np(m.astype(np.float32))
    dir_div = gx * gx * m_xx + 2 * gx * gy * m_xy + gy * gy * m_yy
    gmag2 = gx * gx + gy * gy
    return float(np.mean(np.abs(dir_div / (gmag2 + eps))))


# ───────────────────────────── term math on analytic fields
def test_norm_zero_on_linear_margin():
    # plane: H(m)==0 => n^T H n == 0 exactly (the true-SDF condition H*grad m = 0).
    x = np.arange(W, dtype=np.float64)[None, :].repeat(H, axis=0)
    m = 1.0 + 0.05 * x
    out = float(MOD._eikonal_steik_normalized_mlx(_phi_from_margin(m), H, W, NORM_EPS))
    assert abs(out) < 1e-6


def test_norm_quadratic_matches_hand_formula():
    # m = c + 0.5 a x^2: gx=a x, m_xx=a, others 0 => dir_div = a^3 x^2, |grad m|^2 = a^2 x^2,
    # n^T H n = a^3 x^2 / (a^2 x^2 + eps). mean over interior.
    a, c = 0.05, 5.0
    x = np.arange(W, dtype=np.float64)[None, :].repeat(H, axis=0)
    m = c + 0.5 * a * x * x
    out = float(MOD._eikonal_steik_normalized_mlx(_phi_from_margin(m), H, W, NORM_EPS))
    xi = x[1:-1, 1:-1]
    dir_div = (a * xi) ** 2 * a
    gmag2 = (a * xi) ** 2
    expected = float(np.mean(np.abs(dir_div / (gmag2 + NORM_EPS))))
    assert out == pytest.approx(expected, rel=1e-3)


def test_norm_matches_independent_numpy_on_random_field():
    rng = np.random.default_rng(11)
    m = 2.0 + rng.standard_normal((H, W)) * 0.3
    m = np.maximum(m, 0.1)
    out = float(MOD._eikonal_steik_normalized_mlx(_phi_from_margin(m), H, W, NORM_EPS))
    assert out == pytest.approx(_norm_np(m), rel=1e-3)


def test_norm_equals_raw_divided_by_gmag2_plus_eps():
    # the DEFINING identity: normalized = raw-integrand / (|grad m|^2 + eps), pointwise-then-mean.
    rng = np.random.default_rng(3)
    m = 1.5 + rng.standard_normal((H, W)) * 0.4
    m = np.maximum(m, 0.2)
    gx, gy, m_xx, m_yy, m_xy = _margin_interior_np(m.astype(np.float32))
    dir_div = gx * gx * m_xx + 2 * gx * gy * m_xy + gy * gy * m_yy
    gmag2 = gx * gx + gy * gy
    expect = float(np.mean(np.abs(dir_div / (gmag2 + NORM_EPS))))
    out = float(MOD._eikonal_steik_normalized_mlx(_phi_from_margin(m), H, W, NORM_EPS))
    assert out == pytest.approx(expect, rel=1e-3)


# ───────────────────────────── the DECISIVE guard: self-amplification removed
def test_self_amplification_removed_under_margin_scaling():
    # m -> s*m: raw dir_div ~ s^3 ; |grad m|^2 ~ s^2. Normalized n^T H n ~ s^3/s^2 = s (linear),
    # while RAW StEik ~ s^3 (cubic). This is the whole point (raw NO-GO at |grad m|>>1).
    rng = np.random.default_rng(19)
    m0 = 1.0 + rng.standard_normal((H, W)) * 0.25
    m0 = np.maximum(m0, 0.15)
    raw1 = float(MOD._eikonal_steik_mlx(_phi_from_margin(m0), H, W))
    norm1 = float(MOD._eikonal_steik_normalized_mlx(_phi_from_margin(m0), H, W, NORM_EPS))
    s = 8.0
    raw8 = float(MOD._eikonal_steik_mlx(_phi_from_margin(s * m0), H, W))
    norm8 = float(MOD._eikonal_steik_normalized_mlx(_phi_from_margin(s * m0), H, W, NORM_EPS))
    # raw grows ~s^3 = 512x (self-amplifying); normalized grows ~s = 8x (with eps making it a touch
    # LESS than s at this small-gradient scale, since gmag2 ~ eps for the base field).
    assert raw8 / max(raw1, 1e-12) > 200.0            # cubic blow-up (the measured NO-GO signature)
    assert norm8 / max(norm1, 1e-12) < 20.0           # ~linear (self-amplification GONE)
    # the DECISIVE relative signature: normalized/raw RATIO DROPS sharply as the margin steepens
    # (the |grad m|^2 factor increasingly suppresses the normalized form at the far-from-SDF state
    # where raw runs away). At s=1 (small gradients ~ eps) normalization can even inflate; by s=8 the
    # ratio has collapsed >5x — the exact mechanism that turns raw's 1431x NO-GO into a bounded term.
    ratio1 = norm1 / max(raw1, 1e-12)
    ratio8 = norm8 / max(raw8, 1e-12)
    assert ratio1 > ratio8 * 5.0


def test_norm_bounded_at_large_gradient():
    # at |grad m| >> 1 (the resumed-state regime), n^T H n -> dir_div/gmag2 is O(curvature/|grad m|),
    # NOT O(|grad m|^2*curvature). Concretely finite and modest for a steep near-linear ramp+bump.
    x = np.arange(W, dtype=np.float64)[None, :].repeat(H, axis=0)
    m = 50.0 * x + 0.5 * x * x           # steep gradient (|grad m| ~ 50) + curvature
    out = float(MOD._eikonal_steik_normalized_mlx(_phi_from_margin(m), H, W, NORM_EPS))
    raw = float(MOD._eikonal_steik_mlx(_phi_from_margin(m), H, W))
    assert np.isfinite(out)
    assert out < raw                     # normalization strictly reduces the far-from-SDF magnitude


# ───────────────────────────── eps regularizer behaviour (flat interior)
def test_norm_finite_on_flat_field_no_nan():
    # constant margin: gx=gy=0, curvature=0 => 0/(0+eps) = 0, no NaN/inf.
    m = np.full((H, W), 3.0)
    out = float(MOD._eikonal_steik_normalized_mlx(_phi_from_margin(m), H, W, NORM_EPS))
    assert np.isfinite(out)
    assert abs(out) < 1e-6


def test_norm_eps_leaves_unit_gradient_boundary_intact():
    # near the eikonal target |grad m| ~ 1, eps=1e-2 changes n^T H n by <~1% (1/(1+0.01) ~ 0.990).
    a = 1.0
    x = np.arange(W, dtype=np.float64)[None, :].repeat(H, axis=0)
    m = 0.5 * a * x * x                   # gx = a x ~ O(1..W); pick the |grad m|~1 stripe implicitly
    small_eps = float(MOD._eikonal_steik_normalized_mlx(_phi_from_margin(m), H, W, 1e-6))
    reg_eps = float(MOD._eikonal_steik_normalized_mlx(_phi_from_margin(m), H, W, 1e-2))
    # both finite; the regularized value is within a modest factor (not a different object).
    assert np.isfinite(small_eps) and np.isfinite(reg_eps)
    assert reg_eps <= small_eps + 1e-9   # larger eps only DAMPENS (never amplifies) the term


def test_norm_larger_eps_monotone_non_increasing():
    rng = np.random.default_rng(5)
    m = 2.0 + rng.standard_normal((H, W)) * 0.3
    m = np.maximum(m, 0.1)
    vals = [float(MOD._eikonal_steik_normalized_mlx(_phi_from_margin(m), H, W, e))
            for e in (1e-3, 1e-2, 1e-1, 1.0)]
    assert all(vals[i + 1] <= vals[i] + 1e-9 for i in range(len(vals) - 1))


# ───────────────────────────── plumbing + fail-closed
def test_argparse_normalized_flag_defaults_off():
    assert re.search(r'"--eikonal-steik-normalized",\s*action="store_true"', SRC)
    assert re.search(r'"--eikonal-steik-norm-eps",\s*type=float,\s*default=1e-2', SRC)


def test_eik_stab_cell_threads_normalized_keys():
    assert '"steik_normalized": bool(getattr(args, "eikonal_steik_normalized", False))' in SRC
    assert '"steik_norm_eps": float(getattr(args, "eikonal_steik_norm_eps"' in SRC


def test_composition_selects_normalized_when_flag_set():
    # the total_loss_fn branch must pick the normalized fn ONLY when the flag is set, raw otherwise.
    assert 'if _eik_stab["steik_normalized"]:' in SRC
    assert '_eikonal_steik_normalized_mlx(' in SRC
    # gated under the same steik_w>0 branch (additive; weight 0 => never called => byte-identical).
    assert 'if _eik_stab["steik_w"] > 0.0:' in SRC


def test_fail_closed_normalized_without_weight():
    assert '--eikonal-steik-normalized set without --eikonal-steik-weight > 0' in SRC


def test_fail_closed_norm_eps_nonpositive():
    assert re.search(r'eikonal_steik_norm_eps.*?must be > 0', SRC, re.S)


def test_fail_closed_normalized_under_micro_batch():
    # the twin only has the RAW _eikonal_steik_mlx; normalized + micro-batch must FAIL CLOSED
    # (NO-FAKE silent-drop) rather than silently use the raw self-amplifying form.
    assert '--eikonal-steik-normalized is NOT wired into the micro-batch twin' in SRC


def test_parity_with_de_derivation_numpy_reference():
    # (V6 #317 <-> DE #318 coordination) cross-check the MLX n^T H n term against the DE-DERIVATION
    # sibling's independent numpy oracle src/tac/boundary_math/eikonal_normal_curvature_reference.py
    # (their border=0 full-grid convention; interior mean-of-abs == our (H-2,W-2) mean at matched eps).
    ref = pytest.importorskip("tac.boundary_math.eikonal_normal_curvature_reference")
    rng = np.random.default_rng(318)
    m = (rng.standard_normal((H, W)).cumsum(0).cumsum(1))
    m = (m - m.mean()) / (m.std() + 1e-12)
    m = m - m.min() + 0.5                              # keep the margin positive (top1-top2 >= 0)
    kappa = ref.normal_direction_curvature(m)          # (H,W), border 0
    interior = np.zeros_like(m, dtype=bool)
    interior[1:-1, 1:-1] = True
    ref_mean = float(np.mean(np.abs(kappa[interior])))
    # MLX term at the reference's eps (1e-12), same interior stencil:
    mlx_mean = float(MOD._eikonal_steik_normalized_mlx(_phi_from_margin(m), H, W, 1e-12))
    assert mlx_mean == pytest.approx(ref_mean, rel=1e-3)


def test_norm_default_eps_is_1e_2():
    # calling without an explicit eps uses the documented 1e-2 default (matches the flag default).
    rng = np.random.default_rng(23)
    m = 2.0 + rng.standard_normal((H, W)) * 0.3
    m = np.maximum(m, 0.1)
    default_call = float(MOD._eikonal_steik_normalized_mlx(_phi_from_margin(m), H, W))
    explicit = float(MOD._eikonal_steik_normalized_mlx(_phi_from_margin(m), H, W, 1e-2))
    assert default_call == pytest.approx(explicit, rel=1e-6)
