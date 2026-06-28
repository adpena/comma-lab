"""Tests for the LEVEL-SET witness FiLM-rank-fix (LEVER-A) + thin-lane prior (LEVER-B).

PURE CPU tests (NO GPU training launch; the MLX blocks force the CPU device so they never contend
with a running GPU run). Locks, in priority order:

  * BASELINE PRESERVATION: with the levers OFF the numpy ONE CODEPATH forward is BYTE-IDENTICAL to
    the pre-LEVER-A formula, and the OPTIONAL film_pl/concat_pl routes are no-ops at identity init.
  * The pure-numpy primitives (participation ratio + rank-floor penalty + thin-lane weight map) are
    correct + deterministic, and the rank-floor gradient (MLX) OPPOSES rank-1 collapse.
"""
import numpy as np
import pytest

from tac.boundary_math.lever_b_levelset_generator import (
    _act,
    film_modulation_participation_ratio,
    film_rank_floor_penalty,
    lane_thin_weight_map,
    levelset_rgb_forward_numpy,
)

# ---------------------------------------------------------------------------
# fixtures: a tiny deterministic deploy-param set + feats/code (MLX-free)
# ---------------------------------------------------------------------------
_N_HIDDEN = 2
_HIDDEN = 8
_MOD = 4
_NCLS = 5
_INFEAT = 6
_P = 10


def _base_params(seed: int = 0) -> dict[str, np.ndarray]:
    # small magnitude (0.3) so the relu trunk stays in a finite range (no fp overflow noise); the
    # byte-identical check is unaffected by scale.
    rng = np.random.default_rng(seed)
    s = np.float32(0.3)
    p = {
        "in_proj.weight": (rng.standard_normal((_HIDDEN, _INFEAT)) * s).astype(np.float32),
        "in_proj.bias": (rng.standard_normal(_HIDDEN) * s).astype(np.float32),
        "film.weight": (rng.standard_normal((2 * _HIDDEN * _N_HIDDEN, _MOD)) * s).astype(np.float32),
        "film.bias": (rng.standard_normal(2 * _HIDDEN * _N_HIDDEN) * s).astype(np.float32),
        "out_sdf.weight": (rng.standard_normal((_NCLS, _HIDDEN)) * s).astype(np.float32),
        "out_sdf.bias": (rng.standard_normal(_NCLS) * s).astype(np.float32),
        "out_tex.weight": (rng.standard_normal((3, _HIDDEN)) * s).astype(np.float32),
        "out_tex.bias": (rng.standard_normal(3) * s).astype(np.float32),
        "palette": (rng.standard_normal((_NCLS, 3)) * s).astype(np.float32),
    }
    for li in range(_N_HIDDEN):
        p[f"hidden.{li}.weight"] = (rng.standard_normal((_HIDDEN, _HIDDEN)) * s).astype(np.float32)
        p[f"hidden.{li}.bias"] = (rng.standard_normal(_HIDDEN) * s).astype(np.float32)
    return p


def _feats_code(seed: int = 1):
    rng = np.random.default_rng(seed)
    feats = rng.standard_normal((_P, _INFEAT)).astype(np.float32)
    code = rng.standard_normal(_MOD).astype(np.float32)
    return feats, code


def _fwd(params, feats, code):
    return levelset_rgb_forward_numpy(
        params, feats, code, n_hidden=_N_HIDDEN, hidden_dim=_HIDDEN, n_classes=_NCLS,
        activation="relu", softmax_temp=1.0, wire_w0=20.0, wire_s0=10.0, hosc_beta=4.0,
        hosc_omega=1.0, chroma=True)


def _reference_original_forward(params, feats, code):
    """The PRE-LEVER-A numpy forward formula, using the SAME ``_act``. Byte-identical reference."""
    p = {k: np.asarray(v, np.float64) for k, v in params.items()}
    feats64 = np.asarray(feats, np.float64)
    code64 = np.asarray(code, np.float64)
    akw = dict(w0=20.0, s0=10.0, beta=4.0, omega=1.0)
    h = _act(feats64 @ p["in_proj.weight"].T + p["in_proj.bias"], "relu", **akw)
    film = (code64 @ p["film.weight"].T + p["film.bias"]).reshape(_N_HIDDEN, 2, _HIDDEN)
    for li in range(_N_HIDDEN):
        h = _act((h @ p[f"hidden.{li}.weight"].T + p[f"hidden.{li}.bias"]) * (1.0 + film[li, 0]) + film[li, 1],
                 "relu", **akw)
    phi = h @ p["out_sdf.weight"].T + p["out_sdf.bias"]
    tex = h @ p["out_tex.weight"].T + p["out_tex.bias"]
    z = phi / 1.0
    z = z - z.max(axis=-1, keepdims=True)
    soft = np.exp(z)
    soft = soft / soft.sum(axis=-1, keepdims=True)
    base = soft @ p["palette"]
    rgb = (1.0 / (1.0 + np.exp(-(base + tex)))) * 255.0
    return rgb.astype(np.float32), phi.astype(np.float32)


# ===========================================================================
# 1. BASELINE PRESERVATION (the binding constraint)
# ===========================================================================
def test_numpy_forward_baseline_byte_identical_to_original():
    p = _base_params()
    feats, code = _feats_code()
    rgb, phi = _fwd(p, feats, code)
    rgb_ref, phi_ref = _reference_original_forward(p, feats, code)
    assert np.array_equal(rgb, rgb_ref), "LEVER-A refactor changed the default-off numpy forward"
    assert np.array_equal(phi, phi_ref)


def test_numpy_forward_film_pl_zero_init_is_noop():
    p = _base_params()
    feats, code = _feats_code()
    rgb0, phi0 = _fwd(p, feats, code)
    p_pl = dict(p)
    for li in range(_N_HIDDEN):
        p_pl[f"film_pl.{li}.weight"] = np.zeros((2 * _HIDDEN, _MOD), np.float32)
        p_pl[f"film_pl.{li}.bias"] = np.zeros(2 * _HIDDEN, np.float32)
    rgb1, phi1 = _fwd(p_pl, feats, code)
    assert np.array_equal(rgb0, rgb1), "zero-init film_pl must be an exact no-op (identity-residual)"
    assert np.array_equal(phi0, phi1)


def test_numpy_forward_concat_zero_init_is_noop():
    p = _base_params()
    feats, code = _feats_code()
    rgb0, phi0 = _fwd(p, feats, code)
    p_cc = dict(p)
    for li in range(_N_HIDDEN):
        p_cc[f"concat_pl.{li}.weight"] = np.zeros((_HIDDEN, _MOD), np.float32)
        p_cc[f"concat_pl.{li}.bias"] = np.zeros(_HIDDEN, np.float32)
    rgb1, phi1 = _fwd(p_cc, feats, code)
    assert np.array_equal(rgb0, rgb1), "zero-init concat_pl must be an exact no-op"
    assert np.array_equal(phi0, phi1)


def test_numpy_forward_film_pl_nonzero_changes_output():
    p = _base_params()
    feats, code = _feats_code()
    _, phi0 = _fwd(p, feats, code)
    p_pl = dict(p)
    rng = np.random.default_rng(7)
    for li in range(_N_HIDDEN):
        p_pl[f"film_pl.{li}.weight"] = rng.standard_normal((2 * _HIDDEN, _MOD)).astype(np.float32)
        p_pl[f"film_pl.{li}.bias"] = rng.standard_normal(2 * _HIDDEN).astype(np.float32)
    _, phi1 = _fwd(p_pl, feats, code)
    assert not np.allclose(phi0, phi1), "nonzero film_pl must change the partition (route is live)"


def test_numpy_forward_concat_nonzero_changes_output():
    p = _base_params()
    feats, code = _feats_code()
    _, phi0 = _fwd(p, feats, code)
    p_cc = dict(p)
    rng = np.random.default_rng(9)
    for li in range(_N_HIDDEN):
        p_cc[f"concat_pl.{li}.weight"] = rng.standard_normal((_HIDDEN, _MOD)).astype(np.float32)
        p_cc[f"concat_pl.{li}.bias"] = rng.standard_normal(_HIDDEN).astype(np.float32)
    _, phi1 = _fwd(p_cc, feats, code)
    assert not np.allclose(phi0, phi1), "nonzero concat_pl must change the partition (route is live)"


# ===========================================================================
# 2. participation ratio + rank-floor penalty (LEVER-A loss math)
# ===========================================================================
def test_participation_ratio_rank1_collapse_is_about_one():
    # variation along ONE axis => rank-1 modulation => PR ~= 1 (the measured collapse).
    rng = np.random.default_rng(0)
    a = rng.standard_normal(64)                 # per-pair scalar
    v = rng.standard_normal(48)                 # one fixed direction
    M = np.outer(a, v)                          # (64, 48), rank-1 variation
    pr = film_modulation_participation_ratio(M)
    assert 0.99 <= pr <= 1.02, f"rank-1 modulation PR should be ~1, got {pr}"


def test_participation_ratio_full_rank_much_higher_than_collapse():
    rng = np.random.default_rng(1)
    M_full = rng.standard_normal((64, 48))      # iid => high effective rank
    a = rng.standard_normal(64)
    v = rng.standard_normal(48)
    M_collapse = np.outer(a, v)
    pr_full = film_modulation_participation_ratio(M_full)
    pr_collapse = film_modulation_participation_ratio(M_collapse)
    assert pr_full > 5.0, f"full-rank PR should be well above the rank-1 floor, got {pr_full}"
    assert pr_full > pr_collapse + 3.0


def test_rank_floor_penalty_penalizes_collapse_and_zero_above_target():
    rng = np.random.default_rng(2)
    a = rng.standard_normal(64)
    v = rng.standard_normal(48)
    M_collapse = np.outer(a, v)                 # PR ~ 1
    M_full = rng.standard_normal((64, 48))      # PR high
    target = 5.0
    pen_collapse = film_rank_floor_penalty(M_collapse, target)
    pen_full = film_rank_floor_penalty(M_full, target)
    assert pen_collapse > pen_full, "rank-1 collapse must be penalized MORE than a high-rank M"
    assert pen_collapse > 0.0
    # above-target M => zero penalty (soft floor)
    assert film_rank_floor_penalty(M_full, 1.5) == 0.0


def test_participation_ratio_rejects_bad_shape():
    with pytest.raises(ValueError):
        film_modulation_participation_ratio(np.zeros((3,)))


# ===========================================================================
# 3. thin-lane weight map (LEVER-B)
# ===========================================================================
def _toy_lane_grid():
    """A (40, 40) argmax grid: a 1-px-wide vertical lane line (THIN) at col 5 and a 10x10 wide lane
    block (WIDE). Class 1 = Lane; everything else class 0 = Road."""
    g = np.zeros((40, 40), np.int64)
    g[:, 5] = 1                  # thin 1-px vertical lane line
    g[25:35, 25:35] = 1          # wide 10x10 lane block
    return g


def test_lane_thin_weight_map_thin_gets_higher_than_wide():
    g = _toy_lane_grid()
    w = lane_thin_weight_map(g, lane_class=1, radius=4)
    thin_w = w[20, 5]                    # center of the thin line
    wide_w = w[30, 30]                   # center of the wide block
    assert thin_w > wide_w, f"thin line weight {thin_w} must exceed wide-block weight {wide_w}"
    assert thin_w > 0.5                  # thin -> low local density -> high weight
    assert wide_w < 0.2                  # wide -> high local density -> ~0 weight


def test_lane_thin_weight_map_nonzero_only_on_lane_pixels():
    g = _toy_lane_grid()
    w = lane_thin_weight_map(g, lane_class=1, radius=4)
    lane = (g == 1)
    assert np.all(w[~lane] == 0.0), "non-lane pixels must have zero weight"
    assert np.all(w[lane] >= 0.0)
    assert w[lane].max() > 0.0


def test_lane_thin_weight_map_deterministic():
    g = _toy_lane_grid()
    a = lane_thin_weight_map(g, lane_class=1, radius=4)
    b = lane_thin_weight_map(g, lane_class=1, radius=4)
    assert np.array_equal(a, b), "thin-lane map must be deterministic"


def test_lane_thin_weight_map_no_lane_is_all_zero():
    g = np.zeros((20, 20), np.int64)            # no class-1 pixels
    w = lane_thin_weight_map(g, lane_class=1, radius=3)
    assert np.all(w == 0.0)


def test_lane_thin_weight_map_rejects_non_2d():
    with pytest.raises(ValueError):
        lane_thin_weight_map(np.zeros((2, 3, 4), np.int64))


# ===========================================================================
# 4. MLX twins: identity-at-init + rank-floor gradient sign (force CPU; no GPU contention)
# ===========================================================================
def test_mlx_witness_film_per_layer_and_concat_identity_at_init():
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx.nn")
    mx.set_default_device(mx.cpu)
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parents[3]
    for pth in (repo / "experiments", repo / "src"):
        if str(pth) not in sys.path:
            sys.path.insert(0, str(pth))
    from train_levelset_witness_realized_through_R_mlx import build_levelset_rgb_witness

    model = build_levelset_rgb_witness(
        num_pairs=4, in_feat=_INFEAT, hidden_dim=_HIDDEN, n_hidden=_N_HIDDEN, mod_dim=_MOD,
        n_classes=_NCLS, activation="relu", softmax_temp=1.0, wire_w0=20.0, wire_s0=10.0,
        hosc_beta=4.0, hosc_omega=1.0, chroma=True, film_per_layer=True, film_concat_code=True)
    mx.eval(model.parameters())
    feats = mx.array(np.random.default_rng(3).standard_normal((_P, _INFEAT)).astype(np.float32))
    # both routes ACTIVE (zero-init) -> trunk output
    h_on = np.asarray(model._trunk(feats, 0))
    # toggle the routes OFF -> the branches are skipped
    model.film_per_layer = False
    model.film_concat_code = False
    h_off = np.asarray(model._trunk(feats, 0))
    assert np.array_equal(h_on, h_off), "zero-init film_pl/concat_pl must be an exact no-op at init"


def test_mlx_witness_film_per_layer_nonzero_changes_trunk():
    mx = pytest.importorskip("mlx.core")
    pytest.importorskip("mlx.nn")
    mx.set_default_device(mx.cpu)
    import sys
    from pathlib import Path
    repo = Path(__file__).resolve().parents[3]
    for pth in (repo / "experiments", repo / "src"):
        if str(pth) not in sys.path:
            sys.path.insert(0, str(pth))
    from train_levelset_witness_realized_through_R_mlx import build_levelset_rgb_witness

    model = build_levelset_rgb_witness(
        num_pairs=4, in_feat=_INFEAT, hidden_dim=_HIDDEN, n_hidden=_N_HIDDEN, mod_dim=_MOD,
        n_classes=_NCLS, activation="relu", softmax_temp=1.0, wire_w0=20.0, wire_s0=10.0,
        hosc_beta=4.0, hosc_omega=1.0, chroma=True, film_per_layer=True, film_concat_code=False)
    mx.eval(model.parameters())
    feats = mx.array(np.random.default_rng(4).standard_normal((_P, _INFEAT)).astype(np.float32))
    # give pair 0 a nonzero code so film_pl(code) != 0, and set film_pl[0] nonzero
    model.code[0] = mx.array(np.random.default_rng(5).standard_normal(_MOD).astype(np.float32))
    h_off = np.asarray(model._trunk(feats, 0)).copy()
    model.film_pl[0].weight = mx.array(np.random.default_rng(6).standard_normal((2 * _HIDDEN, _MOD)).astype(np.float32))
    mx.eval(model.parameters())
    h_on = np.asarray(model._trunk(feats, 0))
    assert not np.allclose(h_off, h_on), "a nonzero per-layer FiLM route must change the trunk"


def _mlx_pr(mx, M):
    Mc = M - mx.mean(M, axis=0, keepdims=True)
    tr = mx.sum(Mc * Mc)
    G = Mc @ Mc.T
    fro2 = mx.sum(G * G)
    return (tr * tr) / (fro2 + 1e-12)


def test_mlx_rank_floor_matches_numpy_reference():
    mx = pytest.importorskip("mlx.core")
    mx.set_default_device(mx.cpu)
    M_np = np.random.default_rng(11).standard_normal((40, 24)).astype(np.float32)
    pr_np = film_modulation_participation_ratio(M_np)
    pr_mx = float(_mlx_pr(mx, mx.array(M_np)))
    assert np.isclose(pr_np, pr_mx, rtol=1e-3, atol=1e-3), f"MLX PR {pr_mx} != numpy PR {pr_np}"


def test_mlx_rank_floor_gradient_increases_participation_ratio():
    # THE gradient-sign test: minimizing relu(target - PR(film(code))) must INCREASE PR (oppose the
    # rank-1 collapse). Mirrors the trainer's inline rank-floor math EXACTLY.
    mx = pytest.importorskip("mlx.core")
    mx.set_default_device(mx.cpu)
    rng = np.random.default_rng(12)
    # codes (S, mod) + a film weight (D, mod) initialised NEAR-collapse (PR ~1.03, faithful to the
    # MEASURED state PR~1.19 -- NOT exactly rank-1, which is the global PR minimum / a zero-gradient
    # saddle the rank-floor mathematically cannot escape by gradient alone).
    S, mod, D = 48, 4, 16
    codes = mx.array(rng.standard_normal((S, mod)).astype(np.float32))
    u = rng.standard_normal((D, 1)).astype(np.float32)
    vrow = rng.standard_normal((1, mod)).astype(np.float32)
    W0 = mx.array((u @ vrow + 0.06 * rng.standard_normal((D, mod))).astype(np.float32))  # near rank-1
    target = 6.0

    def penalty(W):
        M = codes @ W.T
        pr = _mlx_pr(mx, M)
        return mx.maximum(target - pr, 0.0)

    pr_before = float(_mlx_pr(mx, codes @ W0.T))
    grad_fn = mx.grad(penalty)
    W = W0
    lr = 0.5
    for _ in range(60):
        g = grad_fn(W)
        W = W - lr * g
        mx.eval(W)
    pr_after = float(_mlx_pr(mx, codes @ W.T))
    assert pr_after > pr_before + 0.25, (
        f"rank-floor gradient must INCREASE participation ratio (oppose collapse): "
        f"{pr_before} -> {pr_after}")
    # and the penalty actually decreased
    assert float(penalty(W)) < float(penalty(W0))
