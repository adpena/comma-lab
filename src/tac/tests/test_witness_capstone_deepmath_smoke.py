# SPDX-License-Identifier: MIT
"""Behavior tests for the WITNESS CAPSTONE deep-math smoke feature builders.

NO-FAKE discipline (CLAUDE.md class 2): these assert BEHAVIOR, not constants. A stub that
returns a fixed array would FAIL — the boundary mask must actually mark inter-class edges, the
proximity must actually peak on the boundary, the oriented PE must actually depend on the tangent,
and the numpy forward must match the MLX forward in ARGMAX.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
_MOD_PATH = REPO_ROOT / "tools" / "witness_capstone_deepmath_smoke.py"
for p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

_spec = importlib.util.spec_from_file_location("witness_capstone_deepmath_smoke", _MOD_PATH)
assert _spec and _spec.loader
wc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(wc)


def test_all_class_boundary_mask_marks_inter_class_edges() -> None:
    # Two horizontal bands: top class 0, bottom class 2. Boundary is the 2 rows straddling row 2/3.
    a = np.zeros((6, 6), dtype=np.int32)
    a[3:, :] = 2
    bnd = wc._all_class_boundary_mask(a)
    # rows 2 and 3 (the pixels straddling the edge) must be marked; interior rows must not.
    assert bnd[2].all() and bnd[3].all()
    assert not bnd[0].any() and not bnd[5].any()


def test_all_class_boundary_mask_empty_when_uniform() -> None:
    a = np.full((5, 5), 3, dtype=np.int32)
    assert not wc._all_class_boundary_mask(a).any()


def test_boundary_proximity_peaks_on_boundary_all_class() -> None:
    a = np.zeros((10, 10), dtype=np.int32)
    a[5:, :] = 4  # boundary between row 4/5
    prox, tang = wc.boundary_proximity_and_tangent(a, lane_class=1, tau=2.0, all_class=True)
    assert prox.shape == (10, 10)
    # proximity at the boundary band must exceed the interior corners (decays with distance).
    assert prox[4, 5] > prox[0, 0]
    assert prox[5, 5] > prox[9, 9]
    # proximity is a valid [0,1] key.
    assert prox.min() >= 0.0 and prox.max() <= 1.0 + 1e-5
    # tangent is unit length (approx) everywhere it is defined.
    norms = np.sqrt((tang**2).sum(-1))
    assert np.allclose(norms, 1.0, atol=1e-4)


def test_boundary_proximity_lane_only_vs_all_class_differ() -> None:
    # A frame with a class-0/class-2 edge but NO class-1 anywhere: all_class must mark a boundary,
    # lane-only must NOT (no lane present -> flat far-away proximity).
    a = np.zeros((12, 12), dtype=np.int32)
    a[6:, :] = 2
    prox_all, _ = wc.boundary_proximity_and_tangent(a, lane_class=1, tau=2.0, all_class=True)
    prox_lane, _ = wc.boundary_proximity_and_tangent(a, lane_class=1, tau=2.0, all_class=False)
    # all-class sees the 0/2 edge -> high proximity there; lane-only sees no lane -> ~0 everywhere.
    assert prox_all[5, 5] > 0.3
    assert prox_lane.max() < 0.05


def test_directional_fourier_feats_depends_on_tangent() -> None:
    coords = np.array([[0.3, -0.2], [0.5, 0.1]], dtype=np.float32)
    t1 = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    t2 = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
    f1 = wc.directional_fourier_feats(coords, t1, n_freqs=4, freq_across=16.0, freq_along=2.0)
    f2 = wc.directional_fourier_feats(coords, t2, n_freqs=4, freq_across=16.0, freq_along=2.0)
    assert f1.shape == (2, 16)  # 4 freqs * 4 (sin/cos x along/across)
    # rotating the tangent must change the encoding (it is genuinely oriented, not a no-op).
    assert not np.allclose(f1, f2)


def test_numpy_argmax_parity_with_mlx_forward() -> None:
    # Build a tiny generator, push the SAME inputs through MLX and a numpy mirror, assert argmax
    # parity (the score-native quantity). This is the portability contract a stub cannot fake.
    import mlx.core as mx

    rng = np.random.default_rng(0)
    P = 3
    model = wc.ImprovedSegGenerator(
        num_pairs=P, n_fourier=8, hidden_dim=24, n_hidden=2, mod_dim=8,
        fourier_sigma=4.0, use_prox=False, use_dir=False, n_dir_freqs=4, activation="relu",
    )
    # random but realistic mod codes so the net is non-degenerate.
    model.mod = mx.array(rng.standard_normal((P, 8)).astype(np.float32))
    in_feat = model.in_feat
    feats = rng.standard_normal((50, in_feat)).astype(np.float32)
    pi = 1
    mlx_logits = np.array(model(mx.array(feats), pi))
    # numpy mirror using the same flattened params.
    from mlx.utils import tree_flatten

    params = {k: np.array(v) for k, v in dict(tree_flatten(model.parameters())).items()}
    h = np.maximum(feats @ params["in_proj.weight"].T + params["in_proj.bias"], 0.0)
    film = params["mod"][pi] @ params["film.weight"].T + params["film.bias"]
    film = film.reshape(model.n_hidden, 2, model.hidden_dim)
    for li in range(model.n_hidden):
        scale = 1.0 + film[li, 0]
        shift = film[li, 1]
        h = np.maximum((h @ params[f"hidden.{li}.weight"].T + params[f"hidden.{li}.bias"]) * scale + shift, 0.0)
    np_logits = h @ params["out.weight"].T + params["out.bias"]
    assert (mlx_logits.argmax(-1) == np_logits.argmax(-1)).mean() == 1.0


def test_gauss_activation_runs_and_differs_from_relu() -> None:
    import mlx.core as mx

    rng = np.random.default_rng(1)
    feats = rng.standard_normal((20, 16)).astype(np.float32)  # 2*8 iso
    m_relu = wc.ImprovedSegGenerator(2, 8, 16, 2, 4, 4.0, False, False, 4, "relu")
    m_gauss = wc.ImprovedSegGenerator(2, 8, 16, 2, 4, 4.0, False, False, 4, "gauss")
    # copy weights so only the activation differs.
    from mlx.utils import tree_flatten, tree_unflatten

    flat = tree_flatten(m_relu.parameters())
    m_gauss.update(tree_unflatten(flat))
    lr = np.array(m_relu(mx.array(feats), 0))
    lg = np.array(m_gauss(mx.array(feats), 0))
    assert not np.allclose(lr, lg)  # the activation genuinely changes the forward.


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
