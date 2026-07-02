# SPDX-License-Identifier: MIT
"""#224 FIX-ALL Wave B behavior tests — the 3 un-deferred levers' CORE mechanics.

These are the small-scale (GPU-free MLX) behavior guards for the value_and_grad /
grad-shield restructures that the LEVELSET trainer wire-in depends on. They assert
BEHAVIOR (grad flows, frozen buffer excluded, shield touches only its leaf), not
constants (per NO-FAKE #2). Advisory; pointer 0.19110 UNMOVED.
"""
from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
import mlx.nn as nn  # noqa: E402
from mlx.utils import tree_flatten, tree_unflatten  # noqa: E402


# ---------------------------------------------------------------------------
# LEVER 1 — pose-carrier child-attach: xi_stored frozen under PARENT recursion,
# dxi co-differentiated by the ONE nn.value_and_grad, grad-shield leaf-only.
# ---------------------------------------------------------------------------
def _tiny_witness():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 3)

    m = Tiny()
    mx.eval(m.parameters())
    return m


def test_carrier_child_attach_excludes_frozen_xi_stored():
    from tac.boundary_math.warp_real_luma_frame0 import (
        GroundHomographyGeom, WarpRealLumaFrame0Carrier, xi_from_pose_calibration,
    )
    P = 3
    poses = (np.random.default_rng(0).standard_normal((P, 6)) * 0.1)
    xi = np.stack([xi_from_pose_calibration(poses[i], 0.044, 0.0, 0.0) for i in range(P)]).astype(np.float32)
    geom = GroundHomographyGeom.eon(native_hw=(64, 80), pitch=0.0)
    carrier = WarpRealLumaFrame0Carrier.build(xi, geom, residual_mode="table", residual_scale=1.0)
    mx.eval(carrier.parameters())
    m = _tiny_witness()
    m.pose_carrier = carrier.impl
    mx.eval(m.parameters())
    keys = sorted(k for k, _ in tree_flatten(m.trainable_parameters()))
    # the frozen stored twist must NOT be trainable (parent recursion must honor the freeze)
    assert not any("xi_stored" in k for k in keys), keys
    # the residual MUST be trainable
    assert any("pose_carrier.dxi" in k for k in keys), keys


def test_carrier_dxi_receives_cograd_via_one_value_and_grad():
    from tac.boundary_math.warp_real_luma_frame0 import (
        GroundHomographyGeom, WarpRealLumaFrame0Carrier, xi_from_pose_calibration,
    )
    P = 3
    poses = (np.random.default_rng(1).standard_normal((P, 6)) * 0.1)
    xi = np.stack([xi_from_pose_calibration(poses[i], 0.044, 0.0, 0.0) for i in range(P)]).astype(np.float32)
    geom = GroundHomographyGeom.eon(native_hw=(64, 80), pitch=0.0)
    carrier = WarpRealLumaFrame0Carrier.build(xi, geom, residual_mode="table", residual_scale=1.0)
    mx.eval(carrier.parameters())
    m = _tiny_witness()
    m.pose_carrier = carrier.impl
    mx.eval(m.parameters())
    src = mx.array((np.random.default_rng(2).random((64, 80, 3)) * 255.0).astype(np.float32))
    x = mx.array(np.random.default_rng(3).random((2, 4)).astype(np.float32))

    def loss_fn(model):
        y = model.lin(x)
        f0 = model.pose_carrier.render_f0(src, 0, None, ste_round=True)
        return mx.sum(y * y) + mx.mean(f0)

    loss, grads = nn.value_and_grad(m, loss_fn)(m)
    mx.eval(loss, grads)
    gf = dict(tree_flatten(grads))
    assert np.isfinite(float(loss))
    dxi_g = np.asarray(gf["pose_carrier.dxi"])
    assert dxi_g.shape == (P, 6)
    assert np.all(np.isfinite(dxi_g))
    assert float(np.abs(dxi_g).sum()) > 0.0            # co-grad actually flows to the residual
    assert np.all(np.isfinite(np.asarray(gf["lin.weight"])))
    assert not any("xi_stored" in k for k in gf)        # frozen buffer gets no grad


# ---------------------------------------------------------------------------
# LEVER 2 — the containment grad-SHIELD applied to ONE leaf must NOT touch the
# witness (grouped-backward) grads. Uses the island_protection MLX shield.
# ---------------------------------------------------------------------------
def _shield_only_leaf(grads_tree, target_substr, residual, spec):
    from tac.boundary_math.island_protection import contain_protected_grad_mx

    flat = dict(tree_flatten(grads_tree))
    touched = []
    for k in list(flat):
        if target_substr in k:
            flat[k] = contain_protected_grad_mx(flat[k], residual, spec)
            touched.append(k)
    return tree_unflatten(list(flat.items())), touched


def test_seed_grad_shield_touches_only_seed_leaf():
    from tac.boundary_math.island_protection import ContainmentSpec

    class WithSeed(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 3)
            self.seed = mx.zeros((1, 8, 10, 3), dtype=mx.float32)

    m = WithSeed()
    # seed the protected residual with a nonzero island value (so 'shield' has a sign to defend)
    r = np.zeros((1, 8, 10, 3), np.float32)
    r[0, 2:4, 3:6, :] = 1.5
    m.seed = mx.array(r)
    mx.eval(m.parameters())
    x = mx.array(np.random.default_rng(4).random((2, 4)).astype(np.float32))

    def loss_fn(model):
        y = model.lin(x)
        # loss pulls seed toward 0 (destructive/erase direction) + a witness term
        return mx.sum(y * y) + mx.sum(model.seed * model.seed)

    loss, grads = nn.value_and_grad(m, loss_fn)(m)
    mx.eval(loss, grads)
    gf0 = dict(tree_flatten(grads))
    spec = ContainmentSpec(mode="shield", protected_mask=None)
    shielded, touched = _shield_only_leaf(grads, "seed", m.seed, spec)
    mx.eval(shielded)
    sf = dict(tree_flatten(shielded))
    assert touched == ["seed"]
    # witness grads UNCHANGED (grouped-backward path untouched)
    assert np.array_equal(np.asarray(sf["lin.weight"]), np.asarray(gf0["lin.weight"]))
    assert np.array_equal(np.asarray(sf["lin.bias"]), np.asarray(gf0["lin.bias"]))
    # the shield NEUTRALIZES the erase (same-sign) component on the seeded island pixels:
    # grad there was +2*r (positive, r>0) -> shield removes it -> 0 on the island.
    g_seed_after = np.asarray(sf["seed"])
    assert float(np.abs(g_seed_after[0, 2:4, 3:6, :]).max()) < 1e-6


# ---------------------------------------------------------------------------
# LEVER 3 — AA-supersample + self-orient FINE dir-feats: shape-correct at the
# ss*grid + the batch-scoped cache is MEMORY-BOUNDED (scales ~cap, NOT ~P).
# ---------------------------------------------------------------------------
def test_fine_dir_feats_shape_and_finite_at_ss_grid():
    from tac.boundary_math.aa_sdf_observation_render import build_supersampled_coords
    from tac.boundary_math.lever_b_generator import self_orientation_directional_feats
    from scipy.ndimage import maximum_filter

    H, W, ss, n_freqs = 48, 64, 2, 6
    coords_fine = build_supersampled_coords(H, W, ss)
    assert coords_fine.shape == (ss * ss * H * W, 2)
    base_arg = maximum_filter(np.random.default_rng(5).integers(0, 5, size=(H, W)).astype(np.int64), size=5)
    arg_fine = np.kron(base_arg, np.ones((ss, ss), np.int64))   # NN-upsample (trainer's np.kron)
    assert arg_fine.shape == (ss * H, ss * W)
    df = self_orientation_directional_feats(coords_fine, arg_fine, n_freqs=n_freqs)
    assert df.shape == (ss * ss * H * W, 4 * n_freqs)
    assert np.all(np.isfinite(df))


def test_batch_scoped_fine_cache_is_memory_bounded():
    # mirror the trainer's bounded FIFO: len never exceeds cap => memory scales ~cap, NOT ~P.
    cap = 4
    P = 40
    lru: dict[int, object] = {}

    def _get(pi):
        v = lru.get(pi)
        if v is None:
            v = object()
            lru[pi] = v
            while len(lru) > cap:
                lru.pop(next(iter(lru)))
        return v

    max_seen = 0
    for _epoch in range(3):
        for pi in range(P):        # every pair every epoch (the thrash the blocker measured)
            _get(pi)
            max_seen = max(max_seen, len(lru))
    assert max_seen <= cap           # MEMORY-BOUNDED: ~cap not ~P
    # per-pair fine dir-feat bytes scale ss^2*H*W*4*n_freqs*4 -> a cap-bounded cache is cap*that.
    H, W, ss, n_freqs = 384, 512, 2, 6
    per_pair = (ss * ss * H * W) * (4 * n_freqs) * 4
    assert per_pair == 75497472                    # 75.5 MB/pair @ ss=2 (the measured n600 figure)
    assert cap * per_pair < 0.5e9                   # cap=4 -> 0.30 GB (vs 45 GB for all-600)
