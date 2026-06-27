"""Tests for the STRUCTURED-PRIOR phi INIT primitives (FEED-ef).

NO-FAKE: synthetic-but-faithful fixtures (canonical comma10k spatial layout — static top sky,
static bottom hood, large mid road, a thin lane stripe, a moving movable blob) exercise the REAL
math the functions name: data-driven role self-detection, scipy-EDT signed distance, the
argmax==partition round-trip, and the closed-form ridge readout. The decisive realized-through-R
verdict lives in the measure script + the FEED-ef memo; these guard the primitives + their
determinism. The functions never hardcode a class index by luma (the FEED-dn mislabel guard).
"""

from __future__ import annotations

import numpy as np

from tac.boundary_math.lever_b_levelset_generator import (
    build_static_core_partition,
    build_static_core_phi_target,
    fit_out_sdf_to_structured_target,
    rebuild_per_pair_feats_in_place,
    signed_distance_fields,
    structured_init_fit_disagree,
)


def _synthetic_lstars(n=16, h=64, w=80, seed=0):
    """Canonical comma10k layout: 2=sky(top static), 4=hood(bottom static, jittery edge),
    0=road(mid background), 1=lane(thin static stripe), 3=movable(moving blob, > lane area)."""
    rng = np.random.default_rng(seed)
    L = np.zeros((n, h, w), np.int64)
    for i in range(n):
        L[i, :, :] = 0  # road
        L[i, : int(h * 0.30), :] = 2  # sky (static top)
        edge = int(h * 0.75) + int(rng.integers(-2, 3))
        L[i, edge:, :] = 4  # hood (static bottom, slight jitter)
        # thin lane stripe (class 1) in the road band — small area, static
        cl = w // 2
        L[i, int(h * 0.35) : int(h * 0.72), cl : cl + 1] = 1
        # a moving movable blob (class 3), larger than the lane stripe
        r = int(rng.integers(int(h * 0.38), int(h * 0.55)))
        c = int(rng.integers(5, w - 12))
        L[i, r : r + 4, c : c + 8] = 3
    return L


# --------------------------- role self-detection ---------------------------
def test_partition_self_detects_canonical_roles():
    L = _synthetic_lstars()
    _part, roles, _meta = build_static_core_partition(L, n_classes=5, include_lane=True)
    d = roles.as_dict()
    assert d["sky"] == 2 and d["hood"] == 4 and d["road"] == 0
    assert d["lane"] == 1 and d["movable"] == 3  # lane thinner than movable -> tiebreak


def test_partition_excludes_movable_and_includes_static_core():
    L = _synthetic_lstars()
    part, roles, meta = build_static_core_partition(L, n_classes=5, include_lane=True)
    present = set(np.unique(part).tolist())
    assert roles.movable not in present  # movable has no static structure -> learned residual
    assert {roles.road, roles.sky, roles.hood} <= present
    assert meta["part_frac"][roles.movable] == 0.0


def test_partition_no_lane_option():
    L = _synthetic_lstars()
    part, roles, _meta = build_static_core_partition(L, n_classes=5, include_lane=False)
    assert roles.lane not in set(np.unique(part).tolist())


# --------------------------- SDF target round-trip ---------------------------
def test_phi_target_argmax_equals_partition():
    L = _synthetic_lstars()
    phi, _roles, _meta = build_static_core_phi_target(L, n_classes=5, include_lane=True)
    part, _, _ = build_static_core_partition(L, n_classes=5, include_lane=True)
    assert phi.shape == (64, 80, 5)
    assert np.array_equal(phi.argmax(-1).astype(part.dtype), part)  # the NO-FAKE round-trip


def test_phi_target_absent_class_is_all_negative():
    L = _synthetic_lstars()
    phi, roles, _meta = build_static_core_phi_target(L, n_classes=5, include_lane=True)
    assert float(phi[..., roles.movable].max()) < 0.0  # movable never wins


def test_signed_distance_fields_is_one_lipschitz_ish():
    # an EDT-based SDF stack: inside class k phi_k>0, outside <0 (the deep locally-supported field)
    part = np.zeros((40, 40), np.int64)
    part[:10] = 2
    part[30:] = 4
    phi = signed_distance_fields(part, 5)
    assert np.array_equal(phi.argmax(-1), part)
    assert phi[..., 2][:10].min() > 0 and phi[..., 0][15:25].min() > 0


# --------------------------- determinism ---------------------------
def test_phi_target_deterministic():
    L = _synthetic_lstars(seed=7)
    a, _, _ = build_static_core_phi_target(L, n_classes=5, include_lane=True)
    b, _, _ = build_static_core_phi_target(L, n_classes=5, include_lane=True)
    assert np.array_equal(a, b)


# --------------------------- closed-form ridge readout ---------------------------
def test_fit_out_sdf_shapes_and_determinism():
    rng = np.random.default_rng(0)
    h = rng.standard_normal((500, 32)).astype(np.float32)
    y = rng.standard_normal((500, 5)).astype(np.float32)
    w1, b1 = fit_out_sdf_to_structured_target(h, y, ridge=1e-2)
    w2, b2 = fit_out_sdf_to_structured_target(h, y, ridge=1e-2)
    assert w1.shape == (5, 32) and b1.shape == (5,)
    assert np.array_equal(w1, w2) and np.array_equal(b1, b2)  # closed-form -> deterministic


def test_fit_reproduces_when_basis_is_representable():
    # when the "trunk features" ARE a one-hot of the target argmax, a linear readout is exact.
    part = np.array([0, 1, 2, 3, 4, 0, 2, 4] * 50, np.int64)
    phi = (np.eye(5, dtype=np.float32)[part] * 10.0 - 5.0)
    h = np.eye(5, dtype=np.float32)[part]  # representable basis
    w, b = fit_out_sdf_to_structured_target(h, phi, ridge=1e-6)
    assert structured_init_fit_disagree(h, phi, w, b) < 1e-6


def test_fit_disagree_is_a_fraction():
    rng = np.random.default_rng(1)
    h = rng.standard_normal((300, 16)).astype(np.float32)
    phi = rng.standard_normal((300, 5)).astype(np.float32)
    w, b = fit_out_sdf_to_structured_target(h, phi, ridge=1e-2)
    d = structured_init_fit_disagree(h, phi, w, b)
    assert 0.0 <= d <= 1.0


# --------------------------- memory-bounded reorient rebuild (FEED-eh) ---------------------------
def _fake_feats(seed):
    return lambda pi: np.random.default_rng(seed + pi).standard_normal((12, 4)).astype(np.float32)


def test_rebuild_in_place_bit_identical_to_naive():
    n = 7
    fp = _fake_feats(0)
    ref = [np.array(fp(pi), copy=True) for pi in range(n)]  # the unbounded reference
    cache = rebuild_per_pair_feats_in_place(
        None, n, fp, mx_array=lambda a: np.array(a, copy=True), mx_eval=lambda a: None)
    assert len(cache) == n
    for r, c in zip(ref, cache):
        assert np.array_equal(r, c)
    # in-place rebuild over an EXISTING cache (the reorient path) -> same values
    cache2 = rebuild_per_pair_feats_in_place(
        cache, n, fp, mx_array=lambda a: np.array(a, copy=True), mx_eval=lambda a: None)
    for r, c in zip(ref, cache2):
        assert np.array_equal(r, c)


def test_rebuild_frees_old_slot_before_alloc_and_evals_each():
    n = 5
    fp = lambda pi: np.full((3, 2), pi, np.float32)
    cache = rebuild_per_pair_feats_in_place(None, n, fp, mx_array=lambda a: np.array(a, copy=True), mx_eval=lambda a: None)
    slot_was_none, evals = [], []

    def probe_array(a):  # the slot for this pi must have been cleared to None before we allocate
        slot_was_none.append(cache[int(a[0, 0])] is None)
        return np.array(a, copy=True)

    rebuild_per_pair_feats_in_place(cache, n, fp, mx_array=probe_array, mx_eval=lambda x: evals.append(int(x[0, 0])))
    assert all(slot_was_none)            # OLD entry freed BEFORE the NEW alloc (peak ~= one cache)
    assert evals == list(range(n))       # mx_eval called per entry (lazy graph bounded), in order


def test_rebuild_reallocates_on_length_mismatch():
    cache = rebuild_per_pair_feats_in_place([1, 2], 4, lambda pi: np.zeros((2, 2), np.float32),
                                            mx_array=lambda a: np.array(a, copy=True), mx_eval=lambda a: None)
    assert len(cache) == 4
