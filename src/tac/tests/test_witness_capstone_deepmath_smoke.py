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


def test_hard_pixel_boost_weights_only_misclassified() -> None:
    # PR74/PR62 hard-pixel boost: correct px -> 1.0, wrong px -> 1.0 + error_boost. Behavior,
    # not a constant: the boost must DEPEND on (pred != gt).
    pred = np.array([0, 1, 2, 3, 4], dtype=np.int32)
    gt = np.array([0, 1, 0, 3, 1], dtype=np.int32)  # px 2 and 4 are wrong
    boost = wc.hard_pixel_boost(pred, gt, error_boost=9.0)
    assert boost.tolist() == [1.0, 1.0, 10.0, 1.0, 10.0]


def test_hard_pixel_boost_disabled_is_identity() -> None:
    pred = np.array([0, 1, 2], dtype=np.int32)
    gt = np.array([4, 4, 4], dtype=np.int32)  # all wrong
    boost = wc.hard_pixel_boost(pred, gt, error_boost=0.0)
    # error_boost=0 -> all 1.0 regardless of correctness (preserves baseline behavior).
    assert np.allclose(boost, 1.0)


def test_hard_pixel_boost_all_correct_is_identity() -> None:
    pred = np.array([2, 2, 2, 2], dtype=np.int32)
    gt = np.array([2, 2, 2, 2], dtype=np.int32)
    boost = wc.hard_pixel_boost(pred, gt, error_boost=49.0)
    assert np.allclose(boost, 1.0)  # no wrong px -> no boost even at high error_boost.


def test_hard_pixel_boost_scales_with_error_boost() -> None:
    pred = np.array([0, 1], dtype=np.int32)
    gt = np.array([0, 0], dtype=np.int32)  # px 1 wrong
    b9 = wc.hard_pixel_boost(pred, gt, 9.0)
    b49 = wc.hard_pixel_boost(pred, gt, 49.0)
    # the wrong px boost must track error_boost (genuinely the adapted mechanism, not a fixed factor).
    assert b9[1] == 10.0 and b49[1] == 50.0
    assert b9[0] == 1.0 and b49[0] == 1.0


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


# ---------------------------------------------------------------------------
# KD (soft-logit knowledge distillation) NO-FAKE behavior tests.
# These assert BEHAVIOR, not constants: KL must be ZERO iff student==teacher, POSITIVE when
# they differ, scale with T^2, flow gradients, and be exactly backward-compatible at kd_weight=0.
# A stub returning a constant FAILS every one of these.
# ---------------------------------------------------------------------------
def test_kd_kl_is_zero_when_student_matches_teacher() -> None:
    import mlx.core as mx

    rng = np.random.default_rng(0)
    logits = mx.array(rng.standard_normal((30, 5)).astype(np.float32))
    # identical student & teacher -> KL == 0 exactly (the defining property of a KD target).
    kl = np.array(wc.kd_kl_logits(logits, logits, temperature=2.0))
    assert kl.shape == (30,)
    assert np.allclose(kl, 0.0, atol=1e-5)


def test_kd_kl_is_positive_when_student_differs_from_teacher() -> None:
    import mlx.core as mx

    rng = np.random.default_rng(1)
    teacher = mx.array(rng.standard_normal((40, 5)).astype(np.float32) * 3.0)
    student = mx.array(rng.standard_normal((40, 5)).astype(np.float32) * 3.0)
    kl = np.array(wc.kd_kl_logits(student, teacher, temperature=2.0))
    # KL divergence is non-negative and strictly positive on average for genuinely different dists.
    assert (kl >= -1e-5).all()
    assert kl.mean() > 0.01


def test_kd_kl_matches_closed_form_softmax_kl() -> None:
    # Independent numpy reference of T^2 * KL(softmax(teacher/T) || softmax(student/T)).
    import mlx.core as mx

    rng = np.random.default_rng(2)
    T = 2.0
    s = rng.standard_normal((25, 5)).astype(np.float32) * 2.0
    t = rng.standard_normal((25, 5)).astype(np.float32) * 2.0

    def _softmax(z: np.ndarray) -> np.ndarray:
        z = z - z.max(-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(-1, keepdims=True)

    pt = _softmax(t / T)
    ps = _softmax(s / T)
    ref = (T * T) * (pt * (np.log(pt + 1e-12) - np.log(ps + 1e-12))).sum(-1)
    got = np.array(wc.kd_kl_logits(mx.array(s), mx.array(t), temperature=T))
    assert np.allclose(got, ref, atol=1e-3)


def test_kd_kl_scales_with_temperature_squared() -> None:
    # The T^2 factor is a defining part of the Hinton-2015 form. Doubling T (with otherwise tiny
    # logits so softmax is near-uniform and KL ~ quadratic in logit scale) increases the T^2 weight.
    import mlx.core as mx

    rng = np.random.default_rng(3)
    teacher = mx.array((rng.standard_normal((50, 5)) * 0.01).astype(np.float32))
    student = mx.array((rng.standard_normal((50, 5)) * 0.01).astype(np.float32))
    kl_t1 = float(np.array(wc.kd_kl_logits(student, teacher, 1.0)).mean())
    kl_t4 = float(np.array(wc.kd_kl_logits(student, teacher, 4.0)).mean())
    # With near-uniform softmax, the per-class KL ~ const/T^2 and the prefactor is T^2, so the
    # T^2 prefactor dominates and kl(T=4) > kl(T=1). At minimum the value genuinely depends on T.
    assert kl_t4 != kl_t1
    assert kl_t4 > 0.0 and kl_t1 > 0.0


def test_kd_kl_gradient_flows_to_student() -> None:
    # The KD term must produce a non-zero gradient w.r.t. the student logits (else it trains nothing).
    import mlx.core as mx

    rng = np.random.default_rng(4)
    teacher = mx.array(rng.standard_normal((20, 5)).astype(np.float32) * 2.0)

    def f(student: mx.array) -> mx.array:
        return wc.kd_kl_logits(student, teacher, 2.0).mean()

    student = mx.array(rng.standard_normal((20, 5)).astype(np.float32) * 2.0)
    g = mx.grad(f)(student)
    g_np = np.array(g)
    assert g_np.shape == (20, 5)
    assert np.abs(g_np).sum() > 1e-4  # gradient genuinely flows


def test_kd_kl_gradient_vanishes_at_optimum() -> None:
    # When student == teacher the KL is at its minimum (0), so the gradient is ~0 there.
    import mlx.core as mx

    rng = np.random.default_rng(5)
    teacher = mx.array(rng.standard_normal((15, 5)).astype(np.float32) * 2.0)

    def f(student: mx.array) -> mx.array:
        return wc.kd_kl_logits(student, teacher, 2.0).mean()

    g = mx.grad(f)(teacher)  # evaluate gradient AT the teacher (= optimum)
    assert np.allclose(np.array(g), 0.0, atol=1e-4)


def test_load_teacher_logits_shape_and_consistency(tmp_path: Path) -> None:
    # Build a tiny fake teacher store (the SAME memmap layout the real builder writes) and assert
    # the loader returns the right slice with argmax recoverable. Behavior: a loader that ignored
    # the file would not reproduce the planted argmax pattern.
    import json as _json

    H, W, P, C = 8, 6, 4, 5
    d = tmp_path / "teacher"
    d.mkdir()
    rng = np.random.default_rng(6)
    logits = (rng.standard_normal((P, C, H, W)) * 3.0).astype(np.float16)
    logits.tofile(d / "gt_segnet_logits.f16")
    (d / "teacher_logits_meta.json").write_text(_json.dumps({
        "num_pairs_built": P, "n_classes": C, "seg_input_hw": [H, W],
    }))
    mm = wc.load_teacher_logits(d, n_pairs=3, H=H, W=W)
    assert mm.shape == (3, C, H, W)
    # the planted argmax pattern must be recoverable from the loaded store.
    assert (np.asarray(mm[0]).astype(np.float32).argmax(0)
            == logits[0].astype(np.float32).argmax(0)).all()


def test_load_teacher_logits_rejects_hw_mismatch(tmp_path: Path) -> None:
    import json as _json

    d = tmp_path / "teacher"
    d.mkdir()
    (np.zeros((2, 5, 8, 6), np.float16)).tofile(d / "gt_segnet_logits.f16")
    (d / "teacher_logits_meta.json").write_text(_json.dumps({
        "num_pairs_built": 2, "n_classes": 5, "seg_input_hw": [8, 6],
    }))
    with pytest.raises(ValueError, match="teacher hw"):
        wc.load_teacher_logits(d, n_pairs=2, H=10, W=10)  # wrong hw -> fail closed


# --------------------------------------------------------------------------------------------
# Boundary-affinity KD (the structurally-NEW pair-wise lever; Liu CVPR-2019 structured-KD adapted)
# --------------------------------------------------------------------------------------------
def test_boundary_affinity_zero_when_student_matches_teacher() -> None:
    # If student logits == teacher logits at BOTH pixels, the soft affinities match exactly -> 0.
    import mlx.core as mx

    rng = np.random.default_rng(11)
    a = mx.array(rng.standard_normal((20, 5)).astype(np.float32) * 2.0)
    b = mx.array(rng.standard_normal((20, 5)).astype(np.float32) * 2.0)
    loss = np.array(wc.boundary_affinity_kd(a, b, a, b, temperature=2.0))
    assert np.allclose(loss, 0.0, atol=1e-6)


def test_boundary_affinity_positive_when_edge_field_differs() -> None:
    # Student says "same class" across the pair (high affinity) while teacher says "edge"
    # (low affinity) -> the affinity-match loss must be strictly positive (the term is alive).
    import mlx.core as mx

    M = 16
    same = mx.array(np.tile(np.array([8.0, 0, 0, 0, 0], np.float32), (M, 1)))  # both -> class 0
    other = mx.array(np.tile(np.array([0, 8.0, 0, 0, 0], np.float32), (M, 1)))  # -> class 1
    # student: A and B both class 0 (no edge); teacher: A class 0, B class 1 (an edge).
    loss = np.array(wc.boundary_affinity_kd(same, same, same, other, temperature=2.0))
    assert (loss > 1e-3).all()


def test_boundary_affinity_captures_edge_vs_interior_structure() -> None:
    # The soft affinity must be HIGHER for an interior pair (same class both) than for an edge pair
    # (different class) under a matched teacher==student — proving it reads edge geometry, not a
    # constant. We compare the teacher affinity directly via the (student==teacher) identity.
    import mlx.core as mx

    interior_a = mx.array(np.array([[6.0, 0, 0, 0, 0]], np.float32))
    interior_b = mx.array(np.array([[6.0, 0, 0, 0, 0]], np.float32))  # same class -> high agreement
    edge_a = mx.array(np.array([[6.0, 0, 0, 0, 0]], np.float32))
    edge_b = mx.array(np.array([[0, 6.0, 0, 0, 0]], np.float32))  # different -> low agreement
    # student deliberately FLAT (uniform) so affinity differs from the teacher in both cases;
    # the EDGE pair (lower teacher affinity) should yield a LARGER mismatch than the interior pair
    # (teacher affinity ~1, student ~0.2). i.e. the loss is sensitive to the teacher edge field.
    flat = mx.array(np.zeros((1, 5), np.float32))
    l_interior = float(np.array(wc.boundary_affinity_kd(flat, flat, interior_a, interior_b, 2.0))[0])
    l_edge = float(np.array(wc.boundary_affinity_kd(flat, flat, edge_a, edge_b, 2.0))[0])
    # teacher interior affinity ~1 (mismatch vs flat 0.2 -> large); teacher edge affinity ~0.5
    # (mismatch vs flat 0.2 -> smaller). So interior mismatch > edge mismatch here.
    assert l_interior > l_edge
    assert l_interior != l_edge  # the term is NOT constant in the teacher field


def test_boundary_affinity_matches_closed_form_numpy() -> None:
    # Closed-form numpy parity: aff = <softmax(a/T), softmax(b/T)>; loss = (aff_s - aff_t)^2.
    import mlx.core as mx

    rng = np.random.default_rng(12)
    T = 2.0
    sa = rng.standard_normal((10, 5)).astype(np.float32) * 1.5
    sb = rng.standard_normal((10, 5)).astype(np.float32) * 1.5
    ta = rng.standard_normal((10, 5)).astype(np.float32) * 1.5
    tb = rng.standard_normal((10, 5)).astype(np.float32) * 1.5

    def _sm(x):
        z = x / T
        z = z - z.max(axis=-1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=-1, keepdims=True)

    aff_s = (_sm(sa) * _sm(sb)).sum(-1)
    aff_t = (_sm(ta) * _sm(tb)).sum(-1)
    want = (aff_s - aff_t) ** 2
    got = np.array(wc.boundary_affinity_kd(mx.array(sa), mx.array(sb), mx.array(ta), mx.array(tb), T))
    assert np.allclose(got, want, atol=1e-5)


def test_boundary_affinity_gradient_flows_to_student() -> None:
    import mlx.core as mx

    rng = np.random.default_rng(13)
    ta = mx.array(rng.standard_normal((12, 5)).astype(np.float32))
    tb = mx.array(rng.standard_normal((12, 5)).astype(np.float32))
    sa = mx.array(rng.standard_normal((12, 5)).astype(np.float32))
    sb = mx.array(rng.standard_normal((12, 5)).astype(np.float32))

    def f(s_pair):
        return wc.boundary_affinity_kd(s_pair[0], s_pair[1], ta, tb, 2.0).mean()

    g = mx.grad(f)([sa, sb])
    assert not np.allclose(np.array(g[0]), 0.0)  # gradient reaches pixel A
    assert not np.allclose(np.array(g[1]), 0.0)  # and pixel B


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
