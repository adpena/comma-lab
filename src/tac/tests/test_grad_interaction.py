"""Tests for per-term gradient interaction telemetry (task #312 Phase A).

Cosine/shares/conflict math is proven on hand-built vectors; the RNG-stream non-perturbation
guard is proven with a bit-compare (draw-restore-draw identity) — the binding score-neutrality
invariant.
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.witness_control import grad_interaction as gi


def test_cosine_matrix_orthogonal_parallel_antiparallel():
    vecs = {
        "a": np.array([1.0, 0.0, 0.0]),
        "b": np.array([0.0, 1.0, 0.0]),   # orthogonal to a
        "c": np.array([2.0, 0.0, 0.0]),   # parallel to a
        "d": np.array([-1.0, 0.0, 0.0]),  # anti-parallel to a
    }
    names, mat, norms = gi.cosine_matrix(vecs)
    idx = {n: i for i, n in enumerate(names)}
    assert mat[idx["a"], idx["b"]] == pytest.approx(0.0, abs=1e-9)
    assert mat[idx["a"], idx["c"]] == pytest.approx(1.0, abs=1e-9)
    assert mat[idx["a"], idx["d"]] == pytest.approx(-1.0, abs=1e-9)
    assert mat[idx["a"], idx["a"]] == pytest.approx(1.0)
    # symmetric
    assert np.allclose(mat, mat.T)
    assert norms["c"] == pytest.approx(2.0)


def test_cosine_matrix_zero_gradient_term():
    names, mat, norms = gi.cosine_matrix({"z": np.zeros(4), "a": np.ones(4)})
    zi = names.index("z")
    assert norms["z"] == 0.0
    assert mat[zi, zi] == 0.0  # zero-grad term: self-cos defined as 0
    assert mat[zi, names.index("a")] == 0.0


def test_cosine_matrix_length_mismatch_raises():
    with pytest.raises(ValueError):
        gi.cosine_matrix({"a": np.ones(3), "b": np.ones(4)})


def test_conflict_pairs_threshold_and_ordering():
    vecs = {
        "seg": np.array([1.0, 0.0]),
        "pose": np.array([-1.0, 0.0]),    # cos -1 vs seg (strong conflict)
        "eik": np.array([0.0, 1.0]),      # cos 0 vs seg (not a conflict)
        "len": np.array([-0.3, 0.95]),    # mild negative vs seg
    }
    names, mat, _ = gi.cosine_matrix(vecs)
    confs = gi.conflict_pairs(names, mat, threshold=-0.2)
    flat = {tuple(sorted(c["pair"])): c["cos"] for c in confs}
    assert ("pose", "seg") in flat
    assert flat[("pose", "seg")] == pytest.approx(-1.0)
    assert ("eik", "seg") not in flat  # cos 0 is not < -0.2
    # most-negative first
    assert confs[0]["cos"] <= confs[-1]["cos"]


def test_dominance_shares_sum_to_one():
    norms = {"seg": 30.0, "pose": 10.0, "eik": 0.0}
    sh = gi.dominance_shares(norms)
    assert sh["seg"] == pytest.approx(0.75)
    assert sh["pose"] == pytest.approx(0.25)
    assert sh["eik"] == 0.0
    assert sum(sh.values()) == pytest.approx(1.0)
    assert gi.dominance_shares({"a": 0.0, "b": 0.0}) == {"a": 0.0, "b": 0.0}


def test_grad_interaction_row_schema_and_active_filter():
    vecs = {
        "seg": np.array([3.0, 0.0]),
        "pose": np.array([0.0, 1.0]),
        "eikonal": np.zeros(2),   # inactive -> dropped from the matrix
    }
    row = gi.grad_interaction_row(vecs, stage="tau_softplus", ep=350, k_pairs=32,
                                  cadence="boundary")
    assert row["stage"] == "grad_interactions"
    assert row["ep"] == 350
    assert row["seg_stage"] == "tau_softplus"
    assert row["k_pairs"] == 32
    assert "eikonal" not in row["terms"]  # inactive lever dropped
    assert set(row["terms"]) == {"seg", "pose"}
    assert row["dominant_term"] == "seg"
    assert row["score_neutral"] is True
    assert "NON-PROMOTABLE" in row["axis"]
    # one pair, orthogonal
    assert row["cosine_upper_triangle"][0]["cos"] == pytest.approx(0.0, abs=1e-9)
    assert row["n_conflicts"] == 0


def test_grad_interaction_row_all_inactive():
    row = gi.grad_interaction_row({"a": np.zeros(3)}, stage="ce", ep=1, k_pairs=8)
    assert row["n_active_terms"] == 0
    assert row["terms"] == []
    assert row["dominant_term"] is None


def test_upper_triangle_count():
    vecs = {"a": np.ones(2), "b": np.ones(2), "c": np.ones(2)}
    names, mat, _ = gi.cosine_matrix(vecs)
    ut = gi.upper_triangle(names, mat)
    assert len(ut) == 3  # C(3,2)
    for p in ut:
        assert p["cos"] == pytest.approx(1.0)


# ─────────────────────────── flatten pytree ─────────────────────────────────
def test_flatten_grad_tree_deterministic_order_and_none():
    tree = {"b": np.array([3.0, 4.0]), "a": {"y": np.array([2.0]), "x": np.array([1.0])},
            "c": None}
    flat = gi.flatten_grad_tree(tree)
    # sorted keys: a(x,y) then b then c(skip) => [1,2,3,4]
    assert list(flat) == [1.0, 2.0, 3.0, 4.0]


def test_flatten_grad_tree_list_and_scalar():
    tree = {"w": [np.array([1.0, 2.0]), np.array([3.0])], "s": 5.0}
    flat = gi.flatten_grad_tree(tree)
    # sorted: s(5) then w(1,2,3)
    assert list(flat) == [5.0, 1.0, 2.0, 3.0]


def test_flatten_grad_tree_empty():
    assert gi.flatten_grad_tree({}).size == 0
    assert gi.flatten_grad_tree(None).size == 0


# ─────────────────────────── RNG non-perturbation (mandatory) ────────────────
def test_rng_fingerprint_changes_on_draw_and_is_restored():
    np.random.seed(123)
    fp0 = gi.rng_fingerprint()
    _ = np.random.random(5)  # advance the numpy stream
    fp1 = gi.rng_fingerprint()
    assert fp0 != fp1  # fingerprint is sensitive to a draw
    np.random.set_state(np.random.RandomState(123).get_state())
    assert gi.rng_fingerprint() == fp0  # restore round-trips


def test_assert_rng_unperturbed_raises_on_change():
    a = {"np_pos": 1, "np_key_hash": 10, "mx_state": None}
    b = {"np_pos": 2, "np_key_hash": 10, "mx_state": None}
    gi.assert_rng_unperturbed(a, dict(a))  # identical -> no raise
    with pytest.raises(AssertionError):
        gi.assert_rng_unperturbed(a, b)


def test_mx_rng_guard_restores_numpy_stream_bit_identical():
    # THE binding score-neutrality proof: a draw taken AFTER a measurement block that itself
    # draws must be BIT-IDENTICAL to the draw taken without the block.
    np.random.seed(2026)
    ref = np.random.random(4)  # the "training" draw that must be unperturbed

    np.random.seed(2026)
    with gi.MxRngGuard(where="test"):
        _ = np.random.random(99)  # a heavy measurement pass drawing from the stream
    after = np.random.random(4)   # same training draw, post-measurement
    assert np.array_equal(ref, after)


def test_mx_rng_guard_forked_key_leaves_global_stream_neutral():
    # A well-behaved measurement uses an EXPLICIT forked key -> the GLOBAL mx stream is
    # untouched -> the guard passes AND the next global draw is bit-identical.
    mx = pytest.importorskip("mlx.core")
    mx.random.seed(7)
    ref = np.asarray(mx.random.uniform(shape=(4,)))

    mx.random.seed(7)
    with gi.MxRngGuard(where="test-mlx"):
        k = mx.random.key(999)
        _ = mx.random.uniform(shape=(50,), key=k)  # forked-key measurement draw
        mx.eval(_)
    after = np.asarray(mx.random.uniform(shape=(4,)))
    assert np.allclose(ref, after)


def test_mx_rng_guard_fails_closed_if_measurement_touches_global_mx_stream():
    # A measurement that draws from the GLOBAL mx stream perturbs it irreversibly -> the guard
    # must FAIL CLOSED (surface the reproducibility bug rather than silently mis-restore).
    mx = pytest.importorskip("mlx.core")
    mx.random.seed(11)
    with pytest.raises(AssertionError):
        with gi.MxRngGuard(where="test-mlx-violation"):
            _ = mx.random.uniform(shape=(50,))  # GLOBAL-stream draw (no explicit key)
            mx.eval(_)


def test_end_to_end_per_term_grads_on_mlx_module_rng_clean():
    """The exact pattern the trainer hook uses: per-term value_and_grad on an MLX module ->
    flatten -> cosine matrix, all inside the RNG guard. Proves (a) real MLX grads flatten and
    build a sensible synergy matrix, and (b) the whole measurement leaves numpy AND the global
    mx stream bit-identical (the mandatory score-neutrality bit-compare)."""
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(4, 3)

        def __call__(self, x):
            return self.lin(x)

    model = Tiny()
    x = mx.array(np.random.default_rng(0).standard_normal((5, 4)).astype(np.float32))

    _dir = mx.array(np.random.default_rng(2).standard_normal((5, 3)).astype(np.float32))

    def term_a(m):  # pushes the output along +_dir
        return mx.sum(m(x) * _dir)

    def term_b(m):  # pushes along -_dir -> gradient is the exact negation of term_a -> cos ~ -1
        return mx.sum(m(x) * (-_dir))

    # establish reference draws BEFORE the measurement
    np.random.seed(4242)
    mx.random.seed(99)
    ref_np = np.random.random(3)
    ref_mx = np.asarray(mx.random.uniform(shape=(3,)))

    np.random.seed(4242)
    mx.random.seed(99)
    with gi.MxRngGuard(where="e2e"):
        grads = {}
        for name, fn in (("a", term_a), ("b", term_b)):
            _, g = nn.value_and_grad(model, fn)(model)
            grads[name] = gi.flatten_grad_tree(g)
        row = gi.grad_interaction_row(grads, stage="ce", ep=10, k_pairs=5)

    # after the guarded measurement, the streams resume bit-identically
    after_np = np.random.random(3)
    after_mx = np.asarray(mx.random.uniform(shape=(3,)))
    assert np.array_equal(ref_np, after_np)
    assert np.allclose(ref_mx, after_mx)

    # the two terms have OPPOSING gradients on the linear weights -> negative cosine (a conflict)
    assert set(row["terms"]) == {"a", "b"}
    cos_ab = row["cosine_upper_triangle"][0]["cos"]
    assert cos_ab < 0.0
    assert row["n_conflicts"] >= 1


def test_mx_rng_guard_still_restores_on_exception():
    np.random.seed(5)
    ref = np.random.random(3)
    np.random.seed(5)
    with pytest.raises(RuntimeError):
        with gi.MxRngGuard():
            _ = np.random.random(10)
            raise RuntimeError("boom")
    # state restored even though the block raised (assertion skipped on exception path)
    assert np.array_equal(ref, np.random.random(3))
