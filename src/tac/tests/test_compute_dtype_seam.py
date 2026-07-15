# SPDX-License-Identifier: MIT
"""#509 batch 3: bf16/fp16 compute-seam mechanism tests.

Behavior (not constants): fp32 masters preserved + restored, gradients returned fp32
through the astype VJP, entry shims inert when inactive / casting when active, the
POST-normalize direction comparator's math, and the admission-gate law."""
from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from tac.canonical_equations.mixed_precision_compute_seam_20260715 import (  # noqa: E402
    build_bf16_compute_seam_gradient_quality_v1,
    gradient_quality_gate,
)
from tac.witness_control.compute_dtype_seam import (  # noqa: E402
    ComputeDtypeSeam,
    flatten_update_tree,
    resolve_compute_dtype,
    update_direction_stats,
)


def _tiny_module():
    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = nn.Linear(4, 3)

        def __call__(self, x):
            return self.lin(x)

        def sdf(self, x):
            return self.lin(x) * 2.0

    m = Tiny()
    mx.eval(m.parameters())
    return m


def _x():
    rng = np.random.default_rng(0)
    return mx.array(rng.standard_normal((5, 4)).astype(np.float32))


# ---- dtype resolution -----------------------------------------------------------------
def test_resolve_compute_dtype():
    assert resolve_compute_dtype("fp32") == mx.float32
    assert resolve_compute_dtype("bf16") == mx.bfloat16
    assert resolve_compute_dtype("fp16") == mx.float16
    with pytest.raises(ValueError, match="unknown compute dtype"):
        resolve_compute_dtype("int8")


def test_seam_refuses_fp32():
    with pytest.raises(ValueError, match="off-is-orphan"):
        ComputeDtypeSeam(_tiny_module(), "fp32", entry_methods=("__call__",))


# ---- entry shims ----------------------------------------------------------------------
def test_shims_inert_when_inactive():
    m = _tiny_module()
    x = _x()
    before = np.asarray(m(x))
    seam = ComputeDtypeSeam(m, "bf16", entry_methods=("__call__", "sdf"))
    assert seam.active is False
    after = np.asarray(m(x))
    # inactive seam => bit-identical pass-through (fp32 masters, fp32 inputs)
    assert np.array_equal(before, after)
    assert m(x).dtype == mx.float32


def test_shim_install_idempotent():
    m = _tiny_module()
    ComputeDtypeSeam(m, "bf16", entry_methods=("__call__",))
    call_1 = type(m).__call__
    ComputeDtypeSeam(m, "bf16", entry_methods=("__call__",))
    assert type(m).__call__ is call_1  # not double-wrapped
    assert getattr(type(m).__call__, "_cdt_orig", None) is not None


def test_shim_active_casts_inputs_and_outputs():
    m = _tiny_module()
    seam = ComputeDtypeSeam(m, "bf16", entry_methods=("__call__", "sdf"))
    # cast masters low-p (as the traced loss does), flip active, call: output must be fp32
    master = m.trainable_parameters()
    m.update(seam.cast_tree(master))
    assert m.lin.weight.dtype == mx.bfloat16
    seam.active = True
    try:
        out = m(_x())
        assert out.dtype == mx.float32  # cast_out_fp32 at the boundary
        out2 = m.sdf(_x())
        assert out2.dtype == mx.float32
    finally:
        seam.active = False
        m.update(master)
    assert m.lin.weight.dtype == mx.float32


def test_cast_out_fp32_maps_containers():
    m = _tiny_module()
    seam = ComputeDtypeSeam(m, "bf16", entry_methods=("__call__",))
    low = mx.array(np.ones(3, np.float32)).astype(mx.bfloat16)
    keep = mx.array(np.arange(3, dtype=np.int32))
    out = seam.cast_out_fp32((low, [low, keep], "tag"))
    assert out[0].dtype == mx.float32
    assert out[1][0].dtype == mx.float32
    assert out[1][1].dtype == mx.int32  # non-float untouched
    assert out[2] == "tag"


def test_cast_tree_only_touches_fp32_floats():
    m = _tiny_module()
    seam = ComputeDtypeSeam(m, "bf16", entry_methods=("__call__",))
    tree = {"w": mx.array(np.ones((2, 2), np.float32)),
            "i": mx.array(np.ones(2, np.int32))}
    cast = seam.cast_tree(tree)
    assert cast["w"].dtype == mx.bfloat16
    assert cast["i"].dtype == mx.int32


# ---- wrapped value_and_grad -----------------------------------------------------------
def test_vag_fp32_masters_preserved_and_grads_fp32():
    m = _tiny_module()
    seam = ComputeDtypeSeam(m, "bf16", entry_methods=("__call__",))
    x = _x()
    master_before = {k: np.asarray(v) for k, v in
                     [("w", m.lin.weight), ("b", m.lin.bias)]}
    seen: dict = {}

    def loss_fn(model, xx):
        seen["dtype"] = model.lin.weight.dtype  # captured INSIDE the trace
        return mx.sum(model(xx) ** 2)

    vag = seam.wrap_module_value_and_grad(loss_fn)
    loss, grads = vag(m, x)
    mx.eval(loss, grads)
    # inside the trace: low-precision compute
    assert seen["dtype"] == mx.bfloat16
    # outside: fp32 masters restored, bit-identical
    assert m.lin.weight.dtype == mx.float32
    assert np.array_equal(master_before["w"], np.asarray(m.lin.weight))
    assert np.array_equal(master_before["b"], np.asarray(m.lin.bias))
    # gradients arrive fp32 (astype VJP) and agree with the fp32 reference in direction
    flat = flatten_update_tree(grads)
    assert flat.size > 0
    ref_vag = nn.value_and_grad(m, loss_fn)
    _, ref_grads = ref_vag(m, x)
    mx.eval(ref_grads)
    stats = update_direction_stats(grads, ref_grads)
    assert stats["cosine"] > 0.98  # bf16 rounding, same direction
    assert 0.8 < stats["rel_norm"] < 1.2
    assert seam.active is False


def test_wrap_dual_restores_masters():
    m = _tiny_module()
    seam = ComputeDtypeSeam(m, "bf16", entry_methods=("__call__",))
    master = m.trainable_parameters()

    def raw_dual(model_params, other):
        # simulate the traced dual loss having left cast params installed
        m.update(seam.cast_tree(model_params))
        assert seam.active is True
        return "out"

    out = seam.wrap_dual_value_and_grad(raw_dual)(master, 1)
    assert out == "out"
    assert seam.active is False
    assert m.lin.weight.dtype == mx.float32


# ---- direction comparator -------------------------------------------------------------
def test_update_direction_stats_identity_and_opposite():
    t = {"a": mx.array(np.ones((2, 2), np.float32)),
         "b": mx.array(np.full(3, 2.0, np.float32))}
    s = update_direction_stats(t, t)
    assert s["cosine"] == pytest.approx(1.0)
    assert s["rel_norm"] == pytest.approx(1.0)
    assert set(s["per_group_cosine"]) == {"a", "b"}
    neg = {k: v * -1.0 for k, v in t.items()}
    s2 = update_direction_stats(neg, t)
    assert s2["cosine"] == pytest.approx(-1.0)


def test_update_direction_stats_size_mismatch_refused():
    a = {"a": mx.array(np.ones(2, np.float32))}
    b = {"a": mx.array(np.ones(3, np.float32))}
    with pytest.raises(ValueError, match="flattened size"):
        update_direction_stats(a, b)


def test_flatten_update_tree_sorted_deterministic():
    t1 = {"b": mx.array(np.array([2.0], np.float32)),
          "a": mx.array(np.array([1.0], np.float32))}
    t2 = {"a": mx.array(np.array([1.0], np.float32)),
          "b": mx.array(np.array([2.0], np.float32))}
    assert np.array_equal(flatten_update_tree(t1), flatten_update_tree(t2))
    assert flatten_update_tree(t1).tolist() == [1.0, 2.0]


# ---- the admission-gate law -----------------------------------------------------------
def test_gradient_quality_gate_admits_and_denies():
    ok = gradient_quality_gate(np.full(50, 0.999), np.full(50, 1.01))
    assert ok["admit"] is True and ok["n_steps"] == 50
    bad_dir = gradient_quality_gate(np.full(50, 0.9), np.full(50, 1.0))
    assert bad_dir["admit"] is False
    bad_mag = gradient_quality_gate(np.full(50, 1.0), np.full(50, 1.5))
    assert bad_mag["admit"] is False


def test_gradient_quality_gate_validation():
    with pytest.raises(ValueError, match="non-empty"):
        gradient_quality_gate(np.zeros(0), np.zeros(0))
    with pytest.raises(ValueError, match="matched|non-empty"):
        gradient_quality_gate(np.ones(3), np.ones(2))
    with pytest.raises(ValueError, match="rel_band"):
        gradient_quality_gate(np.ones(2), np.ones(2), rel_band=(1.1, 0.9))


def test_equation_builds_with_owed_anchor():
    eq = build_bf16_compute_seam_gradient_quality_v1()
    assert eq.equation_id == "bf16_compute_seam_gradient_quality_v1"
    assert len(eq.empirical_anchors) == 1
    assert "OWED" in str(eq.empirical_anchors[0].empirical_output)
