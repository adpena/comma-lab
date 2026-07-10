# SPDX-License-Identifier: MIT
"""Tests for the v8 B1 per-class DECOUPLED-FIELD partition head.

Covers: spec construction/validation, param shapes + counting + byte accounting, the numpy
reference forward (single + batch), the decoupling identity (∂phi_c/∂θ_{c'}=0), the tropical-argmax
composition + the "equals shared-mode when classes tied" invariant, MLX↔numpy parity, and the tied
init. The DSL lever compile/validate + the resume round-trip live in the sister trainer/DSL tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from tac.boundary_math.decoupled_field import (
    DecoupledFieldError,
    DecoupledFieldSpec,
    compose_argmax_partition,
    compose_softmax,
    counted_bytes,
    cross_class_jacobian_is_block_diagonal,
    decoupled_field_numpy_forward,
    init_field_params_numpy,
    param_count,
)

# MLX is optional in some CI images; skip the parity leg cleanly if absent.
mx = pytest.importorskip("mlx.core", reason="MLX required for the parity leg")
from tac.boundary_math.decoupled_field import make_decoupled_field_head_mlx  # noqa: E402


def _spec(**kw):
    base = dict(in_feat=6, mod_dim=8, n_classes=5, field_hidden=16, field_layers=2)
    base.update(kw)
    return DecoupledFieldSpec(**base)


# --------------------------------------------------------------------------- #
# 1. spec construction + validation                                            #
# --------------------------------------------------------------------------- #
def test_spec_defaults_are_canonical():
    s = DecoupledFieldSpec(in_feat=6, mod_dim=8)
    assert s.n_classes == 5  # the measured N_CLASSES (comma10k canonical)
    assert s.field_hidden == 32
    assert s.field_layers == 2
    assert s.activation == "relu"


@pytest.mark.parametrize("bad", [
    dict(in_feat=0), dict(mod_dim=0), dict(n_classes=1),
    dict(field_hidden=0), dict(field_layers=0), dict(activation="sigmoid"),
])
def test_spec_rejects_out_of_contract(bad):
    with pytest.raises(DecoupledFieldError):
        _spec(**bad)


# --------------------------------------------------------------------------- #
# 2. param count + counted bytes                                               #
# --------------------------------------------------------------------------- #
def test_param_count_matches_explicit_shapes():
    s = _spec(field_hidden=16, field_layers=2)
    k, i, m, h, ell = 5, 6, 8, 16, 2
    expected = (k * i * h) + (k * h) + (k * m * 2 * ell * h) + (k * ell * h * h) \
        + (k * ell * h) + (k * h) + k
    assert param_count(s) == expected


def test_counted_bytes_all_params_counted_no_free_table():
    s = _spec()
    cb = counted_bytes(s, quant_bits=8)
    assert cb["n_params"] == param_count(s)
    assert cb["raw_bytes_uncoded"] == pytest.approx(param_count(s) * 1.0)  # 8 bits => 1 byte/param
    assert cb["rate_term_uncoded_S"] > 0.0
    # every param is video-derived => counted (unlike the texture trunk's rule-118-free bank)
    assert "no rule-118-free table" in cb["note"]


# --------------------------------------------------------------------------- #
# 3. numpy reference forward — shapes (single + batch)                         #
# --------------------------------------------------------------------------- #
def test_numpy_forward_single_and_batch_shapes():
    s = _spec()
    params = init_field_params_numpy(s, seed=1)
    coord = np.random.default_rng(0).standard_normal((20, s.in_feat)).astype(np.float32)
    phi_single = decoupled_field_numpy_forward(params, coord, np.zeros(s.mod_dim, np.float32), s)
    assert phi_single.shape == (20, s.n_classes)
    codes = np.random.default_rng(2).standard_normal((4, s.mod_dim)).astype(np.float32)
    phi_batch = decoupled_field_numpy_forward(params, coord, codes, s)
    assert phi_batch.shape == (4, 20, s.n_classes)


def test_numpy_forward_batch_row_equals_single():
    """Batch row b must equal the single forward with code = codes[b] (no cross-pair leakage)."""
    s = _spec()
    params = init_field_params_numpy(s, seed=3)
    coord = np.random.default_rng(0).standard_normal((11, s.in_feat)).astype(np.float32)
    codes = np.random.default_rng(4).standard_normal((3, s.mod_dim)).astype(np.float32)
    phi_batch = decoupled_field_numpy_forward(params, coord, codes, s)
    for b in range(3):
        single = decoupled_field_numpy_forward(params, coord, codes[b], s)
        assert np.allclose(phi_batch[b], single, atol=1e-6)


def test_numpy_forward_rejects_bad_shapes():
    s = _spec()
    params = init_field_params_numpy(s)
    with pytest.raises(DecoupledFieldError):
        decoupled_field_numpy_forward(params, np.zeros((5, s.in_feat + 1), np.float32),
                                      np.zeros(s.mod_dim, np.float32), s)
    with pytest.raises(DecoupledFieldError):
        decoupled_field_numpy_forward(params, np.zeros((5, s.in_feat), np.float32),
                                      np.zeros(s.mod_dim + 1, np.float32), s)


# --------------------------------------------------------------------------- #
# 4. the decoupling identity ∂phi_c/∂θ_{c'} = 0 (NO-FAKE proof)                #
# --------------------------------------------------------------------------- #
def test_cross_class_jacobian_is_block_diagonal():
    assert cross_class_jacobian_is_block_diagonal(_spec()) is True


def test_perturbing_one_class_moves_only_that_class():
    """Direct proof: perturb class kp's w_out, ONLY phi[:, kp] changes."""
    s = _spec()
    params = init_field_params_numpy(s, seed=7, scale=0.3)
    coord = np.random.default_rng(0).standard_normal((9, s.in_feat)).astype(np.float32)
    code = np.random.default_rng(1).standard_normal((s.mod_dim,)).astype(np.float32)
    base = decoupled_field_numpy_forward(params, coord, code, s)
    kp = 2
    pert = {k: v.copy() for k, v in params.items()}
    pert["w_out"][kp] += 0.5
    out = decoupled_field_numpy_forward(pert, coord, code, s)
    delta = np.abs(out - base)
    assert delta[:, kp].max() > 1e-4          # the perturbed class moved
    for kc in range(s.n_classes):
        if kc != kp:
            assert delta[:, kc].max() < 1e-9  # every other class is invariant (decoupled)


# --------------------------------------------------------------------------- #
# 5. composition forward + the "equals shared-mode when tied" invariant         #
# --------------------------------------------------------------------------- #
def test_compose_argmax_is_mode_agnostic_given_same_phi():
    """The tropical-argmax composition is identical regardless of phi's provenance."""
    phi = np.random.default_rng(0).standard_normal((30, 5)).astype(np.float32)
    part_a = compose_argmax_partition(phi)
    part_b = np.argmax(phi, axis=-1)  # the shared-head composition over the SAME phi
    assert np.array_equal(part_a, part_b)


def test_tied_fields_reduce_to_shared_constant_partition():
    """Decoupled fields with tied init produce equal columns => argmax = class 0 everywhere,
    IDENTICAL to a shared head fed the same tied logits (the increment-1 equivalence anchor)."""
    s = _spec()
    tied = init_field_params_numpy(s, seed=5, tied_init=True)
    coord = np.random.default_rng(0).standard_normal((25, s.in_feat)).astype(np.float32)
    code = np.random.default_rng(1).standard_normal((s.mod_dim,)).astype(np.float32)
    phi = decoupled_field_numpy_forward(tied, coord, code, s)
    # all K columns identical (tied fields)
    for kc in range(1, s.n_classes):
        assert np.allclose(phi[:, 0], phi[:, kc], atol=1e-6)
    part = compose_argmax_partition(phi)
    assert np.array_equal(part, np.zeros(25, dtype=part.dtype))  # deterministic tie-break -> 0
    # a shared softmax over the same tied phi gives the SAME argmax
    soft = compose_softmax(phi, temp=0.1)
    assert np.array_equal(np.argmax(soft, axis=-1), part)


def test_compose_argmax_applies_b_c_offset():
    """b_c shifts the tropical argmax (the ~0-byte per-class tie calibration, SPEC §1)."""
    phi = np.zeros((4, 5), np.float32)  # all tied -> argmax 0 without bias
    assert np.array_equal(compose_argmax_partition(phi), np.zeros(4, int))
    b_c = np.array([0.0, 0.0, 5.0, 0.0, 0.0], np.float32)  # class 2 wins with the offset
    assert np.array_equal(compose_argmax_partition(phi, b_c=b_c), np.full(4, 2, int))


def test_compose_softmax_rejects_nonpositive_temp():
    with pytest.raises(DecoupledFieldError):
        compose_softmax(np.zeros((2, 5), np.float32), temp=0.0)


def test_compose_softmax_rows_sum_to_one():
    phi = np.random.default_rng(0).standard_normal((17, 5)).astype(np.float32)
    soft = compose_softmax(phi, temp=0.2)
    assert np.allclose(soft.sum(axis=-1), 1.0, atol=1e-5)


# --------------------------------------------------------------------------- #
# 6. tied init structural check                                                 #
# --------------------------------------------------------------------------- #
def test_tied_init_makes_all_class_slices_identical():
    s = _spec()
    tied = init_field_params_numpy(s, seed=9, tied_init=True)
    for name, arr in tied.items():
        for kc in range(1, s.n_classes):
            assert np.array_equal(arr[0], arr[kc]), name


def test_untied_init_is_class_distinct():
    s = _spec()
    p = init_field_params_numpy(s, seed=9, tied_init=False)
    # weight tensors differ across classes (biases are zero-init, so check a weight)
    assert not np.array_equal(p["w_in"][0], p["w_in"][1])


# --------------------------------------------------------------------------- #
# 7. MLX ↔ numpy parity (compute-substrate law: numpy is the portable authority) #
# --------------------------------------------------------------------------- #
def test_mlx_matches_numpy_single_relu():
    s = _spec(activation="relu")
    params = init_field_params_numpy(s, seed=2, scale=0.2)
    head = make_decoupled_field_head_mlx(s, seed=2, scale=0.2)
    coord = np.random.default_rng(0).standard_normal((30, s.in_feat)).astype(np.float32)
    code = np.random.default_rng(1).standard_normal((s.mod_dim,)).astype(np.float32)
    ref = decoupled_field_numpy_forward(params, coord, code, s)
    got = np.asarray(head.phi_single(mx.array(coord), mx.array(code)))
    # float32-level (numpy runs fp64 then casts); relu sign-flips near zero widen the bound modestly
    assert np.abs(got - ref).max() < 5e-3


def test_mlx_matches_numpy_tanh_tight():
    """tanh (smooth, no relu discontinuity) => a tight parity bound."""
    s = _spec(activation="tanh")
    params = init_field_params_numpy(s, seed=2, scale=0.2)
    head = make_decoupled_field_head_mlx(s, seed=2, scale=0.2)
    coord = np.random.default_rng(0).standard_normal((30, s.in_feat)).astype(np.float32)
    codes = np.random.default_rng(1).standard_normal((3, s.mod_dim)).astype(np.float32)
    ref = decoupled_field_numpy_forward(params, coord, codes, s)
    got = np.asarray(head.phi_batch(mx.array(coord), mx.array(codes)))
    assert np.abs(got - ref).max() < 1e-3


def test_mlx_head_param_count_matches_spec():
    from mlx.utils import tree_flatten
    s = _spec()
    head = make_decoupled_field_head_mlx(s, seed=0)
    n = int(sum(int(np.prod(v.shape)) for _, v in tree_flatten(head.parameters())))
    assert n == param_count(s)
