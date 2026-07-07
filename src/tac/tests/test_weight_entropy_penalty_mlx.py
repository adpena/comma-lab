"""Tests for the MLX weight-entropy rate-in-the-loss port (council draft §22(2) fold).

Pins the port's binding contract:
  * λ=0 is a TRUE no-op — loss AND grads bitwise identical to a closure with no penalty code
    (not a computed-then-zeroed term).
  * Counted-tensor scope — the free curvelet bank (``B``/``*_B``) is EXCLUDED (rule 118);
    everything else is penalized (mirrors ``quantize_levelset_blob`` membership).
  * λ>0 — the term is finite, carries gradient to the counted weights, and descending it lowers
    the surrogate (loss-path smoke, CPU, few steps, tiny module — NOT a training run).
  * Micro-batch routing — ``LeverConfig`` holds ``we_lambda`` (default 0.0 = byte-identical) and
    ``_once_terms`` adds exactly ``λ·rate_term`` once (the serial/batched-twin equivalence).
  * numpy twins — soft MLX == soft numpy (parity); the hard codec-grid entropy metric behaves
    exactly (0 bits for constant symbols, 1 bit for a 2-symbol uniform stream).
  * DSL leg — ``WeightEntropyPenaltyMLX`` emits the trainer flag, is discovered by the
    lever_registry (no unmapped/stale gap), and is NEVER-FIRED in the activation ledger.
"""

from __future__ import annotations

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
mx.set_default_device(mx.cpu)
import mlx.nn as nn  # noqa: E402
from mlx.utils import tree_flatten  # noqa: E402

from tac.boundary_math.levelset_micro_batch_loss import LeverConfig, _once_terms  # noqa: E402
from tac.boundary_math.weight_entropy_penalty_mlx import (  # noqa: E402
    counted_param_items,
    is_counted_param,
    measured_symbol_entropy_bits_numpy,
    soft_symbol_entropy_bits_mlx,
    soft_symbol_entropy_bits_numpy,
    weight_entropy_rate_term_mlx,
)

_RATE_DENOM = 37_545_489


class _TinyCounted(nn.Module):
    """Tiny module with counted params only (deterministic init)."""

    def __init__(self, seed: int = 0):
        super().__init__()
        rng = np.random.default_rng(seed)
        self.lin = nn.Linear(8, 8)
        self.lin.weight = mx.array(rng.normal(size=(8, 8)).astype(np.float32))
        self.lin.bias = mx.array(rng.normal(size=(8,)).astype(np.float32))
        self.code = mx.array(rng.normal(size=(6, 4)).astype(np.float32))

    def __call__(self, x):
        return self.lin(x)


class _BankOnly(nn.Module):
    """Module whose ONLY param is the free bank -> the penalty must refuse (silent no-op ban)."""

    def __init__(self):
        super().__init__()
        self.B = mx.ones((4, 4))


# ---------------------------------------------------------------------------
# counted-vs-free split
# ---------------------------------------------------------------------------
def test_counted_predicate_excludes_free_bank_only():
    assert not is_counted_param("B")
    assert not is_counted_param("front_B")
    assert not is_counted_param("bank_B")
    for name in ("code", "in_proj.weight", "palette", "out_sdf.bias", "hidden.0.weight",
                 "Bx", "b", "film.weight"):
        assert is_counted_param(name), name


def test_counted_param_items_filters_and_sorts():
    flat = [("code", 1), ("B", 2), ("in_proj.weight", 3), ("x_B", 4), ("palette", 5)]
    got = counted_param_items(flat)
    assert got == [("code", 1), ("in_proj.weight", 3), ("palette", 5)]


# ---------------------------------------------------------------------------
# numpy hard metric (the NO-FAKE headline quantity)
# ---------------------------------------------------------------------------
def test_hard_entropy_constant_tensor_is_zero_bits():
    # constant positive value -> every symbol quantizes to 127 -> H = 0 exactly.
    h = measured_symbol_entropy_bits_numpy({"w": np.full((32,), 3.0, np.float32)})
    assert h == pytest.approx(0.0, abs=1e-12)


def test_hard_entropy_two_symbol_uniform_is_one_bit():
    w = np.array([1.0, -1.0] * 64, np.float32)  # symbols {127, -127}, uniform -> 1 bit
    h = measured_symbol_entropy_bits_numpy({"w": w})
    assert h == pytest.approx(1.0, abs=1e-9)


def test_hard_entropy_excludes_free_bank():
    # A high-entropy bank must not move the metric: only "w" (constant -> 0 bits) is counted.
    rng = np.random.default_rng(1)
    params = {"w": np.full((16,), 2.0, np.float32),
              "B": rng.normal(size=(64,)).astype(np.float32)}
    assert measured_symbol_entropy_bits_numpy(params) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# soft surrogate: MLX/numpy parity + sane bounds
# ---------------------------------------------------------------------------
def test_soft_entropy_mlx_matches_numpy_reference():
    rng = np.random.default_rng(2)
    w = rng.normal(size=(50, 7)).astype(np.float32)
    got = float(soft_symbol_entropy_bits_mlx(mx.array(w)))
    ref = soft_symbol_entropy_bits_numpy(w)
    assert got == pytest.approx(ref, rel=1e-4, abs=1e-4)


def test_soft_entropy_tracks_hard_entropy_on_grid_aligned_symbols():
    # Symbols exactly on the int8 grid, well separated vs sigma=0.2 -> soft ~= hard.
    rng = np.random.default_rng(3)
    syms = rng.choice([-127, -60, 0, 60, 127], size=2048).astype(np.float32)
    w = syms / 127.0  # max|w|=1 -> grid == syms (up to the 1e-8 scale eps)
    soft = soft_symbol_entropy_bits_numpy(w)
    hard = measured_symbol_entropy_bits_numpy({"w": w})
    assert soft == pytest.approx(hard, abs=0.05)


# ---------------------------------------------------------------------------
# MLX rate term: scope, scale, gradient, refuse-on-empty
# ---------------------------------------------------------------------------
def test_rate_term_scale_and_composition():
    m = _TinyCounted()
    total_bits, rate_term = weight_entropy_rate_term_mlx(m)
    mx.eval(total_bits, rate_term)
    # rate_term is exactly total_bits/8/N*25 (the contest rate scale).
    assert float(rate_term) == pytest.approx(float(total_bits) / 8.0 / _RATE_DENOM * 25.0,
                                             rel=1e-6)
    # total_bits == sum over counted tensors of H_t * numel_t (per-tensor recomputation).
    expect = 0.0
    for name, arr in counted_param_items(tree_flatten(m.parameters())):
        assert is_counted_param(name)
        expect += soft_symbol_entropy_bits_numpy(np.array(arr)) * arr.size
    assert float(total_bits) == pytest.approx(expect, rel=1e-4)
    assert np.isfinite(float(total_bits))


def test_rate_term_gradient_flows_to_counted_weights():
    m = _TinyCounted()

    def loss_fn(model):
        _bits, rate = weight_entropy_rate_term_mlx(model)
        return 15.0 * rate

    _val, grads = nn.value_and_grad(m, loss_fn)(m)
    flat = dict(tree_flatten(grads))
    gw = np.array(flat["lin.weight"])
    assert np.all(np.isfinite(gw))
    assert float(np.abs(gw).max()) > 0.0
    gc = np.array(flat["code"])
    assert float(np.abs(gc).max()) > 0.0


def test_rate_term_refuses_bank_only_model():
    with pytest.raises(ValueError, match="NO counted params"):
        weight_entropy_rate_term_mlx(_BankOnly())


def test_descending_the_term_lowers_the_surrogate_loss_path_smoke():
    """Loss-path smoke (CPU, tiny, few steps — NOT a training run): SGD on λ·rate_term alone
    strictly lowers the surrogate; the term RUNS in forward+backward+update."""
    import mlx.optimizers as optim

    m = _TinyCounted(seed=7)

    def loss_fn(model):
        _bits, rate = weight_entropy_rate_term_mlx(model)
        return 50.0 * rate

    vg = nn.value_and_grad(m, loss_fn)
    opt = optim.SGD(learning_rate=0.5)
    l0, g = vg(m)
    mx.eval(l0, g)
    for _ in range(30):
        _l, g = vg(m)
        opt.update(m, g)
        mx.eval(m.parameters(), opt.state)
    l1, _ = vg(m)
    mx.eval(l1)
    assert float(l1) < float(l0), (float(l0), float(l1))


# ---------------------------------------------------------------------------
# λ=0 no-op byte-identity (the sealed-config gate)
# ---------------------------------------------------------------------------
def test_lambda_zero_is_bitwise_noop_on_forward_and_backward():
    """The trainer's guard (`if we_lambda > 0.0:`) must make λ=0 a TRUE no-op: loss and every
    grad array bitwise identical to a closure containing NO penalty code at all."""
    x = mx.array(np.random.default_rng(5).normal(size=(4, 8)).astype(np.float32))

    def base_only(model):
        return mx.mean(mx.square(model(x)))

    def with_guarded_branch(model, we_lambda=0.0):
        loss = mx.mean(mx.square(model(x)))
        if we_lambda > 0.0:  # the trainer's exact guard pattern
            _bits, rate = weight_entropy_rate_term_mlx(model)
            loss = loss + we_lambda * rate
        return loss

    m1 = _TinyCounted(seed=11)
    m2 = _TinyCounted(seed=11)
    l1, g1 = nn.value_and_grad(m1, base_only)(m1)
    l2, g2 = nn.value_and_grad(m2, with_guarded_branch)(m2)
    mx.eval(l1, g1, l2, g2)
    assert np.array_equal(np.array(l1), np.array(l2))
    f1, f2 = dict(tree_flatten(g1)), dict(tree_flatten(g2))
    assert f1.keys() == f2.keys()
    for k in f1:
        assert np.array_equal(np.array(f1[k]), np.array(f2[k])), k


# ---------------------------------------------------------------------------
# micro-batch twin routing (LeverConfig + _once_terms)
# ---------------------------------------------------------------------------
def test_lever_config_default_off_and_once_terms_byte_identical():
    lc = LeverConfig()
    assert lc.we_lambda == 0.0
    m = _TinyCounted()
    out = _once_terms(m, lc)
    mx.eval(out)
    assert float(out) == 0.0  # all per-MODEL penalties off -> exact zero scalar


def test_once_terms_adds_exactly_lambda_times_rate_term():
    m = _TinyCounted()
    lam = 12.5
    lc = LeverConfig(we_lambda=lam)
    out = _once_terms(m, lc)
    _bits, rate = weight_entropy_rate_term_mlx(m, sigma=lc.we_sigma)
    mx.eval(out, rate)
    assert float(out) == pytest.approx(lam * float(rate), rel=1e-6)


# ---------------------------------------------------------------------------
# DSL leg: factory + registry coverage + activation ledger never-fired
# ---------------------------------------------------------------------------
def test_dsl_lever_factory_emits_the_trainer_flag():
    from tac.witness_dsl.curriculum_dsl import WeightEntropyPenaltyMLX

    lever = WeightEntropyPenaltyMLX(lam=15.0)
    assert lever.name == "WeightEntropyPenaltyMLX"
    assert lever.overrides == {"--weight-entropy-penalty-lambda": 15.0}
    assert "does NOT transfer" in lever.notes  # the borrowed-number firewall travels with it


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_dsl_lever_factory_rejects_non_positive_lambda(bad):
    from tac.witness_dsl.curriculum_dsl import WeightEntropyPenaltyMLX

    with pytest.raises(ValueError, match="positive finite"):
        WeightEntropyPenaltyMLX(lam=bad)


def test_lever_registry_holds_the_flag_no_unmapped_no_stale():
    from tac.witness_dsl.lever_registry import completeness, lever_factories

    facs = lever_factories()
    assert "WeightEntropyPenaltyMLX" in facs
    assert "--weight-entropy-penalty-lambda" in facs["WeightEntropyPenaltyMLX"]
    comp = completeness()
    assert "--weight-entropy-penalty-lambda" not in comp.unmapped
    assert "--weight-entropy-penalty-lambda" not in comp.stale


def test_activation_ledger_mlx_lever_is_never_fired_torch_history_separate():
    from tac.witness_dsl.activation_ledger import STATE_NEVER_FIRED, activation_status

    st = activation_status("WeightEntropyPenaltyMLX")
    assert st.state == STATE_NEVER_FIRED  # duty-to-measure: its own n600 A/B is owed
    # The torch vehicle's history stays attributed to the DISTINCT torch lever name.
    torch_st = activation_status("WeightEntropyPenalty")
    assert torch_st.lever == "WeightEntropyPenalty"


# ---------------------------------------------------------------------------
# equations leg builds (registration is done once by the landing agent, not the test)
# ---------------------------------------------------------------------------
def test_equations_leg_builds_with_firewalled_anchors():
    from tac.canonical_equations.weight_entropy_rate_in_loss_20260707 import (
        EQUATION_ID,
        build_weight_entropy_rate_in_loss_lever_v1,
        torch_vehicle_measured_live_delta_bytes,
    )

    eq = build_weight_entropy_rate_in_loss_lever_v1()
    assert eq.equation_id == EQUATION_ID
    assert torch_vehicle_measured_live_delta_bytes() == -16_007
    ids = [a.anchor_id for a in eq.empirical_anchors]
    assert "weight_entropy_torch_lambda50_live_decoder_byteclose_20260620" in ids
    assert "weight_entropy_mlx_port_never_fired_20260707" in ids
    # borrowed-number firewall welded into the port anchor
    port = next(a for a in eq.empirical_anchors if "mlx_port" in a.anchor_id)
    assert "NEVER-FIRED" in str(port.empirical_output)
