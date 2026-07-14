# SPDX-License-Identifier: MIT
"""Focused behavioral tests for the M+Adam NumPy/MLX implementation."""

from __future__ import annotations

import inspect
import math
import subprocess
import sys

import numpy as np
import pytest

from tac.optimization.mplus_adam_mlx import (
    RESEARCH_ONLY,
    MPlusAdam,
    mplus_adam_step_numpy,
)


def _kwargs(**overrides: float | int) -> dict[str, float | int]:
    values: dict[str, float | int] = {
        "additive_learning_rate": 1e-2,
        "multiplicative_learning_rate": 2e-2,
        "beta1": 0.9,
        "beta2": 0.999,
        "eps": 1e-8,
        "tau": 1e-1,
        "weight_decay": 0.1,
        "step": 1,
    }
    values.update(overrides)
    return values


def _manual_step(
    parameter: np.ndarray,
    gradient: np.ndarray,
    state: dict[str, np.ndarray],
    *,
    additive_learning_rate: float,
    multiplicative_learning_rate: float,
    beta1: float,
    beta2: float,
    eps: float,
    tau: float,
    weight_decay: float,
    step: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Independent literal transcription used only as a two-step oracle."""

    w = np.array(parameter, dtype=np.float32, copy=True)
    g = np.array(gradient, dtype=np.float32, copy=True)
    m = np.float32(beta1) * state["mean"] + np.float32(1 - beta1) * g
    v = np.float32(beta2) * state["variance"] + np.float32(1 - beta2) * g * g
    ge = np.float32(math.log(2.0)) * w * g
    ve = (
        np.float32(beta2) * state["exponent_variance"]
        + np.float32(1 - beta2) * ge * ge
    )
    mhat = m / np.float32(1 - beta1**step)
    vhat = v / np.float32(1 - beta2**step)
    vehat = ve / np.float32(1 - beta2**step)
    uadd = (
        -np.float32(additive_learning_rate)
        * mhat
        / (np.sqrt(vhat) + np.float32(eps))
    )
    utilde = (
        -np.float32(multiplicative_learning_rate)
        * ge
        / (np.sqrt(vehat) + np.float32(eps))
    )
    sign = np.where(w < 0, np.float32(-1), np.float32(1))
    rho = sign * np.maximum(np.abs(w), np.float32(tau))
    out = (
        np.float32(1 - additive_learning_rate * weight_decay) * w
        + w * (utilde / rho)
        + uadd
    )
    return out.astype(np.float32), {
        "mean": m.astype(np.float32),
        "variance": v.astype(np.float32),
        "exponent_variance": ve.astype(np.float32),
    }


def test_public_constructor_signature_is_trainer_contract() -> None:
    assert RESEARCH_ONLY is True
    signature = inspect.signature(MPlusAdam)
    assert list(signature.parameters) == [
        "learning_rate",
        "multiplicative_learning_rate",
        "tau",
        "betas",
        "eps",
        "weight_decay",
        "bias_correction",
    ]
    assert signature.parameters["tau"].default == pytest.approx(1e-6)
    assert signature.parameters["betas"].default == (0.9, 0.999)
    assert signature.parameters["eps"].default == pytest.approx(1e-8)
    assert signature.parameters["weight_decay"].default == pytest.approx(0.0)
    assert signature.parameters["bias_correction"].default is True


def test_mlx_constructor_refuses_a_second_unverified_bias_contract() -> None:
    probe = subprocess.run(
        [sys.executable, "-c", "import mlx.optimizers"],
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        pytest.skip("MLX optimizer unavailable on this host")
    with pytest.raises(ValueError, match="bias_correction must be exactly True"):
        MPlusAdam(
            learning_rate=1e-2,
            multiplicative_learning_rate=2e-2,
            bias_correction=False,
        )


def test_exact_one_step_algorithm_one_equations() -> None:
    w = np.array([2.0, -0.5, 0.0], dtype=np.float32)
    g = np.array([0.25, -0.125, 1.0], dtype=np.float32)
    out, state = mplus_adam_step_numpy(w, g, {}, **_kwargs())

    # At t=1 both bias-corrected second moments are exact squares of the
    # current gradients, so these closed forms do not reuse the implementation.
    ge = np.float32(math.log(2.0)) * w * g
    uadd = -np.float32(1e-2) * g / (np.abs(g) + np.float32(1e-8))
    utilde = -np.float32(2e-2) * ge / (np.abs(ge) + np.float32(1e-8))
    rho = np.where(w < 0, -1.0, 1.0).astype(np.float32) * np.maximum(
        np.abs(w), np.float32(1e-1)
    )
    expected = np.float32(1 - 1e-2 * 0.1) * w + w * (utilde / rho) + uadd

    np.testing.assert_allclose(out, expected, rtol=0, atol=2e-7)
    np.testing.assert_allclose(state["mean"], np.float32(0.1) * g, rtol=0, atol=0)
    np.testing.assert_allclose(
        state["variance"], np.float32(0.001) * g * g, rtol=0, atol=2e-10
    )
    np.testing.assert_allclose(
        state["exponent_variance"],
        np.float32(0.001) * ge * ge,
        rtol=0,
        atol=2e-10,
    )


def test_exact_two_step_matches_independent_literal_transcription() -> None:
    w0 = np.array([[1.25, -0.75], [0.02, -0.03]], dtype=np.float32)
    g1 = np.array([[0.4, -0.2], [0.8, -0.6]], dtype=np.float32)
    g2 = np.array([[-0.1, 0.3], [0.2, 0.7]], dtype=np.float32)
    zero = {name: np.zeros_like(w0) for name in ("mean", "variance", "exponent_variance")}

    expected1, expected_state1 = _manual_step(w0, g1, zero, **_kwargs())
    actual1, actual_state1 = mplus_adam_step_numpy(w0, g1, zero, **_kwargs())
    expected2, expected_state2 = _manual_step(
        expected1, g2, expected_state1, **_kwargs(step=2)
    )
    actual2, actual_state2 = mplus_adam_step_numpy(
        actual1, g2, actual_state1, **_kwargs(step=2)
    )

    np.testing.assert_allclose(actual1, expected1, rtol=0, atol=3e-7)
    np.testing.assert_allclose(actual2, expected2, rtol=0, atol=3e-7)
    for name in expected_state2:
        np.testing.assert_allclose(
            actual_state2[name], expected_state2[name], rtol=0, atol=3e-7
        )


def test_zero_is_owned_by_additive_branch_and_can_cross_sign() -> None:
    w = np.array(0.0, dtype=np.float32)
    g = np.array(3.0, dtype=np.float32)
    out, state = mplus_adam_step_numpy(
        w,
        g,
        {},
        **_kwargs(
            additive_learning_rate=0.1,
            multiplicative_learning_rate=0.7,
            weight_decay=0.0,
        ),
    )
    assert out.shape == ()
    assert float(out) == pytest.approx(-0.1, abs=2e-7)
    assert float(state["exponent_variance"]) == 0.0

    # The next accepted update crosses back through zero; the signed rho is
    # defined on the negative side and never traps the additive branch.
    out2, _ = mplus_adam_step_numpy(
        out,
        np.array(-30.0, dtype=np.float32),
        state,
        **_kwargs(
            additive_learning_rate=0.2,
            multiplicative_learning_rate=1e-8,
            weight_decay=0.0,
            step=2,
        ),
    )
    assert float(out2) > 0.0


def test_weight_decay_is_decoupled_base_before_branch_sum() -> None:
    w = np.array([1.5], dtype=np.float32)
    g = np.array([0.25], dtype=np.float32)
    kwargs = _kwargs(
        additive_learning_rate=0.05,
        multiplicative_learning_rate=0.03,
        weight_decay=0.4,
    )
    out, _ = mplus_adam_step_numpy(w, g, {}, **kwargs)

    ge = np.float32(math.log(2.0)) * w * g
    uadd = -np.float32(0.05) * g / (np.abs(g) + np.float32(1e-8))
    utilde = -np.float32(0.03) * ge / (np.abs(ge) + np.float32(1e-8))
    umul = utilde / w
    expected = np.float32(1 - 0.05 * 0.4) * w + w * umul + uadd
    wrong_decay_after_all_branches = (w + w * umul + uadd) * np.float32(1 - 0.05 * 0.4)

    np.testing.assert_allclose(out, expected, rtol=0, atol=2e-7)
    assert not np.allclose(out, wrong_decay_after_all_branches, rtol=0, atol=1e-5)


def test_numpy_step_is_fp32_deterministic_and_does_not_mutate_inputs() -> None:
    w = np.array([0.2, -1.7, 4.0], dtype=np.float64)
    g = np.array([-0.7, 0.1, 0.6], dtype=np.float64)
    state = {
        "mean": np.array([0.1, 0.2, -0.3], dtype=np.float32),
        "variance": np.array([0.2, 0.3, 0.4], dtype=np.float32),
        "exponent_variance": np.array([0.5, 0.6, 0.7], dtype=np.float32),
    }
    w_before = w.copy()
    g_before = g.copy()
    state_before = {name: value.copy() for name, value in state.items()}

    out1, state1 = mplus_adam_step_numpy(w, g, state, **_kwargs(step=9))
    out2, state2 = mplus_adam_step_numpy(w, g, state, **_kwargs(step=9))

    assert out1.dtype == np.float32
    assert np.array_equal(out1, out2)
    for name in state1:
        assert state1[name].dtype == np.float32
        assert np.array_equal(state1[name], state2[name])
    assert np.array_equal(w, w_before)
    assert np.array_equal(g, g_before)
    for name in state:
        assert np.array_equal(state[name], state_before[name])


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"additive_learning_rate": 0.0}, "additive_learning_rate"),
        ({"multiplicative_learning_rate": 0.0}, "multiplicative_learning_rate"),
        ({"beta1": -0.1}, "beta1"),
        ({"beta2": 1.0}, "beta2"),
        ({"eps": 0.0}, "eps"),
        ({"tau": 0.0}, "tau"),
        ({"weight_decay": -0.1}, "weight_decay"),
        ({"step": 0}, "step"),
        ({"step": 1.5}, "step"),
        ({"step": True}, "step"),
    ],
)
def test_invalid_hyperparameters_fail_closed(
    override: dict[str, float | int], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        mplus_adam_step_numpy(
            np.array([1.0], dtype=np.float32),
            np.array([0.1], dtype=np.float32),
            {},
            **_kwargs(**override),
        )


def test_malformed_state_and_shape_mismatch_fail_closed() -> None:
    w = np.ones((2,), dtype=np.float32)
    with pytest.raises(ValueError, match="state must be a mapping"):
        mplus_adam_step_numpy(w, w, [0.0, 0.0], **_kwargs())
    with pytest.raises(ValueError, match="state keys must be exactly"):
        mplus_adam_step_numpy(w, w, {"mean": np.zeros_like(w)}, **_kwargs())
    bad_state = {
        "mean": np.zeros((3,), dtype=np.float32),
        "variance": np.zeros((3,), dtype=np.float32),
        "exponent_variance": np.zeros((3,), dtype=np.float32),
    }
    with pytest.raises(ValueError, match="does not match parameter shape"):
        mplus_adam_step_numpy(w, w, bad_state, **_kwargs())
    with pytest.raises(ValueError, match="does not match gradient"):
        mplus_adam_step_numpy(w, np.ones((3,), dtype=np.float32), {}, **_kwargs())


def test_mlx_optimizer_two_step_matches_numpy_when_mlx_is_available() -> None:
    probe = subprocess.run(
        [sys.executable, "-c", "import mlx.optimizers"],
        check=False,
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        pytest.skip("MLX optimizer unavailable on this host")
    try:
        import mlx.core as mx
        import mlx.optimizers as optim
    except Exception as exc:  # headless/no-Metal CI is the expected common path.
        pytest.skip(f"MLX optimizer unavailable on this host: {type(exc).__name__}: {exc}")

    mx.set_default_device(mx.cpu)
    w0 = np.array([1.25, -0.75, 0.02], dtype=np.float32)
    g1 = np.array([0.4, -0.2, 0.8], dtype=np.float32)
    g2 = np.array([-0.1, 0.3, 0.2], dtype=np.float32)
    opt = MPlusAdam(
        learning_rate=1e-2,
        multiplicative_learning_rate=2e-2,
        tau=1e-1,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.1,
        bias_correction=True,
    )
    assert isinstance(opt, optim.Optimizer)
    params = {"w": mx.array(w0)}
    params = opt.apply_gradients({"w": mx.array(g1)}, params)
    params = opt.apply_gradients({"w": mx.array(g2)}, params)
    mx.eval(params, opt.state)

    expected1, state1 = mplus_adam_step_numpy(w0, g1, {}, **_kwargs())
    expected2, _ = mplus_adam_step_numpy(expected1, g2, state1, **_kwargs(step=2))
    np.testing.assert_allclose(np.asarray(params["w"]), expected2, rtol=0, atol=2e-6)
