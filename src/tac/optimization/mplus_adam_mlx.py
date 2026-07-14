# SPDX-License-Identifier: MIT
"""M+Adam additive/multiplicative optimizer with a NumPy-fp32 authority.

This module implements Algorithm 1 from M+Adam (arXiv:2607.10611) exactly as
preregistered by the per-class convergence A/B specification.  The additive
branch is bias-corrected AdamW.  The multiplicative branch preconditions the
exponent-space gradient ``ln(2) * w * g`` with its own second moment and maps
the exponent update back through the paper's signed, thresholded ``rho``.

The deterministic :func:`mplus_adam_step_numpy` implementation is the
portable fp32 reference.  :class:`MPlusAdam` lazily constructs a genuine
``mlx.optimizers.Optimizer`` subclass, so importing the NumPy reference does
not initialize MLX/Metal on headless hosts.  The returned MLX optimizer keeps
all three moment leaves in the standard optimizer state tree, which lets the
witness trainer's existing flatten/restore checkpoint path preserve them.

This is a training-gradient actuator only.  It makes no convergence or score
claim; the fp32 single-video-INR transfer remains an unmeasured INSTANCE until
the real-n600 matched arm is run and adjudicated.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

__all__ = ["RESEARCH_ONLY", "MPlusAdam", "mplus_adam_step_numpy"]

# This primitive is intentionally dormant until a separately reviewed trainer
# lever, resume contract, and real matched-arm receipt exist.
RESEARCH_ONLY = True

_LN2_F32 = np.float32(math.log(2.0))
_STATE_KEYS = ("mean", "variance", "exponent_variance")


def _validate_scalar_hyperparameters(
    *,
    additive_learning_rate: float,
    multiplicative_learning_rate: float,
    beta1: float,
    beta2: float,
    eps: float,
    tau: float,
    weight_decay: float,
) -> None:
    values = {
        "additive_learning_rate": additive_learning_rate,
        "multiplicative_learning_rate": multiplicative_learning_rate,
        "beta1": beta1,
        "beta2": beta2,
        "eps": eps,
        "tau": tau,
        "weight_decay": weight_decay,
    }
    numeric: dict[str, float] = {}
    for name, value in values.items():
        try:
            numeric[name] = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name} must be a real scalar, got {value!r}") from exc
        if not math.isfinite(numeric[name]):
            raise ValueError(f"{name} must be finite, got {value!r}")
    if not numeric["additive_learning_rate"] > 0.0:
        raise ValueError(
            "additive_learning_rate must be > 0, "
            f"got {additive_learning_rate!r}"
        )
    if not numeric["multiplicative_learning_rate"] > 0.0:
        raise ValueError(
            "multiplicative_learning_rate must be > 0, "
            f"got {multiplicative_learning_rate!r}"
        )
    if not 0.0 <= numeric["beta1"] < 1.0:
        raise ValueError(f"beta1 must be in [0, 1), got {beta1!r}")
    if not 0.0 <= numeric["beta2"] < 1.0:
        raise ValueError(f"beta2 must be in [0, 1), got {beta2!r}")
    if not numeric["eps"] > 0.0:
        raise ValueError(f"eps must be > 0, got {eps!r}")
    if not numeric["tau"] > 0.0:
        raise ValueError(f"tau must be > 0, got {tau!r}")
    if numeric["weight_decay"] < 0.0:
        raise ValueError(f"weight_decay must be >= 0, got {weight_decay!r}")


def _fp32_array(value: Any, *, name: str) -> np.ndarray:
    try:
        out = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be convertible to a float32 array") from exc
    if not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain only finite values")
    return out


def _numpy_state(
    state: Mapping[str, Any] | None,
    *,
    shape: tuple[int, ...],
) -> dict[str, np.ndarray]:
    if state is not None and not isinstance(state, Mapping):
        raise ValueError(f"state must be a mapping or None, got {type(state).__name__}")
    if state is None or len(state) == 0:
        return {name: np.zeros(shape, dtype=np.float32) for name in _STATE_KEYS}
    missing = sorted(set(_STATE_KEYS) - set(state))
    extra = sorted(set(state) - set(_STATE_KEYS))
    if missing or extra:
        raise ValueError(
            "state keys must be exactly "
            f"{list(_STATE_KEYS)!r}; missing={missing!r}, extra={extra!r}"
        )
    out: dict[str, np.ndarray] = {}
    for name in _STATE_KEYS:
        value = _fp32_array(state[name], name=f"state[{name!r}]")
        if value.shape != shape:
            raise ValueError(
                f"state[{name!r}] shape {value.shape!r} does not match parameter "
                f"shape {shape!r}"
            )
        out[name] = value
    return out


def mplus_adam_step_numpy(
    parameter: Any,
    gradient: Any,
    state: Mapping[str, Any] | None,
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
    """Apply one deterministic NumPy-fp32 M+Adam Algorithm-1 update.

    ``step`` is the one-indexed accepted optimizer update.  ``state`` is either
    empty/``None`` for a fresh leaf or a mapping containing exactly ``mean``,
    ``variance``, and ``exponent_variance``.  Inputs and state are never
    mutated; returned arrays are new contiguous fp32 values.
    """

    _validate_scalar_hyperparameters(
        additive_learning_rate=additive_learning_rate,
        multiplicative_learning_rate=multiplicative_learning_rate,
        beta1=beta1,
        beta2=beta2,
        eps=eps,
        tau=tau,
        weight_decay=weight_decay,
    )
    try:
        step_i = int(step)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"step must be a one-indexed integer, got {step!r}") from exc
    if isinstance(step, bool) or step_i != step or step_i < 1:
        raise ValueError(f"step must be a one-indexed integer, got {step!r}")

    parameter_f32 = _fp32_array(parameter, name="parameter")
    gradient_f32 = _fp32_array(gradient, name="gradient")
    if parameter_f32.shape != gradient_f32.shape:
        raise ValueError(
            f"parameter shape {parameter_f32.shape!r} does not match gradient "
            f"shape {gradient_f32.shape!r}"
        )
    moments = _numpy_state(state, shape=parameter_f32.shape)

    b1 = np.float32(beta1)
    b2 = np.float32(beta2)
    one = np.float32(1.0)
    # Form the complements from the configured Python scalars, matching MLX's
    # ``(1 - beta) * array`` semantics. Computing ``1-float32(beta)`` instead
    # would introduce a different rounded coefficient and break parity.
    one_minus_b1 = np.float32(1.0 - float(beta1))
    one_minus_b2 = np.float32(1.0 - float(beta2))
    mean = b1 * moments["mean"] + one_minus_b1 * gradient_f32
    variance = b2 * moments["variance"] + one_minus_b2 * np.square(gradient_f32)

    exponent_gradient = _LN2_F32 * parameter_f32 * gradient_f32
    exponent_variance = (
        b2 * moments["exponent_variance"]
        + one_minus_b2 * np.square(exponent_gradient)
    )

    correction1 = np.float32(1.0 - float(beta1) ** step_i)
    correction2 = np.float32(1.0 - float(beta2) ** step_i)
    mean_hat = mean / correction1
    variance_hat = variance / correction2
    exponent_variance_hat = exponent_variance / correction2

    eps_f32 = np.float32(eps)
    additive_update = (
        -np.float32(additive_learning_rate)
        * mean_hat
        / (np.sqrt(variance_hat) + eps_f32)
    )
    exponent_update = (
        -np.float32(multiplicative_learning_rate)
        * exponent_gradient
        / (np.sqrt(exponent_variance_hat) + eps_f32)
    )

    # sign_nonzero(0)=+1 defines rho, while the later w*u_mul product keeps
    # the multiplicative contribution exactly zero at w=0.
    sign_nonzero = np.where(parameter_f32 < np.float32(0.0), -one, one)
    rho = sign_nonzero * np.maximum(np.abs(parameter_f32), np.float32(tau))
    multiplicative_update = exponent_update / rho
    decay_base = (
        one - np.float32(additive_learning_rate) * np.float32(weight_decay)
    ) * parameter_f32
    new_parameter = decay_base + parameter_f32 * multiplicative_update + additive_update

    new_state = {
        "mean": np.array(mean, dtype=np.float32, copy=True, order="C"),
        "variance": np.array(variance, dtype=np.float32, copy=True, order="C"),
        "exponent_variance": np.array(
            exponent_variance, dtype=np.float32, copy=True, order="C"
        ),
    }
    return np.array(new_parameter, dtype=np.float32, copy=True, order="C"), new_state


_MLX_OPTIMIZER_CLASS: type | None = None


def _build_mlx_optimizer_class() -> type:
    """Create and cache the real MLX Optimizer subclass on an MLX-capable host."""

    global _MLX_OPTIMIZER_CLASS
    if _MLX_OPTIMIZER_CLASS is not None:
        return _MLX_OPTIMIZER_CLASS

    try:
        import mlx.core as mx
        import mlx.optimizers as optim
    except Exception as exc:  # pragma: no cover - depends on host Metal access.
        raise RuntimeError(
            "MPlusAdam requires an MLX-capable Apple Silicon host; import the "
            "NumPy reference mplus_adam_step_numpy for headless verification"
        ) from exc

    class _MPlusAdamOptimizer(optim.Optimizer):
        def __init__(
            self,
            learning_rate: float | Callable[[Any], Any],
            multiplicative_learning_rate: float | Callable[[Any], Any],
            tau: float = 1e-6,
            betas: tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 0.0,
            bias_correction: bool = True,
        ) -> None:
            if bias_correction is not True:
                raise ValueError(
                    "bias_correction must be exactly True: the deterministic "
                    "NumPy-fp32 authority is bias-corrected and no second "
                    "reference contract exists"
                )
            try:
                beta_count = len(betas)
            except TypeError as exc:
                raise ValueError(
                    f"betas must contain exactly two values, got {betas!r}"
                ) from exc
            if beta_count != 2:
                raise ValueError(f"betas must contain exactly two values, got {betas!r}")
            beta1, beta2 = float(betas[0]), float(betas[1])
            # Callable schedules are validated by the trainer/config layer at
            # their resolved values.  Scalar constructor inputs fail closed here.
            additive_for_validation = 1.0 if callable(learning_rate) else float(learning_rate)
            multiplicative_for_validation = (
                1.0
                if callable(multiplicative_learning_rate)
                else float(multiplicative_learning_rate)
            )
            _validate_scalar_hyperparameters(
                additive_learning_rate=additive_for_validation,
                multiplicative_learning_rate=multiplicative_for_validation,
                beta1=beta1,
                beta2=beta2,
                eps=float(eps),
                tau=float(tau),
                weight_decay=float(weight_decay),
            )
            super().__init__()
            self._maybe_schedule("learning_rate", learning_rate)
            self._maybe_schedule(
                "multiplicative_learning_rate", multiplicative_learning_rate
            )
            self.betas = (beta1, beta2)
            self.eps = float(eps)
            self.tau = float(tau)
            self.weight_decay = float(weight_decay)
            self.bias_correction = True

        @property
        def multiplicative_learning_rate(self) -> Any:
            return self.state["multiplicative_learning_rate"]

        def init_single(self, parameter: Any, state: dict[str, Any]) -> None:
            state["mean"] = mx.zeros_like(parameter)
            state["variance"] = mx.zeros_like(parameter)
            state["exponent_variance"] = mx.zeros_like(parameter)

        def apply_single(
            self,
            gradient: Any,
            parameter: Any,
            state: dict[str, Any],
        ) -> Any:
            additive_lr = self.learning_rate.astype(gradient.dtype)
            multiplicative_lr = self.multiplicative_learning_rate.astype(gradient.dtype)
            beta1, beta2 = self.betas

            mean = beta1 * state["mean"] + (1.0 - beta1) * gradient
            variance = beta2 * state["variance"] + (1.0 - beta2) * mx.square(gradient)
            exponent_gradient = math.log(2.0) * parameter * gradient
            exponent_variance = (
                beta2 * state["exponent_variance"]
                + (1.0 - beta2) * mx.square(exponent_gradient)
            )
            state["mean"] = mean
            state["variance"] = variance
            state["exponent_variance"] = exponent_variance

            correction1 = 1.0 - beta1**self.step
            correction2 = 1.0 - beta2**self.step
            mean_hat = mean / correction1
            variance_hat = variance / correction2
            exponent_variance_hat = exponent_variance / correction2

            additive_update = (
                -additive_lr * mean_hat / (mx.sqrt(variance_hat) + self.eps)
            )
            exponent_update = (
                -multiplicative_lr
                * exponent_gradient
                / (mx.sqrt(exponent_variance_hat) + self.eps)
            )
            sign_nonzero = mx.where(parameter < 0, -mx.ones_like(parameter), mx.ones_like(parameter))
            rho = sign_nonzero * mx.maximum(mx.abs(parameter), self.tau)
            multiplicative_update = exponent_update / rho
            decay_base = (1.0 - additive_lr * self.weight_decay) * parameter
            return decay_base + parameter * multiplicative_update + additive_update

    _MPlusAdamOptimizer.__name__ = "MPlusAdam"
    _MPlusAdamOptimizer.__qualname__ = "MPlusAdam"
    _MPlusAdamOptimizer.__module__ = __name__
    _MLX_OPTIMIZER_CLASS = _MPlusAdamOptimizer
    return _MPlusAdamOptimizer


class MPlusAdam:
    """Lazy constructor for the real MLX M+Adam ``Optimizer`` subclass.

    The public signature is intentionally identical to the trainer integration
    contract.  Construction, unlike module import, requires working MLX.
    """

    def __new__(
        cls,
        learning_rate: float | Callable[[Any], Any],
        multiplicative_learning_rate: float | Callable[[Any], Any],
        tau: float = 1e-6,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        bias_correction: bool = True,
    ) -> Any:
        implementation = _build_mlx_optimizer_class()
        return implementation(
            learning_rate=learning_rate,
            multiplicative_learning_rate=multiplicative_learning_rate,
            tau=tau,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            bias_correction=bias_correction,
        )
