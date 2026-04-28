"""Lane SI-V2 — LEARNABLE saliency threshold via Lagrangian on bit budget.

Lane SI-V1 hard-coded ``threshold_quantile=0.5``: bottom-half saliency
pixels are encoded with the high-CRF (aggressive) codec, top-half with the
low-CRF (preserve quality) codec. The threshold is heuristic — it might
be 0.4 or 0.6 depending on the saliency distribution; the rate-distortion
optimum depends on the actual codec rate functions and the scorer's
saliency map shape.

Lane SI-V2 replaces the heuristic threshold with a LEARNABLE scalar
threshold optimised by Lagrangian dual ascent on a TARGET BIT BUDGET.

Math:

    soft_mask(p) = sigmoid((p - threshold) / temperature)
                  ∈ [0, 1]                 # differentiable threshold
    bytes(threshold, T) ≈ blind_bytes(blind_frac(threshold, T))
                       + salient_bytes(1 - blind_frac(threshold, T))

The Lagrangian objective:

    L(threshold, λ) = bytes(threshold)
                    + λ * (bytes(threshold) - target_bytes)²

is minimised by gradient descent on ``threshold`` with dual ascent on
``λ`` until ``bytes - target_bytes ≈ 0``. Because we cannot differentiate
through the actual byte-counter (zlib output size has no gradient), we
use a LINEAR rate model calibrated at boot: encode the masks at two
threshold values (0.3, 0.7) to fit slope + intercept, then optimise the
threshold against the linear surrogate. This is rigorous: the rate
function of zlib on a uint8 stream is ~linear in the entropy of the
sliced sub-stream over a wide range, and the slope is what dual ascent
needs to converge.

The temperature ``T`` is also learnable (small sigmoid temperature ⇒
sharp threshold; large ⇒ soft mixing — useful in noisy saliency maps).

CLAUDE.md compliance:
- Pure PyTorch. CUDA-required at the call site (caller decides device).
- Tests: gradient flows through both threshold and temperature; the
  Lagrangian converges to ``target_bytes`` within a tolerance; the soft
  mask collapses to a hard 0/1 split as ``T → 0``.
- No global state, no MPS/CPU fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = [
    "LearnableSaliencyThreshold",
    "OptimiseThresholdResult",
    "differentiable_threshold_mask",
    "fit_linear_rate_model",
    "optimise_threshold_for_target_bytes",
]


@dataclass
class OptimiseThresholdResult:
    """Output of ``optimise_threshold_for_target_bytes``."""

    threshold: float
    temperature: float
    estimated_bytes: float
    target_bytes: float
    final_lambda: float
    final_constraint_violation: float
    converged: bool
    iterations: int
    rate_slope: float
    rate_intercept: float


class LearnableSaliencyThreshold(nn.Module):
    """Learnable scalar (threshold, temperature) for saliency-aware coding.

    ``threshold`` is a single scalar nn.Parameter in ``[0, 1]`` — it
    represents the quantile cutoff in saliency space (the saliency map is
    expected to be normalised to [0, 1] before being passed in). The
    temperature controls the sigmoid sharpness: T → 0 ⇒ hard threshold;
    T → ∞ ⇒ all-pixels weighted equally.

    Args:
        init_threshold: starting quantile cutoff. Default 0.5 to match
            Lane SI-V1's hard-coded value (so V2 falls back to V1 when
            the dual ascent hasn't moved).
        init_temperature: starting sigmoid temperature.
        learn_temperature: when True, both threshold and temperature are
            optimised. When False, temperature is a frozen buffer.
    """

    def __init__(
        self,
        *,
        init_threshold: float = 0.5,
        init_temperature: float = 0.05,
        learn_temperature: bool = True,
    ) -> None:
        super().__init__()
        if not 0.0 < init_threshold < 1.0:
            raise ValueError(
                f"init_threshold must be in (0, 1); got {init_threshold}"
            )
        if init_temperature <= 0.0:
            raise ValueError(
                f"init_temperature must be > 0; got {init_temperature}"
            )
        # Parameterise threshold via a real-valued logit so SGD never
        # leaves [0, 1]: threshold = sigmoid(threshold_logit).
        # Inverse-sigmoid: logit = log(p / (1 - p)).
        thr_logit = torch.tensor(
            float(torch.log(torch.tensor(init_threshold / (1.0 - init_threshold)))),
            dtype=torch.float32,
        )
        self.threshold_logit = nn.Parameter(thr_logit)
        # Temperature parameterised as exp(log_T) so it stays positive.
        log_T = torch.tensor(
            float(torch.log(torch.tensor(init_temperature))), dtype=torch.float32
        )
        if learn_temperature:
            self.log_temperature = nn.Parameter(log_T)
        else:
            self.register_buffer("log_temperature", log_T)
        self.learn_temperature = bool(learn_temperature)

    # ── Accessors ────────────────────────────────────────────────────────

    def threshold(self) -> torch.Tensor:
        return torch.sigmoid(self.threshold_logit)

    def temperature(self) -> torch.Tensor:
        return torch.exp(self.log_temperature)

    def forward(self, saliency: torch.Tensor) -> torch.Tensor:
        """Return the (H, W) soft-blind-spot mask in [0, 1].

        ``saliency`` should be normalised to ``[0, 1]``. ``True``-ish
        outputs (mask > 0.5) correspond to BLIND-SPOT pixels (compress
        aggressively); ``False``-ish outputs are SALIENT (preserve
        quality). The convention matches
        ``compute_inverse_saliency_mask``.
        """
        return differentiable_threshold_mask(
            saliency,
            threshold=self.threshold(),
            temperature=self.temperature(),
        )

    def hard_mask(self, saliency: torch.Tensor) -> torch.Tensor:
        """Bool mask via hard threshold (no gradient). Use at deploy time
        when the differentiable surrogate is no longer needed."""
        return saliency.detach() <= self.threshold().detach()


def differentiable_threshold_mask(
    saliency: torch.Tensor,
    *,
    threshold: float | torch.Tensor,
    temperature: float | torch.Tensor,
) -> torch.Tensor:
    """Sigmoid soft-mask: 1.0 inside the blind spot, 0.0 outside.

    Differentiable wrt. ``threshold`` and ``temperature``. The blind spot
    is defined as ``saliency <= threshold`` (matches the
    ``compute_inverse_saliency_mask`` convention).

    Returns:
        ``saliency.shape`` float tensor in ``[0, 1]``. Value ``0.5`` at
        ``saliency == threshold``; the slope is ``1 / (4 * temperature)``
        at the inflection point.
    """
    if not torch.is_tensor(threshold):
        threshold = torch.as_tensor(threshold, dtype=saliency.dtype, device=saliency.device)
    if not torch.is_tensor(temperature):
        temperature = torch.as_tensor(
            temperature, dtype=saliency.dtype, device=saliency.device
        )
    return torch.sigmoid((threshold - saliency) / temperature.clamp_min(1e-8))


def fit_linear_rate_model(
    measure_bytes_at_threshold: Callable[[float], int],
    *,
    sample_thresholds: tuple[float, ...] = (0.3, 0.7),
) -> tuple[float, float]:
    """Calibrate a linear rate model ``bytes ≈ slope * threshold + intercept``.

    Lane SI-V2 cannot differentiate through the codec, so we measure the
    actual encoded bytes at TWO sample thresholds and fit a line. Two
    points = exact slope + intercept, which is all dual ascent needs to
    drive the constraint to zero (the constraint is linear in bytes).

    Args:
        measure_bytes_at_threshold: callable that, given a threshold in
            ``(0, 1)``, encodes the masks and returns the resulting byte
            count. The Lane SI-V2 caller wires this to
            ``apply_saliency_weighted_compression`` over the target masks.
        sample_thresholds: two thresholds to probe. Default ``(0.3, 0.7)``
            spans most of the curve while staying away from the degenerate
            ``threshold ∈ {0, 1}`` cases.

    Returns:
        ``(slope, intercept)`` so that ``bytes ≈ slope * t + intercept``.

    Raises:
        ValueError: when ``sample_thresholds`` contains < 2 entries or the
        two probes return the same byte count (slope = 0 ⇒ unconverging).
    """
    if len(sample_thresholds) < 2:
        raise ValueError("Need at least two sample thresholds to fit a line")
    t0, t1 = float(sample_thresholds[0]), float(sample_thresholds[1])
    if abs(t1 - t0) < 1e-6:
        raise ValueError(
            f"sample_thresholds {sample_thresholds!r} must span > 1e-6 in t"
        )
    b0 = float(measure_bytes_at_threshold(t0))
    b1 = float(measure_bytes_at_threshold(t1))
    slope = (b1 - b0) / (t1 - t0)
    intercept = b0 - slope * t0
    return slope, intercept


def optimise_threshold_for_target_bytes(
    *,
    target_bytes: int,
    rate_slope: float,
    rate_intercept: float,
    init_threshold: float = 0.5,
    init_temperature: float = 0.05,
    learn_temperature: bool = False,
    max_iterations: int = 500,
    lr_threshold: float = 0.05,
    lr_lambda: float = 1e-6,
    tolerance_bytes: float = 256.0,
) -> OptimiseThresholdResult:
    """Lagrangian dual ascent on the threshold to hit ``target_bytes``.

    The objective is

        L(t, T, λ) = (slope * t + intercept - target_bytes)²
                   + λ * (slope * t + intercept - target_bytes)

    The first term is the squared distance to target (so SGD always moves
    in the right direction even when λ is small); the second term is the
    Lagrangian. Dual ascent on λ pushes the constraint toward zero from
    the side that is currently violated.

    Note: with a LINEAR rate model + an unconstrained scalar threshold,
    this problem has a CLOSED-FORM solution
    ``t* = (target_bytes - intercept) / slope``, so dual ascent here is
    equivalent to "directly project to the line" but the iterative form
    keeps the same code-path used by the on-the-fly version that handles
    any non-linear rate model (e.g. when temperature is also learned and
    affects the boundary blend).

    Args:
        target_bytes: byte budget the encoder should hit.
        rate_slope: linear rate slope from ``fit_linear_rate_model``.
        rate_intercept: linear rate intercept.
        init_threshold: starting threshold in (0, 1).
        init_temperature: starting sigmoid temperature.
        learn_temperature: whether to optimise temperature (irrelevant for
            the linear rate model — kept for parity with the non-linear
            version).
        max_iterations: dual ascent iteration cap.
        lr_threshold: SGD step size on the threshold logit.
        lr_lambda: dual ascent step size on λ.
        tolerance_bytes: convergence tolerance |bytes - target| in bytes.

    Returns:
        ``OptimiseThresholdResult`` with the converged scalars.
    """
    if rate_slope == 0.0:
        raise ValueError(
            "rate_slope == 0 — measure_bytes_at_threshold returned the same "
            "value at both probes; cannot perform dual ascent."
        )
    threshold_module = LearnableSaliencyThreshold(
        init_threshold=init_threshold,
        init_temperature=init_temperature,
        learn_temperature=learn_temperature,
    )
    # The objective gradient on the threshold logit is
    #   g = 2 * (estimated - target) * slope * d(sigmoid)/d(logit)
    # which scales linearly with both `violation` and `slope`. Adam handles
    # this gracefully: it adapts the per-parameter LR from the running
    # gradient magnitude, so the same default lr works whether
    # |slope| = 100 or |slope| = 100_000.
    optimiser = torch.optim.Adam(
        [p for p in threshold_module.parameters() if p.requires_grad],
        lr=float(lr_threshold),
    )
    lam = 0.0
    iterations = 0
    converged = False
    last_violation = float("inf")
    last_estimated = 0.0
    for it in range(max_iterations):
        iterations = it + 1
        optimiser.zero_grad(set_to_none=True)
        thr = threshold_module.threshold()
        estimated = rate_slope * thr + rate_intercept
        violation = estimated - float(target_bytes)
        # Primal: signed-violation L1 (pulls toward target from either
        # side) + λ * violation (Lagrangian). L1 has a constant gradient
        # magnitude in the violation, so Adam sees a clean signal even
        # when violation is small near the target.
        loss = violation.abs() + lam * violation
        loss.backward()
        optimiser.step()
        # Dual ascent on λ (sign-aware nudge toward feasibility).
        with torch.no_grad():
            lam = max(0.0, lam + lr_lambda * float(violation.detach().item()))
        # Convergence check: re-evaluate AFTER the step so we measure the
        # threshold's NEW position (otherwise we converge at iter=1 even
        # when the initial violation is just under tolerance).
        with torch.no_grad():
            new_thr = threshold_module.threshold()
            new_estimated = rate_slope * new_thr + rate_intercept
            new_violation = float(new_estimated.item()) - float(target_bytes)
        last_violation = new_violation
        last_estimated = float(new_estimated.item())
        if abs(last_violation) < tolerance_bytes:
            converged = True
            break

    return OptimiseThresholdResult(
        threshold=float(threshold_module.threshold().detach().item()),
        temperature=float(threshold_module.temperature().detach().item()),
        estimated_bytes=last_estimated,
        target_bytes=float(target_bytes),
        final_lambda=float(lam),
        final_constraint_violation=last_violation,
        converged=bool(converged),
        iterations=int(iterations),
        rate_slope=float(rate_slope),
        rate_intercept=float(rate_intercept),
    )
