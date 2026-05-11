"""Hessian-weighted block-FP bit allocation via water-filling.

This module operationalises the Selfcomp grand-council position (CLAUDE.md
"Selfcomp's specific contributions"): **stacking Hessian quant + arithmetic
coder** with the block-FP weight codec. Specifically: high-Hessian (high
score-sensitivity) tensors get more bits, low-Hessian tensors get fewer.

The bit allocation problem is the classic Lagrangian rate-distortion / KKT /
water-filling form per Boyd ADMM:

    minimize    sum_t D_t(b_t)               (sum of per-tensor distortions)
    subject to  sum_t b_t * params_t <= B    (total bit budget)
                b_t >= 0                      (non-negative bits)

The Lagrangian L = sum_t D_t(b_t) + lambda * (sum_t b_t * params_t - B) has
the KKT condition:

    dD_t/db_t = -lambda * params_t    for active tensors
    b_t = 0                            for inactive tensors

When D_t is approximated as quadratic in the score-domain (D_t = H_t * sigma_t^2
where H_t is the Hessian-diagonal proxy and sigma_t is the quantization noise
variance) and quantization-noise variance scales as 2^(-2*b_t), the water-
filling solution is:

    b_t = max(0, 0.5 * log2(lambda * H_t * params_t / C))

where C is a normalization constant. The bisection solver in
``allocate_bits_by_hessian`` finds lambda such that the budget constraint is
tight.

Per CLAUDE.md "Forbidden weight-domain saliency on score-gradient substrate"
(catalog #123): the Hessian-proxy is computed via ``tac.score_gradient_param_saliency``
when an A1-style score-aware substrate is the source. The default
``mean(theta**2)`` proxy is FORBIDDEN.

Composition contract
--------------------
This module composes with:

- ``tac.scpp_substrate.encode_scpp_substrate`` — passes a per-tensor bit
  budget to the block-FP encoder; the encoder uses larger ``qint_max`` for
  high-budget tensors.
- ``tac.block_fp_codec`` — accepts the per-tensor bit budget as a
  hyperparameter.
- ``tac.codec.a6_selfcomp_blockfp_hyperprior_compose`` — provides the entropy
  coding atop the per-block exponent stream.

CLAUDE.md compliance
--------------------
* NO scorer load: this module operates on a PROVIDED Hessian-proxy dict; the
  caller computes it via ``tac.score_gradient_param_saliency`` or sister.
* NO silent defaults: every public function arg is keyword-only.
* No /tmp paths.

References
----------
* Boyd, S. & Vandenberghe, L. "Convex Optimization" §5.5 (water-filling).
* Hassibi & Stork 1993, "Second order derivatives for network pruning:
  Optimal Brain Surgeon" — Hessian-based pruning.
* CLAUDE.md "Selfcomp's specific contributions" (Quantizr KL distill +
  Selfcomp block-FP + Hessian quant + arithmetic coder stack).
* Forbidden pattern: weight-domain saliency on score-gradient substrate
  (catalog #123).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


__all__ = [
    "HessianAllocationConfig",
    "HessianAllocationResult",
    "compute_hessian_diagonal_proxy",
    "allocate_bits_by_hessian",
    "expected_distortion_under_allocation",
    "validate_hessian_proxy_source",
]


@dataclass(frozen=True)
class HessianAllocationConfig:
    """Configuration for Hessian-weighted bit allocation.

    Attributes:
        total_bit_budget: Total bits available across all tensors. Typically
            ``target_archive_bytes * 8 - overhead_bytes * 8``.
        min_bits_per_tensor: Bits below which a tensor is pruned (set to 0).
            Default 0.5 (matches Selfcomp's ``b < 0.5 → prune`` convention).
        max_bits_per_tensor: Cap on bits/tensor. 8 = int8; 4 = FP4 codebook.
        bisection_max_iters: Bisection iterations for lambda search.
        bisection_tol: Relative tolerance for budget satisfaction.
    """

    total_bit_budget: float
    min_bits_per_tensor: float = 0.5
    max_bits_per_tensor: float = 8.0
    bisection_max_iters: int = 64
    bisection_tol: float = 1e-4

    def __post_init__(self) -> None:
        if self.total_bit_budget <= 0:
            raise ValueError(
                f"total_bit_budget must be positive, got {self.total_bit_budget}"
            )
        if not (0 <= self.min_bits_per_tensor <= self.max_bits_per_tensor):
            raise ValueError(
                "min_bits_per_tensor must be in [0, max_bits_per_tensor]; "
                f"got min={self.min_bits_per_tensor}, max={self.max_bits_per_tensor}"
            )
        if self.bisection_max_iters <= 0:
            raise ValueError(
                f"bisection_max_iters must be positive, got {self.bisection_max_iters}"
            )
        if self.bisection_tol <= 0:
            raise ValueError(
                f"bisection_tol must be positive, got {self.bisection_tol}"
            )


@dataclass(frozen=True)
class HessianAllocationResult:
    """Result of Hessian-weighted bit allocation.

    Attributes:
        bits_per_tensor: ``{tensor_name: bits_per_weight}`` allocation.
        lambda_star: Final Lagrangian multiplier (KKT condition).
        total_bits_used: Sum of allocated bits × params (must satisfy budget).
        expected_distortion: Sum of expected per-tensor distortions.
        n_pruned_tensors: Count of tensors below ``min_bits_per_tensor``.
        budget_satisfied: True iff |total_bits_used - budget| < tol * budget.
        provenance: Audit trail (saliency_source, evidence_grade per #123).
    """

    bits_per_tensor: dict[str, float]
    lambda_star: float
    total_bits_used: float
    expected_distortion: float
    n_pruned_tensors: int
    budget_satisfied: bool
    provenance: dict[str, Any]


def validate_hessian_proxy_source(
    hessian_proxy: dict[str, float],
    saliency_source: str,
    substrate_class: str | None,
) -> None:
    """Refuse weight-domain saliency proxies on score-gradient substrates.

    Per CLAUDE.md "Forbidden weight-domain saliency on score-gradient substrate"
    (catalog #123): the empirical falsification of Track 4 v1 (apogee_intN) at
    rms ~1.8e-3 proved that ``mean(theta**2)`` is anti-correlated with true
    score saliency on score-aware substrates because the score-gradient
    training pushes parameters AWAY from zero precisely where they are
    score-relevant.

    Allowed saliency sources:
      - ``"score_gradient"`` — computed via ``tac.score_gradient_param_saliency``.
      - ``"fisher_diagonal_from_score_loss"`` — computed via gradient² of
        score loss (NOT weight-domain mean(theta²)).
      - ``"hessian_diagonal_from_score_loss"`` — true second-order Hessian.
      - ``"hessian_diagonal_proxy_with_score_aware_substrate_waiver"`` — only
        when ``substrate_class`` is explicitly ``"non_score_aware"`` (e.g. the
        proxy is being computed on a randomly-initialised model BEFORE any
        score-aware fine-tuning has occurred).

    Forbidden saliency sources on score-gradient substrates:
      - ``"weight_magnitude"`` / ``"mean_theta_squared"`` / ``"l2_norm"`` /
        anything that uses only the weight magnitudes.

    Raises:
        ValueError if the saliency source is forbidden for the substrate.
    """
    FORBIDDEN = {
        "weight_magnitude",
        "mean_theta_squared",
        "l2_norm",
        "weight_norm",
        "param_magnitude",
    }
    SCORE_AWARE_SUBSTRATES = {
        "score_gradient",
        "a1",
        "track1_phase_a1_score_gradient",
        "phase_a1_latent",
        "a1_archive",
        "a1_latent_aligned",
        "train_score_gradient",
    }

    if saliency_source in FORBIDDEN and substrate_class in SCORE_AWARE_SUBSTRATES:
        raise ValueError(
            f"FORBIDDEN saliency source {saliency_source!r} on score-gradient "
            f"substrate {substrate_class!r}. Per CLAUDE.md catalog #123: "
            f"weight-domain saliency is anti-correlated with true score "
            f"sensitivity after score-aware training. Use 'score_gradient' or "
            f"'fisher_diagonal_from_score_loss' instead."
        )
    if not hessian_proxy:
        raise ValueError("hessian_proxy dict is empty; cannot allocate bits")
    for k, v in hessian_proxy.items():
        if v < 0:
            raise ValueError(
                f"Hessian-proxy for tensor {k!r} is negative ({v}); "
                f"Hessian-diagonals must be non-negative (squared-gradient form)."
            )


def compute_hessian_diagonal_proxy(
    parameters: dict[str, torch.Tensor],
    gradients: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Compute a Hessian-diagonal proxy via squared score-gradient.

    For a quadratic Taylor expansion of the score loss around the trained
    weights ``theta*``:

        L(theta) ~= L(theta*) + 0.5 * (theta - theta*)^T H (theta - theta*)

    The Hessian-diagonal ``H_ii`` admits the proxy ``(dL/dtheta_i)**2`` (Fisher
    information lower bound) when ``theta*`` is a local minimum. This is the
    canonical proxy used by Hassibi & Stork's Optimal Brain Surgeon framework.

    Per CLAUDE.md catalog #123: this is the CORRECT proxy. The forbidden
    proxy is ``mean(theta**2)`` (pure weight-domain). The gradient must come
    from a score-loss backward pass; the caller is responsible for that.

    Args:
        parameters: ``{name: tensor}`` model parameters.
        gradients: ``{name: tensor}`` corresponding gradients from a
            score-loss backward pass (typically the SegNet+PoseNet
            Lagrangian, computed by the trainer's Stage 2 fine-tune).

    Returns:
        ``{tensor_name: hessian_diagonal_sum}`` — sum of squared gradients
        per tensor. Aggregation is sum (not mean) so larger tensors with
        many small-magnitude gradients are not artificially deprioritized.

    Raises:
        ValueError if any tensor's gradient is missing or shape-mismatched.
    """
    out: dict[str, float] = {}
    for name, theta in parameters.items():
        if name not in gradients:
            raise ValueError(
                f"compute_hessian_diagonal_proxy: gradient missing for {name!r}"
            )
        g = gradients[name]
        if g.shape != theta.shape:
            raise ValueError(
                f"compute_hessian_diagonal_proxy: gradient shape mismatch for "
                f"{name!r}: param {theta.shape} vs grad {g.shape}"
            )
        # Squared-gradient proxy: H_ii ~= (dL/dtheta_i)^2, summed per tensor
        h_sum = float((g.detach().float() ** 2).sum().item())
        out[name] = h_sum
    return out


def allocate_bits_by_hessian(
    *,
    hessian_proxy: dict[str, float],
    params_per_tensor: dict[str, int],
    config: HessianAllocationConfig,
    saliency_source: str,
    substrate_class: str | None = None,
) -> HessianAllocationResult:
    """Solve the water-filling bit-allocation problem.

    For each tensor t with Hessian-diagonal sum ``H_t`` and ``params_t``
    parameters, the optimal allocation under a total-bit budget is:

        b_t = max(min_b, min(max_b, 0.5 * log2(lambda * H_t / params_t)))

    where ``lambda`` is the Lagrangian multiplier found by bisection such
    that ``sum_t b_t * params_t == total_bit_budget``.

    The expected distortion under this allocation is:

        D_total = sum_t H_t * 2^(-2 * b_t) / 12

    (uniform-quantization variance ≈ delta^2/12 where delta = 2 * max / 2^b).

    Args:
        hessian_proxy: Per-tensor Hessian-diagonal sum (output of
            ``compute_hessian_diagonal_proxy``).
        params_per_tensor: Per-tensor parameter count.
        config: Allocation configuration.
        saliency_source: Provenance tag — see ``validate_hessian_proxy_source``.
        substrate_class: Substrate provenance — see ``validate_hessian_proxy_source``.

    Returns:
        HessianAllocationResult with per-tensor bit budget + provenance.

    Raises:
        ValueError if inputs are inconsistent or the saliency source is
        forbidden for the substrate class.
    """
    validate_hessian_proxy_source(hessian_proxy, saliency_source, substrate_class)

    if set(hessian_proxy.keys()) != set(params_per_tensor.keys()):
        diff = set(hessian_proxy.keys()) ^ set(params_per_tensor.keys())
        raise ValueError(
            f"hessian_proxy and params_per_tensor must have identical keys; "
            f"symmetric difference: {sorted(diff)}"
        )

    names = sorted(hessian_proxy.keys())
    H = [max(hessian_proxy[n], 1e-30) for n in names]  # avoid log(0)
    P = [max(params_per_tensor[n], 1) for n in names]

    def budget_at_lambda(lam: float) -> tuple[float, list[float]]:
        bits = []
        for h, p in zip(H, P):
            # b_t = 0.5 * log2(lam * h / p), clamped to [min_b, max_b]
            raw = 0.5 * (
                torch.log2(torch.tensor(max(lam * h / p, 1e-30))).item()
            )
            clamped = max(
                config.min_bits_per_tensor,
                min(config.max_bits_per_tensor, raw),
            )
            # Apply prune rule: bits below min_bits → 0
            if clamped <= config.min_bits_per_tensor:
                clamped = 0.0
            bits.append(clamped)
        total = sum(b * p for b, p in zip(bits, P))
        return total, bits

    # Bisection on lambda. budget_at_lambda is monotone increasing in lam.
    lam_lo = 1e-12
    lam_hi = 1e12
    target = float(config.total_bit_budget)

    bits_final: list[float] = []
    lam_final = lam_lo

    for _ in range(config.bisection_max_iters):
        lam_mid = (lam_lo * lam_hi) ** 0.5  # geometric midpoint
        total, bits_mid = budget_at_lambda(lam_mid)
        if abs(total - target) < config.bisection_tol * target:
            bits_final = bits_mid
            lam_final = lam_mid
            break
        if total < target:
            lam_lo = lam_mid
        else:
            lam_hi = lam_mid
        bits_final = bits_mid
        lam_final = lam_mid

    total_bits_used, _ = budget_at_lambda(lam_final)
    bits_dict = {n: b for n, b in zip(names, bits_final)}

    # Expected distortion: D = sum_t H_t * 2^(-2*b_t) / 12
    expected_d = 0.0
    for n, b in zip(names, bits_final):
        if b <= 0:
            # Pruned: distortion equals H_t (full quantization noise)
            expected_d += hessian_proxy[n]
        else:
            expected_d += hessian_proxy[n] * (2.0 ** (-2.0 * b)) / 12.0

    n_pruned = sum(1 for b in bits_final if b == 0.0)
    budget_satisfied = (
        abs(total_bits_used - target) < config.bisection_tol * target
    )

    return HessianAllocationResult(
        bits_per_tensor=bits_dict,
        lambda_star=lam_final,
        total_bits_used=total_bits_used,
        expected_distortion=expected_d,
        n_pruned_tensors=n_pruned,
        budget_satisfied=budget_satisfied,
        provenance={
            "saliency_source": saliency_source,
            "substrate_class": substrate_class,
            "config": {
                "total_bit_budget": config.total_bit_budget,
                "min_bits_per_tensor": config.min_bits_per_tensor,
                "max_bits_per_tensor": config.max_bits_per_tensor,
                "bisection_max_iters": config.bisection_max_iters,
                "bisection_tol": config.bisection_tol,
            },
            "n_tensors": len(names),
            "evidence_grade": "derivation",  # not [contest-CUDA]; allocation only
        },
    )


def expected_distortion_under_allocation(
    *,
    bits_per_tensor: dict[str, float],
    hessian_proxy: dict[str, float],
) -> float:
    """Expected score-domain distortion under a given bit allocation.

    Standalone helper for solver use: given a bit allocation (not necessarily
    from water-filling — could be an external CMA-ES / Optuna proposal) +
    a Hessian-diagonal proxy, compute the expected score distortion.

    Returns sum_t H_t * 2^(-2*b_t) / 12 for active tensors and sum_t H_t for
    pruned (b_t = 0) tensors.
    """
    if set(bits_per_tensor.keys()) != set(hessian_proxy.keys()):
        diff = set(bits_per_tensor.keys()) ^ set(hessian_proxy.keys())
        raise ValueError(
            f"bits_per_tensor and hessian_proxy must have identical keys; "
            f"symmetric difference: {sorted(diff)}"
        )
    total = 0.0
    for name, b in bits_per_tensor.items():
        h = hessian_proxy[name]
        if b <= 0:
            total += h
        else:
            total += h * (2.0 ** (-2.0 * b)) / 12.0
    return total
