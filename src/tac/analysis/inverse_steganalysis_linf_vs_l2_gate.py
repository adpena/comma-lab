# SPDX-License-Identifier: MIT
"""Falsifiable $0 proof-of-principle: L-inf margin-budget vs L2 allocation.

This module is the canonical helper for the design memo §7 gate
(``.omx/research/inverse_steganalysis_optimal_full_stack_design_20260601.md``):

    **Does L-inf margin-budget (inverse-steganalysis) allocation BEAT L2
    (MSE-optimal) allocation at EQUAL rate, measured on the real scorer?**

The contest scorer is a steganalysis DETECTOR (Yousfi built SegNet from
EfficientNet steganalysis surgery). The contest objective is therefore the
steganographic security objective -- maximize bytes-saved subject to "the
detector's decision does not flip" -- NOT L2 fidelity. This gate isolates THE
OBJECTIVE (allocation rule) by holding the carrier and rate fixed and varying
only WHERE the quantization precision is placed.

Reduction (the rigorous rate model)
------------------------------------
A lossy carrier quantizes its rendered output with a per-pixel uniform
quantizer of step ``delta_i``. Two first-order facts hold for a uniform
quantizer of dynamic range ``R`` (here ``R = 256`` for uint8 RGB):

  * **Distortion** -- the quantization error is uniform on
    ``[-delta_i/2, +delta_i/2]`` so ``|perturbation_i| <= delta_i/2`` (the
    L-inf per-pixel bound) and ``E[perturbation_i^2] = delta_i^2/12`` (the L2
    per-pixel distortion).
  * **Rate** -- the entropy of a uniform-quantized source is approximately
    ``rate_i = log2(R / delta_i)`` bits/pixel (the high-rate approximation;
    Gersho-Gray). Total rate ``B = sum_i rate_i = sum_i log2(R/delta_i)``.

At a **fixed total bit budget B** we choose the steps ``delta_i``:

  * **(A) L2-optimal** -- minimize total MSE ``sum_i delta_i^2/12`` subject to
    ``sum_i log2(R/delta_i) = B``. The Lagrangian stationarity
    ``d/d(delta_i)[delta_i^2/12 - mu*log2(R/delta_i)] = 0`` gives
    ``delta_i^2 = const`` -- i.e. the MSE-optimal allocation under a flat
    (equal-variance) source model is a **UNIFORM step**. Every pixel gets the
    same precision. This is the standard codec objective and the honest L2
    baseline.

  * **(B) L-inf margin-budget (inverse-steganalysis)** -- the contest metric is
    an argmax-flip rate, not MSE, so the correct per-pixel constraint is
    ``|perturbation_i| <= rho_i`` (no pixel exceeds its DeepFool margin /
    Fisher budget). We allocate the SAME total bit budget B but distribute the
    precision by the oracle ``rho_i``: spend fine steps (high local rate) where
    ``rho_i`` is small (decision boundaries; detector-sensitive), and coarse
    steps (the dead-zone; low local rate) where ``rho_i`` is large (the
    detector-blind ~90% of pixels). Concretely the per-pixel step is allocated
    by a reverse-water-fill ``delta_i = clip(c * rho_i, delta_min, delta_max)``
    whose water level ``c`` is solved so that ``sum_i log2(R/delta_i) == B``
    to the requested tolerance.

Both allocations spend the **same total bits B** (the fairness invariant). The
ONLY difference is where the precision goes. We then DECODE (add the
quantization noise at the chosen per-pixel steps) and run the **verified
differentiable scorer mirror** (bit-exact vs the frozen weights, commit
8173b493a) on the SAME real ``upstream/videos/0.mkv`` frames, and measure the
hard contest ``d_seg`` (last-frame argmax-flip rate) + ``d_pose`` (first-6 pose
MSE). The verdict is: at equal rate, does aiming bits by the detector's margin
reduce ``d_seg + d_pose`` more than aiming by L2?

CONTEST COMPLIANCE / authority
------------------------------
This is a COMPRESS-SIDE proof-of-principle. It loads frozen upstream scorers for
OFFLINE allocation analysis only; nothing crosses the receiver boundary. All
numerics are ``[macOS-CPU advisory]`` -- NON-PROMOTABLE, ``score_claim=false``,
``promotable=false`` per Catalog #341/#192/#127/#323. The quantization noise is a
deterministic surrogate for a uniform-quantizer carrier; the proof isolates the
OBJECTIVE (allocation rule) on an existing carrier surface, it does not build or
train a new substrate and makes no score claim.

References
----------
- Moosavi-Dezfooli, Fawzi, Frossard 2016 "DeepFool" (per-pixel margin ``rho_i``).
- Holub, Fridrich, Denemark 2014 "UNIWARD" (cost ~ 1/local-complexity).
- Ker, Pevny, Fridrich 2008 "square-root law" (spread small errors; L-inf budget).
- Gersho, Gray 1992 "Vector Quantization and Signal Compression" (high-rate
  uniform-quantizer entropy ``log2(R/delta)``).
- tac.analysis.score_exact_saliency (the verified P18/P19 oracle producer).
- tac.contest_eval_contract.build_saliency_verification_contract (the proof gates).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

UINT8_DYNAMIC_RANGE = 256.0
GATE_SCHEMA = "inverse_steganalysis_linf_vs_l2_gate.v1"
RATE_TOLERANCE_DEFAULT = 1.0e-3  # <0.1% rate match per the NO-FAKE fairness guard


class InverseSteganalysisGateError(ValueError):
    """Raised when the gate's fairness invariants cannot be satisfied."""


# ---------------------------------------------------------------------------
# Rate model (the high-rate uniform-quantizer entropy approximation).
# ---------------------------------------------------------------------------


def _total_bits_from_steps(
    steps: np.ndarray, *, dynamic_range: float = UINT8_DYNAMIC_RANGE
) -> float:
    """Total rate (bits) of a per-pixel uniform quantizer with steps ``steps``.

    Per-pixel rate ``= log2(R/delta_i)`` clamped at >= 0 (a step larger than the
    range carries no bits). Total ``B = sum_i max(0, log2(R/delta_i))``.
    """
    steps = np.asarray(steps, dtype=np.float64)
    if np.any(steps <= 0.0):
        raise InverseSteganalysisGateError("quantizer steps must be strictly positive")
    per_pixel = np.log2(float(dynamic_range) / steps)
    return float(np.clip(per_pixel, 0.0, None).sum())


def _uniform_step_for_budget(
    n_pixels: int,
    *,
    target_bits: float,
    dynamic_range: float = UINT8_DYNAMIC_RANGE,
) -> float:
    """The single uniform step ``delta`` whose total rate equals ``target_bits``.

    Solve ``n * log2(R/delta) = B`` -> ``delta = R * 2^(-B/n)``. This is the
    L2-optimal (MSE-minimizing under a flat source) allocation: every pixel gets
    the same precision.
    """
    if n_pixels <= 0:
        raise InverseSteganalysisGateError("n_pixels must be > 0")
    if target_bits < 0.0:
        raise InverseSteganalysisGateError("target_bits must be >= 0")
    per_pixel_bits = float(target_bits) / float(n_pixels)
    return float(dynamic_range) * (2.0 ** (-per_pixel_bits))


@dataclass(frozen=True)
class StepAllocation:
    """A per-pixel quantizer step map and its realized rate."""

    steps: np.ndarray  # (N,) per-pixel uniform-quantizer step delta_i
    total_bits: float  # realized sum_i log2(R/delta_i)
    allocation: str  # "l2_uniform" | "linf_margin_budget"
    water_level: float  # the reverse-waterfill constant c (l2: 0.0)
    min_step: float
    max_step: float


def allocate_l2_uniform(
    n_pixels: int,
    *,
    target_bits: float,
    dynamic_range: float = UINT8_DYNAMIC_RANGE,
) -> StepAllocation:
    """L2-optimal allocation: a single uniform step for every pixel.

    This is the MSE-minimizing allocation under a flat (equal-variance) source
    model -- the standard codec objective. The realized rate is exactly
    ``target_bits`` (the uniform step is solved in closed form).
    """
    delta = _uniform_step_for_budget(
        n_pixels, target_bits=target_bits, dynamic_range=dynamic_range
    )
    steps = np.full(int(n_pixels), delta, dtype=np.float64)
    return StepAllocation(
        steps=steps,
        total_bits=_total_bits_from_steps(steps, dynamic_range=dynamic_range),
        allocation="l2_uniform",
        water_level=0.0,
        min_step=delta,
        max_step=delta,
    )


def allocate_linf_margin_budget(
    rho: np.ndarray,
    *,
    target_bits: float,
    dynamic_range: float = UINT8_DYNAMIC_RANGE,
    min_step: float = 1.0,
    max_step: float | None = None,
    rate_tolerance: float = RATE_TOLERANCE_DEFAULT,
    max_bisection_iters: int = 200,
    fairness_direction: str = "disadvantage_linf",
) -> StepAllocation:
    """L-inf margin-budget allocation: ``delta_i = clip(c * rho_i, lo, hi)``.

    The per-pixel STEP is proportional to the oracle margin ``rho_i`` (large
    ``rho_i`` = detector-blind = coarse step = low local rate; small ``rho_i`` =
    boundary = fine step = high local rate). The water level ``c`` is solved by
    bisection so the realized total rate matches ``target_bits`` to within
    ``rate_tolerance`` (relative). This is the inverse-steganalysis allocation:
    the SAME total bits as L2, distributed by the detector's margin instead of
    uniformly.

    ``fairness_direction`` controls which side of the tolerance band the realized
    rate is allowed to land on (the NO-FAKE anti-gaming guard):

      * ``"disadvantage_linf"`` (default) -- the realized L-inf rate must be
        ``>= target_bits`` (within tolerance). The L-inf allocation is FORCED to
        spend AT LEAST as many bits as L2, so a measured L-inf win can NEVER be
        attributed to L-inf secretly spending fewer bits.
      * ``"symmetric"`` -- either side within tolerance (loosest).
      * ``"advantage_linf"`` -- realized rate ``<= target_bits`` (only for the
        adversarial control that proves the win survives an L-inf rate handicap
        in the OTHER direction).

    ``rho`` is the per-pixel oracle margin budget. We use the SALIENCY directly:
    a high-saliency pixel must be perturbed LESS (fine step), so we map
    ``rho_i = 1 / (saliency_i + eps)`` upstream; this function takes the already
    inverse-mapped per-pixel margin budget where LARGER ``rho_i`` => safe to
    coarsen. The mapping is performed by ``margin_budget_from_saliency`` so the
    inversion is auditable and tested separately.
    """
    rho = np.asarray(rho, dtype=np.float64).reshape(-1)
    n = int(rho.size)
    if n == 0:
        raise InverseSteganalysisGateError("rho must be non-empty")
    if np.any(~np.isfinite(rho)) or np.any(rho <= 0.0):
        raise InverseSteganalysisGateError("rho must be finite and strictly positive")
    if min_step <= 0.0:
        raise InverseSteganalysisGateError("min_step must be > 0")
    hi = float(dynamic_range) if max_step is None else float(max_step)
    if hi <= min_step:
        raise InverseSteganalysisGateError("max_step must exceed min_step")
    if target_bits < 0.0:
        raise InverseSteganalysisGateError("target_bits must be >= 0")
    if fairness_direction not in {"disadvantage_linf", "symmetric", "advantage_linf"}:
        raise InverseSteganalysisGateError(
            "fairness_direction must be 'disadvantage_linf', 'symmetric', or 'advantage_linf'"
        )

    def bits_for_level(c: float) -> tuple[float, np.ndarray]:
        steps = np.clip(c * rho, min_step, hi)
        return _total_bits_from_steps(steps, dynamic_range=dynamic_range), steps

    # Bisect the water level c. Larger c => coarser steps => fewer bits (rate is
    # monotonically DECREASING in c). Bracket: c_lo gives >= target bits (fine),
    # c_hi gives <= target bits (coarse).
    c_lo = min_step / float(np.max(rho))  # steps all >= min_step => max bits
    c_hi = hi / float(np.min(rho))  # steps all <= hi => min bits
    bits_lo, _ = bits_for_level(c_lo)
    if bits_lo < target_bits:
        # Even the finest feasible allocation cannot reach target_bits.
        steps = np.clip(c_lo * rho, min_step, hi)
        realized = _total_bits_from_steps(steps, dynamic_range=dynamic_range)
        raise InverseSteganalysisGateError(
            f"target_bits={target_bits:.3f} exceeds the finest feasible rate "
            f"{realized:.3f} (min_step={min_step} caps precision)"
        )
    tol = rate_tolerance * max(target_bits, 1.0)
    c = c_hi
    steps = np.clip(c * rho, min_step, hi)
    for _ in range(int(max_bisection_iters)):
        c = 0.5 * (c_lo + c_hi)
        bits, steps = bits_for_level(c)
        if abs(bits - target_bits) <= tol:
            break
        if bits > target_bits:  # too many bits => steps too fine => raise c
            c_lo = c
        else:
            c_hi = c
    realized = _total_bits_from_steps(steps, dynamic_range=dynamic_range)
    # Anti-gaming fairness guard: never let L-inf land on the cheaper side of the
    # band unless explicitly testing the advantage_linf control. After the main
    # bisection, ``c_lo`` brackets the >=target side and ``c_hi`` the <=target
    # side. We bisect that SAME bracket to find the c whose rate is MINIMALLY
    # above (disadvantage) / below (advantage) target -- never a coarse jump.
    if fairness_direction == "disadvantage_linf" and realized < target_bits:
        lo, hi_c = c_lo, c  # lo gives >=target bits, c gives <target bits
        c_best, steps_best, bits_best = c_lo, np.clip(c_lo * rho, min_step, hi), bits_lo
        for _ in range(int(max_bisection_iters)):
            mid = 0.5 * (lo + hi_c)
            bits, steps_n = bits_for_level(mid)
            if bits >= target_bits:
                # feasible (>=target) and as tight as possible: tighten from above
                c_best, steps_best, bits_best = mid, steps_n, bits
                lo = mid
            else:
                hi_c = mid
            if abs(hi_c - lo) <= 1e-12 * max(abs(hi_c), 1.0):
                break
        c, steps, realized = c_best, steps_best, bits_best
    elif fairness_direction == "advantage_linf" and realized > target_bits:
        lo, hi_c = c, c_hi  # c gives >target bits, c_hi gives <=target bits
        c_best, steps_best, bits_best = c_hi, np.clip(c_hi * rho, min_step, hi), bits_for_level(c_hi)[0]
        for _ in range(int(max_bisection_iters)):
            mid = 0.5 * (lo + hi_c)
            bits, steps_n = bits_for_level(mid)
            if bits <= target_bits:
                c_best, steps_best, bits_best = mid, steps_n, bits
                hi_c = mid
            else:
                lo = mid
            if abs(hi_c - lo) <= 1e-12 * max(abs(hi_c), 1.0):
                break
        c, steps, realized = c_best, steps_best, bits_best
    return StepAllocation(
        steps=steps,
        total_bits=realized,
        allocation="linf_margin_budget",
        water_level=float(c),
        min_step=float(steps.min()),
        max_step=float(steps.max()),
    )


def margin_budget_from_saliency(
    saliency: np.ndarray, *, eps: float = 1.0e-9
) -> np.ndarray:
    """Map oracle saliency -> per-pixel margin budget ``rho_i = 1/(s_i + eps)``.

    A HIGH-saliency pixel (small DeepFool margin, large Fisher) must be perturbed
    LESS -> small ``rho_i`` -> fine step. A LOW-saliency (detector-blind) pixel
    can be coarsened -> large ``rho_i`` -> coarse step. This single inversion is
    the entire inverse-steganalysis prior; it is auditable and tested separately
    so the gate's allocator can be reasoned about without the inversion baked in.
    """
    s = np.asarray(saliency, dtype=np.float64).reshape(-1)
    s = np.clip(s, 0.0, None)
    return 1.0 / (s + float(eps))


# ---------------------------------------------------------------------------
# Quantization-noise application (the "decode" step).
# ---------------------------------------------------------------------------


def apply_uniform_quantization_noise(
    frame: torch.Tensor,
    steps_per_pixel: np.ndarray | torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    clamp_range: tuple[float, float] = (0.0, 255.0),
) -> torch.Tensor:
    """Add deterministic per-pixel uniform quantization noise to ``frame``.

    A uniform quantizer of step ``delta_i`` introduces error uniform on
    ``[-delta_i/2, +delta_i/2]``. We realize that error directly (a faithful
    surrogate for the carrier's quantization) so the rate model and the applied
    distortion are the SAME object. ``steps_per_pixel`` is broadcast over the
    channel dimension (per-pixel step shared across RGB, matching the scorer's
    spatial argmax/pose response). Returns a clamped float tensor.
    """
    if isinstance(steps_per_pixel, np.ndarray):
        steps_t = torch.from_numpy(np.asarray(steps_per_pixel, dtype=np.float64))
    else:
        steps_t = steps_per_pixel.double()
    steps_t = steps_t.to(frame.device)
    # frame: (..., C, H, W); steps: (H, W) or (H*W,) -> (H, W)
    h, w = frame.shape[-2:]
    if steps_t.numel() != h * w:
        raise InverseSteganalysisGateError(
            f"steps_per_pixel has {steps_t.numel()} entries, expected H*W={h * w}"
        )
    steps_hw = steps_t.reshape(h, w)
    u = torch.rand(frame.shape, generator=generator, dtype=torch.float64, device=frame.device)
    noise = (u - 0.5) * steps_hw  # broadcast over leading dims incl channel
    noisy = frame.double() + noise
    lo, hi = clamp_range
    return noisy.clamp(lo, hi).to(frame.dtype)


# ---------------------------------------------------------------------------
# The real-scorer measurement (d_seg argmax-flip + d_pose first-6 MSE).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairScoreMeasurement:
    """Hard contest d_seg/d_pose for one perturbed candidate pair."""

    d_seg: float  # last-frame argmax-flip rate vs gt
    d_pose: float  # first-6 pose MSE vs gt
    allocation: str


def measure_pair_d_seg_d_pose(
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    gt_pair_btchw: torch.Tensor,
    candidate_pair_btchw: torch.Tensor,
) -> tuple[float, float]:
    """Hard contest d_seg + d_pose between a gt pair and a candidate pair.

    ``d_seg = mean(argmax(seg(gt_last)) != argmax(seg(cand_last)))`` exactly per
    ``upstream/modules.py::SegNet.compute_distortion``. ``d_pose = MSE`` over the
    first 6 of the 12 pose-head dims per ``PoseNet.compute_distortion``. Both run
    on the verified differentiable mirror under ``no_grad`` (no gradients are
    needed here; we only need the forward argmax + pose).
    """
    with torch.no_grad():
        seg_in_gt = segnet.preprocess_input(gt_pair_btchw)
        seg_in_cand = segnet.preprocess_input(candidate_pair_btchw)
        gt_arg = segnet(seg_in_gt).argmax(dim=1)
        cand_arg = segnet(seg_in_cand).argmax(dim=1)
        d_seg = float((gt_arg != cand_arg).float().mean().item())

        pose_in_gt = posenet.preprocess_input(gt_pair_btchw)
        pose_in_cand = posenet.preprocess_input(candidate_pair_btchw)
        pose_gt = posenet(pose_in_gt)["pose"][0, :6]
        pose_cand = posenet(pose_in_cand)["pose"][0, :6]
        d_pose = float(((pose_cand - pose_gt) ** 2).mean().item())
    return d_seg, d_pose


def measure_pairs_d_seg_d_pose_batched(
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    gt_pairs_btchw: torch.Tensor,
    candidate_pairs_btchw: torch.Tensor,
    *,
    batch_pairs: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Hard contest d_seg/d_pose for many pairs with exact chunked reduction.

    This is mathematically the same measurement as ``measure_pair_d_seg_d_pose``
    applied independently to each pair. The difference is execution shape: model
    forwards and scalar transfers are chunked, so full600 advisory/replay paths do
    not pay one Torch/MPS synchronization per pair and component.
    """

    if gt_pairs_btchw.shape != candidate_pairs_btchw.shape:
        raise InverseSteganalysisGateError(
            "gt_pairs_btchw and candidate_pairs_btchw must have the same shape"
        )
    if gt_pairs_btchw.ndim != 5:
        raise InverseSteganalysisGateError(
            "expected tensors shaped (pairs, 2, 3, H, W)"
        )
    n_pairs = int(gt_pairs_btchw.shape[0])
    if n_pairs == 0:
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)
    chunk = max(1, int(batch_pairs))
    d_seg_chunks: list[np.ndarray] = []
    d_pose_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, n_pairs, chunk):
            stop = min(n_pairs, start + chunk)
            gt = gt_pairs_btchw[start:stop]
            cand = candidate_pairs_btchw[start:stop]

            seg_in_gt = segnet.preprocess_input(gt)
            seg_in_cand = segnet.preprocess_input(cand)
            gt_arg = segnet(seg_in_gt).argmax(dim=1)
            cand_arg = segnet(seg_in_cand).argmax(dim=1)
            d_seg_chunk = (gt_arg != cand_arg).float().mean(dim=(1, 2))

            pose_in_gt = posenet.preprocess_input(gt)
            pose_in_cand = posenet.preprocess_input(cand)
            pose_gt = posenet(pose_in_gt)["pose"][:, :6]
            pose_cand = posenet(pose_in_cand)["pose"][:, :6]
            d_pose_chunk = ((pose_cand - pose_gt) ** 2).mean(dim=1)

            d_seg_chunks.append(d_seg_chunk.detach().cpu().numpy().astype(np.float64))
            d_pose_chunks.append(d_pose_chunk.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(d_seg_chunks), np.concatenate(d_pose_chunks)


__all__ = [
    "GATE_SCHEMA",
    "RATE_TOLERANCE_DEFAULT",
    "UINT8_DYNAMIC_RANGE",
    "InverseSteganalysisGateError",
    "PairScoreMeasurement",
    "StepAllocation",
    "allocate_l2_uniform",
    "allocate_linf_margin_budget",
    "apply_uniform_quantization_noise",
    "margin_budget_from_saliency",
    "measure_pair_d_seg_d_pose",
    "measure_pairs_d_seg_d_pose_batched",
]
