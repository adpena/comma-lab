# SPDX-License-Identifier: MIT
"""Frontier exact precision-allocation SOLVE — the deterministic reverse-water-fill.

The operator correction (2026-06-19): the int5 "confirmed structural" cap was un-nuanced
— it held our own codec fixed and used a UNIFORM low-nbits grid + per-tensor LSQ. We
possess the frozen SegNet+PoseNet, the frozen 600 pairs, and the closed-form eval, so the
optimal per-tensor precision allocation is a DETERMINISTIC OPTIMIZATION we can SOLVE, not a
feasibility we probe. This module is that solve (design memo:
``.omx/research/frontier_rate_exact_bitalloc_solve_design_20260618.md``).

The three deterministic steps:

1. **Exact per-tensor score-sensitivity** (one backward each, analytic, no training):
   * ``g_seg(W) = ‖∂(Σ SegNet top1−top2 margin)/∂W‖`` — REUSED from
     :func:`tac.margin_saliency_map.compute_decoder_tensor_margin_saliency` (the exact
     frozen-SegNet margin gradient; argmax has 0 gradient so the margin is the a.e.-diff
     surrogate of the argmax-flip d_seg).
   * ``g_pose(W) = ‖∂d_pose/∂W‖`` — the d_pose analogue this module ADDS
     (:func:`compute_decoder_tensor_pose_saliency`), same accumulate-grad-norm structure
     through the frozen PoseNet (exact, MSE-smooth).
   * combined by the master gradient at the operating point (``∂S/∂d_seg = 100``,
     ``∂S/∂d_pose = 5/√(10·d_pose) ≈ 85.8`` at d_pose≈3.4e-5):
     ``s(W) = w_seg·g_seg(W) + w_pose·g_pose(W)``.

2. **Reverse-water-filling / KKT bit allocation** (closed form,
   :func:`waterfill_bit_allocation`): minimize total stored bits subject to a
   score-distortion budget. The per-tensor int-N round-trip L2 error scales as
   ``absmax_t·√numel_t·2^{−b}``; the score-impact coefficient is
   ``c_t = s(t)·absmax_t·√numel_t``. The Lagrangian optimum is
   ``b_t* = log₂(c_t/numel_t) + log₂(λ·ln2)`` — the optimal bit-width rises with the
   per-WEIGHT score-impact ``log₂(s(t)·absmax_t/√numel_t)``, so a large blind tensor
   correctly gets FEW bits/weight. One scalar λ tunes the rate/distortion operating point.

3. **Per-tensor non-uniform requant** (:func:`requant_state_dict_per_tensor_nbits`): each
   weight tensor is quantize→dequantized at its OWN allocated nbits via the canonical
   ``intn_qdq``. The qdq output is a per-tensor symmetric grid representable in the codec's
   existing per-tensor int8 + fp16-scale grammar (a lower-nbits tensor maps to fewer
   distinct int8 symbols → better brotli = the rate win), so the byte-close re-uses the
   proven ``reencode_frontier_archive`` path unchanged — NO per-channel scales (the
   byte-blowup the prior cap hit), NO decoder change for the measurement.

NO-FAKE: the sensitivity is the REAL frozen-scorer autograd gradient; the allocation
ACTUALLY emits non-uniform per-tensor nbits; the requant ACTUALLY snaps each tensor to its
allocated grid. The only authoritative S is the byte-closed archive through
``RealScorerContext.exact_eval`` (CPU, never MPS). This module computes the allocation +
the requant; the harness ``experiments/frontier_exact_bitalloc_solve.py`` byte-closes +
evals. AUTHORITY: ``[contest-CPU advisory]`` NON-PROMOTABLE; pointer UNMOVED 0.19110.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from tac.margin_saliency_map import (
    MarginSaliencyError,
    _stage_for_tensor_name,
    compute_decoder_tensor_margin_saliency,
)
from tac.post_hoc_weight_shrink import intn_qdq

# Master-gradient weights at the frontier operating point (CLAUDE.md "SegNet vs PoseNet
# importance — operating-point dependent"): ∂S/∂d_seg = 100 (constant); ∂S/∂d_pose =
# 5/√(10·d_pose). At the frontier d_pose ≈ 3.4e-5 → 5/√(3.4e-4) ≈ 271; the int8-local
# d_pose ≈ 3e-5 → ≈ 289. We default to the SUB-0.15 design point d_pose≈3.4e-5 → 85.8 is
# WRONG; the correct value at 3.4e-5 is ~271. We expose w_pose so the caller sets the
# operating point explicitly (the allocation's pose-vs-seg balance depends on it).
DEFAULT_W_SEG = 100.0
DEFAULT_W_POSE = 271.0  # 5/sqrt(10 * 3.4e-5); the frontier operating point


class FrontierBitallocError(ValueError):
    """Raised when the bit-allocation solve is malformed."""


# ---------------------------------------------------------------------------
# Step 1b — per-tensor d_pose sensitivity (the d_seg analogue this module adds)
# ---------------------------------------------------------------------------


@dataclass
class DecoderPoseSaliency:
    """Per-decoder-tensor d_pose-sensitivity (the pose half of the bit-alloc prior).

    Attributes
    ----------
    per_tensor : dict[str, float]
        Tensor name -> ``‖∂(Σ pose-MSE)/∂W_t‖`` accumulated over the scored frames.
    n_frames : int
        How many frames were scored.
    tensor_numel : dict[str, int]
        Tensor name -> element count.
    """

    per_tensor: dict[str, float]
    n_frames: int
    tensor_numel: dict[str, int]


def compute_decoder_tensor_pose_saliency(
    decoder,
    latents: torch.Tensor,
    posenet_forward,
    pose_targets: torch.Tensor,
    *,
    render_pair_bchw,
    frame_indices=None,
) -> DecoderPoseSaliency:
    """Per-decoder-tensor ``‖∂d_pose/∂W_t‖`` through a frozen PoseNet on a REAL decoder.

    The pose analogue of :func:`compute_decoder_tensor_margin_saliency`: it backpropagates
    the per-pair pose MSE (the d_pose distortion) through the decoder to EACH decoder weight
    tensor, accumulating the gradient L2-norm per tensor over a handful of scored frames.

    Parameters
    ----------
    decoder : torch.nn.Module
        The frontier decoder (weights loaded). Its params get ``requires_grad_(True)``.
    latents : torch.Tensor
        Decoder latent rows (N, latent_dim).
    posenet_forward : callable
        ``posenet_forward(decoded_bhwc) -> pose6 (B, 6)`` running the frozen PoseNet on the
        eval-roundtripped decoder output (the contest pose path). Vehicle-agnostic.
    pose_targets : torch.Tensor
        The frozen-PoseNet GT pose targets (N, 6) — the d_pose reference.
    render_pair_bchw : callable
        ``render_pair_bchw(decoder, latent_rows) -> decoded_bhwc (B, 2, H, W, 3)`` in
        [0,255] WITH the autograd graph intact (eval-roundtrip applied inside). The same
        roundtripped tensor the scorer consumes.
    frame_indices : iterable[int] | None
        Which latent rows to score. ``None`` -> a deterministic spread of up to 8 rows.

    Returns
    -------
    DecoderPoseSaliency
        The per-tensor d_pose-sensitivity ranking.
    """
    if latents.dim() != 2:
        raise MarginSaliencyError(
            f"compute_decoder_tensor_pose_saliency: latents must be (N, D), got {tuple(latents.shape)}"
        )
    n = int(latents.shape[0])
    if frame_indices is None:
        k = min(8, n)
        frame_indices = (
            [round(i * (n - 1) / max(1, k - 1)) for i in range(k)] if k > 1 else [0]
        )
    frame_indices = [int(i) for i in frame_indices]
    for i in frame_indices:
        if not (0 <= i < n):
            raise MarginSaliencyError(f"frame index {i} out of range for {n} latent rows")

    params = [(name, p) for name, p in decoder.named_parameters()]
    for _name, p in params:
        p.requires_grad_(True)
    names = [name for name, _p in params]
    numel = {name: int(p.numel()) for name, p in params}

    n_tgt = int(pose_targets.shape[0])
    accum = dict.fromkeys(names, 0.0)
    n_scored = 0
    for i in frame_indices:
        if i >= n_tgt:
            # No GT pose target for this frame (capped targets) — skip so we never
            # backprop an empty-target MSE (the warning the smoke surfaced).
            continue
        decoder.zero_grad(set_to_none=True)
        latent_rows = latents[i : i + 1]  # (1, D)
        decoded_bhwc = render_pair_bchw(decoder, latent_rows)  # (1, 2, H, W, 3) WITH grad
        pose6 = posenet_forward(decoded_bhwc)  # (1, 6) WITH grad
        tgt = pose_targets[i : i + 1].to(pose6.device)
        scalar = F.mse_loss(pose6, tgt)
        n_scored += 1
        grads = torch.autograd.grad(
            scalar, [p for _n, p in params], retain_graph=False, allow_unused=True
        )
        for (name, _p), g in zip(params, grads, strict=True):
            if g is not None:
                accum[name] += float(g.detach().norm().item())

    nfr = max(1, n_scored)
    per_tensor = {name: accum[name] / nfr for name in names}
    return DecoderPoseSaliency(
        per_tensor=per_tensor, n_frames=n_scored, tensor_numel=numel
    )


# ---------------------------------------------------------------------------
# Step 1c — combine the two sensitivities by the master gradient
# ---------------------------------------------------------------------------


@dataclass
class CombinedTensorSensitivity:
    """The per-weight-tensor combined score-sensitivity (the water-fill input).

    Attributes
    ----------
    per_tensor : dict[str, float]
        ``s(W) = w_seg·g_seg(W) + w_pose·g_pose(W)`` per weight tensor (dim≥2 only —
        biases/1-D params stay fp32 and are excluded).
    g_seg : dict[str, float]
        The raw d_seg gradient-norm per tensor.
    g_pose : dict[str, float]
        The raw d_pose gradient-norm per tensor.
    absmax : dict[str, float]
        ``|W|.max()`` per tensor (the quant-step scale).
    numel : dict[str, int]
        Element count per tensor.
    w_seg : float
    w_pose : float
    """

    per_tensor: dict[str, float]
    g_seg: dict[str, float]
    g_pose: dict[str, float]
    absmax: dict[str, float]
    numel: dict[str, int]
    w_seg: float
    w_pose: float


def combine_sensitivities(
    decoder,
    seg_saliency,
    pose_saliency: DecoderPoseSaliency,
    *,
    w_seg: float = DEFAULT_W_SEG,
    w_pose: float = DEFAULT_W_POSE,
    weight_min_dim: int = 2,
    normalize: bool = True,
) -> CombinedTensorSensitivity:
    """Combine the d_seg + d_pose per-tensor sensitivities by the master gradient.

    Only weight tensors with ``dim >= weight_min_dim`` are included (biases / 1-D params
    stay fp32 in the codec, so they carry no allocation).

    ``normalize`` (default True — the PRINCIPLED combine): the raw ``g_seg`` and ``g_pose``
    are gradients of DIFFERENT-SCALED scalars (a sum over ~196k pixel margins vs a single
    pose MSE), so their raw norms differ by ~9 orders of magnitude — adding them lets d_seg
    swamp d_pose and STARVES the pose-critical tensors (measured: the raw combine gives
    ``rgb_1`` — frame-1 head, pose-critical but d_seg-blind — int2 → d_pose explodes). To
    put the two axes on a comparable footing, each gradient field is L1-normalized to a
    relative-importance distribution summing to 1, THEN weighted by the master gradient:
    ``s(t) = w_seg·ĝ_seg(t) + w_pose·ĝ_pose(t)`` with ``ĝ = g/Σg``. Now ``w_seg/w_pose``
    (the master gradient ∂S/∂d_metric) directly controls the seg-vs-pose balance, and a
    tensor the POSE path needs (rgb_1) gets protected even though SegNet is blind to it.

    ``normalize=False`` is the raw-magnitude combine (kept for the diagnostic that surfaced
    the starvation; NOT recommended for the allocation).

    The combined sensitivity ``s`` is the score-distortion gradient w.r.t. the tensor's
    weights — the input to the water-fill.
    """
    seg_pt = dict(seg_saliency.per_tensor)
    pose_pt = dict(pose_saliency.per_tensor)
    # L1 normalizers per axis (over the weight tensors only; biases excluded below).
    wnames = [
        name for name, p in decoder.named_parameters() if p.dim() >= weight_min_dim
    ]
    seg_sum = sum(float(seg_pt.get(n, 0.0)) for n in wnames)
    pose_sum = sum(float(pose_pt.get(n, 0.0)) for n in wnames)
    seg_norm = seg_sum if (normalize and seg_sum > 0) else 1.0
    pose_norm = pose_sum if (normalize and pose_sum > 0) else 1.0
    per_tensor: dict[str, float] = {}
    g_seg: dict[str, float] = {}
    g_pose: dict[str, float] = {}
    absmax: dict[str, float] = {}
    numel: dict[str, int] = {}
    for name, p in decoder.named_parameters():
        if p.dim() < weight_min_dim:
            continue
        gs = float(seg_pt.get(name, 0.0))
        gp = float(pose_pt.get(name, 0.0))
        g_seg[name] = gs
        g_pose[name] = gp
        per_tensor[name] = w_seg * (gs / seg_norm) + w_pose * (gp / pose_norm)
        absmax[name] = float(p.detach().abs().max().item())
        numel[name] = int(p.numel())
    return CombinedTensorSensitivity(
        per_tensor=per_tensor,
        g_seg=g_seg,
        g_pose=g_pose,
        absmax=absmax,
        numel=numel,
        w_seg=float(w_seg),
        w_pose=float(w_pose),
    )


# ---------------------------------------------------------------------------
# Step 2 — reverse-water-filling / KKT bit allocation (closed form)
# ---------------------------------------------------------------------------


@dataclass
class BitAllocation:
    """The solved per-tensor bit allocation.

    Attributes
    ----------
    nbits : dict[str, int]
        Tensor name -> allocated integer bit-width (clamped to [b_min, b_max]).
    impact_coeff : dict[str, float]
        ``c_t = s(t)·absmax_t·√numel_t`` per tensor (the score-distortion per ``2^{−b}``).
    per_weight_impact : dict[str, float]
        ``c_t / numel_t = s(t)·absmax_t/√numel_t`` (drives the bit-width; the per-WEIGHT
        score-impact so a large blind tensor gets few bits/weight).
    lam : float
        The Lagrange multiplier (the operating-point knob) used.
    b_min : int
    b_max : int
    total_weight_bits : int
        ``Σ nbits_t·numel_t`` — the rate proxy (total stored int bits).
    continuous_bits : dict[str, float]
        The pre-rounding continuous ``b_t*`` (for observability).
    """

    nbits: dict[str, int]
    impact_coeff: dict[str, float]
    per_weight_impact: dict[str, float]
    lam: float
    b_min: int
    b_max: int
    total_weight_bits: int
    continuous_bits: dict[str, float]


def waterfill_bit_allocation(
    sens: CombinedTensorSensitivity,
    lam: float,
    *,
    b_min: int = 2,
    b_max: int = 8,
) -> BitAllocation:
    """Closed-form reverse-water-fill per-tensor bit allocation for one λ.

    The KKT optimum of ``min Σ numel_t·b_t  s.t.  Σ c_t·2^{−b_t} ≤ D`` (separable convex)
    is, per tensor, ``b_t* = log₂(λ·ln2·c_t/numel_t)`` with
    ``c_t = s(t)·absmax_t·√numel_t``. A larger λ → more bits everywhere (lower distortion,
    more bytes); λ traces the RD curve. Integer-rounded + clamped to ``[b_min, b_max]``.

    A tensor with zero sensitivity OR zero absmax gets ``b_min`` (it carries no score signal
    → coarsest grid).
    """
    if lam <= 0:
        raise FrontierBitallocError(f"lam must be > 0; got {lam}")
    if not (2 <= b_min <= b_max <= 8):
        raise FrontierBitallocError(
            f"require 2 <= b_min <= b_max <= 8; got b_min={b_min} b_max={b_max}"
        )
    nbits: dict[str, int] = {}
    impact: dict[str, float] = {}
    per_w: dict[str, float] = {}
    cont: dict[str, float] = {}
    ln2 = math.log(2.0)
    for name in sens.per_tensor:
        s = float(sens.per_tensor[name])
        am = float(sens.absmax[name])
        n = int(sens.numel[name])
        if n <= 0:
            continue
        c_t = s * am * math.sqrt(n)
        impact[name] = c_t
        per_w[name] = c_t / n
        if c_t <= 0.0:
            cont[name] = float(b_min)
            nbits[name] = b_min
            continue
        b_star = math.log2(lam * ln2 * c_t / n)
        cont[name] = b_star
        nbits[name] = int(min(b_max, max(b_min, round(b_star))))
    total_bits = sum(nbits[k] * sens.numel[k] for k in nbits)
    return BitAllocation(
        nbits=nbits,
        impact_coeff=impact,
        per_weight_impact=per_w,
        lam=float(lam),
        b_min=int(b_min),
        b_max=int(b_max),
        total_weight_bits=int(total_bits),
        continuous_bits=cont,
    )


def lam_for_target_mean_bits(
    sens: CombinedTensorSensitivity,
    target_mean_bits: float,
    *,
    b_min: int = 2,
    b_max: int = 8,
    iters: int = 60,
) -> float:
    """Bisection-solve λ so the param-weighted mean allocated bits ≈ ``target_mean_bits``.

    Convenience to pin the allocation at a chosen average precision (e.g. mean 5.0 bits to
    compare apples-to-apples against the uniform-int5 cap at the SAME total-bit budget but
    NON-uniformly allocated). Monotone in λ (more λ → more bits) so bisection converges.
    """
    total_numel = sum(sens.numel[k] for k in sens.per_tensor if sens.numel[k] > 0)
    if total_numel <= 0:
        raise FrontierBitallocError("no weight tensors to allocate")

    def mean_bits(lam: float) -> float:
        alloc = waterfill_bit_allocation(sens, lam, b_min=b_min, b_max=b_max)
        return alloc.total_weight_bits / total_numel

    # The impact coefficients c_t = s·absmax·√numel can span an enormous magnitude range
    # (s ~ 1e8 for the frontier), so λ's working range is data-dependent: we must expand
    # BOTH bounds (a fixed lo floor like 1e-3 already saturates at b_max for large c_t).
    # Anchor at the λ that puts the MEDIAN per-weight impact at the mid bit-width, then
    # expand outward until lo under-shoots and hi over-shoots the target.
    lo, hi = 1e-30, 1e30
    for _ in range(400):
        lo_under = mean_bits(lo) <= target_mean_bits
        hi_over = mean_bits(hi) >= target_mean_bits
        if lo_under and hi_over:
            break
        if not lo_under:
            lo *= 1e-3  # need a smaller λ (fewer bits)
        if not hi_over:
            hi *= 1e3  # need a larger λ (more bits)
        if lo < 1e-300 or hi > 1e300:
            break
    for _ in range(int(iters)):
        mid = math.sqrt(lo * hi)
        if mean_bits(mid) < target_mean_bits:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


# ---------------------------------------------------------------------------
# Step 3 — per-tensor non-uniform requant (the realization of the allocation)
# ---------------------------------------------------------------------------


def requant_state_dict_per_tensor_nbits(
    dec_sd: dict[str, torch.Tensor],
    nbits: dict[str, int],
    *,
    weight_min_dim: int = 2,
) -> dict[str, torch.Tensor]:
    """Quantize→dequantize each weight tensor at its OWN allocated nbits (REUSE intn_qdq).

    Each tensor in ``nbits`` is snapped to its per-tensor symmetric int-N grid; tensors NOT
    in ``nbits`` (biases, 1-D, or any excluded) are copied through fp32. The qdq output is a
    per-tensor symmetric grid representable by the codec's per-tensor int8 + fp16 scale (a
    lower-nbits tensor → fewer distinct int8 symbols → better brotli), so the result
    re-encodes through ``reencode_frontier_archive`` unchanged. Does NOT mutate the input.
    """
    out: dict[str, torch.Tensor] = {}
    for k, v in dec_sd.items():
        if not torch.is_tensor(v) or v.dim() < weight_min_dim:
            out[k] = v.clone() if torch.is_tensor(v) else v
            continue
        nb = nbits.get(k)
        out[k] = intn_qdq(v, int(nb)) if nb is not None else v.clone()
    return out


def per_stage_bits_summary(nbits: dict[str, int]) -> dict[str, list[int]]:
    """Group the allocated nbits by HNeRV stage (observability)."""
    by_stage: dict[str, list[int]] = {}
    for name, nb in nbits.items():
        stage = _stage_for_tensor_name(name)
        by_stage.setdefault(stage, []).append(int(nb))
    return by_stage


# Score-claim discipline metadata.
SCORE_CLAIM = False
PROMOTION_ELIGIBLE = False
READY_FOR_EXACT_EVAL_DISPATCH = False

__all__ = [
    "DEFAULT_W_POSE",
    "DEFAULT_W_SEG",
    "PROMOTION_ELIGIBLE",
    "READY_FOR_EXACT_EVAL_DISPATCH",
    "SCORE_CLAIM",
    "BitAllocation",
    "CombinedTensorSensitivity",
    "DecoderPoseSaliency",
    "FrontierBitallocError",
    "combine_sensitivities",
    "compute_decoder_tensor_margin_saliency",
    "compute_decoder_tensor_pose_saliency",
    "lam_for_target_mean_bits",
    "per_stage_bits_summary",
    "requant_state_dict_per_tensor_nbits",
    "waterfill_bit_allocation",
]
