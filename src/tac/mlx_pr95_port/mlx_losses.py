# SPDX-License-Identifier: MIT
"""1:1 MLX port of PR95 ``hnerv_muon/src/losses.py`` — the score-aware losses.

These are the FOUR stage segmentation losses + the pose loss + the exact-d_seg
observable, ported verbatim from the proven PR95 torch implementation
(``submissions/hnerv_muon/src/losses.py``), in native MLX so they can be
parity-GATED against the torch reference (bit-/score-close per component).

The parity gate is the whole point of a 1:1 port: every MLX loss here is tested
in :mod:`tac.mlx_pr95_port.tests.test_torch_parity` to match the torch reference
to fp32 epsilon on the SAME logits + targets. A divergence FAILS the gate.

Loss progression (PR95 L14 8-stage curriculum):
  Stage 1 (CE):           :func:`ce_seg_loss_mlx`
  Stage 2 (tau-softplus): :func:`tau_softplus_seg_loss_mlx`  (tau=0.3)
  Stage 3+4 (smooth):     :func:`smooth_disagreement_seg_loss_mlx`  (tau=0.3)
  Stage 5+ (L7):          :func:`l7_softplus_seg_loss_mlx`
  Pose (all stages):      :func:`pose_loss_mlx`  = sqrt(10*MSE)

The seg-loss signature is ``(seg_logits, targets_hard)`` where ``seg_logits`` is
``(B, C, H, W)`` (the torch SegNet layout — we keep it identical so the parity
test feeds the SAME tensor to both backends) and ``targets_hard`` is
``(B, H, W)`` int. The exact d_seg is the argmax-disagreement RATE that
``upstream/evaluate.py`` charges.
"""

from __future__ import annotations

from typing import Any

try:  # pragma: no cover - import guard
    import mlx.core as mx
except Exception:  # pragma: no cover - mlx optional at import
    mx = None  # type: ignore[assignment]


def _require_mlx() -> None:
    if mx is None:  # pragma: no cover - environment guard
        raise RuntimeError(
            "tac.mlx_pr95_port.mlx_losses requires mlx.core (Apple Silicon)."
        )


# A very negative number used to mask the target channel before the max over the
# competing classes (matches PR95's ``-1e9`` scatter sentinel).
_NEG_INF_SENTINEL = -1.0e9


def _target_margin_mlx(seg_logits: Any, targets_hard: Any) -> Any:
    """Return the per-pixel margin ``target_logit - max_competing_logit``.

    1:1 with the PR95 torch ``gather`` + ``scatter(-1e9)`` + ``max`` pattern.
    ``seg_logits`` is ``(B, C, H, W)``; ``targets_hard`` is ``(B, H, W)`` int.
    Returns ``(B, 1, H, W)`` (channel dim kept, as in torch's ``keepdim=True``).
    """
    _require_mlx()
    c = int(seg_logits.shape[1])  # channel (class) count
    # one-hot over the channel axis for the target class.
    tgt = targets_hard.astype(mx.int32)  # (B,H,W)
    onehot = mx.stack(
        [(tgt == k).astype(seg_logits.dtype) for k in range(c)], axis=1
    )  # (B,C,H,W)
    target_logits = mx.sum(seg_logits * onehot, axis=1, keepdims=True)  # (B,1,H,W)
    # mask the target channel to -inf, then max over channels => best competitor.
    masked = seg_logits + onehot * _NEG_INF_SENTINEL
    competitor = mx.max(masked, axis=1, keepdims=True)  # (B,1,H,W)
    return target_logits - competitor


def ce_seg_loss_mlx(seg_logits: Any, targets_hard: Any) -> Any:
    """Stage-1 cross-entropy seg loss (1:1 with ``F.cross_entropy``).

    ``F.cross_entropy`` = mean over all pixels of ``-log_softmax(logit)[target]``.
    """
    _require_mlx()
    log_probs = seg_logits - mx.logsumexp(seg_logits, axis=1, keepdims=True)
    c = int(seg_logits.shape[1])  # channel (class) count
    tgt = targets_hard.astype(mx.int32)
    onehot = mx.stack(
        [(tgt == k).astype(seg_logits.dtype) for k in range(c)], axis=1
    )
    nll = -mx.sum(log_probs * onehot, axis=1)  # (B,H,W)
    return mx.mean(nll)


def tau_softplus_seg_loss_mlx(
    seg_logits: Any, targets_hard: Any, tau: float = 0.3
) -> Any:
    """Stage-2 tau-softplus margin surrogate (1:1 with PR95).

    ``mean(tau * softplus(-margin / tau))``.
    """
    _require_mlx()
    margin = _target_margin_mlx(seg_logits, targets_hard)
    # softplus(z) = log(1 + exp(z)); MLX provides logaddexp for numerical safety.
    z = -margin / tau
    softplus = mx.logaddexp(mx.zeros_like(z), z)
    return mx.mean(tau * softplus)


def smooth_disagreement_seg_loss_mlx(
    seg_logits: Any, targets_hard: Any, tau: float = 0.3
) -> Any:
    """Stage-3/4 smooth disagreement loss (1:1 with PR95): ``mean(sigmoid(-margin/tau))``."""
    _require_mlx()
    margin = _target_margin_mlx(seg_logits, targets_hard)
    return mx.mean(mx.sigmoid(-margin / tau))


def l7_softplus_seg_loss_mlx(
    seg_logits: Any,
    targets_hard: Any,
    tau: float = 0.3,
    l7_threshold: float = 1.0,
    l7_mult: float = 4.0,
) -> Any:
    """Stage-5+ L7-weighted softplus (1:1 with PR95).

    Pixels where ``margin < threshold`` get a ``(1 + l7_mult)`` boost,
    renormalized to mean 1, concentrating gradient on hard pixels. The weight is
    computed under ``stop_gradient`` (PR95 ``torch.no_grad``).
    """
    _require_mlx()
    margin = _target_margin_mlx(seg_logits, targets_hard)
    z = -margin / tau
    per_pixel = tau * mx.logaddexp(mx.zeros_like(z), z)
    weights = 1.0 + l7_mult * (margin < l7_threshold).astype(seg_logits.dtype)
    weights = mx.stop_gradient(weights / mx.mean(weights))
    return mx.mean(per_pixel * weights)


def margin_floor_hinge_mlx(
    seg_logits: Any, targets_hard: Any, *, margin_floor: float
) -> Any:
    """Cross-hardware margin-floor hinge: ``mean(relu(margin_floor - margin))``.

    The MLX-native twin of
    :func:`tac.capstone_vq_nerv.cross_hw_margin_hinge.margin_floor_hinge` (the
    torch-CPU authority hinge). Same math, same ``_target_margin_mlx`` margin
    core (``target_logit - max_competing_logit``), so the two are 1:1 up to the
    measured Metal fp32 reduction-order drift. It pushes every below-floor pixel
    margin PAST ``margin_floor`` so the LOCAL SegNet argmax survives the
    macOS->numpy->Linux/CUDA fp32 logit drift (lever-map L7, the
    numpy-portability guard).

    The ``relu`` (``mx.maximum(.., 0)``) is the boundary selector: an interior
    pixel whose margin already clears ``margin_floor`` contributes EXACTLY 0
    (and 0 gradient), while a below-floor (or argmax-wrong, ``margin < 0``)
    pixel gets a positive penalty whose gradient raises the target logit. This
    is a REAL loss term with a REAL gradient — NOT a constant (a no-op would
    have zero gradient and could not push the margin up).

    Args:
        seg_logits: live MLX SegNet logits ``(B, C, H, W)`` (NCHW; the same
            layout the canonical MLX PR95 seg-loss family consumes).
        targets_hard: GT SegNet argmax ``(B, H, W)`` int.
        margin_floor: required margin (anchor ~0.1 > the measured ~0.096
            cross-hardware logit drift). Must be > 0.

    Returns:
        Scalar MLX hinge loss (``relu(margin_floor - margin)`` averaged over all
        pixels; the relu zeroes the clear-of-floor pixels).
    """
    _require_mlx()
    if margin_floor <= 0.0:
        raise ValueError(f"margin_floor must be > 0; got {margin_floor}")
    margin = _target_margin_mlx(seg_logits, targets_hard)  # (B,1,H,W)
    return mx.mean(mx.maximum(margin_floor - margin, 0.0))


def pose_loss_mlx(pose_pred: Any, pose_target: Any) -> Any:
    """Pose loss ``sqrt(10*MSE)`` (1:1 with PR95, constant across all stages)."""
    _require_mlx()
    mse = mx.mean((pose_pred - pose_target) ** 2)
    return mx.sqrt(10.0 * mse + 1e-12)


def cat_entropy_v2_mlx(
    weight_tensors: Any,
    sigma: float = 0.2,
    sample_size: int = 2000,
    rng_key: Any = None,
) -> Any:
    """Stage-5+ C1a coder-aware entropy regularizer (1:1 MLX port of PR95).

    Size-weighted soft-histogram entropy over the decoder's Conv2d/Linear weight
    tensors. For each weight tensor: per-tensor symmetric INT8 normalization to
    ``[-127, 127]``, a Gaussian soft-assignment to the 255 integer grid bins with
    bandwidth ``sigma``, then categorical entropy (bits/weight). The per-tensor
    entropies are weighted by tensor ``numel`` and averaged.

    Pushing this term down (small sigma + big lambda) biases the post-INT8 weight
    distribution toward integer grid points (brotli-friendly), per PR95 L16.

    This is the EXACT torch ``cat_entropy_v2`` math (``submissions/hnerv_muon/src/
    losses.py``). The torch version iterates ``decoder.named_modules()`` selecting
    ``nn.Conv2d``/``nn.Linear`` and reads ``mod.weight``; here the caller passes the
    equivalent list of weight arrays (the ``*.weight`` ndim>=2 entries of the MLX
    decoder tree). The per-tensor math (abs-max normalize + soft histogram + entropy)
    is layout-invariant, so the NHWC MLX weights give the SAME value as the torch
    OIHW weights for identical weight values (verified bit-exact in the parity gate).

    Args:
        weight_tensors: iterable of MLX weight arrays (Conv2d/Linear weights).
        sigma: Gaussian soft-assignment bandwidth (PR95 0.2 -> 0.1 across stages).
        sample_size: subsample cap per tensor for the soft histogram (PR95 2000).
        rng_key: optional MLX random key for the subsample permutation. When None,
            uses the global MLX RNG (matches the torch ``torch.randperm`` default of
            consuming the ambient RNG stream).

    Returns:
        Scalar MLX array: the size-weighted mean entropy (bits/weight).
    """
    _require_mlx()
    bins = mx.arange(-127, 128).astype(mx.float32)  # (255,)
    total_numel = 0
    weighted_entropy = mx.zeros(())
    for w in weight_tensors:
        numel = int(w.size)
        ma = mx.stop_gradient(mx.max(mx.abs(w)))
        if float(ma.item()) < 1e-12:
            continue
        wn = mx.reshape(w / (ma / 127.0), (-1,))
        if int(wn.size) > sample_size:
            if rng_key is not None:
                rng_key, sub = mx.random.split(rng_key)
                perm = mx.argsort(mx.random.uniform(shape=(int(wn.size),), key=sub))
            else:
                perm = mx.argsort(mx.random.uniform(shape=(int(wn.size),)))
            wn = mx.take(wn, perm[:sample_size], axis=0)
        # Gaussian soft-assignment to the integer grid (B, 255).
        diff = (mx.expand_dims(wn, 1) - mx.expand_dims(bins, 0)) / sigma
        sa = mx.exp(-0.5 * diff * diff)
        sa = sa / (mx.sum(sa, axis=1, keepdims=True) + 1e-12)
        bp = mx.mean(sa, axis=0)
        bp = bp / (mx.sum(bp) + 1e-12)
        entropy = -mx.sum(bp * (mx.log(bp + 1e-12) / mx.log(mx.array(2.0))))
        weighted_entropy = weighted_entropy + numel * entropy
        total_numel += numel
    return weighted_entropy / max(total_numel, 1)


def fake_quantize_mlx(tensor: Any, n_levels: int = 127) -> Any:
    """Per-tensor symmetric INT8 fake-quant with straight-through estimator (1:1 PR95).

    Forward returns the rounded/clamped/dequantized weight; backward passes the
    gradient through unchanged (``(q*scale - tensor).stop_gradient + tensor``).
    The per-tensor abs-max scale is layout-invariant (NHWC == OIHW for identical
    values), so this matches the torch ``fake_quantize`` bit-for-bit.
    """
    _require_mlx()
    ma = mx.max(mx.abs(tensor))
    scale = mx.where(ma > 0, ma / n_levels, mx.array(1.0))
    q = mx.clip(mx.round(tensor / scale), -n_levels, n_levels)
    dequant = q * scale
    return mx.stop_gradient(dequant - tensor) + tensor


def apply_sigma_noise_mlx(weight: Any, sigma: float, rng_key: Any = None) -> Any:
    """Additive Gaussian weight noise ``w + sigma*N(0,1)`` (PR95 L17 sigma schedule).

    The PR95 sigma schedule (0.2 in stages 1-6 -> 0.1 in stages 7-8) injects
    Gaussian noise to simulate the uint8 quantization roundtrip during training
    (sister of the eval_roundtrip non-negotiable). ``sigma<=0`` is a no-op
    (returns the weight unchanged) so a stage with the schedule disabled is byte-
    identical to no-noise.
    """
    _require_mlx()
    if sigma <= 0.0:
        return weight
    if rng_key is not None:
        noise = mx.random.normal(shape=weight.shape, key=rng_key)
    else:
        noise = mx.random.normal(shape=weight.shape)
    return weight + sigma * noise


def exact_d_seg_from_logits_mlx(seg_logits: Any, targets_hard: Any) -> float:
    """Return the EXACT SegNet argmax-disagreement RATE (the evaluator's d_seg).

    ``mean(argmax(seg_logits, axis=channel) != targets_hard)``. This is the
    non-proxy observable; a working loop drives it down.
    """
    _require_mlx()
    pred = mx.argmax(seg_logits, axis=1)  # (B,H,W)
    disagree = (pred != targets_hard.astype(pred.dtype)).astype(mx.float32)
    return float(mx.mean(disagree).item())


# Registry mirrors PR95's per-stage seg-loss family (parity with the torch
# ``STAGE_SEG_LOSS_FNS`` in :mod:`tac.score_aware_loop.live_segnet_loss`).
STAGE_SEG_LOSS_FNS_MLX = {
    "ce_seg_loss": ce_seg_loss_mlx,
    "tau_softplus_seg_loss": tau_softplus_seg_loss_mlx,
    "smooth_disagreement_seg_loss": smooth_disagreement_seg_loss_mlx,
    "l7_softplus_seg_loss": l7_softplus_seg_loss_mlx,
}


__all__ = [
    "STAGE_SEG_LOSS_FNS_MLX",
    "apply_sigma_noise_mlx",
    "cat_entropy_v2_mlx",
    "ce_seg_loss_mlx",
    "exact_d_seg_from_logits_mlx",
    "fake_quantize_mlx",
    "l7_softplus_seg_loss_mlx",
    "margin_floor_hinge_mlx",
    "pose_loss_mlx",
    "smooth_disagreement_seg_loss_mlx",
    "tau_softplus_seg_loss_mlx",
]
