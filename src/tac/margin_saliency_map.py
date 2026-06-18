# SPDX-License-Identifier: MIT
"""Detector margin-saliency map — the unified Yousfi lever asset.

The contest's ``d_seg`` is the output of a frozen DETECTOR (EfficientNet-B2
SegNet, argmax-flip rate on a downsampled 512x384 decision grid). The
detector's OWN per-pixel sensitivity — how much its top1-top2 logit margin
moves per unit input perturbation — is the single map that drives all three
sub-0.15 levers identified in the Yousfi council check-in
(``.omx/research/yousfi_council_checkin_unified_margin_saliency_20260618.md``):

1. **d_seg lever** — a margin-hinge weighted by the detector's own gradient
   is first-order (vs a bare hinge = zeroth-order). This is the
   Yousfi-Fridrich 2022 detector-informed embedding cost, computed from the
   detector's OWN gradient (sharper than UNIWARD, which is the wrong domain
   for a SEMANTIC detector).
2. **rate lever** — the COMPLEMENT (low |saliency| pixels) is where
   quantization / dead-zone bytes can be shed at certified ~0 d_seg cost.
3. **survival certification** — the map lives on the detector's decision grid,
   so it certifies which repairs survive the downsample.

This module is the PRODUCER of that map (``∂margin/∂input`` via autograd
through the frozen scorer). The consumer side already exists:
:mod:`tac.logit_margin_sensitivity_weighted` (sensitivity-weighted logit-margin
loss) takes a per-pixel sensitivity tensor as input — this module produces it.

NO FAKE: the saliency is the REAL autograd gradient of the REAL frozen SegNet's
top1-top2 margin w.r.t. its input frame. ``segnet.preprocess_input`` is a clean
slice + bilinear interpolate (verified gradient-reachable: no ``@torch.no_grad``,
no in-place, no round), so the gradient flows end-to-end. No surrogate.

AUTHORITY: this is a DIAGNOSTIC / loss-weighting asset, never a score claim.
The only authoritative ``d_seg`` is ``upstream/evaluate.py`` on a byte-closed
archive. Run on CPU (MPS would 2x-corrupt the SegNet per CLAUDE.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


class MarginSaliencyError(ValueError):
    """Raised when margin-saliency inputs are malformed."""


def _topk_margin(logits: torch.Tensor) -> torch.Tensor:
    """Per-pixel top1-top2 logit margin, shape (B, H, W), kept differentiable.

    ``logits`` is (B, C, H, W). We use ``torch.topk(k=2)`` along the class dim;
    the difference of the two largest values is a piecewise-linear (hence a.e.
    differentiable) function of the logits, so autograd flows for the
    ``∂margin/∂input`` saliency.
    """
    if logits.dim() != 4:
        raise MarginSaliencyError(
            f"_topk_margin: expected (B, C, H, W) logits, got shape {tuple(logits.shape)}"
        )
    top2 = torch.topk(logits, 2, dim=1)
    return top2.values[:, 0] - top2.values[:, 1]  # (B, H, W)


@dataclass
class MarginSaliency:
    """A computed margin-saliency map on the SegNet decision grid.

    Attributes
    ----------
    saliency : torch.Tensor
        ``|∂(sum_p margin_p) / ∂input|`` reduced over the input RGB channels,
        shape (H, W) on the scorer's decision grid (e.g. 384x512). The detector
        cost: HIGH = the detector is sensitive here (a small input change moves
        the margin a lot = SHARP); LOW = score-blind (the rate-shed band).
    margin : torch.Tensor
        The per-pixel top1-top2 margin of the input frame, shape (H, W). The
        detector's confidence (distance-to-flip) on this specific frame.
    grid_hw : tuple[int, int]
        The (H, W) of the decision grid the maps live on.
    """

    saliency: torch.Tensor
    margin: torch.Tensor
    grid_hw: tuple[int, int]


def compute_margin_saliency_map(
    segnet,
    frame_chw_camera: torch.Tensor,
    *,
    flip_pixel_mask: torch.Tensor | None = None,
) -> MarginSaliency:
    """Compute ``|∂margin/∂input|`` for one camera-res frame through a frozen SegNet.

    Parameters
    ----------
    segnet
        A frozen SegNet exposing ``preprocess_input(x)`` (slice last frame +
        bilinear to the decision grid) and ``__call__`` returning
        ``(1, C, H, W)`` logits. Must be on CPU (CLAUDE.md: never MPS for the
        scorer).
    frame_chw_camera : torch.Tensor
        One camera-res frame, shape (3, H_cam, W_cam), float in [0, 255].
    flip_pixel_mask : torch.Tensor | None
        Optional boolean mask on the decision grid (H, W). When given, the
        scalar that is back-propagated is the margin summed over ONLY those
        pixels (e.g. the current flip pixels), so the saliency answers
        "what input change repairs the flips" rather than the global map.
        When ``None``, all decision-grid pixels contribute (the global map).

    Returns
    -------
    MarginSaliency
        ``saliency`` (H, W) = ``|∂(sum margin)/∂input|`` reduced over RGB,
        ``margin`` (H, W) = the per-pixel margin of this frame.

    Notes
    -----
    The saliency is the gradient of a SUM of per-pixel margins w.r.t. the input
    image. By linearity of ``∂/∂input``, ``∂(sum_p margin_p)/∂input_q`` is the
    superposed influence of input pixel ``q`` on every margin; for the flip-mask
    variant it is the influence on exactly the flip pixels. Reducing over the
    input RGB channels with an L2 norm gives a single per-input-pixel cost.
    """
    import einops

    if frame_chw_camera.dim() != 3 or frame_chw_camera.shape[0] != 3:
        raise MarginSaliencyError(
            "compute_margin_saliency_map: frame_chw_camera must be (3, H, W), "
            f"got {tuple(frame_chw_camera.shape)}"
        )

    frame = frame_chw_camera.detach().clone().requires_grad_(True)  # (3, H_cam, W_cam)
    hwc = frame.permute(1, 2, 0)  # (H_cam, W_cam, 3)
    pair = torch.stack([hwc, hwc]).unsqueeze(0)  # (1, 2, H_cam, W_cam, 3)
    x = einops.rearrange(pair, "b t h w c -> b t c h w")
    seg_in = segnet.preprocess_input(x)  # (1, 3, H, W) on decision grid
    logits = segnet(seg_in)  # (1, C, H, W) WITH grad
    margin = _topk_margin(logits)[0]  # (H, W)

    if flip_pixel_mask is not None:
        if flip_pixel_mask.shape != margin.shape:
            raise MarginSaliencyError(
                "compute_margin_saliency_map: flip_pixel_mask shape "
                f"{tuple(flip_pixel_mask.shape)} != margin grid {tuple(margin.shape)}"
            )
        scalar = margin[flip_pixel_mask.bool()].sum()
    else:
        scalar = margin.sum()

    (grad,) = torch.autograd.grad(scalar, frame, retain_graph=False, create_graph=False)
    # grad: (3, H_cam, W_cam) on the camera grid. Reduce RGB (L2) -> (H_cam, W_cam).
    sal_cam = grad.detach().pow(2).sum(0).sqrt()  # (H_cam, W_cam)

    # Resize the camera-grid saliency onto the decision grid so it aligns 1:1
    # with the margin map (the grid where the argmax actually decides).
    H, W = margin.shape
    sal = torch.nn.functional.interpolate(
        sal_cam.unsqueeze(0).unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
    )[0, 0]
    return MarginSaliency(saliency=sal.detach(), margin=margin.detach(), grid_hw=(H, W))


def saliency_boundary_concentration(
    saliency: torch.Tensor,
    margin: torch.Tensor,
    *,
    boundary_margin: float = 0.5,
) -> dict:
    """Characterize how concentrated the saliency is at the detector boundary.

    Sanity asset for the Yousfi seam: a well-formed margin-saliency map should
    place its mass at LOW-margin (boundary) pixels (where the detector is on a
    class wall and a small input change flips it), not at high-margin interior
    pixels. Returns the fraction of total saliency mass that falls in the
    low-margin band and the ratio of mean-saliency boundary/interior.
    """
    sal = saliency.detach().reshape(-1).double()
    mg = margin.detach().reshape(-1).double()
    boundary = mg < boundary_margin
    total = sal.sum().clamp_min(1e-30)
    mass_boundary = sal[boundary].sum() if boundary.any() else torch.zeros((), dtype=sal.dtype)
    mean_b = sal[boundary].mean() if boundary.any() else torch.zeros((), dtype=sal.dtype)
    mean_i = sal[~boundary].mean() if (~boundary).any() else torch.zeros((), dtype=sal.dtype)
    return {
        "boundary_margin_threshold": float(boundary_margin),
        "frac_pixels_boundary": float(boundary.float().mean()),
        "frac_saliency_mass_in_boundary": float(mass_boundary / total),
        "mean_saliency_boundary": float(mean_b),
        "mean_saliency_interior": float(mean_i),
        "boundary_over_interior_ratio": (
            float(mean_b / mean_i) if float(mean_i) > 1e-30 else float("inf")
        ),
        "saliency_mean": float(sal.mean()),
        "saliency_max": float(sal.max()) if sal.numel() else 0.0,
        "saliency_gini": _gini(sal),
    }


def _gini(x: torch.Tensor) -> float:
    """Gini concentration of a non-negative vector (0=uniform, 1=concentrated)."""
    v = x.detach().reshape(-1).double()
    v = v.clamp_min(0)
    n = v.numel()
    if n == 0 or float(v.sum()) <= 1e-30:
        return 0.0
    sv, _ = torch.sort(v)
    idx = torch.arange(1, n + 1, dtype=torch.double)
    return float((2.0 * (idx * sv).sum()) / (n * sv.sum()) - (n + 1.0) / n)


# ---------------------------------------------------------------------------
# Per-DECODER-TENSOR / per-STAGE d_seg-sensitivity (the QAT bit-allocation prior)
# ---------------------------------------------------------------------------
#
# The Path-B redirect (.omx/research/path_b_recalibration_and_resolve_audit_*) is
# to QAT-shrink the EXISTING frontier decoder, not train a new small-basis vehicle.
# The score-aware QAT (``tac.torch_vehicle.score_aware_qat``) needs a per-tensor
# ``sensitivity`` dict to water-fill the INT8 grid: protect (fine grid → more bytes)
# the d_seg-critical tensors, coarsen (int4 → fewer brotli bytes) the d_seg-blind
# tensors. THIS producer computes that dict on the REAL frontier decoder via
# ``||∂(SegNet top1-top2 margin)/∂w_t||`` — the same first-order detector-informed
# signal as the per-pixel map above, lifted to the WEIGHT domain.
#
# NO FAKE: the per-tensor saliency is the REAL autograd gradient of the REAL frozen
# SegNet's margin w.r.t. each REAL decoder weight tensor (verified in
# ``tests/test_margin_saliency_decoder_tensor.py``). It is a DIAGNOSTIC / bit-alloc
# COST, never a score claim — the only authoritative d_seg is upstream/evaluate.py.

# Canonical HNeRV (PR95/PR101/PR106 family) decoder tensor-name -> upsample-stage
# geometry. The decoder lifts a 6x8 base grid to 384x512 over 6 PixelShuffle x2
# blocks; the SegNet decides on a 384x512 grid after a stride-2 stem (its first
# real decision resolution is ~192x256). So the late stages (blocks.4/blocks.5 at
# 96x128 -> 192x256 -> 384x512, refine + rgb heads at 384x512) carry the
# decision-band detail; the stem + early blocks carry the coarse/low-freq content.
# The map is by name PREFIX so it is robust to ``stem.weight``/``stem.bias`` etc.
HNERV_STAGE_OUTPUT_HW: dict[str, tuple[int, int]] = {
    "stem": (6, 8),
    "blocks.0": (12, 16),
    "blocks.1": (24, 32),
    "blocks.2": (48, 64),
    "blocks.3": (96, 128),
    "blocks.4": (192, 256),
    "blocks.5": (384, 512),
    "skips.2": (48, 64),
    "skips.3": (96, 128),
    "skips.4": (192, 256),
    "refine.0": (384, 512),
    "refine.1": (384, 512),
    "rgb_0": (384, 512),
    "rgb_1": (384, 512),
}


def _stage_for_tensor_name(name: str) -> str:
    """Map a parameter name (e.g. ``blocks.4.weight``) to its HNeRV stage key
    (``blocks.4``). Falls back to the first dotted component for unknown names."""
    for prefix in HNERV_STAGE_OUTPUT_HW:
        if name == prefix or name.startswith(prefix + "."):
            return prefix
    return name.split(".")[0]


@dataclass
class DecoderTensorSaliency:
    """Per-decoder-tensor d_seg-sensitivity ranking — the QAT bit-allocation prior.

    Attributes
    ----------
    per_tensor : dict[str, float]
        Tensor name -> ``||∂(sum margin)/∂w_t||`` accumulated over the scored
        frames. This is the dict consumed by
        ``tac.torch_vehicle.score_aware_qat.per_tensor_levels_from_sensitivity``:
        HIGH = d_seg-critical (protect / keep high precision); LOW = d_seg-blind
        (int4 candidate).
    per_tensor_normalized : dict[str, float]
        ``||∂margin/∂w_t|| / sqrt(numel_t)`` — sensitivity PER WEIGHT, so a large
        tensor is not ranked critical merely for being large. The rank-quantile
        bit-allocator should prefer this normalized form.
    ranking : list[tuple[str, float]]
        ``per_tensor`` sorted DESCENDING (most d_seg-critical first). The protect
        list is the head; the int4 list is the tail.
    per_stage : dict[str, float]
        Stage key -> summed tensor saliency over the tensors in that stage. Lets
        the bit-allocator reason per upsample stage (the spatial geometry).
    stage_output_hw : dict[str, tuple[int, int]]
        Stage key -> its output (H, W) grid (the geometry for the
        decision-band-vs-blind-band check).
    n_frames : int
        How many frames were scored to accumulate the saliency.
    tensor_numel : dict[str, int]
        Tensor name -> element count (for the normalized form + reporting).
    """

    per_tensor: dict[str, float]
    per_tensor_normalized: dict[str, float]
    ranking: list[tuple[str, float]]
    per_stage: dict[str, float]
    stage_output_hw: dict[str, tuple[int, int]]
    n_frames: int
    tensor_numel: dict[str, int]


def compute_decoder_tensor_margin_saliency(
    decoder,
    latents: torch.Tensor,
    segnet,
    *,
    render_frame_chw,
    frame_indices=None,
    score_head: str = "rgb_0",
    flip_pixel_mask_fn=None,
) -> DecoderTensorSaliency:
    """Per-decoder-tensor ``||∂(SegNet margin)/∂w_t||`` on a REAL decoder vehicle.

    This is the WEIGHT-domain analogue of :func:`compute_margin_saliency_map`: it
    backpropagates the SegNet top1-top2 margin (the detector's distance-to-flip)
    through the decoder to EACH decoder weight tensor, accumulating the gradient
    L2-norm per tensor over a handful of scored frames. The result is the
    bit-allocation cost the score-aware QAT consumes (protect the d_seg-critical
    tensors at high precision; coarsen the d_seg-blind tensors to int4).

    Parameters
    ----------
    decoder : torch.nn.Module
        The decoder vehicle (e.g. the frontier ``HNeRVDecoder``) with its
        score-relevant weights ALREADY loaded. This function sets
        ``requires_grad_(True)`` on its parameters and leaves them that way; pass a
        decoder you own. Must be on CPU (CLAUDE.md: never MPS for the scorer).
    latents : torch.Tensor
        The decoder's latent rows, shape (N, latent_dim) (e.g. (600, 28) for PR101).
    segnet
        A frozen SegNet exposing ``preprocess_input`` + ``__call__`` (the same
        contract as :func:`compute_margin_saliency_map`). On CPU.
    render_frame_chw : callable
        ``render_frame_chw(decoder, latent_row, score_head) -> (3, H, W)`` returning
        ONE camera-res frame (float in [0, 255]) WITH the autograd graph intact
        (no ``.detach()``). It selects which head's frame to score (``rgb_0`` /
        ``rgb_1``). This callable is vehicle-specific (the decoder's forward
        signature differs per family) so the producer stays vehicle-agnostic.
    frame_indices : iterable[int] | None
        Which latent rows to score. ``None`` -> a deterministic spread of up to 8
        rows across the sequence (enough to characterize the per-tensor ranking;
        the ranking is stable across frames — boundary tensors dominate every
        frame). More rows = a smoother EMA at higher CPU cost.
    score_head : str
        Which decoder output head's frame to score (passed to ``render_frame_chw``).
        ``rgb_0`` (the first frame of the pair) is the SegNet's scored frame
        (``x[:, -1, ...]`` last-frame slice == frame index 1 in the eval pair, but
        the renderer's two heads are symmetric for the saliency RANKING; the
        consumer may run both heads and sum).
    flip_pixel_mask_fn : callable | None
        Optional ``flip_pixel_mask_fn(logits) -> bool mask (H, W)`` to restrict the
        back-propagated margin to specific decision-grid pixels (e.g. the current
        flip pixels). ``None`` -> the global margin sum (all decision pixels).

    Returns
    -------
    DecoderTensorSaliency
        The per-tensor / per-stage ranking (the QAT bit-allocation prior).
    """
    if latents.dim() != 2:
        raise MarginSaliencyError(
            f"compute_decoder_tensor_margin_saliency: latents must be (N, D), got {tuple(latents.shape)}"
        )
    n = int(latents.shape[0])
    if frame_indices is None:
        k = min(8, n)
        # deterministic even spread across the sequence
        frame_indices = [round(i * (n - 1) / max(1, k - 1)) for i in range(k)] if k > 1 else [0]
    frame_indices = [int(i) for i in frame_indices]
    for i in frame_indices:
        if not (0 <= i < n):
            raise MarginSaliencyError(
                f"frame index {i} out of range for {n} latent rows"
            )

    params = [(name, p) for name, p in decoder.named_parameters()]
    for _name, p in params:
        p.requires_grad_(True)
    names = [name for name, _p in params]
    numel = {name: int(p.numel()) for name, p in params}

    accum = dict.fromkeys(names, 0.0)
    for i in frame_indices:
        decoder.zero_grad(set_to_none=True)
        latent_row = latents[i : i + 1]  # (1, D)
        frame = render_frame_chw(decoder, latent_row, score_head)  # (3, H, W) WITH grad
        if frame.dim() != 3 or frame.shape[0] != 3:
            raise MarginSaliencyError(
                "render_frame_chw must return (3, H, W); got "
                f"{tuple(frame.shape)}"
            )
        import einops

        hwc = frame.permute(1, 2, 0)
        pair = torch.stack([hwc, hwc]).unsqueeze(0)  # (1, 2, H, W, 3)
        x = einops.rearrange(pair, "b t h w c -> b t c h w")
        seg_in = segnet.preprocess_input(x)
        logits = segnet(seg_in)
        margin = _topk_margin(logits)[0]  # (H, W)
        if flip_pixel_mask_fn is not None:
            mask = flip_pixel_mask_fn(logits)
            if mask.shape != margin.shape:
                raise MarginSaliencyError(
                    f"flip_pixel_mask_fn returned shape {tuple(mask.shape)} != margin grid {tuple(margin.shape)}"
                )
            scalar = margin[mask.bool()].sum()
        else:
            scalar = margin.sum()
        grads = torch.autograd.grad(
            scalar, [p for _n, p in params], retain_graph=False, allow_unused=True
        )
        for (name, _p), g in zip(params, grads, strict=True):
            if g is not None:
                accum[name] += float(g.detach().norm().item())

    nfr = len(frame_indices)
    per_tensor = {name: accum[name] / nfr for name in names}
    per_tensor_normalized = {
        name: (per_tensor[name] / (numel[name] ** 0.5) if numel[name] > 0 else 0.0)
        for name in names
    }
    ranking = sorted(per_tensor.items(), key=lambda kv: kv[1], reverse=True)

    per_stage: dict[str, float] = {}
    stage_hw: dict[str, tuple[int, int]] = {}
    for name in names:
        stage = _stage_for_tensor_name(name)
        per_stage[stage] = per_stage.get(stage, 0.0) + per_tensor[name]
        if stage in HNERV_STAGE_OUTPUT_HW:
            stage_hw[stage] = HNERV_STAGE_OUTPUT_HW[stage]

    return DecoderTensorSaliency(
        per_tensor=per_tensor,
        per_tensor_normalized=per_tensor_normalized,
        ranking=ranking,
        per_stage=per_stage,
        stage_output_hw=stage_hw,
        n_frames=nfr,
        tensor_numel=numel,
    )


def verify_stage_saliency_ordering(
    per_stage: dict[str, float],
    stage_output_hw: dict[str, tuple[int, int]],
    *,
    decision_band_min_pixels: int = 192 * 256,
) -> dict:
    """Check the per-stage saliency ordering against the expected geometry.

    The Yousfi seam (``.omx/research/yousfi_council_checkin_unified_margin_saliency_*``)
    predicts the SegNet decision band lives at >= ~192x256 (post stride-2 stem):
    the decision-band stages (output >= 192x256) should carry MORE d_seg-saliency
    mass than the stem-Nyquist-BLIND coarse stages (< 192x256). This returns the
    mass split + the verdict so a build can confirm the cost map matches the
    expected geometry BEFORE it drives the bit-allocation.

    Note: this is a SANITY check on the cost map, not a score claim. A FALSE
    verdict means the saliency is NOT decision-band-concentrated and the bit-alloc
    prior should be re-examined (e.g. the decoder couples coarse stages into the
    boundary via the bilinear skips), not silently trusted.
    """
    decision_mass = 0.0
    blind_mass = 0.0
    decision_stages: list[str] = []
    blind_stages: list[str] = []
    for stage, mass in per_stage.items():
        hw = stage_output_hw.get(stage)
        if hw is None:
            continue
        if hw[0] * hw[1] >= decision_band_min_pixels:
            decision_mass += mass
            decision_stages.append(stage)
        else:
            blind_mass += mass
            blind_stages.append(stage)
    total = decision_mass + blind_mass
    return {
        "decision_band_min_pixels": int(decision_band_min_pixels),
        "decision_stages": sorted(decision_stages),
        "blind_stages": sorted(blind_stages),
        "decision_band_saliency_mass": float(decision_mass),
        "blind_band_saliency_mass": float(blind_mass),
        "decision_band_mass_fraction": float(decision_mass / total) if total > 1e-30 else 0.0,
        "decision_over_blind_ratio": (
            float(decision_mass / blind_mass) if blind_mass > 1e-30 else float("inf")
        ),
        "ordering_matches_geometry": bool(decision_mass >= blind_mass),
    }
