# SPDX-License-Identifier: MIT
"""Lane SI: Saliency-Inversion compression weighting.

Premise (Fridrich UNIWARD, recalled in `project_council_shower_thoughts`):
    "Errors hidden in textured regions are undetectable."

Saliency inversion = compress aggressively in scorer BLIND SPOTS.
The standard saliency map is "where the scorer pays attention". The
INVERSE saliency map is "where the scorer DOESN'T pay attention" — those
are the safe regions to compress aggressively.

This module supplies three primitives:

1. ``compute_pixel_saliency(scorer, frames, output_shape)``
   Backprop the scorer output through the input frames; returns an
   (H, W) tensor of |dscorer/dpixel| (averaged over frames + channels)
   on the camera grid (output_shape).

2. ``compute_inverse_saliency_mask(saliency, threshold_quantile)``
   Returns a boolean (H, W) tensor — ``True`` where the scorer is
   blind (saliency below the q-quantile) → SAFE TO COMPRESS HARD.

3. ``apply_saliency_weighted_compression(masks, saliency_inv,
                                         high_crf, low_crf)``
   Splits mask frames into (high-saliency, low-saliency) regions and
   encodes each region with a different CRF. Returns the packed bytes
   (region-mask header + low-crf payload + high-crf payload).

Notes on usage:
- Scorer access is COMPRESS-TIME ONLY (CLAUDE.md strict-scorer-rule).
  We backprop through PoseNet/SegNet to derive the saliency map. The
  resulting region-mask is small (~hundreds of bytes if quantised) and
  CAN be shipped in the archive; the scorers themselves MUST NOT be
  loaded at inflate time.
- All routines are CUDA-required. CPU/MPS would break the byte-
  determinism contract for the resulting archive.
"""

from __future__ import annotations

import io
import struct
import zlib
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

# Camera grid (matches upstream/frame_utils.py: camera_size = (1164, 874))
CAMERA_W = 1164
CAMERA_H = 874

# Scorer input grid (matches upstream/frame_utils.py:
# segnet_model_input_size = (512, 384); PoseNet uses the same).
SCORER_W = 512
SCORER_H = 384


# ─────────────────────────────────────────────────────────────────────────
# 1. Pixel saliency
# ─────────────────────────────────────────────────────────────────────────


def compute_pixel_saliency(
    scorer: nn.Module,
    frames: torch.Tensor,
    output_shape: tuple[int, int] = (CAMERA_H, CAMERA_W),
    *,
    seq_len: int = 2,
    reduce: str = "mean",
) -> torch.Tensor:
    """Compute per-pixel scorer-gradient saliency on the camera grid.

    Parameters
    ----------
    scorer : nn.Module
        A PoseNet or SegNet instance in ``eval()`` mode on the same
        device as ``frames``. The module must expose
        ``preprocess_input(x)`` and consume the raw ``(B, T, C, H, W)``
        camera-resolution RGB float tensor (matching upstream/modules.py).
    frames : torch.Tensor
        ``(N_pairs, seq_len, 3, H, W)`` float frames in [0, 255]. Will be
        converted to float and grad-tracked internally.
    output_shape : tuple[int, int]
        Target ``(H, W)`` of the saliency map. Defaults to the camera
        grid (874, 1164) so the output aligns with masks.mkv.
    seq_len : int
        Frames per pair (upstream/frame_utils.py: 2). Used as a sanity
        check on the input shape.
    reduce : str
        How to reduce the per-frame gradient maps. ``"mean"`` averages
        over the batch (recommended). ``"max"`` returns the worst-case
        (per-pixel max across frames) — useful for "where is the scorer
        EVER sensitive?" queries.

    Returns
    -------
    torch.Tensor
        ``(output_shape[0], output_shape[1])`` non-negative float tensor.
        NOT normalised — the caller is responsible for thresholding.
    """
    if frames.ndim != 5:
        raise ValueError(
            f"frames must be (N, T, 3, H, W); got {tuple(frames.shape)}"
        )
    if frames.shape[1] != seq_len:
        raise ValueError(
            f"frames.shape[1] must be seq_len={seq_len}; got {frames.shape[1]}"
        )
    if frames.shape[2] != 3:
        raise ValueError(
            f"frames.shape[2] must be 3 (RGB); got {frames.shape[2]}"
        )
    if reduce not in ("mean", "max"):
        raise ValueError(f"reduce must be 'mean' or 'max'; got {reduce!r}")

    # Move to scorer device + grad-enable input tensor
    device = next(scorer.parameters()).device
    frames = frames.to(device=device, dtype=torch.float32).clone()
    frames.requires_grad_(True)

    # Forward through the scorer's preprocess + main network
    pre = scorer.preprocess_input(frames)
    out = scorer(pre)

    # PoseNet returns dict with 'pose' key; SegNet returns the seg logits.
    if isinstance(out, dict):
        # Use the FIRST half of pose (matches upstream compute_distortion:
        # `out[h.name][..., :h.out//2]`)
        head = out["pose"]
        target = head[..., : head.shape[-1] // 2]
    else:
        # Segmentation logits (B, 5, H, W). The contest distortion uses
        # argmax disagreement, which is non-differentiable; instead we
        # backprop the SOFTMAX entropy as a smooth proxy for the
        # decision boundary. High |grad| ⇒ pixel near a class boundary.
        probs = F.softmax(out, dim=1)
        # Entropy per pixel; mean over space + classes ⇒ scalar.
        target = -(probs * (probs.clamp_min(1e-12)).log()).sum(dim=1)

    # Sum-of-squares scalar for clean backprop signal
    loss = target.pow(2).mean()
    grads = torch.autograd.grad(loss, frames, retain_graph=False)[0]

    # Magnitude per pixel: sum over (channels). Shape: (N, T, H, W).
    grad_mag = grads.abs().sum(dim=2)

    # Reduce over (batch, time)
    if reduce == "mean":
        sal = grad_mag.mean(dim=(0, 1))
    else:  # max
        sal = grad_mag.amax(dim=(0, 1))

    # Resize to requested output shape (camera grid by default).
    if sal.shape != output_shape:
        sal = F.interpolate(
            sal[None, None].float(),
            size=output_shape,
            mode="bilinear",
            align_corners=False,
        )[0, 0]

    return sal.detach()


# ─────────────────────────────────────────────────────────────────────────
# 2. Inverse-saliency mask
# ─────────────────────────────────────────────────────────────────────────


def compute_inverse_saliency_mask(
    saliency: torch.Tensor,
    threshold_quantile: float = 0.5,
) -> torch.Tensor:
    """Boolean mask: True where saliency is BELOW the q-quantile.

    These are the scorer's BLIND SPOTS — the safe regions to compress
    aggressively (Fridrich UNIWARD: hide the error where the detector
    can't see).

    Parameters
    ----------
    saliency : torch.Tensor
        ``(H, W)`` non-negative tensor from ``compute_pixel_saliency``.
    threshold_quantile : float
        Quantile in [0, 1]. ``0.5`` ⇒ bottom half of saliency = blind
        spot. Higher values ⇒ smaller blind-spot region (more
        conservative compression).

    Returns
    -------
    torch.Tensor
        Boolean ``(H, W)`` tensor. ``True`` ⇒ low-saliency ⇒ aggressive
        compression OK. ``False`` ⇒ high-saliency ⇒ preserve quality.
    """
    if saliency.ndim != 2:
        raise ValueError(f"saliency must be (H, W); got {tuple(saliency.shape)}")
    if not 0.0 <= threshold_quantile <= 1.0:
        raise ValueError(
            f"threshold_quantile must be in [0, 1]; got {threshold_quantile}"
        )

    sal = saliency.detach().float().flatten()
    # torch.quantile is exact for small tensors; fall back to kthvalue
    # for very large to avoid an OOM (camera grid is 874*1164 ≈ 1M
    # entries — torch.quantile handles that fine on GPU).
    threshold = torch.quantile(sal, threshold_quantile)
    return (saliency <= threshold)


# ─────────────────────────────────────────────────────────────────────────
# 3. Saliency-weighted mask compression
# ─────────────────────────────────────────────────────────────────────────


# Region-mask packing: we ship the boolean (H, W) inverse-saliency mask
# as a header so the inflate-time decoder can recombine the two streams.
# The mask is RLE-then-zlib compressed (small footprint: a smooth
# top-half-sky / bottom-half-road split is ~50-200 bytes).
#
# Container format:
#   magic       4 bytes b"SLI1"
#   header_h    uint32  region-mask H
#   header_w    uint32  region-mask W
#   header_len  uint32  length of compressed region mask
#   high_len    uint32  length of high-saliency (low-CRF) payload
#   low_len     uint32  length of low-saliency  (high-CRF) payload
#   <header bytes> ... <high payload> ... <low payload>
#
# The two payloads are PNG-compressed integer mask frames sliced by the
# region mask (per-frame slicing during decode). For Lane SI v1 we use
# zlib on the raw uint8 bytes of each region — the compression-ratio
# delta versus full-frame AV1 comes from the region mask, not the codec.

_MAGIC = b"SLI1"
_HEADER_FMT = "<4sIIIII"  # magic, h, w, header_len, high_len, low_len
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)


def _rle_encode_bool(mask: torch.Tensor) -> bytes:
    """Run-length encode a flat boolean tensor; return zlib-compressed bytes.

    Format: pairs of (uint32 run-length, uint8 value). The leading bit is
    the starting value; subsequent runs alternate.
    """
    flat = mask.flatten().to(torch.uint8).cpu().numpy()
    if flat.size == 0:
        return zlib.compress(b"", level=9)

    # Find run boundaries
    diffs = (flat[1:] != flat[:-1]).nonzero()[0]
    starts = [0, *(int(d) + 1 for d in diffs)]
    starts.append(len(flat))
    runs = []
    for i in range(len(starts) - 1):
        run_len = starts[i + 1] - starts[i]
        runs.append(struct.pack("<IB", run_len, int(flat[starts[i]])))
    return zlib.compress(b"".join(runs), level=9)


def _rle_decode_bool(blob: bytes, total: int) -> torch.Tensor:
    """Inverse of ``_rle_encode_bool``; returns a flat uint8 tensor."""
    raw = zlib.decompress(blob)
    out = torch.zeros(total, dtype=torch.uint8)
    cursor = 0
    while raw:
        run_len, value = struct.unpack_from("<IB", raw, 0)
        out[cursor : cursor + run_len] = value
        cursor += run_len
        raw = raw[5:]
    return out


def apply_saliency_weighted_compression(
    masks: torch.Tensor,
    saliency_inv: torch.Tensor | None = None,
    high_crf: int = 50,
    low_crf: int = 30,
    *,
    encoder: Callable[[torch.Tensor, int], bytes] | None = None,
    saliency: torch.Tensor | None = None,
    target_bytes: int | None = None,
    target_bytes_tolerance: float = 256.0,
) -> bytes:
    """Encode mask frames with two CRFs split by saliency.

    The high-saliency region (where the scorer pays attention) is
    encoded at ``low_crf`` (preserve quality). The low-saliency region
    (scorer blind spot) is encoded at ``high_crf`` (aggressive
    compression — the scorer can't see the error there).

    The packed bytes carry a small header describing the region mask so
    an inflate-time decoder can recombine the two streams.

    Lane SI-V2 (``target_bytes`` mode): when ``target_bytes`` is provided
    the threshold is OPTIMISED via Lagrangian dual ascent on the byte
    budget instead of taken as a fixed quantile (Lane SI-V1's
    hard-coded 0.5). The encoder is probed at two thresholds, a linear
    rate model is fit, then the optimal threshold is computed in closed
    form. ``saliency`` (the raw, non-thresholded saliency map) MUST be
    supplied in this mode (the threshold derives a NEW
    ``saliency_inv``).

    Parameters
    ----------
    masks : torch.Tensor
        ``(N, H, W)`` uint8 mask frames (5-class argmax outputs from
        SegNet, values in [0, 4]).
    saliency_inv : torch.Tensor, optional
        Boolean ``(H, W)`` mask from ``compute_inverse_saliency_mask``.
        ``True`` ⇒ blind-spot ⇒ compress at ``high_crf``. Required when
        ``target_bytes`` is None; ignored when ``target_bytes`` is set.
    high_crf : int
        CRF for the blind-spot region (higher = more aggressive).
    low_crf : int
        CRF for the salient region (lower = preserve quality).
    encoder : callable, optional
        ``encoder(masks, crf) -> bytes`` for the actual codec. Defaults
        to a zlib-on-raw-uint8 fallback so unit tests have no av1
        dependency. The remote script substitutes a real AV1 encoder.
    saliency : torch.Tensor, optional
        Raw saliency map ``(H, W)`` normalised to ``[0, 1]``. Required
        when ``target_bytes`` is set so the Lagrangian threshold can
        index into the saliency distribution.
    target_bytes : int, optional
        Lane SI-V2 byte budget. When provided the threshold is learned
        via Lagrangian dual ascent (see
        ``tac.learnable_saliency_threshold``). When ``None`` (default),
        Lane SI-V1 behaviour: use the supplied ``saliency_inv`` as-is.
    target_bytes_tolerance : float
        Acceptable |encoded_bytes - target_bytes| in bytes. Default 256
        (typical zlib payload variance for our mask frames).

    Returns
    -------
    bytes
        Packed payload: header + region-mask + high-payload + low-payload.
    """
    if masks.ndim != 3 or masks.dtype != torch.uint8:
        raise ValueError(
            f"masks must be (N, H, W) uint8; got {tuple(masks.shape)} {masks.dtype}"
        )
    if not 0 <= low_crf <= high_crf <= 63:
        raise ValueError(
            f"CRF must satisfy 0 <= low_crf ({low_crf}) <= high_crf "
            f"({high_crf}) <= 63"
        )

    if encoder is None:
        encoder = _default_zlib_encoder

    # Lane SI-V2: derive saliency_inv from a learnable Lagrangian threshold.
    if target_bytes is not None:
        if saliency is None:
            raise ValueError(
                "target_bytes requires `saliency` (the raw saliency map). "
                "Lane SI-V1 used a fixed quantile; Lane SI-V2 needs the "
                "saliency distribution to derive the optimal threshold."
            )
        if saliency.ndim != 2:
            raise ValueError(
                f"saliency must be 2-D (H, W); got {tuple(saliency.shape)}"
            )
        if saliency.shape != masks.shape[1:]:
            raise ValueError(
                f"shape mismatch: masks {tuple(masks.shape)} vs saliency "
                f"{tuple(saliency.shape)}"
            )
        # Normalise saliency into [0, 1] so the threshold is a quantile.
        sal_n = saliency.detach().float().cpu()
        sal_min = float(sal_n.min().item())
        sal_max = float(sal_n.max().item())
        if sal_max > sal_min:
            sal_norm = (sal_n - sal_min) / (sal_max - sal_min)
        else:
            sal_norm = torch.full_like(sal_n, 0.5)

        def _bytes_at_threshold(t: float) -> int:
            mask_inv_t = (sal_norm <= float(t)).bool()
            payload = _encode_with_inv(
                masks=masks, saliency_inv=mask_inv_t,
                high_crf=high_crf, low_crf=low_crf, encoder=encoder,
            )
            return len(payload)

        from tac.learnable_saliency_threshold import (
            fit_linear_rate_model, optimise_threshold_for_target_bytes,
        )
        slope, intercept = fit_linear_rate_model(
            _bytes_at_threshold, sample_thresholds=(0.3, 0.7),
        )
        result = optimise_threshold_for_target_bytes(
            target_bytes=int(target_bytes),
            rate_slope=slope,
            rate_intercept=intercept,
            init_threshold=0.5,
            tolerance_bytes=float(target_bytes_tolerance),
        )
        # Re-encode with the chosen threshold and return the payload.
        chosen_inv = (sal_norm <= result.threshold).bool()
        return _encode_with_inv(
            masks=masks, saliency_inv=chosen_inv,
            high_crf=high_crf, low_crf=low_crf, encoder=encoder,
        )

    # Lane SI-V1 (legacy path) — saliency_inv supplied directly.
    if saliency_inv is None:
        raise ValueError(
            "Either saliency_inv (Lane SI-V1) or (saliency + target_bytes) "
            "(Lane SI-V2) must be provided."
        )
    if saliency_inv.dtype != torch.bool or saliency_inv.ndim != 2:
        raise ValueError(
            f"saliency_inv must be 2-D bool; got {tuple(saliency_inv.shape)} "
            f"{saliency_inv.dtype}"
        )
    if saliency_inv.shape != masks.shape[1:]:
        raise ValueError(
            f"shape mismatch: masks {tuple(masks.shape)} vs saliency_inv "
            f"{tuple(saliency_inv.shape)}"
        )

    return _encode_with_inv(
        masks=masks, saliency_inv=saliency_inv,
        high_crf=high_crf, low_crf=low_crf, encoder=encoder,
    )


def _encode_with_inv(
    masks: torch.Tensor,
    saliency_inv: torch.Tensor,
    high_crf: int,
    low_crf: int,
    encoder: Callable[[torch.Tensor, int], bytes],
) -> bytes:
    """Internal: pack masks + saliency_inv into the SLI1 container.
    Shared by Lane SI-V1 (caller-supplied inv) and Lane SI-V2 (Lagrangian-
    derived inv).
    """

    # Move to CPU (codec libs are CPU-bound)
    masks_c = masks.detach().cpu()
    sal_c = saliency_inv.detach().cpu()

    # Slice each frame: high-saliency region (~scorer cares) goes to
    # low_crf payload; low-saliency region (~blind spot) to high_crf.
    salient_idx = (~sal_c).nonzero(as_tuple=False)  # high-saliency pixels
    blind_idx = sal_c.nonzero(as_tuple=False)       # low-saliency  pixels

    # Per-frame strip: shape (N, n_salient_px) and (N, n_blind_px)
    n = masks_c.shape[0]
    salient_strip = masks_c[:, salient_idx[:, 0], salient_idx[:, 1]]
    blind_strip = masks_c[:, blind_idx[:, 0], blind_idx[:, 1]]

    # Encode each strip with the codec (3-D unsqueeze keeps the encoder
    # signature consistent with full-frame mask encoders).
    salient_payload = encoder(salient_strip.unsqueeze(1).contiguous(), low_crf)
    blind_payload = encoder(blind_strip.unsqueeze(1).contiguous(), high_crf)

    # Region mask header
    region_header = _rle_encode_bool(sal_c)

    h, w = sal_c.shape
    head = struct.pack(
        _HEADER_FMT,
        _MAGIC,
        h,
        w,
        len(region_header),
        len(salient_payload),
        len(blind_payload),
    )
    return head + region_header + salient_payload + blind_payload


def _default_zlib_encoder(masks: torch.Tensor, crf: int) -> bytes:
    """Codec-free fallback: serialize raw uint8 + zlib level chosen by CRF.

    Maps CRF [0, 63] → zlib level [9, 1] (CRF=0 ⇒ best quality ⇒ level=9;
    CRF=63 ⇒ worst quality ⇒ level=1). This is a TEST fallback — the
    remote script substitutes a real AV1 encoder via the ``encoder``
    argument.
    """
    # Linear map: CRF=0 → level=9; CRF=63 → level=1
    level = max(1, min(9, 9 - (crf * 8 // 63)))
    buf = io.BytesIO()
    buf.write(struct.pack("<IIII", *masks.shape, level))
    buf.write(zlib.compress(masks.cpu().numpy().tobytes(), level=level))
    return buf.getvalue()


def unpack_saliency_payload(blob: bytes) -> dict:
    """Parse the bytes produced by ``apply_saliency_weighted_compression``.

    Returns a dict with keys: ``h``, ``w``, ``region_mask`` (bool tensor),
    ``high_payload`` (bytes), ``low_payload`` (bytes). Used by the
    inflate-time decoder (Component 4, deferred — see docstring of
    ``apply_saliency_weighted_compression``).
    """
    if len(blob) < _HEADER_SIZE:
        raise ValueError(f"blob too short: {len(blob)} < {_HEADER_SIZE}")
    magic, h, w, hlen, high_len, low_len = struct.unpack_from(_HEADER_FMT, blob, 0)
    if magic != _MAGIC:
        raise ValueError(f"bad magic: {magic!r} (expected {_MAGIC!r})")
    cursor = _HEADER_SIZE
    region_blob = blob[cursor : cursor + hlen]
    cursor += hlen
    high = blob[cursor : cursor + high_len]
    cursor += high_len
    low = blob[cursor : cursor + low_len]
    region_flat = _rle_decode_bool(region_blob, h * w)
    region_mask = region_flat.view(h, w).bool()
    return {
        "h": h,
        "w": w,
        "region_mask": region_mask,
        "high_payload": high,
        "low_payload": low,
    }
