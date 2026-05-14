"""D1 logit-margin map — the canonical geometric-nullspace primitive.

The SegNet logit margin ``m(x, y) = top1_logit(x, y) - top2_logit(x, y) >= 0``
is the **Newton-step distance** from the current logits to the nearest
decision boundary. By the structural geometry of argmax, perturbations
``δ`` with ``||δ||_inf < m(x, y)`` preserve the argmax decision and are
therefore **scorer-invisible** to SegNet's distortion term.

This module exposes:

* :func:`compute_logit_margin_map` — differentiable computation through
  the canonical SegNet preprocess pipeline (Catalog #164).
* :func:`compute_logit_margin_map_dummy` — test-only constant map.
* :func:`quantize_margin_map_int8` / :func:`dequantize_margin_map_int8` —
  byte-faithful int8 packing for archive storage.

Routes through :func:`tac.substrates.score_aware_common._require_preprocess`
so any SegNet missing the canonical ``preprocess_input`` is refused
explicitly (Catalog #164 compliance).

NO score claims. The margin map is a PROXY signal — the resulting
polytope-allocated noise payload must still be auth-evaluated on
contest-CUDA + contest-CPU before any promotion / ranking / kill verdict.
"""

from __future__ import annotations

from enum import Enum

import torch

from tac.substrates.score_aware_common import _require_preprocess

MARGIN_MAP_DEFAULT_RESOLUTION: tuple[int, int] = (384, 512)
"""Scorer eval resolution per ``upstream/modules.py:109`` SegNet preprocess."""

MARGIN_MAP_SHRUNK_RESOLUTION: tuple[int, int] = (96, 128)
"""16x-shrunk margin map (96, 128) for archive cost reduction.

D1 dispatch finding 2026-05-14 R3 showed full-resolution (384, 512) margin map
brotli'd to ~39 KB (margin_map_int8 = 196,608 bytes pre-brotli; brotli closes
~80% on high-margin-interior plateaus → 39 KB; plus 35 B polytope payload +
meta + 31 B header → 43,296 B total D1POLY1 sidecar).

The rate-axis cost is ``25 * 43,296 / 37,545,489 ≈ +0.029`` per CLAUDE.md
contest scoring formula — too expensive on its own to land sub-A1 without an
equally-strong overlay savings on the seg/pose components.

Shrinking the margin map 16× to (96, 128) before int8-quantization gives
``96 * 128 = 12,288`` pre-brotli bytes → ~2.4 KB brotli'd → ~2.7 KB total
D1POLY1 sidecar. Rate-axis cost is ``25 * 2,700 / 37,545,489 ≈ +0.0018``.

Trade-off: the SegNet margin map has spatial structure at scorer eval grid
(384, 512); area-pool downsample to (96, 128) preserves coarse polytope
geometry but blurs fine boundary detail. The inflate-side bicubic upsample
back to (384, 512) recovers the smooth interior margin but may misallocate
noise near polytope boundaries. The deep-math memo §3.6 polytope theorem
remains valid: at any resolution the per-pixel safe budget is bounded above
by ``m / L`` for the average margin in the pooled region.

Use ``MARGIN_MAP_DEFAULT_RESOLUTION`` for L2-PROOF dispatch; use
``MARGIN_MAP_SHRUNK_RESOLUTION`` for cost-efficient L2-EXPLORATORY dispatch.
"""


class MarginMapMode(Enum):
    """How to derive the per-pixel margin map."""

    SEGNET_TOP1_MINUS_TOP2 = "segnet_top1_minus_top2"
    """Canonical: ``top1_logit - top2_logit`` on the SegNet 5-class output."""

    UNIFORM = "uniform"
    """Uniform margin — degenerates polytope encoder to plain water-fill."""

    DUMMY_CONSTANT = "dummy_constant"
    """Test-only constant; :func:`compute_logit_margin_map_dummy` is canonical."""


def compute_logit_margin_map(
    *,
    seg_scorer: torch.nn.Module,
    rgb_pair_btchw: torch.Tensor,
    target_resolution: tuple[int, int] = MARGIN_MAP_DEFAULT_RESOLUTION,
    eps: float = 1e-6,
    detach_grad: bool = True,
    downsample_mode: str = "bilinear",
) -> torch.Tensor:
    """Compute the canonical SegNet logit margin map.

    The SegNet preprocess (``upstream/modules.py:107-109``) takes a 5D
    ``(B, T, C, H, W)`` pair tensor and code-discards frame 0 via
    ``x[:, -1, ...]`` slicing — so the margin we compute is the **frame-1
    margin**. This is exactly the geometric quantity D1 exploits.

    Per Catalog #164, the input MUST be staged via the canonical 5D pair
    shape ``(B, T=2, C=3, H, W)`` so the upstream preprocess + scorer
    forward observe the contest contract.

    Args:
        seg_scorer: Upstream :class:`SegNet` with ``preprocess_input``.
        rgb_pair_btchw: ``(B, T=2, C=3, H, W)`` pair tensor at camera
            resolution. Per Catalog #164 routing the upstream preprocess
            will downsample to ``(384, 512)`` internally; we still compute
            the margin map at ``target_resolution`` (which the caller
            sets to the scorer plane unless they explicitly want a
            different resolution).
        target_resolution: Output spatial resolution. Defaults to scorer
            eval resolution ``(384, 512)``. Pass
            :data:`MARGIN_MAP_SHRUNK_RESOLUTION` (``(96, 128)``) for the
            16x-shrunk archive cost variant.
        eps: Eps-clamp to avoid div-by-zero in degenerate top1==top2.
        detach_grad: If True (default), the returned margin map is
            grad-detached. Set False only for a research probe of
            margin-aware loss training.
        downsample_mode: torch.nn.functional.interpolate mode used when
            ``target_resolution`` is smaller than the scorer's logit grid.
            ``"area"`` preserves the average margin in each pooled region
            (the canonical choice for shrunk margin maps). ``"bilinear"``
            (default) is interpolation-faithful for the full-resolution
            case where target == logit grid.

    Returns:
        Margin map ``(B, H, W)`` float32 at ``target_resolution``. Values
        are >= 0; large = deep-interior; small = decision-frontier.

    Raises:
        ValueError: when ``rgb_pair_btchw`` shape is wrong.
        :class:`tac.substrates.score_aware_common.ScoreAwareScorerContractError`:
            when the scorer lacks ``preprocess_input``.
    """
    _require_preprocess(seg_scorer, scorer_name="SegNet")
    if rgb_pair_btchw.dim() != 5:
        raise ValueError(
            "compute_logit_margin_map expects 5D (B, T, C, H, W); got "
            f"{tuple(rgb_pair_btchw.shape)}"
        )
    if rgb_pair_btchw.shape[1] != 2:
        raise ValueError(
            "compute_logit_margin_map expects T=2 pair; got T="
            f"{rgb_pair_btchw.shape[1]}"
        )
    if rgb_pair_btchw.shape[2] != 3:
        raise ValueError(
            "compute_logit_margin_map expects C=3 RGB; got C="
            f"{rgb_pair_btchw.shape[2]}"
        )

    # SegNet.preprocess_input slices frame 1 + resamples to (384, 512).
    seg_input = seg_scorer.preprocess_input(rgb_pair_btchw)
    # SegNet.forward returns (B, 5, H_seg, W_seg) logits.
    logits = seg_scorer(seg_input)
    if logits.dim() != 4 or logits.shape[1] < 2:
        raise ValueError(
            "SegNet logits shape unexpected (need (B, C>=2, H, W)); got "
            f"{tuple(logits.shape)}"
        )

    # top2 = the two largest logits per pixel; margin = top1 - top2.
    # torch.topk with k=2 along dim=1 returns descending values by default.
    top2_values, _ = torch.topk(logits, k=2, dim=1, largest=True, sorted=True)
    margin = (top2_values[:, 0] - top2_values[:, 1]).clamp_min(0.0)  # (B, H, W)

    # The margin is non-negative by construction (top1 >= top2). We
    # clamp_min(eps) only at the very end when we need to divide by it;
    # the raw margin map preserves the canonical zero-margin (boundary)
    # pixels for downstream stratification.
    target_h, target_w = target_resolution
    if margin.shape[-2:] != (target_h, target_w):
        if downsample_mode == "area":
            # Area-pool downsample (best for shrunk margin maps; preserves
            # average margin per pooled region, which is the canonical
            # quantity that bounds the polytope safe budget).
            margin = torch.nn.functional.interpolate(
                margin.unsqueeze(1),  # (B, 1, H, W)
                size=(target_h, target_w),
                mode="area",
            ).squeeze(1)
        elif downsample_mode == "bilinear":
            margin = torch.nn.functional.interpolate(
                margin.unsqueeze(1),  # (B, 1, H, W)
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
        else:
            raise ValueError(
                f"compute_logit_margin_map: unsupported downsample_mode="
                f"{downsample_mode!r}; expected 'area' or 'bilinear'"
            )

    if detach_grad:
        margin = margin.detach()

    return margin.contiguous()


def compute_logit_margin_map_dummy(
    resolution: tuple[int, int] = MARGIN_MAP_DEFAULT_RESOLUTION,
    constant_value: float = 1.0,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Test-only constant margin map. NEVER use in a real training path.

    Uniform margin degenerates the polytope encoder to plain water-fill
    (every pixel equally interior). Useful for archive grammar roundtrip
    tests + polytope encoder unit tests where we want to isolate the
    allocator from the margin-grad path. NEVER pass through
    :func:`compose_with_base` for a real archive.
    """
    if constant_value <= 0:
        raise ValueError(
            f"constant_value must be > 0; got {constant_value}. A zero or "
            "negative margin would force every pixel to the decision "
            "boundary — refuse the dummy fixture."
        )
    h, w = resolution
    return torch.full((h, w), constant_value, device=device, dtype=dtype)


def quantize_margin_map_int8(
    margin_map: torch.Tensor,
    *,
    scale: float = 127.0,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, float]:
    """Quantize margin map to int8 for archive storage.

    The margin map is non-negative by construction, so we use the FULL
    positive int8 range ``[0, 127]`` for resolution (vs YUCR cost map
    which is signed). The returned ``(int8_map, recovered_scale)`` lets
    inflate-time recover the float margin map deterministically:
    ``margin_recovered = int8_map.float() * recovered_scale``.

    Args:
        margin_map: Float tensor (typically ``(H, W)`` or ``(B, H, W)``).
            MUST be non-negative (margin is top1 - top2 >= 0).
        scale: Target int8 dynamic range (typically 127 for full-positive
            int8). The encoder side records ``max(margin) / scale`` as
            the recovery scale.
        eps: Eps-clamp for div-by-zero (when the whole map is zero).

    Returns:
        Tuple ``(int8_quantized, recovered_scale)`` where
        ``recovered_scale = max_value / scale``.

    Raises:
        ValueError: when ``scale`` is out of range or ``margin_map`` has
            negative values.
    """
    if scale <= 0 or scale > 127:
        raise ValueError(f"scale must be in (0, 127]; got {scale}")
    if (margin_map.detach() < 0).any():
        raise ValueError(
            "margin_map has negative values — margin = top1 - top2 must "
            "be >= 0 by construction. Refuse the int8 packing."
        )
    margin_max = margin_map.detach().abs().max().clamp_min(eps).item()
    recovered_scale = margin_max / scale
    int8 = (
        (margin_map.detach() / recovered_scale)
        .clamp(0.0, 127.0)
        .round()
        .to(torch.int8)
    )
    return int8, recovered_scale


def upsample_margin_map_for_overlay(
    margin_map_float: torch.Tensor,
    *,
    target_resolution: tuple[int, int] = MARGIN_MAP_DEFAULT_RESOLUTION,
    upsample_mode: str = "bicubic",
) -> torch.Tensor:
    """Upsample a shrunk margin map back to scorer resolution for overlay.

    The L2 overlay path packs the margin map at
    :data:`MARGIN_MAP_SHRUNK_RESOLUTION` (96, 128) at encode time for archive
    cost reduction; inflate-time must restore it to scorer eval resolution
    (384, 512) so per-pixel polytope-interior noise can be allocated at the
    scorer grid. The 16× upsample is bicubic so smooth interior margins
    interpolate without staircase artifacts.

    For full-resolution maps where ``margin_map_float.shape ==
    target_resolution`` this is a no-op pass-through.

    Args:
        margin_map_float: ``(H, W)`` float32 tensor, typically dequantized
            from :func:`dequantize_margin_map_int8`.
        target_resolution: Output spatial resolution. Defaults to
            :data:`MARGIN_MAP_DEFAULT_RESOLUTION`.
        upsample_mode: ``"bicubic"`` (default) or ``"bilinear"``.

    Returns:
        Float32 tensor at ``target_resolution``. Values are
        clamp_min(0.0) so the polytope-interior invariant
        (``B_safe >= 0``) is preserved through the interpolation.

    Raises:
        ValueError: when ``margin_map_float`` is not 2D or
            ``upsample_mode`` is unrecognized.
    """
    if margin_map_float.dim() != 2:
        raise ValueError(
            "upsample_margin_map_for_overlay expects 2D margin map; got "
            f"shape {tuple(margin_map_float.shape)}"
        )
    if upsample_mode not in {"bicubic", "bilinear"}:
        raise ValueError(
            f"upsample_margin_map_for_overlay: unsupported upsample_mode="
            f"{upsample_mode!r}; expected 'bicubic' or 'bilinear'"
        )
    target_h, target_w = target_resolution
    if tuple(margin_map_float.shape) == (target_h, target_w):
        return margin_map_float.clamp_min(0.0).contiguous()
    upsampled = torch.nn.functional.interpolate(
        margin_map_float.unsqueeze(0).unsqueeze(0).float(),
        size=(target_h, target_w),
        mode=upsample_mode,
        align_corners=False,
    ).squeeze(0).squeeze(0)
    # Bicubic can introduce small negative ringing near zero-margin
    # boundary pixels; clamp_min preserves the polytope-interior invariant.
    return upsampled.clamp_min(0.0).contiguous()


def dequantize_margin_map_int8(
    int8_map: torch.Tensor,
    *,
    recovered_scale: float,
) -> torch.Tensor:
    """Inverse of :func:`quantize_margin_map_int8`.

    Args:
        int8_map: Int8 tensor packed by :func:`quantize_margin_map_int8`.
        recovered_scale: Scale recorded at encode time.

    Returns:
        Float32 margin map.

    Raises:
        ValueError: when ``recovered_scale <= 0`` (encoder-side bug; the
            archive is corrupt).
    """
    if recovered_scale <= 0:
        raise ValueError(
            f"recovered_scale must be > 0; got {recovered_scale}. The "
            "encoder side stored a non-positive margin-map scale. Refuse "
            "the archive."
        )
    return int8_map.to(torch.float32) * recovered_scale


__all__ = [
    "MARGIN_MAP_DEFAULT_RESOLUTION",
    "MARGIN_MAP_SHRUNK_RESOLUTION",
    "MarginMapMode",
    "compute_logit_margin_map",
    "compute_logit_margin_map_dummy",
    "dequantize_margin_map_int8",
    "quantize_margin_map_int8",
    "upsample_margin_map_for_overlay",
]
