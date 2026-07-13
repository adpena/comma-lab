# SPDX-License-Identifier: MIT
"""Exact tile-halo feasibility for the frozen EfficientNet-B2 U-Net SegNet.

This module is deliberately architecture-only.  It does not run a scorer and it
does not claim a timing win.  It answers the prerequisite question that must be
settled before an annulus-tile implementation is admitted: can an output logit
have a finite, smaller-than-frame input halo?

For the contest SegNet the answer is no for two independent reasons:

* every EfficientNet-B2 MBConv contains squeeze/excitation, whose spatial mean
  makes the dependency global; and
* even if squeeze/excitation is ignored, exact phase-aware propagation through
  nearest-neighbour decoder upsampling gives 1279- or 1311-pixel support.  The
  worst alignment reaches 685 pixels left and 654 right (safe symmetric halo
  685), larger than both dimensions of the 384x512 scorer input.

The calculation is executable so architecture drift fails loudly instead of
turning an old halo number into false authority.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

SCORER_HW = (384, 512)


@dataclass(frozen=True)
class ReceptiveFieldRow:
    """One input-coordinate receptive-field row.

    ``jump_px`` is the input-pixel spacing between adjacent locations at this
    resolution.  ``local_halo_px`` is the radius required if all operators were
    local.  ``global_se_blocks_seen`` records the stronger full-frame coupling.
    """

    stage: str
    operation: str
    jump_px: int
    receptive_field_px: int
    local_halo_px: int
    global_se_blocks_seen: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExactTileHaloContract:
    image_height: int
    image_width: int
    local_receptive_field_px: int
    local_halo_px: int
    squeeze_excite_blocks: int
    exact_dependency: str
    exact_source_area_fraction: float
    ideal_exact_speedup_upper_bound: float
    verdict: str
    verdict_scope: str
    reformulation_queue: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["reformulation_queue"] = list(self.reformulation_queue)
        return payload


# timm tu-efficientnet_b2 stage block kernels/strides.  Only depthwise spatial
# kernels change the receptive field; pointwise convolutions do not.
_ENCODER_STAGES: tuple[tuple[str, tuple[tuple[int, int], ...]], ...] = (
    ("encoder_stage0", ((3, 1), (3, 1))),
    ("encoder_stage1", ((3, 2), (3, 1), (3, 1))),
    ("encoder_stage2", ((5, 2), (5, 1), (5, 1))),
    ("encoder_stage3", ((3, 2), (3, 1), (3, 1), (3, 1))),
    ("encoder_stage4", ((5, 1), (5, 1), (5, 1), (5, 1))),
    ("encoder_stage5", ((5, 2), (5, 1), (5, 1), (5, 1), (5, 1))),
    ("encoder_stage6", ((3, 1), (3, 1))),
)


def _encoder_spatial_ops() -> tuple[tuple[int, int], ...]:
    return (
        (3, 2),
        *(op for _stage_name, blocks in _ENCODER_STAGES for op in blocks),
    )


def _inverse_conv_interval(
    lo: int, hi: int, *, kernel: int, stride: int
) -> tuple[int, int]:
    padding = (int(kernel) - 1) // 2
    return (
        int(lo) * int(stride) - padding,
        int(hi) * int(stride) - padding + int(kernel) - 1,
    )


def _deep_local_input_interval(
    output_index: int,
    *,
    decoder_blocks: int,
    include_head: bool,
) -> tuple[int, int]:
    """Exact 1-D deepest-path support before image-boundary clipping.

    Nearest x2 maps fine index ``q`` to coarse index ``floor(q/2)``.  Keeping
    this discrete phase is why the final answer is 1279/1311 rather than the
    approximate center-grid recurrence.
    """

    lo = hi = int(output_index)
    if include_head:
        lo, hi = _inverse_conv_interval(lo, hi, kernel=3, stride=1)
    for _ in range(int(decoder_blocks)):
        lo, hi = _inverse_conv_interval(lo, hi, kernel=3, stride=1)
        lo, hi = _inverse_conv_interval(lo, hi, kernel=3, stride=1)
        lo, hi = lo // 2, hi // 2
    for kernel, stride in reversed(_encoder_spatial_ops()):
        lo, hi = _inverse_conv_interval(
            lo, hi, kernel=kernel, stride=stride
        )
    return lo, hi


def derive_receptive_field_rows() -> tuple[ReceptiveFieldRow, ...]:
    """Derive the frozen SegNet's maximum local receptive field stage by stage.

    The standard recurrence for a convolution is ``r' = r + (k-1)j`` and
    ``j' = j*s``.  A nearest-neighbour x2 decoder block halves ``j`` and its two
    3x3 convolutions add ``4*j``.  Skip connections union a shallower support
    with the deep support; the deep support is larger at every merge, so the
    maximum below already includes the skip DAG rather than ignoring it.
    """

    rows: list[ReceptiveFieldRow] = []
    receptive_field = 1
    jump = 1
    se_seen = 0

    # EfficientNet stem: 3x3, stride 2.
    receptive_field += (3 - 1) * jump
    jump *= 2
    rows.append(
        ReceptiveFieldRow(
            stage="stem",
            operation="conv3_stride2",
            jump_px=jump,
            receptive_field_px=receptive_field,
            local_halo_px=(receptive_field - 1) // 2,
            global_se_blocks_seen=se_seen,
        )
    )

    for stage_name, blocks in _ENCODER_STAGES:
        for kernel, stride in blocks:
            receptive_field += (kernel - 1) * jump
            jump *= stride
            # Each listed MBConv has one spatial global-average SE branch.
            se_seen += 1
        rows.append(
            ReceptiveFieldRow(
                stage=stage_name,
                operation=";".join(f"mbconv_k{k}_s{s}" for k, s in blocks),
                jump_px=jump,
                receptive_field_px=receptive_field,
                local_halo_px=(receptive_field - 1) // 2,
                global_se_blocks_seen=se_seen,
            )
        )

    # Five U-Net blocks: nearest x2, concatenate the skip, conv3, conv3.  A
    # center-grid RF recurrence is not exact after nearest upsampling, so
    # enumerate the 32 output phases created by the /32 encoder.
    for index in range(5):
        if jump % 2:
            raise AssertionError(f"decoder jump {jump} cannot be halved exactly")
        jump //= 2
        intervals = [
            _deep_local_input_interval(
                phase, decoder_blocks=index + 1, include_head=False
            )
            for phase in range(64)
        ]
        widths = [hi - lo + 1 for lo, hi in intervals]
        receptive_field = max(widths)
        safe_halo = max(
            max(phase * jump - lo, hi - phase * jump)
            for phase, (lo, hi) in enumerate(intervals)
        )
        rows.append(
            ReceptiveFieldRow(
                stage=f"decoder_block{index}",
                operation="nearest_x2+skip_concat+conv3+conv3",
                jump_px=jump,
                receptive_field_px=receptive_field,
                local_halo_px=safe_halo,
                global_se_blocks_seen=se_seen,
            )
        )

    # SMP SegmentationHead: final 3x3 convolution, no upsample.
    intervals = [
        _deep_local_input_interval(phase, decoder_blocks=5, include_head=True)
        for phase in range(64)
    ]
    widths = [hi - lo + 1 for lo, hi in intervals]
    receptive_field = max(widths)
    safe_halo = max(
        max(phase - lo, hi - phase) for phase, (lo, hi) in enumerate(intervals)
    )
    rows.append(
        ReceptiveFieldRow(
            stage="segmentation_head",
            operation="conv3_stride1",
            jump_px=jump,
            receptive_field_px=receptive_field,
            local_halo_px=safe_halo,
            global_se_blocks_seen=se_seen,
        )
    )
    return tuple(rows)


def derive_exact_tile_halo_contract(
    *,
    image_hw: tuple[int, int] = SCORER_HW,
) -> ExactTileHaloContract:
    """Return the fail-closed exact sparse-forward contract.

    The ideal speedup is an upper bound that grants zero scheduling/copy cost.
    Because an exact evaluated pixel requires the full input frame, selecting a
    subset of output logits cannot reduce frozen-scorer convolutional work.
    """

    height, width = (int(image_hw[0]), int(image_hw[1]))
    if height <= 0 or width <= 0:
        raise ValueError(f"image_hw must be positive, got {image_hw!r}")
    rows = derive_receptive_field_rows()
    last = rows[-1]
    se_blocks = last.global_se_blocks_seen
    if se_blocks <= 0:
        raise AssertionError("expected at least one squeeze/excitation block")
    if last.local_halo_px < max(height, width) // 2:
        raise AssertionError(
            "frozen architecture drifted: local halo no longer independently covers the frame"
        )
    return ExactTileHaloContract(
        image_height=height,
        image_width=width,
        local_receptive_field_px=last.receptive_field_px,
        local_halo_px=last.local_halo_px,
        squeeze_excite_blocks=se_blocks,
        exact_dependency="FULL_FRAME_GLOBAL",
        exact_source_area_fraction=1.0,
        ideal_exact_speedup_upper_bound=1.0,
        verdict="NO_GO",
        verdict_scope=(
            "finite input-crop tile-with-halo exact sparse forward for the frozen "
            "tu-efficientnet_b2 SMP U-Net SegNet at 384x512"
        ),
        reformulation_queue=(
            "sparse decoder after a measured full-encoder/full-SE pass",
            "training-tolerant cached SE statistics with explicit gradient-fidelity gate",
            "local student scorer with n600 real-state Jacobian/argmax fidelity gate",
            "loss/cotangent sparsification after one dense exact forward",
        ),
    )


def inspect_torch_segnet_architecture(segnet: Any) -> dict[str, Any]:
    """Validate the instantiated upstream SegNet against the derived topology.

    Importing torch is intentionally left to the caller.  This function uses
    module metadata only and performs no forward pass.
    """

    named = list(segnet.named_modules())
    se_names = [
        name
        for name, module in named
        if type(module).__name__ in {"SqueezeExcite", "SqueezeExciteCl"}
    ]
    decoder_names = [
        name for name, module in named if type(module).__name__ == "UnetDecoderBlock"
    ]
    stem = dict(named).get("encoder.model.conv_stem")
    head = dict(named).get("segmentation_head.0")
    stem_kernel = tuple(getattr(stem, "kernel_size", ()))
    stem_stride = tuple(getattr(stem, "stride", ()))
    head_kernel = tuple(getattr(head, "kernel_size", ()))
    expected_se = sum(len(blocks) for _, blocks in _ENCODER_STAGES)
    errors: list[str] = []
    if len(se_names) != expected_se:
        errors.append(f"squeeze/excitation count {len(se_names)} != {expected_se}")
    if len(decoder_names) != 5:
        errors.append(f"decoder block count {len(decoder_names)} != 5")
    if stem_kernel != (3, 3) or stem_stride != (2, 2):
        errors.append(f"stem kernel/stride drift: {stem_kernel}/{stem_stride}")
    if head_kernel != (3, 3):
        errors.append(f"segmentation head kernel drift: {head_kernel}")
    if errors:
        raise ValueError("; ".join(errors))
    return {
        "status": "MATCHED",
        "squeeze_excite_blocks": len(se_names),
        "squeeze_excite_module_names": se_names,
        "decoder_blocks": len(decoder_names),
        "stem_kernel": list(stem_kernel),
        "stem_stride": list(stem_stride),
        "head_kernel": list(head_kernel),
    }


def cadence_from_temporal_iou(
    temporal_iou: float,
    *,
    freshness_survival_bar: float = 0.90,
) -> int:
    """Pre-register a refresh cadence from a measured temporal IoU.

    Treating IoU as the one-step survival factor, choose the largest integer
    ``K`` for which ``IoU**K >= freshness_survival_bar``.  Dynamic classes that
    cannot survive even one step refresh every step.  This is a training-path
    tolerance policy, never an exactness or authority claim.
    """

    iou = float(temporal_iou)
    bar = float(freshness_survival_bar)
    if not (0.0 < iou < 1.0):
        raise ValueError(f"temporal_iou must be in (0,1), got {iou}")
    if not (0.0 < bar < 1.0):
        raise ValueError(f"freshness_survival_bar must be in (0,1), got {bar}")
    cadence = math.floor(math.log(bar) / math.log(iou))
    return max(1, cadence)


def class_pair_cadence(
    class_ious: Iterable[tuple[str, float]],
    *,
    freshness_survival_bar: float = 0.90,
) -> dict[str, int]:
    """Return per-class refresh cadences; pair cadence is the minimum of its classes."""

    return {
        str(name): cadence_from_temporal_iou(
            float(iou), freshness_survival_bar=freshness_survival_bar
        )
        for name, iou in class_ious
    }


__all__ = [
    "SCORER_HW",
    "ExactTileHaloContract",
    "ReceptiveFieldRow",
    "cadence_from_temporal_iou",
    "class_pair_cadence",
    "derive_exact_tile_halo_contract",
    "derive_receptive_field_rows",
    "inspect_torch_segnet_architecture",
]
