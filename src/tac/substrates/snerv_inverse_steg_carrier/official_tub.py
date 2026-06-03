# SPDX-License-Identifier: MIT
"""Portable official SNeRV_T temporal-upsampling-branch input primitives.

This module mirrors the official OSS ``model/snerv_t.py`` graph-input contract
without importing torch or ``pytorch_wavelets``.  It is a NumPy-first primitive
for source-faithful TUB input preparation only:

* concatenate current/previous/next frames along batch;
* apply one-level Haar 2D DWT and keep the LF branch;
* normalize LF globally across all three LF tensors;
* build the two official temporal encoder inputs from DWT1D Haar lowpass / 2;
* expose shape metadata for the temporal encoder and ``output_2`` fusion path.

The output is parser/training substrate evidence, not contest score authority.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Final

import numpy as np

OFFICIAL_SNERV_T_SOURCE_SHA: Final[str] = "0844a08f9591eea9625f8b961ed91d08030e06d1"
OFFICIAL_SNERV_T_TUB_SCHEMA: Final[str] = "official_snerv_t_tub_numpy_graph_inputs.v1"
OFFICIAL_SNERV_T_TUB_SOURCE_CONTRACT: Final[str] = (
    "official_snerv_t_lines_125_136_and_148_150_numpy_tub_input_contract"
)


class OfficialTubError(ValueError):
    """Raised when portable official SNeRV_T TUB input preparation is invalid."""


@dataclass(frozen=True)
class OfficialOutput2FusionShape:
    """Shape contract for official ``output_2`` temporal fusion.

    Official SNeRV_T concatenates previous and next temporal encoder outputs on
    channel, splits the halves back apart, concatenates them on batch, runs a
    temporal decoder layer, then spatially shuffles by ``fc_h`` and ``fc_w``.
    """

    temporal_encoder_output_shape: tuple[int, int, int, int]
    emb_ch: int
    prev_half_shape: tuple[int, int, int, int]
    next_half_shape: tuple[int, int, int, int]
    decoder_input_shape: tuple[int, int, int, int]
    fc_hw: tuple[int, int] | None = None
    decoder_output_shape: tuple[int, int, int, int] | None = None
    fused_output2_shape: tuple[int, int, int, int] | None = None

    def as_jsonable(self) -> dict[str, object]:
        return {
            "temporal_encoder_output_shape": list(self.temporal_encoder_output_shape),
            "emb_ch": int(self.emb_ch),
            "prev_half_shape": list(self.prev_half_shape),
            "next_half_shape": list(self.next_half_shape),
            "decoder_input_shape": list(self.decoder_input_shape),
            "fc_hw": list(self.fc_hw) if self.fc_hw is not None else None,
            "decoder_output_shape": (
                list(self.decoder_output_shape)
                if self.decoder_output_shape is not None
                else None
            ),
            "fused_output2_shape": (
                list(self.fused_output2_shape)
                if self.fused_output2_shape is not None
                else None
            ),
        }


@dataclass(frozen=True)
class OfficialTubShapeMetadata:
    """Receiver-visible source-shape metadata for SNeRV_T TUB inputs."""

    source_frame_shape: tuple[int, int, int]
    source_batch_shape: tuple[int, int, int, int]
    lf_triplet_shape: tuple[int, int, int, int]
    normalized_lf_triplet_shape: tuple[int, int, int, int]
    temporal_encoder_input_shape: tuple[int, int, int, int]
    temporal_encoder_input_count: int
    temporal_encoder_concat_axis_after_encoder: int
    output2_fusion: OfficialOutput2FusionShape | None

    def as_jsonable(self) -> dict[str, object]:
        return {
            "source_frame_shape": list(self.source_frame_shape),
            "source_batch_shape": list(self.source_batch_shape),
            "lf_triplet_shape": list(self.lf_triplet_shape),
            "normalized_lf_triplet_shape": list(self.normalized_lf_triplet_shape),
            "temporal_encoder_input_shape": list(self.temporal_encoder_input_shape),
            "temporal_encoder_input_count": int(self.temporal_encoder_input_count),
            "temporal_encoder_concat_axis_after_encoder": int(
                self.temporal_encoder_concat_axis_after_encoder
            ),
            "output2_fusion": (
                self.output2_fusion.as_jsonable()
                if self.output2_fusion is not None
                else None
            ),
        }


@dataclass(frozen=True)
class OfficialTubGraphInputs:
    """Prepared official SNeRV_T TUB graph inputs.

    ``lf_triplet`` and ``normalized_lf`` use official batch order
    ``[current, previous, next]``.  The two temporal encoder inputs correspond
    to official lines 134-135 after the ``DWT1D`` lowpass branch is divided by
    two, but before the torch temporal encoder modules run.
    """

    schema: str
    source_contract: str
    score_claim: bool
    promotion_eligible: bool
    lf_triplet: np.ndarray
    normalized_lf: np.ndarray
    lf_min: float
    lf_max: float
    yl_norm: tuple[float, float]
    current_lf: np.ndarray
    prev_lowpass_over_2: np.ndarray
    next_lowpass_over_2: np.ndarray
    prev_highpass: np.ndarray
    next_highpass: np.ndarray
    temporal_encoder_inputs: tuple[np.ndarray, np.ndarray]
    shape_metadata: OfficialTubShapeMetadata

    def as_jsonable_metadata(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "source_contract": self.source_contract,
            "score_claim": bool(self.score_claim),
            "promotion_eligible": bool(self.promotion_eligible),
            "lf_min": float(self.lf_min),
            "lf_max": float(self.lf_max),
            "yl_norm": [float(self.yl_norm[0]), float(self.yl_norm[1])],
            "shape_metadata": self.shape_metadata.as_jsonable(),
        }


def prepare_official_tub_graph_inputs(
    current: np.ndarray,
    previous: np.ndarray,
    next_frame: np.ndarray,
    *,
    temporal_encoder_output_shape: tuple[int, int, int, int] | None = None,
    fc_hw: tuple[int, int] | None = None,
    output2_decoder_output_shape: tuple[int, int, int, int] | None = None,
) -> OfficialTubGraphInputs:
    """Build official SNeRV_T TUB inputs from current/previous/next frames.

    Inputs may be ``(H, W)``, ``(C, H, W)``, or ``(1, C, H, W)`` arrays.  The
    portable official path requires even spatial dimensions because the upstream
    SNeRV_T crops use even sizes and because this implementation only claims
    source-faithful Haar periodization for complete 2x2 blocks.
    """

    batch = _stack_triplet(current, previous, next_frame)
    lf_triplet = _haar_dwt2_lf_nchw(batch)
    lf_min = float(np.min(lf_triplet))
    lf_max = float(np.max(lf_triplet))
    denom = lf_max - lf_min
    if not np.isfinite(denom) or denom <= 0.0:
        raise OfficialTubError(
            "official SNeRV_T global LF normalization requires non-constant LF "
            "triplet; refusing NaN/Inf graph inputs"
        )
    normalized_lf = ((lf_triplet - lf_min) / denom).astype(np.float64)
    prev_lowpass, prev_highpass = _haar_dwt1d_pair(
        normalized_lf[0:1],
        normalized_lf[1:2],
    )
    next_lowpass, next_highpass = _haar_dwt1d_pair(
        normalized_lf[0:1],
        normalized_lf[2:3],
    )
    prev_lowpass_over_2 = (prev_lowpass / 2.0).astype(np.float64)
    next_lowpass_over_2 = (next_lowpass / 2.0).astype(np.float64)
    output2_fusion = None
    if temporal_encoder_output_shape is not None:
        output2_fusion = official_output2_fusion_shape(
            temporal_encoder_output_shape,
            fc_hw=fc_hw,
            decoder_output_shape=output2_decoder_output_shape,
        )
    metadata = OfficialTubShapeMetadata(
        source_frame_shape=tuple(int(v) for v in batch.shape[1:]),
        source_batch_shape=tuple(int(v) for v in batch.shape),
        lf_triplet_shape=tuple(int(v) for v in lf_triplet.shape),
        normalized_lf_triplet_shape=tuple(int(v) for v in normalized_lf.shape),
        temporal_encoder_input_shape=tuple(int(v) for v in prev_lowpass_over_2.shape),
        temporal_encoder_input_count=2,
        temporal_encoder_concat_axis_after_encoder=1,
        output2_fusion=output2_fusion,
    )
    return OfficialTubGraphInputs(
        schema=OFFICIAL_SNERV_T_TUB_SCHEMA,
        source_contract=OFFICIAL_SNERV_T_TUB_SOURCE_CONTRACT,
        score_claim=False,
        promotion_eligible=False,
        lf_triplet=lf_triplet,
        normalized_lf=normalized_lf,
        lf_min=lf_min,
        lf_max=lf_max,
        yl_norm=(lf_min, lf_max),
        current_lf=normalized_lf[0:1],
        prev_lowpass_over_2=prev_lowpass_over_2,
        next_lowpass_over_2=next_lowpass_over_2,
        prev_highpass=prev_highpass,
        next_highpass=next_highpass,
        temporal_encoder_inputs=(prev_lowpass_over_2, next_lowpass_over_2),
        shape_metadata=metadata,
    )


def official_output2_fusion_shape(
    temporal_encoder_output_shape: tuple[int, int, int, int],
    *,
    fc_hw: tuple[int, int] | None = None,
    decoder_output_shape: tuple[int, int, int, int] | None = None,
) -> OfficialOutput2FusionShape:
    """Return source-faithful metadata for official SNeRV_T ``output_2`` fusion."""

    n, channels, h, w = _validate_nchw_shape(
        temporal_encoder_output_shape,
        name="temporal_encoder_output_shape",
    )
    if n != 1:
        raise OfficialTubError(
            "official SNeRV_T temporal encoder output is concatenated as batch=1"
        )
    if channels % 2:
        raise OfficialTubError(
            "official SNeRV_T output_2 fusion requires even temporal channels"
        )
    emb_ch = channels // 2
    decoder_input_shape = (2, emb_ch, h, w)
    fused_shape = None
    fc_tuple = None
    decoder_tuple = None
    if fc_hw is not None:
        fc_h, fc_w = _validate_hw(fc_hw, name="fc_hw")
        fc_tuple = (fc_h, fc_w)
    if decoder_output_shape is not None:
        if fc_tuple is None:
            raise OfficialTubError("decoder_output_shape requires fc_hw")
        out_n, out_c, out_h, out_w = _validate_nchw_shape(
            decoder_output_shape,
            name="decoder_output_shape",
        )
        if out_n != 2:
            raise OfficialTubError("official output_2 decoder output batch must be 2")
        divisor = fc_tuple[0] * fc_tuple[1]
        if out_c % divisor:
            raise OfficialTubError(
                "official output_2 decoder channels must be divisible by fc_h*fc_w"
            )
        decoder_tuple = (out_n, out_c, out_h, out_w)
        fused_shape = (
            out_n,
            out_c // divisor,
            fc_tuple[0] * out_h,
            fc_tuple[1] * out_w,
        )
    return OfficialOutput2FusionShape(
        temporal_encoder_output_shape=(n, channels, h, w),
        emb_ch=emb_ch,
        prev_half_shape=(1, emb_ch, h, w),
        next_half_shape=(1, emb_ch, h, w),
        decoder_input_shape=decoder_input_shape,
        fc_hw=fc_tuple,
        decoder_output_shape=decoder_tuple,
        fused_output2_shape=fused_shape,
    )


def _stack_triplet(
    current: np.ndarray,
    previous: np.ndarray,
    next_frame: np.ndarray,
) -> np.ndarray:
    frames = [_as_chw(frame, name=name) for frame, name in (
        (current, "current"),
        (previous, "previous"),
        (next_frame, "next_frame"),
    )]
    shape = frames[0].shape
    if any(frame.shape != shape for frame in frames):
        raise OfficialTubError(
            "current/previous/next_frame must have identical CHW shapes"
        )
    _validate_even_hw((shape[1], shape[2]), name="source frame")
    return np.stack(frames, axis=0).astype(np.float64)


def _as_chw(frame: np.ndarray, *, name: str) -> np.ndarray:
    arr = np.asarray(frame, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise OfficialTubError(f"{name} contains NaN or Inf")
    if arr.ndim == 2:
        return arr[np.newaxis, :, :]
    if arr.ndim == 3:
        if arr.shape[0] < 1:
            raise OfficialTubError(f"{name} must have at least one channel")
        return arr
    if arr.ndim == 4:
        if arr.shape[0] != 1:
            raise OfficialTubError(f"{name} NCHW input must have batch=1")
        return arr[0]
    raise OfficialTubError(
        f"{name} must be shaped (H,W), (C,H,W), or (1,C,H,W); got {arr.shape}"
    )


def _haar_dwt2_lf_nchw(batch: np.ndarray) -> np.ndarray:
    arr = np.asarray(batch, dtype=np.float64)
    if arr.ndim != 4:
        raise OfficialTubError(f"batch must be NCHW; got {arr.shape}")
    _validate_even_hw((arr.shape[2], arr.shape[3]), name="source frame")
    a = arr[:, :, 0::2, 0::2]
    b = arr[:, :, 0::2, 1::2]
    c = arr[:, :, 1::2, 0::2]
    d = arr[:, :, 1::2, 1::2]
    return ((a + b + c + d) * 0.5).astype(np.float64)


def _haar_dwt1d_pair(
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if first.shape != second.shape:
        raise OfficialTubError("DWT1D pair tensors must have identical shapes")
    inv_sqrt2 = 1.0 / sqrt(2.0)
    lowpass = (np.asarray(first, dtype=np.float64) + second) * inv_sqrt2
    highpass = (np.asarray(first, dtype=np.float64) - second) * inv_sqrt2
    return lowpass.astype(np.float64), highpass.astype(np.float64)


def _validate_even_hw(hw: tuple[int, int], *, name: str) -> None:
    h, w = _validate_hw(hw, name=name)
    if h % 2 or w % 2:
        raise OfficialTubError(
            f"{name} spatial dims must be even for source-faithful Haar J=1; got {(h, w)}"
        )


def _validate_hw(hw: tuple[int, int], *, name: str) -> tuple[int, int]:
    if len(hw) != 2:
        raise OfficialTubError(f"{name} must contain exactly two dims")
    h, w = int(hw[0]), int(hw[1])
    if h < 1 or w < 1:
        raise OfficialTubError(f"{name} dims must be positive; got {(h, w)}")
    return h, w


def _validate_nchw_shape(
    shape: tuple[int, int, int, int],
    *,
    name: str,
) -> tuple[int, int, int, int]:
    if len(shape) != 4:
        raise OfficialTubError(f"{name} must be NCHW rank-4; got {shape}")
    n, c, h, w = (int(v) for v in shape)
    if min(n, c, h, w) < 1:
        raise OfficialTubError(f"{name} dims must be positive; got {shape}")
    return n, c, h, w


__all__ = [
    "OFFICIAL_SNERV_T_SOURCE_SHA",
    "OFFICIAL_SNERV_T_TUB_SCHEMA",
    "OFFICIAL_SNERV_T_TUB_SOURCE_CONTRACT",
    "OfficialOutput2FusionShape",
    "OfficialTubError",
    "OfficialTubGraphInputs",
    "OfficialTubShapeMetadata",
    "official_output2_fusion_shape",
    "prepare_official_tub_graph_inputs",
]
