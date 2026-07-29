# SPDX-License-Identifier: MIT
"""ddm_xi1 — carried-ξ warp INTER predictor as ONE MORE context/expert on SMEVR.

HOTZ CONSTRAINT (binding): this module does NOT build a new codec.  It EXTENDS
the landed SMEVR coder (``experiments/ddm_r7_token_coder.py``) by importing its
exact arithmetic coder, KT rescale, residual-rank map, and channel views, then
wires a *carried-ξ warped previous token field* in as one additional coding
context (Variant A) or as an innovation alphabet (Variant B).  The counted object
is unchanged: the quantized ``[pair, row, column, channel]`` token array.

The warped previous field is a DECODER-DERIVED context (rule 118):

  * the warp is a generic deterministic numpy ground-homography, a token-grid
    rescale of the vendored engine in
    ``tools/pfs1_recompose_warp_base_and_eval.py`` (EON intrinsics 910/582/437 at
    full frame 1164x874, camera height 1.22 m — the documented ``tac.clip_profile``
    literals);
  * the CARRIED pose ``t_p`` (600x6 float16 target) + per-pair ``s_t`` index are
    the archive's OWN pose-member payload (pfs1 grammar v3), already counted for
    frame_0; using them additionally as token context adds ZERO token bytes.

Therefore the token-frame win (if any) is real bytes saved in the token member at
zero marginal pose cost — but the carried-ξ token predictor and the pfs1 pose
carrier draw from the SAME t_p/s_t pool: they COMPETE for credit, never sum.

No scorer, evaluator, checkpoint value, or fitted histogram is embedded in this
decoder.  Every frame decodes to the exact input and re-encodes byte-identically
before it is admitted.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from typing import Final

import numpy as np

# HOTZ CONSTRAINT: import (do not fork) the landed SMEVR primitives.
from ddm_r7_token_coder import (  # type: ignore[import-not-found]
    DDMR7CoderError,
    _ArithmeticDecoder,
    _ArithmeticEncoder,
    _channel_views,
    _codes,
    _rank_to_residual,
    _raw_lzma_decode,
    _raw_lzma_encode,
    _residual_to_rank,
    _row,
    _SMEVR_LENGTH,
    _update,
    factor_mode_delta,
    pack_nibbles,
    reconstruct_mode_delta,
    unpack_nibbles,
)

MAGIC: Final = b"DX1T"
VERSION: Final = 1
VARIANT_IDS: Final = {
    "smevr_warp_context": 1,
    "smevr_warp_innovation": 2,
}
ID_VARIANTS: Final = {value: key for key, value in VARIANT_IDS.items()}
# 4s magic, B version, B variant, B levels, B ndim, 4H shape, I base_len, I delta_len,
# B warp_direction (0 fwd / 1 bwd), 32s digest
HEADER: Final = struct.Struct("<4sBBBB4HIIB32s")

# --- documented EON / openpilot literals (MEASURED anchors; clip_profile reproduces) ---
_NATIVE_FX: Final = 910.0
_NATIVE_FY: Final = 910.0
_NATIVE_CX: Final = 582.0
_NATIVE_CY: Final = 437.0
_CAMERA_HEIGHT_M: Final = 1.22
_FULL_FRAME_W: Final = 1164.0
_FULL_FRAME_H: Final = 874.0
# the pfs1 grammar-v3 translation-scale grid (indexed by the carried s_t stream)
ST_GRID: Final = (0.0, 0.005, 0.01, 0.02, 0.03, 0.044, 0.06, 0.08, 0.12, 0.16, 0.24)


@dataclass(frozen=True, slots=True)
class XiFrameAccounting:
    variant: str
    framed_bytes: int
    header_bytes: int
    base_bytes: int
    delta_bytes: int
    warp_direction: str
    raw_token_bytes: int
    sha256: str


def _semantic_digest(
    variant_id: int,
    levels: int,
    shape: tuple[int, int, int, int],
    raw: bytes,
) -> bytes:
    metadata = MAGIC + struct.pack("<BBBB4H", VERSION, variant_id, levels, 4, *shape)
    return hashlib.sha256(metadata + raw).digest()


# --------------------------------------------------------------------------- #
# generic token-grid warp (rule 118: deterministic numpy; no video-derived table)
# --------------------------------------------------------------------------- #
def _token_grid_intrinsics(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    scale_x = width / _FULL_FRAME_W
    scale_y = height / _FULL_FRAME_H
    K = np.array(
        [
            [_NATIVE_FX * scale_x, 0.0, _NATIVE_CX * scale_x],
            [0.0, _NATIVE_FY * scale_y, _NATIVE_CY * scale_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return K, np.linalg.inv(K)


def _target_grid(height: int, width: int) -> np.ndarray:
    us, vs = np.meshgrid(np.arange(width), np.arange(height))
    return np.stack([us.ravel(), vs.ravel(), np.ones(height * width)], 0).astype(np.float64)


def _ground_homography(pose6: np.ndarray, K: np.ndarray, Kinv: np.ndarray, s_t: float) -> np.ndarray:
    # ground-plane homography, rotation identity (s_r=0), pitch 0 -> n = [0,-1,0].
    t = s_t * np.array([pose6[2], pose6[1], pose6[0]], dtype=np.float64)
    n = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    M = np.eye(3) - np.outer(t, n) / _CAMERA_HEIGHT_M
    return K @ M @ Kinv


def _warp_channel(
    prev_hw: np.ndarray,
    H_mat: np.ndarray,
    grid: np.ndarray,
    height: int,
    width: int,
    *,
    forward: bool,
) -> np.ndarray:
    """Bilinear-warp one (H,W) integer-code plane by the ground homography.

    forward=True predicts the LATER pair from the EARLIER one (sample at H@grid);
    forward=False uses the inverse (sample at Hinv@grid).  Out-of-frame targets
    fall back to the co-located previous value (the identity predictor there).
    """
    M = H_mat if forward else np.linalg.inv(H_mat)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        src = M @ grid
        z = src[2]
        su = src[0] / z
        sv = src[1] / z
    valid = (
        np.isfinite(su)
        & np.isfinite(sv)
        & (z > 0)
        & (su >= 0)
        & (su <= width - 1)
        & (sv >= 0)
        & (sv <= height - 1)
    )
    suc = np.clip(su, 0.0, width - 1)
    svc = np.clip(sv, 0.0, height - 1)
    x0 = np.floor(suc).astype(np.int64)
    y0 = np.floor(svc).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = suc - x0
    wy = svc - y0
    flat = prev_hw.reshape(-1).astype(np.float64)
    Ia = flat[y0 * width + x0]
    Ib = flat[y0 * width + x1]
    Ic = flat[y1 * width + x0]
    Id = flat[y1 * width + x1]
    top = Ia * (1.0 - wx) + Ib * wx
    bot = Ic * (1.0 - wx) + Id * wx
    sampled = top * (1.0 - wy) + bot * wy
    out = np.where(valid, sampled, flat)
    return out.reshape(height, width)


class _WarpPredictor:
    """Per-channel causal carried-ξ predictor of the current pair's code plane.

    Given the already-reconstructed previous pair's code plane, predict the
    current pair's code plane by the carried ground-homography warp.  Matches the
    SMEVR channel-outer / pair-inner iteration: pair i-1 is complete before pair i.
    """

    def __init__(
        self,
        base_channel_hw: np.ndarray,
        tp: np.ndarray,
        st_idx: np.ndarray,
        height: int,
        width: int,
        levels: int,
        *,
        forward: bool,
    ) -> None:
        self.base = base_channel_hw.astype(np.int16)
        self.tp = np.asarray(tp, dtype=np.float64)
        self.st_idx = np.asarray(st_idx, dtype=np.int64)
        self.height = height
        self.width = width
        self.levels = levels
        self.forward = forward
        self.K, self.Kinv = _token_grid_intrinsics(height, width)
        self.grid = _target_grid(height, width)

    def predict(self, pair: int, prev_code_hw: np.ndarray | None) -> np.ndarray:
        """Predicted current-pair CODE plane (H,W) int16 in [0,levels)."""
        if pair == 0 or prev_code_hw is None:
            return self.base.copy()  # neutral: the temporal mode
        s_t = float(ST_GRID[int(self.st_idx[pair])])
        H_mat = _ground_homography(self.tp[pair], self.K, self.Kinv, s_t)
        warped = _warp_channel(
            prev_code_hw.astype(np.float64),
            H_mat,
            self.grid,
            self.height,
            self.width,
            forward=self.forward,
        )
        pred = np.clip(np.rint(warped), 0, self.levels - 1).astype(np.int16)
        return pred


def _pred_delta(pred_code_hw: np.ndarray, base_hw: np.ndarray, levels: int) -> np.ndarray:
    return ((pred_code_hw - base_hw.astype(np.int16)) % levels).astype(np.int16)


# --------------------------------------------------------------------------- #
# Variant A — SMEVR + carried-ξ warp context ("one more expert")
# --------------------------------------------------------------------------- #
def _encode_warp_context(
    base: np.ndarray,
    delta: np.ndarray,
    levels: int,
    tp: np.ndarray,
    st_idx: np.ndarray,
    *,
    forward: bool,
) -> bytes:
    pairs, height, width, channels = delta.shape
    frame_values = height * width
    sources = _channel_views(delta)
    codes = reconstruct_mode_delta(base, delta, levels)
    code_views = _channel_views(np.ascontiguousarray(codes))
    bases = np.ascontiguousarray(base.transpose(2, 0, 1)).reshape(channels, -1)
    base_planes = np.ascontiguousarray(base.transpose(2, 0, 1))  # (C,H,W)
    occupancy_encoder = _ArithmeticEncoder()
    value_encoder = _ArithmeticEncoder()
    occupancy_rows: dict[int, list[int]] = {}
    value_rows: dict[int, list[int]] = {}
    for channel in range(channels):
        source = sources[channel]
        code_source = code_views[channel]
        base_channel = bases[channel]
        predictor = _WarpPredictor(
            base_planes[channel], tp, st_idx, height, width, levels, forward=forward
        )
        ages = np.zeros(frame_values, dtype=np.uint8)
        pred_delta_flat = np.zeros(frame_values, dtype=np.int16)
        for index, raw_value in enumerate(source):
            cell = index % frame_values
            if cell == 0:
                pair = index // frame_values
                prev = (
                    code_source[(pair - 1) * frame_values : pair * frame_values].reshape(height, width)
                    if pair > 0
                    else None
                )
                pred_code = predictor.predict(pair, prev)
                pred_delta_flat = _pred_delta(
                    pred_code, base_planes[channel], levels
                ).reshape(-1)
            row, column = divmod(cell, width)
            previous_value = int(source[index - frame_values]) if index >= frame_values else 0
            previous_occupancy = int(previous_value != 0)
            left_occupancy = int(source[index - 1] != 0) if column else 0
            upper_occupancy = int(source[index - width] != 0) if row else 0
            age_bucket = min(int(ages[cell]), 3)
            warp_pred = int(pred_delta_flat[cell])
            warp_occ = int(warp_pred != 0)
            context = channel
            for value, radix in (
                (int(base_channel[cell]), levels),
                (previous_occupancy, 2),
                (left_occupancy, 2),
                (upper_occupancy, 2),
                (age_bucket, 4),
                (warp_occ, 2),  # carried-ξ expert: does the warp predict a change?
            ):
                context = context * radix + value
            occupancy = int(raw_value != 0)
            counts = _row(occupancy_rows, context, 2)
            occupancy_encoder.encode(occupancy, counts)
            _update(counts, occupancy)
            if occupancy:
                value_context = (
                    (channel * levels + int(base_channel[cell])) * (levels + 1) + previous_value
                ) * levels + warp_pred  # carried-ξ expert on the value model
                value_counts = _row(value_rows, value_context, levels - 1)
                rank = _residual_to_rank(int(raw_value), levels)
                value_encoder.encode(rank, value_counts)
                _update(value_counts, rank)
            if occupancy == previous_occupancy:
                ages[cell] = min(int(ages[cell]) + 1, 255)
            else:
                ages[cell] = 0
    occupancy_stream = occupancy_encoder.finish()
    value_stream = value_encoder.finish()
    return _SMEVR_LENGTH.pack(len(occupancy_stream)) + occupancy_stream + value_stream


def _decode_warp_context(
    payload: bytes,
    base: np.ndarray,
    shape: tuple[int, int, int, int],
    levels: int,
    tp: np.ndarray,
    st_idx: np.ndarray,
    *,
    forward: bool,
) -> np.ndarray:
    if len(payload) <= _SMEVR_LENGTH.size:
        raise DDMR7CoderError("xi1 warp-context stream is truncated")
    (occupancy_length,) = _SMEVR_LENGTH.unpack_from(payload)
    if occupancy_length <= 0 or _SMEVR_LENGTH.size + occupancy_length >= len(payload):
        raise DDMR7CoderError("xi1 warp-context stream lengths differ")
    occupancy_decoder = _ArithmeticDecoder(
        payload[_SMEVR_LENGTH.size : _SMEVR_LENGTH.size + occupancy_length]
    )
    value_decoder = _ArithmeticDecoder(payload[_SMEVR_LENGTH.size + occupancy_length :])
    pairs, height, width, channels = shape
    frame_values = height * width
    delta_out = np.zeros((channels, math.prod(shape[:-1])), dtype=np.uint8)
    code_out = np.zeros((channels, math.prod(shape[:-1])), dtype=np.uint8)
    bases = np.ascontiguousarray(base.transpose(2, 0, 1)).reshape(channels, -1)
    base_planes = np.ascontiguousarray(base.transpose(2, 0, 1))
    occupancy_rows: dict[int, list[int]] = {}
    value_rows: dict[int, list[int]] = {}
    for channel in range(channels):
        target = delta_out[channel]
        code_target = code_out[channel]
        base_channel = bases[channel]
        predictor = _WarpPredictor(
            base_planes[channel], tp, st_idx, height, width, levels, forward=forward
        )
        ages = np.zeros(frame_values, dtype=np.uint8)
        pred_delta_flat = np.zeros(frame_values, dtype=np.int16)
        for index in range(target.size):
            cell = index % frame_values
            if cell == 0:
                pair = index // frame_values
                prev = (
                    code_target[(pair - 1) * frame_values : pair * frame_values].reshape(height, width)
                    if pair > 0
                    else None
                )
                pred_code = predictor.predict(pair, prev)
                pred_delta_flat = _pred_delta(
                    pred_code, base_planes[channel], levels
                ).reshape(-1)
            row, column = divmod(cell, width)
            previous_value = int(target[index - frame_values]) if index >= frame_values else 0
            previous_occupancy = int(previous_value != 0)
            left_occupancy = int(target[index - 1] != 0) if column else 0
            upper_occupancy = int(target[index - width] != 0) if row else 0
            age_bucket = min(int(ages[cell]), 3)
            warp_pred = int(pred_delta_flat[cell])
            warp_occ = int(warp_pred != 0)
            context = channel
            for value, radix in (
                (int(base_channel[cell]), levels),
                (previous_occupancy, 2),
                (left_occupancy, 2),
                (upper_occupancy, 2),
                (age_bucket, 4),
                (warp_occ, 2),
            ):
                context = context * radix + value
            counts = _row(occupancy_rows, context, 2)
            occupancy = occupancy_decoder.decode(counts)
            _update(counts, occupancy)
            if occupancy:
                value_context = (
                    (channel * levels + int(base_channel[cell])) * (levels + 1) + previous_value
                ) * levels + warp_pred
                value_counts = _row(value_rows, value_context, levels - 1)
                rank = value_decoder.decode(value_counts)
                _update(value_counts, rank)
                target[index] = _rank_to_residual(rank, levels)
            code_target[index] = (int(base_channel[cell]) + int(target[index])) % levels
            if occupancy == previous_occupancy:
                ages[cell] = min(int(ages[cell]) + 1, 255)
            else:
                ages[cell] = 0
    return np.ascontiguousarray(
        delta_out.reshape(channels, *shape[:-1]).transpose(1, 2, 3, 0)
    )


# --------------------------------------------------------------------------- #
# Variant B — carried-ξ innovation alphabet: code (code - warp_pred) mod levels
# --------------------------------------------------------------------------- #
def _encode_warp_innovation(
    base: np.ndarray,
    delta: np.ndarray,
    levels: int,
    tp: np.ndarray,
    st_idx: np.ndarray,
    *,
    forward: bool,
) -> bytes:
    pairs, height, width, channels = delta.shape
    frame_values = height * width
    codes = reconstruct_mode_delta(base, delta, levels)
    code_views = _channel_views(np.ascontiguousarray(codes))
    bases = np.ascontiguousarray(base.transpose(2, 0, 1)).reshape(channels, -1)
    base_planes = np.ascontiguousarray(base.transpose(2, 0, 1))
    occupancy_encoder = _ArithmeticEncoder()
    value_encoder = _ArithmeticEncoder()
    occupancy_rows: dict[int, list[int]] = {}
    value_rows: dict[int, list[int]] = {}
    for channel in range(channels):
        code_source = code_views[channel]
        base_channel = bases[channel]
        predictor = _WarpPredictor(
            base_planes[channel], tp, st_idx, height, width, levels, forward=forward
        )
        ages = np.zeros(frame_values, dtype=np.uint8)
        innov_prev = np.zeros(frame_values, dtype=np.int16)
        innov_cur = np.zeros(frame_values, dtype=np.int16)
        pred_flat = np.zeros(frame_values, dtype=np.int16)
        for index in range(code_source.size):
            cell = index % frame_values
            if cell == 0:
                pair = index // frame_values
                innov_prev = innov_cur.copy()
                innov_cur = np.zeros(frame_values, dtype=np.int16)
                prev = (
                    code_source[(pair - 1) * frame_values : pair * frame_values].reshape(height, width)
                    if pair > 0
                    else None
                )
                pred_code = predictor.predict(pair, prev)
                pred_flat = pred_code.reshape(-1)
            row, column = divmod(cell, width)
            innov = (int(code_source[index]) - int(pred_flat[cell])) % levels
            innov_cur[cell] = innov
            previous_innov = int(innov_prev[cell])
            previous_occupancy = int(previous_innov != 0)
            left_occupancy = int(innov_cur[cell - 1] != 0) if column else 0
            upper_occupancy = int(innov_cur[cell - width] != 0) if row else 0
            age_bucket = min(int(ages[cell]), 3)
            context = channel
            for value, radix in (
                (int(base_channel[cell]), levels),
                (previous_occupancy, 2),
                (left_occupancy, 2),
                (upper_occupancy, 2),
                (age_bucket, 4),
            ):
                context = context * radix + value
            occupancy = int(innov != 0)
            counts = _row(occupancy_rows, context, 2)
            occupancy_encoder.encode(occupancy, counts)
            _update(counts, occupancy)
            if occupancy:
                value_context = (
                    channel * levels + int(base_channel[cell])
                ) * (levels + 1) + previous_innov
                value_counts = _row(value_rows, value_context, levels - 1)
                rank = _residual_to_rank(innov, levels)
                value_encoder.encode(rank, value_counts)
                _update(value_counts, rank)
            if occupancy == previous_occupancy:
                ages[cell] = min(int(ages[cell]) + 1, 255)
            else:
                ages[cell] = 0
    occupancy_stream = occupancy_encoder.finish()
    value_stream = value_encoder.finish()
    return _SMEVR_LENGTH.pack(len(occupancy_stream)) + occupancy_stream + value_stream


def _decode_warp_innovation(
    payload: bytes,
    base: np.ndarray,
    shape: tuple[int, int, int, int],
    levels: int,
    tp: np.ndarray,
    st_idx: np.ndarray,
    *,
    forward: bool,
) -> np.ndarray:
    if len(payload) <= _SMEVR_LENGTH.size:
        raise DDMR7CoderError("xi1 innovation stream is truncated")
    (occupancy_length,) = _SMEVR_LENGTH.unpack_from(payload)
    if occupancy_length <= 0 or _SMEVR_LENGTH.size + occupancy_length >= len(payload):
        raise DDMR7CoderError("xi1 innovation stream lengths differ")
    occupancy_decoder = _ArithmeticDecoder(
        payload[_SMEVR_LENGTH.size : _SMEVR_LENGTH.size + occupancy_length]
    )
    value_decoder = _ArithmeticDecoder(payload[_SMEVR_LENGTH.size + occupancy_length :])
    pairs, height, width, channels = shape
    frame_values = height * width
    code_out = np.zeros((channels, math.prod(shape[:-1])), dtype=np.uint8)
    delta_out = np.zeros((channels, math.prod(shape[:-1])), dtype=np.uint8)
    bases = np.ascontiguousarray(base.transpose(2, 0, 1)).reshape(channels, -1)
    base_planes = np.ascontiguousarray(base.transpose(2, 0, 1))
    occupancy_rows: dict[int, list[int]] = {}
    value_rows: dict[int, list[int]] = {}
    for channel in range(channels):
        code_target = code_out[channel]
        delta_target = delta_out[channel]
        base_channel = bases[channel]
        predictor = _WarpPredictor(
            base_planes[channel], tp, st_idx, height, width, levels, forward=forward
        )
        ages = np.zeros(frame_values, dtype=np.uint8)
        innov_prev = np.zeros(frame_values, dtype=np.int16)
        innov_cur = np.zeros(frame_values, dtype=np.int16)
        pred_flat = np.zeros(frame_values, dtype=np.int16)
        for index in range(code_target.size):
            cell = index % frame_values
            if cell == 0:
                pair = index // frame_values
                innov_prev = innov_cur.copy()
                innov_cur = np.zeros(frame_values, dtype=np.int16)
                prev = (
                    code_target[(pair - 1) * frame_values : pair * frame_values].reshape(height, width)
                    if pair > 0
                    else None
                )
                pred_code = predictor.predict(pair, prev)
                pred_flat = pred_code.reshape(-1)
            row, column = divmod(cell, width)
            previous_innov = int(innov_prev[cell])
            previous_occupancy = int(previous_innov != 0)
            left_occupancy = int(innov_cur[cell - 1] != 0) if column else 0
            upper_occupancy = int(innov_cur[cell - width] != 0) if row else 0
            age_bucket = min(int(ages[cell]), 3)
            context = channel
            for value, radix in (
                (int(base_channel[cell]), levels),
                (previous_occupancy, 2),
                (left_occupancy, 2),
                (upper_occupancy, 2),
                (age_bucket, 4),
            ):
                context = context * radix + value
            counts = _row(occupancy_rows, context, 2)
            occupancy = occupancy_decoder.decode(counts)
            _update(counts, occupancy)
            innov = 0
            if occupancy:
                value_context = (
                    channel * levels + int(base_channel[cell])
                ) * (levels + 1) + previous_innov
                value_counts = _row(value_rows, value_context, levels - 1)
                rank = value_decoder.decode(value_counts)
                _update(value_counts, rank)
                innov = _rank_to_residual(rank, levels)
            innov_cur[cell] = innov
            code_val = (int(pred_flat[cell]) + innov) % levels
            code_target[index] = code_val
            delta_target[index] = (code_val - int(base_channel[cell])) % levels
            if occupancy == previous_occupancy:
                ages[cell] = min(int(ages[cell]) + 1, 255)
            else:
                ages[cell] = 0
    return np.ascontiguousarray(
        delta_out.reshape(channels, *shape[:-1]).transpose(1, 2, 3, 0)
    )


# --------------------------------------------------------------------------- #
# framing
# --------------------------------------------------------------------------- #
def encode_token_codes(
    codes: np.ndarray,
    tp: np.ndarray,
    st_idx: np.ndarray,
    *,
    levels: int = 16,
    variant: str = "smevr_warp_context",
    forward: bool = False,
) -> bytes:
    """Encode one carried-ξ token frame.  tp/st_idx are decoder-side side info."""

    value = _codes(codes, levels)
    if variant not in VARIANT_IDS:
        raise DDMR7CoderError(f"unsupported xi1 variant: {variant}")
    pairs = value.shape[0]
    tp = np.asarray(tp, dtype=np.float64).reshape(pairs, 6)
    st_idx = np.asarray(st_idx, dtype=np.int64).reshape(pairs)
    if np.any(st_idx < 0) or np.any(st_idx >= len(ST_GRID)):
        raise DDMR7CoderError("s_t index outside the carried grid")
    base, delta = factor_mode_delta(value, levels)
    base_stream = _raw_lzma_encode(pack_nibbles(base))
    if variant == "smevr_warp_context":
        delta_stream = _encode_warp_context(base, delta, levels, tp, st_idx, forward=forward)
    else:
        delta_stream = _encode_warp_innovation(base, delta, levels, tp, st_idx, forward=forward)
    header = HEADER.pack(
        MAGIC,
        VERSION,
        VARIANT_IDS[variant],
        levels,
        value.ndim,
        *value.shape,
        len(base_stream),
        len(delta_stream),
        1 if not forward else 0,
        _semantic_digest(VARIANT_IDS[variant], levels, tuple(value.shape), value.tobytes()),
    )
    return header + base_stream + delta_stream


def _decode_token_codes(
    frame: bytes,
    tp: np.ndarray,
    st_idx: np.ndarray,
    *,
    canonical: bool,
) -> np.ndarray:
    if len(frame) < HEADER.size:
        raise DDMR7CoderError("xi1 token frame is truncated")
    (
        magic,
        version,
        variant_id,
        levels,
        rank,
        pair_count,
        height,
        width,
        channels,
        base_length,
        delta_length,
        direction_byte,
        digest,
    ) = HEADER.unpack_from(frame)
    if magic != MAGIC or version != VERSION or rank != 4:
        raise DDMR7CoderError("xi1 frame magic/version/rank differs")
    variant = ID_VARIANTS.get(variant_id)
    if variant is None or not (2 <= levels <= 16):
        raise DDMR7CoderError("xi1 frame variant/levels differs")
    forward = direction_byte == 0
    shape = (pair_count, height, width, channels)
    count = math.prod(shape)
    if count <= 0:
        raise DDMR7CoderError("xi1 frame shape is outside bounds")
    expected = HEADER.size + base_length + delta_length
    if base_length <= 0 or delta_length <= 0 or len(frame) != expected:
        raise DDMR7CoderError("xi1 frame lengths do not close")
    base_payload = frame[HEADER.size : HEADER.size + base_length]
    delta_payload = frame[HEADER.size + base_length :]
    base_count = height * width * channels
    base = unpack_nibbles(
        _raw_lzma_decode(base_payload, expected_length=(base_count + 1) // 2),
        base_count,
    ).reshape(height, width, channels)
    if np.any(base >= levels):
        raise DDMR7CoderError("xi1 mode base exceeds levels")
    tp = np.asarray(tp, dtype=np.float64).reshape(pair_count, 6)
    st_idx = np.asarray(st_idx, dtype=np.int64).reshape(pair_count)
    if variant == "smevr_warp_context":
        delta = _decode_warp_context(
            delta_payload, base, shape, levels, tp, st_idx, forward=forward
        )
    else:
        delta = _decode_warp_innovation(
            delta_payload, base, shape, levels, tp, st_idx, forward=forward
        )
    if delta.shape != shape or np.any(delta >= levels):
        raise DDMR7CoderError("xi1 decoded residual differs from declared lattice")
    restored = reconstruct_mode_delta(base, delta, levels)
    expected_digest = _semantic_digest(variant_id, levels, shape, restored.tobytes())
    if expected_digest != digest:
        raise DDMR7CoderError("xi1 decoded token semantic SHA-256 differs")
    if canonical and encode_token_codes(
        restored, tp, st_idx, levels=levels, variant=variant, forward=forward
    ) != frame:
        raise DDMR7CoderError("xi1 token frame is noncanonical or has inert bytes")
    return restored


def decode_token_codes(frame: bytes, tp: np.ndarray, st_idx: np.ndarray) -> np.ndarray:
    """Decode and canonically re-encode one carried-ξ token frame."""

    try:
        return _decode_token_codes(bytes(frame), tp, st_idx, canonical=True)
    except (OverflowError, struct.error) as exc:
        raise DDMR7CoderError("xi1 token frame structure is invalid") from exc


def frame_accounting(frame: bytes) -> XiFrameAccounting:
    if len(frame) < HEADER.size:
        raise DDMR7CoderError("xi1 token frame is truncated")
    unpacked = HEADER.unpack_from(frame)
    variant = ID_VARIANTS.get(unpacked[2])
    if variant is None:
        raise DDMR7CoderError("xi1 token frame variant differs")
    shape = tuple(int(value) for value in unpacked[5:9])
    return XiFrameAccounting(
        variant=variant,
        framed_bytes=len(frame),
        header_bytes=HEADER.size,
        base_bytes=int(unpacked[9]),
        delta_bytes=int(unpacked[10]),
        warp_direction="forward" if int(unpacked[11]) == 0 else "backward",
        raw_token_bytes=math.prod(shape),
        sha256=hashlib.sha256(frame).hexdigest(),
    )


__all__ = [
    "HEADER",
    "MAGIC",
    "ST_GRID",
    "VARIANT_IDS",
    "XiFrameAccounting",
    "decode_token_codes",
    "encode_token_codes",
    "frame_accounting",
]
