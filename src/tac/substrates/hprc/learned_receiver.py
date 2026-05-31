# SPDX-License-Identifier: MIT
"""Numpy-only compact learned receiver for HPRC packets.

This is the first semantic HPRC receiver mode. It is deliberately small:
archive-contained decoder weights produce a low-resolution learned base, pair
latents modulate that base, selector-weighted residual tokens repair block
structure, and receiver state contributes deterministic temporal bias. The
runtime stays decode-only and scorer-free.
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from typing import Any

import numpy as np

from tac.substrates.hprc.archive import (
    HprcPacket,
    HprcPacketConfig,
    HprcSectionKind,
    pack_hprc_packet,
)

COMPACT_RECEIVER_MODE = "compact_numpy_receiver_v1"
COMPACT_NUMPY_DECODER_FAMILY_ID = 1201
COMPACT_RGB_COLOR_TRANSFORM_ID = 1

_DECODER_MAGIC = b"HPRCDEC1"
_LATENT_MAGIC = b"HPRCLAT1"
_SELECTOR_MAGIC = b"HPRCSEL1"
_RESIDUAL_MAGIC = b"HPRCRES1"
_STATE_MAGIC = b"HPRCSTA1"
_VERSION = 1

_DECODER_HEADER_FMT = "<8sBHHBHf"
_LATENT_HEADER_FMT = "<8sBHHf"
_SELECTOR_HEADER_FMT = "<8sBH"
_RESIDUAL_HEADER_FMT = "<8sBHHHBf"
_STATE_HEADER_FMT = "<8sBHBf"

_DECODER_HEADER_SIZE = struct.calcsize(_DECODER_HEADER_FMT)
_LATENT_HEADER_SIZE = struct.calcsize(_LATENT_HEADER_FMT)
_SELECTOR_HEADER_SIZE = struct.calcsize(_SELECTOR_HEADER_FMT)
_RESIDUAL_HEADER_SIZE = struct.calcsize(_RESIDUAL_HEADER_FMT)
_STATE_HEADER_SIZE = struct.calcsize(_STATE_HEADER_FMT)


class HprcCompactReceiverError(ValueError):
    """Raised when compact receiver sections are malformed."""


@dataclass(frozen=True)
class CompactDecoder:
    height: int
    width: int
    channels: int
    basis_count: int
    basis_scale: float
    mean: np.ndarray
    basis_q: np.ndarray


@dataclass(frozen=True)
class CompactLatents:
    frames: int
    basis_count: int
    scale: float
    q: np.ndarray


@dataclass(frozen=True)
class CompactSelectors:
    frames: int
    values: np.ndarray


@dataclass(frozen=True)
class CompactResidual:
    frames: int
    grid_h: int
    grid_w: int
    channels: int
    scale: float
    q: np.ndarray


@dataclass(frozen=True)
class CompactReceiverState:
    pairs: int
    dims: int
    scale: float
    q: np.ndarray


@dataclass(frozen=True)
class CompactReceiverPacket:
    packet: HprcPacket
    manifest: dict[str, Any]
    rdo_plan: dict[str, Any]
    decoder: CompactDecoder
    latents: CompactLatents
    selectors: CompactSelectors
    residual: CompactResidual
    receiver_state: CompactReceiverState


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )


def _loads_json(payload: bytes, *, section: str) -> dict[str, Any]:
    try:
        out = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HprcCompactReceiverError(f"{section} is not valid JSON") from exc
    if not isinstance(out, dict):
        raise HprcCompactReceiverError(f"{section} JSON root must be an object")
    return out


def _check_magic(magic: bytes, expected: bytes, section: str) -> None:
    if magic != expected:
        raise HprcCompactReceiverError(f"{section} magic mismatch: {magic!r}")


def _require_exact_length(actual: int, expected: int, section: str) -> None:
    if actual != expected:
        raise HprcCompactReceiverError(
            f"{section} length mismatch: expected {expected}, got {actual}"
        )


def pack_compact_decoder(mean: np.ndarray, basis: np.ndarray) -> bytes:
    mean_u8 = np.asarray(mean, dtype=np.uint8)
    if mean_u8.ndim != 3:
        raise HprcCompactReceiverError("mean must be HxWxC")
    if mean_u8.shape[2] != 3:
        raise HprcCompactReceiverError("compact receiver currently requires RGB mean")
    basis_f = np.asarray(basis, dtype=np.float32)
    if basis_f.ndim != 4:
        raise HprcCompactReceiverError("basis must be KxHxWxC")
    if tuple(basis_f.shape[1:]) != tuple(mean_u8.shape):
        raise HprcCompactReceiverError("basis spatial shape must match mean")
    basis_count = int(basis_f.shape[0])
    max_abs = float(np.max(np.abs(basis_f))) if basis_f.size else 0.0
    basis_scale = max(max_abs / 127.0, 1.0 / 127.0)
    basis_q = np.rint(basis_f / basis_scale).clip(-127, 127).astype(np.int8)
    header = struct.pack(
        _DECODER_HEADER_FMT,
        _DECODER_MAGIC,
        _VERSION,
        int(mean_u8.shape[0]),
        int(mean_u8.shape[1]),
        int(mean_u8.shape[2]),
        basis_count,
        basis_scale,
    )
    return header + mean_u8.tobytes(order="C") + basis_q.tobytes(order="C")


def unpack_compact_decoder(payload: bytes) -> CompactDecoder:
    if len(payload) < _DECODER_HEADER_SIZE:
        raise HprcCompactReceiverError("decoder_qw truncated before header")
    magic, version, height, width, channels, basis_count, basis_scale = struct.unpack(
        _DECODER_HEADER_FMT, payload[:_DECODER_HEADER_SIZE]
    )
    _check_magic(magic, _DECODER_MAGIC, "decoder_qw")
    if version != _VERSION:
        raise HprcCompactReceiverError(f"decoder_qw version mismatch: {version}")
    mean_len = int(height) * int(width) * int(channels)
    basis_len = int(basis_count) * mean_len
    _require_exact_length(len(payload), _DECODER_HEADER_SIZE + mean_len + basis_len, "decoder_qw")
    mean = np.frombuffer(
        payload[_DECODER_HEADER_SIZE : _DECODER_HEADER_SIZE + mean_len], dtype=np.uint8
    ).reshape((height, width, channels))
    basis_q = np.frombuffer(
        payload[_DECODER_HEADER_SIZE + mean_len :], dtype=np.int8
    ).reshape((basis_count, height, width, channels))
    return CompactDecoder(
        height=int(height),
        width=int(width),
        channels=int(channels),
        basis_count=int(basis_count),
        basis_scale=float(basis_scale),
        mean=mean,
        basis_q=basis_q,
    )


def pack_compact_latents(latents: np.ndarray) -> bytes:
    latents_f = np.asarray(latents, dtype=np.float32)
    if latents_f.ndim != 2:
        raise HprcCompactReceiverError("latents must be frames x basis_count")
    max_abs = float(np.max(np.abs(latents_f))) if latents_f.size else 0.0
    scale = max(max_abs / 32767.0, 1.0)
    q = np.rint(latents_f / scale).clip(-32767, 32767).astype("<i2")
    header = struct.pack(
        _LATENT_HEADER_FMT,
        _LATENT_MAGIC,
        _VERSION,
        int(q.shape[0]),
        int(q.shape[1]),
        scale,
    )
    return header + q.tobytes(order="C")


def unpack_compact_latents(payload: bytes) -> CompactLatents:
    if len(payload) < _LATENT_HEADER_SIZE:
        raise HprcCompactReceiverError("latents_rc truncated before header")
    magic, version, frames, basis_count, scale = struct.unpack(
        _LATENT_HEADER_FMT, payload[:_LATENT_HEADER_SIZE]
    )
    _check_magic(magic, _LATENT_MAGIC, "latents_rc")
    if version != _VERSION:
        raise HprcCompactReceiverError(f"latents_rc version mismatch: {version}")
    expected = _LATENT_HEADER_SIZE + int(frames) * int(basis_count) * 2
    _require_exact_length(len(payload), expected, "latents_rc")
    q = np.frombuffer(payload[_LATENT_HEADER_SIZE:], dtype="<i2").reshape(
        (frames, basis_count)
    )
    return CompactLatents(
        frames=int(frames),
        basis_count=int(basis_count),
        scale=float(scale),
        q=q,
    )


def pack_compact_selectors(selectors: np.ndarray) -> bytes:
    values = np.asarray(selectors, dtype=np.uint8).reshape(-1)
    header = struct.pack(_SELECTOR_HEADER_FMT, _SELECTOR_MAGIC, _VERSION, int(values.shape[0]))
    return header + values.tobytes(order="C")


def unpack_compact_selectors(payload: bytes) -> CompactSelectors:
    if len(payload) < _SELECTOR_HEADER_SIZE:
        raise HprcCompactReceiverError("selectors_rc truncated before header")
    magic, version, frames = struct.unpack(
        _SELECTOR_HEADER_FMT, payload[:_SELECTOR_HEADER_SIZE]
    )
    _check_magic(magic, _SELECTOR_MAGIC, "selectors_rc")
    if version != _VERSION:
        raise HprcCompactReceiverError(f"selectors_rc version mismatch: {version}")
    _require_exact_length(len(payload), _SELECTOR_HEADER_SIZE + int(frames), "selectors_rc")
    values = np.frombuffer(payload[_SELECTOR_HEADER_SIZE:], dtype=np.uint8)
    return CompactSelectors(frames=int(frames), values=values)


def pack_compact_residual(residual_grid: np.ndarray) -> bytes:
    residual_f = np.asarray(residual_grid, dtype=np.float32)
    if residual_f.ndim != 4:
        raise HprcCompactReceiverError("residual must be frames x grid_h x grid_w x channels")
    if residual_f.shape[3] != 3:
        raise HprcCompactReceiverError("compact residual currently requires RGB")
    max_abs = float(np.max(np.abs(residual_f))) if residual_f.size else 0.0
    scale = max(max_abs / 127.0, 1.0 / 127.0)
    q = np.rint(residual_f / scale).clip(-127, 127).astype(np.int8)
    header = struct.pack(
        _RESIDUAL_HEADER_FMT,
        _RESIDUAL_MAGIC,
        _VERSION,
        int(q.shape[0]),
        int(q.shape[1]),
        int(q.shape[2]),
        int(q.shape[3]),
        scale,
    )
    return header + q.tobytes(order="C")


def unpack_compact_residual(payload: bytes) -> CompactResidual:
    if len(payload) < _RESIDUAL_HEADER_SIZE:
        raise HprcCompactReceiverError("residual_rc truncated before header")
    magic, version, frames, grid_h, grid_w, channels, scale = struct.unpack(
        _RESIDUAL_HEADER_FMT, payload[:_RESIDUAL_HEADER_SIZE]
    )
    _check_magic(magic, _RESIDUAL_MAGIC, "residual_rc")
    if version != _VERSION:
        raise HprcCompactReceiverError(f"residual_rc version mismatch: {version}")
    expected = _RESIDUAL_HEADER_SIZE + int(frames) * int(grid_h) * int(grid_w) * int(channels)
    _require_exact_length(len(payload), expected, "residual_rc")
    q = np.frombuffer(payload[_RESIDUAL_HEADER_SIZE:], dtype=np.int8).reshape(
        (frames, grid_h, grid_w, channels)
    )
    return CompactResidual(
        frames=int(frames),
        grid_h=int(grid_h),
        grid_w=int(grid_w),
        channels=int(channels),
        scale=float(scale),
        q=q,
    )


def pack_compact_receiver_state(state: np.ndarray, *, scale: float = 4.0) -> bytes:
    state_f = np.asarray(state, dtype=np.float32)
    if state_f.ndim != 2:
        raise HprcCompactReceiverError("receiver state must be pairs x dims")
    if state_f.shape[1] > 255:
        raise HprcCompactReceiverError("receiver state dims exceed u8")
    q = np.rint(state_f / float(scale)).clip(-32767, 32767).astype("<i2")
    header = struct.pack(
        _STATE_HEADER_FMT,
        _STATE_MAGIC,
        _VERSION,
        int(q.shape[0]),
        int(q.shape[1]),
        float(scale),
    )
    return header + q.tobytes(order="C")


def unpack_compact_receiver_state(payload: bytes) -> CompactReceiverState:
    if len(payload) < _STATE_HEADER_SIZE:
        raise HprcCompactReceiverError("receiver_state truncated before header")
    magic, version, pairs, dims, scale = struct.unpack(
        _STATE_HEADER_FMT, payload[:_STATE_HEADER_SIZE]
    )
    _check_magic(magic, _STATE_MAGIC, "receiver_state")
    if version != _VERSION:
        raise HprcCompactReceiverError(f"receiver_state version mismatch: {version}")
    expected = _STATE_HEADER_SIZE + int(pairs) * int(dims) * 2
    _require_exact_length(len(payload), expected, "receiver_state")
    q = np.frombuffer(payload[_STATE_HEADER_SIZE:], dtype="<i2").reshape((pairs, dims))
    return CompactReceiverState(pairs=int(pairs), dims=int(dims), scale=float(scale), q=q)


def _manifest_from_packet(packet: HprcPacket) -> dict[str, Any]:
    payload = packet.section_map().get(HprcSectionKind.MANIFEST_JSON)
    if payload is None:
        return {}
    return _loads_json(payload, section="manifest_json")


def is_compact_receiver_packet(packet: HprcPacket) -> bool:
    try:
        manifest = _manifest_from_packet(packet)
    except HprcCompactReceiverError:
        return False
    return manifest.get("hprc_receiver_mode") == COMPACT_RECEIVER_MODE


def decode_compact_receiver_packet(packet: HprcPacket) -> CompactReceiverPacket:
    manifest = _manifest_from_packet(packet)
    if manifest.get("hprc_receiver_mode") != COMPACT_RECEIVER_MODE:
        raise HprcCompactReceiverError("packet is not compact_numpy_receiver_v1")
    section_map = packet.section_map()
    required = (
        HprcSectionKind.DECODER_QW,
        HprcSectionKind.LATENTS_RC,
        HprcSectionKind.SELECTORS_RC,
        HprcSectionKind.RESIDUAL_RC,
        HprcSectionKind.RDO_PLAN,
        HprcSectionKind.RECEIVER_STATE,
    )
    missing = [kind.name.lower() for kind in required if kind not in section_map]
    if missing:
        raise HprcCompactReceiverError(f"compact receiver missing sections: {missing}")
    rdo_plan = _loads_json(section_map[HprcSectionKind.RDO_PLAN], section="rdo_plan")
    decoder = unpack_compact_decoder(section_map[HprcSectionKind.DECODER_QW])
    latents = unpack_compact_latents(section_map[HprcSectionKind.LATENTS_RC])
    selectors = unpack_compact_selectors(section_map[HprcSectionKind.SELECTORS_RC])
    residual = unpack_compact_residual(section_map[HprcSectionKind.RESIDUAL_RC])
    receiver_state = unpack_compact_receiver_state(section_map[HprcSectionKind.RECEIVER_STATE])
    frames = packet.config.frames
    if not (
        latents.frames
        == selectors.frames
        == residual.frames
        == frames
    ):
        raise HprcCompactReceiverError("compact receiver frame counts do not match packet header")
    if latents.basis_count != decoder.basis_count:
        raise HprcCompactReceiverError("latent basis_count does not match decoder")
    expected_pairs = math.ceil(frames / max(packet.config.gop_size, 1))
    if receiver_state.pairs < expected_pairs:
        raise HprcCompactReceiverError("receiver_state has fewer pairs than packet requires")
    return CompactReceiverPacket(
        packet=packet,
        manifest=manifest,
        rdo_plan=rdo_plan,
        decoder=decoder,
        latents=latents,
        selectors=selectors,
        residual=residual,
        receiver_state=receiver_state,
    )


def compact_receiver_section_byte_profile(packet: HprcPacket) -> dict[str, Any]:
    compact = decode_compact_receiver_packet(packet)
    section_map = packet.section_map()
    rows: list[dict[str, Any]] = []
    total = sum(len(payload) for payload in section_map.values())
    for kind in (
        HprcSectionKind.DECODER_QW,
        HprcSectionKind.LATENTS_RC,
        HprcSectionKind.SELECTORS_RC,
        HprcSectionKind.RESIDUAL_RC,
        HprcSectionKind.RDO_PLAN,
        HprcSectionKind.RECEIVER_STATE,
        HprcSectionKind.MANIFEST_JSON,
    ):
        payload = section_map.get(kind, b"")
        rows.append(
            {
                "section": kind.name.lower(),
                "bytes": len(payload),
                "share_of_hprc_payload": round(len(payload) / max(total, 1), 8),
            }
        )
    rows.sort(key=lambda row: int(row["bytes"]), reverse=True)
    residual_bytes = len(section_map.get(HprcSectionKind.RESIDUAL_RC, b""))
    latent_bytes = len(section_map.get(HprcSectionKind.LATENTS_RC, b""))
    return {
        "schema": "hprc_compact_receiver_section_byte_profile.v1",
        "receiver_mode": COMPACT_RECEIVER_MODE,
        "hprc_payload_section_bytes": int(total),
        "frames": int(packet.config.frames),
        "pairs": int(packet.config.pairs),
        "decoder_grid_height": int(compact.decoder.height),
        "decoder_grid_width": int(compact.decoder.width),
        "basis_count": int(compact.decoder.basis_count),
        "residual_grid_height": int(compact.residual.grid_h),
        "residual_grid_width": int(compact.residual.grid_w),
        "section_rows": rows,
        "low_hanging_fruit": [
            {
                "target": "residual_rc",
                "reason": "block residual stream dominates compact HPRC payload",
                "bytes": residual_bytes,
                "next_action": "entropy-code residual int8 tokens with significance map or learned prior",
            },
            {
                "target": "latents_rc",
                "reason": "dense int16 latent stream is small but still directly optimizable",
                "bytes": latent_bytes,
                "next_action": "delta-code pair latents and range/ANS code residual symbols",
            },
            {
                "target": "decoder_grid",
                "reason": "96x128 grid is low enough for rate but risks SegNet/PoseNet loss after camera upsample",
                "bytes": int(total),
                "next_action": "sweep decoder grid and residual grid under exact replay before demotion",
            },
        ],
        "score_claim": False,
        "promotion_eligible": False,
    }


def _nearest_resize(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    src_h, src_w = int(frame.shape[0]), int(frame.shape[1])
    y_idx = (np.arange(height, dtype=np.int64) * src_h // height).clip(0, src_h - 1)
    x_idx = (np.arange(width, dtype=np.int64) * src_w // width).clip(0, src_w - 1)
    return frame[y_idx[:, None], x_idx[None, :], :]


def _residual_to_decoder_grid(residual: CompactResidual, decoder: CompactDecoder, frame_index: int) -> np.ndarray:
    low = residual.q[frame_index].astype(np.float32) * residual.scale
    return _nearest_resize(low, decoder.height, decoder.width)


def render_compact_receiver_frame(
    compact: CompactReceiverPacket,
    frame_index: int,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    if frame_index < 0 or frame_index >= compact.packet.config.frames:
        raise ValueError(f"frame_index outside compact receiver range: {frame_index}")
    decoder = compact.decoder
    rdo = compact.rdo_plan
    frame = decoder.mean.astype(np.float32).copy()
    if decoder.basis_count:
        basis = decoder.basis_q.astype(np.float32) * decoder.basis_scale
        latent = compact.latents.q[frame_index].astype(np.float32) * compact.latents.scale
        frame += float(rdo.get("latent_gain", 1.0)) * np.tensordot(latent, basis, axes=(0, 0))
    selector = float(compact.selectors.values[frame_index]) / 255.0
    frame += (
        float(rdo.get("residual_gain", 1.0))
        * selector
        * _residual_to_decoder_grid(compact.residual, decoder, frame_index)
    )
    state_gain = float(rdo.get("receiver_state_gain", 0.0))
    if state_gain:
        pair_index = min(
            compact.receiver_state.pairs - 1,
            frame_index // max(compact.packet.config.gop_size, 1),
        )
        state = compact.receiver_state.q[pair_index].astype(np.float32) * compact.receiver_state.scale
        if state.shape[0] >= 3:
            frame += state_gain * state[:3].reshape((1, 1, 3))
    out_low = np.clip(np.rint(frame), 0, 255).astype(np.uint8)
    return _nearest_resize(out_low, height, width)


def build_compact_preview_digest(
    compact: CompactReceiverPacket,
    *,
    frame_indices: tuple[int, ...],
    height: int,
    width: int,
) -> str:
    import hashlib

    h = hashlib.sha256()
    for frame_index in frame_indices:
        h.update(
            render_compact_receiver_frame(
                compact,
                frame_index,
                height=height,
                width=width,
            ).tobytes()
        )
    return h.hexdigest()


def compact_receiver_reconstruction_metrics(
    compact: CompactReceiverPacket,
    frames: np.ndarray,
) -> dict[str, Any]:
    """Measure decoder-grid reconstruction against the source frames.

    This is a local materializer audit only. It deliberately reports no contest
    score: SegNet/PoseNet authority requires full contest replay and scorer
    execution on the proper axis.
    """

    arr = np.asarray(frames, dtype=np.float32)
    if arr.ndim == 5:
        pairs, gop, height, width, channels = arr.shape
        arr = arr.reshape((pairs * gop, height, width, channels))
    if arr.ndim != 4:
        raise HprcCompactReceiverError("metrics frames must be FxHxWxC or Px2xHxWxC")
    frame_count, height, width, channels = arr.shape
    if channels != 3:
        raise HprcCompactReceiverError("metrics frames must be RGB")
    if frame_count != compact.packet.config.frames:
        raise HprcCompactReceiverError(
            "metrics frame count does not match compact receiver packet"
        )
    sse = 0.0
    sae = 0.0
    max_abs = 0.0
    count = 0
    for frame_index in range(frame_count):
        rendered = render_compact_receiver_frame(
            compact,
            frame_index,
            height=height,
            width=width,
        ).astype(np.float32)
        diff = rendered - arr[frame_index]
        sse += float(np.sum(diff * diff))
        sae += float(np.sum(np.abs(diff)))
        max_abs = max(max_abs, float(np.max(np.abs(diff))))
        count += int(diff.size)
    mse = sse / max(count, 1)
    mae = sae / max(count, 1)
    psnr = float("inf") if mse <= 0 else 20.0 * math.log10(255.0 / math.sqrt(mse))
    return {
        "schema": "hprc_compact_receiver_decoder_grid_reconstruction_metrics.v1",
        "metric_scope": "decoder_grid_lowres_advisory_not_contest_score",
        "frames": int(frame_count),
        "height": int(height),
        "width": int(width),
        "channels": int(channels),
        "mse_rgb255": float(mse),
        "mae_rgb255": float(mae),
        "max_abs_rgb255": float(max_abs),
        "psnr_rgb255_db": float(psnr),
        "score_claim": False,
        "promotion_eligible": False,
    }


def write_compact_receiver_raw(
    packet: HprcPacket,
    output_path: Any,
    *,
    height: int,
    width: int,
) -> None:
    from pathlib import Path

    compact = decode_compact_receiver_packet(packet)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        for frame_index in range(packet.config.frames):
            f.write(
                render_compact_receiver_frame(
                    compact,
                    frame_index,
                    height=height,
                    width=width,
                ).tobytes()
            )


def _make_basis(height: int, width: int, channels: int, basis_count: int) -> np.ndarray:
    basis = np.zeros((basis_count, height, width, channels), dtype=np.float32)
    filled = 0
    for channel in range(min(channels, basis_count)):
        basis[filled, :, :, channel] = 1.0
        filled += 1
    if filled < basis_count:
        x = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :, None]
        basis[filled, :, :, :] = x
        filled += 1
    if filled < basis_count:
        y = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None, None]
        basis[filled, :, :, :] = y
        filled += 1
    while filled < basis_count:
        basis[filled, :, :, :] = ((filled % 3) - 1) / 3.0
        filled += 1
    return basis


def _block_means(frames: np.ndarray, *, grid_h: int, grid_w: int) -> np.ndarray:
    frame_count, height, width, channels = frames.shape
    out = np.empty((frame_count, grid_h, grid_w, channels), dtype=np.float32)
    y_edges = np.linspace(0, height, grid_h + 1, dtype=np.int64)
    x_edges = np.linspace(0, width, grid_w + 1, dtype=np.int64)
    for gy in range(grid_h):
        y0, y1 = int(y_edges[gy]), int(y_edges[gy + 1])
        for gx in range(grid_w):
            x0, x1 = int(x_edges[gx]), int(x_edges[gx + 1])
            block = frames[:, y0:max(y1, y0 + 1), x0:max(x1, x0 + 1), :]
            out[:, gy, gx, :] = block.mean(axis=(1, 2))
    return out


def build_compact_receiver_packet_from_lowres_frames(
    frames: np.ndarray,
    *,
    basis_count: int = 3,
    residual_grid_h: int = 24,
    residual_grid_w: int = 32,
    source_manifest: dict[str, Any] | None = None,
) -> bytes:
    """Build an HPRC compact receiver from full-video low-resolution frames.

    The builder is deterministic and intentionally modest: channel/spatial
    basis atoms form the learned base; block residual tokens carry remaining
    low-frequency content. It is a rate-axis materializer, not score authority.
    """

    arr = np.asarray(frames, dtype=np.float32)
    if arr.ndim == 5:
        pairs, gop, height, width, channels = arr.shape
        arr = arr.reshape((pairs * gop, height, width, channels))
    if arr.ndim != 4:
        raise HprcCompactReceiverError("frames must be FxHxWxC or Px2xHxWxC")
    frame_count, height, width, channels = arr.shape
    if channels != 3:
        raise HprcCompactReceiverError("compact receiver requires RGB frames")
    if frame_count <= 0 or frame_count > 0xFFFF:
        raise HprcCompactReceiverError("frame count outside u16 range")
    basis_count = max(1, int(basis_count))
    mean = np.rint(arr.mean(axis=0)).clip(0, 255).astype(np.uint8)
    basis = _make_basis(height, width, channels, basis_count)
    centered = arr - mean.astype(np.float32)
    flat_centered = centered.reshape((frame_count, -1))
    flat_basis = basis.reshape((basis_count, -1))
    denom = np.sum(flat_basis * flat_basis, axis=1) + 1e-6
    latents = (flat_centered @ flat_basis.T) / denom[None, :]
    base = mean.astype(np.float32) + np.tensordot(latents, basis, axes=(1, 0))
    residual = _block_means(arr - base, grid_h=int(residual_grid_h), grid_w=int(residual_grid_w))
    selectors = np.full((frame_count,), 255, dtype=np.uint8)
    pairs = math.ceil(frame_count / 2)
    state = np.zeros((pairs, 6), dtype=np.float32)
    if pairs:
        pair_source = np.concatenate([arr, arr[-1:]], axis=0) if frame_count % 2 else arr
        pair_frames = pair_source.reshape((pairs, 2, height, width, channels))
        state[:, :3] = pair_frames.mean(axis=(1, 2, 3)) - arr.mean(axis=(0, 1, 2))
    rdo_plan = {
        "schema": "hprc_compact_receiver_rdo_plan.v1",
        "decoder_mode": COMPACT_RECEIVER_MODE,
        "latent_gain": 1.0,
        "residual_gain": 1.0,
        "receiver_state_gain": 0.25,
        "basis_count": basis_count,
        "residual_grid_h": int(residual_grid_h),
        "residual_grid_w": int(residual_grid_w),
        "score_claim": False,
        "promotion_eligible": False,
    }
    manifest = {
        "schema": "hprc_compact_receiver_manifest.v1",
        "hprc_receiver_mode": COMPACT_RECEIVER_MODE,
        "candidate_kind": "compact_numpy_receiver_with_block_residual_tokens",
        "trained_renderer_export_ready": True,
        "z8_scorer_weighted_residual_sidecar_ready": False,
        "mamba_dreamer_stack_ready": False,
        "exact_cpu_cuda_authority_ready": False,
        "score_claim": False,
        "promotion_eligible": False,
        "source": dict(source_manifest or {}),
    }
    sections = {
        HprcSectionKind.DECODER_QW: pack_compact_decoder(mean, basis),
        HprcSectionKind.LATENTS_RC: pack_compact_latents(latents),
        HprcSectionKind.SELECTORS_RC: pack_compact_selectors(selectors),
        HprcSectionKind.RESIDUAL_RC: pack_compact_residual(residual),
        HprcSectionKind.RDO_PLAN: _json_bytes(rdo_plan),
        HprcSectionKind.RECEIVER_STATE: pack_compact_receiver_state(state),
        HprcSectionKind.MANIFEST_JSON: _json_bytes(manifest),
    }
    return pack_hprc_packet(
        sections,
        config=HprcPacketConfig(
            frames=frame_count,
            pairs=pairs,
            height=int(height),
            width=int(width),
            decoder_family_id=COMPACT_NUMPY_DECODER_FAMILY_ID,
            color_transform_id=COMPACT_RGB_COLOR_TRANSFORM_ID,
            gop_size=2,
        ),
    )


def mutate_compact_receiver_section(
    packet: HprcPacket,
    kind: HprcSectionKind,
    *,
    salt: int,
) -> bytes | None:
    """Return a valid semantic mutation for compact receiver proofing."""

    if not is_compact_receiver_packet(packet):
        return None
    payload = packet.section_map().get(kind)
    if payload is None:
        return None
    data = bytearray(payload)
    if kind == HprcSectionKind.DECODER_QW and len(data) > _DECODER_HEADER_SIZE:
        data[_DECODER_HEADER_SIZE] = (data[_DECODER_HEADER_SIZE] + 7 + salt) & 0xFF
        return bytes(data)
    if kind == HprcSectionKind.LATENTS_RC and len(data) > _LATENT_HEADER_SIZE:
        data[_LATENT_HEADER_SIZE] = (data[_LATENT_HEADER_SIZE] + 13 + salt) & 0xFF
        return bytes(data)
    if kind == HprcSectionKind.SELECTORS_RC and len(data) > _SELECTOR_HEADER_SIZE:
        fill = 0 if any(data[_SELECTOR_HEADER_SIZE:]) else 255
        data[_SELECTOR_HEADER_SIZE:] = bytes([fill]) * (len(data) - _SELECTOR_HEADER_SIZE)
        return bytes(data)
    if kind == HprcSectionKind.RESIDUAL_RC and len(data) > _RESIDUAL_HEADER_SIZE:
        data[_RESIDUAL_HEADER_SIZE:] = bytes([127]) * (len(data) - _RESIDUAL_HEADER_SIZE)
        return bytes(data)
    if kind == HprcSectionKind.RECEIVER_STATE and len(data) > _STATE_HEADER_SIZE:
        data[_STATE_HEADER_SIZE] = (data[_STATE_HEADER_SIZE] + 23 + salt) & 0xFF
        return bytes(data)
    if kind == HprcSectionKind.RDO_PLAN:
        rdo = _loads_json(payload, section="rdo_plan")
        rdo["latent_gain"] = 0.0 if float(rdo.get("latent_gain", 1.0)) else 1.0
        rdo["residual_gain"] = float(rdo.get("residual_gain", 1.0)) + 0.5
        rdo["semantic_mutation_salt"] = int(salt)
        return _json_bytes(rdo)
    if kind == HprcSectionKind.MANIFEST_JSON:
        manifest = _loads_json(payload, section="manifest_json")
        manifest["semantic_mutation_note"] = f"metadata_only_{salt}"
        return _json_bytes(manifest)
    return None


__all__ = [
    "COMPACT_NUMPY_DECODER_FAMILY_ID",
    "COMPACT_RECEIVER_MODE",
    "COMPACT_RGB_COLOR_TRANSFORM_ID",
    "HprcCompactReceiverError",
    "build_compact_preview_digest",
    "build_compact_receiver_packet_from_lowres_frames",
    "compact_receiver_reconstruction_metrics",
    "compact_receiver_section_byte_profile",
    "decode_compact_receiver_packet",
    "is_compact_receiver_packet",
    "mutate_compact_receiver_section",
    "pack_compact_decoder",
    "pack_compact_latents",
    "pack_compact_receiver_state",
    "pack_compact_residual",
    "pack_compact_selectors",
    "render_compact_receiver_frame",
    "write_compact_receiver_raw",
]
