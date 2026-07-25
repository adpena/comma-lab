# SPDX-License-Identifier: MIT
"""Typed counted PC1 pose-stream owner and deterministic multi-depth receiver.

PC1 is deliberately a standalone composition member.  It never reads the R1
``dxi`` payload: R1 is harvest-signal-only and supplies neither bytes nor
weights to this owner.  The only counted video-derived state in this module is
the quantized per-pair pose stream and its small luma-phase residual home.

The receiver works at the frozen scorer geometry.  It constructs a continuous
ground-plane depth field, replaces Movable pixels by a contact-depth stratum,
and writes both camera frames from one decoded frame-0 source:

    frame_0 = W_{-xi/2, depth}(source) + luma_phase_residual
    frame_1 = W_{ xi,   depth}(frame_0)

Thus frame 1 is produced by the same explicit ``W_{xi,depth}`` operator applied
to the generated frame 0.  Inactive packets return the parent bytes exactly.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import struct
import zipfile
import zlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

CAMERA_H = 874
CAMERA_W = 1164
PAIR_H = 384
PAIR_W = 512
POSE_DIMS = 6
LUMA_PHASES = 4
SOURCE_BYTES = 37_545_489
PACKET_MEMBER = "pose/pc1.ddp"
PARENT_MEMBER = "parent/ws1.zip"
MANIFEST_MEMBER = "manifest/pc1.json"
PACKET_MAGIC = b"DDMPC1\x01\x00"
PACKET_HEADER = struct.Struct("<8sBHH6ff")
AXIS_NAMES = ("tx", "ty", "tz", "rx", "ry", "rz")


class PC1PoseStreamError(ValueError):
    """Raised when PC1 custody, packet, or receiver invariants fail."""


def _readonly_array(value: Any, *, dtype: np.dtype[Any], shape_tail: tuple[int, ...]) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True, order="C")
    if result.ndim != len(shape_tail) + 1 or tuple(result.shape[1:]) != shape_tail:
        raise PC1PoseStreamError(f"array must have shape (pairs,{','.join(map(str, shape_tail))})")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PC1PosePacketV1:
    """Single counted owner for the pose stream and its residual home."""

    active: bool
    pair_count: int
    xi_scales: tuple[float, float, float, float, float, float]
    residual_scale: float
    q_xi: np.ndarray
    q_luma_phase: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.active, bool):
            raise PC1PoseStreamError("active must be bool")
        if len(self.xi_scales) != POSE_DIMS:
            raise PC1PoseStreamError("xi_scales must have six values")
        scales = tuple(float(np.float32(value)) for value in self.xi_scales)
        if any(not math.isfinite(value) or value <= 0.0 for value in scales):
            raise PC1PoseStreamError("xi_scales must be finite and positive")
        residual_scale = float(np.float32(self.residual_scale))
        if not math.isfinite(residual_scale) or residual_scale <= 0.0:
            raise PC1PoseStreamError("residual_scale must be finite and positive")
        if (
            isinstance(self.pair_count, bool)
            or not isinstance(self.pair_count, int)
            or self.pair_count <= 1
            or self.pair_count > 65_535
        ):
            raise PC1PoseStreamError("pair_count must fit uint16 and exceed one")
        q_xi = _readonly_array(self.q_xi, dtype=np.dtype("<i2"), shape_tail=(POSE_DIMS,))
        q_luma = _readonly_array(
            self.q_luma_phase,
            dtype=np.dtype("i1"),
            shape_tail=(LUMA_PHASES,),
        )
        if len(q_xi) < 2 or len(q_xi) != len(q_luma):
            raise PC1PoseStreamError("q_xi and q_luma_phase knot counts must agree")
        if len(q_xi) > 65_535 or len(q_xi) > self.pair_count:
            raise PC1PoseStreamError("packet knot count is outside typed custody")
        object.__setattr__(self, "xi_scales", scales)
        object.__setattr__(self, "residual_scale", residual_scale)
        object.__setattr__(self, "q_xi", q_xi)
        object.__setattr__(self, "q_luma_phase", q_luma)

    @property
    def knot_count(self) -> int:
        return len(self.q_xi)

    def decoded_xi(self, pair_ids: Sequence[int] | None = None) -> np.ndarray:
        ids = _validate_pair_ids(pair_ids, self.pair_count)
        scales = np.asarray(self.xi_scales, dtype=np.float64)
        controls = self.q_xi.astype(np.float64) * scales[None, :]
        return _interpolate_controls(controls, ids, self.pair_count)

    def decoded_luma_phase(self, pair_ids: Sequence[int] | None = None) -> np.ndarray:
        ids = _validate_pair_ids(pair_ids, self.pair_count)
        controls = self.q_luma_phase.astype(np.float64) * np.float64(self.residual_scale)
        return np.ascontiguousarray(
            _interpolate_controls(controls, ids, self.pair_count),
            dtype=np.float32,
        )


def _interpolate_controls(
    controls: np.ndarray,
    pair_ids: np.ndarray,
    pair_count: int,
) -> np.ndarray:
    """Linearly decode a smooth knot stream at exact integer pair locations."""

    knot_count = len(controls)
    position = pair_ids.astype(np.float64) * (knot_count - 1) / (pair_count - 1)
    left = np.floor(position).astype(np.int64)
    right = np.minimum(left + 1, knot_count - 1)
    weight = (position - left).reshape(-1, 1)
    return np.ascontiguousarray(controls[left] * (1.0 - weight) + controls[right] * weight)


def _validate_pair_ids(pair_ids: Sequence[int] | None, pair_count: int) -> np.ndarray:
    if pair_ids is None:
        return np.arange(pair_count, dtype=np.int64)
    ids = np.asarray(pair_ids)
    if ids.ndim != 1 or ids.dtype.kind not in "iu":
        raise PC1PoseStreamError("pair_ids must be a one-dimensional integer sequence")
    ids = np.ascontiguousarray(ids, dtype=np.int64)
    if np.any(ids < 0) or np.any(ids >= pair_count):
        raise PC1PoseStreamError("pair_ids fall outside packet custody")
    return ids


def serialize_pc1_packet(packet: PC1PosePacketV1) -> bytes:
    """Serialize a packet deterministically and compress its counted payload."""

    header = PACKET_HEADER.pack(
        PACKET_MAGIC,
        int(packet.active),
        packet.pair_count,
        packet.knot_count,
        *packet.xi_scales,
        packet.residual_scale,
    )
    raw = b"".join(
        (
            header,
            packet.q_xi.astype("<i2", copy=False).tobytes(order="C"),
            packet.q_luma_phase.astype("i1", copy=False).tobytes(order="C"),
        )
    )
    return zlib.compress(raw, level=9)


def parse_pc1_packet(payload: bytes) -> PC1PosePacketV1:
    """Parse, validate, and canonicalize a counted PC1 packet."""

    if not isinstance(payload, bytes):
        raise PC1PoseStreamError("packet payload must be bytes")
    try:
        raw = zlib.decompress(payload)
    except zlib.error as exc:
        raise PC1PoseStreamError("packet zlib stream is invalid") from exc
    if len(raw) < PACKET_HEADER.size:
        raise PC1PoseStreamError("packet is truncated before its header")
    unpacked = PACKET_HEADER.unpack_from(raw)
    magic, active_byte, pair_count, knot_count = unpacked[:4]
    if magic != PACKET_MAGIC or active_byte not in (0, 1):
        raise PC1PoseStreamError("packet header is noncanonical")
    if pair_count <= 1 or knot_count < 2 or knot_count > pair_count:
        raise PC1PoseStreamError("packet pair/knot geometry is invalid")
    expected = PACKET_HEADER.size + knot_count * (POSE_DIMS * 2 + LUMA_PHASES)
    if len(raw) != expected:
        raise PC1PoseStreamError("packet byte count disagrees with typed geometry")
    cursor = PACKET_HEADER.size
    xi_bytes = knot_count * POSE_DIMS * 2
    q_xi = np.frombuffer(raw[cursor : cursor + xi_bytes], dtype="<i2").reshape(
        knot_count,
        POSE_DIMS,
    )
    cursor += xi_bytes
    q_luma = np.frombuffer(raw[cursor:], dtype="i1").reshape(knot_count, LUMA_PHASES)
    packet = PC1PosePacketV1(
        active=bool(active_byte),
        pair_count=int(pair_count),
        xi_scales=tuple(float(value) for value in unpacked[4:10]),
        residual_scale=float(unpacked[10]),
        q_xi=q_xi,
        q_luma_phase=q_luma,
    )
    if serialize_pc1_packet(packet) != payload:
        raise PC1PoseStreamError("packet is valid but not in canonical re-emission form")
    return packet


def make_inactive_packet(packet: PC1PosePacketV1) -> PC1PosePacketV1:
    return PC1PosePacketV1(
        active=False,
        pair_count=packet.pair_count,
        xi_scales=packet.xi_scales,
        residual_scale=packet.residual_scale,
        q_xi=packet.q_xi,
        q_luma_phase=packet.q_luma_phase,
    )


def make_zero_active_packet(packet: PC1PosePacketV1) -> PC1PosePacketV1:
    """Return the active receiver home used for causal counted-q probes."""

    return PC1PosePacketV1(
        active=True,
        pair_count=packet.pair_count,
        xi_scales=packet.xi_scales,
        residual_scale=packet.residual_scale,
        q_xi=np.zeros_like(packet.q_xi),
        q_luma_phase=np.zeros_like(packet.q_luma_phase),
    )


def packet_sha256(packet: PC1PosePacketV1) -> str:
    return hashlib.sha256(serialize_pc1_packet(packet)).hexdigest()


@dataclass(frozen=True)
class PC1ParameterCoordinateV1:
    coordinate_id: str
    knot_id: int
    family: str
    axis: str
    quantization_scale: float
    counted_member: str = PACKET_MEMBER
    output_effect_owner: str = "ddm.pc1.pose_stream"


@dataclass(frozen=True)
class DDMPC1TrainableParameterMapV1:
    """Descent-trainable parameter map consumed by DDM optimizer #366."""

    pair_count: int
    knot_count: int
    xi_scales: tuple[float, float, float, float, float, float]
    residual_scale: float

    def __post_init__(self) -> None:
        if isinstance(self.pair_count, bool) or self.pair_count <= 0:
            raise PC1PoseStreamError("pair_count must be positive")
        if isinstance(self.knot_count, bool) or self.knot_count < 2 or self.knot_count > self.pair_count:
            raise PC1PoseStreamError("knot_count must be in [2,pair_count]")
        if len(self.xi_scales) != POSE_DIMS or any(
            not math.isfinite(float(value)) or float(value) <= 0.0 for value in self.xi_scales
        ):
            raise PC1PoseStreamError("parameter-map xi_scales are invalid")
        if not math.isfinite(float(self.residual_scale)) or self.residual_scale <= 0.0:
            raise PC1PoseStreamError("parameter-map residual_scale is invalid")

    def coordinates(self) -> tuple[PC1ParameterCoordinateV1, ...]:
        rows: list[PC1ParameterCoordinateV1] = []
        for knot_id in range(self.knot_count):
            for axis, scale in zip(AXIS_NAMES, self.xi_scales, strict=True):
                rows.append(
                    PC1ParameterCoordinateV1(
                        coordinate_id=f"ddm.pc1.knot.{knot_id:03d}.xi.{axis}",
                        knot_id=knot_id,
                        family="pose_xi",
                        axis=axis,
                        quantization_scale=float(scale),
                    )
                )
            for phase in range(LUMA_PHASES):
                rows.append(
                    PC1ParameterCoordinateV1(
                        coordinate_id=f"ddm.pc1.knot.{knot_id:03d}.luma_phase.{phase}",
                        knot_id=knot_id,
                        family="luma_phase_residual",
                        axis=str(phase),
                        quantization_scale=float(self.residual_scale),
                    )
                )
        return tuple(rows)

    def project(
        self,
        *,
        xi: np.ndarray,
        luma_phase: np.ndarray,
        active: bool = True,
    ) -> PC1PosePacketV1:
        xi_value = np.asarray(xi, dtype=np.float64)
        residual_value = np.asarray(luma_phase, dtype=np.float64)
        if xi_value.shape != (self.knot_count, POSE_DIMS):
            raise PC1PoseStreamError("xi has wrong parameter-map geometry")
        if residual_value.shape != (self.knot_count, LUMA_PHASES):
            raise PC1PoseStreamError("luma_phase has wrong parameter-map geometry")
        if not np.all(np.isfinite(xi_value)) or not np.all(np.isfinite(residual_value)):
            raise PC1PoseStreamError("parameter map cannot quantize nonfinite values")
        q_xi_f64 = np.rint(xi_value / np.asarray(self.xi_scales)[None, :])
        q_luma_f64 = np.rint(residual_value / self.residual_scale)
        if np.any(np.abs(q_xi_f64) > np.iinfo(np.int16).max):
            raise PC1PoseStreamError("xi falls outside int16 packet range")
        if np.any(np.abs(q_luma_f64) > np.iinfo(np.int8).max):
            raise PC1PoseStreamError("luma residual falls outside int8 packet range")
        return PC1PosePacketV1(
            active=active,
            pair_count=self.pair_count,
            xi_scales=self.xi_scales,
            residual_scale=self.residual_scale,
            q_xi=q_xi_f64.astype("<i2"),
            q_luma_phase=q_luma_f64.astype("i1"),
        )


def fresh_pose_initialization(
    pose_centers: np.ndarray,
    *,
    knot_count: int = 32,
    focal_px: float = 910.0 * PAIR_W / CAMERA_W,
    camera_height_m: float = 1.22,
) -> tuple[np.ndarray, tuple[float, float, float, float, float, float]]:
    """Construct a fresh, geometry-scaled initialization from scorer centers.

    The centers choose directions only.  Their amplitudes are discarded and
    replaced by one frozen-scorer-cell translation/rotation quanta.  This is
    intentionally not a load, composition, or anchor of R1 ``dxi`` bytes.
    """

    centers = np.asarray(pose_centers, dtype=np.float64)
    if centers.ndim != 2 or centers.shape[1] != POSE_DIMS or len(centers) == 0:
        raise PC1PoseStreamError("pose_centers must have shape (pairs,6)")
    if not np.all(np.isfinite(centers)):
        raise PC1PoseStreamError("pose_centers must be finite")
    if not math.isfinite(focal_px) or focal_px <= 0.0:
        raise PC1PoseStreamError("focal_px must be positive")
    if not math.isfinite(camera_height_m) or camera_height_m <= 0.0:
        raise PC1PoseStreamError("camera_height_m must be positive")

    max_abs = np.max(np.abs(centers), axis=0)
    safe = np.where(max_abs > 0.0, max_abs, 1.0)
    unit_direction = centers / safe[None, :]
    translation_quantum = camera_height_m / focal_px
    rotation_quantum = 1.0 / focal_px
    amplitude = np.asarray(
        [translation_quantum] * 3 + [rotation_quantum] * 3,
        dtype=np.float64,
    )
    if isinstance(knot_count, bool) or knot_count < 2 or knot_count > len(centers):
        raise PC1PoseStreamError("knot_count must be in [2,pair_count]")
    knot_pair_ids = np.rint(np.linspace(0, len(centers) - 1, knot_count, dtype=np.float64)).astype(np.int64)
    xi = np.ascontiguousarray(unit_direction[knot_pair_ids] * amplitude[None, :])
    quantization_scales = tuple(float(value / 256.0) for value in amplitude)
    return xi, quantization_scales


def solved_plane_yuv6_target(
    parent_camera: np.ndarray,
    *,
    torch_module: Any | None = None,
) -> Any:
    """Expose the exact solved-plane PoseNet target to #366 without storing it.

    This is the frozen scorer's shared resize followed by its exact full-range
    BT.601/YUV6 polyphase map.  The values are derived for free from the decoded
    W parent, so PC1 descent targets the already-solved plane instead of
    inventing an RGB correction program.
    """

    parent = np.asarray(parent_camera)
    if parent.dtype != np.uint8 or parent.ndim != 5 or parent.shape[1:] != (2, CAMERA_H, CAMERA_W, 3):
        raise PC1PoseStreamError("solved-plane target camera geometry differs")
    if torch_module is None:
        try:
            import torch as torch_module  # type: ignore[no-redef]
        except ImportError as exc:
            raise PC1PoseStreamError("solved-plane target requires torch") from exc
    torch = torch_module
    tensor = torch.from_numpy(np.ascontiguousarray(parent)).permute(0, 1, 4, 2, 3).contiguous().float()
    batch, frames = tensor.shape[:2]
    resized = torch.nn.functional.interpolate(
        tensor.reshape(batch * frames, 3, CAMERA_H, CAMERA_W),
        size=(PAIR_H, PAIR_W),
        mode="bilinear",
        align_corners=False,
    )
    red, green, blue = resized[:, 0], resized[:, 1], resized[:, 2]
    luma = torch.clamp(red * 0.299 + green * 0.587 + blue * 0.114, 0.0, 255.0)
    chroma_u = torch.clamp((blue - luma) / 1.772 + 128.0, 0.0, 255.0)
    chroma_v = torch.clamp((red - luma) / 1.402 + 128.0, 0.0, 255.0)
    u_sub = (
        chroma_u[:, 0::2, 0::2] + chroma_u[:, 1::2, 0::2] + chroma_u[:, 0::2, 1::2] + chroma_u[:, 1::2, 1::2]
    ) * 0.25
    v_sub = (
        chroma_v[:, 0::2, 0::2] + chroma_v[:, 1::2, 0::2] + chroma_v[:, 0::2, 1::2] + chroma_v[:, 1::2, 1::2]
    ) * 0.25
    yuv6 = torch.stack(
        (
            luma[:, 0::2, 0::2],
            luma[:, 1::2, 0::2],
            luma[:, 0::2, 1::2],
            luma[:, 1::2, 1::2],
            u_sub,
            v_sub,
        ),
        dim=1,
    )
    return yuv6.reshape(batch, frames, 6, PAIR_H // 2, PAIR_W // 2)


def active_tube_quadratic(
    *,
    candidate_pose6: np.ndarray,
    centers: np.ndarray,
    low_rank_factors: np.ndarray,
) -> np.ndarray:
    """Evaluate the landed MS4d pose quadratic without asserting membership."""

    candidate = np.asarray(candidate_pose6, dtype=np.float64)
    center = np.asarray(centers, dtype=np.float64)
    factors = np.asarray(low_rank_factors, dtype=np.float64)
    if candidate.shape != center.shape or candidate.ndim != 2 or candidate.shape[1] != POSE_DIMS:
        raise PC1PoseStreamError("candidate_pose6 and centers must have shape (pairs,6)")
    if factors.shape != (len(candidate), POSE_DIMS, POSE_DIMS):
        raise PC1PoseStreamError("low_rank_factors must have shape (pairs,6,6)")
    if not all(np.all(np.isfinite(value)) for value in (candidate, center, factors)):
        raise PC1PoseStreamError("active-tube quadratic inputs must be finite")
    delta = candidate - center
    projected = np.einsum("nij,nj->ni", factors, delta)
    return np.ascontiguousarray(np.sum(projected * projected, axis=1))


def ground_and_movable_depth(
    movable_mask: np.ndarray | None,
    *,
    camera_height_m: float = 1.22,
    focal_px: float = 910.0 * PAIR_H / CAMERA_H,
    horizon_row: float = PAIR_H / 2.0,
    far_depth_m: float = 120.0,
) -> np.ndarray:
    """Derive continuous ground depth plus a contact-depth Movable stratum."""

    if movable_mask is None:
        movable = np.zeros((PAIR_H, PAIR_W), dtype=np.bool_)
    else:
        movable = np.asarray(movable_mask)
        if movable.shape != (PAIR_H, PAIR_W):
            raise PC1PoseStreamError("Movable mask has wrong scorer geometry")
        movable = movable.astype(np.bool_, copy=False)
    rows = np.arange(PAIR_H, dtype=np.float64)[:, None]
    denominator = np.maximum(rows - horizon_row, 1.0)
    ground = camera_height_m * focal_px / denominator
    ground = np.clip(ground, camera_height_m, far_depth_m)
    depth = np.broadcast_to(ground, (PAIR_H, PAIR_W)).copy()
    depth[rows[:, 0] <= horizon_row, :] = far_depth_m
    if np.any(movable):
        contact_row = int(np.max(np.nonzero(movable)[0]))
        contact_denominator = max(float(contact_row) - horizon_row, 1.0)
        contact_depth = float(
            np.clip(
                camera_height_m * focal_px / contact_denominator,
                camera_height_m,
                far_depth_m,
            )
        )
        depth[movable] = contact_depth
    return np.ascontiguousarray(depth.astype(np.float32))


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = vector
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def _se3_exp(xi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    translation = np.asarray(xi[:3], dtype=np.float64)
    omega = np.asarray(xi[3:], dtype=np.float64)
    theta = float(np.linalg.norm(omega))
    omega_hat = _skew(omega)
    identity = np.eye(3, dtype=np.float64)
    if theta < 1e-10:
        rotation = identity + omega_hat + 0.5 * (omega_hat @ omega_hat)
        v_matrix = identity + 0.5 * omega_hat + (omega_hat @ omega_hat) / 6.0
    else:
        theta2 = theta * theta
        rotation = (
            identity + math.sin(theta) / theta * omega_hat + (1.0 - math.cos(theta)) / theta2 * (omega_hat @ omega_hat)
        )
        v_matrix = (
            identity
            + (1.0 - math.cos(theta)) / theta2 * omega_hat
            + (theta - math.sin(theta)) / (theta2 * theta) * (omega_hat @ omega_hat)
        )
    return rotation, v_matrix @ translation


def _warp_scorer_frame(
    frame: Any,
    *,
    xi: np.ndarray,
    depth: np.ndarray,
    torch_module: Any,
) -> Any:
    """Apply W_{xi,depth} with one deterministic inverse-projection grid."""

    torch = torch_module
    functional = torch.nn.functional
    if tuple(frame.shape) != (1, 3, PAIR_H, PAIR_W):
        raise PC1PoseStreamError("scorer frame has wrong tensor geometry")
    xi_value = np.asarray(xi, dtype=np.float64)
    depth_value = np.asarray(depth, dtype=np.float64)
    if (
        xi_value.shape != (POSE_DIMS,)
        or depth_value.shape != (PAIR_H, PAIR_W)
        or not np.all(np.isfinite(xi_value))
        or not np.all(np.isfinite(depth_value))
        or np.any(depth_value <= 0.0)
        or not bool(torch.isfinite(frame).all())
    ):
        raise PC1PoseStreamError("warp inputs must be finite with positive depth")
    try:
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            rotation, translation = _se3_exp(xi_value)
    except (ArithmeticError, ValueError) as exc:
        raise PC1PoseStreamError("SE(3) warp transform is nonfinite") from exc
    if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation)):
        raise PC1PoseStreamError("SE(3) warp transform is nonfinite")
    focal_x = 910.0 * PAIR_W / CAMERA_W
    focal_y = 910.0 * PAIR_H / CAMERA_H
    center_x = (PAIR_W - 1.0) / 2.0
    center_y = (PAIR_H - 1.0) / 2.0
    columns = np.arange(PAIR_W, dtype=np.float64)[None, :]
    rows = np.arange(PAIR_H, dtype=np.float64)[:, None]
    z = depth_value
    x = (columns - center_x) / focal_x * z
    y = (rows - center_y) / focal_y * z
    points = np.stack((x, y, z), axis=0).reshape(3, -1)
    # NumPy/Accelerate emits spurious divide/overflow/invalid warnings for this
    # finite 3xN matmul on some macOS builds. Preserve the exact matmul while
    # making its real authority boundary explicit: warnings are locally
    # contained and every result must still pass a finiteness check.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        source = rotation.T @ (points - translation[:, None])
    if not np.all(np.isfinite(source)):
        raise PC1PoseStreamError("inverse-projected warp coordinates are nonfinite")
    source_z = np.maximum(source[2], 1e-4)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        source_x = focal_x * source[0] / source_z + center_x
        source_y = focal_y * source[1] / source_z + center_y
    grid_x = 2.0 * source_x.reshape(PAIR_H, PAIR_W) / (PAIR_W - 1.0) - 1.0
    grid_y = 2.0 * source_y.reshape(PAIR_H, PAIR_W) / (PAIR_H - 1.0) - 1.0
    grid_f64 = np.stack((grid_x, grid_y), axis=-1)
    if not np.all(np.isfinite(grid_f64)) or np.max(np.abs(grid_f64)) > np.finfo(np.float32).max:
        raise PC1PoseStreamError("warp sampling grid is nonfinite or outside float32")
    grid_np = grid_f64.astype(np.float32)[None, ...]
    grid = torch.from_numpy(np.ascontiguousarray(grid_np)).to(frame.device)
    return functional.grid_sample(
        frame,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


def _luma_phase_image(values: np.ndarray) -> np.ndarray:
    phase = np.asarray(values, dtype=np.float32)
    if phase.shape != (LUMA_PHASES,):
        raise PC1PoseStreamError("luma phase residual has wrong geometry")
    rows = np.arange(PAIR_H)[:, None] & 1
    columns = np.arange(PAIR_W)[None, :] & 1
    phase_index = rows * 2 + columns
    return np.ascontiguousarray(phase[phase_index], dtype=np.float32)


def receive_pc1_camera_pairs(
    *,
    parent_camera: np.ndarray,
    packet: PC1PosePacketV1,
    pair_ids: Sequence[int],
    movable_masks: np.ndarray | None = None,
    torch_module: Any | None = None,
) -> np.ndarray:
    """Render PC1 camera pairs or return exact parent bytes when inactive."""

    parent = np.asarray(parent_camera)
    expected_tail = (2, CAMERA_H, CAMERA_W, 3)
    if parent.dtype != np.uint8 or parent.ndim != 5 or parent.shape[1:] != expected_tail:
        raise PC1PoseStreamError("parent camera has wrong uint8 camera-pair geometry")
    ids = _validate_pair_ids(pair_ids, packet.pair_count)
    if len(ids) != len(parent):
        raise PC1PoseStreamError("pair_ids and parent_camera batch disagree")
    if not packet.active:
        return np.ascontiguousarray(parent)
    if torch_module is None:
        try:
            import torch as torch_module  # type: ignore[no-redef]
        except ImportError as exc:
            raise PC1PoseStreamError("active receiver requires torch") from exc
    torch = torch_module
    functional = torch.nn.functional
    if movable_masks is None:
        masks: list[np.ndarray | None] = [None] * len(parent)
    else:
        movable_value = np.asarray(movable_masks)
        if movable_value.shape != (len(parent), PAIR_H, PAIR_W):
            raise PC1PoseStreamError("movable_masks batch geometry differs")
        masks = [movable_value[index] for index in range(len(parent))]

    xi_rows = packet.decoded_xi(ids)
    residual_rows = packet.decoded_luma_phase(ids)
    rendered: list[Any] = []
    with torch.inference_mode():
        for local_index in range(len(parent)):
            source_u8 = np.ascontiguousarray(parent[local_index, 0])
            source = torch.from_numpy(source_u8).permute(2, 0, 1).unsqueeze(0).float()
            scorer_source = functional.interpolate(
                source,
                size=(PAIR_H, PAIR_W),
                mode="bilinear",
                align_corners=False,
            )
            depth = ground_and_movable_depth(masks[local_index])
            xi = xi_rows[local_index]
            frame0 = _warp_scorer_frame(
                scorer_source,
                xi=-0.5 * xi,
                depth=depth,
                torch_module=torch,
            )
            phase = torch.from_numpy(_luma_phase_image(residual_rows[local_index]))
            frame0 = torch.clamp(frame0 + phase[None, None, :, :], 0.0, 255.0)
            frame1 = _warp_scorer_frame(
                frame0,
                xi=xi,
                depth=depth,
                torch_module=torch,
            )
            pair = torch.cat((frame0, frame1), dim=0)
            camera_pair = functional.interpolate(
                pair,
                size=(CAMERA_H, CAMERA_W),
                mode="bicubic",
                align_corners=False,
            )
            camera_pair = torch.clamp(torch.round(camera_pair), 0.0, 255.0).to(torch.uint8).permute(0, 2, 3, 1).cpu()
            rendered.append(camera_pair)
    result = torch.stack(rendered, dim=0).numpy()
    return np.ascontiguousarray(result, dtype=np.uint8)


def conditional_score_delta(
    *,
    parent_dseg: float,
    parent_dpose: float,
    candidate_dseg: float,
    candidate_dpose: float,
    candidate_archive_bytes: int,
    parent_archive_bytes: int,
) -> float:
    values = (parent_dseg, parent_dpose, candidate_dseg, candidate_dpose)
    if any(not math.isfinite(float(value)) or float(value) < 0.0 for value in values):
        raise PC1PoseStreamError("conditional score inputs must be finite and nonnegative")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 0
        for value in (candidate_archive_bytes, parent_archive_bytes)
    ):
        raise PC1PoseStreamError("archive byte counts must be nonnegative integers")
    return (
        100.0 * (float(candidate_dseg) - float(parent_dseg))
        + math.sqrt(10.0 * float(candidate_dpose))
        - math.sqrt(10.0 * float(parent_dpose))
        + 25.0 * (candidate_archive_bytes - parent_archive_bytes) / SOURCE_BYTES
    )


def build_counted_composition_archive(
    *,
    parent_archive: bytes,
    parent_sha256: str,
    packet: PC1PosePacketV1,
) -> bytes:
    """Build the exact counted nested composition archive deterministically."""

    actual_parent_sha256 = hashlib.sha256(parent_archive).hexdigest()
    if actual_parent_sha256 != parent_sha256:
        raise PC1PoseStreamError("parent archive SHA-256 custody differs")
    packet_bytes = serialize_pc1_packet(packet)
    manifest = {
        "active": packet.active,
        "equation_id": "ddm_pc1_pose_stream_laws_v1",
        "owner": "ddm.pc1.pose_stream",
        "packet_member": PACKET_MEMBER,
        "packet_sha256": hashlib.sha256(packet_bytes).hexdigest(),
        "parent_member": PARENT_MEMBER,
        "parent_sha256": parent_sha256,
        "schema": "ddm_pc1_pose_composition.v1",
    }
    members = (
        (MANIFEST_MEMBER, json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()),
        (PACKET_MEMBER, packet_bytes),
        (PARENT_MEMBER, parent_archive),
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED, strict_timestamps=True) as archive:
        for member_name, member_bytes in members:
            info = zipfile.ZipInfo(member_name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o100644 << 16
            info.create_system = 3
            archive.writestr(info, member_bytes)
    return buffer.getvalue()


def parse_counted_composition_archive(payload: bytes) -> tuple[bytes, PC1PosePacketV1, dict[str, Any]]:
    """Parse back a complete counted PC1 composition and verify all custody."""

    try:
        with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
            names = archive.namelist()
            if names != [MANIFEST_MEMBER, PACKET_MEMBER, PARENT_MEMBER]:
                raise PC1PoseStreamError("composition member order/schema differs")
            manifest = json.loads(archive.read(MANIFEST_MEMBER))
            packet_bytes = archive.read(PACKET_MEMBER)
            parent_bytes = archive.read(PARENT_MEMBER)
    except (zipfile.BadZipFile, KeyError, json.JSONDecodeError) as exc:
        raise PC1PoseStreamError("composition archive is invalid") from exc
    if manifest.get("schema") != "ddm_pc1_pose_composition.v1":
        raise PC1PoseStreamError("composition schema differs")
    if hashlib.sha256(packet_bytes).hexdigest() != manifest.get("packet_sha256"):
        raise PC1PoseStreamError("packet hash differs after archive parse-back")
    if hashlib.sha256(parent_bytes).hexdigest() != manifest.get("parent_sha256"):
        raise PC1PoseStreamError("parent hash differs after archive parse-back")
    packet = parse_pc1_packet(packet_bytes)
    if packet.active is not manifest.get("active"):
        raise PC1PoseStreamError("packet active state differs from manifest")
    rebuilt = build_counted_composition_archive(
        parent_archive=parent_bytes,
        parent_sha256=manifest["parent_sha256"],
        packet=packet,
    )
    if rebuilt != payload:
        raise PC1PoseStreamError("composition archive is not canonical on re-emission")
    return parent_bytes, packet, manifest


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def output_effect_owners() -> tuple[dict[str, str], ...]:
    """Typed #417 owner ledger: each PC1 output effect has exactly one owner."""

    return (
        {
            "effect": "pose_conditioned_two_frame_multi_depth_warp",
            "member": PACKET_MEMBER,
            "owner": "ddm.pc1.pose_stream",
        },
        {
            "effect": "four_phase_luma_residual",
            "member": PACKET_MEMBER,
            "owner": "ddm.pc1.pose_stream",
        },
    )


def verify_unique_output_effect_owners(rows: Iterable[dict[str, str]]) -> bool:
    effects: set[str] = set()
    for row in rows:
        if set(row) != {"effect", "member", "owner"}:
            raise PC1PoseStreamError("output-effect owner row schema differs")
        if row["effect"] in effects or row["member"] != PACKET_MEMBER:
            return False
        if row["owner"] != "ddm.pc1.pose_stream":
            return False
        effects.add(row["effect"])
    return effects == {
        "pose_conditioned_two_frame_multi_depth_warp",
        "four_phase_luma_residual",
    }


__all__ = [
    "AXIS_NAMES",
    "CAMERA_H",
    "CAMERA_W",
    "LUMA_PHASES",
    "MANIFEST_MEMBER",
    "PACKET_MEMBER",
    "PAIR_H",
    "PAIR_W",
    "PARENT_MEMBER",
    "POSE_DIMS",
    "DDMPC1TrainableParameterMapV1",
    "PC1ParameterCoordinateV1",
    "PC1PosePacketV1",
    "PC1PoseStreamError",
    "active_tube_quadratic",
    "build_counted_composition_archive",
    "conditional_score_delta",
    "fresh_pose_initialization",
    "ground_and_movable_depth",
    "make_inactive_packet",
    "make_zero_active_packet",
    "output_effect_owners",
    "packet_sha256",
    "parse_counted_composition_archive",
    "parse_pc1_packet",
    "receive_pc1_camera_pairs",
    "serialize_pc1_packet",
    "sha256_bytes",
    "solved_plane_yuv6_target",
    "verify_unique_output_effect_owners",
]
