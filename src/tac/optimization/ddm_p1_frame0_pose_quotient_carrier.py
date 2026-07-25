# SPDX-License-Identifier: MIT
"""Counted, frame-0-only PC1 PoseNet quotient carrier.

This is an additive and legacy-compatible PC1 subtype.  It does not alter the
existing ``pose/pc1.ddp`` packet or receiver.  Its only video-selected state is
one shared quantized actuator basis plus per-pair quantized coefficients; both
are stored in the counted packet.  The generic nearest-cell receiver is free
decoder code and preserves parent frame 1 byte-for-byte.
"""

from __future__ import annotations

import hashlib
import io
import json
import struct
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from tac.canonical_equations.ddm_p1_frame0_pose_quotient_carrier_20260725 import (
    MAX_CARRIER_BYTES,
    frame0_quotient_law,
)
from tac.optimization.ddm_min_description_contract import (
    LayerHome,
    StreamType,
    TypedStreamTag,
)

CAMERA_H = 874
CAMERA_W = 1164
CHANNELS = 3
PAIR_COUNT = 600
MAX_RANK = 6
GRID_H = 24
GRID_W = 32
PACKET_SCHEMA = "ddm.pc1.frame0_pose_quotient.packet.v1"
COMPOSITION_SCHEMA = "ddm.pc1.frame0_pose_quotient.composition.v1"
PACKET_MEMBER = "pose/pc1_frame0_quotient.ddp"
PARENT_MEMBER = "parent/g4.zip"
MANIFEST_MEMBER = "manifest/pc1_frame0_quotient.json"
OWNER = "ddm.pc1.pose_stream.frame0_quotient"
PACKET_MAGIC = b"DPC1F0Q1"
_HEADER = struct.Struct("<8sBBHBBB")
_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


class PC1Frame0QuotientError(ValueError):
    """Raised when packet, receiver, or counted-composition custody differs."""


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
        + b"\n"
    )


def _readonly(value: Any, *, dtype: str, shape: tuple[int, ...], label: str) -> np.ndarray:
    result = np.array(value, dtype=np.dtype(dtype), copy=True, order="C")
    if result.shape != shape:
        raise PC1Frame0QuotientError(f"{label} must have shape {shape}")
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PC1Frame0QuotientPacketV1:
    """Fixed-geometry counted basis and per-pair coordinate packet."""

    active: bool
    treatment: bool
    pair_count: int
    rank: int
    grid_h: int
    grid_w: int
    term_exponents: np.ndarray
    q_basis: np.ndarray
    q_coefficients: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.active, bool) or not isinstance(self.treatment, bool):
            raise PC1Frame0QuotientError("active and treatment must be bool")
        if (
            isinstance(self.pair_count, bool)
            or not isinstance(self.pair_count, int)
            or self.pair_count < 2
            or self.pair_count > 65_535
        ):
            raise PC1Frame0QuotientError("pair_count must fit uint16 and exceed one")
        if (
            isinstance(self.rank, bool)
            or not isinstance(self.rank, int)
            or not 1 <= self.rank <= MAX_RANK
        ):
            raise PC1Frame0QuotientError("rank must be in [1,6]")
        if (self.grid_h, self.grid_w) != (GRID_H, GRID_W):
            raise PC1Frame0QuotientError("P1 grid geometry must be the sealed 24x32 chart")
        exponents = _readonly(
            self.term_exponents,
            dtype="i1",
            shape=(self.rank,),
            label="term_exponents",
        )
        if np.any(exponents < -24) or np.any(exponents > 8):
            raise PC1Frame0QuotientError("term exponents fall outside exact float64 ldexp custody")
        basis = _readonly(
            self.q_basis,
            dtype="i1",
            shape=(self.rank, CHANNELS, self.grid_h, self.grid_w),
            label="q_basis",
        )
        coefficients = _readonly(
            self.q_coefficients,
            dtype="<i2",
            shape=(self.pair_count, self.rank),
            label="q_coefficients",
        )
        object.__setattr__(self, "term_exponents", exponents)
        object.__setattr__(self, "q_basis", basis)
        object.__setattr__(self, "q_coefficients", coefficients)
        if self.packet_bytes > MAX_CARRIER_BYTES:
            raise PC1Frame0QuotientError("counted carrier exceeds the delegated 30,000-byte cap")

    @property
    def packet_bytes(self) -> int:
        return (
            _HEADER.size
            + self.rank
            + self.rank * CHANNELS * self.grid_h * self.grid_w
            + self.pair_count * self.rank * 2
        )

    def decoded_lowres_residual(self, pair_ids: Sequence[int]) -> np.ndarray:
        ids = _pair_ids(pair_ids, self.pair_count)
        products = np.einsum(
            "nr,rchw->nrchw",
            self.q_coefficients[ids].astype(np.int32),
            self.q_basis.astype(np.int32),
            optimize=False,
        )
        scaled = np.ldexp(
            products.astype(np.float64),
            self.term_exponents.astype(np.int32)[None, :, None, None, None],
        )
        return np.ascontiguousarray(scaled.sum(axis=1, dtype=np.float64))


def _pair_ids(pair_ids: Sequence[int], pair_count: int) -> np.ndarray:
    ids = np.asarray(pair_ids)
    if ids.ndim != 1 or ids.dtype.kind not in "iu":
        raise PC1Frame0QuotientError("pair_ids must be a one-dimensional integer sequence")
    ids = np.ascontiguousarray(ids, dtype=np.int64)
    if np.any(ids < 0) or np.any(ids >= pair_count):
        raise PC1Frame0QuotientError("pair_ids fall outside packet custody")
    return ids


def serialize_packet(packet: PC1Frame0QuotientPacketV1) -> bytes:
    """Serialize fixed-width counted state without content-dependent compression."""

    return b"".join(
        (
            _HEADER.pack(
                PACKET_MAGIC,
                int(packet.active),
                int(packet.treatment),
                packet.pair_count,
                packet.rank,
                packet.grid_h,
                packet.grid_w,
            ),
            packet.term_exponents.astype("i1", copy=False).tobytes(order="C"),
            packet.q_basis.astype("i1", copy=False).tobytes(order="C"),
            packet.q_coefficients.astype("<i2", copy=False).tobytes(order="C"),
        )
    )


def parse_packet(payload: bytes) -> PC1Frame0QuotientPacketV1:
    if not isinstance(payload, bytes) or len(payload) < _HEADER.size:
        raise PC1Frame0QuotientError("P1 packet is not a complete byte string")
    magic, active, treatment, pair_count, rank, grid_h, grid_w = _HEADER.unpack_from(payload)
    if magic != PACKET_MAGIC or active not in (0, 1) or treatment not in (0, 1):
        raise PC1Frame0QuotientError("P1 packet header is noncanonical")
    if not 1 <= rank <= MAX_RANK or (grid_h, grid_w) != (GRID_H, GRID_W):
        raise PC1Frame0QuotientError("P1 packet geometry differs")
    expected = _HEADER.size + rank + rank * CHANNELS * grid_h * grid_w + pair_count * rank * 2
    if len(payload) != expected:
        raise PC1Frame0QuotientError("P1 packet byte count disagrees with typed geometry")
    cursor = _HEADER.size
    exponents = np.frombuffer(payload[cursor : cursor + rank], dtype="i1")
    cursor += rank
    basis_bytes = rank * CHANNELS * grid_h * grid_w
    basis = np.frombuffer(payload[cursor : cursor + basis_bytes], dtype="i1").reshape(
        rank,
        CHANNELS,
        grid_h,
        grid_w,
    )
    cursor += basis_bytes
    coefficients = np.frombuffer(payload[cursor:], dtype="<i2").reshape(pair_count, rank)
    packet = PC1Frame0QuotientPacketV1(
        active=bool(active),
        treatment=bool(treatment),
        pair_count=int(pair_count),
        rank=int(rank),
        grid_h=int(grid_h),
        grid_w=int(grid_w),
        term_exponents=exponents,
        q_basis=basis,
        q_coefficients=coefficients,
    )
    if serialize_packet(packet) != payload:
        raise PC1Frame0QuotientError("P1 packet is not in canonical re-emission form")
    return packet


def _nearest_indices(source: int, target: int) -> np.ndarray:
    # Integer center-coordinate nearest sampling, avoiding cross-host float drift.
    return np.minimum(((2 * np.arange(target, dtype=np.int64) + 1) * source) // (2 * target), source - 1)


def receive_frame0_quotient(
    *,
    parent_camera: np.ndarray,
    packet: PC1Frame0QuotientPacketV1,
    pair_ids: Sequence[int],
) -> np.ndarray:
    """Apply the counted low-rank chart to frame 0 and preserve frame 1 exactly."""

    parent = np.asarray(parent_camera)
    ids = _pair_ids(pair_ids, packet.pair_count)
    if (
        parent.dtype != np.uint8
        or parent.ndim != 5
        or parent.shape != (len(ids), 2, CAMERA_H, CAMERA_W, CHANNELS)
    ):
        raise PC1Frame0QuotientError("parent camera geometry differs")
    if not packet.active:
        return np.array(parent, copy=True, order="C")
    lowres = packet.decoded_lowres_residual(ids)
    y_index = _nearest_indices(packet.grid_h, CAMERA_H)
    x_index = _nearest_indices(packet.grid_w, CAMERA_W)
    residual = lowres[:, :, y_index[:, None], x_index[None, :]]
    residual = np.moveaxis(residual, 1, -1)
    frame0 = np.clip(
        np.rint(parent[:, 0].astype(np.float64) + residual),
        0.0,
        255.0,
    ).astype(np.uint8)
    return frame0_quotient_law(parent, frame0)


def packet_typed_stream_tags(packet: PC1Frame0QuotientPacketV1) -> tuple[TypedStreamTag, TypedStreamTag]:
    """Split counted packet bytes into shared SKELETON and per-pair FIBER homes."""

    fiber_bytes = packet.pair_count * packet.rank * 2
    skeleton_bytes = packet.packet_bytes - fiber_bytes
    citation = "L2_chart -> L3_raster(frame_0 only) -> L4_PoseNet -> L5_d_pose"
    return (
        TypedStreamTag(
            type=StreamType.SKELETON,
            layer_home=LayerHome.L2_CHART,
            evaluate_py_recursion_level_cited=citation,
            counted_bytes=skeleton_bytes,
            free_receiver_code=True,
        ),
        TypedStreamTag(
            type=StreamType.FIBER,
            layer_home=LayerHome.L2_CHART,
            evaluate_py_recursion_level_cited=citation,
            counted_bytes=fiber_bytes,
            free_receiver_code=True,
        ),
    )


def _zip_member(name: str, payload: bytes) -> tuple[zipfile.ZipInfo, bytes]:
    info = zipfile.ZipInfo(name, date_time=_ZIP_TIME)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o100644 << 16
    info.create_system = 3
    return info, payload


def build_counted_composition_archive(
    *,
    parent_archive: bytes,
    parent_sha256: str,
    packet: PC1Frame0QuotientPacketV1,
) -> bytes:
    if sha256_bytes(parent_archive) != parent_sha256:
        raise PC1Frame0QuotientError("P1 parent archive SHA-256 custody differs")
    packet_bytes = serialize_packet(packet)
    tags = packet_typed_stream_tags(packet)
    manifest = {
        "schema": COMPOSITION_SCHEMA,
        "owner": OWNER,
        "active": packet.active,
        "pair_count": packet.pair_count,
        "rank": packet.rank,
        "grid": [packet.grid_h, packet.grid_w],
        "packet_member": PACKET_MEMBER,
        "packet_bytes": len(packet_bytes),
        "packet_sha256": sha256_bytes(packet_bytes),
        "parent_member": PARENT_MEMBER,
        "parent_bytes": len(parent_archive),
        "parent_sha256": parent_sha256,
        "frame0_only": True,
        "frame1_parent_byte_identity": True,
        "typed_stream_tags": [tag.to_dict() for tag in tags],
    }
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", strict_timestamps=True) as archive:
        for name, value in (
            (MANIFEST_MEMBER, canonical_bytes(manifest)),
            (PACKET_MEMBER, packet_bytes),
            (PARENT_MEMBER, parent_archive),
        ):
            info, content = _zip_member(name, value)
            archive.writestr(info, content)
    result = output.getvalue()
    parsed_parent, parsed_packet, _ = parse_counted_composition_archive(result)
    if parsed_parent != parent_archive or serialize_packet(parsed_packet) != packet_bytes:
        raise PC1Frame0QuotientError("P1 composition failed immediate parse-back")
    return result


def parse_counted_composition_archive(
    payload: bytes,
) -> tuple[bytes, PC1Frame0QuotientPacketV1, dict[str, Any]]:
    try:
        with zipfile.ZipFile(io.BytesIO(payload), "r") as archive:
            if archive.namelist() != [MANIFEST_MEMBER, PACKET_MEMBER, PARENT_MEMBER]:
                raise PC1Frame0QuotientError("P1 composition member order/schema differs")
            manifest_bytes = archive.read(MANIFEST_MEMBER)
            packet_bytes = archive.read(PACKET_MEMBER)
            parent_bytes = archive.read(PARENT_MEMBER)
    except (zipfile.BadZipFile, KeyError) as exc:
        raise PC1Frame0QuotientError("P1 composition archive is invalid") from exc
    try:
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError as exc:
        raise PC1Frame0QuotientError("P1 manifest JSON is invalid") from exc
    if not isinstance(manifest, dict) or canonical_bytes(manifest) != manifest_bytes:
        raise PC1Frame0QuotientError("P1 manifest is not canonical JSON")
    packet = parse_packet(packet_bytes)
    tags = packet_typed_stream_tags(packet)
    expected = {
        "schema": COMPOSITION_SCHEMA,
        "owner": OWNER,
        "active": packet.active,
        "pair_count": packet.pair_count,
        "rank": packet.rank,
        "grid": [packet.grid_h, packet.grid_w],
        "packet_member": PACKET_MEMBER,
        "packet_bytes": len(packet_bytes),
        "packet_sha256": sha256_bytes(packet_bytes),
        "parent_member": PARENT_MEMBER,
        "parent_bytes": len(parent_bytes),
        "parent_sha256": sha256_bytes(parent_bytes),
        "frame0_only": True,
        "frame1_parent_byte_identity": True,
        "typed_stream_tags": [tag.to_dict() for tag in tags],
    }
    if manifest != expected:
        raise PC1Frame0QuotientError("P1 manifest custody differs")
    return parent_bytes, packet, manifest


def seeded_matched_control_basis(*, seed: int, rank: int) -> np.ndarray:
    """Return a deterministic untargeted Rademacher basis for the exact control."""

    if isinstance(seed, bool) or not isinstance(seed, int) or not 1 <= rank <= MAX_RANK:
        raise PC1Frame0QuotientError("control seed or rank differs")
    rng = np.random.default_rng(seed)
    values = rng.integers(
        0,
        2,
        size=(rank, CHANNELS, GRID_H, GRID_W),
        dtype=np.int8,
    )
    return np.ascontiguousarray(values * np.int8(2) - np.int8(1))


def make_packet(
    *,
    treatment: bool,
    rank: int,
    q_basis: np.ndarray,
    q_coefficients: np.ndarray,
    term_exponents: np.ndarray | None = None,
    active: bool = True,
) -> PC1Frame0QuotientPacketV1:
    if term_exponents is None:
        term_exponents = np.full(rank, -8, dtype=np.int8)
    return PC1Frame0QuotientPacketV1(
        active=active,
        treatment=treatment,
        pair_count=int(np.asarray(q_coefficients).shape[0]),
        rank=rank,
        grid_h=GRID_H,
        grid_w=GRID_W,
        term_exponents=term_exponents,
        q_basis=q_basis,
        q_coefficients=q_coefficients,
    )


__all__ = [
    "CAMERA_H",
    "CAMERA_W",
    "CHANNELS",
    "COMPOSITION_SCHEMA",
    "GRID_H",
    "GRID_W",
    "MANIFEST_MEMBER",
    "MAX_RANK",
    "OWNER",
    "PACKET_MEMBER",
    "PACKET_SCHEMA",
    "PAIR_COUNT",
    "PARENT_MEMBER",
    "PC1Frame0QuotientError",
    "PC1Frame0QuotientPacketV1",
    "build_counted_composition_archive",
    "canonical_bytes",
    "make_packet",
    "packet_typed_stream_tags",
    "parse_counted_composition_archive",
    "parse_packet",
    "receive_frame0_quotient",
    "seeded_matched_control_basis",
    "serialize_packet",
    "sha256_bytes",
]
