# SPDX-License-Identifier: MIT
"""OBX2 packet grammar: the born QBF sections plus a versioned lattice section.

The OBX2 object replaces the QBF per-pair latent section with a counted,
multiresolution, edge-local feature lattice.  This module owns the byte grammar
and the deterministic NumPy-fp32 reference receiver for that lattice.

Design boundary (contest rule 118).  Every value this module encodes is
video-derived and therefore COUNTED inside the archive: lattice codes, per-level
scales, the fusion weights, and the quantizer metadata.  The query algorithm,
the interpolation, and the gate are generic code and are free.  Nothing here
reads a scorer, the ground truth, or any content outside the packet.

The grammar is deliberately its own magic and version rather than an extension
of `experiments/ddm_qbflow_packet.py`: the born packet is a frozen, pinned input
of other live work and is never mutated here.  Sections 1-4 are carried through
verbatim from the born packet so the born generator decodes bit-identically.
"""

from __future__ import annotations

import hashlib
import io
import lzma
import struct
import zlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import brotli
import numpy as np

MAGIC = b"OBX2PKT1"
VERSION = 1

SECTION_CONFIG = 1
SECTION_MODEL = 2
SECTION_LATENT_META = 3
SECTION_LATENTS = 4
SECTION_LATTICE = 5
SECTION_NAMES = {
    SECTION_CONFIG: "config",
    SECTION_MODEL: "model",
    SECTION_LATENT_META: "latent_meta",
    SECTION_LATENTS: "latents",
    SECTION_LATTICE: "lattice",
}
CARRIED_SECTIONS = (SECTION_CONFIG, SECTION_MODEL, SECTION_LATENT_META)

CODEC_IDS = {"brotli_q11": 1, "zlib9": 2, "lzma9e": 3}
CODEC_NAMES = {value: key for key, value in CODEC_IDS.items()}

LATTICE_MAGIC = b"OBX2LAT1"
LATTICE_VERSION = 1
SUPPORTED_BITS = (4, 8, 16)
# 0 no gate; 1 the correction is multiplied by the interface gate (edge-local
# only); 2 the head emits TWO corrections and blends them with the gate, so the
# object can put capacity at class interfaces AND away from them.  Kind 2 exists
# because the measured binding term is Pose and Pose damage is NOT edge-local:
# uniform noise applied only OUTSIDE the argmax boundary band still drives
# d_pose to 2.82 on a 4-pair smoke, so a purely edge-gated correction cannot
# reach the term that binds.
GATE_KINDS = {0: "none", 1: "interface_softgate", 2: "interface_blend"}
GATED_KINDS = (1, 2)
QUANTIZER_KINDS = {0: "symmetric_uniform_per_level"}
ENTROPY_MODELS = {0: "section_codec_race"}

_PACKET_HEADER = struct.Struct(">8sBBH")
_SECTION_HEADER = struct.Struct(">BBHII32sI")
_LATTICE_HEADER = struct.Struct(">8sBBBBBBBHHHI")
_LEVEL_HEADER = struct.Struct(">HHHBf")
_MLP_HEADER = struct.Struct(">HHHH")


class OBX2PacketError(RuntimeError):
    """Fail-closed refusal for OBX2 packet grammar, integrity, or shape drift."""


@dataclass(frozen=True)
class LatticeSpec:
    """Explicit, self-describing geometry of the counted lattice."""

    levels: tuple[tuple[int, int, int], ...]
    channels: int
    bits: tuple[int, ...]
    gate_kind: int
    quantizer_kind: int
    entropy_model: int
    condition_channels: int
    hidden: int
    outputs: int

    def __post_init__(self) -> None:
        if not self.levels or len(self.levels) > 255:
            raise OBX2PacketError("lattice must carry between 1 and 255 levels")
        if len(self.bits) != len(self.levels):
            raise OBX2PacketError("lattice bit widths must be declared per level")
        for depth, height, width in self.levels:
            if min(depth, height, width) < 1 or max(depth, height, width) > 65_535:
                raise OBX2PacketError("lattice level geometry is out of range")
        if any(width not in SUPPORTED_BITS for width in self.bits):
            raise OBX2PacketError(f"lattice bit width must be one of {SUPPORTED_BITS}")
        if not 1 <= self.channels <= 255:
            raise OBX2PacketError("lattice channel width is out of range")
        if self.gate_kind not in GATE_KINDS:
            raise OBX2PacketError("unknown lattice gate kind")
        if self.quantizer_kind not in QUANTIZER_KINDS:
            raise OBX2PacketError("unknown lattice quantizer kind")
        if self.entropy_model not in ENTROPY_MODELS:
            raise OBX2PacketError("unknown lattice entropy model")
        if not 1 <= self.outputs <= 255 or not 1 <= self.hidden <= 65_535:
            raise OBX2PacketError("lattice fusion geometry is out of range")
        if not 0 <= self.condition_channels <= 255:
            raise OBX2PacketError("lattice condition width is out of range")

    @property
    def feature_width(self) -> int:
        return len(self.levels) * self.channels + self.condition_channels

    def codes_per_level(self) -> tuple[int, ...]:
        return tuple(depth * height * width * self.channels for depth, height, width in self.levels)

    def describe(self) -> dict[str, Any]:
        return {
            "levels": [list(level) for level in self.levels],
            "channels": self.channels,
            "bits": list(self.bits),
            "gate_kind": GATE_KINDS[self.gate_kind],
            "quantizer_kind": QUANTIZER_KINDS[self.quantizer_kind],
            "entropy_model": ENTROPY_MODELS[self.entropy_model],
            "condition_channels": self.condition_channels,
            "hidden": self.hidden,
            "outputs": self.outputs,
            "codes_per_level": list(self.codes_per_level()),
            "total_codes": sum(self.codes_per_level()),
        }


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def fusion_parameter_order(spec: LatticeSpec) -> tuple[str, ...]:
    """Serialization order of the counted fusion parameters for this gate kind."""

    base = ("hidden_w", "hidden_b", "out_w", "out_b")
    return (*base, "gate_tau") if spec.gate_kind in GATED_KINDS else base


def fusion_parameter_names(spec: LatticeSpec) -> set[str]:
    return set(fusion_parameter_order(spec))


def fusion_parameter_shape(spec: LatticeSpec, name: str) -> tuple[int, ...]:
    shapes = {
        "hidden_w": (spec.feature_width, spec.hidden),
        "hidden_b": (spec.hidden,),
        "out_w": (spec.hidden, spec.outputs),
        "out_b": (spec.outputs,),
        "gate_tau": (1,),
    }
    if name not in shapes:
        raise OBX2PacketError(f"unknown lattice fusion parameter: {name}")
    return shapes[name]


def interface_gate(signed_interfaces: np.ndarray, tau: float) -> np.ndarray:
    """Deterministic edge-local gate from the born generator's own decoded state.

    The gate is a generic function of the decoded signed-interface field: it is
    near one where some interface is close to zero (a class boundary) and decays
    away from every interface.  No support map, position list, or mask is
    shipped; the receiver recomputes this from sections 1-4.
    """

    if tau <= 0.0:
        raise OBX2PacketError("interface gate width must be positive")
    nearest = np.abs(np.asarray(signed_interfaces, dtype=np.float32)).min(axis=-1)
    # Clip before squaring so a far-from-every-interface pixel underflows the
    # gate to zero instead of overflowing the square to an invalid float.
    ratio = np.clip(nearest / np.float32(tau), 0.0, 1.0e9).astype(np.float32)
    return np.exp(-np.minimum(np.square(ratio), np.float32(80.0))).astype(np.float32)


def pack_codes(codes: np.ndarray, bits: int) -> bytes:
    """Pack signed integer codes at 4, 8, or 16 bits, little-endian, no padding bias."""

    flat = np.asarray(codes, dtype=np.int64).reshape(-1)
    limit = 1 << (bits - 1)
    if flat.size and (int(flat.min()) < -limit or int(flat.max()) > limit - 1):
        raise OBX2PacketError(f"lattice code exceeds the declared {bits}-bit range")
    if bits == 8:
        return flat.astype(np.int8).tobytes(order="C")
    if bits == 16:
        return flat.astype("<i2").tobytes(order="C")
    unsigned = (flat + limit).astype(np.uint8)
    if unsigned.size % 2:
        unsigned = np.concatenate([unsigned, np.zeros(1, dtype=np.uint8)])
    return ((unsigned[0::2] << 4) | unsigned[1::2]).astype(np.uint8).tobytes(order="C")


def unpack_codes(payload: bytes, count: int, bits: int) -> np.ndarray:
    if bits == 8:
        if len(payload) != count:
            raise OBX2PacketError("packed 8-bit lattice length differs")
        return np.frombuffer(payload, dtype=np.int8).astype(np.int64).copy()
    if bits == 16:
        if len(payload) != 2 * count:
            raise OBX2PacketError("packed 16-bit lattice length differs")
        return np.frombuffer(payload, dtype="<i2").astype(np.int64).copy()
    expected = (count + 1) // 2
    if len(payload) != expected:
        raise OBX2PacketError("packed 4-bit lattice length differs")
    raw = np.frombuffer(payload, dtype=np.uint8)
    high = (raw >> 4).astype(np.int64)
    low = (raw & 0x0F).astype(np.int64)
    interleaved = np.empty(raw.size * 2, dtype=np.int64)
    interleaved[0::2] = high
    interleaved[1::2] = low
    return interleaved[:count] - (1 << (bits - 1))


def packed_length(count: int, bits: int) -> int:
    if bits == 8:
        return count
    if bits == 16:
        return 2 * count
    return (count + 1) // 2


def encode_lattice_section(
    spec: LatticeSpec,
    *,
    codes: Sequence[np.ndarray],
    scales: Sequence[float],
    fusion: Mapping[str, np.ndarray],
) -> bytes:
    """Serialize the counted lattice: explicit geometry, per-level codes, fusion, CRC."""

    if len(codes) != len(spec.levels) or len(scales) != len(spec.levels):
        raise OBX2PacketError("lattice code/scale count differs from the level count")
    required = fusion_parameter_names(spec)
    if set(fusion) != required:
        raise OBX2PacketError(f"lattice fusion parameter set must be exactly {sorted(required)}")
    if fusion["hidden_w"].shape != (spec.feature_width, spec.hidden):
        raise OBX2PacketError("lattice fusion hidden weight shape differs")
    if fusion["hidden_b"].shape != (spec.hidden,):
        raise OBX2PacketError("lattice fusion hidden bias shape differs")
    if fusion["out_w"].shape != (spec.hidden, spec.outputs):
        raise OBX2PacketError("lattice fusion output weight shape differs")
    if fusion["out_b"].shape != (spec.outputs,):
        raise OBX2PacketError("lattice fusion output bias shape differs")
    if "gate_tau" in required:
        tau = np.asarray(fusion["gate_tau"], dtype=np.float64)
        if tau.shape != (1,) or not np.isfinite(tau).all() or float(tau[0]) <= 0.0:
            raise OBX2PacketError("lattice gate width must be a single positive finite value")

    body = io.BytesIO()
    for index, ((depth, height, width), bits, scale) in enumerate(zip(spec.levels, spec.bits, scales, strict=True)):
        level_codes = np.asarray(codes[index])
        if level_codes.ndim == 4 and level_codes.shape != (depth, height, width, spec.channels):
            raise OBX2PacketError("lattice level code array shape differs from its declared geometry")
        if level_codes.size != spec.codes_per_level()[index]:
            raise OBX2PacketError("lattice level code count differs from its declared geometry")
        if not np.isfinite(scale) or scale <= 0.0:
            raise OBX2PacketError("lattice level scale must be finite and positive")
        packed = pack_codes(level_codes, bits)
        body.write(_LEVEL_HEADER.pack(depth, height, width, bits, float(scale)))
        body.write(struct.pack(">I", len(packed)))
        body.write(packed)
    body.write(_MLP_HEADER.pack(spec.feature_width, spec.hidden, spec.outputs, spec.condition_channels))
    for name in fusion_parameter_order(spec):
        values = np.asarray(fusion[name], dtype="<f4")
        if not np.isfinite(values).all():
            raise OBX2PacketError(f"lattice fusion parameter is not finite: {name}")
        body.write(values.tobytes(order="C"))
    payload = body.getvalue()
    header = _LATTICE_HEADER.pack(
        LATTICE_MAGIC,
        LATTICE_VERSION,
        len(spec.levels),
        spec.channels,
        spec.gate_kind,
        spec.quantizer_kind,
        spec.entropy_model,
        spec.condition_channels,
        spec.hidden,
        spec.outputs,
        0,
        zlib.crc32(payload) & 0xFFFFFFFF,
    )
    return header + payload


def decode_lattice_section(raw: bytes) -> tuple[LatticeSpec, list[np.ndarray], list[float], dict[str, np.ndarray]]:
    """Parse the lattice section, refusing unknown, short, or trailing content."""

    if len(raw) < _LATTICE_HEADER.size:
        raise OBX2PacketError("lattice section is truncated")
    (
        magic,
        version,
        level_count,
        channels,
        gate_kind,
        quantizer_kind,
        entropy_model,
        condition_channels,
        hidden,
        outputs,
        reserved,
        crc,
    ) = _LATTICE_HEADER.unpack_from(raw)
    if magic != LATTICE_MAGIC or version != LATTICE_VERSION or reserved != 0:
        raise OBX2PacketError("lattice section header mismatch")
    body = raw[_LATTICE_HEADER.size :]
    if zlib.crc32(body) & 0xFFFFFFFF != crc:
        raise OBX2PacketError("lattice section CRC mismatch")

    offset = 0
    levels: list[tuple[int, int, int]] = []
    bits: list[int] = []
    codes: list[np.ndarray] = []
    scales: list[float] = []
    for _ in range(level_count):
        if offset + _LEVEL_HEADER.size + 4 > len(body):
            raise OBX2PacketError("lattice level header is truncated")
        depth, height, width, level_bits, scale = _LEVEL_HEADER.unpack_from(body, offset)
        offset += _LEVEL_HEADER.size
        (length,) = struct.unpack_from(">I", body, offset)
        offset += 4
        if level_bits not in SUPPORTED_BITS:
            raise OBX2PacketError("lattice level declares an unsupported bit width")
        count = depth * height * width * channels
        if length != packed_length(count, level_bits) or offset + length > len(body):
            raise OBX2PacketError("lattice level payload length differs")
        codes.append(unpack_codes(body[offset : offset + length], count, level_bits))
        offset += length
        levels.append((depth, height, width))
        bits.append(level_bits)
        if not np.isfinite(scale) or scale <= 0.0:
            raise OBX2PacketError("lattice level scale is not a positive finite value")
        scales.append(float(scale))

    if offset + _MLP_HEADER.size > len(body):
        raise OBX2PacketError("lattice fusion header is truncated")
    feature_width, mlp_hidden, mlp_outputs, mlp_condition = _MLP_HEADER.unpack_from(body, offset)
    offset += _MLP_HEADER.size
    spec = LatticeSpec(
        levels=tuple(levels),
        channels=channels,
        bits=tuple(bits),
        gate_kind=gate_kind,
        quantizer_kind=quantizer_kind,
        entropy_model=entropy_model,
        condition_channels=condition_channels,
        hidden=hidden,
        outputs=outputs,
    )
    if (feature_width, mlp_hidden, mlp_outputs, mlp_condition) != (
        spec.feature_width,
        spec.hidden,
        spec.outputs,
        spec.condition_channels,
    ):
        raise OBX2PacketError("lattice fusion header disagrees with the declared geometry")

    fusion: dict[str, np.ndarray] = {}
    for name in fusion_parameter_order(spec):
        shape = fusion_parameter_shape(spec, name)
        count = int(np.prod(shape))
        stop = offset + 4 * count
        if stop > len(body):
            raise OBX2PacketError(f"lattice fusion parameter is truncated: {name}")
        values = np.frombuffer(body[offset:stop], dtype="<f4").reshape(shape).astype(np.float32).copy()
        if not np.isfinite(values).all():
            raise OBX2PacketError(f"decoded lattice fusion parameter is not finite: {name}")
        fusion[name] = values
        offset = stop
    if offset != len(body):
        raise OBX2PacketError("lattice section carries trailing content")
    return spec, codes, scales, fusion


def compress(codec_name: str, raw: bytes) -> bytes:
    if codec_name == "brotli_q11":
        return brotli.compress(raw, quality=11)
    if codec_name == "lzma9e":
        return lzma.compress(raw, format=lzma.FORMAT_XZ, preset=9 | lzma.PRESET_EXTREME)
    if codec_name == "zlib9":
        return zlib.compress(raw, level=9)
    raise OBX2PacketError(f"unknown coder: {codec_name}")


def decompress(codec_name: str, payload: bytes) -> bytes:
    try:
        if codec_name == "brotli_q11":
            return brotli.decompress(payload)
        if codec_name == "lzma9e":
            return lzma.decompress(payload, format=lzma.FORMAT_XZ)
        if codec_name == "zlib9":
            return zlib.decompress(payload)
    except (brotli.error, lzma.LZMAError, zlib.error) as exc:
        raise OBX2PacketError(f"{codec_name} decompression failed") from exc
    raise OBX2PacketError(f"unknown coder: {codec_name}")


@dataclass(frozen=True)
class RacedSection:
    section_id: int
    codec_id: int
    raw_bytes: int
    payload: bytes
    raw_sha256: str
    candidates: tuple[tuple[str, int], ...]


def race_section(section_id: int, raw: bytes) -> RacedSection:
    """Encode one section with every coder, verify each round trip, keep the smallest."""

    if section_id not in SECTION_NAMES:
        raise OBX2PacketError(f"unknown OBX2 section: {section_id}")
    best: RacedSection | None = None
    sizes: list[tuple[str, int]] = []
    for codec_name, codec_id in CODEC_IDS.items():
        payload = compress(codec_name, raw)
        if decompress(codec_name, payload) != raw:
            raise OBX2PacketError(f"coder roundtrip failed: {codec_name}")
        sizes.append((codec_name, len(payload)))
        if best is None or (len(payload), codec_id) < (len(best.payload), best.codec_id):
            best = RacedSection(
                section_id=section_id,
                codec_id=codec_id,
                raw_bytes=len(raw),
                payload=payload,
                raw_sha256=sha256_bytes(raw),
                candidates=(),
            )
    if best is None:
        raise OBX2PacketError("section coder race produced no candidate")
    return RacedSection(
        section_id=best.section_id,
        codec_id=best.codec_id,
        raw_bytes=best.raw_bytes,
        payload=best.payload,
        raw_sha256=best.raw_sha256,
        candidates=tuple(sorted(sizes)),
    )


def pack_obx2_packet(sections: Iterable[RacedSection]) -> bytes:
    ordered = sorted(sections, key=lambda row: row.section_id)
    if len({row.section_id for row in ordered}) != len(ordered):
        raise OBX2PacketError("duplicate OBX2 packet section")
    if not ordered:
        raise OBX2PacketError("OBX2 packet must carry at least one section")
    output = io.BytesIO()
    output.write(_PACKET_HEADER.pack(MAGIC, VERSION, 0, len(ordered)))
    for section in ordered:
        output.write(
            _SECTION_HEADER.pack(
                section.section_id,
                section.codec_id,
                0,
                section.raw_bytes,
                len(section.payload),
                bytes.fromhex(section.raw_sha256),
                zlib.crc32(section.payload) & 0xFFFFFFFF,
            )
        )
        output.write(section.payload)
    return output.getvalue()


def decode_obx2_packet(payload: bytes) -> dict[int, bytes]:
    """Parse an OBX2 packet, refusing unknown, duplicate, disordered, or trailing content."""

    view = memoryview(payload)
    if len(view) < _PACKET_HEADER.size:
        raise OBX2PacketError("truncated OBX2 packet header")
    magic, version, flags, section_count = _PACKET_HEADER.unpack_from(view)
    if magic != MAGIC or version != VERSION or flags != 0 or section_count < 1:
        raise OBX2PacketError("OBX2 packet header mismatch")
    offset = _PACKET_HEADER.size
    sections: dict[int, bytes] = {}
    last_section = 0
    for _ in range(section_count):
        if offset + _SECTION_HEADER.size > len(view):
            raise OBX2PacketError("truncated OBX2 section header")
        section_id, codec_id, reserved, raw_len, coded_len, raw_sha, coded_crc = _SECTION_HEADER.unpack_from(
            view, offset
        )
        offset += _SECTION_HEADER.size
        stop = offset + coded_len
        if (
            section_id not in SECTION_NAMES
            or section_id in sections
            or section_id <= last_section
            or codec_id not in CODEC_NAMES
            or reserved != 0
            or stop > len(view)
        ):
            raise OBX2PacketError("OBX2 section header mismatch")
        coded = bytes(view[offset:stop])
        if zlib.crc32(coded) & 0xFFFFFFFF != coded_crc:
            raise OBX2PacketError("OBX2 section CRC mismatch")
        raw = decompress(CODEC_NAMES[codec_id], coded)
        if len(raw) != raw_len or hashlib.sha256(raw).digest() != raw_sha:
            raise OBX2PacketError("OBX2 section integrity mismatch")
        sections[section_id] = raw
        last_section = section_id
        offset = stop
    if offset != len(view):
        raise OBX2PacketError("OBX2 packet carries trailing content")
    if set(sections) != set(SECTION_NAMES):
        raise OBX2PacketError("OBX2 packet section set differs from the declared grammar")
    return sections


def dequantize_level(codes: np.ndarray, scale: float, shape: tuple[int, ...]) -> np.ndarray:
    return (codes.astype(np.float32) * np.float32(scale)).reshape(shape)


def lattice_grids(spec: LatticeSpec, codes: Sequence[np.ndarray], scales: Sequence[float]) -> list[np.ndarray]:
    """Dequantize every decoded level into its [T,H,W,C] grid, in declaration order."""

    if len(codes) != len(spec.levels) or len(scales) != len(spec.levels):
        raise OBX2PacketError("decoded level count differs from the declared geometry")
    return [
        dequantize_level(code, scale, (*level, spec.channels))
        for code, scale, level in zip(codes, scales, spec.levels, strict=True)
    ]


def trilinear_sample(grid: np.ndarray, t: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Deterministic trilinear sample of a [T,H,W,C] grid at normalized [-1,1] coords."""

    depth, height, width, channels = grid.shape
    axes = []
    for coordinate, size in ((t, depth), (y, height), (x, width)):
        position = (np.asarray(coordinate, dtype=np.float32) + 1.0) * 0.5 * (size - 1)
        position = np.clip(position, 0.0, float(size - 1))
        low = np.floor(position).astype(np.int64)
        high = np.minimum(low + 1, size - 1)
        axes.append((low, high, (position - low).astype(np.float32)))
    (t0, t1, ft), (y0, y1, fy), (x0, x1, fx) = axes
    out = np.zeros((*np.broadcast(t0, y0, x0).shape, channels), dtype=np.float32)
    for ti, wt in ((t0, 1.0 - ft), (t1, ft)):
        for yi, wy in ((y0, 1.0 - fy), (y1, fy)):
            for xi, wx in ((x0, 1.0 - fx), (x1, fx)):
                out += grid[ti, yi, xi] * (wt * wy * wx)[..., None]
    return out


def query_lattice_numpy(
    spec: LatticeSpec,
    grids: Sequence[np.ndarray],
    fusion: Mapping[str, np.ndarray],
    *,
    t: np.ndarray,
    y: np.ndarray,
    x: np.ndarray,
    condition: np.ndarray,
) -> np.ndarray:
    """Reference receiver: sample every level, fuse with the decoded condition, emit outputs.

    `condition` carries the born generator's own decoded state (its signed
    interface field and any derived gate).  It is computed at decode time from
    sections 1-4; no position map or support mask is shipped.
    """

    if len(grids) != len(spec.levels):
        raise OBX2PacketError("lattice grid count differs from the declared levels")
    if condition.shape[-1] != spec.condition_channels:
        raise OBX2PacketError("lattice condition width differs from the declared geometry")
    features = [trilinear_sample(grid, t, y, x) for grid in grids]
    stacked = np.concatenate([*features, condition.astype(np.float32)], axis=-1)
    if stacked.shape[-1] != spec.feature_width:
        raise OBX2PacketError("fused lattice feature width differs from the declared geometry")
    # NumPy's BLAS matmul raises a spurious FE_DIVBYZERO on finite float32
    # operands that contain very small magnitudes (the gate underflows far from
    # every interface).  The invariant that matters is a finite OUTPUT, so the
    # flags are ignored and finiteness is checked explicitly instead.
    with np.errstate(all="ignore"):
        hidden = np.tanh(stacked @ fusion["hidden_w"] + fusion["hidden_b"])
        outputs = (hidden @ fusion["out_w"] + fusion["out_b"]).astype(np.float32)
    if not np.isfinite(outputs).all():
        raise OBX2PacketError("lattice query produced a non-finite correction")
    return outputs


def blend_correction(outputs: np.ndarray, gate: np.ndarray, gate_kind: int) -> np.ndarray:
    """Combine the fusion head's outputs with the decoded gate.

    Kind 1 scales one correction by the gate.  Kind 2 blends two corrections,
    `gate * near + (1 - gate) * far`, so the same lattice can act at class
    interfaces and away from them.  Kind 0 ignores the gate entirely.
    """

    if gate_kind not in GATE_KINDS:
        raise OBX2PacketError("unknown lattice gate kind")
    weights = np.asarray(gate, dtype=np.float32)[..., None]
    if gate_kind == 0:
        return np.asarray(outputs, dtype=np.float32)
    if gate_kind == 1:
        return np.asarray(outputs, dtype=np.float32) * weights
    values = np.asarray(outputs, dtype=np.float32)
    if values.shape[-1] % 2:
        raise OBX2PacketError("interface_blend head must emit an even output width")
    half = values.shape[-1] // 2
    return values[..., :half] * weights + values[..., half:] * (1.0 - weights)


__all__ = [
    "CARRIED_SECTIONS",
    "CODEC_IDS",
    "CODEC_NAMES",
    "GATE_KINDS",
    "LATTICE_VERSION",
    "MAGIC",
    "SECTION_CONFIG",
    "SECTION_LATENTS",
    "SECTION_LATENT_META",
    "SECTION_LATTICE",
    "SECTION_MODEL",
    "SECTION_NAMES",
    "SUPPORTED_BITS",
    "VERSION",
    "LatticeSpec",
    "OBX2PacketError",
    "RacedSection",
    "blend_correction",
    "decode_lattice_section",
    "decode_obx2_packet",
    "dequantize_level",
    "encode_lattice_section",
    "fusion_parameter_names",
    "fusion_parameter_order",
    "fusion_parameter_shape",
    "interface_gate",
    "lattice_grids",
    "pack_codes",
    "pack_obx2_packet",
    "packed_length",
    "query_lattice_numpy",
    "race_section",
    "trilinear_sample",
    "unpack_codes",
]
