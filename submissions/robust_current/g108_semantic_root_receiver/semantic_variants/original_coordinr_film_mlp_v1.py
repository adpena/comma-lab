# SPDX-License-Identifier: MIT
"""Standalone NumPy receiver for the committed G103 original MLP packet."""

from __future__ import annotations

import hashlib
import struct
import zlib
from typing import Final

import numpy as np

VARIANT_ID: Final = "tac.semantic_root_y1.original_coordinr_film_mlp.v1"

MAGIC: Final = b"SRY1V1\x00\x00"
VERSION: Final = 1
PAIR_COUNT: Final = 600
H: Final = 384
W: Final = 512
C: Final = 3
MAX_PACKET_BYTES: Final = 2_000_000
MAX_TEMPLATES: Final = 256
MAX_EVENTS: Final = 1_024
MAX_TENSORS: Final = 24
MAX_BASIS: Final = 128
MAX_QUOTIENT: Final = 2_048
ALL_ROLE_MASK: Final = 0b1_1111

_HEADER = struct.Struct(">8sBBHHHBBHHHHHHH")
_SECTION_META = struct.Struct(">4sI")
_FOOTER = struct.Struct(">I")
_PROFILE = struct.Struct(">4s15B4BI")
_TEMPLATE = struct.Struct(">HBB6h")
_EVENT = struct.Struct(">5H4h")
_MODEL_HEADER = struct.Struct(">4sBBBBHHHHH")
_TENSOR_HEADER = struct.Struct(">HBBbB4HhI")
_LATENT_HEADER = struct.Struct(">4sBBHHhhI")
_RGB_HEADER = struct.Struct(">4sBBHH")
_RGB_BASIS = struct.Struct(">HBBHHHhhhBB")
_PAIR_GAUGE = struct.Struct(">HH6h")
_QUOTIENT_HEADER = struct.Struct(">4sH")
_QUOTIENT = struct.Struct(">HHHBBhhhhHHhhh")
_TAGS: Final = (b"PROF", b"TOPO", b"EVNT", b"MODL", b"LATN", b"RGBF", b"IRRQ")
_FORBIDDEN_MAGICS: Final = (
    b"PK\x03\x04",
    b"TACPVSA",
    b"DDV15S1",
    b"TACV10R",
    b"PVSA",
    b"TSPPV1",
    b"TSPPV2",
)
_FORBIDDEN_SHA256: Final = frozenset(
    {
        "759e2833f31d2182b80e1b2f434214f24d75cb487bbec554dc58abdc7d53e6bb",
        "b9c8ab2a5e2bf6cb775539156be1220d9f3f6b44fce38a2ecae70164027f512b",
        "d50aac6ea72df527f1630485c174b73ed25c2c7b41b685a24a53ccac21e6cf6c",
        "736d9c751b1578cead45bccb5e71a4bab2373353f079f96d5c6ec96694ae8d95",
        "e6f99e435fcbd45673bebea4049f8b8322d927a2276c37c995056e1ac4bbf4fe",
        "2b82e28e23e3b37fc305dc42f2320ed643726d27fe5e6805bf0978ac0e5c8fa8",
        "e4cd154fbd5540bf176102374c968dd9a07f7bd647108a4f24b28d19fb10dad7",
        "e3d0581f70ac91493ed9897e5e3d49819961477c56cac161a3e577010e683c7e",
    }
)

_GRID_Y, _GRID_X = np.indices((H, W), dtype=np.int32)
_FLAT_Y = _GRID_Y.reshape(-1)
_FLAT_X = _GRID_X.reshape(-1)
_Y64 = np.arange(H, dtype=np.int64)[:, None]
_X64 = np.arange(W, dtype=np.int64)[None, :]
_Y_Q4 = np.arange(H, dtype=np.int32)[:, None] * 16
_X_Q4 = np.arange(W, dtype=np.int32)[None, :] * 16


class SemanticVariantError(ValueError):
    """The counted G103 packet or its deterministic render failed closed."""


def accepts_packet(packet: bytes) -> bool:
    """Select by the packet's self-describing magic without counted metadata."""

    return type(packet) is bytes and packet.startswith(MAGIC)


class _Bits:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.offset = 0

    def bit(self) -> int:
        if self.offset >= len(self.payload) * 8:
            raise SemanticVariantError("temporal Rice stream is truncated")
        value = (self.payload[self.offset // 8] >> (7 - self.offset % 8)) & 1
        self.offset += 1
        return value

    def zero_padding(self) -> None:
        while self.offset < len(self.payload) * 8:
            if self.bit():
                raise SemanticVariantError("temporal Rice stream has noncanonical trailing bits")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _reject_foreign(payload: bytes, label: str) -> None:
    if _sha256(payload) in _FORBIDDEN_SHA256 or any(payload.startswith(magic) for magic in _FORBIDDEN_MAGICS):
        raise SemanticVariantError(f"{label} is a forbidden foreign/raster payload home")


def _round_scalar(numerator: int, denominator: int) -> int:
    sign = -1 if numerator < 0 else 1
    return sign * ((abs(numerator) + denominator // 2) // denominator)


def _round_array(numerator: np.ndarray, denominator: int) -> np.ndarray:
    magnitude = (np.abs(numerator) + denominator // 2) // denominator
    return np.where(numerator < 0, -magnitude, magnitude)


def _decode_latents(payload: bytes, latent_dim: int) -> np.ndarray:
    if len(payload) < _LATENT_HEADER.size:
        raise SemanticVariantError("temporal latent section is truncated")
    magic, codec, rice_k, pairs, inner_dim, value_min, value_max, byte_length = _LATENT_HEADER.unpack_from(payload)
    encoded = payload[_LATENT_HEADER.size :]
    if (
        magic != b"SRL1"
        or codec != 0
        or not 0 <= rice_k <= 15
        or pairs != PAIR_COUNT
        or inner_dim != latent_dim
        or len(encoded) != byte_length
        or not encoded
    ):
        raise SemanticVariantError("temporal latent header disagrees with packet")
    bits = _Bits(encoded)
    output = np.empty((PAIR_COUNT, latent_dim), dtype=np.int16)
    previous = np.zeros(latent_dim, dtype=np.int64)
    for pair_id in range(PAIR_COUNT):
        for column in range(latent_dim):
            quotient = 0
            while bits.bit():
                quotient += 1
                if quotient > 0xFFFF:
                    raise SemanticVariantError("temporal Rice quotient exceeds decoder bound")
            remainder = 0
            for _ in range(rice_k):
                remainder = (remainder << 1) | bits.bit()
            unsigned = (quotient << rice_k) | remainder
            delta = unsigned // 2 if not unsigned & 1 else -(unsigned // 2) - 1
            value = int(previous[column]) + delta
            if not -0x8000 <= value <= 0x7FFF:
                raise SemanticVariantError("temporal delta leaves int16 range")
            output[pair_id, column] = value
            previous[column] = value
    bits.zero_padding()
    if int(output.min()) != value_min or int(output.max()) != value_max:
        raise SemanticVariantError("temporal latent declared range differs")
    return output


def _decode_profile(payload: bytes) -> dict[str, object]:
    if len(payload) != _PROFILE.size:
        raise SemanticVariantError("profile section length differs")
    values = _PROFILE.unpack(payload)
    if values[0] != b"SRP1":
        raise SemanticVariantError("profile magic differs")
    role_rgb = tuple(tuple(int(c) for c in values[1 + offset : 4 + offset]) for offset in range(0, 15, 3))
    gains = tuple(int(value) for value in values[16:20])
    if len(set(role_rgb)) != 5 or any(not 1 <= gain <= 31 for gain in gains):
        raise SemanticVariantError("profile prototypes/gains are noncanonical")
    return {
        "role_rgb": role_rgb,
        "texture_gain": gains[0],
        "edge_gain": gains[1],
        "chroma_gain": gains[2],
        "parallax_gain": gains[3],
        "seed": int(values[20]),
    }


def _decode_templates(payload: bytes, count: int) -> tuple[tuple[int, ...], ...]:
    if len(payload) != count * _TEMPLATE.size:
        raise SemanticVariantError("topology template section length differs")
    result: list[tuple[int, ...]] = []
    for index in range(count):
        values = tuple(int(value) for value in _TEMPLATE.unpack_from(payload, index * _TEMPLATE.size))
        template_id, role, shape, *params = values
        if template_id != index or not 0 <= role <= 4 or not 0 <= shape <= 3:
            raise SemanticVariantError("topology template ID/enum differs")
        if shape in (0, 1) and (params[2] <= 0 or params[3] <= 0):
            raise SemanticVariantError("RECT/ELLIPSE extents must be positive")
        if shape == 2 and (params[2] <= params[1] or params[3] <= 0 or params[4] <= 0):
            raise SemanticVariantError("TRAPEZOID parameters are invalid")
        if shape == 3 and (params[4] <= 0 or params[5] <= 0):
            raise SemanticVariantError("QUADRATIC_STRIP parameters are invalid")
        result.append(values)
    return tuple(result)


def _decode_events(
    payload: bytes,
    count: int,
    template_count: int,
) -> tuple[tuple[int, ...], ...]:
    if len(payload) != count * _EVENT.size:
        raise SemanticVariantError("topology event section length differs")
    result: list[tuple[int, ...]] = []
    for index in range(count):
        values = tuple(int(value) for value in _EVENT.unpack_from(payload, index * _EVENT.size))
        event_id, template_id, start, stop = values[:4]
        if event_id != index or not 0 <= template_id < template_count or not 0 <= start < stop <= PAIR_COUNT:
            raise SemanticVariantError("topology event identity/interval differs")
        result.append(values)
    return tuple(result)


def _decode_model(payload: bytes, tensor_count: int) -> dict[str, object]:
    if len(payload) < _MODEL_HEADER.size:
        raise SemanticVariantError("model section is truncated")
    (
        magic,
        architecture,
        numeric_contract,
        activation,
        reserved,
        input_dim,
        hidden_dim,
        layer_count,
        modulation_dim,
        inner_count,
    ) = _MODEL_HEADER.unpack_from(payload)
    if (
        magic != b"SRM1"
        or architecture != 0
        or numeric_contract != 0
        or activation not in (0, 1)
        or reserved != 0
        or not 4 <= input_dim <= 32
        or not 8 <= hidden_dim <= 256
        or not 1 <= layer_count <= 8
        or not 1 <= modulation_dim <= 64
        or inner_count != tensor_count
    ):
        raise SemanticVariantError("model header is unsupported/noncanonical")
    expected: list[tuple[int, tuple[int, ...]]] = [
        (0, (hidden_dim, input_dim)),
        (1, (hidden_dim,)),
    ]
    for _ in range(layer_count):
        expected.extend(((2, (hidden_dim, hidden_dim)), (3, (hidden_dim,))))
    expected.extend(
        (
            (4, (2 * hidden_dim, modulation_dim)),
            (5, (2 * hidden_dim,)),
            (6, (C, hidden_dim)),
            (7, (C,)),
        )
    )
    if len(expected) != tensor_count:
        raise SemanticVariantError("model tensor count differs from reviewed architecture")
    cursor = _MODEL_HEADER.size
    tensors: list[np.ndarray] = []
    all_zero = True
    for tensor_id, (expected_role, expected_shape) in enumerate(expected):
        if cursor + _TENSOR_HEADER.size > len(payload):
            raise SemanticVariantError("tensor header is truncated")
        row = _TENSOR_HEADER.unpack_from(payload, cursor)
        cursor += _TENSOR_HEADER.size
        observed_id, role, dtype, exponent, rank = row[:5]
        padded_shape = tuple(int(value) for value in row[5:9])
        zero_point = int(row[9])
        byte_length = int(row[10])
        shape = padded_shape[:rank]
        if (
            observed_id != tensor_id
            or role != expected_role
            or dtype != (0 if role in (0, 2, 4, 6) else 1)
            or exponent != (-7 if dtype == 0 else -12)
            or zero_point != 0
            or not 1 <= rank <= 4
            or shape != expected_shape
            or any(padded_shape[index] != 0 for index in range(rank, 4))
        ):
            raise SemanticVariantError("tensor role/shape/quantization differs")
        expected_bytes = int(np.prod(shape, dtype=np.int64)) * (1 if dtype == 0 else 2)
        if byte_length != expected_bytes or cursor + byte_length > len(payload):
            raise SemanticVariantError("tensor byte length differs")
        data = payload[cursor : cursor + byte_length]
        cursor += byte_length
        _reject_foreign(data, "model tensor")
        all_zero &= all(value == 0 for value in data)
        array_dtype = np.dtype("i1") if dtype == 0 else np.dtype(">i2")
        tensors.append(np.frombuffer(data, dtype=array_dtype).astype(np.int64).reshape(shape))
    if cursor != len(payload) or all_zero:
        raise SemanticVariantError("model has trailing bytes or is all zero")
    return {
        "activation": int(activation),
        "input_dim": int(input_dim),
        "hidden_dim": int(hidden_dim),
        "layer_count": int(layer_count),
        "modulation_dim": int(modulation_dim),
        "tensors": tuple(tensors),
    }


def _decode_rgb(
    payload: bytes,
    basis_count: int,
    gauge_count: int,
) -> tuple[int, tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...]]:
    expected_size = _RGB_HEADER.size + basis_count * _RGB_BASIS.size + gauge_count * _PAIR_GAUGE.size
    if len(payload) != expected_size:
        raise SemanticVariantError("RGB field section length differs")
    magic, ownership, reserved, inner_basis, inner_gauges = _RGB_HEADER.unpack_from(payload)
    if (
        magic != b"SRF1"
        or ownership not in (0, 1)
        or reserved != 0
        or (inner_basis, inner_gauges) != (basis_count, gauge_count)
        or gauge_count != (0 if ownership == 0 else PAIR_COUNT)
    ):
        raise SemanticVariantError("RGB field header/ownership differs")
    cursor = _RGB_HEADER.size
    basis: list[tuple[int, ...]] = []
    for atom_id in range(basis_count):
        values = tuple(int(value) for value in _RGB_BASIS.unpack_from(payload, cursor))
        cursor += _RGB_BASIS.size
        (
            observed_id,
            role_mask,
            wave_kind,
            frequency_x,
            frequency_y,
            _phase,
            amplitude_r,
            amplitude_g,
            amplitude_b,
            edge_width,
            inner_reserved,
        ) = values
        if (
            observed_id != atom_id
            or not 1 <= role_mask <= ALL_ROLE_MASK
            or wave_kind not in (0, 1, 2)
            or not 0 <= frequency_x <= 64
            or not 0 <= frequency_y <= 64
            or frequency_x == frequency_y == 0
            or not all(-255 <= value <= 255 for value in (amplitude_r, amplitude_g, amplitude_b))
            or (amplitude_r, amplitude_g, amplitude_b) == (0, 0, 0)
            or not 0 <= edge_width <= 16
            or inner_reserved != 0
        ):
            raise SemanticVariantError("RGB basis atom is noncanonical")
        basis.append(values)
    gauges: list[tuple[int, ...]] = []
    for pair_id in range(gauge_count):
        values = tuple(int(value) for value in _PAIR_GAUGE.unpack_from(payload, cursor))
        cursor += _PAIR_GAUGE.size
        if (
            values[0] != pair_id
            or not all(-255 <= value <= 255 for value in values[2:7])
            or not 1 <= values[7] <= 0x7FFF
        ):
            raise SemanticVariantError("pair RGB gauge is noncanonical")
        gauges.append(values)
    if basis and not any(row[1] == ALL_ROLE_MASK and row[9] == 0 for row in basis):
        raise SemanticVariantError("RGB basis omits mandatory all-role non-edge field")
    if ownership == 0 and basis:
        raise SemanticVariantError("derived generator ownership cannot also carry procedural basis")
    return int(ownership), tuple(basis), tuple(gauges)


def _decode_quotient(payload: bytes, count: int) -> tuple[tuple[int, ...], ...]:
    if len(payload) != _QUOTIENT_HEADER.size + count * _QUOTIENT.size:
        raise SemanticVariantError("quotient section length differs")
    magic, inner_count = _QUOTIENT_HEADER.unpack_from(payload)
    if magic != b"SRQ1" or inner_count != count:
        raise SemanticVariantError("quotient header differs")
    result: list[tuple[int, ...]] = []
    cursor = _QUOTIENT_HEADER.size
    for atom_id in range(count):
        values = tuple(int(value) for value in _QUOTIENT.unpack_from(payload, cursor))
        cursor += _QUOTIENT.size
        amplitudes = values[11:14]
        if (
            values[0] != atom_id
            or not 0 <= values[1] < values[2] <= PAIR_COUNT
            or not 1 <= values[3] <= ALL_ROLE_MASK
            or values[4] not in (0, 1)
            or values[9] <= 0
            or values[10] <= 0
            or not all(-255 <= value <= 255 for value in amplitudes)
            or amplitudes == (0, 0, 0)
        ):
            raise SemanticVariantError("quotient atom is noncanonical")
        result.append(values)
    return tuple(result)


def parse_packet(packet: bytes) -> dict[str, object]:
    if type(packet) is not bytes or not 0 < len(packet) <= MAX_PACKET_BYTES:
        raise SemanticVariantError("packet must be nonempty exact bytes inside sparse cap")
    _reject_foreign(packet, "semantic-root packet")
    minimum = _HEADER.size + len(_TAGS) * _SECTION_META.size + _FOOTER.size
    if len(packet) < minimum:
        raise SemanticVariantError("semantic-root packet is truncated")
    values = _HEADER.unpack_from(packet)
    (
        magic,
        version,
        flags,
        pairs,
        scorer_h,
        scorer_w,
        channels,
        background_role,
        template_count,
        event_count,
        tensor_count,
        latent_dim,
        basis_count,
        gauge_count,
        quotient_count,
    ) = values
    if (
        magic != MAGIC
        or version != VERSION
        or flags != 0
        or (pairs, scorer_h, scorer_w, channels) != (PAIR_COUNT, H, W, C)
        or not 0 <= background_role <= 4
        or not 0 <= template_count <= MAX_TEMPLATES
        or not 0 <= event_count <= MAX_EVENTS
        or (template_count == 0) != (event_count == 0)
        or not 1 <= tensor_count <= MAX_TENSORS
        or not 1 <= latent_dim <= 64
        or not 0 <= basis_count <= MAX_BASIS
        or gauge_count not in (0, PAIR_COUNT)
        or not 0 <= quotient_count <= MAX_QUOTIENT
    ):
        raise SemanticVariantError("packet header/cardinality differs from exact n600 wire")
    cursor = _HEADER.size
    metas: list[tuple[bytes, int]] = []
    for expected_tag in _TAGS:
        tag, length = _SECTION_META.unpack_from(packet, cursor)
        cursor += _SECTION_META.size
        if tag != expected_tag:
            raise SemanticVariantError("packet sections are absent, reordered, or duplicated")
        metas.append((tag, int(length)))
    if cursor + sum(length for _, length in metas) + _FOOTER.size != len(packet):
        raise SemanticVariantError("packet exact EOF differs")
    sections: list[bytes] = []
    for tag, length in metas:
        section = packet[cursor : cursor + length]
        cursor += length
        _reject_foreign(section, tag.decode("ascii"))
        sections.append(section)
    (expected_crc,) = _FOOTER.unpack_from(packet, cursor)
    if expected_crc != zlib.crc32(b"".join(sections)) & 0xFFFFFFFF:
        raise SemanticVariantError("packet body CRC32 differs")
    profile = _decode_profile(sections[0])
    templates = _decode_templates(sections[1], int(template_count))
    events = _decode_events(sections[2], int(event_count), int(template_count))
    model = _decode_model(sections[3], int(tensor_count))
    if model["modulation_dim"] != latent_dim:
        raise SemanticVariantError("model modulation width differs from temporal latent width")
    latents = _decode_latents(sections[4], int(latent_dim))
    ownership, basis, gauges = _decode_rgb(sections[5], int(basis_count), int(gauge_count))
    quotient = _decode_quotient(sections[6], int(quotient_count))
    invariant = not events and ownership == 0 and not quotient and bool(np.all(latents == latents[0]))
    return {
        "background_role": int(background_role),
        "profile": profile,
        "templates": templates,
        "events": events,
        "model": model,
        "latents": latents,
        "ownership": ownership,
        "basis": basis,
        "gauges": gauges,
        "quotient": quotient,
        "pair_invariant": invariant,
        "cached_frame": None,
    }


def _labels(root: dict[str, object], pair_id: int) -> np.ndarray:
    labels = np.full((H, W), int(root["background_role"]), dtype=np.uint8)
    templates = {row[0]: row for row in root["templates"]}  # type: ignore[union-attr]
    active = sorted(
        (
            row
            for row in root["events"]  # type: ignore[union-attr]
            if row[2] <= pair_id < row[3]
        ),
        key=lambda row: (row[4], row[0]),
    )
    for event in active:
        template = templates[event[1]]
        dt = pair_id - event[2]
        anchor_x = event[5] + _round_scalar(event[7] * dt, 16)
        anchor_y = event[6] + _round_scalar(event[8] * dt, 16)
        p0, p1, p2, p3, p4, p5 = template[3:]
        local_x = _X_Q4 - anchor_x
        local_y = _Y_Q4 - anchor_y
        if template[2] == 0:
            mask = (np.abs(local_x - p0) <= p2) & (np.abs(local_y - p1) <= p3)
        elif template[2] == 1:
            dx = local_x - p0
            dy = local_y - p1
            mask = dx * dx * p3 * p3 + dy * dy * p2 * p2 <= p2 * p2 * p3 * p3
        elif template[2] == 2:
            span = p2 - p1
            within_y = (local_y >= p1) & (local_y <= p2)
            half_width = p3 + ((local_y - p1) * (p4 - p3)) // span
            mask = within_y & (np.abs(local_x - p0) <= half_width)
        else:
            dy = local_y - p1
            curve_x = p0 + (p2 * dy) // 256 + (p3 * dy * dy) // (4096 * 16)
            mask = (np.abs(dy) <= p5) & (np.abs(local_x - curve_x) <= p4)
        labels[mask] = template[1]
    return labels


def _edges(labels: np.ndarray) -> np.ndarray:
    result = np.zeros_like(labels, dtype=bool)
    result[1:] |= labels[1:] != labels[:-1]
    result[:-1] |= labels[:-1] != labels[1:]
    result[:, 1:] |= labels[:, 1:] != labels[:, :-1]
    result[:, :-1] |= labels[:, :-1] != labels[:, 1:]
    return result


def _dilate(mask: np.ndarray, width: int) -> np.ndarray:
    result = mask.copy()
    for _ in range(max(0, width - 1)):
        expanded = result.copy()
        expanded[1:] |= result[:-1]
        expanded[:-1] |= result[1:]
        expanded[:, 1:] |= result[:, :-1]
        expanded[:, :-1] |= result[:, 1:]
        result = expanded
    return result


def _features(
    x: np.ndarray,
    y: np.ndarray,
    labels: np.ndarray,
    input_dim: int,
    seed: int,
) -> np.ndarray:
    result = np.empty((x.size, input_dim), dtype=np.int64)
    result[:, 0] = x.astype(np.int64) * 8192 // (W - 1) - 4096
    result[:, 1] = y.astype(np.int64) * 8192 // (H - 1) - 4096
    result[:, 2] = labels.astype(np.int64) * 2048 - 4096
    result[:, 3] = 4096
    for column in range(4, input_dim):
        fx = 1 + ((seed + 17 * column) % 31)
        fy = 1 + (((seed >> 8) + 29 * column) % 31)
        phase = (
            fx * x.astype(np.int64) * 256 // W + fy * y.astype(np.int64) * 256 // H + (seed >> ((column % 4) * 8))
        ) & 0xFF
        result[:, column] = (127 - 2 * np.abs(phase - 128)) * 4096 // 127
    return result


def _activate(values: np.ndarray, activation: int) -> np.ndarray:
    return np.clip(values, -4096, 4096) if activation == 0 else np.clip(values, 0, 6 * 4096)


def _generator(root: dict[str, object], pair_id: int, labels: np.ndarray) -> np.ndarray:
    model = root["model"]
    tensors = model["tensors"]  # type: ignore[index]
    layer_count = int(model["layer_count"])  # type: ignore[index]
    output_weight = tensors[2 + 2 * layer_count + 2]
    output_bias = tensors[2 + 2 * layer_count + 3]
    if not np.any(output_weight) and not np.any(output_bias):
        return np.zeros((H, W, C), dtype=np.int32)
    latent = root["latents"][pair_id].astype(np.int64)  # type: ignore[index]
    film_offset = 2 + 2 * layer_count
    film = _round_array(tensors[film_offset] @ latent, 128) + tensors[film_offset + 1]
    hidden_dim = int(model["hidden_dim"])  # type: ignore[index]
    gamma = film[:hidden_dim]
    beta = film[hidden_dim:]
    flat_labels = labels.reshape(-1)
    output = np.empty((flat_labels.size, C), dtype=np.int32)
    for start in range(0, flat_labels.size, 4096):
        stop = min(start + 4096, flat_labels.size)
        features = _features(
            _FLAT_X[start:stop],
            _FLAT_Y[start:stop],
            flat_labels[start:stop],
            int(model["input_dim"]),  # type: ignore[index]
            int(root["profile"]["seed"]),  # type: ignore[index]
        )
        hidden = _round_array(features @ tensors[0].T, 128) + tensors[1]
        hidden = _activate(hidden, int(model["activation"]))  # type: ignore[index]
        for layer in range(layer_count):
            weight = tensors[2 + 2 * layer]
            bias = tensors[3 + 2 * layer]
            hidden = _round_array(hidden @ weight.T, 128) + bias
            hidden += _round_array(hidden * gamma, 4096) + beta
            hidden = _activate(hidden, int(model["activation"]))  # type: ignore[index]
        residual = _round_array(hidden @ output_weight.T, 128) + output_bias
        output[start:stop] = _round_array(residual, 4096).astype(np.int32)
    return output.reshape(H, W, C)


def render_scorer_y1(root: dict[str, object], pair_id: int) -> np.ndarray:
    if type(pair_id) is not int or not 0 <= pair_id < PAIR_COUNT:
        raise SemanticVariantError("pair_id is outside exact n600")
    if root["pair_invariant"] and root["cached_frame"] is not None:
        return root["cached_frame"]  # type: ignore[return-value]
    labels = _labels(root, pair_id)
    profile = root["profile"]
    rgb = np.asarray(profile["role_rgb"], dtype=np.int32)[labels].copy()  # type: ignore[index]
    gauge = (
        root["gauges"][pair_id]  # type: ignore[index]
        if root["ownership"] == 1
        else (pair_id, 0, 0, 0, 0, 0, 0, 256)
    )
    chroma_gain = int(profile["chroma_gain"])  # type: ignore[index]
    rgb[:, :, 0] += gauge[4] + _round_scalar(gauge[6] * chroma_gain, 16)
    rgb[:, :, 1] += gauge[4] - _round_scalar((gauge[5] + gauge[6]) * chroma_gain, 32)
    rgb[:, :, 2] += gauge[4] + _round_scalar(gauge[5] * chroma_gain, 16)
    edge_map: np.ndarray | None = None
    for atom in root["basis"]:  # type: ignore[union-attr]
        phase = (
            atom[3] * _X64 * 256 // W
            + atom[4] * _Y64 * 256 // H
            + atom[5]
            + gauge[1]
            + _round_scalar(
                (atom[3] * gauge[2] + atom[4] * gauge[3]) * int(profile["parallax_gain"]),  # type: ignore[index]
                16,
            )
        ) & 0xFF
        if atom[2] == 0:
            wave = 127 - 2 * np.abs(phase - 128)
        elif atom[2] == 1:
            wave = np.where(phase < 128, -127, 127)
        else:
            wave = 127 - np.minimum(np.abs(phase - 64), np.abs(phase - 192)) * 4
        allowed = ((atom[1] >> labels) & 1).astype(bool)
        if atom[9]:
            if edge_map is None:
                edge_map = _edges(labels)
            allowed &= _dilate(edge_map, atom[9])
            profile_gain = int(profile["edge_gain"])  # type: ignore[index]
        else:
            profile_gain = int(profile["texture_gain"])  # type: ignore[index]
        for channel, amplitude in enumerate(atom[6:9]):
            delta = _round_array(wave * amplitude * gauge[7] * profile_gain, 127 * 256 * 16)
            rgb[:, :, channel] += np.where(allowed, delta, 0).astype(np.int32)
    rgb += _generator(root, pair_id, labels)
    for atom in root["quotient"]:  # type: ignore[union-attr]
        if not atom[1] <= pair_id < atom[2]:
            continue
        dt = pair_id - atom[1]
        center_x = atom[5] + _round_scalar(atom[7] * dt, 16)
        center_y = atom[6] + _round_scalar(atom[8] * dt, 16)
        wx = np.maximum(0, atom[9] - np.abs(_X64 * 16 - center_x))
        wy = np.maximum(0, atom[10] - np.abs(_Y64 * 16 - center_y))
        weight_q8 = wx * wy * 256 // (atom[9] * atom[10])
        allowed = ((atom[3] >> labels) & 1).astype(bool)
        if atom[4]:
            if edge_map is None:
                edge_map = _edges(labels)
            allowed &= edge_map
        for channel, amplitude in enumerate(atom[11:14]):
            delta = _round_array(weight_q8 * amplitude, 256)
            rgb[:, :, channel] += np.where(allowed, delta, 0).astype(np.int32)
    frame = np.clip(rgb, 0, 255).astype(np.uint8)
    if root["pair_invariant"]:
        frame.setflags(write=False)
        root["cached_frame"] = frame
    return frame
