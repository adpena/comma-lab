# SPDX-License-Identifier: MIT
"""Deterministic E1 runtime for the trained TR1 lotto partition renderer.

This module closes the stale TB1 checkpoint to one counted, receiver-consumed
four-section packet:

``tokens``
    Quantized codes for ``clip(EMA.tokens_base + EMA.tokens_delta[p], -1, 1)``.
    The old accounting estimate compressed base and delta separately; this
    runtime intentionally compiles the *combined* field used by the forward.
``lotto_renderer``
    Every effective learned renderer value: binary masks plus deployed fp16
    gains and biases.  The fixed signed bank is regenerated from the counted
    selector seed and is therefore generic/free receiver code.
``selector``
    The counted decode program and TB1 ledger choices, including geometry,
    widths, quantizer, bank seed, mask density, and shared-base temporal mode.
``pose_stub``
    A canonical, consumed declaration that pose is inert and unclaimed.  It
    contains no pose values and cannot be interpreted as a pose result.

The directory stores section offsets, lengths, and SHA-256 digests in a fixed
big-endian binary structure.  Parsing is strict: order, contiguity, hashes,
canonical JSON, Brotli-Q11 closure, padding bits, and the final cursor must all
close exactly.  This is the TR1 application of the counted-consumption rule
Catalog #417; Catalog #402 is telemetry liveness, not this invariant.

The renderer is a NumPy/SciPy fp32 reference.  It needs neither MLX nor Torch:
those backends are used only by the bounded rehearsal parity checker.
Evidence produced by this module is research-only and is never a score.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import struct
import zipfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import numpy as np

try:
    import brotli
except ImportError:  # pragma: no cover - declared E1 runtime dependency
    brotli = None  # type: ignore[assignment]

PACKET_SCHEMA: Final = "ddm_tr1_four_section_packet.v1"
ARCHIVE_SCHEMA: Final = "ddm_tr1_runtime_archive.v1"
SELECTOR_SCHEMA: Final = "ddm_tr1_selector.v1"
POSE_STUB_SCHEMA: Final = "ddm_tr1_inert_pose_stub.v1"

PACKET_MAGIC: Final = b"DDMTR1P1"
PACKET_VERSION: Final = 1
PACKET_HEADER: Final = struct.Struct(">8sHHI")
SECTION_ENTRY: Final = struct.Struct(">16sQQ32s")
COMPRESSED_FRAME_HEADER: Final = struct.Struct(">8sQQ32s")
TOKENS_RAW_HEADER: Final = struct.Struct(">HHHH")
RENDERER_RAW_HEADER: Final = struct.Struct(">II")
TOKENS_FRAME_MAGIC: Final = b"TR1TOK1!"
RENDERER_FRAME_MAGIC: Final = b"TR1REN1!"

SEG_H: Final = 384
SEG_W: Final = 512
CAMERA_H: Final = 874
CAMERA_W: Final = 1164

SECTION_CONTRACT: Final = (
    ("tokens", "TR1Receiver.decode_combined_tokens"),
    ("lotto_renderer", "TR1Receiver.regenerate_bank_and_apply_mask_mods"),
    ("selector", "TR1Receiver.select_decode_program"),
    ("pose_stub", "TR1Receiver.consume_inert_pose_declaration"),
)
SECTION_NAMES: Final = tuple(name for name, _consumer in SECTION_CONTRACT)
SECTION_CONSUMERS: Final = MappingProxyType(dict(SECTION_CONTRACT))
ARCHIVE_MEMBERS: Final = ("manifest.json", "state/tr1.ddt1")

_PACKET_METADATA_KEYS: Final = {
    "pose_claim",
    "pointer_moved",
    "research_only",
    "schema",
    "score_claim",
    "section_consumers",
    "section_order",
}
_SELECTOR_BASE_KEYS: Final = {
    "activation",
    "arch",
    "bank_algorithm",
    "code_width",
    "frame_role",
    "grid_downsample",
    "grid_h",
    "grid_w",
    "lotto_seed",
    "mask_density",
    "num_pairs",
    "output_activation",
    "output_height",
    "output_width",
    "renderer_width",
    "schema",
    "token_encoding",
    "token_quant_levels",
    "token_ste",
    "token_temporal_mode",
}
_POSE_STUB_VALUE: Final = {
    "inert": True,
    "pose_claim": False,
    "schema": POSE_STUB_SCHEMA,
    "values": [],
}
_ARCHIVE_MANIFEST_KEYS: Final = {
    "packet_bytes",
    "packet_member",
    "packet_sha256",
    "pointer_moved",
    "pose_claim",
    "research_only",
    "schema",
    "score_claim",
}


class TR1RuntimeError(ValueError):
    """The packet, archive, checkpoint, or receiver failed closed."""


@dataclass(frozen=True, slots=True)
class CheckpointCustody:
    path: Path
    bytes: int
    sha256: str
    config_hash: str


@dataclass(frozen=True, slots=True)
class CompiledTR1Archive:
    archive_bytes: bytes
    archive_sha256: str
    packet_bytes: bytes
    packet_sha256: str
    checkpoint: CheckpointCustody


@dataclass(frozen=True, slots=True)
class ParsedTR1Packet:
    metadata: Mapping[str, Any]
    section_payloads: tuple[bytes, ...]
    selector: Mapping[str, Any]
    token_codes: np.ndarray
    masks: tuple[np.ndarray, ...]
    gains: tuple[np.ndarray, ...]
    biases: tuple[np.ndarray, ...]
    pose_stub_consumed: bool

    def section(self, name: str) -> bytes:
        try:
            index = SECTION_NAMES.index(name)
        except ValueError as exc:  # pragma: no cover - caller misuse
            raise KeyError(name) from exc
        return self.section_payloads[index]


@dataclass(frozen=True, slots=True)
class ParsedTR1Archive:
    manifest: Mapping[str, Any]
    packet_bytes: bytes
    packet: ParsedTR1Packet


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _file_custody(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(8 * 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            total += len(chunk)
    return total, digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _duplicate_refusing_json(
    payload: bytes,
    *,
    label: str,
    require_canonical: bool,
) -> Any:
    def pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
            if key in result:
                raise TR1RuntimeError(f"{label} has duplicate key {key!r}")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload.decode("utf-8", "strict"),
            object_pairs_hook=pairs,
            parse_constant=lambda token: (_ for _ in ()).throw(
                TR1RuntimeError(f"{label} has non-finite number {token!r}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TR1RuntimeError(f"{label} is malformed JSON") from exc
    if require_canonical and _canonical_json(value) != payload:
        raise TR1RuntimeError(f"{label} is not canonical JSON")
    return value


def _exact_int(value: Any, *, label: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TR1RuntimeError(f"{label} must be an integer >= {minimum}")
    return value


def _exact_float(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TR1RuntimeError(f"{label} must be finite numeric")
    result = float(value)
    if not math.isfinite(result):
        raise TR1RuntimeError(f"{label} must be finite numeric")
    return result


def _config_hash(config: Mapping[str, Any]) -> str:
    return _sha256(_canonical_json(dict(config)))


def _conv_shapes(selector: Mapping[str, Any]) -> tuple[tuple[str, tuple[int, ...]], ...]:
    width = _exact_int(selector["renderer_width"], label="renderer_width", minimum=1)
    code_width = _exact_int(selector["code_width"], label="code_width", minimum=1)
    downsample = _exact_int(
        selector["grid_downsample"],
        label="grid_downsample",
        minimum=1,
    )
    n_upsample = round(math.log2(downsample))
    if 2**n_upsample != downsample:
        raise TR1RuntimeError("grid_downsample must be a power of two")
    rows: list[tuple[str, tuple[int, ...]]] = [("conv0", (width, 3, 3, code_width))]
    rows.extend((f"up{index}", (width, 3, 3, width)) for index in range(n_upsample))
    rows.append(("head", (3, 3, 3, width)))
    return tuple(rows)


def _validate_selector(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise TR1RuntimeError("selector must be an object")
    expected_keys = set(_SELECTOR_BASE_KEYS)
    token_ste = value.get("token_ste")
    if token_ste == "dither":
        expected_keys.add("dither_seed")
    if set(value) != expected_keys:
        raise TR1RuntimeError(f"selector keys differ: {sorted(value)!r} != {sorted(expected_keys)!r}")
    fixed = {
        "activation": "gelu_exact_erf",
        "arch": "tr1_lotto_combined_ema_v1",
        "bank_algorithm": "numpy_default_rng_choice_signed_he_v1",
        "frame_role": "frame1_only",
        "output_activation": "sigmoid_times_255",
        "output_height": SEG_H,
        "output_width": SEG_W,
        "schema": SELECTOR_SCHEMA,
        "token_encoding": "combined_ema_base_plus_delta_uint8_codes_v1",
    }
    for key, expected in fixed.items():
        if value.get(key) != expected:
            raise TR1RuntimeError(f"selector {key} must equal {expected!r}")
    num_pairs = _exact_int(value.get("num_pairs"), label="num_pairs", minimum=1)
    if num_pairs > 65535:
        raise TR1RuntimeError("num_pairs exceeds uint16 token framing")
    grid_h = _exact_int(value.get("grid_h"), label="grid_h", minimum=1)
    grid_w = _exact_int(value.get("grid_w"), label="grid_w", minimum=1)
    downsample = _exact_int(
        value.get("grid_downsample"),
        label="grid_downsample",
        minimum=1,
    )
    if (grid_h * downsample, grid_w * downsample) != (SEG_H, SEG_W):
        raise TR1RuntimeError("selector grid does not close to 384x512")
    _exact_int(value.get("code_width"), label="code_width", minimum=1)
    _exact_int(value.get("renderer_width"), label="renderer_width", minimum=1)
    levels = _exact_int(
        value.get("token_quant_levels"),
        label="token_quant_levels",
        minimum=2,
    )
    if levels > 256:
        raise TR1RuntimeError("uint8 token codes support at most 256 levels")
    if token_ste not in {"round", "dither"}:
        raise TR1RuntimeError("token_ste must be round or dither")
    if token_ste == "dither":
        _exact_int(value.get("dither_seed"), label="dither_seed")
    _exact_int(value.get("lotto_seed"), label="lotto_seed")
    density = _exact_float(value.get("mask_density"), label="mask_density")
    if not 0.0 < density < 1.0:
        raise TR1RuntimeError("mask_density must lie strictly between zero and one")
    if value.get("token_temporal_mode") != "shared_base":
        raise TR1RuntimeError("token_temporal_mode must be shared_base")
    _conv_shapes(value)
    return MappingProxyType(dict(value))


def _validate_packet_metadata(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != _PACKET_METADATA_KEYS:
        raise TR1RuntimeError("packet metadata keys differ from the sealed schema")
    if value.get("schema") != PACKET_SCHEMA:
        raise TR1RuntimeError("packet metadata schema differs")
    if value.get("section_order") != list(SECTION_NAMES):
        raise TR1RuntimeError("packet section order differs")
    if value.get("section_consumers") != dict(SECTION_CONTRACT):
        raise TR1RuntimeError("packet section consumers differ")
    for key, expected in (
        ("research_only", True),
        ("score_claim", False),
        ("pointer_moved", False),
        ("pose_claim", False),
    ):
        if value.get(key) is not expected and value.get(key) != expected:
            raise TR1RuntimeError(f"packet metadata {key} differs")
    return MappingProxyType(dict(value))


def _encode_name(name: str) -> bytes:
    raw = name.encode("ascii")
    if not raw or len(raw) > 16 or b"\0" in raw:
        raise TR1RuntimeError(f"invalid section name {name!r}")
    return raw.ljust(16, b"\0")


def _decode_name(raw: bytes) -> str:
    if len(raw) != 16:
        raise TR1RuntimeError("section name width differs")
    body, separator, padding = raw.partition(b"\0")
    if separator and any(padding):
        raise TR1RuntimeError("section name has nonzero padding")
    try:
        name = body.decode("ascii", "strict")
    except UnicodeDecodeError as exc:
        raise TR1RuntimeError("section name is not ASCII") from exc
    if _encode_name(name) != raw:
        raise TR1RuntimeError("section name is not canonical")
    return name


def _encode_brotli_frame(magic: bytes, raw: bytes) -> bytes:
    if len(magic) != 8:
        raise TR1RuntimeError("compressed frame magic must be eight bytes")
    if brotli is None:
        raise TR1RuntimeError("Brotli is required for TR1 E1 packet compilation")
    coded = brotli.compress(raw, quality=11)
    return (
        COMPRESSED_FRAME_HEADER.pack(
            magic,
            len(raw),
            len(coded),
            hashlib.sha256(raw).digest(),
        )
        + coded
    )


def _decode_brotli_frame(payload: bytes, *, magic: bytes, label: str) -> bytes:
    if len(payload) < COMPRESSED_FRAME_HEADER.size:
        raise TR1RuntimeError(f"{label} frame is truncated")
    observed_magic, raw_length, coded_length, raw_sha = COMPRESSED_FRAME_HEADER.unpack_from(payload)
    if observed_magic != magic:
        raise TR1RuntimeError(f"{label} frame magic differs")
    if len(payload) != COMPRESSED_FRAME_HEADER.size + coded_length:
        raise TR1RuntimeError(f"{label} coded length does not close")
    if brotli is None:
        raise TR1RuntimeError("Brotli is required for TR1 E1 packet decoding")
    try:
        raw = brotli.decompress(payload[COMPRESSED_FRAME_HEADER.size :])
    except brotli.error as exc:
        raise TR1RuntimeError(f"{label} Brotli stream is invalid") from exc
    if len(raw) != raw_length:
        raise TR1RuntimeError(f"{label} raw length differs")
    if hashlib.sha256(raw).digest() != raw_sha:
        raise TR1RuntimeError(f"{label} raw SHA-256 differs")
    return raw


def _token_codes(
    tokens_base: np.ndarray,
    tokens_delta: np.ndarray,
    *,
    levels: int,
    token_ste: str,
    dither_seed: int | None,
) -> np.ndarray:
    combined = np.add(
        np.asarray(tokens_base, dtype=np.float32)[None],
        np.asarray(tokens_delta, dtype=np.float32),
        dtype=np.float32,
    )
    combined = np.clip(combined, np.float32(-1), np.float32(1))
    x01 = (combined + np.float32(1)) * np.float32(0.5)
    scaled = x01 * np.float32(levels - 1)
    if token_ste == "dither":
        if dither_seed is None:
            raise TR1RuntimeError("dither token encoding lacks a seed")
        shape = tuple(int(value) for value in tokens_base.shape)
        dither = (np.random.default_rng(dither_seed).random(shape) - 0.5).astype(np.float32)
        scaled = scaled + dither[None]
    codes_i64 = np.rint(scaled).astype(np.int64)
    if np.any(codes_i64 < 0) or np.any(codes_i64 >= levels):
        raise TR1RuntimeError("quantized token code escaped the declared lattice")
    return np.ascontiguousarray(codes_i64.astype(np.uint8))


def _encode_tokens(codes: np.ndarray) -> bytes:
    if codes.ndim != 4:
        raise TR1RuntimeError("token codes must have shape [P,H,W,C]")
    shape = tuple(int(value) for value in codes.shape)
    if any(value > 65535 for value in shape):
        raise TR1RuntimeError("token shape exceeds uint16 framing")
    raw = (
        TOKENS_RAW_HEADER.pack(*shape)
        + np.ascontiguousarray(
            codes,
            dtype=np.uint8,
        ).tobytes()
    )
    return _encode_brotli_frame(TOKENS_FRAME_MAGIC, raw)


def _decode_tokens(payload: bytes, selector: Mapping[str, Any]) -> np.ndarray:
    raw = _decode_brotli_frame(
        payload,
        magic=TOKENS_FRAME_MAGIC,
        label="tokens",
    )
    if len(raw) < TOKENS_RAW_HEADER.size:
        raise TR1RuntimeError("tokens raw frame is truncated")
    shape = TOKENS_RAW_HEADER.unpack_from(raw)
    expected_shape = (
        selector["num_pairs"],
        selector["grid_h"],
        selector["grid_w"],
        selector["code_width"],
    )
    if shape != expected_shape:
        raise TR1RuntimeError(f"token shape differs: {shape!r} != {expected_shape!r}")
    expected_bytes = math.prod(shape)
    if len(raw) != TOKENS_RAW_HEADER.size + expected_bytes:
        raise TR1RuntimeError("token raw bytes do not close exactly")
    codes = np.frombuffer(raw, dtype=np.uint8, offset=TOKENS_RAW_HEADER.size)
    codes = np.ascontiguousarray(codes.reshape(shape))
    if np.any(codes >= selector["token_quant_levels"]):
        raise TR1RuntimeError("token code exceeds selector quantization levels")
    return codes


def _encode_renderer(
    masks: Sequence[np.ndarray],
    gains: Sequence[np.ndarray],
    biases: Sequence[np.ndarray],
) -> bytes:
    mask_flat = np.concatenate([np.asarray(mask, dtype=np.uint8).reshape(-1) for mask in masks])
    if np.any((mask_flat != 0) & (mask_flat != 1)):
        raise TR1RuntimeError("renderer mask is not binary")
    mask_packed = np.packbits(mask_flat, bitorder="big")
    float_values = np.concatenate(
        [
            item
            for gain, bias in zip(gains, biases, strict=True)
            for item in (
                np.asarray(gain, dtype=np.float32).reshape(-1),
                np.asarray(bias, dtype=np.float32).reshape(-1),
            )
        ]
    )
    raw = (
        RENDERER_RAW_HEADER.pack(mask_flat.size, float_values.size)
        + mask_packed.tobytes()
        + float_values.astype(">f2").tobytes()
    )
    return _encode_brotli_frame(RENDERER_FRAME_MAGIC, raw)


def _decode_renderer(
    payload: bytes,
    selector: Mapping[str, Any],
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    raw = _decode_brotli_frame(
        payload,
        magic=RENDERER_FRAME_MAGIC,
        label="lotto_renderer",
    )
    if len(raw) < RENDERER_RAW_HEADER.size:
        raise TR1RuntimeError("renderer raw frame is truncated")
    mask_count, float_count = RENDERER_RAW_HEADER.unpack_from(raw)
    shapes = _conv_shapes(selector)
    expected_mask_count = sum(math.prod(shape) for _name, shape in shapes)
    expected_float_count = sum(2 * shape[0] for _name, shape in shapes)
    if (mask_count, float_count) != (
        expected_mask_count,
        expected_float_count,
    ):
        raise TR1RuntimeError("renderer raw counts differ from selector geometry")
    mask_bytes = (mask_count + 7) // 8
    expected_length = RENDERER_RAW_HEADER.size + mask_bytes + 2 * float_count
    if len(raw) != expected_length:
        raise TR1RuntimeError("renderer raw bytes do not close exactly")
    packed = np.frombuffer(
        raw,
        dtype=np.uint8,
        count=mask_bytes,
        offset=RENDERER_RAW_HEADER.size,
    )
    all_bits = np.unpackbits(packed, bitorder="big")
    if np.any(all_bits[mask_count:] != 0):
        raise TR1RuntimeError("renderer mask padding contains counted inert bits")
    mask_values = all_bits[:mask_count]
    floats_offset = RENDERER_RAW_HEADER.size + mask_bytes
    float_values = np.frombuffer(
        raw,
        dtype=">f2",
        count=float_count,
        offset=floats_offset,
    ).astype(np.float32)

    masks: list[np.ndarray] = []
    gains: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    mask_cursor = 0
    float_cursor = 0
    for _name, shape in shapes:
        count = math.prod(shape)
        masks.append(np.ascontiguousarray(mask_values[mask_cursor : mask_cursor + count].reshape(shape).astype(bool)))
        mask_cursor += count
        out_channels = shape[0]
        gains.append(np.ascontiguousarray(float_values[float_cursor : float_cursor + out_channels]))
        float_cursor += out_channels
        biases.append(np.ascontiguousarray(float_values[float_cursor : float_cursor + out_channels]))
        float_cursor += out_channels
    if mask_cursor != mask_count or float_cursor != float_count:
        raise TR1RuntimeError("renderer parser cursor did not close")
    return tuple(masks), tuple(gains), tuple(biases)


def _pose_stub() -> bytes:
    return _canonical_json(_POSE_STUB_VALUE)


def _consume_pose_stub(payload: bytes) -> bool:
    value = _duplicate_refusing_json(
        payload,
        label="pose_stub",
        require_canonical=True,
    )
    if value != _POSE_STUB_VALUE:
        raise TR1RuntimeError("pose stub differs from the inert no-claim schema")
    return True


def build_packet(
    metadata: Mapping[str, Any],
    section_payloads: Mapping[str, bytes],
) -> bytes:
    checked_metadata = _validate_packet_metadata(dict(metadata))
    if tuple(section_payloads) != SECTION_NAMES:
        raise TR1RuntimeError("section payload mapping order differs")
    metadata_bytes = _canonical_json(dict(checked_metadata))
    payload_start = PACKET_HEADER.size + len(metadata_bytes) + len(SECTION_NAMES) * SECTION_ENTRY.size
    directory: list[bytes] = []
    payload_rows: list[bytes] = []
    cursor = payload_start
    for name in SECTION_NAMES:
        payload = bytes(section_payloads[name])
        if not payload:
            raise TR1RuntimeError(f"section {name} must not be empty")
        directory.append(
            SECTION_ENTRY.pack(
                _encode_name(name),
                cursor,
                len(payload),
                hashlib.sha256(payload).digest(),
            )
        )
        payload_rows.append(payload)
        cursor += len(payload)
    return (
        PACKET_HEADER.pack(
            PACKET_MAGIC,
            PACKET_VERSION,
            len(SECTION_NAMES),
            len(metadata_bytes),
        )
        + metadata_bytes
        + b"".join(directory)
        + b"".join(payload_rows)
    )


def parse_packet(packet: bytes) -> ParsedTR1Packet:
    if len(packet) < PACKET_HEADER.size:
        raise TR1RuntimeError("TR1 packet is truncated")
    magic, version, section_count, metadata_length = PACKET_HEADER.unpack_from(packet)
    if magic != PACKET_MAGIC or version != PACKET_VERSION:
        raise TR1RuntimeError("TR1 packet magic/version differs")
    if section_count != len(SECTION_NAMES):
        raise TR1RuntimeError("TR1 packet section count differs")
    metadata_start = PACKET_HEADER.size
    metadata_end = metadata_start + metadata_length
    directory_end = metadata_end + section_count * SECTION_ENTRY.size
    if metadata_end < metadata_start or directory_end > len(packet):
        raise TR1RuntimeError("TR1 packet metadata/directory is truncated")
    metadata_value = _duplicate_refusing_json(
        packet[metadata_start:metadata_end],
        label="packet metadata",
        require_canonical=True,
    )
    metadata = _validate_packet_metadata(metadata_value)

    payloads: list[bytes] = []
    cursor = directory_end
    for index, expected_name in enumerate(SECTION_NAMES):
        entry_offset = metadata_end + index * SECTION_ENTRY.size
        raw_name, section_offset, section_length, section_sha = SECTION_ENTRY.unpack_from(packet, entry_offset)
        observed_name = _decode_name(raw_name)
        if observed_name != expected_name:
            raise TR1RuntimeError(f"section order differs at {index}: {observed_name!r}")
        if section_offset != cursor:
            raise TR1RuntimeError(f"section {expected_name} offset is not contiguous")
        section_end = section_offset + section_length
        if section_end < section_offset or section_end > len(packet):
            raise TR1RuntimeError(f"section {expected_name} length escaped packet")
        payload = packet[section_offset:section_end]
        if hashlib.sha256(payload).digest() != section_sha:
            raise TR1RuntimeError(f"section {expected_name} SHA-256 differs")
        payloads.append(payload)
        cursor = section_end
    if cursor != len(packet):
        raise TR1RuntimeError("TR1 packet final cursor differs; trailing or unconsumed bytes exist")

    selector_value = _duplicate_refusing_json(
        payloads[SECTION_NAMES.index("selector")],
        label="selector",
        require_canonical=True,
    )
    selector = _validate_selector(selector_value)
    token_codes = _decode_tokens(
        payloads[SECTION_NAMES.index("tokens")],
        selector,
    )
    masks, gains, biases = _decode_renderer(
        payloads[SECTION_NAMES.index("lotto_renderer")],
        selector,
    )
    pose_consumed = _consume_pose_stub(payloads[SECTION_NAMES.index("pose_stub")])
    return ParsedTR1Packet(
        metadata=metadata,
        section_payloads=tuple(payloads),
        selector=selector,
        token_codes=token_codes,
        masks=masks,
        gains=gains,
        biases=biases,
        pose_stub_consumed=pose_consumed,
    )


def reemit_packet(parsed: ParsedTR1Packet) -> bytes:
    return build_packet(
        parsed.metadata,
        {name: parsed.section_payloads[index] for index, name in enumerate(SECTION_NAMES)},
    )


def _deterministic_stored_zip(members: Mapping[str, bytes]) -> bytes:
    if tuple(members) != ARCHIVE_MEMBERS:
        raise TR1RuntimeError("archive member order differs")
    stream = io.BytesIO()
    with zipfile.ZipFile(
        stream,
        mode="w",
        compression=zipfile.ZIP_STORED,
        allowZip64=False,
    ) as archive:
        for name in ARCHIVE_MEMBERS:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            info.flag_bits = 0
            archive.writestr(info, members[name])
    return stream.getvalue()


def _archive_manifest(packet: bytes) -> dict[str, Any]:
    return {
        "packet_bytes": len(packet),
        "packet_member": "state/tr1.ddt1",
        "packet_sha256": _sha256(packet),
        "pointer_moved": False,
        "pose_claim": False,
        "research_only": True,
        "schema": ARCHIVE_SCHEMA,
        "score_claim": False,
    }


def _validate_archive_manifest(value: Any) -> Mapping[str, Any]:
    if not isinstance(value, dict) or set(value) != _ARCHIVE_MANIFEST_KEYS:
        raise TR1RuntimeError("archive manifest keys differ")
    if value.get("schema") != ARCHIVE_SCHEMA:
        raise TR1RuntimeError("archive manifest schema differs")
    for key, expected in (
        ("packet_member", "state/tr1.ddt1"),
        ("research_only", True),
        ("score_claim", False),
        ("pointer_moved", False),
        ("pose_claim", False),
    ):
        if value.get(key) is not expected and value.get(key) != expected:
            raise TR1RuntimeError(f"archive manifest {key} differs")
    _exact_int(value.get("packet_bytes"), label="packet_bytes", minimum=1)
    if not _is_sha256(value.get("packet_sha256")):
        raise TR1RuntimeError("archive manifest packet_sha256 is invalid")
    return MappingProxyType(dict(value))


def parse_archive(archive_bytes: bytes) -> ParsedTR1Archive:
    try:
        with zipfile.ZipFile(io.BytesIO(archive_bytes), mode="r") as archive:
            infos = archive.infolist()
            if tuple(info.filename for info in infos) != ARCHIVE_MEMBERS:
                raise TR1RuntimeError("TR1 archive members/order differ")
            if len({info.filename for info in infos}) != len(infos):
                raise TR1RuntimeError("TR1 archive has duplicate members")
            for info in infos:
                if info.compress_type != zipfile.ZIP_STORED or info.flag_bits & 0x1 or info.is_dir():
                    raise TR1RuntimeError(f"TR1 archive member {info.filename} is not plain stored data")
            manifest_payload = archive.read("manifest.json")
            packet_bytes = archive.read("state/tr1.ddt1")
    except (zipfile.BadZipFile, RuntimeError, KeyError) as exc:
        if isinstance(exc, TR1RuntimeError):
            raise
        raise TR1RuntimeError("TR1 archive is malformed") from exc
    manifest_value = _duplicate_refusing_json(
        manifest_payload,
        label="archive manifest",
        require_canonical=True,
    )
    manifest = _validate_archive_manifest(manifest_value)
    if len(packet_bytes) != manifest["packet_bytes"]:
        raise TR1RuntimeError("archive packet length differs")
    if _sha256(packet_bytes) != manifest["packet_sha256"]:
        raise TR1RuntimeError("archive packet SHA-256 differs")
    packet = parse_packet(packet_bytes)
    return ParsedTR1Archive(
        manifest=manifest,
        packet_bytes=packet_bytes,
        packet=packet,
    )


def reemit_archive(parsed: ParsedTR1Archive) -> bytes:
    packet_bytes = reemit_packet(parsed.packet)
    if packet_bytes != parsed.packet_bytes:
        raise TR1RuntimeError("packet parse/re-emit changed bytes")
    return _deterministic_stored_zip(
        {
            "manifest.json": _canonical_json(dict(parsed.manifest)),
            "state/tr1.ddt1": packet_bytes,
        }
    )


def parse_extracted_archive(archive_directory: Path) -> ParsedTR1Archive:
    observed = tuple(
        sorted(
            path.relative_to(archive_directory).as_posix() for path in archive_directory.rglob("*") if path.is_file()
        )
    )
    if observed != tuple(sorted(ARCHIVE_MEMBERS)):
        raise TR1RuntimeError(f"extracted TR1 members differ: {observed!r}")
    manifest_path = archive_directory / "manifest.json"
    packet_path = archive_directory / "state/tr1.ddt1"
    if (
        manifest_path.is_symlink()
        or packet_path.is_symlink()
        or not manifest_path.is_file()
        or not packet_path.is_file()
    ):
        raise TR1RuntimeError("extracted TR1 members must be regular files")
    manifest_payload = manifest_path.read_bytes()
    packet_bytes = packet_path.read_bytes()
    manifest_value = _duplicate_refusing_json(
        manifest_payload,
        label="archive manifest",
        require_canonical=True,
    )
    manifest = _validate_archive_manifest(manifest_value)
    if len(packet_bytes) != manifest["packet_bytes"] or _sha256(packet_bytes) != manifest["packet_sha256"]:
        raise TR1RuntimeError("extracted packet custody differs")
    packet = parse_packet(packet_bytes)
    return ParsedTR1Archive(
        manifest=manifest,
        packet_bytes=packet_bytes,
        packet=packet,
    )


def _checkpoint_arrays(
    checkpoint_path: Path,
) -> tuple[CheckpointCustody, dict[str, Any], dict[str, np.ndarray]]:
    if checkpoint_path.is_symlink() or not checkpoint_path.is_file():
        raise TR1RuntimeError("checkpoint must be one regular file")
    source_bytes, source_sha = _file_custody(checkpoint_path)
    try:
        with np.load(checkpoint_path, allow_pickle=False) as stored:
            if "meta::json" not in stored.files:
                raise TR1RuntimeError("checkpoint lacks meta::json")
            meta = _duplicate_refusing_json(
                bytes(stored["meta::json"]),
                label="checkpoint metadata",
                require_canonical=False,
            )
            if not isinstance(meta, dict) or not isinstance(meta.get("cfg"), dict):
                raise TR1RuntimeError("checkpoint metadata lacks cfg")
            config = dict(meta["cfg"])
            config_hash = _config_hash(config)
            if meta.get("config_hash") != config_hash:
                raise TR1RuntimeError("checkpoint config hash differs")
            required_names = ["tokens_base", "tokens_delta"]
            selector_seed = config.get("lotto_seed")
            if config.get("variant") != "lotto" or selector_seed is None:
                raise TR1RuntimeError("checkpoint is not the lotto TR1 variant")
            temp_selector = _selector_from_config(config)
            required_names.extend(
                prefix + layer_name
                for layer_name, _shape in _conv_shapes(temp_selector)
                for prefix in ("s_", "g_", "b_")
            )
            arrays: dict[str, np.ndarray] = {}
            for name in required_names:
                key = f"ema::{name}"
                if key not in stored.files:
                    raise TR1RuntimeError(f"checkpoint lacks {key}")
                value = np.asarray(stored[key])
                if value.dtype != np.float32:
                    raise TR1RuntimeError(f"checkpoint {key} is not fp32")
                if not np.all(np.isfinite(value)):
                    raise TR1RuntimeError(f"checkpoint {key} is non-finite")
                arrays[name] = np.ascontiguousarray(value)
    except (OSError, ValueError) as exc:
        if isinstance(exc, TR1RuntimeError):
            raise
        raise TR1RuntimeError("checkpoint NPZ is malformed") from exc
    custody = CheckpointCustody(
        path=checkpoint_path.resolve(),
        bytes=source_bytes,
        sha256=source_sha,
        config_hash=config_hash,
    )
    return custody, config, arrays


def _selector_from_config(config: Mapping[str, Any]) -> Mapping[str, Any]:
    if config.get("variant") != "lotto":
        raise TR1RuntimeError("only the lotto TR1 variant is supported")
    if config.get("token_temporal_mode") != "shared_base":
        raise TR1RuntimeError("TR1 E1 requires shared_base checkpoint tokens")
    num_pairs = _exact_int(config.get("num_pairs"), label="num_pairs", minimum=1)
    downsample = _exact_int(
        config.get("grid_downsample"),
        label="grid_downsample",
        minimum=1,
    )
    if SEG_H % downsample or SEG_W % downsample:
        raise TR1RuntimeError("checkpoint grid_downsample does not divide output")
    token_ste = config.get("token_ste")
    selector: dict[str, Any] = {
        "activation": "gelu_exact_erf",
        "arch": "tr1_lotto_combined_ema_v1",
        "bank_algorithm": "numpy_default_rng_choice_signed_he_v1",
        "code_width": _exact_int(
            config.get("code_width"),
            label="code_width",
            minimum=1,
        ),
        "frame_role": "frame1_only",
        "grid_downsample": downsample,
        "grid_h": SEG_H // downsample,
        "grid_w": SEG_W // downsample,
        "lotto_seed": _exact_int(
            config.get("lotto_seed"),
            label="lotto_seed",
        ),
        "mask_density": _exact_float(
            config.get("lotto_mask_density_init"),
            label="lotto_mask_density_init",
        ),
        "num_pairs": num_pairs,
        "output_activation": "sigmoid_times_255",
        "output_height": SEG_H,
        "output_width": SEG_W,
        "renderer_width": _exact_int(
            config.get("renderer_width"),
            label="renderer_width",
            minimum=1,
        ),
        "schema": SELECTOR_SCHEMA,
        "token_encoding": "combined_ema_base_plus_delta_uint8_codes_v1",
        "token_quant_levels": _exact_int(
            config.get("token_quant_levels"),
            label="token_quant_levels",
            minimum=2,
        ),
        "token_ste": token_ste,
        "token_temporal_mode": config.get("token_temporal_mode"),
    }
    if token_ste == "dither":
        selector["dither_seed"] = (
            _exact_int(
                config.get("seed"),
                label="seed",
            )
            + 7
        )
    return _validate_selector(selector)


def compile_archive_from_checkpoint(
    checkpoint_path: Path | str,
) -> CompiledTR1Archive:
    path = Path(checkpoint_path)
    custody, config, arrays = _checkpoint_arrays(path)
    selector = _selector_from_config(config)
    base = arrays["tokens_base"]
    delta = arrays["tokens_delta"]
    expected_base_shape = (
        selector["grid_h"],
        selector["grid_w"],
        selector["code_width"],
    )
    expected_delta_shape = (selector["num_pairs"], *expected_base_shape)
    if base.shape != expected_base_shape or delta.shape != expected_delta_shape:
        raise TR1RuntimeError("checkpoint token arrays differ from selector geometry")
    codes = _token_codes(
        base,
        delta,
        levels=selector["token_quant_levels"],
        token_ste=selector["token_ste"],
        dither_seed=selector.get("dither_seed"),
    )
    masks: list[np.ndarray] = []
    gains: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    for name, shape in _conv_shapes(selector):
        score = arrays[f"s_{name}"]
        gain = arrays[f"g_{name}"]
        bias = arrays[f"b_{name}"]
        if score.shape != shape:
            raise TR1RuntimeError(f"checkpoint s_{name} shape differs")
        if gain.shape != (shape[0],) or bias.shape != (shape[0],):
            raise TR1RuntimeError(f"checkpoint {name} modulation shape differs")
        masks.append(score > np.float32(0))
        gains.append(gain)
        biases.append(bias)
    selector_payload = _canonical_json(dict(selector))
    section_payloads = {
        "tokens": _encode_tokens(codes),
        "lotto_renderer": _encode_renderer(masks, gains, biases),
        "selector": selector_payload,
        "pose_stub": _pose_stub(),
    }
    metadata = {
        "pose_claim": False,
        "pointer_moved": False,
        "research_only": True,
        "schema": PACKET_SCHEMA,
        "score_claim": False,
        "section_consumers": dict(SECTION_CONTRACT),
        "section_order": list(SECTION_NAMES),
    }
    packet = build_packet(metadata, section_payloads)
    parsed_packet = parse_packet(packet)
    if reemit_packet(parsed_packet) != packet:
        raise TR1RuntimeError("compiled packet failed exact parse/re-emit")
    manifest = _archive_manifest(packet)
    archive = _deterministic_stored_zip(
        {
            "manifest.json": _canonical_json(manifest),
            "state/tr1.ddt1": packet,
        }
    )
    parsed_archive = parse_archive(archive)
    if reemit_archive(parsed_archive) != archive:
        raise TR1RuntimeError("compiled archive failed exact parse/re-emit")
    return CompiledTR1Archive(
        archive_bytes=archive,
        archive_sha256=_sha256(archive),
        packet_bytes=packet,
        packet_sha256=_sha256(packet),
        checkpoint=custody,
    )


def decode_token_grid(
    parsed: ParsedTR1Packet,
    pair_index: int,
) -> np.ndarray:
    if isinstance(pair_index, bool) or not isinstance(pair_index, int):
        raise TR1RuntimeError("pair_index must be an integer")
    if pair_index < 0 or pair_index >= parsed.token_codes.shape[0]:
        raise TR1RuntimeError("pair_index is outside the packet")
    selector = parsed.selector
    codes = parsed.token_codes[pair_index].astype(np.float32)
    levels = np.float32(selector["token_quant_levels"] - 1)
    if selector["token_ste"] == "dither":
        shape = (
            selector["grid_h"],
            selector["grid_w"],
            selector["code_width"],
        )
        dither = (np.random.default_rng(selector["dither_seed"]).random(shape) - 0.5).astype(np.float32)
        x01 = (codes - dither) / levels
    else:
        x01 = codes / levels
    return np.ascontiguousarray(
        x01 * np.float32(2) - np.float32(1),
        dtype=np.float32,
    )


def _fixed_lotto_weights(
    parsed: ParsedTR1Packet,
) -> tuple[np.ndarray, ...]:
    rng = np.random.default_rng(parsed.selector["lotto_seed"])
    weights: list[np.ndarray] = []
    shapes = _conv_shapes(parsed.selector)
    for index, (_name, shape) in enumerate(shapes):
        fan_in = shape[1] * shape[2] * shape[3]
        magnitude = float(np.sqrt(2.0 / fan_in))
        fixed = (rng.choice([-1.0, 1.0], size=shape) * magnitude).astype(np.float32)
        weight = fixed * parsed.masks[index].astype(np.float32)
        weight = weight * parsed.gains[index].reshape((-1, 1, 1, 1))
        weights.append(np.ascontiguousarray(weight, dtype=np.float32))
    return tuple(weights)


def _conv2d_nhwc(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    if x.ndim != 4 or weight.ndim != 4 or weight.shape[1:3] != (3, 3):
        raise TR1RuntimeError("TR1 conv2d expects NHWC and O33I tensors")
    if x.shape[-1] != weight.shape[-1]:
        raise TR1RuntimeError("TR1 conv2d channel geometry differs")
    batch, height, width, _channels = x.shape
    out_channels = weight.shape[0]
    padded = np.pad(
        np.asarray(x, dtype=np.float32),
        ((0, 0), (1, 1), (1, 1), (0, 0)),
        mode="constant",
    )
    result = np.zeros(
        (batch, height, width, out_channels),
        dtype=np.float32,
    )
    # Apple Accelerate can raise spurious floating warnings from an otherwise
    # finite sgemm.  Suppress only the backend flags and then fail closed on the
    # actual result, rather than letting non-finite arithmetic pass.
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        for row in range(3):
            for column in range(3):
                kernel = weight[:, row, column, :].T
                result += np.matmul(
                    padded[:, row : row + height, column : column + width, :],
                    kernel,
                )
    if not np.all(np.isfinite(result)):
        raise TR1RuntimeError("TR1 conv2d produced non-finite output")
    return result


def _gelu_exact(x: np.ndarray) -> np.ndarray:
    try:
        from scipy.special import erf
    except ImportError as exc:  # pragma: no cover - locked env carries SciPy
        raise TR1RuntimeError("NumPy TR1 receiver requires scipy.special.erf") from exc
    root_two = np.float32(math.sqrt(2.0))
    return np.ascontiguousarray(
        x * (np.float32(1) + erf(x / root_two).astype(np.float32, copy=False)) * np.float32(0.5),
        dtype=np.float32,
    )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    result = np.empty_like(x, dtype=np.float32)
    positive = x >= 0
    result[positive] = np.float32(1) / (np.float32(1) + np.exp(-x[positive]).astype(np.float32, copy=False))
    exp_x = np.exp(x[~positive]).astype(np.float32, copy=False)
    result[~positive] = exp_x / (np.float32(1) + exp_x)
    return result


def render_frame1_float(
    parsed: ParsedTR1Packet,
    pair_index: int,
) -> np.ndarray:
    """Render one pre-R frame1 as float32 ``[384,512,3]`` in ``[0,255]``."""

    if not parsed.pose_stub_consumed:
        raise TR1RuntimeError("pose stub was not consumed")
    x = decode_token_grid(parsed, pair_index)[None]
    weights = _fixed_lotto_weights(parsed)
    shapes = _conv_shapes(parsed.selector)
    x = _conv2d_nhwc(x, weights[0]) + parsed.biases[0]
    x = _gelu_exact(x)
    for index, (name, _shape) in enumerate(shapes[1:-1], start=1):
        if not name.startswith("up"):
            raise TR1RuntimeError("TR1 renderer layer order differs")
        x = np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)
        x = _conv2d_nhwc(x, weights[index]) + parsed.biases[index]
        x = _gelu_exact(x)
    x = _conv2d_nhwc(x, weights[-1]) + parsed.biases[-1]
    result = _sigmoid(x) * np.float32(255)
    if result.shape != (1, SEG_H, SEG_W, 3):
        raise TR1RuntimeError("TR1 renderer output geometry differs")
    return np.ascontiguousarray(result[0], dtype=np.float32)


def _cubic_indices_weights(
    input_size: int,
    output_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    scale = np.float32(input_size / output_size)
    source = (np.arange(output_size, dtype=np.float32) + np.float32(0.5)) * scale - np.float32(0.5)
    base = np.floor(source).astype(np.int64)
    fraction = (source - base.astype(np.float32)).astype(np.float64)
    # PyTorch's bicubic kernel uses A=-0.75.  Calculate coefficients at
    # higher precision, then cast to fp32 for the same receiver arithmetic.
    a = np.float64(-0.75)

    def convolution1(value: np.ndarray) -> np.ndarray:
        return ((a + 2) * value - (a + 3)) * value * value + 1

    def convolution2(value: np.ndarray) -> np.ndarray:
        return ((a * value - 5 * a) * value + 8 * a) * value - 4 * a

    weights = np.stack(
        (
            convolution2(fraction + 1),
            convolution1(fraction),
            convolution1(1 - fraction),
            convolution2(2 - fraction),
        ),
        axis=1,
    ).astype(np.float32)
    indices = np.stack(
        (base - 1, base, base + 1, base + 2),
        axis=1,
    )
    return np.clip(indices, 0, input_size - 1), weights


def bicubic_up_to_camera_float(frame: np.ndarray) -> np.ndarray:
    """NumPy fp32 PyTorch-style bicubic, ``align_corners=False``, A=-0.75."""

    source = np.asarray(frame, dtype=np.float32)
    if source.shape != (SEG_H, SEG_W, 3):
        raise TR1RuntimeError("bicubic input must be [384,512,3]")
    y_indices, y_weights = _cubic_indices_weights(SEG_H, CAMERA_H)
    x_indices, x_weights = _cubic_indices_weights(SEG_W, CAMERA_W)

    horizontal = source[:, x_indices[:, 0], :] * x_weights[None, :, 0, None]
    for index in range(1, 4):
        horizontal = horizontal + (source[:, x_indices[:, index], :] * x_weights[None, :, index, None])
    result = horizontal[y_indices[:, 0], :, :] * y_weights[:, 0, None, None]
    for index in range(1, 4):
        result = result + (horizontal[y_indices[:, index], :, :] * y_weights[:, index, None, None])
    return np.ascontiguousarray(result, dtype=np.float32)


def render_frame1_camera_uint8(
    parsed: ParsedTR1Packet,
    pair_index: int,
) -> np.ndarray:
    """Render one deterministic camera-resolution uint8 frame1."""

    up = bicubic_up_to_camera_float(render_frame1_float(parsed, pair_index))
    return np.ascontiguousarray(np.clip(np.rint(up), 0, 255).astype(np.uint8))


def render_pair_camera_uint8(
    parsed: ParsedTR1Packet,
    pair_index: int,
) -> np.ndarray:
    """Return a research-only pair with a generic zero frame0.

    The zero frame is the executable meaning of the consumed inert pose stub.
    It makes no pose claim and is not evidence that pose is solved.
    """

    frame1 = render_frame1_camera_uint8(parsed, pair_index)
    frame0 = np.zeros_like(frame1, dtype=np.uint8)
    return np.stack((frame0, frame1), axis=0)


def section_ledger(parsed: ParsedTR1Packet) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(SECTION_NAMES):
        payload = parsed.section_payloads[index]
        rows.append(
            {
                "bytes": len(payload),
                "consumer": SECTION_CONSUMERS[name],
                "name": name,
                "sha256": _sha256(payload),
            }
        )
    return rows


def _cli() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("inspect", "render-pair"),
        help="read-only archive inspection or one bounded pair render",
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--pair-index", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    archive_bytes = args.archive.read_bytes()
    parsed = parse_archive(archive_bytes)
    if args.command == "inspect":
        print(
            json.dumps(
                {
                    "archive_bytes": len(archive_bytes),
                    "archive_sha256": _sha256(archive_bytes),
                    "packet_bytes": len(parsed.packet_bytes),
                    "packet_sha256": _sha256(parsed.packet_bytes),
                    "pose_claim": False,
                    "pose_stub_consumed": parsed.packet.pose_stub_consumed,
                    "research_only": True,
                    "score_claim": False,
                    "section_ledger": section_ledger(parsed.packet),
                },
                sort_keys=True,
            )
        )
        return 0
    if args.output is None:
        raise SystemExit("render-pair requires --output")
    pair = render_pair_camera_uint8(parsed.packet, args.pair_index)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        handle.write(pair.tobytes())
        handle.flush()
        os.fsync(handle.fileno())
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by rehearsal CLI
    raise SystemExit(_cli())
