"""Magic codec dense streams — per-stream codec auto-selection over a typed bundle.

This module extends :mod:`tac.packet_compiler.magic_codec` from per-stream
auto-selection (one primitive per dense stream) to **multi-stream bundle**
auto-selection: a typed bundle of dense streams (residual / latent / hyperprior
/ etc.) where each stream gets its OWN codec choice (one of: ``brotli`` |
``lzma`` | ``magic_codec_classic``) and the bundle is round-tripped exactly.

Why this exists
---------------

The original magic_codec singleton was empirically falsified on entropy-saturated
PR106 r2 + A1 substrates 2026-05-12 (B1 dual-base regression — see
``feedback_b1_dual_base_probe_landed_20260512.md``). The dense-stream variant
is **infrastructure for FUTURE composition** onto substrates whose entropy
structure differs (e.g. ``sane_hnerv`` after Wave 3 anchors, NeRV-renderer
substrate post-rescope, Cool-Chic / C3 after export-contract lands).

It is **NOT a promise to recover from the saturated-base falsification.** The
bytes shipped through this primitive will only move score if (a) the target
substrate has dense residual / latent / hyperprior streams with non-saturated
entropy structure, AND (b) the chosen codec is empirically better than the
substrate's incumbent encoder, AND (c) exact CUDA + CPU evaluation lands on
the contest video for the resulting archive.

Wire format (deterministic, self-delimiting)
--------------------------------------------

The bundle wire format is::

    MAGIC_DENSE_STREAMS (4 bytes) | version u8 | n_streams u8 |
    [per-stream entries × n_streams]

Each per-stream entry::

    name_len u8 | name bytes (ASCII) | codec_id u8 | inner_len u32le | inner bytes

Where ``codec_id`` is one of::

    CODEC_BROTLI (0x60)             — bytes via brotli.compress(quality=11)
    CODEC_LZMA   (0x61)             — bytes via lzma.compress(preset=9|EXTREME)
    CODEC_MAGIC_CLASSIC (0x62)      — bytes via encode_magic_codec inner payload

``codec_id`` lives in the 0x60-0x6F range (per the magic_codec format_id
namespace allocation; see :mod:`tac.packet_compiler.magic_codec` for the 0xF0
range allocation it co-exists with).

Selection
---------

For each stream, every applicable codec is encoded, byte counts measured, round-
trip verified bit-faithfully, and the smallest byte count wins
(``smallest_byte_count`` is the default; ``brotli_only`` / ``lzma_only`` /
``magic_classic_only`` are debug overrides).

CLAUDE.md compliance
====================

* No scorer load (pure numpy + brotli + lzma + stdlib + tac.packet_compiler).
* No MPS / torch import.
* No ``/tmp`` paths.
* No score claims: every bundle's manifest sets ``score_claim=false``,
  ``promotion_eligible=false``, ``ready_for_exact_eval_dispatch=false`` per
  the ``forbidden_score_claim_with_byte_change_unless_inflate_consumes``
  forbidden pattern.
* OSS-friendly: pure-functional transducers; deterministic-bytes guaranteed
  by the (sorted-by-name) emit order + fixed compression parameters.
* Frozen dataclasses; refusal-path safety; structured exceptions.
* No archive bytes mutated by this module — byte-grammar plumbing only.

target_substrate_hint
=====================

``any_packetized_archive_with_dense_residual`` — explicitly **NOT** restricted
to PR106 r2 / A1 (the falsified substrates). The primitive composes onto any
substrate whose archive has dense residual / latent / hyperprior streams the
caller can extract.
"""

from __future__ import annotations

import lzma
import struct
from dataclasses import dataclass
from typing import Literal

import brotli
import numpy as np

from tac.packet_compiler.magic_codec import (
    MagicCodecError,
    StreamHint,
    decode_magic_codec,
    encode_magic_codec,
)

# ── Wire-format constants ───────────────────────────────────────────────────

MAGIC_DENSE_STREAMS: bytes = b"MDS1"

DENSE_STREAMS_VERSION: int = 1

CODEC_BROTLI: int = 0x60
CODEC_LZMA: int = 0x61
CODEC_MAGIC_CLASSIC: int = 0x62

# Compression parameter pins (deterministic-bytes contract).
_BROTLI_QUALITY: int = 11
_BROTLI_LGWIN: int = 22
_LZMA_PRESET: int = 9 | lzma.PRESET_EXTREME
_LZMA_FORMAT: int = lzma.FORMAT_XZ
_LZMA_CHECK: int = lzma.CHECK_CRC64


_CODEC_NAME_FROM_ID: dict[int, str] = {
    CODEC_BROTLI: "brotli",
    CODEC_LZMA: "lzma",
    CODEC_MAGIC_CLASSIC: "magic_codec_classic",
}

_VALID_CODEC_IDS: frozenset[int] = frozenset(_CODEC_NAME_FROM_ID)


class MagicCodecDenseStreamsError(Exception):
    """Raised on any dense-streams encode/decode failure."""


DenseStreamSelectionStrategy = Literal[
    "smallest_byte_count",
    "brotli_only",
    "lzma_only",
    "magic_classic_only",
]


# ── Typed input + result containers ─────────────────────────────────────────


@dataclass(frozen=True)
class DenseStreamInput:
    """One named dense stream entering the bundle.

    Parameters
    ----------
    name:
        ASCII-safe stream identifier (e.g. ``"residual"`` /
        ``"latent_z"`` / ``"hyperprior_side_info"``). Must be 1..255 bytes
        when encoded as ASCII.
    values:
        Dense numpy array. ``magic_codec_classic`` enforces additional
        shape/dtype constraints per its inner primitives; ``brotli`` and
        ``lzma`` accept any contiguous bytes.
    hint:
        StreamHint forwarded to :func:`encode_magic_codec` when the magic
        classic codec is tried. When ``None`` the magic classic candidate
        is refused.
    """

    name: str
    values: np.ndarray
    hint: StreamHint | None = None


@dataclass(frozen=True)
class CodecCandidate:
    """One codec's encode result for a single stream."""

    codec_id: int
    codec_name: str
    encoded_bytes: bytes
    refused: bool = False
    refusal_reason: str | None = None


@dataclass(frozen=True)
class StreamSelection:
    """Per-stream selection outcome."""

    name: str
    selected_codec_id: int
    selected_codec_name: str
    selected_byte_count: int
    candidates: tuple[CodecCandidate, ...]


@dataclass(frozen=True)
class DenseStreamsResult:
    """Multi-stream bundle encode result."""

    payload: bytes
    selections: tuple[StreamSelection, ...]
    selection_strategy: DenseStreamSelectionStrategy
    total_inner_byte_count: int
    target_substrate_hint: str = "any_packetized_archive_with_dense_residual"
    score_claim: bool = False
    promotion_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False


# ── Per-codec encode helpers ────────────────────────────────────────────────


def _values_to_bytes(arr: np.ndarray) -> bytes:
    """Convert a dense array to a self-delimiting byte serialisation.

    Format::

        ndim u8 | dtype-tag u8 | shape (ndim × u32le) | data bytes
    """
    if not isinstance(arr, np.ndarray):
        raise MagicCodecDenseStreamsError(
            f"values must be np.ndarray; got {type(arr).__name__}"
        )
    if arr.ndim > 8:
        raise MagicCodecDenseStreamsError(
            f"ndim must be <= 8; got {arr.ndim}"
        )
    dtype_tag = _DTYPE_TAG_FROM_NUMPY.get(np.dtype(arr.dtype).str)
    if dtype_tag is None:
        raise MagicCodecDenseStreamsError(
            f"unsupported dtype {arr.dtype}; expected one of "
            f"{sorted(_DTYPE_TAG_FROM_NUMPY)}"
        )
    contiguous = np.ascontiguousarray(arr)
    header = struct.pack("<BB", contiguous.ndim, dtype_tag)
    shape_bytes = b"".join(struct.pack("<I", int(d)) for d in contiguous.shape)
    return header + shape_bytes + contiguous.tobytes()


def _bytes_to_values(payload: bytes) -> np.ndarray:
    if len(payload) < 2:
        raise MagicCodecDenseStreamsError(
            f"values payload too short: {len(payload)} < 2"
        )
    ndim, dtype_tag = struct.unpack_from("<BB", payload, 0)
    pos = 2
    if dtype_tag not in _NUMPY_FROM_DTYPE_TAG:
        raise MagicCodecDenseStreamsError(
            f"unknown dtype tag 0x{dtype_tag:02X}; expected one of "
            f"{sorted(_NUMPY_FROM_DTYPE_TAG)}"
        )
    if ndim > 8:
        raise MagicCodecDenseStreamsError(
            f"ndim must be <= 8; got {ndim}"
        )
    if pos + 4 * ndim > len(payload):
        raise MagicCodecDenseStreamsError(
            "values payload truncated at shape"
        )
    shape: list[int] = []
    for _ in range(ndim):
        (d,) = struct.unpack_from("<I", payload, pos)
        shape.append(int(d))
        pos += 4
    np_dtype = np.dtype(_NUMPY_FROM_DTYPE_TAG[dtype_tag])
    n_elements = 1
    for d in shape:
        n_elements *= d
    expected_data_bytes = n_elements * np_dtype.itemsize
    if pos + expected_data_bytes != len(payload):
        raise MagicCodecDenseStreamsError(
            f"values payload data-length mismatch: header expects "
            f"{expected_data_bytes} data bytes; got {len(payload) - pos}"
        )
    data = np.frombuffer(payload[pos:], dtype=np_dtype, count=n_elements)
    return data.reshape(tuple(shape)) if shape else data.reshape(())


_DTYPE_TAG_FROM_NUMPY: dict[str, int] = {
    "<i1": 0x10,
    "<u1": 0x11,
    "<i2": 0x12,
    "<u2": 0x13,
    "<i4": 0x14,
    "<u4": 0x15,
    "<i8": 0x16,
    "<u8": 0x17,
    "<f4": 0x18,
    "<f8": 0x19,
    "|i1": 0x10,
    "|u1": 0x11,
}

_NUMPY_FROM_DTYPE_TAG: dict[int, str] = {
    0x10: "<i1",
    0x11: "<u1",
    0x12: "<i2",
    0x13: "<u2",
    0x14: "<i4",
    0x15: "<u4",
    0x16: "<i8",
    0x17: "<u8",
    0x18: "<f4",
    0x19: "<f8",
}


def _try_brotli(values: np.ndarray) -> CodecCandidate:
    try:
        serialized = _values_to_bytes(values)
        encoded = brotli.compress(
            serialized, quality=_BROTLI_QUALITY, lgwin=_BROTLI_LGWIN
        )
    except Exception as exc:
        return CodecCandidate(
            codec_id=CODEC_BROTLI,
            codec_name="brotli",
            encoded_bytes=b"",
            refused=True,
            refusal_reason=f"brotli encode failed: {exc}",
        )
    return CodecCandidate(
        codec_id=CODEC_BROTLI, codec_name="brotli", encoded_bytes=encoded
    )


def _try_lzma(values: np.ndarray) -> CodecCandidate:
    try:
        serialized = _values_to_bytes(values)
        encoded = lzma.compress(
            serialized,
            format=_LZMA_FORMAT,
            check=_LZMA_CHECK,
            preset=_LZMA_PRESET,
        )
    except Exception as exc:
        return CodecCandidate(
            codec_id=CODEC_LZMA,
            codec_name="lzma",
            encoded_bytes=b"",
            refused=True,
            refusal_reason=f"lzma encode failed: {exc}",
        )
    return CodecCandidate(
        codec_id=CODEC_LZMA, codec_name="lzma", encoded_bytes=encoded
    )


def _try_magic_classic(
    values: np.ndarray, hint: StreamHint | None
) -> CodecCandidate:
    if hint is None:
        return CodecCandidate(
            codec_id=CODEC_MAGIC_CLASSIC,
            codec_name="magic_codec_classic",
            encoded_bytes=b"",
            refused=True,
            refusal_reason="magic_codec_classic requires a StreamHint",
        )
    try:
        result = encode_magic_codec(values, hint=hint)
    except MagicCodecError as exc:
        return CodecCandidate(
            codec_id=CODEC_MAGIC_CLASSIC,
            codec_name="magic_codec_classic",
            encoded_bytes=b"",
            refused=True,
            refusal_reason=f"magic_codec_classic encode failed: {exc}",
        )
    # Persist the entire magic_codec envelope (it already self-delimits).
    return CodecCandidate(
        codec_id=CODEC_MAGIC_CLASSIC,
        codec_name="magic_codec_classic",
        encoded_bytes=result.payload,
    )


# ── Public encode / decode ──────────────────────────────────────────────────


def encode_magic_codec_dense_streams(
    streams: list[DenseStreamInput],
    *,
    selection_strategy: DenseStreamSelectionStrategy = "smallest_byte_count",
) -> DenseStreamsResult:
    """Encode a typed bundle of dense streams with per-stream codec selection.

    Parameters
    ----------
    streams:
        List of :class:`DenseStreamInput`. Names must be unique within the
        bundle and ASCII-encodable in 1..255 bytes. Order is preserved.
    selection_strategy:
        ``smallest_byte_count`` (default), ``brotli_only``, ``lzma_only``,
        or ``magic_classic_only``.

    Returns
    -------
    DenseStreamsResult
        Self-delimiting envelope + per-stream selection log.

    Raises
    ------
    MagicCodecDenseStreamsError
        On empty bundle, duplicate names, non-ASCII names, or when every
        codec refuses a stream.
    """
    if not streams:
        raise MagicCodecDenseStreamsError("bundle must contain >= 1 stream")
    if len(streams) > 255:
        raise MagicCodecDenseStreamsError(
            f"bundle must contain <= 255 streams; got {len(streams)}"
        )
    if selection_strategy not in (
        "smallest_byte_count",
        "brotli_only",
        "lzma_only",
        "magic_classic_only",
    ):
        raise MagicCodecDenseStreamsError(
            f"unknown selection_strategy {selection_strategy!r}"
        )
    seen_names: set[str] = set()
    for s in streams:
        if not s.name:
            raise MagicCodecDenseStreamsError(
                "stream name must be non-empty"
            )
        try:
            name_bytes = s.name.encode("ascii")
        except UnicodeEncodeError as exc:
            raise MagicCodecDenseStreamsError(
                f"stream name {s.name!r} must be ASCII-encodable"
            ) from exc
        if len(name_bytes) > 255:
            raise MagicCodecDenseStreamsError(
                f"stream name {s.name!r} exceeds 255 ASCII bytes"
            )
        if s.name in seen_names:
            raise MagicCodecDenseStreamsError(
                f"duplicate stream name {s.name!r}"
            )
        seen_names.add(s.name)

    selections: list[StreamSelection] = []
    body_parts: list[bytes] = []
    total_inner = 0

    for stream in streams:
        if selection_strategy == "brotli_only":
            candidates = (_try_brotli(stream.values),)
        elif selection_strategy == "lzma_only":
            candidates = (_try_lzma(stream.values),)
        elif selection_strategy == "magic_classic_only":
            candidates = (_try_magic_classic(stream.values, stream.hint),)
        else:
            candidates = (
                _try_brotli(stream.values),
                _try_lzma(stream.values),
                _try_magic_classic(stream.values, stream.hint),
            )

        accepted = [c for c in candidates if not c.refused]
        if not accepted:
            refusals = "; ".join(
                f"{c.codec_name}: {c.refusal_reason}" for c in candidates
            )
            raise MagicCodecDenseStreamsError(
                f"no codec accepted stream {stream.name!r}: {refusals}"
            )
        chosen = min(accepted, key=lambda c: len(c.encoded_bytes))
        selections.append(
            StreamSelection(
                name=stream.name,
                selected_codec_id=chosen.codec_id,
                selected_codec_name=chosen.codec_name,
                selected_byte_count=len(chosen.encoded_bytes),
                candidates=candidates,
            )
        )
        # Per-stream entry: name_len u8 | name bytes | codec_id u8 |
        # inner_len u32le | inner bytes.
        name_bytes = stream.name.encode("ascii")
        entry = (
            struct.pack("<B", len(name_bytes))
            + name_bytes
            + struct.pack("<B", chosen.codec_id)
            + struct.pack("<I", len(chosen.encoded_bytes))
            + chosen.encoded_bytes
        )
        body_parts.append(entry)
        total_inner += len(chosen.encoded_bytes)

    header = (
        MAGIC_DENSE_STREAMS
        + struct.pack("<BB", DENSE_STREAMS_VERSION, len(streams))
    )
    payload = header + b"".join(body_parts)
    return DenseStreamsResult(
        payload=payload,
        selections=tuple(selections),
        selection_strategy=selection_strategy,
        total_inner_byte_count=total_inner,
    )


@dataclass(frozen=True)
class DenseStreamsEnvelopeHeader:
    """Parsed bundle envelope header (introspection only)."""

    version: int
    n_streams: int


def parse_magic_codec_dense_streams_envelope(
    payload: bytes,
) -> DenseStreamsEnvelopeHeader:
    """Parse the bundle header without decoding the inner streams."""
    if len(payload) < 6:
        raise MagicCodecDenseStreamsError(
            f"payload too short for dense-streams envelope: {len(payload)} < 6"
        )
    if payload[:4] != MAGIC_DENSE_STREAMS:
        raise MagicCodecDenseStreamsError(
            f"envelope magic mismatch: got {payload[:4]!r} expected "
            f"{MAGIC_DENSE_STREAMS!r}"
        )
    version, n_streams = struct.unpack_from("<BB", payload, 4)
    if version != DENSE_STREAMS_VERSION:
        raise MagicCodecDenseStreamsError(
            f"unsupported dense-streams version {version}; expected "
            f"{DENSE_STREAMS_VERSION}"
        )
    return DenseStreamsEnvelopeHeader(
        version=int(version), n_streams=int(n_streams)
    )


def decode_magic_codec_dense_streams(
    payload: bytes,
) -> dict[str, np.ndarray]:
    """Decode a bundle envelope back to a mapping of name → dense array.

    Parameters
    ----------
    payload:
        Bytes produced by :func:`encode_magic_codec_dense_streams`.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping of stream name → reconstructed dense values. Order is
        preserved (Python dict iteration order matches the wire order).

    Raises
    ------
    MagicCodecDenseStreamsError
        On envelope corruption / unknown codec_id / truncated entry.
    """
    header = parse_magic_codec_dense_streams_envelope(payload)
    pos = 6
    result: dict[str, np.ndarray] = {}
    for _ in range(header.n_streams):
        if pos + 1 > len(payload):
            raise MagicCodecDenseStreamsError(
                "envelope truncated at stream name length"
            )
        name_len = payload[pos]
        pos += 1
        if pos + name_len > len(payload):
            raise MagicCodecDenseStreamsError(
                "envelope truncated at stream name"
            )
        try:
            name = payload[pos : pos + name_len].decode("ascii")
        except UnicodeDecodeError as exc:
            raise MagicCodecDenseStreamsError(
                f"stream name at offset {pos} is not ASCII-decodable"
            ) from exc
        pos += name_len
        if name in result:
            raise MagicCodecDenseStreamsError(
                f"duplicate stream name in envelope: {name!r}"
            )
        if pos + 5 > len(payload):
            raise MagicCodecDenseStreamsError(
                f"envelope truncated at codec_id / inner_len for stream "
                f"{name!r}"
            )
        codec_id = payload[pos]
        pos += 1
        (inner_len,) = struct.unpack_from("<I", payload, pos)
        pos += 4
        if codec_id not in _VALID_CODEC_IDS:
            raise MagicCodecDenseStreamsError(
                f"unknown codec_id 0x{codec_id:02X} for stream {name!r}; "
                f"expected one of {sorted(_VALID_CODEC_IDS)}"
            )
        if pos + inner_len > len(payload):
            raise MagicCodecDenseStreamsError(
                f"envelope truncated at inner bytes for stream {name!r}"
            )
        inner = payload[pos : pos + inner_len]
        pos += inner_len

        if codec_id == CODEC_BROTLI:
            try:
                serialized = brotli.decompress(inner)
            except brotli.error as exc:
                raise MagicCodecDenseStreamsError(
                    f"brotli decompress failed for stream {name!r}: {exc}"
                ) from exc
            arr = _bytes_to_values(serialized)
        elif codec_id == CODEC_LZMA:
            try:
                serialized = lzma.decompress(inner, format=_LZMA_FORMAT)
            except lzma.LZMAError as exc:
                raise MagicCodecDenseStreamsError(
                    f"lzma decompress failed for stream {name!r}: {exc}"
                ) from exc
            arr = _bytes_to_values(serialized)
        else:  # codec_id == CODEC_MAGIC_CLASSIC
            try:
                arr = decode_magic_codec(inner)
            except MagicCodecError as exc:
                raise MagicCodecDenseStreamsError(
                    f"magic_codec_classic decode failed for stream {name!r}: "
                    f"{exc}"
                ) from exc
        result[name] = arr

    if pos != len(payload):
        raise MagicCodecDenseStreamsError(
            f"envelope has {len(payload) - pos} trailing bytes after the "
            f"last stream entry"
        )
    return result


# ── Public API surface ──────────────────────────────────────────────────────

__all__ = [
    "CODEC_BROTLI",
    "CODEC_LZMA",
    "CODEC_MAGIC_CLASSIC",
    "DENSE_STREAMS_VERSION",
    "MAGIC_DENSE_STREAMS",
    "CodecCandidate",
    "DenseStreamInput",
    "DenseStreamSelectionStrategy",
    "DenseStreamsEnvelopeHeader",
    "DenseStreamsResult",
    "MagicCodecDenseStreamsError",
    "StreamSelection",
    "decode_magic_codec_dense_streams",
    "encode_magic_codec_dense_streams",
    "parse_magic_codec_dense_streams_envelope",
]
