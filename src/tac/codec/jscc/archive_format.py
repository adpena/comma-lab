"""JSCC archive-section format.

The JSCC archive section is a self-describing binary blob with the layout:

    +---------+------+-------------+-----------+-------------+--------+
    | MAGIC   | VER  | SIDE_DIM    | ALPHABET  | N_SYMBOLS   | PAYLOAD|
    | (4 B)   | (1B) | (2 B BE)    | (4 B BE)  | (4 B BE)    |        |
    +---------+------+-------------+-----------+-------------+--------+

* ``MAGIC`` — ``b"JSCC"`` (4 bytes), see ``JSCC_MAGIC``.
* ``VER``   — format version byte; current is ``JSCC_FORMAT_VERSION = 1``.
* ``SIDE_DIM`` — uint16 big-endian; the model's side-state dimensionality.
* ``ALPHABET`` — uint32 big-endian; alphabet size K.
* ``N_SYMBOLS`` — uint32 big-endian; number of encoded symbols.
* ``PAYLOAD`` — the range-coded byte stream.

NOTE: this section does NOT carry the model weights. The decoder is expected
to load the model from a sibling location in the archive (or share it across
all sections in a single archive). The model-weight blob format is OUT OF
SCOPE for this primitive — wire it into the archive grammar at integration
time.

Cross-references
----------------
- Sister archive format for unconditional arithmetic coding:
  ``tac.packet_compiler.pr103_arithmetic_coding`` (different magic, different
  layout — JSCC is intentionally a separate primitive).
- Sister magic-codec wrapper grammar:
  ``tac.packet_compiler.magic_codec``.

Lane: ``lane_implement_iglt_ternary_jscc_kc3_canonical_20260513``.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from tac.codec.jscc.entropy_coder import JSCC_FORMAT_VERSION, JSCC_MAGIC

__all__ = [
    "JSCCArchiveSection",
    "JSCCSectionManifest",
    "parse_jscc_section",
    "serialize_jscc_section",
]


@dataclass(frozen=True)
class JSCCSectionManifest:
    """Typed metadata for a JSCC archive section.

    Used by the packet_compiler grammar / parser_section_manifest pipeline
    (per CLAUDE.md HNeRV parity discipline lesson 3 — every monolithic
    archive must declare fixed offsets).

    Attributes:
        magic: ``JSCC_MAGIC`` (4 bytes).
        version: format-version byte.
        side_dim: model's side-state dimensionality.
        alphabet_size: alphabet size K.
        n_symbols: number of encoded symbols.
        payload_offset: byte offset where the encoded payload starts within
            this section.
        payload_length: byte length of the encoded payload.
        total_section_bytes: header_bytes + payload_length.
    """

    magic: bytes
    version: int
    side_dim: int
    alphabet_size: int
    n_symbols: int
    payload_offset: int
    payload_length: int
    total_section_bytes: int


@dataclass(frozen=True)
class JSCCArchiveSection:
    """A parsed JSCC archive section (header + payload).

    Attributes:
        manifest: typed metadata.
        payload: the raw range-coded byte stream.
    """

    manifest: JSCCSectionManifest
    payload: bytes


# Fixed header layout, big-endian:
#   magic[4] || version[1] || side_dim[2] || alphabet[4] || n_symbols[4]
# = 15 bytes header
_HEADER_STRUCT = struct.Struct(">4sBHII")
HEADER_BYTES: int = _HEADER_STRUCT.size  # 15


def serialize_jscc_section(
    payload: bytes,
    side_dim: int,
    alphabet_size: int,
    n_symbols: int,
    version: int = JSCC_FORMAT_VERSION,
) -> bytes:
    """Wrap a JSCC payload in the archive-section header.

    Args:
        payload: range-coded byte stream from
            ``encode_jscc_stream`` / ``ScorerConditionalEntropyCoder.encode``.
        side_dim: model's side-state dimensionality.
        alphabet_size: alphabet size K.
        n_symbols: number of encoded symbols.
        version: format version. Default is current.

    Returns:
        Header + payload bytes (length = HEADER_BYTES + len(payload)).

    Raises:
        ValueError: on out-of-range inputs.
    """
    if not (0 <= version <= 255):
        raise ValueError(f"version must fit in uint8, got {version}")
    if not (0 <= side_dim <= 0xFFFF):
        raise ValueError(f"side_dim must fit in uint16, got {side_dim}")
    if not (0 <= alphabet_size <= 0xFFFFFFFF):
        raise ValueError(
            f"alphabet_size must fit in uint32, got {alphabet_size}"
        )
    if not (0 <= n_symbols <= 0xFFFFFFFF):
        raise ValueError(
            f"n_symbols must fit in uint32, got {n_symbols}"
        )
    if alphabet_size < 2:
        raise ValueError(f"alphabet_size must be >= 2, got {alphabet_size}")
    header = _HEADER_STRUCT.pack(
        JSCC_MAGIC, version, side_dim, alphabet_size, n_symbols
    )
    return header + payload


def parse_jscc_section(section_bytes: bytes) -> JSCCArchiveSection:
    """Parse a JSCC archive section.

    Args:
        section_bytes: header + payload.

    Returns:
        ``JSCCArchiveSection`` with typed manifest + raw payload.

    Raises:
        ValueError: on magic / version / length mismatches.
    """
    if len(section_bytes) < HEADER_BYTES:
        raise ValueError(
            f"section too short ({len(section_bytes)} < {HEADER_BYTES})"
        )
    magic, version, side_dim, alphabet_size, n_symbols = _HEADER_STRUCT.unpack(
        section_bytes[:HEADER_BYTES]
    )
    if magic != JSCC_MAGIC:
        raise ValueError(
            f"bad magic: expected {JSCC_MAGIC!r}, got {magic!r}"
        )
    if version != JSCC_FORMAT_VERSION:
        raise ValueError(
            f"unsupported JSCC version {version}; this build supports "
            f"version {JSCC_FORMAT_VERSION}"
        )
    payload = section_bytes[HEADER_BYTES:]
    manifest = JSCCSectionManifest(
        magic=JSCC_MAGIC,
        version=int(version),
        side_dim=int(side_dim),
        alphabet_size=int(alphabet_size),
        n_symbols=int(n_symbols),
        payload_offset=HEADER_BYTES,
        payload_length=len(payload),
        total_section_bytes=len(section_bytes),
    )
    return JSCCArchiveSection(manifest=manifest, payload=payload)
