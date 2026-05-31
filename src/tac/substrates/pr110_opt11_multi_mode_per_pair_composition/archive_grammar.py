# SPDX-License-Identifier: MIT
"""OPT11MMP archive grammar per Catalog #146 + #220 + #272.

Frozen-offset declaration in source per canonical-leaderboard-binding-depth
discipline L3 (monolithic single-file 0.bin pattern). The OPT11MMP magic +
multi-mode selector payload binds the substrate's distinguishing feature
(per-pair (selector_a, selector_b) multi-mode composition) per Catalog #272
distinguishing-feature integration contract.

Archive layout (substrate-engineering per canonical L7)
=======================================================

::

    +-------------------------------------------------------------+
    | 32-byte OPT11MMP header:                                    |
    |   magic[8] = b"OPT11MMP"                                    |
    |   version u8 = 1                                            |
    |   modes_per_pair u8 (canonical = 2; the multi-mode k)       |
    |   selector_bits_per_mode u8 (canonical = 4 for K=16 modes)  |
    |   family_pair_index u8 (enum into                           |
    |                          CANONICAL_ORTHOGONAL_FAMILY_PAIRS) |
    |   pr110_base_sha256_prefix[16] u128                         |
    |   reserved[4] u32 = 0                                       |
    +-------------------------------------------------------------+
    | u32 multi_mode_selector_blob_len + brotli-q9(per-pair       |
    |   (selector_a, selector_b) indices packed into uint8/uint16 |
    |   stream; canonical 8 bits per pair = 600 pairs × 8 bits =  |
    |   600 raw bytes → ~428B after brotli q=9 per Wave N+34)     |
    +-------------------------------------------------------------+
    | u32 family_pair_metadata_blob_len + brotli-q9(JSON of       |
    |   family-pair labels + per-family mode menu)                |
    +-------------------------------------------------------------+
    | pr110_base_archive_bytes inline (PRESERVED unchanged from   |
    |   PR110 fec6 baseline; rate-term reuses PR110 bytes)        |
    +-------------------------------------------------------------+

Catalog #146 contest 3-arg inflate.sh signature: ``inflate.sh archive_dir
output_dir file_list``. Per canonical-leaderboard-binding-depth discipline L4:
inflate.py ≤200 LOC base budget (this L0 SCAFFOLD stays within budget;
substrate-engineering exception per L7 only when L1 PROMOTION requires more).

Catalog #205 canonical select_inflate_device. Catalog #295 PYTHONPATH
self-containment per Slot EEE NO FAKE.

Canonical structural extinction surfaces:
- Catalog #220 SCAFFOLD_DEFERRED_INTEGRATION_OK pattern (research_only=true
  until paired-CUDA RATIFICATION lands per Catalog #246).
- Catalog #272 distinguishing-feature integration contract (per-pair multi-
  mode composition IS the canonical distinguishing feature; byte-mutation
  smoke per Catalog #139 validates).
- Catalog #146 contest-compliant inflate runtime contract.
- Catalog #205 canonical select_inflate_device.
- Catalog #295 PYTHONPATH self-containment.
"""

from __future__ import annotations

import struct

# Canonical magic = 8 bytes (OPT11MMP = OPT11 Multi-Mode Per-pair).
ARCHIVE_MAGIC: bytes = b"OPT11MMP"
ARCHIVE_VERSION: int = 1

# Canonical fixed-offset header format:
#   8s  = magic[8]
#   B   = version u8
#   B   = modes_per_pair u8
#   B   = selector_bits_per_mode u8
#   B   = family_pair_index u8
#   16s = pr110_base_sha256_prefix[16]
#   4s  = reserved[4]
OPT11MMP_HEADER_FMT: str = "<8sBBBB16s4s"
OPT11MMP_HEADER_LEN: int = struct.calcsize(OPT11MMP_HEADER_FMT)
assert OPT11MMP_HEADER_LEN == 32, (
    "OPT11MMP_HEADER_LEN must be 32 bytes per the frozen grammar "
    f"(got {OPT11MMP_HEADER_LEN})"
)

# Canonical section indices into the 2 length-prefixed brotli sections that
# follow the header.
SECTION_INDEX_MULTI_MODE_SELECTOR: int = 0
SECTION_INDEX_FAMILY_PAIR_METADATA: int = 1
NUM_SECTIONS: int = 2

# Canonical brotli quality per canonical L32 (brotli quality=11 max compression
# for sidecar). We use q=9 as a balanced default for the L0 SCAFFOLD; L1
# PROMOTION sweeps q=11 per Phase 2 sister wave.
DEFAULT_BROTLI_QUALITY: int = 9


def pack_header(
    *,
    version: int,
    modes_per_pair: int,
    selector_bits_per_mode: int,
    family_pair_index: int,
    pr110_base_sha256_prefix: bytes,
    reserved: bytes = b"\x00\x00\x00\x00",
) -> bytes:
    """Pack the canonical 32-byte OPT11MMP header per the frozen grammar.

    Args:
        version: Archive version u8 (canonical = 1).
        modes_per_pair: Number of modes composed per pair (canonical = 2).
        selector_bits_per_mode: Bits per selector (canonical = 4 for K=16).
        family_pair_index: Index into CANONICAL_ORTHOGONAL_FAMILY_PAIRS (0-5).
        pr110_base_sha256_prefix: First 16 bytes of PR110 base archive sha256.
        reserved: 4 reserved bytes (default = b"\x00\x00\x00\x00").

    Returns:
        32-byte canonical OPT11MMP header.

    Raises:
        ValueError: If any field is invalid.
    """
    if version < 0 or version > 255:
        raise ValueError(f"version must fit u8 [0, 255]; got {version}")
    if modes_per_pair < 1 or modes_per_pair > 255:
        raise ValueError(
            f"modes_per_pair must fit u8 [1, 255]; got {modes_per_pair}"
        )
    if selector_bits_per_mode < 1 or selector_bits_per_mode > 16:
        raise ValueError(
            "selector_bits_per_mode must be in [1, 16]; "
            f"got {selector_bits_per_mode}"
        )
    if family_pair_index < 0 or family_pair_index > 255:
        raise ValueError(
            f"family_pair_index must fit u8 [0, 255]; got {family_pair_index}"
        )
    if not isinstance(pr110_base_sha256_prefix, (bytes, bytearray)):
        raise ValueError(
            "pr110_base_sha256_prefix must be bytes; "
            f"got {type(pr110_base_sha256_prefix).__name__}"
        )
    if len(pr110_base_sha256_prefix) != 16:
        raise ValueError(
            "pr110_base_sha256_prefix must be exactly 16 bytes; "
            f"got {len(pr110_base_sha256_prefix)}"
        )
    if not isinstance(reserved, (bytes, bytearray)):
        raise ValueError(
            f"reserved must be bytes; got {type(reserved).__name__}"
        )
    if len(reserved) != 4:
        raise ValueError(
            f"reserved must be exactly 4 bytes; got {len(reserved)}"
        )
    return struct.pack(
        OPT11MMP_HEADER_FMT,
        ARCHIVE_MAGIC,
        version,
        modes_per_pair,
        selector_bits_per_mode,
        family_pair_index,
        bytes(pr110_base_sha256_prefix),
        bytes(reserved),
    )


def unpack_header(header_bytes: bytes) -> dict:
    """Unpack a canonical 32-byte OPT11MMP header.

    Args:
        header_bytes: 32 bytes of OPT11MMP header.

    Returns:
        Dict with keys: magic, version, modes_per_pair,
        selector_bits_per_mode, family_pair_index,
        pr110_base_sha256_prefix, reserved.

    Raises:
        ValueError: If the bytes don't match the canonical grammar.
    """
    if len(header_bytes) != OPT11MMP_HEADER_LEN:
        raise ValueError(
            f"header_bytes must be exactly {OPT11MMP_HEADER_LEN} bytes; "
            f"got {len(header_bytes)}"
        )
    (
        magic,
        version,
        modes_per_pair,
        selector_bits_per_mode,
        family_pair_index,
        pr110_base_sha256_prefix,
        reserved,
    ) = struct.unpack(OPT11MMP_HEADER_FMT, header_bytes)
    if magic != ARCHIVE_MAGIC:
        raise ValueError(
            f"magic mismatch; expected {ARCHIVE_MAGIC!r} got {magic!r}"
        )
    return {
        "magic": magic,
        "version": version,
        "modes_per_pair": modes_per_pair,
        "selector_bits_per_mode": selector_bits_per_mode,
        "family_pair_index": family_pair_index,
        "pr110_base_sha256_prefix": bytes(pr110_base_sha256_prefix),
        "reserved": bytes(reserved),
    }


def pack_selector_stream(selectors: list[tuple[int, int]], *, selector_bits_per_mode: int = 4) -> bytes:
    """Pack per-pair (selector_a, selector_b) into raw uint8 stream.

    For the canonical K=16 / 4-bits-per-mode case: each pair contributes
    (selector_a << 4) | selector_b into a single uint8 byte. For other
    selector_bits_per_mode values the upper-byte / lower-byte packing
    generalizes (modes_per_pair * selector_bits_per_mode bits per pair).

    Args:
        selectors: List of (mode_a_idx, mode_b_idx) tuples per pair.
        selector_bits_per_mode: Canonical 4 for K=16 modes.

    Returns:
        Raw uint8 byte stream of length len(selectors) bytes (canonical 4+4
        case) or 2*len(selectors) bytes for selector_bits_per_mode > 4.

    Raises:
        ValueError: If any selector exceeds the canonical bit budget.
    """
    max_value = (1 << selector_bits_per_mode) - 1
    out = bytearray()
    if selector_bits_per_mode <= 4:
        # Pack two 4-bit selectors per byte (the canonical K=16 case).
        for pid, (a, b) in enumerate(selectors):
            if a < 0 or a > max_value:
                raise ValueError(
                    f"selector_a={a} at pair {pid} exceeds 4-bit budget "
                    f"[0, {max_value}]"
                )
            if b < 0 or b > max_value:
                raise ValueError(
                    f"selector_b={b} at pair {pid} exceeds 4-bit budget "
                    f"[0, {max_value}]"
                )
            out.append(((a & 0x0F) << 4) | (b & 0x0F))
    else:
        # Generalized: 1 byte per selector (handles K up to 256).
        for pid, (a, b) in enumerate(selectors):
            if a < 0 or a > max_value:
                raise ValueError(
                    f"selector_a={a} at pair {pid} exceeds {selector_bits_per_mode}-bit "
                    f"budget [0, {max_value}]"
                )
            if b < 0 or b > max_value:
                raise ValueError(
                    f"selector_b={b} at pair {pid} exceeds {selector_bits_per_mode}-bit "
                    f"budget [0, {max_value}]"
                )
            out.append(a & 0xFF)
            out.append(b & 0xFF)
    return bytes(out)


def unpack_selector_stream(
    stream: bytes,
    *,
    n_pairs: int,
    selector_bits_per_mode: int = 4,
) -> list[tuple[int, int]]:
    """Unpack raw uint8 stream back into per-pair (selector_a, selector_b).

    Inverse of :func:`pack_selector_stream`.

    Args:
        stream: Raw uint8 byte stream.
        n_pairs: Expected number of pairs (for length validation).
        selector_bits_per_mode: Must match the pack_selector_stream call.

    Returns:
        List of (mode_a_idx, mode_b_idx) tuples per pair.

    Raises:
        ValueError: If stream length doesn't match expected.
    """
    out: list[tuple[int, int]] = []
    if selector_bits_per_mode <= 4:
        if len(stream) != n_pairs:
            raise ValueError(
                f"stream length must be n_pairs={n_pairs} for 4-bit packing; "
                f"got {len(stream)}"
            )
        for byte in stream:
            a = (byte >> 4) & 0x0F
            b = byte & 0x0F
            out.append((a, b))
    else:
        if len(stream) != 2 * n_pairs:
            raise ValueError(
                f"stream length must be 2*n_pairs={2*n_pairs} for >4-bit "
                f"packing; got {len(stream)}"
            )
        for i in range(n_pairs):
            a = stream[2 * i]
            b = stream[2 * i + 1]
            out.append((a, b))
    return out
