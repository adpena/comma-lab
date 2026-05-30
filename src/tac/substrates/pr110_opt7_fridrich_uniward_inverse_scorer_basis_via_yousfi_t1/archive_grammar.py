# SPDX-License-Identifier: MIT
"""OPT7VYT1 archive grammar per Catalog #146 + #220 + #272.

Frozen-offset declaration in source per HNeRV parity L3 monolithic-single-file
0.bin pattern. The OPT7VYT1 magic + 4-section length-prefixed payload binds
the substrate's distinguishing feature (5-helper canonical composition) per
Catalog #272 distinguishing-feature integration contract.

Archive layout (substrate-engineering per HNeRV parity L7)
==========================================================

::

    +-------------------------------------------------------------+
    | 32-byte OPT7VYT1 header:                                    |
    |   magic[8] = b"OPT7VYT1"                                    |
    |   version u8 = 1                                            |
    |   alaska_color_branch u8 (enum index into ColorBranchSlice) |
    |   basis_strategy u8 (enum index into InverseScorerBasis)    |
    |   chroma_strategy u8 (enum index into ChromaPerturbation)   |
    |   pr110_base_sha256_prefix[16] u128                         |
    |   reserved[4] u32 = 0                                       |
    +-------------------------------------------------------------+
    | u32 pose_vulnerability_blob_len + brotli-q9(vulnerability   |
    |   map serialized JSON; carries POSE-VULNERABLE pair indices |
    |   + quartile thresholds + vulnerability_ratio)              |
    +-------------------------------------------------------------+
    | u32 alaska_color_branch_blob_len + brotli-q9(YUV6 channel   |
    |   slice metadata serialized JSON)                           |
    +-------------------------------------------------------------+
    | u32 inverse_scorer_basis_blob_len + brotli-q9(Wave N+34     |
    |   UNIWARD per-pair selector indices + costs JSON)           |
    +-------------------------------------------------------------+
    | u32 chroma_perturbation_blob_len + brotli-q9(per-pair       |
    |   chroma magnitude metadata JSON)                           |
    +-------------------------------------------------------------+
    | pr110_base_archive_bytes inline (PRESERVED unchanged from   |
    |   PR110 fec6 baseline; rate-term reuses PR110 bytes)        |
    +-------------------------------------------------------------+

Catalog #146 contest 3-arg inflate.sh signature: ``inflate.sh archive_dir
output_dir file_list``. Per HNeRV parity L4: inflate.py ≤200 LOC base budget
(substrate-engineering exception per L7 for L1 PROMOTION). Per Catalog #205:
canonical select_inflate_device. Per Catalog #295: PYTHONPATH self-containment.

Canonical structural extinction surfaces:
- Catalog #220 SCAFFOLD_DEFERRED_INTEGRATION_OK pattern (L1 PROMOTION
  declares ``research_only=true`` until paired-CUDA RATIFICATION lands).
- Catalog #272 distinguishing-feature integration contract (the 5 helper
  composition IS the canonical distinguishing feature; byte-mutation smoke
  validates per Catalog #139).
- Catalog #146 contest-compliant inflate runtime contract.
- Catalog #205 canonical select_inflate_device.
- Catalog #295 PYTHONPATH self-containment (per Slot EEE NO FAKE).
"""

from __future__ import annotations

import struct


# Canonical magic = 8 bytes (OPT7VYT1 = OPT7 Via Yousfi Tier 1).
ARCHIVE_MAGIC: bytes = b"OPT7VYT1"
ARCHIVE_VERSION: int = 1

# Canonical fixed-offset header format:
#   8s = magic[8]
#   B  = version u8
#   B  = alaska_color_branch u8
#   B  = basis_strategy u8
#   B  = chroma_strategy u8
#   16s = pr110_base_sha256_prefix[16]
#   4s = reserved[4]
OPT7VYT1_HEADER_FMT: str = "<8sBBBB16s4s"
OPT7VYT1_HEADER_LEN: int = struct.calcsize(OPT7VYT1_HEADER_FMT)
assert OPT7VYT1_HEADER_LEN == 32, (
    "OPT7VYT1_HEADER_LEN must be 32 bytes per the frozen grammar "
    f"(got {OPT7VYT1_HEADER_LEN})"
)

# Canonical section indices into the 4 length-prefixed brotli sections that
# follow the header.
SECTION_INDEX_VULNERABILITY_MAP: int = 0
SECTION_INDEX_ALASKA_COLOR_BRANCH: int = 1
SECTION_INDEX_INVERSE_SCORER_BASIS: int = 2
SECTION_INDEX_CHROMA_PERTURBATION: int = 3
NUM_SECTIONS: int = 4

# Canonical brotli quality per HNeRV parity L32 (brotli quality=11 max
# compression for sidecar). We use q=9 as a balanced default; the
# substrate-engineering L1 PROMOTION sweeps q=11 at L2 per Phase 2 sister
# wave.
DEFAULT_BROTLI_QUALITY: int = 9


def pack_header(
    *,
    version: int,
    alaska_color_branch_index: int,
    basis_strategy_index: int,
    chroma_strategy_index: int,
    pr110_base_sha256_prefix: bytes,
    reserved: bytes = b"\x00\x00\x00\x00",
) -> bytes:
    """Pack the canonical 32-byte OPT7VYT1 header per the frozen grammar.

    Args:
        version: Archive version u8 (canonical = 1).
        alaska_color_branch_index: ColorBranchSliceStrategy enum index u8.
        basis_strategy_index: InverseScorerBasisStrategy enum index u8.
        chroma_strategy_index: ChromaPerturbationStrategy enum index u8.
        pr110_base_sha256_prefix: First 16 bytes of PR110 base archive sha256.
        reserved: 4 reserved bytes (default = b"\x00\x00\x00\x00").

    Returns:
        32-byte canonical OPT7VYT1 header.

    Raises:
        ValueError: If any field is invalid.
    """
    if version < 0 or version > 255:
        raise ValueError(f"version must fit u8 [0, 255]; got {version}")
    if alaska_color_branch_index < 0 or alaska_color_branch_index > 255:
        raise ValueError(
            f"alaska_color_branch_index must fit u8 [0, 255]; "
            f"got {alaska_color_branch_index}"
        )
    if basis_strategy_index < 0 or basis_strategy_index > 255:
        raise ValueError(
            f"basis_strategy_index must fit u8 [0, 255]; got {basis_strategy_index}"
        )
    if chroma_strategy_index < 0 or chroma_strategy_index > 255:
        raise ValueError(
            f"chroma_strategy_index must fit u8 [0, 255]; "
            f"got {chroma_strategy_index}"
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
        OPT7VYT1_HEADER_FMT,
        ARCHIVE_MAGIC,
        version,
        alaska_color_branch_index,
        basis_strategy_index,
        chroma_strategy_index,
        bytes(pr110_base_sha256_prefix),
        bytes(reserved),
    )


def unpack_header(header_bytes: bytes) -> dict:
    """Unpack a canonical 32-byte OPT7VYT1 header.

    Args:
        header_bytes: 32 bytes of OPT7VYT1 header.

    Returns:
        Dict with keys: magic, version, alaska_color_branch_index,
        basis_strategy_index, chroma_strategy_index,
        pr110_base_sha256_prefix, reserved.

    Raises:
        ValueError: If the bytes don't match the canonical grammar.
    """
    if len(header_bytes) != OPT7VYT1_HEADER_LEN:
        raise ValueError(
            f"header_bytes must be exactly {OPT7VYT1_HEADER_LEN} bytes; "
            f"got {len(header_bytes)}"
        )
    (
        magic,
        version,
        alaska_color_branch_index,
        basis_strategy_index,
        chroma_strategy_index,
        pr110_base_sha256_prefix,
        reserved,
    ) = struct.unpack(OPT7VYT1_HEADER_FMT, header_bytes)
    if magic != ARCHIVE_MAGIC:
        raise ValueError(
            f"magic mismatch; expected {ARCHIVE_MAGIC!r} got {magic!r}"
        )
    return {
        "magic": magic,
        "version": version,
        "alaska_color_branch_index": alaska_color_branch_index,
        "basis_strategy_index": basis_strategy_index,
        "chroma_strategy_index": chroma_strategy_index,
        "pr110_base_sha256_prefix": bytes(pr110_base_sha256_prefix),
        "reserved": bytes(reserved),
    }
