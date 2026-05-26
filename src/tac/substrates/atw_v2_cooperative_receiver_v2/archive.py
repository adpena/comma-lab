# SPDX-License-Identifier: MIT
"""ATW V2 cooperative-receiver V2 — archive grammar (ATWv2CR2 magic).

Per Phase 3 design memo §1 + §7 + Catalog #124 archive grammar 8 fields
declared inline + Catalog #146 contest-compliant runtime template.

NEW grammar magic ``ATWv2CR2`` (cooperative-receiver V2) distinguishes this
substrate's archive from sister ATW1 / ATW2 (v1) / Z3HP1 / Z4CR1.

Bug class learning per Phase 1 audit CC-8 EMPIRICALLY FALSIFIED
================================================================

ATW V2 v1 cdf_table_blob: codex byte-mutation smoke commit ``057130de4``
proved ``max_abs_raw_byte_delta=0`` across all 2,560 bytes mutated. The
section was parsed + stored but never consumed at decode time → DEAD BYTES.
Canonical equation #26 EXCLUDED context ``direct_byte_substitution_on_decode_
opaque_raw_sections`` registered per Catalog #344 to structurally extinct
the bug class.

THIS substrate grammar (Phase 3) honors the lesson:

- **NO dead sections.** Every section listed in ``PARSER_SECTION_ROLES`` must
  pass Catalog #139/#272 byte-mutation smoke (NON-ZERO max_abs_raw_byte_delta).
- **REMOVED cdf_table_blob** (the v1 dead-section anchor).
- **NEW conditioning surface**: ``ego_motion_proj_blob`` (per-pair ego-motion
  FOE projection coefficients) + ``cond_embed_blob`` (conditioning embedding
  layer weights). Both ARE consumed by inflate.

8 sections (ALL byte-mutation verifiable from byte-zero)
=========================================================

1. ``header``: 16-byte fixed header with magic + version + section offsets
2. ``encoder_blob``: encoder weights (Conv2d stem + Linear projections)
3. ``decoder_blob``: HNeRV-style decoder weights (initial_proj + per-block
   Conv2d + final Conv2d)
4. ``cond_embed_blob``: conditioning embedding head weights (2-layer MLP:
   ego_motion_dim → cond_embed_dim → latent_dim)
5. ``ego_motion_proj_blob``: per-pair ego-motion FOE projection precomputed
   table ``(num_pairs, 6)`` fp16 — consumed by inflate to derive per-pair
   conditioning embedding
6. ``per_pair_latent_blob``: per-pair latent residuals ``(num_pairs, latent_dim)``
   fp16 — consumed by inflate as the encoded substrate content
7. ``class_cond_cdf_blob``: per-pair class-conditional CDF table (placeholder
   for Phase 4 range-coder integration; reserved 0 bytes at L0 scaffold to
   preserve byte-mutation discipline — Phase 4 lands when range coder is wired
   AND byte-mutation smoke confirms decode influence)
8. ``meta_blob``: substrate metadata (substrate_id, version, num_pairs,
   latent_dim, ego_motion_dim, encoder/decoder shape signatures)

Phase 3 L0 SCAFFOLD scope
=========================

The L0 SCAFFOLD provides:
- Magic constants + section role definitions
- pack_archive(...) + parse_archive(...) functions
- Byte-stable serialization for byte-mutation smoke tests

What's deferred to Phase 4:
- Brotli compression of encoder/decoder/conditioning blobs (currently raw bytes)
- Range coder for class_cond_cdf_blob (currently 0 bytes / placeholder)
- Per-pair quantization sweep (currently fp16 raw)

Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY":
``DISPATCH_ENABLED = False``, ``research_only=True``. The L0 SCAFFOLD's
operational mechanism (per-pair latent + ego-motion conditioning consumed at
inflate) IS WIRED + verifiable; Phase 4 council approval required to lift
``_full_main NotImplementedError`` per Catalog #240(c).

Cross-references
----------------

* Phase 1 audit CC-8: cdf_table_blob FALSIFIED → REMOVED from this grammar
* Phase 2 §2.3 CC-8 unwind: NEW grammar magic; NO dead sections
* Phase 3 §1 + §7 (canonical-vs-unique per layer; substrate-specific magic)
* Catalog #139/#272 byte-mutation smoke discipline
* Catalog #220 substrate L1+ operational mechanism declaration
* Catalog #146 contest-compliant inflate runtime template
"""

from __future__ import annotations

import io
import struct
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# Canonical magic + version constants
ATWV2CR2_MAGIC: bytes = b"ATWv2CR2"
"""8-byte magic identifying ATW V2 cooperative-receiver V2 substrate archives."""

ATWV2CR2_SCHEMA_VERSION: int = 1
"""Schema version (bump on grammar-breaking changes)."""

ATWV2CR2_HEADER_FMT: str = "<8sIQ"
"""Header format: 8-byte magic + uint32 schema_version + uint64 num_sections.

Followed by N x (uint64 section_offset + uint64 section_length) pairs in the
section table; total header size = 8 + 4 + 8 + (16 * num_sections) bytes.
"""

ATWV2CR2_HEADER_FIXED_SIZE: int = struct.calcsize(ATWV2CR2_HEADER_FMT)

NUM_SECTIONS: int = 8

PARSER_SECTION_ROLES: tuple[str, ...] = (
    "encoder_blob",
    "decoder_blob",
    "cond_embed_blob",
    "ego_motion_proj_blob",
    "per_pair_latent_blob",
    "class_cond_cdf_blob",
    "meta_blob",
    "reserved_section_for_phase_4_extension",
)
"""8 sections per Phase 3 design memo §1 + §7."""

assert len(PARSER_SECTION_ROLES) == NUM_SECTIONS, "PARSER_SECTION_ROLES count mismatch"


@dataclass
class ATWv2CR2Archive:
    """In-memory representation of an ATWv2CR2 archive.

    Each section is bytes; serialization preserves byte-stability for
    Catalog #139/#272 byte-mutation smoke tests.
    """

    encoder_blob: bytes = b""
    decoder_blob: bytes = b""
    cond_embed_blob: bytes = b""
    ego_motion_proj_blob: bytes = b""
    per_pair_latent_blob: bytes = b""
    class_cond_cdf_blob: bytes = b""
    meta_blob: bytes = b""
    reserved_section_for_phase_4_extension: bytes = b""

    def section_iterator(self) -> list[tuple[str, bytes]]:
        """Iterate sections in canonical order (matches PARSER_SECTION_ROLES)."""
        return [
            ("encoder_blob", self.encoder_blob),
            ("decoder_blob", self.decoder_blob),
            ("cond_embed_blob", self.cond_embed_blob),
            ("ego_motion_proj_blob", self.ego_motion_proj_blob),
            ("per_pair_latent_blob", self.per_pair_latent_blob),
            ("class_cond_cdf_blob", self.class_cond_cdf_blob),
            ("meta_blob", self.meta_blob),
            ("reserved_section_for_phase_4_extension", self.reserved_section_for_phase_4_extension),
        ]

    def total_archive_bytes(self) -> int:
        """Sum of section lengths + header overhead."""
        section_overhead = ATWV2CR2_HEADER_FIXED_SIZE + 16 * NUM_SECTIONS
        section_bytes = sum(len(b) for _, b in self.section_iterator())
        return section_overhead + section_bytes


def pack_archive(archive: ATWv2CR2Archive) -> bytes:
    """Serialize an ATWv2CR2 archive to bytes.

    Byte-stable + deterministic; sister to ``parse_archive``. The serialized
    output is suitable for direct write to ``0.bin`` in the contest archive ZIP.
    """
    sections = archive.section_iterator()
    if len(sections) != NUM_SECTIONS:
        raise ValueError(f"expected {NUM_SECTIONS} sections; got {len(sections)}")

    # Compute section offsets: header (fixed + 16 * N section table) is at byte 0,
    # then sections placed back-to-back.
    section_table_size = 16 * NUM_SECTIONS
    section_data_start = ATWV2CR2_HEADER_FIXED_SIZE + section_table_size
    section_table = []
    offset = section_data_start
    for _, blob in sections:
        section_table.append((offset, len(blob)))
        offset += len(blob)

    buf = io.BytesIO()
    # Header
    buf.write(
        struct.pack(
            ATWV2CR2_HEADER_FMT,
            ATWV2CR2_MAGIC,
            ATWV2CR2_SCHEMA_VERSION,
            NUM_SECTIONS,
        )
    )
    # Section table
    for off, length in section_table:
        buf.write(struct.pack("<QQ", off, length))
    # Section data
    for _, blob in sections:
        buf.write(blob)

    return buf.getvalue()


def parse_archive(archive_bytes: bytes) -> ATWv2CR2Archive:
    """Parse ATWv2CR2 archive bytes into an in-memory ATWv2CR2Archive.

    Validates magic + schema_version + section_count; raises ValueError on
    mismatch. Sister to ``pack_archive``; round-trip is byte-stable.
    """
    if len(archive_bytes) < ATWV2CR2_HEADER_FIXED_SIZE:
        raise ValueError(
            f"archive too short: {len(archive_bytes)} bytes < header fixed size "
            f"{ATWV2CR2_HEADER_FIXED_SIZE}"
        )

    magic, schema_version, num_sections = struct.unpack(
        ATWV2CR2_HEADER_FMT, archive_bytes[:ATWV2CR2_HEADER_FIXED_SIZE]
    )

    if magic != ATWV2CR2_MAGIC:
        raise ValueError(
            f"magic mismatch: got {magic!r}, expected {ATWV2CR2_MAGIC!r}"
        )
    if schema_version != ATWV2CR2_SCHEMA_VERSION:
        raise ValueError(
            f"schema_version mismatch: got {schema_version}, expected "
            f"{ATWV2CR2_SCHEMA_VERSION}"
        )
    if num_sections != NUM_SECTIONS:
        raise ValueError(
            f"num_sections mismatch: got {num_sections}, expected {NUM_SECTIONS}"
        )

    # Parse section table
    table_start = ATWV2CR2_HEADER_FIXED_SIZE
    sections: dict[str, bytes] = {}
    for i, role in enumerate(PARSER_SECTION_ROLES):
        entry_start = table_start + i * 16
        offset, length = struct.unpack("<QQ", archive_bytes[entry_start : entry_start + 16])
        if offset + length > len(archive_bytes):
            raise ValueError(
                f"section {role} out of bounds: offset={offset} length={length} "
                f"archive_len={len(archive_bytes)}"
            )
        sections[role] = archive_bytes[offset : offset + length]

    return ATWv2CR2Archive(
        encoder_blob=sections["encoder_blob"],
        decoder_blob=sections["decoder_blob"],
        cond_embed_blob=sections["cond_embed_blob"],
        ego_motion_proj_blob=sections["ego_motion_proj_blob"],
        per_pair_latent_blob=sections["per_pair_latent_blob"],
        class_cond_cdf_blob=sections["class_cond_cdf_blob"],
        meta_blob=sections["meta_blob"],
        reserved_section_for_phase_4_extension=sections["reserved_section_for_phase_4_extension"],
    )


def build_smoke_archive(num_pairs: int = 600, latent_dim: int = 32) -> ATWv2CR2Archive:
    """Build a smoke-test archive with placeholder bytes in each section.

    Used in test_basic.py for byte-mutation tests + roundtrip tests. NOT a
    production archive; just a deterministic fixture.
    """
    # Per-pair latent residuals: (num_pairs, latent_dim) fp16
    latents = np.zeros((num_pairs, latent_dim), dtype=np.float16)
    latents_bytes = latents.tobytes()

    # Per-pair ego-motion FOE projection: (num_pairs, 6) fp16
    ego_motion_proj = np.zeros((num_pairs, 6), dtype=np.float16)
    ego_motion_proj_bytes = ego_motion_proj.tobytes()

    # Encoder + decoder + cond_embed + meta: small placeholder bytes
    # (Production Phase 4 fills these with actual trained weight bytes via brotli)
    return ATWv2CR2Archive(
        encoder_blob=b"\x00" * 32,
        decoder_blob=b"\x00" * 64,
        cond_embed_blob=b"\x00" * 16,
        ego_motion_proj_blob=ego_motion_proj_bytes,
        per_pair_latent_blob=latents_bytes,
        class_cond_cdf_blob=b"",  # Phase 4 fills with range-coder CDF table
        meta_blob=b'{"substrate_id":"atw_v2_cooperative_receiver_v2","schema_version":1}',
        reserved_section_for_phase_4_extension=b"",
    )


__all__ = [
    "ATWV2CR2_HEADER_FIXED_SIZE",
    "ATWV2CR2_HEADER_FMT",
    "ATWV2CR2_MAGIC",
    "ATWV2CR2_SCHEMA_VERSION",
    "ATWv2CR2Archive",
    "NUM_SECTIONS",
    "PARSER_SECTION_ROLES",
    "build_smoke_archive",
    "pack_archive",
    "parse_archive",
]
