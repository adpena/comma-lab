"""ddm_sz1 -- semantic-blob fp16 metadata byte-split: the transform and its exact inverse.

WHAT THIS IS
------------
The shipped archive's RX1M ``semantic`` section is a Brotli stream whose body is the
F12 stream-ordered WANS1 payload::

    offsets(30 B) || fixed metadata(8_284 B) || stream_area(27_726 B)   = 36_040 B

The fixed metadata is a concatenation of fp16 arrays -- 16 W4 scale arrays and 22 fp16
tensors, 4_142 little-endian ``<f2`` values.  Nothing entropy-codes it: it rides the
outer Brotli as raw interleaved fp16, so an exponent byte alternates with a mantissa
byte and the match distance is destroyed.  Grouping the two byte planes (all high bytes,
then all low bytes) before the container's Brotli is a pure SERIALIZATION change: the
decoded values are identical, only the on-disk layout moves.

This is the mechanism sv2's IX2TOK01 law (#859) predicts a win for -- the live coder is
paid for MATCH STRUCTURE, and the split turns a near-constant exponent plane into long
runs.  It is measured through the real coder at the container's own parameters, never
estimated, so it is not the entropy-surrogate class (#862).

THE RECEIVER
------------
``unsplit_region`` is the exact inverse permutation.  It is free receiver code under
rule 118: it reads no transmitted table, only a frozen offset/length pair, and it is a
single O(n) byte scatter.  It runs BEFORE any parsing, restoring the F12 body
byte-for-byte, so every downstream check (``WANS_BODY_BYTES``, ``decode_f12_wans_body``,
``decode_wans1``) sees exactly the bytes it sees today.

VERSIONING (the absent-tag-is-byte-identity pattern)
----------------------------------------------------
The RX1M 14-byte header already carries a ``reserved`` byte that both the encoder and
the receiver validate as strictly zero.  Bit 0 of that byte is the split flag.  Cost:
ZERO bytes -- the byte is already transmitted.  An archive with ``reserved == 0`` takes
the identical code path it takes today, so inactive is byte-identity by construction;
unknown bits still refuse, so the check stays fail-closed.

THE OFFSET IS A FROZEN CONSTANT, AND HERE IS WHAT IT IS AND IS NOT
------------------------------------------------------------------
Measured on the rr4 semantic section (shipped 34_763 B; control re-Brotli reproduces it
at delta +0), quality=11 lgwin=24:

    profile      (offset, length)   section B   delta
    DERIVED      (30, 8284)           34_265     -498     <- format-derived, 0 fitted params
    FX2_R5C      (41, 8284)           34_248     -515     <- what fx2's r5c actually measured
    TUNED        (49, 8284)           34_243     -520     <- argmax over offsets 0..400

``DERIVED`` takes its offset from the format itself (``_OFFSET_BYTES``) and its length
from ``_fixed_metadata_bytes()``: it splits exactly the metadata region and nothing else.
The other two straddle the region boundary by 11 and 19 bytes.  A joint (offset, length)
sweep does NOT beat (49, 8284), and the deltas swing +-20 B between adjacent offsets, so
**the ~22 B spread above is Brotli alignment noise, not mechanism.**  The mechanism is
worth -498 B; anything beyond that is noise fitted to this one frozen payload.

This also corrects the record.  fx2 reported -515 as "the fp16 metadata byte-split".
It is the same transform, but applied at offset 41 -- ``_HEADER_BYTES``, the CANONICAL
WANS1 header length -- to a body whose real prefix is ``_OFFSET_BYTES`` = 30.  The
region is misaligned against the metadata by 11 bytes.  The bytes are real and the
control is clean; the mechanism attribution was 17 B optimistic.

Every profile is equally correct and equally safe: the un-split is a pure byte
permutation, so a straddling region restores byte-exactly just as an aligned one does.

Provenance
----------
* fx2 receipt ``r5c_scale_split_f12.json`` sha256
  88aee37e349f73b01fedb14dacabc2c3ea21271436fbd5bf7d7e03852e0aa2d6
  ``{"shipped": 34763, "control_rebrotli": 34763, "byte_split": 34248, "delta": -515}``
* fx2 receipt ``r5b_main_relay_rows.json`` sha256
  84d51ab1f15e3a631d17e6a79ea019896862dd40e4ca1ee37f99eb568dcc5eb7
* base archive rr4 ``35ac2b9beb7e6fa81075c7d84b5247d8d24c056fe49ce1cbd22a334bc9618956`` (181_161 B)

Craft: docs/operating_manual_craft_handoff.md -- every number here is MEASURED through
the real coder on the real section, and the miss against fx2's figure is reported rather
than absorbed.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

# --- the F12 semantic body geometry (mirrors runtime/entropy/renderer_weight_codec) ---

WANS_BODY_BYTES = 36_040
"""Fixed length of the F12 stream-ordered semantic body.  The split preserves it."""

F12_OFFSET_TABLE_BYTES = 30
"""``_OFFSET_BYTES`` = 2 * (_W4_COUNT - 1) with _W4_COUNT = 16."""

F12_FIXED_METADATA_BYTES = 8_284
"""``_fixed_metadata_bytes()``: 16 W4 scale arrays + 22 fp16 tensors = 4_142 fp16 values."""

# --- the RX1M container flag (zero transmitted bytes) ---------------------------------

RX1_MODEL_HEADER = struct.Struct("<4sBBBBHHH")
RX1_MAGIC = b"RX1M"
RX1_RESERVED_SEMANTIC_SPLIT = 0x01
"""Bit 0 of the RX1M ``reserved`` byte: the semantic metadata region is byte-split."""

RX1_RESERVED_KNOWN_BITS = RX1_RESERVED_SEMANTIC_SPLIT
"""Any other bit set must still refuse -- the reserved check stays fail-closed."""


class SemanticSplitError(ValueError):
    """A semantic split region or RX1M reserved byte is malformed."""


@dataclass(frozen=True)
class SplitProfile:
    """A frozen (offset, length) region of the F12 body to byte-split."""

    name: str
    offset: int
    length: int
    rationale: str

    def validate(self, body_length: int = WANS_BODY_BYTES) -> None:
        if self.offset < 0 or self.length <= 0:
            raise SemanticSplitError(f"{self.name}: non-positive split region")
        if self.length % 2:
            raise SemanticSplitError(f"{self.name}: split length must be even")
        if self.offset + self.length > body_length:
            raise SemanticSplitError(f"{self.name}: split region overruns the body")


DERIVED = SplitProfile(
    name="derived",
    offset=F12_OFFSET_TABLE_BYTES,
    length=F12_FIXED_METADATA_BYTES,
    rationale=(
        "offset and length taken from the format itself (_OFFSET_BYTES and "
        "_fixed_metadata_bytes): splits exactly the fp16 metadata, zero fitted parameters"
    ),
)

FX2_R5C = SplitProfile(
    name="fx2_r5c",
    offset=41,
    length=F12_FIXED_METADATA_BYTES,
    rationale=(
        "reproduces fx2's r5c -515 B exactly; offset 41 is _HEADER_BYTES, the CANONICAL "
        "WANS1 header, applied to a body whose real prefix is 30 -- misaligned by 11 B"
    ),
)

TUNED = SplitProfile(
    name="tuned",
    offset=49,
    length=F12_FIXED_METADATA_BYTES,
    rationale=(
        "argmax over offsets 0..400; a joint (offset,length) sweep does not beat it. "
        "The +22 B over DERIVED is Brotli alignment noise fitted to this frozen payload, "
        "NOT mechanism -- adjacent offsets swing +-20 B"
    ),
)

PROFILES = {profile.name: profile for profile in (DERIVED, FX2_R5C, TUNED)}

SHIPPED_PROFILE = DERIVED
"""Frozen for the shipping candidate.

A decoder takes no arguments.  Anything configurable at encode time but not at decode
time is a desynchronisation waiting to happen (fx2's own lesson), so exactly one profile
is frozen here and the receiver patch is generated from this same constant.
"""


# --- the transform and its exact inverse ---------------------------------------------


def split_region(body: bytes, profile: SplitProfile = SHIPPED_PROFILE) -> bytes:
    """Group the region's byte planes: all high bytes, then all low bytes.

    The region is read as little-endian ``<u2``, so the high plane is the odd-indexed
    (sign/exponent) bytes and the low plane is the even-indexed (mantissa) bytes.
    Length is preserved exactly, so ``WANS_BODY_BYTES`` still holds.
    """
    profile.validate(len(body))
    start, end = profile.offset, profile.offset + profile.length
    region = np.frombuffer(body[start:end], dtype=np.uint8)
    planes = np.empty(profile.length, dtype=np.uint8)
    half = profile.length // 2
    planes[:half] = region[1::2]
    planes[half:] = region[0::2]
    return body[:start] + planes.tobytes() + body[end:]


def unsplit_region(body: bytes, profile: SplitProfile = SHIPPED_PROFILE) -> bytes:
    """Exact inverse of :func:`split_region` -- the free receiver-side un-split.

    Pure byte scatter: no transmitted table, no schema knowledge, O(n).  Restores the
    F12 body byte-for-byte, so every downstream parse sees today's bytes.
    """
    profile.validate(len(body))
    start, end = profile.offset, profile.offset + profile.length
    planes = np.frombuffer(body[start:end], dtype=np.uint8)
    half = profile.length // 2
    region = np.empty(profile.length, dtype=np.uint8)
    region[1::2] = planes[:half]
    region[0::2] = planes[half:]
    return body[:start] + region.tobytes() + body[end:]


# --- the RX1M reserved-byte flag ------------------------------------------------------


def read_rx1_header(model: bytes) -> tuple[int, int, int, int, int, int]:
    """Return ``(version, codec, table_mode, reserved, ...section lengths)``."""
    if len(model) < RX1_MODEL_HEADER.size or not model.startswith(RX1_MAGIC):
        raise SemanticSplitError("not an RX1M model container")
    magic, version, codec, table_mode, reserved, hpac, semantic, carrier = (
        RX1_MODEL_HEADER.unpack_from(model)
    )
    if magic != RX1_MAGIC:
        raise SemanticSplitError("not an RX1M model container")
    return version, codec, table_mode, reserved, hpac, semantic, carrier


def semantic_split_active(reserved: int) -> bool:
    """Decode the reserved byte, refusing unknown bits (fail-closed, as today)."""
    if reserved & ~RX1_RESERVED_KNOWN_BITS:
        raise SemanticSplitError("RX1 reserved byte carries unknown bits")
    return bool(reserved & RX1_RESERVED_SEMANTIC_SPLIT)


RX1_RESERVED_BYTE_INDEX = 7
"""Layout ``<4sBBBBHHH``: magic[0:4], version[4], codec[5], table_mode[6], reserved[7].

Writing index 8 instead would silently clobber the low byte of ``hpac_bytes`` -- a
section length -- which is why ``set_rx1_reserved`` re-reads the header afterwards and
asserts nothing but the reserved byte moved.
"""


def set_rx1_reserved(model: bytes, reserved: int) -> bytes:
    """Rewrite the RX1M reserved byte in place, leaving every other byte untouched."""
    if reserved & ~RX1_RESERVED_KNOWN_BITS:
        raise SemanticSplitError("refusing to set unknown RX1 reserved bits")
    before = read_rx1_header(model)
    index = RX1_RESERVED_BYTE_INDEX
    updated = model[:index] + bytes((reserved,)) + model[index + 1 :]
    after = read_rx1_header(updated)
    if after[3] != reserved or after[:3] != before[:3] or after[4:] != before[4:]:
        raise SemanticSplitError("reserved-byte rewrite disturbed another header field")
    return updated


# --- the receiver patch ---------------------------------------------------------------

HEADER_ANCHOR = (
    "    if table_mode not in (0, 1) or reserved != 0 "
    "or min(hpac_bytes, semantic_bytes, carrier_bytes) <= 0:\n"
)

HEADER_REPLACEMENT = (
    "    # DDM_SZ1_SEMANTIC_METADATA_SPLIT_V1: reserved bit 0 is the semantic split\n"
    "    # flag.  Unknown bits still refuse, so the check stays fail-closed and an\n"
    "    # archive with reserved == 0 takes exactly the path it takes today.\n"
    "    if table_mode not in (0, 1) or (reserved & ~SZ1_RESERVED_KNOWN_BITS) "
    "or min(hpac_bytes, semantic_bytes, carrier_bytes) <= 0:\n"
)

RECEIVER_ANCHOR = """    tagged_semantic = semantic_body.startswith((b"SD1M", b"SM3R"))
    if tagged_semantic:
        semantic = semantic_body
    else:
        if len(semantic_body) != WANS_BODY_BYTES:
            raise ResidualArchiveError("RX1 semantic section length differs")
"""

RECEIVER_REPLACEMENT = """    # DDM_SZ1_SEMANTIC_METADATA_SPLIT_V1
    # reserved bit 0: the fp16 metadata region is byte-split (high plane, then low
    # plane).  Absent (reserved == 0) is byte-identity: the code path below is
    # unchanged.  The un-split is an exact byte permutation over a frozen region and
    # restores the F12 body byte-for-byte before any parsing, so WANS_BODY_BYTES and
    # every downstream check see exactly the bytes they see today.
    if reserved & SZ1_RESERVED_SEMANTIC_SPLIT:
        semantic_body = _sz1_unsplit_semantic(semantic_body)
    tagged_semantic = semantic_body.startswith((b"SD1M", b"SM3R"))
    if tagged_semantic:
        semantic = semantic_body
    else:
        if len(semantic_body) != WANS_BODY_BYTES:
            raise ResidualArchiveError("RX1 semantic section length differs")
"""

RECEIVER_HELPER = '''

# DDM_SZ1_SEMANTIC_METADATA_SPLIT_V1 -- free receiver code, zero transmitted bytes.
SZ1_RESERVED_SEMANTIC_SPLIT = {flag:#04x}
SZ1_RESERVED_KNOWN_BITS = {known:#04x}
SZ1_SPLIT_OFFSET = {offset}
SZ1_SPLIT_LENGTH = {length}


def _sz1_unsplit_semantic(body: bytes) -> bytes:
    """Restore interleaved fp16 metadata from its two byte planes (exact inverse)."""
    start = SZ1_SPLIT_OFFSET
    end = start + SZ1_SPLIT_LENGTH
    if len(body) < end:
        raise ResidualArchiveError("RX1 semantic body too short for the metadata split")
    planes = np.frombuffer(body[start:end], dtype=np.uint8)
    half = SZ1_SPLIT_LENGTH // 2
    region = np.empty(SZ1_SPLIT_LENGTH, dtype=np.uint8)
    region[1::2] = planes[:half]
    region[0::2] = planes[half:]
    return body[:start] + region.tobytes() + body[end:]
'''


def receiver_helper_source(profile: SplitProfile = SHIPPED_PROFILE) -> str:
    """Render the receiver helper with the frozen profile baked in."""
    profile.validate()
    return RECEIVER_HELPER.format(
        flag=RX1_RESERVED_SEMANTIC_SPLIT,
        known=RX1_RESERVED_KNOWN_BITS,
        offset=profile.offset,
        length=profile.length,
    )


def patch_receiver_source(source: str, profile: SplitProfile = SHIPPED_PROFILE) -> str:
    """Insert the un-split into a ``runtime/residual_archive.py`` source string.

    Fails closed if the anchor is absent or ambiguous: a receiver patch that silently
    does nothing is the inert-flag fake, and it would ship an archive no decoder reads.
    """
    if "DDM_SZ1_SEMANTIC_METADATA_SPLIT_V1" in source:
        raise SemanticSplitError("receiver already carries the sz1 split patch")
    for label, anchor in (("header", HEADER_ANCHOR), ("semantic", RECEIVER_ANCHOR)):
        found = source.count(anchor)
        if found != 1:
            raise SemanticSplitError(
                f"{label} anchor found {found} times, expected exactly 1"
            )
    patched = source.replace(HEADER_ANCHOR, HEADER_REPLACEMENT, 1)
    patched = patched.replace(RECEIVER_ANCHOR, RECEIVER_REPLACEMENT, 1)
    return patched + receiver_helper_source(profile)
