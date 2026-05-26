# SPDX-License-Identifier: MIT
"""BPR1 sidecar archive grammar for boost_nerv_pr110_residual (L0 SCAFFOLD).

Per design memo §"Canonical-vs-unique decision per layer" → archive grammar
FORK because the substrate must STRUCTURALLY BIND the sidecar to a specific
PR110 base archive via SHA256 prefix. Composed archive format:

    [BPR1 header 28 bytes]
      magic[5]   = b"BPR1\\x00"
      version[1] = 1
      num_rounds[1]
      align[1]   = 0 (4-byte alignment padding)
      sha_prefix[16] = sha256(pr110_base_archive_bytes)[:16]
      residual_blob_len[4] = u32 little-endian
      reserved_tail[1] = 0
    [residual_blob_len bytes] brotli-quality9 compressed int8 residual
        learner state_dict + per-pair latent z_pr110 reference
    [PR110_BASE_ARCHIVE_BYTES inline]

The composed archive is contest-loadable as a single ZIP file containing
one member `x` of size (28 + residual_blob_len + len(pr110_base_bytes)).
The inflate runtime knows to:
    1. Read the first 28 bytes; verify BPR1 magic + version.
    2. Read residual_blob_len bytes; brotli-decompress; load residual state_dict.
    3. Slice the rest as PR110_BASE_ARCHIVE_BYTES; write to temp file; invoke
       PR110's inflate.sh subprocess.
    4. Load PR110-produced frames; run residual learner forward per pair;
       compose; write final frames.

Per CLAUDE.md "Bit-level deconstruction and entropy discipline" + Catalog
#139 byte-mutation discipline: the BPR1 magic + sha_prefix binding is the
structural-extinction primitive that prevents the sidecar from being
silently mis-applied to a non-PR110 base archive.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path

from . import ARCHIVE_MAGIC, ARCHIVE_VERSION, BPR1_HEADER_FMT, BPR1_HEADER_LEN


@dataclass(frozen=True)
class Bpr1Header:
    """Parsed BPR1 sidecar header."""

    magic: bytes
    version: int
    num_boosting_rounds: int
    pr110_sha256_prefix: bytes  # 16 bytes
    residual_blob_len: int      # u32

    def __post_init__(self) -> None:
        if self.magic != ARCHIVE_MAGIC:
            raise ValueError(
                f"BPR1 magic mismatch: got {self.magic!r}, expected {ARCHIVE_MAGIC!r}; "
                f"sidecar is not a valid boost_nerv_pr110_residual archive"
            )
        if self.version != ARCHIVE_VERSION:
            raise ValueError(
                f"BPR1 version mismatch: got {self.version}, expected {ARCHIVE_VERSION}"
            )
        if self.num_boosting_rounds < 1 or self.num_boosting_rounds > 4:
            raise ValueError(
                f"num_boosting_rounds out of canonical range [1, 4]: {self.num_boosting_rounds}"
            )
        if len(self.pr110_sha256_prefix) != 16:
            raise ValueError(
                f"pr110_sha256_prefix must be 16 bytes; got {len(self.pr110_sha256_prefix)}"
            )
        if self.residual_blob_len < 0:
            raise ValueError(
                f"residual_blob_len must be non-negative; got {self.residual_blob_len}"
            )


def pack_bpr1_header(
    num_boosting_rounds: int,
    pr110_sha256_prefix: bytes,
    residual_blob_len: int,
) -> bytes:
    """Canonical BPR1 header serialization (28 bytes, byte-stable)."""
    if len(pr110_sha256_prefix) != 16:
        raise ValueError(
            f"pr110_sha256_prefix must be exactly 16 bytes; got {len(pr110_sha256_prefix)}"
        )
    if num_boosting_rounds < 1 or num_boosting_rounds > 4:
        raise ValueError(
            f"num_boosting_rounds must be in [1, 4]; got {num_boosting_rounds}"
        )
    if residual_blob_len < 0 or residual_blob_len > 0xFFFFFFFF:
        raise ValueError(
            f"residual_blob_len must fit in u32; got {residual_blob_len}"
        )
    return struct.pack(
        BPR1_HEADER_FMT,
        ARCHIVE_MAGIC,
        ARCHIVE_VERSION,
        num_boosting_rounds,
        0,  # alignment padding
        pr110_sha256_prefix,
        residual_blob_len,
        0,  # reserved tail
    )


def parse_bpr1_header(buf: bytes) -> Bpr1Header:
    """Parse + validate BPR1 header from buffer (raises if invalid)."""
    if len(buf) < BPR1_HEADER_LEN:
        raise ValueError(
            f"BPR1 header buffer too short: got {len(buf)} bytes, need {BPR1_HEADER_LEN}"
        )
    (
        magic,
        version,
        num_rounds,
        _align,
        sha_prefix,
        blob_len,
        _reserved,
    ) = struct.unpack(BPR1_HEADER_FMT, buf[:BPR1_HEADER_LEN])
    return Bpr1Header(
        magic=magic,
        version=version,
        num_boosting_rounds=num_rounds,
        pr110_sha256_prefix=sha_prefix,
        residual_blob_len=blob_len,
    )


def compose_archive(
    pr110_base_archive_bytes: bytes,
    residual_blob: bytes,
    num_boosting_rounds: int,
) -> bytes:
    """Build composed archive: BPR1 header + residual blob + PR110 base bytes.

    The composed bytes are written into a ZIP member `x` by the canonical
    archive packaging path (caller's responsibility, mirroring PR110's
    monolithic-single-file pattern).

    Per Catalog #139 byte-mutation discipline: the SHA256 prefix binding
    in the header structurally prevents mis-application to a non-PR110 base.
    """
    sha_prefix = hashlib.sha256(pr110_base_archive_bytes).digest()[:16]
    header = pack_bpr1_header(
        num_boosting_rounds=num_boosting_rounds,
        pr110_sha256_prefix=sha_prefix,
        residual_blob_len=len(residual_blob),
    )
    return header + residual_blob + pr110_base_archive_bytes


def split_composed_archive(composed_bytes: bytes) -> tuple[Bpr1Header, bytes, bytes]:
    """Inverse of compose_archive: split into (header, residual_blob, pr110_base_bytes).

    Verifies SHA256 binding at inflate time (refuses if sidecar's
    pr110_sha256_prefix does not match the trailing pr110_base_bytes'
    actual sha256[:16]). This is the runtime closure proof per HNeRV
    parity L9.
    """
    header = parse_bpr1_header(composed_bytes)
    blob_start = BPR1_HEADER_LEN
    blob_end = blob_start + header.residual_blob_len
    if blob_end > len(composed_bytes):
        raise ValueError(
            f"composed_bytes too short: header claims residual_blob_len={header.residual_blob_len} "
            f"but only {len(composed_bytes) - blob_start} bytes remain after header"
        )
    residual_blob = composed_bytes[blob_start:blob_end]
    pr110_base_bytes = composed_bytes[blob_end:]
    # Runtime closure proof: verify sidecar's sha_prefix matches actual base bytes.
    actual_prefix = hashlib.sha256(pr110_base_bytes).digest()[:16]
    if actual_prefix != header.pr110_sha256_prefix:
        raise ValueError(
            f"PR110 base archive sha256 prefix mismatch: header says "
            f"{header.pr110_sha256_prefix.hex()}, actual is {actual_prefix.hex()}; "
            f"sidecar is bound to a different PR110 base archive (refusing per Catalog #139 "
            f"byte-mutation discipline). This is the structural-extinction primitive that "
            f"prevents mis-application to a non-PR110 base."
        )
    return header, residual_blob, pr110_base_bytes


def write_composed_archive_to_zip(
    composed_bytes: bytes,
    output_zip_path: Path,
) -> None:
    """Write composed archive to ZIP file with deterministic-bytes guarantee.

    Per Catalog #19 deterministic ZIP discipline: use ZipInfo + writestr
    with fixed timestamp (1980-01-01) to ensure byte-stable output.
    Mirrors PR110's archive packaging.
    """
    import zipfile

    if not isinstance(composed_bytes, (bytes, bytearray)):
        raise TypeError(f"composed_bytes must be bytes; got {type(composed_bytes)}")

    output_zip_path.parent.mkdir(parents=True, exist_ok=True)
    info = zipfile.ZipInfo("x", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED  # Mirror PR110's STORED member
    with zipfile.ZipFile(output_zip_path, "w") as zf:
        zf.writestr(info, bytes(composed_bytes))
