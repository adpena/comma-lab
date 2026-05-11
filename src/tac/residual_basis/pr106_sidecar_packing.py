"""Shared byte-grammar for byte-closed PR106 + non-HNeRV residual sidecar archives.

This module is the *common substrate* for the five per-family residual sidecar
archive builders introduced 2026-05-11 as the operator-directive
"$100+ candidates ready to dispatch in parallel" pre-stage layer:

* ``tools/materialize_wavelet_residual_pr106_sidecar.py``
* ``tools/materialize_cool_chic_residual_pr106_sidecar.py``
* ``tools/materialize_c3_residual_pr106_sidecar.py``
* ``tools/materialize_siren_residual_pr106_sidecar.py``
* ``tools/materialize_coord_mlp_residual_pr106_sidecar.py``

and the matching inflate runtimes at
``submissions/pr106_<family>_residual_sidecar/{inflate.py, inflate.sh}``.

Each family writes a single-file monolithic ``0.bin`` archive whose first two
bytes carry a magic + family ``format_id``, then a length-prefixed PR106 r2
payload followed by a length-prefixed family-specific residual blob.

Wire format
-----------

::

    magic(1B)         = 0xFD                 # 'PR106 + non-HNeRV residual'
    format_id(1B)     = per-family (see PR106_RESIDUAL_FORMAT_IDS)
    pr106_len(4B LE)  = uint32 byte length of the PR106 payload
    pr106_bytes       = pr106_len bytes (verbatim PR106 0.bin)
    residual_len(4B LE) = uint32 byte length of the residual blob
    residual_bytes    = residual_len bytes (family-specific payload, optionally
                                            brotli-compressed)

The 4B residual length matches the wavelet / SIREN / coord-MLP byte budgets
which can exceed the 2-byte cap PR101's sidecar uses (those are per-pair
nudges, ≤32KB; here we expect 4–20KB residual blobs).

Sister of ``submissions/pr106_latent_sidecar/inflate.py`` (PR100-style 2-byte
per-pair sidecar). The 0xFD magic intentionally differs from 0xFE so a stale
PR100 sidecar inflate cannot accidentally consume one of these archives.

Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline":

* archive_grammar (lesson 3): single-file monolithic 0.bin; offsets/lengths
  declared by this module's parse/build pair.
* parser_section_manifest (lesson 3): see ``parse_archive`` return type plus
  ``ParsedResidualArchive`` dataclass.
* inflate_runtime_loc_budget (lesson 4): each per-family inflate.py uses
  ``parse_archive`` from this module + ≤80 LOC of family-specific decode.
* runtime_dep_closure (lesson 3): numpy + optional brotli (already in the
  PR106 latent-sidecar inflate runtime closure).
* no_op_detector_planned (lesson 11): mutating a residual byte must change
  the inflate.py output deterministically (the materializer test suite pins
  this via byte-mutation parity).

No score claim. No promotion. No exact eval dispatch from this module.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Final

# ── Magic + per-family format_id byte registry ──────────────────────────────

# Differs from 0xFE used by submissions/pr106_latent_sidecar/inflate.py.
PR106_RESIDUAL_MAGIC: Final[int] = 0xFD

# Per-family format_id bytes. Each family owns TWO bytes: 0x1N for dense, 0x2N
# for sparse PacketIR (RLE/AC/temporal-subsampled). Sister inflate.py scripts
# accept both and dispatch on format_id.
PR106_RESIDUAL_FORMAT_IDS: Final[dict[str, int]] = {
    "wavelet": 0x10,
    "cool_chic": 0x11,
    "c3": 0x12,
    "siren": 0x13,
    "coord_mlp": 0x14,
    # Sparse PacketIR variants (closes O's L2 wire-format ceiling).
    "wavelet_sparse": 0x20,
    "cool_chic_sparse": 0x21,
    "c3_sparse": 0x22,
    "siren_sparse": 0x23,
    "coord_mlp_sparse": 0x24,
}

PR106_RESIDUAL_FORMAT_NAMES: Final[dict[int, str]] = {
    v: k for k, v in PR106_RESIDUAL_FORMAT_IDS.items()
}

# Wire-format prefix bytes (magic + format_id + 4B pr106_len).
_HEADER_FIXED_BYTES: Final[int] = 1 + 1 + 4
# Residual length prefix bytes (4B LE uint32).
_RESIDUAL_LEN_BYTES: Final[int] = 4

# 4B residual length budget caps the residual blob at 4 GiB. Real PR106
# residual sidecars are 1–20KB.
_RESIDUAL_LEN_MAX: Final[int] = (1 << 32) - 1


class ResidualArchiveError(ValueError):
    """Raised on wire-format contract violations."""


@dataclass(frozen=True)
class ParsedResidualArchive:
    """Typed result of ``parse_archive()``.

    The ``family`` field is None when the format_id byte is unknown to this
    module — callers that want to refuse unknown families should check it
    explicitly (each per-family inflate.py does so).
    """

    magic: int
    format_id: int
    family: str | None
    pr106_bytes: bytes
    residual_bytes: bytes
    total_bytes: int

    def assert_invariants(self) -> None:
        if self.magic != PR106_RESIDUAL_MAGIC:
            raise ResidualArchiveError(
                f"magic mismatch: got 0x{self.magic:02X} expected 0x{PR106_RESIDUAL_MAGIC:02X}"
            )
        if self.format_id not in PR106_RESIDUAL_FORMAT_NAMES:
            raise ResidualArchiveError(
                f"format_id 0x{self.format_id:02X} not in registered family set "
                f"{sorted(PR106_RESIDUAL_FORMAT_NAMES)}"
            )
        if not self.pr106_bytes:
            raise ResidualArchiveError("pr106_bytes is empty")
        if self.total_bytes != (
            _HEADER_FIXED_BYTES
            + len(self.pr106_bytes)
            + _RESIDUAL_LEN_BYTES
            + len(self.residual_bytes)
        ):
            raise ResidualArchiveError(
                f"total_bytes={self.total_bytes} mismatches header+payload+residual sum"
            )


@dataclass(frozen=True)
class BuildResidualArchiveResult:
    """Typed result of ``build_archive()``."""

    archive_bytes: bytes
    pr106_len: int
    residual_len: int
    family: str
    format_id: int
    score_claim: bool = field(default=False, init=False)
    promotion_eligible: bool = field(default=False, init=False)
    ready_for_exact_eval_dispatch: bool = field(default=False, init=False)
    evidence_grade: str = field(default="research_signal", init=False)


def _ensure_family(family: str) -> int:
    if family not in PR106_RESIDUAL_FORMAT_IDS:
        raise ResidualArchiveError(
            f"unknown family {family!r}; valid: {sorted(PR106_RESIDUAL_FORMAT_IDS)}"
        )
    return PR106_RESIDUAL_FORMAT_IDS[family]


def build_archive(
    *,
    family: str,
    pr106_bytes: bytes,
    residual_bytes: bytes,
) -> BuildResidualArchiveResult:
    """Build a byte-closed PR106 + non-HNeRV residual archive blob.

    Parameters
    ----------
    family
        One of the registered families in ``PR106_RESIDUAL_FORMAT_IDS``
        (wavelet / cool_chic / c3 / siren / coord_mlp).
    pr106_bytes
        Verbatim PR106 0.bin payload (typed as bytes; the materializer reads
        the canonical PR106 r2 archive via ``zipfile``).
    residual_bytes
        Family-specific residual blob. Encoding is the family's choice
        (typically brotli- or LZMA-compressed via PR101/PR103 primitives).

    Returns
    -------
    BuildResidualArchiveResult
        Typed result with the assembled ``archive_bytes``; the
        ``ready_for_exact_eval_dispatch``/``score_claim``/``promotion_eligible``
        fields are pinned False per HNeRV parity discipline (Catalog #100
        ``check_gate2_no_naked_bytes``).
    """

    format_id = _ensure_family(family)
    if not pr106_bytes:
        raise ResidualArchiveError("pr106_bytes is empty")
    if not isinstance(pr106_bytes, (bytes, bytearray)):
        raise ResidualArchiveError(
            f"pr106_bytes must be bytes/bytearray; got {type(pr106_bytes).__name__}"
        )
    if not isinstance(residual_bytes, (bytes, bytearray)):
        raise ResidualArchiveError(
            f"residual_bytes must be bytes/bytearray; got {type(residual_bytes).__name__}"
        )
    if len(residual_bytes) > _RESIDUAL_LEN_MAX:
        raise ResidualArchiveError(
            f"residual_bytes too large: {len(residual_bytes)} > {_RESIDUAL_LEN_MAX}"
        )
    if len(pr106_bytes) > _RESIDUAL_LEN_MAX:
        raise ResidualArchiveError(
            f"pr106_bytes too large: {len(pr106_bytes)} > {_RESIDUAL_LEN_MAX}"
        )

    header = struct.pack(
        "<BBI", PR106_RESIDUAL_MAGIC, format_id, len(pr106_bytes)
    )
    residual_header = struct.pack("<I", len(residual_bytes))
    archive_bytes = bytes(header) + bytes(pr106_bytes) + residual_header + bytes(residual_bytes)
    return BuildResidualArchiveResult(
        archive_bytes=archive_bytes,
        pr106_len=len(pr106_bytes),
        residual_len=len(residual_bytes),
        family=family,
        format_id=format_id,
    )


def parse_archive(blob: bytes) -> ParsedResidualArchive:
    """Inverse of ``build_archive()``.

    Returns a typed ``ParsedResidualArchive``. Raises ``ResidualArchiveError``
    on any contract violation (magic mismatch / truncation / trailing bytes).
    """

    if not blob:
        raise ResidualArchiveError("empty archive")
    if len(blob) < _HEADER_FIXED_BYTES:
        raise ResidualArchiveError(
            f"archive too short for header: {len(blob)} < {_HEADER_FIXED_BYTES}"
        )
    magic, format_id, pr106_len = struct.unpack_from("<BBI", blob, 0)
    if magic != PR106_RESIDUAL_MAGIC:
        raise ResidualArchiveError(
            f"magic mismatch: got 0x{magic:02X} expected 0x{PR106_RESIDUAL_MAGIC:02X}"
        )
    pos = _HEADER_FIXED_BYTES
    if pos + pr106_len > len(blob):
        raise ResidualArchiveError(
            f"truncated PR106 payload: pos={pos} pr106_len={pr106_len} total={len(blob)}"
        )
    pr106_bytes = bytes(blob[pos : pos + pr106_len])
    pos += pr106_len
    if pos + _RESIDUAL_LEN_BYTES > len(blob):
        raise ResidualArchiveError(
            f"truncated residual length prefix: pos={pos} total={len(blob)}"
        )
    (residual_len,) = struct.unpack_from("<I", blob, pos)
    pos += _RESIDUAL_LEN_BYTES
    if pos + residual_len > len(blob):
        raise ResidualArchiveError(
            f"truncated residual payload: pos={pos} residual_len={residual_len} total={len(blob)}"
        )
    residual_bytes = bytes(blob[pos : pos + residual_len])
    pos += residual_len
    if pos != len(blob):
        raise ResidualArchiveError(
            f"trailing bytes after residual: pos={pos} total={len(blob)}"
        )
    parsed = ParsedResidualArchive(
        magic=magic,
        format_id=format_id,
        family=PR106_RESIDUAL_FORMAT_NAMES.get(format_id),
        pr106_bytes=pr106_bytes,
        residual_bytes=residual_bytes,
        total_bytes=len(blob),
    )
    parsed.assert_invariants()
    return parsed


def expect_format_id(blob: bytes, *, family: str) -> ParsedResidualArchive:
    """Parse and verify the format_id matches the expected family.

    Inflate runtimes use this to refuse archives from a different family
    (each family's inflate.py is family-scoped; cross-family dispatch is a
    bug class to refuse loudly).
    """

    expected = _ensure_family(family)
    parsed = parse_archive(blob)
    if parsed.format_id != expected:
        raise ResidualArchiveError(
            f"format_id 0x{parsed.format_id:02X} != expected 0x{expected:02X} "
            f"(family {family!r}). This inflate runtime is family-scoped."
        )
    return parsed


def sparse_family_name(dense_family: str) -> str:
    """Return the sparse-PacketIR sibling family name for a dense family.

    Example: ``"wavelet"`` -> ``"wavelet_sparse"``. Raises if the dense
    family lacks a sparse sibling.
    """

    sparse = f"{dense_family}_sparse"
    if sparse not in PR106_RESIDUAL_FORMAT_IDS:
        raise ResidualArchiveError(
            f"no sparse sibling registered for family {dense_family!r}"
        )
    return sparse


__all__ = [
    "BuildResidualArchiveResult",
    "ParsedResidualArchive",
    "PR106_RESIDUAL_FORMAT_IDS",
    "PR106_RESIDUAL_FORMAT_NAMES",
    "PR106_RESIDUAL_MAGIC",
    "ResidualArchiveError",
    "build_archive",
    "expect_format_id",
    "parse_archive",
    "sparse_family_name",
]
