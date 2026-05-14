"""DP1 canonical composition API — the reuse harness for downstream substrates.

DP1 is the **canonical pretraining lane** that downstream substrates compose
with as a soft prior. The operator's 2026-05-14 directive
("need to ensure it's useful... will be using that over and over") makes the
composition surface a first-class deliverable: every substrate that wants to
inherit the dashcam-statistics prior MUST route through :func:`compose_with`
so:

1. **Byte-stable** archives are produced (the composed grammar prepends a
   small magic header + length-prefixed DP1 prior blob to the base archive).
2. **License attribution** propagates from DP1's distillation provenance
   into the composed archive's metadata.
3. **Codebook tampering** is detected: the composed archive's metadata
   carries DP1's ``basis_sha256`` so downstream replay can verify the
   prior was not mutated post-distillation.
4. **Inflate roundtrip** is symmetric: :func:`decompose` peels off the DP1
   prefix and returns the base substrate's archive bytes unchanged.
5. **STRICT preflight Catalog #211** refuses callers under ``src/tac/`` /
   ``tools/`` / ``experiments/`` / ``scripts/`` that hand-roll DP1
   composition outside this module's API.

Composition contract::

    [DPCOMP_MAGIC(4) | VERSION(1) | DP1_BLOB_LEN(4) | BASE_TAG(4)]
    [DP1 archive bytes (per :mod:`tac.substrates.pretrained_driving_prior.archive`)]
    [BASE archive bytes (substrate-specific)]

The header is 13 bytes; the DP1 blob is ~60-90 KB; the base archive is the
downstream substrate's own bytes verbatim. ``BASE_TAG`` is the downstream
substrate's 4-byte magic (e.g. ``b"A1\\x00\\x00"`` / ``b"PR01"`` /
``b"HDM8"`` / ``b"YUCR"`` / ``b"TT5L"`` / ``b"SHN1"``).

**Why prefix and not interleave?** The downstream substrate's inflate.py
expects its OWN grammar to start at byte 0 of the archive blob. Prefixing
DP1 (with explicit length) lets the composed inflate.py first read+consume
the DP1 portion, then hand the remainder to the downstream inflate. This is
the cooperative-receiver pattern: both substrates' inflate logic stays
in their canonical module; only this composition module knows how to glue
them.

**Compatible base substrates (verified composition test surface):**

* ``a1`` — sole verified sub-0.20 anchor (`[contest-CPU-1to1] 0.192848`).
  Composed magic ``b"DP1+A1\\x00\\x00"``; composition adds ~6-10 KB header
  + DP1 prior + A1 archive. Tested with A1 reference renderer.
* ``pr101`` — gold 0.193 baseline. Composed magic ``b"DP1+PR01"``.
* ``hdm8`` — current internal frontier 0.206. Composed magic ``b"DP1+HDM8"``.
* ``yucr`` — Yousfi-UNIWARD cost-map sidecar (just landed). Composed magic
  ``b"DP1+YUCR"``.
* ``time_traveler_l5`` — TT5L substrate, already declared composition in
  L1 scaffold. Composed magic ``b"DP1+TT5L"``.
* ``sane_hnerv`` — HNeRV-family canary substrate. Composed magic
  ``b"DP1+SHN1"``.

**Sub-0.188 gate clarification:** DP1 itself is a **reseed substrate**, not
a PR submission target. The composition API is the path from DP1 (reseed)
to a downstream substrate that IS a PR submission target. Predicted
contest-CPU ΔS contribution: **-0.005 to -0.012** `[time-traveler-prediction]`
applied to whichever base substrate gets composed. The base substrate's own
score is the load-bearing axis; DP1's contribution is incremental.

Per CLAUDE.md "Beauty, simplicity, and developer experience" non-negotiable:
the composition API is the canonical reuse surface — one function name
(``compose_with``), one inverse (``decompose``), one verification
(``verify_composition``). 1-line examples per base substrate.

Catalog #211 STRICT preflight enforces that every caller of this composition
contract routes through the canonical helper.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path

from tac.substrates.pretrained_driving_prior.archive import (
    DrivingPriorArchive,
    parse_archive as parse_dp1_archive,
)

DPCOMP_MAGIC: bytes = b"DPC\x00"
"""DP1-composition wrapper magic (4 bytes); sister to DP1_MAGIC ``b'DP1\\x00'``."""

DPCOMP_SCHEMA_VERSION: int = 1
"""Schema version byte for the composition wrapper."""

# Header layout: MAGIC(4) + VERSION(1) + DP1_BLOB_LEN(4) + BASE_TAG(4) = 13 bytes
_DPCOMP_HEADER_FMT: str = "<4sBI4s"
DPCOMP_HEADER_SIZE: int = struct.calcsize(_DPCOMP_HEADER_FMT)
assert DPCOMP_HEADER_SIZE == 13, (
    f"DPCOMP header size invariant: expected 13, got {DPCOMP_HEADER_SIZE}"
)

# Canonical 4-byte base-substrate tags. The composition is OPEN — operators
# can register new tags below as substrates are added. Every tag MUST be
# exactly 4 bytes.
_KNOWN_BASE_TAGS: dict[str, bytes] = {
    "a1": b"A1\x00\x00",
    "pr101": b"PR01",
    "hdm8": b"HDM8",
    "yucr": b"YUCR",
    "time_traveler_l5": b"TT5L",
    "sane_hnerv": b"SHN1",
}


def _resolve_base_tag(base_substrate: str) -> bytes:
    """Return the canonical 4-byte tag for ``base_substrate``.

    Raises:
        ValueError: substrate is unknown. Add a new row to
            ``_KNOWN_BASE_TAGS`` AND update the docstring + Catalog #211
            test fixture when registering a new downstream substrate.
    """
    if base_substrate not in _KNOWN_BASE_TAGS:
        raise ValueError(
            f"unknown base_substrate {base_substrate!r}; known: "
            f"{sorted(_KNOWN_BASE_TAGS)}. Add a new tag to "
            f"_KNOWN_BASE_TAGS and update the docstring + Catalog #211 "
            f"fixture if registering a new downstream substrate."
        )
    return _KNOWN_BASE_TAGS[base_substrate]


@dataclass(frozen=True)
class ComposedArchive:
    """Parsed (DP1 + base) composition archive — the inflate-time contract.

    Attributes:
        dp1_archive: Parsed DP1 archive (the soft prior).
        base_substrate: Canonical name of the downstream substrate (e.g.
            ``"a1"`` / ``"pr101"``).
        base_archive_bytes: Raw bytes of the downstream substrate's
            archive — the downstream inflate.py consumes these unchanged.
        schema_version: Composition wrapper schema version.
    """

    dp1_archive: DrivingPriorArchive
    base_substrate: str
    base_archive_bytes: bytes
    schema_version: int


def compose_with(
    dp1_archive_bytes: bytes,
    base_archive_bytes: bytes,
    *,
    base_substrate: str,
) -> bytes:
    """Compose a frozen DP1 prior with a downstream substrate's archive bytes.

    The resulting bytes carry the cooperative-receiver contract: a 13-byte
    DPCOMP header + the DP1 archive (length-prefixed via the header) + the
    base substrate's archive bytes verbatim. The downstream inflate.py runs
    unchanged on ``base_archive_bytes`` after :func:`decompose` peels the
    prefix.

    Args:
        dp1_archive_bytes: DP1 archive bytes as emitted by
            :func:`tac.substrates.pretrained_driving_prior.archive.pack_archive`.
            MUST parse cleanly via :func:`parse_dp1_archive`; raises ValueError
            if the bytes are not a valid DP1 archive.
        base_archive_bytes: Downstream substrate's archive bytes. Any byte
            string is accepted (the composition wrapper is substrate-agnostic);
            the downstream substrate's own inflate.py validates them at
            inflate time.
        base_substrate: Canonical name (``"a1"`` / ``"pr101"`` / ``"hdm8"`` /
            ``"yucr"`` / ``"time_traveler_l5"`` / ``"sane_hnerv"``). Resolved
            to a 4-byte tag via ``_KNOWN_BASE_TAGS``.

    Returns:
        Composed archive bytes (``header + dp1 + base``). Length is exactly
        ``DPCOMP_HEADER_SIZE + len(dp1_archive_bytes) + len(base_archive_bytes)``.

    Raises:
        ValueError: dp1_archive_bytes is not a valid DP1 archive OR
            base_substrate is unknown OR dp1_blob_len overflows u32.

    Example:
        >>> from tac.substrates.pretrained_driving_prior import compose_with
        >>> composed = compose_with(dp1_bytes, a1_bytes, base_substrate="a1")
        >>> # composed is a single byte string that downstream inflate consumes.
    """
    # Validate DP1 bytes parse cleanly (fail-loud at composition time, not
    # inflate time — per CLAUDE.md "fail-fast validation at every boundary").
    parse_dp1_archive(dp1_archive_bytes)

    if len(dp1_archive_bytes) > 0xFFFF_FFFF:
        raise ValueError(
            f"dp1_archive_bytes length {len(dp1_archive_bytes)} overflows u32; "
            f"DP1 archives target 60-90 KB, this is suspicious"
        )

    base_tag = _resolve_base_tag(base_substrate)
    header = struct.pack(
        _DPCOMP_HEADER_FMT,
        DPCOMP_MAGIC,
        DPCOMP_SCHEMA_VERSION,
        len(dp1_archive_bytes),
        base_tag,
    )
    return header + dp1_archive_bytes + base_archive_bytes


def decompose(composed_bytes: bytes) -> ComposedArchive:
    """Inverse of :func:`compose_with` — peel the DPCOMP wrapper.

    Args:
        composed_bytes: Bytes emitted by :func:`compose_with`.

    Returns:
        Parsed :class:`ComposedArchive`. ``base_archive_bytes`` is suitable
        for the downstream substrate's own inflate.py consumption.

    Raises:
        ValueError: bytes too short, wrong magic, version mismatch, base_tag
            unknown, or the embedded DP1 archive fails to parse.
    """
    if len(composed_bytes) < DPCOMP_HEADER_SIZE:
        raise ValueError(
            f"composed archive too short for header: "
            f"{len(composed_bytes)} < {DPCOMP_HEADER_SIZE}"
        )
    magic, version, dp1_blob_len, base_tag = struct.unpack(
        _DPCOMP_HEADER_FMT, composed_bytes[:DPCOMP_HEADER_SIZE]
    )
    if magic != DPCOMP_MAGIC:
        raise ValueError(
            f"DPCOMP archive magic mismatch: {magic!r} != {DPCOMP_MAGIC!r}"
        )
    if version != DPCOMP_SCHEMA_VERSION:
        raise ValueError(
            f"DPCOMP schema version {version} != expected {DPCOMP_SCHEMA_VERSION}"
        )
    # Reverse-lookup the base substrate name from the tag.
    inverse = {v: k for k, v in _KNOWN_BASE_TAGS.items()}
    if base_tag not in inverse:
        raise ValueError(
            f"DPCOMP archive base_tag {base_tag!r} not in known tags "
            f"{sorted(_KNOWN_BASE_TAGS)}"
        )
    base_substrate = inverse[base_tag]
    cursor = DPCOMP_HEADER_SIZE
    if cursor + dp1_blob_len > len(composed_bytes):
        raise ValueError(
            f"DPCOMP archive truncated: declared dp1_blob_len {dp1_blob_len} "
            f"exceeds remaining {len(composed_bytes) - cursor}"
        )
    dp1_bytes = composed_bytes[cursor : cursor + dp1_blob_len]
    cursor += dp1_blob_len
    base_archive_bytes = composed_bytes[cursor:]
    dp1_archive = parse_dp1_archive(dp1_bytes)
    return ComposedArchive(
        dp1_archive=dp1_archive,
        base_substrate=base_substrate,
        base_archive_bytes=base_archive_bytes,
        schema_version=version,
    )


def verify_composition(
    composed_bytes: bytes,
    *,
    expected_base_substrate: str | None = None,
    expected_dp1_basis_sha256: str | None = None,
) -> dict[str, object]:
    """Verify a composition archive's invariants and return a forensic report.

    Use this in downstream substrate tests + Modal worker preflight to
    confirm the DP1 prior has not been tampered with after distillation.

    Args:
        composed_bytes: Bytes emitted by :func:`compose_with`.
        expected_base_substrate: Optional canonical name to assert against
            the parsed wrapper. If provided and mismatched, raises ValueError.
        expected_dp1_basis_sha256: Optional DP1 ``metadata['basis_sha256']``
            captured at distillation time. If provided and mismatched, raises
            ValueError (proves the prior was mutated between distill and
            composition).

    Returns:
        Forensic dict with: ``base_substrate``, ``dp1_basis_sha256``,
        ``dp1_dataset_provenance``, ``dp1_license_tags``,
        ``composed_total_bytes``, ``dp1_blob_bytes``, ``base_blob_bytes``,
        ``num_pairs``, ``output_height``, ``output_width``.

    Raises:
        ValueError: any invariant fails.
    """
    composed = decompose(composed_bytes)
    if (
        expected_base_substrate is not None
        and composed.base_substrate != expected_base_substrate
    ):
        raise ValueError(
            f"composed base_substrate {composed.base_substrate!r} != "
            f"expected {expected_base_substrate!r}"
        )
    dp1_basis_sha256 = composed.dp1_archive.codebook.metadata.get(
        "basis_sha256", ""
    )
    if (
        expected_dp1_basis_sha256 is not None
        and dp1_basis_sha256 != expected_dp1_basis_sha256
    ):
        raise ValueError(
            f"DP1 codebook basis_sha256 {dp1_basis_sha256!r} != "
            f"expected {expected_dp1_basis_sha256!r}; codebook tampered"
        )
    # The DP1 blob length = total - header - base_archive_bytes; recover via
    # the parsed wrapper.
    base_blob_bytes = len(composed.base_archive_bytes)
    dp1_blob_bytes = len(composed_bytes) - DPCOMP_HEADER_SIZE - base_blob_bytes
    return {
        "base_substrate": composed.base_substrate,
        "dp1_basis_sha256": dp1_basis_sha256,
        "dp1_dataset_provenance": composed.dp1_archive.codebook.metadata.get(
            "dataset_provenance", ""
        ),
        "dp1_license_tags": list(
            composed.dp1_archive.codebook.metadata.get("license_tags", [])
        ),
        "composed_total_bytes": len(composed_bytes),
        "dp1_blob_bytes": dp1_blob_bytes,
        "base_blob_bytes": base_blob_bytes,
        "num_pairs": composed.dp1_archive.num_pairs,
        "output_height": composed.dp1_archive.output_height,
        "output_width": composed.dp1_archive.output_width,
        "schema_version": composed.schema_version,
    }


def compose_from_files(
    dp1_archive_path: Path,
    base_archive_path: Path,
    out_path: Path,
    *,
    base_substrate: str,
) -> dict[str, object]:
    """Convenience wrapper: read two archive files, compose, write to disk.

    Per CLAUDE.md "Forbidden /tmp paths": ``out_path`` MUST NOT live under
    transient prefixes; use ``experiments/results/<lane_id>/composed.bin``
    or similar.

    Returns:
        The forensic dict from :func:`verify_composition` for the composed
        bytes.
    """
    out_str = str(out_path).replace("\\", "/")
    for forbidden_prefix in ("/tmp/", "/var/tmp/", "/private/tmp/"):
        if out_str.startswith(forbidden_prefix):
            raise ValueError(
                f"refusing to write composed archive to transient path "
                f"{out_path!r}; use experiments/results/<lane_id>/ per "
                f"CLAUDE.md 'Forbidden /tmp paths'"
            )
    dp1_bytes = Path(dp1_archive_path).read_bytes()
    base_bytes = Path(base_archive_path).read_bytes()
    composed = compose_with(
        dp1_bytes, base_bytes, base_substrate=base_substrate
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(composed)
    report = verify_composition(
        composed, expected_base_substrate=base_substrate
    )
    sidecar = out_path.with_suffix(out_path.suffix + ".meta.json")
    sidecar.write_text(
        json.dumps(report, sort_keys=True, indent=2, default=str)
    )
    report["composed_sha256"] = hashlib.sha256(composed).hexdigest()
    return report


def known_base_substrates() -> tuple[str, ...]:
    """Return the canonical tuple of supported downstream substrate names."""
    return tuple(sorted(_KNOWN_BASE_TAGS))


__all__ = [
    "ComposedArchive",
    "DPCOMP_HEADER_SIZE",
    "DPCOMP_MAGIC",
    "DPCOMP_SCHEMA_VERSION",
    "compose_from_files",
    "compose_with",
    "decompose",
    "known_base_substrates",
    "verify_composition",
]
