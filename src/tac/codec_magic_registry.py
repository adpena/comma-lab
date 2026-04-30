"""Canonical magic-byte registry for tac codecs.

Single source of truth for every 4-byte magic prefix used by any tac codec
that emits self-describing payloads. New codec? Add the magic + the loader
sniff function here so renderer/inflate paths can dispatch deterministically.

Why a registry?
---------------
Magic-byte collisions are silent corruption: if two codecs both emit
``b"OWV2"`` as a header, the loader will dispatch one of them and the
decoder will produce garbage tensors. The registry enforces uniqueness at
import time (any duplicate raises ImportError). It also provides a single
``sniff_codec(blob)`` entry point that loaders can use without importing
every codec module.

Design constraints
------------------
* No imports of heavy modules at registry-import time. Each entry stores
  the codec name + a lazy import path; ``sniff_codec`` resolves only when
  asked.
* Registry is FROZEN — adding a new entry means editing this file and
  bumping the regression test in ``test_codec_magic_registry.py`` (TBD).
* Strict-scorer-rule: registry entries describe DECODE-side codecs only.
  No registry entry may imply scorer load at decode time. The
  ``decode_module`` field MUST point to a pure-math byte→tensor function.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CodecMagicEntry:
    """One row in the canonical registry."""

    magic: bytes
    """4-byte ASCII prefix written at byte 0 of the codec payload."""

    name: str
    """Human-readable codec name (e.g. ``"Lane Ω-W-V2"``)."""

    decode_module: str
    """Dotted import path to the decode function, e.g.
    ``"tac.water_filling_codec_v2:decode_omega_w_v2"``."""

    encode_module: str
    """Dotted import path to the encode function, for documentation."""

    description: str
    """One-line summary of what the payload contains."""


# ── canonical registry table ───────────────────────────────────────────────


_REGISTRY: tuple[CodecMagicEntry, ...] = (
    CodecMagicEntry(
        magic=b"OWV2",
        name="Lane Ω-W-V2",
        decode_module="tac.water_filling_codec_v2:decode_omega_w_v2",
        encode_module="tac.water_filling_codec_v2:encode_omega_w_v2",
        description=(
            "Water-fill bit allocation + static-histogram arithmetic terminal "
            "on block-FP-eligible conv weights. Pre-archive renderer payload."
        ),
    ),
    CodecMagicEntry(
        magic=b"IMPS",
        name="Lane 17 IMP",
        decode_module="tac.imps_renderer_archive:decode_imps_archive",
        encode_module="tac.imps_renderer_archive:encode_imps_archive",
        description=(
            "Iterative-magnitude-pruning sparse-CSR archive — Conv2d weights "
            "at >=78%% sparsity pass through the per-tensor sparse-CSR codec "
            "(uint16 idx + FP4 val); ineligible / low-sparsity / large-numel "
            "tensors fall back to FP16."
        ),
    ),
)


# ── runtime helpers ────────────────────────────────────────────────────────


def all_entries() -> tuple[CodecMagicEntry, ...]:
    """Return the full registry."""
    return _REGISTRY


def find_by_magic(magic: bytes) -> CodecMagicEntry | None:
    """Look up the registry entry for a 4-byte magic prefix."""
    if not isinstance(magic, (bytes, bytearray, memoryview)):
        raise TypeError(
            f"find_by_magic: magic must be bytes-like, got {type(magic).__name__}"
        )
    if len(magic) < 4:
        return None
    needle = bytes(magic[:4])
    for entry in _REGISTRY:
        if entry.magic == needle:
            return entry
    return None


def sniff_codec(blob: bytes | bytearray | memoryview) -> CodecMagicEntry | None:
    """Inspect the first 4 bytes of ``blob`` and return the matching entry.

    Returns None if the prefix is not in the registry — caller must handle
    the legacy / unknown-magic case explicitly.
    """
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise TypeError(
            f"sniff_codec: blob must be bytes-like, got {type(blob).__name__}"
        )
    if len(blob) < 4:
        return None
    return find_by_magic(bytes(blob[:4]))


# ── uniqueness enforcement (raises at module import time on collision) ─────


def _validate_registry_no_duplicates() -> None:
    seen: dict[bytes, str] = {}
    for entry in _REGISTRY:
        if not isinstance(entry.magic, bytes) or len(entry.magic) != 4:
            raise ImportError(
                f"codec_magic_registry: entry '{entry.name}' has malformed "
                f"magic {entry.magic!r}; must be exactly 4 bytes."
            )
        if entry.magic in seen:
            raise ImportError(
                f"codec_magic_registry: DUPLICATE magic {entry.magic!r} "
                f"between '{seen[entry.magic]}' and '{entry.name}'. "
                f"Magic-byte collisions cause silent decode corruption — "
                f"resolve before merging."
            )
        seen[entry.magic] = entry.name


_validate_registry_no_duplicates()


__all__ = [
    "CodecMagicEntry",
    "all_entries",
    "find_by_magic",
    "sniff_codec",
]
