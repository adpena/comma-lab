# SPDX-License-Identifier: MIT
"""Implicit framing adapter for the DDM M6 FP11/CTXR packet.

The selected 177,169-byte vehicle has two nested, fixed-format wrappers:

    FP11 | u32 source_len | CTXR | u8 version | 3 * u24 section lengths
         | sections | u16 selector_len | selector | DQS1 tail

For this one receiver contract, ``FP11``, ``CTXR``, and CTXR version 1 are
generic parser facts, while ``source_len`` is derivable from the retained
section lengths.  Rule 118 therefore permits those 13 bytes to live in generic
receiver code.  All video-derived sections and their boundary lengths remain
counted.

This module is deliberately only a lossless framing adapter.  It neither
re-encodes learned payloads nor claims evaluator score authority.
"""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass

FP11_MAGIC = b"FP11"
CTXR_MAGIC = b"CTXR"
CTXR_VERSION = 1
FECA_MAGIC = b"FECa"
DQS1_MAGIC = b"DQS1"

LEGACY_GENERIC_FRAMING_BYTES = 13
IMPLICIT_SECTION_LENGTH_BYTES = 9
SELECTOR_LENGTH_BYTES = 2
MAX_U24 = (1 << 24) - 1


class DdmM6ImplicitFramingError(ValueError):
    """Raised when an M6 framing transform cannot be proven lossless."""


@dataclass(frozen=True, slots=True)
class ImplicitFP11Parts:
    """Every retained, counted field in the implicit M6 packet."""

    decoder_section: bytes
    latent_section: bytes
    sidecar: bytes
    selector: bytes
    dqs1_tail: bytes

    def __post_init__(self) -> None:
        for name in ("decoder_section", "latent_section", "sidecar"):
            if len(getattr(self, name)) > MAX_U24:
                raise DdmM6ImplicitFramingError(f"{name} exceeds u24")
        if len(self.selector) > 0xFFFF:
            raise DdmM6ImplicitFramingError("selector exceeds u16")
        if not self.selector.startswith(FECA_MAGIC):
            raise DdmM6ImplicitFramingError(
                f"expected FECa selector, got {self.selector[:4]!r}"
            )
        if not self.dqs1_tail.startswith(DQS1_MAGIC):
            raise DdmM6ImplicitFramingError(
                f"expected DQS1 tail, got {self.dqs1_tail[:4]!r}"
            )


def split_legacy_member(member: bytes) -> ImplicitFP11Parts:
    """Parse the exact legacy FP11/CTXR/FECa/DQS1 grammar fail-closed."""

    if len(member) < 8 or member[:4] != FP11_MAGIC:
        raise DdmM6ImplicitFramingError("member is not FP11")
    source_len = int.from_bytes(member[4:8], "little")
    source_start = 8
    source_end = source_start + source_len
    if source_end + SELECTOR_LENGTH_BYTES > len(member):
        raise DdmM6ImplicitFramingError("FP11 source length exceeds member")

    source = member[source_start:source_end]
    if len(source) < 14 or source[:4] != CTXR_MAGIC:
        raise DdmM6ImplicitFramingError("FP11 source is not CTXR")
    if source[4] != CTXR_VERSION:
        raise DdmM6ImplicitFramingError(
            f"unsupported CTXR version {source[4]}"
        )

    decoder_len = int.from_bytes(source[5:8], "little")
    latent_len = int.from_bytes(source[8:11], "little")
    sidecar_len = int.from_bytes(source[11:14], "little")
    sections_end = 14 + decoder_len + latent_len + sidecar_len
    if sections_end != len(source):
        raise DdmM6ImplicitFramingError(
            "CTXR section lengths do not consume the source exactly"
        )

    position = 14
    decoder = source[position : position + decoder_len]
    position += decoder_len
    latent = source[position : position + latent_len]
    position += latent_len
    sidecar = source[position : position + sidecar_len]

    selector_len = int.from_bytes(
        member[source_end : source_end + SELECTOR_LENGTH_BYTES], "little"
    )
    selector_start = source_end + SELECTOR_LENGTH_BYTES
    selector_end = selector_start + selector_len
    if selector_end > len(member):
        raise DdmM6ImplicitFramingError("selector length exceeds member")

    return ImplicitFP11Parts(
        decoder_section=decoder,
        latent_section=latent,
        sidecar=sidecar,
        selector=member[selector_start:selector_end],
        dqs1_tail=member[selector_end:],
    )


def pack_implicit_member(parts: ImplicitFP11Parts) -> bytes:
    """Pack only retained counted fields; the receiver supplies fixed framing."""

    return (
        len(parts.decoder_section).to_bytes(3, "little")
        + len(parts.latent_section).to_bytes(3, "little")
        + len(parts.sidecar).to_bytes(3, "little")
        + parts.decoder_section
        + parts.latent_section
        + parts.sidecar
        + len(parts.selector).to_bytes(2, "little")
        + parts.selector
        + parts.dqs1_tail
    )


def unpack_implicit_member(member: bytes) -> ImplicitFP11Parts:
    """Parse the implicit member without sentinel scanning or guessed lengths."""

    minimum = IMPLICIT_SECTION_LENGTH_BYTES + SELECTOR_LENGTH_BYTES
    if len(member) < minimum:
        raise DdmM6ImplicitFramingError("implicit member is truncated")

    decoder_len = int.from_bytes(member[0:3], "little")
    latent_len = int.from_bytes(member[3:6], "little")
    sidecar_len = int.from_bytes(member[6:9], "little")
    position = IMPLICIT_SECTION_LENGTH_BYTES
    section_total = decoder_len + latent_len + sidecar_len
    selector_len_offset = position + section_total
    if selector_len_offset + SELECTOR_LENGTH_BYTES > len(member):
        raise DdmM6ImplicitFramingError("implicit section lengths exceed member")

    decoder = member[position : position + decoder_len]
    position += decoder_len
    latent = member[position : position + latent_len]
    position += latent_len
    sidecar = member[position : position + sidecar_len]
    position += sidecar_len

    selector_len = int.from_bytes(
        member[position : position + SELECTOR_LENGTH_BYTES], "little"
    )
    selector_start = position + SELECTOR_LENGTH_BYTES
    selector_end = selector_start + selector_len
    if selector_end > len(member):
        raise DdmM6ImplicitFramingError("implicit selector length exceeds member")

    return ImplicitFP11Parts(
        decoder_section=decoder,
        latent_section=latent,
        sidecar=sidecar,
        selector=member[selector_start:selector_end],
        dqs1_tail=member[selector_end:],
    )


def reconstruct_legacy_member(parts: ImplicitFP11Parts) -> bytes:
    """Restore the exact legacy bytes consumed by the existing receiver."""

    source = (
        CTXR_MAGIC
        + bytes((CTXR_VERSION,))
        + len(parts.decoder_section).to_bytes(3, "little")
        + len(parts.latent_section).to_bytes(3, "little")
        + len(parts.sidecar).to_bytes(3, "little")
        + parts.decoder_section
        + parts.latent_section
        + parts.sidecar
    )
    return (
        FP11_MAGIC
        + len(source).to_bytes(4, "little")
        + source
        + len(parts.selector).to_bytes(2, "little")
        + parts.selector
        + parts.dqs1_tail
    )


def stored_archive_bytes(member: bytes, *, member_name: str = "x") -> bytes:
    """Return the deterministic single-member ZIP_STORED contest container."""

    if member_name != "x":
        raise DdmM6ImplicitFramingError("M6 receiver contract requires member 'x'")
    output = io.BytesIO()
    info = zipfile.ZipInfo(member_name)
    info.compress_type = zipfile.ZIP_STORED
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.create_system = 3
    info.external_attr = 0o600 << 16
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(info, member)
    return output.getvalue()


__all__ = [
    "LEGACY_GENERIC_FRAMING_BYTES",
    "DdmM6ImplicitFramingError",
    "ImplicitFP11Parts",
    "pack_implicit_member",
    "reconstruct_legacy_member",
    "split_legacy_member",
    "stored_archive_bytes",
    "unpack_implicit_member",
]
