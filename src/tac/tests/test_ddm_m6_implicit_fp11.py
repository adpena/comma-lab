# SPDX-License-Identifier: MIT
"""Tests for the DDM M6 implicit FP11/CTXR receiver adapter."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from tac.packet_compiler.ddm_m6_implicit_fp11 import (
    LEGACY_GENERIC_FRAMING_BYTES,
    DdmM6ImplicitFramingError,
    ImplicitFP11Parts,
    pack_implicit_member,
    reconstruct_legacy_member,
    split_legacy_member,
    stored_archive_bytes,
    unpack_implicit_member,
)

SOURCE_ARCHIVE = Path(
    "/Volumes/VertigoDataTier/pact/evidence/joint_optimum_575_xhigh_20260720/"
    "n600_r1/n600_r1/candidate_archive.zip"
)


def _parts() -> ImplicitFP11Parts:
    return ImplicitFP11Parts(
        decoder_section=b"decoder",
        latent_section=b"latent",
        sidecar=b"side",
        selector=b"FECa-selector",
        dqs1_tail=b"DQS1-tail",
    )


def test_synthetic_implicit_roundtrip_saves_exactly_13_member_bytes() -> None:
    parts = _parts()
    legacy = reconstruct_legacy_member(parts)
    compact = pack_implicit_member(parts)
    assert len(legacy) - len(compact) == LEGACY_GENERIC_FRAMING_BYTES
    assert unpack_implicit_member(compact) == parts
    assert split_legacy_member(legacy) == parts
    assert reconstruct_legacy_member(unpack_implicit_member(compact)) == legacy


def test_parser_rejects_bad_selector_and_tail_magic() -> None:
    with pytest.raises(DdmM6ImplicitFramingError, match="FECa"):
        ImplicitFP11Parts(b"d", b"l", b"s", b"BAD!", b"DQS1-tail")
    with pytest.raises(DdmM6ImplicitFramingError, match="DQS1"):
        ImplicitFP11Parts(b"d", b"l", b"s", b"FECa-selector", b"BAD!")


@pytest.mark.skipif(not SOURCE_ARCHIVE.is_file(), reason="custodied source archive unavailable")
def test_exact_source_archive_roundtrip_and_byte_delta() -> None:
    source_archive = SOURCE_ARCHIVE.read_bytes()
    with zipfile.ZipFile(SOURCE_ARCHIVE) as archive:
        source_member = archive.read("x")
    parts = split_legacy_member(source_member)
    compact = pack_implicit_member(parts)
    reconstructed = reconstruct_legacy_member(unpack_implicit_member(compact))

    assert reconstructed == source_member
    assert stored_archive_bytes(reconstructed) == source_archive
    assert len(source_archive) == 177_169
    assert len(stored_archive_bytes(compact)) == 177_156
    assert len(source_archive) - len(stored_archive_bytes(compact)) == 13
