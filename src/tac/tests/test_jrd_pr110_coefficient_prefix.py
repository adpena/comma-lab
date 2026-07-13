# SPDX-License-Identifier: MIT
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from tac.packet_compiler.jrd_coefficient_prefix import quantize_prefix
from tac.packet_compiler.jrd_pr110_coefficient_prefix import (
    Pr110CoefficientPacket,
    encode_mapped_i8,
)

ROOT = Path(__file__).resolve().parents[3]
SUBMISSION_DIR = ROOT / "experiments/results/pr110_payload_entropy_recode_20260610/submission_dir"
PRIMARY = ROOT / "experiments/results/clickpolish_pr110_20260710/n8_validation/candidate_archive.zip"
SECONDARY = ROOT / "experiments/results/click_polish_399_import_base.zip"

needs_archives = pytest.mark.skipif(
    not (PRIMARY.is_file() and SECONDARY.is_file() and SUBMISSION_DIR.is_dir()),
    reason="PR110 Phase-1 target archives are absent",
)


@pytest.mark.parametrize("byte_map", ["zig", "negzig", "off", "twos"])
def test_submission_byte_map_inverse_covers_all_int8_values(byte_map: str) -> None:
    values = np.arange(-128, 128, dtype=np.int16).astype(np.int8)
    stored = encode_mapped_i8(values, byte_map)

    def zig_decode(raw: np.ndarray) -> np.ndarray:
        wide = raw.astype(np.int32)
        return np.where(wide % 2 == 0, wide // 2, -(wide // 2) - 1).astype(np.int8)

    if byte_map == "zig":
        decoded = zig_decode(stored)
    elif byte_map == "negzig":
        decoded = (-zig_decode(stored).astype(np.int16)).astype(np.int8)
    elif byte_map == "off":
        decoded = (stored.astype(np.int16) - 128).astype(np.int8)
    else:
        decoded = stored.view(np.int8)
    np.testing.assert_array_equal(decoded, values)


def test_unknown_byte_map_refuses() -> None:
    with pytest.raises(ValueError, match="unknown decoder byte map"):
        encode_mapped_i8(np.array([0], dtype=np.int8), "unknown")


@needs_archives
@pytest.mark.parametrize("archive", [PRIMARY, SECONDARY])
def test_real_archive_derives_28_sections_and_noop_is_byte_exact(archive: Path) -> None:
    packet = Pr110CoefficientPacket(archive, SUBMISSION_DIR)
    assert len(packet.sections) == 28
    assert sum(section.count for section in packet.sections) == 228_958
    assert packet.no_op_archive() == archive.read_bytes()
    assert len({section.name for section in packet.sections}) == 28
    assert {section.byte_map for section in packet.sections} == {
        "zig",
        "negzig",
        "off",
        "twos",
    }


@needs_archives
def test_real_single_tensor_prefix_preserves_every_other_grammar_section(
    tmp_path: Path,
) -> None:
    packet = Pr110CoefficientPacket(PRIMARY, SUBMISSION_DIR)
    section = packet.sections[0]
    q = packet.read_section(section)
    replacement = quantize_prefix(q, bits_removed=1, family="uniform")
    candidate = packet.repack_archive(section, replacement)
    path = tmp_path / "candidate.zip"
    path.write_bytes(candidate)
    parsed = Pr110CoefficientPacket(path, SUBMISSION_DIR)
    assert parsed.packet.original_member != packet.packet.original_member
    assert parsed.packet.sel_bytes == packet.packet.sel_bytes
    assert parsed.packet.dqs1_tail == packet.packet.dqs1_tail
    assert parsed.packet.sidecar == packet.packet.sidecar
    assert parsed.packet._original_lat_sec() == packet.packet._original_lat_sec()
    for other in packet.sections[1:]:
        np.testing.assert_array_equal(parsed.read_section(other), packet.read_section(other))


@needs_archives
def test_replacement_refuses_wrong_dtype_shape_and_forged_section() -> None:
    packet = Pr110CoefficientPacket(PRIMARY, SUBMISSION_DIR)
    section = packet.sections[0]
    q = packet.read_section(section)
    with pytest.raises(TypeError, match="int8"):
        packet.repack_archive(section, q.astype(np.int16))
    with pytest.raises(ValueError, match="replacement shape"):
        packet.repack_archive(section, q.reshape(-1)[:-1])
    forged = dataclasses.replace(section, count=section.count + 1)
    with pytest.raises(ValueError, match="metadata changed"):
        packet.read_section(forged)
