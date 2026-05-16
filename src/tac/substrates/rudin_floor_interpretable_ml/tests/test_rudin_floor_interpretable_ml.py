# SPDX-License-Identifier: MIT
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tac.substrates.rudin_floor_interpretable_ml import (
    RDIF_HEADER_SIZE,
    RDIF_MAGIC,
    RDIF_VERSION,
    RudinFallingRule,
    RudinRuleList,
    inflate_one_video,
    pack_archive,
    parse_archive,
)
from tac.substrates.rudin_floor_interpretable_ml.inflate import main


def _rule_list() -> RudinRuleList:
    return RudinRuleList(
        rules=(
            RudinFallingRule("scene==road", (10, 20, 30), (1, -1, 0)),
            RudinFallingRule("always", (200, 210, 220), (0,)),
        ),
        default_rgb=(1, 2, 3),
    )


def test_rdif_pack_parse_roundtrip() -> None:
    blob = pack_archive(
        rule_list=_rule_list(),
        encoder_tree_blob=b"tree",
        scorer_priors_blob=b"priors",
        rashomon_disagreement_blob=b"sigma",
    )
    assert blob.startswith(RDIF_MAGIC)
    parsed = parse_archive(blob)
    assert parsed.header.version == RDIF_VERSION
    assert parsed.header.section_count == 8
    assert parsed.header.payload_len > 0
    assert RDIF_HEADER_SIZE == 34
    assert parsed.sections["encoder_tree_blob"] == b"tree"
    assert parsed.sections["scorer_priors_blob"] == b"priors"
    assert parsed.sections["rashomon_disagreement_blob"] == b"sigma"
    assert parsed.rule_list.evaluate({"scene": "road"}) == (10, 20, 30)
    assert parsed.rule_list.evaluate({"scene": "sky"}) == (200, 210, 220)


def test_rdif_refuses_mutated_payload() -> None:
    blob = bytearray(pack_archive(rule_list=_rule_list()))
    blob[RDIF_HEADER_SIZE + 4] ^= 0x01
    with pytest.raises(ValueError, match="sha256 mismatch"):
        parse_archive(bytes(blob))


def test_inflate_one_video_writes_deterministic_image(tmp_path: Path) -> None:
    blob = pack_archive(rule_list=_rule_list())
    output = inflate_one_video(blob, tmp_path / "frame", features={"scene": "road"})
    assert output.is_file()
    assert output.suffix in {".png", ".ppm"}
    assert output.stat().st_size > 0


def test_inflate_cli_contract(tmp_path: Path) -> None:
    archive_dir = tmp_path / "archive"
    archive_dir.mkdir()
    (archive_dir / "0.bin").write_bytes(pack_archive(rule_list=_rule_list()))
    output_dir = tmp_path / "out"
    file_list = tmp_path / "files.txt"
    file_list.write_text("0.mkv\n", encoding="utf-8")

    saved = sys.argv
    try:
        sys.argv = ["inflate.py", str(archive_dir), str(output_dir), str(file_list)]
        assert main() == 0
    finally:
        sys.argv = saved
    assert (output_dir / "0.png").is_file() or (output_dir / "0.ppm").is_file()
