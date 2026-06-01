# SPDX-License-Identifier: MIT
"""Tests for compact decoder codec sweep materialization."""

from __future__ import annotations

import zipfile
from pathlib import Path

import torch

from tac.substrates._shared.compact_decoder_codec_sweep import (
    sweep_compact_decoder_codecs,
)
from tac.substrates.pact_nerv_selector_v4.archive import (
    pack_archive as pack_selector_v4_archive,
)
from tac.substrates.pact_nerv_selector_v4.archive import (
    parse_archive as parse_selector_v4_archive,
)
from tac.substrates.pact_nerv_vq.archive import (
    pack_archive as pack_vq_archive,
)
from tac.substrates.pact_nerv_vq.archive import (
    parse_archive as parse_vq_archive,
)


def _source_zip(path: Path, bin_bytes: bytes) -> Path:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("0.bin", bin_bytes)
    return path


def test_sweep_vq_archive_materializes_codec_variants_fail_closed(tmp_path: Path) -> None:
    torch.manual_seed(1)
    decoder = {"conv.weight": torch.randn(4, 3, 3, 3) * 0.02}
    codebook = torch.randn(8, 4) * 0.01
    indices = torch.tensor([0, 3, 5], dtype=torch.long)
    source_bin = pack_vq_archive(
        decoder,
        codebook,
        indices,
        {"embed_dim": 4},
        decoder_codec="int8_mixed",
        indices_codec="auto",
    )
    source = _source_zip(tmp_path / "source_vq.zip", source_bin)

    report = sweep_compact_decoder_codecs(
        source_archive_zip=source,
        output_dir=tmp_path / "sweep",
        decoder_codecs=("int8_mixed", "int8_scale_bundled", "portfolio_auto"),
        repo_root=Path.cwd(),
        run_receiver_proof=False,
    )

    rows = report["variant_rows"]
    assert report["family"] == "pact_nerv_vq"
    assert len(rows) == 3
    assert [row["archive_bytes"] for row in rows] == sorted(
        row["archive_bytes"] for row in rows
    )
    assert report["best_variant"]["archive_bytes"] == min(
        row["archive_bytes"] for row in rows
    )
    for row in rows:
        assert row["score_claim"] is False
        assert row["ready_for_exact_eval_dispatch"] is False
        assert row["charged_bits_changed"] is True
        assert row["score_affecting_payload_changed"] is True
        assert row["exact_axis_score_affecting_adjudication_required"] is True
        assert "receiver_proof_not_run" in row["blockers"]
        assert Path(row["archive_path"]).is_file()
        assert Path(row["bin_path"]).is_file()
        parse_vq_archive(Path(row["bin_path"]).read_bytes())


def test_sweep_selector_archive_materializes_codec_variants_fail_closed(
    tmp_path: Path,
) -> None:
    torch.manual_seed(2)
    decoder = {"conv.weight": torch.randn(4, 3, 3, 3) * 0.02}
    latents = torch.randn(3, 4) * 0.01
    selector_bytes = b"\x00\x01\x01"
    source_bin = pack_selector_v4_archive(
        decoder,
        latents,
        selector_bytes,
        {"embed_dim": 4},
        palette_size=16,
        decoder_codec="int8_mixed",
    )
    source = _source_zip(tmp_path / "source_selector.zip", source_bin)

    report = sweep_compact_decoder_codecs(
        source_archive_zip=source,
        output_dir=tmp_path / "sweep_selector",
        decoder_codecs=("int8_mixed", "int8_scale_bundled"),
        repo_root=Path.cwd(),
        run_receiver_proof=False,
    )

    rows = report["variant_rows"]
    assert report["family"] == "pact_nerv_selector_v4"
    assert len(rows) == 2
    for row in rows:
        assert row["promotion_eligible"] is False
        assert row["charged_bits_changed"] is True
        assert "receiver_proof_not_run" in row["blockers"]
        assert Path(row["archive_path"]).is_file()
        parse_selector_v4_archive(Path(row["bin_path"]).read_bytes())
