# SPDX-License-Identifier: MIT
"""Tests for ATW2 cdf_table_blob dead-section probes."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import torch

from tac.substrates.atw_codec_v2 import (
    ATWv2Codec,
    ATWv2CodecConfig,
    ATWv2Variant,
    parse_atw2_archive_bytes,
    pack_archive,
    parse_archive,
)
from tac.substrates.atw_codec_v2.cdf_dead_section import (
    analyze_atw2_cdf_section,
    compact_atw2_cdf_table_in_archive_zip,
    compose_atw2_archive_without_cdf_table,
    mutate_atw2_cdf_table_bytes,
    prove_atw2_cdf_compaction_parity,
    prove_atw2_cdf_decode_influence,
)

REPO_ROOT = Path(__file__).resolve().parents[5]


def _make_archive() -> bytes:
    cfg = ATWv2CodecConfig(
        variant=ATWv2Variant.B_WZ_ONLY,
        latent_dim=8,
        encoder_input_channels=3,
        encoder_hidden_dim=16,
        decoder_embed_dim=8,
        decoder_initial_grid_h=2,
        decoder_initial_grid_w=2,
        decoder_channels=(6, 4, 4, 4, 4, 4),
        decoder_num_upsample_blocks=2,
        num_pairs=4,
        output_height=16,
        output_width=24,
        scorer_class_prior_dim=8,
        wz_head_hidden_dim=8,
        g1_distill_hidden_dim=8,
    )
    torch.manual_seed(123)
    model = ATWv2Codec(cfg).eval()
    with torch.no_grad():
        model.scorer_class_prior_table.normal_(0.0, 0.2)
        model.cdf_table.copy_(
            torch.linspace(0.001, 0.999, model.cdf_table.numel()).view_as(
                model.cdf_table
            )
        )
    meta: dict[str, Any] = {
        "decoder_embed_dim": cfg.decoder_embed_dim,
        "decoder_initial_grid_h": cfg.decoder_initial_grid_h,
        "decoder_initial_grid_w": cfg.decoder_initial_grid_w,
        "decoder_channels": list(cfg.decoder_channels),
        "decoder_num_upsample_blocks": cfg.decoder_num_upsample_blocks,
        "encoder_input_channels": cfg.encoder_input_channels,
        "encoder_hidden_dim": cfg.encoder_hidden_dim,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
        "wz_head_hidden_dim": cfg.wz_head_hidden_dim,
        "g1_distill_hidden_dim": cfg.g1_distill_hidden_dim,
        "latent_init_std": cfg.latent_init_std,
    }
    return pack_archive(
        model.encoder.state_dict(),
        model.decoder.state_dict(),
        model.wz_side_info_head.state_dict(),
        model.g1_distill_head.state_dict(),
        model.latents.detach().cpu(),
        model.scorer_class_prior_table.detach().cpu(),
        model.cdf_table.detach().cpu(),
        meta,
        variant=1,
    )


def _write_stored_archive_zip(path: Path, member_bytes: bytes) -> None:
    info = zipfile.ZipInfo("0.bin", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr(info, member_bytes)


def test_analyze_atw2_cdf_section_reports_rate_only_estimate() -> None:
    archive = _make_archive()
    analysis = analyze_atw2_cdf_section(archive)
    assert analysis.cdf_classes == 5
    assert analysis.cdf_symbols == 256
    assert analysis.cdf_bytes == 2560
    assert analysis.parser_visible is True
    assert analysis.current_runtime_decode_visible is False
    assert analysis.conservative_bytes_saved == 2528
    assert analysis.conservative_delta_s_rate_only < 0
    assert analysis.score_claim is False


def test_mutate_atw2_cdf_table_changes_only_cdf_table() -> None:
    archive = _make_archive()
    analysis = analyze_atw2_cdf_section(archive)
    mutated = mutate_atw2_cdf_table_bytes(archive)
    assert len(mutated) == len(archive)
    diffs = [i for i, (a, b) in enumerate(zip(archive, mutated)) if a != b]
    assert diffs[0] == analysis.cdf_offset
    assert diffs[-1] == analysis.cdf_offset + analysis.cdf_bytes - 1
    assert len(diffs) == analysis.cdf_bytes
    assert parse_archive(mutated).cdf_table.shape == (5, 256)


def test_compact_cdf_sentinel_parses_and_saves_bytes() -> None:
    archive = _make_archive()
    compact = compose_atw2_archive_without_cdf_table(archive)
    source_analysis = analyze_atw2_cdf_section(archive)
    compact_analysis = analyze_atw2_cdf_section(compact, envelope_bytes=8)
    assert len(archive) - len(compact) == 2552
    assert compact_analysis.cdf_bytes == 8
    assert compact_analysis.cdf_classes == source_analysis.cdf_classes
    assert compact_analysis.cdf_symbols == source_analysis.cdf_symbols
    parsed = parse_archive(compact)
    assert parsed.cdf_table.shape == (5, 256)
    assert float(parsed.cdf_table[0, 0]) == 1.0 / 256.0


def test_compact_cdf_bad_sentinel_rejected() -> None:
    archive = _make_archive()
    compact = bytearray(compose_atw2_archive_without_cdf_table(archive))
    section_offset, _ = parse_atw2_archive_bytes(compact)["cdf_table_blob"]
    compact[section_offset] ^= 0xFF
    bad = bytes(compact)
    for parser in (parse_atw2_archive_bytes, parse_archive):
        try:
            parser(bad)
        except ValueError as exc:
            assert "bad sentinel" in str(exc)
        else:
            raise AssertionError("bad compact cdf_table_blob sentinel was accepted")


def test_cdf_table_xor_preserves_current_inflate_raw_output() -> None:
    archive = _make_archive()
    proof = prove_atw2_cdf_decode_influence(archive, device="cpu")
    assert proof.raw_equal is True
    assert proof.max_abs_raw_byte_delta == 0
    assert proof.mutated_byte_count == 2560
    assert proof.source_archive_sha256 != proof.mutated_archive_sha256
    assert proof.source_raw_sha256 == proof.mutated_raw_sha256
    assert proof.score_claim is False


def test_compact_cdf_sentinel_preserves_current_inflate_raw_output() -> None:
    archive = _make_archive()
    proof = prove_atw2_cdf_compaction_parity(archive, device="cpu")
    assert proof.raw_equal is True
    assert proof.max_abs_raw_byte_delta == 0
    assert proof.archive_bytes_saved == 2552
    assert proof.compact_cdf_bytes == 8
    assert proof.delta_s_rate_only < 0
    assert proof.source_archive_sha256 != proof.compact_archive_sha256
    assert proof.source_raw_sha256 == proof.compact_raw_sha256
    assert proof.score_claim is False


def test_compact_cdf_archive_zip_writes_member_and_reports_zip_savings() -> None:
    archive = _make_archive()
    with tempfile.TemporaryDirectory(prefix="atw2-cdf-zip-test-") as tmp:
        root = Path(tmp)
        source_zip = root / "source" / "archive.zip"
        compact_zip = root / "compact" / "archive.zip"
        _write_stored_archive_zip(source_zip, archive)
        proof = compact_atw2_cdf_table_in_archive_zip(
            source_zip,
            compact_zip,
            device="cpu",
        )
        assert proof.inner_proof.raw_equal is True
        assert proof.member_name == "0.bin"
        assert proof.member_compress_type == zipfile.ZIP_STORED
        assert proof.archive_zip_bytes_saved == 2552
        assert proof.archive_zip_delta_s_rate_only < 0
        assert proof.source_archive_zip_sha256 != proof.compact_archive_zip_sha256
        assert proof.score_claim is False
        with zipfile.ZipFile(compact_zip, "r") as zf:
            compact_member = zf.read("0.bin")
        assert analyze_atw2_cdf_section(compact_member, envelope_bytes=8).cdf_bytes == 8
        assert parse_archive(compact_member).cdf_table.shape == (5, 256)


def test_compact_cdf_archive_zip_refuses_source_overwrite() -> None:
    archive = _make_archive()
    with tempfile.TemporaryDirectory(prefix="atw2-cdf-same-path-test-") as tmp:
        source_zip = Path(tmp) / "archive.zip"
        _write_stored_archive_zip(source_zip, archive)
        try:
            compact_atw2_cdf_table_in_archive_zip(
                source_zip,
                source_zip,
                device="cpu",
            )
        except ValueError as exc:
            assert "must not overwrite" in str(exc)
        else:
            raise AssertionError("source archive.zip overwrite was accepted")


def test_probe_atw2_cdf_dead_section_cli_synthetic_smoke() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "probe_atw2_cdf_dead_section.py"),
            "--synthetic-smoke",
            "--device",
            "cpu",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    payload = json.loads(proc.stdout)
    assert payload["raw_equal"] is True
    assert payload["analysis"]["cdf_bytes"] == 2560
    assert payload["ready_for_exact_eval_dispatch"] is False


def test_probe_atw2_cdf_dead_section_cli_compact_smoke() -> None:
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "probe_atw2_cdf_dead_section.py"),
            "--synthetic-smoke",
            "--compact-cdf",
            "--device",
            "cpu",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    payload = json.loads(proc.stdout)
    assert payload["raw_equal"] is True
    assert payload["archive_bytes_saved"] == 2552
    assert payload["compact_cdf_bytes"] == 8
    assert payload["ready_for_exact_eval_dispatch"] is False


def test_probe_atw2_cdf_dead_section_cli_archive_zip_compact() -> None:
    archive = _make_archive()
    with tempfile.TemporaryDirectory(prefix="atw2-cdf-cli-zip-test-") as tmp:
        root = Path(tmp)
        source_zip = root / "source.zip"
        compact_zip = root / "compact.zip"
        _write_stored_archive_zip(source_zip, archive)
        proc = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "tools" / "probe_atw2_cdf_dead_section.py"),
                str(source_zip),
                "--compact-cdf",
                "--output-archive-zip",
                str(compact_zip),
                "--device",
                "cpu",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        payload = json.loads(proc.stdout)
        assert compact_zip.is_file()
        assert payload["archive_zip_bytes_saved"] == 2552
        assert payload["inner_proof"]["raw_equal"] is True
        assert payload["ready_for_exact_eval_dispatch"] is False


def test_materialize_atw2_cdf_compaction_smoke_cli(tmp_path: Path) -> None:
    out_dir = tmp_path / "atw2-materialized"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "materialize_atw2_cdf_compaction_smoke.py"),
            "--output-dir",
            str(out_dir),
            "--epochs",
            "1",
            "--device",
            "cpu",
            "--variant",
            "B",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    payload = json.loads(proc.stdout)
    assert (out_dir / "materialized_compaction_report.json").is_file()
    assert (out_dir / "materialized_compaction_report.md").is_file()
    assert (out_dir / "source" / "archive.zip").is_file()
    assert (out_dir / "compact" / "archive.zip").is_file()
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["proof"]["archive_zip_bytes_saved"] > 0
    assert payload["proof"]["inner_proof"]["raw_equal"] is True
    assert payload["proof"]["inner_proof"]["max_abs_raw_byte_delta"] == 0


def test_compact_atw2_cdf_candidates_cli(tmp_path: Path) -> None:
    archive = _make_archive()
    input_root = tmp_path / "inputs"
    source_zip = input_root / "case_a" / "archive.zip"
    output_dir = tmp_path / "batch-output"
    _write_stored_archive_zip(source_zip, archive)
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "compact_atw2_cdf_candidates.py"),
            str(input_root),
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    payload = json.loads(proc.stdout)
    assert (output_dir / "batch_compaction_report.json").is_file()
    assert (output_dir / "batch_compaction_report.md").is_file()
    assert payload["scan_candidates_found"] == 1
    assert payload["candidates_seen"] == 1
    assert payload["full_candidates_seen"] == 0
    assert payload["non_full_candidates_seen"] == 1
    assert payload["skipped_non_full_candidate_count"] == 0
    assert payload["compacted_count"] == 1
    assert payload["failure_count"] == 0
    assert payload["total_archive_zip_bytes_saved"] == 2552
    row = payload["compacted"][0]
    assert Path(row["output_archive_zip_path"]).is_file()
    assert row["source_archive_zip_path"] == str(source_zip)
    assert row["num_pairs"] == 4
    assert row["candidate_class"] == "smoke_or_small_candidate"
    assert row["full_candidate"] is False
    assert row["archive_zip_bytes_saved"] == 2552
    assert row["raw_equal"] is True
    assert row["max_abs_raw_byte_delta"] == 0
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False


def test_compact_atw2_cdf_candidates_cli_full_candidate_only_skips_smoke(
    tmp_path: Path,
) -> None:
    archive = _make_archive()
    input_root = tmp_path / "inputs"
    source_zip = input_root / "case_a" / "archive.zip"
    output_dir = tmp_path / "full-only-output"
    _write_stored_archive_zip(source_zip, archive)
    proc = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "compact_atw2_cdf_candidates.py"),
            str(input_root),
            "--output-dir",
            str(output_dir),
            "--device",
            "cpu",
            "--full-candidate-only",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    payload = json.loads(proc.stdout)
    assert payload["scan_candidates_found"] == 1
    assert payload["candidates_seen"] == 0
    assert payload["full_candidates_seen"] == 0
    assert payload["non_full_candidates_seen"] == 0
    assert payload["skipped_non_full_candidate_count"] == 1
    assert payload["compacted_count"] == 0
    assert payload["failure_count"] == 0
    assert payload["total_archive_zip_bytes_saved"] == 0
    assert payload["scan_report"]["candidates"][0]["num_pairs"] == 4
    assert (
        payload["scan_report"]["candidates"][0]["candidate_class"]
        == "smoke_or_small_candidate"
    )
    assert payload["scan_report"]["candidates"][0]["full_candidate"] is False
