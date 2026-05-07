from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import brotli
import pytest

from tac.hnerv_lowlevel_packer import read_strict_single_member_zip, write_stored_single_member_zip
from tac.hnerv_wavelet_compress_time_harness import (
    HnervWaveletCompressTimeHarnessError,
    build_wavelet_compress_time_harness,
)

REPO = Path(__file__).resolve().parents[3]


def test_wavelet_compress_time_harness_manifest_is_deterministic_and_fail_closed(
    tmp_path: Path,
) -> None:
    archive = _source_archive(tmp_path)
    source = read_strict_single_member_zip(archive)
    out_dir = tmp_path / "harness"

    manifest = build_wavelet_compress_time_harness(
        source_archive=archive,
        source_label="PR106x_WR01",
        output_dir=out_dir,
        target_sections=("latents_and_sidecar_brotli",),
        seed=123,
        atom_budget=7,
        block_size=16,
        quant_step=2.0,
        expected_source_archive_sha256=source.archive_sha256,
        expected_source_archive_bytes=source.archive_bytes,
    )
    repeat = build_wavelet_compress_time_harness(
        source_archive=archive,
        source_label="PR106x_WR01",
        output_dir=out_dir,
        target_sections=("latents_and_sidecar_brotli",),
        seed=123,
        atom_budget=7,
        block_size=16,
        quant_step=2.0,
        expected_source_archive_sha256=source.archive_sha256,
        expected_source_archive_bytes=source.archive_bytes,
    )

    manifest_path = out_dir / "hnerv_wavelet_compress_time_harness.json"
    assert manifest == repeat
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest
    assert manifest["score_claim"] is False
    assert manifest["dispatch_attempted"] is False
    assert manifest["ready_for_compress_time_training"] is False
    assert manifest["ready_for_wavelet_sidechannel_candidate"] is False
    assert manifest["ready_for_archive_preflight"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert "missing_exact_decode_validation_manifest" in manifest["blockers"]
    assert "compress_time_atom_training_not_implemented" in manifest["blockers"]
    assert "compress_time_harness_scaffold_only" in manifest["dispatch_blockers"]

    input_manifest = manifest["input_manifest"]
    output_manifest = manifest["output_manifest"]
    assert input_manifest["schema"] == "hnerv_wavelet_compress_time_input.v1"
    assert output_manifest["schema"] == "hnerv_wavelet_compress_time_output.v1"
    assert input_manifest["source"]["source_archive_custody_mode"] == ("operator_expected_archive_identity_verified")
    assert input_manifest["config"]["seed"] == 123
    assert input_manifest["config"]["atom_budget"] == 7
    assert input_manifest["config"]["rng_state_mutated"] is False
    assert output_manifest["trained_atoms_manifest_path"] is None
    assert output_manifest["wavelet_sidechannel_archive_path"] is None
    assert output_manifest["applied_candidate_archive_path"] is None
    assert output_manifest["decode_validation"]["fail_closed"] is True

    section = manifest["source_sections"][0]
    assert section["section_name"] == "latents_and_sidecar_brotli"
    assert section["decode_probe_status"] == "local_brotli_decode_only_not_exact_validation"
    assert section["raw_bytes"] > section["section_bytes"]
    assert section["score_claim"] is False
    assert len(manifest["config_sha256"]) == 64
    assert len(manifest["manifest_sha256_excluding_self"]) == 64


def test_wavelet_compress_time_harness_rejects_source_identity_mismatch(
    tmp_path: Path,
) -> None:
    archive = _source_archive(tmp_path)
    source = read_strict_single_member_zip(archive)

    with pytest.raises(HnervWaveletCompressTimeHarnessError, match="does not match"):
        build_wavelet_compress_time_harness(
            source_archive=archive,
            source_label="PR106x_WR01",
            output_dir=tmp_path / "harness",
            expected_source_archive_sha256="0" * 64,
            expected_source_archive_bytes=source.archive_bytes,
        )


def test_build_hnerv_wavelet_compress_time_harness_cli_writes_same_manifest(
    tmp_path: Path,
) -> None:
    archive = _source_archive(tmp_path)
    source = read_strict_single_member_zip(archive)
    out_dir = tmp_path / "harness"
    json_out = tmp_path / "harness_manifest.json"

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_hnerv_wavelet_compress_time_harness.py"),
            "--source-archive",
            str(archive),
            "--source-label",
            "PR106x_WR01",
            "--output-dir",
            str(out_dir),
            "--target-section",
            "latents_and_sidecar_brotli",
            "--seed",
            "123",
            "--atom-budget",
            "7",
            "--block-size",
            "16",
            "--quant-step",
            "2.0",
            "--expected-source-archive-sha256",
            source.archive_sha256,
            "--expected-source-archive-bytes",
            str(source.archive_bytes),
            "--json-out",
            str(json_out),
        ],
        check=True,
        text=True,
    )

    payload = json.loads(json_out.read_text(encoding="utf-8"))
    persisted = json.loads((out_dir / "hnerv_wavelet_compress_time_harness.json").read_text(encoding="utf-8"))
    assert payload == persisted
    assert payload["score_claim"] is False
    assert payload["dispatch_attempted"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["output_manifest"]["decode_validation"]["exact_validation_available"] is False


def _source_archive(tmp_path: Path) -> Path:
    decoder = brotli.compress(bytes(range(251)) * 40, quality=1)
    latents = brotli.compress((b"alpha-wavelet-compress-time-signal-" * 80), quality=1)
    payload = bytes([0xFF]) + len(decoder).to_bytes(3, "little") + decoder + latents
    archive = tmp_path / "source.zip"
    write_stored_single_member_zip(archive, member_name="x", payload=payload)
    return archive
