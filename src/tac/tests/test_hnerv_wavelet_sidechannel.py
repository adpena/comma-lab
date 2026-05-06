from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import brotli

from tac.hnerv_lowlevel_packer import read_strict_single_member_zip, write_stored_single_member_zip
from tac.hnerv_wavelet_sidechannel import (
    build_wavelet_sidechannel_candidate,
    decode_wavelet_atom_sidechannel,
    parse_wavelet_sidechannel_archive_bytes,
)
from tac.tests.test_hnerv_wavelet_residual import _scorecard

REPO = Path(__file__).resolve().parents[3]


def test_build_wavelet_sidechannel_candidate_preserves_source_and_consumes_atoms(tmp_path: Path) -> None:
    archive = _source_archive(tmp_path)
    source = read_strict_single_member_zip(archive)
    scorecard = _scorecard(source, "PR106x")

    manifest = build_wavelet_sidechannel_candidate(
        source_archive=archive,
        scorecard=scorecard,
        source_label="PR106x",
        output_dir=tmp_path / "candidate",
        target_sections=("latents_and_sidecar_brotli",),
        top_k=6,
        block_size=16,
    )

    assert manifest["score_claim"] is False
    assert manifest["dispatch_attempted"] is False
    assert manifest["ready_for_wavelet_sidechannel_candidate"] is True
    assert manifest["ready_for_archive_preflight"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["blockers"] == []
    assert "candidate_sidechannel_not_applied_by_inflate_runtime" in manifest["dispatch_blockers"]
    assert manifest["candidate_payload_contains_source_payload"] is True
    assert manifest["candidate_payload_byte_delta"] > 0

    candidate = read_strict_single_member_zip(manifest["candidate_archive_path"])
    assert candidate.archive_sha256 == manifest["candidate_archive_sha256"]
    assert candidate.archive_sha256 != source.archive_sha256
    parsed = parse_wavelet_sidechannel_archive_bytes(candidate.payload)
    assert parsed.source_payload == source.payload
    decoded = decode_wavelet_atom_sidechannel(parsed.sidechannel_blob)
    assert decoded["total_atom_count"] == 6
    assert decoded == manifest["decoded_wavelet_sidechannel"]
    assert manifest["runtime_consumption_proof"]["runtime_consumed"] is True
    assert manifest["runtime_consumption_proof"]["decoded_atom_count"] == 6


def test_build_wavelet_sidechannel_candidate_fails_closed_on_stale_scorecard(tmp_path: Path) -> None:
    archive = _source_archive(tmp_path)
    source = read_strict_single_member_zip(archive)
    scorecard = _scorecard(source, "PR106x")
    scorecard["payload_section_manifests"][0]["sections"][2]["sha256"] = "0" * 64

    manifest = build_wavelet_sidechannel_candidate(
        source_archive=archive,
        scorecard=scorecard,
        source_label="PR106x",
        output_dir=tmp_path / "candidate",
        target_sections=("latents_and_sidecar_brotli",),
        top_k=4,
        block_size=16,
    )

    assert manifest["ready_for_wavelet_sidechannel_candidate"] is False
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["candidate_archive_path"] is None
    assert "manifest_section_sha256_mismatch:latents_and_sidecar_brotli" in manifest["blockers"]


def test_build_hnerv_wavelet_sidechannel_candidate_cli(tmp_path: Path) -> None:
    archive = _source_archive(tmp_path)
    source = read_strict_single_member_zip(archive)
    scorecard_path = tmp_path / "scorecard.json"
    json_out = tmp_path / "candidate_manifest.json"
    scorecard_path.write_text(json.dumps(_scorecard(source, "PR106x")), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_hnerv_wavelet_sidechannel_candidate.py"),
            "--source-archive",
            str(archive),
            "--scorecard",
            str(scorecard_path),
            "--source-label",
            "PR106x",
            "--output-dir",
            str(tmp_path / "candidate"),
            "--target-section",
            "latents_and_sidecar_brotli",
            "--top-k",
            "5",
            "--block-size",
            "16",
            "--json-out",
            str(json_out),
            "--fail-if-blocked",
        ],
        check=True,
        text=True,
    )

    payload = json.loads(json_out.read_text())
    assert payload["ready_for_wavelet_sidechannel_candidate"] is True
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["decoded_wavelet_sidechannel"]["total_atom_count"] == 5


def _source_archive(tmp_path: Path) -> Path:
    decoder = brotli.compress(bytes(range(251)) * 40, quality=1)
    latents = brotli.compress((b"latent-wavelet-residual-signal-" * 80), quality=1)
    payload = bytes([0xFF]) + len(decoder).to_bytes(3, "little") + decoder + latents
    archive = tmp_path / "source.zip"
    write_stored_single_member_zip(archive, member_name="x", payload=payload)
    return archive
