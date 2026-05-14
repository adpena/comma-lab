# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import brotli

from tac.hnerv_lowlevel_packer import (
    parse_ff_packed_brotli_hnerv,
    read_strict_single_member_zip,
    sha256_bytes,
    write_stored_single_member_zip,
)
from tac.hnerv_wavelet_residual import build_wavelet_residual_plan, plan_digest

REPO = Path(__file__).resolve().parents[3]


def test_build_wavelet_residual_plan_uses_brotli_raw_domain(tmp_path: Path) -> None:
    archive = _source_archive(tmp_path)
    source = read_strict_single_member_zip(archive)
    scorecard = _scorecard(source, "PR106x")

    plan = build_wavelet_residual_plan(
        source_archive=str(archive),
        scorecard=scorecard,
        source_label="PR106x",
        target_sections=("latents_and_sidecar_brotli",),
        top_k=8,
        block_size=16,
        quant_step=1.0,
    )

    assert plan["score_claim"] is False
    assert plan["dispatch_attempted"] is False
    assert plan["ready_for_wavelet_candidate_build"] is True
    assert plan["ready_for_archive_preflight"] is False
    assert plan["ready_for_exact_eval_dispatch"] is False
    assert "requires_old_new_section_sha256_proof" in plan["dispatch_blockers"]
    assert plan["blockers"] == []
    section = plan["sections"][0]
    assert section["section_name"] == "latents_and_sidecar_brotli"
    assert section["transform_domain"] == "brotli_decompressed_section"
    assert section["raw_bytes"] > section["source_section_bytes"]
    assert section["atom_count"] == 8
    assert section["estimated_atom_bytes"] > 0
    assert section["atoms"] == sorted(
        section["atoms"],
        key=lambda atom: (
            -atom["abs_coefficient_quantized"],
            atom["raw_offset"],
            atom["level"],
            atom["coefficient_index"],
        ),
    )
    assert len(plan_digest(plan)) == 64


def test_build_wavelet_residual_plan_fails_closed_on_stale_scorecard(tmp_path: Path) -> None:
    archive = _source_archive(tmp_path)
    source = read_strict_single_member_zip(archive)
    scorecard = _scorecard(source, "PR106x")
    scorecard["payload_section_manifests"][0]["sections"][2]["sha256"] = "0" * 64

    plan = build_wavelet_residual_plan(
        source_archive=str(archive),
        scorecard=scorecard,
        source_label="PR106x",
        target_sections=("latents_and_sidecar_brotli",),
        top_k=4,
        block_size=16,
    )

    assert plan["ready_for_wavelet_candidate_build"] is False
    assert "manifest_section_sha256_mismatch:latents_and_sidecar_brotli" in plan["blockers"]


def test_plan_hnerv_wavelet_residual_cli(tmp_path: Path) -> None:
    archive = _source_archive(tmp_path)
    source = read_strict_single_member_zip(archive)
    scorecard = tmp_path / "scorecard.json"
    json_out = tmp_path / "wavelet_plan.json"
    scorecard.write_text(json.dumps(_scorecard(source, "PR106x")), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "plan_hnerv_wavelet_residual.py"),
            "--source-archive",
            str(archive),
            "--scorecard",
            str(scorecard),
            "--source-label",
            "PR106x",
            "--target-section",
            "latents_and_sidecar_brotli",
            "--top-k",
            "6",
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
    assert payload["ready_for_wavelet_candidate_build"] is True
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert len(payload["plan_sha256"]) == 64
    assert payload["sections"][0]["atom_count"] == 6


def _source_archive(tmp_path: Path) -> Path:
    decoder = brotli.compress(bytes(range(251)) * 40, quality=1)
    latents = brotli.compress((b"latent-wavelet-residual-signal-" * 80), quality=1)
    payload = bytes([0xFF]) + len(decoder).to_bytes(3, "little") + decoder + latents
    archive = tmp_path / "source.zip"
    write_stored_single_member_zip(archive, member_name="x", payload=payload)
    return archive


def _scorecard(source, label: str) -> dict:
    packed = parse_ff_packed_brotli_hnerv(source.payload)
    sections = []
    start = 0
    for index, (name, data, role) in enumerate(
        [
            ("packed_header_ff_len24", packed.header, "control_or_metadata"),
            ("decoder_packed_brotli", packed.decoder_packed_brotli, "decoder_weight_stream"),
            ("latents_and_sidecar_brotli", packed.latents_and_sidecar_brotli, "latent_stream"),
        ]
    ):
        end = start + len(data)
        sections.append(
            {
                "index": index,
                "name": name,
                "start": start,
                "end": end,
                "bytes": len(data),
                "sha256": sha256_bytes(data),
                "entropy_bits_per_byte": 7.0,
                "optimization_role": role,
            }
        )
        start = end
    return {
        "schema_version": 1,
        "tool": "build_hnerv_frontier_scorecard",
        "score_truth": "byte_fixture",
        "payload_section_manifests": [
            {
                "label": label,
                "archive_sha256": source.archive_sha256,
                "archive_bytes": source.archive_bytes,
                "zip_member": source.member_name,
                "payload_sha256": sha256_bytes(source.payload),
                "member_bytes": source.member_bytes,
                "profile_match_key": "member_sha256",
                "score_claim": False,
                "dispatch_attempted": False,
                "sections": sections,
            }
        ],
    }
