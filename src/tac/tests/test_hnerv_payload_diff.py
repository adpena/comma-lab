# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import brotli

from tac.hnerv_lowlevel_packer import (
    PackedHnervPayload,
    parse_ff_packed_brotli_hnerv,
    read_strict_single_member_zip,
    sha256_bytes,
    write_stored_single_member_zip,
)
from tac.hnerv_payload_diff import build_hnerv_payload_diff

REPO = Path(__file__).resolve().parents[3]


def test_hnerv_payload_diff_records_compressed_and_raw_section_change(tmp_path: Path) -> None:
    source_archive = tmp_path / "source.zip"
    candidate_archive = tmp_path / "candidate.zip"
    source_raw = bytes(range(64))
    candidate_raw = bytes([source_raw[0] + 1]) + source_raw[1:]
    source_payload = PackedHnervPayload(
        header=b"",
        decoder_packed_brotli=brotli.compress(b"decoder", quality=11),
        latents_and_sidecar_brotli=brotli.compress(source_raw, quality=11),
    ).to_bytes()
    candidate_payload = PackedHnervPayload(
        header=b"",
        decoder_packed_brotli=parse_ff_packed_brotli_hnerv(source_payload).decoder_packed_brotli,
        latents_and_sidecar_brotli=brotli.compress(candidate_raw, quality=11),
    ).to_bytes()
    write_stored_single_member_zip(source_archive, member_name="x", payload=source_payload)
    write_stored_single_member_zip(candidate_archive, member_name="x", payload=candidate_payload)
    source = read_strict_single_member_zip(source_archive)

    payload = build_hnerv_payload_diff(
        source_archive,
        candidate_archive,
        source_label="source",
        candidate_label="candidate",
        source_manifest=_manifest(source),
    )

    assert payload["score_claim"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    assert payload["ready_for_archive_preflight"] is True
    assert payload["changed_section_count"] == 1
    sections = {row["name"]: row for row in payload["sections"]}
    assert sections["packed_header_ff_len24"]["changed"] is False
    assert sections["latents_and_sidecar_brotli"]["raw_equal"] is False
    assert sections["latents_and_sidecar_brotli"]["raw_changed_positions"] == 1
    assert sections["latents_and_sidecar_brotli"]["raw_abs_delta_sum"] == 1


def test_hnerv_payload_diff_blocks_noop(tmp_path: Path) -> None:
    archive = tmp_path / "source.zip"
    payload_bytes = PackedHnervPayload(
        header=b"",
        decoder_packed_brotli=brotli.compress(b"decoder", quality=11),
        latents_and_sidecar_brotli=brotli.compress(b"latents", quality=11),
    ).to_bytes()
    write_stored_single_member_zip(archive, member_name="x", payload=payload_bytes)

    payload = build_hnerv_payload_diff(archive, archive)

    assert payload["ready_for_archive_preflight"] is False
    assert "no_payload_section_changed" in payload["blockers"]


def test_compare_hnerv_payload_sections_cli(tmp_path: Path) -> None:
    source_archive = tmp_path / "source.zip"
    candidate_archive = tmp_path / "candidate.zip"
    source_payload = PackedHnervPayload(
        header=b"",
        decoder_packed_brotli=brotli.compress(b"decoder", quality=11),
        latents_and_sidecar_brotli=brotli.compress(b"latents-source", quality=11),
    ).to_bytes()
    candidate_payload = PackedHnervPayload(
        header=b"",
        decoder_packed_brotli=parse_ff_packed_brotli_hnerv(source_payload).decoder_packed_brotli,
        latents_and_sidecar_brotli=brotli.compress(b"latents-target", quality=11),
    ).to_bytes()
    write_stored_single_member_zip(source_archive, member_name="x", payload=source_payload)
    write_stored_single_member_zip(candidate_archive, member_name="x", payload=candidate_payload)
    source = read_strict_single_member_zip(source_archive)
    scorecard = tmp_path / "scorecard.json"
    out = tmp_path / "diff.json"
    scorecard.write_text(
        json.dumps({"payload_section_manifests": [_manifest(source, label="PR106x")]}),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "compare_hnerv_payload_sections.py"),
            "--source-archive",
            str(source_archive),
            "--candidate-archive",
            str(candidate_archive),
            "--source-label",
            "PR106x",
            "--candidate-label",
            "candidate",
            "--source-manifest-json",
            str(scorecard),
            "--json-out",
            str(out),
            "--fail-if-no-section-change",
        ],
        check=True,
        text=True,
    )

    payload = json.loads(out.read_text())
    assert payload["changed_section_count"] >= 1
    assert payload["source_label"] == "PR106x"


def _manifest(source, label: str = "source") -> dict:
    parsed = parse_ff_packed_brotli_hnerv(source.payload)
    sections = []
    start = 0
    for index, (name, data) in enumerate(
        [
            ("packed_header_ff_len24", parsed.header),
            ("decoder_packed_brotli", parsed.decoder_packed_brotli),
            ("latents_and_sidecar_brotli", parsed.latents_and_sidecar_brotli),
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
            }
        )
        start = end
    return {
        "label": label,
        "archive_sha256": source.archive_sha256,
        "archive_bytes": source.archive_bytes,
        "zip_member": source.member_name,
        "payload_sha256": sha256_bytes(source.payload),
        "member_bytes": source.member_bytes,
        "sections": sections,
    }
