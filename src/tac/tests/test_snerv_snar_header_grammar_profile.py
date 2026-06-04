# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

from tac.analysis.snerv_snar_header_grammar_profile import (
    SCHEMA,
    build_snerv_snar_header_grammar_profile,
)
from tac.substrates.snerv_inverse_steg_carrier.archive import pack_snerv_archive
from tools import profile_snerv_snar_header_grammar as cli


def test_snerv_snar_header_grammar_profile_accounts_header_dominance(
    tmp_path: Path,
) -> None:
    packet = pack_snerv_archive(
        metadata_payload=b"m",
        lf_payload=b"l",
        decoder_payload=b"d",
        step_map_packet=b"s",
        metadata={
            "small": 1,
            "dominant_manifest": [
                {"tensor": index, "sha256": "a" * 64, "shape": [12, 16]}
                for index in range(64)
            ],
        },
    )
    packet_path = tmp_path / "candidate.snar"
    packet_path.write_bytes(packet.packet)

    report = build_snerv_snar_header_grammar_profile(
        input_path=packet_path,
        hard_byte_ceilings=(packet.total_bytes - 20,),
        generated_utc="2026-06-04T00:00:00+00:00",
    )

    assert report["schema"] == SCHEMA
    assert report["score_claim"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["packet"]["schema_valid"] is True
    assert report["header"]["bytes"] == packet.header_bytes
    assert report["payload"]["section_total_bytes"] == sum(
        packet.section_bytes.values()
    )
    assert report["payload"]["unreferenced_payload_bytes"] == 0
    assert report["byte_accounting"]["header_dominates_sections"] is True
    assert report["hard_byte_ceiling_rows"][0]["packet_over_ceiling_bytes"] == 20
    assert report["hard_byte_ceiling_rows"][0]["header_bytes_can_cover_overrun"] is True
    assert report["header_rewrite_needed_for_any_ceiling"] is True
    assert "snerv_snar_packet_header_grammar_rewrite_required" in report["blockers"]
    assert report["header"]["metadata_top_contributor_rows"][0]["path"] == (
        "$.metadata"
    )
    assert any(
        row["path"] == "$.metadata.dominant_manifest"
        for row in report["header"]["metadata_top_contributor_rows"]
    )


def test_profile_snerv_snar_header_grammar_cli_writes_json(tmp_path: Path) -> None:
    packet = pack_snerv_archive(
        metadata_payload=b"m",
        lf_payload=b"l",
        decoder_payload=b"d",
        step_map_packet=b"s",
        metadata={"dominant": ["x" * 32 for _ in range(16)]},
    )
    packet_path = tmp_path / "candidate.snar"
    output_json = tmp_path / "profile.json"
    packet_path.write_bytes(packet.packet)

    assert cli.main(
        [
            str(packet_path),
            "--output-json",
            str(output_json),
            "--hard-byte-ceiling",
            str(packet.total_bytes - 1),
        ]
    ) == 0

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["schema"] == SCHEMA
    assert payload["packet"]["bytes"] == packet.total_bytes
    assert payload["header"]["bytes"] == packet.header_bytes
    assert payload["header_rewrite_needed_for_any_ceiling"] is True
