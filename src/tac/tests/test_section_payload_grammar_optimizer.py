# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
import zipfile
from io import BytesIO
from pathlib import Path

import pytest

from tac.optimization.byte_shaving_campaign import build_signal_surface_from_candidate_queue
from tac.packet_compiler.pr101_per_tensor_grammar_solver import DEFAULT_CODERS
from tac.packet_compiler.section_payload_grammar_optimizer import (
    SECTION_PAYLOAD_GRAMMAR_OPTIMIZER_SCHEMA,
    SECTION_PAYLOAD_GRAMMAR_QUEUE_SCHEMA,
    SECTION_PAYLOAD_SOURCE_MANIFEST_SCHEMA,
    build_section_payload_optimizer_queue,
    measure_section_coder_candidates,
    sections_from_single_member_zip_archive,
    select_best_section_candidate,
    solve_section_payload_grammar,
    spans_from_archive_section_telemetry,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _stored_zip(member_name: str, payload: bytes) -> bytes:
    buf = BytesIO()
    info = zipfile.ZipInfo(member_name)
    info.compress_type = zipfile.ZIP_STORED
    info.date_time = (1980, 1, 1, 0, 0, 0)
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(info, payload)
    return buf.getvalue()


def _stored_zip_members(members: dict[str, bytes]) -> bytes:
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, payload in members.items():
            info = zipfile.ZipInfo(name)
            info.compress_type = zipfile.ZIP_STORED
            info.date_time = (1980, 1, 1, 0, 0, 0)
            zf.writestr(info, payload)
    return buf.getvalue()


def test_section_coder_candidates_reuse_shared_portfolio_fail_closed() -> None:
    payload = bytes([0]) * 512 + bytes(range(32)) * 4

    candidates = measure_section_coder_candidates(
        payload,
        section_name="decoder.weights",
        coders=("brotli", "canonical_huffman", "lzma_raw"),
        brotli_quality=4,
    )

    assert candidates
    assert {row["coder"] for row in candidates} == {
        "brotli",
        "canonical_huffman",
        "lzma_raw",
    }
    assert all(
        row["schema"] == "section_payload_grammar_candidate.v1"
        for row in candidates
    )
    assert all(row["roundtrip_exact"] for row in candidates if row["status"] == "ok")
    selected = select_best_section_candidate(candidates)
    assert selected["charged_bytes"] == min(
        row["charged_bytes"] for row in candidates if row["status"] == "ok"
    )
    assert selected["score_claim"] is False
    assert selected["promotion_eligible"] is False
    assert selected["ready_for_exact_eval_dispatch"] is False
    assert "isolated_section_measurement_not_archive_authority" in selected["blockers"]


def test_section_payload_solver_reports_saturation_and_planner_hints() -> None:
    report = solve_section_payload_grammar(
        {
            "zeros": b"\x00" * 1024,
            "pattern": (b"abc123" * 200),
        },
        coders=("brotli", "canonical_huffman", "lzma_raw"),
        brotli_quality=4,
        campaign_id="unit_section_grammar",
    )

    assert report["schema"] == SECTION_PAYLOAD_GRAMMAR_OPTIMIZER_SCHEMA
    assert report["campaign_id"] == "unit_section_grammar"
    assert report["section_count"] == 2
    assert report["coders"] == ["brotli", "canonical_huffman", "lzma_raw"]
    assert report["byte_accounting"]["selected_isolated_section_bytes"] > 0
    assert report["saturation_diagnostic"]["status"] in {
        "entropy_saturated",
        "weak_entropy_gap",
        "unsaturated_entropy_gap",
        "floor_unavailable",
    }
    assert report["planner_feedback"]["operation_hint_count"] == 2
    assert report["planner_feedback"]["posterior_update_hooks"]
    assert report["grouped_brotli_order_diagnostic"]["candidate_count"] == 2
    assert "identity_grouped_brotli_bytes" in report["grouped_brotli_order_diagnostic"]
    assert "selected_isolated_section_bytes" in report["grouped_brotli_order_diagnostic"]
    assert "grouped_delta_bytes_vs_selected_isolated" in report[
        "grouped_brotli_order_diagnostic"
    ]
    assert all(row["selected"]["roundtrip_exact"] for row in report["rows"])
    assert report["score_claim"] is False
    assert report["promotable"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert "runtime_consumption_proof_missing" in report["blockers"]


def test_section_payload_queue_is_planning_only_and_consumable() -> None:
    report = solve_section_payload_grammar(
        {"latents": b"\x01\x02\x03\x04" * 128},
        coders=("brotli", "canonical_huffman"),
        brotli_quality=4,
        baseline_coder="canonical_huffman",
        campaign_id="latents_section_grammar",
    )

    queue = build_section_payload_optimizer_queue(report)

    assert queue["schema"] == SECTION_PAYLOAD_GRAMMAR_QUEUE_SCHEMA
    assert queue["campaign_id"] == "latents_section_grammar"
    assert queue["candidate_count"] == 1
    candidate = queue["candidates"][0]
    assert candidate["operation_family"] == "section_payload_coder_selection"
    assert candidate["consumer_payload"]["byte_accounting_scope"] == (
        "isolated_section_payload_not_archive_authority"
    )
    assert queue["score_claim"] is False
    assert queue["promotion_eligible"] is False
    assert queue["ready_for_provider_dispatch"] is False
    assert "byte_closed_archive_not_materialized" in queue["blockers"]

    surface = build_signal_surface_from_candidate_queue(queue)
    assert surface["score_claim"] is False
    assert surface["ready_for_exact_eval_dispatch"] is False
    assert len(surface["units"]) == 1
    assert all(
        unit["operation_families"] == ["section_payload_coder_selection"]
        for unit in surface["units"]
    )


def test_section_payload_queue_routes_grouped_brotli_order_when_positive() -> None:
    report = solve_section_payload_grammar(
        {"a": b"\x00" * 256, "b": bytes(range(64)) * 4},
        coders=("brotli",),
        brotli_quality=4,
        campaign_id="grouped_section_order",
    )
    # Force a positive grouped-order diagnostic without depending on Brotli's
    # exact version-specific byte count for the fixture above.
    report["grouped_brotli_order_diagnostic"] = {
        "schema": "section_payload_grouped_brotli_order_diagnostic.v1",
        "selected_order_label": "size_desc",
        "selected_section_order": ["b", "a"],
        "selected_grouped_brotli_bytes": 90,
        "identity_grouped_brotli_bytes": 120,
        "selected_isolated_section_bytes": 100,
        "grouped_delta_bytes_vs_identity": -30,
        "grouped_saved_bytes_vs_identity": 30,
        "grouped_delta_bytes_vs_selected_isolated": -10,
        "grouped_saved_bytes_vs_selected_isolated": 10,
    }

    queue = build_section_payload_optimizer_queue(report)

    grouped = [
        row
        for row in queue["candidates"]
        if row["operation_family"] == "section_payload_grouped_brotli_order"
    ]
    assert len(grouped) == 1
    assert grouped[0]["candidate_saved_bytes"] == 10
    assert grouped[0]["predicted_delta_bytes"] == -10
    assert grouped[0]["operation_params"]["selected_section_order"] == ["b", "a"]
    assert grouped[0]["score_claim"] is False
    assert grouped[0]["ready_for_exact_eval_dispatch"] is False


def test_single_member_zip_archive_sections_are_extracted_with_provenance() -> None:
    member_payload = b"A" * 64 + b"BC" * 32
    archive = _stored_zip("0.bin", member_payload)

    sections, manifest = sections_from_single_member_zip_archive(
        archive,
        member_name="0.bin",
        spans=[
            {"name": "alpha", "start": 0, "length": 64},
            {"name": "beta", "start": 64, "length": 64},
        ],
    )
    report = solve_section_payload_grammar(
        sections,
        coders=("brotli", "canonical_huffman"),
        brotli_quality=4,
        source_payload_manifest=manifest,
    )

    assert manifest["schema"] == SECTION_PAYLOAD_SOURCE_MANIFEST_SCHEMA
    assert manifest["zip_member_is_stored"] is True
    assert manifest["member_payload_bytes"] == len(member_payload)
    assert manifest["section_count"] == 2
    assert manifest["sections"][0]["name"] == "alpha"
    assert manifest["sections"][1]["bytes"] == 64
    assert manifest["blockers"] == []
    assert sections[0]["payload"] == b"A" * 64
    assert sections[1]["payload"] == b"BC" * 32
    assert report["source_payload_manifest"]["archive_zip_sha256"] == manifest[
        "archive_zip_sha256"
    ]
    assert report["blockers"]


def test_explicit_zip_member_sections_are_extracted_from_runtime_archive() -> None:
    member_payload = b"A" * 64 + b"BC" * 32
    archive = _stored_zip_members(
        {
            "0.bin": member_payload,
            "inflate.py": b"print('inflate')\n",
            "inflate.sh": b"#!/bin/sh\npython inflate.py \"$@\"\n",
        }
    )

    sections, manifest = sections_from_single_member_zip_archive(
        archive,
        member_name="0.bin",
        spans=[
            {"name": "alpha", "start": 0, "length": 64},
            {"name": "beta", "start": 64, "length": 64},
        ],
    )

    assert manifest["source_kind"] == "zip_archive_member"
    assert manifest["zip_member_count"] == 3
    assert manifest["zip_member_name"] == "0.bin"
    assert manifest["zip_overhead_scope"] == (
        "archive_zip_bytes_minus_selected_member_compress_size"
    )
    assert sections[0]["payload"] == b"A" * 64
    assert sections[1]["payload"] == b"BC" * 32


def test_archive_section_telemetry_converts_to_zip_spans() -> None:
    telemetry = {
        "schema": "hinerv_archive_section_telemetry.v1",
        "sections": [
            {
                "name": "hiv1_header",
                "role": "header",
                "offset": 0,
                "end_offset": 33,
                "bytes": 33,
                "sha256": "a" * 64,
            },
            {
                "name": "decoder_state",
                "role": "decoder",
                "offset": 33,
                "end_offset": 97,
                "bytes": 64,
                "sha256": "b" * 64,
                "codec": "int8_mixed",
            },
        ],
    }

    spans = spans_from_archive_section_telemetry(telemetry)

    assert spans == [
        {
            "schema": "section_payload_telemetry_span_adapter.v1",
            "name": "hiv1_header",
            "start": 0,
            "end": 33,
            "length": 33,
            "source_role": "header",
            "source_sha256": "a" * 64,
            "source_codec": None,
            "source_scale": None,
        },
        {
            "schema": "section_payload_telemetry_span_adapter.v1",
            "name": "decoder_state",
            "start": 33,
            "end": 97,
            "length": 64,
            "source_role": "decoder",
            "source_sha256": "b" * 64,
            "source_codec": "int8_mixed",
            "source_scale": None,
        },
    ]


def test_section_payload_solver_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError, match="duplicate section name"):
        solve_section_payload_grammar(
            [
                {"name": "same", "payload": b"a"},
                {"name": "same", "payload": b"b"},
            ],
            coders=("brotli",),
        )


def test_default_coder_universe_matches_pr101_shared_backend() -> None:
    report = solve_section_payload_grammar(
        {"small": b"\x00\x01\x02\x03"},
        brotli_quality=4,
    )

    assert tuple(report["coders"]) == tuple(DEFAULT_CODERS)
    observed = {row["coder"] for row in report["rows"][0]["top_candidates"]}
    assert observed == set(DEFAULT_CODERS)


def test_section_payload_optimizer_cli_writes_report_and_queue(tmp_path: Path) -> None:
    section_a = tmp_path / "a.bin"
    section_b = tmp_path / "b.bin"
    report_path = tmp_path / "report.json"
    queue_path = tmp_path / "queue.json"
    section_a.write_bytes(b"\x00" * 256)
    section_b.write_bytes(b"abcd" * 128)

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "section_payload_grammar_optimizer.py"),
            "--section",
            f"zeros={section_a}",
            "--section",
            f"pattern={section_b}",
            "--output",
            str(report_path),
            "--queue-output",
            str(queue_path),
            "--campaign-id",
            "cli_section_grammar",
            "--coder",
            "brotli",
            "--coder",
            "canonical_huffman",
            "--brotli-quality",
            "4",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    stdout = json.loads(result.stdout)
    assert stdout["ok"] is True
    report = json.loads(report_path.read_text(encoding="utf-8"))
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    assert report["schema"] == SECTION_PAYLOAD_GRAMMAR_OPTIMIZER_SCHEMA
    assert report["campaign_id"] == "cli_section_grammar"
    assert queue["schema"] == SECTION_PAYLOAD_GRAMMAR_QUEUE_SCHEMA
    assert queue["campaign_id"] == "cli_section_grammar"
    assert report["score_claim"] is False
    assert queue["score_claim"] is False


def test_section_payload_optimizer_cli_reads_single_member_zip_spans(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive.zip"
    report_path = tmp_path / "report.json"
    payload = b"\x00" * 128 + b"abcd" * 64
    archive.write_bytes(_stored_zip("0.bin", payload))

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "section_payload_grammar_optimizer.py"),
            "--zip-archive",
            str(archive),
            "--zip-member",
            "0.bin",
            "--zip-section",
            "zeros:0:128",
            "--zip-section",
            "pattern:128:256",
            "--output",
            str(report_path),
            "--campaign-id",
            "cli_zip_section_grammar",
            "--coder",
            "brotli",
            "--coder",
            "canonical_huffman",
            "--brotli-quality",
            "4",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    stdout = json.loads(result.stdout)
    assert stdout["source_kind"] == "single_member_zip_archive"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["schema"] == SECTION_PAYLOAD_GRAMMAR_OPTIMIZER_SCHEMA
    assert report["source_payload_manifest"]["zip_member_name"] == "0.bin"
    assert report["source_payload_manifest"]["section_count"] == 2
    assert {row["section_name"] for row in report["rows"]} == {"zeros", "pattern"}


def test_section_payload_optimizer_cli_reads_archive_section_telemetry_spans(
    tmp_path: Path,
) -> None:
    archive = tmp_path / "archive.zip"
    telemetry_path = tmp_path / "hi_nerv_archive_section_telemetry.json"
    report_path = tmp_path / "report.json"
    payload = b"H" * 33 + b"\x00" * 96 + b"meta" * 16
    archive.write_bytes(_stored_zip("0.bin", payload))
    telemetry_path.write_text(
        json.dumps(
            {
                "schema": "hinerv_archive_section_telemetry.v1",
                "sections": [
                    {
                        "name": "hiv1_header",
                        "role": "header",
                        "offset": 0,
                        "end_offset": 33,
                        "bytes": 33,
                    },
                    {
                        "name": "decoder_state",
                        "role": "decoder",
                        "offset": 33,
                        "end_offset": 129,
                        "bytes": 96,
                    },
                    {
                        "name": "meta_json",
                        "role": "metadata",
                        "offset": 129,
                        "end_offset": len(payload),
                        "bytes": len(payload) - 129,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "section_payload_grammar_optimizer.py"),
            "--zip-archive",
            str(archive),
            "--zip-member",
            "0.bin",
            "--archive-section-telemetry-json",
            str(telemetry_path),
            "--output",
            str(report_path),
            "--campaign-id",
            "cli_hinerv_telemetry_section_grammar",
            "--coder",
            "brotli",
            "--coder",
            "canonical_huffman",
            "--brotli-quality",
            "4",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = json.loads(report_path.read_text(encoding="utf-8"))
    manifest = report["source_payload_manifest"]
    assert manifest["archive_section_telemetry_path"] == telemetry_path.as_posix()
    assert manifest["archive_section_telemetry_schema"] == (
        "hinerv_archive_section_telemetry.v1"
    )
    assert manifest["section_count"] == 3
    assert {row["section_name"] for row in report["rows"]} == {
        "hiv1_header",
        "decoder_state",
        "meta_json",
    }
