# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.packet_compiler.pr101_per_tensor_grammar_solver import DEFAULT_CODERS
from tac.packet_compiler.section_payload_grammar_optimizer import (
    SECTION_PAYLOAD_GRAMMAR_OPTIMIZER_SCHEMA,
    SECTION_PAYLOAD_GRAMMAR_QUEUE_SCHEMA,
    build_section_payload_optimizer_queue,
    measure_section_coder_candidates,
    select_best_section_candidate,
    solve_section_payload_grammar,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


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
