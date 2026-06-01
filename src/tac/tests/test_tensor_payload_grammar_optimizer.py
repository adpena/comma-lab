# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tac.optimization.byte_shaving_campaign import build_signal_surface_from_candidate_queue
from tac.packet_compiler.tensor_payload_grammar_optimizer import (
    TENSOR_PAYLOAD_GRAMMAR_CANDIDATE_SCHEMA,
    TENSOR_PAYLOAD_GRAMMAR_OPTIMIZER_SCHEMA,
    TENSOR_PAYLOAD_GRAMMAR_QUEUE_SCHEMA,
    TENSOR_PAYLOAD_SOURCE_MANIFEST_SCHEMA,
    build_tensor_payload_optimizer_queue,
    measure_tensor_payload_candidates,
    quantize_tensor_symmetric_int8,
    select_best_tensor_payload_candidate,
    solve_tensor_payload_grammar,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_quantize_tensor_symmetric_int8_rejects_nonfinite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        quantize_tensor_symmetric_int8(np.array([0.0, np.nan], dtype=np.float32))


def test_tensor_payload_candidates_are_roundtrip_and_fail_closed() -> None:
    q = np.array([0, 0, 0, 1, -1, 2, -2, 8, -8], dtype=np.int8)

    candidates = measure_tensor_payload_candidates(
        q,
        tensor_index=3,
        tensor_name="decoder.block.weight",
        scale=0.125,
        scale_dtypes=("fp16", "fp32"),
        storage_perm_mode="identity",
        byte_maps=("zig", "off"),
        coders=("brotli", "canonical_huffman"),
        brotli_quality=4,
    )

    assert candidates
    assert all(row["schema"] == TENSOR_PAYLOAD_GRAMMAR_CANDIDATE_SCHEMA for row in candidates)
    assert all(row["roundtrip_exact"] for row in candidates if row["status"] == "ok")
    assert {row["scale_dtype"] for row in candidates} == {"fp16", "fp32"}
    selected = select_best_tensor_payload_candidate(candidates)
    assert selected["charged_bytes"] == min(
        row["charged_bytes"] for row in candidates if row["status"] == "ok"
    )
    assert selected["score_claim"] is False
    assert selected["promotion_eligible"] is False
    assert selected["ready_for_exact_eval_dispatch"] is False
    assert (
        selected["runtime_consumption_status"]
        == "generic_tensor_payload_receiver_required"
    )
    assert "generic_tensor_payload_receiver_not_bound" in selected["blockers"]


def test_generic_tensor_payload_solver_emits_queue_consumable_signal() -> None:
    tensors = {
        "constant.bias": np.zeros(256, dtype=np.float32),
        "structured.weight": np.linspace(-0.5, 0.5, 64, dtype=np.float32).reshape(4, 4, 4),
    }

    report = solve_tensor_payload_grammar(
        tensors,
        storage_perm_mode="identity",
        coders=("brotli", "canonical_huffman"),
        brotli_quality=4,
        baseline_coder="canonical_huffman",
        campaign_id="generic_tensor_smoke",
    )
    queue = build_tensor_payload_optimizer_queue(report)
    surface = build_signal_surface_from_candidate_queue(queue)

    assert report["schema"] == TENSOR_PAYLOAD_GRAMMAR_OPTIMIZER_SCHEMA
    assert report["source_payload_manifest"]["schema"] == TENSOR_PAYLOAD_SOURCE_MANIFEST_SCHEMA
    assert report["tensor_count"] == 2
    assert report["score_claim"] is False
    assert report["promotion_eligible"] is False
    assert report["byte_accounting"]["selected_isolated_tensor_bytes"] > 0
    order_diag = report["grouped_brotli_order_diagnostic"]
    assert order_diag["schema"] == "tensor_payload_grouped_brotli_order_diagnostic.v1"
    assert order_diag["candidate_count"] >= 1
    assert order_diag["selected_grouped_brotli_bytes"] > 0
    assert "grouped_delta_bytes_vs_identity" in order_diag
    assert "grouped_delta_bytes_vs_selected_isolated" in order_diag
    assert report["planner_feedback"]["operation_hint_count"] == 2
    assert report["planner_feedback"]["grouped_brotli_order_hint"][
        "selected_grouped_brotli_bytes"
    ] == order_diag["selected_grouped_brotli_bytes"]
    assert queue["schema"] == TENSOR_PAYLOAD_GRAMMAR_QUEUE_SCHEMA
    assert queue["candidate_count"] >= 2
    if order_diag["grouped_saved_bytes_vs_selected_isolated"] > 0:
        grouped_rows = [
            row
            for row in queue["candidates"]
            if row["operation_family"] == "tensor_payload_grouped_brotli_order"
        ]
        assert len(grouped_rows) == 1
        assert grouped_rows[0]["candidate_saved_bytes"] == order_diag[
            "grouped_saved_bytes_vs_selected_isolated"
        ]
    assert queue["ready_for_exact_eval_dispatch"] is False
    assert "generic_tensor_payload_receiver_not_bound" in queue["blockers"]
    assert surface["score_claim"] is False
    assert surface["ready_for_exact_eval_dispatch"] is False
    assert len(surface["units"]) >= 1


def test_tensor_payload_optimizer_cli_reads_npz_and_writes_report(
    tmp_path: Path,
) -> None:
    npz_path = tmp_path / "weights.npz"
    report_path = tmp_path / "report.json"
    queue_path = tmp_path / "queue.json"
    np.savez(
        npz_path,
        bias=np.zeros(32, dtype=np.float32),
        weight=np.arange(64, dtype=np.float32).reshape(4, 4, 4) / 64.0,
    )

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "tools" / "tensor_payload_grammar_optimizer.py"),
            "--npz",
            str(npz_path),
            "--output",
            str(report_path),
            "--queue-output",
            str(queue_path),
            "--campaign-id",
            "cli_tensor_payload_grammar",
            "--storage-perm-mode",
            "identity",
            "--coders",
            "brotli,canonical_huffman",
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
    assert stdout["source_kind"] == "npz"
    assert stdout["tensor_count"] == 2
    report = json.loads(report_path.read_text(encoding="utf-8"))
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    assert report["schema"] == TENSOR_PAYLOAD_GRAMMAR_OPTIMIZER_SCHEMA
    assert report["source_payload_manifest"]["source_path"] == npz_path.as_posix()
    assert (
        report["grouped_brotli_order_diagnostic"]["schema"]
        == "tensor_payload_grouped_brotli_order_diagnostic.v1"
    )
    assert {row["tensor_name"] for row in report["rows"]} == {"bias", "weight"}
    assert queue["schema"] == TENSOR_PAYLOAD_GRAMMAR_QUEUE_SCHEMA


def test_grouped_brotli_order_savings_emit_queue_candidate() -> None:
    report = {
        "schema": TENSOR_PAYLOAD_GRAMMAR_OPTIMIZER_SCHEMA,
        "campaign_id": "grouped_fixture",
        "planner_feedback": {"operation_hints": []},
        "grouped_brotli_order_diagnostic": {
            "schema": "tensor_payload_grouped_brotli_order_diagnostic.v1",
            "selected_order_label": "histogram_greedy",
            "selected_tensor_order": ["b", "a"],
            "selected_grouped_brotli_bytes": 83,
            "selected_isolated_tensor_bytes": 100,
            "identity_grouped_brotli_bytes": 100,
            "grouped_delta_bytes_vs_identity": -17,
            "grouped_saved_bytes_vs_identity": 17,
            "grouped_delta_bytes_vs_selected_isolated": -17,
            "grouped_saved_bytes_vs_selected_isolated": 17,
        },
    }

    queue = build_tensor_payload_optimizer_queue(report)

    assert queue["candidate_count"] == 1
    assert queue["top_k"][0]["operation_family"] == "tensor_payload_grouped_brotli_order"
    assert queue["top_k"][0]["candidate_saved_bytes"] == 17
    assert queue["top_k"][0]["predicted_delta_bytes"] == -17
    assert queue["top_k"][0]["operation_params"]["selected_tensor_order"] == ["b", "a"]
    assert queue["score_claim"] is False
    assert queue["ready_for_exact_eval_dispatch"] is False
