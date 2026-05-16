# SPDX-License-Identifier: MIT
"""Tests for the PR106 PacketIR candidate evidence matrix."""

from __future__ import annotations

import json
from pathlib import Path

from tac.packet_compiler.pr106_candidate_matrix import (
    PR106_PACKETIR_CANDIDATE_MATRIX_SCHEMA,
    build_pr106_packetir_candidate_matrix,
    render_pr106_packetir_candidate_matrix_markdown,
    write_pr106_packetir_candidate_matrix,
)


def _row_by_id(matrix: dict[str, object], candidate_id: str) -> dict[str, object]:
    rows = matrix["rows"]
    assert isinstance(rows, list)
    for row in rows:
        assert isinstance(row, dict)
        if row["candidate_id"] == candidate_id:
            return row
    raise AssertionError(candidate_id)


def test_pr106_packetir_candidate_matrix_covers_active_candidates() -> None:
    matrix = build_pr106_packetir_candidate_matrix()

    assert matrix["schema"] == PR106_PACKETIR_CANDIDATE_MATRIX_SCHEMA
    assert matrix["candidate_count"] >= 15
    assert matrix["score_claim"] is False
    assert matrix["promotion_eligible"] is False
    assert matrix["ready_for_exact_eval_dispatch"] is False
    rows = matrix["rows"]
    assert isinstance(rows, list)
    assert all(row["score_claim"] is False for row in rows if isinstance(row, dict))
    assert all(
        row["promotion_eligible"] is False for row in rows if isinstance(row, dict)
    )
    assert all(
        row["ready_for_exact_eval_dispatch"] is False
        for row in rows
        if isinstance(row, dict)
    )

    format_0c = _row_by_id(matrix, "format_0x0c_exact_radix")
    assert format_0c["format_id"] == "0x0C"
    assert format_0c["status"] == "paired_exact_measured"
    exact_0c = format_0c["exact_axis_evidence"]
    assert isinstance(exact_0c, dict)
    assert set(exact_0c) == {"contest_cpu", "contest_cuda"}
    assert exact_0c["contest_cpu"]["valid"] is True
    assert exact_0c["contest_cuda"]["valid"] is True

    format_0d = _row_by_id(matrix, "format_0x0d_latent_score_table")
    assert format_0d["format_id"] == "0x0D"
    assert format_0d["packet_ir_identity"]["passed"] is True
    assert format_0d["runtime_consumption"]["valid"] is True
    assert format_0d["status"] == "paired_exact_measured"

    prefix_1 = _row_by_id(matrix, "prefix_top_1_pr101grammar")
    assert prefix_1["status"] == "runtime_consumed_needs_paired_exact_eval"


def test_pr106_packetir_candidate_matrix_writes_json_and_markdown(
    tmp_path: Path,
) -> None:
    output_json = tmp_path / "matrix.json"
    output_md = tmp_path / "matrix.md"

    matrix = write_pr106_packetir_candidate_matrix(
        output_json=output_json,
        output_md=output_md,
    )

    assert output_json.is_file()
    assert output_md.is_file()
    loaded = json.loads(output_json.read_text(encoding="utf-8"))
    assert loaded["schema"] == PR106_PACKETIR_CANDIDATE_MATRIX_SCHEMA
    assert matrix["artifact_paths"]["json"] == str(output_json)
    markdown = output_md.read_text(encoding="utf-8")
    assert "PR106 PacketIR candidate matrix" in markdown
    assert "format_0x0d_latent_score_table" in markdown


def test_pr106_packetir_candidate_matrix_markdown_is_nonpromotional() -> None:
    matrix = build_pr106_packetir_candidate_matrix()
    markdown = render_pr106_packetir_candidate_matrix_markdown(matrix)

    assert "score_claim=false" in markdown
    assert "promotion_eligible=false" in markdown
    assert "contest_cpu" in markdown
    assert "contest_cuda" in markdown
