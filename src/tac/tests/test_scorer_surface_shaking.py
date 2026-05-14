# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.optimization.scorer_surface_shaking import (
    RATE_SCORE_PER_BYTE,
    ScorerSurfacePlanError,
    build_scorer_surface_shaking_plan,
    default_operating_point,
    render_markdown,
)

REPO = Path(__file__).resolve().parents[3]


def test_default_plan_is_proxy_only_and_ranked_by_local_score_economics() -> None:
    plan = build_scorer_surface_shaking_plan()

    assert plan["schema"] == "scorer_surface_shaking_plan.v1"
    assert plan["score_claim"] is False
    assert plan["dispatch_attempted"] is False
    assert plan["ready_for_exact_eval_dispatch"] is False
    assert "requires_PacketIR_materialized_archive" in plan["dispatch_blockers"]
    assert "requires_exact_cuda_and_paired_cpu_review" in plan["dispatch_blockers"]
    assert plan["operating_point"]["label"] == "pr106_latent_sidecar_r2_pr101_grammar"
    assert plan["operating_point"]["score"] == pytest.approx(0.2066181354574151)
    assert plan["operating_point"]["archive_bytes"] == 186_780
    assert plan["operating_point"]["score_slopes"]["d_score_d_byte"] == pytest.approx(
        RATE_SCORE_PER_BYTE
    )
    assert plan["operating_point"]["score_slopes"]["pose_over_seg_marginal_value"] > 2.7

    rows = plan["ranked_atoms"]
    assert len(rows) >= 5
    assert rows == sorted(
        rows,
        key=lambda row: (
            row["predicted_score_delta"],
            -row["benefit_per_added_byte"],
            row["atom_id"],
        ),
    )
    assert {row["atom_id"] for row in rows} >= {
        "pixel_lsb_threshold_probe",
        "segnet_boundary_run_repair",
        "renderer_training_score_surface_loop",
    }
    pixel = next(row for row in rows if row["atom_id"] == "pixel_lsb_threshold_probe")
    assert pixel["packetir_stream"] == "residual_program_or_score_table_sidecar"
    assert "inflate consumes the charged bytes" in pixel["promotion_gates"]
    assert pixel["expected_added_bytes"] > 0
    assert pixel["score_delta_components"]["rate"] > 0
    assert pixel["rate_positive_if_components_hold"] is (
        pixel["predicted_score_delta"] < 0
    )


def test_custom_operating_point_validation_is_fail_closed() -> None:
    with pytest.raises(ScorerSurfacePlanError, match="unsupported device_axis"):
        default_operating_point().__class__(
            label="bad",
            device_axis="macos_mps",  # type: ignore[arg-type]
            score=1.0,
            archive_bytes=1,
            seg_dist=0.1,
            pose_dist=0.1,
        )


def test_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    json_out = tmp_path / "plan.json"
    md_out = tmp_path / "plan.md"

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_scorer_surface_shaking_plan.py"),
            "--json-out",
            str(json_out),
            "--md-out",
            str(md_out),
            "--max-rows",
            "3",
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert len(payload["ranked_atoms"]) == 3
    assert payload["score_claim"] is False
    markdown = md_out.read_text(encoding="utf-8")
    assert "Scorer Surface Shaking Plan" in markdown
    assert "Planning-only local search plan" in markdown
    assert "requires_PacketIR_materialized_archive" in markdown


def test_render_markdown_names_packetir_and_exact_eval_gates() -> None:
    markdown = render_markdown(build_scorer_surface_shaking_plan(max_rows=2))

    assert "Scorer Surface Shaking Plan" in markdown
    assert "PacketIR" in markdown
    assert "requires_exact_cuda_and_paired_cpu_review" in markdown
