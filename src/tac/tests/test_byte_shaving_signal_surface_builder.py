# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.optimization.byte_shaving_campaign import (
    ByteShavingCampaignError,
    build_byte_shaving_campaign_plan,
)
from tac.optimization.byte_shaving_signal_surface_builder import (
    build_byte_shaving_signal_surface,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL = REPO_ROOT / "tools" / "build_byte_shaving_signal_surface.py"


def _candidate_queue(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "optimizer_candidate_queue_v1",
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "top_k": [
                    {
                        "candidate_id": "drop_pair_0371",
                        "source_candidate_id": "trained_seed7",
                        "candidate_saved_bytes": 120,
                        "predicted_quality_score_cost": 0.00001,
                        "confidence": 0.8,
                        "operation_families": ["drop_pair"],
                        "score_claim": False,
                        "promotion_eligible": False,
                        "rank_or_kill_eligible": False,
                        "ready_for_exact_eval_dispatch": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _auth_eval(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "canonical_score": 0.2,
                "score_recomputed_from_components": 0.2,
                "avg_posenet_dist": 0.01,
                "avg_segnet_dist": 0.001,
                "rate_unscaled": 0.004,
                "archive_size_bytes": 123,
                "n_samples": 600,
                "canonical_score_source": "score_recomputed_from_components",
                "actual_device": "cpu",
                "evidence_grade": "contest-CPU",
                "lane_tag": "[contest-CPU]",
                "score_axis": "contest_cpu",
                "evidence_semantics": "public_leaderboard_cpu_reproduction",
                "cpu_leaderboard_reproduction_eligible": True,
                "score_claim": True,
                "score_claim_valid": True,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "provenance": {
                    "actual_device": "cpu",
                    "platform_system": "Linux",
                    "platform_machine": "x86_64",
                },
            }
        ),
        encoding="utf-8",
    )


def _mlx_calibration(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": "mlx_score_calibration.v1",
                "row_count": 3,
                "evidence_grade": "macOS-MLX",
                "evidence_tag": "[macOS-MLX research-signal]",
                "score_claim": False,
                "score_claim_valid": False,
                "promotion_eligible": False,
                "promotable": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "summary": {"cpu_pairwise_order_accuracy": 1.0},
                "decision_policy": {
                    "allowed_use": "local_spend_triage_only_after_strict_auth_axis_calibration"
                },
            }
        ),
        encoding="utf-8",
    )


def test_builder_merges_queue_and_sanitized_refs_into_plannable_surface(
    tmp_path: Path,
) -> None:
    queue = tmp_path / "queue.json"
    auth = tmp_path / "auth.json"
    mlx = tmp_path / "mlx.json"
    _candidate_queue(queue)
    _auth_eval(auth)
    _mlx_calibration(mlx)

    surface = build_byte_shaving_signal_surface(
        repo_root=tmp_path,
        campaign_id="fixture_surface",
        candidate_queue_paths=[queue],
        auth_eval_paths=[auth],
        mlx_calibration_paths=[mlx],
        xray_hooks=["bit_allocator"],
    )
    plan = build_byte_shaving_campaign_plan(surface, repo_root=tmp_path)

    assert surface["schema"] == "byte_shaving_signal_surface.v1"
    assert surface["score_claim"] is False
    assert surface["units"][0]["unit_id"] == "drop_pair_0371"
    assert surface["auth_eval_refs"][0]["source_score_claim_present"] is True
    assert "score_claim" not in surface["auth_eval_refs"][0]["metrics"]
    assert surface["mlx_calibration_refs"][0]["score_claim"] is False
    assert surface["xray_refs"][0]["primitive_count"] >= 0
    assert plan["ranked_units"][0]["unit_id"] == "drop_pair_0371"
    assert plan["score_claim"] is False


def test_builder_rejects_truthy_proxy_sources(tmp_path: Path) -> None:
    queue = tmp_path / "queue.json"
    _candidate_queue(queue)
    payload = json.loads(queue.read_text(encoding="utf-8"))
    payload["top_k"][0]["score_claim"] = True
    queue.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ByteShavingCampaignError, match="score_claim"):
        build_byte_shaving_signal_surface(
            repo_root=tmp_path,
            campaign_id="fixture_surface",
            candidate_queue_paths=[queue],
        )


def test_cli_writes_surface_and_markdown(tmp_path: Path) -> None:
    queue = tmp_path / "queue.json"
    output = tmp_path / "surface.json"
    md_out = tmp_path / "surface.md"
    _candidate_queue(queue)

    result = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--candidate-queue",
            str(queue),
            "--output",
            str(output),
            "--md-out",
            str(md_out),
            "--repo-root",
            str(tmp_path),
            "--campaign-id",
            "fixture_surface",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "score_claim=false" in result.stdout
    surface = json.loads(output.read_text(encoding="utf-8"))
    assert surface["units"][0]["unit_id"] == "drop_pair_0371"
    assert "Authority Boundary" in md_out.read_text(encoding="utf-8")
