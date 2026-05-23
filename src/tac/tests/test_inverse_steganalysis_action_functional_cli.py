# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tac.optimization.inverse_steganalysis_acquisition import (
    ACTION_FUNCTIONAL_SCHEMA,
    CONTEST_RATE_SCORE_PER_BYTE,
    action_atoms_from_inverse_scorer_surface,
    build_discrete_scorer_action_functional,
)
from tac.optimization.proxy_candidate_contract import PROXY_FALSE_AUTHORITY_FIELDS
from tac.optimization.scorer_inverse_decision_surface import (
    build_inverse_scorer_decision_surface,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
TOOL = REPO_ROOT / "tools" / "build_inverse_steganalysis_action_functional.py"


def _false_authority() -> dict[str, bool]:
    return {
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
    }


def _scorer_response_dataset(path: Path) -> None:
    false_authority = _false_authority()
    path.write_text(
        json.dumps(
            {
                "schema": "scorer_response_dataset.v1",
                "producer": "test",
                **false_authority,
                "authority": {
                    **false_authority,
                    "evidence_grade": "macOS-MLX research-signal",
                },
                "summary": {"row_count": 1},
                "rows": [
                    {
                        "schema": "scorer_response_row.v1",
                        "row_id": "inverse-row-a",
                        "candidate_id": "inverse-row-a",
                        "family": "decoder_q",
                        **false_authority,
                        "authority_source_score_claim": False,
                        "delta_vs_baseline_score": -0.0001,
                        "scorer_delta_vs_baseline": 0.0,
                        "observed_scorer_gain_vs_baseline": 0.0,
                        "added_archive_bytes": -32,
                        "byte_budget_margin_vs_break_even": 32.0,
                        "source_pair_window": [7, 8],
                        "diagnostic_seg_share": 0.15,
                        "diagnostic_pose_share": 0.85,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def test_inverse_surface_cells_become_action_atoms_and_water_buckets(
    tmp_path: Path,
) -> None:
    scorer = tmp_path / "scorer.json"
    _scorer_response_dataset(scorer)
    surface = build_inverse_scorer_decision_surface(
        json.loads(scorer.read_text(encoding="utf-8")),
        source_label="scorer.json",
    )

    atoms = action_atoms_from_inverse_scorer_surface(surface, resource_kind="local_mlx")
    action = build_discrete_scorer_action_functional(
        atoms,
        total_byte_budget=64,
        lambda_rate=CONTEST_RATE_SCORE_PER_BYTE,
    )

    assert atoms[0]["scope_axis"] == "pairs"
    assert atoms[0]["pair_indices"] == [7, 8]
    assert atoms[0]["component"] == "posenet"
    assert atoms[0]["predicted_rate_gain"] == pytest.approx(
        CONTEST_RATE_SCORE_PER_BYTE * 32
    )
    assert action["schema"] == ACTION_FUNCTIONAL_SCHEMA
    assert action["water_bucket"]["selected_count"] == 1
    assert action["water_bucket"]["selected_cells"][0]["atom_id"] == atoms[0]["atom_id"]
    assert action["score_claim"] is False
    for key, value in PROXY_FALSE_AUTHORITY_FIELDS.items():
        assert action[key] is value


def test_cli_builds_inverse_action_functional_from_scorer_response(
    tmp_path: Path,
) -> None:
    scorer = tmp_path / "scorer.json"
    performance = tmp_path / "performance.json"
    runtime_identity = tmp_path / "runtime_identity.json"
    cache_identity = tmp_path / "cache_identity.json"
    output = tmp_path / "action.json"
    md_out = tmp_path / "action.md"
    _scorer_response_dataset(scorer)
    performance.write_text(
        json.dumps(
            {
                "schema": "experiment_queue_performance_summary.v1",
                "queue_id": "inverse_action_queue",
                "telemetry_only": True,
                "score_claim": False,
                "score_claim_valid": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
                "event_count": 1,
                "by_resource_kind": {},
                "by_step": {
                    "inverse-row-a.materialize": {
                        "run_count": 1,
                        "success_count": 1,
                        "failure_count": 0,
                        "resource_kind_counts": {"local_mlx": 1},
                        "dominant_resource_kind": "local_mlx",
                        "elapsed_seconds_mean": 2.25,
                        "artifact_record_bytes_mean": 4096,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    runtime_identity.write_text(
        json.dumps(
            {
                "runtime_tree_sha256": "d" * 64,
                "scorer_version": "local_scheduler.v1",
            }
        ),
        encoding="utf-8",
    )
    cache_identity.write_text(
        json.dumps({"cache_sha256": "e" * 64}),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--scorer-response",
            str(scorer),
            "--queue-performance-summary",
            str(performance),
            "--queue-performance-runtime-identity",
            str(runtime_identity),
            "--queue-performance-cache-identity",
            str(cache_identity),
            "--output",
            str(output),
            "--md-out",
            str(md_out),
            "--repo-root",
            str(tmp_path),
            "--total-byte-budget",
            "64",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    action = json.loads(output.read_text(encoding="utf-8"))
    assert "score_claim=false" in result.stdout
    assert action["schema"] == ACTION_FUNCTIONAL_SCHEMA
    assert action["integral_totals"]["cell_count"] == 1
    assert action["water_bucket"]["selected_count"] == 1
    assert action["cells"][0]["best_observation_id"] == (
        "queue_perf_inverse_action_queue_inverse_row_a_materialize"
    )
    assert action["cells"][0]["priority"]["elapsed_seconds"] == 2.25
    assert action["cells"][0]["priority"]["artifact_bytes"] == 4096
    assert "Selected Water Buckets" in md_out.read_text(encoding="utf-8")


def test_cli_requires_identity_for_queue_performance_summary(tmp_path: Path) -> None:
    scorer = tmp_path / "scorer.json"
    performance = tmp_path / "performance.json"
    output = tmp_path / "action.json"
    _scorer_response_dataset(scorer)
    performance.write_text(
        json.dumps(
            {
                "schema": "experiment_queue_performance_summary.v1",
                "queue_id": "inverse_action_queue",
                "event_count": 0,
                "by_resource_kind": {},
                "by_step": {},
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "--scorer-response",
            str(scorer),
            "--queue-performance-summary",
            str(performance),
            "--output",
            str(output),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode != 0
    assert "--queue-performance-runtime-identity" in result.stderr
