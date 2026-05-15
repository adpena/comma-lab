# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from tac.repo_io import read_json, write_json
from tac.tests.tool_loader import load_repo_tool

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_tool():
    return load_repo_tool(
        REPO_ROOT,
        "tools/pr106_cuda_latent_correction_materializer.py",
        "pr106_cuda_latent_correction_materializer_test",
    )


def _write_jsonl_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _fixture_plan(source_archive: Path) -> dict[str, Any]:
    source_bytes = source_archive.read_bytes()
    return {
        "schema": "pr106_cuda_latent_correction_probe_plan_v1",
        "tool": "tools/build_pr106_cuda_latent_correction_probe.py",
        "from_state_hash": "fixturehash0001",
        "label": "fixture_probe",
        "authority": {
            "research_only": True,
            "score_claim": False,
            "score_claim_valid": False,
            "contest_axis_claim": False,
            "dispatch_attempted": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "ready_for_broad_waterfill_dispatch": False,
            "frontier_language_allowed": False,
        },
        "inputs": {
            "source_archive": {
                "path": source_archive.as_posix(),
                "bytes": len(source_bytes),
                "sha256": hashlib.sha256(source_bytes).hexdigest(),
            },
            "pair_hitlists": [],
            "pair_xrays": [],
            "paired_axis_artifacts": [],
            "pr106_format0c_ledgers": [],
        },
        "selection_policy": {
            "latent_dim_count": 3,
            "delta_q_values": [-2, -1, 1, 2],
            "probe_modes_per_pair": 12,
        },
        "byte_accounting": {
            "archive_mutation_performed": False,
            "materialized_archive_bytes": None,
            "materialized_archive_sha256": None,
        },
        "candidate_pairs": [
            {
                "priority_rank": 1,
                "pair_idx": 5,
                "priority": 0.09,
                "dominant_component": "pose",
                "axis_dominant_component": "pose",
                "planned_mode": "cuda_latent_dim28_delta_q_pm2_grid_then_format0c_selection",
                "latent_dim_count": 3,
                "delta_q_values": [-2, -1, 1, 2],
            },
            {
                "priority_rank": 2,
                "pair_idx": 2,
                "priority": 0.08,
                "dominant_component": "seg",
                "axis_dominant_component": "pose",
                "planned_mode": "cuda_latent_dim28_delta_q_pm2_grid_then_format0c_selection",
                "latent_dim_count": 3,
                "delta_q_values": [-2, -1, 1, 2],
            },
        ],
        "materialization": {
            "supported": False,
            "placeholder_fails_closed": True,
        },
    }


def test_materializer_writes_per_pair_latent_task_artifacts(tmp_path: Path) -> None:
    module = _load_tool()
    source_archive = tmp_path / "archive.zip"
    source_archive.write_bytes(b"fixture archive bytes")
    plan_path = tmp_path / "plan.json"
    write_json(plan_path, _fixture_plan(source_archive))
    output_dir = tmp_path / "out"

    manifest = module.build_materialization(
        plan_path=plan_path,
        output_dir=output_dir,
        python_executable="python",
        n_pairs=600,
        lane_id="lane_pr106_latent_score_table",
    )

    assert manifest["schema"] == "pr106_cuda_latent_correction_materializer_v1"
    assert manifest["authority"]["score_claim"] is False
    assert manifest["authority"]["promotion_eligible"] is False
    assert manifest["authority"]["ready_for_exact_eval_dispatch"] is False
    assert manifest["authority"]["scoretable_commands_execute_gpu"] is False
    assert manifest["counts"] == {
        "candidate_pair_count": 2,
        "scoretable_command_count": 2,
        "probe_task_count": 24,
    }

    tasks = _write_jsonl_rows(output_dir / "pr106_cuda_latent_correction_probe_tasks.jsonl")
    commands = _write_jsonl_rows(
        output_dir / "pr106_cuda_latent_correction_scoretable_commands.jsonl"
    )

    assert len(tasks) == 24
    assert len(commands) == 2
    assert tasks[0]["task_id"] == "pr106_pair_0005_dim_00_dq_m2"
    assert tasks[0]["pair_idx"] == 5
    assert tasks[0]["latent_dim_idx"] == 0
    assert tasks[0]["delta_q"] == -2
    assert tasks[0]["candidate_grid_index"] == 1
    assert tasks[3]["task_id"] == "pr106_pair_0005_dim_00_dq_p2"
    assert tasks[3]["candidate_grid_index"] == 4
    assert tasks[7]["task_id"] == "pr106_pair_0005_dim_01_dq_p2"
    assert tasks[7]["candidate_grid_index"] == 8
    assert all(task["authority"]["score_claim"] is False for task in tasks)
    assert all(task["authority"]["archive_mutation_performed"] is False for task in tasks)

    first_command = commands[0]
    assert first_command["command_id"] == "pair_0005_scoretable_dryrun"
    assert first_command["executes_gpu"] is False
    assert first_command["dispatches_remote"] is False
    assert "--dry-run-plan" in first_command["command_args"]
    assert "--instance-job-id" not in first_command["command_args"]
    assert "claim_lane_dispatch.py" not in first_command["command"]
    assert first_command["command_args"][first_command["command_args"].index("--max-pairs") + 1] == "6"

    script = (output_dir / "pr106_cuda_latent_correction_scoretable_commands.sh").read_text(
        encoding="utf-8"
    )
    assert "Generated dry-run score-table commands only" in script
    assert "--dry-run-plan" in script
    assert "claim_lane_dispatch.py" not in script
    assert (output_dir / "pairs" / "pair_0005.json").is_file()
    assert (output_dir / "pairs" / "pair_0002.json").is_file()

    written_manifest = read_json(
        output_dir / "pr106_cuda_latent_correction_materializer_manifest.json"
    )
    assert written_manifest["outputs"]["manifest"]["sha256"]
    assert written_manifest["dispatch_blockers"][0] == "dry_run_materializer_only"


def test_materializer_rejects_score_authority_plan(tmp_path: Path) -> None:
    module = _load_tool()
    source_archive = tmp_path / "archive.zip"
    source_archive.write_bytes(b"fixture archive bytes")
    plan = _fixture_plan(source_archive)
    plan["authority"]["score_claim"] = True
    plan_path = tmp_path / "plan.json"
    write_json(plan_path, plan)

    with pytest.raises(ValueError, match="score_claim=true"):
        module.build_materialization(plan_path=plan_path, output_dir=tmp_path / "out")


def test_materializer_rejects_source_archive_sha_mismatch(tmp_path: Path) -> None:
    module = _load_tool()
    source_archive = tmp_path / "archive.zip"
    source_archive.write_bytes(b"fixture archive bytes")
    plan = _fixture_plan(source_archive)
    source_archive.write_bytes(b"changed bytes")
    plan_path = tmp_path / "plan.json"
    write_json(plan_path, plan)

    with pytest.raises(ValueError, match="source archive SHA mismatch"):
        module.build_materialization(plan_path=plan_path, output_dir=tmp_path / "out")


def test_cli_writes_outputs_and_refuses_missing_archive_bypass(tmp_path: Path) -> None:
    module = _load_tool()
    source_archive = tmp_path / "archive.zip"
    source_archive.write_bytes(b"fixture archive bytes")
    plan_path = tmp_path / "plan.json"
    write_json(plan_path, _fixture_plan(source_archive))
    output_dir = tmp_path / "out"

    assert module.main(
        [
            "--plan",
            str(plan_path),
            "--output-dir",
            str(output_dir),
            "--python-executable",
            "python",
        ]
    ) == 0

    manifest = read_json(output_dir / "pr106_cuda_latent_correction_materializer_manifest.json")
    assert manifest["scoretable_contract"]["real_cuda_commands_emitted"] is False
    assert "tools/pr106_cuda_latent_correction_materializer.py" in manifest["rebuild_command"]
    assert module.main(
        [
            "--plan",
            str(plan_path),
            "--output-dir",
            str(tmp_path / "blocked"),
            "--allow-missing-source-archive",
        ]
    ) == 2
