# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import tac.optimization.l5_staircase_v2 as l5_v2

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent


def _copy_tool_into_tmp_repo(tmp_path: Path, relpath: str) -> Path:
    source = REPO_ROOT / relpath
    target = tmp_path / relpath
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    return target


def _write_score_axis_artifact(repo_root: Path) -> Path:
    artifact_path = repo_root / l5_v2.TT5L_DYKSTRA_FEASIBILITY_ARTIFACT_PATH
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "schema": l5_v2.TT5L_DYKSTRA_FEASIBILITY_SCHEMA,
                "predicate_id": l5_v2.TT5L_DYKSTRA_FEASIBILITY_PREDICATE_ID,
                "generated_by_tool": l5_v2.TT5L_DYKSTRA_FEASIBILITY_GENERATED_BY_TOOL,
                "substrate_id": l5_v2.TT5L_DYKSTRA_SUBSTRATE_ID,
                "move_level_constraint_proof": False,
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return artifact_path


def _write_proof_artifact(repo_root: Path, *, score_axis_artifact: Path) -> Path:
    proof_tool = _copy_tool_into_tmp_repo(
        repo_root,
        "tools/prove_tt5l_move_level_feasibility.py",
    )
    artifact_path = (
        repo_root
        / "experiments"
        / "results"
        / "time_traveler_l5_v2"
        / "tt5l_move_level_solver_proof.json"
    )
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "schema": "tt5l_move_level_solver_proof_test_v1",
                "subject_id": l5_v2.TT5L_DYKSTRA_SUBSTRATE_ID,
                "predicate_id": l5_v2.TT5L_MOVE_LEVEL_FEASIBILITY_PREDICATE_ID,
                "predicate_passed": True,
                "move_level_constraint_proof": True,
                "residual_max": 0.0,
                "residual_tolerance": 1e-9,
                "constraint_set_ids": sorted(
                    l5_v2.TT5L_DYKSTRA_REQUIRED_CONSTRAINT_IDS
                ),
                "score_axis_sanity_artifact_sha256": hashlib.sha256(
                    score_axis_artifact.read_bytes()
                ).hexdigest(),
                "generated_by_tool": "tools/prove_tt5l_move_level_feasibility.py",
                "tool_sha256": hashlib.sha256(proof_tool.read_bytes()).hexdigest(),
                "mechanism_records": [
                    {
                        "constraint_id": constraint_id,
                        "passed": True,
                        "residual": 0.0,
                        "details": {"fixture": True},
                    }
                    for constraint_id in sorted(
                        l5_v2.TT5L_DYKSTRA_REQUIRED_CONSTRAINT_IDS
                    )
                ],
                "witness_variables": {
                    constraint_id: {"fixture": True}
                    for constraint_id in sorted(
                        l5_v2.TT5L_DYKSTRA_REQUIRED_CONSTRAINT_IDS
                    )
                },
                "score_claim": False,
                "promotion_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return artifact_path


def test_tt5l_move_level_tool_builds_valid_artifact(tmp_path: Path) -> None:
    tool = REPO_ROOT / "tools" / "build_tt5l_move_level_feasibility_artifact.py"
    _copy_tool_into_tmp_repo(
        tmp_path,
        l5_v2.TT5L_MOVE_LEVEL_FEASIBILITY_TOOL_PATH,
    )
    score_axis_artifact = _write_score_axis_artifact(tmp_path)
    proof_artifact = _write_proof_artifact(
        tmp_path,
        score_axis_artifact=score_axis_artifact,
    )
    proof_relpath = str(proof_artifact.relative_to(tmp_path))

    proc = subprocess.run(
        [
            sys.executable,
            str(tool),
            "--repo-root",
            str(tmp_path),
            "--proof-artifact",
            proof_relpath,
            "--proof-command-argv-json",
            json.dumps(["python", "experiments/solve_tt5l_move_level.py"]),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 0, proc.stderr
    artifact_path = tmp_path / l5_v2.TT5L_MOVE_LEVEL_FEASIBILITY_ARTIFACT_PATH
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert payload["schema"] == l5_v2.TT5L_MOVE_LEVEL_FEASIBILITY_SCHEMA
    assert payload["subject_id"] == l5_v2.TT5L_DYKSTRA_SUBSTRATE_ID
    assert payload["predicate_id"] == l5_v2.TT5L_MOVE_LEVEL_FEASIBILITY_PREDICATE_ID
    assert payload["constraint_set_ids"] == sorted(
        l5_v2.TT5L_DYKSTRA_REQUIRED_CONSTRAINT_IDS
    )
    assert payload["proof_artifact_path"] == proof_relpath
    assert payload["proof_artifact_sha256"] == hashlib.sha256(
        proof_artifact.read_bytes()
    ).hexdigest()
    assert payload["score_axis_sanity_artifact_path"] == (
        l5_v2.TT5L_DYKSTRA_FEASIBILITY_ARTIFACT_PATH
    )
    assert payload["score_axis_sanity_artifact_sha256"] == hashlib.sha256(
        score_axis_artifact.read_bytes()
    ).hexdigest()
    assert payload["generated_by_tool"] == l5_v2.TT5L_MOVE_LEVEL_FEASIBILITY_TOOL_PATH
    assert payload["tool_sha256"] == hashlib.sha256(
        (tmp_path / l5_v2.TT5L_MOVE_LEVEL_FEASIBILITY_TOOL_PATH).read_bytes()
    ).hexdigest()
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
    stdout = json.loads(proc.stdout)
    assert stdout["status"]["artifact_valid"] is True


def test_tt5l_move_level_tool_refuses_missing_proof_artifact(tmp_path: Path) -> None:
    tool = REPO_ROOT / "tools" / "build_tt5l_move_level_feasibility_artifact.py"
    _write_score_axis_artifact(tmp_path)

    proc = subprocess.run(
        [
            sys.executable,
            str(tool),
            "--repo-root",
            str(tmp_path),
            "--proof-artifact",
            "experiments/results/time_traveler_l5_v2/missing.json",
            "--proof-command-argv-json",
            json.dumps(["python", "experiments/solve_tt5l_move_level.py"]),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 2
    assert "proof artifact missing" in proc.stderr
    assert not (
        tmp_path / l5_v2.TT5L_MOVE_LEVEL_FEASIBILITY_ARTIFACT_PATH
    ).exists()


def test_tt5l_move_level_tool_refuses_unproven_payload(tmp_path: Path) -> None:
    tool = REPO_ROOT / "tools" / "build_tt5l_move_level_feasibility_artifact.py"
    score_axis_artifact = _write_score_axis_artifact(tmp_path)
    proof_artifact = _write_proof_artifact(
        tmp_path,
        score_axis_artifact=score_axis_artifact,
    )
    payload = json.loads(proof_artifact.read_text(encoding="utf-8"))
    payload["move_level_constraint_proof"] = False
    proof_artifact.write_text(
        json.dumps(payload, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(tool),
            "--repo-root",
            str(tmp_path),
            "--proof-artifact",
            str(proof_artifact.relative_to(tmp_path)),
            "--proof-command-argv-json",
            json.dumps(["python", "experiments/solve_tt5l_move_level.py"]),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert proc.returncode == 2
    assert "move_level_constraint_proof must be true" in proc.stderr
