# SPDX-License-Identifier: MIT
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import tac.optimization.l5_staircase_v2 as l5_v2


def _write_score_axis_artifact(repo_root: Path) -> Path:
    artifact_path = repo_root / l5_v2.TT5L_DYKSTRA_FEASIBILITY_ARTIFACT_PATH
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(
            {
                "schema": l5_v2.TT5L_DYKSTRA_FEASIBILITY_SCHEMA,
                "predicate_id": l5_v2.TT5L_DYKSTRA_FEASIBILITY_PREDICATE_ID,
                "substrate_id": l5_v2.TT5L_DYKSTRA_SUBSTRATE_ID,
                "verdict": "FEASIBLE",
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


def test_tt5l_move_level_structural_proof_tool_emits_witnesses(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    tool = repo_root / "tools" / "prove_tt5l_move_level_feasibility.py"
    tmp_tool = tmp_path / "tools/prove_tt5l_move_level_feasibility.py"
    tmp_tool.parent.mkdir(parents=True, exist_ok=True)
    tmp_tool.write_text(tool.read_text(encoding="utf-8"), encoding="utf-8")
    trainer_source = repo_root / "experiments/train_substrate_time_traveler_l5_autonomy.py"
    trainer_target = tmp_path / "experiments/train_substrate_time_traveler_l5_autonomy.py"
    trainer_target.parent.mkdir(parents=True, exist_ok=True)
    trainer_target.write_text(
        trainer_source.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    score_axis_artifact = _write_score_axis_artifact(tmp_path)
    output_json = (
        tmp_path / ".omx/research/tt5l_move_level_structural_proof_test.json"
    )

    proc = subprocess.run(
        [
            sys.executable,
            str(tool),
            "--repo-root",
            str(tmp_path),
            "--score-axis-sanity-artifact",
            str(score_axis_artifact.relative_to(tmp_path)),
            "--output-json",
            str(output_json.relative_to(tmp_path)),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert proc.returncode == 0, proc.stderr
    payload = json.loads(output_json.read_text(encoding="utf-8"))
    assert payload["schema"] == "tt5l_move_level_structural_feasibility_proof_v1"
    assert payload["predicate_id"] == l5_v2.TT5L_MOVE_LEVEL_FEASIBILITY_PREDICATE_ID
    assert payload["predicate_passed"] is True
    assert payload["move_level_constraint_proof"] is True
    assert payload["constraint_set_ids"] == sorted(
        l5_v2.TT5L_DYKSTRA_REQUIRED_CONSTRAINT_IDS
    )
    assert payload["score_axis_sanity_artifact_sha256"] == hashlib.sha256(
        score_axis_artifact.read_bytes()
    ).hexdigest()
    assert payload["tool_sha256"] == hashlib.sha256(tmp_tool.read_bytes()).hexdigest()
    record_ids = {
        row["constraint_id"]
        for row in payload["mechanism_records"]
        if row["passed"] is True
    }
    assert l5_v2.TT5L_DYKSTRA_REQUIRED_CONSTRAINT_IDS.issubset(record_ids)
    assert payload["witness_variables"]
    assert payload["score_claim"] is False
    assert payload["promotion_eligible"] is False
    assert payload["ready_for_exact_eval_dispatch"] is False
