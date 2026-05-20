# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tac.theoretical_floor_candidates import (
    build_candidate_matrix,
    pr110_score_decomposition,
)

REPO = Path(__file__).resolve().parents[3]


def test_pr110_floor_decomposition_uses_weighted_terms() -> None:
    parts = pr110_score_decomposition()

    assert round(parts["segnet_weighted_score_term"], 6) == 0.056029
    assert round(parts["posenet_weighted_score_term"], 6) == 0.017155
    assert round(parts["rate_weighted_score_term"], 6) == 0.118867
    assert round(parts["distortion_only_zero_archive_score"], 6) == 0.073184
    assert round(parts["total_recomputed_score"], 6) == 0.192051


def test_candidate_matrix_forbids_external_models_as_runtime_deps() -> None:
    matrix = build_candidate_matrix(repo_root=REPO)

    assert matrix["score_claim"] is False
    assert matrix["promotion_eligible"] is False
    assert matrix["ready_for_exact_eval_dispatch"] is False
    source_policy = {
        source["source_id"]: source["runtime_dependency_allowed"]
        for source in matrix["sources"]
    }
    assert source_policy["telescope_hyperbolic_foveation"] is False
    assert source_policy["la_pose"] is False
    assert source_policy["sea_raft"] is False
    assert source_policy["visual_primitives"] is False

    contracts = {item["candidate_id"]: item for item in matrix["runtime_contracts"]}
    foveation = contracts["tf_telescope_lfv1_pose_foveation"]
    assert "sea_raft" in foveation["compress_time_teachers"]
    assert "la_pose" in foveation["compress_time_teachers"]
    assert "visual_primitives" in foveation["compress_time_teachers"]
    assert "raft" not in " ".join(foveation["runtime_dependencies"]).lower()
    assert "la-pose" not in " ".join(foveation["runtime_dependencies"]).lower()
    assert foveation["local_surface_status"]["src/tac/hyperbolic_foveation.py"] is True
    assert foveation["ready_for_exact_eval_dispatch"] is False


def test_candidate_matrix_cli_writes_json_and_markdown(tmp_path: Path) -> None:
    json_out = tmp_path / "matrix.json"
    md_out = tmp_path / "matrix.md"

    subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "build_theoretical_floor_candidate_matrix.py"),
            "--json-out",
            str(json_out),
            "--md-out",
            str(md_out),
        ],
        cwd=REPO,
        check=True,
        text=True,
    )

    matrix = json.loads(json_out.read_text(encoding="utf-8"))
    assert matrix["schema"] == "theoretical_floor_candidate_matrix_v1"
    assert "tf_siren_first_anchor" in {
        item["candidate_id"] for item in matrix["runtime_contracts"]
    }
    markdown = md_out.read_text(encoding="utf-8")
    assert "Source-Backed Theoretical-Floor Candidate Matrix" in markdown
    assert "distortion-only zero-archive score" in markdown
