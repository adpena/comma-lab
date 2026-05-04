from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from tac.submission_archive import validate_archive_seg_tile_actions_payloads

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "experiments" / "profile_pr79_pr77_lossless_s3_candidates.py"
PR79_ARCHIVE = (
    REPO / "experiments/results/top_submission_reverse_engineering_20260503_pr79/archive.zip"
)


def _load_script() -> Any:
    spec = importlib.util.spec_from_file_location("pr79_pr77_s3_profile_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_select_recommendation_requires_byte_improvement_and_semantic_proof() -> None:
    script = _load_script()

    recommendation = script._select_recommendation(
        [
            {
                "archive_bytes": 1000,
                "candidate_id": "noop",
                "delta_bytes_vs_pr79": 0,
                "exact_eval_ready_after_lane_claim": False,
            },
            {
                "archive_bytes": 950,
                "candidate_id": "best",
                "delta_bytes_vs_pr79": -50,
                "exact_eval_ready_after_lane_claim": True,
            },
        ]
    )

    assert recommendation["decision"] == "recommend_exact_cuda_eval_after_lane_claim"
    assert recommendation["candidate"]["candidate_id"] == "best"


def test_pr77_mixed_context_is_not_selected_as_strict_lossless(tmp_path: Path) -> None:
    script = _load_script()
    matrix = tmp_path / "candidate_matrix.json"
    matrix.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "archive_bytes": 276329,
                        "candidate_id": "pr77_actions_pr75mask_renderer_c089pose_fixedslice",
                        "delta_bytes_vs_pr77": -222,
                        "dispatchable_after_gate": True,
                        "manifest_path": str(tmp_path / "manifest.json"),
                        "noop_status": "non_noop_payload",
                        "semantic_contract": "current_runtime_fixedslice_pr77_actions_with_c089_pose",
                    }
                ]
            }
        )
    )

    summary = script._summarise_pr77_matrix(matrix)

    assert summary["status"] == "profiled"
    assert summary["best_delta_bytes_vs_pr77"] == -222
    assert summary["strict_lossless_selected"] is False


@pytest.mark.skipif(not PR79_ARCHIVE.exists(), reason="PR79 reverse-engineering archive missing")
def test_s3_profile_recommends_runtime_closed_pr79_s2_candidate(tmp_path: Path) -> None:
    script = _load_script()

    profile = script.build_profile(
        pr79_archive=PR79_ARCHIVE,
        output_dir=tmp_path,
        pr77_mixed_matrix=tmp_path / "missing_pr77_mixed.json",
        pr77_transplant_matrix=tmp_path / "missing_pr77_transplant.json",
        flatpack_matrix=tmp_path / "missing_flatpack.json",
        force=True,
    )
    recommendation = profile["recommendation"]
    candidate = recommendation["candidate"]

    assert recommendation["decision"] == "recommend_exact_cuda_eval_after_lane_claim"
    assert candidate["candidate_id"] == "pr79_s2_fixed_adaptive_actions"
    assert candidate["delta_bytes_vs_pr79"] == -67
    assert candidate["exact_eval_ready_after_lane_claim"] is True
    assert candidate["decoded_semantics_proof"]["action_record_parity"] is True
    assert candidate["decoded_semantics_proof"]["non_action_streams_preserved"] is True
    assert candidate["zip_profile"]["strict_zip_overhead_bytes"] == 100
    assert validate_archive_seg_tile_actions_payloads(candidate["archive_path"]) == []
    assert (tmp_path / "recommendation.json").exists()
