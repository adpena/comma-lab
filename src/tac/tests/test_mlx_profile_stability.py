# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tac.local_acceleration import EVIDENCE_GRADE_MLX
from tac.local_acceleration.mlx_profile_stability import (
    FAIL_VERDICT,
    PASS_VERDICT,
    MLXProfileStabilityThresholds,
    build_profile_stability_manifest,
)

REPO = Path(__file__).resolve().parents[3]


def _profile(*, score_delta: float = 0.0, evidence_grade: str | None = EVIDENCE_GRADE_MLX):
    payload = {
        "schema_version": "mlx_scorer_response_profile.v1",
        "score_claim": False,
        "promotion_eligible": False,
        "promotable": False,
        "rank_or_kill_eligible": False,
        "reference_cache_dir": "/tmp/ref",
        "candidate_cache_dir": "/tmp/cand",
        "start_pair": 8,
        "max_pairs": 4,
        "rows": [
            {
                "device": "cpu",
                "batch_pairs": 1,
                "n_samples": 4,
                "pair_window": [8, 12],
                "canonical_score": 0.2,
                "avg_posenet_dist": 0.001,
                "avg_segnet_dist": 0.002,
                "posenet_sha256": "p" * 64,
                "segnet_sha256": "s" * 64,
                "pairs_per_second": 1.0,
            },
            {
                "device": "cpu",
                "batch_pairs": 2,
                "n_samples": 4,
                "pair_window": [8, 12],
                "canonical_score": 0.2 + score_delta,
                "avg_posenet_dist": 0.001 + 1.0e-10,
                "avg_segnet_dist": 0.002,
                "posenet_sha256": "q" * 64,
                "segnet_sha256": "s" * 64,
                "pairs_per_second": 2.0,
            },
        ],
    }
    if evidence_grade is not None:
        payload["evidence_grade"] = evidence_grade
    return payload


def test_profile_stability_passes_small_metric_drift_with_sha_warning() -> None:
    manifest = build_profile_stability_manifest(_profile(), baseline_batch_pairs=1)

    assert manifest["passed"] is True
    assert manifest["verdict"] == PASS_VERDICT
    assert manifest["score_claim"] is False
    assert manifest["promotion_eligible"] is False
    assert "profile_row_posenet_sha256_mismatch:index=1" in manifest["warnings"]
    assert manifest["selection"]["eligible_row_indices"] == [0]
    assert manifest["selection"]["recommended_row"]["index"] == 0
    assert manifest["selection"]["recommended_row"]["batch_pairs"] == 1
    assert manifest["selection"]["rejected_rows"][0]["reason"] == (
        "non_singleton_batch_shape_requires_explicit_research_allowance"
    )
    assert manifest["profile_summary"]["archive_size_bytes"] is None


def test_profile_stability_can_select_non_singleton_for_explicit_batch_shape_research() -> None:
    manifest = build_profile_stability_manifest(
        _profile(),
        baseline_batch_pairs=1,
        allow_batch_shape_research_signal=True,
    )

    assert manifest["passed"] is True
    assert manifest["selection"]["eligible_row_indices"] == [0, 1]
    assert manifest["selection"]["recommended_row"]["index"] == 1
    assert manifest["selection"]["recommended_row"]["batch_pairs"] == 2


def test_profile_stability_can_require_component_sha_match() -> None:
    manifest = build_profile_stability_manifest(
        _profile(),
        thresholds=MLXProfileStabilityThresholds(require_component_sha_match=True),
    )

    assert manifest["passed"] is False
    assert manifest["verdict"] == FAIL_VERDICT
    assert "profile_row_posenet_sha256_mismatch:index=1" in manifest["blockers"]
    assert manifest["selection"]["eligible_row_indices"] == [0]
    assert manifest["selection"]["recommended_row"]["index"] == 0


def test_profile_stability_fails_metric_drift_and_false_authority() -> None:
    profile = _profile(score_delta=2.0e-5)
    profile["score_claim"] = True
    manifest = build_profile_stability_manifest(profile)

    assert manifest["passed"] is False
    assert "profile_attempts_score_claim" in manifest["blockers"]
    assert any(
        blocker.startswith("profile_row_score_delta_exceeds_threshold")
        for blocker in manifest["blockers"]
    )
    assert manifest["selection"]["eligible_row_indices"] == []
    assert all(
        row["reason"] == "global_profile_blockers"
        for row in manifest["selection"]["rejected_rows"]
    )


def test_profile_stability_requires_mlx_evidence_grade() -> None:
    manifest = build_profile_stability_manifest(_profile(evidence_grade=None))

    assert manifest["passed"] is False
    assert "profile_evidence_grade_missing" in manifest["blockers"]


def test_profile_stability_cli_writes_manifest(tmp_path: Path) -> None:
    profile_path = tmp_path / "profile.json"
    out_path = tmp_path / "stability.json"
    profile_path.write_text(json.dumps(_profile()), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "check_mlx_scorer_response_profile_stability.py"),
            "--profile",
            str(profile_path),
            "--output",
            str(out_path),
            "--baseline-batch-pairs",
            "1",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert '"passed": true' in completed.stdout
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["passed"] is True
    assert payload["baseline"]["batch_pairs"] == 1
    assert payload["selection"]["recommended_row"]["batch_pairs"] == 1
