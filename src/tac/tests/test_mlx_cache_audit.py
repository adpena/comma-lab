# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from tac.auth_eval_schema import ORIGINAL_VIDEO_BYTES, contest_formula_score
from tac.local_acceleration.mlx_cache_audit import (
    FAIL_VERDICT,
    PASS_VERDICT,
    audit_mlx_scorer_input_cache_against_auth_eval,
)

REPO = Path(__file__).resolve().parents[3]


def _cache(*, aggregate: str = "b" * 64, pair_count: int = 600) -> dict[str, object]:
    return {
        "schema_version": "mlx_scorer_input_cache.v1",
        "archive_sha256": "a" * 64,
        "inflated_outputs_aggregate_sha256": aggregate,
        "raw_sha256": "r" * 64,
        "pair_count": pair_count,
        "segnet_last_rgb_shape": [pair_count, 3, 384, 512],
        "posenet_yuv6_pair_shape": [pair_count, 12, 192, 256],
        "score_claim": False,
        "promotion_eligible": False,
    }


def _auth(*, aggregate: str = "b" * 64, n_samples: int = 600) -> dict[str, object]:
    archive_size = 178_517
    seg = 0.00056029
    pose = 0.00002943
    return {
        "canonical_score": contest_formula_score(
            seg_dist=seg,
            pose_dist=pose,
            archive_bytes=archive_size,
        ),
        "canonical_score_source": "score_recomputed_from_components",
        "avg_segnet_dist": seg,
        "avg_posenet_dist": pose,
        "archive_size_bytes": archive_size,
        "score_rate_contribution": 25.0 * archive_size / ORIGINAL_VIDEO_BYTES,
        "n_samples": n_samples,
        "evidence_grade": "contest-CPU",
        "score_axis": "contest_cpu",
        "provenance": {
            "archive_sha256": "a" * 64,
            "inflated_output_manifest": {
                "payload": {
                    "aggregate_sha256": aggregate,
                    "files": [{"sha256": "r" * 64}],
                }
            },
        },
    }


def test_cache_audit_passes_matching_identity() -> None:
    audit = audit_mlx_scorer_input_cache_against_auth_eval(_cache(), _auth())
    assert audit["passed"] is True
    assert audit["verdict"] == PASS_VERDICT
    assert audit["score_claim"] is False
    assert audit["promotion_eligible"] is False
    assert "local_mlx_training_transfer_calibration" in audit["allowed_use"]


def test_cache_audit_fails_inflated_aggregate_mismatch() -> None:
    audit = audit_mlx_scorer_input_cache_against_auth_eval(_cache(aggregate="c" * 64), _auth())
    assert audit["passed"] is False
    assert audit["verdict"] == FAIL_VERDICT
    assert "inflated_outputs_aggregate_sha256_mismatch_or_missing" in audit["blockers"]
    assert "do_not_use_for_auth_axis_transfer_calibration" in audit["allowed_use"]


def test_cache_audit_fails_pair_count_mismatch() -> None:
    audit = audit_mlx_scorer_input_cache_against_auth_eval(_cache(pair_count=16), _auth())
    assert audit["passed"] is False
    assert "cache_pair_count_mismatch:cache=16:expected=600" in audit["blockers"]


def test_cache_audit_cli_writes_output(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.json"
    auth_path = tmp_path / "auth.json"
    out_path = tmp_path / "audit.json"
    cache_path.write_text(json.dumps(_cache()), encoding="utf-8")
    auth_path.write_text(json.dumps(_auth()), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(REPO / "tools" / "audit_mlx_scorer_input_cache.py"),
            "--cache-manifest",
            str(cache_path),
            "--auth-eval",
            str(auth_path),
            "--output",
            str(out_path),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert '"passed": true' in completed.stdout
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["passed"] is True
