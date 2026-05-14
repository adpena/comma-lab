# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest

from tac.deploy.pr106_yshift import (
    REMOTE_SCRIPT,
    SCORE_TABLE_LANE_ID,
    SCORE_TABLE_ROLE,
    Pr106YshiftScoreTableSpec,
    dispatch_claim_spec,
    score_table_env,
)


def test_pr106_yshift_score_table_env_is_provider_neutral() -> None:
    spec = Pr106YshiftScoreTableSpec(
        job_name="lane_pr106_yshift_score_table_test",
        pr106_archive="experiments/results/pr106/archive.zip",
        candidate_radius=4,
        score_step=0.5,
        n_pairs=512,
        batch_pairs=16,
        candidate_batch_size=64,
    )

    env = score_table_env(spec, output_dir="/workspace/out")

    assert REMOTE_SCRIPT == "scripts/remote_lane_pr106_yshift_sidechannel.sh"
    assert SCORE_TABLE_LANE_ID == "lane_pr106_yshift_score_table"
    assert SCORE_TABLE_ROLE == "pr106_yshift_score_table_cuda"
    assert env == {
        "PR106_YSHIFT_MODE": "score_table",
        "PR106_ARCHIVE": "experiments/results/pr106/archive.zip",
        "PR106_YSHIFT_SCORE_TABLE_LANE_ID": "lane_pr106_yshift_score_table",
        "PR106_YSHIFT_SCORE_TABLE_INSTANCE_JOB_ID": "lane_pr106_yshift_score_table_test",
        "PR106_YSHIFT_CANDIDATE_RADIUS": "4",
        "PR106_YSHIFT_SCORE_STEP": "0.5",
        "PR106_YSHIFT_N_PAIRS": "512",
        "PR106_YSHIFT_SCORE_TABLE_BATCH_PAIRS": "16",
        "PR106_YSHIFT_SCORE_TABLE_CANDIDATE_BATCH_SIZE": "64",
        "PR106_YSHIFT_LOG_DIR": "/workspace/out/yshift_run",
    }


def test_pr106_yshift_score_table_claim_allows_non_lightning_providers() -> None:
    spec = Pr106YshiftScoreTableSpec(
        job_name="kaggle_pr106_yshift",
        pr106_archive="experiments/results/pr106/archive.zip",
    )

    claim = dispatch_claim_spec(
        spec,
        platform="kaggle",
        agent="codex:test",
        predicted_eta_hours=1.0,
        force=True,
        notes="provider-neutral dry run",
    )

    assert claim.lane_id == "lane_pr106_yshift_score_table"
    assert claim.instance_job_id == "kaggle_pr106_yshift"
    assert claim.platform == "kaggle"
    assert claim.agent == "codex:test"
    assert claim.force is True
    assert claim.notes == "provider-neutral dry run"
    assert claim.predicted_eta_utc.endswith("Z")


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("job_name", "", "job_name"),
        ("pr106_archive", "", "pr106_archive"),
        ("pr106_archive", "/tmp/archive.zip", "repo-relative"),
        ("candidate_radius", -1, "candidate_radius"),
        ("score_step", 0.0, "score_step"),
        ("n_pairs", 0, "n_pairs"),
        ("batch_pairs", 0, "batch_pairs"),
        ("candidate_batch_size", 0, "candidate_batch_size"),
        ("lane_id", "", "lane_id"),
    ],
)
def test_pr106_yshift_score_table_spec_fails_closed(
    field: str,
    value: object,
    match: str,
) -> None:
    kwargs = {
        "job_name": "job",
        "pr106_archive": "archive.zip",
        "candidate_radius": 3,
        "score_step": 1.0,
        "n_pairs": 600,
        "batch_pairs": 8,
        "candidate_batch_size": 32,
        "lane_id": SCORE_TABLE_LANE_ID,
    }
    kwargs[field] = value

    with pytest.raises(ValueError, match=match):
        Pr106YshiftScoreTableSpec(**kwargs).validate()


def test_pr106_yshift_score_table_log_dir_requires_output_dir() -> None:
    spec = Pr106YshiftScoreTableSpec(job_name="job", pr106_archive="archive.zip")

    with pytest.raises(ValueError, match="output_dir"):
        score_table_env(spec)

    assert "PR106_YSHIFT_LOG_DIR" not in score_table_env(spec, include_log_dir=False)
