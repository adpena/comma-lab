from __future__ import annotations

import pytest

from tac.deploy.pr106_latent import (
    Pr106LatentScoreTableSpec,
    dispatch_claim_spec,
    score_table_env,
)


def test_pr106_latent_score_table_env_declares_score_table_contract() -> None:
    spec = Pr106LatentScoreTableSpec(
        job_name="kaggle_pr106_latent_test",
        pr106_archive="inputs/pr106_archive.zip",
        delta_radius=2,
        latent_dim=28,
        batch_pairs=3,
        candidate_batch_size=5,
        sidecar_top_k=400,
    )

    env = score_table_env(spec, output_dir="/kaggle/working/pr106_latent")

    assert env["PR106_LATENT_MODE"] == "score_table"
    assert env["PR106_ARCHIVE"] == "inputs/pr106_archive.zip"
    assert env["PR106_LATENT_DELTA_RADIUS"] == "2"
    assert env["PR106_LATENT_DIM"] == "28"
    assert env["PR106_LATENT_SCORE_TABLE_BATCH_PAIRS"] == "3"
    assert env["PR106_LATENT_SCORE_TABLE_CANDIDATE_BATCH_SIZE"] == "5"
    assert env["PR106_LATENT_SCORE_TABLE_INSTANCE_JOB_ID"] == "kaggle_pr106_latent_test"
    assert env["SIDECAR_TOP_K"] == "400"
    assert env["PR106_LATENT_LOG_DIR"] == "/kaggle/working/pr106_latent/latent_run"


def test_pr106_latent_score_table_rejects_nonportable_archive_path() -> None:
    spec = Pr106LatentScoreTableSpec(
        job_name="job",
        pr106_archive="/tmp/archive.zip",
    )

    with pytest.raises(ValueError, match="repo-relative"):
        spec.validate()


def test_pr106_latent_dispatch_claim_uses_lane_and_job() -> None:
    spec = Pr106LatentScoreTableSpec(
        job_name="kaggle_pr106_latent_test",
        pr106_archive="inputs/pr106_archive.zip",
    )

    claim = dispatch_claim_spec(
        spec,
        platform="kaggle",
        agent="codex:test",
        predicted_eta_hours=1.0,
        notes="latent test",
    )

    assert claim.lane_id == "lane_pr106_latent_sidecar"
    assert claim.instance_job_id == "kaggle_pr106_latent_test"
    assert claim.platform == "kaggle"
    assert claim.agent == "codex:test"
