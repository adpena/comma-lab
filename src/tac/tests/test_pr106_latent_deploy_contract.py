# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest

from tac.deploy.pr106_latent import (
    DEFAULT_DELTA_RADIUS,
    DEFAULT_RUNTIME_DIR,
    FORMAT0C_ARCHIVE_MEMBER,
    FORMAT0C_SOURCE_ARCHIVE,
    Pr106LatentScoreTableSpec,
    dispatch_claim_spec,
    score_table_env,
)


def test_pr106_latent_score_table_defaults_target_format0c_packet() -> None:
    spec = Pr106LatentScoreTableSpec(job_name="kaggle_pr106_latent_test")

    env = score_table_env(spec, output_dir="/kaggle/working/pr106_latent")

    assert spec.pr106_archive == FORMAT0C_SOURCE_ARCHIVE
    assert spec.archive_member == FORMAT0C_ARCHIVE_MEMBER
    assert spec.runtime_dir == DEFAULT_RUNTIME_DIR
    assert spec.delta_radius == DEFAULT_DELTA_RADIUS
    assert env["PR106_ARCHIVE"] == FORMAT0C_SOURCE_ARCHIVE
    assert env["PR106_ARCHIVE_MEMBER"] == "x"
    assert env["PR106_RUNTIME_DIR"] == "submissions/pr106_latent_sidecar_r2_pr101_grammar"
    assert env["PR106_LATENT_DELTA_RADIUS"] == "2"


def test_pr106_latent_score_table_env_declares_score_table_contract() -> None:
    spec = Pr106LatentScoreTableSpec(
        job_name="kaggle_pr106_latent_test",
        pr106_archive="inputs/pr106_archive.zip",
        archive_member="0.bin",
        runtime_dir="submissions/pr106_latent_sidecar",
        delta_radius=2,
        latent_dim=28,
        batch_pairs=3,
        candidate_batch_size=5,
        sidecar_top_k=400,
    )

    env = score_table_env(spec, output_dir="/kaggle/working/pr106_latent")

    assert env["PR106_LATENT_MODE"] == "score_table"
    assert env["PR106_ARCHIVE"] == "inputs/pr106_archive.zip"
    assert env["PR106_ARCHIVE_MEMBER"] == "0.bin"
    assert env["PR106_RUNTIME_DIR"] == "submissions/pr106_latent_sidecar"
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


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("archive_member", "", "archive_member"),
        ("archive_member", "nested/x", "single ZIP member"),
        ("runtime_dir", "", "runtime_dir"),
        ("runtime_dir", "/tmp/runtime", "repo-relative"),
    ],
)
def test_pr106_latent_score_table_rejects_ambiguous_source_contract(
    field: str,
    value: str,
    match: str,
) -> None:
    kwargs = {"job_name": "job", field: value}
    spec = Pr106LatentScoreTableSpec(**kwargs)

    with pytest.raises(ValueError, match=match):
        spec.validate()


def test_pr106_latent_dispatch_claim_uses_lane_and_job() -> None:
    spec = Pr106LatentScoreTableSpec(
        job_name="kaggle_pr106_latent_test",
        pr106_archive="inputs/pr106_archive.zip",
        archive_member="0.bin",
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
