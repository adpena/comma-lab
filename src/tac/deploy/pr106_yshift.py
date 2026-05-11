"""Provider-neutral PR106 y-shift score-table dispatch contract.

This module contains only the score-affecting lane contract shared by cloud
providers. Provider adapters own staging, SSH, Batch APIs, notebooks, and log
harvest; this module owns the environment and lane-claim metadata that the
remote y-shift score-table producer consumes.
"""
from __future__ import annotations

from dataclasses import dataclass

from tac.deploy.claims import DispatchClaimSpec, predicted_eta

REMOTE_SCRIPT = "scripts/remote_lane_pr106_yshift_sidechannel.sh"
SCORE_TABLE_LANE_ID = "lane_pr106_yshift_score_table"
SCORE_TABLE_ROLE = "pr106_yshift_score_table_cuda"


@dataclass(frozen=True)
class Pr106YshiftScoreTableSpec:
    """Provider-neutral runtime contract for the PR106 y-shift score table."""

    job_name: str
    pr106_archive: str
    candidate_radius: int = 3
    score_step: float = 1.0
    n_pairs: int = 600
    batch_pairs: int = 8
    candidate_batch_size: int = 32
    lane_id: str = SCORE_TABLE_LANE_ID

    def validate(self) -> None:
        """Fail closed on invalid dispatch-shaping parameters."""

        if not self.job_name.strip():
            raise ValueError("PR106 y-shift score-table dispatch requires job_name")
        if not self.pr106_archive.strip():
            raise ValueError("PR106 y-shift score-table dispatch requires pr106_archive")
        if self.pr106_archive.startswith("/"):
            raise ValueError("pr106_archive must be repo-relative for portable dispatch")
        if self.candidate_radius < 0:
            raise ValueError("candidate_radius must be >= 0")
        if self.score_step <= 0:
            raise ValueError("score_step must be > 0")
        if self.n_pairs <= 0:
            raise ValueError("n_pairs must be > 0")
        if self.batch_pairs <= 0:
            raise ValueError("batch_pairs must be > 0")
        if self.candidate_batch_size <= 0:
            raise ValueError("candidate_batch_size must be > 0")
        if not self.lane_id.strip():
            raise ValueError("lane_id must be non-empty")


def score_table_env(
    spec: Pr106YshiftScoreTableSpec,
    *,
    output_dir: str | None = None,
    include_log_dir: bool = True,
) -> dict[str, str]:
    """Return the canonical environment consumed by the remote y-shift lane."""

    spec.validate()
    env = {
        "PR106_YSHIFT_MODE": "score_table",
        "PR106_ARCHIVE": spec.pr106_archive,
        "PR106_YSHIFT_SCORE_TABLE_LANE_ID": spec.lane_id,
        "PR106_YSHIFT_SCORE_TABLE_INSTANCE_JOB_ID": spec.job_name,
        "PR106_YSHIFT_CANDIDATE_RADIUS": str(spec.candidate_radius),
        "PR106_YSHIFT_SCORE_STEP": str(spec.score_step),
        "PR106_YSHIFT_N_PAIRS": str(spec.n_pairs),
        "PR106_YSHIFT_SCORE_TABLE_BATCH_PAIRS": str(spec.batch_pairs),
        "PR106_YSHIFT_SCORE_TABLE_CANDIDATE_BATCH_SIZE": str(
            spec.candidate_batch_size
        ),
    }
    if include_log_dir:
        if output_dir is None:
            raise ValueError("output_dir is required when include_log_dir=True")
        env["PR106_YSHIFT_LOG_DIR"] = f"{output_dir}/yshift_run"
    return env


def dispatch_claim_spec(
    spec: Pr106YshiftScoreTableSpec,
    *,
    platform: str,
    agent: str,
    predicted_eta_hours: float,
    force: bool = False,
    notes: str = "",
) -> DispatchClaimSpec:
    """Return the canonical lane claim for any provider adapter."""

    spec.validate()
    return DispatchClaimSpec(
        lane_id=spec.lane_id,
        instance_job_id=spec.job_name,
        agent=agent,
        platform=platform,
        predicted_eta_utc=predicted_eta(predicted_eta_hours),
        force=force,
        notes=notes,
    )


__all__ = [
    "Pr106YshiftScoreTableSpec",
    "REMOTE_SCRIPT",
    "SCORE_TABLE_LANE_ID",
    "SCORE_TABLE_ROLE",
    "dispatch_claim_spec",
    "score_table_env",
]
