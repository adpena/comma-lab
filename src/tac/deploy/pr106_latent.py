# SPDX-License-Identifier: MIT
"""Provider-neutral PR106 latent-sidecar score-table dispatch contract."""
from __future__ import annotations

from dataclasses import dataclass

from tac.deploy.claims import DispatchClaimSpec, predicted_eta

REMOTE_SCRIPT = "scripts/remote_lane_pr106_latent_sidecar.sh"
SCORE_TABLE_LANE_ID = "lane_pr106_latent_sidecar"
SCORE_TABLE_ROLE = "pr106_latent_score_table_cuda"
FORMAT0C_SOURCE_ARCHIVE = (
    "experiments/results/pr106_format0c_exact_radix_candidate_20260515_codex/candidates/"
    "pr101_hdm9_hlm3_magicless_exact_radix_dim_fixed_meta_noop_rank_elided_sidecar_format_0x0c.archive.zip"
)
FORMAT0C_ARCHIVE_MEMBER = "x"
DEFAULT_RUNTIME_DIR = "submissions/pr106_latent_sidecar_r2_pr101_grammar"
DEFAULT_DELTA_RADIUS = 2


@dataclass(frozen=True)
class Pr106LatentScoreTableSpec:
    """Provider-neutral runtime contract for PR106 latent sidecar scoring."""

    job_name: str
    pr106_archive: str = FORMAT0C_SOURCE_ARCHIVE
    archive_member: str = FORMAT0C_ARCHIVE_MEMBER
    runtime_dir: str = DEFAULT_RUNTIME_DIR
    delta_radius: int = DEFAULT_DELTA_RADIUS
    latent_dim: int = 28
    n_pairs: int = 600
    batch_pairs: int = 2
    candidate_batch_size: int = 8
    sidecar_top_k: int = 600
    lane_id: str = SCORE_TABLE_LANE_ID

    def validate(self) -> None:
        if not self.job_name.strip():
            raise ValueError("PR106 latent score-table dispatch requires job_name")
        if not self.pr106_archive.strip():
            raise ValueError("PR106 latent score-table dispatch requires pr106_archive")
        if self.pr106_archive.startswith("/"):
            raise ValueError("pr106_archive must be repo-relative for portable dispatch")
        if not self.archive_member.strip():
            raise ValueError("archive_member must be explicit; use 0.bin or x")
        if "/" in self.archive_member or self.archive_member in {".", ".."}:
            raise ValueError("archive_member must be a single ZIP member name")
        if not self.runtime_dir.strip():
            raise ValueError("runtime_dir must be non-empty")
        if self.runtime_dir.startswith("/"):
            raise ValueError("runtime_dir must be repo-relative for portable dispatch")
        if self.delta_radius < 0:
            raise ValueError("delta_radius must be >= 0")
        if self.latent_dim <= 0:
            raise ValueError("latent_dim must be > 0")
        if self.n_pairs <= 0:
            raise ValueError("n_pairs must be > 0")
        if self.batch_pairs <= 0:
            raise ValueError("batch_pairs must be > 0")
        if self.candidate_batch_size <= 0:
            raise ValueError("candidate_batch_size must be > 0")
        if self.sidecar_top_k <= 0:
            raise ValueError("sidecar_top_k must be > 0")
        if not self.lane_id.strip():
            raise ValueError("lane_id must be non-empty")


def score_table_env(
    spec: Pr106LatentScoreTableSpec,
    *,
    output_dir: str | None = None,
    include_log_dir: bool = True,
) -> dict[str, str]:
    """Return the canonical environment consumed by the remote latent lane."""

    spec.validate()
    env = {
        "PR106_LATENT_MODE": "score_table",
        "PR106_ARCHIVE": spec.pr106_archive,
        "PR106_ARCHIVE_MEMBER": spec.archive_member,
        "PR106_RUNTIME_DIR": spec.runtime_dir,
        "PR106_LATENT_DELTA_RADIUS": str(spec.delta_radius),
        "PR106_LATENT_N_PAIRS": str(spec.n_pairs),
        "PR106_LATENT_DIM": str(spec.latent_dim),
        "PR106_LATENT_SCORE_TABLE_BATCH_PAIRS": str(spec.batch_pairs),
        "PR106_LATENT_SCORE_TABLE_CANDIDATE_BATCH_SIZE": str(spec.candidate_batch_size),
        "PR106_LATENT_SCORE_TABLE_LANE_ID": spec.lane_id,
        "PR106_LATENT_SCORE_TABLE_INSTANCE_JOB_ID": spec.job_name,
        "SIDECAR_TOP_K": str(spec.sidecar_top_k),
    }
    if include_log_dir:
        if output_dir is None:
            raise ValueError("output_dir is required when include_log_dir=True")
        env["PR106_LATENT_LOG_DIR"] = f"{output_dir}/latent_run"
    return env


def dispatch_claim_spec(
    spec: Pr106LatentScoreTableSpec,
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
    "DEFAULT_DELTA_RADIUS",
    "DEFAULT_RUNTIME_DIR",
    "FORMAT0C_ARCHIVE_MEMBER",
    "FORMAT0C_SOURCE_ARCHIVE",
    "REMOTE_SCRIPT",
    "SCORE_TABLE_LANE_ID",
    "SCORE_TABLE_ROLE",
    "Pr106LatentScoreTableSpec",
    "dispatch_claim_spec",
    "score_table_env",
]
