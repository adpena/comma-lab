"""Per-archive drift posterior — Bayesian update of CUDA/CPU drift PER ARCHIVE.

Sister-extension of :mod:`tac.optimization.cuda_cpu_axis_profile_registry`
(per-architecture-class drift posterior, landed 2026-05-08 per memory entry
``feedback_cuda_cpu_axis_profile_learning_layer_20260508``).

The per-class layer is the right granularity for *new* archives in a known
architecture class; the per-archive layer captured here lets us track *the
same archive over time* across re-evaluations (different hardware, different
upstream snapshot SHA, different decoder version) so we can detect
archive-specific drift without polluting the per-class posterior.

Why per-archive (in addition to per-class):

  - The per-class layer treats every archive in the class as i.i.d. samples
    of the SAME drift. That is correct for the FIRST eval of an unseen
    archive but loses signal when the SAME archive is re-evaluated.
  - The per-archive layer is the natural Welford running mean / std for
    repeated evals of one archive (e.g. a frontier candidate evaluated
    weekly across substrates).
  - Per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA": shippable
    archives are dual-eval'd; this module records the resulting per-archive
    ε posterior so the next dispatch can predict the gap before re-running.

Persistence: ``.omx/state/per_archive_drift_posterior.json`` is LIVE_STATE
per Catalog #113 (gitignored). The companion lock file ``.lock`` serializes
multi-process updates via :mod:`fcntl`.

Custody discipline (CLAUDE.md non-negotiable):
  - Authoritative tags only (``[contest-CUDA]`` /
    ``[contest-CPU GHA Linux x86_64]`` / sister GHA short forms).
  - macOS substrate refused unless explicit calibrated tag with override.
  - Tag → axis → hardware substrate validated jointly.
  - NaN / Inf / out-of-range refused before the Welford accumulator touches state.
  - Duplicate (archive_sha256, axis, hardware_substrate) refused for idempotence.

Cross-references
----------------
- :mod:`tac.continual_learning` — per-track + per-class posterior layer
- :mod:`tac.optimization.cuda_cpu_axis_profile_registry` — per-class drift profile
- ``feedback_cuda_cpu_axis_profile_learning_layer_20260508``
- ``feedback_dual_cpu_cuda_auth_eval_mandatory_20260508``
- ``feedback_5_beyond_phase4_modules_landed_20260509``

CLAUDE.md compliance tags
-------------------------
- ``custody_validator_required``
- ``no_macos_authoritative``
- ``welford_nan_guard``
- ``no_tmp_paths``
- ``live_state_gitignored_per_catalog_113``
- ``forbidden_score_claims_no_score_returned``
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import math
import os
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

PER_ARCHIVE_DRIFT_SCHEMA_VERSION = "tac_per_archive_drift_posterior_v1"
PER_ARCHIVE_DRIFT_EVIDENCE_GRADE = "[per-archive drift posterior; non-authoritative]"

DEFAULT_PER_ARCHIVE_PATH = Path(".omx/state/per_archive_drift_posterior.json")
DEFAULT_PER_ARCHIVE_LOCK_PATH = Path(".omx/state/.per_archive_drift_posterior.lock")

# Re-export shared validators from continual_learning so the two layers stay
# in lock-step. If the parent layer's custody rules tighten, both layers
# tighten together.
from tac.continual_learning import (  # noqa: E402
    AUTHORITATIVE_TAGS,
    ContestResult,
)


# ── Dataclasses ────────────────────────────────────────────────────────────


@dataclass
class PerArchiveObservation:
    """One per-archive observation that may update the posterior.

    Fields mirror :class:`tac.continual_learning.ContestResult` but are
    explicitly per-archive: the same archive_sha256 across multiple eval
    substrates / dates produces multiple PerArchiveObservation rows.
    """

    archive_sha256: str
    archive_bytes: int
    axis: str  # 'cuda' | 'cpu'
    hardware_substrate: str
    evidence_tag: str
    score_value: float
    cuda_pose: float | None = None
    cuda_seg: float | None = None
    cpu_pose: float | None = None
    cpu_seg: float | None = None
    observed_at_utc: str = ""
    notes: str = ""

    def validate_custody(self) -> tuple[bool, str]:
        """Joint (tag, axis, hardware_substrate) validation per CLAUDE.md.

        Delegate to :class:`tac.continual_learning.ContestResult` so the
        per-track, per-class, and per-archive posterior layers share one
        custody policy. Reimplementing tag predicates here risks drifting from
        the canonical validator and reintroducing Check #127's bug class.
        """
        proxy = ContestResult(
            axis=self.axis,
            hardware_substrate=self.hardware_substrate,
            architecture_class="per_archive_drift_posterior",
            score_value=self.score_value,
            evidence_tag=self.evidence_tag,
            archive_sha256=self.archive_sha256,
            archive_bytes=self.archive_bytes,
            cuda_pose=self.cuda_pose,
            cuda_seg=self.cuda_seg,
            cpu_pose=self.cpu_pose,
            cpu_seg=self.cpu_seg,
            notes=self.notes,
            observed_at_utc=self.observed_at_utc,
        )
        return proxy.validate_custody()


@dataclass
class PerAxisWelfordStats:
    """Welford running mean + variance for one (archive, axis) pair."""

    n: int = 0
    mean: float = 0.0
    sum_squared_dev: float = 0.0  # M2 in Welford notation
    last_updated_utc: str = ""
    last_substrate: str = ""

    def update(self, value: float, substrate: str) -> None:
        if not math.isfinite(value):
            raise ValueError(
                f"non-finite value {value!r} refused before Welford state touched"
            )
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        delta2 = value - self.mean
        self.sum_squared_dev += delta * delta2
        self.last_updated_utc = datetime.now(UTC).isoformat()
        self.last_substrate = substrate

    def variance(self) -> float:
        if self.n < 2:
            return 0.0
        return self.sum_squared_dev / (self.n - 1)

    def std(self) -> float:
        return self.variance() ** 0.5


@dataclass
class PerArchiveDriftPosterior:
    """Aggregate state of all per-archive drift posteriors.

    Keyed by archive_sha256 → :class:`PerArchivePosterior`. Each per-archive
    posterior holds a CUDA-axis Welford and a CPU-axis Welford so the gap
    ε = mean_cuda − mean_cpu has a per-archive estimate.
    """

    schema: str = PER_ARCHIVE_DRIFT_SCHEMA_VERSION
    evidence_grade: str = PER_ARCHIVE_DRIFT_EVIDENCE_GRADE
    per_archive: dict[str, "PerArchivePosterior"] = field(default_factory=dict)
    accepted_observation_count: int = 0
    refused_observation_count: int = 0
    last_updated_utc: str = ""

    def is_consistent(self) -> tuple[bool, list[str]]:
        problems: list[str] = []
        if self.schema != PER_ARCHIVE_DRIFT_SCHEMA_VERSION:
            problems.append(f"schema mismatch: {self.schema!r}")
        if self.refused_observation_count < 0:
            problems.append("refused_observation_count is negative")
        for sha, p in self.per_archive.items():
            if len(sha) != 64:
                problems.append(f"per_archive key {sha!r} not 64-char SHA-256")
            if not math.isfinite(p.cuda_axis.mean):
                problems.append(f"{sha}: cuda mean non-finite")
            if not math.isfinite(p.cpu_axis.mean):
                problems.append(f"{sha}: cpu mean non-finite")
            if p.cuda_axis.sum_squared_dev < 0 or p.cpu_axis.sum_squared_dev < 0:
                problems.append(f"{sha}: Welford accumulator negative")
        return (not problems), problems


@dataclass
class PerArchivePosterior:
    """Per-archive Welford state + provenance audit trail."""

    archive_sha256: str
    archive_bytes: int = 0
    cuda_axis: PerAxisWelfordStats = field(default_factory=PerAxisWelfordStats)
    cpu_axis: PerAxisWelfordStats = field(default_factory=PerAxisWelfordStats)
    observation_history: list[dict[str, Any]] = field(default_factory=list)

    def epsilon_estimate(self) -> Optional[float]:
        """Return the empirical CUDA−CPU gap, or None if either axis untouched."""
        if self.cuda_axis.n == 0 or self.cpu_axis.n == 0:
            return None
        return self.cuda_axis.mean - self.cpu_axis.mean

    def epsilon_uncertainty(self) -> Optional[float]:
        """Return sqrt(var_cuda + var_cpu) as a rough independence-assumption band."""
        if self.cuda_axis.n < 2 and self.cpu_axis.n < 2:
            return None
        var_cuda = self.cuda_axis.variance() if self.cuda_axis.n >= 2 else 0.0
        var_cpu = self.cpu_axis.variance() if self.cpu_axis.n >= 2 else 0.0
        return (var_cuda + var_cpu) ** 0.5


@dataclass
class PerArchiveUpdateResult:
    """Return value of :func:`per_archive_update`."""

    accepted: bool
    refusal_reason: str
    archive_sha256: str
    axis: str
    evidence_tag: str
    n_observations_after: int
    epsilon_estimate_after: Optional[float]
    notes: list[str] = field(default_factory=list)


# ── Read / write the posterior state ───────────────────────────────────────


def load_per_archive_posterior(path: Path | None = None) -> PerArchiveDriftPosterior:
    """Load posterior; return empty fresh state if file missing."""
    p = path or DEFAULT_PER_ARCHIVE_PATH
    if not p.is_file():
        return PerArchiveDriftPosterior()
    raw = json.loads(p.read_text(encoding="utf-8"))
    if raw.get("schema") != PER_ARCHIVE_DRIFT_SCHEMA_VERSION:
        raise ValueError(
            f"per-archive posterior schema mismatch at {p}: "
            f"got {raw.get('schema')!r}, want {PER_ARCHIVE_DRIFT_SCHEMA_VERSION!r}"
        )
    per_archive: dict[str, PerArchivePosterior] = {}
    for sha, body in raw.get("per_archive", {}).items():
        cuda_raw = body.get("cuda_axis", {})
        cpu_raw = body.get("cpu_axis", {})
        per_archive[sha] = PerArchivePosterior(
            archive_sha256=sha,
            archive_bytes=int(body.get("archive_bytes", 0)),
            cuda_axis=PerAxisWelfordStats(**cuda_raw),
            cpu_axis=PerAxisWelfordStats(**cpu_raw),
            observation_history=list(body.get("observation_history", [])),
        )
    return PerArchiveDriftPosterior(
        schema=raw["schema"],
        evidence_grade=raw.get("evidence_grade", PER_ARCHIVE_DRIFT_EVIDENCE_GRADE),
        per_archive=per_archive,
        accepted_observation_count=int(raw.get("accepted_observation_count", 0)),
        refused_observation_count=int(raw.get("refused_observation_count", 0)),
        last_updated_utc=raw.get("last_updated_utc", ""),
    )


def _serialize(posterior: PerArchiveDriftPosterior) -> dict[str, Any]:
    return {
        "schema": posterior.schema,
        "evidence_grade": posterior.evidence_grade,
        "per_archive": {
            sha: {
                "archive_sha256": p.archive_sha256,
                "archive_bytes": p.archive_bytes,
                "cuda_axis": asdict(p.cuda_axis),
                "cpu_axis": asdict(p.cpu_axis),
                "observation_history": p.observation_history,
            }
            for sha, p in posterior.per_archive.items()
        },
        "accepted_observation_count": posterior.accepted_observation_count,
        "refused_observation_count": posterior.refused_observation_count,
        "last_updated_utc": posterior.last_updated_utc,
    }


def save_per_archive_posterior(
    posterior: PerArchiveDriftPosterior,
    path: Path | None = None,
) -> None:
    """Atomic write — uses a unique tmp suffix to avoid clobbering parallel saves.

    For multi-process safety prefer :func:`per_archive_update_locked` which
    re-reads + re-applies the update inside an fcntl exclusive lock.
    """
    p = path or DEFAULT_PER_ARCHIVE_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = _serialize(posterior)
    tmp = p.with_suffix(p.suffix + f".tmp.{uuid.uuid4().hex[:12]}")
    try:
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, p)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


@contextlib.contextmanager
def _per_archive_lock(lock_path: Path | None = None):
    p = lock_path or DEFAULT_PER_ARCHIVE_LOCK_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(p), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield fd
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


# ── Posterior update (transaction) ─────────────────────────────────────────


def per_archive_update(
    posterior: PerArchiveDriftPosterior,
    obs: PerArchiveObservation,
    *,
    forbid_macos_promotion: bool = True,
) -> PerArchiveUpdateResult:
    """Apply one observation to the posterior; return a result record.

    Refusal policy:
      - non-AUTHORITATIVE evidence_tag → REFUSED
      - macOS substrate → REFUSED unless forbid_macos_promotion=False
      - tag/axis/hardware mismatch → REFUSED
      - non-finite score_value → REFUSED before Welford state touched
      - duplicate (archive_sha256, axis, hardware_substrate, observed_at_utc) → REFUSED
    """
    if not isinstance(obs, PerArchiveObservation):
        raise TypeError(
            f"per_archive_update expects PerArchiveObservation; got "
            f"{type(obs).__name__}"
        )

    notes: list[str] = []

    # NaN / Inf guard — protects Welford accumulator from corruption.
    if not math.isfinite(obs.score_value):
        posterior.refused_observation_count += 1
        return PerArchiveUpdateResult(
            accepted=False,
            refusal_reason=f"non-finite score_value {obs.score_value!r}",
            archive_sha256=obs.archive_sha256,
            axis=obs.axis,
            evidence_tag=obs.evidence_tag,
            n_observations_after=0,
            epsilon_estimate_after=None,
            notes=["non-finite score refused per Welford-corruption guard"],
        )

    # Custody validation.
    custody_ok, custody_reason = obs.validate_custody()
    macos_override = (
        not forbid_macos_promotion
        and obs.hardware_substrate.startswith("macos")
        and obs.evidence_tag in AUTHORITATIVE_TAGS
    )
    if not custody_ok and not macos_override:
        posterior.refused_observation_count += 1
        return PerArchiveUpdateResult(
            accepted=False,
            refusal_reason=custody_reason,
            archive_sha256=obs.archive_sha256,
            axis=obs.axis,
            evidence_tag=obs.evidence_tag,
            n_observations_after=0,
            epsilon_estimate_after=None,
            notes=["custody validation failed"],
        )
    if macos_override:
        notes.append("macOS substrate accepted via override; NOT 1:1 contest-compliant")

    # Idempotence — duplicate (sha, axis, substrate, observed_at_utc) refused.
    pa = posterior.per_archive.get(obs.archive_sha256)
    if pa is not None:
        for entry in pa.observation_history:
            if (
                entry.get("axis") == obs.axis
                and entry.get("hardware_substrate") == obs.hardware_substrate
                and entry.get("observed_at_utc") == obs.observed_at_utc
                and obs.observed_at_utc != ""
            ):
                posterior.refused_observation_count += 1
                return PerArchiveUpdateResult(
                    accepted=False,
                    refusal_reason=(
                        f"duplicate observation: archive {obs.archive_sha256[:12]}, "
                        f"axis {obs.axis}, substrate {obs.hardware_substrate}, "
                        f"at {obs.observed_at_utc}"
                    ),
                    archive_sha256=obs.archive_sha256,
                    axis=obs.axis,
                    evidence_tag=obs.evidence_tag,
                    n_observations_after=pa.cuda_axis.n + pa.cpu_axis.n,
                    epsilon_estimate_after=pa.epsilon_estimate(),
                    notes=["duplicate refused; idempotent"],
                )

    if not obs.observed_at_utc:
        obs.observed_at_utc = datetime.now(UTC).isoformat()

    if pa is None:
        pa = PerArchivePosterior(
            archive_sha256=obs.archive_sha256,
            archive_bytes=obs.archive_bytes,
        )
        posterior.per_archive[obs.archive_sha256] = pa

    if pa.archive_bytes == 0 and obs.archive_bytes > 0:
        pa.archive_bytes = obs.archive_bytes

    target_welford = pa.cuda_axis if obs.axis == "cuda" else pa.cpu_axis
    target_welford.update(obs.score_value, obs.hardware_substrate)

    pa.observation_history.append({
        "axis": obs.axis,
        "hardware_substrate": obs.hardware_substrate,
        "evidence_tag": obs.evidence_tag,
        "score_value": obs.score_value,
        "cuda_pose": obs.cuda_pose,
        "cuda_seg": obs.cuda_seg,
        "cpu_pose": obs.cpu_pose,
        "cpu_seg": obs.cpu_seg,
        "observed_at_utc": obs.observed_at_utc,
        "notes": obs.notes,
    })

    posterior.accepted_observation_count += 1
    posterior.last_updated_utc = datetime.now(UTC).isoformat()

    return PerArchiveUpdateResult(
        accepted=True,
        refusal_reason="",
        archive_sha256=obs.archive_sha256,
        axis=obs.axis,
        evidence_tag=obs.evidence_tag,
        n_observations_after=pa.cuda_axis.n + pa.cpu_axis.n,
        epsilon_estimate_after=pa.epsilon_estimate(),
        notes=notes,
    )


def per_archive_update_locked(
    obs: PerArchiveObservation,
    *,
    posterior_path: Path | None = None,
    lock_path: Path | None = None,
    forbid_macos_promotion: bool = True,
) -> PerArchiveUpdateResult:
    """fcntl-locked transactional update — safe for concurrent multi-process callers."""
    p_path = posterior_path or DEFAULT_PER_ARCHIVE_PATH
    l_path = lock_path or DEFAULT_PER_ARCHIVE_LOCK_PATH
    with _per_archive_lock(l_path):
        posterior = load_per_archive_posterior(p_path)
        result = per_archive_update(
            posterior, obs, forbid_macos_promotion=forbid_macos_promotion
        )
        save_per_archive_posterior(posterior, p_path)
        return result


# ── Query helpers ──────────────────────────────────────────────────────────


def query_archive_epsilon(
    posterior: PerArchiveDriftPosterior,
    archive_sha256: str,
) -> tuple[Optional[float], Optional[float], int]:
    """Return (epsilon_mean, epsilon_uncertainty, n_total_observations).

    Returns ``(None, None, 0)`` if the archive has no posterior yet.
    """
    pa = posterior.per_archive.get(archive_sha256)
    if pa is None:
        return None, None, 0
    return (
        pa.epsilon_estimate(),
        pa.epsilon_uncertainty(),
        pa.cuda_axis.n + pa.cpu_axis.n,
    )


def harvest_observations_from_iter(
    posterior: PerArchiveDriftPosterior,
    observations: Iterable[PerArchiveObservation],
    *,
    forbid_macos_promotion: bool = True,
) -> list[PerArchiveUpdateResult]:
    """Bulk apply a sequence of observations; return per-update records."""
    return [
        per_archive_update(
            posterior, obs, forbid_macos_promotion=forbid_macos_promotion
        )
        for obs in observations
    ]


__all__ = [
    "PER_ARCHIVE_DRIFT_SCHEMA_VERSION",
    "PER_ARCHIVE_DRIFT_EVIDENCE_GRADE",
    "DEFAULT_PER_ARCHIVE_PATH",
    "DEFAULT_PER_ARCHIVE_LOCK_PATH",
    "PerArchiveObservation",
    "PerAxisWelfordStats",
    "PerArchivePosterior",
    "PerArchiveDriftPosterior",
    "PerArchiveUpdateResult",
    "load_per_archive_posterior",
    "save_per_archive_posterior",
    "per_archive_update",
    "per_archive_update_locked",
    "query_archive_epsilon",
    "harvest_observations_from_iter",
]
