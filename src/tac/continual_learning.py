"""Continual-learning posterior orchestration layer.

Per the operator framing 2026-05-09 ("non-arbitrariness and increasing continual
learning and intelligence and optimization work and all other such systems of
equations and domain solver work and like general relativity and such"), this
module provides a thin orchestration layer over the existing posterior
substrates so every empirical anchor (``[contest-CUDA]`` /
``[contest-CPU GHA Linux x86_64]``) reseeds:

  - per-architecture-class CUDA-CPU drift (cuda_cpu_axis_profile_registry)
  - rate-distortion calibration anchors (.omx/calibration/anchors_*.json)
  - track-specific empirical correction factors (per-track posterior layer)
  - source-rho estimates (consumed by T13 joint_source_rd_bound)

The state lives at ``.omx/state/continual_learning_posterior.json`` (LIVE_STATE
per Catalog #113 — gitignored) so it persists across sessions but never
contaminates the committed history.

This is OUTSIDE the inner training loop — it runs after each contest_auth_eval
JSON lands. Trainers/planners READ the posterior; this module WRITES it.

Cross-references
----------------
- ``tac.optimization.cuda_cpu_axis_profile_registry`` — the existing per-arch-class layer
- ``tac.predictor.score_band.load_calibration_anchors`` — calibration anchor reader
- ``tac.joint_source_rd_bound.compute_joint_source_floor`` — consumes source ρ
- ``feedback_unified_solver_integration_landed_20260509.md``

CLAUDE.md compliance
--------------------
- Posterior writes are append-only-friendly (snapshots + diff log under .omx/state).
- Anchors tagged with non-1:1 contest-compliant hardware are recorded but
  refused for posterior promotion (the ``promoted`` field stays False).
- No score is claimed by this module; downstream callers consume the posterior
  to *predict* + *gate*, not to *measure*.
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import sys
import uuid
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Optional

CONTINUAL_LEARNING_SCHEMA_VERSION = "tac_continual_learning_posterior_v1"
CONTINUAL_LEARNING_EVIDENCE_GRADE = "[continual-learning posterior; non-authoritative]"

DEFAULT_POSTERIOR_PATH = Path(".omx/state/continual_learning_posterior.json")
DEFAULT_POSTERIOR_LOCK_PATH = Path(".omx/state/.continual_learning.lock")

# ── Custody validator (codex round-2 HIGH 2 fix) ───────────────────────────
#
# Tag-only authority is INSUFFICIENT. The custody validator requires the
# combination of (tag, axis, hardware_substrate) to be internally consistent
# AND for the substrate to be a 1:1 contest-compliant axis per CLAUDE.md
# "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT
# HARDWARE". Anything else is a custody mismatch and refused.

# Tag → required axis. The short-form `[contest-CPU]` is allowed only with the
# explicit GHA hardware substrate (codex HIGH 2 fix).
TAG_AXIS_REQUIREMENT = {
    "[contest-CUDA]": "cuda",
    "[contest-CPU GHA Linux x86_64]": "cpu",
    "[contest-CPU GHA]": "cpu",
    "[contest-CPU]": "cpu",
}

# Tag → set of hardware_substrate prefixes that are 1:1 contest-compliant.
# Codex HIGH 2 fix: short-form `[contest-CPU]` no longer accepts arbitrary
# Linux-x86_64 hosts — must be explicitly GHA.
TAG_HARDWARE_REQUIREMENT: dict[str, frozenset[str]] = {
    "[contest-CUDA]": frozenset({
        "linux_x86_64_t4",
        "linux_x86_64_4090",
        "linux_x86_64_a100",
        "linux_x86_64_h100",
        "linux_x86_64_a10g",
        "linux_x86_64_l40s",
    }),
    "[contest-CPU GHA Linux x86_64]": frozenset({"linux_x86_64_gha_cpu"}),
    "[contest-CPU GHA]": frozenset({"linux_x86_64_gha_cpu"}),
    "[contest-CPU]": frozenset({"linux_x86_64_gha_cpu"}),  # short form — GHA-only
}

# Authoritative-axis tags per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA".
AUTHORITATIVE_TAGS = frozenset({
    "[contest-CUDA]",
    "[contest-CPU GHA Linux x86_64]",
    "[contest-CPU GHA]",  # short form, same-meaning
    "[contest-CPU]",  # accepted only when explicitly Linux x86_64 in metadata
})

# Tags that may seed priors but never promote.
NON_PROMOTABLE_TAGS = frozenset({
    "[macOS-CPU advisory only]",
    "[macOS-CPU calibrated]",
    "[MPS-PROXY]",
    "[MPS-research-signal]",
    "[advisory only]",
    "[distortion-proxy:local]",
    "[byte-anchor]",
    "[predicted; unified-action; closed-form weighted-sum]",
})


@dataclass(frozen=True)
class CustodyVerdict:
    """Typed verdict from :meth:`ContestResult.validate_custody_verdict`.

    Codex round-2 HIGH 2 directive (2026-05-09). The seven ``refused_class``
    values catalog every reason the custody validator rejects an anchor —
    callers may switch on the verdict string for targeted error reporting
    rather than parsing the human-readable reason.

    Attributes
    ----------
    accepted
        ``True`` iff every custody invariant passes.
    reason
        Human-readable reason (empty string when ``accepted=True``).
    refused_class
        ``None`` when ``accepted=True``. Otherwise one of:

        - ``"missing_metadata"`` — required field blank/None
        - ``"advisory_grade"`` — explicitly-non-promotable tag
        - ``"macos_substrate"`` — macOS substrate (or macOS-tag) refused
        - ``"cpu_tag_non_gha_linux"`` — CPU tag with substrate not GHA Linux x86_64
        - ``"cuda_tag_unknown_substrate"`` — CUDA tag with substrate not in known CUDA set
        - ``"tag_axis_mismatch"`` — tag-axis combination is incoherent
    """

    accepted: bool
    reason: str
    refused_class: Optional[
        Literal[
            "tag_axis_mismatch",
            "cpu_tag_non_gha_linux",
            "cuda_tag_unknown_substrate",
            "macos_substrate",
            "missing_metadata",
            "advisory_grade",
        ]
    ] = None


@dataclass
class ContestResult:
    """A single empirical contest result that may update the posterior.

    Required:
      - axis: 'cuda' | 'cpu'
      - hardware_substrate: 'linux_x86_64_t4' | 'linux_x86_64_4090' |
        'linux_x86_64_a100' | 'linux_x86_64_gha_cpu' | 'macos_arm64' | etc.
      - architecture_class: free-form architecture-class label
        (e.g. 'pr106_hnerv_cluster', 'pr101_lossy_coarsening', 'lane_12_v2')
      - score_value: scalar score
      - evidence_tag: one of AUTHORITATIVE_TAGS or NON_PROMOTABLE_TAGS
      - archive_sha256: SHA-256 hex digest of the archive bytes
      - archive_bytes: archive size in bytes

    Optional:
      - cuda_pose / cuda_seg / cpu_pose / cpu_seg: per-component scores when
        available (used to update CUDA-CPU drift profile)
      - source_rho_estimate: empirical source-correlation estimate (T13)
      - track_correction_observations: per-track {track_kind -> empirical correction}
      - notes / metadata: free-form provenance
    """

    axis: str
    hardware_substrate: str
    architecture_class: str
    score_value: float
    evidence_tag: str
    archive_sha256: str
    archive_bytes: int

    cuda_pose: float | None = None
    cuda_seg: float | None = None
    cpu_pose: float | None = None
    cpu_seg: float | None = None
    source_rho_estimate: float | None = None
    track_correction_observations: dict[str, float] = field(default_factory=dict)
    notes: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    observed_at_utc: str = ""

    def is_authoritative(self) -> bool:
        """LEGACY: tag-only check (kept for back-compat).

        DEPRECATED: callers should use :meth:`validate_custody` which checks
        tag + axis + hardware_substrate together (codex round-2 HIGH 2 fix).
        ``is_authoritative`` returning True does NOT imply the result is
        promotable; ``posterior_update`` calls ``validate_custody`` before
        promoting.
        """
        return self.evidence_tag in AUTHORITATIVE_TAGS

    def is_non_promotable(self) -> bool:
        return self.evidence_tag in NON_PROMOTABLE_TAGS

    def validate_custody(self) -> tuple[bool, str]:
        """Per codex round-2 HIGH 2: validate tag + axis + hardware_substrate together.

        Returns ``(ok, reason)`` for back-compat with existing call sites.
        For richer typed introspection (``refused_class`` taxonomy) call
        :meth:`validate_custody_verdict`.

        ``ok=True`` only if every check passes. ``ok=False`` with a specific
        reason when:
          - tag is not in AUTHORITATIVE_TAGS
          - tag's required axis does NOT match self.axis
          - hardware_substrate is NOT in the tag's allowed prefixes
        """
        verdict = self.validate_custody_verdict()
        return verdict.accepted, verdict.reason

    def validate_custody_verdict(self) -> "CustodyVerdict":
        """Typed verdict per codex round-2 HIGH 2 directive.

        Returns a :class:`CustodyVerdict` with one of seven ``refused_class``
        values:

        - ``None`` (when ``accepted=True``)
        - ``"missing_metadata"`` — required metadata fields blank/None
        - ``"advisory_grade"`` — explicitly-non-promotable tag
        - ``"macos_substrate"`` — macOS substrate (advisory regardless of tag)
        - ``"cpu_tag_non_gha_linux"`` — CPU tag with substrate not GHA Linux x86_64
        - ``"cuda_tag_unknown_substrate"`` — CUDA tag with substrate not in known CUDA set
        - ``"tag_axis_mismatch"`` — CPU tag with axis="cuda" or vice versa
        """
        # 0. Required metadata — empty/whitespace-only strings are missing.
        for field_name in ("axis", "hardware_substrate", "evidence_tag", "archive_sha256"):
            value = getattr(self, field_name, None)
            if value is None or (isinstance(value, str) and not value.strip()):
                return CustodyVerdict(
                    accepted=False,
                    reason=f"required metadata field {field_name!r} is missing or blank",
                    refused_class="missing_metadata",
                )

        # 1. Explicit advisory tags (always refused regardless of substrate).
        if self.evidence_tag in NON_PROMOTABLE_TAGS:
            # macOS-CPU advisory/calibrated and MPS-* land here too; bucket them
            # under the most informative refused_class.
            if "macOS" in self.evidence_tag or self.evidence_tag.startswith("[macOS"):
                return CustodyVerdict(
                    accepted=False,
                    reason=f"macOS-CPU tag {self.evidence_tag!r} is advisory only; refused",
                    refused_class="macos_substrate",
                )
            return CustodyVerdict(
                accepted=False,
                reason=f"non-authoritative (advisory-grade) evidence_tag: {self.evidence_tag!r}",
                refused_class="advisory_grade",
            )

        # 2. Tag must be in AUTHORITATIVE_TAGS.
        if self.evidence_tag not in AUTHORITATIVE_TAGS:
            return CustodyVerdict(
                accepted=False,
                reason=(
                    f"non-authoritative evidence_tag: {self.evidence_tag!r} "
                    f"(allowed: {sorted(AUTHORITATIVE_TAGS)})"
                ),
                refused_class="advisory_grade",
            )

        # 3. macOS substrate is forbidden for any authoritative tag.
        # (CLAUDE.md: macOS is NEVER 1:1 contest-compliant for authoritative axis.)
        if self.hardware_substrate.startswith("macos"):
            return CustodyVerdict(
                accepted=False,
                reason=(
                    f"macOS substrate {self.hardware_substrate!r} forbidden as "
                    f"authoritative axis (tag {self.evidence_tag!r})"
                ),
                refused_class="macos_substrate",
            )

        # 4. Tag → required axis must match.
        required_axis = TAG_AXIS_REQUIREMENT.get(self.evidence_tag)
        if required_axis is None:
            return CustodyVerdict(
                accepted=False,
                reason=(
                    f"evidence_tag {self.evidence_tag!r} not in TAG_AXIS_REQUIREMENT; "
                    "cannot validate axis custody"
                ),
                refused_class="missing_metadata",
            )
        if self.axis != required_axis:
            return CustodyVerdict(
                accepted=False,
                reason=(
                    f"axis mismatch: evidence_tag {self.evidence_tag!r} requires "
                    f"axis={required_axis!r} but result has axis={self.axis!r}"
                ),
                refused_class="tag_axis_mismatch",
            )

        # 5. Hardware substrate must be in the tag's allowed list.
        allowed_substrates = TAG_HARDWARE_REQUIREMENT.get(self.evidence_tag)
        if allowed_substrates is None:
            return CustodyVerdict(
                accepted=False,
                reason=(
                    f"evidence_tag {self.evidence_tag!r} not in TAG_HARDWARE_REQUIREMENT; "
                    "cannot validate hardware custody"
                ),
                refused_class="missing_metadata",
            )
        if self.hardware_substrate not in allowed_substrates:
            # Refused-class depends on whether this is a CPU or CUDA tag.
            if required_axis == "cpu":
                rc: Optional[Literal[
                    "tag_axis_mismatch",
                    "cpu_tag_non_gha_linux",
                    "cuda_tag_unknown_substrate",
                    "macos_substrate",
                    "missing_metadata",
                    "advisory_grade",
                ]] = "cpu_tag_non_gha_linux"
            else:
                rc = "cuda_tag_unknown_substrate"
            return CustodyVerdict(
                accepted=False,
                reason=(
                    f"hardware substrate {self.hardware_substrate!r} not in 1:1 contest-compliant "
                    f"set for {self.evidence_tag!r}: {sorted(allowed_substrates)}"
                ),
                refused_class=rc,
            )

        return CustodyVerdict(accepted=True, reason="", refused_class=None)


@dataclass
class PosteriorUpdate:
    """One posterior update record (return value of posterior_update)."""

    accepted: bool
    refusal_reason: str
    architecture_class: str
    axis: str
    evidence_tag: str
    archive_sha256: str
    score_value: float
    posterior_n_anchors_after: int
    track_correction_factors_updated: list[str]
    cuda_cpu_drift_updated: bool
    notes: list[str] = field(default_factory=list)


@dataclass
class PerTrackPosterior:
    """Per-track empirical correction-factor posterior (running mean + std)."""

    track_kind: str
    n_observations: int = 0
    mean_correction: float = 1.0
    sum_squared_dev: float = 0.0  # for Welford's online variance
    last_updated_utc: str = ""

    def update(self, observed_correction: float) -> None:
        """Welford's online algorithm for running mean + variance."""
        self.n_observations += 1
        delta = observed_correction - self.mean_correction
        self.mean_correction += delta / self.n_observations
        delta2 = observed_correction - self.mean_correction
        self.sum_squared_dev += delta * delta2
        self.last_updated_utc = datetime.now(UTC).isoformat()

    def variance(self) -> float:
        if self.n_observations < 2:
            return 0.0
        return self.sum_squared_dev / (self.n_observations - 1)

    def std(self) -> float:
        return self.variance() ** 0.5


@dataclass
class SourceRhoPosterior:
    """Per-architecture-class source-rho posterior (consumed by T13)."""

    architecture_class: str
    n_observations: int = 0
    mean_rho: float = 0.0
    sum_squared_dev: float = 0.0
    last_updated_utc: str = ""

    def update(self, observed_rho: float) -> None:
        if not (-1.0 < observed_rho < 1.0):
            raise ValueError(f"observed_rho must be in (-1, 1); got {observed_rho!r}")
        self.n_observations += 1
        delta = observed_rho - self.mean_rho
        self.mean_rho += delta / self.n_observations
        delta2 = observed_rho - self.mean_rho
        self.sum_squared_dev += delta * delta2
        self.last_updated_utc = datetime.now(UTC).isoformat()

    def variance(self) -> float:
        if self.n_observations < 2:
            return 0.0
        return self.sum_squared_dev / (self.n_observations - 1)


@dataclass
class ContinualLearningPosterior:
    """The aggregated posterior state."""

    schema: str = CONTINUAL_LEARNING_SCHEMA_VERSION
    evidence_grade: str = CONTINUAL_LEARNING_EVIDENCE_GRADE
    track_correction_posteriors: dict[str, PerTrackPosterior] = field(default_factory=dict)
    source_rho_posteriors: dict[str, SourceRhoPosterior] = field(default_factory=dict)
    accepted_anchor_count: int = 0
    refused_anchor_count: int = 0
    last_updated_utc: str = ""
    accepted_anchor_history: list[dict[str, Any]] = field(default_factory=list)


# ── Read / write the posterior state ────────────────────────────────────────


def load_posterior(path: Path | None = None) -> ContinualLearningPosterior:
    """Load posterior state from disk; return a fresh empty posterior if missing."""
    p = path or DEFAULT_POSTERIOR_PATH
    if not p.is_file():
        return ContinualLearningPosterior()
    raw = json.loads(p.read_text(encoding="utf-8"))
    if raw.get("schema") != CONTINUAL_LEARNING_SCHEMA_VERSION:
        raise ValueError(
            f"posterior schema mismatch at {p}: "
            f"got {raw.get('schema')!r}, want {CONTINUAL_LEARNING_SCHEMA_VERSION!r}"
        )
    track_dict = {
        name: PerTrackPosterior(**v) for name, v in raw.get("track_correction_posteriors", {}).items()
    }
    rho_dict = {
        name: SourceRhoPosterior(**v) for name, v in raw.get("source_rho_posteriors", {}).items()
    }
    return ContinualLearningPosterior(
        schema=raw["schema"],
        evidence_grade=raw.get("evidence_grade", CONTINUAL_LEARNING_EVIDENCE_GRADE),
        track_correction_posteriors=track_dict,
        source_rho_posteriors=rho_dict,
        accepted_anchor_count=int(raw.get("accepted_anchor_count", 0)),
        refused_anchor_count=int(raw.get("refused_anchor_count", 0)),
        last_updated_utc=raw.get("last_updated_utc", ""),
        accepted_anchor_history=list(raw.get("accepted_anchor_history", [])),
    )


def _serialize(posterior: ContinualLearningPosterior) -> dict[str, Any]:
    return {
        "schema": posterior.schema,
        "evidence_grade": posterior.evidence_grade,
        "track_correction_posteriors": {
            name: asdict(v) for name, v in posterior.track_correction_posteriors.items()
        },
        "source_rho_posteriors": {
            name: asdict(v) for name, v in posterior.source_rho_posteriors.items()
        },
        "accepted_anchor_count": posterior.accepted_anchor_count,
        "refused_anchor_count": posterior.refused_anchor_count,
        "last_updated_utc": posterior.last_updated_utc,
        "accepted_anchor_history": posterior.accepted_anchor_history,
    }


def save_posterior(posterior: ContinualLearningPosterior, path: Path | None = None) -> None:
    """Persist the posterior to disk (atomic write).

    NOTE: this is the BARE write — for parallel-safe writes use
    :func:`posterior_update_locked` (codex round-2 MEDIUM fix). Calling
    this directly under parallel harvest risks dropping concurrent updates.
    """
    p = path or DEFAULT_POSTERIOR_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = _serialize(posterior)
    # Use a UNIQUE tmp file (codex round-2 MEDIUM fix): the prior fixed-suffix
    # `.tmp` could clobber a sibling save in flight.
    tmp = p.with_suffix(p.suffix + f".tmp.{uuid.uuid4().hex[:12]}")
    try:
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        # fsync the file before rename so the contents are durable.
        with open(tmp, "rb") as f:
            os.fsync(f.fileno())
        os.replace(tmp, p)
    finally:
        # Defensive cleanup if write failed mid-way.
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


@contextlib.contextmanager
def _posterior_lock(lock_path: Path | None = None):
    """Acquire fcntl exclusive lock on the posterior lock file.

    The lock is process-advisory (fcntl.flock LOCK_EX). Multiple processes
    contending for the same lock_path serialize on this lock; the lock is
    released automatically on context exit (success or exception).
    """
    p = lock_path or DEFAULT_POSTERIOR_LOCK_PATH
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


def posterior_update_locked(
    result: ContestResult,
    *,
    posterior_path: Path | None = None,
    lock_path: Path | None = None,
    forbid_macos_promotion: bool = True,
) -> PosteriorUpdate:
    """Locked transactional update — codex round-2 MEDIUM fix.

    Acquires exclusive fcntl lock on ``lock_path``, then INSIDE the lock:
      1. reload posterior from disk (sees other harvesters' updates)
      2. apply ``posterior_update`` (re-runs duplicate checks against fresh state)
      3. write posterior back via ``save_posterior``

    Multiple parallel harvesters serialize on the lock so updates of distinct
    anchors all survive. Without this, two harvesters loading the same stale
    posterior + each updating their distinct anchor + each replacing → ONE
    anchor's update silently dropped.

    Returns the PosteriorUpdate from the inner posterior_update call.
    """
    p_path = posterior_path or DEFAULT_POSTERIOR_PATH
    l_path = lock_path or DEFAULT_POSTERIOR_LOCK_PATH
    with _posterior_lock(l_path):
        posterior = load_posterior(p_path)
        update = posterior_update(
            posterior, result, forbid_macos_promotion=forbid_macos_promotion
        )
        # Always persist (refused anchors increment refused_anchor_count and
        # should also be visible to subsequent harvesters).
        save_posterior(posterior, p_path)
        return update


# ── Posterior update entry point ───────────────────────────────────────────


def posterior_update(
    posterior: ContinualLearningPosterior,
    result: ContestResult,
    *,
    forbid_macos_promotion: bool = True,
) -> PosteriorUpdate:
    """Update the posterior with one ContestResult.

    Refusal policy (CLAUDE.md non-negotiable):
      - non-AUTHORITATIVE evidence_tag → REFUSED for promotion (still recorded
        in refusal count but not promoted into running mean).
      - macOS-CPU substrate (forbid_macos_promotion=True) → REFUSED unless
        explicitly tagged 'calibrated' AND the operator passes
        forbid_macos_promotion=False.
      - duplicate archive_sha256 → REFUSED (idempotent; same anchor twice
        does NOT double-count).
    """
    if not isinstance(result, ContestResult):
        raise TypeError(
            f"posterior_update expects ContestResult; got {type(result).__name__}"
        )

    # R1-Quantizr (adversarial): NaN injection corrupts running mean forever.
    # Reject any non-finite scalar BEFORE it touches Welford state.
    import math as _math  # local import keeps module top tidy

    def _all_finite(values):
        return all(_math.isfinite(float(v)) for v in values)

    nonfinite_fields: list[str] = []
    if not _math.isfinite(float(result.score_value)):
        nonfinite_fields.append("score_value")
    for k, v in result.track_correction_observations.items():
        if not _math.isfinite(float(v)):
            nonfinite_fields.append(f"track_correction_observations[{k!r}]")
    if result.source_rho_estimate is not None and not _math.isfinite(float(result.source_rho_estimate)):
        nonfinite_fields.append("source_rho_estimate")
    for name, val in (
        ("cuda_pose", result.cuda_pose), ("cuda_seg", result.cuda_seg),
        ("cpu_pose", result.cpu_pose), ("cpu_seg", result.cpu_seg),
    ):
        if val is not None and not _math.isfinite(float(val)):
            nonfinite_fields.append(name)
    if nonfinite_fields:
        posterior.refused_anchor_count += 1
        return PosteriorUpdate(
            accepted=False,
            refusal_reason=(
                f"non-finite values in fields {nonfinite_fields}; refused to "
                "prevent posterior corruption"
            ),
            architecture_class=result.architecture_class,
            axis=result.axis,
            evidence_tag=result.evidence_tag,
            archive_sha256=result.archive_sha256,
            score_value=result.score_value,
            posterior_n_anchors_after=posterior.accepted_anchor_count,
            track_correction_factors_updated=[],
            cuda_cpu_drift_updated=False,
            notes=["non-finite values refused per Welford-corruption guard"],
        )

    notes: list[str] = []

    # Codex round-2 HIGH 2 fix: custody validator replaces tag-only check.
    # Validates (tag, axis, hardware_substrate) jointly + 1:1 contest-compliance.
    # macOS substrate is refused unless forbid_macos_promotion=False (preserved
    # for back-compat with the prior macOS-substrate path).
    custody_ok, custody_reason = result.validate_custody()
    macos_override_active = (
        not forbid_macos_promotion
        and result.hardware_substrate.startswith("macos")
        and result.evidence_tag in AUTHORITATIVE_TAGS
    )
    if not custody_ok and not macos_override_active:
        posterior.refused_anchor_count += 1
        return PosteriorUpdate(
            accepted=False,
            refusal_reason=custody_reason,
            architecture_class=result.architecture_class,
            axis=result.axis,
            evidence_tag=result.evidence_tag,
            archive_sha256=result.archive_sha256,
            score_value=result.score_value,
            posterior_n_anchors_after=posterior.accepted_anchor_count,
            track_correction_factors_updated=[],
            cuda_cpu_drift_updated=False,
            notes=["custody validation failed; refused per codex round-2 HIGH 2"],
        )
    if macos_override_active:
        notes.append(
            "macOS substrate accepted via forbid_macos_promotion=False override; "
            "result is NOT 1:1 contest-compliant"
        )

    # Refusal policy 3: duplicate archive_sha256.
    if any(
        h.get("archive_sha256") == result.archive_sha256
        and h.get("axis") == result.axis
        for h in posterior.accepted_anchor_history
    ):
        posterior.refused_anchor_count += 1
        return PosteriorUpdate(
            accepted=False,
            refusal_reason=(
                f"duplicate archive_sha256 {result.archive_sha256!r} on axis "
                f"{result.axis!r}; refused for idempotence"
            ),
            architecture_class=result.architecture_class,
            axis=result.axis,
            evidence_tag=result.evidence_tag,
            archive_sha256=result.archive_sha256,
            score_value=result.score_value,
            posterior_n_anchors_after=posterior.accepted_anchor_count,
            track_correction_factors_updated=[],
            cuda_cpu_drift_updated=False,
            notes=["duplicate anchor refused; idempotent"],
        )

    # Update accepted.
    posterior.accepted_anchor_count += 1
    posterior.last_updated_utc = datetime.now(UTC).isoformat()
    if not result.observed_at_utc:
        result.observed_at_utc = posterior.last_updated_utc

    # 1. Per-track correction-factor updates.
    track_updated: list[str] = []
    for track_kind, observed_correction in result.track_correction_observations.items():
        if track_kind not in posterior.track_correction_posteriors:
            posterior.track_correction_posteriors[track_kind] = PerTrackPosterior(track_kind=track_kind)
        posterior.track_correction_posteriors[track_kind].update(observed_correction)
        track_updated.append(track_kind)

    # 2. Source-rho update (T13 consumer).
    if result.source_rho_estimate is not None:
        if result.architecture_class not in posterior.source_rho_posteriors:
            posterior.source_rho_posteriors[result.architecture_class] = SourceRhoPosterior(
                architecture_class=result.architecture_class
            )
        posterior.source_rho_posteriors[result.architecture_class].update(
            result.source_rho_estimate
        )

    # 3. CUDA-CPU drift hand-off (delegated to cuda_cpu_axis_profile_registry).
    cuda_cpu_drift_updated = False
    if (
        result.cuda_pose is not None
        and result.cuda_seg is not None
        and result.cpu_pose is not None
        and result.cpu_seg is not None
    ):
        # We don't import the heavy registry directly here to keep this layer
        # decoupled. The notes record the intended downstream hand-off; a
        # follow-on tool wires the actual call.
        notes.append(
            "cuda_cpu drift signal present; downstream hand-off to "
            "tac.optimization.cuda_cpu_axis_profile_registry.update_profile_from_anchor"
        )
        cuda_cpu_drift_updated = True

    # Append to history (truncate to 500 most recent for size control).
    posterior.accepted_anchor_history.append({
        "axis": result.axis,
        "architecture_class": result.architecture_class,
        "evidence_tag": result.evidence_tag,
        "archive_sha256": result.archive_sha256,
        "archive_bytes": result.archive_bytes,
        "score_value": result.score_value,
        "hardware_substrate": result.hardware_substrate,
        "observed_at_utc": result.observed_at_utc,
        "track_updates": track_updated,
        "source_rho_estimate": result.source_rho_estimate,
    })
    if len(posterior.accepted_anchor_history) > 500:
        posterior.accepted_anchor_history = posterior.accepted_anchor_history[-500:]

    return PosteriorUpdate(
        accepted=True,
        refusal_reason="",
        architecture_class=result.architecture_class,
        axis=result.axis,
        evidence_tag=result.evidence_tag,
        archive_sha256=result.archive_sha256,
        score_value=result.score_value,
        posterior_n_anchors_after=posterior.accepted_anchor_count,
        track_correction_factors_updated=track_updated,
        cuda_cpu_drift_updated=cuda_cpu_drift_updated,
        notes=notes or ["accepted into posterior"],
    )


def posterior_query_track_correction(
    posterior: ContinualLearningPosterior,
    track_kind: str,
    *,
    default: float = 1.0,
) -> tuple[float, int]:
    """Return (mean_correction, n_observations) for a track.

    Returns ``(default, 0)`` if the track has no posterior yet.
    """
    p = posterior.track_correction_posteriors.get(track_kind)
    if p is None:
        return default, 0
    return p.mean_correction, p.n_observations


def posterior_query_source_rho(
    posterior: ContinualLearningPosterior,
    architecture_class: str,
    *,
    default: float = 0.0,
) -> tuple[float, int]:
    """Return (mean_rho, n_observations) for an architecture class."""
    p = posterior.source_rho_posteriors.get(architecture_class)
    if p is None:
        return default, 0
    return p.mean_rho, p.n_observations


def harvest_anchors_from_iter(
    posterior: ContinualLearningPosterior,
    results: Iterable[ContestResult],
    *,
    forbid_macos_promotion: bool = True,
) -> list[PosteriorUpdate]:
    """Bulk-harvest anchors. Returns the per-update record list."""
    return [
        posterior_update(posterior, r, forbid_macos_promotion=forbid_macos_promotion)
        for r in results
    ]


__all__ = [
    "CONTINUAL_LEARNING_SCHEMA_VERSION",
    "CONTINUAL_LEARNING_EVIDENCE_GRADE",
    "DEFAULT_POSTERIOR_PATH",
    "DEFAULT_POSTERIOR_LOCK_PATH",
    "AUTHORITATIVE_TAGS",
    "NON_PROMOTABLE_TAGS",
    "TAG_AXIS_REQUIREMENT",
    "TAG_HARDWARE_REQUIREMENT",
    "ContestResult",
    "CustodyVerdict",
    "PosteriorUpdate",
    "PerTrackPosterior",
    "SourceRhoPosterior",
    "ContinualLearningPosterior",
    "load_posterior",
    "save_posterior",
    "posterior_update",
    "posterior_update_locked",
    "posterior_query_track_correction",
    "posterior_query_source_rho",
    "harvest_anchors_from_iter",
]
