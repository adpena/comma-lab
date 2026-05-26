# SPDX-License-Identifier: MIT
"""META-LIFT-1 cross-substrate master-gradient analyzer (canonical helper).

Per the 11th standing directive ORDER-MATTERS discipline: this module is the
ONE canonical cross-substrate analyzer; per-substrate consumption happens
SECOND through downstream Catalog #354 exploit consumers + the new cathedral
consumer sister that auto-discovers this analyzer's output.

Mathematical contract (per canonical equation #344 family):

  Per-substrate Taylor expansion (canonical equation
  ``per_pair_master_gradient_score_impact_taylor_v1``, already registered):

    ΔS_i ≈ <∇S_i, Δθ_i>

  Per-axis Taylor projection (NEW META-pattern; FORMALIZATION_PENDING
  per canonical equation
  ``cross_substrate_master_gradient_aggregate_ranking_taylor_savings_v1``):

    ΔS_axis_i = <∇S_axis_i, Δθ_i> where axis ∈ {seg, pose, rate}

  Cauchy-Schwarz upper bound across substrates:

    |Σ_i ΔS_i| ≤ Σ_i ||∇S_i||_2 · ||Δθ_i||_2

  Ranking metric: cross-substrate Taylor residual (potential ΔS per byte
  perturbation budget) per axis. We RANK substrates by per-axis-Taylor-
  projection-per-byte normalised against the L2 norm of the gradient so
  the ranking is comparable across substrates with different byte counts:

    leverage_i_axis = ||∇S_axis_i||_2 / sqrt(N_bytes_i)

  This is the canonical surrogate for "which substrate × which axis has
  the highest per-byte leverage to ΔS in AGGREGATE across the corpus".

All outputs are OBSERVABILITY-ONLY per Catalog #341 + CLAUDE.md
"Apples-to-apples evidence discipline":

  - ``axis_tag = "[predicted]"``
  - ``score_claim = False``
  - ``promotable = False``
  - ``evidence_grade = "[predicted; cross-substrate-aggregate-Taylor]"``

Promotion to a contest score signal REQUIRES paired-CUDA empirical anchor.

Architecture (Catalog #230 sister-disjoint):

  - Inputs: per-substrate authoritative master-gradient anchors via
    :func:`tac.master_gradient_consumers.load_aggregate_gradient_from_anchor`
    + :func:`tac.master_gradient.latest_anchor_for_archive`
  - Outputs: ``CrossSubstrateMasterGradientAnalysis`` frozen dataclass
    persisted to fcntl-locked JSONL at
    ``.omx/state/cross_substrate_master_gradient_analyses.jsonl``
  - Discipline: Catalog #131 / #138 / #245 (fcntl-locked + strict-load +
    canonical 4-layer ledger) + #287 / #323 (placeholder rejection +
    canonical Provenance) + #341 (routing markers) + #356 (per-axis)

Per CLAUDE.md "MLX-first numpy-portable individually-fractal standing
directive": this module is pure-numpy (no MLX or PyTorch dependency at
analysis time). Inputs are loaded from npz/npy sidecars; outputs are
numpy arrays serialized to JSON for the canonical ledger.
"""
from __future__ import annotations

import datetime
import fcntl
import json
import os
import socket
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Module-level constants (canonical paths + schema versions)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]

CROSS_SUBSTRATE_ANALYSES_LEDGER_PATH = (
    REPO_ROOT / ".omx" / "state" / "cross_substrate_master_gradient_analyses.jsonl"
)
"""Canonical fcntl-locked JSONL append-only ledger.

Sister of ``.omx/state/master_gradient_anchors.jsonl`` (Catalog #245 +
#327) at the cross-substrate analysis sub-surface.
"""

_LEDGER_LOCK_PATH = (
    REPO_ROOT / ".omx" / "state" / ".cross_substrate_master_gradient_analyses.lock"
)

SCHEMA_VERSION = "cross_substrate_master_gradient_analysis_v1"

# Per Catalog #356 per-axis decomposition: the canonical axis labels
# match the contest scorer's 3-axis decomposition (seg + pose + rate).
VALID_AXIS_LABELS: frozenset[str] = frozenset({"seg", "pose", "rate"})

# Per Catalog #341 routing markers (Tier A observability-only).
PREDICTED_AXIS_TAG = "[predicted]"

# Per Catalog #287 placeholder rejection.
_PLACEHOLDER_RATIONALES: frozenset[str] = frozenset(
    {"<rationale>", "<reason>", "<rationale_here>", "<reason_here>", ""}
)

# Per CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent":
# the contest scorer canonical coefficients used for per-axis Taylor projection.
CANONICAL_SEG_COEFFICIENT = 100.0
"""ΔS = 100 · Δd_seg term coefficient (canonical contest scorer formula)."""

CANONICAL_RATE_DENOM_BYTES = 37_545_489
"""ΔS = 25 · Δarchive_bytes / 37_545_489 term denominator."""

CANONICAL_RATE_NUMERATOR = 25.0
"""ΔS = 25 · Δarchive_bytes / 37_545_489 term numerator."""

CANONICAL_POSE_SQRT_INNER = 10.0
"""ΔS = sqrt(10 · d_pose) term inner coefficient (non-linear; differenced)."""

# Per Catalog #344 + CLAUDE.md "Canonical equations + models registry":
# the canonical equation id this analyzer enables (FORMALIZATION_PENDING
# until paired-CUDA empirical anchor of cross-substrate ranking accuracy).
CANONICAL_EQUATION_ID = (
    "cross_substrate_master_gradient_aggregate_ranking_taylor_savings_v1"
)


# ---------------------------------------------------------------------------
# Custom exception (canonical strict-load fail-closed sister per Catalog #138)
# ---------------------------------------------------------------------------


class CrossSubstrateMasterGradientAnalysisCorruptError(RuntimeError):
    """Strict-load corruption marker per Catalog #138 fail-closed discipline.

    Sister of :class:`tac.master_gradient.MasterGradientAnchorsCorruptError`
    + :class:`tac.deploy.modal.call_id_ledger.CallIdLedgerCorruptError`. The
    canonical strict-load helper raises this so any future consumer of the
    cross-substrate analyses ledger inherits fail-closed-on-corruption
    semantics — a parse failure does NOT silently coerce missing rows to
    ``[]`` (the bug class Catalog #138 extincts).
    """


# ---------------------------------------------------------------------------
# Frozen dataclasses (canonical contract per Catalog #335 + #323)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CrossSubstrateAxisProjection:
    """Per-axis Taylor projection of a substrate's master-gradient.

    The canonical per-axis projection metric is the L2 norm of the gradient's
    axis column normalized against the byte count (the canonical surrogate
    for "per-byte leverage" comparable across substrates with different
    archive byte counts).

    Per Catalog #356 + CLAUDE.md "SegNet vs PoseNet importance —
    operating-point dependent": per-axis ranking can flip across operating
    points; the analyzer surfaces all three axes separately so downstream
    Pareto + bit-allocator consumers can route per-axis without
    recomputation.
    """

    axis: str
    """One of ``"seg"`` / ``"pose"`` / ``"rate"`` per Catalog #356."""

    gradient_l2_norm: float
    """``||∇S_axis||_2`` over all N_bytes for this substrate × axis."""

    per_byte_leverage: float
    """``||∇S_axis||_2 / sqrt(N_bytes)`` — comparable across substrates."""

    top_k_byte_indices: tuple[int, ...]
    """K highest-absolute-gradient byte indices (canonical K=64 ranked)."""

    top_k_byte_gradient_values: tuple[float, ...]
    """Signed gradient values at ``top_k_byte_indices``."""

    cauchy_schwarz_unit_perturbation_upper_bound: float
    """``||∇S_axis||_2 · ||Δθ||_2`` for unit perturbation (Δθ=1.0 unit norm)."""

    def __post_init__(self) -> None:
        if self.axis not in VALID_AXIS_LABELS:
            raise ValueError(
                f"axis must be in {sorted(VALID_AXIS_LABELS)}; got {self.axis!r}"
            )
        if not np.isfinite(self.gradient_l2_norm):
            raise ValueError("gradient_l2_norm must be finite")
        if self.gradient_l2_norm < 0:
            raise ValueError("gradient_l2_norm must be non-negative")
        if not np.isfinite(self.per_byte_leverage):
            raise ValueError("per_byte_leverage must be finite")
        if self.per_byte_leverage < 0:
            raise ValueError("per_byte_leverage must be non-negative")
        if not np.isfinite(self.cauchy_schwarz_unit_perturbation_upper_bound):
            raise ValueError("cauchy_schwarz_unit_perturbation_upper_bound must be finite")
        if len(self.top_k_byte_indices) != len(self.top_k_byte_gradient_values):
            raise ValueError(
                "top_k_byte_indices length must match top_k_byte_gradient_values length"
            )

    def as_dict(self) -> dict:
        return {
            "axis": self.axis,
            "gradient_l2_norm": float(self.gradient_l2_norm),
            "per_byte_leverage": float(self.per_byte_leverage),
            "top_k_byte_indices": list(self.top_k_byte_indices),
            "top_k_byte_gradient_values": list(self.top_k_byte_gradient_values),
            "cauchy_schwarz_unit_perturbation_upper_bound": float(
                self.cauchy_schwarz_unit_perturbation_upper_bound
            ),
        }


@dataclass(frozen=True)
class CrossSubstrateSubstrateRow:
    """Per-substrate row in a cross-substrate analysis.

    Sister of :class:`tac.master_gradient.MasterGradientAnchor` but at the
    cross-substrate META surface: this row references the substrate's
    authoritative anchor by sha256 + axis + hardware (per Catalog #127
    custody validator + #327 contest-axis authority) and surfaces the
    per-axis Taylor projections + Cauchy-Schwarz contribution.
    """

    archive_sha256: str
    """Canonical archive sha256 (per Catalog #245 + #323 provenance triple)."""

    measurement_axis: str
    """Per Catalog #127: ``[contest-CUDA]`` / ``[contest-CPU]`` / ``[macOS-CPU advisory]``."""

    measurement_hardware: str
    """Per Catalog #190: ``linux_x86_64_t4`` / ``darwin_arm64_m5_max_macos_cpu_advisory`` / etc."""

    measurement_call_id: str
    """Canonical Modal call_id OR ``macos_local_<utc>`` for advisory anchors."""

    n_bytes: int
    """Number of archive bytes covered by this substrate's gradient anchor."""

    aggregate_gradient_l2_norm: float
    """``sqrt(||∇S_seg||² + ||∇S_pose||² + ||∇S_rate||²)`` — aggregate L2."""

    per_axis_projections: tuple[CrossSubstrateAxisProjection, ...]
    """Per-axis Taylor projections (3-tuple: seg, pose, rate)."""

    aggregate_per_byte_leverage: float
    """``aggregate_gradient_l2_norm / sqrt(n_bytes)`` — comparable across substrates."""

    is_authoritative: bool
    """Per Catalog #127 + #327: contest-axis-authority verdict at load time."""

    def __post_init__(self) -> None:
        if not self.archive_sha256:
            raise ValueError("archive_sha256 must be non-empty")
        if not self.measurement_axis:
            raise ValueError("measurement_axis must be non-empty")
        if not self.measurement_hardware:
            raise ValueError("measurement_hardware must be non-empty")
        if not self.measurement_call_id:
            raise ValueError("measurement_call_id must be non-empty")
        if self.n_bytes <= 0:
            raise ValueError("n_bytes must be positive")
        if not np.isfinite(self.aggregate_gradient_l2_norm):
            raise ValueError("aggregate_gradient_l2_norm must be finite")
        if self.aggregate_gradient_l2_norm < 0:
            raise ValueError("aggregate_gradient_l2_norm must be non-negative")
        if len(self.per_axis_projections) != 3:
            raise ValueError(
                f"per_axis_projections must have exactly 3 entries (seg, pose, rate); got {len(self.per_axis_projections)}"
            )
        seen_axes = {proj.axis for proj in self.per_axis_projections}
        if seen_axes != VALID_AXIS_LABELS:
            raise ValueError(
                f"per_axis_projections must cover all axes {sorted(VALID_AXIS_LABELS)}; got {sorted(seen_axes)}"
            )
        if not np.isfinite(self.aggregate_per_byte_leverage):
            raise ValueError("aggregate_per_byte_leverage must be finite")
        if self.aggregate_per_byte_leverage < 0:
            raise ValueError("aggregate_per_byte_leverage must be non-negative")

    def as_dict(self) -> dict:
        return {
            "archive_sha256": self.archive_sha256,
            "measurement_axis": self.measurement_axis,
            "measurement_hardware": self.measurement_hardware,
            "measurement_call_id": self.measurement_call_id,
            "n_bytes": int(self.n_bytes),
            "aggregate_gradient_l2_norm": float(self.aggregate_gradient_l2_norm),
            "aggregate_per_byte_leverage": float(self.aggregate_per_byte_leverage),
            "is_authoritative": bool(self.is_authoritative),
            "per_axis_projections": [proj.as_dict() for proj in self.per_axis_projections],
        }


@dataclass(frozen=True)
class CrossSubstrateMasterGradientOpportunity:
    """A single ranked byte-saving opportunity surfaced by the analyzer.

    The opportunity is a ``(substrate, axis, byte_region)`` tuple with the
    canonical leverage metric. Downstream consumers (cathedral autopilot
    ranker, Pareto polytope solver, bit-allocator) consume these ranked
    opportunities to route per-axis dispatch decisions.

    Per Catalog #341 routing markers: every opportunity is observability-
    only by construction.
    """

    rank: int
    """1-indexed rank within the analysis (1 = highest leverage)."""

    archive_sha256: str
    """Substrate archive sha256 the opportunity targets."""

    axis: str
    """One of ``"seg"`` / ``"pose"`` / ``"rate"``."""

    per_byte_leverage: float
    """Per-byte leverage metric: ``||∇S_axis||_2 / sqrt(n_bytes)``."""

    cauchy_schwarz_unit_perturbation_upper_bound: float
    """Upper bound on ΔS at unit perturbation norm."""

    top_byte_indices: tuple[int, ...]
    """K highest-absolute-gradient byte indices for this opportunity."""

    measurement_axis: str
    """Per Catalog #127 custody axis token."""

    is_authoritative: bool
    """Per Catalog #127 + #327 authority verdict."""

    def __post_init__(self) -> None:
        if self.rank < 1:
            raise ValueError("rank must be 1-indexed (>= 1)")
        if not self.archive_sha256:
            raise ValueError("archive_sha256 must be non-empty")
        if self.axis not in VALID_AXIS_LABELS:
            raise ValueError(f"axis must be in {sorted(VALID_AXIS_LABELS)}; got {self.axis!r}")
        if not np.isfinite(self.per_byte_leverage) or self.per_byte_leverage < 0:
            raise ValueError("per_byte_leverage must be non-negative finite")
        if (
            not np.isfinite(self.cauchy_schwarz_unit_perturbation_upper_bound)
            or self.cauchy_schwarz_unit_perturbation_upper_bound < 0
        ):
            raise ValueError("cauchy_schwarz_unit_perturbation_upper_bound must be non-negative finite")
        if not self.measurement_axis:
            raise ValueError("measurement_axis must be non-empty")

    def as_dict(self) -> dict:
        return {
            "rank": int(self.rank),
            "archive_sha256": self.archive_sha256,
            "axis": self.axis,
            "per_byte_leverage": float(self.per_byte_leverage),
            "cauchy_schwarz_unit_perturbation_upper_bound": float(
                self.cauchy_schwarz_unit_perturbation_upper_bound
            ),
            "top_byte_indices": list(self.top_byte_indices),
            "measurement_axis": self.measurement_axis,
            "is_authoritative": bool(self.is_authoritative),
        }


@dataclass(frozen=True)
class CrossSubstrateMasterGradientAnalysis:
    """Canonical META-LIFT-1 cross-substrate analysis output.

    Sister of :class:`tac.master_gradient.MasterGradientAnchor` at the
    cross-substrate META surface. Persisted to the canonical fcntl-locked
    JSONL ledger at
    ``.omx/state/cross_substrate_master_gradient_analyses.jsonl``.

    Per Catalog #341 + CLAUDE.md "Apples-to-apples evidence discipline":
    every analysis is observability-only by construction (axis_tag,
    score_claim, promotable defaults are FALSE-class). Promotion of an
    analysis to a contest score signal REQUIRES paired-CUDA empirical
    anchor.
    """

    schema_version: str
    """Canonical schema version (current: ``cross_substrate_master_gradient_analysis_v1``)."""

    analysis_id: str
    """Deterministic id ``cross_subst_mg_<utc_compact>_<n_substrates>``."""

    measurement_utc: str
    """ISO-8601 UTC timestamp of analysis emission."""

    substrate_rows: tuple[CrossSubstrateSubstrateRow, ...]
    """Per-substrate rows (one per archive_sha256 covered)."""

    ranked_opportunities: tuple[CrossSubstrateMasterGradientOpportunity, ...]
    """Ranked byte-saving opportunities across substrates (sorted desc by leverage)."""

    cauchy_schwarz_aggregate_upper_bound: float
    """``Σ_i ||∇S_i||_2`` — canonical Cauchy-Schwarz aggregate bound at unit perturbation."""

    target_axes: tuple[str, ...]
    """Axes the analysis was conditioned on (subset of {"seg", "pose", "rate"})."""

    top_k_per_axis: int
    """K used for top-K ranking per axis (canonical default = 64)."""

    axis_tag: str
    """Always ``"[predicted]"`` per Catalog #341 + canonical Provenance."""

    score_claim: bool
    """Always ``False`` per CLAUDE.md "Apples-to-apples evidence discipline"."""

    promotable: bool
    """Always ``False`` per Catalog #341 + #192."""

    evidence_grade: str
    """Always ``"[predicted; cross-substrate-aggregate-Taylor]"``."""

    canonical_helper_invocation: str
    """``"tac.cross_substrate_master_gradient_analyzer.analyze_cross_substrate_master_gradients"``."""

    canonical_equation_id: str
    """``"cross_substrate_master_gradient_aggregate_ranking_taylor_savings_v1"`` per Catalog #344."""

    canonical_equation_status: str
    """``"FORMALIZATION_PENDING"`` until paired-CUDA empirical anchor lands per Catalog #344."""

    written_at_utc: str = ""
    written_pid: int = 0
    written_host: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must equal {SCHEMA_VERSION!r}; got {self.schema_version!r}"
            )
        if not self.analysis_id:
            raise ValueError("analysis_id must be non-empty")
        if not self.measurement_utc:
            raise ValueError("measurement_utc must be non-empty")
        if not self.substrate_rows:
            raise ValueError("substrate_rows must be non-empty")
        if self.axis_tag != PREDICTED_AXIS_TAG:
            raise ValueError(f"axis_tag must equal {PREDICTED_AXIS_TAG!r}; got {self.axis_tag!r}")
        if self.score_claim is not False:
            raise ValueError("score_claim must be False per Catalog #341")
        if self.promotable is not False:
            raise ValueError("promotable must be False per Catalog #341")
        if not self.evidence_grade.startswith("[predicted;"):
            raise ValueError("evidence_grade must start with '[predicted;' per Catalog #287 / #323")
        if self.top_k_per_axis < 1:
            raise ValueError("top_k_per_axis must be >= 1")
        if not self.target_axes:
            raise ValueError("target_axes must be non-empty")
        invalid_axes = set(self.target_axes) - VALID_AXIS_LABELS
        if invalid_axes:
            raise ValueError(
                f"target_axes contains invalid entries: {sorted(invalid_axes)}; "
                f"allowed: {sorted(VALID_AXIS_LABELS)}"
            )
        if self.canonical_equation_id != CANONICAL_EQUATION_ID:
            raise ValueError(
                f"canonical_equation_id must equal {CANONICAL_EQUATION_ID!r}; got {self.canonical_equation_id!r}"
            )
        if self.canonical_equation_status not in {"FORMALIZATION_PENDING", "REGISTERED"}:
            raise ValueError(
                "canonical_equation_status must be 'FORMALIZATION_PENDING' or 'REGISTERED' per Catalog #344"
            )
        if not np.isfinite(self.cauchy_schwarz_aggregate_upper_bound):
            raise ValueError("cauchy_schwarz_aggregate_upper_bound must be finite")
        if self.cauchy_schwarz_aggregate_upper_bound < 0:
            raise ValueError("cauchy_schwarz_aggregate_upper_bound must be non-negative")

    def as_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "analysis_id": self.analysis_id,
            "measurement_utc": self.measurement_utc,
            "substrate_rows": [row.as_dict() for row in self.substrate_rows],
            "ranked_opportunities": [opp.as_dict() for opp in self.ranked_opportunities],
            "cauchy_schwarz_aggregate_upper_bound": float(self.cauchy_schwarz_aggregate_upper_bound),
            "target_axes": list(self.target_axes),
            "top_k_per_axis": int(self.top_k_per_axis),
            "axis_tag": self.axis_tag,
            "score_claim": self.score_claim,
            "promotable": self.promotable,
            "evidence_grade": self.evidence_grade,
            "canonical_helper_invocation": self.canonical_helper_invocation,
            "canonical_equation_id": self.canonical_equation_id,
            "canonical_equation_status": self.canonical_equation_status,
            "written_at_utc": self.written_at_utc,
            "written_pid": int(self.written_pid),
            "written_host": self.written_host,
        }


# ---------------------------------------------------------------------------
# Core analyzer functions (canonical math + helper API)
# ---------------------------------------------------------------------------


def _axis_index(axis: str) -> int:
    """Canonical 3-axis ordering (seg=0, pose=1, rate=2)."""
    if axis == "seg":
        return 0
    if axis == "pose":
        return 1
    if axis == "rate":
        return 2
    raise ValueError(f"axis must be in {sorted(VALID_AXIS_LABELS)}; got {axis!r}")


def _utc_now_iso() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def _project_per_axis(
    gradient_array: np.ndarray,
    axis: str,
    *,
    top_k: int,
) -> CrossSubstrateAxisProjection:
    """Compute the canonical per-axis Taylor projection metrics.

    Per the module-level mathematical contract:

      - ``gradient_l2_norm`` = ``||∇S_axis||_2``
      - ``per_byte_leverage`` = ``||∇S_axis||_2 / sqrt(N_bytes)``
      - ``cauchy_schwarz_unit_perturbation_upper_bound`` =
        ``||∇S_axis||_2 · 1.0`` (unit-norm perturbation upper bound)
      - ``top_k_byte_indices`` = K highest-absolute-gradient indices
    """
    if gradient_array.ndim != 2 or gradient_array.shape[-1] != 3:
        raise ValueError(
            f"gradient_array must have shape (N_bytes, 3); got {gradient_array.shape}"
        )

    axis_col = gradient_array[:, _axis_index(axis)]
    n_bytes = int(axis_col.shape[0])

    if n_bytes == 0:
        raise ValueError("gradient_array must have at least one byte (N_bytes > 0)")

    # L2 norm (canonical Frobenius-restriction to the axis column).
    gradient_l2_norm = float(np.linalg.norm(axis_col, ord=2))
    per_byte_leverage = gradient_l2_norm / float(np.sqrt(n_bytes))

    # Cauchy-Schwarz upper bound at unit perturbation: ||∇S||_2 · ||Δθ||_2
    # = ||∇S||_2 · 1.0
    cauchy_schwarz_unit_perturbation_upper_bound = gradient_l2_norm

    # Top-K highest-absolute-gradient byte indices (canonical greedy ranking
    # surrogate for per-byte sensitivity per CLAUDE.md "Bit-level
    # deconstruction and entropy discipline").
    k_clipped = min(top_k, n_bytes)
    abs_col = np.abs(axis_col)
    # argpartition for O(N + K log K) selection; full sort would be O(N log N).
    top_unsorted = (
        np.argpartition(-abs_col, k_clipped - 1)[:k_clipped]
        if k_clipped < n_bytes
        else np.arange(n_bytes)
    )
    # Sort top-K by descending |gradient|.
    sort_order = np.argsort(-abs_col[top_unsorted])
    top_indices = top_unsorted[sort_order]
    top_gradients = axis_col[top_indices]

    return CrossSubstrateAxisProjection(
        axis=axis,
        gradient_l2_norm=gradient_l2_norm,
        per_byte_leverage=per_byte_leverage,
        top_k_byte_indices=tuple(int(i) for i in top_indices),
        top_k_byte_gradient_values=tuple(float(v) for v in top_gradients),
        cauchy_schwarz_unit_perturbation_upper_bound=cauchy_schwarz_unit_perturbation_upper_bound,
    )


def compute_cauchy_schwarz_cross_substrate_bound(
    substrate_rows: Sequence[CrossSubstrateSubstrateRow],
) -> float:
    """Compute the canonical cross-substrate Cauchy-Schwarz upper bound.

    Mathematical contract:

      |Σ_i ΔS_i| ≤ Σ_i ||∇S_i||_2 · ||Δθ_i||_2

    At unit per-substrate perturbation norm (``||Δθ_i||_2 = 1``) this
    reduces to:

      bound = Σ_i ||∇S_i||_2

    where ``||∇S_i||_2`` is the substrate's aggregate gradient L2 norm
    (Frobenius norm restricted to the 3 axis columns combined).

    Per Catalog #318 raw-byte-authority guard: this bound is NOT a score
    derivative; the canonical Provenance umbrella per Catalog #323 marks
    every analysis row as ``score_claim=False`` + ``promotable=False``.

    Args:
        substrate_rows: per-substrate rows from a cross-substrate analysis.

    Returns:
        The canonical Cauchy-Schwarz aggregate upper bound at unit
        perturbation norm.
    """
    if not substrate_rows:
        raise ValueError("substrate_rows must be non-empty")
    bound = 0.0
    for row in substrate_rows:
        bound += float(row.aggregate_gradient_l2_norm)
    return bound


def rank_byte_saving_opportunities_by_cross_substrate_taylor_residual(
    substrate_rows: Sequence[CrossSubstrateSubstrateRow],
    *,
    target_axes: Sequence[str],
    top_n: int = 16,
) -> tuple[CrossSubstrateMasterGradientOpportunity, ...]:
    """Rank cross-substrate byte-saving opportunities by per-byte leverage.

    The canonical leverage metric per (substrate, axis) pair is:

      leverage = ||∇S_axis||_2 / sqrt(N_bytes)

    Opportunities are ranked DESC by per-byte leverage so highest-EV
    candidates appear first. Per Catalog #356 per-axis decomposition the
    ranking is per-axis-aware (an opportunity entry names the specific
    axis where the leverage was measured).

    Per Catalog #341 routing markers: every returned opportunity is
    observability-only (``is_authoritative`` propagates from the substrate
    row's authority verdict but the OPPORTUNITY rank itself is
    ``[predicted]``-grade).

    Args:
        substrate_rows: per-substrate rows from a cross-substrate analysis.
        target_axes: axes to rank across (subset of {"seg", "pose", "rate"}).
        top_n: number of opportunities to return (default 16).

    Returns:
        Top-N opportunities sorted DESC by per-byte leverage.
    """
    if not substrate_rows:
        raise ValueError("substrate_rows must be non-empty")
    if not target_axes:
        raise ValueError("target_axes must be non-empty")
    invalid = set(target_axes) - VALID_AXIS_LABELS
    if invalid:
        raise ValueError(f"target_axes contains invalid entries: {sorted(invalid)}")
    if top_n < 1:
        raise ValueError("top_n must be >= 1")

    # Collect all (substrate, axis) leverage entries.
    entries: list[tuple[float, CrossSubstrateSubstrateRow, CrossSubstrateAxisProjection]] = []
    for row in substrate_rows:
        for proj in row.per_axis_projections:
            if proj.axis not in target_axes:
                continue
            entries.append((proj.per_byte_leverage, row, proj))

    # Sort DESC by leverage; tie-break by archive_sha256 + axis for determinism.
    entries.sort(key=lambda e: (-e[0], e[1].archive_sha256, e[2].axis))

    # Top-N projection.
    selected = entries[:top_n]

    opportunities: list[CrossSubstrateMasterGradientOpportunity] = []
    for rank, (leverage, row, proj) in enumerate(selected, start=1):
        opportunities.append(
            CrossSubstrateMasterGradientOpportunity(
                rank=rank,
                archive_sha256=row.archive_sha256,
                axis=proj.axis,
                per_byte_leverage=leverage,
                cauchy_schwarz_unit_perturbation_upper_bound=proj.cauchy_schwarz_unit_perturbation_upper_bound,
                top_byte_indices=proj.top_k_byte_indices,
                measurement_axis=row.measurement_axis,
                is_authoritative=row.is_authoritative,
            )
        )
    return tuple(opportunities)


def analyze_cross_substrate_master_gradients(
    substrate_inputs: Sequence[Mapping[str, object]],
    *,
    target_axes: Sequence[str] = ("seg", "pose", "rate"),
    top_k_per_axis: int = 64,
    top_n_opportunities: int = 16,
    measurement_utc: str | None = None,
) -> CrossSubstrateMasterGradientAnalysis:
    """Run the canonical META-LIFT-1 cross-substrate analysis.

    Per the 11th standing directive ORDER discipline: this is the ONE
    canonical cross-substrate analyzer; per-substrate consumption happens
    SECOND via downstream Catalog #354 exploit consumers + the sister
    cathedral consumer in
    :mod:`tac.cathedral_consumers.cross_substrate_master_gradient_analyzer_consumer`.

    Mathematical contract (per module docstring + canonical equation
    ``cross_substrate_master_gradient_aggregate_ranking_taylor_savings_v1``):

      1. For each substrate with an authoritative master-gradient anchor,
         load the (N_bytes, 3) aggregate gradient tensor.
      2. Compute per-axis Taylor projection metrics
         (``||∇S_axis||_2`` / per-byte leverage / top-K bytes).
      3. Compute the cross-substrate Cauchy-Schwarz aggregate upper bound.
      4. Rank byte-saving opportunities by per-byte leverage DESC.
      5. Emit a canonical analysis row with canonical Provenance markers.

    The analysis is observability-only per Catalog #341 + CLAUDE.md
    "Apples-to-apples evidence discipline". Promotion of any opportunity
    to a contest score signal REQUIRES paired-CUDA empirical anchor.

    Args:
        substrate_inputs: list of dicts with keys:
          - ``"gradient_array"``: (N_bytes, 3) numpy array (or callable
            returning one)
          - ``"archive_sha256"``: canonical archive sha256
          - ``"measurement_axis"``: per Catalog #127 axis token
          - ``"measurement_hardware"``: per Catalog #190 hardware substrate
          - ``"measurement_call_id"``: canonical Modal call_id or local id
          - ``"is_authoritative"``: per Catalog #127 + #327 authority bool
        target_axes: axes to include in the ranking (default all 3).
        top_k_per_axis: K for the per-axis top-K byte ranking (default 64).
        top_n_opportunities: top-N cross-substrate opportunities (default 16).
        measurement_utc: optional UTC timestamp; defaults to now.

    Returns:
        Canonical :class:`CrossSubstrateMasterGradientAnalysis`.
    """
    if not substrate_inputs:
        raise ValueError("substrate_inputs must be non-empty")
    invalid_axes = set(target_axes) - VALID_AXIS_LABELS
    if invalid_axes:
        raise ValueError(f"target_axes contains invalid entries: {sorted(invalid_axes)}")
    if top_k_per_axis < 1:
        raise ValueError("top_k_per_axis must be >= 1")
    if top_n_opportunities < 1:
        raise ValueError("top_n_opportunities must be >= 1")

    rows: list[CrossSubstrateSubstrateRow] = []
    for entry in substrate_inputs:
        gradient_value = entry.get("gradient_array")
        gradient_array = gradient_value() if callable(gradient_value) else gradient_value
        if gradient_array is None:
            raise ValueError("each substrate_input must provide 'gradient_array'")
        gradient_array = np.asarray(gradient_array, dtype=np.float64)
        if gradient_array.ndim != 2 or gradient_array.shape[-1] != 3:
            raise ValueError(
                f"gradient_array must have shape (N_bytes, 3); got {gradient_array.shape}"
            )

        # Per-axis projections (always all 3 axes to keep the substrate row
        # canonical; the OPPORTUNITY ranking filters to target_axes).
        projections = tuple(
            _project_per_axis(gradient_array, axis, top_k=top_k_per_axis)
            for axis in ("seg", "pose", "rate")
        )

        # Aggregate gradient L2 norm = Frobenius norm = sqrt(sum of axis L2²).
        agg_l2_sq = sum(proj.gradient_l2_norm ** 2 for proj in projections)
        agg_l2 = float(np.sqrt(agg_l2_sq))
        n_bytes = int(gradient_array.shape[0])
        agg_per_byte_leverage = agg_l2 / float(np.sqrt(n_bytes))

        rows.append(
            CrossSubstrateSubstrateRow(
                archive_sha256=str(entry["archive_sha256"]),
                measurement_axis=str(entry["measurement_axis"]),
                measurement_hardware=str(entry["measurement_hardware"]),
                measurement_call_id=str(entry["measurement_call_id"]),
                n_bytes=n_bytes,
                aggregate_gradient_l2_norm=agg_l2,
                per_axis_projections=projections,
                aggregate_per_byte_leverage=agg_per_byte_leverage,
                is_authoritative=bool(entry.get("is_authoritative", False)),
            )
        )

    # Sort rows deterministically by archive_sha256 so the analysis is
    # reproducible across runs (per Catalog #294 Dim 7 deterministic
    # reproducibility).
    rows.sort(key=lambda r: r.archive_sha256)

    # Cross-substrate Cauchy-Schwarz aggregate upper bound.
    cauchy_bound = compute_cauchy_schwarz_cross_substrate_bound(rows)

    # Ranked opportunities (per-axis aware).
    opportunities = rank_byte_saving_opportunities_by_cross_substrate_taylor_residual(
        rows,
        target_axes=target_axes,
        top_n=top_n_opportunities,
    )

    utc = measurement_utc or _utc_now_iso()
    compact_utc = utc.replace(":", "").replace("-", "")[:15]
    analysis_id = f"cross_subst_mg_{compact_utc}_{len(rows)}"

    return CrossSubstrateMasterGradientAnalysis(
        schema_version=SCHEMA_VERSION,
        analysis_id=analysis_id,
        measurement_utc=utc,
        substrate_rows=tuple(rows),
        ranked_opportunities=opportunities,
        cauchy_schwarz_aggregate_upper_bound=cauchy_bound,
        target_axes=tuple(target_axes),
        top_k_per_axis=top_k_per_axis,
        axis_tag=PREDICTED_AXIS_TAG,
        score_claim=False,
        promotable=False,
        evidence_grade="[predicted; cross-substrate-aggregate-Taylor]",
        canonical_helper_invocation=(
            "tac.cross_substrate_master_gradient_analyzer."
            "analyze_cross_substrate_master_gradients"
        ),
        canonical_equation_id=CANONICAL_EQUATION_ID,
        canonical_equation_status="FORMALIZATION_PENDING",
    )


# ---------------------------------------------------------------------------
# Canonical fcntl-locked JSONL ledger (Catalog #131 / #138 / #245 sister)
# ---------------------------------------------------------------------------


@contextmanager
def _ledger_lock(path: Path | None = None) -> Iterator[None]:
    """fcntl-LOCK_EX scope context manager per Catalog #131."""
    lock_path = path or _LEDGER_LOCK_PATH
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _atomic_append_jsonl(target: Path, line: str) -> None:
    """Atomic append via tmp + rename per Catalog #131 sister discipline.

    Sister of :func:`tac.master_gradient._atomic_write_append`.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    existing = target.read_bytes() if target.exists() else b""
    tmp_name = f".tmp.{uuid.uuid4().hex[:12]}"
    tmp_path = target.parent / (target.name + tmp_name)
    payload = existing + line.encode("utf-8") + b"\n"
    tmp_path.write_bytes(payload)
    os.replace(tmp_path, target)


def append_analysis_locked(
    analysis: CrossSubstrateMasterGradientAnalysis,
    *,
    path: Path | None = None,
) -> dict:
    """Append a cross-substrate analysis row to the canonical ledger.

    Sister of :func:`tac.master_gradient.append_anchor_locked`. fcntl-locked
    via :func:`_ledger_lock`; atomic write via :func:`_atomic_append_jsonl`
    per Catalog #131 sister discipline.

    Per CLAUDE.md HISTORICAL_PROVENANCE Catalog #110/#113: this ledger is
    APPEND-ONLY; existing rows are NEVER mutated. Recalibration produces a
    NEW row with a fresh ``analysis_id`` referencing prior rows by
    timestamp / canonical_equation_id.

    Returns the row dict as written.
    """
    target = path or CROSS_SUBSTRATE_ANALYSES_LEDGER_PATH
    row = analysis.as_dict()
    row["written_at_utc"] = _utc_now_iso()
    row["written_pid"] = os.getpid()
    row["written_host"] = socket.gethostname()
    line = json.dumps(row, sort_keys=True, allow_nan=False)
    with _ledger_lock(target.with_suffix(target.suffix + ".lock")):
        _atomic_append_jsonl(target, line)
    return row


def load_analyses_strict(path: Path | None = None) -> list[dict]:
    """Strict-load the canonical ledger per Catalog #138 fail-closed.

    Sister of :func:`tac.master_gradient.load_anchors_strict`. Raises
    :class:`CrossSubstrateMasterGradientAnalysisCorruptError` on JSON
    parse failure OR non-dict root rather than silently coercing to ``[]``
    (the bug class Catalog #138 extincts).
    """
    target = path or CROSS_SUBSTRATE_ANALYSES_LEDGER_PATH
    if not target.exists():
        return []
    rows: list[dict] = []
    raw = target.read_text(encoding="utf-8")
    for line_no, raw_line in enumerate(raw.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise CrossSubstrateMasterGradientAnalysisCorruptError(
                f"malformed JSON at {target}:{line_no}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise CrossSubstrateMasterGradientAnalysisCorruptError(
                f"non-dict row at {target}:{line_no}: {type(row).__name__}"
            )
        rows.append(row)
    return rows


__all__ = [
    "CANONICAL_EQUATION_ID",
    "CANONICAL_POSE_SQRT_INNER",
    "CANONICAL_RATE_DENOM_BYTES",
    "CANONICAL_RATE_NUMERATOR",
    "CANONICAL_SEG_COEFFICIENT",
    "CROSS_SUBSTRATE_ANALYSES_LEDGER_PATH",
    "PREDICTED_AXIS_TAG",
    "SCHEMA_VERSION",
    "VALID_AXIS_LABELS",
    "CrossSubstrateAxisProjection",
    "CrossSubstrateMasterGradientAnalysis",
    "CrossSubstrateMasterGradientAnalysisCorruptError",
    "CrossSubstrateMasterGradientOpportunity",
    "CrossSubstrateSubstrateRow",
    "analyze_cross_substrate_master_gradients",
    "append_analysis_locked",
    "compute_cauchy_schwarz_cross_substrate_bound",
    "load_analyses_strict",
    "rank_byte_saving_opportunities_by_cross_substrate_taylor_residual",
]
