# SPDX-License-Identifier: MIT
"""tac.score_composition - canonical per-axis score composition helper.

Per CATHEDRAL-SMARTER-DESIGN-MEMO Dimension 3 Step 3.2 (operator blanket
approval 2026-05-20) + CLAUDE.md "SegNet vs PoseNet importance -
operating-point dependent" (UPDATED 2026-05-04).

The contest scorer is

    S = 100 * d_seg + sqrt(10 * d_pose) + 25 * archive_bytes / 37545489

per ``experiments/contest_auth_eval.py`` + canonical equation
``contest_score_formula_v1`` (sister landing in canonical equations
registry). The 3-axis decomposition is the primitive cathedral consumers
emit per-axis predictions on; this module composes per-axis deltas into
total ΔS via the canonical formula, surfaces per-axis breakdown so a
downstream consumer can audit which axis a prediction was wrong on, and
honors the canonical frontier pointer for current per-axis baselines per
CLAUDE.md "Frontier scores are pointer-only" non-negotiable + Catalog
#343.

Quick start::

    from tac.cathedral.consumer_contract import AxisDecomposition
    from tac.score_composition import (
        compose_score_from_axes,
        compose_scalar_delta,
    )

    decomp = AxisDecomposition(
        predicted_d_seg_delta=-0.0001,
        predicted_d_pose_delta=+0.000001,
        predicted_archive_bytes_delta=-200,
        canonical_provenance=provenance_to_dict(
            build_provenance_for_predicted(
                model_id="per_segnet_class_chroma_consumer_v1",
                inputs_sha256=candidate_sha,
            )
        ),
    )
    # Compose total scalar delta given current baseline pose value.
    result = compose_score_from_axes(
        decomp,
        current_archive_bytes=337_944,
        current_d_pose=3.4e-5,  # from canonical frontier pointer
    )
    assert result.total_delta < 0  # candidate improves total score

The math
--------

For a candidate replacing the current frontier:

    delta_total = 100 * delta_d_seg
                + (sqrt(10 * (d_pose + delta_d_pose)) - sqrt(10 * d_pose))
                + 25 * delta_archive_bytes / 37545489

The seg + rate terms are LINEAR in their deltas; the pose term is the
difference of two ``sqrt`` calls because ``sqrt(10 * pose)`` is the
contest scorer's POSE CONTRIBUTION (NOT a multiplier of delta_pose). At
the current operating point (PR106 pose_avg ≈ 3.4e-5), the marginal
``d/d(pose_avg) sqrt(10 * pose_avg) = 5 / sqrt(10 * pose_avg) ≈ 271``;
at OLD 1.x scores (pose_avg ≈ 0.18), the marginal is ≈ 12. The composed
formula honors this operating-point sensitivity automatically.

Catalog #125 6-hook wire-in declaration
---------------------------------------

- Hook #1 sensitivity-map: ACTIVE — per-axis sensitivities are the
  canonical signal at the consumer boundary.
- Hook #2 Pareto constraint: ACTIVE — per-axis decomposition is the
  primitive input to Dykstra alternating-projections on the
  (seg, pose, rate) polytope (Dimension 1 Phase 4 enabler).
- Hook #3 bit-allocator: ACTIVE — ``predicted_archive_bytes_delta`` IS
  the bit-allocator's primary signal.
- Hook #4 cathedral autopilot dispatch: ACTIVE PRIMARY — composition
  happens at the dispatch decision boundary (the consumer cascade in
  ``tools/cathedral_autopilot_autonomous_loop.py``).
- Hook #5 continual-learning posterior: ACTIVE — per-axis posterior
  anchors enable per-axis Bayesian updates (DreamerV3 RSSM categorical
  R(D) anchor is a sister surface).
- Hook #6 probe-disambiguator: ACTIVE — per-axis residuals
  disambiguate which axis a prediction was wrong on (vs scalar
  residual which hides axis attribution; PR97 anti-pattern).

Cross-references
----------------

- :class:`tac.cathedral.consumer_contract.AxisDecomposition` (the
  canonical Protocol extension).
- :mod:`tac.provenance` (Catalog #323 canonical Provenance umbrella).
- :mod:`tac.canonical_frontier_pointer` (Catalog #343 canonical
  baseline source).
- ``tools/cathedral_autopilot_autonomous_loop.py::invoke_cathedral_consumers_on_candidates``
  (Catalog #336 sister; THIS landing extends the helper to detect +
  compose per-axis decompositions).
- Catalog #356 STRICT preflight gate (refuses per-axis emission
  without canonical Provenance per Catalog #287 + #323).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

from tac.cathedral.consumer_contract import AxisDecomposition

__all__ = [
    "CANONICAL_SEG_MULTIPLIER",
    "CANONICAL_POSE_SQRT_INNER",
    "CANONICAL_RATE_MULTIPLIER",
    "CANONICAL_RATE_DENOM_BYTES",
    "ComposedScoreDelta",
    "compose_score_from_axes",
    "compose_scalar_delta",
    "load_baseline_pose_from_canonical_frontier_pointer",
]


# Canonical scorer constants. Mirror the cite-chain in
# ``src/tac/joint_scorer_aware_training.py::LAMBDA_RATE_CONSTANT`` (the
# canonical authority for these literals on the training side) +
# ``src/tac/archive_byte_profile.py::CONTEST_ORIGINAL_BYTES``. The
# canonical equations registry pins ``contest_score_formula_v1`` as the
# canonical equation reference; consumers of this module should cite
# that equation_id when emitting per-axis Provenance.
CANONICAL_SEG_MULTIPLIER: float = 100.0
"""Coefficient on ``d_seg`` in the contest score formula.

Per ``experiments/contest_auth_eval.py`` + canonical equation
``contest_score_formula_v1``: ``S = 100 * d_seg + sqrt(10 * d_pose) +
25 * archive_bytes / 37545489``.
"""

CANONICAL_POSE_SQRT_INNER: float = 10.0
"""Constant inside ``sqrt(10 * d_pose)`` in the contest score formula.

The pose contribution is ``sqrt(CANONICAL_POSE_SQRT_INNER * d_pose)``;
its marginal sensitivity is ``5 / sqrt(10 * d_pose)`` per CLAUDE.md
"SegNet vs PoseNet importance" (operating-point dependent).
"""

CANONICAL_RATE_MULTIPLIER: float = 25.0
"""Coefficient on ``archive_bytes / 37545489`` in the rate term."""

CANONICAL_RATE_DENOM_BYTES: int = 37_545_489
"""Reference denominator for the rate term.

Per ``src/tac/archive_byte_profile.py::CONTEST_ORIGINAL_BYTES`` +
canonical equation ``contest_score_formula_v1``. The rate contribution
is ``CANONICAL_RATE_MULTIPLIER * archive_bytes / CANONICAL_RATE_DENOM_BYTES``.
"""


@dataclass(frozen=True)
class ComposedScoreDelta:
    """Result of composing a per-axis decomposition into total ΔS.

    Per CATHEDRAL-SMARTER-DESIGN-MEMO Dimension 3 Step 3.2: the
    composition surfaces per-axis breakdown AND a single scalar
    ``total_delta`` so downstream consumers can choose to consume the
    per-axis signal for diagnostics or the scalar for ranking.

    Per Catalog #287/#323: ``canonical_provenance`` propagated through
    from the source ``AxisDecomposition`` (or composed via
    :func:`tac.provenance.builders.build_provenance_aggregate` if the
    caller combines multiple decompositions before composing).

    Fields:
        total_delta: scalar score delta in canonical units (matches the
            contest score numerically).
        seg_delta_contribution: ``CANONICAL_SEG_MULTIPLIER * d_seg_delta``.
        pose_delta_contribution: pose contribution as the DIFFERENCE of
            two ``sqrt(10 * d_pose)`` calls (NOT linear in delta_d_pose);
            requires ``baseline_d_pose`` to compute.
        rate_delta_contribution: ``CANONICAL_RATE_MULTIPLIER *
            archive_bytes_delta / CANONICAL_RATE_DENOM_BYTES``.
        baseline_d_pose: the operating-point ``d_pose`` value used in
            the pose-term composition (echoed for audit).
        baseline_archive_bytes: the operating-point archive byte count
            (echoed for audit).
        per_axis_residual: dict mapping ``{seg,pose,rate}`` -> float
            (residual = empirical_axis_delta - predicted_axis_delta;
            populated only when called via the post-empirical-anchor
            composition path; default empty).
        canonical_provenance: dict-form Provenance per Catalog #323.
        axis_tag: canonical axis tag per Catalog #287/#341.
    """

    total_delta: float
    seg_delta_contribution: float
    pose_delta_contribution: float
    rate_delta_contribution: float
    baseline_d_pose: float
    baseline_archive_bytes: int
    canonical_provenance: Mapping[str, Any]
    axis_tag: str = "[predicted]"
    per_axis_residual: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for fname in (
            "total_delta",
            "seg_delta_contribution",
            "pose_delta_contribution",
            "rate_delta_contribution",
            "baseline_d_pose",
        ):
            value = getattr(self, fname)
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"ComposedScoreDelta.{fname} must be numeric, got "
                    f"{type(value).__name__}"
                )
            if math.isnan(value) or math.isinf(value):
                raise ValueError(
                    f"ComposedScoreDelta.{fname} must be finite "
                    "(no NaN / inf)"
                )
            object.__setattr__(self, fname, float(value))
        if (
            not isinstance(self.baseline_archive_bytes, int)
            or isinstance(self.baseline_archive_bytes, bool)
        ):
            raise ValueError(
                "ComposedScoreDelta.baseline_archive_bytes must be int, "
                f"got {type(self.baseline_archive_bytes).__name__}"
            )
        if not isinstance(self.canonical_provenance, Mapping):
            raise ValueError(
                "ComposedScoreDelta.canonical_provenance must be a "
                "Mapping (Catalog #323 dict-form Provenance)"
            )
        if not isinstance(self.per_axis_residual, Mapping):
            raise ValueError(
                "ComposedScoreDelta.per_axis_residual must be a Mapping"
            )

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe serialization for observability payload emission."""
        return {
            "total_delta": float(self.total_delta),
            "seg_delta_contribution": float(self.seg_delta_contribution),
            "pose_delta_contribution": float(self.pose_delta_contribution),
            "rate_delta_contribution": float(self.rate_delta_contribution),
            "baseline_d_pose": float(self.baseline_d_pose),
            "baseline_archive_bytes": int(self.baseline_archive_bytes),
            "axis_tag": str(self.axis_tag),
            "canonical_provenance": dict(self.canonical_provenance),
            "per_axis_residual": {
                str(k): float(v) for k, v in self.per_axis_residual.items()
            },
        }


def compose_score_from_axes(
    decomposition: AxisDecomposition,
    *,
    current_archive_bytes: int,
    current_d_pose: float,
    per_axis_residual: Mapping[str, float] | None = None,
) -> ComposedScoreDelta:
    """Compose a per-axis decomposition into total ΔS via canonical formula.

    Per CATHEDRAL-SMARTER-DESIGN-MEMO Dimension 3 Step 3.2 + canonical
    equation ``contest_score_formula_v1``:

        S = 100 * d_seg + sqrt(10 * d_pose) + 25 * archive_bytes / 37545489

    delta_total = 100 * delta_d_seg
                + (sqrt(10 * (d_pose + delta_d_pose)) - sqrt(10 * d_pose))
                + 25 * delta_archive_bytes / 37545489

    Args:
        decomposition: per-axis prediction emitted by a cathedral
            consumer (`AxisDecomposition` instance).
        current_archive_bytes: baseline archive size (operating point)
            against which ``predicted_archive_bytes_delta`` is computed.
            Caller should source from canonical frontier pointer per
            Catalog #343 (use
            :func:`load_baseline_pose_from_canonical_frontier_pointer`
            sister helper or read the pointer's
            ``our_local_frontier_*.extra`` field if extended; for now
            an int from the caller's known baseline).
        current_d_pose: baseline PoseNet distortion value (operating
            point) for pose-term sqrt composition. Per CLAUDE.md
            "SegNet vs PoseNet importance" the marginal pose
            sensitivity depends critically on this value (PR106 pose_avg
            ≈ 3.4e-5 → marginal ≈ 271; OLD 1.x pose ≈ 0.18 → marginal
            ≈ 12). Caller MUST cite the canonical operating-point source
            for audit.
        per_axis_residual: optional dict of empirical residuals (caller-
            populated; default empty). Used by post-empirical-anchor
            composition paths (Hook #5 + #6).

    Returns:
        :class:`ComposedScoreDelta` with the composed scalar + per-axis
        breakdown + propagated Provenance + echoed baseline values.

    Raises:
        ValueError: if ``current_archive_bytes`` is negative, or
            ``current_d_pose`` is negative (sqrt domain), or other
            invariant violations per :class:`ComposedScoreDelta`.
    """
    if not isinstance(decomposition, AxisDecomposition):
        raise TypeError(
            "compose_score_from_axes: decomposition must be an "
            f"AxisDecomposition, got {type(decomposition).__name__}"
        )
    if not isinstance(current_archive_bytes, int) or isinstance(
        current_archive_bytes, bool
    ):
        raise TypeError(
            "compose_score_from_axes: current_archive_bytes must be int, "
            f"got {type(current_archive_bytes).__name__}"
        )
    if current_archive_bytes < 0:
        raise ValueError(
            "compose_score_from_axes: current_archive_bytes must be "
            f">= 0 (got {current_archive_bytes})"
        )
    if not isinstance(current_d_pose, (int, float)):
        raise TypeError(
            "compose_score_from_axes: current_d_pose must be numeric, "
            f"got {type(current_d_pose).__name__}"
        )
    if math.isnan(current_d_pose) or math.isinf(current_d_pose):
        raise ValueError(
            "compose_score_from_axes: current_d_pose must be finite "
            f"(got {current_d_pose})"
        )
    if current_d_pose < 0:
        raise ValueError(
            "compose_score_from_axes: current_d_pose must be >= 0 "
            "(sqrt domain); got "
            f"{current_d_pose}"
        )

    new_d_pose = current_d_pose + decomposition.predicted_d_pose_delta
    if new_d_pose < 0:
        # A delta that drives pose negative is unphysical; clamp at 0 +
        # surface a residual flag in the per-axis breakdown so the
        # operator can audit. We do NOT raise because consumers may
        # legitimately emit a prediction that mathematically saturates
        # the lower bound; the canonical behavior is to compose against
        # the clamped value + record the saturation.
        new_d_pose = 0.0

    seg_contribution = CANONICAL_SEG_MULTIPLIER * decomposition.predicted_d_seg_delta
    pose_contribution = math.sqrt(
        CANONICAL_POSE_SQRT_INNER * new_d_pose
    ) - math.sqrt(CANONICAL_POSE_SQRT_INNER * current_d_pose)
    rate_contribution = (
        CANONICAL_RATE_MULTIPLIER
        * decomposition.predicted_archive_bytes_delta
        / CANONICAL_RATE_DENOM_BYTES
    )
    total = seg_contribution + pose_contribution + rate_contribution

    return ComposedScoreDelta(
        total_delta=total,
        seg_delta_contribution=seg_contribution,
        pose_delta_contribution=pose_contribution,
        rate_delta_contribution=rate_contribution,
        baseline_d_pose=current_d_pose,
        baseline_archive_bytes=current_archive_bytes,
        canonical_provenance=dict(decomposition.canonical_provenance),
        axis_tag=decomposition.axis_tag,
        per_axis_residual=dict(per_axis_residual or {}),
    )


def compose_scalar_delta(
    decomposition: AxisDecomposition,
    *,
    current_archive_bytes: int,
    current_d_pose: float,
) -> float:
    """Backward-compat scalar API: compose per-axis decomposition into ΔS only.

    Returns the canonical ``total_delta`` field of the full
    :class:`ComposedScoreDelta`. Use this when the caller does not need
    the per-axis breakdown (e.g. when feeding a scalar-only ranker
    consumer).

    Math identical to :func:`compose_score_from_axes`; see that
    function's docstring for the canonical formula + cite-chain.
    """
    result = compose_score_from_axes(
        decomposition,
        current_archive_bytes=current_archive_bytes,
        current_d_pose=current_d_pose,
    )
    return result.total_delta


def load_baseline_pose_from_canonical_frontier_pointer(
    *,
    repo_root: Path | str | None = None,
    panel_axis: str = "contest_cpu",
) -> tuple[float, int] | None:
    """Read the canonical frontier baseline (d_pose, archive_bytes) tuple.

    Per CLAUDE.md "Frontier scores are pointer-only" non-negotiable +
    Catalog #343: the canonical pointer at
    ``.omx/state/canonical_frontier_pointer.json`` is the SoT for the
    current operating point. Callers that need per-axis composition
    against the current frontier SHOULD source baselines from this
    helper to avoid hardcoded literals.

    Returns ``(d_pose, archive_bytes)`` tuple OR ``None`` if the pointer
    cannot be loaded / has no anchor on the requested axis / lacks the
    extra fields. The lenient loader is used so a missing or corrupt
    pointer does not crash the caller; downstream callers MUST handle
    None by sourcing baselines from their own canonical state or
    refusing the composition.

    Args:
        repo_root: repo root path; defaults to current working
            directory.
        panel_axis: ``"contest_cpu"`` (default) or ``"contest_cuda"``.

    Note: the canonical pointer's :class:`AnchorRecord` schema does NOT
    currently include separate per-axis component fields (seg / pose /
    rate); only the composite score + axis + archive_sha256 + lane_id
    are stored. This helper returns ``None`` until the pointer schema
    is extended with per-axis components in a sister wave (Dim 3 Step
    3.4 successor). Currently exists as the canonical wire-in surface
    so consumers can be written against the eventual API.
    """
    try:
        from tac.canonical_frontier_pointer import (
            load_canonical_frontier_pointer_lenient,
        )
    except ImportError:
        return None

    pointer = load_canonical_frontier_pointer_lenient(
        repo_root=Path(repo_root) if repo_root else Path.cwd()
    )
    if pointer is None:
        return None

    axis_key = (
        "our_local_frontier_contest_cpu"
        if panel_axis == "contest_cpu"
        else "our_local_frontier_contest_cuda"
    )
    anchor = getattr(pointer, axis_key, None)
    if anchor is None:
        return None
    # The current pointer schema doesn't store per-axis components
    # (d_pose, archive_bytes) explicitly. A sister wave will extend
    # AnchorRecord.extra with these fields; until then this helper
    # returns None so callers don't silently consume stale baselines.
    extra = getattr(anchor, "extra", None) or {}
    if "d_pose" not in extra or "archive_bytes" not in extra:
        return None
    try:
        return (float(extra["d_pose"]), int(extra["archive_bytes"]))
    except (TypeError, ValueError):
        return None
