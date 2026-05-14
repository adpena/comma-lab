# SPDX-License-Identifier: MIT
"""Axis-level reweighting API for the sensitivity-map composition stack.

This module is the canonical home of the OPERATING-POINT-AWARE per-axis
weight rule. The need was surfaced by two independent landings:

- ``tools/build_composition_ranking_json.py`` (FIX-C bridge): had
  ``DEFAULT_AXIS_WEIGHTS`` defined inline at module level.
- ``tools/probe_seg_pose_weight_at_operating_point.py`` (GGGG A-1 probe):
  recomputed the closed-form ratio inline from
  :mod:`tac.score_geometry`.

Both call sites must consume the SAME canonical operating-point-aware
axis weights — otherwise the bridge (which feeds the autopilot ranker)
disagrees with the probe (which the operator consults). This module
unifies that surface.

The 4-axis vocabulary
─────────────────────

``pose``    — Score-axis where PoseNet distortion contributes via
              ``sqrt(POSE_INSIDE * d_pose)``. Marginal value grows as
              ``d_pose -> 0`` (1/sqrt singularity at the floor).
``seg``     — Score-axis where SegNet distortion contributes
              ``SEG_COEFFICIENT * d_seg`` linearly (constant marginal).
``rate``    — Archive-bytes axis ranked by EV/$ directly.
``mixed``   — Composite axis for substrates that touch BOTH pose and seg
              (e.g. cross-paradigm composition cells). Conservative
              midpoint between pose and seg.

The operating-point-aware closed-form ratio
───────────────────────────────────────────

From :mod:`tac.score_geometry`:

    dS/d(d_seg)  = SEG_COEFFICIENT == 100
    dS/d(d_pose) = 0.5 * sqrt(POSE_INSIDE / d_pose)  with POSE_INSIDE = 10

At the PR106 r2 frontier (``d_pose = 3.4e-5``):

    dS/d(d_pose) = 0.5 * sqrt(10 / 3.4e-5)
                 = 0.5 * sqrt(294117.6)
                 = 271.16

    pose_marginal / seg_marginal = 271.16 / 100 = 2.7116

This is the "2.71×" number cited in CLAUDE.md "SegNet vs PoseNet
importance — operating-point dependent". The flip happens at
``d_pose ≈ 2.5e-4`` (``importance_flip_threshold()``); below that, pose
marginal exceeds seg marginal.

Cross-references
────────────────
- :mod:`tac.score_geometry` — closed-form gradient (``score_gradient``,
  ``importance_flip_threshold``).
- :mod:`tac.optimization.lagrangian_per_tensor_allocation` —
  ``compute_cpu_axis_weights`` (the per-tensor Jacobian-rebalance
  precedent on the CPU-axis side; conceptually orthogonal to this
  per-AXIS rule but the naming is parallel).
- :mod:`tools.build_composition_ranking_json` — FIX-C bridge consumer.
- :mod:`tools.probe_seg_pose_weight_at_operating_point` — GGGG A-1
  probe-disambiguator consumer.

CLAUDE.md compliance tags
─────────────────────────
- ``planning_only_no_score_claim`` (axis weights are PLANNING-side EV
  multipliers; they don't make score claims).
- ``operating_point_aware_axis_weight_rule_v1``
- ``no_tmp_paths``
- ``derived_closed_form_gradient_per_src_tac_score_geometry``
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from tac.score_geometry import (
    POSE_COEFFICIENT_INSIDE_SQRT,
    SEG_COEFFICIENT,
    importance_flip_threshold,
    score_gradient,
)

__all__ = [
    "AXIS_NAMES",
    "OLD_1X_OPERATING_POINT_AXIS_WEIGHTS",
    "OPERATING_POINT_ANCHORS",
    "PR102_CUDA_AXIS_WEIGHTS",
    "PR106_R2_FRONTIER_AXIS_WEIGHTS",
    "PR106_R2_POSE_PER_SEG_MARGINAL_RATIO",
    "AxisWeights",
    "AxisWeightsError",
    "axis_weights_for_named_operating_point",
    "compute_axis_weights",
    "default_axis_weights",
    "validate_axis_weights_mapping",
]

#: Canonical 4-axis vocabulary consumed by the autopilot ranker. Any
#: substrate's ``target_axis`` MUST be one of these strings (the bridge
#: falls back to ``1.0`` for unknown axes, which is the safe default).
AXIS_NAMES: tuple[str, ...] = ("pose", "seg", "rate", "mixed")

#: PR106 r2 marginal ratio derived from the closed-form gradient at the
#: PR106 r2 frontier operating point (``d_pose = 3.4e-5``). Documented in
#: CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent".
PR106_R2_POSE_PER_SEG_MARGINAL_RATIO: float = 2.7116


class AxisWeightsError(ValueError):
    """Raised when an :class:`AxisWeights` value is invalid."""


@dataclass(frozen=True)
class AxisWeights:
    """Operating-point-aware per-axis EV multipliers.

    The four fields ``pose / seg / rate / mixed`` are the axes the
    autopilot ranker (and the composition-cell bridge) consumes to scale
    expected-information-gain. The ``operating_point_tag`` and ``basis``
    fields are PROVENANCE (they let downstream artifacts trace which rule
    produced these weights).

    Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag", any
    artifact that consumes an ``AxisWeights`` instance MUST propagate the
    ``basis`` and ``operating_point_tag`` into its own evidence string.
    """

    pose: float
    seg: float
    rate: float
    mixed: float
    operating_point_tag: str
    basis: str

    def __post_init__(self) -> None:
        for name in ("pose", "seg", "rate", "mixed"):
            value = float(getattr(self, name))
            if math.isnan(value) or math.isinf(value):
                raise AxisWeightsError(
                    f"axis weight {name!r} must be finite; got {value!r}"
                )
            if value < 0.0:
                raise AxisWeightsError(
                    f"axis weight {name!r} must be non-negative; got {value!r}"
                )
        if not isinstance(self.operating_point_tag, str) or not self.operating_point_tag:
            raise AxisWeightsError(
                "operating_point_tag must be a non-empty string"
            )
        if not isinstance(self.basis, str) or not self.basis:
            raise AxisWeightsError("basis must be a non-empty string")

    def as_mapping(self) -> dict[str, float]:
        """Return the four axis weights as a plain ``dict``.

        The :func:`tools.build_composition_ranking_json.build_payload`
        consumer accepts a ``dict[str, float]`` for ``axis_weights``;
        this method bridges the typed dataclass into that interface
        without leaking the provenance fields into the keyspace.
        """
        return {
            "pose": float(self.pose),
            "seg": float(self.seg),
            "rate": float(self.rate),
            "mixed": float(self.mixed),
        }

    def evidence_tag(self) -> str:
        """Return the evidence-tag string downstream artifacts MUST embed."""
        return (
            f"[axis_weights v1; operating_point={self.operating_point_tag}; "
            f"basis={self.basis}]"
        )


# ──────────────────────────────────────────────────────────────────────────
# Named operating-point anchors
# ──────────────────────────────────────────────────────────────────────────

#: PR106 r2 frontier (``d_pose = 3.4e-5``, ``d_seg = 6.7e-4``).
#: Per CLAUDE.md the closed-form ratio is ``pose / seg = 2.7116``. We use
#: ``pose = 2.71`` (3-sig-fig rounding consistent with CLAUDE.md text)
#: and ``seg = 1.00`` as the baseline. ``mixed = 1.50`` is the
#: conservative midpoint (geometric mean of 2.71 and ~0.83 lands near
#: 1.5; sensitivity bounded between seg and pose marginals).
PR106_R2_FRONTIER_AXIS_WEIGHTS = AxisWeights(
    pose=2.71,
    seg=1.00,
    rate=1.00,
    mixed=1.50,
    operating_point_tag="pr106_r2_frontier",
    basis="closed-form gradient per src/tac/score_geometry.py:253-257",
)

#: OLD 1.x scoreband (``d_pose = 0.18``, ``d_seg = 0.001``). The pose
#: marginal here is ``0.5 * sqrt(10 / 0.18) ≈ 3.7``; against the
#: ``seg = 100`` baseline that's ``ratio ≈ 0.037``. Rounded to two-sig-
#: figs gives ``pose = 0.04``, but the legacy default ratio cited
#: throughout the lab is ``0.1`` (the ``10/100`` loss-weight default
#: pair). We expose the LEGACY default here so callers that explicitly
#: opt into "old 1.x" land on the historically calibrated value, not the
#: closed-form-only rounding. The closed-form-only variant is available
#: via :func:`compute_axis_weights` if the caller passes the raw
#: ``d_pose / d_seg`` of the old regime.
OLD_1X_OPERATING_POINT_AXIS_WEIGHTS = AxisWeights(
    pose=0.10,
    seg=1.00,
    rate=1.00,
    mixed=0.55,  # midpoint of 0.10 and 1.00
    operating_point_tag="old_1x",
    basis=(
        "legacy 10/100 loss-weight default (effective ratio 0.1); "
        "closed-form ratio at d_pose=0.18 is ~0.037 — legacy retained "
        "for backward-compat per src/tac/score_geometry.py:253-257"
    ),
)

#: PR102 third-prize CUDA (``d_pose = 5e-5``, ``d_seg = 5.8e-4``).
#: Closed-form pose marginal = ``0.5 * sqrt(10 / 5e-5) ≈ 223.6``;
#: ratio against ``seg = 100`` is ``≈ 2.236``. Rounded to three sig
#: figs and exposed for parity-with-public-frontier diagnostics.
PR102_CUDA_AXIS_WEIGHTS = AxisWeights(
    pose=2.24,
    seg=1.00,
    rate=1.00,
    mixed=1.62,  # midpoint of 2.24 and 1.00
    operating_point_tag="pr102_cuda",
    basis=(
        "closed-form gradient at PR102 third-prize CUDA d_pose=5e-5, "
        "d_seg=5.8e-4 per src/tac/score_geometry.py:253-257"
    ),
)

#: Lookup table for named operating-point anchors. The keys match the
#: ``--operating-point`` choice set in
#: ``tools/probe_seg_pose_weight_at_operating_point.py``.
OPERATING_POINT_ANCHORS: Mapping[str, AxisWeights] = {
    "pr106_r2": PR106_R2_FRONTIER_AXIS_WEIGHTS,
    "old_1x": OLD_1X_OPERATING_POINT_AXIS_WEIGHTS,
    "pr102_cuda": PR102_CUDA_AXIS_WEIGHTS,
}


# ──────────────────────────────────────────────────────────────────────────
# Closed-form computation
# ──────────────────────────────────────────────────────────────────────────


def compute_axis_weights(
    *,
    d_pose: float,
    d_seg: float | None = None,
    operating_point_tag: str = "custom",
    seg_normalizer: float = 1.0,
) -> AxisWeights:
    """Compute operating-point-aware axis weights from a closed-form gradient.

    The returned ``pose`` weight equals the ratio of the pose marginal
    to the seg marginal (i.e. how much more EV a unit of pose-axis
    improvement is worth compared to seg). The ``seg`` weight is fixed
    at ``seg_normalizer`` (default 1.0) so seg is the baseline axis.
    ``rate`` defaults to ``seg_normalizer`` because rate primitives are
    ranked by EV/$ directly (no axis multiplier). ``mixed`` is the
    midpoint between ``pose`` and ``seg``.

    The closed-form gradients come from
    :func:`tac.score_geometry.score_gradient`:

        dS/d(d_seg) = SEG_COEFFICIENT == 100
        dS/d(d_pose) = 0.5 * sqrt(POSE_INSIDE / d_pose)  with POSE_INSIDE = 10

    Args
    ----
    d_pose
        Pose distortion at the current operating point. Must be > 0
        (the d_pose=0 limit is the singularity at the score floor; the
        marginal there is infinite, so the ratio is unbounded and not
        useful as a planning multiplier — callers wanting that limit
        should consult :func:`tac.score_geometry.score_gradient`
        directly).
    d_seg
        Optional seg distortion (default None means "use the closed
        form alone"). When supplied, validates non-negativity but
        otherwise does NOT enter the ratio (the SEG marginal is
        constant at ``SEG_COEFFICIENT == 100`` regardless of d_seg).
    operating_point_tag
        Provenance string (e.g. ``"pr106_r2_frontier"``,
        ``"custom_3.4e-5"``).
    seg_normalizer
        Baseline weight for the seg axis (default 1.0).
    """
    if d_pose <= 0.0:
        raise AxisWeightsError(
            f"d_pose must be > 0 for a finite operating point; got {d_pose}"
        )
    if math.isnan(d_pose) or math.isinf(d_pose):
        raise AxisWeightsError(f"d_pose must be finite; got {d_pose}")
    if d_seg is not None:
        if d_seg < 0.0:
            raise AxisWeightsError(
                f"d_seg must be non-negative; got {d_seg}"
            )
        if math.isnan(d_seg) or math.isinf(d_seg):
            raise AxisWeightsError(f"d_seg must be finite; got {d_seg}")
    if seg_normalizer < 0.0 or math.isnan(seg_normalizer) or math.isinf(seg_normalizer):
        raise AxisWeightsError(
            f"seg_normalizer must be finite and non-negative; got {seg_normalizer}"
        )

    # Use the canonical closed-form gradient. d_seg=0.0 is passed because
    # the seg marginal is CONSTANT regardless of d_seg; we use it only to
    # avoid a None branch inside score_gradient. (score_gradient itself
    # requires non-negative d_pose; we already enforced d_pose > 0.)
    grad = score_gradient(d_seg=0.0, d_pose=float(d_pose))
    pose_marginal = float(grad.d_pose)
    seg_marginal = float(grad.d_seg)
    # ratio = pose_marginal / seg_marginal. With seg_normalizer = 1.0,
    # pose_weight is just that ratio (seg-baselined EV multiplier).
    if seg_marginal <= 0.0:
        raise AxisWeightsError(
            f"seg marginal must be > 0; got {seg_marginal} (corrupt score_geometry?)"
        )
    pose_weight = seg_normalizer * (pose_marginal / seg_marginal)
    seg_weight = seg_normalizer
    rate_weight = seg_normalizer
    # mixed = midpoint between pose and seg (conservative).
    mixed_weight = 0.5 * (pose_weight + seg_weight)
    basis = (
        f"closed-form gradient at d_pose={d_pose:.6g} per "
        f"src/tac/score_geometry.py:253-257"
    )
    return AxisWeights(
        pose=pose_weight,
        seg=seg_weight,
        rate=rate_weight,
        mixed=mixed_weight,
        operating_point_tag=operating_point_tag,
        basis=basis,
    )


def axis_weights_for_named_operating_point(name: str) -> AxisWeights:
    """Return the canonical :class:`AxisWeights` for a named anchor.

    Names match :data:`OPERATING_POINT_ANCHORS` (``"pr106_r2"``,
    ``"old_1x"``, ``"pr102_cuda"``). Unknown names raise
    :exc:`AxisWeightsError`.
    """
    try:
        return OPERATING_POINT_ANCHORS[name]
    except KeyError as exc:
        valid = sorted(OPERATING_POINT_ANCHORS)
        raise AxisWeightsError(
            f"unknown operating-point anchor {name!r}; valid: {valid!r}"
        ) from exc


def default_axis_weights() -> AxisWeights:
    """Return the canonical default :class:`AxisWeights`.

    Defaults to PR106 r2 frontier per CLAUDE.md "operating-point
    dependent" rule. Downstream consumers that target older operating
    points must explicitly opt in via
    :func:`axis_weights_for_named_operating_point` or
    :func:`compute_axis_weights`.
    """
    return PR106_R2_FRONTIER_AXIS_WEIGHTS


def validate_axis_weights_mapping(weights: Mapping[str, float]) -> None:
    """Validate that a plain dict has all 4 axes and finite non-negative values.

    Raises :exc:`AxisWeightsError` on the first violation; otherwise
    returns silently. Used by CLI shims that build an ``AxisWeights``
    from operator-supplied floats.
    """
    missing = [a for a in AXIS_NAMES if a not in weights]
    if missing:
        raise AxisWeightsError(
            f"axis weights mapping missing axes: {missing!r}; required: {list(AXIS_NAMES)!r}"
        )
    for name in AXIS_NAMES:
        v = weights[name]
        try:
            value = float(v)
        except (TypeError, ValueError) as exc:
            raise AxisWeightsError(
                f"axis weight {name!r} must be coercible to float; got {v!r}"
            ) from exc
        if math.isnan(value) or math.isinf(value):
            raise AxisWeightsError(
                f"axis weight {name!r} must be finite; got {value!r}"
            )
        if value < 0.0:
            raise AxisWeightsError(
                f"axis weight {name!r} must be non-negative; got {value!r}"
            )


# Re-export key score_geometry constants so consumers that want to inspect
# the PR106 r2 closed-form derivation don't have to import score_geometry
# separately.
__all__ += ["POSE_COEFFICIENT_INSIDE_SQRT", "SEG_COEFFICIENT", "importance_flip_threshold"]
