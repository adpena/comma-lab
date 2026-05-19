# SPDX-License-Identifier: MIT
"""Multiple-passes per-byte deterministic correction framework.

Per operator standing directive 2026-05-19 verbatim *"we can do multiple
passes and per byte deterministic corrections"* and *"proceed with all
iteration opportunities and recommendations and followup"*, this module
operationalizes the iterative-refinement methodology: each pass measures the
empirical score response to a deterministic per-byte perturbation, then
updates the master-gradient model from prediction-vs-measurement residuals.

The methodology is Bayesian. The master-gradient anchor (per
`tac.master_gradient.MasterGradient`) is the prior; each
IterativeRefinementPass record is a measurement; after N passes the per-byte
sensitivity estimates converge to the true score-response surface (modulo
brotli-cascade nonlinearity captured in the sister grain at
`tac.master_gradient_post_brotli_decompress`).

The "DETERMINISTIC" qualifier is critical: each pass picks the top-K bytes by
master-gradient |sensitivity|, applies a fixed signed delta, and measures.
The selection is REPEATABLE — running pass N twice produces the same byte
list + the same perturbations + the same measurement (modulo runtime noise).
Compare to stochastic sweeps (random byte sampling) which would require many
more empirical anchors to converge.

Per CLAUDE.md "Meta-Lagrangian/Pareto solver" non-negotiable: every refined
master-gradient anchor flows back into the canonical posterior at
`.omx/state/master_gradient_anchors.jsonl` per Catalog #128/#131/#245 sister
discipline. The cathedral autopilot ranker can then consume the refined
anchor without manual re-wire (Catalog #335/#336/#337 paradigm shift).

Per Catalog #287/#323: every numerical field in IterativeRefinementPass
carries canonical Provenance with kind=PREDICTED_FROM_MODEL (for the
predicted Δscore) OR kind=EMPIRICAL anchor (for the measured Δscore). The
prediction-vs-measurement residual is the operator-routable disambiguator
between "the model is correct" and "the model needs refinement".

Sister modules:
- `tac.master_gradient_post_brotli_decompress`: the corrected grain that the
  iteration framework consumes (raw-archive-byte grain is wrong; post-brotli
  decompressed-decoder-weight-byte grain is correct).
- `tac.master_gradient`: canonical MasterGradient dataclass + ledger writer.
- `tac.master_gradient_pr101_mps_axis_probe`: cross-device probe that
  validates per-byte response stability across MPS/CUDA/CPU.
- `tac.mps_diagnostic.drift_predictor`: slot 9 Cauchy-Schwarz bound on MPS
  drift that feeds the pass-1 prediction.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None  # type: ignore[assignment]


__all__ = [
    "DEFAULT_LEARNING_RATE",
    "DEFAULT_TOP_K",
    "IterativeRefinementError",
    "IterativeRefinementPass",
    "compute_prediction_vs_measurement_residuals",
    "deterministic_top_k_byte_selection",
    "summarize_pass_for_next_recommendation",
]

# Defaults per operator directive: top-10 deterministic candidate, +1 perturbation
DEFAULT_TOP_K: int = 10
DEFAULT_PERTURBATION_DELTA: int = 1
DEFAULT_LEARNING_RATE: float = 0.1


class IterativeRefinementError(RuntimeError):
    """Raised when a refinement pass argument or computation is malformed."""


@dataclass(frozen=True)
class IterativeRefinementPass:
    """Record of one deterministic per-byte correction pass.

    Schema is APPEND-ONLY per CLAUDE.md HISTORICAL_PROVENANCE Catalog
    #110/#113 sister discipline; corrections land as NEW pass records with a
    later `pass_index`, never as in-place mutations of an existing pass.

    Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #287/#323:
    `measurement_axis` MUST be lane-tagged (e.g. `[contest-CPU]`,
    `[contest-CUDA]`, `[MPS-research-signal]`, `[macOS-CPU advisory]`,
    `[predicted]`). The two-axis fields enforce the producer/consumer
    separation: `measured_score_deltas` is empirical evidence;
    `predicted_score_deltas` is what the prior model predicted; the residual
    is the structural update signal for pass N+1.
    """

    pass_index: int
    master_gradient_anchor_archive_sha256: str
    master_gradient_anchor_path: str
    mutation_grain: str
    perturbed_byte_indices: tuple[int, ...]
    perturbation_deltas: tuple[int, ...]
    measured_score_deltas: Mapping[str, float]
    predicted_score_deltas: Mapping[str, float]
    prediction_vs_measurement_residual: Mapping[str, float]
    measurement_axis: str
    measurement_hardware: str
    measurement_utc: str
    next_pass_recommendations: tuple[str, ...] = field(default_factory=tuple)
    notes: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.pass_index, int) or self.pass_index < 1:
            raise IterativeRefinementError(
                f"pass_index must be a positive int; got {self.pass_index!r}"
            )
        if not isinstance(self.master_gradient_anchor_archive_sha256, str) or len(
            self.master_gradient_anchor_archive_sha256
        ) < 16:
            raise IterativeRefinementError(
                "master_gradient_anchor_archive_sha256 must be a hex sha256 >=16 chars"
            )
        if not self.master_gradient_anchor_path:
            raise IterativeRefinementError(
                "master_gradient_anchor_path must be non-empty"
            )
        if not self.mutation_grain:
            raise IterativeRefinementError("mutation_grain must be non-empty")
        if len(self.perturbed_byte_indices) != len(self.perturbation_deltas):
            raise IterativeRefinementError(
                "perturbed_byte_indices and perturbation_deltas must have equal length"
            )
        if len(self.perturbed_byte_indices) == 0:
            raise IterativeRefinementError(
                "perturbed_byte_indices must be non-empty (at least 1 byte per pass)"
            )
        # Deterministic invariant: byte indices must be sorted ascending and
        # de-duplicated. A pass that contains the same byte twice cannot be
        # decomposed into atomic per-byte contributions cleanly.
        for i in range(1, len(self.perturbed_byte_indices)):
            if self.perturbed_byte_indices[i] <= self.perturbed_byte_indices[i - 1]:
                raise IterativeRefinementError(
                    "perturbed_byte_indices must be strictly ascending; "
                    f"got duplicate or out-of-order at index {i}: "
                    f"{self.perturbed_byte_indices[i - 1]} -> "
                    f"{self.perturbed_byte_indices[i]}"
                )
        if any(idx < 0 for idx in self.perturbed_byte_indices):
            raise IterativeRefinementError(
                "perturbed_byte_indices must be non-negative"
            )
        if not self.measurement_axis.startswith("["):
            raise IterativeRefinementError(
                f"measurement_axis {self.measurement_axis!r} must be lane-tagged "
                "per CLAUDE.md 'Apples-to-apples evidence discipline' "
                "(e.g. '[contest-CPU]', '[contest-CUDA]', '[MPS-research-signal]', "
                "'[macOS-CPU advisory]', '[predicted]')"
            )
        for label, mapping in (
            ("measured_score_deltas", self.measured_score_deltas),
            ("predicted_score_deltas", self.predicted_score_deltas),
            (
                "prediction_vs_measurement_residual",
                self.prediction_vs_measurement_residual,
            ),
        ):
            if not isinstance(mapping, Mapping):
                raise IterativeRefinementError(
                    f"{label} must be a Mapping; got {type(mapping).__name__}"
                )
            for k, v in mapping.items():
                if not isinstance(k, str) or not k:
                    raise IterativeRefinementError(
                        f"{label} keys must be non-empty strings"
                    )
                if not isinstance(v, (int, float)) or isinstance(v, bool):
                    raise IterativeRefinementError(
                        f"{label}[{k!r}] must be a number; got {type(v).__name__}"
                    )

    def as_dict(self) -> dict[str, Any]:
        return {
            "pass_index": self.pass_index,
            "master_gradient_anchor_archive_sha256": (
                self.master_gradient_anchor_archive_sha256
            ),
            "master_gradient_anchor_path": self.master_gradient_anchor_path,
            "mutation_grain": self.mutation_grain,
            "perturbed_byte_indices": list(self.perturbed_byte_indices),
            "perturbation_deltas": list(self.perturbation_deltas),
            "measured_score_deltas": dict(self.measured_score_deltas),
            "predicted_score_deltas": dict(self.predicted_score_deltas),
            "prediction_vs_measurement_residual": dict(
                self.prediction_vs_measurement_residual
            ),
            "measurement_axis": self.measurement_axis,
            "measurement_hardware": self.measurement_hardware,
            "measurement_utc": self.measurement_utc,
            "next_pass_recommendations": list(self.next_pass_recommendations),
            "notes": self.notes,
        }


def deterministic_top_k_byte_selection(
    sensitivity_array,
    *,
    top_k: int = DEFAULT_TOP_K,
    rank_by: str = "combined_seg_pose_abs",
) -> tuple[int, ...]:
    """Pick top-K byte indices DETERMINISTICALLY from a sensitivity tensor.

    Per the operator's "multiple passes and per byte deterministic
    corrections" methodology, the selection MUST be repeatable: re-running
    the function with the same tensor + the same rank_by yields the same K
    indices in the same order.

    Args:
        sensitivity_array: ndarray (N_bytes, 3) [seg, pose, rate].
        top_k: how many byte indices to return.
        rank_by: ranking criterion. One of:
            - "combined_seg_pose_abs" (default): |seg| + |pose|
            - "seg_abs": |seg| only
            - "pose_abs": |pose| only
            - "absolute_total": |seg| + |pose| + |rate|

    Returns:
        Tuple of top-K byte indices SORTED ASCENDING (deterministic per
        IterativeRefinementPass invariant; tie-break by lower index).
    """
    if np is None:
        raise IterativeRefinementError(
            "numpy required for deterministic_top_k_byte_selection"
        )
    if top_k < 1:
        raise IterativeRefinementError(f"top_k must be >=1; got {top_k}")
    arr = np.asarray(sensitivity_array)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise IterativeRefinementError(
            f"sensitivity tensor shape {arr.shape} != (N, 3)"
        )
    if arr.shape[0] == 0:
        return ()
    if rank_by == "combined_seg_pose_abs":
        scores = np.abs(arr[:, 0]) + np.abs(arr[:, 1])
    elif rank_by == "seg_abs":
        scores = np.abs(arr[:, 0])
    elif rank_by == "pose_abs":
        scores = np.abs(arr[:, 1])
    elif rank_by == "absolute_total":
        scores = np.abs(arr).sum(axis=1)
    else:
        raise IterativeRefinementError(
            f"rank_by={rank_by!r} unrecognized; must be one of "
            "{'combined_seg_pose_abs', 'seg_abs', 'pose_abs', 'absolute_total'}"
        )
    # argsort is deterministic; descending order via [::-1]; tie-break is
    # stable lower-index-first via np.argsort(kind='stable') applied to -scores
    order = np.argsort(-scores, kind="stable")
    k = min(top_k, arr.shape[0])
    top = sorted(int(i) for i in order[:k])  # ascending per dataclass invariant
    return tuple(top)


def compute_prediction_vs_measurement_residuals(
    measured: Mapping[str, float], predicted: Mapping[str, float]
) -> dict[str, float]:
    """Compute per-axis residuals (measured - predicted) for a pass.

    The residual is the structural signal for refining the master-gradient
    model for the next pass: if predicted=+0.001 but measured=+0.005, the
    model UNDERESTIMATED by 0.004 on that axis; pass N+1 should scale the
    sensitivity tensor coefficients to better match the empirical surface.

    Args:
        measured: per-axis empirical Δscore (e.g.
            {"contest_cpu": 0.0016937, "contest_cuda": 0.0013627}).
        predicted: per-axis predicted Δscore from prior model.

    Returns:
        Per-axis residual dict. Axes present in only one mapping are emitted
        with the missing-side as 0.0 (so the residual equals the present
        side; this is a signal that the model failed to predict that axis or
        the measurement is incomplete).
    """
    out: dict[str, float] = {}
    axes = set(measured.keys()) | set(predicted.keys())
    for axis in sorted(axes):
        m = float(measured.get(axis, 0.0))
        p = float(predicted.get(axis, 0.0))
        out[axis] = m - p
    return out


def summarize_pass_for_next_recommendation(
    pass_record: IterativeRefinementPass,
    *,
    large_residual_threshold: float = 0.001,
) -> list[str]:
    """Operator-routable recommendations for the next pass from this pass.

    Examines the prediction-vs-measurement residual and emits canonical
    operator action strings. Sister of the operator briefing surface.

    Args:
        pass_record: a completed IterativeRefinementPass.
        large_residual_threshold: residuals above this magnitude trigger
            model-refinement recommendations.

    Returns:
        List of operator-routable string recommendations.
    """
    recs: list[str] = []
    residuals = pass_record.prediction_vs_measurement_residual
    if not residuals:
        return [
            "no residuals computed; pass record is incomplete or "
            "measurement_axis lacked any axis"
        ]
    max_abs_axis = max(residuals, key=lambda k: abs(residuals[k]))
    max_abs = abs(residuals[max_abs_axis])
    if max_abs > large_residual_threshold:
        recs.append(
            f"Residual on '{max_abs_axis}' axis is {residuals[max_abs_axis]:+.6f} "
            f"(|{max_abs:.6f}| > threshold {large_residual_threshold}); "
            "refine master-gradient anchor before pass "
            f"{pass_record.pass_index + 1}"
        )
        # Sign-aware: positive residual = model under-estimated worsening;
        # negative residual = model over-estimated worsening
        for axis, r in residuals.items():
            if abs(r) > large_residual_threshold:
                direction = "UNDER-ESTIMATED" if r > 0 else "OVER-ESTIMATED"
                recs.append(
                    f"  axis '{axis}': model {direction} the empirical "
                    f"Δscore by {abs(r):.6f}"
                )
    else:
        recs.append(
            f"All residuals within ±{large_residual_threshold:.6f}; "
            "master-gradient model VALIDATED at this pass's perturbation "
            "intensity. Proceed to next pass with finer-resolution byte "
            "selection or larger top_k."
        )
    # Mutation-grain-specific guidance per the post-brotli vs raw-byte
    # distinction (item 3 of codex op7 iteration opportunities)
    if "post_brotli" in pass_record.mutation_grain:
        recs.append(
            "mutation_grain is post-brotli-decompress: the byte-locality "
            "assumption is valid here; residuals are model error, not "
            "cascade nonlinearity."
        )
    else:
        recs.append(
            f"mutation_grain={pass_record.mutation_grain}: if this grain is "
            "compressed (e.g. raw-archive-byte), residuals may reflect "
            "brotli cascade nonlinearity rather than model error. Consider "
            "switching to post-brotli-decompress grain via "
            "tac.master_gradient_post_brotli_decompress."
        )
    return recs
