# SPDX-License-Identifier: MIT
"""Strict finite-lattice rate-distortion continuation for direct descriptions.

The global uint8 lattice is far too large to certify by enumeration.  This
module therefore makes the optimization domain explicit: a SHA-custodied set
of receiver-measured descriptions.  It never upgrades a restricted optimum to
a global-lattice optimum.

Every counted byte belongs to exactly one side of the two-type factorization:

* ``skeleton`` — topology, grammar, framing, and control tokens;
* ``fiber`` — coefficients, exception values, and sampled solved planes.

Continuation starts from a named measured control and uses adjacent candidates
only.  A full-rank check over the finite set is retained as a certificate that
the neighbor corrector reached the restricted global minimizer.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any, Literal

EVIDENCE_AXIS = "[macOS-CPU frozen-scorer advisory]"
FACTOR_KINDS = frozenset({"skeleton", "fiber"})
CUSTODY_ROLES = frozenset({"stored_problem", "solve_exception"})
RECEIVER_CLOSURES = frozenset(
    {
        "archive_receiver_closed",
        "measurement_harness_receiver_closed",
    }
)
SEMANTIC_STRATA = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
SCORER_VISIBILITIES = ("ker(A)-invisible", "seg-visible", "pose-visible")
G4_TEMPORAL_CLASSES = ("STATIC_IN_IMAGE", "STATIC_IN_XI_PROXY", "TRANSIENT")


class LambdaContinuationError(ValueError):
    """Raised when continuation custody or finite-lattice math is invalid."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return the byte-stable JSON representation used by checkpoints."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def realized_distortion(d_seg: float, d_pose: float) -> float:
    """Return ``100*d_seg + sqrt(10*d_pose)`` with strict domain checks."""

    seg = float(d_seg)
    pose = float(d_pose)
    if not math.isfinite(seg) or not math.isfinite(pose) or seg < 0.0 or pose < 0.0:
        raise LambdaContinuationError("realized distortions must be finite and nonnegative")
    return 100.0 * seg + math.sqrt(10.0 * pose)


@dataclass(frozen=True)
class CodedStream:
    """One exact counted byte home in the two-type representation."""

    stream_id: str
    stratum: str
    factor_kind: Literal["skeleton", "fiber"]
    custody_role: Literal["stored_problem", "solve_exception"]
    counted_bytes: int
    sha256: str
    codec: str
    source_path: str

    def __post_init__(self) -> None:
        if not self.stream_id or not self.stratum or not self.codec or not self.source_path:
            raise LambdaContinuationError("coded-stream identity fields must be nonempty")
        if self.factor_kind not in FACTOR_KINDS:
            raise LambdaContinuationError("coded stream must be tagged skeleton or fiber")
        if self.custody_role not in CUSTODY_ROLES:
            raise LambdaContinuationError("coded stream must belong to stored_problem or solve_exception")
        if isinstance(self.counted_bytes, bool) or not isinstance(self.counted_bytes, int) or self.counted_bytes <= 0:
            raise LambdaContinuationError("coded-stream byte counts must be positive integers")
        if len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256):
            raise LambdaContinuationError("coded-stream SHA-256 must be lowercase hex")

    def to_dict(self) -> dict[str, Any]:
        return {
            "stream_id": self.stream_id,
            "stratum": self.stratum,
            "factor_kind": self.factor_kind,
            "custody_role": self.custody_role,
            "counted_bytes": self.counted_bytes,
            "sha256": self.sha256,
            "codec": self.codec,
            "source_path": self.source_path,
        }


@dataclass(frozen=True)
class MeasuredDescription:
    """One n600 realized point in the restricted continuation domain."""

    candidate_id: str
    counted_bytes: int
    d_seg: float
    d_pose: float
    coded_streams: tuple[CodedStream, ...]
    source_artifact: str
    source_sha256: str
    receiver_closure: Literal[
        "archive_receiver_closed",
        "measurement_harness_receiver_closed",
    ]
    pool_id: str = "solve_frontier"
    pair_count: int = 600
    evidence_axis: str = EVIDENCE_AXIS
    score_claim: bool = False
    own_stored_problem: bool = True
    donor_conditioned: bool = False
    per_class: Mapping[str, Mapping[str, float | int]] | None = None
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.candidate_id or not self.source_artifact or not self.pool_id:
            raise LambdaContinuationError("candidate identity fields must be nonempty")
        if isinstance(self.counted_bytes, bool) or not isinstance(self.counted_bytes, int) or self.counted_bytes <= 0:
            raise LambdaContinuationError("candidate counted_bytes must be positive")
        if self.pair_count != 600:
            raise LambdaContinuationError("RD1 headline points must be exact n600")
        if self.evidence_axis != EVIDENCE_AXIS or self.score_claim is not False:
            raise LambdaContinuationError("RD1 authority firewall differs")
        if self.receiver_closure not in RECEIVER_CLOSURES:
            raise LambdaContinuationError("receiver closure is not recognized")
        if not self.own_stored_problem or self.donor_conditioned:
            raise LambdaContinuationError("donor-conditioned descriptions are inadmissible")
        if self.pool_id != "solve_frontier":
            raise LambdaContinuationError("RD1 candidates must live in solve_frontier")
        if len(self.source_sha256) != 64:
            raise LambdaContinuationError("candidate source SHA-256 is malformed")
        if not self.coded_streams:
            raise LambdaContinuationError("candidate has no typed coded streams")
        stream_ids = [stream.stream_id for stream in self.coded_streams]
        if len(stream_ids) != len(set(stream_ids)):
            raise LambdaContinuationError("coded stream ids must be unique per candidate")
        if sum(stream.counted_bytes for stream in self.coded_streams) != self.counted_bytes:
            raise LambdaContinuationError("typed streams do not partition counted bytes exactly")
        realized_distortion(self.d_seg, self.d_pose)

    @property
    def distortion(self) -> float:
        return realized_distortion(self.d_seg, self.d_pose)

    @property
    def skeleton_bytes(self) -> int:
        return sum(stream.counted_bytes for stream in self.coded_streams if stream.factor_kind == "skeleton")

    @property
    def fiber_bytes(self) -> int:
        return sum(stream.counted_bytes for stream in self.coded_streams if stream.factor_kind == "fiber")

    @property
    def description_root_sha256(self) -> str:
        rows = [stream.to_dict() for stream in self.coded_streams]
        return sha256_bytes(canonical_json_bytes(rows))

    def objective(self, lambda_value: float) -> float:
        lam = float(lambda_value)
        if not math.isfinite(lam) or lam < 0.0:
            raise LambdaContinuationError("lambda must be finite and nonnegative")
        return self.counted_bytes + lam * self.distortion

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "counted_bytes": self.counted_bytes,
            "d_seg": self.d_seg,
            "d_pose": self.d_pose,
            "D_realized": self.distortion,
            "S_composed_distortion_only": self.distortion,
            "skeleton_bytes": self.skeleton_bytes,
            "fiber_bytes": self.fiber_bytes,
            "coded_streams": [stream.to_dict() for stream in self.coded_streams],
            "description_root_sha256": self.description_root_sha256,
            "source_artifact": self.source_artifact,
            "source_sha256": self.source_sha256,
            "receiver_closure": self.receiver_closure,
            "pool_id": self.pool_id,
            "competes_with_pool": "solver_member_selection",
            "pool_combination": "competitive_never_additive",
            "pair_count": self.pair_count,
            "evidence_axis": self.evidence_axis,
            "score_claim": self.score_claim,
            "own_stored_problem": self.own_stored_problem,
            "donor_conditioned": self.donor_conditioned,
            "per_class": None if self.per_class is None else dict(self.per_class),
            "metadata": None if self.metadata is None else dict(self.metadata),
        }


def _unique_candidates(
    candidates: Iterable[MeasuredDescription],
) -> tuple[MeasuredDescription, ...]:
    rows = tuple(candidates)
    by_id = {row.candidate_id: row for row in rows}
    if not rows or len(rows) != len(by_id):
        raise LambdaContinuationError("candidate ids must be nonempty and unique")
    return rows


def pareto_nondominated(
    candidates: Iterable[MeasuredDescription],
) -> tuple[MeasuredDescription, ...]:
    """Return byte-ascending points with strictly improving realized distortion."""

    rows = _unique_candidates(candidates)
    best_for_rate: dict[int, MeasuredDescription] = {}
    for row in rows:
        incumbent = best_for_rate.get(row.counted_bytes)
        if incumbent is None or (row.distortion, row.candidate_id) < (
            incumbent.distortion,
            incumbent.candidate_id,
        ):
            best_for_rate[row.counted_bytes] = row
    best_distortion = math.inf
    result: list[MeasuredDescription] = []
    for row in sorted(best_for_rate.values(), key=lambda item: item.counted_bytes):
        if row.distortion < best_distortion:
            result.append(row)
            best_distortion = row.distortion
    return tuple(result)


def lower_supported_hull(
    candidates: Iterable[MeasuredDescription],
) -> tuple[MeasuredDescription, ...]:
    """Return the supported lower convex hull for ``R + lambda*D``."""

    nondominated = pareto_nondominated(candidates)
    hull: list[MeasuredDescription] = []
    for row in nondominated:
        while len(hull) >= 2:
            left, middle = hull[-2:]
            slope_left = (middle.distortion - left.distortion) / (middle.counted_bytes - left.counted_bytes)
            slope_right = (row.distortion - middle.distortion) / (row.counted_bytes - middle.counted_bytes)
            if slope_right <= slope_left:
                hull.pop()
            else:
                break
        hull.append(row)
    return tuple(hull)


def crossover_lambda(left: MeasuredDescription, right: MeasuredDescription) -> float:
    """Return the positive multiplier at which adjacent hull objectives tie."""

    delta_rate = right.counted_bytes - left.counted_bytes
    delta_distortion = left.distortion - right.distortion
    if delta_rate <= 0 or delta_distortion <= 0.0:
        raise LambdaContinuationError("crossover requires more bytes and less distortion")
    return delta_rate / delta_distortion


def geometric_curvature_ladder(
    hull: Sequence[MeasuredDescription],
    *,
    minimum_points: int = 10,
    maximum_points: int = 12,
) -> tuple[float, ...]:
    """Build a geometric ladder with explicit refinement around each crossover."""

    if len(hull) < 2:
        raise LambdaContinuationError("lambda ladder requires at least two hull states")
    if not (8 <= minimum_points <= maximum_points <= 12):
        raise LambdaContinuationError("RD1 ladder bounds must remain within 8..12")
    thresholds = [crossover_lambda(a, b) for a, b in pairwise(hull)]
    if any(b <= a for a, b in pairwise(thresholds)):
        raise LambdaContinuationError("hull crossover multipliers are not strictly increasing")
    values = {0.0, 4.0 * thresholds[-1]}
    for threshold in thresholds:
        values.add(0.9 * threshold)
        values.add(1.1 * threshold)
    for left, right in pairwise(thresholds):
        values.add(math.sqrt(left * right))
    if len(values) < minimum_points:
        low = max(thresholds[0] / 4.0, 1e-12)
        high = thresholds[-1] * 4.0
        fill_count = minimum_points
        for index in range(fill_count):
            fraction = index / max(1, fill_count - 1)
            values.add(math.exp(math.log(low) * (1.0 - fraction) + math.log(high) * fraction))
    ordered = sorted(values)
    if len(ordered) > maximum_points:
        mandatory = {
            0.0,
            4.0 * thresholds[-1],
            *(0.9 * threshold for threshold in thresholds),
            *(1.1 * threshold for threshold in thresholds),
        }
        optional = [value for value in ordered if value not in mandatory]
        while len(mandatory) < maximum_points and optional:
            mandatory.add(optional.pop(len(optional) // 2))
        ordered = sorted(mandatory)
    if len(ordered) < minimum_points:
        raise LambdaContinuationError("could not construct the requested lambda ladder")
    return tuple(ordered)


def _rank(
    candidates: Sequence[MeasuredDescription],
    lambda_value: float,
) -> tuple[MeasuredDescription, ...]:
    return tuple(
        sorted(
            candidates,
            key=lambda row: (
                row.objective(lambda_value),
                row.distortion,
                row.counted_bytes,
                row.candidate_id,
            ),
        )
    )


def _neighbor_descent(
    ordered: Sequence[MeasuredDescription],
    *,
    start_id: str,
    lambda_value: float,
) -> tuple[MeasuredDescription, tuple[str, ...]]:
    by_id = {row.candidate_id: index for index, row in enumerate(ordered)}
    if start_id not in by_id:
        raise LambdaContinuationError(f"warm-start candidate is absent: {start_id}")
    index = by_id[start_id]
    path = [ordered[index].candidate_id]
    while True:
        choices = [index]
        if index > 0:
            choices.append(index - 1)
        if index + 1 < len(ordered):
            choices.append(index + 1)
        best = min(
            choices,
            key=lambda i: (
                ordered[i].objective(lambda_value),
                ordered[i].distortion,
                ordered[i].counted_bytes,
                ordered[i].candidate_id,
            ),
        )
        if best == index:
            break
        index = best
        path.append(ordered[index].candidate_id)
    return ordered[index], tuple(path)


def continuation_rows(
    candidates: Iterable[MeasuredDescription],
    lambdas: Sequence[float],
    *,
    seed_candidate_id: str,
) -> tuple[dict[str, Any], ...]:
    """Solve each finite-lattice multiplier by neighbor-only correction.

    The first correction walks the complete Pareto chain from the named
    describe-line control.  Subsequent corrections walk the supported hull.
    Every row is checked against a full rank of the entire measured domain.
    """

    rows = _unique_candidates(candidates)
    lambda_values = tuple(float(value) for value in lambdas)
    if (
        not lambda_values
        or any(not math.isfinite(value) or value < 0.0 for value in lambda_values)
        or any(b <= a for a, b in pairwise(lambda_values))
    ):
        raise LambdaContinuationError("lambdas must be finite, nonnegative, and increasing")
    pareto = pareto_nondominated(rows)
    hull = lower_supported_hull(rows)
    result: list[dict[str, Any]] = []
    previous_id = seed_candidate_id
    for index, lambda_value in enumerate(lambda_values):
        search_surface = pareto if index == 0 else hull
        selected, path = _neighbor_descent(
            search_surface,
            start_id=previous_id,
            lambda_value=lambda_value,
        )
        ranked = _rank(rows, lambda_value)
        if selected.candidate_id != ranked[0].candidate_id:
            raise LambdaContinuationError("neighbor corrector failed the full-rank restricted-global check")
        result.append(
            {
                "lambda_index": index,
                "lambda": lambda_value,
                "warm_start_from": previous_id,
                "corrector_path": list(path),
                "neighbor_only": True,
                "selected_candidate_id": selected.candidate_id,
                "counted_bytes": selected.counted_bytes,
                "d_seg": selected.d_seg,
                "d_pose": selected.d_pose,
                "D_realized": selected.distortion,
                "S_composed_distortion_only": selected.distortion,
                "objective_bytes_plus_lambda_D": selected.objective(lambda_value),
                "skeleton_bytes": selected.skeleton_bytes,
                "fiber_bytes": selected.fiber_bytes,
                "description_root_sha256": selected.description_root_sha256,
                "receiver_closure": selected.receiver_closure,
                "pair_count": selected.pair_count,
                "full_rank_candidate_ids": [row.candidate_id for row in ranked],
                "restricted_global_rank_verified": True,
                "optimization_domain": "sha_custodied_measured_n600_descriptions_only",
                "global_uint8_lattice_optimality_claim": False,
                "evidence_axis": EVIDENCE_AXIS,
                "score_claim": False,
            }
        )
        previous_id = selected.candidate_id
    return tuple(result)


def discrete_dual_rows(
    hull: Sequence[MeasuredDescription],
) -> tuple[dict[str, Any], ...]:
    """Return exact adjacent secants and exposed class/score deltas."""

    rows: list[dict[str, Any]] = []
    for index, (left, right) in enumerate(pairwise(hull), start=1):
        lambda_star = crossover_lambda(left, right)
        delta_bytes = right.counted_bytes - left.counted_bytes
        delta_seg = right.d_seg - left.d_seg
        delta_pose = right.d_pose - left.d_pose
        row: dict[str, Any] = {
            "dual_index": index,
            "left_candidate_id": left.candidate_id,
            "right_candidate_id": right.candidate_id,
            "constraint_group": (
                str((right.metadata or {}).get("mechanism_bucket"))
                if (right.metadata or {}).get("mechanism_bucket")
                else "global_description"
            ),
            "score_dimension": "joint_realized_distortion",
            "lambda_bytes_per_D": lambda_star,
            "delta_counted_bytes": delta_bytes,
            "delta_d_seg": delta_seg,
            "delta_d_pose": delta_pose,
            "delta_D_realized": right.distortion - left.distortion,
            "marginal_D_reduction_per_byte": ((left.distortion - right.distortion) / delta_bytes),
            "per_class_d_seg": [],
            "epistemic_status": "DERIVED_FROM_TWO_MEASURED_N600_ENDPOINTS",
            "evidence_axis": EVIDENCE_AXIS,
            "score_claim": False,
        }
        if left.per_class is not None and right.per_class is not None:
            common = sorted(set(left.per_class) & set(right.per_class))
            for class_name in common:
                left_class = left.per_class[class_name]
                right_class = right.per_class[class_name]
                if "d_seg" not in left_class or "d_seg" not in right_class:
                    continue
                class_delta = float(right_class["d_seg"]) - float(left_class["d_seg"])
                row["per_class_d_seg"].append(
                    {
                        "class_name": class_name,
                        "stratum": "semantic_frame1",
                        "score_dimension": "d_seg",
                        "delta_d_seg": class_delta,
                        "delta_d_seg_per_byte": class_delta / delta_bytes,
                        "binding_direction": (
                            "improves" if class_delta < 0.0 else "worsens" if class_delta > 0.0 else "flat"
                        ),
                    }
                )
        rows.append(row)
    return tuple(rows)


def effective_quantum_D(
    *,
    uint8_step: float,
    scorer_sensitivity_D_per_uint8_step: float,
) -> float:
    """Return the distortion quantum for one typed receiver dimension.

    The units are explicit so a serialized-coordinate step cannot silently
    substitute for a realized uint8 receiver step.
    """

    step = float(uint8_step)
    sensitivity = float(scorer_sensitivity_D_per_uint8_step)
    if (
        not math.isfinite(step)
        or not math.isfinite(sensitivity)
        or step <= 0.0
        or sensitivity < 0.0
    ):
        raise LambdaContinuationError("effective-quantum inputs must be finite with step > 0")
    return step * sensitivity


def typed_dimension_dual_report(
    hull: Sequence[MeasuredDescription],
) -> dict[str, Any]:
    """Type every adjacent dual by stratum, scorer visibility, and G4 class.

    Existing endpoints identify aggregate and (when error counts are present)
    per-stratum distortion contributions.  They do not identify how a shared
    byte increment or candidate delta partitions across G4 temporal classes.
    The returned cube therefore keeps those cells explicit and blocked.  The
    pooled secant survives only as a non-additive diagnostic control.
    """

    if len(hull) < 2:
        raise LambdaContinuationError("typed dimension duals require two or more hull states")
    component_evidence: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []
    edge_summaries: list[dict[str, Any]] = []
    for edge_index, (left, right) in enumerate(pairwise(hull), start=1):
        delta_bytes = right.counted_bytes - left.counted_bytes
        if delta_bytes <= 0:
            raise LambdaContinuationError("typed dual edges require increasing counted bytes")
        edge_components: list[dict[str, Any]] = []
        if left.per_class is not None and right.per_class is not None:
            if set(left.per_class) != set(SEMANTIC_STRATA) or set(right.per_class) != set(
                SEMANTIC_STRATA
            ):
                raise LambdaContinuationError("per-class dual custody must cover the canonical five strata")
            left_sites = {name: int(left.per_class[name]["sites"]) for name in SEMANTIC_STRATA}
            right_sites = {name: int(right.per_class[name]["sites"]) for name in SEMANTIC_STRATA}
            if left_sites != right_sites or any(value <= 0 for value in left_sites.values()):
                raise LambdaContinuationError("per-class site custody differs across a dual edge")
            total_sites = sum(left_sites.values())
            for stratum in SEMANTIC_STRATA:
                left_errors = int(left.per_class[stratum]["errors"])
                right_errors = int(right.per_class[stratum]["errors"])
                delta_errors = right_errors - left_errors
                delta_global_d_seg = delta_errors / total_sites
                edge_components.append(
                    {
                        "stratum": stratum,
                        "scorer_visibility": "seg-visible",
                        "g4_temporal_class": "UNRESOLVED_G4_MIXTURE",
                        "delta_errors": delta_errors,
                        "sites_in_stratum": left_sites[stratum],
                        "global_seg_sites": total_sites,
                        "delta_d_seg_global_contribution": delta_global_d_seg,
                        "delta_D_dimension": 100.0 * delta_global_d_seg,
                        "epistemic_status": "MEASURED_ENDPOINT_ERRORS_DERIVED_GLOBAL_CONTRIBUTION",
                    }
                )
        else:
            edge_components.append(
                {
                    "stratum": "ALL_SEMANTIC_STRATA_UNRESOLVED",
                    "scorer_visibility": "seg-visible",
                    "g4_temporal_class": "UNRESOLVED_G4_MIXTURE",
                    "delta_errors": None,
                    "sites_in_stratum": None,
                    "global_seg_sites": None,
                    "delta_d_seg_global_contribution": right.d_seg - left.d_seg,
                    "delta_D_dimension": 100.0 * (right.d_seg - left.d_seg),
                    "epistemic_status": "MEASURED_AGGREGATE_SEG_ENDPOINT_STRATUM_SPLIT_ABSENT",
                }
            )
        pose_delta_D = math.sqrt(10.0 * right.d_pose) - math.sqrt(10.0 * left.d_pose)
        edge_components.append(
            {
                "stratum": "POSE6_GLOBAL",
                "scorer_visibility": "pose-visible",
                "g4_temporal_class": "UNRESOLVED_G4_MIXTURE",
                "delta_d_pose": right.d_pose - left.d_pose,
                "delta_D_dimension": pose_delta_D,
                "epistemic_status": "DERIVED_FROM_TWO_MEASURED_POSE_ENDPOINTS",
            }
        )
        for component in edge_components:
            delta_dimension = float(component["delta_D_dimension"])
            component["dual_index"] = edge_index
            component["left_candidate_id"] = left.candidate_id
            component["right_candidate_id"] = right.candidate_id
            component["shared_edge_delta_counted_bytes"] = delta_bytes
            component["shared_cost_nonadditive"] = True
            component["conditional_lambda_bytes_per_D_dimension"] = (
                delta_bytes / -delta_dimension if delta_dimension < 0.0 else None
            )
            component["binding_direction"] = (
                "improves"
                if delta_dimension < 0.0
                else "worsens"
                if delta_dimension > 0.0
                else "flat"
            )
            component["actionable_for_train_decision"] = False
            component["blocker"] = "G4_TEMPORAL_CLASS_AND_SHARED_BYTE_HOME_UNRESOLVED"
            component["score_claim"] = False
            component_evidence.append(component)
        component_sum = sum(float(row["delta_D_dimension"]) for row in edge_components)
        aggregate_delta = right.distortion - left.distortion
        if not math.isclose(component_sum, aggregate_delta, rel_tol=0.0, abs_tol=1e-12):
            raise LambdaContinuationError("dimension contributions do not reconstruct aggregate delta D")
        for stratum in (*SEMANTIC_STRATA, "POSE6_GLOBAL"):
            for scorer_visibility in SCORER_VISIBILITIES:
                for temporal_class in G4_TEMPORAL_CLASSES:
                    if scorer_visibility == "ker(A)-invisible":
                        status = "STRUCTURAL_ZERO_SCORER_EFFECT_NO_TYPED_RATE_HOME"
                        delta_D: float | None = 0.0
                    elif stratum == "POSE6_GLOBAL" and scorer_visibility == "pose-visible":
                        status = "BLOCKED_G4_TO_POSE6_JOINT_ASSIGNMENT_ABSENT"
                        delta_D = None
                    elif stratum in SEMANTIC_STRATA and scorer_visibility == "seg-visible":
                        status = "BLOCKED_CANDIDATE_DELTA_TO_G4_JOINT_ASSIGNMENT_ABSENT"
                        delta_D = None
                    else:
                        status = "NOT_APPLICABLE_TYPED_VISIBILITY_COMBINATION"
                        delta_D = None
                    bucket_rows.append(
                        {
                            "dual_index": edge_index,
                            "left_candidate_id": left.candidate_id,
                            "right_candidate_id": right.candidate_id,
                            "stratum": stratum,
                            "scorer_visibility": scorer_visibility,
                            "g4_temporal_class": temporal_class,
                            "delta_D_dimension": delta_D,
                            "delta_counted_bytes_dimension": None,
                            "lambda_bytes_per_D_dimension": None,
                            "effective_quantum_D": (
                                0.0 if scorer_visibility == "ker(A)-invisible" else None
                            ),
                            "status": status,
                            "actionable_for_train_decision": False,
                            "score_claim": False,
                        }
                    )
        edge_summaries.append(
            {
                "dual_index": edge_index,
                "left_candidate_id": left.candidate_id,
                "right_candidate_id": right.candidate_id,
                "aggregate_delta_D": aggregate_delta,
                "component_delta_D_sum": component_sum,
                "additivity_residual_D": component_sum - aggregate_delta,
                "aggregate_lambda_control": crossover_lambda(left, right),
                "aggregate_lambda_actionable_for_train_decision": False,
                "typed_bucket_count": len((*SEMANTIC_STRATA, "POSE6_GLOBAL"))
                * len(SCORER_VISIBILITIES)
                * len(G4_TEMPORAL_CLASSES),
                "typed_actionable_bucket_count": 0,
                "SOLVE": "BLOCKED_TYPED_G4_AND_DIMENSION_RATE_HOME_CUSTODY",
            }
        )
    return {
        "schema": "ddm_rd1_typed_dimension_duals.v1",
        "axes": {
            "stratum": [*SEMANTIC_STRATA, "POSE6_GLOBAL"],
            "scorer_visibility": list(SCORER_VISIBILITIES),
            "g4_temporal_class": list(G4_TEMPORAL_CLASSES),
        },
        "aggregate_scalarization_policy": (
            "diagnostic control only; never a train-decision price and never "
            "summed with shared-cost component conditionals"
        ),
        "component_evidence": component_evidence,
        "bucket_rows": bucket_rows,
        "edge_summaries": edge_summaries,
        "actionable_bucket_count": 0,
        "blocker": (
            "existing n600 endpoints lack joint candidate-delta x G4-class x "
            "dimension-rate-home custody"
        ),
        "score_claim": False,
    }


def metric_active_continuation_geometry_report(
    continuation: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Classify the geometry used by a completed discrete continuation.

    A previous candidate identifier is a graph-topology warm-start token, not a
    vector-space displacement. Continuous proposal geometry remains blocked
    until its scorer-metric tensors and cross-metric readback are custodied.
    """

    if not continuation:
        raise LambdaContinuationError("metric geometry requires continuation rows")
    for row in continuation:
        if (
            row.get("neighbor_only") is not True
            or row.get("restricted_global_rank_verified") is not True
            or not isinstance(row.get("warm_start_from"), str)
            or not isinstance(row.get("corrector_path"), list)
        ):
            raise LambdaContinuationError("continuation geometry custody differs")
    return {
        "schema": "ddm_rd1_metric_active_continuation_geometry.v1",
        "completed_discrete_solve": {
            "selection_geometry": "ORDERED_FINITE_NEIGHBOR_GRAPH",
            "selection_units": "counted_bytes_plus_lambda_times_realized_contest_D",
            "warm_start_token": "previous_candidate_identifier_only",
            "state_space_distance_used": False,
            "trust_region_used": False,
            "step_direction_used": False,
            "identity_L2_used": False,
            "full_rank_finite_domain_verification": True,
            "status": "VALID_NO_CONTINUOUS_GEOMETRY_INVOKED",
        },
        "continuous_proposal_geometry_contract": {
            "seg": {
                "metric": "margin-Fisher surrogate plus rank-4 head hyperplane geometry",
                "law_ids": [
                    "frozen_scorer_fisher_curvature_margin_colocation_v1",
                    "segnet_head_rank4_linear_flipdist_v1",
                ],
                "identity_L2_allowed": "LABELED_CONTROL_ONLY",
            },
            "pose": {
                "metric": "exact low-rank <=6-dimensional PoseNet output quadratic",
                "parameter_L2_allowed": False,
            },
            "distribution": {
                "metric": "Bregman divergence",
                "binding": "policy_bindings.optimal_metric",
                "identity_L2_allowed": "LABELED_CONTROL_ONLY",
            },
            "composite_readback": {
                "law": "dual-metric Euclidean-versus-Fisher readback",
                "required_fields": [
                    "euclidean_control",
                    "fisher_or_pose_quadratic",
                    "cross_metric_cosine",
                    "relative_norm",
                ],
                "single_metric_verdict_allowed": False,
            },
            "applies_to": [
                "predictor_corrector_step_direction",
                "trust_radius",
                "warm_start_distance",
                "per_bucket_move_proposal_ranking",
            ],
            "actionable": False,
            "blocker": (
                "RD1 endpoint receipts lack per-candidate margin-Fisher pullbacks, "
                "Pose quadratic tensors, Bregman binding receipts, and dual-metric "
                "readbacks for proposed continuous moves"
            ),
        },
        "aggregate_dual_units_unchanged": True,
        "lambda_points_reused_without_restart": True,
        "score_claim": False,
    }


def second_order_metric_geometry_addendum_report(
    metric_geometry: Mapping[str, Any],
) -> dict[str, Any]:
    """Extend the metric contract with order, coordinate, and ladder custody."""

    discrete = metric_geometry.get("completed_discrete_solve")
    continuous = metric_geometry.get("continuous_proposal_geometry_contract")
    if (
        not isinstance(discrete, Mapping)
        or discrete.get("status") != "VALID_NO_CONTINUOUS_GEOMETRY_INVOKED"
        or not isinstance(continuous, Mapping)
        or continuous.get("actionable") is not False
    ):
        raise LambdaContinuationError("base metric geometry custody differs")
    return {
        "schema": "ddm_rd1_second_order_metric_geometry_addendum.v1",
        "completed_discrete_solve": {
            "continuous_approximation_order_used": None,
            "gradient_first_order_naive_used": False,
            "hessian_used": False,
            "reason": (
                "exact finite graph enumeration and full-domain rank require "
                "neither a gradient nor a local quadratic approximation"
            ),
            "status": "VALID_NO_LOCAL_CONTINUOUS_MODEL_INVOKED",
        },
        "future_continuous_move_contract": {
            "second_order_from_step_one_when_quadratic_measured": True,
            "second_order_inventory": [
                {
                    "catalog": 391,
                    "surface": "exact composite-R adjoint",
                },
                {
                    "catalog": 423,
                    "surface": "Hessian-preconditioned TerminalSolve",
                },
                {
                    "catalog": 552,
                    "surface": "SPD normal-coordinate momentum",
                },
            ],
            "coordinate_system": {
                "seg": (
                    "rank-4 scorer-head class-pair hyperplanes and feature "
                    "channels"
                ),
                "flip_distance": "abs(margin)/norm(delta_w_class_pair)",
                "pose": "exact <=6-dimensional PoseNet output chart",
                "parameter_coordinates_allowed": False,
            },
            "ladder_order": [
                "most-optimal measured second-order scorer geometry",
                "measured metric-active simpler geometry",
                "identity-L2 or first-order naive labeled controls",
            ],
            "simpler_geometry_is_control_only": True,
            "actionable": False,
            "blocker": (
                "RD1 endpoints do not retain an exact composite-R adjoint, "
                "per-bucket Hessian/SPD normal chart, scorer-channel proposal "
                "coordinates, or an optimal-first measured geometry ladder"
            ),
        },
        "aggregate_dual_units_unchanged": True,
        "lambda_points_reused_without_restart": True,
        "score_claim": False,
    }


def normalized_knee(
    hull: Sequence[MeasuredDescription],
) -> MeasuredDescription:
    """Choose the maximum normalized elbow in log-rate/distortion coordinates."""

    if len(hull) < 3:
        return hull[0]
    xs = [math.log1p(row.counted_bytes) for row in hull]
    ys = [row.distortion for row in hull]
    x0, x1 = xs[0], xs[-1]
    y0, y1 = ys[0], ys[-1]
    x_scale = x1 - x0
    y_scale = y0 - y1
    if x_scale <= 0.0 or y_scale <= 0.0:
        raise LambdaContinuationError("knee requires increasing rate and decreasing distortion")
    points = [((x - x0) / x_scale, (y - y1) / y_scale) for x, y in zip(xs, ys, strict=True)]
    ax, ay = points[0]
    bx, by = points[-1]
    denom = math.hypot(by - ay, bx - ax)
    distances = [abs((by - ay) * x - (bx - ax) * y + bx * ay - by * ax) / denom for x, y in points]
    interior = range(1, len(hull) - 1)
    return hull[max(interior, key=lambda i: (distances[i], -hull[i].counted_bytes))]


def publish_immutable_json(path: Path, value: Any) -> None:
    """Atomically create an immutable canonical JSON checkpoint."""

    payload = canonical_json_bytes(value) + b"\n"
    if path.exists():
        if path.read_bytes() != payload:
            raise LambdaContinuationError(f"immutable checkpoint differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


__all__ = [
    "EVIDENCE_AXIS",
    "G4_TEMPORAL_CLASSES",
    "SCORER_VISIBILITIES",
    "SEMANTIC_STRATA",
    "CodedStream",
    "LambdaContinuationError",
    "MeasuredDescription",
    "canonical_json_bytes",
    "continuation_rows",
    "crossover_lambda",
    "discrete_dual_rows",
    "effective_quantum_D",
    "geometric_curvature_ladder",
    "lower_supported_hull",
    "metric_active_continuation_geometry_report",
    "normalized_knee",
    "pareto_nondominated",
    "publish_immutable_json",
    "realized_distortion",
    "second_order_metric_geometry_addendum_report",
    "sha256_bytes",
    "typed_dimension_dual_report",
]
