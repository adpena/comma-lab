#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure DDM v17's iterative lattice-aware realized-feedback solve.

One invocation advances one immutable stage: one trust-region iteration, the
full-n600 chunked receiver measurement, or the final receipt.  The scorer is
encode-side only.  Every proposal is an integer carrier state compiled through
the counted v16 receiver before fresh CPU-Torch SegNet/PoseNet measurement.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, model_validator

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "src", REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap  # noqa: E402
from tac.optimization.coupled_margin_levelset import (  # noqa: E402
    CouplingOperator,
    gauss_newton_hessian,
    predicted_margin,
    solve_active_set_kkt,
)
from tac.optimization.direct_description_carrier_compose import (  # noqa: E402
    receive_carrier_compose_archive,
    rfc8785_canonicalize,
)
from tac.optimization.direct_description_coupled_margin import (  # noqa: E402
    encode_coupled_margin_program,
)
from tac.optimization.direct_description_joint_descent import (  # noqa: E402
    clipped_adam_step,
    initial_adam_state,
)
from tac.optimization.direct_description_minimizer import DirectDescriptionError, _read_regular_file_once  # noqa: E402
from tac.optimization.iterative_realized_trust_region import (  # noqa: E402
    HardCandidate,
    LatticeCandidate,
    TrustRegionPolicy,
    bounded_parallel_tempering,
    categorical_fisher_trace_from_margin,
    fisher_margin_debt,
    quantized_babai_candidates,
    ranked_prefix_sign_candidates,
    select_realized_improvement,
    summarize_validity_curve,
    update_trust_radius,
)
from tac.scorer import make_scorers_differentiable  # noqa: E402
from tools.measure_ddm_v14_realization_fidelity import (  # noqa: E402
    EVIDENCE_AXIS,
    POINTER_SCORE_TEXT,
    _load_models,
    _publish_immutable,
    _storage_preflight,
)
from tools.measure_ddm_v16_coupled_joint_solve import (  # noqa: E402
    V14_DSEG,
    DDMV16CoupledJointSolveConfigV1,
    _archive_for_state,
    _assemble_operator,
    _base_v14_bytes,
    _build_problem,
    _byte_diff,
    _dev_measurement,
    _json,
    _ladder_archive,
    _measure_window,
    _portable,
    _required_for_rows,
    _set_state_bank,
    _sha256,
    _sha256_array,
    _target_custody,
    _v15_bindings,
    _write_json,
    _write_npz,
)

RESULT_SCHEMA = "ddm_v17_iterative_realized_trust_region_receipt.v1"
ITERATION_SCHEMA = "ddm_v17_iterative_realized_trust_region_iteration.v1"
N600_SCHEMA = "ddm_v17_iterative_realized_trust_region_n600.v1"
LANE_ID = "ddm_v17_iterative_realized_trust_region_solve"
G3_CORRELATION = 0.5953065905


class DDMV17IterativeRealizedTrustRegionConfigV1(BaseModel):
    """SHA-bound local-only contract for one resumable v17 measurement."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    schema_: Literal["DDMV17IterativeRealizedTrustRegionConfigV1"] = Field(
        default="DDMV17IterativeRealizedTrustRegionConfigV1",
        alias="schema",
        serialization_alias="schema",
    )
    run_id: str = Field(min_length=8)
    seed: StrictInt = 1234
    v16_config_path: str = Field(min_length=1)
    v16_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    hard_pair_registry_path: str = Field(min_length=1)
    hard_pair_registry_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    representative_source_pair_ids: tuple[StrictInt, ...]
    maximum_iterations: StrictInt = Field(default=4, ge=1, le=8)
    maximum_candidates: StrictInt = Field(default=6, ge=2, le=12)
    candidate_scales: tuple[StrictFloat, ...]
    g2f_amplitude_ladder: tuple[StrictFloat, ...]
    initial_trust_radius_u8: StrictFloat = Field(default=1.0, gt=0.0)
    minimum_trust_radius_u8: StrictFloat = Field(default=0.5, gt=0.0)
    maximum_trust_radius_u8: StrictFloat = Field(default=16.0, gt=0.0)
    maximum_target_cells_per_role_pair: StrictInt = Field(default=1, ge=1, le=4)
    maximum_protected_cells_per_pair: StrictInt = Field(default=2, ge=1, le=8)
    maximum_fd_entries: StrictInt = Field(default=8, ge=1, le=32)
    vjp_chunk_rows: StrictInt = Field(default=8, ge=1, le=32)
    target_margin_epsilon: StrictFloat = Field(default=0.05, ge=0.0, le=1.0)
    protected_margin_epsilon: StrictFloat = Field(default=0.0, ge=0.0, le=1.0)
    pose_trust_radius: StrictFloat = Field(default=0.05, gt=0.0, le=1.0)
    scorer_threads: StrictInt = Field(default=4, ge=1, le=16)
    scorer_batch_size: Literal[16] = 16
    archive_box_bytes: Literal[160000] = 160000
    enable_parallel_tempering: Literal[True] = True
    tempering_coordinates: StrictInt = Field(default=8, ge=1, le=32)
    tempering_sweeps: StrictInt = Field(default=4, ge=1, le=16)
    research_only: Literal[True] = True
    execution_allowed: Literal[False] = False
    score_claim: Literal[False] = False
    d_seg_claim: Literal[False] = False
    d_pose_claim: Literal[False] = False

    @model_validator(mode="after")
    def _valid(self) -> DDMV17IterativeRealizedTrustRegionConfigV1:
        if len(self.representative_source_pair_ids) != 24 or len(set(self.representative_source_pair_ids)) != 24:
            raise ValueError("v17 screening must bind the g3 top24 unique pair set")
        if tuple(sorted(set(self.g2f_amplitude_ladder))) != self.g2f_amplitude_ladder or self.g2f_amplitude_ladder != (
            0.5,
            1.0,
            2.0,
            4.0,
            8.0,
            16.0,
        ):
            raise ValueError("v17 must retain the measured g2f amplitude ladder")
        if self.initial_trust_radius_u8 != 1.0:
            raise ValueError("v17 pixel-lattice trust must initialize at the measured 1-LSB knee")
        if not self.minimum_trust_radius_u8 <= self.initial_trust_radius_u8 <= self.maximum_trust_radius_u8:
            raise ValueError("v17 trust radius bounds do not contain the initial radius")
        if not self.candidate_scales or any(value <= 0.0 for value in self.candidate_scales):
            raise ValueError("v17 candidate scales must be positive")
        return self

    def typed_config_hash(self) -> str:
        return _sha256(rfc8785_canonicalize(self.model_dump(mode="json", by_alias=True)))


def _bound_config(config: DDMV17IterativeRealizedTrustRegionConfigV1) -> DDMV16CoupledJointSolveConfigV1:
    path = REPO_ROOT / config.v16_config_path
    payload = _read_regular_file_once(path)
    if _sha256(payload) != config.v16_config_sha256:
        raise DirectDescriptionError("v17 bound v16 config SHA-256 differs")
    return DDMV16CoupledJointSolveConfigV1.model_validate_json(payload)


def _validate_screening_registry(config: DDMV17IterativeRealizedTrustRegionConfigV1) -> dict[str, Any]:
    path = REPO_ROOT / config.hard_pair_registry_path
    payload = _read_regular_file_once(path)
    if _sha256(payload) != config.hard_pair_registry_sha256:
        raise DirectDescriptionError("v17 hard-pair registry SHA-256 differs")
    registry = json.loads(payload)
    if tuple(registry.get("top24", ())) != config.representative_source_pair_ids:
        raise DirectDescriptionError("v17 pair ids differ from the bound g3 top24 registry")
    correlation = registry.get("correlation", {}).get("top24", {}).get("pearson_r")
    if correlation is None:
        correlation = registry.get("replay", {}).get("top24", {}).get("pearson_r")
    # The compact registry schema carries the value in one of several named
    # summary sections across historical versions.  The typed pair list is the
    # hard identity; the published scalar is reasserted explicitly below.
    if correlation is not None and not math.isclose(float(correlation), G3_CORRELATION, abs_tol=5e-10):
        raise DirectDescriptionError("v17 g3 top24 correlation anchor differs")
    return {
        "path": config.hard_pair_registry_path,
        "bytes": len(payload),
        "sha256": _sha256(payload),
        "top24_pair_ids": list(config.representative_source_pair_ids),
        "top24_full_pearson_r": G3_CORRELATION,
        "authority": "SCREENING_ONLY_NEVER_RANK_OR_KILL",
    }


def _policy(config: DDMV17IterativeRealizedTrustRegionConfigV1) -> TrustRegionPolicy:
    return TrustRegionPolicy(
        minimum_radius=config.minimum_trust_radius_u8,
        maximum_radius=config.maximum_trust_radius_u8,
    )


def _state_from_iterations(
    problem: Mapping[str, Any], iterations: Sequence[Mapping[str, Any]]
) -> tuple[np.ndarray, np.ndarray, dict[str, list[int]], float]:
    state = {
        "template_values_u8": problem["initial_template_values_u8"],
        "compensation_rgb_i8": problem["initial_compensation_rgb_i8"],
        "phases": problem["initial_phases"],
    }
    radius = None
    for row in iterations:
        state = row["selected_state"]
        radius = float(row["trust"]["new_radius_u8"])
    return (
        np.asarray(state["template_values_u8"], dtype=np.int16),
        np.asarray(state["compensation_rgb_i8"], dtype=np.int16),
        {key: list(value) for key, value in state["phases"].items()},
        float(radius if radius is not None else 1.0),
    )


def _bounds(values: np.ndarray, compensation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.concatenate((np.zeros(values.size, dtype=np.int64), np.full(compensation.size, -127, dtype=np.int64))),
        np.concatenate((np.full(values.size, 255, dtype=np.int64), np.full(compensation.size, 127, dtype=np.int64))),
    )


def _model_rows(operator: CouplingOperator, rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    weights = np.ones(operator.matrix.shape[0], dtype=np.float64)
    seg_indexes = np.asarray(
        [index for index, row in enumerate(rows) if row["kind"] in {"target", "protected"}], dtype=np.int64
    )
    if seg_indexes.size:
        weights[seg_indexes] = 1.0e-6 + categorical_fisher_trace_from_margin(operator.margin[seg_indexes])
    description = np.eye(operator.matrix.shape[1], dtype=np.float64)
    template_dofs = sum(label.startswith("template:") for label in operator.dof_labels)
    description[template_dofs:, template_dofs:] *= 4.0
    return weights, description


def _proposal_families(
    *,
    config: DDMV17IterativeRealizedTrustRegionConfigV1,
    operator: CouplingOperator,
    rows: Sequence[Mapping[str, Any]],
    current: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    radius: float,
    iteration: int,
) -> tuple[
    tuple[tuple[str, LatticeCandidate], ...],
    dict[str, Any],
    tuple[LatticeCandidate, ...],
]:
    weights, description = _model_rows(operator, rows)
    solved = solve_active_set_kkt(
        operator,
        description_metric=description,
        row_weights=weights,
        damping=1.0e-4,
        trust_radius=radius,
        use_gauss_newton=True,
        tolerance=1.0e-7,
    )
    deficit = np.maximum(operator.required_margin - operator.margin, 0.0)
    active_weight = weights * (deficit > 0.0)
    hessian = gauss_newton_hessian(
        operator.matrix,
        description_metric=description,
        row_weights=weights,
        damping=1.0e-4,
    )
    gradient = -(operator.matrix.T @ active_weight)
    preconditioned = np.linalg.solve(hessian, -gradient)
    norm = float(np.linalg.norm(gradient))
    adam = clipped_adam_step(
        initial_adam_state(current.size),
        gradient.astype(np.float32),
        learning_rate=max(config.minimum_trust_radius_u8, radius * 0.25),
        grad_clip=max(norm, 1.0),
        ema_decay=0.999,
    )
    radii = config.g2f_amplitude_ladder if iteration == 1 else (radius,)
    solve_rows: list[tuple[str, LatticeCandidate]] = []
    seen: set[bytes] = set()
    # First iteration measures the full #599 validity ladder.  Later rounds
    # remain inside the adaptive active radius.
    for source, continuous, metric in (
        ("kkt", solved.step, solved.hessian),
        ("M_preconditioned", preconditioned, hessian),
    ):
        for candidate_radius in radii:
            per_radius_cap = 1 if iteration == 1 else config.maximum_candidates - len(solve_rows)
            candidates = quantized_babai_candidates(
                continuous,
                metric,
                current=current,
                lower=lower,
                upper=upper,
                trust_radius=float(candidate_radius),
                scales=config.candidate_scales,
                maximum_candidates=max(1, per_radius_cap),
            )
            for candidate in candidates:
                digest = candidate.integer_step.tobytes()
                if digest in seen:
                    continue
                seen.add(digest)
                solve_rows.append((f"{source}_r{float(candidate_radius):g}", candidate))
                if len(solve_rows) == config.maximum_candidates:
                    break
            if len(solve_rows) == config.maximum_candidates:
                break
        if len(solve_rows) == config.maximum_candidates:
            break
    collapsed_babai_count = len(solve_rows)
    used_ranked_preconditioned_fallback = collapsed_babai_count < config.maximum_candidates
    if used_ranked_preconditioned_fallback:
        # Equality-constrained template subspaces can map several trust radii
        # to the same Babai point.  Do not pad the budget with duplicate scorer
        # calls.  The damped M^TWM-preconditioned direction remains the model
        # proposal, while stable prefixes provide four distinct lattice points.
        fallback_radii = tuple(float(value) for value in radii[: config.maximum_candidates])
        if len(fallback_radii) < config.maximum_candidates:
            fallback_radii += (float(radius),) * (config.maximum_candidates - len(fallback_radii))
        fallbacks = ranked_prefix_sign_candidates(
            preconditioned,
            current=current,
            lower=lower,
            upper=upper,
            trust_radii=tuple(max(1.0, value) for value in fallback_radii),
        )
        solve_rows = [
            (f"M_preconditioned_ranked_prefix_r{fallback_radii[index]:g}", candidate)
            for index, candidate in enumerate(fallbacks)
        ]
    control_radii = tuple(float(value) for value in radii[: len(solve_rows)])
    if len(control_radii) < len(solve_rows):
        control_radii += (float(radius),) * (len(solve_rows) - len(control_radii))
    descent = ranked_prefix_sign_candidates(
        adam.theta,
        current=current,
        lower=lower,
        upper=upper,
        # A sub-unit trust radius has no nonzero uint8 point.  The matched
        # control therefore records the minimum realizable one-quantum step.
        trust_radii=tuple(max(1.0, value) for value in control_radii),
    )
    solve_receipt = {
        "kkt_diagnostics": asdict(solved.diagnostics),
        "kkt_step_sha256": _sha256_array(solved.step),
        "gauss_newton_hessian_sha256": _sha256_array(hessian),
        "fisher_row_weights_sha256": _sha256_array(weights),
        "preconditioned_step_sha256": _sha256_array(preconditioned),
        "j2_clipped_adam_step_sha256": _sha256_array(adam.theta),
        "model": "Fisher-margin weighted M QP/KKT plus damped M-preconditioned fallback",
        "j2_control": (
            "QP/KKT-disabled clipped-Adam direction, stable ranked-prefix sign lattice controls "
            "with one unique exact receiver call per solve call"
        ),
        "matched_exact_call_budget": len(descent) == len(solve_rows),
        "collapsed_unique_babai_count": collapsed_babai_count,
        "used_ranked_M_preconditioned_fallback": used_ranked_preconditioned_fallback,
    }
    return tuple(solve_rows), solve_receipt, tuple(descent)


def _materialize_candidate(
    *,
    label: str,
    candidate: LatticeCandidate,
    config: DDMV17IterativeRealizedTrustRegionConfigV1,
    problem: Mapping[str, Any],
    values: np.ndarray,
    compensation: np.ndarray,
    phases: Mapping[str, Sequence[int]],
    current: np.ndarray,
    v14_archive: bytes,
    operator: CouplingOperator,
    rows: Sequence[Mapping[str, Any]],
    baseline_measurement: Mapping[str, Any],
    baseline_model_debt: float,
    active_trust_radius: float,
    labels_all: np.ndarray,
    poses_all: np.ndarray,
    segnet: Any,
    posenet: Any,
    root: Path,
    iteration: int,
) -> dict[str, Any]:
    parameters = current + candidate.integer_step
    patch_size = values.size
    new_values = parameters[:patch_size].reshape(values.shape).astype(np.uint8)
    new_compensation = parameters[patch_size:].reshape(compensation.shape).astype(np.int16)
    pair_ids = tuple(int(value) for value in problem["representative_source_pair_ids"])
    archive, nested_base, program = _archive_for_state(
        v14_archive,
        new_values,
        pair_ids,
        phases,
        problem["sparse_compensation_support"],
        new_compensation,
    )
    archive_path = root / "iteration_candidates" / f"iteration_{iteration:02d}_{label}.zip.receipt-bytes"
    _publish_immutable(archive_path, archive)
    required = _required_for_rows(config, rows)
    measurement, realized_margin, _camera = _dev_measurement(
        archive=archive,
        pair_ids=pair_ids,
        rows=rows,
        required=required,
        labels_all=labels_all,
        poses_all=poses_all,
        segnet=segnet,
        posenet=posenet,
    )
    predicted = predicted_margin(operator, candidate.integer_step.astype(np.float64))
    predicted_reduction = baseline_model_debt - fisher_margin_debt(predicted, required)
    realized_reduction = baseline_model_debt - fisher_margin_debt(realized_margin, required)
    rho = realized_reduction / predicted_reduction if predicted_reduction > 0.0 else None
    baseline_constraints = baseline_measurement["constraints"]
    constraints = measurement["constraints"]
    baseline_pose = int(baseline_constraints["pose_upper"]["violations"]) + int(
        baseline_constraints["pose_lower"]["violations"]
    )
    candidate_pose = int(constraints["pose_upper"]["violations"]) + int(constraints["pose_lower"]["violations"])
    inside_active_trust = bool(
        np.all(np.abs(candidate.integer_step.astype(np.float64)) <= active_trust_radius + 1.0e-12)
    )
    admissible = (
        inside_active_trust
        and int(constraints["protected"]["violations"]) <= int(baseline_constraints["protected"]["violations"])
        and candidate_pose <= baseline_pose
        and int(measurement["archive_bytes"]) <= config.archive_box_bytes
    )
    return {
        "label": label,
        "candidate_id": candidate.candidate_id,
        "state": {
            "template_values_u8": new_values.tolist(),
            "compensation_rgb_i8": new_compensation.tolist(),
            "phases": {key: list(value) for key, value in phases.items()},
        },
        "archive": {"path": _portable(archive_path), "bytes": len(archive), "sha256": _sha256(archive)},
        "nested_base_bytes": len(nested_base),
        "program_bytes": len(encode_coupled_margin_program(program)),
        "measurement": measurement,
        "integer_step": candidate.integer_step.tolist(),
        "integer_step_l2": float(np.linalg.norm(candidate.integer_step)),
        "lattice_quanta": int(np.max(np.abs(candidate.integer_step), initial=0)),
        "scale": candidate.scale,
        "quadratic_error": candidate.quadratic_error,
        "covering_bound": candidate.covering_bound,
        "predicted_reduction": predicted_reduction,
        "realized_reduction": realized_reduction,
        "rho": None if rho is None or not math.isfinite(rho) else rho,
        "admissible": admissible,
        "inside_active_trust": inside_active_trust,
        "exact_receiver_realized": True,
        "score_claim": False,
    }


def _hard_candidate(row: Mapping[str, Any]) -> HardCandidate:
    measurement = row["measurement"]
    return HardCandidate(
        candidate_id=str(row["label"]),
        hard_objective=float(measurement["advisory_score_formula_value"]),
        d_seg=float(measurement["d_seg"]),
        archive_bytes=int(measurement["archive_bytes"]),
        admissible=bool(row["admissible"]),
        predicted_reduction=float(row["predicted_reduction"]),
        realized_model_reduction=float(row["realized_reduction"]),
        integer_step=np.asarray(row["integer_step"], dtype=np.int64),
    )


def _selected_row(rows: Sequence[Mapping[str, Any]], candidate_id: str) -> dict[str, Any]:
    return dict(next(row for row in rows if row["label"] == candidate_id))


def _tempering_escape(
    *,
    config: DDMV17IterativeRealizedTrustRegionConfigV1,
    operator: CouplingOperator,
    rows: Sequence[Mapping[str, Any]],
    current: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    radius: float,
    materialize_absolute: Any,
    gradient: np.ndarray,
    baseline_objective: float,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    bound = max(1, math.ceil(radius))
    local_lower = np.maximum(lower, current - bound)
    local_upper = np.minimum(upper, current + bound)
    coordinates = tuple(
        int(value) for value in np.argsort(-np.abs(gradient), kind="stable")[: config.tempering_coordinates]
    )
    required = operator.required_margin
    cache: dict[bytes, dict[str, Any]] = {}

    def cheap_energy(state: np.ndarray) -> float:
        delta = state.astype(np.float64) - current
        return fisher_margin_debt(predicted_margin(operator, delta), required)

    def hard_key(state: np.ndarray) -> tuple[float, ...]:
        if np.array_equal(state, current):
            return (0.0, baseline_objective)
        digest = state.tobytes()
        if digest not in cache:
            cache[digest] = materialize_absolute(state, len(cache))
        row = cache[digest]
        measurement = row["measurement"]
        return (0.0 if row["admissible"] else 1.0, float(measurement["advisory_score_formula_value"]))

    result = bounded_parallel_tempering(
        current,
        lower=local_lower,
        upper=local_upper,
        coordinates=coordinates,
        cheap_energy=cheap_energy,
        hard_key=hard_key,
        seed=config.seed,
        sweeps=config.tempering_sweeps,
    )
    selected = None
    if result.selected_replica is not None:
        terminal = next(row for row in result.terminals if row.replica == result.selected_replica)
        selected = cache[terminal.state.tobytes()]
    receipt = {
        "status": result.status,
        "temperatures": list(result.temperatures),
        "proposals": result.proposals,
        "cheap_accepts": result.cheap_accepts,
        "swaps": result.swaps,
        "hard_terminal_count": len(result.terminals),
        "selected_replica": result.selected_replica,
        "coordinates": list(coordinates),
        "deterministic_seed": config.seed,
        "hard_measurements": list(cache.values()),
        "authority": "cheap_energy_traversal_only_terminal_receiver_measurement_accepts",
    }
    return selected, receipt


def _run_iteration(
    *,
    config: DDMV17IterativeRealizedTrustRegionConfigV1,
    root: Path,
    problem: Mapping[str, Any],
    iterations: Sequence[Mapping[str, Any]],
    n600_cfg: Any,
    labels_all: np.ndarray,
    poses_all: np.ndarray,
    segnet: Any,
    posenet: Any,
) -> dict[str, Any]:
    iteration = len(iterations) + 1
    path = root / "stage_checkpoints" / f"iteration_{iteration:02d}.json"
    if path.exists():
        return _json(path)
    values, compensation, phases, radius = _state_from_iterations(problem, iterations)
    v14_archive = _base_v14_bytes(n600_cfg)
    operator, rows, clusters, current, current_camera = _assemble_operator(
        config=config,
        problem=problem,
        v14_archive=v14_archive,
        values=values,
        compensation=compensation,
        phases=phases,
        segnet=segnet,
        posenet=posenet,
    )
    operator_path = root / "operators" / f"iteration_{iteration:02d}_M.npz"
    _write_npz(
        operator_path,
        M=operator.matrix,
        margin=operator.margin,
        required_margin=operator.required_margin,
        parameters=current,
    )
    pair_ids = tuple(int(value) for value in problem["representative_source_pair_ids"])
    baseline_archive, _nested, _program = _archive_for_state(
        v14_archive,
        values.astype(np.uint8),
        pair_ids,
        phases,
        problem["sparse_compensation_support"],
        compensation,
    )
    baseline_required = _required_for_rows(config, rows)
    baseline_measurement, baseline_margin, _camera = _dev_measurement(
        archive=baseline_archive,
        pair_ids=pair_ids,
        rows=rows,
        required=baseline_required,
        labels_all=labels_all,
        poses_all=poses_all,
        segnet=segnet,
        posenet=posenet,
    )
    if _sha256_array(current_camera) != baseline_measurement["camera_sha256"]:
        raise DirectDescriptionError("v17 linearization origin differs from hard baseline camera")
    baseline_model_debt = fisher_margin_debt(baseline_margin, baseline_required)
    lower, upper = _bounds(values, compensation)
    solve_candidates, solve_receipt, descent_candidates = _proposal_families(
        config=config,
        operator=operator,
        rows=rows,
        current=current.astype(np.int64),
        lower=lower,
        upper=upper,
        radius=radius,
        iteration=iteration,
    )

    def materialize(label: str, candidate: LatticeCandidate) -> dict[str, Any]:
        return _materialize_candidate(
            label=label,
            candidate=candidate,
            config=config,
            problem=problem,
            values=values,
            compensation=compensation,
            phases=phases,
            current=current.astype(np.int64),
            v14_archive=v14_archive,
            operator=operator,
            rows=rows,
            baseline_measurement=baseline_measurement,
            baseline_model_debt=baseline_model_debt,
            active_trust_radius=radius,
            labels_all=labels_all,
            poses_all=poses_all,
            segnet=segnet,
            posenet=posenet,
            root=root,
            iteration=iteration,
        )

    solve_rows = [
        materialize(f"solve_{index:02d}_{source}", candidate)
        for index, (source, candidate) in enumerate(solve_candidates)
    ]
    descent_rows = [
        materialize(f"j2_descent_{index:02d}", candidate)
        for index, candidate in enumerate(descent_candidates[: len(solve_rows)])
    ]
    baseline_objective = float(baseline_measurement["advisory_score_formula_value"])
    selection = select_realized_improvement(baseline_objective, tuple(_hard_candidate(row) for row in solve_rows))
    selected = None if selection.selected is None else _selected_row(solve_rows, selection.selected.candidate_id)
    active_trial_rows = [row for row in solve_rows if row["inside_active_trust"]]
    trial = max(
        active_trial_rows,
        key=lambda row: (
            float(row["predicted_reduction"]),
            -float(row["measurement"]["advisory_score_formula_value"]),
        ),
        default=None,
    )
    trust_rho = selection.rho
    if selected is None and trial is not None and float(trial["predicted_reduction"]) > 0.0:
        trust_rho = float(trial["realized_reduction"]) / float(trial["predicted_reduction"])
    decision = update_trust_radius(radius, rho=trust_rho, accepted=selected is not None, policy=_policy(config))
    gradient = operator.matrix.T @ np.maximum(operator.required_margin - operator.margin, 0.0)
    tempering_receipt: dict[str, Any] = {"status": "NOT_TRIGGERED_ACTIVE_RADIUS_OR_ACCEPTED"}
    if selected is None and config.enable_parallel_tempering and decision.new_radius <= config.minimum_trust_radius_u8:

        def materialize_absolute(state: np.ndarray, terminal_index: int) -> dict[str, Any]:
            delta = state.astype(np.int64) - current.astype(np.int64)
            # PT terminal states are already integer and bounded.  A unit
            # metric wrapper reuses the exact materialization path without a
            # second rounding operation.
            candidate = LatticeCandidate(
                candidate_id=f"pt_terminal_{terminal_index:02d}",
                scale=1.0,
                integer_step=delta,
                continuous_step=delta.astype(np.float64),
                quadratic_error=0.0,
                covering_bound=0.25 * delta.size,
                clipped_coordinate_count=0,
            )
            return materialize(candidate.candidate_id, candidate)

        pt_selected, tempering_receipt = _tempering_escape(
            config=config,
            operator=operator,
            rows=rows,
            current=current.astype(np.int64),
            lower=lower,
            upper=upper,
            radius=decision.new_radius,
            materialize_absolute=materialize_absolute,
            gradient=gradient,
            baseline_objective=baseline_objective,
        )
        if pt_selected is not None:
            pt_choice = select_realized_improvement(baseline_objective, (_hard_candidate(pt_selected),))
            if pt_choice.accepted:
                selected = pt_selected
                selection = pt_choice

    if selected is None:
        selected_state = {
            "template_values_u8": values.astype(np.uint8).tolist(),
            "compensation_rgb_i8": compensation.tolist(),
            "phases": {key: list(value) for key, value in phases.items()},
        }
        selected_archive = {
            "path": None,
            "bytes": len(baseline_archive),
            "sha256": _sha256(baseline_archive),
        }
        selected_measurement = dict(baseline_measurement)
        selected_label = "hold_control"
    else:
        selected_state = selected["state"]
        selected_archive = selected["archive"]
        selected_measurement = selected["measurement"]
        selected_label = selected["label"]
    rejections = 0
    for row in reversed(iterations):
        if row["accepted"]:
            break
        rejections += 1
    if selected is None:
        rejections += 1
    plateau = bool(
        selected is None
        and decision.new_radius <= config.minimum_trust_radius_u8
        and (rejections >= 2 or tempering_receipt.get("status") not in {"NOT_TRIGGERED_ACTIVE_RADIUS_OR_ACCEPTED"})
    )
    descent_best = select_realized_improvement(baseline_objective, tuple(_hard_candidate(row) for row in descent_rows))
    receipt = {
        "schema": ITERATION_SCHEMA,
        "lane_id": LANE_ID,
        "run_id": config.run_id,
        "iteration": iteration,
        "typed_config_sha256": config.typed_config_hash(),
        "linearization": {
            "operator_path": _portable(operator_path),
            "operator_sha256": _sha256(_read_regular_file_once(operator_path)),
            "shape": list(operator.matrix.shape),
            "activation_pattern_sha256": operator.activation_pattern_sha256,
            "origin_archive_sha256": _sha256(baseline_archive),
            "origin_camera_sha256": baseline_measurement["camera_sha256"],
            "current_realized_point": True,
            "fresh_after_prior_accept": not iterations or bool(iterations[-1]["accepted"]),
            "finite_difference_all_clusters_passed": all(row["finite_difference"]["passed"] for row in clusters),
            "pair_clusters": clusters,
        },
        "model_solve": solve_receipt,
        "baseline_measurement": baseline_measurement,
        "baseline_fisher_margin_debt": baseline_model_debt,
        "solve_candidates": solve_rows,
        "j2_descent_candidates": descent_rows,
        "selection": {
            "selected_candidate_id": None if selection.selected is None else selection.selected.candidate_id,
            "accepted": selection.accepted,
            "baseline_objective": selection.baseline_objective,
            "objective_improvement": selection.objective_improvement,
            "rho": selection.rho,
            "evaluated_count": selection.evaluated_count,
            "admissible_count": selection.admissible_count,
        },
        "selected_label": selected_label,
        "selected_state": selected_state,
        "selected_archive": selected_archive,
        "selected_measurement": selected_measurement,
        "accepted": selected is not None,
        "trust": {
            "source": "g2f_amplitude_ladder_pixel_knee_then_classical_rho",
            "g2f_amplitude_ladder": list(config.g2f_amplitude_ladder),
            "old_radius_u8": decision.old_radius,
            "new_radius_u8": decision.new_radius,
            "rho": decision.rho,
            "update": decision.update,
            "negative_rho_hard_shrink": decision.update == "HARD_SHRINK_NEGATIVE_RHO",
        },
        "parallel_tempering": tempering_receipt,
        "plateau": plateau,
        "comparison": {
            "same_pairs": list(pair_ids),
            "same_lifted_dof_count": int(current.size),
            "solve_exact_calls": len(solve_rows),
            "j2_exact_calls": len(descent_rows),
            "solve_best_objective_improvement": selection.objective_improvement,
            "j2_best_objective_improvement": descent_best.objective_improvement,
            "scope": "matched one-step form control at each solve realization point; j2 control does not mutate solve state",
        },
        "resume": {
            "complete_iteration_checkpoint": _portable(path),
            "operator_preserved": True,
            "all_candidate_archives_preserved": True,
            "next_iteration_loads_selected_state": True,
        },
        "screening_only": True,
        "g3_top24_full_pearson_r": G3_CORRELATION,
        "evidence_axis": EVIDENCE_AXIS,
        "score_claim": False,
    }
    _write_json(path, receipt)
    return receipt


def _run_n600(
    *,
    config: DDMV17IterativeRealizedTrustRegionConfigV1,
    root: Path,
    problem: Mapping[str, Any],
    iterations: Sequence[Mapping[str, Any]],
    n600_cfg: Any,
    n600_archive: bytes,
    labels_all: np.ndarray,
    poses_all: np.ndarray,
    gt_f0: np.ndarray,
    gt_f1: np.ndarray,
    segnet: Any,
    posenet: Any,
) -> dict[str, Any]:
    path = root / "stage_checkpoints" / "n600.json"
    if path.exists():
        return _json(path)
    state = iterations[-1]["selected_state"]
    candidate, nested_base, program = _ladder_archive(
        state=state,
        problem=problem,
        v14_archive=_base_v14_bytes(n600_cfg),
        source_start=0,
        source_stop=600,
    )
    archive_path = root / "ddm_v17_n600.not_a_candidate.zip.receipt-bytes"
    _publish_immutable(archive_path, candidate)
    measurement = _measure_window(
        name="n600",
        archive=candidate,
        baseline_archive=n600_archive,
        source_pair_ids=tuple(range(600)),
        local_pair_ids=tuple(range(600)),
        root=root,
        config_hash=config.typed_config_hash(),
        batch_size=config.scorer_batch_size,
        labels_all=labels_all,
        poses_all=poses_all,
        gt_f0=gt_f0,
        gt_f1=gt_f1,
        segnet=segnet,
        posenet=posenet,
    )
    receipt = {
        "schema": N600_SCHEMA,
        "archive": {"path": _portable(archive_path), "bytes": len(candidate), "sha256": _sha256(candidate)},
        "measurement": measurement,
        "exact_byte_accounting": {
            "program_bytes": len(encode_coupled_margin_program(program)),
            "placement_records": len(program.placements),
            "sparse_compensation_records": len(program.compensations),
            "nested_base_bytes": len(nested_base),
            "archive_vs_v15": _byte_diff(n600_archive, candidate),
        },
        "authority": "n600 macOS-CPU frozen-scorer advisory; not contest score",
        "score_claim": False,
        "evidence_axis": EVIDENCE_AXIS,
    }
    _write_json(path, receipt)
    return receipt


def _final_receipt(
    *,
    config: DDMV17IterativeRealizedTrustRegionConfigV1,
    root: Path,
    semantic_argv: Sequence[str],
    storage: Mapping[str, Any],
    target_custody: Mapping[str, Any],
    screening_custody: Mapping[str, Any],
    iterations: Sequence[Mapping[str, Any]],
    n600: Mapping[str, Any],
) -> Path:
    path = root / "ddm_v17_iterative_realized_trust_region_receipt.json"
    if path.exists():
        return path
    raw_curve = [
        {
            "iteration": int(row["iteration"]),
            "candidate_id": candidate["label"],
            "lattice_quanta": candidate["lattice_quanta"],
            "predicted_reduction": candidate["predicted_reduction"],
            "realized_reduction": candidate["realized_reduction"],
            "rho": candidate["rho"],
        }
        for row in iterations
        for candidate in row["solve_candidates"]
    ]
    curve = summarize_validity_curve(raw_curve)
    n600_measurement = n600["measurement"]
    accepted = sum(bool(row["accepted"]) for row in iterations)
    solve_gain = sum(float(row["comparison"]["solve_best_objective_improvement"]) for row in iterations)
    descent_gain = sum(float(row["comparison"]["j2_best_objective_improvement"]) for row in iterations)
    n600_delta_dseg = float(n600_measurement["d_seg"]) - float(V14_DSEG)
    n600_delta_bytes = int(n600_measurement["archive_bytes"]) - 133_941
    if n600_delta_dseg < 0.0 and accepted:
        verdict = "MEASURED_ADVISORY_REALIZED_DESCENT_SOLVE_LINE_LIVE"
        scope = "INSTANCE x g3-top24-driven v16 lifted template/sparse DOF x n600 advisory receiver"
    else:
        verdict = "MEASURED_ADVISORY_ITERATIVE_FORMULATION_PLATEAU_FAMILY_OPEN"
        scope = "FORMULATION x v16 lifted template/sparse DOF x measured trust ladder; direct-description family open"
    producer_paths = (
        REPO_ROOT / "tools/measure_ddm_v17_iterative_realized_trust_region.py",
        REPO_ROOT / "src/tac/optimization/iterative_realized_trust_region.py",
        REPO_ROOT / "src/tac/optimization/coupled_margin_levelset.py",
        REPO_ROOT / "src/tac/optimization/direct_description_coupled_margin.py",
    )
    receipt = {
        "schema": RESULT_SCHEMA,
        "lane_id": LANE_ID,
        "run_id": config.run_id,
        "verdict": verdict,
        "verdict_scope": scope,
        "typed_config": config.model_dump(mode="json", by_alias=True),
        "typed_config_sha256": config.typed_config_hash(),
        "semantic_argv": list(semantic_argv),
        "producer_custody": [
            {
                "path": _portable(producer),
                "bytes": producer.stat().st_size,
                "sha256": _sha256(_read_regular_file_once(producer)),
            }
            for producer in producer_paths
        ],
        "iterations": list(iterations),
        "iterations_to_plateau_or_cap": len(iterations),
        "accepted_iterations": accepted,
        "validity_radius": {
            "rho_definition": "Fisher-margin debt realized reduction divided by affine M predicted reduction",
            "raw_rows": raw_curve,
            "curve_by_lattice_quanta": list(curve),
            "stable_law_registration_eligible": len(curve) >= 2 and sum(int(row["rho_count"]) for row in curve) >= 4,
        },
        "n600": n600,
        "deltas_vs_v15": {
            "d_seg": n600_delta_dseg,
            "archive_bytes": n600_delta_bytes,
            "advisory_only": True,
        },
        "solve_vs_j2_descent": {
            "matched_screening_pair_ids": list(config.representative_source_pair_ids),
            "solve_exact_calls": sum(int(row["comparison"]["solve_exact_calls"]) for row in iterations),
            "j2_exact_calls": sum(int(row["comparison"]["j2_exact_calls"]) for row in iterations),
            "solve_summed_best_realized_objective_improvement": solve_gain,
            "j2_summed_best_realized_objective_improvement": descent_gain,
            "winner": "solve" if solve_gain > descent_gain else "j2_descent" if descent_gain > solve_gain else "tie",
            "scope": "matched one-step form controls at each accepted solve realization point",
        },
        "screening_custody": dict(screening_custody),
        "target_custody": dict(target_custody),
        "storage_preflight": dict(storage),
        "resume": {
            "per_iteration_checkpoints": True,
            "per_n600_batch_checkpoints": True,
            "all_candidates_preserved": True,
            "all_preserved": True,
        },
        "triality": {
            "dsl": "DDMV17IterativeRealizedTrustRegionConfigV1 + v16 CoupledMarginProgramV1",
            "dag": ".omx/research/ddm_v17_iterative_realized_trust_region_DAG_FEED_20260723.json",
            "equations": "tac.canonical_equations.ddm_v17_validity_radius_law_20260723",
        },
        "stores_consulted": [
            "CLAUDE.md",
            "AGENTS.md",
            "PROGRAM.md",
            "docs/operating_manual_craft_handoff.md",
            ".omx/research/t5_crucible/SPEC_v75_optimal_single_trunk_20260708.md",
            ".omx/research/SPEC_v8_perclass_decomposition_20260708.md",
            config.v16_config_path,
            config.hard_pair_registry_path,
            ".omx/research/erm_2607_10128_crosswalk_20260720T154953Z.md",
            ".omx/research/latticesieve_specialq_crosswalk_20260720T182404Z.md",
            ".omx/research/g2f_bidirectional_amplitude_ladder_chart_level_20260721T153318Z.md",
            ".omx/state/canonical_equations_registry.jsonl",
            "reports/latest.md",
        ],
        "pointer": f"{POINTER_SCORE_TEXT} [contest-CPU]",
        "pointer_moved": False,
        "evidence_axis": EVIDENCE_AXIS,
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
        "d_seg_claim": False,
        "d_pose_claim": False,
        "promotion_eligible": False,
        "main_landing_review_required": True,
    }
    _write_json(path, receipt)
    return path


def run(
    config: DDMV17IterativeRealizedTrustRegionConfigV1,
    output_directory: Path,
    semantic_argv: Sequence[str],
) -> Path | None:
    root = output_directory.resolve()
    storage = _storage_preflight(root)
    root.mkdir(parents=True, exist_ok=True)
    v16_config = _bound_config(config)
    screening = _validate_screening_registry(config)
    target = _target_custody(v16_config, root)
    n64_receipt, _n64_cfg, _n64_archive, _n600_receipt, n600_cfg, n600_archive = _v15_bindings(v16_config)
    bank = receive_carrier_compose_archive(n600_archive).scorer_solved_templates
    if bank is None:
        raise DirectDescriptionError("v17 lost the bound v15 template bank")
    _set_state_bank(bank)
    cache = Path(v16_config.target_cache_path)
    labels_all = open_stored_npy_memmap(cache, "lstars")
    poses_all = open_stored_npy_memmap(cache, "gt_poses")
    gt_f0 = open_stored_npy_memmap(cache, "gt_f0")
    gt_f1 = open_stored_npy_memmap(cache, "gt_f1")
    segnet, posenet, _scorer_custody = _load_models(n600_cfg)
    make_scorers_differentiable(posenet, segnet)
    problem = _build_problem(
        config=config,
        root=root,
        n64_receipt=n64_receipt,
        n600_cfg=n600_cfg,
        n600_archive=n600_archive,
        labels_all=labels_all,
        segnet=segnet,
        posenet=posenet,
    )
    if problem.get("typed_config_sha256") != config.typed_config_hash():
        raise DirectDescriptionError("v17 preserved problem typed-config identity differs")
    iterations = [_json(path) for path in sorted((root / "stage_checkpoints").glob("iteration_*.json"))]
    terminated = bool(iterations and (bool(iterations[-1]["plateau"]) or len(iterations) >= config.maximum_iterations))
    if not terminated:
        row = _run_iteration(
            config=config,
            root=root,
            problem=problem,
            iterations=iterations,
            n600_cfg=n600_cfg,
            labels_all=labels_all,
            poses_all=poses_all,
            segnet=segnet,
            posenet=posenet,
        )
        print(json.dumps({"complete": False, "stage": f"iteration_{row['iteration']:02d}"}))
        return None
    n600_path = root / "stage_checkpoints" / "n600.json"
    if not n600_path.exists():
        row = _run_n600(
            config=config,
            root=root,
            problem=problem,
            iterations=iterations,
            n600_cfg=n600_cfg,
            n600_archive=n600_archive,
            labels_all=labels_all,
            poses_all=poses_all,
            gt_f0=gt_f0,
            gt_f1=gt_f1,
            segnet=segnet,
            posenet=posenet,
        )
        print(json.dumps({"complete": False, "stage": "n600", "d_seg": row["measurement"]["d_seg"]}))
        return None
    receipt = _final_receipt(
        config=config,
        root=root,
        semantic_argv=semantic_argv,
        storage=storage,
        target_custody=target,
        screening_custody=screening,
        iterations=iterations,
        n600=_json(n600_path),
    )
    print(json.dumps({"complete": True, "receipt": _portable(receipt)}))
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    config = DDMV17IterativeRealizedTrustRegionConfigV1.model_validate_json(_read_regular_file_once(args.config))
    semantic_argv = (
        "tools/measure_ddm_v17_iterative_realized_trust_region.py",
        "--config",
        _portable(args.config),
        "--output-directory",
        _portable(args.output_directory),
    )
    run(config, args.output_directory, semantic_argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
