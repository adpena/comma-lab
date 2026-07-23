#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Probe A: bounded-collateral contextual realized trust-region solve.

This is the operator-supplemented DDM v17 execution surface.  One invocation
advances one immutable stage.  Proposal models operate only on integer basis
coordinates.  The exact counted receiver and frozen scorers are the sole
acceptance authority; the local Fisher-margin model can never admit a step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "src", REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap  # noqa: E402
from tac.optimization.coupled_margin_levelset import CouplingOperator, predicted_margin  # noqa: E402
from tac.optimization.direct_description_carrier_compose import (  # noqa: E402
    receive_carrier_compose_archive,
    rfc8785_canonicalize,
)
from tac.optimization.direct_description_coupled_margin import (  # noqa: E402
    encode_coupled_margin_program,
    receive_coupled_margin_archive,
)
from tac.optimization.direct_description_minimizer import (  # noqa: E402
    DirectDescriptionError,
    _read_regular_file_once,
)
from tac.optimization.iterative_realized_trust_region import (  # noqa: E402
    BasisProjection,
    HardCandidate,
    TemplateBasis,
    TrustRegionPolicy,
    build_template_basis_projection,
    fisher_margin_debt,
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
    SOURCE_BYTES,
    TARGET_ROLES,
    DDMV16CoupledJointSolveConfigV1,
    _archive_for_state,
    _assemble_operator,
    _base_v14_bytes,
    _build_problem,
    _byte_diff,
    _constraint_summary,
    _hard_margin_vector,
    _json,
    _ladder_archive,
    _measure_window,
    _portable,
    _required_for_rows,
    _set_state_bank,
    _sha256,
    _sha256_array,
    _target_custody,
    _torch_forward_full,
    _v15_bindings,
    _write_json,
    _write_npz,
)
from tools.measure_ddm_v17_iterative_realized_trust_region import _proposal_families  # noqa: E402

SCHEMA = "ddm_a1_bounded_collateral_realized_receipt.v1"
ITERATION_SCHEMA = "ddm_a1_bounded_collateral_realized_iteration.v1"
LANE_ID = "ddm_v17_iterative_realized_trust_region_solve"
AUDIT_MEMO = ".omx/research/ddm_a1_naive_verdict_audit_20260723_codex.md"
AUDIT_MAIN_COMMIT = "fecbefe5a5"
ROLE_DEBT_CEILING_PERCENT = 23.404922
FIXED_PAINT_MOVABLE_GAP_REMAINING_PERCENT = 60.561878
_CANDIDATE_SCALES = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)


class DDMA1BoundedCollateralRealizedConfigV1(BaseModel):
    """Exact Probe-A preregistration plus derived read-only adapter properties."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    schema_: Literal["DDMA1BoundedCollateralRealizedConfigV1"] = Field(
        default="DDMA1BoundedCollateralRealizedConfigV1",
        alias="schema",
        serialization_alias="schema",
    )
    run_id: str = Field(min_length=8)
    seed: Literal[1234] = 1234
    source_v15_receipt: str = Field(min_length=1)
    source_v15_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_v16_receipt: str = Field(min_length=1)
    source_v16_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_cache_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    pair_ids: tuple[StrictInt, ...]
    template_bases: tuple[Literal["1x1_rowband_control", "2x2_contextual", "boundary_normal_2x2"], ...]
    epsilon_harmful_off_target_flips: tuple[StrictInt, ...]
    lattice_radii_u8: tuple[StrictInt, ...]
    maximum_realized_iterations: Literal[8] = 8
    relinearize_after_each_accepted_step: Literal[True] = True
    acceptance_authority: Literal["exact_realized_joint_objective_after_uint8_R"]
    pose_term_in_objective: Literal[True] = True
    maximum_archive_bytes: Literal[160000] = 160000
    research_only: Literal[True] = True
    execution_allowed: Literal[False] = False
    score_claim: Literal[False] = False

    @model_validator(mode="after")
    def _fixed_preregistration(self) -> DDMA1BoundedCollateralRealizedConfigV1:
        if self.pair_ids != (447, 53, 416, 296, 547, 278, 501, 346):
            raise ValueError("Probe A pair_ids differ from the preregistration")
        if self.template_bases != (
            "1x1_rowband_control",
            "2x2_contextual",
            "boundary_normal_2x2",
        ):
            raise ValueError("Probe A template_bases differ from the preregistration")
        if self.epsilon_harmful_off_target_flips != (0, 16, 32, 64):
            raise ValueError("Probe A collateral ladder differs from the preregistration")
        if self.lattice_radii_u8 != (1, 2, 4, 8):
            raise ValueError("Probe A lattice radii differ from the preregistration")
        return self

    def typed_config_hash(self) -> str:
        return _sha256(rfc8785_canonicalize(self.model_dump(mode="json", by_alias=True)))

    # Existing v16 helpers are pure duck-typed consumers of these immutable
    # apparatus constants.  They are not extra DSL fields or invented flags.
    @property
    def representative_source_pair_ids(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.pair_ids)

    @property
    def maximum_target_cells_per_role_pair(self) -> int:
        return 2

    @property
    def maximum_protected_cells_per_pair(self) -> int:
        return 4

    @property
    def maximum_fd_entries(self) -> int:
        return 8

    @property
    def vjp_chunk_rows(self) -> int:
        return 8

    @property
    def target_margin_epsilon(self) -> float:
        return 0.05

    @property
    def protected_margin_epsilon(self) -> float:
        return 0.0

    @property
    def pose_trust_radius(self) -> float:
        return 0.05

    @property
    def scorer_threads(self) -> int:
        return 4

    @property
    def scorer_batch_size(self) -> int:
        return 16

    @property
    def archive_box_bytes(self) -> int:
        return self.maximum_archive_bytes

    @property
    def maximum_candidates(self) -> int:
        return len(self.lattice_radii_u8)

    @property
    def candidate_scales(self) -> tuple[float, ...]:
        return _CANDIDATE_SCALES

    @property
    def g2f_amplitude_ladder(self) -> tuple[float, ...]:
        return tuple(float(value) for value in self.lattice_radii_u8)

    @property
    def minimum_trust_radius_u8(self) -> float:
        return float(min(self.lattice_radii_u8))

    @property
    def maximum_trust_radius_u8(self) -> float:
        return float(max(self.lattice_radii_u8))


def _bound_json(path: Path, expected_sha256: str, name: str) -> dict[str, Any]:
    payload = _read_regular_file_once(path)
    if _sha256(payload) != expected_sha256:
        raise DirectDescriptionError(f"Probe A {name} SHA-256 differs")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise DirectDescriptionError(f"Probe A {name} is not a JSON object")
    return value


def _bindings(
    config: DDMA1BoundedCollateralRealizedConfigV1,
) -> tuple[dict[str, Any], dict[str, Any], DDMV16CoupledJointSolveConfigV1]:
    v15 = _bound_json(REPO_ROOT / config.source_v15_receipt, config.source_v15_receipt_sha256, "v15 receipt")
    v16 = _bound_json(REPO_ROOT / config.source_v16_receipt, config.source_v16_receipt_sha256, "v16 receipt")
    v16_config = DDMV16CoupledJointSolveConfigV1.model_validate(v16["typed_config"])
    if v16_config.n64_receipt_path != config.source_v15_receipt:
        raise DirectDescriptionError("Probe A source v15 receipt differs from v16 custody")
    if v16_config.target_cache_sha256 != config.target_cache_sha256:
        raise DirectDescriptionError("Probe A target-cache SHA differs from v16 custody")
    if tuple(v16_config.representative_source_pair_ids) != config.pair_ids:
        raise DirectDescriptionError("Probe A pair ids differ from v16 custody")
    return v15, v16, v16_config


def _boundary_axes(
    current_base: bytes,
    pair_ids: tuple[int, ...],
) -> tuple[tuple[str, ...], list[dict[str, Any]]]:
    receiver = receive_carrier_compose_archive(current_base)
    bank = receiver.scorer_solved_templates
    if bank is None:
        raise DirectDescriptionError("Probe A boundary basis lost the solved-template bank")
    axes = []
    receipt = []
    for template in range(len(bank.templates)):
        masks = receiver.template_camera_masks(pair_ids, bank.templates[template])
        energy_x = 0.0
        energy_y = 0.0
        for mask in masks:
            value = np.asarray(mask, dtype=np.float64)
            energy_x += float(np.abs(np.diff(value, axis=1)).sum())
            energy_y += float(np.abs(np.diff(value, axis=0)).sum())
        axis = "x" if energy_x >= energy_y else "y"
        axes.append(axis)
        receipt.append(
            {
                "template_index": template,
                "normal_axis": axis,
                "mask_boundary_energy_x": energy_x,
                "mask_boundary_energy_y": energy_y,
                "authority": "MEASURED_FROM_BOUND_TEMPLATE_MASKS",
            }
        )
    return tuple(axes), receipt


def _project_operator(operator: CouplingOperator, projection: BasisProjection) -> CouplingOperator:
    # NumPy/Accelerate on this host can leak stale floating-status flags from
    # prior Torch/BLAS calls.  The constructor below still rejects any actual
    # nonfinite result, so suppressing those false warnings does not weaken the
    # numerical gate.
    with np.errstate(all="ignore"):
        matrix = operator.matrix @ projection.matrix
    return CouplingOperator(
        matrix=matrix,
        margin=operator.margin,
        required_margin=operator.required_margin,
        targeted_count=operator.targeted_count,
        row_labels=operator.row_labels,
        dof_labels=projection.dof_labels,
        activation_pattern_sha256=hashlib.sha256(
            (operator.activation_pattern_sha256 + projection.basis + ":" + ",".join(projection.boundary_axes)).encode()
        ).hexdigest(),
    )


def _latent_bounds(projection: BasisProjection) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.concatenate(
            (
                np.zeros(projection.template_latent_count, dtype=np.int64),
                np.full(projection.current.size - projection.template_latent_count, -127, dtype=np.int64),
            )
        ),
        np.concatenate(
            (
                np.full(projection.template_latent_count, 255, dtype=np.int64),
                np.full(projection.current.size - projection.template_latent_count, 127, dtype=np.int64),
            )
        ),
    )


def _hard_measurement(
    *,
    archive: bytes,
    pair_ids: tuple[int, ...],
    rows: list[dict[str, Any]],
    required: np.ndarray,
    labels_all: np.ndarray,
    poses_all: np.ndarray,
    segnet: Any,
    posenet: Any,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    receiver = receive_coupled_margin_archive(archive)
    camera = receiver.render_camera_pairs(pair_ids)
    logits, cells, pose6 = _torch_forward_full(segnet, posenet, camera)
    indexes = np.asarray(pair_ids, dtype=np.int64)
    labels = np.asarray(labels_all[indexes])
    poses = np.asarray(poses_all[indexes])
    errors = cells != labels
    margins = _hard_margin_vector(rows, pair_ids, logits, pose6)
    d_seg = float(np.mean(errors))
    d_pose = float(np.mean(np.square(pose6 - poses), dtype=np.float64))
    per_role = {}
    for role, role_id in TARGET_ROLES.items():
        mask = labels == role_id
        count = int(np.count_nonzero(mask))
        role_errors = int(np.count_nonzero(errors & mask))
        per_role[role] = {
            "errors": role_errors,
            "sites": count,
            "d_seg": f"{role_errors / max(1, count):.12f}",
        }
    objective = 100.0 * d_seg + math.sqrt(10.0 * d_pose) + 25.0 * len(archive) / SOURCE_BYTES
    return (
        {
            "archive_bytes": len(archive),
            "archive_sha256": _sha256(archive),
            "errors": int(np.count_nonzero(errors)),
            "sites": int(errors.size),
            "d_seg": f"{d_seg:.12f}",
            "d_pose": f"{d_pose:.12f}",
            "advisory_score_formula_value": f"{objective:.12f}",
            "per_role": per_role,
            "constraints": _constraint_summary(rows, margins, required),
            "cells_sha256": _sha256_array(cells),
            "pose6_sha256": _sha256_array(pose6),
            "camera_sha256": _sha256_array(camera),
            "score_claim": False,
            "evidence_axis": EVIDENCE_AXIS,
        },
        margins,
        cells,
        camera,
    )


def _state(
    iterations: list[dict[str, Any]], problem: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray, dict[str, list[int]], str | None, int | None, float]:
    if not iterations:
        return (
            np.asarray(problem["initial_template_values_u8"], dtype=np.int16),
            np.asarray(problem["initial_compensation_rgb_i8"], dtype=np.int16),
            {key: list(value) for key, value in problem["initial_phases"].items()},
            None,
            None,
            1.0,
        )
    selected = iterations[-1]["selected_state"]
    return (
        np.asarray(selected["template_values_u8"], dtype=np.int16),
        np.asarray(selected["compensation_rgb_i8"], dtype=np.int16),
        {key: list(value) for key, value in selected["phases"].items()},
        str(iterations[-1]["active_basis"]),
        int(iterations[-1]["active_epsilon_harmful_off_target_flips"]),
        float(iterations[-1]["trust"]["new_radius_u8"]),
    )


def _materialize(
    *,
    label: str,
    basis: str,
    candidate: Any,
    projection: BasisProjection,
    operator: CouplingOperator,
    config: DDMA1BoundedCollateralRealizedConfigV1,
    problem: dict[str, Any],
    values: np.ndarray,
    compensation: np.ndarray,
    phases: dict[str, list[int]],
    v14_archive: bytes,
    rows: list[dict[str, Any]],
    baseline_measurement: dict[str, Any],
    baseline_model_debt: float,
    baseline_cells: np.ndarray,
    labels_all: np.ndarray,
    poses_all: np.ndarray,
    segnet: Any,
    posenet: Any,
    root: Path,
    iteration: int,
) -> dict[str, Any]:
    latent_step = np.asarray(candidate.integer_step, dtype=np.int64)
    physical_step = projection.lift_step(latent_step)
    full_current = np.concatenate((values.reshape(-1), compensation.reshape(-1))).astype(np.int64)
    full_candidate = full_current + physical_step
    patch_size = int(values.size)
    new_values = full_candidate[:patch_size].reshape(values.shape).astype(np.uint8)
    new_compensation = full_candidate[patch_size:].reshape(compensation.shape).astype(np.int16)
    pair_ids = config.representative_source_pair_ids
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
    measurement, realized_margin, cells, _camera = _hard_measurement(
        archive=archive,
        pair_ids=pair_ids,
        rows=rows,
        required=required,
        labels_all=labels_all,
        poses_all=poses_all,
        segnet=segnet,
        posenet=posenet,
    )
    labels = np.asarray(labels_all[np.asarray(pair_ids, dtype=np.int64)])
    baseline_correct = baseline_cells == labels
    candidate_correct = cells == labels
    harmful = int(np.count_nonzero(baseline_correct & ~candidate_correct))
    helpful = int(np.count_nonzero(~baseline_correct & candidate_correct))
    admissible_epsilons = [
        int(epsilon)
        for epsilon in config.epsilon_harmful_off_target_flips
        if harmful <= int(epsilon) and len(archive) <= config.maximum_archive_bytes
    ]
    predicted = predicted_margin(operator, latent_step.astype(np.float64))
    predicted_reduction = baseline_model_debt - fisher_margin_debt(predicted, required)
    realized_reduction = baseline_model_debt - fisher_margin_debt(realized_margin, required)
    rho = realized_reduction / predicted_reduction if predicted_reduction > 0.0 else None
    objective = float(measurement["advisory_score_formula_value"])
    baseline_objective = float(baseline_measurement["advisory_score_formula_value"])
    return {
        "label": label,
        "basis": basis,
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
        "latent_integer_step": latent_step.tolist(),
        "physical_integer_step": physical_step.tolist(),
        "latent_dof_count": int(latent_step.size),
        "physical_dof_count": int(physical_step.size),
        "integer_step_l2": float(np.linalg.norm(latent_step)),
        "lattice_quanta": int(np.max(np.abs(latent_step), initial=0)),
        "scale": candidate.scale,
        "quadratic_error": candidate.quadratic_error,
        "covering_bound": candidate.covering_bound,
        "predicted_reduction": predicted_reduction,
        "realized_reduction": realized_reduction,
        "rho": None if rho is None or not math.isfinite(rho) else rho,
        "hard_objective_delta": objective - baseline_objective,
        "d_seg_delta": float(measurement["d_seg"]) - float(baseline_measurement["d_seg"]),
        "archive_byte_delta": int(measurement["archive_bytes"]) - int(baseline_measurement["archive_bytes"]),
        "harmful_off_target_flips": harmful,
        "helpful_flips": helpful,
        "harmful_definition": "baseline-correct scorer cells made incorrect after exact uint8/R replay",
        "admissible_epsilons": admissible_epsilons,
        "exact_receiver_realized": True,
        "pose_term_in_objective": True,
        "score_claim": False,
    }


def _basis_projection(
    basis: str,
    values: np.ndarray,
    compensation: np.ndarray,
    boundary_axes: tuple[str, ...],
) -> BasisProjection:
    return build_template_basis_projection(
        values,
        compensation,
        basis=TemplateBasis(basis),
        boundary_axes=boundary_axes if basis == TemplateBasis.BOUNDARY_NORMAL_2X2 else (),
    )


def _hard_candidate(row: dict[str, Any]) -> HardCandidate:
    minimum_epsilon = min(row["admissible_epsilons"], default=None)
    return HardCandidate(
        candidate_id=row["label"],
        hard_objective=float(row["measurement"]["advisory_score_formula_value"]),
        d_seg=float(row["measurement"]["d_seg"]),
        archive_bytes=int(row["measurement"]["archive_bytes"]),
        admissible=minimum_epsilon is not None,
        predicted_reduction=float(row["predicted_reduction"]),
        realized_model_reduction=float(row["realized_reduction"]),
        integer_step=np.asarray(row["latent_integer_step"], dtype=np.int64),
    )


def _selected_row(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    return next(dict(row) for row in rows if row["label"] == label)


def _run_iteration(
    *,
    config: DDMA1BoundedCollateralRealizedConfigV1,
    root: Path,
    problem: dict[str, Any],
    iterations: list[dict[str, Any]],
    n600_config: Any,
    labels_all: np.ndarray,
    poses_all: np.ndarray,
    segnet: Any,
    posenet: Any,
) -> dict[str, Any]:
    iteration = len(iterations) + 1
    path = root / "stage_checkpoints" / f"iteration_{iteration:02d}.json"
    if path.exists():
        return _json(path)
    values, compensation, phases, active_basis, active_epsilon, radius = _state(iterations, problem)
    v14_archive = _base_v14_bytes(n600_config)
    full_operator, rows, clusters, full_current, current_camera = _assemble_operator(
        config=config,
        problem=problem,
        v14_archive=v14_archive,
        values=values,
        compensation=compensation,
        phases=phases,
        segnet=segnet,
        posenet=posenet,
    )
    operator_path = root / "operators" / f"iteration_{iteration:02d}_M_full.npz"
    _write_npz(
        operator_path,
        M=full_operator.matrix,
        margin=full_operator.margin,
        required_margin=full_operator.required_margin,
        parameters=full_current,
    )
    pair_ids = config.representative_source_pair_ids
    baseline_archive, current_base, _program = _archive_for_state(
        v14_archive,
        values.astype(np.uint8),
        pair_ids,
        phases,
        problem["sparse_compensation_support"],
        compensation,
    )
    required = _required_for_rows(config, rows)
    baseline_measurement, baseline_margin, baseline_cells, _camera = _hard_measurement(
        archive=baseline_archive,
        pair_ids=pair_ids,
        rows=rows,
        required=required,
        labels_all=labels_all,
        poses_all=poses_all,
        segnet=segnet,
        posenet=posenet,
    )
    if _sha256_array(current_camera) != baseline_measurement["camera_sha256"]:
        raise DirectDescriptionError("Probe A linearization origin differs from hard baseline")
    baseline_model_debt = fisher_margin_debt(baseline_margin, required)
    boundary_axes, boundary_receipt = _boundary_axes(current_base, pair_ids)
    bases = (active_basis,) if active_basis is not None else config.template_bases
    solve_rows: list[dict[str, Any]] = []
    descent_rows: list[dict[str, Any]] = []
    model_receipts = []
    for basis in bases:
        projection = _basis_projection(basis, values, compensation, boundary_axes)
        operator = _project_operator(full_operator, projection)
        lower, upper = _latent_bounds(projection)
        with np.errstate(all="ignore"):
            solve_candidates, model_receipt, descent_candidates = _proposal_families(
                config=config,
                operator=operator,
                rows=rows,
                current=projection.current,
                lower=lower,
                upper=upper,
                radius=radius,
                iteration=iteration,
            )
        expected = config.maximum_candidates
        if len(solve_candidates) != expected or len(descent_candidates) != expected:
            raise DirectDescriptionError("Probe A solve/control exact-call budget is not the preregistered K=4")
        model_receipts.append(
            {
                "basis": basis,
                "latent_dof_count": int(projection.current.size),
                "physical_dof_count": int(projection.matrix.shape[0]),
                "projection_sha256": _sha256_array(projection.matrix),
                **model_receipt,
            }
        )
        for index, (source, candidate) in enumerate(solve_candidates):
            solve_rows.append(
                _materialize(
                    label=f"{basis}_solve_{index:02d}_{source}",
                    basis=basis,
                    candidate=candidate,
                    projection=projection,
                    operator=operator,
                    config=config,
                    problem=problem,
                    values=values,
                    compensation=compensation,
                    phases=phases,
                    v14_archive=v14_archive,
                    rows=rows,
                    baseline_measurement=baseline_measurement,
                    baseline_model_debt=baseline_model_debt,
                    baseline_cells=baseline_cells,
                    labels_all=labels_all,
                    poses_all=poses_all,
                    segnet=segnet,
                    posenet=posenet,
                    root=root,
                    iteration=iteration,
                )
            )
        for index, candidate in enumerate(descent_candidates):
            descent_rows.append(
                _materialize(
                    label=f"{basis}_model_disabled_j2_{index:02d}",
                    basis=basis,
                    candidate=candidate,
                    projection=projection,
                    operator=operator,
                    config=config,
                    problem=problem,
                    values=values,
                    compensation=compensation,
                    phases=phases,
                    v14_archive=v14_archive,
                    rows=rows,
                    baseline_measurement=baseline_measurement,
                    baseline_model_debt=baseline_model_debt,
                    baseline_cells=baseline_cells,
                    labels_all=labels_all,
                    poses_all=poses_all,
                    segnet=segnet,
                    posenet=posenet,
                    root=root,
                    iteration=iteration,
                )
            )
    if len(solve_rows) != len(descent_rows):
        raise DirectDescriptionError("Probe A solve and model-disabled controls used unequal exact-call budgets")
    baseline_objective = float(baseline_measurement["advisory_score_formula_value"])
    selection = select_realized_improvement(baseline_objective, tuple(_hard_candidate(row) for row in solve_rows))
    selected = None if selection.selected is None else _selected_row(solve_rows, selection.selected.candidate_id)
    descent_selection = select_realized_improvement(
        baseline_objective, tuple(_hard_candidate(row) for row in descent_rows)
    )
    if selected is None:
        selected_state = {
            "template_values_u8": values.astype(np.uint8).tolist(),
            "compensation_rgb_i8": compensation.tolist(),
            "phases": {key: list(value) for key, value in phases.items()},
        }
        selected_basis = active_basis or "hold_control"
        selected_epsilon = active_epsilon if active_epsilon is not None else 0
        selected_measurement = baseline_measurement
        selected_archive = {"path": None, "bytes": len(baseline_archive), "sha256": _sha256(baseline_archive)}
        selected_radius = radius
    else:
        selected_state = selected["state"]
        selected_basis = selected["basis"]
        selected_epsilon = min(int(value) for value in selected["admissible_epsilons"])
        selected_measurement = selected["measurement"]
        selected_archive = selected["archive"]
        selected_radius = float(selected["lattice_quanta"])
    trust_rho = selection.rho
    if selected is None:
        modeled = [row for row in solve_rows if float(row["predicted_reduction"]) > 0.0]
        if modeled:
            trial = max(modeled, key=lambda row: float(row["predicted_reduction"]))
            trust_rho = float(trial["realized_reduction"]) / float(trial["predicted_reduction"])
    decision = update_trust_radius(
        selected_radius,
        rho=trust_rho,
        accepted=selected is not None,
        policy=TrustRegionPolicy(minimum_radius=1.0, maximum_radius=8.0),
    )
    contextual_or_priced = bool(
        selected is not None and (selected_basis != TemplateBasis.ROWBAND_1X1_CONTROL or selected_epsilon > 0)
    )
    receipt = {
        "schema": ITERATION_SCHEMA,
        "run_id": config.run_id,
        "lane_id": LANE_ID,
        "iteration": iteration,
        "typed_config_sha256": config.typed_config_hash(),
        "linearization": {
            "operator_path": _portable(operator_path),
            "operator_sha256": _sha256(_read_regular_file_once(operator_path)),
            "shape": list(full_operator.matrix.shape),
            "origin_archive_sha256": _sha256(baseline_archive),
            "origin_camera_sha256": baseline_measurement["camera_sha256"],
            "fresh_current_realized_point": True,
            "relinearized_after_prior_accept": bool(iterations and iterations[-1]["accepted"]),
            "finite_difference_all_clusters_passed": all(row["finite_difference"]["passed"] for row in clusters),
            "pair_clusters": clusters,
        },
        "boundary_normal_basis": boundary_receipt,
        "model_solves": model_receipts,
        "baseline_measurement": baseline_measurement,
        "baseline_fisher_margin_debt": baseline_model_debt,
        "solve_candidates": solve_rows,
        "model_disabled_j2_candidates": descent_rows,
        "selection": {
            "selected_candidate_id": None if selection.selected is None else selection.selected.candidate_id,
            "accepted": selection.accepted,
            "baseline_objective": selection.baseline_objective,
            "objective_improvement": selection.objective_improvement,
            "rho": selection.rho,
            "evaluated_count": selection.evaluated_count,
            "admissible_count": selection.admissible_count,
            "contextual_or_epsilon_positive": contextual_or_priced,
        },
        "selected_state": selected_state,
        "selected_measurement": selected_measurement,
        "selected_archive": selected_archive,
        "active_basis": selected_basis,
        "active_epsilon_harmful_off_target_flips": selected_epsilon,
        "accepted": selected is not None,
        "trust": {
            "old_radius_u8": decision.old_radius,
            "new_radius_u8": decision.new_radius,
            "rho": decision.rho,
            "update": decision.update,
            "negative_rho_hard_shrink": decision.update == "HARD_SHRINK_NEGATIVE_RHO",
        },
        "plateau": selected is None,
        "comparison": {
            "same_pairs": list(pair_ids),
            "solve_exact_calls": len(solve_rows),
            "model_disabled_j2_exact_calls": len(descent_rows),
            "same_budget": len(solve_rows) == len(descent_rows),
            "solve_best_objective_improvement": selection.objective_improvement,
            "model_disabled_j2_best_objective_improvement": descent_selection.objective_improvement,
            "control_definition": "QP/KKT disabled; clipped first-order direction plus ranked-prefix sign lattice",
        },
        "collateral_pricing": {
            "epsilon_ladder": list(config.epsilon_harmful_off_target_flips),
            "priced_by": "full exact d_seg term inside 100*d_seg + sqrt(10*d_pose) + 25*bytes/N",
            "hard_cap": "harmful_off_target_flips <= epsilon",
            "v15_extreme_point": "epsilon=0",
        },
        "parallel_tempering": {
            "status": "NOT_TRIGGERED_INITIAL_PREREGISTERED_GRID",
            "scope": "optional escape remains available; no duplicate same-point iteration is claimed",
        },
        "resume": {
            "complete_iteration_checkpoint": _portable(path),
            "operator_preserved": True,
            "all_candidate_archives_preserved": True,
            "next_iteration_loads_selected_state": True,
        },
        "evidence_axis": EVIDENCE_AXIS,
        "score_claim": False,
    }
    _write_json(path, receipt)
    return receipt


def _source_control(v15_receipt: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in v15_receipt["solved_template_ladder"] if row["candidate"] == "v15_solved_templates"]
    if len(rows) != 1:
        raise DirectDescriptionError("Probe A v15 control row is not unique")
    row = rows[0]
    objective = (
        100.0 * float(row["d_seg"])
        + math.sqrt(10.0 * float(row["d_pose"]))
        + 25.0 * int(row["archive_bytes"]) / SOURCE_BYTES
    )
    return {
        "candidate": row["candidate"],
        "archive_bytes": int(row["archive_bytes"]),
        "archive_sha256": row["archive_sha256"],
        "d_seg": row["d_seg"],
        "d_pose": row["d_pose"],
        "advisory_score_formula_value": f"{objective:.12f}",
        "source_receipt_sha256": v15_receipt["producer_custody"][0]["sha256"]
        if v15_receipt.get("producer_custody")
        else None,
    }


def _run_ladder(
    *,
    rung: Literal["n64", "n600"],
    config: DDMA1BoundedCollateralRealizedConfigV1,
    root: Path,
    problem: dict[str, Any],
    final_iteration: dict[str, Any],
    v15_receipt: dict[str, Any],
    v15_n600_receipt: dict[str, Any],
    n64_config: Any,
    n64_archive: bytes,
    n600_config: Any,
    n600_archive: bytes,
    labels_all: np.ndarray,
    poses_all: np.ndarray,
    gt_f0: np.ndarray,
    gt_f1: np.ndarray,
    segnet: Any,
    posenet: Any,
) -> dict[str, Any]:
    path = root / "stage_checkpoints" / f"ladder_{rung}.json"
    if path.exists():
        return _json(path)
    state = final_iteration["selected_state"]
    if rung == "n64":
        candidate, nested_base, program = _ladder_archive(
            state=state,
            problem=problem,
            v14_archive=_base_v14_bytes(n64_config),
            source_start=448,
            source_stop=512,
        )
        source_ids = tuple(range(448, 512))
        local_ids = tuple(range(64))
        baseline = n64_archive
    else:
        candidate, nested_base, program = _ladder_archive(
            state=state,
            problem=problem,
            v14_archive=_base_v14_bytes(n600_config),
            source_start=0,
            source_stop=600,
        )
        source_ids = tuple(range(600))
        local_ids = source_ids
        baseline = n600_archive
    archive_path = root / f"ddm_a1_{rung}.not_a_candidate.zip.receipt-bytes"
    _publish_immutable(archive_path, candidate)
    measurement = _measure_window(
        name=f"ladder_{rung}",
        archive=candidate,
        baseline_archive=baseline,
        source_pair_ids=source_ids,
        local_pair_ids=local_ids,
        root=root,
        config_hash=config.typed_config_hash(),
        batch_size=16,
        labels_all=labels_all,
        poses_all=poses_all,
        gt_f0=gt_f0,
        gt_f1=gt_f1,
        segnet=segnet,
        posenet=posenet,
    )
    control = _source_control(v15_receipt if rung == "n64" else v15_n600_receipt)
    expected_control_sha = _sha256(n64_archive if rung == "n64" else n600_archive)
    if control["archive_sha256"] != expected_control_sha:
        raise DirectDescriptionError(f"Probe A {rung} bound-control archive SHA differs")
    control_objective = float(control["advisory_score_formula_value"])
    objective = float(measurement["advisory_score_formula_value"])
    no_worse = objective <= control_objective + 1e-12
    receipt = {
        "schema": "ddm_a1_bounded_collateral_realized_ladder.v1",
        "rung": rung,
        "archive": {"path": _portable(archive_path), "bytes": len(candidate), "sha256": _sha256(candidate)},
        "measurement": measurement,
        "control": control,
        "no_worse_than_bound_control": no_worse,
        "objective_delta_vs_control": objective - control_objective,
        "exact_byte_accounting": {
            "nested_base_bytes": len(nested_base),
            "program_bytes": len(encode_coupled_margin_program(program)),
            "archive_vs_v15": _byte_diff(baseline, candidate),
        },
        "evidence_axis": EVIDENCE_AXIS,
        "score_claim": False,
    }
    _write_json(path, receipt)
    return receipt


def _final_receipt(
    *,
    config: DDMA1BoundedCollateralRealizedConfigV1,
    root: Path,
    semantic_argv: tuple[str, ...],
    storage: dict[str, Any],
    target_custody: dict[str, Any],
    v15_receipt: dict[str, Any],
    v16_receipt: dict[str, Any],
    iterations: list[dict[str, Any]],
    n64: dict[str, Any] | None,
    n600: dict[str, Any] | None,
) -> Path:
    path = root / "ddm_a1_bounded_collateral_realized_receipt.json"
    if path.exists():
        return path
    raw_curve = [
        {
            "iteration": int(iteration["iteration"]),
            "basis": candidate["basis"],
            "candidate_id": candidate["label"],
            "lattice_quanta": candidate["lattice_quanta"],
            "predicted_reduction": candidate["predicted_reduction"],
            "realized_reduction": candidate["realized_reduction"],
            "rho": candidate["rho"],
            "hard_objective_delta": candidate["hard_objective_delta"],
            "harmful_off_target_flips": candidate["harmful_off_target_flips"],
        }
        for iteration in iterations
        for candidate in iteration["solve_candidates"]
    ]
    curve = summarize_validity_curve(raw_curve)
    accepted = [row for row in iterations if row["accepted"]]
    final_measurement = iterations[-1]["selected_measurement"]
    baseline = iterations[0]["baseline_measurement"]
    solve_gain = sum(float(row["comparison"]["solve_best_objective_improvement"]) for row in iterations)
    descent_gain = sum(float(row["comparison"]["model_disabled_j2_best_objective_improvement"]) for row in iterations)
    probe_admitted = bool(
        accepted
        and any(bool(row["selection"]["contextual_or_epsilon_positive"]) for row in accepted)
        and n64 is not None
        and bool(n64["no_worse_than_bound_control"])
        and n600 is not None
    )
    if probe_admitted:
        verdict = "MEASURED_ADVISORY_PROBE_A_ADMITTED_N600_REPLAYED"
        scope = "INSTANCE x Probe-A basis/collateral/radius contract x macOS-CPU advisory"
    elif accepted and n64 is not None and not bool(n64["no_worse_than_bound_control"]):
        verdict = "MEASURED_ADVISORY_DEV_WINNER_FAILED_N64_REPLAY"
        scope = "INSTANCE x selected Probe-A path; contextual/template family open"
    elif accepted:
        verdict = "MEASURED_ADVISORY_DEV_DESCENT_FULL_WINDOW_CUSTODY_INCOMPLETE"
        scope = "INSTANCE x eight-pair Probe-A screen; no n600 claim"
    else:
        verdict = "MEASURED_ADVISORY_INITIAL_GRID_PLATEAU_FORMULATION_OPEN"
        scope = (
            "INSTANCE x KKT/Babai and matched-prefix proposals at one realized point across the full "
            "Probe-A basis/epsilon/radius grid; contextual template and joint-training families open"
        )
    producer_paths = (
        REPO_ROOT / "tools/probe_ddm_a1_bounded_collateral_realized.py",
        REPO_ROOT / "tools/measure_ddm_v17_iterative_realized_trust_region.py",
        REPO_ROOT / "src/tac/optimization/iterative_realized_trust_region.py",
    )
    receipt = {
        "schema": SCHEMA,
        "run_id": config.run_id,
        "lane_id": LANE_ID,
        "verdict": verdict,
        "verdict_scope": scope,
        "typed_config": config.model_dump(mode="json", by_alias=True),
        "typed_config_sha256": config.typed_config_hash(),
        "semantic_argv": list(semantic_argv),
        "operator_supplement": {
            "directive_utc": "2026-07-23T02:55:15Z",
            "audit_memo": AUDIT_MEMO,
            "audit_main_commit": AUDIT_MAIN_COMMIT,
            "consumed": True,
        },
        "source_receipts": {
            "v15": {"path": config.source_v15_receipt, "sha256": config.source_v15_receipt_sha256},
            "v16": {"path": config.source_v16_receipt, "sha256": config.source_v16_receipt_sha256},
            "v16_pointer_unmodified": v16_receipt.get("pointer"),
            "v15_control": _source_control(v15_receipt),
        },
        "producer_custody": [
            {
                "path": _portable(producer),
                "bytes": producer.stat().st_size,
                "sha256": _sha256(_read_regular_file_once(producer)),
            }
            for producer in producer_paths
        ],
        "iterations": iterations,
        "iterations_to_plateau_or_cap": len(iterations),
        "accepted_iterations": len(accepted),
        "validity_radius": {
            "rho_definition": "Fisher-margin debt realized reduction divided by affine projected-M predicted reduction",
            "raw_rows": raw_curve,
            "curve_by_lattice_quanta": list(curve),
            "stable_law_registration_eligible": len(curve) >= 2 and sum(int(row["rho_count"]) for row in curve) >= 4,
        },
        "realized_deltas_on_preregistered_eight_pairs": {
            "d_seg": float(final_measurement["d_seg"]) - float(baseline["d_seg"]),
            "archive_bytes": int(final_measurement["archive_bytes"]) - int(baseline["archive_bytes"]),
            "joint_objective": float(final_measurement["advisory_score_formula_value"])
            - float(baseline["advisory_score_formula_value"]),
        },
        "solve_vs_model_disabled_j2": {
            "solve_exact_calls": sum(int(row["comparison"]["solve_exact_calls"]) for row in iterations),
            "model_disabled_j2_exact_calls": sum(
                int(row["comparison"]["model_disabled_j2_exact_calls"]) for row in iterations
            ),
            "same_budget": all(bool(row["comparison"]["same_budget"]) for row in iterations),
            "solve_summed_best_realized_objective_improvement": solve_gain,
            "model_disabled_j2_summed_best_realized_objective_improvement": descent_gain,
            "winner": "solve"
            if solve_gain > descent_gain
            else "model_disabled_j2"
            if descent_gain > solve_gain
            else "tie",
        },
        "n64": n64 if n64 is not None else {"status": "NOT_RUN_NO_DEV_WINNER"},
        "n600": n600 if n600 is not None else {"status": "NOT_RUN_NO_N64_ADMITTED_WINNER"},
        "probe_a_admitted": probe_admitted,
        "debt_ceiling": {
            "Lane_plus_Movable_percent_of_remaining_in_box_debt": ROLE_DEBT_CEILING_PERCENT,
            "fixed_paint_Movable_projection_gap_remaining_percent": FIXED_PAINT_MOVABLE_GAP_REMAINING_PERCENT,
            "authority": "DERIVED_CEILINGS_NOT_EXPECTED_GAIN",
        },
        "triality": {
            "dsl": "DDMA1BoundedCollateralRealizedConfigV1",
            "dag": ".omx/research/ddm_v17_iterative_realized_trust_region_DAG_FEED_20260723.json",
            "equations": "tac.canonical_equations.ddm_v17_validity_radius_law_20260723",
        },
        "storage_preflight": storage,
        "target_custody": target_custody,
        "resume": {
            "per_iteration_checkpoints": True,
            "per_n64_n600_batch_checkpoints": True,
            "all_candidate_archives_preserved": True,
            "interrupted_pre_directive_run_separate": True,
        },
        "stores_consulted": [
            "CLAUDE.md",
            "AGENTS.md",
            "PROGRAM.md",
            "docs/operating_manual_craft_handoff.md",
            AUDIT_MEMO,
            config.source_v15_receipt,
            config.source_v16_receipt,
            ".omx/research/g2f_bidirectional_amplitude_ladder_chart_level_20260721T153318Z.md",
            ".omx/research/erm_2607_10128_crosswalk_20260720T154953Z.md",
        ],
        "pointer": f"{POINTER_SCORE_TEXT} [contest-CPU]",
        "pointer_moved": False,
        "evidence_axis": EVIDENCE_AXIS,
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
        "promotion_eligible": False,
        "main_landing_review_required": True,
    }
    _write_json(path, receipt)
    return path


def run(
    config: DDMA1BoundedCollateralRealizedConfigV1,
    output_directory: Path,
    semantic_argv: tuple[str, ...],
) -> Path | None:
    root = output_directory.resolve()
    storage = _storage_preflight(root)
    root.mkdir(parents=True, exist_ok=True)
    v15_receipt, v16_receipt, v16_config = _bindings(config)
    target = _target_custody(v16_config, root)
    n64_receipt, n64_config, n64_archive, n600_receipt, n600_config, n600_archive = _v15_bindings(v16_config)
    bank = receive_carrier_compose_archive(n600_archive).scorer_solved_templates
    if bank is None:
        raise DirectDescriptionError("Probe A lost the bound v15 template bank")
    _set_state_bank(bank)
    cache = Path(v16_config.target_cache_path)
    labels_all = open_stored_npy_memmap(cache, "lstars")
    poses_all = open_stored_npy_memmap(cache, "gt_poses")
    gt_f0 = open_stored_npy_memmap(cache, "gt_f0")
    gt_f1 = open_stored_npy_memmap(cache, "gt_f1")
    segnet, posenet, _scorer_custody = _load_models(n600_config)
    make_scorers_differentiable(posenet, segnet)
    problem = _build_problem(
        config=config,
        root=root,
        n64_receipt=n64_receipt,
        n600_cfg=n600_config,
        n600_archive=n600_archive,
        labels_all=labels_all,
        segnet=segnet,
        posenet=posenet,
    )
    if problem.get("typed_config_sha256") != config.typed_config_hash():
        raise DirectDescriptionError("Probe A preserved problem typed-config identity differs")
    iterations = [_json(path) for path in sorted((root / "stage_checkpoints").glob("iteration_*.json"))]
    terminated = bool(
        iterations and (bool(iterations[-1]["plateau"]) or len(iterations) >= config.maximum_realized_iterations)
    )
    if not terminated:
        row = _run_iteration(
            config=config,
            root=root,
            problem=problem,
            iterations=iterations,
            n600_config=n600_config,
            labels_all=labels_all,
            poses_all=poses_all,
            segnet=segnet,
            posenet=posenet,
        )
        print(json.dumps({"complete": False, "stage": f"iteration_{row['iteration']:02d}"}))
        return None
    accepted = any(bool(row["accepted"]) for row in iterations)
    n64_path = root / "stage_checkpoints" / "ladder_n64.json"
    n64 = _json(n64_path) if n64_path.exists() else None
    if accepted and n64 is None:
        row = _run_ladder(
            rung="n64",
            config=config,
            root=root,
            problem=problem,
            final_iteration=iterations[-1],
            v15_receipt=v15_receipt,
            v15_n600_receipt=n600_receipt,
            n64_config=n64_config,
            n64_archive=n64_archive,
            n600_config=n600_config,
            n600_archive=n600_archive,
            labels_all=labels_all,
            poses_all=poses_all,
            gt_f0=gt_f0,
            gt_f1=gt_f1,
            segnet=segnet,
            posenet=posenet,
        )
        print(json.dumps({"complete": False, "stage": "n64", "d_seg": row["measurement"]["d_seg"]}))
        return None
    n600_path = root / "stage_checkpoints" / "ladder_n600.json"
    n600 = _json(n600_path) if n600_path.exists() else None
    if accepted and n64 is not None and n64["no_worse_than_bound_control"] and n600 is None:
        row = _run_ladder(
            rung="n600",
            config=config,
            root=root,
            problem=problem,
            final_iteration=iterations[-1],
            v15_receipt=v15_receipt,
            v15_n600_receipt=n600_receipt,
            n64_config=n64_config,
            n64_archive=n64_archive,
            n600_config=n600_config,
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
        v15_receipt=v15_receipt,
        v16_receipt=v16_receipt,
        iterations=iterations,
        n64=n64,
        n600=n600,
    )
    print(json.dumps({"complete": True, "receipt": _portable(receipt)}))
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    config = DDMA1BoundedCollateralRealizedConfigV1.model_validate_json(_read_regular_file_once(args.config))
    semantic_argv = (
        "tools/probe_ddm_a1_bounded_collateral_realized.py",
        "--config",
        _portable(args.config),
        "--output-directory",
        _portable(args.output_directory),
    )
    run(config, args.output_directory, semantic_argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
