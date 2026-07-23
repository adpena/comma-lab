#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure a receiver-closed coupled SegNet-margin solve for DDM v16.

This is an encode-only, local CPU, false-authority harness.  It differentiates
through the frozen scorer while assembling the candidate, but the counted
archive contains only shared 2x2 RGB templates, pair-local template phases,
and sparse camera-R compensation bytes.  Every proposed SQP step is rounded to
the uint8 lattice and remeasured through the real receiver and frozen scorer.

One invocation advances exactly one durable stage: SQP round 1/2, the
eight-island rung, n64, n600, or the final receipt.  Thus every nonlinear round
and every expensive ladder rung is independently resumable from immutable
disk state.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictFloat, StrictInt, model_validator

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
for _path in (SRC, REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap  # noqa: E402
from tac.optimization.coupled_margin_levelset import (  # noqa: E402
    CouplingOperator,
    babai_nearest_plane,
    predicted_margin,
    solve_active_set_kkt,
    validate_coupling_operator_fd,
)
from tac.optimization.direct_description_carrier_compose import (  # noqa: E402
    RowBandScorerTemplateV1,
    ScorerSolvedTemplateBankV1,
    encode_scorer_solved_template_bank,
    receive_carrier_compose_archive,
    rfc8785_canonicalize,
)
from tac.optimization.direct_description_coupled_margin import (  # noqa: E402
    CoupledMarginProgramV1,
    SparseCameraCompensationV1,
    TemplatePlacementV1,
    compile_coupled_margin_archive,
    coupled_margin_byte_rows,
    encode_coupled_margin_program,
    receive_coupled_margin_archive,
)
from tac.optimization.direct_description_minimizer import (  # noqa: E402
    SOURCE_BYTES,
    DirectDescriptionError,
    _read_regular_file_once,
)
from tac.optimization.uint8_lattice_feasibility import DisjointResizeOperator  # noqa: E402
from tac.scorer import make_scorers_differentiable  # noqa: E402
from tac.through_r.resolution_chain import CAMERA_H, CAMERA_W, SEG_H, SEG_W  # noqa: E402
from tools.measure_ddm_v14_realization_fidelity import (  # noqa: E402
    EVIDENCE_AXIS,
    POINTER_SCORE_TEXT,
    _forward,
    _load_models,
    _publish_immutable,
    _storage_preflight,
)
from tools.measure_ddm_v15_scorer_solved_templates import (  # noqa: E402
    REPRESENTATIVE_ISLANDS,
    DDMV15ScorerSolvedTemplateConfigV1,
    _bound_bytes,
    _bound_json,
    _compile_from_v14,
)

RESULT_SCHEMA = "ddm_v16_coupled_joint_solve_receipt.v1"
PROBLEM_SCHEMA = "ddm_v16_coupled_joint_problem.v1"
ROUND_SCHEMA = "ddm_v16_coupled_joint_round.v1"
LADDER_SCHEMA = "ddm_v16_receiver_ladder_rung.v1"
LANE_ID = "ddm_v16_coupled_joint_solve"
# Canonical comma10k SegNet order is
# Road=0, Lane=1, Undrivable=2, Movable=3, MyCar=4.  Keep this local mapping
# explicit because confusing Lane with MyCar invalidates both the coupling
# rows and the per-stratum ladder while leaving aggregate d_seg plausible.
TARGET_ROLES: Mapping[str, int] = {"Movable": 3, "Lane": 1}
V14_DSEG = "0.027470296224"
V14_MOVABLE_DSEG = "0.291615222639"
V14_LANE_DSEG = "0.435195521828"


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).view(np.uint8)).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(_read_regular_file_once(path))
    except (OSError, json.JSONDecodeError) as exc:
        raise DirectDescriptionError(f"malformed JSON: {path}") from exc
    if not isinstance(value, dict):
        raise DirectDescriptionError(f"JSON must contain one object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    _publish_immutable(path, rfc8785_canonicalize(dict(value)))


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    payload = _read_regular_file_once(temporary)
    temporary.unlink()
    _publish_immutable(path, payload)


def _portable(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


class DDMV16CoupledJointSolveConfigV1(BaseModel):
    """SHA-bound local-only staged solve contract."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    schema_: Literal["DDMV16CoupledJointSolveConfigV1"] = Field(
        default="DDMV16CoupledJointSolveConfigV1", alias="schema", serialization_alias="schema"
    )
    run_id: str = Field(min_length=8)
    seed: StrictInt = 1234
    n64_receipt_path: str = Field(min_length=1)
    n64_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    n64_archive_path: str = Field(min_length=1)
    n64_archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    n600_receipt_path: str = Field(min_length=1)
    n600_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    n600_archive_path: str = Field(min_length=1)
    n600_archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    target_cache_path: str = Field(min_length=1)
    target_cache_bytes: StrictInt = Field(gt=0)
    target_cache_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    upstream_root: str = Field(min_length=1)
    representative_source_pair_ids: tuple[StrictInt, ...] = REPRESENTATIVE_ISLANDS
    nonlinear_rounds: StrictInt = Field(default=2, ge=2, le=4)
    maximum_target_cells_per_role_pair: StrictInt = Field(default=2, ge=1, le=8)
    maximum_protected_cells_per_pair: StrictInt = Field(default=4, ge=1, le=16)
    maximum_fd_entries: StrictInt = Field(default=8, ge=1, le=32)
    vjp_chunk_rows: StrictInt = Field(default=8, ge=1, le=32)
    template_trust_radius_u8: StrictFloat = Field(default=4.0, gt=0.0, le=32.0)
    compensation_trust_radius_u8: StrictFloat = Field(default=16.0, gt=0.0, le=127.0)
    target_margin_epsilon: StrictFloat = Field(default=0.05, ge=0.0, le=10.0)
    protected_margin_epsilon: StrictFloat = Field(default=0.0, ge=0.0, le=10.0)
    pose_trust_radius: StrictFloat = Field(default=0.05, gt=0.0, le=1.0)
    scorer_threads: StrictInt = Field(default=4, ge=1, le=16)
    scorer_batch_size: Literal[16] = 16
    archive_box_bytes: Literal[160000] = 160000
    movable_gate: StrictFloat = Field(default=0.05, ge=0.0, le=1.0)
    research_only: Literal[True] = True
    execution_allowed: Literal[False] = False
    score_claim: Literal[False] = False
    d_seg_claim: Literal[False] = False
    d_pose_claim: Literal[False] = False

    @model_validator(mode="after")
    def _valid(self) -> DDMV16CoupledJointSolveConfigV1:
        if self.representative_source_pair_ids != REPRESENTATIVE_ISLANDS:
            raise ValueError("v16 must use the preregistered eight-island development set")
        if len(set(self.representative_source_pair_ids)) != 8:
            raise ValueError("v16 development pair ids must be unique")
        return self

    def typed_config_hash(self) -> str:
        return _sha256(rfc8785_canonicalize(self.model_dump(mode="json", by_alias=True)))


def _v15_bindings(
    config: DDMV16CoupledJointSolveConfigV1,
) -> tuple[
    dict[str, Any], DDMV15ScorerSolvedTemplateConfigV1, bytes, dict[str, Any], DDMV15ScorerSolvedTemplateConfigV1, bytes
]:
    n64_receipt = _bound_json(REPO_ROOT / config.n64_receipt_path, config.n64_receipt_sha256, "v15 n64 receipt")
    n600_receipt = _bound_json(REPO_ROOT / config.n600_receipt_path, config.n600_receipt_sha256, "v15 n600 receipt")
    n64_cfg = DDMV15ScorerSolvedTemplateConfigV1.model_validate(n64_receipt["typed_config"])
    n600_cfg = DDMV15ScorerSolvedTemplateConfigV1.model_validate(n600_receipt["typed_config"])
    n64_archive = _bound_bytes(REPO_ROOT / config.n64_archive_path, config.n64_archive_sha256, "v15 n64 archive")
    n600_archive = _bound_bytes(REPO_ROOT / config.n600_archive_path, config.n600_archive_sha256, "v15 n600 archive")
    if n64_cfg.pair_count != 64 or n600_cfg.pair_count != 600:
        raise DirectDescriptionError("bound v15 receipts do not form the n64/n600 ladder")
    if n600_cfg.target_cache_path != config.target_cache_path or n600_cfg.upstream_root != config.upstream_root:
        raise DirectDescriptionError("v16 scorer/target custody differs from v15")
    return n64_receipt, n64_cfg, n64_archive, n600_receipt, n600_cfg, n600_archive


def _expanded_bank(bank: ScorerSolvedTemplateBankV1, values: np.ndarray | None = None) -> ScorerSolvedTemplateBankV1:
    rows = []
    for index, template in enumerate(bank.templates):
        if values is None:
            patch = np.tile(np.frombuffer(template.rgb_u8, dtype=np.uint8), (4, 1))
        else:
            patch = np.asarray(values[index], dtype=np.uint8).reshape(4, 3)
        rows.append(
            RowBandScorerTemplateV1(
                template.role,
                template.application,
                template.scorer_row_start,
                template.scorer_row_stop,
                2,
                2,
                bytes(patch.reshape(-1).tolist()),
            )
        )
    return ScorerSolvedTemplateBankV1(tuple(rows))


def _bank_values(bank: ScorerSolvedTemplateBankV1) -> np.ndarray:
    return np.asarray([row.patch() for row in bank.templates], dtype=np.uint8)


def _candidate_attempt_values(receipt: Mapping[str, Any], bank: ScorerSolvedTemplateBankV1) -> dict[str, np.ndarray]:
    result: dict[str, np.ndarray] = {}
    for solve in receipt["solver"]:
        improving = [row for row in solve["attempted_steps"] if int(row["role_error_improvement"]) > 0]
        chosen = min(
            improving,
            key=lambda row: (int(row["harmful_off_target_flips"]), -int(row["role_error_improvement"])),
        )
        value = _bank_values(bank)
        role_indexes = [index for index, template in enumerate(bank.templates) if template.role == solve["role"]]
        for local_index, template_index in enumerate(role_indexes):
            value[template_index] = np.tile(np.asarray(chosen["rgb_u8"][local_index], dtype=np.uint8), (2, 2, 1))
        result[str(solve["role"])] = value
    return result


def _program(
    pair_ids: Sequence[int],
    template_count: int,
    phases: Mapping[str, Sequence[int]],
    supports: Sequence[Mapping[str, int]],
    compensation: np.ndarray,
) -> CoupledMarginProgramV1:
    placements = tuple(
        sorted(
            TemplatePlacementV1(
                int(pair_id),
                template_index,
                int(phases.get(f"{pair_id}:{template_index}", (0, 0))[0]),
                int(phases.get(f"{pair_id}:{template_index}", (0, 0))[1]),
            )
            for pair_id in pair_ids
            for template_index in range(template_count)
        )
    )
    rows = []
    for index, support in enumerate(supports):
        delta = tuple(int(value) for value in np.asarray(compensation[index]).tolist())
        if delta != (0, 0, 0):
            rows.append(
                SparseCameraCompensationV1(
                    int(support["source_pair_id"]),
                    int(support["frame_index"]),
                    int(support["camera_y"]),
                    int(support["camera_x"]),
                    delta,
                )
            )
    return CoupledMarginProgramV1(placements, tuple(sorted(rows)))


def _archive_for_state(
    v14_archive: bytes,
    values: np.ndarray,
    pair_ids: Sequence[int],
    phases: Mapping[str, Sequence[int]],
    supports: Sequence[Mapping[str, int]],
    compensation: np.ndarray,
) -> tuple[bytes, bytes, CoupledMarginProgramV1]:
    bank = _expanded_bank(_expanded_bank_from_values_shape(values), values)
    base = _compile_from_v14(v14_archive, bank)
    program = _program(pair_ids, len(bank.templates), phases, supports, compensation)
    return compile_coupled_margin_archive(base, program), base, program


def _expanded_bank_from_values_shape(values: np.ndarray) -> ScorerSolvedTemplateBankV1:
    """Placeholder metadata carrier; callers replace this through ``_STATE_BANK``.

    This function is rebound by ``_set_state_bank`` after custody validation.
    Keeping the archive builder pure makes its uint8/byte accounting easy to
    exercise in unit tests without scorer imports.
    """

    if _STATE_BANK is None or len(_STATE_BANK.templates) != int(values.shape[0]):
        raise DirectDescriptionError("v16 template metadata has not been SHA-bound")
    return _STATE_BANK


_STATE_BANK: ScorerSolvedTemplateBankV1 | None = None


def _set_state_bank(bank: ScorerSolvedTemplateBankV1) -> None:
    global _STATE_BANK
    _STATE_BANK = bank


def _torch_forward_full(segnet: Any, posenet: Any, camera: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import torch

    value = torch.from_numpy(np.ascontiguousarray(camera)).permute(0, 1, 4, 2, 3).contiguous().float()
    with torch.inference_mode():
        logits = segnet(segnet.preprocess_input(value))
        pose_output = posenet(posenet.preprocess_input(value))
        pose = pose_output["pose"] if isinstance(pose_output, dict) else pose_output
    logits_np = logits.cpu().numpy().astype(np.float64)
    return logits_np, logits.argmax(dim=1).cpu().numpy().astype(np.uint8), pose[:, :6].cpu().numpy().astype(np.float64)


def _base_v14_bytes(config: DDMV15ScorerSolvedTemplateConfigV1) -> bytes:
    return _bound_bytes(REPO_ROOT / config.v14_archive_path, config.v14_archive_sha256, "v14 base archive")


def _support_coordinate(operator: DisjointResizeOperator, scorer_y: int, scorer_x: int) -> tuple[int, int]:
    row = operator.row_supports[scorer_y]
    col = operator.col_supports[scorer_x]
    y = row.indices[int(np.argmax(np.asarray(row.weights, dtype=np.float64)))]
    x = col.indices[int(np.argmax(np.asarray(col.weights, dtype=np.float64)))]
    return int(y), int(x)


def _select_target_cells(
    logits: np.ndarray,
    cells: np.ndarray,
    labels: np.ndarray,
    pair_ids: Sequence[int],
    limit: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for local, pair_id in enumerate(pair_ids):
        for role, role_id in TARGET_ROLES.items():
            eligible = np.argwhere((labels[local] == role_id) & (cells[local] != role_id))
            ranked: list[tuple[float, int, int, int]] = []
            for y_raw, x_raw in eligible:
                y, x = int(y_raw), int(x_raw)
                rival = int(np.argmax(logits[local, :, y, x]))
                margin = float(logits[local, role_id, y, x] - logits[local, rival, y, x])
                ranked.append((abs(margin), y, x, rival))
            for _distance, y, x, rival in sorted(ranked)[:limit]:
                rows.append(
                    {
                        "source_pair_id": int(pair_id),
                        "scorer_y": y,
                        "scorer_x": x,
                        "target_class": role_id,
                        "initial_rival_class": rival,
                        "role": role,
                    }
                )
    return rows


def _build_problem(
    *,
    config: DDMV16CoupledJointSolveConfigV1,
    root: Path,
    n64_receipt: Mapping[str, Any],
    n600_cfg: DDMV15ScorerSolvedTemplateConfigV1,
    n600_archive: bytes,
    labels_all: np.ndarray,
    segnet: Any,
    posenet: Any,
) -> dict[str, Any]:
    path = root / "stage_checkpoints" / "00_problem.json"
    original_receiver = receive_carrier_compose_archive(n600_archive)
    original_bank = original_receiver.scorer_solved_templates
    if original_bank is None or len(original_bank.templates) == 0:
        raise DirectDescriptionError("bound v15 archive lacks scorer-solved templates")
    _set_state_bank(original_bank)
    if path.exists():
        return _json(path)
    expanded = _expanded_bank(original_bank)
    initial_values = _bank_values(expanded)
    v14_archive = _base_v14_bytes(n600_cfg)
    expanded_base = _compile_from_v14(v14_archive, expanded)
    pair_ids = tuple(int(value) for value in config.representative_source_pair_ids)
    phases: dict[str, tuple[int, int]] = {
        f"{pair_id}:{template_index}": (0, 0)
        for pair_id in pair_ids
        for template_index in range(len(expanded.templates))
    }
    initial_program = _program(pair_ids, len(expanded.templates), phases, (), np.zeros((0, 3), dtype=np.int16))
    initial_archive = compile_coupled_margin_archive(expanded_base, initial_program)
    initial_receiver = receive_coupled_margin_archive(initial_archive)
    local_ids = pair_ids
    camera = initial_receiver.render_camera_pairs(local_ids)
    v15_camera = original_receiver.render_camera_pairs(local_ids)
    if not np.array_equal(camera, v15_camera):
        raise DirectDescriptionError("uniform 2x2 v16 initialization does not reproduce v15 pixels")
    labels = np.asarray(labels_all[np.asarray(pair_ids, dtype=np.int64)], dtype=np.int64)
    logits, cells, pose6 = _torch_forward_full(segnet, posenet, camera)
    targets = _select_target_cells(
        logits,
        cells,
        labels,
        pair_ids,
        config.maximum_target_cells_per_role_pair,
    )

    attempted = _candidate_attempt_values(n64_receipt, expanded)
    protected_keys: set[tuple[int, int, int, int, int]] = set()
    collateral_rows: list[dict[str, Any]] = []
    for role, role_id in TARGET_ROLES.items():
        candidate_bank = _expanded_bank(original_bank, attempted[role])
        candidate_archive = compile_coupled_margin_archive(
            _compile_from_v14(v14_archive, candidate_bank),
            initial_program,
        )
        candidate_camera = receive_coupled_margin_archive(candidate_archive).render_camera_pairs(local_ids)
        candidate_logits, candidate_cells, _candidate_pose = _torch_forward_full(segnet, posenet, candidate_camera)
        harmful = (labels != role_id) & (cells == labels) & (candidate_cells != labels)
        for local, pair_id in enumerate(pair_ids):
            locations = np.argwhere(harmful[local])
            # Deterministic severity order: largest lost GT-vs-winning margin first.
            ranked = []
            for y_raw, x_raw in locations:
                y, x = int(y_raw), int(x_raw)
                target_class = int(labels[local, y, x])
                rival = int(candidate_cells[local, y, x])
                severity = float(candidate_logits[local, rival, y, x] - candidate_logits[local, target_class, y, x])
                ranked.append((-severity, y, x, target_class, rival))
            for _neg_severity, y, x, target_class, rival in sorted(ranked)[: config.maximum_protected_cells_per_pair]:
                key = (int(pair_id), y, x, target_class, rival)
                if key in protected_keys:
                    continue
                protected_keys.add(key)
                collateral_rows.append(
                    {
                        "source_pair_id": int(pair_id),
                        "scorer_y": y,
                        "scorer_x": x,
                        "target_class": target_class,
                        "initial_rival_class": rival,
                        "discovered_by_role_proposal": role,
                    }
                )

    operator = DisjointResizeOperator.build(
        camera_h=CAMERA_H,
        camera_w=CAMERA_W,
        scorer_h=SEG_H,
        scorer_w=SEG_W,
    )
    support_by_key: dict[tuple[int, int, int], dict[str, int]] = {}
    for row in collateral_rows:
        camera_y, camera_x = _support_coordinate(operator, int(row["scorer_y"]), int(row["scorer_x"]))
        support_by_key.setdefault(
            (int(row["source_pair_id"]), camera_y, camera_x),
            {
                "source_pair_id": int(row["source_pair_id"]),
                "frame_index": 1,
                "camera_y": camera_y,
                "camera_x": camera_x,
            },
        )
    supports = [support_by_key[key] for key in sorted(support_by_key)]
    template_payload = encode_scorer_solved_template_bank(expanded)
    _publish_immutable(root / "expanded_templates_2x2.ddst", template_payload)
    _publish_immutable(root / "initial_v16.not_a_candidate.zip.receipt-bytes", initial_archive)
    problem = {
        "schema": PROBLEM_SCHEMA,
        "run_id": config.run_id,
        "typed_config_sha256": config.typed_config_hash(),
        "representative_source_pair_ids": list(pair_ids),
        "target_cells": targets,
        "protected_collateral_cells": collateral_rows,
        "sparse_compensation_support": supports,
        "initial_template_values_u8": initial_values.tolist(),
        "initial_compensation_rgb_i8": np.zeros((len(supports), 3), dtype=np.int8).tolist(),
        "initial_phases": {key: list(value) for key, value in phases.items()},
        "initial_pose6": pose6.tolist(),
        "pose_trust_radius": config.pose_trust_radius,
        "initial_cells_sha256": _sha256_array(cells),
        "initial_camera_sha256": _sha256_array(camera),
        "initial_archive": {
            "path": _portable(root / "initial_v16.not_a_candidate.zip.receipt-bytes"),
            "bytes": len(initial_archive),
            "sha256": _sha256(initial_archive),
        },
        "expanded_template_bank": {
            "path": _portable(root / "expanded_templates_2x2.ddst"),
            "bytes": len(template_payload),
            "sha256": _sha256(template_payload),
            "template_count": len(expanded.templates),
            "continuous_template_dofs": int(initial_values.size),
        },
        "coupling_scope": {
            "active": [
                "shared_2x2_template_bytes_at_all_pair_placements",
                "pair_local_template_phase",
                "sparse_frame1_camera_compensation",
            ],
            "frozen_inherited": ["v14_lane_profile", "v14_worldsheet", "predictor", "pose6_codes"],
            "frozen_reason": "v16 extends the v15 receiver; inherited lane/worldsheet edits are outside the counted v16 program",
        },
        "collateral_discovery": {
            "source": "minimum-harmful improving v15 proposal per target role, rerendered on the preregistered eight islands",
            "Movable_v15_minimum_harmful": 13,
            "Lane_v15_minimum_harmful": 23,
            "measured_protected_cells_retained": len(collateral_rows),
            "unique_sparse_camera_supports": len(supports),
        },
        "receiver_equivalence": {
            "uniform_2x2_equals_v15_camera": True,
            "v15_camera_sha256": _sha256_array(v15_camera),
            "exact_R": "camera_uint8_then_frozen_SegNet.preprocess_input_bilinear_384x512",
        },
        "score_claim": False,
        "evidence_axis": EVIDENCE_AXIS,
    }
    _write_json(path, problem)
    return problem


def _state(
    problem: Mapping[str, Any], previous: Mapping[str, Any] | None
) -> tuple[np.ndarray, np.ndarray, dict[str, list[int]]]:
    if previous is None:
        return (
            np.asarray(problem["initial_template_values_u8"], dtype=np.int16),
            np.asarray(problem["initial_compensation_rgb_i8"], dtype=np.int16),
            {key: list(value) for key, value in problem["initial_phases"].items()},
        )
    selected = previous["selected_state"]
    return (
        np.asarray(selected["template_values_u8"], dtype=np.int16),
        np.asarray(selected["compensation_rgb_i8"], dtype=np.int16),
        {key: list(value) for key, value in selected["phases"].items()},
    )


def _dof_labels(template_count: int, supports: Sequence[Mapping[str, int]]) -> tuple[str, ...]:
    labels = [
        f"template:{template}:patch:{patch_y}:{patch_x}:channel:{channel}@all_placements"
        for template in range(template_count)
        for patch_y in range(2)
        for patch_x in range(2)
        for channel in range(3)
    ]
    labels.extend(
        f"sparse_comp:{row['source_pair_id']}:f{row['frame_index']}:y{row['camera_y']}:x{row['camera_x']}:channel:{channel}"
        for row in supports
        for channel in range(3)
    )
    return tuple(labels)


def _render_pair_torch(
    *,
    torch: Any,
    base_camera: np.ndarray,
    template_masks: Sequence[np.ndarray],
    parameters: Any,
    phases: Mapping[str, Sequence[int]],
    pair_id: int,
    supports: Sequence[Mapping[str, int]],
    template_count: int,
) -> Any:
    patch_size = template_count * 2 * 2 * 3
    patch = parameters[:patch_size].reshape(template_count, 2, 2, 3)
    compensation = parameters[patch_size:].reshape(len(supports), 3)
    output = torch.from_numpy(np.ascontiguousarray(base_camera[None])).float()
    yy = np.arange(CAMERA_H, dtype=np.int16)[:, None]
    xx = np.arange(CAMERA_W, dtype=np.int16)[None, :]
    for template_index, raw_mask in enumerate(template_masks):
        phase_y, phase_x = phases.get(f"{pair_id}:{template_index}", (0, 0))
        mask = np.asarray(raw_mask, dtype=bool)
        for patch_y in range(2):
            for patch_x in range(2):
                active_np = mask & (((yy + int(phase_y)) % 2) == patch_y) & (((xx + int(phase_x)) % 2) == patch_x)
                if not np.any(active_np):
                    continue
                active = torch.from_numpy(active_np)[None, None, :, :, None]
                colour = patch[template_index, patch_y, patch_x].reshape(1, 1, 1, 1, 3)
                output = torch.where(active, colour, output)
    if supports:
        delta = torch.zeros_like(output)
        for index, support in enumerate(supports):
            if int(support["source_pair_id"]) != pair_id:
                continue
            delta[
                0,
                int(support["frame_index"]),
                int(support["camera_y"]),
                int(support["camera_x"]),
            ] = compensation[index]
        output = output + delta
    return output.clamp(0.0, 255.0)


def _pair_margin_tensor(
    *,
    torch: Any,
    segnet: Any,
    posenet: Any,
    camera: Any,
    rows: Sequence[Mapping[str, Any]],
) -> Any:
    chw = camera.permute(0, 1, 4, 2, 3).contiguous()
    logits = segnet(segnet.preprocess_input(chw))
    pose_output = posenet(posenet.preprocess_input(chw))
    pose = pose_output["pose"] if isinstance(pose_output, dict) else pose_output
    values = []
    for row in rows:
        if row["kind"] in {"target", "protected"}:
            values.append(
                logits[0, int(row["target_class"]), int(row["scorer_y"]), int(row["scorer_x"])]
                - logits[0, int(row["rival_class"]), int(row["scorer_y"]), int(row["scorer_x"])]
            )
        elif row["kind"] == "pose_upper":
            values.append(float(row["radius"]) - (pose[0, int(row["coordinate"])] - float(row["reference"])))
        elif row["kind"] == "pose_lower":
            values.append(float(row["radius"]) + (pose[0, int(row["coordinate"])] - float(row["reference"])))
        else:
            raise DirectDescriptionError("unknown v16 margin row kind")
    if not values:
        raise DirectDescriptionError("v16 pair cluster has no constraints")
    return torch.stack(values)


def _linearization_rows(
    *,
    problem: Mapping[str, Any],
    pair_ids: Sequence[int],
    logits: np.ndarray,
) -> list[dict[str, Any]]:
    pair_to_local = {int(pair_id): local for local, pair_id in enumerate(pair_ids)}
    rows: list[dict[str, Any]] = []
    for kind, source in (("target", problem["target_cells"]), ("protected", problem["protected_collateral_cells"])):
        for source_row in source:
            row = dict(source_row)
            local = pair_to_local[int(row["source_pair_id"])]
            y, x = int(row["scorer_y"]), int(row["scorer_x"])
            target = int(row["target_class"])
            rivals = np.array(logits[local, :, y, x], copy=True)
            rivals[target] = -np.inf
            row.update(
                {
                    "kind": kind,
                    "rival_class": int(np.argmax(rivals)),
                    "required_margin": 0.0,
                }
            )
            rows.append(row)
    initial_pose = np.asarray(problem["initial_pose6"], dtype=np.float64)
    for local, pair_id in enumerate(pair_ids):
        for coordinate in range(6):
            for kind in ("pose_upper", "pose_lower"):
                rows.append(
                    {
                        "kind": kind,
                        "source_pair_id": int(pair_id),
                        "coordinate": coordinate,
                        "reference": float(initial_pose[local, coordinate]),
                        "radius": float(problem["pose_trust_radius"]),
                        "required_margin": 0.0,
                    }
                )
    # The CouplingOperator contract requires all target crossings first.
    order = {"target": 0, "protected": 1, "pose_upper": 2, "pose_lower": 3}
    return sorted(
        rows,
        key=lambda row: (
            order[str(row["kind"])],
            int(row["source_pair_id"]),
            int(row.get("scorer_y", row.get("coordinate", 0))),
            int(row.get("scorer_x", 0)),
        ),
    )


def _row_label(row: Mapping[str, Any]) -> str:
    if row["kind"] in {"target", "protected"}:
        return (
            f"{row['kind']}:{row['source_pair_id']}:y{row['scorer_y']}:x{row['scorer_x']}:"
            f"c{row['target_class']}-c{row['rival_class']}"
        )
    return f"{row['kind']}:{row['source_pair_id']}:pose{row['coordinate']}"


def _assemble_operator(
    *,
    config: DDMV16CoupledJointSolveConfigV1,
    problem: Mapping[str, Any],
    v14_archive: bytes,
    values: np.ndarray,
    compensation: np.ndarray,
    phases: Mapping[str, Sequence[int]],
    segnet: Any,
    posenet: Any,
) -> tuple[CouplingOperator, list[dict[str, Any]], list[dict[str, Any]], np.ndarray, np.ndarray]:
    import torch

    pair_ids = tuple(int(value) for value in problem["representative_source_pair_ids"])
    supports = list(problem["sparse_compensation_support"])
    template_count = int(values.shape[0])
    current_archive, current_base, _program_value = _archive_for_state(
        v14_archive,
        values.astype(np.uint8),
        pair_ids,
        phases,
        supports,
        compensation.astype(np.int16),
    )
    receiver = receive_coupled_margin_archive(current_archive)
    camera = receiver.render_camera_pairs(pair_ids)
    logits_np, cells_np, _pose_np = _torch_forward_full(segnet, posenet, camera)
    rows = _linearization_rows(problem=problem, pair_ids=pair_ids, logits=logits_np)
    parameters_np = np.concatenate((values.reshape(-1), compensation.reshape(-1))).astype(np.float64)
    dof_labels = _dof_labels(template_count, supports)
    if len(dof_labels) != parameters_np.size:
        raise DirectDescriptionError("v16 parameter/DOF label geometry differs")

    base_receiver = receive_carrier_compose_archive(v14_archive)
    mask_receiver = receive_carrier_compose_archive(current_base)
    local_rows: list[np.ndarray] = []
    local_margins: list[np.ndarray] = []
    cluster_receipts: list[dict[str, Any]] = []
    row_indexes_by_pair: dict[int, list[int]] = {
        pair_id: [index for index, row in enumerate(rows) if int(row["source_pair_id"]) == pair_id]
        for pair_id in pair_ids
    }
    for pair_id in pair_ids:
        pair_rows = [rows[index] for index in row_indexes_by_pair[pair_id]]
        base_camera = base_receiver.render_camera_pairs((pair_id,))[0]
        masks = [
            mask_receiver.base.template_camera_masks((pair_id,), template)[0]
            if hasattr(mask_receiver, "base")
            else mask_receiver.template_camera_masks((pair_id,), template)[0]
            for template in mask_receiver.scorer_solved_templates.templates
        ]
        parameters = torch.tensor(parameters_np, dtype=torch.float32, requires_grad=True)
        rendered = _render_pair_torch(
            torch=torch,
            base_camera=base_camera,
            template_masks=masks,
            parameters=parameters,
            phases=phases,
            pair_id=pair_id,
            supports=supports,
            template_count=template_count,
        )
        margin_tensor = _pair_margin_tensor(
            torch=torch,
            segnet=segnet,
            posenet=posenet,
            camera=rendered,
            rows=pair_rows,
        )
        chunks = []
        for start in range(0, int(margin_tensor.numel()), config.vjp_chunk_rows):
            stop = min(start + config.vjp_chunk_rows, int(margin_tensor.numel()))
            grad_outputs = torch.eye(int(margin_tensor.numel()), dtype=margin_tensor.dtype)[start:stop]
            try:
                gradient = torch.autograd.grad(
                    margin_tensor,
                    parameters,
                    grad_outputs=grad_outputs,
                    retain_graph=stop < int(margin_tensor.numel()),
                    is_grads_batched=True,
                )[0]
            except TypeError:
                gradient = torch.stack(
                    [
                        torch.autograd.grad(
                            margin_tensor[index],
                            parameters,
                            retain_graph=index + 1 < int(margin_tensor.numel()),
                        )[0]
                        for index in range(start, stop)
                    ]
                )
            chunks.append(gradient.detach().cpu().numpy().astype(np.float64))
        matrix = np.concatenate(chunks, axis=0)
        margin = margin_tensor.detach().cpu().numpy().astype(np.float64)
        required = np.asarray([float(row["required_margin"]) for row in pair_rows], dtype=np.float64)
        for index, row in enumerate(pair_rows):
            if row["kind"] == "target":
                required[index] = float(config.target_margin_epsilon)
            elif row["kind"] == "protected":
                required[index] = float(config.protected_margin_epsilon)
        activation_hash = hashlib.sha256(
            cells_np[pair_ids.index(pair_id)].tobytes()
            + rfc8785_canonicalize({key: list(value) for key, value in phases.items() if key.startswith(f"{pair_id}:")})
        ).hexdigest()
        local_operator = CouplingOperator(
            matrix=matrix,
            margin=margin,
            required_margin=required,
            targeted_count=sum(row["kind"] == "target" for row in pair_rows),
            row_labels=tuple(_row_label(row) for row in pair_rows),
            dof_labels=dof_labels,
            activation_pattern_sha256=activation_hash,
        )

        def margin_callback(
            delta: np.ndarray,
            *,
            bound_base_camera: np.ndarray = base_camera,
            bound_masks: Sequence[np.ndarray] = masks,
            bound_pair_id: int = pair_id,
            bound_pair_rows: Sequence[Mapping[str, Any]] = pair_rows,
        ) -> np.ndarray:
            candidate = torch.tensor(parameters_np + delta, dtype=torch.float32)
            candidate_camera = _render_pair_torch(
                torch=torch,
                base_camera=bound_base_camera,
                template_masks=bound_masks,
                parameters=candidate,
                phases=phases,
                pair_id=bound_pair_id,
                supports=supports,
                template_count=template_count,
            )
            with torch.no_grad():
                return (
                    _pair_margin_tensor(
                        torch=torch,
                        segnet=segnet,
                        posenet=posenet,
                        camera=candidate_camera,
                        rows=bound_pair_rows,
                    )
                    .cpu()
                    .numpy()
                    .astype(np.float64)
                )

        fd = validate_coupling_operator_fd(
            local_operator,
            margin_callback,
            epsilon=1e-2,
            maximum_entries=max(1, math.ceil(config.maximum_fd_entries / len(pair_ids))),
            seed=config.seed + pair_id,
            absolute_tolerance=5e-2,
            relative_tolerance=1e-1,
        )
        cluster_receipts.append(
            {
                "source_pair_id": pair_id,
                "rows": matrix.shape[0],
                "columns": matrix.shape[1],
                "matrix_sha256": _sha256_array(matrix),
                "margin_sha256": _sha256_array(margin),
                "activation_pattern_sha256": activation_hash,
                "finite_difference": asdict(fd),
                "derivative_producer": "torch_vjp_through_exact_receiver_composition_and_frozen_scorers",
            }
        )
        local_rows.append(matrix)
        local_margins.append(margin)

    # Pair-local assembly order differs from the global target-first contract.
    pair_stacked_rows = [row for pair_id in pair_ids for row in (rows[index] for index in row_indexes_by_pair[pair_id])]
    stacked_matrix = np.concatenate(local_rows, axis=0)
    stacked_margin = np.concatenate(local_margins, axis=0)
    lookup = {_row_label(row): index for index, row in enumerate(pair_stacked_rows)}
    permutation = np.asarray([lookup[_row_label(row)] for row in rows], dtype=np.int64)
    matrix = stacked_matrix[permutation]
    margin = stacked_margin[permutation]
    required = np.asarray(
        [
            config.target_margin_epsilon
            if row["kind"] == "target"
            else config.protected_margin_epsilon
            if row["kind"] == "protected"
            else 0.0
            for row in rows
        ],
        dtype=np.float64,
    )
    activation_hash = hashlib.sha256(
        cells_np.tobytes() + rfc8785_canonicalize({key: list(value) for key, value in sorted(phases.items())})
    ).hexdigest()
    operator = CouplingOperator(
        matrix=matrix,
        margin=margin,
        required_margin=required,
        targeted_count=sum(row["kind"] == "target" for row in rows),
        row_labels=tuple(_row_label(row) for row in rows),
        dof_labels=dof_labels,
        activation_pattern_sha256=activation_hash,
    )
    return operator, rows, cluster_receipts, parameters_np, camera


def _hard_margin_vector(
    rows: Sequence[Mapping[str, Any]], pair_ids: Sequence[int], logits: np.ndarray, pose6: np.ndarray
) -> np.ndarray:
    pair_to_local = {int(pair_id): local for local, pair_id in enumerate(pair_ids)}
    values = []
    for row in rows:
        local = pair_to_local[int(row["source_pair_id"])]
        if row["kind"] in {"target", "protected"}:
            values.append(
                float(
                    logits[local, int(row["target_class"]), int(row["scorer_y"]), int(row["scorer_x"])]
                    - logits[local, int(row["rival_class"]), int(row["scorer_y"]), int(row["scorer_x"])]
                )
            )
        elif row["kind"] == "pose_upper":
            values.append(
                float(row["radius"]) - (float(pose6[local, int(row["coordinate"])]) - float(row["reference"]))
            )
        else:
            values.append(
                float(row["radius"]) + (float(pose6[local, int(row["coordinate"])]) - float(row["reference"]))
            )
    return np.asarray(values, dtype=np.float64)


def _required_for_rows(config: DDMV16CoupledJointSolveConfigV1, rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray(
        [
            config.target_margin_epsilon
            if row["kind"] == "target"
            else config.protected_margin_epsilon
            if row["kind"] == "protected"
            else 0.0
            for row in rows
        ],
        dtype=np.float64,
    )


def _constraint_summary(rows: Sequence[Mapping[str, Any]], margins: np.ndarray, required: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for kind in ("target", "protected", "pose_upper", "pose_lower"):
        indexes = np.asarray([index for index, row in enumerate(rows) if row["kind"] == kind], dtype=np.int64)
        if indexes.size == 0:
            result[kind] = {"rows": 0, "violations": 0, "debt": "0.000000000000", "minimum_slack": None}
            continue
        slack = margins[indexes] - required[indexes]
        result[kind] = {
            "rows": int(indexes.size),
            "violations": int(np.count_nonzero(slack < 0.0)),
            "debt": f"{float(np.maximum(-slack, 0.0).sum()):.12f}",
            "minimum_slack": f"{float(slack.min()):.12f}",
        }
    return result


def _dev_measurement(
    *,
    archive: bytes,
    pair_ids: Sequence[int],
    rows: Sequence[Mapping[str, Any]],
    required: np.ndarray,
    labels_all: np.ndarray,
    poses_all: np.ndarray,
    segnet: Any,
    posenet: Any,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    receiver = receive_coupled_margin_archive(archive)
    camera = receiver.render_camera_pairs(pair_ids)
    logits, cells, pose6 = _torch_forward_full(segnet, posenet, camera)
    labels = np.asarray(labels_all[np.asarray(pair_ids, dtype=np.int64)])
    poses = np.asarray(poses_all[np.asarray(pair_ids, dtype=np.int64)])
    errors = cells != labels
    margins = _hard_margin_vector(rows, pair_ids, logits, pose6)
    dseg = float(np.mean(errors))
    dpose = float(np.mean(np.square(pose6 - poses), dtype=np.float64))
    per_role = {}
    for role, role_id in TARGET_ROLES.items():
        role_mask = labels == role_id
        per_role[role] = {
            "errors": int(np.count_nonzero(errors & role_mask)),
            "sites": int(np.count_nonzero(role_mask)),
            "d_seg": f"{float(np.count_nonzero(errors & role_mask)) / max(1, int(np.count_nonzero(role_mask))):.12f}",
        }
    score = 100.0 * dseg + math.sqrt(10.0 * dpose) + 25.0 * len(archive) / SOURCE_BYTES
    return (
        {
            "archive_bytes": len(archive),
            "archive_sha256": _sha256(archive),
            "errors": int(np.count_nonzero(errors)),
            "sites": int(errors.size),
            "d_seg": f"{dseg:.12f}",
            "d_pose": f"{dpose:.12f}",
            "advisory_score_formula_value": f"{score:.12f}",
            "per_role": per_role,
            "constraints": _constraint_summary(rows, margins, required),
            "cells_sha256": _sha256_array(cells),
            "pose6_sha256": _sha256_array(pose6),
            "camera_sha256": _sha256_array(camera),
            "score_claim": False,
            "evidence_axis": EVIDENCE_AXIS,
        },
        margins,
        camera,
    )


def _byte_diff(before: bytes, after: bytes) -> dict[str, Any]:
    overlap = min(len(before), len(after))
    changed = int(
        np.count_nonzero(
            np.frombuffer(before[:overlap], dtype=np.uint8) != np.frombuffer(after[:overlap], dtype=np.uint8)
        )
    )
    return {
        "before_bytes": len(before),
        "after_bytes": len(after),
        "byte_delta": len(after) - len(before),
        "changed_positions_in_common_prefix": changed,
        "before_sha256": _sha256(before),
        "after_sha256": _sha256(after),
    }


def _candidate_from_step(
    *,
    config: DDMV16CoupledJointSolveConfigV1,
    label: str,
    continuous_step: np.ndarray,
    hessian: np.ndarray,
    current_parameters: np.ndarray,
    values_shape: tuple[int, ...],
    compensation_shape: tuple[int, ...],
    phases: Mapping[str, Sequence[int]],
    problem: Mapping[str, Any],
    v14_archive: bytes,
    rows: Sequence[Mapping[str, Any]],
    operator: CouplingOperator,
    labels_all: np.ndarray,
    poses_all: np.ndarray,
    segnet: Any,
    posenet: Any,
    root: Path,
    round_index: int,
) -> dict[str, Any]:
    projection = babai_nearest_plane(continuous_step, hessian)
    step = projection.integer_step.astype(np.int64)
    patch_size = int(np.prod(values_shape))
    candidate_parameters = current_parameters.astype(np.int64) + step
    candidate_parameters[:patch_size] = np.clip(candidate_parameters[:patch_size], 0, 255)
    candidate_parameters[patch_size:] = np.clip(candidate_parameters[patch_size:], -127, 127)
    realized_step = candidate_parameters.astype(np.float64) - current_parameters
    values = candidate_parameters[:patch_size].reshape(values_shape).astype(np.uint8)
    compensation = candidate_parameters[patch_size:].reshape(compensation_shape).astype(np.int16)
    pair_ids = tuple(int(value) for value in problem["representative_source_pair_ids"])
    archive, base, program = _archive_for_state(
        v14_archive,
        values,
        pair_ids,
        phases,
        problem["sparse_compensation_support"],
        compensation,
    )
    archive_path = root / "round_candidates" / f"round_{round_index:02d}_{label}.not_a_candidate.zip.receipt-bytes"
    _publish_immutable(archive_path, archive)
    required = _required_for_rows(config, rows)
    measurement, realized_margin, camera = _dev_measurement(
        archive=archive,
        pair_ids=pair_ids,
        rows=rows,
        required=required,
        labels_all=labels_all,
        poses_all=poses_all,
        segnet=segnet,
        posenet=posenet,
    )
    predicted = predicted_margin(operator, realized_step)
    residual = realized_margin - predicted
    if float(np.std(predicted)) > 0.0 and float(np.std(realized_margin)) > 0.0:
        correlation = float(np.corrcoef(predicted, realized_margin)[0, 1])
    else:
        correlation = float("nan")
    return {
        "label": label,
        "state": {
            "template_values_u8": values.tolist(),
            "compensation_rgb_i8": compensation.tolist(),
            "phases": {key: list(value) for key, value in phases.items()},
        },
        "archive": {
            "path": _portable(archive_path),
            "bytes": len(archive),
            "sha256": _sha256(archive),
        },
        "base_archive_bytes": len(base),
        "program_bytes": len(encode_coupled_margin_program(program)),
        "placement_count": len(program.placements),
        "sparse_compensation_count": len(program.compensations),
        "projection": {
            "integer_step": projection.integer_step.tolist(),
            "quadratic_error": projection.quadratic_error,
            "covering_bound": projection.covering_bound,
            "inside_bound": projection.inside_bound,
            "basis_rank": projection.basis_rank,
        },
        "continuous_step_l2": float(np.linalg.norm(continuous_step)),
        "integer_step_l2": float(np.linalg.norm(realized_step)),
        "linearization": {
            "predicted_margin_sha256": _sha256_array(predicted),
            "realized_margin_sha256": _sha256_array(realized_margin),
            "mean_absolute_error": f"{float(np.mean(np.abs(residual))):.12f}",
            "maximum_absolute_error": f"{float(np.max(np.abs(residual), initial=0.0)):.12f}",
            "correlation": None if not math.isfinite(correlation) else f"{correlation:.12f}",
            "prediction_model": "first_order_M_with_exact_active_QP_then_uint8_Babai",
            "receiver_realized": True,
        },
        "measurement": measurement,
        "camera_sha256": _sha256_array(camera),
    }


def _measurement_key(measurement: Mapping[str, Any]) -> tuple[int, int, int, float, float]:
    constraints = measurement["constraints"]
    target = constraints["target"]
    protected = constraints["protected"]
    pose_violations = int(constraints["pose_upper"]["violations"]) + int(constraints["pose_lower"]["violations"])
    return (
        int(protected["violations"]),
        pose_violations,
        int(target["violations"]),
        float(target["debt"]),
        float(measurement["advisory_score_formula_value"]),
    )


def _bounded_phase_search(
    *,
    config: DDMV16CoupledJointSolveConfigV1,
    selected: dict[str, Any],
    problem: Mapping[str, Any],
    v14_archive: bytes,
    rows: Sequence[Mapping[str, Any]],
    labels_all: np.ndarray,
    poses_all: np.ndarray,
    segnet: Any,
    posenet: Any,
    root: Path,
    round_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    values = np.asarray(selected["state"]["template_values_u8"], dtype=np.uint8)
    compensation = np.asarray(selected["state"]["compensation_rgb_i8"], dtype=np.int16)
    phases = {key: list(value) for key, value in selected["state"]["phases"].items()}
    pair_ids = tuple(int(value) for value in problem["representative_source_pair_ids"])
    nonuniform = [
        index for index in range(values.shape[0]) if np.unique(values[index].reshape(4, 3), axis=0).shape[0] > 1
    ]
    if not nonuniform:
        return selected, {
            "reduction_status": "SEARCHED_PATTERN_SWITCH",
            "status": "NO_NONUNIFORM_TEMPLATE_AFTER_INTEGER_PROJECTION",
            "trials": [],
        }
    template_index = nonuniform[0]
    role = _STATE_BANK.templates[template_index].role if _STATE_BANK is not None else None
    eligible_pairs = [
        int(row["source_pair_id"]) for row in problem["target_cells"] if role is None or row.get("role") == role
    ]
    pair_id = eligible_pairs[0] if eligible_pairs else pair_ids[0]
    key = f"{pair_id}:{template_index}"
    current_phase = tuple(int(value) for value in phases[key])
    best = selected
    best_key = _measurement_key(selected["measurement"])
    trials = []
    for phase in ((0, 0), (0, 1), (1, 0), (1, 1)):
        if phase == current_phase:
            continue
        trial_phases = {name: list(value) for name, value in phases.items()}
        trial_phases[key] = list(phase)
        archive, base, program = _archive_for_state(
            v14_archive,
            values,
            pair_ids,
            trial_phases,
            problem["sparse_compensation_support"],
            compensation,
        )
        required = _required_for_rows(config, rows)
        measurement, _margin, camera = _dev_measurement(
            archive=archive,
            pair_ids=pair_ids,
            rows=rows,
            required=required,
            labels_all=labels_all,
            poses_all=poses_all,
            segnet=segnet,
            posenet=posenet,
        )
        trial = {
            "source_pair_id": pair_id,
            "template_index": template_index,
            "phase": list(phase),
            "archive_bytes": len(archive),
            "archive_sha256": _sha256(archive),
            "program_bytes": len(encode_coupled_margin_program(program)),
            "measurement": measurement,
            "camera_sha256": _sha256_array(camera),
        }
        trials.append(trial)
        trial_key = _measurement_key(measurement)
        if trial_key < best_key:
            archive_path = (
                root
                / "round_candidates"
                / f"round_{round_index:02d}_phase_{pair_id}_{template_index}_{phase[0]}{phase[1]}.not_a_candidate.zip.receipt-bytes"
            )
            _publish_immutable(archive_path, archive)
            best = {
                **selected,
                "label": f"{selected['label']}+phase_search",
                "state": {
                    "template_values_u8": values.tolist(),
                    "compensation_rgb_i8": compensation.tolist(),
                    "phases": trial_phases,
                },
                "archive": {"path": _portable(archive_path), "bytes": len(archive), "sha256": _sha256(archive)},
                "base_archive_bytes": len(base),
                "program_bytes": len(encode_coupled_margin_program(program)),
                "placement_count": len(program.placements),
                "sparse_compensation_count": len(program.compensations),
                "measurement": measurement,
                "camera_sha256": _sha256_array(camera),
            }
            best_key = trial_key
    return best, {
        "reduction_status": "SEARCHED_PATTERN_SWITCH",
        "status": "ENUMERATED_ONE_MEASURED_BOUNDARY_TEMPLATE_PLACEMENT",
        "source_pair_id": pair_id,
        "template_index": template_index,
        "role": role,
        "trial_count": len(trials),
        "trials": trials,
        "selected_phase": best["state"]["phases"][key],
    }


def _run_round(
    *,
    config: DDMV16CoupledJointSolveConfigV1,
    root: Path,
    problem: Mapping[str, Any],
    previous: Mapping[str, Any] | None,
    round_index: int,
    n600_cfg: DDMV15ScorerSolvedTemplateConfigV1,
    labels_all: np.ndarray,
    poses_all: np.ndarray,
    segnet: Any,
    posenet: Any,
) -> dict[str, Any]:
    path = root / "stage_checkpoints" / f"round_{round_index:02d}.json"
    if path.exists():
        return _json(path)
    values, compensation, phases = _state(problem, previous)
    v14_archive = _base_v14_bytes(n600_cfg)
    reused_unchanged_pattern = bool(previous is not None and previous["selected_label"] == "hold_control")
    if reused_unchanged_pattern:
        pair_ids = tuple(int(value) for value in problem["representative_source_pair_ids"])
        current_archive, _base, _program_value = _archive_for_state(
            v14_archive,
            values.astype(np.uint8),
            pair_ids,
            phases,
            problem["sparse_compensation_support"],
            compensation,
        )
        current_camera = receive_coupled_margin_archive(current_archive).render_camera_pairs(pair_ids)
        logits, cells, _pose = _torch_forward_full(segnet, posenet, current_camera)
        rows = _linearization_rows(problem=problem, pair_ids=pair_ids, logits=logits)
        activation_hash = hashlib.sha256(
            cells.tobytes() + rfc8785_canonicalize({key: list(value) for key, value in sorted(phases.items())})
        ).hexdigest()
        if activation_hash != previous["operator"]["activation_pattern_sha256"]:
            reused_unchanged_pattern = False
    if reused_unchanged_pattern:
        previous_npz = np.load(REPO_ROOT / previous["operator"]["path"])
        parameters = np.concatenate((values.reshape(-1), compensation.reshape(-1))).astype(np.float64)
        operator = CouplingOperator(
            matrix=np.asarray(previous_npz["M"]),
            margin=np.asarray(previous_npz["margin"]),
            required_margin=np.asarray(previous_npz["required_margin"]),
            targeted_count=sum(row["kind"] == "target" for row in rows),
            row_labels=tuple(_row_label(row) for row in rows),
            dof_labels=tuple(previous["operator"]["dof_labels"]),
            activation_pattern_sha256=previous["operator"]["activation_pattern_sha256"],
        )
        clusters = [
            {**row, "reused_unchanged_pattern": True, "reused_from_round": int(previous["round"])}
            for row in previous["operator"]["pair_clusters"]
        ]
    else:
        operator, rows, clusters, parameters, current_camera = _assemble_operator(
            config=config,
            problem=problem,
            v14_archive=v14_archive,
            values=values,
            compensation=compensation,
            phases=phases,
            segnet=segnet,
            posenet=posenet,
        )
    matrix_path = root / "operators" / f"round_{round_index:02d}_M.npz"
    _write_npz(
        matrix_path,
        M=operator.matrix,
        margin=operator.margin,
        required_margin=operator.required_margin,
        parameters=parameters,
    )
    patch_size = int(values.size)
    trust_scale = 0.5 ** (round_index - 1)
    trust = trust_scale * np.concatenate(
        (
            np.full(patch_size, config.template_trust_radius_u8, dtype=np.float64),
            np.full(compensation.size, config.compensation_trust_radius_u8, dtype=np.float64),
        )
    )
    description_metric = np.diag(
        np.concatenate(
            (
                np.ones(patch_size, dtype=np.float64),
                np.full(compensation.size, 4.0, dtype=np.float64),
            )
        )
    )
    candidates: list[dict[str, Any]] = []
    solve_receipts: dict[str, Any] = {}
    for label, use_gn in (("gauss_newton", True), ("first_order", False)):
        solved = solve_active_set_kkt(
            operator,
            description_metric=description_metric,
            row_weights=np.ones(operator.matrix.shape[0], dtype=np.float64),
            damping=1e-4,
            trust_radius=trust,
            use_gauss_newton=use_gn,
            tolerance=1e-7,
        )
        solve_receipts[label] = {
            "diagnostics": asdict(solved.diagnostics),
            "continuous_step_sha256": _sha256_array(solved.step),
            "hessian_sha256": _sha256_array(solved.hessian),
            "hessian_kind": "G_plus_MtWM_damped_Gauss_Newton"
            if use_gn
            else "description_metric_only_first_order_control",
        }
        candidates.append(
            _candidate_from_step(
                config=config,
                label=label,
                continuous_step=solved.step,
                hessian=solved.hessian,
                current_parameters=parameters,
                values_shape=values.shape,
                compensation_shape=compensation.shape,
                phases=phases,
                problem=problem,
                v14_archive=v14_archive,
                rows=rows,
                operator=operator,
                labels_all=labels_all,
                poses_all=poses_all,
                segnet=segnet,
                posenet=posenet,
                root=root,
                round_index=round_index,
            )
        )
    hold = _candidate_from_step(
        config=config,
        label="hold_control",
        continuous_step=np.zeros_like(parameters),
        hessian=np.eye(parameters.size, dtype=np.float64),
        current_parameters=parameters,
        values_shape=values.shape,
        compensation_shape=compensation.shape,
        phases=phases,
        problem=problem,
        v14_archive=v14_archive,
        rows=rows,
        operator=operator,
        labels_all=labels_all,
        poses_all=poses_all,
        segnet=segnet,
        posenet=posenet,
        root=root,
        round_index=round_index,
    )
    candidates.append(hold)
    selected = min(candidates, key=lambda row: _measurement_key(row["measurement"]))
    selected, phase_search = _bounded_phase_search(
        config=config,
        selected=selected,
        problem=problem,
        v14_archive=v14_archive,
        rows=rows,
        labels_all=labels_all,
        poses_all=poses_all,
        segnet=segnet,
        posenet=posenet,
        root=root,
        round_index=round_index,
    )
    selected_archive = _read_regular_file_once(REPO_ROOT / selected["archive"]["path"])
    current_archive = _read_regular_file_once(REPO_ROOT / hold["archive"]["path"])
    receipt = {
        "schema": ROUND_SCHEMA,
        "run_id": config.run_id,
        "round": round_index,
        "typed_config_sha256": config.typed_config_hash(),
        "operator": {
            "path": _portable(matrix_path),
            "bytes": matrix_path.stat().st_size,
            "sha256": _sha256(_read_regular_file_once(matrix_path)),
            "shape": list(operator.matrix.shape),
            "matrix_sha256": _sha256_array(operator.matrix),
            "activation_pattern_sha256": operator.activation_pattern_sha256,
            "targeted_rows": operator.targeted_count,
            "protected_plus_pose_rows": operator.matrix.shape[0] - operator.targeted_count,
            "dof_labels": list(operator.dof_labels),
            "row_labels": list(operator.row_labels),
            "pair_clusters": clusters,
            "reused_unchanged_activation_pattern": reused_unchanged_pattern,
            "trust_radius_scale_after_receiver_invalid_step": trust_scale,
            "finite_difference_all_clusters_passed": all(row["finite_difference"]["passed"] for row in clusters),
        },
        "mathematical_authority": {
            "receiver_and_R": "EXACT",
            "backprop_VJP_Jacobian": "MEASURED_AT_CURRENT_POINT_AND_FD_VALIDATED",
            "frozen_active_set_KKT_QP": "EXACT_CONDITIONAL_ON_LOCAL_M",
            "Gauss_Newton_Hessian": "MODELED_LOCAL_CURVATURE",
            "uint8_Babai_projection": "BOUNDED_NEAREST_PLANE_NOT_GLOBAL_INTEGER_OPTIMUM",
            "activation_pattern_switch": "SEARCHED_BOUNDED_ENUMERATION",
            "nonlinear_global_optimum": "NOT_CLAIMED",
        },
        "solve": solve_receipts,
        "candidates": candidates,
        "phase_search": phase_search,
        "selected_label": selected["label"],
        "selected_state": selected["state"],
        "selected_archive": selected["archive"],
        "selected_measurement": selected["measurement"],
        "selected_vs_round_input_archive_diff": _byte_diff(current_archive, selected_archive),
        "selected_vs_round_input_camera": {
            "before_sha256": _sha256_array(current_camera),
            "after_sha256": selected["camera_sha256"],
            "changed": _sha256_array(current_camera) != selected["camera_sha256"],
        },
        "relinearize_next_round": round_index < config.nonlinear_rounds,
        "resume": {
            "complete_round_checkpoint": _portable(path),
            "operator_preserved": True,
            "all_candidate_archives_preserved": True,
            "next_round_loads_selected_state": True,
        },
        "score_claim": False,
        "evidence_axis": EVIDENCE_AXIS,
    }
    _write_json(path, receipt)
    return receipt


def _window_batch_receipt(
    *,
    name: str,
    archive: bytes,
    baseline_archive: bytes,
    source_ids: np.ndarray,
    local_ids: tuple[int, ...],
    labels_all: np.ndarray,
    poses_all: np.ndarray,
    gt_f0: np.ndarray,
    gt_f1: np.ndarray,
    segnet: Any,
    posenet: Any,
) -> dict[str, Any]:
    receiver = receive_coupled_margin_archive(archive)
    baseline = receive_carrier_compose_archive(baseline_archive)
    camera = receiver.render_camera_pairs(local_ids)
    baseline_camera = baseline.render_camera_pairs(local_ids)
    cells, pose6 = _forward(segnet, posenet, camera)
    replay_cells, replay_pose = _forward(segnet, posenet, camera)
    if not np.array_equal(cells, replay_cells) or not np.array_equal(pose6, replay_pose):
        raise DirectDescriptionError(f"{name} deterministic batch replay failed")
    labels = np.asarray(labels_all[source_ids])
    poses = np.asarray(poses_all[source_ids])
    errors = cells != labels
    class_rows = {}
    for role, role_id in TARGET_ROLES.items():
        mask = labels == role_id
        class_rows[role] = {
            "errors": int(np.count_nonzero(errors & mask)),
            "sites": int(np.count_nonzero(mask)),
        }
    diff = camera.astype(np.int16) - baseline_camera.astype(np.int16)
    gt = np.stack((np.asarray(gt_f0[source_ids]), np.asarray(gt_f1[source_ids])), axis=1).astype(np.int16)
    candidate_gt = camera.astype(np.int16) - gt
    baseline_gt = baseline_camera.astype(np.int16) - gt
    return {
        "schema": "ddm_v16_receiver_batch.v1",
        "candidate": name,
        "archive_sha256": _sha256(archive),
        "source_pair_ids": source_ids.tolist(),
        "errors": int(np.count_nonzero(errors)),
        "sites": int(errors.size),
        "pose_squared_error_sum": f"{float(np.square(pose6 - poses).sum(dtype=np.float64)):.12f}",
        "pose_coordinates": int(pose6.size),
        "class_rows": class_rows,
        "cells_sha256": _sha256_array(cells),
        "pose6_sha256": _sha256_array(pose6),
        "camera_diff": {
            "changed_channel_values_vs_v15": int(np.count_nonzero(diff)),
            "changed_rgb_pixels_vs_v15": int(np.count_nonzero(np.any(diff != 0, axis=-1))),
            "l1_channel_sum_vs_v15": int(np.abs(diff).sum(dtype=np.int64)),
            "candidate_l1_channel_sum_vs_gt": int(np.abs(candidate_gt).sum(dtype=np.int64)),
            "v15_l1_channel_sum_vs_gt": int(np.abs(baseline_gt).sum(dtype=np.int64)),
            "candidate_camera_sha256": _sha256_array(camera),
            "v15_camera_sha256": _sha256_array(baseline_camera),
        },
        "score_claim": False,
        "evidence_axis": EVIDENCE_AXIS,
    }


def _measure_window(
    *,
    name: str,
    archive: bytes,
    baseline_archive: bytes,
    source_pair_ids: Sequence[int],
    local_pair_ids: Sequence[int],
    root: Path,
    config_hash: str,
    batch_size: int,
    labels_all: np.ndarray,
    poses_all: np.ndarray,
    gt_f0: np.ndarray,
    gt_f1: np.ndarray,
    segnet: Any,
    posenet: Any,
) -> dict[str, Any]:
    stage = root / "stage_checkpoints" / name
    source = np.asarray(source_pair_ids, dtype=np.int64)
    local = np.asarray(local_pair_ids, dtype=np.int64)
    if source.shape != local.shape or source.size == 0:
        raise DirectDescriptionError("v16 ladder pair geometry differs")
    for start in range(0, int(source.size), batch_size):
        stop = min(start + batch_size, int(source.size))
        checkpoint = stage / f"batch_{start:04d}_{stop:04d}.json"
        if checkpoint.exists():
            row = _json(checkpoint)
            if row["archive_sha256"] != _sha256(archive) or row["typed_config_sha256"] != config_hash:
                raise DirectDescriptionError("v16 ladder batch identity differs")
            continue
        row = _window_batch_receipt(
            name=name,
            archive=archive,
            baseline_archive=baseline_archive,
            source_ids=source[start:stop],
            local_ids=tuple(int(value) for value in local[start:stop]),
            labels_all=labels_all,
            poses_all=poses_all,
            gt_f0=gt_f0,
            gt_f1=gt_f1,
            segnet=segnet,
            posenet=posenet,
        )
        row["typed_config_sha256"] = config_hash
        _write_json(checkpoint, row)
    batches = [_json(path) for path in sorted(stage.glob("batch_*.json"))]
    expected = math.ceil(source.size / batch_size)
    if len(batches) != expected:
        raise DirectDescriptionError("v16 ladder batch coverage incomplete")
    errors = sum(int(row["errors"]) for row in batches)
    sites = sum(int(row["sites"]) for row in batches)
    pose_sse = sum(float(row["pose_squared_error_sum"]) for row in batches)
    pose_coordinates = sum(int(row["pose_coordinates"]) for row in batches)
    classes = {
        role: {
            "errors": sum(int(row["class_rows"][role]["errors"]) for row in batches),
            "sites": sum(int(row["class_rows"][role]["sites"]) for row in batches),
        }
        for role in TARGET_ROLES
    }
    for row in classes.values():
        row["d_seg"] = f"{int(row['errors']) / max(1, int(row['sites'])):.12f}"
    camera_diff = {
        key: sum(int(row["camera_diff"][key]) for row in batches)
        for key in (
            "changed_channel_values_vs_v15",
            "changed_rgb_pixels_vs_v15",
            "l1_channel_sum_vs_v15",
            "candidate_l1_channel_sum_vs_gt",
            "v15_l1_channel_sum_vs_gt",
        )
    }
    dseg = errors / sites
    dpose = pose_sse / pose_coordinates
    return {
        "candidate": name,
        "archive_bytes": len(archive),
        "archive_sha256": _sha256(archive),
        "d_seg": f"{dseg:.12f}",
        "d_pose": f"{dpose:.12f}",
        "errors": errors,
        "sites": sites,
        "per_role": classes,
        "advisory_score_formula_value": f"{100.0 * dseg + math.sqrt(10.0 * dpose) + 25.0 * len(archive) / SOURCE_BYTES:.12f}",
        "batch_count": len(batches),
        "batch_size": batch_size,
        "all_batches_checkpointed_and_preserved": True,
        "batch_digest_chain_sha256": hashlib.sha256(
            "".join(row["cells_sha256"] + row["pose6_sha256"] for row in batches).encode()
        ).hexdigest(),
        "camera_diff_vs_v15_and_gt": camera_diff,
        "byte_streams": coupled_margin_byte_rows(archive),
        "receiver_custody": dict(receive_coupled_margin_archive(archive).custody),
        "score_claim": False,
        "evidence_axis": EVIDENCE_AXIS,
    }


def _ladder_archive(
    *,
    state: Mapping[str, Any],
    problem: Mapping[str, Any],
    v14_archive: bytes,
    source_start: int,
    source_stop: int,
) -> tuple[bytes, bytes, CoupledMarginProgramV1]:
    values = np.asarray(state["template_values_u8"], dtype=np.uint8)
    all_supports = list(problem["sparse_compensation_support"])
    all_compensation = np.asarray(state["compensation_rgb_i8"], dtype=np.int16)
    selected_indexes = [
        index for index, row in enumerate(all_supports) if source_start <= int(row["source_pair_id"]) < source_stop
    ]
    supports = [all_supports[index] for index in selected_indexes]
    compensation = (
        all_compensation[np.asarray(selected_indexes, dtype=np.int64)]
        if selected_indexes
        else np.zeros((0, 3), dtype=np.int16)
    )
    pair_ids = [
        int(pair_id)
        for pair_id in problem["representative_source_pair_ids"]
        if source_start <= int(pair_id) < source_stop
    ]
    return _archive_for_state(
        v14_archive,
        values,
        pair_ids,
        state["phases"],
        supports,
        compensation,
    )


def _run_ladder_rung(
    *,
    name: Literal["eight_island", "n64", "n600"],
    config: DDMV16CoupledJointSolveConfigV1,
    root: Path,
    problem: Mapping[str, Any],
    final_round: Mapping[str, Any],
    n64_cfg: DDMV15ScorerSolvedTemplateConfigV1,
    n64_archive: bytes,
    n600_cfg: DDMV15ScorerSolvedTemplateConfigV1,
    n600_archive: bytes,
    labels_all: np.ndarray,
    poses_all: np.ndarray,
    gt_f0: np.ndarray,
    gt_f1: np.ndarray,
    segnet: Any,
    posenet: Any,
) -> dict[str, Any]:
    path = root / "stage_checkpoints" / f"ladder_{name}.json"
    if path.exists():
        return _json(path)
    state = final_round["selected_state"]
    if name == "n64":
        v14_archive = _base_v14_bytes(n64_cfg)
        candidate, nested_base, program = _ladder_archive(
            state=state,
            problem=problem,
            v14_archive=v14_archive,
            source_start=448,
            source_stop=512,
        )
        source_ids = tuple(range(448, 512))
        local_ids = tuple(range(64))
        baseline = n64_archive
    else:
        v14_archive = _base_v14_bytes(n600_cfg)
        candidate, nested_base, program = _ladder_archive(
            state=state,
            problem=problem,
            v14_archive=v14_archive,
            source_start=0,
            source_stop=600,
        )
        if name == "eight_island":
            source_ids = tuple(int(value) for value in problem["representative_source_pair_ids"])
            local_ids = source_ids
        else:
            source_ids = tuple(range(600))
            local_ids = source_ids
        baseline = n600_archive
    archive_path = root / f"ddm_v16_{name}.not_a_candidate.zip.receipt-bytes"
    _publish_immutable(archive_path, candidate)
    measurement = _measure_window(
        name=f"ladder_{name}",
        archive=candidate,
        baseline_archive=baseline,
        source_pair_ids=source_ids,
        local_pair_ids=local_ids,
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
    template_payload = encode_scorer_solved_template_bank(
        _expanded_bank(_STATE_BANK, np.asarray(state["template_values_u8"], dtype=np.uint8))
    )
    program_payload = encode_coupled_margin_program(program)
    rung = {
        "schema": LADDER_SCHEMA,
        "rung": name,
        "typed_config_sha256": config.typed_config_hash(),
        "archive": {"path": _portable(archive_path), "bytes": len(candidate), "sha256": _sha256(candidate)},
        "measurement": measurement,
        "exact_byte_accounting": {
            "shared_template_payload_bytes": len(template_payload),
            "shared_template_payload_sha256": _sha256(template_payload),
            "placement_plus_sparse_compensation_payload_bytes": len(program_payload),
            "placement_plus_sparse_compensation_payload_sha256": _sha256(program_payload),
            "placement_records": len(program.placements),
            "sparse_compensation_records": len(program.compensations),
            "nested_v15_compatible_base_bytes": len(nested_base),
            "outer_archive_bytes": len(candidate),
            "archive_vs_v15": _byte_diff(baseline, candidate),
            "nested_base_vs_v15": _byte_diff(baseline, nested_base),
            "sparse_support_density_vs_camera_rgb": f"{len(program.compensations) / max(1, len(source_ids) * 2 * CAMERA_H * CAMERA_W):.12e}",
        },
        "controls": {
            "v14_d_seg": V14_DSEG,
            "v14_Movable_d_seg": V14_MOVABLE_DSEG,
            "v14_Lane_d_seg": V14_LANE_DSEG,
            "pointer": POINTER_SCORE_TEXT,
            "comparison_axis": EVIDENCE_AXIS,
        },
        "resume": {
            "per_scorer_batch_checkpoints": True,
            "all_preserved": True,
            "complete_rung_checkpoint": _portable(path),
        },
        "score_claim": False,
        "evidence_axis": EVIDENCE_AXIS,
    }
    _write_json(path, rung)
    return rung


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _target_custody(config: DDMV16CoupledJointSolveConfigV1, root: Path) -> dict[str, Any]:
    path = root / "stage_checkpoints" / "00_target_custody.json"
    if path.exists():
        return _json(path)
    target = Path(config.target_cache_path).resolve()
    stat = target.stat()
    if stat.st_size != config.target_cache_bytes:
        raise DirectDescriptionError("v16 target cache byte count differs")
    digest = _sha256_file(target)
    if digest != config.target_cache_sha256:
        raise DirectDescriptionError("v16 target cache SHA-256 differs")
    row = {
        "path": str(target),
        "bytes": stat.st_size,
        "sha256": digest,
        "members": ["lstars", "margins", "gt_poses", "gt_f0", "gt_f1"],
        "mutated": False,
        "read_only_memmap": True,
    }
    _write_json(path, row)
    return row


def _final_receipt(
    *,
    config: DDMV16CoupledJointSolveConfigV1,
    root: Path,
    semantic_argv: Sequence[str],
    storage: Mapping[str, Any],
    target_custody: Mapping[str, Any],
    rounds: Sequence[Mapping[str, Any]],
    ladder: Sequence[Mapping[str, Any]],
) -> Path:
    path = root / "ddm_v16_coupled_joint_solve_receipt.json"
    if path.exists():
        return path
    n600 = next(row for row in ladder if row["rung"] == "n600")
    measurement = n600["measurement"]
    meaningful_dseg = float(measurement["d_seg"]) < float(V14_DSEG) - 1e-6
    gate_a = (
        float(measurement["per_role"]["Movable"]["d_seg"]) <= config.movable_gate
        and int(measurement["archive_bytes"]) <= config.archive_box_bytes
        and meaningful_dseg
    )
    fd_pass = all(bool(round_row["operator"]["finite_difference_all_clusters_passed"]) for round_row in rounds)
    correlations = [
        float(candidate["linearization"]["correlation"])
        for round_row in rounds
        for candidate in round_row["candidates"]
        if candidate["label"] != "hold_control" and candidate["linearization"]["correlation"] is not None
    ]
    linearization_invalid = not fd_pass or (bool(correlations) and min(correlations) < 0.25)
    sqp_works = False
    for round_row in rounds:
        hold = next(candidate for candidate in round_row["candidates"] if candidate["label"] == "hold_control")
        selected_debt = float(round_row["selected_measurement"]["constraints"]["target"]["debt"])
        hold_debt = float(hold["measurement"]["constraints"]["target"]["debt"])
        sqp_works |= selected_debt + 1e-9 < hold_debt
    compensation_bytes_kill = sqp_works and (
        int(measurement["archive_bytes"]) > config.archive_box_bytes or not meaningful_dseg
    )
    if gate_a:
        fork = {
            "case": "A",
            "verdict_scope": "MEASURED full-n600 macOS-CPU frozen-scorer advisory receiver ladder",
            "disposition": "FLAG_MAIN_FOR_R6_AND_CATALOG_366_LIVE_JOINT_TRAINING_NO_DISPATCH_PERFORMED",
            "main_action_required": True,
        }
    elif linearization_invalid:
        fork = {
            "case": "C",
            "verdict_scope": "INSTANCE x measured local activation patterns and configured trust radii",
            "disposition": "INSTANCE_VALIDITY_RADII_COSTATE_PRECONDITIONER_FOR_CATALOG_366",
            "main_action_required": False,
        }
    elif compensation_bytes_kill:
        fork = {
            "case": "B",
            "verdict_scope": "FORMULATION x counted sparse-compensation grammar on n600",
            "disposition": "FORMULATION_M_AND_WARM_START_FOR_CATALOG_366",
            "main_action_required": False,
        }
    else:
        fork = {
            "case": "B",
            "verdict_scope": "FORMULATION x local KKT plus uint8 nearest-plane admission",
            "disposition": "FORMULATION_M_AND_WARM_START_FOR_CATALOG_366_NO_MEANINGFUL_N600_GAIN",
            "main_action_required": False,
        }
    producer_paths = (
        REPO_ROOT / "tools/measure_ddm_v16_coupled_joint_solve.py",
        REPO_ROOT / "src/tac/optimization/coupled_margin_levelset.py",
        REPO_ROOT / "src/tac/optimization/direct_description_coupled_margin.py",
        REPO_ROOT / "src/tac/canonical_equations/ddm_v16_coupled_margin_law_20260723.py",
    )
    receipt = {
        "schema": RESULT_SCHEMA,
        "lane_id": LANE_ID,
        "run_id": config.run_id,
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
        "coupling_operator_rounds": list(rounds),
        "receiver_realized_ladder": list(ladder),
        "fork": fork,
        "conditionals": {
            "full_n600_Movable_le_0_05": float(measurement["per_role"]["Movable"]["d_seg"]) <= config.movable_gate,
            "full_n600_total_dseg_meaningfully_improves_v14": meaningful_dseg,
            "exact_archive_le_160KB": int(measurement["archive_bytes"]) <= config.archive_box_bytes,
            "sqp_reduced_measured_target_margin_debt": sqp_works,
            "finite_difference_all_clusters_passed": fd_pass,
            "linearization_invalid": linearization_invalid,
            "compensation_bytes_kill": compensation_bytes_kill,
        },
        "law_anchor": "tac.canonical_equations.ddm_v16_coupled_margin_law_20260723",
        "triality": {
            "dsl": "direct_description_coupled_margin.CoupledMarginProgramV1",
            "dag": ".omx/research/ddm_v16_coupled_joint_solve_dag_20260723.json",
            "equations": ".omx/research/ddm_v16_coupled_joint_solve_equations_20260723.json",
        },
        "target_custody": dict(target_custody),
        "storage_preflight": dict(storage),
        "resume": {
            "per_sqp_round_checkpoints": True,
            "per_scorer_batch_checkpoints": True,
            "rounds_preserved": len(rounds),
            "ladder_rungs_preserved": len(ladder),
            "all_preserved": True,
        },
        "stores_consulted": [
            "CLAUDE.md",
            "AGENTS.md",
            "PROGRAM.md",
            "docs/operating_manual_craft_handoff.md",
            ".omx/research/t5_crucible/SPEC_v75_optimal_single_trunk_20260708.md",
            ".omx/research/SPEC_v8_perclass_decomposition_20260708.md",
            config.n64_receipt_path,
            config.n64_archive_path,
            config.n600_receipt_path,
            config.n600_archive_path,
            config.target_cache_path,
            ".omx/research/joint_seg_pose_inverse_solve_20260719_codex.md",
            "reports/latest.md",
            ".omx/state/lane_registry.json",
            ".omx/state/subagent_progress.jsonl",
        ],
        "self_review_round1": {
            "performed": True,
            "found_and_fixed": [
                "separated exact conditional KKT authority from modeled nonlinear linearization",
                "kept pair-local phase as bounded combinatorial search rather than a fictitious derivative",
                "counted outer ZIP, nested base, template payload, placement payload, and sparse records separately",
                "preserved v14 lane profile as frozen inherited state instead of silently claiming it as an optimized DOF",
            ],
            "remaining_scope_limit": "shared template bytes plus measured sparse collateral support; inherited lane/worldsheet profile is frozen",
        },
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
    config: DDMV16CoupledJointSolveConfigV1,
    output_directory: Path,
    semantic_argv: Sequence[str],
) -> Path | None:
    root = output_directory.resolve()
    storage = _storage_preflight(root)
    root.mkdir(parents=True, exist_ok=True)
    target = _target_custody(config, root)
    n64_receipt, n64_cfg, n64_archive, _n600_receipt, n600_cfg, n600_archive = _v15_bindings(config)
    bank = receive_carrier_compose_archive(n600_archive).scorer_solved_templates
    if bank is None:
        raise DirectDescriptionError("v16 lost the bound v15 template bank")
    _set_state_bank(bank)
    cache = Path(config.target_cache_path)
    labels_all = open_stored_npy_memmap(cache, "lstars")
    poses_all = open_stored_npy_memmap(cache, "gt_poses")
    gt_f0 = open_stored_npy_memmap(cache, "gt_f0")
    gt_f1 = open_stored_npy_memmap(cache, "gt_f1")
    rounds = [
        _json(root / "stage_checkpoints" / f"round_{index:02d}.json")
        for index in range(1, config.nonlinear_rounds + 1)
        if (root / "stage_checkpoints" / f"round_{index:02d}.json").exists()
    ]
    ladder_names = ("eight_island", "n64", "n600")
    ladder = [
        _json(root / "stage_checkpoints" / f"ladder_{name}.json")
        for name in ladder_names
        if (root / "stage_checkpoints" / f"ladder_{name}.json").exists()
    ]
    if len(rounds) == config.nonlinear_rounds and len(ladder) == len(ladder_names):
        receipt = _final_receipt(
            config=config,
            root=root,
            semantic_argv=semantic_argv,
            storage=storage,
            target_custody=target,
            rounds=rounds,
            ladder=ladder,
        )
        print(json.dumps({"complete": True, "receipt": str(receipt)}))
        return receipt

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
    if len(rounds) < config.nonlinear_rounds:
        round_index = len(rounds) + 1
        completed = _run_round(
            config=config,
            root=root,
            problem=problem,
            previous=rounds[-1] if rounds else None,
            round_index=round_index,
            n600_cfg=n600_cfg,
            labels_all=labels_all,
            poses_all=poses_all,
            segnet=segnet,
            posenet=posenet,
        )
        print(
            json.dumps(
                {
                    "complete": False,
                    "stage": f"round_{round_index:02d}",
                    "checkpoint": completed["resume"]["complete_round_checkpoint"],
                }
            )
        )
        return None
    next_name = ladder_names[len(ladder)]
    completed = _run_ladder_rung(
        name=next_name,
        config=config,
        root=root,
        problem=problem,
        final_round=rounds[-1],
        n64_cfg=n64_cfg,
        n64_archive=n64_archive,
        n600_cfg=n600_cfg,
        n600_archive=n600_archive,
        labels_all=labels_all,
        poses_all=poses_all,
        gt_f0=gt_f0,
        gt_f1=gt_f1,
        segnet=segnet,
        posenet=posenet,
    )
    print(
        json.dumps(
            {
                "complete": False,
                "stage": f"ladder_{next_name}",
                "checkpoint": completed["resume"]["complete_rung_checkpoint"],
            }
        )
    )
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    config = DDMV16CoupledJointSolveConfigV1.model_validate_json(_read_regular_file_once(args.config))
    semantic_argv = [
        "tools/measure_ddm_v16_coupled_joint_solve.py",
        "--config",
        _portable(args.config),
        "--output-directory",
        _portable(args.output_directory),
    ]
    run(config, args.output_directory, semantic_argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
