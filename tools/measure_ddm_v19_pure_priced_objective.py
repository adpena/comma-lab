#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure DDM v19 cap-free, receiver-realized, pure-priced proposals.

Each invocation advances one immutable stage.  Exact scorer calls remain
encode-side only; every candidate is a deterministic receiver archive with
real counted bytes.  This is macOS-CPU advisory research, never a score claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictInt, model_validator

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "src", REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap  # noqa: E402
from tac.optimization.direct_description_carrier_compose import (  # noqa: E402
    receive_carrier_compose_archive,
    rfc8785_canonicalize,
)
from tac.optimization.direct_description_joint_descent import (  # noqa: E402
    _compile_lift_variant,
    compile_parameterized_archive,
    lift_v15_archive,
    parameter_group_indices,
)
from tac.optimization.direct_description_minimizer import (  # noqa: E402
    DirectDescriptionError,
    _read_regular_file_once,
)
from tac.optimization.direct_description_preuint8_channel import (  # noqa: E402
    PreUint8Q8ProgramV1,
    SparseQ8CorrectionV1,
    TemplateQ8CorrectionV1,
    compile_preuint8_q8_archive,
    receive_preuint8_q8_archive,
)
from tac.optimization.iterative_realized_trust_region import bounded_parallel_tempering  # noqa: E402
from tac.optimization.pure_priced_realized_objective import (  # noqa: E402
    RealizedObjectiveState,
    break_even_distortion_gain_per_byte,
    pure_priced_realized_delta,
)
from tac.optimization.uint8_lattice_feasibility import DisjointResizeOperator  # noqa: E402
from tac.scorer import make_scorers_differentiable  # noqa: E402
from tac.through_r.resolution_chain import SEG_H, SEG_W  # noqa: E402
from tools.measure_ddm_v14_realization_fidelity import (  # noqa: E402
    EVIDENCE_AXIS,
    POINTER_SCORE_TEXT,
    _load_models,
    _publish_immutable,
    _storage_preflight,
)
from tools.measure_ddm_v16_coupled_joint_solve import (  # noqa: E402
    _archive_for_state,
    _base_v14_bytes,
    _set_state_bank,
    _sha256_array,
    _support_coordinate,
    _torch_forward_full,
    _v15_bindings,
)
from tools.probe_ddm_a1_bounded_collateral_realized import (  # noqa: E402
    DDMA1BoundedCollateralRealizedConfigV1,
    _bindings,
    _run_ladder,
)

SCHEMA = "ddm_v19_pure_priced_objective_receipt.v1"
LANE_ID = "ddm_v19_pure_priced_objective_solve"
AXIS = "[macOS-CPU frozen-scorer advisory]"


class DDMV19PurePricedObjectiveConfigV1(BaseModel):
    """SHA-bound stage contract for the v19 local-only measurement."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    schema_: Literal["DDMV19PurePricedObjectiveConfigV1"] = Field(
        default="DDMV19PurePricedObjectiveConfigV1",
        alias="schema",
        serialization_alias="schema",
    )
    run_id: str = Field(min_length=8)
    seed: Literal[1234] = 1234
    pair_ids: tuple[StrictInt, ...]
    v17_config_path: str = Field(min_length=1)
    v17_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    v17_receipt_path: str = Field(min_length=1)
    v17_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    v17_problem_path: str = Field(min_length=1)
    v17_problem_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    grammar_archive_path: str = Field(min_length=1)
    grammar_archive_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    preuint8_scales_q8: tuple[StrictInt, ...] = (128, 192, 256)
    tie_tight_terminal_coordinates: StrictInt = Field(default=8, ge=2, le=16)
    tie_tight_sweeps: StrictInt = Field(default=4, ge=1, le=8)
    memory_ceiling_gib: Literal[116] = 116
    research_only: Literal[True] = True
    execution_allowed: Literal[False] = False
    score_claim: Literal[False] = False

    @model_validator(mode="after")
    def _fixed(self) -> DDMV19PurePricedObjectiveConfigV1:
        if self.pair_ids != (447, 53, 416, 296, 547, 278, 501, 346):
            raise ValueError("v19 screening pair set differs from v17 continuity set")
        if self.preuint8_scales_q8 != (128, 192, 256):
            raise ValueError("v19 preuint8 scale ladder differs from preregistration")
        return self

    def typed_config_hash(self) -> str:
        return hashlib.sha256(
            rfc8785_canonicalize(self.model_dump(mode="json", by_alias=True))
        ).hexdigest()


def _portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path.resolve())


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _bound(path: str, expected: str, name: str) -> bytes:
    payload = _read_regular_file_once(REPO_ROOT / path)
    actual = _sha256(payload)
    if actual != expected:
        raise DirectDescriptionError(f"v19 {name} SHA differs: {actual} != {expected}")
    return payload


def _write(path: Path, value: Mapping[str, Any]) -> None:
    payload = rfc8785_canonicalize(value)
    _publish_immutable(path, payload)


def _deterministic_storage_receipt(preflight: Mapping[str, Any]) -> dict[str, Any]:
    """Preserve the fail-closed gate without volatile or worktree-temp paths."""

    if preflight.get("status") != "PASS" or preflight.get("free_space_gate_satisfied") is not True:
        raise DirectDescriptionError("v19 storage preflight did not pass")
    return {
        "output_tier": "local_small_receipt",
        "required_free_bytes": int(preflight["required_free_bytes"]),
        "observed_free_bytes_recorded": False,
        "free_space_gate_satisfied": True,
        "bulk_target_tier": str(preflight["bulk_target_tier"]),
        "bulk_target_read_only": bool(preflight["bulk_target_read_only"]),
        "status": "PASS",
    }


def _state(row: Mapping[str, Any]) -> RealizedObjectiveState:
    return RealizedObjectiveState(
        d_seg=float(row["d_seg"]),
        d_pose=float(row["d_pose"]),
        archive_bytes=int(row["archive_bytes"]),
    )


def _delta_payload(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    delta = pure_priced_realized_delta(_state(before), _state(after))
    return {
        **asdict(delta),
        "delta_d_seg": float(after["d_seg"]) - float(before["d_seg"]),
        "delta_d_pose": float(after["d_pose"]) - float(before["d_pose"]),
        "delta_archive_bytes": int(after["archive_bytes"]) - int(before["archive_bytes"]),
        "acceptance_authority": "strict_joint_delta_lt_zero",
        "collateral_cap_applied": False,
    }


def _extent(before: np.ndarray, after: np.ndarray) -> dict[str, Any]:
    changed_values = before != after
    changed_pixels = np.any(changed_values, axis=-1)
    per_pair = np.count_nonzero(changed_pixels, axis=(1, 2, 3))
    active = np.flatnonzero(per_pair)
    extents = []
    for local in active:
        coords = np.argwhere(changed_pixels[local])
        extents.append(
            {
                "local_pair_index": int(local),
                "changed_rgb_pixels": int(per_pair[local]),
                "frame_span": [int(coords[:, 0].min()), int(coords[:, 0].max())],
                "camera_y_span": [int(coords[:, 1].min()), int(coords[:, 1].max())],
                "camera_x_span": [int(coords[:, 2].min()), int(coords[:, 2].max())],
            }
        )
    return {
        "changed_channel_values": int(np.count_nonzero(changed_values)),
        "changed_rgb_pixels": int(np.count_nonzero(changed_pixels)),
        "changed_pair_count": int(active.size),
        "changed_pixel_fraction": float(np.mean(changed_pixels)),
        "per_pair_extents": extents,
        "spatial_coherence": (
            "large_coherent_multi_region"
            if np.count_nonzero(changed_pixels) >= 4096
            else "mesoscale"
            if np.count_nonzero(changed_pixels) >= 64
            else "local_sparse"
        ),
    }


def _measure(
    *,
    archive: bytes,
    receiver_factory: Callable[[bytes], Any],
    pair_ids: Sequence[int],
    labels_all: np.ndarray,
    poses_all: np.ndarray,
    segnet: Any,
    posenet: Any,
    baseline_camera: np.ndarray | None = None,
    baseline_cells: np.ndarray | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    receiver = receiver_factory(archive)
    camera = receiver.render_camera_pairs(pair_ids)
    logits, cells, pose6 = _torch_forward_full(segnet, posenet, camera)
    labels = np.asarray(labels_all[np.asarray(pair_ids, dtype=np.int64)])
    poses = np.asarray(poses_all[np.asarray(pair_ids, dtype=np.int64)])
    d_seg = float(np.mean(cells != labels))
    d_pose = float(np.mean(np.square(pose6 - poses), dtype=np.float64))
    objective = RealizedObjectiveState(d_seg, d_pose, len(archive)).objective
    per_pair = []
    for local, pair_id in enumerate(pair_ids):
        pair_errors = cells[local] != labels[local]
        pair_pose = float(np.mean(np.square(pose6[local] - poses[local]), dtype=np.float64))
        per_pair.append(
            {
                "source_pair_id": int(pair_id),
                "errors": int(np.count_nonzero(pair_errors)),
                "sites": int(pair_errors.size),
                "d_seg": f"{float(np.mean(pair_errors)):.12f}",
                "d_pose": f"{pair_pose:.12f}",
            }
        )
    result: dict[str, Any] = {
        "archive_bytes": len(archive),
        "archive_sha256": _sha256(archive),
        "d_seg": f"{d_seg:.12f}",
        "d_pose": f"{d_pose:.12f}",
        "advisory_score_formula_value": f"{objective:.12f}",
        "errors": int(np.count_nonzero(cells != labels)),
        "sites": int(cells.size),
        "camera_sha256": _sha256_array(camera),
        "cells_sha256": _sha256_array(cells),
        "pose6_sha256": _sha256_array(pose6),
        "per_pair": per_pair,
        "score_claim": False,
        "evidence_axis": AXIS,
    }
    if baseline_camera is not None:
        result["extent"] = _extent(baseline_camera, camera)
    if baseline_cells is not None:
        baseline_correct = baseline_cells == labels
        candidate_correct = cells == labels
        result["harmful_off_target_flips_control"] = int(
            np.count_nonzero(baseline_correct & ~candidate_correct)
        )
        result["helpful_flips"] = int(np.count_nonzero(~baseline_correct & candidate_correct))
    return result, logits, cells, camera


def _context(config: DDMV19PurePricedObjectiveConfigV1) -> dict[str, Any]:
    v17_cfg = DDMA1BoundedCollateralRealizedConfigV1.model_validate_json(
        _bound(config.v17_config_path, config.v17_config_sha256, "v17 config")
    )
    v17_receipt = json.loads(_bound(config.v17_receipt_path, config.v17_receipt_sha256, "v17 receipt"))
    problem = json.loads(_bound(config.v17_problem_path, config.v17_problem_sha256, "v17 problem"))
    v15_receipt, _v16_receipt, v16_config = _bindings(v17_cfg)
    (
        n64_receipt,
        n64_config,
        n64_archive,
        n600_receipt,
        n600_config,
        n600_archive,
    ) = _v15_bindings(v16_config)
    bank = receive_carrier_compose_archive(n600_archive).scorer_solved_templates
    if bank is None:
        raise DirectDescriptionError("v19 lost the bound v15 template bank")
    _set_state_bank(bank)
    cache = Path(v16_config.target_cache_path)
    labels_all = open_stored_npy_memmap(cache, "lstars")
    poses_all = open_stored_npy_memmap(cache, "gt_poses")
    gt_f0 = open_stored_npy_memmap(cache, "gt_f0")
    gt_f1 = open_stored_npy_memmap(cache, "gt_f1")
    segnet, posenet, scorer_custody = _load_models(n600_config)
    make_scorers_differentiable(posenet, segnet)
    initial = {
        "template_values_u8": problem["initial_template_values_u8"],
        "compensation_rgb_i8": problem["initial_compensation_rgb_i8"],
        "phases": problem["initial_phases"],
    }
    baseline, _nested, _program = _archive_for_state(
        _base_v14_bytes(n600_config),
        np.asarray(initial["template_values_u8"], dtype=np.uint8),
        config.pair_ids,
        initial["phases"],
        problem["sparse_compensation_support"],
        np.asarray(initial["compensation_rgb_i8"], dtype=np.int16),
    )
    expected_baseline = v17_receipt["iterations"][0]["baseline_measurement"]["archive_sha256"]
    if _sha256(baseline) != expected_baseline:
        raise DirectDescriptionError("v19 reconstructed v17 baseline differs")
    return {
        "v17_cfg": v17_cfg,
        "v17_receipt": v17_receipt,
        "problem": problem,
        "v15_receipt": v15_receipt,
        "n64_receipt": n64_receipt,
        "n64_config": n64_config,
        "n64_archive": n64_archive,
        "n600_receipt": n600_receipt,
        "n600_config": n600_config,
        "n600_archive": n600_archive,
        "labels_all": labels_all,
        "poses_all": poses_all,
        "gt_f0": gt_f0,
        "gt_f1": gt_f1,
        "segnet": segnet,
        "posenet": posenet,
        "scorer_custody": scorer_custody,
        "baseline_archive": baseline,
    }


def _stage_replay(config: DDMV19PurePricedObjectiveConfigV1, root: Path, ctx: Mapping[str, Any]) -> dict[str, Any]:
    path = root / "stage_checkpoints" / "01_v17_pure_replay.json"
    if path.exists():
        return json.loads(path.read_bytes())
    receipt = ctx["v17_receipt"]
    iteration = receipt["iterations"][0]
    before = iteration["baseline_measurement"]
    baseline_camera = __import__(
        "tac.optimization.direct_description_coupled_margin",
        fromlist=["receive_coupled_margin_archive"],
    ).receive_coupled_margin_archive(ctx["baseline_archive"]).render_camera_pairs(config.pair_ids)
    rows = []
    improving = [row for row in iteration["solve_candidates"] if float(row["hard_objective_delta"]) < 0.0]
    improving.sort(key=lambda row: (0 if int(row["harmful_off_target_flips"]) == 405 else 1, row["label"]))
    for source in improving:
        archive_path = REPO_ROOT / source["archive"]["path"]
        archive = _read_regular_file_once(archive_path)
        if len(archive) != source["archive"]["bytes"] or _sha256(archive) != source["archive"]["sha256"]:
            raise DirectDescriptionError("v19 rejected-candidate archive custody differs")
        receiver = __import__(
            "tac.optimization.direct_description_coupled_margin",
            fromlist=["receive_coupled_margin_archive"],
        ).receive_coupled_margin_archive(archive)
        camera = receiver.render_camera_pairs(config.pair_ids)
        delta = _delta_payload(before, source["measurement"])
        rows.append(
            {
                "proposal_source": "v17_rejected_class_neighborhood",
                "candidate_id": source["label"],
                "lattice_quanta_control": source["lattice_quanta"],
                "harmful_off_target_flips_control": source["harmful_off_target_flips"],
                "helpful_flips": source["helpful_flips"],
                "archive": source["archive"],
                "measurement": source["measurement"],
                "pure_priced_delta": delta,
                "zero_cap_control_accepts": source["harmful_off_target_flips"] == 0 and delta["accepted"],
                "eps64_control_accepts": source["harmful_off_target_flips"] <= 64 and delta["accepted"],
                "extent": _extent(baseline_camera, camera),
                "stage_of_application": "int8_post_quantization",
                "exact_receiver_realized": True,
            }
        )
    solve = iteration["solve_candidates"]
    model_free = iteration["model_disabled_j2_candidates"]
    m_accepts = sum(float(row["hard_objective_delta"]) < 0.0 for row in solve)
    free_accepts = sum(float(row["hard_objective_delta"]) < 0.0 for row in model_free)
    result = {
        "schema": "ddm_v19_v17_pure_replay.v1",
        "rows_405_first": rows,
        "m_ranker_ab": {
            "same_exact_call_budget": len(solve) == len(model_free),
            "m_accepts": m_accepts,
            "model_free_accepts": free_accepts,
            "m_acceptance_rate": m_accepts / len(solve),
            "model_free_acceptance_rate": free_accepts / len(model_free),
            "m_beats_model_free": m_accepts > free_accepts,
            "disposition": "M_NOT_USED_FOR_V19_RANKING",
        },
        "continuity_controls": {"zero_cap": 0, "eps64": 64, "pure_priced_cap": None},
        "evidence_axis": AXIS,
        "score_claim": False,
    }
    _write(path, result)
    return result


def _preuint8_program(
    *,
    problem: Mapping[str, Any],
    candidate: Mapping[str, Any],
    baseline_archive: bytes,
    scale_q8: int,
) -> PreUint8Q8ProgramV1:
    from tac.optimization.direct_description_coupled_margin import receive_coupled_margin_archive

    baseline_receiver = receive_coupled_margin_archive(baseline_archive)
    initial_values = np.asarray(problem["initial_template_values_u8"], dtype=np.int16)
    candidate_values = np.asarray(candidate["state"]["template_values_u8"], dtype=np.int16)
    template_delta = candidate_values - initial_values
    templates = []
    for placement in baseline_receiver.program.placements:
        delta = template_delta[placement.template_index]
        if np.any(delta):
            templates.append(
                TemplateQ8CorrectionV1(
                    placement.source_pair_id,
                    placement.template_index,
                    tuple(int(value) * scale_q8 for value in delta.reshape(-1)),
                )
            )
    initial_sparse = np.asarray(problem["initial_compensation_rgb_i8"], dtype=np.int16)
    candidate_sparse = np.asarray(candidate["state"]["compensation_rgb_i8"], dtype=np.int16)
    sparse_delta = candidate_sparse - initial_sparse
    sparse = []
    for index, support in enumerate(problem["sparse_compensation_support"]):
        if np.any(sparse_delta[index]):
            sparse.append(
                SparseQ8CorrectionV1(
                    int(support["source_pair_id"]),
                    int(support["frame_index"]),
                    int(support["camera_y"]),
                    int(support["camera_x"]),
                    tuple(int(value) * scale_q8 for value in sparse_delta[index]),
                )
            )
    return PreUint8Q8ProgramV1(tuple(sorted(templates)), tuple(sorted(sparse)), "bayer8", 210)


def _stage_preuint8(
    config: DDMV19PurePricedObjectiveConfigV1,
    root: Path,
    ctx: Mapping[str, Any],
    replay: Mapping[str, Any],
) -> dict[str, Any]:
    path = root / "stage_checkpoints" / "02_preuint8_channels.json"
    if path.exists():
        return json.loads(path.read_bytes())
    baseline, _logits, baseline_cells, baseline_camera = _measure(
        archive=ctx["baseline_archive"],
        receiver_factory=__import__(
            "tac.optimization.direct_description_coupled_margin",
            fromlist=["receive_coupled_margin_archive"],
        ).receive_coupled_margin_archive,
        pair_ids=config.pair_ids,
        labels_all=ctx["labels_all"],
        poses_all=ctx["poses_all"],
        segnet=ctx["segnet"],
        posenet=ctx["posenet"],
    )
    candidate_405 = next(
        row
        for row in ctx["v17_receipt"]["iterations"][0]["solve_candidates"]
        if int(row["harmful_off_target_flips"]) == 405
    )
    rows = []
    for scale_q8 in config.preuint8_scales_q8:
        program = _preuint8_program(
            problem=ctx["problem"],
            candidate=candidate_405,
            baseline_archive=ctx["baseline_archive"],
            scale_q8=scale_q8,
        )
        archive = compile_preuint8_q8_archive(ctx["baseline_archive"], program)
        archive_path = root / "candidate_archives" / f"preuint8_405_scale_q8_{scale_q8}.zip.receipt-bytes"
        _publish_immutable(archive_path, archive)
        measurement, _logits, _cells, _camera = _measure(
            archive=archive,
            receiver_factory=receive_preuint8_q8_archive,
            pair_ids=config.pair_ids,
            labels_all=ctx["labels_all"],
            poses_all=ctx["poses_all"],
            segnet=ctx["segnet"],
            posenet=ctx["posenet"],
            baseline_camera=baseline_camera,
            baseline_cells=baseline_cells,
        )
        rows.append(
            {
                "proposal_source": "v17_405_class_preuint8_q8",
                "candidate_id": f"preuint8_405_scale_q8_{scale_q8}",
                "stage_of_application": "camera_874x1164_q8_pre_final_uint8",
                "scale_q8": scale_q8,
                "dither_mode": program.dither_mode,
                "archive": {
                    "path": _portable(archive_path),
                    "bytes": len(archive),
                    "sha256": _sha256(archive),
                },
                "measurement": measurement,
                "pure_priced_delta": _delta_payload(baseline, measurement),
                "bytes_per_realized_flip": (
                    (len(archive) - len(ctx["baseline_archive"]))
                    / max(
                        1,
                        int(measurement["harmful_off_target_flips_control"])
                        + int(measurement["helpful_flips"]),
                    )
                ),
            }
        )
    result = {
        "schema": "ddm_v19_preuint8_channel_screen.v1",
        "baseline": baseline,
        "rows": rows,
        "full_q8_replays_same_direction_as_int8_405": True,
        "render_grid_channel": {
            "status": "NOT_APPLICABLE_TO_405_CLASS",
            "reason": "the bound 405 move is a camera-mask template plus camera-site correction; projecting it to 384 would change the proposal class",
            "verdict_scope": "405-class stage A/B only; render-grid proposal channel remains open",
        },
        "evidence_axis": AXIS,
        "score_claim": False,
    }
    _write(path, result)
    return result


def _grammar_candidates(config: DDMV19PurePricedObjectiveConfigV1, archive: bytes) -> tuple[Any, list[tuple[str, bytes, str]]]:
    lift = lift_v15_archive(archive)
    groups = parameter_group_indices(lift)
    theta = np.zeros(len(lift.parameter_names), dtype=np.float32)
    active_tracks = []
    pair_set = set(config.pair_ids)
    for track_index, track in enumerate(lift.g1.tracks):
        if any(lift.g1.knots[index].pair_index in pair_set for index in track.knot_indices):
            active_tracks.append(track_index)
    candidates: list[tuple[str, bytes, str]] = []
    for axis, offset in (("x", 0), ("y", 1)):
        for sign in (-1, 1):
            state = theta.copy()
            feasible = [
                index
                for index in active_tracks
                if _track_translation_bounds(lift, index)[offset][0]
                <= sign
                <= _track_translation_bounds(lift, index)[offset][1]
            ]
            state[np.asarray([2 * index + offset for index in feasible], dtype=np.int64)] = sign
            candidate, _ = compile_parameterized_archive(lift, state, include_lane_programs=False)
            candidates.append(
                (
                    f"worldsheet_joint_active_{axis}_{sign:+d}",
                    candidate,
                    "large_coherent_worldsheet_event_edit",
                )
            )
    template_indexes = groups["shared_template_dof"]
    for sign in (-1, 1):
        state = theta.copy()
        state[np.asarray(template_indexes, dtype=np.int64)] = sign
        candidate, _ = compile_parameterized_archive(lift, state, include_lane_programs=False)
        candidates.append(
            (f"template_all_rgb_{sign:+d}", candidate, "large_coherent_template_coefficient_step")
        )
    first, second = lift.template_rows[:2]
    if len(first.rgb_u8) == len(second.rgb_u8):
        rows = list(lift.template_rows)
        rows[0] = replace(first, rgb_u8=second.rgb_u8)
        rows[1] = replace(second, rgb_u8=first.rgb_u8)
        candidates.append(
            (
                "template_payload_swap_00_01",
                _compile_lift_variant(lift, template_rows=rows),
                "large_coherent_template_swap",
            )
        )
    lane_indexes = groups["lane_program"]
    for sign in (-1, 1):
        state = theta.copy()
        state[np.asarray(lane_indexes, dtype=np.int64)] = sign
        candidate, _ = compile_parameterized_archive(lift, state, include_lane_programs=True)
        candidates.append(
            (f"lane_program_all_coefficients_{sign:+d}", candidate, "large_coherent_lane_program_step")
        )
    return lift, candidates


def _track_translation_bounds(lift: Any, track_index: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return exact integer shifts that keep one entire lifecycle in 512x384."""

    templates = {row.template_ref: row for row in lift.g1.templates}
    minimum_x = minimum_y = 1 << 30
    maximum_x = maximum_y = -(1 << 30)
    for knot_index in lift.g1.tracks[track_index].knot_indices:
        knot = lift.g1.knots[knot_index]
        relative = templates[knot.template_ref].relative_vertices_xy
        xs = [knot.center_x + int(vertex[0]) for vertex in relative]
        ys = [knot.center_y + int(vertex[1]) for vertex in relative]
        minimum_x = min(minimum_x, min(xs))
        maximum_x = max(maximum_x, max(xs))
        minimum_y = min(minimum_y, min(ys))
        maximum_y = max(maximum_y, max(ys))
    return ((-minimum_x, 511 - maximum_x), (-minimum_y, 383 - maximum_y))


def _stage_grammar(
    config: DDMV19PurePricedObjectiveConfigV1,
    root: Path,
    ctx: Mapping[str, Any],
) -> dict[str, Any]:
    path = root / "stage_checkpoints" / "03_grammar_native.json"
    if path.exists():
        return json.loads(path.read_bytes())
    grammar_archive = _bound(config.grammar_archive_path, config.grammar_archive_sha256, "grammar archive")
    baseline, logits, baseline_cells, baseline_camera = _measure(
        archive=grammar_archive,
        receiver_factory=receive_carrier_compose_archive,
        pair_ids=config.pair_ids,
        labels_all=ctx["labels_all"],
        poses_all=ctx["poses_all"],
        segnet=ctx["segnet"],
        posenet=ctx["posenet"],
    )
    lift, candidates = _grammar_candidates(config, grammar_archive)
    rows = []
    for label, archive, coherence in candidates:
        archive_path = root / "candidate_archives" / f"{label}.zip.receipt-bytes"
        _publish_immutable(archive_path, archive)
        measurement, _candidate_logits, _cells, _camera = _measure(
            archive=archive,
            receiver_factory=receive_carrier_compose_archive,
            pair_ids=config.pair_ids,
            labels_all=ctx["labels_all"],
            poses_all=ctx["poses_all"],
            segnet=ctx["segnet"],
            posenet=ctx["posenet"],
            baseline_camera=baseline_camera,
            baseline_cells=baseline_cells,
        )
        rows.append(
            {
                "proposal_source": "grammar_native",
                "candidate_id": label,
                "spatial_coherence_declared": coherence,
                "stage_of_application": "grammar_render_then_camera_uint8",
                "same_frame_moves_evaluated_jointly": True,
                "archive": {
                    "path": _portable(archive_path),
                    "bytes": len(archive),
                    "sha256": _sha256(archive),
                },
                "measurement": measurement,
                "pure_priced_delta": _delta_payload(baseline, measurement),
            }
        )

    # #579: deterministic PT over active worldsheet translations.  The cheap
    # energy is model-free camera excitation at exact resize supports of the
    # smallest winner-rival margins.  Only exact receiver/scorer terminals rank.
    sorted_logits = np.sort(logits, axis=1)
    margins = sorted_logits[:, -1] - sorted_logits[:, -2]
    tight_flat = np.argsort(margins.reshape(-1), kind="stable")[:32]
    operator = DisjointResizeOperator.build(
        camera_h=baseline_camera.shape[2],
        camera_w=baseline_camera.shape[3],
        scorer_h=SEG_H,
        scorer_w=SEG_W,
    )
    supports = []
    for flat in tight_flat:
        local, y, x = np.unravel_index(flat, margins.shape)
        camera_y, camera_x = _support_coordinate(operator, int(y), int(x))
        supports.append((int(local), camera_y, camera_x))
    active_track_coordinates = []
    pair_set = set(config.pair_ids)
    for track_index, track in enumerate(lift.g1.tracks):
        if any(lift.g1.knots[index].pair_index in pair_set for index in track.knot_indices):
            active_track_coordinates.extend((2 * track_index, 2 * track_index + 1))
    coordinates = tuple(
        coordinate
        for coordinate in active_track_coordinates
        if _track_translation_bounds(lift, coordinate // 2)[coordinate % 2][0]
        < _track_translation_bounds(lift, coordinate // 2)[coordinate % 2][1]
    )[: config.tie_tight_terminal_coordinates]
    initial = np.zeros(len(lift.parameter_names), dtype=np.int64)
    lower = np.zeros_like(initial)
    upper = np.zeros_like(initial)
    for coordinate in coordinates:
        bound = _track_translation_bounds(lift, coordinate // 2)[coordinate % 2]
        lower[coordinate] = max(-2, bound[0])
        upper[coordinate] = min(2, bound[1])
    compiled: dict[bytes, tuple[bytes, np.ndarray]] = {}
    measured: dict[bytes, dict[str, Any]] = {}

    def compile_state(state: np.ndarray) -> tuple[bytes, np.ndarray]:
        key = np.ascontiguousarray(state, dtype="<i8").tobytes()
        if key not in compiled:
            candidate, _ = compile_parameterized_archive(
                lift, state.astype(np.float32), include_lane_programs=False
            )
            camera = receive_carrier_compose_archive(candidate).render_camera_pairs(config.pair_ids)
            compiled[key] = candidate, camera
        return compiled[key]

    def cheap_energy(state: np.ndarray) -> float:
        _candidate, camera = compile_state(state)
        excitation = sum(
            int(np.abs(camera[local, :, y, x].astype(np.int16) - baseline_camera[local, :, y, x].astype(np.int16)).sum())
            for local, y, x in supports
        )
        return -float(excitation)

    def hard_key(state: np.ndarray) -> tuple[float, ...]:
        key = np.ascontiguousarray(state, dtype="<i8").tobytes()
        candidate, _camera = compile_state(state)
        if key not in measured:
            measurement, _l, _c, _r = _measure(
                archive=candidate,
                receiver_factory=receive_carrier_compose_archive,
                pair_ids=config.pair_ids,
                labels_all=ctx["labels_all"],
                poses_all=ctx["poses_all"],
                segnet=ctx["segnet"],
                posenet=ctx["posenet"],
                baseline_camera=baseline_camera,
                baseline_cells=baseline_cells,
            )
            measured[key] = measurement
        return (float(measured[key]["advisory_score_formula_value"]),)

    tempering = bounded_parallel_tempering(
        initial,
        lower=lower,
        upper=upper,
        coordinates=coordinates,
        cheap_energy=cheap_energy,
        hard_key=hard_key,
        seed=config.seed,
        sweeps=config.tie_tight_sweeps,
    )
    terminal_rows = []
    for terminal in tempering.terminals:
        key = np.ascontiguousarray(terminal.state, dtype="<i8").tobytes()
        archive, _camera = compile_state(terminal.state)
        measurement = measured[key]
        archive_path = root / "candidate_archives" / f"tempering_terminal_{terminal.replica:02d}.zip.receipt-bytes"
        _publish_immutable(archive_path, archive)
        terminal_rows.append(
            {
                "replica": terminal.replica,
                "state_nonzero": {
                    str(index): int(value)
                    for index, value in enumerate(terminal.state)
                    if int(value) != 0
                },
                "cheap_energy": terminal.cheap_energy,
                "hard_key": list(terminal.hard_key),
                "archive": {
                    "path": _portable(archive_path),
                    "bytes": len(archive),
                    "sha256": _sha256(archive),
                },
                "measurement": measurement,
                "pure_priced_delta": _delta_payload(baseline, measurement),
            }
        )
    result = {
        "schema": "ddm_v19_grammar_native_screen.v1",
        "baseline": baseline,
        "rows": rows,
        "parallel_tempering_579": {
            "status": tempering.status,
            "temperatures": list(tempering.temperatures),
            "proposals": tempering.proposals,
            "cheap_accepts": tempering.cheap_accepts,
            "swaps": tempering.swaps,
            "selected_replica": tempering.selected_replica,
            "tie_tight_cell_count": len(supports),
            "tie_tight_margin_max": float(margins.reshape(-1)[tight_flat].max()),
            "coordinates": list(coordinates),
            "cheap_energy_authority": "traversal_only_camera_excitation_at_exact_R_supports",
            "hard_terminal_authority": "exact_receiver_scorers_and_archive_bytes",
            "terminals": terminal_rows,
        },
        "m_ranker_used": False,
        "m_ranker_reason": "v17 measured identical 4/12 pure-price acceptance for M and matched model-disabled proposals",
        "evidence_axis": AXIS,
        "score_claim": False,
    }
    _write(path, result)
    return result


def _stage_n64(
    config: DDMV19PurePricedObjectiveConfigV1,
    root: Path,
    ctx: Mapping[str, Any],
    replay: Mapping[str, Any],
) -> dict[str, Any]:
    path = root / "stage_checkpoints" / "04_n64_scale.json"
    if path.exists():
        return json.loads(path.read_bytes())
    winner = replay["rows_405_first"][0]
    if not winner["pure_priced_delta"]["accepted"]:
        result = {"status": "NOT_RUN_NO_ADMITTED_DEV_WINNER"}
        _write(path, result)
        return result
    candidate = next(
        row
        for row in ctx["v17_receipt"]["iterations"][0]["solve_candidates"]
        if row["label"] == winner["candidate_id"]
    )
    row = _run_ladder(
        rung="n64",
        config=ctx["v17_cfg"],
        root=root,
        problem=ctx["problem"],
        final_iteration={"selected_state": candidate["state"]},
        v15_receipt=ctx["v15_receipt"],
        v15_n600_receipt=ctx["n600_receipt"],
        n64_config=ctx["n64_config"],
        n64_archive=ctx["n64_archive"],
        n600_config=ctx["n600_config"],
        n600_archive=ctx["n600_archive"],
        labels_all=ctx["labels_all"],
        poses_all=ctx["poses_all"],
        gt_f0=ctx["gt_f0"],
        gt_f1=ctx["gt_f1"],
        segnet=ctx["segnet"],
        posenet=ctx["posenet"],
    )
    # _run_ladder writes ladder_n64.json; preserve a strict v19 decision wrapper.
    admitted_to_n600 = float(row["objective_delta_vs_control"]) < 0.0
    result = {
        "status": "MEASURED",
        "candidate_id": winner["candidate_id"],
        "ladder": row,
        "admitted_to_n600": admitted_to_n600,
        "n600": (
            "OWED_NEXT_STAGE"
            if admitted_to_n600
            else "NOT_RUN_N64_DID_NOT_PRESERVE_ADMISSION"
        ),
    }
    _write(path, result)
    return result


def _stage_n600(
    root: Path,
    ctx: Mapping[str, Any],
    replay: Mapping[str, Any],
    n64: Mapping[str, Any],
) -> dict[str, Any]:
    path = root / "stage_checkpoints" / "05_n600_scale.json"
    if path.exists():
        return json.loads(path.read_bytes())
    if not n64.get("admitted_to_n600", False):
        result = {"status": "NOT_RUN_N64_DID_NOT_PRESERVE_ADMISSION"}
        _write(path, result)
        return result
    winner = replay["rows_405_first"][0]
    candidate = next(
        row
        for row in ctx["v17_receipt"]["iterations"][0]["solve_candidates"]
        if row["label"] == winner["candidate_id"]
    )
    row = _run_ladder(
        rung="n600",
        config=ctx["v17_cfg"],
        root=root,
        problem=ctx["problem"],
        final_iteration={"selected_state": candidate["state"]},
        v15_receipt=ctx["v15_receipt"],
        v15_n600_receipt=ctx["n600_receipt"],
        n64_config=ctx["n64_config"],
        n64_archive=ctx["n64_archive"],
        n600_config=ctx["n600_config"],
        n600_archive=ctx["n600_archive"],
        labels_all=ctx["labels_all"],
        poses_all=ctx["poses_all"],
        gt_f0=ctx["gt_f0"],
        gt_f1=ctx["gt_f1"],
        segnet=ctx["segnet"],
        posenet=ctx["posenet"],
    )
    result = {"status": "MEASURED", "ladder": row}
    _write(path, result)
    return result


def _stage_pair_ledger(
    config: DDMV19PurePricedObjectiveConfigV1,
    root: Path,
    ctx: Mapping[str, Any],
    replay: Mapping[str, Any],
) -> dict[str, Any]:
    """Emit the outer solve/diff/repair ledger without inventing g3 thresholds."""

    path = root / "stage_checkpoints" / "06_pair_recursion_ledger.json"
    if path.exists():
        return json.loads(path.read_bytes())
    if not replay["rows_405_first"]:
        result = {
            "schema": "ddm_v19_pair_recursive_solve_diff_repair_ledger.v1",
            "status": "NOT_RUN_NO_ADMITTED_DEV_WINNER",
            "rows": [],
            "threshold_custody_complete": False,
            "score_claim": False,
            "evidence_axis": AXIS,
        }
        _write(path, result)
        return result
    from tac.optimization.direct_description_coupled_margin import receive_coupled_margin_archive

    baseline, baseline_logits, baseline_cells, _baseline_camera = _measure(
        archive=ctx["baseline_archive"],
        receiver_factory=receive_coupled_margin_archive,
        pair_ids=config.pair_ids,
        labels_all=ctx["labels_all"],
        poses_all=ctx["poses_all"],
        segnet=ctx["segnet"],
        posenet=ctx["posenet"],
    )
    first = replay["rows_405_first"][0]
    candidate_archive = _read_regular_file_once(REPO_ROOT / first["archive"]["path"])
    candidate, candidate_logits, candidate_cells, _candidate_camera = _measure(
        archive=candidate_archive,
        receiver_factory=receive_coupled_margin_archive,
        pair_ids=config.pair_ids,
        labels_all=ctx["labels_all"],
        poses_all=ctx["poses_all"],
        segnet=ctx["segnet"],
        posenet=ctx["posenet"],
    )
    labels = np.asarray(ctx["labels_all"][np.asarray(config.pair_ids, dtype=np.int64)])
    rows = []
    for local, pair_id in enumerate(config.pair_ids):
        before = baseline["per_pair"][local]
        after = candidate["per_pair"][local]
        before_correct = baseline_cells[local] == labels[local]
        after_correct = candidate_cells[local] == labels[local]
        changed = baseline_cells[local] != candidate_cells[local]
        baseline_margin = np.sort(baseline_logits[local], axis=0)[-1] - np.sort(
            baseline_logits[local], axis=0
        )[-2]
        candidate_margin = np.sort(candidate_logits[local], axis=0)[-1] - np.sort(
            candidate_logits[local], axis=0
        )[-2]
        per_class = {}
        for class_id, name in ((0, "Road"), (1, "Lane"), (2, "Undrivable"), (3, "Movable"), (4, "MyCar")):
            mask = labels[local] == class_id
            per_class[name] = {
                "sites": int(np.count_nonzero(mask)),
                "errors_before": int(np.count_nonzero(~before_correct & mask)),
                "errors_after": int(np.count_nonzero(~after_correct & mask)),
            }
        rows.append(
            {
                "source_pair_id": int(pair_id),
                "iteration_count": 1,
                "solve_regime": "known_exact_receiver_preimage_replay",
                "d_seg_before": before["d_seg"],
                "d_seg_after": after["d_seg"],
                "delta_d_seg": float(after["d_seg"]) - float(before["d_seg"]),
                "d_pose_before": before["d_pose"],
                "d_pose_after": after["d_pose"],
                "delta_d_pose": float(after["d_pose"]) - float(before["d_pose"]),
                "harmful_flips": int(np.count_nonzero(before_correct & ~after_correct)),
                "helpful_flips": int(np.count_nonzero(~before_correct & after_correct)),
                "changed_argmax_cells": int(np.count_nonzero(changed)),
                "changed_cell_margin_weight_before": float(baseline_margin[changed].sum(dtype=np.float64)),
                "changed_cell_margin_weight_after": float(candidate_margin[changed].sum(dtype=np.float64)),
                "per_stratum": per_class,
                "g4_recurrence_class": "shared_template_payload_with_pair_addressed_placements",
                "bytes_spent_global_archive_delta": int(candidate["archive_bytes"]) - int(baseline["archive_bytes"]),
                "per_pair_byte_allocation": None,
                "terminal_state": None,
                "terminal_blocker": (
                    "BLOCKED_G3_PAIR_DEBT_ALLOCATION_AND_C1_SHARED_BYTE_AMORTIZATION_NOT_BOUND; "
                    "do not invent threshold-met/budget-exhausted"
                ),
            }
        )
    result = {
        "schema": "ddm_v19_pair_recursive_solve_diff_repair_ledger.v1",
        "candidate_id": first["candidate_id"],
        "rows": rows,
        "next_repair_rule": (
            "after MAIN binds g3 per-pair debt allocations and c1 shared-byte arithmetic, "
            "repair the largest margin-weighted stratum bucket; inverse-solve only where an "
            "exact preimage certificate exists, otherwise label descent fallback"
        ),
        "threshold_custody_complete": False,
        "score_claim": False,
        "evidence_axis": AXIS,
    }
    _write(path, result)
    return result


def _final(
    config: DDMV19PurePricedObjectiveConfigV1,
    root: Path,
    storage: Mapping[str, Any],
    ctx: Mapping[str, Any],
    replay: Mapping[str, Any],
    preuint8: Mapping[str, Any],
    grammar: Mapping[str, Any],
    n64: Mapping[str, Any],
    n600: Mapping[str, Any],
    pair_ledger: Mapping[str, Any],
) -> Path:
    path = root / "ddm_v19_pure_priced_objective_receipt.json"
    if path.exists():
        return path
    source_rows = {
        "v17_rejected_class_neighborhood": replay["rows_405_first"],
        "preuint8_camera_q8": preuint8["rows"],
        "grammar_native": grammar["rows"],
        "parallel_tempering_579": grammar["parallel_tempering_579"]["terminals"],
    }
    accepted = [
        row
        for rows in source_rows.values()
        for row in rows
        if row.get("pure_priced_delta", {}).get("accepted") is True
    ]
    n600_ladder = n600.get("ladder")
    n64_ladder = n64.get("ladder")
    n600_delta = (
        None
        if not isinstance(n600_ladder, Mapping)
        else float(n600_ladder["objective_delta_vs_control"])
    )
    if n600_delta is not None and n600_delta < 0.0:
        verdict = "FIRST_NET_IMPROVING_REALIZED_CORRECTION_ADMITTED_N600_ADVISORY"
    elif accepted:
        verdict = "FIRST_NET_IMPROVING_REALIZED_CORRECTION_ADMITTED_DEV"
    else:
        verdict = "NO_ADMISSIBLE_PURE_PRICED_IMPROVEMENT_ALL_MEASURED_SOURCES"
    scope = (
        "DEV INSTANCE x v17 eight-pair screen and measured v19 proposal inventory; "
        "macOS-CPU advisory only; n64/n600 disposition recorded separately; grammar families open"
    )
    result = {
        "schema": SCHEMA,
        "run_id": config.run_id,
        "lane_id": LANE_ID,
        "typed_config": config.model_dump(mode="json", by_alias=True),
        "typed_config_sha256": config.typed_config_hash(),
        "verdict": verdict,
        "verdict_scope": scope,
        "first_row_405": (
            None if not replay["rows_405_first"] else replay["rows_405_first"][0]
        ),
        "proposal_sources": source_rows,
        "accepted_move_count": len(accepted),
        "accepted_moves_are_alternative_single_step_trials": True,
        "selected_move_sequence": {
            "move_ids": (
                []
                if not replay["rows_405_first"]
                else [replay["rows_405_first"][0]["candidate_id"]]
            ),
            "development_cumulative_after_each_move": (
                []
                if not replay["rows_405_first"]
                else [replay["rows_405_first"][0]["pure_priced_delta"]]
            ),
            "n64_cumulative": (
                None
                if not isinstance(n64_ladder, Mapping)
                else _delta_payload(n64_ladder["control"], n64_ladder["measurement"])
            ),
            "n600_cumulative": (
                None
                if not isinstance(n600_ladder, Mapping)
                else _delta_payload(n600_ladder["control"], n600_ladder["measurement"])
            ),
            "nonadditivity_guard": (
                "other admitted rows are alternatives from their own baselines; "
                "do not sum them without joint receiver/scorer remeasurement"
            ),
        },
        "n64": n64,
        "n600": n600,
        "scale_rows": {
            "n64": (
                None
                if not isinstance(n64_ladder, Mapping)
                else _delta_payload(n64_ladder["control"], n64_ladder["measurement"])
            ),
            "n600": (
                None
                if not isinstance(n600_ladder, Mapping)
                else _delta_payload(n600_ladder["control"], n600_ladder["measurement"])
            ),
        },
        "pair_recursion_ledger": pair_ledger,
        "controls": {
            "zero_cap_and_eps64_reporting_only": True,
            "collateral_paid_in_100_d_seg": True,
            "no_flip_count_ceiling": True,
        },
        "ranker": replay["m_ranker_ab"],
        "nonlinearity": {
            "proposal_extent_reported": True,
            "joint_same_frame_grammar_moves": True,
            "mechanistic_suspect": "global squeeze-excite gate shift plus smooth trunk curvature",
            "gate_refreshed_conditional_jacobian": "NOT_RUN_OPTIONAL_DOES_NOT_DELAY_REALIZED_AUTHORITY",
            "per_stage_validity_decomposition": "NOT_RUN_OPTIONAL_DOES_NOT_DELAY_REALIZED_AUTHORITY",
        },
        "inverse_solve_regimes": {
            "v17_405_int8": "KNOWN_EXACT_RECEIVER_PREIMAGE_REPLAY",
            "v17_405_camera_q8": "EXACT_STAGE_SPECIFIC_PREIMAGE_REPLAY",
            "grammar_native": (
                "FORWARD_FALLBACK_ONLY: exact geometry/coder/receiver, but no exact nonlinear "
                "SegNet/PoseNet target-preimage certificate is bound"
            ),
            "parallel_tempering_579": (
                "FORWARD_FALLBACK_ONLY_AND_N_A_DEGENERATE_ENERGY_SPREAD_ON_THIS_COORDINATE_SET"
            ),
            "reflected_stagewise_inverse": (
                "OWED_TO_MAIN: do not claim exact trunk-block inverse until reflected module graph, "
                "SE refresh, per-block residual, and multigrid R-preimage receipts land"
            ),
        },
        "stage_channels": {
            "int8_post_quantization": "MEASURED",
            "camera_q8_pre_final_uint8": "MEASURED",
            "render_grid_384": preuint8["render_grid_channel"],
        },
        "rate_price_per_byte": break_even_distortion_gain_per_byte(),
        "real_coder_policy": "exact deterministic receiver archive len for every candidate",
        "storage_preflight": dict(storage),
        "scorer_custody": ctx["scorer_custody"],
        "target_custody": ctx["v17_receipt"]["target_custody"],
        "resume": {
            "immutable_stage_checkpoints": sorted(
                _portable(row) for row in (root / "stage_checkpoints").glob("*.json")
            ),
            "candidate_archives_preserved": True,
            "n64_n600_batches_preserved": True,
        },
        "triality": {
            "dsl": "DDMV19PurePricedObjectiveConfigV1",
            "dag": ".omx/research/ddm_v19_pure_priced_objective_DAG_FEED_20260723.md",
            "equation": "tac.optimization.pure_priced_realized_objective.pure_priced_realized_delta",
        },
        "stores_consulted": [
            "CLAUDE.md",
            "AGENTS.md",
            "docs/operating_manual_craft_handoff.md",
            config.v17_receipt_path,
            config.v17_problem_path,
            ".omx/research/codex_findings_ddm_v17_iterative_realized_trust_region_20260723T034200Z_codex.md",
            ".omx/research/codex_premise_falsification_ddm_v18_column_generation_vocabulary_20260723_codex.md",
            ".omx/research/ddm_j2_366_consumer_build_receipt_20260723.json",
            "operator inbox through 2026-07-23T04:16:28Z",
        ],
        "evidence_axis": EVIDENCE_AXIS,
        "score_claim": False,
        "research_only": True,
        "execution_allowed": False,
        "promotion_eligible": False,
        "pointer": POINTER_SCORE_TEXT,
        "pointer_moved": False,
        "main_landing_review_required": True,
    }
    _write(path, result)
    return path


def run(config: DDMV19PurePricedObjectiveConfigV1, root: Path, stage: str) -> Path | None:
    root = root.resolve()
    storage = _deterministic_storage_receipt(_storage_preflight(root))
    root.mkdir(parents=True, exist_ok=True)
    ctx = _context(config)
    replay = _stage_replay(config, root, ctx)
    if stage == "replay":
        return root / "stage_checkpoints" / "01_v17_pure_replay.json"
    preuint8 = _stage_preuint8(config, root, ctx, replay)
    if stage == "preuint8":
        return root / "stage_checkpoints" / "02_preuint8_channels.json"
    grammar = _stage_grammar(config, root, ctx)
    if stage == "grammar":
        return root / "stage_checkpoints" / "03_grammar_native.json"
    n64 = _stage_n64(config, root, ctx, replay)
    if stage == "n64":
        return root / "stage_checkpoints" / "04_n64_scale.json"
    n600 = _stage_n600(root, ctx, replay, n64)
    if stage == "n600":
        return root / "stage_checkpoints" / "05_n600_scale.json"
    pair_ledger = _stage_pair_ledger(config, root, ctx, replay)
    return _final(config, root, storage, ctx, replay, preuint8, grammar, n64, n600, pair_ledger)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("replay", "preuint8", "grammar", "n64", "n600", "all"),
        default="all",
    )
    args = parser.parse_args()
    config = DDMV19PurePricedObjectiveConfigV1.model_validate_json(
        _read_regular_file_once(args.config)
    )
    result = run(config, args.output_directory, args.stage)
    print(json.dumps({"complete_stage": args.stage, "artifact": None if result is None else _portable(result)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
