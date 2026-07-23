#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Greedily compose DDM v19 winners under exact joint remeasurement.

The v19 rows were alternative single-step trials.  This runner starts from the
405-flip compact-int8 winner and admits every later move only after compiling
the entire current stack into one receiver archive and replaying both frozen
scorers.  Every result remains macOS-CPU advisory research, never a score claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "src", REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.optimization.direct_description_carrier_compose import (  # noqa: E402
    REALIZATION_STATIC_RULE_MEMBER,
    SCORER_SOLVED_TEMPLATE_MEMBER,
    WORLDSHEET_G1_MEMBER,
    compile_carrier_compose_archive,
    encode_scorer_solved_template_bank,
    parse_carrier_compose_archive,
    receive_carrier_compose_archive,
    rfc8785_canonicalize,
)
from tac.optimization.direct_description_coupled_margin import (  # noqa: E402
    compile_coupled_margin_archive,
    coupled_margin_byte_rows,
    receive_coupled_margin_archive,
)
from tac.optimization.direct_description_g1_worldsheet import (  # noqa: E402
    encode_lifted_g1_movable_worldsheet,
    lift_g1_movable_worldsheet,
)
from tac.optimization.direct_description_minimizer import (  # noqa: E402
    SOURCE_BYTES,
    DirectDescriptionError,
    _read_regular_file_once,
)
from tac.optimization.direct_description_preuint8_channel import (  # noqa: E402
    PreUint8Q8ProgramV1,
    SparseQ8CorrectionV1,
    TemplateQ8CorrectionV1,
    compile_preuint8_q8_archive,
    parse_preuint8_q8_archive,
    receive_preuint8_q8_archive,
)
from tac.optimization.pure_priced_realized_objective import (  # noqa: E402
    break_even_distortion_gain_per_byte,
)
from tools.measure_ddm_v14_realization_fidelity import (  # noqa: E402
    POINTER_SCORE_TEXT,
    _forward,
    _publish_immutable,
    _storage_preflight,
)
from tools.measure_ddm_v16_coupled_joint_solve import (  # noqa: E402
    _base_v14_bytes,
    _ladder_archive,
    _sha256_array,
)
from tools.measure_ddm_v19_pure_priced_objective import (  # noqa: E402
    AXIS,
    DDMV19PurePricedObjectiveConfigV1,
    _context,
    _delta_payload,
    _deterministic_storage_receipt,
    _measure,
    _portable,
    _sha256,
    _write,
)
from tools.probe_ddm_a1_bounded_collateral_realized import _source_control  # noqa: E402

SCHEMA = "ddm_v19b_joint_remeasure_stack_receipt.v1"
LANE_ID = "ddm_v19b_joint_remeasure_stack"
FIRST_MOVE_ID = "1x1_rowband_control_solve_02_M_preconditioned_ranked_prefix_r4"
ROLE_CLASS_IDS = frozenset({1, 3})
CLASS_NAMES = {
    0: "Road",
    1: "Lane",
    2: "Undrivable",
    3: "Movable",
    4: "MyCar",
}
TARGET_MAX_ERRORS = 136_839
C1_DOWNSTREAM_BUDGET_BYTES = 16_384


class DDMV19BJointRemeasureStackConfigV1(BaseModel):
    """SHA-bound local-only contract for the v19b composition measurement."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    schema_: Literal["DDMV19BJointRemeasureStackConfigV1"] = Field(
        default="DDMV19BJointRemeasureStackConfigV1",
        alias="schema",
        serialization_alias="schema",
    )
    run_id: str = Field(min_length=8)
    seed: Literal[1234] = 1234
    v19_config_path: str = Field(min_length=1)
    v19_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    v19_receipt_path: str = Field(min_length=1)
    v19_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scorer_batch_size: Literal[16] = 16
    greedy_order: Literal["forced_405_then_single_step_joint_delta"] = (
        "forced_405_then_single_step_joint_delta"
    )
    post_int8_merge: Literal["add_candidate_delta_then_clip_wire_domain"] = (
        "add_candidate_delta_then_clip_wire_domain"
    )
    grammar_merge: Literal["add_integer_track_translation"] = (
        "add_integer_track_translation"
    )
    preuint8_merge: Literal["sum_q8_before_one_final_uint8"] = (
        "sum_q8_before_one_final_uint8"
    )
    memory_ceiling_gib: Literal[116] = 116
    research_only: Literal[True] = True
    execution_allowed: Literal[False] = False
    score_claim: Literal[False] = False

    @model_validator(mode="after")
    def _sealed_contract(self) -> DDMV19BJointRemeasureStackConfigV1:
        if self.scorer_batch_size != 16:
            raise ValueError("v19b batch size must preserve the measured batch-16 scorer path")
        return self

    def typed_config_hash(self) -> str:
        return hashlib.sha256(
            rfc8785_canonicalize(self.model_dump(mode="json", by_alias=True))
        ).hexdigest()


@dataclass(frozen=True, slots=True)
class StackState:
    """One additive proposal state before receiver-domain clipping."""

    template_delta: np.ndarray
    compensation_delta: np.ndarray
    grammar_move_ids: tuple[str, ...] = ()
    preuint8_scale_q8: int = 0
    move_ids: tuple[str, ...] = ()


def _initial_stack(problem: Mapping[str, Any]) -> StackState:
    return StackState(
        np.zeros_like(np.asarray(problem["initial_template_values_u8"], dtype=np.int16)),
        np.zeros_like(np.asarray(problem["initial_compensation_rgb_i8"], dtype=np.int16)),
    )


def _bound_json(path: str, expected_sha256: str, label: str) -> dict[str, Any]:
    payload = _read_regular_file_once(REPO_ROOT / path)
    actual = _sha256(payload)
    if actual != expected_sha256:
        raise DirectDescriptionError(
            f"v19b {label} SHA differs: {actual} != {expected_sha256}"
        )
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DirectDescriptionError(f"v19b {label} is not strict JSON") from exc
    if not isinstance(value, dict):
        raise DirectDescriptionError(f"v19b {label} is not a JSON object")
    return value


def _load_sources(
    config: DDMV19BJointRemeasureStackConfigV1,
) -> tuple[DDMV19PurePricedObjectiveConfigV1, dict[str, Any], dict[str, Any]]:
    v19_payload = _read_regular_file_once(REPO_ROOT / config.v19_config_path)
    if _sha256(v19_payload) != config.v19_config_sha256:
        raise DirectDescriptionError("v19b source v19 config SHA differs")
    v19_config = DDMV19PurePricedObjectiveConfigV1.model_validate_json(v19_payload)
    receipt = _bound_json(
        config.v19_receipt_path, config.v19_receipt_sha256, "source v19 receipt"
    )
    if receipt.get("schema") != "ddm_v19_pure_priced_objective_receipt.v1":
        raise DirectDescriptionError("v19b source receipt schema differs")
    if receipt.get("typed_config_sha256") != v19_config.typed_config_hash():
        raise DirectDescriptionError("v19b source receipt/config binding differs")
    if receipt.get("score_claim") is not False or receipt.get("pointer_moved") is not False:
        raise DirectDescriptionError("v19b source receipt has false-authority drift")
    ctx = _context(v19_config)
    return v19_config, receipt, ctx


def _accepted_inventory(receipt: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family, family_rows in receipt["proposal_sources"].items():
        for source in family_rows:
            delta = source.get("pure_priced_delta", {})
            candidate_id = source.get("candidate_id")
            if delta.get("accepted") is True and isinstance(candidate_id, str):
                rows.append(
                    {
                        "candidate_id": candidate_id,
                        "family": family,
                        "single_step_delta": dict(delta),
                        "source_archive": dict(source["archive"]),
                    }
                )
    first = [row for row in rows if row["candidate_id"] == FIRST_MOVE_ID]
    if len(first) != 1:
        raise DirectDescriptionError("v19b source inventory lost the unique 405 winner")
    remaining = [row for row in rows if row["candidate_id"] != FIRST_MOVE_ID]
    remaining.sort(
        key=lambda row: (
            float(row["single_step_delta"]["joint_delta"]),
            str(row["candidate_id"]),
        )
    )
    result = [first[0], *remaining]
    if len(result) != 10:
        raise DirectDescriptionError(
            f"v19b expected 10 admitted v19 alternatives, found {len(result)}"
        )
    return result


def _candidate_state(ctx: Mapping[str, Any], candidate_id: str) -> Mapping[str, Any]:
    matches = [
        row["state"]
        for row in ctx["v17_receipt"]["iterations"][0]["solve_candidates"]
        if row["label"] == candidate_id
    ]
    if len(matches) != 1:
        raise DirectDescriptionError(
            f"v19b post-int8 candidate state is not unique: {candidate_id}"
        )
    return matches[0]


def _apply_move(
    state: StackState,
    move: Mapping[str, Any],
    *,
    problem: Mapping[str, Any],
    ctx: Mapping[str, Any],
) -> StackState:
    candidate_id = str(move["candidate_id"])
    family = str(move["family"])
    template_delta = state.template_delta.copy()
    compensation_delta = state.compensation_delta.copy()
    grammar_move_ids = state.grammar_move_ids
    preuint8_scale_q8 = state.preuint8_scale_q8
    if family == "v17_rejected_class_neighborhood":
        candidate = _candidate_state(ctx, candidate_id)
        template_delta += np.asarray(
            candidate["template_values_u8"], dtype=np.int16
        ) - np.asarray(problem["initial_template_values_u8"], dtype=np.int16)
        compensation_delta += np.asarray(
            candidate["compensation_rgb_i8"], dtype=np.int16
        ) - np.asarray(problem["initial_compensation_rgb_i8"], dtype=np.int16)
    elif family == "grammar_native":
        if not (
            candidate_id.startswith("worldsheet_joint_active_x_")
            or candidate_id.startswith("worldsheet_joint_active_y_")
        ):
            raise DirectDescriptionError(
                f"v19b admitted unsupported grammar move: {candidate_id}"
            )
        grammar_move_ids = (*grammar_move_ids, candidate_id)
    elif family == "preuint8_camera_q8":
        prefix = "preuint8_405_scale_q8_"
        if not candidate_id.startswith(prefix):
            raise DirectDescriptionError(
                f"v19b admitted unsupported preuint8 move: {candidate_id}"
            )
        preuint8_scale_q8 += int(candidate_id.removeprefix(prefix))
    else:
        raise DirectDescriptionError(f"v19b admitted unsupported family: {family}")
    return StackState(
        template_delta,
        compensation_delta,
        grammar_move_ids,
        preuint8_scale_q8,
        (*state.move_ids, candidate_id),
    )


def _realized_post_state(
    stack: StackState, problem: Mapping[str, Any]
) -> dict[str, Any]:
    templates = np.clip(
        np.asarray(problem["initial_template_values_u8"], dtype=np.int16)
        + stack.template_delta,
        0,
        255,
    ).astype(np.uint8)
    compensations = np.clip(
        np.asarray(problem["initial_compensation_rgb_i8"], dtype=np.int16)
        + stack.compensation_delta,
        -127,
        127,
    ).astype(np.int16)
    return {
        "template_values_u8": templates.tolist(),
        "compensation_rgb_i8": compensations.tolist(),
        "phases": problem["initial_phases"],
    }


def _worldsheet_payload(
    archive: bytes,
    *,
    active_pair_ids: Sequence[int],
    grammar_move_ids: Sequence[str],
) -> tuple[bytes, dict[str, Any]]:
    members, _homes = parse_carrier_compose_archive(archive)
    payload = members[WORLDSHEET_G1_MEMBER]
    if not grammar_move_ids:
        return payload, {
            "active_track_count": 0,
            "shifted_track_count": 0,
            "grammar_move_ids": [],
            "track_dx_range": [0, 0],
            "track_dy_range": [0, 0],
            "source_payload_sha256": _sha256(payload),
            "compiled_payload_sha256": _sha256(payload),
        }
    lift = lift_g1_movable_worldsheet(payload)
    pair_set = {int(value) for value in active_pair_ids}
    active = [
        index
        for index, track in enumerate(lift.tracks)
        if any(lift.knots[knot].pair_index in pair_set for knot in track.knot_indices)
    ]
    if not active:
        raise DirectDescriptionError("v19b grammar move has no active track on this rung")
    translations = {index: [0, 0] for index in active}
    for candidate_id in grammar_move_ids:
        if candidate_id.startswith("worldsheet_joint_active_x_"):
            axis = 0
        elif candidate_id.startswith("worldsheet_joint_active_y_"):
            axis = 1
        else:
            raise DirectDescriptionError(
                f"v19b stack contains unsupported grammar move: {candidate_id}"
            )
        sign = int(candidate_id.rsplit("_", 1)[1])
        feasible = 0
        for track_index in active:
            bounds = _g1_track_translation_bounds(lift, track_index)[axis]
            if bounds[0] <= sign <= bounds[1]:
                translations[track_index][axis] += sign
                feasible += 1
        if feasible == 0:
            raise DirectDescriptionError(
                f"v19b grammar move has no feasible track on this rung: {candidate_id}"
            )
    knots = list(lift.knots)
    for track_index in active:
        dx, dy = translations[track_index]
        if dx == 0 and dy == 0:
            continue
        x_bounds, y_bounds = _g1_track_translation_bounds(lift, track_index)
        if not x_bounds[0] <= dx <= x_bounds[1] or not y_bounds[0] <= dy <= y_bounds[1]:
            raise DirectDescriptionError("v19b joint grammar translation left the scorer grid")
        for knot_index in lift.tracks[track_index].knot_indices:
            knot = knots[knot_index]
            knots[knot_index] = replace(
                knot, center_x=knot.center_x + dx, center_y=knot.center_y + dy
            )
    compiled = encode_lifted_g1_movable_worldsheet(replace(lift, knots=tuple(knots)))
    dx_values = [value[0] for value in translations.values()]
    dy_values = [value[1] for value in translations.values()]
    return compiled, {
        "active_track_count": len(active),
        "shifted_track_count": sum(value != [0, 0] for value in translations.values()),
        "grammar_move_ids": list(grammar_move_ids),
        "track_dx_range": [min(dx_values), max(dx_values)],
        "track_dy_range": [min(dy_values), max(dy_values)],
        "source_payload_sha256": _sha256(payload),
        "compiled_payload_sha256": _sha256(compiled),
    }


def _g1_track_translation_bounds(
    lift: Any, track_index: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return integer shifts that keep one direct G1 lifecycle in 512x384."""

    templates = {row.template_ref: row for row in lift.templates}
    minimum_x = minimum_y = 1 << 30
    maximum_x = maximum_y = -(1 << 30)
    for knot_index in lift.tracks[track_index].knot_indices:
        knot = lift.knots[knot_index]
        relative = templates[knot.template_ref].relative_vertices_xy
        xs = [knot.center_x + int(vertex[0]) for vertex in relative]
        ys = [knot.center_y + int(vertex[1]) for vertex in relative]
        minimum_x = min(minimum_x, min(xs))
        maximum_x = max(maximum_x, max(xs))
        minimum_y = min(minimum_y, min(ys))
        maximum_y = max(maximum_y, max(ys))
    return ((-minimum_x, 511 - maximum_x), (-minimum_y, 383 - maximum_y))


def _carrier_with_worldsheet(expanded_base: bytes, worldsheet_payload: bytes) -> bytes:
    members, _homes = parse_carrier_compose_archive(expanded_base)
    expected = {
        "manifest.json",
        "predictor.zip",
        WORLDSHEET_G1_MEMBER,
        "render/receiver_realization.ddrp",
        SCORER_SOLVED_TEMPLATE_MEMBER,
    }
    unexpected = set(members) - expected - {REALIZATION_STATIC_RULE_MEMBER}
    if unexpected:
        raise DirectDescriptionError(
            f"v19b refuses to drop unmodeled carrier members: {sorted(unexpected)}"
        )
    receiver = receive_carrier_compose_archive(expanded_base)
    archive, _rows = compile_carrier_compose_archive(
        members["predictor.zip"],
        worldsheet_g1_payload=worldsheet_payload,
        realization_profile=receiver.realization_profile,
        realization_static_rule_payload=members.get(REALIZATION_STATIC_RULE_MEMBER, b""),
        realization_static_rule_id=receiver.realization_static_rule_id,
        scorer_solved_templates=receiver.scorer_solved_templates,
    )
    if worldsheet_payload == members[WORLDSHEET_G1_MEMBER] and archive != expanded_base:
        raise DirectDescriptionError("v19b carrier identity recompile changed bytes")
    return archive


def _preuint8_program(
    *,
    problem: Mapping[str, Any],
    ctx: Mapping[str, Any],
    coupled_archive: bytes,
    scale_q8: int,
    source_start: int,
    source_stop: int,
) -> PreUint8Q8ProgramV1:
    candidate = _candidate_state(ctx, FIRST_MOVE_ID)
    initial_templates = np.asarray(
        problem["initial_template_values_u8"], dtype=np.int16
    )
    candidate_templates = np.asarray(
        candidate["template_values_u8"], dtype=np.int16
    )
    template_delta = candidate_templates - initial_templates
    receiver = receive_coupled_margin_archive(coupled_archive)
    templates = []
    for placement in receiver.program.placements:
        delta = template_delta[placement.template_index]
        if np.any(delta):
            templates.append(
                TemplateQ8CorrectionV1(
                    placement.source_pair_id,
                    placement.template_index,
                    tuple(int(value) * scale_q8 for value in delta.reshape(-1)),
                )
            )
    initial_sparse = np.asarray(
        problem["initial_compensation_rgb_i8"], dtype=np.int16
    )
    candidate_sparse = np.asarray(
        candidate["compensation_rgb_i8"], dtype=np.int16
    )
    sparse_delta = candidate_sparse - initial_sparse
    sparse = []
    for index, support in enumerate(problem["sparse_compensation_support"]):
        pair_id = int(support["source_pair_id"])
        if not source_start <= pair_id < source_stop or not np.any(sparse_delta[index]):
            continue
        sparse.append(
            SparseQ8CorrectionV1(
                pair_id,
                int(support["frame_index"]),
                int(support["camera_y"]),
                int(support["camera_x"]),
                tuple(int(value) * scale_q8 for value in sparse_delta[index]),
            )
        )
    return PreUint8Q8ProgramV1(
        tuple(sorted(templates)), tuple(sorted(sparse)), "bayer8", 210
    )


def _rung_geometry(
    rung: Literal["dev", "n64", "n600"],
    v19_config: DDMV19PurePricedObjectiveConfigV1,
    ctx: Mapping[str, Any],
) -> dict[str, Any]:
    if rung == "n64":
        return {
            "v14_archive": _base_v14_bytes(ctx["n64_config"]),
            "source_archive": ctx["n64_archive"],
            "source_start": 448,
            "source_stop": 512,
            "source_pair_ids": tuple(range(448, 512)),
            "local_pair_ids": tuple(range(64)),
            "grammar_pair_ids": tuple(
                pair_id - 448
                for pair_id in v19_config.pair_ids
                if 448 <= pair_id < 512
            ),
        }
    return {
        "v14_archive": _base_v14_bytes(ctx["n600_config"]),
        "source_archive": ctx["n600_archive"],
        "source_start": 0,
        "source_stop": 600,
        "source_pair_ids": (
            tuple(v19_config.pair_ids) if rung == "dev" else tuple(range(600))
        ),
        "local_pair_ids": (
            tuple(v19_config.pair_ids) if rung == "dev" else tuple(range(600))
        ),
        "grammar_pair_ids": tuple(v19_config.pair_ids),
    }


def _compile_stack(
    stack: StackState,
    *,
    rung: Literal["dev", "n64", "n600"],
    v19_config: DDMV19PurePricedObjectiveConfigV1,
    ctx: Mapping[str, Any],
) -> tuple[bytes, Callable[[bytes], Any], dict[str, Any]]:
    geometry = _rung_geometry(rung, v19_config, ctx)
    state = _realized_post_state(stack, ctx["problem"])
    _compact, expanded_base, program = _ladder_archive(
        state=state,
        problem=ctx["problem"],
        v14_archive=geometry["v14_archive"],
        source_start=geometry["source_start"],
        source_stop=geometry["source_stop"],
    )
    worldsheet, grammar = _worldsheet_payload(
        geometry["source_archive"],
        active_pair_ids=geometry["grammar_pair_ids"],
        grammar_move_ids=stack.grammar_move_ids,
    )
    carrier = _carrier_with_worldsheet(expanded_base, worldsheet)
    coupled = compile_coupled_margin_archive(carrier, program)
    if stack.preuint8_scale_q8:
        q8_program = _preuint8_program(
            problem=ctx["problem"],
            ctx=ctx,
            coupled_archive=coupled,
            scale_q8=stack.preuint8_scale_q8,
            source_start=geometry["source_start"],
            source_stop=geometry["source_stop"],
        )
        archive = compile_preuint8_q8_archive(coupled, q8_program)
        receiver_factory = receive_preuint8_q8_archive
        wrapper = "preuint8_q8(coupled_margin(carrier_compose))"
        q8_records = {
            "template_records": len(q8_program.templates),
            "sparse_records": len(q8_program.sparse),
            "scale_q8_sum": stack.preuint8_scale_q8,
        }
    else:
        archive = coupled
        receiver_factory = receive_coupled_margin_archive
        wrapper = "coupled_margin(carrier_compose)"
        q8_records = None
    return archive, receiver_factory, {
        "rung": rung,
        "move_ids": list(stack.move_ids),
        "wrapper": wrapper,
        "archive_bytes": len(archive),
        "archive_sha256": _sha256(archive),
        "grammar": grammar,
        "post_int8": {
            "template_delta_nonzero": int(np.count_nonzero(stack.template_delta)),
            "compensation_delta_nonzero": int(
                np.count_nonzero(stack.compensation_delta)
            ),
            "placement_records": len(program.placements),
            "sparse_compensation_records": len(program.compensations),
        },
        "preuint8": q8_records,
        "source_start": geometry["source_start"],
        "source_stop": geometry["source_stop"],
        "source_pair_ids": list(geometry["source_pair_ids"]),
        "local_pair_ids": list(geometry["local_pair_ids"]),
    }


def _bucket_transition(
    before_cells: np.ndarray, after_cells: np.ndarray, labels: np.ndarray
) -> dict[str, Any]:
    before_correct = before_cells == labels
    after_correct = after_cells == labels
    role_mask = np.isin(labels, tuple(ROLE_CLASS_IDS))
    result = {}
    for name, mask in (
        ("role_bucket_Lane_plus_Movable", role_mask),
        ("residual_bucket_Road_Undrivable_MyCar", ~role_mask),
    ):
        errors_before = int(np.count_nonzero(~before_correct & mask))
        errors_after = int(np.count_nonzero(~after_correct & mask))
        helpful = int(np.count_nonzero(~before_correct & after_correct & mask))
        harmful = int(np.count_nonzero(before_correct & ~after_correct & mask))
        result[name] = {
            "sites": int(np.count_nonzero(mask)),
            "errors_before": errors_before,
            "errors_after": errors_after,
            "helpful_flips": helpful,
            "harmful_flips": harmful,
            "realized_net_flips": helpful - harmful,
            "delta_errors": errors_after - errors_before,
            "delta_d_seg_global_denominator": (
                errors_after - errors_before
            ) / labels.size,
        }
    return result


def _nonadditivity(
    single_step_delta: Mapping[str, Any],
    joint_incremental_delta: Mapping[str, Any],
) -> dict[str, Any]:
    original_gain = max(0.0, -float(single_step_delta["joint_delta"]))
    realized_gain = max(0.0, -float(joint_incremental_delta["joint_delta"]))
    survived = min(original_gain, realized_gain)
    degraded = max(0.0, original_gain - realized_gain)
    amplified = max(0.0, realized_gain - original_gain)
    if realized_gain == 0.0:
        token = "DEGRADED_TO_REJECTION"
    elif amplified > 1e-15:
        token = "SURVIVED_AND_AMPLIFIED"
    elif degraded > 1e-15:
        token = "SURVIVED_PARTIALLY_DEGRADED"
    else:
        token = "SURVIVED"
    return {
        "single_step_gain": original_gain,
        "joint_incremental_gain": realized_gain,
        "survived_gain": survived,
        "degraded_gain": degraded,
        "amplified_gain": amplified,
        "survival_fraction": (
            None if original_gain == 0.0 else realized_gain / original_gain
        ),
        "verdict": token,
    }


def _load_stage(path: Path, config_sha256: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = json.loads(path.read_bytes())
    if value.get("typed_config_sha256") != config_sha256:
        raise DirectDescriptionError(f"v19b resumed stage identity differs: {path}")
    return value


def _measure_dev(
    *,
    archive: bytes,
    receiver_factory: Callable[[bytes], Any],
    v19_config: DDMV19PurePricedObjectiveConfigV1,
    ctx: Mapping[str, Any],
    baseline_camera: np.ndarray | None = None,
    baseline_cells: np.ndarray | None = None,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    measurement, _logits, cells, camera = _measure(
        archive=archive,
        receiver_factory=receiver_factory,
        pair_ids=v19_config.pair_ids,
        labels_all=ctx["labels_all"],
        poses_all=ctx["poses_all"],
        segnet=ctx["segnet"],
        posenet=ctx["posenet"],
        baseline_camera=baseline_camera,
        baseline_cells=baseline_cells,
    )
    return measurement, cells, camera


def _stage_greedy(
    config: DDMV19BJointRemeasureStackConfigV1,
    root: Path,
    v19_config: DDMV19PurePricedObjectiveConfigV1,
    source_receipt: Mapping[str, Any],
    ctx: Mapping[str, Any],
) -> dict[str, Any]:
    final_path = root / "stage_checkpoints" / "01_greedy_joint_screen.json"
    resumed = _load_stage(final_path, config.typed_config_hash())
    if resumed is not None:
        return resumed
    inventory = _accepted_inventory(source_receipt)
    row_root = root / "stage_checkpoints" / "01_greedy_candidates"
    rows: list[dict[str, Any]] = []
    stack = _initial_stack(ctx["problem"])
    for index, move in enumerate(inventory):
        checkpoint = row_root / f"candidate_{index:02d}_{move['candidate_id']}.json"
        row = _load_stage(checkpoint, config.typed_config_hash())
        if row is None:
            break
        if row.get("candidate_id") != move["candidate_id"]:
            raise DirectDescriptionError("v19b greedy checkpoint order differs")
        rows.append(row)
        if row["accepted"]:
            stack = _apply_move(stack, move, problem=ctx["problem"], ctx=ctx)
    if len(rows) not in (0, len(inventory)) and any(
        _load_stage(
            row_root / f"candidate_{later:02d}_{inventory[later]['candidate_id']}.json",
            config.typed_config_hash(),
        )
        is not None
        for later in range(len(rows) + 1, len(inventory))
    ):
        raise DirectDescriptionError("v19b greedy checkpoints contain a gap")

    labels = np.asarray(
        ctx["labels_all"][np.asarray(v19_config.pair_ids, dtype=np.int64)]
    )
    if not rows:
        baseline_measurement, _baseline_logits, baseline_cells, baseline_camera = _measure(
            archive=ctx["baseline_archive"],
            receiver_factory=receive_coupled_margin_archive,
            pair_ids=v19_config.pair_ids,
            labels_all=ctx["labels_all"],
            poses_all=ctx["poses_all"],
            segnet=ctx["segnet"],
            posenet=ctx["posenet"],
        )
        first_stack = _apply_move(
            stack, inventory[0], problem=ctx["problem"], ctx=ctx
        )
        archive, factory, compile_receipt = _compile_stack(
            first_stack, rung="dev", v19_config=v19_config, ctx=ctx
        )
        source_first = source_receipt["first_row_405"]
        if _sha256(archive) != source_first["archive"]["sha256"]:
            raise DirectDescriptionError("v19b common master did not reproduce the 405 archive")
        measurement, cells, camera = _measure_dev(
            archive=archive,
            receiver_factory=factory,
            v19_config=v19_config,
            ctx=ctx,
            baseline_camera=baseline_camera,
            baseline_cells=baseline_cells,
        )
        if measurement["cells_sha256"] != source_first["measurement"]["cells_sha256"]:
            raise DirectDescriptionError("v19b 405 scorer replay differs from v19")
        archive_path = (
            root
            / "candidate_archives"
            / f"greedy_00_{inventory[0]['candidate_id']}.zip.receipt-bytes"
        )
        _publish_immutable(archive_path, archive)
        incremental = _delta_payload(baseline_measurement, measurement)
        row = {
            "schema": "ddm_v19b_joint_move.v1",
            "typed_config_sha256": config.typed_config_hash(),
            "move_index": 0,
            "candidate_id": inventory[0]["candidate_id"],
            "family": inventory[0]["family"],
            "forced_start_from_v19_admission": True,
            "accepted": True,
            "archive": {
                "path": _portable(archive_path),
                "bytes": len(archive),
                "sha256": _sha256(archive),
            },
            "compile": compile_receipt,
            "before": baseline_measurement,
            "trial": measurement,
            "current_after": measurement,
            "joint_incremental_delta": incremental,
            "cumulative_delta_vs_v19_control": incremental,
            "c1_bucket_attribution": _bucket_transition(
                baseline_cells, cells, labels
            ),
            "nonadditivity": _nonadditivity(
                inventory[0]["single_step_delta"], incremental
            ),
            "single_step_v19_delta": inventory[0]["single_step_delta"],
            "evidence_axis": AXIS,
            "score_claim": False,
        }
        _write(row_root / f"candidate_00_{inventory[0]['candidate_id']}.json", row)
        rows.append(row)
        stack = first_stack
        current_measurement = measurement
        current_cells = cells
        current_camera = camera
    else:
        archive, factory, _compile_receipt = _compile_stack(
            stack, rung="dev", v19_config=v19_config, ctx=ctx
        )
        current_measurement, current_cells, current_camera = _measure_dev(
            archive=archive,
            receiver_factory=factory,
            v19_config=v19_config,
            ctx=ctx,
        )
        expected = rows[-1]["current_after"]
        if current_measurement["cells_sha256"] != expected["cells_sha256"]:
            raise DirectDescriptionError("v19b resumed current stack scorer state differs")

    control = rows[0]["before"]
    for index in range(len(rows), len(inventory)):
        move = inventory[index]
        trial_stack = _apply_move(
            stack, move, problem=ctx["problem"], ctx=ctx
        )
        archive, factory, compile_receipt = _compile_stack(
            trial_stack, rung="dev", v19_config=v19_config, ctx=ctx
        )
        archive_path = (
            root
            / "candidate_archives"
            / f"greedy_{index:02d}_{move['candidate_id']}.zip.receipt-bytes"
        )
        _publish_immutable(archive_path, archive)
        trial, trial_cells, trial_camera = _measure_dev(
            archive=archive,
            receiver_factory=factory,
            v19_config=v19_config,
            ctx=ctx,
            baseline_camera=current_camera,
            baseline_cells=current_cells,
        )
        incremental = _delta_payload(current_measurement, trial)
        accepted = bool(incremental["accepted"])
        row = {
            "schema": "ddm_v19b_joint_move.v1",
            "typed_config_sha256": config.typed_config_hash(),
            "move_index": index,
            "candidate_id": move["candidate_id"],
            "family": move["family"],
            "forced_start_from_v19_admission": False,
            "accepted": accepted,
            "archive": {
                "path": _portable(archive_path),
                "bytes": len(archive),
                "sha256": _sha256(archive),
            },
            "compile": compile_receipt,
            "before": current_measurement,
            "trial": trial,
            "current_after": trial if accepted else current_measurement,
            "joint_incremental_delta": incremental,
            "cumulative_delta_vs_v19_control": _delta_payload(control, trial),
            "c1_bucket_attribution": _bucket_transition(
                current_cells, trial_cells, labels
            ),
            "nonadditivity": _nonadditivity(
                move["single_step_delta"], incremental
            ),
            "single_step_v19_delta": move["single_step_delta"],
            "evidence_axis": AXIS,
            "score_claim": False,
        }
        checkpoint = (
            row_root / f"candidate_{index:02d}_{move['candidate_id']}.json"
        )
        _write(checkpoint, row)
        rows.append(row)
        if accepted:
            stack = trial_stack
            current_measurement = trial
            current_cells = trial_cells
            current_camera = trial_camera

    final_archive, _factory, final_compile = _compile_stack(
        stack, rung="dev", v19_config=v19_config, ctx=ctx
    )
    final_archive_path = root / "ddm_v19b_final_dev.zip.receipt-bytes"
    _publish_immutable(final_archive_path, final_archive)
    accepted_rows = [row for row in rows if row["accepted"]]
    original_gain = sum(
        float(row["nonadditivity"]["single_step_gain"]) for row in rows[1:]
    )
    survived = sum(float(row["nonadditivity"]["survived_gain"]) for row in rows[1:])
    degraded = sum(float(row["nonadditivity"]["degraded_gain"]) for row in rows[1:])
    amplified = sum(float(row["nonadditivity"]["amplified_gain"]) for row in rows[1:])
    result = {
        "schema": "ddm_v19b_greedy_joint_screen.v1",
        "typed_config_sha256": config.typed_config_hash(),
        "status": "MEASURED",
        "candidate_order": [row["candidate_id"] for row in rows],
        "forced_first_move": FIRST_MOVE_ID,
        "per_move_joint_table": rows,
        "accepted_move_ids": [row["candidate_id"] for row in accepted_rows],
        "accepted_move_count": len(accepted_rows),
        "rejected_move_ids": [
            row["candidate_id"] for row in rows if not row["accepted"]
        ],
        "final_stack": {
            "archive": {
                "path": _portable(final_archive_path),
                "bytes": len(final_archive),
                "sha256": _sha256(final_archive),
            },
            "measurement": current_measurement,
            "compile": final_compile,
            "cumulative_delta_vs_v19_control": _delta_payload(
                control, current_measurement
            ),
        },
        "nonadditivity_aggregate_remaining_winners": {
            "single_step_gain_total": original_gain,
            "survived_gain_total": survived,
            "degraded_gain_total": degraded,
            "amplified_gain_total": amplified,
            "survival_fraction": (
                None if original_gain == 0.0 else survived / original_gain
            ),
            "strict_joint_remeasurement_used": True,
            "same_frame_candidates_sequentially_joint_confirmed": True,
        },
        "chunked_verdict": (
            "MULTI_MOVE_DEV_STACK"
            if len(accepted_rows) >= 2
            else "ONLY_405_SURVIVED_DEV"
        ),
        "score_claim": False,
        "evidence_axis": AXIS,
    }
    _write(final_path, result)
    return result


def _stack_from_ids(
    move_ids: Sequence[str],
    inventory: Sequence[Mapping[str, Any]],
    *,
    problem: Mapping[str, Any],
    ctx: Mapping[str, Any],
) -> StackState:
    by_id = {str(row["candidate_id"]): row for row in inventory}
    stack = _initial_stack(problem)
    for candidate_id in move_ids:
        if candidate_id not in by_id:
            raise DirectDescriptionError(f"v19b final stack move is absent: {candidate_id}")
        stack = _apply_move(stack, by_id[candidate_id], problem=problem, ctx=ctx)
    return stack


def _archive_byte_rows(
    archive: bytes, receiver_factory: Callable[[bytes], Any]
) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
    if receiver_factory is receive_coupled_margin_archive:
        receiver = receive_coupled_margin_archive(archive)
        return coupled_margin_byte_rows(archive), receiver.custody
    members, homes = parse_preuint8_q8_archive(archive)
    receiver = receive_preuint8_q8_archive(archive)
    rows = [
        {
            "stratum": {
                "manifest.json": "outer_manifest",
                "base/ddm_v16_receiver.zip": "nested_coupled_margin_receiver",
                "render/preuint8_q8_program.ddq8": "preuint8_q8_program",
            }[row["name"]],
            **row,
        }
        for row in homes
    ]
    rows.extend(
        {
            "nested_under": "base/ddm_v16_receiver.zip",
            **row,
        }
        for row in coupled_margin_byte_rows(members["base/ddm_v16_receiver.zip"])
    )
    return rows, receiver.custody


def _measure_window(
    *,
    name: str,
    archive: bytes,
    receiver_factory: Callable[[bytes], Any],
    baseline_archive: bytes,
    source_pair_ids: Sequence[int],
    local_pair_ids: Sequence[int],
    root: Path,
    config_hash: str,
    batch_size: int,
    labels_all: np.ndarray,
    poses_all: np.ndarray,
    segnet: Any,
    posenet: Any,
) -> dict[str, Any]:
    stage = root / "stage_checkpoints" / name
    source = np.asarray(source_pair_ids, dtype=np.int64)
    local = np.asarray(local_pair_ids, dtype=np.int64)
    if source.shape != local.shape or source.size == 0:
        raise DirectDescriptionError("v19b ladder pair geometry differs")
    for start in range(0, int(source.size), batch_size):
        stop = min(start + batch_size, int(source.size))
        checkpoint = stage / f"batch_{start:04d}_{stop:04d}.json"
        resumed = _load_stage(checkpoint, config_hash)
        if resumed is not None:
            if resumed["archive_sha256"] != _sha256(archive):
                raise DirectDescriptionError("v19b ladder batch archive identity differs")
            continue
        receiver = receiver_factory(archive)
        baseline = receive_carrier_compose_archive(baseline_archive)
        local_ids = tuple(int(value) for value in local[start:stop])
        source_ids = source[start:stop]
        camera = receiver.render_camera_pairs(local_ids)
        baseline_camera = baseline.render_camera_pairs(local_ids)
        cells, pose6 = _forward(segnet, posenet, camera)
        replay_cells, replay_pose6 = _forward(segnet, posenet, camera)
        if not np.array_equal(cells, replay_cells) or not np.array_equal(
            pose6, replay_pose6
        ):
            raise DirectDescriptionError(f"v19b {name} deterministic batch replay failed")
        labels = np.asarray(labels_all[source_ids])
        poses = np.asarray(poses_all[source_ids])
        errors = cells != labels
        class_rows = {}
        for class_id, class_name in CLASS_NAMES.items():
            mask = labels == class_id
            class_rows[class_name] = {
                "errors": int(np.count_nonzero(errors & mask)),
                "sites": int(np.count_nonzero(mask)),
            }
        diff = camera.astype(np.int16) - baseline_camera.astype(np.int16)
        row = {
            "schema": "ddm_v19b_receiver_batch.v1",
            "typed_config_sha256": config_hash,
            "candidate": name,
            "archive_sha256": _sha256(archive),
            "source_pair_ids": source_ids.tolist(),
            "errors": int(np.count_nonzero(errors)),
            "sites": int(errors.size),
            "pose_squared_error_sum": (
                f"{float(np.square(pose6 - poses).sum(dtype=np.float64)):.12f}"
            ),
            "pose_coordinates": int(pose6.size),
            "class_rows": class_rows,
            "cells_sha256": _sha256_array(cells),
            "pose6_sha256": _sha256_array(pose6),
            "camera_diff_vs_v15": {
                "changed_channel_values": int(np.count_nonzero(diff)),
                "changed_rgb_pixels": int(
                    np.count_nonzero(np.any(diff != 0, axis=-1))
                ),
                "l1_channel_sum": int(np.abs(diff).sum(dtype=np.int64)),
                "candidate_camera_sha256": _sha256_array(camera),
                "v15_camera_sha256": _sha256_array(baseline_camera),
            },
            "score_claim": False,
            "evidence_axis": AXIS,
        }
        _write(checkpoint, row)
    batches = [
        json.loads(path.read_bytes()) for path in sorted(stage.glob("batch_*.json"))
    ]
    expected = math.ceil(source.size / batch_size)
    if len(batches) != expected:
        raise DirectDescriptionError("v19b ladder batch coverage incomplete")
    if any(
        row["archive_sha256"] != _sha256(archive)
        or row["typed_config_sha256"] != config_hash
        for row in batches
    ):
        raise DirectDescriptionError("v19b ladder aggregate batch identity differs")
    errors = sum(int(row["errors"]) for row in batches)
    sites = sum(int(row["sites"]) for row in batches)
    pose_sse = sum(float(row["pose_squared_error_sum"]) for row in batches)
    pose_coordinates = sum(int(row["pose_coordinates"]) for row in batches)
    classes = {
        class_name: {
            "errors": sum(
                int(row["class_rows"][class_name]["errors"]) for row in batches
            ),
            "sites": sum(
                int(row["class_rows"][class_name]["sites"]) for row in batches
            ),
        }
        for class_name in CLASS_NAMES.values()
    }
    for row in classes.values():
        row["d_seg"] = f"{row['errors'] / max(1, row['sites']):.12f}"
    d_seg = errors / sites
    d_pose = pose_sse / pose_coordinates
    byte_rows, custody = _archive_byte_rows(archive, receiver_factory)
    return {
        "candidate": name,
        "archive_bytes": len(archive),
        "archive_sha256": _sha256(archive),
        "d_seg": f"{d_seg:.12f}",
        "d_pose": f"{d_pose:.12f}",
        "errors": errors,
        "sites": sites,
        "per_class": classes,
        "c1_buckets": {
            "role_bucket_Lane_plus_Movable": {
                "errors": classes["Lane"]["errors"] + classes["Movable"]["errors"],
                "sites": classes["Lane"]["sites"] + classes["Movable"]["sites"],
            },
            "residual_bucket_Road_Undrivable_MyCar": {
                "errors": (
                    classes["Road"]["errors"]
                    + classes["Undrivable"]["errors"]
                    + classes["MyCar"]["errors"]
                ),
                "sites": (
                    classes["Road"]["sites"]
                    + classes["Undrivable"]["sites"]
                    + classes["MyCar"]["sites"]
                ),
            },
        },
        "advisory_score_formula_value": (
            f"{100.0 * d_seg + math.sqrt(10.0 * d_pose) + 25.0 * len(archive) / SOURCE_BYTES:.12f}"
        ),
        "batch_count": len(batches),
        "batch_size": batch_size,
        "all_batches_checkpointed_and_preserved": True,
        "batch_digest_chain_sha256": hashlib.sha256(
            "".join(
                row["cells_sha256"] + row["pose6_sha256"] for row in batches
            ).encode()
        ).hexdigest(),
        "camera_diff_vs_v15": {
            key: sum(
                int(row["camera_diff_vs_v15"][key]) for row in batches
            )
            for key in ("changed_channel_values", "changed_rgb_pixels", "l1_channel_sum")
        },
        "byte_streams": byte_rows,
        "receiver_custody": dict(custody),
        "score_claim": False,
        "evidence_axis": AXIS,
    }


def _control_with_errors(receipt: Mapping[str, Any]) -> dict[str, Any]:
    control = _source_control(dict(receipt))
    matches = [
        row
        for row in receipt["solved_template_ladder"]
        if row["candidate"] == "v15_solved_templates"
    ]
    if len(matches) != 1:
        raise DirectDescriptionError("v19b v15 control row is not unique")
    row = matches[0]
    control.update(
        {
            "errors": int(row["errors"]),
            "sites": int(row["sites"]),
            "per_role": {
                name: row["per_stratum"][name] for name in ("Lane", "Movable")
            },
        }
    )
    return control


def _rung_bucket_delta(
    control: Mapping[str, Any], measurement: Mapping[str, Any]
) -> dict[str, Any]:
    role_before = sum(
        int(control["per_role"][name]["errors"]) for name in ("Lane", "Movable")
    )
    role_after = int(
        measurement["c1_buckets"]["role_bucket_Lane_plus_Movable"]["errors"]
    )
    residual_before = int(control["errors"]) - role_before
    residual_after = int(
        measurement["c1_buckets"]["residual_bucket_Road_Undrivable_MyCar"][
            "errors"
        ]
    )
    return {
        "role_bucket_Lane_plus_Movable": {
            "errors_before": role_before,
            "errors_after": role_after,
            "realized_net_flips": role_before - role_after,
        },
        "residual_bucket_Road_Undrivable_MyCar": {
            "errors_before": residual_before,
            "errors_after": residual_after,
            "realized_net_flips": residual_before - residual_after,
        },
    }


def _stage_rung(
    *,
    rung: Literal["n64", "n600"],
    config: DDMV19BJointRemeasureStackConfigV1,
    root: Path,
    v19_config: DDMV19PurePricedObjectiveConfigV1,
    source_receipt: Mapping[str, Any],
    ctx: Mapping[str, Any],
    greedy: Mapping[str, Any],
    n64: Mapping[str, Any] | None,
) -> dict[str, Any]:
    ordinal = "02" if rung == "n64" else "03"
    path = root / "stage_checkpoints" / f"{ordinal}_{rung}_joint_stack.json"
    resumed = _load_stage(path, config.typed_config_hash())
    if resumed is not None:
        return resumed
    if rung == "n600" and not bool((n64 or {}).get("admitted_to_n600")):
        result = {
            "schema": "ddm_v19b_joint_stack_ladder.v1",
            "typed_config_sha256": config.typed_config_hash(),
            "rung": rung,
            "status": "NOT_RUN_N64_DID_NOT_PRESERVE_ADMISSION",
            "score_claim": False,
            "evidence_axis": AXIS,
        }
        _write(path, result)
        return result
    inventory = _accepted_inventory(source_receipt)
    stack = _stack_from_ids(
        greedy["accepted_move_ids"],
        inventory,
        problem=ctx["problem"],
        ctx=ctx,
    )
    archive, receiver_factory, compile_receipt = _compile_stack(
        stack, rung=rung, v19_config=v19_config, ctx=ctx
    )
    archive_path = root / f"ddm_v19b_final_{rung}.zip.receipt-bytes"
    _publish_immutable(archive_path, archive)
    geometry = _rung_geometry(rung, v19_config, ctx)
    measurement = _measure_window(
        name=f"{ordinal}_{rung}_batches",
        archive=archive,
        receiver_factory=receiver_factory,
        baseline_archive=geometry["source_archive"],
        source_pair_ids=geometry["source_pair_ids"],
        local_pair_ids=geometry["local_pair_ids"],
        root=root,
        config_hash=config.typed_config_hash(),
        batch_size=config.scorer_batch_size,
        labels_all=ctx["labels_all"],
        poses_all=ctx["poses_all"],
        segnet=ctx["segnet"],
        posenet=ctx["posenet"],
    )
    source_v15_receipt = ctx["n64_receipt"] if rung == "n64" else ctx["n600_receipt"]
    control = _control_with_errors(source_v15_receipt)
    expected_control_sha = _sha256(geometry["source_archive"])
    if control["archive_sha256"] != expected_control_sha:
        raise DirectDescriptionError(f"v19b {rung} v15 control identity differs")
    delta = _delta_payload(control, measurement)
    result = {
        "schema": "ddm_v19b_joint_stack_ladder.v1",
        "typed_config_sha256": config.typed_config_hash(),
        "rung": rung,
        "status": "MEASURED",
        "move_ids": list(stack.move_ids),
        "archive": {
            "path": _portable(archive_path),
            "bytes": len(archive),
            "sha256": _sha256(archive),
        },
        "compile": compile_receipt,
        "control": control,
        "measurement": measurement,
        "joint_delta_vs_v15_control": delta,
        "c1_bucket_delta_vs_v15_control": _rung_bucket_delta(
            control, measurement
        ),
        "admitted_to_n600": bool(delta["accepted"]) if rung == "n64" else None,
        "chunked_verdict": (
            "N64_STACK_PRESERVED_STRICT_JOINT_ADMISSION"
            if rung == "n64" and delta["accepted"]
            else "N64_STACK_FAILED_STRICT_JOINT_ADMISSION"
            if rung == "n64"
            else "N600_STACK_STRICT_JOINT_ADMISSION_CONFIRMED"
            if delta["accepted"]
            else "N600_STACK_NOT_ADMITTED"
        ),
        "score_claim": False,
        "evidence_axis": AXIS,
    }
    _write(path, result)
    return result


def _template_order_gauge(
    archive: bytes, receiver_factory: Callable[[bytes], Any]
) -> dict[str, Any]:
    receiver = receiver_factory(archive)
    coupled = receiver if receiver_factory is receive_coupled_margin_archive else receiver.base
    bank = coupled.base.scorer_solved_templates
    if bank is None:
        raise DirectDescriptionError("v19b final archive lost its template payload")
    payload = encode_scorer_solved_template_bank(bank)
    return {
        "scope": "shared_template_atom_order_only",
        "bytes_as_emitted": len(payload),
        "bytes_canonical_order": len(payload),
        "byte_delta": 0,
        "as_emitted_sha256": _sha256(payload),
        "status": "ZERO_RATE_ACTUATOR_IN_CURRENT_FIXED_WIDTH_ZIP_STORED_PAYLOAD",
        "reason": (
            "all six 2x2 RGB records have fixed width and the receiver stores the "
            "payload without an order-sensitive entropy code; atom-order matching "
            "is routed to c1 CODE, where any future delta coder must remap placement "
            "indices and prove camera-byte identity"
        ),
        "frozen_scorer_permutation_claim_imported": False,
    }


def _c1_handoff(
    greedy: Mapping[str, Any], n600: Mapping[str, Any]
) -> dict[str, Any]:
    count = int(greedy["accepted_move_count"])
    if n600.get("status") != "MEASURED":
        return {
            "status": "BLOCKED_NO_N600_FINAL_STACK",
            "division_of_labor": (
                "v19b retains local correction admission; #366 owns the unmeasured residual finish"
            ),
        }
    control = n600["control"]
    measurement = n600["measurement"]
    archive_delta = int(measurement["archive_bytes"]) - int(control["archive_bytes"])
    net_flips = int(control["errors"]) - int(measurement["errors"])
    remaining = max(0, int(measurement["errors"]) - TARGET_MAX_ERRORS)
    rate = None if archive_delta <= 0 else net_flips / archive_delta
    bytes_per_flip = None if net_flips <= 0 else archive_delta / net_flips
    if count >= 2:
        division = (
            "v19b owns the admitted multi-move REALIZE correction line; v18b and "
            "#366/J3 consume the exact residual state without additive credit"
        )
    else:
        division = (
            "v19b owns only the 405 compact-int8 REALIZE move; all remaining "
            "residual Seg/Pose closure stays with #366/J3"
        )
    return {
        "status": "EXACT_N600_HANDOFF",
        "division_of_labor": division,
        "accepted_correction_moves": count,
        "correction_line": {
            "input_archive_bytes": int(control["archive_bytes"]),
            "output_archive_bytes": int(measurement["archive_bytes"]),
            "delta_archive_bytes": archive_delta,
            "input_d_seg": control["d_seg"],
            "output_d_seg": measurement["d_seg"],
            "delta_d_seg": float(measurement["d_seg"]) - float(control["d_seg"]),
            "realized_net_flips": net_flips,
            "realized_net_flips_per_added_byte": rate,
            "bytes_per_realized_net_flip": bytes_per_flip,
            "remaining_errors_above_c1_integer_target": remaining,
        },
        "downstream_budgets": [
            {
                "budget_id": "v18b_first_exact_pricing_rung",
                "budget_bytes": C1_DOWNSTREAM_BUDGET_BYTES,
                "exact_input_archive_bytes": int(measurement["archive_bytes"]),
                "exact_input_d_seg": measurement["d_seg"],
                "exact_input_errors": int(measurement["errors"]),
                "solo_full_close_net_flips_per_budget_byte": (
                    remaining / C1_DOWNSTREAM_BUDGET_BYTES
                ),
            },
            {
                "budget_id": "j3_xi_template_worldsheet_finish",
                "budget_bytes": C1_DOWNSTREAM_BUDGET_BYTES,
                "exact_input_archive_bytes": int(measurement["archive_bytes"]),
                "exact_input_d_seg": measurement["d_seg"],
                "exact_input_errors": int(measurement["errors"]),
                "conditional_credit_law": (
                    "required J3 flips = remaining errors after exact v18b replay; "
                    "independent deltas must not be summed"
                ),
            },
        ],
    }


def _final(
    *,
    config: DDMV19BJointRemeasureStackConfigV1,
    root: Path,
    storage: Mapping[str, Any],
    source_receipt: Mapping[str, Any],
    greedy: Mapping[str, Any],
    n64: Mapping[str, Any],
    n600: Mapping[str, Any],
    v19_config: DDMV19PurePricedObjectiveConfigV1,
    ctx: Mapping[str, Any],
) -> Path:
    path = root / "ddm_v19b_joint_remeasure_stack_receipt.json"
    if path.exists():
        value = json.loads(path.read_bytes())
        if value.get("typed_config_sha256") != config.typed_config_hash():
            raise DirectDescriptionError("v19b final receipt identity differs")
        return path
    inventory = _accepted_inventory(source_receipt)
    stack = _stack_from_ids(
        greedy["accepted_move_ids"],
        inventory,
        problem=ctx["problem"],
        ctx=ctx,
    )
    final_rung: Literal["n600", "n64", "dev"]
    if n600.get("status") == "MEASURED":
        final_rung = "n600"
    elif n64.get("status") == "MEASURED":
        final_rung = "n64"
    else:
        final_rung = "dev"
    final_archive, receiver_factory, _compile_receipt = _compile_stack(
        stack, rung=final_rung, v19_config=v19_config, ctx=ctx
    )
    n600_delta = n600.get("joint_delta_vs_v15_control", {})
    if n600_delta.get("accepted") is True:
        verdict = (
            "MULTI_MOVE_JOINT_STACK_ADMITTED_N600_ADVISORY"
            if greedy["accepted_move_count"] >= 2
            else "ONLY_405_SURVIVED_JOINT_REMEASUREMENT_N600_ADVISORY"
        )
    elif n64.get("joint_delta_vs_v15_control", {}).get("accepted") is True:
        verdict = "JOINT_STACK_ADMITTED_N64_N600_NOT_ADMITTED"
    else:
        verdict = "JOINT_STACK_DID_NOT_PRESERVE_N64_ADMISSION"
    result = {
        "schema": SCHEMA,
        "run_id": config.run_id,
        "lane_id": LANE_ID,
        "typed_config": config.model_dump(mode="json", by_alias=True),
        "typed_config_sha256": config.typed_config_hash(),
        "source_v19": {
            "config_path": config.v19_config_path,
            "config_sha256": config.v19_config_sha256,
            "receipt_path": config.v19_receipt_path,
            "receipt_sha256": config.v19_receipt_sha256,
            "first_move_id": FIRST_MOVE_ID,
        },
        "verdict": verdict,
        "verdict_scope": (
            "INSTANCE:V19B x source-v19 ten admitted alternative moves x deterministic "
            "greedy common-master composition x n64/n600 exact receiver replay; "
            "macOS-CPU advisory only; no contest-axis, family, score, or promotion verdict"
        ),
        "greedy_screen": greedy,
        "n64": n64,
        "n600": n600,
        "final_stack_move_ids": list(greedy["accepted_move_ids"]),
        "accepted_move_count": int(greedy["accepted_move_count"]),
        "c1_bucket_attribution_per_admitted_move": [
            {
                "move_index": row["move_index"],
                "candidate_id": row["candidate_id"],
                "buckets": row["c1_bucket_attribution"],
            }
            for row in greedy["per_move_joint_table"]
            if row["accepted"]
        ],
        "nonadditivity": greedy["nonadditivity_aggregate_remaining_winners"],
        "c1_handoff": _c1_handoff(greedy, n600),
        "template_order_gauge": _template_order_gauge(
            final_archive, receiver_factory
        ),
        "rate_price_per_byte": break_even_distortion_gain_per_byte(),
        "common_master_order": [
            "carrier_compose_with_joint_worldsheet_translation",
            "coupled_margin_compact_int8_templates_and_sparse_records",
            "preuint8_q8_sum_before_one_final_uint8_if_admitted",
            "exact_R",
            "frozen_SegNet_and_official_YUV6_PoseNet_encode_side_only",
        ],
        "kernel_contract": {
            "receiver_numpy_integer_path": "BIT_IDENTICAL_PARSE_BACK_CHECKED",
            "numpy_fp32_reference_relation": (
                "receiver emits exact uint8 camera bytes; scorer calls use the frozen "
                "CPU torch path bound by the v19 custody receipt"
            ),
            "mlx_or_mps_authority_used": False,
        },
        "storage_preflight": dict(storage),
        "resume": {
            "immutable_stage_checkpoints": sorted(
                _portable(row) for row in (root / "stage_checkpoints").rglob("*.json")
            ),
            "per_candidate_greedy_checkpoints": True,
            "per_scorer_batch_checkpoints": True,
            "candidate_and_final_archives_preserved": True,
        },
        "triality": {
            "dsl": "DDMV19BJointRemeasureStackConfigV1",
            "dag": ".omx/research/ddm_v19b_joint_remeasure_stack_DAG_FEED_20260723.md",
            "equation": ".omx/research/ddm_v19b_joint_remeasure_stack_canonical_equations_20260723.md",
        },
        "stores_consulted": [
            "CLAUDE.md",
            "AGENTS.md",
            "docs/operating_manual_craft_handoff.md",
            config.v19_receipt_path,
            ".omx/research/codex_findings_ddm_v19_pure_priced_objective_20260723_codex.md",
            ".omx/research/ddm_c1_composed_candidate_spec_603_613_20260723.md",
            ".omx/research/ddm_c1_composed_candidate_ledger_603_613_20260723.json",
            "operator inbox through 2026-07-23T05:27:21Z",
        ],
        "evidence_axis": AXIS,
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


def run(
    config: DDMV19BJointRemeasureStackConfigV1, root: Path, stage: str
) -> Path:
    root = root.resolve()
    storage = _deterministic_storage_receipt(_storage_preflight(root))
    root.mkdir(parents=True, exist_ok=True)
    v19_config, source_receipt, ctx = _load_sources(config)
    greedy = _stage_greedy(config, root, v19_config, source_receipt, ctx)
    if stage == "greedy":
        return root / "stage_checkpoints" / "01_greedy_joint_screen.json"
    n64 = _stage_rung(
        rung="n64",
        config=config,
        root=root,
        v19_config=v19_config,
        source_receipt=source_receipt,
        ctx=ctx,
        greedy=greedy,
        n64=None,
    )
    if stage == "n64":
        return root / "stage_checkpoints" / "02_n64_joint_stack.json"
    n600 = _stage_rung(
        rung="n600",
        config=config,
        root=root,
        v19_config=v19_config,
        source_receipt=source_receipt,
        ctx=ctx,
        greedy=greedy,
        n64=n64,
    )
    if stage == "n600":
        return root / "stage_checkpoints" / "03_n600_joint_stack.json"
    return _final(
        config=config,
        root=root,
        storage=storage,
        source_receipt=source_receipt,
        greedy=greedy,
        n64=n64,
        n600=n600,
        v19_config=v19_config,
        ctx=ctx,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("greedy", "n64", "n600", "all"), default="all"
    )
    args = parser.parse_args()
    config = DDMV19BJointRemeasureStackConfigV1.model_validate_json(
        _read_regular_file_once(args.config)
    )
    artifact = run(config, args.output_directory, args.stage)
    print(
        json.dumps(
            {
                "complete_stage": args.stage,
                "artifact": _portable(artifact),
                "evidence_axis": AXIS,
                "score_claim": False,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
