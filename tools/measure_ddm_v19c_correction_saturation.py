#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Saturate the DDM v19b correction line under exact realized joint pricing.

This local-only runner starts from the SHA-bound v19b common-master state,
interleaves substantially wider correction families, and recurses until either
K consecutive exact proposals fail or the additional 200 kB correction budget
is exhausted.  DEV is the broad proposal screen.  Every DEV admission is then
replayed sequentially on n600 and retained only when its n600 incremental
joint objective is strictly negative.

All measurements are macOS-CPU frozen-scorer advisory research.  They are not
contest scores and cannot move the frontier pointer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import zip_longest
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "src", REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.optimization.direct_description_carrier_compose import (  # noqa: E402
    WORLDSHEET_G1_MEMBER,
    parse_carrier_compose_archive,
    rfc8785_canonicalize,
)
from tac.optimization.direct_description_coupled_margin import (  # noqa: E402
    compile_coupled_margin_archive,
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
    receive_preuint8_q8_archive,
)
from tac.optimization.pure_priced_realized_objective import (  # noqa: E402
    break_even_distortion_gain_per_byte,
)
from tools.measure_ddm_v14_realization_fidelity import (  # noqa: E402
    POINTER_SCORE_TEXT,
    _publish_immutable,
    _storage_preflight,
)
from tools.measure_ddm_v16_coupled_joint_solve import (  # noqa: E402
    _base_v14_bytes,
    _ladder_archive,
)
from tools.measure_ddm_v19_pure_priced_objective import (  # noqa: E402
    AXIS,
    DDMV19PurePricedObjectiveConfigV1,
    _delta_payload,
    _deterministic_storage_receipt,
    _measure,
    _portable,
    _sha256,
    _write,
)
from tools.measure_ddm_v19b_joint_remeasure_stack import (  # noqa: E402
    FIRST_MOVE_ID,
    DDMV19BJointRemeasureStackConfigV1,
    _accepted_inventory,
    _bucket_transition,
    _candidate_state,
    _carrier_with_worldsheet,
    _g1_track_translation_bounds,
    _load_sources,
    _load_stage,
    _measure_dev,
    _measure_window,
    _rung_geometry,
    _stack_from_ids,
    _template_order_gauge,
)

SCHEMA = "ddm_v19c_correction_saturation_receipt.v1"
LANE_ID = "ddm_v19c_correction_saturation"
C1_ROLE_CEILING_ERRORS = 726_416
C1_CONTINUOUS_DEBT = 3_103_688.832
C1_TARGET_ERRORS = 136_839
C1_RESIDUAL_AFTER_PERFECT_ROLE = 2_377_273
FAMILIES = (
    "inverse_solved_rowband",
    "preuint8_q8_region",
    "worldsheet_track_event",
    "worldsheet_pair_event",
    "scorer_template_swap",
    "grammar_event_template_swap",
)


class DDMV19CCorrectionSaturationConfigV1(BaseModel):
    """SHA-bound local-only contract for recursive correction saturation."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    schema_: Literal["DDMV19CCorrectionSaturationConfigV1"] = Field(
        default="DDMV19CCorrectionSaturationConfigV1",
        alias="schema",
        serialization_alias="schema",
    )
    run_id: str = Field(min_length=8)
    seed: Literal[1234] = 1234
    v19b_config_path: str = Field(min_length=1)
    v19b_config_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    v19b_receipt_path: str = Field(min_length=1)
    v19b_receipt_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    scorer_batch_size: Literal[16] = 16
    consecutive_failure_limit: Literal[64] = 64
    correction_budget_bytes: Literal[200000] = 200_000
    memory_ceiling_gib: Literal[116] = 116
    rowband_direction_scales: tuple[Literal[-1, 1], ...] = (-1, 1)
    q8_all_scales: tuple[int, ...] = (
        -256,
        -192,
        -128,
        -96,
        -64,
        -32,
        32,
        64,
        96,
        128,
        192,
        256,
    )
    q8_region_scales: tuple[int, ...] = (-128, -64, -32, 32, 64, 128)
    q8_pair_scales: tuple[int, ...] = (-128, -64, 64, 128)
    proposal_order: Literal["family_interleaved_fisher_margin_reverse_waterfill_proxy"] = (
        "family_interleaved_fisher_margin_reverse_waterfill_proxy"
    )
    dev_acceptance: Literal["strict_realized_joint_delta_lt_zero"] = "strict_realized_joint_delta_lt_zero"
    n600_acceptance: Literal["sequential_strict_realized_joint_delta_lt_zero"] = (
        "sequential_strict_realized_joint_delta_lt_zero"
    )
    per_frame_batching: Literal["two_frame_pair_lifecycle_se_coupled"] = "two_frame_pair_lifecycle_se_coupled"
    research_only: Literal[True] = True
    execution_allowed: Literal[False] = False
    score_claim: Literal[False] = False

    @model_validator(mode="after")
    def _sealed_contract(self) -> DDMV19CCorrectionSaturationConfigV1:
        if self.consecutive_failure_limit < len(FAMILIES):
            raise ValueError("failure limit must expose every interleaved family")
        if 0 in self.q8_all_scales + self.q8_region_scales + self.q8_pair_scales:
            raise ValueError("zero Q8 proposals are inert and forbidden")
        if any(value % 32 for value in self.q8_all_scales):
            raise ValueError("Q8 all-region quanta must remain on the preregistered /32 lattice")
        return self

    def typed_config_hash(self) -> str:
        return hashlib.sha256(rfc8785_canonicalize(self.model_dump(mode="json", by_alias=True))).hexdigest()


@dataclass(frozen=True, slots=True)
class SaturationState:
    """Composable state before receiver-domain clipping and final uint8."""

    template_delta: np.ndarray
    compensation_delta: np.ndarray
    base_grammar_move_ids: tuple[str, ...]
    base_preuint8_scale_q8: int
    track_translations: tuple[tuple[int, int, int], ...] = ()
    q8_directives: tuple[tuple[str, int], ...] = ()
    grammar_ref_swaps: tuple[tuple[int, str], ...] = ()
    move_ids: tuple[str, ...] = ()


def _bound_v19b(
    config: DDMV19CCorrectionSaturationConfigV1,
) -> tuple[
    DDMV19BJointRemeasureStackConfigV1,
    dict[str, Any],
    DDMV19PurePricedObjectiveConfigV1,
    dict[str, Any],
    dict[str, Any],
]:
    payload = _read_regular_file_once(REPO_ROOT / config.v19b_config_path)
    if _sha256(payload) != config.v19b_config_sha256:
        raise DirectDescriptionError("v19c source v19b config SHA differs")
    v19b_config = DDMV19BJointRemeasureStackConfigV1.model_validate_json(payload)
    receipt_payload = _read_regular_file_once(REPO_ROOT / config.v19b_receipt_path)
    if _sha256(receipt_payload) != config.v19b_receipt_sha256:
        raise DirectDescriptionError("v19c source v19b receipt SHA differs")
    receipt = json.loads(receipt_payload)
    if receipt.get("schema") != "ddm_v19b_joint_remeasure_stack_receipt.v1":
        raise DirectDescriptionError("v19c source v19b receipt schema differs")
    if receipt.get("typed_config_sha256") != v19b_config.typed_config_hash():
        raise DirectDescriptionError("v19c source v19b receipt/config binding differs")
    if (
        receipt.get("score_claim") is not False
        or receipt.get("pointer_moved") is not False
        or receipt.get("accepted_move_count") != 10
    ):
        raise DirectDescriptionError("v19c source v19b authority or move count drifted")
    v19_config, v19_receipt, ctx = _load_sources(v19b_config)
    return v19b_config, receipt, v19_config, v19_receipt, ctx


def _initial_state(
    v19b_receipt: Mapping[str, Any],
    v19_receipt: Mapping[str, Any],
    ctx: Mapping[str, Any],
) -> SaturationState:
    inventory = _accepted_inventory(v19_receipt)
    source = _stack_from_ids(
        v19b_receipt["final_stack_move_ids"],
        inventory,
        problem=ctx["problem"],
        ctx=ctx,
    )
    return SaturationState(
        template_delta=source.template_delta.copy(),
        compensation_delta=source.compensation_delta.copy(),
        base_grammar_move_ids=tuple(source.grammar_move_ids),
        base_preuint8_scale_q8=int(source.preuint8_scale_q8),
        move_ids=tuple(source.move_ids),
    )


def _realized_post_state(state: SaturationState, problem: Mapping[str, Any]) -> dict[str, Any]:
    templates = np.clip(
        np.asarray(problem["initial_template_values_u8"], dtype=np.int16) + state.template_delta,
        0,
        255,
    ).astype(np.uint8)
    compensations = np.clip(
        np.asarray(problem["initial_compensation_rgb_i8"], dtype=np.int16) + state.compensation_delta,
        -127,
        127,
    ).astype(np.int16)
    return {
        "template_values_u8": templates.tolist(),
        "compensation_rgb_i8": compensations.tolist(),
        "phases": problem["initial_phases"],
    }


def _active_worldsheet(archive: bytes, pair_ids: Sequence[int]) -> tuple[Any, list[int]]:
    members, _homes = parse_carrier_compose_archive(archive)
    lift = lift_g1_movable_worldsheet(members[WORLDSHEET_G1_MEMBER])
    pair_set = {int(value) for value in pair_ids}
    active = [
        index
        for index, track in enumerate(lift.tracks)
        if any(lift.knots[knot].pair_index in pair_set for knot in track.knot_indices)
    ]
    if not active:
        raise DirectDescriptionError("v19c proposal inventory has no active worldsheet tracks")
    return lift, active


def _interleave(groups: Sequence[Sequence[dict[str, Any]]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for rows in zip_longest(*groups):
        result.extend(row for row in rows if row is not None)
    return result


def _proposal_inventory(
    config: DDMV19CCorrectionSaturationConfigV1,
    v19_config: DDMV19PurePricedObjectiveConfigV1,
    ctx: Mapping[str, Any],
    pair_ev: Mapping[int, float] | None = None,
) -> list[dict[str, Any]]:
    """Return family-complete coordinates ranked by measured Fisher-margin EV.

    EV is normalized only within each proposal family because the coordinate
    coders differ in dimensionality and shared-byte amortization. The strict
    realized joint objective remains the cross-family authority.
    """

    lift, active_tracks = _active_worldsheet(ctx["n600_archive"], v19_config.pair_ids)
    ev_by_pair = {int(pair_id): float((pair_ev or {}).get(int(pair_id), 0.0)) for pair_id in v19_config.pair_ids}
    rowbands = [
        {
            "base_id": f"rowband_{row['label']}_dir_{scale:+d}",
            "family": "inverse_solved_rowband",
            "regime": "inverse_solved_v17_receiver_preimage_direction",
            "rowband_candidate_id": row["label"],
            "scale": scale,
            "ev_proxy": (
                max(0.0, float(row.get("predicted_reduction", 0.0)))
                / max(1, abs(int(row.get("archive_byte_delta", 0))))
            ),
            "ev_rank_basis": "v17_predicted_reduction_per_archive_byte",
        }
        for row in ctx["v17_receipt"]["iterations"][0]["solve_candidates"]
        for scale in config.rowband_direction_scales
    ]
    rowbands.sort(key=lambda row: (-float(row["ev_proxy"]), row["base_id"]))
    q8 = [
        {
            "base_id": f"q8_{scope}_{scale:+d}",
            "family": "preuint8_q8_region",
            "regime": "model_free_exact_stage_coordinate",
            "scope": scope,
            "scale_q8": scale,
            "ev_proxy": (2.0 if scale > 0 else 0.5) / (1.0 + abs(scale) / 32.0),
            "ev_rank_basis": "v19_positive_q8_prior_times_small_quantum",
        }
        for scope, scales in (
            ("all", config.q8_all_scales),
            ("templates", config.q8_region_scales),
            ("sparse", config.q8_region_scales),
        )
        for scale in scales
    ]
    q8.extend(
        {
            "base_id": f"q8_pair_{pair_id:03d}_{scale:+d}",
            "family": "preuint8_q8_region",
            "regime": "model_free_pair_lifecycle_se_coupled_coordinate",
            "scope": f"pair:{pair_id}",
            "scale_q8": scale,
            "ev_proxy": ev_by_pair[pair_id] / (1.0 + abs(scale) / 32.0),
            "ev_rank_basis": "measured_fisher_margin_pair_ev_per_quantum",
        }
        for pair_id in v19_config.pair_ids
        for scale in config.q8_pair_scales
    )
    q8.sort(key=lambda row: (-float(row["ev_proxy"]), row["base_id"]))
    track_events = [
        {
            "base_id": f"worldsheet_track_{track_index:03d}_{axis}_{sign:+d}",
            "family": "worldsheet_track_event",
            "regime": "model_free_exact_forward_coordinate",
            "track_indices": [track_index],
            "axis": axis,
            "sign": sign,
            "ev_proxy": sum(
                ev_by_pair.get(int(lift.knots[knot].pair_index), 0.0) for knot in lift.tracks[track_index].knot_indices
            ),
            "ev_rank_basis": "measured_fisher_margin_track_lifecycle_ev",
        }
        for track_index in active_tracks
        for axis in ("x", "y")
        for sign in (-1, 1)
    ]
    track_events.sort(key=lambda row: (-float(row["ev_proxy"]), row["base_id"]))
    pair_events: list[dict[str, Any]] = []
    for pair_id in v19_config.pair_ids:
        tracks = [
            index
            for index in active_tracks
            if any(lift.knots[knot].pair_index == pair_id for knot in lift.tracks[index].knot_indices)
        ]
        for axis in ("x", "y"):
            for sign in (-1, 1):
                pair_events.append(
                    {
                        "base_id": f"worldsheet_pair_{pair_id:03d}_{axis}_{sign:+d}",
                        "family": "worldsheet_pair_event",
                        "regime": "model_free_pair_lifecycle_se_coupled_coordinate",
                        "track_indices": tracks,
                        "axis": axis,
                        "sign": sign,
                        "ev_proxy": ev_by_pair[pair_id],
                        "ev_rank_basis": "measured_fisher_margin_pair_ev",
                    }
                )
    pair_events.sort(key=lambda row: (-float(row["ev_proxy"]), row["base_id"]))
    template_swaps = [
        {
            "base_id": f"scorer_template_swap_{left:02d}_{right:02d}",
            "family": "scorer_template_swap",
            "regime": "model_free_exact_forward_template_assignment",
            "left": left,
            "right": right,
            "ev_proxy": 1.0 / (right - left),
            "ev_rank_basis": "early_adjacent_atom_alignment",
        }
        for left in range(6)
        for right in range(left + 1, 6)
    ]
    template_swaps.sort(key=lambda row: (-float(row["ev_proxy"]), row["base_id"]))
    grammar_swaps: list[dict[str, Any]] = []
    for pair_id in v19_config.pair_ids:
        candidates = [
            (track_index, position, knot_index)
            for track_index in active_tracks
            for position, knot_index in enumerate(lift.tracks[track_index].knot_indices)
            if lift.knots[knot_index].pair_index == pair_id
        ]
        if not candidates:
            continue
        track_index, position, knot_index = candidates[0]
        knots = lift.tracks[track_index].knot_indices
        for offset in (-1, 1):
            neighbor = knots[(position + offset) % len(knots)]
            replacement = lift.knots[neighbor].template_ref
            if replacement == lift.knots[knot_index].template_ref:
                continue
            grammar_swaps.append(
                {
                    "base_id": (f"grammar_event_pair_{pair_id:03d}_knot_{knot_index:04d}_neighbor_{offset:+d}"),
                    "family": "grammar_event_template_swap",
                    "regime": "model_free_exact_forward_grammar_template_substitution",
                    "knot_index": knot_index,
                    "replacement_template_ref": replacement,
                    "ev_proxy": ev_by_pair[pair_id],
                    "ev_rank_basis": "measured_fisher_margin_pair_ev",
                }
            )
    grammar_swaps.sort(key=lambda row: (-float(row["ev_proxy"]), row["base_id"]))
    groups = (
        rowbands,
        q8,
        track_events,
        pair_events,
        template_swaps,
        grammar_swaps,
    )
    if any(not group for group in groups):
        raise DirectDescriptionError("v19c proposal inventory lost a required family")
    inventory = _interleave(groups)
    if len(inventory) <= 10 or {row["family"] for row in inventory} != set(FAMILIES):
        raise DirectDescriptionError("v19c proposal inventory is not wider than v19b")
    return inventory


def _proposal_at(inventory: Sequence[Mapping[str, Any]], proposal_index: int) -> dict[str, Any]:
    cycle, offset = divmod(proposal_index, len(inventory))
    proposal = dict(inventory[offset])
    proposal["cycle"] = cycle
    proposal["proposal_index"] = proposal_index
    proposal["candidate_id"] = f"cycle_{cycle:03d}_{proposal['base_id']}"
    return proposal


def _apply_proposal(
    state: SaturationState,
    proposal: Mapping[str, Any],
    *,
    problem: Mapping[str, Any],
    ctx: Mapping[str, Any],
) -> SaturationState:
    templates = state.template_delta.copy()
    compensations = state.compensation_delta.copy()
    translations = {int(track): [int(dx), int(dy)] for track, dx, dy in state.track_translations}
    q8 = list(state.q8_directives)
    ref_swaps = {int(index): ref for index, ref in state.grammar_ref_swaps}
    family = str(proposal["family"])
    if family == "inverse_solved_rowband":
        candidate = _candidate_state(ctx, str(proposal["rowband_candidate_id"]))
        scale = int(proposal["scale"])
        templates += scale * (
            np.asarray(candidate["template_values_u8"], dtype=np.int16)
            - np.asarray(problem["initial_template_values_u8"], dtype=np.int16)
        )
        compensations += scale * (
            np.asarray(candidate["compensation_rgb_i8"], dtype=np.int16)
            - np.asarray(problem["initial_compensation_rgb_i8"], dtype=np.int16)
        )
    elif family == "preuint8_q8_region":
        q8.append((str(proposal["scope"]), int(proposal["scale_q8"])))
    elif family in {"worldsheet_track_event", "worldsheet_pair_event"}:
        axis = 0 if proposal["axis"] == "x" else 1
        sign = int(proposal["sign"])
        if not proposal["track_indices"]:
            raise DirectDescriptionError("v19c worldsheet proposal has no target track")
        for track_index in proposal["track_indices"]:
            value = translations.setdefault(int(track_index), [0, 0])
            value[axis] += sign
    elif family == "scorer_template_swap":
        realized = np.clip(
            np.asarray(problem["initial_template_values_u8"], dtype=np.int16) + templates,
            0,
            255,
        )
        left, right = int(proposal["left"]), int(proposal["right"])
        realized[[left, right]] = realized[[right, left]]
        templates = realized - np.asarray(problem["initial_template_values_u8"], dtype=np.int16)
    elif family == "grammar_event_template_swap":
        ref_swaps[int(proposal["knot_index"])] = str(proposal["replacement_template_ref"])
    else:
        raise DirectDescriptionError(f"v19c unsupported proposal family: {family}")
    return SaturationState(
        template_delta=templates,
        compensation_delta=compensations,
        base_grammar_move_ids=state.base_grammar_move_ids,
        base_preuint8_scale_q8=state.base_preuint8_scale_q8,
        track_translations=tuple(
            (track, value[0], value[1]) for track, value in sorted(translations.items()) if value != [0, 0]
        ),
        q8_directives=tuple(q8),
        grammar_ref_swaps=tuple(sorted(ref_swaps.items())),
        move_ids=(*state.move_ids, str(proposal["candidate_id"])),
    )


def _worldsheet_payload(
    archive: bytes,
    *,
    active_pair_ids: Sequence[int],
    base_grammar_move_ids: Sequence[str],
    track_translations: Sequence[tuple[int, int, int]],
    grammar_ref_swaps: Sequence[tuple[int, str]],
) -> tuple[bytes, dict[str, Any]]:
    from tools.measure_ddm_v19b_joint_remeasure_stack import (
        _worldsheet_payload as _v19b_worldsheet_payload,
    )

    payload, base_receipt = _v19b_worldsheet_payload(
        archive,
        active_pair_ids=active_pair_ids,
        grammar_move_ids=base_grammar_move_ids,
    )
    if not track_translations and not grammar_ref_swaps:
        return payload, {
            **base_receipt,
            "v19c_track_translation_count": 0,
            "v19c_grammar_ref_swap_count": 0,
        }
    lift = lift_g1_movable_worldsheet(payload)
    knots = list(lift.knots)
    known_refs = {row.template_ref for row in lift.templates}
    for knot_index, template_ref in grammar_ref_swaps:
        if not 0 <= knot_index < len(knots) or template_ref not in known_refs:
            raise DirectDescriptionError("v19c grammar template substitution is invalid")
        knots[knot_index] = replace(knots[knot_index], template_ref=template_ref)
    lift = replace(lift, knots=tuple(knots))
    knots = list(lift.knots)
    for track_index, dx, dy in track_translations:
        if not 0 <= track_index < len(lift.tracks):
            raise DirectDescriptionError("v19c worldsheet track index is invalid")
        x_bounds, y_bounds = _g1_track_translation_bounds(lift, track_index)
        if not x_bounds[0] <= dx <= x_bounds[1] or not y_bounds[0] <= dy <= y_bounds[1]:
            raise DirectDescriptionError("v19c worldsheet event leaves the scorer grid")
        for knot_index in lift.tracks[track_index].knot_indices:
            knot = knots[knot_index]
            knots[knot_index] = replace(knot, center_x=knot.center_x + dx, center_y=knot.center_y + dy)
    compiled = encode_lifted_g1_movable_worldsheet(replace(lift, knots=tuple(knots)))
    return compiled, {
        **base_receipt,
        "compiled_payload_sha256": _sha256(compiled),
        "v19c_track_translation_count": len(track_translations),
        "v19c_track_translations": [list(row) for row in track_translations],
        "v19c_grammar_ref_swap_count": len(grammar_ref_swaps),
        "v19c_grammar_ref_swap_knot_indices": [int(index) for index, _ref in grammar_ref_swaps],
    }


def _scope_scale(
    directives: Sequence[tuple[str, int]],
    *,
    kind: Literal["templates", "sparse"],
    pair_id: int,
) -> int:
    result = 0
    for scope, scale in directives:
        if scope == "all" or scope == kind or scope == f"pair:{pair_id}":
            result += int(scale)
    return result


def _preuint8_program(
    *,
    state: SaturationState,
    problem: Mapping[str, Any],
    ctx: Mapping[str, Any],
    coupled_archive: bytes,
    source_start: int,
    source_stop: int,
) -> PreUint8Q8ProgramV1:
    candidate = _candidate_state(ctx, FIRST_MOVE_ID)
    template_delta = np.asarray(candidate["template_values_u8"], dtype=np.int16) - np.asarray(
        problem["initial_template_values_u8"], dtype=np.int16
    )
    sparse_delta = np.asarray(candidate["compensation_rgb_i8"], dtype=np.int16) - np.asarray(
        problem["initial_compensation_rgb_i8"], dtype=np.int16
    )
    receiver = receive_coupled_margin_archive(coupled_archive)
    templates = []
    for placement in receiver.program.placements:
        pair_id = int(placement.source_pair_id)
        scale = state.base_preuint8_scale_q8 + _scope_scale(state.q8_directives, kind="templates", pair_id=pair_id)
        delta = template_delta[placement.template_index]
        if scale and np.any(delta):
            templates.append(
                TemplateQ8CorrectionV1(
                    pair_id,
                    placement.template_index,
                    tuple(int(value) * scale for value in delta.reshape(-1)),
                )
            )
    sparse = []
    for index, support in enumerate(problem["sparse_compensation_support"]):
        pair_id = int(support["source_pair_id"])
        if not source_start <= pair_id < source_stop:
            continue
        scale = state.base_preuint8_scale_q8 + _scope_scale(state.q8_directives, kind="sparse", pair_id=pair_id)
        if scale and np.any(sparse_delta[index]):
            sparse.append(
                SparseQ8CorrectionV1(
                    pair_id,
                    int(support["frame_index"]),
                    int(support["camera_y"]),
                    int(support["camera_x"]),
                    tuple(int(value) * scale for value in sparse_delta[index]),
                )
            )
    return PreUint8Q8ProgramV1(tuple(sorted(templates)), tuple(sorted(sparse)), "bayer8", 210)


def _compile_state(
    state: SaturationState,
    *,
    rung: Literal["dev", "n600"],
    v19_config: DDMV19PurePricedObjectiveConfigV1,
    ctx: Mapping[str, Any],
) -> tuple[bytes, Callable[[bytes], Any], dict[str, Any]]:
    geometry = _rung_geometry(rung, v19_config, ctx)
    realized = _realized_post_state(state, ctx["problem"])
    _compact, expanded_base, program = _ladder_archive(
        state=realized,
        problem=ctx["problem"],
        v14_archive=_base_v14_bytes(ctx["n600_config"]),
        source_start=geometry["source_start"],
        source_stop=geometry["source_stop"],
    )
    worldsheet, grammar = _worldsheet_payload(
        geometry["source_archive"],
        active_pair_ids=geometry["grammar_pair_ids"],
        base_grammar_move_ids=state.base_grammar_move_ids,
        track_translations=state.track_translations,
        grammar_ref_swaps=state.grammar_ref_swaps,
    )
    carrier = _carrier_with_worldsheet(expanded_base, worldsheet)
    coupled = compile_coupled_margin_archive(carrier, program)
    q8_program = _preuint8_program(
        state=state,
        problem=ctx["problem"],
        ctx=ctx,
        coupled_archive=coupled,
        source_start=geometry["source_start"],
        source_stop=geometry["source_stop"],
    )
    archive = compile_preuint8_q8_archive(coupled, q8_program)
    return (
        archive,
        receive_preuint8_q8_archive,
        {
            "rung": rung,
            "move_count_total_including_v19b": len(state.move_ids),
            "v19c_move_count": len(state.move_ids) - 10,
            "wrapper": "preuint8_q8(coupled_margin(carrier_compose))",
            "archive_bytes": len(archive),
            "archive_sha256": _sha256(archive),
            "grammar": grammar,
            "post_int8": {
                "template_delta_nonzero": int(np.count_nonzero(state.template_delta)),
                "compensation_delta_nonzero": int(np.count_nonzero(state.compensation_delta)),
                "placement_records": len(program.placements),
                "sparse_compensation_records": len(program.compensations),
            },
            "preuint8": {
                "base_scale_q8": state.base_preuint8_scale_q8,
                "directive_count": len(state.q8_directives),
                "template_records": len(q8_program.templates),
                "sparse_records": len(q8_program.sparse),
            },
            "source_start": geometry["source_start"],
            "source_stop": geometry["source_stop"],
            "source_pair_ids": list(geometry["source_pair_ids"]),
            "local_pair_ids": list(geometry["local_pair_ids"]),
        },
    )


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)[:180]


def _family_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for family in FAMILIES:
        members = [row for row in rows if row["proposal"]["family"] == family]
        admitted = [row for row in members if row.get("accepted") is True]
        summary[family] = {
            "proposals_measured_or_classified": len(members),
            "admitted": len(admitted),
            "strict_joint_gain": sum(
                max(0.0, -float(row["joint_incremental_delta"]["joint_delta"])) for row in admitted
            ),
            "compile_infeasible": sum(row.get("disposition") == "INFEASIBLE_COMPILE" for row in members),
        }
    return summary


def _stage_dev(
    config: DDMV19CCorrectionSaturationConfigV1,
    root: Path,
    v19b_receipt: Mapping[str, Any],
    v19_config: DDMV19PurePricedObjectiveConfigV1,
    v19_receipt: Mapping[str, Any],
    ctx: Mapping[str, Any],
) -> dict[str, Any]:
    final_path = root / "stage_checkpoints" / "01_dev_saturation.json"
    resumed = _load_stage(final_path, config.typed_config_hash())
    if resumed is not None:
        return resumed
    state = _initial_state(v19b_receipt, v19_receipt, ctx)
    initial_archive, initial_factory, initial_compile = _compile_state(
        state, rung="dev", v19_config=v19_config, ctx=ctx
    )
    expected = v19b_receipt["greedy_screen"]["final_stack"]
    if _sha256(initial_archive) != expected["archive"]["sha256"]:
        raise DirectDescriptionError("v19c did not reproduce the v19b DEV archive")
    current, current_logits, current_cells, current_camera = _measure(
        archive=initial_archive,
        receiver_factory=initial_factory,
        pair_ids=v19_config.pair_ids,
        labels_all=ctx["labels_all"],
        poses_all=ctx["poses_all"],
        segnet=ctx["segnet"],
        posenet=ctx["posenet"],
    )
    if current["cells_sha256"] != expected["measurement"]["cells_sha256"]:
        raise DirectDescriptionError("v19c did not reproduce v19b DEV scorer cells")
    labels = np.asarray(ctx["labels_all"][np.asarray(v19_config.pair_ids, dtype=np.int64)])
    sorted_logits = np.sort(current_logits, axis=1)
    margins = sorted_logits[:, -1] - sorted_logits[:, -2]
    pair_ev_values: dict[int, float] = {}
    for local_index, pair_id in enumerate(v19_config.pair_ids):
        errors = current_cells[local_index] != labels[local_index]
        role_weight = np.where(labels[local_index] == 1, 1.25, 1.0)
        pair_ev_values[int(pair_id)] = float(
            np.sum(
                role_weight[errors] / np.maximum(margins[local_index][errors], 1e-8),
                dtype=np.float64,
            )
        )
    inventory = _proposal_inventory(config, v19_config, ctx, pair_ev=pair_ev_values)
    row_root = root / "stage_checkpoints" / "01_dev_candidates"
    rows: list[dict[str, Any]] = []
    consecutive_failures = 0
    proposal_index = 0
    initial_bytes = len(initial_archive)
    existing = sorted(row_root.glob("proposal_*.json"))
    for expected_index, checkpoint in enumerate(existing):
        proposal = _proposal_at(inventory, expected_index)
        prior = _load_stage(checkpoint, config.typed_config_hash())
        if prior is None or prior.get("proposal", {}).get("candidate_id") != proposal["candidate_id"]:
            raise DirectDescriptionError("v19c DEV resume proposal order differs")
        rows.append(prior)
        if prior["accepted"]:
            state = _apply_proposal(state, proposal, problem=ctx["problem"], ctx=ctx)
        consecutive_failures = int(prior["consecutive_failures_after"])
        proposal_index += 1
    if rows:
        resumed_archive, resumed_factory, _resumed_compile = _compile_state(
            state, rung="dev", v19_config=v19_config, ctx=ctx
        )
        current, current_cells, current_camera = _measure_dev(
            archive=resumed_archive,
            receiver_factory=resumed_factory,
            v19_config=v19_config,
            ctx=ctx,
        )
        resumed_expected = rows[-1]["current_after"]
        if (
            current["archive_sha256"] != resumed_expected["archive_sha256"]
            or current["cells_sha256"] != resumed_expected["cells_sha256"]
        ):
            raise DirectDescriptionError("v19c DEV resumed scorer endpoint differs")
    while True:
        if consecutive_failures >= config.consecutive_failure_limit:
            stop_reason = "K_CONSECUTIVE_PROPOSALS_FAILED"
            break
        if int(current["archive_bytes"]) - initial_bytes >= config.correction_budget_bytes:
            stop_reason = "CORRECTION_BUDGET_EXHAUSTED"
            break
        proposal = _proposal_at(inventory, proposal_index)
        checkpoint = row_root / (f"proposal_{proposal_index:05d}_{_safe_name(proposal['base_id'])}.json")
        if checkpoint.exists():
            raise DirectDescriptionError("v19c DEV checkpoint gap or duplicate detected")
        trial_state = _apply_proposal(state, proposal, problem=ctx["problem"], ctx=ctx)
        try:
            archive, factory, compile_receipt = _compile_state(trial_state, rung="dev", v19_config=v19_config, ctx=ctx)
        except (DirectDescriptionError, ValueError, OverflowError) as exc:
            consecutive_failures += 1
            row = {
                "schema": "ddm_v19c_dev_proposal.v1",
                "typed_config_sha256": config.typed_config_hash(),
                "proposal": proposal,
                "accepted": False,
                "disposition": "INFEASIBLE_COMPILE",
                "compile_error": f"{type(exc).__name__}: {exc}",
                "current_after": current,
                "consecutive_failures_after": consecutive_failures,
                "score_claim": False,
                "evidence_axis": AXIS,
            }
            _write(checkpoint, row)
            rows.append(row)
            proposal_index += 1
            continue
        added_bytes = len(archive) - initial_bytes
        archive_path = (
            root
            / "candidate_archives"
            / (f"dev_{proposal_index:05d}_{_safe_name(proposal['base_id'])}.zip.receipt-bytes")
        )
        _publish_immutable(archive_path, archive)
        if added_bytes > config.correction_budget_bytes:
            consecutive_failures += 1
            row = {
                "schema": "ddm_v19c_dev_proposal.v1",
                "typed_config_sha256": config.typed_config_hash(),
                "proposal": proposal,
                "accepted": False,
                "disposition": "REJECTED_CORRECTION_BUDGET",
                "archive": {
                    "path": _portable(archive_path),
                    "bytes": len(archive),
                    "sha256": _sha256(archive),
                },
                "compile": compile_receipt,
                "added_bytes_vs_v19b": added_bytes,
                "current_after": current,
                "consecutive_failures_after": consecutive_failures,
                "score_claim": False,
                "evidence_axis": AXIS,
            }
            _write(checkpoint, row)
            rows.append(row)
            proposal_index += 1
            continue
        trial, trial_cells, trial_camera = _measure_dev(
            archive=archive,
            receiver_factory=factory,
            v19_config=v19_config,
            ctx=ctx,
            baseline_camera=current_camera,
            baseline_cells=current_cells,
        )
        delta = _delta_payload(current, trial)
        accepted = bool(delta["accepted"])
        consecutive_failures = 0 if accepted else consecutive_failures + 1
        row = {
            "schema": "ddm_v19c_dev_proposal.v1",
            "typed_config_sha256": config.typed_config_hash(),
            "proposal": proposal,
            "accepted": accepted,
            "disposition": ("ADMITTED_STRICT_DEV_JOINT_DELTA" if accepted else "REJECTED_NONNEGATIVE_DEV_JOINT_DELTA"),
            "archive": {
                "path": _portable(archive_path),
                "bytes": len(archive),
                "sha256": _sha256(archive),
            },
            "compile": compile_receipt,
            "before": current,
            "trial": trial,
            "current_after": trial if accepted else current,
            "joint_incremental_delta": delta,
            "bucket_transition": _bucket_transition(current_cells, trial_cells, labels),
            "added_bytes_vs_v19b": added_bytes,
            "consecutive_failures_after": consecutive_failures,
            "score_claim": False,
            "evidence_axis": AXIS,
        }
        _write(checkpoint, row)
        rows.append(row)
        if accepted:
            state = trial_state
            current = trial
            current_cells = trial_cells
            current_camera = trial_camera
        proposal_index += 1
    final_archive, _factory, final_compile = _compile_state(state, rung="dev", v19_config=v19_config, ctx=ctx)
    final_archive_path = root / "ddm_v19c_final_dev.zip.receipt-bytes"
    _publish_immutable(final_archive_path, final_archive)
    accepted = [row for row in rows if row.get("accepted") is True]
    result = {
        "schema": "ddm_v19c_dev_saturation.v1",
        "typed_config_sha256": config.typed_config_hash(),
        "status": "MEASURED",
        "stop_reason": stop_reason,
        "proposal_inventory_unique_coordinate_count": len(inventory),
        "fisher_margin_ev_rank": {
            "law_ids": [
                "frozen_scorer_fisher_curvature_margin_colocation_v1",
                "fisher_curvature_equals_categorical_fisher_trace_caustic_v1",
                "witness_measured_reverse_waterfill_v1",
            ],
            "pair_ev": {str(pair_id): pair_ev_values[pair_id] for pair_id in sorted(pair_ev_values)},
            "pair_ev_formula": (
                "sum_error_cells class_weight/max(top1_minus_top2_margin,1e-8); "
                "Lane class_weight=1.25 from registered larger-normal cheap-flip prior"
            ),
            "cross_family_authority": ("strict realized joint delta; family EV proxies are not comparable units"),
            "fourier_basis_used": False,
        },
        "proposal_count": len(rows),
        "consecutive_failures_at_stop": consecutive_failures,
        "accepted_count": len(accepted),
        "accepted_proposal_ids": [row["proposal"]["candidate_id"] for row in accepted],
        "per_family": _family_summary(rows),
        "initial": {
            "archive_bytes": initial_bytes,
            "archive_sha256": _sha256(initial_archive),
            "measurement": expected["measurement"],
            "compile": initial_compile,
        },
        "final": {
            "archive": {
                "path": _portable(final_archive_path),
                "bytes": len(final_archive),
                "sha256": _sha256(final_archive),
            },
            "measurement": current,
            "compile": final_compile,
            "added_bytes_vs_v19b": len(final_archive) - initial_bytes,
        },
        "accepted_curve_dev": [
            {
                "admission_index": index,
                "proposal_id": row["proposal"]["candidate_id"],
                "family": row["proposal"]["family"],
                "archive_bytes": row["current_after"]["archive_bytes"],
                "d_seg": row["current_after"]["d_seg"],
                "d_pose": row["current_after"]["d_pose"],
                "incremental_joint_delta": row["joint_incremental_delta"]["joint_delta"],
                "cumulative_joint_delta_vs_v19b": _delta_payload(expected["measurement"], row["current_after"])[
                    "joint_delta"
                ],
            }
            for index, row in enumerate(accepted)
        ],
        "candidate_checkpoint_glob": _portable(row_root / "proposal_*.json"),
        "score_claim": False,
        "evidence_axis": AXIS,
    }
    _write(final_path, result)
    return result


def _bucket_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    for name in (
        "role_bucket_Lane_plus_Movable",
        "residual_bucket_Road_Undrivable_MyCar",
    ):
        errors_before = int(before["c1_buckets"][name]["errors"])
        errors_after = int(after["c1_buckets"][name]["errors"])
        result[name] = {
            "errors_before": errors_before,
            "errors_after": errors_after,
            "realized_net_flips": errors_before - errors_after,
        }
    return result


def _restore_n600_decisions(
    *,
    decision_root: Path,
    config_hash: str,
    proposal_indices: Sequence[int],
    inventory: Sequence[Mapping[str, Any]],
    state: SaturationState,
    problem: Mapping[str, Any],
    ctx: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], SaturationState, Mapping[str, Any] | None]:
    rows: list[dict[str, Any]] = []
    current: Mapping[str, Any] | None = None
    for expected_index, checkpoint in enumerate(sorted(decision_root.glob("candidate_*.json"))):
        if expected_index >= len(proposal_indices):
            raise DirectDescriptionError("v19c n600 resume has more decisions than DEV admissions")
        proposal_index = proposal_indices[expected_index]
        proposal = _proposal_at(inventory, proposal_index)
        prior = _load_stage(checkpoint, config_hash)
        if (
            prior is None
            or int(prior.get("dev_admission_index", -1)) != expected_index
            or prior.get("proposal", {}).get("candidate_id") != proposal["candidate_id"]
        ):
            raise DirectDescriptionError("v19c n600 resume proposal order differs")
        rows.append(prior)
        if prior["accepted"]:
            state = _apply_proposal(state, proposal, problem=problem, ctx=ctx)
        current = prior["current_after"]
    return rows, state, current


def _stage_n600(
    config: DDMV19CCorrectionSaturationConfigV1,
    root: Path,
    v19b_receipt: Mapping[str, Any],
    v19_config: DDMV19PurePricedObjectiveConfigV1,
    v19_receipt: Mapping[str, Any],
    ctx: Mapping[str, Any],
    dev: Mapping[str, Any],
) -> dict[str, Any]:
    final_path = root / "stage_checkpoints" / "02_n600_saturation_curve.json"
    resumed = _load_stage(final_path, config.typed_config_hash())
    if resumed is not None:
        return resumed
    inventory = _proposal_inventory(
        config,
        v19_config,
        ctx,
        pair_ev={int(pair_id): float(value) for pair_id, value in dev["fisher_margin_ev_rank"]["pair_ev"].items()},
    )
    state = _initial_state(v19b_receipt, v19_receipt, ctx)
    baseline = v19b_receipt["n600"]["measurement"]
    initial_bytes = int(baseline["archive_bytes"])
    accepted_dev = {
        row["proposal"]["proposal_index"]: row
        for row in (
            json.loads(path.read_bytes())
            for path in sorted((root / "stage_checkpoints" / "01_dev_candidates").glob("proposal_*.json"))
        )
        if row.get("accepted") is True
    }
    geometry = _rung_geometry("n600", v19_config, ctx)
    proposal_indices = sorted(accepted_dev)
    decision_root = root / "stage_checkpoints" / "02_n600_decisions"
    rows, state, restored_current = _restore_n600_decisions(
        decision_root=decision_root,
        config_hash=config.typed_config_hash(),
        proposal_indices=proposal_indices,
        inventory=inventory,
        state=state,
        problem=ctx["problem"],
        ctx=ctx,
    )
    current = restored_current or baseline
    if rows:
        resumed_archive, _resumed_factory, _resumed_compile = _compile_state(
            state, rung="n600", v19_config=v19_config, ctx=ctx
        )
        if (
            _sha256(resumed_archive) != current["archive_sha256"]
            or len(resumed_archive) != int(current["archive_bytes"])
        ):
            raise DirectDescriptionError("v19c n600 resumed archive endpoint differs")
    for n600_index, proposal_index in enumerate(proposal_indices[len(rows) :], start=len(rows)):
        proposal = _proposal_at(inventory, proposal_index)
        trial_state = _apply_proposal(state, proposal, problem=ctx["problem"], ctx=ctx)
        archive, factory, compile_receipt = _compile_state(trial_state, rung="n600", v19_config=v19_config, ctx=ctx)
        archive_path = (
            root / "candidate_archives" / (f"n600_{n600_index:04d}_{_safe_name(proposal['base_id'])}.zip.receipt-bytes")
        )
        _publish_immutable(archive_path, archive)
        measurement = _measure_window(
            name=f"02_n600_candidates/candidate_{n600_index:04d}",
            archive=archive,
            receiver_factory=factory,
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
        delta = _delta_payload(current, measurement)
        budget_ok = int(measurement["archive_bytes"]) - initial_bytes <= config.correction_budget_bytes
        accepted = bool(delta["accepted"]) and budget_ok
        row = {
            "schema": "ddm_v19c_n600_proposal.v1",
            "typed_config_sha256": config.typed_config_hash(),
            "dev_admission_index": n600_index,
            "proposal": proposal,
            "accepted": accepted,
            "disposition": (
                "ADMITTED_STRICT_N600_JOINT_DELTA"
                if accepted
                else "REJECTED_N600_BUDGET"
                if not budget_ok
                else "REJECTED_NONNEGATIVE_N600_JOINT_DELTA"
            ),
            "archive": {
                "path": _portable(archive_path),
                "bytes": len(archive),
                "sha256": _sha256(archive),
            },
            "compile": compile_receipt,
            "before": current,
            "trial": measurement,
            "current_after": measurement if accepted else current,
            "joint_incremental_delta": delta,
            "bucket_transition": _bucket_delta(current, measurement),
            "score_claim": False,
            "evidence_axis": AXIS,
        }
        checkpoint = decision_root / f"candidate_{n600_index:04d}.json"
        if checkpoint.exists():
            raise DirectDescriptionError("v19c n600 checkpoint gap or duplicate detected")
        _write(checkpoint, row)
        rows.append(row)
        if accepted:
            state = trial_state
            current = measurement
    final_archive, final_factory, final_compile = _compile_state(state, rung="n600", v19_config=v19_config, ctx=ctx)
    final_path_archive = root / "ddm_v19c_final_n600.zip.receipt-bytes"
    _publish_immutable(final_path_archive, final_archive)
    if _sha256(final_archive) != current["archive_sha256"]:
        raise DirectDescriptionError("v19c n600 final archive/measurement identity differs")
    accepted = [row for row in rows if row["accepted"]]
    result = {
        "schema": "ddm_v19c_n600_saturation_curve.v1",
        "typed_config_sha256": config.typed_config_hash(),
        "status": "MEASURED",
        "dev_admission_count": len(rows),
        "n600_admission_count": len(accepted),
        "n600_rejection_count": len(rows) - len(accepted),
        "decisions": rows,
        "accepted_curve_n600": [
            {
                "admission_index": index,
                "proposal_id": row["proposal"]["candidate_id"],
                "family": row["proposal"]["family"],
                "archive_bytes": row["current_after"]["archive_bytes"],
                "d_seg": row["current_after"]["d_seg"],
                "d_pose": row["current_after"]["d_pose"],
                "incremental_joint_delta": row["joint_incremental_delta"]["joint_delta"],
                "cumulative_joint_delta_vs_v19b": _delta_payload(baseline, row["current_after"])["joint_delta"],
                "bucket_transition": row["bucket_transition"],
            }
            for index, row in enumerate(accepted)
        ],
        "initial_v19b": baseline,
        "final": {
            "archive": {
                "path": _portable(final_path_archive),
                "bytes": len(final_archive),
                "sha256": _sha256(final_archive),
            },
            "measurement": current,
            "compile": final_compile,
            "added_bytes_vs_v19b": len(final_archive) - initial_bytes,
            "cumulative_joint_delta_vs_v19b": _delta_payload(baseline, current),
            "bucket_delta_vs_v19b": _bucket_delta(baseline, current),
        },
        "per_family": _family_summary(rows),
        "template_order_gauge": _template_order_gauge(final_archive, final_factory),
        "score_claim": False,
        "evidence_axis": AXIS,
    }
    _write(final_path, result)
    return result


def _artifact_hygiene(root: Path) -> dict[str, Any]:
    """Certify retained bytes; this runner has no throwaway scratch tree."""

    durable_files = [path for path in root.rglob("*") if path.is_file() and ".tmp" not in path.name]
    total = sum(path.stat().st_size for path in durable_files)
    return {
        "schema": "ddm_v19c_artifact_hygiene.v1",
        "durable_file_count": len(durable_files),
        "durable_bytes": total,
        "scratch_bytes_deleted": 0,
        "scratch_policy": "context-free; runner writes atomically to durable checkpoints",
        "large_artifact_action": (
            "LOCAL_DURABLE_BELOW_512_MIB"
            if total <= 512 * 1024 * 1024
            else "BLOCKED_REQUIRES_CERTIFIED_SSD_COLD_STORE_BEFORE_CLEANUP"
        ),
        "candidate_archives_rebuildable_from": ("typed config + v19b/v19 SHA chain + proposal checkpoint sequence"),
        "destructive_cleanup_performed": False,
    }


def _final(
    config: DDMV19CCorrectionSaturationConfigV1,
    root: Path,
    storage: Mapping[str, Any],
    v19b_receipt: Mapping[str, Any],
    dev: Mapping[str, Any],
    n600: Mapping[str, Any],
) -> Path:
    path = root / "ddm_v19c_correction_saturation_receipt.json"
    if path.exists():
        value = json.loads(path.read_bytes())
        if value.get("typed_config_sha256") != config.typed_config_hash():
            raise DirectDescriptionError("v19c final receipt identity differs")
        return path
    baseline = n600["initial_v19b"]
    final = n600["final"]["measurement"]
    role_before = int(baseline["c1_buckets"]["role_bucket_Lane_plus_Movable"]["errors"])
    role_after = int(final["c1_buckets"]["role_bucket_Lane_plus_Movable"]["errors"])
    residual_before = int(baseline["c1_buckets"]["residual_bucket_Road_Undrivable_MyCar"]["errors"])
    residual_after = int(final["c1_buckets"]["residual_bucket_Road_Undrivable_MyCar"]["errors"])
    role_gain = role_before - role_after
    residual_gain = residual_before - residual_after
    total_gain = role_gain + residual_gain
    if n600["n600_admission_count"]:
        verdict = "V19C_CORRECTION_SATURATION_ADMITTED_N600_ADVISORY"
    else:
        verdict = "V19C_WIDE_SCREEN_NO_ADDITIONAL_N600_ADMISSION_INSTANCE_ONLY"
    result = {
        "schema": SCHEMA,
        "run_id": config.run_id,
        "lane_id": LANE_ID,
        "typed_config": config.model_dump(mode="json", by_alias=True),
        "typed_config_sha256": config.typed_config_hash(),
        "source_v19b": {
            "config_path": config.v19b_config_path,
            "config_sha256": config.v19b_config_sha256,
            "receipt_path": config.v19b_receipt_path,
            "receipt_sha256": config.v19b_receipt_sha256,
            "starting_move_count": int(v19b_receipt["accepted_move_count"]),
            "starting_archive_sha256": baseline["archive_sha256"],
        },
        "verdict": verdict,
        "verdict_scope": (
            "INSTANCE:V19C x SHA-bound v19b start x interleaved finite coordinate "
            "families x recursive K-failure stop x sequential n600 replay; family "
            "negatives remain open; macOS-CPU advisory only; no contest score, "
            "promotion, or pointer movement"
        ),
        "asymptote": {
            "dev_stop_reason": dev["stop_reason"],
            "consecutive_failures_at_stop": dev["consecutive_failures_at_stop"],
            "unique_coordinate_inventory": dev["proposal_inventory_unique_coordinate_count"],
            "dev_proposals": dev["proposal_count"],
            "dev_admissions": dev["accepted_count"],
            "n600_admissions": n600["n600_admission_count"],
            "correction_budget_bytes": config.correction_budget_bytes,
            "correction_bytes_used_n600": n600["final"]["added_bytes_vs_v19b"],
            "family_optimum_claimed": False,
        },
        "curve": {
            "dev_per_admitted_move": dev["accepted_curve_dev"],
            "n600_per_admitted_move": n600["accepted_curve_n600"],
            "n600_endpoint": {
                "archive_bytes": final["archive_bytes"],
                "archive_sha256": final["archive_sha256"],
                "d_seg": final["d_seg"],
                "d_pose": final["d_pose"],
                "joint_delta_vs_v19b": n600["final"]["cumulative_joint_delta_vs_v19b"],
            },
        },
        "family_attribution": {
            "dev": dev["per_family"],
            "n600": n600["per_family"],
        },
        "fisher_margin_reverse_waterfill": dev["fisher_margin_ev_rank"],
        "c1_bucket_attribution": {
            "canonical_ceiling_reference": {
                "continuous_in_box_debt": C1_CONTINUOUS_DEBT,
                "lane_plus_movable_ceiling_errors": C1_ROLE_CEILING_ERRORS,
                "integer_target_errors": C1_TARGET_ERRORS,
                "residual_after_perfect_role_errors": C1_RESIDUAL_AFTER_PERFECT_ROLE,
                "derivation": ("3,240,528 v15 errors - 726,416 role ceiling - 136,839 integer target = 2,377,273"),
            },
            "measured_v19b_start": {
                "role_errors": role_before,
                "residual_errors": residual_before,
            },
            "measured_v19c_final": {
                "role_errors": role_after,
                "residual_errors": residual_after,
            },
            "v19c_incremental_realized_net_flips": {
                "role_bucket_Lane_plus_Movable": role_gain,
                "residual_bucket_Road_Undrivable_MyCar": residual_gain,
                "total": total_gain,
                "residual_fraction": (None if total_gain == 0 else residual_gain / total_gain),
            },
            "classification": (
                "RESIDUAL_DOMINANT"
                if residual_gain > role_gain
                else "ROLE_DOMINANT"
                if role_gain > residual_gain
                else "BALANCED_OR_ZERO"
            ),
        },
        "atom_order_gauge": {
            **n600["template_order_gauge"],
            "operator_application_order_note": (
                "row-band clipping and stateful template substitutions are "
                "noncommutative coordinates, not a free atom-order gauge"
            ),
        },
        "application_stage": {
            "post_int8": "MEASURED",
            "camera_q8_pre_final_uint8": "MEASURED_AT_MULTIPLE_QUANTA_AND_REGIONS",
            "worldsheet_grammar": "MEASURED_PER_TRACK_PER_PAIR_AND_EVENT_TEMPLATE",
            "joint_same_frame_se_effects": "PRESERVED_BY_TWO_FRAME_PAIR_LIFECYCLE_BATCHING",
        },
        "inverse_and_recursion_custody": {
            "inverse_solved": (
                "v17 row-band states are exact receiver-preimage directions; "
                "their repeated signed use is still exact forward-measured"
            ),
            "model_free": (
                "Q8, worldsheet, scorer-template, and grammar-event coordinates "
                "are exact forward proposals with no inverse certificate claimed"
            ),
            "receiver_omniscience": (
                "every admission uses realized receiver/scorer/rate terms; "
                "omniscience is finite measured coordinate enumeration only"
            ),
            "recursion": (
                "inventory cycles repeat from the current accepted state until "
                "64 consecutive failures or 200 kB exhaustion"
            ),
        },
        "rate_price_per_byte": break_even_distortion_gain_per_byte(),
        "source_bytes": SOURCE_BYTES,
        "storage_preflight": dict(storage),
        "artifact_hygiene": _artifact_hygiene(root),
        "resume": {
            "immutable_dev_proposal_checkpoints": True,
            "immutable_n600_per_batch_checkpoints": True,
            "candidate_and_final_archives_preserved": True,
            "maximum_lost_work": "one proposal or one 16-pair scorer batch",
        },
        "triality": {
            "dsl": "DDMV19CCorrectionSaturationConfigV1",
            "dag": ".omx/research/ddm_v19c_correction_saturation_DAG_FEED_20260723.md",
            "equation": ".omx/research/ddm_v19c_correction_saturation_canonical_equations_20260723.md",
        },
        "stores_consulted": [
            "CLAUDE.md",
            "AGENTS.md",
            "docs/operating_manual_craft_handoff.md",
            config.v19b_receipt_path,
            ".omx/research/codex_findings_ddm_v19b_joint_remeasure_stack_20260723_codex.md",
            ".omx/research/codex_session_summary_20260723_ddm_v19b_codex.md",
            ".omx/research/ddm_c1_composed_candidate_spec_603_613_20260723.md",
            ".omx/research/ddm_c1_composed_candidate_ledger_603_613_20260723.json",
            "operator inbox and Codex broadcast through run start",
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


def run(config: DDMV19CCorrectionSaturationConfigV1, root: Path, stage: str) -> Path:
    root = root.resolve()
    storage = _deterministic_storage_receipt(_storage_preflight(root))
    root.mkdir(parents=True, exist_ok=True)
    (
        _v19b_config,
        v19b_receipt,
        v19_config,
        v19_receipt,
        ctx,
    ) = _bound_v19b(config)
    dev = _stage_dev(config, root, v19b_receipt, v19_config, v19_receipt, ctx)
    if stage == "dev":
        return root / "stage_checkpoints" / "01_dev_saturation.json"
    n600 = _stage_n600(
        config,
        root,
        v19b_receipt,
        v19_config,
        v19_receipt,
        ctx,
        dev,
    )
    if stage == "n600":
        return root / "stage_checkpoints" / "02_n600_saturation_curve.json"
    return _final(config, root, storage, v19b_receipt, dev, n600)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--stage", choices=("dev", "n600", "all"), default="all")
    args = parser.parse_args()
    config = DDMV19CCorrectionSaturationConfigV1.model_validate_json(_read_regular_file_once(args.config))
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
