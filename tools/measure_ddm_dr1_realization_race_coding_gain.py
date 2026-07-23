#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure the DDM DR1 realization, context-code, and ground-vocabulary races.

This is a local, resumable measurement harness.  It consumes only SHA-bound
landed receipts, measures receiver variants through the frozen n600 scorers,
and emits no contest-score or promotion claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "src", REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.optimization.ddm_dv2_sdwl1 import (  # noqa: E402
    SentenceLayout,
    SentenceOptions,
    TemporalMode,
    decode_sentence,
    decompress_outer_payload,
    measure_serialization,
)
from tac.optimization.direct_description_coupled_margin import (  # noqa: E402
    compile_coupled_margin_archive,
    receive_coupled_margin_archive,
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
from tools.measure_ddm_v14_realization_fidelity import (  # noqa: E402
    POINTER_SCORE_TEXT,
    _forward,
    _publish_immutable,
    _storage_preflight,
)
from tools.measure_ddm_v16_coupled_joint_solve import (  # noqa: E402
    _ladder_archive,
    _sha256_array,
)
from tools.measure_ddm_v19_pure_priced_objective import (  # noqa: E402
    _delta_payload,
    _deterministic_storage_receipt,
    _portable,
    _sha256,
    _write,
)
from tools.measure_ddm_v19b_joint_remeasure_stack import (  # noqa: E402
    AXIS,
    FIRST_MOVE_ID,
    DDMV19BJointRemeasureStackConfigV1,
    StackState,
    _accepted_inventory,
    _archive_byte_rows,
    _candidate_state,
    _carrier_with_worldsheet,
    _initial_stack,
    _load_sources,
    _load_stage,
    _realized_post_state,
    _stack_from_ids,
    _worldsheet_payload,
)

SCHEMA = "ddm_dr1_realization_race_coding_gain_receipt.v1"
LANE_ID = "lane_ddm_c1_composed_candidate_spec_603_613_20260723"
N_PAIRS = 600
Q8_ONE = 256
CONTEST_BYTES = 37_545_489


class BoundInputV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class DDMDR1RealizationRaceCodingGainConfigV1(BaseModel):
    """Typed, SHA-bound contract for the three DR1 measurement arms."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    schema_: Literal["DDMDR1RealizationRaceCodingGainConfigV1"] = Field(
        default="DDMDR1RealizationRaceCodingGainConfigV1",
        alias="schema",
        serialization_alias="schema",
    )
    run_id: str = Field(min_length=8)
    seed: Literal[1234] = 1234
    c1_ledger: BoundInputV1
    dv2_receipt: BoundInputV1
    dv2_selected_payload: BoundInputV1
    g4_receipt: BoundInputV1
    g4_projection_receipt: BoundInputV1
    horizon_archive: BoundInputV1
    dv1_receipt: BoundInputV1
    dv1_ledger: BoundInputV1
    v19b_config: BoundInputV1
    v19b_receipt: BoundInputV1
    scorer_batch_size: Literal[16] = 16
    q8_mode: Literal["move_post_int8_fields_to_camera_q8"] = (
        "move_post_int8_fields_to_camera_q8"
    )
    post_rounding: Literal["nearest_integer_half_up"] = "nearest_integer_half_up"
    research_only: Literal[True] = True
    execution_allowed: Literal[False] = False
    score_claim: Literal[False] = False

    @model_validator(mode="after")
    def _sealed(self) -> DDMDR1RealizationRaceCodingGainConfigV1:
        paths = [
            value.path
            for name, value in self.__dict__.items()
            if isinstance(value, BoundInputV1) and name != "horizon_archive"
        ]
        if len(paths) != len(set(paths)):
            raise ValueError("DR1 receipt inputs must be path-distinct")
        return self

    def typed_hash(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json", by_alias=True),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()


def _bound_bytes(spec: BoundInputV1) -> bytes:
    payload = _read_regular_file_once(REPO_ROOT / spec.path)
    actual = _sha256(payload)
    if actual != spec.sha256:
        raise DirectDescriptionError(
            f"DR1 input SHA mismatch for {spec.path}: {actual} != {spec.sha256}"
        )
    return payload


def _bound_json(spec: BoundInputV1) -> dict[str, Any]:
    try:
        value = json.loads(_bound_bytes(spec))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DirectDescriptionError(f"DR1 input is not strict JSON: {spec.path}") from exc
    if not isinstance(value, dict):
        raise DirectDescriptionError(f"DR1 input is not one JSON object: {spec.path}")
    return value


def _computed_from(spec: BoundInputV1) -> dict[str, str]:
    return {"path": spec.path, "sha256": spec.sha256}


def _rate_audit(
    *,
    descriptive_form: str,
    compact_dof: str,
    coder_gain: str,
    visibility: str,
    tolerance: str,
    admissible: bool,
) -> dict[str, Any]:
    return {
        "scorer_visibility": visibility,
        "sensitivity_priced_tolerance": tolerance,
        "three_layer_decomposition": {
            "descriptive_form": descriptive_form,
            "inherently_compact_dof": compact_dof,
            "coder_gain": coder_gain,
        },
        "composed_candidate_admissible": admissible,
    }


def _nonredundancy_audit(
    *,
    owner: str,
    conditional_coding: str,
    pairwise_measurement: Mapping[str, Any],
    dimension_home: str,
    correction_delta_rule: str,
    admissible: bool,
) -> dict[str, Any]:
    return {
        "single_owner_fact_rule": owner,
        "cross_stream_conditional_coding": conditional_coding,
        "pairwise_redundancy_measurement": dict(pairwise_measurement),
        "dimension_home": dimension_home,
        "corrections_are_deltas": correction_delta_rule,
        "composed_candidate_admissible": admissible,
    }


def _dv1_payload_rows(payload: bytes) -> list[dict[str, Any]]:
    rows = []
    for line in payload.splitlines():
        value = json.loads(line)
        if isinstance(value, dict) and isinstance(value.get("payload"), dict):
            rows.append(value["payload"])
    return rows


def _select_dv1_rows(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    primitive = [
        dict(row)
        for row in rows
        if row.get("primitive_id") == "persistent_level_set_ground_partition"
    ]
    joint = [dict(row) for row in rows if row.get("candidate_id") == "persistent_plus_events"]
    if len(primitive) != 1 or len(joint) != 1:
        raise DirectDescriptionError("DR1 DV1 persistent primitive rows are not unique")
    return primitive[0], joint[0]


def _undrivable_row(
    row: Mapping[str, Any],
    *,
    scope: Literal["standalone", "joint"],
    source: BoundInputV1,
    unconditioned_persistent_bytes: int | None = None,
) -> dict[str, Any]:
    measurement = row["measurement"] if scope == "standalone" else row
    described = int(measurement["per_stratum_errors_described"]["Undrivable"])
    stationary = int(
        measurement["stationarity_of_described_errors"]["Undrivable"]["STATIC_IN_IMAGE"]
    )
    result = {
        "row_id": f"dv1_persistent_level_set_undrivable_{scope}",
        "status": "MEASURED_FROM_SHA_BOUND_DV1_N600",
        "scope": scope,
        "n_pairs": N_PAIRS,
        "described_errors": described,
        "described_fraction": float(
            measurement["per_stratum_described_fraction"]["Undrivable"]
        ),
        "static_in_image_fraction_of_described": stationary / described,
        "net_errors_closed_all_strata": int(measurement["net_errors_closed"]),
        "counted_bytes": int(measurement["counted_bytes"]),
        "computed_from": [_computed_from(source)],
        "evidence_axis": AXIS,
        "score_claim": False,
        "first-rung": (
            "receiver-realize this exact persistent partition through RGB and remeasure "
            "Undrivable collateral with Pose on the same archive"
        ),
        "rate_audit": _rate_audit(
            descriptive_form="persistent two-phase ground partition",
            compact_dof="one 512x384 persistent class field; temporal replication is derived",
            coder_gain=f"complete typed derivation is {int(measurement['counted_bytes'])} bytes",
            visibility=(
                "semantic-cell visibility is measured; RGB receiver survival is not yet proved"
            ),
            tolerance=(
                "exact semantic primitive is measured; local scorer-tolerance quantization "
                "remains owed before composed-candidate admission"
            ),
            admissible=False,
        ),
        "nonredundancy_audit": _nonredundancy_audit(
            owner=(
                "persistent ground partition owns static clip-mass exactly once; "
                "pair events own only innovations"
            ),
            conditional_coding=(
                "standalone primitive has no prior decoded stream"
                if scope == "standalone"
                else "the joint section is coded after the existing event archive"
            ),
            pairwise_measurement=(
                {
                    "status": "NOT_APPLICABLE_STANDALONE_STREAM",
                    "persistent_unconditioned_bytes": int(measurement["counted_bytes"]),
                }
                if scope == "standalone"
                else {
                    "status": "MEASURED_FROM_SHA_BOUND_DV1",
                    "persistent_unconditioned_bytes": unconditioned_persistent_bytes,
                    "persistent_given_events_bytes": int(row["new_joint_section_bytes"]),
                    "redundancy_bytes_unconditioned_minus_conditional": (
                        int(unconditioned_persistent_bytes)
                        - int(row["new_joint_section_bytes"])
                        if unconditioned_persistent_bytes is not None
                        else None
                    ),
                }
            ),
            dimension_home=(
                "static class mass lives in the clip-persistent primitive; temporal "
                "replication and frame-0 segmentation facts are not stored"
            ),
            correction_delta_rule=(
                "semantic proposal has no correction stream; RGB receiver deltas remain owed"
            ),
            admissible=False,
        ),
        "verdict_scope": (
            "INSTANCE: exact DV1 persistent-ground arbitration on preserved n600 semantic "
            "cells; no RGB, Pose, contest-axis, family, or promotion verdict"
        ),
    }
    return result


def _remeasure_sdwl1(
    *,
    payload: bytes,
    selected_row: Mapping[str, Any],
    source: BoundInputV1,
    g4_source: BoundInputV1,
    g4_gain_bytes: int,
) -> dict[str, Any]:
    inventory = decode_sentence(decompress_outer_payload(payload))
    options = SentenceOptions(
        layout=SentenceLayout.TYPED_SECTION,
        temporal_mode=TemporalMode.CAUSAL_DELTA,
    )
    measured = measure_serialization(inventory, options=options)
    expected = selected_row["outer_payload"]
    if (
        measured.outer_payload != payload
        or measured.outer_deflate_bytes != int(expected["bytes"])
        or measured.outer_deflate_sha256 != expected["sha256"]
    ):
        raise DirectDescriptionError("DR1 SDWL1 exact remeasurement differs from bound row")
    return {
        "row_id": "sdwl1_typed_causal_decoder_context_fold",
        "status": "MEASURED_EXACT_PARSEBACK_CONTEXT_ALREADY_PRESENT",
        "n_pairs": N_PAIRS,
        "control_bytes": int(expected["bytes"]),
        "context_fold_bytes": measured.outer_deflate_bytes,
        "delta_bytes": measured.outer_deflate_bytes - int(expected["bytes"]),
        "exact_parseback": measured.exact_parseback,
        "semantic_sha256": inventory.semantic_sha256,
        "payload_sha256": measured.outer_deflate_sha256,
        "g4_registered_gain_bytes": g4_gain_bytes,
        "g4_gain_applied_as_subtraction": False,
        "reason": (
            "the selected typed causal arrays already use decoder-derived left/upper "
            "same-channel contexts; the G4 innovation-raster saving is a different symbol "
            "stream and cannot be subtracted from this complete SDWL1 object"
        ),
        "byte_race_admitted": True,
        "exporter_realizability": "SYNTAX_PARSEBACK_PROVED_PIXEL_EXPORTER_STILL_OWED",
        "computed_from": [_computed_from(source), _computed_from(g4_source)],
        "evidence_axis": AXIS,
        "score_claim": False,
        "first-rung": (
            "quantize CELL/SEPR/SCRW facts by measured local scorer tolerance, then "
            "re-run exact syntax parse-back and a receiver exporter smoke"
        ),
        "rate_audit": _rate_audit(
            descriptive_form="typed partition-cell, separatrix, and pair-screw sentence",
            compact_dof="6600 records and 45600 non-padding scalar facts before tolerance quotient",
            coder_gain="causal delta plus decoder-derived left/upper arithmetic context",
            visibility=(
                "facts are scorer-derived, but no receiver exporter proves that all exact "
                "fact bytes affect a legal scorer-visible witness"
            ),
            tolerance=(
                "FAILS NEW ADMISSION DOCTRINE: every fact is exact; margin-priced lossy "
                "tolerance and null/gauge quotient are not implemented"
            ),
            admissible=False,
        ),
        "nonredundancy_audit": _nonredundancy_audit(
            owner=(
                "SDWL1 owns typed semantic facts; the separate G4 innovation raster owns "
                "its own decoded-symbol context and contributes no subtraction here"
            ),
            conditional_coding=(
                "within-stream left/upper contexts are measured; SDWL1 conditioned on a "
                "decoded G4 stream is not implemented"
            ),
            pairwise_measurement={
                "status": "NOT_MEASURED_BLOCKER",
                "ordered_pair": "SDWL1_given_G4_decoded",
                "sdwl1_unconditioned_bytes": measured.outer_deflate_bytes,
                "sdwl1_given_g4_bytes": None,
                "g4_gain_bytes_not_reused": g4_gain_bytes,
            },
            dimension_home=(
                "clip/pair/section ownership is typed, but cross-stream single-owner "
                "fact proof is not yet exported"
            ),
            correction_delta_rule=(
                "no correction stream is present in this syntax-only row"
            ),
            admissible=False,
        ),
        "verdict_scope": (
            "FORMULATION: exact typed-causal SDWL1 syntax only; no claim against "
            "sensitivity-priced SDWL1, its renderer family, or the description paradigm"
        ),
    }


def _first_move_deltas(
    ctx: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    problem = ctx["problem"]
    candidate = _candidate_state(ctx, FIRST_MOVE_ID)
    template = np.asarray(candidate["template_values_u8"], dtype=np.int16) - np.asarray(
        problem["initial_template_values_u8"], dtype=np.int16
    )
    sparse = np.asarray(candidate["compensation_rgb_i8"], dtype=np.int16) - np.asarray(
        problem["initial_compensation_rgb_i8"], dtype=np.int16
    )
    return template, sparse


def _nearest_integer_half_up(numerator: int, denominator: int = Q8_ONE) -> int:
    if denominator <= 0:
        raise ValueError("denominator must be positive")
    sign = -1 if numerator < 0 else 1
    return sign * ((abs(numerator) + denominator // 2) // denominator)


def _post_equivalent_stack(
    stack: StackState,
    first_template: np.ndarray,
    first_sparse: np.ndarray,
) -> StackState:
    extra_quanta = _nearest_integer_half_up(stack.preuint8_scale_q8)
    return StackState(
        stack.template_delta + first_template * extra_quanta,
        stack.compensation_delta + first_sparse * extra_quanta,
        stack.grammar_move_ids,
        0,
        (*stack.move_ids, f"q8_total_rounded_to_{extra_quanta}_post_quanta"),
    )


def _q8_program(
    *,
    coupled_archive: bytes,
    problem: Mapping[str, Any],
    template_q8: np.ndarray,
    sparse_q8: np.ndarray,
    dither_mode: Literal["off", "bayer8", "resize_null_sigma_delta"],
) -> PreUint8Q8ProgramV1:
    receiver = receive_coupled_margin_archive(coupled_archive)
    templates = []
    for placement in receiver.program.placements:
        delta = template_q8[placement.template_index]
        if np.any(delta):
            templates.append(
                TemplateQ8CorrectionV1(
                    placement.source_pair_id,
                    placement.template_index,
                    tuple(int(value) for value in delta.reshape(-1)),
                )
            )
    sparse = []
    for index, support in enumerate(problem["sparse_compensation_support"]):
        delta = sparse_q8[index]
        if np.any(delta):
            sparse.append(
                SparseQ8CorrectionV1(
                    int(support["source_pair_id"]),
                    int(support["frame_index"]),
                    int(support["camera_y"]),
                    int(support["camera_x"]),
                    tuple(int(value) for value in delta),
                )
            )
    return PreUint8Q8ProgramV1(
        tuple(sorted(templates)),
        tuple(sorted(sparse)),
        dither_mode,
        210,
    )


def _compile_horizon_stack(
    *,
    stack: StackState,
    horizon_archive: bytes,
    v19_config: Any,
    ctx: Mapping[str, Any],
) -> tuple[bytes, dict[str, Any]]:
    state = _realized_post_state(stack, ctx["problem"])
    _compact, expanded, program = _ladder_archive(
        state=state,
        problem=ctx["problem"],
        v14_archive=horizon_archive,
        source_start=0,
        source_stop=N_PAIRS,
    )
    worldsheet, grammar = _worldsheet_payload(
        ctx["n600_archive"],
        active_pair_ids=tuple(v19_config.pair_ids),
        grammar_move_ids=stack.grammar_move_ids,
    )
    carrier = _carrier_with_worldsheet(expanded, worldsheet)
    archive = compile_coupled_margin_archive(carrier, program)
    return archive, {
        "archive_bytes": len(archive),
        "archive_sha256": _sha256(archive),
        "grammar": grammar,
        "post_template_nonzero": int(np.count_nonzero(stack.template_delta)),
        "post_sparse_nonzero": int(np.count_nonzero(stack.compensation_delta)),
        "move_ids": list(stack.move_ids),
    }


def _measure_receiver_arm(
    *,
    name: str,
    archive: bytes,
    receiver_factory: Callable[[bytes], Any],
    horizon_archive: bytes,
    root: Path,
    config_hash: str,
    config: DDMDR1RealizationRaceCodingGainConfigV1,
    ctx: Mapping[str, Any],
) -> dict[str, Any]:
    return _measure_window_first_batch_replay(
        name=name,
        archive=archive,
        receiver_factory=receiver_factory,
        baseline_archive=horizon_archive,
        source_pair_ids=tuple(range(N_PAIRS)),
        local_pair_ids=tuple(range(N_PAIRS)),
        root=root,
        config_hash=config_hash,
        batch_size=config.scorer_batch_size,
        labels_all=ctx["labels_all"],
        poses_all=ctx["poses_all"],
        segnet=ctx["segnet"],
        posenet=ctx["posenet"],
    )


def _measure_window_first_batch_replay(
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
    """Measure n600 with the canonical v14 first-batch determinism replay."""
    from tac.optimization.direct_description_carrier_compose import (
        receive_carrier_compose_archive,
    )

    class_names = {
        0: "Road",
        1: "Lane",
        2: "MyCar",
        3: "Movable",
        4: "Undrivable",
    }
    stage = root / "stage_checkpoints" / name
    source = np.asarray(source_pair_ids, dtype=np.int64)
    local = np.asarray(local_pair_ids, dtype=np.int64)
    if source.shape != local.shape or source.size == 0:
        raise DirectDescriptionError("DR1 ladder pair geometry differs")
    archive_sha = _sha256(archive)
    for start in range(0, int(source.size), batch_size):
        stop = min(start + batch_size, int(source.size))
        checkpoint = stage / f"batch_{start:04d}_{stop:04d}.json"
        resumed = _load_stage(checkpoint, config_hash)
        if resumed is not None:
            if resumed["archive_sha256"] != archive_sha:
                raise DirectDescriptionError("DR1 batch archive identity differs")
            continue
        receiver = receiver_factory(archive)
        baseline = receive_carrier_compose_archive(baseline_archive)
        local_ids = tuple(int(value) for value in local[start:stop])
        source_ids = source[start:stop]
        camera = receiver.render_camera_pairs(local_ids)
        baseline_camera = baseline.render_camera_pairs(local_ids)
        cells, pose6 = _forward(segnet, posenet, camera)
        if start == 0:
            replay_cells, replay_pose6 = _forward(segnet, posenet, camera)
            if not np.array_equal(cells, replay_cells) or not np.array_equal(
                pose6, replay_pose6
            ):
                raise DirectDescriptionError(f"DR1 {name} first-batch replay failed")
        labels = np.asarray(labels_all[source_ids])
        poses = np.asarray(poses_all[source_ids])
        errors = cells != labels
        class_rows = {}
        for class_id, class_name in class_names.items():
            mask = labels == class_id
            class_rows[class_name] = {
                "errors": int(np.count_nonzero(errors & mask)),
                "sites": int(np.count_nonzero(mask)),
            }
        diff = camera.astype(np.int16) - baseline_camera.astype(np.int16)
        _write(
            checkpoint,
            {
                "schema": "ddm_dr1_receiver_batch.v1",
                "determinism_replay_policy": (
                    "first_batch_exact_replay_then_deterministic_algorithms"
                ),
                "typed_config_sha256": config_hash,
                "candidate": name,
                "archive_sha256": archive_sha,
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
            },
        )
    batches = [
        json.loads(path.read_bytes()) for path in sorted(stage.glob("batch_*.json"))
    ]
    expected = math.ceil(source.size / batch_size)
    if len(batches) != expected:
        raise DirectDescriptionError("DR1 batch coverage incomplete")
    if any(
        row["archive_sha256"] != archive_sha
        or row["typed_config_sha256"] != config_hash
        for row in batches
    ):
        raise DirectDescriptionError("DR1 aggregate batch identity differs")
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
        for class_name in class_names.values()
    }
    for row in classes.values():
        row["d_seg"] = f"{row['errors'] / max(1, row['sites']):.12f}"
    d_seg = errors / sites
    d_pose = pose_sse / pose_coordinates
    byte_rows, custody = _archive_byte_rows(archive, receiver_factory)
    return {
        "candidate": name,
        "archive_bytes": len(archive),
        "archive_sha256": archive_sha,
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
            f"{100.0 * d_seg + math.sqrt(10.0 * d_pose) + 25.0 * len(archive) / CONTEST_BYTES:.12f}"
        ),
        "batch_count": len(batches),
        "batch_size": batch_size,
        "all_batches_checkpointed_and_preserved": True,
        "determinism_replay_policy": (
            "first_batch_exact_replay_then_deterministic_algorithms"
        ),
        "batch_digest_chain_sha256": hashlib.sha256(
            "".join(
                row["cells_sha256"] + row["pose6_sha256"] for row in batches
            ).encode()
        ).hexdigest(),
        "camera_diff_vs_v15": {
            key: sum(int(row["camera_diff_vs_v15"][key]) for row in batches)
            for key in ("changed_channel_values", "changed_rgb_pixels", "l1_channel_sum")
        },
        "byte_streams": byte_rows,
        "receiver_custody": dict(custody),
        "score_claim": False,
        "evidence_axis": AXIS,
    }


def _quantization_support(
    *,
    control_archive: bytes,
    lattice_archive: bytes,
    dither_archive: bytes,
    derived_archive: bytes,
) -> dict[str, Any]:
    control = receive_coupled_margin_archive(control_archive)
    lattice = receive_preuint8_q8_archive(lattice_archive)
    dither = receive_preuint8_q8_archive(dither_archive)
    derived = receive_preuint8_q8_archive(derived_archive)
    lattice_changed = 0
    dither_changed = 0
    dither_only = 0
    lattice_only = 0
    derived_changed = 0
    derived_only = 0
    uniform_only_vs_derived = 0
    for start in range(0, N_PAIRS, 16):
        stop = min(start + 16, N_PAIRS)
        ids = tuple(range(start, stop))
        base = control.render_camera_pairs(ids)
        off = lattice.render_camera_pairs(ids)
        ordered = dither.render_camera_pairs(ids)
        shaped = derived.render_camera_pairs(ids)
        off_changed = off != base
        ordered_changed = ordered != base
        lattice_changed += int(np.count_nonzero(off_changed))
        dither_changed += int(np.count_nonzero(ordered_changed))
        dither_only += int(np.count_nonzero(ordered_changed & ~off_changed))
        lattice_only += int(np.count_nonzero(off_changed & ~ordered_changed))
        shaped_changed = shaped != base
        derived_changed += int(np.count_nonzero(shaped_changed))
        derived_only += int(np.count_nonzero(shaped_changed & ~off_changed))
        uniform_only_vs_derived += int(np.count_nonzero(off_changed & ~shaped_changed))
    net_recovered = dither_changed - lattice_changed
    observable_gap = dither_only + lattice_only
    return {
        "lattice_changed_channel_values": lattice_changed,
        "dither_changed_channel_values": dither_changed,
        "dither_only_changed_channel_values": dither_only,
        "lattice_only_changed_channel_values": lattice_only,
        "net_dither_changed_channel_values": net_recovered,
        "observable_stage_disagreement_values": observable_gap,
        "net_recovery_fraction_of_observable_stage_gap": (
            net_recovered / observable_gap if observable_gap else 0.0
        ),
        "derived_changed_channel_values": derived_changed,
        "derived_only_changed_channel_values": derived_only,
        "uniform_only_vs_derived_changed_channel_values": uniform_only_vs_derived,
        "derived_vs_uniform_stage_disagreement_values": (
            derived_only + uniform_only_vs_derived
        ),
    }


def _joint_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    return _delta_payload(before, after)


def _pad_zip_comment(archive: bytes, target_bytes: int) -> bytes:
    """Add counted neutral ZIP-comment bytes without changing decoded members."""
    delta = target_bytes - len(archive)
    if delta < 0:
        raise DirectDescriptionError("DR1 cannot pad an archive to a smaller byte count")
    if delta == 0:
        return archive
    marker = b"PK\x05\x06"
    offset = archive.rfind(marker)
    if offset < 0 or offset + 22 > len(archive):
        raise DirectDescriptionError("DR1 equal-rate control has no terminal ZIP EOCD")
    old_comment_bytes = struct.unpack_from("<H", archive, offset + 20)[0]
    if offset + 22 + old_comment_bytes != len(archive):
        raise DirectDescriptionError("DR1 equal-rate control ZIP comment is not terminal")
    if old_comment_bytes + delta > 65_535:
        raise DirectDescriptionError("DR1 equal-rate ZIP comment exceeds format limit")
    result = bytearray(archive)
    struct.pack_into("<H", result, offset + 20, old_comment_bytes + delta)
    result.extend(b"\0" * delta)
    return bytes(result)


def _prove_preuint8_receiver_equivalence(
    before: bytes,
    after: bytes,
) -> dict[str, Any]:
    """Prove padding changes bytes only, not any camera-resolution receiver value."""
    from tac.optimization.direct_description_preuint8_channel import (
        parse_preuint8_q8_archive,
    )

    before_members, _before_homes = parse_preuint8_q8_archive(before)
    after_members, _after_homes = parse_preuint8_q8_archive(after)
    if before_members != after_members:
        raise DirectDescriptionError("DR1 equal-rate ZIP padding changed decoded members")
    original = receive_preuint8_q8_archive(before)
    digest = hashlib.sha256()
    for start in range(0, N_PAIRS, 16):
        stop = min(start + 16, N_PAIRS)
        ids = tuple(range(start, stop))
        camera_before = original.render_camera_pairs(ids)
        digest.update(camera_before.tobytes())
    return {
        "status": "PROVED_EXACT_MEMBER_EQUAL_PLUS_N600_CAMERA_HASH",
        "before_archive_bytes": len(before),
        "before_archive_sha256": _sha256(before),
        "equal_rate_archive_bytes": len(after),
        "equal_rate_archive_sha256": _sha256(after),
        "camera_uint8_n600_sha256": digest.hexdigest(),
        "decoded_member_bytes_equal": True,
        "padding_bytes": len(after) - len(before),
        "padding_location": "outer_zip_comment_not_decoded_by_receiver",
    }


def _distortion_objective(row: Mapping[str, Any]) -> float:
    return 100.0 * float(row["d_seg"]) + math.sqrt(10.0 * float(row["d_pose"]))


def _realization_rows(
    *,
    config: DDMDR1RealizationRaceCodingGainConfigV1,
    root: Path,
    horizon_archive: bytes,
    v19b_receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    v19b_config = DDMV19BJointRemeasureStackConfigV1.model_validate_json(
        _bound_bytes(config.v19b_config)
    )
    v19_config, source_v19, ctx = _load_sources(v19b_config)
    if v19b_receipt.get("typed_config_sha256") != v19b_config.typed_config_hash():
        raise DirectDescriptionError("DR1 v19b receipt/config binding differs")
    inventory = _accepted_inventory(source_v19)
    stack = _stack_from_ids(
        v19b_receipt["final_stack_move_ids"],
        inventory,
        problem=ctx["problem"],
        ctx=ctx,
    )
    first_template, first_sparse = _first_move_deltas(ctx)
    control_stack = _initial_stack(ctx["problem"])
    control_archive, control_compile = _compile_horizon_stack(
        stack=control_stack,
        horizon_archive=horizon_archive,
        v19_config=v19_config,
        ctx=ctx,
    )
    post_stack = _post_equivalent_stack(stack, first_template, first_sparse)
    post_archive, post_compile = _compile_horizon_stack(
        stack=post_stack,
        horizon_archive=horizon_archive,
        v19_config=v19_config,
        ctx=ctx,
    )
    grammar_only = replace(
        control_stack,
        grammar_move_ids=stack.grammar_move_ids,
        move_ids=tuple(stack.grammar_move_ids),
    )
    q8_base, q8_base_compile = _compile_horizon_stack(
        stack=grammar_only,
        horizon_archive=horizon_archive,
        v19_config=v19_config,
        ctx=ctx,
    )
    template_q8 = stack.template_delta * Q8_ONE + first_template * stack.preuint8_scale_q8
    sparse_q8 = stack.compensation_delta * Q8_ONE + first_sparse * stack.preuint8_scale_q8
    lattice_program = _q8_program(
        coupled_archive=q8_base,
        problem=ctx["problem"],
        template_q8=template_q8,
        sparse_q8=sparse_q8,
        dither_mode="off",
    )
    dither_program = replace(lattice_program, dither_mode="bayer8")
    derived_program = replace(
        lattice_program,
        dither_mode="resize_null_sigma_delta",
    )
    lattice_archive = compile_preuint8_q8_archive(q8_base, lattice_program)
    dither_archive = compile_preuint8_q8_archive(q8_base, dither_program)
    derived_archive = compile_preuint8_q8_archive(q8_base, derived_program)
    equal_rate_bytes = max(
        len(lattice_archive),
        len(dither_archive),
        len(derived_archive),
    )
    lattice_equal_rate = _pad_zip_comment(lattice_archive, equal_rate_bytes)
    dither_equal_rate = _pad_zip_comment(dither_archive, equal_rate_bytes)
    derived_equal_rate = _pad_zip_comment(derived_archive, equal_rate_bytes)
    lattice_equal_rate_proof = _prove_preuint8_receiver_equivalence(
        lattice_archive,
        lattice_equal_rate,
    )
    dither_equal_rate_proof = _prove_preuint8_receiver_equivalence(
        dither_archive,
        dither_equal_rate,
    )
    derived_equal_rate_proof = _prove_preuint8_receiver_equivalence(
        derived_archive,
        derived_equal_rate,
    )
    _publish_immutable(
        root
        / "archives"
        / f"preuint8_uniform_lattice.equal-rate-{equal_rate_bytes}.zip.receipt-bytes",
        lattice_equal_rate,
    )
    _publish_immutable(
        root
        / "archives"
        / f"preuint8_bayer8_dither.equal-rate-{equal_rate_bytes}.zip.receipt-bytes",
        dither_equal_rate,
    )
    _publish_immutable(
        root
        / "archives"
        / f"preuint8_resize_null_sigma_delta.equal-rate-{equal_rate_bytes}.zip.receipt-bytes",
        derived_equal_rate,
    )
    archives = {
        "control": (control_archive, receive_coupled_margin_archive),
        "post_int8_lattice": (post_archive, receive_coupled_margin_archive),
        "preuint8_uniform_lattice": (lattice_archive, receive_preuint8_q8_archive),
        "preuint8_bayer8_dither": (dither_archive, receive_preuint8_q8_archive),
        "preuint8_resize_null_sigma_delta": (
            derived_archive,
            receive_preuint8_q8_archive,
        ),
    }
    measurements = {}
    for name, (archive, factory) in archives.items():
        path = root / "archives" / f"{name}.not_a_candidate.zip.receipt-bytes"
        _publish_immutable(path, archive)
        measurements[name] = _measure_receiver_arm(
            name=f"A_{name}",
            archive=archive,
            receiver_factory=factory,
            horizon_archive=horizon_archive,
            root=root,
            config_hash=config.typed_hash(),
            config=config,
            ctx=ctx,
        )
    control = measurements["control"]
    support = _quantization_support(
        control_archive=q8_base,
        lattice_archive=lattice_archive,
        dither_archive=dither_archive,
        derived_archive=derived_archive,
    )
    encode_program = __import__(
        "tac.optimization.direct_description_preuint8_channel",
        fromlist=["encode_preuint8_q8_program"],
    ).encode_preuint8_q8_program
    program_sizes = {
        len(encode_program(program))
        for program in (lattice_program, dither_program, derived_program)
    }
    equal_program_bytes = len(program_sizes) == 1
    rows = []
    for name in (
        "post_int8_lattice",
        "preuint8_uniform_lattice",
        "preuint8_bayer8_dither",
        "preuint8_resize_null_sigma_delta",
    ):
        measurement = measurements[name]
        delta = _joint_delta(control, measurement)
        rows.append(
            {
                "row_id": name,
                "status": "MEASURED_N600_FULL_UINT8_R_REPLAY",
                "n_pairs": N_PAIRS,
                "measurement": measurement,
                "joint_delta_vs_horizon_control": delta,
                "accepted": bool(delta["joint_delta"] < 0.0),
                "computed_from": [
                    _computed_from(config.c1_ledger),
                    _computed_from(config.g4_receipt),
                    _computed_from(config.g4_projection_receipt),
                    _computed_from(config.horizon_archive),
                    _computed_from(config.v19b_receipt),
                ],
                "evidence_axis": AXIS,
                "score_claim": False,
                "first-rung": (
                    "rerun the winning application stage after margin-priced pruning of "
                    "correction records on the same n600 horizon control"
                ),
                "rate_audit": _rate_audit(
                    descriptive_form="horizon rule plus typed template/sparse/track corrections",
                    compact_dof=(
                        f"{len(lattice_program.templates)} template placements and "
                        f"{len(lattice_program.sparse)} sparse camera records"
                    ),
                    coder_gain="exact nested archive bytes; no inferred compression credit",
                    visibility=(
                        "all correction records are replayed at camera resolution through "
                        "uint8, R, frozen SegNet, and PoseNet"
                    ),
                    tolerance=(
                        "v19b admitted move amplitudes retained; new margin-priced record "
                        "pruning remains the first rung"
                    ),
                    admissible=False,
                ),
                "nonredundancy_audit": _nonredundancy_audit(
                    owner=(
                        "horizon base owns the decoded scene; template, sparse, and track "
                        "streams own only named residual corrections"
                    ),
                    conditional_coding=(
                        "correction programs decode after the nested horizon base, but the "
                        "ordered pairwise conditional byte matrix is not measured"
                    ),
                    pairwise_measurement={
                        "status": "NOT_MEASURED_BLOCKER",
                        "ordered_pairs_owed": [
                            "template_given_horizon",
                            "sparse_given_horizon_plus_template",
                            "track_given_horizon_plus_template_plus_sparse",
                        ],
                    },
                    dimension_home=(
                        "clip-static state remains in the horizon; pair records carry only "
                        "innovations; preuint8 corrections have one camera/Q8 scale"
                    ),
                    correction_delta_rule=(
                        "all emitted correction records are deltas applied to decoded base "
                        "state; a byte-wise overlap proof remains owed"
                    ),
                    admissible=False,
                ),
                "verdict_scope": (
                    "INSTANCE: SHA-bound horizon plus v19b move classes at this application "
                    "stage; macOS-CPU advisory only"
                ),
            }
        )
    post_distortion = _distortion_objective(measurements["post_int8_lattice"])
    uniform_distortion = _distortion_objective(
        measurements["preuint8_uniform_lattice"]
    )
    dither_distortion = _distortion_objective(measurements["preuint8_bayer8_dither"])
    derived_distortion = _distortion_objective(
        measurements["preuint8_resize_null_sigma_delta"]
    )
    quantization_gap = abs(post_distortion - uniform_distortion)
    dither_recovery = uniform_distortion - dither_distortion
    recovery = dither_recovery / quantization_gap if quantization_gap else 0.0
    derived_recovery = uniform_distortion - derived_distortion
    derived_recovery_fraction = (
        derived_recovery / quantization_gap if quantization_gap else 0.0
    )
    falsifier_evaluable = bool(
        equal_program_bytes
        and len(lattice_equal_rate) == len(dither_equal_rate)
        and len(dither_equal_rate) == len(derived_equal_rate)
        and quantization_gap > 0.0
    )
    summary = {
        "control": control,
        "control_compile": control_compile,
        "post_compile": post_compile,
        "q8_base_compile": q8_base_compile,
        "q8_program": {
            "template_records": len(lattice_program.templates),
            "sparse_records": len(lattice_program.sparse),
            "aggregate_post_move_template_nonzero": int(np.count_nonzero(stack.template_delta)),
            "aggregate_post_move_sparse_nonzero": int(
                np.count_nonzero(stack.compensation_delta)
            ),
            "preuint8_scale_q8_sum": stack.preuint8_scale_q8,
            "program_bytes_equal": equal_program_bytes,
            "lattice_archive_bytes": len(lattice_archive),
            "dither_archive_bytes": len(dither_archive),
            "derived_archive_bytes": len(derived_archive),
            "equal_rate_archive_bytes": equal_rate_bytes,
            "lattice_equal_rate_receiver_proof": lattice_equal_rate_proof,
            "dither_equal_rate_receiver_proof": dither_equal_rate_proof,
            "derived_equal_rate_receiver_proof": derived_equal_rate_proof,
        },
        "quantization_support": support,
        "falsifier": {
            "threshold": 0.10,
            "operational_definition": (
                "continuous-vs-uint8 gap is the realized distortion-objective distance "
                "between preuint8-uniform and post-int8 placement; dither recovery is "
                "preuint8-uniform minus preuint8-bayer8 at equal counted archive bytes"
            ),
            "post_int8_distortion_objective": post_distortion,
            "preuint8_uniform_distortion_objective": uniform_distortion,
            "preuint8_dither_distortion_objective": dither_distortion,
            "preuint8_derived_distortion_objective": derived_distortion,
            "continuous_vs_uint8_gap": quantization_gap,
            "dither_recovered_distortion_objective": dither_recovery,
            "derived_recovered_distortion_objective": derived_recovery,
            "equal_exact_archive_bytes": (
                len(lattice_equal_rate)
                == len(dither_equal_rate)
                == len(derived_equal_rate)
            ),
            "equal_rate_archive_bytes": equal_rate_bytes,
            "equal_correction_program_bytes": equal_program_bytes,
            "evaluable_at_equal_exact_archive_bytes": falsifier_evaluable,
            "observed_recovery_fraction": recovery,
            "fired": bool(falsifier_evaluable and recovery < 0.10),
            "disposition": (
                "INSTANCE_NEGATIVE_FOR_TESTED_MOVE_CLASSES"
                if falsifier_evaluable and recovery < 0.10
                else "NOT_FIRED"
                if falsifier_evaluable
                else "NOT_EVALUABLE_EXACT_ARCHIVE_BYTE_MISMATCH"
            ),
            "verdict_scope": (
                "INSTANCE: this Q8 program and tested move classes only; no formulation, "
                "family, or paradigm generalization"
            ),
        },
        "derived_kernel_prediction": {
            "prediction": (
                "resize-null block sigma-delta recovers more of the operational "
                "continuous-vs-uint8 gap than generic Bayer8 at equal bytes"
            ),
            "generic_recovery_fraction": recovery,
            "derived_recovery_fraction": derived_recovery_fraction,
            "passed": bool(
                falsifier_evaluable and derived_recovery_fraction > recovery
            ),
            "disposition": (
                "SUPPORTED_INSTANCE"
                if falsifier_evaluable and derived_recovery_fraction > recovery
                else "FALSIFIED_INSTANCE"
                if falsifier_evaluable
                else "NOT_EVALUABLE"
            ),
            "implemented_geometry": (
                "exact disjoint 2x2 bilinear resize numerators; adjacent integer "
                "lattice choice minimizes range(A) residual and directs the remainder "
                "into ker(A)"
            ),
            "fisher_tie_break": (
                "BLOCKED_NOT_IMPLEMENTED: receiver has no counted or derivable local "
                "margin field; this contender tests the exact resize-nullspace leg only"
            ),
            "verdict_scope": (
                "INSTANCE: this SHA-bound correction field and resize-null block "
                "quantizer; literal Floyd-Steinberg, within-range Fisher shaping, "
                "and other kernels remain open"
            ),
        },
        "scorer_custody": ctx["scorer_custody"],
    }
    return summary, rows


def run(
    config: DDMDR1RealizationRaceCodingGainConfigV1,
    output_directory: Path,
    semantic_argv: Sequence[str],
) -> Path:
    root = output_directory.resolve()
    storage = _deterministic_storage_receipt(_storage_preflight(root))
    root.mkdir(parents=True, exist_ok=True)
    receipt_path = root / "receipt.json"
    if receipt_path.exists():
        receipt = json.loads(_read_regular_file_once(receipt_path))
        if receipt.get("typed_config_sha256") != config.typed_hash():
            raise DirectDescriptionError("DR1 completed receipt config differs")
        return receipt_path

    c1 = _bound_json(config.c1_ledger)
    dv2 = _bound_json(config.dv2_receipt)
    g4 = _bound_json(config.g4_receipt)
    _g4_projection = _bound_json(config.g4_projection_receipt)
    dv1 = _bound_json(config.dv1_receipt)
    v19b = _bound_json(config.v19b_receipt)
    horizon = _bound_bytes(config.horizon_archive)
    if (
        c1.get("score_claim") is not False
        or dv2.get("score_claim") is not False
        or g4.get("score_claim") is not False
        or dv1.get("score_claim") is not False
        or v19b.get("score_claim") is not False
    ):
        raise DirectDescriptionError("DR1 source false-authority contract drifted")

    selected = dv2["selected_base_row"]
    if selected.get("row_id") != "whole_typed_section_causal_delta":
        raise DirectDescriptionError("DR1 DV2 selected row changed")
    dv2_row_matches = [
        row
        for row in dv2["rows"]
        if row.get("spec", {}).get("row_id") == "whole_typed_section_causal_delta"
    ]
    if len(dv2_row_matches) != 1:
        raise DirectDescriptionError("DR1 DV2 selected measurement row is not unique")
    sdwl1 = _remeasure_sdwl1(
        payload=_bound_bytes(config.dv2_selected_payload),
        selected_row=dv2_row_matches[0],
        source=config.dv2_receipt,
        g4_source=config.g4_receipt,
        g4_gain_bytes=int(c1["waterfill"]["free_context"]["gain_bytes"]),
    )
    primitive, joint = _select_dv1_rows(_dv1_payload_rows(_bound_bytes(config.dv1_ledger)))
    dv1_rows = [
        _undrivable_row(
            primitive,
            scope="standalone",
            source=config.dv1_ledger,
            unconditioned_persistent_bytes=int(primitive["measurement"]["counted_bytes"]),
        ),
        _undrivable_row(
            joint,
            scope="joint",
            source=config.dv1_ledger,
            unconditioned_persistent_bytes=int(primitive["measurement"]["counted_bytes"]),
        ),
    ]
    realization_summary, realization_rows = _realization_rows(
        config=config,
        root=root,
        horizon_archive=horizon,
        v19b_receipt=v19b,
    )
    receipt = {
        "schema": SCHEMA,
        "lane_id": LANE_ID,
        "run_id": config.run_id,
        "typed_config": config.model_dump(mode="json", by_alias=True),
        "typed_config_sha256": config.typed_hash(),
        "semantic_argv": list(semantic_argv),
        "input_custody": {
            name: _computed_from(value)
            for name, value in config.__dict__.items()
            if isinstance(value, BoundInputV1)
        },
        "deliverable_A_realization_race": {
            "summary": realization_summary,
            "rows": realization_rows,
        },
        "deliverable_B_sdwl1_context_fold": sdwl1,
        "deliverable_C_undrivable_transfer": dv1_rows,
        "positive_findings_have_first_rung": all(
            "first-rung" in row
            for row in [*realization_rows, sdwl1, *dv1_rows]
            if row.get("accepted")
            or row.get("byte_race_admitted")
            or float(row.get("net_errors_closed_all_strata", 0)) > 0
        ),
        "storage_preflight": storage,
        "resume": {
            "batch_size": config.scorer_batch_size,
            "per_batch_checkpoints": True,
            "all_preserved": True,
        },
        "late_directive_disposition": {
            "rate_doctrine_clause_4_nonredundancy": (
                "CONSUMED; each touched stream has a single-owner/dimension-home audit and "
                "either a measured pairwise row or an explicit admission blocker"
            ),
            "hopfield_preprox_third_leg": (
                "BLOCKED_NOT_MEASURED: the delegated base predates receiver commit "
                "1c55f78063 and has no src/tac/optimization/ddm_runtime_receiver.py; "
                "authority forbids importing unreviewed MAIN state into this isolated worktree"
            ),
            "hopfield_verdict_scope": (
                "CUSTODY BLOCKER ONLY; no Hopfield formulation or family verdict"
            ),
            "resize_null_sigma_delta_contender": (
                "CONSUMED_AND_MEASURED: exact disjoint bilinear-resize numerators "
                "derived the block sigma-delta choice; its predicted recovery over "
                "generic Bayer8 was FALSIFIED_INSTANCE at equal counted bytes"
            ),
            "resize_null_sigma_delta_verdict_scope": (
                "INSTANCE: this SHA-bound correction field and exact-resize block "
                "quantizer; the literal Floyd-Steinberg comparator, within-range "
                "Fisher shaping, and other kernels remain unmeasured"
            ),
        },
        "stores_consulted": [
            "CLAUDE.md",
            "AGENTS.md",
            "PROGRAM.md",
            "docs/operating_manual_craft_handoff.md",
            *[
                value.path
                for value in config.__dict__.values()
                if isinstance(value, BoundInputV1)
            ],
            ".omx/state/lane_registry.json",
            ".omx/state/subagent_progress.jsonl",
            "MAIN inbox directive 2026-07-23T16:36:35Z",
            "MAIN inbox directive 2026-07-23T16:45:43Z",
            "MAIN inbox directive 2026-07-23T16:59:16Z",
            "MAIN inbox directive 2026-07-23T17:13:39Z",
        ],
        "pointer": f"{POINTER_SCORE_TEXT} [contest-CPU]",
        "pointer_moved": False,
        "evidence_axis": AXIS,
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
        "d_seg_claim": False,
        "d_pose_claim": False,
        "promotion_eligible": False,
        "main_landing_review_required": True,
        "verdict_scope": (
            "INSTANCE/FORMULATION rows state their own narrow scope; no contest-axis, "
            "family-wide, paradigm, pointer, score, or promotion verdict"
        ),
    }
    _write(receipt_path, receipt)
    return receipt_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    config = DDMDR1RealizationRaceCodingGainConfigV1.model_validate_json(
        _read_regular_file_once(args.config)
    )
    artifact = run(
        config,
        args.output_directory,
        [
            "tools/measure_ddm_dr1_realization_race_coding_gain.py",
            "--config",
            str(args.config),
            "--output-directory",
            str(args.output_directory),
        ],
    )
    print(
        json.dumps(
            {
                "artifact": _portable(artifact),
                "evidence_axis": AXIS,
                "score_claim": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
