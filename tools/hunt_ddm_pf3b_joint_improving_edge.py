#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Rank and measure the 52 open support-positive PF3 coordinates.

The ranking pass replays the already-sealed MS6 event supports under their
batch-one geometry to determine the actual frozen-SegNet event-error direction.
It then uses non-event argmax changes and the n600 AT1x PoseNet input-Gram trace
as spill guards.  Authority measurements reuse PF3's real receiver, uint8,
batch-16 frozen scorer, and fixed-owner exact-parseback E4 path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO / "src", REPO):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap  # noqa: E402
from tac.optimization.ddm_pf3_finite_price_materialization import (  # noqa: E402
    RATE_DENOMINATOR_BYTES,
)
from tac.optimization.ddm_runtime_sensitivity import forward_seg_argmax  # noqa: E402
from tools.materialize_ddm_pf3_finite_prices import (  # noqa: E402
    PAIR_COUNT,
    DDMF3FinitePriceConfigV1,
    PF3MeasurementError,
    _artifact,
    _endpoint,
    _fast_receiver,
    _load_inventory,
    _load_models,
    _load_or_race_base_coders,
    _measure_candidate,
    _probe_context,
    _publish,
    _read_bound_json,
    _resolve,
    _sha256_file,
    _verify_scorer_config,
)
from tools.materialize_ddm_pf3_finite_prices import (  # noqa: E402
    _read_config as _read_pf3_config,
)

RUN_ID: Final = "ddm_pf3b_52probe_joint_improving_hunt_20260725T202800Z"
LANE_ID: Final = "ddm_pf3b_52probe_joint_improving_hunt"
DELEGATION_KEY: Final = (
    "codex_delegate:ddm_pf3b_52probe_joint_improving_hunt:20260725T202800Z"
)
CONFIG: Final = (
    REPO / ".omx/research/configs/ddm_pf3b_52probe_joint_improving_hunt_20260725.json"
)
OUTPUT: Final = REPO / ".omx/research" / RUN_ID
CHECKPOINT_SCHEMA: Final = "ddm_pf3b_joint_measurement_checkpoint.v1"
RECEIPT_SCHEMA: Final = "ddm_pf3b_joint_improving_hunt_receipt.v1"
RANK_SCHEMA: Final = "ddm_pf3b_prescore_rank_checkpoint.v2"
HEIGHT: Final = 384
WIDTH: Final = 512
_MAGNITUDE = re.compile(r"^(.*)\.mag([12])$")


class PF3BConfigV1(BaseModel):
    """Strict local-only contract for the PF3B endpoint hunt."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema: Literal["DDMPF3BJointHuntConfigV1"]
    run_id: Literal[RUN_ID]
    lane_id: Literal[LANE_ID]
    delegation_checkpoint_key: Literal[DELEGATION_KEY]
    pf3_config_path: StrictStr
    pf3_config_sha256: StrictStr
    pf3_receipt_path: StrictStr
    pf3_receipt_sha256: StrictStr
    at1x_receipt_path: StrictStr
    at1x_receipt_sha256: StrictStr
    at1x_gaze_path: StrictStr
    at1x_gaze_sha256: StrictStr
    bulk_root: StrictStr
    minimum_free_bytes: StrictInt
    expected_event_hit_probe_count: Literal[68]
    expected_predecessor_measured_count: Literal[16]
    expected_remaining_probe_count: Literal[52]
    rank_batch_size: Literal[1]
    authority_batch_size: Literal[16]
    seed: Literal[1234]
    scorer_threads: Literal[4]
    stop_on_first_strict_joint_improvement: Literal[True]
    measure_sign_and_adjacent_magnitude_neighborhood: Literal[True]
    research_only: Literal[True]
    execution_allowed: Literal[False]
    score_claim: Literal[False]
    main_review_required: Literal[True]
    local_cost_usd: Literal[0]
    no_rg5: Literal[True]
    campaign_fire_allowed: Literal[False]

    @property
    def scorer_batch_size(self) -> int:
        """Expose the PF3 runtime field without duplicating config authority."""

        return self.authority_batch_size


def _read_config(path: Path) -> tuple[PF3BConfigV1, str]:
    payload = path.read_bytes()
    config = PF3BConfigV1.model_validate_json(payload)
    if not Path(config.bulk_root).is_absolute() or not config.bulk_root.startswith(
        "/Volumes/VertigoDataTier/pact/"
    ):
        raise PF3MeasurementError("PF3B bulk root must use the primary SSD")
    return config, hashlib.sha256(payload).hexdigest()


def _storage_preflight(
    config: PF3BConfigV1,
    config_sha256: str,
) -> dict[str, Any]:
    bulk = Path(config.bulk_root)
    bulk.mkdir(parents=True, exist_ok=True)
    path = bulk / "stage_checkpoints/00_storage_preflight.json"
    if path.exists():
        stored = json.loads(path.read_bytes())
        if (
            stored.get("schema") != "ddm_pf3b_storage_preflight.v1"
            or stored.get("typed_config_sha256") != config_sha256
            or stored.get("selected_tier") != str(bulk)
            or stored.get("minimum_free_bytes") != config.minimum_free_bytes
            or stored.get("passed") is not True
        ):
            raise PF3MeasurementError("PF3B storage-preflight resume differs")
        return stored
    usage = shutil.disk_usage(bulk)
    if usage.free < config.minimum_free_bytes:
        raise PF3MeasurementError("PF3B SSD preflight has insufficient free bytes")
    value = {
        "schema": "ddm_pf3b_storage_preflight.v1",
        "typed_config_sha256": config_sha256,
        "selected_tier": str(bulk),
        "free_bytes": usage.free,
        "minimum_free_bytes": config.minimum_free_bytes,
        "passed": True,
        "cleanup_policy": (
            "candidate archives, E4 frames, and per-probe checkpoints are small "
            "durable evidence; only atomic temporary files are success-deleted"
        ),
    }
    _publish(path, value)
    return value


def _identity(row: Mapping[str, Any]) -> tuple[str, str]:
    coordinate = row.get("receiver_actuator_id", row.get("coordinate_id"))
    if not isinstance(coordinate, str):
        raise PF3MeasurementError("PF3B row lacks a coordinate identity")
    return coordinate, str(row["direction_id"])


def _recover_all_event_hits(
    *,
    pf3_config: DDMF3FinitePriceConfigV1,
    rg3_receipt: Mapping[str, Any],
    assignment: Mapping[str, Any],
) -> list[dict[str, Any]]:
    table_ref = rg3_receipt["assignment_table"]
    table = _read_bound_json(
        _resolve(str(table_ref["path"])),
        str(table_ref["file_sha256"]),
        "PF2/MS6 assignment table",
    )
    sweep = rg3_receipt["probe_sweep"]
    checkpoint_root = Path(str(sweep["checkpoint_root"]))
    owner = {
        str(actuator_id): str(row["bucket_id"])
        for row in assignment["rows"]
        for actuator_id in row["receiver_actuator_ids"]
    }
    rows: list[dict[str, Any]] = []
    for result in sorted(
        (
            dict(value)
            for value in table["probe_results"]
            if value.get("receiver_actuator_id") in owner
            and value.get("status") == "MEASURED_ARGMAX_PERTURBATION"
        ),
        key=_identity,
    ):
        safe = str(result["receiver_actuator_id"]).replace(".", "_")
        checkpoint_path = (
            checkpoint_root / f"{safe}__{result['direction_id']}.json"
        )
        checkpoint = _read_bound_json(
            checkpoint_path,
            str(result["checkpoint_sha256"]),
            "RG3 event-hit checkpoint",
        )
        if (
            checkpoint.get("receiver_actuator_id") != result["receiver_actuator_id"]
            or checkpoint.get("direction_id") != result["direction_id"]
            or checkpoint.get("status") != "MEASURED_ARGMAX_PERTURBATION"
        ):
            raise PF3MeasurementError("RG3 event-hit checkpoint identity differs")
        event = checkpoint.get("event_artifact")
        if (
            not isinstance(event, Mapping)
            or not Path(str(event["path"])).is_file()
            or Path(str(event["path"])).is_symlink()
            or _sha256_file(Path(str(event["path"]))) != event["sha256"]
        ):
            raise PF3MeasurementError("RG3 event-hit artifact custody differs")
        rows.append(
            {
                **result,
                "owner_bucket_id": owner[str(result["receiver_actuator_id"])],
                "source_checkpoint": checkpoint,
                "source_checkpoint_path": str(checkpoint_path),
            }
        )
    if len(rows) != pf3_config.expected_event_hit_probe_count:
        raise PF3MeasurementError("PF3B did not recover the sealed 68 event hits")
    if len({_identity(row) for row in rows}) != len(rows):
        raise PF3MeasurementError("PF3B event-hit identities are not unique")
    return rows


def _load_previous_measurements(
    *,
    config: PF3BConfigV1,
    receipt: Mapping[str, Any],
) -> list[dict[str, Any]]:
    custody = receipt.get("inventory", {}).get("candidate_checkpoint_custody", {})
    artifacts = custody.get("artifacts")
    if (
        custody.get("count") != config.expected_predecessor_measured_count
        or not isinstance(artifacts, list)
        or len(artifacts) != config.expected_predecessor_measured_count
    ):
        raise PF3MeasurementError("PF3 predecessor measurement count differs")
    rows = []
    for artifact in artifacts:
        path = Path(str(artifact["path"]))
        checkpoint = _read_bound_json(
            path,
            str(artifact["sha256"]),
            "PF3 predecessor candidate checkpoint",
        )
        rows.append(_attach_checkpoint_path(checkpoint, path))
    if len({_identity(row) for row in rows}) != len(rows):
        raise PF3MeasurementError("PF3 predecessor measurement identities repeat")
    return rows


def _load_at1_pose_trace(config: PF3BConfigV1) -> tuple[dict[int, float], dict[str, Any]]:
    receipt_path = _resolve(config.at1x_receipt_path)
    receipt = _read_bound_json(
        receipt_path,
        config.at1x_receipt_sha256,
        "AT1x tracked receipt",
    )
    gaze_ref = receipt.get("gaze_contraction", {}).get("atlas")
    gaze_path = Path(config.at1x_gaze_path)
    if (
        not isinstance(gaze_ref, Mapping)
        or gaze_ref.get("path") != str(gaze_path)
        or gaze_ref.get("sha256") != config.at1x_gaze_sha256
    ):
        raise PF3MeasurementError("AT1x receipt/gaze binding differs")
    gaze = _read_bound_json(
        gaze_path,
        config.at1x_gaze_sha256,
        "AT1x n600 gaze atlas",
    )
    pair_rows = gaze.get("pair_rows")
    if (
        gaze.get("full_n600_coverage") is not True
        or not isinstance(pair_rows, list)
        or len(pair_rows) != PAIR_COUNT
    ):
        raise PF3MeasurementError("AT1x pose atlas is not full n600")
    traces = {
        int(row["pair_id"]): float(row["pose"]["camera_input_x"]["trace"])
        for row in pair_rows
    }
    if (
        set(traces) != set(range(PAIR_COUNT))
        or any(not math.isfinite(value) or value < 0.0 for value in traces.values())
    ):
        raise PF3MeasurementError("AT1x camera-input pose traces are malformed")
    return traces, {
        "tracked_receipt": _artifact(receipt_path, "AT1x tracked atlas receipt"),
        "gaze_atlas": _artifact(gaze_path, "AT1x n600 camera-input Gram atlas"),
    }


def _event_ids(source: Mapping[str, Any], pair_id: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    event_ref = source["event_artifact"]
    path = Path(str(event_ref["path"]))
    per_bucket = []
    arrays = []
    with np.load(path, allow_pickle=False) as stored:
        expected = {str(row["bucket_id"]) for row in source["bucket_hits"]}
        if set(stored.files) != expected:
            raise PF3MeasurementError("PF3B event bucket vocabulary differs")
        for hit in source["bucket_hits"]:
            bucket_id = str(hit["bucket_id"])
            values = np.asarray(stored[bucket_id], dtype="<u4")
            if (
                values.ndim != 1
                or values.size != int(hit["event_count"])
                or hashlib.sha256(values.tobytes()).hexdigest()
                != hit["event_ids_sha256"]
                or not np.all(values.astype(np.uint64) // (HEIGHT * WIDTH) == pair_id)
            ):
                raise PF3MeasurementError("PF3B event ID custody differs")
            arrays.append(values)
            per_bucket.append(
                {
                    "bucket_id": bucket_id,
                    "event_count": int(values.size),
                    "event_ids_sha256": hit["event_ids_sha256"],
                }
            )
    return np.unique(np.concatenate(arrays)).astype("<u4", copy=False), per_bucket


def _rank_checkpoint_name(row: Mapping[str, Any]) -> str:
    return f"{str(row['checkpoint_sha256'])[:16]}__{str(row['direction_id']).lower()}.json"


def _prescore_probe(
    *,
    row: Mapping[str, Any],
    config: PF3BConfigV1,
    config_sha256: str,
    context: Any,
    base_receiver: Any,
    labels_all: np.ndarray,
    segnet: Any,
    pose_traces: Mapping[int, float],
) -> dict[str, Any]:
    checkpoint = (
        Path(config.bulk_root)
        / "stage_checkpoints/01_ranking"
        / _rank_checkpoint_name(row)
    )
    if checkpoint.exists():
        value = json.loads(checkpoint.read_bytes())
        if (
            value.get("schema") != RANK_SCHEMA
            or value.get("typed_config_sha256") != config_sha256
            or value.get("source_checkpoint_sha256") != row["checkpoint_sha256"]
        ):
            raise PF3MeasurementError("PF3B ranking resume checkpoint differs")
        return value
    archive, pair_ids, infeasible = _compile_rank_probe(context, row)
    source = row["source_checkpoint"]
    expected = source["candidate_archive"]
    if (
        archive is None
        or infeasible is not None
        or len(pair_ids) != 1
        or len(archive) != expected["bytes"]
        or hashlib.sha256(archive).hexdigest() != expected["sha256"]
    ):
        raise PF3MeasurementError("PF3B rank probe did not rebuild exact source archive")
    pair_id = int(pair_ids[0])
    base_camera = base_receiver.render_camera_pairs((pair_id,))
    candidate_camera = _fast_receiver(archive).render_camera_pairs((pair_id,))
    base_cells = forward_seg_argmax(segnet=segnet, camera=base_camera)
    candidate_cells = forward_seg_argmax(segnet=segnet, camera=candidate_camera)
    if base_cells.shape != (1, HEIGHT, WIDTH) or candidate_cells.shape != base_cells.shape:
        raise PF3MeasurementError("PF3B ranking scorer geometry differs")
    events, bucket_rows = _event_ids(source, pair_id)
    local = (events.astype(np.uint64) % (HEIGHT * WIDTH)).astype(np.intp)
    base_event = base_cells[0].reshape(-1)[local]
    candidate_event = candidate_cells[0].reshape(-1)[local]
    target_event = np.asarray(labels_all[pair_id], dtype=np.uint8).reshape(-1)[local]
    if not np.all(base_event != candidate_event):
        raise PF3MeasurementError("PF3B failed to reproduce every sealed MS6 event hit")
    base_error = base_event != target_event
    candidate_error = candidate_event != target_event
    changed_total = int(np.count_nonzero(base_cells != candidate_cells))
    event_count = int(events.size)
    support = source["raster_support"]
    support_cells = int(support["composite_r_cell_count"])
    if event_count > changed_total:
        raise PF3MeasurementError("PF3B event IDs escaped changed argmax cells")
    value = {
        "schema": RANK_SCHEMA,
        "run_id": RUN_ID,
        "lane_id": LANE_ID,
        "typed_config_sha256": config_sha256,
        "coordinate_id": row["receiver_actuator_id"],
        "direction_id": row["direction_id"],
        "source_checkpoint_path": row["source_checkpoint_path"],
        "source_checkpoint_sha256": row["checkpoint_sha256"],
        "pair_id": pair_id,
        "source_candidate_archive": {
            "bytes": len(archive),
            "sha256": hashlib.sha256(archive).hexdigest(),
            "preservation": "REBUILDABLE_FROM_SHA_BOUND_V19C_PLUS_TYPED_RG3_QUANTUM",
        },
        "pf2_ms6_event_direction": {
            "geometry": "SEALED_MS6_BATCH1_EVENT_INTERSECTION_REPLAY",
            "unique_changed_event_count": event_count,
            "base_event_errors": int(np.count_nonzero(base_error)),
            "candidate_event_errors": int(np.count_nonzero(candidate_error)),
            "event_delta_errors": int(
                np.count_nonzero(candidate_error) - np.count_nonzero(base_error)
            ),
            "corrected_event_count": int(np.count_nonzero(base_error & ~candidate_error)),
            "broken_event_count": int(np.count_nonzero(~base_error & candidate_error)),
            "changed_to_other_wrong_class_count": int(
                np.count_nonzero(base_error & candidate_error)
            ),
            "bucket_rows": bucket_rows,
        },
        "joint_spill_guard": {
            "changed_argmax_cells_batch1": changed_total,
            "changed_argmax_cells_outside_pf2_ms6_event_union": changed_total - event_count,
            "composite_r_support_cells": support_cells,
            "support_vs_argmax_relation": (
                "SEPARATE_COORDINATE_NOT_SUBTRACTED: composite-R support is the "
                "actuation footprint; SegNet receptive fields can expand argmax changes"
            ),
            "changed_camera_values": int(support["camera_value_count"]),
            "at1x_camera_input_pose_gram_trace": float(pose_traces[pair_id]),
            "pose_guard_scope": (
                "PAIR-LEVEL AT1X TRACE RANK ONLY; realized Pose6 delta remains "
                "owed to the batch16 joint measurement"
            ),
        },
        "rank_authority": (
            "exact event-error direction under sealed MS6 batch1 geometry; "
            "spill and AT1x trace rank only, never a score"
        ),
        "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
        "score_claim": False,
        "research_only": True,
        "verdict_scope": "INSTANCE ranking evidence for one sealed support-positive RG3 sign",
    }
    _publish(checkpoint, value)
    return value


def _compile_rank_probe(
    context: Any,
    row: Mapping[str, Any],
) -> tuple[bytes | None, tuple[int, ...], str | None]:
    # Local import keeps this runner's scorer/receiver dependency identical to PF3.
    from tools.measure_ddm_ms6_receiver_support import _compile_probe

    return _compile_probe(
        context,
        actuator_id=str(row["receiver_actuator_id"]),
        direction_id=str(row["direction_id"]),
    )


def prescore_rank_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """Order correction-first, then guard against Seg and Pose spill."""

    direction = row["pf2_ms6_event_direction"]
    spill = row["joint_spill_guard"]
    delta = int(direction["event_delta_errors"])
    direction_class = 0 if delta < 0 else 1 if delta == 0 else 2
    return (
        direction_class,
        delta,
        int(spill["changed_argmax_cells_outside_pf2_ms6_event_union"]),
        int(spill["composite_r_support_cells"]),
        float(spill["at1x_camera_input_pose_gram_trace"]),
        -int(direction["unique_changed_event_count"]),
        str(row["coordinate_id"]),
        str(row["direction_id"]),
    )


def _magnitude_family(coordinate_id: str) -> tuple[str, int] | None:
    match = _MAGNITUDE.fullmatch(coordinate_id)
    return None if match is None else (match.group(1), int(match.group(2)))


def neighborhood_identities(
    winner: Mapping[str, Any],
    inventory: list[Mapping[str, Any]],
) -> set[tuple[str, str]]:
    """Return sealed sign twins and adjacent magnitudes for one RG3 address."""

    family = _magnitude_family(str(winner["coordinate_id"]))
    if family is None:
        return {_identity(winner)}
    prefix, magnitude = family
    allowed_magnitudes = {magnitude}
    if magnitude > 1:
        allowed_magnitudes.add(magnitude - 1)
    if magnitude < 2:
        allowed_magnitudes.add(magnitude + 1)
    output = set()
    for row in inventory:
        candidate = _magnitude_family(str(row["receiver_actuator_id"]))
        if candidate is not None and candidate[0] == prefix and candidate[1] in allowed_magnitudes:
            output.add(_identity(row))
    return output


def _measurement_summary(
    checkpoint: Mapping[str, Any],
    *,
    source: str,
) -> dict[str, Any]:
    delta = checkpoint["five_pf3_edges"]["candidate_delta"]
    rate = checkpoint["five_pf3_edges"]["dimension_rate_home"]
    delta_joint = float(delta["delta_D_joint"])
    delta_bytes = int(rate["delta_counted_bytes_dimension"])
    delta_rate = 25.0 * delta_bytes / RATE_DENOMINATOR_BYTES
    return {
        "coordinate_id": checkpoint["coordinate_id"],
        "direction_id": checkpoint["direction_id"],
        "measurement_source": source,
        "checkpoint_sha256": _sha256_file(Path(str(checkpoint["_checkpoint_path"]))),
        "pair_id": int(checkpoint["pair_id"]),
        "delta_global_errors": int(delta["delta_global_errors"]),
        "delta_global_pose_sse_6d": float(delta["delta_global_pose_sse_6d"]),
        "delta_D_seg": float(delta["delta_D_seg"]),
        "delta_D_pose": float(delta["delta_D_pose"]),
        "delta_D_joint": delta_joint,
        "delta_counted_bytes_E4": delta_bytes,
        "delta_S_rate": delta_rate,
        "delta_S_exact_formula": delta_joint + delta_rate,
        "strict_joint_improving": delta_joint < 0.0,
        "total_score_improving_after_rate": delta_joint + delta_rate < 0.0,
        "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
    }


def _attach_checkpoint_path(value: dict[str, Any], path: Path) -> dict[str, Any]:
    return {**value, "_checkpoint_path": str(path)}


def _write_outputs(
    *,
    config: PF3BConfigV1,
    config_sha256: str,
    storage: Mapping[str, Any],
    rank_rows: list[dict[str, Any]],
    previous: list[dict[str, Any]],
    measured: list[dict[str, Any]],
    all_hits: list[dict[str, Any]],
    at1_custody: Mapping[str, Any],
    pf3_config: DDMF3FinitePriceConfigV1,
    pf3_receipt_path: Path,
    winner: dict[str, Any] | None,
    neighborhood: set[tuple[str, str]],
) -> dict[str, Any]:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rank_path = OUTPUT / "ranked_inventory.json"
    rank_payload = {
        "schema": "ddm_pf3b_ranked_inventory.v1",
        "typed_config_sha256": config_sha256,
        "ranking_policy": (
            "exact MS6-batch1 event-error direction; then non-event argmax spill, "
            "composite-R actuation footprint, AT1x camera-input Pose Gram trace, "
            "event mass"
        ),
        "rank_batch_size": config.rank_batch_size,
        "authority_batch_size": config.authority_batch_size,
        "rows": rank_rows,
        "score_claim": False,
        "verdict_scope": "INSTANCE ranking of the 52 open support-positive RG3 signs",
    }
    _publish(rank_path, rank_payload)
    prior_summaries = [
        _measurement_summary(row, source="PF3_PREDECESSOR_BATCH16_E4")
        for row in previous
    ]
    current_summaries = [
        _measurement_summary(row, source="PF3B_RANKED_BATCH16_E4")
        for row in measured
    ]
    by_identity = {
        (row["coordinate_id"], row["direction_id"]): row
        for row in [*prior_summaries, *current_summaries]
    }
    sealed_rows = []
    rank_by_identity = {
        (row["coordinate_id"], row["direction_id"]): row for row in rank_rows
    }
    for source in sorted(all_hits, key=_identity):
        identity = _identity(source)
        summary = by_identity.get(identity)
        sealed_rows.append(
            {
                "coordinate_id": identity[0],
                "direction_id": identity[1],
                "source_checkpoint_sha256": source["checkpoint_sha256"],
                "ranking": rank_by_identity.get(identity),
                "measurement": summary,
                "measurement_status": (
                    "MEASURED_BATCH16_E4"
                    if summary is not None
                    else "UNMEASURED_AFTER_STOP_RULE"
                ),
            }
        )
    all_measured = len(by_identity) == config.expected_event_hit_probe_count
    all_nonimproving = all(
        float(row["delta_D_joint"]) >= 0.0 for row in by_identity.values()
    )
    if winner is None:
        if not all_measured or not all_nonimproving:
            raise PF3MeasurementError("PF3B exhaustion verdict preconditions differ")
        verdict = (
            "SEALED_EXISTING_RG3_SUPPORT_POSITIVE_ALPHABET_EXHAUSTED_"
            "NO_STRICT_JOINT_IMPROVEMENT"
        )
    else:
        verdict = "FOUND_STRICT_JOINT_IMPROVING_EXISTING_RG3_EDGE"
    measured_checkpoint_artifacts = [
        _artifact(
            Path(str(row["_checkpoint_path"])),
            "immutable PF3B ranked batch16/E4 measurement",
        )
        for row in measured
    ]
    winner_summary = (
        by_identity[(winner["coordinate_id"], winner["direction_id"])]
        if winner is not None
        else None
    )
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "run_id": RUN_ID,
        "lane_id": LANE_ID,
        "delegation_checkpoint_key": config.delegation_checkpoint_key,
        "typed_config": config.model_dump(mode="json"),
        "typed_config_sha256": config_sha256,
        "authority": {
            "research_only": True,
            "execution_allowed": False,
            "score_claim": False,
            "promotion_eligible": False,
            "main_landing_review_required": True,
            "local_cost_usd": 0,
            "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
            "competitive_frontier": (
                "official leaderboard displayed 0.172; no competitive pointer "
                "mutation or campaign fire is authorized here"
            ),
            "competitive_frontier_moved_by_this_arm": False,
        },
        "storage_preflight": dict(storage),
        "source_custody": {
            "pf3_typed_config": _artifact(
                _resolve(config.pf3_config_path),
                "PF3 physical edge and scorer contract",
            ),
            "pf3_receipt": _artifact(
                pf3_receipt_path,
                "PF3 first 16 batch16/E4 measurements",
            ),
            "pf3_base_coder_root": str(Path(pf3_config.bulk_root) / "coder_base_frames"),
            "at1x": dict(at1_custody),
        },
        "counts": {
            "sealed_event_hit_probe_count": len(all_hits),
            "predecessor_measured_count": len(previous),
            "ranked_remaining_probe_count": len(rank_rows),
            "new_batch16_E4_measurement_count": len(measured),
            "total_batch16_E4_measured_count": len(by_identity),
            "remaining_after_stop_count": len(all_hits) - len(by_identity),
            "strict_joint_improving_count_measured": sum(
                float(row["delta_D_joint"]) < 0.0 for row in by_identity.values()
            ),
        },
        "ranking": {
            "artifact": _artifact(
                rank_path,
                "ranked 52-probe PF2/MS6 event and AT1x spill inventory",
            ),
            "rank_authority": (
                "batch1 ranking evidence only; every verdict delta uses PF3 batch16 "
                "n600 endpoint geometry"
            ),
        },
        "stop_rule": {
            "policy": "stop on first strict negative realized batch16 delta_D_joint",
            "triggered": winner is not None,
            "winner": winner_summary,
            "neighborhood_identities": [
                {"coordinate_id": coordinate, "direction_id": direction}
                for coordinate, direction in sorted(neighborhood)
            ],
        },
        "measurement_checkpoint_custody": {
            "count": len(measured_checkpoint_artifacts),
            "artifacts": measured_checkpoint_artifacts,
            "digest_chain_sha256": hashlib.sha256(
                "".join(row["sha256"] for row in measured_checkpoint_artifacts).encode()
            ).hexdigest(),
        },
        "sealed_event_hit_probe_rows": sealed_rows,
        "measurement_extrema": {
            "delta_D_joint_min": min(float(row["delta_D_joint"]) for row in by_identity.values()),
            "delta_D_joint_max": max(float(row["delta_D_joint"]) for row in by_identity.values()),
            "delta_S_exact_formula_min": min(
                float(row["delta_S_exact_formula"]) for row in by_identity.values()
            ),
            "delta_S_exact_formula_max": max(
                float(row["delta_S_exact_formula"]) for row in by_identity.values()
            ),
            "delta_counted_bytes_E4_min": min(
                int(row["delta_counted_bytes_E4"]) for row in by_identity.values()
            ),
            "delta_counted_bytes_E4_max": max(
                int(row["delta_counted_bytes_E4"]) for row in by_identity.values()
            ),
        },
        "triality": {
            "dsl": str(CONFIG.relative_to(REPO)),
            "dag": f".omx/research/{RUN_ID}/DAG_FEED.json",
            "equations": ["cgauge_master_action_v1"],
            "equation_registration": (
                "NO_NEW_PRICE_OR_DIRECTION_LAW; PF3B reuses the registered joint "
                "action and reports empirical event-direction ordering only"
            ),
        },
        "stretch": {
            "composed_pairs_ran": False,
            "reason": (
                "not authorized before a strict single-edge improvement"
                if winner is None
                else "single-edge neighborhood consumes this bounded delegated unit"
            ),
        },
        "verdict": verdict,
        "verdict_scope": (
            "INSTANCE: the sealed 68 support-positive signed RG3 probes around "
            "the SHA-bound V19C endpoint; not an RG3 family or paradigm negative"
        ),
        "main_landing_review_required": True,
    }
    receipt_path = OUTPUT / "receipt.json"
    _publish(receipt_path, receipt)
    dag = {
        "schema": "ddm_pf3b_DAG_feed.v1",
        "node_id": RUN_ID,
        "status": "MEASURED_ADVISORY",
        "parents": [
            "ddm_pf3_finite_price_materialization_20260725T193409Z",
            "ddm_rg3_residual_family_productions_20260724T110418Z",
            "ddm_at1x_atlas_materialize_20260723",
        ],
        "receipt": _artifact(receipt_path, "PF3B sealed hunt receipt"),
        "verdict": verdict,
        "pointer_delta": "UNCHANGED",
        "next": (
            "MAIN review the strict joint-improving edge and its E4 total price"
            if winner is not None
            else "Treat only this existing V19C RG3 endpoint alphabet as sealed exhausted"
        ),
        "verdict_scope": receipt["verdict_scope"],
        "main_landing_review_required": True,
    }
    _publish(OUTPUT / "DAG_FEED.json", dag)
    return receipt


def hunt(config_path: Path = CONFIG) -> dict[str, Any]:
    config, config_sha256 = _read_config(config_path)
    storage = _storage_preflight(config, config_sha256)
    pf3_config_path = _resolve(config.pf3_config_path)
    if _sha256_file(pf3_config_path) != config.pf3_config_sha256:
        raise PF3MeasurementError("PF3 predecessor config SHA differs")
    pf3_config, pf3_config_sha = _read_pf3_config(pf3_config_path)
    if (
        pf3_config_sha != config.pf3_config_sha256
        or pf3_config.scorer_batch_size != config.authority_batch_size
        or pf3_config.seed != config.seed
        or pf3_config.scorer_threads != config.scorer_threads
    ):
        raise PF3MeasurementError("PF3B/PF3 scorer contract differs")
    pf3_receipt_path = _resolve(config.pf3_receipt_path)
    pf3_receipt = _read_bound_json(
        pf3_receipt_path,
        config.pf3_receipt_sha256,
        "PF3 predecessor receipt",
    )
    selected, rg3_receipt, assignment, direct, _rd1 = _load_inventory(pf3_config)
    previous = _load_previous_measurements(config=config, receipt=pf3_receipt)
    selected_identities = {_identity(row) for row in selected}
    previous_identities = {_identity(row) for row in previous}
    if selected_identities != previous_identities:
        raise PF3MeasurementError("PF3 selected inventory and measurement identities differ")
    all_hits = _recover_all_event_hits(
        pf3_config=pf3_config,
        rg3_receipt=rg3_receipt,
        assignment=assignment,
    )
    remaining = [row for row in all_hits if _identity(row) not in previous_identities]
    if (
        len(previous) != config.expected_predecessor_measured_count
        or len(remaining) != config.expected_remaining_probe_count
        or len(all_hits) != config.expected_event_hit_probe_count
    ):
        raise PF3MeasurementError("PF3B 16 + 52 = 68 partition differs")
    pose_traces, at1_custody = _load_at1_pose_trace(config)
    base_archive_path = _resolve(pf3_config.base_archive_path)
    if (
        not base_archive_path.is_file()
        or base_archive_path.is_symlink()
        or _sha256_file(base_archive_path) != pf3_config.base_archive_sha256
    ):
        raise PF3MeasurementError("PF3B V19C base archive custody differs")
    base_archive = base_archive_path.read_bytes()
    base_receipt = _read_bound_json(
        _resolve(pf3_config.base_receipt_path),
        pf3_config.base_receipt_sha256,
        "V19C base receipt",
    )
    endpoint_errors, endpoint_d_pose = _endpoint(base_receipt)
    scorer_config, _target_custody = _verify_scorer_config(pf3_config)
    labels_all = open_stored_npy_memmap(Path(scorer_config.target_cache_path), "lstars")
    poses_all = open_stored_npy_memmap(Path(scorer_config.target_cache_path), "gt_poses")
    segnet, posenet, scorer_custody = _load_models(scorer_config)
    context = _probe_context(base_archive)
    base_receiver = _fast_receiver(base_archive)
    raw_rank_rows = [
        _prescore_probe(
            row=row,
            config=config,
            config_sha256=config_sha256,
            context=context,
            base_receiver=base_receiver,
            labels_all=labels_all,
            segnet=segnet,
            pose_traces=pose_traces,
        )
        for row in remaining
    ]
    raw_rank_rows.sort(key=prescore_rank_key)
    trace_order = {
        pair_id: index
        for index, (pair_id, _value) in enumerate(
            sorted(pose_traces.items(), key=lambda item: (item[1], item[0])),
            start=1,
        )
    }
    rank_rows = []
    inventory_by_identity = {_identity(row): row for row in remaining}
    for rank, row in enumerate(raw_rank_rows, start=1):
        pair_id = int(row["pair_id"])
        rank_rows.append(
            {
                "rank": rank,
                "authority_measurement_index": config.expected_predecessor_measured_count + rank,
                "coordinate_id": row["coordinate_id"],
                "direction_id": row["direction_id"],
                "source_checkpoint_sha256": row["source_checkpoint_sha256"],
                "pair_id": pair_id,
                "pf2_ms6_event_direction": row["pf2_ms6_event_direction"],
                "joint_spill_guard": {
                    **row["joint_spill_guard"],
                    "at1x_pose_null_rank_of_600": trace_order[pair_id],
                    "at1x_pose_null_percentile": trace_order[pair_id] / PAIR_COUNT,
                },
                "rank_checkpoint": _artifact(
                    Path(config.bulk_root)
                    / "stage_checkpoints/01_ranking"
                    / _rank_checkpoint_name(inventory_by_identity[(row["coordinate_id"], row["direction_id"])]),
                    "immutable PF3B pre-score ranking checkpoint",
                ),
            }
        )
    rank_stage = {
        "schema": "ddm_pf3b_rank_stage_complete.v1",
        "typed_config_sha256": config_sha256,
        "ranked_probe_count": len(rank_rows),
        "ordered_source_checkpoint_sha256": [
            row["source_checkpoint_sha256"] for row in rank_rows
        ],
        "status": "COMPLETE",
    }
    _publish(
        Path(config.bulk_root) / "stage_checkpoints/01_rank_complete.json",
        rank_stage,
    )
    base_race, base_frames, _base_frame_artifacts = _load_or_race_base_coders(
        base_archive,
        Path(pf3_config.bulk_root),
    )
    if base_race["winner"]["codec"] != "E4_BROTLI_Q11":
        raise PF3MeasurementError("PF3B predecessor base coder owner is no longer E4")
    occupied = [
        dict(row)
        for row in direct["rows"]
        if isinstance(row.get("event_count"), int) and row["event_count"] > 0
    ]
    g4_by_bucket = {
        str(row["bucket_id"]): str(row["g4_temporal_class"]) for row in occupied
    }
    base_batch_cache: dict[
        int,
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ] = {}
    replayed_batch_starts: set[int] = set()
    measured: list[dict[str, Any]] = []
    measured_identities: set[tuple[str, str]] = set()
    winner: dict[str, Any] | None = None
    rank_by_identity = {
        (row["coordinate_id"], row["direction_id"]): row for row in rank_rows
    }

    def measure(identity: tuple[str, str]) -> dict[str, Any]:
        rank_row = rank_by_identity[identity]
        inventory_row = inventory_by_identity[identity]
        index = int(rank_row["authority_measurement_index"])
        checkpoint_path = (
            Path(config.bulk_root)
            / "stage_checkpoints/02_candidates"
            / f"{index:03d}.json"
        )
        value = _measure_candidate(
            index=index,
            inventory_row=inventory_row,
            config=config,
            config_sha256=config_sha256,
            context=context,
            base_receiver=base_receiver,
            endpoint_errors=endpoint_errors,
            endpoint_d_pose=endpoint_d_pose,
            labels_all=labels_all,
            poses_all=poses_all,
            segnet=segnet,
            posenet=posenet,
            scorer_custody=scorer_custody,
            base_batch_cache=base_batch_cache,
            replayed_batch_starts=replayed_batch_starts,
            base_coder_frames=base_frames,
            g4_by_bucket=g4_by_bucket,
            run_id=RUN_ID,
            lane_id=LANE_ID,
            checkpoint_schema=CHECKPOINT_SCHEMA,
            candidate_prefix="pf3b",
        )
        attached = _attach_checkpoint_path(value, checkpoint_path)
        if identity not in measured_identities:
            measured.append(attached)
            measured_identities.add(identity)
        return attached

    for rank_row in rank_rows:
        identity = (rank_row["coordinate_id"], rank_row["direction_id"])
        value = measure(identity)
        if float(value["five_pf3_edges"]["candidate_delta"]["delta_D_joint"]) < 0.0:
            winner = value
            break
    neighborhood: set[tuple[str, str]] = set()
    if winner is not None:
        neighborhood = neighborhood_identities(winner, all_hits)
        for identity in sorted(neighborhood):
            if identity in inventory_by_identity and identity not in measured_identities:
                measure(identity)
    complete = {
        "schema": "ddm_pf3b_measurement_stage_complete.v1",
        "typed_config_sha256": config_sha256,
        "measured_count": len(measured),
        "winner_identity": (
            {
                "coordinate_id": winner["coordinate_id"],
                "direction_id": winner["direction_id"],
            }
            if winner is not None
            else None
        ),
        "stop_rule_triggered": winner is not None,
        "status": "COMPLETE",
    }
    _publish(
        Path(config.bulk_root) / "stage_checkpoints/03_measurement_complete.json",
        complete,
    )
    return _write_outputs(
        config=config,
        config_sha256=config_sha256,
        storage=storage,
        rank_rows=rank_rows,
        previous=previous,
        measured=measured,
        all_hits=all_hits,
        at1_custody=at1_custody,
        pf3_config=pf3_config,
        pf3_receipt_path=pf3_receipt_path,
        winner=winner,
        neighborhood=neighborhood,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    receipt = hunt(args.config)
    print(
        json.dumps(
            {
                "verdict": receipt["verdict"],
                "counts": receipt["counts"],
                "measurement_extrema": receipt["measurement_extrema"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
