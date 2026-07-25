# SPDX-License-Identifier: MIT
"""Fail-closed C1 byte-home attribution for the EV2 waterfill edge.

The current C1 member is a deterministic, receiver-closed object, but its
counted streams are compressed jointly across 600 pairs.  Exact ZIP homes can
therefore be proved at the stream level without implying a finer
``pair x scorer cell`` ownership.  This module preserves that distinction:
all measured C1 mass is conserved, inseparable mass is typed UNALLOCATED, and
the 162 RD1 prices remain NULL unless an exclusive final-byte foreign key is
present.
"""

from __future__ import annotations

import hashlib
import io
import json
import struct
import zipfile
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Final

from tac.optimization.ddm_min_description_contract import (
    build_minimum_description_headline,
)
from tac.optimization.ddm_pf2_bucket_assignment import (
    canonical_sha256 as ms5_canonical_sha256,
)
from tac.optimization.ddm_pf2_bucket_assignment import validate_assignment_table

SCHEMA: Final = "ddm_ev2_per_pair_allocation_receipt.v1"
TABLE_SCHEMA: Final = "ddm_ev2_ms5_loader_assignment_table.v1"
ROW_SCHEMA: Final = "ddm_ev2_ms5_loader_assignment_row.v1"
PAIR_SCHEMA: Final = "ddm_ev2_per_pair_allocation_row.v1"
COARSE_HOME_SCHEMA: Final = "ddm_ev2_coarse_stream_home.v1"
BACKFILL_SCHEMA: Final = "ddm_ev2_rd1_162_dual_backfill.v1"
HEADLINE_SCHEMA: Final = "ddm_ev2_headline_replay.v1"
PAIR_COUNT: Final = 600
CELL_COUNT: Final = 162
MEASURED_C1_BYTES: Final = 134_211
ARCHIVE_BYTES: Final = 133_941
LANE_SEED_BYTES: Final = 270
FALSIFIER_UNALLOCATED_FRACTION: Final = 0.30
MS5_TABLE_SCHEMA: Final = "ddm_ms5_pf2_bucket_assignment_table.v1"
UNALLOCATED_STATUS: Final = (
    "UNALLOCATED_NO_EXCLUSIVE_FINAL_BYTE_PAIR_AND_CELL_FOREIGN_KEY"
)
EXPECTED_HEADLINE_BLOCKERS: Final = (
    "POSE_TUBE_NOT_ACTIVE_IN_SOLVE",
    "TYPED_SUBPROBLEM_ALTERNATION_NOT_ACTIVE",
    "TYPED_BLOCK_ATLAS_NOT_ACTIVE",
    "PER_DIMENSION_EFFECTIVE_QUANTA_NOT_ACTIVE",
)
EXPECTED_OUTER_HOMES: Final = {
    "manifest.json": ("manifest", 3_345),
    "predictor.zip": ("v15_predictor_zip_outer_home", 100_099),
    "predict/movable_polygon_worldsheet.g1s": (
        "g1_movable_worldsheet_outer_home",
        29_878,
    ),
    "render/receiver_realization.ddrp": ("receiver_realization_profile", 85),
    "render/scorer_solved_templates.ddst": ("solved_template_outer_home", 151),
    "__central_directory_and_eocd__": ("central_directory_and_eocd", 383),
}
COUNTED_LP1_STREAMS: Final = {
    "v15_predictor_zip_outer_home",
    "g1_movable_worldsheet_outer_home",
    "receiver_realization_profile",
    "solved_template_outer_home",
    "manifest",
    "central_directory_and_eocd",
    "lane_program_seed",
}
CELL_KEY_FIELDS: Final = (
    "dual_index",
    "stratum",
    "scorer_visibility",
    "g4_temporal_class",
)
DUAL_INDICES: Final = (1, 2, 3)
SEMANTIC_STRATA: Final = (
    "Lane",
    "Movable",
    "MyCar",
    "POSE6_GLOBAL",
    "Road",
    "Undrivable",
)
SCORER_VISIBILITIES: Final = (
    "ker(A)-invisible",
    "pose-visible",
    "seg-visible",
)
G4_TEMPORAL_CLASSES: Final = (
    "STATIC_IN_IMAGE",
    "STATIC_IN_XI_PROXY",
    "TRANSIENT",
)
EXPECTED_CELL_KEYS: Final = frozenset(
    product(
        DUAL_INDICES,
        SEMANTIC_STRATA,
        SCORER_VISIBILITIES,
        G4_TEMPORAL_CLASSES,
    )
)


class EV2AllocationError(ValueError):
    """An input drifted or a finer byte attribution was asserted without proof."""


@dataclass(frozen=True, slots=True)
class EV2BuildResult:
    allocation_table: dict[str, Any]
    ms5_loader_table: dict[str, Any]
    rd1_backfill: dict[str, Any]
    headline_replay: dict[str, Any]
    receipt: dict[str, Any]


def canonical_bytes(value: Any) -> bytes:
    """Return deterministic JSON bytes for receipts and content hashes."""

    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _exact_nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise EV2AllocationError(f"{label} must be an exact nonnegative integer")
    return value


def _cell_key(row: Mapping[str, Any]) -> tuple[int, str, str, str]:
    key = (
        _exact_nonnegative_int(row.get("dual_index"), "dual_index"),
        str(row.get("stratum")),
        str(row.get("scorer_visibility")),
        str(row.get("g4_temporal_class")),
    )
    if key not in EXPECTED_CELL_KEYS:
        raise EV2AllocationError(f"cell key is outside the sealed cube: {key}")
    return key


def _false_authority(value: Mapping[str, Any], label: str) -> None:
    if not isinstance(value, Mapping):
        raise EV2AllocationError(f"{label} authority must be an object")
    for key in ("score_claim", "promotion_eligible", "pointer_moved"):
        if key in value and value.get(key) is not False:
            raise EV2AllocationError(f"{label}.{key} must remain false")


def _zip_home_rows(archive: bytes) -> tuple[list[dict[str, Any]], dict[str, bytes]]:
    """Return exact non-overlapping outer ZIP byte ranges and member payloads."""

    try:
        with zipfile.ZipFile(io.BytesIO(archive), "r") as reader:
            infos = reader.infolist()
            members = {row.filename: reader.read(row) for row in infos}
            start_dir = reader.start_dir
    except (KeyError, OSError, zipfile.BadZipFile) as exc:
        raise EV2AllocationError("C1 archive is malformed") from exc
    if [row.filename for row in infos] != list(EXPECTED_OUTER_HOMES)[:-1]:
        raise EV2AllocationError("C1 outer member order differs")
    boundaries = [row.header_offset for row in infos[1:]] + [start_dir]
    rows = []
    for info, stop in zip(infos, boundaries, strict=True):
        stream, expected_bytes = EXPECTED_OUTER_HOMES[info.filename]
        home_bytes = stop - info.header_offset
        if home_bytes != expected_bytes:
            raise EV2AllocationError(f"C1 outer home differs for {info.filename}")
        rows.append(
            {
                "archive_member": info.filename,
                "stream": stream,
                "byte_range": [info.header_offset, stop],
                "counted_bytes": home_bytes,
                "payload_bytes": info.file_size,
                "payload_sha256": hashlib.sha256(members[info.filename]).hexdigest(),
            }
        )
    stream, expected_bytes = EXPECTED_OUTER_HOMES[
        "__central_directory_and_eocd__"
    ]
    if len(archive) - start_dir != expected_bytes:
        raise EV2AllocationError("C1 central-directory home differs")
    rows.append(
        {
            "archive_member": "__central_directory_and_eocd__",
            "stream": stream,
            "byte_range": [start_dir, len(archive)],
            "counted_bytes": len(archive) - start_dir,
            "payload_bytes": 0,
            "payload_sha256": None,
        }
    )
    if sum(row["counted_bytes"] for row in rows) != len(archive):
        raise EV2AllocationError("C1 outer ZIP homes do not conserve archive bytes")
    return rows, members


def _joint_coder_custody(members: Mapping[str, bytes]) -> dict[str, Any]:
    """Prove that the two large C1 homes expose no final pair-byte boundary."""

    predictor = members.get("predictor.zip")
    worldsheet = members.get("predict/movable_polygon_worldsheet.g1s")
    if predictor is None or worldsheet is None:
        raise EV2AllocationError("C1 joint-coded homes are absent")
    try:
        with zipfile.ZipFile(io.BytesIO(predictor), "r") as reader:
            predictor_members = {
                row.filename: reader.read(row) for row in reader.infolist()
            }
        manifest = json.loads(predictor_members["manifest.json"])
    except (KeyError, OSError, UnicodeDecodeError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
        raise EV2AllocationError("C1 predictor member is malformed") from exc
    if manifest.get("pair_count") != PAIR_COUNT:
        raise EV2AllocationError("C1 predictor pair count differs")
    coded_predictor_members = sorted(
        name
        for name in predictor_members
        if name == "chart.zip" or name.endswith((".br", ".lz"))
    )
    if len(coded_predictor_members) != 11:
        raise EV2AllocationError("C1 predictor coded-stream inventory differs")

    if len(worldsheet) < 5 or worldsheet[:4] != b"G1S1" or worldsheet[4] != 3:
        raise EV2AllocationError("C1 G1 worldsheet envelope differs")
    offset = 5
    worldsheet_sections = []
    for _ in range(3):
        if offset + 10 > len(worldsheet):
            raise EV2AllocationError("C1 G1 worldsheet envelope is truncated")
        production_id, codec_id, raw_bytes, coded_bytes = struct.unpack_from(
            "<BBII", worldsheet, offset
        )
        start = offset
        offset += 10
        stop = offset + coded_bytes
        if stop > len(worldsheet):
            raise EV2AllocationError("C1 G1 coded stream is truncated")
        worldsheet_sections.append(
            {
                "production_id": production_id,
                "codec_id": codec_id,
                "raw_bytes": raw_bytes,
                "coded_bytes": coded_bytes,
                "envelope_byte_range": [start, stop],
                "pair_byte_boundaries": None,
            }
        )
        offset = stop
    if offset != len(worldsheet):
        raise EV2AllocationError("C1 G1 envelope has trailing bytes")
    return {
        "predictor": {
            "pair_count": PAIR_COUNT,
            "joint_coded_member_count": len(coded_predictor_members),
            "joint_coded_members": coded_predictor_members,
            "exclusive_final_byte_pair_boundaries_present": False,
            "derivation_method": "EXACT_CONSTRUCTION_LINEAGE",
        },
        "g1_worldsheet": {
            "pair_count": PAIR_COUNT,
            "joint_coded_production_count": len(worldsheet_sections),
            "coded_productions": worldsheet_sections,
            "exclusive_final_byte_pair_boundaries_present": False,
            "derivation_method": "EXACT_CONSTRUCTION_LINEAGE",
        },
    }


def _lp1_rows(lp1: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    if lp1.get("schema") != "ddm_lp1_layer_pricing.v1":
        raise EV2AllocationError("LP1 receipt schema differs")
    _false_authority(lp1, "LP1")
    waterfill = lp1.get("c1_corrected_waterfill")
    rows = waterfill.get("rows") if isinstance(waterfill, Mapping) else None
    if (
        not isinstance(rows, Sequence)
        or waterfill.get("corrected_measured_allocated_bytes") != MEASURED_C1_BYTES
        or waterfill.get("source_exact_control_subtotal_bytes") != ARCHIVE_BYTES
    ):
        raise EV2AllocationError("LP1 measured C1 allocation differs")
    counted = {
        str(row.get("stream")): row
        for row in rows
        if row.get("counted_in_corrected_total") is True
    }
    if set(counted) != COUNTED_LP1_STREAMS:
        raise EV2AllocationError("LP1 counted stream inventory differs")
    if (
        sum(
            _exact_nonnegative_int(
                row.get("corrected_allocated_bytes"),
                f"LP1 {name}.corrected_allocated_bytes",
            )
            for name, row in counted.items()
        )
        != MEASURED_C1_BYTES
    ):
        raise EV2AllocationError("LP1 counted homes do not conserve 134211 bytes")
    return counted


def _ev1_cells(ev1: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], dict[str, Any]]:
    if ev1.get("schema") != "ddm_ev1_campaign_evidence_join_receipt.v1":
        raise EV2AllocationError("EV1 receipt schema differs")
    _false_authority(ev1, "EV1")
    rd1 = ev1.get("rd1_evidence")
    cells = rd1.get("bucket_rows") if isinstance(rd1, Mapping) else None
    summaries = rd1.get("edge_summaries") if isinstance(rd1, Mapping) else None
    if (
        not isinstance(cells, list)
        or len(cells) != CELL_COUNT
        or not isinstance(summaries, list)
        or len(summaries) != 3
    ):
        raise EV2AllocationError("EV1 162-cell accounting cube differs")
    keys = [_cell_key(row) for row in cells]
    if set(keys) != EXPECTED_CELL_KEYS:
        raise EV2AllocationError("EV1 cell identities do not cover the sealed cube")
    per_dual = {}
    for summary in summaries:
        dual = _exact_nonnegative_int(summary.get("dual_index"), "EV1 dual_index")
        if dual in per_dual:
            raise EV2AllocationError("EV1 edge summary dual identities are duplicated")
        expected = _exact_nonnegative_int(
            summary.get("delta_counted_bytes"), "EV1 edge bytes"
        )
        observed = sum(
            _exact_nonnegative_int(
                row.get("delta_counted_bytes_dimension"),
                "EV1 cell bytes",
            )
            for row in cells
            if row.get("dual_index") == dual
        )
        if observed != expected:
            raise EV2AllocationError(f"EV1 dual {dual} accounting homes do not conserve")
        per_dual[str(dual)] = {
            "edge_delta_counted_bytes": expected,
            "cell_home_sum_bytes": observed,
            "reconciled": True,
        }
    if set(per_dual) != {str(value) for value in DUAL_INDICES}:
        raise EV2AllocationError("EV1 edge summaries do not cover all duals")
    return cells, {
        "cell_count": CELL_COUNT,
        "per_dual": per_dual,
        "all_exclusive_accounting_homes_reconciled": True,
        "same_object_as_c1_allocation": False,
        "forbidden_join": (
            "EV1 endpoint-edge accounting bytes are not C1 archive byte homes."
        ),
    }


def _rd1_rows(rd1: Mapping[str, Any]) -> dict[tuple[int, str, str, str], Mapping[str, Any]]:
    if rd1.get("schema") != "ddm_rd1_dimension_duals_effective_quantum.v1":
        raise EV2AllocationError("RD1 source schema differs")
    _false_authority(rd1, "RD1")
    dimensions = rd1.get("dimension_duals")
    rows = dimensions.get("bucket_rows") if isinstance(dimensions, Mapping) else None
    if not isinstance(rows, list) or len(rows) != CELL_COUNT:
        raise EV2AllocationError("RD1 source must contain 162 cells")
    result = {_cell_key(row): row for row in rows}
    if set(result) != EXPECTED_CELL_KEYS:
        raise EV2AllocationError("RD1 source cells do not cover the sealed cube")
    if any(row.get("lambda_bytes_per_D_dimension") is not None for row in rows):
        raise EV2AllocationError("RD1 source unexpectedly contains a finite cell price")
    return result


def _coarse_home_rows(
    *,
    lp1_rows: Mapping[str, Mapping[str, Any]],
    outer_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_stream = {str(row["stream"]): row for row in outer_rows}
    rows = []
    for stream in sorted(COUNTED_LP1_STREAMS):
        lp1 = lp1_rows[stream]
        counted = _exact_nonnegative_int(
            lp1.get("corrected_allocated_bytes"),
            f"LP1 {stream}.corrected_allocated_bytes",
        )
        archive = by_stream.get(stream)
        if stream == "lane_program_seed":
            if counted != LANE_SEED_BYTES:
                raise EV2AllocationError("LP1 lane seed bytes differ")
            byte_range = None
            archive_member = None
            source_home = "LP1 measured receiver-owning delta outside v15 control ZIP"
        else:
            if archive is None or archive["counted_bytes"] != counted:
                raise EV2AllocationError(f"LP1/archive bytes differ for {stream}")
            byte_range = archive["byte_range"]
            archive_member = archive["archive_member"]
            source_home = "exact v15 outer ZIP unique home"
        rows.append(
            {
                "schema": COARSE_HOME_SCHEMA,
                "stream": stream,
                "archive_member": archive_member,
                "byte_range": byte_range,
                "counted_bytes": counted,
                "typed_home": lp1.get("typed_home"),
                "pair_id": None,
                "cell_key": None,
                "assignment_status": UNALLOCATED_STATUS,
                "derivation_method": "EXACT_CONSTRUCTION_LINEAGE",
                "same_object": True,
                "source_home": source_home,
                "unallocated_reason": (
                    "The exact counted home is shared, container-level, or jointly "
                    "compressed across pairs; no exclusive final-byte interval is "
                    "owned by one pair and one EV1 cell."
                ),
            }
        )
    if sum(row["counted_bytes"] for row in rows) != MEASURED_C1_BYTES:
        raise EV2AllocationError("coarse C1 homes do not conserve 134211 bytes")
    return rows


def _allocation_table(
    cells: Sequence[Mapping[str, Any]],
    coarse_rows: Sequence[Mapping[str, Any]],
    *,
    ms5: Mapping[str, Any],
    ms5_sha256: str,
) -> dict[str, Any]:
    if ms5.get("schema") != MS5_TABLE_SCHEMA:
        raise EV2AllocationError("MS5 loader source schema differs")
    _false_authority(ms5, "MS5")
    rows = []
    for cell in sorted(cells, key=_cell_key):
        dual, stratum, visibility, g4 = _cell_key(cell)
        rows.append(
            {
                "schema": ROW_SCHEMA,
                "bucket_id": (
                    f"dual{dual}__{stratum}__{visibility}__{g4}"
                ),
                "atlas_key": {
                    "dual_index": dual,
                    "stratum": stratum,
                    "scorer_visibility": visibility,
                    "g4_temporal_class": g4,
                },
                "assignment_status": UNALLOCATED_STATUS,
                "pair_ids": [],
                "receiver_actuator_ids": [],
                "measured_probe_assignments": [],
                "byte_runs": [],
                "assigned_counted_bytes": 0,
                "derivation_method": "EXACT_CONSTRUCTION_LINEAGE_REFUSAL",
                "same_object": True,
                "unallocated_reason": (
                    "No exact final C1 byte run owns this pair/cell identity."
                ),
            }
        )
    table = {
        "schema": TABLE_SCHEMA,
        "loader_lineage": {
            "source_schema": MS5_TABLE_SCHEMA,
            "source_sha256": ms5_sha256,
            "method": (
                "MS5-shaped atlas_key/assignment_status/pair_ids/"
                "receiver_actuator_ids/measured_probe_assignments rows"
            ),
            "existing_ms5_validator_claimed_compatible": False,
            "reason": (
                "EV2 is a 162-cell accounting cube, not PF2's 1200-row atlas; "
                "the causal-assignment row law is reused without schema spoofing."
            ),
        },
        "cell_count": CELL_COUNT,
        "rows": rows,
        "pair_rows": [
            {
                "schema": PAIR_SCHEMA,
                "source_pair_id": pair_id,
                "allocated_counted_bytes": 0,
                "exclusive_archive_section_counted_bytes": 0,
                "exclusive_archive_section_present": False,
                "cell_ids": [],
                "assignment_status": UNALLOCATED_STATUS,
                "derivation_method": "EXACT_CONSTRUCTION_LINEAGE_REFUSAL",
            }
            for pair_id in range(PAIR_COUNT)
        ],
        "coarse_lawful_partition": {
            "schema": "ddm_ev2_coarse_stream_partition.v1",
            "rows": list(coarse_rows),
            "counted_bytes": MEASURED_C1_BYTES,
            "partition_level": "LP1_TYPED_STREAM_HOME",
        },
        "mass_conservation": {
            "lp1_measured_bytes": MEASURED_C1_BYTES,
            "assigned_pair_cell_bytes": 0,
            "unallocated_bytes": MEASURED_C1_BYTES,
            "separable_fraction": 0.0,
            "unallocated_fraction": 1.0,
            "conserved": True,
        },
        "falsifier": {
            "threshold_unallocated_fraction": FALSIFIER_UNALLOCATED_FRACTION,
            "observed_unallocated_fraction": 1.0,
            "fired": True,
            "verdict": "FORMULATION_MISPOSED_FOR_CURRENT_C1_COMPOSITION",
            "verdict_scope": "FORMULATION",
        },
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        "main_landing_review_required": True,
    }
    table["table_content_sha256"] = canonical_sha256(table)
    return table


def _ms5_loader_table(
    source: Mapping[str, Any],
    *,
    allocation_table: Mapping[str, Any],
) -> dict[str, Any]:
    """Emit a strict MS5-loadable projection without pretending 162 == 1,200."""

    table = deepcopy(dict(source))
    table["ev2_rate_home_extension"] = {
        "schema": "ddm_ev2_ms5_rate_home_extension.v1",
        "allocation_table_schema": TABLE_SCHEMA,
        "allocation_table_content_sha256": allocation_table[
            "table_content_sha256"
        ],
        "cell_count": CELL_COUNT,
        "pair_count": PAIR_COUNT,
        "assigned_counted_bytes": 0,
        "unallocated_counted_bytes": MEASURED_C1_BYTES,
        "assignment_status": UNALLOCATED_STATUS,
        "derivation_method": "EXACT_CONSTRUCTION_LINEAGE_REFUSAL",
        "score_claim": False,
    }
    table.pop("table_content_sha256", None)
    table["table_content_sha256"] = ms5_canonical_sha256(table)
    expected_pf2_sha256 = table.get("pf2_receipt_sha256")
    if not isinstance(expected_pf2_sha256, str):
        raise EV2AllocationError("MS5 source PF2 binding is absent")
    validate_assignment_table(
        table,
        expected_pf2_sha256=expected_pf2_sha256,
    )
    return table


def _backfill(
    *,
    ev1_cells: Sequence[Mapping[str, Any]],
    rd1_rows: Mapping[tuple[int, str, str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    cells = []
    for ev1 in sorted(ev1_cells, key=_cell_key):
        key = _cell_key(ev1)
        rd1 = rd1_rows.get(key)
        if rd1 is None:
            raise EV2AllocationError("EV1 and RD1 cell identities differ")
        cells.append(
            {
                **dict(zip(CELL_KEY_FIELDS, key, strict=True)),
                "left_candidate_id": rd1.get("left_candidate_id"),
                "right_candidate_id": rd1.get("right_candidate_id"),
                "effective_quantum_D": rd1.get("effective_quantum_D"),
                "ev1_delta_D_dimension": ev1.get("delta_D_dimension"),
                "ev1_accounting_bytes_dimension": ev1.get(
                    "delta_counted_bytes_dimension"
                ),
                "c1_allocated_bytes_dimension": 0,
                "lambda_bytes_per_D_dimension": None,
                "costate_status": (
                    "STILL_NULL_FORMULATION_MISPOSED_NO_EXCLUSIVE_C1_CELL_HOME"
                ),
                "actionable_for_train_decision": False,
                "score_claim": False,
            }
        )
    return {
        "schema": BACKFILL_SCHEMA,
        "source_cell_count": CELL_COUNT,
        "computable_cell_count": 0,
        "still_null_cell_count": CELL_COUNT,
        "cells": cells,
        "verdict": "NO_BACKFILL_PERFORMED; 162_OF_162_STILL_NULL",
        "verdict_scope": "FORMULATION",
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
    }


def _headline_replay(
    r3: Mapping[str, Any],
    *,
    bundle_path: Path,
    repository_root: Path,
) -> dict[str, Any]:
    if r3.get("schema") != "ddm_ms2r_r3_box_tolerance_solve_receipt.v1":
        raise EV2AllocationError("R3 receipt schema differs")
    _false_authority(r3.get("authority", {}), "R3")
    source = r3.get("minimum_description_headline")
    if not isinstance(source, Mapping):
        raise EV2AllocationError("R3 minimum-description headline is absent")
    declarations = source["recursive_solve_typing"]["declarations"]
    replay = build_minimum_description_headline(
        stored_problem_bytes=source["stored_problem"]["bytes"],
        stored_problem_sha256=source["stored_problem"]["sha256"],
        exception_bytes=source["solve_mandated_exceptions"]["bytes"],
        exception_sha256=source["solve_mandated_exceptions"]["sha256"],
        realized_d_seg=source["diagnostic_distortions"]["realized_d_seg"],
        realized_d_pose=source["diagnostic_distortions"]["realized_d_pose"],
        stored_problem_own_lineage=source["stored_problem"]["own_lineage"],
        donor_conditioned=source["donor_conditioned"],
        expansion_receiver_closed=source["stored_problem"][
            "receiver_expansion_closed"
        ],
        pose_tube_active=source["joint_constraints"]["pose_tube_active"],
        realized_uint8_r_frozen_scorers=source["joint_constraints"][
            "realized_uint8_r_frozen_scorers"
        ],
        quotient_coordinates_only=declarations["quotient_coordinates_only"],
        scorer_metric_active=declarations["scorer_metric_active"],
        alternating_typed_subproblems=declarations[
            "alternating_typed_subproblems"
        ],
        typed_blocks_active=declarations["typed_blocks_active"],
        per_dimension_quanta_active=declarations[
            "per_dimension_quanta_active"
        ],
        typed_stream_tags=source["typed_stream_custody"]["tags"],
        strict_typed_stream_tags=True,
        metric_custody_bundle_path=bundle_path,
        metric_custody_repository_root=repository_root,
    )
    blockers = tuple(replay.get("blockers", ()))
    if blockers != EXPECTED_HEADLINE_BLOCKERS:
        raise EV2AllocationError("headline replay blocker set differs")
    return {
        "schema": HEADLINE_SCHEMA,
        "builder": (
            "tac.optimization.ddm_min_description_contract:"
            "build_minimum_description_headline"
        ),
        "source_status": source["status"],
        "replay_status": replay["status"],
        "edge_cleared_blockers": [],
        "remaining_blockers": list(blockers),
        "exact_remaining_set_unchanged": True,
        "reason": (
            "EV2 measured zero lawful pair-cell bytes, so the edge does not "
            "activate a typed block atlas or per-dimension quanta."
        ),
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
    }


def build_ev2_allocation(
    *,
    c1_archive: bytes,
    lp1: Mapping[str, Any],
    ev1: Mapping[str, Any],
    rd1: Mapping[str, Any],
    r3: Mapping[str, Any],
    ms5: Mapping[str, Any],
    ms5_sha256: str,
    bundle_path: Path,
    repository_root: Path,
) -> EV2BuildResult:
    """Build the measured falsifier receipt and its consumer-facing tables."""

    if len(c1_archive) != ARCHIVE_BYTES:
        raise EV2AllocationError("C1 archive byte length differs")
    lp1_rows = _lp1_rows(lp1)
    outer_rows, members = _zip_home_rows(c1_archive)
    joint_coder = _joint_coder_custody(members)
    coarse_rows = _coarse_home_rows(lp1_rows=lp1_rows, outer_rows=outer_rows)
    ev1_cells, ev1_conservation = _ev1_cells(ev1)
    rd1_rows = _rd1_rows(rd1)
    table = _allocation_table(
        ev1_cells,
        coarse_rows,
        ms5=ms5,
        ms5_sha256=ms5_sha256,
    )
    ms5_loader_table = _ms5_loader_table(ms5, allocation_table=table)
    backfill = _backfill(ev1_cells=ev1_cells, rd1_rows=rd1_rows)
    headline = _headline_replay(
        r3,
        bundle_path=bundle_path,
        repository_root=repository_root,
    )
    receipt = {
        "schema": SCHEMA,
        "verdict": "FORMULATION_MISPOSED_FOR_CURRENT_C1_COMPOSITION",
        "verdict_scope": "FORMULATION",
        "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer": "0.1910828242 [contest-CPU]",
        "pointer_moved": False,
        "main_landing_review_required": True,
        "derivation_method": "EXACT_CONSTRUCTION_LINEAGE",
        "receiver_probe_status": (
            "NOT_RUN_CONSTRUCTION_LINEAGE_ALREADY_PROVES_NO_EXCLUSIVE_FINAL_BYTE_RUN"
        ),
        "same_object_firewall": {
            "c1_allocation_object": "v15 exact control plus LP1 lane seed",
            "ev1_accounting_object": "three RD1 endpoint-edge deltas",
            "objects_identical": False,
            "cross_object_byte_smearing": 0,
        },
        "joint_coder_custody": joint_coder,
        "mass_conservation": table["mass_conservation"],
        "lp1_coarse_lawful_partition": table["coarse_lawful_partition"],
        "ev1_accounting_home_conservation": ev1_conservation,
        "falsifier": table["falsifier"],
        "per_pair_allocation": {
            "status": "REFUSED_FORMULATION_MISPOSED",
            "row_count": PAIR_COUNT,
            "assigned_counted_bytes": 0,
            "unallocated_counted_bytes": MEASURED_C1_BYTES,
        },
        "allocation_table_content_sha256": table["table_content_sha256"],
        "ms5_loader_projection": {
            "schema": ms5_loader_table["schema"],
            "table_content_sha256": ms5_loader_table[
                "table_content_sha256"
            ],
            "strict_validator_passed": True,
            "source_row_count": len(ms5_loader_table["rows"]),
            "ev2_extension_cell_count": CELL_COUNT,
        },
        "rd1_dual_backfill": {
            "computable_cell_count": backfill["computable_cell_count"],
            "still_null_cell_count": backfill["still_null_cell_count"],
            "verdict": backfill["verdict"],
        },
        "headline_replay": headline,
        "waterfill": {
            "preflight_eligible_from_ms3_bundle": True,
            "full_solve_allowed": False,
            "reason": (
                "The bundle is complete, but the EV2 attribution falsifier "
                "leaves every per-dimension price null."
            ),
        },
        "stores_consulted": [
            "LP1 measured C1 typed-home receipt",
            "EV1 600-pair and 162-cell accounting receipt",
            "RD1 162-cell null-dual source",
            "MS5 causal assignment loader lineage",
            "MS6/RG1-RG3 probe methodology",
            "R3 BOX headline receipt",
            "MS3 BUNDLE-COMPLETE loader",
        ],
    }
    receipt["receipt_content_sha256"] = canonical_sha256(receipt)
    return EV2BuildResult(
        allocation_table=table,
        ms5_loader_table=ms5_loader_table,
        rd1_backfill=backfill,
        headline_replay=headline,
        receipt=receipt,
    )


__all__ = [
    "ARCHIVE_BYTES",
    "BACKFILL_SCHEMA",
    "CELL_COUNT",
    "COARSE_HOME_SCHEMA",
    "FALSIFIER_UNALLOCATED_FRACTION",
    "HEADLINE_SCHEMA",
    "MEASURED_C1_BYTES",
    "PAIR_COUNT",
    "ROW_SCHEMA",
    "SCHEMA",
    "TABLE_SCHEMA",
    "UNALLOCATED_STATUS",
    "EV2AllocationError",
    "EV2BuildResult",
    "build_ev2_allocation",
    "canonical_bytes",
    "canonical_sha256",
]
