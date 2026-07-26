# SPDX-License-Identifier: MIT
"""Canonicalize prior original-work signal for the bounded G1 compiler.

This module is an evidence bridge, not a candidate codec.  It joins the exact
PBR2 encoder-side teacher with the already-measured V10 lattice and
V13/V14/V19c direct-description campaigns so a successor does not rediscover
settled primitive families or silently import target truth into candidate
bytes.

The output contains measurements and acquisition priors only.  PBR packets,
target labels, scorer artifacts, and dense planes remain encoder-only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import stat
import subprocess
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from tac.witness_dsl.progressive_geometry_residual import (
    ProgressiveGeometryResidualError,
)
from tac.witness_dsl.progressive_geometry_residual import (
    packet_accounting as pbr2_packet_accounting,
)

SCHEMA = "tac.g1_prior_signal_harvest.v1"
ENVELOPE_SCHEMA = "tac.g1_prior_signal_harvest_receipt.v1"

PBR2_SCHEMA = "tac.c0b_pbr2_progressive_geometry_measurement.v2"
V13_SCHEMA = "direct_description_v13_worldsheet_predictor_receipt.v1"
V14_SCHEMA = "ddm_v14_realization_fidelity_receipt.v1"
V19C_SCHEMA = "ddm_v19c_correction_saturation_receipt.v1"
V10_LATTICE_SCHEMA = "v10_uint8_lattice_feasibility_receipt.v1"
CENSUS_ENVELOPE_SCHEMA = "tac.v9_target_partition_grammar_census_receipt.v1"
CENSUS_BODY_SCHEMA = "tac.v9_target_partition_grammar_census.v1"

DEFAULT_PBR2 = Path(
    ".omx/research/original_taskspace_inverse_witness_codec_20260725/c0b_pbr2_progressive_geometry_n64.json"
)
DEFAULT_PBR2_PACKET = Path(
    ".omx/research/original_taskspace_inverse_witness_codec_20260725/c0b_pbr2_progressive_geometry_n64.pbr2"
)
DEFAULT_V13 = Path(
    ".omx/research/ddm_v13_g1_worldsheet_predictor_n600_20260722T201500Z/"
    "ddm_v13_worldsheet_predictor_n600_receipt_v2.json"
)
DEFAULT_V14 = Path(
    ".omx/research/ddm_v14_realization_fidelity_n600_20260722T215500Z/ddm_v14_realization_fidelity_n600_receipt.json"
)
DEFAULT_V19C = Path(
    ".omx/research/ddm_v19c_correction_saturation_20260723T063500Z/ddm_v19c_correction_saturation_receipt.json"
)
DEFAULT_V10_LATTICE = Path(".omx/research/v10_uint8_lattice_feasibility_receipt_20260718.json")
DEFAULT_CENSUS = Path(
    ".omx/research/original_taskspace_inverse_witness_codec_20260725/v9_target_partition_grammar_census_n600.json"
)
DEFAULT_OUTPUT = Path(".omx/research/original_taskspace_inverse_witness_codec_20260725/g1_prior_signal_harvest_v1.json")

_REPO_ROOT = Path(__file__).resolve().parents[3]

_MAX_RECEIPT_BYTES = 64 << 20
_SHA256_HEX_LEN = 64
_ADVISORY_AXIS = "[macOS-CPU frozen-scorer advisory]"
_V10_AXIS = "[macOS-CPU advisory subset]"
_V10_VERDICT_SCOPE = (
    "selected real n600-cache pairs, frozen CPU SegNet, uint8 frame1/A factor only; "
    "no PoseNet, receiver archive, contest CPU/CUDA, or full-n600 claim"
)
_PBR2_ACCOUNTING_FIELDS = frozenset(
    {
        "schema",
        "packet_bytes",
        "packet_sha256",
        "packet_prefix_header_bytes",
        "header_bytes",
        "crc_bytes",
        "strata",
        "initial_error_cells",
        "final_error_cells",
        "source_pair_start",
        "source_pair_stop_exclusive",
        "predictor_program_sha256",
        "predictor_semantic_sha256",
        "target_semantic_sha256",
        "separate_dense_target_table_section_bytes",
        "pbr2_is_target_derived",
        "pbr2_target_derived_section_bytes",
        "pbr2_event_count",
        "pbr2_event_density_numerator",
        "pbr2_event_density_denominator",
        "target_derived_residual_promotion_admitted",
        "research_only",
        "artifact_role",
        "candidate_archive_admissible",
        "exact_target_semantic_reconstruction",
        "target_semantic_lineage",
        "pbr2_reconstructs_exact_gt_argmax",
        "reconstructed_target_semantic_bytes",
        "candidate_archive_blocker",
        "generic_apply_requires_external_predictor_semantics",
        "physical_prefix_decode_supported",
        "staged_application_requires_complete_packet",
        "decode_scorer_dependency",
        "score_claim",
        "promotion_eligible",
    }
)
_BODY_FIELDS = frozenset(
    {
        "schema",
        "research_only",
        "score_claim",
        "promotion_eligible",
        "candidate_payload_emitted",
        "pointer_moved",
        "input_custody",
        "producer_custody",
        "semantic_argv",
        "frozen_target_cache_join",
        "teacher",
        "target_partition_prior",
        "worldsheet_prior",
        "realization_prior",
        "inverse_preimage_prior",
        "correction_saturation_prior",
        "composition_decision",
        "verdict",
    }
)


class G1SignalHarvestError(ValueError):
    """Raised when prior evidence or its authority boundary is malformed."""


def canonical_json_bytes(value: Any) -> bytes:
    """Return the repository's compact, deterministic JSON representation."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _exact_dict(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise G1SignalHarvestError(f"{label} must be a JSON object")
    return value


def _exact_list(value: Any, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise G1SignalHarvestError(f"{label} must be a JSON list")
    return value


def _require_schema(document: Mapping[str, Any], expected: str, label: str) -> None:
    if document.get("schema") != expected:
        raise G1SignalHarvestError(f"{label} schema must be {expected!r}; got {document.get('schema')!r}")


def _require_research_only(
    document: Mapping[str, Any],
    label: str,
    *,
    expected_axis: str | None = None,
) -> None:
    if document.get("research_only") is not True:
        raise G1SignalHarvestError(f"{label} must be research_only=true")
    if document.get("score_claim") is not False:
        raise G1SignalHarvestError(f"{label} must have score_claim=false")
    if document.get("promotion_eligible") is not False:
        raise G1SignalHarvestError(f"{label} must have promotion_eligible=false")
    if "pointer_moved" in document and document.get("pointer_moved") is not False:
        raise G1SignalHarvestError(f"{label} must have pointer_moved=false")
    if "execution_allowed" in document and document.get("execution_allowed") is not False:
        raise G1SignalHarvestError(f"{label} must have execution_allowed=false")
    if expected_axis is not None and document.get("evidence_axis") != expected_axis:
        raise G1SignalHarvestError(f"{label} evidence_axis must be {expected_axis!r}")


def _require_v10_false_authority(document: Mapping[str, Any]) -> None:
    authority = _exact_dict(document.get("authority"), "V10 lattice authority")
    required_false = (
        "score_claim",
        "pointer_moved",
        "promotion_eligible",
        "rank_or_kill_eligible",
        "ready_for_exact_eval_dispatch",
    )
    for field in required_false:
        if authority.get(field) is not False:
            raise G1SignalHarvestError(f"V10 lattice authority must have {field}=false")
    if authority.get("subset_non_promotable") is not True:
        raise G1SignalHarvestError("V10 lattice authority must remain subset_non_promotable=true")
    if authority.get("verdict_scope") != _V10_VERDICT_SCOPE:
        raise G1SignalHarvestError(f"V10 lattice verdict_scope must be {_V10_VERDICT_SCOPE!r}")
    if document.get("axis") != _V10_AXIS:
        raise G1SignalHarvestError(f"V10 lattice axis must be {_V10_AXIS!r}")


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_HEX_LEN
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_git_sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(character in "0123456789abcdef" for character in value)


def _positive_int(value: Any, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise G1SignalHarvestError(f"{label} must be a positive integer")
    return value


def _nonnegative_int(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise G1SignalHarvestError(f"{label} must be a nonnegative exact integer")
    return value


def _exact_int_list(value: Any, expected: list[int], label: str) -> list[int]:
    values = _exact_list(value, label)
    if values != expected or any(type(item) is not int for item in values):
        raise G1SignalHarvestError(f"{label} must be the exact integer list {expected!r}")
    return values


def _zero_number(value: Any, label: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)) or float(value) != 0.0:
        raise G1SignalHarvestError(f"{label} must be an exact finite numeric zero")
    return float(value)


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(_REPO_ROOT))
    except ValueError:
        return str(resolved)


def _read_regular_bytes(
    path: Path,
    *,
    max_bytes: int = _MAX_RECEIPT_BYTES,
) -> tuple[bytes, dict[str, Any]]:
    supplied = Path(os.path.abspath(path))
    supplied_stat = supplied.lstat()
    if stat.S_ISLNK(supplied_stat.st_mode) or not stat.S_ISREG(supplied_stat.st_mode):
        raise G1SignalHarvestError(f"input must be a non-symlink regular file: {supplied}")
    descriptor = os.open(supplied, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        before = os.fstat(descriptor)
        if before.st_size > max_bytes:
            raise G1SignalHarvestError(f"input exceeds {max_bytes} byte ceiling: {supplied}")
        with os.fdopen(descriptor, "rb", closefd=False) as handle:
            payload = handle.read(max_bytes + 1)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    path_after = supplied.lstat()
    before_identity = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
    after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    path_identity = (
        path_after.st_dev,
        path_after.st_ino,
        path_after.st_size,
        path_after.st_mtime_ns,
        path_after.st_ctime_ns,
    )
    if before_identity != after_identity or after_identity != path_identity or len(payload) != after.st_size:
        raise G1SignalHarvestError(f"input mutated while read: {supplied}")
    return payload, {"path": _display_path(supplied), "bytes": len(payload), "sha256": _sha256(payload)}


def _read_regular_bytes_custody(path: Path, *, max_bytes: int = _MAX_RECEIPT_BYTES) -> dict[str, Any]:
    return _read_regular_bytes(path, max_bytes=max_bytes)[1]


def read_json_with_custody(path: Path, *, expected_schema: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Read one stable regular JSON file and bind its exact bytes."""

    payload, custody = _read_regular_bytes(path, max_bytes=_MAX_RECEIPT_BYTES)
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise G1SignalHarvestError(f"input is not valid JSON: {path}") from exc
    document = dict(_exact_dict(value, str(path)))
    _require_schema(document, expected_schema, str(path))
    return document, {**custody, "schema": expected_schema}


def _selected_row(
    rows: list[Any],
    candidate: str,
    label: str,
    *,
    selector_key: str = "candidate",
) -> Mapping[str, Any]:
    matches = [
        _exact_dict(row, f"{label} row")
        for row in rows
        if isinstance(row, Mapping) and row.get(selector_key) == candidate
    ]
    if len(matches) != 1:
        raise G1SignalHarvestError(f"{label} must contain exactly one {selector_key}={candidate!r} row")
    return matches[0]


def _worldsheet_signal(v13: Mapping[str, Any]) -> dict[str, Any]:
    selected = str(v13.get("selected_rung"))
    selected_row = _selected_row(
        _exact_list(v13.get("composition_ladder"), "v13 composition_ladder"),
        selected,
        "v13 composition_ladder",
        selector_key="rung",
    )
    inventory = _exact_dict(v13.get("natural_production_inventory"), "v13 inventory")
    movable = _exact_dict(inventory.get("movable_g1"), "v13 movable_g1")
    bridge = _exact_dict(selected_row.get("bridge"), "v13 selected bridge")
    segmentation = _exact_dict(bridge.get("segmentation"), "v13 segmentation")
    pose = _exact_dict(bridge.get("pose"), "v13 pose")
    archive = _exact_dict(selected_row.get("archive"), "v13 archive")
    return {
        "selected_rung": selected,
        "archive_bytes": archive.get("bytes"),
        "archive_sha256": archive.get("sha256"),
        "measured_advisory_d_seg": segmentation.get("d_seg"),
        "measured_advisory_d_pose": pose.get("d_pose"),
        "movable_worldsheet": {
            "payload_bytes": movable.get("payload_bytes"),
            "payload_sha256": movable.get("payload_sha256"),
            "decoded_mask_errors": movable.get("decoded_mask_errors"),
            "decoded_clean_rest_dseg": movable.get("decoded_clean_rest_dseg"),
            "births": movable.get("births"),
            "deaths": movable.get("deaths"),
            "persists": movable.get("persists"),
            "vertices": movable.get("vertices"),
        },
        "binding_mechanism": _exact_dict(v13.get("falsifier"), "v13 falsifier").get("binding_mechanism"),
        "authority": v13.get("evidence_axis"),
    }


def _realization_signal(v14: Mapping[str, Any], *, pair_start: int, pair_stop: int) -> dict[str, Any]:
    selected = str(v14.get("selected_candidate"))
    row = _selected_row(
        _exact_list(v14.get("fixed_ladder"), "v14 fixed_ladder"),
        selected,
        "v14 fixed_ladder",
    )
    diagnostics = _exact_dict(v14.get("diagnostics"), "v14 diagnostics")
    anchors: list[dict[str, Any]] = []
    for raw in _exact_list(diagnostics.get("lane_windows"), "v14 lane_windows"):
        item = _exact_dict(raw, "v14 lane window")
        source_pair_id = item.get("source_pair_id")
        local_pair_id = item.get("local_pair_id")
        if not isinstance(source_pair_id, int) or isinstance(source_pair_id, bool):
            raise G1SignalHarvestError("v14 source_pair_id must be an integer")
        if not isinstance(local_pair_id, int) or isinstance(local_pair_id, bool):
            raise G1SignalHarvestError("v14 local_pair_id must be an integer")
        if pair_start <= source_pair_id < pair_stop:
            anchors.append(
                {
                    "source_pair_id": source_pair_id,
                    "local_pair_id": local_pair_id,
                    "delta_d_seg": item.get("delta_d_seg"),
                    "fixed_islands_d_seg": item.get("fixed_islands_d_seg"),
                    "fixed_both_d_seg": item.get("fixed_both_d_seg"),
                }
            )
    anchors.sort(key=lambda value: int(value["source_pair_id"]))
    if not anchors:
        raise G1SignalHarvestError("v14 has no measured anchor inside the PBR2 window")
    return {
        "selected_candidate": selected,
        "archive_bytes": row.get("archive_bytes"),
        "measured_advisory_d_seg": row.get("d_seg"),
        "measured_advisory_d_pose": row.get("d_pose"),
        "measured_mechanism": diagnostics.get("measured_mechanism"),
        "pbr2_window_anchor_rows": anchors,
        "anchor_rows_are_family_claim": False,
        "authority": v14.get("evidence_axis"),
    }


def _v19c_signal(v19c: Mapping[str, Any]) -> dict[str, Any]:
    curve = _exact_dict(v19c.get("curve"), "v19c curve")
    endpoint = _exact_dict(curve.get("n600_endpoint"), "v19c n600 endpoint")
    families = _exact_dict(v19c.get("family_attribution"), "v19c family attribution")
    n600 = _exact_dict(families.get("n600"), "v19c n600 family attribution")
    rows: list[dict[str, Any]] = []
    for family, raw in n600.items():
        row = _exact_dict(raw, f"v19c family {family}")
        rows.append(
            {
                "family": family,
                "admitted": row.get("admitted"),
                "compile_infeasible": row.get("compile_infeasible"),
                "measured_or_classified": row.get("proposals_measured_or_classified"),
                "strict_joint_gain": row.get("strict_joint_gain"),
            }
        )
    rows.sort(key=lambda value: (-float(value["strict_joint_gain"]), str(value["family"])))
    asymptote = _exact_dict(v19c.get("asymptote"), "v19c asymptote")
    bucket = _exact_dict(v19c.get("c1_bucket_attribution"), "v19c c1 bucket")
    flips = _exact_dict(
        bucket.get("v19c_incremental_realized_net_flips"),
        "v19c incremental flips",
    )
    return {
        "endpoint": {
            "archive_bytes": endpoint.get("archive_bytes"),
            "archive_sha256": endpoint.get("archive_sha256"),
            "measured_advisory_d_seg": endpoint.get("d_seg"),
            "measured_advisory_d_pose": endpoint.get("d_pose"),
            "joint_delta_vs_v19b": endpoint.get("joint_delta_vs_v19b"),
        },
        "family_priors_ranked_by_measured_n600_strict_gain": rows,
        "finite_coordinate_certificate": {
            "unique_coordinate_inventory": asymptote.get("unique_coordinate_inventory"),
            "dev_proposals": asymptote.get("dev_proposals"),
            "dev_admissions": asymptote.get("dev_admissions"),
            "n600_admissions": asymptote.get("n600_admissions"),
            "consecutive_failures_at_stop": asymptote.get("consecutive_failures_at_stop"),
            "family_optimum_claimed": asymptote.get("family_optimum_claimed"),
        },
        "residual_bucket_fraction_of_incremental_flips": flips.get("residual_fraction"),
        "authority": v19c.get("evidence_axis"),
    }


def _v10_preimage_signal(v10: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the exact measured existence result without importing its dense sidecar."""

    _require_v10_false_authority(v10)
    aggregate = _exact_dict(v10.get("aggregate"), "V10 lattice aggregate")
    search = _exact_dict(aggregate.get("exact_search"), "V10 lattice exact search")
    arms = _exact_dict(aggregate.get("arms"), "V10 lattice arms")
    exact_arm = _exact_dict(
        arms.get("exact_uint8_lattice_candidate"),
        "V10 exact uint8 lattice arm",
    )
    configuration = _exact_dict(v10.get("configuration"), "V10 lattice configuration")
    pair_ids = _exact_list(configuration.get("pair_ids"), "V10 lattice pair_ids")
    if (
        len(pair_ids) != 6
        or len(set(pair_ids)) != 6
        or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in pair_ids)
    ):
        raise G1SignalHarvestError("V10 lattice pair_ids must be six unique nonnegative integers")
    scorer_hw = _exact_int_list(configuration.get("scorer_hw"), [384, 512], "V10 scorer_hw")
    camera_hw = _exact_int_list(configuration.get("camera_hw"), [874, 1164], "V10 camera_hw")
    frame_count = len(pair_ids)
    expected_pixels = frame_count * scorer_hw[0] * scorer_hw[1]
    expected_blocks = expected_pixels * 3
    exact_blocks = _nonnegative_int(search.get("exact_blocks"), "V10 exact_blocks")
    exact_candidate_blocks = _nonnegative_int(search.get("exact_candidate_blocks"), "V10 exact_candidate_blocks")
    certified_frames = _nonnegative_int(search.get("certified_exact_frames"), "V10 certified frames")
    decoded_frames = _nonnegative_int(
        search.get("decoded_frames_with_exact_numerator_equality"), "V10 decoded exact frames"
    )
    budget_blocks = _nonnegative_int(search.get("budget_blocks"), "V10 budget blocks")
    heuristic_blocks = _nonnegative_int(search.get("heuristic_blocks"), "V10 heuristic blocks")
    residual_cells = _nonnegative_int(
        search.get("nonzero_decoded_numerator_residual_cells"), "V10 decoded residual cells"
    )
    max_residual = _nonnegative_int(search.get("max_abs_decoded_numerator_residual"), "V10 maximum decoded residual")
    mismatched_pixels = _nonnegative_int(exact_arm.get("mismatched_pixels"), "V10 mismatched pixels")
    total_pixels = _positive_int(exact_arm.get("total_pixels"), "V10 total pixels")
    exact_d_seg = _zero_number(exact_arm.get("d_seg"), "V10 exact-arm d_seg")
    nodes_visited = _positive_int(search.get("nodes_visited"), "V10 nodes visited")
    if (
        search.get("aggregate_statuses") != ["FEASIBLE_EXACT"]
        or exact_blocks != expected_blocks
        or exact_candidate_blocks != exact_blocks
        or certified_frames != frame_count
        or decoded_frames != frame_count
        or budget_blocks != 0
        or heuristic_blocks != 0
        or residual_cells != 0
        or max_residual != 0
        or mismatched_pixels != 0
        or exact_d_seg != 0.0
        or total_pixels != expected_pixels
    ):
        raise G1SignalHarvestError("V10 lattice receipt does not prove the claimed exact measured subset")
    per_class = _exact_dict(exact_arm.get("per_class"), "V10 exact arm per-class rows")
    if set(per_class) != {"0", "1", "2", "3", "4"}:
        raise G1SignalHarvestError("V10 exact arm must contain the five exact class rows")
    target_pixel_sum = 0
    for class_id, raw in per_class.items():
        row = _exact_dict(raw, f"V10 exact arm class {class_id}")
        target_pixel_sum += _positive_int(row.get("target_pixels"), f"V10 class {class_id} target_pixels")
        class_mismatches = _nonnegative_int(row.get("mismatched_pixels"), f"V10 class {class_id} mismatched pixels")
        class_d_seg = _zero_number(row.get("d_seg"), f"V10 class {class_id} d_seg")
        if class_mismatches != 0 or class_d_seg != 0.0:
            raise G1SignalHarvestError("V10 exact arm per-class hard-Seg closure differs")
    if target_pixel_sum != expected_pixels:
        raise G1SignalHarvestError("V10 exact arm per-class target pixels do not close")
    sidecar = _exact_dict(v10.get("sidecar"), "V10 lattice sidecar")
    sidecar_frame_count = _nonnegative_int(sidecar.get("frame_count"), "V10 sidecar frame_count")
    if (
        sidecar.get("parse_back_all_frame_hashes_match") is not True
        or sidecar_frame_count != frame_count
        or not isinstance(sidecar.get("bytes"), int)
        or isinstance(sidecar.get("bytes"), bool)
        or sidecar.get("bytes") <= 0
        or not _is_sha256(sidecar.get("sha256"))
        or sidecar.get("honest_name") != "incremental uint8 lattice feasibility sidecar; NOT a contest archive"
        or sidecar.get("candidate_payload_allowed") not in (None, False)
    ):
        raise G1SignalHarvestError("V10 lattice sidecar parse-back is not closed")
    blockers = _exact_list(v10.get("remaining_blockers"), "V10 lattice blockers")
    return {
        "measured_pair_ids": pair_ids,
        "camera_hw": camera_hw,
        "scorer_hw": scorer_hw,
        "certified_exact_frames": certified_frames,
        "decoded_frames_with_exact_numerator_equality": decoded_frames,
        "exact_blocks": exact_blocks,
        "nodes_visited": nodes_visited,
        "max_abs_decoded_numerator_residual": max_residual,
        "measured_hard_seg_mismatches": mismatched_pixels,
        "measured_advisory_d_seg": exact_d_seg,
        "dense_measurement_sidecar": {
            "bytes": sidecar.get("bytes"),
            "sha256": sidecar.get("sha256"),
            "frame_count": sidecar_frame_count,
            "candidate_payload_allowed_by_harvest": False,
            "source_candidate_payload_allowed": sidecar.get("candidate_payload_allowed"),
            "role": "encoder_only_existence_and_solver_evidence",
        },
        "remaining_blockers": blockers,
        "derived_composition_consequence": {
            "exact_preimage_existence": "measured_for_frame1_on_selected_six_pair_subset_only",
            "generic_receiver_operation": "canonical_factor2_uint8_support_fill",
            "counted_object": "compact_Y0_Y1_obligation_generator_program",
            "dense_y_or_camera_preimage_payload": False,
            "independent_frame0_pose_still_required": True,
        },
        "authority": _V10_VERDICT_SCOPE,
        "evidence_axis": _V10_AXIS,
    }


def _validate_pbr2_teacher(
    pbr2: Mapping[str, Any],
    pbr2_row: Mapping[str, Any],
    input_custody: Mapping[str, Mapping[str, Any]],
    packet_accounting: Mapping[str, Any],
) -> tuple[int, int, list[dict[str, Any]]]:
    """Close exact frozen-teacher lineage and staged byte/error arithmetic."""

    required = {
        "pbr2_is_target_derived": True,
        "candidate_archive_admissible": False,
        "exact_target_semantic_reconstruction": True,
        "pbr2_reconstructs_exact_gt_argmax": True,
        "target_derived_residual_promotion_admitted": False,
        "promotion_eligible": False,
        "research_only": True,
        "score_claim": False,
        "decode_scorer_dependency": False,
    }
    if any(pbr2_row.get(key) is not expected for key, expected in required.items()):
        raise G1SignalHarvestError("PBR2 target-derived lineage is not closed")
    if pbr2_row.get("target_semantic_lineage") != "frozen_gt_argmax":
        raise G1SignalHarvestError("PBR2 target_semantic_lineage must be frozen_gt_argmax")
    receiver_closure = _exact_dict(pbr2.get("receiver_closure"), "PBR2 receiver closure")
    if (
        receiver_closure.get("candidate_payload_allowed") is not False
        or receiver_closure.get("exact_target_recovered_without_gt_cache_at_decode") is not True
        or receiver_closure.get("predictor_semantics_rederived_from_counted_program") is not True
    ):
        raise G1SignalHarvestError("PBR2 receiver closure is not exact and candidate-forbidden")

    pair_start = pbr2_row.get("source_pair_start")
    pair_stop = pbr2_row.get("source_pair_stop_exclusive")
    if (
        not isinstance(pair_start, int)
        or isinstance(pair_start, bool)
        or not isinstance(pair_stop, int)
        or isinstance(pair_stop, bool)
        or pair_start < 0
        or pair_stop <= pair_start
    ):
        raise G1SignalHarvestError("PBR2 source-pair window is invalid")
    difference = _exact_dict(pbr2.get("exact_difference"), "PBR2 difference")
    mismatch = _positive_int(difference.get("mismatch_cells"), "PBR2 mismatch_cells")
    total = _positive_int(difference.get("total_cells"), "PBR2 total_cells")
    if mismatch > total:
        raise G1SignalHarvestError("PBR2 mismatch cells exceed total cells")
    fraction = difference.get("mismatch_fraction")
    if (
        not isinstance(fraction, (int, float))
        or isinstance(fraction, bool)
        or not math.isclose(float(fraction), mismatch / total, rel_tol=0.0, abs_tol=1e-15)
    ):
        raise G1SignalHarvestError("PBR2 mismatch fraction arithmetic differs")
    if (
        pbr2_row.get("initial_error_cells") != mismatch
        or pbr2_row.get("final_error_cells") != 0
        or pbr2_row.get("pbr2_event_count") != mismatch
        or pbr2_row.get("pbr2_event_density_numerator") != mismatch
        or pbr2_row.get("pbr2_event_density_denominator") != total
        or pbr2_row.get("reconstructed_target_semantic_bytes") != total
    ):
        raise G1SignalHarvestError("PBR2 difference/header arithmetic differs")

    raw_strata = _exact_list(pbr2_row.get("strata"), "PBR2 strata")
    if len(raw_strata) != 3:
        raise G1SignalHarvestError("PBR2 must contain exactly three measured teacher strata")
    expected_before = mismatch
    payload_sum = 0
    strata: list[dict[str, Any]] = []
    for order, raw in enumerate(raw_strata, start=1):
        row = dict(_exact_dict(raw, "PBR2 stratum"))
        before = row.get("errors_before")
        after = row.get("errors_after")
        corrected = row.get("corrected_cells")
        payload_bytes = row.get("payload_bytes")
        if (
            row.get("order") != order
            or before != expected_before
            or not isinstance(after, int)
            or isinstance(after, bool)
            or after < 0
            or corrected != before - after
            or not isinstance(payload_bytes, int)
            or isinstance(payload_bytes, bool)
            or payload_bytes <= 0
            or not _is_sha256(row.get("payload_sha256"))
        ):
            raise G1SignalHarvestError("PBR2 stratum error/byte continuity differs")
        expected_before = after
        payload_sum += payload_bytes
        strata.append(row)
    if expected_before != 0 or sum(int(row["corrected_cells"]) for row in strata) != mismatch:
        raise G1SignalHarvestError("PBR2 staged correction cells do not close")
    prefix = pbr2_row.get("packet_prefix_header_bytes")
    header = pbr2_row.get("header_bytes")
    crc = pbr2_row.get("crc_bytes")
    packet_bytes = pbr2_row.get("packet_bytes")
    if (
        payload_sum != pbr2_row.get("pbr2_target_derived_section_bytes")
        or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in (prefix, header, crc))
        or packet_bytes != prefix + header + payload_sum + crc
        or not _is_sha256(pbr2_row.get("packet_sha256"))
    ):
        raise G1SignalHarvestError("PBR2 packet byte ownership does not close")
    packet_custody = _exact_dict(input_custody.get("pbr2_packet"), "PBR2 packet custody")
    if packet_custody.get("bytes") != packet_bytes or packet_custody.get("sha256") != pbr2_row.get("packet_sha256"):
        raise G1SignalHarvestError("PBR2 packet identity does not match exact input bytes")
    if frozenset(packet_accounting) != _PBR2_ACCOUNTING_FIELDS:
        raise G1SignalHarvestError("PBR2 strict packet accounting fields differ")
    for field in _PBR2_ACCOUNTING_FIELDS:
        if canonical_json_bytes(packet_accounting.get(field)) != canonical_json_bytes(pbr2_row.get(field)):
            raise G1SignalHarvestError(f"PBR2 receipt field {field!r} differs from strict packet contents")
    return pair_start, pair_stop, strata


def _target_cache_join(
    *,
    pbr2: Mapping[str, Any],
    census_body: Mapping[str, Any],
    v10_lattice: Mapping[str, Any],
    v13: Mapping[str, Any],
    v14: Mapping[str, Any],
) -> dict[str, Any]:
    pbr2_gt = _exact_dict(_exact_dict(pbr2.get("inputs"), "PBR2 inputs").get("gt_cache"), "PBR2 GT cache")
    census_gt = _exact_dict(census_body.get("input_custody"), "target census input custody")
    v10_hashes = _exact_dict(
        _exact_dict(v10_lattice.get("configuration"), "V10 configuration").get("input_hashes"),
        "V10 input hashes",
    )
    v13_gt = _exact_dict(v13.get("target_custody"), "V13 target custody")
    v14_gt = _exact_dict(v14.get("target_custody"), "V14 target custody")
    rows = {
        "pbr2": (pbr2_gt.get("sha256"), pbr2_gt.get("bytes")),
        "target_partition_census": (census_gt.get("sha256"), census_gt.get("bytes")),
        "v10_lattice": (v10_hashes.get("gt_n600_npz_sha256"), None),
        "v13": (v13_gt.get("cache_sha256"), v13_gt.get("cache_bytes")),
        "v14": (v14_gt.get("sha256"), v14_gt.get("bytes")),
    }
    for source, (digest, byte_count) in rows.items():
        if not _is_sha256(digest):
            raise G1SignalHarvestError(f"{source} frozen target digest is malformed")
        if byte_count is not None:
            _positive_int(byte_count, f"{source} frozen target bytes")
    hashes = {value[0] for value in rows.values()}
    declared_bytes = {value[1] for value in rows.values() if value[1] is not None}
    if len(hashes) != 1 or len(declared_bytes) != 1:
        raise G1SignalHarvestError("PBR2/census/V10/V13/V14 frozen target custody does not join")
    return {
        "sha256": next(iter(hashes)),
        "bytes": next(iter(declared_bytes)),
        "joined_sources": list(rows),
        "v19c_target_cache_join": "not_declared_by_source_receipt; correction prior only",
    }


def build_signal_harvest_body(
    *,
    pbr2: Mapping[str, Any],
    v13: Mapping[str, Any],
    v14: Mapping[str, Any],
    v19c: Mapping[str, Any],
    v10_lattice: Mapping[str, Any],
    census_envelope: Mapping[str, Any],
    input_custody: Mapping[str, Mapping[str, Any]],
    pbr2_packet_accounting: Mapping[str, Any],
    producer_custody: Mapping[str, Any] | None = None,
    semantic_argv: Iterable[str] = (),
) -> dict[str, Any]:
    """Join exact prior evidence without converting it into payload authority."""

    _require_schema(pbr2, PBR2_SCHEMA, "PBR2 receipt")
    _require_schema(v13, V13_SCHEMA, "V13 receipt")
    _require_schema(v14, V14_SCHEMA, "V14 receipt")
    _require_schema(v19c, V19C_SCHEMA, "V19c receipt")
    _require_schema(v10_lattice, V10_LATTICE_SCHEMA, "V10 lattice receipt")
    _require_schema(census_envelope, CENSUS_ENVELOPE_SCHEMA, "target census")
    _require_research_only(pbr2, "PBR2 receipt")
    _require_research_only(v13, "V13 receipt", expected_axis=_ADVISORY_AXIS)
    _require_research_only(v14, "V14 receipt", expected_axis=_ADVISORY_AXIS)
    _require_research_only(v19c, "V19c receipt", expected_axis=_ADVISORY_AXIS)
    _require_v10_false_authority(v10_lattice)
    if pbr2.get("candidate_payload_allowed") is not False:
        raise G1SignalHarvestError("PBR2 must remain forbidden candidate payload")
    pbr2_row = _exact_dict(pbr2.get("pbr2"), "PBR2 packet")

    census_body = _exact_dict(census_envelope.get("body"), "target census body")
    _require_schema(census_body, CENSUS_BODY_SCHEMA, "target census body")
    _require_research_only(census_body, "target census body")
    expected_body_sha = census_envelope.get("body_sha256")
    actual_body_sha = _sha256(canonical_json_bytes(census_body))
    if expected_body_sha != actual_body_sha:
        raise G1SignalHarvestError("target census body hash mismatch")

    pair_start, pair_stop, validated_strata = _validate_pbr2_teacher(
        pbr2,
        pbr2_row,
        input_custody,
        _exact_dict(pbr2_packet_accounting, "PBR2 strict packet accounting"),
    )
    target_cache_join = _target_cache_join(
        pbr2=pbr2,
        census_body=census_body,
        v10_lattice=v10_lattice,
        v13=v13,
        v14=v14,
    )

    difference = _exact_dict(pbr2.get("exact_difference"), "PBR2 difference")
    strata = []
    for row in validated_strata:
        strata.append(
            {
                "order": row.get("order"),
                "name": row.get("name"),
                "mode": row.get("mode"),
                "corrected_cells": row.get("corrected_cells"),
                "errors_before": row.get("errors_before"),
                "errors_after": row.get("errors_after"),
                "record_count": row.get("record_count"),
                "span_count": row.get("span_count"),
                "payload_bytes_teacher_only": row.get("payload_bytes"),
            }
        )

    measurements = _exact_dict(census_body.get("measurements"), "target census measurements")
    aggregate = _exact_dict(measurements.get("aggregate"), "target census aggregate")
    target_prior = {
        "total_sites": aggregate.get("total_sites"),
        "successive_pair_end_changed_sites": _exact_dict(
            aggregate.get("temporal_changed_sites"), "temporal changed sites"
        ).get("sum"),
        "successive_pair_end_changed_fraction": aggregate.get("temporal_changed_fraction"),
        "row_runs": _exact_dict(aggregate.get("row_runs"), "target row runs").get("sum"),
        "temporal_interpretation": measurements.get("temporal_interpretation"),
    }

    body = {
        "schema": SCHEMA,
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "candidate_payload_emitted": False,
        "pointer_moved": False,
        "input_custody": dict(input_custody),
        "producer_custody": dict(producer_custody or {}),
        "semantic_argv": list(semantic_argv),
        "frozen_target_cache_join": target_cache_join,
        "teacher": {
            "source_pair_start": pair_start,
            "source_pair_stop_exclusive": pair_stop,
            "mismatch_cells": difference.get("mismatch_cells"),
            "total_cells": difference.get("total_cells"),
            "mismatch_fraction": difference.get("mismatch_fraction"),
            "packet_bytes_teacher_only": pbr2_row.get("packet_bytes"),
            "packet_sha256": pbr2_row.get("packet_sha256"),
            "strata": strata,
            "candidate_payload_allowed": False,
            "role": "encoder_side_teacher_and_conditional_entropy_bound",
        },
        "target_partition_prior": target_prior,
        "worldsheet_prior": _worldsheet_signal(v13),
        "realization_prior": _realization_signal(v14, pair_start=pair_start, pair_stop=pair_stop),
        "inverse_preimage_prior": _v10_preimage_signal(v10_lattice),
        "correction_saturation_prior": _v19c_signal(v19c),
        "composition_decision": {
            "legal_stack": "P_plus_G_plus_A",
            "reuse_existing_original_primitive_wire_codecs": True,
            "replay_historical_search_from_zero": False,
            "pbr_packets_in_candidate": False,
            "target_labels_in_candidate": False,
            "dense_y_in_candidate": False,
            "dense_camera_preimage_in_candidate": False,
            "candidate_legality_surface": "payload_lineage_receiver_closure_and_exact_byte_custody",
            "compact_generator_exact_target_output_allowed": True,
            "sparse_hard_pixel_sidecars_allowed_when_counted": True,
            "exact_output_is_not_target_lineage": True,
            "counted_preimage_object": "compact_Y0_Y1_obligation_generator_program",
            "factor2_support_fill_role": "generic_deterministic_receiver_logic",
            "independent_component_admission_thresholds": False,
            "admission_equation": "delta(100*d_seg+sqrt(10*d_pose)+25*archive_bytes/37545489)<0",
            "archive_byte_delta_is_signed": True,
            "family_negative_scope": "finite_measured_instances_only",
            "next_build_order": [
                "compile_bounded_G_from_existing_V9_to_V19c_primitives",
                "bind_one_PairPopulation_across_P_G_IR_A",
                "measure_complete_object_joint_score_value_per_byte",
                "train_only_typed_terminal_remainder_after_matched_byte_controls",
            ],
        },
        "verdict": "USE_EXISTING_ORIGINAL_PRIMITIVES_AS_G_PRIORS_NEVER_AS_SCORE_AUTHORITY",
    }
    return body


def make_envelope(body: Mapping[str, Any]) -> dict[str, Any]:
    """Bind one harvest body to its canonical content identity."""

    body_dict = dict(body)
    return {
        "schema": ENVELOPE_SCHEMA,
        "body": body_dict,
        "body_sha256": _sha256(canonical_json_bytes(body_dict)),
    }


def _reject_unowned_payload_aliases(value: Any, *, path: str = "body") -> None:
    if isinstance(value, Mapping):
        for raw_key, child in value.items():
            key = str(raw_key).lower()
            if key in {"payload", "blob", "data", "base64", "encoded_payload", "raw_byte_values"} or key.endswith(
                "_base64"
            ):
                raise G1SignalHarvestError(f"harvest contains an unowned payload field at {path}.{raw_key}")
            _reject_unowned_payload_aliases(child, path=f"{path}.{raw_key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_unowned_payload_aliases(child, path=f"{path}[{index}]")


def _strict_rebuild_from_custody(body: Mapping[str, Any]) -> None:
    """Reopen every canonical source and compare the fully re-derived body."""

    custody = _exact_dict(body.get("input_custody"), "harvest input custody")
    expected_sources = {
        "pbr2": (DEFAULT_PBR2, PBR2_SCHEMA),
        "v13": (DEFAULT_V13, V13_SCHEMA),
        "v14": (DEFAULT_V14, V14_SCHEMA),
        "v19c": (DEFAULT_V19C, V19C_SCHEMA),
        "v10_lattice": (DEFAULT_V10_LATTICE, V10_LATTICE_SCHEMA),
        "target_partition_census": (DEFAULT_CENSUS, CENSUS_ENVELOPE_SCHEMA),
    }
    if set(custody) != {*expected_sources, "pbr2_packet"}:
        raise G1SignalHarvestError("harvest input custody source set differs")
    documents: dict[str, dict[str, Any]] = {}
    reopened_custody: dict[str, dict[str, Any]] = {}
    for name, (path, schema) in expected_sources.items():
        documents[name], reopened_custody[name] = read_json_with_custody(_REPO_ROOT / path, expected_schema=schema)
        if canonical_json_bytes(custody.get(name)) != canonical_json_bytes(reopened_custody[name]):
            raise G1SignalHarvestError(f"harvest {name} custody differs from exact canonical source bytes")
    packet_payload, reopened_custody["pbr2_packet"] = _read_regular_bytes(_REPO_ROOT / DEFAULT_PBR2_PACKET)
    if canonical_json_bytes(custody.get("pbr2_packet")) != canonical_json_bytes(reopened_custody["pbr2_packet"]):
        raise G1SignalHarvestError("harvest PBR2 packet custody differs from exact canonical source bytes")
    try:
        strict_pbr2_accounting = pbr2_packet_accounting(packet_payload)
    except ProgressiveGeometryResidualError as exc:
        raise G1SignalHarvestError("harvest canonical PBR2 packet no longer strict-reopens") from exc

    producer = _exact_dict(body.get("producer_custody"), "harvest producer custody")
    if set(producer) != {"module", "git_head"}:
        raise G1SignalHarvestError("harvest producer custody fields differ")
    module = _exact_dict(producer.get("module"), "harvest producer module")
    if set(module) != {"path", "bytes", "sha256"}:
        raise G1SignalHarvestError("harvest producer module custody fields differ")
    reopened_module = _read_regular_bytes_custody(Path(__file__))
    if canonical_json_bytes(module) != canonical_json_bytes(reopened_module):
        raise G1SignalHarvestError("harvest producer module differs from current exact source bytes")
    if not _is_git_sha(producer.get("git_head")):
        raise G1SignalHarvestError("harvest producer git_head is malformed")

    expected_argv = [
        ".venv/bin/python",
        "-m",
        "tac.witness_dsl.g1_prior_signal_harvest",
        "--pbr2",
        str(DEFAULT_PBR2),
        "--v13",
        str(DEFAULT_V13),
        "--v14",
        str(DEFAULT_V14),
        "--v19c",
        str(DEFAULT_V19C),
        "--v10-lattice",
        str(DEFAULT_V10_LATTICE),
        "--target-census",
        str(DEFAULT_CENSUS),
        "--output",
        str(DEFAULT_OUTPUT),
    ]
    if body.get("semantic_argv") != expected_argv:
        raise G1SignalHarvestError("harvest semantic argv differs from the canonical replay command")
    rebuilt = build_signal_harvest_body(
        pbr2=documents["pbr2"],
        v13=documents["v13"],
        v14=documents["v14"],
        v19c=documents["v19c"],
        v10_lattice=documents["v10_lattice"],
        census_envelope=documents["target_partition_census"],
        input_custody=reopened_custody,
        pbr2_packet_accounting=strict_pbr2_accounting,
        producer_custody=producer,
        semantic_argv=expected_argv,
    )
    if canonical_json_bytes(body) != canonical_json_bytes(rebuilt):
        raise G1SignalHarvestError("harvest body differs from exact source-derived reconstruction")


def validate_envelope(envelope: Mapping[str, Any], *, reopen_sources: bool = True) -> None:
    """Fail closed on envelope schema or canonical-body drift."""

    if set(envelope) != {"schema", "body", "body_sha256"}:
        raise G1SignalHarvestError("harvest envelope keys are not exact")
    _require_schema(envelope, ENVELOPE_SCHEMA, "harvest envelope")
    body = _exact_dict(envelope.get("body"), "harvest body")
    _require_schema(body, SCHEMA, "harvest body")
    if frozenset(body) != _BODY_FIELDS:
        raise G1SignalHarvestError("harvest body keys are not exact")
    _reject_unowned_payload_aliases(body)
    if envelope.get("body_sha256") != _sha256(canonical_json_bytes(body)):
        raise G1SignalHarvestError("harvest body hash mismatch")
    required_body_authority = {
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "candidate_payload_emitted": False,
        "pointer_moved": False,
    }
    if any(body.get(key) is not expected for key, expected in required_body_authority.items()):
        raise G1SignalHarvestError("harvest body authority boundary differs")
    teacher = _exact_dict(body.get("teacher"), "harvest teacher")
    composition = _exact_dict(body.get("composition_decision"), "harvest composition decision")
    worldsheet = _exact_dict(body.get("worldsheet_prior"), "harvest worldsheet prior")
    realization = _exact_dict(body.get("realization_prior"), "harvest realization prior")
    saturation = _exact_dict(body.get("correction_saturation_prior"), "harvest correction saturation prior")
    inverse = _exact_dict(body.get("inverse_preimage_prior"), "harvest inverse preimage prior")
    sidecar = _exact_dict(inverse.get("dense_measurement_sidecar"), "harvest V10 sidecar")
    if (
        teacher.get("candidate_payload_allowed") is not False
        or composition.get("pbr_packets_in_candidate") is not False
        or composition.get("target_labels_in_candidate") is not False
        or composition.get("dense_y_in_candidate") is not False
        or composition.get("dense_camera_preimage_in_candidate") is not False
        or composition.get("candidate_legality_surface") != "payload_lineage_receiver_closure_and_exact_byte_custody"
        or composition.get("compact_generator_exact_target_output_allowed") is not True
        or composition.get("sparse_hard_pixel_sidecars_allowed_when_counted") is not True
        or composition.get("exact_output_is_not_target_lineage") is not True
        or composition.get("independent_component_admission_thresholds") is not False
        or composition.get("admission_equation") != "delta(100*d_seg+sqrt(10*d_pose)+25*archive_bytes/37545489)<0"
        or composition.get("archive_byte_delta_is_signed") is not True
        or sidecar.get("candidate_payload_allowed_by_harvest") is not False
        or sidecar.get("source_candidate_payload_allowed") not in (None, False)
    ):
        raise G1SignalHarvestError("harvest nested candidate-lineage boundary differs")
    if (
        worldsheet.get("authority") != _ADVISORY_AXIS
        or realization.get("authority") != _ADVISORY_AXIS
        or saturation.get("authority") != _ADVISORY_AXIS
        or inverse.get("evidence_axis") != _V10_AXIS
        or inverse.get("authority") != _V10_VERDICT_SCOPE
    ):
        raise G1SignalHarvestError("harvest nested evidence-axis boundary differs")
    producer = _exact_dict(body.get("producer_custody"), "harvest producer custody")
    module = _exact_dict(producer.get("module"), "harvest producer module")
    if not _is_sha256(module.get("sha256")) or not _is_git_sha(producer.get("git_head")):
        raise G1SignalHarvestError("harvest producer custody is incomplete")
    argv = _exact_list(body.get("semantic_argv"), "harvest semantic argv")
    if not argv or any(not isinstance(value, str) or not value for value in argv):
        raise G1SignalHarvestError("harvest semantic argv is incomplete")
    if reopen_sources:
        _strict_rebuild_from_custody(body)


def _write_once(path: Path, payload: bytes, *, reopen_sources: bool = True) -> None:
    """Crash-atomically publish complete bytes without overwriting a peer."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing_payload, _ = _read_regular_bytes(path)
        if existing_payload != payload:
            raise G1SignalHarvestError(f"output exists with different bytes: {path}")
        reopened = _exact_dict(json.loads(existing_payload), "existing harvest envelope")
        validate_envelope(reopened, reopen_sources=reopen_sources)
        final_payload, _ = _read_regular_bytes(path)
        if final_payload != existing_payload:
            raise G1SignalHarvestError(f"existing harvest changed during validation: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            concurrent_payload, _ = _read_regular_bytes(path)
            if concurrent_payload != payload:
                raise G1SignalHarvestError(f"concurrent output differs: {path}") from None
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    reopened_payload, _ = _read_regular_bytes(path)
    if reopened_payload != payload:
        raise G1SignalHarvestError("published harvest bytes differ")
    try:
        reopened = _exact_dict(json.loads(reopened_payload), "published harvest envelope")
    except json.JSONDecodeError as exc:
        raise G1SignalHarvestError("published harvest is not valid JSON") from exc
    validate_envelope(reopened, reopen_sources=reopen_sources)
    final_payload, _ = _read_regular_bytes(path)
    if final_payload != reopened_payload:
        raise G1SignalHarvestError("published harvest changed during validation")


def _git_head() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    if not _is_git_sha(value):
        raise G1SignalHarvestError("git HEAD must be a 40-hex SHA-1")
    return value


def run(
    *,
    pbr2_path: Path,
    v13_path: Path,
    v14_path: Path,
    v19c_path: Path,
    v10_lattice_path: Path,
    census_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Read, validate, join, and write the canonical frozen-source receipt."""

    supplied = (pbr2_path, v13_path, v14_path, v19c_path, v10_lattice_path, census_path, output_path)
    canonical = (
        DEFAULT_PBR2,
        DEFAULT_V13,
        DEFAULT_V14,
        DEFAULT_V19C,
        DEFAULT_V10_LATTICE,
        DEFAULT_CENSUS,
        DEFAULT_OUTPUT,
    )
    if any(
        path.resolve() != (_REPO_ROOT / expected).resolve() for path, expected in zip(supplied, canonical, strict=True)
    ):
        raise G1SignalHarvestError(
            "g1 prior harvest is canonical-source-only; custom CLI paths are intentionally unsupported"
        )

    documents: dict[str, dict[str, Any]] = {}
    custody: dict[str, dict[str, Any]] = {}
    for name, path, schema in (
        ("pbr2", pbr2_path, PBR2_SCHEMA),
        ("v13", v13_path, V13_SCHEMA),
        ("v14", v14_path, V14_SCHEMA),
        ("v19c", v19c_path, V19C_SCHEMA),
        ("v10_lattice", v10_lattice_path, V10_LATTICE_SCHEMA),
        ("target_partition_census", census_path, CENSUS_ENVELOPE_SCHEMA),
    ):
        documents[name], custody[name] = read_json_with_custody(path, expected_schema=schema)
    packet_path_raw = _exact_dict(documents["pbr2"].get("pbr2"), "PBR2 packet").get("artifact_path")
    if not isinstance(packet_path_raw, str) or not packet_path_raw:
        raise G1SignalHarvestError("PBR2 receipt lacks artifact_path")
    packet_path = Path(packet_path_raw)
    if not packet_path.is_absolute():
        packet_path = _REPO_ROOT / packet_path
    packet_payload, custody["pbr2_packet"] = _read_regular_bytes(packet_path)
    try:
        strict_pbr2_accounting = pbr2_packet_accounting(packet_payload)
    except ProgressiveGeometryResidualError as exc:
        raise G1SignalHarvestError("PBR2 packet failed strict reopen/accounting") from exc
    semantic_argv = [
        ".venv/bin/python",
        "-m",
        "tac.witness_dsl.g1_prior_signal_harvest",
        "--pbr2",
        str(pbr2_path),
        "--v13",
        str(v13_path),
        "--v14",
        str(v14_path),
        "--v19c",
        str(v19c_path),
        "--v10-lattice",
        str(v10_lattice_path),
        "--target-census",
        str(census_path),
        "--output",
        str(output_path),
    ]
    body = build_signal_harvest_body(
        pbr2=documents["pbr2"],
        v13=documents["v13"],
        v14=documents["v14"],
        v19c=documents["v19c"],
        v10_lattice=documents["v10_lattice"],
        census_envelope=documents["target_partition_census"],
        input_custody=custody,
        pbr2_packet_accounting=strict_pbr2_accounting,
        producer_custody={
            "module": _read_regular_bytes_custody(Path(__file__)),
            "git_head": _git_head(),
        },
        semantic_argv=semantic_argv,
    )
    envelope = make_envelope(body)
    validate_envelope(envelope)
    _write_once(output_path.resolve(), canonical_json_bytes(envelope))
    return envelope


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pbr2", type=Path, default=DEFAULT_PBR2)
    parser.add_argument("--v13", type=Path, default=DEFAULT_V13)
    parser.add_argument("--v14", type=Path, default=DEFAULT_V14)
    parser.add_argument("--v19c", type=Path, default=DEFAULT_V19C)
    parser.add_argument("--v10-lattice", type=Path, default=DEFAULT_V10_LATTICE)
    parser.add_argument("--target-census", type=Path, default=DEFAULT_CENSUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    envelope = run(
        pbr2_path=args.pbr2,
        v13_path=args.v13,
        v14_path=args.v14,
        v19c_path=args.v19c,
        v10_lattice_path=args.v10_lattice,
        census_path=args.target_census,
        output_path=args.output,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "body_sha256": envelope["body_sha256"],
                "candidate_payload_emitted": False,
                "score_claim": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
