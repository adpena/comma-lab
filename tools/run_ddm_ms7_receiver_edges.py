#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize DDM MS7 R0, one PF3 receiver control, and its real coder race."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, StrictBool, StrictInt

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO / "src", REPO):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap  # noqa: E402
from tac.canonical_equations.ddm_dynamic_quantum_calibration_20260724 import (  # noqa: E402
    dynamic_quantum_calibration,
)
from tac.canonical_equations.ddm_ms2r_tolerance_capped_solve_20260724 import (  # noqa: E402
    tolerance_capped_rung_score,
)
from tac.optimization.ddm_ms7_receiver_edges import (  # noqa: E402
    POINTER,
    MS7ReceiverEdgesError,
    build_r0_reach_table,
    race_same_receiver_object,
    read_bound_atlas,
    read_bound_json,
    sha256_file,
)
from tac.optimization.ddm_rg1_receiver_grammar import (  # noqa: E402
    RG3ResidualCoordinateV1,
    compile_rg3_receiver_grammar,
    receive_rg1_receiver_grammar,
)
from tac.optimization.direct_description_entropy_priced_member import (  # noqa: E402
    rfc8785_canonicalize,
)
from tools.measure_ddm_ms6_receiver_support import _fast_receiver, _probe_context  # noqa: E402
from tools.measure_ddm_v14_realization_fidelity import (  # noqa: E402
    DDMV14RealizationFidelityConfigV1,
    _forward,
    _load_models,
)
from tools.measure_ddm_v19c_correction_saturation import (  # noqa: E402
    _compile_coupled_margin_fast,
    _compile_preuint8_fast,
)

RUN_ID: Final = "ddm_ms7_receiver_edges_and_25bucket_reach_20260724T172249Z"
LANE_ID: Final = "lane_ddm_ms7_receiver_edges_20260724"
RECEIPT_SCHEMA: Final = "ddm_ms7_receiver_edges_receipt.v1"
PF3_SCHEMA: Final = "ddm_ms7_pf3_receiver_control.v1"
CONFIG = REPO / ".omx/research/configs/ddm_ms7_receiver_edges_20260724.json"
OUTPUT = REPO / ".omx/research" / RUN_ID
R0_PATH = OUTPUT / "r0_25_bucket_reach_table.json"
RECEIPT_PATH = OUTPUT / "ddm_ms7_receiver_edges_receipt.json"
ALLOWED_ERRORS: Final = 136_839
SCORED_PIXELS: Final = 600 * 512 * 384
POSE_COORDINATES: Final = 600 * 6
_ACTUATOR = re.compile(
    r"^rg3\.(?P<family>class_birth|finer_event|fisher_stratum)\."
    r"pair(?P<pair>\d{3})\.class(?P<class_a>\d)_(?P<class_b>\d)\."
    r"(?P<stratum>boundary|cell)\.(?P<temporal>static_in_image|transient)\."
    r"band(?P<row_band>\d{2})\.fine(?P<fine_band>\d{2})\.mag(?P<magnitude>\d+)$"
)
_FAMILY = {
    "class_birth": "EVENT_LOCAL_SKELETON_CLASS_BIRTH_PRODUCTION",
    "finer_event": "FINER_EVENT_LOCAL_SKELETON_AMPLITUDE_CODEBOOK",
    "fisher_stratum": "FISHER_MARGIN_PER_STRATUM_SKELETON_AMPLITUDE_CODEBOOK",
}
_VALIDITY_RADIUS = {"class_birth": 1, "finer_event": 2, "fisher_stratum": 2}


class DDMMS7ReceiverEdgesConfigV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema: Literal["DDMMS7ReceiverEdgesConfigV1"]
    run_id: Literal[RUN_ID]
    lane_id: Literal[LANE_ID]
    direct_metric_path: str
    direct_metric_sha256: str
    bundle_complete_path: str
    bundle_complete_sha256: str
    g3_atlas_path: str
    g3_atlas_sha256: str
    dm4_receipt_path: str
    dm4_receipt_sha256: str
    base_archive_path: str
    base_archive_sha256: str
    base_receipt_path: str
    base_receipt_sha256: str
    scorer_config_path: str
    control_selection_policy: Literal[
        "maximum_flip_weighted_S_leverage_terminal_row_then_positive_measured_existing_coordinate"
    ]
    expected_control_pair_id: StrictInt
    expected_control_bucket_id: str
    bulk_root: str
    seed: StrictInt
    scorer_threads: Literal[4]
    scorer_batch_size: Literal[16]
    minimum_free_bytes: StrictInt
    research_only: StrictBool
    execution_allowed: StrictBool
    score_claim: StrictBool
    main_review_required: StrictBool


def _resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else REPO / value


def _read_config(path: Path) -> tuple[DDMMS7ReceiverEdgesConfigV1, str]:
    payload = path.read_bytes()
    value = DDMMS7ReceiverEdgesConfigV1.model_validate_json(payload)
    if (
        value.seed != 1234
        or value.research_only is not True
        or value.execution_allowed is not False
        or value.score_claim is not False
        or value.main_review_required is not True
    ):
        raise MS7ReceiverEdgesError("MS7 typed execution boundary differs")
    return value, hashlib.sha256(payload).hexdigest()


def _publish(path: Path, value: Mapping[str, Any] | bytes) -> None:
    payload = bytes(value) if isinstance(value, bytes) else rfc8785_canonicalize(dict(value))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise MS7ReceiverEdgesError(f"immutable output differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _artifact(path: Path, role: str) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "role": role,
    }


def _load_sources(
    config: DDMMS7ReceiverEdgesConfigV1,
) -> tuple[dict[str, Any], dict[str, Any], dict[int, dict[str, Any]], dict[str, Any]]:
    direct_path = _resolve(config.direct_metric_path)
    dm4_path = _resolve(config.dm4_receipt_path)
    atlas_path = _resolve(config.g3_atlas_path)
    bundle_path = _resolve(config.bundle_complete_path)
    direct = read_bound_json(direct_path, config.direct_metric_sha256)
    dm4 = read_bound_json(dm4_path, config.dm4_receipt_sha256)
    atlas = read_bound_atlas(atlas_path, config.g3_atlas_sha256)
    bundle = read_bound_json(bundle_path, config.bundle_complete_sha256)
    admissibility = bundle.get("headline_admissibility")
    if (
        bundle.get("schema") != "ddm_metric_custody_bundle.v1"
        or bundle.get("status") != "COMPLETE"
        or not isinstance(admissibility, Mapping)
        or admissibility.get("bundle_complete") is not True
        or admissibility.get("scorer_metric_active") is not True
        or admissibility.get("pose_tube_active") is not True
        or admissibility.get("score_claim") is not False
    ):
        raise MS7ReceiverEdgesError("MS4D bundle is not BUNDLE-COMPLETE")
    sources = {
        "direct_metric": _artifact(direct_path, "MS4D exact 25 direct blocks"),
        "dm4_receipt": _artifact(dm4_path, "measured guaranteed-reach prices"),
        "g3_atlas": _artifact(atlas_path, "pair score mass and flip population"),
        "bundle_complete": _artifact(bundle_path, "PF3 metric admission gate"),
    }
    return direct, dm4, atlas, sources


def _gain(block: Mapping[str, Any]) -> float:
    values = block.get("composite_r_adjoint_readback")
    if not isinstance(values, list) or len(values) != 4 or any(not isinstance(value, (int, float)) for value in values):
        raise MS7ReceiverEdgesError("MS4D block lacks exact rank-4 adjoint readback")
    gain = math.sqrt(sum(float(value) ** 2 for value in values))
    if not math.isfinite(gain) or gain <= 0.0:
        raise MS7ReceiverEdgesError("MS4D composite-R gain is not finite positive")
    return gain


def _positive_probe(block: Mapping[str, Any]) -> tuple[str, str, int]:
    custody = block.get("probe_custody")
    probes = custody.get("probes") if isinstance(custody, Mapping) else None
    if not isinstance(probes, list):
        raise MS7ReceiverEdgesError("MS4D block lacks MS6 probe custody")
    positive = [
        row
        for row in probes
        if isinstance(row, Mapping)
        and row.get("direction_id") == "POSITIVE_ONE_QUANTUM"
        and isinstance(row.get("receiver_actuator_id"), str)
    ]
    if not positive:
        raise MS7ReceiverEdgesError("MS4D block lacks a positive measured RG3 coordinate")
    row = min(positive, key=lambda item: str(item["receiver_actuator_id"]))
    match = _ACTUATOR.fullmatch(str(row["receiver_actuator_id"]))
    if match is None:
        raise MS7ReceiverEdgesError("MS6 RG3 actuator ID does not parse")
    return str(row["receiver_actuator_id"]), match.group("family"), int(match.group("magnitude"))


def _calibration_rows(direct: Mapping[str, Any]) -> list[dict[str, Any]]:
    blocks = direct.get("direct_blocks")
    if not isinstance(blocks, list) or len(blocks) != 25:
        raise MS7ReceiverEdgesError("dynamic calibration requires exact 25 blocks")
    output = []
    for block in blocks:
        if not isinstance(block, Mapping):
            raise MS7ReceiverEdgesError("dynamic calibration block is malformed")
        actuator_id, family, _prior_magnitude = _positive_probe(block)
        calibration = dynamic_quantum_calibration(
            composite_r_gain=_gain(block),
            realized_uint8_deadzone=1.0,
            lattice=(1, 2, 4, 8, 16),
            validity_radius=_VALIDITY_RADIUS[family],
        )
        output.append(
            {
                "pair_id": block["pair_id"],
                "bucket_id": block["bucket_id"],
                "receiver_coordinate": actuator_id,
                "receiver_family": family,
                "gain_authority": "MS4D_COMPOSITE_R_ADJOINT_L2",
                "deadzone_authority": "G2F_WHOLE_LSB_KNEE_AND_580_532_UINT8_LAW",
                "validity_authority": ("MS6_RG3_MEASURED_MAGNITUDE_RANGE_INTERSECT_V16_V17_RADIUS_LAW"),
                "calibration": calibration,
                "predicted_vs_realized": None,
                "price_bytes": None,
            }
        )
    return output


def materialize_r0(
    config: DDMMS7ReceiverEdgesConfigV1,
    config_sha256: str,
) -> dict[str, Any]:
    direct, dm4, atlas, sources = _load_sources(config)
    value = build_r0_reach_table(
        direct_metric=direct,
        dm4_receipt=dm4,
        atlas=atlas,
        sources=sources,
    )
    value.update(
        {
            "run_id": RUN_ID,
            "lane_id": LANE_ID,
            "typed_config": config.model_dump(mode="json"),
            "typed_config_sha256": config_sha256,
            "dynamic_quantum_calibration": {
                "equation_id": "dynamic_quantum_calibration_v1",
                "rows": _calibration_rows(direct),
            },
            "directive_consumption": [
                {
                    "utc": "2026-07-24T14:45:16Z",
                    "application": (
                        "control coordinate is an existing scorer-recursive class-birth "
                        "production; no generic disk/menu construction"
                    ),
                },
                {
                    "utc": "2026-07-24T17:26:45Z",
                    "application": ("fixed ladder superseded by gain/deadzone/validity-derived k-star"),
                },
                {
                    "utc": "2026-07-24T17:42:58Z",
                    "application": ("first coder race adds exact constriction and counted zstd-dictionary rows"),
                },
            ],
        }
    )
    _publish(R0_PATH, value)
    return value


def _control_block(direct: Mapping[str, Any], config: DDMMS7ReceiverEdgesConfigV1) -> dict[str, Any]:
    rows = [
        row
        for row in direct.get("direct_blocks", ())
        if isinstance(row, Mapping)
        and row.get("pair_id") == config.expected_control_pair_id
        and row.get("bucket_id") == config.expected_control_bucket_id
    ]
    if len(rows) != 1:
        raise MS7ReceiverEdgesError("expected PF3 control row identity differs")
    return dict(rows[0])


def _coordinate(block: Mapping[str, Any]) -> tuple[RG3ResidualCoordinateV1, dict[str, Any]]:
    actuator_id, family_slug, _prior_magnitude = _positive_probe(block)
    match = _ACTUATOR.fullmatch(actuator_id)
    if match is None:
        raise MS7ReceiverEdgesError("PF3 control actuator ID does not parse")
    calibration = dynamic_quantum_calibration(
        composite_r_gain=_gain(block),
        realized_uint8_deadzone=1.0,
        lattice=(1, 2, 4, 8, 16),
        validity_radius=_VALIDITY_RADIUS[family_slug],
    )
    selected = calibration["selected_k_star"]
    if selected != 1:
        raise MS7ReceiverEdgesError("PF3 control no longer derives the sealed class-birth quantum")
    coordinate = RG3ResidualCoordinateV1(
        pair_index=int(match.group("pair")),
        class_a=int(match.group("class_a")),
        class_b=int(match.group("class_b")),
        family=_FAMILY[family_slug],
        temporal_class=match.group("temporal").upper(),
        row_band=int(match.group("row_band")),
        fine_band=int(match.group("fine_band")),
        signed_quanta=selected,
    )
    return coordinate, calibration


def _storage_preflight(config: DDMMS7ReceiverEdgesConfigV1) -> dict[str, Any]:
    bulk = Path(config.bulk_root)
    bulk.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(bulk)
    if usage.free < config.minimum_free_bytes:
        raise MS7ReceiverEdgesError("SSD preflight has insufficient free bytes")
    return {
        "selected_tier": str(bulk),
        "free_bytes": usage.free,
        "minimum_free_bytes": config.minimum_free_bytes,
        "passed": True,
    }


def _load_base(config: DDMMS7ReceiverEdgesConfigV1) -> tuple[bytes, dict[str, Any]]:
    archive_path = _resolve(config.base_archive_path)
    if (
        not archive_path.is_file()
        or archive_path.is_symlink()
        or sha256_file(archive_path) != config.base_archive_sha256
    ):
        raise MS7ReceiverEdgesError("V19C base archive custody differs")
    receipt = read_bound_json(
        _resolve(config.base_receipt_path),
        config.base_receipt_sha256,
    )
    return archive_path.read_bytes(), receipt


def _endpoint(receipt: Mapping[str, Any]) -> tuple[int, float]:
    curve = receipt.get("curve")
    endpoint = curve.get("n600_endpoint") if isinstance(curve, Mapping) else None
    c1 = receipt.get("c1_bucket_attribution")
    final = c1.get("measured_v19c_final") if isinstance(c1, Mapping) else None
    if (
        not isinstance(endpoint, Mapping)
        or not isinstance(final, Mapping)
        or not isinstance(final.get("residual_errors"), int)
        or not isinstance(final.get("role_errors"), int)
        or not isinstance(endpoint.get("d_pose"), str)
        or not isinstance(endpoint.get("d_seg"), str)
    ):
        raise MS7ReceiverEdgesError("V19C endpoint custody is malformed")
    errors = int(final["residual_errors"]) + int(final["role_errors"])
    d_pose = float(endpoint["d_pose"])
    d_seg = float(endpoint["d_seg"])
    if not math.isclose(errors / SCORED_PIXELS, d_seg, rel_tol=0.0, abs_tol=5e-13):
        raise MS7ReceiverEdgesError("V19C endpoint integer errors and d_seg differ")
    return errors, d_pose


def _verify_large_cache(path: Path, expected_bytes: int, expected_sha256: str) -> dict[str, Any]:
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != expected_bytes
        or sha256_file(path) != expected_sha256
    ):
        raise MS7ReceiverEdgesError("target cache custody differs")
    return _artifact(path, "n600 GT labels and Pose targets")


def _scorer_measurement(
    *,
    config: DDMMS7ReceiverEdgesConfigV1,
    base_archive: bytes,
    candidate_archive: bytes,
    endpoint_errors: int,
    endpoint_d_pose: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    scorer_config_path = _resolve(config.scorer_config_path)
    scorer_payload = scorer_config_path.read_bytes()
    scorer_config = DDMV14RealizationFidelityConfigV1.model_validate_json(scorer_payload)
    if (
        scorer_config.seed != config.seed
        or scorer_config.scorer_threads != config.scorer_threads
        or scorer_config.scorer_batch_size != config.scorer_batch_size
        or scorer_config.pair_count != 600
    ):
        raise MS7ReceiverEdgesError("frozen scorer config differs from MS7 contract")
    target_path = Path(scorer_config.target_cache_path)
    target_custody = _verify_large_cache(
        target_path,
        scorer_config.target_cache_bytes,
        scorer_config.target_cache_sha256,
    )
    labels_all = open_stored_npy_memmap(target_path, "lstars")
    poses_all = open_stored_npy_memmap(target_path, "gt_poses")
    segnet, posenet, scorer_custody = _load_models(scorer_config)

    selected_pair = config.expected_control_pair_id
    start = selected_pair // config.scorer_batch_size * config.scorer_batch_size
    pair_ids = tuple(range(start, start + config.scorer_batch_size))
    selected_local = pair_ids.index(selected_pair)
    base_receiver = _fast_receiver(base_archive)
    candidate_receiver = _fast_receiver(candidate_archive)
    base_camera = base_receiver.render_camera_pairs(pair_ids)
    candidate_camera = candidate_receiver.render_camera_pairs(pair_ids)
    changed = candidate_camera.astype(np.int16) - base_camera.astype(np.int16)
    per_pair_changed = np.count_nonzero(changed, axis=(1, 2, 3, 4))
    if per_pair_changed[selected_local] <= 0 or np.count_nonzero(per_pair_changed) != 1:
        raise MS7ReceiverEdgesError("PF3 receiver coordinate escaped or missed its exact pair")
    nonzero = np.abs(changed[changed != 0])
    if nonzero.size == 0:
        raise MS7ReceiverEdgesError("PF3 receiver coordinate died before uint8 realization")

    base_cells, base_pose = _forward(segnet, posenet, base_camera)
    candidate_cells, candidate_pose = _forward(segnet, posenet, candidate_camera)
    replay_cells, replay_pose = _forward(segnet, posenet, candidate_camera)
    if not np.array_equal(candidate_cells, replay_cells) or not np.array_equal(candidate_pose, replay_pose):
        raise MS7ReceiverEdgesError("PF3 exact scorer replay is nondeterministic")
    labels = np.asarray(labels_all[np.asarray(pair_ids, dtype=np.intp)], dtype=np.uint8)
    poses = np.asarray(poses_all[np.asarray(pair_ids, dtype=np.intp)], dtype=np.float64)
    base_batch_errors = int(np.count_nonzero(base_cells != labels))
    candidate_batch_errors = int(np.count_nonzero(candidate_cells != labels))
    base_batch_pose_sse = float(np.square(base_pose - poses).sum(dtype=np.float64))
    candidate_batch_pose_sse = float(np.square(candidate_pose - poses).sum(dtype=np.float64))
    delta_errors = candidate_batch_errors - base_batch_errors
    delta_pose_sse = candidate_batch_pose_sse - base_batch_pose_sse
    candidate_errors = endpoint_errors + delta_errors
    candidate_d_pose = (endpoint_d_pose * POSE_COORDINATES + delta_pose_sse) / POSE_COORDINATES
    if candidate_errors < 0 or candidate_d_pose < 0.0:
        raise MS7ReceiverEdgesError("PF3 global endpoint splice is invalid")
    measurement = {
        "batch_geometry": {
            "pair_ids": list(pair_ids),
            "batch_size": config.scorer_batch_size,
            "selected_pair_local_index": selected_local,
            "authority": "MATCHES_V19C_N600_BATCH16_ENDPOINT_GEOMETRY",
        },
        "realized_uint8_quantum": {
            "minimum_nonzero_absolute_level": int(nonzero.min()),
            "maximum_nonzero_absolute_level": int(nonzero.max()),
            "changed_channel_values": int(nonzero.size),
            "changed_pair_ids": [selected_pair],
            "deadzone_crossed": int(nonzero.min()) >= 1,
        },
        "same_object_candidate_delta": {
            "base_batch_errors": base_batch_errors,
            "candidate_batch_errors": candidate_batch_errors,
            "delta_global_errors": delta_errors,
            "base_batch_pose_sse_6d": base_batch_pose_sse,
            "candidate_batch_pose_sse_6d": candidate_batch_pose_sse,
            "delta_global_pose_sse_6d": delta_pose_sse,
            "base_global_errors": endpoint_errors,
            "candidate_global_errors": candidate_errors,
            "base_global_d_pose": endpoint_d_pose,
            "candidate_global_d_pose": candidate_d_pose,
            "changed_argmax_cells": int(np.count_nonzero(candidate_cells != base_cells)),
            "unchanged_pair_outputs_identical": bool(
                np.array_equal(
                    np.delete(candidate_cells, selected_local, axis=0),
                    np.delete(base_cells, selected_local, axis=0),
                )
                and np.array_equal(
                    np.delete(candidate_pose, selected_local, axis=0),
                    np.delete(base_pose, selected_local, axis=0),
                )
            ),
            "splice_authority": (
                "fresh exact batch16 delta plus SHA-bound V19C n600 endpoint; "
                "coordinate structurally changes one pair only"
            ),
        },
        "deterministic_replay": True,
        "target_cache": target_custody,
        "scorer_custody": scorer_custody,
        "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
        "score_claim": False,
    }
    return measurement, {
        "seg_errors": candidate_errors,
        "d_pose": candidate_d_pose,
    }


def materialize_pf3(
    config: DDMMS7ReceiverEdgesConfigV1,
    config_sha256: str,
    r0: Mapping[str, Any],
) -> dict[str, Any]:
    storage = _storage_preflight(config)
    direct, _dm4, _atlas, sources = _load_sources(config)
    block = _control_block(direct, config)
    coordinate, calibration = _coordinate(block)
    base_archive, base_receipt = _load_base(config)
    endpoint_errors, endpoint_d_pose = _endpoint(base_receipt)
    context = _probe_context(base_archive)
    candidate_carrier = compile_rg3_receiver_grammar(
        context.base_carrier,
        rg3_residuals=(coordinate,),
    )
    candidate_coupled = _compile_coupled_margin_fast(
        candidate_carrier,
        context.coupled_program,
    )
    candidate_archive = _compile_preuint8_fast(
        candidate_coupled,
        context.preuint8_program,
    )
    nested_receiver = receive_rg1_receiver_grammar(
        candidate_carrier,
        verify_member_effects=True,
    )
    if nested_receiver.archive != candidate_carrier:
        raise MS7ReceiverEdgesError("PF3 receiver parse-back changed candidate bytes")

    bulk = Path(config.bulk_root)
    candidate_path = bulk / "stage_checkpoints/02_pf3/candidate_rg3_receiver.zip"
    _publish(candidate_path, candidate_archive)
    race, frames = race_same_receiver_object(candidate_archive)
    frame_artifacts = {}
    for codec, frame in sorted(frames.items()):
        suffix = "zip" if codec == "RAW_COMPACT" else "bin"
        path = bulk / "stage_checkpoints/03_coder_race" / f"{codec}.{suffix}"
        _publish(path, frame)
        frame_artifacts[codec] = _artifact(path, f"{codec} exact same-object frame")
    race["frame_artifacts"] = frame_artifacts

    measurement, callable_inputs = _scorer_measurement(
        config=config,
        base_archive=base_archive,
        candidate_archive=candidate_archive,
        endpoint_errors=endpoint_errors,
        endpoint_d_pose=endpoint_d_pose,
    )
    realized = measurement["realized_uint8_quantum"]
    predicted_vs_realized = {
        "predicted_k_star": calibration["predicted_k_star"],
        "selected_k_star": calibration["selected_k_star"],
        "predicted_deadzone_crossed": calibration["selected_k_star"] is not None,
        "realized_minimum_nonzero_uint8_level": realized["minimum_nonzero_absolute_level"],
        "realized_deadzone_crossed": realized["deadzone_crossed"],
        "amplitude_validity_check_passed": bool(
            calibration["selected_k_star"] is not None and realized["deadzone_crossed"]
        ),
    }
    winner = race["winner"]
    rung = tolerance_capped_rung_score(
        seg_errors=int(callable_inputs["seg_errors"]),
        scored_pixels=SCORED_PIXELS,
        d_pose=float(callable_inputs["d_pose"]),
        raw_compact_bytes=len(candidate_archive),
        best_coded_bytes=int(winner["framed_bytes"]),
        allowed_errors=ALLOWED_ERRORS,
        bundle_complete=True,
        parseback_exact=True,
        uint8_reverified=True,
    )
    control_r0 = [
        row
        for row in r0["rows"]
        if row["pair_id"] == config.expected_control_pair_id and row["bucket_id"] == config.expected_control_bucket_id
    ]
    if len(control_r0) != 1 or control_r0[0]["verdict"] != "UNREACHABLE-AND-IGNORED":
        raise MS7ReceiverEdgesError("PF3 control R0 disposition differs")
    value = {
        "schema": PF3_SCHEMA,
        "run_id": RUN_ID,
        "lane_id": LANE_ID,
        "pointer": POINTER,
        "pointer_moved": False,
        "control_identity": {
            "pair_id": config.expected_control_pair_id,
            "bucket_id": config.expected_control_bucket_id,
            "r0_verdict": control_r0[0]["verdict"],
            "measurement_role": ("MANDATED_PF3_PRICING_CONTROL_ONLY_NOT_R0_ADMISSION"),
        },
        "dynamic_quantum_calibration": {
            "equation_id": "dynamic_quantum_calibration_v1",
            "composite_r_gain": _gain(block),
            "calibration": calibration,
            "predicted_vs_realized": predicted_vs_realized,
        },
        "coordinate": {
            "actuator_id": coordinate.actuator_id,
            "pair_index": coordinate.pair_index,
            "class_a": coordinate.class_a,
            "class_b": coordinate.class_b,
            "family": coordinate.family,
            "temporal_class": coordinate.temporal_class,
            "row_band": coordinate.row_band,
            "fine_band": coordinate.fine_band,
            "signed_quanta": coordinate.signed_quanta,
            "scorer_recursive_derivation": (
                "MS6 existing class-birth coordinate derived from receiver geometry; "
                "MS4D post-R rank-4 adjoint gain; no generic spatial menu"
            ),
        },
        "five_pf3_edges": {
            "receiver_object_builder": {
                "callable_pipeline": [
                    "tac.optimization.ddm_rg1_receiver_grammar:compile_rg3_receiver_grammar",
                    "tools.measure_ddm_v19c_correction_saturation:_compile_coupled_margin_fast",
                    "tools.measure_ddm_v19c_correction_saturation:_compile_preuint8_fast",
                ],
                "composition": (
                    "insert RG3 into the nested carrier, then deterministically rewrap the "
                    "frozen coupled-margin and pre-uint8 V19C receiver programs"
                ),
                "candidate": _artifact(candidate_path, "deterministic RG3 receiver object"),
                "parseback_exact": True,
            },
            "realized_uint8_quantum": measurement["realized_uint8_quantum"],
            "same_object_candidate_delta": measurement["same_object_candidate_delta"],
            "dimension_rate_home": {
                "stream_type": "SKELETON",
                "layer_home": "L3_RASTER",
                "member": "production/residual_family_coordinates.rg3rf",
                "same_object_raw_bytes": len(candidate_archive),
            },
            "coder_payload_owner": {
                "owner": race["coder_payload_owner"],
                "winning_codec": winner["codec"],
                "winning_counted_bytes": winner["framed_bytes"],
                "winning_frame_sha256": winner["frame_sha256"],
            },
        },
        "coder_race": race,
        "registered_waterfill_callable": {
            "equation_id": "ddm_tolerance_capped_min_score_waterfill_v1",
            "inputs": {
                **callable_inputs,
                "scored_pixels": SCORED_PIXELS,
                "raw_compact_bytes": len(candidate_archive),
                "best_coded_bytes": winner["framed_bytes"],
                "allowed_errors": ALLOWED_ERRORS,
                "bundle_complete": True,
                "parseback_exact": True,
                "uint8_reverified": True,
            },
            "output": rung,
            "priced_rung_status": ("MEASURED_PRICED_CONTROL_NONADMISSIBLE_R0_AND_ERROR_CAP"),
        },
        "measurement": measurement,
        "storage_preflight": storage,
        "sources": sources,
        "typed_config_sha256": config_sha256,
        "score_claim": False,
        "research_only": True,
        "main_review_required": True,
        "verdict_scope": (
            "INSTANCE one dynamic class-birth coordinate x exact V19C endpoint x "
            "local macOS-CPU scorer; not a family reach or contest score claim"
        ),
    }
    checkpoint = bulk / "stage_checkpoints/04_complete/pf3_receiver_control.json"
    _publish(checkpoint, value)
    value["bulk_checkpoint"] = _artifact(
        checkpoint,
        "complete resumable PF3 measurement checkpoint",
    )
    return value


def materialize_all(config_path: Path = CONFIG) -> dict[str, Any]:
    config, config_sha256 = _read_config(config_path)
    r0 = materialize_r0(config, config_sha256)
    pf3 = materialize_pf3(config, config_sha256, r0)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "run_id": RUN_ID,
        "lane_id": LANE_ID,
        "pointer": POINTER,
        "pointer_moved": False,
        "r0": {
            "artifact": _artifact(R0_PATH, "mandatory 25-row mass/reach table"),
            "row_count": r0["row_count"],
            "mass_paying_row_count": r0["mass_paying_row_count"],
            "unreachable_and_ignored_row_count": r0["unreachable_and_ignored_row_count"],
            "r1_execution_disposition": r0["r1_execution_disposition"],
            "r2_execution_disposition": r0["r2_execution_disposition"],
            "r3_execution_disposition": r0["r3_execution_disposition"],
        },
        "pf3": pf3,
        "triality": {
            "dsl": str(CONFIG.relative_to(REPO)),
            "dag": (f".omx/research/{RUN_ID}/ddm_ms7_receiver_edges_DAG_FEED.json"),
            "equations": [
                "dynamic_quantum_calibration_v1",
                "ddm_tolerance_capped_min_score_waterfill_v1",
            ],
        },
        "directive_consumption": r0["directive_consumption"],
        "score_claim": False,
        "promotion_eligible": False,
        "research_only": True,
        "execution_allowed": False,
        "main_review_required": True,
        "verdict": (
            "R0_25_OF_25_UNREACHABLE_AND_IGNORED_UNDER_MEASURED_REACH_PRICES; "
            "PF3_FIVE_EDGES_BOUND_AND_FIRST_REAL_CODER_RACE_PRICED_AS_NONADMITTED_CONTROL"
        ),
        "verdict_scope": (
            "INSTANCE exact 25 terminal rows x measured DM4 guaranteed-reach prices; "
            "dynamic R1 and T-residual prices remain NULL; pointer unchanged"
        ),
    }
    _publish(RECEIPT_PATH, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--stage", choices=("r0", "all"), default="all")
    args = parser.parse_args()
    config, config_sha256 = _read_config(args.config)
    if args.stage == "r0":
        value = materialize_r0(config, config_sha256)
        output = {
            "stage": "r0",
            "artifact": str(R0_PATH),
            "row_count": value["row_count"],
            "mass_paying_row_count": value["mass_paying_row_count"],
        }
    else:
        value = materialize_all(args.config)
        output = {
            "stage": "all",
            "receipt": str(RECEIPT_PATH),
            "verdict": value["verdict"],
        }
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
