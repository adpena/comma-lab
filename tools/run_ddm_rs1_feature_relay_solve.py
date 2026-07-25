#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Strict admission and execution gate for DDM RS1 feature relay.

This runner does not manufacture internal-layer dynamics from endpoint
summaries.  It rehashes every declared source, audits the three-station
multiple-shooting prerequisites, and emits either:

* ``READY_FOR_G3_TOP24_REALIZED_RADIUS_MEASUREMENT`` when every measured
  station edge is present and both predictive solves have completed; or
* a typed, formulation-scoped blocker naming the missing edges.

Actual realized acceptance remains delegated to the existing J8F n600
receiver/scorer harness after this gate passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tac.optimization.ddm_feature_relay_multiple_shooting import (  # noqa: E402
    ACCEPTANCE_AUTHORITY,
    METRIC_KIND,
    RelayProblemV1,
    RelaySegmentV1,
    RelayStationV1,
    solve_direct_final_station,
    solve_multiple_shooting,
)

CONFIG_SCHEMA: Final = "ddm_rs1_feature_relay_solve_config.v1"
RECEIPT_SCHEMA: Final = "ddm_rs1_feature_relay_solve_receipt.v1"
BLOCKER_SCHEMA: Final = "ddm_rs1_feature_relay_solve_blocker.v1"
STATION_BUNDLE_SCHEMA: Final = "ddm_rs1_measured_station_bundle.v1"
EVIDENCE_AXIS: Final = "[macOS-CPU frozen-scorer advisory]"
POINTER: Final = "0.1910828242 [contest-CPU]"
EXPECTED_BINDINGS: Final = frozenset(
    {
        "authority",
        "pre_se_locus_memo",
        "pre_se_locus_equation",
        "at1_tracked_receipt",
        "at1_gaze_atlas",
        "sn1_tracked_receipt",
        "sn1_telemetry",
        "ms4_bundle",
        "ms4_seg_metric",
        "v17_validity_law",
        "v17_validity_receipt",
        "j8f_receipt",
        "range_a_projector",
        "solver_source",
        "runner_source",
    }
)


class RelayAdmissionError(ValueError):
    """Fail-closed malformed RS1 config or source custody."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _valid_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(char in "0123456789abcdef" for char in value)
    )


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write(path, _canonical_bytes(payload) + b"\n")


@dataclass(frozen=True, slots=True)
class BoundSource:
    path: str
    sha256: str
    bytes: int | None

    @classmethod
    def from_payload(
        cls,
        payload: Any,
        *,
        name: str,
        allow_unknown_bytes: bool = False,
    ) -> BoundSource:
        if not isinstance(payload, Mapping):
            raise RelayAdmissionError(f"{name} binding must be a mapping")
        path = payload.get("path")
        digest = payload.get("sha256")
        byte_count = payload.get("bytes")
        if (
            not isinstance(path, str)
            or not path
            or not _valid_sha256(digest)
            or (
                byte_count is None
                and not allow_unknown_bytes
            )
            or (
                byte_count is not None
                and (
                    isinstance(byte_count, bool)
                    or not isinstance(byte_count, int)
                    or byte_count <= 0
                )
            )
        ):
            raise RelayAdmissionError(f"{name} binding is malformed")
        return cls(path=path, sha256=digest, bytes=byte_count)

    def resolve(self, repo_root: Path) -> Path:
        path = Path(self.path)
        return path if path.is_absolute() else repo_root / path

    def read(self, repo_root: Path) -> bytes:
        path = self.resolve(repo_root)
        if not path.is_file() or path.is_symlink():
            raise RelayAdmissionError(f"bound source is unavailable: {path}")
        payload = path.read_bytes()
        if (
            self.bytes is not None
            and len(payload) != self.bytes
        ) or _sha256(payload) != self.sha256:
            raise RelayAdmissionError(f"bound source identity differs: {path}")
        return payload

    def to_payload(self) -> dict[str, Any]:
        return {"path": self.path, "sha256": self.sha256, "bytes": self.bytes}


@dataclass(frozen=True, slots=True)
class RelayAuditConfig:
    path: Path
    lane_id: str
    run_id: str
    source_bindings: Mapping[str, BoundSource]
    station_bundle: BoundSource | None
    missing_pre_se_receipts: tuple[BoundSource, ...]
    hard_tail_block_count: int
    bounded_n600_block_count: int

    @classmethod
    def from_path(cls, path: str | Path) -> RelayAuditConfig:
        config_path = Path(path).resolve()
        payload = json.loads(config_path.read_bytes())
        required = {
            "schema": CONFIG_SCHEMA,
            "research_only": True,
            "score_claim": False,
            "promotion_eligible": False,
            "pointer_moved": False,
            "pointer": f"{POINTER} UNMOVED",
            "evidence_axis": EVIDENCE_AXIS,
            "main_landing_review_required": True,
            "metric_primary": METRIC_KIND,
            "euclidean_control_only": True,
            "realized_acceptance": ACCEPTANCE_AUTHORITY,
            "hard_tail_block_count": 24,
        }
        drift = {
            key: (payload.get(key), expected)
            for key, expected in required.items()
            if payload.get(key) != expected
        }
        if drift:
            raise RelayAdmissionError(f"RS1 typed/authority contract differs: {drift}")
        bounded = payload.get("bounded_n600_block_count")
        if isinstance(bounded, bool) or not isinstance(bounded, int) or not 1 <= bounded <= 600:
            raise RelayAdmissionError("bounded_n600_block_count must be in [1,600]")
        lane_id = payload.get("lane_id")
        run_id = payload.get("run_id")
        if not all(isinstance(value, str) and value for value in (lane_id, run_id)):
            raise RelayAdmissionError("RS1 lane/run identity differs")
        bindings_payload = payload.get("source_bindings")
        if not isinstance(bindings_payload, Mapping) or set(bindings_payload) != EXPECTED_BINDINGS:
            raise RelayAdmissionError("RS1 source binding set differs")
        if "station_bundle" not in payload:
            raise RelayAdmissionError("RS1 station_bundle declaration is required")
        station_bundle_payload = payload["station_bundle"]
        station_bundle = (
            None
            if station_bundle_payload is None
            else BoundSource.from_payload(
                station_bundle_payload,
                name="station_bundle",
            )
        )
        missing_payload = payload.get("declared_pre_se_receipts")
        if not isinstance(missing_payload, list) or len(missing_payload) != 2:
            raise RelayAdmissionError("RS1 must declare both #484 receipt identities")
        missing = tuple(
            BoundSource.from_payload(
                row,
                name=f"declared_pre_se_receipts[{index}]",
                allow_unknown_bytes=True,
            )
            for index, row in enumerate(missing_payload)
        )
        return cls(
            path=config_path,
            lane_id=str(lane_id),
            run_id=str(run_id),
            source_bindings={
                name: BoundSource.from_payload(row, name=name)
                for name, row in bindings_payload.items()
            },
            station_bundle=station_bundle,
            missing_pre_se_receipts=missing,
            hard_tail_block_count=24,
            bounded_n600_block_count=bounded,
        )

    @property
    def repo_root(self) -> Path:
        return self.path.parents[3]

    def typed_hash(self) -> str:
        return _sha256(
            _canonical_bytes(
                {
                    "schema": CONFIG_SCHEMA,
                    "lane_id": self.lane_id,
                    "run_id": self.run_id,
                    "hard_tail_block_count": self.hard_tail_block_count,
                    "bounded_n600_block_count": self.bounded_n600_block_count,
                    "source_bindings": {
                        key: value.to_payload()
                        for key, value in sorted(self.source_bindings.items())
                    },
                    "station_bundle": (
                        None
                        if self.station_bundle is None
                        else self.station_bundle.to_payload()
                    ),
                    "declared_pre_se_receipts": [
                        value.to_payload() for value in self.missing_pre_se_receipts
                    ],
                    "metric_primary": METRIC_KIND,
                    "realized_acceptance": ACCEPTANCE_AUTHORITY,
                    "score_claim": False,
                }
            )
        )


def _json_source(config: RelayAuditConfig, name: str) -> dict[str, Any]:
    payload = json.loads(config.source_bindings[name].read(config.repo_root))
    if not isinstance(payload, dict):
        raise RelayAdmissionError(f"{name} must contain a JSON object")
    return payload


def _measured_problem(
    config: RelayAuditConfig,
) -> tuple[RelayProblemV1, dict[str, Any]] | None:
    """Load an optional SHA-bound G3 station bundle into the typed solver."""

    if config.station_bundle is None:
        return None
    payload = json.loads(config.station_bundle.read(config.repo_root))
    if not isinstance(payload, dict):
        raise RelayAdmissionError("station_bundle must contain a JSON object")
    required = {
        "schema": STATION_BUNDLE_SCHEMA,
        "measurement_status": "MEASURED",
        "research_only": True,
        "score_claim": False,
        "evidence_axis": EVIDENCE_AXIS,
        "hard_tail_block_count": 24,
        "input_station_id": "range_a_input",
        "station_chain": [
            "block2_pre_se",
            "block3_pre_se",
            "rank4_head",
        ],
    }
    drift = {
        key: (payload.get(key), expected)
        for key, expected in required.items()
        if payload.get(key) != expected
    }
    if drift:
        raise RelayAdmissionError(f"measured station bundle contract differs: {drift}")
    actuator_control = payload.get("actuator_euclidean_control")
    if (
        not isinstance(actuator_control, Mapping)
        or actuator_control.get("metric_kind") != "euclidean_l2_control"
        or actuator_control.get("measurement_status") != "MEASURED"
    ):
        raise RelayAdmissionError("station bundle actuator control custody differs")

    station_rows = payload.get("stations")
    segment_rows = payload.get("segments")
    secant_rows = payload.get("continuity_secants")
    if (
        not isinstance(station_rows, list)
        or len(station_rows) != 3
        or not isinstance(segment_rows, list)
        or len(segment_rows) != 3
        or not isinstance(secant_rows, list)
        or len(secant_rows) != config.hard_tail_block_count
    ):
        raise RelayAdmissionError("measured station bundle row counts differ")

    stations: list[RelayStationV1] = []
    expected_layers = {
        "block2_pre_se": "encoder.model.blocks.1.2.se.forward_pre",
        "block3_pre_se": "encoder.model.blocks.2.2.se.forward_pre",
        "rank4_head": "POST_R_PENULTIMATE_HEAD_QUOTIENT",
    }
    for row in station_rows:
        if not isinstance(row, Mapping):
            raise RelayAdmissionError("station bundle row must be a mapping")
        station_id = row.get("station_id")
        control = row.get("euclidean_control")
        if (
            station_id not in expected_layers
            or row.get("layer_path") != expected_layers[station_id]
            or not isinstance(control, Mapping)
            or control.get("metric_kind") != "euclidean_l2_control"
            or control.get("measurement_status") != "MEASURED"
        ):
            raise RelayAdmissionError("station bundle layer/control custody differs")
        station = RelayStationV1(
            station_id=str(station_id),
            layer_path=str(row["layer_path"]),
            target_delta=row.get("target_delta"),
            metric_gram=row.get("metric_gram"),
            metric_kind=str(row.get("metric_kind")),
            evidence_sha256=str(row.get("evidence_sha256")),
            measurement_status=str(row.get("measurement_status")),
        )
        reported_l2 = control.get("target_delta_l2")
        observed_l2 = float(np.linalg.norm(station.target_delta))
        tolerance = np.finfo(np.float64).eps * max(1.0, observed_l2) * 64.0
        if (
            isinstance(reported_l2, bool)
            or not isinstance(reported_l2, (int, float))
            or not np.isfinite(float(reported_l2))
            or abs(float(reported_l2) - observed_l2) > tolerance
        ):
            raise RelayAdmissionError("station Euclidean control readback differs")
        stations.append(station)

    segments: list[RelaySegmentV1] = []
    expected_edges = (
        ("range_a_input", "block2_pre_se"),
        ("block2_pre_se", "block3_pre_se"),
        ("block3_pre_se", "rank4_head"),
    )
    for row, (source_id, target_id) in zip(segment_rows, expected_edges, strict=True):
        if (
            not isinstance(row, Mapping)
            or row.get("source_id") != source_id
            or row.get("target_id") != target_id
        ):
            raise RelayAdmissionError("station bundle segment chain differs")
        segments.append(
            RelaySegmentV1(
                segment_id=str(row.get("segment_id")),
                source_id=str(row.get("source_id")),
                target_id=str(row.get("target_id")),
                jacobian=row.get("jacobian"),
                evidence_sha256=str(row.get("evidence_sha256")),
                measurement_status=str(row.get("measurement_status")),
            )
        )

    actuator_dimension = payload.get("actuator_dimension")
    if isinstance(actuator_dimension, bool) or not isinstance(actuator_dimension, int):
        raise RelayAdmissionError("station bundle actuator dimension differs")
    problem = RelayProblemV1(
        stations=tuple(stations),
        segments=tuple(segments),
        actuator_dimension=actuator_dimension,
        actuator_metric=payload.get("actuator_metric"),
    )

    candidate_ids: set[str] = set()
    for row in secant_rows:
        if not isinstance(row, Mapping):
            raise RelayAdmissionError("continuity secant row must be a mapping")
        candidate_id = row.get("candidate_id")
        candidate_segments = row.get("segments")
        if (
            not isinstance(candidate_id, str)
            or not candidate_id
            or candidate_id in candidate_ids
            or not isinstance(candidate_segments, list)
            or len(candidate_segments) != len(problem.segments)
            or row.get("measurement_status") != "MEASURED"
        ):
            raise RelayAdmissionError("continuity secant custody differs")
        candidate_ids.add(candidate_id)
        for secant, segment in zip(candidate_segments, problem.segments, strict=True):
            if (
                not isinstance(secant, Mapping)
                or secant.get("segment_id") != segment.segment_id
                or secant.get("measurement_status") != "MEASURED"
                or not _valid_sha256(secant.get("evidence_sha256"))
            ):
                raise RelayAdmissionError("continuity secant segment custody differs")
            source_delta = np.asarray(secant.get("source_delta"), dtype=np.float64)
            realized_target = np.asarray(
                secant.get("realized_target_delta"),
                dtype=np.float64,
            )
            reported_linearized = np.asarray(
                secant.get("linearized_target_delta"),
                dtype=np.float64,
            )
            if (
                source_delta.shape != (segment.jacobian.shape[1],)
                or realized_target.shape != (segment.jacobian.shape[0],)
                or reported_linearized.shape != (segment.jacobian.shape[0],)
                or not np.all(np.isfinite(source_delta))
                or not np.all(np.isfinite(realized_target))
                or not np.all(np.isfinite(reported_linearized))
            ):
                raise RelayAdmissionError("continuity secant vector geometry differs")
            derived_linearized = segment.jacobian @ source_delta
            scale = max(
                1.0,
                float(np.linalg.norm(derived_linearized)),
                float(np.linalg.norm(realized_target)),
            )
            tolerance = np.finfo(np.float64).eps * scale * 128.0
            residual_l2 = float(np.linalg.norm(realized_target - derived_linearized))
            reported_residual = secant.get("residual_l2")
            if (
                not np.allclose(
                    reported_linearized,
                    derived_linearized,
                    rtol=0.0,
                    atol=tolerance,
                )
                or isinstance(reported_residual, bool)
                or not isinstance(reported_residual, (int, float))
                or not np.isfinite(float(reported_residual))
                or abs(float(reported_residual) - residual_l2) > tolerance
            ):
                raise RelayAdmissionError("continuity secant readback differs")
    return problem, payload


def _binding_rows(config: RelayAuditConfig) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    for name, binding in sorted(config.source_bindings.items()):
        payload = binding.read(config.repo_root)
        rows[name] = {
            **binding.to_payload(),
            "resolved_path": str(binding.resolve(config.repo_root)),
            "observed_sha256": _sha256(payload),
            "validated": True,
        }
    return rows


def _candidate_paths(config: RelayAuditConfig, binding: BoundSource) -> tuple[Path, ...]:
    original = Path(binding.path)
    name = original.name
    return (
        binding.resolve(config.repo_root),
        Path("/Volumes/VertigoDataTier/pact/experiments/results")
        / original.parent.name
        / name,
        Path("/Volumes/APDataStore/pact/experiments/results")
        / original.parent.name
        / name,
    )


def _declared_receipt_rows(config: RelayAuditConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for binding in config.missing_pre_se_receipts:
        paths = _candidate_paths(config, binding)
        matches: list[str] = []
        mismatches: list[dict[str, Any]] = []
        for path in paths:
            if not path.is_file() or path.is_symlink():
                continue
            payload = path.read_bytes()
            if (
                binding.bytes is None
                or len(payload) == binding.bytes
            ) and _sha256(payload) == binding.sha256:
                matches.append(str(path))
            else:
                mismatches.append(
                    {
                        "path": str(path),
                        "bytes": len(payload),
                        "sha256": _sha256(payload),
                    }
                )
        rows.append(
            {
                **binding.to_payload(),
                "waterfall_paths_checked": [str(path) for path in paths],
                "exact_matches": matches,
                "identity_mismatches": mismatches,
                "available": bool(matches),
            }
        )
    return rows


def build_admission_receipt(config: RelayAuditConfig) -> dict[str, Any]:
    """Rehash current stores and derive the relay execution gates."""

    bindings = _binding_rows(config)
    pre_se_memo = config.source_bindings["pre_se_locus_memo"].read(config.repo_root).decode(
        "utf-8"
    )
    at1_tracked = _json_source(config, "at1_tracked_receipt")
    at1_gaze = _json_source(config, "at1_gaze_atlas")
    sn1 = _json_source(config, "sn1_tracked_receipt")
    sn1_telemetry = config.source_bindings["sn1_telemetry"].read(config.repo_root)
    ms4_bundle = _json_source(config, "ms4_bundle")
    ms4_seg = _json_source(config, "ms4_seg_metric")
    j8f = _json_source(config, "j8f_receipt")
    pre_se_receipts = _declared_receipt_rows(config)
    measured_problem = _measured_problem(config)
    problem = None if measured_problem is None else measured_problem[0]
    station_bundle_payload = (
        None if measured_problem is None else measured_problem[1]
    )

    block2_shape = "(1,144,96,128)"
    block3_shape = "(1,288,48,64)"
    loci_defined = (
        "encoder.model.blocks.1.2.se" in pre_se_memo
        and "encoder.model.blocks.2.2.se" in pre_se_memo
        and block2_shape in pre_se_memo.replace(" ", "")
        and block3_shape in pre_se_memo.replace(" ", "")
    )
    measured_depths = at1_gaze.get("measured_relay_depths")
    at1_internal_absent = (
        measured_depths == ["scorer_plane_y", "camera_input_x"]
        and at1_gaze.get("unmeasured_internal_layers_claimed") is False
    )
    sn1_aggregate_only = (
        sn1.get("schema") == "ddm_sn1_segnet_telemetry_asymmetry_receipt.v1"
        and b"ddm_sn1_segnet_telemetry.aggregate.v1" in sn1_telemetry
        and b"intermediate_margin_fisher_gram" not in sn1_telemetry
        and b"segment_jacobian" not in sn1_telemetry
    )
    final_metric_complete = (
        ms4_bundle.get("schema") == "ddm_metric_custody_bundle.v1"
        and ms4_bundle.get("status") == "COMPLETE"
        and ms4_seg.get("schema") == "ddm_seg_metric_custody.direct_scorer_intrinsic.v2"
        and ms4_seg.get("head_rank") == 4
        and ms4_seg.get("metric_mode") == "DIRECT_SCORER_INTRINSIC_NO_ACTUATOR_INPUT"
    )
    j8f_ready = (
        j8f.get("verdict") == "READY_TO_FIRE_DDM_EVENT_CONTINUATION"
        and j8f.get("score_claim") is False
        and j8f.get("pointer_moved") is False
        and j8f.get("range_gauge_projected_arm", {})
        .get("archive", {})
        .get("parseback_exact")
        is True
    )
    tracked_at1_consistent = (
        at1_tracked.get("gaze_contraction", {})
        .get("measured_relay_depths")
        == measured_depths
    )
    raw_pre_se_available = all(row["available"] for row in pre_se_receipts)

    if problem is None:
        station_rows = [
            {
                "station_id": "range_a_input",
                "layer_path": "#580 exact receiver input projector",
                "definition_measured": True,
                "target_present": False,
                "fisher_gram_present": False,
                "dual_metric_readback_present": False,
                "status": "PROJECTOR_PRESENT_STATION_SOLVE_ROWS_ABSENT",
            },
            {
                "station_id": "block2_pre_se",
                "layer_path": "encoder.model.blocks.1.2.se.forward_pre",
                "definition_measured": loci_defined,
                "target_present": False,
                "fisher_gram_present": False,
                "dual_metric_readback_present": False,
                "status": "LOCUS_AGGREGATE_MEASURED_SEGMENT_DYNAMICS_ABSENT",
            },
            {
                "station_id": "block3_pre_se",
                "layer_path": "encoder.model.blocks.2.2.se.forward_pre",
                "definition_measured": loci_defined,
                "target_present": False,
                "fisher_gram_present": False,
                "dual_metric_readback_present": False,
                "status": "LOCUS_AGGREGATE_MEASURED_SEGMENT_DYNAMICS_ABSENT",
            },
            {
                "station_id": "rank4_head",
                "layer_path": "POST_R_PENULTIMATE_HEAD_QUOTIENT",
                "definition_measured": final_metric_complete,
                "target_present": final_metric_complete,
                "fisher_gram_present": final_metric_complete,
                "dual_metric_readback_present": final_metric_complete,
                "status": "MEASURED_FINAL_STATION_COMPLETE",
            },
        ]
        segment_rows = [
            {
                "segment_id": segment_id,
                "jacobian_present": False,
                "candidate_secants_present": False,
                "continuity_readback_present": False,
                "status": "MISSING",
            }
            for segment_id in (
                "range_a_input_to_block2_pre_se",
                "block2_pre_se_to_block3_pre_se",
                "block3_pre_se_to_rank4_head",
            )
        ]
        missing_edges = [
            "G3_TOP24_TO_BLOCK2_BLOCK3_STATION_TARGET_JOIN",
            "BLOCK2_MARGIN_FISHER_GRAM_AND_EUCLIDEAN_CONTROL_READBACK",
            "BLOCK3_MARGIN_FISHER_GRAM_AND_EUCLIDEAN_CONTROL_READBACK",
            "RANGE_A_INPUT_TO_BLOCK2_MEASURED_SEGMENT_JACOBIAN",
            "BLOCK2_TO_BLOCK3_MEASURED_SEGMENT_JACOBIAN",
            "BLOCK3_TO_RANK4_HEAD_MEASURED_SEGMENT_JACOBIAN",
            "PER_CANDIDATE_STATION_CONTINUITY_SECANTS",
        ]
        if not raw_pre_se_available:
            missing_edges.insert(
                0,
                "DECLARED_484_RAW_RECEIPTS_UNAVAILABLE_IN_STORAGE_WATERFALL",
            )
        relay_prediction = None
        direct_prediction = None
    else:
        station_rows = [
            {
                "station_id": "range_a_input",
                "layer_path": "#580 exact receiver input projector",
                "definition_measured": True,
                "target_present": False,
                "fisher_gram_present": True,
                "dual_metric_readback_present": True,
                "status": "MEASURED_INPUT_METRIC_COMPLETE",
            },
            *[
                {
                    "station_id": station.station_id,
                    "layer_path": station.layer_path,
                    "definition_measured": True,
                    "target_present": True,
                    "fisher_gram_present": True,
                    "dual_metric_readback_present": True,
                    "status": "MEASURED_STATION_COMPLETE",
                }
                for station in problem.stations
            ],
        ]
        segment_rows = [
            {
                "segment_id": segment.segment_id,
                "jacobian_present": True,
                "candidate_secants_present": True,
                "continuity_readback_present": True,
                "status": "MEASURED_COMPLETE",
            }
            for segment in problem.segments
        ]
        missing_edges = []
        relay_prediction = solve_multiple_shooting(problem)
        direct_prediction = solve_direct_final_station(problem)
    ready = (
        not missing_edges
        and loci_defined
        and at1_internal_absent
        and sn1_aggregate_only
        and final_metric_complete
        and j8f_ready
        and tracked_at1_consistent
    )
    verdict = (
        "READY_FOR_G3_TOP24_REALIZED_RADIUS_MEASUREMENT"
        if ready
        else "BLOCKED_INTERNAL_STATION_DYNAMICS_NOT_CUSTODIED"
    )
    return {
        "schema": RECEIPT_SCHEMA,
        "lane_id": config.lane_id,
        "run_id": config.run_id,
        "typed_config_hash": config.typed_hash(),
        "verdict": verdict,
        "verdict_scope": (
            "FORMULATION x current SHA-bound #484/AT1/SN1/MS4D/J8F custody; "
            "feature-relay and multiple-shooting families remain open"
        ),
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer": POINTER,
        "pointer_moved": False,
        "evidence_axis": EVIDENCE_AXIS,
        "main_landing_review_required": True,
        "metric_primary": METRIC_KIND,
        "euclidean_control_only": True,
        "acceptance_authority": ACCEPTANCE_AUTHORITY,
        "execution": {
            "g3_top24_run": (
                "PREDICTIVE_SOLVES_COMPLETE_AWAIT_REALIZED_ENDPOINT_LADDER"
                if ready
                else "NOT_RUN_INPUT_ADMISSION_REFUSED"
            ),
            "bounded_n600_run": (
                "NOT_RUN_G3_REALIZED_ENDPOINT_LADDER_OWED"
                if ready
                else "NOT_RUN_G3_FIRST_GATE_REFUSED"
            ),
            "receiver_scorer_replay": (
                "NOT_RUN_REALIZED_ENDPOINTS_OWED"
                if ready
                else "NOT_RUN_NO_RELAY_CANDIDATE"
            ),
            "relay_radius": None,
            "direct_radius": None,
            "delta_d_seg": None,
            "delta_d_pose": None,
            "delta_bytes": None,
            "score_claim": False,
        },
        "predicted_solve": {
            "relay": relay_prediction,
            "direct": direct_prediction,
            "used_for_acceptance": False,
        },
        "station_plan": {
            "relay_station_count": 3,
            "input_boundary": "range_a_input",
            "stations": station_rows[1:],
            "all_rows": station_rows,
        },
        "segment_plan": segment_rows,
        "source_interpretation": {
            "pre_se_loci_defined_and_measured_aggregate": loci_defined,
            "declared_pre_se_raw_receipts_available": raw_pre_se_available,
            "at1_tracked_and_ssd_depths_consistent": tracked_at1_consistent,
            "at1_measured_relay_depths": measured_depths,
            "at1_internal_station_layers_explicitly_unmeasured": at1_internal_absent,
            "sn1_telemetry_is_aggregate_without_internal_fisher_or_jacobian": (
                sn1_aggregate_only
            ),
            "ms4_final_rank4_station_complete": final_metric_complete,
            "j8f_end_realized_harness_ready": j8f_ready,
            "measured_g3_station_bundle_present": problem is not None,
        },
        "gates": {
            "hard_tail_block_count": config.hard_tail_block_count,
            "bounded_n600_block_count": config.bounded_n600_block_count,
            "all_internal_station_targets_present": problem is not None,
            "all_internal_fisher_grams_present": problem is not None,
            "all_segment_jacobians_present": problem is not None,
            "all_continuity_secants_present": problem is not None,
            "end_realized_harness_present": j8f_ready,
            "admitted": ready,
        },
        "missing_edges": missing_edges,
        "declared_pre_se_receipts": pre_se_receipts,
        "source_bindings": bindings,
        "station_bundle": (
            None
            if config.station_bundle is None
            else {
                **config.station_bundle.to_payload(),
                "resolved_path": str(
                    config.station_bundle.resolve(config.repo_root)
                ),
                "observed_sha256": config.station_bundle.sha256,
                "validated": True,
                "schema": station_bundle_payload["schema"],
                "candidate_count": len(
                    station_bundle_payload["continuity_secants"]
                ),
            }
        ),
        "next_exact_measurement": (
            "Materialize one SHA-bound G3-top24 station bundle carrying, for the same "
            "J8F integer candidate rows, block2/block3 target deltas, categorical "
            "margin-Fisher Grams, Euclidean controls, input->block2, block2->block3, "
            "block3->rank4 Jacobians, and realized continuity secants. Re-run this gate; "
            "only then solve and send the winning endpoint to J8F n600 replay."
        ),
        "no_family_kill": True,
    }


def _blocker(receipt: Mapping[str, Any], receipt_path: Path) -> dict[str, Any]:
    payload = receipt_path.read_bytes()
    return {
        "schema": BLOCKER_SCHEMA,
        "verdict": receipt["verdict"],
        "verdict_scope": receipt["verdict_scope"],
        "receipt": {
            "path": str(receipt_path),
            "bytes": len(payload),
            "sha256": _sha256(payload),
        },
        "missing_edges": receipt["missing_edges"],
        "next_exact_measurement": receipt["next_exact_measurement"],
        "execution_allowed": False,
        "score_claim": False,
        "pointer": POINTER,
        "pointer_moved": False,
        "main_landing_review_required": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-directory", required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)

    config = RelayAuditConfig.from_path(args.config)
    output = Path(args.output_directory).resolve()
    receipt = build_admission_receipt(config)
    receipt_path = output / "receipt.json"
    _write_json(receipt_path, receipt)
    if receipt["verdict"] != "READY_FOR_G3_TOP24_RELAY_MEASUREMENT":
        _write_json(output / "BLOCKER.json", _blocker(receipt, receipt_path))
    print(
        json.dumps(
            {
                "verdict": receipt["verdict"],
                "missing_edge_count": len(receipt["missing_edges"]),
                "g3_top24_run": receipt["execution"]["g3_top24_run"],
                "bounded_n600_run": receipt["execution"]["bounded_n600_run"],
                "validate_only": bool(args.validate_only),
                "receipt_path": str(receipt_path),
            },
            sort_keys=True,
        )
    )
    return (
        0
        if receipt["verdict"]
        == "READY_FOR_G3_TOP24_REALIZED_RADIUS_MEASUREMENT"
        else 2
    )


if __name__ == "__main__":
    raise SystemExit(main())
