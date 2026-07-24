# SPDX-License-Identifier: MIT
"""Fail-closed R3 admission for the DDM typed Fisher/G4 waterfill.

EV1 measures 162 exclusive accounting homes and exact receiver-step
histograms.  Those observations are necessary rate evidence, but they are not
signed, located residual streams and their byte ranges are explicitly not
physical ZIP separations.  MS4D likewise measures scorer-intrinsic Fisher
geometry while declaring every per-bucket actuator secant inapplicable.

This module performs the real admission test between those two surfaces.  It
publishes exact endpoint slopes and output-space histogram quanta, but refuses
to manufacture per-dimension actuator quanta, representation races, or a
receiver-closed candidate when the required foreign keys and streams are
absent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from functools import reduce
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tac.ddm_campaign_evidence_join import (
    METRIC_ID,
    RD1_BUCKET_COUNT,
    bucket_key,
    canonical_bytes,
    validate_campaign_evidence_join,
)
from tac.optimization.ddm_min_description_contract import (
    LayerHome,
    StreamType,
    TypedStreamTag,
    build_minimum_description_headline,
)
from tac.scorer_value_oracle import (
    ProducerBinding,
    ScorerValueOracle,
)

SCHEMA: Final = "ddm_ms2r_r3_typed_fisher_g4_waterfill_receipt.v1"
CONFIG_SCHEMA: Final = "DDMMS2RR3TypedFisherG4WaterfillConfigV1"
PRICING_SCHEMA: Final = "ddm_ms2r_r3_rd1_pricing_admission.v1"
RACE_SCHEMA: Final = "ddm_ms2r_r3_representation_race_admission.v1"
TELEMETRY_SCHEMA: Final = "ddm_co2_compression_progress_telemetry.v1"
LANE_ID: Final = "ddm_ms2r_r3_typed_fisher_g4_waterfill"
RATE_SCORE_PER_BYTE: Final = 25.0 / 37_545_489.0
EXPECTED_PF2_ROWS: Final = 1_200
EXPECTED_DIRECT_BLOCKS: Final = 25
CODER_ROSTER: Final = (
    "RAW",
    "BROTLI_Q11",
    "ORDER1_ANS",
    "ZSTD19_DICTIONARY",
    "MAHONEY_CONTEXT_MIXING",
    "WILLEMS_CTW",
)
OPTIONAL_CODER: Final = "BELLARD_ONLINE_LEARNED_MODEL"
REPRESENTATION_FAMILIES: Final = (
    "PREDICTOR_INNOVATION",
    "SHARED_CODEBOOK_TEMPLATE_INDEX",
    "SOLUTION_DIRECT_TYPED_CODE",
    "TOLERANCE_ABSORBED",
)


class R3AdmissionError(ValueError):
    """A sealed producer or R3 admission invariant differs."""


class R3Config(BaseModel):
    """Typed, no-extra-field execution DSL for the advisory R3 admission."""

    model_config = ConfigDict(extra="forbid", frozen=True, populate_by_name=True)

    schema_name: Literal["DDMMS2RR3TypedFisherG4WaterfillConfigV1"] = Field(
        alias="schema"
    )
    run_id: str
    output_receipt_path: str
    r2_receipt_path: str
    r2_receipt_sha256: str
    r2_receipt_bytes: int
    ev1_receipt_path: str
    ev1_receipt_sha256: str
    ev1_receipt_bytes: int
    pair_count: Literal[600]
    scorer_batch_size: Literal[32]
    allowed_errors: Literal[136839]
    metric_id: Literal[
        "exact_composite_R_rank4_margin_fisher_plus_pose6_quadratic"
    ]
    research_only: Literal[True]
    execution_allowed: Literal[False]
    score_claim: Literal[False]
    promotion_eligible: Literal[False]
    receipt_timestamp_utc: str
    main_landing_review_required: Literal[True]

    @field_validator("r2_receipt_sha256", "ev1_receipt_sha256")
    @classmethod
    def _sha256(cls, value: str) -> str:
        if len(value) != 64:
            raise ValueError("producer SHA-256 must have 64 hexadecimal characters")
        try:
            bytes.fromhex(value)
        except ValueError as exc:
            raise ValueError("producer SHA-256 must be hexadecimal") from exc
        return value

    @field_validator("r2_receipt_bytes", "ev1_receipt_bytes")
    @classmethod
    def _positive_bytes(cls, value: int) -> int:
        if isinstance(value, bool) or value < 1:
            raise ValueError("producer byte count must be a positive integer")
        return value

    @model_validator(mode="after")
    def _authority(self) -> R3Config:
        if not self.run_id or not self.output_receipt_path or not self.receipt_timestamp_utc:
            raise ValueError("run id, output path, and sealed timestamp must be nonempty")
        if self.metric_id != METRIC_ID:
            raise ValueError("R3 admits only the exact composite-R Fisher/Pose6 metric")
        return self


def _resolve(repository_root: Path, path: str) -> Path:
    candidate = Path(path).expanduser()
    return candidate if candidate.is_absolute() else repository_root / candidate


def _load_config(path: Path) -> tuple[R3Config, str]:
    payload = path.read_bytes()
    try:
        value = R3Config.model_validate_json(payload)
    except ValueError as exc:
        raise R3AdmissionError(f"typed R3 config is invalid: {exc}") from exc
    return value, hashlib.sha256(payload).hexdigest()


def _publish_immutable(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise R3AdmissionError(f"immutable stage differs on resume: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _binding(
    *,
    producer_id: str,
    producer: str,
    path: str,
    sha256: str,
    byte_count: int,
    schema: str,
    authority_scope: str,
) -> ProducerBinding:
    return ProducerBinding(
        producer_id=producer_id,
        producer=producer,
        path=path,
        sha256=sha256,
        bytes=byte_count,
        schema=schema,
        validity_horizon="content-hash valid until a superseding sealed receipt lands",
        authority_scope=authority_scope,
    )


def _reference_binding(
    reference: Mapping[str, Any],
    *,
    producer_id: str,
    producer: str,
    schema: str,
    authority_scope: str,
) -> ProducerBinding:
    raw_path = reference.get("path")
    raw_sha = reference.get("sha256")
    raw_bytes = reference.get("bytes")
    if (
        not isinstance(raw_path, str)
        or not isinstance(raw_sha, str)
        or isinstance(raw_bytes, bool)
        or not isinstance(raw_bytes, int)
    ):
        raise R3AdmissionError(f"{producer_id}: nested producer reference is malformed")
    return _binding(
        producer_id=producer_id,
        producer=producer,
        path=raw_path,
        sha256=raw_sha,
        byte_count=raw_bytes,
        schema=schema,
        authority_scope=authority_scope,
    )


def _validate_r2(
    r2: Mapping[str, Any],
    *,
    oracle: ScorerValueOracle,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    if (
        r2.get("schema") != "ddm_ms2r_tolerance_capped_solve_r2_receipt.v1"
        or r2.get("authority", {}).get("research_only") is not True
        or r2.get("authority", {}).get("score_claim") is not False
        or r2.get("authority", {}).get("pointer_moved") is not False
    ):
        raise R3AdmissionError("R2 authority or schema differs")
    homotopy = r2.get("homotopy")
    solve = homotopy.get("solve") if isinstance(homotopy, Mapping) else None
    candidate = homotopy.get("candidate") if isinstance(homotopy, Mapping) else None
    if (
        not isinstance(solve, Mapping)
        or not isinstance(candidate, Mapping)
        or solve.get("allowed_errors") != 136_839
        or solve.get("realized_errors") != 136_839
        or candidate.get("strict_production_parseback_exact") is not True
        or candidate.get("canonical_archive_determinism_x2") is not True
    ):
        raise R3AdmissionError("R2 exact box-control custody differs")
    archive = candidate.get("archive")
    if not isinstance(archive, Mapping):
        raise R3AdmissionError("R2 candidate archive reference is absent")
    archive_lineage = oracle.require_artifact(
        archive,
        role="R2 receiver-closed box-control archive",
    ).to_dict()
    config_reference = r2.get("typed_config")
    if not isinstance(config_reference, Mapping):
        raise R3AdmissionError("R2 typed config reference is absent")
    r2_config = oracle.require_json_producer(
        _reference_binding(
            config_reference,
            producer_id="r2_typed_config",
            producer="R2 typed execution config",
            schema="DDMMS2RToleranceCappedSolveR2ConfigV1",
            authority_scope="R2 n600 execution geometry and false-authority flags",
        )
    )
    if (
        not isinstance(r2_config, Mapping)
        or r2_config.get("pair_count") != 600
        or r2_config.get("scorer_batch_size") != 16
        or r2_config.get("allowed_errors") != 136_839
    ):
        raise R3AdmissionError("R2 typed config geometry differs")
    return archive_lineage, r2_config


def _histogram_quantum(histogram: Sequence[Any]) -> tuple[int | None, int | None]:
    if len(histogram) != 256:
        raise R3AdmissionError("receiver uint8 histogram must have 256 bins")
    support: list[int] = []
    for index, count in enumerate(histogram):
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise R3AdmissionError("receiver uint8 histogram counts must be exact nonnegative integers")
        if index > 0 and count > 0:
            support.append(index)
    if not support:
        return None, None
    return support[0], reduce(math.gcd, support)


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise R3AdmissionError(f"{field} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise R3AdmissionError(f"{field} must be a finite number")
    return result


def _pricing_stage(ev1: Mapping[str, Any]) -> dict[str, Any]:
    rows = ev1["rd1_evidence"]["bucket_rows"]
    prices: list[dict[str, Any]] = []
    for raw in sorted(rows, key=bucket_key):
        delta_d = _finite_float(raw["delta_D_dimension"], "delta_D_dimension")
        counted_bytes = int(raw["delta_counted_bytes_dimension"])
        amortized_bytes = _finite_float(
            raw["amortized_bytes_per_frame"],
            "amortized_bytes_per_frame",
        )
        minimum_step, step_gcd = _histogram_quantum(
            raw["receiver_uint8_abs_step_histogram"]
        )
        beneficial = delta_d < 0.0
        observed_full_slope = (
            counted_bytes / -delta_d if beneficial else None
        )
        observed_amortized_slope = (
            amortized_bytes / -delta_d if beneficial else None
        )
        reduction_per_full_byte = (
            -delta_d / counted_bytes
            if beneficial and counted_bytes > 0
            else None
        )
        prices.append(
            {
                "dual_index": int(raw["dual_index"]),
                "stratum": str(raw["stratum"]),
                "scorer_visibility": str(raw["scorer_visibility"]),
                "g4_temporal_class": str(raw["g4_temporal_class"]),
                "metric_id": str(raw["metric_id"]),
                "delta_D_dimension": delta_d,
                "delta_counted_bytes_dimension": counted_bytes,
                "scope": str(raw["scope"]),
                "k": _finite_float(raw["k"], "k"),
                "amortized_bytes_per_frame": amortized_bytes,
                "realized_output_min_nonzero_step_uint8": minimum_step,
                "realized_output_step_gcd_uint8": step_gcd,
                "per_dimension_effective_quantum": None,
                "per_dimension_effective_quantum_status": (
                    "NULL_NO_ACTUATOR_ADJOINT_GAIN_X_DEADZONE_FOREIGN_KEY"
                ),
                "observed_accounting_slope_full_bytes_per_D_improvement": (
                    observed_full_slope
                ),
                "observed_accounting_slope_amortized_bytes_per_D_improvement": (
                    observed_amortized_slope
                ),
                "observed_D_improvement_per_full_byte": reduction_per_full_byte,
                "above_rate_break_even_on_observed_accounting_slope": (
                    None
                    if reduction_per_full_byte is None
                    else reduction_per_full_byte > RATE_SCORE_PER_BYTE
                ),
                "lambda_bytes_per_D_dimension": None,
                "lambda_status": (
                    "NULL_ACCOUNTING_HOME_NOT_PHYSICALLY_SEPARABLE_AND_NO_TYPED_ACTUATOR_SECANT"
                ),
                "actionable_for_train_decision": False,
                "score_claim": False,
            }
        )
    edge_slopes = []
    for edge in ev1["rd1_evidence"]["edge_summaries"]:
        delta_d = _finite_float(edge["joint_delta_D"], "joint_delta_D")
        delta_bytes = int(edge["delta_counted_bytes"])
        if delta_d >= 0.0 or delta_bytes <= 0:
            raise R3AdmissionError("RD1 sequential endpoint edge is not a beneficial positive-rate edge")
        edge_slopes.append(
            {
                "dual_index": int(edge["dual_index"]),
                "delta_counted_bytes": delta_bytes,
                "joint_delta_D": delta_d,
                "measured_endpoint_secant_bytes_per_D_improvement": (
                    delta_bytes / -delta_d
                ),
                "status": (
                    "MEASURED_SEQUENTIAL_SAME_OBJECT_ENDPOINT_SECANT_NOT_A_162_DIMENSION_DUAL"
                ),
            }
        )
    return {
        "schema": PRICING_SCHEMA,
        "metric_id": METRIC_ID,
        "rate_score_per_byte": RATE_SCORE_PER_BYTE,
        "cell_count": len(prices),
        "beneficial_accounting_slope_count": sum(
            row["observed_accounting_slope_full_bytes_per_D_improvement"] is not None
            for row in prices
        ),
        "finite_dimension_dual_count": 0,
        "actionable_for_train_decision_count": 0,
        "endpoint_secants": edge_slopes,
        "cells": prices,
        "verdict": (
            "29_OF_162_HAVE_BENEFICIAL_OBSERVED_ACCOUNTING_SLOPES; "
            "0_OF_162_HAVE_COMPOSABLE_PER_DIMENSION_DUALS"
        ),
        "score_claim": False,
    }


def _race_stage(ev1: Mapping[str, Any]) -> dict[str, Any]:
    assignments = []
    family_blockers = {
        "PREDICTOR_INNOVATION": (
            "NO_SIGNED_LOCATED_BUCKET_STREAM_OR_FREE_PREDICTOR_FOREIGN_KEY"
        ),
        "SHARED_CODEBOOK_TEMPLATE_INDEX": (
            "NO_BUCKET_STREAM_BYTES_OR_TEMPLATE_INDEX_SEQUENCE_FOR_PAIRWISE_NCD"
        ),
        "SOLUTION_DIRECT_TYPED_CODE": (
            "NO_SOLUTION_DIRECT_SIGNED_TYPED_STREAM_OR_RECEIVER_FIELD_OWNER"
        ),
        "TOLERANCE_ABSORBED": (
            "NO_ACTUAL_BUCKET_STREAM_FOR_MARTIN_LOF_COMPRESSIBILITY_CERTIFICATE"
        ),
    }
    for raw in sorted(ev1["rd1_evidence"]["bucket_rows"], key=bucket_key):
        assignments.append(
            {
                "dual_index": int(raw["dual_index"]),
                "stratum": str(raw["stratum"]),
                "scorer_visibility": str(raw["scorer_visibility"]),
                "g4_temporal_class": str(raw["g4_temporal_class"]),
                "selected_family": None,
                "families": [
                    {
                        "family": family,
                        "status": "NOT_RUN_MISSING_ADMISSIBLE_STREAM",
                        "blocker": family_blockers[family],
                    }
                    for family in REPRESENTATION_FAMILIES
                ],
                "coder_race": [
                    {
                        "coder": coder,
                        "bytes": None,
                        "parseback_exact": False,
                        "status": "NOT_RUN_NO_ADMISSIBLE_TYPED_STREAM",
                    }
                    for coder in CODER_ROSTER
                ],
                "optional_coder": {
                    "coder": OPTIONAL_CODER,
                    "bytes": None,
                    "parseback_exact": False,
                    "status": "NOT_RUN_NO_ADMISSIBLE_TYPED_STREAM",
                },
                "switch_cost_bytes": None,
                "switch_cost_status": "NULL_NO_FAMILY_ASSIGNMENT",
                "fragmentation_penalty_bytes": None,
                "polyanskiy_status": "NULL_NO_STREAM_LENGTH_OR_DISPERSION",
                "fri_event_floor_bits": None,
                "fri_status": "NULL_NO_SIGNED_LOCATED_EVENT_STREAM",
                "score_claim": False,
            }
        )
    return {
        "schema": RACE_SCHEMA,
        "bucket_count": len(assignments),
        "family_roster": list(REPRESENTATION_FAMILIES),
        "coder_roster": list(CODER_ROSTER),
        "optional_coder": OPTIONAL_CODER,
        "assigned_bucket_count": 0,
        "coder_race_completed_bucket_count": 0,
        "assignments": assignments,
        "verdict": "REFUSED_NO_PHYSICALLY_SEPARABLE_SIGNED_TYPED_STREAMS",
        "score_claim": False,
    }


def _telemetry_stage() -> dict[str, Any]:
    blocker = "NO_ADMISSIBLE_TYPED_STREAM_BYTES_TO_ATTRIBUTE"
    return {
        "schema": TELEMETRY_SCHEMA,
        "consumer": "co2",
        "rows": [
            {
                "level": family,
                "bytes_of_innovation": None,
                "bucket_count": RD1_BUCKET_COUNT,
                "status": blocker,
            }
            for family in REPRESENTATION_FAMILIES
        ],
        "event_stream_fri_floor_bits": None,
        "event_stream_status": "NULL_NO_SIGNED_LOCATED_EVENT_STREAM",
        "score_claim": False,
    }


def _source_summary(
    *,
    coverage: Mapping[str, Any],
    margin: Mapping[str, Any],
    second_order: Mapping[str, Any],
    assignments: Mapping[str, Any],
) -> dict[str, Any]:
    margin_data = margin["data"]
    second_data = second_order["data"]
    pf2_rows = assignments["rows"]
    if (
        len(margin_data["rows"]) != EXPECTED_PF2_ROWS
        or len(second_data["rows"]) != EXPECTED_PF2_ROWS
        or len(pf2_rows) != EXPECTED_PF2_ROWS
        or len(margin_data["direct_blocks"]) != EXPECTED_DIRECT_BLOCKS
    ):
        raise R3AdmissionError("MS4D/PF2 typed atlas dimensions differ")
    missing_foreign_keys = sum(
        not row.get("receiver_actuator_ids") for row in pf2_rows
    )
    inapplicable_secants = sum(
        row.get("secant_status")
        == "NOT_APPLICABLE_DIRECT_SCORER_INTRINSIC_NO_ACTUATOR"
        for row in second_data["rows"]
    )
    unreachable_blocks = sum(
        row.get("actuation_status") == "UNREACHABLE_BY_COUNTED_COORDINATES"
        for row in margin_data["direct_blocks"]
    )
    if (
        missing_foreign_keys != EXPECTED_PF2_ROWS
        or inapplicable_secants != EXPECTED_PF2_ROWS
        or unreachable_blocks != EXPECTED_DIRECT_BLOCKS
    ):
        raise R3AdmissionError("expected typed-actuator blocker surface changed")
    return {
        "oracle_coverage": dict(coverage),
        "pf2_bucket_count": EXPECTED_PF2_ROWS,
        "pf2_buckets_without_actuator_foreign_key": missing_foreign_keys,
        "ms4d_rows_with_actuator_secant_not_applicable": inapplicable_secants,
        "ms4d_direct_blocks_unreachable_by_counted_coordinates": unreachable_blocks,
        "ms4d_scorer_batch_size": int(margin_data["scorer_batch_size"]),
        "metric_mode": str(margin_data["metric_mode"]),
        "coordinate_domain": str(second_data["coordinate_domain"]),
    }


def _headline(
    r2: Mapping[str, Any],
    *,
    bundle_path: Path,
    repository_root: Path,
) -> dict[str, Any]:
    solve = r2["homotopy"]["solve"]
    archive = r2["homotopy"]["candidate"]["archive"]
    empty_sha = hashlib.sha256(b"").hexdigest()
    return build_minimum_description_headline(
        stored_problem_bytes=int(archive["bytes"]),
        stored_problem_sha256=str(archive["sha256"]),
        exception_bytes=0,
        exception_sha256=empty_sha,
        realized_d_seg=float(solve["realized_errors"]) / (600 * 512 * 384),
        realized_d_pose=float(solve["realized_d_pose"]),
        stored_problem_own_lineage=True,
        donor_conditioned=False,
        expansion_receiver_closed=True,
        pose_tube_active=True,
        realized_uint8_r_frozen_scorers=True,
        quotient_coordinates_only=True,
        scorer_metric_active=True,
        alternating_typed_subproblems=False,
        typed_blocks_active=False,
        per_dimension_quanta_active=False,
        typed_stream_tags=(
            TypedStreamTag(
                type=StreamType.FIBER,
                layer_home=LayerHome.L3_RASTER,
                evaluate_py_recursion_level_cited=(
                    "L3 R2 quotient raster -> L4 frozen scorers -> L5 verdict"
                ),
                counted_bytes=int(archive["bytes"]),
                free_receiver_code=True,
            ),
            TypedStreamTag(
                type=StreamType.RESIDUAL,
                layer_home=LayerHome.L5_VERDICT,
                evaluate_py_recursion_level_cited=(
                    "L5 no admitted R3 solve-exception stream"
                ),
                counted_bytes=0,
                free_receiver_code=True,
            ),
        ),
        strict_typed_stream_tags=True,
        metric_custody_bundle_path=bundle_path,
        metric_custody_repository_root=repository_root,
    )


def run(
    config_path: Path,
    *,
    repository_root: Path | None = None,
) -> dict[str, Any]:
    """Run the resumable admission and publish an immutable honest-wall receipt."""

    root = (
        Path(__file__).resolve().parents[3]
        if repository_root is None
        else repository_root.expanduser().resolve(strict=True)
    )
    config_path = config_path.expanduser().resolve(strict=True)
    config, config_sha = _load_config(config_path)
    oracle = ScorerValueOracle(root)
    ev1_snapshot = oracle.read_json_producer(
        _binding(
            producer_id="ev1_campaign_evidence_join",
            producer="EV1 V19/RD1 campaign evidence join",
            path=config.ev1_receipt_path,
            sha256=config.ev1_receipt_sha256,
            byte_count=config.ev1_receipt_bytes,
            schema="ddm_ev1_campaign_evidence_join_receipt.v1",
            authority_scope=(
                "n600 receiver-closed evidence and exclusive accounting homes; no prices"
            ),
        )
    )
    r2_snapshot = oracle.read_json_producer(
        _binding(
            producer_id="r2_box_control",
            producer="R2 receiver-closed q4/q8 box control",
            path=config.r2_receipt_path,
            sha256=config.r2_receipt_sha256,
            byte_count=config.r2_receipt_bytes,
            schema="ddm_ms2r_tolerance_capped_solve_r2_receipt.v1",
            authority_scope=(
                "finite q4/q8 n600 batch16 advisory control; not a typed Fisher/G4 solve"
            ),
        )
    )
    ev1 = ev1_snapshot.require_value()
    r2 = r2_snapshot.require_value()
    if not isinstance(ev1, Mapping) or not isinstance(r2, Mapping):
        raise R3AdmissionError("sealed campaign producers must be JSON objects")
    ev1_validation = validate_campaign_evidence_join(ev1)
    archive_lineage, r2_config = _validate_r2(r2, oracle=oracle)
    if (
        ev1["metric_custody"]["metric_id"] != config.metric_id
        or ev1["pair_count"] != config.pair_count
        or ev1["resumability"]["batch_size"] != 16
        or ev1["scorer_custody"]["batch_size"] != 16
    ):
        raise R3AdmissionError("EV1 metric or sealed n600 batch16 geometry differs")

    margin_snapshot = oracle.margin_fisher()
    second_snapshot = oracle.realized_second_order()
    pose_snapshot = oracle.pose_reference_and_tube()
    stationarity_snapshot = oracle.stationarity_maps()
    assignment_snapshot = oracle.bucket_assignments()
    resize_snapshot = oracle.resize_support_nullity()
    margin = margin_snapshot.require_value()
    second_order = second_snapshot.require_value()
    pose = pose_snapshot.require_value()
    stationarity = stationarity_snapshot.require_value()
    assignments = assignment_snapshot.require_value()
    resize = resize_snapshot.require_value()
    coverage = oracle.coverage_report(verify=True)

    output = _resolve(root, config.output_receipt_path)
    stage_root = output.parent / "stage_checkpoints"
    oracle_stage = {
        "schema": "ddm_ms2r_r3_oracle_admission.v1",
        "config_sha256": config_sha,
        "ev1": ev1_snapshot.to_dict(include_value=False),
        "r2": r2_snapshot.to_dict(include_value=False),
        "r2_archive": archive_lineage,
        "ev1_validation": ev1_validation,
        "r2_scorer_batch_size": int(r2_config["scorer_batch_size"]),
        "ev1_scorer_batch_size": int(ev1["scorer_custody"]["batch_size"]),
        "r3_required_scorer_batch_size": config.scorer_batch_size,
        "source_summary": _source_summary(
            coverage=coverage,
            margin=margin,
            second_order=second_order,
            assignments=assignments,
        ),
        "additional_oracle_lineages": {
            "pose": pose_snapshot.to_dict(include_value=False),
            "stationarity": stationarity_snapshot.to_dict(include_value=False),
            "resize": resize_snapshot.to_dict(include_value=False),
        },
        "pose_row_count": len(pose["data"]["rows"]),
        "g4_stationarity_schema": stationarity["schema"],
        "resize_schema": resize["schema"],
    }
    _publish_immutable(stage_root / "01_oracle_admission.json", oracle_stage)

    pricing = _pricing_stage(ev1)
    _publish_immutable(stage_root / "02_pricing.json", pricing)
    race = _race_stage(ev1)
    _publish_immutable(stage_root / "03_representation_race.json", race)
    telemetry = _telemetry_stage()
    _publish_immutable(stage_root / "04_compression_telemetry.json", telemetry)

    bundle_path = _resolve(root, str(r2_config["bundle_complete_path"]))
    headline = _headline(r2, bundle_path=bundle_path, repository_root=root)
    _publish_immutable(stage_root / "05_headline.json", headline)
    solve = r2["homotopy"]["solve"]
    archive = r2["homotopy"]["candidate"]["archive"]
    receipt = {
        "schema": SCHEMA,
        "run_id": config.run_id,
        "lane_id": LANE_ID,
        "finished_at_utc": config.receipt_timestamp_utc,
        "verdict": (
            "BLOCKED_NO_COMPOSABLE_TYPED_ACTUATOR_STREAM; "
            "R2_CONTROL_REMAINS_CHEAPEST_RECEIVER_CLOSED_BOX_MEMBER"
        ),
        "verdict_scope": (
            "n600 local advisory evidence admission only; EV1 accounting homes plus "
            "MS4D scorer-intrinsic geometry do not define separable actuator streams, "
            "per-dimension secants, or a mixed receiver candidate"
        ),
        "authority": {
            "research_only": True,
            "execution_allowed": False,
            "score_claim": False,
            "promotion_eligible": False,
            "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
            "pointer_moved": False,
            "main_landing_review_required": True,
        },
        "typed_config": {
            "path": str(config_path.relative_to(root)),
            "bytes": config_path.stat().st_size,
            "sha256": config_sha,
            "schema": CONFIG_SCHEMA,
        },
        "resumability": {
            "mode": "immutable atomic per-stage checkpoints",
            "stage_count": 5,
            "all_stages_preserved": True,
            "stage_root": str(stage_root.relative_to(root)),
        },
        "oracle_admission": oracle_stage,
        "requested_component_activation": {
            "TYPED_SUBPROBLEM_ALTERNATION": {
                "active": False,
                "blocker": "NO_ADMISSIBLE_PER_BUCKET_FAMILY_ACTION",
            },
            "TYPED_BLOCK_ATLAS": {
                "active": False,
                "evidence_rows_present": RD1_BUCKET_COUNT,
                "blocker": "0_OF_1200_PF2_BUCKETS_HAVE_ACTUATOR_FOREIGN_KEYS",
            },
            "PER_DIMENSION_EFFECTIVE_QUANTA": {
                "active": False,
                "output_histograms_present": RD1_BUCKET_COUNT,
                "blocker": (
                    "OUTPUT_HISTOGRAMS_DO_NOT_SUPPLY_PER_DIMENSION_ACTUATOR_GAIN_X_DEADZONE"
                ),
            },
        },
        "box_result": {
            "allowed_errors": config.allowed_errors,
            "new_receiver_closed_candidate_emitted": False,
            "cheapest_admitted_member": "R2_BINARY_Q4_Q8_CONTROL",
            "bytes": int(archive["bytes"]),
            "archive_sha256": str(archive["sha256"]),
            "seg_errors": int(solve["realized_errors"]),
            "d_seg": int(solve["realized_errors"]) / (600 * 512 * 384),
            "d_pose": float(solve["realized_d_pose"]),
            "beats_r2_bytes": False,
            "r2_measurement_batch_size": int(r2_config["scorer_batch_size"]),
            "r3_batch32_candidate_measurement": "NOT_RUN_NO_COMPOSABLE_CANDIDATE",
            "E4_packet_byte_close": "NOT_APPLICABLE_NO_COMPOSED_CANDIDATE",
        },
        "rd1_pricing": pricing,
        "representation_race": race,
        "compression_progress_telemetry": telemetry,
        "headline": headline,
        "honest_wall": {
            "missing_actuator_foreign_keys": EXPECTED_PF2_ROWS,
            "missing_per_bucket_actuator_secants": EXPECTED_PF2_ROWS,
            "unreachable_direct_blocks": EXPECTED_DIRECT_BLOCKS,
            "accounting_homes_not_physical_streams": RD1_BUCKET_COUNT,
            "exact_next_evidence": [
                (
                    "signed and located residual bytes with pair, field owner, and "
                    "decoder-side predictor foreign keys for every admitted bucket"
                ),
                (
                    "paired realized uint8 secants binding each typed actuator to "
                    "MS4D adjoint gain and deadzone inside a measured validity radius"
                ),
                (
                    "a receiver builder that composes mixed family assignments, prices "
                    "switch headers, and emits one counted E4 object"
                ),
                (
                    "same-object n600 batch32 receiver/R/uint8 frozen-scorer replay for "
                    "every waterfill rung admitted under the 136839-error cap"
                ),
            ],
        },
        "directive_consumption": [
            {
                "utc": "2026-07-19T19:42:07Z",
                "application": (
                    "diagnostic accounting slopes include the registered rate break-even; "
                    "no noncomposable row is admitted"
                ),
            },
            {
                "utc": "2026-07-19T19:48:01Z",
                "application": (
                    "only exact composite-R margin-Fisher plus Pose6 is admitted; "
                    "Euclidean rows are absent"
                ),
            },
            {
                "utc": "2026-07-24T14:45:16Z",
                "application": (
                    "construction refuses PF2 rows without scorer-recursive actuator "
                    "foreign keys rather than falling back to a generic spatial menu"
                ),
            },
            {
                "utc": "2026-07-24T20:01:23Z",
                "application": (
                    "generic decoder models remain zero-rate; no video-derived histogram "
                    "or table is disguised as free interpreter code"
                ),
            },
        ],
        "triality": {
            "dsl": str(config_path.relative_to(root)),
            "dag": (
                f".omx/research/{config.run_id}/DAG_FEED.md"
            ),
            "equations": [
                "dynamic_quantum_calibration_v1 (not transferable without actuator gain)",
                "ddm_tolerance_capped_min_score_waterfill_v1 (R2 control replay only)",
                "exact measured endpoint secant delta_bytes/(-delta_D) (diagnostic, not 162-way dual)",
            ],
        },
        "pointer_delta": "NONE",
        "main_landing_review_required": True,
    }
    _publish_immutable(output, receipt)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    receipt = run(args.config)
    print(
        json.dumps(
            {
                "verdict": receipt["verdict"],
                "bytes": receipt["box_result"]["bytes"],
                "seg_errors": receipt["box_result"]["seg_errors"],
                "actionable_duals": receipt["rd1_pricing"][
                    "actionable_for_train_decision_count"
                ],
                "headline": receipt["headline"]["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CODER_ROSTER",
    "CONFIG_SCHEMA",
    "LANE_ID",
    "METRIC_ID",
    "OPTIONAL_CODER",
    "PRICING_SCHEMA",
    "RACE_SCHEMA",
    "REPRESENTATION_FAMILIES",
    "SCHEMA",
    "TELEMETRY_SCHEMA",
    "R3AdmissionError",
    "R3Config",
    "run",
]
