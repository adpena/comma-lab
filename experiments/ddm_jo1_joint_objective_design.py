#!/usr/bin/env python3
"""Typed, scorer-free preparation for the rc2 JO1 joint objective.

This module is deliberately incapable of dispatching work.  It validates and
seals the complete workload description, verifies retained input custody, and
emits a dependency-ordered MAIN fire order.  Missing scorer or memory payloads
produce a typed ``BLOCKED`` result; they are never replaced by guessed hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import shutil
import struct
import zipfile
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA = "ddm_jo1_compiled_config.v1"
READINESS_SCHEMA = "ddm_jo1_readiness.v1"
MEMORY_RECEIPT_SCHEMA = "ddm_jo1_memory_preflight.v1"
CHECKPOINT_SCHEMA = "ddm_jo1_checkpoint.v1"
PAYLOAD_MANIFEST_SCHEMA = "ddm_jo1_retained_payload_manifest.v1"
OUTPUT_ROOT = Path("/Volumes/APDataStore/pact/ddm_jo1_joint_objective_design")
MATERIALIZER_OUTPUT_ROOT = Path(
    "/Volumes/APDataStore/pact/ddm_gs3_unbridled_gestalt/jo1_payload_unblock"
)

N_PAIRS = 600
SEG_H = 384
SEG_W = 512
SEG_DENOMINATOR = N_PAIRS * SEG_H * SEG_W
BASE_ARCHIVE_BYTES = 180_456
BASE_FLIPS = 23_757
BASE_DPOSE = 6.370359e-6
SCORE_DENOMINATOR = 37_545_489
COLLATERAL_CAP = 0.89
RATE_INCREMENT_ANCHOR = 1_176
RATE_INCREMENT_BAND = (1_174, 1_191)
STRICT_TEN_X_FLIPS = 966
PREREGISTERED_LIVE_FLIPS = 965
RC2_ARCHIVE_SHA256 = "df7fd266e1b7488cdec02c7b5c1201c40628804260286001f38b51d7ed9e2080"
FX5_ARCHIVE_BYTES = 180_386
FX5_ARCHIVE_SHA256 = "4b54fccc25f100cb68030db317791ba5e58936bb9b491f9ee9a020e695b79841"
FX5_RUNTIME_TREE_SHA256 = "8eff613ecec2c371a6fa4cc580b8af9df131f45dc33f5d3c9b829faac1a513a5"
AUTH_CACHE_VOLUME_NAME = "comma-auth-eval-cache-artifacts"
MATERIALIZER_BATCH_PAIRS = 16
MATERIALIZER_MAX_CHUNK_PAIRS = 120
MATERIALIZER_STORAGE_RESERVE_BYTES = 4 * 1024**3
TRAINING_MIN_AP_FREE_BYTES = 44 * 1024**3
# Two retained exact-receiver raws plus every tensor produced by one SegNet and
# one PoseNet pass.  The scorer functions retain their per-batch inputs and
# full outputs, not only the final field/vectors.  NPY headers and runtime/log
# metadata are covered by the separate reserve.
MATERIALIZER_EXPECTED_RETAINED_PAYLOAD_BYTES = (
    2 * (N_PAIRS * 2 * 874 * 1164 * 3)
    + N_PAIRS * 3 * SEG_H * SEG_W * 4
    + N_PAIRS * 5 * SEG_H * SEG_W * 4
    + N_PAIRS * SEG_H * SEG_W
    + N_PAIRS * 12 * SEG_H * SEG_W * 4
    + N_PAIRS * 12 * 4
    + N_PAIRS * 6 * 4
)
IMPLEMENTATION_BLOCKER = "RC2_FRESH_SCHUR_RECEIVER_CLOSE_NOT_IMPLEMENTED"

REQUIRED_STAGE_IDS = ("target_birth", "joint_balance", "collateral_finish")
REQUIRED_FIELD_OUTPUTS = frozenset(
    {
        "candidate_argmax_field",
        "bhw_decomposition",
        "pose6_outputs",
        "exact_package",
        "decoded_render_identity",
        "metrics_json",
    }
)


class JO1Error(RuntimeError):
    """Fail-closed configuration, custody, packaging, or admission error."""


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ConstantValue(StrictModel):
    value: int | float | str | bool
    value_type: Literal["int", "float", "str", "bool"]
    unit: str = Field(min_length=1)
    provenance_class: Literal["MEASURED", "DERIVED", "OPERATOR_REGISTERED", "CONTRACT"]
    source_citation: str = Field(min_length=1)
    rederivation_trigger: str = Field(min_length=1)

    @model_validator(mode="after")
    def value_matches_type(self) -> ConstantValue:
        expected = {"int": int, "float": float, "str": str, "bool": bool}[self.value_type]
        # bool is an int subclass, so require exact type for all authority values.
        if type(self.value) is not expected:
            raise ValueError(f"constant value/type differ: {self.value!r} vs {self.value_type}")
        return self


class AuthorityConstants(StrictModel):
    base_score: ConstantValue
    base_archive_bytes: ConstantValue
    base_dseg: ConstantValue
    base_flips: ConstantValue
    base_dpose: ConstantValue
    score_denominator_bytes: ConstantValue
    repinned_increment_bytes: ConstantValue
    repinned_increment_low_bytes: ConstantValue
    repinned_increment_high_bytes: ConstantValue
    collateral_cap: ConstantValue
    live_band_flips: ConstantValue
    strict_ten_x_flips: ConstantValue
    free_receiver_header_bytes: ConstantValue


def _sha256_path(path: Path) -> str:
    if path.is_file():
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    if path.is_dir():
        digest = hashlib.sha256()
        for child in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
            relative = child.relative_to(path).as_posix().encode()
            digest.update(struct.pack(">I", len(relative)))
            digest.update(relative)
            digest.update(struct.pack(">Q", child.stat().st_size))
            with child.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
        return digest.hexdigest()
    raise JO1Error(f"artifact path is absent: {path}")


def _path_bytes(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())
    raise JO1Error(f"artifact path is absent: {path}")


class ArtifactRef(StrictModel):
    path: str = Field(min_length=1)
    bytes: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    kind: Literal["file", "tree"]
    axis: str = Field(min_length=1)
    source_object_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    shape: tuple[int, ...] | None = None
    dtype: str | None = None

    @property
    def resolved_path(self) -> Path:
        return Path(self.path).expanduser().resolve()


def artifact_ref(
    path: Path,
    *,
    axis: str,
    source_object_sha256: str,
    shape: tuple[int, ...] | None = None,
    dtype: str | None = None,
) -> ArtifactRef:
    resolved = path.expanduser().resolve()
    return ArtifactRef(
        path=str(resolved),
        bytes=_path_bytes(resolved),
        sha256=_sha256_path(resolved),
        kind="tree" if resolved.is_dir() else "file",
        axis=axis,
        source_object_sha256=source_object_sha256,
        shape=shape,
        dtype=dtype,
    )


def verify_artifact(record: ArtifactRef) -> dict[str, Any]:
    path = record.resolved_path
    if record.kind == "file" and not path.is_file():
        raise JO1Error(f"expected file is absent: {path}")
    if record.kind == "tree" and not path.is_dir():
        raise JO1Error(f"expected tree is absent: {path}")
    observed_bytes = _path_bytes(path)
    observed_sha256 = _sha256_path(path)
    if observed_bytes != record.bytes or observed_sha256 != record.sha256:
        raise JO1Error(
            f"artifact drift: {path}; bytes={observed_bytes}/{record.bytes}, "
            f"sha256={observed_sha256}/{record.sha256}"
        )
    return record.model_dump(mode="json")


class InputBindings(StrictModel):
    rc2_archive: ArtifactRef
    rc2_runtime: ArtifactRef
    rc2_decoded_semantic_tokens: ArtifactRef | None = None
    gt_argmax_field: ArtifactRef | None = None
    rc2_base_argmax_field: ArtifactRef | None = None
    source_pose6_targets: ArtifactRef | None = None
    source_object: ArtifactRef
    segnet_weights: ArtifactRef
    posenet_weights: ArtifactRef
    compiler_source: ArtifactRef
    worker_source: ArtifactRef
    dispatcher_source: ArtifactRef
    materializer_worker_source: ArtifactRef | None = None
    memory_preflight_receipt: ArtifactRef | None = None

    @model_validator(mode="after")
    def rc2_anchor_is_exact(self) -> InputBindings:
        if self.rc2_archive.bytes != BASE_ARCHIVE_BYTES or self.rc2_archive.sha256 != RC2_ARCHIVE_SHA256:
            raise ValueError("rc2 archive authority pin differs")
        expected_field = (N_PAIRS, SEG_H, SEG_W)
        for name, record in (
            ("rc2_decoded_semantic_tokens", self.rc2_decoded_semantic_tokens),
            ("gt_argmax_field", self.gt_argmax_field),
            ("rc2_base_argmax_field", self.rc2_base_argmax_field),
        ):
            if record is not None and (record.shape != expected_field or record.dtype != "uint8"):
                raise ValueError(f"{name} must declare shape={expected_field}, dtype=uint8")
        pose = self.source_pose6_targets
        if pose is not None and (
            pose.shape is None
            or len(pose.shape) < 2
            or pose.shape[0] != N_PAIRS
            or pose.shape[-1] != 6
            or pose.dtype not in {"float32", "float64"}
        ):
            raise ValueError("source_pose6_targets must declare n600 x ... x 6 float shape")
        return self


class MaterializerConfig(StrictModel):
    vehicle_id: Literal["fx5_e1", "rc2"]
    archive: ArtifactRef
    runtime: ArtifactRef
    expected_runtime_tree_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    batch_pairs: Literal[16]
    chunk_pair_limit: Literal[120]
    remote_volume_name: Literal["comma-auth-eval-cache-artifacts"]
    remote_volume_run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]+$")
    harvest_root: str
    rc2_fallback_reason: str | None = None

    @model_validator(mode="after")
    def exact_live_base_or_reasoned_fallback(self) -> MaterializerConfig:
        if self.vehicle_id == "fx5_e1":
            if (
                self.archive.bytes != FX5_ARCHIVE_BYTES
                or self.archive.sha256 != FX5_ARCHIVE_SHA256
            ):
                raise ValueError("fx5_e1 materializer archive pin differs")
            if self.expected_runtime_tree_sha256 != FX5_RUNTIME_TREE_SHA256:
                raise ValueError("fx5_e1 materializer runtime-tree pin differs")
            if self.rc2_fallback_reason is not None:
                raise ValueError("fx5_e1 must not carry an rc2 fallback reason")
        else:
            if (
                self.archive.bytes != BASE_ARCHIVE_BYTES
                or self.archive.sha256 != RC2_ARCHIVE_SHA256
            ):
                raise ValueError("rc2 fallback materializer archive pin differs")
            if not (self.rc2_fallback_reason or "").strip():
                raise ValueError("rc2 fallback requires a written reason")
        harvest = Path(self.harvest_root).expanduser().resolve()
        allowed = MATERIALIZER_OUTPUT_ROOT.resolve()
        if allowed != harvest and allowed not in harvest.parents:
            raise ValueError("materializer harvest root is outside the chartered AP store")
        return self


class ValueKnob(StrictModel):
    value: float = Field(ge=0.0)
    unit: str = Field(min_length=1)
    provenance_class: Literal["MEASURED", "DERIVED", "HYPOTHESIS", "OPERATOR_REGISTERED"]
    source_citation: str = Field(min_length=1)
    rederivation_trigger: str = Field(min_length=1)


class FieldPass(StrictModel):
    at_end: Literal[True]
    pair_start: Literal[0]
    pair_count: Literal[600]
    batch_pairs: int = Field(gt=0, le=600)
    retained_outputs: tuple[str, ...]

    @field_validator("retained_outputs")
    @classmethod
    def all_outputs_are_retained(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if set(value) != REQUIRED_FIELD_OUTPUTS:
            raise ValueError(
                "stage field retained outputs differ; "
                f"missing={sorted(REQUIRED_FIELD_OUTPUTS - set(value))}, "
                f"unknown={sorted(set(value) - REQUIRED_FIELD_OUTPUTS)}"
            )
        return value


class StageConfig(StrictModel):
    stage_id: Literal["target_birth", "joint_balance", "collateral_finish"]
    boundary_event: str = Field(min_length=1)
    fail_safe_steps: int = Field(gt=0)
    learning_rate: ValueKnob
    benefit_weight: ValueKnob
    harm_weight: ValueKnob
    pose_weight: ValueKnob
    rate_proxy_weight: ValueKnob
    field_pass: FieldPass
    checkpoint_every_steps: int = Field(gt=0)

    @model_validator(mode="after")
    def checkpoint_is_intra_stage(self) -> StageConfig:
        if self.checkpoint_every_steps > self.fail_safe_steps:
            raise ValueError("checkpoint cadence exceeds the stage fail-safe cap")
        return self


class ActuationConfig(StrictModel):
    family: Literal["hybrid_oriented_context_output_rgb_residual"]
    injection_point: Literal["semantic_renderer_output_before_exact_R"]
    hidden_channels: int = Field(gt=0, le=64)
    max_rgb_delta: ValueKnob
    derive_context_from_tokens: Literal[True]
    token_blocks_after_actuation: Literal[0]
    exact_roundtrip_in_loop: Literal[True]


class ObjectiveConfig(StrictModel):
    benefit_on_base_errors: Literal[True]
    harm_on_base_correct: Literal[True]
    collateral_rho: Literal[0.89]
    collateral_augmented_lagrangian: Literal[True]
    pose_augmented_lagrangian: Literal[True]
    pose_hard_cap: Literal[6.370359e-06]
    rate_proxy_name: Literal["rate_proxy"]
    realized_rate_at_stage_boundary: Literal[True]
    dual_updates: Literal["stage_boundary_only"]
    exact_bhw_admission: Literal[True]


class CheckpointConfig(StrictModel):
    schema: Literal["ddm_jo1_checkpoint.v1"]
    atomic_replace: Literal[True]
    distinct_stage_paths: Literal[True]
    required_state: tuple[str, ...]

    @field_validator("required_state")
    @classmethod
    def complete_resume_state(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        required = {
            "stage_id",
            "step",
            "field_pass_cursor",
            "package_cursor",
            "live",
            "ema",
            "optimizer",
            "rng",
            "duals",
            "config_sha256",
        }
        if set(value) != required:
            raise ValueError("checkpoint resume state is incomplete")
        return value


class MemoryPreflightConfig(StrictModel):
    required: Literal[True]
    device_class: Literal["NVIDIA T4"]
    real_config: Literal[True]
    requested_memory_bytes: int = Field(gt=0)
    minimum_headroom_bytes: int = Field(gt=0)
    minimum_ap_free_bytes: int = Field(gt=0)
    max_age_hours: int = Field(gt=0, le=24)


class DispatchConfig(StrictModel):
    lane_id: str = Field(pattern=r"^ddm_jo1_[a-z0-9_]+$")
    claim_agent: Literal["MAIN"]
    platform: Literal["modal"]
    gpu: Literal["T4"]
    detach_required: Literal[True]
    provider_detach_ack_required: Literal[True]
    single_flight: Literal[True]
    durable_call_id: Literal[True]
    automatic_terminal_closure: Literal[True]


class CompiledConfig(StrictModel):
    schema: Literal["ddm_jo1_compiled_config.v1"]
    action: Literal["prepare", "materialize_scorer_payloads", "memory_preflight", "train"]
    run_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]+$")
    output_root: str
    seed: int
    retain_all_payloads: Literal[True]
    authority: AuthorityConstants
    inputs: InputBindings
    actuation: ActuationConfig
    objective: ObjectiveConfig
    stages: tuple[StageConfig, StageConfig, StageConfig]
    checkpoint: CheckpointConfig
    memory_preflight: MemoryPreflightConfig
    dispatch: DispatchConfig
    materializer: MaterializerConfig | None = None
    workload_config_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def canonical_schedule(self) -> CompiledConfig:
        if tuple(stage.stage_id for stage in self.stages) != REQUIRED_STAGE_IDS:
            raise ValueError("JO1 stages must be target_birth, joint_balance, collateral_finish")
        output = Path(self.output_root).expanduser().resolve()
        allowed = (
            MATERIALIZER_OUTPUT_ROOT.resolve()
            if self.action == "materialize_scorer_payloads"
            else OUTPUT_ROOT.resolve()
        )
        if allowed != output and allowed not in output.parents:
            raise ValueError("JO1 output is outside its chartered APDataStore root")
        if self.action == "materialize_scorer_payloads" and self.materializer is None:
            raise ValueError("materialize_scorer_payloads requires a materializer config")
        if self.memory_preflight.minimum_ap_free_bytes < TRAINING_MIN_AP_FREE_BYTES:
            raise ValueError("JO1 training storage bar is below 44 GiB")
        final = self.stages[-1]
        if self.objective.pose_hard_cap != BASE_DPOSE or self.objective.collateral_rho != COLLATERAL_CAP:
            raise ValueError("final objective weakens the pose or collateral cap")
        if final.field_pass.pair_count != N_PAIRS:
            raise ValueError("final stage is not an n600 field pass")
        return self


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def attach_workload_sha256(config: CompiledConfig) -> CompiledConfig:
    body = config.model_dump(mode="json")
    body["workload_config_sha256"] = None
    body["inputs"]["memory_preflight_receipt"] = None
    digest = canonical_sha256(body)
    return config.model_copy(update={"workload_config_sha256": digest})


def authority_constants() -> AuthorityConstants:
    source = ".omx/tmp/jo1_charter.md and ddm_ec2p_conditioning_repin_20260821.md"
    def c(value: Any, value_type: str, unit: str, provenance: str, trigger: str) -> ConstantValue:
        return ConstantValue(
            value=value,
            value_type=value_type,
            unit=unit,
            provenance_class=provenance,
            source_citation=source,
            rederivation_trigger=trigger,
        )
    return AuthorityConstants(
        base_score=c(0.14827847122030852, "float", "score", "MEASURED", "new promoted exact rc2 row"),
        base_archive_bytes=c(BASE_ARCHIVE_BYTES, "int", "bytes", "MEASURED", "rc2 archive changes"),
        base_dseg=c(BASE_FLIPS / SEG_DENOMINATOR, "float", "fraction", "DERIVED", "base field changes"),
        base_flips=c(BASE_FLIPS, "int", "pixels", "MEASURED", "base field changes"),
        base_dpose=c(BASE_DPOSE, "float", "MSE", "MEASURED", "base pose6 receipt changes"),
        score_denominator_bytes=c(SCORE_DENOMINATOR, "int", "bytes", "CONTRACT", "contest evaluator changes"),
        repinned_increment_bytes=c(RATE_INCREMENT_ANCHOR, "int", "bytes", "MEASURED", "real coder or body changes"),
        repinned_increment_low_bytes=c(RATE_INCREMENT_BAND[0], "int", "bytes", "MEASURED", "real coder rerun"),
        repinned_increment_high_bytes=c(RATE_INCREMENT_BAND[1], "int", "bytes", "MEASURED", "real coder rerun"),
        collateral_cap=c(COLLATERAL_CAP, "float", "introduced/fixed", "OPERATOR_REGISTERED", "operator changes preregistration"),
        live_band_flips=c(PREREGISTERED_LIVE_FLIPS, "int", "net pixels", "OPERATOR_REGISTERED", "operator changes preregistration"),
        strict_ten_x_flips=c(STRICT_TEN_X_FLIPS, "int", "net pixels", "DERIVED", "strict score threshold changes"),
        free_receiver_header_bytes=c(633, "int", "bytes", "CONTRACT", "receiver/code charging rule changes"),
    )


def delta_score(*, fixed: int, introduced: int, d_pose_candidate: float, candidate_archive_bytes: int) -> float:
    if min(fixed, introduced, candidate_archive_bytes) < 0 or not math.isfinite(d_pose_candidate) or d_pose_candidate < 0:
        raise JO1Error("exact score inputs are outside their domains")
    return (
        100.0 * (introduced - fixed) / SEG_DENOMINATOR
        + math.sqrt(10.0 * d_pose_candidate)
        - math.sqrt(10.0 * BASE_DPOSE)
        + 25.0 * (candidate_archive_bytes - BASE_ARCHIVE_BYTES) / SCORE_DENOMINATOR
    )


def preregistered_band(net_fixed_flips: int) -> str:
    if net_fixed_flips >= PREREGISTERED_LIVE_FLIPS:
        return "LIVE"
    if net_fixed_flips >= 924:
        return "MARGINAL"
    if net_fixed_flips >= 0:
        return "CLOSED-neutral"
    return "CLOSED-harmful"


def prior_law_diagnostics() -> dict[str, float]:
    measured_gross_fraction = 12_075 / 34_970
    measured_old_ratio = 52_854 / 12_075
    predicted_fixes = BASE_FLIPS * measured_gross_fraction
    exact_ratio = (predicted_fixes - PREREGISTERED_LIVE_FLIPS) / predicted_fixes
    exact_suppression = measured_old_ratio / exact_ratio
    return {
        "transferred_gross_recovery_fraction": measured_gross_fraction,
        "predicted_fixes": predicted_fixes,
        "net_live_flips": float(PREREGISTERED_LIVE_FLIPS),
        "introduced_per_fixed_required_exact": exact_ratio,
        "measured_old_introduced_per_fixed": measured_old_ratio,
        "suppression_from_measured_ratio_required_exact": exact_suppression,
        "registered_collateral_cap": COLLATERAL_CAP,
        "registered_suppression_label": 4.93,
    }


def stage_admission(
    *,
    fixed: int,
    introduced: int,
    wrong_to_wrong: int,
    d_pose_candidate: float,
    candidate_archive_bytes: int,
    single_p: bool,
    package_parseback_identity: bool,
) -> dict[str, Any]:
    if min(fixed, introduced, wrong_to_wrong) < 0:
        raise JO1Error("B/H/W counts must be nonnegative")
    delta = delta_score(
        fixed=fixed,
        introduced=introduced,
        d_pose_candidate=d_pose_candidate,
        candidate_archive_bytes=candidate_archive_bytes,
    )
    ratio = 0.0 if fixed == 0 and introduced == 0 else (math.inf if fixed == 0 else introduced / fixed)
    blockers = []
    if d_pose_candidate > BASE_DPOSE:
        blockers.append("POSE_CAP_EXCEEDED")
    if ratio > COLLATERAL_CAP:
        blockers.append("COLLATERAL_CAP_EXCEEDED")
    if not single_p:
        blockers.append("ARCHIVE_NOT_SINGLE_P")
    if not package_parseback_identity:
        blockers.append("PACKAGE_PARSEBACK_DIFFERS")
    if delta >= 0.0:
        blockers.append("EXACT_DELTA_NONNEGATIVE")
    net = fixed - introduced
    return {
        "schema": "ddm_jo1_stage_admission.v1",
        "fixed": fixed,
        "introduced": introduced,
        "wrong_to_wrong": wrong_to_wrong,
        "net_fixed_flips": net,
        "collateral_per_fix": ratio,
        "d_pose_candidate": d_pose_candidate,
        "candidate_archive_bytes": candidate_archive_bytes,
        "delta_score": delta,
        "preregistered_band": preregistered_band(net),
        "admissible": not blockers,
        "blockers": blockers,
    }


def deterministic_single_p_archive(member: bytes) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", allowZip64=False) as archive:
        info = zipfile.ZipInfo("p", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        archive.writestr(info, member)
    return output.getvalue()


def read_single_p_archive(path: Path) -> bytes:
    with zipfile.ZipFile(path) as archive:
        infos = archive.infolist()
        if len(infos) != 1 or infos[0].filename != "p" or infos[0].compress_type != zipfile.ZIP_STORED:
            raise JO1Error("archive must contain exactly one stored member p")
        member = archive.read(infos[0])
        if archive.testzip() is not None:
            raise JO1Error("archive CRC validation failed")
    return member


def _atomic_bytes(path: Path, payload: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return {
        "path": str(path.resolve()),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def atomic_json(path: Path, value: Any) -> dict[str, Any]:
    return _atomic_bytes(path, canonical_json_bytes(value))


def verify_memory_receipt(config: CompiledConfig) -> dict[str, Any]:
    reference = config.inputs.memory_preflight_receipt
    if reference is None:
        raise JO1Error("memory preflight receipt is absent")
    verify_artifact(reference)
    value = json.loads(reference.resolved_path.read_text(encoding="utf-8"))
    required = {
        "schema",
        "passed",
        "device",
        "training_batch_pairs",
        "field_batch_pairs",
        "geometry",
        "max_memory_allocated_bytes",
        "max_memory_reserved_bytes",
        "requested_memory_bytes",
        "headroom_bytes",
        "workload_config_sha256",
        "producer_command",
        "created_at_utc",
    }
    if set(value) != required:
        raise JO1Error("memory receipt fields differ")
    if value["schema"] != MEMORY_RECEIPT_SCHEMA or value["passed"] is not True:
        raise JO1Error("memory preflight did not pass")
    if value["device"] != config.memory_preflight.device_class:
        raise JO1Error("memory preflight device differs")
    if value["geometry"] != [N_PAIRS, SEG_H, SEG_W]:
        raise JO1Error("memory preflight geometry differs")
    if value["workload_config_sha256"] != config.workload_config_sha256:
        raise JO1Error("memory receipt is bound to a different workload")
    if int(value["requested_memory_bytes"]) != config.memory_preflight.requested_memory_bytes:
        raise JO1Error("memory receipt requested capacity differs")
    if int(value["headroom_bytes"]) < config.memory_preflight.minimum_headroom_bytes:
        raise JO1Error("memory receipt has insufficient headroom")
    try:
        created = datetime.fromisoformat(str(value["created_at_utc"]).replace("Z", "+00:00"))
    except ValueError as error:
        raise JO1Error("memory receipt timestamp is invalid") from error
    age_seconds = (datetime.now(UTC) - created.astimezone(UTC)).total_seconds()
    if age_seconds < 0 or age_seconds > config.memory_preflight.max_age_hours * 3600:
        raise JO1Error("memory receipt is stale or future-dated")
    return value


def readiness(config: CompiledConfig) -> dict[str, Any]:
    # This blocker is removed only when the remote worker performs the complete
    # rc2 render -> exact R -> both scorers -> fresh same-object Schur -> real
    # single-p package loop.  Validation primitives alone are not readiness.
    blockers: list[str] = [IMPLEMENTATION_BLOCKER]
    output_root = Path(config.output_root).resolve()
    storage_probe = output_root if output_root.exists() else output_root.parent
    free_bytes: int | None = None
    try:
        free_bytes = shutil.disk_usage(storage_probe).free
    except OSError as error:
        blockers.append(f"AP_STORAGE_PREFLIGHT_BLOCKED:{error}")
    else:
        if free_bytes < config.memory_preflight.minimum_ap_free_bytes:
            blockers.append(
                "AP_STORAGE_PREFLIGHT_BLOCKED:"
                f"free={free_bytes},required={config.memory_preflight.minimum_ap_free_bytes}"
            )
    required_inputs = {
        "RC2_DECODED_SEMANTIC_TOKENS_MISSING": config.inputs.rc2_decoded_semantic_tokens,
        "GT_ARGMAX_FIELD_MISSING": config.inputs.gt_argmax_field,
        "RC2_BASE_ARGMAX_FIELD_MISSING": config.inputs.rc2_base_argmax_field,
        "SOURCE_POSE6_TARGETS_MISSING": config.inputs.source_pose6_targets,
    }
    for blocker, record in required_inputs.items():
        if record is None:
            blockers.append(blocker)
        else:
            try:
                verify_artifact(record)
            except JO1Error:
                blockers.append(blocker.replace("_MISSING", "_DRIFT"))
    for record in (
        config.inputs.rc2_archive,
        config.inputs.rc2_runtime,
        config.inputs.source_object,
        config.inputs.segnet_weights,
        config.inputs.posenet_weights,
        config.inputs.compiler_source,
        config.inputs.worker_source,
        config.inputs.dispatcher_source,
    ):
        try:
            verify_artifact(record)
        except JO1Error:
            blockers.append(f"PIN_DRIFT:{Path(record.path).name}")
    try:
        verify_memory_receipt(config)
    except JO1Error as error:
        blockers.append(f"MEMORY_PREFLIGHT_BLOCKED:{error}")
    return {
        "schema": READINESS_SCHEMA,
        "status": "READY_TO_FIRE" if not blockers else "BLOCKED",
        "blockers": blockers,
        "workload_config_sha256": config.workload_config_sha256,
        "frontier_moved": False,
        "frontier_line": "rc2 S=0.14827847122030852 remains unchanged",
        "storage_probe": {
            "path": str(storage_probe),
            "free_bytes": free_bytes,
            "required_free_bytes": config.memory_preflight.minimum_ap_free_bytes,
        },
    }


def materializer_readiness(config: CompiledConfig) -> dict[str, Any]:
    """Verify only the producer dependencies; training gates do not apply."""
    blockers: list[str] = []
    materializer = config.materializer
    if materializer is None:
        blockers.append("MATERIALIZER_CONFIG_MISSING")
        return {
            "schema": "ddm_jo1_materializer_readiness.v1",
            "status": "BLOCKED",
            "blockers": blockers,
            "workload_config_sha256": config.workload_config_sha256,
            "frontier_moved": False,
            "frontier_line": (
                "effective frontier pointer remains unchanged; "
                "this is a component-only materializer"
            ),
            "storage_probe": None,
        }

    harvest_root = Path(materializer.harvest_root).expanduser().resolve()
    storage_probe = harvest_root if harvest_root.exists() else harvest_root.parent
    free_bytes: int | None = None
    already_retained = 0
    try:
        free_bytes = shutil.disk_usage(storage_probe).free
        if harvest_root.exists():
            already_retained = sum(
                child.stat().st_size
                for child in harvest_root.rglob("*")
                if child.is_file()
            )
    except OSError as error:
        blockers.append(f"MATERIALIZER_STORAGE_PREFLIGHT_BLOCKED:{error}")
    remaining_payload = max(
        0, MATERIALIZER_EXPECTED_RETAINED_PAYLOAD_BYTES - already_retained
    )
    required_free = remaining_payload + MATERIALIZER_STORAGE_RESERVE_BYTES
    if free_bytes is not None and free_bytes < required_free:
        blockers.append(
            "MATERIALIZER_STORAGE_PREFLIGHT_BLOCKED:"
            f"free={free_bytes},required={required_free}"
        )

    required_inputs = {
        "GT_ARGMAX_FIELD_MISSING": config.inputs.gt_argmax_field,
        "SOURCE_POSE6_TARGETS_MISSING": config.inputs.source_pose6_targets,
        "MATERIALIZER_WORKER_SOURCE_MISSING": config.inputs.materializer_worker_source,
    }
    for blocker, record in required_inputs.items():
        if record is None:
            blockers.append(blocker)
        else:
            try:
                verify_artifact(record)
            except JO1Error:
                blockers.append(blocker.replace("_MISSING", "_DRIFT"))
    for record in (
        materializer.archive,
        materializer.runtime,
        config.inputs.source_object,
        config.inputs.segnet_weights,
        config.inputs.posenet_weights,
        config.inputs.compiler_source,
        config.inputs.worker_source,
        config.inputs.dispatcher_source,
    ):
        try:
            verify_artifact(record)
        except JO1Error:
            blockers.append(f"PIN_DRIFT:{Path(record.path).name}")
    return {
        "schema": "ddm_jo1_materializer_readiness.v1",
        "status": "READY_TO_FIRE" if not blockers else "BLOCKED",
        "blockers": blockers,
        "workload_config_sha256": config.workload_config_sha256,
        "frontier_moved": False,
        "frontier_line": (
            "effective frontier pointer remains unchanged; "
            "this is a component-only materializer"
        ),
        "vehicle_id": materializer.vehicle_id,
        "storage_probe": {
            "path": str(storage_probe),
            "free_bytes": free_bytes,
            "already_retained_bytes": already_retained,
            "expected_total_retained_payload_bytes": (
                MATERIALIZER_EXPECTED_RETAINED_PAYLOAD_BYTES
            ),
            "remaining_payload_bytes": remaining_payload,
            "reserve_bytes": MATERIALIZER_STORAGE_RESERVE_BYTES,
            "required_free_bytes": required_free,
            "training_requirement_bytes": config.memory_preflight.minimum_ap_free_bytes,
            "training_requirement_applied": False,
        },
    }


def readiness_for_action(config: CompiledConfig) -> dict[str, Any]:
    if config.action == "materialize_scorer_payloads":
        return materializer_readiness(config)
    return readiness(config)


def fire_order(config_path: Path, config_sha256: str, config: CompiledConfig) -> dict[str, Any]:
    common = [
        ".venv/bin/modal",
        "run",
    ]
    def command(entrypoint: str) -> list[str]:
        return [
            *common,
            f"experiments/ddm_jo1_modal_joint_objective.py::{entrypoint}",
            "--compiled-config",
            str(config_path.resolve()),
            "--expected-config-sha256",
            config_sha256,
            "--main-owned-dispatch-authorization",
            "--detach",
            "--provider-detach-ack",
        ]
    materializer_readiness_value = (
        materializer_readiness(config)
        if config.action == "materialize_scorer_payloads"
        else None
    )
    materializer_ready = bool(
        materializer_readiness_value is not None
        and materializer_readiness_value["status"] == "READY_TO_FIRE"
    )
    materializer = config.materializer
    harvest_command = None
    if materializer is not None:
        harvest_command = [
            ".venv/bin/modal",
            "volume",
            "get",
            "--force",
            materializer.remote_volume_name,
            f"{materializer.remote_volume_run_id}/",
            str(Path(materializer.harvest_root).resolve() / "harvest"),
        ]
    return {
        "schema": "ddm_jo1_fire_order.v1",
        "owner": "MAIN",
        "lane_id": config.dispatch.lane_id,
        "current_disposition": "READY" if materializer_ready else "BLOCKED",
        "current_blocker": (
            None
            if materializer_ready
            else (
                ";".join(materializer_readiness_value["blockers"])
                if materializer_readiness_value is not None
                else IMPLEMENTATION_BLOCKER
            )
        ),
        "stop_after_each_async_fire": True,
        "commands": [
            {
                "ordinal": 1,
                "purpose": "materialize_scorer_payloads",
                "argv": command("materialize_scorer_payloads"),
                "fire_trigger": (
                    "materializer storage preflight PASS; no active n600 scorer job; "
                    "MAIN holds a unique lane claim"
                    if materializer_ready
                    else "materializer backend implemented and reviewed"
                ),
                "requires_reseal_after_harvest": True,
            },
            {
                "ordinal": "1H",
                "purpose": "harvest_materialized_payloads",
                "argv": harvest_command,
                "fire_trigger": "ordinal 1 call is terminal and the volume final receipt is COMPLETE",
                "requires_reseal_after_harvest": True,
            },
            {
                "ordinal": 2,
                "purpose": "real_scale_memory_preflight",
                "argv": None,
                "fire_trigger": "ordinal 1 payloads harvested into a newly sealed config",
                "requires_reseal_after_harvest": True,
            },
            {
                "ordinal": 3,
                "purpose": "train",
                "argv": None,
                "fire_trigger": "ordinal 2 fresh matching memory receipt harvested into a newly sealed config",
                "requires_reseal_after_harvest": False,
            },
        ],
    }


def prepare(config: CompiledConfig, *, destination: Path) -> dict[str, Any]:
    compiled = attach_workload_sha256(config)
    destination.mkdir(parents=True, exist_ok=True)
    config_path = destination / "compiled_config.json"
    config_record = _atomic_bytes(config_path, canonical_json_bytes(compiled.model_dump(mode="json")))
    readiness_value = readiness_for_action(compiled)
    readiness_record = atomic_json(destination / "READINESS.json", readiness_value)
    order = fire_order(config_path, config_record["sha256"], compiled)
    order_record = atomic_json(destination / "FIRE_ORDER.json", order)
    return {
        "schema": "ddm_jo1_prepare.v1",
        "status": readiness_value["status"],
        "compiled_config": config_record,
        "readiness": readiness_record,
        "fire_order": order_record,
        "dispatch_performed": False,
    }


def load_compiled_config(path: Path, expected_sha256: str) -> CompiledConfig:
    if not path.is_file():
        raise JO1Error(f"compiled config is absent: {path}")
    observed = _sha256_path(path)
    if observed != expected_sha256:
        raise JO1Error(f"compiled config sha256 differs: {observed} != {expected_sha256}")
    config = CompiledConfig.model_validate_json(path.read_text(encoding="utf-8"))
    attached = attach_workload_sha256(config)
    if attached.workload_config_sha256 != config.workload_config_sha256:
        raise JO1Error("compiled workload identity differs")
    return config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        config = CompiledConfig.model_validate_json(args.author_config.read_text(encoding="utf-8"))
        result = prepare(config, destination=args.output.resolve())
    except (OSError, ValueError, JO1Error, json.JSONDecodeError) as error:
        print(json.dumps({"status": "BLOCKED", "reason": str(error)}, sort_keys=True))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] == "READY_TO_FIRE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
