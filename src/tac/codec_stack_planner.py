# SPDX-License-Identifier: MIT
"""Deterministic HStack/VStack/multipass codec planning primitives.

This module is deliberately scoreless. It describes candidate codec-stack
repairs, byte/evidence semantics, blockers, and promotion gates before any
archive is allowed to leave planning. In this planner:

* HStack means horizontal, parallel composition across parser-proven logical
  streams inside the monolithic contest packet. It must not be inferred from
  ZIP member names on PR101/PR106-style HNeRV frontier archives.
* VStack means vertical, serial transforms inside one component stream:
  ``representation -> prediction -> quantization -> hyperprior -> arithmetic -> pack``.
* Multipass means compress-time repeated planning/eval passes that emit one
  final packet; no inflate-time scorer or sidecar is implied.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from tac.omega_opt_claims import (
    OMEGA_OPT_CLAIM_SCHEMA,
    omega_opt_claim_rows,
)

STACK_PLANNER_SCHEMA: str = "tac_hstack_vstack_multipass_plan_v2"
PROMOTION_POLICY_SCHEMA: str = "tac_codec_stack_fail_closed_promotion_policy_v1"

DEFAULT_STATIC_PMF_DELTA_BYTES: int = 793
DEFAULT_MAX_PASSES: int = 3
ABSOLUTE_MAX_PASSES: int = 5

SCORE_PROMOTION_EVIDENCE_GRADES: tuple[str, ...] = ("A", "A++")

HSTACK_VSTACK_AXIS_SEMANTICS: dict[str, str] = {
    "hstack_parallel": "parallel parser-proven logical stream codecs merged into one monolithic packet",
    "vstack_serial": "serial representation-to-pack transforms inside one component stream",
    "multipass": "compress-time repeated planning and training passes only",
}

NESTED_OPTIMIZATION_LEVELS: tuple[str, ...] = (
    "meta_pass_score_feedback",
    "bilevel_substrate_training",
    "multipass_refinement",
    "hstack_parallel_components",
    "vstack_serial_transforms",
    "per_tensor_hstack_parallel_substreams",
)

CANONICAL_QAT_PASSES: tuple[str, ...] = (
    "anchor",
    "finetune",
    "joint",
    "qat",
    "final",
)

QUALITY_MANDATE_11: tuple[str, ...] = (
    "beautiful",
    "elegant",
    "human_readable",
    "composable",
    "creative",
    "reusable",
    "expressive",
    "canonical",
    "production_hardened",
    "oss_ready",
    "paper_ready",
)

RENDERER_WEIGHT_BLOCK_FP_STREAMS: tuple[str, ...] = (
    "renderer_weights_qint",
    "renderer_weight_exponents",
)
MODEL_WEIGHT_BLOCK_FP_STREAMS: tuple[str, ...] = (
    "model_weight_qint",
    "model_weight_exponents",
)

PREDICTED_NESTED_SCORE_BAND: dict[str, Any] = {
    "schema": OMEGA_OPT_CLAIM_SCHEMA,
    "claim_count": 8,
    "evidence_grade": "prediction",
    "score_claim": False,
    "promotion_allowed": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "claims": omega_opt_claim_rows(),
}

BALLE_HYPERPRIOR_BLOCKERS: tuple[str, ...] = (
    "learned_model_overhead_not_amortized",
    "model_overhead_accounting_missing",
    "exact_reconstruction_roundtrip_missing",
    "runtime_packet_consumption_missing",
    "charged_archive_byte_delta_missing",
    "exact_cuda_auth_eval_missing",
    "static_shared_pmf_k12_negative_plus_793_bytes_requires_repair",
)

BALLE_HYPERPRIOR_REPAIR_CRITERIA: tuple[str, ...] = (
    "prove_model_overhead_bytes_are_less_than_conditional_entropy_savings",
    "record_model_weight_bytes_z_stream_bytes_headers_and_archive_delta",
    "prove_encode_decode_exact_reconstruction_on_canonical_qint_vectors",
    "prove_runtime_packet_contains_and_consumes_all_hyperprior_side_info",
    "record_old_new_archive_sha256_and_charged_byte_delta",
    "run_full_sample_exact_cuda_auth_eval_before_any_score_claim",
)

MULTIPASS_FAIL_CLOSED_CRITERIA: tuple[str, ...] = (
    "planner_outputs_zero_score_claim",
    "each_score_affecting_transform_has_charged_byte_delta_proof",
    "each_codec_component_has_exact_roundtrip_or_is_marked_non_decoding",
    "inflate_runtime_consumes_packet_without_score_sidecars",
    "exact_cuda_auth_eval_required_before_score_promotion",
)


def _canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    """Return the SHA-256 of a canonical JSON mapping."""
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def _require_id(value: str, *, field_name: str) -> str:
    text = str(value)
    if not text:
        raise ValueError(f"{field_name} must be non-empty")
    if "\x00" in text:
        raise ValueError(f"{field_name} must not contain NUL bytes")
    return text


def _string_tuple(values: Sequence[str] | None, *, field_name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    out = tuple(str(value) for value in values)
    if any(not value for value in out):
        raise ValueError(f"{field_name} must not contain empty strings")
    if len(set(out)) != len(out):
        raise ValueError(f"{field_name} must not contain duplicates")
    return out


def _optional_int(value: int | None, *, field_name: str, minimum: int | None = None) -> int | None:
    if value is None:
        return None
    out = int(value)
    if minimum is not None and out < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}, got {out}")
    return out


def _finite_optional_float(value: float | None, *, field_name: str) -> float | None:
    if value is None:
        return None
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{field_name} must be finite")
    return out


@dataclass(frozen=True)
class ByteEvidenceSemantics:
    """Byte accounting and evidence claims for one planning object."""

    evidence_grade: str
    evidence_semantics: str
    score_claim: bool = False
    score_affecting_payload_changed: bool = False
    charged_bits_changed: bool = False
    dispatchable: bool = False
    promotion_eligible: bool = False
    measured_archive_bytes: int | None = None
    estimated_archive_bytes: int | None = None
    byte_delta_vs_baseline: int | None = None
    model_overhead_bytes: int | None = None
    blockers: tuple[str, ...] = ()
    repair_criteria: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_id(self.evidence_grade, field_name="evidence_grade")
        _require_id(self.evidence_semantics, field_name="evidence_semantics")
        object.__setattr__(
            self,
            "measured_archive_bytes",
            _optional_int(self.measured_archive_bytes, field_name="measured_archive_bytes", minimum=0),
        )
        object.__setattr__(
            self,
            "estimated_archive_bytes",
            _optional_int(self.estimated_archive_bytes, field_name="estimated_archive_bytes", minimum=0),
        )
        object.__setattr__(
            self,
            "byte_delta_vs_baseline",
            None if self.byte_delta_vs_baseline is None else int(self.byte_delta_vs_baseline),
        )
        object.__setattr__(
            self,
            "model_overhead_bytes",
            _optional_int(self.model_overhead_bytes, field_name="model_overhead_bytes", minimum=0),
        )
        object.__setattr__(self, "blockers", _string_tuple(self.blockers, field_name="blockers"))
        object.__setattr__(
            self,
            "repair_criteria",
            _string_tuple(self.repair_criteria, field_name="repair_criteria"),
        )
        if self.score_claim and self.evidence_grade not in SCORE_PROMOTION_EVIDENCE_GRADES:
            raise ValueError(
                "score_claim=True requires exact CUDA-grade evidence; "
                f"got {self.evidence_grade!r}"
            )

    def to_manifest(self) -> dict[str, Any]:
        return {
            "evidence_grade": self.evidence_grade,
            "evidence_semantics": self.evidence_semantics,
            "score_claim": bool(self.score_claim),
            "score_affecting_payload_changed": bool(self.score_affecting_payload_changed),
            "charged_bits_changed": bool(self.charged_bits_changed),
            "dispatchable": bool(self.dispatchable),
            "promotion_eligible": bool(self.promotion_eligible),
            "measured_archive_bytes": self.measured_archive_bytes,
            "estimated_archive_bytes": self.estimated_archive_bytes,
            "byte_delta_vs_baseline": self.byte_delta_vs_baseline,
            "model_overhead_bytes": self.model_overhead_bytes,
            "blockers": list(self.blockers),
            "repair_criteria": list(self.repair_criteria),
        }


@dataclass(frozen=True)
class CodecComponent:
    """One component candidate in a codec stack plan."""

    component_id: str
    family: str
    role: str
    stack_axis: str
    streams: tuple[str, ...]
    byte_semantics: ByteEvidenceSemantics
    depends_on: tuple[str, ...] = ()
    exact_roundtrip_required: bool = True
    runtime_packet_required: bool = True
    deterministic: bool = True
    blockers: tuple[str, ...] = ()
    repair_criteria: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "component_id", _require_id(self.component_id, field_name="component_id"))
        object.__setattr__(self, "family", _require_id(self.family, field_name="family"))
        object.__setattr__(self, "role", _require_id(self.role, field_name="role"))
        object.__setattr__(self, "stack_axis", _require_id(self.stack_axis, field_name="stack_axis"))
        object.__setattr__(self, "streams", _string_tuple(self.streams, field_name="streams"))
        object.__setattr__(self, "depends_on", _string_tuple(self.depends_on, field_name="depends_on"))
        object.__setattr__(self, "blockers", _string_tuple(self.blockers, field_name="blockers"))
        object.__setattr__(
            self,
            "repair_criteria",
            _string_tuple(self.repair_criteria, field_name="repair_criteria"),
        )
        object.__setattr__(self, "notes", _string_tuple(self.notes, field_name="notes"))

    @property
    def all_blockers(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.blockers, *self.byte_semantics.blockers)))

    @property
    def all_repair_criteria(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.repair_criteria, *self.byte_semantics.repair_criteria)))

    def to_manifest(self) -> dict[str, Any]:
        return {
            "component_id": self.component_id,
            "family": self.family,
            "role": self.role,
            "stack_axis": self.stack_axis,
            "streams": list(self.streams),
            "depends_on": list(self.depends_on),
            "exact_roundtrip_required": bool(self.exact_roundtrip_required),
            "runtime_packet_required": bool(self.runtime_packet_required),
            "deterministic": bool(self.deterministic),
            "byte_semantics": self.byte_semantics.to_manifest(),
            "blockers": list(self.blockers),
            "repair_criteria": list(self.repair_criteria),
            "notes": list(self.notes),
        }


@dataclass(frozen=True)
class SerialTransform:
    """One VStack serial transform from input component IDs to outputs."""

    transform_id: str
    order_index: int
    transform_kind: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    byte_semantics: ByteEvidenceSemantics
    blockers: tuple[str, ...] = ()
    repair_criteria: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "transform_id", _require_id(self.transform_id, field_name="transform_id"))
        object.__setattr__(self, "order_index", _optional_int(self.order_index, field_name="order_index", minimum=0))
        object.__setattr__(
            self,
            "transform_kind",
            _require_id(self.transform_kind, field_name="transform_kind"),
        )
        object.__setattr__(self, "inputs", _string_tuple(self.inputs, field_name="inputs"))
        object.__setattr__(self, "outputs", _string_tuple(self.outputs, field_name="outputs"))
        object.__setattr__(self, "blockers", _string_tuple(self.blockers, field_name="blockers"))
        object.__setattr__(
            self,
            "repair_criteria",
            _string_tuple(self.repair_criteria, field_name="repair_criteria"),
        )

    @property
    def all_blockers(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.blockers, *self.byte_semantics.blockers)))

    def to_manifest(self) -> dict[str, Any]:
        return {
            "transform_id": self.transform_id,
            "order_index": self.order_index,
            "transform_kind": self.transform_kind,
            "inputs": list(self.inputs),
            "outputs": list(self.outputs),
            "byte_semantics": self.byte_semantics.to_manifest(),
            "blockers": list(self.blockers),
            "repair_criteria": list(self.repair_criteria),
        }


@dataclass(frozen=True)
class ParallelComponentGroup:
    """One HStack group of independently encoded components."""

    group_id: str
    merge_policy: str
    component_ids: tuple[str, ...]
    byte_semantics: ByteEvidenceSemantics
    blockers: tuple[str, ...] = ()
    repair_criteria: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_id", _require_id(self.group_id, field_name="group_id"))
        object.__setattr__(self, "merge_policy", _require_id(self.merge_policy, field_name="merge_policy"))
        object.__setattr__(self, "component_ids", _string_tuple(self.component_ids, field_name="component_ids"))
        object.__setattr__(self, "blockers", _string_tuple(self.blockers, field_name="blockers"))
        object.__setattr__(
            self,
            "repair_criteria",
            _string_tuple(self.repair_criteria, field_name="repair_criteria"),
        )

    @property
    def all_blockers(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.blockers, *self.byte_semantics.blockers)))

    def to_manifest(self) -> dict[str, Any]:
        return {
            "group_id": self.group_id,
            "merge_policy": self.merge_policy,
            "component_ids": list(self.component_ids),
            "byte_semantics": self.byte_semantics.to_manifest(),
            "blockers": list(self.blockers),
            "repair_criteria": list(self.repair_criteria),
        }


@dataclass(frozen=True)
class CodecPass:
    """One compress-time pass in the multipass plan."""

    pass_id: str
    pass_index: int
    purpose: str
    vstack_transform_ids: tuple[str, ...]
    hstack_group_ids: tuple[str, ...]
    evidence_gate: str
    byte_semantics: ByteEvidenceSemantics
    blockers: tuple[str, ...] = ()
    fail_closed_criteria: tuple[str, ...] = MULTIPASS_FAIL_CLOSED_CRITERIA

    def __post_init__(self) -> None:
        object.__setattr__(self, "pass_id", _require_id(self.pass_id, field_name="pass_id"))
        object.__setattr__(self, "pass_index", _optional_int(self.pass_index, field_name="pass_index", minimum=0))
        object.__setattr__(self, "purpose", _require_id(self.purpose, field_name="purpose"))
        object.__setattr__(
            self,
            "vstack_transform_ids",
            _string_tuple(self.vstack_transform_ids, field_name="vstack_transform_ids"),
        )
        object.__setattr__(
            self,
            "hstack_group_ids",
            _string_tuple(self.hstack_group_ids, field_name="hstack_group_ids"),
        )
        object.__setattr__(self, "evidence_gate", _require_id(self.evidence_gate, field_name="evidence_gate"))
        object.__setattr__(self, "blockers", _string_tuple(self.blockers, field_name="blockers"))
        object.__setattr__(
            self,
            "fail_closed_criteria",
            _string_tuple(self.fail_closed_criteria, field_name="fail_closed_criteria"),
        )

    @property
    def all_blockers(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.blockers, *self.byte_semantics.blockers)))

    def to_manifest(self) -> dict[str, Any]:
        return {
            "pass_id": self.pass_id,
            "pass_index": self.pass_index,
            "purpose": self.purpose,
            "vstack_transform_ids": list(self.vstack_transform_ids),
            "hstack_group_ids": list(self.hstack_group_ids),
            "evidence_gate": self.evidence_gate,
            "byte_semantics": self.byte_semantics.to_manifest(),
            "blockers": list(self.blockers),
            "fail_closed_criteria": list(self.fail_closed_criteria),
        }


@dataclass(frozen=True)
class FailClosedPromotionPolicy:
    """Policy separating exact-eval dispatch readiness from score promotion."""

    policy_id: str = "hstack_vstack_hyperprior_fail_closed_policy"
    schema: str = PROMOTION_POLICY_SCHEMA
    exact_eval_dispatch_requires: tuple[str, ...] = (
        "score_affecting_payload_changed",
        "charged_bits_changed",
        "archive_old_new_sha256_recorded",
        "exact_reconstruction_roundtrip_passed",
        "runtime_packet_contains_all_required_model_and_side_info",
        "inflate_path_consumes_packet_without_external_sidecars",
    )
    score_promotion_requires: tuple[str, ...] = (
        "full_sample_exact_cuda_auth_eval_json",
        "archive_zip_sha256_matches_eval_manifest",
        "component_distances_recomputed_from_cuda_eval",
        "dispatch_claim_closed_with_terminal_status",
    )
    default_blockers: tuple[str, ...] = (
        "planning_artifact_only",
        "score_claim_forbidden_until_exact_cuda_auth_eval",
    )

    def to_manifest(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "policy_id": self.policy_id,
            "fail_closed": True,
            "exact_eval_dispatch_requires": list(self.exact_eval_dispatch_requires),
            "score_promotion_requires": list(self.score_promotion_requires),
            "default_blockers": list(self.default_blockers),
        }


@dataclass(frozen=True)
class PromotionStatus:
    """Computed promotion status for a plan."""

    dispatchable: bool
    promotion_eligible: bool
    score_claim: bool
    blockers: tuple[str, ...]
    required_repairs: tuple[str, ...]

    def to_manifest(self) -> dict[str, Any]:
        return {
            "dispatchable": bool(self.dispatchable),
            "promotion_eligible": bool(self.promotion_eligible),
            "score_claim": bool(self.score_claim),
            "blockers": list(self.blockers),
            "required_repairs": list(self.required_repairs),
        }


@dataclass(frozen=True)
class CodecStackPlan:
    """A deterministic HStack/VStack/multipass planning artifact."""

    plan_id: str
    anchor_id: str
    static_pmf_delta_bytes: int
    components: tuple[CodecComponent, ...]
    serial_transforms: tuple[SerialTransform, ...]
    parallel_groups: tuple[ParallelComponentGroup, ...]
    passes: tuple[CodecPass, ...]
    promotion_policy: FailClosedPromotionPolicy = field(default_factory=FailClosedPromotionPolicy)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "plan_id", _require_id(self.plan_id, field_name="plan_id"))
        object.__setattr__(self, "anchor_id", _require_id(self.anchor_id, field_name="anchor_id"))
        object.__setattr__(self, "static_pmf_delta_bytes", int(self.static_pmf_delta_bytes))
        self.validate()

    def validate(self) -> None:
        component_ids = _ids_unique(
            [component.component_id for component in self.components],
            field_name="component.component_id",
        )
        transform_ids = _ids_unique(
            [transform.transform_id for transform in self.serial_transforms],
            field_name="serial_transform.transform_id",
        )
        group_ids = _ids_unique(
            [group.group_id for group in self.parallel_groups],
            field_name="parallel_group.group_id",
        )
        pass_ids = _ids_unique([codec_pass.pass_id for codec_pass in self.passes], field_name="pass.pass_id")
        if len(pass_ids) > ABSOLUTE_MAX_PASSES:
            raise ValueError(f"at most {ABSOLUTE_MAX_PASSES} passes are allowed")

        for component in self.components:
            unknown = [dep for dep in component.depends_on if dep not in component_ids]
            if unknown:
                raise ValueError(f"component {component.component_id!r} depends on unknown components: {unknown}")
        for transform in self.serial_transforms:
            unknown_inputs = [component_id for component_id in transform.inputs if component_id not in component_ids]
            unknown_outputs = [component_id for component_id in transform.outputs if component_id not in component_ids]
            if unknown_inputs or unknown_outputs:
                raise ValueError(
                    f"transform {transform.transform_id!r} references unknown components: "
                    f"inputs={unknown_inputs}, outputs={unknown_outputs}"
                )
        for group in self.parallel_groups:
            unknown = [component_id for component_id in group.component_ids if component_id not in component_ids]
            if unknown:
                raise ValueError(f"group {group.group_id!r} references unknown components: {unknown}")
        for codec_pass in self.passes:
            unknown_transforms = [
                transform_id
                for transform_id in codec_pass.vstack_transform_ids
                if transform_id not in transform_ids
            ]
            unknown_groups = [group_id for group_id in codec_pass.hstack_group_ids if group_id not in group_ids]
            if unknown_transforms or unknown_groups:
                raise ValueError(
                    f"pass {codec_pass.pass_id!r} references unknown objects: "
                    f"transforms={unknown_transforms}, groups={unknown_groups}"
                )
        pass_indices = [codec_pass.pass_index for codec_pass in self.passes]
        if pass_indices != sorted(pass_indices):
            raise ValueError("passes must be sorted by pass_index")
        if len(set(pass_indices)) != len(pass_indices):
            raise ValueError("pass_index values must be unique")

    def promotion_status(self) -> PromotionStatus:
        blockers: list[str] = list(self.promotion_policy.default_blockers)
        repairs: list[str] = []
        score_claim = False

        for obj in (*self.components, *self.serial_transforms, *self.parallel_groups, *self.passes):
            byte_semantics = obj.byte_semantics
            score_claim = score_claim or byte_semantics.score_claim
            blockers.extend(obj.all_blockers)
            repairs.extend(byte_semantics.repair_criteria)
            repairs.extend(getattr(obj, "repair_criteria", ()))
            repairs.extend(getattr(obj, "fail_closed_criteria", ()))
            if byte_semantics.score_affecting_payload_changed and not byte_semantics.charged_bits_changed:
                blockers.append("score_affecting_payload_changed_without_charged_bits_proof")

        if score_claim:
            blockers.append("score_claim_present_in_planning_manifest")
        if not any(component.byte_semantics.charged_bits_changed for component in self.components):
            blockers.append("no_component_has_charged_bits_changed_proof")

        unique_blockers = tuple(dict.fromkeys(blockers))
        unique_repairs = tuple(dict.fromkeys(repairs))
        dispatchable = not unique_blockers
        promotion_eligible = dispatchable and score_claim
        return PromotionStatus(
            dispatchable=dispatchable,
            promotion_eligible=promotion_eligible,
            score_claim=score_claim,
            blockers=unique_blockers,
            required_repairs=unique_repairs,
        )

    def to_manifest(self) -> dict[str, Any]:
        payload = {
            "schema": STACK_PLANNER_SCHEMA,
            "plan_id": self.plan_id,
            "anchor_id": self.anchor_id,
            "static_pmf_delta_bytes": self.static_pmf_delta_bytes,
            "metadata": dict(self.metadata),
            "components": [component.to_manifest() for component in self.components],
            "serial_transforms": [transform.to_manifest() for transform in self.serial_transforms],
            "parallel_groups": [group.to_manifest() for group in self.parallel_groups],
            "passes": [codec_pass.to_manifest() for codec_pass in self.passes],
            "promotion_policy": self.promotion_policy.to_manifest(),
            "promotion_status": self.promotion_status().to_manifest(),
        }
        payload["manifest_sha256"] = canonical_json_sha256(payload)
        return payload


def _ids_unique(values: Sequence[str], *, field_name: str) -> set[str]:
    out = set(values)
    if len(out) != len(values):
        duplicates = sorted({value for value in values if values.count(value) > 1})
        raise ValueError(f"{field_name} values must be unique; duplicates: {duplicates}")
    return out


def _planning_semantics(
    *,
    evidence_semantics: str,
    blockers: Sequence[str] = (),
    repair_criteria: Sequence[str] = (),
    evidence_grade: str = "prediction",
    byte_delta_vs_baseline: int | None = None,
    model_overhead_bytes: int | None = None,
) -> ByteEvidenceSemantics:
    return ByteEvidenceSemantics(
        evidence_grade=evidence_grade,
        evidence_semantics=evidence_semantics,
        score_claim=False,
        score_affecting_payload_changed=False,
        charged_bits_changed=False,
        dispatchable=False,
        promotion_eligible=False,
        byte_delta_vs_baseline=byte_delta_vs_baseline,
        model_overhead_bytes=model_overhead_bytes,
        blockers=tuple(blockers),
        repair_criteria=tuple(repair_criteria),
    )


def learned_hyperprior_component(
    *,
    component_id: str = "balle_full_learned_hyperprior",
    static_pmf_delta_bytes: int = DEFAULT_STATIC_PMF_DELTA_BYTES,
    model_overhead_bytes: int | None = None,
) -> CodecComponent:
    """Return the fail-closed Ballé/full learned hyperprior candidate."""
    blockers = tuple(
        blocker
        if blocker != "static_shared_pmf_k12_negative_plus_793_bytes_requires_repair"
        else f"static_shared_pmf_k12_negative_plus_{int(static_pmf_delta_bytes)}_bytes_requires_repair"
        for blocker in BALLE_HYPERPRIOR_BLOCKERS
    )
    return CodecComponent(
        component_id=component_id,
        family="balle_full_learned_hyperprior",
        role="hyperprior",
        stack_axis="vstack_serial",
        streams=(*RENDERER_WEIGHT_BLOCK_FP_STREAMS, *MODEL_WEIGHT_BLOCK_FP_STREAMS),
        depends_on=("quantized_symbol_streams",),
        exact_roundtrip_required=True,
        runtime_packet_required=True,
        byte_semantics=_planning_semantics(
            evidence_semantics="learned_hyperprior_candidate_plan_only_zero_score_claim",
            blockers=blockers,
            repair_criteria=BALLE_HYPERPRIOR_REPAIR_CRITERIA,
            byte_delta_vs_baseline=int(static_pmf_delta_bytes),
            model_overhead_bytes=model_overhead_bytes,
        ),
        blockers=blockers,
        repair_criteria=BALLE_HYPERPRIOR_REPAIR_CRITERIA,
        notes=(
            "full_balle_family_may_replace_static_pmf_only_after_mdl_overhead_win",
            "do_not_touch_shared_pmf_or_balle_codec_implementation_from_planner",
        ),
    )


def build_hstack_vstack_multipass_plan(
    *,
    anchor_id: str = "current_frontier_archive",
    static_pmf_delta_bytes: int = DEFAULT_STATIC_PMF_DELTA_BYTES,
    max_passes: int = DEFAULT_MAX_PASSES,
    include_balle_hyperprior: bool = True,
    learned_model_overhead_bytes: int | None = None,
) -> CodecStackPlan:
    """Build the default deterministic repair plan.

    The returned plan is intentionally non-dispatchable until a later worker
    supplies charged-byte, exact-roundtrip, runtime-packet, and exact CUDA
    evidence. The static PMF delta is recorded as a negative control, not as a
    learned-hyperprior measurement.
    """
    if max_passes < 1 or max_passes > ABSOLUTE_MAX_PASSES:
        raise ValueError(f"max_passes must be in [1, {ABSOLUTE_MAX_PASSES}], got {max_passes}")
    _finite_optional_float(
        None if learned_model_overhead_bytes is None else float(learned_model_overhead_bytes),
        field_name="learned_model_overhead_bytes",
    )

    baseline = CodecComponent(
        component_id="baseline_packet",
        family="contest_archive_packet",
        role="input_anchor",
        stack_axis="source",
        streams=("archive_zip",),
        exact_roundtrip_required=False,
        runtime_packet_required=False,
        byte_semantics=_planning_semantics(
            evidence_grade="empirical",
            evidence_semantics="existing_anchor_for_planning_no_new_score_claim",
            blockers=("old_new_archive_boundary_missing",),
            repair_criteria=("record_anchor_archive_bytes_and_sha256_before_rewrite",),
        ),
        blockers=("old_new_archive_boundary_missing",),
        repair_criteria=("record_anchor_archive_bytes_and_sha256_before_rewrite",),
    )
    quantized = CodecComponent(
        component_id="quantized_symbol_streams",
        family="qint_symbol_streams",
        role="quantization",
        stack_axis="vstack_serial",
        streams=(
            *RENDERER_WEIGHT_BLOCK_FP_STREAMS,
            *MODEL_WEIGHT_BLOCK_FP_STREAMS,
            "mask_qint",
            "pose_qint",
            "residual_qint",
        ),
        depends_on=("baseline_packet",),
        byte_semantics=_planning_semantics(
            evidence_semantics="symbol_stream_contract_plan_only",
            blockers=("canonical_qint_vector_manifest_missing", "symbol_stream_roundtrip_vectors_missing"),
            repair_criteria=("emit_cross_language_qint_vectors_with_shapes_offsets_and_sha256",),
        ),
        blockers=("canonical_qint_vector_manifest_missing", "symbol_stream_roundtrip_vectors_missing"),
        repair_criteria=("emit_cross_language_qint_vectors_with_shapes_offsets_and_sha256",),
    )
    static_control = CodecComponent(
        component_id="static_shared_pmf_k12_negative_control",
        family="static_shared_pmf",
        role="negative_control",
        stack_axis="vstack_serial",
        streams=(*RENDERER_WEIGHT_BLOCK_FP_STREAMS, *MODEL_WEIGHT_BLOCK_FP_STREAMS),
        depends_on=("quantized_symbol_streams",),
        exact_roundtrip_required=True,
        runtime_packet_required=True,
        byte_semantics=_planning_semantics(
            evidence_grade="empirical",
            evidence_semantics="static_shared_pmf_k12_negative_control_after_charged_bytes",
            blockers=("negative_control_not_a_promotion_candidate",),
            repair_criteria=("use_as_floor_and_overhead_warning_only",),
            byte_delta_vs_baseline=int(static_pmf_delta_bytes),
        ),
        blockers=("negative_control_not_a_promotion_candidate",),
        repair_criteria=("use_as_floor_and_overhead_warning_only",),
        notes=(f"static_shared_pmf_k12_measured_delta_vs_brotli_bytes={int(static_pmf_delta_bytes)}",),
    )
    hyperprior = learned_hyperprior_component(
        static_pmf_delta_bytes=static_pmf_delta_bytes,
        model_overhead_bytes=learned_model_overhead_bytes,
    )
    arithmetic = CodecComponent(
        component_id="arithmetic_terminal_coder",
        family="range_arithmetic_terminal",
        role="arithmetic",
        stack_axis="vstack_serial",
        streams=(
            *RENDERER_WEIGHT_BLOCK_FP_STREAMS,
            *MODEL_WEIGHT_BLOCK_FP_STREAMS,
            "mask_qint",
            "pose_qint",
            "residual_qint",
        ),
        depends_on=("quantized_symbol_streams",),
        byte_semantics=_planning_semantics(
            evidence_semantics="terminal_entropy_coder_contract_plan_only",
            blockers=("terminal_coder_payload_vectors_missing", "no_old_new_payload_sha256"),
            repair_criteria=("prove_payload_decode_identity_and_fail_closed_malformed_vectors",),
        ),
        blockers=("terminal_coder_payload_vectors_missing", "no_old_new_payload_sha256"),
        repair_criteria=("prove_payload_decode_identity_and_fail_closed_malformed_vectors",),
    )
    mask = _parallel_stream_component("mask_stream_component", "mask_video_stream", "mask")
    pose = _parallel_stream_component("pose_stream_component", "pose_stream", "pose")
    residual = _parallel_stream_component("residual_stream_component", "residual_correction_stream", "residual")
    runtime_packet = CodecComponent(
        component_id="runtime_packet_materializer",
        family="contest_runtime_packet",
        role="pack",
        stack_axis="vstack_serial",
        streams=("archive_zip", "inflate_runtime"),
        depends_on=("arithmetic_terminal_coder",),
        exact_roundtrip_required=True,
        runtime_packet_required=True,
        byte_semantics=_planning_semantics(
            evidence_semantics="runtime_packet_contract_plan_only",
            blockers=("runtime_packet_not_built", "inflate_consumption_probe_missing", "no_sidecar_audit_missing"),
            repair_criteria=(
                "archive_zip_contains_all_score_affecting_model_side_info",
                "inflate_sh_consumes_packet_without_external_state_or_network",
                "pre_submission_compliance_check_strict_passes",
            ),
        ),
        blockers=("runtime_packet_not_built", "inflate_consumption_probe_missing", "no_sidecar_audit_missing"),
        repair_criteria=(
            "archive_zip_contains_all_score_affecting_model_side_info",
            "inflate_sh_consumes_packet_without_external_state_or_network",
            "pre_submission_compliance_check_strict_passes",
        ),
    )

    components = [baseline, quantized, static_control, arithmetic, mask, pose, residual, runtime_packet]
    if include_balle_hyperprior:
        components.insert(3, hyperprior)

    serial_transforms = _build_serial_transforms(include_balle_hyperprior=include_balle_hyperprior)
    parallel_groups = (
        ParallelComponentGroup(
            group_id="hstack_parallel_stream_components",
            merge_policy="independent_streams_then_deterministic_packet_merge",
            component_ids=(
                "mask_stream_component",
                "pose_stream_component",
                "residual_stream_component",
                "arithmetic_terminal_coder",
            ),
            byte_semantics=_planning_semantics(
                evidence_semantics="hstack_parallel_component_plan_only",
                blockers=(
                    "single_member_internal_section_map_missing",
                    "parallel_stream_byte_ledger_missing",
                    "cross_stream_budget_reconciliation_missing",
                ),
                repair_criteria=(
                    "record_per_stream_internal_offsets_lengths_sha256_and_merge_order_sha256",
                ),
            ),
            blockers=(
                "single_member_internal_section_map_missing",
                "parallel_stream_byte_ledger_missing",
                "cross_stream_budget_reconciliation_missing",
            ),
            repair_criteria=("record_per_stream_internal_offsets_lengths_sha256_and_merge_order_sha256",),
        ),
    )
    passes = _build_passes(max_passes=max_passes, include_balle_hyperprior=include_balle_hyperprior)

    return CodecStackPlan(
        plan_id="hstack_vstack_hyperprior_repair_20260507_worker_h",
        anchor_id=anchor_id,
        static_pmf_delta_bytes=int(static_pmf_delta_bytes),
        components=tuple(components),
        serial_transforms=serial_transforms,
        parallel_groups=parallel_groups,
        passes=passes,
        metadata={
            "target_modes": ["contest_exact_eval"],
            "score_claim": False,
            "dispatch_attempted": False,
            "planner_scope": "components_serial_transforms_parallel_components_passes_byte_semantics_blockers",
            "axis_semantics": dict(HSTACK_VSTACK_AXIS_SEMANTICS),
            "nested_optimization": {
                "levels": list(NESTED_OPTIMIZATION_LEVELS),
                "score_band_prediction": dict(PREDICTED_NESTED_SCORE_BAND),
                "status": "design_contract_only_no_score_claim",
            },
            "canonical_qat_pipeline": {
                "passes": list(CANONICAL_QAT_PASSES),
                "status": "required_pipeline_shape_not_training_evidence",
            },
            "quality_mandate": list(QUALITY_MANDATE_11),
            "archive_layout": {
                "shape": "single_member_monolithic_packet_with_internal_parser_proven_logical_sections",
                "status": "planning_contract_only_after_frontier_monolith_review",
                "member_level_component_budgets_valid": False,
                "logical_stream_budget_requires_internal_parser_proof": True,
                "physical_packet_target": "one_stored_zip_member_for_pr101_pr106_style_frontier_archives",
                "dispatch_requires": [
                    "codecop_dag_manifest",
                    "single_member_name_size_sha256",
                    "internal_section_offset_len_sha256",
                    "no_member_level_mask_pose_budget_claim",
                    "charged_byte_delta_per_node",
                    "decode_roundtrip_per_leaf",
                ],
            },
            "negative_prior": {
                "family": "static_shared_pmf_k12",
                "delta_bytes_vs_brotli_after_charged_bytes": int(static_pmf_delta_bytes),
                "semantics": "negative_control_not_family_kill_for_learned_hyperprior",
            },
        },
    )


def _parallel_stream_component(component_id: str, family: str, stream_name: str) -> CodecComponent:
    return CodecComponent(
        component_id=component_id,
        family=family,
        role="parallel_stream",
        stack_axis="hstack_parallel",
        streams=(stream_name,),
        depends_on=("baseline_packet",),
        exact_roundtrip_required=True,
        runtime_packet_required=True,
        byte_semantics=_planning_semantics(
            evidence_semantics=f"{stream_name}_parallel_stream_plan_only",
            blockers=(
                f"{stream_name}_internal_parser_section_missing",
                f"{stream_name}_charged_byte_ledger_missing",
                f"{stream_name}_roundtrip_vector_missing",
            ),
            repair_criteria=(f"record_{stream_name}_offset_len_sha256_and_decode_vector",),
        ),
        blockers=(
            f"{stream_name}_internal_parser_section_missing",
            f"{stream_name}_charged_byte_ledger_missing",
            f"{stream_name}_roundtrip_vector_missing",
        ),
        repair_criteria=(f"record_{stream_name}_offset_len_sha256_and_decode_vector",),
    )


def _build_serial_transforms(*, include_balle_hyperprior: bool) -> tuple[SerialTransform, ...]:
    transforms = [
        SerialTransform(
            transform_id="vstack_deconstruct_to_qint_streams",
            order_index=0,
            transform_kind="representation_to_quantized_symbols",
            inputs=("baseline_packet",),
            outputs=("quantized_symbol_streams",),
            byte_semantics=_planning_semantics(
                evidence_semantics="deconstruction_transform_plan_only",
                blockers=("qint_deconstruction_not_proven",),
                repair_criteria=("emit_qint_deconstruction_manifest_with_sha256",),
            ),
            blockers=("qint_deconstruction_not_proven",),
            repair_criteria=("emit_qint_deconstruction_manifest_with_sha256",),
        )
    ]
    if include_balle_hyperprior:
        transforms.append(
            SerialTransform(
                transform_id="vstack_learned_hyperprior_repair",
                order_index=1,
                transform_kind="quantized_symbols_to_learned_hyperprior_sideinfo",
                inputs=("quantized_symbol_streams",),
                outputs=("balle_full_learned_hyperprior",),
                byte_semantics=_planning_semantics(
                    evidence_semantics="learned_hyperprior_transform_plan_only",
                    blockers=BALLE_HYPERPRIOR_BLOCKERS,
                    repair_criteria=BALLE_HYPERPRIOR_REPAIR_CRITERIA,
                ),
                blockers=BALLE_HYPERPRIOR_BLOCKERS,
                repair_criteria=BALLE_HYPERPRIOR_REPAIR_CRITERIA,
            )
        )
        arithmetic_inputs = ("balle_full_learned_hyperprior",)
        order_offset = 0
    else:
        arithmetic_inputs = ("static_shared_pmf_k12_negative_control",)
        order_offset = -1
    transforms.extend(
        [
            SerialTransform(
                transform_id="vstack_arithmetic_terminal_encode",
                order_index=2 + order_offset,
                transform_kind="entropy_model_to_arithmetic_payload",
                inputs=arithmetic_inputs,
                outputs=("arithmetic_terminal_coder",),
                byte_semantics=_planning_semantics(
                    evidence_semantics="arithmetic_terminal_transform_plan_only",
                    blockers=("arithmetic_payload_identity_vectors_missing",),
                    repair_criteria=("prove_arithmetic_payload_roundtrip_and_malformed_fail_closed",),
                ),
                blockers=("arithmetic_payload_identity_vectors_missing",),
                repair_criteria=("prove_arithmetic_payload_roundtrip_and_malformed_fail_closed",),
            ),
            SerialTransform(
                transform_id="vstack_packet_materialize",
                order_index=3 + order_offset,
                transform_kind="payloads_to_runtime_packet",
                inputs=("arithmetic_terminal_coder",),
                outputs=("runtime_packet_materializer",),
                byte_semantics=_planning_semantics(
                    evidence_semantics="packet_materialization_transform_plan_only",
                    blockers=("runtime_packet_materialization_missing",),
                    repair_criteria=("prove_runtime_packet_contains_all_score_affecting_bytes",),
                ),
                blockers=("runtime_packet_materialization_missing",),
                repair_criteria=("prove_runtime_packet_contains_all_score_affecting_bytes",),
            ),
        ]
    )
    return tuple(transforms)


def _build_passes(*, max_passes: int, include_balle_hyperprior: bool) -> tuple[CodecPass, ...]:
    hyperprior_transforms = ("vstack_learned_hyperprior_repair",) if include_balle_hyperprior else ()
    blueprints = [
        (
            "pass_00_anchor_deconstruction",
            "anchor deconstruction plus static PMF negative-control accounting",
            ("vstack_deconstruct_to_qint_streams",),
            (),
            ("old_new_archive_boundary_missing", "qint_deconstruction_not_proven"),
        ),
        (
            "pass_01_vstack_hyperprior_repair",
            "serial VStack learned-hyperprior repair and arithmetic payload contract",
            (*hyperprior_transforms, "vstack_arithmetic_terminal_encode"),
            (),
            ("learned_hyperprior_repair_not_proven", "arithmetic_payload_identity_vectors_missing"),
        ),
        (
            "pass_02_hstack_packet_closure",
            "parallel HStack byte reconciliation and runtime packet closure",
            ("vstack_packet_materialize",),
            ("hstack_parallel_stream_components",),
            ("parallel_stream_byte_ledger_missing", "runtime_packet_not_built"),
        ),
    ]
    while len(blueprints) < max_passes:
        idx = len(blueprints)
        blueprints.append(
            (
                f"pass_{idx:02d}_multipass_refinement",
                "compress-time multipass refinement with no inflate-time scorer",
                ("vstack_arithmetic_terminal_encode", "vstack_packet_materialize"),
                ("hstack_parallel_stream_components",),
                ("multipass_refinement_exact_eval_gate_missing",),
            )
        )

    passes: list[CodecPass] = []
    for idx, (pass_id, purpose, vstack_ids, hstack_ids, blockers) in enumerate(blueprints[:max_passes]):
        passes.append(
            CodecPass(
                pass_id=pass_id,
                pass_index=idx,
                purpose=purpose,
                vstack_transform_ids=tuple(vstack_ids),
                hstack_group_ids=tuple(hstack_ids),
                evidence_gate="planning_only_no_score_claim_exact_roundtrip_runtime_packet_required",
                byte_semantics=_planning_semantics(
                    evidence_semantics="multipass_plan_stage_no_archive_score_claim",
                    blockers=blockers,
                    repair_criteria=MULTIPASS_FAIL_CLOSED_CRITERIA,
                ),
                blockers=blockers,
                fail_closed_criteria=MULTIPASS_FAIL_CLOSED_CRITERIA,
            )
        )
    return tuple(passes)


def write_plan_manifest(plan: CodecStackPlan, path: str) -> None:
    """Write a plan manifest as canonical pretty JSON."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(plan.to_manifest(), f, indent=2, sort_keys=True)
        f.write("\n")


def plan_from_manifest(manifest: Mapping[str, Any]) -> CodecStackPlan:
    """Reconstruct a plan from a manifest emitted by :meth:`to_manifest`."""
    supported_schemas = {STACK_PLANNER_SCHEMA, "tac_hstack_vstack_multipass_plan_v1"}
    if manifest.get("schema") not in supported_schemas:
        raise ValueError(f"unsupported codec stack plan schema: {manifest.get('schema')!r}")
    components = tuple(
        CodecComponent(
            component_id=item["component_id"],
            family=item["family"],
            role=item["role"],
            stack_axis=item["stack_axis"],
            streams=tuple(item["streams"]),
            depends_on=tuple(item["depends_on"]),
            exact_roundtrip_required=bool(item["exact_roundtrip_required"]),
            runtime_packet_required=bool(item["runtime_packet_required"]),
            deterministic=bool(item["deterministic"]),
            byte_semantics=_byte_semantics_from_manifest(item["byte_semantics"]),
            blockers=tuple(item["blockers"]),
            repair_criteria=tuple(item["repair_criteria"]),
            notes=tuple(item.get("notes", ())),
        )
        for item in manifest["components"]
    )
    serial_transforms = tuple(
        SerialTransform(
            transform_id=item["transform_id"],
            order_index=int(item["order_index"]),
            transform_kind=item["transform_kind"],
            inputs=tuple(item["inputs"]),
            outputs=tuple(item["outputs"]),
            byte_semantics=_byte_semantics_from_manifest(item["byte_semantics"]),
            blockers=tuple(item["blockers"]),
            repair_criteria=tuple(item["repair_criteria"]),
        )
        for item in manifest["serial_transforms"]
    )
    parallel_groups = tuple(
        ParallelComponentGroup(
            group_id=item["group_id"],
            merge_policy=item["merge_policy"],
            component_ids=tuple(item["component_ids"]),
            byte_semantics=_byte_semantics_from_manifest(item["byte_semantics"]),
            blockers=tuple(item["blockers"]),
            repair_criteria=tuple(item["repair_criteria"]),
        )
        for item in manifest["parallel_groups"]
    )
    passes = tuple(
        CodecPass(
            pass_id=item["pass_id"],
            pass_index=int(item["pass_index"]),
            purpose=item["purpose"],
            vstack_transform_ids=tuple(item.get("vstack_transform_ids", item.get("hstack_transform_ids", ()))),
            hstack_group_ids=tuple(item.get("hstack_group_ids", item.get("vstack_group_ids", ()))),
            evidence_gate=item["evidence_gate"],
            byte_semantics=_byte_semantics_from_manifest(item["byte_semantics"]),
            blockers=tuple(item["blockers"]),
            fail_closed_criteria=tuple(item["fail_closed_criteria"]),
        )
        for item in manifest["passes"]
    )
    return CodecStackPlan(
        plan_id=manifest["plan_id"],
        anchor_id=manifest["anchor_id"],
        static_pmf_delta_bytes=int(manifest["static_pmf_delta_bytes"]),
        components=components,
        serial_transforms=serial_transforms,
        parallel_groups=parallel_groups,
        passes=passes,
        metadata=dict(manifest.get("metadata", {})),
    )


def _byte_semantics_from_manifest(item: Mapping[str, Any]) -> ByteEvidenceSemantics:
    return ByteEvidenceSemantics(
        evidence_grade=str(item["evidence_grade"]),
        evidence_semantics=str(item["evidence_semantics"]),
        score_claim=bool(item["score_claim"]),
        score_affecting_payload_changed=bool(item["score_affecting_payload_changed"]),
        charged_bits_changed=bool(item["charged_bits_changed"]),
        dispatchable=bool(item["dispatchable"]),
        promotion_eligible=bool(item["promotion_eligible"]),
        measured_archive_bytes=item.get("measured_archive_bytes"),
        estimated_archive_bytes=item.get("estimated_archive_bytes"),
        byte_delta_vs_baseline=item.get("byte_delta_vs_baseline"),
        model_overhead_bytes=item.get("model_overhead_bytes"),
        blockers=tuple(item.get("blockers", ())),
        repair_criteria=tuple(item.get("repair_criteria", ())),
    )


def summarize_plan(plan: CodecStackPlan) -> dict[str, Any]:
    """Return a compact deterministic operator summary."""
    status = plan.promotion_status()
    return {
        "schema": "tac_hstack_vstack_multipass_plan_summary_v2",
        "plan_id": plan.plan_id,
        "anchor_id": plan.anchor_id,
        "components": len(plan.components),
        "serial_transforms": len(plan.serial_transforms),
        "parallel_groups": len(plan.parallel_groups),
        "passes": len(plan.passes),
        "score_claim": status.score_claim,
        "dispatchable": status.dispatchable,
        "promotion_eligible": status.promotion_eligible,
        "blocker_count": len(status.blockers),
        "required_repair_count": len(status.required_repairs),
        "static_pmf_delta_bytes": plan.static_pmf_delta_bytes,
        "nested_optimization_levels": list(NESTED_OPTIMIZATION_LEVELS),
        "predicted_nested_score_band": dict(PREDICTED_NESTED_SCORE_BAND),
        "manifest_sha256": plan.to_manifest()["manifest_sha256"],
    }
