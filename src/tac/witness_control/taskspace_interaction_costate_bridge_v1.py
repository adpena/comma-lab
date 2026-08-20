# SPDX-License-Identifier: MIT
"""Exact-base G8/A feedback -> advisory costate-control intelligence (v1).

This is a pure adapter over the frozen G18 feedback receipt.  It does not read a
pointer, archive, run directory, scorer, or ledger.  The caller must supply the
independently observed current pointer/base/population binding; any mismatch is
fatal.  A transition and a four-cell G8-by-A interaction are separate, indivisible
control observations.  In particular, an interaction residual is never spread over
or added to atomic action marginals.

The only live consumer invoked here is the side-effect-free
``control_alphabet.powerplay_acquisition`` ranker.  The emitted ``stage`` rows are
accepted unchanged by ``telemetry_binding.parse_log_lines``.  State-changing
consumer APIs receive typed blocked handoffs because n2 macOS advisory feedback is
not an n600 costate posterior, a master-gradient tensor, or an allocator proposal.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, Final, Literal

from tac.witness_control.control_alphabet import powerplay_acquisition
from tac.witness_control.taskspace_receding_horizon_controller_v1 import (
    ACTION_FAMILIES as TERMINAL_ACTION_FAMILIES,
)
from tac.witness_control.taskspace_receding_horizon_controller_v1 import (
    ACTION_SCALES as TERMINAL_ACTION_SCALES,
)
from tac.witness_control.taskspace_receding_horizon_controller_v1 import (
    SCHEMA as TERMINAL_CONTROLLER_SCHEMA,
)
from tac.witness_dsl.pair_population_envelope import PairDomain, PairPopulation
from tac.witness_dsl.taskspace_g8_a3_interaction_feedback import (
    AXIS as G18_AXIS,
)
from tac.witness_dsl.taskspace_g8_a3_interaction_feedback import (
    SCHEMA as G18_SCHEMA,
)
from tac.witness_dsl.taskspace_g8_a3_interaction_feedback import (
    feedback_receipt_bytes,
)

SCHEMA: Final = "tac.taskspace_interaction_costate_bridge.v1"
STATE_SCHEMA: Final = "tac.taskspace_interaction_costate_controller_state.v1"
TELEMETRY_STAGE: Final = "taskspace_interaction_costate_calibration"
SOURCE_SCHEMA: Final = "tac.taskspace_g8_a3_n2_allocator.v1"
RATE_DENOMINATOR: Final = 37_545_489
POPULATION_PAIR_COUNT: Final = 600
G22_RECEIPT_SCHEMA: Final = "tac.g22_ep725_xcodec_decode_receipt.v1"
G20_RECEIPT_SCHEMA: Final = "tac.ep725_lossless_xcodec_recode.v1"
G25_RECEIPT_SCHEMA: Final = "tac.ep725_population_global_recode.v2"
G28_RECEIPT_SCHEMA: Final = "tac.taskspace_ep725_same_object_bridge_eval.v1"
G22_STRUCTURAL_AXIS: Final = "[macOS-CPU frozen receiver structural proof]"
_G22_CLOSURE_PROOF: Final = object()

EffectKind = Literal["transition", "interaction"]
TeacherReplayMode = Literal["HISTORICAL_HINT_ONLY", "CURRENT_EXACT_RECEIVER_REPLAY"]
AnalyticLearnedFactor = Literal["analytic", "learned", "hybrid"]
ActionSemantics = Literal[
    "ADD",
    "DELETE_PRUNE",
    "MERGE_SHARE",
    "FACTOR",
    "REPLACE",
    "MIGRATE",
    "REQUANTIZE_STORAGE",
    "GAUGE_REPRESENTATIVE",
]

ACTION_SEMANTICS: Final = (
    "ADD",
    "DELETE_PRUNE",
    "MERGE_SHARE",
    "FACTOR",
    "REPLACE",
    "MIGRATE",
    "REQUANTIZE_STORAGE",
    "GAUGE_REPRESENTATIVE",
)
COSTATE_GRANULARITIES: Final = ("atom", "group", "shard", "global")
COSTATE_FACTOR_AXES: Final = (
    "weights",
    "codes",
    "time",
    "population",
    "semantic",
    "realization",
    "pose",
    "entropy",
    "analytic_vs_learned",
)
GENERIC_DECODER_OPERATIONS: Final = (
    "solve",
    "repair",
    "synthesize",
    "project",
    "factor_expand",
    "gauge_select",
    "iterate",
    "entropy_decode",
    "procedural_basis",
)
RESERVOIRS: Final = (
    "base",
    "population_code",
    "semantic_grammar",
    "realization",
    "A",
    "learned_residual",
    "entropy_context",
)

_SHA256_RE: Final = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER_RE: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,191}\Z")
_TOP_FIELDS: Final = {
    "schema",
    "source_receipt_sha256",
    "canonical_receipt_sha256",
    "source_schema",
    "source_lane_id",
    "axis",
    "truth",
    "latest_pointer_comparison",
    "pointer_observation_custody",
    "four_way_z_t_h_partition",
    "g12_acquisition",
    "diagnostic_exact_semantic_g_control",
    "row_inventory",
    "transition_inventory",
    "interaction_inventory",
    "downstream_payloads",
    "hook_coverage",
    "dense_frames_or_masks_serialized",
    "live_run_scorer_eval_or_ledger_access_performed",
}
_TRUTH: Final = {
    "authoritative_contest_cpu_evaluation": False,
    "authoritative_contest_cuda_evaluation": False,
    "candidate_archive_eligible": False,
    "dense_frames_persisted": False,
    "macos_cpu_advisory": True,
    "n2_only": True,
    "n600_evaluation": False,
    "originality_claim": False,
    "pointer_moved": False,
    "promotion_eligible": False,
    "public_archive_payload_reused": False,
    "ready_for_exact_eval_dispatch": False,
    "research_only": True,
    "score_claim": False,
}
_MEASUREMENT_REQUIRED: Final = {
    "measurement_id",
    "baseline_bundle_sha256",
    "bundle_sha256",
    "d_seg",
    "d_pose",
    "selected_archive_bytes",
    "selected_archive_sha256",
    "selected_encoding",
    "stored_archive_bytes",
    "stored_archive_sha256",
    "deflated_archive_bytes",
    "deflated_archive_sha256",
    "member_bytes",
    "member_sha256",
    "decoded_output_sha256",
    "receiver_receipt_sha256",
    "derived_component_total",
    "scorer_evidence",
    "a_mode",
    "a_row_count",
    "a_program_sha256",
    "a_source_binding_sha256",
    "camera_y1_sha256",
    "candidate_seg_labels_sha256",
}
_TRANSITION_FIELDS: Final = {
    "transition_id",
    "kind",
    "after_occurrence_id",
    "before_measurement_id",
    "after_measurement_id",
    "delta_d_seg",
    "delta_d_pose",
    "delta_selected_archive_bytes",
    "seg_term_delta",
    "pose_term_delta",
    "rate_term_delta",
    "distortion_term_delta",
    "exact_score_delta",
    "exact_score_improvement",
    "finite_byte_ceiling_real",
    "greatest_strict_integer_byte_delta",
    "improves_score",
}


class TaskspaceInteractionCostateBridgeError(ValueError):
    """Raised before output when feedback, binding, or state loses custody."""


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise TaskspaceInteractionCostateBridgeError("value is not finite canonical JSON") from exc


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _require_sha256(value: object, field_name: str) -> str:
    if type(value) is not str or _SHA256_RE.fullmatch(value) is None:
        raise TaskspaceInteractionCostateBridgeError(f"{field_name} must be canonical lowercase SHA-256")
    return value


def _require_identifier(value: object, field_name: str) -> str:
    if type(value) is not str or _IDENTIFIER_RE.fullmatch(value) is None:
        raise TaskspaceInteractionCostateBridgeError(f"{field_name} is not a bounded identifier")
    return value


def _finite(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TaskspaceInteractionCostateBridgeError(f"{field_name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise TaskspaceInteractionCostateBridgeError(f"{field_name} must be finite")
    return result


def _close(left: object, right: object, field_name: str, *, atol: float = 1e-12) -> None:
    if not math.isclose(
        _finite(left, field_name),
        _finite(right, field_name),
        rel_tol=0.0,
        abs_tol=atol,
    ):
        raise TaskspaceInteractionCostateBridgeError(f"{field_name} arithmetic drifted")


def _score(d_seg: float, d_pose: float, archive_bytes: int) -> float:
    if d_seg < 0.0 or d_pose < 0.0 or archive_bytes < 1:
        raise TaskspaceInteractionCostateBridgeError("score components left their physical domain")
    return 100.0 * d_seg + math.sqrt(10.0 * d_pose) + 25.0 * archive_bytes / RATE_DENOMINATOR


def _distortion(d_seg: float, d_pose: float) -> float:
    if d_seg < 0.0 or d_pose < 0.0:
        raise TaskspaceInteractionCostateBridgeError("distortion components left their physical domain")
    return 100.0 * d_seg + math.sqrt(10.0 * d_pose)


def _validate_action_signature(
    action_semantics: object,
    source_reservoirs: object,
    target_reservoirs: object,
    semantic_evidence_sha256: object,
    *,
    collection_type: type,
) -> None:
    if action_semantics not in ACTION_SEMANTICS:
        raise TaskspaceInteractionCostateBridgeError("action_semantics left the closed vocabulary")
    if type(source_reservoirs) is not collection_type or type(target_reservoirs) is not collection_type:
        raise TaskspaceInteractionCostateBridgeError(f"reservoir collections must be {collection_type.__name__}")
    if any(item not in RESERVOIRS for item in (*source_reservoirs, *target_reservoirs)):
        raise TaskspaceInteractionCostateBridgeError("action names an unknown byte/semantic reservoir")
    if len(set(source_reservoirs)) != len(source_reservoirs) or len(set(target_reservoirs)) != len(target_reservoirs):
        raise TaskspaceInteractionCostateBridgeError("reservoir lists may not repeat entries")
    if action_semantics == "ADD" and (source_reservoirs or not target_reservoirs):
        raise TaskspaceInteractionCostateBridgeError("ADD requires target reservoirs and no source")
    if action_semantics == "DELETE_PRUNE" and (not source_reservoirs or target_reservoirs):
        raise TaskspaceInteractionCostateBridgeError("DELETE_PRUNE requires source reservoirs only")
    if action_semantics == "MERGE_SHARE" and (len(source_reservoirs) < 2 or not target_reservoirs):
        raise TaskspaceInteractionCostateBridgeError("MERGE_SHARE requires >=2 sources and a target")
    if action_semantics == "FACTOR" and (
        not source_reservoirs or len(target_reservoirs) < 2 or set(source_reservoirs) == set(target_reservoirs)
    ):
        raise TaskspaceInteractionCostateBridgeError("FACTOR requires source reservoirs and >=2 different targets")
    if action_semantics in ("REPLACE", "MIGRATE") and (not source_reservoirs or not target_reservoirs):
        raise TaskspaceInteractionCostateBridgeError(f"{action_semantics} requires source and target reservoirs")
    if action_semantics == "MIGRATE" and source_reservoirs == target_reservoirs:
        raise TaskspaceInteractionCostateBridgeError("MIGRATE source and target must differ")
    if action_semantics == "REQUANTIZE_STORAGE" and (
        not source_reservoirs or source_reservoirs != target_reservoirs or semantic_evidence_sha256 is None
    ):
        raise TaskspaceInteractionCostateBridgeError(
            "REQUANTIZE_STORAGE requires unchanged reservoirs and an equivalence receipt SHA"
        )
    if action_semantics == "GAUGE_REPRESENTATIVE" and (
        len(source_reservoirs) != 1 or source_reservoirs != target_reservoirs or semantic_evidence_sha256 is None
    ):
        raise TaskspaceInteractionCostateBridgeError(
            "GAUGE_REPRESENTATIVE requires one unchanged reservoir and an equivalence receipt SHA"
        )
    if semantic_evidence_sha256 is not None:
        _require_sha256(semantic_evidence_sha256, "semantic_evidence_sha256")


@dataclass(frozen=True, slots=True)
class FeedbackExactBaseBindingV1:
    """Independent current-context binding required before G18 can inform control."""

    feedback_sha256: str
    source_receipt_sha256: str
    canonical_receipt_sha256: str
    source_lane_id: str
    axis: str
    baseline_bundle_sha256: str
    baseline_archive_sha256: str
    baseline_archive_nbytes: int
    selected_research_archive_sha256: str
    frozen_scorer_sha256: str
    pointer_manifest_sha256: str
    pointer_start_sha256: str
    pointer_start_target_score: float
    pointer_latest_sha256: str
    pointer_latest_target_score: float
    population_manifest_sha256: str
    source_pair_ids: tuple[int, ...]
    population_pair_count: int = POPULATION_PAIR_COUNT

    def __post_init__(self) -> None:
        for field_name in (
            "feedback_sha256",
            "source_receipt_sha256",
            "canonical_receipt_sha256",
            "baseline_bundle_sha256",
            "baseline_archive_sha256",
            "selected_research_archive_sha256",
            "frozen_scorer_sha256",
            "pointer_manifest_sha256",
            "pointer_start_sha256",
            "pointer_latest_sha256",
            "population_manifest_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name)
        _require_identifier(self.source_lane_id, "source_lane_id")
        if self.axis != G18_AXIS:
            raise TaskspaceInteractionCostateBridgeError(f"axis must remain the frozen G18 advisory axis {G18_AXIS!r}")
        if type(self.baseline_archive_nbytes) is not int or self.baseline_archive_nbytes < 1:
            raise TaskspaceInteractionCostateBridgeError("baseline_archive_nbytes must be positive int")
        _finite(self.pointer_start_target_score, "pointer_start_target_score")
        _finite(self.pointer_latest_target_score, "pointer_latest_target_score")
        if self.population_pair_count != POPULATION_PAIR_COUNT:
            raise TaskspaceInteractionCostateBridgeError("v1 population_pair_count must remain 600")
        if (
            type(self.source_pair_ids) is not tuple
            or len(self.source_pair_ids) != 2
            or any(type(pair_id) is not int or pair_id < 0 for pair_id in self.source_pair_ids)
            or len(set(self.source_pair_ids)) != len(self.source_pair_ids)
        ):
            raise TaskspaceInteractionCostateBridgeError("v1 source_pair_ids must be two unique n600 indices")


def _canonical_receipt_payload(payload: bytes, *, label: str) -> tuple[dict[str, Any], bytes]:
    if type(payload) is not bytes:
        raise TaskspaceInteractionCostateBridgeError(f"{label} must be exact bytes")
    try:
        value = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise TaskspaceInteractionCostateBridgeError(f"{label} is not JSON") from exc
    if type(value) is not dict:
        raise TaskspaceInteractionCostateBridgeError(f"{label} root must be one object")
    canonical = _canonical_json(value) + b"\n"
    if payload != canonical:
        raise TaskspaceInteractionCostateBridgeError(f"{label} bytes are not canonical")
    return value, canonical


def _custody_row(value: object, *, label: str, absolute_path: bool = False) -> dict[str, Any]:
    if type(value) is not dict or set(value) != {"path", "bytes", "sha256"}:
        raise TaskspaceInteractionCostateBridgeError(f"{label} custody row drifted")
    path = value["path"]
    if type(path) is not str or not path or (absolute_path and not path.startswith("/")):
        raise TaskspaceInteractionCostateBridgeError(f"{label} path custody drifted")
    if type(value["bytes"]) is not int or value["bytes"] < 1:
        raise TaskspaceInteractionCostateBridgeError(f"{label} byte custody drifted")
    _require_sha256(value["sha256"], f"{label}.sha256")
    return value


@dataclass(frozen=True, slots=True)
class ReceiverEqualityAdmissionControlV1:
    """Mutable frontier context kept outside immutable decode-proof dependencies."""

    pointer_artifact_sha256: str
    pointer_artifact_nbytes: int
    semantic_target_sha256: str

    def __post_init__(self) -> None:
        _require_sha256(self.pointer_artifact_sha256, "admission pointer artifact SHA-256")
        _require_sha256(self.semantic_target_sha256, "admission semantic target SHA-256")
        if type(self.pointer_artifact_nbytes) is not int or self.pointer_artifact_nbytes < 1:
            raise TaskspaceInteractionCostateBridgeError("admission pointer artifact bytes must be positive int")

    @classmethod
    def derive(
        cls,
        *,
        pointer_artifact_sha256: str,
        pointer_artifact_nbytes: int,
        semantic_target: Mapping[str, Any],
    ) -> ReceiverEqualityAdmissionControlV1:
        if not isinstance(semantic_target, Mapping):
            raise TaskspaceInteractionCostateBridgeError("semantic target control must be an object")
        return cls(
            pointer_artifact_sha256=pointer_artifact_sha256,
            pointer_artifact_nbytes=pointer_artifact_nbytes,
            semantic_target_sha256=_sha256(_canonical_json(dict(semantic_target))),
        )


@dataclass(frozen=True, slots=True)
class FullN600ReceiverEqualityClosureV1:
    """Typed G22 full-population decode equality derived from exact receipt bytes.

    Pointer artifacts and competitive targets are retained as observations only;
    neither participates in ``decode_dependency_sha256``.
    """

    g22_receipt_sha256: str
    g22_receipt_nbytes: int
    g20_receipt_sha256: str
    g20_receipt_nbytes: int
    pair_population_sha256: str
    pair_population_envelope_sha256: str
    source_archive_sha256: str
    source_archive_nbytes: int
    source_member_sha256: str
    source_member_nbytes: int
    selected_archive_sha256: str
    selected_archive_nbytes: int
    selected_member_sha256: str
    selected_member_nbytes: int
    decoder_runtime_sha256: str
    decoder_runtime_nbytes: int
    decoded_quantized_state_sha256: str
    realized_output_sha256: str
    realized_output_nbytes: int
    pair_order_sha256: str
    replay_tool_sha256: str
    decode_dependency_sha256: str
    pointer_start_artifact_sha256: str
    pointer_end_artifact_sha256: str
    semantic_target_start_sha256: str
    semantic_target_end_sha256: str
    receipt_custody_sha256: str
    _proof: object = None

    def __post_init__(self) -> None:
        if self._proof is not _G22_CLOSURE_PROOF:
            raise TaskspaceInteractionCostateBridgeError(
                "full-n600 receiver equality must be derived from exact G20/G22/population bytes"
            )
        for field_name in (
            "g22_receipt_sha256",
            "g20_receipt_sha256",
            "pair_population_sha256",
            "pair_population_envelope_sha256",
            "source_archive_sha256",
            "source_member_sha256",
            "selected_archive_sha256",
            "selected_member_sha256",
            "decoder_runtime_sha256",
            "decoded_quantized_state_sha256",
            "realized_output_sha256",
            "pair_order_sha256",
            "replay_tool_sha256",
            "decode_dependency_sha256",
            "pointer_start_artifact_sha256",
            "pointer_end_artifact_sha256",
            "semantic_target_start_sha256",
            "semantic_target_end_sha256",
            "receipt_custody_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name)
        for field_name in (
            "g22_receipt_nbytes",
            "g20_receipt_nbytes",
            "source_archive_nbytes",
            "source_member_nbytes",
            "selected_archive_nbytes",
            "selected_member_nbytes",
            "decoder_runtime_nbytes",
            "realized_output_nbytes",
        ):
            if type(getattr(self, field_name)) is not int or getattr(self, field_name) < 1:
                raise TaskspaceInteractionCostateBridgeError(f"{field_name} must be positive int")

    @classmethod
    def from_receipts(
        cls,
        *,
        g20_receipt: bytes,
        g22_receipt: bytes,
        pair_population: bytes,
    ) -> FullN600ReceiverEqualityClosureV1:
        g20, g20_canonical = _canonical_receipt_payload(g20_receipt, label="G20 receipt")
        g22, g22_canonical = _canonical_receipt_payload(g22_receipt, label="G22 receipt")
        if g20.get("schema") != G20_RECEIPT_SCHEMA or g22.get("schema") != G22_RECEIPT_SCHEMA:
            raise TaskspaceInteractionCostateBridgeError("G20/G22 receipt schema differs")
        try:
            population = PairPopulation.from_bytes(pair_population)
        except Exception as exc:
            raise TaskspaceInteractionCostateBridgeError(
                "PairPopulation bytes are not strict canonical serialized custody"
            ) from exc
        canonical_ids = tuple(range(POPULATION_PAIR_COUNT))
        if population.source_pair_ids != canonical_ids or any(
            population.domain_index(domain).local_to_source_pair_ids != canonical_ids for domain in PairDomain
        ):
            raise TaskspaceInteractionCostateBridgeError(
                "G22 closure requires canonical ordered n600 PairPopulation in every domain"
            )

        archive = g22.get("archive_artifact")
        runtime = g22.get("generic_decoder_runtime")
        rewrite = g22.get("encoder_only_rewrite_evidence")
        decode = g22.get("decode_receipt")
        output = g22.get("output_witness")
        authority = g22.get("authority_pointer_status")
        cleanup = g22.get("cleanup")
        reproducibility = g22.get("reproducibility")
        if any(
            type(row) is not dict
            for row in (archive, runtime, rewrite, decode, output, authority, cleanup, reproducibility)
        ):
            raise TaskspaceInteractionCostateBridgeError("G22 full receipt sections drifted")
        source = archive.get("source")
        selected = archive.get("selected")
        if type(source) is not dict or type(selected) is not dict:
            raise TaskspaceInteractionCostateBridgeError("G22 archive artifacts drifted")
        source_member = source.get("member")
        selected_member = selected.get("member")
        if type(source_member) is not dict or type(selected_member) is not dict:
            raise TaskspaceInteractionCostateBridgeError("G22 archive members drifted")
        for label, row in (("source member", source_member), ("selected member", selected_member)):
            if row.get("name") != "0.bin" or type(row.get("bytes")) is not int or row["bytes"] < 1:
                raise TaskspaceInteractionCostateBridgeError(f"G22 {label} shape drifted")
            _require_sha256(row.get("sha256"), f"G22 {label} SHA-256")

        g20_source = g20.get("source")
        g20_selected = g20.get("selected")
        g20_runtime = g20.get("runtime")
        g20_proof = g20.get("proof")
        if any(type(row) is not dict for row in (g20_source, g20_selected, g20_runtime, g20_proof)):
            raise TaskspaceInteractionCostateBridgeError("G20 closure sections drifted")
        exact_matches = {
            "source archive SHA": (source.get("sha256"), g20_source.get("archive_sha256")),
            "source archive bytes": (source.get("bytes"), g20_source.get("archive_bytes")),
            "source member SHA": (source_member.get("sha256"), g20_source.get("member_sha256")),
            "source member bytes": (source_member.get("bytes"), g20_source.get("member_bytes")),
            "selected archive SHA": (selected.get("sha256"), g20_selected.get("archive_sha256")),
            "selected archive bytes": (selected.get("bytes"), g20_selected.get("archive_bytes")),
            "selected member SHA": (selected_member.get("sha256"), g20_selected.get("member_sha256")),
            "selected member bytes": (selected_member.get("bytes"), g20_selected.get("member_bytes")),
            "runtime SHA": (runtime.get("sha256"), g20_runtime.get("sha256")),
            "runtime bytes": (runtime.get("bytes"), g20_runtime.get("bytes")),
        }
        for label, (observed, expected) in exact_matches.items():
            if observed != expected:
                raise TaskspaceInteractionCostateBridgeError(f"G22/G20 {label} custody mismatch")
        g20_sha = _sha256(g20_canonical)
        g20_row = rewrite.get("g20_receipt")
        if (
            type(g20_row) is not dict
            or g20_row.get("sha256") != g20_sha
            or g20_row.get("bytes") != len(g20_canonical)
            or rewrite.get("decoded_state_equal") is not True
            or rewrite.get("source_state_sha256") != rewrite.get("selected_state_sha256")
            or rewrite.get("source_state_sha256") != g20_proof.get("source_state_sha256")
            or g20_proof.get("source_state_sha256") != g20_proof.get("selected_state_sha256")
        ):
            raise TaskspaceInteractionCostateBridgeError("G22/G20 full-state or receipt custody mismatch")
        state_sha = _require_sha256(rewrite.get("source_state_sha256"), "decoded state SHA-256")

        pair_ids = decode.get("pair_ids")
        checkpoints = decode.get("chunk_checkpoints")
        if (
            pair_ids != list(canonical_ids)
            or decode.get("pair_population_exact_ordered_0_through_599") is not True
            or decode.get("all_chunks_immutable_and_validated") is not True
            or type(checkpoints) is not list
            or not checkpoints
            or decode.get("chunk_count") != len(checkpoints)
        ):
            raise TaskspaceInteractionCostateBridgeError("G22 receipt is bounded or lacks exact n600 pair order")
        custody_rows = [_custody_row(decode.get("run_manifest"), label="G22 run manifest", absolute_path=True)]
        custody_rows.extend(
            _custody_row(row, label=f"G22 chunk checkpoint {index}", absolute_path=True)
            for index, row in enumerate(checkpoints)
        )
        custody_rows.extend(
            (
                _custody_row(g22.get("precleanup_receipt"), label="G22 precleanup receipt", absolute_path=True),
                _custody_row(cleanup.get("certificate"), label="G22 cleanup certificate", absolute_path=True),
                _custody_row(cleanup.get("completion"), label="G22 cleanup completion", absolute_path=True),
            )
        )
        if (
            cleanup.get("completed") is not True
            or cleanup.get("scratch_absent") is not True
            or cleanup.get("per_chunk_checkpoints_preserved") is not True
            or cleanup.get("success_only") is not True
        ):
            raise TaskspaceInteractionCostateBridgeError("G22 final cleanup/receipt custody is incomplete")

        expected_raw_bytes = POPULATION_PAIR_COUNT * 2 * 874 * 1164 * 3
        raw_sha = output.get("source_raw_sha256")
        if (
            output.get("axis") != G22_STRUCTURAL_AXIS
            or output.get("pairs_compared") != POPULATION_PAIR_COUNT
            or output.get("frames_compared") != POPULATION_PAIR_COUNT * 2
            or output.get("raw_bytes_each") != expected_raw_bytes
            or output.get("all_bytes_directly_compared") is not True
            or output.get("uint8_exact_equal") is not True
            or raw_sha != output.get("selected_raw_sha256")
        ):
            raise TaskspaceInteractionCostateBridgeError("G22 full-n600 realized decode equality drifted")
        _require_sha256(raw_sha, "G22 realized output SHA-256")
        expected_authority = {
            "candidate_claim": False,
            "contest_cpu_cuda_same_bytes_owed": True,
            "decode_equality_independent_of_pointer_change": True,
            "decode_equality_invalidated": False,
            "exact_eval_invoked": False,
            "full_n600_receiver_replay_owed": False,
            "promotion_eligible": False,
            "research_only": True,
            "score_claim": False,
        }
        for key, expected in expected_authority.items():
            if authority.get(key) is not expected:
                raise TaskspaceInteractionCostateBridgeError(f"G22 authority field {key} drifted")

        def pointer_parts(name: str) -> tuple[str, int, str]:
            observation = authority.get(name)
            if type(observation) is not dict or observation.get("schema") != "tac.pointer_artifact_observation.v1":
                raise TaskspaceInteractionCostateBridgeError(f"G22 {name} observation drifted")
            artifact = observation.get("artifact")
            target = observation.get("competitive_target")
            if type(artifact) is not dict or type(target) is not dict:
                raise TaskspaceInteractionCostateBridgeError(f"G22 {name} identity split drifted")
            artifact_sha = _require_sha256(artifact.get("sha256"), f"G22 {name} artifact SHA")
            artifact_bytes = artifact.get("bytes")
            preserved = artifact.get("preserved_bytes")
            if (
                type(artifact_bytes) is not int
                or artifact_bytes < 1
                or type(preserved) is not dict
                or preserved.get("sha256") != artifact_sha
                or preserved.get("bytes") != artifact_bytes
            ):
                raise TaskspaceInteractionCostateBridgeError(f"G22 {name} artifact custody drifted")
            return artifact_sha, artifact_bytes, _sha256(_canonical_json(target))

        start_artifact_sha, _start_bytes, start_target_sha = pointer_parts("pointer_start")
        end_artifact_sha, _end_bytes, end_target_sha = pointer_parts("pointer_end")
        if authority.get("pointer_artifact_changed") is not (start_artifact_sha != end_artifact_sha):
            raise TaskspaceInteractionCostateBridgeError("G22 pointer artifact classification drifted")
        target_changed = start_target_sha != end_target_sha
        if (
            authority.get("competitive_target_changed") is not target_changed
            or authority.get("rebase_required_before_admission") is not target_changed
        ):
            raise TaskspaceInteractionCostateBridgeError("G22 semantic admission classification drifted")
        tool = _custody_row(reproducibility.get("tool"), label="G22 replay tool", absolute_path=True)
        argv = reproducibility.get("exact_resume_argv")
        if (
            type(argv) is not list
            or "--confirm-full-n600" not in argv
            or not ("--pair-count" in argv and argv[argv.index("--pair-count") + 1] == "600")
        ):
            raise TaskspaceInteractionCostateBridgeError("G22 full-n600 reviewed invocation custody drifted")

        pair_order_sha = _sha256(_canonical_json(list(canonical_ids)))
        dependency = {
            "schema": "tac.g22_decoded_bytes_equal_dependencies.v1",
            "source_archive": [source["sha256"], source["bytes"]],
            "source_member": [source_member["sha256"], source_member["bytes"]],
            "selected_archive": [selected["sha256"], selected["bytes"]],
            "selected_member": [selected_member["sha256"], selected_member["bytes"]],
            "decoder_runtime": [runtime["sha256"], runtime["bytes"]],
            "pair_population_sha256": population.population_sha256,
            "pair_order_sha256": pair_order_sha,
            "replay_tool_sha256": tool["sha256"],
            "comparison": "direct_all_realized_uint8_bytes.v1",
            "realized_output": [raw_sha, expected_raw_bytes],
        }
        return cls(
            g22_receipt_sha256=_sha256(g22_canonical),
            g22_receipt_nbytes=len(g22_canonical),
            g20_receipt_sha256=g20_sha,
            g20_receipt_nbytes=len(g20_canonical),
            pair_population_sha256=population.population_sha256,
            pair_population_envelope_sha256=_sha256(pair_population),
            source_archive_sha256=source["sha256"],
            source_archive_nbytes=source["bytes"],
            source_member_sha256=source_member["sha256"],
            source_member_nbytes=source_member["bytes"],
            selected_archive_sha256=selected["sha256"],
            selected_archive_nbytes=selected["bytes"],
            selected_member_sha256=selected_member["sha256"],
            selected_member_nbytes=selected_member["bytes"],
            decoder_runtime_sha256=runtime["sha256"],
            decoder_runtime_nbytes=runtime["bytes"],
            decoded_quantized_state_sha256=state_sha,
            realized_output_sha256=raw_sha,
            realized_output_nbytes=expected_raw_bytes,
            pair_order_sha256=pair_order_sha,
            replay_tool_sha256=tool["sha256"],
            decode_dependency_sha256=_sha256(_canonical_json(dependency)),
            pointer_start_artifact_sha256=start_artifact_sha,
            pointer_end_artifact_sha256=end_artifact_sha,
            semantic_target_start_sha256=start_target_sha,
            semantic_target_end_sha256=end_target_sha,
            receipt_custody_sha256=_sha256(_canonical_json(custody_rows)),
            _proof=_G22_CLOSURE_PROOF,
        )

    def admission_status(self, current: ReceiverEqualityAdmissionControlV1 | None = None) -> dict[str, Any]:
        if current is not None and type(current) is not ReceiverEqualityAdmissionControlV1:
            raise TaskspaceInteractionCostateBridgeError("receiver-equality admission control has wrong typed input")
        pointer_sha = self.pointer_end_artifact_sha256 if current is None else current.pointer_artifact_sha256
        target_sha = self.semantic_target_end_sha256 if current is None else current.semantic_target_sha256
        return {
            "pointer_artifact_changed_since_proof": pointer_sha != self.pointer_end_artifact_sha256,
            "semantic_target_changed_since_proof": target_sha != self.semantic_target_end_sha256,
            "rebase_required_before_admission": target_sha != self.semantic_target_end_sha256,
            "decode_equality_invalidated": False,
        }

    def as_dict(self, current: ReceiverEqualityAdmissionControlV1 | None = None) -> dict[str, Any]:
        return {
            "schema": "tac.g22_full_n600_receiver_equality_closure.v1",
            **{
                field_name: getattr(self, field_name)
                for field_name in self.__dataclass_fields__
                if field_name != "_proof"
            },
            "pair_count": POPULATION_PAIR_COUNT,
            "pair_order": "canonical_contiguous_0_through_599",
            "axis": G22_STRUCTURAL_AXIS,
            "decode_proof_dependencies_include_pointer": False,
            "admission_control": self.admission_status(current),
            "score_claim": False,
            "promotion_eligible": False,
        }


@dataclass(frozen=True, slots=True)
class CostateFactorizationV1:
    """Hierarchical identity and the complete closed factor-coordinate product.

    These are labels supplied by the proposing controller, not mechanism claims.
    G19 indexes one indivisible exact effect at atom/group/shard/global scope; it
    never sums the repeated hierarchical views.
    """

    atom_key: str
    group_key: str
    shard_key: str
    global_key: str
    weights: str
    codes: str
    time: str
    population: str
    semantic: str
    realization: str
    pose: str
    entropy: str
    analytic_vs_learned: AnalyticLearnedFactor

    def __post_init__(self) -> None:
        for field_name in (
            "atom_key",
            "group_key",
            "shard_key",
            "global_key",
            *COSTATE_FACTOR_AXES[:-1],
        ):
            _require_identifier(getattr(self, field_name), f"factorization.{field_name}")
        if self.analytic_vs_learned not in ("analytic", "learned", "hybrid"):
            raise TaskspaceInteractionCostateBridgeError(
                "factorization.analytic_vs_learned left the closed factor axis"
            )


@dataclass(frozen=True, slots=True)
class CurrentExactReceiverReplayV1:
    """Current-context replay needed before a historical lattice becomes usable.

    V1 deliberately binds the replay to the same two G18 source-pair IDs.  It is
    therefore exact receiver evidence for homotopy telemetry, but remains n2
    macOS advisory and cannot enter an n600 posterior or promotion store.
    """

    replay_id: str
    measurement_receipt_sha256: str
    receiver_receipt_sha256: str
    decoded_output_sha256: str
    archive_sha256: str
    archive_nbytes: int
    d_seg: float
    d_pose: float
    baseline_bundle_sha256: str
    population_manifest_sha256: str
    frozen_scorer_sha256: str
    pointer_latest_sha256: str
    axis: str
    source_pair_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_identifier(self.replay_id, "teacher_replay.replay_id")
        for field_name in (
            "measurement_receipt_sha256",
            "receiver_receipt_sha256",
            "decoded_output_sha256",
            "archive_sha256",
            "baseline_bundle_sha256",
            "population_manifest_sha256",
            "frozen_scorer_sha256",
            "pointer_latest_sha256",
        ):
            _require_sha256(getattr(self, field_name), f"teacher_replay.{field_name}")
        if type(self.archive_nbytes) is not int or self.archive_nbytes < 1:
            raise TaskspaceInteractionCostateBridgeError("teacher_replay.archive_nbytes must be positive int")
        for field_name in ("d_seg", "d_pose"):
            value = _finite(getattr(self, field_name), f"teacher_replay.{field_name}")
            if value < 0.0:
                raise TaskspaceInteractionCostateBridgeError(f"teacher_replay.{field_name} left the physical domain")
        if type(self.axis) is not str or not self.axis:
            raise TaskspaceInteractionCostateBridgeError("teacher_replay.axis must be non-empty")
        if (
            type(self.source_pair_ids) is not tuple
            or not self.source_pair_ids
            or any(type(pair_id) is not int or pair_id < 0 for pair_id in self.source_pair_ids)
            or len(set(self.source_pair_ids)) != len(self.source_pair_ids)
        ):
            raise TaskspaceInteractionCostateBridgeError(
                "teacher_replay.source_pair_ids must be unique nonnegative indices"
            )


@dataclass(frozen=True, slots=True)
class FullLatticeTeacherEndpointV1:
    """A distortion-zero/high-rate lattice teacher, historical or replayed.

    The reported endpoint is retained for lineage.  Only ``current_replay`` can
    make it a numeric homotopy reference, and build-time validation then binds
    every current receiver/scorer/base/population field independently.
    """

    teacher_key: str
    historical_receipt_sha256: str
    lattice_payload_sha256: str
    lattice_payload_nbytes: int
    reported_d_seg: float
    reported_d_pose: float
    current_replay: CurrentExactReceiverReplayV1 | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.teacher_key, "teacher_key")
        _require_sha256(self.historical_receipt_sha256, "historical_receipt_sha256")
        _require_sha256(self.lattice_payload_sha256, "lattice_payload_sha256")
        if type(self.lattice_payload_nbytes) is not int or self.lattice_payload_nbytes < 1:
            raise TaskspaceInteractionCostateBridgeError("lattice_payload_nbytes must be positive int")
        for field_name in ("reported_d_seg", "reported_d_pose"):
            if _finite(getattr(self, field_name), field_name) != 0.0:
                raise TaskspaceInteractionCostateBridgeError(
                    "full-lattice teacher must report the distortion-zero endpoint"
                )
        if self.current_replay is not None and type(self.current_replay) is not CurrentExactReceiverReplayV1:
            raise TaskspaceInteractionCostateBridgeError(
                "current_replay must have exact CurrentExactReceiverReplayV1 type"
            )


@dataclass(frozen=True, slots=True)
class DecoderPlacementV1:
    """Declared generic-decoder versus counted-payload placement for one action.

    Decoder source bytes and runtime remain visible guardrail costs, while only
    the selected archive's post-compression bytes enter the contest score.  The
    evidence hashes are retained but not dereferenced by this pure adapter.
    """

    generic_decoder_program_sha256: str
    generic_decoder_source_nbytes: int
    generic_operations: tuple[str, ...]
    video_specific_payload_fields: tuple[str, ...]
    decoder_runtime_seconds: float
    decoder_runtime_receipt_sha256: str
    deterministic_portability_receipt_sha256: str
    placement_receipt_sha256: str
    factorability_receipt_sha256: str | None = None

    def __post_init__(self) -> None:
        for field_name in (
            "generic_decoder_program_sha256",
            "decoder_runtime_receipt_sha256",
            "deterministic_portability_receipt_sha256",
            "placement_receipt_sha256",
        ):
            _require_sha256(getattr(self, field_name), f"decoder_placement.{field_name}")
        if self.factorability_receipt_sha256 is not None:
            _require_sha256(
                self.factorability_receipt_sha256,
                "decoder_placement.factorability_receipt_sha256",
            )
        if type(self.generic_decoder_source_nbytes) is not int or self.generic_decoder_source_nbytes < 1:
            raise TaskspaceInteractionCostateBridgeError(
                "decoder_placement.generic_decoder_source_nbytes must be positive int"
            )
        if (
            type(self.generic_operations) is not tuple
            or not self.generic_operations
            or len(set(self.generic_operations)) != len(self.generic_operations)
            or any(operation not in GENERIC_DECODER_OPERATIONS for operation in self.generic_operations)
        ):
            raise TaskspaceInteractionCostateBridgeError(
                "decoder_placement.generic_operations left the closed active-decode vocabulary"
            )
        if (
            type(self.video_specific_payload_fields) is not tuple
            or not self.video_specific_payload_fields
            or len(set(self.video_specific_payload_fields)) != len(self.video_specific_payload_fields)
        ):
            raise TaskspaceInteractionCostateBridgeError(
                "decoder_placement requires unique counted video-specific payload fields"
            )
        for field_name in self.video_specific_payload_fields:
            _require_identifier(field_name, "decoder_placement.video_specific_payload_field")
        runtime = _finite(self.decoder_runtime_seconds, "decoder_placement.decoder_runtime_seconds")
        if runtime < 0.0:
            raise TaskspaceInteractionCostateBridgeError(
                "decoder_placement.decoder_runtime_seconds must be nonnegative"
            )


@dataclass(frozen=True, slots=True)
class ActionEffectPredictionV1:
    """One prediction bound to one measured effect and its exact base object.

    For ``transition``, the component fields are endpoint deltas and the score
    effect is recomputed from the nonlinear score law.  For ``interaction``, the
    component fields are four-cell residuals and ``predicted_joint_score_effect``
    is mandatory: a nonlinear score interaction cannot be reconstructed by adding
    component residuals or multiplying them by local partial derivatives.
    """

    proposal_key: str
    effect_id: str
    effect_kind: EffectKind
    exact_base_measurement_id: str
    exact_base_archive_sha256: str
    baseline_bundle_sha256: str
    population_manifest_sha256: str
    action_semantics: ActionSemantics
    source_reservoirs: tuple[str, ...]
    target_reservoirs: tuple[str, ...]
    factorization: CostateFactorizationV1
    decoder_placement: DecoderPlacementV1
    homotopy_path_key: str
    homotopy_step_index: int
    predicted_delta_d_seg: float
    predicted_delta_d_pose: float
    predicted_delta_archive_bytes: int
    predicted_postcompression_archive_nbytes: int | None
    predicted_joint_score_effect: float | None = None
    semantic_evidence_sha256: str | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.proposal_key, "proposal_key")
        _require_identifier(self.effect_id, "effect_id")
        _require_identifier(self.exact_base_measurement_id, "exact_base_measurement_id")
        _require_identifier(self.homotopy_path_key, "homotopy_path_key")
        for field_name in (
            "exact_base_archive_sha256",
            "baseline_bundle_sha256",
            "population_manifest_sha256",
        ):
            _require_sha256(getattr(self, field_name), field_name)
        if self.effect_kind not in ("transition", "interaction"):
            raise TaskspaceInteractionCostateBridgeError("effect_kind must be transition or interaction")
        _validate_action_signature(
            self.action_semantics,
            self.source_reservoirs,
            self.target_reservoirs,
            self.semantic_evidence_sha256,
            collection_type=tuple,
        )
        if type(self.factorization) is not CostateFactorizationV1:
            raise TaskspaceInteractionCostateBridgeError("factorization must have exact CostateFactorizationV1 type")
        if type(self.decoder_placement) is not DecoderPlacementV1:
            raise TaskspaceInteractionCostateBridgeError("decoder_placement must have exact DecoderPlacementV1 type")
        if type(self.homotopy_step_index) is not int or self.homotopy_step_index < 0:
            raise TaskspaceInteractionCostateBridgeError("homotopy_step_index must be a nonnegative int")
        _finite(self.predicted_delta_d_seg, "predicted_delta_d_seg")
        _finite(self.predicted_delta_d_pose, "predicted_delta_d_pose")
        if type(self.predicted_delta_archive_bytes) is not int:
            raise TaskspaceInteractionCostateBridgeError("predicted_delta_archive_bytes must be int")
        if self.effect_kind == "transition":
            if (
                type(self.predicted_postcompression_archive_nbytes) is not int
                or self.predicted_postcompression_archive_nbytes < 1
            ):
                raise TaskspaceInteractionCostateBridgeError(
                    "transition requires positive predicted post-compression archive bytes"
                )
        elif self.predicted_postcompression_archive_nbytes is not None:
            raise TaskspaceInteractionCostateBridgeError("interaction has no endpoint post-compression archive size")
        if self.effect_kind == "transition" and self.predicted_joint_score_effect is not None:
            raise TaskspaceInteractionCostateBridgeError(
                "transition score effect is recomputed; caller-supplied score effect is forbidden"
            )
        if self.effect_kind == "interaction" and self.predicted_joint_score_effect is None:
            raise TaskspaceInteractionCostateBridgeError(
                "interaction requires an independently predicted joint score residual"
            )
        if self.predicted_joint_score_effect is not None:
            _finite(self.predicted_joint_score_effect, "predicted_joint_score_effect")


def _prediction_dict(prediction: ActionEffectPredictionV1) -> dict[str, Any]:
    result = asdict(prediction)
    result["source_reservoirs"] = list(prediction.source_reservoirs)
    result["target_reservoirs"] = list(prediction.target_reservoirs)
    result["factorization"] = _factorization_dict(prediction.factorization)
    result["decoder_placement"] = _placement_dict(prediction.decoder_placement)
    return result


def _factorization_dict(factorization: CostateFactorizationV1) -> dict[str, Any]:
    return asdict(factorization)


def _placement_dict(placement: DecoderPlacementV1) -> dict[str, Any]:
    result = asdict(placement)
    result["generic_operations"] = list(placement.generic_operations)
    result["video_specific_payload_fields"] = list(placement.video_specific_payload_fields)
    return result


def _factorization_from_mapping(value: object) -> CostateFactorizationV1:
    expected = {
        "atom_key",
        "group_key",
        "shard_key",
        "global_key",
        *COSTATE_FACTOR_AXES,
    }
    if type(value) is not dict or set(value) != expected:
        raise TaskspaceInteractionCostateBridgeError("prior factorization schema drifted")
    return CostateFactorizationV1(**value)


def _placement_from_mapping(value: object) -> DecoderPlacementV1:
    expected = {
        "generic_decoder_program_sha256",
        "generic_decoder_source_nbytes",
        "generic_operations",
        "video_specific_payload_fields",
        "decoder_runtime_seconds",
        "decoder_runtime_receipt_sha256",
        "deterministic_portability_receipt_sha256",
        "placement_receipt_sha256",
        "factorability_receipt_sha256",
    }
    if type(value) is not dict or set(value) != expected:
        raise TaskspaceInteractionCostateBridgeError("prior decoder placement schema drifted")
    if type(value["generic_operations"]) is not list or type(value["video_specific_payload_fields"]) is not list:
        raise TaskspaceInteractionCostateBridgeError("prior decoder placement collections drifted")
    return DecoderPlacementV1(
        **{
            **value,
            "generic_operations": tuple(value["generic_operations"]),
            "video_specific_payload_fields": tuple(value["video_specific_payload_fields"]),
        }
    )


def _binding_dict(binding: FeedbackExactBaseBindingV1) -> dict[str, Any]:
    result = asdict(binding)
    result["source_pair_ids"] = list(binding.source_pair_ids)
    return result


def bridge_receipt_bytes(value: Mapping[str, Any]) -> bytes:
    """Return canonical bytes for a completed v1 bridge record."""

    if type(value) is not dict or value.get("schema") != SCHEMA:
        raise TaskspaceInteractionCostateBridgeError("bridge record schema changed")
    return _canonical_json(value) + b"\n"


def _load_feedback(feedback: bytes | Mapping[str, Any]) -> tuple[bytes, dict[str, Any]]:
    if type(feedback) is bytes:
        source = feedback
        try:
            value = json.loads(source)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise TaskspaceInteractionCostateBridgeError("G18 feedback bytes are not JSON") from exc
        if type(value) is not dict:
            raise TaskspaceInteractionCostateBridgeError("G18 feedback root must be one object")
        try:
            canonical = feedback_receipt_bytes(value)
        except Exception as exc:
            raise TaskspaceInteractionCostateBridgeError("G18 feedback schema/canonicalization failed") from exc
        if source != canonical:
            raise TaskspaceInteractionCostateBridgeError("G18 feedback bytes are not the canonical receipt bytes")
    elif type(feedback) is dict:
        value = feedback
        try:
            canonical = feedback_receipt_bytes(value)
        except Exception as exc:
            raise TaskspaceInteractionCostateBridgeError("G18 feedback schema/canonicalization failed") from exc
    else:
        raise TaskspaceInteractionCostateBridgeError("feedback must be exact bytes or a plain dict")
    return canonical, value


def _measurement_score(row: Mapping[str, Any]) -> float:
    return _score(
        _finite(row["d_seg"], "measurement.d_seg"),
        _finite(row["d_pose"], "measurement.d_pose"),
        row["selected_archive_bytes"],
    )


def _conditional_sign_reversal(
    action_effect_without_context: float,
    action_effect_with_context: float,
) -> str | None:
    without_context_improves = action_effect_without_context < 0.0
    with_context_improves = action_effect_with_context < 0.0
    if without_context_improves == with_context_improves:
        return None
    if with_context_improves:
        return "LOCALLY_HARMFUL_BUT_COMPOSITIONALLY_BENEFICIAL"
    return "LOCALLY_BENEFICIAL_BUT_COMPOSITIONALLY_HARMFUL"


def _validate_measurement(row: object, *, sample_count: int) -> dict[str, Any]:
    if type(row) is not dict or not set(row) >= _MEASUREMENT_REQUIRED:
        raise TaskspaceInteractionCostateBridgeError("G18 measurement schema is incomplete")
    _require_identifier(row["measurement_id"], "measurement_id")
    for field_name in (
        "baseline_bundle_sha256",
        "bundle_sha256",
        "selected_archive_sha256",
        "stored_archive_sha256",
        "deflated_archive_sha256",
        "member_sha256",
        "decoded_output_sha256",
        "receiver_receipt_sha256",
        "a_program_sha256",
        "a_source_binding_sha256",
        "camera_y1_sha256",
        "candidate_seg_labels_sha256",
    ):
        _require_sha256(row[field_name], f"measurement.{field_name}")
    for field_name in (
        "selected_archive_bytes",
        "stored_archive_bytes",
        "deflated_archive_bytes",
        "member_bytes",
    ):
        if type(row[field_name]) is not int or row[field_name] < 1:
            raise TaskspaceInteractionCostateBridgeError(f"measurement.{field_name} must be positive int")
    d_seg = _finite(row["d_seg"], "measurement.d_seg")
    d_pose = _finite(row["d_pose"], "measurement.d_pose")
    if d_seg < 0.0 or d_pose < 0.0:
        raise TaskspaceInteractionCostateBridgeError("measurement distances must be nonnegative")
    if row["selected_encoding"] == "STORE":
        selected = (row["stored_archive_sha256"], row["stored_archive_bytes"])
    elif row["selected_encoding"] == "DEFLATE":
        selected = (row["deflated_archive_sha256"], row["deflated_archive_bytes"])
    else:
        raise TaskspaceInteractionCostateBridgeError("measurement selected encoding changed")
    if selected != (row["selected_archive_sha256"], row["selected_archive_bytes"]):
        raise TaskspaceInteractionCostateBridgeError("selected archive hash/bytes drifted from encoding")
    _close(row["derived_component_total"], _measurement_score(row), "derived_component_total")
    evidence = row["scorer_evidence"]
    if type(evidence) is not dict or evidence.get("sample_count") != sample_count:
        raise TaskspaceInteractionCostateBridgeError("scorer evidence population count drifted")
    for field_name in (
        "measurement_receipt_sha256",
        "candidate_forward_receipt_sha256",
        "candidate_pose6_sha256",
        "frozen_scorer_sha256",
        "target_forward_receipt_sha256",
    ):
        _require_sha256(evidence.get(field_name), f"scorer_evidence.{field_name}")
    if evidence.get("dense_frames_logits_rgb_serialized") is not False:
        raise TaskspaceInteractionCostateBridgeError("dense scorer evidence serialization changed")
    for field_name in ("per_pair_d_seg", "per_pair_d_pose"):
        values = evidence.get(field_name)
        if type(values) is not list or len(values) != sample_count:
            raise TaskspaceInteractionCostateBridgeError(f"{field_name} population scope drifted")
        for index, value in enumerate(values):
            if _finite(value, f"{field_name}[{index}]") < 0.0:
                raise TaskspaceInteractionCostateBridgeError(f"{field_name} contains a negative distance")
    return row


def _validate_transition(
    row: object,
    *,
    index: int,
    occurrence_by_id: Mapping[str, dict[str, Any]],
    measurement_by_id: Mapping[str, dict[str, Any]],
) -> dict[str, Any]:
    if type(row) is not dict or not set(row) >= _TRANSITION_FIELDS:
        raise TaskspaceInteractionCostateBridgeError("transition schema is incomplete")
    if row["transition_id"] != f"transition:{index}":
        raise TaskspaceInteractionCostateBridgeError("transition IDs are not canonical and ordered")
    before = measurement_by_id.get(row["before_measurement_id"])
    occurrence = occurrence_by_id.get(row["after_occurrence_id"])
    if before is None or occurrence is None:
        raise TaskspaceInteractionCostateBridgeError("transition lost its before/after row")
    after = occurrence["measurement"]
    if row["after_measurement_id"] != after["measurement_id"]:
        raise TaskspaceInteractionCostateBridgeError("transition after measurement identity drifted")
    if before["baseline_bundle_sha256"] != after["baseline_bundle_sha256"]:
        raise TaskspaceInteractionCostateBridgeError("transition crosses exact baseline bundles")
    delta_seg = after["d_seg"] - before["d_seg"]
    delta_pose = after["d_pose"] - before["d_pose"]
    delta_bytes = after["selected_archive_bytes"] - before["selected_archive_bytes"]
    seg_term = 100.0 * delta_seg
    pose_term = math.sqrt(10.0 * after["d_pose"]) - math.sqrt(10.0 * before["d_pose"])
    rate_term = 25.0 * delta_bytes / RATE_DENOMINATOR
    distortion_term = seg_term + pose_term
    exact_delta = distortion_term + rate_term
    for field_name, expected in (
        ("delta_d_seg", delta_seg),
        ("delta_d_pose", delta_pose),
        ("seg_term_delta", seg_term),
        ("pose_term_delta", pose_term),
        ("rate_term_delta", rate_term),
        ("distortion_term_delta", distortion_term),
        ("exact_score_delta", exact_delta),
        ("exact_score_improvement", -exact_delta),
    ):
        _close(row[field_name], expected, f"transition.{field_name}")
    if row["delta_selected_archive_bytes"] != delta_bytes:
        raise TaskspaceInteractionCostateBridgeError("transition byte delta drifted")
    ceiling = -distortion_term * RATE_DENOMINATOR / 25.0
    _close(row["finite_byte_ceiling_real"], ceiling, "transition.finite_byte_ceiling_real", atol=1e-9)
    if row["greatest_strict_integer_byte_delta"] != math.ceil(ceiling) - 1:
        raise TaskspaceInteractionCostateBridgeError("transition strict integer byte ceiling drifted")
    if row["improves_score"] is not (exact_delta < 0.0):
        raise TaskspaceInteractionCostateBridgeError("transition improvement flag drifted")
    return {
        "effect_id": row["transition_id"],
        "effect_kind": "transition",
        "component_effect_semantics": "exact_endpoint_delta_on_one_measured_before_after_archive_pair.v1",
        "exact_base_measurement_id": before["measurement_id"],
        "exact_base_archive_sha256": before["selected_archive_sha256"],
        "exact_base_archive_nbytes": before["selected_archive_bytes"],
        "baseline_bundle_sha256": before["baseline_bundle_sha256"],
        "realized_archive_custody": {
            "before_archive_sha256": before["selected_archive_sha256"],
            "before_archive_nbytes": before["selected_archive_bytes"],
            "after_archive_sha256": after["selected_archive_sha256"],
            "after_archive_nbytes": after["selected_archive_bytes"],
        },
        "homotopy_endpoint": {
            "before": {
                "measurement_id": before["measurement_id"],
                "archive_sha256": before["selected_archive_sha256"],
                "postcompression_mdl_nbytes": before["selected_archive_bytes"],
                "d_seg": before["d_seg"],
                "d_pose": before["d_pose"],
                "global_score": _measurement_score(before),
            },
            "after": {
                "measurement_id": after["measurement_id"],
                "archive_sha256": after["selected_archive_sha256"],
                "postcompression_mdl_nbytes": after["selected_archive_bytes"],
                "d_seg": after["d_seg"],
                "d_pose": after["d_pose"],
                "global_score": _measurement_score(after),
            },
        },
        "realized": {
            "delta_d_seg": delta_seg,
            "delta_d_pose": delta_pose,
            "delta_archive_bytes": delta_bytes,
            "joint_score_effect": exact_delta,
        },
    }


def _interaction_effect(
    row: object,
    *,
    index: int,
    baseline: dict[str, Any],
    occurrences: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if type(row) is not dict or row.get("interaction_id") != f"interaction:{index}":
        raise TaskspaceInteractionCostateBridgeError("interaction identity/order drifted")
    required = {
        "interaction_id",
        "treatment_id",
        "g8_proposal_id",
        "a_program_id",
        "a_program_sha256",
        "family",
        "prefix_order",
        "palette_bound_per_class",
        "prefix_cell_count",
        "interaction_I",
        "definition",
    }
    if not required <= set(row) or row["definition"] != ("S_g8_a_minus_S_g8_pass_minus_S_g0_a_plus_S_g0_pass.v1"):
        raise TaskspaceInteractionCostateBridgeError("interaction schema/definition drifted")
    treatment_id = row["treatment_id"]
    g8_a = [item for item in occurrences if item.get("kind") == "g8_a" and item.get("treatment_id") == treatment_id]
    g0_a = [
        item for item in occurrences if item.get("kind") == "matched_g0_a" and item.get("treatment_id") == treatment_id
    ]
    g8_pass = [
        item
        for item in occurrences
        if item.get("kind") == "g8_pass_a"
        and type(item.get("g8_branch")) is dict
        and item["g8_branch"].get("proposal_id") == row["g8_proposal_id"]
    ]
    if len(g8_a) != 1 or len(g0_a) != 1 or len(g8_pass) != 1:
        raise TaskspaceInteractionCostateBridgeError("interaction four-cell support is incomplete")
    cells = {
        "g0_pass": baseline,
        "g8_pass": g8_pass[0]["measurement"],
        "matched_g0_a": g0_a[0]["measurement"],
        "g8_a": g8_a[0]["measurement"],
    }
    if len({cell["baseline_bundle_sha256"] for cell in cells.values()}) != 1:
        raise TaskspaceInteractionCostateBridgeError("interaction four cells cross baseline bundles")

    def four_cell(field_name: str) -> float:
        return (
            float(cells["g8_a"][field_name])
            - float(cells["g8_pass"][field_name])
            - float(cells["matched_g0_a"][field_name])
            + float(cells["g0_pass"][field_name])
        )

    component = {
        "delta_d_seg": four_cell("d_seg"),
        "delta_d_pose": four_cell("d_pose"),
        "delta_archive_bytes": int(four_cell("selected_archive_bytes")),
    }
    joint_score = (
        _measurement_score(cells["g8_a"])
        - _measurement_score(cells["g8_pass"])
        - _measurement_score(cells["matched_g0_a"])
        + _measurement_score(cells["g0_pass"])
    )
    action_effect_without_g8 = _measurement_score(cells["matched_g0_a"]) - _measurement_score(
        cells["g0_pass"]
    )
    action_effect_with_g8 = _measurement_score(cells["g8_a"]) - _measurement_score(cells["g8_pass"])
    sign_reversal = _conditional_sign_reversal(
        action_effect_without_g8,
        action_effect_with_g8,
    )
    _close(row["interaction_I"], joint_score, "interaction.interaction_I")
    return {
        "effect_id": row["interaction_id"],
        "effect_kind": "interaction",
        "component_effect_semantics": "four_cell_g8_by_a_residual_not_an_atomic_delta.v1",
        "exact_base_measurement_id": baseline["measurement_id"],
        "exact_base_archive_sha256": baseline["selected_archive_sha256"],
        "exact_base_archive_nbytes": baseline["selected_archive_bytes"],
        "baseline_bundle_sha256": baseline["baseline_bundle_sha256"],
        "realized_archive_custody": {
            name: {
                "archive_sha256": cell["selected_archive_sha256"],
                "archive_nbytes": cell["selected_archive_bytes"],
                "measurement_id": cell["measurement_id"],
            }
            for name, cell in cells.items()
        },
        "realized": {**component, "joint_score_effect": joint_score},
        "conditional_action_effect": {
            "action_effect_without_g8": action_effect_without_g8,
            "action_effect_with_g8": action_effect_with_g8,
            "interaction_residual": joint_score,
            "sign_reversal": sign_reversal,
            "local_marginal_pruning_safe": sign_reversal is None,
        },
        "treatment_id": treatment_id,
        "g8_proposal_id": row["g8_proposal_id"],
        "family": row["family"],
        "prefix_order": row["prefix_order"],
        "palette_bound_per_class": row["palette_bound_per_class"],
        "prefix_cell_count": row["prefix_cell_count"],
    }


def _validate_feedback(value: dict[str, Any], *, sample_count: int) -> dict[str, Any]:
    if set(value) != _TOP_FIELDS or value.get("schema") != G18_SCHEMA:
        raise TaskspaceInteractionCostateBridgeError("frozen G18 top-level schema drifted")
    if value.get("source_schema") != SOURCE_SCHEMA or value.get("axis") != G18_AXIS:
        raise TaskspaceInteractionCostateBridgeError("G18 source schema or axis drifted")
    if value.get("truth") != _TRUTH:
        raise TaskspaceInteractionCostateBridgeError("G18 authority/originality truth drifted")
    if (
        value.get("dense_frames_or_masks_serialized") is not False
        or value.get("live_run_scorer_eval_or_ledger_access_performed") is not False
    ):
        raise TaskspaceInteractionCostateBridgeError("G18 containment truth drifted")
    for field_name in ("source_receipt_sha256", "canonical_receipt_sha256"):
        _require_sha256(value[field_name], field_name)
    _require_identifier(value["source_lane_id"], "source_lane_id")

    comparison = value["latest_pointer_comparison"]
    if comparison != {
        "classification": "noncomparable_n2_omitted_runtime_research_lower_bound",
        "comparable": False,
        "pairwise_common_basis_deltas_valid": True,
    }:
        raise TaskspaceInteractionCostateBridgeError("G18 pointer-comparison scope drifted")
    custody = value["pointer_observation_custody"]
    if type(custody) is not dict or set(custody) != {
        "pointer_start",
        "pointer_latest",
        "manifest_sha256",
        "pointer_observation_paths",
        "pointer_mutation_performed",
    }:
        raise TaskspaceInteractionCostateBridgeError("pointer observation custody schema drifted")
    _require_sha256(custody["manifest_sha256"], "pointer manifest")
    for name in ("pointer_start", "pointer_latest"):
        pointer = custody[name]
        if type(pointer) is not dict or not {"pointer_sha256", "target_score"}.issubset(pointer):
            raise TaskspaceInteractionCostateBridgeError(f"{name} schema drifted")
        _require_sha256(pointer["pointer_sha256"], f"{name}.pointer_sha256")
        _finite(pointer["target_score"], f"{name}.target_score")
        if "pointer_bytes" in pointer and (
            type(pointer["pointer_bytes"]) is not int or pointer["pointer_bytes"] < 1
        ):
            raise TaskspaceInteractionCostateBridgeError(f"{name}.pointer_bytes drifted")
        if "pointer_path" in pointer and (
            type(pointer["pointer_path"]) is not str or not pointer["pointer_path"].startswith("/")
        ):
            raise TaskspaceInteractionCostateBridgeError(f"{name}.pointer_path drifted")
        for field_name in ("pointer_device", "pointer_inode", "pointer_mtime_ns"):
            if field_name in pointer and type(pointer[field_name]) is not int:
                raise TaskspaceInteractionCostateBridgeError(f"{name}.{field_name} drifted")
        for field_name, expected in (
            ("derived_planning_only", True),
            ("evaluation_claim", False),
            ("pointer_moved", False),
            ("promotion_eligible", False),
            ("ready_for_exact_eval_dispatch", False),
            ("research_only", True),
            ("score_claim", False),
        ):
            if field_name in pointer and pointer[field_name] is not expected:
                raise TaskspaceInteractionCostateBridgeError(f"{name}.{field_name} authority drifted")
        if "selected_archive_sha256" in pointer and pointer["selected_archive_sha256"] is not None:
            _require_sha256(pointer["selected_archive_sha256"], f"{name}.selected_archive_sha256")
        for field_name in (
            "last_refreshed_utc",
            "selected_axis",
            "selected_custody",
            "selected_evidence_grade",
            "selected_score_precision",
            "selected_source",
            "selected_source_kind",
            "selection_rule",
            "source_snapshot_at_utc",
        ):
            if field_name in pointer and (type(pointer[field_name]) is not str or not pointer[field_name]):
                raise TaskspaceInteractionCostateBridgeError(f"{name}.{field_name} drifted")
    paths = custody["pointer_observation_paths"]
    if (
        custody["pointer_mutation_performed"] is not False
        or type(paths) is not list
        or not paths
        or any(type(path) is not str or not path for path in paths)
    ):
        raise TaskspaceInteractionCostateBridgeError("pointer observation mutation/path custody drifted")

    partition = value["four_way_z_t_h_partition"]
    source_pair_ids = partition.get("source_pair_ids") if type(partition) is dict else None
    if type(source_pair_ids) is not list or len(source_pair_ids) != sample_count:
        raise TaskspaceInteractionCostateBridgeError("four-way population scope drifted")

    inventory = value["row_inventory"]
    if type(inventory) is not dict or set(inventory) != {
        "occurrence_count",
        "unique_measurement_id_count",
        "occurrences",
        "selected_research_row",
    }:
        raise TaskspaceInteractionCostateBridgeError("row inventory schema drifted")
    occurrences = inventory["occurrences"]
    if type(occurrences) is not list or inventory["occurrence_count"] != len(occurrences):
        raise TaskspaceInteractionCostateBridgeError("row occurrence count drifted")
    occurrence_by_id: dict[str, dict[str, Any]] = {}
    measurement_by_id: dict[str, dict[str, Any]] = {}
    measurement_semantic_by_id: dict[str, dict[str, Any]] = {}
    measurement_proof_receipts_by_id: dict[str, set[str]] = {}
    measurement_occurrence_count_by_id: dict[str, int] = {}
    for occurrence in occurrences:
        if type(occurrence) is not dict:
            raise TaskspaceInteractionCostateBridgeError("row occurrence is not an object")
        occurrence_id = _require_identifier(occurrence.get("occurrence_id"), "occurrence_id")
        if occurrence_id in occurrence_by_id:
            raise TaskspaceInteractionCostateBridgeError("row occurrence ID repeats")
        measurement = _validate_measurement(occurrence.get("measurement"), sample_count=sample_count)
        measurement_id = measurement["measurement_id"]
        semantic_measurement = dict(measurement)
        semantic_scorer = dict(measurement["scorer_evidence"])
        proof_receipt_sha = semantic_scorer.pop("measurement_receipt_sha256")
        semantic_measurement["scorer_evidence"] = semantic_scorer
        previous_semantic = measurement_semantic_by_id.get(measurement_id)
        if previous_semantic is not None and previous_semantic != semantic_measurement:
            raise TaskspaceInteractionCostateBridgeError(
                "measurement semantic-control ID aliases different realized objects"
            )
        measurement_semantic_by_id[measurement_id] = semantic_measurement
        measurement_proof_receipts_by_id.setdefault(measurement_id, set()).add(proof_receipt_sha)
        measurement_occurrence_count_by_id[measurement_id] = (
            measurement_occurrence_count_by_id.get(measurement_id, 0) + 1
        )
        previous = measurement_by_id.get(measurement_id)
        if (
            previous is None
            or proof_receipt_sha < previous["scorer_evidence"]["measurement_receipt_sha256"]
        ):
            measurement_by_id[measurement_id] = measurement
        occurrence_by_id[occurrence_id] = occurrence
    if inventory["unique_measurement_id_count"] != len(measurement_by_id):
        raise TaskspaceInteractionCostateBridgeError("unique measurement count drifted")
    baseline_occurrences = [item for item in occurrences if item.get("occurrence_id") == "g0:0"]
    if len(baseline_occurrences) != 1:
        raise TaskspaceInteractionCostateBridgeError("canonical G0-pass baseline occurrence is missing")
    baseline = baseline_occurrences[0]["measurement"]
    if baseline["a_mode"] != "PASS_A_V1" or baseline.get("g8_program_sha256") is not None:
        raise TaskspaceInteractionCostateBridgeError("canonical baseline is not G0/pass-A")
    if len({item["measurement"]["baseline_bundle_sha256"] for item in occurrences}) != 1:
        raise TaskspaceInteractionCostateBridgeError("G18 rows cross baseline bundles")
    scorer_shas = {item["measurement"]["scorer_evidence"]["frozen_scorer_sha256"] for item in occurrences}
    target_shas = {item["measurement"]["scorer_evidence"]["target_forward_receipt_sha256"] for item in occurrences}
    if len(scorer_shas) != 1 or len(target_shas) != 1:
        raise TaskspaceInteractionCostateBridgeError("scorer/target-forward custody changes across rows")
    selected = inventory["selected_research_row"]
    if type(selected) is not dict:
        raise TaskspaceInteractionCostateBridgeError("selected research row is absent")
    expected_selected = min(
        (item["measurement"] for item in occurrences),
        key=lambda row: (row["derived_component_total"], row["measurement_id"]),
    )

    def without_path(row: Mapping[str, Any]) -> dict[str, Any]:
        return {key: item for key, item in row.items() if key != "selected_archive_path"}

    if without_path(selected) != without_path(expected_selected):
        raise TaskspaceInteractionCostateBridgeError("selected research row is not the complete-row minimum")

    transition_inventory = value["transition_inventory"]
    transitions = transition_inventory.get("transitions") if type(transition_inventory) is dict else None
    if type(transitions) is not list or transition_inventory.get("transition_count") != len(transitions):
        raise TaskspaceInteractionCostateBridgeError("transition inventory count drifted")
    transition_effects = [
        _validate_transition(
            row,
            index=index,
            occurrence_by_id=occurrence_by_id,
            measurement_by_id=measurement_by_id,
        )
        for index, row in enumerate(transitions)
    ]
    interaction_inventory = value["interaction_inventory"]
    interactions = interaction_inventory.get("interactions") if type(interaction_inventory) is dict else None
    if type(interactions) is not list or interaction_inventory.get("interaction_count") != len(interactions):
        raise TaskspaceInteractionCostateBridgeError("interaction inventory count drifted")
    interaction_effects = [
        _interaction_effect(row, index=index, baseline=baseline, occurrences=occurrences)
        for index, row in enumerate(interactions)
    ]

    transition_ids = {effect["effect_id"] for effect in transition_effects}
    interaction_ids = {effect["effect_id"] for effect in interaction_effects}
    bit_allocator = value["downstream_payloads"].get("bit_allocator")
    if (
        type(bit_allocator) is not dict
        or bit_allocator.get("additive_independence_assumed") is not False
        or bit_allocator.get("dynamic_archive_cost_preserved") is not True
        or bit_allocator.get("mutually_exclusive_or_composed_actions_require_bundle_remeasurement") is not True
        or set(bit_allocator.get("transition_ids") or ()) != transition_ids
        or set(bit_allocator.get("interaction_ids") or ()) != interaction_ids
    ):
        raise TaskspaceInteractionCostateBridgeError("G18 nonadditive bit-allocation contract drifted")
    hooks = value["hook_coverage"]
    if type(hooks) is not dict or set(hooks) != {"1", "2", "3", "4", "5", "6"}:
        raise TaskspaceInteractionCostateBridgeError("G18 six-hook closure drifted")
    for hook_number in ("3", "4", "6"):
        if set(hooks[hook_number].get("transition_ids") or ()) != transition_ids:
            raise TaskspaceInteractionCostateBridgeError("transition signal orphaned from G18 hook")
    for hook_number in ("3", "4", "5", "6"):
        if set(hooks[hook_number].get("interaction_ids") or ()) != interaction_ids:
            raise TaskspaceInteractionCostateBridgeError("interaction signal orphaned from G18 hook")
    return {
        "baseline": baseline,
        "selected": selected,
        "source_pair_ids": tuple(source_pair_ids),
        "frozen_scorer_sha256": next(iter(scorer_shas)),
        "transition_effects": transition_effects,
        "interaction_effects": interaction_effects,
        "pointer_custody": custody,
        "measurement_identity_domains": [
            {
                "measurement_id": measurement_id,
                "semantic_control_identity_sha256": _sha256(
                    _canonical_json(measurement_semantic_by_id[measurement_id])
                ),
                "proof_dependency_receipt_sha256s": sorted(
                    measurement_proof_receipts_by_id[measurement_id]
                ),
                "proof_dependency_receipt_count": len(
                    measurement_proof_receipts_by_id[measurement_id]
                ),
                "occurrence_count": measurement_occurrence_count_by_id[measurement_id],
            }
            for measurement_id in sorted(measurement_semantic_by_id)
        ],
    }


def _validate_binding(
    binding: FeedbackExactBaseBindingV1,
    *,
    canonical_feedback: bytes,
    feedback: Mapping[str, Any],
    view: Mapping[str, Any],
) -> str:
    if type(binding) is not FeedbackExactBaseBindingV1:
        raise TaskspaceInteractionCostateBridgeError("binding must have exact FeedbackExactBaseBindingV1 type")
    baseline = view["baseline"]
    selected = view["selected"]
    custody = view["pointer_custody"]
    observed = {
        "feedback_sha256": _sha256(canonical_feedback),
        "source_receipt_sha256": feedback["source_receipt_sha256"],
        "canonical_receipt_sha256": feedback["canonical_receipt_sha256"],
        "source_lane_id": feedback["source_lane_id"],
        "axis": feedback["axis"],
        "baseline_bundle_sha256": baseline["baseline_bundle_sha256"],
        "baseline_archive_sha256": baseline["selected_archive_sha256"],
        "baseline_archive_nbytes": baseline["selected_archive_bytes"],
        "selected_research_archive_sha256": selected["selected_archive_sha256"],
        "frozen_scorer_sha256": view["frozen_scorer_sha256"],
        "pointer_manifest_sha256": custody["manifest_sha256"],
        "pointer_start_sha256": custody["pointer_start"]["pointer_sha256"],
        "pointer_start_target_score": custody["pointer_start"]["target_score"],
        "pointer_latest_sha256": custody["pointer_latest"]["pointer_sha256"],
        "pointer_latest_target_score": custody["pointer_latest"]["target_score"],
        "source_pair_ids": view["source_pair_ids"],
    }
    expected = _binding_dict(binding)
    expected["source_pair_ids"] = tuple(expected["source_pair_ids"])
    for field_name, observed_value in observed.items():
        if expected[field_name] != observed_value:
            raise TaskspaceInteractionCostateBridgeError(f"{field_name} drifted from exact binding")
    return _sha256(_canonical_json(_binding_dict(binding)))


def _teacher_endpoint_payload(
    teacher: FullLatticeTeacherEndpointV1 | None,
    *,
    binding: FeedbackExactBaseBindingV1,
) -> dict[str, Any]:
    contract = {
        "required_endpoint": "full_lattice_distortion_zero_high_rate_feasible_endpoint_oracle.v1",
        "historical_without_replay_policy": "HINT_ONLY_ZERO_CONTROLLER_INFLUENCE",
        "current_replay_scope": "same_exact_G18_source_pairs_receiver_scorer_base_pointer_population",
        "teacher_is_selected_solution": False,
        "selection_role": "feasible_endpoint_oracle_for_rate_distortion_homotopy_only",
    }
    if teacher is None:
        return {
            **contract,
            "status": "ABSENT",
            "replay_mode": None,
            "controller_influence_performed": False,
            "numeric_homotopy_reference_eligible": False,
            "endpoint": None,
        }
    if type(teacher) is not FullLatticeTeacherEndpointV1:
        raise TaskspaceInteractionCostateBridgeError(
            "teacher_endpoint must have exact FullLatticeTeacherEndpointV1 type"
        )
    reported = {
        "teacher_key": teacher.teacher_key,
        "historical_receipt_sha256": teacher.historical_receipt_sha256,
        "lattice_payload_sha256": teacher.lattice_payload_sha256,
        "lattice_payload_nbytes": teacher.lattice_payload_nbytes,
        "reported_d_seg": teacher.reported_d_seg,
        "reported_d_pose": teacher.reported_d_pose,
        "reported_distortion_zero": True,
    }
    replay = teacher.current_replay
    if replay is None:
        return {
            **contract,
            "status": "HISTORICAL_HINT_ONLY_NOT_REPLAYED_ON_CURRENT_EXACT_RECEIVER",
            "replay_mode": "HISTORICAL_HINT_ONLY",
            "controller_influence_performed": False,
            "numeric_homotopy_reference_eligible": False,
            "endpoint": reported,
            "current_replay": None,
            "distortion_zero_verified_current_receiver": False,
            "high_rate_verified_against_current_compact_base": False,
        }

    exact_bindings = {
        "baseline_bundle_sha256": binding.baseline_bundle_sha256,
        "population_manifest_sha256": binding.population_manifest_sha256,
        "frozen_scorer_sha256": binding.frozen_scorer_sha256,
        "pointer_latest_sha256": binding.pointer_latest_sha256,
        "axis": binding.axis,
        "source_pair_ids": binding.source_pair_ids,
    }
    for field_name, expected in exact_bindings.items():
        if getattr(replay, field_name) != expected:
            raise TaskspaceInteractionCostateBridgeError(
                f"teacher current replay {field_name} drifted from exact binding"
            )
    if replay.d_seg != 0.0 or replay.d_pose != 0.0:
        raise TaskspaceInteractionCostateBridgeError("teacher current replay is not the distortion-zero endpoint")
    if replay.archive_nbytes <= binding.baseline_archive_nbytes:
        raise TaskspaceInteractionCostateBridgeError("teacher current replay is not a measured high-rate endpoint")
    replay_payload = asdict(replay)
    replay_payload["source_pair_ids"] = list(replay.source_pair_ids)
    replay_payload["exact_score"] = _score(replay.d_seg, replay.d_pose, replay.archive_nbytes)
    replay_payload["postcompression_mdl_nbytes"] = replay.archive_nbytes
    return {
        **contract,
        "status": "CURRENT_EXACT_RECEIVER_REPLAY_BOUND_N2_ADVISORY",
        "replay_mode": "CURRENT_EXACT_RECEIVER_REPLAY",
        "controller_influence_performed": False,
        "numeric_homotopy_reference_eligible": True,
        "endpoint": reported,
        "current_replay": replay_payload,
        "distortion_zero_verified_current_receiver": True,
        "high_rate_verified_against_current_compact_base": True,
    }


def _lossless_xcodec_placement_payload(
    receipt: bytes | Mapping[str, Any],
    *,
    receiver_equality: FullN600ReceiverEqualityClosureV1 | None = None,
    admission_control: ReceiverEqualityAdmissionControlV1 | None = None,
) -> dict[str, Any]:
    if type(receipt) is bytes:
        receipt_bytes = receipt
        try:
            value = json.loads(receipt_bytes)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise TaskspaceInteractionCostateBridgeError("REQUANTIZE_STORAGE receipt bytes are not JSON") from exc
    elif type(receipt) is dict:
        value = receipt
        receipt_bytes = _canonical_json(value)
    else:
        raise TaskspaceInteractionCostateBridgeError("representation placement receipt must be exact bytes or dict")
    if type(value) is not dict or value.get("schema") != "tac.ep725_lossless_xcodec_recode.v1":
        raise TaskspaceInteractionCostateBridgeError("representation placement receipt schema is unsupported")

    def exact_dict(raw: object, label: str) -> dict[str, Any]:
        if type(raw) is not dict:
            raise TaskspaceInteractionCostateBridgeError(f"{label} must be an exact object")
        return raw

    source = exact_dict(value.get("source"), "placement source")
    selected = exact_dict(value.get("selected"), "placement selected")
    artifact = exact_dict(value.get("artifact"), "placement artifact")
    delta = exact_dict(value.get("exact_delta"), "placement exact_delta")
    proof = exact_dict(value.get("proof"), "placement proof")
    search = exact_dict(value.get("search"), "placement search")
    runtime = exact_dict(value.get("runtime"), "placement runtime")
    bounded = exact_dict(runtime.get("bounded_equivalence"), "placement bounded equivalence")
    truth = exact_dict(value.get("truth"), "placement truth")
    wire = exact_dict(value.get("system_wire_in"), "placement system wire")
    if (
        value.get("lane_id") != "ep725_lossless_xcodec_recode_20260726"
        or value.get("authority") != "structural exact archive and full quantized-state equality"
        or artifact.get("classification") != "not_a_candidate"
        or search.get("selection_surface") != "exact complete archive.zip bytes"
        or wire.get("bit_allocator_hook") != "REQUANTIZE_STORAGE proposal with exact whole-object price"
    ):
        raise TaskspaceInteractionCostateBridgeError(
            "REQUANTIZE_STORAGE whole-object authority/selection contract drifted"
        )
    for row_name, row in (("source", source), ("selected", selected), ("artifact", artifact)):
        _require_sha256(row.get("archive_sha256") or row.get("sha256"), f"placement {row_name} archive SHA")
        bytes_value = row.get("archive_bytes") if row_name != "artifact" else row.get("bytes")
        if type(bytes_value) is not int or bytes_value < 1:
            raise TaskspaceInteractionCostateBridgeError(f"placement {row_name} archive bytes must be positive int")
    if (
        artifact["sha256"] != selected["archive_sha256"]
        or artifact["bytes"] != selected["archive_bytes"]
        or type(delta.get("archive_bytes")) is not int
        or delta["archive_bytes"] != selected["archive_bytes"] - source["archive_bytes"]
        or delta["archive_bytes"] >= 0
    ):
        raise TaskspaceInteractionCostateBridgeError("REQUANTIZE_STORAGE final archive delta/custody drifted")
    if type(delta.get("member_bytes")) is not int or delta["member_bytes"] != (
        selected["member_bytes"] - source["member_bytes"]
    ):
        raise TaskspaceInteractionCostateBridgeError("REQUANTIZE_STORAGE member delta drifted")
    expected_rate = 25.0 * delta["archive_bytes"] / RATE_DENOMINATOR
    _close(delta.get("rate_score_units"), expected_rate, "placement exact rate score delta")
    if delta.get("formula") != "25*archive_delta_bytes/37545489":
        raise TaskspaceInteractionCostateBridgeError("placement rate formula drifted")
    for field_name in (
        "base_arrays_equal",
        "code_array_equal",
        "deterministic_rebuild_equal",
        "full_quantized_state_equal",
        "manifest_equal_after_removing_xcodec",
        "pose_bytes_equal",
    ):
        if proof.get(field_name) is not True:
            raise TaskspaceInteractionCostateBridgeError(f"placement proof {field_name} is not exact")
    if proof.get("source_state_sha256") != proof.get("selected_state_sha256"):
        raise TaskspaceInteractionCostateBridgeError("placement full-state hashes differ")
    _require_sha256(proof.get("source_state_sha256"), "placement full-state SHA")
    if (
        bounded.get("uint8_raw_equal") is not True
        or bounded.get("full_n600_replay_owed") is not True
        or bounded.get("pair_ids") != [0]
        or bounded.get("frames_compared") != 2
    ):
        raise TaskspaceInteractionCostateBridgeError("placement bounded receiver equality/scope drifted")
    expected_truth = {
        "candidate_claim": False,
        "contest_cpu_cuda_same_bytes_owed": True,
        "exact_eval_invoked": False,
        "full_n600_receiver_replay_owed": True,
        "pointer_moved": False,
        "promotion_eligible": False,
        "public_payload_reused": False,
        "research_only": True,
        "score_claim": False,
    }
    if truth != expected_truth:
        raise TaskspaceInteractionCostateBridgeError("placement research-only authority truth drifted")
    runtime_sha = _require_sha256(runtime.get("sha256"), "placement runtime SHA")
    if type(runtime.get("bytes")) is not int or runtime["bytes"] < 1:
        raise TaskspaceInteractionCostateBridgeError("placement runtime bytes drifted")
    transformed_points = search.get("transformed_points_measured")
    if type(transformed_points) is not int or transformed_points < 1:
        raise TaskspaceInteractionCostateBridgeError("placement search was not finite/exhaustive")
    receipt_sha = _sha256(receipt_bytes)
    equality_payload = None
    if receiver_equality is not None:
        if type(receiver_equality) is not FullN600ReceiverEqualityClosureV1:
            raise TaskspaceInteractionCostateBridgeError(
                "representation placement receiver equality has wrong typed input"
            )
        expected_closure = {
            "g20 receipt": (receiver_equality.g20_receipt_sha256, receipt_sha),
            "source archive SHA": (receiver_equality.source_archive_sha256, source["archive_sha256"]),
            "source archive bytes": (receiver_equality.source_archive_nbytes, source["archive_bytes"]),
            "source member SHA": (receiver_equality.source_member_sha256, source["member_sha256"]),
            "source member bytes": (receiver_equality.source_member_nbytes, source["member_bytes"]),
            "selected archive SHA": (receiver_equality.selected_archive_sha256, selected["archive_sha256"]),
            "selected archive bytes": (receiver_equality.selected_archive_nbytes, selected["archive_bytes"]),
            "selected member SHA": (receiver_equality.selected_member_sha256, selected["member_sha256"]),
            "selected member bytes": (receiver_equality.selected_member_nbytes, selected["member_bytes"]),
            "decoder runtime SHA": (receiver_equality.decoder_runtime_sha256, runtime_sha),
            "decoder runtime bytes": (receiver_equality.decoder_runtime_nbytes, runtime["bytes"]),
            "decoded state SHA": (
                receiver_equality.decoded_quantized_state_sha256,
                proof["source_state_sha256"],
            ),
        }
        for label, (observed, expected) in expected_closure.items():
            if observed != expected:
                raise TaskspaceInteractionCostateBridgeError(
                    f"G22 receiver-equality closure {label} mismatched G20 placement"
                )
        equality_payload = receiver_equality.as_dict(admission_control)
    elif admission_control is not None:
        raise TaskspaceInteractionCostateBridgeError(
            "placement admission control cannot exist without full-n600 receiver equality"
        )
    return {
        "schema": "tac.taskspace_whole_object_representation_placement.v1",
        "source_schema": value["schema"],
        "source_receipt_sha256": receipt_sha,
        "source_lane_id": value["lane_id"],
        "action_semantics": "REQUANTIZE_STORAGE",
        "action_atomicity": "one_exact_whole_object_recode_not_section_marginals",
        "source_reservoirs": ["population_code", "learned_residual", "entropy_context"],
        "target_reservoirs": ["population_code", "learned_residual", "entropy_context"],
        "exact_source_archive": {
            "sha256": source["archive_sha256"],
            "postcompression_mdl_nbytes": source["archive_bytes"],
        },
        "selected_solution": {
            "sha256": selected["archive_sha256"],
            "postcompression_mdl_nbytes": selected["archive_bytes"],
            "decoded_quantized_state_sha256": proof["selected_state_sha256"],
            "selection_surface": "exact_complete_archive_zip_bytes_not_raw_tensor_size",
            "first_feasible_selection_used": False,
            "transformed_points_measured": transformed_points,
        },
        "exact_effect": {
            "delta_postcompression_mdl_nbytes": delta["archive_bytes"],
            "rate_score_delta": expected_rate,
            "distortion_state_delta": "ZERO_FULL_QUANTIZED_STATE_DELTA",
            "distortion_replay_scope": (
                "FULL_N600_RECEIVER_EQUALITY_CLOSED"
                if equality_payload is not None
                else "N1_BOUNDED_RECEIVER_EQUAL_FULL_N600_REPLAY_OWED"
            ),
            "global_score_claim": False,
        },
        "costate": {
            "realized_rate_benefit_s": -expected_rate,
            "finite_whole_action_not_derivative": True,
            "additive_section_attribution_forbidden": True,
            "controller_confidence_update_performed": False,
        },
        "factorization": {
            "weights": "lossless_selected_weight_transposes",
            "codes": "frame_delta_mod256",
            "time": "full_code_time_axis_recode",
            "population": "shared_global_payload_full_n600_replay_owed",
            "semantic": "unchanged_full_quantized_state",
            "realization": "unchanged_full_quantized_state",
            "pose": "unchanged_pose_bytes",
            "entropy": "brotli_members_plus_final_zip",
            "analytic_vs_learned": "analytic_storage_transform_of_counted_learned_payload",
        },
        "placement": {
            "generic_decoder_program_sha256": runtime_sha,
            "generic_decoder_source_nbytes": runtime["bytes"],
            "generic_decoder_rate_cost_in_archive_bytes": 0,
            "generic_operations": ["entropy_decode", "factor_expand"],
            "video_specific_payload_is_counted": True,
            "selected_video_specific_payload_archive_nbytes": selected["archive_bytes"],
            "bounded_receiver_uint8_equal": True,
            "deterministic_rebuild_equal": True,
        },
        "homotopy": {
            "path_role": "exact_rate_only_representation_step_toward_compact_codec",
            "teacher_or_source_archive_sha256": source["archive_sha256"],
            "student_or_selected_archive_sha256": selected["archive_sha256"],
            "bytes_removed": -delta["archive_bytes"],
            "distortion_debt_added": 0.0,
            "rate_score_delta": expected_rate,
        },
        "full_n600_receiver_equality": equality_payload,
        "receiver_equality_handoff_unblocked": equality_payload is not None,
        "contest_score_authority_unblocked": False,
        "truth": expected_truth,
    }


def _population_global_recode_payload(
    receipt: bytes | Mapping[str, Any],
    *,
    parent_placement: Mapping[str, Any],
) -> dict[str, Any]:
    """Project one exact G25 rate endpoint without laundering output equality.

    G25 proves that a population-global spelling decodes to the same quantized
    state as its G20 parent.  It does *not* prove that the public ``inflate.sh``
    runtime consumes LVPG2 or emits the required n600 video.  The controller may
    therefore use the measured ZIP delta as conditional rate telemetry, while
    score/candidate admission remains blocked on a dedicated public-output proof.
    """

    if type(receipt) is bytes:
        receipt_bytes = receipt
        try:
            value = json.loads(receipt_bytes)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise TaskspaceInteractionCostateBridgeError("G25 receipt bytes are not JSON") from exc
    elif type(receipt) is dict:
        value = receipt
        receipt_bytes = _canonical_json(value)
    else:
        raise TaskspaceInteractionCostateBridgeError("G25 receipt must be exact bytes or dict")
    if type(value) is not dict or value.get("schema") != G25_RECEIPT_SCHEMA:
        raise TaskspaceInteractionCostateBridgeError("G25 population-global receipt schema is unsupported")

    def exact_dict(raw: object, label: str) -> dict[str, Any]:
        if type(raw) is not dict:
            raise TaskspaceInteractionCostateBridgeError(f"{label} must be an exact object")
        return raw

    source = exact_dict(value.get("source"), "G25 source")
    selected = exact_dict(value.get("selected"), "G25 selected")
    artifact = exact_dict(value.get("artifact"), "G25 artifact")
    controls = exact_dict(value.get("complete_object_controls"), "G25 controls")
    g20_control = exact_dict(controls.get("g20"), "G25 G20 control")
    delta = exact_dict(value.get("exact_delta"), "G25 exact delta")
    proof = exact_dict(value.get("proof"), "G25 proof")
    runtime = exact_dict(value.get("runtime"), "G25 runtime")
    search = exact_dict(value.get("search"), "G25 search")
    action = exact_dict(value.get("substitutive_action"), "G25 substitutive action")
    action_parent = exact_dict(action.get("parent"), "G25 action parent")
    action_candidate = exact_dict(action.get("candidate"), "G25 action candidate")
    action_effect = exact_dict(action.get("exact_effect"), "G25 action effect")
    truth = exact_dict(value.get("truth"), "G25 truth")
    cleanup = exact_dict(value.get("cleanup_certificate"), "G25 cleanup certificate")

    if (
        value.get("lane_id") != "lane_g25_population_global_same_solution_recode_20260726"
        or value.get("authority") != "structural exact archive and full quantized-state equality"
        or controls.get("selected_control_name") != "g25_v2"
        or artifact.get("classification") != "not_a_candidate"
    ):
        raise TaskspaceInteractionCostateBridgeError("G25 authority/control classification drifted")

    for label, row in (("source", source), ("selected", selected), ("artifact", artifact)):
        digest = row.get("archive_sha256") if label != "artifact" else row.get("sha256")
        nbytes = row.get("archive_bytes") if label != "artifact" else row.get("bytes")
        _require_sha256(digest, f"G25 {label} archive SHA")
        if type(nbytes) is not int or nbytes < 1:
            raise TaskspaceInteractionCostateBridgeError(f"G25 {label} archive bytes must be positive int")
    _require_sha256(g20_control.get("archive_sha256"), "G25 G20 control archive SHA")
    if type(g20_control.get("archive_bytes")) is not int or g20_control["archive_bytes"] < 1:
        raise TaskspaceInteractionCostateBridgeError("G25 G20 control archive bytes must be positive int")
    for label, row in (("source", source), ("selected", selected)):
        _require_sha256(row.get("member_sha256"), f"G25 {label} member SHA")
        if type(row.get("member_bytes")) is not int or row["member_bytes"] < 1:
            raise TaskspaceInteractionCostateBridgeError(f"G25 {label} member bytes must be positive int")
    if (
        artifact["sha256"] != selected["archive_sha256"]
        or artifact["bytes"] != selected["archive_bytes"]
        or selected["archive_bytes"] >= g20_control.get("archive_bytes", 0)
    ):
        raise TaskspaceInteractionCostateBridgeError("G25 selected artifact/cost control drifted")

    parent_source = parent_placement.get("exact_source_archive")
    parent_selected = parent_placement.get("selected_solution")
    if type(parent_source) is not dict or type(parent_selected) is not dict:
        raise TaskspaceInteractionCostateBridgeError("G25 parent G20 placement is incomplete")
    parent_bindings = {
        "source archive SHA": (source["archive_sha256"], parent_source.get("sha256")),
        "source archive bytes": (source["archive_bytes"], parent_source.get("postcompression_mdl_nbytes")),
        "G20 archive SHA": (g20_control.get("archive_sha256"), parent_selected.get("sha256")),
        "G20 archive bytes": (
            g20_control.get("archive_bytes"),
            parent_selected.get("postcompression_mdl_nbytes"),
        ),
        "action parent SHA": (action_parent.get("archive_sha256"), parent_selected.get("sha256")),
        "action parent bytes": (
            action_parent.get("archive_bytes"),
            parent_selected.get("postcompression_mdl_nbytes"),
        ),
    }
    for label, (observed, expected) in parent_bindings.items():
        if observed != expected:
            raise TaskspaceInteractionCostateBridgeError(f"G25 {label} mismatched exact G20 parent")

    required_proof = {
        "base_arrays_equal",
        "canonical_member_reencode_equal",
        "code_array_equal",
        "deterministic_archive_rebuild_equal",
        "full_quantized_state_equal",
        "manifest_equal_after_removing_recode_descriptor",
        "pose_bytes_equal",
    }
    if any(proof.get(field_name) is not True for field_name in required_proof):
        raise TaskspaceInteractionCostateBridgeError("G25 exact same-state proof is incomplete")
    state_sha = _require_sha256(proof.get("source_state_sha256"), "G25 state SHA")
    parent_state_sha = _require_sha256(
        parent_selected.get("decoded_quantized_state_sha256"),
        "G20 parent state SHA",
    )
    if proof.get("selected_state_sha256") != state_sha:
        raise TaskspaceInteractionCostateBridgeError("G25 source/selected v2 quantized states differ")

    delta_g20 = selected["archive_bytes"] - g20_control["archive_bytes"]
    delta_source = selected["archive_bytes"] - source["archive_bytes"]
    if (
        delta.get("formula") != "25*archive_delta_bytes/37545489"
        or delta.get("versus_g20_archive_bytes") != delta_g20
        or delta.get("versus_source_archive_bytes") != delta_source
        or delta_g20 >= 0
    ):
        raise TaskspaceInteractionCostateBridgeError("G25 exact archive delta drifted")
    rate_g20 = 25.0 * delta_g20 / RATE_DENOMINATOR
    rate_source = 25.0 * delta_source / RATE_DENOMINATOR
    _close(delta.get("versus_g20_rate_score_units"), rate_g20, "G25 rate delta versus G20")
    _close(delta.get("versus_source_rate_score_units"), rate_source, "G25 rate delta versus source")

    if (
        action.get("schema") != "tac.selected_solution_substitutive_action.v1"
        or action.get("action_semantics") != "REQUANTIZE_STORAGE"
        or action.get("action_atomicity") != "one_exact_whole_object_recode_not_section_marginals"
        or action.get("effect_observation_kind") != "ENDPOINT"
        or action.get("section_marginal_attribution_forbidden") is not True
        or action_candidate.get("archive_sha256") != selected["archive_sha256"]
        or action_candidate.get("archive_bytes") != selected["archive_bytes"]
        or action_effect.get("delta_archive_bytes") != delta_g20
        or action_effect.get("distortion_state_delta") != "ZERO_FULL_QUANTIZED_STATE_DELTA"
    ):
        raise TaskspaceInteractionCostateBridgeError("G25 substitutive action drifted")
    _close(action_effect.get("rate_score_delta"), rate_g20, "G25 action rate delta")

    parent_runtime = parent_placement.get("placement")
    if type(parent_runtime) is not dict:
        raise TaskspaceInteractionCostateBridgeError("G25 parent decoder placement is incomplete")
    if (
        _require_sha256(runtime.get("sha256"), "G25 runtime SHA")
        != parent_runtime.get("generic_decoder_program_sha256")
        or runtime.get("bytes") != parent_runtime.get("generic_decoder_source_nbytes")
        or runtime.get("full_n600_output_replay_owed") is not True
        or runtime.get("v2_state_receiver")
        != "tac.witness_dsl.ep725_population_global_recode_v2.parse_population_global_member"
    ):
        raise TaskspaceInteractionCostateBridgeError("G25 runtime/public-output blocker drifted")

    if (
        search.get("search_kind") != "deterministic_exact_whole_object_coordinate_descent"
        or search.get("selection_surface") != "exact complete archive.zip bytes"
        or search.get("converged_whole_cycle") is not True
        or search.get("global_minimum_outside_sealed_menu_claimed") is not False
        or type(search.get("points_measured")) is not int
        or search["points_measured"] < 1
        or type(search.get("stages")) is not list
        or not search["stages"]
    ):
        raise TaskspaceInteractionCostateBridgeError("G25 finite search custody drifted")
    hyperedges = value.get("interaction_hyperedges")
    if type(hyperedges) is not list or len(hyperedges) < 2:
        raise TaskspaceInteractionCostateBridgeError("G25 population-rate interactions are missing")
    for hyperedge in hyperedges:
        if (
            type(hyperedge) is not dict
            or hyperedge.get("schema") != "tac.complete_object_rate_interaction_hyperedge.v1"
            or hyperedge.get("support") != "COMPLETE"
            or hyperedge.get("effect_observation_kind") != "INDIVISIBLE_HYPEREDGE"
            or hyperedge.get("additive_attribution_forbidden") is not True
            or type(hyperedge.get("interaction_archive_bytes")) is not int
        ):
            raise TaskspaceInteractionCostateBridgeError("G25 population-rate hyperedge drifted")

    expected_truth = {
        "candidate_claim": False,
        "contest_cpu_cuda_same_bytes_owed": True,
        "exact_eval_invoked": False,
        "full_n600_quantized_state_used": True,
        "full_n600_runtime_output_replay_owed": True,
        "pointer_moved": False,
        "promotion_eligible": False,
        "public_payload_reused": False,
        "research_only": True,
        "score_claim": False,
    }
    if truth != expected_truth:
        raise TaskspaceInteractionCostateBridgeError("G25 research-only truth drifted")
    if (
        cleanup.get("schema") != "tac.disk_hygiene_certificate.v1"
        or cleanup.get("bulk_scratch_created") is not False
        or cleanup.get("destructive_cleanup_performed") is not False
        or cleanup.get("temporary_paths_remaining") != []
    ):
        raise TaskspaceInteractionCostateBridgeError("G25 cleanup custody drifted")

    return {
        "schema": "tac.taskspace_population_global_recode_costate.v1",
        "source_schema": G25_RECEIPT_SCHEMA,
        "source_receipt_sha256": _sha256(receipt_bytes),
        "source_lane_id": value["lane_id"],
        "action_semantics": "REQUANTIZE_STORAGE",
        "action_atomicity": "population_chronology_x_tensor_coordinates_x_inner_codec_x_outer_zip",
        "parent_g20": {
            "sha256": parent_selected["sha256"],
            "postcompression_mdl_nbytes": parent_selected["postcompression_mdl_nbytes"],
            "v1_decoded_quantized_state_sha256": parent_state_sha,
            "full_n600_private_receiver_equality_closed": parent_placement.get("receiver_equality_handoff_unblocked")
            is True,
        },
        "conditional_selected_solution": {
            "sha256": selected["archive_sha256"],
            "postcompression_mdl_nbytes": selected["archive_bytes"],
            "member_sha256": selected["member_sha256"],
            "member_nbytes": selected["member_bytes"],
            "decoded_quantized_state_sha256": state_sha,
            "decoded_state_identity_domain": "ep725_population_global_recode_v2",
            "selection_surface": "exact_complete_archive_zip_bytes",
            "complete_archives_measured": search["points_measured"],
        },
        "exact_effect": {
            "delta_postcompression_mdl_nbytes_versus_g20": delta_g20,
            "delta_postcompression_mdl_nbytes_versus_source": delta_source,
            "rate_score_delta_versus_g20": rate_g20,
            "rate_score_delta_versus_source": rate_source,
            "distortion_state_delta": "ZERO_FULL_QUANTIZED_STATE_DELTA",
            "finite_whole_action_not_derivative": True,
            "additive_section_attribution_forbidden": True,
        },
        "factorization": {
            "weights": "selected_tensor_coordinate_transforms",
            "codes": "population_global_frame_latent_pair_trajectories",
            "time": "long_temporal_population_trajectories",
            "population": "one_n600_shared_recode",
            "semantic": "unchanged_quantized_state",
            "realization": "public_runtime_adapter_owed",
            "pose": "unchanged_pose_bytes",
            "entropy": "inner_layout_codec_x_outer_zip_nonadditive",
            "analytic_vs_learned": "analytic_storage_transform_of_counted_learned_payload",
        },
        "interaction_hyperedges": [
            {
                "hyperedge_id": row.get("hyperedge_id"),
                "interaction_archive_bytes": row["interaction_archive_bytes"],
                "support": row["support"],
                "additive_attribution_forbidden": True,
            }
            for row in hyperedges
        ],
        "public_video_contract": {
            "expected_frame_count": 1_200,
            "expected_height": 874,
            "expected_width": 1_164,
            "expected_channels": 3,
            "expected_dtype": "uint8",
            "public_inflate_sh_output_equality_proved": False,
            "upstream_evaluate_recursive_closure_proved": False,
            "g20_v1_to_g25_v2_state_identity_join_proved": False,
            "blocker": "G25_LVPG2_PUBLIC_DECODER_AND_FULL_N600_OUTPUT_EQUALITY_OWED",
        },
        "telemetry_admission": {
            "conditional_rate_planning_eligible": True,
            "exact_score_endpoint_eligible": False,
            "candidate_eligible": False,
            "training_decision_authority": False,
            "reason": (
                "measured global ZIP benefit is usable as conditional rate telemetry; "
                "the selected archive is not a scored video until public decode closure"
            ),
        },
        "truth": expected_truth,
    }


def _same_object_bridge_score_payload(
    receipt: bytes | Mapping[str, Any],
    *,
    binding: FeedbackExactBaseBindingV1,
    parent_placement: Mapping[str, Any],
    receiver_equality: FullN600ReceiverEqualityClosureV1,
    population_global_recodes: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Admit an exact G28 row for routing while preserving advisory authority.

    G28 is the first measured full-population ``(d_seg, d_pose, archive bytes)``
    triple on the exact G20/G22 object.  It may route the next inverse action and
    reprice a G25 spelling conditionally.  It may not become a contest score,
    public-decoder proof, promotion row, or independent per-axis threshold.
    """

    if type(receipt) is bytes:
        value, canonical = _canonical_receipt_payload(receipt, label="G28 receipt")
    elif type(receipt) is dict:
        value = receipt
        canonical = _canonical_json(value) + b"\n"
    else:
        raise TaskspaceInteractionCostateBridgeError("G28 receipt must be exact bytes or dict")
    if value.get("schema") != G28_RECEIPT_SCHEMA:
        raise TaskspaceInteractionCostateBridgeError("G28 same-object score receipt schema is unsupported")

    identity = _require_sha256(value.get("receipt_identity_sha256"), "G28 receipt identity")
    identity_payload = dict(value)
    identity_payload.pop("receipt_identity_sha256")
    if identity != _sha256(_canonical_json(identity_payload)):
        raise TaskspaceInteractionCostateBridgeError("G28 receipt identity arithmetic drifted")

    def exact_dict(raw: object, label: str) -> dict[str, Any]:
        if type(raw) is not dict:
            raise TaskspaceInteractionCostateBridgeError(f"{label} must be an exact object")
        return raw

    artifact = exact_dict(value.get("artifact_identity"), "G28 artifact identity")
    authority = exact_dict(value.get("authority"), "G28 authority")
    proof_dependencies = exact_dict(value.get("proof_dependency_set"), "G28 proof dependency set")
    score = exact_dict(value.get("score_row"), "G28 score row")
    semantic = exact_dict(value.get("semantic_control_identity"), "G28 semantic control identity")
    run = exact_dict(value.get("run_custody"), "G28 run custody")

    expected_authority = {
        "axis": "[macOS-CPU frozen-scorer advisory]",
        "contest_cpu_cuda_same_bytes_owed": True,
        "pointer_moved": False,
        "promotion_eligible": False,
        "research_only": True,
        "score_claim": False,
    }
    if authority != expected_authority:
        raise TaskspaceInteractionCostateBridgeError("G28 advisory authority classification drifted")
    expected_dependencies = {
        "competitive_admission": ["current canonical frontier pointer"],
        "decode_equality": ["selected archive", "selected member", "generic runtime", "G22 receipt"],
        "pointer_change_invalidates_decode_equality": False,
        "score_row": ["selected raw SHA-256", "frozen scorer", "source video", "scorer arithmetic"],
    }
    if proof_dependencies != expected_dependencies:
        raise TaskspaceInteractionCostateBridgeError("G28 proof dependency domains drifted")

    selected_archive = _custody_row(
        artifact.get("selected_archive"), label="G28 selected archive", absolute_path=True
    )
    scored_archive = _custody_row(
        artifact.get("scored_archive"), label="G28 scored archive", absolute_path=True
    )
    runtime = _custody_row(artifact.get("runtime"), label="G28 runtime", absolute_path=True)
    g22_receipt = _custody_row(artifact.get("receipt"), label="G28 G22 receipt", absolute_path=True)
    selected_member = exact_dict(artifact.get("selected_member"), "G28 selected member")
    _require_sha256(selected_member.get("sha256"), "G28 selected member SHA")
    if (
        selected_member.get("name") != "0.bin"
        or type(selected_member.get("bytes")) is not int
        or selected_member["bytes"] < 1
    ):
        raise TaskspaceInteractionCostateBridgeError("G28 selected member shape drifted")

    parent_selected = parent_placement.get("selected_solution")
    if type(parent_selected) is not dict:
        raise TaskspaceInteractionCostateBridgeError("G28 exact G20 parent placement is incomplete")
    exact_identity_matches = {
        "selected archive SHA": (selected_archive["sha256"], receiver_equality.selected_archive_sha256),
        "selected archive bytes": (selected_archive["bytes"], receiver_equality.selected_archive_nbytes),
        "scored archive SHA": (scored_archive["sha256"], receiver_equality.selected_archive_sha256),
        "scored archive bytes": (scored_archive["bytes"], receiver_equality.selected_archive_nbytes),
        "G20 parent archive SHA": (selected_archive["sha256"], parent_selected.get("sha256")),
        "G20 parent archive bytes": (
            selected_archive["bytes"],
            parent_selected.get("postcompression_mdl_nbytes"),
        ),
        "selected member SHA": (selected_member["sha256"], receiver_equality.selected_member_sha256),
        "selected member bytes": (selected_member["bytes"], receiver_equality.selected_member_nbytes),
        "runtime SHA": (runtime["sha256"], receiver_equality.decoder_runtime_sha256),
        "runtime bytes": (runtime["bytes"], receiver_equality.decoder_runtime_nbytes),
        "G22 receipt SHA": (g22_receipt["sha256"], receiver_equality.g22_receipt_sha256),
        "G22 receipt bytes": (g22_receipt["bytes"], receiver_equality.g22_receipt_nbytes),
        "decoded state SHA": (
            artifact.get("decoded_quantized_state_sha256"),
            receiver_equality.decoded_quantized_state_sha256,
        ),
        "realized output SHA": (
            artifact.get("selected_raw_sha256"),
            receiver_equality.realized_output_sha256,
        ),
        "realized output bytes": (artifact.get("raw_bytes"), receiver_equality.realized_output_nbytes),
    }
    for label, (observed, expected) in exact_identity_matches.items():
        if observed != expected:
            raise TaskspaceInteractionCostateBridgeError(f"G28 {label} mismatched exact G20/G22 parent")
    if (
        artifact.get("pair_count") != POPULATION_PAIR_COUNT
        or artifact.get("frame_count") != 2 * POPULATION_PAIR_COUNT
        or artifact.get("scored_archive_matches_g22_selected") is not True
        or artifact.get("scored_output_matches_g22_reference") is not True
        or artifact.get("decode_equality_independent_of_pointer_change") is not True
    ):
        raise TaskspaceInteractionCostateBridgeError("G28 full-population same-object identity drifted")

    d_seg = _finite(score.get("d_seg"), "G28 d_seg")
    d_pose = _finite(score.get("d_pose"), "G28 d_pose")
    target_score = _finite(score.get("target_score"), "G28 target score")
    if d_seg < 0.0 or d_pose < 0.0 or target_score <= 0.0:
        raise TaskspaceInteractionCostateBridgeError("G28 score values left their physical domain")
    if (
        score.get("archive_bytes") != selected_archive["bytes"]
        or score.get("raw_sha256") != receiver_equality.realized_output_sha256
        or score.get("pair_count") != POPULATION_PAIR_COUNT
        or score.get("batch_size") != 16
        or score.get("batch_count") != math.ceil(POPULATION_PAIR_COUNT / 16)
        or score.get("device") != "cpu"
        or score.get("deterministic_algorithms") is not True
        or score.get("formula") != "100*d_seg + sqrt(10*d_pose) + 25*archive_bytes/37545489"
        or score.get("independent_axis_thresholds_forbidden") is not True
    ):
        raise TaskspaceInteractionCostateBridgeError("G28 scorer shape or formula drifted")
    seg_component = 100.0 * d_seg
    pose_component = math.sqrt(10.0 * d_pose)
    rate_component = 25.0 * selected_archive["bytes"] / RATE_DENOMINATOR
    measured_score = seg_component + pose_component + rate_component
    _close(score.get("seg_component"), seg_component, "G28 Seg component")
    _close(score.get("pose_component"), pose_component, "G28 Pose component")
    _close(score.get("rate_component"), rate_component, "G28 rate component")
    _close(score.get("score"), measured_score, "G28 score")
    _close(score.get("score_minus_target"), measured_score - target_score, "G28 target delta")
    inside = measured_score < target_score
    if score.get("inside_target_sublevel") is not inside:
        raise TaskspaceInteractionCostateBridgeError("G28 target classification drifted")
    expected_verdict = (
        "MEASURED_FULL_N600_SAME_OBJECT_BRIDGE_INSIDE_ADVISORY_TARGET"
        if inside
        else "MEASURED_FULL_N600_SAME_OBJECT_BRIDGE_OUTSIDE_TARGET"
    )
    if value.get("verdict") != expected_verdict:
        raise TaskspaceInteractionCostateBridgeError("G28 verdict drifted from recomputed action")

    boundaries = exact_dict(score.get("conditional_boundaries"), "G28 conditional boundaries")
    distortion_component = seg_component + pose_component
    _close(
        boundaries.get("rate_score_headroom_at_same_distortions"),
        target_score - distortion_component,
        "G28 rate headroom",
    )
    pose_residual = target_score - seg_component - rate_component
    seg_residual = target_score - pose_component - rate_component
    expected_pose_boundary = pose_residual * pose_residual / 10.0 if pose_residual >= 0.0 else None
    expected_seg_boundary = seg_residual / 100.0 if seg_residual >= 0.0 else None
    for field_name, expected in (
        ("max_d_pose_at_same_d_seg_and_bytes", expected_pose_boundary),
        ("max_d_seg_at_same_d_pose_and_bytes", expected_seg_boundary),
    ):
        observed = boundaries.get(field_name)
        if expected is None:
            if observed is not None:
                raise TaskspaceInteractionCostateBridgeError(f"G28 {field_name} should be infeasible")
        else:
            _close(observed, expected, f"G28 {field_name}")

    semantic_pointer = _custody_row(
        semantic.get("pointer_evidence"), label="G28 semantic pointer", absolute_path=True
    )
    if (
        semantic.get("role") != "competitive_score_to_beat"
        or semantic.get("selection_rule")
        != "min(our_local_frontier_contest_cpu, our_local_frontier_contest_cuda, upstream_official_leaderboard.best_entry)"
        or semantic.get("custody") != "external target only; no local archive authority implied"
        or semantic.get("score_precision") != "official_display"
    ):
        raise TaskspaceInteractionCostateBridgeError("G28 semantic target custody drifted")
    _close(semantic.get("score"), target_score, "G28 semantic target")

    cleanup = exact_dict(run.get("cleanup"), "G28 cleanup")
    raw_before_cleanup = _custody_row(
        run.get("raw_before_cleanup"), label="G28 raw before cleanup", absolute_path=True
    )
    scorer_stage = _custody_row(run.get("scorer_stage"), label="G28 scorer stage", absolute_path=True)
    scorer_evidence = _custody_row(score.get("scorer_evidence"), label="G28 score evidence", absolute_path=True)
    pointer_observations = exact_dict(run.get("pointer_observations"), "G28 pointer observations")
    pointer_start = _custody_row(
        pointer_observations.get("start"), label="G28 pointer start", absolute_path=True
    )
    pointer_end = _custody_row(
        pointer_observations.get("end"), label="G28 pointer end", absolute_path=True
    )
    if (
        cleanup.get("completed") is not True
        or cleanup.get("success_only") is not True
        or cleanup.get("raw_absent") is not True
        or cleanup.get("decode_checkpoints_preserved") is not True
        or cleanup.get("scorer_stage_preserved") is not True
        or run.get("decode_checkpoint_count") != 50
        or raw_before_cleanup["sha256"] != receiver_equality.realized_output_sha256
        or raw_before_cleanup["bytes"] != receiver_equality.realized_output_nbytes
        or scorer_stage != scorer_evidence
        or pointer_start != pointer_end
        or pointer_end != semantic_pointer
    ):
        raise TaskspaceInteractionCostateBridgeError("G28 run, scorer, cleanup, or pointer custody drifted")

    matching_recodes = [
        row
        for row in population_global_recodes
        if row.get("parent_g20", {}).get("sha256") == selected_archive["sha256"]
        and row.get("parent_g20", {}).get("postcompression_mdl_nbytes") == selected_archive["bytes"]
    ]
    conditional_archive_bytes = min(
        (
            row["conditional_selected_solution"]["postcompression_mdl_nbytes"]
            for row in matching_recodes
        ),
        default=selected_archive["bytes"],
    )
    conditional_rate = 25.0 * conditional_archive_bytes / RATE_DENOMINATOR
    conditional_score = distortion_component + conditional_rate
    coupled_budget = target_score - conditional_rate
    max_pose_at_zero_seg = coupled_budget * coupled_budget / 10.0 if coupled_budget >= 0.0 else None
    max_seg_at_zero_pose = coupled_budget / 100.0 if coupled_budget >= 0.0 else None
    pose_reduction_to_zero_seg_feasibility = (
        None if max_pose_at_zero_seg is None else max(0.0, 1.0 - max_pose_at_zero_seg / d_pose)
    )
    seg_reduction_to_zero_pose_feasibility = (
        None if max_seg_at_zero_pose is None else max(0.0, 1.0 - max_seg_at_zero_pose / d_seg)
    )
    if coupled_budget < 0.0:
        uniform_scale_boundary = None
    elif seg_component == 0.0:
        uniform_scale_boundary = (coupled_budget / pose_component) ** 2 if pose_component else 1.0
    else:
        root = (-pose_component + math.sqrt(pose_component**2 + 4.0 * seg_component * coupled_budget)) / (
            2.0 * seg_component
        )
        uniform_scale_boundary = max(0.0, root * root)

    components = {
        "seg": seg_component,
        "pose": pose_component,
        "rate": rate_component,
    }
    priority = sorted(components, key=lambda name: (-components[name], name))
    pointer_rebased = (
        semantic_pointer["sha256"] != binding.pointer_latest_sha256
        or not math.isclose(target_score, binding.pointer_latest_target_score, rel_tol=0.0, abs_tol=1e-15)
    )
    return {
        "schema": "tac.taskspace_same_object_score_costate.v1",
        "source_schema": G28_RECEIPT_SCHEMA,
        "source_receipt_sha256": _sha256(canonical),
        "source_receipt_identity_sha256": identity,
        "authority": expected_authority,
        "exact_object": {
            "g20_receipt_sha256": receiver_equality.g20_receipt_sha256,
            "g22_receipt_sha256": receiver_equality.g22_receipt_sha256,
            "archive_sha256": selected_archive["sha256"],
            "archive_nbytes": selected_archive["bytes"],
            "runtime_sha256": runtime["sha256"],
            "realized_output_sha256": receiver_equality.realized_output_sha256,
            "realized_output_nbytes": receiver_equality.realized_output_nbytes,
            "pair_count": POPULATION_PAIR_COUNT,
            "frame_count": 2 * POPULATION_PAIR_COUNT,
        },
        "measured_action": {
            "d_seg": d_seg,
            "d_pose": d_pose,
            "archive_nbytes": selected_archive["bytes"],
            "seg_component": seg_component,
            "pose_component": pose_component,
            "rate_component": rate_component,
            "distortion_component": distortion_component,
            "score": measured_score,
            "observed_target_score": target_score,
            "score_minus_observed_target": measured_score - target_score,
            "inside_observed_target_sublevel": inside,
        },
        "conditional_population_global_rate_projection": {
            "available": bool(matching_recodes),
            "archive_nbytes": conditional_archive_bytes,
            "rate_component": conditional_rate,
            "same_distortion_conditional_score": conditional_score,
            "score_minus_observed_target": conditional_score - target_score,
            "score_endpoint_eligible": False,
            "reason": "G25 rate is exact same-state telemetry but its public video output remains unproved",
        },
        "coupled_frontier_geometry": {
            "independent_axis_thresholds_forbidden": True,
            "target_distortion_budget_at_conditional_rate": coupled_budget,
            "max_d_pose_at_zero_seg": max_pose_at_zero_seg,
            "max_d_seg_at_zero_pose": max_seg_at_zero_pose,
            "required_pose_reduction_fraction_at_zero_seg": pose_reduction_to_zero_seg_feasibility,
            "required_seg_reduction_fraction_at_zero_pose": seg_reduction_to_zero_pose_feasibility,
            "uniform_joint_distortion_scale_boundary": uniform_scale_boundary,
            "required_uniform_joint_distortion_reduction_fraction": (
                None if uniform_scale_boundary is None else max(0.0, 1.0 - uniform_scale_boundary)
            ),
        },
        "controller_route": {
            "component_priority": priority,
            "dominant_component": priority[0],
            "next_phase": "MAXIMAL_INVERSE_POSE_PRESERVING_REPAIR",
            "then": [
                "JOINT_TASKSPACE_SEMANTIC_AND_REALIZATION_REPAIR",
                "FUNCTIONAL_QUOTIENT_MERGE_PRUNE_AND_MACRO_EVICTION",
                "TRAIN_ONLY_TELEMETRY_PROVEN_IRREDUCIBLE_QUOTIENT",
                "TERMINAL_WHOLE_OBJECT_JOINT_DESCENT",
            ],
            "uncorrected_ep725_bridge_retry_admitted": False,
            "selection_rule": "globally_best_exact_whole_object_endpoint_then_invalidate_and_remeasure",
            "reason": "Pose is the recomputed dominant component; rate-only recoding cannot enter the target sublevel",
        },
        "decoder_placement_policy": {
            "generic_repair_projection_iteration_may_live_in_inflate": True,
            "generic_decode_compute_has_zero_rate_cost": True,
            "video_specific_parameters_latents_selectors_exceptions_are_counted": True,
            "scorer_teacher_original_video_or_gt_access_at_decode_forbidden": True,
            "runtime_memory_determinism_and_public_graph_closure_are_hard_constraints": True,
        },
        "target_rebase": {
            "observed_pointer_artifact_sha256": semantic_pointer["sha256"],
            "current_binding_pointer_artifact_sha256": binding.pointer_latest_sha256,
            "measurement_geometry_rebase_required": pointer_rebased,
            "decode_equality_invalidated": False,
        },
        "telemetry_admission": {
            "component_routing_eligible": True,
            "conditional_rate_planning_eligible": bool(matching_recodes),
            "contest_score_endpoint_eligible": False,
            "candidate_eligible": False,
            "promotion_eligible": False,
            "actuation": "NONE",
        },
    }


def _symmetric_agreement(predicted: float, realized: float) -> float:
    denominator = abs(predicted) + abs(realized)
    if denominator == 0.0:
        return 1.0
    return max(0.0, 1.0 - abs(predicted - realized) / denominator)


def _parse_state(
    state: Mapping[str, Any] | None,
    *,
    binding: FeedbackExactBaseBindingV1,
) -> dict[str, dict[str, Any]]:
    if state is None:
        return {}
    if type(state) is not dict or set(state) != {
        "schema",
        "axis",
        "population_manifest_sha256",
        "aggregation",
        "proposals",
    }:
        raise TaskspaceInteractionCostateBridgeError("prior controller state schema drifted")
    if (
        state["schema"] != STATE_SCHEMA
        or state["axis"] != binding.axis
        or state["population_manifest_sha256"] != binding.population_manifest_sha256
        or state["aggregation"] != "arithmetic_mean_of_content_addressed_joint_score_agreements_not_probability.v1"
        or type(state["proposals"]) is not list
    ):
        raise TaskspaceInteractionCostateBridgeError("prior controller state axis/population drifted")
    result: dict[str, dict[str, Any]] = {}
    for entry in state["proposals"]:
        if type(entry) is not dict or set(entry) != {
            "proposal_key",
            "effect_kind",
            "action_semantics",
            "source_reservoirs",
            "target_reservoirs",
            "factorization",
            "decoder_placement",
            "homotopy_path_key",
            "homotopy_step_index",
            "semantic_evidence_sha256",
            "observations",
            "observation_count",
            "advisory_confidence",
        }:
            raise TaskspaceInteractionCostateBridgeError("prior proposal state schema drifted")
        key = _require_identifier(entry["proposal_key"], "prior proposal_key")
        if key in result or type(entry["observations"]) is not list:
            raise TaskspaceInteractionCostateBridgeError("prior proposal repeats or lacks observations")
        if entry["effect_kind"] not in ("transition", "interaction"):
            raise TaskspaceInteractionCostateBridgeError("prior proposal effect_kind drifted")
        _require_identifier(entry["homotopy_path_key"], "prior homotopy_path_key")
        if type(entry["homotopy_step_index"]) is not int or entry["homotopy_step_index"] < 0:
            raise TaskspaceInteractionCostateBridgeError("prior homotopy_step_index drifted")
        _factorization_from_mapping(entry["factorization"])
        _placement_from_mapping(entry["decoder_placement"])
        _validate_action_signature(
            entry["action_semantics"],
            entry["source_reservoirs"],
            entry["target_reservoirs"],
            entry["semantic_evidence_sha256"],
            collection_type=list,
        )
        observations: list[dict[str, Any]] = []
        seen: set[str] = set()
        for observation in entry["observations"]:
            expected_fields = {
                "observation_id",
                "binding_fingerprint_sha256",
                "feedback_sha256",
                "exact_base_archive_sha256",
                "pointer_latest_sha256",
                "target_score",
                "joint_score_agreement",
                "component_agreement",
            }
            if type(observation) is not dict or set(observation) != expected_fields:
                raise TaskspaceInteractionCostateBridgeError("prior observation schema drifted")
            observation_id = _require_sha256(observation["observation_id"], "observation_id")
            for field_name in (
                "binding_fingerprint_sha256",
                "feedback_sha256",
                "exact_base_archive_sha256",
                "pointer_latest_sha256",
            ):
                _require_sha256(observation[field_name], field_name)
            agreement = _finite(observation["joint_score_agreement"], "joint_score_agreement")
            _finite(observation["target_score"], "observation.target_score")
            if not 0.0 <= agreement <= 1.0 or observation_id in seen:
                raise TaskspaceInteractionCostateBridgeError("prior observation agreement/identity drifted")
            components = observation["component_agreement"]
            if type(components) is not dict or set(components) != {
                "d_seg",
                "d_pose",
                "archive_bytes",
            }:
                raise TaskspaceInteractionCostateBridgeError("prior component agreement schema drifted")
            if any(not 0.0 <= _finite(value, "component agreement") <= 1.0 for value in components.values()):
                raise TaskspaceInteractionCostateBridgeError("prior component agreement left [0,1]")
            seen.add(observation_id)
            observations.append(dict(observation))
        if entry["observation_count"] != len(observations):
            raise TaskspaceInteractionCostateBridgeError("prior observation_count drifted")
        confidence = (
            sum(row["joint_score_agreement"] for row in observations) / len(observations) if observations else 0.0
        )
        _close(entry["advisory_confidence"], confidence, "prior advisory_confidence")
        result[key] = {**entry, "observations": observations}
    return result


def _state_payload(entries: Mapping[str, dict[str, Any]], binding: FeedbackExactBaseBindingV1) -> dict[str, Any]:
    proposals: list[dict[str, Any]] = []
    for key in sorted(entries):
        entry = entries[key]
        observations = sorted(entry["observations"], key=lambda row: row["observation_id"])
        confidence = (
            sum(row["joint_score_agreement"] for row in observations) / len(observations) if observations else 0.0
        )
        proposals.append(
            {
                **{
                    field: entry[field]
                    for field in (
                        "proposal_key",
                        "effect_kind",
                        "action_semantics",
                        "source_reservoirs",
                        "target_reservoirs",
                        "factorization",
                        "decoder_placement",
                        "homotopy_path_key",
                        "homotopy_step_index",
                        "semantic_evidence_sha256",
                    )
                },
                "observations": observations,
                "observation_count": len(observations),
                "advisory_confidence": confidence,
            }
        )
    return {
        "schema": STATE_SCHEMA,
        "axis": binding.axis,
        "population_manifest_sha256": binding.population_manifest_sha256,
        "aggregation": "arithmetic_mean_of_content_addressed_joint_score_agreements_not_probability.v1",
        "proposals": proposals,
    }


def _blocked_handoffs(
    predictions: Sequence[ActionEffectPredictionV1],
    *,
    teacher: Mapping[str, Any],
    placements: Sequence[Mapping[str, Any]],
    population_global_recodes: Sequence[Mapping[str, Any]],
    same_object_scores: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    gauge_present = any(row.action_semantics == "GAUGE_REPRESENTATIVE" for row in predictions)
    placement_count = len(placements)
    placement_equality_closed = bool(placements) and all(
        row.get("receiver_equality_handoff_unblocked") is True for row in placements
    )
    return [
        {
            "consumer_api": "tac.master_gradient.append_anchor_locked",
            "status": "BLOCKED_INCOMPATIBLE_NO_CALL",
            "reason": (
                "G18 supplies n2 whole-object finite differences, not an n600 per-byte gradient tensor, "
                "sidecar path, runtime call ID, or contest-axis archive anchor."
            ),
            "required_evidence": [
                "content-bound aggregate/per-pair gradient sidecar",
                "n_pairs_used=n_pairs_total=600",
                "scored archive SHA/bytes and runtime custody",
            ],
            "write_performed": False,
        },
        {
            "consumer_api": "tac.witness_control.costate_organ_v3.append_realized_delta_row",
            "status": "BLOCKED_INCOMPATIBLE_NO_CALL",
            "reason": (
                "The v3 corpus requires exact_gap/visibility/realizability/byte_price factors and a "
                "durable repo-relative receipt. G18 does not identify those factors, and its interaction "
                "must not be collapsed into one additive realized_benefit_s row."
            ),
            "required_evidence": ["canonical v3 factors", "durable reviewed receipt", "n600 scope"],
            "write_performed": False,
        },
        {
            "consumer_api": "tac.witness_control.costate_posterior.record_costate_observation",
            "status": "BLOCKED_INCOMPATIBLE_NO_CALL",
            "reason": "n2 effect agreement is not an independent n600 costate estimate with a standard error.",
            "required_evidence": ["n600 identifiable costate", "uncertainty estimate", "byte-close run reference"],
            "write_performed": False,
        },
        {
            "consumer_api": "tac.witness_control.continual_costate.append_trajectory_record",
            "status": "BLOCKED_INCOMPATIBLE_NO_CALL",
            "reason": "G18 is a discrete n2 action grid, not a campaign trajectory/backtest/prototype record.",
            "required_evidence": ["n600 trajectory", "architecture backtest reports", "prototype record"],
            "write_performed": False,
        },
        {
            "consumer_api": "tac.witness_dsl.taskspace_whole_archive_allocator.allocate_taskspace_whole_archive",
            "status": "BLOCKED_REMEASUREMENT_REQUIRED_NO_CALL",
            "reason": (
                "G18 preserves hashes and outcomes but not P/G/A/T packet bytes, transforms, or receiver/measurement "
                "callbacks. ADD, DELETE/PRUNE, MERGE/SHARE, FACTOR, REPLACE, MIGRATE, and REQUANTIZE_STORAGE must "
                "each rebuild and remeasure the "
                "whole current prefix; atomic marginals cannot substitute."
            ),
            "required_evidence": [
                "TaskspaceSectionBundleV1 bytes",
                "typed proposal transform",
                "double-decode receiver receipt",
                "exact whole-object measurement receipt",
            ],
            "write_performed": False,
        },
        {
            "consumer_api": (
                "tac.witness_control.taskspace_receding_horizon_controller_v1."
                "select_receding_horizon_action_v1"
            ),
            "status": "BLOCKED_COMPLETE_EXACT_ENDPOINT_UNIVERSE_OWED_NO_CALL",
            "reason": (
                "The terminal controller is implemented, but G18/G20/G22/G25/G28 do not form a complete "
                "same-base micro-through-system action universe with public decoder and recursive auth-eval "
                "closure on every endpoint. Calling it on a partial universe would turn caller order and "
                "missing macro actions into an unauthorized selection prior."
            ),
            "required_evidence": [
                "one content-addressed exact public whole-object base",
                "complete typed current-epoch action-family and scale coverage",
                "public double-decode and recursive auth-eval closure per endpoint",
                "same explicit authority mode and hardware axis as the exact base",
                "whole-object Seg/Pose/final-archive measurements per endpoint",
                "evaluator-cell identity plus exact proof receipt per endpoint",
                "continuation-equivalence identity plus exact proof receipt per endpoint",
            ],
            "write_performed": False,
        },
        {
            "consumer_api": "tac.ddm_costate_organ.build_live_ddm_costate",
            "status": "HINT_ONLY_NOT_CONSUMED",
            "reason": (
                "The live DDM organ discovers the historical V19-family receipt fleet and has no typed external "
                "action-observation input for DELETE/PRUNE, MERGE/SHARE, FACTOR, REPLACE, MIGRATE, "
                "REQUANTIZE_STORAGE, or gauge selection. "
                "Its fixed family-local ordering cannot calibrate this exact-base G18 evidence."
            ),
            "required_evidence": ["new typed DDM action-observation consumer", "current exact baseline hash"],
            "write_performed": False,
        },
        {
            "consumer_api": "evaluator_equivalent_gauge_representative_selector",
            "status": "BLOCKED_NO_LAWFUL_CONSUMER" if gauge_present else "NOT_REQUESTED",
            "reason": (
                "A supplied equivalence-receipt hash is preserved but not dereferenced. G18 has no proof that two "
                "representatives are evaluator-equivalent through R; confidence therefore covers observed effect "
                "agreement only, never the gauge claim."
            ),
            "required_evidence": [
                "content-validated equivalence receipt",
                "same target cells through R",
                "whole-object byte-close remeasurement",
            ],
            "write_performed": False,
        },
        {
            "consumer_api": "full_lattice_teacher_rate_distortion_costate_ingest",
            "status": (
                "BLOCKED_N2_ADVISORY_NO_WRITE"
                if teacher.get("numeric_homotopy_reference_eligible") is True
                else "HINT_ONLY_NOT_CONSUMED"
                if teacher.get("status") != "ABSENT"
                else "NOT_REQUESTED"
            ),
            "reason": (
                "A current exact receiver replay may anchor n2 homotopy telemetry, but neither it nor an "
                "unreplayed historical lattice is an n600 posterior or promotion row."
            ),
            "required_evidence": [
                "current exact n600 receiver replay",
                "final archive bytes and runtime custody",
                "contest-axis same-byte evaluation before authority",
            ],
            "write_performed": False,
        },
        {
            "consumer_api": "whole_object_representation_placement_costate_ingest",
            "status": (
                "READY_RECEIVER_EQUALITY_CLOSED_CONTEST_EVAL_SEPARATELY_OWED_NO_WRITE"
                if placement_equality_closed
                else "BLOCKED_FULL_N600_REPLAY_OWED_NO_WRITE"
                if placement_count
                else "NOT_REQUESTED"
            ),
            "reason": (
                "The lossless xcodec remains one indivisible whole-object observation. Receiver equality is "
                "usable only when exact G20/G22 archive, runtime, member, pair-order, decoded-output, and final "
                "receipt custody close; contest score authority remains a separate same-byte receipt."
            ),
            "required_evidence": [
                "full n600 frozen-receiver replay",
                "same selected archive bytes",
                "no additive section double-counting",
            ],
            "write_performed": False,
        },
        {
            "consumer_api": "population_global_recode_costate_ingest",
            "status": (
                "BLOCKED_PUBLIC_VIDEO_OUTPUT_CLOSURE_NO_WRITE" if population_global_recodes else "NOT_REQUESTED"
            ),
            "reason": (
                "G25 is an exact whole-population same-state rate observation, but its LVPG2 inverse is not "
                "yet consumed by the public inflate.sh runtime. It may set a conditional physical-rate floor; "
                "it may not become a score endpoint or training-decision authority until the exact 1200-frame "
                "public output and upstream recursive dependency closure are proved."
            ),
            "required_evidence": [
                "generic LVPG2 inverse in public decoder",
                "bit-identical full n600 public output",
                "recursive upstream evaluate dependency closure",
            ],
            "write_performed": False,
        },
        {
            "consumer_api": "same_object_bridge_score_costate_ingest",
            "status": (
                "READY_ADVISORY_COMPONENT_ROUTING_NO_PROMOTION_NO_WRITE"
                if same_object_scores
                else "NOT_REQUESTED"
            ),
            "reason": (
                "G28 closes a full-n600 same-object advisory Seg/Pose/rate triple and may route the next "
                "inverse action. Its macOS CPU frozen-scorer axis is not contest CPU/CUDA authority, and the "
                "private receiver is not public inflate/evaluate closure."
            ),
            "required_evidence": [
                "generic public decoder emits exact 1200-frame video",
                "two fresh deterministic public decodes",
                "upstream evaluate.sh recursive runtime closure",
                "contest CPU/CUDA same-archive measurements",
            ],
            "write_performed": False,
        },
    ]


def build_taskspace_interaction_costate_bridge_v1(
    feedback: bytes | Mapping[str, Any],
    *,
    binding: FeedbackExactBaseBindingV1,
    predictions: Sequence[ActionEffectPredictionV1] = (),
    teacher_endpoint: FullLatticeTeacherEndpointV1 | None = None,
    representation_placement_receipts: Sequence[bytes | Mapping[str, Any]] = (),
    representation_placement_receiver_equalities: Sequence[FullN600ReceiverEqualityClosureV1] = (),
    representation_placement_admission_controls: Mapping[str, ReceiverEqualityAdmissionControlV1] | None = None,
    population_global_recode_receipts: Sequence[bytes | Mapping[str, Any]] = (),
    same_object_bridge_score_receipts: Sequence[bytes | Mapping[str, Any]] = (),
    prior_controller_state: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build deterministic advisory calibration/acquisition telemetry from G18."""

    if type(binding) is not FeedbackExactBaseBindingV1:
        raise TaskspaceInteractionCostateBridgeError("binding must have exact FeedbackExactBaseBindingV1 type")
    canonical_feedback, feedback_value = _load_feedback(feedback)
    view = _validate_feedback(feedback_value, sample_count=len(binding.source_pair_ids))
    binding_fingerprint = _validate_binding(
        binding,
        canonical_feedback=canonical_feedback,
        feedback=feedback_value,
        view=view,
    )
    teacher_payload = _teacher_endpoint_payload(teacher_endpoint, binding=binding)
    try:
        placement_sources = tuple(representation_placement_receipts)
    except TypeError as exc:
        raise TaskspaceInteractionCostateBridgeError(
            "representation_placement_receipts must be a finite sequence"
        ) from exc
    try:
        equality_sources = tuple(representation_placement_receiver_equalities)
    except TypeError as exc:
        raise TaskspaceInteractionCostateBridgeError(
            "representation placement receiver equalities must be a finite sequence"
        ) from exc
    if any(type(item) is not FullN600ReceiverEqualityClosureV1 for item in equality_sources):
        raise TaskspaceInteractionCostateBridgeError(
            "every receiver equality must have exact FullN600ReceiverEqualityClosureV1 type"
        )
    equality_by_g20 = {item.g20_receipt_sha256: item for item in equality_sources}
    if len(equality_by_g20) != len(equality_sources):
        raise TaskspaceInteractionCostateBridgeError("receiver equality closure is duplicated")
    controls = (
        {} if representation_placement_admission_controls is None else dict(representation_placement_admission_controls)
    )
    if any(
        type(key) is not str
        or _SHA256_RE.fullmatch(key) is None
        or type(value) is not ReceiverEqualityAdmissionControlV1
        for key, value in controls.items()
    ):
        raise TaskspaceInteractionCostateBridgeError(
            "placement admission controls must map G22 receipt SHA to typed control"
        )
    placement_rows = []
    consumed_equalities: set[str] = set()
    consumed_controls: set[str] = set()
    for row in placement_sources:
        raw = row if type(row) is bytes else _canonical_json(row)
        g20_sha = _sha256(raw)
        equality = equality_by_g20.get(g20_sha)
        control = None
        if equality is not None:
            consumed_equalities.add(g20_sha)
            control = controls.get(equality.g22_receipt_sha256)
            if control is not None:
                consumed_controls.add(equality.g22_receipt_sha256)
        placement_rows.append(
            _lossless_xcodec_placement_payload(
                row,
                receiver_equality=equality,
                admission_control=control,
            )
        )
    if consumed_equalities != set(equality_by_g20):
        raise TaskspaceInteractionCostateBridgeError(
            "receiver equality closure has no exact matching G20 placement receipt"
        )
    if consumed_controls != set(controls):
        raise TaskspaceInteractionCostateBridgeError("placement admission control has no exact matching G22 closure")
    if len({row["source_receipt_sha256"] for row in placement_rows}) != len(placement_rows):
        raise TaskspaceInteractionCostateBridgeError("representation placement receipt is duplicated")
    try:
        global_recode_sources = tuple(population_global_recode_receipts)
    except TypeError as exc:
        raise TaskspaceInteractionCostateBridgeError(
            "population_global_recode_receipts must be a finite sequence"
        ) from exc
    parent_by_selected = {
        (
            row["selected_solution"]["sha256"],
            row["selected_solution"]["postcompression_mdl_nbytes"],
        ): row
        for row in placement_rows
    }
    population_global_rows: list[dict[str, Any]] = []
    for source in global_recode_sources:
        if type(source) is bytes:
            try:
                source_value = json.loads(source)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise TaskspaceInteractionCostateBridgeError("G25 receipt bytes are not JSON") from exc
        elif type(source) is dict:
            source_value = source
        else:
            raise TaskspaceInteractionCostateBridgeError("G25 receipt must be exact bytes or dict")
        controls_value = source_value.get("complete_object_controls")
        g20_value = controls_value.get("g20") if type(controls_value) is dict else None
        if type(g20_value) is not dict:
            raise TaskspaceInteractionCostateBridgeError("G25 receipt lacks an exact G20 parent control")
        parent = parent_by_selected.get((g20_value.get("archive_sha256"), g20_value.get("archive_bytes")))
        if parent is None:
            raise TaskspaceInteractionCostateBridgeError(
                "G25 population-global recode has no exact ingested G20 parent placement"
            )
        population_global_rows.append(_population_global_recode_payload(source, parent_placement=parent))
    if len({row["source_receipt_sha256"] for row in population_global_rows}) != len(population_global_rows):
        raise TaskspaceInteractionCostateBridgeError("G25 population-global receipt is duplicated")
    try:
        same_object_score_sources = tuple(same_object_bridge_score_receipts)
    except TypeError as exc:
        raise TaskspaceInteractionCostateBridgeError(
            "same_object_bridge_score_receipts must be a finite sequence"
        ) from exc
    equality_by_g22 = {item.g22_receipt_sha256: item for item in equality_sources}
    same_object_score_rows: list[dict[str, Any]] = []
    for source in same_object_score_sources:
        if type(source) is bytes:
            try:
                source_value = json.loads(source)
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                raise TaskspaceInteractionCostateBridgeError("G28 receipt bytes are not JSON") from exc
        elif type(source) is dict:
            source_value = source
        else:
            raise TaskspaceInteractionCostateBridgeError("G28 receipt must be exact bytes or dict")
        artifact_value = source_value.get("artifact_identity")
        g22_value = artifact_value.get("receipt") if type(artifact_value) is dict else None
        selected_value = artifact_value.get("selected_archive") if type(artifact_value) is dict else None
        if type(g22_value) is not dict or type(selected_value) is not dict:
            raise TaskspaceInteractionCostateBridgeError("G28 receipt lacks exact G22/G20 object identity")
        equality = equality_by_g22.get(g22_value.get("sha256"))
        parent = parent_by_selected.get((selected_value.get("sha256"), selected_value.get("bytes")))
        if equality is None or parent is None:
            raise TaskspaceInteractionCostateBridgeError(
                "G28 same-object score has no exact ingested G20 placement and G22 equality parents"
            )
        same_object_score_rows.append(
            _same_object_bridge_score_payload(
                source,
                binding=binding,
                parent_placement=parent,
                receiver_equality=equality,
                population_global_recodes=population_global_rows,
            )
        )
    if len({row["source_receipt_sha256"] for row in same_object_score_rows}) != len(same_object_score_rows):
        raise TaskspaceInteractionCostateBridgeError("G28 same-object score receipt is duplicated")
    try:
        ordered_predictions = tuple(predictions)
    except TypeError as exc:
        raise TaskspaceInteractionCostateBridgeError("predictions must be a finite sequence") from exc
    if any(type(item) is not ActionEffectPredictionV1 for item in ordered_predictions):
        raise TaskspaceInteractionCostateBridgeError("every prediction must have exact ActionEffectPredictionV1 type")
    if len({item.effect_id for item in ordered_predictions}) != len(ordered_predictions):
        raise TaskspaceInteractionCostateBridgeError("one effect may not receive multiple predictions")
    if len({(item.homotopy_path_key, item.homotopy_step_index) for item in ordered_predictions}) != len(
        ordered_predictions
    ):
        raise TaskspaceInteractionCostateBridgeError("one homotopy path step may not receive multiple predictions")

    effects = {effect["effect_id"]: effect for effect in (*view["transition_effects"], *view["interaction_effects"])}
    sign_reversal_rows = [
        {
            "interaction_id": effect["effect_id"],
            "treatment_id": effect["treatment_id"],
            "g8_proposal_id": effect["g8_proposal_id"],
            "family": effect["family"],
            "prefix_order": effect["prefix_order"],
            "palette_bound_per_class": effect["palette_bound_per_class"],
            "prefix_cell_count": effect["prefix_cell_count"],
            "classification": effect["conditional_action_effect"]["sign_reversal"],
            "action_effect_without_g8": effect["conditional_action_effect"][
                "action_effect_without_g8"
            ],
            "action_effect_with_g8": effect["conditional_action_effect"][
                "action_effect_with_g8"
            ],
            "interaction_residual": effect["conditional_action_effect"][
                "interaction_residual"
            ],
            "four_cell_archive_custody": effect["realized_archive_custody"],
        }
        for effect in view["interaction_effects"]
        if effect["conditional_action_effect"]["sign_reversal"] is not None
    ]
    state_entries = _parse_state(prior_controller_state, binding=binding)
    calibrations: list[dict[str, Any]] = []
    costate_rows: list[dict[str, Any]] = []
    homotopy_rows: list[dict[str, Any]] = []
    lam_field: dict[str, float] = {}
    measured_binding: dict[str, float] = {}
    acquisition_meta: dict[str, tuple[str, str]] = {}
    feedback_sha = _sha256(canonical_feedback)

    for prediction in sorted(ordered_predictions, key=lambda item: (item.effect_id, item.proposal_key)):
        effect = effects.get(prediction.effect_id)
        if effect is None:
            raise TaskspaceInteractionCostateBridgeError(
                f"prediction effect_id {prediction.effect_id!r} is absent from exact G18 feedback"
            )
        for field_name, predicted, realized in (
            ("effect_kind", prediction.effect_kind, effect["effect_kind"]),
            ("exact_base_measurement_id", prediction.exact_base_measurement_id, effect["exact_base_measurement_id"]),
            ("exact_base_archive_sha256", prediction.exact_base_archive_sha256, effect["exact_base_archive_sha256"]),
            ("baseline_bundle_sha256", prediction.baseline_bundle_sha256, effect["baseline_bundle_sha256"]),
            ("population_manifest_sha256", prediction.population_manifest_sha256, binding.population_manifest_sha256),
        ):
            if predicted != realized:
                raise TaskspaceInteractionCostateBridgeError(
                    f"prediction {prediction.effect_id} {field_name} drifted from exact effect base"
                )
        if prediction.effect_kind == "transition" and (
            prediction.predicted_postcompression_archive_nbytes
            != effect["exact_base_archive_nbytes"] + prediction.predicted_delta_archive_bytes
        ):
            raise TaskspaceInteractionCostateBridgeError(
                "predicted post-compression archive bytes drifted from exact base plus delta"
            )

        predicted_components = {
            "delta_d_seg": float(prediction.predicted_delta_d_seg),
            "delta_d_pose": float(prediction.predicted_delta_d_pose),
            "delta_archive_bytes": prediction.predicted_delta_archive_bytes,
        }
        if prediction.effect_kind == "transition":
            base = next(row for row in view["transition_effects"] if row["effect_id"] == prediction.effect_id)
            # Reconstruct the base components from the realized endpoint delta.
            realized_after = base["realized"]
            before_d_seg = next(
                occurrence["measurement"]
                for occurrence in feedback_value["row_inventory"]["occurrences"]
                if occurrence["measurement"]["measurement_id"] == prediction.exact_base_measurement_id
            )["d_seg"]
            before_d_pose = next(
                occurrence["measurement"]
                for occurrence in feedback_value["row_inventory"]["occurrences"]
                if occurrence["measurement"]["measurement_id"] == prediction.exact_base_measurement_id
            )["d_pose"]
            before_bytes = effect["exact_base_archive_nbytes"]
            predicted_score = _score(
                before_d_seg + prediction.predicted_delta_d_seg,
                before_d_pose + prediction.predicted_delta_d_pose,
                before_bytes + prediction.predicted_delta_archive_bytes,
            ) - _score(before_d_seg, before_d_pose, before_bytes)
            del realized_after  # documents that no realized component is substituted into prediction
            score_derivation = "nonlinear_score_law_on_predicted_endpoint_components.v1"
        else:
            predicted_score = float(prediction.predicted_joint_score_effect)
            score_derivation = "independently_predicted_four_cell_score_residual_not_component_sum.v1"

        realized = effect["realized"]
        component_agreement = {
            "d_seg": _symmetric_agreement(predicted_components["delta_d_seg"], realized["delta_d_seg"]),
            "d_pose": _symmetric_agreement(predicted_components["delta_d_pose"], realized["delta_d_pose"]),
            "archive_bytes": _symmetric_agreement(
                float(predicted_components["delta_archive_bytes"]),
                float(realized["delta_archive_bytes"]),
            ),
        }
        joint_agreement = _symmetric_agreement(predicted_score, realized["joint_score_effect"])
        prediction_payload = _prediction_dict(prediction)
        observation_seed = {
            "binding_fingerprint_sha256": binding_fingerprint,
            "feedback_sha256": feedback_sha,
            "prediction": prediction_payload,
            "realized": realized,
        }
        observation_id = _sha256(_canonical_json(observation_seed))
        observation = {
            "observation_id": observation_id,
            "binding_fingerprint_sha256": binding_fingerprint,
            "feedback_sha256": feedback_sha,
            "exact_base_archive_sha256": prediction.exact_base_archive_sha256,
            "pointer_latest_sha256": binding.pointer_latest_sha256,
            "target_score": binding.pointer_latest_target_score,
            "joint_score_agreement": joint_agreement,
            "component_agreement": component_agreement,
        }
        action_signature = {
            "effect_kind": prediction.effect_kind,
            "action_semantics": prediction.action_semantics,
            "source_reservoirs": list(prediction.source_reservoirs),
            "target_reservoirs": list(prediction.target_reservoirs),
            "factorization": _factorization_dict(prediction.factorization),
            "decoder_placement": _placement_dict(prediction.decoder_placement),
            "homotopy_path_key": prediction.homotopy_path_key,
            "homotopy_step_index": prediction.homotopy_step_index,
            "semantic_evidence_sha256": prediction.semantic_evidence_sha256,
        }
        entry = state_entries.get(prediction.proposal_key)
        if entry is None:
            entry = {
                "proposal_key": prediction.proposal_key,
                **action_signature,
                "observations": [],
            }
            state_entries[prediction.proposal_key] = entry
        elif any(entry[field_name] != value for field_name, value in action_signature.items()):
            raise TaskspaceInteractionCostateBridgeError(
                f"proposal_key {prediction.proposal_key!r} changed effect/action semantics"
            )
        observations = entry["observations"]
        confidence_before = (
            sum(row["joint_score_agreement"] for row in observations) / len(observations) if observations else None
        )
        duplicate = observation_id in {row["observation_id"] for row in observations}
        if not duplicate:
            observations.append(observation)
        confidence_after = sum(row["joint_score_agreement"] for row in observations) / len(observations)
        semantic_status = (
            "DECLARED_WITH_UNVERIFIED_EXTERNAL_EVIDENCE_HASH"
            if prediction.semantic_evidence_sha256 is not None
            else "DECLARED_ONLY_G18_HAS_NO_OPERATOR_SEMANTICS"
        )
        impact_envelope = max(abs(predicted_score), abs(realized["joint_score_effect"]))
        calibration = {
            "effect_id": prediction.effect_id,
            "proposal_key": prediction.proposal_key,
            "effect_kind": prediction.effect_kind,
            "component_effect_semantics": effect["component_effect_semantics"],
            "action_semantics": prediction.action_semantics,
            "source_reservoirs": list(prediction.source_reservoirs),
            "target_reservoirs": list(prediction.target_reservoirs),
            "factorization": _factorization_dict(prediction.factorization),
            "decoder_placement": {
                **_placement_dict(prediction.decoder_placement),
                "generic_decoder_rate_cost_in_archive_bytes": 0,
                "video_specific_payload_counted_in_final_archive": True,
                "evidence_hashes_content_validated_by_g19": False,
                "runtime_within_30_min_declared": prediction.decoder_placement.decoder_runtime_seconds <= 1800.0,
            },
            "homotopy_path_key": prediction.homotopy_path_key,
            "homotopy_step_index": prediction.homotopy_step_index,
            "semantic_claim_status": semantic_status,
            "semantic_evidence_sha256": prediction.semantic_evidence_sha256,
            "exact_base": {
                "measurement_id": effect["exact_base_measurement_id"],
                "archive_sha256": effect["exact_base_archive_sha256"],
                "archive_nbytes": effect["exact_base_archive_nbytes"],
                "baseline_bundle_sha256": effect["baseline_bundle_sha256"],
                "population_manifest_sha256": binding.population_manifest_sha256,
                "source_pair_ids": list(binding.source_pair_ids),
                "population_pair_count": binding.population_pair_count,
            },
            "realized_archive_custody": effect["realized_archive_custody"],
            "predicted": {
                **predicted_components,
                "joint_score_effect": predicted_score,
                "joint_score_derivation": score_derivation,
            },
            "realized": realized,
            "residual_realized_minus_predicted": {
                "delta_d_seg": realized["delta_d_seg"] - predicted_components["delta_d_seg"],
                "delta_d_pose": realized["delta_d_pose"] - predicted_components["delta_d_pose"],
                "delta_archive_bytes": realized["delta_archive_bytes"] - predicted_components["delta_archive_bytes"],
                "joint_score_effect": realized["joint_score_effect"] - predicted_score,
            },
            "agreement": {"joint_score": joint_agreement, **component_agreement},
            "observation_id": observation_id,
            "duplicate_observation": duplicate,
            "advisory_confidence_before": confidence_before,
            "advisory_confidence_after": confidence_after,
            "controller_acquisition_priority": (1.0 - confidence_after) * impact_envelope,
            "confidence_semantics": "mean symmetric agreement; advisory calibration, not probability or authority",
            "additive_independence_assumed": False,
        }
        calibrations.append(calibration)
        hierarchy_keys = (
            ("atom", prediction.factorization.atom_key),
            ("group", prediction.factorization.group_key),
            ("shard", prediction.factorization.shard_key),
            ("global", prediction.factorization.global_key),
        )
        for granularity, costate_key in hierarchy_keys:
            costate_rows.append(
                {
                    "granularity": granularity,
                    "costate_key": costate_key,
                    "proposal_key": prediction.proposal_key,
                    "effect_id": prediction.effect_id,
                    "effect_kind": prediction.effect_kind,
                    "action_semantics": prediction.action_semantics,
                    "factorization": _factorization_dict(prediction.factorization),
                    "predicted_finite_action_score_costate": -predicted_score,
                    "realized_finite_action_score_costate": -realized["joint_score_effect"],
                    "joint_score_agreement": joint_agreement,
                    "observation_id": observation_id,
                    "same_effect_reindexed_across_hierarchy": True,
                    "additive_rollup_performed": False,
                    "derivative_claim": False,
                    "authority": "n2_macos_advisory_exact_whole_effect",
                }
            )
        if prediction.effect_kind == "transition":
            endpoint = effect["homotopy_endpoint"]
            after = endpoint["after"]
            teacher_metrics: dict[str, Any] | None = None
            if teacher_payload["numeric_homotopy_reference_eligible"] is True:
                teacher_replay = teacher_payload["current_replay"]
                bytes_removed = teacher_replay["postcompression_mdl_nbytes"] - after["postcompression_mdl_nbytes"]
                teacher_metrics = {
                    "teacher_archive_sha256": teacher_replay["archive_sha256"],
                    "teacher_postcompression_mdl_nbytes": teacher_replay["postcompression_mdl_nbytes"],
                    "bytes_removed_from_teacher": bytes_removed,
                    "compact_rate_fraction_of_teacher": after["postcompression_mdl_nbytes"]
                    / teacher_replay["postcompression_mdl_nbytes"],
                    "distortion_debt_from_zero": _distortion(after["d_seg"], after["d_pose"]),
                    "global_score_delta_from_teacher": after["global_score"] - teacher_replay["exact_score"],
                    "distortion_debt_per_removed_byte": (
                        _distortion(after["d_seg"], after["d_pose"]) / bytes_removed if bytes_removed > 0 else None
                    ),
                }
            homotopy_rows.append(
                {
                    "homotopy_path_key": prediction.homotopy_path_key,
                    "homotopy_step_index": prediction.homotopy_step_index,
                    "path_role": "exact_current_base_transition_endpoint",
                    "proposal_key": prediction.proposal_key,
                    "effect_id": prediction.effect_id,
                    "action_semantics": prediction.action_semantics,
                    "factorization": _factorization_dict(prediction.factorization),
                    "before": endpoint["before"],
                    "after": endpoint["after"],
                    "postcompression_mdl_source": "selected_complete_archive_zip_bytes",
                    "raw_tensor_size_used": False,
                    "first_feasible_selection_used": False,
                    "teacher_comparison": teacher_metrics,
                    "teacher_comparison_status": (
                        "CURRENT_EXACT_RECEIVER_BOUND"
                        if teacher_metrics is not None
                        else "BLOCKED_HISTORICAL_OR_ABSENT_TEACHER"
                    ),
                    "authority": "n2_macos_advisory",
                }
            )
        else:
            homotopy_rows.append(
                {
                    "homotopy_path_key": prediction.homotopy_path_key,
                    "homotopy_step_index": prediction.homotopy_step_index,
                    "path_role": "nonadditive_four_cell_interaction_costate_not_endpoint",
                    "proposal_key": prediction.proposal_key,
                    "effect_id": prediction.effect_id,
                    "action_semantics": prediction.action_semantics,
                    "factorization": _factorization_dict(prediction.factorization),
                    "before": None,
                    "after": None,
                    "postcompression_mdl_source": "four_exact_complete_archive_cells",
                    "raw_tensor_size_used": False,
                    "first_feasible_selection_used": False,
                    "teacher_comparison": None,
                    "teacher_comparison_status": "NOT_AN_ENDPOINT_COMPARISON",
                    "authority": "n2_macos_advisory",
                }
            )
        lever_id = f"{prediction.proposal_key}@{prediction.effect_id}"
        lam_field[lever_id] = predicted_score
        measured_binding[lever_id] = realized["joint_score_effect"]
        acquisition_meta[lever_id] = (prediction.proposal_key, prediction.effect_id)

    acquisition_rows: list[dict[str, Any]] = []
    if lam_field:
        ranked = powerplay_acquisition(
            lam_field,
            measured_binding=measured_binding,
            ever_fired=dict.fromkeys(lam_field, True),
            costs=dict.fromkeys(lam_field, 1.0),
        )
        for row in ranked:
            proposal_key, effect_id = acquisition_meta[row.lever]
            acquisition_rows.append(
                {
                    **asdict(row),
                    "proposal_key": proposal_key,
                    "effect_id": effect_id,
                    "atomic_control_unit": (
                        "one exact transition"
                        if effect_id.startswith("transition:")
                        else "one four-cell interaction residual"
                    ),
                }
            )

    state = _state_payload(state_entries, binding)
    selected = view["selected"]
    selected_owner = next(
        (
            row
            for row in calibrations
            if row["effect_kind"] == "transition"
            and row["realized_archive_custody"]["after_archive_sha256"] == selected["selected_archive_sha256"]
        ),
        None,
    )
    selected_solution = {
        "selection_role": "selected_complete_evaluator_measured_solution_not_problem_inventory",
        "measurement_id": selected["measurement_id"],
        "archive_sha256": selected["selected_archive_sha256"],
        "postcompression_mdl_nbytes": selected["selected_archive_bytes"],
        "d_seg": selected["d_seg"],
        "d_pose": selected["d_pose"],
        "global_score": _measurement_score(selected),
        "selection_objective": ("minimize_100_dseg_plus_sqrt_10_dpose_plus_25_final_archive_zip_bytes_over_37545489"),
        "global_score_recomputed": True,
        "final_archive_bytes_used": True,
        "raw_tensor_or_uncompressed_member_size_used": False,
        "first_feasible_selection_used": False,
        "factorability_priced_via_realized_postcompression_mdl_not_proxy": True,
        "evaluator_equivalence_claim": False,
        "evaluator_equivalence_status": "MEASURED_DISTORTION_RETAINED_EXACTLY_NOT_CALLED_ZERO",
        "factorization": selected_owner["factorization"] if selected_owner is not None else None,
        "decoder_placement": selected_owner["decoder_placement"] if selected_owner is not None else None,
        "placement_attribution_status": (
            "BOUND_TO_PREDICTED_TRANSITION"
            if selected_owner is not None
            else "NO_PREDICTION_BOUND_TO_SELECTED_MEASUREMENT"
        ),
        "authority": "n2_macos_cpu_advisory_not_candidate_or_score_claim",
    }
    telemetry_rows: list[dict[str, Any]] = [
        {
            "stage": TELEMETRY_STAGE,
            "event": "exact_binding_closed",
            "schema": SCHEMA,
            "axis": binding.axis,
            "feedback_sha256": feedback_sha,
            "binding_fingerprint_sha256": binding_fingerprint,
            "population_manifest_sha256": binding.population_manifest_sha256,
            "source_pair_count": len(binding.source_pair_ids),
            "population_pair_count": binding.population_pair_count,
            "transition_count": len(view["transition_effects"]),
            "interaction_count": len(view["interaction_effects"]),
            "calibration_count": len(calibrations),
            "additive_independence_assumed": False,
            "score_claim": False,
            "promotion_eligible": False,
            "actuation": "NONE",
        },
        {
            "stage": TELEMETRY_STAGE,
            "event": "selected_evaluator_solution_observed",
            "schema": SCHEMA,
            "axis": binding.axis,
            "measurement_id": selected_solution["measurement_id"],
            "archive_sha256": selected_solution["archive_sha256"],
            "postcompression_mdl_nbytes": selected_solution["postcompression_mdl_nbytes"],
            "global_score": selected_solution["global_score"],
            "raw_tensor_size_used": False,
            "first_feasible_selection_used": False,
            "score_claim": False,
            "promotion_eligible": False,
            "actuation": "NONE",
        },
        {
            "stage": TELEMETRY_STAGE,
            "event": "full_lattice_teacher_status",
            "schema": SCHEMA,
            "axis": binding.axis,
            "status": teacher_payload["status"],
            "numeric_homotopy_reference_eligible": teacher_payload["numeric_homotopy_reference_eligible"],
            "controller_influence_performed": False,
            "score_claim": False,
            "promotion_eligible": False,
            "actuation": "NONE",
        },
    ]
    telemetry_rows.extend(
        {
            "stage": TELEMETRY_STAGE,
            "event": "proposal_effect_calibrated",
            "schema": SCHEMA,
            "axis": binding.axis,
            "proposal_key": row["proposal_key"],
            "effect_id": row["effect_id"],
            "effect_kind": row["effect_kind"],
            "action_semantics": row["action_semantics"],
            "predicted_joint_score_effect": row["predicted"]["joint_score_effect"],
            "realized_joint_score_effect": row["realized"]["joint_score_effect"],
            "joint_score_agreement": row["agreement"]["joint_score"],
            "advisory_confidence_after": row["advisory_confidence_after"],
            "controller_acquisition_priority": row["controller_acquisition_priority"],
            "observation_id": row["observation_id"],
            "additive_independence_assumed": False,
            "score_claim": False,
            "promotion_eligible": False,
            "actuation": "NONE",
        }
        for row in calibrations
    )
    telemetry_rows.extend(
        {
            "stage": TELEMETRY_STAGE,
            "event": "hierarchical_finite_action_costate_harvested",
            "schema": SCHEMA,
            "axis": binding.axis,
            "granularity": row["granularity"],
            "costate_key": row["costate_key"],
            "proposal_key": row["proposal_key"],
            "effect_id": row["effect_id"],
            "action_semantics": row["action_semantics"],
            "realized_finite_action_score_costate": row["realized_finite_action_score_costate"],
            "additive_rollup_performed": False,
            "derivative_claim": False,
            "score_claim": False,
            "promotion_eligible": False,
            "actuation": "NONE",
        }
        for row in costate_rows
    )
    telemetry_rows.extend(
        {
            "stage": TELEMETRY_STAGE,
            "event": "rate_distortion_homotopy_step_observed",
            "schema": SCHEMA,
            "axis": binding.axis,
            "homotopy_path_key": row["homotopy_path_key"],
            "homotopy_step_index": row["homotopy_step_index"],
            "path_role": row["path_role"],
            "proposal_key": row["proposal_key"],
            "effect_id": row["effect_id"],
            "action_semantics": row["action_semantics"],
            "raw_tensor_size_used": False,
            "teacher_comparison_status": row["teacher_comparison_status"],
            "score_claim": False,
            "promotion_eligible": False,
            "actuation": "NONE",
        }
        for row in homotopy_rows
    )
    telemetry_rows.extend(
        {
            "stage": TELEMETRY_STAGE,
            "event": "whole_object_representation_placement_observed",
            "schema": SCHEMA,
            "axis": placement["truth"]["research_only"] and "[macOS-CPU frozen receiver structural proof]",
            "source_receipt_sha256": placement["source_receipt_sha256"],
            "action_semantics": placement["action_semantics"],
            "delta_postcompression_mdl_nbytes": placement["exact_effect"]["delta_postcompression_mdl_nbytes"],
            "rate_score_delta": placement["exact_effect"]["rate_score_delta"],
            "additive_section_attribution_forbidden": True,
            "score_claim": False,
            "promotion_eligible": False,
            "actuation": "NONE",
        }
        for placement in placement_rows
    )
    telemetry_rows.extend(
        {
            "stage": TELEMETRY_STAGE,
            "event": "population_global_same_state_rate_observed",
            "schema": SCHEMA,
            "axis": "[exact archive/state structural proof]",
            "source_receipt_sha256": row["source_receipt_sha256"],
            "archive_nbytes": row["conditional_selected_solution"]["postcompression_mdl_nbytes"],
            "rate_score_delta_versus_g20": row["exact_effect"]["rate_score_delta_versus_g20"],
            "public_video_output_closure_proved": False,
            "score_claim": False,
            "promotion_eligible": False,
            "actuation": "NONE",
        }
        for row in population_global_rows
    )
    telemetry_rows.extend(
        {
            "stage": TELEMETRY_STAGE,
            "event": "full_n600_same_object_component_debt_observed",
            "schema": SCHEMA,
            "axis": row["authority"]["axis"],
            "source_receipt_sha256": row["source_receipt_sha256"],
            "archive_sha256": row["exact_object"]["archive_sha256"],
            "d_seg": row["measured_action"]["d_seg"],
            "d_pose": row["measured_action"]["d_pose"],
            "archive_nbytes": row["measured_action"]["archive_nbytes"],
            "seg_component": row["measured_action"]["seg_component"],
            "pose_component": row["measured_action"]["pose_component"],
            "rate_component": row["measured_action"]["rate_component"],
            "global_score": row["measured_action"]["score"],
            "dominant_component": row["controller_route"]["dominant_component"],
            "next_phase": row["controller_route"]["next_phase"],
            "target_rebase_required": row["target_rebase"]["measurement_geometry_rebase_required"],
            "independent_axis_thresholds_used": False,
            "score_claim": False,
            "promotion_eligible": False,
            "actuation": "NONE",
        }
        for row in same_object_score_rows
    )

    output = {
        "schema": SCHEMA,
        "feedback_sha256": feedback_sha,
        "binding": _binding_dict(binding),
        "binding_fingerprint_sha256": binding_fingerprint,
        "population_scope": {
            "source_pair_ids": list(binding.source_pair_ids),
            "source_pair_count": len(binding.source_pair_ids),
            "population_pair_count": binding.population_pair_count,
            "population_manifest_sha256": binding.population_manifest_sha256,
            "evidence_scope": (
                "n2_feedback_plus_exact_n600_same_object_macos_cpu_advisory"
                if same_object_score_rows
                else "n2_subset_of_bound_n600_population_macos_cpu_advisory"
            ),
            "membership_proof_in_g18": False,
            "generalization_to_population": False,
        },
        "truth": {
            "research_only": True,
            "n2_only": not same_object_score_rows,
            "n600_evaluation": bool(same_object_score_rows),
            "n600_evaluation_axis": (
                "[macOS-CPU frozen-scorer advisory]" if same_object_score_rows else None
            ),
            "score_claim": False,
            "candidate_claim": False,
            "originality_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "pointer_mutation_performed": False,
            "scorer_or_live_dispatch_performed": False,
            "ledger_append_performed": False,
            "actuation": "NONE",
        },
        "action_semantics_contract": {
            "closed_actions": list(ACTION_SEMANTICS),
            "closed_reservoirs": list(RESERVOIRS),
            "costate_granularities": list(COSTATE_GRANULARITIES),
            "costate_factor_axes": list(COSTATE_FACTOR_AXES),
            "semantic_labels_proven_by_g18": False,
            "effect_calibration_does_not_upgrade_semantic_claim": True,
            "whole_object_remeasurement_required_for_composition": True,
            "additive_independence_assumed": False,
        },
        "selected_evaluator_solution": selected_solution,
        "measurement_identity_domains": {
            "artifact_identity": "full occurrence object including exact measurement receipt custody",
            "semantic_control_identity": "realized measurement with proof receipt identity projected out",
            "proof_dependency_set": "all exact measurement receipt hashes supporting one semantic control identity",
            "same_semantic_measurement_may_have_multiple_proof_receipts": True,
            "semantic_aliasing_forbidden": True,
            "rows": view["measurement_identity_domains"],
        },
        "solution_representation_policy": {
            "controller_object": "selected_evaluator_measured_solution_not_problem_inventory",
            "score_objective_uses_final_archive_zip_bytes": True,
            "postcompression_mdl_is_archive_nbytes": True,
            "raw_tensor_size_is_never_an_objective": True,
            "first_feasible_solution_is_never_the_selection_rule": True,
            "factorability_is_priced_by_realized_final_archive_bytes_not_label_count": True,
            "generic_decoder_rate_cost_in_archive": 0,
            "generic_decoder_guardrail_costs": [
                "shipped_decoder_source_nbytes",
                "measured_runtime_seconds",
                "deterministic_portability_receipt",
            ],
            "generic_active_decode_operations": list(GENERIC_DECODER_OPERATIONS),
            "generic_machinery_placement": "inflate_py_or_inflate_sh",
            "counted_video_specific_payload_classes": [
                "selector",
                "parameter",
                "atom",
                "latent",
                "exception",
                "target_dependent_branch",
            ],
            "placement_evidence_hashes_dereferenced_by_g19": False,
            "whole_object_receiver_remeasurement_required": True,
        },
        "full_lattice_teacher_endpoint": teacher_payload,
        "rate_distortion_homotopy": {
            "teacher_role": "feasible_endpoint_oracle_not_automatic_selected_solution",
            "selection_objective": "global_score_with_final_postcompression_archive_bytes",
            "rows": homotopy_rows,
        },
        "hierarchical_costate_harvest": {
            "granularities": list(COSTATE_GRANULARITIES),
            "factor_axes": list(COSTATE_FACTOR_AXES),
            "finite_action_costates_not_derivatives": True,
            "same_effect_hierarchy_rollup_is_nonadditive": True,
            "rows": costate_rows,
        },
        "whole_object_representation_placements": {
            "count": len(placement_rows),
            "rows": placement_rows,
            "additive_double_counting_forbidden": True,
            "controller_confidence_update_performed": False,
            "receiver_equality_closed_count": sum(row["receiver_equality_handoff_unblocked"] for row in placement_rows),
        },
        "population_global_representation_recodes": {
            "count": len(population_global_rows),
            "rows": population_global_rows,
            "conditional_best_archive_nbytes": (
                min(
                    row["conditional_selected_solution"]["postcompression_mdl_nbytes"] for row in population_global_rows
                )
                if population_global_rows
                else None
            ),
            "rate_floor_is_conditional_on_public_video_output_closure": True,
            "controller_confidence_update_performed": False,
            "additive_double_counting_forbidden": True,
        },
        "same_object_score_measurements": {
            "count": len(same_object_score_rows),
            "rows": same_object_score_rows,
            "component_routing_eligible_count": sum(
                row["telemetry_admission"]["component_routing_eligible"] for row in same_object_score_rows
            ),
            "contest_score_endpoint_eligible_count": 0,
            "independent_axis_thresholds_forbidden": True,
            "controller_confidence_update_performed": False,
        },
        "composition_sign_reversal_protection": {
            "interaction_count": len(view["interaction_effects"]),
            "sign_reversal_count": len(sign_reversal_rows),
            "locally_harmful_compositionally_beneficial_count": sum(
                row["classification"]
                == "LOCALLY_HARMFUL_BUT_COMPOSITIONALLY_BENEFICIAL"
                for row in sign_reversal_rows
            ),
            "locally_beneficial_compositionally_harmful_count": sum(
                row["classification"]
                == "LOCALLY_BENEFICIAL_BUT_COMPOSITIONALLY_HARMFUL"
                for row in sign_reversal_rows
            ),
            "local_marginal_filter_may_prune_sign_reversal_rows": False,
            "paired_whole_object_endpoint_enumeration_required": True,
            "rows": sign_reversal_rows,
        },
        "terminal_controller_contract": {
            "consumer_api": (
                "tac.witness_control.taskspace_receding_horizon_controller_v1."
                "select_receding_horizon_action_v1"
            ),
            "consumer_schema": TERMINAL_CONTROLLER_SCHEMA,
            "required_action_families": list(TERMINAL_ACTION_FAMILIES),
            "required_action_scales": list(TERMINAL_ACTION_SCALES),
            "selection_rule": (
                "branch_and_bound_globally_lowest_exact_public_whole_object_endpoint_then_"
                "invalidate_all_other_marginals"
            ),
            "exhaustive_endpoint_materialization_required": False,
            "unmeasured_branch_requires_exact_optimistic_score_lower_bound": True,
            "same_authority_mode_and_axis_as_base_required": True,
            "identity_hashes_without_reopenable_proof_receipts_forbidden": True,
            "controller_authorizes_pointer_promotion": False,
            "selected_archive_requires_paired_contest_cpu_cuda_replay": True,
            "present_evaluator_cell_equivalence_alone_is_sufficient": False,
            "functional_quotient_identity": "evaluator_cells_x_continuation_equivalence",
            "independent_seg_pose_rate_thresholds_forbidden": True,
            "sign_reversal_hyperedges_must_be_enumerated_atomically": True,
            "partial_action_universe_selection_forbidden": True,
            "call_performed": False,
            "blocker": "COMPLETE_EXACT_PUBLIC_ENDPOINT_UNIVERSE_OWED",
        },
        "calibrations": calibrations,
        "unpredicted_effect_ids": sorted(set(effects) - {row.effect_id for row in ordered_predictions}),
        "controller_state": state,
        "controller_state_sha256": _sha256(_canonical_json(state)),
        "acquisition": {
            "api": "tac.witness_control.control_alphabet.powerplay_acquisition",
            "api_invoked": bool(lam_field),
            "unit_cost_semantics": "one already-measured indivisible effect observation",
            "additive_independence_assumed": False,
            "rows": acquisition_rows,
        },
        "telemetry": {
            "consumer_api": "tac.witness_control.telemetry_binding.parse_log_lines",
            "compatible_stage_rows": True,
            "write_performed": False,
            "rows": telemetry_rows,
        },
        "blocked_handoffs": _blocked_handoffs(
            ordered_predictions,
            teacher=teacher_payload,
            placements=placement_rows,
            population_global_recodes=population_global_rows,
            same_object_scores=same_object_score_rows,
        ),
        "historical_hint_policy": {
            "v19c_and_fixed_ddm_additive_lists": "HINT_ONLY_NOT_INGESTED",
            "full_lattice_teacher_without_current_receiver_replay": "HINT_ONLY_ZERO_CONTROLLER_INFLUENCE",
            "confidence_sources": "only content-addressed exact-base G18 prediction/realization pairs",
        },
    }
    round_trip = json.loads(bridge_receipt_bytes(output).decode("ascii"))
    if round_trip != output:
        raise TaskspaceInteractionCostateBridgeError("bridge canonical round trip changed output")
    return output


__all__ = [
    "ACTION_SEMANTICS",
    "COSTATE_FACTOR_AXES",
    "COSTATE_GRANULARITIES",
    "GENERIC_DECODER_OPERATIONS",
    "RESERVOIRS",
    "SCHEMA",
    "STATE_SCHEMA",
    "ActionEffectPredictionV1",
    "CostateFactorizationV1",
    "CurrentExactReceiverReplayV1",
    "DecoderPlacementV1",
    "FeedbackExactBaseBindingV1",
    "FullLatticeTeacherEndpointV1",
    "FullN600ReceiverEqualityClosureV1",
    "ReceiverEqualityAdmissionControlV1",
    "TaskspaceInteractionCostateBridgeError",
    "bridge_receipt_bytes",
    "build_taskspace_interaction_costate_bridge_v1",
]
