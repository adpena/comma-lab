# SPDX-License-Identifier: MIT
"""Receiver-runtime binding proof for SNeRV LF/HF payload families.

The payload proof builders demonstrate that each family can emit byte-charged
receiver payloads.  This module is the next, deliberately small, handoff: it
re-opens those exact payload bytes by path/SHA and decodes them through the
family receiver module that an inflate/runtime path would call.

The emitted artifact is false-authority.  It can clear receiver-runtime binding
blockers in planning queues, but it is not a score, source-forward, replay, or
promotion claim.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from tac.analysis.snerv_source_forward_proof import (
    DROP_OUTPUT2_USE_MFU_HFR_TUB_BASIS,
    SNERV_OUTPUT2_BOUNDARY_VERDICT_SCHEMA,
    SOURCE_FORWARD_SURFACES,
    SOURCE_IDENTICAL,
    validate_snerv_output2_boundary_verdict,
)
from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY
from tac.substrates.snerv_inverse_steg_carrier.joint_lf_hf_codebook import (
    decode_joint_lf_hf_factorized_codebook_payload,
)
from tac.substrates.snerv_inverse_steg_carrier.lf_conditioned_hf_residual import (
    decode_lf_conditioned_hf_residual_payload,
)
from tac.substrates.snerv_inverse_steg_carrier.lf_latent_hyperprior import (
    decode_lf_latent_hyperprior_payload,
)
from tac.substrates.snerv_inverse_steg_carrier.lf_super_resolution_tiny_anchor import (
    decode_lf_super_resolution_tiny_anchor_payload,
)
from tac.substrates.snerv_inverse_steg_carrier.spectral_band_allocator import (
    decode_score_tethered_spectral_band_allocator_payload,
)
from tac.substrates.snerv_inverse_steg_carrier.temporal_lf_predictor import (
    decode_temporal_lf_predictor_payload,
)

SCHEMA = "snerv_lf_hf_runtime_binding_proof.v1"
ROW_SCHEMA = "snerv_lf_hf_runtime_binding_row.v1"
NATIVE_TUB_LF_HF_OUTPUT2_BINDING_SCHEMA = (
    "snerv_native_tub_lf_hf_output2_runtime_binding.v1"
)
AXIS_TAG = "[receiver-runtime-binding:false-authority]"

QUEUE_FALSE_AUTHORITY = {
    **FALSE_AUTHORITY,
    "score_claim": False,
    "score_claim_valid": False,
    "frontier_score_claim": False,
    "promotion_eligible": False,
    "promotable": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "production_hardened_claim": False,
    "dispatch_attempted": False,
    "gpu_launched": False,
    "local_mlx_long_training_allowed": False,
    "dispatch_allowed": False,
    "exact_or_full_video_cuda_allowed": False,
}


class SnervLfHfRuntimeBindingError(ValueError):
    """Raised when a runtime-binding proof cannot be built."""


@dataclass(frozen=True)
class _FamilySpec:
    solution_family: str
    proof_schema: str
    cli_flag: str
    false_authority_blocker: str
    runtime_binding_blocker: str | None
    bounded_training_blocker: str
    decoder_name: str
    decoder: Callable[[bytes], np.ndarray]


FAMILY_SPECS: tuple[_FamilySpec, ...] = (
    _FamilySpec(
        solution_family="lf_conditioned_hf_residual_generator",
        proof_schema="snerv_lf_conditioned_hf_residual_receiver_proof.v1",
        cli_flag="--hf-residual-receiver-payload-proof",
        false_authority_blocker="snerv_lf_conditioned_hf_residual_payload_false_authority",
        runtime_binding_blocker=(
            "snerv_lf_conditioned_hf_residual_receiver_runtime_binding_missing"
        ),
        bounded_training_blocker="snerv_lf_conditioned_hf_bounded_training_binding_missing",
        decoder_name="decode_lf_conditioned_hf_residual_payload",
        decoder=decode_lf_conditioned_hf_residual_payload,
    ),
    _FamilySpec(
        solution_family="joint_lf_hf_factorized_codebook",
        proof_schema="snerv_joint_lf_hf_factorized_codebook_receiver_proof.v1",
        cli_flag="--joint-codebook-receiver-payload-proof",
        false_authority_blocker="snerv_joint_lf_hf_factorized_codebook_false_authority",
        runtime_binding_blocker=(
            "snerv_joint_lf_hf_factorized_codebook_receiver_runtime_binding_missing"
        ),
        bounded_training_blocker="snerv_joint_lf_hf_bounded_training_binding_missing",
        decoder_name="decode_joint_lf_hf_factorized_codebook_payload",
        decoder=decode_joint_lf_hf_factorized_codebook_payload,
    ),
    _FamilySpec(
        solution_family="temporal_lf_predictor_gate",
        proof_schema="snerv_temporal_lf_predictor_receiver_proof.v1",
        cli_flag="--temporal-lf-predictor-receiver-payload-proof",
        false_authority_blocker="snerv_temporal_lf_predictor_payload_false_authority",
        runtime_binding_blocker="snerv_temporal_lf_predictor_receiver_runtime_binding_missing",
        bounded_training_blocker="snerv_temporal_lf_predictor_bounded_training_binding_missing",
        decoder_name="decode_temporal_lf_predictor_payload",
        decoder=decode_temporal_lf_predictor_payload,
    ),
    _FamilySpec(
        solution_family="lf_super_resolution_from_tiny_anchor",
        proof_schema="snerv_lf_super_resolution_tiny_anchor_receiver_proof.v1",
        cli_flag="--lf-super-resolution-receiver-payload-proof",
        false_authority_blocker="snerv_lf_super_resolution_tiny_anchor_payload_false_authority",
        runtime_binding_blocker="snerv_lf_super_resolution_receiver_runtime_binding_missing",
        bounded_training_blocker="snerv_lf_super_resolution_bounded_training_binding_missing",
        decoder_name="decode_lf_super_resolution_tiny_anchor_payload",
        decoder=decode_lf_super_resolution_tiny_anchor_payload,
    ),
    _FamilySpec(
        solution_family="score_tethered_spectral_band_allocator",
        proof_schema="snerv_score_tethered_spectral_band_allocator_receiver_proof.v1",
        cli_flag="--spectral-band-allocator-receiver-payload-proof",
        false_authority_blocker="snerv_score_tethered_spectral_band_allocator_false_authority",
        runtime_binding_blocker="snerv_score_tethered_lf_hf_band_allocator_runtime_binding_missing",
        bounded_training_blocker="snerv_score_tethered_lf_hf_band_allocator_bounded_training_binding_missing",
        decoder_name="decode_score_tethered_spectral_band_allocator_payload",
        decoder=decode_score_tethered_spectral_band_allocator_payload,
    ),
    _FamilySpec(
        solution_family="entropy_modeled_lf_latent_hyperprior",
        proof_schema="snerv_lf_latent_hyperprior_receiver_proof.v1",
        cli_flag="--lf-latent-hyperprior-receiver-payload-proof",
        false_authority_blocker="snerv_lf_latent_hyperprior_payload_false_authority",
        runtime_binding_blocker="snerv_lf_latent_hyperprior_runtime_binding_missing",
        bounded_training_blocker="snerv_lf_latent_hyperprior_bounded_training_binding_missing",
        decoder_name="decode_lf_latent_hyperprior_payload",
        decoder=decode_lf_latent_hyperprior_payload,
    ),
)

_SPEC_BY_SCHEMA = {spec.proof_schema: spec for spec in FAMILY_SPECS}
_SPEC_BY_FAMILY = {spec.solution_family: spec for spec in FAMILY_SPECS}


def build_snerv_lf_hf_runtime_binding_proof(
    receiver_payload_proofs: Sequence[Mapping[str, Any]],
    *,
    generated_utc: str | None = None,
) -> dict[str, Any]:
    """Build a false-authority runtime-binding proof from payload proof JSON."""

    generated = generated_utc or datetime.now(UTC).isoformat()
    rows = [
        _runtime_binding_row(proof)
        for proof in receiver_payload_proofs
        if isinstance(proof, Mapping) and proof.get("schema") in _SPEC_BY_SCHEMA
    ]
    blockers: list[str] = []
    if not rows:
        blockers.append("snerv_lf_hf_runtime_binding_payload_proofs_missing")
    for row in rows:
        blockers.extend(
            blocker
            for blocker in row.get("blockers") or ()
            if blocker != row.get("false_authority_blocker")
        )
    closed = _dedupe(
        blocker
        for row in rows
        for blocker in row.get("closed_campaign_blockers") or ()
    )
    return {
        "schema": SCHEMA,
        "generated_utc": generated,
        "axis_tag": AXIS_TAG,
        "runtime_contract": (
            "exact payload bytes are reopened, hash-checked, and decoded "
            "through the family NumPy receiver module"
        ),
        "runtime_binding_row_count": len(rows),
        "runtime_binding_rows": rows,
        "runtime_bound_solution_families": [
            row["solution_family"] for row in rows if row.get("runtime_binding_proven")
        ],
        "closed_campaign_blockers": closed,
        "blockers": _dedupe(blockers),
        **QUEUE_FALSE_AUTHORITY,
    }


def build_snerv_native_tub_lf_hf_output2_runtime_binding(
    native_export_artifact: Mapping[str, Any] | None,
    *,
    lf_hf_solution_family: str | None = None,
    generated_utc: str | None = None,
) -> dict[str, Any]:
    """Bind native SNeRV TUB output_2 through exported receiver bytes.

    This closes only the native output_2 migration boundary. It deliberately
    keeps score, promotion, exact-eval, and source-forward authority false.
    """

    generated = generated_utc or datetime.now(UTC).isoformat()
    artifact = native_export_artifact if isinstance(native_export_artifact, Mapping) else {}
    maps = _shallow_artifact_maps(artifact)
    selected_family = str(
        lf_hf_solution_family
        or artifact.get("snerv_lf_hf_solution_family")
        or artifact.get("lf_hf_solution_family")
        or ""
    ).strip()
    tub_binding = _first_mapping(maps, "snerv_official_tub_source_fixture_binding")
    output2_storage = _first_mapping(maps, "official_tub_output2_storage")

    output2_payload_export_bound = _any_true(
        maps,
        "official_tub_output2_payload_export_bound",
        "official_tub_output2_payload_stored",
    )
    output2_payload_source_available = _any_true(
        maps,
        "official_tub_output2_payload_source_available",
    ) or bool(output2_storage.get("source_payload_present") is True)
    output2_receiver_fusion_from_payload = _any_true(
        maps,
        "official_tub_output2_receiver_fusion_from_payload",
    ) or bool(output2_storage.get("receiver_executes_output2_fusion_from_payload") is True)
    output2_receiver_executed = _any_true(
        maps,
        "official_tub_output2_receiver_executed",
        "official_tub_output2_fusion_executed",
    )
    output2_receiver_consumes = _any_true(
        maps,
        "receiver_frame_decode_consumes_output2",
    ) or bool(output2_storage.get("receiver_frame_decode_consumes_output2") is True)
    output2_shape_matches = _any_true(
        maps,
        "official_tub_output2_receiver_output2_frame_shape_match",
    ) or bool(output2_storage.get("receiver_output2_frame_shape_match") is True)
    output2_receiver_frame_bound = _any_true(
        maps,
        "official_tub_output2_receiver_frame_bound",
    ) or bool(
        output2_payload_export_bound
        and output2_receiver_fusion_from_payload
        and output2_receiver_executed
        and output2_receiver_consumes
        and output2_shape_matches
    )

    tub_source_fixture_bound = _any_true(
        maps,
        "snerv_official_tub_source_fixture_replay_bound",
        "snerv_official_tub_source_forward_fixture_bound",
        "source_fixture_replay_bound",
    ) or bool(tub_binding.get("source_fixture_replay_bound") is True)
    tub_source_fixture_passed = _any_true(
        maps,
        "snerv_official_tub_source_fixture_replay_passed",
        "official_tub_temporal_encoder_output2_source_fixture_replay_passed",
    ) or bool(
        tub_binding.get("official_tub_temporal_encoder_output2_source_fixture_replay_passed")
        is True
    )
    receiver_payload_bound = _any_true(
        maps,
        "snerv_official_mfu_hfr_tub_receiver_payload_bound",
        "official_mfu_hfr_tub_receiver_payload_bound",
        "official_receiver_payload_bound",
    )
    frame_producing_export = _any_true(
        maps,
        "snerv_official_mfu_hfr_tub_frame_producing_export",
        "official_mfu_hfr_tub_frame_producing_export",
        "selected_packet_frame_producing_official_export",
        "frame_producing_official_export",
    )
    receiver_frame_replay_passed = _any_true(
        maps,
        "selected_packet_receiver_payload_frame_replay_passed",
        "receiver_payload_frame_replay_passed",
    )

    blockers: list[str] = []
    if not selected_family:
        blockers.append("snerv_native_lf_hf_solution_family_missing")
    if not tub_source_fixture_bound:
        blockers.append("snerv_native_tub_source_fixture_not_bound")
    if not tub_source_fixture_passed:
        blockers.append("snerv_native_tub_source_fixture_replay_not_passed")
    if not receiver_payload_bound:
        blockers.append("snerv_native_lf_hf_receiver_payload_not_bound")
    if not frame_producing_export:
        blockers.append("snerv_native_lf_hf_frame_producing_export_missing")
    if not receiver_frame_replay_passed:
        blockers.append("snerv_native_lf_hf_receiver_payload_frame_replay_missing")
    if not output2_payload_export_bound:
        blockers.append("snerv_native_output2_payload_not_export_bound")
    if not output2_payload_source_available:
        blockers.append("snerv_native_output2_source_payload_missing")
    if not output2_receiver_fusion_from_payload:
        blockers.append("snerv_native_output2_fusion_not_payload_bound")
    if not output2_receiver_executed:
        blockers.append("snerv_native_output2_fusion_not_executed")
    if not output2_receiver_consumes:
        blockers.append("snerv_native_output2_not_consumed_by_receiver")
    if not output2_shape_matches:
        blockers.append("snerv_native_output2_frame_shape_mismatch")
    if not output2_receiver_frame_bound:
        blockers.append("snerv_native_output2_receiver_frame_not_bound")

    source_identical = not _dedupe(blockers)
    archive_storage = {
        **dict(output2_storage),
        "section": output2_storage.get("section", "decoder_payload.output_2"),
        "stored": output2_payload_export_bound,
        "source_payload_present": output2_payload_source_available,
        "receiver_executes_output2_fusion_from_payload": (
            output2_receiver_fusion_from_payload
        ),
        "receiver_frame_decode_consumes_output2": output2_receiver_consumes,
        "receiver_output2_frame_shape_match": output2_shape_matches,
        "shape_adapter_forbidden": True,
        "shape_adapter_applied": False,
    }
    boundary = {
        "schema": SNERV_OUTPUT2_BOUNDARY_VERDICT_SCHEMA,
        "verdict": (
            SOURCE_IDENTICAL
            if source_identical
            else DROP_OUTPUT2_USE_MFU_HFR_TUB_BASIS
        ),
        "passed": source_identical,
        "has_output2_by_surface": dict.fromkeys(
            SOURCE_FORWARD_SURFACES,
            source_identical,
        ),
        "output2_shapes_by_surface": {
            surface: _output2_shape(artifact) for surface in SOURCE_FORWARD_SURFACES
        },
        "archive_tub_output2_storage": archive_storage,
        "minimal_causal_basis_recommendation": (
            ["keep_output2_source_forward_bound"]
            if source_identical
            else [
                "lf_carrier",
                "hf_carrier",
                "mfu_state",
                "hfr_state",
                "tub_temporal_state",
                "pair_adapter",
                "derive_output_2",
            ]
        ),
        "blockers": [],
        "required_next_step": (
            "output2_boundary_closed"
            if source_identical
            else "store_lf_hf_mfu_hfr_tub_pair_adapter_and_derive_output2"
        ),
        **QUEUE_FALSE_AUTHORITY,
    }
    boundary_status = validate_snerv_output2_boundary_verdict(boundary)
    if boundary_status["passed"] is not True:
        blockers.extend(
            f"snerv_native_output2_boundary:{blocker}"
            for blocker in boundary_status["blockers"]
        )
    proven = bool(source_identical and boundary_status["passed"] is True)
    return {
        "schema": NATIVE_TUB_LF_HF_OUTPUT2_BINDING_SCHEMA,
        "generated_utc": generated,
        "axis_tag": AXIS_TAG,
        "lf_hf_solution_family": selected_family or None,
        "native_export_artifact_schema": artifact.get("schema"),
        "packet_path": artifact.get("packet_path"),
        "packet_sha256": artifact.get("packet_sha256"),
        "archive_path": artifact.get("archive_path"),
        "archive_sha256": artifact.get("archive_sha256"),
        "tub_source_fixture_bound": tub_source_fixture_bound,
        "tub_source_fixture_passed": tub_source_fixture_passed,
        "lf_hf_receiver_payload_bound": receiver_payload_bound,
        "lf_hf_frame_producing_export": frame_producing_export,
        "lf_hf_receiver_payload_frame_replay_passed": receiver_frame_replay_passed,
        "output2_payload_export_bound": output2_payload_export_bound,
        "output2_receiver_frame_bound": output2_receiver_frame_bound,
        "output2_boundary_verdict": boundary_status[
            "normalized_output2_boundary_verdict"
        ],
        "output2_source_identical": proven,
        "runtime_binding_proven": proven,
        "source_forward_replay_bound": False,
        "source_forward_replay_verified": False,
        "source_forward_replay_authority": False,
        "full_stack_source_forward_replay_proven": False,
        "official_authority": False,
        "closed_campaign_blockers": (
            [
                "snerv_native_export_output2_boundary_missing",
                "snerv_native_export_output2_source_identical_missing",
                "migration_required_before_runner_execution",
            ]
            if proven
            else []
        ),
        "blockers": _dedupe(blockers),
        **QUEUE_FALSE_AUTHORITY,
    }


def proof_cli_flag_for_solution_family(solution_family: str) -> str | None:
    """Return the payload-proof CLI flag for a solution family."""

    spec = _SPEC_BY_FAMILY.get(str(solution_family))
    return None if spec is None else spec.cli_flag


def runtime_binding_blocker_for_solution_family(solution_family: str) -> str | None:
    """Return the runtime-binding blocker closed by a family proof."""

    spec = _SPEC_BY_FAMILY.get(str(solution_family))
    return None if spec is None else spec.runtime_binding_blocker


def bounded_training_blocker_for_solution_family(solution_family: str) -> str | None:
    """Return the post-runtime bounded-training blocker for a family."""

    spec = _SPEC_BY_FAMILY.get(str(solution_family))
    return None if spec is None else spec.bounded_training_blocker


def _runtime_binding_row(proof: Mapping[str, Any]) -> dict[str, Any]:
    spec = _SPEC_BY_SCHEMA[str(proof["schema"])]
    payload_path_text = str(proof.get("payload_path") or "").strip()
    payload_path = Path(payload_path_text).expanduser()
    payload_bytes_expected = _positive_int(proof.get("payload_bytes"))
    payload_sha_expected = str(proof.get("payload_sha256") or "").strip()
    blockers = [spec.false_authority_blocker]
    decoded_summary: dict[str, Any] | None = None
    actual_payload_bytes: int | None = None
    actual_payload_sha: str | None = None
    if not payload_path_text:
        blockers.append("snerv_lf_hf_runtime_binding_payload_path_missing")
    elif not payload_path.is_file():
        blockers.append("snerv_lf_hf_runtime_binding_payload_path_not_file")
    else:
        payload = payload_path.read_bytes()
        actual_payload_bytes = len(payload)
        actual_payload_sha = _sha256(payload)
        if payload_bytes_expected is None:
            blockers.append("snerv_lf_hf_runtime_binding_payload_bytes_missing")
        elif payload_bytes_expected != actual_payload_bytes:
            blockers.append("snerv_lf_hf_runtime_binding_payload_bytes_mismatch")
        if not payload_sha_expected:
            blockers.append("snerv_lf_hf_runtime_binding_payload_sha256_missing")
        elif payload_sha_expected != actual_payload_sha:
            blockers.append("snerv_lf_hf_runtime_binding_payload_sha256_mismatch")
        try:
            decoded = spec.decoder(payload)
        except Exception as exc:
            blockers.append(
                "snerv_lf_hf_runtime_binding_receiver_decode_failed:"
                f"{type(exc).__name__}"
            )
        else:
            decoded_summary = _decoded_summary(decoded)
            if decoded_summary["all_finite"] is not True:
                blockers.append("snerv_lf_hf_runtime_binding_decoded_nonfinite")
            if int(decoded_summary["element_count"]) <= 0:
                blockers.append("snerv_lf_hf_runtime_binding_decoded_empty")
    structural_blockers = [
        blocker for blocker in blockers if blocker != spec.false_authority_blocker
    ]
    runtime_binding_proven = not structural_blockers
    closed = (
        [spec.runtime_binding_blocker]
        if runtime_binding_proven and spec.runtime_binding_blocker is not None
        else []
    )
    return {
        "schema": ROW_SCHEMA,
        "solution_family": spec.solution_family,
        "source_payload_proof_schema": proof.get("schema"),
        "source_payload_proof_path": (
            proof.get("_source_path") or proof.get("source_path") or proof.get("report_path")
        ),
        "source_payload_proof_sha256": proof.get("_source_sha256"),
        "payload_path": payload_path.as_posix(),
        "payload_bytes_expected": payload_bytes_expected,
        "payload_bytes_actual": actual_payload_bytes,
        "payload_sha256_expected": payload_sha_expected or None,
        "payload_sha256_actual": actual_payload_sha,
        "payload_pair_indices": proof.get("pair_indices"),
        "payload_sample_shape": proof.get("sample_shape_b2chw")
        or proof.get("lf_shape_b2chw"),
        "receiver_decoder": spec.decoder_name,
        "decoded_summary": decoded_summary,
        "runtime_binding_proven": runtime_binding_proven,
        "closed_campaign_blockers": closed,
        "false_authority_blocker": spec.false_authority_blocker,
        "blockers": _dedupe(blockers),
        **QUEUE_FALSE_AUTHORITY,
    }


def _decoded_summary(decoded: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(decoded)
    finite = np.isfinite(arr)
    arr64 = arr.astype(np.float64, copy=False)
    return {
        "decoded_shape": [int(value) for value in arr.shape],
        "decoded_dtype": str(arr.dtype),
        "element_count": int(arr.size),
        "all_finite": bool(np.all(finite)),
        "min": None if arr.size == 0 else float(np.min(arr64)),
        "max": None if arr.size == 0 else float(np.max(arr64)),
        "mean": None if arr.size == 0 else float(np.mean(arr64)),
        "std": None if arr.size == 0 else float(np.std(arr64)),
    }


def _shallow_artifact_maps(root: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    maps: list[Mapping[str, Any]] = [root]
    for key in (
        "artifact",
        "official_primitive_binding",
        "selected_official_authority",
        "official_receiver_tensor_map_custody",
        "official_receiver_payload_frame_replay",
    ):
        child = root.get(key)
        if isinstance(child, Mapping):
            maps.append(child)
    selected = root.get("selected_official_authority")
    if isinstance(selected, Mapping):
        replay = selected.get("official_receiver_payload_frame_replay")
        if isinstance(replay, Mapping):
            maps.append(replay)
    return maps


def _first_mapping(maps: Sequence[Mapping[str, Any]], key: str) -> Mapping[str, Any]:
    for row in maps:
        child = row.get(key)
        if isinstance(child, Mapping):
            return child
    return {}


def _any_true(maps: Sequence[Mapping[str, Any]], *keys: str) -> bool:
    for row in maps:
        for key in keys:
            if row.get(key) is True:
                return True
    return False


def _output2_shape(artifact: Mapping[str, Any]) -> list[int]:
    raw = artifact.get("output2_decoder_output_shape")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raw = artifact.get("official_tub_output2_receiver_output_tensor_shape")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return []
    try:
        return [int(value) for value in raw]
    except (TypeError, ValueError):
        return []


def _positive_int(value: Any) -> int | None:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _dedupe(values: Sequence[Any]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if not text or text in seen:
            continue
        out.append(text)
        seen.add(text)
    return out


def load_json_with_source_identity(path: str | Path) -> dict[str, Any]:
    """Load JSON and attach source path/sha for reproducible queue rebuilds."""

    resolved = Path(path).expanduser().resolve(strict=False)
    data = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise SnervLfHfRuntimeBindingError(
            f"expected JSON object at {resolved.as_posix()}"
        )
    data["_source_path"] = resolved.as_posix()
    data["_source_sha256"] = hashlib.sha256(
        resolved.read_bytes()
    ).hexdigest()
    return data
