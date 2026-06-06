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
        runtime_binding_blocker=None,
        bounded_training_blocker="snerv_lf_conditioned_hf_bounded_training_binding_missing",
        decoder_name="decode_lf_conditioned_hf_residual_payload",
        decoder=decode_lf_conditioned_hf_residual_payload,
    ),
    _FamilySpec(
        solution_family="joint_lf_hf_factorized_codebook",
        proof_schema="snerv_joint_lf_hf_factorized_codebook_receiver_proof.v1",
        cli_flag="--joint-codebook-receiver-payload-proof",
        false_authority_blocker="snerv_joint_lf_hf_factorized_codebook_false_authority",
        runtime_binding_blocker=None,
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
