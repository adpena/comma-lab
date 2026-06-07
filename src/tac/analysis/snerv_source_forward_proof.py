# SPDX-License-Identifier: MIT
"""SNeRV source-forward proof action rows.

The long-run gate needs a numerical proof that bytes are causal, not metadata
that merely says a section was present.  This module is intentionally small and
backend-agnostic: producers supply real tensors from official Torch, Pact MLX,
archive parse-back, and NumPy receiver surfaces; the contract validates the
fixed tensor/scorer comparisons and the destructive payload bit-flip.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from hashlib import sha256
from typing import Any

import numpy as np

from tac.optimization.proxy_candidate_contract import PROXY_FALSE_AUTHORITY_FIELDS

SNERV_SOURCE_FORWARD_PROOF_ACTION_EFFECT_SCHEMA = (
    "snerv_source_forward_proof_action_effect.v1"
)
SNERV_PAYLOAD_BITFLIP_FALSIFICATION_SCHEMA = (
    "snerv_payload_bitflip_falsification.v1"
)
SNERV_PAYLOAD_BITFLIP_FALSIFICATION_MATRIX_SCHEMA = (
    "snerv_payload_bitflip_falsification_matrix.v1"
)
SNERV_PAYLOAD_BITFLIP_REQUIRED_SECTIONS: tuple[str, ...] = (
    "metadata_payload",
    "lf_payload",
    "decoder_payload",
    "step_map_packet",
)
SNERV_OUTPUT2_BOUNDARY_VERDICT_SCHEMA = "snerv_output2_boundary_verdict.v1"
SOURCE_IDENTICAL = "SOURCE_IDENTICAL"
REPARAMETERIZED_RENAME_REQUIRED = "REPARAMETERIZED_RENAME_REQUIRED"
DROP_OUTPUT2_USE_MFU_HFR_TUB_BASIS = "DROP_OUTPUT2_USE_MFU_HFR_TUB_BASIS"
SOURCE_FORWARD_OUTPUT2_VERDICTS: tuple[str, ...] = (
    SOURCE_IDENTICAL,
    REPARAMETERIZED_RENAME_REQUIRED,
    DROP_OUTPUT2_USE_MFU_HFR_TUB_BASIS,
)

SOURCE_FORWARD_SURFACES: tuple[str, ...] = (
    "official_torch",
    "pact_mlx",
    "archive_parseback",
    "numpy_receiver",
)
SOURCE_FORWARD_REFERENCE_SURFACE = "official_torch"
SOURCE_FORWARD_COMPARISON_SURFACES: tuple[str, ...] = (
    "pact_mlx",
    "archive_parseback",
    "numpy_receiver",
)
SOURCE_FORWARD_TENSOR_NAMES: tuple[str, ...] = (
    "coord_time_embedding",
    "mfu_in",
    "mfu_out",
    "hfr_in",
    "hfr_out",
    "tub_in",
    "tub_out",
    "output_2",
    "rgb_pair_float",
    "rgb_pair_uint8",
    "segnet_input",
    "posenet_input",
    "segnet_logits",
    "segnet_argmax",
    "posenet_output",
)
SOURCE_FORWARD_SCORER_FIELDS: tuple[str, ...] = ("d_seg", "d_pose")
SOURCE_FORWARD_SURFACE_PROVENANCE_SCHEMA = "snerv_source_forward_surface_provenance.v1"
SOURCE_FORWARD_PROVENANCE_REQUIRED_FIELDS: tuple[str, ...] = (
    "schema",
    "surface",
    "producer",
    "backend",
    "pair_ids",
    "tensor_capture_authority",
    "scorer_capture_authority",
)
SOURCE_FORWARD_ARCHIVE_BOUND_SURFACES: tuple[str, ...] = (
    "archive_parseback",
    "numpy_receiver",
)
SOURCE_FORWARD_FORBIDDEN_PROVENANCE_TOKENS: tuple[str, ...] = (
    "fixture",
    "synthetic",
    "mock",
    "metadata",
    "sidecar",
    "proxy",
    "placeholder",
    "receiver_bound",
    "not_upstream",
)
SOURCE_FORWARD_OFFICIAL_TORCH_ALLOWED_TRAINED_LINEAGES: tuple[str, ...] = (
    "official_trained_checkpoint_state_dict",
    "checkpoint_export_official_trained_checkpoint_state_dict",
    "official_trained_checkpoint_state_dict_slice",
)
SOURCE_FORWARD_OFFICIAL_TORCH_ALLOWED_CONFIG_LINEAGES: tuple[str, ...] = (
    "official_trained_run_config",
    "checkpoint_export_official_trained_run_config",
    "official_submission_config",
)
SOURCE_FORWARD_OPTIONAL_PROVENANCE_FIELDS: tuple[str, ...] = (
    "trained_checkpoint_lineage",
    "checkpoint_sha256",
    "state_dict_sha256",
    "model_source_sha256",
    "source_config_lineage",
    "source_config_sha256",
    "source_config_kind",
    "source_config_source",
    "source_config_is_fixture",
    "source_scope",
    "capture_origin",
)


def build_snerv_source_forward_surface_provenance(
    *,
    pair_ids: Sequence[int],
    archive_sha256: str,
    producer_by_surface: Mapping[str, str] | None = None,
    backend_by_surface: Mapping[str, str] | None = None,
    tensor_capture_authority_by_surface: Mapping[str, str] | None = None,
    scorer_capture_authority_by_surface: Mapping[str, str] | None = None,
    extra_by_surface: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build canonical per-surface custody for SourceForwardProof producers."""

    producers = dict(producer_by_surface or {})
    backends = dict(backend_by_surface or {})
    tensor_authority = dict(tensor_capture_authority_by_surface or {})
    scorer_authority = dict(scorer_capture_authority_by_surface or {})
    extras = {
        str(surface): dict(value)
        for surface, value in dict(extra_by_surface or {}).items()
        if isinstance(value, Mapping)
    }
    normalized_pair_ids = _normalize_pair_ids(pair_ids)
    return {
        surface: {
            **extras.get(surface, {}),
            "schema": SOURCE_FORWARD_SURFACE_PROVENANCE_SCHEMA,
            "surface": surface,
            "producer": str(producers.get(surface) or f"{surface}_source_forward"),
            "backend": str(backends.get(surface) or surface),
            "pair_ids": list(normalized_pair_ids),
            "tensor_capture_authority": str(
                tensor_authority.get(surface) or "real_surface_forward_capture"
            ),
            "scorer_capture_authority": str(
                scorer_authority.get(surface) or "real_surface_scorer_capture"
            ),
            **(
                {"archive_sha256": str(archive_sha256)}
                if surface in SOURCE_FORWARD_ARCHIVE_BOUND_SURFACES
                else {}
            ),
        }
        for surface in SOURCE_FORWARD_SURFACES
    }


def build_snerv_source_forward_proof_action_effect(
    *,
    action_id: str,
    archive_sha256: str,
    payload_section_hashes: Mapping[str, str],
    pair_ids: Sequence[int],
    tensors_by_surface: Mapping[str, Mapping[str, Any]],
    scorer_deltas: Mapping[str, Any],
    destructive_payload_bit_flip: Mapping[str, Any],
    destructive_payload_bit_flip_matrix: Mapping[str, Any] | None = None,
    output2_boundary_verdict: Mapping[str, Any] | None = None,
    surface_provenance: Mapping[str, Mapping[str, Any]] | None = None,
    tolerance_by_tensor: Mapping[str, float] | None = None,
    archive_bytes: int | None = None,
    generated_utc: str | None = None,
) -> dict[str, Any]:
    """Build a validated SourceForwardProof action row from real tensors."""

    tolerance_by_tensor = dict(tolerance_by_tensor or {})
    tensor_hashes: dict[str, dict[str, str]] = {}
    tensor_deltas: dict[str, dict[str, float | None]] = {}
    blockers: list[str] = []

    surfaces = {
        surface: {
            str(name): np.asarray(value)
            for name, value in dict(tensors_by_surface.get(surface) or {}).items()
        }
        for surface in SOURCE_FORWARD_SURFACES
    }

    for surface in SOURCE_FORWARD_SURFACES:
        tensor_hashes[surface] = {}
        for name in SOURCE_FORWARD_TENSOR_NAMES:
            tensor = surfaces[surface].get(name)
            if tensor is None:
                blockers.append(f"source_forward_tensor_missing:{surface}:{name}")
                continue
            tensor_hashes[surface][name] = _hash_array_exact(tensor)

    reference = surfaces[SOURCE_FORWARD_REFERENCE_SURFACE]
    for name in SOURCE_FORWARD_TENSOR_NAMES:
        tensor_deltas[name] = {}
        ref = reference.get(name)
        if ref is None:
            tensor_deltas[name][SOURCE_FORWARD_REFERENCE_SURFACE] = None
            continue
        tensor_deltas[name][SOURCE_FORWARD_REFERENCE_SURFACE] = 0.0
        tolerance = float(tolerance_by_tensor.get(name, 0.0))
        for surface in SOURCE_FORWARD_COMPARISON_SURFACES:
            other = surfaces[surface].get(name)
            delta = _max_abs_delta_or_none(ref, other)
            tensor_deltas[name][surface] = delta
            if delta is None:
                blockers.append(f"source_forward_tensor_shape_mismatch:{surface}:{name}")
            elif delta > tolerance:
                blockers.append(f"source_forward_tensor_delta_exceeds_tolerance:{surface}:{name}")

    bitflip_status = validate_snerv_payload_bitflip_falsification(
        destructive_payload_bit_flip
    )
    blockers.extend(bitflip_status["blockers"])
    bitflip_matrix_status = validate_snerv_payload_bitflip_falsification_matrix(
        destructive_payload_bit_flip_matrix
    )
    blockers.extend(bitflip_matrix_status["blockers"])

    score_status = _validate_scorer_deltas(scorer_deltas)
    blockers.extend(score_status["blockers"])
    normalized_scorer_deltas = score_status["normalized_scorer_deltas"]
    provenance_status = _validate_surface_provenance(
        surface_provenance,
        archive_sha256=archive_sha256,
        pair_ids=pair_ids,
    )
    blockers.extend(provenance_status["blockers"])
    normalized_surface_provenance = provenance_status["normalized_surface_provenance"]
    output2_status = validate_snerv_output2_boundary_verdict(output2_boundary_verdict)
    blockers.extend(output2_status["blockers"])
    normalized_output2_boundary = output2_status["normalized_output2_boundary_verdict"]

    if not _looks_like_sha256(archive_sha256):
        blockers.append("source_forward_archive_sha256_invalid")
    if not payload_section_hashes:
        blockers.append("source_forward_payload_section_hashes_missing")
    for name, value in payload_section_hashes.items():
        if not str(name):
            blockers.append("source_forward_payload_section_hash_name_missing")
        if not _looks_like_sha256(value):
            blockers.append(f"source_forward_payload_section_hash_invalid:{name}")
    if not pair_ids:
        blockers.append("source_forward_pair_ids_missing")
    if archive_bytes is not None and int(archive_bytes) <= 0:
        blockers.append("source_forward_archive_bytes_invalid")

    blockers = _ordered_unique(blockers)
    first_failed_tensor = _first_failed_tensor(blockers)
    passed = not blockers
    launch_gate_clearable = bool(
        passed and normalized_output2_boundary.get("verdict") == SOURCE_IDENTICAL
    )
    return {
        "schema": SNERV_SOURCE_FORWARD_PROOF_ACTION_EFFECT_SCHEMA,
        "action_id": str(action_id),
        "family": "snerv",
        "authority": "source_forward_action_effect",
        "generated_utc": generated_utc,
        "pair_ids": [int(value) for value in pair_ids],
        "archive_sha256": str(archive_sha256),
        "archive_bytes": None if archive_bytes is None else int(archive_bytes),
        "payload_section_hashes": {str(k): str(v) for k, v in payload_section_hashes.items()},
        "surfaces": list(SOURCE_FORWARD_SURFACES),
        "reference_surface": SOURCE_FORWARD_REFERENCE_SURFACE,
        "comparison_surfaces": list(SOURCE_FORWARD_COMPARISON_SURFACES),
        "tensor_names": list(SOURCE_FORWARD_TENSOR_NAMES),
        "tensor_hashes": tensor_hashes,
        "tensor_deltas": tensor_deltas,
        "tolerance_by_tensor": {str(k): float(v) for k, v in tolerance_by_tensor.items()},
        "scorer_deltas": normalized_scorer_deltas,
        "surface_provenance": normalized_surface_provenance,
        "destructive_payload_bit_flip": dict(destructive_payload_bit_flip),
        "destructive_payload_bit_flip_matrix": bitflip_matrix_status[
            "normalized_matrix"
        ],
        "output2_boundary_verdict": normalized_output2_boundary,
        "rgb_uint8_and_scorer_compared": _scorer_surfaces_present(tensor_hashes),
        "parseback_receiver_surface_compared": _parseback_receiver_surfaces_present(
            tensor_hashes
        ),
        "source_forward_replay_bound": passed,
        "source_forward_replay_verified": passed,
        "source_forward_replay_authority": passed,
        "full_stack_source_forward_replay_proven": passed,
        "launch_gate_clearable": launch_gate_clearable,
        "first_failed_tensor": first_failed_tensor,
        "passed": passed,
        "blockers": blockers,
        **PROXY_FALSE_AUTHORITY_FIELDS,
    }


def build_snerv_payload_bitflip_falsification(
    *,
    bitflip_section: str,
    baseline_section_sha256: str,
    mutated_section_sha256: str,
    proof_passed_after_bitflip: bool,
    first_failed_tensor: str | None,
    first_failed_surface: str | None = None,
    bit_offset: int | None = None,
    bit_mask: int | None = None,
    failure: str | None = None,
) -> dict[str, Any]:
    """Return the destructive payload mutation row consumed by the proof."""

    row = {
        "schema": SNERV_PAYLOAD_BITFLIP_FALSIFICATION_SCHEMA,
        "bitflip_section": str(bitflip_section),
        "baseline_section_sha256": str(baseline_section_sha256),
        "mutated_section_sha256": str(mutated_section_sha256),
        "proof_passed": bool(proof_passed_after_bitflip),
        "first_failed_tensor": first_failed_tensor,
        "first_failed_surface": first_failed_surface,
        "bit_offset": bit_offset,
        "bit_mask": bit_mask,
        "failure": failure,
        **PROXY_FALSE_AUTHORITY_FIELDS,
    }
    status = validate_snerv_payload_bitflip_falsification(row)
    return {**row, "passed": not status["blockers"], "blockers": status["blockers"]}


def build_snerv_payload_bitflip_falsification_matrix(
    section_proofs: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Bundle destructive payload mutations across every charged SNeRV section."""

    normalized = {str(key): dict(value) for key, value in section_proofs.items()}
    row = {
        "schema": SNERV_PAYLOAD_BITFLIP_FALSIFICATION_MATRIX_SCHEMA,
        "required_sections": list(SNERV_PAYLOAD_BITFLIP_REQUIRED_SECTIONS),
        "section_proofs": normalized,
        "covered_sections": sorted(normalized),
        "noncausal_sections": _bitflip_noncausal_sections(normalized),
        **PROXY_FALSE_AUTHORITY_FIELDS,
    }
    status = validate_snerv_payload_bitflip_falsification_matrix(row)
    return {
        **row,
        "passed": status["passed"],
        "blockers": status["blockers"],
    }


def validate_snerv_source_forward_proof_action_effect(
    row: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate that a SNeRV proof row is numerical and payload-causal."""

    blockers: list[str] = []
    if not isinstance(row, Mapping):
        return {"passed": False, "blockers": ["snerv_source_forward_proof_missing"]}
    if row.get("schema") != SNERV_SOURCE_FORWARD_PROOF_ACTION_EFFECT_SCHEMA:
        blockers.append("snerv_source_forward_proof_schema_invalid")
    if not str(row.get("action_id") or ""):
        blockers.append("snerv_source_forward_action_id_missing")
    if not _looks_like_sha256(row.get("archive_sha256")):
        blockers.append("snerv_source_forward_archive_sha256_invalid")
    archive_bytes = row.get("archive_bytes")
    if archive_bytes is not None:
        try:
            if int(archive_bytes) <= 0:
                blockers.append("snerv_source_forward_archive_bytes_invalid")
        except (TypeError, ValueError):
            blockers.append("snerv_source_forward_archive_bytes_invalid")
    payload_hashes = row.get("payload_section_hashes")
    if not isinstance(payload_hashes, Mapping) or not payload_hashes:
        blockers.append("snerv_source_forward_payload_section_hashes_missing")
    elif any(not _looks_like_sha256(value) for value in payload_hashes.values()):
        blockers.append("snerv_source_forward_payload_section_hash_invalid")
    if not _valid_pair_ids(row.get("pair_ids")):
        blockers.append("snerv_source_forward_pair_ids_missing")
    if row.get("rgb_uint8_and_scorer_compared") is not True:
        blockers.append("snerv_source_forward_rgb_uint8_scorer_not_compared")
    if row.get("parseback_receiver_surface_compared") is not True:
        blockers.append("snerv_source_forward_parseback_receiver_not_compared")

    tensor_hashes = row.get("tensor_hashes")
    if not isinstance(tensor_hashes, Mapping):
        blockers.append("snerv_source_forward_tensor_hashes_missing")
    else:
        for surface in SOURCE_FORWARD_SURFACES:
            surface_hashes = tensor_hashes.get(surface)
            if not isinstance(surface_hashes, Mapping):
                blockers.append(f"snerv_source_forward_surface_missing:{surface}")
                continue
            missing = [name for name in SOURCE_FORWARD_TENSOR_NAMES if name not in surface_hashes]
            blockers.extend(f"snerv_source_forward_tensor_hash_missing:{surface}:{name}" for name in missing)
            bad = [
                name
                for name, value in surface_hashes.items()
                if not _looks_like_sha256(value)
            ]
            blockers.extend(f"snerv_source_forward_tensor_hash_invalid:{surface}:{name}" for name in bad)

    tensor_deltas = row.get("tensor_deltas")
    tolerance_by_tensor = row.get("tolerance_by_tensor") or {}
    if not isinstance(tensor_deltas, Mapping):
        blockers.append("snerv_source_forward_tensor_deltas_missing")
    else:
        for name in SOURCE_FORWARD_TENSOR_NAMES:
            deltas = tensor_deltas.get(name)
            if not isinstance(deltas, Mapping):
                blockers.append(f"snerv_source_forward_tensor_delta_missing:{name}")
                continue
            for surface in SOURCE_FORWARD_COMPARISON_SURFACES:
                value = deltas.get(surface)
                if not _nonnegative_finite_float(value):
                    blockers.append(f"snerv_source_forward_tensor_delta_invalid:{surface}:{name}")
                    continue
                tolerance = _float_or_default(
                    tolerance_by_tensor.get(name) if isinstance(tolerance_by_tensor, Mapping) else None,
                    0.0,
                )
                if float(value) > tolerance:
                    blockers.append(
                        f"snerv_source_forward_tensor_delta_exceeds_tolerance:{surface}:{name}"
                    )

    score_status = _validate_scorer_deltas(row.get("scorer_deltas"))
    blockers.extend(score_status["blockers"])
    provenance_status = _validate_surface_provenance(
        row.get("surface_provenance"),
        archive_sha256=str(row.get("archive_sha256") or ""),
        pair_ids=row.get("pair_ids") if isinstance(row.get("pair_ids"), Sequence) else (),
    )
    blockers.extend(provenance_status["blockers"])
    bitflip_status = validate_snerv_payload_bitflip_falsification(
        row.get("destructive_payload_bit_flip")
    )
    blockers.extend(bitflip_status["blockers"])
    bitflip_matrix_status = validate_snerv_payload_bitflip_falsification_matrix(
        row.get("destructive_payload_bit_flip_matrix")
    )
    blockers.extend(bitflip_matrix_status["blockers"])
    output2_status = validate_snerv_output2_boundary_verdict(
        row.get("output2_boundary_verdict")
    )
    blockers.extend(output2_status["blockers"])
    if row.get("launch_gate_clearable") is not True:
        blockers.append("snerv_source_forward_launch_gate_clearable_false")
    if row.get("passed") is not True:
        blockers.append("snerv_source_forward_proof_not_passed")
    if row.get("source_forward_replay_authority") is not True:
        blockers.append("snerv_source_forward_replay_authority_false")
    return {"passed": not _ordered_unique(blockers), "blockers": _ordered_unique(blockers)}


def validate_snerv_output2_boundary_verdict(
    row: Mapping[str, Any] | None,
) -> dict[str, Any]:
    blockers: list[str] = []
    if not isinstance(row, Mapping):
        return {
            "passed": False,
            "blockers": ["snerv_output2_boundary_verdict_missing"],
            "normalized_output2_boundary_verdict": {},
        }
    verdict = str(row.get("verdict") or "")
    if row.get("schema") != SNERV_OUTPUT2_BOUNDARY_VERDICT_SCHEMA:
        blockers.append("snerv_output2_boundary_verdict_schema_invalid")
    if verdict not in SOURCE_FORWARD_OUTPUT2_VERDICTS:
        blockers.append("snerv_output2_boundary_verdict_invalid")
    if verdict != SOURCE_IDENTICAL:
        blockers.append(
            f"snerv_output2_boundary_not_source_identical:{verdict or 'missing'}"
        )
    if (row.get("passed") is True) != (verdict == SOURCE_IDENTICAL):
        blockers.append("snerv_output2_boundary_passed_flag_mismatch")
    nested_blockers = [str(value) for value in row.get("blockers") or []]
    if nested_blockers:
        blockers.append("snerv_output2_boundary_nested_blockers_present")
    has_output2 = dict(row.get("has_output2_by_surface") or {})
    missing_surfaces = [
        surface
        for surface in SOURCE_FORWARD_SURFACES
        if has_output2.get(surface) is not True
    ]
    if missing_surfaces:
        blockers.extend(
            f"snerv_output2_missing_source_forward_surface:{surface}"
            for surface in missing_surfaces
        )
    raw_storage = row.get("archive_tub_output2_storage")
    storage = raw_storage if isinstance(raw_storage, Mapping) else {}
    if storage.get("receiver_frame_decode_consumes_output2") is not True:
        blockers.append("snerv_output2_not_consumed_by_receiver_frame_decode")
    if storage.get("receiver_output2_frame_shape_match") is not True:
        blockers.append("snerv_output2_receiver_frame_shape_mismatch")
    if storage.get("receiver_executes_output2_fusion_from_payload") is not True:
        blockers.append("snerv_output2_receiver_fusion_not_payload_bound")
    shape_adapter_applied = any(
        bool(storage.get(key))
        for key in (
            "shape_adapter_applied",
            "output2_shape_adapter",
            "receiver_output2_shape_adapter_applied",
        )
    )
    if shape_adapter_applied:
        blockers.append("snerv_output2_shape_adapter_forbidden")
    normalized = {
        "schema": str(row.get("schema") or ""),
        "verdict": verdict,
        "passed": row.get("passed") is True,
        "has_output2_by_surface": dict(row.get("has_output2_by_surface") or {}),
        "output2_shapes_by_surface": dict(row.get("output2_shapes_by_surface") or {}),
        "archive_tub_output2_storage": {
            **dict(storage),
            "shape_adapter_forbidden": True,
            "shape_adapter_applied": shape_adapter_applied,
        },
        "minimal_causal_basis_recommendation": list(
            row.get("minimal_causal_basis_recommendation") or []
        ),
        "blockers": nested_blockers,
        "required_next_step": row.get("required_next_step"),
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    return {
        "passed": not _ordered_unique(blockers),
        "blockers": _ordered_unique(blockers),
        "normalized_output2_boundary_verdict": normalized,
    }


def validate_snerv_payload_bitflip_falsification(
    row: Mapping[str, Any] | None,
) -> dict[str, Any]:
    blockers: list[str] = []
    if not isinstance(row, Mapping):
        return {"passed": False, "blockers": ["snerv_payload_bitflip_falsification_missing"]}
    if row.get("schema") != SNERV_PAYLOAD_BITFLIP_FALSIFICATION_SCHEMA:
        blockers.append("snerv_payload_bitflip_schema_invalid")
    if not str(row.get("bitflip_section") or ""):
        blockers.append("snerv_payload_bitflip_section_missing")
    if not _looks_like_sha256(row.get("baseline_section_sha256")):
        blockers.append("snerv_payload_bitflip_baseline_hash_invalid")
    if not _looks_like_sha256(row.get("mutated_section_sha256")):
        blockers.append("snerv_payload_bitflip_mutated_hash_invalid")
    if (
        _looks_like_sha256(row.get("baseline_section_sha256"))
        and row.get("baseline_section_sha256") == row.get("mutated_section_sha256")
    ):
        blockers.append("snerv_payload_bitflip_section_hash_unchanged")
    if row.get("proof_passed") is not False:
        blockers.append("snerv_payload_bitflip_did_not_falsify_proof")
    failed_tensor = row.get("first_failed_tensor")
    if not isinstance(failed_tensor, str) or not failed_tensor:
        blockers.append("snerv_payload_bitflip_first_failed_tensor_missing")
    return {"passed": not _ordered_unique(blockers), "blockers": _ordered_unique(blockers)}


def validate_snerv_payload_bitflip_falsification_matrix(
    row: Mapping[str, Any] | None,
) -> dict[str, Any]:
    blockers: list[str] = []
    if not isinstance(row, Mapping):
        return {
            "passed": False,
            "blockers": ["snerv_payload_bitflip_matrix_missing"],
            "normalized_matrix": {},
        }
    if row.get("schema") != SNERV_PAYLOAD_BITFLIP_FALSIFICATION_MATRIX_SCHEMA:
        blockers.append("snerv_payload_bitflip_matrix_schema_invalid")
    section_proofs = row.get("section_proofs")
    if not isinstance(section_proofs, Mapping):
        blockers.append("snerv_payload_bitflip_matrix_section_proofs_missing")
        section_proofs = {}
    normalized_proofs: dict[str, dict[str, Any]] = {}
    covered_sections = {str(section) for section in section_proofs}
    missing_sections = sorted(set(SNERV_PAYLOAD_BITFLIP_REQUIRED_SECTIONS) - covered_sections)
    for section in missing_sections:
        blockers.append(f"snerv_payload_bitflip_matrix_section_missing:{section}")
    for section, proof_value in section_proofs.items():
        section_name = str(section)
        proof = dict(proof_value) if isinstance(proof_value, Mapping) else {}
        if str(proof.get("bitflip_section") or "") != section_name:
            blockers.append(
                f"snerv_payload_bitflip_matrix_section_name_mismatch:{section_name}"
            )
        proof_status = validate_snerv_payload_bitflip_falsification(proof)
        blockers.extend(
            f"snerv_payload_bitflip_matrix_{section_name}:{blocker}"
            for blocker in proof_status["blockers"]
        )
        normalized_proofs[section_name] = proof
    normalized = {
        "schema": SNERV_PAYLOAD_BITFLIP_FALSIFICATION_MATRIX_SCHEMA,
        "required_sections": list(SNERV_PAYLOAD_BITFLIP_REQUIRED_SECTIONS),
        "covered_sections": sorted(covered_sections),
        "missing_sections": missing_sections,
        "noncausal_sections": _bitflip_noncausal_sections(normalized_proofs),
        "section_proofs": normalized_proofs,
        **PROXY_FALSE_AUTHORITY_FIELDS,
    }
    return {
        "passed": not _ordered_unique(blockers),
        "blockers": _ordered_unique(blockers),
        "normalized_matrix": normalized,
    }


def _bitflip_noncausal_sections(
    section_proofs: Mapping[str, Mapping[str, Any]],
) -> list[str]:
    noncausal: list[str] = []
    for section, proof in section_proofs.items():
        if (
            proof.get("proof_passed") is not False
            or not str(proof.get("first_failed_tensor") or "")
        ):
            noncausal.append(str(section))
    return sorted(noncausal)


def find_snerv_source_forward_proof_rows(payload: Any) -> list[dict[str, Any]]:
    """Recursively collect SourceForwardProof action rows from JSON payloads."""

    hits: list[dict[str, Any]] = []
    if isinstance(payload, Mapping):
        if payload.get("schema") == SNERV_SOURCE_FORWARD_PROOF_ACTION_EFFECT_SCHEMA:
            hits.append(dict(payload))
        for value in payload.values():
            hits.extend(find_snerv_source_forward_proof_rows(value))
    elif isinstance(payload, (list, tuple)):
        for value in payload:
            hits.extend(find_snerv_source_forward_proof_rows(value))
    return hits


def _validate_surface_provenance(
    row: Any,
    *,
    archive_sha256: str,
    pair_ids: Sequence[int] | Any,
) -> dict[str, Any]:
    blockers: list[str] = []
    if not isinstance(row, Mapping):
        return {
            "passed": False,
            "blockers": ["snerv_source_forward_surface_provenance_missing"],
            "normalized_surface_provenance": {},
        }
    normalized_pair_ids = _normalize_pair_ids(pair_ids)
    normalized: dict[str, dict[str, Any]] = {}
    for surface in SOURCE_FORWARD_SURFACES:
        surface_row = row.get(surface)
        if not isinstance(surface_row, Mapping):
            blockers.append(
                f"snerv_source_forward_surface_provenance_surface_missing:{surface}"
            )
            continue
        surface_map = dict(surface_row)
        for field in SOURCE_FORWARD_PROVENANCE_REQUIRED_FIELDS:
            if field not in surface_map:
                blockers.append(
                    f"snerv_source_forward_surface_provenance_field_missing:{surface}:{field}"
                )
        if surface_map.get("schema") != SOURCE_FORWARD_SURFACE_PROVENANCE_SCHEMA:
            blockers.append(
                f"snerv_source_forward_surface_provenance_schema_invalid:{surface}"
            )
        if str(surface_map.get("surface") or "") != surface:
            blockers.append(
                f"snerv_source_forward_surface_provenance_surface_mismatch:{surface}"
            )
        if not str(surface_map.get("producer") or ""):
            blockers.append(
                f"snerv_source_forward_surface_provenance_producer_missing:{surface}"
            )
        if not str(surface_map.get("backend") or ""):
            blockers.append(
                f"snerv_source_forward_surface_provenance_backend_missing:{surface}"
            )
        surface_pair_ids = _normalize_pair_ids(surface_map.get("pair_ids"))
        if surface_pair_ids != normalized_pair_ids:
            blockers.append(
                f"snerv_source_forward_surface_provenance_pair_ids_mismatch:{surface}"
            )
        for field in ("tensor_capture_authority", "scorer_capture_authority"):
            value = str(surface_map.get(field) or "")
            if not value:
                blockers.append(
                    f"snerv_source_forward_surface_provenance_authority_missing:{surface}:{field}"
                )
                continue
            lowered = value.lower()
            if any(token in lowered for token in SOURCE_FORWARD_FORBIDDEN_PROVENANCE_TOKENS):
                blockers.append(
                    f"snerv_source_forward_surface_provenance_authority_not_real:{surface}:{field}"
                )
        if surface in SOURCE_FORWARD_ARCHIVE_BOUND_SURFACES:
            value = str(surface_map.get("archive_sha256") or "")
            if value != str(archive_sha256):
                blockers.append(
                    f"snerv_source_forward_surface_provenance_archive_sha256_mismatch:{surface}"
                )
        if surface == SOURCE_FORWARD_REFERENCE_SURFACE:
            _validate_official_torch_trained_checkpoint_lineage(
                surface_map,
                blockers=blockers,
            )
        normalized[surface] = {
            "schema": str(surface_map.get("schema") or ""),
            "surface": str(surface_map.get("surface") or ""),
            "producer": str(surface_map.get("producer") or ""),
            "backend": str(surface_map.get("backend") or ""),
            "pair_ids": surface_pair_ids,
            "tensor_capture_authority": str(surface_map.get("tensor_capture_authority") or ""),
            "scorer_capture_authority": str(surface_map.get("scorer_capture_authority") or ""),
            **(
                {"archive_sha256": str(surface_map.get("archive_sha256") or "")}
                if "archive_sha256" in surface_map
                else {}
            ),
            **{
                field: str(surface_map.get(field) or "")
                for field in SOURCE_FORWARD_OPTIONAL_PROVENANCE_FIELDS
                if field in surface_map
            },
        }
    return {
        "passed": not _ordered_unique(blockers),
        "blockers": _ordered_unique(blockers),
        "normalized_surface_provenance": normalized,
    }


def _validate_official_torch_trained_checkpoint_lineage(
    surface_map: Mapping[str, Any],
    *,
    blockers: list[str],
) -> None:
    if str(surface_map.get("tensor_capture_authority") or "") != "upstream_snerv_t_forward_source_graph":
        blockers.append(
            "snerv_source_forward_official_torch_upstream_tensor_capture_authority_missing"
        )
    lineage = str(surface_map.get("trained_checkpoint_lineage") or "")
    if lineage not in SOURCE_FORWARD_OFFICIAL_TORCH_ALLOWED_TRAINED_LINEAGES:
        blockers.append(
            "snerv_source_forward_official_torch_trained_checkpoint_lineage_missing"
        )
    if str(surface_map.get("source_scope") or "") != "official_trained_checkpoint":
        blockers.append(
            "snerv_source_forward_official_torch_trained_checkpoint_source_scope_missing"
        )
    if str(surface_map.get("capture_origin") or "") != "official_upstream_trained_checkpoint":
        blockers.append(
            "snerv_source_forward_official_torch_capture_origin_missing"
        )
    config_lineage = str(surface_map.get("source_config_lineage") or "")
    if config_lineage not in SOURCE_FORWARD_OFFICIAL_TORCH_ALLOWED_CONFIG_LINEAGES:
        blockers.append(
            "snerv_source_forward_official_torch_trained_config_lineage_missing"
        )
    if _truthy_flag(surface_map.get("source_config_is_fixture")):
        blockers.append(
            "snerv_source_forward_official_torch_source_config_fixture_forbidden"
        )
    for field in ("checkpoint_sha256", "state_dict_sha256", "model_source_sha256"):
        if not _looks_like_sha256(surface_map.get(field)):
            blockers.append(
                f"snerv_source_forward_official_torch_{field}_invalid"
            )
    if not _looks_like_sha256(surface_map.get("source_config_sha256")):
        blockers.append(
            "snerv_source_forward_official_torch_source_config_sha256_invalid"
        )


def _truthy_flag(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _validate_scorer_deltas(row: Any) -> dict[str, Any]:
    blockers: list[str] = []
    if not isinstance(row, Mapping):
        return {
            "passed": False,
            "blockers": ["snerv_source_forward_scorer_deltas_missing"],
            "normalized_scorer_deltas": {},
        }
    for field in ("d_seg", "d_pose", "delta_score_nonrate"):
        if not _finite_float(row.get(field)):
            blockers.append(f"snerv_source_forward_scorer_delta_invalid:{field}")
    by_surface = row.get("by_surface")
    if not isinstance(by_surface, Mapping):
        blockers.append("snerv_source_forward_scorer_by_surface_missing")
        return {
            "passed": False,
            "blockers": _ordered_unique(blockers),
            "normalized_scorer_deltas": dict(row),
        }

    normalized_by_surface: dict[str, dict[str, float]] = {}
    for surface in SOURCE_FORWARD_SURFACES:
        metrics = by_surface.get(surface)
        if not isinstance(metrics, Mapping):
            blockers.append(f"snerv_source_forward_scorer_surface_missing:{surface}")
            continue
        normalized_by_surface[surface] = {}
        for field in SOURCE_FORWARD_SCORER_FIELDS:
            value = metrics.get(field)
            if not _nonnegative_finite_float(value):
                blockers.append(
                    f"snerv_source_forward_scorer_surface_metric_invalid:{surface}:{field}"
                )
                continue
            normalized_by_surface[surface][field] = float(value)

    surface_deltas: dict[str, dict[str, float | None]] = {}
    reference = normalized_by_surface.get(SOURCE_FORWARD_REFERENCE_SURFACE)
    tolerance_by_field = row.get("tolerance_by_field")
    tolerance_by_field = tolerance_by_field if isinstance(tolerance_by_field, Mapping) else {}
    if reference is None:
        blockers.append("snerv_source_forward_scorer_reference_surface_missing")
    else:
        for field in SOURCE_FORWARD_SCORER_FIELDS:
            surface_deltas.setdefault(SOURCE_FORWARD_REFERENCE_SURFACE, {})[field] = 0.0
        for surface in SOURCE_FORWARD_COMPARISON_SURFACES:
            surface_metrics = normalized_by_surface.get(surface)
            surface_deltas[surface] = {}
            for field in SOURCE_FORWARD_SCORER_FIELDS:
                if surface_metrics is None or field not in surface_metrics or field not in reference:
                    surface_deltas[surface][field] = None
                    blockers.append(
                        f"snerv_source_forward_scorer_surface_delta_missing:{surface}:{field}"
                    )
                    continue
                delta = abs(float(surface_metrics[field]) - float(reference[field]))
                surface_deltas[surface][field] = delta
                tolerance = _float_or_default(tolerance_by_field.get(field), 0.0)
                if delta > tolerance:
                    blockers.append(
                        f"snerv_source_forward_scorer_surface_delta_exceeds_tolerance:{surface}:{field}"
                    )

    normalized = {
        "d_seg": float(row.get("d_seg")) if _finite_float(row.get("d_seg")) else row.get("d_seg"),
        "d_pose": float(row.get("d_pose")) if _finite_float(row.get("d_pose")) else row.get("d_pose"),
        "delta_score_nonrate": (
            float(row.get("delta_score_nonrate"))
            if _finite_float(row.get("delta_score_nonrate"))
            else row.get("delta_score_nonrate")
        ),
        "by_surface": normalized_by_surface,
        "surface_deltas": surface_deltas,
        "tolerance_by_field": {
            str(key): float(value)
            for key, value in tolerance_by_field.items()
            if _nonnegative_finite_float(value)
        },
    }
    return {
        "passed": not _ordered_unique(blockers),
        "blockers": _ordered_unique(blockers),
        "normalized_scorer_deltas": normalized,
    }


def _scorer_surfaces_present(tensor_hashes: Mapping[str, Mapping[str, str]]) -> bool:
    required = {
        "rgb_pair_uint8",
        "segnet_input",
        "posenet_input",
        "segnet_logits",
        "segnet_argmax",
        "posenet_output",
    }
    return all(
        required.issubset(set(tensor_hashes.get(surface, {})))
        for surface in SOURCE_FORWARD_SURFACES
    )


def _parseback_receiver_surfaces_present(
    tensor_hashes: Mapping[str, Mapping[str, str]]
) -> bool:
    required = set(SOURCE_FORWARD_TENSOR_NAMES)
    return required.issubset(set(tensor_hashes.get("archive_parseback", {}))) and required.issubset(
        set(tensor_hashes.get("numpy_receiver", {}))
    )


def _first_failed_tensor(blockers: Sequence[str]) -> str | None:
    for blocker in blockers:
        parts = str(blocker).split(":")
        if len(parts) >= 3 and parts[-1] in SOURCE_FORWARD_TENSOR_NAMES:
            return parts[-1]
    return None


def _valid_pair_ids(value: Any) -> bool:
    return bool(_normalize_pair_ids(value))


def _normalize_pair_ids(value: Any) -> list[int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        return []
    if not value:
        return []
    try:
        out = [int(item) for item in value]
    except (TypeError, ValueError):
        return []
    if any(item < 0 for item in out):
        return []
    return out


def _hash_array_exact(array: Any) -> str:
    arr = np.ascontiguousarray(np.asarray(array))
    h = sha256()
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(b"\0")
    h.update(json.dumps(list(arr.shape), sort_keys=True).encode("utf-8"))
    h.update(b"\0")
    h.update(arr.tobytes())
    return h.hexdigest()


def _max_abs_delta_or_none(left: Any, right: Any) -> float | None:
    if right is None:
        return None
    left_arr = np.asarray(left, dtype=np.float64)
    right_arr = np.asarray(right, dtype=np.float64)
    if left_arr.shape != right_arr.shape:
        return None
    delta = np.abs(left_arr - right_arr)
    if not np.all(np.isfinite(delta)):
        return None
    return float(np.max(delta)) if delta.size else 0.0


def _looks_like_sha256(value: Any) -> bool:
    text = str(value or "").strip().lower()
    return len(text) == 64 and all(ch in "0123456789abcdef" for ch in text)


def _finite_float(value: Any) -> bool:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(parsed))


def _nonnegative_finite_float(value: Any) -> bool:
    if not _finite_float(value):
        return False
    return float(value) >= 0.0


def _float_or_default(value: Any, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return parsed if np.isfinite(parsed) and parsed >= 0.0 else float(default)


def _ordered_unique(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        text = str(value)
        if text and text not in seen:
            out.append(text)
            seen.add(text)
    return out


__all__ = [
    "DROP_OUTPUT2_USE_MFU_HFR_TUB_BASIS",
    "REPARAMETERIZED_RENAME_REQUIRED",
    "SNERV_OUTPUT2_BOUNDARY_VERDICT_SCHEMA",
    "SNERV_PAYLOAD_BITFLIP_FALSIFICATION_MATRIX_SCHEMA",
    "SNERV_PAYLOAD_BITFLIP_FALSIFICATION_SCHEMA",
    "SNERV_PAYLOAD_BITFLIP_REQUIRED_SECTIONS",
    "SNERV_SOURCE_FORWARD_PROOF_ACTION_EFFECT_SCHEMA",
    "SOURCE_FORWARD_OUTPUT2_VERDICTS",
    "SOURCE_FORWARD_SCORER_FIELDS",
    "SOURCE_FORWARD_SURFACES",
    "SOURCE_FORWARD_SURFACE_PROVENANCE_SCHEMA",
    "SOURCE_FORWARD_TENSOR_NAMES",
    "SOURCE_IDENTICAL",
    "build_snerv_payload_bitflip_falsification",
    "build_snerv_payload_bitflip_falsification_matrix",
    "build_snerv_source_forward_proof_action_effect",
    "build_snerv_source_forward_surface_provenance",
    "find_snerv_source_forward_proof_rows",
    "validate_snerv_output2_boundary_verdict",
    "validate_snerv_payload_bitflip_falsification",
    "validate_snerv_payload_bitflip_falsification_matrix",
    "validate_snerv_source_forward_proof_action_effect",
]
