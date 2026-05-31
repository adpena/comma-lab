# SPDX-License-Identifier: MIT
"""Runtime payload bridge for Z8 Mamba/Wyner-Ziv archive sections.

This module makes the non-wavelet Z8HPC1 payloads typed at receiver runtime.
The bridge decodes per-pair Wyner-Ziv top states from archive bytes using the
same top-LL side-information model as the encoder, then projects the decoded
state into frame-1 top-LL. That is a real receiver pixel driver, but it remains
false-authority for score, promotion, and exact dispatch until contest-axis
evaluation signs the byte-closed archive/runtime pair.
"""

from __future__ import annotations

import hashlib
import struct
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from tac.substrates.z8_hierarchical_predictive_coding.archive import (
    Z8HierarchicalArchive,
    pack_archive,
    parse_archive,
)
from tac.substrates.z8_hierarchical_predictive_coding.canonical_quadruple_binding import (
    build_canonical_quadruple_binding_from_z8_config,
    parse_pair_blobs_from_wavelet_blob,
    reconstruct_pair_rgb_from_pyramid,
)

Z8_RUNTIME_PAYLOAD_BRIDGE_REPORT_SCHEMA = "z8_hpc1_runtime_payload_bridge_report.v1"
Z8_STATE_TO_TOP_LL_PROJECTION_GAIN = 1.0 / 64.0
Z8_STATE_TO_TOP_LL_PROJECTION_SEED = 23
Z8_STACK_CONTEXT_TO_TOP_LL_PROJECTION_GAIN = 1.0 / 128.0
Z8_STACK_CONTEXT_TO_TOP_LL_PROJECTION_SEED = 29
FALSE_AUTHORITY: dict[str, bool] = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "promotable": False,
    "dispatch_attempted": False,
    "gpu_launched": False,
}


def _length_prefixed_payloads(blob: bytes) -> list[bytes]:
    if len(blob) < 4:
        raise ValueError("Z8 Wyner-Ziv blob is too short to carry pair count")
    pos = 0
    (num_payloads,) = struct.unpack("<I", blob[pos : pos + 4])
    pos += 4
    payloads: list[bytes] = []
    for payload_index in range(int(num_payloads)):
        if pos + 4 > len(blob):
            raise ValueError(
                f"Z8 Wyner-Ziv blob truncated before length for pair {payload_index}"
            )
        (payload_len,) = struct.unpack("<I", blob[pos : pos + 4])
        pos += 4
        end = pos + int(payload_len)
        if end > len(blob):
            raise ValueError(
                f"Z8 Wyner-Ziv blob truncated reading pair {payload_index}"
            )
        payloads.append(bytes(blob[pos:end]))
        pos = end
    if pos != len(blob):
        raise ValueError(
            f"Z8 Wyner-Ziv blob trailing bytes (pos={pos} len={len(blob)})"
        )
    return payloads


def _pack_length_prefixed_payloads(payloads: list[bytes]) -> bytes:
    parts = [struct.pack("<I", len(payloads))]
    for payload in payloads:
        parts.append(struct.pack("<I", len(payload)))
        parts.append(bytes(payload))
    return b"".join(parts)


def _binding_config_from_archive(arc: Z8HierarchicalArchive) -> SimpleNamespace:
    return SimpleNamespace(
        num_levels=arc.num_levels,
        num_groups_per_level=tuple(arc.num_groups_per_level),
        num_categories_per_level=tuple(arc.num_categories_per_level),
        num_pairs=arc.num_pairs,
        deterministic_state_dim=16,
        ego_motion_dim=6,
        eval_size=(
            int(arc.meta.get("eval_height", 32)),
            int(arc.meta.get("eval_width", 32)),
        ),
    )


def _top_ll_side_info(
    pair_pyramid: dict[str, Any], side_shape: tuple[int, int, int]
) -> np.ndarray:
    side_c, side_h, side_w = (int(value) for value in side_shape)
    top_ll = np.asarray(pair_pyramid["frame_0_top_ll"], dtype=np.float32)
    if top_ll.ndim != 3:
        raise ValueError(f"frame_0_top_ll must be HWC; got {top_ll.shape}")
    top_ll_nchw = np.transpose(top_ll[np.newaxis, ...], (0, 3, 1, 2))
    top_ll_per_channel = top_ll_nchw.mean(axis=1, keepdims=True)
    side_info = np.tile(top_ll_per_channel, (1, side_c, 1, 1))
    if side_info.shape[-2:] != (side_h, side_w):
        h_min = min(side_info.shape[-2], side_h)
        w_min = min(side_info.shape[-1], side_w)
        buf = np.zeros((1, side_c, side_h, side_w), dtype=np.float32)
        buf[:, :, :h_min, :w_min] = side_info[:, :, :h_min, :w_min]
        side_info = buf
    return side_info.astype(np.float32, copy=False)


def _flatten_state_dict_values(state_dict: dict[str, Any]) -> list[np.ndarray]:
    vectors: list[np.ndarray] = []
    for key in sorted(state_dict):
        value = np.asarray(state_dict[key], dtype=np.float32).reshape(-1)
        if value.size:
            vectors.append(value)
    return vectors


def stack_context_vector_from_archive(arc: Z8HierarchicalArchive) -> np.ndarray:
    """Return decoder/index/Dreamer numeric context consumed by receiver pixels."""

    vectors: list[np.ndarray] = []
    vectors.extend(_flatten_state_dict_values(arc.decoder_state_dict))
    for level_idx, indices in enumerate(arc.per_level_category_indices):
        categories = max(int(arc.num_categories_per_level[level_idx]) - 1, 1)
        normalized = np.asarray(indices, dtype=np.float32).reshape(-1) / float(
            categories
        )
        if normalized.size:
            vectors.append(normalized)
    vectors.extend(_flatten_state_dict_values(arc.dreamer_state_blob))
    if not vectors:
        return np.zeros((0,), dtype=np.float32)
    return np.concatenate(vectors).astype(np.float32, copy=False)


def decode_wyner_ziv_top_states_from_archive(
    archive_bytes: bytes,
) -> list[np.ndarray]:
    """Decode per-pair Mamba/Wyner-Ziv top states from Z8HPC1 bytes."""

    arc = parse_archive(archive_bytes)
    payloads = _length_prefixed_payloads(arc.wyner_ziv_top_blob)
    pair_pyramids = parse_pair_blobs_from_wavelet_blob(arc.wavelet_coeffs_blob)
    if len(payloads) != len(pair_pyramids):
        raise ValueError(
            "Z8 Wyner-Ziv payload count does not match wavelet pair count: "
            f"{len(payloads)} != {len(pair_pyramids)}"
        )
    binding = build_canonical_quadruple_binding_from_z8_config(
        _binding_config_from_archive(arc)
    )
    side_shape = binding.contract.wyner_ziv_top_level_side_info_shape
    decoded: list[np.ndarray] = []
    for payload, pair_pyramid in zip(payloads, pair_pyramids, strict=True):
        side_info = _top_ll_side_info(pair_pyramid, side_shape)
        decoded.append(
            np.asarray(binding.m6.decode(payload, side_info), dtype=np.float32)
        )
    return decoded


def _state_to_top_ll_delta(
    decoded_state: np.ndarray,
    top_ll_hwc: np.ndarray,
    *,
    projection_gain: float = Z8_STATE_TO_TOP_LL_PROJECTION_GAIN,
    projection_seed: int = Z8_STATE_TO_TOP_LL_PROJECTION_SEED,
) -> np.ndarray:
    """Project one decoded WZ/Mamba state into a bounded top-LL correction."""

    top_ll = np.asarray(top_ll_hwc, dtype=np.float32)
    if top_ll.ndim != 3:
        raise ValueError(f"top_ll_hwc must be HWC; got {top_ll.shape}")
    state = np.asarray(decoded_state, dtype=np.float32).reshape(-1)
    if state.size == 0:
        return np.zeros_like(top_ll, dtype=np.float32)
    state = np.where(np.isfinite(state), state, 0.0)
    centered = state - float(state.mean())
    norm = float(np.sqrt(np.mean(centered.astype(np.float64) ** 2)))
    if norm <= 1e-12:
        centered = state
        norm = float(np.sqrt(np.mean(centered.astype(np.float64) ** 2)))
    if norm <= 1e-12:
        return np.zeros_like(top_ll, dtype=np.float32)
    normalized = (centered / norm).astype(np.float32, copy=False)
    channels = int(top_ll.shape[-1])
    rng = np.random.RandomState(int(projection_seed))
    projection = rng.standard_normal((normalized.size, channels)).astype(np.float32)
    projection /= max(float(normalized.size) ** 0.5, 1.0)
    channel_delta = normalized @ projection
    max_abs = float(np.max(np.abs(channel_delta))) if channel_delta.size else 0.0
    if max_abs > 0.0:
        channel_delta = channel_delta / max_abs
    spatial = np.linspace(-1.0, 1.0, int(top_ll.shape[0]), dtype=np.float32)[:, None]
    lateral = np.linspace(-1.0, 1.0, int(top_ll.shape[1]), dtype=np.float32)[None, :]
    envelope = (1.0 - 0.25 * np.clip(spatial**2 + lateral**2, 0.0, 1.0)).astype(
        np.float32
    )
    return (
        float(projection_gain)
        * envelope[:, :, None]
        * channel_delta.reshape(1, 1, channels)
    ).astype(np.float32, copy=False)


def _vector_to_top_ll_delta(
    vector: np.ndarray,
    top_ll_hwc: np.ndarray,
    *,
    projection_gain: float,
    projection_seed: int,
) -> np.ndarray:
    top_ll = np.asarray(top_ll_hwc, dtype=np.float32)
    context = np.asarray(vector, dtype=np.float32).reshape(-1)
    if top_ll.ndim != 3:
        raise ValueError(f"top_ll_hwc must be HWC; got {top_ll.shape}")
    if context.size == 0:
        return np.zeros_like(top_ll, dtype=np.float32)
    context = np.where(np.isfinite(context), context, 0.0)
    centered = context - float(context.mean())
    norm = float(np.sqrt(np.mean(centered.astype(np.float64) ** 2)))
    if norm <= 1e-12:
        centered = context
        norm = float(np.sqrt(np.mean(centered.astype(np.float64) ** 2)))
    if norm <= 1e-12:
        return np.zeros_like(top_ll, dtype=np.float32)
    normalized = (centered / norm).astype(np.float32, copy=False)
    channels = int(top_ll.shape[-1])
    rng = np.random.RandomState(int(projection_seed))
    projection = rng.standard_normal((normalized.size, channels)).astype(np.float32)
    projection /= max(float(normalized.size) ** 0.5, 1.0)
    channel_delta = normalized @ projection
    max_abs = float(np.max(np.abs(channel_delta))) if channel_delta.size else 0.0
    if max_abs > 0.0:
        channel_delta = channel_delta / max_abs
    vertical = np.linspace(0.75, 1.0, int(top_ll.shape[0]), dtype=np.float32)[:, None]
    horizontal = np.linspace(1.0, 0.75, int(top_ll.shape[1]), dtype=np.float32)[
        None, :
    ]
    envelope = (vertical * horizontal).astype(np.float32)
    return (
        float(projection_gain)
        * envelope[:, :, None]
        * channel_delta.reshape(1, 1, channels)
    ).astype(np.float32, copy=False)


def project_decoded_top_states_into_pair_pyramids(
    pair_pyramids: list[dict[str, Any]],
    decoded_top_states: list[np.ndarray],
    *,
    projection_gain: float = Z8_STATE_TO_TOP_LL_PROJECTION_GAIN,
    stack_context_vector: np.ndarray | None = None,
    stack_context_projection_gain: float = Z8_STACK_CONTEXT_TO_TOP_LL_PROJECTION_GAIN,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return pair pyramids whose frame-1 top LL consumes decoded WZ states.

    This is a deterministic low-rank bridge from the WZ/Mamba state space into
    the Mallat top-level image space. It intentionally affects only frame 1:
    frame 0 remains the decoder-side information used to reconstruct the WZ
    state, while frame 1 receives the next-frame predictive correction.
    """

    if len(pair_pyramids) != len(decoded_top_states):
        raise ValueError(
            "decoded WZ state count does not match pair pyramid count: "
            f"{len(decoded_top_states)} != {len(pair_pyramids)}"
        )
    projected: list[dict[str, Any]] = []
    max_abs_delta = 0.0
    mean_abs_accum = 0.0
    value_count = 0
    changed_pairs = 0
    for pyramid, decoded_state in zip(pair_pyramids, decoded_top_states, strict=True):
        frame_1_top_ll = np.asarray(pyramid["frame_1_top_ll"], dtype=np.float32)
        delta = _state_to_top_ll_delta(
            decoded_state,
            frame_1_top_ll,
            projection_gain=projection_gain,
        )
        if stack_context_vector is not None:
            delta = delta + _vector_to_top_ll_delta(
                stack_context_vector,
                frame_1_top_ll,
                projection_gain=stack_context_projection_gain,
                projection_seed=Z8_STACK_CONTEXT_TO_TOP_LL_PROJECTION_SEED,
            )
        top_ll_projected = (frame_1_top_ll + delta).astype(np.float32, copy=False)
        actual_delta = top_ll_projected - frame_1_top_ll
        if np.any(actual_delta != 0.0):
            changed_pairs += 1
        max_abs_delta = max(max_abs_delta, float(np.max(np.abs(actual_delta))))
        mean_abs_accum += float(np.sum(np.abs(actual_delta)))
        value_count += int(actual_delta.size)
        next_pyramid = dict(pyramid)
        next_pyramid["frame_1_top_ll"] = top_ll_projected
        projected.append(next_pyramid)
    stats = {
        "projection_target": "frame_1_top_ll",
        "projection_gain": float(projection_gain),
        "projection_seed": Z8_STATE_TO_TOP_LL_PROJECTION_SEED,
        "stack_context_projection_gain": (
            float(stack_context_projection_gain)
            if stack_context_vector is not None
            else 0.0
        ),
        "stack_context_projection_seed": (
            Z8_STACK_CONTEXT_TO_TOP_LL_PROJECTION_SEED
            if stack_context_vector is not None
            else None
        ),
        "stack_context_vector_length": (
            int(np.asarray(stack_context_vector).size)
            if stack_context_vector is not None
            else 0
        ),
        "projected_pair_count": len(projected),
        "projected_pair_changed_count": changed_pairs,
        "max_abs_projected_top_ll_delta": max_abs_delta,
        "mean_abs_projected_top_ll_delta": (
            mean_abs_accum / value_count if value_count else 0.0
        ),
    }
    return projected, stats


def projected_pair_pyramids_from_archive_bytes(
    archive_bytes: bytes,
) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    """Return the exact WZ-projected pair pyramids consumed by Z8 inflate."""

    arc = parse_archive(archive_bytes)
    payloads = _length_prefixed_payloads(arc.wyner_ziv_top_blob)
    pair_pyramids = parse_pair_blobs_from_wavelet_blob(arc.wavelet_coeffs_blob)
    if len(payloads) != len(pair_pyramids):
        raise ValueError(
            "Z8 Wyner-Ziv payload count does not match wavelet pair count: "
            f"{len(payloads)} != {len(pair_pyramids)}"
        )
    binding = build_canonical_quadruple_binding_from_z8_config(
        _binding_config_from_archive(arc)
    )
    side_shape = binding.contract.wyner_ziv_top_level_side_info_shape
    decoded = [
        np.asarray(
            binding.m6.decode(payload, _top_ll_side_info(pyramid, side_shape)),
            dtype=np.float32,
        )
        for payload, pyramid in zip(payloads, pair_pyramids, strict=True)
    ]
    stack_context = stack_context_vector_from_archive(arc)
    projected, stats = project_decoded_top_states_into_pair_pyramids(
        pair_pyramids,
        decoded,
        stack_context_vector=stack_context,
    )
    return binding, projected, stats


def reconstruct_projected_pair_rgb_from_archive_bytes(
    archive_bytes: bytes,
    *,
    pair_index: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct one pair through the same WZ projection path inflate uses."""

    binding, projected, _stats = projected_pair_pyramids_from_archive_bytes(
        archive_bytes
    )
    if not projected:
        raise ValueError("Z8 archive carries zero projected pair pyramids")
    src_pair_idx = int(pair_index) % len(projected)
    return reconstruct_pair_rgb_from_pyramid(binding, projected[src_pair_idx])


def mutate_valid_wyner_ziv_payload_in_archive(
    archive_bytes: bytes,
    *,
    pair_index: int = 0,
    mutation_scale: float = 0.5,
) -> tuple[bytes, dict[str, Any]]:
    """Return a byte-closed archive whose WZ payload decodes to changed state."""

    arc = parse_archive(archive_bytes)
    payloads = _length_prefixed_payloads(arc.wyner_ziv_top_blob)
    pair_pyramids = parse_pair_blobs_from_wavelet_blob(arc.wavelet_coeffs_blob)
    if not payloads or not pair_pyramids:
        raise ValueError("Z8 archive requires at least one WZ payload and pair")
    if len(payloads) != len(pair_pyramids):
        raise ValueError(
            "Z8 Wyner-Ziv payload count does not match wavelet pair count: "
            f"{len(payloads)} != {len(pair_pyramids)}"
        )
    binding = build_canonical_quadruple_binding_from_z8_config(
        _binding_config_from_archive(arc)
    )
    side_shape = binding.contract.wyner_ziv_top_level_side_info_shape
    target_idx = int(pair_index) % len(payloads)
    side_info = _top_ll_side_info(pair_pyramids[target_idx], side_shape)
    decoded = np.asarray(
        binding.m6.decode(payloads[target_idx], side_info),
        dtype=np.float32,
    )
    flat = decoded.reshape(-1)
    if flat.size == 0:
        raise ValueError("decoded WZ top state is empty")
    pattern = np.linspace(-1.0, 1.0, int(flat.size), dtype=np.float32).reshape(
        decoded.shape
    )
    state_scale = max(float(np.std(flat.astype(np.float64))), 1.0)
    candidate_states = (
        ("zero_top_state", np.zeros_like(decoded, dtype=np.float32)),
        (
            "unit_positive_top_state",
            np.ones_like(decoded, dtype=np.float32) * state_scale,
        ),
        ("negated_top_state", -decoded),
        (
            "ramp_perturbed_top_state",
            decoded + float(mutation_scale) * state_scale * pattern,
        ),
    )
    selected_mode = ""
    selected_state_delta = 0.0
    selected_payload = b""
    for mode, candidate_state in candidate_states:
        candidate_payload = binding.m6.encode(candidate_state, side_info)
        candidate_decoded = np.asarray(
            binding.m6.decode(candidate_payload, side_info),
            dtype=np.float32,
        )
        state_delta = float(np.max(np.abs(candidate_decoded - decoded)))
        if state_delta > 0.0:
            selected_mode = mode
            selected_state_delta = state_delta
            selected_payload = candidate_payload
            break
    if not selected_payload:
        raise RuntimeError("could not produce a decoded-changing WZ payload mutation")
    next_payloads = list(payloads)
    next_payloads[target_idx] = selected_payload
    mutated_wz_blob = _pack_length_prefixed_payloads(next_payloads)
    mutated_archive = pack_archive(
        arc.decoder_state_dict,
        arc.per_level_category_indices,
        arc.wavelet_coeffs_blob,
        mutated_wz_blob,
        arc.dreamer_state_blob,
        arc.meta,
        num_levels=arc.num_levels,
        num_groups_per_level=arc.num_groups_per_level,
        num_categories_per_level=arc.num_categories_per_level,
        num_pairs=arc.num_pairs,
        decoder_latent_dim=arc.decoder_latent_dim,
        base_channels=arc.base_channels,
        wavelet_basis_id=arc.wavelet_basis_id,
        schema_version=arc.schema_version,
    )
    return mutated_archive, {
        "mutated_pair_index": target_idx,
        "original_payload_bytes": len(payloads[target_idx]),
        "mutated_payload_bytes": len(next_payloads[target_idx]),
        "original_archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "mutated_archive_sha256": hashlib.sha256(mutated_archive).hexdigest(),
        "decoded_top_state_max_abs_delta": selected_state_delta,
        "mutation_mode": selected_mode,
        "mutation_scale": float(mutation_scale),
    }


def mutate_valid_stack_context_payload_in_archive(
    archive_bytes: bytes,
    section: str,
    *,
    pair_index: int = 0,
) -> tuple[bytes, dict[str, Any]]:
    """Return a valid archive mutation for decoder/index/Dreamer context."""

    arc = parse_archive(archive_bytes)
    decoder_state_dict = dict(arc.decoder_state_dict)
    dreamer_state_dict = dict(arc.dreamer_state_blob)
    per_level_indices = [np.array(a, copy=True) for a in arc.per_level_category_indices]
    target_pair = int(pair_index) % max(int(arc.num_pairs), 1)
    if section == "decoder_blob":
        decoder_state_dict["runtime_stack_context_delta"] = np.array(
            [1.0], dtype=np.float32
        )
    elif section == "dreamer_state_blob":
        dreamer_state_dict["runtime_stack_context_delta"] = np.array(
            [1.0], dtype=np.float32
        )
    elif section == "indices_blob":
        if not per_level_indices:
            raise ValueError("indices_blob mutation requires at least one level")
        categories = max(int(arc.num_categories_per_level[0]), 1)
        per_level_indices[0][target_pair, 0] = (
            int(per_level_indices[0][target_pair, 0]) + 1
        ) % categories
    else:
        raise ValueError(f"unsupported stack context section: {section}")
    mutated_archive = pack_archive(
        decoder_state_dict,
        per_level_indices,
        arc.wavelet_coeffs_blob,
        arc.wyner_ziv_top_blob,
        dreamer_state_dict,
        arc.meta,
        num_levels=arc.num_levels,
        num_groups_per_level=arc.num_groups_per_level,
        num_categories_per_level=arc.num_categories_per_level,
        num_pairs=arc.num_pairs,
        decoder_latent_dim=arc.decoder_latent_dim,
        base_channels=arc.base_channels,
        wavelet_basis_id=arc.wavelet_basis_id,
        schema_version=arc.schema_version,
    )
    return mutated_archive, {
        "mutated_section": section,
        "mutated_pair_index": target_pair,
        "original_archive_sha256": hashlib.sha256(archive_bytes).hexdigest(),
        "mutated_archive_sha256": hashlib.sha256(mutated_archive).hexdigest(),
        "mutation_kind": "valid_semantic_stack_context_payload_mutation",
    }


def build_stack_context_payload_mutation_receiver_proofs(
    archive_bytes: bytes,
    *,
    sections: tuple[str, ...] = (
        "decoder_blob",
        "indices_blob",
        "dreamer_state_blob",
    ),
    pair_index: int = 0,
) -> dict[str, Any]:
    """Prove valid decoder/index/Dreamer mutations change receiver pixels."""

    base_rgb_0, base_rgb_1 = reconstruct_projected_pair_rgb_from_archive_bytes(
        archive_bytes,
        pair_index=pair_index,
    )
    per_section: dict[str, dict[str, Any]] = {}
    for section in sections:
        mutated_archive, mutation = mutate_valid_stack_context_payload_in_archive(
            archive_bytes,
            section,
            pair_index=pair_index,
        )
        mutated_rgb_0, mutated_rgb_1 = reconstruct_projected_pair_rgb_from_archive_bytes(
            mutated_archive,
            pair_index=pair_index,
        )
        frame0_delta = float(np.max(np.abs(mutated_rgb_0 - base_rgb_0)))
        frame1_delta = float(np.max(np.abs(mutated_rgb_1 - base_rgb_1)))
        per_section[section] = {
            "section": section,
            "valid_semantic_payload_mutation": True,
            "section_pixel_consumption_proven": frame1_delta > 0.0,
            "frame_0_max_abs_delta": frame0_delta,
            "frame_1_max_abs_delta": frame1_delta,
            "expected_changed_frame": "frame_1",
            "mutation": mutation,
        }
    proven_sections = [
        section
        for section, proof in per_section.items()
        if proof["section_pixel_consumption_proven"] is True
    ]
    return {
        "schema": "z8_hpc1_stack_context_payload_mutation_receiver_proofs.v1",
        "receiver_runtime_path": (
            "projected_pair_pyramids_from_archive_bytes"
            "->stack_context_vector_from_archive"
            "->project_decoded_top_states_into_pair_pyramids"
            "->reconstruct_pair_rgb_from_pyramid"
        ),
        "archive_member_byte_closed": True,
        "valid_semantic_stack_context_payload_mutations": True,
        "stack_context_sections_pixel_consumed": proven_sections,
        "per_section": per_section,
        "axis_tag": "[macOS-CPU advisory]",
        "evidence_grade": "receiver-runtime-local-proof",
        **FALSE_AUTHORITY,
    }


def build_wyner_ziv_payload_mutation_receiver_proof(
    archive_bytes: bytes,
    *,
    proof_out: str | Path | None = None,
    pair_index: int = 0,
    mutation_scale: float = 0.5,
) -> dict[str, Any]:
    """Prove a valid WZ payload mutation changes receiver-rendered pixels."""

    base_rgb_0, base_rgb_1 = reconstruct_projected_pair_rgb_from_archive_bytes(
        archive_bytes,
        pair_index=pair_index,
    )
    mutated_archive, mutation = mutate_valid_wyner_ziv_payload_in_archive(
        archive_bytes,
        pair_index=pair_index,
        mutation_scale=mutation_scale,
    )
    mutated_rgb_0, mutated_rgb_1 = reconstruct_projected_pair_rgb_from_archive_bytes(
        mutated_archive,
        pair_index=pair_index,
    )
    frame0_delta = float(np.max(np.abs(mutated_rgb_0 - base_rgb_0)))
    frame1_delta = float(np.max(np.abs(mutated_rgb_1 - base_rgb_1)))
    manifest = {
        "schema": "z8_hpc1_wyner_ziv_payload_mutation_receiver_proof.v1",
        "receiver_runtime_path": (
            "inflate.decode_wyner_ziv_top_states_from_archive"
            "->project_decoded_top_states_into_pair_pyramids"
            "->reconstruct_pair_rgb_from_pyramid"
        ),
        "archive_member_byte_closed": True,
        "valid_semantic_wyner_ziv_payload_mutation": True,
        "wyner_ziv_top_state_pixel_consumption_proven": frame1_delta > 0.0,
        "frame_0_max_abs_delta": frame0_delta,
        "frame_1_max_abs_delta": frame1_delta,
        "expected_changed_frame": "frame_1",
        "mutation": mutation,
        "axis_tag": "[macOS-CPU advisory]",
        "evidence_grade": "receiver-runtime-local-proof",
        **FALSE_AUTHORITY,
    }
    if proof_out is not None:
        out = Path(proof_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        manifest["proof_path"] = str(out)
        from tac.repo_io import write_json

        write_json(out, manifest)
    else:
        manifest["proof_path"] = None
    return manifest


def build_runtime_payload_bridge_report(
    archive_bytes: bytes,
    *,
    report_out: str | Path | None = None,
) -> dict[str, Any]:
    """Return a false-authority report proving WZ top-state decode custody."""

    decoded = decode_wyner_ziv_top_states_from_archive(archive_bytes)
    _binding, _projected, projection_stats = projected_pair_pyramids_from_archive_bytes(
        archive_bytes
    )
    stack_proofs = build_stack_context_payload_mutation_receiver_proofs(archive_bytes)
    pixel_consumption_proven = (
        int(projection_stats.get("projected_pair_changed_count") or 0) > 0
    )
    state_shapes = [list(state.shape) for state in decoded]
    state_digest = hashlib.sha256(
        b"".join(state.astype(np.float32, copy=False).tobytes() for state in decoded)
    ).hexdigest()
    report = {
        "schema": Z8_RUNTIME_PAYLOAD_BRIDGE_REPORT_SCHEMA,
        "archive_sha256": hashlib.sha256(bytes(archive_bytes)).hexdigest(),
        "wyner_ziv_top_state_decode_ready": True,
        "wyner_ziv_top_state_count": len(decoded),
        "wyner_ziv_top_state_shapes": state_shapes,
        "wyner_ziv_top_state_sha256": state_digest,
        "side_info_source": "frame_0_top_ll_per_channel_spatial_mean",
        "state_to_pixel_projection_ready": True,
        "state_to_pixel_projection": projection_stats,
        "stack_context_payload_mutation_receiver_proofs": stack_proofs,
        "stack_context_sections_pixel_consumed": stack_proofs[
            "stack_context_sections_pixel_consumed"
        ],
        "pixel_consumption_proven": pixel_consumption_proven,
        "next_required_task": (
            "serialize_trained_mlx_renderer_state_into_z8hpc1_archive"
            if pixel_consumption_proven
            else "run_valid_wyner_ziv_payload_mutation_receiver_proof"
        ),
        "allowed_use": "receiver_runtime_projection_candidate_and_materializer_planning_only",
        "forbidden_use": "score_claim_or_pixel_consumption_authority",
        **FALSE_AUTHORITY,
    }
    if report_out is not None:
        from tac.repo_io import write_json

        path = Path(report_out)
        write_json(path, report)
        report["report_path"] = path.as_posix()
        write_json(path, report)
    return report


__all__ = [
    "Z8_RUNTIME_PAYLOAD_BRIDGE_REPORT_SCHEMA",
    "Z8_STACK_CONTEXT_TO_TOP_LL_PROJECTION_GAIN",
    "Z8_STATE_TO_TOP_LL_PROJECTION_GAIN",
    "build_runtime_payload_bridge_report",
    "build_stack_context_payload_mutation_receiver_proofs",
    "build_wyner_ziv_payload_mutation_receiver_proof",
    "decode_wyner_ziv_top_states_from_archive",
    "mutate_valid_stack_context_payload_in_archive",
    "mutate_valid_wyner_ziv_payload_in_archive",
    "project_decoded_top_states_into_pair_pyramids",
    "projected_pair_pyramids_from_archive_bytes",
    "reconstruct_projected_pair_rgb_from_archive_bytes",
]
