# SPDX-License-Identifier: MIT
"""Runtime payload bridge for Z8 Mamba/Wyner-Ziv archive sections.

This module makes the non-wavelet Z8HPC1 payloads typed at receiver runtime
without pretending they are pixel-consuming yet. The bridge decodes the
per-pair Wyner-Ziv top states from archive bytes using the same top-LL side
information model as the encoder. A later state-to-pixel adapter can consume
these decoded states; until then the report remains false-authority custody
and planning signal only.
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
    parse_archive,
)
from tac.substrates.z8_hierarchical_predictive_coding.canonical_quadruple_binding import (
    build_canonical_quadruple_binding_from_z8_config,
    parse_pair_blobs_from_wavelet_blob,
)

Z8_RUNTIME_PAYLOAD_BRIDGE_REPORT_SCHEMA = "z8_hpc1_runtime_payload_bridge_report.v1"
Z8_STATE_TO_TOP_LL_PROJECTION_GAIN = 1.0 / 64.0
Z8_STATE_TO_TOP_LL_PROJECTION_SEED = 23
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


def project_decoded_top_states_into_pair_pyramids(
    pair_pyramids: list[dict[str, Any]],
    decoded_top_states: list[np.ndarray],
    *,
    projection_gain: float = Z8_STATE_TO_TOP_LL_PROJECTION_GAIN,
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
        top_ll_projected = np.clip(frame_1_top_ll + delta, 0.0, 1.0).astype(
            np.float32, copy=False
        )
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
        "projected_pair_count": len(projected),
        "projected_pair_changed_count": changed_pairs,
        "max_abs_projected_top_ll_delta": max_abs_delta,
        "mean_abs_projected_top_ll_delta": (
            mean_abs_accum / value_count if value_count else 0.0
        ),
    }
    return projected, stats


def build_runtime_payload_bridge_report(
    archive_bytes: bytes,
    *,
    report_out: str | Path | None = None,
) -> dict[str, Any]:
    """Return a false-authority report proving WZ top-state decode custody."""

    decoded = decode_wyner_ziv_top_states_from_archive(archive_bytes)
    arc = parse_archive(archive_bytes)
    pair_pyramids = parse_pair_blobs_from_wavelet_blob(arc.wavelet_coeffs_blob)
    _projected, projection_stats = project_decoded_top_states_into_pair_pyramids(
        pair_pyramids, decoded
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
        "pixel_consumption_proven": False,
        "next_required_task": "run_valid_wyner_ziv_payload_mutation_receiver_proof",
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
    "Z8_STATE_TO_TOP_LL_PROJECTION_GAIN",
    "build_runtime_payload_bridge_report",
    "decode_wyner_ziv_top_states_from_archive",
    "project_decoded_top_states_into_pair_pyramids",
]
