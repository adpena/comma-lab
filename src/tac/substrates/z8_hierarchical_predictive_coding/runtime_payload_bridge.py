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

from tac.repo_io import write_json
from tac.substrates.z8_hierarchical_predictive_coding.archive import (
    Z8HierarchicalArchive,
    parse_archive,
)
from tac.substrates.z8_hierarchical_predictive_coding.canonical_quadruple_binding import (
    build_canonical_quadruple_binding_from_z8_config,
    parse_pair_blobs_from_wavelet_blob,
)

Z8_RUNTIME_PAYLOAD_BRIDGE_REPORT_SCHEMA = "z8_hpc1_runtime_payload_bridge_report.v1"
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


def build_runtime_payload_bridge_report(
    archive_bytes: bytes,
    *,
    report_out: str | Path | None = None,
) -> dict[str, Any]:
    """Return a false-authority report proving WZ top-state decode custody."""

    decoded = decode_wyner_ziv_top_states_from_archive(archive_bytes)
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
        "pixel_consumption_proven": False,
        "state_to_pixel_projection_ready": False,
        "next_required_task": "fit_and_archive_state_to_top_ll_projection",
        "allowed_use": "runtime_payload_decode_custody_and_materializer_planning_only",
        "forbidden_use": "score_claim_or_pixel_consumption_authority",
        **FALSE_AUTHORITY,
    }
    if report_out is not None:
        path = Path(report_out)
        write_json(path, report)
        report["report_path"] = path.as_posix()
        write_json(path, report)
    return report


__all__ = [
    "Z8_RUNTIME_PAYLOAD_BRIDGE_REPORT_SCHEMA",
    "build_runtime_payload_bridge_report",
    "decode_wyner_ziv_top_states_from_archive",
]
