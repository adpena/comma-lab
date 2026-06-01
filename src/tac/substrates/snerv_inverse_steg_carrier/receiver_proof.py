# SPDX-License-Identifier: MIT
"""Scorer-free receiver proof for the SNeRV archive grammar."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from tac.analysis.snerv_step_map_coder import decode_step_maps, encode_step_maps
from tac.substrates.snerv_inverse_steg_carrier.archive import (
    SnervArchivePacket,
    encode_decoder_payload,
    encode_lf_metadata_payload,
    encode_lf_quant_payload,
    pack_snerv_archive,
    unpack_snerv_archive,
)
from tac.substrates.snerv_inverse_steg_carrier.carrier import (
    SnervFrameCode,
    decode_frame,
    encode_frame_lf,
    fit_hf_decoder_least_squares,
    quantize_lf,
)

SCHEMA = "snerv_receiver_archive_proof.v1"
AXIS_TAG = "[receiver-proof:false-authority]"


@dataclass(frozen=True)
class SnervReceiverArchiveProof:
    """Machine-readable proof that receiver archive sections reconstruct frames."""

    schema: str
    axis_tag: str
    receiver_contract_satisfied: bool
    runtime_consumption_proof_ready: bool
    archive_packet_bytes: int
    archive_packet_sha256: str
    section_bytes: dict[str, int]
    section_sha256: dict[str, str]
    step_map_codec: str
    step_map_bins: int
    step_map_packet_bytes: int
    lf_plane_count: int
    levels: int
    wavelet: str
    orig_hw: tuple[int, int]
    receiver_output_shape: tuple[int, ...]
    receiver_output_bytes: int
    receiver_output_sha256: str
    direct_output_sha256: str
    receiver_matches_direct: bool
    max_abs_diff: float
    score_claim: bool = False
    frontier_score_claim: bool = False
    promotion_eligible: bool = False
    rank_or_kill_eligible: bool = False
    ready_for_exact_eval_dispatch: bool = False
    blockers: tuple[str, ...] = (
        "toy_receiver_proof_not_full_600_pair_replay",
        "not_packaged_as_contest_archive_zip",
        "paired_contest_cpu_cuda_auth_eval_missing",
    )

    def as_jsonable(self) -> dict[str, Any]:
        d = asdict(self)
        d["orig_hw"] = list(self.orig_hw)
        d["receiver_output_shape"] = list(self.receiver_output_shape)
        return d


def build_snerv_receiver_archive_proof(
    *,
    bins: int = 4,
    levels: int = 2,
    hw: tuple[int, int] = (32, 48),
    seed: int = 7,
    full_frame_packet: bool = False,
    n_pairs: int = 1,
) -> tuple[SnervReceiverArchiveProof, SnervArchivePacket]:
    """Build a deterministic scorer-free receiver proof packet.

    The proof constructs a tiny SNeRV archive packet, decodes every receiver
    section through the archive API, reconstructs a frame from decoded LF/decoder/
    step-map state, and checks it matches direct reconstruction from the same
    archive-visible decoded state.
    """

    if full_frame_packet:
        frames_tensor = _deterministic_rgb_pair_frames(
            n_pairs=n_pairs,
            hw=hw,
            seed=seed,
        )
        frames = [
            frames_tensor[pair_idx, frame_idx, channel_idx]
            for pair_idx in range(int(n_pairs))
            for frame_idx in range(2)
            for channel_idx in range(3)
        ]
    else:
        frames_tensor = None
        frames = _deterministic_frames(hw=hw, seed=seed)
    pyrs = [encode_frame_lf(frame, levels=levels) for frame in frames]
    decoder = fit_hf_decoder_least_squares(pyrs, levels=levels)
    step_maps = [
        _deterministic_step_map(pyr.lf.shape, phase=float(idx) * 0.17)
        for idx, pyr in enumerate(pyrs)
    ]
    step_packet = encode_step_maps(step_maps, bins=bins)
    receiver_step_maps = decode_step_maps(step_packet.packet)
    q_planes = []
    zero_points = []
    for pyr, receiver_steps in zip(pyrs, receiver_step_maps, strict=True):
        q, _scale, zero = quantize_lf(pyr.lf, per_element_steps=receiver_steps)
        q_planes.append(q)
        zero_points.append(zero)
    decoder_payload = encode_decoder_payload(decoder)
    metadata: dict[str, Any] = {
        "lf_plane_count": len(q_planes),
        "levels": levels,
        "wavelet": pyrs[0].wavelet,
        "orig_hw": list(hw),
        "receiver_proof_seed": seed,
        "step_map_bins": bins,
    }
    if full_frame_packet:
        metadata.update(
            {
                "n_pairs": int(n_pairs),
                "frames_per_pair": 2,
                "channels": 3,
                "carrier_hw": list(hw),
            }
        )
    archive = pack_snerv_archive(
        metadata_payload=encode_lf_metadata_payload(lf_zero_points=zero_points),
        lf_payload=encode_lf_quant_payload(q_planes),
        decoder_payload=decoder_payload,
        step_map_packet=step_packet.packet,
        metadata=metadata,
    )
    decoded = unpack_snerv_archive(archive.packet)
    receiver_decoder = decoded.decode_decoder()
    receiver_planes = decoded.decode_frame_planes()
    direct_planes = []
    for q, zero, steps in zip(
        decoded.decode_lf_quant_planes(),
        decoded.decode_lf_zero_points(),
        decoded.decode_step_maps(),
        strict=True,
    ):
        direct_code = SnervFrameCode(
            lf_quant=q,
            lf_scale=1.0,
            lf_zero=float(zero),
            lf_shape=q.shape,
            levels=levels,
            wavelet=pyrs[0].wavelet,
            orig_hw=hw,
            per_element_steps=steps,
        )
        direct_planes.append(
            np.clip(decode_frame(direct_code, receiver_decoder), 0.0, 255.0).astype(
                np.float32
            )
        )
    if full_frame_packet:
        receiver_output = decoded.decode_frames()
        direct_output = np.stack(direct_planes, axis=0).reshape(
            int(n_pairs),
            2,
            3,
            int(hw[0]),
            int(hw[1]),
        )
    else:
        receiver_output = receiver_planes[0]
        direct_output = direct_planes[0]
    max_abs_diff = float(np.max(np.abs(receiver_output - direct_output)))
    receiver_matches = bool(np.allclose(receiver_output, direct_output, atol=0.0, rtol=0.0))
    receiver_bytes = _array_bytes(receiver_output)
    direct_bytes = _array_bytes(direct_output)
    proof = SnervReceiverArchiveProof(
        schema=SCHEMA,
        axis_tag=AXIS_TAG,
        receiver_contract_satisfied=receiver_matches,
        runtime_consumption_proof_ready=receiver_matches,
        archive_packet_bytes=archive.total_bytes,
        archive_packet_sha256=_sha256(archive.packet),
        section_bytes=dict(archive.section_bytes),
        section_sha256=dict(archive.section_sha256),
        step_map_codec=step_packet.schema,
        step_map_bins=step_packet.bins,
        step_map_packet_bytes=step_packet.total_bytes,
        lf_plane_count=len(q_planes),
        levels=levels,
        wavelet=pyrs[0].wavelet,
        orig_hw=hw,
        receiver_output_shape=tuple(int(v) for v in receiver_output.shape),
        receiver_output_bytes=len(receiver_bytes),
        receiver_output_sha256=_sha256(receiver_bytes),
        direct_output_sha256=_sha256(direct_bytes),
        receiver_matches_direct=receiver_matches,
        max_abs_diff=max_abs_diff,
    )
    return proof, archive


def _deterministic_frames(
    *,
    hw: tuple[int, int],
    seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    h, w = hw
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    return [
        np.clip(
            120.0
            + 25.0 * np.sin(xx / 7.0 + i * 0.15)
            + 15.0 * np.cos(yy / 5.0)
            + rng.standard_normal((h, w)) * 0.5,
            0.0,
            255.0,
        )
        for i in range(4)
    ]


def _deterministic_rgb_pair_frames(
    *,
    n_pairs: int,
    hw: tuple[int, int],
    seed: int,
) -> np.ndarray:
    if int(n_pairs) < 1:
        raise ValueError(f"n_pairs must be >= 1, got {n_pairs}")
    rng = np.random.default_rng(seed)
    h, w = hw
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    out = np.empty((int(n_pairs), 2, 3, h, w), dtype=np.float32)
    for pair_idx in range(int(n_pairs)):
        for frame_idx in range(2):
            for channel_idx in range(3):
                out[pair_idx, frame_idx, channel_idx] = np.clip(
                    112.0
                    + 19.0 * np.sin(xx / (5.0 + channel_idx) + pair_idx * 0.13)
                    + 11.0 * np.cos(yy / (4.0 + frame_idx) + channel_idx * 0.07)
                    + 3.0 * channel_idx
                    + 1.5 * frame_idx
                    + rng.standard_normal((h, w)) * 0.25,
                    0.0,
                    255.0,
                )
    return out


def _deterministic_step_map(
    shape: tuple[int, int],
    *,
    phase: float = 0.0,
) -> np.ndarray:
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float32)
    steps = np.exp2(
        0.5
        + 0.08 * np.sin(xx / 5.0 + phase)
        + 0.04 * np.cos(yy / 3.0 - phase)
    )
    steps[0, 0] *= 4.0
    return steps.astype(np.float32)


def _array_bytes(arr: np.ndarray) -> bytes:
    return np.asarray(arr, dtype="<f4").tobytes()


def _sha256(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()
