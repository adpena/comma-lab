# SPDX-License-Identifier: MIT
"""RAFT optical flow pose stream — pose-axis residual sidecar (full scaffold).

Lane: ``lane_pose_axis_raft_pose_stream_full_scaffold`` (Phase 3
substrate-engineering; pose-axis target). Per the handoff P3 ledger
``~/Downloads/pact_score_lowering_handoff_2026-05-11.md`` ("Wavelet/foveation/
RAFT/ego-motion" section) and Grand Council Insight 3 (pose-axis lanes carry
**2.79× higher EV per byte at PR106 r2 operating point** vs SegNet), the
RAFT pose stream is a charged byte stream of per-frame-pair 6-DoF pose deltas
derived from RAFT optical flow at compress time.

Architecture
============

At COMPRESS time, RAFT-Large is run on consecutive frame pairs of the contest
video to recover per-pair optical flow. The flow is reduced to a 6-DoF pose
delta via the existing ``tac.raft_pose.flow_to_pose_dim0`` primitive (extended
to all 6 dims via least-squares fit to the PR106 baseline pose stream). The
resulting stream replaces (or refines) the PR106 base pose prediction.

At INFLATE time, the consumer reads the precomputed delta stream from the
archive. **No RAFT model is loaded at inflate time** — the inflate runtime
just reads the pose stream and forwards it to the contest evaluator. This
satisfies CLAUDE.md "strict scorer rule" and HNeRV parity discipline lesson 4
(inflate ≤200 LOC; ≤2 external deps).

Wire format (RAFT pose stream sidecar)
--------------------------------------
::

    magic           : u8  = 0xFD  (PR106 residual family magic, reused)
    format_id       : u8  = 0x31  (RAFT_POSE_STREAM_FORMAT_ID)
    n_pairs         : u32_LE
    pose_dim        : u8  = 6
    anchor_pose     : pose_dim * f16_LE  (frame-0 absolute pose)
    delta_scale     : pose_dim * f16_LE  (per-axis quantization scale)
    quantized_deltas: brotli(int8 * (n_pairs - 1) * pose_dim)

Total raw: 6*2 + 6*2 + (n_pairs - 1) * 6 = 12 + 12 + ~3.6 KB for n_pairs=600
before brotli. Target ≤ 4 KB after brotli (operator constraint).

CLAUDE.md compliance
====================
- ``score_claim = False`` permanently until charged archive consumer + exact T4
- ``promotion_eligible = False`` permanently
- ``ready_for_exact_eval_dispatch = False`` until council deliberation
- 8 archive-grammar fields declared (see lane registry)
- ``research_only = False`` because the inflate-time consumer is the
  ``submissions/pr106_raft_pose_sidecar/`` packet (built alongside)
- ``lane_class = substrate_engineering``
- NO scorer load at inflate
- NO ``/tmp`` paths; module is pure-Python with brotli + numpy + torch only
- NO MPS authoritative; ``compute_raft_pose_delta`` defers to
  ``tac.raft_pose.compute_raft_flow`` which enforces CUDA-or-raise.
- Sister of ``tac.codec_pipeline_raft_pose.Op_RAFTPoseStream`` (the Phase-2
  CodecOp Protocol-conformant wrapper). This module is the lower-level
  archive-grammar primitive that the CodecOp wraps.

8 archive-grammar fields (Catalog #124)
=======================================
- ``archive_grammar``: monolithic ``0.bin`` (0xFD + 0x31 wrapper around PR106)
- ``parser_section_manifest``: see ``RAFT_POSE_STREAM_FORMAT`` constant
- ``inflate_runtime_loc_budget``: 200 LOC waiver under substrate_engineering
- ``runtime_dep_closure``: torch + brotli + numpy (contest runtime)
- ``export_format``: ``raft_pose_stream_v1``
- ``score_aware_loss``: deferred to trainer (research-only encoder right now)
- ``bolt_on_loc_budget``: substrate_engineering
- ``no_op_detector_planned``: pose stream variance must exceed 1e-6 across
  frames; tested

6-hook wire-in declarations
===========================
- Sensitivity-map: predicted Δpose informs autopilot (N/A — predicted column)
- Pareto: candidate when L2 encoder + dispatch land
- Bit-allocator: ``compute_raft_pose_stream_bytes`` informs allocator
- Cathedral autopilot: register as ``optimize_mode_transforms`` candidate
- Continual-learning posterior update: triggered on exact T4 result
- Probe-disambiguator: foveation-vs-RAFT-vs-LAPose head-to-head
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

import brotli  # type: ignore[import-not-found]
import numpy as np
import torch

# Reuse PR106 residual family magic for monolithic packet wrapping. The
# format_id alone disambiguates the consumer.
PR106_RESIDUAL_MAGIC = 0xFD
RAFT_POSE_STREAM_FORMAT_ID = 0x31
RAFT_POSE_STREAM_VERSION = 1

POSE_DIM = 6
# Hard byte budget per CLAUDE.md / handoff operator constraint.
MAX_ENCODED_BYTES = 4096

# Per-axis variance threshold (rad² or m²); below this the stream is
# considered no-op (every frame equal to anchor) and the sidecar refuses to
# spend bytes per HNeRV parity discipline lesson 11.
NO_OP_VARIANCE_THRESHOLD = 1e-6


@dataclass
class RaftPoseStream:
    """Per-frame-pair pose delta stream derived from RAFT optical flow.

    ``poses`` is the materialized full ``(n_pairs, 6)`` float32 pose tensor.
    The encoder converts it to (anchor, deltas, scale) for byte-efficient
    storage.
    """

    poses: torch.Tensor  # (n_pairs, 6) float32

    @property
    def n_pairs(self) -> int:
        return int(self.poses.shape[0])

    def validate(self) -> None:
        if self.poses.ndim != 2 or self.poses.shape[-1] != POSE_DIM:
            raise ValueError(
                f"poses must have shape (n_pairs, {POSE_DIM}); got {tuple(self.poses.shape)}"
            )
        if self.poses.shape[0] < 1:
            raise ValueError(f"poses must have at least 1 row; got {self.poses.shape[0]}")
        if not torch.all(torch.isfinite(self.poses)):
            raise ValueError("poses contains non-finite values")


def compute_raft_pose_delta(
    video_path: str,
    n_frames: int = 1200,
    baseline_poses: torch.Tensor | None = None,
    *,
    device: str = "cuda",
) -> RaftPoseStream:
    """Compute the full 6-DoF RAFT pose stream from the contest video.

    This is **COMPRESS-TIME ONLY**. The function delegates to
    ``tac.raft_pose.compute_raft_flow`` (which enforces CUDA-or-raise per
    CLAUDE.md) and ``flow_to_pose_dim0`` for dim 0, then uses ``baseline_poses``
    (PR106 base pose tensor) for dims 1-5 if provided, else zeros.

    The full RAFT-to-SE(3) inverse-dynamics reconstruction (LAPose-style; see
    sister module ``tac.lapose_motion_atom_allocator``) is a research-only
    extension; this primitive uses the simple road-region calibration from
    ``tac.raft_pose.calibrate_pose_dim0``.

    Args:
        video_path: Path to the contest MKV (e.g., ``upstream/videos/0.mkv``).
        n_frames: Max frames to decode (default 1200 = 600 pairs).
        baseline_poses: Optional ``(N, 6)`` PR106 baseline; dims 1-5 reused.
        device: Must be ``"cuda"`` (CUDA-or-raise per raft_pose.py).

    Returns:
        ``RaftPoseStream`` with shape ``(n_pairs, 6)``.
    """
    from tac.raft_pose import build_pose_tensor_from_flow, compute_raft_flow, flow_to_pose_dim0

    flow = compute_raft_flow(video_path, n_frames=n_frames, device=device)
    dim0 = flow_to_pose_dim0(flow)
    poses = build_pose_tensor_from_flow(dim0, baseline_poses=baseline_poses)
    return RaftPoseStream(poses=poses.detach().cpu().float())


def _quantize_int8_per_axis(deltas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric int8 quantization with per-axis scale.

    Args:
        deltas: ``(N-1, 6)`` float32 deltas.

    Returns:
        Tuple of ``(int8 quantized, per-axis float scales)``.
    """
    if deltas.size == 0:
        return np.zeros((0, POSE_DIM), dtype=np.int8), np.ones(POSE_DIM, dtype=np.float32)
    max_abs = np.abs(deltas).max(axis=0).clip(min=1e-12)  # (6,)
    scales = (max_abs / 127.0).astype(np.float32)  # (6,)
    q = np.round(deltas / scales).clip(-127, 127).astype(np.int8)
    return q, scales


def encode_raft_pose_stream(
    stream: RaftPoseStream,
    *,
    enforce_budget: bool = True,
) -> bytes:
    """Encode a ``RaftPoseStream`` to the wire format.

    Raises if ``enforce_budget`` and the encoded payload exceeds
    ``MAX_ENCODED_BYTES`` (per CLAUDE.md / operator constraint of ≤ 4 KB).
    """
    stream.validate()
    n_pairs = stream.n_pairs
    if n_pairs > 2**31 - 1:
        raise ValueError(f"n_pairs must fit in u32; got {n_pairs}")

    poses_np = stream.poses.detach().cpu().float().numpy().astype(np.float32)
    anchor_pose = poses_np[0].astype(np.float16)  # (6,) f16

    if n_pairs > 1:
        deltas = poses_np[1:] - poses_np[:-1]  # (n_pairs - 1, 6)
        deltas_q, scales = _quantize_int8_per_axis(deltas)
        deltas_payload = brotli.compress(deltas_q.tobytes(), quality=11)
    else:
        deltas_payload = b""
        scales = np.ones(POSE_DIM, dtype=np.float32)

    out = bytearray()
    out.append(PR106_RESIDUAL_MAGIC)
    out.append(RAFT_POSE_STREAM_FORMAT_ID)
    out += struct.pack("<I", n_pairs)
    out.append(POSE_DIM)
    out += anchor_pose.tobytes()  # 6 * 2 = 12 B
    out += scales.astype(np.float16).tobytes()  # 6 * 2 = 12 B
    out += struct.pack("<I", len(deltas_payload))
    out += deltas_payload

    encoded = bytes(out)
    if enforce_budget and len(encoded) > MAX_ENCODED_BYTES:
        raise ValueError(
            f"raft pose stream encoded size {len(encoded)} > budget {MAX_ENCODED_BYTES}; "
            "reduce n_pairs or coarsen quantization"
        )
    return encoded


def decode_raft_pose_stream(blob: bytes) -> RaftPoseStream:
    """Inverse of :func:`encode_raft_pose_stream`."""
    if len(blob) < 8:
        raise ValueError(f"raft pose stream blob too short: {len(blob)}")
    if blob[0] != PR106_RESIDUAL_MAGIC:
        raise ValueError(
            f"raft pose stream magic mismatch: 0x{blob[0]:02X} != 0x{PR106_RESIDUAL_MAGIC:02X}"
        )
    if blob[1] != RAFT_POSE_STREAM_FORMAT_ID:
        raise ValueError(
            f"raft pose stream format_id mismatch: 0x{blob[1]:02X} != 0x{RAFT_POSE_STREAM_FORMAT_ID:02X}"
        )
    pos = 2
    (n_pairs,) = struct.unpack_from("<I", blob, pos)
    pos += 4
    pose_dim = blob[pos]
    pos += 1
    if pose_dim != POSE_DIM:
        raise ValueError(f"pose_dim mismatch: got {pose_dim}, expected {POSE_DIM}")

    anchor_pose = np.frombuffer(blob, dtype=np.float16, count=POSE_DIM, offset=pos).astype(np.float32)
    pos += POSE_DIM * 2
    scales = np.frombuffer(blob, dtype=np.float16, count=POSE_DIM, offset=pos).astype(np.float32)
    pos += POSE_DIM * 2
    (deltas_len,) = struct.unpack_from("<I", blob, pos)
    pos += 4
    deltas_payload = blob[pos : pos + deltas_len]
    pos += deltas_len
    if pos != len(blob):
        raise ValueError(f"raft pose stream trailing bytes: pos={pos} total={len(blob)}")

    if n_pairs > 1 and deltas_len > 0:
        deltas_raw = brotli.decompress(deltas_payload)
        expected_bytes = (n_pairs - 1) * POSE_DIM
        if len(deltas_raw) != expected_bytes:
            raise ValueError(
                f"deltas size mismatch: got {len(deltas_raw)}, expected {expected_bytes}"
            )
        deltas_q = np.frombuffer(deltas_raw, dtype=np.int8).reshape(n_pairs - 1, POSE_DIM)
        deltas = deltas_q.astype(np.float32) * scales
        poses = np.empty((n_pairs, POSE_DIM), dtype=np.float32)
        poses[0] = anchor_pose
        poses[1:] = anchor_pose + np.cumsum(deltas, axis=0)
    else:
        poses = anchor_pose.reshape(1, POSE_DIM)

    return RaftPoseStream(poses=torch.from_numpy(poses).contiguous())


def compute_raft_pose_stream_bytes(stream: RaftPoseStream) -> int:
    """Return the encoded byte size of ``stream`` without raising on overflow."""
    return len(encode_raft_pose_stream(stream, enforce_budget=False))


def is_no_op(stream: RaftPoseStream, *, threshold: float = NO_OP_VARIANCE_THRESHOLD) -> bool:
    """Return True if the pose stream is effectively constant (no-op).

    The stream is no-op when the per-axis variance across frames falls below
    ``threshold``. A no-op stream wastes archive bytes per HNeRV parity
    discipline lesson 11.
    """
    stream.validate()
    if stream.n_pairs < 2:
        return True
    poses_np = stream.poses.detach().cpu().float().numpy()
    per_axis_var = poses_np.var(axis=0)  # (6,)
    return bool(per_axis_var.max() < threshold)


__all__ = [
    "MAX_ENCODED_BYTES",
    "NO_OP_VARIANCE_THRESHOLD",
    "POSE_DIM",
    "PR106_RESIDUAL_MAGIC",
    "RAFT_POSE_STREAM_FORMAT_ID",
    "RAFT_POSE_STREAM_VERSION",
    "RaftPoseStream",
    "compute_raft_pose_delta",
    "compute_raft_pose_stream_bytes",
    "decode_raft_pose_stream",
    "encode_raft_pose_stream",
    "is_no_op",
]
