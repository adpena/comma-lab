"""Op_RAFTPoseStream - Phase 2 keystone pose-stream codec for the canonical
:class:`tac.codec_pipeline.CodecPipeline`.

Per Grand Council bilevel-optimization deliberation
(``.omx/research/grand_council_optimal_path_to_shannon_floor_20260507.md``),
**Phase 2** of the bilevel trajectory targets score 0.183 by replacing the
PR100 pose stream with RAFT-derived 6DOF SE(3) pose estimates. The
gap-decomposition memo (PR103-on-PR106 vs PR101 gold) attributes **69% of
the score gap to pose**; this op is the canonical Phase 2 keystone.

This module defines :class:`Op_RAFTPoseStream`, a CodecOp Protocol-conformant
encoder/decoder that wraps a per-axis int16 quantization + Brotli back end. It
is byte-deterministic, decoder-symmetric, and Joint-ADMM-compatible (mirrors
:mod:`tac.joint_admm_proximal_pose_delta` patterns).

Wire format (CPL1 sub-blob, deterministic, byte-exact):

    magic              : 4 bytes  = b"RPS1"  (RAFT-Pose-Stream v1)
    n_frames           : u32_LE
    pose_dim           : u16_LE   = 6
    per_axis_scales    : pose_dim x f64_LE  (per-axis quantization scale)
    quantized_payload  : brotli(int16_LE * n_frames * pose_dim)

The decoder is bit-faithful within the int16 quantization grid; i.e. the
roundtrip preserves the integer codes exactly; the float reconstruction is
``q * scale``. Reconstruction error is bounded by ``scale / 32767`` per axis.

State_dict contract (Op encode/decode):
    Input must contain exactly one tensor named ``"poses_se3"`` with shape
    ``(N_frames, 6)`` and dtype castable to float32. Output state_dict has
    the same key with reconstructed tensor.

Op_state schema (threaded through CodecPipeline manifest):
    {
        "per_axis_scales": list[float],   # length-6, the quantization scales
        "n_frames": int,
    }

CLAUDE.md compliance:
    - Strict-scorer-rule: pure CPU + numpy + brotli + torch (no scorers).
    - No MPS, no CUDA dispatch - encoder runs in plain CPU.
    - No /tmp paths.
    - Score claims: this op reports BYTES only; predicted score impact tagged
      ``[predicted-band only]`` in council deliberation, NOT in module output.

Cross-references:
    - Council memo: ``.omx/research/grand_council_optimal_path_to_shannon_floor_20260507.md``
    - Sibling Joint-ADMM proximal codec: :mod:`tac.joint_admm_proximal_pose_delta`
    - Phase 2 phase trajectory: :mod:`tools.run_bilevel_optimization`
    - Lane registry id: ``lane_op_raft_pose_stream`` (phase 2)
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Any

import brotli
import torch

from tac.codec_pipeline import EncodeResult, ValidationReport

_RPS_MAGIC = b"RPS1"
_POSE_DIM = 6
# Per-axis SE(3) magnitude sanity bounds. RAFT-derived poses for a driving
# video (small per-frame deltas, smooth trajectory) should never exceed
# ~30 radians of cumulative rotation or ~1000 meters of translation in any
# single axis on a 600-frame chunk. These are wide tolerance bounds; the
# validate step ``warns`` (records a finding but still passes) when exceeded
# rather than failing outright, since exotic test inputs may be passed
# during synthetic-substrate development.
_MAX_REASONABLE_MAGNITUDE = 1024.0
_INT16_MAX = 32767


def _quantize_int16(
    poses: torch.Tensor,
    *,
    per_axis_scales: list[float] | None = None,
) -> tuple[torch.Tensor, list[float]]:
    """Quantize ``(N, 6)`` poses to int16 with per-axis symmetric scales.

    If ``per_axis_scales`` is None, derive scales from the data:
    ``scale_i = max(|x_i|) / INT16_MAX`` with a numerical floor.

    Returns:
        (quantized_int16_tensor, scales_list)
    """
    if poses.ndim != 2 or poses.shape[1] != _POSE_DIM:
        raise ValueError(
            f"poses must have shape (N, {_POSE_DIM}); got {tuple(poses.shape)}"
        )
    flat = poses.detach().to(torch.float64)
    if per_axis_scales is None:
        # Derive symmetric per-axis scale from max abs.
        max_abs = flat.abs().amax(dim=0).clamp_min(1e-12)
        scales = (max_abs / float(_INT16_MAX)).tolist()
    else:
        if len(per_axis_scales) != _POSE_DIM:
            raise ValueError(
                f"per_axis_scales must have length {_POSE_DIM}; "
                f"got {len(per_axis_scales)}"
            )
        scales = [float(s) for s in per_axis_scales]
        if any(s <= 0.0 for s in scales):
            raise ValueError(
                f"per_axis_scales must be > 0; got {scales}"
            )
    scales_t = torch.tensor(scales, dtype=torch.float64)
    quantized = (flat / scales_t.unsqueeze(0)).round().clamp(
        -_INT16_MAX, _INT16_MAX
    ).to(torch.int16)
    return quantized, scales


def _dequantize_int16(
    quantized: torch.Tensor,
    *,
    per_axis_scales: list[float],
) -> torch.Tensor:
    """Reconstruct float poses from int16 grid + per-axis scales."""
    if quantized.ndim != 2 or quantized.shape[1] != _POSE_DIM:
        raise ValueError(
            f"quantized must have shape (N, {_POSE_DIM}); "
            f"got {tuple(quantized.shape)}"
        )
    if len(per_axis_scales) != _POSE_DIM:
        raise ValueError(
            f"per_axis_scales must have length {_POSE_DIM}; "
            f"got {len(per_axis_scales)}"
        )
    scales_t = torch.tensor(per_axis_scales, dtype=torch.float64)
    out = quantized.to(torch.float64) * scales_t.unsqueeze(0)
    return out.to(torch.float32)


def _encode_blob(
    poses: torch.Tensor,
    *,
    brotli_quality: int,
    per_axis_scales: list[float] | None,
) -> tuple[bytes, list[float], int]:
    """Build the RPS1 wire-format blob.

    Returns:
        (blob_bytes, per_axis_scales, n_frames)
    """
    quantized, scales = _quantize_int16(poses, per_axis_scales=per_axis_scales)
    n_frames = int(quantized.shape[0])

    # Pack int16 LE payload deterministically.
    np_view = quantized.contiguous().cpu().numpy().astype("<i2")
    raw = np_view.tobytes()
    compressed = brotli.compress(raw, quality=int(brotli_quality))

    out = bytearray()
    out += _RPS_MAGIC
    out += struct.pack("<I", n_frames)
    out += struct.pack("<H", _POSE_DIM)
    for s in scales:
        out += struct.pack("<d", float(s))
    out += compressed
    return bytes(out), scales, n_frames


def _decode_blob(
    blob: bytes,
    *,
    expected_n_frames: int | None = None,
    expected_scales: list[float] | None = None,
) -> torch.Tensor:
    """Decode an RPS1 blob to a (N, 6) float32 tensor."""
    if blob[:4] != _RPS_MAGIC:
        raise ValueError(
            f"Op_RAFTPoseStream.decode: bad magic {blob[:4]!r}, "
            f"expected {_RPS_MAGIC!r}"
        )
    cursor = 4
    n_frames = struct.unpack_from("<I", blob, cursor)[0]
    cursor += 4
    pose_dim = struct.unpack_from("<H", blob, cursor)[0]
    cursor += 2
    if pose_dim != _POSE_DIM:
        raise ValueError(
            f"Op_RAFTPoseStream.decode: pose_dim {pose_dim} != {_POSE_DIM}"
        )
    scales: list[float] = []
    for _ in range(pose_dim):
        s = struct.unpack_from("<d", blob, cursor)[0]
        cursor += 8
        scales.append(float(s))

    if expected_n_frames is not None and expected_n_frames != n_frames:
        raise ValueError(
            f"Op_RAFTPoseStream.decode: n_frames {n_frames} != "
            f"expected {expected_n_frames}"
        )
    if expected_scales is not None:
        if len(expected_scales) != pose_dim:
            raise ValueError(
                f"expected_scales length {len(expected_scales)} != "
                f"pose_dim {pose_dim}"
            )
        for i, (got, exp) in enumerate(zip(scales, expected_scales, strict=True)):
            if abs(got - float(exp)) > 1e-12 * max(1.0, abs(float(exp))):
                raise ValueError(
                    f"Op_RAFTPoseStream.decode: scale axis {i} "
                    f"{got!r} != expected {exp!r}"
                )

    compressed = blob[cursor:]
    raw = brotli.decompress(compressed)
    expected_n_bytes = n_frames * pose_dim * 2
    if len(raw) != expected_n_bytes:
        raise ValueError(
            f"Op_RAFTPoseStream.decode: decompressed payload {len(raw)}B != "
            f"expected {expected_n_bytes}B (n_frames={n_frames}, "
            f"pose_dim={pose_dim})"
        )
    import numpy as np
    arr = np.frombuffer(raw, dtype="<i2").reshape(n_frames, pose_dim).copy()
    quantized = torch.from_numpy(arr.astype("int16"))
    return _dequantize_int16(quantized, per_axis_scales=scales)


# ---------------------------------------------------------------------------
# Op
# ---------------------------------------------------------------------------

POSE_KEY = "poses_se3"


@dataclass
class Op_RAFTPoseStream:
    """Op: per-axis int16 quantization + Brotli of RAFT-derived SE(3) poses.

    Phase 2 keystone in the bilevel-optimization phase trajectory; the input
    state_dict must carry a ``"poses_se3"`` tensor of shape ``(N, 6)``.

    Attributes:
        name: registered op name ``"raft_pose_stream"`` (used in CPL1 manifest).
        brotli_quality: Brotli compressor quality (1-11). 11 = best ratio.
        per_axis_scales: optional override of the per-axis quantization scale
            (length-6 list of floats > 0). When None, scales are derived from
            the input poses' per-axis ``max(|x|)``.
        magnitude_warn_threshold: per-axis magnitude warn threshold for
            the validate gate (default _MAX_REASONABLE_MAGNITUDE).
    """
    name: str = "raft_pose_stream"
    brotli_quality: int = 11
    per_axis_scales: list[float] | None = None
    magnitude_warn_threshold: float = _MAX_REASONABLE_MAGNITUDE

    # Joint-ADMM-compatible: mark as substitutional (does not transform the
    # state_dict downstream; pose stream is independent from decoder weights).
    transforms_state_dict: bool = field(default=False, init=False)

    def encode(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        context: dict[str, Any],
    ) -> EncodeResult:
        if POSE_KEY not in state_dict:
            raise ValueError(
                f"Op_RAFTPoseStream.encode: state_dict missing required key "
                f"{POSE_KEY!r}; have {sorted(state_dict)}"
            )
        poses = state_dict[POSE_KEY]
        if not isinstance(poses, torch.Tensor):
            raise TypeError(
                f"Op_RAFTPoseStream.encode: {POSE_KEY!r} must be a "
                f"torch.Tensor, got {type(poses).__name__}"
            )
        bytes_in = sum(t.numel() * t.element_size() for t in state_dict.values())
        blob, scales, n_frames = _encode_blob(
            poses,
            brotli_quality=self.brotli_quality,
            per_axis_scales=self.per_axis_scales,
        )
        op_state: dict[str, Any] = {
            "per_axis_scales": [float(s) for s in scales],
            "n_frames": int(n_frames),
        }
        return EncodeResult(
            blob=blob,
            bytes_in=bytes_in,
            bytes_out=len(blob),
            op_name=self.name,
            op_state=op_state,
        )

    def decode(
        self,
        blob: bytes,
        *,
        op_state: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, torch.Tensor]:
        scales = op_state.get("per_axis_scales")
        if scales is None:
            raise ValueError(
                "Op_RAFTPoseStream.decode: op_state missing per_axis_scales"
            )
        n_frames = op_state.get("n_frames")
        if n_frames is None:
            raise ValueError(
                "Op_RAFTPoseStream.decode: op_state missing n_frames"
            )
        poses = _decode_blob(
            blob,
            expected_n_frames=int(n_frames),
            expected_scales=[float(s) for s in scales],
        )
        return {POSE_KEY: poses}

    def validate(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        context: dict[str, Any],
    ) -> ValidationReport:
        findings: list[str] = []
        # Hard gate: required key present.
        if POSE_KEY not in state_dict:
            findings.append(
                f"missing required key {POSE_KEY!r} for raft pose stream"
            )
            return ValidationReport(
                passed=False, op_name=self.name, findings=findings
            )
        poses = state_dict[POSE_KEY]
        if not isinstance(poses, torch.Tensor):
            findings.append(
                f"{POSE_KEY!r} must be a torch.Tensor, got {type(poses).__name__}"
            )
            return ValidationReport(
                passed=False, op_name=self.name, findings=findings
            )
        # Hard gate: shape (N, 6).
        if poses.ndim != 2 or poses.shape[1] != _POSE_DIM:
            findings.append(
                f"{POSE_KEY!r} shape {tuple(poses.shape)} != (N, {_POSE_DIM})"
            )
            return ValidationReport(
                passed=False, op_name=self.name, findings=findings
            )
        # Hard gate: at least 1 frame.
        if poses.shape[0] < 1:
            findings.append(
                f"{POSE_KEY!r} has 0 frames; need >= 1"
            )
            return ValidationReport(
                passed=False, op_name=self.name, findings=findings
            )
        # Hard gate: numeric dtype, not bool/object.
        if not poses.is_floating_point() and poses.dtype not in (
            torch.int16, torch.int32, torch.int64,
        ):
            findings.append(
                f"{POSE_KEY!r} dtype {poses.dtype} not numeric (float/int)"
            )
            return ValidationReport(
                passed=False, op_name=self.name, findings=findings
            )
        # Hard gate: reject NaN/Inf; they would corrupt the int16 round.
        as_float = poses.to(torch.float64)
        if torch.isnan(as_float).any():
            findings.append(f"{POSE_KEY!r} contains NaN values")
            return ValidationReport(
                passed=False, op_name=self.name, findings=findings
            )
        if torch.isinf(as_float).any():
            findings.append(f"{POSE_KEY!r} contains Inf values")
            return ValidationReport(
                passed=False, op_name=self.name, findings=findings
            )
        # Soft warn: extreme magnitude. Logged as a finding but still passes;
        # exotic synthetic substrates may legitimately exceed the bound.
        max_abs = float(as_float.abs().max().item())
        warning_only = False
        if max_abs > self.magnitude_warn_threshold:
            findings.append(
                f"{POSE_KEY!r} magnitude {max_abs:.4g} exceeds "
                f"warn-threshold {self.magnitude_warn_threshold:.4g} (warning)"
            )
            warning_only = True
        return ValidationReport(
            passed=(not findings) or warning_only,
            op_name=self.name,
            findings=findings,
        )


__all__ = [
    "POSE_KEY",
    "Op_RAFTPoseStream",
]
