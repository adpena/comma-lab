"""Op_KLPoseStream — Karhunen-Loève basis pose-stream codec.

First tractable Hilbert-manifold subitem from the
``feedback_hilbert_manifolds_research_direction_20260507`` queue:
**pose Karhunen-Loève basis as new CodecOp**.

Why this exists alongside :class:`Op_RAFTPoseStream`:

  - Op_RAFTPoseStream encodes the FULL 6-DOF trajectory at int16 per axis
    per frame (~600 frames × 6 axes × 2 B = 7,200 B raw, ~5,000 B after
    Brotli). It treats poses as 600 independent 6-vectors.
  - Op_KLPoseStream exploits temporal correlation across frames. Driving
    trajectories are smooth — adjacent poses are highly correlated.
    A KL (Karhunen-Loève) basis derived from the trajectory's covariance
    captures this with a low-rank projection: top-k principal components
    of the (N×6) trajectory matrix.

Substrate-adaptive: the basis is COMPUTED from the input trajectory, not
hardcoded. The basis itself is part of the wire format (k × 6 floats),
plus the k-dim coefficients for each frame (N × k integers). The total
bytes is ``k*6*8 + N*k*2`` (basis as f64, coefficients as int16) ≈
``48k + 1200k`` = ``1248k`` for our 600 frames. Compared to RAFT's
``600 * 12 = 7200`` raw + Brotli, KL with k=4 stores ``1248 * 4 = 4992``
B uncompressed → similar Brotli output, with the *quality lever*
being k.

The right k depends on the trajectory's effective rank. For smooth
driving trajectories, the effective rank in 6-DOF is typically 2-3
(forward translation + yaw + small pitch contributions).

CLAUDE.md compliance:
  - Strict-scorer-rule: pure CPU + numpy + brotli + torch (no scorers).
  - No /tmp paths, no MPS, no CUDA dispatch.
  - Score claims: this op reports BYTES + reconstruction RMS only;
    predicted score impact tagged ``[predicted-band only]`` in council
    deliberation, NOT in module output.

Wire format (CPL1 sub-blob, deterministic, byte-exact):

    magic              : 4 bytes  = b"KPS1"  (KL-Pose-Stream v1)
    n_frames           : u32_LE
    pose_dim           : u16_LE   = 6
    n_components       : u16_LE   = k (number of KL basis vectors kept)
    basis_payload      : k × 6 × f64_LE      (the basis matrix, row-major)
    mean_payload       : 6 × f64_LE          (per-axis mean of the trajectory)
    coef_scale_payload : k × f64_LE          (per-component quantization scale)
    coef_payload       : brotli(int16_LE × n_frames × k)

Decoder reconstruction:

    poses[t, :] = mean + sum_i (coef[t, i] * coef_scales[i] * basis[i, :])

Reconstruction error has two sources:

  1. Truncation: low-rank projection drops the (D-k) tail components.
     Bounded by sqrt(sum_{i>k} σ_i²) per frame in 2-norm.
  2. Quantization: int16 coefficients introduce per-component error
     ≤ scale_i / 32767. Per-frame 2-norm error ≤ sum_i ||basis[i]||_2 *
     scale_i / 32767 ≤ k * max_scale / 32767 (since basis vectors
     are unit norm).

Cross-references:
  - Sibling: :mod:`tac.codec_pipeline_raft_pose`
  - Hilbert-manifold queue: ``feedback_hilbert_manifolds_research_direction_20260507``
  - Bayesian decision theory paper synthesis (sister direction):
    ``feedback_bayesian_gp_paper_synthesis_STUDY_verdict_20260507``
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Any

import brotli
import numpy as np
import torch

from tac.codec_pipeline import EncodeResult, ValidationReport

POSE_KEY = "poses_se3"
_KPS_MAGIC = b"KPS1"
_POSE_DIM = 6
_INT16_MAX = 32767


def _derive_kl_basis(
    poses: torch.Tensor, *, n_components: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive the KL basis from the trajectory.

    Returns ``(basis, mean, sing_vals)`` where:
      - ``basis``: shape ``(k, 6)``, top-k right singular vectors (unit-norm).
      - ``mean``: shape ``(6,)``, per-axis mean of the trajectory.
      - ``sing_vals``: shape ``(min(N, 6),)``, all singular values (used by
        callers to estimate truncation error).

    The KL basis is the right singular vectors of the centered trajectory
    matrix. Equivalent to PCA on the trajectory's frame-axis covariance.
    """
    if poses.dim() != 2 or poses.size(1) != _POSE_DIM:
        raise ValueError(
            f"_derive_kl_basis: expected (N, {_POSE_DIM}), got {tuple(poses.shape)}"
        )
    if n_components < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components}")
    n_components_eff = min(n_components, _POSE_DIM)
    arr = poses.detach().to(dtype=torch.float64).cpu().numpy()
    mean = arr.mean(axis=0)  # shape (6,)
    centered = arr - mean
    # SVD: centered = U S Vt, where Vt has shape (min(N,6), 6).
    # The KL basis is the rows of Vt (right singular vectors).
    _u, sing_vals, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:n_components_eff].copy()  # shape (k, 6)
    return basis, mean.copy(), sing_vals.copy()


def _project_quantize(
    poses: torch.Tensor,
    basis: np.ndarray,
    mean: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project the trajectory onto the KL basis and quantize coefficients.

    Returns ``(quantized_coefs, per_component_scales)`` where
    ``quantized_coefs`` has shape ``(N, k)`` int16 and
    ``per_component_scales`` has shape ``(k,)`` f64. The reconstructed
    coefficient is ``q * scale``.
    """
    arr = poses.detach().to(dtype=torch.float64).cpu().numpy()
    centered = arr - mean
    coefs = centered @ basis.T  # shape (N, k)
    # Per-component quantization scale: max_abs / INT16_MAX. If the
    # component is identically zero, use scale=1.0 to avoid div-by-zero;
    # the quantized coefs will all be zero anyway.
    abs_max = np.maximum(np.max(np.abs(coefs), axis=0), 1e-30)
    scales = abs_max / _INT16_MAX
    quantized = np.round(coefs / scales).astype(np.int16)
    quantized = np.clip(quantized, -_INT16_MAX, _INT16_MAX)
    return quantized, scales


def _dequantize_reconstruct(
    quantized_coefs: np.ndarray,
    coef_scales: np.ndarray,
    basis: np.ndarray,
    mean: np.ndarray,
) -> torch.Tensor:
    """Reconstruct the trajectory from quantized KL coefficients."""
    coefs = quantized_coefs.astype(np.float64) * coef_scales
    centered_recon = coefs @ basis  # shape (N, 6)
    arr = centered_recon + mean
    return torch.from_numpy(arr).to(dtype=torch.float32)


def _encode_blob(
    poses: torch.Tensor,
    *,
    n_components: int,
    brotli_quality: int,
) -> tuple[bytes, np.ndarray, np.ndarray, np.ndarray, int]:
    basis, mean, _sing = _derive_kl_basis(poses, n_components=n_components)
    quantized, scales = _project_quantize(poses, basis, mean)
    n_frames = quantized.shape[0]
    n_comp = quantized.shape[1]
    header = (
        _KPS_MAGIC
        + struct.pack("<I", n_frames)
        + struct.pack("<H", _POSE_DIM)
        + struct.pack("<H", n_comp)
    )
    basis_payload = basis.astype("<f8").tobytes()
    mean_payload = mean.astype("<f8").tobytes()
    scale_payload = scales.astype("<f8").tobytes()
    coef_raw = quantized.astype("<i2").tobytes()
    coef_compressed = brotli.compress(coef_raw, quality=brotli_quality)
    blob = (
        header
        + basis_payload
        + mean_payload
        + scale_payload
        + struct.pack("<I", len(coef_compressed))
        + coef_compressed
    )
    return blob, basis, mean, scales, n_frames


def _decode_blob(blob: bytes) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    if len(blob) < 4 + 4 + 2 + 2:
        raise ValueError("KL pose blob too short for header")
    if blob[:4] != _KPS_MAGIC:
        raise ValueError(f"KL pose blob bad magic: got {blob[:4]!r}, want {_KPS_MAGIC!r}")
    cur = 4
    n_frames = struct.unpack_from("<I", blob, cur)[0]; cur += 4
    pose_dim = struct.unpack_from("<H", blob, cur)[0]; cur += 2
    n_comp = struct.unpack_from("<H", blob, cur)[0]; cur += 2
    if pose_dim != _POSE_DIM:
        raise ValueError(f"KL pose blob: pose_dim={pose_dim}, expected {_POSE_DIM}")
    basis_len = n_comp * _POSE_DIM * 8
    basis = np.frombuffer(blob, dtype="<f8", count=n_comp * _POSE_DIM, offset=cur).reshape(n_comp, _POSE_DIM).copy()
    cur += basis_len
    mean = np.frombuffer(blob, dtype="<f8", count=_POSE_DIM, offset=cur).copy()
    cur += _POSE_DIM * 8
    scales = np.frombuffer(blob, dtype="<f8", count=n_comp, offset=cur).copy()
    cur += n_comp * 8
    coef_compressed_len = struct.unpack_from("<I", blob, cur)[0]; cur += 4
    coef_compressed = blob[cur:cur + coef_compressed_len]
    if len(coef_compressed) != coef_compressed_len:
        raise ValueError(
            f"KL pose blob: coef payload length mismatch "
            f"(declared {coef_compressed_len}, got {len(coef_compressed)})"
        )
    coef_raw = brotli.decompress(coef_compressed)
    expected_coef_bytes = n_frames * n_comp * 2
    if len(coef_raw) != expected_coef_bytes:
        raise ValueError(
            f"KL pose blob: decoded coef bytes={len(coef_raw)} != expected {expected_coef_bytes}"
        )
    quantized = np.frombuffer(coef_raw, dtype="<i2").reshape(n_frames, n_comp).copy()
    poses = _dequantize_reconstruct(quantized, scales, basis, mean)
    return poses, basis, mean, scales


@dataclass
class Op_KLPoseStream:
    """Op: Karhunen-Loève basis projection + int16 quantization + Brotli of poses.

    Hilbert-manifold-direction subitem #1: pose trajectory as a curve in
    a low-dim KL basis; transmits basis (substrate-adaptive) + coefficients
    instead of raw poses. Targets the pose marginal axis, which is
    2.71x SegNet's at PR106 frontier.

    Attributes:
        name: registered op name ``"kl_pose_stream"``.
        n_components: number of KL basis vectors to keep (1-6). Defaults
            to 4 — captures ~99%+ of typical driving trajectory variance.
        brotli_quality: Brotli compressor quality (1-11).
        transforms_state_dict: False (substitutional op; pose stream is
            independent from decoder weights).
    """
    name: str = "kl_pose_stream"
    n_components: int = 4
    brotli_quality: int = 11
    transforms_state_dict: bool = field(default=False, init=False)

    def encode(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        context: dict[str, Any],
    ) -> EncodeResult:
        if POSE_KEY not in state_dict:
            raise ValueError(
                f"Op_KLPoseStream.encode: state_dict missing required key "
                f"{POSE_KEY!r}; have {sorted(state_dict)}"
            )
        poses = state_dict[POSE_KEY]
        if not isinstance(poses, torch.Tensor):
            raise TypeError(
                f"Op_KLPoseStream.encode: {POSE_KEY!r} must be torch.Tensor, "
                f"got {type(poses).__name__}"
            )
        bytes_in = sum(t.numel() * t.element_size() for t in state_dict.values())
        blob, basis, mean, scales, n_frames = _encode_blob(
            poses,
            n_components=self.n_components,
            brotli_quality=self.brotli_quality,
        )
        op_state: dict[str, Any] = {
            "n_components": int(min(self.n_components, _POSE_DIM)),
            "n_frames": int(n_frames),
            # The basis/mean/scales are in the blob too; we don't echo them
            # in op_state to keep CPL1 wire JSON small. Decoder reads them
            # from the blob.
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
        poses, _basis, _mean, _scales = _decode_blob(blob)
        return {POSE_KEY: poses}

    def validate(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        context: dict[str, Any],
    ) -> ValidationReport:
        findings: list[str] = []
        if POSE_KEY not in state_dict:
            findings.append(f"missing required state_dict key {POSE_KEY!r}")
        else:
            poses = state_dict[POSE_KEY]
            if not isinstance(poses, torch.Tensor):
                findings.append(
                    f"{POSE_KEY!r} not a torch.Tensor (got {type(poses).__name__})"
                )
            elif poses.dim() != 2:
                findings.append(
                    f"{POSE_KEY!r} must be 2D (N, {_POSE_DIM}); got shape {tuple(poses.shape)}"
                )
            elif poses.size(1) != _POSE_DIM:
                findings.append(
                    f"{POSE_KEY!r} second dim must be {_POSE_DIM}; got {poses.size(1)}"
                )
            elif poses.size(0) < self.n_components:
                findings.append(
                    f"{POSE_KEY!r} has {poses.size(0)} frames < n_components={self.n_components}; "
                    f"SVD will silently truncate to min(N, k)"
                )
        if self.n_components < 1 or self.n_components > _POSE_DIM:
            findings.append(
                f"n_components={self.n_components} must be in [1, {_POSE_DIM}]"
            )
        return ValidationReport(passed=not findings, op_name=self.name, findings=findings)


def estimate_truncation_rms(
    poses: torch.Tensor, *, n_components: int
) -> dict[str, float]:
    """Diagnostic: return the truncation RMS error for a given k, plus the
    cumulative variance ratio. Useful for picking n_components empirically.

    Returns dict with keys:
      - ``truncation_rms_per_frame``: sqrt(sum_{i>k} σ_i²) / sqrt(N), the
        per-frame RMS error from truncating the (D-k)-rank tail.
      - ``cumulative_variance_ratio``: sum_{i<=k} σ_i² / sum_i σ_i², the
        fraction of trajectory variance captured by the top-k components.
      - ``singular_values``: list of all singular values (sorted desc).
    """
    _basis, _mean, sing_vals = _derive_kl_basis(poses, n_components=n_components)
    n_frames = int(poses.size(0))
    n_keep = min(n_components, len(sing_vals))
    total_var = float(np.sum(sing_vals**2))
    kept_var = float(np.sum(sing_vals[:n_keep] ** 2))
    truncated_var = max(total_var - kept_var, 0.0)
    return {
        "truncation_rms_per_frame": float(np.sqrt(truncated_var / max(n_frames, 1))),
        "cumulative_variance_ratio": kept_var / total_var if total_var > 0 else 1.0,
        "singular_values": [float(s) for s in sing_vals],
    }


__all__ = [
    "Op_KLPoseStream",
    "POSE_KEY",
    "estimate_truncation_rms",
]
