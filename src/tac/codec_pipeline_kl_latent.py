# SPDX-License-Identifier: MIT
"""Op_KLLatent — Karhunen-Loève basis codec for arbitrary 2D latent matrices.

Generalization of :class:`tac.codec_pipeline_kl_pose.Op_KLPoseStream` from
the fixed (N, 6) pose shape to any (N, D) latent matrix. Targets substrates
that ship raw-tensor latents in the cathedral wire format (PR100, PR106 in
their non-LZMA encoding, internal training checkpoints).

**Substrate compatibility note**: PR101 specifically encodes its (600, 28)
latents via LZMA + per-dim min/scale + temporal-delta uint8 codes; the
cathedral's KL pattern is INCOMPATIBLE with PR101's wire format and CANNOT
substitute into a stock PR101 archive without a forked inflate. See
``tools/pr101_archive_substitution_surgery.py`` docstring for the full
substrate-compatibility table.

For substrates that DO permit raw-tensor latent substitution (PR100/PR106),
Op_KLLatent provides the same KL-projection lever as Op_KLPoseStream:
exploits low-rank structure in the (N, D) matrix to compress with a
substrate-adaptive basis + int16 coefficients.

Wire format (CPL1 sub-blob, deterministic, byte-exact):

    magic              : 4 bytes  = b"KLT1"  (KL-LaTent v1)
    n_frames           : u32_LE
    latent_dim         : u16_LE   (D, e.g. 28 for PR101's latent shape)
    n_components       : u16_LE   = k
    basis_payload      : k × D × f64_LE
    mean_payload       : D × f64_LE
    coef_scale_payload : k × f64_LE
    coef_payload       : brotli(int16_LE × n_frames × k)

Decoder reconstruction:

    latents[t, :] = mean + sum_i (coef[t, i] * coef_scales[i] * basis[i, :])

Reconstruction error has two sources, identical to Op_KLPoseStream:

  1. Truncation: low-rank projection drops (D-k) tail components.
  2. Quantization: int16 coefficients introduce per-component error
     ≤ scale_i / 32767.

CLAUDE.md compliance:
  - Strict-scorer-rule: pure CPU + numpy + brotli + torch (no scorers).
  - No /tmp paths, no MPS, no CUDA dispatch.
  - Score claims: this op reports BYTES + reconstruction RMS only;
    predicted score impact tagged ``[predicted-band only]`` in council
    deliberation, NOT in module output.

Predicted savings (for council reference, NOT module output):

  PR101's (600, 28) latents currently ship at ~15.4 KB after LZMA.
  KL with k=8 stores: 8*28*8 = 1792 B basis + 28*8 mean/scales +
  brotli(600*8*2 = 9600 B int16 coefs) ≈ ~10-12 KB. Marginal savings
  (~3-5 KB if effective rank << 28); requires forked inflate.

Cross-references:
  - Sibling: :mod:`tac.codec_pipeline_kl_pose` (template)
  - Hilbert-manifold queue:
    ``feedback_hilbert_manifolds_research_direction_20260507``
  - PR101 wire-format-incompatibility:
    ``tools/pr101_archive_substitution_surgery.py`` docstring
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import Any

import brotli
import numpy as np
import torch

from tac.codec_pipeline import EncodeResult, ValidationReport

LATENT_KEY = "latents"
_KLT_MAGIC = b"KLT1"
_INT16_MAX = 32767


def _derive_kl_basis(
    latents: torch.Tensor, *, n_components: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derive the KL basis from the latent matrix.

    Returns ``(basis, mean, sing_vals)`` where:
      - ``basis``: shape ``(k, D)``, top-k right singular vectors
        (unit-norm).
      - ``mean``: shape ``(D,)``, per-dim mean across frames.
      - ``sing_vals``: shape ``(min(N, D),)``, full SVD spectrum
        (used by ``estimate_truncation_rms`` to estimate the dropped
        tail's L²).
    """
    if latents.dim() != 2:
        raise ValueError(
            f"_derive_kl_basis: expected 2D (N, D), got shape {tuple(latents.shape)}"
        )
    n_frames, latent_dim = int(latents.size(0)), int(latents.size(1))
    if n_components < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components}")
    n_components_eff = min(n_components, latent_dim, n_frames)
    arr = latents.detach().to(dtype=torch.float64).cpu().numpy()
    mean = arr.mean(axis=0)
    centered = arr - mean
    _u, sing_vals, vt = np.linalg.svd(centered, full_matrices=False)
    basis = vt[:n_components_eff].copy()
    return basis, mean.copy(), sing_vals.copy()


def _project_quantize(
    latents: torch.Tensor,
    basis: np.ndarray,
    mean: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project latents onto the KL basis and quantize to int16.

    Returns ``(quantized_coefs, per_component_scales)``: quantized has
    shape (N, k) int16; scales has shape (k,) f64. Reconstructed
    coefficient = ``q * scale``.
    """
    arr = latents.detach().to(dtype=torch.float64).cpu().numpy()
    centered = arr - mean
    coefs = centered @ basis.T  # (N, k)
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
    """Reconstruct the latent matrix from quantized KL coefficients."""
    coefs = quantized_coefs.astype(np.float64) * coef_scales
    centered_recon = coefs @ basis  # (N, D)
    arr = centered_recon + mean
    return torch.from_numpy(arr).to(dtype=torch.float32)


def _encode_blob(
    latents: torch.Tensor,
    *,
    n_components: int,
    brotli_quality: int,
) -> tuple[bytes, np.ndarray, np.ndarray, np.ndarray, int, int]:
    basis, mean, _sing = _derive_kl_basis(latents, n_components=n_components)
    quantized, scales = _project_quantize(latents, basis, mean)
    n_frames, n_comp = quantized.shape
    latent_dim = int(mean.shape[0])
    if latent_dim > 0xFFFF:
        raise ValueError(
            f"latent_dim {latent_dim} exceeds u16 wire-format max (65535)"
        )
    if n_comp > 0xFFFF:
        raise ValueError(
            f"n_components {n_comp} exceeds u16 wire-format max (65535)"
        )
    header = (
        _KLT_MAGIC
        + struct.pack("<I", n_frames)
        + struct.pack("<H", latent_dim)
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
    return blob, basis, mean, scales, n_frames, latent_dim


def _decode_blob(
    blob: bytes,
) -> tuple[torch.Tensor, np.ndarray, np.ndarray, np.ndarray]:
    if len(blob) < 4 + 4 + 2 + 2:
        raise ValueError("KL latent blob too short for header")
    if blob[:4] != _KLT_MAGIC:
        raise ValueError(
            f"KL latent blob bad magic: got {blob[:4]!r}, want {_KLT_MAGIC!r}"
        )
    cur = 4
    n_frames = struct.unpack_from("<I", blob, cur)[0]
    cur += 4
    latent_dim = struct.unpack_from("<H", blob, cur)[0]
    cur += 2
    n_comp = struct.unpack_from("<H", blob, cur)[0]
    cur += 2
    basis_len = n_comp * latent_dim * 8
    basis = np.frombuffer(
        blob, dtype="<f8", count=n_comp * latent_dim, offset=cur
    ).reshape(n_comp, latent_dim).copy()
    cur += basis_len
    mean = np.frombuffer(blob, dtype="<f8", count=latent_dim, offset=cur).copy()
    cur += latent_dim * 8
    scales = np.frombuffer(blob, dtype="<f8", count=n_comp, offset=cur).copy()
    cur += n_comp * 8
    coef_compressed_len = struct.unpack_from("<I", blob, cur)[0]
    cur += 4
    coef_compressed = blob[cur : cur + coef_compressed_len]
    if len(coef_compressed) != coef_compressed_len:
        raise ValueError(
            f"KL latent blob: coef payload length mismatch "
            f"(declared {coef_compressed_len}, got {len(coef_compressed)})"
        )
    coef_raw = brotli.decompress(coef_compressed)
    expected_coef_bytes = n_frames * n_comp * 2
    if len(coef_raw) != expected_coef_bytes:
        raise ValueError(
            f"KL latent blob: decoded coef bytes={len(coef_raw)} != "
            f"expected {expected_coef_bytes}"
        )
    quantized = (
        np.frombuffer(coef_raw, dtype="<i2").reshape(n_frames, n_comp).copy()
    )
    latents = _dequantize_reconstruct(quantized, scales, basis, mean)
    return latents, basis, mean, scales


@dataclass
class Op_KLLatent:
    """Op: Karhunen-Loève basis projection + int16 quantization + Brotli of
    a 2D latent matrix.

    Generalizes the (N, 6) pose pattern to any (N, D) latent matrix.
    Substrate compatibility: ONLY for substrates that ship raw-tensor
    latents in cathedral wire format. NOT compatible with PR101's
    LZMA-coded latent format (would require forked inflate).

    Attributes:
        name: registered op name ``"kl_latent"``.
        n_components: number of KL basis vectors to keep (1..min(N, D)).
            Defaults to 8 for typical (600, 28) latents (~99%+ variance
            retained on smooth substrates per pose-codec empirical).
        brotli_quality: Brotli compressor quality (1-11).
        transforms_state_dict: False (substitutional op; latent stream
            is independent from decoder weights).
    """
    name: str = "kl_latent"
    n_components: int = 8
    brotli_quality: int = 11
    transforms_state_dict: bool = field(default=False, init=False)

    def encode(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        context: dict[str, Any],
    ) -> EncodeResult:
        if LATENT_KEY not in state_dict:
            raise ValueError(
                f"Op_KLLatent.encode: state_dict missing required key "
                f"{LATENT_KEY!r}; have {sorted(state_dict)}"
            )
        latents = state_dict[LATENT_KEY]
        if not isinstance(latents, torch.Tensor):
            raise TypeError(
                f"Op_KLLatent.encode: {LATENT_KEY!r} must be torch.Tensor, "
                f"got {type(latents).__name__}"
            )
        bytes_in = sum(t.numel() * t.element_size() for t in state_dict.values())
        blob, basis, mean, scales, n_frames, latent_dim = _encode_blob(
            latents,
            n_components=self.n_components,
            brotli_quality=self.brotli_quality,
        )
        op_state: dict[str, Any] = {
            "n_components": int(min(self.n_components, latent_dim, n_frames)),
            "n_frames": int(n_frames),
            "latent_dim": int(latent_dim),
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
        latents, _basis, _mean, _scales = _decode_blob(blob)
        return {LATENT_KEY: latents}

    def validate(
        self,
        state_dict: dict[str, torch.Tensor],
        *,
        context: dict[str, Any],
    ) -> ValidationReport:
        findings: list[str] = []
        if LATENT_KEY not in state_dict:
            findings.append(f"missing required state_dict key {LATENT_KEY!r}")
        else:
            latents = state_dict[LATENT_KEY]
            if not isinstance(latents, torch.Tensor):
                findings.append(
                    f"{LATENT_KEY!r} not a torch.Tensor (got {type(latents).__name__})"
                )
            elif latents.dim() != 2:
                findings.append(
                    f"{LATENT_KEY!r} must be 2D (N, D); got shape {tuple(latents.shape)}"
                )
            else:
                n_frames, latent_dim = int(latents.size(0)), int(latents.size(1))
                if n_frames < self.n_components:
                    findings.append(
                        f"{LATENT_KEY!r} has {n_frames} rows < n_components="
                        f"{self.n_components}; SVD will silently truncate"
                    )
                if latent_dim > 0xFFFF:
                    findings.append(
                        f"latent_dim {latent_dim} exceeds u16 wire-format max"
                    )
        if self.n_components < 1:
            findings.append(f"n_components={self.n_components} must be >= 1")
        return ValidationReport(
            passed=not findings, op_name=self.name, findings=findings
        )


def estimate_truncation_rms(
    latents: torch.Tensor, *, n_components: int
) -> dict[str, float]:
    """Diagnostic: return the truncation RMS error for a given k, plus the
    cumulative variance ratio. Useful for picking ``n_components``
    empirically before committing to a sweep.

    Returns dict with keys:
      - ``truncation_rms_per_frame``: sqrt(sum_{i>k} σ_i²) / sqrt(N)
      - ``cumulative_variance_ratio``: sum_{i<=k} σ_i² / sum_i σ_i²
      - ``singular_values``: list of all singular values (sorted desc)
    """
    _basis, _mean, sing_vals = _derive_kl_basis(latents, n_components=n_components)
    n_frames = int(latents.size(0))
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
    "LATENT_KEY",
    "Op_KLLatent",
    "estimate_truncation_rms",
]
