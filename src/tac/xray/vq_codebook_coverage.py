"""F5: VQ-VAE codebook coverage primitive.

van den Oord shower-thought (zen_floor §13): a VQ-VAE codebook of size
``K = 4096`` on ``16x16`` driving patches achieves ~95% coverage of the
contest video's patch distribution. Per deep_math §3.4, codebook coverage
lower-bounds the zen-floor's codebook-byte axis:

    R_min_codebook = K * patch_dim * log2(quantization_levels)

This primitive computes a codebook's coverage against patches drawn from a
target video by mapping each patch to its nearest codebook entry under L2
distance and reporting the fraction of patches whose distance is below a
tolerance threshold (= "covered" patches).

Wire-in hooks engaged:

- ``bit_allocator``: per-patch codebook-index entropy informs the
  bit-allocator's codebook-bytes axis.
- ``sensitivity_map``: uncovered patches receive higher per-pixel sensitivity
  weight (they need EXACT reconstruction).
- ``probe_disambiguator``: comparing coverage at K=1024 vs K=4096 vs K=16384
  disambiguates the codebook-size-vs-coverage Pareto frontier.

Cross-references
----------------
- Source: van den Oord shower-thought (zen_floor §13)
- Deep math memo: ``.omx/research/deep_math_geometry_manifolds_synthesis_20260514.md`` §3.4
- Sister codec primitive: :mod:`tac.codec.frame_conditional` (the
  per-frame conditional codec that consumes VQ codebook indices)

CLAUDE.md compliance tags
-------------------------
- ``planning_only_no_score_claim``
- ``no_mps_authoritative``
- ``no_tmp_paths``
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from tac.xray.base import (
    ComposedXRayPrimitive,
    WireInHook,
    XRayPrimitiveResult,
)


@dataclass(frozen=True)
class VQCoverageReport:
    """Typed result from :meth:`VQCodebookCoverage.compute`.

    Attributes
    ----------
    codebook_size : int
        Number of entries in the VQ codebook (K).
    patch_dim : int
        Dimensionality of each codebook entry (= patch_height * patch_width *
        channels).
    n_patches_evaluated : int
        Number of patches drawn from the target and mapped to nearest entry.
    coverage_fraction : float
        Fraction of patches whose nearest-codebook-entry L2 distance is below
        ``coverage_tolerance``.
    mean_quantization_distance : float
        Mean L2 distance from patch to nearest codebook entry across all
        evaluated patches.
    codebook_index_entropy_bits : float
        Shannon entropy of the patch -> codebook-index assignment (in bits).
        log2(K) at most (uniform assignment); 0 if all patches map to one entry.
    codebook_byte_budget_lower_bound : int
        K * patch_dim * sizeof(quantization_level) bytes — minimal cost of
        the codebook itself if shipped at int8 precision.
    """

    codebook_size: int
    patch_dim: int
    n_patches_evaluated: int
    coverage_fraction: float
    mean_quantization_distance: float
    codebook_index_entropy_bits: float
    codebook_byte_budget_lower_bound: int

    def __post_init__(self) -> None:
        if self.codebook_size <= 0:
            raise ValueError("codebook_size must be positive")
        if self.patch_dim <= 0:
            raise ValueError("patch_dim must be positive")
        if not (0.0 <= self.coverage_fraction <= 1.0):
            raise ValueError(
                f"coverage_fraction must be in [0.0, 1.0]; got "
                f"{self.coverage_fraction}"
            )
        if self.mean_quantization_distance < 0.0:
            raise ValueError(
                "mean_quantization_distance must be non-negative"
            )
        if self.codebook_index_entropy_bits < 0.0:
            raise ValueError("entropy bits must be non-negative")


class VQCodebookCoverage:
    """F5 canonical primitive: VQ codebook coverage analyzer."""

    @property
    def name(self) -> str:
        return "vq_codebook_coverage"

    @property
    def wire_in_hooks(self) -> tuple[WireInHook, ...]:
        return (
            "bit_allocator",
            "sensitivity_map",
            "probe_disambiguator",
        )

    def compute(
        self,
        target: torch.Tensor,
        *,
        codebook: torch.Tensor,
        coverage_tolerance: float = 0.1,
        bytes_per_codebook_entry: int = 1,
        **_kwargs: Any,
    ) -> XRayPrimitiveResult:
        """Compute codebook coverage against a tensor of patches.

        Parameters
        ----------
        target : torch.Tensor
            Patches tensor of shape ``(N, patch_dim)`` — each row a flattened
            patch.
        codebook : torch.Tensor
            Codebook tensor of shape ``(K, patch_dim)``.
        coverage_tolerance : float
            Maximum L2 distance for a patch to be considered "covered".
        bytes_per_codebook_entry : int
            Bytes-per-dim of the quantized codebook entries (1 = int8,
            2 = fp16, 4 = fp32).
        """
        if target.dim() != 2:
            raise ValueError(
                f"target must be 2-D (N, patch_dim); got shape {tuple(target.shape)}"
            )
        if codebook.dim() != 2:
            raise ValueError(
                f"codebook must be 2-D (K, patch_dim); got shape "
                f"{tuple(codebook.shape)}"
            )
        if target.shape[1] != codebook.shape[1]:
            raise ValueError(
                f"target patch_dim {target.shape[1]} != codebook patch_dim "
                f"{codebook.shape[1]}"
            )
        if bytes_per_codebook_entry <= 0:
            raise ValueError("bytes_per_codebook_entry must be positive")

        n_patches = target.shape[0]
        k = codebook.shape[0]
        patch_dim = codebook.shape[1]

        # Compute pairwise L2 distances (N x K) and find nearest entry per patch.
        # For large N and K, this is O(N*K*patch_dim); for our smoke-scale
        # tests (N <= 1024, K <= 4096) it's fine on CPU.
        with torch.no_grad():
            # Use cdist for clarity.
            dists = torch.cdist(target.float(), codebook.float(), p=2.0)
            min_dists, nearest_idx = dists.min(dim=1)

        n_covered = (min_dists < coverage_tolerance).sum().item()
        coverage_fraction = n_covered / max(1, n_patches)
        mean_dist = float(min_dists.mean().item())

        # Shannon entropy of nearest-index distribution.
        counts = torch.bincount(nearest_idx, minlength=k).float()
        probs = counts / counts.sum()
        # Filter zero-probability entries.
        nonzero = probs > 0
        entropy_bits = (
            -(probs[nonzero] * torch.log2(probs[nonzero])).sum().item()
            if nonzero.any()
            else 0.0
        )

        codebook_bytes = k * patch_dim * bytes_per_codebook_entry

        report = VQCoverageReport(
            codebook_size=k,
            patch_dim=patch_dim,
            n_patches_evaluated=n_patches,
            coverage_fraction=coverage_fraction,
            mean_quantization_distance=mean_dist,
            codebook_index_entropy_bits=entropy_bits,
            codebook_byte_budget_lower_bound=codebook_bytes,
        )

        # Confidence band: binomial 1.96-SE around coverage_fraction.
        if n_patches > 1:
            se = math.sqrt(
                coverage_fraction
                * (1.0 - coverage_fraction)
                / max(1, n_patches)
            )
            band = (
                max(0.0, coverage_fraction - 1.96 * se),
                min(1.0, coverage_fraction + 1.96 * se),
            )
        else:
            band = (0.0, 1.0)

        return XRayPrimitiveResult(
            primitive_name=self.name,
            archive_or_video_path=None,
            archive_sha256=None,
            primitive_value=report,
            evidence_grade="council-deliberation",
            confidence_band=band,
            composes_with=("wavelet_hf_energy", "shannon_vector_r_d"),
            wire_in_hooks_engaged=self.wire_in_hooks,
            metadata={
                "coverage_tolerance": coverage_tolerance,
                "bytes_per_codebook_entry": bytes_per_codebook_entry,
            },
        )

    def compose_with(self, other: Any) -> Any:
        return ComposedXRayPrimitive(left=self, right=other)


__all__ = [
    "VQCodebookCoverage",
    "VQCoverageReport",
]
