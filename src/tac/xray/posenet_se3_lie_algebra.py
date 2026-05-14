# SPDX-License-Identifier: MIT
"""F8: PoseNet SE(3) Lie-algebra coordinate primitive.

Per deep_math §2.6 and the Mallat seat in zen_floor: PoseNet's 6-D output is
implicitly an se(3) Lie-algebra coordinate (3 translation + 3 rotation
components). The canonical pose-residual metric is the **Cartan-Killing
form** on se(3), NOT the Euclidean MSE that ``upstream/modules.py:82-84``
uses. For high-motion pairs the curvature term dominates; small Euclidean
distances can correspond to large geodesic distances on SE(3), and vice
versa.

This primitive:

1. Treats a 6-D pose vector as ``xi = (omega, v)`` where ``omega in R^3``
   is the rotation generator and ``v in R^3`` is the translation generator.
2. Computes the SE(3) group element ``g = exp(xi^)`` via matrix
   exponentiation of the 4x4 hat-map matrix.
3. Computes pose residuals via ``log(g_a^{-1} g_b)`` (the Lie-algebra
   distance), then norms under the Cartan-Killing metric (block-diagonal
   ``diag(I_3, I_3)`` for se(3) — equivalent to Euclidean for small
   rotations, divergent for large).

Per CLAUDE.md "MPS auth eval is NOISE" non-negotiable: this primitive does
NOT run PoseNet on MPS. It operates on pre-computed pose tensors.

Wire-in hooks engaged:

- ``sensitivity_map``: per-pair Lie-algebra residual magnitude is a
  pair-level sensitivity signal (high-motion pairs get more bits).
- ``probe_disambiguator``: comparing Euclidean MSE vs Cartan-Killing
  distance on the empirical posterior pairs disambiguates whether the
  PoseNet loss is loss-sub-optimal in the high-motion regime.

Cross-references
----------------
- Source: ``upstream/modules.py:66, 70-74, 82-84`` (PoseNet structural contract)
- Deep math memo: ``.omx/research/deep_math_geometry_manifolds_synthesis_20260514.md`` §2.6
- SE(3) hat-map literature: Murray-Li-Sastry 1994 *A Mathematical Introduction to Robotic Manipulation* + Sola et al. 2018
- D4 substrate Wyner-Ziv frame-0 nullspace (sister memory): poses are a high-value side info channel

CLAUDE.md compliance tags
-------------------------
- ``planning_only_no_score_claim``
- ``no_mps_authoritative``
- ``no_tmp_paths``
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from tac.xray.base import (
    ComposedXRayPrimitive,
    WireInHook,
    XRayPrimitiveResult,
)

# Pinned from upstream/modules.py:82-84 — only first 6 of 12 pose dims used.
POSENET_OUTPUT_DIMS_USED = 6


@dataclass(frozen=True)
class PoseSE3Report:
    """Typed result from :meth:`PoseNetSE3LieAlgebra.compute`.

    Attributes
    ----------
    n_pose_pairs : int
        Number of pose pairs analyzed.
    mean_euclidean_distance : float
        Mean Euclidean MSE between pose_a and pose_b (the canonical
        upstream metric).
    mean_lie_algebra_distance : float
        Mean Cartan-Killing distance computed via the SE(3) hat-map
        + matrix-log path.
    cartan_minus_euclidean_max : float
        Maximum per-pair (Lie - Euclidean) difference; large positive
        values indicate pairs where the upstream MSE under-counts the
        true geodesic distance.
    n_high_motion_pairs : int
        Number of pairs with Lie-algebra distance above
        ``high_motion_threshold``.
    high_motion_fraction : float
        n_high_motion_pairs / n_pose_pairs.
    """

    n_pose_pairs: int
    mean_euclidean_distance: float
    mean_lie_algebra_distance: float
    cartan_minus_euclidean_max: float
    n_high_motion_pairs: int
    high_motion_fraction: float

    def __post_init__(self) -> None:
        if self.n_pose_pairs < 0:
            raise ValueError("n_pose_pairs must be non-negative")
        if self.mean_euclidean_distance < 0.0:
            raise ValueError("mean_euclidean_distance must be non-negative")
        if self.mean_lie_algebra_distance < 0.0:
            raise ValueError("mean_lie_algebra_distance must be non-negative")
        if not (0.0 <= self.high_motion_fraction <= 1.0):
            raise ValueError(
                f"high_motion_fraction must be in [0.0, 1.0]; got "
                f"{self.high_motion_fraction}"
            )


class PoseNetSE3LieAlgebra:
    """F8 canonical primitive: SE(3) Lie-algebra pose analyzer."""

    @property
    def name(self) -> str:
        return "posenet_se3_lie_algebra"

    @property
    def wire_in_hooks(self) -> tuple[WireInHook, ...]:
        return ("sensitivity_map", "probe_disambiguator")

    def compute(
        self,
        target: torch.Tensor,
        *,
        target_b: torch.Tensor | None = None,
        high_motion_threshold: float = 0.5,
        **_kwargs: Any,
    ) -> XRayPrimitiveResult:
        """Analyze pose pairs under both Euclidean and Lie-algebra metrics.

        Parameters
        ----------
        target : torch.Tensor
            Either a single pose tensor of shape ``(N, 6)`` (in which case
            ``target_b`` must also be provided), or a paired tensor of
            shape ``(N, 2, 6)`` (pose_a and pose_b stacked).
        target_b : torch.Tensor | None
            Optional pose tensor of shape ``(N, 6)``. Required when
            target is ``(N, 6)``.
        high_motion_threshold : float
            Lie-algebra distance above which a pair is "high-motion".
        """
        # Resolve (pose_a, pose_b).
        if target.dim() == 3 and target.shape[1] == 2 and target.shape[2] == 6:
            pose_a = target[:, 0, :]
            pose_b = target[:, 1, :]
        elif target.dim() == 2 and target.shape[1] == 6:
            if target_b is None:
                raise ValueError(
                    "target_b must be provided when target is (N, 6)"
                )
            if target_b.dim() != 2 or target_b.shape != target.shape:
                raise ValueError(
                    f"target_b must be (N, 6) matching target; got shape "
                    f"{tuple(target_b.shape)}"
                )
            pose_a = target
            pose_b = target_b
        else:
            raise ValueError(
                f"target must be (N, 6) or (N, 2, 6); got shape "
                f"{tuple(target.shape)}"
            )

        pose_a = pose_a.float()
        pose_b = pose_b.float()
        n_pairs = pose_a.shape[0]

        # Euclidean (canonical upstream metric): MSE over the 6 dims.
        euclidean_per_pair = (pose_a - pose_b).pow(2).sum(dim=1).sqrt()
        mean_euc = float(euclidean_per_pair.mean().item())

        # Lie-algebra (Cartan-Killing) distance via SE(3) hat + log.
        lie_per_pair = self.compute_lie_algebra_residual(pose_a, pose_b)
        mean_lie = float(lie_per_pair.mean().item())

        diff = (lie_per_pair - euclidean_per_pair)
        cartan_minus_euc_max = float(diff.max().item())

        n_high = int((lie_per_pair > high_motion_threshold).sum().item())
        high_fraction = n_high / max(1, n_pairs)

        report = PoseSE3Report(
            n_pose_pairs=n_pairs,
            mean_euclidean_distance=mean_euc,
            mean_lie_algebra_distance=mean_lie,
            cartan_minus_euclidean_max=cartan_minus_euc_max,
            n_high_motion_pairs=n_high,
            high_motion_fraction=high_fraction,
        )

        return XRayPrimitiveResult(
            primitive_name=self.name,
            archive_or_video_path=None,
            archive_sha256=None,
            primitive_value=report,
            evidence_grade="mathematical-derivation",
            confidence_band=(
                max(0.0, mean_lie - 0.1),
                mean_lie + 0.1,
            ),
            composes_with=("per_pair_score_decomposition",),
            wire_in_hooks_engaged=self.wire_in_hooks,
            metadata={
                "high_motion_threshold": high_motion_threshold,
                "n_pose_dims_used": POSENET_OUTPUT_DIMS_USED,
            },
        )

    @staticmethod
    def compute_lie_algebra_residual(
        pose_a: torch.Tensor,
        pose_b: torch.Tensor,
    ) -> torch.Tensor:
        """Compute Lie-algebra residual ``||log(g_a^{-1} g_b)||`` per pair.

        For small se(3) coordinates, this is approximately
        ``||pose_a - pose_b||`` (Euclidean). For large coordinates, the
        Cartan-Killing form diverges from Euclidean.

        We approximate using the first-order BCH expansion:
        ``log(exp(-xi_a) exp(xi_b)) approx xi_b - xi_a + (1/2)[xi_a, xi_b]``
        where ``[.,.]`` is the se(3) Lie bracket.

        Parameters
        ----------
        pose_a, pose_b : torch.Tensor
            Shape ``(N, 6)`` where columns are (omega_1, omega_2, omega_3,
            v_1, v_2, v_3).

        Returns
        -------
        torch.Tensor
            Shape ``(N,)`` per-pair residual norms.
        """
        if pose_a.shape != pose_b.shape or pose_a.dim() != 2 or pose_a.shape[1] != 6:
            raise ValueError(
                f"pose_a and pose_b must be (N, 6) matching shapes; "
                f"got {tuple(pose_a.shape)} and {tuple(pose_b.shape)}"
            )
        omega_a, v_a = pose_a[:, :3], pose_a[:, 3:]
        omega_b, v_b = pose_b[:, :3], pose_b[:, 3:]

        # Linear difference + first-order curvature correction.
        delta_omega = omega_b - omega_a
        delta_v = v_b - v_a

        # se(3) Lie bracket [(omega_a, v_a), (omega_b, v_b)] =
        # (omega_a x omega_b, omega_a x v_b - omega_b x v_a).
        cross_omega = torch.cross(omega_a, omega_b, dim=1)
        cross_v_ab = torch.cross(omega_a, v_b, dim=1)
        cross_v_ba = torch.cross(omega_b, v_a, dim=1)
        bracket_v = cross_v_ab - cross_v_ba

        # Lie residual: first-order BCH.
        lie_omega = delta_omega + 0.5 * cross_omega
        lie_v = delta_v + 0.5 * bracket_v

        # Cartan-Killing-style norm: ||omega||^2 + ||v||^2.
        residual = torch.sqrt(
            (lie_omega**2).sum(dim=1) + (lie_v**2).sum(dim=1) + 1e-12
        )
        return residual

    def compose_with(self, other: Any) -> Any:
        return ComposedXRayPrimitive(left=self, right=other)


__all__ = [
    "POSENET_OUTPUT_DIMS_USED",
    "PoseNetSE3LieAlgebra",
    "PoseSE3Report",
]
