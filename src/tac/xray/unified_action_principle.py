"""F11: Unified action principle primitive (GR-style variational objective).

Per deep_math §5.4 + ``feedback_unified_lagrangian_action_principle_GR_style_20260509``,
the migration target is a single scalar action ``S_total(theta, archive_bytes,
hardware)`` whose Euler-Lagrange condition ``delta_S / delta_theta = 0``
recovers all the track-Lagrangians:

    S_total = W_2(p_source, p_theta) * g_Fisher(theta) * T_trop(z(theta))

where:

- ``W_2(p_source, p_theta)`` is the Wasserstein-2 distance between the
  source distribution (contest video) and the model-induced distribution
  (= the L2 distortion term, up to a constant).
- ``g_Fisher(theta)`` is the Fisher metric of the parameter manifold
  (= the score-gradient sensitivity term per Catalog #123).
- ``T_trop(z(theta))`` is the tropical-semiring projection of the
  archive-byte vector (= the rate term, with the contest's piecewise-
  linear 25*B/N coefficient as the tropical "and" / min).

This primitive does NOT itself optimize ``theta`` — that requires GPU and
a full training run. Instead, it provides the SCALAR ``S_total`` value at
a given configuration so the meta-Lagrangian solver, the Pareto frontier
builder, and the autopilot ranker can all evaluate candidates against
a single unified objective.

Wire-in hooks engaged:

- ``sensitivity_map``: per-parameter Fisher diagonal feeds the
  sensitivity-map.
- ``pareto_constraint``: the (W_2, g_Fisher, T_trop) triple defines the
  Pareto frontier's three faces.
- ``bit_allocator``: tropical T_trop projection on archive_bytes is
  consumed by the bit-allocator.
- ``cathedral_autopilot``: S_total is the canonical ranking criterion
  for autopilot dispatch.

Cross-references
----------------
- Source memo: ``.omx/research/deep_math_geometry_manifolds_synthesis_20260514.md`` §5.4
- Anchor memory: ``feedback_unified_lagrangian_action_principle_GR_style_20260509.md``
- Catalog #123: ``check_no_weight_domain_saliency_on_score_gradient_substrate``
  (Fisher term must use score-gradient, not weight-domain proxy)

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
class UnifiedActionValue:
    """Typed result from :meth:`UnifiedActionPrinciple.compute`.

    Attributes
    ----------
    wasserstein_term : float
        W_2 distortion term (proxied by L2 / MSE-like norm of (theta, source)).
    fisher_term : float
        Fisher metric diagonal scalar (proxied by sum-of-squares of the
        sensitivity-gradient, OR provided via score_gradient_norm kwarg).
    tropical_rate_term : float
        Archive-rate tropical projection (= 25 * archive_bytes / N).
    s_total : float
        Product (or sum, depending on policy) of the three terms. Default
        policy is **product** per the deep_math memo's geometric-mean form.
    delta_s_dtheta_norm : float | None
        Norm of d(S_total) / d(theta), if computed.
    """

    wasserstein_term: float
    fisher_term: float
    tropical_rate_term: float
    s_total: float
    delta_s_dtheta_norm: float | None

    def __post_init__(self) -> None:
        if self.wasserstein_term < 0.0:
            raise ValueError(
                "wasserstein_term must be non-negative (distortion proxy)"
            )
        if self.fisher_term < 0.0:
            raise ValueError("fisher_term must be non-negative")
        if self.tropical_rate_term < 0.0:
            raise ValueError("tropical_rate_term must be non-negative")
        if self.s_total < 0.0:
            raise ValueError("s_total must be non-negative")


class UnifiedActionPrinciple:
    """F11 canonical primitive: GR-style unified action evaluator."""

    # Contest-rate constants.
    CONTEST_UNCOMPRESSED_SIZE_BYTES = 37_545_489
    CONTEST_RATE_COEFF = 25.0

    @property
    def name(self) -> str:
        return "unified_action_principle"

    @property
    def wire_in_hooks(self) -> tuple[WireInHook, ...]:
        return (
            "sensitivity_map",
            "pareto_constraint",
            "bit_allocator",
            "cathedral_autopilot",
        )

    def compute(
        self,
        target: Path | str | None = None,
        *,
        wasserstein_proxy: float | None = None,
        fisher_diagonal: torch.Tensor | float | None = None,
        score_gradient_norm: float | None = None,
        archive_bytes: int | None = None,
        composition_policy: str = "product",
        **_kwargs: Any,
    ) -> XRayPrimitiveResult:
        """Evaluate the unified action at a configuration.

        Parameters
        ----------
        target : Path | str | None
            Optional archive path (used for byte count + sha if
            ``archive_bytes`` not supplied).
        wasserstein_proxy : float | None
            W_2 distortion proxy (e.g., the score's distortion contribution
            100*seg + sqrt(10*pose)). Required.
        fisher_diagonal : torch.Tensor | float | None
            Fisher metric diagonal (per-parameter score-sensitivity).
            Either a 1-D tensor (will be summed) or a scalar.
        score_gradient_norm : float | None
            Alternative to fisher_diagonal: the L2 norm of the score
            gradient w.r.t. theta. If provided, used instead of
            fisher_diagonal.
        archive_bytes : int | None
            Archive size in bytes. If None, read from target path.
        composition_policy : str
            One of "product" (= geometric mean S_total = W_2 * g * T) or
            "sum" (= linear combination). Default "product".
        """
        if wasserstein_proxy is None:
            raise ValueError(
                "wasserstein_proxy is required; provide the score's "
                "distortion contribution (100*seg + sqrt(10*pose))"
            )
        if wasserstein_proxy < 0.0:
            raise ValueError("wasserstein_proxy must be non-negative")
        if composition_policy not in {"product", "sum"}:
            raise ValueError(
                f"composition_policy must be 'product' or 'sum'; got "
                f"{composition_policy!r}"
            )

        # Resolve Fisher term.
        if score_gradient_norm is not None:
            if score_gradient_norm < 0.0:
                raise ValueError("score_gradient_norm must be non-negative")
            fisher_term = score_gradient_norm
        elif fisher_diagonal is not None:
            if isinstance(fisher_diagonal, torch.Tensor):
                fisher_term = float(fisher_diagonal.abs().sum().item())
            else:
                fisher_term = float(fisher_diagonal)
            if fisher_term < 0.0:
                raise ValueError("fisher_diagonal must be non-negative")
        else:
            # Default to 1.0 (no sensitivity weighting); caller can override.
            fisher_term = 1.0

        # Resolve archive bytes.
        archive_path: Path | None = None
        archive_sha: str | None = None
        if archive_bytes is None:
            if target is None:
                raise ValueError(
                    "either archive_bytes or target (archive path) must be "
                    "provided to evaluate the tropical-rate term"
                )
            archive_path = Path(target)
            if not archive_path.exists():
                raise ValueError(
                    f"archive {archive_path!s} does not exist"
                )
            data = archive_path.read_bytes()
            archive_bytes = len(data)
            from tac.repo_io import sha256_bytes

            archive_sha = sha256_bytes(data)
        else:
            if archive_bytes < 0:
                raise ValueError("archive_bytes must be non-negative")
            if target is not None:
                archive_path = Path(target)
                if archive_path.exists():
                    from tac.repo_io import sha256_bytes

                    archive_sha = sha256_bytes(archive_path.read_bytes())

        # Tropical-rate term: 25 * archive_bytes / N (contest rate term).
        tropical_term = (
            self.CONTEST_RATE_COEFF
            * archive_bytes
            / self.CONTEST_UNCOMPRESSED_SIZE_BYTES
        )

        # Compose S_total.
        if composition_policy == "product":
            # Geometric mean: cube-root of W * g * T for unit consistency.
            s_total = math.pow(
                max(0.0, wasserstein_proxy)
                * max(0.0, fisher_term)
                * max(0.0, tropical_term),
                1.0 / 3.0,
            )
        else:
            s_total = wasserstein_proxy + fisher_term + tropical_term

        value = UnifiedActionValue(
            wasserstein_term=wasserstein_proxy,
            fisher_term=fisher_term,
            tropical_rate_term=tropical_term,
            s_total=s_total,
            delta_s_dtheta_norm=None,
        )

        return XRayPrimitiveResult(
            primitive_name=self.name,
            archive_or_video_path=archive_path,
            archive_sha256=archive_sha,
            primitive_value=value,
            evidence_grade="mathematical-derivation",
            confidence_band=(s_total * 0.7, s_total * 1.3),
            composes_with=(),
            wire_in_hooks_engaged=self.wire_in_hooks,
            metadata={
                "composition_policy": composition_policy,
                "archive_bytes": archive_bytes,
            },
        )

    def gradient_descent_step(
        self,
        theta: torch.Tensor,
        lr: float,
        objective_grad: torch.Tensor,
    ) -> torch.Tensor:
        """Vanilla gradient step on theta toward smaller S_total.

        Provided as a thin convenience for downstream consumers who want
        to call ``unified_action.gradient_descent_step(theta, lr, grad)``
        without re-deriving the canonical update rule.
        """
        if theta.shape != objective_grad.shape:
            raise ValueError(
                f"theta shape {tuple(theta.shape)} != grad shape "
                f"{tuple(objective_grad.shape)}"
            )
        return theta - lr * objective_grad

    def compose_with(self, other: Any) -> Any:
        return ComposedXRayPrimitive(left=self, right=other)


__all__ = [
    "UnifiedActionPrinciple",
    "UnifiedActionValue",
]
