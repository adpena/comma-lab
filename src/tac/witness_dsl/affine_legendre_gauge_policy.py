# SPDX-License-Identifier: MIT
"""Typed affine-Legendre gauge pair composed by the V9 genuine-frame arms.

This closes implementation custody for the mathematical transform pair only.  It does not
claim that a trained V9 latent has measured affine covariance; that remains an empirical gate.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from tac.information_geometry.bregman_v9_surfaces import (
    GeometryValidationError,
    affine_legendre_logsumexp_summary,
)

AFFINE_LEGENDRE_GAUGE_SCHEMA = "v9_affine_legendre_gauge_pair.v1"
AFFINE_LEGENDRE_GAUGE_EQUATION_ID = "cgauge_categorical_bregman_hessian_covariance_v1"


class AffineLegendreGaugePolicyError(ValueError):
    """Raised when a transform pair or its R/xi custody is incomplete."""


@dataclass(frozen=True, slots=True)
class GaugeChartCustody:
    pre_chart: str
    post_chart: str
    r_operator: str
    xi_chart: str

    def validate(self) -> None:
        for name, value in asdict(self).items():
            if not isinstance(value, str) or not value.strip():
                raise AffineLegendreGaugePolicyError(f"{name} custody must be a non-empty string")


@dataclass(frozen=True, slots=True)
class AffineLegendreGaugeTransform:
    matrix: tuple[tuple[float, ...], ...]
    offset: tuple[float, ...]
    scale: float
    linear_term: tuple[float, ...]
    constant: float = 0.0

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        matrix = np.asarray(self.matrix, dtype=np.float64)
        offset = np.asarray(self.offset, dtype=np.float64)
        linear = np.asarray(self.linear_term, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
            raise AffineLegendreGaugePolicyError("affine gauge matrix must be non-empty and square")
        if offset.shape != (matrix.shape[0],) or linear.shape != (matrix.shape[1],):
            raise AffineLegendreGaugePolicyError("offset/linear_term dimensions must match matrix")
        if not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(offset)) or not np.all(
            np.isfinite(linear)
        ):
            raise AffineLegendreGaugePolicyError("affine gauge values must be finite")
        if abs(float(np.linalg.det(matrix))) <= 1.0e-12:
            raise AffineLegendreGaugePolicyError("affine gauge matrix must be invertible")
        if not np.isfinite(float(self.scale)) or float(self.scale) <= 0.0:
            raise AffineLegendreGaugePolicyError("affine gauge scale must be finite and positive")
        if not np.isfinite(float(self.constant)):
            raise AffineLegendreGaugePolicyError("affine gauge constant must be finite")
        return matrix, offset, linear


@dataclass(frozen=True, slots=True)
class AffineLegendreGaugePair:
    transform: AffineLegendreGaugeTransform
    custody: GaugeChartCustody
    equation_id: str = AFFINE_LEGENDRE_GAUGE_EQUATION_ID
    tolerance: float = 1.0e-12

    def verify(self, point: tuple[float, ...], reference: tuple[float, ...]) -> dict[str, Any]:
        self.custody.validate()
        if self.equation_id != AFFINE_LEGENDRE_GAUGE_EQUATION_ID:
            raise AffineLegendreGaugePolicyError("affine-Legendre LawRef equation_id drifted")
        tolerance = float(self.tolerance)
        if not np.isfinite(tolerance) or tolerance < 0.0:
            raise AffineLegendreGaugePolicyError("tolerance must be finite and non-negative")
        matrix, offset, linear = self.transform.arrays()
        theta_p = np.asarray(point, dtype=np.float64)
        theta_q = np.asarray(reference, dtype=np.float64)
        if theta_p.shape != (matrix.shape[1],) or theta_q.shape != theta_p.shape:
            raise AffineLegendreGaugePolicyError("point/reference dimensions must match the transform")
        try:
            summary = affine_legendre_logsumexp_summary(
                theta_p,
                theta_q,
                matrix,
                offset,
                scale=float(self.transform.scale),
                linear_term=linear,
                constant=float(self.transform.constant),
            )
        except GeometryValidationError as exc:
            raise AffineLegendreGaugePolicyError(str(exc)) from exc
        divergence_error = float(summary["covariance_abs_error"])
        # The categorical Bregman contribution is the local action term at this policy seam;
        # consequently its pre/post scaled-action residual is the same independently derived scalar.
        action_error = abs(
            float(summary["gauged_bregman"]) - float(summary["scaled_base_bregman"])
        )
        if divergence_error > tolerance or action_error > tolerance:
            raise AffineLegendreGaugePolicyError(
                "affine-Legendre covariance residual exceeds tolerance: "
                f"divergence={divergence_error}, action={action_error}, tol={tolerance}"
            )
        payload: dict[str, Any] = {
            "schema": AFFINE_LEGENDRE_GAUGE_SCHEMA,
            "equation_id": self.equation_id,
            "custody": asdict(self.custody),
            "transform": asdict(self.transform),
            "point": theta_p.tolist(),
            "reference": theta_q.tolist(),
            "pre_divergence_scaled": float(summary["scaled_base_bregman"]),
            "post_divergence": float(summary["gauged_bregman"]),
            "divergence_abs_error": divergence_error,
            "pre_action_scaled": float(summary["scaled_base_bregman"]),
            "post_action": float(summary["gauged_bregman"]),
            "action_abs_error": action_error,
            "tolerance": tolerance,
            "status": "TYPED_PAIR_VERIFIED_LIVE_AFFINE_COVARIANCE_UNMEASURED",
            "verdict_scope": "implementation custody only; no trained-model or basis-family verdict",
            "score_claim": False,
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        payload["receipt_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
        return payload


def canonical_v9_affine_legendre_gauge_pair() -> AffineLegendreGaugePair:
    """Return the deterministic nontrivial V9 transform-pair declaration."""

    return AffineLegendreGaugePair(
        transform=AffineLegendreGaugeTransform(
            matrix=((1.0, 0.2, -0.1), (0.0, 1.1, 0.3), (0.2, -0.2, 0.9)),
            offset=(0.1, -0.2, 0.05),
            scale=1.7,
            linear_term=(0.3, -0.4, 0.2),
            constant=2.5,
        ),
        custody=GaugeChartCustody(
            pre_chart="categorical_logit_quotient_zero_mean",
            post_chart="v9_affine_legendre_categorical_chart",
            r_operator="contest_bilinear_uint8_R_874x1164_to_384x512",
            xi_chart="v9_modulation_xi_typed_config",
        ),
    )


def canonical_v9_affine_legendre_receipt() -> dict[str, Any]:
    return canonical_v9_affine_legendre_gauge_pair().verify(
        (0.8, -0.3, 0.2),
        (-0.1, 0.6, -0.4),
    )


__all__ = [
    "AFFINE_LEGENDRE_GAUGE_EQUATION_ID",
    "AFFINE_LEGENDRE_GAUGE_SCHEMA",
    "AffineLegendreGaugePair",
    "AffineLegendreGaugePolicyError",
    "AffineLegendreGaugeTransform",
    "GaugeChartCustody",
    "canonical_v9_affine_legendre_gauge_pair",
    "canonical_v9_affine_legendre_receipt",
]
