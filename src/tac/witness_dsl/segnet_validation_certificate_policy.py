# SPDX-License-Identifier: MIT
"""Typed event-conditioned policy for frozen-SegNet cheap validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from tac.boundary_math.segnet_validation_certificate import FeatureTrustRegion, ValidationDecision


@dataclass(frozen=True)
class SegNetValidationCertificatePolicy:
    mode: Literal["rigorous", "empirical"]
    rigorous_bound_artifact_sha256: str | None = None
    empirical_calibration_receipt_sha256: str | None = None
    fallback: Literal["full_teacher_and_refresh"] = "full_teacher_and_refresh"
    research_only: Literal[True] = True
    score_claim: Literal[False] = False
    promotion_eligible: Literal[False] = False

    def __post_init__(self) -> None:
        if self.mode == "rigorous":
            if self.rigorous_bound_artifact_sha256 is None or self.empirical_calibration_receipt_sha256 is not None:
                raise ValueError("rigorous mode requires only a rigorous-bound artifact")
        elif self.mode == "empirical":
            if self.empirical_calibration_receipt_sha256 is None or self.rigorous_bound_artifact_sha256 is not None:
                raise ValueError("empirical mode requires only a calibration receipt")
        else:
            raise ValueError("mode must be rigorous or empirical")
        value = self.rigorous_bound_artifact_sha256 or self.empirical_calibration_receipt_sha256
        if len(value or "") != 64 or any(c not in "0123456789abcdef" for c in value or ""):
            raise ValueError("policy artifact identity must be lowercase SHA-256")

    def validate_region(self, region: FeatureTrustRegion) -> None:
        expected = "rigorous_upper_bound" if self.mode == "rigorous" else "empirical_local_estimate"
        if region.authority != expected:
            raise ValueError("policy mode and trust-region authority differ")
        if self.mode == "rigorous" and region.bound_artifact_sha256 != self.rigorous_bound_artifact_sha256:
            raise ValueError("rigorous bound custody differs")
        if self.mode == "empirical" and region.calibration_receipt_sha256 != self.empirical_calibration_receipt_sha256:
            raise ValueError("empirical calibration custody differs")

    def select_action(self, decision: ValidationDecision) -> str:
        if decision.status == "ACCEPT" and self.mode == "rigorous" and decision.certificate_authoritative:
            return "reuse"
        if decision.status == "PROXY_ACCEPT" and self.mode == "empirical" and not decision.certificate_authoritative:
            return "reuse_advisory"
        return self.fallback

    def compile_measurement_contract(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "control_law": "reuse while gate accepts; refresh immediately on exit, rejection, nonfinite data, or custody change",
            "accept_authority": "certificate" if self.mode == "rigorous" else "advisory_proxy_only",
            "fallback": self.fallback,
            "research_only": self.research_only,
            "score_claim": self.score_claim,
            "promotion_eligible": self.promotion_eligible,
            "live_trainer_argv": [],
        }


__all__ = ["SegNetValidationCertificatePolicy"]
