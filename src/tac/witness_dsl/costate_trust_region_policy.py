# SPDX-License-Identifier: MIT
"""Typed, flag-free policy for costate trust-region validation economics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from tac.scorer_surrogate.costate_trust_region import (
        CostateReuseDecision,
        CostateTrustRegion,
        DirectCostateCertificate,
        DirectCostateDecision,
    )


@dataclass(frozen=True)
class CostateTrustRegionPolicy:
    """Compile an anchor-derived control law without a radius knob."""

    mode: Literal["rigorous", "empirical"]
    rigorous_bound_artifact_sha256: str | None = None
    empirical_calibration_receipt_sha256: str | None = None
    validation_cadence: Literal["anchor_only"] = "anchor_only"
    fallback: Literal["full_teacher_refresh"] = "full_teacher_refresh"
    research_only: Literal[True] = True
    score_claim: Literal[False] = False
    promotion_eligible: Literal[False] = False

    def __post_init__(self) -> None:
        if self.mode == "rigorous":
            value = self.rigorous_bound_artifact_sha256
            if value is None or self.empirical_calibration_receipt_sha256 is not None:
                raise ValueError("rigorous mode requires only the rigorous bound artifact")
        elif self.mode == "empirical":
            value = self.empirical_calibration_receipt_sha256
            if value is None or self.rigorous_bound_artifact_sha256 is not None:
                raise ValueError("empirical mode requires only the calibration receipt")
        else:
            raise ValueError("mode must be rigorous or empirical")
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("policy custody must be a lowercase SHA-256")

    def validate_region(self, region: CostateTrustRegion) -> None:
        expected = "rigorous_upper_bound" if self.mode == "rigorous" else "empirical_local_estimate"
        if region.authority != expected:
            raise ValueError("policy mode and region authority differ")
        if self.mode == "rigorous" and region.bound_artifact_sha256 != self.rigorous_bound_artifact_sha256:
            raise ValueError("rigorous bound custody differs")
        if self.mode == "empirical" and (
            region.calibration_receipt_sha256 != self.empirical_calibration_receipt_sha256
        ):
            raise ValueError("empirical calibration custody differs")

    def select_action(self, decision: CostateReuseDecision) -> str:
        if self.mode == "rigorous" and decision.status == "CERTIFIED_REUSE":
            if not (decision.label_cell_authoritative and decision.descent_authoritative):
                raise ValueError("rigorous reuse decision lacks both authority legs")
            return "reuse_costate"
        if self.mode == "empirical" and decision.status == "PROXY_REUSE":
            if decision.label_cell_authoritative or decision.descent_authoritative:
                raise ValueError("empirical reuse decision cannot carry certificate authority")
            return "reuse_costate_advisory"
        return self.fallback

    def compile_measurement_contract(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "radius_control_law": (
                "min(anchor-margin ball inverted through first-block Jacobian Taylor bound, "
                "projected-costate descent ball); no radius literal"
            ),
            "per_step_gate": "O(pixels) frame displacement plus predicted margin/Fisher field",
            "provider_mode": "current_prefix_vjp_banked_suffix",
            "validation_cadence": self.validation_cadence,
            "fallback": self.fallback,
            "rigorous_requires": (
                "first-block Jacobian norm and Lipschitz upper bounds; suffix pairwise-logit and "
                "costate Lipschitz upper bounds; renderer-VJP norm upper bound; projected-gradient floor"
            ),
            "empirical_authority": "training-signal advisory only",
            "research_only": self.research_only,
            "score_claim": self.score_claim,
            "promotion_eligible": self.promotion_eligible,
            "live_trainer_argv": [],
        }


@dataclass(frozen=True)
class DirectFullCostatePolicy:
    """Default-off, flag-free policy for the Jacobian-drift certificate."""

    enabled: bool = False
    mode: Literal["rigorous", "empirical"] = "empirical"
    rigorous_bound_artifact_sha256: str | None = None
    empirical_calibration_receipt_sha256: str | None = None
    validation_cadence: Literal["anchor_and_admitted_reuse"] = "anchor_and_admitted_reuse"
    fallback: Literal["full_teacher_refresh"] = "full_teacher_refresh"
    research_only: Literal[True] = True
    score_claim: Literal[False] = False
    promotion_eligible: Literal[False] = False

    def __post_init__(self) -> None:
        if self.mode == "rigorous":
            value = self.rigorous_bound_artifact_sha256
            if value is None or self.empirical_calibration_receipt_sha256 is not None:
                raise ValueError("rigorous direct mode requires only the bound artifact")
        elif self.mode == "empirical":
            value = self.empirical_calibration_receipt_sha256
            if value is None or self.rigorous_bound_artifact_sha256 is not None:
                raise ValueError("empirical direct mode requires only the calibration receipt")
        else:
            raise ValueError("mode must be rigorous or empirical")
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("direct policy custody must be a lowercase SHA-256")

    def validate_certificate(self, certificate: DirectCostateCertificate) -> None:
        expected = "rigorous_upper_bound" if self.mode == "rigorous" else "empirical_local_estimate"
        if certificate.authority != expected:
            raise ValueError("direct policy mode and certificate authority differ")
        if self.mode == "rigorous":
            if certificate.bound_artifact_sha256 != self.rigorous_bound_artifact_sha256:
                raise ValueError("direct rigorous bound custody differs")
        elif certificate.calibration_receipt_sha256 != self.empirical_calibration_receipt_sha256:
            raise ValueError("direct empirical calibration custody differs")

    def select_action(
        self,
        decision: DirectCostateDecision,
        *,
        certificate: DirectCostateCertificate | None = None,
        current_target_scorer_input_sha256: str | None = None,
        current_displacement_vector_sha256: str | None = None,
        current_correction_tensor_sha256: str | None = None,
        current_corrected_costate_sha256: str | None = None,
        current_anchor_costate_sha256: str | None = None,
    ) -> str:
        if not self.enabled:
            return self.fallback
        if certificate is None:
            return self.fallback
        self.validate_certificate(certificate)
        from tac.scorer_surrogate.costate_trust_region import direct_certificate_sha256

        required = (
            decision.target_scorer_input_sha256,
            decision.displacement_vector_sha256,
            decision.correction_tensor_sha256,
            decision.corrected_costate_sha256,
            decision.anchor_costate_sha256,
            decision.certificate_sha256,
            current_target_scorer_input_sha256,
            current_displacement_vector_sha256,
            current_correction_tensor_sha256,
            current_corrected_costate_sha256,
            current_anchor_costate_sha256,
        )
        if any(value is None for value in required):
            return self.fallback
        if (
            decision.certificate_sha256 != direct_certificate_sha256(certificate)
            or decision.anchor_costate_sha256 != certificate.anchor_costate_sha256
            or decision.target_scorer_input_sha256 != current_target_scorer_input_sha256
            or decision.displacement_vector_sha256 != current_displacement_vector_sha256
            or decision.correction_tensor_sha256 != current_correction_tensor_sha256
            or decision.corrected_costate_sha256 != current_corrected_costate_sha256
            or decision.anchor_costate_sha256 != current_anchor_costate_sha256
        ):
            return self.fallback
        if not decision.correction_derivation_authoritative:
            return self.fallback
        if self.mode == "rigorous" and decision.status == "CERTIFIED_REUSE":
            if not decision.descent_authoritative or not certificate.certificate_authoritative:
                raise ValueError("certified direct reuse lacks descent authority")
            return "reuse_corrected_full_costate"
        if self.mode == "empirical" and decision.status == "PROXY_REUSE":
            if decision.descent_authoritative:
                raise ValueError("empirical direct reuse cannot carry authority")
            return "reuse_corrected_full_costate_advisory"
        return self.fallback

    def compile_measurement_contract(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "provider_mode": "direct_full_input_costate",
            "radius_control_law": (
                "min(monotone root of E(r)<gamma_theta/B_R, derived geometry radius, "
                "custody calibration cap); no radius or tolerance literal"
            ),
            "rigorous_requires": (
                "custody-bearing B_J, B_H, Lip(DJ), Q_a, L_q, L_c, B_R, gamma_theta, "
                "geometry radius, and calibration cap"
            ),
            "fixed_adjoint_correction": "(DJ(a)[d])^T q_a with q_a detached",
            "correction_derivation_gate": (
                "trusted integrated HVP derivation receipt required; external correction tensors "
                "remain MISSING_INTEGRATED_HVP_DERIVATION_RECEIPT"
            ),
            "full_ce_hvp_role": "adjoint-drift diagnostic only",
            "validation_cadence": self.validation_cadence,
            "fallback": self.fallback,
            "research_only": self.research_only,
            "score_claim": self.score_claim,
            "promotion_eligible": self.promotion_eligible,
            "live_trainer_argv": [],
        }


__all__ = ["CostateTrustRegionPolicy", "DirectFullCostatePolicy"]
