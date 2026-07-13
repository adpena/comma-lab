# SPDX-License-Identifier: MIT
"""Typed, fail-closed policy for reusing JRD coefficient-response priors.

This policy deliberately emits no trainer argv.  The PR110 response curves are
an n=1 macOS-CPU screen, so they may order future measurements but cannot set a
deployed precision or bit allocation until a real-GT n600 NumPy-fp32 receipt
passes the activation gate below.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal


class PriorState(StrEnum):
    DORMANT_N1_SCREEN = "DORMANT_N1_SCREEN"
    ACTIVE_N600_CONFIRMED = "ACTIVE_N600_CONFIRMED"


@dataclass(frozen=True, slots=True)
class N600PriorConfirmation:
    """Minimum authority receipt required before a screen prior can actuate."""

    eval_pairs: int
    real_gt: bool
    numpy_fp32_bit_identical: bool
    exact_r: bool
    separate_dseg_dpose: bool
    exact_archive_bytes: bool
    positive_repeat_noise_floor_zero: bool
    archive_sha256: str
    receipt_path: str
    receipt_sha256: str

    @property
    def admissible(self) -> bool:
        return (
            self.eval_pairs == 600
            and self.real_gt
            and self.numpy_fp32_bit_identical
            and self.exact_r
            and self.separate_dseg_dpose
            and self.exact_archive_bytes
            and self.positive_repeat_noise_floor_zero
            and len(self.archive_sha256) == 64
            and all(char in "0123456789abcdef" for char in self.archive_sha256)
            and self.receipt_path.startswith("experiments/results/")
            and self.receipt_path.endswith(".json")
            and len(self.receipt_sha256) == 64
            and all(char in "0123456789abcdef" for char in self.receipt_sha256)
        )


@dataclass(frozen=True, slots=True)
class JrdReusablePriorPolicy:
    """Machine-readable JRD routing prior with an explicit evidence firewall.

    ``bias_plane_hypothesis`` and ``weight_plane_hypothesis`` are the
    operator-supplied warm-start ranges, not measurements established by the
    PR110 JSON.  They remain dormant until :class:`N600PriorConfirmation` is
    admissible.  The coefficient masses and pose-only section identities are
    exact facts re-derived from the named PR110 packet and n=1 curve JSON.
    """

    bias_plane_hypothesis: tuple[int, int] = (5, 6)
    weight_plane_hypothesis: tuple[int, int] = (3, 4)
    hypothesis_evidence_label: Literal["ASSUMED_OPERATOR_WARM_START"] = "ASSUMED_OPERATOR_WARM_START"
    screen_eval_pairs: Literal[1] = 1
    activation_eval_pairs: Literal[600] = 600
    measured_bias_coefficients: Literal[2446] = 2446
    measured_weight_coefficients: Literal[226512] = 226512
    measured_total_coefficients: Literal[228958] = 228958
    pose_only_screen_sections: tuple[str, ...] = (
        "refine.1.bias",
        "rgb_0.bias",
        "rgb_0.weight",
    )
    research_only: Literal[True] = True
    score_claim: Literal[False] = False
    promotion_eligible: Literal[False] = False

    def __post_init__(self) -> None:
        if self.measured_bias_coefficients + self.measured_weight_coefficients != self.measured_total_coefficients:
            raise ValueError("bias and weight coefficient counts must close exactly")
        for low, high in (self.bias_plane_hypothesis, self.weight_plane_hypothesis):
            if not (0 <= low <= high <= 8):
                raise ValueError("int8 precision-plane hypotheses must lie in [0, 8]")

    @property
    def derived_mass_prior(self) -> dict[str, float]:
        """Coefficient-mass-only byte prior; it carries no distortion authority."""

        total = float(self.measured_total_coefficients)
        return {
            "bias": self.measured_bias_coefficients / total,
            "weight": self.measured_weight_coefficients / total,
        }

    def state(self, confirmation: N600PriorConfirmation | None = None) -> PriorState:
        if confirmation is not None and confirmation.admissible:
            return PriorState.ACTIVE_N600_CONFIRMED
        return PriorState.DORMANT_N1_SCREEN

    def compile_warm_start(self, confirmation: N600PriorConfirmation | None = None) -> dict[str, object]:
        """Compile a measurement order; refuse precision actuation before n600."""

        state = self.state(confirmation)
        active = state is PriorState.ACTIVE_N600_CONFIRMED
        return {
            "state": state.value,
            "active": active,
            "screen_eval_pairs": self.screen_eval_pairs,
            "activation_eval_pairs": self.activation_eval_pairs,
            "bias_plane_hypothesis": list(self.bias_plane_hypothesis),
            "weight_plane_hypothesis": list(self.weight_plane_hypothesis),
            "hypothesis_evidence_label": self.hypothesis_evidence_label,
            "derived_mass_prior": self.derived_mass_prior,
            "mass_prior_evidence_label": "DERIVED_FROM_MEASURED_PR110_PACKET_SCHEMA",
            "pose_only_screen_sections": list(self.pose_only_screen_sections),
            "allowed_use": (
                "initialize n600 measurement ordering"
                if not active
                else "initialize a receiver-closed n600 candidate search"
            ),
            "precision_actuation": ("REFUSED_PENDING_N600_CONFIRMATION" if not active else "N600_GATE_PASSED"),
            "activation_receipt": (
                None
                if not active or confirmation is None
                else {
                    "path": confirmation.receipt_path,
                    "sha256": confirmation.receipt_sha256,
                    "archive_sha256": confirmation.archive_sha256,
                }
            ),
            "live_trainer_argv": [],
            "research_only": self.research_only,
            "score_claim": self.score_claim,
            "promotion_eligible": self.promotion_eligible,
        }

    def search_family_route(
        self,
        *,
        same_pr110_archive: bool,
        family: str,
        witness_integrated_training: bool,
    ) -> str:
        """Stop exact reruns while leaving structurally different families open."""

        exhausted = {"uniform", "laplace_dead_zone"}
        if same_pr110_archive and family in exhausted and not witness_integrated_training:
            return "STOP_EXACT_PR110_POSTHOC_RERUN"
        if witness_integrated_training:
            return "ROUTE_TO_N600_WITNESS_MEASUREMENT"
        return "SEARCH_ONLY_IF_STRUCTURE_DIFFERS_AND_EXACT_BYTES_ARE_MEASURED"


__all__ = ["JrdReusablePriorPolicy", "N600PriorConfirmation", "PriorState"]
