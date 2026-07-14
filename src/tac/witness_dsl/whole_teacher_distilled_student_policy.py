# SPDX-License-Identifier: MIT
"""Typed, argv-inert policy for the whole-teacher distilled student.

The student is a throughput *means*: this module can admit an offline cached
measurement and prepare a governed integration packet, but it cannot activate
the live witness trainer.  The two authorities stay separate deliberately so
forward-only fidelity, a local timing row, or an operator-GO packet cannot be
laundered into a training-gradient or score claim.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from tac.witness_dsl.curriculum_dsl import Lever

POLICY_NAME = "whole_teacher_distilled_student"
REQUIRED_N_PAIRS = 600
MEASUREMENT_AXIS = "[n600 macOS-MLX advisory; NumPy-fp32 reference; no score authority]"
EXACT_COSTATE_REUSE_KMAX = 2
STUDENT_ANCHOR_CADENCE_CHOICES = (2, 4, 8, 20, 32, 64, 128)
VERDICT_SCOPE = (
    "INSTANCE x INPUT-CACHE x STUDENT-SIZE x CADENCE; a failed or missing receipt "
    "does not close the whole-teacher distilled-student family"
)
REQ_R = (
    "reconstruct a content-bound n600 real rendered-state bundle through the actual R "
    "surface, including full teacher quotient and exact input-VJP custody; then rerun the "
    "preregistered worst-pair value/VJP gates and matched in-loop timing before activation"
)


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
        and len(set(value)) > 1
        and value != hashlib.sha256(b"").hexdigest()
    )


class StudentTarget(StrEnum):
    """Only the JEPA-settled decision quotient is admitted in this lane."""

    CENTERED_LOGIT_QUOTIENT_4D = "centered_logit_quotient"


class StudentSize(StrEnum):
    """Preregistered architecture-size arms; exact layouts live in the student module."""

    TINY = "tiny"
    SMALL = "small"
    MEDIUM = "medium"


class EvidenceTier(StrEnum):
    """Forward advisory and differentiable training authority are not interchangeable."""

    FORWARD_ADVISORY = "forward_advisory"
    TRAINING_GRADIENT = "training_gradient"


class AdmissionState(StrEnum):
    """Fail-closed offline states plus the sole non-actuating green state."""

    BLOCKED_DATA_CUSTODY = "BLOCKED_DATA_CUSTODY"
    BLOCKED_FORWARD_FIDELITY = "BLOCKED_FORWARD_FIDELITY"
    BLOCKED_VJP_FIDELITY = "BLOCKED_VJP_FIDELITY"
    BLOCKED_ECONOMICS = "BLOCKED_ECONOMICS"
    GO_READY_NOT_FIRED = "GO_READY_NOT_FIRED"


VERDICT_SCOPE_BY_STATE: dict[AdmissionState, str] = {
    AdmissionState.BLOCKED_DATA_CUSTODY: (
        "INSTANCE x INPUT-CACHE x SEMANTIC-CUSTODY; missing or drifted evidence says "
        "nothing about whole-teacher distilled-student fidelity"
    ),
    AdmissionState.BLOCKED_FORWARD_FIDELITY: (
        "FORMULATION x STUDENT-SIZE x FIT-POLICY x REAL-N600-REPLAY; one value-fidelity "
        "failure does not close the whole-teacher distilled-student family"
    ),
    AdmissionState.BLOCKED_VJP_FIDELITY: (
        "FORMULATION x STUDENT-SIZE x SCALAR-OBJECTIVE x FIT-POLICY x REAL-N600-REPLAY; "
        "one Jacobian failure does not close whole-teacher distillation"
    ),
    AdmissionState.BLOCKED_ECONOMICS: (
        "INSTANCE x TIER x STUDENT-SIZE x K-STUDENT x HARDWARE x TIMING-RECEIPT; one "
        "non-paying operating point does not close the architecture or family"
    ),
    AdmissionState.GO_READY_NOT_FIRED: VERDICT_SCOPE,
}

REQ_R_BY_STATE: dict[AdmissionState, str] = {
    AdmissionState.BLOCKED_DATA_CUSTODY: REQ_R,
    AdmissionState.BLOCKED_FORWARD_FIDELITY: (
        "change the typed student architecture, size, or preregistered fit policy; reseal its "
        "parameters and repeat the same real-n600 NumPy-fp32 worst-pair quotient and parity gates"
    ),
    AdmissionState.BLOCKED_VJP_FIDELITY: (
        "change the typed differentiable architecture or preregistered scalar-composed Sobolev "
        "fit; then repeat the exact full-input-VJP and NumPy/MLX parity gates on the same real n600"
    ),
    AdmissionState.BLOCKED_ECONOMICS: (
        "measure another preregistered tier, size, or K_student on content-bound matched hardware, "
        "or reduce fully charged student/update cost before requesting in-loop activation"
    ),
    AdmissionState.GO_READY_NOT_FIRED: (
        "obtain explicit operator GO, land the reviewed provider seam, and pass the governed "
        "matched-window/full-facet treatment before any activation"
    ),
}


@dataclass(frozen=True)
class FidelityGates:
    """Immutable thresholds preregistered before reading heldout student results.

    The values are ASSUMED design thresholds rather than empirical laws.  Their
    provenance travels in every compiled contract; changing one creates a new
    policy hash and therefore cannot silently reinterpret an existing receipt.
    """

    forward_worst_pair_min_cosine: float = 0.995
    forward_worst_pair_max_relative_l2: float = 0.05
    forward_worst_pair_max_argmax_disagreement: float = 0.005
    vjp_worst_pair_min_cosine: float = 0.95
    vjp_worst_pair_max_relative_l2: float = 0.25
    numpy_framework_min_cosine: float = 0.9997
    provenance: str = (
        "ASSUMED_AWAITING_VERIFICATION; preregistered 2026-07-13 before any "
        "whole-teacher student heldout bundle was available"
    )

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if name == "provenance":
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise ValueError(f"{name} must be numeric")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        cosine_fields = (
            self.forward_worst_pair_min_cosine,
            self.vjp_worst_pair_min_cosine,
            self.numpy_framework_min_cosine,
        )
        if any(not -1.0 <= float(value) <= 1.0 for value in cosine_fields):
            raise ValueError("cosine gates must lie in [-1, 1]")
        error_fields = (
            self.forward_worst_pair_max_relative_l2,
            self.forward_worst_pair_max_argmax_disagreement,
            self.vjp_worst_pair_max_relative_l2,
        )
        if any(float(value) < 0.0 for value in error_fields):
            raise ValueError("error gates must be non-negative")
        if self.forward_worst_pair_max_argmax_disagreement > 1.0:
            raise ValueError("argmax disagreement is a fraction in [0, 1]")
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise ValueError("fidelity gates require non-empty provenance")


@dataclass(frozen=True)
class StudentAdmissionEvidence:
    """One content-bound n600 offline admission summary.

    Raw values are optional only so a missing bundle can be represented without
    placeholders.  A gate requiring a missing value refuses admission.
    """

    n_pairs: int = 0
    cache_manifest_valid: bool = False
    real_rendered_states: bool = False
    exact_teacher_quotient_custody: bool = False
    exact_teacher_input_vjp_custody: bool = False
    actual_r_custody_verified: bool = False
    frozen_teacher_custody_verified: bool = False
    scalar_objective_custody_verified: bool = False
    deterministic_repeat_verified: bool = False
    replay_states: tuple[str, ...] = ()
    measured_tier: EvidenceTier | None = None
    measured_student_size: StudentSize | None = None
    measured_student_anchor_cadence: int | None = None
    backend: str = "UNMEASURED"
    teacher_calls: int | None = None
    cache_manifest_sha256: str | None = None
    cache_manifest_file_sha256: str | None = None
    cache_validated_sha256: str | None = None
    source_custody_sha256: str | None = None
    teacher_source_custody_sha256: str | None = None
    actual_r_operator_sha256: str | None = None
    post_r_input_surface_sha256: str | None = None
    frozen_teacher_weights_sha256: str | None = None
    quotient_basis_sha256: str | None = None
    scalar_objective_sha256: str | None = None
    fit_policy_sha256: str | None = None
    parameter_layout_sha256: str | None = None
    student_parameters_sha256: str | None = None
    deterministic_repeat_sha256: str | None = None
    measurement_contract_sha256: str | None = None
    teacher_timing_receipt_sha256: str | None = None
    forward_worst_pair_cosine: float | None = None
    forward_worst_pair_relative_l2: float | None = None
    forward_worst_pair_argmax_disagreement: float | None = None
    vjp_worst_pair_cosine: float | None = None
    vjp_worst_pair_relative_l2: float | None = None
    numpy_framework_forward_worst_pair_cosine: float | None = None
    numpy_framework_vjp_worst_pair_cosine: float | None = None
    charged_timing_measured: bool = False
    student_forward_cost_ms: float | None = None
    student_forward_vjp_cost_ms: float | None = None
    exact_teacher_forward_cost_ms: float | None = None
    exact_teacher_forward_vjp_cost_ms: float | None = None
    anchor_update_cost_ms: float | None = None
    student_timing_axis: str = "UNMEASURED"
    teacher_timing_axis: str = "UNMEASURED"
    measurement_axis: str = "UNMEASURED"

    def __post_init__(self) -> None:
        if isinstance(self.n_pairs, bool) or not isinstance(self.n_pairs, int):
            raise ValueError("n_pairs must be an integer")
        if self.n_pairs < 0:
            raise ValueError("n_pairs must be non-negative")
        boolean_fields = {
            "cache_manifest_valid": self.cache_manifest_valid,
            "real_rendered_states": self.real_rendered_states,
            "exact_teacher_quotient_custody": self.exact_teacher_quotient_custody,
            "exact_teacher_input_vjp_custody": self.exact_teacher_input_vjp_custody,
            "actual_r_custody_verified": self.actual_r_custody_verified,
            "frozen_teacher_custody_verified": self.frozen_teacher_custody_verified,
            "scalar_objective_custody_verified": self.scalar_objective_custody_verified,
            "deterministic_repeat_verified": self.deterministic_repeat_verified,
            "charged_timing_measured": self.charged_timing_measured,
        }
        if any(not isinstance(value, bool) for value in boolean_fields.values()):
            raise ValueError("custody and timing flags must be boolean")
        optional_metrics = (
            self.forward_worst_pair_cosine,
            self.forward_worst_pair_relative_l2,
            self.forward_worst_pair_argmax_disagreement,
            self.vjp_worst_pair_cosine,
            self.vjp_worst_pair_relative_l2,
            self.numpy_framework_forward_worst_pair_cosine,
            self.numpy_framework_vjp_worst_pair_cosine,
            self.student_forward_cost_ms,
            self.student_forward_vjp_cost_ms,
            self.exact_teacher_forward_cost_ms,
            self.exact_teacher_forward_vjp_cost_ms,
            self.anchor_update_cost_ms,
        )
        if any(
            value is not None
            and (not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(float(value)))
            for value in optional_metrics
        ):
            raise ValueError("fidelity and timing metrics must be finite numbers or None")
        for name in (
            "forward_worst_pair_cosine",
            "vjp_worst_pair_cosine",
            "numpy_framework_forward_worst_pair_cosine",
            "numpy_framework_vjp_worst_pair_cosine",
        ):
            value = getattr(self, name)
            if value is not None and not -1.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must lie in [-1, 1]")
        for name in (
            "forward_worst_pair_relative_l2",
            "vjp_worst_pair_relative_l2",
        ):
            value = getattr(self, name)
            if value is not None and float(value) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        disagreement = self.forward_worst_pair_argmax_disagreement
        if disagreement is not None and not 0.0 <= float(disagreement) <= 1.0:
            raise ValueError("forward_worst_pair_argmax_disagreement must lie in [0, 1]")
        for name in (
            "student_forward_cost_ms",
            "student_forward_vjp_cost_ms",
            "exact_teacher_forward_cost_ms",
            "exact_teacher_forward_vjp_cost_ms",
            "anchor_update_cost_ms",
        ):
            value = getattr(self, name)
            if value is not None and float(value) < 0.0:
                raise ValueError(f"{name} must be non-negative")
        if self.measured_tier is not None and not isinstance(self.measured_tier, EvidenceTier):
            raise ValueError("measured_tier must be a typed EvidenceTier or None")
        if self.measured_student_size is not None and not isinstance(self.measured_student_size, StudentSize):
            raise ValueError("measured_student_size must be a typed StudentSize or None")
        if self.measured_student_anchor_cadence is not None and (
            isinstance(self.measured_student_anchor_cadence, bool)
            or not isinstance(self.measured_student_anchor_cadence, int)
            or self.measured_student_anchor_cadence < 1
        ):
            raise ValueError("measured_student_anchor_cadence must be an integer >= 1 or None")
        if self.teacher_calls is not None and (
            isinstance(self.teacher_calls, bool) or not isinstance(self.teacher_calls, int) or self.teacher_calls < 0
        ):
            raise ValueError("teacher_calls must be a non-negative integer or None")
        for name in ("backend", "measurement_axis", "student_timing_axis", "teacher_timing_axis"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
        for name in (
            "cache_manifest_sha256",
            "cache_manifest_file_sha256",
            "cache_validated_sha256",
            "source_custody_sha256",
            "teacher_source_custody_sha256",
            "actual_r_operator_sha256",
            "post_r_input_surface_sha256",
            "frozen_teacher_weights_sha256",
            "quotient_basis_sha256",
            "scalar_objective_sha256",
            "fit_policy_sha256",
            "parameter_layout_sha256",
            "student_parameters_sha256",
            "deterministic_repeat_sha256",
            "measurement_contract_sha256",
            "teacher_timing_receipt_sha256",
        ):
            value = getattr(self, name)
            if value is not None and not _is_sha256(value):
                raise ValueError(f"{name} must be a lowercase SHA-256 or None")

    @property
    def measurement_axis_valid(self) -> bool:
        """Accept only the exact registered cached-n600 MLX advisory axis."""

        return self.measurement_axis == MEASUREMENT_AXIS

    def binding_valid_for(self, policy: WholeTeacherDistilledStudentPolicy) -> bool:
        """Bind evidence to one exact policy, student, source, and semantic surface."""

        required_hashes = (
            self.cache_manifest_sha256,
            self.cache_manifest_file_sha256,
            self.cache_validated_sha256,
            self.source_custody_sha256,
            self.teacher_source_custody_sha256,
            self.actual_r_operator_sha256,
            self.post_r_input_surface_sha256,
            self.frozen_teacher_weights_sha256,
            self.quotient_basis_sha256,
            self.scalar_objective_sha256,
            self.fit_policy_sha256,
            self.parameter_layout_sha256,
            self.student_parameters_sha256,
            self.deterministic_repeat_sha256,
        )
        return (
            self.measured_tier is policy.tier
            and self.measured_student_size is policy.selected_size
            and self.measured_student_anchor_cadence == policy.student_anchor_cadence
            and self.backend == "mlx"
            and self.teacher_calls == 0
            and all(_is_sha256(value) for value in required_hashes)
            and self.measurement_contract_sha256 == policy.measurement_contract_sha256()
            and self.actual_r_custody_verified
            and self.frozen_teacher_custody_verified
            and self.scalar_objective_custody_verified
            and self.deterministic_repeat_verified
        )

    def derived_economics_for(
        self, policy: WholeTeacherDistilledStudentPolicy
    ) -> dict[str, float | bool | str | int | None]:
        """Derive the exact tier/K charged law from raw timing components."""

        student = (
            self.student_forward_cost_ms
            if policy.tier is EvidenceTier.FORWARD_ADVISORY
            else self.student_forward_vjp_cost_ms
        )
        teacher = (
            self.exact_teacher_forward_cost_ms
            if policy.tier is EvidenceTier.FORWARD_ADVISORY
            else self.exact_teacher_forward_vjp_cost_ms
        )
        timing_bound = (
            self.charged_timing_measured
            and _is_sha256(self.teacher_timing_receipt_sha256)
            and self.student_timing_axis == self.teacher_timing_axis
            and self.student_timing_axis == self.measurement_axis
            and student is not None
            and teacher is not None
            and teacher > 0.0
            and self.anchor_update_cost_ms is not None
        )
        charged = (
            float(student) + (float(teacher) + float(self.anchor_update_cost_ms)) / policy.student_anchor_cadence
            if timing_bound
            else None
        )
        strict_pays = bool(timing_bound and charged is not None and charged < float(teacher))
        inclusive_95 = bool(timing_bound and charged is not None and charged <= 0.05 * float(teacher))
        return {
            "tier": policy.tier.value,
            "student_size": policy.selected_size.value,
            "student_anchor_cadence": policy.student_anchor_cadence,
            "timing_bound": timing_bound,
            "student_cost_ms": float(student) if student is not None else None,
            "exact_teacher_cost_ms": float(teacher) if teacher is not None else None,
            "anchor_update_cost_ms": (
                float(self.anchor_update_cost_ms) if self.anchor_update_cost_ms is not None else None
            ),
            "charged_cost_ms": charged,
            "strict_pays": strict_pays,
            "inclusive_95": inclusive_95,
        }

    @property
    def forward_custody_valid(self) -> bool:
        """Require real n600 frames plus exact teacher quotient custody."""

        return (
            self.n_pairs == REQUIRED_N_PAIRS
            and self.cache_manifest_valid
            and self.real_rendered_states
            and self.exact_teacher_quotient_custody
            and self.measurement_axis_valid
            and len(self.replay_states) == 3
            and set(self.replay_states) == {"ep150", "ep251", "ep275"}
        )

    @property
    def training_gradient_custody_valid(self) -> bool:
        """Add full exact teacher input-VJP custody for the decisive tier."""

        return self.forward_custody_valid and self.exact_teacher_input_vjp_custody


@dataclass(frozen=True)
class AdmissionDecision:
    """Machine-readable scoped decision; never a live activation authority."""

    state: AdmissionState
    tier: EvidenceTier
    admitted: bool
    reasons: tuple[str, ...]
    verdict_scope: str = VERDICT_SCOPE
    req_R: str = REQ_R

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["state"] = self.state.value
        payload["tier"] = self.tier.value
        return payload


def _scoped_decision(
    *,
    state: AdmissionState,
    tier: EvidenceTier,
    admitted: bool,
    reasons: tuple[str, ...],
) -> AdmissionDecision:
    return AdmissionDecision(
        state=state,
        tier=tier,
        admitted=admitted,
        reasons=reasons,
        verdict_scope=VERDICT_SCOPE_BY_STATE[state],
        req_R=REQ_R_BY_STATE[state],
    )


@dataclass(frozen=True)
class WholeTeacherDistilledStudentPolicy:
    """Default-OFF research policy for a governed future in-loop surrogate."""

    enabled: bool = False
    research_only: bool = True
    target: StudentTarget = StudentTarget.CENTERED_LOGIT_QUOTIENT_4D
    selected_size: StudentSize = StudentSize.SMALL
    size_candidates: tuple[StudentSize, ...] = (
        StudentSize.TINY,
        StudentSize.SMALL,
        StudentSize.MEDIUM,
    )
    student_anchor_cadence: int = 20
    student_anchor_cadence_candidates: tuple[int, ...] = STUDENT_ANCHOR_CADENCE_CHOICES
    exact_costate_reuse_kmax: int | None = EXACT_COSTATE_REUSE_KMAX
    tier: EvidenceTier = EvidenceTier.TRAINING_GRADIENT
    seed: int = 455
    gates: FidelityGates = field(default_factory=FidelityGates)
    exact_anchor_fallback: str = "full_frozen_teacher_forward_and_input_vjp"
    operator_go_recorded: bool = False
    provider_current: bool = False
    teacher_recomputation_enabled: bool = False
    live_training_enabled: bool = False
    paid_or_remote_dispatch_enabled: bool = False
    live_run_mutation_enabled: bool = False
    evaluator_enabled: bool = False
    score_claim: bool = False
    promotion_eligible: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be boolean")
        if self.research_only is not True:
            raise ValueError("this isolated policy is research_only")
        if self.target is not StudentTarget.CENTERED_LOGIT_QUOTIENT_4D:
            raise ValueError("the JEPA target decision is sealed to the centered-logit quotient")
        if self.size_candidates != (
            StudentSize.TINY,
            StudentSize.SMALL,
            StudentSize.MEDIUM,
        ):
            raise ValueError("student size candidates are preregistered")
        if self.selected_size not in self.size_candidates:
            raise ValueError("selected_size must be a preregistered candidate")
        if not isinstance(self.tier, EvidenceTier):
            raise ValueError("tier must be a typed EvidenceTier")
        if not isinstance(self.gates, FidelityGates):
            raise ValueError("gates must be a FidelityGates contract")
        if self.student_anchor_cadence_candidates != STUDENT_ANCHOR_CADENCE_CHOICES:
            raise ValueError("student anchor cadence candidates are preregistered")
        if self.student_anchor_cadence not in self.student_anchor_cadence_candidates:
            raise ValueError("student_anchor_cadence must be a preregistered candidate")
        if self.exact_costate_reuse_kmax not in (None, EXACT_COSTATE_REUSE_KMAX):
            raise ValueError("the optional inner exact-costate reuse controller is sealed to #487 K_max=2")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.exact_anchor_fallback != "full_frozen_teacher_forward_and_input_vjp":
            raise ValueError("fallback must restore the exact whole teacher")
        if not isinstance(self.operator_go_recorded, bool):
            raise ValueError("operator_go_recorded must be boolean")
        if self.provider_current is not False:
            raise ValueError("the live trainer has no admitted student provider seam")
        containment_flags = {
            "teacher_recomputation_enabled": self.teacher_recomputation_enabled,
            "live_training_enabled": self.live_training_enabled,
            "paid_or_remote_dispatch_enabled": self.paid_or_remote_dispatch_enabled,
            "live_run_mutation_enabled": self.live_run_mutation_enabled,
            "evaluator_enabled": self.evaluator_enabled,
        }
        if any(value is not False for value in containment_flags.values()):
            raise ValueError("this policy is sealed to cached-only local measurement")
        if self.score_claim is not False or self.promotion_eligible is not False:
            raise ValueError("a throughput surrogate cannot claim score or promotion")

    def evaluate_evidence(self, evidence: StudentAdmissionEvidence | None = None) -> AdmissionDecision:
        """Apply the preregistered worst-pair and charged-economics gates."""

        observed = evidence or StudentAdmissionEvidence()
        custody_valid = (
            observed.forward_custody_valid
            if self.tier is EvidenceTier.FORWARD_ADVISORY
            else observed.training_gradient_custody_valid
        )
        if not custody_valid or not observed.binding_valid_for(self):
            return _scoped_decision(
                state=AdmissionState.BLOCKED_DATA_CUSTODY,
                tier=self.tier,
                admitted=False,
                reasons=(
                    "requires exactly 600 unique real rendered states over ep150/ep251/ep275",
                    (
                        "requires content-bound teacher quotient custody; the training-gradient "
                        "tier additionally requires the full exact input VJP"
                    ),
                    (
                        "requires receipt binding to the selected tier/size/K, MLX backend, zero "
                        "teacher calls, deterministic repeat, policy/parameter/source hashes, "
                        "and actual-R/frozen-teacher/scalar-objective semantic custody"
                    ),
                ),
            )

        forward_values = (
            observed.forward_worst_pair_cosine,
            observed.forward_worst_pair_relative_l2,
            observed.forward_worst_pair_argmax_disagreement,
            observed.numpy_framework_forward_worst_pair_cosine,
        )
        forward_passed = all(value is not None for value in forward_values) and (
            float(observed.forward_worst_pair_cosine)  # type: ignore[arg-type]
            >= self.gates.forward_worst_pair_min_cosine
            and float(observed.forward_worst_pair_relative_l2)  # type: ignore[arg-type]
            <= self.gates.forward_worst_pair_max_relative_l2
            and float(observed.forward_worst_pair_argmax_disagreement)  # type: ignore[arg-type]
            <= self.gates.forward_worst_pair_max_argmax_disagreement
            and float(observed.numpy_framework_forward_worst_pair_cosine)  # type: ignore[arg-type]
            >= self.gates.numpy_framework_min_cosine
        )
        if not forward_passed:
            return _scoped_decision(
                state=AdmissionState.BLOCKED_FORWARD_FIDELITY,
                tier=self.tier,
                admitted=False,
                reasons=("forward worst-pair or NumPy/framework parity gate failed or is unmeasured",),
            )

        if self.tier is EvidenceTier.TRAINING_GRADIENT:
            vjp_values = (
                observed.vjp_worst_pair_cosine,
                observed.vjp_worst_pair_relative_l2,
                observed.numpy_framework_vjp_worst_pair_cosine,
            )
            vjp_passed = all(value is not None for value in vjp_values) and (
                float(observed.vjp_worst_pair_cosine)  # type: ignore[arg-type]
                >= self.gates.vjp_worst_pair_min_cosine
                and float(observed.vjp_worst_pair_relative_l2)  # type: ignore[arg-type]
                <= self.gates.vjp_worst_pair_max_relative_l2
                and float(observed.numpy_framework_vjp_worst_pair_cosine)  # type: ignore[arg-type]
                >= self.gates.numpy_framework_min_cosine
            )
            if not vjp_passed:
                return _scoped_decision(
                    state=AdmissionState.BLOCKED_VJP_FIDELITY,
                    tier=self.tier,
                    admitted=False,
                    reasons=(
                        "decisive full exact input-VJP worst-pair cosine/relative-L2 or NumPy/framework VJP parity gate failed or is unmeasured",
                    ),
                )

        economics = observed.derived_economics_for(self)
        if not economics["strict_pays"]:
            return _scoped_decision(
                state=AdmissionState.BLOCKED_ECONOMICS,
                tier=self.tier,
                admitted=False,
                reasons=("fully charged matched-device student/teacher/update timing is unmeasured or does not pay",),
            )
        return _scoped_decision(
            state=AdmissionState.GO_READY_NOT_FIRED,
            tier=self.tier,
            admitted=True,
            reasons=(
                "offline measurement gates passed; operator-GO and a separately reviewed provider integration remain required",
            ),
        )

    def compile_trainer_argv(self) -> tuple[str, ...]:
        """Return no argv: the governed trainer integration is intentionally not built."""

        return ()

    def compile_measurement_contract(self) -> dict[str, Any]:
        """Compile immutable offline requirements and value provenance."""

        return {
            "policy": POLICY_NAME,
            "enabled": self.enabled,
            "research_only": self.research_only,
            "containment": {
                "teacher_recomputation_enabled": self.teacher_recomputation_enabled,
                "live_training_enabled": self.live_training_enabled,
                "paid_or_remote_dispatch_enabled": self.paid_or_remote_dispatch_enabled,
                "live_run_mutation_enabled": self.live_run_mutation_enabled,
                "evaluator_enabled": self.evaluator_enabled,
            },
            "target": self.target.value,
            "selected_size": self.selected_size.value,
            "size_candidates": [value.value for value in self.size_candidates],
            "student_anchor_cadence": self.student_anchor_cadence,
            "student_anchor_cadence_candidates": list(self.student_anchor_cadence_candidates),
            "exact_costate_reuse_kmax": self.exact_costate_reuse_kmax,
            "cadence_composition_law": (
                "K_student controls periodic whole-teacher anchors independently; optional "
                "#487 K_max=2 controls only its own inner stale-costate reuse attempt and "
                "contributes no inherited speed claim"
            ),
            "tier": self.tier.value,
            "seed": self.seed,
            "required_n_pairs": REQUIRED_N_PAIRS,
            "required_replay_states": ["ep150", "ep251", "ep275"],
            "required_cache_fields": {
                "all_tiers": [
                    "rendered_frame",
                    "teacher_quotient4",
                    "labels",
                    "per_tensor_sha256",
                ],
                "training_gradient_additional": ["teacher_input_costate"],
            },
            "required_evidence_binding_fields": [
                "measured_tier",
                "measured_student_size",
                "measured_student_anchor_cadence",
                "cache_manifest_sha256",
                "cache_manifest_file_sha256",
                "cache_validated_sha256",
                "source_custody_sha256",
                "teacher_source_custody_sha256",
                "actual_r_operator_sha256",
                "post_r_input_surface_sha256",
                "frozen_teacher_weights_sha256",
                "quotient_basis_sha256",
                "scalar_objective_sha256",
                "fit_policy_sha256",
                "parameter_layout_sha256",
                "student_parameters_sha256",
                "deterministic_repeat_sha256",
                "measurement_contract_sha256",
                "teacher_timing_receipt_sha256",
            ],
            "semantic_custody_requirements": {
                "actual_r_custody_verified": True,
                "frozen_teacher_custody_verified": True,
                "scalar_objective_custody_verified": True,
                "deterministic_repeat_verified": True,
                "backend": "mlx",
                "measurement_axis": MEASUREMENT_AXIS,
                "teacher_calls": 0,
            },
            "economics_law": "C_S,t + (C_T,t + U) / K_student",
            "economics_raw_timing_fields": {
                "forward_advisory": [
                    "student_forward_cost_ms",
                    "exact_teacher_forward_cost_ms",
                ],
                "training_gradient": [
                    "student_forward_vjp_cost_ms",
                    "exact_teacher_forward_vjp_cost_ms",
                ],
                "all_tiers": [
                    "anchor_update_cost_ms",
                    "student_timing_axis",
                    "teacher_timing_axis",
                ],
            },
            "gates": asdict(self.gates),
            "exact_anchor_fallback": self.exact_anchor_fallback,
            "forced_refresh_boundaries": ["event", "stage", "custody_change", "trust_region"],
            "numerical_authority": (
                "teacher fidelity is decided by the NumPy-fp32 reference; MLX is advisory "
                "for matched-device timing and owes separate forward and input-VJP parity"
            ),
            "measurement_authority": (
                "real cached n600 rendered states through actual R; worst-pair QoI; exact teacher input-VJP"
            ),
            "live_trainer_argv": list(self.compile_trainer_argv()),
            "verdict_scope": VERDICT_SCOPE,
            "req_R": REQ_R,
            "negative_dispositions": {
                state.value: {
                    "verdict_scope": VERDICT_SCOPE_BY_STATE[state],
                    "req_R": REQ_R_BY_STATE[state],
                }
                for state in (
                    AdmissionState.BLOCKED_DATA_CUSTODY,
                    AdmissionState.BLOCKED_FORWARD_FIDELITY,
                    AdmissionState.BLOCKED_VJP_FIDELITY,
                    AdmissionState.BLOCKED_ECONOMICS,
                )
            },
            "constant_provenance": {
                "target": "DERIVED JEPA decision-quotient memo and softmax gauge law",
                "student_anchor_cadence_candidates": (
                    "SOURCE operator K economics request; K=20 comparison plus DERIVED "
                    "power-of-two bracketing above the inclusive-95 feasibility boundary"
                ),
                "exact_costate_reuse_kmax": (
                    "MEASURED-INHERITED #487 guarded exact-costate-reuse policy; separate "
                    "inner controller with no cadence cap or speed transfer to K_student"
                ),
                "seed": "MEASURED-INHERITED sealed Round-5 deterministic seed",
                "fidelity_gates": self.gates.provenance,
            },
            "score_claim": False,
            "promotion_eligible": False,
        }

    def measurement_contract_sha256(self) -> str:
        """Hash the exact immutable contract that a measurement must cite."""

        encoded = json.dumps(
            self.compile_measurement_contract(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def compile_activation_contract(self, evidence: StudentAdmissionEvidence | None = None) -> dict[str, Any]:
        """Expose a GO-packet state while refusing all live activation."""

        decision = self.evaluate_evidence(evidence)
        activation_errors = [
            "policy is default-off" if not self.enabled else "policy enable is not sufficient",
            "live student provider is not integrated",
            "live trainer argv is empty",
        ]
        if not self.operator_go_recorded:
            activation_errors.append("operator-GO is not recorded")
        if not decision.admitted:
            activation_errors.extend(decision.reasons)
        return {
            **self.compile_measurement_contract(),
            "offline_admission": decision.to_dict(),
            "go_packet_state": (AdmissionState.GO_READY_NOT_FIRED.value if decision.admitted else decision.state.value),
            "operator_go_recorded": self.operator_go_recorded,
            "provider_current": self.provider_current,
            "trainer_activation_admitted": False,
            "trainer_activation_authority": "REFUSED_NO_PROVIDER_OR_ARGV",
            "trainer_activation_errors": activation_errors,
        }


def whole_teacher_distilled_student_lever(
    policy: WholeTeacherDistilledStudentPolicy | None = None,
    evidence: StudentAdmissionEvidence | None = None,
) -> Lever:
    """Return the named default-OFF DSL leg with zero trainer overrides."""

    selected = policy or WholeTeacherDistilledStudentPolicy()
    contract = selected.compile_activation_contract(evidence)
    decision = contract["offline_admission"]
    return Lever(
        name=POLICY_NAME,
        overrides={},
        epochs_delta=0,
        notes=(
            f"research-only argv-inert student policy; offline={decision['state']}; "
            "trainer=REFUSED; exact-anchor fallback and operator-GO remain binding"
        ),
    )


__all__ = [
    "EXACT_COSTATE_REUSE_KMAX",
    "MEASUREMENT_AXIS",
    "POLICY_NAME",
    "REQUIRED_N_PAIRS",
    "REQ_R",
    "REQ_R_BY_STATE",
    "STUDENT_ANCHOR_CADENCE_CHOICES",
    "VERDICT_SCOPE",
    "VERDICT_SCOPE_BY_STATE",
    "AdmissionDecision",
    "AdmissionState",
    "EvidenceTier",
    "FidelityGates",
    "StudentAdmissionEvidence",
    "StudentSize",
    "StudentTarget",
    "WholeTeacherDistilledStudentPolicy",
    "whole_teacher_distilled_student_lever",
]
