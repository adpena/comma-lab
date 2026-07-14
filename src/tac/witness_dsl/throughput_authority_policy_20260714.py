# SPDX-License-Identifier: MIT
"""Typed op/substrate/precision policy for Task #494.

The compiler is pure and default-safe: it never launches work and it never
promotes a research-signal backend to score authority.  Receipt predicates are
deliberately conjunctive so a QDQ feasibility result, a fast kernel, or a
cross-process digest cannot stand in for the other two.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any


class Operation(StrEnum):
    WITNESS_FORWARD_BACKWARD = "witness_forward_backward"
    R_FORWARD = "render_R_forward"
    R_ADJOINT = "render_R_adjoint"
    SEGNET_VERDICT = "segnet_forward_verdict"
    POSENET_VERDICT_PRE_FINISH = "posenet_verdict_pre_pose_finish"
    POSENET_VERDICT_ACTIVE = "posenet_verdict_pose_active"
    ARCHIVE_DECODE = "archive_decode"
    TERMINAL_EVALUATOR = "terminal_evaluator"


class Substrate(StrEnum):
    NUMPY_CPU = "numpy_cpu"
    TORCH_CPU_ONE_THREAD = "torch_cpu_one_thread"
    MLX_METAL = "mlx_metal"
    CUSTOM_METAL = "custom_metal"
    COREML_ANE = "coreml_ane"
    TORCH_MPS = "torch_mps"
    CONTEST_CUDA = "contest_cuda"
    BANKED_TELEMETRY = "banked_telemetry"


class Precision(StrEnum):
    NUMPY_FP32 = "numpy_fp32"
    TORCH_FP32 = "torch_fp32"
    MLX_FP32 = "mlx_fp32"
    INTEGER_Q_INT32 = "integer_q_int32_accum"
    FIXEDPOINT_MIXED = "fixedpoint_receipt_selected"
    COREML_FP32 = "coreml_fp32"
    COREML_W8A8 = "coreml_w8a8"
    BANKED_FP64_SCALAR = "banked_fp64_scalar"


class AssignmentState(StrEnum):
    ACTIVE = "active"
    DEFAULT_OFF_CANDIDATE = "default_off_candidate"
    HELD_OWED = "held_owed"
    ADVISORY_ONLY = "advisory_only"
    FORBIDDEN = "forbidden"
    OPERATOR_GO_ONLY = "operator_go_only"


class AuthorityGrade(StrEnum):
    TRAINING_SIGNAL = "training_signal"
    LOCAL_REFERENCE = "local_reference"
    LOCAL_CANDIDATE_FILTER = "local_candidate_filter"
    LOCAL_ADVISORY = "local_advisory"
    CONTEST_AUTHORITY = "contest_authority"
    NO_AUTHORITY = "no_authority"


@dataclass(frozen=True)
class Assignment:
    operation: Operation
    substrate: Substrate
    precision: Precision
    state: AssignmentState
    authority_grade: AuthorityGrade
    reason: str
    evidence: str
    required_next_gate: str | None = None
    selected_bits: int | None = None
    activation_scale_mode: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for key in ("operation", "substrate", "precision", "state", "authority_grade"):
            payload[key] = getattr(self, key).value
        return payload


@dataclass(frozen=True)
class ThroughputAuthorityPolicy:
    assignments: tuple[Assignment, ...]
    pose_gate_enabled: bool
    pose_canary_every: int
    banked_r1_dpose: float
    research_only: bool = True
    score_claim: bool = False
    pointer_moved: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": "throughput_authority_policy.v1",
            "assignments": [row.to_dict() for row in self.assignments],
            "pose_gate_enabled": self.pose_gate_enabled,
            "pose_canary_every": self.pose_canary_every,
            "banked_r1_dpose": self.banked_r1_dpose,
            "research_only": self.research_only,
            "score_claim": self.score_claim,
            "pointer_moved": self.pointer_moved,
        }


def _fixedpoint_qdq_gate(
    receipt: Mapping[str, Any] | None,
) -> tuple[bool, int | None, str | None, str]:
    if not receipt:
        return False, None, None, "full real-n600 QDQ receipt is absent"
    contract = receipt.get("contract", {})
    summary = receipt.get("summary", {})
    activation_scale_mode = contract.get("activation_scale_mode") or (
        "fixed_calibration"
        if receipt.get("schema") == "fixedpoint_scorer_forward_n600.v2"
        else None
    )
    expected_schema = {
        "fixed_calibration": "fixedpoint_scorer_forward_n600.v2",
        "dynamic_exact_absmax": "dynamic_fixedpoint_scorer_forward_n600.v1",
    }.get(activation_scale_mode)
    if receipt.get("schema") != expected_schema:
        return False, None, None, "QDQ receipt schema/activation-scale-mode mismatch"
    if contract.get("native_integer_speed_claim") is not False:
        return False, None, None, "QDQ receipt must explicitly disclaim native integer speed"
    if summary.get("full_real_n600") is not True:
        return False, None, None, "QDQ receipt lacks exact 0..599 custody"
    custody = summary.get("cache_custody", {})
    if not (
        custody.get("status") == "MEASURED"
        and int(custody.get("pairs", -1)) == 600
        and int(custody.get("unique_pair_indices", -1)) == 600
        and custody.get("observed_pair_indices_sha256")
        == custody.get("expected_pair_indices_sha256")
    ):
        return False, None, None, "QDQ receipt exact pair-index custody is incomplete"
    arm = summary.get("minimum_argmax_exact_arm")
    if not isinstance(arm, str) or not arm.startswith("w") or "a" not in arm:
        return False, None, None, "no full-n600 exact-argmax fixed-point arm"
    try:
        bits = int(arm[1 : arm.index("a")])
    except ValueError:
        return False, None, None, "malformed exact-argmax arm"
    row = summary.get("arms", {}).get(arm, {})
    if row.get("argmax_exact_admitted") is not True:
        return False, None, None, "selected QDQ arm is not exact-argmax admitted"
    return (
        True,
        bits,
        str(activation_scale_mode),
        f"full-n600 {activation_scale_mode} QDQ feasibility selects {arm}; speed remains unclaimed",
    )


def _integer_scorer_gate(
    receipt: Mapping[str, Any] | None,
    *,
    required_qdq_fingerprint: str | None,
) -> tuple[bool, int | None, str | None, str | None, str]:
    if not receipt:
        return False, None, None, None, "exact-int64 scorer receipt is absent"
    schema = receipt.get("schema")
    summary = receipt.get("summary", {})
    manifest = receipt.get("model_manifest", {})
    contract = receipt.get("contract", {})
    custody = summary.get("cache_custody", {})
    common_ok = bool(
        summary.get("status") == "MEASURED"
        and summary.get("full_real_n600") is True
        and summary.get("argmax_exact_admitted") is True
        and custody.get("status") == "MEASURED"
        and int(custody.get("pairs", -1)) == 600
        and int(custody.get("unique_pair_indices", -1)) == 600
        and custody.get("observed_pair_indices_sha256")
        == custody.get("expected_pair_indices_sha256")
        and int(manifest.get("converted_conv2d_count", -1)) == 125
        and manifest.get("accumulation") == "exact_signed_int64"
        and contract.get("native_integer_speed_claim") is True
        and contract.get("activation_scale_mode") == "dynamic_exact_absmax"
    )
    if not common_ok:
        return False, None, None, None, "exact-int64 scorer n600 conjunction is incomplete"
    if (
        required_qdq_fingerprint is None
        or receipt.get("custody", {}).get("qdq_precursor_fingerprint")
        != required_qdq_fingerprint
    ):
        return False, None, None, None, "exact-int64 scorer is not bound to this QDQ receipt"
    if schema == "exact_int64_fixedpoint_scorer_n600.v1":
        bits = int(summary.get("bits", -1))
        if bits < 2 or int(manifest.get("bits", -1)) != bits:
            return False, None, None, None, "uniform exact-int64 precision manifest differs"
        assignment = f"uniform_W{bits}A{bits}"
    elif schema == "mixed_int64_fixedpoint_scorer_n600.v1":
        bits = int(summary.get("minimum_bits", -1))
        maximum_bits = int(summary.get("maximum_bits", -1))
        if (
            bits != 26
            or maximum_bits != 30
            or int(manifest.get("minimum_bits", -1)) != bits
            or int(manifest.get("maximum_bits", -1)) != maximum_bits
            or manifest.get("assignment_rule")
            != "largest_geometry_safe_bits_with_signed_int64_static_bound"
        ):
            return False, None, None, None, "mixed exact-int64 precision manifest differs"
        assignment = "geometry_safe_W26_to_W30"
    elif schema == "weight_l1_int64_fixedpoint_scorer_n600.v1":
        bits = int(summary.get("minimum_bits", -1))
        maximum_bits = int(summary.get("maximum_bits", -1))
        if (
            bits != 26
            or maximum_bits != 31
            or int(manifest.get("minimum_bits", -1)) != bits
            or int(manifest.get("maximum_bits", -1)) != maximum_bits
            or manifest.get("assignment_rule")
            != "largest_frozen_weight_l1_safe_bits_with_signed_int64_bound"
            or manifest.get("bound_kind")
            != "activation_qmax_times_max_output_quantized_weight_l1"
            or manifest.get("label_or_frame_dependent") is not False
        ):
            return False, None, None, None, "weight-L1 exact-int64 precision manifest differs"
        assignment = "frozen_weight_l1_safe_W26_to_W31"
    elif schema == "weight_l1_tie_snap_scorer_n600.v1":
        bits = int(manifest.get("minimum_bits", -1))
        maximum_bits = int(manifest.get("maximum_bits", -1))
        epsilon = summary.get("minimum_calibration_exact_epsilon")
        if (
            bits != 26
            or maximum_bits != 31
            or int(manifest.get("converted_conv2d_count", -1)) != 125
            or manifest.get("assignment_rule")
            != "largest_frozen_weight_l1_safe_bits_with_signed_int64_bound"
            or manifest.get("bound_kind")
            != "activation_qmax_times_max_output_quantized_weight_l1"
            or manifest.get("label_or_frame_dependent") is not False
            or not summary.get("minimum_calibration_exact_arm")
            or epsilon is None
            or float(epsilon) <= 0.0
            or summary.get("selected_heldout_exact") is not True
            or summary.get("selected_full_exact") is not True
            or contract.get("decision_rule")
            != "lowest class index within epsilon of candidate maximum"
            or contract.get("epsilon_selection")
            != "minimum calibration-exact epsilon; no heldout reselection"
            or contract.get("runtime_label_or_frame_dependent") is not False
        ):
            return False, None, None, None, "weight-L1 tie-snap precision manifest differs"
        assignment = f"frozen_weight_l1_safe_W26_to_W31_tie_snap_{float(epsilon).hex()}"
    elif schema == "weight_l1_class_pair_tie_snap_scorer_n600.v1":
        bits = int(manifest.get("minimum_bits", -1))
        maximum_bits = int(manifest.get("maximum_bits", -1))
        epsilon = contract.get("epsilon")
        winner = contract.get("candidate_winner_class")
        runner = contract.get("candidate_runner_class")
        replacement = contract.get("replacement_class")
        if (
            bits != 26
            or maximum_bits != 31
            or int(manifest.get("converted_conv2d_count", -1)) != 125
            or manifest.get("assignment_rule")
            != "largest_frozen_weight_l1_safe_bits_with_signed_int64_bound"
            or manifest.get("bound_kind")
            != "activation_qmax_times_max_output_quantized_weight_l1"
            or manifest.get("label_or_frame_dependent") is not False
            or summary.get("design_exact") is not True
            or summary.get("second_validation_exact") is not True
            or contract.get("design_split") != [0, 264]
            or contract.get("second_validation_split") != [264, 600]
            or winner != 4
            or runner != 0
            or replacement != 0
            or epsilon is None
            or float(epsilon) != float(2.0**-19)
            or contract.get("rule_frozen_before_second_validation_access") is not True
            or contract.get("second_validation_reselection") is not False
            or contract.get("runtime_label_or_frame_dependent") is not False
        ):
            return (
                False,
                None,
                None,
                None,
                "weight-L1 class-pair tie-snap precision manifest differs",
            )
        assignment = (
            f"frozen_weight_l1_safe_W26_to_W31_class_pair_tie_snap_w{winner}_r{runner}"
            f"_to{replacement}_eps_{float(epsilon).hex()}"
        )
    else:
        return False, None, None, None, "exact-int64 scorer receipt schema mismatch"
    return (
        True,
        bits,
        "dynamic_exact_absmax",
        assignment,
        f"full-n600 exact-int64 scorer admits {assignment}",
    )


def _metal_gate(
    receipt: Mapping[str, Any] | None,
    *,
    required_bits: int | None,
    required_scale_mode: str | None,
    required_qdq_fingerprint: str | None,
    required_precision_assignment: str | None,
    required_integer_fingerprint: str | None,
) -> tuple[bool, str]:
    if not receipt:
        return False, "custom-Metal receipt is absent"
    if receipt.get("schema") != "metal_fixedpoint_segnet_n600.v1":
        return False, "custom-Metal receipt schema mismatch"
    contract = receipt.get("contract", {})
    if required_bits is None or contract.get("bits") != required_bits:
        return False, "custom-Metal bits do not match the admitted QDQ arm"
    if required_scale_mode is None or contract.get("activation_scale_mode") != required_scale_mode:
        return False, "custom-Metal activation scale mode does not match QDQ"
    if (
        required_precision_assignment is None
        or contract.get("precision_assignment") != required_precision_assignment
    ):
        return False, "custom-Metal precision assignment does not match the admitted precursor"
    if required_qdq_fingerprint is None or contract.get("qdq_receipt_fingerprint") != required_qdq_fingerprint:
        return False, "custom-Metal receipt is not bound to the admitted QDQ receipt"
    if (
        required_integer_fingerprint is not None
        and contract.get("exact_int64_cpu_precursor_fingerprint")
        != required_integer_fingerprint
    ):
        return False, "custom-Metal receipt is not bound to the admitted exact-int64 precursor"
    summary = receipt.get("summary", {})
    gates = (
        "complete",
        "full_real_n600",
        "cross_process_argmax_identical",
        "argmax_exact",
        "positive_speed",
        "admitted_candidate_authority_filter",
    )
    missing = [name for name in gates if summary.get(name) is not True]
    if missing:
        return False, "custom-Metal conjunction failed: " + ",".join(missing)
    return True, "n600 exact/certified/cross-process/speed conjunction passed"


def _integer_r_gate(receipt: Mapping[str, Any] | None) -> tuple[bool, str]:
    if not receipt:
        return False, "integer R-adjoint receipt is absent"
    if receipt.get("schema") != "integer_r_adjoint_backend_benchmark.v1":
        return False, "integer R-adjoint receipt schema mismatch"
    admission = receipt.get("admission", {})
    if admission.get("admitted_for_training") is not True:
        return False, "integer R-adjoint n600 parity/determinism/speed conjunction is incomplete"
    return True, "full-n600 bounded-error/determinism/speed conjunction passed"


def compile_throughput_authority_policy(
    *,
    fixedpoint_qdq_receipt: Mapping[str, Any] | None = None,
    integer_scorer_receipt: Mapping[str, Any] | None = None,
    metal_fixedpoint_receipt: Mapping[str, Any] | None = None,
    integer_r_receipt: Mapping[str, Any] | None = None,
    pose_gate_enabled: bool = True,
    pose_canary_every: int = 8,
    banked_r1_dpose: float = 0.001610,
) -> ThroughputAuthorityPolicy:
    """Compile the fastest assignment that preserves the declared authority ladder."""

    if isinstance(pose_canary_every, bool) or pose_canary_every < 1:
        raise ValueError("pose_canary_every must be a positive integer")
    if not (0.0 <= float(banked_r1_dpose) < float("inf")):
        raise ValueError("banked_r1_dpose must be finite and non-negative")

    qdq_ok, qdq_bits, qdq_scale_mode, qdq_reason = _fixedpoint_qdq_gate(
        fixedpoint_qdq_receipt
    )
    qdq_fingerprint = (
        str(fixedpoint_qdq_receipt.get("fingerprint"))
        if fixedpoint_qdq_receipt and fixedpoint_qdq_receipt.get("fingerprint")
        else None
    )
    (
        integer_scorer_ok,
        integer_scorer_bits,
        integer_scorer_scale_mode,
        integer_scorer_assignment,
        integer_scorer_reason,
    ) = _integer_scorer_gate(
        integer_scorer_receipt,
        required_qdq_fingerprint=qdq_fingerprint,
    )
    scorer_numerics_ok = qdq_ok or integer_scorer_ok
    scorer_bits = integer_scorer_bits if integer_scorer_ok else qdq_bits
    scorer_scale_mode = (
        integer_scorer_scale_mode if integer_scorer_ok else qdq_scale_mode
    )
    scorer_assignment = (
        integer_scorer_assignment
        if integer_scorer_ok
        else f"uniform_W{qdq_bits}A{qdq_bits}"
        if qdq_bits is not None
        else None
    )
    metal_ok, metal_reason = _metal_gate(
        metal_fixedpoint_receipt,
        required_bits=scorer_bits,
        required_scale_mode=scorer_scale_mode,
        required_qdq_fingerprint=qdq_fingerprint,
        required_precision_assignment=scorer_assignment,
        required_integer_fingerprint=(
            str(integer_scorer_receipt.get("fingerprint"))
            if integer_scorer_ok
            and integer_scorer_receipt
            and integer_scorer_receipt.get("fingerprint")
            else None
        ),
    )
    integer_r_ok, integer_r_reason = _integer_r_gate(integer_r_receipt)

    seg_state = (
        AssignmentState.DEFAULT_OFF_CANDIDATE
        if scorer_numerics_ok and metal_ok
        else AssignmentState.HELD_OWED
    )
    r_state = AssignmentState.DEFAULT_OFF_CANDIDATE if integer_r_ok else AssignmentState.HELD_OWED
    rows = (
        Assignment(
            Operation.WITNESS_FORWARD_BACKWARD,
            Substrate.MLX_METAL,
            Precision.MLX_FP32,
            AssignmentState.ACTIVE,
            AuthorityGrade.TRAINING_SIGNAL,
            "fast differentiable witness and frozen-teacher training path",
            "standing MLX path; MPS remains non-authority",
        ),
        Assignment(
            Operation.R_FORWARD,
            Substrate.MLX_METAL,
            Precision.MLX_FP32,
            AssignmentState.ACTIVE,
            AuthorityGrade.TRAINING_SIGNAL,
            "realized-through-R training signal",
            "existing render_through_R_mlx path",
            "NumPy-fp32 byte-close remains terminal reference",
        ),
        Assignment(
            Operation.R_ADJOINT,
            Substrate.CUSTOM_METAL,
            Precision.INTEGER_Q_INT32,
            r_state,
            AuthorityGrade.TRAINING_SIGNAL if integer_r_ok else AuthorityGrade.NO_AUTHORITY,
            "order-independent exact integer accumulation; reproducibility lever, not verdict throughput",
            integer_r_reason,
            None if integer_r_ok else "run tools/run_integer_r_adjoint_backend_host.command",
        ),
        Assignment(
            Operation.SEGNET_VERDICT,
            Substrate.CUSTOM_METAL,
            Precision.FIXEDPOINT_MIXED,
            seg_state,
            (
                AuthorityGrade.LOCAL_CANDIDATE_FILTER
                if scorer_numerics_ok and metal_ok
                else AuthorityGrade.NO_AUTHORITY
            ),
            "candidate replacement for the slow one-thread local SegNet verdict",
            f"QDQ: {qdq_reason}; exact-int64: {integer_scorer_reason}; Metal: {metal_reason}",
            (
                (
                    "shadow/certify the admitted kernel on actual evolving witness frames "
                    "before suppressing the CPU reference"
                )
                if scorer_numerics_ok and metal_ok
                else "complete a full-n600 exact-argmax exact-int64 scorer precursor"
                if not scorer_numerics_ok
                else "complete the bound custom-Metal n600/cross-process/speed receipt"
            ),
            selected_bits=scorer_bits,
            activation_scale_mode=scorer_scale_mode,
        ),
        Assignment(
            Operation.SEGNET_VERDICT,
            Substrate.TORCH_CPU_ONE_THREAD,
            Precision.TORCH_FP32,
            AssignmentState.ACTIVE,
            AuthorityGrade.LOCAL_REFERENCE,
            "current deterministic local verdict backstop",
            "frozen upstream scorer; exact thread geometry",
        ),
        Assignment(
            Operation.SEGNET_VERDICT,
            Substrate.COREML_ANE,
            Precision.COREML_W8A8,
            AssignmentState.FORBIDDEN,
            AuthorityGrade.NO_AUTHORITY,
            "settled calibrated CoreML W8A8 formulation has catastrophic label drift",
            "#482 held-out 1,081,426/2,359,296 flips = 45.836809%; FORMULATION negative",
            "a distinct ANE precision/runtime formulation, not a rerun of W8A8 PTQ",
        ),
        Assignment(
            Operation.SEGNET_VERDICT,
            Substrate.COREML_ANE,
            Precision.COREML_FP32,
            AssignmentState.ADVISORY_ONLY,
            AuthorityGrade.LOCAL_ADVISORY,
            "banked local forward-only advisory rail; placement and contest equivalence are unproved",
            "#482 CoreML fp32 CPU_AND_GPU held-out flip gate passed at 3.609x matched local speed",
        ),
        Assignment(
            Operation.SEGNET_VERDICT,
            Substrate.TORCH_MPS,
            Precision.TORCH_FP32,
            AssignmentState.FORBIDDEN,
            AuthorityGrade.NO_AUTHORITY,
            "MPS numeric drift is a distinct mechanism and never carries score authority",
            "standing deterministic-reproducibility contract",
        ),
        Assignment(
            Operation.POSENET_VERDICT_PRE_FINISH,
            Substrate.BANKED_TELEMETRY,
            Precision.BANKED_FP64_SCALAR,
            AssignmentState.ACTIVE if pose_gate_enabled else AssignmentState.DEFAULT_OFF_CANDIDATE,
            AuthorityGrade.LOCAL_ADVISORY,
            "pose weight is zero pre-finish; skip live PoseNet except periodic drift canaries",
            f"banked_r1_dpose={banked_r1_dpose:.6f}; live canary every {pose_canary_every}",
            "MAIN governed n96 dry-start before live use",
        ),
        Assignment(
            Operation.POSENET_VERDICT_ACTIVE,
            Substrate.TORCH_CPU_ONE_THREAD,
            Precision.TORCH_FP32,
            AssignmentState.ACTIVE,
            AuthorityGrade.LOCAL_REFERENCE,
            "continuous first-six PoseNet output needs its own nonlinear debt certificate",
            "no ANE/custom-Metal pose authority receipt exists",
        ),
        Assignment(
            Operation.ARCHIVE_DECODE,
            Substrate.NUMPY_CPU,
            Precision.NUMPY_FP32,
            AssignmentState.ACTIVE,
            AuthorityGrade.LOCAL_REFERENCE,
            "bit-identical portable receiver reference",
            "deterministic NumPy-fp32 contract",
        ),
        Assignment(
            Operation.TERMINAL_EVALUATOR,
            Substrate.TORCH_CPU_ONE_THREAD,
            Precision.TORCH_FP32,
            AssignmentState.ACTIVE,
            AuthorityGrade.CONTEST_AUTHORITY,
            "contest-CPU terminal replay on exact archive bytes",
            "upstream/evaluate.py exact-byte custody required",
        ),
        Assignment(
            Operation.TERMINAL_EVALUATOR,
            Substrate.CONTEST_CUDA,
            Precision.TORCH_FP32,
            AssignmentState.OPERATOR_GO_ONLY,
            AuthorityGrade.CONTEST_AUTHORITY,
            "separate contest-CUDA terminal axis",
            "paid/off-device dispatch remains contained",
            "claim lane and obtain operator GO",
        ),
    )
    return ThroughputAuthorityPolicy(
        assignments=rows,
        pose_gate_enabled=bool(pose_gate_enabled),
        pose_canary_every=int(pose_canary_every),
        banked_r1_dpose=float(banked_r1_dpose),
    )


__all__ = [
    "Assignment",
    "AssignmentState",
    "AuthorityGrade",
    "Operation",
    "Precision",
    "Substrate",
    "ThroughputAuthorityPolicy",
    "compile_throughput_authority_policy",
]
