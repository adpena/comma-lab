# SPDX-License-Identifier: MIT
"""Typed default-OFF stubs for the int8 teacher and witness-QAT rungs.

These proposals emit no argv because the live trainer has no parser-backed
int8 lever.  Enabling an unwired proposal is structurally refused.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum


class ProposalState(StrEnum):
    OFF_UNWIRED = "off_unwired"
    READY_FOR_AB_WIRING = "ready_for_ab_wiring"


class TeacherBackend(StrEnum):
    MLX_NATIVE_INT8_CONV = "mlx_native_int8_conv"
    COREML_ANE_W8A8 = "coreml_ane_w8a8"
    MLX_W8A8_QDQ_STE = "mlx_w8a8_qdq_ste"


@dataclass(frozen=True)
class Int8TeacherForwardProposal:
    enabled: bool = False
    state: ProposalState = ProposalState.OFF_UNWIRED
    backend: TeacherBackend = TeacherBackend.MLX_W8A8_QDQ_STE
    weights: str = "int8_symmetric_per_operator_tensor"
    activations: str = "int8_symmetric_dynamic_per_operator_input"
    accumulation: str = "float32"
    gradient: str = "identity_ste_through_qdq"
    required_quality_pairs: int = 600
    minimum_global_gradient_cosine: float = 0.99
    minimum_pair_gradient_cosine: float = 0.99
    minimum_speedup: float = 1.5
    measured_anchor: str | None = None

    def __post_init__(self) -> None:
        if self.enabled:
            raise ValueError("int8 teacher proposal is unwired and must remain default-OFF")
        if self.state is not ProposalState.OFF_UNWIRED:
            raise ValueError("disabled int8 teacher proposal must be OFF_UNWIRED")
        if self.required_quality_pairs != 600:
            raise ValueError("teacher admission requires exactly n600 real states")
        if not (0.0 <= self.minimum_global_gradient_cosine <= 1.0):
            raise ValueError("minimum global gradient cosine must be in [0,1]")
        if not (0.0 <= self.minimum_pair_gradient_cosine <= 1.0):
            raise ValueError("minimum pair gradient cosine must be in [0,1]")
        if self.minimum_speedup <= 1.0:
            raise ValueError("minimum speedup must be >1")

    def to_display_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["state"] = self.state.value
        payload["backend"] = self.backend.value
        payload["wired"] = False
        payload["live_trainer_argv"] = []
        return payload


@dataclass(frozen=True)
class Int8WitnessQATProposal:
    enabled: bool = False
    state: ProposalState = ProposalState.OFF_UNWIRED
    stage: str = "finishing_stage_only"
    grid: str = "lvls1_per_tensor_symmetric_absmax_over_127"
    backward: str = "fake_quant_identity_ste"
    quantized_groups: str = "all_learned_params_except_rule118_free_B"
    required_quality_pairs: int = 600
    admission_surface: str = "parsed_lvls1_receiver_realized_dseg_and_exact_archive_bytes"
    control: str = "fp32_training_then_posthoc_lvls1_int8"
    measured_posthoc_gap_anchor: str | None = None

    def __post_init__(self) -> None:
        if self.enabled:
            raise ValueError("int8 witness QAT proposal is unwired and must remain default-OFF")
        if self.state is not ProposalState.OFF_UNWIRED:
            raise ValueError("disabled int8 witness QAT proposal must be OFF_UNWIRED")
        if self.required_quality_pairs != 600:
            raise ValueError("QAT admission requires exactly n600 real states")
        if self.stage != "finishing_stage_only":
            raise ValueError("QAT changes may only begin at a declared stage boundary")

    def to_display_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["state"] = self.state.value
        payload["wired"] = False
        payload["live_trainer_argv"] = []
        return payload


__all__ = [
    "Int8TeacherForwardProposal",
    "Int8WitnessQATProposal",
    "ProposalState",
    "TeacherBackend",
]
