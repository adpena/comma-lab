# SPDX-License-Identifier: MIT
"""RR9 Metal mem-probe fire-protocol law."""
from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_SOURCE_INSPECTION,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "ddm_rr9_mem_probe_fire_protocol_v1"
SOURCE_ARTIFACT = ".omx/research/ddm_rr9_20260807/ROUND9_FINDINGS.md"


def metal_fire_protocol_status(
    *,
    mem_probe_receipt_required: bool,
    mem_probe_receipt_exists: bool,
    mem_probe_status: str | None,
    has_load_stage_samples: bool,
    launch_kind: str,
) -> dict[str, bool | str]:
    """Classify whether a Metal/MLX fire may proceed under the RR9 protocol."""

    is_metal_train = launch_kind in {"mlx-train", "metal-train", "gpu-train"}
    if not is_metal_train:
        return {"allowed": True, "reason": "not_metal_training_fire"}
    if not mem_probe_receipt_required:
        return {"allowed": True, "reason": "receipt_not_required"}
    if not mem_probe_receipt_exists:
        return {"allowed": False, "reason": "missing_required_mem_probe_receipt"}
    if mem_probe_status != "passed":
        return {"allowed": False, "reason": "mem_probe_not_passed"}
    if not has_load_stage_samples:
        return {"allowed": False, "reason": "missing_load_stage_samples"}
    return {"allowed": True, "reason": "passed_required_mem_probe_receipt"}


def build_ddm_rr9_mem_probe_fire_protocol_v1() -> CanonicalEquation:
    provenance = build_provenance_for_research_sidecar(
        SOURCE_ARTIFACT,
        reactivation_criteria=(
            "append anchors for every governed Metal training fire with a required "
            "mem-probe receipt; refuse future fires before safe_run when the receipt is absent"
        ),
        measurement_axis="[apparatus / scorer-free]",
        hardware_substrate="source_and_state_inspection",
        captured_at_utc="2026-08-07T11:55:23Z",
    )
    predicted = metal_fire_protocol_status(
        mem_probe_receipt_required=True,
        mem_probe_receipt_exists=False,
        mem_probe_status=None,
        has_load_stage_samples=False,
        launch_kind="mlx-train",
    )
    anchor = EmpiricalAnchor(
        anchor_id="rr9_mx1c_required_mem_probe_absent_before_arm_cap_fire_20260807",
        measurement_utc="2026-08-07T11:55:23Z",
        inputs={
            "ticket": ".omx/research/ddm_mx1c_20260807/launch_ticket_v2_two_arm_governed.json",
            "mem_probe_receipt_required": True,
            "mem_probe_receipt_path": (
                ".omx/research/ddm_mx1c_20260807/row1_v2_two_arm/"
                "mem_probe_receipt.json"
            ),
            "mem_probe_receipt_exists": False,
            "launch_kind": "mlx-train",
            "safe_run_projection_gib": 66.268951,
        },
        predicted_output=predicted,
        empirical_output={
            "training_fire_observed_without_required_receipt": True,
            "safe_run_projection_arithmetic_held": True,
            "projection_is_not_metal_load_stage_receipt": True,
            "follow_on": "rr9_f1_mx1c_mem_probe_fire_protocol_guard",
        },
        residual=0.0,
        source_artifact=SOURCE_ARTIFACT,
        measurement_method=(
            "RR9 source/state inspection of mx1c ticket, live launch manifest, durable daemon "
            "admission row, and absent mem-probe receipt path"
        ),
        provenance=provenance,
        empirical_verification_status=VERIFIED_VIA_SOURCE_INSPECTION,
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Required Metal mem-probe receipt gates training fire",
        one_line_summary=(
            "For Metal/MLX training tickets, safe_run admission is not a substitute for a "
            "required passed mem-probe receipt with load-stage samples."
        ),
        latex_form=(
            r"\mathrm{allow}=\neg R\vee(E\wedge status=passed\wedge samples)"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.ddm_rr9_mem_probe_fire_protocol_20260807:"
            "metal_fire_protocol_status"
        ),
        domain_of_validity={
            "included": [
                "Metal/MLX training fires with mem_probe_receipt_required=true",
                "governed safe_run launches where admission and load-stage receipt are separate",
            ],
            "excluded": [
                "CPU-only scorer-free jobs",
                "post-hoc process liveness claims",
                "using projected peak as a measured Metal load-stage receipt",
            ],
            "verdict_scope": "APPARATUS: pre-fire protocol",
            "score_claim": False,
            "promotion_eligible": False,
        },
        units_in={
            "mem_probe_receipt_required": "bool",
            "mem_probe_receipt_exists": "bool",
            "mem_probe_status": "string_or_none",
            "has_load_stage_samples": "bool",
            "launch_kind": "string",
        },
        units_out={"allowed": "bool", "reason": "typed refusal/allow token"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"rr9_mem_probe_guard_residual": 0.0},
        last_calibration_utc="2026-08-07T11:55:23Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "mx1c launch-protocol guard",
            "tools/launch_detached_process.py training argv classifier",
            "safe_run scheduling wrapper",
        ),
        canonical_producers=(
            ".omx/research/ddm_rr9_20260807/ROUND9_FINDINGS.md",
            ".omx/research/ddm_mx1c_20260807/launch_ticket_v2_two_arm_governed.json",
        ),
        provenance=provenance,
    )


def populate_ddm_rr9_mem_probe_fire_protocol_v1(
    *, path=None, lock_path=None, agent: str | None = None, subagent_id: str | None = None
) -> CanonicalEquation:
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_ddm_rr9_mem_probe_fire_protocol_v1()
    register_canonical_equation(
        eq,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes="ddm_cq1 registration: RR9 mem-probe fire protocol guard",
    )
    return eq


__all__ = [
    "EQUATION_ID",
    "SOURCE_ARTIFACT",
    "build_ddm_rr9_mem_probe_fire_protocol_v1",
    "metal_fire_protocol_status",
    "populate_ddm_rr9_mem_probe_fire_protocol_v1",
]
