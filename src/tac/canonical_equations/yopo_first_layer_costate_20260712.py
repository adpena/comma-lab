# SPDX-License-Identifier: MIT
"""YOPO first-layer frozen-SegNet provider law (research-only).

Zhang, Zhang, Lu, Zhu, and Dong, *You Only Propagate Once: Accelerating
Adversarial Training via Maximal Principle* (2019), arXiv:1905.00877, gives
the clean-room control-law inspiration.  This module makes no claim that its
adversarial-training results transfer to renderer-parameter optimization.
"""

from __future__ import annotations

from tac.canonical_equations.equation import RECALIBRATE_NEVER_AUTO, CanonicalEquation
from tac.provenance.builders import build_provenance_for_research_sidecar

YOPO_FIRST_LAYER_COSTATE_EQUATION_ID = "yopo_first_layer_costate_v1"
_MEMO = ".omx/research/goldmine_hunt_20260712.md"


def build_yopo_first_layer_costate_v1() -> CanonicalEquation:
    """Build the isolated provider law; fresh teacher checks remain mandatory."""

    return CanonicalEquation(
        equation_id=YOPO_FIRST_LAYER_COSTATE_EQUATION_ID,
        name="YOPO first-layer current-frame input-costate provider",
        one_line_summary=(
            "Bank p1=dL/dz1 at a full-teacher refresh, then use "
            "lambda_hat_t=J_prefix(x_t)^T p1 for the current rendered frame."
        ),
        latex_form=(
            r"z_1=f_0(x),\ p_1^r=\nabla_{z_1}L_{teacher}(x_r),\ "
            r"\hat\lambda_t=J_{f_0}(x_t)^\top p_1^r,\ "
            r"\nabla_\theta L_{inj}=J_x(\theta)^\top\hat\lambda_t"
        ),
        python_callable_module_path=("tac.boundary_math.segnet_gradient_replacement:yopo_first_layer_costate_torch"),
        domain_of_validity={
            "citation": (
                "Zhang, Zhang, Lu, Zhu, Dong (2019), You Only Propagate Once: "
                "Accelerating Adversarial Training via Maximal Principle, arXiv:1905.00877"
            ),
            "cut": "encoder.model.conv_stem -> bn1 -> blocks[0] output only",
            "included": (
                "finite current frame and finite detached p1",
                "content-addressed NPZ bank re-hashed at every provider evaluation",
                "matching objective/scorer/anchor/split/source-step identities",
                "valid global and optional-annulus metrics plus a finite, decreasing "
                "one-step teacher check with measured regret",
                "regime replay/Pareto admission rather than universal agreement or regret thresholds",
            ),
            "excluded": (
                "a claim that a frozen p1 is exact away from its refresh frame",
                "a universal cosine threshold",
                "trainer wiring, score claims, or throughput claims without measurement",
            ),
            "fallback": "any custody, topology, frame, objective, step, or finite failure selects full_teacher",
            "authority": "research provider contract; score_claim=false",
        },
        units_in={
            "x": "rendered_frame_units",
            "p1": "teacher_loss_per_split_activation_unit",
            "objective_context_fingerprint": "sha256",
            "split_identity_sha256": "sha256",
            "bank_sha256": "sha256",
        },
        units_out={"lambda_hat": "teacher_loss_per_rendered_frame_unit"},
        empirical_anchors=(),
        predicted_vs_empirical_residual={},
        last_calibration_utc="2026-07-12T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_NEVER_AUTO,
        canonical_consumers=("tac.witness_dsl.scorer_gradient_policy",),
        canonical_producers=("tac.boundary_math.segnet_gradient_replacement",),
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO,
            reactivation_criteria=(
                "fresh non-refresh global and annulus agreement, real-teacher descent/regret, "
                "wall-time, and governed full-P=600 evidence"
            ),
            measurement_axis="[derived YOPO provider law; local research only]",
            hardware_substrate="torch/MLX portable contract",
        ),
    )


def populate_yopo_first_layer_costate_v1(
    *, path=None, lock_path=None, agent=None, subagent_id=None
) -> CanonicalEquation:
    """Idempotently append the isolated YOPO law through the canonical writer."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_yopo_first_layer_costate_v1()
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes="yopo_first_layer_costate_contract_20260712",
    )
    return equation


__all__ = [
    "YOPO_FIRST_LAYER_COSTATE_EQUATION_ID",
    "build_yopo_first_layer_costate_v1",
    "populate_yopo_first_layer_costate_v1",
]
