#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Register the 2nd EMPIRICAL anchor (3rd lifecycle event) on canonical
equation #2 ``hinton_kl_distill_enables_qat_catalyst_composition_savings_v1``
from the Cascade B CATALYST sister wave 2 production-scale 600f x 1000ep +
post-train QAT 100ep run.

This anchor is the canonical RESOLUTION of the 5th-order synthetic-fixture
IMPLEMENTATION_LEVEL_FALSIFIED anchor (commit `fcfad9331`) which the 5th-order
landing memo flagged as missing the CATALYST training-time mechanism
(reason #2: "the QAT path to BE TRAINED so the head adapts to the FP4 codebook
discretization"). Wave 2 TRAINS the QAT path (Stage C 100ep fine-tune).

Per CLAUDE.md "MLX portable-local-substrate authority" + Catalog #192/#317:
  axis_tag `[macOS-MLX research-signal]`; NEVER promotable.
Per Catalog #323 canonical Provenance: build_provenance_for_predicted with
  the macOS-MLX research-signal axis (matching the sister synthetic anchor
  exactly; MLX-local advisory grade is non-promotable by construction).
Per Catalog #344 canonical equations registry: 3rd lifecycle event; the
  registered next_recalibration_trigger `when_3+_new_empirical_anchors_in_domain`
  is SATISFIED at 3 in-domain anchors (operator-routable recalibration via
  tools/recalibrate_equation.py; NOT auto-fired on append).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.canonical_equations.equation import EmpiricalAnchor  # noqa: E402
from tac.canonical_equations.registry import (  # noqa: E402
    update_equation_with_empirical_anchor,
)
from tac.provenance.builders import build_provenance_for_predicted  # noqa: E402

EQUATION_ID = "hinton_kl_distill_enables_qat_catalyst_composition_savings_v1"
# Canonical equation #2 latex predicted-lift form (per 5th-order anchor):
#   ΔS_cat = ΔS_{P4}^{alone} · (1 + α · ΔH_{logits}^{T=2}),  α ∈ [0.1, 0.2]
# The synthetic anchor used qat_savings_lift = 0.15 (midpoint of α band).
PREDICTED_QAT_SAVINGS_LIFT = 0.15
PREDICTED_ENTROPY_TIGHTENING_RATIO = 0.85


def main(argv: list[str]) -> int:
    if len(argv) < 1:
        sweep_path = (
            REPO_ROOT
            / "experiments"
            / "results"
            / "cascade_b_catalyst_sister_wave_2_production_scale_post_train_qat_real_segnet_20260526"
            / "sweep_results.json"
        )
    else:
        sweep_path = Path(argv[0]).resolve()

    summary = json.loads(sweep_path.read_text())
    arms = summary["arms"]
    deltas = summary["deltas"]
    verdict = summary["catalog_307_verdict"]

    path_a_composite = arms["path_a_alone"]["composite_proxy"]
    catalyst_composite = arms["catalyst_composition"]["composite_proxy"]
    path_a_eval_kl = arms["path_a_alone"]["eval_kl"]
    catalyst_eval_kl = arms["catalyst_composition"]["eval_kl"]

    # Canonical residual per the 5th-order anchor formulation:
    #   residual = |composite_catalyst − composite_path_a · (1 − qat_savings_lift)|
    # i.e. how far the empirical CATALYST composite is from the predicted
    # 15%-lift improvement over Path A alone.
    predicted_catalyst_composite = path_a_composite * (1.0 - PREDICTED_QAT_SAVINGS_LIFT)
    residual = abs(catalyst_composite - predicted_catalyst_composite)

    inputs_blob = json.dumps(summary["configuration"], sort_keys=True).encode("utf-8")
    inputs_sha = hashlib.sha256(inputs_blob).hexdigest()

    # Provenance: matches the sister synthetic anchor (build_provenance_for_predicted
    # with the macOS-MLX research-signal axis; non-promotable per Catalog #192/#317).
    prov = build_provenance_for_predicted(
        model_id="cascade_b_catalyst_wave2_production_real_segnet_kl_t2_plus_post_train_qat_fp4",
        inputs_sha256=inputs_sha,
        measurement_axis="[macOS-MLX research-signal]",
        hardware_substrate="macos_arm64",
        captured_at_utc=summary.get("captured_at_utc"),
    )

    anchor = EmpiricalAnchor(
        anchor_id=(
            "hinton_kl_distill_qat_catalyst_6th_order_production_600f_1000ep_"
            "post_train_qat_100ep_real_segnet_mlx_local_20260527"
        ),
        measurement_utc="2026-05-27T13:30:00Z",
        inputs={
            "in_domain_context": (
                "hinton_kl_t2_mlx_600f_1000ep_real_segnet_teacher_student_head_mode_learnable_"
                "plus_post_train_qat_fp4_100ep_fine_tune_cascade_b_catalyst_6th_order_2026_05_27"
            ),
            "catalyst_position": ["p2_hinton_kl_distill", "p5_qat_fp4_fine_tune"],
            "n_frames": summary["configuration"]["n_frames"],
            "path_a_n_epochs": summary["configuration"]["path_a_n_epochs"],
            "qat_n_epochs": summary["configuration"]["qat_n_epochs"],
            "distill_temperature": summary["configuration"]["temperature"],
            "predicted_qat_savings_lift_alpha_midpoint": PREDICTED_QAT_SAVINGS_LIFT,
            "predicted_entropy_tightening_ratio": PREDICTED_ENTROPY_TIGHTENING_RATIO,
        },
        predicted_output={
            "qat_savings_lift": PREDICTED_QAT_SAVINGS_LIFT,
            "post_quantization_scorer_entropy_tightening_ratio": PREDICTED_ENTROPY_TIGHTENING_RATIO,
            "predicted_catalyst_composite": predicted_catalyst_composite,
        },
        empirical_output={
            "path_a_eval_kl": path_a_eval_kl,
            "catalyst_eval_kl": catalyst_eval_kl,
            "path_a_composite": path_a_composite,
            "catalyst_composite": catalyst_composite,
            "delta_catalyst_kl_minus_path_a_kl": deltas["delta_catalyst_kl_minus_path_a_kl"],
            "delta_catalyst_composite_minus_path_a_composite": deltas[
                "delta_catalyst_composite_minus_path_a_composite"
            ],
            "delta_catalyst_rate_minus_path_a_rate": deltas["delta_catalyst_rate_minus_path_a_rate"],
            "verdict_per_catalog_307": verdict,
            "catalyst_path_was_trained": True,
            "resolves_synthetic_falsification_reason_2_missing_qat_training_mechanism": True,
        },
        residual=float(residual),
        source_artifact=str(sweep_path.relative_to(REPO_ROOT)),
        measurement_method=(
            "mlx_local_3_arm_catalyst_production_600f_1000ep_real_segnet_teacher_"
            "p2_hinton_kl_t2_plus_p5_post_train_qat_fp4_fake_quant_ste_100ep_fine_tune"
        ),
        provenance=prov,
    )

    updated = update_equation_with_empirical_anchor(
        EQUATION_ID,
        anchor,
        agent="claude",
        subagent_id="cascade_b_wave2_RESUME1",
        notes=(
            "Cascade B CATALYST sister wave 2 6th-order production-scale "
            "real-SegNet post-train QAT empirical anchor (3rd lifecycle event; "
            "2nd EMPIRICAL anchor). RESOLVES the 5th-order synthetic "
            "IMPLEMENTATION_LEVEL_FALSIFIED missing-QAT-training-mechanism. "
            f"verdict={verdict}; delta_kl="
            f"{deltas['delta_catalyst_kl_minus_path_a_kl']:+.4f}; "
            f"delta_rate={deltas['delta_catalyst_rate_minus_path_a_rate']:+.6f}. "
            "[macOS-MLX research-signal] non-promotable per Catalog #192/#317."
        ),
    )
    n_anchors = len(updated.empirical_anchors)
    print(f"[eq2-anchor] registered 3rd lifecycle event; n_anchors now {n_anchors}", flush=True)
    print(f"[eq2-anchor] verdict={verdict} residual={residual:.6f}", flush=True)
    trig = updated.next_recalibration_trigger
    print(f"[eq2-anchor] next_recalibration_trigger={trig}", flush=True)
    print(
        f"[eq2-anchor] recalibration condition (3+ empirical in-domain anchors) "
        f"{'SATISFIED' if n_anchors >= 3 else 'NOT YET'}; "
        "operator-routable via tools/recalibrate_equation.py (NOT auto-fired)",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
