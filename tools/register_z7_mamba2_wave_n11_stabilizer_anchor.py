# SPDX-License-Identifier: MIT
"""Register Wave N+11 Z7-Mamba-2 stabilizer empirical anchor.

Per Catalog #344 canonical equations registry + Catalog #371 auto-recalibrator
+ Catalog #323 canonical Provenance umbrella. Appends to the existing
``z7_mamba2_state_space_predictive_coding_pose_axis_savings_v1`` equation
(5 anchors → 6 anchors; well-calibrated; the auto-recalibrator fires when
3+ NEW anchors land in domain).

[verified-against: .omx/research/z7_mamba2_wave_n11_stabilizer_600pair_50ep_20260530T233603Z/training_artifact.json]
[verified-against: tac.canonical_equations.update_equation_with_empirical_anchor canonical helper]
[verified-against: tac.provenance.builders.build_provenance_for_mps_proxy canonical helper]

Run::

    .venv/bin/python tools/register_z7_mamba2_wave_n11_stabilizer_anchor.py
"""
from __future__ import annotations

from pathlib import Path

from tac.canonical_equations import (
    EmpiricalAnchor,
    update_equation_with_empirical_anchor,
)
from tac.provenance.builders import build_provenance_for_mps_proxy

ANCHOR_RUN_DIR = Path(
    ".omx/research/z7_mamba2_wave_n11_stabilizer_600pair_50ep_20260530T233603Z"
)


def main() -> int:
    artifact_sha = "a98da26facfac1e7eb4137a54db75c438213094dff51a0b127777524ca7aabea"
    archive_sha = "c5d81af0eca1f1e762115712d7d30c93a447097aa96c9b642e414244993bbb31"
    artifact_path = ANCHOR_RUN_DIR / "training_artifact.json"

    # Wave N+11 stabilizer measurement: SUCCESS — completed 50/50 epochs
    # without NaN. This is the canonical empirical-anchor entry for the
    # z7_mamba2_state_space_predictive_coding_pose_axis_savings_v1 canonical
    # equation per Catalog #344.
    #
    # Empirical numbers from training_artifact.json + telemetry.jsonl:
    #   - epochs: 50/50 (no early-stop / no NaN)
    #   - wall: 210.4s
    #   - first loss: 0.3772
    #   - last loss:  0.3711
    #   - first ema_drift_l2: 4.21e-6
    #   - last ema_drift_l2:  0.130
    #   - archive bytes: 1,316,597
    #   - mock-scorer-teacher (so pose=0 throughout; the stabilizer is
    #     validated at the dynamics surface, not at the score surface)
    #
    # Compare to Wave N+10 unstabilized 600pair × 50ep:
    #   - ep15: ema_drift_l2 = 0.895 (vs ep15 stabilized = 0.0014)
    #   - ep18: ema_drift_l2 = 6.86  (vs ep18 stabilized = 0.0061)
    #   - ep19: ema_drift_l2 = 9.92 → NaN crash by ep ~20
    # The stabilizer reduced ep19 ema_drift_l2 by ~1320x and prevented the
    # NaN failure mode identified in the Slot 1 RESUME IMPLEMENTATION-LEVEL
    # falsification (per Catalog #307 paradigm-vs-implementation classification).
    #
    # NOTE: this anchor uses --allow-mock-scorer-teacher so it does NOT speak
    # to pose-axis reduction (the equation's primary quantity). The stabilizer
    # bind step is the PRECONDITION for a future real-teacher anchor; the
    # numerical anchor is for the STABILITY HORIZON measurement.

    predicted_output = {
        # Predicted: 50 epochs completes WITHOUT NaN if stabilizer recipe
        # (grad-clip max_norm=1.0 + warmup 5 epochs + weight_decay 1e-4 +
        # smaller d_state=8 + smaller d_inner=32 + AdamW) is applied per
        # Wave N+11 plan. Per Gu+Dao 2023 Mamba canonical stability +
        # Loshchilov+Hutter 2019 AdamW canonical.
        "wave_n11_stabilizer_completes_50ep_without_nan": True,
        "predicted_stability_horizon_epochs": 50,
        "predicted_ema_drift_l2_at_ep_50_upper_bound": 0.5,
        "predicted_loss_monotonic_or_flat_within_pct": 10.0,
        "predicted_classification": (
            "stabilizer_recipe_extincts_nan_at_ep_16_18_signature_via_"
            "canonical_mamba2_stability_recipe_gu_dao_2023"
        ),
    }
    empirical_output = {
        "wave_n11_stabilizer_completes_50ep_without_nan_empirical": True,
        "actual_stability_horizon_epochs": 50,
        "actual_ema_drift_l2_at_ep_50": 0.13014995008149802,
        "actual_loss_pct_change_ep0_to_ep49": -1.62,  # 0.3772 -> 0.3711
        "actual_first_loss": 0.3771716058254242,
        "actual_last_loss": 0.37109893560409546,
        "actual_min_loss": 0.36085450649261475,
        "actual_max_loss": 0.3832758963108063,
        "actual_first_ema_drift_l2": 4.207030875844035e-06,
        "actual_last_ema_drift_l2": 0.13014995008149802,
        "actual_wall_clock_seconds": 210.37502098083496,
        "actual_archive_bytes": 1316597,
        "actual_archive_sha256": archive_sha,
        "wave_n10_unstabilized_ep19_ema_drift_l2_comparison": 9.92,
        "wave_n11_stabilized_ep19_ema_drift_l2": 0.0075,
        "stabilizer_ema_drift_reduction_at_ep19_x": 1322.7,
        "nan_first_epoch": None,
        "scorer_teacher_kind": "mock",
        "stabilizer_kwargs": {
            "grad_clip_max_norm": 1.0,
            "warmup_epochs": 5,
            "weight_decay": 1e-4,
            "ema_decay": 0.997,
            "optimizer_kind": "adamw",
            "d_state": 8,
            "d_model": 32,
            "expand": 1,
            "d_inner_actual": 32,
        },
        "evidence_grade": "[macOS-MLX research-signal]",
        "promotion_eligible": False,
        "score_claim_valid": False,
    }

    # residual: predicted exactly matched empirical (50/50 no NaN; horizon met).
    # The stability surface is binary {completes | NaN-crash}; predicted=True
    # and empirical=True; residual = 0.0.
    residual = 0.0

    anchor = EmpiricalAnchor(
        anchor_id="wave_n11_stabilizer_600pair_50ep_mlx_local_anchor_20260530T233603Z",
        measurement_utc="2026-05-30T23:39:52Z",
        inputs={
            "method": "wave_n11_stabilizer_recipe_grad_clip_warmup_weight_decay_smaller_dims",
            "num_pairs": 600,
            "epochs_planned": 50,
            "epochs_completed": 50,
            "learning_rate": 3e-4,
            "in_domain_context": "z7_mamba2_state_space_recurrence_mlx_local_stabilizer_smoke",
            "wave_n10_predecessor_nan_signature": "ema_drift_l2_explodes_0.118_to_9.92_at_ep13_to_ep19_then_nan_at_ep_~20",
            "wave_n11_plan_task": "1481",
            "lane_id": "lane_z7_mamba2_wave_n11_stabilizer_re_fire_20260530",
            "operator_directive": (
                "Wave N+11 plan: apply training stabilizer + re-fire to land "
                "canonical equation anchor; unblocks Wave N+11 quad composition "
                "test (Z6-v2 + Z7-Mamba-2 + NSCS06 v8 + Compound C) sub-0.15 cascade."
            ),
        },
        predicted_output=predicted_output,
        empirical_output=empirical_output,
        residual=residual,
        source_artifact=str(artifact_path),
        measurement_method="mlx_local_canonical_harness_wave_n11_stabilizer_600pair_50ep_completes_without_nan_anchor",
        provenance=build_provenance_for_mps_proxy(
            artifact_sha256=artifact_sha,
            source_path=str(artifact_path),
            captured_at_utc="2026-05-30T23:39:52Z",
        ),
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
    )

    eq = update_equation_with_empirical_anchor(
        equation_id="z7_mamba2_state_space_predictive_coding_pose_axis_savings_v1",
        anchor=anchor,
        agent="claude",
        subagent_id="z7_mamba2_wave_n11_stabilizer_20260530",
        notes=(
            "Wave N+11 Z7-Mamba-2 stabilizer recipe 600pair x 50ep MLX-LOCAL "
            "re-fire anchor per task #1481. Empirically proves the canonical "
            "stabilizer recipe (grad-clip max_norm=1.0 + warmup 5ep + "
            "weight_decay 1e-4 + smaller d_state=8 d_model=32 expand=1 + "
            "AdamW + EMA 0.997) extincts the Slot 1 RESUME NaN-at-ep-~20 "
            "IMPLEMENTATION-LEVEL falsification per Catalog #307 "
            "paradigm-vs-implementation classification. ema_drift_l2 "
            "reduced 1322x at ep19 vs Wave N+10 unstabilized. Anchor uses "
            "mock-scorer-teacher; this validates the STABILITY HORIZON "
            "surface, not the pose-axis reduction (next wave: real-teacher "
            "anchor with same stabilizer)."
        ),
    )

    print(
        f"REGISTERED Wave N+11 stabilizer anchor: equation has {len(eq.empirical_anchors)} anchors"
    )
    print(f"  trigger: {eq.next_recalibration_trigger}")
    print(f"  last_calibration_utc: {eq.last_calibration_utc}")
    print(f"  is_well_calibrated: {eq.is_well_calibrated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
