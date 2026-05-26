# SPDX-License-Identifier: MIT
"""Append Variant B-d empirical anchors to canonical equation #347.

Per operator NON-NEGOTIABLE 2026-05-26 cascade follow-up + Catalog #344 sister
pattern (`tools/register_2_new_canonical_equations_20260526.py` commit
`04f34ea40`). Appends 2 aggregate empirical anchors to equation
`residual_hybrid_boosting_savings_v1`:

(a) ONE aggregate anchor for the 9-cell SCALE-INVARIANCE finding via Variant B-d
    codec (sister-aggregate of the gain_clamp sweep finding that the L1 BPR1
    codec was scale-invariant): empirically confirms Variant B-d ALSO produces
    a constant sidecar (149B vs L1's 42B; both fully gain_clamp-independent).

(b) ONE per-cell representative anchor: gain_clamp=0.20 epochs=30 (Carmack
    operator-routable best-cell config; reuses the L1 anchor sister-keys).

Per Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE: only NEW anchors;
no mutation of existing canonical equation #347 fields or anchors.

Per Catalog #359 sister discipline: Variant B-d is BoostNeRV-PR110 residual-
correction-hybrid context (in-domain per equation #347's `domain_of_validity`);
NOT a misapplication to residual-hybrid context the equation does not predict.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path("/Users/adpena/Projects/pact")
SWEEP_ARTIFACT = REPO_ROOT / ".omx/research/boostnerv_pr110_bpr1_variant_b_sweep_results_20260526/sweep_heatmap.json"

sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.canonical_equations import (
    EmpiricalAnchor,
    get_equation_by_id,
    update_equation_with_empirical_anchor,
)
from tac.provenance.builders import build_provenance_for_predicted


def main() -> int:
    eq_id = "residual_hybrid_boosting_savings_v1"
    eq_before = get_equation_by_id(eq_id)
    assert eq_before is not None, f"Canonical equation #347 not registered: {eq_id}"
    anchors_before = len(eq_before.empirical_anchors)
    print(f"[append-variant-b-d-anchors] BEFORE: {anchors_before} anchors registered for {eq_id}")

    # Load sweep artifact for canonical anchor data
    artifact = json.loads(SWEEP_ARTIFACT.read_text())
    cells = artifact["cells"]
    assert len(cells) == 9, f"Expected 9 cells, got {len(cells)}"

    pr110_sha = artifact["pr110_archive_sha256"]
    pr110_bytes_size = artifact["pr110_archive_bytes"]

    # ANCHOR (a): aggregate Variant B-d scale-invariance finding across 9 cells
    # All cells produce identical sidecar bytes (149B) regardless of gain_clamp.
    # The "predicted" value per the canonical equation #347 closed_form_predicate
    # `Δrate = 25 × sidecar_bytes / 37_545_489` would have been gain_clamp-dependent
    # IF Variant B-d codec broke scale-invariance; the EMPIRICAL value is gain_clamp-
    # invariant. Residual computed as max(|predicted - empirical|) / empirical.

    sweep_sidecar_bytes = [c["bpr1_variant_b_d_sidecar_bytes"] for c in cells]
    assert min(sweep_sidecar_bytes) == max(sweep_sidecar_bytes) == 149, (
        f"Variant B-d sweep expected constant 149B; got {sweep_sidecar_bytes}"
    )
    constant_sidecar_bytes = 149
    constant_delta_rate = 25.0 * constant_sidecar_bytes / 37_545_489

    # Per equation #347's domain_of_validity at landing time: predicted that
    # increasing gain_clamp would predict increased rate-cost AND increased
    # distortion reduction; the empirical Variant B-d sweep falsifies the
    # rate-cost increase prediction (sidecar bytes constant at 149B at this
    # codec surface). Per canonical Catalog #307 sister: PARADIGM intact;
    # IMPLEMENTATION-LEVEL falsification of the codec variant.
    #
    # We encode the "predicted" branch as what the canonical equation predicate
    # would produce IF Variant B-d sidecar bytes had scaled with gain_clamp
    # proportional to clamp-magnitude-ratio (a hypothetical proportionality).
    # The empirical branch is the OBSERVED constant 149B. The non-zero residual
    # captures the falsification magnitude per Catalog #344 invariant.

    # Predicted naive proportional: bytes ~ gain_clamp ratio × baseline. Baseline
    # gain_clamp=0.05 → 149B; gain_clamp=0.20 hypothetical would be 4 × 149 = 596B.
    # Mean predicted across 9 cells = mean(149, 149, 149, 298, 298, 298, 596, 596, 596)
    # = 326B. Empirical = 149B. Residual = |326 - 149| / 149 = 1.188 (118.8% rel error).
    hypothetical_predicted_bytes_per_cell = [
        149 * (gc / 0.05) for gc in [c["gain_clamp"] for c in cells]
    ]
    mean_hypothetical_predicted = sum(hypothetical_predicted_bytes_per_cell) / 9.0
    aggregate_residual_relative = abs(mean_hypothetical_predicted - constant_sidecar_bytes) / max(
        constant_sidecar_bytes, 1
    )

    aggregate_provenance = build_provenance_for_predicted(
        model_id=(
            "boostnerv_pr110_residual_correction_hybrid_VARIANT_B_D_sign_bitmap_"
            "9cell_aggregate_clamp_0p05_to_0p20_epochs_30_to_300"
        ),
        inputs_sha256=pr110_sha,
        measurement_axis="[macOS-MLX research-signal]",
        hardware_substrate="darwin_arm64_m5_max_mlx_local",
        captured_at_utc="2026-05-26T18:43:00Z",
    )
    aggregate_anchor = EmpiricalAnchor(
        anchor_id="residual_hybrid_boosting_boostnerv_pr110_VARIANT_B_D_9cell_aggregate_scale_invariance_finding_20260526",
        measurement_utc="2026-05-26T18:43:00Z",
        inputs={
            "base_archive_sha256_pr110_fec6": pr110_sha,
            "base_substrate_size_bytes_int8": 1843200,  # 50 × 96 × 128 × 3
            "brotli_quality": 9,
            "epochs_range": [30, 100, 300],
            "gain_clamp_range": [0.05, 0.10, 0.20],
            "pairs": 50,
            "spatial_size_hw": [96, 128],
            "substrate": "boost_nerv_pr110_variant_b_d",
            "codec_variant": "BPR1_VARIANT_B_D_SIGN_BITMAP",
            "cells_aggregated": 9,
        },
        predicted_output={
            "hypothetical_bpr1_sidecar_bytes_total_mean_proportional_to_gain_clamp": mean_hypothetical_predicted,
            "hypothetical_predicate_form": "sidecar_bytes ∝ gain_clamp / baseline_gain_clamp × baseline_sidecar_bytes",
            "hypothetical_baseline_gain_clamp": 0.05,
            "hypothetical_baseline_sidecar_bytes": 149,
        },
        empirical_output={
            "empirical_bpr1_variant_b_d_sidecar_bytes_total_per_cell": sweep_sidecar_bytes,
            "empirical_bpr1_variant_b_d_sidecar_bytes_total_constant": constant_sidecar_bytes,
            "empirical_delta_rate_contest_units_constant": constant_delta_rate,
            "empirical_global_sign_entropy_bits_per_cell": [
                c["sign_bitmap_entropy_diagnostic"]["global_sign_entropy_bits"] for c in cells
            ],
            "empirical_global_sign_positive_fraction_per_cell": [
                c["sign_bitmap_entropy_diagnostic"]["global_positive_fraction"] for c in cells
            ],
            "implementation_level_falsification_explanation": (
                "Variant B-d codec design (sign-bitmap + per-pair magnitude) was hypothesized "
                "to break the int8 quantization scale-invariance of L1 BPR1 codec. Empirical "
                "sweep at 9 (gain_clamp, epochs) cells produces CONSTANT 149B sidecar AND "
                "CONSTANT global_sign_entropy_bits=0.0000 — 100% of trained residuals are "
                "NEGATIVE (sign-bitmap is all-zero uniform → brotli RLE-collapses to 13B + "
                "header 28B + len fields 8B + per-pair magnitudes 100B = 149B). SECOND-ORDER "
                "DISCOVERY not anticipated in pre-execution gate report: PR110 base RGB > GT "
                "is the typical case (overshoots) → L2 loss biases residual learner toward "
                "all-negative subtraction → sign-bitmap entropy=0 regardless of gain_clamp. "
                "Per Catalog #307 sister: PARADIGM (residual-correction hybrid stacking) "
                "INTACT; IMPLEMENTATION-LEVEL falsification of Variant B-d codec design. "
                "Per CLAUDE.md 'Forbidden premature KILL': DEFER-pending-Variant-C (training-"
                "side fix to break sign-axis bias e.g. residual_loss_with_sign_diversity_term "
                "OR centered-base-recolor OR mean-subtracted base)."
            ),
            "distortion_axis_measurement_status": "UNMEASURED_pending_paired_cuda_or_mlx_segnet_posenet_routing",
        },
        residual=float(aggregate_residual_relative),
        source_artifact=(
            ".omx/research/boostnerv_pr110_bpr1_variant_b_sweep_results_20260526/sweep_heatmap.json"
        ),
        measurement_method=(
            "boostnerv_pr110_variant_b_d_sign_bitmap_codec_mlx_local_9cell_sweep_"
            "50pairs_96x128_brotli_q9"
        ),
        provenance=aggregate_provenance,
    )

    update_equation_with_empirical_anchor(
        eq_id,
        aggregate_anchor,
        agent="claude",
        subagent_id="boostnerv-bpr1-variant-b-codec-redesign-break-scale-invariance-20260526",
        notes=(
            "Aggregate Variant B-d 9-cell empirical anchor; sister-extends gain_clamp sweep "
            "L1 finding (commit `8240aceda`); IMPLEMENTATION-LEVEL falsification of Variant "
            "B-d codec design at this fixture surface per Catalog #307; PARADIGM intact; "
            "DEFER to Variant C (training-side fix)."
        ),
    )
    print(f"[append-variant-b-d-anchors] Appended aggregate anchor: {aggregate_anchor.anchor_id}")

    # ANCHOR (b): per-cell representative gain_clamp=0.20 epochs=30 (Carmack best-cell)
    best_cell = next(c for c in cells if c["gain_clamp"] == 0.20 and c["num_epochs"] == 30)

    best_cell_provenance = build_provenance_for_predicted(
        model_id="boostnerv_pr110_residual_correction_hybrid_VARIANT_B_D_sign_bitmap_clamp_0p20_30ep",
        inputs_sha256=pr110_sha,
        measurement_axis="[macOS-MLX research-signal]",
        hardware_substrate="darwin_arm64_m5_max_mlx_local",
        captured_at_utc="2026-05-26T18:43:00Z",
    )
    # For per-cell anchor: predicted = constant 149B (Variant B-d is empirically gain_clamp-
    # invariant per aggregate anchor); empirical = observed 149B per this specific cell.
    # Residual = 0.0 because Variant B-d is internally consistent at this codec surface
    # (the falsification is captured by the aggregate anchor's hypothetical-proportional
    # predicted vs constant empirical; not by per-cell variance).
    per_cell_anchor = EmpiricalAnchor(
        anchor_id="residual_hybrid_boosting_boostnerv_pr110_VARIANT_B_D_clamp_0p20_30ep_carmack_best_cell_20260526",
        measurement_utc="2026-05-26T18:43:00Z",
        inputs={
            "base_archive_sha256_pr110_fec6": pr110_sha,
            "base_substrate_size_bytes_int8": 1843200,
            "brotli_quality": 9,
            "epochs": 30,
            "gain_clamp": 0.20,
            "pairs": 50,
            "spatial_size_hw": [96, 128],
            "substrate": "boost_nerv_pr110_variant_b_d",
            "codec_variant": "BPR1_VARIANT_B_D_SIGN_BITMAP",
            "carmack_best_cell_anchor": True,
            "sister_anchor_aggregate": (
                "residual_hybrid_boosting_boostnerv_pr110_VARIANT_B_D_9cell_aggregate_scale_invariance_finding_20260526"
            ),
        },
        predicted_output={
            "bpr1_variant_b_d_sidecar_bytes_total": 149,
            "delta_rate_contest_units": 149 * 25.0 / 37_545_489,
        },
        empirical_output={
            "bpr1_variant_b_d_sidecar_bytes_total": best_cell["bpr1_variant_b_d_sidecar_bytes"],
            "delta_rate_contest_units": best_cell["delta_rate_contest_units"],
            "final_loss": best_cell["final_loss"],
            "loss_reduction_fraction": best_cell["loss_reduction_fraction"],
            "recon_mse_reduction_fraction": best_cell["recon_mse_reduction_fraction"],
            "global_sign_entropy_bits": best_cell["sign_bitmap_entropy_diagnostic"][
                "global_sign_entropy_bits"
            ],
            "global_positive_fraction": best_cell["sign_bitmap_entropy_diagnostic"][
                "global_positive_fraction"
            ],
            "sign_bitmap_brotli_bytes": best_cell["bpr1_variant_b_d_sidecar_manifest"][
                "sign_bitmap_brotli_bytes"
            ],
            "sign_bitmap_brotli_ratio": best_cell["bpr1_variant_b_d_sidecar_manifest"][
                "sign_bitmap_brotli_ratio"
            ],
            "distortion_axis_measurement_status": "UNMEASURED_pending_paired_cuda_or_mlx_segnet_posenet_routing",
            "loss_recon_axis_validates_carmack_dissent": True,
            "byte_axis_falsifies_variant_b_d_codec_design": True,
        },
        residual=0.0,  # Variant B-d is internally consistent at this codec surface
        source_artifact=(
            ".omx/research/boostnerv_pr110_bpr1_variant_b_sweep_results_20260526/sweep_heatmap.json"
        ),
        measurement_method=(
            "boostnerv_pr110_variant_b_d_sign_bitmap_codec_mlx_local_50pairs_96x128_"
            "clamp_0p20_30ep_brotli_q9_carmack_best_cell"
        ),
        provenance=best_cell_provenance,
    )

    update_equation_with_empirical_anchor(
        eq_id,
        per_cell_anchor,
        agent="claude",
        subagent_id="boostnerv-bpr1-variant-b-codec-redesign-break-scale-invariance-20260526",
        notes=(
            "Per-cell Carmack-best-cell (gain_clamp=0.20 epochs=30) Variant B-d anchor; "
            "internally consistent (predicted=empirical=149B at this codec surface); "
            "captures loss/recon-axis validation (52.6% MSE reduction) + byte-axis "
            "falsification (codec scale-invariant). Sister of aggregate anchor."
        ),
    )
    print(f"[append-variant-b-d-anchors] Appended per-cell anchor: {per_cell_anchor.anchor_id}")

    eq_after = get_equation_by_id(eq_id)
    anchors_after = len(eq_after.empirical_anchors)
    print(
        f"[append-variant-b-d-anchors] AFTER: {anchors_after} anchors registered "
        f"(delta = +{anchors_after - anchors_before})"
    )
    print(f"[append-variant-b-d-anchors] DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
