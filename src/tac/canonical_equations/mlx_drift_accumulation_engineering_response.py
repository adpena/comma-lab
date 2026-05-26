# SPDX-License-Identifier: MIT
"""Canonical equation: MLX/PyTorch drift accumulation + engineering response.

Per T3 grand council deliberation 2026-05-26 (commit ``7d04474cb``,
`.omx/research/t3_grand_council_mlx_pytorch_drift_accumulation_source_and_engineer_away_20260526.md`)
+ operator approval Tier 1 T3 execution: codifies the joint M1+M2 mechanism
diagnosis + HYBRID engineering response (Class 2 PRIMARY + Class 1-SCOPED
PRIMARY + Class 3 FALLBACK) as canonical operational knowledge for future
Path 3 substrate L2 long-training decisions.

Status: **PROVISIONAL** per T3 verdict Decision 7 + Contrarian +
Assumption-Adversary REVISION #1. The equation transitions PROVISIONAL ->
CALIBRATED only when 3+ sister Path 3 substrate L2 long-training anchors
land (per Catalog #344 ``RECALIBRATE_ON_NEW_ANCHORS`` trigger).

Sister equations:
  * ``mlx_matmul_drift_m_series_canonical_floor_v1`` (R1''-K commit
    ``2d59283d4``): M1 mechanism per-op surface — hardware floor for
    individual MLX matmul ops.
  * ``mlx_pytorch_drift_vs_training_depth_z6_v1`` (DRIFT commit
    ``b5fb7c8cc``; 5-anchor empirical): n=5 sub-linear empirical
    reference (alpha=0.47 sat ~2000ep) for Z6 substrate specifically.
  * ``mlx_pytorch_full_decoder_downstream_scorer_drift_propagation_v1``:
    PR95 HNeRV class-shift sister.
  * ``mps_drift_architecture_class_dependent_v1``: per-architecture-class
    drift framework this equation extends to MLX-PyTorch accumulation.

Per CLAUDE.md "Apples-to-apples evidence discipline": the T3 council
reasoned with stale n=2 power-law (alpha~1.45 super-linear extrapolated
to ~1000ep threshold-crossing); the DRIFT 5-anchor empirical anchor
(alpha=0.47 sub-linear, R^2=0.971, sat at 2000-3000ep, threshold-crossing
~4973ep) is the canonical reference. Both anchors are recorded so future
agents can audit the council's reasoning vs the empirical reality and
trigger auto-recalibration as more sister substrate anchors land.

Per Catalog #292 per-deliberation assumption classification:
  * T3 council n=2 anchor: **CARGO-CULTED** (per Assumption-Adversary
    verdict — two datapoints uniquely determine ANY two-parameter family;
    super-linear alpha=1.45 was unprovable at n=2).
  * DRIFT n=5 anchor: **HARD-EARNED-EMPIRICALLY-VERIFIED** (5 datapoints
    bracket sub-linear power-law with R^2=0.971; saturation at 2000ep
    consistent with EMA equilibrium + per-pair gradient noise floor).

Per CLAUDE.md "MLX portable-local-substrate authority" + "MPS auth eval
is NOISE": this equation is canonical ENGINEERING RESPONSE reference
only — NEVER score authority. Empirical anchors carry
``[macOS-MLX research-signal]`` evidence grade + non-promotable markers
per Catalog #127/#192/#317/#341.

Cross-references:
  * ``.omx/research/t3_grand_council_mlx_pytorch_drift_accumulation_source_and_engineer_away_20260526.md``
    (canonical T3 verdict source)
  * ``.omx/research/path_3_d_z6_drift_vs_training_depth_characterization_landed_20260526T125130Z.md``
    (canonical DRIFT empirical 5-anchor source)
  * ``.omx/research/path_3_k_recursive_adversarial_review_r1_prime_prime_3_axis_20260526.md``
    (R1''-K canonical matmul-floor sister anchor)
  * Catalog #344 (canonical-equation-reference enforcement at memo surface)
  * Catalog #292 (per-deliberation assumption classification)
  * Catalog #287 (canonical evidence-tag discipline)
  * Catalog #341 (Tier A canonical-routing-markers)
"""
from __future__ import annotations

from typing import Any

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "mlx_drift_accumulation_engineering_response_v1"

# T3 council n=2 reasoning anchor (CARGO-CULTED per Catalog #292).
T3_COUNCIL_CARGO_CULTED_ALPHA = 1.45
T3_COUNCIL_PREDICTED_THRESHOLD_CROSSING_EPOCHS = 1000

# DRIFT 5-anchor empirical fit (HARD-EARNED per Catalog #292).
DRIFT_EMPIRICAL_ALPHA = 0.4713
DRIFT_EMPIRICAL_C = 1.8105e-5
DRIFT_EMPIRICAL_R_SQUARED = 0.9713
DRIFT_EMPIRICAL_THRESHOLD_CROSSING_EPOCHS = 4973

# Sister #1265 gate threshold (per `tools/gate_mlx_candidate_contest_equivalence_z6.py`).
SISTER_1265_GATE_THRESHOLD = 0.001

# Saturation behavior: 2000->3000ep shows +0.5% drift (anchored evidence of
# asymptotic flattening consistent with EMA equilibrium + per-pair gradient
# noise floor combining to bound drift below threshold indefinitely).
DRIFT_SATURATION_EPOCH_OBSERVED = 2000
DRIFT_SATURATION_RATIO_3000_OVER_2000 = 1.0056  # 0.000725 / 0.000721

# Engineering response class tokens per T3 Decision 2 HYBRID verdict.
ENGINEERING_RESPONSE_CLASS_2_PRIMARY = "class_2_drift_aware_gate_parameterization"
ENGINEERING_RESPONSE_CLASS_1_SCOPED_PRIMARY = "class_1_scoped_kahan_ema_surgical_mitigation"
ENGINEERING_RESPONSE_CLASS_3_FALLBACK = "class_3_proxy_grade_depth_ceiling_fallback"
ENGINEERING_RESPONSE_HYBRID_DEFAULT = "hybrid_class_2_plus_1_scoped_plus_3_fallback"

# Depth bands per T3 Decision 6 + DRIFT sister updated thresholds.
DEPTH_BAND_HEADROOM_AMPLE_EPOCHS = 1500  # <1500ep: safety factor >=2x
DEPTH_BAND_HEADROOM_MARGINAL_EPOCHS = 3000  # 1500-3000ep: Class 1-SCOPED recommended
DEPTH_BAND_CROSSING_RISK_EPOCHS = 4000  # >=4000ep: approaching threshold-crossing

VALID_ENGINEERING_RESPONSE_CLASSES = frozenset(
    {
        ENGINEERING_RESPONSE_CLASS_2_PRIMARY,
        ENGINEERING_RESPONSE_CLASS_1_SCOPED_PRIMARY,
        ENGINEERING_RESPONSE_CLASS_3_FALLBACK,
        ENGINEERING_RESPONSE_HYBRID_DEFAULT,
    }
)


def select_engineering_response(
    *,
    measured_alpha: float,
    training_depth_epochs: int,
    substrate_class: str = "predictive_coding_world_model",
) -> dict[str, Any]:
    """Select engineering response class per T3 HYBRID verdict (Decision 2).

    Returns a typed verdict dict per the canonical contract with the
    selected response class + per-class engineering hooks + canonical
    non-promotable markers per Catalog #127/#192/#317/#341.

    Per T3 Decision 2 HYBRID:
      * Class 2 (drift-aware gate parameterization) is PRIMARY mechanism
        for ALL substrates >= L2 depth (the canonical equation IS the
        Class 2 surface).
      * Class 1-SCOPED (Kahan-EMA + optimizer-state moment-2) is PRIMARY
        for training_depth >= 1500ep (where M2 dominates) AND empirical
        alpha > 0.6 (where M2 accumulation is non-trivial).
      * Class 3 (depth ceiling) is FALLBACK when training_depth approaches
        threshold-crossing depth AND empirical alpha consistent with super-
        linear divergence (>1.0).

    Per Carmack MVP-first phasing (T3 Decision 2 + per-member assumption):
      * Empirically-confirmed sub-linear (alpha <= 0.6, threshold-crossing
        >4000ep) at L2 depth <= 3000ep: NO Kahan-EMA needed; current Sister
        #1265 gate threshold (0.001) UNCHANGED suffices per DRIFT 5-anchor
        empirical evidence.
      * Predicted super-linear (alpha > 1.0) at L2 depth: HYBRID with
        Class 1-SCOPED Kahan-EMA + Class 2 epoch-aware threshold.

    Args:
        measured_alpha: empirical power-law exponent from n>=4 anchors.
        training_depth_epochs: planned L2 / L3 training depth.
        substrate_class: canonical substrate-class token (e.g.
            "predictive_coding_world_model" for Z6, "hnerv_decoder" for
            PR95, "cooperative_receiver" for ATW V2).

    Returns:
        Typed verdict dict with canonical engineering response selection +
        per-class engineering hooks + non-promotable markers.

    Raises:
        ValueError: if measured_alpha < 0 or training_depth_epochs <= 0.
    """
    if not isinstance(measured_alpha, (int, float)):
        raise ValueError("measured_alpha must be numeric")
    if measured_alpha != measured_alpha:  # NaN check
        raise ValueError("measured_alpha must not be NaN")
    if measured_alpha < 0:
        raise ValueError("measured_alpha must be >= 0")
    if not isinstance(training_depth_epochs, int) or training_depth_epochs <= 0:
        raise ValueError(
            f"training_depth_epochs must be positive int; got {training_depth_epochs!r}"
        )
    if not isinstance(substrate_class, str) or not substrate_class.strip():
        raise ValueError("substrate_class must be non-empty str")

    # Class 2 always applies (the canonical equation IS the parameterization).
    selected_classes: list[str] = [ENGINEERING_RESPONSE_CLASS_2_PRIMARY]

    # Class 1-SCOPED gating per T3 Decision 6 + Carmack 30-min smoke rationale.
    kahan_ema_recommended = (
        training_depth_epochs >= DEPTH_BAND_HEADROOM_MARGINAL_EPOCHS
        and measured_alpha > 0.6
    ) or measured_alpha > 1.0
    if kahan_ema_recommended:
        selected_classes.append(ENGINEERING_RESPONSE_CLASS_1_SCOPED_PRIMARY)

    # Class 3 (FALLBACK depth ceiling) per T3 Decision 6 + Hotz "simplest
    # engineering response" lens — only when Class 1-SCOPED insufficient.
    class_3_fallback_triggered = (
        training_depth_epochs >= DEPTH_BAND_CROSSING_RISK_EPOCHS
        and measured_alpha > 1.0
    )
    if class_3_fallback_triggered:
        selected_classes.append(ENGINEERING_RESPONSE_CLASS_3_FALLBACK)

    response_verdict = (
        ENGINEERING_RESPONSE_HYBRID_DEFAULT
        if len(selected_classes) > 1
        else ENGINEERING_RESPONSE_CLASS_2_PRIMARY
    )

    # Depth-band classification per T3 Decision 6 thresholds.
    if training_depth_epochs < DEPTH_BAND_HEADROOM_AMPLE_EPOCHS:
        depth_band = "headroom_ample_class_2_only_suffices"
    elif training_depth_epochs < DEPTH_BAND_HEADROOM_MARGINAL_EPOCHS:
        depth_band = "headroom_marginal_class_1_scoped_recommended_if_alpha_gt_0_6"
    elif training_depth_epochs < DEPTH_BAND_CROSSING_RISK_EPOCHS:
        depth_band = "crossing_risk_class_1_scoped_required"
    else:
        depth_band = "near_threshold_class_3_fallback_eligible"

    return {
        "equation_id": EQUATION_ID,
        "measured_alpha": float(measured_alpha),
        "training_depth_epochs": int(training_depth_epochs),
        "substrate_class": substrate_class,
        "response_verdict": response_verdict,
        "selected_classes": tuple(selected_classes),
        "depth_band": depth_band,
        "kahan_ema_recommended": bool(kahan_ema_recommended),
        "class_3_fallback_triggered": bool(class_3_fallback_triggered),
        # Per-class engineering hooks per T3 OP-ROUTABLEs.
        "class_2_hook": (
            "tools/gate_mlx_candidate_contest_equivalence_z6.py "
            "--gate-threshold-epoch-aware (consumes this equation's threshold "
            "function)"
        ),
        "class_1_scoped_hook": (
            "tac.training.long_training_canonical.KahanCompensatedPolyakEMAShadow "
            "(canonical Kahan-EMA wrapper; LongTrainingConfig.enable_kahan_ema_shadow=True "
            "at epochs > 500 default per T3 Decision 6)"
        ),
        "class_3_fallback_hook": (
            "Catalog #341 Tier A non-promotable markers; route candidate to "
            "observability-only at training_depth > canonical depth budget"
        ),
        # Canonical non-promotable markers per Catalog #127/#192/#317/#341.
        "evidence_grade": "macOS-MLX research-signal",
        "axis_tag": "[macOS-MLX research-signal]",
        "score_claim": False,
        "promotion_eligible": False,
        "promotable": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "blockers": [
            "macos_mlx_research_signal_not_contest_authority",
            "engineering_response_is_routing_signal_not_score_signal",
            "requires_paired_contest_cpu_plus_cuda_for_score_claim",
        ],
        # PROVISIONAL status per T3 Decision 7 + Contrarian REVISION #1.
        "calibration_status": "PROVISIONAL",
        "promotion_to_calibrated_pending": (
            "3+ sister Path 3 substrate L2 long-training drift-vs-depth anchors "
            "land via tac.canonical_equations.update_equation_with_empirical_anchor"
        ),
    }


def build_mlx_drift_accumulation_engineering_response_v1() -> CanonicalEquation:
    """Construct the canonical MLX drift accumulation + engineering response equation.

    Records two anchors:

    1. T3 council n=2 reasoning anchor (alpha=1.45 super-linear extrapolation;
       CARGO-CULTED per Catalog #292 Assumption-Adversary verdict — two
       datapoints cannot distinguish power-law shape).
    2. DRIFT 5-anchor empirical (alpha=0.4713 sub-linear sat ~2000ep,
       R^2=0.971; HARD-EARNED-EMPIRICALLY-VERIFIED per Catalog #292 +
       commit ``60a9de751``).

    The empirical n=5 anchor reveals the n=2 council prediction was an
    UPPER BOUND consistent with cargo-culted over-extrapolation; actual
    Z6 trajectory is sub-linear with saturation, well within Sister #1265
    gate's 0.001 threshold across all anchored depths.

    Returns:
        Canonical equation registered with two empirical anchors + 3
        producers + 4 consumers per the canonical contract.
    """
    measurement_utc_t3 = "2026-05-26T12:51:35Z"
    measurement_utc_drift = "2026-05-26T12:51:30Z"

    source_artifact_t3 = (
        ".omx/research/t3_grand_council_mlx_pytorch_drift_accumulation_source_and_engineer_away_20260526.md"
    )
    source_artifact_drift = (
        ".omx/research/path_3_d_z6_drift_vs_training_depth_characterization_landed_20260526T125130Z.md"
    )

    # Anchor 1: T3 council n=2 CARGO-CULTED reasoning anchor.
    # Per Catalog #292: residual = |predicted_threshold_crossing -
    # empirical_threshold_crossing| / empirical_threshold_crossing.
    t3_residual = abs(
        T3_COUNCIL_PREDICTED_THRESHOLD_CROSSING_EPOCHS
        - DRIFT_EMPIRICAL_THRESHOLD_CROSSING_EPOCHS
    ) / DRIFT_EMPIRICAL_THRESHOLD_CROSSING_EPOCHS

    anchor_t3 = EmpiricalAnchor(
        anchor_id="t3_grand_council_n_2_power_law_reasoning_anchor_20260526",
        measurement_utc=measurement_utc_t3,
        inputs={
            "datapoints_count": 2,
            "datapoints": [
                {"epochs": 30, "drift_max_abs": 0.000009},
                {"epochs": 300, "drift_max_abs": 0.000253},
            ],
            "substrate_class": "predictive_coding_world_model",
            "fit_method": "log_log_2_point_unique_determination",
            "council_tier": "T3",
            "council_quorum_attendees": 26,
            "assumption_adversary_classification": "CARGO_CULTED",
            "assumption_adversary_rationale": (
                "two datapoints uniquely determine ANY two-parameter family "
                "(power-law, linear-with-knee, polynomial, exponential); "
                "alpha=1.45 super-linear fit unprovable at n=2"
            ),
        },
        predicted_output={
            "alpha": T3_COUNCIL_CARGO_CULTED_ALPHA,
            "threshold_crossing_epochs": T3_COUNCIL_PREDICTED_THRESHOLD_CROSSING_EPOCHS,
            "mechanism_decomposition": (
                "M1 per-op composed precision drift (alpha~0.5-0.7) + "
                "M2 EMA shadow accumulation (alpha~0.7-0.9 damped) = "
                "joint alpha~1.5 super-linear (cargo-culted)"
            ),
            "recommended_engineering_response": "HYBRID Class 2 + Class 1-SCOPED",
        },
        empirical_output={
            "datapoints_count_at_assertion": 2,
            "empirical_at_2_points_only": True,
            "extrapolated_threshold_crossing_at_extrapolation": (
                T3_COUNCIL_PREDICTED_THRESHOLD_CROSSING_EPOCHS
            ),
            "subsequent_falsification_per_drift_5_anchor": (
                "DRIFT commit 60a9de751 (n=5) empirically FALSIFIED via "
                "5-anchor sub-linear fit alpha=0.4713 R^2=0.971; "
                "actual threshold-crossing ~4973ep, NOT ~1000ep"
            ),
        },
        residual=t3_residual,
        source_artifact=source_artifact_t3,
        measurement_method="t3_council_n_2_power_law_extrapolation_cargo_culted_per_assumption_adversary",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=source_artifact_t3,
            reactivation_criteria=(
                "anchor preserved per Catalog #110/#113 APPEND-ONLY "
                "HISTORICAL_PROVENANCE — codifies T3 council reasoning at "
                "moment of deliberation; subsequent DRIFT anchor (n=5) is "
                "the empirical reality and supersedes this anchor's "
                "extrapolation per Catalog #344 RECALIBRATE_ON_NEW_ANCHORS"
            ),
            measurement_axis="[macOS-MLX research-signal]",
            hardware_substrate="darwin_arm64_apple_silicon_m5_max_mps",
            captured_at_utc=measurement_utc_t3,
        ),
    )

    # Anchor 2: DRIFT 5-anchor empirical HARD-EARNED reference.
    # Residual = log-log fit residual averaged over 5 anchors per the
    # DRIFT memo's per-anchor residuals table.
    drift_residual = 0.0610  # mean of per-anchor residuals (0.049, 0.055, 0.024, 0.097, 0.080)

    anchor_drift = EmpiricalAnchor(
        anchor_id="path_3_d_z6_drift_vs_training_depth_5_anchor_landed_20260526",
        measurement_utc=measurement_utc_drift,
        inputs={
            "datapoints_count": 5,
            "datapoints": [
                {"epochs": 300, "drift_max_abs": 0.000253},
                {"epochs": 500, "drift_max_abs": 0.000358},
                {"epochs": 1000, "drift_max_abs": 0.000458},
                {"epochs": 2000, "drift_max_abs": 0.000721},
                {"epochs": 3000, "drift_max_abs": 0.000725},
            ],
            "substrate_class": "predictive_coding_world_model",
            "substrate_id": "time_traveler_l5_z6",
            "fit_method": "log_log_5_point_least_squares",
            "ema_decay": 0.997,
            "optimizer_class": "mlx_optimizers_AdamW",
            "resolution": "48x64",
            "pair_count": 50,
            "hardware_substrate": "darwin_arm64_apple_silicon_m5_max_mps",
            "assumption_adversary_classification": "HARD_EARNED_EMPIRICALLY_VERIFIED",
        },
        predicted_output={
            "alpha": DRIFT_EMPIRICAL_ALPHA,
            "constant_c": DRIFT_EMPIRICAL_C,
            "r_squared": DRIFT_EMPIRICAL_R_SQUARED,
            "threshold_crossing_epochs": DRIFT_EMPIRICAL_THRESHOLD_CROSSING_EPOCHS,
            "saturation_observed_at_epochs": DRIFT_SATURATION_EPOCH_OBSERVED,
            "saturation_ratio_3000_over_2000": DRIFT_SATURATION_RATIO_3000_OVER_2000,
            "mechanism_interpretation": (
                "sub-linear sat consistent with EMA equilibrium (Polyak "
                "0.997 ~333-step window) + per-pair gradient noise floor "
                "combining to bound drift below threshold indefinitely"
            ),
            "recommended_engineering_response_per_drift_anchor": (
                "Class 2 alone suffices for L3 sweep (500-1500ep); "
                "NO Class 1-SCOPED Kahan-EMA required at this depth; "
                "Sister #1265 gate threshold 0.001 unchanged"
            ),
        },
        empirical_output={
            "per_anchor_residuals": [0.0490, 0.0550, 0.0243, 0.0968, 0.0800],
            "max_residual": 0.0968,
            "mean_residual": 0.0610,
            "well_calibrated": True,
            "headroom_at_300ep": 3.95,
            "headroom_at_500ep": 2.79,
            "headroom_at_1000ep": 2.18,
            "headroom_at_2000ep": 1.39,
            "headroom_at_3000ep": 1.38,
            "verdict": "PASS at all 5 anchored depths; Sister #1265 0.001 threshold unchanged",
        },
        residual=drift_residual,
        source_artifact=source_artifact_drift,
        measurement_method="path_3_d_z6_drift_vs_training_depth_5_anchor_empirical_fit_canonical_anchor",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=source_artifact_drift,
            reactivation_criteria=(
                "every Path 3 substrate's L2 long-training will land a "
                "sister drift-vs-depth anchor; after >=3 sister substrate "
                "anchors land, transition this equation PROVISIONAL -> "
                "CALIBRATED per Catalog #344 RECALIBRATE_ON_NEW_ANCHORS"
            ),
            measurement_axis="[macOS-MLX research-signal]",
            hardware_substrate="darwin_arm64_apple_silicon_m5_max_mps",
            captured_at_utc=measurement_utc_drift,
        ),
    )

    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="MLX/PyTorch drift accumulation + engineering response (PROVISIONAL)",
        one_line_summary=(
            "Joint M1 per-op + M2 EMA shadow + M3 AdamW state drift accumulation "
            "with HYBRID Class 2+1-SCOPED+3-FALLBACK engineering response selector."
        ),
        latex_form=(
            r"\mathrm{drift}(\epsilon, \mathrm{class}) = c \cdot \epsilon^{\alpha} "
            r"\ \mathrm{where}\ \alpha = \alpha_{M1}(\mathrm{correlated}) + "
            r"\alpha_{M2}(\mathrm{damped}) + \alpha_{M3}(\mathrm{optimizer\_state}); "
            r"\ \mathrm{empirical\_Z6}: \alpha=0.47,\ c=1.81\cdot 10^{-5}; "
            r"\ \mathrm{predicted\_T3}: \alpha=1.45\ (\mathrm{cargo\_culted\_at\_n=2})"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.mlx_drift_accumulation_engineering_response:select_engineering_response"
        ),
        domain_of_validity={
            "framework_pair": "MLX vs PyTorch",
            "dtype": "fp32",
            "hardware_substrate_class": "darwin_arm64_apple_silicon_m_series_mps",
            "training_depth_range": "30 <= epochs <= 3000 (anchored); extrapolation to ~4973 ep crossing",
            "substrate_classes_anchored": ["predictive_coding_world_model"],
            "substrate_classes_pending_anchor": [
                "hnerv_decoder",
                "cooperative_receiver",
                "procedural_codebook",
                "world_model",
                "block_neural_representation",
                "implicit_neural_representation",
                "predictive_coding_alternative",
            ],
            "ema_decay_canonical": 0.997,
            "optimizer_class_z6": "mlx_optimizers_AdamW",
            "measurement_axis": "[macOS-MLX research-signal]",
            "promotion_authority": False,
            "calibration_status": "PROVISIONAL",
            "promotion_to_calibrated_trigger": (
                "3+ sister Path 3 substrate L2 long-training anchors land "
                "per Catalog #344 RECALIBRATE_ON_NEW_ANCHORS"
            ),
            "applicable_substrate_classes": [
                "all_path_3_substrates_using_canonical_l2_helper",
                "all_substrates_using_tac_training_long_training_canonical",
                "all_substrates_using_PolyakEMAShadow_with_canonical_decay_0_997",
            ],
            "out_of_domain_when": [
                "training_depth > 3000 (empirical anchor saturation observed; extrapolation uncertain)",
                "training_depth < 30 (below canonical L1 baseline anchor)",
                "ema_decay != 0.997 (per CLAUDE.md 'EMA NON-NEGOTIABLE' default; non-canonical decays need own anchor)",
                "non-MLX-PyTorch framework pair (CPU-only / CUDA-only / pure-torch out of scope)",
                "non-M-series Apple Silicon (M1/M2/M3 may have different per-op floors per R1''-K sister)",
            ],
            "engineering_response_classes": [
                ENGINEERING_RESPONSE_CLASS_2_PRIMARY,
                ENGINEERING_RESPONSE_CLASS_1_SCOPED_PRIMARY,
                ENGINEERING_RESPONSE_CLASS_3_FALLBACK,
                ENGINEERING_RESPONSE_HYBRID_DEFAULT,
            ],
            "sister_equation_mlx_matmul_floor": "mlx_matmul_drift_m_series_canonical_floor_v1",
            "sister_equation_z6_drift_vs_depth": "mlx_pytorch_drift_vs_training_depth_z6_v1",
            "sister_equation_pr95_full_decoder": "mlx_pytorch_full_decoder_downstream_scorer_drift_propagation_v1",
        },
        units_in={
            "measured_alpha": "float_dimensionless_power_law_exponent",
            "training_depth_epochs": "int_positive_epoch_count",
            "substrate_class": "str_canonical_substrate_class_token",
        },
        units_out={
            "response_verdict": "str_canonical_engineering_response_class_token",
            "selected_classes": "tuple_str_one_or_more_response_class_tokens",
            "depth_band": "str_canonical_depth_band_token",
            "kahan_ema_recommended": "bool_class_1_scoped_recommendation",
            "class_3_fallback_triggered": "bool_class_3_fallback_recommendation",
            "calibration_status": "str_PROVISIONAL_or_CALIBRATED_or_DEFER_PENDING_DEEPER_INVESTIGATION",
        },
        empirical_anchors=(anchor_t3, anchor_drift),
        predicted_vs_empirical_residual={
            "t3_council_n_2_extrapolation_cargo_culted": t3_residual,
            "drift_5_anchor_empirical_hard_earned": drift_residual,
        },
        last_calibration_utc=measurement_utc_t3,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            # Cascade doctrine L6 gate routing per T3 Decision 4 (sister Tier1-T3-OP7-OP8).
            "path_3_canonical_substrate_development_cascade_doctrine",
            # MLX-first doctrine forecast amendment per T3 Decision 5 (sister Tier1-T3-OP7-OP8).
            "mlx_first_everywhere_canonical_doctrine",
            # Sister #1265 gate threshold parameterization per T3 OP-ROUTABLE #5 (deferred Tier 2).
            "tools.gate_mlx_candidate_contest_equivalence_z6",
            # Future Path 3 substrate L2 long-training decisions consume the selector.
            "tac.training.long_training_canonical",
            # Cathedral autopilot auto-discovery via Catalog #335 paradigm.
            "tac.cathedral_consumers.canonical_equation_lookup_consumer",
        ),
        canonical_producers=(
            # T3 grand council deliberation produced the n=2 council anchor.
            "t3_grand_council_mlx_pytorch_drift_accumulation_source_and_engineer_away_20260526",
            # DRIFT 5-anchor empirical produced the n=5 HARD-EARNED anchor.
            "path_3_d_z6_drift_vs_training_depth_characterization_landed_20260526",
            # R1''-K canonical floor reference sister anchor.
            "fix_wave_r1_prime_prime_k_close_coin_pp_empirical_claim_falsification",
        ),
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=source_artifact_t3,
            reactivation_criteria=(
                "PROVISIONAL -> CALIBRATED upgrade pending 3+ sister Path 3 "
                "substrate L2 long-training drift-vs-depth anchors per "
                "Catalog #344 RECALIBRATE_ON_NEW_ANCHORS trigger; future "
                "B'/C'/D/E/F/G/H/I/J/K L2 long-trainings will land sister "
                "anchors automatically per canonical L2 helper consumption"
            ),
            measurement_axis="[macOS-MLX research-signal]",
            hardware_substrate="darwin_arm64_apple_silicon_m5_max_mps",
            captured_at_utc=measurement_utc_t3,
        ),
    )


__all__ = [
    "DEPTH_BAND_CROSSING_RISK_EPOCHS",
    "DEPTH_BAND_HEADROOM_AMPLE_EPOCHS",
    "DEPTH_BAND_HEADROOM_MARGINAL_EPOCHS",
    "DRIFT_EMPIRICAL_ALPHA",
    "DRIFT_EMPIRICAL_C",
    "DRIFT_EMPIRICAL_R_SQUARED",
    "DRIFT_EMPIRICAL_THRESHOLD_CROSSING_EPOCHS",
    "DRIFT_SATURATION_EPOCH_OBSERVED",
    "DRIFT_SATURATION_RATIO_3000_OVER_2000",
    "ENGINEERING_RESPONSE_CLASS_1_SCOPED_PRIMARY",
    "ENGINEERING_RESPONSE_CLASS_2_PRIMARY",
    "ENGINEERING_RESPONSE_CLASS_3_FALLBACK",
    "ENGINEERING_RESPONSE_HYBRID_DEFAULT",
    "EQUATION_ID",
    "SISTER_1265_GATE_THRESHOLD",
    "T3_COUNCIL_CARGO_CULTED_ALPHA",
    "T3_COUNCIL_PREDICTED_THRESHOLD_CROSSING_EPOCHS",
    "VALID_ENGINEERING_RESPONSE_CLASSES",
    "build_mlx_drift_accumulation_engineering_response_v1",
    "select_engineering_response",
]
