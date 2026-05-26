# SPDX-License-Identifier: MIT
"""Canonical equation: M-series MPS fp32 matmul drift hardware floor.

Per FIX-WAVE-R1''-K landing 2026-05-26 (per `.omx/research/path_3_k_recursive_adversarial_review_r1_prime_prime_3_axis_20260526.md`).

R1'' independent verification empirically falsified K=COIN++ landing memo's
claimed `5e-3` matmul drift anchor. Independent verification at K-typical
substrate dimensions (32x32 .. 256x64 fp32) on M-series MPS shows:

* **abs_max** consistently **O(1e-2)** (range 1.4e-2 to 5.2e-2)
* **rms** consistently **O(1e-3)** (range 4.5e-3 to 1.2e-2)
* **rel_median** consistently **~7.6e-4** (canonical hardware-class floor)
* **sinusoidal-encoding** drift is **bit-exact** (~1.2e-7) — special case

This canonical equation establishes the M-series MPS fp32 matmul drift
hardware floor as the binding reference for ALL future Path 3 (and Path N)
substrates routing through MLX matmul on Apple Silicon. The floor is
HARD-EARNED-EMPIRICALLY-VERIFIED via independent R1'' verification across
canonical substrate dimensions.

**Canonical-substrate-design implication**: substrates requiring per-matmul
absolute precision tighter than O(1e-2) MUST route through fp32 +
Kahan-compensated summation OR accept the drift as PROXY-grade per
Catalog #341 Tier A observability-only. Substrates that operate on
near-bit-exact primitives (sin/cos positional encodings, elementwise
multiply/add, sigmoid output activation) inherit a much tighter local
floor (~1e-7) and can be claimed clean independently of matmul accumulation.

Per CLAUDE.md "MLX portable-local-substrate authority" + "MPS auth eval is
NOISE": this equation is canonical hardware-FLOOR reference only — NEVER
score authority. Empirical anchors carry `[macOS-MLX research-signal]`
evidence grade + non-promotable markers per Catalog #127/#192/#317/#341.

Cross-references:
  * `.omx/research/path_3_k_recursive_adversarial_review_r1_prime_prime_3_axis_20260526.md`
    (canonical R1'' empirical anchor source)
  * `.omx/research/path_3_recursive_adversarial_review_r1_prime_prime_aggregate_3_axis_landings_h_i_j_k_20260526.md`
    (sister H aggregate; same hardware-floor independently verified across H+K wave)
  * `.omx/research/mlx_first_everywhere_canonical_doctrine_20260526.md`
    (MLX-first doctrine; this equation IS the canonical hardware-floor anchor)
  * `tac.canonical_equations.mlx_pytorch_drift` (downstream scorer-drift sister at decoder-output granularity)
  * Catalog #344 (canonical-equation-reference enforcement at memo surface)
  * Catalog #287 (canonical Provenance umbrella; every empirical claim carries axis + hardware tags)
"""
from __future__ import annotations

from typing import Any

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "mlx_matmul_drift_m_series_canonical_floor_v1"

# Canonical M-series MPS fp32 matmul drift hardware floor (per FIX-WAVE-R1''-K
# independent verification 2026-05-26 across K-typical substrate dimensions).
CANONICAL_ABS_MAX_UPPER_BOUND = 6.0e-2  # 6e-2: covers (64,256)@(256,64) worst-case
CANONICAL_ABS_MAX_TYPICAL = 3.0e-2  # 3e-2: median observed across K-typical dims
CANONICAL_RMS_UPPER_BOUND = 1.5e-2  # 1.5e-2: covers (64,256)@(256,64) worst-case rms
CANONICAL_RMS_TYPICAL = 6.0e-3  # 6e-3: median observed rms
CANONICAL_REL_MEDIAN = 7.6e-4  # 7.6e-4: dimension-independent canonical floor
CANONICAL_SINUSOIDAL_ENCODING_BIT_EXACT = 1.2e-7  # ~1.2e-7: sin/cos special case


def classify_mlx_matmul_drift(
    *,
    measured_abs_max: float,
    measured_rms: float | None = None,
    measured_rel_median: float | None = None,
    matmul_shape: tuple[int, int, int] | None = None,
) -> dict[str, Any]:
    """Classify an MLX matmul drift measurement vs canonical M-series floor.

    Returns a typed verdict dict with all canonical non-promotable markers
    per Catalog #127/#192/#317/#341. Verdict taxonomy:

    * ``BIT_EXACT_LIKE_SINUSOIDAL``: abs_max <= 1e-6 (special primitive class)
    * ``WITHIN_CANONICAL_FLOOR``: abs_max <= CANONICAL_ABS_MAX_UPPER_BOUND
      AND rms <= CANONICAL_RMS_UPPER_BOUND (M-series MPS fp32 hardware floor)
    * ``ABOVE_CANONICAL_FLOOR_NEEDS_MITIGATION``: exceeds canonical floor;
      substrate MUST route through fp32 + Kahan-compensated summation OR
      accept the drift as PROXY-grade per Catalog #341 Tier A observability-only

    Per CLAUDE.md "MPS auth eval is NOISE": the classification is ENGINEERING-
    BRIDGE only. Never score authority; always paired with non-promotable
    markers + axis tag = `[macOS-MLX research-signal]`.
    """
    if not isinstance(measured_abs_max, (int, float)):
        raise ValueError("measured_abs_max must be numeric")
    if measured_abs_max != measured_abs_max:  # NaN check
        raise ValueError("measured_abs_max must not be NaN")
    if measured_abs_max < 0:
        raise ValueError("measured_abs_max must be >= 0")

    if measured_abs_max <= 1e-6:
        verdict = "BIT_EXACT_LIKE_SINUSOIDAL"
    elif (
        measured_abs_max <= CANONICAL_ABS_MAX_UPPER_BOUND
        and (measured_rms is None or measured_rms <= CANONICAL_RMS_UPPER_BOUND)
    ):
        verdict = "WITHIN_CANONICAL_FLOOR"
    else:
        verdict = "ABOVE_CANONICAL_FLOOR_NEEDS_MITIGATION"

    return {
        "equation_id": EQUATION_ID,
        "measured_abs_max": float(measured_abs_max),
        "measured_rms": float(measured_rms) if measured_rms is not None else None,
        "measured_rel_median": (
            float(measured_rel_median) if measured_rel_median is not None else None
        ),
        "matmul_shape": list(matmul_shape) if matmul_shape is not None else None,
        "canonical_abs_max_upper_bound": CANONICAL_ABS_MAX_UPPER_BOUND,
        "canonical_rms_upper_bound": CANONICAL_RMS_UPPER_BOUND,
        "canonical_rel_median": CANONICAL_REL_MEDIAN,
        "verdict": verdict,
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
            "requires_paired_contest_cpu_plus_cuda_for_score_claim",
        ],
    }


def build_mlx_matmul_drift_m_series_canonical_floor_v1() -> CanonicalEquation:
    """Construct the canonical M-series MPS fp32 matmul drift floor equation.

    Empirical anchor: FIX-WAVE-R1''-K independent verification 2026-05-26
    across K-typical substrate dimensions (32x32, 64x64, 128x128, 256x64,
    64x256, MOD_DIM=64 hidden=64 depth=3 chain). Per-dim measurements:

    | (m,k)@(k,n)           | abs_max  | rms      | rel_median |
    |-----------------------|----------|----------|------------|
    | (32,32)@(32,32)       | 1.54e-2  | 4.52e-3  | 7.76e-4    |
    | (64,64)@(64,64)       | 2.42e-2  | 6.16e-3  | 7.66e-4    |
    | (128,128)@(128,128)   | 3.62e-2  | 8.81e-3  | 7.59e-4    |
    | (256,64)@(64,256)     | 2.97e-2  | 6.20e-3  | 7.64e-4    |
    | (64,256)@(256,64)     | 4.60e-2  | 1.24e-2  | 7.75e-4    |

    Sinusoidal encoding (sin+cos): abs_max = 1.19e-7 (bit-exact special case)

    Source artifact: `.omx/research/path_3_k_recursive_adversarial_review_r1_prime_prime_3_axis_20260526.md`
    + R1'' aggregate memo §8 "Empirical anchor — MLX drift measurement"
    + sister FIX-WAVE-R1''-K landing memo §5 "Independent verification table".
    """
    measurement_utc = "2026-05-26T12:45:00Z"
    source_artifact = (
        ".omx/research/path_3_k_recursive_adversarial_review_r1_prime_prime_3_axis_20260526.md"
    )

    # Canonical anchor: K-typical dim sweep (5 representative dims).
    # The anchor records the OBSERVED (1.54e-2 .. 4.60e-2) abs_max vs the
    # PREDICTED CANONICAL_ABS_MAX_UPPER_BOUND (6e-2 declared as bound from
    # the empirical observation set + safety margin). The residual is the
    # normalized fraction of observed/predicted at worst-case dim.
    worst_case_observed_abs_max = 4.60e-2  # (64,256)@(256,64)
    residual = worst_case_observed_abs_max / CANONICAL_ABS_MAX_UPPER_BOUND

    anchor = EmpiricalAnchor(
        anchor_id="r1_pp_k_independent_verification_5_substrate_dims_20260526",
        measurement_utc=measurement_utc,
        inputs={
            "framework": "MLX (mx.matmul) vs numpy (np.matmul)",
            "dtype": "fp32",
            "hardware_substrate": "darwin_arm64_apple_silicon_m_series_mps",
            "substrate_dims_tested": [
                {"shape_a": [32, 32], "shape_b": [32, 32]},
                {"shape_a": [64, 64], "shape_b": [64, 64]},
                {"shape_a": [128, 128], "shape_b": [128, 128]},
                {"shape_a": [256, 64], "shape_b": [64, 256]},
                {"shape_a": [64, 256], "shape_b": [256, 64]},
            ],
            "seed": 42,
            "verification_source": "FIX-WAVE-R1''-K independent reproduction",
        },
        predicted_output={
            "abs_max_upper_bound": CANONICAL_ABS_MAX_UPPER_BOUND,
            "rms_upper_bound": CANONICAL_RMS_UPPER_BOUND,
            "rel_median_canonical_floor": CANONICAL_REL_MEDIAN,
            "hypothesis": "M-series MPS fp32 matmul drift bounded by O(1e-2) abs / O(1e-3) rel across canonical substrate dims",
        },
        empirical_output={
            "per_dim_measurements": [
                {"shape": "(32,32)@(32,32)", "abs_max": 1.54e-2, "rms": 4.52e-3, "rel_median": 7.76e-4},
                {"shape": "(64,64)@(64,64)", "abs_max": 2.42e-2, "rms": 6.16e-3, "rel_median": 7.66e-4},
                {"shape": "(128,128)@(128,128)", "abs_max": 3.62e-2, "rms": 8.81e-3, "rel_median": 7.59e-4},
                {"shape": "(256,64)@(64,256)", "abs_max": 2.97e-2, "rms": 6.20e-3, "rel_median": 7.64e-4},
                {"shape": "(64,256)@(256,64)", "abs_max": 4.60e-2, "rms": 1.24e-2, "rel_median": 7.75e-4},
            ],
            "worst_case_abs_max_observed": worst_case_observed_abs_max,
            "worst_case_rms_observed": 1.24e-2,
            "median_rel_median_observed": 7.66e-4,
            "verdict": "WITHIN_CANONICAL_FLOOR for all 5 dims",
            "sinusoidal_encoding_abs_max": CANONICAL_SINUSOIDAL_ENCODING_BIT_EXACT,
        },
        residual=residual,
        source_artifact=source_artifact,
        measurement_method=(
            "r1_pp_k_independent_verification_mlx_matmul_vs_numpy_fp32_5_substrate_dims"
        ),
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=source_artifact,
            reactivation_criteria=(
                "rerun on wider dim sweep + per-Apple-Silicon-class characterization "
                "(M1/M2/M3/M4/M5/M-Ultra/M-Pro/M-Max) before promoting to "
                "cross-machine canonical hardware-floor reference"
            ),
            measurement_axis="[macOS-MLX research-signal]",
            hardware_substrate="darwin_arm64_apple_silicon_m_series_mps",
            captured_at_utc=measurement_utc,
        ),
    )

    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="M-series MPS fp32 matmul drift canonical hardware floor",
        one_line_summary=(
            "M-series MPS fp32 matmul drift bounded by O(1e-2) abs / O(1e-3) "
            "rms / 7.6e-4 rel-median across canonical substrate dims; "
            "sinusoidal encoding bit-exact (~1e-7) special case."
        ),
        latex_form=(
            r"\|\mathrm{MLX}_{\mathrm{matmul}}(A,B) - \mathrm{numpy}(A,B)\|_\infty "
            r"\leq 6\cdot 10^{-2} \mathrm{\ for\ fp32\ on\ M\text{-}series\ MPS}; "
            r"\mathrm{median}(|\Delta|/|y|) \approx 7.6\cdot 10^{-4}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.mlx_matmul_m_series_floor:classify_mlx_matmul_drift"
        ),
        domain_of_validity={
            "framework_pair": "MLX vs numpy",
            "dtype": "fp32",
            "hardware_substrate_class": "darwin_arm64_apple_silicon_m_series_mps",
            "matmul_dim_range": "32 <= m,k,n <= 256 (canonical substrate dims)",
            "measurement_axis": "[macOS-MLX research-signal]",
            "promotion_authority": False,
            "requires_paired_contest_cpu_cuda_for_promotion": True,
            "applicable_substrate_classes": [
                "coin_pp_implicit_neural_representation",
                "any_path_3_substrate_routing_through_mlx_matmul",
                "any_substrate_using_mx_matmul_at_canonical_dims",
            ],
            "out_of_domain_when": [
                "matmul dim > 256 (compounding error may exceed canonical floor)",
                "fp16 matmul (anti-pattern per axis 2 discipline)",
                "non-M-series Apple Silicon (M1-Ultra/M2-Pro/etc may differ)",
                "non-Apple-Silicon hardware (out of scope; use CUDA/CPU reference)",
            ],
            "bit_exact_primitive_exception": "sin/cos/sigmoid/elementwise are bit-exact (~1e-7) and inherit tighter local floor independent of matmul accumulation",
        },
        units_in={
            "measured_abs_max": "float_dimensionless_max_absolute_drift",
            "measured_rms": "float_dimensionless_rms_drift_optional",
            "measured_rel_median": "float_dimensionless_relative_median_drift_optional",
            "matmul_shape": "tuple_int_int_int_m_k_n_optional",
        },
        units_out={
            "verdict": "str_BIT_EXACT_LIKE_SINUSOIDAL_or_WITHIN_CANONICAL_FLOOR_or_ABOVE_CANONICAL_FLOOR_NEEDS_MITIGATION",
            "canonical_abs_max_upper_bound": "float_dimensionless_6e-2",
            "canonical_rms_upper_bound": "float_dimensionless_1_5e-2",
            "canonical_rel_median": "float_dimensionless_7_6e-4",
        },
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={
            "k_typical_dims_worst_case_abs_max": residual,
        },
        last_calibration_utc=measurement_utc,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            # K substrate test threshold consumes this floor.
            "tac.substrates.coin_pp_implicit_neural_representation.tests.test_basic",
            # R1'' axis 2 reviewer consumes this floor.
            "path_3_recursive_adversarial_review_r1_prime_prime_axis_2_reviewer",
            # Sister #1265 gate threshold rationale consumes this floor.
            "tools.gate_mlx_candidate_contest_equivalence",
            # Cathedral autopilot consumer auto-discovers via Catalog #344.
            "tac.cathedral_consumers.canonical_equation_lookup_consumer",
            # MLX-first doctrine cites this floor as canonical hardware reference.
            "mlx_first_everywhere_canonical_doctrine",
        ),
        canonical_producers=(
            # R1'' independent verification produced this anchor.
            "path_3_fix_wave_r1_prime_prime_k_independent_verification",
            # Future per-class characterization will produce new anchors.
            "tools.measure_pr95_mlx_pytorch_full_decoder_downstream_scorer_drift",
        ),
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=source_artifact,
            reactivation_criteria=(
                "rerun on wider dim sweep + per-Apple-Silicon-class characterization "
                "before promoting to cross-machine canonical hardware-floor reference"
            ),
            measurement_axis="[macOS-MLX research-signal]",
            hardware_substrate="darwin_arm64_apple_silicon_m_series_mps",
            captured_at_utc=measurement_utc,
        ),
    )
