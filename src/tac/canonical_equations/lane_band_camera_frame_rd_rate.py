# SPDX-License-Identifier: MIT
"""Canonical equation: camera-frame lane-band rate-distortion rate law (Wave-F Stage-1).

MEASURED (n600, real gt_n600 lstars, byte-closed serialization, [macOS-CPU advisory]
NON-PROMOTABLE, NEVER MPS): the per-pair analytic-lane manifold serialized NAIVELY
(each pair independent, float64 coeffs, brotli) costs 156,340 B = rate_term 0.1041
(launch-unusable vs a 0.19110 frontier). The OPTIMAL LBND2 rate-distortion codec
(L4 lateral-sorted fixed slots + carry-forward hold -> L3 geometric-tolerance quantize
each coeff to its OWN step -> L2 temporal-delta across the 600 pairs -> zigzag int32 ->
outer brotli) costs 41,526 B = rate_term 0.02765 = a 3.76x rate reduction, decode-consistent
(bit-exact through R) by construction. The delta-stream Shannon floor is 26,179 B (0.01743) --
so EVEN a perfect entropy coder on the camera-frame per-pair coeffs CANNOT cross the +0.005
rate floor (7,509 B): the residual is INFORMATION-bound, not coding-bound (=> the next lever
is a SOURCE re-parameterization, the ego-factorization sister equation, NOT a better coder).

The rate term is the exact contest rate law over the byte-closed archive stream:

    rate_term(bytes) = 25 * bytes / 37_545_489

PTC1 / constriction range-coding is MEASURED-DOMINATED at this data shape (43,153 B >
brotli 41,526 B: the per-dim transmitted-PMF header cost swamps the coded stream) => brotli
on zigzag-int32 is the correct backend (KILLS the "add a range coder" gold-plating +
keeps inflate dep-free). Induced lateral RMS from quantization = 0.0212 m (~2 cm, sub-pixel).

Consumers: the byte-close 5th-block codec (tools/levelset_byte_close_and_eval.py) + the DSL
LaneGauge byte-close realization + the ego-factorization sister equation.
Producers: tools/wave_f_lane_band_rd_rate_n600.py (the $0 n600 measured-rate tool).
"""
from __future__ import annotations

from tac.canonical_equations.equation import (
    CanonicalEquation,
    EmpiricalAnchor,
    RECALIBRATE_ON_NEW_ANCHORS,
)
from tac.provenance.builders import (
    build_provenance_for_predicted,
    build_provenance_for_research_sidecar,
)

EQUATION_ID = "lane_band_camera_frame_rd_rate_v1"

# MEASURED anchors (n600, real gt_n600, byte-closed serialization;
# .omx/research/wave_f_lane_band_rd_rate_n600_measured.json).
_RATE_DENOM = 37_545_489
_NAIVE_BYTES = 156_340       # naive LBND1 (per-pair float64, brotli)
_RD_BYTES = 41_526           # optimal LBND2 RD codec
_SHANNON_FLOOR_BYTES = 26_179  # delta-stream Shannon floor (perfect entropy coder on the coeffs)
_PTC1_BYTES = 43_153         # constriction range-coder (DOMINATED: > brotli 41_526)


def predict_rate_term(archive_bytes: float) -> float:
    """The exact contest rate-term for a byte-closed lane-band stream of ``archive_bytes``.

    ``rate_term = 25 * bytes / 37_545_489`` -- the third term of the contest score
    ``S = 100*d_seg + sqrt(10*d_pose) + rate_term``. Exact, not a fit."""

    return 25.0 * float(archive_bytes) / _RATE_DENOM


def build_lane_band_camera_frame_rd_rate_v1() -> CanonicalEquation:
    """Build the Wave-F Stage-1 camera-frame lane-band RD rate-law canonical equation."""

    report = ".omx/research/wave_f_lane_band_rd_rate_n600_measured.json"

    anchor_naive = EmpiricalAnchor(
        anchor_id="lane_band_naive_lbnd1_rate_n600_20260702",
        measurement_utc="2026-07-02T00:00:00Z",
        inputs={"codec": "naive_LBND1_float64_brotli", "n_pairs": 600, "byte_closed": True},
        predicted_output={"rate_term": predict_rate_term(_NAIVE_BYTES)},
        empirical_output={"brotli_bytes": _NAIVE_BYTES, "rate_term": 0.10410038873112026},
        residual=abs(predict_rate_term(_NAIVE_BYTES) - 0.10410038873112026),
        source_artifact=report,
        measurement_method="n600_real_gt_byte_closed_serialization",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=report,
            reactivation_criteria="re-measure if the lane manifold parameterization changes",
            measurement_axis="[macOS-CPU advisory]",
            hardware_substrate="m5_max_cpu",
        ),
    )
    anchor_rd = EmpiricalAnchor(
        anchor_id="lane_band_rd_lbnd2_rate_n600_20260702",
        measurement_utc="2026-07-02T00:00:00Z",
        inputs={"codec": "optimal_LBND2_rd", "levers": "L4_slots+L3_geomtol_quantize+L2_temporal_delta+zigzag+brotli",
                "n_pairs": 600, "byte_closed": True, "induced_lateral_rms_m": 0.0212},
        predicted_output={"rate_term": predict_rate_term(_RD_BYTES)},
        empirical_output={"brotli_bytes": _RD_BYTES, "rate_term": 0.02765045888735129,
                          "rd_vs_naive_ratio": 0.2656, "note": "3.76x rate reduction, decode-consistent"},
        residual=abs(predict_rate_term(_RD_BYTES) - 0.02765045888735129),
        source_artifact=report,
        measurement_method="n600_real_gt_byte_closed_serialization",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=report,
            reactivation_criteria="re-measure through R after a fine-tune with --lane-render-band active",
            measurement_axis="[macOS-CPU advisory]",
            hardware_substrate="m5_max_cpu",
        ),
    )
    anchor_floor = EmpiricalAnchor(
        anchor_id="lane_band_delta_stream_shannon_floor_n600_20260702",
        measurement_utc="2026-07-02T00:00:00Z",
        inputs={"quantity": "delta_stream_shannon_floor", "coding_bound": True, "n_pairs": 600,
                "ptc1_dominated_bytes": _PTC1_BYTES},
        predicted_output={"rate_term": predict_rate_term(_SHANNON_FLOOR_BYTES)},
        empirical_output={"floor_bytes": _SHANNON_FLOOR_BYTES, "rate_term": 0.01743160809544923,
                          "note": "even a PERFECT entropy coder cannot cross +0.005 (7509 B): residual is INFORMATION-bound, not coding-bound"},
        residual=abs(predict_rate_term(_SHANNON_FLOOR_BYTES) - 0.01743160809544923),
        source_artifact=report,
        measurement_method="n600_real_gt_shannon_entropy_of_delta_stream",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=report,
            reactivation_criteria="the floor moves only via a SOURCE re-parameterization (ego-factorization), not a better coder",
            measurement_axis="[macOS-CPU advisory]",
            hardware_substrate="m5_max_cpu",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Camera-frame lane-band rate-distortion rate law (Wave-F Stage-1)",
        one_line_summary=(
            "Optimal LBND2 RD codec cuts the naive lane-band rate 3.76x (0.1041->0.02765, "
            "156340->41526 B); Shannon floor 0.01743 => residual is INFORMATION-bound."
        ),
        latex_form=(
            r"\text{rate\_term}(b) = 25\,b / 37{,}545{,}489;\ "
            r"b_{naive}{=}156340 \to b_{RD}{=}41526 \to b_{floor}{=}26179"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.lane_band_camera_frame_rd_rate:predict_rate_term"
        ),
        domain_of_validity={
            "vehicle": ["analytic_lane_render_band", "softmax_of_sdf_levelset_witness"],
            "measurement_axis": ["macOS-CPU advisory"],
            "byte_closed": True,
            "note": "the RATE half; the d_seg WIN is the SEPARATE #205 trained-in run (a post-hoc band is break-even)",
        },
        units_in={"archive_bytes": "bytes"},
        units_out={"rate_term": "score_units_contest_rate_term"},
        empirical_anchors=(anchor_naive, anchor_rd, anchor_floor),
        predicted_vs_empirical_residual={
            "n600_real_gt_byte_closed_serialization": 0.0,
        },
        last_calibration_utc="2026-07-02T00:00:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tools/levelset_byte_close_and_eval.py",
            "tac.witness_dsl.gauge",
            "tac.canonical_equations.lane_band_ego_factorization_source_reparam",
        ),
        canonical_producers=(
            "tools/wave_f_lane_band_rd_rate_n600.py",
        ),
        provenance=build_provenance_for_predicted(
            model_id="lane_band_camera_frame_rd_rate.v1",
            inputs_sha256="0" * 64,
            measurement_axis="[macOS-CPU advisory]",
            hardware_substrate="m5_max_cpu",
        ),
    )


__all__ = [
    "EQUATION_ID",
    "predict_rate_term",
    "build_lane_band_camera_frame_rd_rate_v1",
]
