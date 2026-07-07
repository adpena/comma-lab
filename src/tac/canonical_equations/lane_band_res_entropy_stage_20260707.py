# SPDX-License-Identifier: MIT
"""Canonical equation: lane-coeff ENTROPY-STAGE law (LBND4; Mallat/Ballé review row 7 BUILD 1).

Extends the ``lane_band_camera_frame_rd_rate_v1`` family (which established LBND2 41,526 B with
a delta-stream Shannon floor of 26,179 B and concluded the residual is INFORMATION-bound BELOW
the floor). The floor left a ~15 KB CODING gap ABOVE it: LBND2 ships the temporal-delta matrix
as raw uint32 zigzag words and delegates ALL entropy coding to the outer brotli. MEASURED
(n600, real gt_n600 lane fit — the byte-close tool's own build path; brotli-counted,
[macOS-CPU advisory] NON-PROMOTABLE, NEVER MPS): re-coding the IDENTICAL delta matrix through
the ξ delta/context residual entropy stage (``tac.boundary_math.xi_spline_residual_coder``,
best-of-three {varint, zlib9, rice}, self-describing scheme id — the SAME single-source coder
that measured −486 B on the ξ table, commit a44a06fb8) closes most of that gap:

    LBND2 (raw zigzag + brotli):   41,526 B  -> rate_term 0.02765   (the shipping default)
    LBND4 varint  + brotli:        30,892 B  -> rate_term 0.02057   (auto-picked, post-brotli)
    LBND4 rice    + brotli:        33,229 B
    LBND4 zlib9   + brotli:        34,701 B
    Shannon floor (order-0):       26,179 B  -> rate_term 0.01743   (still respected: 30,892 > 26,179)

    measured_byte_delta = 30,892 − 41,526 = **−10,634 B (−25.6%)** = rate_term −0.00708

Decode-reencode is BYTE-IDENTICAL for all three schemes and the dequantized LaneLine statistic
is BIT-IDENTICAL to LBND2's (same L3/L4 quantization grid — the choice is PURE RATE; d_seg and
d_pose are invariant by construction). The post-brotli winner is VARINT, not the raw-smallest
zlib9: brotli finds context in the varint byte stream that the pre-compressed zlib9 stream
denies it — the pick objective must be the COUNTED (post-brotli) bytes, which the auto-pick
implements. Predictive-context coding transfers to the lane payload; the hyperprior does NOT
(side-info calculus inverted at this byte scale — Mallat/Ballé review row 7).

DEFAULT OFF (sealed-config discipline): ``--lane-band-res`` on the byte-close tool selects it;
shipping requires inlining the LBND4 decode half into _INFLATE_PY (the parity gate fails closed
on the unknown magic until then). DSL leg: ``tac.witness_dsl.gauge.LaneBandCoderGauge`` (+
``lane_band_coder_byte_close_flags``); activation-ledger lever ``LaneBandResCoder`` (measured).

means != ends: a rate-lever measurement (advisory, NON-PROMOTABLE); pointer 0.19110 UNMOVED.
"""
from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import (
    build_provenance_for_predicted,
    build_provenance_for_research_sidecar,
)

EQUATION_ID = "lane_band_res_entropy_stage_v1"

_UTC = "2026-07-07T00:00:00Z"
_RATE_DENOM = 37_545_489
_LBND2_BYTES = 41_526     # shipping default (matches lane_band_camera_frame_rd_rate_v1 anchor exactly)
_LBND4_VARINT = 30_892    # auto-picked post-brotli winner
_LBND4_RICE = 33_229
_LBND4_ZLIB9 = 34_701
_SHANNON_FLOOR = 26_179   # order-0 floor from the sister equation (still respected)
_REPORT = "experiments/results/lane_band_res_coder_20260707/lane_band_res_coder_n600_measured.json"


def predict_rate_term(archive_bytes: float) -> float:
    """The exact contest rate-term for a byte-closed stream of ``archive_bytes`` (exact law)."""
    return 25.0 * float(archive_bytes) / _RATE_DENOM


def measured_entropy_stage_delta_bytes() -> int:
    """The MEASURED n600 counted-byte delta of the LBND4 entropy stage vs LBND2 (negative =
    bytes saved). Source artifact: the gitignored-durable JSON at ``_REPORT``."""
    return _LBND4_VARINT - _LBND2_BYTES


def build_lane_band_res_entropy_stage_v1() -> CanonicalEquation:
    """Build the LBND4 lane-coeff entropy-stage canonical equation (Mallat/Ballé row 7)."""

    anchor_lbnd4 = EmpiricalAnchor(
        anchor_id="lane_band_res_lbnd4_varint_rate_n600_20260707",
        measurement_utc=_UTC,
        inputs={"codec": "LBND4_res_varint", "grid": "identical_LBND2_L3_L4",
                "entropy_stage": "xi_spline_residual_coder best-of-three (post-brotli pick)",
                "n_pairs": 600, "byte_closed_serialization": True},
        predicted_output={"rate_term": predict_rate_term(_LBND4_VARINT)},
        empirical_output={
            "brotli_bytes": _LBND4_VARINT,
            "rate_term": predict_rate_term(_LBND4_VARINT),
            "delta_vs_lbnd2_bytes": _LBND4_VARINT - _LBND2_BYTES,
            "per_scheme_brotli_bytes": {"varint": _LBND4_VARINT, "rice": _LBND4_RICE,
                                        "zlib9": _LBND4_ZLIB9},
            "decode_reencode_byte_identical": True,
            "statistic_bit_identical_to_lbnd2": True,
            "note": "-10634 B / -25.6% vs LBND2; floor 26179 B still respected (30892 > 26179)",
        },
        residual=0.0,
        source_artifact=_REPORT,
        measurement_method="n600_real_gt_byte_closed_serialization",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_REPORT,
            reactivation_criteria="re-measure if the lane manifold parameterization, the L3/L4 "
                                  "grid, or the residual-scheme set changes; inline the LBND4 "
                                  "decode into _INFLATE_PY before any shipped selection",
            measurement_axis="[macOS-CPU advisory]",
            hardware_substrate="m5_max_cpu",
        ),
    )
    anchor_pick = EmpiricalAnchor(
        anchor_id="lane_band_res_post_brotli_pick_objective_n600_20260707",
        measurement_utc=_UTC,
        inputs={"quantity": "scheme_pick_objective", "candidates": ["varint", "zlib9", "rice"],
                "raw_smallest": "zlib9", "post_brotli_smallest": "varint"},
        predicted_output={"picked_scheme": "varint"},
        empirical_output={
            "picked_scheme": "varint",
            "note": "the COUNTED objective is post-brotli: brotli finds context in the varint "
                    "stream that a pre-compressed zlib9 stream denies it (raw zlib9 35,139 B < "
                    "varint 48,194 B, yet post-brotli varint 30,892 B < zlib9 34,701 B)",
        },
        residual=0.0,
        source_artifact=_REPORT,
        measurement_method="n600_real_gt_byte_closed_serialization",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_REPORT,
            reactivation_criteria="re-verify the pick if the outer entropy backend (brotli q11) changes",
            measurement_axis="[macOS-CPU advisory]",
            hardware_substrate="m5_max_cpu",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Lane-coeff entropy-stage law (LBND4: ξ delta/context residual stage over the LBND2 grid)",
        one_line_summary=(
            "xi residual entropy stage on the LBND2 delta matrix saves a MEASURED 10,634 B "
            "(41,526->30,892; -25.6%; rate -0.00708) at a bit-identical statistic; floor 26,179 B holds."
        ),
        latex_form=(
            r"b_{LBND4} = \min_{s\in\{v,z,r\}} |\mathrm{brotli}(\mathrm{res}_s(\Delta Q))|;\ "
            r"b_{LBND2}{=}41526 \to b_{LBND4}{=}30892 > b_{floor}{=}26179"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.lane_band_res_entropy_stage_20260707:measured_entropy_stage_delta_bytes"
        ),
        domain_of_validity={
            "vehicle": ["analytic_lane_render_band", "softmax_of_sdf_levelset_witness"],
            "measurement_axis": ["macOS-CPU advisory"],
            "byte_closed": True,
            "note": "PURE-RATE lever (statistic bit-identical to LBND2 => d_seg/d_pose invariant); "
                    "default OFF until the LBND4 decode is inlined into _INFLATE_PY",
        },
        units_in={"archive_bytes": "bytes"},
        units_out={"rate_term": "score_units_contest_rate_term", "delta_bytes": "bytes"},
        empirical_anchors=(anchor_lbnd4, anchor_pick),
        predicted_vs_empirical_residual={
            "n600_real_gt_byte_closed_serialization": 0.0,
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tools/levelset_byte_close_and_eval.py",
            "tac.witness_dsl.gauge",
        ),
        canonical_producers=(
            "experiments/results/lane_band_res_coder_20260707/measure_lane_band_res_n600.py",
            "tac.boundary_math.analytic_lane_render_band",
        ),
        provenance=build_provenance_for_predicted(
            model_id="lane_band_res_entropy_stage.v1",
            inputs_sha256="0" * 64,
            measurement_axis="[macOS-CPU advisory]",
            hardware_substrate="m5_max_cpu",
        ),
    )


def populate_lane_band_res_entropy_stage_equation(
    *,
    path=None,
    lock_path=None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration of the LBND4 entropy-stage law into the canonical
    registry (latest-row-wins query semantics). This is the EQUATIONS leg of DAG FEED-08h;
    the DSL leg is ``tac.witness_dsl.gauge.LaneBandCoderGauge``."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_lane_band_res_entropy_stage_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id,
        notes="lane_band_res_entropy_stage_20260707 (equations leg of DAG FEED-08h; DSL leg = "
              "LaneBandCoderGauge; sister of lane_band_camera_frame_rd_rate_v1 + the xi "
              "delta_res coder a44a06fb8)",
    )
    return eq


__all__ = [
    "EQUATION_ID",
    "build_lane_band_res_entropy_stage_v1",
    "measured_entropy_stage_delta_bytes",
    "populate_lane_band_res_entropy_stage_equation",
    "predict_rate_term",
]
