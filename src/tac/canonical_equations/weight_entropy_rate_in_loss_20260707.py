# SPDX-License-Identifier: MIT
"""Canonical equation: the Ballé rate-in-the-loss WEIGHT-ENTROPY lever (torch-measured;
MLX-ported 2026-07-07 — council draft ``DRAFT_derived_optimal_next_run_for_council_20260707.md``
§22(2) fold-blocker CLOSED).

THE LAW (train-time entropy descent sets the deployed byte floor)
-----------------------------------------------------------------
The counted decoder/witness payload is int8+brotli of the learned weights; brotli sits near the
order-0 floor, so ``archive_bytes ≈ Σ_t H(symbols_t)·numel_t/8`` — the ONLY rate lever left is
lowering ``H`` itself, which is set by TRAINING. Adding ``λ·rate_term`` (the expected symbol
codelength on the contest rate scale ``25·bits/8/37_545_489``) to the training loss pulls the
weight-symbol distribution toward low entropy → a lower deployed byte floor.

MEASURED anchors (TORCH VEHICLE, 2026-06-20 — learned per-channel Ballé prior + STE noise;
``tac.torch_vehicle.weight_entropy_penalty``; [contest-CPU advisory], NON-PROMOTABLE):
  * λ50 cut the LIVE-decoder order-0 H by −1.55 bits/wt and LIVE archive bytes by −16,007 B
    (−19.6%) through brotli (the ancestor-vehicle bit-spend proof).
  * The SHIPPED EMA shadow at decay 0.999 did NOT shrink in short runs (+72..87 B — EMA lag);
    the ema0.9 A/B PROVED the H-cut translates to shipped bytes.
  * C1a stacking is NET-NEGATIVE (same-quantity estimators interfere; ``supersedes_c1a=True``).
  * λ* open in {5,15,30} (λ50/ema0.9 overshoots into d_seg harm).

PORT FACT (2026-07-07, this module's registration event): the lever now ALSO exists on the
LEVELSET MLX trainer (``--weight-entropy-penalty-lambda`` on
``experiments/train_levelset_witness_realized_through_R_mlx.py``) as a DETERMINISTIC
soft-histogram surrogate (``tac.boundary_math.weight_entropy_penalty_mlx`` — state-free, no RNG,
no learned prior; counted weights only, free bank excluded per rule 118), routed into BOTH the
serial and micro-batch loss paths, held by the DSL ``Lever`` factory
``tac.witness_dsl.curriculum_dsl.WeightEntropyPenaltyMLX``. BORROWED-NUMBER FIREWALL: the torch
anchors above are attributed to the TORCH vehicle + its learned-prior term; NO byte/score number
transfers to the MLX lever. The MLX lever is NEVER-FIRED; its OWED anchor (the reactivation /
duty-to-measure) is a byte-closed n600 A/B on the witness (λ-on vs λ=0 at equal d_seg/d_pose,
real ``quantize_levelset_blob`` bytes + the hard
``measured_symbol_entropy_bits_numpy`` metric).

means != ends: rate-lever rows (advisory, NON-PROMOTABLE); pointer contest-CPU 0.19110 UNMOVED.
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

EQUATION_ID = "weight_entropy_rate_in_loss_lever_v1"

_UTC_TORCH = "2026-06-20T00:00:00Z"
_RATE_DENOM = 37_545_489
_TORCH_LIVE_DELTA_BYTES = -16_007   # λ50, LIVE decoder, torch vehicle (NOT transferable)
_TORCH_LIVE_DELTA_H = -1.55         # bits/weight, hard order-0, torch vehicle
_MEMO_BYTECLOSE = ".omx/research/weight_entropy_penalty_balle_adversarial_review_byteclose_20260620.md"
_MEMO_INTEGRATION = ".omx/research/weight_entropy_penalty_balle_rate_term_integration_20260620T202332Z.md"


def predict_rate_term(archive_bytes: float) -> float:
    """The exact contest rate-term for a byte-closed payload of ``archive_bytes`` (exact law)."""
    return 25.0 * float(archive_bytes) / _RATE_DENOM


def torch_vehicle_measured_live_delta_bytes() -> int:
    """The MEASURED torch-vehicle λ50 LIVE-decoder byte delta (the ancestor anchor). Attributed
    to the torch vehicle's learned-prior term ONLY — NOT transferable to the MLX lever (whose
    own n600 A/B is owed)."""
    return _TORCH_LIVE_DELTA_BYTES


def build_weight_entropy_rate_in_loss_lever_v1() -> CanonicalEquation:
    """Build the rate-in-the-loss weight-entropy lever equation (torch anchors + MLX port fact)."""

    anchor_torch_live = EmpiricalAnchor(
        anchor_id="weight_entropy_torch_lambda50_live_decoder_byteclose_20260620",
        measurement_utc=_UTC_TORCH,
        inputs={"vehicle": "torch_split_by_head_basin", "term": "learned_prior_balle_ste_noise",
                "lambda": 50.0, "decoder_state": "LIVE (not the shipped EMA shadow)",
                "byte_closed_codec": True},
        predicted_output={"direction": "archive_bytes decrease at equal epochs"},
        empirical_output={
            "delta_live_archive_bytes": _TORCH_LIVE_DELTA_BYTES,
            "delta_live_archive_frac": -0.196,
            "delta_hard_order0_entropy_bits_per_weight": _TORCH_LIVE_DELTA_H,
            "ema_0999_shipped_delta_bytes": "+72..87 (EMA lag; shadow did NOT shrink in short runs)",
            "ema_09_ab": "PROVED the H-cut translates to shipped bytes",
            "c1a_stacking": "NET-NEGATIVE (supersedes_c1a=True landed)",
            "lambda_star": "open in {5,15,30} (lambda50/ema0.9 overshoots into d_seg harm)",
            "note": "TORCH-VEHICLE anchor; NOT transferable to the MLX levelset lever",
        },
        residual=0.0,
        source_artifact=_MEMO_BYTECLOSE,
        measurement_method="torch_vehicle_byte_closed_codec_ab",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO_BYTECLOSE,
            reactivation_criteria="net-S n600 A/B on the torch vehicle still OWED; re-measure if "
                                  "the codec grid or EMA decay policy changes",
            measurement_axis="[contest-CPU advisory]",
            hardware_substrate="m5_max_cpu",
        ),
    )
    anchor_mlx_port = EmpiricalAnchor(
        anchor_id="weight_entropy_mlx_port_never_fired_20260707",
        measurement_utc="2026-07-07T00:00:00Z",
        inputs={"vehicle": "levelset_mlx_witness", "term": "deterministic_soft_histogram_surrogate",
                "flag": "--weight-entropy-penalty-lambda",
                "trainer": "experiments/train_levelset_witness_realized_through_R_mlx.py",
                "module": "tac.boundary_math.weight_entropy_penalty_mlx",
                "dsl_lever": "tac.witness_dsl.curriculum_dsl.WeightEntropyPenaltyMLX"},
        predicted_output={"state": "PORT LANDED; no byte/score prediction carried over"},
        empirical_output={
            "state": "NEVER-FIRED (activation ledger); lambda=0 default proven a true no-op "
                     "(branch never constructed); micro-batch twin routed (_once_terms)",
            "owed": "byte-closed n600 A/B (lambda-on vs lambda=0 at equal d_seg/d_pose; real "
                    "quantize_levelset_blob bytes + measured_symbol_entropy_bits_numpy)",
            "note": "SOURCE-INSPECTION row, not a measurement of the lever's effect",
        },
        residual=0.0,
        source_artifact="src/tac/boundary_math/weight_entropy_penalty_mlx.py",
        measurement_method="source_inspection_port_landed",
        provenance=build_provenance_for_research_sidecar(
            sidecar_path="src/tac/boundary_math/weight_entropy_penalty_mlx.py",
            reactivation_criteria="fire the WeightEntropyPenaltyMLX lever as an n600 A/B arm "
                                  "(duty-to-measure); the torch -19.6% must NOT be cited for it",
            measurement_axis="[research-signal]",
            hardware_substrate="m5_max_cpu",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="Ballé rate-in-the-loss weight-entropy lever (torch-measured; MLX-ported, A/B owed)",
        one_line_summary=(
            "Train-time weight-entropy penalty lowers the byte floor: torch MEASURED -16,007 B "
            "(-19.6%) live at lambda50 (EMA-lag caveat); MLX port landed, never-fired, n600 A/B owed."
        ),
        latex_form=(
            r"\mathcal{L} \mathrel{+}= \lambda\cdot\frac{25}{8\,N}\sum_t H_t\,n_t;\ "
            r"b_{arch}\approx\tfrac{1}{8}\sum_t H_t n_t;\ "
            r"\Delta b^{torch,live}_{\lambda 50}{=}{-}16007\,(-19.6\%)"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.weight_entropy_rate_in_loss_20260707:"
            "torch_vehicle_measured_live_delta_bytes"
        ),
        domain_of_validity={
            "vehicle_measured": ["torch_split_by_head_basin (learned-prior term)"],
            "vehicle_ported_unmeasured": ["levelset_mlx_witness (deterministic soft-histogram term)"],
            "measurement_axis": ["contest-CPU advisory"],
            "byte_closed": True,
            "note": "TRAIN-TIME lever (changes dynamics, not bytes directly); the empirical "
                    "bit-spend proof is the paired lambda-on/off A/B on REAL codec bytes "
                    "(Catalog #304); torch numbers do NOT transfer across vehicles/terms",
        },
        units_in={"lambda": "score_units_per_rate_term", "weights": "counted decoder/witness params"},
        units_out={"delta_bytes": "bytes", "delta_H": "bits/weight"},
        empirical_anchors=(anchor_torch_live, anchor_mlx_port),
        predicted_vs_empirical_residual={
            "torch_vehicle_byte_closed_codec_ab": 0.0,
        },
        last_calibration_utc=_UTC_TORCH,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_dsl.curriculum_dsl",
            "experiments/train_levelset_witness_realized_through_R_mlx.py",
        ),
        canonical_producers=(
            "tac.torch_vehicle.weight_entropy_penalty",
            "tac.boundary_math.weight_entropy_penalty_mlx",
        ),
        provenance=build_provenance_for_predicted(
            model_id="weight_entropy_rate_in_loss_lever.v1",
            inputs_sha256="0" * 64,
            measurement_axis="[contest-CPU advisory]",
            hardware_substrate="m5_max_cpu",
        ),
    )


def populate_weight_entropy_rate_in_loss_equation(
    *,
    path=None,
    lock_path=None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """APPEND-ONLY registration of the rate-in-the-loss lever law into the canonical registry
    (latest-row-wins query semantics). Equations leg of DAG FEED-08j; DSL leg =
    ``WeightEntropyPenaltyMLX``; trainer leg = ``--weight-entropy-penalty-lambda`` (MLX port)."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_weight_entropy_rate_in_loss_lever_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id,
        notes="weight_entropy_rate_in_loss_20260707 (equations leg of DAG FEED-08j; the council "
              "draft §22(2) fold: torch-vehicle anchors welded with the borrowed-number firewall; "
              "MLX levelset port never-fired, n600 A/B owed)",
    )
    return eq


__all__ = [
    "EQUATION_ID",
    "build_weight_entropy_rate_in_loss_lever_v1",
    "populate_weight_entropy_rate_in_loss_equation",
    "predict_rate_term",
    "torch_vehicle_measured_live_delta_bytes",
]
