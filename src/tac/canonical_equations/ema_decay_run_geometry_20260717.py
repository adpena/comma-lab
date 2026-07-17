# SPDX-License-Identifier: MIT
"""Canonical equation: EMA decay from RUN GEOMETRY (ARM-C p0_ema_calibration; SPEC_v10 §13.3).

The law
-------
A constant-decay EMA over ``U`` optimizer updates obeys three exact geometric identities:

    seed_fraction   eps = d**U            (weight the initial shadow seed retains at the end)
    warmup_updates  W   = 2/(1-d)         (the registered two-time-constant warmup,
                                           ``tac.confound_observability.ema_warmup_updates``)
    warmup_fraction phi = W/U = 2/((1-d)*U)

Inverting for ``d`` gives the CALIBRATED decay from run geometry:

    d = eps**(1/U)          (pin the terminal seed fraction eps)
    d = 1 - 2/(phi*U)       (pin the run fraction phi at which warmup completes)

MEASURED basis (SPEC_v10 §13.3; live c2 run ``levelset_n600_witness_20260717T113932Z``):
the incumbent ``--ema-decay 0.997`` is 0.997/update at 1 update/epoch (full-batch accum)
=> tau ~= 333 EPOCHS; declared warmup 2/(1-d) = 667 updates ~= ep1318 on a 1400-ep run —
the shadow spends the WHOLE run inside warmup; the warm-start seed (ep651 checkpoint)
still holds d**149 = 0.6391 (~64%) of the shadow at ep800 and d**749 = 0.1054 (~10.5%)
at the ep1400 terminal; EMA-live verdict gap -0.00095 @ ep775 (SPEC-recorded). The 0.997
provenance (Quantizr per-step minibatch EMA) does NOT transfer to the deterministic
full-batch regime: the noise-averaging rationale vanishes at 1 update/epoch. This law
replaces the inherited constant with a DERIVED value; the incumbent remains available as
``historical_non_authorizing`` context (CLAUDE.md EMA non-negotiable mechanics are SOUND —
this is calibration, not a bug).

Evaluator: ``tac.canonical_equations.evaluators.eval_ema_decay_run_geometry``
(LawRef-executable, ``ema_decay_run_geometry_v1``). DSL consumer:
``tac.witness_dsl.curriculum_dsl.EmaDecayCalibrated`` (emits a LawRef-resolved
``--ema-decay``; the default trainer path is byte-identical when the Lever is not composed).

Axis: derivation + config/source inspection; research_only; NO score claim; the d_seg/d_pose
effect of a calibrated decay is RUN-GATED (the arm-C shadow-vs-live byte-close comparator +
the per-stage A/B decide it empirically). Pointer UNMOVED (means).
"""
from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.canonical_equations.evaluators import eval_ema_decay_run_geometry
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "ema_decay_run_geometry_v1"

#: Live c2 run whose telemetry is the measured basis (READ-ONLY custody).
LIVE_RUN_DIR = "experiments/results/levelset_n600_witness_20260717T113932Z"
#: The executable warmup registrar this law extends (W = 2/(1-d)).
WARMUP_HELPER = "src/tac/confound_observability.py"

#: MEASURED/DERIVED incumbent numbers (exact closed forms of d=0.997 over the c2 geometry).
INCUMBENT_DECAY = 0.997
C2_WARM_SEED_EPOCH = 651          # weights-only warm start (the shadow seed epoch)
C2_RUN_END_EPOCH = 1400
C2_EP800_UPDATES = 800 - C2_WARM_SEED_EPOCH        # 149 updates after the seed
C2_TERMINAL_UPDATES = C2_RUN_END_EPOCH - C2_WARM_SEED_EPOCH   # 749


def build_ema_decay_run_geometry_v1() -> CanonicalEquation:
    provenance = build_provenance_for_research_sidecar(
        LIVE_RUN_DIR,
        reactivation_criteria=(
            "re-derive if the accum regime changes (updates/epoch != 1), if the EMA update law "
            "changes (non-constant decay / Polyak), or when the shadow-vs-live byte-close A/B "
            "lands a measured verdict on the calibrated decay"
        ),
        measurement_axis="[derivation + config/source inspection; macOS-MLX telemetry basis]",
        hardware_substrate="n/a (closed-form law)",
        captured_at_utc="2026-07-17T20:30:00Z",
    )
    anchors = (
        EmpiricalAnchor(
            anchor_id="ema_decay_run_geometry_warmup_667_20260717",
            measurement_utc="2026-07-17T20:30:00Z",
            inputs={"ema_decay": INCUMBENT_DECAY, "updates_per_run": C2_TERMINAL_UPDATES,
                    "warmup_helper": WARMUP_HELPER,
                    "launch_flag": f"{LIVE_RUN_DIR}/launch.sh --ema-decay 0.997"},
            predicted_output={"warmup_updates": 667,
                              "law": "W = ceil(2/(1-d)); phi = W/U"},
            empirical_output={
                "warmup_updates_registered": 667,   # ema_warmup_updates(0.997) == ceil(2/0.003)
                "warmup_completes_epoch": C2_WARM_SEED_EPOCH + 667,   # 1318 of 1400
                "warmup_fraction_of_warm_window": round(667 / C2_TERMINAL_UPDATES, 4),  # 0.8905
            },
            residual=0.0,
            source_artifact=WARMUP_HELPER,
            measurement_method=(
                "executable cross-check: tac.confound_observability.ema_warmup_updates(0.997) "
                "== 667; run flag --ema-decay 0.997 read from the live run's launch.sh; "
                "warmup completion epoch = seed epoch + W"
            ),
            provenance=provenance,
            empirical_verification_status="VERIFIED_VIA_SOURCE_INSPECTION",
        ),
        EmpiricalAnchor(
            anchor_id="ema_decay_run_geometry_seed_fraction_20260717",
            measurement_utc="2026-07-17T20:30:00Z",
            inputs={"ema_decay": INCUMBENT_DECAY,
                    "updates_ep800": C2_EP800_UPDATES, "updates_terminal": C2_TERMINAL_UPDATES},
            predicted_output={"law": "seed_fraction = d**U",
                              "spec_recorded": "~64% warm-start seed @ep800 (SPEC_v10 §13.3)"},
            empirical_output={
                "seed_fraction_ep800": round(INCUMBENT_DECAY ** C2_EP800_UPDATES, 4),   # 0.6391
                "seed_fraction_terminal": round(INCUMBENT_DECAY ** C2_TERMINAL_UPDATES, 4),  # 0.1054
                "ema_minus_live_dseg_ep775_spec_recorded": -0.00095,
            },
            residual=abs(round(INCUMBENT_DECAY ** C2_EP800_UPDATES, 4) - 0.64),
            source_artifact=LIVE_RUN_DIR,
            measurement_method=(
                "exact closed form d**U evaluated at the c2 warm-run geometry; cross-checked "
                "against the SPEC_v10 §13.3 recorded ~64% @ep800 and the recorded EMA-live "
                "verdict gap -0.00095 @ep775 (telemetry-recorded, advisory axis)"
            ),
            provenance=provenance,
            empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="EMA decay from run geometry (seed-fraction / warmup-fraction closed forms)",
        one_line_summary=(
            "d = eps**(1/U) (pin terminal seed fraction) or d = 1 - 2/(phi*U) (pin warmup "
            "completion fraction); the Quantizr 0.997/update provenance does not transfer to "
            "the 1-update/epoch full-batch regime"
        ),
        latex_form=(
            r"\varepsilon = d^{U},\quad W = \frac{2}{1-d},\quad \varphi = \frac{W}{U}"
            r"\;\Longrightarrow\; d = \varepsilon^{1/U} \;=\; 1 - \frac{2}{\varphi U}"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.evaluators:eval_ema_decay_run_geometry"
        ),
        domain_of_validity={
            "regime": "constant-decay EMA, deterministic full-batch training "
                      "(updates_per_run counted in actual optimizer updates)",
            "vehicle": "level-set witness trainer (MlxEMA; 1 update/epoch under full accum)",
            "verdict_scope": "FORMULATION — value-provenance law; the d_seg effect of a "
                             "calibrated decay is RUN-GATED (shadow-vs-live byte-close A/B "
                             "+ per-stage A/B decide empirically)",
            "research_only": True,
            "score_claim": False,
            "promotion_eligible": False,
        },
        units_in={"mode": "derivation selector", "updates_per_run": "optimizer updates",
                  "target_seed_fraction": "dimensionless (0,1)",
                  "warmup_fraction": "fraction of run", "ema_decay": "per-update decay"},
        units_out={"value": "per-update decay | fraction (mode-dependent)"},
        empirical_anchors=anchors,
        predicted_vs_empirical_residual={"warmup_updates": 0.0},
        last_calibration_utc="2026-07-17T20:30:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_dsl.curriculum_dsl.EmaDecayCalibrated (LawRef-resolved --ema-decay)",
            "tools/compare_shadow_vs_live_byte_close.py (the empirical decider for the EMA "
            "question, SPEC_v10 §13.5)",
            "duty-to-measure queue (ema_decay_finisher registration; costate SENSE layer)",
        ),
        canonical_producers=(
            WARMUP_HELPER,
            f"{LIVE_RUN_DIR}/launch.sh",
            "experiments/train_levelset_witness_realized_through_R_mlx.py (MlxEMA + "
            "--ema-decay/--ema-decay-finisher argparse)",
        ),
        provenance=provenance,
    )


__all__ = [
    "EQUATION_ID",
    "build_ema_decay_run_geometry_v1",
    "eval_ema_decay_run_geometry",
]
