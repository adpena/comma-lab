# SPDX-License-Identifier: MIT
"""Adam second-moment variance warmup length — the DERIVED LR-rewarmup window law.

Registered 2026-07-17 for the ``p0_resume_warmup_geometry_20260717`` set (task #518, item 2).

THE LAW (β₂-derived, not hand-typed):

    warmup_steps  ≈ c / (1 - β₂)          (optimizer steps; c ≈ 2 default)
    warmup_epochs = ceil(warmup_steps / steps_per_epoch)

Rationale (INFERRED_FROM_DOMAIN_LITERATURE — the RAdam variance-rectification argument,
Liu et al. 2019, arXiv:1908.03265): with fresh/reset moments, the variance of Adam's adaptive
learning rate is divergently large until the second-moment estimate ``v`` has accumulated
O(1/(1-β₂)) observations — early steps are quasi-isotropic (v≈0 → the preconditioner is the ε
floor), so full-LR steps push un-shaped energy into separatrix-normal directions (the rank-4
head law ``segnet_head_rank4_linear_flipdist_v1``: flip distance d = |m|/‖Δw‖). ``c=1`` is the
bare memory bound (the ALREADY-REGISTERED sister law ``rewarmup_beta2_memory_window_v1``:
rewarmup_steps ≥ 1/(1-β₂)); ``c≈2`` is the conservative default multiple this law adds so the
ramp SPANS the window rather than merely reaching it (RAdam's ρ-threshold sits at a small
multiple of the memory scale).

MEASURED context (c2 fork, run levelset_n600_witness_20260717T113932Z): steps_per_epoch = 75
(600 pairs / accum 8), β₂ = 0.999 → memory 1000 steps ≈ 13.4 ep; c=2 → 2000 steps → 27 ep.
The config-of-record constant 8 ep (600 steps) sits UNDER the c=1 bound — the same deficit the
sister law's anchor already flagged ("cert8_satisfies": false). The 8 stays the recorded
CONFIG-rung fallback on the value-provenance ladder (DERIVED > CONFIG); this law is the DERIVED
rung the ``ResumeLRWarmup`` DSL lever resolves through.

Verdict scope: FORMULATION — a config-derivation law (value provenance), PROVISIONAL-PENDING-
VERIFICATION like its sister (the isolating A/B — 8 vs derived epochs at fixed β₂ — is the owed
anchor). Not a score claim; the pointer moves only through a byte-closed n600
``upstream/evaluate.py`` row.
"""
from __future__ import annotations

import math

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "adam_v_variance_warmup_length_v1"

#: default conservative multiple of the second-moment memory 1/(1-beta2) (RAdam-motivated).
DEFAULT_C = 2.0

#: the c2 run's MEASURED optimizer-steps-per-epoch (600 pairs / --accum-pairs 8; the same value
#: the sister law's run-2 anchor records as "20 ep x 75 steps/ep").
C2_STEPS_PER_EPOCH = 75

#: the config-of-record constant (mod32cap/c2 launch.sh: --stage-transition-rewarmup-epochs 8) —
#: preserved as the CONFIG-rung fallback on the value-provenance ladder, NOT the derived value.
CONFIG_OF_RECORD_REWARMUP_EPOCHS = 8


def adam_v_variance_warmup_epochs(
    beta2: float, steps_per_epoch: int, c: float = DEFAULT_C
) -> int:
    """The law's pure callable: ceil(c/(1-beta2) / steps_per_epoch) epochs.

    ``c=1`` reproduces the sister bound ``min_rewarmup_epochs`` (rewarmup_beta2_memory_window_v1)
    exactly; ``c=2`` (default) spans twice the second-moment memory (RAdam rationale).
    """
    if not 0.0 < float(beta2) < 1.0:
        raise ValueError(f"beta2={beta2} must be in (0, 1)")
    if int(steps_per_epoch) <= 0:
        raise ValueError(f"steps_per_epoch={steps_per_epoch} must be > 0")
    if not float(c) > 0.0:
        raise ValueError(f"c={c} must be > 0")
    return math.ceil((float(c) / (1.0 - float(beta2))) / float(steps_per_epoch))


def build_adam_v_variance_warmup_length_v1() -> CanonicalEquation:
    provenance = build_provenance_for_research_sidecar(
        ".omx/research/p0_resume_warmup_geometry_build_20260717.md",
        reactivation_criteria=(
            "PROVISIONAL like the sister rewarmup_beta2_memory_window_v1: the isolating A/B "
            "(config 8 ep vs derived c/(1-beta2) epochs at fixed beta2=0.999, steps/ep=75) "
            "converts INFERRED to VERIFIED and calibrates c"
        ),
        measurement_axis="[macOS-CPU advisory]",
        hardware_substrate="apple_m5_max_cpu_mlx",
        captured_at_utc="2026-07-17T13:40:00Z",
    )
    anchors = (
        EmpiricalAnchor(
            anchor_id="adam_v_variance_warmup_c2_fork_20260717",
            measurement_utc="2026-07-17T13:40:00Z",
            inputs={"beta2": 0.999, "steps_per_epoch": C2_STEPS_PER_EPOCH, "c": DEFAULT_C,
                    "config_of_record_epochs": CONFIG_OF_RECORD_REWARMUP_EPOCHS,
                    "incident": "c2 warm-start fork (run 20260717T113932Z) entered ep651 at "
                                "full scheduled LR on fresh AdamW moments (no rewarmup "
                                "boundary registered)"},
            predicted_output={"warmup_epochs_c2": 27,
                              "warmup_epochs_c1_matches_sister_bound": 14},
            empirical_output={"pose_gate_sigma_min_transient": "0.0025 -> 0.0084 (3.4x), "
                                                              "settling by ep674 (~23 ep)",
                              "transient_epochs_observed": 23,
                              "config_8ep_under_c1_bound": True},
            residual=4.0,  # |derived 27 - observed transient ~23| epochs (coarse; A/B owed)
            source_artifact="experiments/results/levelset_n600_witness_20260717T113932Z "
                            "(read-only reference; measured on main pre-dispatch)",
            measurement_method=(
                "arithmetic against the verified c2 config (beta2/accum/pairs) + the run's "
                "MEASURED pose-gate sigma_min transient window; no new runs"
            ),
            provenance=provenance,
            empirical_verification_status="INFERRED_FROM_DOMAIN_LITERATURE",
        ),
    )
    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=("Adam second-moment variance warmup length: warmup_steps ~= c/(1-beta2) "
              "(c~=2; c=1 == the sister memory bound)"),
        one_line_summary=(
            "post-reset/fork LR-rewarmup length DERIVED from beta2: ceil(c/(1-beta2)/steps_per_"
            "epoch) epochs (RAdam variance-rectification rationale); c2: 27 ep vs config 8"
        ),
        latex_form=r"T_{\mathrm{rw}} = \left\lceil \frac{c/(1-\beta_2)}{S_{ep}} \right\rceil,\ c \approx 2",
        python_callable_module_path=(
            "tac.canonical_equations.adam_v_variance_warmup_20260717:"
            "adam_v_variance_warmup_epochs"
        ),
        domain_of_validity={
            "lever": "--stage-transition-rewarmup-epochs (via the ResumeLRWarmup DSL lever)",
            "vehicle": ["any AdamW trainer with fresh/reset second moments at a boundary "
                        "(warm-start fork, --stage-transition-reset-moments, pose-engage)"],
            "measurement_axis": ["macOS-CPU advisory"],
            "note": "PROVISIONAL-PENDING-VERIFICATION (mirrors the sister law): literature-"
                    "derived + consistent with the measured c2 transient (~23 ep vs derived 27);"
                    " no isolated A/B has confirmed c",
            "verdict_scope": "FORMULATION — config-derivation law (value provenance), not a "
                             "score claim",
            "research_only": True,
            "score_claim": False,
            "promotion_eligible": False,
        },
        units_in={"beta2": "dimensionless_decay", "steps_per_epoch": "optimizer_steps_per_epoch",
                  "c": "dimensionless_multiple_of_memory"},
        units_out={"warmup_epochs": "epochs"},
        empirical_anchors=anchors,
        predicted_vs_empirical_residual={"derived_vs_observed_transient_epochs": 4.0},
        last_calibration_utc="2026-07-17T13:40:00Z",
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_dsl.curriculum_dsl (ResumeLRWarmup lever LawRef)",
            "experiments/train_levelset_witness_realized_through_R_mlx.py "
            "(--stage-transition-rewarmup-epochs consumer via _stage_rewarmup_factor)",
        ),
        canonical_producers=(
            ".omx/research/p0_resume_warmup_geometry_build_20260717.md",
            "tac.canonical_equations.curriculum_derivation_laws_20260705:min_rewarmup_epochs "
            "(the c=1 sister bound this law generalizes)",
        ),
        provenance=provenance,
    )


def populate_adam_v_variance_warmup_length_equation(
    *, path=None, lock_path=None, agent: str | None = None, subagent_id: str | None = None
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration (EQUATIONS leg of p0_resume_warmup_geometry item 2)."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_adam_v_variance_warmup_length_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id
    )
    return eq


__all__ = [
    "C2_STEPS_PER_EPOCH",
    "CONFIG_OF_RECORD_REWARMUP_EPOCHS",
    "DEFAULT_C",
    "EQUATION_ID",
    "adam_v_variance_warmup_epochs",
    "build_adam_v_variance_warmup_length_v1",
    "populate_adam_v_variance_warmup_length_equation",
]
