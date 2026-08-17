# SPDX-License-Identifier: MIT
"""CW1: DSL custody of the semantic renderer's curriculum-shape and observation budget.

These four flags produced the whole ``ce1`` result of 2026-08-17 and **none of them was
held by the DSL**.  ``lever_registry.completeness()`` on
``src/tac/pr130_lift/train_semantic_quantized_resumable.py`` reported 34 of 38 flags
UNMAPPED, including ``--ce-fraction`` and ``--softplus-fraction`` -- the exact
config-orphan the triality law names.  Nine retained runs swept three decades of
learning rate while holding a curriculum shape nobody had ever varied, because the shape
lived only in an argparse default.

What is MEASURED (re-derived by ``ddm_cw1`` from the retained ``result.json`` payloads,
never from a memo headline; same init sha, same seed 20260715, same lr 2e-5, same
600 steps, differing ONLY in these two flags):

===================  ==================  ======================  =========================
arm                  curriculum          flips above init @600   aligned LR budget
===================  ==================  ======================  =========================
``A2_repeat``        0.50 / 0.85         +8,654                  26.87%
``CE0``              0.00 / 0.85         +4,852                  52.43%
``EF0``              0.00 / 0.00         **+636**                **61.85%**
===================  ==================  ======================  =========================

13.607x, monotone at every one of the six shared checkpoints, 92.651% of the surcharge
removed.  Extended to 3,000 and 6,000 steps the same shape crosses BELOW the
initialization (-2,286 and -2,620 endpoint flips) where ten prior runs could not.

**The aligned budget is a CLOSED FORM, not a fit.**  ``curriculum_loss`` selects the
phase from ``progress = step / (total_steps - 1)`` and the scheduler is
``CosineAnnealingLR(T_max=steps, eta_min=0.01*lr)``, so every phase's share of the
integrated learning rate is a function of the FRACTIONS ALONE.  Verified numerically at
T = 600 / 3,000 / 30,000: the stock shape spends 81.15 / 81.19 / 81.20% of its LR on the
phase whose gradient sign agrees with the metric only 60.4% of the time
(``cos(sign g) = 0.2087``, rg1b Sec 6.2), and 0.84% on the best-aligned phase (0.6185).
**That is why lengthening the window changed nothing** -- a fraction of the run cannot be
moved by changing the run.

WHAT THESE LEVERS DO NOT CLAIM.  ``expected_flip`` is the best of THREE measured
objectives, not a proven optimum; nothing here says the alignment axis is exhausted.
The ``tau`` anneal inside ``expected_flip`` (0.15 -> 0.05 across the phase) is itself a
fraction-of-run constant with no flag, and at ``softplus_fraction = 0.0`` it now spans
the whole run instead of its last 15% -- the same class of inherited constant, UNSWEPT.
And every number above is the trainer's advisory ``quantized_exact_seg`` on the SEMANTIC
RENDERER (38 tensors, 66,339 params, ``blocks.{0..3}`` + FiLM), which is a DIFFERENT
ARCHITECTURE from the ``hv1`` frontier vehicle (37 tensors, 39,375 params, no FiLM) --
not a different checkpoint.  ``score_claim`` is False on every lever here.
"""

from __future__ import annotations

from tac.witness_dsl.curriculum_dsl import Lever
from tac.witness_dsl.lawref import (
    LADDER_DERIVED_AT_CONFIG,
    LADDER_HARDCODED_WAIVER,
    LADDER_MEASURED_ANCHOR,
    InputRef,
    LawRef,
)

TRAINER_RELPATH = "src/tac/pr130_lift/train_semantic_quantized_resumable.py"

#: Vehicle tag carried on every constant here.  A verdict on this renderer is NOT a
#: verdict on the frontier archive; the two are different architectures.
VEHICLE = "pr130_lift_semantic_renderer_w96_b4"

#: The trainer's own hand-typed defaults, kept ONLY so the notes can price the delta.
TRAINER_DEFAULT_CE_FRACTION = 0.50
TRAINER_DEFAULT_SOFTPLUS_FRACTION = 0.85

#: MEASURED, rg1b Sec 6.2, real gradient path at the shared init: sign-agreement cosine
#: between each phase's gradient and the argmax metric's own gradient.
COS_SIGN_AT_INIT = {"ce": 0.2087, "softplus_margin": 0.5235, "expected_flip": 0.6185}

#: DERIVED closed form (ddm_cw1, verified at T = 600 / 3,000 / 30,000).  Share of the
#: integrated CosineAnnealingLR budget landing in each phase, per curriculum shape.
ALIGNED_LR_BUDGET_STOCK = 0.2687
ALIGNED_LR_BUDGET_EF = 0.6185

#: MEASURED endpoint flips above init at 600 steps, same init/seed/lr, ddm_cw1
#: re-derivation from the retained result.json payloads.
LADDER_FLIPS_AT_600 = {"stock": 8654, "ce0": 4852, "ef0": 636}

#: MEASURED (ddm_cw1, 8 launcher done-receipts, one instrument).  Per-evaluation
#: wall-clock on this trainer at n600 / eval-batch 8, local Metal.
SECONDS_PER_EVAL = 24.758
SECONDS_PER_STEP_EXPECTED_FLIP = 0.19429
FIXED_SECONDS = 126.198

#: MEASURED A/A run-to-run difference at step 600 (A2 vs A2_repeat, same config, same
#: seed 20260715; MPS nondeterminism alone).  The two-run comparison band is this x sqrt(2).
AA_FLOOR_FLIPS = 605


def _ladder_ref(flag: str, value: float) -> LawRef:
    """Custody the curriculum fraction against the MEASURED three-arm ladder."""
    return LawRef(
        equation_id="semantic_curriculum_alignment_ladder_v1",
        inputs={
            "value": InputRef.literal(
                value,
                (
                    "ddm_cw1 re-derivation from retained result.json: A2_repeat +8,654 / "
                    "CE0 +4,852 / EF0 +636 flips above a shared 33,757-flip init at 600 "
                    "steps, same seed 20260715, same lr 2e-5, differing only in "
                    "--ce-fraction/--softplus-fraction; 13.607x, 92.651% of the surcharge "
                    "removed, monotone at all six shared checkpoints"
                ),
                config_tags={"vehicle": VEHICLE, "flag": flag},
            ),
            "aligned_lr_budget": InputRef.literal(
                ALIGNED_LR_BUDGET_EF,
                (
                    "DERIVED closed form: phase shares of the integrated "
                    "CosineAnnealingLR budget depend only on the fractions, so the split "
                    "is scale-invariant (verified at T = 600 / 3,000 / 30,000). Stock "
                    f"{ALIGNED_LR_BUDGET_STOCK:.4f} -> aligned {ALIGNED_LR_BUDGET_EF:.4f}"
                ),
                config_tags={"vehicle": VEHICLE},
            ),
        },
        ladder_class=LADDER_MEASURED_ANCHOR,
    )


def lever_cw1_aligned_objective() -> Lever:
    """Spend the whole LR budget on ``expected_flip``, the best-aligned measured objective.

    Emits ``--ce-fraction 0.0 --softplus-fraction 0.0``.  ``_phase_for_step`` reads both
    as fractions of the run and ``progress < 0.0`` is False at every step, so the run is
    ``expected_flip`` from step 1.  No code change: both flags already existed with no
    validator, which is precisely why the shape was never swept.

    Falsifier for any arm composing this lever: against a step-matched, seed-matched
    stock control, the endpoint must beat it by more than the two-run band
    (605 * sqrt(2) = 856 flips).  It does, by 9.5x, at 600 steps.
    """
    return Lever(
        name="cw1_aligned_objective",
        overrides={"--ce-fraction": 0.0, "--softplus-fraction": 0.0},
        notes=(
            "CW1: 100% expected_flip. The stock shape (ce 0.50 / softplus 0.85) spends "
            f"81.19% of its integrated LR on cos(sign g)={COS_SIGN_AT_INIT['ce']} and "
            f"0.84% on {COS_SIGN_AT_INIT['expected_flip']}; aligned budget "
            f"{ALIGNED_LR_BUDGET_STOCK:.4f} -> {ALIGNED_LR_BUDGET_EF:.4f} (2.30x). The "
            "split is a CLOSED FORM in the fractions alone, hence SCALE-INVARIANT -- "
            "which is why 5x the window changed nothing and no window would. MEASURED "
            f"ladder at 600 steps: stock +{LADDER_FLIPS_AT_600['stock']} / ce0 "
            f"+{LADDER_FLIPS_AT_600['ce0']} / ef0 +{LADDER_FLIPS_AT_600['ef0']} flips "
            "above a shared init. NOT a proven optimum: expected_flip is the best of "
            "three measured objectives, and the tau anneal inside it (0.15->0.05 across "
            "the phase, no flag) is an inherited fraction-of-run constant now stretched "
            "over the whole run -- UNSWEPT. Judge any composing arm on the ENDPOINT "
            f"against a two-run band of {AA_FLOOR_FLIPS * 2 ** 0.5:.0f} flips "
            f"({AA_FLOOR_FLIPS} MEASURED A/A x sqrt(2)). Advisory; score_claim=False."
        ),
        lawrefs={
            "--ce-fraction": _ladder_ref("--ce-fraction", 0.0),
            "--softplus-fraction": _ladder_ref("--softplus-fraction", 0.0),
        },
        constant_manifest={
            "--ce-fraction": {
                "value": 0.0,
                "rung": "MEASURED_ANCHOR (ddm_cw1 three-arm ladder, n600 advisory)",
                "equation_id": "semantic_curriculum_alignment_ladder_v1",
                "trainer_default_superseded": TRAINER_DEFAULT_CE_FRACTION,
                "single_value_owner": "semantic_curriculum_alignment_ladder_v1",
                "vehicle": VEHICLE,
            },
            "--softplus-fraction": {
                "value": 0.0,
                "rung": "MEASURED_ANCHOR (ddm_cw1 three-arm ladder, n600 advisory)",
                "equation_id": "semantic_curriculum_alignment_ladder_v1",
                "trainer_default_superseded": TRAINER_DEFAULT_SOFTPLUS_FRACTION,
                "single_value_owner": "semantic_curriculum_alignment_ladder_v1",
                "vehicle": VEHICLE,
            },
        },
        runtime_receipt_schemas={
            "cw1_aligned_objective_phase_trace": (
                "per-eval trainer telemetry already emitted in result.json::history "
                "('phase' key): asserts every logged step reports phase == "
                "'expected_flip', which is the ON-only proof the fractions took effect"
            ),
        },
        policy_contracts={
            "score_claim": False,
            "adds_new_loss_term": False,
            "default_off_byte_identity": True,
            "vehicle": VEHICLE,
            "is_frontier_vehicle": False,
        },
    )


def lever_cw1_lr_budget(base_lr: float) -> Lever:
    """The learning rate, carried EXPLICITLY as a swept axis with its optimum UNMEASURED.

    This lever exists to stop ``--lr 2e-5`` behaving like a constant.  Three decades of
    learning rate were swept (2e-7 / 2e-6 / 2e-5 / 2e-4) **only under the misaligned
    curriculum**, where the measured response was damage rising as ``lr**0.372``.  The
    aligned curriculum has been measured at EXACTLY ONE learning rate.  2e-5 was never
    chosen FOR this objective; it is inherited from the arm the stock sweep centred on,
    and CONSTANTS-ARE-POISON applies to it until it is re-raced here.

    Why it is load-bearing rather than cosmetic: ``eta_min = 0.01 * lr``, so ``--lr`` sets
    the tail of the schedule as well as its peak.  Both measured aligned windows spent
    over 99% of their integrated LR before their best step (EF3000 at the cap with 0.00%
    left, EF6000 at step 5,200 with 0.64% left), so "the descent decelerated" and "the
    anneal ran out" are not separable on the evidence, and raising ``--lr`` is the only
    zero-code lever that lifts the floor as well as the peak.
    """
    if not (float(base_lr) > 0.0):
        raise ValueError(
            f"lever_cw1_lr_budget: base_lr must be > 0, got {base_lr!r}; a non-positive "
            "peak makes eta_min = 0.01*lr non-positive and the cosine schedule undefined."
        )
    return Lever(
        name=f"cw1_lr_budget_{base_lr:g}",
        overrides={"--lr": float(base_lr)},
        notes=(
            f"CW1: peak lr {base_lr:g} (eta_min = {base_lr * 0.01:g}). Integrated LR over "
            "a cosine run is exactly steps * 0.505 * lr, so lr and steps are two ways to "
            "buy the SAME budget -- which is what makes an lr arm a cheaper substitute "
            "for a longer window at matched budget. The aligned-objective optimum is "
            "UNMEASURED: the only lr sweep on this trainer ran under the stock "
            "curriculum, where higher lr meant more damage (flips ~ lr**0.372). Under an "
            "aligned objective that sign is untested. Report the ENDPOINT, not the best: "
            "the best is a max over the eval samples and carries a selection bias of "
            "roughly 1.7 sd of the tail scatter at 14 evals."
        ),
        lawrefs={
            "--lr": LawRef(
                equation_id="semantic_aligned_lr_optimum_unmeasured_v1",
                inputs={
                    "value": InputRef.literal(
                        float(base_lr),
                        (
                            "SWEPT AXIS, not a derived constant. The lr x curriculum grid "
                            "is measured at four lr under the stock shape and at exactly "
                            "ONE lr (2e-5) under the aligned shape; this lever's value is "
                            "the point being raced, and it is illegal to cite it as an "
                            "optimum until the aligned column is filled."
                        ),
                        config_tags={"vehicle": VEHICLE, "axis": "swept"},
                    )
                },
                ladder_class=LADDER_HARDCODED_WAIVER,
                fallback=float(base_lr),
                fallback_waiver_reason=(
                    "CLASS-4 typed custody. reason: the aligned-objective lr optimum has "
                    "never been measured -- the only sweep ran under the misaligned "
                    "curriculum. owner: ddm_cw1. rederivation_trigger: any arm that "
                    "fills two or more cells of the (lr x aligned-curriculum) column "
                    "supersedes this waiver with a MEASURED_ANCHOR."
                ),
            )
        },
        constant_manifest={
            "--lr": {
                "value": float(base_lr),
                "rung": "HARDCODED_WAIVER (swept axis; optimum UNMEASURED under alignment)",
                "equation_id": "semantic_aligned_lr_optimum_unmeasured_v1",
                "owner": "ddm_cw1",
                "rederivation_trigger": (
                    "two or more measured cells in the (lr x aligned-curriculum) column"
                ),
                "eta_min": float(base_lr) * 0.01,
                "integrated_lr_per_step": 0.505 * float(base_lr),
                "vehicle": VEHICLE,
            }
        },
        runtime_receipt_schemas={
            "cw1_lr_budget_schedule_trace": (
                "per-eval trainer telemetry already emitted in result.json::history "
                "('lr' key): the realized CosineAnnealingLR value, so the fraction of the "
                "integrated LR budget remaining after the best step is recomputable "
                "post-hoc -- the check that separates 'found a floor' from 'anneal ran out'"
            ),
        },
        policy_contracts={
            "score_claim": False,
            "adds_new_loss_term": False,
            "vehicle": VEHICLE,
            "is_frontier_vehicle": False,
            "optimum_measured": False,
        },
    )


def lever_cw1_observation_budget(
    eval_every: int = 250, checkpoint_every: int = 200
) -> Lever:
    """Choose the observation cadence deliberately, at its MEASURED price.

    The cadence is not free and it was never chosen.  MEASURED on this trainer from 8
    launcher done-receipts: ``e = 24.758 s`` per evaluation against
    ``r = 0.19429 s`` per ``expected_flip`` step, so a run at ``--eval-every 25`` spends
    **70.6%** of its wall clock observing itself.

    It is also a BIAS knob, which is the part nobody priced.  Evaluation is
    ``@torch.no_grad()``, restores live weights exactly through ``ema_eval_scope``, and
    consumes no RNG, so cadence cannot perturb the trained trajectory -- **the endpoint
    is cadence-invariant**.  But ``best`` is a minimum over the eval samples: at 14 tail
    evals the pure selection component is about 1.7 sd of the tail scatter, and on
    ``EF6000`` that scatter was 465 flips, so roughly 790 flips of the -2,755 "best"
    is selection over noise rather than descent.  The endpoint (-2,620) carries none of it.

    Defaults deliberately keep ``eval_every != checkpoint_every``.  Two runs whose eval
    and save cadences were equal made the two terms collinear and produced a wall-clock
    law that attributed 53% of its training rate to evaluation for three runs.
    """
    if eval_every < 1 or checkpoint_every < 1:
        raise ValueError(
            "lever_cw1_observation_budget: cadences must be >= 1 "
            f"(got eval_every={eval_every!r}, checkpoint_every={checkpoint_every!r}); "
            "the trainer refuses < 1 and a zero cadence would divide by zero."
        )
    if eval_every == checkpoint_every:
        raise ValueError(
            "lever_cw1_observation_budget: eval_every == checkpoint_every makes the eval "
            "and save terms COLLINEAR in any wall-clock fit. That collinearity is exactly "
            "what let the campaign price a save-dominated model that implied a negative "
            "fixed cost. Choose different cadences."
        )
    return Lever(
        name=f"cw1_observation_budget_e{eval_every}_c{checkpoint_every}",
        overrides={"--eval-every": int(eval_every), "--checkpoint-every": int(checkpoint_every)},
        notes=(
            f"CW1: eval every {eval_every}, checkpoint every {checkpoint_every} "
            "(deliberately unequal, to keep the wall-clock terms separable). MEASURED "
            f"price e={SECONDS_PER_EVAL} s/eval against r={SECONDS_PER_STEP_EXPECTED_FLIP} "
            f"s/step and F={FIXED_SECONDS} s fixed. Cadence is trajectory-NEUTRAL "
            "(no_grad, exact weight restore, no RNG draw), so the ENDPOINT is "
            "cadence-invariant and is the number to report; 'best' is a min over the eval "
            "samples and must be quoted with its eval count. Checkpoints are cheap "
            "(~1.6 MB, ~1.4 s each) and are P0 resume state -- never economise on them."
        ),
        lawrefs={
            "--eval-every": LawRef(
                equation_id="semantic_observation_cadence_price_v1",
                inputs={
                    "value": InputRef.literal(
                        int(eval_every),
                        (
                            "DERIVED at config from the MEASURED cadence law "
                            f"(F={FIXED_SECONDS}, r_expected_flip="
                            f"{SECONDS_PER_STEP_EXPECTED_FLIP}, e={SECONDS_PER_EVAL}; "
                            "8 launcher done-receipts, one instrument, per-curriculum "
                            "rate separated) balanced against the best-of-N selection "
                            "bias measured on the EF6000 tail"
                        ),
                        config_tags={"vehicle": VEHICLE},
                    ),
                    "seconds_per_eval": InputRef.literal(
                        SECONDS_PER_EVAL,
                        "MEASURED: least-squares over 8 detached_local_process_done.v2 "
                        "receipts; residual rms 6.35 s, max 0.95%",
                        config_tags={"vehicle": VEHICLE},
                    ),
                },
                ladder_class=LADDER_DERIVED_AT_CONFIG,
            ),
        },
        constant_manifest={
            "--eval-every": {
                "value": int(eval_every),
                "rung": "DERIVED_AT_CONFIG (measured cadence price x selection-bias budget)",
                "equation_id": "semantic_observation_cadence_price_v1",
                "seconds_per_eval": SECONDS_PER_EVAL,
                "vehicle": VEHICLE,
            },
            "--checkpoint-every": {
                "value": int(checkpoint_every),
                "rung": "DERIVED_AT_CONFIG (P0 resume granularity; save cost ~1.4 s, ~1.6 MB)",
                "equation_id": "semantic_observation_cadence_price_v1",
                "collinearity_guard": "must differ from --eval-every",
                "vehicle": VEHICLE,
            },
        },
        runtime_receipt_schemas={
            "cw1_observation_budget_receipt": (
                "launcher detached_local_process_done.v2 elapsed_s plus "
                "result.json::checkpoints_written and len(history): the three counts a "
                "cadence-aware wall-clock refit needs, non-collinear by construction"
            ),
        },
        policy_contracts={
            "score_claim": False,
            "adds_new_loss_term": False,
            "trajectory_neutral": True,
            "vehicle": VEHICLE,
            "is_frontier_vehicle": False,
        },
    )
