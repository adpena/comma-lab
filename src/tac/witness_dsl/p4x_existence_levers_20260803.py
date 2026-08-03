# SPDX-License-Identifier: MIT
"""ddm_p4x (#920) — the LANE EXISTENCE primitive as real DSL ``Lever`` factories.

The mechanism lives in ``tac.optimization.existence_hinge`` and is wired into the LIVE
TR1 vehicle (``experiments/train_tr1_partition_renderer_mlx.py``).  These factories are
how it becomes reachable, raceable and duty-to-measure tracked; a lever that is not a
``Lever`` factory is orphaned by definition.

**What this force is, in one line.**  A SEPARATE loss term at COMPONENT granularity:
``s(c) = logsumexp_beta(live margin over GT component c)`` -> the component's WITNESS
pixel; ``loss = mean_c w_c * relu(target - s(c))``.

**Why it is not another per-pixel weight, which is the whole design.**  ``cg1r``
(``ee848e88cd``) MEASURED realized per-flip GT-margin depth as direction-SYMMETRIC on all
nine class edges (Road<->Lane 1.074x) while the COUNT asymmetry runs to 15.88x.  The
lane-erasure discount is therefore VOLUMETRIC/verb-level, not per-flip pricing: a Lane
word dies at ~2.5 px of depth because Lane has no interior.  Every existing lane_guard
mechanism (``lambda_lane``, born-mask, margin-floor) is an ADDITIVE per-pixel addend in
``seg_pixel_w``, which is exactly why the force ledger records ``protection=ABSENT`` for
the ANNIHILATE verb specifically.  A per-pixel lever aimed at an already-symmetric
quantity is expected to measure NULL on this channel; component-level is the shape that
can see a whole word being lost.

**Prior art on the ANCESTOR vehicle, and why these are NOT a port** (this distinction is
load-bearing per the L18 ancestor rule).  The island-protection family
(``island_seed_birth`` = ``--seed-islands`` + ``--witness-alone-island-loss``,
``SeedIslandEased`` #323, ``AmplifyIsland``) attacks the same debt and MEASURED a real
n600 effect (``experiments/results/island_survival_n600.log``: Lane survival 0.5646 ->
0.9304, ``seed_birth_gain`` 0.3658; Movable 0.9102 -> 0.9773, gain 0.0671).  Two facts
keep that from being importable here:

  1. **Those flags do not exist on TR1.**  VERIFIED by grep: ``--seed-islands``,
     ``--witness-alone-island-loss`` and ``--amplify-weight`` each occur ZERO times in
     the TR1 trainer and once in the RETIRED levelset trainer.  There is nothing to
     rename; a TR1 seed/amplify path would have to be BUILT (named as OWED below).
  2. **Its numbers are ancestor-vehicle numbers.**  The 0.3658 survival gain was measured
     against a SIMULATED erasure (``erase_factor 4``), not against this decode's realized
     annihilation, and on a different vehicle.  It is cited as a DIRECTIONAL prior that
     the debt is attackable, never as a predicted effect size here.

What DOES transfer is the geometric split, and it is a genuine independent convergence:
#323 eases births *"movable via SDF forward-Euler DILATION ... + lane via openpilot
VP-TANGENT along-tangent widening (manifold-preserving; isotropic-of-a-curve is the
NO-GO)"*.  This module's ``BIRTH_MATRIX`` reached the same partition from the gt2 verb
masses alone (Lane has no interior -> uniform/area-blind; Movable does, GOUGE 16,718 px
-> sqrt_area).  Two independent instruments agreeing on the per-class geometry is
evidence about the geometry, not about either implementation.

**Registry contract.**  Every factory here is default-OFF and score-affecting, so each is
a duty-to-measure row the moment this module exists: ``package_known_levers()`` AST-scans
the package, so these enter ``never_fired()`` / ``duty_to_measure()`` automatically.
``TRAINER_RELPATH`` files them under the vehicle we ship (ddm_lr2 §1: an undeclared module
silently defaults to the RETIRED trainer, and a queue cannot drain a lever filed under the
wrong vehicle).

Pointer honesty: ``0.1910828242`` [contest-CPU] UNMOVED.  No arm here has been raced
against the scorer, so no row claims a d_seg effect.  ``score_claim=False``.
"""

from __future__ import annotations

from tac.optimization.existence_hinge import (
    BIRTH_MATRIX,
    CONNECTIVITY_4,
    CONNECTIVITY_8,
    LANE,
    MOVABLE,
    WEIGHT_POLICIES,
    annihilate_ceiling_s,
    protected_ceiling_s,
)
from tac.witness_dsl.curriculum_dsl import Lever

#: ddm_lr2 §1 — declare the vehicle these levers target, or the registry files them under
#: the retired trainer by silent default and no TR1-scoped query can surface them.
TRAINER_RELPATH = "experiments/train_tr1_partition_renderer_mlx.py"

#: The gap denominator every S figure below is quoted against (ddm_p4x, 2026-08-03).
GAP_DENOMINATOR = 0.6189279

#: Scale note, stated once and referenced by every factory rather than re-guessed.
#: The term is ``mean_c relu(target - s(c))`` in MARGIN units.  At init most protected
#: components are being annihilated (Lane 54.38% of words at 8-conn), so ``s(c) < 0`` and
#: the term starts at O(|margin|) ~ O(1-10).  Meanwhile the seg leg contributes
#: ``w_seg * seg_l`` ~ ``100 * O(0.01)`` = O(1).  So an existence weight of 1.0 would make
#: this force COMPARABLE TO OR LARGER THAN the entire seg objective on step 0 -- which is
#: not obviously wrong, but it is certainly not a neutral default.  The race therefore
#: starts an order of magnitude below that and sweeps UP.  This is arithmetic about the
#: term's units, not a measured optimum, and it is labelled as such in every manifest.
EXISTENCE_WEIGHT_RACE_START = 0.1


def _manifest(flag: str, value, rung: str, provenance: str) -> dict:
    return {flag: {"value": value, "rung": rung, "provenance": provenance}}


def lever_lane_existence_hinge(weight: float = EXISTENCE_WEIGHT_RACE_START) -> Lever:
    """Existence protection for LANE ONLY — the ranked head of the protection debt.

    Lane is the single largest protection debt in the force ledger
    (``tr1.lane.annihilate``, ``protection=ABSENT``, magnitude 0.1575 S) and the only
    class whose words are annihilated at a majority rate.

    **Addressable mass, with its denominator carried.**  This term targets the ANNIHILATE
    channel SPECIFICALLY -- not Lane's whole 0.1575 S debt, 73.0% of whose flip PIXELS are
    ERODE on SURVIVING components (the rim guard's instrument, untouched here).  MEASURED
    ceiling at 8-connectivity: 0.037276 S = 6.02% of the gap.  That is a CEILING at 100%
    capture, which will not happen; it bounds the row rather than predicting it.

    Falsifier (pre-registered, and stated in relative units so it cannot drift with the
    baseline): a matched-epoch, matched-bytes, matched-base A/B shows realized n600 d_seg
    no better than the weight=0 control across the swept weight ladder => component-level
    existence protection does not bind on THIS vehicle and the row closes at INSTANCE for
    the Lane instantiation (never for the existence FAMILY, and never for Movable, whose
    geometry differs and which is a separate arm below).

    Typed outcome on a null, per the realizability doctrine: NOT ``ROW_DEAD`` but
    ``DEBT_NAMED(stage=<where the word is lost>, cure=<what would have to carry it>)``.
    """
    if float(weight) <= 0.0:
        raise ValueError("weight must be > 0 to arm the lever (0.0 is the OFF control, "
                         "which is the trainer default -- do not build a Lever for it)")
    return Lever(
        name=f"tr1_existence_lane_w{weight:g}",
        overrides={"--existence-hinge-weight": str(weight),
                   "--existence-hinge-classes": "lane"},
        notes="p4x #920 COMPONENT-level existence hinge, Lane only. Separate loss term "
              "(NOT a seg_pixel_w addend) per the cg1r volumetric law. Ceiling "
              f"{annihilate_ceiling_s(LANE, CONNECTIVITY_8):.6f} S = "
              f"{100 * annihilate_ceiling_s(LANE, CONNECTIVITY_8) / GAP_DENOMINATOR:.2f}% "
              "of gap at 8-conn. Default-OFF => byte-identical.",
        constant_manifest={
            **_manifest("--existence-hinge-weight", float(weight),
                        "RACED-NOT-ASSERTED (no measured optimum on TR1)",
                        "race start derived from the term's UNITS, not from a measurement: "
                        "the hinge is O(1-10) in margin units at init while w_seg*seg_l is "
                        "O(1), so 1.0 would let existence dominate seg on step 0. Start 0.1 "
                        "and sweep UP. The own-optimum law requires sweeping before any "
                        "adopt/kill verdict is admissible."),
            **_manifest("--existence-hinge-classes", "lane", "MEASURED (gt2 verb masses)",
                        "Lane is the top-ranked ANNIHILATE debt: 54.38% of its words "
                        "annihilated at 8-conn (7,789 of 14,323), vs Road 5.45% and "
                        "MyCar 0.00%."),
        })


def lever_existence_birth_matrix(weight: float = EXISTENCE_WEIGHT_RACE_START) -> Lever:
    """The full PER-CLASS BIRTH MATRIX: ONE mechanism, two class geometries.

    Lane and Movable are the only two classes with a materially non-zero word-annihilation
    rate (54.38% and 16.20% at 8-conn).  They receive the SAME mechanism instantiated on
    their own measured geometry:

      * **Lane** -- 75.04% of its GT pixels sit at depth<=1 and GOUGE is 2,926 px vs ERODE
        135,683 px: no interior.  Existence rides the witness pixel alone, and ``w_c`` is
        ``uniform`` (area-blind), which is the volumetric law applied literally.
      * **Movable** -- the ONLY class where GOUGE (16,718 px) is a large fraction of ERODE
        (53,940 px) = 31.0%: it loses INTERIOR, so it is a blob with something inside to
        keep.  ``w_c`` is ``sqrt_area``, the compromise between treating a 22.7 px word as
        a 4.9 px one and re-importing full per-pixel pricing.

    Independently corroborated: #323 ``SeedIslandEased`` splits the same two classes the
    same way (Movable SDF dilation vs Lane VP-tangent widening) on the ancestor vehicle,
    and ``birth_completion`` telemetry records ``classes: [1, 3]`` -- Lane and Movable.
    Three surfaces, one partition, reached independently.

    Ceiling at 8-conn: 0.044175 S = 7.14% of gap (Lane 0.037276 + Movable 0.006900).

    Falsifier (pre-registered): matched A/B shows no realized n600 d_seg improvement over
    the weight=0 control AND the per-class telemetry shows ``at_risk`` counts unchanged =>
    the term is not reaching the words it names, which is an IMPLEMENTATION verdict
    (wiring/scale), not a verdict on existence protection. Only an A/B where ``at_risk``
    demonstrably falls while d_seg does not can close the mechanism at INSTANCE.
    """
    if float(weight) <= 0.0:
        raise ValueError("weight must be > 0 to arm the lever (0.0 is the OFF control)")
    ceiling = protected_ceiling_s((LANE, MOVABLE), CONNECTIVITY_8)
    return Lever(
        name=f"tr1_existence_birthmatrix_w{weight:g}",
        overrides={"--existence-hinge-weight": str(weight),
                   "--existence-hinge-classes": "lane,movable"},
        notes="p4x #920 per-class BIRTH MATRIX (Lane uniform / Movable sqrt_area, each "
              f"from its own measured geometry). Ceiling {ceiling:.6f} S = "
              f"{100 * ceiling / GAP_DENOMINATOR:.2f}% of gap at 8-conn. Default-OFF.",
        constant_manifest={
            **_manifest("--existence-hinge-weight", float(weight),
                        "RACED-NOT-ASSERTED", "see lever_lane_existence_hinge: the start "
                        "is derived from the term's units, not measured."),
            **_manifest("--existence-hinge-classes", "lane,movable",
                        "MEASURED (gt2 verb masses; corroborated by #323 + birth_completion)",
                        "the only two classes with a materially non-zero word-annihilation "
                        "rate. MyCar annihilates 0 words in 600 frames, so including it "
                        "could only add gradient noise; the exclusion is a measurement."),
        })


def lever_existence_grammar(connectivity: int = CONNECTIVITY_8,
                            weight: float = EXISTENCE_WEIGHT_RACE_START) -> Lever:
    """RACE the component GRAMMAR: 8-connected (receiver-physical) vs 4-connected (gt2's).

    ddm_p4x MEASURED that gt2's published word grammar is **4-CONNECTED** -- reproduced
    EXACTLY on all eight controls (Lane 16,581 components / 9,655 annihilated / 47,226 px;
    Movable 2,207 / 361 / 8,180) and NOT under 8-connectivity (Lane 14,323 = 0.864x), with
    GT pixel totals matching 1.0000 so the corpus is identical and only the labelling rule
    differs.  For Lane the two grammars differ by 13.6%: 2,258 diagonal joins.

    This is a real fork, not a formatting choice:

      * **8-conn** is what the RECEIVER can represent (Rosenfeld: a seed that does not
        respect 8-connectivity is deleted by the measured consolidation; gt2 measured a
        FRAGMENT negative on 4 of 5 classes).  Defending a word the receiver will delete
        is wasted gradient.
      * **4-conn** is the grammar every published per-word rate is denominated in.  An A/B
        that wants to speak in gt2's units must select it EXPLICITLY.

    The S-arithmetic is grammar-INVARIANT per pixel (0.044175 S at 8-conn vs 0.046968 at
    4-conn -- the 0.002793 difference does not vanish, it MOVES to the ERODE/GOUGE channel
    where merged words retain a surviving fragment).  Per-WORD rates are NOT invariant and
    must never be quoted across grammars.

    Falsifier (pre-registered): matched A/B across the two grammars shows realized d_seg
    within the seed noise floor => the receiver-consolidation argument does not bind at
    this operating point, and the cheaper/likelier-comparable 4-conn grammar should be
    preferred for interpretability against gt2's published rates.

    Reproduce the grammar measurement: ``tools/ddm_p4x_connectivity_control.py``.
    """
    if connectivity not in (CONNECTIVITY_4, CONNECTIVITY_8):
        raise ValueError(f"connectivity must be {CONNECTIVITY_4} or {CONNECTIVITY_8}")
    if float(weight) <= 0.0:
        raise ValueError("weight must be > 0 to arm the lever (0.0 is the OFF control)")
    return Lever(
        name=f"tr1_existence_conn{connectivity}_w{weight:g}",
        overrides={"--existence-hinge-weight": str(weight),
                   "--existence-hinge-connectivity": connectivity},
        notes=f"p4x #920 component-grammar race arm ({connectivity}-connected). "
              "gt2's published per-word rates are 4-conn (p4x MEASURED, 8/8 exact "
              "controls); the receiver argues 8-conn. Ceiling "
              f"{protected_ceiling_s((LANE, MOVABLE), connectivity):.6f} S.",
        constant_manifest={
            **_manifest("--existence-hinge-connectivity", int(connectivity),
                        "MEASURED (p4x connectivity control, 8/8 exact vs gt2)",
                        "4-conn reproduces gt2 EXACTLY on components/annihilated/px for "
                        "both classes; 8-conn is the Rosenfeld receiver-consolidation "
                        "constraint. Named explicitly because per-WORD capture fractions "
                        "are meaningless across grammars."),
        })


def lever_existence_weight_policy(policy: str,
                                  weight: float = EXISTENCE_WEIGHT_RACE_START) -> Lever:
    """RACE the per-component weight ``w_c``: uniform vs sqrt_area vs area.

    This is the knob that decides how much of the per-pixel pricing the term re-imports,
    so it is the most direct test of the volumetric law itself:

      * ``uniform``   -- pure existence; a 1-px word counts like a 40-px word.
      * ``sqrt_area`` -- partial area sensitivity.
      * ``area``      -- full area weighting, i.e. this term becomes a (component-gated)
        per-pixel objective again.

    ``area`` is deliberately included as the ADVERSARIAL arm: if it wins, the volumetric
    reading is wrong at this operating point and the whole component-level premise needs
    re-examination -- which is a result worth having and is why the arm exists rather than
    being assumed away.

    Falsifier (pre-registered): ``area`` matching or beating ``uniform`` on realized n600
    d_seg falsifies the volumetric premise AS APPLIED to the training force (never the
    cg1r depth-symmetry measurement itself, which is independently established).
    """
    if policy not in WEIGHT_POLICIES or policy == "":
        raise ValueError(f"policy must be one of {[p for p in WEIGHT_POLICIES]}")
    if float(weight) <= 0.0:
        raise ValueError("weight must be > 0 to arm the lever (0.0 is the OFF control)")
    return Lever(
        name=f"tr1_existence_wpolicy_{policy}_w{weight:g}",
        overrides={"--existence-hinge-weight": str(weight),
                   "--existence-hinge-weight-policy": policy},
        notes=f"p4x #920 w_c policy race arm ({policy}). 'area' is the ADVERSARIAL arm: "
              "if it wins, the volumetric premise is wrong for the training force. "
              "Per-class defaults are uniform (Lane) / sqrt_area (Movable).",
        constant_manifest={
            **_manifest("--existence-hinge-weight-policy", policy,
                        "RACED-NOT-ASSERTED",
                        "the per-class BIRTH_MATRIX defaults are DERIVED from measured "
                        "geometry (Lane no interior -> uniform; Movable GOUGE 31% of "
                        "ERODE -> sqrt_area); this lever OVERRIDES both with one policy "
                        "so the geometry hypothesis is itself falsifiable."),
        })


def lever_existence_target(target: float,
                           weight: float = EXISTENCE_WEIGHT_RACE_START) -> Lever:
    """RACE the survival CUSHION: how far above the decision boundary a word must sit.

    ``target=0.0`` is bare existence -- the decision boundary itself, which is the exact
    semantic definition of the word surviving argmax.  ``target>0`` demands a margin
    cushion, protecting words that are alive but marginal (and therefore likely to die on
    the next optimizer step or under the uint8/resize round trip).

    Falsifier (pre-registered): if no positive target beats ``0.0`` on realized n600
    d_seg, then marginal-but-alive words are not the loss channel and the cushion is
    spending gradient on words that were never going to die.
    """
    if float(target) < 0.0:
        raise ValueError("target must be >= 0 margin units (0.0 = bare existence)")
    if float(weight) <= 0.0:
        raise ValueError("weight must be > 0 to arm the lever (0.0 is the OFF control)")
    return Lever(
        name=f"tr1_existence_target{target:g}_w{weight:g}",
        overrides={"--existence-hinge-weight": str(weight),
                   "--existence-hinge-target": str(target)},
        notes="p4x #920 survival-cushion race arm. 0.0 = bare existence (the decision "
              "boundary = the exact definition of surviving argmax); >0 protects "
              "alive-but-marginal words against the next step and the R round trip.",
        constant_manifest={
            **_manifest("--existence-hinge-target", float(target),
                        "RACED-NOT-ASSERTED",
                        "0.0 is DERIVED (it is the decision boundary itself, not a tuned "
                        "value); any positive cushion is a raced hyperparameter with no "
                        "measured optimum on this vehicle."),
        })


def lever_existence_beta(beta: float,
                         weight: float = EXISTENCE_WEIGHT_RACE_START) -> Lever:
    """RACE the softmax sharpness ``beta`` against its DERIVED per-class default.

    The default is a derivation, not a literal: ``logsumexp_beta`` over ``n`` values
    overestimates their max by at most ``log(n)/beta``, so holding that slack inside a
    declared tolerance gives ``beta >= log(mean_component_area)/tolerance``.  At tolerance
    0.5 margin units that is Lane 7.4587 and Movable 12.9896 -- different per class
    because their area laws differ, which is the birth matrix doing its job.

    Raising beta approaches the exact witness pixel (hard max) and concentrates all
    gradient on one pixel per word; lowering it spreads gradient over the word and starts
    to resemble a mean, i.e. drifts back toward the per-pixel shape this design rejects.

    Falsifier (pre-registered): if low beta (mean-like) beats the derived beta, the
    witness-pixel reading of existence is wrong and the term is really acting as an
    area-weighted reweight under another name.
    """
    if float(beta) <= 0.0:
        raise ValueError("beta must be > 0")
    if float(weight) <= 0.0:
        raise ValueError("weight must be > 0 to arm the lever (0.0 is the OFF control)")
    return Lever(
        name=f"tr1_existence_beta{beta:g}_w{weight:g}",
        overrides={"--existence-hinge-weight": str(weight),
                   "--existence-hinge-beta": str(beta)},
        notes="p4x #920 sharpness race arm. Derived per-class defaults: Lane "
              f"{BIRTH_MATRIX[LANE].beta:.4f}, Movable {BIRTH_MATRIX[MOVABLE].beta:.4f} "
              "(log(mean_area)/tolerance at tolerance 0.5). Low beta drifts toward a mean "
              "= back toward the per-pixel shape this design rejects.",
        constant_manifest={
            **_manifest("--existence-hinge-beta", float(beta),
                        "RACED-NOT-ASSERTED (overrides a DERIVED default)",
                        "the default is derived from the measured component-area law plus "
                        "a declared 0.5-margin-unit tolerance; this lever overrides it so "
                        "the derivation is itself falsifiable."),
        })
