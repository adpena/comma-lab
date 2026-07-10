# PAPERS-CHECKED — "A Stochastic–Geometric Theory of Scaling Laws in Grokking" (arXiv 2606.30388)

**Pulled:** 2026-07-10 (operator-supplied, #395 texture-trunk P0 training-dynamics input). `[CITED — abstract
verified via WebFetch; PDF scaling-law forms NOT yet read]`. **MEANS; pointer 0.19110 UNMOVED.**

**Authors:** Róisín Luo, Christian Gagné, Jonas Ngnawé, Ihsan Ullah, Karyn Morrissey.
**arXiv:** https://arxiv.org/abs/2606.30388 (the ID resolves — the abstract MATCHES the operator's
description exactly: shell-core geometry, Adam+weight-shrinkage, stopping-time scaling laws for lr/batch/ℓ2).

## Honesty note on the ID (operator flagged a possible slip)
The operator wrote *"possibly an operator ID-slip for a texture paper — say so honestly."* VERDICT: **NOT a
slip in the sense of a wrong fetch** — the ID resolves to the paper the operator described (a TRAINING-DYNAMICS
/ grokking paper), and its content is exactly what was cited for the training-dynamics application. It is
**NOT a texture paper** — if the operator additionally intended a separate texture reference, that is a
distinct pull still owed (none was found under this ID). The training-dynamics application below stands on
this paper's own merits.

## What it says (abstract, verified)
Delayed generalization ("grokking"): nets MEMORIZE first, then generalize after prolonged training with an
ABRUPT transition. The solution-space topology is a **shell-core**: random inits cluster on an outer
spherical SHELL, enclosing a MEMORIZATION region, which contains a CORE of GENERALIZATION solutions.
Adam + weight-shrinkage (weight decay) drives the shell→core descent. Using **stopping-time theory** they
derive **scaling laws for the transition time in learning-rate, batch-size, and regularization-strength**;
predictions match experiments and recover prior empirical grokking findings.

## Application to the texture trunk (#395) — DERIVED, the WHEN + HOW-fast input
A tiny FRESH texture trunk T bolted beside a CONVERGED partition trunk G is a textbook shell-core setup:
T's coefficients start on the random-init shell while G already sits in its generalization core. Consequences,
each a design lever (not a guess):

1. **Warm-start deployment risks a grokking delay.** If T is added at a `--resume-from` of a converged G and
   trained with the run's default lr/decay, T may sit in a memorization/shell regime for a long transient
   before its texture generalizes — the abrupt-transition risk. ⇒ the `TextureTrunk` lever carries a
   `window` warm-start-epochs parameter; the A/B's CONDITION-style arm must budget enough tail epochs to
   clear the transition, not judge T at a mid-transient checkpoint (the "EMA-shadow lag / early-run rise"
   confound sister, MEMORY early-run row).
2. **Derive T's lr + weight-decay from the scaling laws, NOT copy G's.** The transition time scales with
   (lr, batch, ℓ2); a fresh sub-module wants a HIGHER effective lr and/or a TUNED decay to reach its core
   fast (G's converged-phase small lr would strand T on the shell). Concrete pre-registered arm: T's
   coefficient group gets its own lr (a small multiplier > the trunk's) — a per-group optimizer setting, to
   be pinned from the paper's law once the PDF forms are read (OWED: read §scaling-laws for the exact
   lr/ℓ2 exponents before the warm-start arm; the from-scratch arm §does not need it).
3. **Connect to island-birth timing + the event-triggered schedule (#315).** The abrupt shell→core transition
   is the SAME shape as our measured island-birth / stage-transition events; the event-triggered τ/schedule
   controller (#315) is the natural WHEN-to-engage gate for T (engage the texture trunk at/after the
   partition's island-birth, not before — the geometry must exist before it can be textured). Pre-registered:
   an event-gated `window` (engage T when G's per-class d_seg settles) vs a fixed-epoch engage.

## Owed
- Read the PDF's exact scaling-law forms (lr/batch/ℓ2 exponents) BEFORE the warm-start A/B arm, to pin T's
  per-group lr + decay rather than sweep them (P: derive-don't-sweep).
- License check at any code-port time (none needed for the CITED reasoning here).

**Cross-refs:** `.omx/research/texture_trunk_p0_design_20260710.md` (§training-dynamics) · #315 event-triggered
schedule · MEMORY early-run EMA-shadow-lag row (mid-transient confound). Advisory; pointer 0.19110 UNMOVED.
