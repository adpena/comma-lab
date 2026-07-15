# PAPER CHECKED — arXiv 2607.12140 "Causal Graphs, Markov Properties and Do-calculus for Stochastic Differential Equations"

**Assessed 2026-07-15** · Philip Boeken and Joris M. Mooij ·
`research_only=true` · pointer **0.19108 / 0.18804 UNMOVED** · no launch, scorer,
evaluator, archive mutation, or score claim.

## Answer first

**Verdict: `WARM-START-SEED`, not a direct theorem transfer.** The paper supplies a rigorous way to
turn a causal SDE into a cyclic structural causal model on whole sample paths, then split and
marginalize that model without changing its observational/interventional semantics. Pact should
transfer that *intervention-and-time-split manifest* into the C0/treatment and costate-organ
apparatus. It must not claim the paper's Markov or do-calculus conclusions for current V9 runs:
the required SDE, solvability, noise, graph, and multi-run distribution custody do not exist.

`verdict_scope=DIRECT-MECHANISM x CURRENT-V9-TRAINING-CORPUS`. This does not reject causal process
models, SDE control, randomized campaigns, or the paper's family. It scopes only the unlicensed
use of these theorems on today's deterministic, n=1-like campaign telemetry.

## Recall-first result

**MEASURED (local corpus inspection):** exact searches for `2607.12140` in `.omx/research/`,
`.omx/state/`, `docs/`, `src/`, and `tools/` returned no prior assessment. The existing
`papers_checked_*` corpus contains no memo for this identifier. `tools/graph_memory_recall.py`
returned fuzzy neighboring causal/control records but no exact paper node. This is a fresh
assessment, not a re-opened result.

## Source custody and method deep-read

**SOURCE-VERIFIED:** the arXiv record and the authors' primary PDF were read. The method—not just
the abstract—was checked through the causal-SDE definition, causal graph, solvability,
marginalization, Markov-property, do-calculus, time-splitting, subsampling, and
Granger/local-independence sections.

The mechanism is:

1. A causal SDE permits each endogenous component to depend on parent histories through its drift,
   diffusion, and jump integrands, with exogenous semimartingales representing external noise.
   Directed edges encode nonconstant functional dependence; shared exogenous causes induce
   bidirected edges.
2. A pathwise stochastic-integral map converts the SDE into a (possibly cyclic) SCM whose variables
   are entire càdlàg sample paths. A perfect intervention replaces the target path equation rather
   than merely perturbing one observed coordinate.
3. **Essential unique solvability** requires an adapted measurable solution function, unique almost
   surely even when complementary processes are supplied as arbitrary adapted inputs. The paper
   gives SCC-wise sufficient conditions and a stronger additive-noise Brownian specialization.
4. Marginalization substitutes solution functions for hidden processes. Under the stated
   conditions it preserves observational and interventional distributions, is order-independent,
   commutes with intervention, and preserves essential solvability.
5. For essentially uniquely solvable systems, the induced path law obeys the global
   sigma-separation Markov property after SCC acyclification. Under the paper's stronger
   additive-noise assumptions, a d-separation result follows by approximating the continuous
   process with acyclic Euler SCMs and passing conditional independence through total-variation
   convergence.
6. The sigma-separation graph supports three do-calculus rules and the no-directed-path/no-effect
   conclusion. The authors do **not** silently generalize the strongest d-separation result to all
   cyclic SDEs; that extension is explicitly beyond the proven scope.
7. **Time splitting** replaces a process by interval-path variables, paired with independent noise
   increments, so causal order across intervals is explicit. Subsampling is then graph
   marginalization onto selected times. The paper also records the important caveat that
   subsampling can erase conditional independences or hide a causal edge through perfect
   adaptation.

These are structural results. They do not estimate a causal effect from one path, manufacture
positivity, or turn repeated epochs from one run into independent treatment assignments.

Primary sources: arXiv `2607.12140` and the authors' PDF
`philipboeken.github.io/assets/pdf/papers/boeken2026causal.pdf`.

## Honest fork versus V9·CGauge

### Why the direct mechanism is not currently applicable

**DERIVED:** V9·CGauge training is a seeded optimizer evolving a level-set INR through a frozen
SegNet/PoseNet receiver. Current treatment arms are typed once at run/stage boundaries; they are
not a custodied causal SDE with independently characterized semimartingale noises. The campaign
currently has neither:

- a declared path-space structural equation for every mutable controller/optimizer state;
- a verified causal graph including exogenous/shared-noise edges;
- proof of essential unique solvability or the paper's additive-noise conditions;
- replicated treatment support sufficient to identify run-level effects; nor
- a time-split intervention object that distinguishes pre-treatment state, intervention semantics,
  descendants, and measurement apparatus.

Therefore using sigma-separation, d-separation, or do-calculus on the current telemetry would be
fake authority. The existing HCM audit independently found zero identified run-level lever effects;
this paper does not pay that missing assignment/support debt.

### Warm start from the assumption fork

**INFERRED, unmeasured:** transfer the paper's *construction discipline*, not its causal verdict.
For every C0-versus-treatment run, define a typed stage-intervention path manifest containing:

- the preserved checkpoint path and hashes before intervention;
- the exact DSL lever/config and whether it replaces a whole path law, a stage transition, or a
  measurement-only field;
- seed and every declared stochastic/noise stream;
- optimizer, EMA, controller, guard, curriculum, scorer, resize/uint8, and receiver states that can
  carry effects across the boundary;
- interval boundaries aligned with preserved stage checkpoints; and
- the target outcome vector (`d_seg`, `d_pose`, exact bytes) plus axis and apparatus custody.

This makes the intervention estimand and invalid adjustment paths explicit. A future multi-run
matched/randomized campaign could then ask whether its graph and support justify HCM/FORE or
SDE-specific identification. Until then, the manifest is a strict **no-false-credit guard** and a
reproducibility surface, not an estimator.

## Route

- **Primary cluster:** organ / causal-attribution apparatus — `#426`, `#436`,
  `.omx/research/hcm_causal_attribution_dig_20260713.md`, and the existing
  `costate_organ_duty_queue_20260711`.
- **Campaign edge:** the Phase-2 C0 treatment A/B queue in
  `.omx/research/P0_campaign_queue_20260715.md`. The proposal does not add or reorder a launch; it
  specifies the future custody required before assigning causal credit to any completed A/B.
- **Admission gate:** a typed manifest plus at least the support/identification conditions already
  demanded by FORE/HCM. Large within-run `pair x epoch` counts do not substitute for independent
  treatment-bearing runs.
- **Not routed:** curvelet/basis, rate, and pose. This paper supplies attribution structure, not a
  representation, bit-allocation law, or pose carrier.

## Triality and authority

- **DAG:** `.omx/research/causal_sde_time_split_routing_DAG_FEED_20260715.md` stages the route.
- **DSL:** no lever or flag is invented. A future implementation must extend the typed campaign/run
  manifest and compile exact intervention semantics from the existing DSL.
- **Equations:** no canonical equation is registered. This arm measured no Pact law and did not
  close a causal identification formula.

`# FORMALIZATION_PENDING: literature-derived intervention-manifest seed; formalize only after a typed producer/consumer and an identified multi-run backtest exist.`

## STORES CONSULTED

`CLAUDE.md` · `AGENTS.md` · `docs/operating_manual_craft_handoff.md` · top current-state memory
entries · `tools/graph_memory_recall.py` · `.omx/research/papers_checked_*` ·
`.omx/research/P0_campaign_queue_20260715.md` ·
`.omx/research/hcm_causal_attribution_dig_20260713.md` ·
`.omx/research/organ_regime_conditional_dispatch_436_20260711.md` · canonical task status for
`#426`/the organ duty queue · arXiv metadata/abstract · the authors' full primary PDF.

**Pointer delta:** none. **Raw Pact measurements:** none. **Promotion authority:** none.
