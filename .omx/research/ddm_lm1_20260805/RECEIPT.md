# ddm_lm1 — cmpnd.ai "Let the Model Write the Code" (Flex/DSPy/GEPA) crosswalk
# Operator drop 2026-08-05 (mid-turn URL). MAIN inline deep-read (blog-scale; fleet at 2 arms).
# Source: https://www.cmpnd.ai/blog/let-the-model-write-the-code.html
# STORES CONSULTED: GEPA priors (#481 CL-reconciliation, #655 q1 GEPA-omni crosswalk), rd1
# λ-continuation (#667), ct1 ΔS-per-hour telemetry (#707), workflow v2 (#590), harvest contract
# (tac.subagent_contract), solve-first doctrine (m13/sc1/od-line), waterfill laws (m66, ph3 §7).

## The article's doctrine (their words)
Flex exposes SOURCE CODE to the GEPA optimizer: the reflection model may "decompose your
program, write helper functions, implement routing logic, and rewrite your prompts." Metric =
task score − λ·(LLM-call cost): each call must "buy back more accuracy than it costs."
Four recurring optimizer moves: decomposition · method selection (code-vs-model) · routing
(clear→deterministic, ambiguous→model) · evolution. Sandboxed execution of model-written code.
Measured: conflation 90.4%→95.0% at −28% cost (λ=0); λ=0.4 → 92.1% at $0.01/1k (1 LLM call per
240 records). SWE-bench pilot: Flex-optimized Haiku 4.5 4/12 vs 0/12 unoptimized. Doctrine:
"continually compile our harness" as data/models/tactics arrive.

## Crosswalk table (per-row disposition; consumers named)
1. λ-priced calls ("buy back more than it costs") → ALREADY-EMBODIED. This IS the contest
   objective's structure (every byte buys back distortion) and our waterfill/exchange-rate law
   (m66 gap denominators, ph3 §7 joint realized exchange rate). At the apparatus level it is
   the caps-genus/VOI spawn test + ct1 ΔS-per-hour telemetry. Independent convergent
   validation; no build. Consumer: none needed.
2. λ-sweep spectrum of operating points → ALREADY-EMBODIED as rd1 λ-continuation R(D)
   frontier (#667) — the article rediscovers λ-continuation for harness economics. No build.
3. Routing (deterministic cheap path / model for ambiguity) → ALREADY-EMBODIED as the
   solve-first/train-least doctrine (m13; "seg ~100% solvable → gap = CARRIAGE"; QA43
   tail-targeted per-pair pose = exactly their routing move at pair granularity). No build.
4. "Continually compile the harness" → ALREADY-EMBODIED as the triality cycle
   (DAG→DSL→run→rows→equations→next-DAG) + workflow v2. Delta they'd push: let the OPTIMIZER
   author DSL programs with the metric in-loop. We deliberately hold actuation at
   advisory-only (CONTAINMENT law — heavy launches are operator-GO); the PROPOSE side already
   lives in the costate duty queue. NOT-ADOPT for autonomy expansion (containment binds).
5. Optimizer-authored sub-programs on a bounded measurable surface → TRACKED-CANDIDATE
   (fire-order recorded, like rate_crush #949): a Flex-style metric-in-loop proposer for the
   tq1-family token-edit MOVE GENERATORS (priority ordering is hand-derived; judge candidate
   generators on realized ΔS per scorer-second). Honest EV check: the whole tq1 family has
   yielded −1.9e-4 S — optimizer overhead likely dominates at current yield. Fires ONLY if
   tq1c's larger-menu measurement shows the family's yield curve is ordering-limited (its
   receipt will show accepted-rate vs rank). Consumer: tq1c receipt adjudication.
6. Sandboxed untrusted model code → ALREADY-EMBODIED (codex arm sandbox git-block; locked-env
   receiver smokes). No build.
7. SWE-bench "optimized harness ≈ mature hand-built harness" → LESSON-ONLY: supports the
   harvest-engineered subagent contract (a manual compile of the same kind); no new surface.

## Verdict
CONVERGENT-VALIDATION drop: 5 of 7 rows already embodied by measured campaign law (the
article independently arrives at λ-continuation + call-pricing + routing + continuous harness
compilation). One containment-bound NOT-ADOPT (optimizer actuation). One tracked candidate
with a named falsifier-consumer (row 5 → tq1c). No fire-now build. Axis n/a ($0 doctrine read).
