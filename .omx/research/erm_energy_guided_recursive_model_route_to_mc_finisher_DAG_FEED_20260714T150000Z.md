# DAG FEED — arXiv 2607.10128 ERM (Energy-guided Recursive Model) → routes to the terminal-band K-candidate selector (#396/#400/#319)

**Leg:** DAG (routed intake). Operator share 2026-07-14. Re-pointed against the full object inventory —
UNLIKE MorphoHDL / 2605.29035 (conceptual-only, no lever), **this one has a genuine measurable fit** with a
LIVE arm. Disposition: route (not dismiss, not a fresh 6th dispatch) — the consumer is the MC-finisher.

## What ERM is (Zhao & Tang)
Recursive reasoning models update a small net's latent state over D steps; test-time scaling generates K
candidate trajectories but LACKS a principled selector (prior work = heuristic voting / extra q-heads).
ERM's fix: an **explicit Hopfield energy** = a memory of VALID local/global structures, used as the
selector/ranker over the K candidates, composable with **parallel tempering** for sampling efficiency.
D=64, K=128 → 98.97% Sudoku / 88.04% PPBench / 99.30% Maze. Domain: energy-guided candidate selection for
discrete constraint-satisfaction (NOT video/seg — the TRANSFER is the ALGORITHM, not the results).

## The genuine fit — our terminal-band candidate selection (verdict-scope: FORMULATION-level lever, unmeasured)
Our live surfaces are EXACTLY the "generate K candidates → select the best" regime ERM formalizes:
- **#396 (in_progress)** — exact-metric MONTE-CARLO FINISHER (arXiv 2607.08406): gradient-free terminal
  optimization of the discrete argmax d_seg. Generates candidate discrete perturbations, selects by the
  exact through-R d_seg.
- **#400 (in_progress)** — MC-finisher pair-local DIAGONAL mode: d_seg polish + ξ (dxi) pose polish.
- **#319 (pending, campaign-layer, backtest-gated)** — SimpleTES K>1 candidate emission when the through-R
  evaluator band spans 0.

**What ERM ADDS (the marginal lever, honestly scoped):** our MC finisher ALREADY has the exact energy —
the true through-R d_seg IS the selector. ERM's transferable value is therefore NOT "use an energy"; it is:
1. **Parallel tempering over the K candidates** — replace i.i.d./heuristic sampling with tempered-swap
   sampling ⇒ better coverage of the discrete flip-configuration space per exact-eval budget (the exact
   eval is our expensive step; PT spends it better).
2. **A CHEAP Hopfield "valid-structure" pre-rank prior** — a memory of VALID local SegNet-argmax partition
   patches, used to pre-rank/prune candidate flip-configs BEFORE the expensive through-R eval (don't waste
   exact-evals on obviously-invalid partitions). "Valid local structure" here = argmax-consistent boundary
   patches; the memory is built from the frozen-scorer's own partition statistics (video-derived, but a
   DECODE-TIME selector prior, not shipped payload — so rule-118-clean IF used only to guide search).

## Honest limits (NO-FAKE / over-credit guard)
- Results are Sudoku/Maze (hard discrete CSP with crisp valid/invalid). Our terminal band is a NOISY
  argmax boundary at the flicker-floor (0.005318, GT-side oracle floor per L85) — "valid structure" is
  fuzzier; the Hopfield memory may not have crisp attractors. This is the reformulation risk to measure.
- The MC finisher already selects by the EXACT metric; PT + pre-rank only help if SAMPLING (not the
  energy) is the current bottleneck of #396's terminal search. That is the $0-measurable question.
- No pointer implication until measured through R at n600, byte-closed.

## Disposition (per per-task-cap ≤3 + don't-over-spawn + 5 live arms)
ROUTE to the #396/#400 MC-finisher arm as a specific in-context lever — NOT a fresh 6th concurrent dispatch
(this is a REFINEMENT of a live arm, best evaluated by that arm's owner in-context). The MC-finisher arm,
when it settles its base exact-search, should evaluate: (a) is candidate SAMPLING the bottleneck? if yes →
parallel-tempering the K-candidate draw; (b) build the cheap Hopfield valid-partition-patch pre-rank prior
and measure exact-evals-saved at equal terminal d_seg. Both $0, through-R, n600. If the operator wants it
pursued as its own arm now, say so; default = queued lever for the live MC-finisher.

**Pointer:** 0.19108 / 0.18804 UNMOVED. This is a routed algorithmic lever for the terminal-band selector,
not a score-mover until #396/#400 measure it byte-closed through R.
