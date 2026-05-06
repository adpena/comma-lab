# Grand Council deliberation: Meta-Lagrangian Pareto-gate design decisions

**Date:** 2026-05-06
**Convened by:** claude:main per user directive ("we found a new floor remember
proceed with that the new code and have grand council provide design decisions")
**Triggering artifacts (codex 2026-05-06):**
- `src/tac/optimization/cross_paradigm_atoms.py` — 5-family atom adapter layer
  (HNeRV rate recode + WR01 wavelet + categorical/openpilot mask + LA-pose + foveation)
- `src/tac/optimization/meta_lagrangian_allocator.py` — Pareto-gated atom ledger
- `tools/build_cross_paradigm_atom_ledger.py` — CLI to build cross-paradigm ledger
- `.omx/research/meta_lagrangian_pareto_gate_20260506_codex.md`
- `.omx/research/cross_paradigm_atoms_20260506_codex.md`

## What the new architecture does

The meta-Lagrangian Pareto-gate moves us from "pick the next paradigm to
dispatch" to "compute the Pareto frontier of stacked atom combinations across
5 orthogonal families, then dispatch the frontier set." The implicit new
floor is **not a single number** — it's the Pareto-optimal combination
along the (byte_delta, expected_seg_dist_delta, expected_pose_dist_delta,
confidence) axes, with `pareto_scope=family_group` so orthogonal families are
compared within their scope before cross-stack interaction review.

This explicitly extends the Wave-Ω 3-stack model (SJ-KL + NeRV-mask +
block-FP) into an N-family combinatorial space. The "new floor" is whatever
the Pareto frontier achieves once empirical-grade atoms enter the ledger.

## Design decisions surfaced

### Q1: Dispatch ordering — Pareto-frontier-first vs single-best-atom-first?

**Options:**
- A) Dispatch every atom on the Pareto frontier in parallel (race-mode rule 1).
- B) Sequentially advance one atom from each family (orthogonal coverage).
- C) Dispatch the single atom with highest predicted Δscore × confidence.

**Council positions** (5 inner-ten):

- **Shannon (LEAD):** A. The Pareto frontier IS the R(D) feasible region; any
  point on the frontier is achievable, but only an empirical dispatch
  collapses the prediction band to a measurement. Race-mode parallel-fan-out
  per CLAUDE.md NON-NEGOTIABLE Rule 1.
- **Dykstra (CO-LEAD):** A with caveat. Dykstra alternating projections
  guarantee the intersection of family-feasibility sets is reached; we need
  AT LEAST one atom from each family on the frontier to converge. Pure A
  satisfies this if the frontier spans families.
- **Yousfi:** A. The contest is decided by score, not by atom-count. Dispatch
  every frontier atom in parallel; harvest only the contest-CUDA results.
- **Contrarian:** **B**. Single-best risks concentration (Hassabis trap from
  2026-04-29). Sequential family-coverage protects against the case where
  the frontier collapses to a single dominant family that turns out to have
  a measurement bug.
- **Hotz:** A. Burn the GPU. Race-mode is the rule. The Contrarian is
  worrying about a failure mode that the parallel-actuator + harvest-loop
  already catches (per `tools/parallel_dispatch_top_k.py` →
  `tools/harvest_and_reseed.py`).

**Verdict: 4 for A / 1 for B → A wins.** Dispatch every Pareto-frontier
atom in parallel. The Contrarian's caveat is partially addressed by the
fact that `pareto_scope=family_group` already enforces per-family
representation.

### Q2: Empirical-grade gate — when does an atom graduate from `[predicted]` to dispatchable?

**Options:**
- A) Atom needs a closed-byte archive manifest (just enough to dispatch).
- B) Atom needs a closed-byte archive manifest + per-component delta
     measurement (Lane G v3 anchor or equivalent).
- C) Atom needs B + Pareto-frontier membership in its family scope.

**Council positions:**

- **Shannon:** C. Without Pareto membership, dispatching the atom is a waste
  of GPU — there's a strictly-better candidate in the same scope.
- **Dykstra:** C. Same logic via convex feasibility: if your atom is
  Pareto-dominated, projection onto the feasibility set lands on the
  dominator, not on you.
- **Selfcomp:** B. Pareto-frontier membership requires the OTHER atoms to
  have measured deltas; the FIRST dispatch in a family has nothing to
  compare against. C creates a chicken-and-egg deadlock.
- **MacKay:** B + C compromise: first-of-family dispatches under B, all
  subsequent dispatches under C. This avoids the deadlock while preserving
  Pareto discipline.
- **Ballé:** C, with the operator override that the FIRST atom in a family
  is auto-Pareto-frontier (trivially, it's the only point).

**Verdict: 1 for B / 3 for C / 1 for B+C compromise → C wins** with the
operator-override that first-of-family is auto-frontier (Ballé's clarification
addresses Selfcomp's deadlock).

### Q3: Cross-family stacking — when do orthogonal-family atoms compose?

**Options:**
- A) Always — stack everything; the contest scorer measures the combined effect.
- B) Only when interaction assumptions are explicitly verified per atom pair.
- C) Stack within a "verified interaction zone" defined by prior empirical
     stacks (Wave-Ω 3-stack predicts 0.180; anything beyond needs new evidence).

**Council positions:**

- **Shannon:** B. Per the May 4 race postmortem, stacking claims must show
  overlap analysis, not just naive sum. Naive A is the Hassabis-trap.
- **Dykstra:** C. The convex-hull intersection collapses if family
  assumptions don't hold. Verified interaction zones bound the
  optimization to known-feasible regions.
- **Yousfi:** B. Anti-naive-stacking is a contest-faithfulness rule.
- **Quantizr:** A but only with a "stacking adjudication" empirical pass
  (run the stack, measure, retract if interaction kills score).
- **Contrarian:** B. Quantizr's "measure and retract" assumes you have
  cheap GPU; we don't.

**Verdict: 3 for B / 1 for C / 1 for A → B wins.** Stack only when
interaction assumptions are explicitly verified (the
`interaction_assumptions` field codex added to atoms is exactly this).

### Q4: Pareto scope choice — `family_group` (default) vs global?

**Options:**
- A) `family_group` (default): atoms compared only within their family
     before cross-stack review.
- B) `global`: atoms compared across all families immediately.
- C) Tiered: `family_group` for first-pass ranking, `global` for final
     dispatch selection.

**Council positions:**

- **Shannon:** A. Global comparison conflates orthogonal axes (a
  better-pose atom is not "dominated" by a better-mask atom; they're
  complementary).
- **Dykstra:** A. Per Dykstra alternating projections, intra-family
  optimization happens FIRST, then cross-family.
- **Selfcomp:** A. The PR#56 paradigm is a cross-family stack of
  Gaussian-LUT mask + xz-int8 weights + AV1 video; you don't pick "best"
  among them.
- **Hotz:** C. Pragmatic — first round narrows by family, second round
  picks the dispatch wave.
- **Quantizr:** A. Hotz's C is just A with a manual second pass; the
  default is fine.

**Verdict: 4 for A / 1 for C → A wins** (codex's default is correct).

## Concrete dispatch recommendation

Per the verdicts:

1. Run `tools/build_cross_paradigm_atom_ledger.py` to generate the current
   Pareto frontier across 5 families.
2. For each Pareto-frontier atom:
   - If `archive_manifest_sha256` is present and `dispatch_blockers` is empty
     → eligible for parallel dispatch.
   - Otherwise → record blockers, do NOT dispatch.
3. Fan out the eligible-frontier atoms via
   `tools/parallel_dispatch_top_k.py` (race-mode Rule 1 actuator).
4. Harvest results via `tools/harvest_and_reseed.py` and re-feed empirical
   anchors back into `.omx/calibration/anchors_*.json`.
5. Re-run step 1 with the updated empirical anchors to expand the Pareto
   frontier.

## What is NOT yet decided (deferred to operator)

- **GPU budget cap** for the first parallel dispatch wave. The
  `tools/parallel_dispatch_top_k.py --max-total-cost` flag exists; the
  Council recommends $5-10 for the first wave (matches prior Wave-Ω
  recommendation), but this is operator-gated.
- **First-of-family dispatches**: which families dispatch first? Per the
  Q2 verdict (with Ballé's operator-override), every family contributes ≥1
  atom in the first wave. Operator picks the prioritization within
  budget.
- **Floor target**: the predicted floor under stacking is operator-set.
  No single number — the empirical floor is whatever the harvested wave
  measures on contest-CUDA.

## Cross-references

- `feedback_may_4_hnerv_race_postmortem_20260505.md` — race-mode rule 1
- `project_grand_council_FIELDS_MEDAL_shannon_floor_obsession_20260501.md`
  — Wave-Ω 3-stack predicting 0.180 (the previous reference)
- `feedback_goal_is_lowest_score_not_quantizr_paradigm_match_20260506.md`
  — strategic re-frame
- CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first" non-negotiable
- CLAUDE.md "Adversarial council review of design decisions" non-negotiable

## Council adjournment

Verdicts on Q1/Q2/Q3/Q4 reached. Implementation falls to the operator's
next dispatch decision. The Pareto-gate + cross-paradigm atom adapter
landed by codex are the canonical planning surface going forward.

---

**Council members consulted:** Shannon (LEAD), Dykstra (CO-LEAD), Yousfi,
Contrarian, Hotz, Selfcomp, MacKay, Ballé, Quantizr — 9 of inner-ten
positioned. Carmack and Schmidhuber not consulted (Carmack is grand-council;
Schmidhuber's joint-ADMM lane was deferred per 2026-04-29 senior engineer
review).
