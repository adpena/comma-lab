---
council_tier: T2
council_attendees: [Shannon, Dykstra, Rudin, Daubechies, Schmidhuber, Contrarian, Assumption-Adversary]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_predicted_mission_contribution: apparatus_maintenance
council_override_invoked: false
council_dissent:
  - member: Contrarian
    verbatim: "SimpleTES's headline idea — credit an attempt by the trajectory's FINAL score — is a Monte-Carlo RETURN, the categorical opposite of our costate λ=∂S/∂x (a local adjoint gradient). Do not let the vocabulary ('evaluation-driven', 'discovery loop', 'credit assignment') smuggle a campaign-layer SFT method into the in-run costate estimator, where it does NOT belong. The only honest DRAW-FROM is a CAMPAIGN-layer complement, and its flagship (outcome-credit) is corpus-gated to near-uselessness until a fleet of completed runs exists."
council_assumption_adversary_verdict:
  - assumption: "'costate controller' and 'SimpleTES' operate on the same object because both are 'evaluation-driven'"
    classification: CARGO-CULTED
    rationale: "SimpleTES post-trains an LLM POLICY on discovery trajectories (SFT with outcome-weighted CE). Our costate controller estimates marginal-ΔS sensitivities of a training run and selects levers. Different agents, different objects, different math (MC-return vs adjoint). The overlap is at the META/campaign layer (which config to try next), NOT the estimator core."
    empirical_verification_status: VERIFIED_VIA_SOURCE_INSPECTION
  - assumption: "the 'stepping stone' insight overturns our NEVER-REGRESS/POWERPLAY guard"
    classification: HARD-EARNED
    rationale: "It does NOT — POWERPLAY keeps the FRONTIER (best archive/checkpoint) and explores forward; SimpleTES keeps STEPPING STONES and credits by outcome. They COMPOSE (Schmidhuber-lineage cousins). But naming the tension prevents a future misread of never-regress as 'never try a locally-regressing lever'."
    empirical_verification_status: VERIFIED_VIA_SOURCE_INSPECTION
council_decisions_recorded:
  - "op-routable 1: SPLIT verdict — DRAW-FROM (campaign/Phase-B lever-selection layer) + NOT-RELEVANT (costate estimator core + the #315 per-class fix)"
  - "op-routable 2: the one non-corpus-gated actionable idea = K>1 candidate emission for a NOISY through-R evaluator (measured wide bands) — a Phase-B design note, not an estimator change"
  - "op-routable 3: the corpus-gated idea = campaign outcome-credit sibling (Monte-Carlo return complement to in-run adjoint λ); GATED on a fleet of completed runs, degenerates to rollback_gain below that"
  - "op-routable 4: NO controller change adopted until backtested against v1-v5 + #205 forensic logs; backtest spec in §5"
---

# SimpleTES vs our COSTATE CONTROLLER — assessment (operator-flagged 2026-07-05)

Operator: *"Research this paper as well, this is may inform our costate controller — https://haotianye.com/blog/simpletes/"*. Sources: the blog post + arXiv **2604.19341** (Ye, Lin, et al., 2026; code `github.com/wq-will/SimpleTES`). Read beyond the abstract per the RESEARCH-DEPTH binding. All numbers below [external]; pointer **0.19110 UNMOVED** — this is a MEANS assessment (what to build), not a byte off the archive.

## 1. What SimpleTES IS (precise, with its math)

SimpleTES ("Simple Test-time Evaluation-driven Scaling") is a **test-time scaling framework for open-ended scientific discovery** (math constructions, GPU-kernel/quantum/algorithm optimization) built on a **propose–evaluate–refine** loop scaled along three axes: **C** parallel exploration trajectories × **L** sequential refinement steps × **K** local candidates per step (budget ≈ C·L·K evaluator calls). It assumes only a task-specific **evaluator V** (the sole non-shared component) and that the **MAXIMUM** solution found matters (not the mean), so early low-scoring attempts are kept as **stepping stones**. It then closes a learning loop: the trajectory histories **post-train the policy** via a single weighted-cross-entropy objective

```
ℒ(θ) = − E_{(x, ŷ, w) ~ 𝒟} [ w · Σ_i log π_θ(ŷ_i | x, ŷ_{<i}) ]
```

where the crucial detail is that **`w` is set by the FINAL score of the whole trajectory the attempt belongs to, not by the attempt's own in-the-moment score** — "an attempt is rewarded for the discovery it ultimately leads to." Empirically it reports SOTA on 20/21 problems across six domains using only open-source gpt-oss-120b (e.g. Erdős min-overlap 0.380868 beating 0.380927; a 1.122 ms H100 TriMul kernel).

**In one line:** SimpleTES is an *outcome-weighted (Monte-Carlo-return) SFT recipe wrapped around a C×L×K parallel-search discovery loop*. There is **no** control-theoretic / adjoint / costate / eligibility-trace / value-function / advantage formalism — credit assignment is entirely "final trajectory score → uniform weight on every token of that trajectory." That absence is the load-bearing fact for our question.

## 2. Mapping onto OUR costate frame (drawn explicitly)

Our costate frame (`project_meta_layer_above_triality_hamiltonian_control_costate`, `costate_controller_design_20260705.md`): the campaign = ONE controlled learning dynamics; **λ = ∂S/∂x** is the *local adjoint / shadow-price* (marginal-ΔS-per-lever), estimated by windowed OLS over measured n600 verdict rows, feeding a Pontryagin-style `u* = argmax_ready [ΔS·effect − cost]` under NEVER-REGRESS. Correspondence, term by term:

| Our object | SimpleTES analogue | Same or different? |
|---|---|---|
| costate **λ = ∂S/∂x** (local adjoint gradient) | trajectory **final-score weight w** (Monte-Carlo return) | **DIFFERENT KIND** — a derivative vs a return. Not the same math; not a canonical-equation correspondence. |
| Pontryagin adjoint / switching function | (none) | SimpleTES has **no** adjoint/optimal-control layer. |
| temporal credit across *stages* (per-stage dS/dep) | temporal credit across *attempts* (outcome→all steps) | **Weak analogue** — both push credit backward in time, but ours is *differential-per-stage*, theirs is *uniform-final-outcome*. |
| **per-class / per-term** attribution (#253/#255, the #315 fix) | (none — scalar final score only) | **NOT-RELEVANT.** SimpleTES has no per-term credit; it cannot inform the binding-term/per-class deadlock fix. |
| NEVER-REGRESS = POWERPLAY (keep frontier) | keep **stepping stones**, credit by outcome | **COMPOSE** (Schmidhuber-lineage cousins): frontier-preservation + forward exploration are compatible; naming the tension guards against a greedy misread. |
| in-run actuation (rollback / lever-gate) | (none — SimpleTES acts at the search+SFT layer) | **NOT-RELEVANT** to in-run actuation. |
| **campaign lever/config selection** (Phase-B, #216 instrument) | **C×L×K propose-evaluate-refine** + outcome-credit | **THIS is the real correspondence** — a concrete discovery-loop *algorithm* for the campaign layer. |

The honest picture: SimpleTES lives at the **campaign / DSL-controller layer** (which config to try next, given a noisy evaluator), NOT at the **costate-estimator core** (adjoint λ, per-class attribution). It is the *algorithmic* sibling of the EdgeBench entry (which gave the *scaling law* of the same discovery process) and a cousin of our own POWERPLAY. It informs the DSL leg's *search structure*, not the costate λ.

## 3. VERDICT — SPLIT

- **NOT-RELEVANT** to the costate *estimator* core (`costate_estimator.py`) and to the just-shipped **#315 per-class / binding-term** fix. Reason: SimpleTES credit is a scalar Monte-Carlo return with no per-term/per-class decomposition and no adjoint — it cannot see the term that our scalar-S classifier missed (the v5 frozen-descending-S deadlock). Adopting its weighting there would be the exact category error the Assumption-Adversary flagged.
- **DRAW-FROM (campaign / Phase-B lever-selection layer only)**, two ideas ranked by EV, both as **design notes for the shadow_controller's Phase-B**, NOT estimator edits (sibling #315 owns `witness_control/`):

  **DF-1 (EV: medium; actionable NOW, corpus-free) — K>1 candidate emission for a NOISY evaluator.** SimpleTES's K-local-candidates axis exists precisely because "noisy local estimates benefit from sampling K before commit." Our through-R verdicts ARE measured-noisy (design memo §4: live recommendation band `ΔS −1.78 [−4.45, +0.89]`, n=3 spanning 0). The controller today emits **top-1** (greedy). The SimpleTES-informed Phase-B change: when the leading recommendation's band **spans 0** (a measured, already-computed trigger), emit the **top-K** candidate configs for *parallel* through-R evaluation and select by the evaluated result — i.e. propose-evaluate-refine instead of propose-commit. This is a `shadow_controller._recommendations` return-shape parameter (`emit_k`, default 1 = today's behavior) + a Phase-B note; it changes no estimator math.

  **DF-2 (EV: low now, higher at horizon; CORPUS-GATED) — campaign outcome-credit as the Monte-Carlo complement to in-run λ.** SimpleTES's flagship (credit a lever/config by the FINAL score of the run it seeded, not its local slope) is a genuinely different signal from our local windowed-OLS λ, and it is **decay-immune by construction** — it would NOT have made the ep450 over-prediction miss (design memo §3, the one honest backtest failure: local λ overpredicts creep under deceleration). BUT: with **one-to-few** completed runs it degenerates to `rollback_gain` (already built), and cross-run lever attribution is CONFOUNDED (design memo §2, item 4: the #205↔seed-fix pair is 13 simultaneous diffs → UNIDENTIFIABLE). So outcome-credit only earns its keep once a **fleet of completed runs shares lever-families** — the same corpus gate as #211 (amortized meta-init) and the EdgeBench ≥8-curves gate. Proposed home: a NEW sibling `witness_control/campaign_outcome_credit.py` (spec below), NOT an estimator edit, ingesting completed-run endpoints (final implied-S / byte-closed S + lever vector) and emitting per-lever-family outcome-credit **with a corpus-size refusal gate** (mirrors the estimator's UNIDENTIFIABLE discipline). It feeds Phase-B lever ranking as a SECOND signal beside local-λ — never replacing it.

  **DF-3 (EV: framing only) — stepping-stone / never-regress reconciliation.** Add a one-paragraph invariant to the Phase-B design: NEVER-REGRESS binds the FRONTIER (best checkpoint/archive is never lost), NOT the exploration policy — a lever with locally-negative ΔS that seeds a lower final S is *exactly* a POWERPLAY-legal stepping stone (the CE→tau→l7→Muon stage breakthroughs are our own measured instances; the estimator's existing `transition_transient` "don't act on a recent post-boundary rise" guard is already a stepping-stone-aware primitive). Prevents a future greedy misread.

## 4. The named controller integration (design/verdict — for a FUTURE task, NOT built here)

Sibling #315 is actively editing `src/tac/witness_control/` (estimator + shadow controller) → **this assessment does NOT touch that code** (collision avoidance). The consumable design:

1. **`witness_control/campaign_outcome_credit.py` (NEW sibling; DF-2):** `credit_levers_by_outcome(completed_runs) -> list[LeverCredit]` where each `LeverCredit = {lever_family, mean_final_S, n_runs, band, status}` and `status ∈ {MEASURED, UNIDENTIFIABLE(corpus<N or confounded)}`. Refuses (UNIDENTIFIABLE + the gap) below N completed runs OR when the lever-family co-varies with ≥k others across the corpus (reuses the estimator's confound logic). Emits the Monte-Carlo-return complement to `stage_epoch_costates`' adjoint λ.
2. **`shadow_controller._recommendations` param `emit_k` (DF-1):** default 1 (today). When the top candidate band spans 0, return top-K; each carries the same evidence chain + never-regress guard. Pure return-shape change.
3. **Phase-B design doc paragraph (DF-3):** the frontier-vs-exploration invariant.

Ranked by EV toward the two live goals: **DF-1 > DF-3 > DF-2** for the *binding-term-diagnosis* goal (DF-2 gives nothing there — per-class is the wrong axis for a scalar-return method); **DF-2 > DF-1 > DF-3** for the *Phase-B cross-run lever-selection* goal (once the corpus exists). Neither touches the #315 per-class fix, which SimpleTES is orthogonal to.

## 5. "If measured" discipline — the BACKTEST SPEC (binding before any adoption)

Per the design memo's own NO-FAKE validation (§3 the 5/7-rediscovery backtest), **no DF is adopted until it is replayed against the v1–v5 + #205 forensic logs and shown to catch what the scalar-S controller missed.** Concrete specs:

- **Backtest DF-2 (outcome-credit vs local-λ):** replay the completed-run endpoints — v1 (organic death ~ep92), #205 (crept to ep525, net +40.4% over CE-best), seed-fix (descending), v5 (0.026 gold then re-deadlock). Two acceptance criteria: **(a)** does an outcome-weighted lever-credit ranker *reproduce* the known verdicts (rank the crept/deadlocked configs below the descending ones)? **(b) the discriminating test:** at the **ep450** decision where local-λ overpredicted (+0.0060 [central] vs realized +0.0004), outcome-credit uses the *endpoint* → it structurally *cannot* make that mid-run over-prediction, BUT it *also cannot act mid-run* (no endpoint yet) → the crisp expected finding is **"outcome-credit is a CROSS-RUN lever-selection complement, useless for in-run stopping; local-λ remains the in-run signal."** If the backtest shows outcome-credit merely re-deriving `rollback_gain` on the available (few) runs, it is **NOT adopted** (records the corpus gate). This is a $0 replay (endpoints already in the run dirs).
- **Backtest DF-1 (K>1 for a noisy evaluator):** this is NOT a historical-log backtest — we never ran parallel-K arms, so it cannot be validated against v1–v5. Honest status: the *noisy-evaluator premise* is **SUPPORTED** by our measured band widths (n=3 spanning 0, design memo §4), but the *K>1 win* is **UNMEASURED** until a forward parallel-arm experiment runs (a $-costed multi-config through-R batch). Do NOT assert the win; ship it as a Phase-B design note whose value is *contingent* on that future measurement. This honesty is the point — SimpleTES's own K>1 claim is on LLM discovery with a cheap evaluator; our through-R evaluator is expensive (n600), so the C·L·K budget arithmetic must be re-derived at our costs before any parallel-K commit.
- **Backtest DF-3:** no measurement — it is a framing invariant; its correctness is that it contradicts no measured verdict (the transition_transient guard already embodies it).

## 6. Observability surface

Inspectable: every DF is a named return-field or a named sibling module with a status enum. Decomposable: outcome-credit vs local-λ are two separately-printed signals (never blended). Diff-able: the backtest replays are `--as-of-epoch`-style historical passes. Queryable: outcome-credit rows would live in a JSONL sidecar mirroring `costate_shadow.jsonl`. Cite-able: each credit names the completed runs behind it. Counterfactual-able: the ep450 discriminator IS the counterfactual (what would each signal have said).

## Canonical-vs-unique decision per layer
- Verdict framing: ADOPT_CANONICAL (the papers-checked ledger format + the split DRAW-FROM/NOT-RELEVANT discipline).
- Correspondence math: UNIQUE (no prior surface maps MC-return credit onto our adjoint λ).
- Backtest harness: ADOPT_CANONICAL by deferral (the design memo's §3 as-of-replay scorecard is the template).

## 7. Canonical-equation status
**NO canonical equation.** SimpleTES's credit (MC-return outcome weight) is NOT a formal correspondence to `costate_lambda_marginal_ds_v1` (adjoint ∂S/∂x) — asserting one would be a false-friend. If DF-2 is ever built AND backtested to beat local-λ on cross-run lever selection, a *new* equation `campaign_outcome_credit_monte_carlo_return_v1` could be drafted FORMALIZATION_PENDING with its anchor OWED to that backtest — do NOT register now (no anchor, orphan, exactly the EdgeBench `campaign_frontier_expansion_log_sigmoid_v1` GATED precedent).

Cross-refs: `project_meta_layer_above_triality_hamiltonian_control_costate_20260703` · `costate_controller_design_20260705.md` · `reference_papers_checked_not_relevant_or_watch_item_ledger_20260701` (EdgeBench sibling entry) · `litsweep_training_dynamics_control_20260705.md` (the optimal-control-of-training DRAW-FROMs this complements). Axis: all [external]/[macOS advisory] NON-PROMOTABLE. **pointer 0.19110 UNMOVED — MEANS; it moves only when a controller-selected lever lands a lower exact byte-closed row.**
