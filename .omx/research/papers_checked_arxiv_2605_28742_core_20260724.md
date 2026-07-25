# Papers-checked — alphaXiv 2605.28742 "CORE: Contrastive Reflection Enables Rapid Improvements in Reasoning" (Nasvytis, Han, Prystawski, Grant, Goodman, Fan)

UTC: 2026-07-24 · Harvested by: MAIN (Fable, inline) · $0
Evidence class: MEASURED_EXTERNAL (LLM reasoning benchmarks). Lessons-only. Target surface:
the KNOWLEDGE APPARATUS + costate organ — not the witness math.

## What the paper shows

Non-parametric improvement from verifiable rewards at tiny-n (5–100 samples): (1) CONTRASTIVE
reflection — pair each FAILED trace with a semantically-similar SUCCESSFUL one and extract the
contrast as a compact natural-language insight; (2) dual memory — rollout store + insight store
where each insight carries an EMPIRICAL UTILITY estimate updated by credit assignment;
(3) utility-aware retrieval (similarity × utility); (4) ADMISSION TEST — an insight is stored
only if it demonstrably improves its originating problem. Beats GRPO/GEPA/MemRL/episodic-RAG
in the low-sample regime; +59.9% within 350 rollouts; 35–36× fewer context tokens than
episodic traces.

## Crosswalk vs live surfaces (5 rows)

| # | Their mechanism | Our surface | Disposition |
|---|---|---|---|
| 1 | Utility scores + credit assignment on stored insights | #319 campaign_outcome_credit (DESIGNED, backtest-gated, pending) · organ activation ledger {ever_fired, last_measured_verdict} · co4 bandit-allocation DESIGN_ONLY row (AWAITING J8F ΔS-per-hour telemetry) | **FEED-INTO-#319**: CORE's utility-update + credit-assignment design is a concrete external template for the pending outcome-credit build — same tiny-n regime as the organ's n=1 starvation (#499/#434). When #319 fires (post-J8F telemetry), its design consults this memo. No new arm |
| 2 | Contrastive pairing at failure time (failed trace × matched successful trace → typed insight) | NEG↔CURE adjacency law · verdict-scope ladder · a1/a2 naive-verdict audits (periodic sweeps) | **NAMED GAP (apparatus, non-critical-path)**: we extract contrast in periodic AUDITS, not structurally at LANDING time. Candidate: when a gate/arm fails and a sibling formulation later passes the same gate, the disposition step emits a typed CONTRAST row (what differed, scoped). Queue into the knowledge-apparatus line (#346/#569 family) at a quiet boundary — fails the DISTANCE test today, recorded not built |
| 3 | Insight ADMISSION test (must improve originating problem before storage) | canonical_equations EmpiricalAnchor requirement · two-landing rule · MEASURED/DERIVED-never-guessed | **CORROBORATION**: our equations-leg already has admission (anchors); the discipline generalizes. No action |
| 4 | Compact insights ≫ episodic traces (35–36× token cut) | MEMORY.md <17KB one-line-hooks + topic-file detail + consolidation cadence | **CORROBORATION** of the index/detail split. No action |
| 5 | Non-parametric beats parametric at tiny-n | Organ maturity=_dev, advisory, backtest-gated; no learned policy in the loop | **CORROBORATION** of keeping the organ non-parametric/advisory in the n≈1 regime. No action |

## Verdict

`LESSONS_HARVESTED_INLINE; ONE_DESIGN_INPUT_ROUTED (#319 outcome-credit template);
ONE_APPARATUS_GAP_NAMED (landing-time contrastive rows — recorded, not built); NO_ARM_SPAWNED`.
Coherence check: NOVELTY — utility-tracked insight memory + landing-time contrast are new
external datums; DERIVATION — rows map to the named pending #319 build and the NEG↔CURE law;
DISTANCE — apparatus-only, nothing on the critical path (j8f unchanged); the gap is recorded
per no-signal-loss, deliberately NOT spawned per the drain-circling guard. Pointer
0.1910828242 [contest-CPU] UNMOVED — this is means.

STORES CONSULTED: #319 task row + SimpleTES draw-from memo · organ activation-ledger doctrine
(#247/#405) · co4 bandit DESIGN_ONLY receipt row · NEG↔CURE memory · #569 Cerebras-crosswalk
hardening (RecallEvidence ranking) · #499/#434 n=1 starvation rows · MEMORY.md discipline ·
papers_checked_* precedent.
