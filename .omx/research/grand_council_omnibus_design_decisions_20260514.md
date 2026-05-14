# Grand Council Omnibus Design Decisions — 2026-05-14

**Subagent:** `grand_council_omnibus_design_decisions_20260514`
**Lane:** `lane_grand_council_omnibus_design_decisions_20260514` (Phase 4)
**Parent session:** operator-session
**Inherited directives:** `recovery_session_20260514_directive_absolute_no_signal_loss_20260514` + `recursive_no_signal_loss_protocol_20260514` + `journal_lab_grade_documentation_standard_directive_20260514` + `harness_rigor_deterministic_reproducibility_directive_20260514` + `holistic_engineering_picture_seven_factor_directive_20260514` + `grand_council_tiered_parallel_plan_full_authority_20260514` + **`all_design_decisions_through_grand_council_directive_20260514` (the canonical referral)**
**Tag:** `journal_grade_v1=true`; `research_only=true`; NO score claims; NO GPU spend; deliberation + binding verdicts only.
**Operator directive verbatim 2026-05-14**: *"ask the grand council to weigh in on all design decisions"*. This is the standing-referral omnibus sweep of 14 accumulated open design decisions (15 surfaced; #5 C1↔Z5 routing is in flight under `C1-COUNCIL-RECONVENE` and is excluded).

---

## Council roster (canonical eleven + grand-bench when invoked)

Inner ten (quintet-pact extended + peer): **Shannon LEAD**, **Dykstra CO-LEAD**, **Yousfi**, **Fridrich**, **Contrarian**, **Quantizr**, **Hotz**, **Selfcomp**, **MacKay**, **Ballé**, **Time-Traveler peer** (= 11 voices). Grand bench consulted on demand: Hassabis (long-horizon strategy), Carmack (engineering shortcuts), Hinton (KD/temperature/capsules), Karpathy (arch-search), Schmidhuber (compression-as-intelligence/predictive coding), Boyd (ADMM/proximal), Tao (math omniscience), Mallat (wavelets/scattering), van den Oord (VQ-VAE/WaveNet), Filler (STC).

Contrarian SUPER-VETO available on any decision. Council is **non-conservative by charter**: only mathematical / scientific / geometric / empirical arguments accepted. "Don't change working code" / "Ship what we have" / "That's too aggressive" are NOT valid arguments.

---

## 7-factor frame (Holistic Engineering Picture directive 2026-05-14)

Every decision is scored across:

| # | Factor | Weight in this omnibus |
|---|---|---|
| 1 | Curriculum (training schedule + multi-stage) | binding for stage-ordering decisions |
| 2 | Substrate (within-class saturated vs across-class shift) | binding for promotion + composition |
| 3 | Engineering (8 harness pillars) | gating for any paid dispatch |
| 4 | Process (parallel-dispatch + NO-SIGNAL-LOSS + checkpoint) | binding for sequencing |
| 5 | Time (3600s Modal hard-kill, race window) | binding for sweep design |
| 6 | Complexity (≤350 LOC bolt-on; 30-sec review) | binding for refactor decisions |
| 7 | Spend (cost-band posterior, tier envelopes) | binding for dispatch authorization |

---

## Aggregate verdict summary

| # | Decision | Verdict | Spend impact | Wall-clock impact |
|---|---|---|---|---|
| 1 | 10-grammar registry refactor | **PROCEED (Option A: registry pattern, ≤120 LOC)** | $0 | -50% future grammar wiring time |
| 2 | F3 GTScorerCache default → True | **DEFER (default stays False; opt-in per trainer)** | $0 | none (cache already wired) |
| 3 | Z3 v1 byte-identical vs v2 priority | **PROCEED with v2 (latent-replacement) over v1** | shift $2 → $5-10 | +2-3d landing |
| 4 | Z3 → Z4 → Z5 staircase ordering | **PROCEED strict staircase; D4/D1 fire in parallel** | within Tier 1 envelope | no critical-path delay |
| 6 | Catalog #227-#231 strict-flip timing | **PROCEED warn-only 7d → STRICT atomically (per-gate)** | $0 | structural |
| 7 | MDL Tier C extension to PR106 + others | **PROCEED (PR106 next; then DP1/D4/C6 after smoke-anchors)** | $0 | +30-90 min per substrate |
| 8 | L1 → L2 substrate-class promotion criteria | **PROCEED 4-gate canonical (smoke + Tier C + 100ep + auth-eval custody)** | $0 | structural |
| 9 | Provider routing per dispatch class | **PROCEED (Modal T4 default smokes; Vast.ai 4090 fulls; Lightning A100 long-burns)** | -15-25% / dispatch | none |
| 10 | Phase B-2 / Wave 2 sweep design | **PROCEED 7-substrate fan-out within $15 cap** | $9-15 | ~12h harvest |
| 11 | DARTS-SuperNet C7 Tier 3 timing | **DEFER (gate behind Z3+Z4 anchors; reactivate Q3-2026)** | -$100-300 saved this wave | none |
| 12 | OSS announcement timing | **PROCEED with rc1 push now; v0.2.0 stable gated on medal-band trigger** | $0 | community signal NOW |
| 13 | F3 backport sister-trainer wave | **PROCEED (vq_vae flag-declare immediate; PDP substrate-side opt-in)** | $0 | ~2-3h editor work |
| 14 | MDL Tier D / Tier E future extension | **PROCEED (Tier D = scorer-conditional class transition; Tier E = predictive-coding residual)** | $0 | research-design hook |
| 15 | Sister-substrate parser P2/P3 sequencing | **PROCEED batched (P2: 6 substrates 1 wave; P3: 11 substrates split into 2 waves)** | $0 | parser-coverage canvas |
| 16 | v0.2.0-rc1 vs v0.2.0 stable timing | **PROCEED rc1 now; stable on first medal-band [contest-CUDA] or 30d quiet** | $0 | aligned with #12 |

**Net:** 13 PROCEED + 2 DEFER (#2 GTScorerCache default; #11 DARTS-SuperNet). Contrarian SUPER-VETO **NOT invoked** on any decision (3 challenges issued and addressed by math/empirical).

---

## Decision 1: 10-grammar registry refactor in `tac.analysis.hnerv_packet_sections`

**Context.** IBPS1-PARSER-WAVE-P0 op-routable #5: the analysis module currently has 10 grammar branches via if/elif over substrate class tokens. Adding a new grammar (C6 IBPS1 just landed at commit `100e9c585`) requires another branch. Question: refactor to a registry pattern (per-grammar plug-in dispatch table) vs leave the if/elif.

**Options.**
- **A** Registry pattern with explicit dispatch table (~80-120 LOC change; new grammar = 5-10 LOC plug-in).
- **B** Keep if/elif (current); add a comment block listing all 10.
- **C** Defer; revisit when adding grammar #11 (next substrate-class shift lands).

**11-voice positions.**

- **Shannon LEAD**: Option A. The if/elif dispatch is structurally identical to a finite-state lookup; making the lookup explicit eliminates the constant cognitive-overhead cost. Per `[Shannon 1948]` channel-capacity framing, the registry IS the codec front-end's parser-section manifest in code form. Refactor cost is bounded.
- **Dykstra CO-LEAD**: Option A. The Dykstra-feasibility argument is **per-grammar isolation**: a registry pattern makes each grammar's section-role manifest independently testable. If/elif couples them by control flow — adding grammar 11 risks bug-class spread to grammars 1-10 (per CLAUDE.md "META-meta finding from a8bc7e79's proactive sweep: bug classes have 6-7× spread"). Registry isolates the spread.
- **Yousfi**: Option A. From a contest-scorer audit perspective: when a new grammar lands and you want to verify it doesn't leak SegNet/PoseNet weights into the section roles, you want **one place** to look. Registry centralizes the audit surface.
- **Fridrich**: Option A, modulo budget. Inverse-steganalysis discipline: keep the parser narrow + auditable. Registry is narrower per grammar. But the LOC budget must stay ≤120 (HNeRV parity lesson 7 ≤350 ceiling; this is a tooling module so even ≤120 is generous).
- **Contrarian**: **CHALLENGE Option A**. "Refactor before need" is a bias. With 10 grammars already shipped, what's the empirical evidence the if/elif has caused a real bug? **Response (Shannon)**: not refactor-before-need; refactor at the **N+1 boundary** where each new grammar carries marginal cost of 1 branch + N test-touch points. The cost is the **on-going future maintenance**, not the current state.
- **Quantizr**: Option A. From the competitive-intelligence side: every leaderboard winner has touched the grammar surface to add a new substrate (Quantizr 0.33 hnerv_lc; PR101 hnerv_lc_v2; PR106 latent-sidecar; etc). The registry is the **public-PR-intake surface** when public PRs land that we want to deconstruct. Make it pluggable.
- **Hotz**: Option A, but cap the change at ~80 LOC. Carmack-style: "if it takes more than a day to refactor, you're doing it wrong." Keep the dispatch dict literal + 10 small `parse_*_grammar(...)` functions.
- **Selfcomp**: Option A. The grayscale-LUT codec's `pack_*` and `unpack_*` functions are already in this registry pattern (his block-FP analog mask paradigm). Sister consistency.
- **MacKay**: Option A. MDL framing: the registry is **shorter description** of the parser. Per `[Rissanen 1978]`, the canonical description length of the parser-section manifest is minimized by factoring out the dispatch.
- **Ballé**: Option A. The Ballé hyperprior side-info parser is already plug-in in `compressai`. Sister consistency with the public-frontier codec ecosystem.
- **Time-Traveler peer**: Option A, but **scope-restrictive amendment**: the registry should expose a `register_grammar(class_token, parser_fn, section_role_map)` API rather than a hard-coded dict. Future predictive-receiver substrates need to register their grammars at import time, not at module-edit time.

**Cross-debate.**
- Contrarian → Shannon: "If/elif works. Refactor cost is real ~3-5h. Why now?" Shannon: "Cost amortizes over N future grammars. Empirical N already 10; expected next 5-10 (Tier 1+2 substrates pre-registered)."
- Hotz → Time-Traveler: "register_grammar at import time risks circular imports." Time-Traveler: "Late binding via lazy registration sidesteps this; canonical pattern in compressai."

**Vote tally**: **10 of 11 for Option A** (Contrarian challenge → addressed → joins on amendment). Time-Traveler's `register_grammar` API amendment **adopted**. Contrarian SUPER-VETO **not invoked**.

**Binding verdict**: **PROCEED Option A**. Refactor to registry pattern with `register_grammar(class_token, parser_fn, section_role_map)` API; ≤120 LOC budget. Implementation gate: **same-commit-batch as the next grammar addition** (P2/P3 wave) to avoid drive-by churn.

**Math + rationale.** Description length of parser (in LOC + cognitive load):
- if/elif current: 10 × ~50 LOC = 500 LOC + 10 × test-touch coupling
- Registry: 80-120 LOC dispatch table + 10 × ~30 LOC parser fn = ~380-420 LOC + isolated tests

Net description-length saving: ~100-200 LOC + **isolation invariant** (per-grammar tests don't risk regression of unrelated grammars). Per Rissanen MDL: the registry minimizes K(parser).

**Reactivation criteria.** None — this is a structural improvement. If the refactor itself introduces a regression (parser test failures), revert + revisit.

**Operator-routable checkpoints (3):**
1. Refactor PR lands in P2/P3 wave commit-batch ($0; ~3-5h editor work).
2. The 10 existing grammar parsers each remain ≤30 LOC post-refactor.
3. Test coverage: every grammar gets one `test_<grammar>_parser_roundtrip.py` mirror in `src/tac/analysis/tests/`.

---

## Decision 2: F3 GTScorerCache canonical default flip to True

**Context.** F3 GTScorerCache landing 2026-05-14: `score_pair_components_with_cache` wired into 7 substrate trainers (sane_hnerv + balle_renderer + 5 NeRV-family + cool_chic + self_compress_nn + hybrid_renderer_residual). Question: flip canonical default in `tac.training_optimization.scorer_cache.score_pair_components_with_cache(..., enable=False → True)`.

**Options.**
- **A** Flip default to True (cache ON by default).
- **B** Keep default False (cache OFF; explicit opt-in per trainer).
- **C** Per-substrate default in the substrate's recipe YAML.

**11-voice positions.**

- **Shannon LEAD**: Option B. The cache is correct **mathematically identical to direct GT forward** only when the GT decode is deterministic and the cache key fully captures the inputs (frame indices + scorer weight sha). If a future substrate introduces non-determinism (e.g., dropout in GT decode, augmentation), the cache silently breaks identity. Opt-in keeps the invariant explicit.
- **Dykstra CO-LEAD**: Option B. The "opt-in is structurally safer" feasibility argument: per CLAUDE.md "Bugs must be permanently fixed AND self-protected against", flipping default-True risks a sister-substrate that didn't backport to silently use stale cache. Default-False forces the trainer author to think about cache invariants.
- **Yousfi**: Option B. From contest-scorer audit: every substrate's auth-eval relies on the GT decode being byte-deterministic. A cache MISS → recompute is safe; a cache HIT with wrong key → silently wrong score. Default-False is fail-safe.
- **Fridrich**: Option B. UNIWARD discipline (don't perturb regions you didn't intend to perturb): default-True perturbs the trainer behavior for every substrate without opt-in. That's a forbidden-pattern shape.
- **Contrarian**: Option A. "Speed matters. Per F3 landing memo, ~50% per-step scorer compute saving. Multiply across 14 substrates × 10 trainings = real wall-clock. Don't be conservative." **Response (Yousfi)**: "We're not being conservative — we're being correct. Default-False means each trainer author writes ONE line `enable=True` after they verify the cache invariant holds. That's not slow."
- **Quantizr**: Option B. From competitive-intelligence: leaderboard winners never share a hidden state by default. Quantizr's KL distillation is opt-in per stage. Default-False is the canonical leaderboard pattern.
- **Hotz**: Option B with caveat. "Default-False, but the cache-enable line should be a 1-token bool in the recipe YAML, not a 5-line trainer block. Make opt-in cheap." **Adopted** as Option C amendment.
- **Selfcomp**: Option B. Block-FP 1.017-bpw weight self-compression is **always** opt-in per substrate; never a global flag. Sister discipline.
- **MacKay**: Option B. MDL argument: the cache is a side-info channel that needs explicit declaration. Default-False respects the contract.
- **Ballé**: Option B. The compressai entropy-bottleneck is opt-in per encoder; the parallel cache pattern.
- **Time-Traveler peer**: Option B with **enhancement**: build a `tools/probe_gt_scorer_cache_invariant.py` that runs a 10-frame smoke comparing cached vs uncached scores and refuses if they diverge by > 1e-9. THIS probe is the canonical regression guard. Then per-substrate opt-in is structurally safe.

**Cross-debate.**
- Contrarian → Yousfi: "If the cache is mathematically identical, why not flip?" Yousfi: "Mathematically identical ASSUMING the cache key is correct. The bug class is the key-construction logic, not the cache itself." Contrarian: "Concede — Option B preserves the audit surface."
- Hotz → Time-Traveler: "Cost of the probe?" Time-Traveler: "~$0 (10 frames CPU), ~30s wall-clock. Run it as a STRICT preflight gate when GT decode changes."

**Vote tally**: **11 of 11 for Option B** (Contrarian challenge resolved). Time-Traveler probe enhancement **adopted**. Contrarian SUPER-VETO **not invoked**.

**Binding verdict**: **DEFER default flip; keep default False (opt-in per trainer)**. Build `tools/probe_gt_scorer_cache_invariant.py` as the structural safeguard for future default-flip consideration. Reactivation criteria: after the probe lands AND ≥3 substrate trainers have run 100ep auth-eval comparing cached-vs-uncached identical (within 1e-9), the council reconvenes on this decision.

**Math + rationale.** The cache identity holds iff `cache_key = sha256(frame_idx || scorer_weight_sha || decode_config_sha)` and `cache_value = GT_forward(frame_idx, scorer)`. If any decode-config is omitted from the key (e.g., new augmentation flag), cache hit returns stale value with the wrong gradient. Default-False is the **structural fail-closed**.

**Reactivation criteria.** Probe lands → 3 substrates' 100ep auth-eval comparison agrees within 1e-9 → council revisits.

**Operator-routable checkpoints (3):**
1. Probe build ($0; ~2h editor work) → lands at `tools/probe_gt_scorer_cache_invariant.py`.
2. Three substrate trainers run 100ep dual cached/uncached on smoke (~$2-3 total Modal A100).
3. Council reconvenes on default-flip after probe + 3-substrate evidence (no GPU spend for the deliberation).

---

## Decision 3: Z3 v1 byte-identical strategy vs v2 priority

**Context.** Z3 Ballé hyperprior bolt-on landing 2026-05-14: v1 strategy = leave A1 latent_blob bytes UNCHANGED and add a parallel hyperprior side-channel; v2 strategy = REPLACE A1 latent_blob with hyperprior-coded version. Question: which path lands first?

**Options.**
- **A** v1 first (byte-identical on A1; demonstrates the runtime contract; predicted ΔS ≈ 0 since no bytes change).
- **B** v2 first (latent replacement; predicted ΔS = -0.0003 to -0.0009 per `[Ballé 2018 §IV.A]` 5-15% byte savings).
- **C** Both in parallel (separate substrates, separate archives).

**11-voice positions.**

- **Shannon LEAD**: Option B. v1 ships zero ΔS by construction — it's an integration test for the hyperprior wiring, NOT a score-lowering substrate. The Tiered Parallel Plan's Z3 line predicts ΔS ∈ [-0.0003, -0.0009] **only under v2 (latent replacement)**. Per CLAUDE.md "Long-burn score-lowering campaign default", the campaign field "score-lowering hypothesis" requires real bytes changing.
- **Dykstra CO-LEAD**: Option B + concurrent v1 as test scaffold. Feasibility: v2's archive grammar must be derived from the Ballé hyperprior's bit-rate model; v1 is the **integration test** that proves the parser can read both legacy A1 bytes AND the new hyperprior side-info. So v1 = test fixture, v2 = real score path.
- **Yousfi**: Option B. From contest-CUDA / contest-CPU axis discipline: a v1 ΔS=0 anchor is wasted dispatch ($2 Modal smoke for zero signal). v2's predicted ΔS is small (-0.0003 to -0.0009) BUT it's the only actionable score change.
- **Fridrich**: Option B. UNIWARD analogy: the hyperprior **must be where the score-distortion live**, which is in the latent bytes. v1 doesn't perturb the score-bytes.
- **Contrarian**: **CHALLENGE Option B**. "v2 is the harder engineering. v1 demonstrates the wire-up cheaply. Land v1 first to de-risk." **Response (Shannon)**: "We've already de-risked the parser via P0 (parser-WAVE-P0 landed 2026-05-14 commit `7fb4d079a`). v1 is now redundant. Direct-to-v2 is correct sequencing." **Contrarian withdraws challenge**.
- **Quantizr**: Option B. From competitive-intelligence: leaderboard winners ship ACTUAL score-changing substrates. v1 = integration milestone (good for OSS demo); v2 = leaderboard delta.
- **Hotz**: Option B + 2-day budget cap. "If v2 takes more than 2 dev-days post-substrate-build, fall back to v1 + dispatch + retry." Engineering pragma.
- **Selfcomp**: Option B. The grayscale-LUT codec went directly to score-changing bytes; never had a v1 byte-identical phase. Sister discipline.
- **MacKay**: Option B. MDL: v1's K=0 (zero bytes changed) is uninformative as a Bayesian update. v2's K = -Δb is the actionable posterior anchor.
- **Ballé**: Option B with **strong endorsement**. The hyperprior IS the entropy-bottleneck factorization per `[Ballé 2018]`. The whole point is the **rate model** captures non-trivial structure in the latent that a flat code misses. The 5-15% savings prediction is the published lower bound on natural image latents; comma2k19 driving video is in-distribution.
- **Time-Traveler peer**: Option B. Predictive-receiver staircase math: Z3 is Step 1; the Step-1 ΔS must be nonzero (even if small) for Step 2 (Z4 cooperative-receiver) to compose meaningfully. v1's zero ΔS does NOT seed the staircase posterior.

**Cross-debate.**
- Contrarian → Hotz: "v2 timeline?" Hotz: "Per Z3 landing memo, the hyperprior network is ~1764 params (small). Training is ~30 min smoke / ~4h full on Modal A100. Replacement integration is ~1 day editor work. Total ~3-4 days from substrate ready."
- Ballé → Shannon: "v2's predicted ΔS [-0.0003, -0.0009] is small. Is this worth a Tier-1 dispatch?" Shannon: "Yes — per Z1 ablation A1 is 99.29% MDL-saturated within-class. Any positive ΔS from within-class refinement is bounded near this regime. Z3 v2 is calibrating the within-class slope."

**Vote tally**: **11 of 11 for Option B**. Contrarian challenge resolved (parser P0 already landed). Contrarian SUPER-VETO **not invoked**.

**Binding verdict**: **PROCEED Option B (v2 latent-replacement)**. v1 retired as redundant (P0 parser landing supersedes its scaffold purpose). Hotz's 2-day cap accepted; fallback path is **regenerate-A1-anchor-with-Ballé-latent** if v2 integration takes >2 dev-days.

**Math + rationale.** Per `[Ballé 2018 §IV.A]`, entropy-bottleneck + scale hyperprior saves 5-15% bits over flat factorized prior on natural images. A1's latent_blob is 8528 bytes (per Catalog #226 archive_sha256 `87ec7ca5...492b5`). 5-15% savings → 426-1280 bytes saved → ΔS = -25·Δb/N = -25·{426,1280}/37,545,489 = `[-0.0003, -0.0009]` `[predicted; Ballé 2018 §IV.A bound]`.

**Reactivation criteria.** If v2 smoke shows ΔS regression (worse than A1 anchor by >0.001), DEFER and council reconvenes; investigate whether A1's latent has unusual statistics (low entropy, learned-quantized) that the hyperprior over-fits. Build `tools/probe_ballé_hyperprior_overfit_diagnostic.py`.

**Operator-routable checkpoints (5):**
1. Z3 v2 substrate trainer completes (per existing substrate plan; ~$2 smoke + $3-5 full on Modal A100; ~5h wall-clock total).
2. v2 smoke ΔS in [-0.0001, -0.001]: green-light full.
3. v2 smoke ΔS > +0.001 (regression): DEFER per reactivation criteria.
4. v1 retired in lane registry (mark `lane_z3_balle_v1_byte_identical_20260514` if pre-registered → mark `three_clean_review` with "council retired: redundant with P0 parser").
5. Hotz fallback path activated if dev-time >2 days: regenerate A1 with Ballé latent inline as A1-Ballé anchor.

---

## Decision 4: Z3 → Z4 → Z5 staircase ordering

**Context.** TIME-TRAVELER landing op-routable #2: Z3 (Ballé hyperprior bolt-on) → Z4 (cooperative-receiver loss) → Z5 (predictive-coding world model) staircase. Question: strict sequential vs partial parallel vs opportunistic?

**Options.**
- **A** Strict staircase (Z3 anchors before Z4 starts; Z4 anchors before Z5).
- **B** Opportunistic parallel (all three substrates train concurrently; harvest informs ordering).
- **C** Partial parallel: Z3+Z4 parallel; Z5 gated on Z3 OR Z4 anchor.

**11-voice positions.**

- **Shannon LEAD**: Option C (partial parallel). Z3 and Z4 are mathematically independent — Z3 changes the **rate model** of an existing latent; Z4 changes the **loss formulation** without changing archive grammar. They can train in parallel. Z5 is a **world-model substrate** (architecture class shift) that needs Z3 OR Z4 as a wired baseline to compose against.
- **Dykstra CO-LEAD**: Option C. Feasibility: Z3 and Z4 occupy disjoint feasibility sets (rate-model space vs loss-formulation space). Parallel projection onto each is the correct Dykstra alternating-projections algorithm. Z5's feasibility set requires both Z3 and Z4 priors **but only one needs to land as anchor first**.
- **Yousfi**: Option C. Contest-scorer-audit: Z3 and Z4 produce orthogonal evidence; running them in parallel doubles the per-day signal harvest without doubling complexity.
- **Fridrich**: Option C with timing constraint. UNIWARD inverse-steganalysis: Z4's cooperative-receiver loss is the **right** loss formulation per `[Atick-Redlich 1990]`. If Z4 lands first with strong ΔS, Z3 might be subsumed. So parallel run, but Z4's signal could retire Z3 if it dominates.
- **Contrarian**: **CHALLENGE Option C**. "Parallel = double cost. Why not strict?" **Response (Dykstra)**: "Each lane is Tier-1 ~$5-10. Parallel doubles spend to ~$10-20 BUT halves wall-clock to ~12h. Per CLAUDE.md 'Race-mode rigor inversion' the operator's directive 'all tiers in parallel' supersedes the conservative sequence. Contrarian: concedes parallel sequencing is operator-mandated."
- **Quantizr**: Option C. From competitive-intelligence: every leaderboard winner's PR family had 2-3 substrates training simultaneously in the 4h race window.
- **Hotz**: Option C with 12h hard timeout on Z3+Z4. "If neither lands ΔS within 12h, both retire to research-only; Z5 promoted regardless."
- **Selfcomp**: Option C. The block-FP 1.017-bpw and grayscale-LUT and 94K-param SegMap shipped in parallel within ONE PR (#56). Sister discipline.
- **MacKay**: Option C. The Bayesian posterior update is **stronger** when two parallel substrates give independent evidence. Sequential ordering biases the posterior.
- **Ballé**: Option C, but with **strong recommendation Z3 fires first by 1-2 hours**. Z3's hyperprior is well-understood; expected wall-clock for smoke is 30 min. Z4's cooperative-receiver loss is novel-to-the-codebase; expected wall-clock smoke is harder to predict. Front-loading Z3 gives early signal for Z4 tuning.
- **Time-Traveler peer**: Option C. Predictive-receiver staircase math: Step 1 (Z3) and Step 2 (Z4) can both train in parallel because they're independent operators on the same posterior; Step 3 (Z5) is the cooperative composition of Steps 1+2 and needs at least one to anchor first. **D4 (Wyner-Ziv frame-0) is sister to Z3/Z4 in this fan-out — D4 fires in parallel with Z3+Z4 since it's an across-class substrate**.

**Cross-debate.**
- Contrarian → Hotz: "12h timeout — what's the fallback?" Hotz: "If Z3 OR Z4 lands ΔS < -0.0001, fire Z5. If both fail, retire both, fire Z5 anyway as the across-class fallback."
- Time-Traveler → Ballé: "Z3 first by 1-2h or full parallel?" Ballé: "Concurrent fan-out; Z3 should COMPLETE smoke first by 1-2h due to maturity, but FIRE simultaneously."

**Vote tally**: **11 of 11 for Option C (partial parallel)**. D4 inclusion in the fan-out **adopted** (Time-Traveler amendment). Contrarian SUPER-VETO **not invoked**.

**Binding verdict**: **PROCEED Option C**: Z3 + Z4 fire concurrently (Tier 1 envelope $5-10 each); D4 fires concurrently as the across-class sister. Z5 gated on at least one of Z3/Z4 anchoring a non-trivial ΔS within 12h.

**Math + rationale.** Per the Tiered Parallel Plan's compound trajectory:
- Z3: ΔS [-0.0003, -0.0009] (rate-model refinement)
- Z4: ΔS [-0.005, -0.010] (loss reformulation per `[Atick-Redlich 1990]`)
- D4: ΔS [-0.025, -0.045] (across-class Wyner-Ziv per `[Wyner-Ziv 1976]`)
- Z5: ΔS [-0.030, -0.060] (across-class predictive-coding per `[Rao & Ballard 1999; Ha & Schmidhuber 2018]`)

Parallel fan-out across Z3+Z4+D4 has expected wall-clock 12h; sequential would be 24-36h. Per CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first" non-negotiable + operator directive "all tiers in parallel", parallel is mandatory.

**Reactivation criteria.** If Z3 AND Z4 BOTH land ΔS > -0.0001 (no improvement or regression), the within-class hypothesis is partially falsified and council reconvenes to question whether the staircase trajectory needs to skip directly to Z5 or D4. (Note this is research-direction-reactivation, NOT a kill.)

**Operator-routable checkpoints (5):**
1. Z3 v2 + Z4 + D4 smokes fire concurrently within Tier 1 envelope $15-30 (3 × $5-10).
2. 12h harvest window; harvest all 3 artifacts regardless of rc per CLAUDE.md "HARVEST OR LOSE".
3. Z5 fire decision at 12h based on which of Z3/Z4/D4 anchored.
4. If all three regress: council reconvenes (no kill, just re-strategize).
5. Tier 2 dispatcher upgrade (multi-stage curriculum + Phase 3 Z5+C6 composition) gated on at least one Tier 1 success.

---

## Decision 6: STRICT gate strict-flip timing for Catalog #227-#231

**Context.** CATALOG-227-231-WAVE in flight: 5 strict gates landing with various warn-only vs immediate STRICT decisions per gate. Per CLAUDE.md "Strict-flip atomicity rule", STRICT-flip should land in the SAME commit-batch as the canonical fix when live count = 0; otherwise warn-only with planned strict-flip.

**Options.**
- **A** Each gate flips strict atomically per CLAUDE.md rule (no batch warn-only delays).
- **B** All 5 land warn-only first; STRICT flip in 7-day cadence after backfill sweep.
- **C** Per-gate decision based on live count at landing.

**11-voice positions.**

- **Shannon LEAD**: Option C. Per CLAUDE.md "Strict-flip atomicity rule" verbatim: *"If the fix subagent achieves live count = 0 in the same landing, the strict-flip should land in the SAME commit-batch."* So per-gate: live count 0 → atomic STRICT; live count > 0 → warn-only with backfill plan.
- **Dykstra CO-LEAD**: Option C. The Dykstra-feasibility argument is **per-gate**: each gate's accept set is convex; live count 0 means the codebase is already inside the set. Forcing premature STRICT on a non-zero live count is a non-convex jump that breaks builds.
- **Yousfi**: Option C. Audit symmetry: warn-only purgatory destroys the "what's strict?" operator-facing manifest. Cataolog #176 (`check_strict_preflight_callsites_have_claude_md_catalog_row`) and #185 (`check_strict_flipped_catalog_entries_have_live_count_zero`) jointly enforce the atomic-flip contract.
- **Fridrich**: Option C. Each gate's strictness IS a "where to plant the safe-set boundary" decision. Per-gate is correct.
- **Contrarian**: Option C **with audit**. "Make sure every warn-only gate has a documented backfill plan with target strict-flip date." **Adopted** as enforcement amendment.
- **Quantizr**: Option C. From competitive-intelligence: every winning PR has a "fix + protect" landing pair. The protection (STRICT gate) must land atomically when feasible.
- **Hotz**: Option C. 30-second-review discipline: warn-only purgatory is a debt that compounds. Pay it down atomically when live count = 0.
- **Selfcomp**: Option C. Block-FP 1.017-bpw QAT codec's strict-bounds checks all landed atomically. Sister discipline.
- **MacKay**: Option C. MDL: warn-only adds an extra "is this strict?" dimension to the catalog; atomic STRICT removes the dimension when feasible.
- **Ballé**: Option C. compressai's rate-distortion check is strict-from-byte-one. Sister discipline.
- **Time-Traveler peer**: Option C, with **enforcement**: every warn-only gate landing MUST cite the planned strict-flip trigger condition in its CLAUDE.md catalog row (e.g., "STRICT-flip pending: legacy backfill sweep on 5 files; ETA <date>"). Catalog #183 (`check_legacy_allowlist_backfill_cadence_ledger_current`) is the structural enforcement.

**Cross-debate.**
- Contrarian → Time-Traveler: "Is the backfill cadence ledger enough?" Time-Traveler: "Yes — per Catalog #183, if a gate sits warn-only > 30 days without ledger movement, the ledger gate itself fails."
- Hotz → Shannon: "What about #229 specifically (which I heard sits at live count > 0)?" Shannon: "If live count > 0 at landing, warn-only is correct per CLAUDE.md rule. Strict-flip waits for backfill."

**Vote tally**: **11 of 11 for Option C**. Time-Traveler enforcement + Contrarian audit amendments **adopted**.

**Binding verdict**: **PROCEED Option C** (per-gate decision):
- Atomic STRICT-flip if live count = 0 at landing.
- Warn-only with documented backfill plan + target STRICT-flip date if live count > 0.
- Catalog #183 (backfill cadence ledger) is the enforcement.

**Math + rationale.** Per CLAUDE.md "Strict-flip atomicity rule" + Catalog #176/#185: the atomic-flip contract is structurally optimal. Each gate's strictness boundary is convex in the codebase state; live count 0 IS the precondition for atomic STRICT.

**Reactivation criteria.** None — this is a process discipline, not a research question.

**Operator-routable checkpoints (3):**
1. Each of #227-#231 landing records its live count at landing in the CLAUDE.md catalog row.
2. Each warn-only gate cites its backfill plan + target STRICT-flip date.
3. Backfill cadence ledger (Catalog #183) maintains ≤30-day rolling movement.

---

## Decision 7: MDL Tier C extension to PR106_latent_sidecar + others

**Context.** MDL-ABLATION-TIER-C op-routable #3: tier-C extends the scorer-conditional MDL ablation to substrates beyond A1. Initial commit `100e9c585` added IBPS1 (C6) grammar. Question: which substrates next, and in what order?

**Options.**
- **A** PR106_latent_sidecar next (closest to A1's HNeRV-family; calibrates within-class slope).
- **B** DP1 (driving-prior; cross-domain reference).
- **C** D4 / D1 (across-class shift; tests Z1 hypothesis).
- **D** All in parallel.

**11-voice positions.**

- **Shannon LEAD**: Option A then C in sequence. PR106 calibrates within-class — confirms A1's 99.29% saturation generalizes to sibling HNeRV-family. Then D4/D1 (across-class) tests the across-class density hypothesis (predicted < 50% per Z1). DP1 is cross-domain (driving prior) — useful but not on the critical path.
- **Dykstra CO-LEAD**: Option A then C. Feasibility: within-class density posterior gains tight error bars from PR106 (sister anchor to A1); across-class density posterior gains separation from D4/D1.
- **Yousfi**: Option A. Contest-scorer-audit: PR106 is the **next-closest medal-band anchor** to A1 (PR106 r2 score 0.197 vs A1 0.193). Calibrating it is the cheapest empirical extension.
- **Fridrich**: Option A then C. From inverse-steganalysis: PR106's latent_sidecar uses the same coding scheme as A1; D4 uses a different code (Wyner-Ziv side info). Stratification by scheme.
- **Contrarian**: **CHALLENGE Option A**. "Why not D4 first? It tests the ACROSS-class hypothesis, which is the score-lowering hypothesis. PR106 just confirms what we already suspect." **Response (Shannon)**: "PR106 is the **null hypothesis** test. If PR106's density is 99% like A1, the within-class saturation generalizes. If it's 80%, the saturation is A1-specific. We need this signal BEFORE betting across-class." **Contrarian concedes**.
- **Quantizr**: Option A. From competitive-intelligence: PR106 is the actively-competing sister anchor; understanding its density profile informs future PR106-class bolt-ons (Catalog #226 sidecar packets).
- **Hotz**: Option A with $0 estimated cost. "Tier C extension is editor-side; no GPU. ~30-90 min per substrate. Sequential is fine."
- **Selfcomp**: Option D **partial** — within-class anchors (PR106) in parallel with cross-domain (DP1); across-class (D4/D1) gated on within-class results.
- **MacKay**: Option A then C. Bayesian: PR106 is the within-class sister; running it first tightens the within-class posterior, then across-class measurements yield maximum information gain.
- **Ballé**: Option A. PR106 is the canonical sidecar pattern; understanding its density profile is foundational.
- **Time-Traveler peer**: Option A then C (in that order) **with** parallel DP1 as cross-domain control. The 7-factor frame's substrate axis: PR106 anchors within-class certainty; D4 anchors across-class certainty; DP1 anchors cross-domain certainty. All three are needed but sequenced.

**Cross-debate.**
- Contrarian → Selfcomp: "Parallel saves time?" Selfcomp: "Tier C is $0; parallel = same wall-clock as sequential editor-wise. Sequential is cleaner for the Bayesian update."
- Time-Traveler → Shannon: "DP1 first or alongside?" Shannon: "Alongside PR106 — they're independent (PR106 = within-class HNeRV; DP1 = cross-domain prior). Across-class (D4/D1) waits for within-class signal."

**Vote tally**: **11 of 11 for Option A → C ordering**. DP1 parallel-with-PR106 **adopted** (Time-Traveler/Selfcomp amendment). Contrarian SUPER-VETO **not invoked**.

**Binding verdict**: **PROCEED with the following sequencing**:
1. Tier C extension: PR106_latent_sidecar (~30-60 min editor; $0 GPU).
2. Tier C extension: DP1 (parallel with PR106; ~30-60 min editor; $0 GPU).
3. Across-class extensions (D4 / D1 / C6 — C6 already landed): gated on PR106 + DP1 evidence (~30-60 min each; $0 GPU).
4. Council reconvenes if PR106 density deviates from A1 by >5% (suggests A1-specific saturation, not within-class universal).

**Math + rationale.** Per `[Rissanen 1978]` MDL framing + Z1 ablation methodology, scorer-conditional MDL density `ρ = 1 - K(payload | scorer) / K(payload)`. A1: ρ = 0.9929. Hypothesis: ρ_HNeRV ≈ 0.95-0.99 (within-class saturated); ρ_across_class < 0.7 (room for class-shift extraction). PR106 + DP1 test the within-class generalization; D4/D1 test the across-class separation.

**Reactivation criteria.** If PR106 density deviates >5% from A1 (e.g., 0.94 instead of 0.99), council reconvenes on within-class universality. If DP1 density is >0.99 (cross-domain trivially saturated), DP1 substrate path is downgraded (no class-shift potential).

**Operator-routable checkpoints (5):**
1. PR106 Tier C extension lands ($0; ~30-60 min editor).
2. DP1 Tier C extension lands ($0; ~30-60 min editor).
3. Council reconvenes if either deviates >5% from A1 (no GPU spend).
4. Across-class extensions (D4/D1) gated on within-class signal.
5. Updated Z1 v2 ledger memo `feedback_z1_tier_c_extension_pr106_dp1_landed_YYYYMMDD.md`.

---

## Decision 8: L1 → L2 substrate-class promotion criteria (canonical)

**Context.** C6 5ep ACROSS-CLASS architectural finding (C6 substrate trains structurally but smoke shows mixed posterior). Question: what evidence formally promotes a lane from L1 (impl_complete) to L2 (real_archive_empirical)?

**Options.**
- **A** Tier C alone (MDL density measurement).
- **B** Tier B (100ep auth-eval anchor) + Tier C.
- **C** Tier B + Tier C + converged-substrate Tier C (post-training MDL).
- **D** 4-gate canonical: (i) smoke green (rc=0); (ii) Tier C measured; (iii) 100ep auth-eval with byte-deterministic archive; (iv) custody validated per Catalog #127.

**11-voice positions.**

- **Shannon LEAD**: Option D. Per CLAUDE.md "Forbidden score claims" + Catalog #127 (`check_authoritative_tag_requires_custody_metadata`) + lane maturity 7-gate definition: L2 requires both `impl_complete` AND `real_archive_empirical`. The 4-gate canonical IS the operationalization.
- **Dykstra CO-LEAD**: Option D. Feasibility: each gate is a separate convex set; intersection is the L2 promotion set. All four must be satisfied.
- **Yousfi**: Option D. Contest-scorer-audit: the L2 promotion gates the byte-level archive's auth-eval custody. All four gates are structurally necessary for the archive to be reproducible + comparable.
- **Fridrich**: Option D. UNIWARD: each gate guards a distinct invariant (smoke → architecture-correct; Tier C → measurement-correct; auth-eval → score-correct; custody → audit-correct).
- **Contrarian**: **CHALLENGE Option D**. "4 gates is heavy. Why not 2 (smoke + auth-eval)?" **Response (Yousfi)**: "Tier C measures density INDEPENDENTLY of architecture; it catches a substrate whose smoke passes but is within-class saturated (which would NEVER produce class-shift score). Catalog #127 custody catches the rest. The 4 are non-redundant."
- **Quantizr**: Option D. From competitive-intelligence: leaderboard winners have all 4 implicitly. Codifying them is just making the existing standard explicit.
- **Hotz**: Option D with cost cap. "Each gate must be <1h editor + <1h GPU. Smoke + Tier C are <30 min each; 100ep auth-eval is ~4-12h GPU; custody is editor. Total per substrate: ~4-12h GPU + ~2h editor."
- **Selfcomp**: Option D. The block-FP 0.38 substrate (PR #56) passed all 4 implicitly.
- **MacKay**: Option D. MDL: each gate provides a separate term in the description-length budget. Removing any one weakens the L2 anchor.
- **Ballé**: Option D. compressai's evaluation gates include all 4 analogs (smoke-test on toy dataset, kodak-eval, KOdak-roundtrip, custody-verified).
- **Time-Traveler peer**: Option D, with **5th-gate enhancement**: probe-disambiguator (Catalog #125 hook #6) for any substrate with 2+ defensible interpretations. The 4-gate canonical + Catalog #125 hook = full operationalization.

**Cross-debate.**
- Contrarian → Yousfi: "Tier C is $0 measurement; can a substrate be L2 without it?" Yousfi: "No — Tier C IS the falsification anchor. Without it, you can't say 'this substrate is across-class' empirically."
- Hotz → Time-Traveler: "5th gate adds cost." Time-Traveler: "Only when 2+ interpretations exist; if not, the gate is auto-satisfied (N/A trivially)."

**Vote tally**: **11 of 11 for Option D (4-gate canonical)**. Time-Traveler 5th-gate amendment **adopted as conditional**. Contrarian SUPER-VETO **not invoked**.

**Binding verdict**: **PROCEED Option D**. The L1 → L2 substrate-class promotion canonical:
1. Smoke green (rc=0 + auth-eval JSON parseable).
2. Tier C MDL density measured (`tools/mdl_scorer_conditional_ablation.py` on the smoke artifact).
3. 100ep auth-eval anchor with byte-deterministic archive (sha256 stable across re-runs).
4. Custody validated per Catalog #127 (`evidence_grade` matches axis + hardware).
5. (Conditional) Probe-disambiguator built if 2+ defensible interpretations exist.

This codifies the lane-maturity `real_archive_empirical` gate's evidence requirement.

**Math + rationale.** Per the 7-factor frame + Catalog #127 + Catalog #219 (MDL density gate):
- Smoke green: architecture correctness gate.
- Tier C: substrate-class hypothesis falsification gate.
- 100ep auth-eval: score-claim custody gate.
- Catalog #127: axis/hardware custody gate.

Each gate is **structurally non-redundant** with the others. Together they prevent the "substrate trained but never validated" + "substrate validated but within-class saturated" + "score claim without custody" failure modes.

**Reactivation criteria.** None — this is a process canonical. If a substrate fails Tier C, the substrate is NOT promoted to L2; it stays L1 with `notes: within-class saturated per Tier C density`.

**Operator-routable checkpoints (3):**
1. Tier C extension lands for next 3 candidate L2 substrates (PR106, DP1, C6).
2. Lane registry audit: any L2-claimed lane WITHOUT all 4 gates is forensically reviewed and downgraded if necessary.
3. Catalog #220 or new sister gate enforces the 4-gate canonical at lane-registry level.

---

## Decision 9: Provider routing — Modal vs Vast.ai vs Lightning per dispatch class

**Context.** Cost-band posterior accumulating evidence; provider choice per dispatch class drives cost + wall-clock.

**Options.**
- **A** Modal default for everything (single-provider simplicity).
- **B** Per-class routing: Modal T4 default for smokes (<$2); Vast.ai 4090 for fulls; Lightning A100 for long-burns.
- **C** Cost-band-posterior-driven (autopilot ranker chooses provider per dispatch).

**11-voice positions.**

- **Shannon LEAD**: Option B. Per CLAUDE.md "GPU budget and compute resources — non-negotiable" + cost-band posterior empirical anchors: Modal T4 ($0.59/h) is optimal for ≤30-min smokes; Vast.ai RTX 4090 ($0.25/h) is 4-5× faster than T4 → optimal for 4-12h fulls; Lightning A100 is optimal for very-long >12h runs (operator subscription on monthly plan).
- **Dykstra CO-LEAD**: Option B with cost-band overlay. The Pareto frontier in {$/dispatch, wall-clock} is provider-dependent; per-class routing is structurally on the frontier.
- **Yousfi**: Option B. Contest-scorer-audit: every public-PR replay anchor uses contest-CUDA T4 (the bot's reference). Smokes match the contest reference; fulls go to faster hardware.
- **Fridrich**: Option B.
- **Contrarian**: **CHALLENGE Option B**. "Why not pure Modal? Single-provider = simpler ops." **Response (Hotz)**: "Vast.ai 4090 at $0.25/h is 4-5× faster than Modal T4 at $0.59/h. For a 4h full → $1.00 on Vast vs $2.36 on Modal. Scaling across 10+ fulls = $10-15 saved. Plus Vast.ai 4090 doesn't have the Modal 3600s hard-kill."
- **Quantizr**: Option B. From competitive-intelligence: leaderboard winners used multiple providers depending on the run type.
- **Hotz**: Option B with **canonical ordering**: smoke = Modal T4 (single-provider simplicity for quick iterate); full = Vast.ai 4090 (cost + wall-clock); long-burn = Lightning A100 (operator subscription); H100 = Vast.ai or RunPod when A100 saturated.
- **Selfcomp**: Option B.
- **MacKay**: Option B with cost-band posterior anchor. Each provider's cost-band updates after each dispatch; the autopilot v2 ranker already consumes this.
- **Ballé**: Option B.
- **Time-Traveler peer**: Option B with **dynamic re-routing**: the autopilot v2 ranker should consult the cost-band posterior at dispatch time; if Vast.ai 4090 capacity is saturated, fallback to Modal A100 instead of waiting. This is in flight per Catalog #219's autopilot wiring.

**Cross-debate.**
- Contrarian → Hotz: "Multi-provider = complexity." Hotz: "Yes, but cost-saved offsets. The complexity is in `tools/operator_authorize.py` _dispatch_* functions, which are already canonicalized per Catalog #162."

**Vote tally**: **11 of 11 for Option B**. Time-Traveler dynamic re-routing **adopted**. Contrarian SUPER-VETO **not invoked**.

**Binding verdict**: **PROCEED Option B (per-class canonical routing)**:
- Smokes (≤30 min, ≤$2): Modal T4 default.
- Fulls (1-12h, $2-$15): Vast.ai RTX 4090 default; Modal A100 fallback.
- Long-burns (12h+, $50+): Lightning A100 default (operator subscription); Vast.ai H100 for race-mode urgency.
- Dynamic re-routing per autopilot v2 + cost-band posterior at dispatch time (saturation-aware).

**Math + rationale.** Cost-per-dispatch table:
| Provider | GPU | $/hr | 4h full cost | wall-clock vs T4 |
|---|---|---:|---:|---|
| Modal | T4 | $0.59 | $2.36 | 1.0x |
| Vast.ai | RTX 4090 | $0.25 | $1.00 | 0.20-0.25x (4-5× faster) |
| Modal | A100 | $1.10 | $4.40 | 0.4x |
| Lightning | A100 | sub/$0 | $0 (subscription) | 0.4x |
| Vast.ai | H100 | $1.50-1.99 | $6-8 | 0.15-0.20x |

**Reactivation criteria.** If Vast.ai 4090 cost rises >$0.40/h (capacity contraction) OR Modal T4 cost falls <$0.30/h (provider promo), re-evaluate.

**Operator-routable checkpoints (3):**
1. `tools/operator_authorize.py` honors per-class routing default (already canonical).
2. Autopilot v2 ranker consults cost-band posterior per dispatch.
3. Cost-band posterior updates after each dispatch (already canonical per Catalog #175 + #177).

---

## Decision 10: Phase B-2 / Wave 2 sweep design (6-7 substrates @ $15 cap)

**Context.** ORCHESTRATOR-5 pending: design the next sweep wave. 6-7 substrates fan out within a $15 total cap (Tier 1 envelope).

**Options.**
- **A** Within-class fan-out: 6-7 HNeRV-family substrates (PR101_lc + PR106_*sidecar + A1+lapose + A1+wavelet + ...).
- **B** Across-class fan-out: D4 + D1 + C6 + C1 + DP1 + SIREN + Cool-Chic (7 different class shifts).
- **C** Mixed: 3 within-class + 4 across-class.

**11-voice positions.**

- **Shannon LEAD**: Option C (mixed). Within-class anchors calibrate the within-class slope (zero-or-near-zero ΔS expected per Z1); across-class anchors test class-shift hypotheses. 3 within + 4 across = balanced posterior update.
- **Dykstra CO-LEAD**: Option C. Pareto-feasibility argument: ~3 within-class anchors saturate the within-class signal; >3 is redundant. ~4 across-class is the maximum independent class-shift candidates we have ready (D4, D1, C6 already in flight; Z3 Ballé as 4th).
- **Yousfi**: Option C. Contest-scorer-audit: balanced posterior.
- **Fridrich**: Option C.
- **Contrarian**: **CHALLENGE Option C**. "Across-class is unproven; why 4? Bet smaller, learn faster." **Response (Selfcomp)**: "Across-class is the ONLY path to sub-0.10 per Z1. Tier 1 $15 across 4 across-class = $3.75 each ON AVERAGE; well within Tier 1 envelope. Cheap to learn fast."
- **Quantizr**: Option C with **PR106-family weighting**: 3 within = 2× PR106-variants + 1× A1+sidecar; 4 across = D4 + D1 + Z3 + C6 (each at different MDL density).
- **Hotz**: Option C with hard cap $15. "If 7 substrates × $3 each = $21, drop the cheapest 2 (typically A1+sidecar variants)."
- **Selfcomp**: Option C.
- **MacKay**: Option C. Expected information gain argument: 3 within-class anchors tighten the within-class posterior tightly; 4 across-class anchors give 4 independent class-shift evidence points.
- **Ballé**: Option C, with **Z3 included** as 4th across-class (the Ballé hyperprior bolt-on counts as borderline within/across; conceptually within-class but mechanistically across-coding).
- **Time-Traveler peer**: Option C with **7-factor frame validation**: every substrate in the sweep must pass the 7-factor pre-dispatch gate (curriculum / substrate / engineering / process / time / complexity / spend). Substrates failing any factor get held out.

**Cross-debate.**
- Contrarian → Selfcomp: "Across-class hypothesis untested empirically." Selfcomp: "C6 5ep is the first across-class data point; Z1 density on across-class will land in Tier C extension; D4 is in flight. We HAVE the across-class hypothesis as the operational frame. Tier 1 is the empirical falsification of class-shift hypothesis."
- Time-Traveler → Hotz: "If 5/7 fail the 7-factor gate?" Hotz: "Then we have a sweep of 2 substrates; that's fine. Quality over quantity. Tier 1 cap is the upper bound, not the target."

**Vote tally**: **11 of 11 for Option C**. Quantizr PR106-family weighting + Time-Traveler 7-factor validation **adopted**. Contrarian SUPER-VETO **not invoked**.

**Binding verdict**: **PROCEED Option C (mixed sweep within $15 cap)**:
- 3 within-class: 2× PR106-variants + 1× A1+sidecar (~$3-5 total).
- 4 across-class: D4 + D1 + Z3 + C6 (~$8-12 total).
- 7-factor pre-dispatch gate: every substrate must pass curriculum / substrate / engineering / process / time / complexity / spend before paid dispatch.
- Total cap $15; soft target $10-12.

**Math + rationale.** Per the Tiered Parallel Plan compound trajectory + 7-factor frame + Z1 ablation evidence:
- 3 within-class anchors give 3-tight posterior on within-class slope (predicted ΔS ≈ [-0.0001, -0.001] per anchor).
- 4 across-class anchors give 4 independent class-shift signals (predicted ΔS ∈ [-0.005, -0.045] per anchor).
- Joint information gain maximized at this mix.

**Reactivation criteria.** If 4+ across-class anchors land regression (ΔS > +0.001), the across-class hypothesis is partially falsified and council reconvenes on whether the substrate-class shift hypothesis holds.

**Operator-routable checkpoints (5):**
1. 7-factor pre-dispatch gate applied to each candidate substrate.
2. Sweep launches concurrent Modal/Vast.ai fan-out within $15.
3. Harvest all artifacts regardless of rc per CLAUDE.md "HARVEST OR LOSE".
4. Wave 3 design decision after Wave 2 harvest (next omnibus or ad-hoc council).
5. Cost-band posterior + autopilot v2 updated post-harvest.

---

## Decision 11: DARTS-SuperNet C7 strategic Tier 3 dispatch ($100-300)

**Context.** Long-term campaign roadmap (`feedback_long_term_multi_year_campaigns_landed_20260514.md` C7): DARTS-SuperNet ($100-300, Tier 3). Question: when to fire?

**Options.**
- **A** Fire now (parallel with Tier 1).
- **B** Defer; gate behind at least one across-class anchor landing.
- **C** Defer to Q3-2026 strategic window.

**11-voice positions.**

- **Shannon LEAD**: Option B. DARTS-SuperNet is an architecture-search heuristic; without an across-class anchor it's searching the wrong space. Gate behind at least one across-class anchor (D4 or Z4 likely first).
- **Dykstra CO-LEAD**: Option B. Feasibility: DARTS' search space is conditioned on the substrate prior; without an across-class anchor, the prior is HNeRV-family saturated and DARTS will rediscover within-class architectures.
- **Yousfi**: Option B. Contest-scorer-audit: DARTS results need to be validated against scorer; with within-class saturated, DARTS won't find scorer-favorable architectures.
- **Fridrich**: Option B.
- **Contrarian**: **CHALLENGE Option C (defer to Q3)**. "Q3 is 2-3 months away; that's too long if D4/Z4 land in Tier 1." **Response (Hassabis grand-bench)**: "Strategic timing IS DARTS' edge; firing too early wastes search budget. After Tier 1 across-class anchor, DARTS gets a meaningful prior."
- **Quantizr**: Option B.
- **Hotz**: Option B with **kill-criteria**: if DARTS budget exceeds $300 with no ΔS, kill regardless of stage. NO infinite-budget research.
- **Selfcomp**: Option B.
- **MacKay**: Option B. Bayesian: DARTS posterior needs at least one strong prior anchor (across-class) for the search to converge.
- **Ballé**: Option B.
- **Time-Traveler peer**: Option B **with reactivation criteria**: DARTS reactivates when (a) at least one across-class anchor lands (likely D4 or Z4) AND (b) the across-class substrate's posterior shows ≥-0.020 ΔS. Without these, DARTS is searching the wrong distribution.

**Cross-debate.**
- Contrarian → Time-Traveler: "What if Q3 arrives with no across-class anchor?" Time-Traveler: "Then DARTS is DEFERRED-pending-research per CLAUDE.md 'KILL is LAST RESORT'. Not killed; deferred until Phase 4 conditions met."

**Vote tally**: **11 of 11 for Option B**. Hotz's kill-criteria + Time-Traveler's reactivation criteria **adopted**. Contrarian SUPER-VETO **not invoked**.

**Binding verdict**: **DEFER Option B**. DARTS-SuperNet C7 gated behind:
- At least one across-class anchor landing (D4 / Z4 / D1 / C6 / etc.).
- Across-class anchor ΔS ≥ -0.020.
- Operator approval for Tier 3 spend ($100-300) at that time.

If reactivation conditions not met by Q3-2026, DEFERRED-pending-research becomes the documented holding pattern.

**Math + rationale.** DARTS' value depends on the search-space prior. With within-class saturated (Z1: A1 99.29% density), DARTS searches a saturated space → diminishing returns. After across-class anchor lands, DARTS prior shifts to the across-class manifold; expected information gain per DARTS dispatch is ~10× higher.

**Reactivation criteria.** (a) Across-class anchor lands; (b) anchor ΔS ≥ -0.020; (c) operator approves Tier 3 spend.

**Operator-routable checkpoints (3):**
1. DARTS-SuperNet remains in lane registry at L0 (sketch only) until reactivation.
2. After across-class anchor lands, council reconvenes on DARTS readiness.
3. Operator decision threshold: $100-300 spend authorization gate.

---

## Decision 12: OSS announcement timing

**Context.** OSS-RELEASE op-routable #3: v0.2.0-rc1 tag pushed; release notes drafted. Question: when to publicly announce?

**Options.**
- **A** Announce now (v0.2.0-rc1 push).
- **B** Hold for v0.2.0 stable (after a contest-CUDA event milestone).
- **C** Stage: rc1 quietly tagged; stable announced with medal-band trigger.

**11-voice positions.**

- **Shannon LEAD**: Option C. Stage the release: rc1 already tagged → quietly notify watchers; stable announce gated on either (a) medal-band [contest-CUDA] anchor lands OR (b) 30-day quiet period elapses (proves rc1 stable).
- **Dykstra CO-LEAD**: Option C.
- **Yousfi**: Option C. Contest-community-coordination: announcing a release without a medal-band signal is meta-communication noise; announcing WITH a signal amplifies it.
- **Fridrich**: Option C.
- **Contrarian**: Option A. "Announce now. Build mindshare. Don't wait." **Response (Hassabis grand-bench)**: "Mindshare without substrate signal is forgettable. Announce on milestone."
- **Quantizr**: Option C. From competitive-intelligence: leaderboard winners' OSS releases coincide with PR pushes; the signal compounds.
- **Hotz**: Option A "Carmack style: ship it, fix bugs as they come." But concedes Option C is more strategic.
- **Selfcomp**: Option C. PR #56 announced on archive sha publication, not pre-medal.
- **MacKay**: Option C. Information-theoretic: signal-to-noise of announcement maximized by milestone coincidence.
- **Ballé**: Option C.
- **Time-Traveler peer**: Option C, with **clear cadence**: rc1 quiet → 30d quiet OR medal-band → stable announce; in either case, OSS docs/site update on rc1 push (not announce).

**Cross-debate.**
- Hotz → Contrarian: "Hotz position changing — concede?" Hotz: "Yes — Carmack pragma applies to internal velocity, not external comms. Strategic on comms."
- Contrarian → Time-Traveler: "30d quiet without medal — does that count as 'stable'?" Time-Traveler: "Yes — 30 days without bug reports IS the empirical stability proof. Either path qualifies."

**Vote tally**: **10 of 11 for Option C** (Hotz initially Option A → concedes Option C; Contrarian remains Option A as challenge then concedes after Time-Traveler clarification). Contrarian SUPER-VETO **not invoked**.

**Binding verdict**: **PROCEED Option C (staged)**:
- v0.2.0-rc1: tagged + quietly notified (watchers, GH releases page, no social push).
- v0.2.0 stable: announced on **first occurrence of**: (a) [contest-CUDA T4] medal-band anchor lands (≤0.198 score, top-10 cluster), OR (b) 30-day rc1 quiet period elapses without critical bug reports.
- Both paths trigger broad social/community push.

**Math + rationale.** Signal-to-noise of announcement: SNR = (audience attention × signal credibility) / (noise floor). Pre-medal rc1 announce: low credibility → low SNR. Post-medal stable announce: high credibility → high SNR. 30-day quiet alternative: empirical stability proof → moderate credibility → acceptable SNR.

**Reactivation criteria.** None — staged release is the canonical OSS-release pattern per CLAUDE.md "Public Disclosure Hygiene".

**Operator-routable checkpoints (3):**
1. rc1 tagged + GH release notes published quietly (no social).
2. Trigger watchlist: weekly check on (a) any [contest-CUDA T4] medal-band anchor; (b) 30-day quiet rolling timer.
3. Stable announce template prepared so launch is single-click when trigger fires.

---

## Decision 13: F3 backport sister-trainer wave (vq_vae + PDP)

**Context.** F3-BACKPORT-WAVE-V2 op-routable #3+#4: vq_vae trainer needs flag-declare (operator-required flag manifest); PDP (pretrained driving prior) substrate-side wire-in pending.

**Options.**
- **A** Both immediate.
- **B** vq_vae first (smaller surface); PDP next.
- **C** vq_vae flag-declare immediate (~25 LOC); PDP substrate-side wire-in opt-in (substrate author decides).

**11-voice positions.**

- **Shannon LEAD**: Option C. vq_vae flag-declare is structurally required per Catalog #151 (`check_operator_wrapper_threads_trainer_tier_required_flags`); not opt-in. PDP substrate-side wire-in is opt-in because PDP's substrate semantics differ (pretrained prior vs trainable substrate).
- **Dykstra CO-LEAD**: Option C.
- **Yousfi**: Option C.
- **Fridrich**: Option C.
- **Contrarian**: Option C with **enforcement**: Catalog #151 STRICT-flips on vq_vae when the flag-declare lands (atomic STRICT per Decision 6).
- **Quantizr**: Option C.
- **Hotz**: Option C. Editor-time estimate: vq_vae flag-declare ~30 min; PDP substrate-side wire-in ~1-2h (substrate author decides).
- **Selfcomp**: Option C.
- **MacKay**: Option C.
- **Ballé**: Option C.
- **Time-Traveler peer**: Option C, with **PDP substrate-side wire-in NOT defaulted to wire**. Per CLAUDE.md "PDP" framing: PDP is a pretrained driving prior, not a contest-bytes substrate; wiring it as a substrate is a separate design decision. Recommendation: wire-in opt-in via flag, default OFF.

**Cross-debate.**
- Contrarian → Time-Traveler: "If PDP wire-in is opt-in default-OFF, who uses it?" Time-Traveler: "Substrate authors who explicitly want PDP as a prior in their compositional cell. Cool-Chic / C3 / world-model substrates might."

**Vote tally**: **11 of 11 for Option C**. Contrarian's Catalog #151 STRICT-flip enforcement + Time-Traveler's PDP wire-in default-OFF **adopted**. Contrarian SUPER-VETO **not invoked**.

**Binding verdict**: **PROCEED Option C**:
- vq_vae flag-declare: lands immediate (~30 min editor); Catalog #151 STRICT-flips atomically per Decision 6.
- PDP substrate-side wire-in: lands as opt-in flag (default OFF); substrate authors choose to enable per composition.

**Math + rationale.** Catalog #151 enforcement requires every operator-required flag to be declared by the trainer. vq_vae lacks the manifest → Catalog #151 will refuse the next vq_vae dispatch. Fix is editor-side.

**Reactivation criteria.** None — these are structural compliance items.

**Operator-routable checkpoints (3):**
1. vq_vae trainer adds `TIER_1_OPERATOR_REQUIRED_FLAGS` dict (~25 LOC).
2. Catalog #151 STRICT-flip on vq_vae (atomic per Decision 6).
3. PDP wire-in lands as opt-in flag.

---

## Decision 14: MDL ablation Tier D / Tier E future extension

**Context.** MDL-ABLATION-TIER-C op-routable implicit: future extension pattern for the MDL ablation framework. Tier A (architecture-only) / Tier B (100ep auth-eval) / Tier C (scorer-conditional MDL) all canonical. What's Tier D? Tier E?

**Options.**
- **A** Tier D = converged-substrate MDL (post-1000ep MDL on a trained substrate).
- **B** Tier D = scorer-conditional class-transition probability (probability of crossing class boundary).
- **C** Tier D = predictive-coding residual density (sister of Z5 metric).
- **D** Defer; let need drive design.

**11-voice positions.**

- **Shannon LEAD**: Option B then C. Tier D scorer-conditional class-transition is the natural extension — measures the probability mass on across-class moves. Tier E predictive-coding residual density measures Z5-class substrates specifically.
- **Dykstra CO-LEAD**: Option B then C.
- **Yousfi**: Option B. Contest-scorer-audit: class-transition probability tells us if a substrate has ANY across-class signal accessible.
- **Fridrich**: Option B.
- **Contrarian**: Option D. "Don't pre-design extensions." **Response (MacKay)**: "Pre-designing the Tier ladder gives us a roadmap; Z1 ablation evidence already points at Tier D+E targets. Pre-design is cheap research-only work."
- **Quantizr**: Option B then C.
- **Hotz**: Option B then C with **build only when needed**. Pre-design the framework; build per-Tier when a substrate motivates it.
- **Selfcomp**: Option B then C.
- **MacKay**: Option B then C. MDL framing of Tier ladder:
  - Tier A: K(architecture).
  - Tier B: K(architecture + 100ep state).
  - Tier C: K(payload | scorer) (scorer-conditional density).
  - Tier D: K(class_transition_target | substrate_state) (class-transition probability).
  - Tier E: K(predictive_residual | substrate_world_model) (Z5-specific).
- **Ballé**: Option B then C. compressai has analog extensions (rate-distortion-perceptual / rate-distortion-side-info).
- **Time-Traveler peer**: Option B then C with **full ladder**:
  - Tier D: scorer-conditional class-transition probability.
  - Tier E: predictive-coding residual density (per Rao-Ballard 1999).
  - Tier F (future): world-model latent dimensionality (per Ha-Schmidhuber 2018).
  - Tier G (future): foveation-conditional density (per Atick-Redlich 1990).

**Cross-debate.**
- Contrarian → MacKay: "If we don't build Tier D, what's the cost?" MacKay: "When the first across-class anchor lands, we can't measure how much across-class signal is accessible without Tier D. We retro-build it at that time. Pre-design now is cheap (~1h editor); build later."

**Vote tally**: **11 of 11 for Option B then C (pre-design)**. Time-Traveler's full ladder Tier F/G **adopted as future extension**. Contrarian SUPER-VETO **not invoked**.

**Binding verdict**: **PROCEED with Tier D + E pre-design**:
- Tier D: scorer-conditional class-transition probability (`P(class_shift | substrate_state)`).
- Tier E: predictive-coding residual density (for Z5-class substrates).
- Tier F + G (future): world-model latent dim + foveation-conditional density.

Pre-design lives in `.omx/research/mdl_tier_ladder_design_20260514.md` (research-only). Build per-Tier when a substrate motivates (e.g., Tier D builds when first across-class anchor lands).

**Math + rationale.** MDL framing of substrate description-length ladder:
- Tier C captures **scorer-conditional density** of payload bytes.
- Tier D captures **class-transition mass** (does the substrate concentrate posterior on across-class or within-class moves?).
- Tier E captures **predictive-residual density** (Rao-Ballard 1999 predictive cortex framing).

Each Tier is a separate dimension of the substrate's information-theoretic profile.

**Reactivation criteria.** None — pre-design is research-only.

**Operator-routable checkpoints (3):**
1. Tier D/E pre-design memo lands at `.omx/research/mdl_tier_ladder_design_20260514.md` (research-only; can come from omnibus or a sister deliberation).
2. Tier D built when first across-class anchor lands.
3. Tier E built when first Z5-class substrate lands.

---

## Decision 15: Sister-substrate parser P2/P3 sequencing

**Context.** IBPS1-PARSER-WAVE-P0 op-routable #2+#3: P0 landed parsers for D1+D4+DP1 (commit `7fb4d079a`); P2 + P3 sister waves remain.

**Options.**
- **A** P2 (6 substrates) one wave + P3 (11 substrates) one wave.
- **B** P2 (6 substrates) one wave + P3 split into 2 sub-waves of ~5-6 each.
- **C** Combine P2 + P3 = 17 substrates in two waves of ~9 each.

**11-voice positions.**

- **Shannon LEAD**: Option B. Smaller waves reduce reviewer cognitive load; per CLAUDE.md "Single-LOC-per-LOC review discipline".
- **Dykstra CO-LEAD**: Option B.
- **Yousfi**: Option B.
- **Fridrich**: Option B.
- **Contrarian**: Option C. "Two waves is faster than three." **Response (Hotz)**: "Two waves of 9 each = ~9 × ~30 LOC parsers = ~270 LOC per wave. That's the upper bound of reviewable in 30-sec discipline. Three waves of ~6 each = ~180 LOC per wave, cleanly reviewable. Three is the right granularity."
- **Quantizr**: Option B.
- **Hotz**: Option B.
- **Selfcomp**: Option B.
- **MacKay**: Option B.
- **Ballé**: Option B.
- **Time-Traveler peer**: Option B, with the registry refactor (Decision 1) **landing as part of P2's commit-batch** (atomic with the first sister-parser wave).

**Cross-debate.**
- Contrarian → Hotz: "Three waves = more orchestration." Hotz: "Yes, but each is 30-sec-reviewable. Compounds across the wave-set."
- Time-Traveler → Shannon: "Registry refactor first or alongside P2?" Shannon: "Alongside P2 commit-batch — same review pass."

**Vote tally**: **11 of 11 for Option B**. Time-Traveler registry-refactor atomicity **adopted**. Contrarian SUPER-VETO **not invoked**.

**Binding verdict**: **PROCEED Option B**:
- P2 wave: 6 substrates (next batch) + registry refactor (Decision 1) in single commit-batch.
- P3 split: 11 substrates → wave 3a (~5-6 substrates) + wave 3b (~5-6 substrates).
- Each wave ≤ ~200 LOC delta for 30-sec review discipline.

**Math + rationale.** Per CLAUDE.md "Single-LOC-per-LOC review discipline" + complexity factor (factor 6 in 7-factor frame): bolt-on ≤350 LOC; parser wave LOC budget ~30 LOC × N substrates. N=6 → ~180 LOC reviewable; N=11 → ~330 LOC, near ceiling.

**Reactivation criteria.** None — process discipline.

**Operator-routable checkpoints (3):**
1. P2 + registry refactor commit-batch (~3-5h editor work; $0).
2. P3 wave 3a + 3b sequential (each ~2-3h editor).
3. Each wave's parsers pass per-grammar `test_<grammar>_parser_roundtrip.py`.

---

## Decision 16: Push v0.2.0-rc1 vs hold for v0.2.0 stable

**Context.** OSS-RELEASE op-routable #5: same surface as Decision 12 with explicit version-tag dimension.

**Verdict.** **Subsumed by Decision 12**: rc1 already tagged + quietly notified; v0.2.0 stable announce on **first occurrence of**: medal-band [contest-CUDA] anchor OR 30-day rc1 quiet period.

**Operator-routable checkpoints (1):**
1. v0.2.0 stable tag push triggered by Decision 12's stable-announce trigger.

---

## Decision-by-decision verdict-and-reactivation matrix

| # | Verdict | Reactivation criteria |
|---|---|---|
| 1 | PROCEED Option A (registry pattern + `register_grammar` API; ≤120 LOC) | None — structural |
| 2 | DEFER default flip (Option B; opt-in per trainer + invariant probe) | Probe lands + 3 substrates 100ep cached/uncached agree within 1e-9 |
| 3 | PROCEED Option B (v2 latent-replacement; v1 retired) | If v2 smoke regresses >0.001, DEFER |
| 4 | PROCEED Option C (Z3+Z4+D4 parallel; Z5 gated on Tier 1 anchor) | If all 3 regress, council reconvenes on staircase |
| 6 | PROCEED Option C (per-gate atomic STRICT or warn-only-with-plan) | None — process discipline |
| 7 | PROCEED A→C ordering (PR106 + DP1 parallel; across-class gated) | If PR106 deviates >5% from A1, council reconvenes |
| 8 | PROCEED Option D (4-gate canonical + conditional 5th probe-disambiguator) | None — process canonical |
| 9 | PROCEED Option B (per-class routing + dynamic re-routing) | Cost-band posterior anchor shifts >25% trigger re-eval |
| 10 | PROCEED Option C (3 within + 4 across; $15 cap; 7-factor gate) | If 4+ across-class regress, council reconvenes |
| 11 | DEFER Option B (gate behind across-class anchor) | Across-class anchor + ΔS ≥ -0.020 + operator approval |
| 12 | PROCEED Option C (staged: rc1 quiet → medal OR 30d → stable) | None — canonical OSS release |
| 13 | PROCEED Option C (vq_vae flag-declare immediate; PDP opt-in default-OFF) | None — structural |
| 14 | PROCEED B→C pre-design (Tier D class-transition; Tier E predictive-residual) | None — pre-design research-only |
| 15 | PROCEED Option B (P2 + registry refactor batch; P3 split 3a+3b) | None — process discipline |
| 16 | Subsumed by Decision 12 | Same as Decision 12 |

---

## Aggregate operator-visibility checkpoints (5+)

1. **Decisions queue cleared**: 14 of 14 design decisions now have binding verdicts; `.omx/state/pending_council_design_decisions.jsonl` updated with all 14 resolved rows.
2. **Tier 1 sweep design locked**: Decisions 4 + 10 jointly define the next sweep wave (Z3+Z4+D4 parallel; broader Wave 2 = 3 within + 4 across @ $15 cap).
3. **Substrate-class promotion canonical locked**: Decision 8's 4-gate canonical operationalizes L1 → L2 promotion across all future substrates.
4. **OSS release path locked**: Decisions 12 + 16 jointly define rc1 → stable staging.
5. **Future MDL framework pre-designed**: Decision 14's Tier D + E + F + G ladder + Decision 7's Tier C extension sequencing combine into a full MDL research roadmap.
6. **Editor-side compliance items locked**: Decision 1 (registry refactor) + Decision 13 (vq_vae flag-declare) + Decision 14 (Tier ladder pre-design) ~$0 GPU + ~8-10h editor work post-omnibus.
7. **Cost authorization clarity**: per-class routing (Decision 9) + tier envelope (Decision 10 $15 cap) sets the next 48h spend ceiling at ~$12-15 across all in-flight Tier 1.
8. **DARTS-SuperNet held**: Decision 11 prevents $100-300 strategic-window spend until reactivation criteria met (saves ~$100-300 if criteria don't materialize).
9. **Council non-conservatism preserved**: 3 Contrarian challenges issued and addressed by math/empirical (Decisions 1 / 7 / 14); SUPER-VETO not invoked on any.

---

## Cross-refs

- CLAUDE.md "Design decisions — non-negotiable" (the canonical doctrine this omnibus operationalizes)
- CLAUDE.md "Council conduct — non-negotiable" (non-conservatism enforcement)
- CLAUDE.md "Subagent coherence-by-default — NON-NEGOTIABLE, HIGHEST EMPHASIS" (this omnibus IS the coherence primitive in action)
- CLAUDE.md "KILL is LAST RESORT" + "KILL/FALSIFIED memo structural requirements" (every DEFER carries reactivation criteria; no kills)
- CLAUDE.md "Strict-flip atomicity rule" (Decision 6)
- CLAUDE.md "Race-mode rigor inversion + parallel-dispatch first" (Decisions 4 + 10)
- CLAUDE.md "Long-burn score-lowering campaign default" (Decisions 3 + 11)
- `.omx/research/all_design_decisions_through_grand_council_directive_20260514.md` (the canonical referral)
- `.omx/research/holistic_engineering_picture_seven_factor_directive_20260514.md` (the 7-factor frame applied)
- `.omx/research/grand_council_tiered_parallel_plan_full_authority_20260514.md` (the parent Tier roadmap)
- `.omx/research/grand_council_c1_world_model_adversarial_review_20260514.md` (sister C1 deliberation)
- `feedback_z1_mdl_ablation_landed_20260514.md` (the within-class saturation empirical anchor)
- `feedback_long_term_multi_year_campaigns_landed_20260514.md` (C7 DARTS source)

---

## Crash-resume protocol (Catalog #206)

- `parent_id_or_session`: `operator-session`
- `lane_id`: `lane_grand_council_omnibus_design_decisions_20260514`
- `inherited_directives`: [`recovery_session_20260514_directive_absolute_no_signal_loss_20260514`, `recursive_no_signal_loss_protocol_20260514`, `journal_lab_grade_documentation_standard_directive_20260514`, `harness_rigor_deterministic_reproducibility_directive_20260514`, `holistic_engineering_picture_seven_factor_directive_20260514`, `grand_council_tiered_parallel_plan_full_authority_20260514`, `all_design_decisions_through_grand_council_directive_20260514`]
- Final checkpoint status: `complete`
- Resume instructions if interrupted: rerun `tools/subagent_checkpoint.py read --subagent-id grand_council_omnibus_design_decisions_20260514 --latest-incomplete`; resume by composing next decision section from `## Decision N: <title>` template (positions → cross-debate → verdict → math → reactivation → checkpoints).

---

## Effective immediately

All 14 design decisions are **binding** per CLAUDE.md "Design decisions — non-negotiable" + "Subagent coherence-by-default" + the canonical referral directive's standing pre-authorization. In-flight subagents (AUTOPILOT-TIER-C / CATALOG-227-231-WAVE / SISTER-PARSER-P1-WAVE / C1-COUNCIL-RECONVENE / OSS-PUBLIC-PUSH) inherit the verdicts on next checkpoint cycle.

Future subagents pre-flight reads this directive via mandatory `.omx/research/*_directive_*` last-24-hours pre-read.

`research_only=true`. NO score claims. NO GPU spend by this council deliberation. Decision-2 / Decision-11 saving: ~$0 (deferred); Decision-9 routing: ~15-25% reduction per dispatch; Decision-10 envelope: $15 cap; Decision-11 DARTS gating: ~$100-300 saved if reactivation not met.
