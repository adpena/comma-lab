# NSCS03 OPERATOR-DECISION items investigation — 2026-05-15

**Subagent:** `20260516T033335Z_nscs03_op_decision_investigator`
**Lane:** `lane_nscs03_operator_decision_items_investigation_20260515`
**Scope:** Enumerate every OPERATOR-DECISION-REQUIRED item against NSCS03 per
Subagent A (audit `.omx/research/4_substrate_9_dim_and_contest_compliance_audit_20260515.md`)
and Subagent C (plan `.omx/research/4_substrate_plus_3_stack_dispatch_plan_20260515.md`,
Section 2.4); classify each per the canonical
share-vs-unique decision flowchart
(`feedback_canonical_share_when_serves_unique_when_suppresses_standing_directive_20260515.md`)
and the HARD-EARNED-vs-CARGO-CULTED addendum
(`feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515.md`);
flag council-grade items per CLAUDE.md "Design decisions — non-negotiable"
and "Council conduct" (sextet pact + Assumption-Adversary seat per
`feedback_adversarial_review_apparatus_blind_to_shared_assumption_failure_meta_meta_meta_meta_20260515.md`);
recommend pre-dispatch gating + unblock sequence.
**Mode:** READ-ONLY; writes only this ledger (Catalog #230 disjoint from sister
subagents).

---

## Section 0 — State summary (verified pre-write)

| Surface | State |
|---|---|
| Trainer `_full_main` | IMPLEMENTED 2026-05-15 per UNIQUE-AND-COMPLETE-PER-METHOD PR95+Ballé2018 (~880 LOC body inside 1343 LOC file) |
| Substrate inflate.py | 214 LOC (substrate-engineering ≤200 LOC waiver region) `[empirical:wc -l src/tac/substrates/nscs03_end_to_end_balle_joint_codec/inflate.py]` |
| Submission inflate.py | 226 LOC (substrate-engineering waiver) `[empirical:wc -l submissions/nscs03_end_to_end_balle_joint_codec/inflate.py]` |
| Recipe `dispatch_enabled` | **false** |
| Recipe `research_only` | **true** |
| Recipe `smoke_only` | **true** |
| Recipe `dispatch_blockers` | `[phase_2_council_approval_required_to_lift_full_main_NotImplementedError, lambda_R_sweep_calibration_pending_first_smoke_anchor, sigma_floor_sensitivity_pending_low_rate_op_point_validation]` |
| Recipe `min_smoke_gpu` | A100 (Catalog #215; cannot downgrade) |
| Recipe `gpu` | A100 |
| Recipe `hand_calibrated_fallback_p50_usd` | 80.00 |
| Recipe `timeout_hours` | 12.0 |
| Recipe `target_modes` | `[research_substrate]` |
| Operator-authorize wrapper | EXISTS (`scripts/operator_authorize_substrate_nscs03_end_to_end_balle_joint_codec_modal_a100_dispatch.sh`, 3.8 KB) — **Subagent C's Section 2.4 prereq-op #1 is OBSOLETE; wrapper already generated** |
| Remote lane driver | EXISTS (`scripts/remote_lane_substrate_nscs03_end_to_end_balle_joint_codec.sh`, 9.6 KB) |
| TIER_1_OPERATOR_REQUIRED_FLAGS | 10 keys including `--lambda-R`, `--sigma-floor`, `--main-latent-channels`, `--hyper-latent-channels`, `--gdn-eps` |
| Lane registry gates | 2/8 PASS (`impl_complete`, `memory_entry`) → Level 1 |
| Cohort verdict | OPERATOR-DECISION-REQUIRED (per audit Section 3) |

---

## Section 1 — Enumerated OPERATOR-DECISION items

The investigation surfaces **8 distinct items** needing operator/council decision
before NSCS03 can advance from Level 1 → Level 2 → Level 3. Item numbers index
into the resolution recommendations + gate matrix + council-grade flags in
Sections 2–4.

---

### Item 1 — Recipe gate-flip (`dispatch_enabled: true` + clear `dispatch_blockers`)

- **Description:** Recipe currently `dispatch_enabled: false / research_only: true / smoke_only: true / dispatch_blockers: [3 items]`. Trainer's `_full_main` is fully wired; the gate is now operator-choice not trainer-readiness. Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" + Catalog #240, the recipe is the OPERATOR safety surface; flipping `dispatch_enabled: true` is a deliberate authorization.
- **Cost:** ~5 min editor (3-line YAML edit + clear blockers list).
- **Reversibility:** Easy revert (single commit; no GPU spend yet).
- **Empirical evidence:** `[empirical:experiments/results/lane_nscs03_*_smoke__1ep_cpu/stats.json]` — CPU 1-epoch smoke rc=0 + parseable stats + 76/76 tests pass per landing memo. The trainer's runtime contract is validated locally; gate-flip is the structural authorization, not a technical readiness question.
- **Council-grade?** **YES** — directly gates ~$60–80 A100 spend per dispatch (long_burn boundary per Catalog #239). CLAUDE.md "Design decisions — non-negotiable" threshold ($1+ GPU spend OR 2+ defensible alternatives).

### Item 2 — λ_R sweep council adjudication (Items 2 + 3 + 4 are the three explicit reactivation_criteria from the recipe + landing memo)

- **Description:** Ballé 2018 recipe's rate-distortion tradeoff parameter; current default `--lambda-R 0.5` (env var `NSCS03_LAMBDA_R="0.5"`). Landing memo line 117 + recipe `dispatch_blockers[1]` cite λ_R sweep as Phase 2 council blocker. Subagent C plan Section 2.4 cites typical logspace `[0.01, 0.1, 0.5, 1.0, 5.0]` (5-config sweep).
- **Cost:** 4–5× cost of single A100 smoke (`~$1.50–4.00` per smoke × 4–5 candidates = `$6–20` for the sweep on Lightning T4 per landing memo line 117). NOT the full $80 each. Wall-clock 4–5h parallel.
- **Reversibility:** Easy — sweep is empirical measurement; each smoke is independent + cancellable.
- **Empirical evidence:** NONE for the substrate currently — the landing memo cites the sweep AS the empirical evidence that will close the question; no prior NSCS03-specific λ_R measurement exists. The default `0.5` is theoretical-mid (Ballé 2018 paper uses `[0.001, 0.05]` range for natural images at much LOWER bit-rates; comma video at our rate envelope may need different).
- **Council-grade?** **YES** — Ballé 2018 specific to natural-image compression; comma driving video is OOD and the optimal λ_R is operating-point-dependent. Council must weigh: (a) skip sweep + use default `0.5` ($0 sweep cost / unknown ΔS) vs (b) run 4-config Lightning T4 sweep ($6–20 cost / measured ΔS surface) vs (c) defer sweep + run NSCS03 full A100 at default ($60–80 / uncalibrated risk).

### Item 3 — σ-floor sensitivity council adjudication

- **Description:** EntropyBottleneck logistic CDF lower bound on the scale parameter. Current `--sigma-floor 1e-4` (env var `NSCS03_SIGMA_FLOOR="1e-4"`). Landing memo line 119: "1e-4 may be too tight at low rate operating points". Recipe `dispatch_blockers[2]` cites σ-floor as Phase 2 council blocker.
- **Cost:** ~3× smoke cost (paired comparison `1e-3 / 1e-4 / 1e-5` on Lightning T4 per landing memo, `~$4.50–12` for the sweep).
- **Reversibility:** Easy — three independent smokes.
- **Empirical evidence:** Ballé 2018 reference uses analytical lower bound `~2^-18 ≈ 3.8e-6` at low rate. Our `1e-4` is 25× tighter (more constrained scale parameter) which may pinch the entropy bottleneck and force suboptimal latent quantization. [first-principles bound; literature anchor only; no measured NSCS03-on-comma-video data]
- **Council-grade?** **YES** — same reasoning as Item 2. The σ-floor interacts with λ_R (low λ_R + tight σ-floor = under-rate; high λ_R + loose σ-floor = under-fitting). Both must be co-calibrated.

### Item 4 — EMA decay differentiated 0.999/0.997 split (DOCUMENTED FORK deferred to Phase 2)

- **Description:** Ballé 2018 reference uses TWO EMA decay rates: 0.999 for hyperprior + 0.997 for main encoder. NSCS03 trainer currently uses single `--ema-decay 0.997` for the joint state_dict. Landing memo lines 67–68 + 120 explicitly defer this as "Phase 2 calibration item".
- **Cost:** Engineering: ~2 hours editor work to split EMA into two scoped instances + thread through validation + archive emission. Compute: marginal — same training cost (EMA is post-step).
- **Reversibility:** Moderate — touches `_full_main` snapshot+restore + archive build helpers; revert requires another edit.
- **Empirical evidence:** Ballé 2018 paper uses the differentiated split (citation: landing memo line 32 trainer comment). No internal paired comparison; CLAUDE.md "EMA — non-negotiable" canon is `0.997` for weights; codebook EMA is `0.99` (van den Oord). The differentiated split is HARD-EARNED in the Ballé-2018 lineage but CARGO-CULTED in ours (we adopted the canonical `0.997` without empirical proof it's optimal for joint encoder+hyperprior).
- **Council-grade?** **YES** — touches CLAUDE.md "EMA — non-negotiable" canonical value. Per Catalog #88 EMA correctness gate + sister Catalog #128 (locked posterior writes) — any divergence from canonical EMA needs council sign-off + waiver documentation.

### Item 5 — Trainer LOC budget (1343 LOC vs PR101's 600-LOC target)

- **Description:** Audit Dim 2 + Dim 5 flagged the trainer is `1343 LOC` — "well above PR101 600 target". Per CLAUDE.md "HNeRV / leaderboard-implementation parity discipline" L7: "Bolt-on size ≤ 350 LOC (substrate engineering may exceed; tag `lane_class=substrate_engineering` explicitly)." NSCS03 IS substrate-engineering, so 1343 LOC is admissible — BUT every line ≥350 must be reviewable in 30s per L12.
- **Cost:** Trimming: 4–8 hours editor. Cost of NOT trimming: each council review takes longer; 30s-reviewability degrades.
- **Reversibility:** Hard to reverse a trim (refactor cost); easy to leave as-is.
- **Empirical evidence:** Sister trainer LOC counts: `balle_renderer=1502 LOC / nscs01=1047 LOC / nscs06=785 LOC` `[empirical:wc -l experiments/train_substrate_*.py]`. NSCS03 sits between Ballé renderer (1502) and NSCS01 (1047). The 30s-reviewability claim is structural-discipline not score-affecting.
- **Council-grade?** **NO** — pure engineering hygiene; HNeRV parity L7 explicitly permits substrate-engineering exception. Defer to follow-up subagent only if reviewability becomes a blocker.

### Item 6 — Memo backfill: missing `## 9-dimension success checklist evidence` section per Catalog #290

- **Description:** Per audit Section 4 + Catalog #290 sister-discipline + the standing directive's "Operator-facing standing rule": every substrate scaffold landing memo dated >= 2026-05-15 MUST include the literal section header. NSCS03 landing memo has the canonical-vs-unique table at lines 19–46 but NOT under the literal `## Canonical-vs-unique decision per layer` heading; also lacks `## 9-dimension success checklist evidence`, `## Predicted ΔS band`, `## Stack-of-stacks composition matrix`. Cohort-wide gap.
- **Cost:** ~30 min editor (4 sister substrates × ~2h = 8h total via sister subagent).
- **Reversibility:** Trivial (memo edit only).
- **Empirical evidence:** Catalog #290 STRICT preflight gate is WARN-ONLY at landing; the gate IS the gap evidence.
- **Council-grade?** **NO** — purely structural-discipline backfill. Handle via sister memo-backfill subagent (not this investigator's scope).

### Item 7 — Composition matrix for 3-stack (NSCS01+02+03) — DEFERRED-pending-design

- **Description:** Audit Dim 6 MISSING; Subagent C Section 3 + OR-5: NSCS03's monolithic codec doesn't naturally compose. Three trainer-design paths (sequential-cascade / joint-multi-objective / 3-substrate substrate-engineering scaffold) await council adjudication. Per CLAUDE.md "Forbidden cross-archive composition without verified [contest-CUDA] substrate anchor": composition is BLOCKED until ≥2 of {NSCS01, NSCS02, NSCS03} have landed contest-CUDA anchors.
- **Cost:** Council: ~1h deliberation. Composition trainer authoring: ~8–16h after council picks design path. Compute: ~$40–120 for composition full per OR-5.
- **Reversibility:** Hard — composition trainer is NEW code; revert = throw away the implementation.
- **Empirical evidence:** Subagent C composition orthogonality analysis (independent ΔS bands `[-0.020, -0.030]` per substrate per ASSUMPTIONS-CHALLENGE-AUDIT). Additive upper bound `[-0.060, -0.090]` / saturating lower bound `[-0.030, -0.045]` per Tier C composition rule. Confidence: LOW until ≥2 NSCS individual anchors land.
- **Council-grade?** **YES** — entire substrate-class composition + cost-band > $40 + 3+ defensible design paths.

### Item 8 — Cost-band classification jump to `long_burn` (Catalog #239 boundary) + Catalog #271 codex pre-dispatch review trigger

- **Description:** NSCS03 full cost band `$60–80` crosses the `long_burn` boundary (`> $50`) per Catalog #239. Per Catalog #271, ANY paid dispatch `> $1` triggers codex pre-dispatch review (cost-gated). NSCS03 full will fire BOTH protections; the codex review may surface findings that need to be addressed BEFORE the GPU meter starts.
- **Cost:** Codex review: ~$0.50–2.00 codex tokens (per Catalog #271 cache schema; cache key includes git_HEAD_sha + recipe_sha + trainer_sha + dirty_tree_fingerprint + untracked_relevant_fingerprint per Catalog #282; fresh review on every dispatch). Operator response to findings: variable.
- **Reversibility:** Easy — codex review is read-only; findings are advisory unless `no-ship` verdict.
- **Empirical evidence:** Sister Z3-G1 codex review bkrbqet3p surfaced 2 HIGH findings (F1+F2) that operator-approved as paired-empirical-confirmation-then-research_only verdict. NSCS03 has analogous risk surfaces (entropy bottleneck weights actually consumed at decode? σ-floor actually clamped? λ_R warmup deterministic?). Cannot predict findings until codex actually runs.
- **Council-grade?** **NO** at the trigger surface — Catalog #271 IS the canonical gate; just route through it. Council-grade ONLY if codex returns `needs-attention` or `no-ship`.

---

## Section 2 — Recommended resolution per item (canonical decision flowchart)

The canonical decision flowchart per
`feedback_canonical_share_when_serves_unique_when_suppresses_standing_directive_20260515.md`:

1. **EMPIRICAL** — paired-comparison smoke run → adopt the lower-scoring path.
2. **PRINCIPLED** — design assumption fits substrate's math? YES adopt / NO fork / UNCLEAR run smoke OR fork with rationale.
3. **UNCLEAR / UNMEASURED** — burden on PROVING canonical-is-better.
4. **OBVIOUS-FIT** — adopt canonical (default).

| Item | Classification | Recommended resolution |
|---|---|---|
| **1** Recipe gate-flip | OBVIOUS-FIT (operator authorization gate; not a canonical-vs-unique design question) | Flip after Items 2 + 3 council adjudication lands. If operator chooses "skip sweep + use defaults", flip immediately + accept Item 8 codex review risk. |
| **2** λ_R sweep | UNCLEAR — empirical measurement required | Run 4–5 config Lightning T4 paired smoke (`λ_R ∈ [0.01, 0.1, 0.5, 1.0, 5.0]`, 100ep each, ~$6–20 total). Pick lowest-scoring config per CLAUDE.md "Apples-to-apples evidence discipline" `[contest-CUDA Lightning T4]` tag. |
| **3** σ-floor sensitivity | UNCLEAR — empirical measurement required + Ballé2018 reference suggests `1e-4` is 25× tighter than analytical lower bound | Co-sweep with Item 2 (or run standalone 3-config `σ ∈ [1e-3, 1e-4, 1e-5]` ~$4.50–12). Pick lowest-scoring. |
| **4** EMA decay differentiated split | PRINCIPLED FORK — design assumption mismatch: Ballé 2018 differentiated split is the canonical Ballé recipe; our single 0.997 is CARGO-CULTED canonical. BUT CLAUDE.md "EMA — non-negotiable" pins 0.997. **Decompose**: HARD-EARNED core = "EMA shadow at inference + 0.997 for weights"; CARGO-CULTED shell = "single decay across joint encoder + hyperprior". | Implement differentiated split (0.999 hyperprior + 0.997 main) as **UNIQUE engineering** per HNeRV parity L7 substrate-engineering exception. Document via same-line `# EMA_DECAY_DIFFERENTIATED_OK:balle2018_recipe_canonical_for_joint_codec` waiver in trainer. Council sign-off required (Item 4 is council-grade). |
| **5** Trainer LOC budget | OBVIOUS-FIT (substrate-engineering exception per HNeRV parity L7) | Leave as-is; 1343 LOC sits between sister Ballé renderer (1502) and NSCS01 (1047). Add 30s-reviewability annotations to `_full_main` if reviewer flags it. |
| **6** Memo backfill (Catalog #290) | OBVIOUS-FIT (structural-discipline backfill) | Defer to sister memo-backfill subagent (separate task; not blocking dispatch). |
| **7** 3-stack composition | UNCLEAR — 3 defensible design paths; council adjudication required | DEFERRED until ≥2 of {NSCS01, NSCS02, NSCS03} have landed `[contest-CUDA]` anchors per CLAUDE.md cross-archive composition rule. After anchors land: full council convenes + picks one of {sequential-cascade, joint-multi-objective, 3-substrate substrate-engineering scaffold}. |
| **8** Cost-band trigger + Catalog #271 codex review | OBVIOUS-FIT (route through canonical gate; act on findings) | Let Catalog #271 + Catalog #270 + Catalog #243 fire automatically via `tools/operator_authorize.py`. If codex returns `approve` → proceed; `advisory` → review findings then proceed; `needs-attention` → pause + address; `no-ship` → pause + escalate. |

---

## Section 3 — Pre-dispatch gate matrix

For each Item, mark whether it BLOCKS each dispatch / promotion stage.

| Item | Smoke (cheap) | Full CUDA (~$60–80) | Paired CPU (~$0.10–0.50) | L2 promotion | L3 promotion | 3-stack composition |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** Recipe gate-flip | BLOCKS (smoke runs through `run_modal_smoke_before_full.py` which respects `dispatch_enabled`) | BLOCKS | BLOCKS (paired CPU depends on prior CUDA archive) | BLOCKS (no CUDA = no `real_archive_empirical` evidence) | BLOCKS | BLOCKS |
| **2** λ_R sweep | RECOMMENDED-FIRST (cheaper to sweep at smoke-scale than to discover at full-scale) | RECOMMENDED-FIRST (sweep informs which λ_R to use for full) | does not block (CPU paired runs on archive that emerged from full) | RECOMMENDED-FIRST | RECOMMENDED-FIRST | RECOMMENDED-FIRST |
| **3** σ-floor calibration | RECOMMENDED-FIRST | RECOMMENDED-FIRST | does not block | RECOMMENDED-FIRST | RECOMMENDED-FIRST | RECOMMENDED-FIRST |
| **4** EMA differentiated split | does not block smoke (single decay still passes integration) | DOES NOT block (substrate-engineering optionality; document FORK in landing memo + waiver) | does not block | DOES NOT block L2 | RECOMMENDED for L3 (canonical Ballé reference; landing memo lists as Phase 2 reactivation criterion) | RECOMMENDED (composition trainer would inherit) |
| **5** LOC budget trim | does not block | does not block | does not block | does not block | RECOMMENDED for L3 (production-hardened standard) | does not block |
| **6** Memo backfill | does not block | does not block | does not block | BLOCKS L2 (Catalog #290 strict-flip pending; will refuse) | BLOCKS L3 | does not block |
| **7** 3-stack composition design | does not block | does not block | does not block | does not block (NSCS03 L2 is single-substrate) | does not block | BLOCKS composition smoke/full |
| **8** Catalog #271 codex review | does not block (smoke `<$1`; cost-gated review not triggered) | BLOCKS until codex `approve` or `advisory` | does not block | BLOCKS L2 implicitly (no full CUDA = no L2) | BLOCKS L3 | BLOCKS implicitly |

**Summary of blockers per dispatch stage:**

- **Smoke**: Item 1 (only blocker).
- **Full CUDA**: Items 1 + 8 (Item 8 fires after gate-flip; Items 2+3 recommended-first).
- **Paired CPU**: Items 1 + 8 + prior full CUDA archive existence.
- **L2 promotion**: Items 1 + 6 + 8 + prior full CUDA empirical evidence.
- **L3 promotion (production-hardened)**: ALL items except 7 (composition is separate substrate-class).
- **3-stack composition**: Items 1 + 6 + 7 + 8 + ≥2-of-3 individual CUDA anchors.

---

## Section 4 — Council-grade items (escalation)

**4 of 8 items are council-grade** (Items 1, 2, 3, 4, 7 — but Item 7 is composition not strictly per-substrate). Per CLAUDE.md "Council conduct" + sextet pact + Assumption-Adversary seat:

### Item 1 (Recipe gate-flip)

- **Council members required:** Shannon LEAD (R(D) feasibility) + Dykstra CO-LEAD (Pareto-region projection given uncalibrated λ_R) + Yousfi (steganalysis perspective on whether the codec is contest-eligible at default config) + Fridrich (rate-distortion perspective on $80 burn for uncalibrated dispatch) + Contrarian (challenge "ship now without calibration") + Assumption-Adversary (challenge the FRAMING: must we sweep BEFORE flipping?).
- **HARD-EARNED-vs-CARGO-CULTED classification:**
  - HARD-EARNED: `dispatch_enabled: false` until trainer ready + recipe-vs-trainer-state consistency per Catalog #240 (PRESERVED — citation: Z3 v2 / Z4 / Z5 paired $2 waste anchor; trainer is now ready, gate-flip is operator choice).
  - CARGO-CULTED: requiring Phase 2 council adjudication BEFORE any smoke fire. The recipe `smoke_only: true` already permits smoke; the question is whether to ALSO allow full at default config OR sweep-first. Subagent C's plan implicitly assumes "sweep first" but never PROVES that default `λ_R=0.5` is suboptimal.
- **Per-round assumption-statement discipline (Catalog #292):** EACH council member must explicitly state the shared assumption they're operating within. Example expected statements:
  - Shannon: "The shared assumption I am operating within is that Ballé 2018's rate-distortion theorem applies at our operating point of `25*archive_bytes/37545489` rate scaling."
  - Dykstra: "The shared assumption I am operating within is that the Pareto-feasibility region at λ_R=0.5 is empirically non-empty for comma video at 384×512."
  - Assumption-Adversary: "The shared assumption is that we MUST sweep before dispatching. Violation hypothesis: default `0.5` may be within ±5% of optimal; the sweep cost ($6–20) may exceed the ΔS improvement at the chosen full-dispatch budget."

### Item 2 (λ_R sweep)

- **Council members required:** Same quintet + Ballé (rate-distortion lineage) + MacKay (MDL perspective on λ_R's role in achieving the rate-distortion floor) + Assumption-Adversary.
- **HARD-EARNED-vs-CARGO-CULTED classification:**
  - HARD-EARNED: "Rate-distortion has an optimal operating point per substrate" (PRESERVED — Shannon canonical).
  - CARGO-CULTED: The specific sweep range `[0.01, 0.1, 0.5, 1.0, 5.0]` is inherited from Ballé 2018 natural-image regime; our comma video at 25× rate scaling may need different. Plus: assuming a sweep is required AT ALL (vs first-principles bisection).
- **Per-round assumption-statement discipline:** Ballé member must state "the shared assumption is that the λ_R sweep range from Ballé 2018 maps to our rate normalization"; Assumption-Adversary challenges whether the Ballé sweep range is appropriate.

### Item 3 (σ-floor sensitivity)

- **Council members required:** Same as Item 2 + Hotz (engineering-shortcut perspective on whether σ-floor matters at all vs analytical lower bound).
- **HARD-EARNED-vs-CARGO-CULTED classification:**
  - HARD-EARNED: "Entropy bottleneck needs a lower bound on scale to avoid log(0) numerical instability" (PRESERVED — Ballé 2018 canonical + GDN numerics).
  - CARGO-CULTED: The SPECIFIC value `1e-4` is engineering-convenience; Ballé 2018 paper uses `~3.8e-6`; comma at 384×512 may need different.

### Item 4 (EMA differentiated split)

- **Council members required:** Quintet + Ballé (specific Ballé recipe member) + Quantizr (empirical EMA-shadow practitioner; CLAUDE.md "EMA — non-negotiable" canon source) + Assumption-Adversary.
- **HARD-EARNED-vs-CARGO-CULTED classification:**
  - HARD-EARNED: "EMA shadow at inference + decay value ≥ 0.99 + apply at eval only + snapshot+restore pattern" (PRESERVED — citation: CLAUDE.md "EMA — non-negotiable"; cost of violating = single-epoch noise dominates).
  - CARGO-CULTED: "Single EMA decay across joint encoder + hyperprior" (ELIGIBLE — Ballé 2018 uses differentiated; we adopted single without empirical proof).
- **Decomposition allowed:** Yes — keep `0.997` for main encoder + add `0.999` for hyperprior; document via `# EMA_DECAY_DIFFERENTIATED_OK:balle2018_recipe` waiver. Council blesses the decomposition.

### Item 7 (3-stack composition design)

- **Council members required:** Full inner quintet + Carmack (engineering shortcut perspective on cascade vs joint) + Selfcomp (block-FP composition lineage) + MacKay (cross-paradigm composition MDL perspective) + Assumption-Adversary (challenge: is 3-stack the right composition, or should we wait for 4th/5th individual anchor?).
- **HARD-EARNED-vs-CARGO-CULTED classification:**
  - HARD-EARNED: "Composition requires ≥1 verified [contest-CUDA] anchor per component" (PRESERVED — CLAUDE.md "Forbidden cross-archive composition" anchor).
  - CARGO-CULTED: The specific composition pattern (sequential cascade vs joint multi-objective vs substrate-engineering scaffold) is engineering convenience; no empirical evidence yet on which works best for orthogonal-axis substrates.
- **Note:** This item is DEFERRED until at least 2 individual NSCS substrates land contest-CUDA anchors; council adjudication is BLOCKED-pending-empirical-evidence per the canonical forbidden pattern.

---

## Section 5 — Recommended next-action sequence (ranked by EV/$)

Ranked by `|predicted ΔS lower bound| / cost` per CLAUDE.md
META-ASSUMPTION REVIEW + canonical decision flowchart.

### Path A (RECOMMENDED): Calibrate-first, then dispatch (~$8–22 total before any A100 spend)

1. **(council deliberation; ~1 hour)** Council convenes for Items 1+2+3+4 simultaneously per CLAUDE.md "Council conduct" sextet pact + per-round assumption-statement discipline (Catalog #292). Assumption-Adversary challenges the FRAMING ("must we sweep before flipping?"). Output: council-approved λ_R sweep range + σ-floor candidates + EMA differentiated split decision + gate-flip authorization conditional on sweep success.

2. **(Item 4 implementation; ~2h editor)** Implement EMA differentiated 0.999 hyperprior + 0.997 main split in trainer + document via `# EMA_DECAY_DIFFERENTIATED_OK:balle2018_recipe_canonical_for_joint_codec` waiver. Update landing memo's canonical-vs-unique table (Layer 7) from CANONICAL ADOPT to UNIQUE per Phase 2.

3. **(Items 2+3 co-sweep; ~$6–20 / 4–5h wall-clock)** Run paired λ_R + σ-floor sweep on Lightning T4 (cheaper than A100 smoke per substrate-engineering tradeoff): `λ_R ∈ {0.1, 0.5, 1.0, 5.0} × σ_floor ∈ {1e-3, 1e-4, 1e-5}` = 12-config matrix OR reduced 5-config (sister recipes default `λ_R=0.5/σ=1e-4` + 4 perturbations). Tag every result `[contest-CUDA Lightning T4]`. Pick lowest-scoring config per "Apples-to-apples evidence discipline".

4. **(Item 1 gate-flip; ~5 min editor)** Edit recipe: `dispatch_enabled: true`, `dispatch_blockers: []`, update `env_overrides.NSCS03_LAMBDA_R` + `NSCS03_SIGMA_FLOOR` to council-chosen values from Step 3. Update lane registry notes with sweep evidence.

5. **(Item 8 codex pre-dispatch review; ~$0.50–2.00 / ~5–10 min)** Automatic via `tools/operator_authorize.py`. Operator addresses findings if any.

6. **(NSCS03 A100 smoke; ~$1.50–4.00 / 30–60 min)** Per Subagent C plan Section 2.4. Validates calibrated config at A100 scale.

7. **(NSCS03 A100 full; ~$60–80 / 2–12h)** Per Subagent C plan Section 2.4. Produces first `[contest-CUDA Modal A100]` anchor.

8. **(Paired CPU eval; ~$0.10–0.50 / 30–90 min)** Per Catalog #246 anchor-skip-aware after CUDA full lands. Produces `[contest-CPU Linux x86_64 Modal]` anchor for L2 promotion completeness per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable.

9. **(Memo backfill; ~30 min)** Sister subagent adds `## 9-dimension success checklist evidence` + `## Predicted ΔS band` + `## Stack-of-stacks composition matrix` sections + literal `## Canonical-vs-unique decision per layer` heading to landing memo (Item 6).

10. **(L2 promotion; ~5 min)** Update lane registry: `real_archive_empirical=true`, `contest_cuda=true`, `contest_cpu=true` evidence fields. Per Catalog #233 4-gate canonical: smoke green + Tier C MDL density + 100ep auth-eval anchor + Catalog #127 custody validated.

**Total cost (Path A):** $8–22 sweep + $1.50–4.00 smoke + $60–80 full + $0.10–0.50 paired = `$70–107`. Wall-clock: 8–18h (council 1h + EMA impl 2h + sweep 4–5h parallel + smoke 1h + full 2–12h + paired 1.5h).

### Path B (FAST-AND-LOOSE): Skip calibration, dispatch at default

1. Council Item 1 gate-flip only; defer Items 2+3+4 as documented research_only opt-outs.
2. Recipe edit: `dispatch_enabled: true`; keep `dispatch_blockers: [lambda_R_sweep_calibration_pending, sigma_floor_sensitivity_pending]` as known gaps; add notes "default λ_R=0.5 / σ=1e-4 used pending Phase 2 council".
3. Codex pre-dispatch review.
4. A100 smoke + full at default config. ~$62–84.
5. Paired CPU + L2 promotion (but L2 will carry "calibration deferred" tag).

**Total cost (Path B):** $62–84. Wall-clock: 4–14h.

**Path B risk:** Default `λ_R=0.5` may be far from optimal; full A100 burn could land at score significantly above the achievable floor for this substrate. The $50–80 burn is unrecoverable if score is poor. Catalog #239 `long_burn` boundary classifies this as the highest-cost dispatch class — sweep-first is the canonical risk mitigation.

### Path C (DEFERRED-AND-STACK): Wait for NSCS01/NSCS02 anchors first

1. Land NSCS06 (already ready), NSCS01, NSCS02 contest-CUDA anchors first per Subagent C OR-1 / OR-3.
2. Re-evaluate NSCS03 priority after those anchors reseed the cathedral autopilot ranker per Catalog #227 Tier C + cost-band posterior.
3. If NSCS03 still highest-EV after re-ranking, proceed with Path A.

**Total cost (Path C):** $0 NSCS03-specific until after re-evaluation; Subagent C OR-1/OR-2/OR-3 envelope `~$25–50` for the prior anchors.

### Recommended path: **Path A** (calibrate-first)

Reasoning:
1. NSCS03 is the HIGHEST cost-band substrate (~$60–80 full + A100 vs T4 for sisters). The $8–22 calibration sweep is `12–28%` of the full dispatch cost — cheap insurance per Catalog #239 `long_burn` discipline.
2. Per HARD-EARNED-vs-CARGO-CULTED classification: λ_R + σ-floor + EMA decay are ALL CARGO-CULTED defaults; the canonical decision flowchart says "burden of proof on PROVING canonical-is-better" (Rule 3) when unmeasured. Calibration produces the empirical evidence.
3. The recipe itself flagged 3 of these 4 items as `dispatch_blockers` — the recipe is the canonical operator-decision surface; honor it.
4. Per CLAUDE.md "Apples-to-apples evidence discipline": better to spend $20 measuring than $80 guessing.
5. Path B's $80 risk dominates Path A's $22 + EV gain from picking the right config.

### Top-3 op-routables (this session, executable order):

1. **Council deliberation (Items 1+2+3+4)** — INVOKE the sextet pact + Assumption-Adversary seat + Ballé + Quantizr + MacKay. Per-round assumption-statement discipline mandatory. Output: ledger at `.omx/research/grand_council_nscs03_phase_2_calibration_<DATE>.md` documenting decisions per Catalog #292.

2. **EMA differentiated split implementation** (Item 4) — sister subagent edits `experiments/train_substrate_nscs03_end_to_end_balle_joint_codec.py` `_full_main` to instantiate two EMA scopes; updates landing memo Layer 7; ~2h editor.

3. **λ_R + σ-floor co-sweep dispatch** (Items 2+3) — `tools/run_modal_smoke_before_full.py --recipe substrate_nscs03_end_to_end_balle_joint_codec_modal_a100_dispatch --smoke-only --smoke-gpu A100` per CLAUDE.md "NEVER invent CLI flags" + Catalog #167. Iterate across `λ_R ∈ {0.1, 0.5, 1.0, 5.0}` via env override `NSCS03_LAMBDA_R`. Sister sweep via env override `NSCS03_SIGMA_FLOOR ∈ {1e-3, 1e-4, 1e-5}`.

---

## Section 6 — Final summary

- **Total OPERATOR-DECISION items found:** 8
- **Decision-flowchart classification per item:**
  - OBVIOUS-FIT: 4 (Items 1, 5, 6, 8)
  - UNCLEAR (empirical measurement required): 2 (Items 2, 3)
  - PRINCIPLED FORK: 1 (Item 4)
  - UNCLEAR + council adjudication required: 1 (Item 7)
- **Council-grade item count:** 4 (Items 1, 2, 3, 4) — Item 7 is council-grade but DEFERRED-pending-empirical-evidence
- **Pre-dispatch blocker count per stage:**
  - Smoke: 1 (Item 1)
  - Full CUDA: 2 (Items 1, 8)
  - Paired CPU: 2 (same) + prior full archive
  - L2 promotion: 3 (Items 1, 6, 8) + prior CUDA evidence
  - L3 promotion: 6 (Items 1, 2, 3, 4, 6, 8) + L2 evidence
  - 3-stack composition: 5 (Items 1, 6, 7, 8) + ≥2-of-3 individual anchors
- **Recommended unblock sequence:** Path A (calibrate-first) — `Council → EMA differentiated split → λ_R+σ-floor co-sweep → recipe gate-flip → codex review → smoke → full → paired CPU → memo backfill → L2 promotion`. Total `$70–107` / `8–18h` wall-clock.

---

## Process compliance footer

- **Pre-flight reads completed (Catalog #229 premise verification):**
  CLAUDE.md "Design decisions" + "Submission auth eval — BOTH CPU AND CUDA" + "Council conduct" + "UNIQUE-AND-COMPLETE-PER-METHOD" + "Forbidden empirical-claim-without-evidence-tag"; landing memo `feedback_nscs03_full_main_implementation_pr95_balle_2018_paradigm_landed_20260515.md`; Subagent A audit ledger; Subagent C dispatch plan; recipe YAML; trainer file + TIER_1 manifest + EMA / λ_R / σ-floor argparse surface; lane registry NSCS03 entry; sister recipes (NSCS01/NSCS02 dispatch_blockers compared); operator-authorize wrappers (all 4 exist; Subagent C plan Section 2.4 prereq-op #1 obsolete).
- **Premise verification corrections vs Subagent C:** Subagent C Section 2.4 prereq-op #1 ("Generate `scripts/operator_authorize_substrate_nscs03_..._.sh`") is INCORRECT — wrapper already exists (3.8 KB, generated). Updated Section 0 of this ledger to reflect actual state.
- **Checkpoint discipline (Catalog #206):** 4 checkpoints recorded at `.omx/state/subagent_progress.jsonl` with `lane_id=lane_nscs03_operator_decision_items_investigation_20260515`.
- **Disjoint sister-subagent scope (Catalog #230):** READ-ONLY against source; only this ledger written under `.omx/research/`. No overlap with NSCS06 dispatch / wrapper-generator / memo-backfill / NSCS02-trimmer sister subagents.
- **Apples-to-apples evidence discipline:** every score / cost claim tagged: `[empirical:<path>]` for in-repo file measurements; `[first-principles bound; literature anchor only; no measured NSCS03-on-comma-video data]` for predictions; `[contest-CUDA Lightning T4]` / `[contest-CUDA Modal A100]` / `[contest-CPU Linux x86_64 Modal]` for dispatch-stage scoring axes.
- **No CLI flags invented (CLAUDE.md "NEVER invent CLI flags"):** All CLI mentions verified against trainer argparse surface (`grep "add_argument"` in trainer file) + Subagent C plan's verified canonical entry points.
- **6-hook wire-in declaration (CLAUDE.md Catalog #125):**
  1. **Sensitivity-map contribution**: N/A — this is an investigation ledger, not a measurement; downstream council deliberation may produce sensitivity outputs.
  2. **Pareto constraint**: N/A — no new constraint declared; existing NSCS03 substrate contract carries `rate_distortion_v1` constraint.
  3. **Bit-allocator hook**: N/A — no per-tensor importance change at the investigation surface; NSCS03 substrate carries `ibps_kkt` hook.
  4. **Cathedral autopilot dispatch hook**: ACTIVE — investigation output feeds council deliberation, which feeds dispatch decisions; the recommended Path A sequence is consumable by `tools/cathedral_autopilot_autonomous_loop.py` via the recipe gate.
  5. **Continual-learning posterior update**: ACTIVE — each Path A dispatch will trigger `posterior_update_locked` per existing trainer wiring + `cost_band_calibration.append_anchor(outcome=...)` per Catalog #175/#177.
  6. **Probe-disambiguator**: N/A — this is an investigation ledger; the 4-config λ_R sweep IS the probe-disambiguator for Item 2; the 3-config σ-floor sweep IS the disambiguator for Item 3; these are recommended-actions, not landed-probes.
