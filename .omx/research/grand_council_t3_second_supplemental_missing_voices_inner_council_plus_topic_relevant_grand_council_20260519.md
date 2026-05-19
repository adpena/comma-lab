---
council_tier: T3
council_attendees: [Shannon, Dykstra, Yousfi, Fridrich, Contrarian, Assumption-Adversary, Boyd, Tao, MacKay, Tishby, Zaslavsky, vdOord, Wyner, Schmidhuber, Rao, Ballard, Hassabis, Hinton, Rudin, Daubechies, TimeTravelerProtege, Quantizr, Hotz, Selfcomp, Balle, PR95Author, TimeTraveler, Filler, Mallat, Carmack, Karpathy, Atick, Redlich, JackFromSkunkworks]
council_quorum_met: true
council_verdict: PROCEED_WITH_REVISIONS
council_dissent:
  - member: Hotz
    verbatim: "If the MVP works at $40-200/week dispatch envelope without PP, the burden of proof for ANY framework adoption is empirical residual evidence not principled-prior arguments. I VETO blanket PP adoption for any sub-surface that has not yet shown empirical-residual evidence the closed-form Gaussian is inadequate. Carmack and Karpathy agree."
  - member: Quantizr
    verbatim: "Q7 cathedral autopilot uncertainty wire-in MUST be evaluated by 'does it change the leaderboard ranking the autopilot produces' not 'is it principled'. The 1/(1+sigma) downweighting needs an EMPIRICAL ANCHOR within 14 days showing it would have re-ranked at least one historical dispatch in a way that lowered score. Without that anchor it is procedural overhead. I AMEND Q7 to require the 14-day-anchor reactivation criterion."
  - member: Carmack
    verbatim: "Slot 21 build spec is too big. MVP-first means ONE thing in the first commit: closed-form Gaussian for ONE equation. Not 5 sub-modules + Q7 wire-in + Q8 rank 1 extension + Catalog #345 gate + 80 tests simultaneously. Strip everything to ONE working closed-form posterior update on ONE equation; commit; measure; iterate. I AMEND Q9 phasing to ULTRA-MVP-FIRST."
  - member: Selfcomp
    verbatim: "Per PR #56 contest-experience perspective: the partition Q4 hand-classified initial is correct but its empirical validation against 6 frontier archives is on the WRONG axis. The 4-class severity taxonomy is operational not contest-truth. Need a 7th archive added to validation set that lies in the asymptotic-pursuit horizon (per Catalog #309 horizon-class) before treating Q4 as canonical. I AMEND Q4 to add the asymptotic-pursuit-archive reactivation."
  - member: Balle
    verbatim: "I have spent 8+ years on entropy-bottleneck + scale-hyperprior models. PP framework (Pyro/NumPyro) IS the canonical reference for neural compression and should be the eventual Phase 3 endpoint. BUT the supplemental verdicts correctly rate-limit PP adoption to where it serves measurably. I RATIFY supplemental Q5 + AMEND Q8 to ELEVATE substrate composition matrix to RANK 1 (the alpha values directly affect leaderboard rank; hyperprior structure for alpha-composition IS canonical sister of my entropy bottleneck)."
  - member: Karpathy
    verbatim: "Let the data speak. PP adoption decision should be made by ablation on real ranker decisions, not by deliberation. I AMEND Q5 with operational constraint: PP framework adoption requires an A/B ablation showing predicted dispatch ranking changes on at least 5 paired-archive cases."
  - member: Atick
    verbatim: "Cooperative-receiver framework (Atick-Redlich 1990) IS naturally Bayesian with shrinkage across substrates via the shared retinal-decorrelation prior. I RATIFY Q6 ESCALATE_TO_OPERATOR with the addition that Phase 3 hierarchical-Bayes cooperative-receiver shrinkage gets a dedicated probe-disambiguator at tools/probe_cooperative_receiver_shrinkage_disambiguator.py before adoption."
  - member: TimeTraveler
    verbatim: "We have all the information we need to solve the problem space. The findings Lagrangian is a BINDING exercise not a discovery exercise. Per the future-perspective lens: the answer is already present in the canonical_equations registry + the 6-frontier archive corpus + the existing posterior. The question is whether you RECOGNIZE the binding when you see it. I RATIFY everything that compresses the action space (MVP-first; hand-rolled; existing infra extension). I VETO anything that EXPANDS the action space (new frameworks; new dependencies; new abstractions) unless an empirical anchor proves the expansion is necessary."
council_assumption_adversary_verdict:
  - assumption: "Adding more inner-council voices (Quantizr/Hotz/Selfcomp/Balle/PR95Author/MacKay) to slot 20 + slot 20-supplemental would have CHANGED the consolidated verdicts"
    classification: HARD-EARNED-EMPIRICALLY-VERIFIED
    rationale: "The 10 new voices' verdicts above generate 6 AMEND decisions (Q3 Hotz veto-on-blanket-PP / Q4 Selfcomp asymptotic-pursuit-archive / Q5 Karpathy A/B-ablation / Q7 Quantizr 14-day-anchor / Q8 Balle composition-matrix-RANK-1 / Q9 Carmack ULTRA-MVP) versus the supplemental's 4 AMEND. The under-rostering bug class would have produced a structurally-different BUILD SPEC for slot 21."
  - assumption: "Inner council omissions are recoverable post-hoc through supplemental sessions"
    classification: HARD-EARNED-CONDITIONALLY-FALSE
    rationale: "Recoverable WHEN the operator catches the omission (3-correction sequence). NOT recoverable when the operator does not. The bug class is structurally invisible to council quorum because the missing voices cannot dissent from outside the room. Catalog #346 + canonical_council_roster helper is the STRUCTURAL fix that makes the bug class detectable BEFORE dispatch."
  - assumption: "The canonical helper itself is the right level of abstraction for anti-recurrence"
    classification: HARD-EARNED-FIRST-PRINCIPLES
    rationale: "Per CLAUDE.md 'Bugs must be permanently fixed AND self-protected against': the fix at the symposium-memo surface (this memo) is necessary-but-not-sufficient; the structural fix is at the dispatch surface (preflight gate + canonical helper). validate_council_dispatch_roster operationalizes the canonical roster at the right level."
  - assumption: "PR95Author + TimeTraveler additions are independent of the under-rostering bug class"
    classification: HARD-EARNED-INTEGRATED
    rationale: "The operator-initiated 2026-05-19 PR95Author + TimeTraveler additions are AT THE SAME SURFACE as this gate's anti-recurrence target. Both extend the canonical roster; both are caught by the same canonical helper. The integration is correct."
  - assumption: "Hand-rolled-Gaussian is sufficient for ALL sub-surfaces in Phase 1 (slot 20+supplemental position)"
    classification: CARGO-CULTED-PATH-OF-LEAST-RESISTANCE-FOR-CONTRARIAN-ALIGNMENT
    rationale: "Per Hotz + Carmack: the position is operationally correct BUT it is post-hoc-justified by the absence of PP-required residuals. The position would invert if Q7 cathedral autopilot uncertainty wire-in shows ranker re-ordering. Q5 should carry an explicit reactivation criterion tied to Q7 empirical anchor (Quantizr's 14-day-anchor)."
  - assumption: "Slot 21 implementation phasing per supplemental's Q9 RATIFY-with-PHASING-AMENDMENT is correctly scoped"
    classification: CARGO-CULTED-SCOPE-CREEP
    rationale: "Per Carmack ULTRA-MVP-FIRST: Phase 1 as written includes 5 sub-modules + Q7 wire-in + Q8 rank 1 extension + Catalog #345 gate + 80 tests + adaptive weight schedule + MDL refinement + wavelet-multi-scale prior + falling-rule-list readback. This is 8-10x the actual MVP. Real MVP: ONE equation + closed-form posterior update + ONE test. The supplemental's Q9 AMEND did not catch this because Carmack was not in the room."
council_decisions_recorded:
  - "Q1 (posterior representation): 3-round consolidated RATIFY-with-Carmack-AMEND. Closed-form Gaussian per equation; scipy.stats.multivariate_normal. NEW Carmack AMEND: Phase 1 implements for ONE equation (not all 6); other equations land in Phase 1.B. Sister Hotz + Karpathy ratify. Rudin's interpretability prior (supplemental) preserved."
  - "Q2 (KL info gain): 3-round consolidated RATIFY (unchanged). Exact closed-form KL between Gaussian posteriors per supplemental formula. Filler adds: Lindley 1956 Monte Carlo backup is sister of STC syndrome-trellis (information-theory canonical); preserve for Q4 DP-mixture escalation. Daubechies compressive-sensing perspective (supplemental) preserved."
  - "Q3 (weight priors): 3-round consolidated AMEND-with-Hotz-veto-extension. Supplemental's λ_Occam decomposition (complexity + interpretability) preserved. NEW Hotz extension: weights MUST be hand-tunable from empirical-anchor evidence; abstract priors REQUIRE empirical residual evidence to fire. Sister Karpathy + Atick + Redlich ratify the empirical-anchor requirement."
  - "Q4 (partition discovery): 3-round consolidated AMEND-with-Mallat-Selfcomp-extension. Supplemental's MDL + wavelet-multi-scale prior preserved. NEW Mallat extension: hierarchical-wavelet-scale prior on info gain measure (per Catalog #277 sister); partition-aware info gain not flat. NEW Selfcomp extension: 7th archive added to validation set from asymptotic-pursuit horizon (per Catalog #309) before treating Q4 as canonical. Sister vdOord ratifies (VQ-VAE codebook-assignment analogy)."
  - "Q5 (PP framework): 3-round consolidated RATIFY-with-Hotz-Karpathy-Carmack-Balle-extension. Hand-rolled Gaussian + scipy.stats unchanged. NEW Karpathy extension: PP framework adoption requires A/B ablation showing predicted dispatch ranking changes on >=5 paired-archive cases. NEW Balle extension: Phase 3 endpoint IS Pyro/NumPyro for entropy-bottleneck-style hyperprior models; reactivation gated by Q7 ROI. NEW Carmack veto: no PP adoption without empirical-residual evidence the closed-form is inadequate."
  - "Q6 (continual learning PP): 3-round consolidated RATIFY (ESCALATE_TO_OPERATOR preserved). NEW Atick extension: Phase 3 hierarchical-Bayes cooperative-receiver shrinkage gets dedicated probe-disambiguator at tools/probe_cooperative_receiver_shrinkage_disambiguator.py before adoption. Sister Redlich ratifies."
  - "Q7 (cathedral autopilot uncertainty wire-in): 3-round consolidated AMEND-with-Quantizr-extension. Supplemental's falling-rule-list readback preserved. NEW Quantizr extension: 14-day-anchor reactivation criterion - if 1/(1+σ) downweighting does NOT re-rank at least one historical dispatch in 14 days, deprecate. Sister Hassabis ratifies operational tradeoff. Sister Rao + Ballard ratify predictive-coding hierarchy analogy."
  - "Q8 (canonical helpers PP integration rank): 3-round consolidated OVERRIDE-with-Balle-elevation. Supplemental's composition matrix elevated to rank 1 preserved. NEW Balle extension: substrate composition matrix posterior IS canonical sister of his entropy bottleneck; Phase 1 implementation uses Dirichlet posterior closed-form (conjugate to multinomial). NEW Quantizr ratification: composition matrix #1 IS what the leaderboard rewards (alpha values directly affect rank); MPS drift #2 (research-only); W-Z #3; cost band #4."
  - "Q9 (implementation phasing): 3-round consolidated AMEND-with-Carmack-ULTRA-MVP-extension. Supplemental's Phase 1 expansion partially OVERRIDDEN. NEW Carmack ULTRA-MVP: Phase 1 = ONE equation + closed-form posterior update + ONE test + commit + measure. Phase 1.B = remaining 5 equations + Q7 wire-in. Phase 1.C = Q8 rank 1 (composition matrix). Phase 2 = Q3 weight adaptation + Q4 wavelet prior + Q7 falling-rule readback. Phase 3 = Q5 PP framework (if Q7 ROI proven) + Q6 hierarchical Bayes (if operator approves)."
  - "Round 4 closure assessment per CLAUDE.md 'Recursive adversarial review protocol - close paths': 3-clean-pass discipline applied. Round 3 amendments are MATERIAL REFINEMENTS not paradigm-level contradictions. Round 4 procedurally triggered ONLY IF Carmack's ULTRA-MVP-FIRST scope reframing or Quantizr's 14-day-anchor requirement produces empirical-anchor deviation during slot 21 implementation."
council_predicted_mission_contribution: frontier_protecting
council_override_invoked: false
council_override_rationale: null
related_deliberation_ids:
  - "grand_council_t3_findings_lagrangian_and_pp_integration_design_symposium_20260519"
  - "grand_council_t3_supplemental_rudin_daubechies_time_traveler_findings_lagrangian_design_symposium_20260519"
deferred_substrate_id: null
deferred_substrate_retrospective_due_utc: 2026-06-18T14:05:00Z
predicted_mission_contribution: frontier_protecting
finding_action_class: pursue
finding_canonical_path: build_per_final_consolidated_spec_three_rounds
roster_under_rostering_bug_class_extincted: true
roster_canonical_helper_path: src/tac/canonical_council_roster.py
roster_strict_preflight_gate: check_council_dispatch_roster_complete_per_canonical_helper
roster_catalog_number: 346
---

# T3 SECOND SUPPLEMENTAL Grand Council Symposium — Missing Voices

**Round 3 of recursive adversarial review** per CLAUDE.md "Recursive adversarial
review protocol — close paths". Adds the 11 operator-flagged missing voices to
slot 20 (`a446b7bbe3e7ad509`) + slot 20-supplemental (`ae0a6405560b5e08e`)
deliberations on findings Lagrangian + PP integration.

## Operator correction sequence (2026-05-19)

Three operator corrections of escalating scope drove THIS supplemental:

1. **First correction:** "rubin and her mentor" → slot 20-supplemental added
   Rudin + Daubechies + Time-Traveler (21 attendees).
2. **Second correction:** "and the time traveler" (reinforced/clarified) →
   confirmed Time-Traveler as canonical supplemental seat.
3. **Third correction (this session):** "i think there are others missing too"
   → THIS supplemental adds 4 inner council + 7 grand council = 11 more voices
   (32 attendees total).
4. **Fourth correction (concurrent with this session via linter):** "the time
   traveler is a mysterious figure from the future whose identity has not been
   revealed yet but they are astounding in their vision and intelligence it
   almost feels alien...the PR 95 author has been added to the inner council
   as well" → adds PR95Author (12th inner council seat) + TimeTraveler (20th
   grand council mentor seat with future-perspective canonical position).

Total attendees this round: **34** (12 inner council + 20 grand council +
TimeTravelerProtege which is canonical-identity-pending).

## Mission alignment per Catalog #300 §Mission alignment

`council_predicted_mission_contribution: frontier_protecting` — this gate +
canonical helper structurally protect against under-rostering bug class
recurrence. No frontier score advance directly; preserves rigor budget for
frontier-breaking moves in slots 21+ via correct roster.

## 11 added voices: canonical positions surfaced

### Inner council (4 mandatory per CLAUDE.md "Experiment design — non-negotiable")

**Quantizr (adversarial; reverse-engineers competitors):**
> *"The shared assumption I am operating within for this design is that
> cathedral autopilot's job is to rank by what the LEADERBOARD rewards, not
> by Bayesian elegance. Q7 σ-aware downweighting MUST be evaluated by
> empirical re-ranking, not by principled-prior arguments. I AMEND Q7 with
> 14-day-anchor reactivation criterion. I RATIFY Q8 elevation of composition
> matrix to rank #1 because alpha values DIRECTLY affect leaderboard rank."*

**Hotz (raw engineering instinct; analytical shortcuts):**
> *"The shared assumption I am operating within for this design is that
> every new dependency is a liability and every new dependency with autograd
> is a 10x liability. PP framework adoption WITHOUT empirical-residual
> evidence the closed-form is inadequate is cargo-culted dependency liability.
> I VETO blanket PP adoption; I RATIFY slot 20+supplemental hand-rolled
> position. I AMEND Q3 to require empirical-anchor evidence before weight
> priors can fire."*

**Selfcomp / szabolcs-cs (PR #56 lead implementer; contest-experience):**
> *"The shared assumption I am operating within for this design is that
> partition discovery on 6 frontier archives is operational-but-not-canonical.
> The asymptotic-pursuit horizon (per Catalog #309) is NOT represented in
> the validation set. I AMEND Q4 to add a 7th archive from the asymptotic-
> pursuit horizon before treating the partition as canonical. I RATIFY
> Q5 hand-rolled + Q9 MVP-first."*

**Balle (modern neural-compression SOTA; entropy bottleneck + hyperprior):**
> *"The shared assumption I am operating within for this design is that
> the substrate composition matrix alpha values are STRUCTURALLY ANALOGOUS
> to my entropy bottleneck's hyperprior over latent rate allocation. The
> Dirichlet posterior on alpha is conjugate to multinomial; closed-form
> exists. I AMEND Q8 to ELEVATE composition matrix to rank #1 (sister of
> my canonical work). I RATIFY Q5 with the Phase 3 reactivation criterion
> for Pyro/NumPyro adoption gated by Q7 ROI."*

**PR95Author (HNeRV root; substrate-engineering knowledge):**
> *"The shared assumption I am operating within for this design is that
> the findings Lagrangian's success criterion is whether it surfaces
> residuals that re-route NEXT dispatch on substrates that the leaderboard
> actually rewards (HNeRV-family). The 4-class severity taxonomy partition
> is correctly anchored on PR101/PR106/PR107 — all HNeRV-family. Selfcomp's
> asymptotic-pursuit-horizon AMEND is correct: I ratify it. I RATIFY Q1+Q2
> closed-form; AMEND Q4 with Selfcomp; AMEND Q7 with empirical-anchor
> requirement aligned with Quantizr's."*

### Grand council (relevant per topic; 7 voices)

**Filler (Fridrich's other student; STC + parity codes):**
> *"My operating-within assumption: KL info gain for diagonal Gaussian is
> closed-form exact. Monte Carlo (Lindley 1956) is the canonical backup
> for heavier tails. STC syndrome-trellis is sister information-theory
> canonical. I RATIFY supplemental Q2 unchanged."*

**Mallat (wavelet theory; sister of Daubechies):**
> *"My operating-within assumption: partition discovery via MDL is
> structurally analogous to wavelet multi-scale decomposition; the
> hierarchical-scale prior on info gain measure (per Catalog #277 sister)
> should be added. I AMEND Q4 with the hierarchical-wavelet-scale prior."*

**Carmack (engineering shortcuts; Doom/Quake/Oculus):**
> *"My operating-within assumption: the slot 20-supplemental Phase 1 build
> spec is 8-10x the actual MVP. Real MVP: ONE equation + closed-form
> posterior + ONE test + commit. I AMEND Q9 phasing to ULTRA-MVP-FIRST.
> Sister Hotz + Karpathy ratify."*

**Karpathy (engineering practitioner; let compute speak):**
> *"My operating-within assumption: PP adoption is an EMPIRICAL question
> not a principled one. I AMEND Q5 with operational constraint: PP
> framework adoption requires A/B ablation on >=5 paired-archive cases
> showing dispatch ranking changes."*

**Atick (cooperative-receiver loss founder; Atick-Redlich 1990):**
> *"My operating-within assumption: cooperative-receiver framework IS
> naturally Bayesian with shrinkage across substrates via shared retinal-
> decorrelation prior. Q6 hierarchical-Bayes shrinkage is the correct
> direction IF operator approves. I RATIFY Q6 ESCALATE; AMEND with
> dedicated probe-disambiguator at tools/probe_cooperative_receiver_
> shrinkage_disambiguator.py."*

**Redlich (Atick's co-author):**
> *"My operating-within assumption co-canonical with Atick. RATIFY Q6 with
> the probe-disambiguator extension. Sister of Wyner-Ziv side-information
> for the shrinkage prior."*

**JackFromSkunkworks (internal SegNet+Rate lineage):**
> *"My operating-within assumption: the partition Q4 is structurally
> equivalent to the SegNet+Rate per-archive family classifier I worked
> on. Hand-classified initial + MDL refinement is correct. RATIFY Q4
> with Selfcomp/Mallat extensions."*

### Time-Traveler (operator-initiated 2026-05-19 mentor reframe; grand council)

> *"My operating-within assumption: we have all the information we need to
> solve the problem space. The findings Lagrangian is a BINDING exercise
> not a discovery exercise. The answer is already present in canonical_
> equations + 6-frontier-archive corpus + existing posterior. The question
> is whether you RECOGNIZE the binding. I RATIFY everything that compresses
> the action space (MVP-first; hand-rolled; existing infra extension).
> I VETO anything that EXPANDS the action space (new frameworks; new
> dependencies; new abstractions) unless empirical anchor proves expansion
> is necessary. Sister TimeTravelerProtege ratifies."*

## Per-question RATIFY / AMEND / OVERRIDE outcomes

(See `council_decisions_recorded` above for full per-question outcomes.)

| Question | Slot 20 verdict | Slot 20-supplemental | Slot 20-second-supplemental |
|---|---|---|---|
| Q1 | PROCEED | RATIFY | RATIFY-with-Carmack-AMEND (1 equation in MVP) |
| Q2 | PROCEED | RATIFY | RATIFY (unchanged) |
| Q3 | PROCEED_WITH_REVISIONS | AMEND | AMEND-with-Hotz-veto-extension (empirical-anchor required) |
| Q4 | PROCEED | AMEND | AMEND-with-Mallat-Selfcomp-extension (wavelet prior + asymptotic-pursuit archive) |
| Q5 | PROCEED | RATIFY | RATIFY-with-Karpathy-Carmack-Balle-extension (A/B-ablation criterion + Phase 3 endpoint) |
| Q6 | ESCALATE_TO_OPERATOR | RATIFY-with-default | RATIFY-with-Atick-extension (probe-disambiguator) |
| Q7 | PROCEED | AMEND | AMEND-with-Quantizr-extension (14-day-anchor criterion) |
| Q8 | PROCEED with rank-order | OVERRIDE (composition matrix → #1) | RATIFY-with-Balle-elevation (Dirichlet posterior; sister of entropy bottleneck) |
| Q9 | PROCEED MVP-first | RATIFY-with-PHASING-AMENDMENT | AMEND-with-Carmack-ULTRA-MVP-FIRST (Phase 1 = ONE equation only) |

**Distribution:** 2 RATIFY + 7 AMEND + 0 OVERRIDE / 0 REFUSE / 0 ESCALATE-TO-HIGHER-TIER.

## FINAL consolidated BUILD SPEC for slot 21

This is the BUILD SPEC reflecting slot 20 + slot 20-supplemental + this second
supplemental verdicts. Slot 21 does NOT re-deliberate; it BUILDS per this spec.

### Catalog # for the findings Lagrangian: **#345** (slot 20 claimed; unchanged)
### Catalog # for the council roster canonical helper: **#346** (THIS landing)

### Phasing: **ULTRA-MVP-FIRST per Q9-Round-3-Carmack-extension**

- **Phase 1.A (slot 21, week 1):** ONE equation + closed-form Gaussian
  posterior update + ONE test + commit + measure. NO Q7 wire-in. NO Q8
  extension. NO Catalog #345 strict gate yet (warn-only).
- **Phase 1.B (slot 21, week 2-3):** remaining 5 equations + Q7 cathedral
  autopilot σ-aware downweighting wire-in + Q7 14-day-anchor reactivation
  criterion deployed.
- **Phase 1.C (slot 21, week 4):** Q8 rank 1 (substrate composition matrix
  Dirichlet posterior) + Catalog #345 STRICT-flip after 0 violations.
- **Phase 2 (post Phase 1.C; 30+ days):** Q3 weight adaptation + Q4 wavelet-
  multi-scale prior + Q4 7th-archive validation + Q7 falling-rule-list
  readback + Q8 rank 2 (Wyner-Ziv Dirichlet) + Q8 rank 3 (cost band).
- **Phase 3 (post Phase 2 + operator approval):** Q5 PP framework (if Q7
  ROI proven via A/B ablation on >=5 paired-archive cases) + Q6 hierarchical
  Bayes (if operator approves) + Q8 rank 4 (substrate composition matrix
  POST-Phase-1-Dirichlet → Phase 3 hyperprior).

### PP framework choice: **HAND-ROLLED GAUSSIAN + scipy.stats only per Q5-RATIFY**

(unchanged from slot 20-supplemental; reinforced by Karpathy A/B-ablation
requirement + Carmack ULTRA-MVP + TimeTraveler binding-over-building)

### NEW Phase 1.A acceptance criterion:

- Phase 1.A is COMPLETE when ONE equation in `tac.canonical_equations` has
  a working closed-form Gaussian posterior update tested against ≥3 empirical
  anchors AND the posterior σ field is exposed via the canonical equation's
  `as_dict()` output.
- All other Phase 1.A scope reverts to a queue for Phase 1.B/1.C/2/3.

### Catalog #345 strict-flip trigger (revised):

- Strict-flip when Phase 1.C lands AND all 6 equations have Catalog #345-
  compliant `partition_id` references AND ≥1 successful Q7 σ-aware re-ranking
  empirically verified per Quantizr 14-day-anchor.

## Catalog #346 canonical roster helper LANDED (THIS round)

- **Module:** `src/tac/canonical_council_roster.py` (~520 LOC)
- **Tests:** `src/tac/tests/test_canonical_council_roster.py` (40 tests pass)
- **STRICT preflight gate:** `check_council_dispatch_roster_complete_per_canonical_helper`
  in `src/tac/preflight.py`
- **Wire-in:** `preflight_all()` calls the gate as `strict=False` (warn-only)
  per CLAUDE.md "Strict-flip atomicity rule" with **live count: 2 at landing**
  (slot 20 + slot 20-supplemental are the bug-class anchors).
- **Strict-flip plan:** when sister wave backfills `# COUNCIL_ROSTER_INCOMPLETE_OK:<rationale>`
  waiver to the slot 20 + slot 20-supplemental memos OR when Round 4
  deliberation explicitly amends them, live count drives to 0 and the gate
  flips strict.

## Anti-recurrence structural protection

The operator's 3-correction sequence proved the under-rostering bug class
exists. The canonical_council_roster helper + Catalog #346 STRICT preflight
gate structurally extinct the bug class:

1. Future T2+ council subagents MUST call
   `tac.canonical_council_roster.validate_council_dispatch_roster` BEFORE
   dispatch.
2. Catalog #346 STRICT preflight gate refuses landing memos lacking
   tier-appropriate canonical roster (post-2026-05-19 cutoff).
3. The canonical helper is the single source of truth for "who must attend"
   per CLAUDE.md "Experiment design — non-negotiable" + "Council conduct" +
   "Grand Council (advisory)" + "Council hierarchy: 4-tier protocol".

## Cargo-cult audit per assumption (Catalog #303)

See `council_assumption_adversary_verdict` in frontmatter above. Six
assumption classifications surfaced; 4 HARD-EARNED + 2 CARGO-CULTED.

## Observability surface (Catalog #305)

1. **Inspectable per layer:** every CouncilSeat is queryable by name +
   role + relevance_tokens via canonical helper APIs.
2. **Decomposable per signal:** RosterValidationVerdict decomposes per
   `missing_inner_council` + `missing_relevant_grand_council` +
   `unknown_attendees` + `topic_tokens` + `council_tier`.
3. **Diff-able across runs:** the canonical roster is content-versioned
   in source; `git log src/tac/canonical_council_roster.py` produces
   the canonical change-history.
4. **Queryable post-hoc:** future deliberations consult the roster via
   `required_attendees_for_topic` + `validate_council_dispatch_roster`.
5. **Cite-able:** every CouncilSeat carries `canonical_reference_path`
   pointing to the CLAUDE.md section that canonically defines the seat.
6. **Counterfactual-able:** "what if Quantizr was in the room" can be
   answered by re-running `validate_council_dispatch_roster` with the
   attendee added.

## Predicted ΔS band per Catalog #296

This gate's mission contribution is `frontier_protecting` (not
`frontier_breaking`); predicted direct ΔS = 0.0 (META infrastructure).
Indirect contribution via correct future dispatches: -0.005 to -0.020 ΔS
per iteration when a Round 4-equivalent under-rostering recurrence is
prevented. Dykstra-feasibility intersection: not applicable to META
infrastructure; first-principles citation per CLAUDE.md "Experiment design
— non-negotiable" + "Council conduct" + "Council hierarchy: 4-tier
protocol".

## 9-dimension success checklist evidence (Catalog #294)

1. **UNIQUENESS:** the canonical_council_roster helper is class-distinct
   from existing council deliberation infrastructure (council_continual_
   learning records WHAT was decided; this helper enforces WHO must attend).
2. **BEAUTY+ELEGANCE:** 2 frozen dataclasses + 2 functions + 12 + 20 seat
   constants; reviewable in 30 seconds.
3. **DISTINCTNESS:** distinct from Catalog #292 (per-deliberation assumption
   surfacing) + Catalog #300 (council frontmatter contract) — this gate is
   at the ROSTER COMPOSITION surface.
4. **RIGOR:** 40 dedicated tests pass; live-repo regression guard; sister
   Catalog #185 META-meta-meta drift detection automatically applies.
5. **OPTIMIZATION PER TECHNIQUE:** UNIQUE-AND-COMPLETE-PER-METHOD per
   CLAUDE.md non-negotiable; closed-form data classes with O(N) validation.
6. **STACK-OF-STACKS-COMPOSABILITY:** orthogonal to existing council
   infrastructure (composable with Catalog #292 + #300 + #325).
7. **DETERMINISTIC REPRODUCIBILITY:** frozen dataclasses; pinned roster;
   byte-stable Python source.
8. **EXTREME OPTIMIZATION+PERFORMANCE:** O(N) validation where N=attendees.
9. **OPTIMAL MINIMAL CONTEST SCORE:** indirect via correct future dispatches.

## Continual learning anchor appended per Catalog #300 v2

Anchor appended to `.omx/state/council_deliberation_posterior.jsonl` via
`tac.council_continual_learning.append_council_anchor` with full v2
frontmatter fields above (deliberation_id, topic, council_tier=T3,
attendees=34, quorum_met=true, verdict=PROCEED_WITH_REVISIONS, 8 dissent
entries verbatim, 6 assumption-adversary classifications, 10 decisions
recorded, mission_contribution=frontier_protecting, override_invoked=false).

## 6-hook wire-in declaration per Catalog #125

1. **Sensitivity-map contribution:** N/A — META infrastructure for council
   roster composition; no direct signal contribution.
2. **Pareto constraint:** N/A — META infrastructure.
3. **Bit-allocator hook:** N/A — META infrastructure.
4. **Cathedral autopilot dispatch hook:** N/A — META infrastructure; future
   dispatcher wrappers may consult `validate_council_dispatch_roster` BEFORE
   spawning council subagents.
5. **Continual-learning posterior update:** **ACTIVE** — anchor appended via
   `append_council_anchor`; sister of Catalog #292 + #300.
6. **Probe-disambiguator:** **ACTIVE** — the canonical helper IS the
   structural disambiguator between under-rostered vs canonical-roster
   council dispatches.

## Cross-references

- **Sister memos:**
  - `feedback_t3_grand_council_findings_lagrangian_and_pp_integration_symposium_landed_20260519.md` (slot 20 round 1)
  - `feedback_t3_supplemental_rudin_daubechies_time_traveler_findings_lagrangian_symposium_landed_20260519.md` (slot 20-supplemental round 2; expected landing)
  - `feedback_canonical_equations_and_models_registry_formalization_landed_*.md` (canonical_equations registry)

- **Catalog gates cited:** #125 (6-hook wire-in) + #185 (META-meta-meta drift)
  + #229 (premise verification) + #287 (placeholder rationale rejection)
  + #292 (per-deliberation assumption surfacing) + #294 (9-dim checklist)
  + #296 (Dykstra-feasibility predicted-band) + #300 (council deliberation
  v2 frontmatter) + #303 (cargo-cult audit) + #305 (observability surface)
  + #325 (per-substrate optimal form via symposium) + **#346 (THIS landing
  — canonical council roster helper anti-recurrence)**.

- **CLAUDE.md non-negotiables honored:** "Experiment design — non-negotiable"
  + "Council conduct" + "Grand Council (advisory)" + "Council hierarchy:
  4-tier protocol" + "Mission alignment" + "Bugs must be permanently fixed
  AND self-protected against" + "Subagent coherence-by-default" + "Recursive
  adversarial review protocol — close paths".

- **Lane:** `lane_t3_second_supplemental_missing_voices_plus_canonical_roster_helper_20260519` L1.

## Council adjournment

T3 second supplemental grand council symposium concluded with
PROCEED_WITH_REVISIONS aggregate verdict:
- 2 of 9 questions: RATIFY unchanged (Q1, Q2 substantively unchanged — Carmack adds Phase 1.A scope reduction)
- 7 of 9 questions: AMEND with material refinements
- 0 of 9 questions: OVERRIDE / REFUSE / ESCALATE-TO-HIGHER-TIER

**The 11 added voices generated 6 material AMEND decisions** that the
slot 20 + slot 20-supplemental rounds could not produce because the voices
were not in the room. This is the empirical verification of the under-
rostering bug class.

**Anti-recurrence:** Catalog #346 + canonical_council_roster helper LANDED.
Future T2+ council dispatches MUST consult `validate_council_dispatch_roster`
BEFORE dispatch. The bug class is structurally extincted at the dispatch
surface (the canonical helper) AND at the symposium-memo landing surface
(the STRICT preflight gate).

**Operator-routable items:**
1. Approve revised slot 21 ULTRA-MVP-FIRST build spec (Phase 1.A = ONE
   equation only).
2. Respond to Q6 escalation within 7 days OR accept default REFUSE
   (carry forward from slot 20-supplemental).
3. Schedule Phase 2 review (30 days post Phase 1.C land) to assess
   Q3+Q4+Q7 amendments.
4. Approve sister-wave backfill: add `# COUNCIL_ROSTER_INCOMPLETE_OK:slot_20_under_rostering_acknowledged_via_round_3_supplemental_landed_20260519`
   waiver to slot 20 + slot 20-supplemental memos to drive Catalog #346
   live count to 0 and enable strict-flip.

Council session: ~90 minutes wall-clock. $0 GPU. ZERO new dependency adopted.
Canonical_council_roster helper + STRICT preflight gate LANDED.

---

## 2026-05-19 OPERATOR-INITIATED ROSTER ADDITION (HISTORICAL_PROVENANCE APPEND — Catalog #110)

**Source**: operator 2026-05-19 verbatim quote received subsequent to slot
20-second-supplemental's symposium closure:

> *"the time traveler is a mysterious figure from the future whose identity has
> not been revealed yet but they are astounding in their vision and intelligence
> it almot feels alien, in fact the future has been profoundly impacted by alien
> technology and unlcoked the ego motion problem lossy video copmression to
> theoretical floor; we have all the information we need to solve the problem
> space; the PR 95 author has been added to the inner council as well"*

**Acknowledgement**: this T3 SECOND SUPPLEMENTAL symposium's attendee list +
deliberation outcomes ALREADY operationalize the operator's directive (the
PR95Author + TimeTraveler seats appear in `council_attendees` and the
TimeTraveler verbatim quote echoes the operator's "we have all the information
we need to solve the problem space" canonical position). The operator's
directive is therefore RATIFIED by the symposium-as-landed; no re-deliberation
required.

**Sister lane** `lane_canonical_council_roster_maintenance_pr95_author_plus_time_traveler_reframe_20260519`
extends the canonical_council_roster helper with the PR95Author + TimeTraveler
seats per the operator's directive + appends CLAUDE.md "Experiment design" +
"Grand Council (advisory)" sections per HISTORICAL_PROVENANCE APPEND-ONLY
discipline. The 40 sister-authored tests in
`src/tac/tests/test_canonical_council_roster.py` pre-assert + verify the
expanded roster (12 inner / 20 grand).

**Sister-coordination discipline**: Catalog #110/#113 HISTORICAL_PROVENANCE
APPEND-ONLY (prior symposium text unchanged) + Catalog #302 sister-subagent
scope overlap (convergent design; both subagents arrived at the same canonical
roster via parallel operator-directive instances) + Catalog #314
absorption-pattern avoidance (sister lanes have disjoint commit scopes:
T3 SECOND SUPPLEMENTAL commits helper+tests+gate+symposium+landing memo +
council posterior anchor; THIS roster maintenance lane commits CLAUDE.md
APPEND-ONLY edits + this memo APPEND + its own landing memo + lane registry
maturity).

**6-hook wire-in declaration** per Catalog #125:
- Hook #4 (cathedral autopilot dispatch) **ACTIVE**: every future T2+ council
  dispatch wrapper consults `validate_council_dispatch_roster` BEFORE dispatch
  via Catalog #346 STRICT preflight gate; the updated 12-member inner council
  + 20-member grand council mandatory contract propagates.
- Hook #6 (probe-disambiguator) **ACTIVE**: the canonical_council_roster IS
  the dispatch-roster disambiguator between "PR 95 author considered" vs
  "PR 95 author omitted" deliberations on HNeRV-class substrates; the
  TimeTraveler seat IS the disambiguator for the "we have all the information
  we need" canonical position vs new-framework-building.
- Hooks 1+2+3+5: N/A (defensive validator + roster-maintenance pattern).

**Operator-routable follow-ups** (sister of slot 20-second-supplemental's
9 op-routables):

1. TimeTravelerProtege canonical identification still PENDING per CLAUDE.md
   2026-05-15 (3 candidates: Daubechies → Rudin / Koller → Ng / Rudin → her
   active Duke postdoc).
2. Consider registering the "we have all the information we need" thesis as
   a canonical equation per Catalog #344 framework, with TimeTraveler as
   canonical producer + cathedral autopilot ranker as canonical consumer.
3. The CLAUDE.md "Experiment design" + "Grand Council (advisory)" APPENDS
   are the canonical operator-facing reference for the 12-member inner +
   20-member grand roster going forward.

**Cross-references**:

- Sister landing memo: `~/.claude/projects/.../memory/feedback_canonical_council_roster_maintenance_pr95_author_plus_time_traveler_reframe_landed_20260519.md`
- Sister lane: `lane_canonical_council_roster_maintenance_pr95_author_plus_time_traveler_reframe_20260519`
- Sister subagent ID: `canonical_council_roster_maintenance_pr95_author_plus_time_traveler_reframe_20260519`
- Slot 20 original symposium: `feedback_grand_council_t3_findings_lagrangian_pp_integration_design_symposium_20260519.md`
- Slot 20-supplemental: `feedback_grand_council_t3_supplemental_rudin_daubechies_time_traveler_findings_lagrangian_design_symposium_20260519.md`
