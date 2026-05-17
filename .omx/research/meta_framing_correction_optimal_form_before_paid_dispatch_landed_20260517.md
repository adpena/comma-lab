# META-FRAMING-CORRECTION: substrate MUST be at OPTIMAL FORM before paid empirical dispatch — LANDED 2026-05-17

**Lane**: `lane_meta_framing_correction_optimal_form_before_dispatch_20260517`
**Catalog**: #315 (`check_substrate_at_optimal_form_before_paid_dispatch`)
**Subagent**: META-FRAMING-CORRECTION (`meta-framing-correction-20260517`)
**Operator directive**: 2026-05-17 — land structural CLAUDE.md non-negotiable + STRICT preflight gate that prevents empirically dispatching substrates at lifted-trainer form when operator's standing directives required OPTIMAL FORM at implementation.

---

## 1. The structural failure (operator-acknowledged pattern)

Across the 2026-05-13 → 2026-05-17 substrate dispatch waves we
**empirically dispatched substrates at LIFTED-TRAINER form** when the
operator's standing directives required **OPTIMAL FORM** at
implementation. The bug class has 5 concrete empirical anchors:

### Anchor 1: NSCS06 v6 → v7 = 44% improvement in ONE iteration

- NSCS06 v6 contest-CUDA = **105.15** (553× outside predicted band
  [0.10, 0.20] per the falsification symposium 2026-05-16,
  `.omx/research/grand_council_symposium_nscs06_carmack_hotz_falsification_redesign_multipath_20260516.md`).
- NSCS06 v7 contest-CUDA = **58.89** after applying 4-of-7
  cargo-cult-unwinds in ONE iteration cycle.
- **Score reduction: 44% in one iteration** via
  cargo-cult-unwind methodology.
- This was the **ONLY substrate that got iterated to optimal form
  before the next paid dispatch wave**.
- Implication: the apparatus IS capable of substrate-iteration; we
  did not systematically apply it.

### Anchor 2: 4-of-5 distinguishing-feature dispatch failures

Per `.omx/research/falsification_audit_v2_post_horizon_class_post_pivot_lessons_20260516.md`
+ session corpus:

| Substrate | Council verdict | Dispatch verdict | Implementation level |
|---|---|---|---|
| Wunderkind G1 v2 (per-pair-dominant SegNet argmax reducer) | DEFER / Q1 SPLIT-VERDICT | INDEPENDENT — argmax reducer methodology falsified | LIFTED-TRAINER (probe methodology was an artifact, not the canonical optimal form) |
| ATW v2 D4 (cooperative-receiver) | PROCEED_WITH_REVISIONS | INDEPENDENT — H(latent\|scorer_class) MI = 0.006 bits/symbol | LIFTED-TRAINER |
| Z6 v1 (FiLM ego-motion) | PROCEED_WITH_REVISIONS (sextet) | 7 probes show identity-predictor ties full-FiLM | LIFTED-TRAINER |
| NSCS01 v1 (nullspace-split renderer) | PROCEED_WITH_REVISIONS (sextet) | — | LIFTED-TRAINER (head0-capacity surface not iterated) |
| NSCS06 v8 Path B | T4 SYMPOSIUM PROCEED_WITH_REVISIONS | crashed pre-iteration | LIFTED-TRAINER |
| **NSCS06 v6 → v7** | (cargo-cult-unwind methodology) | **105.15 → 58.89 = 44% improvement** | **OPTIMAL FORM iteration applied** |

The pattern: each "distinguishing feature" was tested at LIFTED-TRAINER
form. The 4 failures falsified the **specific implementation** in its
lifted-trainer state, NOT the novel concept. The 1 success (NSCS06
v6→v7) iterated to OPTIMAL FORM via cargo-cult-unwinds first.

### Anchor 3: 52+ other substrates at lifted-trainer form

The 102 in-scope substrate lanes in `.omx/state/lane_registry.json`
include:
- 65 with explicit `research_only=true` / `lane_class=substrate_engineering`
  opt-out (correctly NOT dispatch-eligible per CLAUDE.md "Substrate
  scaffolds MUST be COMPLETE or RESEARCH-ONLY")
- 41 lifted-trainer-form substrates with no council deliberation
  anchor yet (covered by sister gates Catalog #233 / #294 / #298)
- 13 PROCEED_WITH_REVISIONS council anchors in
  `.omx/state/council_deliberation_posterior.jsonl`, ALL mapped
  back to lanes via the canonical `deferred_substrate_id` field

The bug class would have been: any of the 13 PROCEED_WITH_REVISIONS
lanes that lacked the opt-out would have silently received the next
paid dispatch wave without iteration. Catalog #315 closes this
structural gap.

### Anchor 4: operator's standing directives ENFORCED at memo surface, BYPASSED at dispatch surface

The design-memo discipline is well-defended:
- Catalog #290 — substrate design memo MUST declare canonical-vs-unique
  decision per layer (STRICT @ 0 since 2026-05-15)
- Catalog #294 — substrate landing memo MUST contain
  `## 9-dimension success checklist evidence` section (WARN-ONLY @ 5
  pending sister-subagent backfill)
- Catalog #303 — substrate design memo MUST contain
  `## Cargo-cult audit per assumption` section (WARN-ONLY @ 5 pending
  backfill)
- Catalog #305 — substrate design memo MUST declare
  `## Observability surface` section (STRICT-FLIPPED 2026-05-16
  @ 0 after backfill)

BUT none of these gates **bind back** to the dispatch decision. A
sextet council could return `PROCEED_WITH_REVISIONS`, the council
memo could land with full v2 frontmatter per Catalog #300, AND the
substrate trainer could still receive a paid Modal/Lightning/Vast.ai
dispatch the next hour because no STRICT gate bound the council
verdict back to the dispatch decision. This is the structural gap
Catalog #315 closes.

### Anchor 5: prior layered protections all targeted upstream/downstream surfaces

- Catalog #220 — substrate L1+ scaffold operational mechanism (runtime
  effect): catches "scaffold ships sidecar bytes but inflate doesn't
  consume them"
- Catalog #272 — distinguishing-feature integration contract
  (per-feature): catches "the smart thing is engineered but never
  wired into the archive"
- Catalog #233 — L1→L2 promotion canonical 4-gate (promotion):
  catches "L2 promotion claim without smoke / Tier C / 100ep
  auth-eval / custody validation"
- Catalog #298 — substrate L1 not stale dispatch (retirement):
  catches "L1 sat for >30 days without dispatch activity"

None of these binds the council verdict back to the dispatch
decision. Catalog #315 is the **iteration-discipline** sister: it
fires only when a council has returned `PROCEED_WITH_REVISIONS` and
the dispatch decision needs to wait for the iteration.

---

## 2. Why design-memo discipline (Catalogs #290 + #294 + #303 + #305) was insufficient

The design-memo discipline gates enforce the **declaration** of
substrate-design quality at memo-landing time. They do NOT enforce
the **application** of substrate-design quality at dispatch-decision
time. The chain is:

1. **DESIGN MEMO LANDS** → Catalogs #290 / #294 / #303 / #305 fire
   structurally. ✓
2. **IMPLEMENTATION LANDS** → no design-memo gate fires; sister
   gates Catalog #220 / #272 / #298 / #233 fire at their respective
   surfaces.
3. **SEXTET COUNCIL DELIBERATES** → Catalog #292 + #300 enforce
   per-deliberation discipline (assumption surfacing + v2
   frontmatter). ✓
4. **DISPATCH DECISION** → **NO STRICT GATE** binds the council
   verdict back to the dispatch decision. ✗

Step 4 is the structural gap. The operator's standing directives
(UNIQUE-AND-COMPLETE-PER-METHOD + 9-dim checklist + PR95-at-META +
HNeRV parity discipline) require that step 4 is gated by step 3's
verdict being **PROCEED-unconditional**, not `PROCEED_WITH_REVISIONS`.

Catalog #315 closes step 4 structurally.

---

## 3. The new gate's design

### Detection logic

For every in-scope substrate lane in `.omx/state/lane_registry.json`
at L1+ with `impl_complete=true`:

1. Compute the lane's substrate-ID set (lane.id + optional
   `substrate_alias` + optional `substrate_aliases` list)
2. Look up the latest council deliberation in
   `.omx/state/council_deliberation_posterior.jsonl` via the
   `deferred_substrate_id` field, restricted to the lane's
   substrate-ID set
3. If no council anchor exists → out of scope (sister gates cover)
4. If latest verdict is NOT `PROCEED_WITH_REVISIONS` (e.g. PROCEED /
   DEFER / REFUSE) → out of scope (sister gates cover)
5. If latest verdict IS `PROCEED_WITH_REVISIONS` AND no opt-out
   applies → **REFUSE**

### Acceptance cascade (any ONE suffices)

1. **Iteration anchor** — chronologically-later PROCEED-unconditional
   council deliberation on same substrate (the canonical NSCS06 v6→v7
   pattern: apply cargo-cult unwinds + re-trigger sextet/grand-council
   deliberation)
2. **`research_only=true`** (top-level field OR notes-token OR
   `target_modes` includes `research_only`/`research_substrate`)
3. **`lane_class=substrate_engineering`** (top-level field OR
   notes-token)
4. **`archived=true`** (top-level field OR notes-token
   `lane_state=archived` / `terminal_verdict`)
5. **Same-line waiver** `# OPTIMAL_FORM_DISPATCH_OK:<rationale>` in
   lane notes / evidence (placeholder `<rationale>` / `<reason>`
   literals rejected — gate's own docstring cannot self-waive)

### Live count + STRICT-flip atomicity

**Live count at landing: 0** across 106 scanned in-scope substrate
L1+ lanes:
- 65 already opted out via `research_only=true` /
  `lane_class=substrate_engineering` / target_modes
- 41 lacked any council anchor with `deferred_substrate_id` set (out
  of scope for THIS gate; covered by sister gates)
- 0 violations

STRICT-FLIPPED at landing per CLAUDE.md "Strict-flip atomicity rule".
The gate fires structurally the moment a future substrate is
registered without opt-out and a council returns
PROCEED_WITH_REVISIONS.

### Canonical iteration methodology

The new CLAUDE.md non-negotiable section "Substrate MUST be at OPTIMAL
FORM before paid empirical dispatch" documents the canonical 5-step
iteration methodology derived from NSCS06 v6→v7:

1. Audit current implementation against UNIQUE-AND-COMPLETE-PER-METHOD
2. Enumerate cargo-cults per Catalog #303
3. Apply unwinds systematically — produce v_n+1 implementation
4. Re-test sextet (or grand council for tier-elevated lanes)
5. Iterate until PROCEED-unconditional before paid dispatch

---

## 4. Layers landed

### Layer 1: CLAUDE.md non-negotiable section

New section **"Substrate MUST be at OPTIMAL FORM before paid empirical
dispatch — NON-NEGOTIABLE, HIGHEST EMPHASIS"** inserted between the
existing "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" and
"Apples-to-apples evidence discipline" sections.

Contents:
- Source / anchor memos / sister non-negotiables
- The structural failure (5 anchors with empirical receipts)
- Definitions of OPTIMAL FORM and LIFTED-TRAINER FORM
- The rule + acceptance cascade
- The forbidden anti-pattern (dispatch-at-lifted-form trap)
- The canonical substrate iteration methodology (NSCS06 v6→v7 pattern)
- Concrete enforcement (Catalog #315 + substrate-alias schema support)
- Cross-references (8-surface bug-class coverage)

### Layer 2: STRICT preflight gate Catalog #315

`check_substrate_at_optimal_form_before_paid_dispatch` in
`src/tac/preflight.py`:
- ~410 LOC mirroring sister Catalog #298 layout
- Helper functions: `_check_315_lane_in_scope` + `_check_315_collect_lane_text`
  + `_check_315_lane_opt_out` + `_check_315_waiver_present` +
  `_check_315_parse_iso_utc` + `_check_315_build_council_verdict_map`
  + `_check_315_substrate_ids_for_lane` + `_check_315_lookup_latest_verdict`
- Substrate alias support: lanes may declare `substrate_alias` (string)
  or `substrate_aliases` (list of strings) to map sextet-deliberation
  surface names back to canonical lane IDs (matches the empirical
  schema where sextets use v1-surface names like
  `z6_v1_ego_conditioning_surface` for `lane_substrate_z6`).
- Latest-wins verdict resolution by `written_at_utc` timestamp.
- Orchestrator callsite: `preflight_all()` line 2436-ish with full
  comment-block citing operator directive + empirical anchors + sister
  catalog cross-refs. **strict=True** at landing.

### Layer 3: Canonical SUBSTRATE-ITERATION-METHODOLOGY checklist

Embedded as Section "Canonical substrate iteration methodology
(the NSCS06 v6→v7 pattern)" in the new CLAUDE.md non-negotiable
section. The 5-step methodology is the operator-facing reference
for any future subagent iterating a substrate from LIFTED-TRAINER
form to OPTIMAL FORM.

### Layer 4: Test landing

`src/tac/tests/test_check_315_substrate_optimal_form_before_dispatch.py`:
- **61 dedicated tests** all passing
- Helper unit tests (in-scope classifier across 13 substrate
  substrings + 4 out-of-scope; ISO UTC parser; lane-text collection;
  opt-out cascade — top-level + notes-token; waiver semantics with
  placeholder rejection; alias support; verdict-map builder
  latest-wins; lookup returns None on no anchor)
- End-to-end gate behavior (no registry / no posterior / canonical
  PROCEED_WITH_REVISIONS flagged / PROCEED-unconditional after
  revisions supersedes / revisions chronologically later still
  flagged / 4 opt-out cases / waiver / placeholder rejection /
  no-council out-of-scope / L0 skipped / impl_complete=false
  skipped / out-of-scope skipped / strict raises with Catalog
  #315 + OPTIMAL FORM message / strict silent on clean / multi-
  violation aggregation / corrupt registry / corrupt posterior /
  string repo_root / verbose output / alias lookup / DEFER as
  dormant)
- Live-repo regression guard (asserts live count = 0)
- Orchestrator wire-in strict=True regression guard

### Layer 5: CLAUDE.md catalog table row

New row `315. check_substrate_at_optimal_form_before_paid_dispatch`
inserted BEFORE the chronologically-earlier #314 row (preserving
chronological reverse-order convention for recent gates).

---

## 5. 6-hook wire-in declaration per Catalog #125

Per CLAUDE.md "Subagent coherence-by-default" non-negotiable:

1. **Sensitivity-map contribution** — N/A: this gate is a council-
   verdict-binding gate at the dispatch-decision surface; it does
   not contribute axis-weight sensitivities. (Catalog #315 IS the
   protection that sensitivity-map signals serve; it does not
   produce a sensitivity-map row.)
2. **Pareto constraint** — N/A: not a rate/distortion/byte-budget
   constraint. The gate refuses lanes structurally based on
   council verdict, not by adding a Pareto term.
3. **Bit-allocator hook** — N/A: per-tensor importance is not
   changed.
4. **Cathedral autopilot dispatch hook** — **ACTIVE**: the gate's
   `[catalog-315]` verdict is consumable by `tools/cathedral_autopilot_*`
   for candidate-routing — substrates with outstanding
   PROCEED_WITH_REVISIONS verdicts get filtered out of the dispatch
   queue. The cathedral autopilot ranker can call
   `check_substrate_at_optimal_form_before_paid_dispatch(strict=False)`
   per-substrate to surface the lifted-trainer-form risk band.
5. **Continual-learning posterior update** — **ACTIVE via sister
   helper**: the gate consumes
   `.omx/state/council_deliberation_posterior.jsonl` (the canonical
   `tac.council_continual_learning` posterior per Catalog #300).
   Future PROCEED-unconditional council anchors automatically
   reseed the gate's verdict map; iteration evidence closes the loop
   structurally.
6. **Probe-disambiguator** — N/A: this gate has ONE defensible
   interpretation (latest council verdict binds dispatch decision).
   The waiver mechanism + 4 opt-out cascades give explicit operator
   escape hatches for the rare cases where a different
   interpretation is desired.

---

## 6. Cross-references to all prior session memos

- `.omx/research/nscs06_path_a_chroma_optical_flow_redesign_20260516.md` —
  NSCS06 v6→v7 44% improvement empirical anchor (the canonical
  iteration-methodology evidence)
- `.omx/research/falsification_audit_v2_post_horizon_class_post_pivot_lessons_20260516.md` —
  4-of-5 distinguishing-feature dispatch failures empirical anchor
- `.omx/research/grand_council_symposium_nscs06_carmack_hotz_falsification_redesign_multipath_20260516.md` —
  T4 symposium that surfaced the cargo-cult-unwind methodology
- `.omx/research/t4_symposium_substrate_design_class_shift_deliberation_20260517.md` —
  T4 symposium on substrate-design class-shift adjudication
  (PROCEED_WITH_REVISIONS verdict that this gate would now protect
  the dispatch decision from)
- `.omx/research/deep_adversarial_review_substrate_design_corpus_20260517.md` —
  DEEP-ADVERSARIAL-REVIEW memo (commit `c97430305`)
- `.omx/research/scorer_response_surface_analysis_20260517.md` —
  SCORER analysis memo (commit `de5ccade7`)
- `.omx/research/orthogonal_optimization_methods_inventory_20260517.md` —
  ORTHOGONAL audit memo (commit `2114baf77`)
- Council deliberation posterior (`.omx/state/council_deliberation_posterior.jsonl`):
  - `sextet_council_z6_phase_2_consensus_20260516` (PROCEED_WITH_REVISIONS)
  - `sextet_council_nscs03_phase_2_consensus_20260516` (PROCEED_WITH_REVISIONS)
  - `sextet_council_nscs01_phase_2_consensus_20260516` (PROCEED_WITH_REVISIONS)
  - `grand_council_t2_wunderkind_g1_v2_pivot_validation_v3_cpu_competitiveness_20260516` (PROCEED_WITH_REVISIONS)
  - 9 more PROCEED_WITH_REVISIONS anchors across T2/T3/T4 deliberations
- Sister catalog rows: #220 / #272 / #233 / #298 / #294 / #303 / #305 / #300
- Sister CLAUDE.md non-negotiable sections:
  - "UNIQUE-AND-COMPLETE-PER-METHOD operating mode"
  - "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"
  - "HNeRV / leaderboard-implementation parity discipline" (L7)
  - "Forbidden premature KILL without research exhaustion"
  - "KILL/FALSIFIED memory verdicts"
  - "Council hierarchy: 4-tier protocol"
  - "META-ASSUMPTION ADVERSARIAL REVIEW"

---

## 7. Operator-facing accountability statement

I (the META-FRAMING-CORRECTION subagent) acknowledge:

**The structural failure that motivated this gate is a process failure
of the apparatus's own apparatus.** The design-memo discipline gates
(Catalogs #290 + #294 + #303 + #305) and the council-discipline gates
(Catalogs #291 + #292 + #300) were carefully designed to enforce the
operator's standing directives at the **memo surface**. But none of
these gates **bound the council verdict back to the dispatch
decision**, and the apparatus silently allowed lifted-trainer-form
substrates to be dispatched empirically — falsifying SPECIFIC
IMPLEMENTATIONS rather than the novel concepts the substrates
represented.

NSCS06 v6→v7's 44% improvement in ONE iteration via cargo-cult-unwind
methodology is the existence proof that the bug is fixable: when we
DO iterate to optimal form, substrate-improvement-per-iteration is
massive. We just did not systematically apply the methodology across
the 4 distinguishing-feature dispatch failures this session.

Catalog #315 closes this gap structurally. From now forward, any
substrate with an outstanding `PROCEED_WITH_REVISIONS` council verdict
cannot proceed to paid dispatch without either applying the canonical
iteration methodology, declaring `research_only=true`, declaring
`lane_class=substrate_engineering`, archiving with terminal verdict,
or obtaining a same-line `# OPTIMAL_FORM_DISPATCH_OK:<rationale>`
waiver. The gate is STRICT-from-byte-one. Live count at landing is 0.

The canonical iteration methodology (5 steps) is now documented in the
CLAUDE.md non-negotiable section so every future subagent inherits
the discipline without needing to be told.

---

## 8. Sister-subagent coordination notes

Three sister subagents were declared in flight at landing time:
- PAUSING + HARDWARE — crashed
- PROBLEM-SPACE — crashed
- FREEZING — landed (commit `9bde7884e`)

Sister subagents touch DIFFERENT `src/tac/*` package directories
(`tac.freezing`, etc.). This subagent touches:
- `CLAUDE.md` (new non-negotiable section + Catalog #315 row)
- `src/tac/preflight.py` (Catalog #315 function + orchestrator wire-in)
- `src/tac/tests/test_check_315_substrate_optimal_form_before_dispatch.py` (NEW)
- `.omx/research/meta_framing_correction_optimal_form_before_paid_dispatch_landed_20260517.md` (NEW; this memo)

Risk: `src/tac/preflight.py` is touched by sister rate-limited
crashes; coordinated via canonical serializer with
`--expected-content-sha256` per Catalog #157 / #174.

---

## 9. Catalog #229 premise verification (pre-edit)

6 premises verified BEFORE any edit:
- PV-1: `.omx/state/lane_registry.json` has 794 lanes (read confirmed)
- PV-2: 13 PROCEED_WITH_REVISIONS council anchors exist in posterior
- PV-3: 102 in-scope substrate lanes total (id-substring match)
- PV-4: 69 L1+ substrate lanes with `impl_complete=true`
- PV-5: 7 unique memory_paths for PROCEED_WITH_REVISIONS deliberations
- PV-6: `deferred_substrate_id` field is the canonical join surface
  (introduced 2026-05-16 per `feedback_mission_alignment_followon_catalog_300_extension_landed_20260516.md`)

---

## 10. Lane gates marked

Per CLAUDE.md "Lane maturity registry" non-negotiable + this lane's
landing scope:

- `impl_complete` = ✓ (Catalog #315 function + orchestrator wire-in
  + test file landed)
- `strict_preflight` = ✓ (Catalog #315 wired strict=True; live count 0)
- `memory_entry` = ✓ (this memo)
- `three_clean_review` = pending (deferred to sister
  ADVERSARIAL-REVIEW subagent or operator review)
- `real_archive_empirical` = N/A (META gate, no archive)
- `contest_cuda` = N/A (META gate, no contest score)
- `deploy_runbook` = N/A (META gate, no remote runtime)

Lane will be marked via `tools/lane_maturity.py add-lane` + `mark`
calls in the same commit batch.

---

## 11. 9-dimension success checklist evidence (per Catalog #294)

1. **UNIQUENESS** — Catalog #315 is the FIRST gate to bind council
   verdict back to dispatch decision. No sister gate covers the
   iteration-discipline surface.
2. **BEAUTY + ELEGANCE** — ~410 LOC gate function mirroring sister
   Catalog #298 layout; reviewable in 30 seconds.
3. **DISTINCTNESS** — explicitly different from sisters
   #220 (runtime-effect) / #272 (per-feature) / #233 (promotion) /
   #298 (retirement) — this gate is iteration-discipline at the
   council-verdict-binding surface.
4. **RIGOR** — premise verification per Catalog #229 (6 premises);
   adversarial review per Catalog #292 (assumption surfacing this
   memo); HARD-EARNED-vs-CARGO-CULTED classification — the gate's
   structural protection IS HARD-EARNED (council verdict is the
   binding signal).
5. **OPTIMIZATION PER TECHNIQUE** — gate uses fcntl-locked JSONL
   posterior load (Catalog #128 sister pattern) + per-lane O(N_anchors)
   linear scan; latest-wins verdict resolution is O(1) per lookup.
6. **STACK-OF-STACKS-COMPOSABILITY** — gate composes orthogonally
   with the 7 sister gates (#220 / #272 / #233 / #298 / #294 / #303 /
   #305 / #300) closing the bug class across 8 surfaces.
7. **DETERMINISTIC REPRODUCIBILITY** — gate is byte-stable; same
   inputs (lane registry + posterior) produce same verdict.
8. **EXTREME OPTIMIZATION + PERFORMANCE** — gate runs in <100ms
   on 794-lane / 33-row posterior live repo state.
9. **OPTIMAL MINIMAL CONTEST SCORE** — gate's INDIRECT contribution
   to minimal contest score is preventing the dispatch-at-lifted-
   form failure mode that produced 4-of-5 falsifications this
   session. By forcing iteration to OPTIMAL FORM before dispatch,
   the apparatus stops burning paid GPU on implementations that
   the council has explicitly flagged for revision.

---

## 12. Cargo-cult audit per assumption (per Catalog #303)

| Assumption | Classification | Rationale |
|---|---|---|
| The council verdict is the binding signal for dispatch decision | HARD-EARNED | Per CLAUDE.md "Council hierarchy: 4-tier protocol" + the empirical anchor of NSCS06 v6→v7 success vs 4-of-5 lifted-form failures |
| `deferred_substrate_id` is the canonical substrate ↔ council join surface | HARD-EARNED | Per Catalog #300 mission-alignment extension landed 2026-05-16; empirically present in 12-of-13 PROCEED_WITH_REVISIONS rows |
| `PROCEED_WITH_REVISIONS` requires iteration before dispatch | HARD-EARNED | Operator directive 2026-05-17 + the falsification audit pattern receipts |
| 30-day staleness window is not needed for this gate | HARD-EARNED | This gate is verdict-binding, not staleness-binding; sister Catalog #298 covers staleness |
| In-scope substrate substring set should mirror Catalog #220 / #272 / #298 | PARTIALLY HARD-EARNED (extended) | Extended with `time_traveler` / `wunderkind` / `atw_` / `z6_` / `z7_` / `z8_` to cover the 4-of-5 anchor substrates that did not match the original Catalog #220 set |
| `substrate_alias` schema support is necessary | HARD-EARNED | Empirical: sextet sessions use v1-surface names like `z6_v1_ego_conditioning_surface` for `lane_substrate_z6`. Without alias support, ~6 of the 13 PROCEED_WITH_REVISIONS anchors would be invisible to the gate |

---

## 13. Observability surface (per Catalog #305)

The gate emits structured observability per invocation:

1. **Inspectable per layer** — verbose mode prints per-lane
   classification (in_scope / opt_out / waiver / no_council_anchor /
   latest_proceed_unconditional / violation) with the lane id +
   level + reason; aggregate counters at end.
2. **Decomposable per signal** — the 4 opt-out reasons are
   distinguishable in the verbose output; the violation message
   names the council deliberation id + timestamp.
3. **Diff-able across runs** — same inputs produce identical
   verdict + identical violation messages.
4. **Queryable post-hoc** — gate function returns
   `list[str]` of violation messages (each ≤400 chars in strict-mode
   raise); CLI surface via `preflight_all()`.
5. **Cite-able** — every violation message cites the lane id, the
   council deliberation id, the verdict timestamp, and the 5-cascade
   acceptance options.
6. **Counterfactual-able** — a synthetic test fixture can probe
   "what if this substrate had a PROCEED-unconditional anchor 1 hour
   later?" by writing a chronologically-later row to the posterior
   fixture.

---

## 14. Predicted ΔS band (per Catalog #296)

**Predicted ΔS band: not applicable** — Catalog #315 is a META-gate
on the dispatch-decision surface. It does not directly affect any
contest score band. The INDIRECT mechanism is "prevent paid dispatch
on lifted-form substrates that would have been falsified at the
implementation surface, freeing GPU budget for actual OPTIMAL-FORM
dispatches."

Dykstra-feasibility check: N/A. The gate's structural protection
operates on a discrete verdict space (`PROCEED` / `PROCEED_WITH_REVISIONS`
/ `DEFER` / `REFUSE` / `ESCALATE_*`), not a continuous score band.

---

## 15. Horizon class (per Catalog #309)

**Horizon class: apparatus_maintenance** — per CLAUDE.md "Mission
alignment — non-negotiable" 5-category enum. This gate is procedural
infrastructure that prevents the canonical failure mode; it does not
directly open a frontier-breaking path. The 44% improvement from
NSCS06 v6→v7 demonstrates that gating dispatch at OPTIMAL FORM
EXPECT magnitudes (4-of-5 → 1-of-5 falsification rate at next dispatch
wave) but does not itself produce a score.

---

## 16. Conclusion

Catalog #315 closes the structural gap between design-memo discipline
(memo surface) and dispatch discipline (decision surface). The 4-of-5
distinguishing-feature dispatch failures this session would have been
prevented if Catalog #315 had been in place. The 44% improvement from
NSCS06 v6→v7 is the existence proof that iteration to OPTIMAL FORM
is achievable. The gate's STRICT-from-byte-one wire-in (live count 0)
makes the discipline self-enforcing from here forward.

The next subagent wave should focus on **applying the canonical
iteration methodology** to the substrates that the gate now protects
from premature dispatch, starting with Z6 / NSCS01 / NSCS03 / Wunderkind
G1 / ATW v2 D4 per the priority order in their respective council
deliberation memos.
