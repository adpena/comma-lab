# Codex routing directive: cargo-cult-audit-section backfill sweep across existing design memos
# Date: 2026-05-18
# Authority: META-audit (commit `e86ca6d0c`) §2 + cargo-cult burn-down supplement (commit `fb102933b`) + operator standing directive "burn down all cargo culted and keep pushing" + "all operator decisions approved"
# Pairs with: prospective STRICT preflight gate `check_rate_attack_strategic_claim_has_receiver_path_evidence` (routing at `.omx/research/codex_routing_directive_strict_preflight_gate_rate_attack_strategic_claim_receiver_path_evidence_20260518.md` commit `fb102933b`)
# Companion to (NOT duplicating): Catalog #303 (`check_substrate_design_memo_has_cargo_cult_audit_section`; WARN-ONLY at landing; gate-side enforcement) — THIS directive is the OPERATIONAL BACKFILL sweep that drives the live count to 0 enabling strict-flip per CLAUDE.md "Strict-flip atomicity rule"

## STRATEGIC FRAMING

Catalog #303 currently lives WARN-ONLY at landing with **5 known violations** (per CLAUDE.md row: `atw_codec_v1_cargo_cult_unwind_design_20260516.md` + `c6_e4_mdl_ibps_cargo_cult_unwind_design_20260516.md` + `nscs02_downsampled_renderer_cargo_cult_unwind_design_20260516.md` + `sane_hnerv_cargo_cult_unwind_design_20260516.md` + `time_traveler_l5_cargo_cult_unwind_design_20260516.md`). These predate the gate landing date (>=2026-05-16). New design memos landed since (>=2026-05-17 forward) MAY have the section; gate scan = canonical source of truth.

Per the META-audit (12-claim self-audit CONFLATE_DECLARATIVE_WITH_PHYSICAL pattern) + the cargo-cult burn-down supplement (extending the META-pattern across 9 today's landings), the cargo-cult-audit section is the SINGLE STRUCTURAL ARTIFACT that surfaces per-assumption HARD-EARNED-vs-CARGO-CULTED classification per the addendum `feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515.md`. Without it, design memos slip cargo-culted assumptions past every adversarial review apparatus.

Per the Assumption-Adversary council seat (Catalog #292 sister): per-deliberation explicit assumption surfacing is the runtime mechanism; the cargo-cult audit section is the per-design-memo persistent artifact that captures the same discipline.

Empirical anchor: NSCS06 v6 → v7 = 44% improvement in ONE iteration via cargo-cult-unwind methodology. Every design memo lacking the audit section is leaving ~10-40% predicted-band improvement on the table (per the NSCS06 anchor + Z3-G1 / D1 / Z6-v2 sister anchors).

## CANONICAL POINTERS

1. `/Users/adpena/Projects/pact/CLAUDE.md` — especially:
   - "## Cargo-cult audit per assumption" canonical section spec
   - Catalog #303 `check_substrate_design_memo_has_cargo_cult_audit_section`
   - CLAUDE.md "FORBIDDEN PATTERNS" → "Forbidden force-canonical-without-evaluation-of-suppression" + "Forbidden symposium-band-prediction-without-Dykstra-feasibility-check" + "Forbidden closed-form-CDF-allocator-without-empirical-bit-spend-proof"
   - Catalog #292 `check_grand_council_deliberation_has_explicit_assumption_statements` (sister at per-deliberation surface)
   - Catalog #110 / #113 HISTORICAL_PROVENANCE APPEND-ONLY discipline
2. `/Users/adpena/Projects/pact/AGENTS.md`
3. `feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515.md` (HARD-EARNED-vs-CARGO-CULTED classification framework)
4. `.omx/research/meta_audit_conflate_declarative_with_physical_error_pattern_12_claim_self_audit_20260518.md` (commit `e86ca6d0c`; the META-audit)
5. `.omx/research/cargo_cult_burn_down_supplement_extending_meta_audit_across_session_20260518.md` (commit `fb102933b`; supplement)
6. `.omx/research/codex_routing_directive_strict_preflight_gate_rate_attack_strategic_claim_receiver_path_evidence_20260518.md` (commit `fb102933b`; sister gate routing — different surface)

## WHAT CODEX EXECUTES

### Phase 1: Inventory sweep

For every file under `.omx/research/*_design_<YYYYMMDD>.md` (also catching `*_scaffold_<YYYYMMDD>.md` per the Catalog #303 scan scope), determine:

1. Date suffix (`<YYYYMMDD>`) parsed from filename
2. Cargo-cult-audit-section present? (case-insensitive search for `## Cargo-cult audit per assumption`)
3. If absent AND date >= 2026-05-16 (Catalog #303 enforcement window): canonical backfill candidate
4. If absent AND date < 2026-05-16: legacy-exempt (Catalog #303 grandfathers pre-cutoff memos)
5. If present: PASS — no action

Output: `experiments/results/cargo_cult_audit_backfill_inventory_<utc>/inventory.json` with per-file verdict (PASS / CANONICAL_BACKFILL_CANDIDATE / LEGACY_EXEMPT / NOT_DESIGN_MEMO).

### Phase 2: Backfill HISTORICAL_PROVENANCE-compliant for each CANONICAL_BACKFILL_CANDIDATE

Per Catalog #110 / #113 HISTORICAL_PROVENANCE APPEND-ONLY discipline: the section is APPENDED to the existing memo body; the original memo content is PRESERVED untouched. Sister approach to the 2026-05-16 WAVE-1 APPARATUS HARDENING backfill of Catalog #305 (`## Observability surface`) sections.

Per-backfill protocol:
1. Read the existing memo
2. Identify the per-design assumptions (typically 5-10) the memo's design embodies — extract from the prose + the choice list + the predicted-band derivation
3. For each assumption, classify HARD-EARNED-VERIFIED vs CARGO-CULTED-PENDING-EMPIRICAL per the addendum framework:
   - HARD-EARNED-VERIFIED: cite source evidence (paper / commit / empirical anchor)
   - CARGO-CULTED-PENDING-EMPIRICAL: propose unwind-test plan
4. Append `## Cargo-cult audit per assumption` section with the per-assumption matrix
5. Each row format:
   ```
   ### Assumption N: <one-line statement>
   - **Classification**: HARD-EARNED-VERIFIED | CARGO-CULTED-PENDING-EMPIRICAL
   - **Source / rationale**: <citation OR null>
   - **Unwind-test plan** (if CARGO-CULTED): <concrete probe spec; $X budget; predicted band>
   ```
6. Commit each backfill SEPARATELY via canonical serializer with POST-EDIT working-tree sha256 per Catalog #117/#157/#174 (one backfill per commit avoids absorption-pattern per Catalog #314)

### Phase 3: Catalog #303 strict-flip readiness check

After all 5 known violations backfilled (NSCS06-Carmack-Hotz v6 + ATW V1 + C6-E4 MDL-IBPS + NSCS02-downsampled + sane_hnerv + time_traveler_l5):
1. Run `.venv/bin/python -m tac.preflight check_substrate_design_memo_has_cargo_cult_audit_section --strict`
2. If live count == 0: propose strict-flip in `src/tac/preflight.py` (operator-routable; do NOT auto-flip; produce CLAUDE.md row + preflight.py edit as a SEPARATE directive for operator review)
3. If live count > 0: surface remaining violations as op-routables (likely NEW landings between Phase 1 sweep and Phase 3 check)

### Phase 4: Recurring-cadence operationalization

Per CLAUDE.md "META-ASSUMPTION ADVERSARIAL REVIEW" non-negotiable + Catalog #291 (`check_session_has_recent_meta_assumption_review` — 7-day cadence): the cargo-cult audit backfill should become RECURRING (every new design memo triggers the audit at write-time, OR every 7-day cadence sweep validates that all post-cutoff memos have the section).

Phase 4 deliverable: routing directive for the recurring sweep (operator-tentative — write directive, do NOT execute Phase 4 in this session).

## DISCIPLINE

- Catalog #229 premise verification: read each existing memo BEFORE backfilling
- Catalog #287 evidence tags on every classification (HARD-EARNED requires `[empirical:<path>]` or paper citation)
- Catalog #117/#157/#174 commit serializer with POST-EDIT working-tree sha256
- Catalog #110/#113 HISTORICAL_PROVENANCE APPEND-ONLY (do NOT rewrite memo bodies; only APPEND the audit section)
- Catalog #186 NO catalog # claim needed (this is an operational sweep, not a new gate)
- Catalog #206 checkpoint discipline (Codex /goal LOOP already provides this)
- Catalog #314 absorption avoidance: one backfill per commit; declare files_touched
- Catalog #292 the canonical Assumption-Adversary classification framework is THE source-of-truth for per-assumption HARD-EARNED-vs-CARGO-CULTED verdict
- Catalog #325 if any backfilled memo's audit reveals CARGO-CULTED assumption requiring symposium: cite Catalog #325 per-substrate-symposium-evidence requirement
- Catalog #313 if any backfilled memo references probe outcomes: consult `.omx/state/probe_outcomes.jsonl`

## EXIT CRITERIA

- [ ] Phase 1 inventory complete + JSON artifact lands
- [ ] Phase 2: all 5 known violations backfilled (each via canonical serializer + POST-EDIT sha)
- [ ] Phase 3 strict-flip readiness check + verdict (PROPOSE_STRICT_FLIP / OPS_REMAINING)
- [ ] Phase 4 routing directive lands at `.omx/research/codex_routing_directive_recurring_cargo_cult_audit_sweep_<utc>.md`
- [ ] codex_persistent_session_state row appended per /goal LOOP discipline
- [ ] Memory entry `feedback_cargo_cult_audit_backfill_sweep_landed_<YYYYMMDD>.md` documents per-memo verdict + Phase 3 readiness

## OPERATOR-FACING NOTE

This routing directive operationalizes the META-audit §2 recommendation at the EXISTING-design-memo backfill surface. Sister of the prospective STRICT preflight gate routing directive (which protects FUTURE rate-attack strategic-claim memos at the source-text scan surface). Together they close the cargo-cult-discipline surface at BOTH the gate-protection level (prospective) AND the operational-debt-cleanup level (retrospective).

Per CLAUDE.md "Forbidden premature KILL without research exhaustion": this backfill IS the canonical research path for previously-classified-CARGO-CULTED assumptions; classification flips from CARGO-CULTED to HARD-EARNED-VERIFIED via empirical anchors are the natural outcome of the sweep.

Per CLAUDE.md "Gate consolidation discipline" Catalog #299: NO new gate needed (sweep operationalizes EXISTING Catalog #303 enforcement) — quota brake preserved.

Per operator standing directive 2026-05-18 "all operator decisions approved" + "burn down all cargo culted" + "continue meta consolidation and similar meta work across" — strategic mandate alignment confirmed.

— Main-Claude 2026-05-18 (META-audit §2 + cargo-cult burn-down operational sweep routing per operator burn-down + meta-consolidation mandate)
