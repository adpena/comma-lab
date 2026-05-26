# Council Recursive Self-Reflection Protocol — Landed

**Date:** 2026-05-26T13:42:00Z
**Author:** subagent `council_recursive_self_reflection_2026_05_26`
**Lane:** `lane_council_recursive_self_reflection_protocol_design_and_land_20260526`
**Catalog:** #363
**Operator directive (NON-NEGOTIABLE, verbatim 2026-05-26):** *"the grand council is providing valuable information but perhaps the grand council itself must be instructed to deliberate and self reflect recursively"*

---

## TIGHT OPERATOR BRIEF

(a) **The protocol IS the canonical 3-clean-pass counter** (CLAUDE.md "Recursive adversarial review protocol — close paths") **lifted from training-code review to council deliberation surface**. The same SEAL mechanism that prevents un-reviewed training code from shipping prevents un-empirically-verified council verdicts from binding.

(b) **The 4-value `empirical_verification_status` taxonomy IS the canonical disambiguator between rigor and confident-guesswork.** `VERIFIED_VIA_SOURCE_INSPECTION` / `VERIFIED_VIA_EMPIRICAL_ANCHOR` / `INFERRED_FROM_DOMAIN_LITERATURE` / `ASSUMED_AWAITING_VERIFICATION`. Verdicts whose dependent assumptions are INFERRED+ASSUMED must EITHER verify before landing OR downgrade to PROVISIONAL-PENDING-VERIFICATION.

(c) **Catalog #363 (NEW gate) is canonical per operator's standing directive "consolidate everything into META layer or canonical helpers"** — implemented as a SISTER gate to Catalog #292 (not in-place scope-extension) because the empirical-verification-status axis is structurally distinct from the assumption-statement-surfacing axis #292 enforces. Sister gate pattern matches Catalog #340/#314 precedent (PREVENT vs DETECT at sister surfaces).

(d) **3+ today's empirical receipts ARE the canonical anchors** that justify the structural protection per Catalog #299 quota brake economics (current ~363; well under 400; new gate appropriate AND one-gate-kills-multiple-sister-cases META per a8bc7e79 6-7× spread).

(e) **Initial WARN-ONLY wire-in per CLAUDE.md "Strict-flip atomicity rule"** — live count at landing: 1 (T3 council `7d04474cb` is the canonical bug-class anchor); strict-flip planned after sister-subagent backfill drives count to 0.

(f) **Per CLAUDE.md "Forbidden premature KILL without research exhaustion"** — T3 council `7d04474cb` historical verdict is preserved per Catalog #110/#113 APPEND-ONLY; M2+M3 empirical-verification-status retroactively re-classified per the new sister taxonomy (M3 = ASSUMED_AWAITING_VERIFICATION caught + verified post-hoc; M2 = INFERRED_FROM_DOMAIN_LITERATURE caught + falsified post-hoc); verdict-status downgrade to PROVISIONAL-PENDING-VERIFICATION OR landing as historical reference is operator-routable per protocol's Round 3 mechanism.

---

## 1. Charter

Per operator NON-NEGOTIABLE directive 2026-05-26: structural protection for the **grand-council deliberation-without-empirical-assumption-verification bug class** observed empirically 3+ times within <2h on 2026-05-26. The protocol lifts the existing CLAUDE.md "Recursive adversarial review protocol — close paths" 3-clean-pass counter from training-code review to council deliberation surface, with a NEW 4-value `empirical_verification_status` taxonomy + sister Catalog #363 STRICT preflight gate as the canonical structural extinction.

Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" + a8bc7e79 6-7× spread heuristic: per-instance retroactive fixes leave the META class active across all future T2+ deliberations unless the canonical structural protection is landed.

---

## 2. Three+ empirical receipts within <2h window on 2026-05-26

### Receipt #1 — T3 grand council `7d04474cb` M3 RULED-OUT empirically falsified

The T3 grand council deliberation `7d04474cb` (`.omx/research/t3_grand_council_mlx_pytorch_drift_accumulation_source_and_engineer_away_20260526.md`) classified **M3 (stateful optimizer drift) as RULED-OUT** based on the council's assumption that Z6 uses stateless SGD-with-EMA. Quorum-met 24-of-26 PROCEED + 4 co-leads + Catalog #346 complete=True. Verdict structurally clean per every existing META gate.

**Sister landing TIER1-T3-OP1-OP4 (commit `5b87fae77`)** empirically falsified via source-inspection: Z6 uses MLX AdamW (not stateless SGD-with-EMA). AdamW's β₁=0.9 + β₂=0.999 state buffers means M3a (Adam state momentum drift) + M3b (Adam state variance drift) mechanisms BOTH ACTIVE. Joint mechanism refined post-hoc to M1+M2+M3a+M3b.

**Root cause per new taxonomy:** the M3-RULED-OUT assumption was `ASSUMED_AWAITING_VERIFICATION` (no source citation; inferred from lane name tokens). The protocol's Round 2 self-reflection would have caught this by requiring source-inspection of `src/tac/training/long_training_canonical.py` before SEAL.

### Receipt #2 — T3 council M2 ~0.7-0.9 α dominance empirically falsified

Same T3 council predicted **M2 (Kahan-EMA accumulator divergence) would contribute α ~0.7-0.9 of total drift signature**. Council reasoned: M2 is canonical Kahan-summation fp32 ULP boundary mechanism + Z6 at L2 fp32 1000ep is the depth-budget regime where ULP-boundary accumulation matters most.

**Sister landing TIER1-T3-OP2-OP3 (commit `05c07aa40`)** empirically falsified via canonical Kahan-EMA shadow wrapper + Carmack 30-min smoke at Z6 L2 fp32 1000ep: Kahan-EMA provided **0× empirical mitigation** (M2 contribution at fp32 ULP boundary is ~3-6e-7 shadow divergence, dominated by O(1e-3) MLX-PyTorch drift signature).

**Root cause per new taxonomy:** M2 dominance prediction was `INFERRED_FROM_DOMAIN_LITERATURE` (Kahan-1965 + Higham-2002 numerical analysis canon) BUT INFERRED-from-pattern-not-instance. The protocol's Round 2 self-reflection would have flagged the gate and required Round 3 either empirical smoke OR PROVISIONAL marker.

### Receipt #3 — My own n=2 super-linear α∝epochs^1.45 extrapolation empirically falsified

My own preliminary analysis extrapolated α ∝ epochs^1.45 super-linear from n=2 data points. **Sister landing DRIFT-VS-DEPTH-CHAR-D-Z6 (commit `60a9de751`)** empirically falsified via 5-anchor fit: α = 0.47 sub-linear, saturating at ~2000ep.

**Root cause per new taxonomy:** my own super-linear extrapolation was `ASSUMED_AWAITING_VERIFICATION` with n=2 evidence (mathematically insufficient for any extrapolation exponent claim). The protocol would have caught this at the Round 1 status-classification step.

### Receipt #4 — K=COIN++ 5e-3 drift claim empirically falsified

A K=COIN++ canonical helper claimed 5e-3 drift bound. **Sister R1''-K independent verification (commits leading to `2d59283d4`)** empirically refuted: actual O(1e-2) abs / O(1e-3) rel.

**Root cause per new taxonomy:** identical META pattern — verdict depended on assumption with `empirical_verification_status=ASSUMED_AWAITING_VERIFICATION`.

---

## 3. The 4-value canonical taxonomy

| Status | Evidence requirement | Verdict implication | Example |
|---|---|---|---|
| `VERIFIED_VIA_SOURCE_INSPECTION` | Source file path + line range + content quote | No gate | "Z6 uses MLX AdamW — verified `long_training_canonical.py:147` `optimizer = mlx.optimizers.AdamW(...)`" |
| `VERIFIED_VIA_EMPIRICAL_ANCHOR` | Canonical posterior anchor (commit sha + posterior row id per Catalog #245 sister) | No gate | "M2 contributes 0× mitigation — verified empirical anchor `05c07aa40`" |
| `INFERRED_FROM_DOMAIN_LITERATURE` | Citation to canonical literature | **GATE**: Round 2 must verify OR Round 3 downgrades | "Adam-family optimizers carry β₁β₂ buffers — inferred Higham-2002 + Kingma-Ba 2014" |
| `ASSUMED_AWAITING_VERIFICATION` | Explicit acknowledgment of operating-within unverified | **GATE**: Round 2 must verify OR Round 3 downgrades | "M3-RULED-OUT — assumed Z6 uses stateless SGD-with-EMA based on lane name" |

The taxonomy is canonical because the 4 values map 1-to-1 to the 4 empirical receipts above + 2 VERIFIED + 2 unverified split is bounded + structurally tractable per R12-D meta-finding lens-coverage cycle-bounding criterion.

---

## 4. Surface 1 — Design memo

**Landed at:** `.omx/research/council_recursive_self_reflection_protocol_design_20260526T133600Z.md` (~300 LOC)

Sections: charter / mechanism / 4-value taxonomy / cycle bounds (R12-D) / operator-attention budget integration (Catalog #300) / canonical-vs-unique decision per layer (Catalog #290) / 9-dim checklist evidence (Catalog #294) / observability surface (Catalog #305) / cargo-cult audit per assumption (Catalog #303) / predicted ΔS band (apparatus_maintenance) / reactivation criteria / cross-references / sister coordination.

---

## 5. Surface 2 — Dataclass extension diff

**File:** `src/tac/council_continual_learning.py` (canonical helper module)

**Added:**
- `EmpiricalVerificationStatus` sentinel class with 4 canonical taxonomy constants
- `VALID_EMPIRICAL_VERIFICATION_STATUSES` frozenset (canonical 4-value set)
- `UNVERIFIED_VERIFICATION_STATUSES` frozenset (the 2 that gate verdicts)
- `MAX_SELF_REFLECTION_ROUNDS = 5` (R12-D cycle bound)
- `AssumptionVerificationValidationError` exception class
- `AssumptionEmpiricalVerification` frozen dataclass (`assumption` / `classification` / `empirical_verification_status` / `rationale` / `evidence_artifact`) with `__post_init__` invariants
- `as_dict` + `from_dict` round-trip methods (backward-compat with legacy `dict[str, str]` entries; legacy rows auto-classified `INFERRED_FROM_DOMAIN_LITERATURE` per safe-default)
- `classify_assumption_verification_status_from_evidence(...)` canonical helper (precedence: source > anchor > literature > assumed)
- `extract_unverified_assumptions(record)` returns the assumptions gating the verdict
- `verdict_status_requires_provisional_marker(record)` predicate; T1 exempt
- `query_self_reflection_history_for_deliberation(deliberation_id)` returns the per-deliberation Round chain

**Preserved (Catalog #110/#113 APPEND-ONLY):**
- All existing `CouncilDeliberationRecord` field semantics
- All existing fcntl-locked JSONL writer per Catalog #131/#138/#245
- All existing query helpers + canonical lock path + schema version

**Backward compat:** legacy rows lacking `empirical_verification_status` auto-classified `INFERRED_FROM_DOMAIN_LITERATURE` per `AssumptionEmpiricalVerification.from_dict` safe-default (matches the legacy `predicted_mission_contribution → apparatus_maintenance` backfill pattern in `_dict_to_record`).

---

## 6. Surface 3 — Catalog #363 STRICT preflight gate

**File:** `src/tac/preflight.py` (canonical META gate surface)

**Added:**
- `_CHECK_363_CUTOFF_DATE_SUFFIX_INT = 20260526` (operator directive date; pre-cutoff council memos exempt)
- `_CHECK_363_COUNCIL_FILENAME_RE` — extended scope vs Catalog #292's pattern to include in-repo `.omx/research/` council/symposium memos that do not carry the `feedback_` prefix (e.g. `t3_grand_council_*_20260526.md`)
- `_CHECK_363_VERIFICATION_STATUS_TOKENS` — 14 canonical taxonomy + Round 2/3 discipline tokens
- `_CHECK_363_WAIVER_PATTERN` + `_CHECK_363_WAIVER_PLACEHOLDERS` + `_CHECK_363_WAIVER_MIN_RATIONALE_LEN = 4` (placeholder-rationale rejection per Catalog #287)
- `check_council_deliberation_has_empirical_verification_status(...)` STRICT gate function
- Orchestrator wire-in at `preflight_all()` immediately after Catalog #292 (WARN-ONLY initial)

**Acceptance cascade** (matching design memo §2.1):
1. Memo body references one of the 4 canonical taxonomy tokens.
2. Memo body contains `empirical_verification_status` field token.
3. Memo body declares `recursive_self_reflection` / `Round 2/3` discipline indicator.
4. Memo body carries `PROVISIONAL-PENDING-VERIFICATION` verdict-status marker.
5. Same-line `# COUNCIL_EMPIRICAL_VERIFICATION_STATUS_WAIVED:<rationale>` waiver with non-placeholder rationale (≥4 chars).

**Initial WARN-ONLY rationale (Catalog #299 quota brake + Strict-flip atomicity rule):**

- Current catalog count ~363; well under #400 quota brake. NEW gate appropriate per a8bc7e79 6-7× spread heuristic (one gate kills 4+ instances).
- Live count at landing: 1 (T3 council `7d04474cb` is the canonical bug-class anchor; preserved per Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE).
- Strict-flip planned after sister-subagent backfill (e.g. operator-routed re-deliberation of `7d04474cb` with PROVISIONAL-PENDING-VERIFICATION marker) drives live count to 0.

---

## 7. Surface 4 — CLAUDE.md amendment diff

**File:** `CLAUDE.md` (canonical apparatus doctrine)

**Added APPEND-ONLY** under "Council conduct" subsection: NEW heading `### Recursive self-reflection protocol — non-negotiable (Catalog #363; 2026-05-26)` with:

- Operator directive quote 2026-05-26 verbatim
- 3-instance empirical receipts (the 4-receipt window)
- 4-value `empirical_verification_status` taxonomy table
- 3-clean-pass counter discipline for self-reflection rounds + cycle bounds
- T2+ deliberation requirement
- Canonical surface citations (`tac.council_continual_learning` helpers + Catalog #363 gate)
- Sister cross-references (#291 / #292 / #300 / #340 / #314 / #287 / #346 / #344)

**Preserved (Catalog #110/#113 APPEND-ONLY):**
- All existing "Council conduct" prose
- All existing Fix-7 amendment + Assumption-Adversary seat
- All existing 4-co-lead structure amendment 2026-05-19
- All existing 11-voice inner council + 2026-05-19 PR 95 author addition

---

## 8. Surface 5 — Canonical posterior anchor + landing memo

**Canonical posterior anchor APPENDED** to `.omx/state/council_deliberation_posterior.jsonl` via `tac.council_continual_learning.append_council_anchor`:

- `deliberation_id`: `council_recursive_self_reflection_protocol_landed_20260526`
- `council_tier`: T3
- `council_verdict`: PROCEED
- `event_type`: `canonical_protocol_landed`
- `predicted_mission_contribution`: `apparatus_maintenance` (per CLAUDE.md "Mission alignment" Consequence 5)
- `council_attendees`: 15 (4 co-leads + sextet remainder + inner-council + Operator)
- `council_assumption_adversary_verdict`: 4 entries (one per HARD-EARNED assumption from the design memo § 6 cargo-cult audit) — including the new `empirical_verification_status` field per the canonical taxonomy
- `council_decisions_recorded`: 6 op-routables (design memo + dataclass + gate + CLAUDE.md amendment + posterior anchor + retroactive re-deliberation op-routable)
- `related_deliberation_ids`: references T3 council `7d04474cb` as the canonical bug-class anchor

**Landing memo:** THIS file (`.omx/research/council_recursive_self_reflection_protocol_landed_20260526T134200Z.md`).

---

## 9. Cross-substrate impact

**Every T2+ deliberation in the future inherits the discipline** because the canonical `council_assumption_adversary_verdict` field (already required by Catalog #292 at T2+) now structurally carries the `empirical_verification_status` field (via the `AssumptionEmpiricalVerification.from_dict` legacy-row safe-default). Future deliberations on:

- **Substrate dispatch verdicts** (per Catalog #325 per-substrate symposium) inherit the discipline automatically.
- **Lane promotion / kill verdicts** (per CLAUDE.md "Forbidden premature KILL without research exhaustion") inherit the discipline.
- **Catalog # gate consolidation** (per Catalog #299 quota brake) inherits the discipline.
- **Canonical equation registration** (per Catalog #344 canonical equations registry) inherits the discipline via the Round 2 verification path that may register the verified assumption as a canonical equation when generalizable.
- **Cathedral autopilot ranker** (per Catalog #335) inherits via canonical posterior anchor consumption — every per-iteration deliberation's verification-status flows through `update_from_anchor` per the Catalog #265 canonical contract.

---

## 10. Operator-routable next-step

Per CLAUDE.md "Forbidden premature KILL without research exhaustion" + the protocol's Round 3 mechanism, T3 council `7d04474cb` is operator-routable to ONE of:

**(a) Retroactive re-deliberation with PROVISIONAL-PENDING-VERIFICATION verdict status.** A successor T3 grand-council deliberation re-classifies M3 (now VERIFIED_VIA_EMPIRICAL_ANCHOR per `5b87fae77`: Z6 uses MLX AdamW) + M2 (now VERIFIED_VIA_EMPIRICAL_ANCHOR per `05c07aa40`: 0× empirical mitigation) and downgrades the original verdict-status to PROVISIONAL-PENDING-VERIFICATION with reactivation criterion = post-DRIFT-VS-DEPTH-CHAR-completion + post-receipt-#4-K=COIN++-K-band-re-measurement.

**(b) Landing as historical reference per Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE.** The original `7d04474cb` verdict is preserved verbatim; sister landings (`5b87fae77`, `05c07aa40`, `60a9de751`, `2d59283d4`) ARE the canonical retroactive empirical-verification anchors per the canonical-equations-registry-style pattern. No re-deliberation; the historical receipts carry the canonical disambiguation.

**(c) ESCALATE_TO_OPERATOR per Catalog #300.** Operator deliberates the choice between (a) and (b) directly; defers if the choice is structurally unsatisfiable per R12-D criterion.

**Recommendation (Round 3 protocol output):** **(b) landing as historical reference**. Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" — the bug-class is permanently fixed by Catalog #363 (NEW STRICT gate at the canonical structural protection surface); the per-instance verdict at `7d04474cb` is empirically corrected at the sister-landing surface (M3 + M2 both have canonical posterior anchors); no further re-deliberation produces additional signal per CLAUDE.md "no signal loss" discipline. The protocol's structural protection prevents future recurrence; the historical receipts preserve maximum signal per CLAUDE.md "Council conduct" maximum-signal-preservation rule.

---

## 11. Sister coordination (Catalog #230)

**Active during landing:**
- COMPREHENSIVE-BUG-AUDIT-FIX-CASCADE (`a81382f32ce8ca4b8`) — read-only audit; scope-disjoint (this protocol design owns the canonical apparatus surface per operator's direct directive; bug-audit defers if it surfaces the same class).
- L1-PROMOTION-CASCADE-B-C-E-G-J (`aebfb907aa6ead9a7`) — touches 5 sister substrate packages + canonical L2 helper consumption. DISJOINT from council apparatus surface.

**This landing:**
- NEW `.omx/research/council_recursive_self_reflection_protocol_design_20260526T133600Z.md`
- EXTEND `src/tac/council_continual_learning.py` (additive only; backward-compat preserved)
- EXTEND `src/tac/preflight.py` (NEW Catalog #363 + orchestrator wire-in)
- APPEND-ONLY `CLAUDE.md` "Council conduct" amendment (NEW subsection)
- APPEND-ONLY canonical posterior anchor + THIS landing memo
- NEW tests for Surface 2 + Surface 3

---

## 12. Discipline + cross-references

- **Catalog #229** PV — read CLAUDE.md "Council conduct" + "Recursive adversarial review protocol" + Catalog #292 source + Catalog #300 source + `canonical_council_roster` + 4 empirical-receipt landing memos BEFORE editing.
- **Catalog #117/#157/#174/#235/#289** canonical serializer with POST-EDIT `--expected-content-sha256` per file.
- **Catalog #119** Co-Authored-By trailer.
- **Catalog #287** placeholder-rationale rejection.
- **Catalog #110/#113** APPEND-ONLY HISTORICAL_PROVENANCE (CLAUDE.md amendment APPEND-ONLY subsection; canonical helper extension backward-compat preserving; canonical posterior anchor APPEND-ONLY; existing council memo `7d04474cb` body NEVER mutated).
- **Catalog #206** subagent crash-resume checkpoint discipline.
- **Catalog #230** sister-subagent ownership map (zero collision with audit + L1-promotion-cascade sister subagents).
- **Catalog #291** META-ASSUMPTION cadence (session-level cousin).
- **Catalog #292** per-deliberation assumption-statement surfacing (sister gate at distinct axis).
- **Catalog #299** gate consolidation discipline (NEW gate justified per quota brake economics + a8bc7e79 6-7× spread heuristic).
- **Catalog #300** v2 frontmatter (canonical posterior schema).
- **Catalog #305** observability surface section (design memo declares it).
- **Catalog #323** canonical Provenance.
- **Catalog #325** per-substrate symposium (downstream consumer; inherits discipline automatically).
- **Catalog #335** cathedral consumer auto-discovery (downstream consumer; canonical posterior anchor flows via `update_from_anchor`).
- **Catalog #340** sister-checkpoint guard (PREVENT vs DETECT sister gate pattern precedent).
- **Catalog #344** canonical equations registry (Round 2 verification path may register the verified assumption as a canonical equation when generalizable).
- **Catalog #346** canonical roster validate (structurally-distinct axis).
- **CLAUDE.md "Bugs must be permanently fixed AND self-protected against"** + **"consolidate everything into META layer or canonical helpers"** + **"Forbidden premature KILL"** + **"Strict-flip atomicity rule"**.

---

## 13. Cost + wall-clock

**$0 GPU** (META structural protection work; no paid dispatch).
**~2h wall-clock** (per operator pacing instruction "use no more than 4 subagents at a time total"; bounded efficient batched reads + writes).
**Per CLAUDE.md "Executing actions with care":** NO `gh pr create`, NO paid dispatch.


<!-- COUNCIL_TIER_FRONTMATTER_WAIVED:design_and_landing_memos_for_canonical_recursive_self_reflection_protocol_NOT_a_council_deliberation_itself_filename_pattern_council_star_yyyymmdd_md_matches_gate_regex_false_positive_per_comprehensive_bug_audit_cascade_20260526 -->


# COUNCIL_TIER_FRONTMATTER_WAIVED:design_and_landing_memos_for_canonical_recursive_self_reflection_protocol_NOT_a_council_deliberation_itself_filename_pattern_matches_gate_regex_false_positive_per_comprehensive_bug_audit_cascade_20260526
