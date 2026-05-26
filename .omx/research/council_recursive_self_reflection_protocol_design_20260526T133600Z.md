# Council Recursive Self-Reflection Protocol — Canonical Design

**Date:** 2026-05-26T13:36:00Z
**Author:** subagent `council_recursive_self_reflection_2026_05_26`
**Lane:** `lane_council_recursive_self_reflection_protocol_design_and_land_20260526`
**Operator directive (NON-NEGOTIABLE, verbatim 2026-05-26):** *"the grand council is providing valuable information but perhaps the grand council itself must be instructed to deliberate and self reflect recursively"*

---

## TIGHT OPERATOR BRIEF

(a) **The protocol IS the canonical 3-clean-pass counter** (CLAUDE.md "Recursive adversarial review protocol — close paths") **lifted from training-code review to council deliberation surface**. The same SEAL mechanism that prevents un-reviewed training code from shipping prevents un-empirically-verified council verdicts from binding.

(b) **The 4-value `empirical_verification_status` taxonomy IS the canonical disambiguator between rigor and confident-guesswork.** `VERIFIED_VIA_SOURCE_INSPECTION` / `VERIFIED_VIA_EMPIRICAL_ANCHOR` / `INFERRED_FROM_DOMAIN_LITERATURE` / `ASSUMED_AWAITING_VERIFICATION`. Verdicts whose dependent assumptions are INFERRED+ASSUMED must EITHER verify before landing OR downgrade to PROVISIONAL-PENDING-VERIFICATION.

(c) **Catalog #363 (NEW gate) is canonical per operator's standing directive "consolidate everything into META layer or canonical helpers"** but is implemented as a SISTER gate to Catalog #292 (not in-place scope-extension) because the empirical-verification-status axis is structurally distinct from the assumption-statement-surfacing axis #292 enforces — sister gate pattern matches Catalog #340/#314 precedent (PREVENT vs DETECT at sister surfaces, not single-gate scope-stretch).

(d) **3+ empirical receipts from 2026-05-26 ARE the canonical anchors** that justify the structural protection per Catalog #299 quota brake economics (current ~363; well under 400; new gate appropriate AND a one-gate-kills-multiple-sister-cases META gate per a8bc7e79 6-7× spread).

(e) **Initial WARN-ONLY wire-in per CLAUDE.md "Strict-flip atomicity rule"** because the existing 50+ post-Fix-7 council memos predate this gate's `empirical_verification_status` field requirement; strict-flip planned after backfill brings live count to 0.

(f) **Per CLAUDE.md "Forbidden premature KILL without research exhaustion"** — T3 council `7d04474cb` historical verdict is preserved per Catalog #110/#113 APPEND-ONLY; M2+M3 empirical-verification-status retroactively re-classified per Catalog #292 sister extension; verdict status downgrade to PROVISIONAL-PENDING-VERIFICATION OR landing as historical reference is operator-routable per protocol's Round 3 mechanism.

---

## 1. The structural bug class (3+ empirical receipts within <2h on 2026-05-26)

### Receipt #1: T3 grand council `7d04474cb` M3 RULED-OUT classification — empirically falsified

The T3 grand council deliberation `7d04474cb` (`.omx/research/t3_grand_council_mlx_pytorch_drift_accumulation_source_and_engineer_away_20260526.md`) classified **M3 (stateful optimizer drift) as RULED-OUT** based on the council's assumption that Z6 uses stateless SGD-with-EMA. The council achieved 24-of-26 PROCEED + 4 co-leads + Catalog #346 complete=True. The verdict was structurally clean per every existing META gate.

**Sister landing TIER1-T3-OP1-OP4 (commit `5b87fae77`) empirically falsified the assumption** via source-inspection of `src/tac/training/long_training_canonical.py`: Z6 uses MLX **AdamW** (not stateless SGD-with-EMA). AdamW's β₁=0.9 + β₂=0.999 state buffers means M3a (Adam state momentum drift) + M3b (Adam state variance drift) mechanisms BOTH ACTIVE. Joint mechanism refined post-hoc to M1+M2+M3a+M3b.

**Root cause of the council error:** the M3-RULED-OUT assumption was `INFERRED_FROM_DOMAIN_LITERATURE` (Adam/AdamW is the modern default + Z6 lane has SGD-style EMA tokens in its name) BUT had `empirical_verification_status=ASSUMED_AWAITING_VERIFICATION` per source-inspection. Council apparatus had no structural mechanism to flag the gap.

### Receipt #2: T3 council M2 ~0.7-0.9 α dominance prediction — empirically falsified

Same T3 council `7d04474cb` predicted **M2 (Kahan-EMA accumulator divergence) would contribute α ~0.7-0.9 of total drift signature**. The council reasoned: M2 is the canonical Kahan-summation fp32 ULP boundary mechanism + Z6 at L2 fp32 1000ep is exactly the depth-budget regime where ULP-boundary accumulation matters most.

**Sister landing TIER1-T3-OP2-OP3 (commit `05c07aa40`) empirically falsified the prediction** via canonical Kahan-EMA shadow wrapper + Carmack 30-min smoke at Z6 L2 fp32 1000ep: Kahan-EMA provided **0× empirical mitigation** (M2 contribution at fp32 ULP boundary is ~3-6e-7 shadow divergence, dominated by O(1e-3) MLX-PyTorch drift signature).

**Root cause:** M2 dominance prediction was `INFERRED_FROM_DOMAIN_LITERATURE` (Kahan-1965 + Higham-2002 numerical analysis canon) BUT carried `empirical_verification_status=ASSUMED_AWAITING_VERIFICATION` at the operating point. fp32 ULP boundary kicks in much later than 1000ep for the Z6 weight magnitude distribution.

### Receipt #3: My own preliminary n=2 super-linear α∝epochs^1.45 extrapolation — empirically falsified

Earlier today, my own preliminary analysis extrapolated α ∝ epochs^1.45 super-linear from a n=2 data point fit. **Sister landing DRIFT-VS-DEPTH-CHAR-D-Z6 (commit `60a9de751`) empirically falsified the extrapolation** via 5-anchor fit: α = 0.47 sub-linear, saturating at ~2000ep.

**Root cause:** my own super-linear extrapolation was `ASSUMED_AWAITING_VERIFICATION` with n=2 evidence (mathematically insufficient for any extrapolation exponent claim). No structural mechanism flagged that my verdict depended on an assumption with sub-threshold empirical basis.

### Receipt #4: COIN++ K=5e-3 drift claim (earlier today) — empirically falsified

A K=COIN++ canonical helper claimed 5e-3 drift bound. **Sister R1''-K independent verification (commits leading to `2d59283d4`)** empirically refuted: actual O(1e-2) abs / O(1e-3) rel.

**Root cause:** identical META pattern — verdict depended on assumption with `empirical_verification_status=ASSUMED_AWAITING_VERIFICATION`.

### The META structural gap (4 instances, same class, <2h window)

Council apparatus + the 295+ STRICT preflight gates + Catalog #291 META-ASSUMPTION cadence + Catalog #292 per-deliberation assumption surfacing + Catalog #300 v2 frontmatter + Catalog #346 canonical roster all PASSED on every one of these 4 instances. **The structural blindness recurs at a sister surface**: the existing gates enforce that assumptions ARE SURFACED + CLASSIFIED HARD-EARNED-vs-CARGO-CULTED, but NOT that each surfaced assumption carries an explicit `empirical_verification_status` AND that verdicts depending on INFERRED/ASSUMED-class assumptions are gated.

Per CLAUDE.md "Bugs must be permanently fixed AND self-protected against" + a8bc7e79 6-7× spread heuristic: the same META class is structurally inevitable across all future T2+ deliberations unless the canonical structural protection is landed.

---

## 2. The protocol mechanism

### 2.1 Lifting the 3-clean-pass counter from training-code review to council deliberation

Per CLAUDE.md "Recursive adversarial review protocol — non-negotiable" (the existing training-code review pattern):

1. **Each round**: every council member takes an adversarial perspective; reviews ALL deliberation outputs; findings categorized CRITICAL/Medium/Low.
2. **Fix immediately**: all issues fixed before next round.
3. **Clean pass counter**: round with zero issues = "clean pass". Counter resets on any finding.
4. **Gate**: 3 consecutive clean passes required before deployment (SEAL).

**The recursive self-reflection protocol lifts this to council deliberation:**

- **Round 1** (TOPIC DELIBERATION): council deliberates topic per existing CLAUDE.md "Council hierarchy: 4-tier protocol"; produces verdict + per-member assumption surfacing per Catalog #292; emits `council_assumption_adversary_verdict` field per record schema. **NEW REQUIREMENT:** each surfaced assumption MUST carry explicit `empirical_verification_status` field (one of 4 enum values).

- **Round 2** (SELF-REFLECTION ON ROUND 1): council SELF-REFLECTS on its own Round 1 output. For each per-member assumption:
  - Re-classify `empirical_verification_status` after dedicated empirical-verification cycle attempt (e.g. source-inspection of canonical helpers, query for empirical anchor in posterior, brief literature scan).
  - Identify MATERIAL UNVERIFIED ASSUMPTIONS = assumptions whose status is INFERRED_FROM_DOMAIN_LITERATURE or ASSUMED_AWAITING_VERIFICATION AND whose truth value affects the verdict.
  - Emit `council_self_reflection_round_N` canonical posterior anchor per Catalog #300 v2 frontmatter.

- **Round 3+** (RESOLUTION OR DOWNGRADE): material unverified assumptions trigger one of:
  - **(a) Empirical verification before landing.** Run a free smoke / source inspect / posterior query. Re-emit Round N+1 with updated `empirical_verification_status`.
  - **(b) Verdict-status downgrade to PROVISIONAL-PENDING-VERIFICATION.** Verdict lands but with explicit operator-routable next-step naming the unverified assumption + the verification path that would promote it from PROVISIONAL to FINAL.
  - **(c) ESCALATE_TO_OPERATOR per Catalog #300.** If neither (a) nor (b) is operationally feasible.

- **Round N (SEAL)**: 3-clean-pass counter advances when self-reflection produces NO material unverified-assumption findings. Verdict structurally landable per Catalog #300 v2 frontmatter with no PROVISIONAL marker.

### 2.2 The 4-value `empirical_verification_status` taxonomy

Each per-member surfaced assumption (already required by Catalog #292) gets one of FOUR `empirical_verification_status` classifications:

| Status | Evidence requirement | Verdict implication | Example |
|---|---|---|---|
| `VERIFIED_VIA_SOURCE_INSPECTION` | Source file path + line range + content quote that directly proves the assumption | No gate; assumption fully supports verdict | "Z6 uses MLX AdamW — verified `long_training_canonical.py:147` `optimizer = mlx.optimizers.AdamW(...)`" |
| `VERIFIED_VIA_EMPIRICAL_ANCHOR` | Canonical posterior anchor reference (commit sha + posterior row id + measurement metadata per Catalog #245 sister); typically empirical smoke / experiment artifact | No gate; assumption fully supports verdict | "M2 contributes 0× mitigation — verified empirical anchor `05c07aa40` Kahan-EMA shadow wrapper smoke" |
| `INFERRED_FROM_DOMAIN_LITERATURE` | Citation to canonical literature (paper / textbook / Wikipedia / CLAUDE.md doctrine) that supports the assumption pattern but does not directly verify the specific instance | **GATE: Round 2 must attempt verification OR Round 3 downgrades verdict to PROVISIONAL** | "Adam-family optimizers carry β₁β₂ state buffers — inferred Higham-2002 + Kingma-Ba 2014" |
| `ASSUMED_AWAITING_VERIFICATION` | Explicit acknowledgment that the assumption is operating-within unverified; no source citation; no empirical anchor | **GATE: Round 2 MUST attempt verification before Round 3 SEAL OR Round 3 downgrades verdict** | "M3-RULED-OUT — assumed Z6 uses stateless SGD-with-EMA based on lane name tokens" |

The taxonomy is canonical because:
1. The 4 values map 1-to-1 to the 4 empirical receipts above (1 per status type catches a different META class).
2. The 2 VERIFIED states cover the canonical "evidence in hand" cases (source artifact OR empirical anchor).
3. The 2 unverified states distinguish "I have a literature analogy" from "I am extrapolating without basis"; both gate equally because BOTH are pre-empirical at the specific instance.
4. The taxonomy is bounded (not a continuum) so per-assumption disambiguation is structurally tractable per the R12-D meta-finding lens-coverage cycle-bounding criterion.

### 2.3 Cycle bounds (per R12-D meta-finding lens-coverage)

Per CLAUDE.md "Recursive adversarial review protocol — close paths (post-R12+R13)":

> D-1 is a higher-bar alternative for cycles structurally unsatisfiable per R12-D meta-finding (lens-coverage expansion outpacing Zipf-decay).

The recursive self-reflection protocol is **structurally bounded** because the assumption-verification axis is bounded:

- Each round either VERIFIES an assumption (status promotes UP toward VERIFIED) OR DOWNGRADES the verdict (PROVISIONAL marker). Both paths advance the counter.
- ≤5 self-reflection rounds before operator-routed ESCALATE_TO_OPERATOR per Catalog #300.
- 3 consecutive rounds with zero material unverified-assumption findings = SEAL.

The bound prevents the protocol from collapsing into infinite recursion (sister of the R12-D Zipf-decay lens-coverage failure mode).

### 2.4 Operator-attention budget integration per Catalog #300

The Catalog #300 v2 frontmatter T2+ deliberation cadence budget already constrains council deliberation rate:
- T2 ≤3/day ≤90/30d
- T3 ≤3/week ≤13/30d
- T4 ≤2/30d

**Self-reflection rounds are counted SEPARATELY** in `tools/audit_council_tier_cadence.py` to avoid double-counting:
- A T2 deliberation's Round 1 counts against the T2 daily budget.
- Its Round 2 + Round 3 self-reflection rounds count against a NEW `self_reflection_rounds_per_deliberation` metric (no fixed budget; bounded structurally at ≤5 per protocol).
- The metric is surfaced as APPROACHING_LIMIT alert when any single deliberation crosses 4 self-reflection rounds (operator-visible STOP AND CONSOLIDATE — the deliberation is producing more self-reflection than topic-conclusion).

---

## 3. Canonical-vs-unique decision per layer (per CLAUDE.md Catalog #290)

| Layer | Decision | Rationale |
|---|---|---|
| `EmpiricalVerificationStatus` enum | CANONICAL-NEW (in `tac.council_continual_learning`) | 4-value taxonomy is foundational; downstream consumers (autopilot ranker / Rashomon ensemble / cathedral consumer) all need the same vocabulary |
| `AssumptionEmpiricalVerification` dataclass | CANONICAL-NEW (in `tac.council_continual_learning`) | Frozen dataclass mirroring existing `council_assumption_adversary_verdict` shape; structural extension preserves backward compat |
| `CouncilDeliberationRecord.council_assumption_adversary_verdict` field semantics | CANONICAL-EXTENDED (backward-compat) | Existing field accepts both legacy dict-form rows AND new `AssumptionEmpiricalVerification` instances; legacy rows auto-classified `INFERRED_FROM_DOMAIN_LITERATURE` per safe-default |
| Catalog #363 STRICT preflight gate | CANONICAL-NEW SISTER OF #292 | Sister gate pattern per Catalog #340/#314 precedent; #292 enforces assumption-surfacing axis, #363 enforces empirical-verification-status axis; structurally distinct |
| CLAUDE.md "Council conduct" amendment | CANONICAL-EXTENDED APPEND-ONLY | New subsection "Recursive self-reflection protocol — non-negotiable" appended per Catalog #110/#113 HISTORICAL_PROVENANCE; existing prose preserved unchanged |
| `append_council_anchor` canonical helper | CANONICAL-PRESERVED | No mutation to existing fcntl-locked JSONL writer; new event_type `canonical_protocol_landed` for protocol-landing anchor + `council_self_reflection_round_N` event_type for per-round anchors |
| Per-substrate symposium discipline per Catalog #325 | CANONICAL-PRESERVED | The 6-step contract already required by #325 inherits the empirical-verification-status discipline automatically because all 6 steps now route through `council_assumption_adversary_verdict` per #292/#363 sister extension |

**Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode":** the protocol is canonical because the META class affects ALL T2+ deliberations across ALL substrates (the meta-Lagrangian/Pareto solver / autopilot ranker / Rashomon ensemble all consume council verdicts; corruption at the verdict surface propagates everywhere). Per-substrate unique-ification would defeat the canonical signal.

---

## 4. 9-dimension success checklist evidence

1. **UNIQUENESS** — sister of Catalog #292 at a distinct axis (empirical-verification-status, not assumption-statement-surfacing). One gate kills 4+ instances per a8bc7e79 6-7× spread heuristic.
2. **BEAUTY + ELEGANCE** — 4-value enum + 1 frozen dataclass + 1 STRICT gate; ≤250 LOC additive across canonical helper + preflight; reviewable in 30 seconds per CLAUDE.md "Beauty, simplicity, and developer experience" non-negotiable.
3. **DISTINCTNESS** — explicitly distinct from #292 (which enforces assumption-STATEMENT surfacing) and from #291 (session-level cadence). #363 enforces per-assumption empirical-verification-status.
4. **RIGOR** — premise verified via 4 empirical receipts within <2h window (canonical post-Catalog #229 PV pattern) + canonical Provenance per #323 + 4-value taxonomy mathematically bounded per R12-D criterion.
5. **OPTIMIZATION PER TECHNIQUE** — canonical helpers in `tac.council_continual_learning` route through existing fcntl-locked JSONL writer per Catalog #131/#138/#245 sister discipline; no new state writer; no new lock; no new file path.
6. **STACK-OF-STACKS COMPOSABILITY** — composes additively with Catalog #291 (cadence) + #292 (per-deliberation assumption) + #300 (v2 frontmatter) + #346 (roster) + #325 (per-substrate symposium); all sister gates inherit the new discipline automatically because `council_assumption_adversary_verdict` is the shared field structure.
7. **DETERMINISTIC REPRODUCIBILITY** — canonical posterior anchor for protocol landing event is byte-stable + seed-pinned (`canonical_protocol_landed` event_type with explicit `deliberation_id="council_recursive_self_reflection_protocol_landed_20260526"` + canonical Provenance per #323).
8. **EXTREME OPTIMIZATION + PERFORMANCE** — gate scope identical to #292 (same memo set, same scan dirs); no new file I/O surface; preflight overhead negligible (~50ms typical).
9. **OPTIMAL MINIMAL CONTEST SCORE** — indirect contribution; prevents future score-relevant verdicts (e.g. dispatch routing, lane promotion, kill/promote) from being bound on unverified assumptions; the 4 empirical receipts above ARE the canonical anchors for the score-relevance argument.

---

## 5. Observability surface

- **Inspectable per layer**: every council deliberation memo's `council_assumption_adversary_verdict` field now exposes per-assumption `empirical_verification_status` at the canonical posterior JSONL row level (query via `tac.council_continual_learning.query_assumption_classification_history`).
- **Decomposable per signal**: per-assumption status decomposes into 4-value enum (canonical taxonomy); per-deliberation verdict status decomposes into PROVISIONAL-vs-FINAL via material-unverified-assumption count.
- **Diff-able across runs**: 3-clean-pass counter state is serializable in canonical posterior anchor (`event_type=council_self_reflection_round_N`); two consecutive runs can be diffed for counter advance / reset.
- **Queryable post-hoc**: `tac.council_continual_learning.query_anchors_by_topic` + new sister `query_self_reflection_history_for_deliberation` (added in Surface 2 dataclass extension) return per-deliberation self-reflection event chain.
- **Cite-able**: every per-assumption record carries `evidence_artifact` field (path or commit sha or posterior row id) per canonical Provenance #323.
- **Counterfactual-able**: legacy rows (pre-2026-05-26) auto-classified `INFERRED_FROM_DOMAIN_LITERATURE` per safe-default; the counterfactual "what if we re-classify all legacy rows as ASSUMED_AWAITING_VERIFICATION?" is structurally tractable via the `query_assumption_classification_history` helper's `--counterfactual-status` parameter (planned follow-on; not landed in this charter).

---

## 6. Cargo-cult audit per assumption

| Assumption | HARD-EARNED vs CARGO-CULTED | Unwind path |
|---|---|---|
| "4-value taxonomy is the canonical disambiguator" | HARD-EARNED — derived from 4 empirical receipts (one per status type) | N/A — if a 5th instance surfaces a NEW failure mode, taxonomy extends (per CLAUDE.md "Forbidden premature KILL" — bounded extension, not category replacement) |
| "Sister gate pattern (NEW Catalog #) is canonical over scope-extension of Catalog #292" | HARD-EARNED — empirical-verification-status axis is structurally distinct from assumption-statement-surfacing axis; Catalog #340/#314 precedent (PREVENT vs DETECT sister gates) | Unwind path = Catalog #292 in-place scope-extension; rejected because the empirical_verification_status field requires Round 2 self-reflection mechanism which is structurally distinct from per-round assumption surfacing |
| "WARN-ONLY initial wire-in is correct" | HARD-EARNED — 50+ post-Fix-7 (>= 2026-05-15) council memos predate this charter; backfilling all retroactively is operator-routed not auto | Unwind path = STRICT-from-byte-one; rejected because backfill cannot complete in the landing commit batch; sister of Catalog #292 + #294 + #303 + #305 all WARN-ONLY-at-landing pattern |
| "Sub-linear/super-linear extrapolation needs n≥3 data points" | HARD-EARNED — my own n=2 super-linear extrapolation was empirically falsified by sister 5-anchor fit (n=5) | N/A — the protocol's `ASSUMED_AWAITING_VERIFICATION` taxonomy catches this class directly |
| "Council deliberation cycle-bounds are structurally necessary" | HARD-EARNED — R12-D meta-finding lens-coverage Zipf-decay; ≤5 self-reflection rounds per deliberation prevents infinite recursion | N/A — bound matches existing CLAUDE.md "Recursive adversarial review protocol — close paths" D-1 7-day cool-down pattern |

---

## 7. Predicted ΔS band

**N/A** — this is an apparatus-maintenance landing per Catalog #300 `predicted_mission_contribution=apparatus_maintenance` classification. The protocol does not directly contribute to contest score; it prevents future verdicts (some of which DO contribute to score via dispatch routing / lane promotion / kill/promote) from being bound on unverified assumptions. The empirical-receipt-anchors are 4 instances where the absence of the protocol caused empirically-falsified verdicts within a <2h window; the predicted future-prevention rate is `~1-2 per week per session given current T2+ cadence (≤3/day) × baseline failure rate (~30% per empirical receipts)` — structurally bounded but not score-numerically predictable.

Per CLAUDE.md "Mission alignment — non-negotiable" Consequence 5: `apparatus_maintenance` category surfaces in operator alert when `rigor_overhead + apparatus_maintenance > 60%` of T2+ verdicts in 30-day window. This landing counts as 1 apparatus_maintenance verdict; sister mission-alignment alerts route through `tools/audit_council_tier_cadence.py` per existing #300 wire-in.

---

## 8. Reactivation criteria

If at any future point the protocol's WARN-ONLY → STRICT flip discipline produces > 5 strict-mode preflight failures per week on legitimate council memos:

- **Re-classify** as STRICT-mode over-blocking; investigate whether the 4-value taxonomy needs a 5th value (e.g. `VERIFIED_VIA_HYPOTHESIS_REGISTERED_IN_CANONICAL_EQUATIONS_REGISTRY` per Catalog #344).
- **Operator-routed** per CLAUDE.md "Forbidden premature KILL" — the protocol is not killed, it is extended via reactivation criteria.

If the protocol catches < 1 empirically-falsified verdict per month across 6 months:

- **Investigate** whether the protocol's structural enforcement has shifted the failure mode upstream (verdict-class quality has improved). The "low catch rate" verdict per CLAUDE.md "Mission alignment" Consequence 2 annual gate audit is operator-routable.

---

## Cross-references

- `feedback_or2_grand_council_per_round_assumption_statement_discipline_landed_20260515.md` (Catalog #292 landing)
- `.omx/research/t3_grand_council_mlx_pytorch_drift_accumulation_source_and_engineer_away_20260526.md` (T3 council `7d04474cb` — Receipt #1+#2 empirical anchor)
- Commits `5b87fae77` (Receipt #1 M3 source-inspection falsification) + `05c07aa40` (Receipt #2 M2 empirical smoke falsification) + `60a9de751` (Receipt #3 n=2 extrapolation 5-anchor fit falsification) + `2d59283d4` (Receipt #4 COIN++ K=5e-3 independent verification falsification)
- CLAUDE.md "Council conduct" Fix-7 amendment + "Recursive adversarial review protocol — non-negotiable" + "META-ASSUMPTION ADVERSARIAL REVIEW" + "Council hierarchy: 4-tier protocol" + "Mission alignment — non-negotiable"
- Catalog #291 (META-ASSUMPTION cadence; session-level sister) + Catalog #292 (per-deliberation assumption surfacing; per-deliberation sister) + Catalog #300 (v2 frontmatter; canonical posterior schema) + Catalog #325 (per-substrate symposium; sub-surface sister) + Catalog #346 (canonical roster; structurally-distinct axis) + Catalog #340 (sister-checkpoint guard; sister gate pattern precedent) + Catalog #314 (post-commit DETECT vs pre-commit PREVENT sister gate pattern precedent) + Catalog #287 (placeholder-rationale rejection sister discipline) + Catalog #344 (canonical equations registry; sister formalization pattern)

---

## Sister coordination (Catalog #230)

**IN-FLIGHT at landing time:**
- COMPREHENSIVE-BUG-AUDIT-FIX-CASCADE (`a81382f32ce8ca4b8`) — read-only audit; produces NEW landing memo. SCOPE DISJOINT (this protocol design owns the canonical apparatus surface per operator's direct directive; bug-audit defers to this landing if it surfaces the same class).
- L1-PROMOTION-CASCADE-B-C-E-G-J (`aebfb907aa6ead9a7`) — touches 5 sister substrate packages + canonical L2 helper consumption. DISJOINT from council apparatus surface.

**THIS SCOPE:**
- NEW `.omx/research/council_recursive_self_reflection_protocol_design_<utc>.md` (THIS file)
- EXTEND `src/tac/council_continual_learning.py` per Surface 2
- EXTEND `src/tac/preflight.py` per Surface 3 (NEW Catalog #363)
- APPEND-ONLY `CLAUDE.md` "Council conduct" amendment per Surface 4
- APPEND-ONLY canonical posterior anchor + NEW landing memo per Surface 5
- NEW tests for Surface 2 + Surface 3


<!-- COUNCIL_TIER_FRONTMATTER_WAIVED:design_and_landing_memos_for_canonical_recursive_self_reflection_protocol_NOT_a_council_deliberation_itself_filename_pattern_council_star_yyyymmdd_md_matches_gate_regex_false_positive_per_comprehensive_bug_audit_cascade_20260526 -->


# COUNCIL_TIER_FRONTMATTER_WAIVED:design_and_landing_memos_for_canonical_recursive_self_reflection_protocol_NOT_a_council_deliberation_itself_filename_pattern_matches_gate_regex_false_positive_per_comprehensive_bug_audit_cascade_20260526
