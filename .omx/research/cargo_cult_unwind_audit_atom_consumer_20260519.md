# Cargo-cult unwind audit: `atom_consumer` (Wave 2C)

- **Date**: 2026-05-19
- **Subagent**: WAVE-2C-CARGO-CULT-UNWIND-AUDIT (`wave_2c_cargo_cult_audit_20260519`)
- **Target**: `src/tac/cathedral_consumers/atom_consumer/__init__.py` (64 LOC; commit `179dc2501`)
- **Canonical reference**: `src/tac/cathedral_consumers/_example_consumer/__init__.py` (64 LOC)
- **Auditor lens**: META-ASSUMPTION ADVERSARIAL REVIEW per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" + the hard-earned-vs-cargo-culted addendum (`feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515.md`).

## Source-text differential vs `_example_consumer`

`atom_consumer` differs from the template by only ~10 lines: module docstring naming `tac.atom` + `CONSUMER_NAME` literal + rationale string citing "cargo-cult / premise-verification / probe-outcome / council-deliberation / meta-Lagrangian atoms". Body of `update_from_anchor` and `consume_candidate` is byte-identical-template (both discard input via `_ = anchor` / `_ = candidate` and return the canonical zero-adjustment dict).

The wrapper IS the template — there is no per-consumer engineering beyond the docstring + literal.

## Canonical-vs-unique decision per layer

Per Catalog #290 falling-rule list. Layers identified by inspecting the source:

| Layer | Canonical decision | Verdict | Evidence |
|---|---|---|---|
| L1: `CONSUMER_NAME` literal | adopt `"atom_consumer"` | **ADOPT_CANONICAL_BECAUSE_SERVES** | Required for auto-discovery registration per `tac.cathedral.consumer_contract.ConsumerRegistration`. No score-suppressing alternative. |
| L2: `CONSUMER_HOOK_NUMBERS` | adopt `(CATHEDRAL_AUTOPILOT_DISPATCH, CONTINUAL_LEARNING_POSTERIOR)` | **CARGO-CULTED-INHERITED-DEFAULT** | Template default; the rationale comment claims hook #5 but `update_from_anchor` is NO-OP, contradicting the declared hook #5 semantic. See cargo-cult #2 below. |
| L3: `update_from_anchor(anchor)` body | adopt NO-OP template | **CARGO-CULTED-INHERITED-DEFAULT** | Docstring says "atoms are already persisted via `tac.atom` helpers fcntl-locked JSONL store; no additional posterior update is required here". This is HARD-EARNED-FIRST-PRINCIPLES (atom ledger IS already the canonical posterior surface; double-persisting would violate Catalog #110 HISTORICAL_PROVENANCE APPEND-ONLY). But declaring hook #5 then no-op-ing it is structurally inconsistent. |
| L4: `consume_candidate(candidate)` body — discards candidate via `_ = candidate` | adopt zero-adjustment template | **CARGO-CULTED-PATH-OF-LEAST-RESISTANCE** | The template was meant as a starting scaffold per `_example_consumer` docstring (*"A production consumer would compute a bounded adjustment from the candidate's payload"*). atom_consumer ships the scaffold as production. See cargo-cult #1 below. |
| L5: `axis_tag = "[predicted]"` | adopt canonical | **ADOPT_CANONICAL_BECAUSE_SERVES** | HARD-EARNED-EMPIRICALLY-VERIFIED per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #287 + #323. Atom emissions ARE predicted/diagnostic by construction; promoting to `[contest-CUDA]` requires paired Linux x86_64 anchor. Forking would re-introduce phantom-score class. |
| L6: `promotable = False` | adopt canonical | **ADOPT_CANONICAL_BECAUSE_SERVES** | HARD-EARNED-FIRST-PRINCIPLES per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA" non-negotiable. Atom emissions have no archive sha256 binding to a contest-axis score. |
| L7: `confidence = 0.0` | adopt canonical | **CARGO-CULTED-INHERITED-DEFAULT** | Template literal. Atom emissions have rich structured payload (KIND / archive_sha / timestamp / probe-outcome); a confidence > 0 reflecting "this annotation cites N atoms attached to this archive" would be more informative. See cargo-cult #3 below. |
| L8: docstring rationale citing 5 atom KINDs | adopt canonical | **ADOPT_CANONICAL_BECAUSE_SERVES** | HARD-EARNED — accurate enumeration of tac.atom emission KINDs serves operator audit. |

## Cargo-cult audit per assumption

Per Catalog #303. Five inherited assumptions, each classified:

### Assumption #1: "An observability-only consumer that discards the candidate is the canonical Wave 2C pattern"

- **Classification**: CARGO-CULTED-PATH-OF-LEAST-RESISTANCE
- **Evidence**: The `_example_consumer` docstring explicitly labels itself a "reference no-op consumer" and says "A production consumer would compute a bounded adjustment from the candidate's payload". Wave 2C cloned the reference, not a production consumer.
- **Empirical anchor**: 22-of-24 cathedral consumers (all except `mps_viable_prescreen_consumer` from earlier today + `procedural_codebook_generator_consumer`) discard the candidate via `_ = candidate`. Confirmed via `grep -L "_ = candidate" src/tac/cathedral_consumers/*/__init__.py`.
- **Why the apparatus may have suppressed substrate-optimal engineering**: per CLAUDE.md "Subagent coherence-by-default" the 6-hook wire-in non-negotiable demanded SOMETHING be wired into cathedral autopilot for every new `tac.*` namespace. The path-of-least-resistance discharge was a wrapper that satisfies Catalog #335 structural validation but contributes zero ranking signal. Empirically the autopilot ranker is unchanged (line 6236: *"the invocation does NOT mutate predicted_score_delta on the candidate rows"*).
- **Unwind hypothesis**: replace `_ = candidate` with a real query against `tac.atom.ledger.query_atoms_by_archive_sha(candidate["archive_sha256"])`. Return `confidence` proportional to atom count attached to this archive; surface KIND breakdown in `rationale` (e.g. "5 atoms attached: 2 cargo_cult / 1 premise_verification / 2 probe_outcome — [predicted]"). The result remains `predicted_delta_adjustment = 0.0` (the gate is observability-only) but it gives operators a queryable annotation surface vs an identical template literal.
- **Unwind cost**: ~$0 (no GPU). ~15 LOC + 2 tests.
- **Predicted signal contribution**: operator-velocity acceleration during dispatch ranking review (post-Modal harvest, the operator currently has to grep tac.atom.ledger manually); estimated **-0.002 to -0.005 ΔS/month** indirect via faster dispatch decisions per Catalog #287 evidence-tag discipline.
- **Reactivation criterion (per Catalog #308 N≥3 alternatives)**: alternative reducers if the count-based confidence proves noisy = (a) per-KIND confidence weighting / (b) recency-weighted (latest N atoms only) / (c) cite-chain depth via `related_atom_ids`.

### Assumption #2: "Declaring `CONTINUAL_LEARNING_POSTERIOR` hook #5 but no-op-ing `update_from_anchor` is acceptable"

- **Classification**: CARGO-CULTED-INHERITED-DEFAULT (structurally inconsistent with hook declaration)
- **Evidence**: The Protocol contract docstring (`consumer_contract.py:130`) says hook #5 fires *"when a new empirical anchor (contest-CUDA / contest-CPU / diagnostic) is appended to the canonical posterior store"*. atom_consumer claims hook #5 but ignores anchors. If atom emissions are truly already persisted independently, hook #5 should not be declared.
- **Why apparatus suppressed**: the template `_example_consumer` declares both hooks #4 and #5 and the auto-discovery loop validates structural compliance (Catalog #335 STRICT preflight) WITHOUT verifying that declared hooks have non-trivial implementations. Catalog #335's structural-only validation is HARD-EARNED-CORRECT (per the Protocol design that runtime correctness is the consumer's responsibility) but enables hook-declaration cargo-culting.
- **Unwind hypothesis (option A)**: drop `HookNumber.CONTINUAL_LEARNING_POSTERIOR` from `CONSUMER_HOOK_NUMBERS` to match the actual NO-OP implementation. This is HONEST and removes a documentation lie.
- **Unwind hypothesis (option B)**: implement hook #5 as `tac.atom.ledger.append_atom(KIND="continual_learning_anchor_observed", anchor_ref=anchor)` so every contest-CUDA/CPU/diagnostic anchor triggers an audit atom. This binds the declared hook to a real surface.
- **Unwind cost**: option A ~$0, 2 LOC. Option B ~$0, ~10 LOC + 1 test.
- **Predicted signal contribution**: option B closes the audit-trail loop (every empirical anchor produces a queryable atom citation). Indirect operator-velocity contribution ~**-0.001 to -0.003 ΔS/month**.
- **Reactivation criterion**: option B alternative = per-anchor KIND emission (cargo_cult_anchor vs premise_verification_anchor vs probe_outcome_anchor).

### Assumption #3: "`confidence = 0.0` is appropriate for observability-only consumers"

- **Classification**: CARGO-CULTED-INHERITED-DEFAULT
- **Evidence**: All 22 template-clone consumers ship `confidence = 0.0`. The Protocol docstring (`consumer_contract.py:159`) defines `confidence: float in [0, 1]` but the template defaults to 0.
- **Why apparatus suppressed**: a non-zero confidence on an observability-only contribution might be misread as authoritative; the template defended against misuse by zeroing it. This is HARD-EARNED-DEFENSIVE in the abstract but cargo-culted at the specific atom_consumer surface where confidence has a natural empirical meaning (atom count / freshness).
- **Unwind hypothesis**: `confidence = min(1.0, atom_count / 10.0)` per the unwind in cargo-cult #1. Confidence > 0 signals "this annotation has actual evidence" vs "this consumer always returns the same template".
- **Unwind cost**: ~$0, 3 LOC (depends on cargo-cult #1 landing).
- **Predicted signal contribution**: subsidiary to cargo-cult #1.

### Assumption #4: "Rationale text is the right channel for atom KIND breakdown"

- **Classification**: HARD-EARNED-FIRST-PRINCIPLES
- **Evidence**: The Protocol contract has `rationale: str` with a 512-char bound (line 6194 `[:512]`). Operators read the rationale; structured fields below it are queryable. Per CLAUDE.md "Max observability — non-negotiable" the canonical disclosure is a human-readable + machine-queryable annotation.
- **Verdict**: ADOPT. No unwind needed.

### Assumption #5: "tac.atom emissions never warrant promotion to contest-grade"

- **Classification**: HARD-EARNED-FIRST-PRINCIPLES
- **Evidence**: tac.atom emissions are typed-atom annotations attached to design-time + audit-time events. They have no archive_sha256 + paired-axis empirical evidence binding required for `[contest-CUDA]` / `[contest-CPU]` per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #127.
- **Verdict**: ADOPT. Forking would re-introduce phantom-score class per Catalog #287 / #323.

## Observability surface

Per Catalog #305. Six facets:

1. **Inspectable per layer**: rationale text + `confidence` field + hook declarations exposed via the canonical `ConsumerRegistration` dataclass. Module docstring + source readable in 30 sec.
2. **Decomposable per signal**: today, NO — the rationale string is monolithic. Post cargo-cult #1 unwind, per-KIND counts would be exposed.
3. **Diff-able across runs**: today, NO — every invocation returns the identical template. Post unwind, per-archive atom counts would diff naturally.
4. **Queryable post-hoc**: today, partial — `consumer_invocations` is persisted in autopilot loop output (line 6761 + 6856) so the rationale survives. Atom citations would survive a re-grep against `tac.atom.ledger`.
5. **Cite-able**: today, NO — no cite-chain field. Post unwind, atom IDs would be embedded in the rationale.
6. **Counterfactual-able**: today, NO — discarding the candidate means there is no counterfactual (the contribution is the same regardless of input). Post unwind, mutating an atom in the ledger would change the rationale on the next invocation.

**Conclusion**: 1-of-6 facets satisfied today. Cargo-cult #1 unwind would lift to 4-of-6.

## Unwind priority queue

Ranked by `|predicted ΔS-signal-contribution| / cost`:

| Rank | Unwind | Cargo-cult # | Cost | Predicted ΔS | EV ratio |
|---|---|---|---|---|---|
| 1 | Query `tac.atom.ledger` by archive_sha + return per-KIND counts in rationale + non-zero confidence | #1 + #3 (composite) | ~$0, 15 LOC, 2 tests | -0.002 to -0.005 ΔS/month indirect | HIGH |
| 2 | Drop `CONTINUAL_LEARNING_POSTERIOR` from `CONSUMER_HOOK_NUMBERS` OR wire hook #5 to append an atom per anchor | #2 | ~$0, 2-10 LOC | -0.001 to -0.003 ΔS/month indirect | MEDIUM |
| 3 | (subsidiary to #1) Add per-KIND confidence weighting | #3 | ~$0, 3 LOC | subsidiary | LOW |

**Top-1 unwind**: cargo-cult #1+#3 composite. Estimated 30-45 min implementation per the Wave 2C consumer template.

## Cross-references

- Sister of Catalog #265 (symposium impls canonical contract — same template-clone class at the symposium surface; we should audit those too)
- Sister of Catalog #335 (cathedral auto-ingest paradigm shift — STRUCTURAL validation only; runtime correctness is consumer's responsibility, so this audit is the runtime-correctness layer)
- Sister of Catalog #287 (`[predicted]` discipline — HARD-EARNED at the axis_tag layer, CARGO-CULTED at the "treat everything as template" layer)
- Anchor memo: `feedback_pr95_lesson_now_at_meta_level_unique_and_complete_per_method_default_20260515.md`
- Hard-earned-vs-cargo-culted addendum: `feedback_assumptions_classification_hard_earned_vs_cargo_culted_critical_addendum_20260515.md`
- Empirical evidence that ranker is observability-only: `tools/cathedral_autopilot_autonomous_loop.py:6236` *"the invocation does NOT mutate predicted_score_delta on the candidate rows"*

## Verdict

`atom_consumer` is a Catalog #335 contract-compliant template-clone. Two CARGO-CULTED assumptions identified (#1 candidate discard, #2 hook #5 declared without implementation). Both are unwindable at ~$0 cost. Per CLAUDE.md "Forbidden premature KILL without research exhaustion" the lane is DEFER + REQUEST-REINVESTIGATION-OF-ALTERNATIVES per Catalog #308 (3 reactivation criteria documented above).


<!-- # FORMALIZATION_PENDING:pre_framework_memo_dated_2026-05-19_predates_canonical_equations_birthday_registry_population_in_progress_appended_by_strict_flip_enablers_per_operator_blanket_approval_per_claude_md_forbidden_premature_kill_without_research_exhaustion_this_is_DEFER_pending_canonical_equation_backfill_NOT_kill -->
