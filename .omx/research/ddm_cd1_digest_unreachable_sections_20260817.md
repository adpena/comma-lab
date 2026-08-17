# ddm_cd1 — the costate digest's recall organs never ran on the live path

**Date:** 2026-08-17 · **Arm:** ddm_cd1 · **Axis:** apparatus (pointer UNMOVED) ·
**Evidence:** MEASURED at source + executed controls

---

## ANSWER FIRST

The defect is **real, live, and worse than "a dropped section"** — but its shape is not
what the filing said, and the correction matters for anyone building a detector.

`tools/costate_digest.py::build_digest` opened `if ddm_live:` at `:2200` and closed it
with `return lines, data` at `:2251`, orphaning `:2253`–`:2330`. `ddm_live` is **True in
the live state** (MEASURED). But the branch *deliberately re-provided a total 16-key
schema*, so **nothing looked absent**. The harm was four keys carrying **plausible wrong
values**:

| key | shipped on the live path | what the section actually returns |
|---|---|---|
| `verdict_scope` | a *provenance* dict (`evidence_axis`, `score_claim`, …) | `{'count_14d': 7, 'count_all': 14, 'recent': [...]}` |
| `corpus_recall` | `[]` — violating its own `dict \| None` signature | `dict` with 3 corpus hits |
| `active_convening` | `None` | `dict` (live t5_crucible3 convening) |
| `graph_memory` | `None` | `dict` (37,519 nodes / 156,277 edges) |

So the standing belief that the recall advisory reaches a reader is **falsified**: the
7 advisories in 14 days reached a sink whose only reader was dead code. Confirmed absent
in the default invocation, `--full`, and `--session-start` (all route through
`build_digest`, `main():2399`).

**Fixed** (if/else + shared tail; +7 digest lines; 3.498 s → 4.663 s, still under the
documented <5 s budget). **Guarded** by a STRICT gate at live-count 0 plus three
mutation-verified regression tests.

---

## 1. VERIFICATION — re-derived, and two corrections to the filing

Both corrections are the same genus: *the detector's shape must match how the code
expresses the thing* (`ddm_qd1` §2).

**Correction 1 — it is NOT a bare `return`.** `:2251` is `return lines, data`, a
value-returning return. A guard keyed on `node.value is None` would have reported a
clean scan **over its own anchor incident**. My gate matches returns by POSITION only.

**Correction 2 — "18 sections dropped" overstates the data-key harm.** Re-derived by
AST: **31 top-level statements** sit after the if-block, assigning **16 distinct `data`
keys**. Of those 16, six are set to explicit `DOMINATED_STALE` markers by the loop at
`:2209`–`:2217` (deliberate, documented) and six more are legitimately re-provided from
the live report. **Four were wrong.** The digest *text* lost **7 lines** (31 → 38 after
the fix). The honest statement is "4 keys poisoned + 7 lines lost", not "18 sections
dropped".

**The accident is provable.** `git blame`: commit `7fac2e7475` (2026-07-23) added the
`return` at `:2251` **and, in the same commit**, an `if ddm_live:` at `:2304` *inside the
region it had just orphaned*. You do not write a branch into code you meant to make
unreachable. This is the fingerprint the guard keys on.

`verdict_scope: INSTANCE` — one function, one file, verified at source and by execution.

---

## 2. THE FIX

`if ddm_live: … return` → explicit `if/else` with a **shared tail**. Branch-specific
closing text travels in `boundary_line`/`deeper_line` locals.

- The four DDM-independent organs (advisory sink, convening ledger, corpus, graph
  memory) now run on **both** paths.
- The provenance dict moved to its own key, **`ddm_authority_provenance`** — it was never
  verdict-scope data. No external consumer reads it (checked; see §5).
- The dead `if ddm_live:` at `:2304` collapsed to its live branch.
- `resume_spine`'s *line* is deliberately **not** restored on the live path: its data
  comes from the live report, and the file's own contract says partial DDM state is never
  mixed into the legacy lineage.

**A regression I introduced, caught in review:** the first splice left `boundary_line`
assigned but appended the *legacy* literals — the live path would have been relabelled
with the wrong actuation BOUNDARY. `ruff --select F` (F841) caught it. Recorded because
one's own fixes are unreviewed new code.

---

## 3. THE GUARD — and the shape I rejected first

### Rejected: "early return truncates a partially-built accumulator"

Implemented fully, then **MEASURED**: **30 sites across 30 files**. On inspection
essentially all are legitimate error-cascade guards that *record* why they bailed
(`blockers.append(...); return summary, blockers`). Critically, **neither accumulator
shape nor dict-key-set parity separates them from the anchor** — the anchor's author
maintained a total schema, so key-parity passes on the pre-fix code. A gate on this shape
would be permanently amber over benign code, and a gate readers ignore is not protection.

`verdict_scope: FORMULATION` — this formulation of the truncation detector is not
strictly gateable. That is **not** a claim that no gateable form exists; I found one.

### Landed: `check_no_dead_conditional_retest_after_early_return`

`src/tac/confound_gates.py` (extends the existing AST-gate machinery — `_func_defs`,
`_span_source`, `_waiver_present`, `_finish` — rather than building a parallel surface).

Refuses: a top-level `if <name>:` preceded by another top-level `if <name>:` that returns
unconditionally with no `else`, where `<name>` is neither rebound nor mutated between.
That is machine-provable dead code and the accidental-truncation fingerprint. Deliberate
early returns do not leave dead re-tests behind.

- **STRICT** from byte one, wired into `preflight_all` riding the confound-gate
  immune-system row (`# CLAUDE_MD_ENTRY_OK`). **No new catalog number**: the quota gate is
  strict-wired and already breached (registered max 407 vs the #400 brake, 5 violations);
  claiming #408 would add a sixth.
- **Waiver:** `# DEAD_CONDITIONAL_RETEST_OK:<rationale>`.
- **Denominator declared** (VACUITY==PASS cure): `10,782 file(s) considered, 1,531 parsed
  after the necessary-condition pre-filter, 42,527 function(s), 691 unconditional-return
  guard(s) tracked`. The pre-filter is a *provably necessary* condition (a violation needs
  two top-level bare-name `if`s), not a heuristic; both counts ship so a shrunken scope
  cannot hide. 28 s → 10.5 s.

**A real bug caught in the gate's own bring-up:** the first draft flagged
`harvest_cuda_cpu_axis_profile_registry.build_combined_payload_from_pair`, where
`if blockers: return` is followed by more `blockers.append(...)` and a second
`if blockers:` — very much alive. Clearing the decision only on *name rebinding* misses
method-call mutation. `_name_mutated_between` now treats `x.append(...)` and item stores
as invalidating.

---

## 4. THE CONTROLS — both executed, both shown

**Class guard, run against the two source versions:**

```
=== PREFIX  ===  1 violation
  tools/costate_digest.py:2304: `if ddm_live:` is DEAD — the guard at line 2200 in
  'build_digest' already returns unconditionally when ddm_live is truthy …
=== POSTFIX ===  0 violations
  OK (1 file(s) considered, 1 parsed after the necessary-condition pre-filter,
      60 function(s), 1 unconditional-return guard(s) tracked)
```

The post-fix denominator proves it *looked* — not a vacuous green.

**Registered `PositiveControl`**, planting the exact anchor shape (value-returning
`return lines, data` + a second `if ddm_live:`). It is **executed** by
`check_refusal_gates_have_live_positive_control` in a temp dir; coverage ratcheted
**13/31 → 14/32**, and the meta-gate raised no "POSITIVE CONTROL NO LONGER FIRES" entry.

**Three regression tests**, `src/tac/tests/test_costate_digest_shared_tail_reaches_live_path.py`,
**mutation-verified** by restoring the pre-fix file: all three go **red**, then green
again on restore.

> The first draft of the structural test **passed on the pre-fix source** — it asserted
> the organs sat at build_digest's *top level* rather than inside an `if`, and the
> truncated statements were top-level, merely unreachable. Position was the *named*
> object; reachability was the *measured* one. Caught only by mutation-checking the test
> instead of trusting it. The test now checks for a preceding unconditional-return guard.

---

## 5. SWEEP — with its denominator

| detector | scope | result |
|---|---|---|
| dead conditional re-test | 10,782 files considered / 1,531 parsed / 42,527 functions | **1** live: `tools/gc_experiments_results.py:404` |
| early-return truncates accumulator (rejected shape) | 10,780 files / ~42k functions | **30** across 30 files, judged benign |
| same, unrefined, over the full tree incl. `experiments/results/**` | 51,361 files / 901,714 functions | 1,450 (upper bound, mostly generated code) |

`experiments/results/**` is excluded from the shipped gate as generated-artifact
territory (~40k files, +60 s); the exclusion is stated in code and the denominator ships.

The one dead-re-test hit is **genuine**: `_classify_dir`'s second `if is_tracked:` is a
"defense-in-depth" fail-safe the author believes is live and which provably is not. Waived
truthfully rather than silently — the waiver says it is unreachable *today* and retained
against future control-flow changes.

**I did not find further instances of the dead-re-test class in the scope above.** That is
a scoped non-finding, not a claim that none exist.

---

## 6. FOUND BUT NOT FIXED

1. **`test_costate_digest_ncde.py::test_section_omitted_on_short_telemetry` is RED, and it
   was red before I touched anything** (verified by stashing). Adjudication: the
   **BEHAVIOR is wrong, not the fixture.** The test asserts `section_ncde` returns
   `(None, None)` on fewer than 8 verdict points; it now emits
   `ncde-trajectory: ep? d_seg NO-FIRE (still descending)`. Emitting a verdict over 4
   points is precisely VACUITY==PASS. The fixture encodes the correct contract; the code
   drifted. Do **not** fixture-edit this green. Unowned.

2. **The positive-control ratchet is already breached**: uncovered REFUSE-capable gates =
   **18**, ceiling **17** (`MAX_UNCOVERED_REFUSE_GATES`). Verified pre-existing by
   stashing. My gate is covered and does not worsen the count, but
   `check_refusal_gates_have_live_positive_control` is STRICT, so full-scope preflight
   raises. The commit hook does not surface this — it runs `--no-codebase` and examines
   **0 gates** (announcing so, which is the `ddm_vc1` cure working as designed). Unowned.

3. **`test_confound_gates.py` carries 7 pre-existing red bounds.** Verified by running the
   suite against a stash of my changes: baseline **7 failed / 24 passed**, with my landing
   **7 failed / 25 passed** — the identical seven
   (`check_levelset_hosc_requires_beta_end`, `check_no_raw_virtual_memory_safety_basis`,
   `check_process_guard_excludes_observer_flag_values`, `check_no_stub_lever_factories`,
   `check_refusal_gates_have_live_positive_control`,
   `check_checkpoint_saves_do_not_silently_drop_optimizer_state`,
   `check_no_bulk_write_strands_the_ready_record`), plus my new bound passing. Each is a
   gate whose declared live-count bound no longer matches what it measures — the same
   docstring-says-0-but-measures-N shape `ddm_gh1` closed for one gate in July. Unowned.

4. **Adoption decay, one layer out.** No consumer anywhere in the repo reads
   `verdict_scope`, `corpus_recall`, `active_convening`, or `graph_memory` out of the
   digest's `--json` payload. The fix restores them to the *human* digest, where a reader
   exists. The JSON keys remain producer-only. Consistent with the measured write-only-API
   genus (`verdict.v1`: 486 producers, 0 readers) and a reason not to build further
   machinery on these keys until a reader exists.

5. **`graph_memory` now displays `recalls 24h: 0 ⚠ UNDER-USED`** on the live path — a
   standing adoption-decay signal that has been invisible since 2026-07-23.

---

## 7. Pointer

**UNMOVED.** This is apparatus, not score. What it buys: a recall/advisory surface that
had been dark for 25 days now reaches a reader, and the accidental-truncation fingerprint
is refused structurally rather than nagged about.
