# Strict-gate violation-count DRIFT ALARM — Catalog #185 scope extension (landed)

**Date:** 2026-07-02T23:09:03Z · **Status:** LANDED (structural self-protect: alarm CHECK + operator
TOOL + committed baseline manifest + 29 tests). **Advisory / apparatus-hardening; pointer 0.19110
UNMOVED** — this landing moves no exact row; it is the META fix that stops a strict gate from silently
decaying red unnoticed. `score_claim=false`, `promotable=false`.

Origin: the orphan-guard agent's adjacent finding
(`.omx/research/orphaned_measured_win_bug_class_selfprotect_and_sweep_20260702T224934Z.md` §0) — Catalog
**#344** silently drifted **0 → 480** and `preflight_all(strict=True)` is red on it with **no alarm**:
the apparatus-blindness meta-bug recursing one layer up (a strict gate decayed red, unnoticed).

---

## 1. TASK 1 — the #185 gap (diagnosed, MEASURED, not asserted)

**Catalog #185 = `check_strict_flipped_catalog_entries_have_live_count_zero`** IS a per-gate-violation
watcher (it invokes gates), but it is **narrowly triggered**. `_check_185_extract_strict_zero_entries`
keeps an entry ONLY when its catalog row body contains BOTH:
1. a `_CHECK_185_LIVE_COUNT_ZERO_PHRASES` phrase (`"live count: 0"`, `"0 -> strict"`, …), AND
2. a `_CHECK_159_STRICT_PHRASES` phrase (`"strict-flipped"`, `"strict @ 0"`, …).

Only for those matched entries does it look up the gate and invoke `fn(strict=False, verbose=False)`.

**WHY it missed #344 (measured):** #344's catalog row (`docs/meta_bug_class_catalog.md:554`) contains
`strict-flipped` (a STRICT phrase) but **does NOT contain the `"live count: 0"` phrase** (measured:
`grep -ci "live count"` on the row = **0**; `grep -ci "strict-flipped"` = 1). So `has_zero_claim=False`
→ `_check_185_extract_strict_zero_entries` `continue`s past it → **#344 is NEVER invoked** → its 0→480
decay is invisible to #185. This is not a scope-of-file bug (#185 correctly reads BOTH `CLAUDE.md` and
`docs/meta_bug_class_catalog.md`), not a signature bug (#344 accepts `(repo_root, strict, verbose)`) —
it is a **TRIGGER-scope bug**: #185 only verifies gates that literally RE-ASSERT "live count: 0" in
their row, and most strict-flipped gates' rows mention only the strict-flip date, not the zero-claim.

**Second structural hole:** #185 has **no committed baseline** — it trusts the row-text's implicit "0",
so an INTENTIONAL non-zero backlog cannot be declared; there is no place to say "#344 is at 480 on
purpose, alarm only if it grows."

**Live measurement (this session):**
- `#185` non-strict currently returns **3** violations → its OWN live-repo tests are RED, pre-existing:
  `#172` (2, autocast-fp16), `#298` (61, stale L1), `#343` (6, hardcoded frontier literals). (These rows
  DO self-claim "live count: 0", so #185 catches them — but nobody acted; the apparatus-blindness again.)
- `#344` live count = **exactly 480**, memos dated **2026-05-20 .. 2026-07-02** — invisible to #185.
- `263` strict callsites total in `preflight_all` (via `_check_176_collect_strict_callsites`); #344 is one
  of them and IS enumerable there.

## 2. TASK 2 — the DRIFT ALARM (EXTEND #185, no new number)

Per the gate-consolidation discipline (Catalog #299, quota ~397/400) and the task's explicit
"EXTEND/REPAIR #185 rather than add a new number", this **widens #185's scope** with a companion that
watches gates regardless of whether their row self-claims "live count: 0", driven by a **committed
baseline manifest** so intentional non-zero counts are declared. **NO new catalog number claimed.**

- **Manifest** `.omx/state/strict_gate_violation_baseline.json` (committed; `.gitignore` `!`-allowlisted):
  per watched gate `{catalog, baseline_max, reason, first_seen_utc}`. Seeded with the 4 known drifted
  strict gates at their MEASURED current counts: **#344→480, #298→61, #172→2, #343→6**. Alarm semantics:
  `live > baseline_max` = OVER_BASELINE (alarm); `== ` = AT_BASELINE (ok); `<` = UNDER_BASELINE
  (advisory: burned down, tighten); declared-but-not-callable = MISSING_CALLABLE (stale watchlist).
- **CHECK** `tac.preflight.check_strict_gate_violation_counts_within_declared_baseline` — wired
  **WARN-ONLY** in `preflight_all` right after the #185 strict callsite. Flags OVER_BASELINE +
  MISSING_CALLABLE. WARN-only because (a) Strict-flip atomicity and (b) no new catalog number ⇒ no
  Catalog #176 strict-callsite-needs-row obligation. Now **#344 IS WATCHED** (baseline 480) — the exact
  blind spot, closed. Green at landing (all 4 gates AT_BASELINE, 0 violations).
- **TOOL** `tools/audit_strict_gate_violation_drift.py` — the rc-bearing operator alarm. Default:
  snapshot the declared watchlist, **rc=1** on any OVER_BASELINE/MISSING_CALLABLE (rc=0 at landing).
  `--full`: best-effort snapshot of **EVERY** strict callsite (finds NOT-YET-declared drift — the true
  blind-spot closer; slow, operator/nightly on-demand). `--json`; `--update-baseline` (re-baseline after
  a burndown / seed new gates).
- **Shared evaluator** `evaluate_strict_gate_violation_drift` powers BOTH the check and the tool so they
  never drift (same pattern as the sister #396 `classify_findings_memo_orphan_status`).

## 3. TASK 3 — the #344 backlog, FLAGGED for the operator (NOT done here)

**#344 backlog = 480 memos** dated **2026-05-20 .. 2026-07-02** in `.omx/research/*.md` that carry a #344
empirical-finding trigger token with no `canonical_equation` reference and no `# FORMALIZATION_PENDING:`
waiver. Burndown = register the equation OR add a substantive `FORMALIZATION_PENDING` waiver per memo
(operator-scoped, task #225). It is DECLARED in the manifest (baseline_max 480) so the alarm is green;
**lower baseline_max as it burns down.** This alarm does NOT touch the backfill.

**Adjacent PRE-EXISTING backlog also flagged** (same class, operator-scoped, NOT this landing's job):
`#298`=61, `#172`=2, `#343`=6 — these make #185's own live-repo tests
(`test_live_repo_strict_zero_entries_have_zero_violations`, `test_live_repo_strict_mode_passes`) RED
today (2 failures, verified pre-existing, none created here). Burn down or update those rows.

**Warn-first status:** the alarm does NOT go red on #344 (declared 480) and is wired WARN-only anyway, so
Strict-flip atomicity is honored twice over. Strict-flip is a future follow-up (claim a number or fold
into #185's strict function) once the operator sets backlog policy.

## 4. TASK 4 — tests + docs

- **Tests** `src/tac/tests/test_check_185_strict_gate_violation_drift_alarm.py` — **29 pass**: verdict
  constants distinct; manifest loader (missing/malformed/missing-gates/valid/bad-baseline_max/non-dict);
  snapshot (missing-callable→None, list-count, non-list→0, bare-signature fallback, incompatible-sig→None);
  classifier (over/at/under/None); evaluate (MISSING_CALLABLE, over-baseline record shape); check
  (no-manifest→[], at-baseline clean, over-baseline alarm w/ count+catalog named, missing-callable alarm,
  under-baseline advisory-not-violation, strict-raise, strict-clean, verbose ALARM, str repo_root); LIVE
  repo alarm GREEN + #344 in watched set.
- **Catalog doc**: appended a SCOPE-EXTENSION note to the #185 row in `docs/meta_bug_class_catalog.md`
  (no new number; documents the companion check + manifest + tool + #344 anchor + WARN-ONLY status).

## 5. Verification (measured)

- `ruff check --select F821,F401` clean on `src/tac/preflight.py`, `tools/audit_strict_gate_violation_drift.py`,
  and the test file.
- New alarm on live repo: **0 violations** (all 4 declared gates AT_BASELINE). Tool default rc=**0**.
- Catalog-integrity gates AFTER this landing: **#118=0, #159=0, #176=0** (my WARN-only callsite is NOT
  demanded a row); **#185=3** (the PRE-EXISTING #172/#298/#343 — unchanged; my doc edit added no new
  matched entries; #185 self-skips its own row).
- 29/29 new tests pass.

## 6. 6-hook wire-in (Catalog #125)

- #1 sensitivity-map / #2 Pareto / #3 bit-allocator / #5 continual-learning = **N/A** (an
  apparatus-hardening gate, not a score-axis contributor).
- #4 cathedral autopilot dispatch = **ACTIVE** — the tool is the operator-facing rc-bearing drift alarm
  the autopilot / nightly regression can consult.
- #6 probe-disambiguator = **ACTIVE** — `evaluate_strict_gate_violation_drift` / `classify_strict_gate_drift`
  ARE the disambiguator between OVER/AT/UNDER_BASELINE / MISSING_CALLABLE / NOT_INVOKABLE.

Mission contribution (Catalog #300): `frontier_protecting` — extincts the "a strict gate decays red
unnoticed" apparatus-blindness class (canonical anchor #344's 0→480) so a red strict gate now trips an
alarm instead of rotting.
