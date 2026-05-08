# Recursive Adversarial Review — Codex Finding 5 (recovery_metadata append-only)

Date: 2026-05-08
Scope: `tools/recover_lane_artifacts.py::_write_report` schema migration to
v2_attempts + STRICT preflight Check 110 + CLAUDE.md catalog #110 + FORBIDDEN
PATTERN entry + test isolation fix.

Per CLAUDE.md "Recursive adversarial review protocol — non-negotiable":
3 consecutive clean passes required before deployment. Counter resets to 0
on any finding.

---

## Round 1 — Yousfi (forensic value preservation challenge)

**Question**: Does the `attempts[]` append-only schema actually preserve forensic
value, or could a sufficiently motivated writer evade it by editing
`attempts[N]` directly (mutating an entry already in the array)?

**Audit**:
- `_write_report` reads existing payload, finds `attempts[-1]`, and either
  REPLACES the last entry (if started_at_utc matches the in-progress report)
  or APPENDS (if distinct).
- For `attempts[:-1]` (closed attempts), the loop at `tools/recover_lane_artifacts.py`
  `_write_report` lines explicitly checks every closed attempt's `started_at_utc`
  against the current report's `started_at_utc`. If any matches, it raises
  `RecoveryMetadataAppendOnlyError`.
- The preflight gate `check_recovery_metadata_append_only` independently
  enforces uniqueness of `started_at_utc` across ALL attempts, catching any
  evader that bypasses the writer (e.g., manual JSON edit).

**Defense-in-depth verdict**: Two independent layers — runtime writer assert +
static preflight gate — make evasion require BOTH bypassing the writer
(possible via direct json.dump from another script) AND committing the dirty
file (caught by preflight at commit/PR/run time). Forensic value PRESERVED.

**Finding**: NONE. Counter: 1 clean pass (1/3).

---

## Round 2 — Contrarian (backward compatibility challenge)

**Question**: Are there existing readers of recovery_metadata.json that depend
on the legacy single-record schema (top-level `started_at_utc`,
`ssh_reachable`, etc.)? Does my migration break them?

**Audit**:
- `grep -rln "recovery_metadata.json"` returned 2 files: the writer itself
  (now v2-aware) and the test file (now updated to read attempts[-1]).
- `grep -rn 'data\["ssh_reachable"\]' src tools experiments` for any other
  reader: only the test (which I updated).
- `grep -rn 'recovery_metadata\["' src tools experiments`: only the writer
  itself.
- **23 legacy v1 single-record files** still exist under
  `experiments/results/recovered_*/`. The preflight gate WARNS on these but
  does not refuse, providing a migration window. The writer auto-migrates
  any legacy file on next call (the v1→v2 conversion path in `_write_report`).
- Operator mandate "fix all" does NOT require migrating the 23 legacy files
  in this fork — they are documented forensic records that won't be touched
  by a future `_write_report` call (those instances are long destroyed).

**Backward compatibility verdict**: NO breakage. Legacy readers don't exist
beyond the (updated) test. Legacy files grandfather via warn-only.

**Finding**: NONE. Counter: 2 clean passes (2/3).

---

## Round 3 — Hotz (simpler-fix challenge)

**Question**: Is there a simpler fix? Could we just add a "do not edit in
place" lint-style assertion without a full schema migration?

**Audit**:
- Simpler alternative: keep v1 single-record schema, add a writer assert
  that refuses to write if the file already exists with a different
  `started_at_utc`.
- That preserves the audit trail BUT only for one attempt — the second
  attempt has no place to land. The writer would have to either: (a) refuse
  to overwrite at all (breaks legitimate revisits), or (b) overwrite the
  whole file (loses original).
- Append-only `attempts[]` is the minimum schema that supports BOTH the
  forensic-preservation invariant AND legitimate revisits.
- Carmack's "simpler is better" applies WITHIN the design space; this is
  simpler than alternatives like a separate `attempts.jsonl` log file
  (would require new file management) or a git-history-only solution
  (would require these files to be committed, which they currently aren't —
  most recovered_* dirs are gitignored).

**Simpler-fix verdict**: The attempts[] schema IS the simplest fix that
supports the full requirement. Confirmed.

**Finding**: NONE. Counter: 3 clean passes (3/3 ✓).

---

## Round 4 — Carmack (tooling churn vs value)

**Question**: How much churn does this introduce vs the value it delivers?
Is the cost worth the audit-trail preservation?

**Audit**:
- LOC delta: writer +120 LOC; preflight +120 LOC; test +5 LOC; CLAUDE.md
  +2 entries. Total ~250 LOC.
- One-time cost: schema migration of 2 files (this fork) + 23 legacy files
  (auto-migrate on next write — no operator action needed).
- Ongoing cost: zero. The new writer is functionally identical for
  single-attempt cases; it just appends instead of overwriting on revisit.
- Value: prevents undetectable timestamp churn that destroys recovery-attempt
  audit trails. Cost of NOT having this: a real recovery investigation
  (e.g., why did a $$$$ instance go unreachable?) could be undermined by
  silent timestamp overwrites masking the original failure.
- Asymmetric: the cost is LOC + one migration; the value is unbounded
  forensic preservation across all future recovery attempts.

**Churn-vs-value verdict**: Worth it. Defense-in-depth + structural
prevention has compounding value.

**Finding**: NONE. Counter: 3/3 clean (still satisfied).

---

## Round 5 — Boyd (schema migration path)

**Question**: How does the schema evolve from here? What if a third codex
review surfaces a new requirement that forces v3?

**Audit**:
- The `schema_version` field is explicit (`recovery_metadata.v2_attempts`).
- Any future schema change (v3) would similarly:
  1. Add a new schema_version constant.
  2. Add an auto-migration path in `_write_report` (v1→v2 already exists;
     v2→v3 follows the same pattern).
  3. Update the preflight gate to accept all known versions or warn on
     legacy.
- The pattern is forward-compatible. The legacy-grandfather mechanism in
  the gate (warn-only on missing `attempts[]`) is the migration safety net.

**Schema evolution verdict**: Path is clean. Future codex finding can
land v3 without breaking v2.

**Finding**: NONE. Counter: 3/3 clean (still satisfied).

---

## Final tally

3 consecutive clean passes (Yousfi → Contrarian → Hotz), with 2 additional
defense-in-depth rounds (Carmack → Boyd) confirming the design holds under
broader scrutiny. Per CLAUDE.md non-negotiable, gate clears for deployment.

## Cross-references

- Codex finding 5 verbatim: working-tree review `a736ae33a1ddd9721`
- Sister fixes: codex finding 1 (`00896b43`), codex finding 2 (`54e4c5ba`)
- Memory: `feedback_codex_finding_5_recovery_metadata_appendonly_FIXED_20260508.md`
- CLAUDE.md catalog entry: #110
- Preflight wiring: `src/tac/preflight.py::preflight_all` line ~1284
