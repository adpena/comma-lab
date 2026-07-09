# MAGNITUDE-DISMISSAL self-protection hook + gate — build ledger (2026-07-08)

**Landing #2** of the "bugs must be permanently fixed AND self-protected against"
pattern for the MAGNITUDE-DISMISSAL bug class. Landing #1 = the memory
`relative-not-absolute-significance-near-goal-dont-orphan-small-deltaS` + the live
re-audit. Operator directive: *"we need a gate or hook or to wire up fmtools
accordingly to trigger examine/audit/adversarial-review in those cases."*

`pointer 0.19110 UNMOVED` — this is apparatus; the live #205 run was not touched;
nothing was launched.

## The bug class

A DEFER / DOWNGRADE-to-WEAK / ORPHAN / KILL verdict on a lever/finding justified by
ABSOLUTE MAGNITUDE ("weak / negligible / noise / small ΔS / little to gain / not
worth it") WITHOUT either:
- **(a)** a RELATIVE-significance computation — ΔS / (S_current − S_target), the
  fraction of the remaining descent this lever buys at the current operating point; OR
- **(b)** a cited MEASUREMENT of un-recoverability — the #141 label-noise / noise-floor
  case, WITH a measurement / exit criterion.

The two legitimate dismissals — **"measured un-recoverable"** and **"structurally
superseded"** — must NOT trip the alarm; nor must non-dismissal magnitude usages
("weak supervision", "noise floor", "noise injection"). Only the un-justified
magnitude-dismissal does.

## What landed

| Surface | File | Grade |
|---|---|---|
| Stop-hook (runtime) | `tools/magnitude_dismissal_detector.py` | fmtools-wired |
| Static preflight gate | `tac.confound_gates.check_no_unjustified_magnitude_dismissal` (#404) | WARN-ONLY |
| Hook registration | `.claude/settings.json` Stop → sibling of triality_drift_detector | wired |
| Tests | `src/tac/tests/test_magnitude_dismissal_detector.py` (28) + confound registry/live-count (86 pass) | green |
| Catalog row | `docs/meta_bug_class_catalog.md` #404 | landed |
| Memory | `feedback_relative_not_absolute_significance_near_goal_dont_orphan_small_deltaS_20260708.md` (landing #1) | cross-ref'd |
| DAG | FEED-magdismiss in `sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md` | landed |

## Design (mirrors the triality-drift-detector precedent)

**Two-stage, robust:**
1. **Deterministic PRE-FILTER** (`magnitude_dismissal_candidates`): slides a 3-line
   window; a window is a candidate iff it co-locates a dismissal verb AND a magnitude
   word AND is NOT exempt (relative-sig / measured-un-recoverability / non-dismissal
   usage / valid waiver / discussion-cue) within a wider 5-line window. Adjacent
   candidates collapse. This carries the weight so the self-protection is live even
   without fmtools.
2. **fmtools SEMANTIC confirm** (`fm_confirm`): runs the on-device Apple FM
   (`apple_fm_sdk` + `fmtools.local_extract`, closed 2-value schema) in a SUBPROCESS
   under the fmtools venv — the pact venv gains zero deps — over ONLY the pre-filtered
   candidates (never per-line). It prunes false positives the regex missed. **Verified
   live on this box**: fmtools venv PRESENT, `fm_confirm` `ran=True` and correctly
   classified a real positive as `unjustified_magnitude_dismissal`.

**Authority boundary:** ADVISORY only. A Stop-hook `{"decision":"block"}` re-engages the
agent at turn-end to trigger the examine/audit/adversarial-review; it does NOT block a
launch (governor + launch_guard_hook keep launch/heavy/paid authority — the correct
boundary). WARN-ONLY on the static gate.

**Fail-open discipline (mirrors triality_drift_detector exactly):** every path wrapped;
non-git dir / bad stdin / timeout / missing fmtools ⇒ exit 0. A last-resort `except:
sys.exit(0)` guarantees the hook can NEVER wedge or crash a session. The static gate
fail-opens to a no-op if the hook module (the classifier SoT) is absent.

**Loop-safety:** `stop_hook_active` flag + persisted `last_block_head` backstop → one
nudge per drift state. On block, `last_head` is NOT advanced, so the fix/waiver commit
lands inside the next window's union and clears the candidate.

**Escape valves ("never binary"):** window-wide `[magnitude-ok]` / `[skip-magnitude]`
commit token; per-line `# MAGNITUDE_DISMISSAL_OK:<rationale>` (placeholder rationales
rejected, ≥4 chars, per the Catalog #287 sister discipline).

## fmtools status: WIRED, not fallback

The fmtools venv is present on this machine and the semantic layer ran live and
confirmed the deterministic candidate. The deterministic fallback path (fmtools absent →
deterministic verdict stands with an honest "confirmation owed" label) is fully
implemented and unit-tested (`test_fm_confirm_absent_returns_not_ran`,
`test_build_reason_labels_owed_when_fm_absent`) so the hook is honest on any machine —
never a faked FM call.

## One classifier, one source of truth

The deterministic predicates live ONLY in the hook module (Claude-workflow apparatus,
not `tac` — per the CLAUDE.md `tac`-cleanliness rule). The confound gate opportunistically
imports them (`_load_magnitude_detector`) so there is no duplicated regex to drift; if the
hook is missing, the gate fail-opens to a no-op rather than crashing preflight.

## Live-count + strict-flip

Static gate on the real `.omx/research` corpus: **15 hits** (max_report cap) — historical
memos that predate the discipline (this is exactly the memory-point-3 re-audit backlog).
WARN-ONLY at landing. **Strict-flip condition**: after the re-audit sweep re-ranks every
prior absolute-magnitude DEFER/DOWNGRADE/ORPHAN at the current operating point (relative
significance) and drains live-count to 0.

## Verification run

- `pytest test_magnitude_dismissal_detector.py` — 28 passed.
- `pytest test_confound_gates.py` — 86 passed (registry + live-count-bound updated for #404).
- `ruff check --select F` — clean on both edited files.
- Hook exit-0 smoke on the live repo + garbage-stdin — both rc=0 (fail-open).
- fmtools live: `fm_confirm` ran=True, confirmed the positive.
