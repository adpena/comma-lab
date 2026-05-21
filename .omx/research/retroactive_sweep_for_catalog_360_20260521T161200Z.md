# Retroactive Sweep For Catalog #360 - Pre-Spawn-Fatal Observability Extinction

Author: Claude (OVERNIGHT-HH bug-audit-fix-cascade subagent)
UTC: 2026-05-21T16:12:00Z
Catalog: #360
Gate: `check_modal_dispatcher_pre_spawn_fatal_observability`
Source directive: operator NON-NEGOTIABLE 2026-05-21 "Fix all bugs"
Lane: `lane_overnight_hh_comprehensive_bug_audit_fix_cascade_20260521`

## Bug-Class Symptom Signature

`@app.local_entrypoint() def main()` inside `experiments/modal_train_lane.py`
contains 12+ `sys.exit(2)` FATAL paths upstream of `fn.spawn(...)`. When ANY
of them fires, Modal's behavior is:

1. App initializes (creates mounts + functions; ~5s prelude)
2. `main()` calls `sys.exit(2)` for the FATAL precondition
3. Modal sees a clean exit with no `fn.spawn()` call queued
4. App transitions to "stopped" state with `0 tasks`
5. **NO row appears in `.omx/state/modal_call_id_ledger.jsonl`**
6. **NO `modal_metadata.json` is written** (the writer only runs post-spawn)
7. **NO recovery dump is created** at `.omx/state/modal_call_id_ledger_recovery_tmp/`

Result: the dispatch is INVISIBLE to the harvester per CLAUDE.md "Modal
`.spawn()` HARVEST OR LOSE" non-negotiable. Operator observes silent
failure with no structured signal. The print() to stderr goes to local
terminal but is LOST if the operator is running in background or
dispatch was via subprocess.

The symptom matches Catalog #339's "silent-no-spawn" pattern, but at a
DIFFERENT surface: #339 catches POST-spawn registration silent-swallow,
#360 catches PRE-spawn sys.exit silent-no-spawn.

## Pre-Fix Window

Pre-fix window: every `experiments/modal_train_lane.py` commit between
2026-05-15 (Catalog #245 canonical Modal call_id ledger landing) and
2026-05-21T16:00:00Z (this fix). During this ~6 day window any dispatch
that hit one of the 12 sys.exit(2) paths in main() produced a silent-no-
spawn dispatch invisible to the canonical ledger.

## Historical-KILL/DEFER/FALSIFY Search Results

Direct empirical anchors (all 5 STC v2 silent-no-spawn dispatches per
`.omx/research/stc_v2_ratify_or_defer_path_b_dispatch_landed_20260521.md`
5-failure pattern table):

| # | Date UTC | Modal app | Call ID | Verdict | Path |
|---|---|---|---|---|---|
| 1 | 2026-05-16T21:31Z | — | fc-01KRSB76H04HM4958V2HX2JZZ4 | rc=25 (failed_dispatch) | Catalog #146 / #204 / #220 fix path |
| 2 | 2026-05-17T02:17Z | — | fc-01KRSVKF9VEESQY2FS33FF4WDM | rc=25 | Driver path-layer fix attempt |
| 3 | 2026-05-19T06:31Z | ap-rlIMf5jMhPaF1FbwNVLpZq | **(no row)** | DEFER (advisory) | First silent-no-spawn |
| 4 | 2026-05-19 | (cheap_signal_first_wave) | **(no row)** | (silent) | 2nd silent-no-spawn |
| 5 | 2026-05-21T07:41Z | ap-KA1LFP69IGthTDNrXGXRie | **(no row)** | DEFER (advisory) | 5th consecutive silent-no-spawn |

Historical verdict status per CLAUDE.md "Forbidden premature KILL":
all 5 anchors classified IMPLEMENTATION-LEVEL operational failures per
Catalog #307 — STC v2 substrate paradigm INTACT; dispatch surface for
THIS recipe structurally broken. Probe-outcomes ledger row
`overnight_j_stc_v2_path_b_dispatch_silent_no_spawn_5th_consecutive_20260521T074400Z`
adjudicated DEFER per Catalog #313. No KILL verdicts were rotted by
the pre-fix silent-no-spawn window because the operator + Assumption-
Adversary correctly classified the symptoms as OPERATIONAL rather than
PARADIGM falsification.

Sister-cross-substrate empirical: DP1 paired dispatches
`fc-01KS4KJGDXVXZ9NYRD4HKZ9CET` + `fc-01KS4KKYQ09DEEW6BCDRGPBE93`
landed 2026-05-21T01:29Z via SAME `experiments/modal_train_lane.py` +
`tools/operator_authorize.py` code path — confirming the bug class is
STC v2 specific (or substrate-class specific) NOT Modal-wide. This
narrows the failure mode to ONE of the 12 sys.exit(2) paths upstream
of fn.spawn, most likely the trainer-discovery block (lines 1781 /
1795 / 1817 / 1838) since DP1's trainer + extra-mount-payload differs
from STC v2's.

## Per-Finding RE-EVAL-Priority Assignment

- **HIGH**: All 5 STC v2 silent-no-spawn anchors retroactively gain
  ledger visibility from this point forward via the new
  `EVENT_PRE_SPAWN_FATAL` event type. Future operator-routable: re-fire
  STC v2 (or its sister path 3a A1 residual) with the canonical helper
  active; if it fires silent-no-spawn AGAIN, the new ledger row will
  carry `sys_exit_line_number` + `sys_exit_helper_source` + `fatal_reason`
  identifying WHICH of the 12 paths fired. This converts the failure
  from undiagnosable to operator-actionable.
- **MEDIUM**: All other Modal dispatches from the pre-fix window are
  EITHER: (a) succeeded (have canonical ledger rows; no impact) OR
  (b) failed silent-no-spawn (no ledger row; no operator memory loss
  since the symptom was always "stopped/0 tasks" with no recovery
  dump). No KILL or DEFER verdicts hinged on the missing observability.
- **LOW**: Pre-Catalog #245 (before 2026-05-15) dispatches predate the
  canonical ledger entirely and are out of scope for retroactive sweep.

## Authority And Wire-In

Catalog #360 lands STRICT-from-byte-one per CLAUDE.md "Strict-flip
atomicity rule" — live count at landing: 0 (the fix routes 9 HIGH-RISK
pre-spawn sys.exit paths in main() through the inline `_pre_spawn_fatal`
helper which delegates to `register_pre_spawn_fatal` in the same commit
batch).

Wire-in surfaces:
- `experiments/modal_train_lane.py::main` carries the canonical inline
  `_pre_spawn_fatal(reason, line_no, helper_source)` helper at the top
  of main() (before the first FATAL path).
- All 9 in-scope sys.exit paths route through the helper before exit.
- `src/tac/deploy/modal/call_id_ledger.py::register_pre_spawn_fatal`
  is the canonical helper (~150 LOC fail-OPEN observability writer).
- `src/tac/preflight.py::check_modal_dispatcher_pre_spawn_fatal_observability`
  is the STRICT preflight gate wired into `preflight_all(strict=True)`.
- `src/tac/tests/test_check_360_pre_spawn_fatal_observability.py` carries
  17 dedicated tests covering all canonical surfaces.

Sister of Catalog #339 at the POST-spawn surface; together they extinct
the silent-no-spawn bug class at BOTH orthogonal surfaces.

## Mission Contribution

`frontier_protecting` per Catalog #300: the fix preserves the harvester
invariant per CLAUDE.md "Modal `.spawn()` HARVEST OR LOSE"
non-negotiable so future paid Modal dispatches cannot become invisible
orphans. The 5 historical STC v2 silent-no-spawn anchors retroactively
gain ledger visibility from this point forward; future dispatches that
hit the same bug class produce structured signal the operator can
diagnose without manual app-id ↔ failure-pattern correlation.

## Cross-References

- `feedback_overnight_hh_comprehensive_bug_audit_fix_cascade_landed_20260521.md` (landing memo)
- `.omx/research/stc_v2_ratify_or_defer_path_b_dispatch_landed_20260521.md` (OVERNIGHT-J 5-failure pattern empirical anchor)
- `feedback_silent_no_spawn_structural_extinction_landed_20260519.md` (Catalog #339 sister wave)
- `feedback_modal_call_id_ledger_canonical_landed_20260515.md` (Catalog #245 canonical 4-layer pattern this gate operationalizes for the pre-spawn surface)
