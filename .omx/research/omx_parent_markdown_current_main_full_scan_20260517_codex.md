# OMX Parent Markdown Current-Main Full Scan - 2026-05-17

## Why This Exists

The operator re-raised that relevant OMX/Claude signal may live outside
`.omx/research`. This pass rechecked every Markdown file under the `.omx`
parent tree, including ignored snapshot and temporary trees, before continuing
L5/L5-v2 and Rule #6 routing.

## Commands

```bash
find .omx -name '*.md' -print | wc -l
find .omx -path .omx/research -prune -o -name '*.md' -print | wc -l
find .omx -path .omx/research -prune -o -name '*.md' -print
rg -l -i --hidden --no-ignore 'l5|tt5l|time[-_ ]?trav|staircase|cargo[-_ ]?cult|local minima|local minimum|rule #?6|rule6|score[-_ ]?lower|no signal loss|stack|arithmetic|entropy|ball[eé]|nerv|hnerv|frontier|FEC6|PR101|PR95|PR106|MPS|Modal' .omx --glob '*.md' --glob '!.omx/research/**'
```

## Inventory

Observed on current `main` after the latest parent-scope ledgers landed:

| Bucket | Markdown files |
|---|---:|
| total `.omx/**/*.md` | 2410 |
| non-research `.omx` Markdown | 636 |
| keyword-matching non-research Markdown | 379 |
| `.omx/auto_memory_snapshot_20260504T230223Z` | 562 |
| `.omx/context` | 28 |
| `.omx/state` | 22 |
| `.omx/tmp` | 16 |
| `.omx/plans` | 4 |
| root `.omx/*.md` | 2 |

Root `.omx/*.md` files remain `.omx/notepad.md` and
`.omx/release_manifest_v0.2.0-rc1.md`; neither supersedes the May 17 L5-v2 /
Rule #6 routing.

## Documents Checked For Authority

- `.omx/state/current_focus.md`
- `.omx/state/next_experiments.md`
- `.omx/state/active_lane_dispatch_claims.md`
- `.omx/state/active_dispatches.md`
- `.omx/state/dispatch_queue.md`
- `.omx/state/dispatch_claims_archive/dispatch_claims_2026-05.md`
- `.omx/notepad.md`
- `.omx/release_manifest_v0.2.0-rc1.md`
- `.omx/tmp/observability_section_append_20260516.md`
- `.omx/auto_memory_snapshot_20260504T230223Z/feedback_no_signal_loss.md`
- `.omx/auto_memory_snapshot_20260504T230223Z/feedback_default_to_convenience_trap.md`
- `.omx/auto_memory_snapshot_20260504T230223Z/feedback_no_local_mps_for_authoritative_kill_or_promote_20260429.md`
- `.omx/auto_memory_snapshot_20260504T230223Z/feedback_mps_cuda_drift_critical.md`

## Routing Result

No non-research `.omx` Markdown supersedes the active May 17 control plane:

1. `.omx/state/current_focus.md`
2. `.omx/state/next_experiments.md`
3. `.omx/state/active_lane_dispatch_claims.md`

The parent-scope material does strengthen two standing constraints:

- MPS and macOS/local proxy results must remain fail-closed, axis-labeled, and
  non-promotional. They may support smoke/proxy shape work only.
- Proxy/advisory state writes must use append-only, locked custody. The
  non-research Claude memory explicitly treats no-signal-loss provenance and
  proxy-axis discipline as binding, not optional.

## Actionable Finding And Fix

The scan exposed a concrete code/doc mismatch:

- `.omx/research/one_arg_local_mps_vs_modal_dispatch_switch_design_20260517.md`
  and the parent Claude memories require proxy/advisory manifest rows to be
  append-only, fail-closed, and locked.
- `src/tac/optimization/mps_research_signal.py::append_manifest_row_to_jsonl`
  and the sister macOS-CPU helper used bare `open("a")` appends.

Fix landed in this pass:

- `src/tac/optimization/mps_research_signal.py` now appends under a sibling
  `fcntl.flock(LOCK_EX)` lock file.
- `src/tac/optimization/macos_cpu_advisory_signal.py` now does the same.
- Focused tests assert both fail-closed semantics and lock-file creation.

## Verification

```bash
.venv/bin/python -m pytest src/tac/tests/test_mps_research_signal.py src/tac/tests/test_macos_cpu_advisory_signal_manifest.py -q
```

Result: `33 passed`.

```bash
.venv/bin/python -m ruff check src/tac/optimization/mps_research_signal.py src/tac/optimization/macos_cpu_advisory_signal.py src/tac/tests/test_mps_research_signal.py src/tac/tests/test_macos_cpu_advisory_signal_manifest.py
```

Result: `All checks passed!`.

## Authority

- `score_claim=false`
- `promotion_eligible=false`
- `ready_for_exact_eval_dispatch=false`
- `ready_for_provider_dispatch=false`
- `dispatch_attempted=false`

No provider dispatch was launched. No lane claim was opened. This is a
control-plane scan and proxy/advisory custody hardening patch, not a score or
promotion claim.
