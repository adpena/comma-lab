Cured and committed as `de22c5cd565787bc9d240e275626732d0c986bfe`.

The real cause was argv-visible PATH pollution containing standalone `claude`/`codex` components—not `tools/host_shims` itself. Both canonical launchers now inspect the stable post-`exec` argv and refuse unsafe detached launches. The previous transient `env REAPER_KEEPALIVE=1` workaround was removed.

Artifacts:

- [Evidence memo](/Users/adpena/Projects/pact/.omx/research/ddm_rp2_reaper_shim_cure_20260825.md)
- [Shared launch guard](/Users/adpena/Projects/pact/tools/fleet_reaper_guard.py)
- [Durable SSD receipt](/Volumes/APDataStore/pact/ddm_rp2_reaper_shim_cure/evidence_receipt.json)
- [Passing control manifest](/Volumes/APDataStore/pact/ddm_rp2_reaper_shim_cure/controls/executed_20260825/negative_passed/launch_manifest.json)

Verification:

- Polluted shim-routed control: refused with rc 5 before directory creation.
- Shim-only control: passed with rc 0.
- 46 tests passed; two existing `ps`-based TTY tests were deselected because the sandbox prohibits `ps`.
- Two review passes recorded at 100% coverage.
- Live r7 remained read-only and safe: running after 7,542.959 seconds, `kill_reason=null`, no reaper predicate hits.
- Repository-wide developer preflight remains red on eight global gates after a concurrent sibling commit; this landing’s commit hook passed and its files remain unchanged in current HEAD.

No scorer or Modal job ran. The frontier remains **gb1 — S 0.14811799921260607 @ 180,215 B [contest-CUDA T4, n600]**.

## LIVE-HYPOTHESES

None within this charter’s scope. Source inspection and executed controls closed the identified canonical-launcher mechanism.

## DEAD-ENDS

- **Host Python shim as the causal trigger:** closed because it is an exec-wrapper with no `claude`/`codex` token, and the shim-only control passed.
- **Transient `/usr/bin/env REAPER_KEEPALIVE=1` as immunity:** closed because `env` execs its target, so the marker is not necessarily visible in the stable process argv.
- **Capacity, permission, or harness foreground reaper:** closed by the fleet log’s three exact 344–359 second reap receipts.
- **`--allow-reaper-name-match` escape:** closed; the compatibility flag no longer bypasses refusal.