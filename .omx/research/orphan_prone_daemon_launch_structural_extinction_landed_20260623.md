# Orphan-prone daemon launch — structural extinction LANDED (Catalog #389)

Date: 2026-06-23
Subagent: orphan_daemon_fix_20260623
Operator binding: "permanently fix that bug class"
Non-negotiables honored: "Durable detached daemons, not session-watchers" +
"Bugs must be permanently fixed AND self-protected against" (the 2-landing
discipline) + NO-FAKE (the headline test spawns a REAL parent+child daemon and
proves no-orphan).

## Bug class
`nohup bash -c '<worker> | tee LOG > /dev/null' & disown` => process tree
`bash-wrapper -> {python worker, tee}`. Wrapper PID != worker PID, so killing
the wrapper ORPHANS the worker (reparented to init, keeps running).
**Anchor 2026-06-23:** dashboard bash-wrapper 84946 killed, python child 84948
survived; two renderer pythons then both rewrote the dashboard index.html every
20s and the operator saw a STALE stopped run.

## Landing 1 — the fix
`tools/spawn_durable_daemon.py` extended (start behavior byte-compatible by
default; the worker remains its own session/process-group leader via
`Popen(start_new_session=True)` — no bash wrapper):
- registry `.omx/state/durable_daemons.json` (fcntl-locked, lock-load-mutate-
  save + unique-tmp + fsync + os.replace per the Catalog #128/#131/#245 pattern;
  mirrors `tac.deploy.azure.active_vms_state`).
- `--label NAME` (required-recommended; synthesized + warned if omitted; upsert
  by label).
- `--stop NAME`: pgid resolve -> `os.killpg(pgid, SIGTERM)` WHOLE-GROUP kill ->
  SIGKILL escalation -> verify dead -> mark stopped. NO ORPHAN.
- `--status`: registry + live pid/pgid check.

## Landing 2 — the self-protect gate
`check_no_orphan_prone_daemon_launch` (Catalog #389) in `src/tac/preflight.py`,
wired into `preflight_all()` `strict=True`. Scans tools/scripts/experiments for
the SPECIFIC `nohup + bash -c + | tee + bg` signature; LOW FP (plain nohup,
short backgrounded cmds, run_in_background NOT flagged). Same-line waiver
`# ORPHAN_PRONE_DAEMON_LAUNCH_OK:<rationale>` (placeholder rejected).
**STRICT @ 0** — live count 0 at landing.

## Tests
- `src/tac/tests/test_spawn_durable_daemon_lifecycle.py` (14): HEADLINE
  `test_stop_kills_whole_group_no_orphan` (real parent+child, both dead after
  --stop) + orphan-proof + registry round-trip + status LIVE->DEAD + upsert +
  edge cases. **PASS.**
- `src/tac/tests/test_check_389_orphan_prone_daemon_launch.py` (21): signature
  positive/negative, placeholder rejection, scanner catch/strict, allow-list,
  waiver-respect, live-repo regression count==0. **PASS.**

## Worked example + follow-ups
Dashboard renderer ALREADY migrated to spawn_durable_daemon.py. Follow-up
migration candidates (NOT migrated now — would disrupt live infra): http.server
(8733), cloudflared (47775), Monitor, training daemons.

## 6-hook wire-in declaration (Catalog #125)
1. sensitivity-map — N/A (infrastructure validator gate; no per-byte saliency).
2. Pareto constraint — N/A.
3. bit-allocator hook — N/A.
4. cathedral autopilot dispatch — **ACTIVE** (prevents future orphan-prone
   daemon launches from shipping; steers all detached daemons to the canonical
   lifecycle so background work never races stale zombies).
5. continual-learning posterior — N/A (no empirical score anchor; the registry
   is the durable consumed surface).
6. probe-disambiguator — **ACTIVE** (the registry pgid + group-kill IS the
   disambiguator between "killed the daemon" and "orphaned the worker").

Mission contribution (Catalog #300): `apparatus_maintenance` /
`frontier_protecting` (prevents zombie-daemon races from corrupting the
operator's live observability + wasting CPU; pointer UNMOVED 0.19110, no score
claim — this is infrastructure).
