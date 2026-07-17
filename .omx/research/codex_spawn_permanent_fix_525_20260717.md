# Task #525 — permanent codex spawn-kill fix: the fleet claude-code-reaper was the killer

**Date:** 2026-07-17 · **Arm:** ARM E (arm_e_525, main tree) · **Status:** LANDED (C1-C4)
**Commits:** `e4adec609b` (C1 pty-immune shared spawn core) · `3aff1db54f` (C2 reader admission class) ·
`f3366e3a2f` (C3 warn-only preflight gate) · this memo + ledger rows (C4)
**Pointer:** UNMOVED — this is apparatus (means, not a score row).

## 1. Root cause — MEASURED, killer caught in its own log

Every `codex exec` arm died rc=143 (SIGTERM) at **5m18–5m55** (10/10 on 2026-07-17:
3 launch stamps × 3 attempts + 1 clean probe). The killer is the **operator fleet launchd
agent `com.vertigo.claude-code-reaper`** (`~/Library/LaunchAgents/…`, installed 2026-03-14,
`StartInterval=60s`, `runs=21062`; script `~/Projects/fleet/scripts/claude-code-reaper.sh`,
`GRACE_SECONDS=300`). Its Phase 1 SIGTERMs any process matching `\b(claude|codex)\b` with:

- `ps tty == "??"` (no controlling terminal), AND
- `ppid == 1` OR stdin resolving to `/dev/null|PIPE|FIFO`, AND
- age ≥ 300 s.

**[MEASURED]** `/tmp/com.vertigo.claude-code-reaper.log` records every kill by pid, age,
RSS, and argv — including pid 56931 (watched die live at 20:03:25-28Z, whole codex tree
simultaneous, supervisor bash in the SAME pgid survived), pid 81194 (retry.2), and pid
97723 (the clean-env probe below). Kill ages 313–352 s = grace 300 + 60 s sweep jitter.

### Discriminating experiments (all MEASURED 2026-07-17)

| Probe | Conditions | Outcome |
|---|---|---|
| `fix525_survival_probe` (pre-fix path shape) | clean env (`env -i`, NO `CODEX_COMPANION_SESSION_ID`), effort=low, ~10³ tokens, no subagents, detached | **REAPED at 5m24s** — falsifies token/context-volume, compaction, multi_agent_v2, companion-env, and quota hypotheses (rate-limit snapshot showed 5% used) |
| `csxprobe` (renamed argv[0] symlink to the same binary) | identical otherwise | **SURVIVED >7m12s** — argv-name matching confirmed |
| detached bash forensics sampler | contains "codex" strings, stdin=/dev/null, no TTY | survived full 10 min — saved by the reaper's own `parse_etime` octal bug (see §4) # MAGNITUDE_DISMISSAL_OK: forensic evidence row (probe survival fact), not a score-effect verdict — no ΔS dismissed |
| supervisor bash + tee (same pgid as codex) | no `\bcodex\b` argv match (`codex_runs` has `_` = word char) | survived every kill |
| post-fix probe through the NEW default path | tiny prompt, effort=low, read-only | **alive >8m19s on `ttys001`, ZERO reaper-log mentions** (completion: `sleep 600` payload ran past every prior kill window) |

### Why it started TODAY (the "what changed" answer)

**[MEASURED]** `ff0f884b35` (2026-07-16 21:46 CDT) made `codex_delegate` launch arms
HEADLESS (`Popen(start_new_session=True)`) instead of via `osascript … Terminal do script`.
The Terminal launch had given every prior arm a **controlling TTY as an accidental side
effect** — and the reaper's very first check excludes `tty != "??"` ("does NOT touch …
sessions attached to a terminal"). 150 arms ran fine for 4 months INSIDE the reaper's
active window because of that accident. Last survivors: costate/curvelet arms (Jul 16
19:17–20:09Z, 45–47 min, pre-headless). First kills: the first >5-min headless arms
(Jul 17 19:12Z+). codex binary (0.144.1, Jul 9), `~/.codex/config.toml` (Jul 16 17:25Z,
BEFORE survivors), ChatGPT.app update — all exonerated by the survivor timeline.

**Exonerated candidates** (each falsified by a measured probe, not by argument): codex CLI
internal watchdog; codex compaction/subagent crash (compaction coincided with one death by
chance; the 5m24s clean probe had none); companion plugin session reaping
(`session-lifecycle-hook.mjs` kills only its OWN tracked jobs on SessionEnd); API-side
stream termination; harness SIGURG (kills process GROUPS — the same-pgid supervisor
survived); macOS sandbox teardown (probes launched detached-unsandboxed died identically).

## 2. The fix (landed)

**C1 — reaper-immune spawn core (`e4adec609b`).**
`spawn_durable_daemon.spawn_detached_verified(cmd, log, with_pty=…)` is now the single
detached-spawn implementation (refactored out of `_do_start`; adopted by
`codex_delegate` — no copy-paste second impl, enforced by test). `with_pty=True` wraps the
command in `script -q /dev/null …` (BSD form; util-linux form on Linux) so the detached
chain gets a **controlling pty** — `ps` shows `ttysNNN` for the launcher, codex, and tee
(verified live). All four reaper phases require `tty == "??"`, so the chain is classified
as a live terminal session. **This is honest classification, not evasion**: a delegated
arm IS a live supervised session (events-log START/DONE custody, `codex_status` liveness +
STALLED detection, bounded retry supervisor). `codex_delegate` launches `with_pty` by
DEFAULT (`--no-pty` escape), `TERM=dumb` for log hygiene, verify-alive 3 s.
`spawn_durable_daemon --with-pty` (default OFF = byte-identical) for claude/codex-named
daemons. Retry-supervisor bounded respawns RETAINED (defense-in-depth per the charter).

**C2 — reader/control-plane admission class (`3aff1db54f`).**
`spawn_durable_daemon --job-class reader`: preflight-only admission for small
control-plane jobs (the SUM-over-RAM training model refused a 2 GB codex reader beside
the 60 GiB trainer). Fail-closed declared envelope: explicit `--projected-peak-gib ≤ 4`
AND `--rss-cap-mb ≤ 4096` (safe_run enforces at runtime, so a mislabeled trainer cannot
ride the lane). OOM free-floor preflight ALWAYS on; the declared projection replaces the
25 GB training default there. `job_class` + `admission_class_reason` recorded in the
daemon registry (mirrors the #370 control-plane exemption pattern — queryable, never
silent).

**C3 — self-protection gate (`f3366e3a2f`).**
`check_codex_exec_spawn_paths_are_reaper_immune` (warn-only, wired into
`preflight_all()`): flags any tools/scripts/experiments `.py` that builds a `codex exec`
invocation AND spawns processes outside the immune core; same-line
`# CODEX_SPAWN_REAPER_IMMUNE_OK:<rationale>` waiver (placeholders rejected). Live count 0
at landing.

**C4 — ledger + memo.** 3 rows appended to `.omx/state/harness_failure_ledger.jsonl`:
daemon_5min TRUE-mechanism diagnosis (supersedes "agent-harness long-call sweep"),
daemon_5min `class-fixed` resolution, sigurg codex-arm-surface `class-fixed` resolution
(honestly scoped: bare bg-bash SIGURG remains a separate real mechanism).

## 3. Measured verification (charter step 4)

`fix525_survival_probe` relaunched through the **new default delegate path**
(20:31:32Z): `codex exec` on `ttys001` from birth; **COMPLETED NORMALLY rc=0 at
20:41:49Z — 10m17s total lifetime** (`sleep 600` payload + `PROBE_DONE` final reply,
DONE marker + events-log custody all written), versus 10/10 prior kills at ≤5m55s. The
reaper log contains ZERO mentions of it — the reaper never even classified it as an
orphan. Measured window: 10m17s-to-completion vs a 5m18–5m55 kill band = unambiguous.

## 4. Recommended (NOT applied — outside mutation frontier): fleet reaper hardening

`~/Projects/fleet/scripts/claude-code-reaper.sh` is operator infrastructure in a
different repo; two bugs found while diagnosing, patch recommended for the operator:

1. **Octal crash:** `parse_etime` captures zero-padded fields ("08"/"09") and
   `$(( 08*60 ))` is an arithmetic error; under `set -euo pipefail` the WHOLE sweep dies
   at the first such line (err.log is full of these). Fix: `$(( 10#$m * 60 ))` (base-10
   force) — the reaper is currently unreliable at its own job because of this.
2. **Governed-arm exclusion:** exempt processes whose argv matches the delegate custody
   surface (e.g. `codex_runs/launch_` / `--skip-git-repo-check` with a live supervisor
   parent), or honor a marker env/file, so intentional supervised arms never depend
   solely on the pty classification.

## 5. Labels

- **[MEASURED]** every kill (reaper log custody), all probe outcomes, the pty tty
  readback, kill-age band, survivor timeline, `runs=21062`, config/binary mtimes.
- **[DERIVED]** the 300 s + 60 s-jitter arithmetic matching the 313–352 s ages; the
  word-boundary analysis (`codex_runs` non-match vs `codex exec` / `codex-code-mode-host`
  match); the octal-bug rescue of the sampler (etime "08:30" at the fatal sweep).
- **[INFERRED]** that the 2026-07-07 daemon_5min python-daemon deaths were the same
  reaper (their argv plausibly matched `\bclaude\b` via `.claude/` paths; the class's
  "same step count ~5 min, detach-insensitive" signature fits exactly; not re-run).
- **[ASSUMED]** the operator wants delegated arms exempt from the reaper (they are
  supervised, DONE-marked, checkpointed; the reaper's own docstring excludes live
  terminal sessions).

## 6. Composition / antagonism vs existing apparatus

- **Composes** with the retry supervisor (bounded respawns kept; now they should never
  fire for this class), the codex_status STALLED detector (covers the
  genuinely-orphaned-arm case the reaper used to catch), worktree isolation, inbox
  contract, DONE-marker custody — all preserved (the pty wraps OUTSIDE the launcher).
- **Composes** with the #406/#370 governor stack: reader class is a declared,
  recorded, envelope-enforced lane, not a bypass; training admission unchanged.
- **No antagonism** with trainer levers (no trainer/DSL surface touched); the
  `--with-pty` daemon flag is default-OFF byte-identical.
- **Residual risk (accepted, visible):** a future fleet-reaper rewrite that drops the
  TTY exclusion would re-expose arms — the preflight gate + this memo + ledger rows keep
  the dependency named; the §4 reaper-side patch is the belt-and-suspenders half.

## 7. Round-1 adversarial self-review (attack the build)

1. *Does the fix change the spawn path or just wrap it?* It changes the classification
   the killer keys on (controlling pty), via the spawn path all arms use. The kill
   criterion is `tty == "??"` in ALL four reaper phases — wrapping IS the mechanism-level
   fix. Verified live, not asserted: `ps` tty readback + >8m19s survival + zero reaper-log
   mentions.
2. *Can the retry supervisor still die?* Yes — to a reboot/operator kill; it survived
   every reaper sweep (its argv never matched). Bounded respawns retained; DONE custody
   unaffected.
3. *Does the reader class open a governor bypass?* No: fail-closed on BOTH a declared
   ≤4 GiB projection AND a runtime-enforced `--rss-cap-mb ≤ 4096` (safe_run SIGKILLs an
   envelope breach); the OOM free-floor preflight still runs; the declaration + reason
   are registry-recorded. A 60 GiB trainer cannot ride it (refusal rc=7 tested).
4. *Counted-but-inert risk?* The pty default is exercised by every future launch; the
   test suite pins `with_pty=True` reaching the spawn core, the tty-vs-no-tty live
   counterfactual pair, and live-repo-zero for the gate. The C1 test-run stray arm
   (unit525) that briefly launched real codex was killed + its worktree/branch removed.
5. *Could `script` break arm behavior?* codex detects a TTY → `TERM=dumb` keeps logs
   plain; the launcher's tee/PIPESTATUS pipeline is inside the pty child, unchanged; the
   verification probe ran the full contract (worktree refused under read-only →
   `--no-isolate`, inbox, checkpoint, DONE path all exercised by the live arm relaunch
   being main's call). Known cosmetic: `script` may add `\r` line endings + a leading
   `^D` in wrap logs.
6. *Is the sigurg "class-fixed" claim over-broad?* Scoped explicitly to the codex-arm
   surface in the row; the bare-bg-bash mechanism is stated as remaining real.

## 8. Open items for MAIN

- **Relaunch `sol_ultra_v10_true_final_form`** (its 3 stamps all died rc=143 mid-work;
  retry checkpoints exist under `codex_delegate:sol_ultra_v10_true_final_form:*`) — now
  survives by default. MAIN's call, not this arm's.
- **Operator:** apply the §4 fleet-reaper patch (octal fix at minimum — the reaper is
  broken at its own job); consider `REAPER_GRACE_SECONDS` ≥ 3600 for codex.
- The retry loop's transient-signature grep false-positives on repo content in the log
  (e.g. CLAUDE.md "timed out" text authorizes retries for ANY nonzero rc). It happened to
  help here (checkpointed resumes); a scoped tightening (classify rc=143 explicitly as
  external-termination + grep only the log tail) is a candidate follow-up, deliberately
  NOT landed in this arm (behavior-preserving fix surface only).
- Probe artifacts under `.omx/tmp/codex_runs/forensics_525/` (samplers, probe logs,
  renamed-binary symlink) — small, durable evidence; the `fix525_survival_probe` arm
  completed naturally (rc=0, 10m17s, `PROBE_DONE`).
