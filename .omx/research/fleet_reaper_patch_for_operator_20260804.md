# Fleet reaper patch — FOR OPERATOR TO APPLY (2026-08-04)

**Target:** `~/Projects/fleet/scripts/claude-code-reaper.sh` (launchd `com.vertigo.claude-code-reaper`, fires every 60s).
**Why:** MEASURED today — it SIGTERM'd every codex-arm generation at age 300–360s (receipts: `signal=TERM elapsed=335/337/337`),
and it is killing the **ChatGPT.app Codex helper processes every ~5 min despite a LIVE parent** (PPID=66195 — violating the
script's own "does not touch processes whose parent is still alive" header, via the `stdin_is_dead` secondary path).
The pact side is already cured (keeper spawn, commit `0c2a6965cd`) — this patch is belt-and-suspenders for pact **and the
only fix for the ChatGPT-app collateral**, which I cannot touch.

## The patch (one insertion, after `PS_SNAPSHOT=` is built, ~line 137)

```bash
# ── Intentional work + GUI app helpers are NOT orphans ─────────────────────
# 1) pact's canonical codex-arm spawner detaches long-running arms via
#    fork+setsid (PPID==1, no TTY) — orphan-shaped BY DESIGN. They self-mark
#    with `codex_runs/` in argv; future daemons may carry REAPER_KEEPALIVE.
#    MEASURED 2026-08-04: this reaper SIGTERM'd three working arms at age
#    335-337s, twice in one day.
# 2) /Applications/*.app/ helpers (ChatGPT's "Codex Framework") were being
#    reaped every ~5min via stdin_is_dead DESPITE a live parent — violating
#    the header's "does not touch processes whose parent is still alive".
PS_SNAPSHOT="$(echo "$PS_SNAPSHOT" | grep -vE 'codex_runs/|REAPER_KEEPALIVE|/Applications/[^ ]*\.app/' || true)"
```

## Second defect (surfaced by dry-run): parse_etime octal bug

`claude-code-reaper.sh: line 85: 09: value too great for base` — the `$(( ))` arithmetic treats
zero-padded etime fields ("09") as octal. Fix inside `parse_etime` by stripping leading zeros or
forcing base-10: `echo $(( 10#$d*86400 + 10#$h*3600 + 10#$m*60 + 10#$s ))`.

## Verification after applying (no reload needed — script is read fresh each 60s tick)

```bash
bash ~/Projects/fleet/scripts/claude-code-reaper.sh --dry-run --grace 1 --verbose
# Expect: no DRY-RUN lines for codex_runs/ arms and none for /Applications/…Codex Framework…
```

Evidence: receipts in `.omx/tmp/codex_runs/*.done` · control-survival log `.omx/tmp/reaper_probe/control_sleep.log`
· kill log `/tmp/com.vertigo.claude-code-reaper.log` · memory `arm_killer_was_the_fleet_launchd_reaper_keeper_cure_20260804`.
