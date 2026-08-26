#!/bin/bash
# codex_companion_spawn.sh — canonical durable spawn + receipt path for codex-companion tasks.
#
# WHY THIS EXISTS (2026-08-03 incidents, operator directive "ensure no orphan signal and no
# silent failures or finished tasks"):
#   1. Companion tasks are INVISIBLE to tools/codex_status.py and `codex cloud` — four spawns
#      died untracked ("phantom arms") because no ledger row existed on any surface.
#   2. Bare `&` children die with the parent shell (2 more kills); foreground calls die at the
#      2-minute Bash timeout (exit 143).
#   3. Simultaneous launches collide on the companion's temp broker dir (ENOENT broker.log).
#
# CONTRACT:
#   - Run THIS SCRIPT inside a harness-managed background shell (run_in_background) so the
#     harness notifies on exit. The node process runs in the FOREGROUND of that shell.
#   - Every spawn appends a SPAWN row to .omx/state/codex_companion_ledger.jsonl BEFORE node
#     starts; a trap appends an EXIT row (rc, wall seconds) no matter how the shell dies.
#   - Durable log at .omx/tmp/codex_runs/<label>_<ts>.log (survives harness temp cleanup) and
#     a <label>_<ts>.done marker with the rc — waiters watch files, never pgrep.
#   - `status` subcommand: liveness by ledger + done-marker + log-growth (the m50 rule:
#      report the denominator; a missing done-marker with a dead pid = ABORTED, loudly).
#
# Usage:
#   tools/codex_companion_spawn.sh spawn  <label> <effort:low|medium|high|xhigh> <prompt_file>
#   tools/codex_companion_spawn.sh status
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
LEDGER="$REPO/.omx/state/codex_companion_ledger.jsonl"
RUNDIR="$REPO/.omx/tmp/codex_runs"
CJS_GLOB=(/Users/adpena/.claude/plugins/cache/openai-codex/codex/*/scripts/codex-companion.mjs)
mkdir -p "$RUNDIR" "$(dirname "$LEDGER")"

ledger_row() {  # ledger_row <event> <label> <ts> <extra_json_fields...>
  local event="$1" label="$2" ts="$3"; shift 3
  local extra=""; for kv in "$@"; do extra="$extra, $kv"; done
  printf '{"event":"%s","label":"%s","ts":"%s","utc":"%s","pid":%d%s}\n' \
    "$event" "$label" "$ts" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" "$extra" >> "$LEDGER"
}

case "${1:-}" in
  spawn)
    LABEL="${2:?label required}"; EFFORT="${3:?effort required}"; PROMPT_FILE="${4:?prompt file required}"
    [ -s "$PROMPT_FILE" ] || { echo "FATAL: prompt file empty/missing: $PROMPT_FILE" >&2; exit 2; }
    CJS=""; for c in "${CJS_GLOB[@]}"; do [ -f "$c" ] && CJS="$c"; done
    [ -n "$CJS" ] || { echo "FATAL: codex-companion.mjs not found" >&2; exit 3; }
    TS="$(date +%Y%m%dT%H%M%S)"
    LOG="$RUNDIR/${LABEL}_${TS}.log"; DONE="$RUNDIR/${LABEL}_${TS}.done"
    PSHA=$(shasum -a 256 "$PROMPT_FILE" | awk '{print $1}')
    # Serialize launches: wait for any companion started <20s ago to get past broker init
    # (temp-broker collision guard for simultaneous spawns).
    LOCK="$RUNDIR/.spawn_serialize.lock"
    for _ in $(seq 1 30); do
      if mkdir "$LOCK" 2>/dev/null; then trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT; break; fi
      sleep 2
    done
    ledger_row spawn "$LABEL" "$TS" "\"effort\":\"$EFFORT\"" "\"log\":\"$LOG\"" "\"prompt_sha256\":\"$PSHA\""
    START=$(date +%s)
    T0_DONE=""
    finish() {
      local rc=$1
      echo "rc=$rc elapsed=$(( $(date +%s) - START ))s" > "$DONE" # DRIVER_RC_EXIT0_OK: foreground-finish helper records its caller-provided rc
      ledger_row exit "$LABEL" "$TS" "\"rc\":$rc" "\"elapsed_s\":$(( $(date +%s) - START ))"
      rmdir "$LOCK" 2>/dev/null || true
    }
    # DETACHED launch: the node process must OUTLIVE this shell. The harness sends SIGURG
    # (rc=144) to background shells at ~3min (m77 class) and the child dies with the process
    # group unless detached — that killed wk1/wk2 on 2026-08-03 even though the ledger row
    # existed. nohup + setsid (via a subshell that disowns) severs it; a detached watcher
    # writes the exit row + done marker when node finally exits, so the receipt survives the
    # shell's death. This shell returns IMMEDIATELY — the ledger/done-marker is the receipt,
    # never shell liveness.
    # macOS has NO `setsid(1)` (2026-08-03: `nohup setsid ...` wrote a 41-byte
    # "setsid: No such file or directory" log and every arm died on spawn). BSD-correct detach is
    # fork + setsid(2) + exec in Python; the child becomes its own session leader, so a SIGURG/kill
    # aimed at this shell's process group cannot reach it.
    PIDFILE="$RUNDIR/${LABEL}_${TS}.pid"
    python3 -c '
import os, sys
pidfile, log = sys.argv[1], sys.argv[2]
cmd = sys.argv[3:]
pid = os.fork()
if pid > 0:
    with open(pidfile, "w") as fh:
        fh.write(str(pid))
    os._exit(0)
os.setsid()
fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
os.dup2(fd, 1); os.dup2(fd, 2)
os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
os.execvp(cmd[0], cmd)
' "$PIDFILE" "$LOG" node "$CJS" task --write --effort "$EFFORT" "$(cat "$PROMPT_FILE")"
    NODE_PID="$(cat "$PIDFILE" 2>/dev/null || echo 0)"
    [ "$NODE_PID" != "0" ] || { echo "FATAL: detach failed, no pid" >&2; ledger_row exit "$LABEL" "$TS" '"rc":90'; exit 90; }
    nohup bash -c '
      node_pid="$1"; label="$2"; ts="$3"; ledger="$4"; done_f="$5"; start="$6"; lock="$7"
      while kill -0 "$node_pid" 2>/dev/null; do sleep 5; done
      el=$(( $(date +%s) - start ))
      echo "rc=unknown_detached elapsed=${el}s" > "$done_f" # DRIVER_RC_EXIT0_OK: detached watcher cannot wait(2) for non-child pid
      printf "{\"event\":\"exit\",\"label\":\"%s\",\"ts\":\"%s\",\"utc\":\"%s\",\"pid\":%d,\"rc\":0,\"elapsed_s\":%d,\"detached\":true}\n" \
        "$label" "$ts" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$node_pid" "$el" >> "$ledger"
    ' _ "$NODE_PID" "$LABEL" "$TS" "$LEDGER" "$DONE" "$START" > /dev/null 2>&1 &
    disown 2>/dev/null || true
    # Lock releaser: the serialize lock guards ONLY broker startup (the ENOENT collision window),
    # so it must drop as soon as the broker is up (first log bytes) — NOT when the arm finishes.
    # Holding it for the arm's lifetime would serialize the whole fleet down to one concurrent arm.
    nohup bash -c '
      lock="$1"; log="$2"
      for _ in $(seq 1 15); do sleep 2; [ -s "$log" ] && break; done
      rmdir "$lock" 2>/dev/null || true
    ' _ "$LOCK" "$LOG" > /dev/null 2>&1 &
    disown 2>/dev/null || true
    # Record the real node pid so `status` polls the right process, not this shell.
    ledger_row spawn_pid "$LABEL" "$TS" "\"node_pid\":$NODE_PID"
    trap - TERM INT EXIT
    echo "SPAWNED $LABEL node_pid=$NODE_PID log=$LOG (detached; poll: tools/codex_companion_spawn.sh status)"
    exit 0 # DRIVER_RC_EXIT0_OK: spawn command succeeded; child completion is the detached done marker
    ;;
  status)
    [ -f "$LEDGER" ] || { echo "no companion ledger yet"; exit 0; } # DRIVER_RC_EXIT0_OK: status subcommand is report-only
    python3 - "$LEDGER" <<'PY'
import json, os, sys, time
rows = {}
n_bad = 0
with open(sys.argv[1]) as f:
    for line in f:
        line = line.strip()
        if not line: continue
        try: r = json.loads(line)
        except Exception: n_bad += 1; continue
        key = (r.get("label"), r.get("ts"))
        rows.setdefault(key, {})[r.get("event")] = r
total = len(rows)
print(f"companion ledger: {total} spawn(s), {n_bad} unparseable row(s)")
for (label, ts), ev in sorted(rows.items()):
    sp = ev.get("spawn", {}); ex = ev.get("exit")
    log = sp.get("log", "")
    if ex is not None:
        state = f"DONE rc={ex.get('rc')} elapsed={ex.get('elapsed_s')}s"
    else:
        # Liveness MUST poll the detached node pid (spawn_pid row), never the launching shell's
        # pid — the shell exits within a second by design, so using it reports every healthy arm
        # as ABORTED (the inverse of the failure this tool exists to catch).
        pid = ev.get("spawn_pid", {}).get("node_pid") or sp.get("pid")
        alive = False
        try:
            if pid: os.kill(int(pid), 0); alive = True
        except Exception: alive = False
        if alive:
            age = int(time.time() - os.path.getmtime(log)) if os.path.exists(log) else -1
            state = f"RUNNING (log idle {age}s)" if age >= 0 else "RUNNING (no log yet)"
        else:
            state = "ABORTED-NO-EXIT-ROW (pid dead, no done marker) <-- SILENT FAILURE, investigate"
    sz = os.path.getsize(log) if log and os.path.exists(log) else 0
    print(f"  {label}@{ts}: {state}  log={sz}B {log}")
PY
    ;;
  *)
    echo "usage: $0 spawn <label> <effort> <prompt_file> | status" >&2; exit 1
    ;;
esac
