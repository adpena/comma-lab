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
set -uo pipefail

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
      if mkdir "$LOCK" 2>/dev/null; then trap 'rmdir "$LOCK" 2>/dev/null' EXIT; break; fi
      sleep 2
    done
    ledger_row spawn "$LABEL" "$TS" "\"effort\":\"$EFFORT\"" "\"log\":\"$LOG\"" "\"prompt_sha256\":\"$PSHA\""
    START=$(date +%s)
    T0_DONE=""
    finish() {
      local rc=$1
      echo "rc=$rc elapsed=$(( $(date +%s) - START ))s" > "$DONE"
      ledger_row exit "$LABEL" "$TS" "\"rc\":$rc" "\"elapsed_s\":$(( $(date +%s) - START ))"
      rmdir "$LOCK" 2>/dev/null || true
    }
    trap 'finish 143' TERM; trap 'finish 130' INT
    # Release the serialize lock once the broker is up (first log line), not at process end:
    ( for _ in $(seq 1 15); do sleep 2; [ -s "$LOG" ] && { rmdir "$LOCK" 2>/dev/null; exit 0; }; done
      rmdir "$LOCK" 2>/dev/null ) &
    node "$CJS" task --write --effort "$EFFORT" "$(cat "$PROMPT_FILE")" > "$LOG" 2>&1 < /dev/null
    RC=$?
    trap - TERM INT EXIT
    finish "$RC"
    exit "$RC"
    ;;
  status)
    [ -f "$LEDGER" ] || { echo "no companion ledger yet"; exit 0; }
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
        pid = sp.get("pid"); alive = False
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
