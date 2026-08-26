#!/bin/bash
# ddm_gd5 — governed continuation-window chain for the ds=32 (grid_downsample=32) TR1 run.
#
# WHY THIS EXISTS: the sealed ds=32 ticket carries --max-wall-minutes 30, so a 666-epoch
# from-scratch run needs ~17 windows (~8.5 h at the MEASURED 37-48 s/epoch). Firing those by
# hand is 17 chances to resume from the wrong checkpoint -- and 672 real ds=16 checkpoints sit
# on disk one tab-completion away (ddm_gd4 §1.3). This driver picks the resume point BY TENSOR
# SHAPE, never by filename, and fires every window through the GOVERNED launcher so G1..G5
# adjudicate each one.
#
# IT NEVER KILLS ANYTHING. A launcher REFUSE (rc=4) is information: the driver waits and retries.
# It also NEVER softens the pre-registered falsifier; it only records and applies bounded
# stop rules so the chain cannot burn 8 h confirming a plateau.
#
# score_claim=false. Every number this emits is [macOS-CPU/MLX advisory].
#
# Usage:
#   nohup bash tools/ddm_gd5_ds32_window_chain.sh > /dev/null 2>&1 & disown
#   ... or with overrides:
#   GD5_MAX_WINDOWS=4 bash tools/ddm_gd5_ds32_window_chain.sh

set -euo pipefail

REPO="${GD5_REPO:-/Users/adpena/Projects/pact}"
ROOT="${GD5_ROOT:-/Volumes/VertigoDataTier/pact/ddm_gd4_ds32_20260803}"
TICKET="${GD5_TICKET:-/Volumes/VertigoDataTier/pact/ddm_gd4_20260803/ticket_ds32_window_01.json}"
PY="${GD5_PY:-$REPO/.venv/bin/python}"
LEDGER="$ROOT/gd5_chain_ledger.jsonl"
DRIVER_LOG="$ROOT/gd5_chain_driver.log"

MAX_WINDOWS="${GD5_MAX_WINDOWS:-17}"     # 17 x ~40 ep ~= the sealed 666-epoch budget
REFUSE_RETRY_S="${GD5_REFUSE_RETRY_S:-60}"
REFUSE_MAX_TRIES="${GD5_REFUSE_MAX_TRIES:-60}"   # up to ~60 min waiting on a busy slot
POLL_S="${GD5_POLL_S:-60}"
POLL_MAX="${GD5_POLL_MAX:-90}"           # 90 min hard ceiling per 30-min window

# PRE-REGISTERED FALSIFIER (ddm_gd4 §6, `any` mask). NOT to be softened by this driver.
BAR="0.00594179"

log() { echo "[gd5-chain $(date -u +%Y-%m-%dT%H:%M:%SZ)] $*" | tee -a "$DRIVER_LOG"; }

cd "$REPO" || { echo "cannot cd $REPO"; exit 2; }
mkdir -p "$ROOT"

# --- resume point BY SHAPE, never by filename (the 672-ds16-checkpoint hazard) --------------
latest_ds32_ckpt() {
  "$PY" - "$ROOT" <<'PYEOF'
import sys, glob, os, numpy as np
root = sys.argv[1]
best = None
for p in glob.glob(os.path.join(root, "window_*", "checkpoints", "*.npz")):
    try:
        with np.load(p, allow_pickle=True) as z:
            if "param::tokens_base" not in z.files:
                continue
            if tuple(z["param::tokens_base"].shape) != (12, 16, 4):
                continue          # ds=32 geometry, MEASURED by shape -- never by filename
    except Exception:
        continue
    # Order by WRITE TIME, not by the epoch digits in the name: the stage-exit checkpoints
    # (stage_seg_trunk_tau_final.npz) carry no epoch digits yet are the most-trained state,
    # and checkpoints are written in monotone training order both within and across windows.
    key = os.path.getmtime(p)
    if best is None or key > best[0]:
        best = (key, p)
print(best[1] if best else "")
PYEOF
}

# --- last gate of a window, as a JSON line ---------------------------------------------------
last_gate_json() {
  "$PY" - "$1" <<'PYEOF'
import sys, json, os
p = os.path.join(sys.argv[1], "telemetry.jsonl")
gates, eps = [], []
if os.path.exists(p):
    for line in open(p):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("event") == "a1_gate" and r.get("realized_gate_dseg_mean") is not None:
            gates.append(r)
        elif r.get("event") == "epoch":
            eps.append(r)
out = {"gates_in_window": len(gates)}
if gates:
    g = gates[-1]
    bc = g.get("realized_gate_dseg_by_gt_class") or []
    tp = g.get("topology_per_class") or {}
    out.update({
        "epoch": int(g["epoch"]),
        "d_seg": float(g["realized_gate_dseg_mean"]),
        "a1_classification": g.get("a1_classification"),
        "by_gt_class": [float(x) for x in bc],
        # LOAD-BEARING (MEASURED 2026-08-03, ddm_gd5 §3.5): the gate does NOT always read the
        # same object. window_01 read LIVE weights ("live_ema_warmup"); every RESUMED window
        # reads the EMA SHADOW ("ema_shadow", gate_basis_mode="resumed_warm_shadow"). At
        # ema_decay=0.99992 the shadow's time constant is 166.5 epochs, so early-resume shadow
        # readings are dominated by INIT (ep49 read 0.5119 against live ep44's 0.0158 -- a 32x
        # apparent regression that is entirely a basis switch). d_seg is therefore NOT
        # comparable across differing gate_basis, and the stop rules below must not try.
        "gate_basis": g.get("gate_params"),
        "lane_betti0_realized": (tp.get("betti0_realized") or [None]*5)[1],
        "lane_components_erased": (tp.get("gt_components_erased") or [None]*5)[1],
        "tokens_bytes": g.get("tokens_bytes"),
        "total_counted_bytes": (g.get("tokens_bytes", 0) or 0) + (g.get("renderer_bytes", 0) or 0)
                               + (g.get("selector_ledger_bytes", 0) or 0),
    })
    if len(gates) > 1:
        out["d_seg_window_start"] = float(gates[0]["realized_gate_dseg_mean"])
        out["rel_drop_in_window"] = (float(gates[0]["realized_gate_dseg_mean"])
                                     - float(g["realized_gate_dseg_mean"])) / max(
                                         abs(float(gates[0]["realized_gate_dseg_mean"])), 1e-12)
# CONFOUND ALARMS (CLAUDE.md L1): liveness is not optional telemetry.
if eps:
    tail = eps[-8:]
    out["weights_stepped_all"] = all(bool(r.get("weights_stepped")) for r in tail)
    out["any_zero_ep_loss"] = any(float(r.get("ep_loss", 1.0)) == 0.0 for r in tail)
    out["last_epoch"] = int(eps[-1]["epoch"])
    out["last_t_wall_min"] = round(float(eps[-1].get("t_wall", 0.0)) / 60.0, 2)
print(json.dumps(out))
PYEOF
}

# Self-test: exercise the two helpers against the live tree and exit WITHOUT launching.
# A driver whose resume-picker has never been run is exactly the untested-guard class.
if [[ -n "${GD5_SELF_TEST:-}" ]]; then
  echo "SELF-TEST latest_ds32_ckpt -> $(latest_ds32_ckpt)"
  for d in "$ROOT"/window_*; do
    [[ -d "$d" ]] && echo "SELF-TEST last_gate_json($(basename "$d")) -> $(last_gate_json "$d")"
  done
  exit 0
fi

log "START chain: root=$ROOT ticket=$TICKET max_windows=$MAX_WINDOWS bar=$BAR"

for ((w = 2; w <= MAX_WINDOWS + 1; w++)); do
  OUT="$ROOT/$(printf 'window_%02d' "$w")"
  if [[ -f "$OUT/launch_receipt.json" ]]; then
    log "window $w already launched at $OUT -- skipping to next"
    continue
  fi

  CKPT="$(latest_ds32_ckpt)"
  if [[ -z "$CKPT" ]]; then
    log "ABORT: no ds=32 checkpoint (tokens_base shape (12,16,4)) found under $ROOT"
    exit 3
  fi
  log "window $w resume-from (SHAPE-verified ds=32): $CKPT"

  # Fire through the GOVERNED launcher. rc=4 is a REFUSE -> wait, never force.
  tries=0
  while true; do
    rc=0
    "$PY" tools/launch_tr1_run.py \
      --ticket "$TICKET" --out-dir "$OUT" --resume-from "$CKPT" \
      --allow-outdir-drift --purpose "ddm_gd5 ds32 continuation window $w" \
      >>"$DRIVER_LOG" 2>&1 || rc=$?
    if [[ $rc -eq 0 ]]; then break; fi
    if [[ $rc -eq 4 ]]; then
      tries=$((tries + 1))
      if [[ $tries -ge $REFUSE_MAX_TRIES ]]; then
        log "ABORT: launcher REFUSED $tries times for window $w (governor said no; not forcing)"
        exit 4
      fi
      log "launcher REFUSE (rc=4) for window $w, try $tries -- waiting ${REFUSE_RETRY_S}s"
      sleep "$REFUSE_RETRY_S"
      continue
    fi
    log "ABORT: launcher exited rc=$rc for window $w"
    exit "$rc"
  done

  PIDF="$OUT/run.pid"
  for ((i = 0; i < 20; i++)); do [[ -f "$PIDF" ]] && break; sleep 3; done
  PID="$(cat "$PIDF" 2>/dev/null || echo '')"
  log "window $w launched pid=${PID:-unknown}"

  # Wait for exit. Liveness anchored on the EXACT pid, never a loose pattern (m50).
  polls=0
  while [[ -n "$PID" ]] && ps -p "$PID" >/dev/null 2>&1; do
    polls=$((polls + 1))
    if [[ $polls -ge $POLL_MAX ]]; then
      log "WARN: window $w still alive after $((polls * POLL_S / 60)) min -- leaving it running, chain stops here"
      exit 5
    fi
    sleep "$POLL_S"
  done
  log "window $w exited"

  ROW="$(last_gate_json "$OUT")"
  "$PY" - "$LEDGER" "$w" "$OUT" "$BAR" "$ROW" <<'PYEOF'
import sys, json, datetime
ledger, w, out, bar, row = sys.argv[1], int(sys.argv[2]), sys.argv[3], float(sys.argv[4]), json.loads(sys.argv[5])
row.update({"window": w, "out_dir": out, "bar": bar, "arm": "ddm_gd5",
            "axis": "[macOS-CPU/MLX advisory]", "score_claim": False,
            "utc": datetime.datetime.now(datetime.UTC).isoformat()})
if "d_seg" in row:
    row["x_bar"] = row["d_seg"] / bar
with open(ledger, "a") as fh:
    fh.write(json.dumps(row, sort_keys=True) + "\n")
print(json.dumps({k: row.get(k) for k in
                  ("window", "epoch", "d_seg", "x_bar", "a1_classification",
                   "rel_drop_in_window", "lane_betti0_realized", "total_counted_bytes",
                   "weights_stepped_all", "any_zero_ep_loss")}))
PYEOF

  # --- BOUNDED STOP RULES (pre-registered; they never soften the falsifier) -----------------
  VERDICT="$("$PY" - "$LEDGER" "$BAR" <<'PYEOF'
import sys, json
rows = [json.loads(l) for l in open(sys.argv[1])]
bar = float(sys.argv[2])
rows = [r for r in rows if "d_seg" in r]
v = "CONTINUE"
why = ""
if rows:
    last = rows[-1]
    # Liveness alarms are basis-independent -- they are about the OPTIMIZER, not the readout.
    if last.get("any_zero_ep_loss"):
        v, why = "STOP", "frozen-epoch alarm: ep_loss == 0.0"
    elif last.get("weights_stepped_all") is False:
        v, why = "STOP", "liveness alarm: weights_stepped False in-window"
    else:
        # EVERYTHING BELOW COMPARES d_seg, so it may only look at rows sharing the CURRENT
        # gate_basis (see the ledger comment above: live-vs-shadow differ by ~32x here, and a
        # cross-basis comparison would either fake a plateau or fake a spectacular descent).
        same = [r for r in rows if r.get("gate_basis") == last.get("gate_basis")]
        if last["d_seg"] <= bar:
            v, why = ("BAR_MET",
                      f"d_seg {last['d_seg']:.7f} <= bar {bar} on basis {last.get('gate_basis')!r} "
                      "-- HAND-OFF ONLY: the real adjudication is an n600 byte-closed reading (§3.4)")
        elif len(same) >= 3 and all((r.get("rel_drop_in_window") or 0.0) < 0.01 for r in same[-3:]):
            v, why = "STOP", ("plateau above bar: 3 consecutive same-basis windows with <1% "
                              f"relative d_seg drop on basis {last.get('gate_basis')!r}, last "
                              f"d_seg {last['d_seg']:.7f} = {last['d_seg']/bar:.2f}x bar")
print(json.dumps({"verdict": v, "why": why}))
PYEOF
)"
  log "window $w verdict: $VERDICT"
  V="$(printf '%s' "$VERDICT" | "$PY" -c 'import sys,json; print(json.load(sys.stdin)["verdict"])')"
  if [[ "$V" == "BAR_MET" ]]; then
    log "STOP: falsifier bar met -- hand off to byte-close + pose re-solve"; exit 0
  elif [[ "$V" == "STOP" ]]; then
    log "STOP: bounded stop rule fired -- adjudicate, do not keep burning"; exit 0
  fi
done

log "chain complete: $MAX_WINDOWS windows fired"
