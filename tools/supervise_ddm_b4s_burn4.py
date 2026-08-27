#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# no-argparse-OK: no argv consumed — __main__ runs on pinned inputs/env; nothing for --help to discover
"""ddm_b4s burn-4 DETACHED window-loop SUPERVISOR (task #807; gc12 §3 terminal deliverable).

Burn-4 is the ladder's terminal deliverable: a bounded GOVERNED CONTINUATION from the
r1c rung-1 ep641 endpoint (the warm-QA24 base that WON the from-birth-vs-warm race —
r1c seg+rate 0.6082 beat from-birth bc1 0.686), engaging the R6 LANE-BIRTH lever
(``--class-weight-lane`` 1.0 -> 1.3, fh1 #1 ranked, never-fired duty-to-measure row)
with the tp1 v9 telemetry port ON, to attack the erased super-nucleus Lane pool
(xp1 EXACT P = 0.04401 S). Pose stays TERMINAL (#383); the burn trains SEG ONLY. NO
injection/paint anywhere (ERF-collateral law: recovery must be BORN in-loop).

The daemon+marker pattern (verbatim the r1c substrate, adapted): runs DETACHED
(nohup+disown) and supervises with ZERO agent attention. Loop:

  window_01 = the bounded RE-SMOKE (25 ep). After it ends, the ΔS SAFETY GATE:
      final_gate_dseg >= PARENT_FINAL_GATE_DSEG + RESMOKE_NOISE  => ALARM (R6 REGRESSES
          this renderer post-plateau; the R6 falsifier fires at INSTANCE) — do NOT fire
          the full burn.
      else (not regressing beyond noise)                        => extend to the full
          windows (the endpoint measures whether R6 BEATS the parent d_seg 0.00426).
  windows 02+ = FULL continuation (140 ep). Standard birth-completion/caps stop logic
      (reuse the sealed r1c rules — deviations are refusals):
        birth_completion fired  OR  extension cap  OR  8h wall  => STOP (endpoint stage)
        any confound_alarm / a1_stage_exit_refuse / crash       => ALARM (never extend)

Every decision writes a per-window receipt JSON. On STOP: the terminal endpoint stage
(n600 realized verdict via experiments/ddm_pa1r_endpoint_verdict.py + QA80 staleness
re-check + final birth key) writes ``burn4_endpoint_manifest.json`` + the ``burn4.done``
marker. On ALARM: ``burn4.ALARM`` (diagnosis is MAIN's; no endpoint stage).

Crash-resumable from disk: state is DERIVED from the run dirs/receipts every iteration
(stateless loop); a restart never double-launches (live-pid + receipt spawn-guard;
pidfile singleton). Sealed config untouched: window tickets are DSL-compiled with ONLY
{--class-weight-lane, --telemetry-v9-port, window fields, out-dir, resume-from} vs the
r1c ep641 parent (the seal tool ASSERTS it).

MAIN consumes the endpoint via ``burn4.done``. score_claim=False; every number
[macOS-CPU/MLX advisory]. Pointer 0.1910828242 [contest-CPU] UNMOVED.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = Path("/Users/adpena/Projects/pact")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# NB: after the sys.path bootstrap above -- import position is deliberate.
from tac import process_liveness  # noqa: E402

VENV_PY = str(REPO / ".venv" / "bin" / "python")
ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_b4s_20260731")
# the r1c rung-1 ep641 endpoint ticket IS the burn-4 parent (the warm-QA24 winner).
PARENT_TICKET = Path(
    "/Volumes/VertigoDataTier/pact/ddm_r1c_20260731/tickets/window_01_ticket.json")
QA91_INV = Path("/Volumes/VertigoDataTier/pact/ddm_fp1_20260731/qa91_erased_lane.json")
GT_CACHE = str(REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz")

# r1c ep641 endpoint (the parent baseline the endpoint deltas are read against; the
# manifest-verified warm-QA24 winner).
PARENT_BASELINE = {"n600_d_seg": 0.00426407708062066, "s_additive": 0.6081898657002313,
                   "tokens_bytes_smevr": 269504, "total_counted_bytes": 273004}
# r1c ep641 window_01 final realized gate d_seg (same instrument the smoke gate reads).
PARENT_FINAL_GATE_DSEG = 0.004277829770688657
PARENT_CKPT = Path("/Volumes/VertigoDataTier/pact/ddm_r1c_20260731/window_01/"
                   "checkpoints/stage_seg_trunk_tau_final.npz")

BURN4_CLASS_WEIGHT_LANE = 1.3    # fh1 R6 derived S-share tilt (Lane 0.126/0.494); raced
# --- MAIN charter amendment 2026-07-31: CONSTRAIN-AND-PROTECT layer (xp1 Lane-erosion finding) ---
# lg1 (#808) LANDED (c009a2e123 + c66acf4d79): the trainer-side protection (λ_Lane primal-dual
# constraint + born-lane protection + margin-floor emphasis) is folded into the sealed ticket via
# the three lever_lane_guard_* DSL factories (see _build_ticket). LG1_DUAL_ENGAGED=True wires the
# ROLLBACK-AND-RAISE path: on a LANE_EROSION window verdict, rollback to the window's START ckpt
# and relaunch the SAME window with --lane-guard-lambda-init = last λ + one capped step (lg1 amend
# c66acf4d79; caps-law honored), escalating to an operator ALARM at λ ≥ 1.0 (the natural full
# unit, lg1 engagement spec). The supervisor-side erosion guard (P2-inverted betti0 slope key) is
# the TOPOLOGY-trend constraint; lg1's dual is the LEVEL constraint on the same gate cadence —
# complementary, no double-counting (the guard STOPS/rolls a window; the dual PRICES erosion
# inside it). Constants-are-poison: ε derived per-window (OLS+quant); λ step/max lg1-derived.
LG1_DUAL_ENGAGED = True          # lg1 landed; λ_Lane flags are in the sealed ticket
LG1_LAMBDA_STEP = 0.1            # lg1 derive_lambda_step_cap() (caps-law; one raise per rollback)
LG1_LAMBDA_ESCALATE = 1.0        # lg1 engagement spec: ALARM-ESCALATE to operator at λ >= 1.0
LG1_LAMBDA_MAX = 5.0             # lg1 bounded safety ceiling (trainer clamps too)
LANE_GUARD_BORN_W = 0.25         # lg1 engagement-spec race {0.25,0.5} LOW end (raced-pending)
LANE_GUARD_MARGIN_FLOOR_V = 0.5  # lg1 engagement-spec race {0.5,1.0} LOW end (raced-pending)
SMOKE_EPOCHS = 25                # window_01 bounded re-smoke extent
FULL_EPOCHS = 140                # windows 02+ (gc12 §5 ~120-150 ep)
SMOKE_WALL_MINUTES = 30.0        # 25 ep x ~0.64 min/ep ~16 min + slack
FULL_WALL_MINUTES = 130.0        # 140 ep x ~0.64 min/ep ~90 min + slack
RESMOKE_NOISE = 3.0e-5           # dw1 B gate-residual std (the measured noise floor)
MAX_WINDOWS = 4                  # smoke + <=3 full windows
TOTAL_CAP_HOURS = 6.0            # hard burn-4 wall cap
EPOCHS_PER_GATE = 5              # r1c --gate-every 5
POLL_SECONDS = 120

DONE_MARKER = ROOT / "burn4.done"
ALARM_MARKER = ROOT / "burn4.ALARM"
STATE_PATH = ROOT / "supervisor_state.json"
PIDFILE = ROOT / "supervisor" / "supervisor.pid"

POINTER = "0.1910828242 [contest-CPU] UNMOVED"
HALT_EVENTS = ("confound_alarm", "a1_stage_exit_refuse")
OK_STOP_REASONS = ("epochs_complete", "max_wall_minutes")

# --- S-arithmetic (upstream/evaluate.py): S = 100*d_seg + sqrt(10*d_pose) + 25*bytes/DEN.
# The burn trains SEG ONLY (pose TERMINAL, #383) => the pose term is constant across
# windows and cancels in a WINDOW-NET delta; only the seg + rate terms move.
S_RATE_DENOMINATOR = 37_545_489.0  # upstream/evaluate.py:63 compressed_size denominator
S_SEG_COEF = 100.0
S_RATE_COEF = 25.0

# --- MAIN adjudication 2026-07-31 (ddm_b4r, successor to ddm_b4s): UNDRIV PRE-AUTHORIZATION.
# MAIN adjudicated the window_02 UNDRIV_EROSION fire CONTINUE: the reading is REAL (not
# spurious — contrast the window_01 term_domination fire, SPURIOUS_PORTED_PREDICATE), but it
# is a PRICED TRADE: fl1 (#813) prices Undrivable's TOTAL reachable non-flicker headroom at
# +0.0164 S, which does not justify halting a measured -0.02 S/window descent. To keep a
# repeat fire from dead-stopping the chain between MAIN wakes, the standing adjudication is
# encoded as a BOUNDED pre-authorization rather than a blanket mute.
#
# PROVENANCE (honesty, per constants-are-poison): UNDRIV_PREAUTH_MAX_ABS_NET is NOT a derived
# physical threshold and is not used to decide whether erosion is REAL — the DERIVED epsilon
# (t_crit*max(SE_ols,SE_quant), P2 machinery) still decides ``eroding``, unchanged. It is a
# GOVERNANCE bound: how much erosion MAIN's standing verdict covers before a FRESH adjudication
# is owed. Class-4 value, owner MAIN, re-derivation trigger = trainer-side lambda_undriv lands
# (routes to cg1 #809, calibrated by this burn's endpoint) or an endpoint re-pricing of the
# Undrivable pool. Outside the bounds => ALARM+HOLD exactly as before.
UNDRIV_PREAUTH_MAX_ABS_NET = 10   # |net betti0_realized delta| components over the window
UNDRIV_PREAUTH_REQUIRES_NET_DS_NEGATIVE = True  # the window must still be net score-lowering


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _log(msg: str) -> None:
    print(f"[{_now()}] {msg}", flush=True)


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def _pid_alive(pid: int) -> bool:
    """Delegates to the canonical tri-state read (``tac.process_liveness``).

    Behaviour CHANGED in three ways (undocumented drift in the old local copy):

    * ``PermissionError`` was explicitly DEAD, now ALIVE.
    * A zombie was ALIVE forever, now DEAD -- a finished-but-unreaped window
      worker used to keep this supervisor waiting on a process that can never
      run again, which is the silent-stall shape this file already fights.
    * ``pid <= 0`` had NO guard: ``os.kill(0, 0)`` targets our OWN process
      group and SUCCEEDS, so a ``launch_manifest.json`` carrying ``"pid": 0``
      read the window as RUNNING forever.  It is now UNREADABLE -> False.
    """
    return process_liveness.pid_state(pid) == process_liveness.ALIVE


def _window_dir(n: int) -> Path:
    return ROOT / f"window_{n:02d}"


def _existing_windows() -> list[Path]:
    # EXACT window_NN dirs only: retired/rejected debris (window_01_preamendment_*,
    # rollback-retired copies) must never be picked up as the current window (the
    # 2026-07-31 false WINDOW_CRASHED_NO_RECEIPT lesson — glob matched a retired dir).
    return sorted(p for p in ROOT.glob("window_*")
                  if p.is_dir() and re.fullmatch(r"window_\d{2}", p.name))


def _window_pid(wdir: Path) -> int | None:
    mf = wdir / "launch_manifest.json"
    if not mf.is_file():
        return None
    try:
        return int(json.loads(mf.read_text()).get("pid"))
    except Exception:
        return None


def _window_done(wdir: Path) -> bool:
    return (wdir / "tr1_window_receipt.json").is_file()


def _scan_halt_events(wdir: Path) -> list[dict]:
    tele = wdir / "telemetry.jsonl"
    hits: list[dict] = []
    if not tele.is_file():
        return hits
    for line in tele.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("event") in HALT_EVENTS:
            hits.append({k: row.get(k) for k in ("event", "kind", "epoch")})
    return hits


def _ckpt_epoch(ckpt: Path) -> int:
    import numpy as np
    return int(np.load(ckpt, allow_pickle=False)["meta::epoch"][0])


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_ticket(epochs: int, wall_minutes: float, out_dir: Path, resume_from: Path,
                  out_ticket: Path, lambda_init: float = 0.0) -> dict:
    """DSL-compile the burn-4 window ticket: r1c ep641 parent levers VERBATIM, swapping
    ONLY the seg-physics --class-weight-lane (-> 1.3), adding telemetry_v9_port('on'),
    the lg1 (#808) CONSTRAIN-AND-PROTECT trio (lambda dual w/ LawRef-custodied budget +
    born W=0.25 + margin-floor V=0.5 — engagement-spec low-end, raced-pending), the
    optional --lane-guard-lambda-init (rollback+raise relaunch), and the window lever.
    Fail-closed never-invent-flags + ticket_hash + no-stray-store_true-token check."""
    sys.path.insert(0, str(REPO / "src"))
    from tac.witness_dsl.curriculum_dsl import Lever
    from tac.witness_dsl.spec_tr1_renderer_20260728 import (
        TR1RendererProgramV1, lever_lane_guard_born, lever_lane_guard_lambda,
        lever_lane_guard_margin_floor, lever_seg_physics, lever_telemetry_v9_port,
        lever_window)

    parent = json.loads(PARENT_TICKET.read_text())
    levers: list[Lever] = []
    saw_tel = False
    for d in parent["levers"]:
        ov = d["overrides"]
        if "--class-weight-lane" in ov:
            levers.append(lever_seg_physics(
                form_start=ov.get("--seg-form-start", "ce"),
                w_seg=float(ov.get("--w-seg", "100.0")),
                class_weight_lane=BURN4_CLASS_WEIGHT_LANE,
                margin_target=float(ov.get("--margin-target", "1.0"))))
        elif "--epochs" in ov:
            levers.append(lever_window(epochs, wall_minutes, batch_pairs=8, lr=2e-3))
        elif "--telemetry-v9-port" in ov:
            levers.append(lever_telemetry_v9_port("on"))
            saw_tel = True
        elif "--lane-guard" in ov:
            continue  # parent never carries lane-guard; re-added canonically below
        else:
            levers.append(Lever(name=d["name"], overrides=dict(ov), notes=d.get("notes", "")))
    if not saw_tel:
        levers.append(lever_telemetry_v9_port("on"))
    levers.append(lever_lane_guard_lambda())          # 0.0 sentinels => derived/custodied
    levers.append(lever_lane_guard_born(LANE_GUARD_BORN_W))
    levers.append(lever_lane_guard_margin_floor(LANE_GUARD_MARGIN_FLOOR_V))
    if lambda_init > 0.0:
        levers.append(Lever(
            name="tr1_lane_guard_lambda_init",
            overrides={"--lane-guard-lambda-init": str(lambda_init)},
            notes="b4s rollback+raise-lambda relaunch (lg1 amend c66acf4d79): warm-start "
                  "the dual at last lambda + one capped step; clamped trainer-side"))

    prog = TR1RendererProgramV1(levers=tuple(levers), num_pairs=600, out_dir=str(out_dir),
                                seed=0, gt_cache=GT_CACHE, resume_from=str(resume_from),
                                full_confirm=True)
    ticket = prog.sealed_ticket()
    if "True" in ticket["argv"] or "False" in ticket["argv"]:
        raise RuntimeError(f"stray store_true token in compiled argv: {ticket['argv']}")
    _atomic_write(out_ticket, ticket)
    return ticket


def _p2_key(wdir: Path) -> dict:
    out_json = wdir / "birth_completion_gate.json"
    r = subprocess.run(
        [VENV_PY, "tools/run_ddm_lp2_birth_completion_key.py",
         "--telemetry", str(wdir / "telemetry.jsonl"),
         "--qa91-inventory", str(QA91_INV),
         "--epochs-per-gate", str(EPOCHS_PER_GATE),
         "--output-json", str(out_json)],
        cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"P2 key FAIL-CLOSED rc={r.returncode}: {(r.stderr or r.stdout).strip()[:400]}")
    return json.loads(out_json.read_text())


def _is_resmoke(wdir: Path) -> bool:
    return wdir.name == "window_01"


def _launch_window(n: int, resume_ckpt: Path, lambda_init: float = 0.0) -> dict:
    wdir = _window_dir(n)
    epochs_add = SMOKE_EPOCHS if n == 1 else FULL_EPOCHS
    wall = SMOKE_WALL_MINUTES if n == 1 else FULL_WALL_MINUTES
    epochs = _ckpt_epoch(resume_ckpt) + epochs_add
    suffix = f"_lam{lambda_init:.1f}" if lambda_init > 0.0 else ""
    ticket_path = ROOT / "tickets" / f"window_{n:02d}{suffix}_ticket.json"
    ticket = _build_ticket(epochs, wall, wdir, resume_ckpt, ticket_path,
                           lambda_init=lambda_init)
    purpose = (f"ddm_b4s burn-4 window_{n:02d} "
               f"{'re-smoke' if n == 1 else 'full'} (R6=1.3, lg1-protected, telemetry on"
               + (f", lambda_init={lambda_init:.1f}" if lambda_init > 0.0 else "") + ")")
    r = subprocess.run(
        [VENV_PY, "tools/launch_tr1_run.py",
         "--ticket", str(ticket_path), "--out-dir", str(wdir),
         "--resume-from", str(resume_ckpt), "--purpose", purpose],
        cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"governed launcher REFUSED window_{n:02d} rc={r.returncode}: "
            f"{(r.stderr or r.stdout).strip()[:600]}")
    _log(f"window_{n:02d} launched (epochs->{epochs}, ticket {ticket['ticket_hash'][:12]}"
         + (f", lambda_init={lambda_init:.1f}" if lambda_init > 0.0 else "") + ")")
    return {"window": n, "epochs_arg": epochs, "ticket_hash": ticket["ticket_hash"],
            "ticket_path": str(ticket_path), "lambda_init": lambda_init}


def _last_lane_guard_lambda(wdir: Path) -> float | None:
    """Last ``lambda_lane`` from the window's ``event=lane_guard`` telemetry rows (lg1
    engagement spec: one row per a1 gate). None if the window emitted none."""
    tele = wdir / "telemetry.jsonl"
    lam: float | None = None
    if not tele.is_file():
        return None
    for line in tele.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if row.get("event") == "lane_guard" and row.get("lambda_lane") is not None:
            try:
                lam = float(row["lambda_lane"])
            except (TypeError, ValueError):
                continue
    return lam


T0_PATH = ROOT / "burn4_t0.json"


def _ensure_t0() -> None:
    """Persist the burn's t0 ONCE (window_01's launch time if present, else now). The
    rollback path may RETIRE window_01, so the wall-cap clock must not depend on it."""
    if T0_PATH.is_file():
        return
    rec = _window_dir(1) / "launch_receipt.json"
    if rec.is_file():
        t0 = json.loads(rec.read_text())["launched_utc"]
    else:
        t0 = _now()
    _atomic_write(T0_PATH, {"schema": "ddm_b4s_burn4_t0.v1", "launched_utc": t0,
                            "pointer": POINTER, "score_claim": False})


def _elapsed_hours() -> float:
    _ensure_t0()
    t0 = datetime.strptime(json.loads(T0_PATH.read_text())["launched_utc"],
                           "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    return (datetime.now(UTC) - t0).total_seconds() / 3600.0


def _cap_would_be_exceeded() -> tuple[bool, float]:
    """(would_exceed, elapsed_hours) for STARTING one more FULL window now.

    Projected against ``FULL_WALL_MINUTES`` (the trainer's OWN wall cap) rather than the measured
    window duration: a window MAY legally run to its wall cap, so only the wall cap is a
    GUARANTEED-completion bound. Binds EVERY launch path (extend AND the lane-erosion
    rollback-and-raise relaunch) — a cap that only one caller honors is not a cap."""
    e = _elapsed_hours()
    return (e + (FULL_WALL_MINUTES / 60.0) > TOTAL_CAP_HOURS, e)


def _heartbeat(phase: str, window: int) -> None:
    _atomic_write(STATE_PATH, {"ts_utc": _now(), "phase": phase, "current_window": window,
                               "supervisor_pid": os.getpid(), "pointer": POINTER,
                               "score_claim": False})


def _alarm(reason: str, details: dict) -> None:
    _atomic_write(ALARM_MARKER, {"schema": "ddm_b4s_burn4_alarm.v1", "ts_utc": _now(),
                                 "reason": reason, "details": details,
                                 "action_owed": "MAIN diagnose + report (gc11 W4)",
                                 "pointer": POINTER, "score_claim": False})
    _log(f"ALARM: {reason} -> {ALARM_MARKER}")


def _qa80_staleness_recheck(wdir: Path) -> dict:
    cfg = json.loads((wdir / "tr1_config.json").read_text())
    consumers = {"token_quant_coupling_field": cfg.get("token_quant_coupling_field"),
                 "token_quant_margin_coupling": cfg.get("token_quant_margin_coupling"),
                 "delta_sparsity_weight": cfg.get("delta_sparsity_weight"),
                 "renderer_head_mode": cfg.get("renderer_head_mode")}
    fresh = (consumers["token_quant_coupling_field"] in (None, "") and
             consumers["token_quant_margin_coupling"] == "off" and
             float(consumers["delta_sparsity_weight"] or 0.0) == 0.0 and
             consumers["renderer_head_mode"] == "rgb")
    return {"verdict": ("NOT_CONSUMED_FRESH_BY_CONSTRUCTION" if fresh
                        else "CONSUMER_ACTIVE_FLAG_FOR_MAIN"),
            "consumers": consumers,
            "note": "margin_weighted_loss uses LIVE realized-SegNet margins "
                    "(_live_margin_weight, per-step, stop-grad) — fresh by construction"}


def _endpoint_stage(final_dir: Path, stop_reason: str, decisions: list[dict]) -> None:
    _heartbeat("endpoint_stage", int(final_dir.name.split("_")[1]))
    ckpt = final_dir / "checkpoints" / "stage_seg_trunk_tau_final.npz"
    manifest: dict = {
        "schema": "ddm_b4s_burn4_endpoint_manifest.v1", "ts_utc": _now(),
        "stop_reason": stop_reason, "final_window_dir": str(final_dir),
        "final_ckpt": str(ckpt),
        "final_ckpt_sha256": _sha256(ckpt) if ckpt.is_file() else None,
        "final_ckpt_meta_epoch": _ckpt_epoch(ckpt) if ckpt.is_file() else None,
        "windows": decisions, "parent_baseline": PARENT_BASELINE,
        "class_weight_lane": BURN4_CLASS_WEIGHT_LANE,
        "lane_guard": {
            "dual_engaged": LG1_DUAL_ENGAGED,
            "born_weight": LANE_GUARD_BORN_W,
            "margin_floor_weight": LANE_GUARD_MARGIN_FLOOR_V,
            "lambda_step": LG1_LAMBDA_STEP, "lambda_escalate": LG1_LAMBDA_ESCALATE,
            "lambda_max": LG1_LAMBDA_MAX,
            "rejected_windows": sorted(p.name for p in (ROOT / "rejected").glob("window_*"))
            if (ROOT / "rejected").is_dir() else [],
            "lambda_trajectory": [{"window": w.name,
                                   "last_lambda_lane": _last_lane_guard_lambda(w)}
                                  for w in _existing_windows()],
        },
        "window_history": {
            "superseded": sorted(p.name for p in
                                 ROOT.glob("window_*_decision_supersession.json")),
            "undriv_preauth": sorted(p.name for p in ROOT.glob("window_*_undriv_preauth.json")),
            "alarms_resolved": sorted(p.name for p in (ROOT / "alarms_resolved").glob("*.json"))
            if (ROOT / "alarms_resolved").is_dir() else [],
        },
        "window_net_delta_s_chain": [
            _window_net_delta_s(decisions[:i + 1]) for i in range(len(decisions))],
        "r6_falsifier": "beat parent n600 d_seg 0.00426407708 at matched compute; "
                        "else R6 (class-weight-lane LANE-BIRTH tilt) closes at INSTANCE",
        "pointer": POINTER, "score_claim": False,
        "evidence_axis": "[macOS-CPU/MLX advisory]",
    }
    try:
        r = subprocess.run(
            [VENV_PY, "experiments/ddm_pa1r_endpoint_verdict.py", str(final_dir),
             "--chunk", "100"],
            cwd=REPO, capture_output=True, text=True, timeout=3600,
            env={**os.environ, "PYTHONPATH": f"{REPO}/src:{REPO}:{REPO}/upstream"})
        if r.returncode == 0:
            v = json.loads(r.stdout)
            v["delta_vs_parent"] = {
                "d_seg": v["n600_d_seg"] - PARENT_BASELINE["n600_d_seg"],
                "s_additive": v["s_additive"] - PARENT_BASELINE["s_additive"],
                "total_counted_bytes": (v["total_counted_bytes"]
                                        - PARENT_BASELINE["total_counted_bytes"])}
            v["r6_verdict"] = ("R6_PAYS" if v["n600_d_seg"] < PARENT_BASELINE["n600_d_seg"]
                               else "R6_CLOSES_AT_INSTANCE")
            manifest["endpoint_verdict"] = v
        else:
            manifest["endpoint_verdict_error"] = (r.stderr or r.stdout).strip()[-800:]
            manifest["endpoint_verdict_owed"] = "MAIN: re-run ddm_pa1r_endpoint_verdict.py"
    except Exception as exc:
        manifest["endpoint_verdict_error"] = repr(exc)
        manifest["endpoint_verdict_owed"] = "MAIN: re-run ddm_pa1r_endpoint_verdict.py"
    key = decisions[-1].get("birth_key", {}) if decisions else {}
    manifest["p_remeasure"] = {
        "erased_count_endpoint": key.get("erased_count"),
        "above_nucleus_erased_estimate": key.get("above_nucleus_erased_estimate"),
        "label": "DERIVED-ESTIMATE (tr1 4-conn betti0 x QA91 8-conn area frac)",
        "exact_owed_to_main": "QA92 base-pass method on this endpoint render "
                              "(experiments/ddm_qa92_carrier_discriminator.py) -> P in S units",
        "parent_exact_P_S_units_xp1": 0.04401,
    }
    manifest["qa80_staleness_recheck"] = _qa80_staleness_recheck(final_dir)
    # --- ENDPOINT EXTRAS (ddm_b4r): the gc13 R1-bundle measurements MAIN consumes beyond the
    # scalar verdict — per-class n600 d_seg (all 5 classes), the 5x5 class-PAIR flip matrix,
    # per-class descent rates across the accepted windows, and the Undrivable pool priced in S
    # across the whole burn. Runs in the free scorer slot (nothing else is live at endpoint).
    # FAIL-SOFT by construction: a missing/failing extras tool records the debt and NEVER
    # invalidates or blocks the manifest + done marker (the chain's payload survives).
    extras_tool = REPO / "experiments" / "ddm_b4r_endpoint_extras.py"
    try:
        if not extras_tool.is_file():
            raise FileNotFoundError(str(extras_tool))
        r2 = subprocess.run(
            [VENV_PY, str(extras_tool), str(final_dir), "--root", str(ROOT),
             "--parent-ckpt", str(PARENT_CKPT), "--chunk", "100"],
            cwd=REPO, capture_output=True, text=True, timeout=3600,
            env={**os.environ, "PYTHONPATH": f"{REPO}/src:{REPO}:{REPO}/upstream"})
        if r2.returncode == 0:
            manifest["endpoint_extras"] = json.loads(r2.stdout)
        else:
            manifest["endpoint_extras_error"] = (r2.stderr or r2.stdout).strip()[-800:]
            manifest["endpoint_extras_owed"] = f"MAIN: re-run {extras_tool.name}"
    except Exception as exc:
        manifest["endpoint_extras_error"] = repr(exc)
        manifest["endpoint_extras_owed"] = f"MAIN: re-run {extras_tool.name}"
    _atomic_write(ROOT / "burn4_endpoint_manifest.json", manifest)
    _atomic_write(DONE_MARKER, {"schema": "ddm_b4s_burn4_done.v1", "ts_utc": _now(),
                                "endpoint_manifest": str(ROOT / "burn4_endpoint_manifest.json"),
                                "stop_reason": stop_reason, "pointer": POINTER,
                                "score_claim": False})
    _log(f"STOP ({stop_reason}) -> {DONE_MARKER}")


def _decision_receipts() -> list[dict]:
    """Window decision receipts with APPEND-ONLY SUPERSESSION overlay: a companion
    ``window_NN_decision_supersession.json`` (written by an adjudication, e.g. the
    2026-07-31 MAIN false-positive verdict on the ported term_domination predicate)
    overlays ONLY its declared ``supersedes_fields`` onto the original receipt at READ
    time — the original decision file is never mutated (append-only law)."""
    out = []
    for p in sorted(ROOT.glob("window_*_decision.json")):
        if not re.fullmatch(r"window_\d{2}_decision\.json", p.name):
            continue  # companions/debris are overlays, not receipts
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        sup = p.with_name(p.name[:-len(".json")] + "_supersession.json")
        if sup.is_file():
            try:
                s = json.loads(sup.read_text())
                for k in s.get("supersedes_fields", []):
                    if k in s:
                        d[k] = s[k]
                d["superseded_by"] = sup.name
            except Exception:
                pass  # a malformed supersession never hides the original receipt
        out.append(d)
    return out


def _window_net_delta_s(decisions: list[dict]) -> dict:
    """WINDOW-NET ΔS of the LAST window vs the prior ACCEPTED state.

    ``100*(d_seg_end - d_seg_start) + 25*(bytes_end - bytes_start)/37_545_489``, read off the
    per-window decision receipts (the same realized-gate instrument both ends). The prior
    accepted state is the previous window's decision receipt, or the r1c ep641 parent baseline
    for the chain's first window. The pose term cancels (seg-only burn, pose TERMINAL #383), so
    this IS the window's full score-arithmetic movement on the two axes the burn touches.

    Returns the decomposition (never a bare composite — decompose-every-headline). ``net_delta_s``
    is None when either endpoint is missing; a None NEVER satisfies a pre-authorization bound.
    [macOS-CPU/MLX advisory]; score_claim False."""
    if not decisions:
        return {"net_delta_s": None, "note": "no decision receipts"}
    cur = decisions[-1]
    d1, b1 = cur.get("final_gate_dseg"), cur.get("total_counted_bytes")
    if len(decisions) >= 2:
        prev = decisions[-2]
        d0, b0 = prev.get("final_gate_dseg"), prev.get("total_counted_bytes")
        basis = f"prior window receipt ({prev.get('window')})"
    else:
        d0, b0 = PARENT_FINAL_GATE_DSEG, PARENT_BASELINE["total_counted_bytes"]
        basis = "r1c ep641 parent baseline"
    if any(v is None for v in (d0, d1, b0, b1)):
        return {"net_delta_s": None, "basis": basis,
                "note": "missing d_seg/bytes endpoint on one side"}
    seg_term = S_SEG_COEF * (float(d1) - float(d0))
    rate_term = S_RATE_COEF * (int(b1) - int(b0)) / S_RATE_DENOMINATOR
    return {"net_delta_s": seg_term + rate_term, "seg_term_delta_s": seg_term,
            "rate_term_delta_s": rate_term, "d_seg_start": d0, "d_seg_end": d1,
            "bytes_start": b0, "bytes_end": b1, "basis": basis,
            "formula": "100*d_seg + 25*bytes/37545489 (pose cancels: seg-only burn)"}


UNDRIV_CLASS_INDEX = 2  # comma10k CANONICAL order [Road, Lane, Undrivable, Movable, MyCar]


def _class_erosion_verdict(wdir: Path, class_index: int, class_name: str) -> dict:
    """gc13 (71659bd3d7) UNDRIV watch: 'Lane = the one measured need' is FALSIFIED by
    receipt — Undrivable eroded +0.00204 S over the same unprotected ep499→641 window
    (qa92 ep499 vs xp1 ep641, both n600-exact; INSTANCE scope), MORE than Lane's
    +0.00151 — and the R6+λ_Lane config protects Lane only.  Same derived-ε machinery
    as the Lane key (betti0_realized[class] OLS slope over the last window gates;
    ε = t_crit·max(SE_ols, SE_quant); eroding ⇔ slope < −ε; REUSES the P2 module's
    fit + constants verbatim — no hand-set threshold).  Consumed as ALARM+HOLD ONLY
    (no auto-rollback: there is no trainer-side λ_undriv to raise; MAIN adjudicates on
    fire).  Never raises — a malformed/short window returns eroding=False with a note
    (the watch must not kill the loop); daemon-decision-logic only, sealed config
    untouched."""
    try:
        sys.path.insert(0, str(REPO / "src"))
        import numpy as np
        from scipy import stats as sci_stats
        from tac.optimization.ddm_lp2_birth_completion import (
            _QUANT_ROUNDING_VAR, DEFAULT_ALPHA_ONE_SIDED, DEFAULT_WINDOW_GATES,
            MIN_WINDOW_POINTS, _ols_slope_fit)

        tele = wdir / "telemetry.jsonl"
        pts: dict[int, int] = {}
        for line in tele.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            topo = row.get("topology_per_class")
            if not isinstance(topo, dict) or "epoch" not in row:
                continue
            realized = topo.get("betti0_realized")
            if isinstance(realized, list) and len(realized) > class_index:
                pts[int(row["epoch"])] = int(realized[class_index])
        ordered = sorted(pts.items())[-int(DEFAULT_WINDOW_GATES):]
        if len(ordered) < int(MIN_WINDOW_POINTS):
            return {"eroding": False, "class_name": class_name,
                    "note": f"insufficient topology points ({len(ordered)})"}
        epochs = np.array([e for e, _ in ordered], dtype=np.float64)
        counts = np.array([c for _, c in ordered], dtype=np.float64)
        gates = epochs / float(EPOCHS_PER_GATE)
        slope, se_ols, s_xx, _ssr, dof = _ols_slope_fit(gates, counts)
        se_quant = float(np.sqrt(float(_QUANT_ROUNDING_VAR) / s_xx))
        se = max(se_ols, se_quant)
        eps = float(sci_stats.t.ppf(1.0 - float(DEFAULT_ALPHA_ONE_SIDED), dof)) * se
        return {
            "eroding": bool(slope < -abs(eps)), "class_name": class_name,
            "class_index": int(class_index),
            "slope_comp_per_gate": float(slope), "epsilon_comp_per_gate": float(eps),
            "net_betti0_realized_delta": int(ordered[-1][1] - ordered[0][1]),
            "n_points": len(ordered),
            "rule": "eroding <=> betti0_realized OLS slope < -epsilon; epsilon = "
                    "t_crit*max(SE_ols, SE_quant) (P2 machinery reused verbatim; "
                    "gc13 receipt: Undriv +0.00204 S unprotected ep499->641)",
        }
    except Exception as exc:  # the watch must never kill the supervisor loop
        return {"eroding": False, "class_name": class_name,
                "note": f"watch error (degraded, non-fatal): {exc!r}"}


def _resmoke_gate(decision: dict) -> str:
    """The ΔS SAFETY GATE after window_01: not-regressing-beyond-noise => PROCEED; else ALARM.
    Returns 'PROCEED' or 'REGRESSED'."""
    dseg = decision.get("final_gate_dseg")
    if dseg is None:
        return "PROCEED"  # missing gate metric -> defer to the full-window verdict (never block)
    return "REGRESSED" if dseg >= PARENT_FINAL_GATE_DSEG + RESMOKE_NOISE else "PROCEED"


def _lane_erosion_verdict(key: dict) -> dict:
    """MAIN items #4 (LANE-EROSION EVENT KEY) + #5 (WINDOW REALIZED ACCEPTANCE), supervisor-side.

    The P2 birth key INVERTED: the same OLS fit of Lane ``betti0_realized`` vs gate over the window,
    but the erosion test is ``slope < -epsilon`` (Lane components net ERASED, a significant NEGATIVE
    trend) — vs the birth key's ``slope <= epsilon`` (flattening). epsilon is DERIVED per-window from
    the SAME OLS+quant machinery (constants-are-poison: no hand-set threshold). ``net_delta`` (last −
    first betti0_realized_lane over the window) is the item-#5 realized-acceptance signal reported
    alongside. ``eroding`` True ⇒ the window WORSENED Lane beyond the derived noise band ⇒ do NOT
    extend; REJECT (rollback target = the window's START ckpt = the last accepted state)."""
    fit = key.get("fit") or {}
    slope = fit.get("slope_comp_per_gate")
    eps = fit.get("epsilon_comp_per_gate")
    trend = list(key.get("window_betti0_realized") or ())
    net_delta = (trend[-1] - trend[0]) if len(trend) >= 2 else None
    eroding = (slope is not None and eps is not None and slope < -abs(eps))
    return {
        "eroding": bool(eroding),
        "slope_comp_per_gate": slope,
        "epsilon_comp_per_gate": eps,
        "net_betti0_realized_lane_delta": net_delta,
        "window_betti0_realized_lane": trend,
        "rule": "eroding <=> Lane betti0_realized OLS slope < -epsilon (significant net erasure over "
                "the window); epsilon DERIVED (t_crit*max(SE_ols,SE_quant), same machinery as the P2 "
                "birth key); no hand-set threshold (constants-are-poison)",
    }


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    if PIDFILE.is_file():
        try:
            other = int(PIDFILE.read_text().strip())
        except ValueError:
            other = -1
        if other > 0 and _pid_alive(other):
            _log(f"another supervisor is live (pid {other}); exiting")
            return 0
    PIDFILE.parent.mkdir(parents=True, exist_ok=True)
    PIDFILE.write_text(str(os.getpid()))
    _log(f"burn-4 supervisor start pid {os.getpid()}")

    # bootstrap: if no window exists yet, launch window_01 (the re-smoke) from the r1c
    # ep641 parent ckpt (the daemon owns the first launch so the fire is single-command).
    if not _existing_windows():
        try:
            _launch_window(1, PARENT_CKPT)
        except RuntimeError as exc:
            _alarm("WINDOW_01_LAUNCH_REFUSED", {"error": str(exc)})
            return 0
    _ensure_t0()  # persist the wall-cap clock ONCE (rollback may retire window_01)

    while True:
        if DONE_MARKER.is_file() or ALARM_MARKER.is_file():
            _log("terminal marker present; exiting")
            return 0
        wins = _existing_windows()
        if not wins:
            _alarm("NO_WINDOWS_FOUND", {"root": str(ROOT)})
            return 0
        cur = wins[-1]
        n = int(cur.name.split("_")[1])
        if not _window_done(cur):
            pid = _window_pid(cur)
            if pid is not None and _pid_alive(pid):
                _heartbeat("waiting_window", n)
                time.sleep(POLL_SECONDS)
                continue
            _alarm("WINDOW_CRASHED_NO_RECEIPT",
                   {"window": str(cur), "pid": pid, "halt_events": _scan_halt_events(cur)})
            return 0
        pid = _window_pid(cur)
        if pid is not None and _pid_alive(pid):
            _heartbeat("window_done_awaiting_exit", n)
            time.sleep(15)
            continue
        # evaluate + decide (idempotent per-window decision receipt)
        dec_path = ROOT / f"{cur.name}_decision.json"
        if not dec_path.is_file():
            receipt = json.loads((cur / "tr1_window_receipt.json").read_text())
            halt = _scan_halt_events(cur)
            try:
                key = _p2_key(cur)
            except RuntimeError as exc:
                _alarm("P2_KEY_FAIL_CLOSED", {"window": str(cur), "error": str(exc)})
                return 0
            fg = receipt.get("final_gate", {})
            decision = {
                "schema": "ddm_b4s_burn4_window_decision.v1", "ts_utc": _now(),
                "window": n, "window_dir": str(cur), "is_resmoke": _is_resmoke(cur),
                "stop_reason": receipt.get("stop_reason"),
                "epochs_ran": receipt.get("epochs_ran"),
                "final_gate_epoch": fg.get("epoch"),
                "final_gate_dseg": fg.get("realized_gate_dseg_mean"),
                "final_gate_a1": fg.get("a1_classification"),
                "full_confirm_dseg": (receipt.get("full_confirm") or {}).get(
                    "realized_dseg_mean"),
                "total_counted_bytes": fg.get("total_counted_bytes"),
                "parent_final_gate_dseg": PARENT_FINAL_GATE_DSEG,
                "delta_gate_dseg_vs_parent": (
                    (fg.get("realized_gate_dseg_mean") - PARENT_FINAL_GATE_DSEG)
                    if fg.get("realized_gate_dseg_mean") is not None else None),
                "halt_events": halt,
                "birth_key": {k: key.get(k) for k in
                              ("fired", "slope_le_epsilon",
                               "above_nucleus_erasure_persists", "erased_count",
                               "above_nucleus_erased_estimate", "betti0_realized_endpoint")},
                "birth_key_fit": key.get("fit"),
                "lane_erosion": _lane_erosion_verdict(key),  # MAIN items #4/#5 (protection)
                "undriv_erosion": _class_erosion_verdict(    # gc13: the unwatched eroder
                    cur, UNDRIV_CLASS_INDEX, "Undrivable"),
                "elapsed_hours": round(_elapsed_hours(), 3),
                "pointer": POINTER, "score_claim": False,
            }
            _atomic_write(dec_path, decision)
            _log(f"decision written for {cur.name}: resmoke={_is_resmoke(cur)} "
                 f"fired={key.get('fired')} stop={receipt.get('stop_reason')} halt={len(halt)}")
        decisions = _decision_receipts()
        last = decisions[-1]
        if last["halt_events"]:
            _alarm("HALT_EVENTS_IN_WINDOW",
                   {"window": last["window_dir"], "halt_events": last["halt_events"]})
            return 0
        if last["stop_reason"] not in OK_STOP_REASONS:
            _alarm("UNEXPECTED_STOP_REASON",
                   {"window": last["window_dir"], "stop_reason": last["stop_reason"]})
            return 0
        # --- MAIN protection: LANE-EROSION GUARD (items #4/#5; every window) ---------
        # A window whose Lane betti0_realized OLS slope is significantly NEGATIVE (< -epsilon)
        # WORSENED the protected Lane pool beyond the derived noise band. REJECT it: do NOT extend
        # from its (eroded) end ckpt; the last ACCEPTED state is its START ckpt (the prior window's
        # end, or the parent for window_01).
        le = last.get("lane_erosion") or {}
        if le.get("eroding"):
            n_cur = int(Path(last["window_dir"]).name.split("_")[1])
            prior = (_window_dir(n_cur - 1) / "checkpoints" / "stage_seg_trunk_tau_final.npz"
                     if n_cur > 1 else PARENT_CKPT)
            details = {"window": last["window_dir"],
                       "slope_comp_per_gate": le.get("slope_comp_per_gate"),
                       "epsilon_comp_per_gate": le.get("epsilon_comp_per_gate"),
                       "net_betti0_realized_lane_delta": le.get("net_betti0_realized_lane_delta"),
                       "rollback_target_ckpt": str(prior),
                       "lg1_dual_engaged": LG1_DUAL_ENGAGED}
            if LG1_DUAL_ENGAGED:
                # ROLLBACK-AND-RAISE (lg1 amend c66acf4d79): rollback to the START ckpt and
                # relaunch the SAME window with --lane-guard-lambda-init = last λ + one capped
                # step (caps-law). Escalate to an operator ALARM at λ >= 1.0 (engagement spec).
                lam = _last_lane_guard_lambda(Path(last["window_dir"]))
                new_init = min((lam if lam is not None else 0.0) + LG1_LAMBDA_STEP,
                               LG1_LAMBDA_MAX)
                details.update({"last_lambda_lane": lam, "raise_lambda_init_to": new_init})
                if new_init >= LG1_LAMBDA_ESCALATE:
                    _alarm("LANE_EROSION_LAMBDA_ESCALATE",
                           {**details, "reading": "Lane still erodes with the dual at "
                                                  f"lambda>={LG1_LAMBDA_ESCALATE} (the natural "
                                                  "full unit) — the loss-weight dual form is "
                                                  "too weak on this vehicle (lg1 falsifier "
                                                  "direction: #208 containment-projection "
                                                  "successor). OPERATOR decision owed."})
                    return 0
                # PROSPECTIVE cap on the ROLLBACK path too (ddm_b4r): the protective relaunch is
                # a full window and must also fit inside the cap. Checked BEFORE the retire/rename
                # so custody is untouched when we cannot proceed — an unresolvable-in-budget
                # protection failure is a genuine judgment call, so ALARM+HOLD for MAIN rather
                # than silently endpointing on a half-executed rollback.
                would_exceed, elapsed_now = _cap_would_be_exceeded()
                if would_exceed:
                    _alarm("LANE_EROSION_ROLLBACK_EXCEEDS_CAP", {
                        **details, "elapsed_hours": round(elapsed_now, 3),
                        "cap_hours": TOTAL_CAP_HOURS,
                        "projected_hours_if_relaunched":
                            round(elapsed_now + FULL_WALL_MINUTES / 60.0, 3),
                        "reading": "Lane eroded and the lg1 rollback-and-raise relaunch is owed, "
                                   "but that window cannot COMPLETE inside the burn cap. The "
                                   "eroded window is left IN PLACE (not retired) so custody is "
                                   "intact. MAIN decides: endpoint on the rollback-target ckpt "
                                   "(the last ACCEPTED state), or extend the cap and relaunch."})
                    return 0
                # retire the eroded window + its decision receipt (custody preserved), then
                # relaunch the SAME window number from the rollback target with raised λ.
                rej = ROOT / "rejected"
                rej.mkdir(exist_ok=True)
                stamp = _now().replace(":", "")
                wdir_p = Path(last["window_dir"])
                wdir_p.rename(rej / f"{wdir_p.name}_eroded_{stamp}")
                dec_p = ROOT / f"{wdir_p.name}_decision.json"
                if dec_p.is_file():
                    dec_p.rename(rej / f"{wdir_p.name}_decision_{stamp}.json")
                _log(f"LANE_EROSION rollback: retired {wdir_p.name} -> rejected/; "
                     f"relaunching with lambda_init={new_init:.1f} from {prior}")
                try:
                    _launch_window(n_cur, prior, lambda_init=new_init)
                except RuntimeError as exc:
                    _alarm("GOVERNED_LAUNCH_REFUSED",
                           {"window": n_cur, "lambda_init": new_init, "error": str(exc)})
                    return 0
                time.sleep(POLL_SECONDS)
                continue
            _alarm("LANE_EROSION",
                   {**details, "reading": "R6 (class-weight-lane) ERODES the protected Lane pool "
                                          "beyond noise (xp1's finding realized); the burn is NOT "
                                          "extended. Accepted state = the rollback target ckpt. "
                                          "Needs lg1's λ_Lane constraint to proceed (MAIN)."})
            return 0
        # --- gc13 UNDRIV-EROSION WATCH (ALARM+HOLD; MAIN-PRE-AUTHORIZED continuation) ---
        # No rollback actuator exists (no trainer-side lambda_undriv). MAIN's 2026-07-31
        # standing adjudication CONTINUES a fire that stays inside the pre-authorization
        # bounds (small erosion inside a still-net-negative window); anything outside them
        # is a FRESH adjudication and still ALARM+HOLDs.
        ue = last.get("undriv_erosion") or {}
        if ue.get("eroding"):
            net_comp = ue.get("net_betti0_realized_delta")
            ds = _window_net_delta_s(decisions)
            net_ds = ds.get("net_delta_s")
            within_comp = (net_comp is not None
                           and abs(int(net_comp)) <= UNDRIV_PREAUTH_MAX_ABS_NET)
            within_ds = (not UNDRIV_PREAUTH_REQUIRES_NET_DS_NEGATIVE
                         or (net_ds is not None and net_ds < 0.0))
            common = {"window": last["window_dir"],
                      "slope_comp_per_gate": ue.get("slope_comp_per_gate"),
                      "epsilon_comp_per_gate": ue.get("epsilon_comp_per_gate"),
                      "net_betti0_realized_delta": net_comp,
                      "window_net_delta_s": ds,
                      "preauth_bounds": {
                          "max_abs_net_betti0": UNDRIV_PREAUTH_MAX_ABS_NET,
                          "requires_net_delta_s_negative":
                              UNDRIV_PREAUTH_REQUIRES_NET_DS_NEGATIVE,
                          "within_component_bound": within_comp,
                          "within_net_delta_s_bound": within_ds}}
            if within_comp and within_ds:
                wname = Path(last["window_dir"]).name
                _atomic_write(ROOT / f"{wname}_undriv_preauth.json", {
                    "schema": "ddm_b4r_undriv_preauth.v1", "ts_utc": _now(),
                    "classification": "UNDRIV_EROSION_ADJUDICATED_CONTINUE_PREAUTHORIZED",
                    "reading_is_real_not_spurious": True,
                    "trade": "PRICED: fl1 (#813) prices Undrivable's TOTAL reachable "
                             "non-flicker headroom at +0.0164 S; this window's realized net "
                             "is score-LOWERING. MAIN adjudicated CONTINUE (2026-07-31); the "
                             "trainer-side lambda_undriv routes to cg1 (#809), calibrated by "
                             "this burn's endpoint — NOT a mid-burn build.",
                    **common, "pointer": POINTER, "score_claim": False})
                _log(f"UNDRIV_EROSION PRE-AUTHORIZED (net {net_comp} comp, window net "
                     f"dS {'n/a' if net_ds is None else f'{net_ds:.6f}'}); chain continues")
            else:
                _alarm("UNDRIV_EROSION", {
                    **common,
                    "reading": "Undrivable betti0 erodes beyond the derived noise band AND "
                               "outside MAIN's standing pre-authorization bounds (gc13 "
                               "receipt: Undriv +0.00204 S over the unprotected ep499->641 "
                               "window, MORE than Lane's +0.00151). NO auto-rollback — there "
                               "is no trainer-side lambda_undriv to raise; the burn HOLDS "
                               "here and MAIN adjudicates FRESH."})
                return 0
        # --- window_01 RE-SMOKE ΔS SAFETY GATE -------------------------------------
        if last.get("is_resmoke"):
            verdict = _resmoke_gate(last)
            if verdict == "REGRESSED":
                _alarm("RESMOKE_REGRESSED",
                       {"window": last["window_dir"],
                        "final_gate_dseg": last["final_gate_dseg"],
                        "parent_final_gate_dseg": PARENT_FINAL_GATE_DSEG,
                        "delta": last["delta_gate_dseg_vs_parent"],
                        "noise_floor": RESMOKE_NOISE,
                        "reading": "R6 (class-weight-lane=1.3) REGRESSES this renderer "
                                   "post-plateau beyond noise; R6 falsifier fires at INSTANCE. "
                                   "The full burn is NOT fired."})
                return 0
            _log(f"re-smoke GATE PASS (delta_gate_dseg={last['delta_gate_dseg_vs_parent']}); "
                 "extending to full windows")
            # fall through to the extend branch (do NOT endpoint on the smoke)
        else:
            # full-window STOP conditions
            if last["birth_key"]["fired"]:
                _endpoint_stage(cur, "birth_completion_fired", decisions)
                return 0
        if len(wins) >= MAX_WINDOWS:
            _endpoint_stage(cur, "extension_cap_reached", decisions)
            return 0
        would_exceed, elapsed_h = _cap_would_be_exceeded()
        if elapsed_h >= TOTAL_CAP_HOURS:
            _endpoint_stage(cur, "total_wall_cap_reached", decisions)
            return 0
        # PROSPECTIVE wall cap (ddm_b4r 2026-07-31): the retrospective check above admits a
        # window that STARTS inside the cap and RUNS ~2.2h past it — the cap is only read at
        # window boundaries, so a 5.1h-elapsed boundary would launch a window ending at ~7.3h.
        # Never START a window that cannot COMPLETE inside the ORIGINAL cap; go to the endpoint
        # stage instead (a shorter chain with a real endpoint beats a truncated final window).
        # The cap itself is UNCHANGED (TOTAL_CAP_HOURS, t0 from burn4_t0.json).
        if would_exceed:
            _log(f"PROSPECTIVE cap: elapsed {elapsed_h:.2f}h + window "
                 f"{FULL_WALL_MINUTES / 60.0:.2f}h > cap {TOTAL_CAP_HOURS}h; endpoint now")
            _endpoint_stage(cur, "total_wall_cap_would_be_exceeded", decisions)
            return 0
        # extend: spawn-guard then governed launch of window n+1
        nxt = _window_dir(n + 1)
        if nxt.exists() and (_window_done(nxt) or (
                (p := _window_pid(nxt)) is not None and _pid_alive(p))):
            time.sleep(POLL_SECONDS)
            continue
        resume_ckpt = cur / "checkpoints" / "stage_seg_trunk_tau_final.npz"
        if not resume_ckpt.is_file():
            _alarm("FINAL_CKPT_MISSING", {"window": str(cur), "ckpt": str(resume_ckpt)})
            return 0
        try:
            _launch_window(n + 1, resume_ckpt)
        except RuntimeError as exc:
            _alarm("GOVERNED_LAUNCH_REFUSED", {"window": n + 1, "error": str(exc)})
            return 0
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
