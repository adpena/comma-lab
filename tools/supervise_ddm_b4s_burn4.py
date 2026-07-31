#!/usr/bin/env python
# SPDX-License-Identifier: MIT
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
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO = Path("/Users/adpena/Projects/pact")
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
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    except Exception:
        return False
    return True


def _window_dir(n: int) -> Path:
    return ROOT / f"window_{n:02d}"


def _existing_windows() -> list[Path]:
    return sorted(p for p in ROOT.glob("window_*") if p.is_dir())


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
        },
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
    _atomic_write(ROOT / "burn4_endpoint_manifest.json", manifest)
    _atomic_write(DONE_MARKER, {"schema": "ddm_b4s_burn4_done.v1", "ts_utc": _now(),
                                "endpoint_manifest": str(ROOT / "burn4_endpoint_manifest.json"),
                                "stop_reason": stop_reason, "pointer": POINTER,
                                "score_claim": False})
    _log(f"STOP ({stop_reason}) -> {DONE_MARKER}")


def _decision_receipts() -> list[dict]:
    out = []
    for p in sorted(ROOT.glob("window_*_decision.json")):
        try:
            out.append(json.loads(p.read_text()))
        except Exception:
            continue
    return out


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
        if _elapsed_hours() >= TOTAL_CAP_HOURS:
            _endpoint_stage(cur, "total_wall_cap_reached", decisions)
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
