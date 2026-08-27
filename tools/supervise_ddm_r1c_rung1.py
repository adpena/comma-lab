#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# no-argparse-OK: no argv consumed — __main__ runs on pinned inputs/env; nothing for --help to discover
"""ddm_r1c rung-1 DETACHED window-loop SUPERVISOR (task #803; gc12 §3 rung-1 / §5 seal).

The daemon+marker pattern: this script runs DETACHED (nohup+disown) and supervises the
rung-1 continuation windows with ZERO agent attention. Loop:

  wait-for-window-end (poll the run dir's ``tr1_window_receipt.json`` + telemetry mtime)
    -> run the P2 birth-completion key CLI (tools/run_ddm_lp2_birth_completion_key.py,
       epochs_per_gate=5 = the control_tail gate cadence) on the window's telemetry
    -> decide per the SEALED rules (gc12 §5, mechanical — deviations are refusals):
         fired=False ∧ windows<3 (base + ≤2 extensions) ∧ elapsed<8h  => launch next window
             via the governed resume path (tools/launch_tr1_run.py; a REFUSE is information)
         fired=True ∨ caps hit                                        => STOP
         any confound_alarm / a1_stage_exit_refuse / crashed window   => ALARM (never extend)
    -> write a per-window decision receipt JSON after EVERY decision
    -> on STOP: run the terminal endpoint stage (n600 realized verdict via
       experiments/ddm_pa1r_endpoint_verdict.py + QA80 staleness re-check + final key row),
       write ``rung1_endpoint_manifest.json`` + the ``rung1.done`` marker.
    -> on ALARM: write the ``rung1.ALARM`` marker (diagnosis is MAIN's; no endpoint stage).

Crash-resumable from disk: state is DERIVED from the run dirs/receipts on every iteration
(stateless loop); a restart never double-launches (live-pid + receipt spawn-guard; pidfile
singleton). Sealed config untouched: window tickets are DSL-compiled copies of the parent
control_tail ticket with ONLY the window lever swapped (argv-diff = the gc12 §5 four).

MAIN consumes the endpoint via the ``rung1.done`` marker. score_claim=False; every number
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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# NB: after the sys.path bootstrap above -- import position is deliberate.
from tac import process_liveness  # noqa: E402

VENV_PY = str(REPO / ".venv" / "bin" / "python")
ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_r1c_20260731")
PARENT_TICKET = Path(
    "/Volumes/VertigoDataTier/pact/ddm_pa1r_20260730/tickets/control_tail_ticket.json")
QA91_INV = Path("/Volumes/VertigoDataTier/pact/ddm_fp1_20260731/qa91_erased_lane.json")
GT_CACHE = str(REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz")

# pa1r control_tail endpoint (the parent baseline the endpoint deltas are read against).
PARENT_BASELINE = {"n600_d_seg": 0.0049411, "s_additive": 0.67325,
                   "tokens_bytes_smevr": 265528, "total_counted_bytes": 269028}

WINDOW_EPOCHS = 140          # gc12 §5 window extent (~120-150 ep)
WALL_MINUTES = 130.0         # per-window wall cap (measured 36.8 s/ep x 140 ~= 86 min + slack)
MAX_WINDOWS = 3              # base window + <=2 extensions (gc12 §5)
TOTAL_CAP_HOURS = 8.0        # hard rung-1 wall cap (gc12 §5)
EPOCHS_PER_GATE = 5          # control_tail --gate-every 5 (NOT the bc1 default 10)
POLL_SECONDS = 120

DONE_MARKER = ROOT / "rung1.done"
ALARM_MARKER = ROOT / "rung1.ALARM"
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
    """Delegates to the canonical tri-state read (``tac.process_liveness``).

    Behaviour CHANGED in three ways (undocumented drift in the old local copy):

    * ``PermissionError`` was explicitly DEAD, now ALIVE.
    * A zombie was ALIVE forever, now DEAD -- a finished-but-unreaped window
      worker used to keep this supervisor waiting on a process that can never
      run again.
    * ``pid <= 0`` had NO guard: ``os.kill(0, 0)`` targets our OWN process
      group and SUCCEEDS, so a ``launch_manifest.json`` carrying ``"pid": 0``
      read the window as RUNNING forever.  It is now UNREADABLE -> False.
    """
    return process_liveness.pid_state(pid) == process_liveness.ALIVE


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
    """F1-F4/A1 halt semantics: confound_alarm + a1_stage_exit_refuse rows halt the loop."""
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


def _build_ticket(epochs: int, out_dir: Path, resume_from: Path, out_ticket: Path) -> dict:
    """DSL-compile the window ticket: parent levers VERBATIM, only the window lever swapped."""
    sys.path.insert(0, str(REPO / "src"))
    from tac.witness_dsl.curriculum_dsl import Lever
    from tac.witness_dsl.spec_tr1_renderer_20260728 import TR1RendererProgramV1, lever_window

    parent = json.loads(PARENT_TICKET.read_text())
    levers: list[Lever] = []
    for d in parent["levers"]:
        if "--epochs" in d["overrides"]:
            levers.append(lever_window(epochs, WALL_MINUTES, batch_pairs=8, lr=2e-3))
        else:
            levers.append(Lever(name=d["name"], overrides=dict(d["overrides"]),
                                notes=d.get("notes", "")))
    prog = TR1RendererProgramV1(levers=tuple(levers), num_pairs=600, out_dir=str(out_dir),
                                seed=0, gt_cache=GT_CACHE, resume_from=str(resume_from),
                                full_confirm=True)
    ticket = prog.sealed_ticket()  # compiles + fail-closed validate + ticket_hash
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


def _launch_window(n: int, resume_ckpt: Path) -> dict:
    wdir = _window_dir(n)
    epochs = _ckpt_epoch(resume_ckpt) + 1 + WINDOW_EPOCHS
    ticket_path = ROOT / "tickets" / f"window_{n:02d}_ticket.json"
    ticket = _build_ticket(epochs, wdir, resume_ckpt, ticket_path)
    r = subprocess.run(
        [VENV_PY, "tools/launch_tr1_run.py",
         "--ticket", str(ticket_path), "--out-dir", str(wdir),
         "--resume-from", str(resume_ckpt),
         "--purpose", f"ddm_r1c rung1 window_{n:02d} continuation"],
        cwd=REPO, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(
            f"governed launcher REFUSED window_{n:02d} rc={r.returncode}: "
            f"{(r.stderr or r.stdout).strip()[:600]}")
    _log(f"window_{n:02d} launched (epochs->{epochs}, ticket {ticket['ticket_hash'][:12]})")
    return {"window": n, "epochs_arg": epochs, "ticket_hash": ticket["ticket_hash"],
            "ticket_path": str(ticket_path)}


def _elapsed_hours() -> float:
    rec = _window_dir(1) / "launch_receipt.json"
    t0 = datetime.strptime(json.loads(rec.read_text())["launched_utc"],
                           "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=UTC)
    return (datetime.now(UTC) - t0).total_seconds() / 3600.0


def _heartbeat(phase: str, window: int) -> None:
    _atomic_write(STATE_PATH, {"ts_utc": _now(), "phase": phase, "current_window": window,
                               "supervisor_pid": os.getpid(), "pointer": POINTER,
                               "score_claim": False})


def _alarm(reason: str, details: dict) -> None:
    _atomic_write(ALARM_MARKER, {"schema": "ddm_r1c_rung1_alarm.v1", "ts_utc": _now(),
                                 "reason": reason, "details": details,
                                 "action_owed": "MAIN diagnose + report (gc12 W4)",
                                 "pointer": POINTER, "score_claim": False})
    _log(f"ALARM: {reason} -> {ALARM_MARKER}")


def _qa80_staleness_recheck(wdir: Path) -> dict:
    """Mechanical endpoint re-check: the sealed lineage must still NOT consume the stored
    QA80 field (its margin term is the LIVE per-step realized-SegNet margin)."""
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
        "schema": "ddm_r1c_rung1_endpoint_manifest.v1", "ts_utc": _now(),
        "stop_reason": stop_reason, "final_window_dir": str(final_dir),
        "final_ckpt": str(ckpt),
        "final_ckpt_sha256": _sha256(ckpt) if ckpt.is_file() else None,
        "final_ckpt_meta_epoch": _ckpt_epoch(ckpt) if ckpt.is_file() else None,
        "windows": decisions, "parent_baseline": PARENT_BASELINE,
        "pointer": POINTER, "score_claim": False,
        "evidence_axis": "[macOS-CPU/MLX advisory]",
    }
    # Obligation 1 (FOLDED IN): n600 realized verdict through the pa1r endpoint harness
    # (EMA shadow + re-engaged STE + R->uint8->frozen CPU-torch SegNet + real SMEVR byte-close).
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
            manifest["endpoint_verdict"] = v
        else:
            manifest["endpoint_verdict_error"] = (r.stderr or r.stdout).strip()[-800:]
            manifest["endpoint_verdict_owed"] = "MAIN: re-run ddm_pa1r_endpoint_verdict.py"
    except Exception as exc:  # never lose the manifest to a verdict failure
        manifest["endpoint_verdict_error"] = repr(exc)
        manifest["endpoint_verdict_owed"] = "MAIN: re-run ddm_pa1r_endpoint_verdict.py"
    # Obligation 2 (SPLIT): DERIVED-ESTIMATE from the final key row here; the EXACT S-unit
    # pool P (QA92 base-pass method, 8-conn per-component flip-mass) is left to MAIN.
    key = decisions[-1].get("birth_key", {}) if decisions else {}
    manifest["p_remeasure"] = {
        "erased_count_endpoint": key.get("erased_count"),
        "above_nucleus_erased_estimate": key.get("above_nucleus_erased_estimate"),
        "label": "DERIVED-ESTIMATE (tr1 4-conn betti0 x QA91 8-conn area frac)",
        "exact_owed_to_main": "QA92 base-pass method on this endpoint render "
                              "(experiments/ddm_qa92_carrier_discriminator.py) -> P in S units",
    }
    # Obligation 3 (FOLDED IN): QA80 staleness re-check at the endpoint config.
    manifest["qa80_staleness_recheck"] = _qa80_staleness_recheck(final_dir)
    _atomic_write(ROOT / "rung1_endpoint_manifest.json", manifest)
    _atomic_write(DONE_MARKER, {"schema": "ddm_r1c_rung1_done.v1", "ts_utc": _now(),
                                "endpoint_manifest": str(ROOT / "rung1_endpoint_manifest.json"),
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


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    # singleton guard
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
    _log(f"supervisor start pid {os.getpid()}")

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
                   {"window": str(cur), "pid": pid,
                    "halt_events": _scan_halt_events(cur)})
            return 0
        # window done -> let the trainer process FULLY exit before evaluating/launching
        # (the receipt is written moments before process exit; launching into that gap
        # would trip the governed launcher's G4 scorer-slot check -> spurious REFUSE).
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
                "schema": "ddm_r1c_rung1_window_decision.v1", "ts_utc": _now(),
                "window": n, "window_dir": str(cur),
                "stop_reason": receipt.get("stop_reason"),
                "epochs_ran": receipt.get("epochs_ran"),
                "final_gate_epoch": fg.get("epoch"),
                "final_gate_dseg": fg.get("realized_gate_dseg_mean"),
                "final_gate_a1": fg.get("a1_classification"),
                "full_confirm_dseg": (receipt.get("full_confirm") or {}).get(
                    "realized_dseg_mean"),
                "total_counted_bytes": fg.get("total_counted_bytes"),
                "halt_events": halt,
                "birth_key": {k: key.get(k) for k in
                              ("fired", "slope_le_epsilon",
                               "above_nucleus_erasure_persists", "erased_count",
                               "above_nucleus_erased_estimate",
                               "betti0_realized_endpoint")},
                "birth_key_fit": key.get("fit"),
                "elapsed_hours": round(_elapsed_hours(), 3),
                "pointer": POINTER, "score_claim": False,
            }
            _atomic_write(dec_path, decision)
            _log(f"decision written for {cur.name}: fired={key.get('fired')} "
                 f"stop_reason={receipt.get('stop_reason')} halt={len(halt)}")
        decisions = _decision_receipts()
        last = decisions[-1]
        if last["halt_events"]:
            _alarm("HALT_EVENTS_IN_WINDOW", {"window": last["window_dir"],
                                             "halt_events": last["halt_events"]})
            return 0
        if last["stop_reason"] not in OK_STOP_REASONS:
            _alarm("UNEXPECTED_STOP_REASON", {"window": last["window_dir"],
                                              "stop_reason": last["stop_reason"]})
            return 0
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
        _heartbeat("launched_window", n + 1)
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
