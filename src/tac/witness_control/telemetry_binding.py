"""Binding-vs-inert telemetry audit over a witness run's ALREADY-EMITTED rows (#404, P0 2026-07-10).

READ-ONLY / POST-HOC / SCORE-NEUTRAL / DEFAULT-ON (CLAUDE.md "'Off' is a tracked queue": read-only
telemetry defaults ON). This module never touches the trainer, never writes into a run dir, and is
safe to run against the SEALED v7.5.2 relaunch chain mid-flight. It is the L1-alarm READ side of the
Confound self-protection non-negotiable: its verdicts answer "is this lever BINDING or INERT?" for
the new v7.5.2 levers from the rows the trainer already prints to ``run.log``:

* (b) AMBER stability stack  — global grad-clip binding rate from ``loss_terms.gnorm`` vs the
  EFFECTIVE ``grad_clip`` stamped by the ``witness_stability_resolved`` row (the inherited
  ``--grad-clip 1.0`` defeat is exactly the confound this catches). Per-GROUP clip rates are NOT
  emitted by the trainer yet (queued trainer patch; see the #404 audit memo).
* (c) CHROMA rung            — ``chroma_boundary`` share of the post-weight loss total (INERT_ZERO
  floor alarm + DOMINATING >40% mirror of the in-run ``term_domination`` alarm, which does NOT
  include chroma in its ``_reg_keys``).
* (d) POSE-FINISH detector   — ``jacobian_basin`` sensor cadence liveness (the SigmaMinPlateauDetector
  AttributeError crash class died SILENTLY for 2.5 h: a stalled sensor now reads as DETECTOR_STALLED)
  + the ``pose_finish_*`` decision rows + the four pose-gate confound alarms.
* (e) EMA-LAG               — verdict d_pose trend (EMA-graded) vs live ``loss_terms.pose`` trend
  (the CONFIRMED run-1 confound: verdict d_pose rises while training pose falls).
* (f) SOLVE-UPON-BASIN      — the D27b terminal-band trigger signal: muon fired AND rolling d_seg
  relative slope ~ 0 over the trailing verdict window (plus TAIL PowerPlay stop / Polyak arm as
  terminal-band confirmation). ``d27b_ready`` is the machine-readable trigger.
* (h) TAIL cycle endpoints  — per-cycle endpoint stats joined from ``tail_cycle_begin`` boundaries
  and the nearest preceding verdict rows (feeds the later SWA-soup candidate evaluation).
* (a) EVENT decision table  — one queryable table normalizing every event/gate decision row
  (``start_event_fired`` / ``cap_fired_before_event`` / engage rows / tail rows / would-fire rows).

Everything here is MEANS (advisory, ``[macOS-MLX advisory] NON-PROMOTABLE`` inputs); the pointer
moves only through a byte-closed ``upstream/evaluate.py`` row. Cited discipline:
``docs/operating_manual_craft_handoff.md`` (label MEASURED vs DERIVED; a verdict from a corrupted
instrument is not a verdict — hence the liveness checks on the instruments themselves).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# ── row-schema constants (verified against the trainer's emit sites 2026-07-10) ──────────────────
# experiments/train_levelset_witness_realized_through_R_mlx.py:
#   loss_terms row   ~L3012 (_loss_terms_row)      verdict row ~L6949/L9726
#   jacobian_basin   ~L6649                        muon_finisher_switch ~L8790
#   tail_cycle_begin ~L9292  tail_powerplay_stop ~L9300  tail_early_stop ~L9966
#   witness_stability_resolved ~L11939 (startup)   confound_alarm ~L8260 (_emit_confound_alarm)
# src/tac/witness_control/event_wirings.py: start_event_fired L191 / cap_fired_before_event L207 /
#   lane_band_would_fire L387.  sigma_min_plateau.py: pose_finish_conditioning_gate L564.
EVENT_STAGES: tuple[str, ...] = (
    "start_event_fired", "cap_fired_before_event", "lane_band_would_fire",
    "curriculum_transition", "tau_advance_armed", "muon_finisher_switch",
    "pose_finish_armed", "pose_finish_engage", "pose_finish_conditioning_gate",
    "lane_render_band_engage", "seg_chroma_boundary_engage", "seg_temporal_screw_engage",
    "birth_completion_ramp", "ladder_rung",
    "tail_controller_armed", "tail_cycle_begin", "tail_powerplay_stop", "tail_early_stop",
    "event_curriculum_inert_under_unify", "resume_event_curriculum",
    "polyak_finisher_armed", "handoff_readiness",
)
POSE_GATE_ALARMS: tuple[str, ...] = (
    "pose_finish_gate_no_sigma_source", "pose_finish_gate_canary_failed",
    "pose_finish_disengaged_shipped_banked_r1", "pose_finish_backstop_overridden_banked_r1",
)

_MIN_ROWS_FOR_VERDICT = 8  # below this the honest answer is UNKNOWN, not a verdict


# ── loading ───────────────────────────────────────────────────────────────────────────────────────
def parse_log_lines(lines: "list[str]") -> list[dict]:
    """Extract the JSON telemetry rows from raw log lines. The trainer prints one JSON object per
    row; wrapper lines (safe_run, admission guard) may prefix text before the first ``{`` — parse
    from the first brace and skip anything that does not decode to a dict with a ``stage`` key."""
    rows: list[dict] = []
    for line in lines:
        i = line.find("{")
        if i < 0:
            continue
        try:
            d = json.loads(line[i:])
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(d, dict) and "stage" in d:
            rows.append(d)
    return rows


def load_run_rows(run_dir: "Path | str", *, tail_bytes: int = 0) -> list[dict]:
    """Load telemetry rows from every ``run.log`` under ``run_dir`` (the run root plus nested
    dry_start/resume dirs), in path-sorted order. ``tail_bytes > 0`` bounds the read per file
    (cheap digest cadence); 0 reads whole files (CLI / offline audit)."""
    root = Path(run_dir)
    logs = sorted(root.glob("run.log")) + sorted(p for p in root.glob("*/run.log"))
    rows: list[dict] = []
    for log in logs:
        try:
            if tail_bytes > 0:
                with log.open("rb") as fh:
                    fh.seek(max(0, log.stat().st_size - tail_bytes))
                    text = fh.read().decode("utf-8", errors="replace")
            else:
                text = log.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        rows.extend(parse_log_lines(text.splitlines()))
    return rows


def _row_epoch(r: dict) -> "int | None":
    for k in ("epoch", "ep"):
        v = r.get(k)
        if isinstance(v, (int, float)):
            return int(v)
    return None


def _stage_rows(rows: "list[dict]", stage: str) -> list[dict]:
    return [r for r in rows if r.get("stage") == stage]


def _verdict_rows(rows: "list[dict]") -> list[dict]:
    out = [r for r in rows if r.get("stage") == "verdict"
           and isinstance(r.get("d_seg"), (int, float)) and isinstance(r.get("d_pose"), (int, float))]
    out.sort(key=lambda r: _row_epoch(r) or -1)
    return out


def _alarms(rows: "list[dict]") -> list[dict]:
    return [r for r in rows if r.get("stage") == "confound_alarm"]


# ── (a) event decision table ──────────────────────────────────────────────────────────────────────
def event_decision_table(rows: "list[dict]") -> list[dict]:
    """Normalize every event/gate decision row into one queryable, epoch-sorted table:
    ``{epoch, event, fired_by/state, sensor, detail}``. This is the post-hoc DECISION-PROVENANCE
    query surface the P0 asks for — built entirely from rows the trainer already emits."""
    table: list[dict] = []
    for r in rows:
        st = r.get("stage")
        if st not in EVENT_STAGES:
            continue
        detail = {k: v for k, v in r.items()
                  if k not in ("stage", "epoch", "ep", "note", "axis") and not isinstance(v, (dict, list))}
        table.append({
            "epoch": _row_epoch(r),
            "event": st if st != "start_event_fired" else f"start_event_fired:{r.get('transition')}",
            "fired_by": r.get("fired_by") or r.get("engage_via") or r.get("reason"),
            "sensor": r.get("sensor"),
            "sensor_data_epoch": r.get("sensor_data_epoch"),
            "sensor_lag_epochs": r.get("sensor_lag_epochs"),
            "detail": detail,
        })
    table.sort(key=lambda e: (e["epoch"] is None, e["epoch"]))
    return table


# ── (b) amber stability stack ─────────────────────────────────────────────────────────────────────
def amber_binding(rows: "list[dict]") -> dict:
    """GLOBAL grad-clip binding rate (MEASURED from ``loss_terms.gnorm`` vs the effective
    ``grad_clip``). Verdicts: BINDING (clips sometimes), INERT_NEVER_BINDS (never clips — the amber
    knob is decorative at this config), SATURATED_ALWAYS_CLIPS (>90% of rows clip — the step size is
    entirely clip-determined), UNKNOWN (insufficient rows). Per-GROUP rates need the queued trainer
    row (per-group norms are not emitted); this is stated, not silently omitted."""
    stab = _stage_rows(rows, "witness_stability_resolved")
    grad_clip = float(stab[-1]["grad_clip"]) if stab and isinstance(stab[-1].get("grad_clip"), (int, float)) else None
    per_group = bool(stab[-1].get("per_group_grad_clip")) if stab else None
    gnorms = [float(r["gnorm"]) for r in _stage_rows(rows, "loss_terms")
              if isinstance(r.get("gnorm"), (int, float))]
    hijacks = [a for a in _alarms(rows) if a.get("alarm") == "gnorm_hijack"]
    out: dict[str, Any] = {
        "grad_clip_effective": grad_clip, "per_group_grad_clip": per_group,
        "n_gnorm_rows": len(gnorms), "gnorm_hijack_alarms": len(hijacks),
        "per_group_rates_available": False,
        "queued": "per-group clip-activation row (trainer patch, gated on relaunch/resume boundary)",
    }
    if grad_clip is None or grad_clip <= 0 or len(gnorms) < _MIN_ROWS_FOR_VERDICT:
        out.update(verdict="UNKNOWN", clip_binding_frac=None)
        return out
    binds = sum(1 for g in gnorms if g > grad_clip)
    frac = binds / len(gnorms)
    out["clip_binding_frac"] = round(frac, 4)
    out["gnorm_max"] = round(max(gnorms), 4)
    if frac == 0.0:
        out["verdict"] = "INERT_NEVER_BINDS"
    elif frac > 0.9:
        out["verdict"] = "SATURATED_ALWAYS_CLIPS"
    else:
        out["verdict"] = "BINDING"
    return out


# ── (c) chroma rung ───────────────────────────────────────────────────────────────────────────────
def term_share_series(rows: "list[dict]", term: str) -> list[tuple[int, float]]:
    """Per-``loss_terms``-row |term| / (|total| + eps) share series, epoch-tagged."""
    out: list[tuple[int, float]] = []
    for r in _stage_rows(rows, "loss_terms"):
        terms = r.get("terms")
        if not isinstance(terms, dict) or term not in terms:
            continue
        tot = abs(float(r.get("total", 0.0))) + 1e-12
        ep = _row_epoch(r)
        if ep is not None:
            out.append((ep, abs(float(terms[term])) / tot))
    return out


def chroma_binding(rows: "list[dict]", *, domination_frac: float = 0.40,
                   inert_floor: float = 1e-6) -> dict:
    """Chroma-boundary term share of the loss total AFTER its engage event. Verdicts: PENDING (not
    engaged yet), INERT_ZERO (engaged but the term contributes ~0 — the floor alarm the in-run
    ``term_domination`` alarm does not provide), DOMINATING (>40%, mirroring the in-run alarm which
    does NOT list chroma in its ``_reg_keys``), BINDING otherwise, UNKNOWN (too few rows)."""
    engage = _stage_rows(rows, "seg_chroma_boundary_engage") or [
        r for r in _stage_rows(rows, "seg_chroma_boundary") if r.get("active")]
    engaged_ep = min((e for e in (_row_epoch(r) for r in engage) if e is not None), default=None)
    series = term_share_series(rows, "chroma_boundary")
    out: dict[str, Any] = {"engaged_epoch": engaged_ep, "n_rows": len(series)}
    if engaged_ep is None:
        out["verdict"] = "PENDING"
        return out
    post = [s for ep, s in series if ep >= engaged_ep]
    out["n_rows_post_engage"] = len(post)
    if len(post) < _MIN_ROWS_FOR_VERDICT:
        out["verdict"] = "UNKNOWN"
        return out
    mean_share = sum(post) / len(post)
    out["mean_share_post_engage"] = round(mean_share, 6)
    out["latest_share"] = round(post[-1], 6)
    if mean_share < inert_floor:
        out["verdict"] = "INERT_ZERO"
    elif mean_share > domination_frac:
        out["verdict"] = "DOMINATING"
    else:
        out["verdict"] = "BINDING"
    return out


# ── (d) pose-finish detector health ───────────────────────────────────────────────────────────────
def pose_gate_health(rows: "list[dict]", *, stall_cadence_mult: float = 4.0,
                     no_row_grace_verdicts: int = 6) -> dict:
    """Instrument-liveness for the σ_min plateau chain: the ``jacobian_basin`` sensor must keep
    emitting at its own cadence while verdicts advance — a sensor that stops (the silent
    SigmaMinPlateauDetector-crash class) reads DETECTOR_STALLED. Also surfaces the conditioning-gate
    observer rows, engage/armed decisions, and the four pose-gate confound alarms. Grace: with ZERO
    sensor rows the stall call waits for ``no_row_grace_verdicts`` verdicts (an early run whose T1
    sensor has not fired yet is not stalled); an explicitly DISABLED sensor is DISABLED, not stalled."""
    jb = sorted(_stage_rows(rows, "jacobian_basin"), key=lambda r: _row_epoch(r) or -1)
    gates = _stage_rows(rows, "pose_finish_conditioning_gate")
    verds = _verdict_rows(rows)
    alarms = [a for a in _alarms(rows) if a.get("alarm") in POSE_GATE_ALARMS]
    disabled = bool(_stage_rows(rows, "jacobian_basin_disabled"))
    out: dict[str, Any] = {
        "n_jacobian_basin_rows": len(jb), "n_gate_observer_rows": len(gates),
        "sensor_disabled": disabled,
        "armed": bool(_stage_rows(rows, "pose_finish_armed")),
        "engaged": bool(_stage_rows(rows, "pose_finish_engage")),
        "alarms": [{"alarm": a.get("alarm"), "epoch": _row_epoch(a)} for a in alarms],
    }
    if jb:
        last = jb[-1]
        out["last_epoch"] = _row_epoch(last)
        out["median_sigma_min_last"] = last.get("median_sigma_min")
        out["sigma_min_plateau_est"] = last.get("sigma_min_plateau_est")
        out["would_have_fired_any"] = any(r.get("would_have_fired") for r in jb)
    if len(jb) >= 2 and verds:
        eps = [e for e in (_row_epoch(r) for r in jb) if e is not None]
        diffs = [b - a for a, b in zip(eps, eps[1:]) if b > a]
        cadence = sorted(diffs)[len(diffs) // 2] if diffs else None
        last_v = _row_epoch(verds[-1])
        if cadence and last_v is not None:
            gap = last_v - eps[-1]
            out["sensor_gap_epochs"] = gap
            out["stalled"] = bool(gap > stall_cadence_mult * cadence)
        else:
            out["stalled"] = None
    elif not jb and not disabled and len(verds) >= no_row_grace_verdicts:
        out["stalled"] = True  # many verdicts elapsed, sensor never emitted: the silent-crash class
    else:
        out["stalled"] = None
    out["verdict"] = ("DISABLED" if disabled
                      else "DETECTOR_STALLED" if out.get("stalled")
                      else "ALARMED" if alarms
                      else "OK" if jb else "NO_SENSOR_ROWS")
    return out


# ── (e) EMA-lag ───────────────────────────────────────────────────────────────────────────────────
def ema_lag(rows: "list[dict]", *, window: int = 8) -> dict:
    """Verdict d_pose trend (EMA-graded weights) vs live training-pose-term trend over the SAME
    trailing epoch span. DIVERGING = verdict d_pose RISING while the live pose term FALLS — the
    CONFIRMED run-1 EMA-shadow-lag confound signature (check EMA lag FIRST before narrating a pose
    regression). Trends are endpoint deltas over the window (sign is what matters, not slope size)."""
    verds = _verdict_rows(rows)
    out: dict[str, Any] = {"window": window, "n_verdicts": len(verds)}
    if len(verds) < max(3, window // 2):
        out["verdict"] = "UNKNOWN"
        return out
    vwin = verds[-window:]
    ep_lo, ep_hi = _row_epoch(vwin[0]), _row_epoch(vwin[-1])
    d_pose_delta = float(vwin[-1]["d_pose"]) - float(vwin[0]["d_pose"])
    live = [(ep, float(r["terms"]["pose"])) for r in _stage_rows(rows, "loss_terms")
            if isinstance(r.get("terms"), dict) and isinstance(r["terms"].get("pose"), (int, float))
            and (ep := _row_epoch(r)) is not None and ep_lo is not None and ep_hi is not None
            and ep_lo <= ep <= ep_hi]
    out["epoch_span"] = [ep_lo, ep_hi]
    out["verdict_d_pose_delta"] = round(d_pose_delta, 8)
    if len(live) < 2:
        out["verdict"] = "UNKNOWN"
        return out
    live.sort()
    live_delta = live[-1][1] - live[0][1]
    out["live_pose_term_delta"] = round(live_delta, 8)
    out["diverging"] = bool(d_pose_delta > 0 and live_delta < 0)
    out["verdict"] = "EMA_LAG_DIVERGING" if out["diverging"] else "CONSISTENT"
    return out


# ── (f) solve-upon-basin / D27b terminal-band trigger ─────────────────────────────────────────────
def terminal_band_status(rows: "list[dict]", *, slope_window: int = 8, rel_eps: float = 5e-3) -> dict:
    """The D27b trigger signal, made measurable: ``in_basin`` = Muon finisher fired AND the trailing
    ``slope_window``-verdict d_seg relative change is within ``rel_eps`` (Muon-stage plateau).
    ``terminal_band`` additionally requires a TAIL PowerPlay/k_max stop or the Polyak finisher armed
    (governed-stop adjacency). ``d27b_ready`` == in_basin — fire the terminal solve stack on the
    NEXT checkpoint/byte-close, never mid-run (deferral ledger D27b)."""
    muon = _stage_rows(rows, "muon_finisher_switch")
    verds = _verdict_rows(rows)
    tail_stop = _stage_rows(rows, "tail_powerplay_stop") + _stage_rows(rows, "tail_early_stop")
    polyak = _stage_rows(rows, "polyak_finisher_armed") + _stage_rows(rows, "polyak_finisher_resumed")
    out: dict[str, Any] = {
        "muon_fired_epoch": _row_epoch(muon[0]) if muon else None,
        "n_verdicts": len(verds), "slope_window": slope_window, "rel_eps": rel_eps,
        "tail_stop_seen": bool(tail_stop), "polyak_armed": bool(polyak),
    }
    reasons: list[str] = []
    plateau = None
    if len(verds) >= slope_window:
        win = verds[-slope_window:]
        d0, d1 = float(win[0]["d_seg"]), float(win[-1]["d_seg"])
        rel = (d0 - d1) / d0 if d0 > 0 else 0.0
        out["d_seg_window"] = [round(d0, 6), round(d1, 6)]
        out["d_seg_rel_improvement_window"] = round(rel, 6)
        plateau = abs(rel) < rel_eps
        reasons.append(f"d_seg rel change {rel:.2e} over last {slope_window} verdicts "
                       f"({'plateau' if plateau else 'still moving'})")
    else:
        reasons.append(f"only {len(verds)} verdicts (< window {slope_window}) — too early")
    if muon:
        reasons.append(f"muon finisher fired @ep{out['muon_fired_epoch']}")
    else:
        reasons.append("muon finisher not fired")
    in_basin = bool(muon) and bool(plateau)
    out["in_basin"] = in_basin
    out["terminal_band"] = in_basin and (bool(tail_stop) or bool(polyak))
    out["d27b_ready"] = in_basin
    out["reasons"] = reasons
    return out


# ── (h) TAIL cycle endpoints ──────────────────────────────────────────────────────────────────────
def tail_cycle_endpoints(rows: "list[dict]") -> list[dict]:
    """Per-TAIL-cycle endpoint stats: at each ``tail_cycle_begin`` boundary, the endpoint of the
    segment ENDING there = the last verdict at/before the boundary epoch (plus the run-final
    endpoint). Per-segment best d_seg included. Feeds the SWA-soup candidate evaluation later."""
    verds = _verdict_rows(rows)
    begins = sorted(_stage_rows(rows, "tail_cycle_begin"), key=lambda r: _row_epoch(r) or -1)
    if not verds:
        return []
    endpoints: list[dict] = []
    seg_start_ep = -1
    for b in begins:
        bep = _row_epoch(b)
        if bep is None:
            continue
        seg = [v for v in verds if seg_start_ep < (_row_epoch(v) or -1) <= bep]
        if seg:
            last = seg[-1]
            endpoints.append({
                "segment_end": "tail_cycle_begin", "cycle_next": b.get("cycle"),
                "boundary_epoch": bep, "endpoint_epoch": _row_epoch(last),
                "d_seg": last.get("d_seg"), "d_pose": last.get("d_pose"),
                "implied_S": last.get("implied_S"),
                "best_d_seg_in_segment": min(float(v["d_seg"]) for v in seg),
            })
        seg_start_ep = bep
    tail = [v for v in verds if (_row_epoch(v) or -1) > seg_start_ep]
    if tail:
        last = tail[-1]
        endpoints.append({
            "segment_end": "final", "cycle_next": None,
            "boundary_epoch": _row_epoch(last), "endpoint_epoch": _row_epoch(last),
            "d_seg": last.get("d_seg"), "d_pose": last.get("d_pose"),
            "implied_S": last.get("implied_S"),
            "best_d_seg_in_segment": min(float(v["d_seg"]) for v in tail),
        })
    return endpoints


# ── combined audit ────────────────────────────────────────────────────────────────────────────────
def audit_rows(rows: "list[dict]") -> dict:
    """Run every analyzer; each is independently fail-open (an exception in one section must never
    hide the others — the digest consumer depends on that)."""
    out: dict[str, Any] = {"n_rows": len(rows)}
    for key, fn in (("events", event_decision_table), ("amber", amber_binding),
                    ("chroma", chroma_binding), ("pose_gate", pose_gate_health),
                    ("ema_lag", ema_lag), ("terminal_band", terminal_band_status),
                    ("tail_endpoints", tail_cycle_endpoints)):
        try:
            out[key] = fn(rows)
        except Exception as exc:  # fail-open per section: telemetry reading must never crash a consumer
            out[key] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def format_summary(audit: dict) -> str:
    """One digest line: the binding/inert verdict per lever + the D27b basin state."""
    def _v(section: str) -> str:
        s = audit.get(section) or {}
        return str(s.get("verdict", "err" if "error" in s else "n/a"))
    tb = audit.get("terminal_band") or {}
    basin = ("TERMINAL-BAND" if tb.get("terminal_band")
             else "IN-BASIN" if tb.get("in_basin") else "not-yet")
    n_ev = len(audit.get("events") or []) if isinstance(audit.get("events"), list) else 0
    n_tail = len(audit.get("tail_endpoints") or []) if isinstance(audit.get("tail_endpoints"), list) else 0
    return (f"telem-binding: amber={_v('amber')} chroma={_v('chroma')} pose-gate={_v('pose_gate')} "
            f"ema-lag={_v('ema_lag')} basin={basin} events={n_ev} tail-endpoints={n_tail} "
            f"(read-only; advisory NON-PROMOTABLE)")
