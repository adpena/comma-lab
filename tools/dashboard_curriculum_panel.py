"""Truth-rendering models for the dashboard's CURRICULUM POSITION + POSE-DESCENT
READINESS panels — DERIVED from the run's own config + emitted evidence, never a
hardcoded PR95-style epoch skeleton.

Operator complaint (verbatim, 2026-07-10): the curriculum panel rendered a
3-segment epoch bar (CE | tau | Muon) with HARDCODED epoch positions — hiding the
event-gated DERIVED schedule that is actually running, which made our derived
curriculum look cargo-culted from PR95. Sister complaint: "It also doesn't display
the R1" — the pose panel showed only the jacobian_basin readiness, never the BANKED
R1 fallback (the sealed program's shippable pose custody).

This module builds two JSON-able display MODELS the (JS) panels render verbatim:

1. :func:`build_curriculum_panel_model` — the curriculum as DERIVED:
   * stage transitions shown as EVENTS with their primary trigger; the epoch is
     labeled explicitly as a FAIL-SAFE CAP, never as the trigger;
   * the tau-path as ONE continuous geometric anneal (CE = the tau=1 limit);
   * the event-gated MECHANISM LANES (lane-band, chroma-boundary, temporal-screw,
     birth-completion, LADDER, TAIL, Polyak, terminal solve) each with
     {trigger, cap, state: pending|armed|fired@ep};
   * a provenance line that answers the cargo-cult question on the panel itself.

2. :func:`build_pose_readiness_model` — the BANKED R1 fallback + the fallback
   contract state (detector mode / state / decision tree). The R1 numbers are READ
   from the #238 byte-close artifact JSON (never hardcoded); the contribution is
   DERIVED as sqrt(10*d_pose).

VALUE-PROVENANCE DISCIPLINE (CLAUDE.md): every epoch/cap comes from the schedule
read-back or the parsed launch.sh flags; every R1 number comes from the artifact
JSON. The only literals here are DESCRIPTIVE TEXT (lane names, trigger prose,
the provenance line) — documentation, not run-values.

Authority: OBSERVABILITY only (score-neutral, read-only). The frontier pointer
(contest-CPU 0.19110) is UNMOVED by anything here. Every entry point is fail-open
(returns a partial/empty model, never raises) because the consumer is a
load-bearing multi-day live dashboard.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

__all__ = [
    "MECHANISM_LANES",
    "R1_BYTECLOSE_JSON",
    "SCHEDULE_PROVENANCE_LINE",
    "build_curriculum_panel_model",
    "build_pose_readiness_model",
    "read_mechanism_event_states",
]

# ── The provenance one-liner (documentation text; answers the cargo-cult question
# on the panel itself). NOT a run-value — the schedule design lineage. ──
SCHEDULE_PROVENANCE_LINE = (
    "schedule: derived (#302/#315/#286; PR95 residue: 3 gradient-forms "
    "retained-by-measurement — tau_softplus top-2 L_tau · Muon -32%; "
    "removed-by-measurement: l7, smooth, QAT/c1a/lambda/sigma)"
)

# ── The #238 R1 byte-close artifact (machine-readable) — the BANKED pose custody. ──
R1_BYTECLOSE_JSON = "reports/r1_dxi_238/n600_shipdxi.json"

# ── Event-gated MECHANISM LANE specs. Each lane's cap + trigger are READ from the
# named launch.sh flags (never hardcoded here); the descriptive name/prose is
# documentation. ``armed_stage`` / ``fired_stage`` are the trainer log ``stage``
# rows whose appearance means the lane armed / engaged (see the trainer's emitted
# rows). ``gated`` lanes are condition-gated (no epoch cap flag) — their trigger is
# the condition, and "cap" is honestly None.
MECHANISM_LANES: tuple[dict, ...] = (
    {"key": "lane_band", "name": "lane-band",
     "trigger_flag": "lane-band-start-event", "cap_flag": "lane-band-start-epoch",
     "enabled_flag": "lane-render-band",
     "armed_stage": "lane_render_band", "fired_stage": "lane_render_band_engage"},
    {"key": "chroma_boundary", "name": "chroma-boundary",
     "trigger_flag": "seg-chroma-boundary-start-event",
     "cap_flag": "seg-chroma-boundary-start-epoch",
     "enabled_flag": "seg-chroma-boundary-weight",
     "armed_stage": "seg_chroma_boundary", "fired_stage": "seg_chroma_boundary_engage"},
    {"key": "temporal_screw", "name": "temporal-screw",
     "trigger_flag": "seg-temporal-screw-start-event",
     "cap_flag": "seg-temporal-screw-start-epoch",
     "enabled_flag": "seg-temporal-screw-weight",
     "armed_stage": "seg_temporal_screw", "fired_stage": "seg_temporal_screw_engage"},
    {"key": "birth_completion", "name": "birth-completion",
     "trigger_prose": "tau-persist + area-band condition", "cap_flag": None,
     "enabled_flag": "birth-completion-event", "gated": True,
     "armed_stage": "birth_completion_setup", "fired_stage": "birth_completion_ramp"},
    {"key": "ladder", "name": "LADDER island homotopy",
     "trigger_prose": "island-birth radius ramps (lane/movable)", "cap_flag": None,
     "enabled_flag": "ladder-island-homotopy", "gated": True,
     "armed_stage": "ladder_island_homotopy", "fired_stage": "ladder_rung"},
    {"key": "tail", "name": "TAIL turnpike cycles",
     "trigger_prose": "turnpike dwell cycle (marginal-S gated)",
     "cap_flag": "tail-start-epoch", "enabled_flag": "tail-cycles-max",
     "armed_stage": "tail_controller_armed", "fired_stage": "tail_cycle_begin"},
    {"key": "polyak", "name": "Polyak finisher arm",
     "trigger_prose": "iterate-averaging finisher arm",
     "cap_flag": "polyak-finisher-start-epoch", "enabled_flag": "polyak-finisher-arm",
     "armed_stage": "polyak_finisher_armed", "fired_stage": None},
    {"key": "terminal_solve", "name": "terminal solve stack (pose-finish)",
     "trigger_flag": "pose-finish-engage-on", "cap_flag": "pose-finish-start-epoch",
     "enabled_flag": "pose-finish-start-epoch",
     "armed_stage": "pose_finish_armed", "fired_stage": "pose_finish_engage"},
)

# stage rows scanned for mechanism-lane state (union of all armed/fired stages +
# the two curriculum/optimizer boundary rows the stages themselves fire on).
_LANE_STAGES: frozenset[str] = frozenset(
    [s for lane in MECHANISM_LANES for s in (lane.get("armed_stage"), lane.get("fired_stage")) if s]
    + ["muon_finisher_switch", "curriculum_transition_fired"])

_TAIL_BYTES = 524_288  # bounded run.log tail — O(1) in file size for the multi-day run


def _row_epoch(d: dict):
    """A stage row's epoch (rows use ``epoch`` or the terser ``ep``). None when absent."""
    for k in ("epoch", "ep"):
        v = d.get(k)
        if isinstance(v, (int, float)):
            return int(v)
    return None


def read_mechanism_event_states(log_paths, tail_bytes: int = _TAIL_BYTES) -> dict:
    """Latest fired/armed epoch of every mechanism-lane stage row from the run-log
    TAIL(s). Returns ``{stage_name: {"epoch": int}}`` (latest wins). Bounded tail
    read (the log grows to MBs over 3000 epochs). READ-ONLY, fail-open to ``{}`` so a
    parse error never kills the daemon — a lane with no row is then honestly
    'pending'."""
    out: dict = {}
    for lp in log_paths or ():
        try:
            p = Path(lp)
            size = p.stat().st_size
            with p.open("rb") as fh:
                if size > tail_bytes:
                    fh.seek(size - tail_bytes)
                    fh.readline()  # drop the clipped partial line
                block = fh.read()
            text = block.decode("utf-8", "replace")
        except Exception:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            # cheap prefilter: the tail is dense with non-lane stages
            if '"stage"' not in line:
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            st = d.get("stage")
            if st in _LANE_STAGES:
                ep = _row_epoch(d)
                if ep is not None:
                    out[st] = {"epoch": ep}  # later line wins
    return out


def _int_flag(flags: dict, key: str | None):
    if key is None:
        return None
    v = flags.get(key)
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _lane_state(lane: dict, event_states: dict) -> dict:
    """Resolve a lane's live state from the emitted evidence: FIRED@ep if its
    engage row appeared, else ARMED@ep if its arm/setup row appeared, else PENDING.
    Honest: a lane with NO consumable row is 'pending (no event row yet)'."""
    fired = event_states.get(lane.get("fired_stage")) if lane.get("fired_stage") else None
    armed = event_states.get(lane.get("armed_stage")) if lane.get("armed_stage") else None
    if fired is not None:
        return {"status": "fired", "at_epoch": fired.get("epoch"),
                "evidence_stage": lane.get("fired_stage")}
    if armed is not None:
        return {"status": "armed", "at_epoch": armed.get("epoch"),
                "evidence_stage": lane.get("armed_stage")}
    return {"status": "pending", "at_epoch": None,
            "evidence_stage": None, "note": "no event row yet"}


def _lane_enabled(lane: dict, flags: dict) -> bool:
    ef = lane.get("enabled_flag")
    if ef is None:
        return True
    v = flags.get(ef)
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    # numeric weights: enabled iff nonzero; tail-cycles-max: enabled iff > 0
    try:
        return float(v) != 0.0
    except (TypeError, ValueError):
        return True  # a present non-numeric flag (e.g. a mode string) counts as on


def build_curriculum_panel_model(readback, flags, event_states) -> dict:
    """The CURRICULUM POSITION panel model, DERIVED from the schedule read-back +
    the parsed launch.sh flags + the emitted event evidence.

    ``readback``: the :class:`ScheduleReadback` ``to_dict()`` (or the object).
    ``flags``: parsed launch.sh flags (keys sans ``--``, hyphenated).
    ``event_states``: :func:`read_mechanism_event_states` output.

    Returns ``{ok, event_triggered, epochs, stages, tau_anneal, lanes, provenance}``
    — the epochs/caps all sourced from the read-back/flags (no hardcoded literal).
    Fail-open: a bad input yields ``{ok: False, reason}`` with the provenance line."""
    model: dict = {"ok": False, "provenance": SCHEDULE_PROVENANCE_LINE,
                   "stages": [], "lanes": [], "tau_anneal": {}}
    try:
        rb = readback.to_dict() if hasattr(readback, "to_dict") else dict(readback or {})
        flags = flags or {}
        event_states = event_states or {}
        epochs = rb.get("epochs")
        model["epochs"] = epochs
        model["eval_every"] = rb.get("eval_every")
        model["event_triggered"] = bool(rb.get("event_triggered"))

        # ── STAGES as EVENTS (cap = fail-safe). CE is the anneal origin; tau/Muon
        # carry their trigger + hard-ceiling cap. Muon's event trigger lives in the
        # launch flag (the read-back keeps Muon 'fixed' because the trainer's Muon
        # event-switch is a cap+event hybrid) — surface it here. ──
        muon_evt = flags.get("muon-start-event")
        for st in rb.get("stages", []):
            name = st.get("name")
            entry = {"name": name, "mode": st.get("mode", "fixed"),
                     "status": st.get("status", "scheduled"),
                     "start": st.get("start"), "cap": st.get("cap"),
                     "fired_epoch": st.get("fired_epoch"),
                     "trigger": st.get("trigger")}
            if name == "Muon":
                # cap = the fixed start (the hard ceiling); trigger from the launch flag.
                entry["cap"] = st.get("start") if st.get("cap") is None else st.get("cap")
                if muon_evt:
                    entry["mode"] = "event"
                    entry["trigger"] = f"{muon_evt} event"
                    sw = event_states.get("muon_finisher_switch")
                    if sw is not None:
                        entry["status"] = "fired"
                        entry["fired_epoch"] = sw.get("epoch")
                    elif entry.get("status") not in ("fired",):
                        entry["status"] = "pending"
                    entry["start"] = None if entry["status"] != "fired" else sw.get("epoch")
            elif name == "tau":
                # corroborate a pending tau with the fired transition row if present.
                tr = event_states.get("curriculum_transition_fired")
                if tr is not None and entry.get("status") != "fired":
                    entry["status"] = "fired"
                    entry["fired_epoch"] = tr.get("epoch")
                    entry["start"] = tr.get("epoch")
            model["stages"].append(entry)

        # ── tau-path as ONE continuous anneal (CE = tau=1 limit). Span ends where the
        # finishing optimizer begins (Muon cap) or end-of-run. Params from flags. ──
        muon_cap = None
        for st in model["stages"]:
            if st["name"] == "Muon":
                muon_cap = st.get("cap") if st.get("cap") is not None else st.get("start")
        model["tau_anneal"] = {
            "shape": flags.get("tau-anneal-shape"),
            "temp_start": _flt(flags.get("softmax-temp-start")),
            "temp_end": _flt(flags.get("softmax-temp-end")),
            "softplus_tau": _flt(flags.get("tau-softplus-tau")),
            "span_start_epoch": 0,
            "span_end_epoch": muon_cap if muon_cap is not None else epochs,
            "ce_limit_note": "CE = tau=1 limit (one continuous geometric anneal)",
        }

        # ── MECHANISM LANES (event-gated engagements). ──
        for lane in MECHANISM_LANES:
            if not _lane_enabled(lane, flags):
                continue
            trigger = (flags.get(lane["trigger_flag"]) if lane.get("trigger_flag")
                       else lane.get("trigger_prose"))
            cap = _int_flag(flags, lane.get("cap_flag"))
            state = _lane_state(lane, event_states)
            model["lanes"].append({
                "key": lane["key"], "name": lane["name"],
                "trigger": trigger, "cap": cap,
                "cap_kind": ("condition-gated" if lane.get("gated") else "fail-safe cap"),
                "status": state["status"], "at_epoch": state["at_epoch"],
                "evidence_stage": state.get("evidence_stage"),
                "note": state.get("note"),
            })
        model["ok"] = True
    except Exception as exc:
        model["reason"] = f"curriculum-panel build failed: {exc}"
    return model


def _flt(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def build_pose_readiness_model(flags, event_states, sensors,
                               r1_path: str | Path = R1_BYTECLOSE_JSON) -> dict:
    """The POSE-DESCENT READINESS model: the BANKED R1 fallback card + the fallback
    contract state (detector mode / state / decision tree).

    The R1 numbers are READ from the #238 byte-close artifact JSON (value-provenance:
    never hardcoded); the pose contribution is DERIVED as ``sqrt(10*d_pose)``. The
    detector mode is the ``--pose-finish-engage-on`` flag; the detector state derives
    from the pose_finish evidence rows + the jacobian_basin sensor. Fail-open:
    a missing artifact yields ``banked_r1: {ok: False}`` (the panel still shows the
    contract + honest pending)."""
    flags = flags or {}
    event_states = event_states or {}
    sensors = sensors or {}

    # ── 1) BANKED R1 artifact card (read from JSON; contribution derived). ──
    banked: dict = {"ok": False}
    try:
        d = json.loads(Path(r1_path).read_text(errors="replace"))
        parity = d.get("parity_on_inflated_frames") or {}
        bc = d.get("byte_close") or {}
        dpose = parity.get("d_pose_realized_on_inflated")
        if dpose is None:
            dpose = (d.get("pose_carrier_confirmation") or {}).get(
                "d_pose_carrier_warp_f0_witness_f1")
        xi_bytes = bc.get("pose_carrier_counted_bytes")
        contribution = math.sqrt(10.0 * float(dpose)) if isinstance(dpose, (int, float)) else None
        banked = {
            "ok": dpose is not None,
            "d_pose": dpose,
            "contribution": contribution,          # DERIVED sqrt(10*d_pose)
            "xi_bytes": xi_bytes,                   # xi_eff / dxi section (COUNTED)
            "n_pairs": parity.get("pairs_scored") or d.get("n_pairs_total"),
            "authority": d.get("authority") or "[macOS-CPU advisory] NON-PROMOTABLE",
            "promotable": bool(d.get("promotion_claim", False)),
            "source": str(r1_path),
        }
    except Exception as exc:
        banked = {"ok": False, "reason": f"R1 artifact unreadable: {exc}",
                  "source": str(r1_path)}

    # ── 2) fallback contract state. ──
    detector_mode = flags.get("pose-finish-engage-on")  # e.g. sigma_min_plateau
    pose_finish_cap = _int_flag(flags, "pose-finish-start-epoch")
    engage = event_states.get("pose_finish_engage")
    armed = event_states.get("pose_finish_armed")
    jb = sensors.get("jacobian_basin") or {}
    # detector state precedence: fired > degenerate(basin unidentifiable) > armed > pending.
    if engage is not None:
        det_state, det_at = "fired", engage.get("epoch")
    elif armed is not None:
        det_state, det_at = "armed", armed.get("epoch")
    else:
        det_state, det_at = "pending", None
    # DEGENERATE flag: basin probe present but sigma below floor / would-not-fire.
    degenerate = False
    if jb:
        sm, floor = jb.get("median_sigma_min"), jb.get("sigma_floor")
        if isinstance(sm, (int, float)) and isinstance(floor, (int, float)) and sm <= floor:
            degenerate = True
    # should_ship_banked_r1: read from telemetry if the trainer emits it; else honest None.
    ssbr = None
    for src in (engage, armed):
        if isinstance(src, dict) and "should_ship_banked_r1" in src:
            ssbr = bool(src["should_ship_banked_r1"])
            break

    contract = {
        "detector_mode": detector_mode,
        "detector_cap": pose_finish_cap,             # fail-safe cap epoch (from flag)
        "detector_state": det_state,
        "detector_at_epoch": det_at,
        "degenerate": degenerate,
        "should_ship_banked_r1": ssbr,
        "basin_probe_present": bool(jb),
        "decision_tree": ("detector fires -> joint pose finish · "
                          "degenerate/never-fires -> ship banked R1 · "
                          "regression -> rollback"),
        # honest pending copy while pose is terminal (no live pose telemetry yet).
        "pending_note": "joint finish: awaiting conditioning (first sigma rows >= ~ep400)",
    }
    return {"ok": True, "banked_r1": banked, "contract": contract,
            "pose_blind_by_design": True}
