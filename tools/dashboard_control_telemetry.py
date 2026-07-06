#!/usr/bin/env python3
"""Control-system telemetry + lever/config panels for the level-set witness dashboards.

DASHBOARD PASS 2026-07-04 (operator: "enhance and extend to reflect new levers and how
they're working or not and control system telemetry"). One module, pure + testable fns
(NO IO in the pure core; one small guarded log-reader wrapper), consumed by BOTH:

  * ``tools/render_levelset_dashboard.py`` — the static self-contained index.html
    (panels + the extra PNG rows), and
  * ``tools/dashboard_server.py`` — automatically, via the pre-rendered
    ``rld._run_info_html`` strip it already ships in META (``#runinfostrip`` on the
    DESCENT tab). All panel HTML is INLINE-STYLED so both surfaces render identically
    with zero server CSS/tab changes (the Yousfi-arc tab structure is untouched).

What it reads (ALL real trainer telemetry — grep-verified emission sites in
``experiments/train_levelset_witness_realized_through_R_mlx.py``; NEVER fabricated):

  * ``{"stage":"verdict",...}``                     — d_seg/d_pose/ep_loss/seg_form rows
  * ``{"stage":"curriculum_transition_fired",...}`` — event-triggered boundary (build-2;
      carries ``from``/``to``/``epoch``/``trigger`` ∈ {loss_plateau, cap})
  * ``{"stage":"curriculum_transition",...}``       — LEGACY hardcoded boundary (#205
      emits this; ``from_seg_form``/``to_seg_form``; rendered trigger="hardcoded")
  * ``{"stage":"closed_loop",...}``                 — build-3 controller decisions
      (classification / d_seg_slope / eikonal_bump=cumulative bump_add / action)
  * ep0 ``{"stage":"structured_init",...}``         — ``part_frac`` dict + ``lane_px``
      (+ ``roles`` for the lane class index) — THE nucleation acceptance gate
  * ``{"stage":"lane_prior_phi1",...}``             — seed mechanism ``mode``
      (paint vs replace — replace measured NO-OP on #205)

The eikonal EFFECTIVE weight is DERIVED (not logged directly): a verbatim pure replica
of the trainer's ``_scheduled_eikonal_weight`` (base → cosine-eased STEP to end at the
tau/MCF onset over the rewarmup window; the step tracks the FIRED tau boundary when
event-triggering is ON) composed with the build-3 bump via the trainer's
``_cl_effective_eikonal`` contract ``min(scheduled+bump, max(cl_max, scheduled))``.

Classification math is NOT forked: the monitor-replay lane calls
``tools/witness_control_monitor.classify_trajectory`` directly (the canonical
classifier); live ``closed_loop`` rows win when present (the trainer's in-run replica
is parity-regression-guarded against the same monitor).

AUTHORITY: everything rendered is [macOS-MLX advisory · NON-PROMOTABLE]; the frontier
pointer is contest-CPU 0.19110 and UNMOVED — no panel here is a score claim.
"""
from __future__ import annotations

import html as _html
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import witness_control_monitor as wcm  # noqa: E402  (canonical classifier — never forked)

# ── palette (mirrors render_levelset_dashboard's dark theme) ──
_FG = "#d8dde6"
_MUTED = "#8b93a3"
_PANEL = "#1b1e24"
_GRIDC = "#2c313b"
_GOOD = "#46d369"
_WARN = "#e6cf7a"
_BAD = "#ff6b6b"
_ACC = "#5ab0ff"
_VOLA = "#c08cff"

#: classification → display color (the closed-loop lane color code)
CLASS_COLORS: dict[str, str] = {
    wcm.CONVERGING: _GOOD,
    wcm.PLATEAU: _ACC,
    wcm.TRANSITION_TRANSIENT: _WARN,
    wcm.DIVERGING_ERASING: _BAD,
    wcm.VOLATILE: _VOLA,
}

_ADVISORY = "[macOS-MLX advisory · NON-PROMOTABLE] · pointer 0.19110 UNMOVED"


# ─────────────────────────── parsing (pure) ───────────────────────────
def parse_stage_rows(lines) -> dict:
    """Extract the control-system telemetry rows from raw run.log lines (pure).

    Returns ``{"transitions": [...], "closed_loop": [...], "structured_init": dict|None,
    "lane_prior_phi1": dict|None}``. Transitions merge the NEW event-triggered
    ``curriculum_transition_fired`` rows (trigger loss_plateau|cap) with the LEGACY
    hardcoded ``curriculum_transition`` rows (#205 emits these; normalized to the same
    shape with ``trigger="hardcoded"``), de-duplicated on (epoch, to), sorted by epoch.
    Tolerant of every malformed line (a dashboard reader must never crash)."""
    transitions: list[dict] = []
    closed_loop: list[dict] = []
    structured_init: dict | None = None
    lane_prior: dict | None = None
    island_seed: dict | None = None
    for raw in lines:
        s = raw.strip()
        if '"stage"' not in s:
            continue
        try:
            d = json.loads(s)
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        st = d.get("stage")
        if st == "curriculum_transition_fired":
            transitions.append({"epoch": d.get("epoch"), "from": d.get("from"),
                                "to": d.get("to"), "trigger": d.get("trigger", "?")})
        elif st == "curriculum_transition":  # legacy hardcoded boundary (#205 path)
            transitions.append({"epoch": d.get("epoch"), "from": d.get("from_seg_form"),
                                "to": d.get("to_seg_form"), "trigger": "hardcoded"})
        elif st == "closed_loop":
            closed_loop.append(d)
        elif st == "structured_init":
            structured_init = d
        elif st == "lane_prior_phi1":
            lane_prior = d
        elif st == "island_seed":
            island_seed = d
    # de-dup (an event-fired row + a legacy row for the same boundary): fired wins.
    seen: dict[tuple, dict] = {}
    for t in transitions:
        key = (t.get("epoch"), t.get("to"))
        if key not in seen or seen[key]["trigger"] == "hardcoded":
            seen[key] = t
    out_tr = sorted(seen.values(), key=lambda t: (t.get("epoch") is None, t.get("epoch") or 0))
    closed_loop.sort(key=lambda r: r.get("epoch") or 0)
    return {"transitions": out_tr, "closed_loop": closed_loop,
            "structured_init": structured_init, "lane_prior_phi1": lane_prior,
            "island_seed": island_seed}


# ── nucleation-gate MEASURED constants (provenance in each comment) ────────────────────────
# Lane pixel fraction of the frame (measured: 0.59-0.64% across n600; nucleation memory +
# FEED-04x). A witness with ZERO lane scores d_seg >= this bound (all lane pixels wrong), so a
# verdict BELOW it PROVES partial lane presence — the birth-confirmation pixel argument.
LANE_FRAC_BOUND = 0.0059
# Paint-signal discriminator on structured_init's pretrain disagree: a paint-carrying target the
# pretrain hasn't absorbed leaves disagree ~= the lane band (fresh run MEASURED 0.00589); the #205
# replace-no-op (lane-LESS target, fit near-perfectly) measured 0.00035. 0.003 sits ~2x/8x between.
PAINT_SIGNAL_MIN_DISAGREE = 0.003


def nucleation_gate(structured_init: dict | None, lane_prior: dict | None = None,
                    island_seed: dict | None = None,
                    verdicts: list[dict] | None = None) -> dict:
    """The ep0 NUCLEATION ACCEPTANCE GATE — CORRECTED semantics (FEED-04x, 2026-07-04).

    ``part_frac`` in the structured_init row is the WITNESS partition, which starts at 0 in
    ANY run (including a perfectly seeded one) — so part_frac[lane]==0 at init is NOT by
    itself a failure; rendering it as FAIL was the pre-correction bug (a wrong verdict shown
    with high confidence — worse than 'insufficient data'). The honest, fully MEASURED
    falling-rule states:

      * BIRTH_CONFIRMED — any verdict d_seg < LANE_FRAC_BOUND (0.0059): a lane-less witness
        CANNOT score below the lane pixel fraction, so this PROVES partial lane presence.
      * PASS — part_frac[lane] > 0 at init (the witness partition itself starts seeded).
      * SEEDED_WATCH — init partition is 0 BUT the seeding MECHANISM is measured-present:
        paint mode with band px > 0 AND pretrain disagree >= PAINT_SIGNAL_MIN_DISAGREE
        (the paint IS in the target as a standing signal), and/or an island_seed row with
        support > 0 (the render-level seed #205 lacked). Birth watch armed: CE verdicts
        must out-descend #205's CE trace by ep50-75.
      * FAIL — a seeding MECHANISM was CONFIGURED (lane-prior mode set / island_seed row) but
        is inert: the true #205 signature (replace-no-op target fit near-perfectly, paint band
        produced no in-target signal). A configured mechanism that does nothing is a real defect.
      * UNSEEDED_WATCH — structured_init present, part_frac[lane]==0, and NO seeding mechanism
        configured AT ALL (the council-approved clean baseline deliberately dropped the measured
        no-op lane-prior). This is NOT a failure: lane birth is deferred to CE TRAINING, not an
        init seed. Honest watch (neither pass nor fail): verdicts must descend to birth.
      * UNMEASURED — no structured_init row yet.

    Pure; never raises. Returns the state + every measured input it judged from."""
    mode = None
    band_px = None
    if isinstance(lane_prior, dict):
        mode = lane_prior.get("mode")
        band_px = lane_prior.get("lane_band_px")
    seed_support = None
    if isinstance(island_seed, dict):
        try:
            seed_support = float(island_seed.get("mean_support_frac"))
        except (TypeError, ValueError):
            seed_support = None
    # birth confirmation from verdicts (usable even before structured_init parses)
    birth_ep, birth_dseg = None, None
    for v in (verdicts or []):
        try:
            ds = float(v.get("d_seg"))
        except (TypeError, ValueError):
            continue
        if ds < LANE_FRAC_BOUND and (birth_dseg is None or ds < birth_dseg):
            birth_ep, birth_dseg = v.get("epoch"), ds
    base = {"lane_px": None, "lane_cls": None, "mode": mode, "lane_band_px": band_px,
            "seed_support": seed_support, "pretrain_disagree": None,
            "birth_epoch": birth_ep, "birth_d_seg": birth_dseg, "lane_part_frac": None}
    if not isinstance(structured_init, dict):
        state = "BIRTH_CONFIRMED" if birth_dseg is not None else "UNMEASURED"
        return {"state": state, **base}
    roles = structured_init.get("roles") or {}
    lane_cls = roles.get("lane", 1)
    pf = structured_init.get("part_frac") or {}
    lane_frac = pf.get(str(lane_cls), pf.get(lane_cls))
    lane_px = structured_init.get("lane_px")
    try:
        disagree = float(structured_init.get("pretrain_direct_argmax_disagree_vs_part"))
    except (TypeError, ValueError):
        disagree = None
    base.update({"lane_px": lane_px, "lane_cls": lane_cls, "pretrain_disagree": disagree,
                 "lane_part_frac": (float(lane_frac) if lane_frac is not None else None)})
    paint_signal = (mode == "paint" and (band_px or 0) > 0
                    and disagree is not None and disagree >= PAINT_SIGNAL_MIN_DISAGREE)
    seed_present = seed_support is not None and seed_support > 0.0
    # was a seeding MECHANISM even configured? mode set (replace/paint/add/…) OR an island_seed
    # row exists. Distinguishes the two part_frac[lane]==0 cases the old single FAIL conflated:
    #   mechanism_configured=True  → #205 signature (replace-no-op fit perfectly / paint inert) = FAIL
    #   mechanism_configured=False → clean baseline (lane-prior deliberately dropped) = UNSEEDED_WATCH
    mechanism_configured = mode is not None or seed_support is not None
    if birth_dseg is not None:
        state = "BIRTH_CONFIRMED"
    elif lane_frac is not None and float(lane_frac) > 0.0:
        state = "PASS"
    elif paint_signal or seed_present:
        state = "SEEDED_WATCH"
    elif lane_frac is None:
        state = "UNMEASURED"
    elif mechanism_configured:
        state = "FAIL"
    else:
        state = "UNSEEDED_WATCH"
    return {"state": state, **base}


# ─────────────── eikonal effective-weight derivation (pure trainer replicas) ───────────────
def scheduled_eikonal_weight(ep: int, *, base: float, end: float | None,
                             step_epoch: int | None, ease_epochs: int,
                             curriculum: bool = True) -> float:
    """Verbatim pure replica of the trainer's ``_scheduled_eikonal_weight`` math:
    constant ``base`` unless ``end`` is set != base (and curriculum is on); then a STEP
    base→end at ``step_epoch`` (the tau/MCF onset), cosine-eased over ``ease_epochs``.
    ``step_epoch`` None/<=0 → base (unfired-sentinel behavior)."""
    import math
    if end is None:
        return float(base)
    end = float(end)
    if end == float(base) or not curriculum:
        return float(base)
    if step_epoch is None or int(step_epoch) <= 0 or ep < int(step_epoch):
        return float(base)
    step_ep = int(step_epoch)
    ease = int(ease_epochs or 0)
    if ease <= 0 or ep >= step_ep + ease:
        return end
    frac = (ep - step_ep) / float(ease)
    w = 0.5 * (1.0 - math.cos(math.pi * frac))
    return float(base) + (end - float(base)) * w


def effective_eikonal_series(max_epoch: int, flags: dict, transitions: list[dict],
                             closed_loop: list[dict]) -> list[tuple[int, float]]:
    """Effective eikonal weight per epoch 0..max_epoch: the build-1 schedule (the step
    tracks the FIRED tau boundary when one exists in ``transitions``; else the hardcoded
    ``--tau-softplus-start-epoch``) composed with the build-3 cumulative bump via the
    trainer's ``_cl_effective_eikonal`` contract ``min(sched+bump, max(cl_max, sched))``.
    Pure; ``flags`` is the parsed launch.sh dict."""
    def _f(key, default=None):
        v = flags.get(key)
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    base = _f("eikonal-weight", 0.01)
    end = _f("eikonal-weight-end", None)
    ease = int(_f("stage-transition-rewarmup-epochs", 0) or 0)
    curriculum = "curriculum" in flags
    cl_max = _f("closed-loop-eikonal-max", 0.20)
    # step epoch: the FIRED tau boundary wins (event-triggered); hardcoded fallback.
    step_epoch = None
    for t in transitions:
        if t.get("to") == "tau_softplus" and isinstance(t.get("epoch"), (int, float)):
            step_epoch = int(t["epoch"])
            break
    if step_epoch is None:
        hc = _f("tau-softplus-start-epoch", None)
        step_epoch = int(hc) if hc is not None else None
    # cumulative bump timeline: latest closed_loop row at-or-before ep wins.
    bumps = [(int(r["epoch"]), float(r.get("eikonal_bump") or 0.0))
             for r in closed_loop if isinstance(r.get("epoch"), (int, float))]
    bumps.sort()
    out: list[tuple[int, float]] = []
    bi = 0
    bump_now = 0.0
    for ep in range(0, max(int(max_epoch), 0) + 1):
        while bi < len(bumps) and bumps[bi][0] <= ep:
            bump_now = bumps[bi][1]
            bi += 1
        w = scheduled_eikonal_weight(ep, base=base, end=end, step_epoch=step_epoch,
                                     ease_epochs=ease, curriculum=curriculum)
        if bump_now > 0.0:
            w = min(w + bump_now, max(cl_max if cl_max is not None else w, w))
        out.append((ep, w))
    return out


# ─────────────── per-stage slopes + the classification lane (pure) ───────────────
def per_stage_slopes(verdicts: list[dict]) -> list[dict]:
    """Per-stage d_seg slope (the Lyapunov dV/dt) ALONGSIDE the ep_loss slope, so the
    surrogate↔verdict DECOUPLING (#205's failure: d_seg RISING while ep_loss FALLS) is
    visible at a glance. Groups verdict rows by ``seg_form`` in first-appearance order
    (rows without seg_form — the ep0 priming verdict — are skipped). Uses the monitor's
    canonical least-squares slope. Pure."""
    groups: dict[str, list[dict]] = {}
    order: list[str] = []
    for v in verdicts:
        sf = v.get("seg_form")
        if not sf or not isinstance(v.get("d_seg"), (int, float)):
            continue
        if sf not in groups:
            groups[sf] = []
            order.append(sf)
        groups[sf].append(v)
    out = []
    for sf in order:
        g = groups[sf]
        eps = [float(v["epoch"]) for v in g]
        d_slope = wcm._lstsq_slope(eps, [float(v["d_seg"]) for v in g])
        losses = [(float(v["epoch"]), float(v["ep_loss"])) for v in g
                  if isinstance(v.get("ep_loss"), (int, float))]
        l_slope = wcm._lstsq_slope([e for e, _ in losses], [x for _, x in losses])
        decoupled = bool(len(g) >= 3 and d_slope > 0.0 and l_slope < 0.0)
        out.append({"stage": sf, "n": len(g), "ep_lo": int(min(eps)), "ep_hi": int(max(eps)),
                    "d_seg_slope": d_slope, "ep_loss_slope": l_slope,
                    "decoupled": decoupled})
    return out


def classification_lane(verdicts: list[dict], closed_loop: list[dict],
                        *, window: int = 5, min_sustained_windows: int = 3) -> list[dict]:
    """The closed-loop lane: one classification per eval point. LIVE ``closed_loop``
    rows win (the fresh run's build-3 telemetry, with actions). Without them (#205)
    the lane is REPLAYED through the canonical monitor classifier on verdict prefixes
    (``source="monitor_replay"``) so the #205 failure is legible retrospectively.
    Never forks the classifier math. Pure."""
    if closed_loop:
        return [{"epoch": int(r["epoch"]), "classification": r.get("classification"),
                 "action": r.get("action"), "eikonal_bump": r.get("eikonal_bump"),
                 "d_seg_slope": r.get("d_seg_slope"), "source": "closed_loop"}
                for r in closed_loop if isinstance(r.get("epoch"), (int, float))]
    usable = [v for v in verdicts
              if v.get("seg_form") and isinstance(v.get("d_seg"), (int, float))]
    out: list[dict] = []
    for i in range(2, len(usable) + 1):
        try:
            cv = wcm.classify_trajectory(usable[:i], window=window,
                                         min_sustained_windows=min_sustained_windows)
        except (ValueError, KeyError):
            continue
        out.append({"epoch": cv.epoch_latest, "classification": cv.classification,
                    "action": None, "eikonal_bump": None,
                    "d_seg_slope": cv.d_seg_slope_per_ep, "source": "monitor_replay"})
    return out


# ─────────────── run role (config currency) + the lever ledger ───────────────
#: #205 measured diagnosis (transition-analysis memo + FEED-04i; all MEASURED on the real log)
_205_DIAGNOSIS = ("DIAGNOSED erosion run: ep0 part_frac[lane]=0 (nucleation FAIL, "
                  "seed mode=replace NO-OP) · tau-creep +4.68e-5/ep onset → steady "
                  "+6.0e-6/ep, net +40.4% over CE-best by ep450 WHILE ep_loss fell "
                  "148→130 (surrogate↔verdict decoupling) — will NOT reach goal at "
                  "current trajectory [measured]")
_FRESH_HEADLINE = ("FRESH SEEDED RUN (successor): 13-change calibrated config, "
                   "control system ON (paint-seed · eikonal step-ramp · event-triggered "
                   "curriculum · closed loop)")

#: markers whose presence in launch.sh flags identifies the fresh seeded config
_FRESH_MARKER_FLAGS = ("seed-islands", "eikonal-weight-end", "curriculum-event-triggered",
                       "closed-loop-control", "film-stiefel", "muon-warm-start-momentum")


def run_role(flags: dict) -> dict:
    """Classify the watched run's CONFIG ROLE so the banner reflects current reality:
    ``fresh`` (the seeded successor — ≥2 fresh markers present), ``205`` (the diagnosed
    erosion config: seed mode=replace + eikonal 0.01 flat), ``unknown``, or ``none``
    (no flags parsed). Pure."""
    if not flags:
        return {"role": "none", "headline": "", "markers": {}}
    markers = {k: (k in flags) for k in _FRESH_MARKER_FLAGS}
    markers["paint-seed"] = (flags.get("lane-prior-phi1-mode") == "paint")
    n_fresh = sum(1 for v in markers.values() if v)
    if n_fresh >= 2:
        return {"role": "fresh", "headline": _FRESH_HEADLINE, "markers": markers}
    if (flags.get("lane-prior-phi1-mode") == "replace"
            and str(flags.get("eikonal-weight")) in ("0.01", "0.010")
            and "eikonal-weight-end" not in flags):
        return {"role": "205", "headline": _205_DIAGNOSIS, "markers": markers}
    return {"role": "unknown", "headline": "", "markers": markers}


#: THE PRIMARY LEVER TABLE (master ledger §1, fresh_run_master_lever_ledger_20260704.md).
#: rows: (lever label, flag keys to read from the run, #205 value, fresh value, grounding)
LEVER_LEDGER: tuple[tuple[str, tuple[str, ...], str, str, str], ...] = (
    ("--seed-islands", ("seed-islands",), "OFF", "ON",
     "the nucleation fix; amplify was a no-op without it (sweep-A #1)"),
    ("seed mechanism", ("lane-prior-phi1-mode",), "replace (NO-OP)", "paint-then-SDF (#291)",
     "paint: part_frac[lane] 0→0.0064, lane_FN 0.00713→0.00211 [measured]"),
    ("--island-dilate-px", ("island-dilate-px",), "1", "1 (NOT 2)",
     "+2px FP-costly ~15:1 — survival gained < FP lost [measured facet-3]"),
    ("--eikonal-weight (+end)", ("eikonal-weight", "eikonal-weight-end"), "0.01 flat",
     "0.05 → step 0.10 @ tau/MCF onset",
     "survival knee: σ0.8/93% vs σ1.5/49% as MCF narrows [measured]"),
    ("--length-weight", ("length-weight",), "0.001", "0.001 (KEEP)",
     "it IS the MCF surface-tension erosion term — do NOT raise"),
    ("--tau-anneal-shape / --softmax-temp-end", ("tau-anneal-shape", "softmax-temp-end"),
     "cosine / 0.05", "geometric / 1.0",
     "τ_end=1.0 MEASURED resolution floor (0.05 = 20× sub-pixel aliasing)"),
    ("--mod-dim", ("mod-dim",), "32", "19",
     "TwoNN m≈7.15; Whitney 2m+1=15.3 < 19 → SAFE, no cliff; 32 = waste [measured]"),
    ("--bank-n-scales", ("bank-n-scales",), "4", "6",
     "full stem-Nyquist f_max=64 (facet-2; slope-gated)"),
    ("--film-stiefel", ("film-stiefel",), "OFF", "ON",
     "byte-free rank fix, PR 1.19→4.57 [measured facet-2]"),
    ("--muon-warm-start-momentum / --muon-lr-final-frac",
     ("muon-warm-start-momentum", "muon-lr-final-frac"), "OFF / 1.0", "ON / 0.1",
     "kills the Muon cold-start spike + escapes the flat-LR plateau (#270/facet-1)"),
    ("--lane-band-start-epoch + rewarmup",
     ("lane-band-start-epoch", "stage-transition-rewarmup-epochs",
      "stage-transition-rewarmup-shape"), "300 / 8-linear", "350 / 20-cosine",
     "deconflict the MEASURED ep300 collision (3.4× harm) [Ch.6]"),
    ("--curriculum-event-triggered", ("curriculum-event-triggered",),
     "OFF (hardcoded epochs)", "ON (fire on loss plateau, cap-bounded)",
     "dynamical epochs; deterministic — fired epochs are logged OUTPUTS (build-2)"),
    ("--closed-loop-control", ("closed-loop-control",), "OFF",
     "ON (bounded eikonal bump → early-stop+preserve)",
     "would have caught #205's erosion @ep325 instead of ep425 (build-3)"),
)


def lever_run_value(flags: dict, keys: tuple[str, ...]) -> str:
    """The run's OWN value for a lever: join the parsed launch.sh values for ``keys``
    (boolean flags render ON; absent flags render OFF for boolean-style keys, else —)."""
    parts: list[str] = []
    for k in keys:
        if k not in flags:
            parts.append("—")
            continue
        v = flags[k]
        parts.append("ON" if v is True else str(v))
    return " / ".join(parts)


# ─────────────── lever status board (live "how they're working or not") ───────────────
def lever_status_rows(flags: dict, control: dict, schedule: dict | None = None,
                      max_epoch: int | None = None) -> list[dict]:
    """Per-lever LIVE status once a run streams (pre-launch / awaiting-run states when
    telemetry is absent). Sources: run.log rows (via ``control`` = parse_stage_rows +
    nucleation) + the parsed launch.sh flags. Returns rows
    ``{lever, state ∈ ok|warn|bad|pending|off, value, detail}``. Pure."""
    sched = schedule or {}
    nuc = control.get("nucleation") or nucleation_gate(control.get("structured_init"),
                                                       control.get("lane_prior_phi1"))
    transitions = control.get("transitions") or []
    closed_loop = control.get("closed_loop") or []
    rows: list[dict] = []

    # 1) paint-seed (the CORRECTED gate, FEED-04x: mechanism + birth, not init-part_frac alone)
    mode = nuc.get("mode") or flags.get("lane-prior-phi1-mode") or "?"
    if nuc["state"] == "BIRTH_CONFIRMED":
        rows.append({"lever": "paint-seed", "state": "ok",
                     "value": f"BIRTH d_seg={nuc['birth_d_seg']:.4f}@ep{nuc.get('birth_epoch')}",
                     "detail": (f"mode={mode} · d_seg < lane-frac bound {LANE_FRAC_BOUND} PROVES "
                                f"partial lane presence (pixel arithmetic)")})
    elif nuc["state"] == "PASS":
        rows.append({"lever": "paint-seed", "state": "ok",
                     "value": f"part_frac[lane]={nuc['lane_part_frac']:.4f}",
                     "detail": f"mode={mode} · lane_px={nuc.get('lane_px')} · init partition seeded (measured)"})
    elif nuc["state"] == "SEEDED_WATCH":
        dis = nuc.get("pretrain_disagree")
        sup = nuc.get("seed_support")
        ev = " + ".join(filter(None, [
            (f"paint-signal disagree {dis:.4f}≈band" if dis is not None
             and dis >= PAINT_SIGNAL_MIN_DISAGREE else None),
            (f"seed support {sup:.3f}" if sup else None)]))
        rows.append({"lever": "paint-seed", "state": "warn",
                     "value": "SEEDED · birth watch",
                     "detail": (f"mode={mode} · init partition 0 (expected) but mechanism MEASURED "
                                f"present: {ev} · confirm = d_seg<{LANE_FRAC_BOUND}")})
    elif nuc["state"] == "FAIL":
        rows.append({"lever": "paint-seed", "state": "bad",
                     "value": "mechanism inert",
                     "detail": (f"mode={mode} · a seeding mechanism WAS configured but produced no "
                                f"paint-signal / seed support, partition 0 (the #205 replace-no-op signature)")})
    elif nuc["state"] == "UNSEEDED_WATCH":
        rows.append({"lever": "paint-seed", "state": "warn",
                     "value": "unseeded · birth via training",
                     "detail": (f"mode={mode} · NO init-seed mechanism by design (clean baseline — "
                                f"lane-prior no-op dropped); birth deferred to CE, watch d_seg<{LANE_FRAC_BOUND}")})
    else:
        rows.append({"lever": "paint-seed", "state": "pending", "value": "awaiting run",
                     "detail": f"mode={mode} · gate = mechanism present at ep0, birth = d_seg<{LANE_FRAC_BOUND}"})

    # 2) eikonal-ramp (build-1): base/end config; stepped yet?
    base = flags.get("eikonal-weight", "?")
    end = flags.get("eikonal-weight-end")
    if end is None:
        rows.append({"lever": "eikonal-ramp", "state": "off",
                     "value": f"flat {base}", "detail": "no --eikonal-weight-end (schedule OFF)"})
    else:
        step_ep = None
        for t in transitions:
            if t.get("to") == "tau_softplus":
                step_ep = t.get("epoch")
                break
        if step_ep is None:
            try:
                step_ep = int(flags.get("tau-softplus-start-epoch"))
            except (TypeError, ValueError):
                step_ep = None
        stepped = (max_epoch is not None and step_ep is not None and max_epoch >= step_ep)
        rows.append({"lever": "eikonal-ramp", "state": "ok" if stepped else "pending",
                     "value": f"{base}→{end}" + (f" @ep{step_ep}" if step_ep is not None else ""),
                     "detail": ("STEPPED (tau/MCF onset passed)" if stepped
                                else "armed; steps at the tau/MCF onset")})

    # 3) event-triggered curriculum (build-2): fired boundaries + trigger types
    if "curriculum-event-triggered" in flags:
        fired = [t for t in transitions if t.get("trigger") in ("loss_plateau", "cap")]
        if fired:
            desc = " · ".join(f"{t.get('to')}@ep{t.get('epoch')} ({t.get('trigger')})"
                              for t in fired)
            rows.append({"lever": "event-trigger", "state": "ok",
                         "value": f"{len(fired)} fired", "detail": desc})
        else:
            rows.append({"lever": "event-trigger", "state": "pending", "value": "armed",
                         "detail": "no boundary fired yet (fires on loss plateau, cap-bounded)"})
    else:
        hard = [t for t in transitions if t.get("trigger") == "hardcoded"]
        rows.append({"lever": "event-trigger", "state": "off",
                     "value": "OFF (hardcoded epochs)",
                     "detail": (" · ".join(f"{t.get('to')}@ep{t.get('epoch')}" for t in hard)
                                or "boundaries at the hardcoded epochs")})

    # 4) closed-loop (build-3): actions taken
    if "closed-loop-control" in flags:
        acts = [r for r in closed_loop if r.get("action") not in (None, "none")]
        bumps = max((int(r.get("bumps_used") or 0) for r in closed_loop), default=0)
        if closed_loop:
            last = closed_loop[-1]
            rows.append({"lever": "closed-loop",
                         "state": ("warn" if acts else "ok"),
                         "value": f"{len(acts)} action(s) · {bumps} bump(s)",
                         "detail": (f"last: {last.get('classification')} @ep{last.get('epoch')} "
                                    f"(bump_add={last.get('eikonal_bump')})")})
        else:
            rows.append({"lever": "closed-loop", "state": "pending", "value": "armed",
                         "detail": "no decision row yet (emits per eval)"})
    else:
        rows.append({"lever": "closed-loop", "state": "off", "value": "OFF",
                     "detail": "controller not enabled on this run"})

    # 5) muon warm-start: armed; fires at the muon boundary
    muon_start = sched.get("muon_start") or flags.get("muon-start-epoch")
    if "muon-warm-start-momentum" in flags:
        try:
            fired = (max_epoch is not None and muon_start is not None
                     and max_epoch >= int(muon_start))
        except (TypeError, ValueError):
            fired = False
        rows.append({"lever": "muon warm-start", "state": "ok" if fired else "pending",
                     "value": ("FIRED" if fired else "armed"),
                     "detail": f"fires at muon boundary ep{muon_start} · "
                               f"lr-final-frac {flags.get('muon-lr-final-frac', '1.0')}"})
    else:
        rows.append({"lever": "muon warm-start", "state": "off",
                     "value": "OFF (cold Muon)",
                     "detail": f"muon @ep{muon_start} starts with zero momentum (the #205 spike)"})

    # 6) static config levers (render values)
    rows.append({"lever": "mod-dim / bank-6 / stiefel", "state": "ok",
                 "value": (f"mod {flags.get('mod-dim', '?')} · scales "
                           f"{flags.get('bank-n-scales', '4')} · stiefel "
                           f"{'ON' if 'film-stiefel' in flags else 'OFF'}"),
                 "detail": "static config (measured-safe: mod-19 Whitney-gated; scales 6 = "
                           "full stem-Nyquist; stiefel = byte-free rank fix)"})
    return rows


# ─────────────── composition (one IO wrapper + one pure builder) ───────────────
def build_control(stage_rows: dict, flags: dict, verdicts: list[dict],
                  schedule: dict | None = None) -> dict:
    """Compose the full control-telemetry dict from parsed pieces (pure)."""
    nuc = nucleation_gate(stage_rows.get("structured_init"), stage_rows.get("lane_prior_phi1"),
                          island_seed=stage_rows.get("island_seed"), verdicts=verdicts)
    max_ep = max((int(v["epoch"]) for v in verdicts
                  if isinstance(v.get("epoch"), (int, float))), default=None)
    lane = classification_lane(verdicts, stage_rows.get("closed_loop") or [])
    control = {
        "transitions": stage_rows.get("transitions") or [],
        "closed_loop": stage_rows.get("closed_loop") or [],
        "structured_init": stage_rows.get("structured_init"),
        "lane_prior_phi1": stage_rows.get("lane_prior_phi1"),
        "nucleation": nuc,
        "classification_lane": lane,
        "stage_slopes": per_stage_slopes(verdicts),
        "eikonal_series": (effective_eikonal_series(max_ep, flags,
                                                    stage_rows.get("transitions") or [],
                                                    stage_rows.get("closed_loop") or [])
                           if max_ep is not None else []),
        "max_epoch": max_ep,
    }
    control["levers"] = lever_status_rows(flags, control, schedule, max_epoch=max_ep)
    return control


def collect_control(log_path, flags: dict, verdicts: list[dict],
                    schedule: dict | None = None) -> dict:
    """IO wrapper: read the run.log, parse the control stage rows, build the control
    dict. NEVER raises (an unreadable/absent log yields the empty-telemetry shape)."""
    lines: list[str] = []
    try:
        p = Path(log_path) if log_path is not None else None
        if p is not None and p.is_file():
            lines = p.read_text(errors="replace").splitlines()
    except Exception:
        lines = []
    try:
        return build_control(parse_stage_rows(lines), flags or {}, verdicts or [], schedule)
    except Exception:
        return {"transitions": [], "closed_loop": [], "structured_init": None,
                "lane_prior_phi1": None,
                "nucleation": {"state": "UNMEASURED", "lane_part_frac": None,
                               "lane_px": None, "lane_cls": None, "mode": None},
                "classification_lane": [], "stage_slopes": [], "eikonal_series": [],
                "max_epoch": None, "levers": []}


# ─────────────────────────── HTML renderers (pure str fns) ───────────────────────────
def _esc(x) -> str:
    return _html.escape(str(x))


def _sec(title: str, body: str, sub: str = "") -> str:
    sub_html = (f'<div style="font-size:10.5px;color:{_MUTED};margin:2px 0 8px">{sub}</div>'
                if sub else "")
    return (f'<div style="background:{_PANEL};border:1px solid {_GRIDC};border-radius:10px;'
            f'padding:12px 14px;margin:10px 0;text-align:left">'
            f'<div style="font-size:10px;color:{_MUTED};letter-spacing:.6px;'
            f'text-transform:uppercase">{title}</div>{sub_html}{body}</div>')


_TD = f'padding:3px 8px;border-bottom:1px solid {_GRIDC};font-size:11px;color:{_FG};text-align:left;vertical-align:top'
_TH = f'padding:3px 8px;border-bottom:1px solid {_GRIDC};font-size:9.5px;color:{_MUTED};text-transform:uppercase;letter-spacing:.5px;text-align:left'


def render_role_line_html(role: dict) -> str:
    """One compact run-role line: the #205 diagnosis or the fresh-successor headline."""
    r = role.get("role")
    if r == "205":
        color, tag = _BAD, "#205"
    elif r == "fresh":
        color, tag = _GOOD, "FRESH"
    else:
        return ""
    return (f'<div style="font-size:11.5px;color:{color};margin:8px auto 0;max-width:1000px;'
            f'text-align:left;line-height:1.55"><b>[{tag}]</b> {_esc(role.get("headline", ""))}</div>')


def render_control_panel_html(control: dict) -> str:
    """The CONTROL-SYSTEM TELEMETRY panel: nucleation gate · stage timeline (fired
    transitions) · closed-loop lane · per-stage slope table (dV/dt vs ep_loss —
    the surrogate↔verdict decoupling made legible). Inline-styled; pure."""
    if not control:
        return ""
    nuc = control.get("nucleation") or {}
    # (c) nucleation gate — rendered PROMINENTLY, with the CORRECTED (FEED-04x) semantics:
    # init part_frac[lane]==0 is the EXPECTED starting point of every run; only mechanism-absent
    # is FAIL, and birth is PROVABLE from verdicts (d_seg < the lane pixel fraction).
    st = nuc.get("state", "UNMEASURED")
    nuc_color = {"PASS": _GOOD, "BIRTH_CONFIRMED": _GOOD, "SEEDED_WATCH": _WARN,
                 "UNSEEDED_WATCH": _WARN, "FAIL": _BAD}.get(st, _MUTED)
    st_label = {"BIRTH_CONFIRMED": "BIRTH ✓", "SEEDED_WATCH": "SEEDED · WATCH",
                "UNSEEDED_WATCH": "UNSEEDED · WATCH"}.get(st, st)
    pf = nuc.get("lane_part_frac")
    pf_s = f"{pf:.4f}" if isinstance(pf, float) else "—"
    dis = nuc.get("pretrain_disagree")
    sup = nuc.get("seed_support")
    if st == "BIRTH_CONFIRMED":
        evidence = (f'd_seg <b>{nuc.get("birth_d_seg"):.4f}</b>@ep{_esc(nuc.get("birth_epoch"))} '
                    f'&lt; lane-frac bound {LANE_FRAC_BOUND} ⟹ lane PRESENT (pixel arithmetic)')
        gate_note = ('a lane-less witness cannot score below the lane pixel fraction — '
                     'birth is PROVEN, not inferred')
    elif st == "SEEDED_WATCH":
        ev_bits = []
        if dis is not None and dis >= PAINT_SIGNAL_MIN_DISAGREE:
            ev_bits.append(f'paint-signal IN target (pretrain disagree <b>{dis:.4f}</b> ≈ the '
                           f'lane band; the #205 no-op measured 0.0004)')
        if sup:
            ev_bits.append(f'island-seed support <b>{sup:.3f}</b> (#205 had none)')
        evidence = (f'init part_frac[lane] = {pf_s} (EXPECTED 0 at init — the witness partition '
                    f'starts empty in every run) · ' + " · ".join(ev_bits))
        gate_note = (f'mechanism MEASURED present → birth watch armed: CE verdicts must '
                     f'out-descend #205 by ep50-75; BIRTH confirms at d_seg &lt; {LANE_FRAC_BOUND}')
    elif st == "FAIL":
        evidence = (f'part_frac[lane] = <b>{pf_s}</b> · a seeding mechanism WAS configured but is '
                    f'inert (disagree {dis if dis is not None else "—"}, no seed support)')
        gate_note = 'the #205 replace-no-op signature: a configured mechanism that births nothing'
    elif st == "UNSEEDED_WATCH":
        evidence = (f'init part_frac[lane] = {pf_s} (EXPECTED 0 at init) · NO init-seed mechanism '
                    f'configured — the clean baseline deliberately dropped the measured no-op lane-prior')
        gate_note = (f'unseeded BY DESIGN → lane birth deferred to CE TRAINING (not an init seed); '
                     f'birth watch armed: verdicts must reach d_seg &lt; {LANE_FRAC_BOUND}')
    else:
        evidence = (f'part_frac[lane] = <b>{pf_s}</b> · lane_px {_esc(nuc.get("lane_px", "—"))} · '
                    f'seed mode {_esc(nuc.get("mode") or "—")}')
        gate_note = ('gate = seeding MECHANISM measured at ep0 (paint-signal / seed support / '
                     'partition); flag presence is NOT evidence')
    nuc_html = (f'<span style="font-size:19px;font-weight:700;color:{nuc_color}">{_esc(st_label)}</span>'
                f'<span style="font-size:11.5px;color:{_FG};margin-left:12px">{evidence}</span>'
                f'<div style="font-size:10px;color:{_MUTED};margin-top:2px">{gate_note}</div>')
    # (a) stage timeline with FIRED transitions
    trs = control.get("transitions") or []
    if trs:
        chips = []
        for t in trs:
            trig = t.get("trigger", "?")
            tc = {"loss_plateau": _GOOD, "cap": _WARN}.get(trig, _MUTED)
            chips.append(f'<span style="color:{_FG}">{_esc(t.get("from") or "?")}</span>'
                         f'<span style="color:{tc}"> →(ep{_esc(t.get("epoch"))}·{_esc(trig)})→ </span>'
                         f'<span style="color:{_FG}">{_esc(t.get("to") or "?")}</span>')
        tl_html = ('<div style="font-size:11.5px">' + " &nbsp; ".join(chips) + "</div>"
                   f'<div style="font-size:10px;color:{_MUTED};margin-top:2px">'
                   f'<span style="color:{_GOOD}">loss_plateau</span> = event-FIRED (dynamical '
                   f'epoch) · <span style="color:{_WARN}">cap</span> = hit the epoch cap · '
                   f'<span style="color:{_MUTED}">hardcoded</span> = legacy fixed boundary</div>')
    else:
        tl_html = f'<div style="font-size:11px;color:{_MUTED}">no stage transition yet</div>'
    # (b) closed-loop lane
    lane = control.get("classification_lane") or []
    if lane:
        src = lane[-1].get("source")
        dots = []
        for e in lane[-40:]:
            c = CLASS_COLORS.get(e.get("classification"), _MUTED)
            act = e.get("action")
            mark = "▲" if act not in (None, "none") else "●"
            dots.append(f'<span title="ep{_esc(e.get("epoch"))} {_esc(e.get("classification"))}'
                        f'{(" · " + _esc(act)) if act not in (None, "none") else ""}" '
                        f'style="color:{c};font-size:12px">{mark}</span>')
        last = lane[-1]
        lc = CLASS_COLORS.get(last.get("classification"), _MUTED)
        lane_html = ('<div style="letter-spacing:2px">' + "".join(dots) + "</div>"
                     f'<div style="font-size:11px;margin-top:3px">latest: '
                     f'<b style="color:{lc}">{_esc(last.get("classification", "?")).upper()}</b> '
                     f'@ep{_esc(last.get("epoch"))} · source {_esc(src)}'
                     + (f' · bump_add {_esc(last.get("eikonal_bump"))}'
                        if last.get("eikonal_bump") not in (None, 0, 0.0) else "")
                     + "</div>")
        legend = " · ".join(f'<span style="color:{c}">{n}</span>'
                            for n, c in CLASS_COLORS.items())
        lane_html += f'<div style="font-size:9.5px;color:{_MUTED};margin-top:2px">{legend} · ▲ = action taken</div>'
    else:
        lane_html = f'<div style="font-size:11px;color:{_MUTED}">no classified eval yet</div>'
    # eikonal effective weight now
    eik = control.get("eikonal_series") or []
    if eik:
        w0, wn = eik[0][1], eik[-1][1]
        lane_html += (f'<div style="font-size:11px;color:{_FG};margin-top:5px">eikonal effective '
                      f'weight: base {w0:g} → now <b>{wn:g}</b> @ep{eik[-1][0]}</div>')
    # (d) per-stage slope table (the Lyapunov dV/dt vs surrogate)
    slopes = control.get("stage_slopes") or []
    if slopes:
        rows_html = ""
        for s in slopes:
            flag = (f'<span style="color:{_BAD};font-weight:700">⚠ DECOUPLED</span>'
                    if s["decoupled"] else f'<span style="color:{_GOOD}">✓ coupled</span>')
            dcol = _BAD if s["d_seg_slope"] > 0 else _GOOD
            rows_html += (f'<tr><td style="{_TD}">{_esc(s["stage"])}</td>'
                          f'<td style="{_TD}">ep{s["ep_lo"]}–{s["ep_hi"]} (n{s["n"]})</td>'
                          f'<td style="{_TD};color:{dcol}">{s["d_seg_slope"]:+.2e}/ep</td>'
                          f'<td style="{_TD}">{s["ep_loss_slope"]:+.3g}/ep</td>'
                          f'<td style="{_TD}">{flag}</td></tr>')
        slope_html = (f'<table style="border-collapse:collapse;width:100%"><tr>'
                      f'<th style="{_TH}">stage</th><th style="{_TH}">window</th>'
                      f'<th style="{_TH}">d_seg slope (dV/dt)</th>'
                      f'<th style="{_TH}">ep_loss slope</th><th style="{_TH}">surrogate↔verdict</th>'
                      f'</tr>{rows_html}</table>'
                      f'<div style="font-size:10px;color:{_MUTED};margin-top:3px">DECOUPLED = '
                      f'd_seg RISING while ep_loss FALLS — the #205 erosion signature (a minority '
                      f'class eroded by the flow while the surrogate still improves)</div>')
    else:
        slope_html = f'<div style="font-size:11px;color:{_MUTED}">no staged verdicts yet</div>'

    body = (_sec("ep0 nucleation gate", nuc_html)
            + _sec("curriculum stage timeline", tl_html)
            + _sec("closed-loop lane", lane_html)
            + _sec("per-stage descent (Lyapunov dV/dt vs surrogate)", slope_html))
    return (f'<div style="max-width:1000px;margin:16px auto 0;text-align:left">'
            f'<div style="font-size:11px;color:{_MUTED};letter-spacing:1.2px;'
            f'text-transform:uppercase">Control-system telemetry '
            f'<span style="color:{_MUTED};text-transform:none;letter-spacing:0">'
            f'· {_ADVISORY}</span></div>{body}</div>')


_STATE_COLORS = {"ok": _GOOD, "warn": _WARN, "bad": _BAD, "pending": _MUTED, "off": _MUTED}


def render_lever_board_html(lever_rows: list[dict]) -> str:
    """The LEVER STATUS board — per-lever live status ('how they're working or not')."""
    if not lever_rows:
        return ""
    rows_html = ""
    for r in lever_rows:
        c = _STATE_COLORS.get(r.get("state"), _MUTED)
        rows_html += (f'<tr><td style="{_TD}">{_esc(r["lever"])}</td>'
                      f'<td style="{_TD};color:{c};font-weight:600">{_esc(r["value"])}</td>'
                      f'<td style="{_TD};color:{_MUTED}">{_esc(r["detail"])}</td></tr>')
    table = (f'<table style="border-collapse:collapse;width:100%"><tr>'
             f'<th style="{_TH}">lever</th><th style="{_TH}">status</th>'
             f'<th style="{_TH}">detail</th></tr>{rows_html}</table>')
    return ('<div style="max-width:1000px;margin:14px auto 0;text-align:left">'
            + _sec("lever status board", table,
                   sub="live per-lever state from run.log + launch.sh (pre-launch levers "
                       "render 'awaiting run')") + "</div>")


def render_config_lever_table_html(flags: dict, role: dict | None = None) -> str:
    """The CONFIG panel: the primary lever table (lever → #205 → fresh → THIS RUN →
    measured grounding) from the master lever ledger."""
    rows_html = ""
    for lever, keys, v205, vfresh, grounding in LEVER_LEDGER:
        run_v = lever_run_value(flags, keys) if flags else "—"
        rows_html += (f'<tr><td style="{_TD}">{_esc(lever)}</td>'
                      f'<td style="{_TD};color:{_MUTED}">{_esc(v205)}</td>'
                      f'<td style="{_TD};color:{_GOOD}">{_esc(vfresh)}</td>'
                      f'<td style="{_TD}">{_esc(run_v)}</td>'
                      f'<td style="{_TD};color:{_MUTED}">{_esc(grounding)}</td></tr>')
    table = (f'<table style="border-collapse:collapse;width:100%"><tr>'
             f'<th style="{_TH}">lever</th><th style="{_TH}">#205</th>'
             f'<th style="{_TH}">fresh</th><th style="{_TH}">this run</th>'
             f'<th style="{_TH}">measured grounding</th></tr>{rows_html}</table>')
    role_note = ""
    if role and role.get("role") in ("205", "fresh"):
        rc = _BAD if role["role"] == "205" else _GOOD
        role_note = (f'<div style="font-size:11px;color:{rc};margin-bottom:6px">this run '
                     f'matches the <b>{"#205 erosion" if role["role"] == "205" else "fresh seeded"}'
                     f'</b> config</div>')
    return ('<div style="max-width:1000px;margin:14px auto 0;text-align:left">'
            + _sec("config — primary lever table (master ledger §1)", role_note + table,
                   sub="#205 = the diagnosed erosion run · fresh = the 13-change calibrated "
                       "successor · groundings are MEASURED (ledger "
                       "fresh_run_master_lever_ledger_20260704.md)") + "</div>")


def render_all_html(control: dict | None, flags: dict, role: dict | None) -> str:
    """All four panel blocks in display order (role line renders separately near the
    banner via render_role_line_html). Empty-safe."""
    out = []
    if control:
        out.append(render_control_panel_html(control))
        out.append(render_lever_board_html(control.get("levers") or []))
    out.append(render_config_lever_table_html(flags or {}, role))
    return "".join(out)
