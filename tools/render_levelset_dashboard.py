#!/usr/bin/env python3
"""Lightweight LIVE dashboard for the level-set witness row (reads the daemon LOG).

The level-set trainer (experiments/train_levelset_witness_realized_through_R_mlx.py)
writes verdicts as JSON lines (``{"stage":"verdict","epoch":N,"d_seg":..,"d_pose":..,
"blob_bytes":..,"implied_S":..}``) to its daemon log — NOT the torch_vehicle_trajectory.jsonl
that tools/render_decisive_run_dashboard.py expects. This renders a self-contained
``index.html`` (embedded PNG, meta-refresh) of the d_seg / d_pose / bytes / implied_S
trajectory vs epoch, with the curriculum stage boundaries + the sub-0.19 goal d_seg drawn.

Advisory only ([macOS-MLX training] NON-PROMOTABLE); the exact row is byte-closed
contest-CPU/CUDA. Usage:
    render_levelset_dashboard.py --log <verdict.log> --out .omx/tmp/dash_levelset/index.html \
        --watch --refresh-seconds 30 --tau 300 --l7 900 --goal-dseg 0.00112

SELF-FOLLOWING (2026-06-27): pass ``--log-glob PATTERN`` (default
``.omx/tmp/levelset_*.log``) instead of (or in addition to) ``--log``. Each
refresh cycle the watched log is RE-RESOLVED to the NEWEST-mtime file matching
the glob that actually CONTAINS verdict lines — so when a new run starts (new
log file) the dashboard auto-switches to it, and the dashboard never follows its
OWN daemon log / a non-verdict log (the confounder that let a DEAD run render as
"live" for hours). ``--log`` remains the explicit-override path (back-compat).

SELF-CALIBRATING STALENESS (2026-06-27, FEED-gf — supersedes the hand-set
``--stale-min 45`` band-aid). The trainer writes in CLUSTERS (checkpoint +
reorient + verdict + verdict_async_done) roughly every N epochs; the async
verdict itself takes minutes; quiet stretches BETWEEN clusters can exceed any
fixed minute threshold on a perfectly healthy run, so a fixed mtime threshold
false-alarms "STOPPED". Telemetry accuracy is VITAL (operator: "accurate, or we
understand+flag when it isn't"), so the live/stale JUDGMENT is now DATA-DERIVED
from the observed verdict cadence:

  * The dashboard SELF-OBSERVES when each NEW verdict EPOCH first appears (the
    arrival wall-time), persisting a tiny state file (``--cadence-state``,
    default ``.omx/tmp/dash_levelset_cadence.json``) keyed by log name so the
    calibration survives a dashboard restart but NOT a new run. When the trainer
    embeds a ``ts`` UTC field on its verdict lines (the durable root-cause fix —
    see FEED note), arrival times are read DIRECTLY from ``ts`` and calibration
    is instantaneous.
  * cadence = MEDIAN inter-arrival of new verdict epochs. Until enough samples
    (``< _MIN_CADENCE_SAMPLES`` measured arrivals) the dashboard shows a
    "● warming up (calibrating cadence)" state using a labeled cadence PRIOR —
    never a false STALE.
  * STALE fires only when time-since-last-NEW-verdict > max(floor, K × cadence)
    (K≈2.5, floor≈10 min) — so it auto-adapts to CE/tau/l7/Muon stages having
    different cadences. A truly dead process during warmup is still caught via a
    secondary floor guard on the log file mtime.
  * The banner DISTINGUISHES two honest facts: "last verdict ep<N> (<Mm> ago)"
    [the d_seg recency — what matters] vs "log active <Ns> ago" [the file mtime,
    a process-writing proxy]. It also shows "expected next verdict ~ep<N+spacing>
    (~<Mm>)" from the measured cadence.

``--stale-min`` remains an optional HARD FLOOR override (back-compat): when set
it replaces the self-calibrated floor in ``max(floor, K × cadence)``; when unset
(default) the floor is ``--stale-floor-min`` (10 min) and the logic is fully
self-calibrated.
"""
from __future__ import annotations

import argparse
import base64
import glob as _glob
import io
import json
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# ── self-calibration: ALL time-telemetry DERIVED FROM THE RUN'S OWN DATA ──
# (operator "Telemetry accuracy vital" 2026-06-30: confident-wrong is the worst failure.)
# There is deliberately NO hardcoded cadence/ETA TIME constant here. Cadence is the run's
# MEASURED inter-verdict gap (from the verdict ``ts`` or self-observed arrivals); until a
# gap exists we say "calibrating" (no number) — never a fabricated value. The retired
# ``_CADENCE_PRIOR_MIN`` (18 min) was the "hardcoded garbage" that false-flagged the live
# n600 run (real cadence ~43 min = 104.5 s/epoch × eval-every-25). The only tunables that
# remain are DIMENSIONLESS POLICY knobs (a stale multiple, a min-sample count) — not times.
_CADENCE_K = 2.0           # STALE multiple: verdict overdue past K × MEASURED cadence
_MIN_CADENCE_GAPS = 1      # >=1 measured inter-verdict gap (i.e. >=2 verdicts) => "measured"
_DEFAULT_EPOCH_SPACING = 25  # next-EPOCH label spacing only (not a time); refined from data
_ASYNC_GRACE_FALLBACK_S = 0.0  # async-eval grace is MEASURED from verdict_async_done secs
_CADENCE_STATE_DEFAULT = ".omx/tmp/dash_levelset_cadence.json"
# Back-compat shims — these no longer drive ANY judgment; retained so external --flags and
# legacy callers don't crash. Defaults are inert (no hardcoded floor / no fabricated prior).
_STALE_FLOOR_MIN = 0.0     # default: NO floor (the binding bound is K × MEASURED cadence)
_CADENCE_PRIOR_MIN = 0.0   # DEPRECATED/unused — retained only for --cadence-prim back-compat
_MIN_CADENCE_SAMPLES = 2   # DEPRECATED alias (>=2 verdicts); superseded by _MIN_CADENCE_GAPS

# ── dark theme palette (clean, legible, comma10k-ish accents) ──
_BG = "#14161a"
_PANEL = "#1b1e24"
_FG = "#d8dde6"
_MUTED = "#8b93a3"
_GRIDC = "#2c313b"
_ACC = "#5ab0ff"  # d_seg / primary curve
_GOAL = "#46d369"  # goal/target hlines
_POSE = "#ffb454"  # d_pose
_BYTES = "#c08cff"  # blob bytes
_SVAL = "#ff6b6b"  # implied_S
# curriculum stage shading (CE / tau / l7+Muon)
_STAGE_CE = "#1f3b5f"
_STAGE_TAU = "#3a2a5f"
_STAGE_L7 = "#5f3320"


def _parse_verdicts(log_path: Path) -> list[dict]:
    rows: list[dict] = []
    if not log_path.exists():
        return rows
    for line in log_path.read_text(errors="replace").splitlines():
        line = line.strip()
        if '"stage": "verdict"' not in line and '"stage":"verdict"' not in line:
            continue
        try:
            d = json.loads(line)
        except Exception:
            continue
        if d.get("stage") != "verdict" or "epoch" not in d:
            continue
        rows.append(d)
    rows.sort(key=lambda r: r["epoch"])
    return rows


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _has_verdict(path: Path) -> bool:
    """True iff the file carries >=1 verdict line. Used so the self-follow
    resolver never latches onto the dashboard's OWN daemon log or any other
    non-verdict log that happens to match the glob with a newer mtime."""
    try:
        if not path.is_file():
            return False
        for line in path.read_text(errors="replace").splitlines():
            if '"stage": "verdict"' in line or '"stage":"verdict"' in line:
                return True
    except Exception:
        return False
    return False


def _resolve_watched_log(log: str | None, log_glob: str | None) -> Path | None:
    """Resolve the log to watch.

    Explicit ``--log`` wins (back-compat, returned verbatim). Otherwise return
    the NEWEST-mtime file matching ``--log-glob`` that contains verdict lines,
    so a freshly-started run (new log file) is auto-followed and the dashboard
    never self-follows its own non-verdict daemon log. Returns ``None`` when no
    verdict-bearing file matches (caller renders a "no run log found" banner).
    Ties on mtime break deterministically by filename (lexicographically last).
    """
    if log:
        return Path(log)
    if not log_glob:
        return None
    verdict_logs = [Path(p) for p in _glob.glob(log_glob) if _has_verdict(Path(p))]
    if not verdict_logs:
        return None
    verdict_logs.sort(key=lambda p: (p.stat().st_mtime, p.name))
    return verdict_logs[-1]


def _is_run_log(path: Path) -> bool:
    """True iff the file is a WITNESS RUN log (verdict-bearing OR still warming up).

    The run-identity marker is the trainer's early ``{"stage": "gt"}`` line, emitted
    a few seconds after launch — BEFORE the first verdict epoch. A verdict line also
    qualifies (verdict-bearing is trivially a run log). This is the warming-up
    sibling of ``_has_verdict``: it recognizes a run that is still in
    ``structured_init`` (0 verdicts) while still EXCLUDING the dashboard's own daemon
    log / any non-run log (no ``gt`` and no ``verdict`` line)."""
    try:
        if not path.is_file():
            return False
        for line in path.read_text(errors="replace").splitlines():
            if '"stage": "gt"' in line or '"stage":"gt"' in line:
                return True
            if '"stage": "verdict"' in line or '"stage":"verdict"' in line:
                return True
    except Exception:
        return False
    return False


def _resolve_run_log(log: str | None, log_glob: str | None) -> Path | None:
    """Resolve the NEWEST-mtime WITNESS RUN log (verdict-bearing OR warming up).

    Mirror of ``_resolve_watched_log`` but keyed on ``_is_run_log`` instead of
    ``_has_verdict``, so a freshly-launched run that has not yet emitted a verdict
    (still in ``structured_init``) is surfaced IMMEDIATELY rather than being invisible
    until its first verdict epoch (~ep25, ~45 min later). Explicit ``--log`` wins.
    Ties on mtime break deterministically by filename (lexicographically last)."""
    if log:
        return Path(log)
    if not log_glob:
        return None
    run_logs = [Path(p) for p in _glob.glob(log_glob) if _is_run_log(Path(p))]
    if not run_logs:
        return None
    run_logs.sort(key=lambda p: (p.stat().st_mtime, p.name))
    return run_logs[-1]


# ─────────────────── run CONFIG + CURRICULUM SCHEDULE (full observability) ───────────────────
# Curated display groups: which parsed launch.sh flags belong in which panel section.
# Generalizable — any flag NOT listed is still kept in ``flags`` (raw), it just is not
# placed in a named display group.
_CONFIG_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("architecture", ("mod-dim", "hidden-dim", "n-hidden", "activation",
                       "hosc-beta", "hosc-omega", "siren-init", "render-h", "render-w")),
    ("basis", ("self-orient", "n-dir-freqs", "freq-across", "freq-along",
               "max-bank-freq", "reorient-every", "chroma", "palette-anchor")),
    ("optimizer", ("muon-lr", "muon-momentum", "muon-ns-steps", "grad-clip",
                   "accum-pairs", "ema-decay", "softmax-temp-start", "softmax-temp-end")),
    ("seed", ("seed", "structured-init", "structured-init-include-lane",
              "lane-prior-phi1", "lane-prior-phi1-mode", "lane-prior-phi1-dash-gate")),
    ("regularizers", ("eikonal-weight", "length-weight",
                      "stage-transition-rewarmup-epochs", "stage-transition-rewarmup-floor",
                      "stage-transition-rewarmup-shape", "stage-transition-reset-moments")),
    ("loss", ("w-seg", "w-pose", "score-domain-loss", "tau-softplus-tau",
              "num-pairs", "verdict-pairs")),
)

# run.log config-bearing stage lines used for the launch.sh-absent fallback.
_CONFIG_STAGE_LINES = ("gt", "front_end", "custom_grouped_backward", "palette_anchor",
                       "lane_prior_phi1", "structured_init", "stem_nyquist")


def _parse_launch_sh_flags(text: str) -> dict:
    """Parse ``--flag [value]`` pairs out of a launch.sh trainer invocation.

    Generalizable token walk: a ``--flag`` followed by a non-flag token takes that
    token as its value; a ``--flag`` followed by another ``--flag`` (or EOL) is a
    boolean ``True``; ``--flag=value`` is split. Leading env assignments and the
    python/script path (tokens not starting with ``--``) are ignored. NEVER raises —
    a malformed script yields whatever parsed cleanly."""
    toks: list[str] = []
    for raw in text.splitlines():
        s = raw.strip()
        if s.endswith("\\"):
            s = s[:-1].strip()
        if not s or s.startswith("#"):
            continue
        toks.extend(t for t in s.split() if t)
    flags: dict = {}
    i = 0
    while i < len(toks):
        t = toks[i]
        if t.startswith("--"):
            key = t[2:]
            if "=" in key:
                k, v = key.split("=", 1)
                flags[k] = v
            elif i + 1 < len(toks) and not toks[i + 1].startswith("--"):
                flags[key] = toks[i + 1]
                i += 1
            else:
                flags[key] = True
        i += 1
    return flags


def _int_flag(flags: dict, key: str):
    v = flags.get(key)
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _schedule_from_flags(flags: dict) -> dict:
    """Curriculum stage epoch-boundaries from the trainer flags. Builds the ordered
    CE / tau / l7 / Muon stages from whatever boundary flags are present (future
    stages included, so the full schedule is visible up front). All boundaries None
    -> empty stages (the client then shows only what it can)."""
    epochs = _int_flag(flags, "epochs")
    eval_every = _int_flag(flags, "eval-every")
    tau_start = _int_flag(flags, "tau-softplus-start-epoch")
    l7_start = _int_flag(flags, "l7-start-epoch")
    muon_start = _int_flag(flags, "muon-start-epoch")
    bounds = [("CE", 0, tau_start), ("tau", tau_start, l7_start),
              ("l7", l7_start, muon_start), ("Muon", muon_start, epochs)]
    stages = [{"name": n, "start": a, "end": b} for (n, a, b) in bounds if a is not None]
    return {"epochs": epochs, "eval_every": eval_every,
            "curriculum": ("curriculum" in flags),
            "tau_start": tau_start, "l7_start": l7_start, "muon_start": muon_start,
            "stages": stages}


def _stage_lines_from_run_log(run_dir: Path) -> list:
    """Fallback config: the config-bearing ``{"stage": ...}`` JSON lines from the
    run's primary log. Returns ``[[stage, dict], ...]``."""
    cand = _resolve_run_log(None, str(Path(run_dir) / "*.log"))
    if cand is None:
        return []
    out = []
    try:
        for line in cand.read_text(errors="replace").splitlines():
            if '"stage"' not in line:
                continue
            try:
                d = json.loads(line.strip())
            except Exception:
                continue
            if d.get("stage") in _CONFIG_STAGE_LINES:
                out.append([d.get("stage"), d])
    except Exception:
        return out
    return out


def parse_run_config(run_dir) -> dict:
    """Parse the run's CONFIG + CURRICULUM SCHEDULE for the dashboard, AUTOMATICALLY,
    from the run's OWN artifacts — so it generalizes to ANY future run with zero
    hand-config. Source of truth = ``<run_dir>/launch.sh`` (the full flag-validated
    command); falls back to the run.log structured-stage emissions when launch.sh is
    absent. Returns a JSON-able ``{source, flags, groups, schedule, stages?}``.
    NEVER raises."""
    out = {"source": "none", "flags": {}, "groups": {}, "schedule": {}}
    if run_dir is None:
        return out
    run_dir = Path(run_dir)
    flags: dict = {}
    launch = run_dir / "launch.sh"
    if launch.is_file():
        try:
            flags = _parse_launch_sh_flags(launch.read_text(errors="replace"))
            if flags:
                out["source"] = "launch.sh"
        except Exception:
            flags = {}
    if not flags:
        stages = _stage_lines_from_run_log(run_dir)
        if stages:
            out["source"] = "run.log"
            out["stages"] = stages
    out["flags"] = flags
    groups: dict = {}
    for gname, keys in _CONFIG_GROUPS:
        rows = [[k, flags[k]] for k in keys if k in flags]
        if rows:
            groups[gname] = rows
    out["groups"] = groups
    out["schedule"] = _schedule_from_flags(flags)
    return out


def _fmt_age(seconds: float | None) -> str:
    if seconds is None:
        return "?"
    s = max(0, int(seconds))
    if s < 90:
        return f"{s}s"
    m = s / 60.0
    if m < 90:
        return f"{m:.1f}m"
    return f"{m / 60.0:.1f}h"


def _staleness(watched: Path | None, stale_min: float) -> dict:
    """File-mtime LOG-ACTIVITY signal (NOT the headline judgment): is the data
    source still being WRITTEN? Returns ``age_s = now - mtime`` and an mtime-based
    state at the given threshold. The headline live/stale judgment is the
    DATA-DERIVED cadence logic in ``_compute_liveness``; this remains the honest
    secondary 'process is touching the file' proxy (and the warmup dead-process
    guard)."""
    if watched is None or not watched.exists():
        return {"state": "missing", "age_s": None, "mtime": None}
    mtime = watched.stat().st_mtime
    age = time.time() - mtime
    state = "stale" if age > stale_min * 60.0 else "live"
    return {"state": state, "age_s": age, "mtime": mtime}


def _detect_switch(prev_name: str | None, watched: Path | None, now_utc: str) -> str | None:
    """Return a switch note when the resolved watched log differs from the
    previous cycle (or None to keep the prior note). First resolution reads as
    '▶ following <name>'; a later change reads as '▶ following <name> (switched <utc>)'."""
    if watched is None:
        return None
    name = watched.name
    if prev_name is None:
        return f"▶ following {name}"
    if prev_name != name:
        return f"▶ following {name} (switched {now_utc})"
    return None


# ───────────────────────── self-calibrating cadence ─────────────────────────
def _median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        return 0.0
    m = n // 2
    return s[m] if n % 2 else (s[m - 1] + s[m]) / 2.0


def _verdict_ts_epoch(row: dict) -> float | None:
    """Parse an embedded UTC timestamp (``ts``) from a verdict row into epoch
    seconds. Returns None when absent/unparseable — the timestamp-less current
    run falls back to self-observed arrival times. Forward-compatible with the
    durable trainer fix that stamps ``ts`` on each verdict line."""
    ts = row.get("ts")
    if not isinstance(ts, str) or not ts:
        return None
    try:
        return datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def _arrival_times(rows: list[dict], sub: dict, now: float) -> tuple[list[tuple[int, float, str]], set]:
    """Return ``[(epoch, arrival_wall_s, source)]`` for verdict rows, where
    arrival is the embedded ``ts`` wall-time when present (source="ts"), else the
    dashboard's self-observed first-seen time (source="observed", persisted in
    ``sub['first_seen']``). Mutates ``sub``: records first_seen for any epoch not
    yet seen, and on the FIRST observation records ``sub['baseline_epochs']`` —
    the epochs already present at launch, which are latched at ``now`` and must
    NOT be counted as measured inter-arrivals (they did not arrive on our clock).
    """
    first_seen: dict = sub.setdefault("first_seen", {})
    if "first_observation_at" not in sub:
        sub["first_observation_at"] = now
        sub["baseline_epochs"] = sorted({int(r["epoch"]) for r in rows})
    baseline = set(sub.get("baseline_epochs", []))
    out: list[tuple[int, float, str]] = []
    for r in rows:
        ep = int(r["epoch"])
        ts = _verdict_ts_epoch(r)
        if ts is not None:
            out.append((ep, ts, "ts"))
            continue
        key = str(ep)
        if key not in first_seen:
            first_seen[key] = now
        out.append((ep, float(first_seen[key]), "observed"))
    out.sort(key=lambda t: t[0])
    return out, baseline


def _measure_cadence(arrivals: list[tuple[int, float, str]], baseline: set, cfg: dict) -> tuple[float | None, str, int]:
    """MEASURED cadence = median of POSITIVE consecutive inter-arrival gaps (seconds)
    of verdict epochs, derived ENTIRELY from the run's own data.

    A sample's arrival time is KNOWN iff it carries an embedded ``ts`` (source "ts")
    OR the dashboard observed the epoch APPEAR after launch (not a launch baseline
    epoch latched at "now"). Returns ``(cadence_s_or_None, source, n_known)``:

      * ``(median(gaps), "measured", n)`` as soon as there is at least
        ``cfg["min_gaps"]`` positive gap (i.e. >= 2 known arrivals) — NEVER a
        fabricated prior. With the trainer's per-verdict ``ts`` this is the very
        SECOND verdict, so the live n600 run reads its real ~43-min cadence
        immediately instead of the retired hardcoded 18 min.
      * ``(None, "calibrating", n)`` while < 1 gap is measurable — the honest
        "not enough data yet" state (the banner/UI shows 'calibrating', not a
        number, and NEVER flags stale on a fabricated threshold).

    No hardcoded minutes anywhere: the only input is the run's own arrival series.
    ``cfg`` accepts ``min_gaps`` (preferred) or the legacy ``min_samples`` (which is
    interpreted as a sample count -> ``min_samples - 1`` gaps) for back-compat.
    """
    if "min_gaps" in cfg:
        min_gaps = int(cfg["min_gaps"])
    else:  # legacy cfg used a sample count; N samples => N-1 gaps
        min_gaps = max(1, int(cfg.get("min_samples", 2)) - 1)
    known = sorted(
        [(ep, a) for (ep, a, src) in arrivals if src == "ts" or ep not in baseline],
        key=lambda t: t[0],
    )
    n = len(known)
    diffs = [b[1] - a[1] for a, b in zip(known, known[1:]) if b[1] > a[1]]
    if len(diffs) >= min_gaps:
        return _median(diffs), "measured", n
    return None, "calibrating", n


def _async_grace_s(log_path: Path | None) -> float:
    """MEASURED async-eval grace (seconds) = max observed ``verdict_async_done`` secs
    in the log — the time an async verdict itself takes (~200-240 s for n600). It is
    added to the stale thresholds so a run is NEVER flagged stale merely because an
    async verdict is in flight. Returns 0.0 when no such line exists (synchronous
    eval) or on any error (telemetry must never crash the live dashboard)."""
    if log_path is None:
        return _ASYNC_GRACE_FALLBACK_S
    try:
        p = Path(log_path)
        if not p.is_file():
            return _ASYNC_GRACE_FALLBACK_S
        best = 0.0
        for line in p.read_text(errors="replace").splitlines():
            if "verdict_async_done" not in line:
                continue
            try:
                d = json.loads(line.strip())
            except Exception:
                continue
            if d.get("stage") == "verdict_async_done":
                s = d.get("secs")
                if isinstance(s, (int, float)) and float(s) > best:
                    best = float(s)
        return best
    except Exception:
        return _ASYNC_GRACE_FALLBACK_S


def _epoch_spacing(rows: list[dict], default: int) -> int:
    eps = sorted({int(r["epoch"]) for r in rows})
    diffs = [b - a for a, b in zip(eps, eps[1:]) if b > a]
    return int(_median(diffs)) if diffs else default


def _compute_liveness(rows: list[dict], watched_mtime: float | None, now: float,
                      sub: dict, cfg: dict, async_grace_s: float = 0.0) -> dict:
    """The HEADLINE, DATA-DERIVED live/stale judgment — NO hardcoded time constants.

    ``watched_mtime`` is the watched log's file mtime (None when no log). ``sub`` is
    the per-log cadence sub-state (mutated with new observations). ``async_grace_s``
    is the MEASURED async-verdict latency (``_async_grace_s``) added to the stale
    thresholds so an in-flight async verdict never reads stale. ``cfg`` may carry
    ``preferred_cadence_s``/``preferred_cadence_source`` (the CURRENT-stage cadence,
    stage-aware) and ``bootstrap_cadence_s`` (a labeled estimate before any gap).
    Returns a rich dict the banner/UI renders verbatim. NO IO here (pure + testable)."""
    log_age = (now - watched_mtime) if watched_mtime is not None else None
    base = {
        "kind": "missing", "calibrating": True, "last_epoch": None, "verdict_age_s": None,
        "log_age_s": log_age, "cadence_s": None, "cadence_source": None,
        "n_samples": 0, "next_epoch": None, "next_eta_s": None,
        "threshold_s": None, "log_threshold_s": None, "async_grace_s": async_grace_s,
        "epoch_spacing": cfg.get("default_spacing", _DEFAULT_EPOCH_SPACING),
    }
    if watched_mtime is None:
        return base
    arrivals, baseline = _arrival_times(rows, sub, now)
    cadence_s, cad_src, n = _measure_cadence(arrivals, baseline, cfg)
    # Stage-aware PREFERRED cadence: when the server measured the CURRENT stage's
    # epoch-time (Muon epochs are slower than CE; tau/l7 differ), use it for the
    # next-verdict ETA + stale threshold so we never carry a stale-stage rate across
    # a curriculum boundary. Only honored when it is itself MEASURED.
    pref = cfg.get("preferred_cadence_s")
    if pref and cfg.get("preferred_cadence_source", "measured") == "measured":
        cadence_s, cad_src = float(pref), "measured"
    # Bootstrap ESTIMATE (operator): before ANY measured gap, if an independent
    # seconds-per-epoch is available derive cadence ~ eval_every × spe and LABEL it
    # 'estimate'. None when unavailable -> stays 'calibrating' (we never invent a
    # number, never a hardcoded prior).
    if cadence_s is None and cfg.get("bootstrap_cadence_s"):
        cadence_s, cad_src = float(cfg["bootstrap_cadence_s"]), "estimate"
    calibrating = cad_src != "measured"  # estimate / calibrating are both "not yet measured"
    spacing = _epoch_spacing(rows, cfg.get("default_spacing", _DEFAULT_EPOCH_SPACING))
    last_epoch = arrivals[-1][0] if arrivals else None
    last_src = arrivals[-1][2] if arrivals else None
    last_arr = arrivals[-1][1] if arrivals else None

    # HONEST verdict recency. A verdict line IS a log write, so the true age of the
    # last verdict can never be FRESHER than the log mtime — the invariant
    # verdict_age >= log_age must always hold. Exact when we KNOW the wall-time
    # (embedded ts, or an epoch we watched APPEAR after launch); for a pre-launch
    # BASELINE epoch whose emit time we never observed, fall back to the LOG mtime.
    if last_epoch is None:
        verdict_age = log_age
    elif last_src == "observed" and last_epoch in baseline:
        verdict_age = log_age
    else:
        verdict_age = now - last_arr
    if verdict_age is not None and log_age is not None:
        verdict_age = max(verdict_age, log_age)

    next_epoch = (last_epoch + spacing) if last_epoch is not None else None
    # next-verdict ETA = last_verdict + cadence (countdown). Honest: it can legitimately
    # be ~40 min for n600. None while calibrating (no cadence to count down from).
    next_eta = (max(0.0, cadence_s - verdict_age)
                if (cadence_s is not None and verdict_age is not None) else None)

    # STALE is DATA-DERIVED, DOUBLE-GATED, and async-aware:
    #  * threshold_s     = max(opt-in floor, K × MEASURED cadence) + async grace
    #  * log_threshold_s = MEASURED cadence + async grace   (the file is touched at
    #                      least once per ~cadence: ckpt-every == eval-every)
    #  * STALE iff cadence is KNOWN (measured/estimate) AND the verdict is overdue
    #    past threshold_s AND the LOG FILE itself is also quiet past log_threshold_s
    #    (genuinely hung). So a healthy run mid-cadence — or mid in-flight async eval
    #    (log still being written by checkpoints) — is NEVER flagged. While cadence
    #    is unknown (calibrating, no gap yet) we NEVER flag stale on a fabricated
    #    threshold; a dead process in that window surfaces via the separate
    #    training-alive signal (meta.training_alive).
    threshold = None
    log_threshold = None
    kind = "live"
    if cadence_s is not None:
        threshold = max(cfg.get("floor_s", 0.0), cfg["k"] * cadence_s) + async_grace_s
        log_threshold = cadence_s + async_grace_s
        verdict_overdue = verdict_age is not None and verdict_age > threshold
        log_quiet = log_age is not None and log_age > log_threshold
        if verdict_overdue and log_quiet:
            kind = "stale"

    return {
        "kind": kind, "calibrating": calibrating, "last_epoch": last_epoch,
        "verdict_age_s": verdict_age, "log_age_s": log_age, "cadence_s": cadence_s,
        "cadence_source": cad_src, "n_samples": n, "next_epoch": next_epoch,
        "next_eta_s": next_eta, "threshold_s": threshold, "log_threshold_s": log_threshold,
        "async_grace_s": async_grace_s, "epoch_spacing": spacing,
    }


def _cfg_from_args(args) -> dict:
    """Build the cadence POLICY config from DIMENSIONLESS knobs only — no hardcoded
    time. The binding stale bound is ``K × MEASURED cadence + async grace``.
    ``--stale-floor-min`` (or legacy ``--stale-min``) is an OPT-IN ABSOLUTE floor
    (minutes), default 0 (no floor). Optional ``eval_every`` + ``seconds_per_epoch``
    (e.g. a measured per-epoch rate) yield a labeled bootstrap-cadence ESTIMATE used
    only before the first measured gap; ``preferred_cadence_s`` (stage-aware) wins
    when present + measured."""
    stale_min = getattr(args, "stale_min", None)
    floor_min = stale_min if stale_min is not None else getattr(args, "stale_floor_min", _STALE_FLOOR_MIN)
    cfg = {
        "k": float(getattr(args, "cadence_k", _CADENCE_K)),
        "floor_s": float(floor_min or 0.0) * 60.0,
        "min_gaps": _MIN_CADENCE_GAPS,
        "default_spacing": _DEFAULT_EPOCH_SPACING,
    }
    spe = getattr(args, "seconds_per_epoch", None)
    ee = getattr(args, "eval_every", None)
    if spe and ee:
        cfg["bootstrap_cadence_s"] = float(spe) * float(ee)
    pref = getattr(args, "preferred_cadence_s", None)
    if pref:
        cfg["preferred_cadence_s"] = float(pref)
        cfg["preferred_cadence_source"] = getattr(args, "preferred_cadence_source", "measured")
    return cfg


def _load_cadence_state(path: str | None, log_name: str) -> tuple[dict, dict]:
    """Load the per-log cadence sub-state from disk (best-effort). Keyed by log
    NAME so calibration survives a dashboard restart but a NEW run (new log
    name) starts fresh — never inheriting a dead run's cadence."""
    all_state: dict = {}
    if path:
        try:
            loaded = json.loads(Path(path).read_text())
            if isinstance(loaded, dict):
                all_state = loaded
        except Exception:
            all_state = {}
    sub = all_state.get(log_name)
    if not isinstance(sub, dict):
        sub = {}
    return all_state, sub


def _save_cadence_state(path: str | None, all_state: dict, log_name: str, sub: dict) -> None:
    """Persist the cadence sub-state atomically (best-effort — telemetry state
    must NEVER crash the live dashboard)."""
    if not path:
        return
    all_state[log_name] = sub
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(all_state))
        os.replace(tmp, p)
    except Exception:
        pass


# ───────────────────────────── rendering ─────────────────────────────
def _stage_short(last_epoch: int | None, tau: int, l7: int) -> str:
    """Compact curriculum stage word for the one-line status."""
    if last_epoch is None:
        return "starting"
    if last_epoch < tau:
        return "CE"
    if last_epoch < l7:
        return "tau"
    return "l7/Muon"


def _trend(rows: list[dict], key: str = "d_seg", look: int = 4) -> tuple[str, str]:
    vals = [r.get(key) for r in rows if r.get(key) is not None]
    if len(vals) < 2:
        return "·", "—"
    recent, prev = vals[-1], vals[-min(look, len(vals))]
    if recent < prev:
        return "▼", "descending"
    if recent > prev:
        return "▲", "rising"
    return "▬", "flat"


def _style_ax(ax) -> None:
    ax.set_facecolor(_PANEL)
    for sp in ax.spines.values():
        sp.set_color(_GRIDC)
    ax.tick_params(colors=_MUTED, labelsize=8)
    ax.grid(alpha=0.18, color=_GRIDC)


def _render_png(rows: list[dict], tau: int, l7: int, goal_dseg: float) -> bytes:
    # NO figure suptitle — the HTML header already carries the title + epoch; a
    # bold matplotlib suptitle on top of it reads as redundant clutter. Each panel
    # is self-labelled (title + caption), which is where the per-metric meaning
    # belongs (clean > complete).
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2))
    fig.patch.set_facecolor(_BG)
    ep = [r["epoch"] for r in rows]
    xlo, xhi = (min(ep), max(ep)) if ep else (0, l7)
    if xhi <= xlo:
        xhi = xlo + max(1, tau)

    def _stage_spans(ax):
        # shade CE / tau / l7+Muon regions on the epoch axis, clipped to data range
        lo, hi = ax.get_xlim()
        segs = [(lo, min(tau, hi), _STAGE_CE), (max(tau, lo), min(l7, hi), _STAGE_TAU),
                (max(l7, lo), hi, _STAGE_L7)]
        for a, b, col in segs:
            if b > a:
                ax.axvspan(a, b, color=col, alpha=0.22, lw=0)
        for x, lab, col in ((tau, "tau", _STAGE_TAU), (l7, "l7", _STAGE_L7)):
            if lo <= x <= hi:
                ax.axvline(x, ls="--", lw=1, color=_MUTED, alpha=0.7)
                ax.text(x, ax.get_ylim()[1], f" {lab}", color=_FG, fontsize=8,
                        va="top", ha="left", alpha=0.85)

    def _panel(ax, key, title, sub, color, logy=False, hline=None, hlabel=None):
        y = [r.get(key) for r in rows]
        xy = [(e, v) for e, v in zip(ep, y) if v is not None]
        _style_ax(ax)
        ax.set_xlim(xlo, xhi)
        if xy:
            xs, ys = zip(*xy)
            ax.plot(xs, ys, "-o", ms=3.5, lw=1.6, color=color)
            if len(xs) == 1:
                ax.annotate(f"{ys[0]:.4g}", (xs[0], ys[0]), color=_FG, fontsize=8)
        ax.set_title(title, fontsize=10.5, color=_FG, loc="left")
        ax.set_xlabel(sub, fontsize=8, color=_MUTED)
        # log scale only with >=1 strictly-positive value (an empty min() on a
        # generator with no positives raises ValueError; d_seg/d_pose CAN be 0 in
        # a degenerate/early verdict, and this is also called once OUTSIDE the
        # watch-loop try/except, so guard it so a 0-valued point never crashes).
        if logy and xy and any(v > 0 for v in ys):
            ax.set_yscale("log")
        if hline is not None:
            ax.axhline(hline, ls=":", lw=1.3, color=_GOAL, alpha=0.9)
            ax.text(xlo, hline, f" {hlabel or ''}", color=_GOAL, fontsize=8, va="bottom")
        _stage_spans(ax)

    _panel(axes[0, 0], "d_seg", "d_seg — realized SegNet-argmax disagreement",
           "epoch · GOAL line = 0.00112 (sub-0.19) · LOWER is better", _ACC,
           logy=True, hline=goal_dseg, hlabel="sub-0.19 goal")
    _panel(axes[0, 1], "d_pose", "d_pose — realized PoseNet MSE (6-dim)",
           "epoch · target ~9e-4 (parent existence-proof) · LOWER is better", _POSE,
           logy=True, hline=9e-4, hlabel="existence-proof ~9e-4")
    _panel(axes[1, 0], "blob_bytes", "blob_bytes — LEARNED payload (counted in archive)",
           "epoch · smaller payload = lower rate term", _BYTES)
    _panel(axes[1, 1], "implied_S", "implied_S — ADVISORY mid-training estimate (NOT the contest score)",
           "epoch · frontier pointer = 0.19110 · the real row is byte-closed CPU/CUDA", _SVAL,
           logy=True, hline=0.19110, hlabel="pointer 0.19110")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=92, facecolor=_BG)
    plt.close(fig)
    return buf.getvalue()


def _banner_html(live: dict | None, watched: Path | None, switched_note: str | None,
                 log_glob: str | None) -> str:
    """Build the data-derived status banner + the two honest recency facts +
    the next-verdict estimate + watched-log line + switch note. The live/stale
    judgment is keyed to ONE threshold = max(floor, K × cadence) and the banner
    cites that SAME cadence (displayed == used) — never a hand-set band. A
    HEALTHY run mid-cadence reads ``● live`` (with a ``(calibrating)`` tag while
    the cadence is still the prior, ``(measured)`` once observed)."""
    if live is None:
        return ""
    kind = live.get("kind")
    verdict_age = live.get("verdict_age_s")
    log_age = live.get("log_age_s")
    last_ep = live.get("last_epoch")
    cadence_s = live.get("cadence_s")
    threshold_s = live.get("threshold_s")
    calibrating = live.get("calibrating", False)
    next_ep = live.get("next_epoch")
    next_eta = live.get("next_eta_s")

    # cadence display: a real measured/estimate number, or the honest word
    # "calibrating" — NEVER a fabricated "~?m" or a hardcoded prior.
    cad_tag = "calibrating" if calibrating else "measured"
    cad_phrase = (f"cadence ~{cadence_s / 60.0:.0f}m ({cad_tag})"
                  if cadence_s else "cadence calibrating (measuring 1st gap)")
    cad_m = f"{cadence_s / 60.0:.0f}m" if cadence_s else "calibrating"
    # the two honest facts. verdict_age >= log_age by construction (a verdict IS
    # a log write), so they are always internally consistent.
    verdict_fact = (f"last verdict ep{last_ep} ({_fmt_age(verdict_age)} ago)"
                    if last_ep is not None else "no verdict yet")
    log_fact = f"log active {_fmt_age(log_age)} ago" if log_age is not None else ""
    next_fact = (f"next ~ep{next_ep} (~{_fmt_age(next_eta)})"
                 if next_ep is not None and next_eta is not None else "")

    if kind == "missing":
        glob_hint = f" (glob: {log_glob})" if log_glob else ""
        banner = f'<div class="banner stale">⚠ no run log found{glob_hint}</div>'
    elif kind == "stale":
        # stale fires only with a KNOWN cadence; cite the binding multiple honestly.
        note = (f">{_CADENCE_K:g}x the ~{cad_m} {cad_tag} cadence"
                if cadence_s else f"> the {(threshold_s or 0) / 60.0:.0f}m bound")
        banner = (
            f'<div class="banner stale">⚠ STALE — no verdict in {_fmt_age(verdict_age)} '
            f'({note}) — likely STOPPED · {log_fact}</div>'
        )
    else:  # live (healthy; cadence measured, estimate, or still calibrating)
        banner = (
            f'<div class="banner live">● live · {verdict_fact} · {next_fact} · '
            f'{cad_phrase} · {log_fact}</div>'
        )

    watched_line = ""
    if watched is not None:
        watched_line = f'<div class="w">watched: {Path(watched).name}</div>'
    switch_line = f'<div class="sw">{switched_note}</div>' if switched_note else ""
    return banner + watched_line + switch_line


def _pill_html(live: dict | None) -> str:
    """ONE small color-coded status pill: ● live / ◐ warming up / ⚠ stale /
    ⚠ no log. 'warming up' is a CALM (non-red) calibrating state, NOT an alarm —
    the run is live; only the cadence is still the prior."""
    if live is None:
        return ""
    kind = live.get("kind")
    if kind == "missing":
        cls, txt = "stale", "⚠ no log"
    elif kind == "stale":
        cls, txt = "stale", "⚠ stale"
    elif live.get("calibrating"):
        cls, txt = "warm", "◐ warming up"
    else:
        cls, txt = "live", "● live"
    return f'<span class="pill {cls}">{txt}</span>'


def _status_line(rows: list[dict], live: dict | None, tau: int, l7: int) -> str:
    """ONE compact line: stage · ep · next-verdict (or the stale/missing reason)."""
    if live is not None and live.get("kind") == "missing":
        return "no run log found"
    if live is None or not rows:
        return "waiting for ep0 verdict…"
    kind = live.get("kind")
    last_ep = live.get("last_epoch")
    parts = [f"{_stage_short(last_ep, tau, l7)} stage"]
    if last_ep is not None:
        parts.append(f"ep{last_ep}")
    if kind == "stale":
        parts.append(f"no verdict in {_fmt_age(live.get('verdict_age_s'))} — likely stopped")
    elif live.get("next_epoch") is not None and live.get("next_eta_s") is not None:
        parts.append(f"next verdict ~{_fmt_age(live.get('next_eta_s'))}")
    return " · ".join(parts)


def _detail_line(live: dict | None) -> str:
    """ONE tiny muted line carrying the two honest telemetry facts (verdict
    recency vs log activity — never blurred) + the cadence + its calibrated/prior
    source. verdict_age >= log_age by construction (a verdict IS a log write)."""
    if live is None or live.get("kind") == "missing":
        return ""
    bits = []
    va, la, cs = live.get("verdict_age_s"), live.get("log_age_s"), live.get("cadence_s")
    if va is not None:
        bits.append(f"verdict {_fmt_age(va)} ago")
    if la is not None:
        bits.append(f"log {_fmt_age(la)} ago")
    if cs:
        tag = "calibrating" if live.get("calibrating") else "measured"
        bits.append(f"cadence ~{cs / 60.0:.0f}m ({tag})")
    else:
        bits.append("cadence calibrating")  # honest: no gap measured yet (no fabricated number)
    return " · ".join(bits)


def _write_html(out: Path, png: bytes, rows: list[dict], refresh: int,
                watched: Path | None = None, live: dict | None = None,
                switched_note: str | None = None, log_glob: str | None = None,
                tau: int = 300, l7: int = 900, goal_dseg: float = 0.00112) -> None:
    """Clean, minimal single-page dashboard: title + status pill, one big d_seg
    headline, one compact status line, one tiny telemetry detail line, the 4
    panels (each self-explanatory in its own caption), a minimal footer. No JS."""
    b64 = base64.b64encode(png).decode("ascii")
    last = rows[-1] if rows else {}
    d_seg = last.get("d_seg")
    dseg_str = f"{d_seg:.5f}" if isinstance(d_seg, (int, float)) else "—"
    arrow, _direction = _trend(rows, "d_seg")
    pill = _pill_html(live)
    status = _status_line(rows, live, tau, l7)
    detail = _detail_line(live)
    headline = (
        f'<div class="hl">d_seg <b>{dseg_str}</b> '
        f'<span class="arr">{arrow}</span>'
        f'<span class="goal">goal &lt;{goal_dseg:g}</span></div>'
        if rows else '<div class="hl">waiting for ep0 verdict…</div>'
    )
    detail_html = f'<div class="detail">{detail}</div>' if detail else ""
    switch_html = (f'<div class="sw">{switched_note}</div>'
                   if switched_note and "switched" in switched_note else "")
    watched_name = Path(watched).name if watched is not None else ""
    html = (
        '<!doctype html><html><head><meta charset="utf-8">'
        f'<meta http-equiv="refresh" content="{refresh}">'
        "<title>Level-Set Witness</title>"
        "<style>"
        f"body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;"
        f"background:{_BG};color:{_FG};text-align:center;margin:0;padding:24px 16px;line-height:1.5}}"
        f".title{{font-size:13px;color:{_MUTED};letter-spacing:1.5px;text-transform:uppercase;"
        "margin-bottom:16px}"
        ".pill{font-size:12px;font-weight:600;padding:3px 11px;border-radius:11px;"
        "margin-left:10px;vertical-align:middle}"
        ".pill.live{background:#173d22;color:#7fe0a0}"
        ".pill.warm{background:#3a3413;color:#e6cf7a}"
        ".pill.stale{background:#4a1717;color:#ff9b9b}"
        f".hl{{font-size:20px;color:{_FG};font-weight:300;margin:4px 0}}"
        f".hl b{{color:{_ACC};font-size:34px;font-weight:600;font-variant-numeric:tabular-nums;"
        "margin:0 4px}"
        f".arr{{font-size:15px;color:{_MUTED};margin-left:2px}}"
        f".goal{{font-size:12px;color:{_MUTED};margin-left:12px}}"
        f".status{{font-size:14px;color:{_FG};margin:10px 0 2px}}"
        f".detail{{font-size:11px;color:{_MUTED};margin-bottom:20px;font-variant-numeric:tabular-nums}}"
        "img{max-width:100%;border-radius:8px}"
        f".foot{{font-size:10.5px;color:{_MUTED};opacity:.75;margin-top:18px;line-height:1.7}}"
        f".sw{{font-size:10.5px;color:{_ACC};opacity:.8;margin-top:4px}}"
        "</style></head>"
        "<body>"
        f'<div class="title">Level-Set Witness{pill}</div>'
        f"{headline}"
        f'<div class="status">{status}</div>'
        f"{detail_html}"
        f'<img src="data:image/png;base64,{b64}">'
        f'<div class="foot">[macOS-MLX advisory · NON-PROMOTABLE] · pointer 0.19110 · '
        f"stages CE · tau · l7 · Muon · refresh {refresh}s"
        + (f" · {watched_name}" if watched_name else "")
        + "</div>"
        f"{switch_html}"
        "</body></html>"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(html)
    os.replace(tmp, out)


def _render_once(args, state: dict | None = None) -> dict:
    """Render one cycle. ``state`` persists across watch cycles (in-process) so
    the watched log can be re-resolved (self-follow) and switches detected. The
    cadence calibration is persisted to DISK (``--cadence-state``) keyed by log
    name so it also survives a dashboard restart."""
    if state is None:
        state = {}
    watched = _resolve_watched_log(getattr(args, "log", None), getattr(args, "log_glob", None))
    note = _detect_switch(state.get("watched_name"), watched, _utc_now())
    if note is not None:
        state["switched_note"] = note
    if watched is not None:
        state["watched_name"] = watched.name
    rows = _parse_verdicts(watched) if watched is not None else []
    now = time.time()
    cfg = _cfg_from_args(args)
    log_name = watched.name if watched is not None else "_none_"
    all_state, sub = _load_cadence_state(getattr(args, "cadence_state", _CADENCE_STATE_DEFAULT), log_name)
    mtime = watched.stat().st_mtime if (watched is not None and watched.exists()) else None
    async_grace = _async_grace_s(watched)
    live = _compute_liveness(rows, mtime, now, sub, cfg, async_grace_s=async_grace)
    _save_cadence_state(getattr(args, "cadence_state", _CADENCE_STATE_DEFAULT), all_state, log_name, sub)
    png = _render_png(rows, args.tau, args.l7, args.goal_dseg)
    _write_html(Path(args.out), png, rows, args.refresh_seconds, watched, live,
                state.get("switched_note"), getattr(args, "log_glob", None),
                tau=args.tau, l7=args.l7, goal_dseg=args.goal_dseg)
    return {"verdicts": len(rows),
            "watched": (watched.name if watched is not None else None),
            "kind": live["kind"], "cadence_source": live.get("cadence_source")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", default=None,
                    help="explicit verdict log to watch (override; back-compat)")
    ap.add_argument("--log-glob", default=".omx/tmp/levelset_*.log",
                    help="self-follow: each cycle watch the NEWEST verdict-bearing "
                         "log matching this glob (ignored when --log is given)")
    ap.add_argument("--out", default=".omx/tmp/dash_levelset/index.html")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--refresh-seconds", type=int, default=30)
    ap.add_argument("--tau", type=int, default=300)
    ap.add_argument("--l7", type=int, default=900)
    ap.add_argument("--goal-dseg", type=float, default=0.00112)
    ap.add_argument("--stale-min", type=float, default=None,
                    help="OPTIONAL absolute floor (minutes) for the stale threshold; "
                         "replaces --stale-floor-min in max(floor, K×cadence)+grace. "
                         "UNSET (default) = fully data-derived (no floor).")
    ap.add_argument("--stale-floor-min", type=float, default=_STALE_FLOOR_MIN,
                    help=f"OPT-IN absolute floor (minutes; default {_STALE_FLOOR_MIN:g} = NONE). "
                         "STALE fires at max(floor, K×MEASURED_cadence)+async_grace — the "
                         "binding bound is the measured cadence, never a hardcoded minute value.")
    ap.add_argument("--cadence-k", type=float, default=_CADENCE_K,
                    help=f"DIMENSIONLESS stale multiple on the MEASURED cadence (default {_CADENCE_K:g})")
    ap.add_argument("--cadence-prior-min", type=float, default=_CADENCE_PRIOR_MIN,
                    help="DEPRECATED/IGNORED — there is no hardcoded cadence prior; cadence is "
                         "the run's MEASURED inter-verdict gap ('calibrating' until the 1st gap). "
                         "Retained only so legacy invocations do not crash.")
    ap.add_argument("--cadence-state", default=_CADENCE_STATE_DEFAULT,
                    help="tiny JSON state file persisting observed verdict arrival "
                         "times (keyed by log name) so cadence survives restarts")
    args = ap.parse_args()

    stop = {"v": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("v", True))
    state: dict = {}
    # The FIRST render must be as crash-proof as the watch-loop ones: a transient startup error
    # (TOCTOU log race, matplotlib/IO hiccup, malformed verdict line) must NOT kill the
    # "load-bearing" live daemon before it ever enters the watch loop. On failure we log and
    # continue into the loop, which will retry every refresh (recovering once the source settles).
    try:
        info = _render_once(args, state)
        print(json.dumps({"stage": "dashboard", "rendered": True, "verdicts": info["verdicts"],
                          "watched": info["watched"], "kind": info["kind"],
                          "cadence_source": info["cadence_source"], "out": args.out}))
    except Exception as e:
        print(json.dumps({"stage": "dashboard", "rendered": False, "error": str(e), "out": args.out}))
        if not args.watch:
            raise  # non-watch one-shot: surface the failure (rc!=0) instead of pretending success
    if not args.watch:
        return
    while not stop["v"]:
        for _ in range(args.refresh_seconds):
            if stop["v"]:
                break
            time.sleep(1)
        try:
            _render_once(args, state)
        except Exception as e:  # a transient read/render error must not kill the live dash
            print(json.dumps({"stage": "dashboard", "error": str(e)}))


if __name__ == "__main__":
    main()
