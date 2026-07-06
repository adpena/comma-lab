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
import re
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.transforms as mtransforms  # noqa: E402

# ── DASHBOARD PASS 2026-07-04: control-system telemetry + lever panels + the fresh-run
# A/B projection. Guarded imports (sister modules in tools/): when either is unimportable
# the dashboard renders WITHOUT the new panels — never crashes the load-bearing daemon.
import sys as _sys  # noqa: E402

_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    import dashboard_control_telemetry as _dct  # noqa: E402
except Exception:  # pragma: no cover - degraded-but-alive path
    _dct = None
try:
    import dashboard_trajectory_model as _dtm  # noqa: E402
except Exception:  # pragma: no cover - degraded-but-alive path
    _dtm = None

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


_LAUNCH_TS_RE = re.compile(r"(\d{8}T\d{6}Z)")


def _launch_ts(path: Path) -> str | None:
    """Extract the launch timestamp from a run-log's FULL PATH.

    Run logs carry a UTC launch stamp ``YYYYMMDDTHHMMSSZ`` either in the filename
    (``.omx/tmp/levelset_<label>_<TS>.log``) OR in the parent directory
    (``experiments/results/levelset_<label>_<TS>/run.log`` — where the filename is
    the bare ``run.log`` and the stamp lives on the dir). We search the WHOLE path
    and take the MAX match so BOTH layouts order by launch time. This is the robust
    identity of "which run is current" — a strictly-newer launch supersedes an
    older run (we run serially), immune to the mtime race at the swap instant.
    ISO-8601 basic form sorts chronologically as a plain string.

    CRITICAL: searching only ``path.name`` was a bug — a ``run.log`` (no stamp in
    the name) then sorted as ``""`` and LOST to any timestamped-filename ``.log``,
    even a much older one, surfacing an ancient run. Full-path fixes that.
    Returns ``None`` when no token is present (caller falls back to mtime)."""
    matches = _LAUNCH_TS_RE.findall(str(path))
    return max(matches) if matches else None


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


def _multi_glob(log_glob: str) -> list[str]:
    """Expand a COMMA-SEPARATED list of globs into a deduped path list.

    Live runs launched via the governed daemon log to ``.omx/tmp/levelset_*.log``;
    other launch paths write ``experiments/results/levelset_*/run.log``. A single
    glob pattern cannot match both, so the default watch glob is a comma list and
    the resolvers union all matches — so a live run is NEVER invisible to the
    dashboard just because it logged to the other location."""
    seen: dict[str, None] = {}
    for pat in log_glob.split(","):
        pat = pat.strip()
        if not pat:
            continue
        for p in _glob.glob(pat):
            seen[p] = None
    return list(seen)


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
    verdict_logs = [Path(p) for p in _multi_glob(log_glob) if _has_verdict(Path(p))]
    if not verdict_logs:
        return None
    # Newest LAUNCH wins (robust to the mtime race at the run-swap instant); mtime
    # then name break ties among same-launch / un-timestamped logs.
    verdict_logs.sort(key=lambda p: (_launch_ts(p) or "", p.stat().st_mtime, p.name))
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
    run_logs = [Path(p) for p in _multi_glob(log_glob) if _is_run_log(Path(p))]
    if not run_logs:
        return None
    # Newest LAUNCH is the current run (we run serially, never 2 concurrent); this
    # is robust to the mtime race at the swap instant where a just-killed run can
    # flush a final write whose mtime briefly beats the fresh run's first write.
    run_logs.sort(key=lambda p: (_launch_ts(p) or "", p.stat().st_mtime, p.name))
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
    -> empty stages (the client then shows only what it can).

    (L7-F1 fix, throughput review 2026-07-04) A DEMOTED l7 — ``--l7-start-epoch >=
    epochs`` (the "never" form, e.g. 1001 @ epochs=1000 in fresh_seeded) or ``>
    muon_start`` (unreachable under Muon precedence) — emits NO l7 segment and Muon
    follows tau directly. The naive chain otherwise produced an INVERTED [1001, 726)
    l7 segment + a tau stage overlapping Muon (the exact class
    ``dashboard_trajectory_model._stage_segments`` fixes on its surface)."""
    epochs = _int_flag(flags, "epochs")
    eval_every = _int_flag(flags, "eval-every")
    tau_start = _int_flag(flags, "tau-softplus-start-epoch")
    l7_start = _int_flag(flags, "l7-start-epoch")
    muon_start = _int_flag(flags, "muon-start-epoch")
    l7_demoted = l7_start is not None and (
        (epochs is not None and l7_start >= epochs)
        or (muon_start is not None and l7_start > muon_start))
    if l7_demoted:
        bounds = [("CE", 0, tau_start), ("tau", tau_start, muon_start),
                  ("Muon", muon_start, epochs)]
    else:
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


def _fmt_bytes(n: float | int | None) -> str:
    """Human byte size (KB/MB/GB) for the checkpoint-footprint display. None -> '?'."""
    if n is None:
        return "?"
    n = float(n)
    for unit, div in (("GB", 1024 ** 3), ("MB", 1024 ** 2), ("KB", 1024)):
        if n >= div:
            return f"{n / div:.1f}{unit}"
    return f"{int(n)}B"


# ─────────────── #205 MULTI-DAY RUN OBSERVABILITY (all REAL trainer telemetry) ───────────────
# Every reader below sources a REAL file/line the trainer emits (never a fabricated field):
#   * levelset_best.json                    -> _read_best_json          (best-so-far EMA-shadow)
#   * {"stage":"checkpoint",...} log lines  -> _parse_checkpoints       (per-stage ckpt ledger)
#   * levelset_resume_state.npz / _ema_mlx  -> _resume_status           (resumability RIGHT NOW)
#   * levelset_*.npz on disk                -> _ckpt_disk_footprint      (disk hygiene visibility)
#   * {"stage":"custom_grouped_backward"}   -> _perf_env_status         (~17x fast-path active?)
#   * launch.sh schedule flags              -> parse_run_config/schedule (curriculum boundaries)
# A missing/partial/mid-write file yields a graceful {} or None — NEVER a crash (the dashboard is
# a load-bearing multi-day daemon). No reader fabricates a value it cannot source.

def _run_dir_for(watched: Path | None, log_glob: str | None, run_dir_arg: str | None) -> Path | None:
    """Resolve the RUN DIR (where launch.sh + the npz/json artifacts live).

    Explicit ``--run-dir`` wins; else the watched log's PARENT (the trainer writes
    run.log + all artifacts into out_dir); else the dir component of ``--log-glob``.
    Returns None when none is resolvable (readers then no-op)."""
    if run_dir_arg:
        return Path(run_dir_arg)
    if watched is not None:
        return Path(watched).parent
    if log_glob:
        d = os.path.dirname(log_glob)
        if d:
            return Path(d)
    return None


def _read_best_json(run_dir: Path | None) -> dict | None:
    """The BEST realized-through-R EMA-shadow pointer the trainer atomically writes to
    ``levelset_best.json`` ({d_seg, epoch, path, ts}) on each new best. This IS the
    deploy artifact the byte-close / next-arm warm-start consumes. None when absent."""
    if run_dir is None:
        return None
    p = Path(run_dir) / "levelset_best.json"
    try:
        if not p.is_file():
            return None
        d = json.loads(p.read_text(errors="replace"))
        return d if isinstance(d, dict) else None
    except Exception:
        return None


def _parse_checkpoints(log_path: Path | None) -> dict:
    """Parse the trainer's ``{"stage":"checkpoint", ...}`` log lines into a ledger.

    Kinds emitted by the trainer: ``best`` (new best EMA shadow), ``stage_transition``
    (PRESERVED stage-encoded ckpt at a curriculum boundary), ``intra_stage`` (rolling
    --ckpt-every crash window), ``final``. Returns
    ``{best, last, preserved:[...], counts:{...}, last_intra}`` — the crash-insurance
    visibility (which per-stage ckpts landed + the last resumable position). NEVER
    raises (a mid-write/truncated line is skipped)."""
    out: dict = {"best": None, "last": None, "preserved": [], "counts": {},
                 "last_intra": None}
    if log_path is None or not Path(log_path).exists():
        return out
    counts: dict = {}
    try:
        for line in Path(log_path).read_text(errors="replace").splitlines():
            if '"checkpoint"' not in line:
                continue
            try:
                d = json.loads(line.strip())
            except Exception:
                continue
            if d.get("stage") != "checkpoint":
                continue
            kind = d.get("kind")
            counts[kind] = counts.get(kind, 0) + 1
            if kind == "best":
                out["best"] = d
                continue
            out["last"] = d  # latest non-best ckpt line = most-recent resumable write
            if kind == "intra_stage":
                out["last_intra"] = d
            # PRESERVED stage-encoded ckpts (stage_transition + final carry ema_preserved)
            if d.get("ema_preserved"):
                out["preserved"].append(
                    {"kind": kind, "epoch": d.get("epoch"),
                     "ema": d.get("ema_preserved"), "resume": d.get("resume_preserved")})
    except Exception:
        return out
    out["counts"] = counts
    return out


def _ckpt_disk_footprint(run_dir: Path | None) -> dict:
    """Disk footprint of the run's checkpoint npzs (``levelset_*.npz``) — the
    disk-hygiene visibility for a multi-day run. Returns
    ``{total_bytes, count, biggest:[(name, bytes)...]}``. NEVER raises."""
    out = {"total_bytes": 0, "count": 0, "biggest": []}
    if run_dir is None:
        return out
    try:
        files = sorted(Path(run_dir).glob("levelset_*.npz"))
        sized = []
        for f in files:
            try:
                sized.append((f.name, f.stat().st_size))
            except OSError:
                continue
        out["count"] = len(sized)
        out["total_bytes"] = sum(b for _, b in sized)
        out["biggest"] = sorted(sized, key=lambda t: -t[1])[:4]
    except Exception:
        return out
    return out


def _resume_status(run_dir: Path | None, checkpoints: dict, now: float | None = None) -> dict:
    """Is the run resumable-from-disk RIGHT NOW? Reads the ACTUAL artifacts + the last
    checkpoint line. Returns ``{resumable, resume_epoch, resume_npz, resume_age_s,
    deploy_npz, deploy_age_s, best_npz}``. The resume sidecar (``levelset_resume_state.npz``)
    is the crash-continue target; the deploy npz (``levelset_witness_ema_mlx.npz``) is the
    EMA-shadow byte-close artifact. NEVER raises."""
    now = time.time() if now is None else now
    out = {"resumable": False, "resume_epoch": None, "resume_npz": False,
           "resume_age_s": None, "deploy_npz": False, "deploy_age_s": None,
           "best_npz": False}
    if run_dir is None:
        return out
    run_dir = Path(run_dir)

    def _age(name: str):
        p = run_dir / name
        try:
            if p.is_file():
                return True, now - p.stat().st_mtime
        except OSError:
            pass
        return False, None

    out["resume_npz"], out["resume_age_s"] = _age("levelset_resume_state.npz")
    out["deploy_npz"], out["deploy_age_s"] = _age("levelset_witness_ema_mlx.npz")
    out["best_npz"], _ = _age("levelset_witness_ema_BEST.npz")
    last = checkpoints.get("last") or checkpoints.get("last_intra")
    if isinstance(last, dict):
        out["resume_epoch"] = last.get("epoch")
    # resumable iff a resume sidecar OR the deploy npz exists on disk (the trainer's
    # --resume-from accepts either), matching _resolve_resume_path in the trainer.
    out["resumable"] = bool(out["resume_npz"] or out["deploy_npz"])
    return out


def _perf_env_status(log_path: Path | None) -> dict:
    """Is the ~17x custom-grouped-backward fast path ACTIVE (the launch-gate perf
    invariant)? Reads the trainer's ``{"stage":"custom_grouped_backward","active":..}``
    line. Returns ``{active, reason}`` (active None => line not seen yet). NEVER raises."""
    out: dict = {"active": None, "reason": None}
    if log_path is None or not Path(log_path).exists():
        return out
    try:
        for line in Path(log_path).read_text(errors="replace").splitlines():
            if "custom_grouped_backward" not in line:
                continue
            try:
                d = json.loads(line.strip())
            except Exception:
                continue
            if d.get("stage") == "custom_grouped_backward":
                out["active"] = bool(d.get("active"))
                out["reason"] = d.get("reason")
                # keep scanning: the LAST such line wins (most recent truth)
    except Exception:
        return out
    return out


def _first_verdict_ts(rows: list[dict]) -> float | None:
    """Wall-clock (epoch seconds) of the EARLIEST verdict carrying an embedded ``ts``
    — used to compute run elapsed. None when no verdict has a ts (elapsed then N/A)."""
    for r in rows:
        t = _verdict_ts_epoch(r)
        if t is not None:
            return t
    return None


def _stage_progress(last_epoch: int | None, last_seg_form: str | None, schedule: dict) -> dict:
    """Curriculum stage + progress from the REAL schedule (launch.sh boundaries) and the
    last verdict. Prefers the verdict row's own ``seg_form`` for the stage NAME (the
    trainer's authoritative label) and the schedule for the %-progress boundaries.
    Returns ``{stage, stage_start, stage_end, stage_pct, total_pct, epochs, next_name,
    next_epoch, muon_active}`` — any unknown field is None (never fabricated)."""
    epochs = schedule.get("epochs")
    tau = schedule.get("tau_start")
    l7 = schedule.get("l7_start")
    muon = schedule.get("muon_start")
    out = {"stage": None, "stage_start": None, "stage_end": None, "stage_pct": None,
           "total_pct": None, "epochs": epochs, "next_name": None, "next_epoch": None,
           "muon_active": None}
    if last_epoch is None:
        return out
    # Stage name: trust the verdict's seg_form when present; else derive from boundaries.
    _seg_name = {"ce": "CE", "tau_softplus": "tau", "l7_softplus": "l7"}
    stage = _seg_name.get(str(last_seg_form)) if last_seg_form else None
    # boundaries (may be partially None; treat missing future stages as absent)
    b_tau = tau if tau is not None else None
    b_l7 = l7 if l7 is not None else None
    if stage is None:
        if b_tau is not None and last_epoch < b_tau:
            stage = "CE"
        elif b_l7 is not None and (b_tau is None or last_epoch >= b_tau) and last_epoch < b_l7:
            stage = "tau"
        elif b_l7 is not None and last_epoch >= b_l7:
            stage = "l7"
    # stage span from the ordered boundaries
    span = {"CE": (0, b_tau), "tau": (b_tau, b_l7), "l7": (b_l7, epochs)}.get(stage, (None, None))
    s0, s1 = span
    out["stage"] = stage
    out["stage_start"], out["stage_end"] = s0, s1
    if s0 is not None and s1 is not None and s1 > s0:
        out["stage_pct"] = max(0.0, min(1.0, (last_epoch - s0) / (s1 - s0)))
    if epochs and epochs > 0:
        out["total_pct"] = max(0.0, min(1.0, last_epoch / epochs))
    # next stage boundary (for the ETA)
    for name, bnd in (("tau", b_tau), ("l7", b_l7), ("Muon", muon), ("end", epochs)):
        if bnd is not None and bnd > last_epoch:
            out["next_name"], out["next_epoch"] = name, bnd
            break
    # Muon finisher overlay (optimizer switch; runs concurrently with l7 seg-form)
    out["muon_active"] = (muon is not None and last_epoch >= muon)
    return out


def _throughput_eta(live: dict | None, schedule: dict, stage_prog: dict,
                    first_ts: float | None, now: float | None = None) -> dict:
    """Throughput + ETA for the multi-day run, DERIVED from the MEASURED verdict cadence.
    ``s_per_epoch = cadence / eval_every`` (verdicts land every eval_every epochs). ETA to
    the next stage boundary + to total completion use s_per_epoch × epochs-remaining.
    Elapsed = now - first_verdict_ts. Any input unknown => that output stays None (never
    a fabricated number). Returns ``{s_per_epoch, epochs_per_hour, elapsed_s,
    eta_next_boundary_s, eta_total_s}``."""
    now = time.time() if now is None else now
    out = {"s_per_epoch": None, "epochs_per_hour": None, "elapsed_s": None,
           "eta_next_boundary_s": None, "eta_total_s": None}
    cadence = (live or {}).get("cadence_s")
    eval_every = schedule.get("eval_every")
    last_epoch = (live or {}).get("last_epoch")
    epochs = schedule.get("epochs")
    if first_ts is not None:
        out["elapsed_s"] = max(0.0, now - first_ts)
    if cadence and eval_every and eval_every > 0 and (live or {}).get("cadence_source") == "measured":
        spe = cadence / float(eval_every)
        out["s_per_epoch"] = spe
        out["epochs_per_hour"] = (3600.0 / spe) if spe > 0 else None
        if last_epoch is not None:
            nb = stage_prog.get("next_epoch")
            if nb is not None and nb > last_epoch:
                out["eta_next_boundary_s"] = (nb - last_epoch) * spe
            if epochs is not None and epochs > last_epoch:
                out["eta_total_s"] = (epochs - last_epoch) * spe
    return out


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


def _render_control_panel(ax, control: dict, tau: int) -> None:
    """PNG panel: CONTROL SYSTEM — eikonal effective weight (line) + fired stage
    transitions (vlines, trigger-coded) + the closed-loop classification lane
    (color-coded dots along the bottom) + action markers. All from run.log telemetry
    (dashboard_control_telemetry). Empty control renders an honest placeholder."""
    _style_ax(ax)
    ax.set_title("control system — eikonal weight · transitions · closed-loop lane",
                 fontsize=10.5, color=_FG, loc="left")
    ax.set_xlabel("epoch · lane dots: green=converging blue=plateau amber=transient "
                  "red=ERASING · ▲=action", fontsize=8, color=_MUTED)
    if not control or _dct is None:
        ax.text(0.5, 0.5, "awaiting control-system telemetry", color=_MUTED,
                fontsize=9, ha="center", va="center", transform=ax.transAxes)
        return
    eik = control.get("eikonal_series") or []
    if eik:
        xs, ws = zip(*eik)
        ax.plot(xs, ws, "-", lw=1.6, color=_BYTES, label="eikonal effective weight")
        ax.set_ylim(0, max(max(ws) * 1.35, 0.02))
    # fired transitions (trigger-coded)
    for t in control.get("transitions") or []:
        epx = t.get("epoch")
        if epx is None:
            continue
        col = {"loss_plateau": _GOAL, "cap": "#e6cf7a"}.get(t.get("trigger"), _MUTED)
        ax.axvline(epx, ls="--", lw=1.1, color=col, alpha=0.85)
        ax.text(epx, ax.get_ylim()[1], f' {t.get("to")}@{epx}·{t.get("trigger")}',
                color=col, fontsize=7, va="top", ha="left", rotation=0)
    # classification lane (dots at a fixed axes-fraction height; x in data coords)
    lane = control.get("classification_lane") or []
    if lane:
        trans = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
        for e in lane:
            c = _dct.CLASS_COLORS.get(e.get("classification"), _MUTED)
            acted = e.get("action") not in (None, "none")
            ax.plot([e["epoch"]], [0.06], marker=("^" if acted else "o"),
                    ms=(6 if acted else 4), color=c, transform=trans, clip_on=False)
    if eik:  # only the eikonal line carries a legend label (lane dots are caption-coded)
        ax.legend(loc="upper left", fontsize=7, facecolor=_PANEL, edgecolor=_GRIDC,
                  labelcolor=_FG, framealpha=0.9)


def _render_projection_panel(ax, rows: list[dict], proj: dict | None, tau: int,
                             goal_dseg: float) -> None:
    """PNG panel: PROJECTED TRAJECTORIES — the measured d_seg points + the #205
    measured-slope erosion continuation vs the fresh SEEDED run's HYPOTHESIS band,
    both from dtm.build_fresh_run_projection (measured-anchored; the below-CE-floor
    segment is explicitly a hypothesis). The A/B is visually falsifiable."""
    _style_ax(ax)
    ax.set_title("projected trajectories — #205 erosion vs fresh-run HYPOTHESIS band",
                 fontsize=10.5, color=_FG, loc="left")
    ax.set_xlabel("epoch · red dashes = #205 measured-slope continuation · green band = "
                  "seeded HYPOTHESIS [projected/advisory]", fontsize=8, color=_MUTED)
    xy = [(r["epoch"], r.get("d_seg")) for r in rows
          if isinstance(r.get("d_seg"), (int, float)) and r["d_seg"] > 0]
    if xy:
        xs, ys = zip(*xy)
        ax.plot(xs, ys, "o", ms=3.0, color=_ACC, label="measured d_seg [this run]")
    if proj and proj.get("ok"):
        pe = proj["epochs"]
        ax.plot(pe, proj["eroding_205"], "--", lw=1.5, color=_SVAL,
                label="#205 erosion [measured slopes]")
        ax.fill_between(pe, proj["fresh_lo"], proj["fresh_hi"], color=_GOAL, alpha=0.18,
                        label="fresh seeded band [HYPOTHESIS]")
        ax.plot(pe, proj["fresh_lo"], "-", lw=1.0, color=_GOAL, alpha=0.7)
        dv = proj.get("divergence_epoch")
        if dv is not None:
            ax.axvline(dv, ls=":", lw=1.2, color=_FG, alpha=0.6)
            _btr = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
            ax.text(dv, 0.97, f" divergence @ep{dv} (tau/MCF onset)", color=_FG,
                    fontsize=7, va="top", ha="left", alpha=0.85, transform=_btr)
    ax.axhline(goal_dseg, ls=":", lw=1.3, color=_GOAL, alpha=0.9)
    ax.text(0, goal_dseg, f" goal {goal_dseg:g}", color=_GOAL, fontsize=8, va="bottom")
    if xy or (proj and proj.get("ok")):
        ax.set_yscale("log")
        ax.legend(loc="upper right", fontsize=7, facecolor=_PANEL, edgecolor=_GRIDC,
                  labelcolor=_FG, framealpha=0.9)
    else:
        ax.text(0.5, 0.5, "no data / projection yet", color=_MUTED, fontsize=9,
                ha="center", va="center", transform=ax.transAxes)


def _render_png(rows: list[dict], tau: int, l7: int, goal_dseg: float,
                muon: int | None = None, best_dseg: float | None = None,
                control: dict | None = None, fresh_proj: dict | None = None) -> bytes:
    # NO figure suptitle — the HTML header already carries the title + epoch; a
    # bold matplotlib suptitle on top of it reads as redundant clutter. Each panel
    # is self-labelled (title + caption), which is where the per-metric meaning
    # belongs (clean > complete). Rows 1-2 = the original 4 metric panels; row 3 =
    # the control-system panel + the fresh-run A/B projection (2026-07-04 pass).
    fig, axes = plt.subplots(3, 2, figsize=(12, 10.8))
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
        _bounds = [(tau, "tau", _STAGE_TAU), (l7, "l7", _STAGE_L7)]
        if muon is not None:
            _bounds.append((muon, "Muon", _GOAL))  # optimizer-switch boundary (overlays l7)
        for x, lab, col in _bounds:
            if x is not None and lo <= x <= hi:
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
    # BEST-so-far marker on the d_seg panel (the deploy artifact's realized d_seg).
    if best_dseg is not None and best_dseg > 0:
        axes[0, 0].axhline(best_dseg, ls="-", lw=1.1, color=_GOAL, alpha=0.55)
        axes[0, 0].text(xhi, best_dseg, f"best {best_dseg:.5f} ", color=_GOAL, fontsize=7.5,
                        va="bottom", ha="right", alpha=0.9)
    _panel(axes[0, 1], "d_pose", "d_pose — realized PoseNet MSE (6-dim)",
           "epoch · target ~9e-4 (parent existence-proof) · LOWER is better", _POSE,
           logy=True, hline=9e-4, hlabel="existence-proof ~9e-4")
    _panel(axes[1, 0], "blob_bytes", "blob_bytes — LEARNED payload (counted in archive)",
           "epoch · smaller payload = lower rate term", _BYTES)
    _panel(axes[1, 1], "implied_S", "implied_S — ADVISORY mid-training estimate (NOT the contest score)",
           "epoch · frontier pointer = 0.19110 · the real row is byte-closed CPU/CUDA", _SVAL,
           logy=True, hline=0.19110, hlabel="pointer 0.19110")
    # row 3: CONTROL-SYSTEM TELEMETRY + the fresh-run A/B projection (measured-anchored)
    _render_control_panel(axes[2, 0], control or {}, tau)
    axes[2, 0].set_xlim(xlo, xhi)
    _render_projection_panel(axes[2, 1], rows, fresh_proj, tau, goal_dseg)
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


def _fmt_pct(x: float | None) -> str:
    return f"{x * 100:.0f}%" if isinstance(x, (int, float)) else "?"


def _fmt_sig(x, nd: int = 5) -> str:
    """Compact numeric format for a score/telemetry value; '?' when unknown."""
    if isinstance(x, bool) or x is None:
        return "?" if x is None else str(x)
    if isinstance(x, (int, float)):
        return f"{x:.{nd}f}" if isinstance(x, float) else str(x)
    return str(x)


def _progress_bar_html(frac: float | None, color: str) -> str:
    """A tiny inline progress bar (0..1). None => empty track."""
    pct = max(0.0, min(1.0, frac)) * 100.0 if isinstance(frac, (int, float)) else 0.0
    return (f'<div class="bar"><div class="fill" style="width:{pct:.1f}%;'
            f'background:{color}"></div></div>')


def _run_info_html(info: dict | None) -> str:
    """The #205 multi-day observability strip: a compact card grid BELOW the trajectory
    PNG surfacing stage-progress / best-so-far / throughput+ETA / checkpoint ledger /
    resumability / perf-fast-path / config — ALL from the REAL trainer telemetry. Any
    unknown field renders '?' / 'N/A' (never a fabricated number). ``info=None`` (or an
    empty run) yields '' so the page is unchanged (back-compat)."""
    if not info:
        return ""
    sched = info.get("schedule") or {}
    sp = info.get("stage_prog") or {}
    best = info.get("best") or {}
    ck = info.get("checkpoints") or {}
    res = info.get("resume") or {}
    thr = info.get("throughput") or {}
    perf = info.get("perf") or {}
    disk = info.get("disk") or {}
    cfg_flags = (info.get("config") or {}).get("flags") or {}
    cfg_src = (info.get("config") or {}).get("source", "none")

    cards: list[str] = []

    def _card(label: str, body: str, sub: str = "", bar: str = "") -> str:
        sub_html = f'<div class="csub">{sub}</div>' if sub else ""
        return (f'<div class="card"><div class="clabel">{label}</div>'
                f'<div class="cval">{body}</div>{bar}{sub_html}</div>')

    # 1) STAGE + progress ────────────────────────────────────────────────
    stage = sp.get("stage") or "—"
    epochs = sp.get("epochs")
    muon_badge = ' <span class="badge">+Muon</span>' if sp.get("muon_active") else ""
    stage_body = f"{stage}{muon_badge}"
    tot = _fmt_pct(sp.get("total_pct"))
    stage_pct = _fmt_pct(sp.get("stage_pct"))
    ep_now = sp.get("__last_epoch")
    ep_str = f"ep{ep_now}" if ep_now is not None else "ep?"
    ep_tot = f"/{epochs}" if epochs else ""
    stage_sub = f"{ep_str}{ep_tot} · {stage_pct} of stage · {tot} total"
    cards.append(_card("curriculum stage", stage_body, stage_sub,
                       _progress_bar_html(sp.get("total_pct"), _ACC)))

    # 2) BEST-so-far (deploy artifact) ───────────────────────────────────
    if best:
        bd = best.get("d_seg")
        bep = best.get("epoch")
        # need-band for sub-0.15 d_seg is 0.00077-0.00118 (MEMORY CURRENT-STATE).
        band = ""
        if isinstance(bd, (int, float)):
            # sub-0.19 goal d_seg (0.00112) sits INSIDE the sub-0.15 need-band (0.00077-0.00118),
            # so the need-band is the binding annotation; below the floor = even better than needed.
            if bd < 0.00077:
                band = "✓ below sub-0.15 need-band floor"
            elif bd <= 0.00118:
                band = "✓ in sub-0.15 need-band"
        cards.append(_card("best d_seg (deploy)",
                           _fmt_sig(bd), f"@ ep{bep} · {best.get('path', '')} {band}".strip()))
    else:
        cards.append(_card("best d_seg (deploy)", "—", "no finite best yet"))

    # 3) THROUGHPUT + ETA ────────────────────────────────────────────────
    spe = thr.get("s_per_epoch")
    eph = thr.get("epochs_per_hour")
    thr_body = (f"{spe:.1f}s/ep" if isinstance(spe, (int, float)) else "calibrating")
    thr_sub_bits = []
    if isinstance(eph, (int, float)):
        thr_sub_bits.append(f"{eph:.0f} ep/hr")
    if thr.get("elapsed_s") is not None:
        thr_sub_bits.append(f"elapsed {_fmt_age(thr['elapsed_s'])}")
    if thr.get("eta_next_boundary_s") is not None:
        thr_sub_bits.append(f"→{sp.get('next_name', 'next')} ~{_fmt_age(thr['eta_next_boundary_s'])}")
    if thr.get("eta_total_s") is not None:
        thr_sub_bits.append(f"→done ~{_fmt_age(thr['eta_total_s'])}")
    cards.append(_card("throughput · ETA", thr_body, " · ".join(thr_sub_bits) or "measuring cadence"))

    # 4) CHECKPOINT ledger (crash insurance) ─────────────────────────────
    counts = ck.get("counts") or {}
    n_pres = len(ck.get("preserved") or [])
    n_intra = counts.get("intra_stage", 0)
    ck_every = sched.get("__ckpt_every")
    ck_body = f"{n_pres} per-stage · {n_intra} intra"
    ck_sub_bits = []
    if ck_every is not None:
        ck_sub_bits.append(f"ckpt-every {ck_every if ck_every else 'off'}")
    if disk.get("count"):
        ck_sub_bits.append(f"{disk['count']} npz · {_fmt_bytes(disk.get('total_bytes'))} on disk")
    # name the preserved stage ckpts (crash-resume + per-stage A/B visibility)
    pres_names = [f"ep{p.get('epoch')}" for p in (ck.get("preserved") or [])[-4:]]
    if pres_names:
        ck_sub_bits.append("preserved @ " + ",".join(pres_names))
    cards.append(_card("checkpoints", ck_body, " · ".join(ck_sub_bits) or "none yet"))

    # 5) RESUMABILITY (resume-from-disk RIGHT NOW) ───────────────────────
    resumable = res.get("resumable")
    res_body = "✓ resumable" if resumable else "✗ not yet"
    res_bits = []
    if res.get("resume_epoch") is not None:
        res_bits.append(f"from ep{res['resume_epoch']}")
    if res.get("resume_npz"):
        res_bits.append(f"resume npz {_fmt_age(res.get('resume_age_s'))}")
    if res.get("deploy_npz"):
        res_bits.append(f"deploy npz {_fmt_age(res.get('deploy_age_s'))}")
    cards.append(_card("resumability", res_body, " · ".join(res_bits) or "no checkpoint on disk"))

    # 6) PERF fast-path (~17x grouped-backward) ──────────────────────────
    active = perf.get("active")
    if active is True:
        perf_body, perf_sub = "✓ active", "custom_grouped_backward ~17x"
    elif active is False:
        perf_body, perf_sub = "✗ INACTIVE", "SLOW — TAC_MLX_CUSTOM_GROUPED_BACKWARD unset"
    else:
        perf_body, perf_sub = "?", "grouped-backward line not seen yet"
    cards.append(_card("MLX fast path", perf_body, perf_sub))

    # 7) CONFIG + provenance (loss weights / arch / seed) ─────────────────
    def _g(k, default="?"):
        v = cfg_flags.get(k)
        return v if v is not None else default
    cfg_bits = [f"w_seg {_g('w-seg')}", f"w_pose {_g('w-pose')}", f"act {_g('activation')}",
                f"mod {_g('mod-dim')}", f"hid {_g('hidden-dim')}", f"seed {_g('seed')}",
                f"pairs {_g('num-pairs')}"]
    cards.append(_card("config", cfg_src, " · ".join(cfg_bits)))

    out_html = '<div class="grid">' + "".join(cards) + "</div>"
    # ── DASHBOARD PASS 2026-07-04: run-role line + control-system telemetry panel +
    # lever status board + the primary-lever CONFIG table. All inline-styled pure HTML
    # (dashboard_control_telemetry) so the server's #runinfostrip renders them too —
    # panels slot INTO the DESCENT tab with zero server/tab changes. Guarded: absent
    # module or absent telemetry keys leave the strip exactly as before (back-compat).
    if _dct is not None:
        try:
            role = info.get("role") or {}
            out_html = (_dct.render_role_line_html(role) + out_html
                        + _dct.render_all_html(info.get("control"), cfg_flags, role))
        except Exception:
            pass  # a panel bug must never take down the run-info strip
    return out_html


def _write_html(out: Path, png: bytes, rows: list[dict], refresh: int,
                watched: Path | None = None, live: dict | None = None,
                switched_note: str | None = None, log_glob: str | None = None,
                tau: int = 300, l7: int = 900, goal_dseg: float = 0.00112,
                run_info: dict | None = None) -> None:
    """Clean, minimal single-page dashboard: title + status pill, one big d_seg
    headline, one compact status line, one tiny telemetry detail line, the 4
    panels (each self-explanatory in its own caption), the #205 run-info card grid
    (stage progress · best · throughput/ETA · checkpoints · resumability · perf ·
    config — all REAL telemetry; omitted when ``run_info`` is None), a minimal
    footer. No JS."""
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
        # #205 run-info card grid (stage / best / throughput / checkpoints / resume / perf / config)
        ".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));"
        "gap:10px;max-width:1000px;margin:18px auto 0;text-align:left}"
        f".card{{background:{_PANEL};border:1px solid {_GRIDC};border-radius:8px;padding:10px 12px}}"
        f".clabel{{font-size:10px;color:{_MUTED};letter-spacing:.6px;text-transform:uppercase}}"
        f".cval{{font-size:17px;color:{_FG};font-weight:600;margin:3px 0;font-variant-numeric:tabular-nums}}"
        f".csub{{font-size:10.5px;color:{_MUTED};line-height:1.5}}"
        f".bar{{height:5px;background:{_GRIDC};border-radius:3px;margin:5px 0 4px;overflow:hidden}}"
        ".fill{height:100%;border-radius:3px}"
        f".badge{{font-size:10px;font-weight:600;color:{_GOAL};background:#173d22;"
        "padding:1px 6px;border-radius:8px;vertical-align:middle}"
        f".foot{{font-size:10.5px;color:{_MUTED};opacity:.75;margin-top:18px;line-height:1.7}}"
        f".sw{{font-size:10.5px;color:{_ACC};opacity:.8;margin-top:4px}}"
        "</style></head>"
        "<body>"
        f'<div class="title">Level-Set Witness{pill}</div>'
        f"{headline}"
        f'<div class="status">{status}</div>'
        f"{detail_html}"
        f'<img src="data:image/png;base64,{b64}">'
        f"{_run_info_html(run_info)}"
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


def _collect_run_info(watched: Path | None, log_glob: str | None, run_dir_arg: str | None,
                      rows: list[dict], live: dict, now: float) -> dict:
    """Compose the #205 run-info dict from the REAL run artifacts (launch.sh schedule +
    levelset_best.json + checkpoint log lines + resume/deploy npzs on disk +
    custom_grouped_backward perf line). Purely READS + composes the crash-safe readers
    above; NEVER raises (an empty dict/None per field on any failure). Returns a
    JSON-able dict consumed by ``_run_info_html``."""
    run_dir = _run_dir_for(watched, log_glob, run_dir_arg)
    try:
        config = parse_run_config(run_dir)
    except Exception:
        config = {"source": "none", "flags": {}, "groups": {}, "schedule": {}}
    schedule = dict(config.get("schedule") or {})
    # ckpt-every isn't a curriculum boundary; surface it for the checkpoint card.
    schedule["__ckpt_every"] = _int_flag(config.get("flags") or {}, "ckpt-every")
    last_epoch = live.get("last_epoch")
    last_seg_form = rows[-1].get("seg_form") if rows else None
    stage_prog = _stage_progress(last_epoch, last_seg_form, schedule)
    stage_prog["__last_epoch"] = last_epoch
    checkpoints = _parse_checkpoints(watched)
    # prefer the atomically-written levelset_best.json; fall back to the log 'best' line.
    best = _read_best_json(run_dir) or checkpoints.get("best")
    resume = _resume_status(run_dir, checkpoints, now=now)
    perf = _perf_env_status(watched)
    disk = _ckpt_disk_footprint(run_dir)
    first_ts = _first_verdict_ts(rows)
    throughput = _throughput_eta(live, schedule, stage_prog, first_ts, now=now)
    info = {"run_dir": (str(run_dir) if run_dir is not None else None),
            "config": config, "schedule": schedule, "stage_prog": stage_prog,
            "best": best, "checkpoints": checkpoints, "resume": resume,
            "perf": perf, "disk": disk, "throughput": throughput}
    # ── 2026-07-04: control-system telemetry (fired transitions / closed-loop rows /
    # ep0 nucleation gate / per-stage slopes / eikonal effective-weight series) + the
    # run's CONFIG ROLE (#205 diagnosed-erosion vs fresh seeded successor). All from
    # the run's OWN run.log + launch.sh; crash-safe (collect_control never raises).
    if _dct is not None:
        try:
            flags = config.get("flags") or {}
            info["control"] = _dct.collect_control(watched, flags, rows, schedule)
            info["role"] = _dct.run_role(flags)
        except Exception:
            info["control"], info["role"] = None, {}
    return info


def _render_once(args, state: dict | None = None) -> dict:
    """Render one cycle. ``state`` persists across watch cycles (in-process) so
    the watched log can be re-resolved (self-follow) and switches detected. The
    cadence calibration is persisted to DISK (``--cadence-state``) keyed by log
    name so it also survives a dashboard restart. The curriculum stage boundaries
    (tau/l7/muon) + the #205 run-info strip are read AUTOMATICALLY from the run's OWN
    launch.sh + artifacts (falling back to the --tau/--l7 CLI defaults for the PNG
    shading when no schedule is resolvable)."""
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
    # #205: read the REAL curriculum schedule + run-info from the run's own artifacts.
    run_info = _collect_run_info(watched, getattr(args, "log_glob", None),
                                 getattr(args, "run_dir", None), rows, live, now)
    sched = run_info.get("schedule") or {}
    # PNG stage shading uses the REAL boundaries when resolvable; else the CLI defaults.
    tau_png = sched.get("tau_start") if sched.get("tau_start") is not None else args.tau
    l7_png = sched.get("l7_start") if sched.get("l7_start") is not None else args.l7
    muon_png = sched.get("muon_start")
    best_dseg = (run_info.get("best") or {}).get("d_seg")
    # 2026-07-04: fresh-run A/B projection (measured-anchored; hypothesis band labeled)
    fresh_proj = None
    if _dtm is not None:
        try:
            fresh_proj = _dtm.build_fresh_run_projection(sched, args.goal_dseg)
        except Exception:
            fresh_proj = None
    png = _render_png(rows, tau_png, l7_png, args.goal_dseg, muon=muon_png,
                      best_dseg=best_dseg, control=run_info.get("control"),
                      fresh_proj=fresh_proj)
    _write_html(Path(args.out), png, rows, args.refresh_seconds, watched, live,
                state.get("switched_note"), getattr(args, "log_glob", None),
                tau=tau_png, l7=l7_png, goal_dseg=args.goal_dseg, run_info=run_info)
    return {"verdicts": len(rows),
            "watched": (watched.name if watched is not None else None),
            "kind": live["kind"], "cadence_source": live.get("cadence_source"),
            "stage": (run_info.get("stage_prog") or {}).get("stage"),
            "perf_active": (run_info.get("perf") or {}).get("active")}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", default=None,
                    help="explicit verdict log to watch (override; back-compat)")
    ap.add_argument("--log-glob", default=".omx/tmp/levelset_*.log",
                    help="self-follow: each cycle watch the NEWEST verdict-bearing "
                         "log matching this glob (ignored when --log is given)")
    ap.add_argument("--out", default=".omx/tmp/dash_levelset/index.html")
    ap.add_argument("--run-dir", default=None,
                    help="run directory holding launch.sh + the checkpoint npzs + "
                         "levelset_best.json (the #205 run-info source). Default None => "
                         "derived from the watched log's parent (or the --log-glob dir).")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--refresh-seconds", type=int, default=30)
    ap.add_argument("--tau", type=int, default=300,
                    help="FALLBACK CE->tau boundary for PNG shading when no launch.sh "
                         "schedule is resolvable (the REAL boundary is read from the run's "
                         "own launch.sh --tau-softplus-start-epoch when present).")
    ap.add_argument("--l7", type=int, default=900,
                    help="FALLBACK tau->l7 boundary for PNG shading (REAL value read from "
                         "the run's launch.sh --l7-start-epoch when present).")
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
