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

# ── self-calibration defaults (data-derived staleness; NOT hand-tuned bands) ──
_CADENCE_K = 2.5  # STALE only when verdict-age > K × measured cadence
_STALE_FLOOR_MIN = 10.0  # sane minimum stale threshold (minutes), even for fast cadences
_CADENCE_PRIOR_MIN = 18.0  # labeled prior used until cadence is measured (~25 ep cluster)
_MIN_CADENCE_SAMPLES = 3  # measured NEW-verdict arrivals needed before cadence is "measured"
_DEFAULT_EPOCH_SPACING = 25  # fallback verdict-epoch spacing for the next-verdict estimate
_CADENCE_STATE_DEFAULT = ".omx/tmp/dash_levelset_cadence.json"

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


def _measure_cadence(arrivals: list[tuple[int, float, str]], baseline: set, cfg: dict) -> tuple[float, str, int]:
    """Median inter-arrival (seconds) of NEW verdict epochs.

    A sample counts as MEASURED iff it has an embedded ts OR it was observed
    AFTER launch (epoch not in the launch baseline). Returns
    ``(cadence_s, source, n_measured)``: ('measured', median) once
    ``>= cfg['min_samples']`` measured arrivals exist (>= min_samples-1 positive
    intervals); otherwise ('prior', cfg prior) with the running sample count.
    """
    measured = sorted(
        [(ep, a) for (ep, a, src) in arrivals if src == "ts" or ep not in baseline],
        key=lambda t: t[0],
    )
    n = len(measured)
    if n >= cfg["min_samples"]:
        diffs = [b[1] - a[1] for a, b in zip(measured, measured[1:]) if b[1] > a[1]]
        if len(diffs) >= cfg["min_samples"] - 1:
            return _median(diffs), "measured", n
    return cfg["prior_s"], "prior", n


def _epoch_spacing(rows: list[dict], default: int) -> int:
    eps = sorted({int(r["epoch"]) for r in rows})
    diffs = [b - a for a, b in zip(eps, eps[1:]) if b > a]
    return int(_median(diffs)) if diffs else default


def _compute_liveness(rows: list[dict], watched_mtime: float | None, now: float,
                      sub: dict, cfg: dict) -> dict:
    """The HEADLINE, DATA-DERIVED live/stale judgment.

    ``watched_mtime`` is the watched log's file mtime (None when no log). ``sub``
    is the per-log cadence sub-state (mutated with new observations). Returns a
    rich dict the banner renders verbatim. NO IO here (pure + testable)."""
    log_age = (now - watched_mtime) if watched_mtime is not None else None
    base = {
        "kind": "missing", "last_epoch": None, "verdict_age_s": None,
        "log_age_s": log_age, "cadence_s": None, "cadence_source": None,
        "n_samples": 0, "next_epoch": None, "next_eta_s": None,
        "threshold_s": None, "epoch_spacing": cfg["default_spacing"],
    }
    if watched_mtime is None:
        return base
    arrivals, baseline = _arrival_times(rows, sub, now)
    cadence_s, cad_src, n = _measure_cadence(arrivals, baseline, cfg)
    spacing = _epoch_spacing(rows, cfg["default_spacing"])
    last_epoch = arrivals[-1][0] if arrivals else None
    last_arr = arrivals[-1][1] if arrivals else None
    verdict_age = (now - last_arr) if last_arr is not None else None
    threshold = max(cfg["floor_s"], cfg["k"] * cadence_s)
    next_epoch = (last_epoch + spacing) if last_epoch is not None else None
    next_eta = max(0.0, cadence_s - verdict_age) if verdict_age is not None else None

    if last_epoch is None:
        kind = "warming_up"
    elif cad_src == "measured":
        kind = "stale" if (verdict_age is not None and verdict_age > threshold) else "live"
    else:  # cadence not yet calibrated → warming up, but still catch a dead process
        dead = (verdict_age is not None and verdict_age > threshold) or \
               (log_age is not None and log_age > cfg["floor_s"])
        kind = "stale" if dead else "warming_up"

    return {
        "kind": kind, "last_epoch": last_epoch, "verdict_age_s": verdict_age,
        "log_age_s": log_age, "cadence_s": cadence_s, "cadence_source": cad_src,
        "n_samples": n, "next_epoch": next_epoch, "next_eta_s": next_eta,
        "threshold_s": threshold, "epoch_spacing": spacing,
    }


def _cfg_from_args(args) -> dict:
    """Build the cadence config. ``--stale-min`` (when given) is the HARD FLOOR
    override; otherwise the floor is ``--stale-floor-min``. Self-calibration
    (K × measured cadence) applies on top either way via ``max(floor, ...)``."""
    stale_min = getattr(args, "stale_min", None)
    floor_min = stale_min if stale_min is not None else getattr(args, "stale_floor_min", _STALE_FLOOR_MIN)
    return {
        "k": float(getattr(args, "cadence_k", _CADENCE_K)),
        "floor_s": float(floor_min) * 60.0,
        "prior_s": float(getattr(args, "cadence_prior_min", _CADENCE_PRIOR_MIN)) * 60.0,
        "min_samples": _MIN_CADENCE_SAMPLES,
        "default_spacing": _DEFAULT_EPOCH_SPACING,
    }


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
def _stage_label(last_epoch: int | None, seg_form: str | None, tau: int, l7: int) -> str:
    if last_epoch is None:
        return "starting"
    if last_epoch < tau:
        return f"CE warmup (→ tau@{tau})"
    if last_epoch < l7:
        return f"tau stage ({seg_form or 'softplus'}) (→ l7@{l7})"
    return f"l7 finisher / Muon (ep ≥ {l7})"


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
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.2))
    fig.patch.set_facecolor(_BG)
    head = (
        f"level-set witness — {len(rows)} verdicts"
        + (f" — last ep{rows[-1]['epoch']}" if rows else " — (waiting for ep0)")
    )
    fig.suptitle(head, fontsize=13, color=_FG, fontweight="bold")
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
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=92, facecolor=_BG)
    plt.close(fig)
    return buf.getvalue()


def _banner_html(live: dict | None, watched: Path | None, switched_note: str | None,
                 log_glob: str | None) -> str:
    """Build the data-derived status banner + the two honest recency facts +
    the next-verdict estimate + watched-log line + switch note. The judgment is
    keyed to the OBSERVED verdict cadence (live/warming/stale), never a hand-set
    minute band — so it stays accurate across CE/tau/l7/Muon cadences, or flags
    when it is still calibrating."""
    if live is None:
        return ""
    kind = live.get("kind")
    verdict_age = live.get("verdict_age_s")
    log_age = live.get("log_age_s")
    last_ep = live.get("last_epoch")
    cadence_s = live.get("cadence_s")
    cad_src = live.get("cadence_source")
    next_ep = live.get("next_epoch")
    next_eta = live.get("next_eta_s")
    n_samples = live.get("n_samples", 0)

    cad_m = f"{cadence_s / 60.0:.0f}m" if cadence_s else "?"
    # the two honest, distinct facts (never blurred)
    verdict_fact = (f"last verdict ep{last_ep} ({_fmt_age(verdict_age)} ago)"
                    if last_ep is not None else "no verdict yet")
    log_fact = f"log active {_fmt_age(log_age)} ago" if log_age is not None else ""
    next_fact = (f"next ~ep{next_ep} (~{_fmt_age(next_eta)})"
                 if next_ep is not None and next_eta is not None else "")

    if kind == "missing":
        glob_hint = f" (glob: {log_glob})" if log_glob else ""
        banner = f'<div class="banner stale">⚠ no run log found{glob_hint}</div>'
    elif kind == "stale":
        cad_note = f">{_CADENCE_K:g}x the ~{cad_m} {cad_src or ''} cadence".strip()
        banner = (
            f'<div class="banner stale">⚠ STALE — no verdict in '
            f'{_fmt_age(verdict_age)} ({cad_note}) — likely STOPPED · {log_fact}</div>'
        )
    elif kind == "warming_up":
        prog = f"{n_samples}/{_MIN_CADENCE_SAMPLES}"
        banner = (
            f'<div class="banner warm">● warming up — calibrating cadence '
            f'(prior ~{cad_m}, {prog} samples) · {verdict_fact} · {next_fact} · {log_fact}</div>'
        )
    else:  # live (cadence measured)
        banner = (
            f'<div class="banner live">● live · {verdict_fact} · {next_fact} · '
            f'cadence ~{cad_m} (measured) · {log_fact}</div>'
        )

    watched_line = ""
    if watched is not None:
        watched_line = f'<div class="w">watched: {Path(watched).name}</div>'
    switch_line = f'<div class="sw">{switched_note}</div>' if switched_note else ""
    return banner + watched_line + switch_line


def _headline_html(rows: list[dict], live: dict | None, tau: int, l7: int,
                   goal_dseg: float) -> str:
    """Big, self-explanatory headline: current d_seg + trend arrow + plain-language
    'what stage / which direction' status line."""
    if not rows:
        return '<div class="hl">waiting for ep0 verdict…</div>'
    last = rows[-1]
    d_seg = last.get("d_seg")
    arrow, direction = _trend(rows, "d_seg")
    last_ep = live.get("last_epoch") if live else last.get("epoch")
    stage = _stage_label(last_ep, last.get("seg_form"), tau, l7)
    dseg_str = f"{d_seg:.5f}" if isinstance(d_seg, (int, float)) else "?"
    ratio = (d_seg / goal_dseg) if isinstance(d_seg, (int, float)) and goal_dseg else None
    ratio_str = f"{ratio:.1f}× the sub-0.19 goal" if ratio else ""
    cls = "down" if direction == "descending" else ("up" if direction == "rising" else "flat")
    return (
        f'<div class="hl">d_seg <b>{dseg_str}</b> '
        f'<span class="arr {cls}">{arrow} {direction}</span>'
        f'<span class="goal">goal &lt;{goal_dseg:g} · {ratio_str}</span></div>'
        f'<div class="status">de-confound witness · {stage} · '
        f'l7 finisher at ep{l7}, Muon after</div>'
    )


def _legend_html() -> str:
    return (
        '<div class="legend">'
        'stage shading: <span class="lg ce">CE</span> '
        '<span class="lg tau">tau</span> '
        '<span class="lg l7">l7/Muon</span> &nbsp;|&nbsp; '
        '<b>● live</b> cadence measured, fresh · '
        '<b>● warming up</b> calibrating cadence (prior) · '
        '<b>⚠ stale</b> no verdict in &gt;2.5× the cadence (likely stopped) &nbsp;|&nbsp; '
        'panels: d_seg/d_pose LOWER=better · bytes=learned payload · '
        'implied_S is ADVISORY (not the contest score)'
        '</div>'
    )


def _write_html(out: Path, png: bytes, rows: list[dict], refresh: int,
                watched: Path | None = None, live: dict | None = None,
                switched_note: str | None = None, log_glob: str | None = None,
                tau: int = 300, l7: int = 900, goal_dseg: float = 0.00112) -> None:
    b64 = base64.b64encode(png).decode("ascii")
    last = rows[-1] if rows else {}
    summary = (
        f"ep{last.get('epoch', '?')} | d_seg={last.get('d_seg', '?')} | "
        f"d_pose={last.get('d_pose', '?')} | bytes={last.get('blob_bytes', '?')} | "
        f"implied_S={last.get('implied_S', '?')}"
        if rows else "waiting for ep0 verdict…"
    )
    banner = _banner_html(live, watched, switched_note, log_glob)
    headline = _headline_html(rows, live, tau, l7, goal_dseg)
    legend = _legend_html()
    html = (
        '<!doctype html><html><head><meta charset="utf-8">'
        f'<meta http-equiv="refresh" content="{refresh}">'
        "<title>level-set row dashboard</title>"
        "<style>"
        f"body{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',monospace;"
        f"background:{_BG};color:{_FG};text-align:center;margin:0;padding:14px 12px}}"
        "img{max-width:98%;border-radius:6px}"
        f".s{{font-size:12px;color:{_MUTED};margin:6px;font-variant-numeric:tabular-nums}}"
        f".hl{{font-size:24px;color:{_FG};margin:6px;font-weight:600;letter-spacing:.2px}}"
        f".hl b{{color:{_ACC};font-size:30px}}"
        ".arr{font-size:14px;margin-left:10px;padding:2px 8px;border-radius:10px}"
        f".arr.down{{background:#14431f;color:#b6f5c4}}.arr.up{{background:#5a1f1f;color:#ffd0d0}}"
        f".arr.flat{{background:#33384a;color:{_FG}}}"
        f".goal{{font-size:12px;color:{_GOAL};margin-left:12px}}"
        f".status{{font-size:13px;color:{_ACC};margin:4px 0 10px}}"
        ".banner{padding:9px;font-size:14px;font-weight:bold;border-radius:6px;"
        "margin:8px auto;max-width:94%}"
        ".banner.stale{background:#7a1f1f;color:#ffdede}"
        ".banner.live{background:#14431f;color:#b6f5c4}"
        ".banner.warm{background:#5a4a14;color:#fff0c2}"
        f".w{{font-size:11px;color:{_MUTED};margin:2px}}.sw{{font-size:11px;color:{_ACC};margin:2px}}"
        f".legend{{font-size:11px;color:{_MUTED};margin:8px auto;max-width:94%;line-height:1.6}}"
        ".lg{padding:1px 6px;border-radius:4px;color:#fff}"
        f".lg.ce{{background:{_STAGE_CE}}}.lg.tau{{background:{_STAGE_TAU}}}.lg.l7{{background:{_STAGE_L7}}}"
        f".a{{font-size:11px;color:{_MUTED};margin-top:8px}}"
        "</style></head>"
        f"<body>{banner}{headline}"
        f'<div class="s">{summary}</div>'
        f'<img src="data:image/png;base64,{b64}">'
        f"{legend}"
        f'<div class="a">advisory [macOS-MLX training] NON-PROMOTABLE · '
        f"pointer UNMOVED 0.19110 · auto-refresh {refresh}s</div></body></html>"
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
    live = _compute_liveness(rows, mtime, now, sub, cfg)
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
                    help="OPTIONAL hard floor (minutes) override for the stale "
                         "threshold; replaces --stale-floor-min in max(floor, "
                         "K×cadence). UNSET (default) = fully self-calibrated.")
    ap.add_argument("--stale-floor-min", type=float, default=_STALE_FLOOR_MIN,
                    help=f"sane minimum stale threshold in minutes (default {_STALE_FLOOR_MIN:g}); "
                         "STALE fires at max(this, K×measured_cadence)")
    ap.add_argument("--cadence-k", type=float, default=_CADENCE_K,
                    help=f"STALE multiplier on measured cadence (default {_CADENCE_K:g})")
    ap.add_argument("--cadence-prior-min", type=float, default=_CADENCE_PRIOR_MIN,
                    help=f"labeled cadence prior (minutes) used until measured (default {_CADENCE_PRIOR_MIN:g})")
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
