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

SELF-FOLLOWING + STALENESS-HONEST (2026-06-27): pass ``--log-glob PATTERN``
(default ``.omx/tmp/levelset_*.log``) instead of (or in addition to) ``--log``.
Each refresh cycle the watched log is RE-RESOLVED to the NEWEST-mtime file
matching the glob that actually CONTAINS verdict lines — so when a new run
starts (new log file) the dashboard auto-switches to it, and the dashboard
never follows its OWN daemon log / a non-verdict log (the confounder that let
a DEAD run render as "live" for hours). A staleness banner keyed to the
WATCHED LOG's mtime makes a stopped/crashed source impossible to miss:
``● live`` when fresh, a LOUD ``⚠ STALE`` banner when age > ``--stale-min``
(default 5 min). ``--log`` remains the explicit-override path (back-compat).
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
    """Compute live/stale/missing state keyed to the WATCHED LOG's real mtime
    (verdict lines carry no timestamp, so file mtime is the honest 'is the data
    source still being written?' signal). ``age_s`` is now - mtime."""
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


def _render_png(rows: list[dict], tau: int, l7: int, goal_dseg: float) -> bytes:
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    fig.suptitle(
        f"level-set witness row — {len(rows)} verdicts"
        + (f" — last ep{rows[-1]['epoch']}" if rows else " — (waiting for ep0)"),
        fontsize=13,
    )
    ep = [r["epoch"] for r in rows]

    def _panel(ax, key, title, logy=False, hline=None, hlabel=None):
        y = [r.get(key) for r in rows]
        xy = [(e, v) for e, v in zip(ep, y) if v is not None]
        if xy:
            xs, ys = zip(*xy)
            ax.plot(xs, ys, "-o", ms=4, color="#1f77b4")
            if len(xs) == 1:
                ax.annotate(f"{ys[0]:.4g}", (xs[0], ys[0]))
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("epoch")
        # log scale only when there is >=1 strictly-positive value (an empty `min(...)` on a
        # generator with no positives raises ValueError; d_seg/d_pose CAN be exactly 0 in a
        # degenerate/early verdict, and this render is also called once OUTSIDE the watch-loop
        # try/except, so guard it so a 0-valued point can never crash the live daemon).
        if logy and xy and any(v > 0 for v in ys):
            ax.set_yscale("log")
        for x, lab, col in ((tau, "tau", "#ff7f0e"), (l7, "l7", "#d62728")):
            ax.axvline(x, ls="--", lw=1, color=col, alpha=0.6)
            ax.text(x, ax.get_ylim()[1], lab, color=col, fontsize=8, va="top")
        if hline is not None:
            ax.axhline(hline, ls=":", lw=1.2, color="green", alpha=0.8)
            ax.text(0, hline, hlabel or "", color="green", fontsize=8, va="bottom")
        ax.grid(alpha=0.25)

    _panel(axes[0, 0], "d_seg", "d_seg (realized argmax) — goal <0.00112", logy=True,
           hline=goal_dseg, hlabel="sub-0.19 goal")
    _panel(axes[0, 1], "d_pose", "d_pose (realized) — target ~9e-4", logy=True,
           hline=9e-4, hlabel="parent existence-proof")
    _panel(axes[1, 0], "blob_bytes", "blob_bytes (learned payload)")
    _panel(axes[1, 1], "implied_S", "implied_S (advisory) — frontier 0.19110", logy=True,
           hline=0.19110, hlabel="pointer")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=90)
    plt.close(fig)
    return buf.getvalue()


def _staleness_banner_html(stale: dict | None, watched: Path | None,
                           switched_note: str | None, log_glob: str | None) -> str:
    """Build the LOUD staleness banner + watched-log line + switch note. This
    is the honesty layer: a stopped/crashed data source can never render as
    'live' because the banner is keyed to the watched log's real mtime."""
    if stale is None:
        return ""
    state = stale.get("state")
    if state == "missing":
        glob_hint = f" (glob: {log_glob})" if log_glob else ""
        banner = f'<div class="banner stale">⚠ no run log found{glob_hint}</div>'
    elif state == "stale":
        banner = (
            f'<div class="banner stale">⚠ STALE — no new verdict in '
            f'{_fmt_age(stale.get("age_s"))} — run may be STOPPED/crashed</div>'
        )
    else:  # live
        banner = (
            f'<div class="banner live">● live (updated '
            f'{_fmt_age(stale.get("age_s"))} ago)</div>'
        )
    watched_line = ""
    if watched is not None and stale.get("age_s") is not None:
        watched_line = (
            f'<div class="w">watched: {Path(watched).name} · '
            f'last update {_fmt_age(stale.get("age_s"))} ago</div>'
        )
    switch_line = f'<div class="sw">{switched_note}</div>' if switched_note else ""
    return banner + watched_line + switch_line


def _write_html(out: Path, png: bytes, rows: list[dict], refresh: int,
                watched: Path | None = None, stale: dict | None = None,
                switched_note: str | None = None, log_glob: str | None = None) -> None:
    b64 = base64.b64encode(png).decode("ascii")
    last = rows[-1] if rows else {}
    summary = (
        f"ep{last.get('epoch','?')} | d_seg={last.get('d_seg','?')} | "
        f"d_pose={last.get('d_pose','?')} | bytes={last.get('blob_bytes','?')} | "
        f"implied_S={last.get('implied_S','?')}"
        if rows else "waiting for ep0 verdict…"
    )
    banner = _staleness_banner_html(stale, watched, switched_note, log_glob)
    html = (
        f'<!doctype html><html><head><meta charset="utf-8">'
        f'<meta http-equiv="refresh" content="{refresh}">'
        f"<title>level-set row dashboard</title>"
        f'<style>body{{font-family:-apple-system,monospace;background:#111;color:#ddd;'
        f"text-align:center;margin:0;padding:12px}}img{{max-width:98%}}"
        f".s{{font-size:13px;color:#9cf;margin:8px}}.a{{font-size:11px;color:#888}}"
        f".banner{{padding:9px;font-size:15px;font-weight:bold;border-radius:5px;"
        f"margin:8px auto;max-width:92%}}"
        f".stale{{background:#7a1f1f;color:#ffdede}}.live{{background:#14431f;color:#b6f5c4}}"
        f".w{{font-size:12px;color:#aaa;margin:2px}}.sw{{font-size:11px;color:#6cf;margin:2px}}"
        f"</style></head>"
        f"<body>{banner}"
        f'<div class="s">{summary}</div>'
        f'<img src="data:image/png;base64,{b64}">'
        f'<div class="a">advisory [macOS-MLX training] NON-PROMOTABLE · pointer UNMOVED 0.19110 · '
        f"auto-refresh {refresh}s</div></body></html>"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(html)
    os.replace(tmp, out)


def _render_once(args, state: dict | None = None) -> dict:
    """Render one cycle. ``state`` persists across watch cycles so the watched
    log can be re-resolved (self-follow) and switches detected."""
    if state is None:
        state = {}
    watched = _resolve_watched_log(getattr(args, "log", None), getattr(args, "log_glob", None))
    note = _detect_switch(state.get("watched_name"), watched, _utc_now())
    if note is not None:
        state["switched_note"] = note
    if watched is not None:
        state["watched_name"] = watched.name
    rows = _parse_verdicts(watched) if watched is not None else []
    stale = _staleness(watched, getattr(args, "stale_min", 5.0))
    png = _render_png(rows, args.tau, args.l7, args.goal_dseg)
    _write_html(Path(args.out), png, rows, args.refresh_seconds, watched, stale,
                state.get("switched_note"), getattr(args, "log_glob", None))
    return {"verdicts": len(rows),
            "watched": (watched.name if watched is not None else None),
            "stale": stale["state"]}


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
    ap.add_argument("--stale-min", type=float, default=5.0,
                    help="age (minutes) of the watched log's last write beyond "
                         "which the LOUD stale banner fires (default 5)")
    args = ap.parse_args()

    stop = {"v": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("v", True))
    state: dict = {}
    info = _render_once(args, state)
    print(json.dumps({"stage": "dashboard", "rendered": True, "verdicts": info["verdicts"],
                      "watched": info["watched"], "stale": info["stale"], "out": args.out}))
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
