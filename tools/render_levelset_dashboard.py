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
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import signal
import time
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
        if logy and xy and min(v for v in ys if v > 0) > 0:
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


def _write_html(out: Path, png: bytes, rows: list[dict], refresh: int) -> None:
    b64 = base64.b64encode(png).decode("ascii")
    last = rows[-1] if rows else {}
    summary = (
        f"ep{last.get('epoch','?')} | d_seg={last.get('d_seg','?')} | "
        f"d_pose={last.get('d_pose','?')} | bytes={last.get('blob_bytes','?')} | "
        f"implied_S={last.get('implied_S','?')}"
        if rows else "waiting for ep0 verdict…"
    )
    html = (
        f'<!doctype html><html><head><meta charset="utf-8">'
        f'<meta http-equiv="refresh" content="{refresh}">'
        f"<title>level-set row dashboard</title>"
        f'<style>body{{font-family:-apple-system,monospace;background:#111;color:#ddd;'
        f"text-align:center;margin:0;padding:12px}}img{{max-width:98%}}"
        f".s{{font-size:13px;color:#9cf;margin:8px}}.a{{font-size:11px;color:#888}}</style></head>"
        f'<body><div class="s">{summary}</div>'
        f'<img src="data:image/png;base64,{b64}">'
        f'<div class="a">advisory [macOS-MLX training] NON-PROMOTABLE · pointer UNMOVED 0.19110 · '
        f"auto-refresh {refresh}s</div></body></html>"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp")
    tmp.write_text(html)
    os.replace(tmp, out)


def _render_once(args) -> int:
    rows = _parse_verdicts(Path(args.log))
    png = _render_png(rows, args.tau, args.l7, args.goal_dseg)
    _write_html(Path(args.out), png, rows, args.refresh_seconds)
    return len(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", required=True)
    ap.add_argument("--out", default=".omx/tmp/dash_levelset/index.html")
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--refresh-seconds", type=int, default=30)
    ap.add_argument("--tau", type=int, default=300)
    ap.add_argument("--l7", type=int, default=900)
    ap.add_argument("--goal-dseg", type=float, default=0.00112)
    args = ap.parse_args()

    stop = {"v": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("v", True))
    n = _render_once(args)
    print(json.dumps({"stage": "dashboard", "rendered": True, "verdicts": n, "out": args.out}))
    if not args.watch:
        return
    while not stop["v"]:
        for _ in range(args.refresh_seconds):
            if stop["v"]:
                break
            time.sleep(1)
        try:
            _render_once(args)
        except Exception as e:  # a transient read/render error must not kill the live dash
            print(json.dumps({"stage": "dashboard", "error": str(e)}))


if __name__ == "__main__":
    main()
