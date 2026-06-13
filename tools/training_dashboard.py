# SPDX-License-Identifier: MIT
"""Real-time training telemetry dashboard for torch-vehicle runs (one or many).

Reads each run's ``torch_vehicle_trajectory.jsonl`` (via
:func:`tac.torch_vehicle.telemetry.read_trajectory`) + ``torch_vehicle_summary.json``
and renders a 4-panel matplotlib figure with EVERY run overlaid (one color per run):

  1. training LOSS vs global epoch         (dense — logged every epoch)
  2. d_seg  vs global epoch                (eval epochs; the binding contest term)
  3. d_pose vs global epoch (log y)        (eval epochs; the pose axis)
  4. contest SCORE vs global epoch         (eval epochs; 100*d_seg + sqrt(10*d_pose)
                                            + 25*bytes/37_545_489 — the `score` field)

Output is a SELF-CONTAINED HTML (the PNG embedded base64) with a ``<meta refresh>`` so a
browser viewing the local file auto-updates. In ``--watch`` mode the dashboard regenerates
every ``--refresh-seconds`` — run it as a detached daemon and just keep the HTML open.

AUTHORITY (CLAUDE.md "Generated reports must preserve the axis label"): every score shown
is the IN-LOOP CPU-authority advisory metric — ``[contest-CPU advisory] NON-PROMOTABLE``
until ``upstream/evaluate.py`` on the byte-closed archive. The banner says so.

Usage:
    # one-shot (open the HTML it prints):
    .venv/bin/python tools/training_dashboard.py --once

    # live (detached daemon; regenerates every 20s):
    nohup .venv/bin/python tools/training_dashboard.py --watch --refresh-seconds 20 \
        --runs 'experiments/results/from0_ab_v2/*' \
        --out reports/training_dashboard.html < /dev/null >/dev/null 2>&1 & disown

    # then open reports/training_dashboard.html in a browser.
"""

from __future__ import annotations

import argparse
import base64
import glob
import io
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — no display needed
import matplotlib.pyplot as plt  # noqa: E402

from tac.torch_vehicle.telemetry import read_summary, read_trajectory  # noqa: E402

_RATE_DENOM = 37_545_489  # contest archive-size normalizer (CLAUDE.md score formula)
# A stable, color-blind-friendly palette; cycled if there are more runs than colors.
_PALETTE = [
    "#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e",
    "#17becf", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22",
]


def _resolve_run_dirs(patterns: list[str]) -> list[Path]:
    """Expand the --runs globs/paths to run dirs that have a trajectory file (or could).
    A run dir is any dir directly containing (or that will contain) the trajectory JSONL."""
    out: list[Path] = []
    seen: set[str] = set()
    for pat in patterns:
        for m in sorted(glob.glob(pat)):
            p = Path(m)
            if not p.is_dir():
                continue
            rp = str(p.resolve())
            if rp in seen:
                continue
            seen.add(rp)
            out.append(p)
    return out


def _series(rows: list[dict], yfield: str, *, eval_only: bool) -> tuple[list[int], list[float]]:
    """Extract (global_epoch, yfield) pairs from trajectory rows. ``eval_only`` keeps only
    rows whose ``yfield`` is a finite number (the eval-epoch fields are null off-eval)."""
    xs: list[int] = []
    ys: list[float] = []
    for r in rows:
        e = r.get("global_epoch", r.get("epoch"))
        v = r.get(yfield)
        if e is None:
            continue
        if v is None or not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            if eval_only:
                continue
            continue
        xs.append(int(e))
        ys.append(float(v))
    return xs, ys


def _latest_eval(rows: list[dict]) -> dict | None:
    """The most recent row with a finite ``score`` (an eval epoch)."""
    for r in reversed(rows):
        s = r.get("score")
        if isinstance(s, (int, float)) and math.isfinite(float(s)):
            return r
    return None


def _render_png(run_data: list[tuple[str, list[dict], str]]) -> bytes:
    """Render the 4-panel figure for all runs. ``run_data`` = [(label, rows, color)]."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    (ax_loss, ax_seg), (ax_pose, ax_score) = axes

    panels = [
        (ax_loss, "loss", "training loss", False, False),
        (ax_seg, "d_seg", "d_seg  (100·d_seg = binding term)", True, False),
        (ax_pose, "d_pose", "d_pose  (√(10·d_pose) term)", True, True),
        (ax_score, "score", "contest score  [contest-CPU advisory]", True, False),
    ]
    any_data = False
    for ax, field, title, eval_only, logy in panels:
        for label, rows, color in run_data:
            xs, ys = _series(rows, field, eval_only=eval_only)
            if not xs:
                continue
            any_data = True
            marker = "o" if eval_only else None
            ms = 4 if eval_only else 0
            ax.plot(xs, ys, color=color, label=label, marker=marker, markersize=ms, linewidth=1.5)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("global epoch", fontsize=9)
        ax.grid(True, alpha=0.3)
        if logy:
            ax.set_yscale("log")
        ax.tick_params(labelsize=8)
    # one shared legend (top-left panel) if there is anything to show
    handles, labels = ax_loss.get_legend_handles_labels()
    if not handles:  # loss may be empty very early; fall back to seg panel
        handles, labels = ax_seg.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 6),
                   fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.99))
    fig.suptitle("", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if not any_data:
        ax_loss.text(0.5, 0.5, "no telemetry yet…", transform=ax_loss.transAxes,
                     ha="center", va="center", fontsize=12)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return buf.getvalue()


def _stats_table(run_data: list[tuple[str, list[dict], str]], run_dirs: list[Path]) -> str:
    """An HTML table of per-run latest/best stats."""
    rows_html = []
    for (label, rows, color), rd in zip(run_data, run_dirs, strict=True):
        summ = read_summary(rd) or {}
        last = rows[-1] if rows else {}
        le = _latest_eval(rows)
        ep = last.get("global_epoch", last.get("epoch", "—"))
        loss = last.get("loss")
        best = summ.get("best_score")
        wc = last.get("wall_clock_s")

        def f(v, nd=4):
            return f"{v:.{nd}f}" if isinstance(v, (int, float)) and math.isfinite(v) else "—"

        seg = le.get("d_seg") if le else None
        pose = le.get("d_pose") if le else None
        sc = le.get("score") if le else None
        eep = le.get("global_epoch") if le else None
        rows_html.append(
            f"<tr>"
            f"<td><span style='display:inline-block;width:10px;height:10px;background:{color};"
            f"border-radius:50%'></span> {label}</td>"
            f"<td>{ep}</td><td>{f(loss,3)}</td>"
            f"<td>{f(seg,5)}</td><td>{f(pose,3)}</td><td>{f(sc,4)}</td>"
            f"<td>{f(best,4)}</td>"
            f"<td>{'@'+str(eep) if eep is not None else '—'}</td>"
            f"<td>{(str(int(wc))+'s') if isinstance(wc,(int,float)) else '—'}</td>"
            f"</tr>"
        )
    return (
        "<table><thead><tr><th>run</th><th>epoch</th><th>loss</th>"
        "<th>d_seg</th><th>d_pose</th><th>score</th><th>best_score</th>"
        "<th>eval@</th><th>wall</th></tr></thead><tbody>"
        + "".join(rows_html)
        + "</tbody></table>"
    )


def build_html(run_dirs: list[Path], refresh_seconds: int) -> str:
    run_data: list[tuple[str, list[dict], str]] = []
    for i, rd in enumerate(run_dirs):
        rows = read_trajectory(rd)
        label = rd.name
        run_data.append((label, rows, _PALETTE[i % len(_PALETTE)]))
    png = _render_png(run_data)
    b64 = base64.b64encode(png).decode("ascii")
    table = _stats_table(run_data, run_dirs)
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    meta = (
        f'<meta http-equiv="refresh" content="{int(refresh_seconds)}">'
        if refresh_seconds > 0 else ""
    )
    return f"""<!DOCTYPE html><html><head><meta charset="utf-8">{meta}
<title>training telemetry</title>
<style>
 body{{font-family:-apple-system,system-ui,sans-serif;margin:18px;background:#0d1117;color:#c9d1d9}}
 h1{{font-size:16px;margin:0 0 4px}} .sub{{color:#8b949e;font-size:12px;margin-bottom:12px}}
 .banner{{background:#3d2c00;color:#f0c674;padding:6px 10px;border-radius:6px;font-size:12px;margin-bottom:12px}}
 img{{width:100%;max-width:1300px;border-radius:8px;background:#fff}}
 table{{border-collapse:collapse;margin-top:14px;font-size:12px}}
 th,td{{border:1px solid #30363d;padding:4px 9px;text-align:right}} th{{background:#161b22}}
 td:first-child,th:first-child{{text-align:left}}
</style></head><body>
<h1>torch-vehicle training telemetry — {len(run_dirs)} run(s)</h1>
<div class="sub">updated {now} · auto-refresh {refresh_seconds}s · runs: {", ".join(r.name for r in run_dirs)}</div>
<div class="banner">⚠ in-loop CPU-authority advisory metric — <b>[contest-CPU advisory] NON-PROMOTABLE</b>
 until upstream/evaluate.py on the byte-closed archive. score = 100·d_seg + √(10·d_pose) + 25·bytes/{_RATE_DENOM:,}</div>
<img src="data:image/png;base64,{b64}"/>
{table}
</body></html>"""


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+",
                    default=["experiments/results/from0_ab_v2/*"],
                    help="glob(s)/paths to run dirs (default: the from-0 5-arm sweep).")
    ap.add_argument("--out", default="reports/training_dashboard.html")
    ap.add_argument("--refresh-seconds", type=int, default=20)
    ap.add_argument("--watch", action="store_true",
                    help="regenerate forever every --refresh-seconds (run detached).")
    ap.add_argument("--once", action="store_true", help="render once and exit.")
    args = ap.parse_args(argv)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def render_once() -> int:
        run_dirs = _resolve_run_dirs(args.runs)
        if not run_dirs:
            # still write a placeholder so the page exists
            out.write_text(
                f"<html><body style='font-family:sans-serif'>no run dirs match "
                f"{args.runs} yet…</body></html>"
            )
            return 0
        html = build_html(run_dirs, 0 if args.once else args.refresh_seconds)
        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.write_text(html)
        os.replace(tmp, out)  # atomic so the browser never reads a half-written file
        return len(run_dirs)

    if args.watch:
        print(f"[dashboard] watching {args.runs} -> {out} every {args.refresh_seconds}s "
              f"(Ctrl-C / kill to stop)", flush=True)
        while True:
            try:
                n = render_once()
                print(f"[dashboard] rendered {n} run(s) at "
                      f"{datetime.now(timezone.utc).strftime('%H:%M:%SZ')}", flush=True)
            except Exception as e:  # never let a transient read error kill the daemon
                print(f"[dashboard] WARN render error: {e}", flush=True)
            time.sleep(max(2, args.refresh_seconds))
    else:
        n = render_once()
        print(f"[dashboard] wrote {out} ({n} run(s)). Open it in a browser.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
