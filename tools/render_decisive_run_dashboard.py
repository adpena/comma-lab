#!/usr/bin/env python
"""Beautiful, stage-aware, estimated-vs-actual live dashboard for the decisive PR95-curriculum run.

Renders a self-contained ``index.html`` (embedded PNG, meta-refresh) showing, for ONE training run:
  * a header with the live operating point (stage, epoch/total, d_seg, d_pose, S, best) + ETA + thresholds,
  * 4 panels over global_epoch with the 8 curriculum STAGES drawn as labelled colored bands + boundary lines:
      1. contest score S          (+ target lines: beat-frontier 0.19110, sub-0.15)
      2. d_seg  (the binding term) (+ ESTIMATED power-law projection (dashed) + target d_seg lines)
      3. d_pose (log y)
      4. training loss (per-epoch)

"Estimated vs actual": ACTUAL = solid measured exact-eval points; ESTIMATED = the schedule's stage layout
(bands) + a power-law projection of d_seg fit to the actual points + the target threshold lines. PR95 itself
published no trajectory, so no author curve exists to overlay — these are the honest estimates.

ALL values are ``[contest-CPU advisory]`` (macOS), NON-PROMOTABLE — NOT an authoritative contest score.

Usage:
    .venv/bin/python tools/render_decisive_run_dashboard.py \
        --run-dir experiments/results/<run> --schedule-csv docs/.../pr95_8stage_schedule.csv \
        --out .omx/tmp/dashboard_serve/index.html --watch --refresh-seconds 20
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

_RATE_DENOM = 37_545_489
_FRONTIER_S = 0.19110  # the borrowed exact frontier (contest-CPU) we must beat
_TARGET_S = 0.15

# Stage band palette (8 stages, distinct but muted for a dark theme).
_STAGE_COLORS = [
    "#1f3a5f", "#243f5a", "#2a4858", "#345b4f", "#4a5d3a",
    "#5f5630", "#6b4630", "#7a2f3a",
]


def _read_trajectory(run_dir: Path) -> list[dict]:
    p = run_dir / "torch_vehicle_trajectory.jsonl"
    rows = []
    if p.exists():
        for line in p.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _read_summary(run_dir: Path) -> dict:
    p = run_dir / "torch_vehicle_summary.json"
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def _read_schedule(csv_path: Path) -> list[dict]:
    out = []
    if csv_path.exists():
        with csv_path.open() as f:
            for r in csv.DictReader(f):
                out.append(r)
    return out


def _stage_bounds(schedule: list[dict]) -> list[tuple[str, int, int]]:
    """Cumulative [start,end) global-epoch extent per stage from the schedule (the ESTIMATED layout)."""
    bounds, cum = [], 0
    for s in schedule:
        ep = int(s["epochs"])
        bounds.append((s["stage"], cum, cum + ep))
        cum += ep
    return bounds


def _powerlaw_projection(eps: list[float], ys: list[float], x_end: float):
    """Fit y = a * x^b on the (eps, ys) actual d_seg points; return (xs, ys_proj) to x_end. Honest estimate."""
    import numpy as np

    e = np.asarray(eps, float)
    y = np.asarray(ys, float)
    m = (e > 0) & (y > 0)
    if m.sum() < 3:
        return None
    b, loga = np.polyfit(np.log(e[m]), np.log(y[m]), 1)
    a = float(np.exp(loga))
    xs = np.linspace(float(e[m].min()), float(x_end), 200)
    return xs, a * xs ** b


def _fmt(x, nd=5):
    return "n/a" if x is None else f"{x:.{nd}f}"


def _measure_spe(traj: list[dict], window_ep: int = 1500) -> float | None:
    """Measured seconds/epoch from cumulative wall_clock_s over the most-recent window.

    CRITICAL (encodes a real prior error): use the CURRENT-STAGE recent-window rate, NEVER
    the whole-trace average and NEVER a hardcoded constant. Stages 1-4 (CE, no C1a/QAT) run
    ~2-4x faster than stages 5-8 (C1a sample_size=2000 + QAT + Muon), so the whole-trace mean
    (~5.7 s/ep) badly UNDER-estimates the remaining all-slow-stage time. A stale hardcoded
    1.78 s/ep previously produced a ~7h ETA when the truth was ~2.3 days. The recent window is
    the honest rate for the epochs that remain.
    """
    wc = sorted(
        (r["global_epoch"], r["wall_clock_s"])
        for r in traj
        if r.get("wall_clock_s") is not None and r.get("global_epoch") is not None
    )
    if len(wc) < 2:
        return None
    cur_ep = wc[-1][0]
    cand = [(e, t) for e, t in wc if e >= cur_ep - window_ep]
    if len(cand) < 2:
        cand = wc
    (e0, t0), (e1, t1) = cand[0], cand[-1]
    return (t1 - t0) / (e1 - e0) if e1 > e0 else None


def _fmt_dur(hours: float | None) -> str:
    if hours is None:
        return "n/a"
    return f"{hours:.1f} h" if hours < 48 else f"{hours / 24:.1f} d"


def render(run_dir: Path, schedule_csv: Path, out: Path) -> None:
    traj = _read_trajectory(run_dir)
    summ = _read_summary(run_dir)
    sched = _read_schedule(schedule_csv)
    bounds = _stage_bounds(sched)
    total_epochs = bounds[-1][2] if bounds else None

    evals = [r for r in traj if r.get("d_seg") is not None]
    le = summ.get("last_eval") or {}  # summary may carry last_eval=null pre-first-eval (async eval lag); None.get crashes
    cur_dseg = le.get("d_seg")
    cur_dpose = le.get("d_pose")
    cur_S = le.get("score")
    cur_ep = le.get("global_epoch")
    cur_stage = le.get("stage_name", "?")
    best = summ.get("best_score")
    best_ep = summ.get("best_ep")

    # target d_seg lines using the run's CURRENT pose+rate terms (so they're the live break-even).
    pose_term = (10 * cur_dpose) ** 0.5 if cur_dpose else 0.0
    rate_term = 25 * (le.get("archive_bytes", 0) or 0) / _RATE_DENOM
    dseg_beat = (_FRONTIER_S - pose_term - rate_term) / 100 if cur_dpose else None
    dseg_sub015 = (_TARGET_S - pose_term - rate_term) / 100 if cur_dpose else None

    plt.style.use("dark_background")
    fig, axes = plt.subplots(4, 1, figsize=(13, 16), sharex=True)
    fig.subplots_adjust(hspace=0.12, top=0.93, bottom=0.05, left=0.09, right=0.97)

    def draw_stage_bands(ax, ymin=None, ymax=None):
        for i, (_name, a, b) in enumerate(bounds):
            ax.axvspan(a, b, color=_STAGE_COLORS[i % len(_STAGE_COLORS)], alpha=0.35, lw=0)
            ax.axvline(a, color="#666", lw=0.5, ls=":")
        if cur_ep is not None:
            ax.axvline(cur_ep, color="#ff5577", lw=1.4, label="now")

    # --- panel 1: S ---
    ax = axes[0]
    draw_stage_bands(ax)
    if evals:
        ax.plot([r["global_epoch"] for r in evals], [r["score"] for r in evals],
                "-o", color="#7fd1ff", ms=3, lw=1.4, label="S (actual)")
    ax.axhline(_FRONTIER_S, color="#ffd24a", ls="--", lw=1.2, label=f"frontier {_FRONTIER_S} (beat)")
    ax.axhline(_TARGET_S, color="#7CFC00", ls="--", lw=1.2, label=f"target {_TARGET_S}")
    ax.set_ylabel("contest S")
    ax.set_yscale("log")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("contest score S = 100·d_seg + √(10·d_pose) + 25·bytes/37.5M   [contest-CPU advisory — NON-PROMOTABLE]",
                 fontsize=10)

    # --- panel 2: d_seg (binding) + ESTIMATED projection ---
    ax = axes[1]
    draw_stage_bands(ax)
    if evals:
        e = [r["global_epoch"] for r in evals]
        y = [r["d_seg"] for r in evals]
        ax.plot(e, y, "-o", color="#ff9f5a", ms=3, lw=1.4, label="d_seg (actual)")
        proj = _powerlaw_projection(e, y, total_epochs or max(e))
        if proj is not None:
            ax.plot(proj[0], proj[1], ls="--", color="#ffcf9a", lw=1.2, label="d_seg (estimated power-law)")
    if dseg_beat:
        ax.axhline(dseg_beat, color="#ffd24a", ls="--", lw=1.0, label=f"beat-0.191 d_seg≈{dseg_beat:.1e}")
    if dseg_sub015:
        ax.axhline(dseg_sub015, color="#7CFC00", ls="--", lw=1.0, label=f"sub-0.15 d_seg≈{dseg_sub015:.1e}")
    ax.set_ylabel("d_seg (binding)")
    ax.set_yscale("log")
    ax.legend(loc="upper right", fontsize=8)

    # --- panel 3: d_pose ---
    ax = axes[2]
    draw_stage_bands(ax)
    if evals:
        ax.plot([r["global_epoch"] for r in evals], [r["d_pose"] for r in evals],
                "-o", color="#b48cff", ms=3, lw=1.4, label="d_pose (actual)")
    ax.set_ylabel("d_pose")
    ax.set_yscale("log")
    ax.legend(loc="upper right", fontsize=8)

    # --- panel 4: loss (per-epoch) ---
    ax = axes[3]
    draw_stage_bands(ax)
    losses = [(r.get("global_epoch"), r.get("loss")) for r in traj if r.get("loss") is not None and r.get("global_epoch") is not None]
    if losses:
        ax.plot([a for a, _ in losses], [b for _, b in losses], color="#9aa7b3", lw=0.7, label="train loss")
    ax.set_ylabel("loss")
    ax.set_xlabel("global epoch")
    ax.legend(loc="upper right", fontsize=8)

    # stage-band legend across the top
    handles = [Patch(facecolor=_STAGE_COLORS[i % len(_STAGE_COLORS)], alpha=0.5, label=b[0]) for i, b in enumerate(bounds)]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=7, frameon=False, bbox_to_anchor=(0.5, 0.995))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, facecolor="#0d1117")
    plt.close(fig)
    png_b64 = base64.b64encode(buf.getvalue()).decode()

    pct = (100.0 * cur_ep / total_epochs) if (cur_ep and total_epochs) else 0.0
    # MEASURED current-stage rate (NOT a hardcoded constant; see _measure_spe docstring).
    spe = _measure_spe(traj)
    stage8_start = bounds[-1][1] if bounds else None  # Muon (last scheduled stage) = the decisive d_seg read

    def _eta_h(target_ep):
        if not (spe and cur_ep and target_ep and target_ep > cur_ep):
            return None
        return (target_ep - cur_ep) * spe / 3600.0

    next_bound = next(((nm, a) for nm, a, _ in bounds if a > (cur_ep or 0)), None)
    eta_next = _eta_h(next_bound[1]) if next_bound else None
    eta_stage8 = _eta_h(stage8_start)
    eta_done = _eta_h(total_epochs)
    spe_txt = f"{spe:.1f} s/ep" if spe else "n/a"
    next_txt = (f"{next_bound[0].split('_')[0]} {_fmt_dur(eta_next)}" if next_bound and eta_next else "—")
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<meta http-equiv="refresh" content="20">
<title>Decisive run — {run_dir.name}</title>
<style>
 body{{background:#0d1117;color:#c9d1d9;font-family:-apple-system,Segoe UI,Helvetica,Arial,sans-serif;margin:0;padding:18px}}
 .hdr{{display:flex;flex-wrap:wrap;gap:14px;align-items:baseline;margin-bottom:10px}}
 .k{{color:#8b949e;font-size:12px}} .v{{font-size:20px;font-weight:600}}
 .card{{background:#161b22;border:1px solid #30363d;border-radius:10px;padding:10px 14px}}
 .warn{{color:#ffd24a;font-size:12px;margin:6px 0 12px}}
 img{{width:100%;max-width:1300px;border-radius:10px;border:1px solid #30363d}}
 .bar{{height:8px;background:#21262d;border-radius:4px;overflow:hidden;width:280px;display:inline-block;vertical-align:middle}}
 .fill{{height:100%;background:linear-gradient(90deg,#7fd1ff,#7CFC00)}}
</style></head><body>
<h2 style="margin:0 0 4px">Decisive PR95-curriculum run &nbsp;<span class="k">{run_dir.name}</span></h2>
<div class="hdr">
 <div class="card"><div class="k">stage</div><div class="v">{cur_stage}</div></div>
 <div class="card"><div class="k">epoch</div><div class="v">{cur_ep}/{total_epochs} &nbsp;<span class="bar"><span class="fill" style="width:{pct:.1f}%"></span></span> {pct:.1f}%</div></div>
 <div class="card"><div class="k">d_seg (binding)</div><div class="v">{_fmt(cur_dseg)}</div></div>
 <div class="card"><div class="k">d_pose</div><div class="v">{_fmt(cur_dpose)}</div></div>
 <div class="card"><div class="k">S (advisory)</div><div class="v">{_fmt(cur_S,5)}</div></div>
 <div class="card"><div class="k">best S</div><div class="v">{_fmt(best,5)} <span class="k">@ep {best_ep}</span></div></div>
 <div class="card"><div class="k">rate (measured)</div><div class="v">{spe_txt}</div></div>
 <div class="card"><div class="k">ETA→next stage</div><div class="v">{next_txt}</div></div>
 <div class="card"><div class="k">ETA→stage8 Muon</div><div class="v">{_fmt_dur(eta_stage8)}</div></div>
 <div class="card"><div class="k">ETA→done</div><div class="v">{_fmt_dur(eta_done)}</div></div>
</div>
<div class="warn">⚠ All values [contest-CPU advisory] (macOS) — NON-PROMOTABLE. Authoritative score only from
 upstream/evaluate.py on the byte-closed archive. Frontier pointer (separate, byte-closed): {_FRONTIER_S}.
 To beat it: d_seg &lt; {_fmt(dseg_beat,6) if dseg_beat else 'n/a'} · sub-0.15: d_seg &lt; {_fmt(dseg_sub015,6) if dseg_sub015 else 'n/a'}.
 ETAs use the MEASURED current-stage rate ({spe_txt}), not a constant — stages 5-8 run ~2-4× slower than 1-4, so the
 whole-trace mean would lie. stage-8 (Muon) adds Newton-Schulz per step, so ETA→done is a LOWER bound until stage 8 starts.
 Updated {summ.get('updated_at_utc','?')} · dashboard {time.strftime('%H:%M:%SZ', time.gmtime())} · auto-refresh 20s.</div>
<img src="data:image/png;base64,{png_b64}">
</body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(html)
    tmp.replace(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--schedule-csv", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--refresh-seconds", type=float, default=20.0)
    args = ap.parse_args(argv)

    if not args.watch:
        render(args.run_dir, args.schedule_csv, args.out)
        print(f"[dashboard] rendered {args.run_dir.name} -> {args.out}")
        return 0
    while True:
        try:
            render(args.run_dir, args.schedule_csv, args.out)
            print(f"[dashboard] rendered {args.run_dir.name} at {time.strftime('%H:%M:%SZ', time.gmtime())}", flush=True)
        except Exception as exc:  # a watch loop must not die on a transient read
            print(f"[dashboard] WARN {exc!r}", flush=True)
        time.sleep(args.refresh_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
