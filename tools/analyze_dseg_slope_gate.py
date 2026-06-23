#!/usr/bin/env python
"""Deep, measured d_seg-slope analysis for the decisive run's stage-5→8 decision gate.

PURPOSE. Provide a MEASURED, multi-angle instrument for the ~1-day stage-5 slope checkpoint so the decision
("continue to the Muon finisher" / "a measured pathology to fix" / "defer pending Muon") is grounded in data,
not a vibe. Per CLAUDE.md "Forbidden premature KILL without research exhaustion" + "Measurement-first":

  THIS TOOL HAS NO "KILL" / "WALLED" / "DEAD" VERDICT. A flat d_seg in the AdamW stages (1–7) is PREDICTED
  by the conditioning thesis (diagonal preconditioner → power-law-slow d_seg; the bulk of d_seg closing is
  reserved for the SPECTRAL Muon stage 8, O(ln 1/ε)). So flatness alone is never a conclusion. The verdicts
  are: ON_TRACK_STEEPENING / ADAMW_PHASE_FLAT_AS_EXPECTED / MEASURED_PATHOLOGY_INVESTIGATE / DEFER_PENDING_MUON.
  Any negative beyond these requires research exhaustion + grand-council consensus + reactivation criteria
  (the protocol memo), NOT this tool.

Measured dimensions: per-stage d_seg slope (linear + log), recent-window slopes, power-law fit + epochs-to-
threshold projection, d_seg monotonicity (is it rising = a real anomaly?), pose/rate context, and a
conditioning-thesis-consistency flag. Authority: [contest-CPU advisory]; NON-PROMOTABLE.

Usage: .venv/bin/python tools/analyze_dseg_slope_gate.py --run-dir experiments/results/<run> [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from tac.contest_score import break_even_d_seg

_FRONTIER_S = 0.19110
_TARGET_S = 0.15
# Muon (stage 8) is the spectral d_seg finisher; AdamW stages 1-7 are expected power-law-slow on d_seg.
_ADAMW_STAGE_INDICES = set(range(0, 7))  # 0..6 (stages 1-7); stage_index 7 = stage8 muon


def _read_evals(run_dir: Path) -> list[dict]:
    rows = []
    p = run_dir / "torch_vehicle_trajectory.jsonl"
    for line in p.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("d_seg") is not None and d.get("global_epoch") is not None:
            rows.append(d)
    rows.sort(key=lambda r: r["global_epoch"])
    return rows


def _loglog_slope(eps: list[float], ys: list[float]) -> float | None:
    """Power-law exponent b in y ~ a*x^b (the honest 'is it descending and how fast' on a log-log)."""
    pts = [(e, y) for e, y in zip(eps, ys, strict=False) if e > 0 and y > 0]
    if len(pts) < 3:
        return None
    n = len(pts)
    lx = [math.log(e) for e, _ in pts]
    ly = [math.log(y) for _, y in pts]
    mx = sum(lx) / n
    my = sum(ly) / n
    den = sum((x - mx) ** 2 for x in lx)
    if den == 0:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(lx, ly, strict=False)) / den


def _window(rows: list[dict], last_n_epochs: int) -> list[dict]:
    if not rows:
        return []
    cut = rows[-1]["global_epoch"] - last_n_epochs
    return [r for r in rows if r["global_epoch"] >= cut]


def analyze(run_dir: Path) -> dict:
    rows = _read_evals(run_dir)
    if not rows:
        return {"error": "no exact-eval rows yet"}
    cur = rows[-1]
    cur_dseg = cur["d_seg"]
    cur_dpose = cur.get("d_pose")
    cur_ep = cur["global_epoch"]
    cur_stage_index = int(cur.get("stage_index", -1))

    # live break-even d_seg targets given the current pose+rate, via the
    # canonical tac.contest_score helper (carries the x25 on the rate term).
    cur_bytes = cur.get("archive_bytes", 0) or 0
    dseg_beat = (
        break_even_d_seg(_FRONTIER_S, cur_dpose, cur_bytes) if cur_dpose else None
    )
    dseg_sub015 = (
        break_even_d_seg(_TARGET_S, cur_dpose, cur_bytes) if cur_dpose else None
    )

    # per-stage d_seg endpoints + log-log slope.
    by_stage: dict[str, list[dict]] = {}
    for r in rows:
        by_stage.setdefault(r.get("stage_name", "?"), []).append(r)
    stage_summ = {}
    for name, rs in by_stage.items():
        eps = [r["global_epoch"] for r in rs]
        ys = [r["d_seg"] for r in rs]
        stage_summ[name] = {
            "n": len(rs),
            "d_seg_first": ys[0],
            "d_seg_last": ys[-1],
            "rel_change_pct": round(100 * (ys[-1] - ys[0]) / ys[0], 2) if ys[0] else None,
            "loglog_slope_b": (round(_loglog_slope(eps, ys), 3) if _loglog_slope(eps, ys) is not None else None),
        }

    # recent-window slopes (multi-scale).
    windows = {}
    for w in (500, 1000, 2000):
        ws = _window(rows, w)
        if len(ws) >= 3:
            eps = [r["global_epoch"] for r in ws]
            ys = [r["d_seg"] for r in ws]
            windows[f"last_{w}ep"] = {
                "n": len(ws),
                "d_seg_from": ys[0], "d_seg_to": ys[-1],
                "rel_change_pct": round(100 * (ys[-1] - ys[0]) / ys[0], 3) if ys[0] else None,
                "loglog_slope_b": (round(_loglog_slope(eps, ys), 3) if _loglog_slope(eps, ys) is not None else None),
            }

    # monotonicity over the last 1000 ep: is d_seg RISING (a real anomaly worth investigating)?
    w1k = _window(rows, 1000)
    rising = len(w1k) >= 3 and w1k[-1]["d_seg"] > w1k[0]["d_seg"] * 1.02

    # power-law projection: at the recent trend, when (if) does d_seg reach the beat threshold?
    proj = {}
    ws = _window(rows, 4000) or rows
    b = _loglog_slope([r["global_epoch"] for r in ws], [r["d_seg"] for r in ws])
    if b is not None and b < 0 and dseg_beat:
        # y=a*x^b ; solve x for y=dseg_beat from the last point.
        x0, y0 = cur_ep, cur_dseg
        a = y0 / (x0 ** b)
        try:
            x_beat = (dseg_beat / a) ** (1.0 / b)
            proj["epochs_to_beat_threshold_at_recent_trend"] = int(max(0, x_beat - cur_ep))
        except (ValueError, ZeroDivisionError):
            proj["epochs_to_beat_threshold_at_recent_trend"] = None
    proj["recent_loglog_slope_b"] = round(b, 3) if b is not None else None

    # conditioning-thesis consistency: in the AdamW stages, flat/slow d_seg is EXPECTED (Muon-reserved).
    in_adamw_phase = cur_stage_index in _ADAMW_STAGE_INDICES
    muon_reached = cur_stage_index >= 7

    # VERDICT (NO KILL — see module docstring).
    if rising:
        verdict = "MEASURED_PATHOLOGY_INVESTIGATE"
        rationale = ("d_seg is RISING over the last ~1000 ep (>2%). This is a real anomaly (not expected "
                     "flatness) — investigate: EMA-shadow lag, loss-form gradient death, C1a/QAT interaction, "
                     "or eval noise. NOT a kill; a fix target.")
    elif muon_reached:
        verdict = "ON_TRACK_STEEPENING" if (b is not None and b < -0.05) else "MUON_STAGE_MEASURE_DIRECTLY"
        rationale = "In the Muon (stage 8) finisher — read d_seg directly; this is the decisive spectral phase."
    elif in_adamw_phase:
        slow = (windows.get("last_2000ep", {}).get("rel_change_pct") or 0) > -3
        verdict = "ADAMW_PHASE_FLAT_AS_EXPECTED" if slow else "ON_TRACK_STEEPENING"
        rationale = ("Still in the AdamW stages (1-7). Power-law-slow / flat d_seg here is PREDICTED by the "
                     "conditioning thesis (diagonal preconditioner; the bulk of d_seg closing is reserved for "
                     "the spectral Muon stage 8). Flatness is NOT a wall — continue to the finisher. "
                     "Re-evaluate WITH the Muon stage-8 read, not before.")
    else:
        verdict = "DEFER_PENDING_MUON"
        rationale = "Indeterminate phase; defer the verdict to the Muon stage-8 read."

    return {
        "authority": "[contest-CPU advisory] NON-PROMOTABLE",
        "no_kill_discipline": "This tool has NO kill/walled/dead verdict (CLAUDE.md no-premature-kill).",
        "current": {
            "global_epoch": cur_ep, "stage": cur.get("stage_name"), "stage_index": cur_stage_index,
            "d_seg": cur_dseg, "d_pose": cur_dpose, "score_S": cur.get("score"),
        },
        "thresholds": {"beat_0.19110_needs_d_seg_below": dseg_beat, "sub_0.15_needs_d_seg_below": dseg_sub015},
        "per_stage_d_seg": stage_summ,
        "recent_windows": windows,
        "projection": proj,
        "d_seg_rising_last_1000ep": rising,
        "verdict": verdict,
        "rationale": rationale,
        "reactivation_note": ("Any verdict beyond CONTINUE/INVESTIGATE/DEFER (e.g. a kill) requires research "
                              "exhaustion + grand-council consensus + reactivation criteria per the gate protocol."),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args(argv)
    res = analyze(args.run_dir)
    print(json.dumps(res, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
