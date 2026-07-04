"""Witness control monitor — the self-converging safety net (scaling-law facet-5, task #289).

Reads a level-set witness run's verdict log (the JSONL ``{"stage":"verdict", ...}`` lines the
trainer emits) and classifies the CURRENT within-stage trajectory into a Lyapunov / tau-creep
control decision. It EMITS a decision + config-diffs ONLY — it NEVER launches, stops, or mutates
any run (CONTAINMENT + the P0 system-memory governor own all actuation). The operator (or a
governed launcher) consumes its JSON.

The two certificates (both MEASURED from the log; no proxy):
  * tau-CREEP detector (the #205 erosion signature): within a stage, d_seg SLOPE > +eps WHILE
    ep_loss SLOPE < 0 == the smooth-surrogate <-> hard-verdict DECOUPLING (a minority class is
    being ERODED by the mean-curvature flow while the training loss still falls). This is exactly
    what #205's tau_softplus stage does (d_seg 0.00475@ep300 -> 0.00667@ep450 while ep_loss
    148->130). Root cause: a sub-critical / zero-mass class (the LANE NUCLEATION FAILURE); fix =
    paint-then-SDF seed (#291) + raised eikonal + per-class area constraint.
  * Lyapunov descent certificate (the convergence test): V := d_seg (the distortion we minimize);
    dV/dt := the within-stage d_seg slope. Converging iff dV/dt < 0; PLATEAU iff |dV/dt| ~ 0 for
    the window (candidate early-stop / stage-advance). (The OT-dual-gap V_OT is the PROVEN-tier
    sibling certificate; this tool implements the MEASURED-tier descent-rate certificate from the
    log alone, which needs no extra compute.)

Usage:
  .venv/bin/python tools/witness_control_monitor.py --run-log <path/to/run.log>
  .venv/bin/python tools/witness_control_monitor.py --run-dir <experiments/results/levelset_*>
  .venv/bin/python tools/witness_control_monitor.py --run-log <...> --json   # machine-readable
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Classification sentinels (a falling-rule list, most-severe first).
DIVERGING_ERASING = "diverging_erasing"   # tau-creep: d_seg UP while loss DOWN (a class eroding)
VOLATILE = "volatile"                       # high within-window variance -> no clean slope
PLATEAU = "plateau"                         # |dV/dt| ~ 0 -> converged within stage
CONVERGING = "converging"                   # dV/dt < 0 -> healthy descent


@dataclass(frozen=True)
class ControlVerdict:
    classification: str
    stage: str
    n_window: int
    epoch_latest: int
    d_seg_latest: float
    d_seg_slope_per_ep: float       # dV/dt (V = d_seg); <0 healthy, >0 erosion
    ep_loss_slope_per_ep: float     # <0 = loss still falling
    d_seg_cv: float                 # within-window coefficient of variation (volatility)
    recommendation: str
    config_diffs: tuple[str, ...]


def _lstsq_slope(xs: list[float], ys: list[float]) -> float:
    """Least-squares slope dy/dx (0.0 if <2 points or zero x-spread). Pure numpy-free."""
    n = len(xs)
    if n < 2:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0:
        return 0.0
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return sxy / sxx


def _read_verdicts(run_log: Path) -> list[dict]:
    """Parse the ``"stage":"verdict"`` JSONL lines from a run.log (tolerant of other lines)."""
    out: list[dict] = []
    for line in run_log.read_text(errors="replace").splitlines():
        line = line.strip()
        if '"stage": "verdict"' not in line and '"stage":"verdict"' not in line:
            continue
        try:
            row = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if row.get("stage") == "verdict" and "d_seg" in row and "epoch" in row:
            out.append(row)
    return out


def classify_trajectory(
    verdicts: list[dict], *, window: int = 5, creep_eps: float = 1e-6,
    plateau_eps: float = 5e-7, volatile_cv: float = 0.5,
) -> ControlVerdict:
    """Classify the CURRENT within-stage trajectory (the last ``window`` same-stage verdicts).

    creep_eps / plateau_eps are per-epoch d_seg slopes; volatile_cv is the CV above which the
    window is 'volatile' (no clean slope). All thresholds MEASURED-tunable, defaults from the
    #205 trace (its tau slope ~ +9e-6/ep, well above creep_eps)."""

    if not verdicts:
        raise ValueError("no verdicts to classify")
    latest_stage = str(verdicts[-1].get("seg_form", ""))
    same = [v for v in verdicts if str(v.get("seg_form", "")) == latest_stage]
    win = same[-int(window):] if window > 0 else same
    eps = [float(v["epoch"]) for v in win]
    dsegs = [float(v["d_seg"]) for v in win]
    losses = [float(v.get("ep_loss", 0.0)) for v in win]

    d_slope = _lstsq_slope(eps, dsegs)
    l_slope = _lstsq_slope(eps, losses)
    mean_ds = sum(dsegs) / len(dsegs)
    if len(dsegs) >= 2 and mean_ds > 0:
        var = sum((d - mean_ds) ** 2 for d in dsegs) / len(dsegs)
        cv = (var ** 0.5) / mean_ds
    else:
        cv = 0.0

    # Falling-rule classification (most-severe first).
    if d_slope > creep_eps and l_slope < 0.0:
        cls = DIVERGING_ERASING
        rec = ("tau-CREEP: d_seg is RISING while ep_loss FALLS (surrogate<->verdict DECOUPLING) — "
               "a minority class is being ERODED by the flow (the #205 nucleation-failure signature). "
               "For a live run this is low-EV to continue; for the fresh run apply the seed fix.")
        diffs = ("--lane-prior-phi1-mode paint (nucleate the lane, #291)",
                 "--eikonal-weight 0.05 (hold the thin interface sharp)",
                 "per-class area constraint (auction-MBO, pin mass != 0)",
                 "verify part_frac[rare-class] > 0 at ep0 (the acceptance gate)")
    elif cv > volatile_cv:
        cls = VOLATILE
        rec = ("high within-window variance — no clean slope; widen the window or reduce LR / "
               "check for stage-transition collision (Ch.6 easing).")
        diffs = ("--stage-transition-rewarmup-epochs 20 --stage-transition-rewarmup-shape cosine",)
    elif abs(d_slope) <= plateau_eps:
        cls = PLATEAU
        rec = ("converged within stage (|dV/dt| ~ 0) — candidate EARLY-STOP or STAGE-ADVANCE "
               "(the Lyapunov descent certificate has flattened).")
        diffs = ("advance to the next curriculum stage OR early-stop this stage",)
    else:  # d_slope < -plateau_eps
        cls = CONVERGING
        rec = "healthy descent (dV/dt < 0) — continue; no action."
        diffs = ()

    return ControlVerdict(
        classification=cls, stage=latest_stage, n_window=len(win),
        epoch_latest=int(win[-1]["epoch"]), d_seg_latest=float(win[-1]["d_seg"]),
        d_seg_slope_per_ep=d_slope, ep_loss_slope_per_ep=l_slope, d_seg_cv=cv,
        recommendation=rec, config_diffs=diffs,
    )


def _resolve_run_log(args: argparse.Namespace) -> Path:
    if args.run_log:
        return Path(args.run_log)
    if args.run_dir:
        p = Path(args.run_dir) / "run.log"
        if p.exists():
            return p
    # newest levelset run.log
    cands = sorted(glob.glob(str(REPO / "experiments/results/levelset_*/run.log")))
    if not cands:
        raise SystemExit("no run.log found; pass --run-log or --run-dir")
    return Path(cands[-1])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-log", type=str, default="", help="path to a witness run.log")
    ap.add_argument("--run-dir", type=str, default="", help="a run dir (uses <dir>/run.log)")
    ap.add_argument("--window", type=int, default=5, help="within-stage verdicts to fit (default 5)")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args(argv)

    run_log = _resolve_run_log(args)
    verdicts = _read_verdicts(run_log)
    if not verdicts:
        raise SystemExit(f"no verdict rows in {run_log}")
    v = classify_trajectory(verdicts, window=args.window)

    if args.json:
        print(json.dumps({"run_log": str(run_log), "n_verdicts": len(verdicts), **asdict(v),
                          "note": "CONTAINMENT: decision-only; this tool NEVER launches or stops a run."},
                         indent=2))
    else:
        print(f"run: {run_log}")
        print(f"  stage={v.stage}  window={v.n_window}  ep={v.epoch_latest}  d_seg={v.d_seg_latest:.6f}")
        print(f"  d_seg slope = {v.d_seg_slope_per_ep:+.3e}/ep   ep_loss slope = {v.ep_loss_slope_per_ep:+.3e}/ep   CV={v.d_seg_cv:.3f}")
        print(f"  ==> {v.classification.upper()}")
        print(f"  {v.recommendation}")
        for d in v.config_diffs:
            print(f"    - {d}")
        print("  [CONTAINMENT: decision-only — this tool never launches or stops a run.]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
