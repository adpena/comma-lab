#!/usr/bin/env python3
"""owed-14 GOVERNED TELEMETRY REPLAY (SYNTHESIS_v3_v752 §C item 14, v3 AMENDMENT-2).

Replay the v7.5.2 observer stack — the pose-finish σ_min rolling-slope plateau detector, the
verdict-trend alarm, and the #247 costate-shadow observer — against a STOPPED run's REAL telemetry,
and PROVE the observers (a) parse the real rows and (b) fire/hold where the run's history says they
should. This is the $0 NEGATIVE-control leg of the pose-gate canary (SYNTHESIS §A.4 v3 AMENDMENT-2:
"the detector must NOT fire on the stopped run's rising σ_min rows") plus a parse+fire smoke of the
verdict-trend + costate observers on the same real telemetry.

READ-ONLY on the target run dir (never writes into it). Advisory / `[macOS-MLX advisory]` — MEANS, not
a score. Default target = the #205-lineage stopped run
``experiments/results/levelset_n600_witness_20260709T105312Z``.

Usage:
    .venv/bin/python tools/witness_observer_replay.py --run-dir <dir> [--json]

Exit 0 iff every leg's expectation holds (pose-gate does NOT fire on the rising σ_min; verdict-trend +
costate observers parse the real rows); 1 otherwise.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from tac import witness_run_artifacts as _wra  # noqa: E402

_DEFAULT_RUN = "experiments/results/levelset_n600_witness_20260709T105312Z"


# ----------------------------------------------------------------------------- parse (PURE) --
def load_jsonl_rows(path: Path) -> list[dict]:
    """Parse a JSONL telemetry file into the list of dict rows (skips non-JSON / blank lines).
    PURE; a missing file returns []."""
    out: list[dict] = []
    try:
        text = Path(path).read_text()
    except OSError:
        return out
    for ln in text.splitlines():
        ln = ln.strip()
        if not ln.startswith("{"):
            continue
        try:
            d = json.loads(ln)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(d, dict):
            out.append(d)
    return out


def sigma_min_series(rows: list[dict]) -> tuple[list[float], list[float]]:
    """(epochs, median_sigma_min) from the trainer's ``jacobian_basin`` telemetry rows, in log order.
    PURE."""
    eps: list[float] = []
    smins: list[float] = []
    for r in rows:
        if r.get("stage") != "jacobian_basin":
            continue
        e = r.get("epoch")
        s = r.get("median_sigma_min")
        if isinstance(e, (int, float)) and isinstance(s, (int, float)):
            eps.append(float(e))
            smins.append(float(s))
    return eps, smins


def verdict_rows(rows: list[dict]) -> list[dict]:
    """The trainer's advisory ``verdict`` rows (carry d_seg/d_pose/epoch). PURE."""
    return [r for r in rows if r.get("stage") == "verdict"
            and isinstance(r.get("d_seg"), (int, float)) and isinstance(r.get("epoch"), (int, float))]


# ----------------------------------------------------------------------------- replay legs --
def replay_pose_gate(eps: list[float], smins: list[float]) -> dict:
    """NEGATIVE CONTROL (SYNTHESIS §A.4 v3 AMENDMENT-2): the de-noised σ_min rolling-slope plateau
    detector must NOT fire on the stopped run's rising/oscillating σ_min series. PASS = not fired."""
    from tac.witness_control.sigma_min_plateau import (
        PLATEAU_FIRED,
        SigmaMinPlateauConfig,
        evaluate_plateau,
    )
    if len(smins) < 2:
        return {"leg": "pose_gate_negative_control", "parsed_points": len(smins),
                "classification": "INSUFFICIENT_DATA", "fired": False,
                "expected": "MUST NOT FIRE (rising σ_min)", "pass": False,
                "detail": "too few σ_min points to evaluate the detector"}
    cfg = SigmaMinPlateauConfig()
    v = evaluate_plateau(eps, smins, cfg)
    fired = (v.classification == PLATEAU_FIRED)
    rising = smins[-1] > smins[0]
    return {
        "leg": "pose_gate_negative_control", "parsed_points": len(smins),
        "sigma_min_first": round(smins[0], 5), "sigma_min_last": round(smins[-1], 5),
        "sigma_min_rising_overall": rising,
        "classification": v.classification, "fired": fired,
        "should_ship_banked_r1": v.should_ship_banked_r1(),
        "latest_rel_slope_per_ep": v.latest_rel_slope_per_ep,
        "expected": "MUST NOT FIRE (stopped-run σ_min is rising/oscillating — not a converged plateau)",
        # negative control passes iff the detector does NOT fire (either NOT_PLATEAUED or the degenerate
        # guard trips → ship banked; both correctly withhold the in-basin pose finish)
        "pass": (not fired),
    }


def replay_verdict_trend(vrows: list[dict]) -> dict:
    """Parse+fire the verdict-trend alarm on the real verdict rows. History: d_seg descends then mildly
    rises in the tail; the leg PASSES if the alarm parses the real rows and returns a known
    classification (reporting whether it caught the tail rise)."""
    from tac.witness_control.verdict_trend_alarm import (
        NO_ALARM,
        RISING_VERDICT,
        RISING_VERDICT_UNIDENTIFIABLE,
        TRAIN_VERDICT_DECOUPLING,
        verdict_trend_alarm,
    )
    known = {NO_ALARM, RISING_VERDICT, RISING_VERDICT_UNIDENTIFIABLE, TRAIN_VERDICT_DECOUPLING}
    a = verdict_trend_alarm(vrows)
    d_segs = [round(float(r["d_seg"]), 5) for r in vrows]
    parsed = a.classification in known
    return {
        "leg": "verdict_trend", "parsed_rows": len(vrows),
        "d_seg_trajectory": d_segs,
        "classification": a.classification, "fired": a.fired(),
        "expected": "PARSE the real verdict rows + return a known classification (d_seg mildly rises "
                    "in the tail: best→last)",
        "pass": bool(parsed and len(vrows) >= 1),
    }


def replay_costate_shadow(cs_rows: list[dict]) -> dict:
    """Prove the #247 costate-shadow observer's real output parses: >=1 row with named costates."""
    names: list[str] = []
    for r in cs_rows:
        cs = r.get("costates")
        if isinstance(cs, list) and cs:
            names = [c.get("name") for c in cs if isinstance(c, dict) and c.get("name")]
            break
    return {
        "leg": "costate_shadow", "parsed_rows": len(cs_rows),
        "costate_names": names,
        "expected": "PARSE the real costate_shadow.jsonl rows + expose the named costates",
        "pass": bool(len(cs_rows) >= 1 and names),
    }


def replay_disengaged_alarm(n_sigma_points: int, pose_gate_fired: bool) -> dict:
    """Prove the LOUD pose-DISENGAGED-at-end alarm (SYNTHESIS §A.4 Repair 4) builds a valid row for a
    run that never plateaued (the stopped run): the pose-finish never engaged → a real run would ship
    the banked R1 dxi and fire this alarm, never silently."""
    from tac.witness_control.sigma_min_plateau import disengaged_alarm_row
    row = disengaged_alarm_row(
        epoch=-1, reason="replay: stopped run σ_min never plateaued (negative control)",
        canary_passed=None, n_points=int(n_sigma_points))
    valid = (row.get("stage") == "confound_alarm"
             and row.get("alarm") == "pose_finish_disengaged_shipped_banked_r1")
    return {
        "leg": "disengaged_alarm_build", "alarm": row.get("alarm"),
        "expected": "the disengaged alarm builds a valid confound_alarm row (run shipped pose-blind → "
                    "never silent)",
        # only meaningful when the pose gate did NOT fire (the disengaged path); if it DID fire this leg
        # is N/A but still proves the row builder
        "pass": bool(valid),
        "applicable": (not pose_gate_fired),
    }


# ----------------------------------------------------------------------------- driver --
def replay_run(run_dir: Path) -> dict:
    """Replay all observer legs against a run dir's REAL telemetry (READ-ONLY). Returns a report dict."""
    run_dir = Path(run_dir)
    rows = load_jsonl_rows(run_dir / "run.log")
    cs_rows = load_jsonl_rows(run_dir / _wra.COSTATE_JSONL)
    eps, smins = sigma_min_series(rows)
    vrows = verdict_rows(rows)

    pose = replay_pose_gate(eps, smins)
    verdict = replay_verdict_trend(vrows)
    costate = replay_costate_shadow(cs_rows)
    diseng = replay_disengaged_alarm(len(smins), pose.get("fired", False))

    legs = [pose, verdict, costate, diseng]
    all_pass = all(leg["pass"] for leg in legs)
    return {
        "gate": "owed-14 governed telemetry replay",
        "run_dir": str(run_dir), "read_only": True,
        "run_log_rows": len(rows), "jacobian_basin_points": len(smins),
        "verdict_rows": len(vrows), "costate_shadow_rows": len(cs_rows),
        "legs": legs, "all_pass": all_pass,
        "axis": "[macOS-MLX advisory] NON-PROMOTABLE (MEANS, not a score; pointer 0.19110 UNMOVED)",
    }


def format_report(rep: dict) -> str:
    lines = [
        f"owed-14 OBSERVER REPLAY — {rep['run_dir']} (READ-ONLY)",
        f"  parsed: {rep['run_log_rows']} run.log rows | {rep['jacobian_basin_points']} σ_min points | "
        f"{rep['verdict_rows']} verdict rows | {rep['costate_shadow_rows']} costate rows",
        "",
    ]
    for leg in rep["legs"]:
        mark = "PASS" if leg["pass"] else "FAIL"
        extra = ""
        if leg["leg"] == "pose_gate_negative_control":
            extra = (f" [{leg['classification']}, fired={leg['fired']}, "
                     f"σ_min {leg.get('sigma_min_first')}→{leg.get('sigma_min_last')} "
                     f"rising={leg.get('sigma_min_rising_overall')}]")
        elif leg["leg"] == "verdict_trend":
            extra = f" [{leg['classification']}, fired={leg['fired']}, d_seg={leg['d_seg_trajectory']}]"
        elif leg["leg"] == "costate_shadow":
            extra = f" [{leg['parsed_rows']} rows, costates={leg['costate_names']}]"
        elif leg["leg"] == "disengaged_alarm_build":
            extra = f" [alarm={leg['alarm']}, applicable={leg.get('applicable')}]"
        lines.append(f"  [{mark}] {leg['leg']}{extra}")
        lines.append(f"         expected: {leg['expected']}")
    lines.append("")
    lines.append(f"  ALL LEGS: {'PASS' if rep['all_pass'] else 'FAIL'}")
    lines.append(f"  {rep['axis']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default=_DEFAULT_RUN,
                    help=f"the STOPPED run dir to replay (READ-ONLY; default {_DEFAULT_RUN})")
    ap.add_argument("--json", action="store_true", help="emit the machine-readable report")
    args = ap.parse_args(argv)

    run_dir = Path(args.run_dir)
    if not run_dir.is_absolute():
        run_dir = _REPO / run_dir
    if not (run_dir / "run.log").exists():
        print(f"witness_observer_replay: no run.log under {run_dir} — nothing to replay.",
              file=sys.stderr)
        return 2

    rep = replay_run(run_dir)
    if args.json:
        print(json.dumps(rep, indent=2))
    else:
        print(format_report(rep))
    return 0 if rep["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
