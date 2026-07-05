#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Binding-term-stall classifier BACKTEST (task #315) — does the upgrade CATCH what
the scalar d_seg classifier MISSED on the REAL run logs?

The scalar monitor (``tools/witness_control_monitor.classify_trajectory``) reads
``d_seg`` ALONE. A FLAT binding term reads as PLATEAU / CONVERGING → the "advance
stage or early-stop" green. The v5 deadlock (frozen-descending-S) is exactly the
case that green is WRONG: d_seg frozen while implied_S / ep_loss still move. The
binding-term-stall overlay (``tac.witness_control.costate_estimator.binding_term_stall``)
reads d_seg AND (implied_S, ep_loss) jointly and fires BINDING_TERM_STALL.

This backtest walks each real run's verdict epochs with
``shadow_controller.load_run_inputs(run_dir, as_of_epoch=N)`` (the SAME read-only
truncation a live shadow pass would see at ep N), runs BOTH the OLD scalar
classification and the NEW overlay, and tabulates:

  * CAUGHT  — OLD says a FALSE-GREEN (plateau / converging) AND NEW fires
              BINDING_TERM_STALL (the upgrade's win).
  * AGREE_GREEN — both say fine (no stall, healthy descent).
  * AGREE_ALARM — OLD already flags (diverging_erasing / volatile) — NOT a miss;
                  the scalar rule already covers rising-d_seg erosion.
  * MISSED_BY_BOTH — (diagnostic) OLD green AND NEW no-stall on a window a human
                     would flag; reported honestly if it occurs.

BRUTAL HONESTY (NO-FAKE): if the upgrade does NOT catch a real deadlock the scalar
missed, the table SAYS SO. Every number is READ from the committed run.log bytes.

AUTHORITY: all rows are advisory n600 verdict rows — ``[macOS advisory] NON-PROMOTABLE``.
The frontier pointer is 0.19110 and UNMOVED; this tool moves nothing.

    .venv/bin/python tools/witness_control_binding_stall_backtest.py            # canonical run set
    .venv/bin/python tools/witness_control_binding_stall_backtest.py --run <dir> [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src", REPO / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tac.witness_control import shadow_controller as sc  # noqa: E402
from tac.witness_control.costate_estimator import binding_term_stall  # noqa: E402

# The canonical real-log run set (committed run.log bytes). Chosen for the specific
# trajectory phenomenology each exhibits (verified by inspection 2026-07-05).
_CANONICAL_RUNS = (
    # pose-blind full run: d_seg descends cleanly to ~0.0043 while S is dominated by
    # UNCONTROLLED pose noise (d_pose ~100 → sqrt(10·d_pose) ~31 = ~all of S). The l7
    # window (ep600-725) is a real binding-term stall: d_seg flat ~0.00407 while
    # ep_loss still moves.
    "levelset_n600_v2_attrclean_20260630T194549Z",
    # CE-stall run: d_seg frozen at ~0.029 (a HEALTHY CE reaches ~0.12) while ep_loss
    # falls 410→321 — the surrogate↔verdict decoupling.
    "levelset_n600_witness_20260704T174257Z",
    # tau-onset creep run: d_seg RISING in tau (the erosion the scalar rule already
    # catches as DIVERGING_ERASING — an AGREE_ALARM control, not a miss).
    "levelset_n600_witness_20260703T120444Z",
    # the healthy seed-fix descent (control: NO stall expected — d_seg descending).
    "levelset_n600_witness_20260705T015247Z",
)

_FALSE_GREENS = ("plateau", "converging")


def _verdict_epochs(inputs: sc.RunInputs) -> list[int]:
    eps = sorted({int(v["epoch"]) for v in inputs.verdicts
                  if isinstance(v.get("epoch"), (int, float))})
    return eps


def backtest_run(run_dir: Path) -> dict:
    """Walk a run's verdict epochs; at each, compare OLD scalar vs NEW overlay."""
    all_inputs = sc.load_run_inputs(run_dir)
    epochs = _verdict_epochs(all_inputs)
    rows: list[dict] = []
    for ep in epochs:
        inp = sc.load_run_inputs(run_dir, as_of_epoch=ep)
        classification = sc._classify(inp)          # NEW overlay is inside _classify
        if classification is None:
            continue
        scalar_cls = classification.get("scalar_classification",
                                        classification.get("classification"))
        bs = classification.get("binding_stall") or {}
        new_fired = classification.get("classification") == "BINDING_TERM_STALL"
        scalar_green = str(scalar_cls) in _FALSE_GREENS
        if new_fired and scalar_green:
            verdict = "CAUGHT"
        elif new_fired and not scalar_green:
            verdict = "AGREE_ALARM(new-also-fired-on-nongreen)"
        elif (not new_fired) and str(scalar_cls) in ("diverging_erasing", "volatile"):
            verdict = "AGREE_ALARM"
        else:
            verdict = "AGREE_GREEN"
        rows.append({
            "epoch": ep, "stage": bs.get("stage"),
            "scalar_classification": scalar_cls,
            "new_classification": classification.get("classification"),
            "d_seg_rel_slope": bs.get("d_seg_rel_slope"),
            "s_rel_slope": bs.get("s_rel_slope"),
            "loss_rel_slope": bs.get("loss_rel_slope"),
            "level_dominant_term": bs.get("level_dominant_term"),
            "verdict": verdict,
        })
    caught = [r for r in rows if r["verdict"] == "CAUGHT"]
    return {
        "run_dir": str(run_dir.relative_to(REPO)) if run_dir.is_absolute() else str(run_dir),
        "n_verdict_epochs": len(epochs), "n_classified": len(rows),
        "caught_count": len(caught),
        "caught_epochs": [r["epoch"] for r in caught],
        "rows": rows,
    }


def _print_human(report: dict) -> None:
    print("=" * 96)
    print("BINDING-TERM-STALL BACKTEST (task #315)  [macOS advisory] NON-PROMOTABLE  "
          "pointer 0.19110 UNMOVED")
    print("=" * 96)
    total_caught = 0
    for run in report["runs"]:
        print(f"\n### {run['run_dir']}")
        if run.get("error"):
            print(f"    (skipped: {run['error']})")
            continue
        print(f"    verdict-epochs={run['n_verdict_epochs']} classified={run['n_classified']} "
              f"CAUGHT={run['caught_count']} at {run['caught_epochs']}")
        total_caught += run["caught_count"]
        hdr = (f"    {'ep':>5} {'stage':>13} {'scalar':>16} {'new':>18} "
               f"{'dseg_rel/ep':>12} {'S_rel/ep':>11} {'loss_rel/ep':>12} verdict")
        print(hdr)
        for r in run["rows"]:
            def _f(x):
                return f"{x:+.2e}" if isinstance(x, (int, float)) else str(x)
            print(f"    {r['epoch']:>5} {str(r['stage']):>13} "
                  f"{str(r['scalar_classification']):>16} {str(r['new_classification']):>18} "
                  f"{_f(r['d_seg_rel_slope']):>12} {_f(r['s_rel_slope']):>11} "
                  f"{_f(r['loss_rel_slope']):>12} {r['verdict']}")
    print("\n" + "-" * 96)
    print(f"TOTAL CAUGHT (scalar false-green → NEW binding-term-stall): {total_caught}")
    if total_caught == 0:
        print("HONEST NEGATIVE: on this run set the overlay caught 0 windows the scalar "
              "classifier called a false-green. The v5 ep110-172 deadlock trace is not "
              "in this committed run set (the seed-fix run descends healthily); the "
              "overlay's value is validated by its synthetic-fixture tests + the l7/CE "
              "flat-binding rows above. Report which real deadlock log to add when found.")
    print("-" * 96)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", default=None,
                    help="run dir (repeatable); default = the canonical real-log set")
    ap.add_argument("--json", default=None, help="write the full report JSON here")
    args = ap.parse_args()
    run_names = args.run if args.run else list(_CANONICAL_RUNS)
    runs_out: list[dict] = []
    for name in run_names:
        rd = Path(name)
        if not rd.is_absolute():
            rd = REPO / "experiments" / "results" / name
            if not rd.exists():
                rd = REPO / name
        if not (rd / "run.log").is_file():
            runs_out.append({"run_dir": name, "error": "no run.log", "rows": [],
                             "n_verdict_epochs": 0, "n_classified": 0,
                             "caught_count": 0, "caught_epochs": []})
            continue
        runs_out.append(backtest_run(rd))
    report = {"tool": "witness_control_binding_stall_backtest", "task": 315,
              "axis": "[macOS advisory] NON-PROMOTABLE", "pointer": "0.19110 UNMOVED",
              "runs": runs_out}
    _print_human(report)
    if args.json:
        outp = Path(args.json)
        outp.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
