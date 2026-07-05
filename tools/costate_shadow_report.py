#!/usr/bin/env python3
"""costate_shadow_report — thin CLI over the θ* costate SHADOW controller (task #303 Phase A).

Reads a witness run directory READ-ONLY (run.log + launch.sh), estimates the estimable
costates with uncertainty, classifies the trajectory through the canonical monitor,
and prints ranked recommendations. ``--write`` appends the row to
``<run_dir>/costate_shadow.jsonl`` (the only write; an advisory sidecar).

SHADOW MODE ONLY: ``actuation`` is always ``"NONE"``. This tool (and the package under
it) has NO capability to launch, signal, or mutate a run — CONTAINMENT is structural.

Usage:
  .venv/bin/python tools/costate_shadow_report.py --run-dir experiments/results/levelset_...
  .venv/bin/python tools/costate_shadow_report.py --run-dir <dir> --as-of-epoch 325 --json
  .venv/bin/python tools/costate_shadow_report.py --run-dir <dir> --write
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tac.witness_control import (  # noqa: E402
    build_shadow_report,
    load_run_inputs,
    write_shadow_row,
)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="witness run dir (reads run.log + launch.sh)")
    ap.add_argument("--as-of-epoch", type=int, default=None,
                    help="backtest: truncate all telemetry to epoch <= N")
    ap.add_argument("--horizon-epochs", type=int, default=25,
                    help="epochs over which per-epoch costates project a ΔS band (default 25)")
    ap.add_argument("--json", action="store_true", help="emit the full machine-readable row")
    ap.add_argument("--write", action="store_true",
                    help="append the row to <run_dir>/costate_shadow.jsonl")
    args = ap.parse_args(argv)

    inputs = load_run_inputs(args.run_dir, as_of_epoch=args.as_of_epoch)
    report = build_shadow_report(inputs, horizon_epochs=args.horizon_epochs)
    row = report.to_row()

    if args.write:
        out = write_shadow_row(args.run_dir, report)
        print(f"[costate-shadow] appended -> {out}", file=sys.stderr)
    if args.json:
        print(json.dumps(row, indent=2, sort_keys=True))
        return 0

    st = report.state
    print(f"run: {report.run_dir}"
          + (f"  (as-of ep{report.as_of_epoch})" if report.as_of_epoch is not None else ""))
    print(f"  state: ep{st.get('epoch')} stage={st.get('stage')} "
          f"d_seg={st.get('d_seg')} d_pose={st.get('d_pose')} "
          f"bytes={st.get('blob_bytes')} implied_S={st.get('implied_S')} "
          f"(best d_seg {st.get('best_d_seg')}@ep{st.get('best_epoch')})")
    if report.classification:
        c = report.classification
        print(f"  classification: {c['classification'].upper()} "
              f"(d_seg slope {c['d_seg_slope_per_ep']:+.3e}/ep, "
              f"ep_loss slope {c['ep_loss_slope_per_ep']:+.3e}/ep) "
              f"[{c['source']}]")
    print("  costates:")
    for cst in report.costates:
        b = cst.band()
        band_s = f" band[{b[0]:+.3e},{b[1]:+.3e}]" if b else ""
        val_s = f"{cst.value:+.4e}" if cst.value is not None else "—"
        print(f"    {cst.status:<15} {cst.name:<38} {val_s}{band_s}  ({cst.units})")
    print("  recommendations (ranked; ΔS<0 = improvement; NEVER-REGRESS enforced):")
    if not report.recommendations:
        print("    (none identifiable)")
    for r in report.recommendations:
        band_s = (f" band[{r['predicted_dS_band'][0]:+.4f},{r['predicted_dS_band'][1]:+.4f}]"
                  if r.get("predicted_dS_band") else "")
        print(f"    {r['action']}: predicted ΔS {r['predicted_dS']:+.4f}{band_s} "
              f"over {r['horizon_epochs']}ep")
        print(f"      why: {r['rationale']}")
        for ev in r["evidence"]:
            print(f"      evidence: {ev}")
    for r in report.refused:
        print(f"    REFUSED {r['action']}: {r['refusal_reason']}")
    if report.probe_queue:
        print("  probe queue (UNIDENTIFIABLE costates — honest gaps, not guesses):")
        for p in report.probe_queue:
            print(f"    {p['costate']}: {p['why_unidentifiable']}")
    print(f"  actuation: {row['actuation']}   axis: {row['axis']}   {row['pointer']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
