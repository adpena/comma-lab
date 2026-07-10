#!/usr/bin/env python3
"""witness_telemetry_audit — binding-vs-inert + decision-provenance readback for a witness run (#404 P0).

READ-ONLY post-hoc CLI over ``run.log`` rows the levelset trainer already emits (safe against the
sealed mid-flight v7.5.2 relaunch chain — it never writes into a run dir). Thin wrapper around
``tac.witness_control.telemetry_binding`` (the unit-tested analyzers):

* (a) event decision table (start_event_fired / cap_fired_before_event / engage rows, queryable)
* (b) amber grad-clip binding rate (BINDING / INERT_NEVER_BINDS / SATURATED)
* (c) chroma_boundary term share (PENDING / INERT_ZERO / BINDING / DOMINATING)
* (d) pose-gate sensor liveness (OK / DETECTOR_STALLED / ALARMED — the silent-crash class fix)
* (e) EMA-lag verdict-vs-live divergence (the run-1 confound signature)
* (f) D27b terminal-band / solve-upon-basin trigger status (in_basin / terminal_band / d27b_ready)
* (h) TAIL per-cycle endpoint stats (SWA-soup candidate inputs)

Advisory MEANS only — every input is ``[macOS-MLX advisory] NON-PROMOTABLE``; the pointer moves only
through a byte-closed ``upstream/evaluate.py`` row. Discipline: docs/operating_manual_craft_handoff.md.

Usage:
    .venv/bin/python tools/witness_telemetry_audit.py --run-dir experiments/results/<run> [--json]
    .venv/bin/python tools/witness_telemetry_audit.py --run-dir <run> --section terminal_band --json

Exit 0 on a produced audit (verdicts inside), 2 when no telemetry rows were found.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from tac.witness_control import telemetry_binding as tb  # noqa: E402

_SECTIONS = ("events", "amber", "chroma", "pose_gate", "ema_lag", "terminal_band", "tail_endpoints")


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="run dir containing run.log (nested */run.log also read)")
    ap.add_argument("--tail-bytes", type=int, default=0,
                    help="bound the per-file read to the last N bytes (0 = whole file)")
    ap.add_argument("--section", choices=_SECTIONS, default=None,
                    help="print only one section (default: full audit)")
    ap.add_argument("--json", action="store_true", help="machine-readable JSON output")
    args = ap.parse_args(argv)

    rows = tb.load_run_rows(args.run_dir, tail_bytes=args.tail_bytes)
    if not rows:
        print(f"no telemetry rows found under {args.run_dir} (looked for run.log + */run.log)",
              file=sys.stderr)
        return 2

    audit = tb.audit_rows(rows)
    audit["run_dir"] = str(args.run_dir)
    audit["axis"] = "[macOS-MLX advisory] NON-PROMOTABLE (read-only telemetry readback)"

    if args.section:
        payload = audit.get(args.section)
        print(json.dumps(payload, indent=None if args.json else 2, default=str))
        return 0
    if args.json:
        print(json.dumps(audit, default=str))
        return 0

    print(tb.format_summary(audit))
    for key in ("amber", "chroma", "pose_gate", "ema_lag", "terminal_band"):
        print(f"\n── {key} " + "─" * max(1, 70 - len(key)))
        print(json.dumps(audit.get(key), indent=2, default=str))
    events = audit.get("events") or []
    print(f"\n── events ({len(events)} decision rows; last 20) " + "─" * 30)
    for e in events[-20:]:
        print(f"  ep{e.get('epoch')}: {e.get('event')} fired_by={e.get('fired_by')} "
              f"sensor={e.get('sensor')} lag={e.get('sensor_lag_epochs')}")
    tails = audit.get("tail_endpoints") or []
    print(f"\n── tail endpoints ({len(tails)}) " + "─" * 40)
    for t in tails:
        print(f"  {t.get('segment_end')} @ep{t.get('endpoint_epoch')}: d_seg={t.get('d_seg')} "
              f"d_pose={t.get('d_pose')} implied_S={t.get('implied_S')} "
              f"best_in_seg={t.get('best_d_seg_in_segment')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
