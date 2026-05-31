#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Operator-facing CLI for multi-granularity contest-score sensitivity.

Thin delegate to ``tac.multi_granularity_sensitivity`` (canonical implementation
lives in the library per CLAUDE.md "tac stays clean"). Produces:

  * REAL-MEASURED byte × (seg/pose/rate) sensitivity for an archive whose
    master-gradient anchor is on disk ($0, no GPU); and
  * the DESIGNED-PENDING-MEASUREMENT recipes for the input-domain
    (frame/pair/region/boundary) sensitivities that need a scorer forward pass.

Every emitted score-relevant number is NON-PROMOTABLE ``[predicted]`` /
``[research-signal]`` per Catalog #341 / #192 / #127 / #323.

Usage:
    .venv/bin/python tools/multi_granularity_sensitivity_report.py \\
        --archive-sha256 6bae0201fb082457... --axis '[contest-CUDA]' --json
    .venv/bin/python tools/multi_granularity_sensitivity_report.py --list-pending
"""

from __future__ import annotations

import argparse
import json
import sys

from tac.multi_granularity_sensitivity import (
    GRANULARITIES,
    SCORE_AXES,
    MultiGranularitySensitivityError,
    byte_axis_sensitivity_from_master_gradient,
    design_input_domain_sensitivity_measurement,
)


def _build_pending_catalog() -> list[dict]:
    rows: list[dict] = []
    # boundary/region are seg-only (d_seg is the argmax-flip-rate, pixel-spatial)
    for axis in ("seg",):
        for gran in ("boundary", "region"):
            rows.append(design_input_domain_sensitivity_measurement(gran, axis).as_dict())
    # frame/pair are native contest granularities for all axes
    for gran in ("frame", "pair"):
        for axis in SCORE_AXES:
            rows.append(design_input_domain_sensitivity_measurement(gran, axis).as_dict())
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--archive-sha256",
        help="archive whose master-gradient anchor to read (REAL byte-axis sensitivity)",
    )
    ap.add_argument(
        "--axis",
        default=None,
        help="contest axis filter, e.g. '[contest-CUDA]' / '[macOS-CPU advisory]'",
    )
    ap.add_argument(
        "--list-pending",
        action="store_true",
        help="list the designed-pending-measurement input-domain recipes (no GPU)",
    )
    ap.add_argument("--json", action="store_true", help="machine-readable JSON output")
    args = ap.parse_args(argv)

    if not args.archive_sha256 and not args.list_pending:
        ap.error("supply --archive-sha256 (real byte-axis) and/or --list-pending")

    payload: dict = {
        "schema": "multi_granularity_sensitivity_report_v1",
        "score_axes": list(SCORE_AXES),
        "granularities": list(GRANULARITIES),
        "non_promotable": True,
        "evidence_axis": "[predicted] / [research-signal] — NON-PROMOTABLE per Catalog #341/#192/#127/#323",
    }

    if args.archive_sha256:
        try:
            rep = byte_axis_sensitivity_from_master_gradient(
                args.archive_sha256, axis=args.axis
            )
        except MultiGranularitySensitivityError as exc:
            print(f"[multi-granularity-sensitivity] ERROR: {exc}", file=sys.stderr)
            return 1
        payload["byte_axis_sensitivity"] = rep.as_dict()

    if args.list_pending:
        payload["pending_input_domain_measurements"] = _build_pending_catalog()

    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    # human-readable summary
    print("Multi-granularity contest-score sensitivity")
    print("  axes:", ", ".join(SCORE_AXES), " granularities:", ", ".join(GRANULARITIES))
    print("  evidence: NON-PROMOTABLE [predicted]/[research-signal] (Catalog #341/#192/#127/#323)")
    if "byte_axis_sensitivity" in payload:
        b = payload["byte_axis_sensitivity"]
        print(f"\n  REAL byte-axis sensitivity (archive {args.archive_sha256[:12]}, "
              f"axis {b['measurement_axis']}, n_bytes {b['n_bytes']}):")
        for a in b["per_axis"]:
            print(f"    {a['axis']:5s}  share={a['share']:.4f}  gini={a['gini']:.4f}  "
                  f"top10%mass={a['top_decile_mass']:.4f}  dom_bytes={a['dominant_byte_count']}")
    if "pending_input_domain_measurements" in payload:
        print(f"\n  DESIGNED-PENDING input-domain measurements "
              f"({len(payload['pending_input_domain_measurements'])} recipes, need scorer forward):")
        for m in payload["pending_input_domain_measurements"]:
            print(f"    {m['granularity']:8s}/{m['score_axis']:4s}  "
                  f"steps={len(m['measurement_recipe'])}  research_only={m['research_only']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
