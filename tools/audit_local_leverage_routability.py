#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Operator-facing CLI for local-leverage routability audit.

Per operator directive 2026-05-21 *"Let's make sure we are leveraging local
cpu and mps and metal and mlx as much as possible"*: this CLI audits every
substrate operator-authorize recipe + classifies into local-routability
buckets (LOCAL_MLX_TRAINABLE / LOCAL_MPS_TRAINABLE / LOCAL_CPU_PROXY /
PAID_ONLY) per M5 Max 128GB unified memory + Metal GPU + MLX framework.

Usage:

    .venv/bin/python tools/audit_local_leverage_routability.py
    .venv/bin/python tools/audit_local_leverage_routability.py --json
    .venv/bin/python tools/audit_local_leverage_routability.py \\
        --report-out .omx/state/local_leverage_routability_audit_<utc>.json

Exit codes:
    0 = audit complete (clean or with verdicts; output is informational)
    2 = CLI argument error

Per CLAUDE.md non-negotiables PRESERVED:
- Output is ADVISORY only per Catalog #1/#192/#317; local routings are
  NON-PROMOTABLE without paired Linux x86_64 + NVIDIA.
- Every emitted manifest row carries the canonical non-promotable triple
  (evidence_grade + promotable=False + score_claim=False) per Catalog
  #287/#323.
- Routing decisions remain with the operator; this audit informs but
  does not auto-route per CLAUDE.md "Executing actions with care".

Sister of:
- :mod:`tac.optimization.macos_cpu_advisory_signal` (macOS-CPU advisory)
- :mod:`tac.optimization.mps_research_signal` (MPS curve discovery)
- :mod:`tac.local_acceleration.mlx_integration` (MLX framework scaffold)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Ensure src/ is on path for canonical helpers.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from tac.local_acceleration.mlx_integration import mlx_smoke_test
from tac.local_acceleration.routability_audit import (
    audit_all_substrate_recipes,
    verdict_summary_text,
    write_audit_manifest,
)


def _utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Local-leverage routability audit (M5 Max + MLX + Metal + MPS)",
    )
    parser.add_argument(
        "--repo-root",
        default=str(_REPO_ROOT),
        help="repo root (default: %(default)s)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit JSON to stdout instead of human-readable summary",
    )
    parser.add_argument(
        "--report-out",
        default=None,
        help=(
            "write canonical audit manifest JSON to this path "
            "(default: .omx/state/local_leverage_routability_audit_<utc>.json)"
        ),
    )
    parser.add_argument(
        "--mlx-smoke",
        action="store_true",
        help="run MLX availability + Metal device smoke test only",
    )
    args = parser.parse_args(argv)

    if args.mlx_smoke:
        smoke = mlx_smoke_test()
        print(json.dumps(smoke, indent=2, sort_keys=True))
        return 0

    verdicts = audit_all_substrate_recipes(args.repo_root)

    if args.report_out is not None:
        report_path = Path(args.report_out)
    else:
        report_path = (
            Path(args.repo_root)
            / ".omx"
            / "state"
            / f"local_leverage_routability_audit_{_utc_stamp()}.json"
        )

    write_audit_manifest(verdicts, report_path)

    if args.json:
        payload = {
            "schema_version": "local_leverage_routability_audit.v1",
            "report_path": str(report_path),
            "verdicts": [v.as_dict() for v in verdicts],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(verdict_summary_text(verdicts))
        print(f"\nCanonical manifest written to: {report_path}")
        print(
            "Per CLAUDE.md: this audit is ADVISORY only. Local routings "
            "are NON-PROMOTABLE per Catalog #1/#192/#317."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
