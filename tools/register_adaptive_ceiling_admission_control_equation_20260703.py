#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Register the SYSTEM-aware adaptive-ceiling + admission-control canonical equation into the JSONL
registry (``.omx/state/canonical_equations_registry.jsonl``) so the EQUATIONS leg of the triality
matches the DAG + DSL legs for the 2026-07-02 machine-protection landing.

Equation: ``adaptive_ceiling_admission_control_v1`` — the SUM-over-RAM crash law:
``ADMIT iff system_used + sum_active max(0, peak-current) + new_peak <= (T - max(8, 0.08T))``.
Producer: tools/system_memory_governor.py. Consumers: launch_witness_run / spawn_durable_daemon /
witness_memory_preflight / memory_blackbox.

CONTAINMENT: pure decision + JSONL append; NO GPU, NO launch, NO trainer edits. Pointer 0.19110
UNMOVED — apparatus / machine-protection. Idempotent: re-running appends another 'registered' event;
``query_equations`` returns the latest-payload-per-equation_id.

Usage:
    .venv/bin/python tools/register_adaptive_ceiling_admission_control_equation_20260703.py [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tac.canonical_equations.adaptive_ceiling_admission_control_20260703 import (  # noqa: E402
    build_adaptive_ceiling_admission_control_v1,
)
from tac.canonical_equations.registry import register_canonical_equation  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="build + validate but do NOT append to the registry")
    args = ap.parse_args(argv)

    eq = build_adaptive_ceiling_admission_control_v1()
    print(f"built {eq.equation_id}: {eq.one_line_summary}")
    print(f"  producers={eq.canonical_producers}  consumers={eq.canonical_consumers}")
    if args.dry_run:
        print("DRY-RUN: not registered.")
        return 0
    register_canonical_equation(
        eq,
        agent="memory-blackbox-and-governor-landing-20260703",
        notes="FEED-mem system-aware adaptive-ceiling + admission-control (SUM-over-RAM crash law); "
              "score-neutral machine-protection; ships ADVISORY pending independent review.",
    )
    print(f"registered {eq.equation_id} into .omx/state/canonical_equations_registry.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
