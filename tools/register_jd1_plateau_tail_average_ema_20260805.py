#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Register the dy2 JD1 plateau-tail EMA live-weight law.

Idempotent: skips ``jd1_plateau_tail_average_ema_v1`` if the append-only
canonical-equation registry already contains it.  Apparatus-only; no scorer,
archive, or frontier claim.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for candidate in (REPO, REPO / "src"):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from tac.canonical_equations.jd1_plateau_tail_average_ema_20260805 import (  # noqa: E402
    EQUATION_ID,
    build_equation,
)
from tac.canonical_equations.registry import (  # noqa: E402
    load_registry_events_lenient,
    register_canonical_equation,
)

_SUBAGENT_ID = "ddm_dy1r"
_NOTES = (
    "dy1r closes dy2 FORMALIZATION_PENDING: tail-average law registered under "
    "T3_LIVE_ADAPTED scope-law surface. Apparatus-only; pointer unmoved."
)


def _registered_ids() -> set[str]:
    return {str(ev.get("equation_id", "")) for ev in load_registry_events_lenient()}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    eq = build_equation()
    if eq.equation_id != EQUATION_ID:
        raise SystemExit("equation id mismatch")
    if eq.equation_id in _registered_ids():
        result = {"equation_id": eq.equation_id, "skipped": "already_registered"}
    elif args.dry_run:
        result = {"equation_id": eq.equation_id, "dry_run": True}
    else:
        register_canonical_equation(
            eq,
            agent="codex",
            subagent_id=_SUBAGENT_ID,
            notes=_NOTES,
        )
        result = {"equation_id": eq.equation_id, "registered": True}
    print(json.dumps({"stage": "register_jd1_plateau_tail_average_ema_20260805",
                      "result": result}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
