#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Operator-facing CLI + library API: check predecessor probe outcome.

Per CLAUDE.md "Operator gates must be wired and used" non-negotiable + Catalog
#313 (sister of #245 Modal call_id ledger 4-layer pattern). Operator-facing
helper that answers the question:

    "Has this substrate / recipe already been adjudicated by a probe
    within the last 30 days, and if so what was the verdict?"

Usage::

    # Query by recipe path:
    .venv/bin/python tools/check_predecessor_probe_outcome.py \\
        --recipe .omx/operator_authorize_recipes/substrate_atw_codec_v2_modal_a100_dispatch.yaml

    # Query by substrate name:
    .venv/bin/python tools/check_predecessor_probe_outcome.py \\
        --substrate atw_codec_v2

    # JSON output for downstream consumers:
    .venv/bin/python tools/check_predecessor_probe_outcome.py \\
        --recipe <path> --json

    # Show all blocking outcomes (operator dashboard):
    .venv/bin/python tools/check_predecessor_probe_outcome.py --list-blocking

Exit codes::

    0   no blocking predecessor outcome (dispatch OK to proceed)
    1   blocking predecessor outcome found (dispatch should be refused)
    2   query argument error (no --recipe / --substrate / --list-blocking)

Library API: ``check_predecessor_probe_outcome(recipe_path=...)`` returns the
canonical ``ProbeOutcomeView`` or ``None``. ``tools/operator_authorize.py``
imports the helper to gate paid dispatch.

Memory: feedback_probe_outcomes_canonical_ledger_landed_20260516.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.probe_outcomes_ledger import (  # noqa: E402
    PROBE_OUTCOMES_LEDGER_PATH,
    ProbeOutcomeView,
    latest_blocking_outcome_by_recipe,
    latest_blocking_outcome_by_substrate,
    query_blocking_outcomes,
)


def check_predecessor_probe_outcome(
    *,
    recipe_path: str | Path | None = None,
    substrate: str | None = None,
    ledger_path: Path | None = None,
) -> ProbeOutcomeView | None:
    """Library API: return the most-recent blocking outcome for the recipe
    OR substrate, or None.

    Resolution order:
      1. If ``recipe_path`` is provided AND the ledger has a matching outcome,
         return that view.
      2. Else if ``substrate`` is provided, return the most-recent blocking
         outcome by substrate (fallback for renamed recipes).
      3. Else return None.

    Per CLAUDE.md "Forbidden premature KILL": a blocking outcome does NOT
    mean the lane is killed — it means the apparatus has already adjudicated
    this probe within the staleness window. Operator override is via paired-
    env ``OPERATOR_AUTHORIZE_PROBE_PREDECESSOR_BYPASS_VERDICT=1`` + paired
    rationale, per Catalog #199 sister discipline.
    """
    if recipe_path is not None:
        view = latest_blocking_outcome_by_recipe(recipe_path, path=ledger_path)
        if view is not None:
            return view
    if substrate is not None:
        return latest_blocking_outcome_by_substrate(substrate, path=ledger_path)
    return None


def _format_human(view: ProbeOutcomeView) -> str:
    """Render a blocking ProbeOutcomeView as a multi-line human-readable string."""
    lines = [
        "[probe-predecessor] BLOCKING outcome found:",
        f"  probe_id            = {view.probe_id}",
        f"  substrate           = {view.substrate}",
        f"  recipe_path         = {view.recipe_path}",
        f"  probe_kind          = {view.probe_kind}",
        f"  verdict             = {view.verdict}",
        f"  metric_name         = {view.metric_name}",
        f"  metric_value        = {view.metric_value}",
        f"  threshold           = {view.threshold}",
        f"  threshold_token     = {view.threshold_token}",
        f"  evidence_path       = {view.evidence_path}",
        f"  next_action         = {view.next_action}",
        f"  blocker_status      = {view.blocker_status}",
        f"  adjudicated_at_utc  = {view.adjudicated_at_utc}",
        f"  expires_at_utc      = {view.expires_at_utc}",
    ]
    return "\n".join(lines)


def _to_dict(view: ProbeOutcomeView) -> dict[str, Any]:
    return {
        "probe_id": view.probe_id,
        "substrate": view.substrate,
        "recipe_path": view.recipe_path,
        "probe_kind": view.probe_kind,
        "verdict": view.verdict,
        "metric_name": view.metric_name,
        "metric_value": view.metric_value,
        "threshold": view.threshold,
        "threshold_token": view.threshold_token,
        "evidence_path": view.evidence_path,
        "next_action": view.next_action,
        "blocker_status": view.blocker_status,
        "adjudicated_at_utc": view.adjudicated_at_utc,
        "expires_at_utc": view.expires_at_utc,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check predecessor probe-outcome for a recipe / substrate.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--recipe",
        type=str,
        default=None,
        help="Operator-authorize recipe path to query.",
    )
    parser.add_argument(
        "--substrate",
        type=str,
        default=None,
        help="Substrate name to query (fallback when recipe_path differs).",
    )
    parser.add_argument(
        "--list-blocking",
        action="store_true",
        help="List ALL active blocking outcomes across all recipes/substrates.",
    )
    parser.add_argument(
        "--ledger-path",
        type=str,
        default=str(PROBE_OUTCOMES_LEDGER_PATH),
        help=f"Override ledger path (default: {PROBE_OUTCOMES_LEDGER_PATH}).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON output instead of human-readable text.",
    )
    args = parser.parse_args(argv)

    ledger = Path(args.ledger_path)

    if args.list_blocking:
        blockers = query_blocking_outcomes(path=ledger)
        if args.json:
            payload = {
                "blocking_outcomes": [dict(row) for row in blockers],
                "count": len(blockers),
            }
            print(json.dumps(payload, sort_keys=True, indent=2))
        else:
            if not blockers:
                print("[probe-predecessor] no active blocking outcomes.")
            else:
                print(f"[probe-predecessor] {len(blockers)} blocking outcome(s):")
                for row in blockers:
                    print(
                        f"  - {row.get('probe_id')} | {row.get('substrate')} | "
                        f"{row.get('verdict')} | {row.get('adjudicated_at_utc')}"
                    )
        return 0

    if not args.recipe and not args.substrate:
        parser.print_help(sys.stderr)
        print(
            "\n[probe-predecessor] ERROR: must specify one of "
            "--recipe / --substrate / --list-blocking",
            file=sys.stderr,
        )
        return 2

    view = check_predecessor_probe_outcome(
        recipe_path=args.recipe,
        substrate=args.substrate,
        ledger_path=ledger,
    )
    if view is None:
        if args.json:
            print(json.dumps({"blocking_outcome": None}, sort_keys=True))
        else:
            recipe_or_substrate = args.recipe or args.substrate
            print(
                f"[probe-predecessor] OK: no blocking predecessor outcome "
                f"for {recipe_or_substrate!r}"
            )
        return 0

    if args.json:
        print(json.dumps({"blocking_outcome": _to_dict(view)}, sort_keys=True))
    else:
        print(_format_human(view))
        print(
            "\n[probe-predecessor] DISPATCH SHOULD BE REFUSED. "
            "Per CLAUDE.md Catalog #313: the apparatus has already adjudicated "
            "this probe within the 30-day staleness window. Either:"
        )
        print(
            "  1. Address the blocker (sister probe with alternative reducer; "
            "fresh evidence; council adjudication)."
        )
        print(
            "  2. Operator override: set "
            "OPERATOR_AUTHORIZE_PROBE_PREDECESSOR_BYPASS_VERDICT=1 + paired "
            "OPERATOR_AUTHORIZE_PROBE_PREDECESSOR_BYPASS_RATIONALE=<text>."
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
