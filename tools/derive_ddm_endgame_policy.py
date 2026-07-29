#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Derive the pure DDM endgame policy or its canonical arithmetic receipt.

The default mode prints the deterministic arithmetic receipt.  ``--request``
reads one JSON request and prints the resume-serializable advisory decision.
This tool performs no scoring, training, launch, pointer mutation, or file
write.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
POLICY_PATH = REPO / "src" / "tac" / "witness_control" / "ddm_endgame_policy.py"


def _load_policy_module():
    """Load the stdlib-only leaf without importing witness_control's heavy package init."""

    module_name = "_ddm_endgame_policy_leaf"
    spec = importlib.util.spec_from_file_location(module_name, POLICY_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load policy module from {POLICY_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_POLICY = _load_policy_module()
ActionQuote = _POLICY.ActionQuote
AdvisorySignals = _POLICY.AdvisorySignals
OperatingPoint = _POLICY.OperatingPoint
build_endgame_arithmetic_receipt = _POLICY.build_endgame_arithmetic_receipt
decide_endgame_policy = _POLICY.decide_endgame_policy


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--request",
        type=Path,
        help="JSON mapping with operating_point, quotes, optional target_score, and optional advisory_signals",
    )
    return parser.parse_args(argv)


def derive_request(payload: dict[str, Any]) -> dict[str, Any]:
    """Derive one advisory decision from a decoded request payload."""

    if not isinstance(payload, dict):
        raise ValueError("request root must be a JSON object")
    allowed_keys = {"operating_point", "quotes", "target_score", "advisory_signals"}
    unexpected = sorted(repr(key) for key in payload if key not in allowed_keys)
    if unexpected:
        raise ValueError(f"request contains unexpected keys: {', '.join(unexpected)}")
    point = OperatingPoint.from_payload(payload["operating_point"])
    quotes = tuple(ActionQuote.from_payload(row) for row in payload.get("quotes", ()))
    signals = AdvisorySignals.from_payload(payload.get("advisory_signals"))
    decision = decide_endgame_policy(
        point,
        quotes,
        target_score=payload.get("target_score", 0.172),
        advisory_signals=signals,
    )
    return {
        "schema": "ddm_endgame_policy.tool_result.v1",
        "operating_point": point.to_payload(),
        "quotes": [row.to_payload() for row in quotes],
        "decision": decision.to_payload(),
        "research_only": True,
        "score_claim": False,
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.request is None:
        result = build_endgame_arithmetic_receipt()
    else:
        result = derive_request(json.loads(args.request.read_text(encoding="utf-8")))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
