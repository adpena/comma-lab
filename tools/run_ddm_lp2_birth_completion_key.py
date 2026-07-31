#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""CLI: emit the typed ``birth_completion`` gate-key verdict from telemetry + QA91 inventory.

The rung-1 continuation window loop calls this BETWEEN windows (trainer stays sealed, gc12 §5):

    .venv/bin/python tools/run_ddm_lp2_birth_completion_key.py \
        --telemetry /Volumes/VertigoDataTier/pact/<run>/telemetry.jsonl \
        --qa91-inventory /Volumes/VertigoDataTier/pact/ddm_fp1_20260731/qa91_erased_lane.json \
        --output-json /Volumes/VertigoDataTier/pact/<run>/birth_completion_gate.json

Exit code: 0 always on a valid verdict (fired True or False is DATA, not an error); 3 on fail-closed
telemetry/inventory error. Delegates entirely to ``tac.optimization.ddm_lp2_birth_completion``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tac.optimization.ddm_lp2_birth_completion import (
    DEFAULT_ALPHA_ONE_SIDED,
    DEFAULT_EPOCHS_PER_GATE,
    DEFAULT_WINDOW_GATES,
    BirthCompletionTelemetryError,
    evaluate_from_paths,
    verdict_to_row,
)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--telemetry", required=True, help="path to the run's telemetry.jsonl")
    p.add_argument(
        "--qa91-inventory",
        required=True,
        help="path to qa91_erased_lane.json (ddm_fp1_qa91_erased_lane.v1)",
    )
    p.add_argument("--output-json", default=None, help="optional path to write the verdict row")
    p.add_argument("--window-gates", type=int, default=DEFAULT_WINDOW_GATES)
    p.add_argument("--epochs-per-gate", type=int, default=DEFAULT_EPOCHS_PER_GATE)
    p.add_argument("--alpha-one-sided", type=float, default=DEFAULT_ALPHA_ONE_SIDED)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        verdict = evaluate_from_paths(
            args.telemetry,
            args.qa91_inventory,
            window_gates=args.window_gates,
            epochs_per_gate=args.epochs_per_gate,
            alpha_one_sided=args.alpha_one_sided,
        )
    except BirthCompletionTelemetryError as exc:
        print(f"FAIL-CLOSED: {exc}", file=sys.stderr)
        return 3
    row = verdict_to_row(verdict)
    payload = json.dumps(row, indent=1, sort_keys=True)
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(out.suffix + ".tmp")
        tmp.write_text(payload + "\n", encoding="utf-8")
        tmp.replace(out)
    print(payload)
    print(
        f"\ngate_key=birth_completion fired={verdict.fired} "
        f"slope={verdict.fit.slope_comp_per_gate:.4f} epsilon={verdict.fit.epsilon_comp_per_gate:.4f} "
        f"erased={verdict.erased_count} above_nucleus_est={verdict.above_nucleus_erased_estimate}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
