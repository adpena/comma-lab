#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Append one cost-band measurement anchor to the posterior at
``.omx/state/cost_band_posterior.jsonl``.

Wrapper integration (one-liner at end of operator dispatch script, after
``wall_clock_sec`` + ``actual_cost_usd`` are known):

    python tools/append_cost_band_anchor.py \\
        --dispatch-label "$INSTANCE_JOB_ID" \\
        --trainer experiments/train_<name>.py \\
        --platform modal --gpu T4 \\
        --epochs 3000 --batch-size 32 --all-flags-on \\
        --actual-wall-clock-sec 5400 \\
        --actual-cost-usd 1.85 \\
        --predicted-cost-usd-low 1.50 --predicted-cost-usd-high 3.50

The append is fcntl-locked (sister of Catalog #128 atomic-write pattern).
The next ``tac.cost_band_calibration.predict(...)`` call will incorporate
this measurement into its p10/p50/p90 estimate.
"""
from __future__ import annotations

import argparse
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

_REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(_REPO_ROOT)

from tac.cost_band_calibration import (  # noqa: E402
    POSTERIOR_PATH,
    SUCCESSFUL_DISPATCH,
    VALID_OUTCOMES,
    CostBandAnchor,
    _now_utc_iso,
    append_anchor,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Append one cost-band measurement anchor to the JSONL posterior."
    )
    parser.add_argument("--dispatch-label", required=True,
                        help="Operator-facing unique label (e.g. INSTANCE_JOB_ID)")
    parser.add_argument("--trainer", required=True,
                        help="Trainer script path, e.g. experiments/train_x.py")
    parser.add_argument("--platform", required=True,
                        choices=["modal", "vastai", "lightning", "azure", "kaggle", "github", "local"],
                        help="Dispatch platform")
    parser.add_argument("--gpu", required=True,
                        help="GPU class: T4 | A10G | A100 | H100 | 4090 | etc.")
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--all-flags-on", action="store_true",
                        help="Set if every TIER_1_OPERATOR_REQUIRED_FLAGS entry was threaded")
    parser.add_argument("--actual-wall-clock-sec", type=float, required=True)
    parser.add_argument("--actual-cost-usd", type=float, required=True)
    parser.add_argument("--predicted-cost-usd-low", type=float, default=None)
    parser.add_argument("--predicted-cost-usd-high", type=float, default=None)
    parser.add_argument(
        "--outcome",
        choices=sorted(VALID_OUTCOMES),
        default=SUCCESSFUL_DISPATCH,
        help="Dispatch outcome; failed/timed-out rows are retained but excluded from predictions by default.",
    )
    parser.add_argument("--returncode", type=int, default=None)
    parser.add_argument("--notes", default="",
                        help="Optional free-text annotation")
    parser.add_argument("--posterior-path", type=Path, default=None,
                        help="Override (testing only); default is .omx/state/cost_band_posterior.jsonl")
    args = parser.parse_args()

    prediction_in_band = None
    if args.predicted_cost_usd_low is not None and args.predicted_cost_usd_high is not None:
        prediction_in_band = (
            args.predicted_cost_usd_low <= args.actual_cost_usd <= args.predicted_cost_usd_high
        )

    anchor = CostBandAnchor(
        logged_at_utc=_now_utc_iso(),
        dispatch_label=args.dispatch_label,
        trainer=args.trainer,
        platform=args.platform,
        gpu=args.gpu,
        epochs=args.epochs,
        batch_size=args.batch_size,
        all_flags_on=args.all_flags_on,
        actual_wall_clock_sec=args.actual_wall_clock_sec,
        actual_cost_usd=args.actual_cost_usd,
        predicted_cost_usd_low=args.predicted_cost_usd_low,
        predicted_cost_usd_high=args.predicted_cost_usd_high,
        prediction_in_band=prediction_in_band,
        outcome=args.outcome,
        returncode=args.returncode,
        notes=args.notes,
    )
    append_anchor(anchor, posterior_path=args.posterior_path)
    target = args.posterior_path or POSTERIOR_PATH
    print(
        f"[cost-band-anchor] appended {args.dispatch_label} "
        f"(platform={args.platform},gpu={args.gpu},epochs={args.epochs},"
        f"cost=${args.actual_cost_usd:.2f}) to {target}"
    )
    if prediction_in_band is not None:
        tag = "IN-BAND" if prediction_in_band else "OUT-OF-BAND"
        print(
            f"[cost-band-anchor] prediction band ${args.predicted_cost_usd_low:.2f}-"
            f"${args.predicted_cost_usd_high:.2f} vs actual ${args.actual_cost_usd:.2f}: {tag}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
