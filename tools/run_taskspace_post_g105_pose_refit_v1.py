#!/usr/bin/env python3
"""Run the governed exact-public n600 post-G105 generated-Y1 pose refit."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tac.witness_control.taskspace_post_g105_pose_refit_population_v1 import (
    PostG105PoseRefitPopulationError,
    load_population_config,
    run_g121_retained_pose_population,
)
from tac.witness_control.taskspace_post_g105_pose_refit_v1 import (
    write_blocker_receipt,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="strict self-hashed G121-retained population refit config",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        required=True,
        help="SSD run root; required for both cold start and crash resume",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command = [sys.argv[0], *(sys.argv[1:] if argv is None else argv)]
    try:
        config = load_population_config(args.config)
        result = run_g121_retained_pose_population(
            config=config,
            resume_from=args.resume_from,
            command=command,
        )
    except (OSError, ValueError, PostG105PoseRefitPopulationError) as exc:
        blocker = write_blocker_receipt(
            config_path=args.config,
            resume_from=args.resume_from,
            command=command,
            error=exc,
        )
        detail = f"; blocker={blocker}" if blocker is not None else ""
        print(f"REFUSE: {exc}{detail}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            {
                "joint_ledger_path": str(result.joint_ledger_path),
                "joint_ledger_sha256": result.joint_ledger_sha256,
                "retained_stage_count": result.retained_stage_count,
                "every_retained_stage_processed": True,
                "cross_stage_winner_selected": False,
                "score_claim": False,
                "pointer_moved": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
