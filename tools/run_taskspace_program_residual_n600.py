#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the governed, resumable G49 residual-transport/G58 producer seam."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.admission_guard import assert_governed_admission  # noqa: E402
from tac.witness_dsl.taskspace_program_residual_producer_v1 import (  # noqa: E402
    PRIMARY_CODEC_BLOCKERS,
    ProgramResidualProducerError,
    canonical_json,
    load_config,
    publish_write_once,
    run_structural_producer,
    stable_file_identity,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument(
        "--resume-from",
        required=True,
        type=Path,
        help="Exact output_root containing immutable per-stage checkpoints.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    assert_governed_admission("taskspace_program_residual_n600")
    try:
        config = load_config(args.config)
        if config.test_only_small_fixture:
            raise ProgramResidualProducerError("n600 runner refuses test-only fixtures")
        if args.resume_from.expanduser().resolve() != config.output_paths["output_root"]:
            raise ProgramResidualProducerError("--resume-from must equal typed config output_root")
        parent = subprocess.run(
            ["ps", "-p", str(os.getppid()), "-o", "command="],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        launch = {
            "schema": "tac.taskspace_program_residual_launch.v1",
            "status": "STRUCTURAL_RESIDUAL_TRANSPORT_ONLY",
            "run_id": config.run_id,
            "campaign_id": config.campaign_id,
            "producer_role": config.raw["producer_role"],
            "config": config.file_identity.to_mapping(),
            "resume_from": str(args.resume_from.expanduser().resolve()),
            "resumable": True,
            "immutable_stage_checkpoints": config.stage_count,
            "no_prior_stage_redecode_required": True,
            "governed_admission": os.environ.get("TAC_GOVERNED_ADMISSION") == "1",
            "argv": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
            "parent_command": parent,
            "python_version": platform.python_version(),
            "tool": stable_file_identity(
                Path(__file__).resolve(),
                label="program residual runner source",
            ).to_mapping(),
            "candidate_admission": False,
            "score_claim": False,
            "promotion_eligible": False,
            "expected_terminal_blockers": list(PRIMARY_CODEC_BLOCKERS),
        }
        publish_write_once(
            config.output_paths["output_root"] / "launch_receipt.json",
            canonical_json(launch),
            label="program residual launch receipt",
        )
        result = run_structural_producer(config)
    except ProgramResidualProducerError as exc:
        print(
            json.dumps(
                {
                    "status": "REFUSED",
                    "candidate_admission": False,
                    "error": str(exc),
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(result, sort_keys=True))
    # The current honest terminal condition is a named G59 candidate refusal.
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
