#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the governed full-n600 research-only selected-plane codec diagnostic."""

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
from tac.witness_dsl.c0b_semantic_quotient import storage_preflight  # noqa: E402
from tac.witness_dsl.taskspace_lossy_selected_plane_codec_v1 import (  # noqa: E402
    LossySelectedPlaneCodecError,
    canonical_json,
    config_identity,
    load_config,
    run_full_experiment,
    sha256_file,
    write_once_or_equal,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    assert_governed_admission("taskspace_lossy_codec_n600")
    try:
        config = load_config(args.config)
        if config["test_only_small_fixture"]:
            raise LossySelectedPlaneCodecError("n600 runner refuses test-only fixtures")
        preflight = storage_preflight(
            args.output_root,
            required_bytes=config["required_free_bytes"],
            test_only_small_fixture=False,
            allow_local_storage=False,
        )
        args.output_root.mkdir(parents=True, exist_ok=True)
        parent_command = subprocess.run(
            ["ps", "-p", str(os.getppid()), "-o", "command="],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        launch_receipt = {
            "schema": "taskspace_lossy_selected_plane_launch_receipt.v1",
            "research_only": True,
            "candidate_lineage_allowed": False,
            "resumable": True,
            "immutable_segment_checkpoints": 5,
            "argv": [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
            "parent_command": parent_command,
            "governed_admission": os.environ.get("TAC_GOVERNED_ADMISSION") == "1",
            "python_version": platform.python_version(),
            "tool_path": str(Path(__file__).resolve()),
            "tool_sha256": sha256_file(Path(__file__).resolve()),
            "config_path": str(args.config.resolve()),
            "config_sha256": config_identity(config),
            "output_root": str(args.output_root.resolve()),
            "storage_preflight": preflight,
        }
        write_once_or_equal(args.output_root / "launch_receipt.json", canonical_json(launch_receipt))
        result = run_full_experiment(config, output_root=args.output_root)
    except LossySelectedPlaneCodecError as exc:
        print(json.dumps({"status": "refused", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
