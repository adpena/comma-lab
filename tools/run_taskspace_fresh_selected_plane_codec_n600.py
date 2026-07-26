#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run the governed fresh full-n600 selected-plane codec and final recode."""

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
from tac.witness_dsl.taskspace_fresh_selected_plane_codec_v1 import (  # noqa: E402
    FreshSelectedPlaneCodecError,
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


def _open_provider(config: dict[str, object]):
    try:
        from tac.witness_control.taskspace_fresh_scorer_plane_materializer_v1 import (
            FreshScorerPlaneOperandLoaderV1,
        )
    except ImportError as exc:
        raise FreshSelectedPlaneCodecError("fresh scorer-plane operand loader is unavailable") from exc
    source = config["operand_provider"]
    if not isinstance(source, dict):
        raise FreshSelectedPlaneCodecError("operand_provider config is malformed")
    try:
        return FreshScorerPlaneOperandLoaderV1.open(
            source["aggregate_receipt_path"],
            expected_sha256=source["aggregate_receipt_sha256"],
        )
    except Exception as exc:
        raise FreshSelectedPlaneCodecError("fresh operand loader refused configured custody") from exc


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    assert_governed_admission("taskspace_fresh_selected_plane_codec_n600")
    try:
        config = load_config(args.config)
        if config["test_only_small_fixture"]:
            raise FreshSelectedPlaneCodecError("n600 runner refuses test-only fixtures")
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
            "schema": "taskspace_fresh_selected_plane_codec_launch.v1",
            "research_only": True,
            "candidate_lineage_allowed": True,
            "historical_payload_reused": False,
            "score_claim": False,
            "promotion_eligible": False,
            "resumable": True,
            "immutable_encoder_stage_checkpoints": 5,
            "whole_population_final_recode": True,
            "public_decode_authority": "PyAV",
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
        provider = _open_provider(dict(config))
        result = run_full_experiment(config, provider, output_root=args.output_root, repo_root=REPO_ROOT)
    except FreshSelectedPlaneCodecError as exc:
        print(json.dumps({"status": "refused", "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
