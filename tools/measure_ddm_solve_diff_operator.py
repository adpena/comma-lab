#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Thin typed CLI for scorer-free DDM solve-difference instrumentation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = _REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tac.optimization.solve_diff_operator_mining import (  # noqa: E402
    SolveDiffMiningConfigV1,
    SolveDiffMiningError,
    run_mining_pass,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-stage-module-sha256")
    parser.add_argument("--pair-limit", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    semantic_argv = [str(Path(__file__).resolve().relative_to(_REPO_ROOT))]
    semantic_argv.extend(sys.argv[1:] if argv is None else argv)
    try:
        raw = args.config.read_bytes()
        config = SolveDiffMiningConfigV1.model_validate_json(raw)
        summary = run_mining_pass(
            config,
            args.output_root,
            pair_limit=args.pair_limit,
            resume=args.resume,
            resume_stage_module_sha256=args.resume_stage_module_sha256,
            argv=semantic_argv,
            config_path=args.config,
        )
    except (OSError, ValueError, json.JSONDecodeError, SolveDiffMiningError) as exc:
        print(f"FAIL_CLOSED: {exc}", file=sys.stderr)
        return 2
    print(
        json.dumps(
            summary.model_dump(mode="json", by_alias=True),
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
