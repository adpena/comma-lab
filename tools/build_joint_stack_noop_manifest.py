#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Write the PARADIGM-gamma typed no-op stack fixture manifest.

This tool creates an integration contract only. It does not build an archive,
dispatch a job, or make a score claim.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.repo_io import write_json  # noqa: E402
from tac.stack_compositions import (  # noqa: E402
    build_joint_admm_balle_arithmetic_noop_manifest,
)


DEFAULT_OUTPUT = (
    "experiments/results/joint_stack_noop_manifest_20260506_codex/manifest.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Write the deterministic, non-dispatching PARADIGM-gamma "
            "ADMM/Balle/AQ stack fixture manifest."
        )
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / DEFAULT_OUTPUT,
        help="Destination JSON path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_joint_admm_balle_arithmetic_noop_manifest(
        repo_root=REPO_ROOT
    )
    write_json(args.output, manifest)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
