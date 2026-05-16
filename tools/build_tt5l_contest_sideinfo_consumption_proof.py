#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a contest-full-frame TT5L side-info consumption proof.

The tool is intentionally narrow: it consumes already-built TT5L baseline and
side-info-mutated archives plus their already-inflated output directories. It
does not run auth eval and does not claim score movement.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)


def _reexec_repo_venv_if_available(repo_root: Path) -> None:
    """Run under the repo venv when invoked as an executable tool."""

    venv_python = repo_root / ".venv" / "bin" / "python"
    if os.environ.get("PACT_ALLOW_SYSTEM_PYTHON") == "1" or not venv_python.is_file():
        return
    if Path(sys.executable).resolve() == venv_python.resolve():
        return
    os.execv(str(venv_python), [str(venv_python), *sys.argv])


_reexec_repo_venv_if_available(REPO_ROOT)
ensure_repo_imports(REPO_ROOT)

from tac.substrates.time_traveler_l5_autonomy.consumption_proof import (  # noqa: E402
    build_tt5l_contest_full_frame_sideinfo_consumption_proof,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-archive", required=True, type=Path)
    parser.add_argument("--mutated-archive", required=True, type=Path)
    parser.add_argument("--baseline-output-dir", required=True, type=Path)
    parser.add_argument("--mutated-output-dir", required=True, type=Path)
    parser.add_argument("--file-list", required=True, type=Path)
    parser.add_argument("--artifact-out", required=True, type=Path)
    parser.add_argument("--manifest-out", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_tt5l_contest_full_frame_sideinfo_consumption_proof(
        baseline_archive_path=args.baseline_archive,
        mutated_archive_path=args.mutated_archive,
        baseline_output_dir=args.baseline_output_dir,
        mutated_output_dir=args.mutated_output_dir,
        file_list_path=args.file_list,
        artifact_path=args.artifact_out,
        manifest_path=args.manifest_out,
        repo_root=REPO_ROOT,
    )
    print(f"wrote {result.proof_path}")
    print(f"wrote {result.manifest_path}")
    print(f"predicate_passed={str(result.proof['predicate_passed']).lower()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
