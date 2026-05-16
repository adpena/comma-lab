#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a TT5L inflate provenance manifest for proof-gate inputs.

This tool does not run auth eval and does not claim score movement. It binds an
already-run TT5L inflate output directory to the archive, file list, runtime
tree, and command that produced it so downstream L5 v2 proofs can reject
caller-supplied output directories without provenance.
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
    TT5L_CONTEST_FRAME_NBYTES,
    build_tt5l_inflate_provenance_manifest,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--file-list", required=True, type=Path)
    parser.add_argument("--artifact-out", required=True, type=Path)
    parser.add_argument("--command", required=True)
    parser.add_argument("--exit-code", default=0, type=int)
    parser.add_argument(
        "--frame-nbytes",
        default=TT5L_CONTEST_FRAME_NBYTES,
        type=int,
        help=(
            "raw output bytes per frame; defaults to contest camera size. "
            "Non-default values are for tests and remain non-promotional."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_tt5l_inflate_provenance_manifest(
        archive_path=args.archive,
        output_dir=args.output_dir,
        file_list_path=args.file_list,
        artifact_path=args.artifact_out,
        command=args.command,
        exit_code=args.exit_code,
        repo_root=REPO_ROOT,
        frame_nbytes=args.frame_nbytes,
    )
    print(f"wrote {result.provenance_path}")
    print(f"output_aggregate_sha256={result.provenance['output_aggregate_sha256']}")
    print("score_claim=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
