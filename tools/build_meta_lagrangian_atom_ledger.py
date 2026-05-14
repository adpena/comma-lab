#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a planning-only meta-Lagrangian atom ledger."""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.meta_lagrangian_allocator import (  # noqa: E402
    atoms_from_hnerv_decoder_recode_profile,
    build_atom_ledger,
)
from tac.repo_io import json_text, read_json  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atoms-json", type=Path, help="JSON list of atom records.")
    parser.add_argument(
        "--hnerv-decoder-profile",
        type=Path,
        help="Structural-recode profile to convert into rate-only atoms.",
    )
    parser.add_argument("--base-pose-dist", type=float, required=True)
    parser.add_argument("--source", default="manual")
    parser.add_argument("--json-out", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    atoms = []
    if args.atoms_json:
        loaded = read_json(args.atoms_json)
        if not isinstance(loaded, list):
            raise SystemExit("--atoms-json must contain a JSON list")
        atoms.extend(loaded)
    if args.hnerv_decoder_profile:
        atoms.extend(atoms_from_hnerv_decoder_recode_profile(read_json(args.hnerv_decoder_profile)))
    if not atoms:
        raise SystemExit("provide --atoms-json or --hnerv-decoder-profile")
    ledger = build_atom_ledger(atoms, base_pose_dist=args.base_pose_dist, source=args.source)
    text = json_text(ledger)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
