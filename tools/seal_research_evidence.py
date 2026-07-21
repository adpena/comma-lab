#!/usr/bin/env python3
"""CLI for deterministic, content-addressed research-evidence sealing."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from tac.research_evidence import (  # noqa: E402
    EvidenceSealError,
    default_output_dir,
    restore_bundle,
    seal_research_evidence,
    verify_bundle,
)


def _repo_root() -> Path:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=REPO,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise EvidenceSealError("repository root unavailable; refusing unscoped evidence operation") from exc
    return Path(result.stdout.strip()).resolve()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    seal = commands.add_parser("seal", help="seal a regular-file research directory without mutation")
    seal.add_argument("--source", required=True, type=Path)
    seal.add_argument("--output-dir", type=Path)
    verify = commands.add_parser("verify", help="verify every manifest and payload byte in a bundle")
    verify.add_argument("--bundle", required=True, type=Path)
    restore = commands.add_parser("restore", help="restore a verified bundle to a new directory only")
    restore.add_argument("--bundle", required=True, type=Path)
    restore.add_argument("--destination", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        root = _repo_root()
        if args.command == "seal":
            result = seal_research_evidence(
                args.source,
                output_dir=args.output_dir or default_output_dir(args.source),
                repo_root=root,
            )
            print(json.dumps(result.as_dict(), sort_keys=True))
        elif args.command == "verify":
            manifest = verify_bundle(args.bundle, repo_root=root)
            print(json.dumps(manifest.as_dict(), sort_keys=True))
        else:
            manifest = restore_bundle(args.bundle, args.destination, repo_root=root)
            print(json.dumps(manifest.as_dict(), sort_keys=True))
    except EvidenceSealError as exc:
        print(f"evidence seal refused: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
