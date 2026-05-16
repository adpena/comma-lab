#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the TT5L byte-closed temporal side-info consumption proof."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.substrates.time_traveler_l5_autonomy.consumption_proof import (  # noqa: E402
    TT5L_SIDEINFO_CONSUMPTION_DEFAULT_ARTIFACT,
    TT5L_SIDEINFO_CONSUMPTION_DEFAULT_MANIFEST,
    TT5L_SIDEINFO_CONSUMPTION_DEFAULT_WORK_DIR,
    build_tt5l_sideinfo_consumption_proof,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-json",
        type=Path,
        default=Path(TT5L_SIDEINFO_CONSUMPTION_DEFAULT_ARTIFACT),
        help="Repo-relative or absolute proof JSON output path.",
    )
    parser.add_argument(
        "--manifest-json",
        type=Path,
        default=Path(TT5L_SIDEINFO_CONSUMPTION_DEFAULT_MANIFEST),
        help="Repo-relative or absolute inflated-output manifest JSON path.",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path(TT5L_SIDEINFO_CONSUMPTION_DEFAULT_WORK_DIR),
        help="Repo-relative or absolute ignored work directory for toy archives/raw outputs.",
    )
    return parser.parse_args(argv)


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPO_ROOT))
    except ValueError:
        return str(resolved)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = build_tt5l_sideinfo_consumption_proof(
        artifact_path=args.artifact_json,
        manifest_path=args.manifest_json,
        work_dir=args.work_dir,
        repo_root=REPO_ROOT,
    )
    print(
        json.dumps(
            {
                "artifact_json": _repo_relative(result.proof_path),
                "manifest_json": _repo_relative(result.manifest_path),
                "artifact_sha256": _sha256_file(result.proof_path),
                "predicate_passed": result.proof["predicate_passed"],
                "runtime_tree_sha256": result.proof["runtime_tree_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result.proof["predicate_passed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
