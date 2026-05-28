#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize a byte-closed Pact-NeRV-IA3 PIA3 archive candidate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.repo_io import ArtifactWriteError, json_text, sha256_file, write_json_artifact  # noqa: E402
from tac.substrates.pact_nerv_ia3.archive_candidate import (  # noqa: E402
    PACT_NERV_IA3_BYTE_CLOSED_CANDIDATE_SCHEMA,
    PactNervIa3ArchiveCandidateError,
    materialize_pact_nerv_ia3_byte_closed_candidate,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pytorch-state-dict", required=True, type=Path)
    parser.add_argument("--parity-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--manifest-out", required=True, type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        manifest = materialize_pact_nerv_ia3_byte_closed_candidate(
            pytorch_state_dict_path=args.pytorch_state_dict,
            parity_report_path=args.parity_report,
            output_dir=args.output_dir,
            repo_root=REPO_ROOT,
            label=args.label,
            overwrite=args.overwrite,
        )
        manifest_out = _resolve(args.manifest_out)
        expected_existing_sha256 = sha256_file(manifest_out) if manifest_out.is_file() and args.overwrite else None
        write = write_json_artifact(
            manifest_out,
            manifest,
            allow_overwrite=bool(args.overwrite),
            expected_existing_sha256=expected_existing_sha256,
        )
    except (
        ArtifactWriteError,
        OSError,
        PactNervIa3ArchiveCandidateError,
        ValueError,
    ) as exc:
        print(f"FATAL: Pact-NeRV-IA3 byte-closed materialization failed: {exc}", file=sys.stderr)
        return 2
    print(
        json_text(
            {
                "schema": "pact_nerv_ia3_byte_closed_candidate_cli_result.v1",
                "candidate_schema": PACT_NERV_IA3_BYTE_CLOSED_CANDIDATE_SCHEMA,
                "manifest_out": str(args.manifest_out),
                "archive_zip": manifest["candidate_archive_path"],
                "archive_zip_bytes": manifest["candidate_archive_bytes"],
                "receiver_contract_satisfied": manifest["receiver_contract_satisfied"],
                "bytes_written": write.bytes_written,
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
