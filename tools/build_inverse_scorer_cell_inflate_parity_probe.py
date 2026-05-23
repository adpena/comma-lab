#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build an inverse-scorer IAS1 inflate-output parity proof."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.inverse_scorer_cell_inflate_parity import (  # noqa: E402
    DEFAULT_PARITY_SCOPE,
    InverseScorerCellInflateParityError,
    build_inverse_scorer_cell_inflate_parity_probe,
    build_inverse_scorer_cell_inflate_parity_probe_from_archives,
)
from tac.repo_io import ArtifactWriteError, write_json_artifact  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--source-output-dir", type=Path)
    parser.add_argument("--candidate-output-dir", type=Path)
    parser.add_argument("--inflate-runtime-dir", type=Path)
    parser.add_argument("--source-archive", type=Path)
    parser.add_argument("--candidate-archive", type=Path)
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--file-list-entry", action="append", default=[])
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--keep-work-dir", action="store_true")
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--proof-scope", default=DEFAULT_PARITY_SCOPE)
    parser.add_argument("--allow-output-byte-difference", action="store_true")
    parser.add_argument("--fail-if-blocked", action="store_true")
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--expected-json-sha256", default=None)
    args = parser.parse_args(argv)

    try:
        if args.inflate_runtime_dir is not None:
            if args.source_output_dir is not None or args.candidate_output_dir is not None:
                raise SystemExit("use either --inflate-runtime-dir or output dirs, not both")
            proof = build_inverse_scorer_cell_inflate_parity_probe_from_archives(
                candidate_manifest=args.candidate_manifest,
                inflate_runtime_dir=args.inflate_runtime_dir,
                source_archive=args.source_archive,
                candidate_archive=args.candidate_archive,
                repo_root=args.repo_root,
                proof_scope=args.proof_scope,
                expect_output_byte_identical=not args.allow_output_byte_difference,
                timeout_seconds=args.timeout_seconds,
                file_list_entries=tuple(args.file_list_entry or ["0.mkv"]),
                work_dir=args.work_dir,
                keep_work_dir=args.keep_work_dir,
            )
        else:
            if args.source_output_dir is None or args.candidate_output_dir is None:
                raise SystemExit(
                    "either --inflate-runtime-dir or both --source-output-dir and "
                    "--candidate-output-dir are required"
                )
            proof = build_inverse_scorer_cell_inflate_parity_probe(
                candidate_manifest=args.candidate_manifest,
                source_output_dir=args.source_output_dir,
                candidate_output_dir=args.candidate_output_dir,
                repo_root=args.repo_root,
                proof_scope=args.proof_scope,
                expect_output_byte_identical=not args.allow_output_byte_difference,
            )
        blockers = proof.get("blockers") or []
        if args.fail_if_blocked and blockers:
            raise SystemExit("inverse scorer cell inflate parity blocked: " + ",".join(blockers))
        write_json_artifact(
            args.json_out,
            proof,
            allow_overwrite=bool(args.allow_overwrite),
            expected_existing_sha256=args.expected_json_sha256,
        )
    except (InverseScorerCellInflateParityError, ArtifactWriteError) as exc:
        raise SystemExit(str(exc)) from exc

    print(
        json.dumps(
            {
                "schema": "inverse_scorer_cell_inflate_parity_result.v1",
                "json_out": str(args.json_out),
                "full_frame_inflate_output_parity_claim": proof[
                    "full_frame_inflate_output_parity_claim"
                ],
                "source_output_tree_sha256": proof["source_output_tree"]["tree_sha256"],
                "candidate_output_tree_sha256": proof["candidate_output_tree"]["tree_sha256"],
                "blockers": proof.get("blockers") or [],
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
