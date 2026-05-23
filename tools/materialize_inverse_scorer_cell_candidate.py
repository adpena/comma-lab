#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize a planning-only inverse-scorer cell candidate archive."""

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

from tac.optimization.inverse_scorer_cell_materializer import (  # noqa: E402
    InverseScorerCellMaterializerError,
    materialize_inverse_scorer_cell_candidate,
)
from tac.repo_io import ArtifactWriteError, write_json_artifact  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-archive-template", type=Path, required=True)
    parser.add_argument("--inverse-action-functional", type=Path, required=True)
    parser.add_argument("--raw-contest-video-digest", required=True)
    parser.add_argument("--output-archive", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--runtime-consumption-proof", type=Path, default=None)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--atom-id", action="append", default=[])
    parser.add_argument("--selected-limit", type=int, default=None)
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--expected-output-sha256", default=None)
    parser.add_argument("--expected-manifest-sha256", default=None)
    args = parser.parse_args(argv)

    try:
        manifest = materialize_inverse_scorer_cell_candidate(
            raw_contest_video_digest=args.raw_contest_video_digest,
            candidate_archive_template=args.candidate_archive_template,
            inverse_action_functional=args.inverse_action_functional,
            output_archive=args.output_archive,
            runtime_consumption_proof=args.runtime_consumption_proof,
            atom_ids=tuple(args.atom_id),
            selected_limit=args.selected_limit,
            repo_root=args.repo_root,
            allow_overwrite=bool(args.allow_overwrite),
            expected_existing_output_sha256=args.expected_output_sha256,
        )
        write_json_artifact(
            args.manifest_out,
            manifest,
            allow_overwrite=bool(args.allow_overwrite),
            expected_existing_sha256=args.expected_manifest_sha256,
        )
    except (InverseScorerCellMaterializerError, ArtifactWriteError) as exc:
        raise SystemExit(str(exc)) from exc

    print(
        json.dumps(
            {
                "schema": "inverse_scorer_cell_candidate_materialize_result.v1",
                "manifest_out": str(args.manifest_out),
                "output_archive": str(args.output_archive),
                "candidate_archive_sha256": manifest["candidate_archive"]["sha256"],
                "selected_cell_count": manifest[
                    "inverse_scorer_cell_descriptor"
                ]["selected_cell_count"],
                "receiver_contract_satisfied": manifest[
                    "receiver_contract_satisfied"
                ],
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
