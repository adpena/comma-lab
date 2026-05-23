#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build an inverse-scorer cell receiver proof from a runtime adapter manifest."""

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
    build_inverse_scorer_cell_receiver_proof,
)
from tac.repo_io import ArtifactWriteError, write_json_artifact  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-adapter-manifest", type=Path, required=True)
    parser.add_argument("--candidate-manifest", type=Path, default=None)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--fail-if-not-ready", action="store_true")
    parser.add_argument("--allow-overwrite", action="store_true")
    parser.add_argument("--expected-json-sha256", default=None)
    args = parser.parse_args(argv)

    try:
        proof = build_inverse_scorer_cell_receiver_proof(
            runtime_adapter_manifest=args.runtime_adapter_manifest,
            candidate_manifest=args.candidate_manifest,
            repo_root=args.repo_root,
        )
        if args.fail_if_not_ready and proof["ready_for_exact_eval_runtime"] is not True:
            raise SystemExit(
                "inverse scorer cell receiver proof not ready: "
                + ",".join(proof.get("blockers") or [])
            )
        write_json_artifact(
            args.json_out,
            proof,
            allow_overwrite=bool(args.allow_overwrite),
            expected_existing_sha256=args.expected_json_sha256,
        )
    except (InverseScorerCellMaterializerError, ArtifactWriteError) as exc:
        raise SystemExit(str(exc)) from exc

    print(
        json.dumps(
            {
                "schema": "inverse_scorer_cell_receiver_proof_result.v1",
                "json_out": str(args.json_out),
                "ready_for_exact_eval_runtime": proof[
                    "ready_for_exact_eval_runtime"
                ],
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
