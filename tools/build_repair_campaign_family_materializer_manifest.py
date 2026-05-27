#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a family-specific repair materializer manifest for one allocation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    _TOOL_DIR = Path(__file__).resolve().parent
    _REPO_ROOT = _TOOL_DIR.parent
    for _path in (str(_REPO_ROOT), str(_TOOL_DIR)):
        if _path not in sys.path:
            sys.path.insert(0, _path)
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.optimization.repair_family_materializers import (  # noqa: E402
    RepairFamilyMaterializerError,
    build_repair_campaign_family_materializer_manifest,
)
from tac.repo_io import (  # noqa: E402
    ArtifactWriteError,
    json_text,
    sha256_file,
    write_json_artifact,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--materialization-plan", required=True, type=Path)
    parser.add_argument("--score-report", required=True, type=Path)
    parser.add_argument("--typed-response-id", required=True)
    parser.add_argument("--candidate-id", default="")
    parser.add_argument("--materializer-manifest-out", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RepairFamilyMaterializerError(f"{path} must contain a JSON object")
    return payload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        plan_path = _resolve(args.materialization_plan)
        score_report_path = _resolve(args.score_report)
        manifest = build_repair_campaign_family_materializer_manifest(
            repo_root=REPO_ROOT,
            materialization_plan=_load_json(plan_path),
            materialization_plan_path=args.materialization_plan,
            score_report=_load_json(score_report_path),
            score_report_path=args.score_report,
            typed_response_id=args.typed_response_id,
            candidate_id=args.candidate_id,
        )
        manifest_out = _resolve(args.materializer_manifest_out)
        expected_existing_sha256 = None
        write_result = None
        skipped_identical_existing_artifact = False
        if manifest_out.exists() and args.overwrite:
            existing_text = manifest_out.read_text(encoding="utf-8")
            next_text = json_text(manifest)
            if existing_text == next_text:
                skipped_identical_existing_artifact = True
            else:
                expected_existing_sha256 = sha256_file(manifest_out)
        if not skipped_identical_existing_artifact:
            write_result = write_json_artifact(
                manifest_out,
                manifest,
                allow_overwrite=bool(args.overwrite),
                expected_existing_sha256=expected_existing_sha256,
            )
    except (
        ArtifactWriteError,
        OSError,
        RepairFamilyMaterializerError,
        ValueError,
    ) as exc:
        print(f"FATAL: repair family materializer failed: {exc}", file=sys.stderr)
        return 2
    print(
        json_text(
            {
                "schema": "repair_campaign_family_materializer_manifest_cli_result.v1",
                "materialization_plan": str(args.materialization_plan),
                "score_report": str(args.score_report),
                "materializer_manifest_out": str(args.materializer_manifest_out),
                "typed_response_id": args.typed_response_id,
                "candidate_id": args.candidate_id or None,
                "target_kind": manifest.get("target_kind"),
                "byte_closed_candidate_emitted": (
                    manifest.get("byte_closed_candidate_emitted") is True
                ),
                "component_response_replayed": (
                    manifest.get("component_response_replayed") is True
                ),
                "blockers": manifest.get("readiness_blockers") or [],
                "bytes_written": (
                    write_result.bytes_written if write_result is not None else 0
                ),
                "skipped_identical_existing_artifact": (
                    skipped_identical_existing_artifact
                ),
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "budget_spend_allowed": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
