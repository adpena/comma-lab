#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""CLI wrapper for Z1 scorer-conditional MDL ablation manifests.

This tool is an operator surface around ``tac.analysis.scorer_conditional_mdl``.
It emits measurement artifacts only. It never builds a candidate archive, never
dispatches, and never claims a score.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ImportError:  # pragma: no cover - direct execution from tools/
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.analysis.scorer_conditional_mdl import (  # noqa: E402
    ScorerConditionalMdlError,
    attach_eval_jsons,
    build_scorer_conditional_mdl_ablation,
    dumps_manifest,
    parse_archive_spec,
    render_markdown,
)
from tac.repo_io import json_text  # noqa: E402

DEFAULT_SOURCE_DOCUMENTS = (
    ".omx/research/zen_floor_field_medal_grade_council_20260514.md#Decision-Z1",
    ".omx/research/grand_council_maximize_value_with_time_traveler_seat_20260514.md#Decision-1",
)


def _utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _default_output_dir(repo_root: Path) -> Path:
    return (
        repo_root
        / "experiments"
        / "results"
        / f"lane_zen_floor_scorer_conditional_mdl_ablation_{_utc_stamp()}"
    )


def _resolve_outputs(args: argparse.Namespace, repo_root: Path) -> tuple[Path, Path]:
    output_dir = args.output_dir or _default_output_dir(repo_root)
    output_json = args.output_json or output_dir / "scorer_conditional_mdl_ablation.json"
    output_md = args.output_md or output_dir / "scorer_conditional_mdl_ablation.md"
    return Path(output_json), Path(output_md)


def _eval_spec_labels(eval_specs: list[str]) -> set[str]:
    labels: set[str] = set()
    for spec in eval_specs:
        if "=" not in spec:
            raise ScorerConditionalMdlError("--eval-json must use label=path syntax")
        label, _ = spec.split("=", 1)
        labels.add(label.strip())
    return labels


def _write_outputs(json_path: Path, md_path: Path, payload: dict[str, Any], markdown: str) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json_text(payload), encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")


def _error_payload(message: str) -> dict[str, Any]:
    return {
        "schema": "tac_scorer_conditional_mdl_ablation_cli_error_v1",
        "schema_version": 1,
        "tool": "tools/compute_scorer_conditional_mdl_ablation.py",
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "score_claim": False,
        "score_evidence_grade": "invalid_no_score",
        "dispatch_attempted": False,
        "gpu_required": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "true_scorer_conditional_entropy_claim": False,
        "error": {
            "class": "fail_closed_input_error",
            "message": message,
        },
        "dispatch_blockers": [
            "input_manifest_not_constructed",
            "no_score_claim",
            "no_candidate_archive",
            "no_exact_cuda_eval",
        ],
    }


def _error_markdown(payload: dict[str, Any]) -> str:
    message = payload.get("error", {}).get("message", "unknown error")
    return "\n".join(
        [
            "# Z1 Scorer-Conditional MDL Ablation",
            "",
            "- score_claim: `false`",
            "- promotion_eligible: `false`",
            "- ready_for_exact_eval_dispatch: `false`",
            "- status: `failed_closed`",
            "",
            "## Error",
            "",
            str(message),
            "",
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a proxy-safe Z1 scorer-conditional MDL ablation manifest. "
            "Archives use label=path or label=path,parser=name syntax."
        )
    )
    parser.add_argument(
        "--archive",
        action="append",
        default=[],
        metavar="LABEL=PATH[,parser=NAME]",
        help="Archive spec. May be repeated.",
    )
    parser.add_argument(
        "--eval-json",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Optional exact-eval JSON spec. May be repeated.",
    )
    parser.add_argument(
        "--section-evidence-json",
        type=Path,
        default=None,
        help=(
            "Optional tac_section_scorer_evidence_map_v1 JSON binding parser "
            "sections to scorer component-response or penultimate-feature artifacts."
        ),
    )
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-md", type=Path, default=None)
    parser.add_argument(
        "--source-document",
        action="append",
        default=[],
        help="Additional source document or ledger cross-reference.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    repo_root = Path(args.repo_root).resolve()
    output_json, output_md = _resolve_outputs(args, repo_root)

    try:
        if not args.archive:
            raise ScorerConditionalMdlError("at least one --archive is required")
        archives = [parse_archive_spec(spec) for spec in args.archive]
        archive_labels = {archive.label for archive in archives}
        unknown_eval_labels = _eval_spec_labels(args.eval_json) - archive_labels
        if unknown_eval_labels:
            joined = ", ".join(sorted(unknown_eval_labels))
            raise ScorerConditionalMdlError(
                f"--eval-json label has no matching --archive: {joined}"
            )
        archives = attach_eval_jsons(archives, args.eval_json)
        source_documents = list(DEFAULT_SOURCE_DOCUMENTS) + list(args.source_document)
        manifest = build_scorer_conditional_mdl_ablation(
            archives,
            repo_root=repo_root,
            chunk_size=args.chunk_size,
            source_documents=source_documents,
            section_scorer_evidence=args.section_evidence_json,
        )
        markdown = render_markdown(manifest)
        _write_outputs(output_json, output_md, manifest, markdown)
        print(f"wrote_json={output_json}")
        print(f"wrote_markdown={output_md}")
        return 0
    except (
        ScorerConditionalMdlError,
        ValueError,
        OSError,
        json.JSONDecodeError,
    ) as exc:
        payload = _error_payload(str(exc))
        _write_outputs(output_json, output_md, payload, _error_markdown(payload))
        print(f"failed_closed_json={output_json}", file=sys.stderr)
        print(f"failed_closed_markdown={output_md}", file=sys.stderr)
        print(f"error={exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
