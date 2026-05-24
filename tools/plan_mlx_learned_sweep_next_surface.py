#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Emit the next queue/executor surface for an MLX learned-sweep plan."""

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

from tac.optimization.mlx_learned_sweep_next_surface import (  # noqa: E402
    MLXLearnedSweepNextSurfaceError,
    build_mlx_learned_sweep_next_surface_report,
    render_mlx_learned_sweep_next_surface_markdown,
)
from tac.repo_io import (  # noqa: E402
    ArtifactWriteError,
    sha256_file,
    write_json_artifact,
    write_text_artifact,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--repo-root", default=REPO_ROOT, type=Path)
    parser.add_argument("--max-top-rows", default=8, type=int)
    parser.add_argument("--allow-overwrite", action="store_true")
    return parser.parse_args(argv)


def _repo_rel(path: Path, *, repo_root: Path) -> str:
    try:
        return path.resolve(strict=False).relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    plan_path = args.plan if args.plan.is_absolute() else repo_root / args.plan
    output_path = args.output if args.output.is_absolute() else repo_root / args.output
    markdown_path = (
        None
        if args.markdown_output is None
        else (
            args.markdown_output
            if args.markdown_output.is_absolute()
            else repo_root / args.markdown_output
        )
    )

    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    report = build_mlx_learned_sweep_next_surface_report(
        plan,
        source_plan={
            "path": _repo_rel(plan_path, repo_root=repo_root),
            "sha256": sha256_file(plan_path),
            "bytes": plan_path.stat().st_size,
            "schema": plan.get("schema"),
        },
        max_top_rows=args.max_top_rows,
    )
    json_result = write_json_artifact(
        output_path,
        report,
        allow_overwrite=args.allow_overwrite,
    )
    markdown_result = None
    if markdown_path is not None:
        markdown_result = write_text_artifact(
            markdown_path,
            render_mlx_learned_sweep_next_surface_markdown(report),
            allow_overwrite=args.allow_overwrite,
        )
    print(
        json.dumps(
            {
                "schema": "mlx_dynamic_learned_sweep_next_surface_cli.v1",
                "report": json_result.__dict__,
                "markdown": None if markdown_result is None else markdown_result.__dict__,
                "recommended_next_surface": report["recommended_next_surface"],
                "blockers": report["blockers"],
                "routing_notes": report["routing_notes"],
                "authority": {
                    "score_claim": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                    "dispatch_attempted": False,
                    "gpu_launched": False,
                },
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ArtifactWriteError, MLXLearnedSweepNextSurfaceError) as exc:
        raise SystemExit(f"FATAL: {exc}") from exc
