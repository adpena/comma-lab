#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Execute one HPRC pair-scoped runner row through incremental MLX response."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.substrates.hprc.incremental_runner_execution import (  # noqa: E402
    build_hprc_incremental_runner_execution_report,
    prepare_hprc_incremental_runner_execution,
    write_hprc_incremental_runner_execution_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner-plan", required=True, type=Path)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--repo-root", default=REPO_ROOT, type=Path)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--scorer-batch-pairs", type=int, default=1)
    parser.add_argument("--cache-render-batch-pairs", type=int, default=8)
    parser.add_argument("--proof-root", action="append", type=Path, default=[])
    parser.add_argument("--no-allow-large-tensor-cache", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve(strict=False)
    output_dir = _resolve(args.output_dir, base=repo_root)
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise SystemExit(f"output dir exists; pass --force: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    prep = prepare_hprc_incremental_runner_execution(
        runner_plan_path=args.runner_plan,
        candidate_id=str(args.candidate_id),
        output_dir=output_dir,
        repo_root=repo_root,
        scorer_batch_pairs=int(args.scorer_batch_pairs),
        cache_render_batch_pairs=int(args.cache_render_batch_pairs),
        device=str(args.device),
        allow_large_tensor_cache=not bool(args.no_allow_large_tensor_cache),
    )
    prep_path = output_dir / "hprc_incremental_runner_execution_prep.json"
    prep_path.write_text(json.dumps(prep, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    if args.plan_only:
        print(
            json.dumps(
                {
                    "prep": prep_path.as_posix(),
                    "archive_sha256": prep["archive"]["sha256"],
                    "plan_only": True,
                    "score_claim": False,
                },
                sort_keys=True,
            )
        )
        return 0

    started = time.time()
    completed = subprocess.run(
        list(prep["incremental_command_argv"]),
        cwd=repo_root,
        text=True,
        capture_output=True,
    )
    elapsed = time.time() - started
    if completed.returncode != 0:
        raise SystemExit(
            "incremental runner command failed:\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    retention_path = Path(prep["expected_cache_retention_plan"])
    retention = _run_retention_plan(
        repo_root=repo_root,
        cache_root=Path(prep["incremental_output_dir"]) / "mlx_incremental_cache",
        output_path=retention_path,
    )
    report = build_hprc_incremental_runner_execution_report(
        prep=prep,
        incremental_report_path=prep["expected_incremental_report"],
        incremental_stdout=completed.stdout,
        incremental_stderr_tail=completed.stderr[-4000:],
        incremental_elapsed_seconds=elapsed,
        retention_plan_path=retention_path,
        retention_stdout=retention.stdout,
        retention_stderr_tail=retention.stderr[-4000:],
        proof_roots=[_resolve(path, base=repo_root) for path in args.proof_root],
    )
    report_path = output_dir / "hprc_incremental_runner_execution_report.json"
    write_hprc_incremental_runner_execution_report(
        output_path=report_path,
        report=report,
        allow_overwrite=True,
    )
    print(
        json.dumps(
            {
                "report": report_path.as_posix(),
                "archive_sha256": report["archive"]["sha256"],
                "changed_pair_count": report["incremental_summary"]["changed_pair_count"],
                "delta_total_mlx_score_advisory": report["incremental_summary"][
                    "delta_total_mlx_score_advisory"
                ],
                "ready_for_exact_eval_dispatch": False,
                "score_claim": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _run_retention_plan(*, repo_root: Path, cache_root: Path, output_path: Path) -> subprocess.CompletedProcess[str]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(repo_root / "tools" / "compact_experiment_artifacts.py"),
        cache_root.as_posix(),
        "--repo-root",
        repo_root.as_posix(),
        "--include-kind",
        "mlx_scorer_input_cache",
        "--min-bytes",
        "1mb",
        "--json-output",
        output_path.as_posix(),
    ]
    completed = subprocess.run(cmd, cwd=repo_root, text=True, capture_output=True)
    if completed.returncode != 0:
        raise SystemExit(
            "artifact-retention plan failed:\n"
            f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )
    return completed


def _resolve(path: Path, *, base: Path) -> Path:
    expanded = path.expanduser()
    return expanded if expanded.is_absolute() else (base / expanded).resolve(strict=False)


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
