#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a queue-owned final-rate attack against frontier archives.

The default path resolves the current canonical contest-CPU frontier archive,
accepts any explicit comparison archives, and emits an ``experiment_queue.v1``
that runs family-agnostic materializer sweeps locally. It is local/advisory
only; exact auth eval is still required before any score or promotion claim.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from comma_lab.scheduler.experiment_queue import default_state_path  # noqa: E402
from comma_lab.scheduler.frontier_rate_attack_bootstrap import (  # noqa: E402
    DEFAULT_EXECUTABLE_TARGET_KINDS,
    DEFAULT_FRONTIER_POINTER,
    FrontierRateAttackBootstrapError,
    build_frontier_rate_attack_payloads,
    parse_archive_spec,
    resolve_current_frontier_archive,
)
from tac.repo_io import ArtifactWriteError, json_text, write_json_artifact  # noqa: E402


def _utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _display_path(path: str | Path) -> str:
    value = Path(path)
    try:
        return value.resolve(strict=False).relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return value.as_posix()


def _default_results_root() -> Path:
    for root in (
        Path("/Volumes/VertigoDataTier/experiments/results"),
        Path("/Volumes/APDataStore/experiments/results"),
    ):
        if root.exists():
            return root
    return REPO_ROOT / "experiments" / "results"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-id", default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--archive", action="append", default=[], help="Explicit label=path archive.")
    parser.add_argument("--frontier-axis", default="contest_cpu")
    parser.add_argument("--frontier-pointer", default=DEFAULT_FRONTIER_POINTER)
    parser.add_argument("--request-search-root", action="append", default=[])
    parser.add_argument("--archive-search-root", action="append", default=[])
    parser.add_argument(
        "--no-current-frontier",
        dest="include_current_frontier",
        action="store_false",
        help="Do not resolve and include the canonical current frontier archive.",
    )
    parser.set_defaults(include_current_frontier=True)
    parser.add_argument("--target-kind", action="append", default=None)
    parser.add_argument(
        "--no-optional-target-blockers",
        dest="include_optional_target_blockers",
        action="store_false",
        help="Do not emit typed blockers for section/tensor targets lacking manifests.",
    )
    parser.set_defaults(include_optional_target_blockers=True)
    parser.add_argument("--member-name", default=None)
    parser.add_argument("--section-manifest", default=None)
    parser.add_argument("--section-name", action="append", default=[])
    parser.add_argument("--tensor-manifest", default=None)
    parser.add_argument("--factorization-contract", default=None)
    parser.add_argument("--tensor-factorize-rank", type=int, default=None)
    parser.add_argument("--zip-compression-method", action="append", default=[])
    parser.add_argument("--zip-compresslevel", action="append", type=int, default=[])
    parser.add_argument("--min-free-bytes", type=int, default=0)
    parser.add_argument(
        "--allow-materializer-overwrite",
        action="store_true",
        help="Pass --allow-overwrite to materializer sweep outputs.",
    )
    parser.add_argument("--local-cpu-concurrency", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--max-parallel", type=int, default=0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--poll-interval-seconds", type=float, default=0.05)
    parser.add_argument("--idle-sleep-seconds", type=float, default=0.0)
    parser.add_argument("--max-idle-cycles", type=int, default=1)
    return parser.parse_args(argv)


def _run_command(command: list[str]) -> dict[str, Any]:
    started = time.monotonic()
    result = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True)
    return {
        "command": command,
        "returncode": result.returncode,
        "elapsed_seconds": time.monotonic() - started,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def _write_outputs(output_dir: Path, payloads: dict[str, Any]) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "bootstrap": output_dir / "frontier_rate_attack_bootstrap.json",
        "materializer_contexts": output_dir / "materializer_contexts.json",
        "materializer_backlog": output_dir / "materializer_backlog.json",
        "materializer_work_queue": output_dir / "materializer_work_queue.json",
        "experiment_queue": output_dir / "experiment_queue.json",
    }
    write_json_artifact(paths["materializer_contexts"], payloads["contexts"])
    write_json_artifact(paths["materializer_backlog"], payloads["backlog"])
    write_json_artifact(paths["materializer_work_queue"], payloads["work_queue"])
    write_json_artifact(paths["experiment_queue"], payloads["queue"])
    bootstrap = dict(payloads["bootstrap"])
    bootstrap["artifacts"] = {key: _display_path(path) for key, path in paths.items() if key != "bootstrap"}
    bootstrap["operator_commands"] = {
        "validate": [
            ".venv/bin/python",
            "tools/experiment_queue.py",
            "--queue",
            _display_path(paths["experiment_queue"]),
            "validate",
        ],
        "init": [
            ".venv/bin/python",
            "tools/experiment_queue.py",
            "--queue",
            _display_path(paths["experiment_queue"]),
            "init",
        ],
        "run_worker": [
            ".venv/bin/python",
            "tools/experiment_queue.py",
            "--queue",
            _display_path(paths["experiment_queue"]),
            "run-worker",
            "--execute",
            "--max-steps",
            "<N>",
            "--max-parallel",
            "<P>",
        ],
    }
    write_json_artifact(paths["bootstrap"], bootstrap)
    return {key: _display_path(path) for key, path in paths.items()}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    stamp = _utc_stamp()
    queue_id = args.queue_id or f"frontier_final_rate_attack_{stamp}"
    output_dir = args.output_dir or (
        REPO_ROOT / ".omx" / "research" / f"frontier_final_rate_attack_{stamp}"
    )
    results_root = args.results_root or (_default_results_root() / "frontier_final_rate_attack")
    try:
        archive_records: list[dict[str, Any]] = []
        frontier_resolution = None
        if args.include_current_frontier:
            frontier_resolution = resolve_current_frontier_archive(
                repo_root=REPO_ROOT,
                frontier_axis=args.frontier_axis,
                pointer_path=args.frontier_pointer,
                request_search_roots=tuple(args.request_search_root),
                archive_search_roots=tuple(args.archive_search_root),
            )
            archive_records.append(dict(frontier_resolution["archive_record"]))
        for spec in args.archive:
            archive_records.append(parse_archive_spec(spec, repo_root=REPO_ROOT))
        target_kinds = tuple(args.target_kind or DEFAULT_EXECUTABLE_TARGET_KINDS)
        payloads = build_frontier_rate_attack_payloads(
            repo_root=REPO_ROOT,
            queue_id=queue_id,
            archive_records=archive_records,
            results_root=results_root,
            target_kinds=target_kinds,
            include_optional_target_blockers=args.include_optional_target_blockers,
            member_name=args.member_name,
            section_manifest=args.section_manifest,
            section_names=tuple(args.section_name),
            tensor_manifest=args.tensor_manifest,
            factorization_contract=args.factorization_contract,
            tensor_factorize_rank=args.tensor_factorize_rank,
            zip_compression_methods=tuple(args.zip_compression_method or ("stored", "deflated")),
            zip_compresslevels=tuple(args.zip_compresslevel or (1, 6, 9)),
            min_free_bytes=args.min_free_bytes,
            allow_overwrite=args.allow_materializer_overwrite,
            local_cpu_concurrency=args.local_cpu_concurrency,
            lane_id=f"lane_{queue_id}",
        )
        if frontier_resolution is not None:
            payloads["bootstrap"]["frontier_resolution"] = frontier_resolution
        artifact_paths = _write_outputs(output_dir, payloads)
        queue_path = Path(artifact_paths["experiment_queue"])
        if not queue_path.is_absolute():
            queue_path = REPO_ROOT / queue_path
        execution_report: dict[str, Any] | None = None
        if args.execute:
            state_path = default_state_path(REPO_ROOT, queue_id)
            commands = [
                [
                    ".venv/bin/python",
                    "tools/experiment_queue.py",
                    "--queue",
                    _display_path(queue_path),
                    "--state",
                    _display_path(state_path),
                    "init",
                ],
                [
                    ".venv/bin/python",
                    "tools/experiment_queue.py",
                    "--queue",
                    _display_path(queue_path),
                    "--state",
                    _display_path(state_path),
                    "run-worker",
                    "--execute",
                    "--max-steps",
                    str(args.max_steps),
                    "--max-parallel",
                    str(args.max_parallel),
                    "--poll-interval-seconds",
                    str(args.poll_interval_seconds),
                    "--idle-sleep-seconds",
                    str(args.idle_sleep_seconds),
                    "--max-idle-cycles",
                    str(args.max_idle_cycles),
                ],
                [
                    ".venv/bin/python",
                    "tools/experiment_queue.py",
                    "--queue",
                    _display_path(queue_path),
                    "--state",
                    _display_path(state_path),
                    "observe",
                    "--format",
                    "json",
                ],
            ]
            results = [_run_command(command) for command in commands]
            execution_report = {
                "schema": "frontier_final_rate_attack_execution_report.v1",
                "queue_id": queue_id,
                "state_path": _display_path(state_path),
                "commands": results,
                "failed_command_count": sum(1 for result in results if result["returncode"] != 0),
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
            report_path = output_dir / "execution_report.json"
            write_json_artifact(report_path, execution_report)
            artifact_paths["execution_report"] = _display_path(report_path)
        result = {
            "schema": "frontier_final_rate_attack_queue_builder_result.v1",
            "queue_id": queue_id,
            "output_dir": _display_path(output_dir),
            "artifacts": artifact_paths,
            "execute": bool(args.execute),
            "execution_report": execution_report,
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
    except (
        OSError,
        ArtifactWriteError,
        FrontierRateAttackBootstrapError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"FATAL: frontier final-rate attack bootstrap failed: {exc}", file=sys.stderr)
        return 2
    print(json_text(result), end="")
    if execution_report is not None and execution_report["failed_command_count"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
