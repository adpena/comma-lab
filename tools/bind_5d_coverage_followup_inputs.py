#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Bind 5D coverage follow-up requests to custody-checked local inputs."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import tempfile
from pathlib import Path

from comma_lab.scheduler.experiment_queue import ExperimentQueueError
from comma_lab.scheduler.pair_frame_5d_coverage_acquisition_queue import (
    build_coverage_followup_input_binding_report,
    build_coverage_followup_readiness_report,
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", action="append", type=Path, default=[])
    parser.add_argument("--plans-dir", type=Path, default=None)
    parser.add_argument("--search-root", action="append", type=Path, default=[])
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--refreshed-readiness-output", type=Path, default=None)
    parser.add_argument("--submission-bundle", type=Path, default=None)
    parser.add_argument("--reference-mlx-cache-dir", type=Path, default=None)
    parser.add_argument("--candidate-mlx-cache-dir", type=Path, default=None)
    parser.add_argument("--archive-size-bytes", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _resolve(path: Path, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def _collect_plan_paths(
    *,
    explicit_plans: list[Path],
    plans_dir: Path | None,
    repo_root: Path,
) -> list[Path]:
    plan_paths = [_resolve(path, repo_root) for path in explicit_plans]
    if plans_dir is not None:
        resolved_dir = _resolve(plans_dir, repo_root)
        if not resolved_dir.is_dir():
            raise ExperimentQueueError(f"plans-dir not found: {resolved_dir}")
        dir_plans = sorted(resolved_dir.glob("*_acquisition_plan.json"))
        plan_paths.extend(dir_plans or sorted(resolved_dir.glob("*.json")))
    deduped: list[Path] = []
    seen: set[Path] = set()
    for path in plan_paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(path)
    if not deduped:
        raise ExperimentQueueError("no acquisition plan paths provided")
    return deduped


def _write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp_name)
        raise


def _select_arg(
    explicit: Path | None,
    selected_inputs: dict[str, object],
    key: str,
    *,
    repo_root: Path,
) -> Path | None:
    if explicit is not None:
        return explicit
    value = selected_inputs.get(key)
    if isinstance(value, str) and value:
        return _resolve(Path(value), repo_root)
    return None


def _select_archive_size(
    explicit: int | None,
    selected_inputs: dict[str, object],
) -> int | None:
    if explicit is not None:
        return explicit
    value = selected_inputs.get("archive_size_bytes")
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    repo_root = _repo_root()
    output_path = _resolve(args.output, repo_root)
    refreshed_path = (
        None
        if args.refreshed_readiness_output is None
        else _resolve(args.refreshed_readiness_output, repo_root)
    )
    for path in (output_path, refreshed_path):
        if path is not None and path.exists() and not args.overwrite:
            raise SystemExit(
                f"[bind_5d_coverage_followup_inputs] FATAL: output exists: {path}"
            )
    try:
        plan_paths = _collect_plan_paths(
            explicit_plans=args.plan,
            plans_dir=args.plans_dir,
            repo_root=repo_root,
        )
        report = build_coverage_followup_input_binding_report(
            repo_root=repo_root,
            plan_paths=plan_paths,
            search_roots=args.search_root,
            submission_bundle_path=args.submission_bundle,
            reference_mlx_cache_dir=args.reference_mlx_cache_dir,
            candidate_mlx_cache_dir=args.candidate_mlx_cache_dir,
            archive_size_bytes=args.archive_size_bytes,
        )
        if refreshed_path is not None:
            selected_inputs = report.get("selected_inputs")
            if not isinstance(selected_inputs, dict):
                selected_inputs = {}
            readiness = build_coverage_followup_readiness_report(
                repo_root=repo_root,
                plan_paths=plan_paths,
                submission_bundle_path=_select_arg(
                    args.submission_bundle,
                    selected_inputs,
                    "submission_bundle_path",
                    repo_root=repo_root,
                ),
                reference_mlx_cache_dir=_select_arg(
                    args.reference_mlx_cache_dir,
                    selected_inputs,
                    "reference_mlx_cache_dir",
                    repo_root=repo_root,
                ),
                candidate_mlx_cache_dir=_select_arg(
                    args.candidate_mlx_cache_dir,
                    selected_inputs,
                    "candidate_mlx_cache_dir",
                    repo_root=repo_root,
                ),
                archive_size_bytes=_select_archive_size(
                    args.archive_size_bytes,
                    selected_inputs,
                ),
            )
            _write_text_atomic(
                refreshed_path,
                json.dumps(readiness, indent=2, sort_keys=True) + "\n",
            )
    except (OSError, json.JSONDecodeError, ExperimentQueueError) as exc:
        print(f"[bind_5d_coverage_followup_inputs] FATAL: {exc}")
        return 2
    _write_text_atomic(output_path, json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(
        "[bind_5d_coverage_followup_inputs] OK: "
        f"{report['bound_request_count']} bound, "
        f"{report['blocked_request_count']} blocked -> {output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
