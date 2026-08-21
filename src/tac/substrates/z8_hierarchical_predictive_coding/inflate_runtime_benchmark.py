# SPDX-License-Identifier: MIT
"""Full ``inflate.sh`` runtime benchmark for Z8 archive candidates.

The wavelet-blob parser benchmark is useful for codec hot spots, but it does
not exercise the receiver boundary.  This module times the actual contest
runtime contract:

    inflate.sh <archive_dir> <output_dir> <file_list>

It is deliberately advisory-only.  The report measures decode/runtime pressure
and output custody on the local machine, but it is not an auth-eval score and
never grants promotion authority.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from tac.process_group_kill import run_in_process_group

NON_PROMOTABLE_MARKERS: dict[str, Any] = {
    "evidence_grade": "macOS-CPU-advisory",
    "axis_tag": "[macOS-CPU advisory]",
    "score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "promotable": False,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_file_list(file_list_path: Path) -> list[str]:
    names = [
        line.strip()
        for line in file_list_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not names:
        raise ValueError(f"file_list is empty: {file_list_path}")
    return names


def manifest_tree(root: Path) -> dict[str, Any]:
    files: list[dict[str, Any]] = []
    total_bytes = 0
    if root.exists():
        for path in sorted(p for p in root.rglob("*") if p.is_file()):
            size = path.stat().st_size
            total_bytes += size
            files.append(
                {
                    "rel_path": path.relative_to(root).as_posix(),
                    "bytes": int(size),
                    "sha256": sha256_file(path),
                }
            )
    tree_hash = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "file_count": len(files),
        "total_bytes": int(total_bytes),
        "tree_sha256": tree_hash,
        "files": files,
    }


def _text_tail(value: str | None, *, limit: int = 4000) -> str:
    if not value:
        return ""
    if len(value) <= limit:
        return value
    return value[-limit:]


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def _require_empty_or_missing(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(
            f"benchmark output directory must be empty or missing: {path}"
        )


def benchmark_z8_submission_inflate_runtime(
    *,
    inflate_sh: str | Path,
    archive_dir: str | Path,
    file_list: str | Path,
    output_dir: str | Path,
    repeat: int = 1,
    timeout_seconds: float = 1800.0,
    auth_eval_window_seconds: float = 1800.0,
    inflate_device: str = "cpu",
    env: Mapping[str, str] | None = None,
    retain_output: bool = False,
) -> dict[str, Any]:
    """Run the full Z8 receiver shell and return an advisory timing report."""

    if repeat <= 0:
        raise ValueError("repeat must be positive")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    inflate_sh_path = Path(inflate_sh)
    archive_dir_path = Path(archive_dir)
    file_list_path = Path(file_list)
    output_root = Path(output_dir)
    _require_file(inflate_sh_path, "inflate.sh")
    if not archive_dir_path.is_dir():
        raise FileNotFoundError(f"archive_dir not found: {archive_dir_path}")
    _require_file(file_list_path, "file_list")
    _require_empty_or_missing(output_root)
    names = read_file_list(file_list_path)

    base_env = dict(os.environ if env is None else env)
    if inflate_device:
        base_env["PACT_INFLATE_DEVICE"] = inflate_device

    runs: list[dict[str, Any]] = []
    blockers: list[str] = [
        "auth_evaluator_not_run",
        "contest_cpu_cuda_score_not_measured",
    ]
    output_root.mkdir(parents=True, exist_ok=True)
    for index in range(repeat):
        run_output_dir = output_root / f"run_{index:03d}"
        _require_empty_or_missing(run_output_dir)
        run_output_dir.mkdir(parents=True, exist_ok=False)
        argv = [
            "bash",
            str(inflate_sh_path),
            str(archive_dir_path),
            str(run_output_dir),
            str(file_list_path),
        ]
        start = time.perf_counter()
        timed_out = False
        try:
            proc = run_in_process_group(
                argv,
                check=False,
                capture_output=True,
                env=base_env,
                text=True,
                timeout=float(timeout_seconds),
            )
            returncode: int | None = int(proc.returncode)
            stdout_tail = _text_tail(proc.stdout)
            stderr_tail = _text_tail(proc.stderr)
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            returncode = None
            stdout_tail = _text_tail(
                exc.stdout.decode("utf-8", errors="replace")
                if isinstance(exc.stdout, bytes)
                else exc.stdout
            )
            stderr_tail = _text_tail(
                exc.stderr.decode("utf-8", errors="replace")
                if isinstance(exc.stderr, bytes)
                else exc.stderr
            )
        elapsed = time.perf_counter() - start
        output_manifest = manifest_tree(run_output_dir)
        output_cleanup_blocker: str | None = None
        if not retain_output:
            try:
                shutil.rmtree(run_output_dir)
            except OSError as exc:
                output_cleanup_blocker = f"inflate_output_cleanup_failed:{exc}"
                if "inflate_output_cleanup_failed" not in blockers:
                    blockers.append("inflate_output_cleanup_failed")
        if timed_out and "inflate_sh_timed_out" not in blockers:
            blockers.append("inflate_sh_timed_out")
        if returncode not in {0} and "inflate_sh_returned_nonzero" not in blockers:
            blockers.append("inflate_sh_returned_nonzero")
        if (
            output_manifest["file_count"] < len(names)
            and "inflate_output_count_below_file_list_count" not in blockers
        ):
            blockers.append("inflate_output_count_below_file_list_count")
        runs.append(
            {
                "run_index": int(index),
                "argv": argv,
                "elapsed_seconds": float(elapsed),
                "returncode": returncode,
                "timed_out": timed_out,
                "stdout_tail": stdout_tail,
                "stderr_tail": stderr_tail,
                "output_dir": run_output_dir.as_posix(),
                "output_manifest": output_manifest,
                "output_retained": bool(retain_output),
                "output_cleanup_blocker": output_cleanup_blocker,
            }
        )

    successful = [
        run["elapsed_seconds"]
        for run in runs
        if run["returncode"] == 0 and run["timed_out"] is False
    ]
    best = min(successful) if successful else None
    mean = sum(successful) / len(successful) if successful else None
    return {
        "schema": "z8_submission_inflate_runtime_benchmark.v1",
        "purpose": (
            "Full receiver-runtime benchmark for Z8 codec candidates through "
            "the contest inflate.sh contract."
        ),
        **NON_PROMOTABLE_MARKERS,
        "benchmark_scope": "full_submission_inflate_sh_runtime",
        "receiver_path_exercised": True,
        "inflate_sh": inflate_sh_path.as_posix(),
        "archive_dir": archive_dir_path.as_posix(),
        "archive_member_manifest": manifest_tree(archive_dir_path),
        "file_list": file_list_path.as_posix(),
        "file_list_entries": names,
        "output_root": output_root.as_posix(),
        "repeat": int(repeat),
        "timeout_seconds": float(timeout_seconds),
        "auth_eval_window_seconds": float(auth_eval_window_seconds),
        "inflate_device": inflate_device,
        "output_retention_policy": (
            "retained_by_request" if retain_output else "manifest_then_delete"
        ),
        "large_artifact_cleanup_default": True,
        "successful_runs": len(successful),
        "inflate_seconds_best": float(best) if best is not None else None,
        "inflate_seconds_mean": float(mean) if mean is not None else None,
        "auth_eval_window_fraction_best": (
            float(best / auth_eval_window_seconds)
            if best is not None and auth_eval_window_seconds > 0
            else None
        ),
        "runs": runs,
        "blockers": blockers,
    }


__all__ = [
    "NON_PROMOTABLE_MARKERS",
    "benchmark_z8_submission_inflate_runtime",
    "manifest_tree",
    "read_file_list",
    "sha256_file",
]
