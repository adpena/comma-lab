#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build LFV1 queue candidates and run sparse uint8 visibility probes.

This is the high-throughput pre-scorer filter for PR110-compatible LFV1
search. It materializes byte-closed archives without full raw inflate, probes
only selected frames, and writes compact cacheable JSON results.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)
SAFE_CANDIDATE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


@dataclass(frozen=True)
class SparseBatchResult:
    candidate_id: str
    returncode: int
    elapsed_seconds: float
    candidate_dir: str
    manifest_path: str | None
    visibility_path: str | None
    stdout_log: str
    stderr_log: str
    uint8_visible: bool | None = None
    changed_frame_count: int | None = None
    archive_delta_bytes: int | None = None
    error: str | None = None


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_candidate_id(candidate_id: str) -> str:
    if not SAFE_CANDIDATE_ID.fullmatch(candidate_id) or candidate_id in {".", ".."}:
        raise ValueError(f"unsafe candidate id: {candidate_id!r}")
    return candidate_id


def _candidate_dir(args: argparse.Namespace, candidate_id: str) -> Path:
    safe_id = _validate_candidate_id(candidate_id)
    root = (args.output_root / "candidates").resolve()
    path = (root / safe_id).resolve()
    if path.parent != root:
        raise ValueError(f"candidate path escaped output root: {path}")
    return path


def _run_logged(cmd: list[str], *, log_dir: Path, prefix: str, timeout: int) -> tuple[int, float, Path, Path]:
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_log = log_dir / f"{prefix}.stdout.log"
    stderr_log = log_dir / f"{prefix}.stderr.log"
    start = time.monotonic()
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    elapsed = time.monotonic() - start
    stdout_log.write_text(proc.stdout, encoding="utf-8")
    stderr_log.write_text(proc.stderr, encoding="utf-8")
    return proc.returncode, elapsed, stdout_log, stderr_log


def _candidate_rows(queue: dict[str, Any], *, max_candidates: int) -> list[dict[str, Any]]:
    rows = queue.get("candidates")
    if not isinstance(rows, list):
        raise SystemExit("queue JSON missing candidates list")
    out = [row for row in rows if isinstance(row, dict)]
    if max_candidates > 0:
        out = out[:max_candidates]
    return out


def _build_cmd(candidate: dict[str, Any], args: argparse.Namespace, candidate_id: str) -> list[str]:
    if candidate.get("family") != "lfv1_sparse_pair_micro_foveation":
        raise ValueError(f"unsupported candidate family: {candidate.get('family')!r}")
    params = candidate.get("params")
    if not isinstance(params, dict):
        raise ValueError(f"{candidate_id}: missing params")
    selected_pairs = candidate.get("selected_pairs")
    if not isinstance(selected_pairs, list) or not selected_pairs:
        raise ValueError(f"{candidate_id}: missing selected_pairs")
    max_sidecar_bytes = (
        candidate.get("archive_delta_budget", {}).get("max_archive_delta_bytes")
        if isinstance(candidate.get("archive_delta_budget"), dict)
        else None
    )
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "build_hfv1_sidecar_candidate.py"),
        "--runtime-dir",
        str(args.runtime_dir),
        "--output-root",
        str(args.output_root / "candidates"),
        "--candidate-id",
        candidate_id,
        "--pair-list",
        ",".join(str(int(pair)) for pair in selected_pairs),
        "--alpha",
        str(float(params["alpha"])),
        "--radius-scale",
        str(float(params["radius_scale"])),
        "--power",
        str(float(params["power"])),
        "--origin-y-frac",
        str(float(params["origin_y_frac"])),
        "--sidecar-format",
        "lfv1",
        "--lfv1-version",
        str(int(candidate.get("lfv1_version") or 2)),
        "--python-bin",
        str(args.python_bin),
        "--file-list-name",
        args.file_list_name,
        "--upstream-dir",
        str(args.upstream_dir),
    ]
    if max_sidecar_bytes is not None:
        cmd.extend(["--max-sidecar-bytes", str(int(max_sidecar_bytes))])
    return cmd


def _probe_cmd(candidate_id: str, args: argparse.Namespace) -> list[str]:
    candidate_dir = _candidate_dir(args, candidate_id)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "probe_lfv1_sparse_visibility.py"),
        "--candidate-dir",
        str(candidate_dir),
        "--runtime-dir",
        str(args.runtime_dir),
        "--baseline-raw",
        str(args.baseline_raw),
        "--output",
        str(args.output_root / "sparse_visibility" / candidate_id / "visibility.json"),
        "--device",
        args.device,
    ]
    if args.baseline_raw_sha256:
        cmd.extend(["--baseline-raw-sha256", str(args.baseline_raw_sha256)])
    if args.reuse_cache:
        cmd.append("--reuse-cache")
    return cmd


def _run_one(candidate: dict[str, Any], args: argparse.Namespace) -> SparseBatchResult:
    candidate_id = str(candidate.get("candidate_id") or "")
    if not candidate_id:
        raise ValueError("candidate row missing candidate_id")
    candidate_id = _validate_candidate_id(candidate_id)
    start = time.monotonic()
    log_dir = args.output_root / "logs" / candidate_id
    candidate_dir = _candidate_dir(args, candidate_id)
    manifest_path = candidate_dir / "manifest.json"
    visibility_path = args.output_root / "sparse_visibility" / candidate_id / "visibility.json"
    try:
        if not (args.reuse_cache and manifest_path.is_file()):
            build_rc, _build_elapsed, build_stdout, build_stderr = _run_logged(
                _build_cmd(candidate, args, candidate_id),
                log_dir=log_dir,
                prefix="build",
                timeout=args.build_timeout,
            )
            if build_rc != 0:
                return SparseBatchResult(
                    candidate_id=candidate_id,
                    returncode=build_rc,
                    elapsed_seconds=time.monotonic() - start,
                    candidate_dir=str(candidate_dir),
                    manifest_path=str(manifest_path) if manifest_path.is_file() else None,
                    visibility_path=None,
                    stdout_log=str(build_stdout),
                    stderr_log=str(build_stderr),
                    error="candidate_build_failed",
                )
        probe_rc, _probe_elapsed, probe_stdout, probe_stderr = _run_logged(
            _probe_cmd(candidate_id, args),
            log_dir=log_dir,
            prefix="sparse_visibility",
            timeout=args.probe_timeout,
        )
        if probe_rc != 0:
            return SparseBatchResult(
                candidate_id=candidate_id,
                returncode=probe_rc,
                elapsed_seconds=time.monotonic() - start,
                candidate_dir=str(candidate_dir),
                manifest_path=str(manifest_path) if manifest_path.is_file() else None,
                visibility_path=str(visibility_path) if visibility_path.is_file() else None,
                stdout_log=str(probe_stdout),
                stderr_log=str(probe_stderr),
                error="sparse_visibility_failed",
            )
        visibility = _read_json(visibility_path)
        manifest = _read_json(manifest_path)
        archive = manifest.get("archive") if isinstance(manifest.get("archive"), dict) else {}
        changed = visibility.get("changed_frame_indices")
        return SparseBatchResult(
            candidate_id=candidate_id,
            returncode=0,
            elapsed_seconds=time.monotonic() - start,
            candidate_dir=str(candidate_dir),
            manifest_path=str(manifest_path),
            visibility_path=str(visibility_path),
            stdout_log=str(probe_stdout),
            stderr_log=str(probe_stderr),
            uint8_visible=bool(visibility.get("uint8_visible")),
            changed_frame_count=len(changed) if isinstance(changed, list) else None,
            archive_delta_bytes=archive.get("delta_bytes_vs_source_archive"),
        )
    except Exception as exc:  # noqa: BLE001
        return SparseBatchResult(
            candidate_id=candidate_id,
            returncode=1,
            elapsed_seconds=time.monotonic() - start,
            candidate_dir=str(candidate_dir),
            manifest_path=str(manifest_path) if manifest_path.is_file() else None,
            visibility_path=str(visibility_path) if visibility_path.is_file() else None,
            stdout_log=str(log_dir / "exception.stdout.log"),
            stderr_log=str(log_dir / "exception.stderr.log"),
            error=f"{type(exc).__name__}: {exc}",
        )


def _hardware_summary() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--baseline-raw", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-candidates", type=int, default=24)
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--python-bin", type=Path, default=REPO_ROOT / ".venv" / "bin" / "python")
    parser.add_argument("--upstream-dir", type=Path, default=REPO_ROOT / "upstream")
    parser.add_argument("--file-list-name", default="0.mkv")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--build-timeout", type=int, default=120)
    parser.add_argument("--probe-timeout", type=int, default=120)
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    queue = _read_json(args.queue)
    if not isinstance(queue, dict):
        raise SystemExit("--queue must be a JSON object")
    rows = _candidate_rows(queue, max_candidates=args.max_candidates)
    args.baseline_raw_sha256 = _sha256_file(args.baseline_raw.resolve())
    manifest_path = args.output_root / "sparse_batch_manifest.json"
    planned = {
        "schema": "lfv1_sparse_visibility_batch_v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "queue": str(args.queue),
        "runtime_dir": str(args.runtime_dir),
        "baseline_raw": str(args.baseline_raw),
        "baseline_raw_sha256": args.baseline_raw_sha256,
        "output_root": str(args.output_root),
        "max_candidates": args.max_candidates,
        "max_parallel": args.max_parallel,
        "dry_run": bool(args.dry_run),
        "reuse_cache": bool(args.reuse_cache),
        "hardware": _hardware_summary(),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "candidate_ids": [str(row.get("candidate_id")) for row in rows],
    }
    if args.dry_run:
        _write_json(manifest_path, {**planned, "results": []})
        print(json.dumps({"dry_run": True, "manifest": str(manifest_path)}, indent=2, sort_keys=True))
        return 0

    results: list[SparseBatchResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.max_parallel)) as pool:
        futures = [pool.submit(_run_one, row, args) for row in rows]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                json.dumps(
                    {
                        "candidate_id": result.candidate_id,
                        "returncode": result.returncode,
                        "uint8_visible": result.uint8_visible,
                        "changed_frame_count": result.changed_frame_count,
                        "elapsed_seconds": result.elapsed_seconds,
                        "error": result.error,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    ranked = sorted(
        [result for result in results if result.returncode == 0],
        key=lambda item: (
            not bool(item.uint8_visible),
            int(item.archive_delta_bytes or 0),
            int(item.changed_frame_count or 0),
            item.candidate_id,
        ),
    )
    payload = {
        **planned,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "results": [result.__dict__ for result in sorted(results, key=lambda item: item.candidate_id)],
        "success_count": sum(1 for result in results if result.returncode == 0),
        "failure_count": sum(1 for result in results if result.returncode != 0),
        "visible_count": sum(1 for result in results if result.uint8_visible),
        "no_op_count": sum(1 for result in results if result.returncode == 0 and not result.uint8_visible),
        "first_visible": ranked[0].__dict__ if ranked and ranked[0].uint8_visible else None,
    }
    _write_json(manifest_path, payload)
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "success_count": payload["success_count"],
                "failure_count": payload["failure_count"],
                "visible_count": payload["visible_count"],
                "first_visible": payload["first_visible"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
