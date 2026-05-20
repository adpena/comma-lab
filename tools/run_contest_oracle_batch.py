#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize and optionally CPU-score a contest oracle candidate queue.

This is the local-Mac saturation runner for planning queues emitted by
``tools/plan_contest_oracle_search.py``.  It keeps the search layer canonical:

* candidate queue JSON in;
* byte-closed candidate archive + official inflate locality control out;
* optional cached CPU advisory scorer result out;
* no score claims, no promotion readiness, no GPU dispatch.

The first supported family is ``lfv1_sparse_pair_micro_foveation`` because it
uses the PR110-compatible LFV1 sidecar builder.
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
class BatchResult:
    candidate_id: str
    returncode: int
    elapsed_seconds: float
    candidate_dir: str
    manifest_path: str | None
    advisory_eval_path: str | None
    stdout_log: str
    stderr_log: str
    error: str | None = None
    cleanup: dict[str, Any] | None = None
    eval_skipped_reason: str | None = None


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
        "--run-inflate",
        "--baseline-raw",
        str(args.baseline_raw),
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


def _eval_cmd(candidate_id: str, args: argparse.Namespace) -> list[str]:
    candidate_dir = _candidate_dir(args, candidate_id)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "run_raw_advisory_eval.py"),
        "--raw",
        str(candidate_dir / "inflated" / "0.raw"),
        "--archive",
        str(candidate_dir / "archive" / "archive.zip"),
        "--output-dir",
        str(args.output_root / "advisory_raw_eval" / candidate_id),
        "--upstream-dir",
        str(args.upstream_dir),
        "--device",
        args.device,
        "--axis-label",
        args.axis_label,
        "--batch-size",
        str(args.batch_size),
        "--num-threads",
        str(args.num_threads),
        "--timeout",
        str(args.eval_timeout),
        "--reuse-cache",
    ]
    return cmd


def _cleanup_generated_raw(candidate_id: str, args: argparse.Namespace) -> dict[str, Any]:
    """Delete large generated raw outputs after their hashes are persisted."""
    candidate_dir = _candidate_dir(args, candidate_id)
    raw_path = candidate_dir / "inflated" / "0.raw"
    advisory_path = args.output_root / "advisory_raw_eval" / candidate_id / "raw_advisory_eval.json"
    manifest_path = candidate_dir / "manifest.json"
    cleanup: dict[str, Any] = {
        "requested": True,
        "raw_path": str(raw_path),
        "advisory_eval_path": str(advisory_path),
        "deleted": False,
        "skipped_reason": None,
    }
    if not advisory_path.is_file():
        cleanup["skipped_reason"] = "advisory_eval_json_missing"
        return cleanup
    if not raw_path.is_file():
        cleanup["skipped_reason"] = "raw_missing"
        return cleanup
    advisory = _read_json(advisory_path)
    manifest = _read_json(manifest_path) if manifest_path.is_file() else {}
    raw_meta = advisory.get("raw") if isinstance(advisory, dict) else None
    manifest_raw_meta = (
        manifest.get("official_inflate_control", {}).get("output_raw")
        if isinstance(manifest, dict)
        else None
    )
    expected_sha = raw_meta.get("sha256") if isinstance(raw_meta, dict) else None
    manifest_sha = (
        manifest_raw_meta.get("sha256") if isinstance(manifest_raw_meta, dict) else None
    )
    if not expected_sha or expected_sha != manifest_sha:
        cleanup["skipped_reason"] = "persisted_raw_hash_mismatch_or_missing"
        cleanup["advisory_raw_sha256"] = expected_sha
        cleanup["manifest_raw_sha256"] = manifest_sha
        return cleanup
    current_sha = _sha256_file(raw_path)
    if current_sha != expected_sha:
        cleanup["skipped_reason"] = "current_raw_hash_mismatch"
        cleanup["current_raw_sha256"] = current_sha
        cleanup["advisory_raw_sha256"] = expected_sha
        return cleanup
    cleanup["raw_bytes"] = raw_path.stat().st_size
    cleanup["raw_sha256"] = expected_sha
    raw_path.unlink()
    cleanup["deleted"] = True
    return cleanup


def _cleanup_generated_raw_after_build(candidate_id: str, args: argparse.Namespace) -> dict[str, Any]:
    """Delete generated raw after inflate locality metadata is persisted."""
    candidate_dir = _candidate_dir(args, candidate_id)
    raw_path = candidate_dir / "inflated" / "0.raw"
    manifest_path = candidate_dir / "manifest.json"
    cleanup: dict[str, Any] = {
        "requested": True,
        "stage": "after_build",
        "raw_path": str(raw_path),
        "manifest_path": str(manifest_path),
        "deleted": False,
        "skipped_reason": None,
    }
    if not manifest_path.is_file():
        cleanup["skipped_reason"] = "manifest_missing"
        return cleanup
    if not raw_path.is_file():
        cleanup["skipped_reason"] = "raw_missing"
        return cleanup
    manifest = _read_json(manifest_path)
    output_raw = (
        manifest.get("official_inflate_control", {}).get("output_raw")
        if isinstance(manifest, dict)
        else None
    )
    if not isinstance(output_raw, dict) or not output_raw.get("sha256"):
        cleanup["skipped_reason"] = "manifest_raw_hash_missing"
        return cleanup
    expected_sha = str(output_raw["sha256"])
    current_sha = _sha256_file(raw_path)
    if current_sha != expected_sha:
        cleanup["skipped_reason"] = "current_raw_hash_mismatch"
        cleanup["current_raw_sha256"] = current_sha
        cleanup["manifest_raw_sha256"] = expected_sha
        return cleanup
    cleanup["raw_bytes"] = raw_path.stat().st_size
    cleanup["raw_sha256"] = expected_sha
    raw_path.unlink()
    cleanup["deleted"] = True
    return cleanup


def _noop_eval_skip_reason(manifest_path: Path) -> str | None:
    if not manifest_path.is_file():
        return None
    manifest = _read_json(manifest_path)
    control = manifest.get("official_inflate_control") if isinstance(manifest, dict) else None
    if not isinstance(control, dict):
        return None
    comparison = control.get("raw_comparison")
    if not isinstance(comparison, dict):
        return None
    baseline_sha = comparison.get("baseline_raw_sha256")
    candidate_sha = comparison.get("candidate_raw_sha256")
    changed_frames = comparison.get("changed_frame_indices")
    if baseline_sha and candidate_sha and baseline_sha == candidate_sha:
        return "candidate_raw_sha256_matches_baseline"
    if isinstance(changed_frames, list) and len(changed_frames) == 0:
        return "changed_frame_indices_empty"
    return None


def _fatal_locality_skip_reason(manifest_path: Path) -> str | None:
    if not manifest_path.is_file():
        return None
    manifest = _read_json(manifest_path)
    control = manifest.get("official_inflate_control") if isinstance(manifest, dict) else None
    if not isinstance(control, dict):
        return None
    comparison = control.get("raw_comparison")
    if not isinstance(comparison, dict):
        return None
    unexpected = comparison.get("unexpected_changed_frame_indices")
    if isinstance(unexpected, list) and unexpected:
        return "unexpected_changed_frames_outside_selection"
    if comparison.get("changed_frames_within_selection") is False:
        return "changed_frames_outside_selection"
    return None


def _run_one(candidate: dict[str, Any], args: argparse.Namespace) -> BatchResult:
    candidate_id = str(candidate.get("candidate_id") or "")
    if not candidate_id:
        raise ValueError("candidate row missing candidate_id")
    candidate_id = _validate_candidate_id(candidate_id)
    start = time.monotonic()
    log_dir = args.output_root / "logs" / candidate_id
    candidate_dir = _candidate_dir(args, candidate_id)
    manifest_path = candidate_dir / "manifest.json"
    try:
        build_rc, _build_elapsed, build_stdout, build_stderr = _run_logged(
            _build_cmd(candidate, args, candidate_id),
            log_dir=log_dir,
            prefix="build",
            timeout=args.build_timeout,
        )
        if build_rc != 0:
            return BatchResult(
                candidate_id=candidate_id,
                returncode=build_rc,
                elapsed_seconds=time.monotonic() - start,
                candidate_dir=str(candidate_dir),
                manifest_path=str(manifest_path) if manifest_path.is_file() else None,
                advisory_eval_path=None,
                stdout_log=str(build_stdout),
                stderr_log=str(build_stderr),
                error="candidate_build_failed",
            )
        advisory_path: Path | None = None
        stdout_log = build_stdout
        stderr_log = build_stderr
        cleanup = None
        if args.delete_raw_after_build and not args.run_eval:
            cleanup = _cleanup_generated_raw_after_build(candidate_id, args)
        if args.run_eval:
            eval_skipped_reason = None if args.allow_noop_control_eval else _noop_eval_skip_reason(manifest_path)
            if eval_skipped_reason is None and not args.allow_failed_locality_control_eval:
                eval_skipped_reason = _fatal_locality_skip_reason(manifest_path)
            if args.skip_eval_if_noop:
                eval_skipped_reason = eval_skipped_reason or _noop_eval_skip_reason(manifest_path)
            if eval_skipped_reason is not None:
                if args.delete_raw_after_eval:
                    cleanup = _cleanup_generated_raw_after_build(candidate_id, args)
                    cleanup["stage"] = "after_build_noop_eval_skip"
                return BatchResult(
                    candidate_id=candidate_id,
                    returncode=0,
                    elapsed_seconds=time.monotonic() - start,
                    candidate_dir=str(candidate_dir),
                    manifest_path=str(manifest_path),
                    advisory_eval_path=None,
                    stdout_log=str(stdout_log),
                    stderr_log=str(stderr_log),
                    cleanup=cleanup,
                    eval_skipped_reason=eval_skipped_reason,
                )
            eval_rc, _eval_elapsed, eval_stdout, eval_stderr = _run_logged(
                _eval_cmd(candidate_id, args),
                log_dir=log_dir,
                prefix="eval",
                timeout=args.eval_timeout + 60,
            )
            stdout_log = eval_stdout
            stderr_log = eval_stderr
            advisory_path = args.output_root / "advisory_raw_eval" / candidate_id / "raw_advisory_eval.json"
            if eval_rc != 0:
                return BatchResult(
                    candidate_id=candidate_id,
                    returncode=eval_rc,
                    elapsed_seconds=time.monotonic() - start,
                    candidate_dir=str(candidate_dir),
                    manifest_path=str(manifest_path),
                    advisory_eval_path=str(advisory_path) if advisory_path.is_file() else None,
                    stdout_log=str(stdout_log),
                    stderr_log=str(stderr_log),
                    error="advisory_eval_failed",
                )
        if args.delete_raw_after_eval:
            if not args.run_eval:
                cleanup = {
                    "requested": True,
                    "stage": "after_eval",
                    "deleted": False,
                    "skipped_reason": "run_eval_false",
                }
            elif advisory_path is not None:
                cleanup = _cleanup_generated_raw(candidate_id, args)
        return BatchResult(
            candidate_id=candidate_id,
            returncode=0,
            elapsed_seconds=time.monotonic() - start,
            candidate_dir=str(candidate_dir),
            manifest_path=str(manifest_path),
            advisory_eval_path=str(advisory_path) if advisory_path else None,
            stdout_log=str(stdout_log),
            stderr_log=str(stderr_log),
            cleanup=cleanup,
        )
    except Exception as exc:  # noqa: BLE001 - batch rows must return structured failures
        return BatchResult(
            candidate_id=candidate_id,
            returncode=1,
            elapsed_seconds=time.monotonic() - start,
            candidate_dir=str(candidate_dir),
            manifest_path=str(manifest_path) if manifest_path.is_file() else None,
            advisory_eval_path=None,
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
        "memory_bytes_env_hint": None,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--baseline-raw", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--max-candidates", type=int, default=4)
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument(
        "--skip-eval-if-noop",
        action="store_true",
        help=(
            "Deprecated compatibility flag. No-op rows are skipped by default; "
            "use --allow-noop-control-eval to force advisory scoring."
        ),
    )
    parser.add_argument(
        "--allow-noop-control-eval",
        action="store_true",
        help="Allow advisory eval when full inflate proves the candidate raw equals baseline.",
    )
    parser.add_argument(
        "--allow-failed-locality-control-eval",
        action="store_true",
        help="Allow advisory eval when raw locality control failed for a non-noop reason.",
    )
    parser.add_argument("--python-bin", type=Path, default=REPO_ROOT / ".venv" / "bin" / "python")
    parser.add_argument("--upstream-dir", type=Path, default=REPO_ROOT / "upstream")
    parser.add_argument("--file-list-name", default="0.mkv")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--axis-label", default="[macOS-CPU advisory]")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--build-timeout", type=int, default=600)
    parser.add_argument("--eval-timeout", type=int, default=1800)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--delete-raw-after-eval",
        action="store_true",
        help=(
            "Delete each generated candidate inflated/0.raw after a successful advisory "
            "eval, but only after matching raw SHA-256 is present in both candidate "
            "manifest and raw_advisory_eval.json."
        ),
    )
    parser.add_argument(
        "--delete-raw-after-build",
        action="store_true",
        help=(
            "Delete each generated candidate inflated/0.raw after build/inflate "
            "metadata is persisted. Use for materialization-only no-op and "
            "visibility sweeps; incompatible with --run-eval."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.run_eval and args.axis_label.startswith("[contest-"):
        raise SystemExit(
            "run_contest_oracle_batch.py is an advisory raw-output runner; "
            "use an advisory axis label, not [contest-*]"
        )
    if args.delete_raw_after_build and args.run_eval:
        raise SystemExit("--delete-raw-after-build is for materialization-only sweeps; use --delete-raw-after-eval with --run-eval")
    queue = _read_json(args.queue)
    if not isinstance(queue, dict):
        raise SystemExit("--queue must be a JSON object")
    rows = _candidate_rows(queue, max_candidates=args.max_candidates)
    manifest_path = args.output_root / "batch_manifest.json"
    planned = {
        "schema": "contest_oracle_batch_run_v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "queue": str(args.queue),
        "runtime_dir": str(args.runtime_dir),
        "baseline_raw": str(args.baseline_raw),
        "output_root": str(args.output_root),
        "max_candidates": args.max_candidates,
        "max_parallel": args.max_parallel,
        "run_eval": bool(args.run_eval),
        "skip_eval_if_noop": bool(args.skip_eval_if_noop),
        "allow_noop_control_eval": bool(args.allow_noop_control_eval),
        "allow_failed_locality_control_eval": bool(args.allow_failed_locality_control_eval),
        "dry_run": bool(args.dry_run),
        "delete_raw_after_eval": bool(args.delete_raw_after_eval),
        "delete_raw_after_build": bool(args.delete_raw_after_build),
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

    results: list[BatchResult] = []
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
                        "error": result.error,
                        "eval_skipped_reason": result.eval_skipped_reason,
                        "elapsed_seconds": result.elapsed_seconds,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    payload = {
        **planned,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "results": [result.__dict__ for result in sorted(results, key=lambda item: item.candidate_id)],
        "success_count": sum(1 for result in results if result.returncode == 0),
        "failure_count": sum(1 for result in results if result.returncode != 0),
    }
    _write_json(manifest_path, payload)
    print(json.dumps({"manifest": str(manifest_path), "success_count": payload["success_count"], "failure_count": payload["failure_count"]}, indent=2, sort_keys=True))
    return 0 if payload["failure_count"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
