#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run bounded macOS advisory component response for decoder-q candidates."""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
import os
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tools.run_decoder_q_candidate_inflate_controls import (  # noqa: E402
    _compare_raws,
    _extract_single_member,
    _run_inflate,
    _sha256_file,
    _write_json,
)


def _candidate_dirs(candidate_root: Path, candidate_ids: list[str], max_candidates: int | None) -> list[Path]:
    if candidate_ids:
        rows = [candidate_root / candidate_id for candidate_id in candidate_ids]
    else:
        rows = [
            path
            for path in sorted(candidate_root.iterdir())
            if path.is_dir() and (path / "archive.zip").is_file()
        ]
    for path in rows:
        if not (path / "archive.zip").is_file():
            raise SystemExit(f"candidate archive missing: {path / 'archive.zip'}")
    if max_candidates is not None:
        rows = rows[: int(max_candidates)]
    return rows


def _run_raw_advisory_eval(
    *,
    raw: Path,
    archive: Path,
    output_dir: Path,
    upstream_dir: Path,
    video_names_file: Path,
    file_list_name: str,
    device: str,
    axis_label: str,
    batch_size: int,
    num_threads: int,
    timeout: int,
) -> dict[str, Any]:
    env = {
        **dict(os.environ),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "run_raw_advisory_eval.py"),
        "--raw",
        str(raw),
        "--archive",
        str(archive),
        "--output-dir",
        str(output_dir),
        "--upstream-dir",
        str(upstream_dir),
        "--video-names-file",
        str(video_names_file),
        "--file-list-name",
        file_list_name,
        "--device",
        device,
        "--axis-label",
        axis_label,
        "--batch-size",
        str(batch_size),
        "--num-threads",
        str(num_threads),
        "--timeout",
        str(timeout),
    ]
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout + 120,
        check=False,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "advisory_wrapper.stdout.log").write_text(proc.stdout, encoding="utf-8")
    (output_dir / "advisory_wrapper.stderr.log").write_text(proc.stderr, encoding="utf-8")
    payload_path = output_dir / "raw_advisory_eval.json"
    payload = json.loads(payload_path.read_text(encoding="utf-8")) if payload_path.is_file() else {}
    payload.setdefault("wrapper_returncode", proc.returncode)
    payload.setdefault("wrapper_stdout_log", str(output_dir / "advisory_wrapper.stdout.log"))
    payload.setdefault("wrapper_stderr_log", str(output_dir / "advisory_wrapper.stderr.log"))
    return payload


def _run_one(candidate_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    runtime_dir = args.runtime_dir.resolve()
    baseline_raw = args.baseline_raw.resolve()
    archive_zip = candidate_dir / "archive.zip"
    out_dir = output_root / "candidates" / candidate_dir.name
    file_list = output_root / "file_list.txt"
    frame_bytes = int(args.frame_height) * int(args.frame_width) * 3

    blockers = [
        "raw_eval_advisory_not_full_archive_inflate_custody",
        "not_contest_cuda",
        "exact_cuda_auth_eval_missing",
    ]
    extract = _extract_single_member(archive_zip, out_dir / "data_dir")
    inflate = _run_inflate(
        runtime_dir=runtime_dir,
        data_dir=out_dir / "data_dir",
        output_dir=out_dir / "inflated",
        file_list=file_list,
        python_bin=args.python_bin,
        timeout=args.inflate_timeout,
    )
    (out_dir / "inflate.stdout.log").write_text(inflate["stdout"], encoding="utf-8")
    (out_dir / "inflate.stderr.log").write_text(inflate["stderr"], encoding="utf-8")
    raw_path = out_dir / "inflated" / Path(args.file_list_name).with_suffix(".raw")
    comparison = None
    advisory = None
    if inflate["returncode"] != 0 or not raw_path.is_file():
        blockers.append("official_inflate_failed")
    else:
        comparison = _compare_raws(
            baseline_raw=baseline_raw,
            candidate_raw=raw_path,
            frame_bytes=frame_bytes,
            sample_limit=args.sample_limit,
        )
        if not comparison["passed_visible_change"]:
            blockers.append("no_visible_raw_change")
        else:
            advisory = _run_raw_advisory_eval(
                raw=raw_path,
                archive=archive_zip,
                output_dir=out_dir / "advisory_eval",
                upstream_dir=args.upstream_dir.resolve(),
                video_names_file=args.video_names_file.resolve(),
                file_list_name=args.file_list_name,
                device=args.device,
                axis_label=args.axis_label,
                batch_size=args.batch_size,
                num_threads=args.num_threads,
                timeout=args.eval_timeout,
            )
            if advisory.get("returncode") != 0:
                blockers.append("upstream_evaluate_failed")
    cleanup = {"candidate_raw_deleted": False}
    advisory_success = bool(isinstance(advisory, dict) and advisory.get("returncode") == 0)
    if raw_path.is_file() and args.cleanup_candidate_raw and advisory_success:
        cleanup = {
            "candidate_raw_deleted": True,
            "candidate_raw_sha256_before_delete": _sha256_file(raw_path),
            "candidate_raw_bytes_before_delete": raw_path.stat().st_size,
        }
        raw_path.unlink()
    elif raw_path.is_file() and args.cleanup_candidate_raw and not advisory_success:
        cleanup = {
            "candidate_raw_deleted": False,
            "reason": "advisory_not_successful_keep_raw_for_debug",
            "candidate_raw_sha256": _sha256_file(raw_path),
            "candidate_raw_bytes": raw_path.stat().st_size,
        }
    if args.cleanup_extracted_member and (out_dir / "data_dir").exists():
        shutil.rmtree(out_dir / "data_dir")
        cleanup["extracted_member_deleted"] = True

    mutation_manifest_path = candidate_dir / "mutation_manifest.json"
    mutation_manifest = (
        json.loads(mutation_manifest_path.read_text(encoding="utf-8"))
        if mutation_manifest_path.is_file()
        else None
    )
    row = {
        "candidate_id": candidate_dir.name,
        "source_candidate_dir": str(candidate_dir.resolve()),
        "mutation_manifest": mutation_manifest,
        "archive_extract": extract,
        "inflate": {
            "cmd": inflate["cmd"],
            "returncode": inflate["returncode"],
            "elapsed_seconds": inflate["elapsed_seconds"],
            "stdout_log": str((out_dir / "inflate.stdout.log").resolve()),
            "stderr_log": str((out_dir / "inflate.stderr.log").resolve()),
            "output_raw_path": str(raw_path.resolve()),
            "output_raw_exists_after_cleanup": raw_path.is_file(),
        },
        "raw_comparison": comparison,
        "advisory_eval": advisory,
        "cleanup": cleanup,
        "blockers": blockers,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    _write_json(out_dir / "advisory_candidate_manifest.json", row)
    return row


def run_batch(args: argparse.Namespace) -> dict[str, Any]:
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    file_list = output_root / "file_list.txt"
    file_list.write_text(args.file_list_name + "\n", encoding="utf-8")
    candidates = _candidate_dirs(
        args.candidate_root.resolve(),
        args.candidate_id,
        args.max_candidates,
    )
    jobs = max(1, int(args.jobs))
    if jobs > 1 and not args.allow_parallel_scorer:
        raise SystemExit(
            "--jobs > 1 launches parallel upstream/evaluate.py scorer subprocesses; "
            "pass --allow-parallel-scorer to acknowledge CPU/RAM/disk pressure"
        )
    if jobs == 1:
        rows = [_run_one(candidate_dir, args) for candidate_dir in candidates]
    else:
        with futures.ThreadPoolExecutor(max_workers=jobs) as pool:
            rows = list(pool.map(lambda candidate_dir: _run_one(candidate_dir, args), candidates))

    successful = [
        row
        for row in rows
        if isinstance(row.get("advisory_eval"), dict) and row["advisory_eval"].get("returncode") == 0
    ]
    baseline_score = args.baseline_score
    for row in successful:
        score = float(row["advisory_eval"]["canonical_score"])
        row["delta_vs_baseline_score"] = score - baseline_score if baseline_score is not None else None
    best = None
    if successful:
        best = min(successful, key=lambda row: float(row["advisory_eval"]["canonical_score"]))

    return {
        "schema": "fec6_decoder_q_candidate_advisory_batch_v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "producer": "tools/run_decoder_q_candidate_advisory_batch.py",
        "inputs": {
            "runtime_dir": str(args.runtime_dir.resolve()),
            "candidate_root": str(args.candidate_root.resolve()),
            "baseline_raw": str(args.baseline_raw.resolve()),
            "baseline_raw_sha256": _sha256_file(args.baseline_raw.resolve()),
            "baseline_score": baseline_score,
            "jobs": jobs,
            "device": args.device,
            "axis_label": args.axis_label,
            "batch_size": args.batch_size,
            "num_threads": args.num_threads,
            "max_candidates": args.max_candidates,
            "candidate_id": args.candidate_id,
        },
        "summary": {
            "candidate_count": len(rows),
            "advisory_success_count": len(successful),
            "visible_change_count": sum(
                1
                for row in rows
                if isinstance(row.get("raw_comparison"), dict)
                and row["raw_comparison"].get("passed_visible_change")
            ),
            "best_candidate_id": best["candidate_id"] if best else None,
            "best_score": best["advisory_eval"]["canonical_score"] if best else None,
            "best_delta_vs_baseline_score": best.get("delta_vs_baseline_score") if best else None,
        },
        "candidates": rows,
        "authority": {
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "notes": "macOS raw-output advisory component response; exact contest eval still required.",
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--baseline-raw", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--upstream-dir", type=Path, default=REPO_ROOT / "upstream")
    parser.add_argument("--video-names-file", type=Path, default=REPO_ROOT / "upstream" / "public_test_video_names.txt")
    parser.add_argument("--candidate-id", action="append", default=[])
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--file-list-name", default="0.hevc")
    parser.add_argument("--frame-height", type=int, default=874)
    parser.add_argument("--frame-width", type=int, default=1164)
    parser.add_argument("--python-bin", default=".venv/bin/python")
    parser.add_argument("--inflate-timeout", type=int, default=900)
    parser.add_argument("--eval-timeout", type=int, default=1800)
    parser.add_argument("--sample-limit", type=int, default=16)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--axis-label", default="[macOS-CPU advisory decoder-q]")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--baseline-score", type=float, default=None)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--allow-parallel-scorer", action="store_true")
    parser.add_argument("--cleanup-candidate-raw", action="store_true")
    parser.add_argument("--cleanup-extracted-member", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = run_batch(args)
    _write_json(args.output, payload)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "candidate_count": payload["summary"]["candidate_count"],
                "advisory_success_count": payload["summary"]["advisory_success_count"],
                "visible_change_count": payload["summary"]["visible_change_count"],
                "best_candidate_id": payload["summary"]["best_candidate_id"],
                "best_score": payload["summary"]["best_score"],
                "best_delta_vs_baseline_score": payload["summary"]["best_delta_vs_baseline_score"],
                "score_claim": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
