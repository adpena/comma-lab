#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run multiple MLX scorer responses from caches in one scorer process."""

from __future__ import annotations

import argparse
import json
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

from tac.local_acceleration.mlx_scorer_response import (  # noqa: E402
    MLXScorerResponseBatchJob,
    build_mlx_scorer_response_payload_batch,
    write_mlx_scorer_response_payload,
)
from tac.optimization.proxy_candidate_contract import require_no_truthy_authority_fields  # noqa: E402
from tac.repo_io import ArtifactWriteError, json_text, write_json_artifact  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-cache-dir", required=True, type=Path)
    parser.add_argument(
        "--job-json",
        action="append",
        default=[],
        help=(
            "JSON object with candidate_cache_dir, output, archive or "
            "archive_size_bytes, optional components_dir and response_family. May repeat."
        ),
    )
    parser.add_argument("--summary-out", required=True, type=Path)
    parser.add_argument("--repo-root", default=Path("."), type=Path)
    parser.add_argument(
        "--upstream-dir",
        type=Path,
        help=(
            "Explicit upstream scorer snapshot directory. Defaults to "
            "<repo-root>/upstream. Use this for SSD worktrees whose code checkout "
            "is separate from the canonical upstream snapshot."
        ),
    )
    parser.add_argument("--batch-pairs", type=int, default=1)
    parser.add_argument("--start-pair", type=int, default=0)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument("--device", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--allow-gpu-research-signal", action="store_true")
    parser.add_argument("--allow-batch-shape-research-signal", action="store_true")
    parser.add_argument("--allow-unaudited-candidate-cache-debug", action="store_true")
    parser.add_argument("--allow-local-cpu-advisory-cache-identity", action="store_true")
    parser.add_argument(
        "--cache-integrity-mode",
        choices=("strict", "manifest"),
        default="manifest",
        help=(
            "manifest avoids hot-path multi-GB cache rehashing by trusting "
            "recorded cache-manifest hashes and size-checking files; use "
            "strict for authority/parity probes."
        ),
    )
    parser.add_argument("--progress-every", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def _job_from_json(raw: str, *, repo_root: Path) -> MLXScorerResponseBatchJob:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--job-json must be a JSON object: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("--job-json must be a JSON object")
    candidate_cache_dir = _required_path(payload, "candidate_cache_dir")
    output = _required_path(payload, "output")
    archive_size_bytes = payload.get("archive_size_bytes")
    archive = payload.get("archive")
    if archive_size_bytes is None:
        if not isinstance(archive, str) or not archive.strip():
            raise ValueError("job requires archive_size_bytes or archive")
        archive_path = _resolve(Path(archive), repo_root=repo_root)
        archive_size_bytes = archive_path.stat().st_size
    components_dir = payload.get("components_dir")
    response_family = payload.get("response_family")
    if response_family is not None and not isinstance(response_family, str):
        raise ValueError("job response_family must be a string when provided")
    return MLXScorerResponseBatchJob(
        candidate_cache_dir=Path(candidate_cache_dir),
        archive_size_bytes=int(archive_size_bytes),
        output=Path(output),
        components_dir=Path(components_dir) if isinstance(components_dir, str) and components_dir else None,
        response_family=response_family,
    )


def _required_path(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"job requires non-empty {key}")
    return value


def _resolve(path: Path, *, repo_root: Path) -> Path:
    return path if path.is_absolute() else repo_root / path


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root if args.repo_root.is_absolute() else REPO_ROOT / args.repo_root
    try:
        jobs = [_job_from_json(raw, repo_root=repo_root) for raw in args.job_json]
        started = time.time()
        payloads = build_mlx_scorer_response_payload_batch(
            reference_cache_dir=args.reference_cache_dir,
            jobs=jobs,
            repo_root=args.repo_root,
            upstream_dir=args.upstream_dir,
            batch_pairs=args.batch_pairs,
            device_type=args.device,
            progress_every=args.progress_every,
            start_pair=args.start_pair,
            max_pairs=args.max_pairs,
            allow_gpu_research_signal=args.allow_gpu_research_signal,
            allow_batch_shape_research_signal=args.allow_batch_shape_research_signal,
            allow_unaudited_candidate_cache_debug=(
                args.allow_unaudited_candidate_cache_debug
            ),
            allow_local_cpu_advisory_cache_identity=(
                args.allow_local_cpu_advisory_cache_identity
            ),
            cache_integrity_mode=args.cache_integrity_mode,
        )
        rows: list[dict[str, Any]] = []
        for job, payload in zip(jobs, payloads, strict=True):
            output = Path(job.output)
            if output.exists() and not args.overwrite:
                raise FileExistsError(f"output exists; pass --overwrite: {output}")
            write_mlx_scorer_response_payload(payload, output)
            require_no_truthy_authority_fields(
                payload,
                context=f"mlx_scorer_response_batch:{output}",
            )
            rows.append(
                {
                    "schema": "mlx_scorer_response_batch_row.v1",
                    "output": str(output),
                    "candidate_cache_dir": str(job.candidate_cache_dir),
                    "canonical_score": payload["canonical_score"],
                    "n_samples": payload["n_samples"],
                    "elapsed_seconds": payload["elapsed_seconds"],
                    "batch_pairs": payload["batch_pairs"],
                    "device": args.device,
                    "score_claim": False,
                    "promotion_eligible": False,
                    "rank_or_kill_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                }
            )
        summary = {
            "schema": "mlx_scorer_response_batch_run.v1",
            "job_count": len(rows),
            "rows": rows,
            "wall_seconds": time.time() - started,
            "batch_pairs": int(args.batch_pairs),
            "device": args.device,
            "cache_integrity_mode": args.cache_integrity_mode,
            "score_claim": False,
            "score_claim_valid": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "candidate_generation_only": True,
            "allowed_use": "local_mlx_research_signal_batch_execution_only",
        }
        write = write_json_artifact(
            args.summary_out,
            summary,
            allow_overwrite=bool(args.overwrite),
        )
    except (
        ArtifactWriteError,
        FileExistsError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ) as exc:
        print(f"FATAL: MLX scorer response batch failed: {exc}", file=sys.stderr)
        return 2
    print(
        json_text(
            {
                "summary_out": str(args.summary_out),
                "summary_bytes": write.bytes_written,
                "job_count": len(rows),
                "wall_seconds": summary["wall_seconds"],
                "score_claim": False,
                "promotion_eligible": False,
                "rank_or_kill_eligible": False,
                "ready_for_exact_eval_dispatch": False,
            }
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
