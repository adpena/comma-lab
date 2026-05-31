#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize an unaudited MLX scorer-input cache from a submission inflate.

This is the MLX-first acquisition path for local candidate generation.  It
runs ``archive.zip -> inflate.sh`` and writes scorer-input tensors, but it does
not run the CPU scorer and never creates score authority.  Downstream MLX
response runners must opt in to the unaudited-cache contract explicitly.
"""

from __future__ import annotations

import argparse
import json
import shutil
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

from experiments.contest_auth_eval import (  # noqa: E402  # pyright: ignore[reportPrivateUsage]
    _ensure_uv_available,
    _extract_archive,
    _record_inflated_output_artifacts,
    _run_inflate,
    _sha256,
    _validate_archive_members,
)
from tac.local_acceleration.mlx_preprocess import (  # noqa: E402
    load_raw_video_memmap,
    non_overlapping_pair_indices,
    write_scorer_input_cache_from_raw_file,
)
from tac.repo_io import write_json  # noqa: E402

SCHEMA = "mlx_scorer_cache_from_submission_inflate_only.v1"
OWNED_MARKER = ".mlx_scorer_cache_from_submission_owned.json"
FALSE_AUTHORITY = {
    "score_claim": False,
    "score_claim_valid": False,
    "promotion_eligible": False,
    "promotable": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--submission-dir", required=True, type=Path)
    parser.add_argument("--inflate-sh", default="inflate.sh")
    parser.add_argument("--upstream-dir", default=Path("upstream"), type=Path)
    parser.add_argument("--video-names-file", type=Path)
    parser.add_argument("--output-cache-dir", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--report-output", required=True, type=Path)
    parser.add_argument("--inflate-timeout", type=int, default=1800)
    parser.add_argument("--batch-pairs", type=int, default=1)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument(
        "--local-acquisition-max-pairs",
        type=int,
        help=(
            "Set PACT_LOCAL_ACQUISITION_MAX_PAIRS during inflate and validate "
            "a partial raw surface. This is for queue-owned MLX acquisition "
            "only and is never exact-eval authority."
        ),
    )
    parser.add_argument("--large-cache-pair-threshold", type=int, default=64)
    parser.add_argument("--allow-large-tensor-cache", action="store_true")
    parser.add_argument("--keep-raw", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.inflate_timeout < 1:
        raise SystemExit("--inflate-timeout must be >= 1")
    if args.batch_pairs < 1:
        raise SystemExit("--batch-pairs must be >= 1")
    if args.max_pairs is not None and args.max_pairs < 1:
        raise SystemExit("--max-pairs must be >= 1")
    if args.local_acquisition_max_pairs is not None and args.local_acquisition_max_pairs < 1:
        raise SystemExit("--local-acquisition-max-pairs must be >= 1")
    if args.large_cache_pair_threshold < 1:
        raise SystemExit("--large-cache-pair-threshold must be >= 1")

    archive = args.archive.resolve()
    submission_dir = args.submission_dir.resolve()
    inflate_sh = (submission_dir / args.inflate_sh).resolve()
    upstream_dir = args.upstream_dir.resolve()
    video_names_file = (
        args.video_names_file.resolve()
        if args.video_names_file is not None
        else (upstream_dir / "public_test_video_names.txt").resolve()
    )
    _require_file(archive, "archive")
    _require_file(inflate_sh, "inflate_sh")
    _require_file(video_names_file, "video_names_file")

    work_dir = args.work_dir.resolve()
    output_cache = args.output_cache_dir.resolve()
    report_output = args.report_output.resolve()
    _prepare_owned_dir(work_dir, force=args.force, label="work_dir")
    _prepare_owned_dir(output_cache, force=args.force, label="output_cache_dir")
    if report_output.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite report: {report_output}")
    report_output.parent.mkdir(parents=True, exist_ok=True)

    _ensure_uv_available()
    started = time.time()
    extracted_dir = work_dir / "archive"
    inflated_dir = work_dir / "inflated"
    members = _extract_archive(archive, extracted_dir)
    _validate_archive_members(members)
    local_acquisition_env = (
        {"PACT_LOCAL_ACQUISITION_MAX_PAIRS": str(int(args.local_acquisition_max_pairs))}
        if args.local_acquisition_max_pairs is not None
        else None
    )
    inflate_expected_frames = (
        int(args.local_acquisition_max_pairs) * 2
        if args.local_acquisition_max_pairs is not None
        else 1200
    )
    inflate_elapsed = _run_inflate(
        inflate_sh,
        extracted_dir,
        inflated_dir,
        video_names_file,
        timeout=int(args.inflate_timeout),
        extra_env=local_acquisition_env,
        expected_num_frames=inflate_expected_frames,
    )
    provenance: dict[str, Any] = {
        "archive_sha256": _sha256(archive, prefix=0),
        "archive_size_bytes": archive.stat().st_size,
        "archive_path": str(archive),
        "submission_dir": str(submission_dir),
        "inflate_sh": str(inflate_sh),
        "video_names_file": str(video_names_file),
        **FALSE_AUTHORITY,
    }
    inflated_manifest = _record_inflated_output_artifacts(
        provenance,
        work_dir,
        inflated_dir,
        video_names_file,
    )
    raw_path = _single_raw_path(inflated_dir, video_names_file)
    raw = load_raw_video_memmap(raw_path)
    raw_pair_count = len(non_overlapping_pair_indices(raw.shape[0]))
    cached_pair_count = (
        raw_pair_count
        if args.max_pairs is None
        else min(raw_pair_count, int(args.max_pairs))
    )
    if cached_pair_count > args.large_cache_pair_threshold and not args.allow_large_tensor_cache:
        raise SystemExit(
            "refusing full MLX tensor cache for "
            f"{cached_pair_count} pairs (> threshold {args.large_cache_pair_threshold}); "
            "pass --allow-large-tensor-cache after confirming disk budget"
        )
    manifest = write_scorer_input_cache_from_raw_file(
        raw_path,
        output_cache,
        archive_sha256=provenance["archive_sha256"],
        inflated_outputs_aggregate_sha256=str(inflated_manifest["aggregate_sha256"]),
        max_pairs=args.max_pairs,
        batch_pairs=int(args.batch_pairs),
    )
    manifest["eligible_for_local_mlx_transfer_calibration"] = False
    manifest["eligible_for_unaudited_mlx_candidate_generation"] = True
    manifest["unaudited_candidate_cache_contract"] = {
        "schema": "unaudited_mlx_candidate_cache_contract.v1",
        "created_by": "tools/materialize_mlx_scorer_cache_from_submission.py",
        "allowed_use": "local_mlx_candidate_generation_prior_before_cpu_gate",
        "forbidden_use": "score_claim_or_promotion_or_rank_or_exact_dispatch",
        **FALSE_AUTHORITY,
    }
    write_json(output_cache / "manifest.json", manifest)
    report = {
        "schema": SCHEMA,
        "archive": {
            "path": str(archive),
            "bytes": archive.stat().st_size,
            "sha256": provenance["archive_sha256"],
        },
        "submission_dir": str(submission_dir),
        "inflate_sh": str(inflate_sh),
        "work_dir": str(work_dir),
        "output_cache_dir": str(output_cache),
        "cache_manifest": str(output_cache / "manifest.json"),
        "inflated_outputs_manifest": str(work_dir / "inflated_outputs_manifest.json"),
        "inflated_outputs_aggregate_sha256": inflated_manifest["aggregate_sha256"],
        "raw_path": str(raw_path),
        "raw_sha256": manifest.get("raw_sha256"),
        "raw_pair_count": int(raw_pair_count),
        "cached_pair_count": int(manifest["pair_count"]),
        "pair_count": int(manifest["pair_count"]),
        "max_pairs": args.max_pairs,
        "local_acquisition_max_pairs": args.local_acquisition_max_pairs,
        "local_acquisition_partial_raw": args.local_acquisition_max_pairs is not None,
        "inflate_elapsed_seconds": float(inflate_elapsed),
        "elapsed_seconds": time.time() - started,
        "candidate_cache_identity_mode": "unaudited_candidate_generation_prior",
        "cpu_score_computed": False,
        "mlx_first_acquisition_only": True,
        "requires_local_cpu_gate_before_exact_auth": True,
        **FALSE_AUTHORITY,
    }
    write_json(report_output, report)
    if not args.keep_raw:
        shutil.rmtree(inflated_dir, ignore_errors=True)
    print(
        json.dumps(
            {
                "report": str(report_output),
                "raw_pair_count": raw_pair_count,
                "cached_pair_count": int(manifest["pair_count"]),
            },
            sort_keys=True,
        )
    )
    return 0


def _prepare_owned_dir(path: Path, *, force: bool, label: str) -> None:
    if path.exists():
        marker = path / OWNED_MARKER
        if not marker.exists() and not force:
            raise SystemExit(f"refusing to reuse non-owned {label}: {path}")
        if force:
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    marker = path / OWNED_MARKER
    marker.write_text(
        json.dumps({"schema": "owned_directory_marker.v1", "tool": __file__}) + "\n",
        encoding="utf-8",
    )


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"{label} missing: {path}")


def _single_raw_path(inflated_dir: Path, video_names_file: Path) -> Path:
    names = [line.strip() for line in video_names_file.read_text().splitlines() if line.strip()]
    if len(names) != 1:
        raise SystemExit(f"expected exactly one video for MLX cache materialization, got {len(names)}")
    raw_path = inflated_dir / Path(names[0]).with_suffix(".raw")
    if not raw_path.is_file():
        raise SystemExit(f"inflated raw file missing: {raw_path}")
    return raw_path


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
