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
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

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
    _validate_zip_container_integrity,
)
from tac.local_acceleration.mlx_preprocess import (  # noqa: E402
    CAMERA_HW,
    load_raw_video_memmap,
    non_overlapping_pair_indices,
    write_scorer_input_cache_from_pair_batches,
    write_scorer_input_cache_from_raw_file,
)
from tac.process_group_kill import run_in_process_group  # noqa: E402
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
    parser.add_argument(
        "--preinflated-output-dir",
        type=Path,
        help=(
            "Use an already proven shell-inflate output directory instead of "
            "running inflate.sh again. The directory must contain raw files "
            "or, with --allow-png-frame-tree-output, PNG frame trees matching "
            "--video-names-file and must not live inside --work-dir."
        ),
    )
    parser.add_argument(
        "--preinflated-proof-manifest",
        type=Path,
        help=(
            "Required with preinflated PNG frame trees. Must bind the "
            "preinflated bytes to the archive/runtime proof that produced them."
        ),
    )
    parser.add_argument(
        "--allow-png-frame-tree-output",
        action="store_true",
        help=(
            "Accept receiver-proof-style output_dir/<video>.raw/*.png frame "
            "trees as an MLX advisory scorer-cache source. This never creates "
            "score authority and exists for compact NeRV/PACT candidates whose "
            "runtime emits PNG frame trees instead of contest-sized raw files."
        ),
    )
    parser.add_argument("--inflate-timeout", type=int, default=1800)
    parser.add_argument("--batch-pairs", type=int, default=1)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument(
        "--pair-ranges",
        help=(
            "For --receiver-direct-cache/--hprc-direct-cache, render only "
            "explicit pair indices/ranges such as 1-2,4,9-12. The manifest "
            "preserves source pair indices so MLX response can align the "
            "subset against the full reference cache."
        ),
    )
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
    parser.add_argument(
        "--hprc-direct-cache",
        action="store_true",
        help=(
            "For compact HPRC receiver archives, render scorer-input tensors "
            "directly from archive bytes instead of writing a multi-GB raw "
            "scratch file. Advisory only; receiver proof remains mandatory."
        ),
    )
    parser.add_argument(
        "--receiver-direct-cache",
        action="store_true",
        help=(
            "Render scorer-input tensors directly from supported deterministic "
            "receiver archive bytes instead of writing a multi-GB raw scratch "
            "file. Currently supports compact HPRC packets and HiNeRV HIV1 "
            "archives. Advisory only; receiver proof remains mandatory."
        ),
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    if args.inflate_timeout < 1:
        raise SystemExit("--inflate-timeout must be >= 1")
    if args.batch_pairs < 1:
        raise SystemExit("--batch-pairs must be >= 1")
    if args.max_pairs is not None and args.max_pairs < 1:
        raise SystemExit("--max-pairs must be >= 1")
    if args.local_acquisition_max_pairs is not None and args.local_acquisition_max_pairs < 1:
        raise SystemExit("--local-acquisition-max-pairs must be >= 1")
    receiver_direct_cache = bool(args.hprc_direct_cache or args.receiver_direct_cache)
    if args.pair_ranges and not receiver_direct_cache:
        raise SystemExit(
            "--pair-ranges currently requires --receiver-direct-cache "
            "or --hprc-direct-cache"
        )
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
    preinflated_output_dir = (
        args.preinflated_output_dir.resolve()
        if args.preinflated_output_dir is not None
        else None
    )
    preinflated_proof_manifest = (
        args.preinflated_proof_manifest.resolve()
        if args.preinflated_proof_manifest is not None
        else None
    )
    if preinflated_output_dir is not None:
        if receiver_direct_cache:
            raise SystemExit(
                "--receiver-direct-cache/--hprc-direct-cache cannot be combined "
                "with --preinflated-output-dir"
            )
        if not preinflated_output_dir.is_dir():
            raise SystemExit(
                f"--preinflated-output-dir is not a directory: {preinflated_output_dir}"
            )
        if _is_relative_to(preinflated_output_dir, work_dir):
            raise SystemExit(
                "--preinflated-output-dir must not be inside --work-dir; "
                f"work-dir cleanup would destroy the source: {preinflated_output_dir}"
            )
    if preinflated_proof_manifest is not None and not preinflated_proof_manifest.is_file():
        raise SystemExit(f"--preinflated-proof-manifest missing: {preinflated_proof_manifest}")
    if (
        preinflated_output_dir is not None
        and args.allow_png_frame_tree_output
        and preinflated_proof_manifest is None
    ):
        raise SystemExit(
            "--preinflated-proof-manifest is required when reusing "
            "preinflated PNG frame trees"
        )
    _prepare_owned_dir(work_dir, force=args.force, label="work_dir")
    _prepare_owned_dir(output_cache, force=args.force, label="output_cache_dir")
    if report_output.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite report: {report_output}")
    report_output.parent.mkdir(parents=True, exist_ok=True)

    _ensure_uv_available()
    started = time.time()
    extracted_dir = work_dir / "archive"
    inflated_dir = work_dir / "inflated"
    inflate_executed = preinflated_output_dir is None and not receiver_direct_cache
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
    archive_sha256 = _sha256(archive, prefix=0)
    direct_cache_report: dict[str, Any] | None = None
    png_frame_count: int | None = None
    inflated_surface_kind: str | None = None
    png_tree_cache_blockers: list[str] = []
    preinflated_proof = (
        _load_preinflated_proof_manifest(preinflated_proof_manifest)
        if preinflated_proof_manifest is not None
        else None
    )
    if receiver_direct_cache:
        _validate_archive_without_extracting(archive, validate_members=False)
        inflate_elapsed = 0.0
    elif inflate_executed:
        members = _extract_archive(archive, extracted_dir)
        _validate_archive_members(members)
        if args.allow_png_frame_tree_output:
            inflate_elapsed = _run_inflate_png_frame_tree_permitted(
                inflate_sh,
                extracted_dir,
                inflated_dir,
                video_names_file,
                timeout=int(args.inflate_timeout),
                extra_env=local_acquisition_env,
            )
        else:
            inflate_elapsed = _run_inflate(
                inflate_sh,
                extracted_dir,
                inflated_dir,
                video_names_file,
                timeout=int(args.inflate_timeout),
                extra_env=local_acquisition_env,
                expected_num_frames=inflate_expected_frames,
            )
    else:
        _validate_archive_without_extracting(archive)
        inflated_dir = preinflated_output_dir
        inflate_elapsed = 0.0
    provenance: dict[str, Any] = {
        "archive_sha256": archive_sha256,
        "archive_size_bytes": archive.stat().st_size,
        "archive_path": str(archive),
        "submission_dir": str(submission_dir),
        "inflate_sh": str(inflate_sh),
        "video_names_file": str(video_names_file),
        "inflate_executed": inflate_executed,
        "allow_png_frame_tree_output": bool(args.allow_png_frame_tree_output),
        "preinflated_output_dir": (
            str(preinflated_output_dir) if preinflated_output_dir is not None else None
        ),
        "preinflated_proof_manifest": (
            None if preinflated_proof_manifest is None else str(preinflated_proof_manifest)
        ),
        **FALSE_AUTHORITY,
    }
    if receiver_direct_cache:
        direct_cache_report, manifest = _write_direct_receiver_cache(
            archive,
            output_cache,
            max_pairs=args.max_pairs,
            local_acquisition_max_pairs=args.local_acquisition_max_pairs,
            batch_pairs=int(args.batch_pairs),
            archive_sha256=provenance["archive_sha256"],
            pair_indices_filter=_parse_pair_ranges(args.pair_ranges),
        )
        raw_path: Path | None = None
        inflated_manifest = {
            "aggregate_sha256": direct_cache_report["direct_render_raw_sha256"],
            "raw_file_count": 0,
            "total_bytes": direct_cache_report["direct_render_raw_bytes"],
            "files": [],
        }
        raw_pair_count = int(direct_cache_report["raw_pair_count"])
        local_acquisition_partial_raw = _local_acquisition_is_partial_raw(
            raw_pair_count=raw_pair_count,
            local_acquisition_max_pairs=args.local_acquisition_max_pairs,
            inflate_executed=True,
        )
        cached_pair_count = int(manifest["pair_count"])
    else:
        raw_path = _single_inflated_surface_path(inflated_dir, video_names_file)
        if raw_path.is_dir():
            if not args.allow_png_frame_tree_output:
                raise SystemExit(
                    "inflated output is a PNG frame tree; pass "
                    "--allow-png-frame-tree-output to materialize an MLX advisory cache"
                )
            png_paths = _png_frame_paths(raw_path)
            png_frame_count = len(png_paths)
            raw_pair_count = png_frame_count // 2
            inflated_surface_kind = "png_frame_tree"
            if preinflated_output_dir is not None:
                assert preinflated_proof is not None
                _validate_preinflated_png_tree_proof(
                    preinflated_proof,
                    archive_sha256=archive_sha256,
                    video_names_file=video_names_file,
                    frame_tree=raw_path,
                )
            inflated_manifest = _record_png_tree_inflated_output_artifacts(
                provenance,
                work_dir,
                inflated_dir,
                video_names_file,
                raw_path,
                png_paths,
            )
        else:
            inflated_surface_kind = "raw_file"
            inflated_manifest = _record_inflated_output_artifacts(
                provenance,
                work_dir,
                inflated_dir,
                video_names_file,
            )
            raw = load_raw_video_memmap(raw_path)
            raw_pair_count = len(non_overlapping_pair_indices(raw.shape[0]))
        local_acquisition_partial_raw = _local_acquisition_is_partial_raw(
            raw_pair_count=raw_pair_count,
            local_acquisition_max_pairs=args.local_acquisition_max_pairs,
            inflate_executed=inflate_executed,
        )
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
    if not receiver_direct_cache:
        if raw_path is not None and raw_path.is_dir():
            assert png_frame_count is not None
            png_paths = _png_frame_paths(raw_path)
            selected_frame_paths = png_paths[: cached_pair_count * 2]
            frame_shape_hwc = _png_frame_shape_hwc(selected_frame_paths[0])
            png_tree_cache_blockers = _png_tree_cache_blockers(
                frame_shape_hwc=frame_shape_hwc,
                png_frame_count=png_frame_count,
                cached_pair_count=cached_pair_count,
                expected_full_frames=1200,
            )
            manifest = write_scorer_input_cache_from_pair_batches(
                _iter_png_pair_batches(
                    selected_frame_paths,
                    pair_count=cached_pair_count,
                    batch_pairs=int(args.batch_pairs),
                    frame_shape_hwc=frame_shape_hwc,
                ),
                output_cache,
                pair_count=cached_pair_count,
                pair_indices=non_overlapping_pair_indices(cached_pair_count * 2),
                frame_shape_hwc=frame_shape_hwc,
                source=str(raw_path),
                source_kind="png_frame_tree_inflate",
                archive_sha256=provenance["archive_sha256"],
                inflated_outputs_aggregate_sha256=str(
                    inflated_manifest["aggregate_sha256"]
                ),
                batch_pairs=int(args.batch_pairs),
                compute_raw_sha256=True,
            )
            manifest["png_frame_tree_contract"] = {
                "schema": "png_frame_tree_mlx_cache_contract.v1",
                "surface_kind": "png_frame_tree",
                "contest_raw_geometry": frame_shape_hwc == (CAMERA_HW[0], CAMERA_HW[1], 3),
                "full_contest_frame_count": png_frame_count == 1200,
                "png_frame_count": int(png_frame_count),
                "cached_pair_count": int(cached_pair_count),
                "cache_blockers": png_tree_cache_blockers,
                "preinflated_proof_manifest": (
                    None
                    if preinflated_proof_manifest is None
                    else str(preinflated_proof_manifest)
                ),
                **FALSE_AUTHORITY,
            }
        else:
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
        "cache_blockers": png_tree_cache_blockers,
        **FALSE_AUTHORITY,
    }
    write_json(output_cache / "manifest.json", manifest)
    cache_identity_mode = (
        str(direct_cache_report["candidate_cache_identity_mode"])
        if direct_cache_report is not None
        else "unaudited_candidate_generation_prior"
    )
    report = {
        "schema": SCHEMA,
        "tool_command": [sys.executable, str(Path(__file__).resolve()), *raw_argv],
        "tool_cwd": str(Path.cwd()),
        "relevant_env": {
            "PACT_LOCAL_ACQUISITION_MAX_PAIRS": os.environ.get(
                "PACT_LOCAL_ACQUISITION_MAX_PAIRS"
            ),
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
        },
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
        "inflated_outputs_manifest": (
            None if receiver_direct_cache else str(work_dir / "inflated_outputs_manifest.json")
        ),
        "inflated_outputs_aggregate_sha256": inflated_manifest["aggregate_sha256"],
        "raw_path": str(raw_path) if raw_path is not None else None,
        "inflated_surface_kind": inflated_surface_kind,
        "png_frame_count": png_frame_count,
        "png_tree_cache_blockers": png_tree_cache_blockers,
        "raw_sha256": manifest.get("raw_sha256"),
        "raw_pair_count": int(raw_pair_count),
        "cached_pair_count": int(manifest["pair_count"]),
        "pair_count": int(manifest["pair_count"]),
        "max_pairs": args.max_pairs,
        "local_acquisition_max_pairs": args.local_acquisition_max_pairs,
        "allow_png_frame_tree_output": bool(args.allow_png_frame_tree_output),
        "local_acquisition_partial_raw": local_acquisition_partial_raw,
        "local_acquisition_full_raw_pair_floor": 600,
        "inflate_executed": inflate_executed,
        "hprc_direct_cache": bool(args.hprc_direct_cache),
        "hprc_direct_cache_report": (
            direct_cache_report
            if direct_cache_report is not None
            and direct_cache_report.get("source_family") == "hprc"
            else None
        ),
        "receiver_direct_cache": bool(receiver_direct_cache),
        "direct_receiver_cache_report": direct_cache_report,
        "preinflated_output_dir": (
            str(preinflated_output_dir) if preinflated_output_dir is not None else None
        ),
        "preinflated_proof_manifest": (
            None if preinflated_proof_manifest is None else str(preinflated_proof_manifest)
        ),
        "preinflated_proof_manifest_sha256": (
            None if preinflated_proof is None else preinflated_proof["sha256"]
        ),
        "inflate_elapsed_seconds": float(inflate_elapsed),
        "elapsed_seconds": time.time() - started,
        "candidate_cache_identity_mode": cache_identity_mode,
        "cpu_score_computed": False,
        "mlx_first_acquisition_only": True,
        "requires_local_cpu_gate_before_exact_auth": True,
        **FALSE_AUTHORITY,
    }
    write_json(report_output, report)
    if not args.keep_raw and inflate_executed and not receiver_direct_cache:
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


def _local_acquisition_is_partial_raw(
    *,
    raw_pair_count: int,
    local_acquisition_max_pairs: int | None,
    inflate_executed: bool,
) -> bool:
    if local_acquisition_max_pairs is None or not inflate_executed:
        return False
    return int(raw_pair_count) < 600


def _write_direct_receiver_cache(
    archive: Path,
    output_cache: Path,
    *,
    max_pairs: int | None,
    local_acquisition_max_pairs: int | None,
    batch_pairs: int,
    archive_sha256: str,
    pair_indices_filter: list[int] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    member_name, packet_bytes = _read_single_member_zip(archive)
    if packet_bytes.startswith(b"HIV1"):
        return _write_hi_nerv_direct_cache_from_payload(
            archive,
            output_cache,
            member_name=member_name,
            archive_payload=packet_bytes,
            max_pairs=max_pairs,
            local_acquisition_max_pairs=local_acquisition_max_pairs,
            batch_pairs=batch_pairs,
            archive_sha256=archive_sha256,
            pair_indices_filter=pair_indices_filter,
        )
    if packet_bytes.startswith(b"SNAR1"):
        return _write_snerv_direct_cache_from_payload(
            archive,
            output_cache,
            member_name=member_name,
            packet_bytes=packet_bytes,
            max_pairs=max_pairs,
            local_acquisition_max_pairs=local_acquisition_max_pairs,
            batch_pairs=batch_pairs,
            archive_sha256=archive_sha256,
            pair_indices_filter=pair_indices_filter,
        )
    return _write_hprc_direct_cache_from_payload(
        archive,
        output_cache,
        member_name=member_name,
        packet_bytes=packet_bytes,
        max_pairs=max_pairs,
        local_acquisition_max_pairs=local_acquisition_max_pairs,
        batch_pairs=batch_pairs,
        archive_sha256=archive_sha256,
        pair_indices_filter=pair_indices_filter,
    )


def _write_hprc_direct_cache_from_payload(
    archive: Path,
    output_cache: Path,
    *,
    member_name: str,
    packet_bytes: bytes,
    max_pairs: int | None,
    local_acquisition_max_pairs: int | None,
    batch_pairs: int,
    archive_sha256: str,
    pair_indices_filter: list[int] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from tac.substrates.hprc.archive import parse_hprc_packet
    from tac.substrates.hprc.learned_receiver import (
        decode_compact_receiver_packet,
        is_compact_receiver_packet,
        render_compact_receiver_frame_batch,
    )

    packet = parse_hprc_packet(packet_bytes)
    if not is_compact_receiver_packet(packet):
        raise SystemExit("--hprc-direct-cache requires compact_numpy_receiver_v1 archive bytes")
    compact = decode_compact_receiver_packet(packet)
    raw_pair_count = int(packet.config.frames) // 2
    if local_acquisition_max_pairs is not None:
        raw_pair_count = min(raw_pair_count, int(local_acquisition_max_pairs))
    selected_pair_indices = (
        list(range(raw_pair_count))
        if pair_indices_filter is None
        else _validate_selected_pair_indices(pair_indices_filter, raw_pair_count=raw_pair_count)
    )
    if max_pairs is not None:
        selected_pair_indices = selected_pair_indices[: int(max_pairs)]
    pair_count = len(selected_pair_indices)
    if pair_count < 1:
        raise SystemExit("HPRC direct cache has no complete frame pairs")

    h, w = CAMERA_HW
    scorer_pair_indices = np.array(
        [[2 * idx, 2 * idx + 1] for idx in selected_pair_indices],
        dtype=np.int64,
    )

    def pair_batches():
        for start in range(0, pair_count, int(batch_pairs)):
            chunk_indices = selected_pair_indices[start : start + int(batch_pairs)]
            chunks = [
                render_compact_receiver_frame_batch(
                    compact,
                    pair_index * 2,
                    2,
                    height=h,
                    width=w,
                ).reshape(1, 2, h, w, 3)
                for pair_index in chunk_indices
            ]
            yield np.concatenate(chunks, axis=0)

    manifest = write_scorer_input_cache_from_pair_batches(
        pair_batches(),
        output_cache,
        pair_count=pair_count,
        pair_indices=scorer_pair_indices,
        frame_shape_hwc=(h, w, 3),
        source=str(archive),
        source_kind="hprc_direct_receiver_render",
        archive_sha256=archive_sha256,
        inflated_outputs_aggregate_sha256=None,
        batch_pairs=batch_pairs,
        compute_raw_sha256=True,
    )
    manifest["inflated_outputs_aggregate_sha256"] = manifest.get("raw_sha256")
    audit_path = output_cache / "hprc_direct_receiver_render_cache_identity_audit.json"
    audit = {
        "schema_version": "hprc_direct_receiver_render_cache_identity_audit.v1",
        "verdict": "PASS_HPRC_DIRECT_RECEIVER_RENDER_CACHE_IDENTITY",
        "passed": True,
        "created_by": "tools/materialize_mlx_scorer_cache_from_submission.py",
        "allowed_use": "certify_hprc_direct_mlx_cache_rebuildability_for_disk_retention",
        "forbidden_use": "score_claim_or_promotion_or_rank_or_exact_dispatch",
        "cache": {
            "archive_sha256": manifest.get("archive_sha256"),
            "inflated_outputs_aggregate_sha256": manifest.get(
                "inflated_outputs_aggregate_sha256"
            ),
            "raw_sha256": manifest.get("raw_sha256"),
            "pair_count": manifest.get("pair_count"),
            "hash_domain": manifest.get("hash_domain"),
            "array_sha256": manifest.get("array_sha256"),
        },
        "source": {
            "archive_path": str(archive),
            "archive_sha256": archive_sha256,
            "zip_member": member_name,
            "packet_config": packet.config.as_dict(),
        },
        "direct_render": {
            "raw_pair_count": raw_pair_count,
            "selected_pair_count": int(pair_count),
            "selected_pair_ranges": _format_pair_ranges(selected_pair_indices),
            "pair_index_scope": (
                "explicit_pair_ranges" if pair_indices_filter is not None else "prefix_from_zero"
            ),
            "frame_shape_hwc": [h, w, 3],
            "batch_pairs": int(batch_pairs),
            "max_pairs": max_pairs,
            "local_acquisition_max_pairs": local_acquisition_max_pairs,
            "raw_file_written": False,
            "rebuilds_from_archive_bytes": True,
        },
        "receiver_proof_required_for_promotion": True,
        **FALSE_AUTHORITY,
    }
    write_json(audit_path, audit)
    manifest["hprc_direct_receiver_render_cache_identity_audit"] = {
        "schema_version": audit["schema_version"],
        "path": str(audit_path),
        "sha256": _sha256(audit_path, prefix=0),
        "verdict": audit["verdict"],
        "passed": True,
        "archive_path": str(archive),
        "archive_sha256": archive_sha256,
        **FALSE_AUTHORITY,
    }
    manifest["eligible_for_hprc_direct_rebuild_cleanup"] = True
    write_json(output_cache / "manifest.json", manifest)
    report = {
        "schema": "hprc_direct_mlx_scorer_cache_render.v1",
        "source_family": "hprc",
        "archive_path": str(archive),
        "archive_sha256": archive_sha256,
        "zip_member": member_name,
        "packet_config": packet.config.as_dict(),
        "raw_pair_count": raw_pair_count,
        "cached_pair_count": int(manifest["pair_count"]),
        "selected_pair_count": int(pair_count),
        "selected_pair_ranges": _format_pair_ranges(selected_pair_indices),
        "pair_index_scope": (
            "explicit_pair_ranges" if pair_indices_filter is not None else "prefix_from_zero"
        ),
        "frame_shape_hwc": [h, w, 3],
        "direct_render_raw_bytes": int(manifest["pair_count"]) * 2 * h * w * 3,
        "direct_render_raw_pair_count": int(manifest["pair_count"]),
        "direct_render_raw_sha256": manifest.get("raw_sha256"),
        "direct_render_raw_sha256_scope": manifest.get("raw_sha256_scope"),
        "raw_file_written": False,
        "receiver_proof_required_for_promotion": True,
        "identity_audit_path": str(audit_path),
        "identity_audit_sha256": manifest[
            "hprc_direct_receiver_render_cache_identity_audit"
        ]["sha256"],
        "candidate_cache_identity_mode": (
            "hprc_direct_receiver_render_cache_identity_audited_false_authority"
        ),
        **FALSE_AUTHORITY,
    }
    return report, manifest


def _write_snerv_direct_cache_from_payload(
    archive: Path,
    output_cache: Path,
    *,
    member_name: str,
    packet_bytes: bytes,
    max_pairs: int | None,
    local_acquisition_max_pairs: int | None,
    batch_pairs: int,
    archive_sha256: str,
    pair_indices_filter: list[int] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from tac.substrates.snerv_inverse_steg_carrier.archive import (
        decode_snerv_archive_pair_frames_from_decoded,
        unpack_snerv_archive,
    )

    decoded = unpack_snerv_archive(packet_bytes)
    raw_pair_count = int(decoded.metadata.get("n_pairs") or 0)
    if raw_pair_count < 1:
        raise SystemExit("SNeRV direct cache has no complete frame pairs")
    if local_acquisition_max_pairs is not None:
        raw_pair_count = min(raw_pair_count, int(local_acquisition_max_pairs))
    selected_pair_indices = (
        list(range(raw_pair_count))
        if pair_indices_filter is None
        else _validate_selected_pair_indices(pair_indices_filter, raw_pair_count=raw_pair_count)
    )
    if max_pairs is not None:
        selected_pair_indices = selected_pair_indices[: int(max_pairs)]
    pair_count = len(selected_pair_indices)
    if pair_count < 1:
        raise SystemExit("SNeRV direct cache has no complete frame pairs")

    frames = decode_snerv_archive_pair_frames_from_decoded(
        decoded,
        selected_pair_indices,
        clip_to_uint8_range=True,
    )
    if frames.ndim != 5 or frames.shape[1] != 2 or frames.shape[2] != 3:
        raise SystemExit(f"SNeRV direct cache expected frames (pairs,2,3,H,W), got {frames.shape}")
    if frames.shape[0] != pair_count:
        raise SystemExit(
            f"SNeRV direct cache decoded {frames.shape[0]} selected pairs, expected {pair_count}"
        )
    h, w = int(frames.shape[3]), int(frames.shape[4])
    scorer_pair_indices = np.array(
        [[2 * idx, 2 * idx + 1] for idx in selected_pair_indices],
        dtype=np.int64,
    )

    def pair_batches():
        for start in range(0, pair_count, int(batch_pairs)):
            chunk = frames[start : start + int(batch_pairs)]
            chunk = np.transpose(chunk, (0, 1, 3, 4, 2))
            yield np.ascontiguousarray(np.rint(chunk).clip(0, 255).astype(np.uint8))

    manifest = write_scorer_input_cache_from_pair_batches(
        pair_batches(),
        output_cache,
        pair_count=pair_count,
        pair_indices=scorer_pair_indices,
        frame_shape_hwc=(h, w, 3),
        source=str(archive),
        source_kind="snerv_direct_receiver_render",
        archive_sha256=archive_sha256,
        inflated_outputs_aggregate_sha256=None,
        batch_pairs=batch_pairs,
        compute_raw_sha256=True,
    )
    manifest["inflated_outputs_aggregate_sha256"] = manifest.get("raw_sha256")
    audit_path = output_cache / "snerv_direct_receiver_render_cache_identity_audit.json"
    audit = {
        "schema_version": "snerv_direct_receiver_render_cache_identity_audit.v1",
        "verdict": "PASS_SNERV_DIRECT_RECEIVER_RENDER_CACHE_IDENTITY",
        "passed": True,
        "created_by": "tools/materialize_mlx_scorer_cache_from_submission.py",
        "allowed_use": "certify_snerv_direct_mlx_cache_rebuildability_for_disk_retention",
        "forbidden_use": "score_claim_or_promotion_or_rank_or_exact_dispatch",
        "cache": {
            "archive_sha256": manifest.get("archive_sha256"),
            "inflated_outputs_aggregate_sha256": manifest.get(
                "inflated_outputs_aggregate_sha256"
            ),
            "raw_sha256": manifest.get("raw_sha256"),
            "pair_count": manifest.get("pair_count"),
            "hash_domain": manifest.get("hash_domain"),
            "array_sha256": manifest.get("array_sha256"),
        },
        "source": {
            "archive_path": str(archive),
            "archive_sha256": archive_sha256,
            "zip_member": member_name,
            "archive_magic": "SNAR1",
            "packet_sha256": decoded.packet_sha256,
            "metadata": dict(decoded.metadata),
        },
        "direct_render": {
            "raw_pair_count": raw_pair_count,
            "selected_pair_count": int(pair_count),
            "selected_pair_ranges": _format_pair_ranges(selected_pair_indices),
            "pair_index_scope": (
                "explicit_pair_ranges" if pair_indices_filter is not None else "prefix_from_zero"
            ),
            "frame_shape_hwc": [h, w, 3],
            "batch_pairs": int(batch_pairs),
            "max_pairs": max_pairs,
            "local_acquisition_max_pairs": local_acquisition_max_pairs,
            "raw_file_written": False,
            "rebuilds_from_archive_bytes": True,
            "lowering": "decode_snerv_archive_pair_frames_nchw_to_uint8_hwc",
        },
        "receiver_proof_required_for_promotion": True,
        **FALSE_AUTHORITY,
    }
    write_json(audit_path, audit)
    manifest["snerv_direct_receiver_render_cache_identity_audit"] = {
        "schema_version": audit["schema_version"],
        "path": str(audit_path),
        "sha256": _sha256(audit_path, prefix=0),
        "verdict": audit["verdict"],
        "passed": True,
        "archive_path": str(archive),
        "archive_sha256": archive_sha256,
        **FALSE_AUTHORITY,
    }
    manifest["eligible_for_snerv_direct_rebuild_cleanup"] = True
    write_json(output_cache / "manifest.json", manifest)
    report = {
        "schema": "snerv_direct_mlx_scorer_cache_render.v1",
        "source_family": "snerv",
        "archive_path": str(archive),
        "archive_sha256": archive_sha256,
        "zip_member": member_name,
        "archive_magic": "SNAR1",
        "packet_sha256": decoded.packet_sha256,
        "raw_pair_count": raw_pair_count,
        "cached_pair_count": int(manifest["pair_count"]),
        "selected_pair_count": int(pair_count),
        "selected_pair_ranges": _format_pair_ranges(selected_pair_indices),
        "pair_index_scope": (
            "explicit_pair_ranges" if pair_indices_filter is not None else "prefix_from_zero"
        ),
        "frame_shape_hwc": [h, w, 3],
        "direct_render_raw_bytes": int(manifest["pair_count"]) * 2 * h * w * 3,
        "direct_render_raw_pair_count": int(manifest["pair_count"]),
        "direct_render_raw_sha256": manifest.get("raw_sha256"),
        "direct_render_raw_sha256_scope": manifest.get("raw_sha256_scope"),
        "raw_file_written": False,
        "receiver_proof_required_for_promotion": True,
        "identity_audit_path": str(audit_path),
        "identity_audit_sha256": manifest[
            "snerv_direct_receiver_render_cache_identity_audit"
        ]["sha256"],
        "candidate_cache_identity_mode": (
            "snerv_direct_receiver_render_cache_identity_audited_false_authority"
        ),
        **FALSE_AUTHORITY,
    }
    return report, manifest


def _write_hi_nerv_direct_cache_from_payload(
    archive: Path,
    output_cache: Path,
    *,
    member_name: str,
    archive_payload: bytes,
    max_pairs: int | None,
    local_acquisition_max_pairs: int | None,
    batch_pairs: int,
    archive_sha256: str,
    pair_indices_filter: list[int] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    from tac.substrates._shared.inflate_runtime import (
        CAMERA_HW as INFLATE_CAMERA_HW,
    )
    from tac.substrates._shared.inflate_runtime import (
        rgb_pair_to_uint8_frames,
    )
    from tac.substrates.hi_nerv.inflate import build_model_from_archive

    arc, cfg, model = build_model_from_archive(archive_payload, device="cpu")
    raw_pair_count = int(cfg.num_pairs)
    if local_acquisition_max_pairs is not None:
        raw_pair_count = min(raw_pair_count, int(local_acquisition_max_pairs))
    selected_pair_indices = (
        list(range(raw_pair_count))
        if pair_indices_filter is None
        else _validate_selected_pair_indices(pair_indices_filter, raw_pair_count=raw_pair_count)
    )
    if max_pairs is not None:
        selected_pair_indices = selected_pair_indices[: int(max_pairs)]
    pair_count = len(selected_pair_indices)
    if pair_count < 1:
        raise SystemExit("HiNeRV direct cache has no complete frame pairs")

    h, w = INFLATE_CAMERA_HW
    scorer_pair_indices = np.array(
        [[2 * idx, 2 * idx + 1] for idx in selected_pair_indices],
        dtype=np.int64,
    )

    def pair_batches():
        with torch.no_grad():
            for start in range(0, pair_count, int(batch_pairs)):
                chunk_indices = selected_pair_indices[start : start + int(batch_pairs)]
                rendered_pairs: list[np.ndarray] = []
                for pair_index in chunk_indices:
                    idx_tensor = torch.tensor([pair_index], device="cpu", dtype=torch.long)
                    rgb_0, rgb_1 = model(idx_tensor)
                    rendered_pairs.append(
                        rgb_pair_to_uint8_frames(
                            rgb_0,
                            rgb_1,
                            input_range="unit",
                        ).reshape(1, 2, h, w, 3)
                    )
                yield np.concatenate(rendered_pairs, axis=0)

    manifest = write_scorer_input_cache_from_pair_batches(
        pair_batches(),
        output_cache,
        pair_count=pair_count,
        pair_indices=scorer_pair_indices,
        frame_shape_hwc=(h, w, 3),
        source=str(archive),
        source_kind="hi_nerv_direct_receiver_render",
        archive_sha256=archive_sha256,
        inflated_outputs_aggregate_sha256=None,
        batch_pairs=batch_pairs,
        compute_raw_sha256=True,
    )
    manifest["inflated_outputs_aggregate_sha256"] = manifest.get("raw_sha256")
    audit_path = output_cache / "hi_nerv_direct_receiver_render_cache_identity_audit.json"
    audit = {
        "schema_version": "hi_nerv_direct_receiver_render_cache_identity_audit.v1",
        "verdict": "PASS_HI_NERV_DIRECT_RECEIVER_RENDER_CACHE_IDENTITY",
        "passed": True,
        "created_by": "tools/materialize_mlx_scorer_cache_from_submission.py",
        "allowed_use": "certify_hi_nerv_direct_mlx_cache_rebuildability_for_disk_retention",
        "forbidden_use": "score_claim_or_promotion_or_rank_or_exact_dispatch",
        "cache": {
            "archive_sha256": manifest.get("archive_sha256"),
            "inflated_outputs_aggregate_sha256": manifest.get(
                "inflated_outputs_aggregate_sha256"
            ),
            "raw_sha256": manifest.get("raw_sha256"),
            "pair_count": manifest.get("pair_count"),
            "hash_domain": manifest.get("hash_domain"),
            "array_sha256": manifest.get("array_sha256"),
        },
        "source": {
            "archive_path": str(archive),
            "archive_sha256": archive_sha256,
            "zip_member": member_name,
            "archive_magic": "HIV1",
            "schema_version": int(arc.schema_version),
            "config": {
                "num_pairs": int(cfg.num_pairs),
                "latent_dim_coarse": int(cfg.latent_dim_coarse),
                "latent_dim_mid": int(cfg.latent_dim_mid),
                "latent_dim_fine": int(cfg.latent_dim_fine),
                "embed_dim": int(cfg.embed_dim),
                "initial_grid_h": int(cfg.initial_grid_h),
                "initial_grid_w": int(cfg.initial_grid_w),
                "decoder_channels": [int(c) for c in cfg.decoder_channels],
                "sin_frequency": float(cfg.sin_frequency),
                "num_upsample_blocks": int(cfg.num_upsample_blocks),
                "mid_injection_block_index": int(cfg.mid_injection_block_index),
                "fine_injection_block_index": int(cfg.fine_injection_block_index),
                "output_height": int(cfg.output_height),
                "output_width": int(cfg.output_width),
            },
        },
        "direct_render": {
            "raw_pair_count": raw_pair_count,
            "selected_pair_count": int(pair_count),
            "selected_pair_ranges": _format_pair_ranges(selected_pair_indices),
            "pair_index_scope": (
                "explicit_pair_ranges" if pair_indices_filter is not None else "prefix_from_zero"
            ),
            "frame_shape_hwc": [h, w, 3],
            "batch_pairs": int(batch_pairs),
            "max_pairs": max_pairs,
            "local_acquisition_max_pairs": local_acquisition_max_pairs,
            "raw_file_written": False,
            "rebuilds_from_archive_bytes": True,
            "lowering": "rgb_pair_to_uint8_frames_input_range_unit_bicubic",
        },
        "receiver_proof_required_for_promotion": True,
        **FALSE_AUTHORITY,
    }
    write_json(audit_path, audit)
    manifest["hi_nerv_direct_receiver_render_cache_identity_audit"] = {
        "schema_version": audit["schema_version"],
        "path": str(audit_path),
        "sha256": _sha256(audit_path, prefix=0),
        "verdict": audit["verdict"],
        "passed": True,
        "archive_path": str(archive),
        "archive_sha256": archive_sha256,
        **FALSE_AUTHORITY,
    }
    manifest["eligible_for_hi_nerv_direct_rebuild_cleanup"] = True
    write_json(output_cache / "manifest.json", manifest)
    report = {
        "schema": "hi_nerv_direct_mlx_scorer_cache_render.v1",
        "source_family": "hi_nerv",
        "archive_path": str(archive),
        "archive_sha256": archive_sha256,
        "zip_member": member_name,
        "archive_magic": "HIV1",
        "schema_version": int(arc.schema_version),
        "raw_pair_count": raw_pair_count,
        "cached_pair_count": int(manifest["pair_count"]),
        "selected_pair_count": int(pair_count),
        "selected_pair_ranges": _format_pair_ranges(selected_pair_indices),
        "pair_index_scope": (
            "explicit_pair_ranges" if pair_indices_filter is not None else "prefix_from_zero"
        ),
        "frame_shape_hwc": [h, w, 3],
        "direct_render_raw_bytes": int(manifest["pair_count"]) * 2 * h * w * 3,
        "direct_render_raw_pair_count": int(manifest["pair_count"]),
        "direct_render_raw_sha256": manifest.get("raw_sha256"),
        "direct_render_raw_sha256_scope": manifest.get("raw_sha256_scope"),
        "raw_file_written": False,
        "receiver_proof_required_for_promotion": True,
        "identity_audit_path": str(audit_path),
        "identity_audit_sha256": manifest[
            "hi_nerv_direct_receiver_render_cache_identity_audit"
        ]["sha256"],
        "candidate_cache_identity_mode": (
            "hi_nerv_direct_receiver_render_cache_identity_audited_false_authority"
        ),
        **FALSE_AUTHORITY,
    }
    return report, manifest


def _parse_pair_ranges(value: str | None) -> list[int] | None:
    if value is None or not value.strip():
        return None
    out: list[int] = []
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item:
            start_s, end_s = item.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                raise SystemExit(f"invalid descending pair range: {item}")
            out.extend(range(start, end + 1))
        else:
            out.append(int(item))
    return sorted(set(out))


def _validate_selected_pair_indices(
    pair_indices: list[int],
    *,
    raw_pair_count: int,
) -> list[int]:
    if not pair_indices:
        raise SystemExit("--pair-ranges selected no pairs")
    invalid = [idx for idx in pair_indices if idx < 0 or idx >= raw_pair_count]
    if invalid:
        preview = invalid[:8]
        suffix = "" if len(invalid) <= 8 else f" ... +{len(invalid) - 8} more"
        raise SystemExit(
            f"--pair-ranges contains pairs outside [0,{raw_pair_count}): {preview}{suffix}"
        )
    return list(pair_indices)


def _format_pair_ranges(indices: list[int]) -> list[list[int]]:
    if not indices:
        return []
    ranges: list[list[int]] = []
    start = prev = int(indices[0])
    for value in indices[1:]:
        idx = int(value)
        if idx == prev + 1:
            prev = idx
            continue
        ranges.append([start, prev])
        start = prev = idx
    ranges.append([start, prev])
    return ranges


def _read_single_member_zip(archive: Path) -> tuple[str, bytes]:
    with zipfile.ZipFile(archive, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        _validate_zip_container_integrity(archive, zf.infolist())
        if not infos:
            raise SystemExit(f"archive has no file members: {archive}")
        payload_infos = [info for info in infos if Path(info.filename).name == "0.bin"]
        if len(payload_infos) != 1:
            raise SystemExit(
                "--receiver-direct-cache requires exactly one archive member named "
                f"0.bin, got {[info.filename for info in infos]}"
            )
        info = payload_infos[0]
        return info.filename, zf.read(info)


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


def _is_relative_to(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _validate_archive_without_extracting(
    archive: Path,
    *,
    validate_members: bool = True,
) -> None:
    with zipfile.ZipFile(archive, "r") as zf:
        infos = zf.infolist()
        _validate_zip_container_integrity(archive, infos)
        if validate_members:
            _validate_archive_members([info.filename for info in infos])


def _require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise SystemExit(f"{label} missing: {path}")


def _run_inflate_png_frame_tree_permitted(
    inflate_sh: Path,
    archive_dir: Path,
    inflated_dir: Path,
    video_names_file: Path,
    *,
    timeout: int,
    extra_env: dict[str, str] | None,
) -> float:
    """Run inflate without contest-raw byte validation for advisory PNG-tree caches."""

    inflated_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "bash",
        str(inflate_sh),
        str(archive_dir),
        str(inflated_dir),
        str(video_names_file),
    ]
    env = {**os.environ}
    env.setdefault("PYTHON", sys.executable)
    env.setdefault("PYTHON_BIN", sys.executable)
    env.setdefault("PACT_PYTHON_BIN", sys.executable)
    env.setdefault("UV_PYTHON", sys.executable)
    if extra_env:
        env.update(extra_env)
    t0 = time.monotonic()
    try:
        result = run_in_process_group(cmd, timeout=timeout, check=False, env=env)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(
            f"[inflate/png-tree] TIMED OUT after {timeout}s while building an "
            "MLX advisory frame-tree cache"
        ) from exc
    elapsed = time.monotonic() - t0
    if result.returncode != 0:
        raise RuntimeError(f"[inflate/png-tree] FAILED with returncode={result.returncode}")
    return elapsed


def _single_inflated_surface_path(inflated_dir: Path, video_names_file: Path) -> Path:
    names = [line.strip() for line in video_names_file.read_text().splitlines() if line.strip()]
    if len(names) != 1:
        raise SystemExit(f"expected exactly one video for MLX cache materialization, got {len(names)}")
    raw_path = inflated_dir / Path(names[0]).with_suffix(".raw")
    if raw_path.is_file() or raw_path.is_dir():
        return raw_path
    stem_dir = inflated_dir / Path(names[0]).stem
    if stem_dir.is_dir():
        return stem_dir
    raise SystemExit(
        "inflated surface missing; expected raw file or PNG frame tree at "
        f"{raw_path} (or frame tree {stem_dir})"
    )


def _load_preinflated_proof_manifest(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"preinflated proof manifest is not a JSON object: {path}")
    return {
        "path": str(path),
        "sha256": _sha256(path, prefix=0),
        "payload": payload,
    }


def _validate_preinflated_png_tree_proof(
    proof: dict[str, Any],
    *,
    archive_sha256: str,
    video_names_file: Path,
    frame_tree: Path,
) -> None:
    payload = proof["payload"]
    proof_archive_sha = (
        payload.get("archive_sha256")
        or (payload.get("archive") or {}).get("sha256")
        or (payload.get("archive") or {}).get("archive_sha256")
    )
    if proof_archive_sha != archive_sha256:
        raise SystemExit(
            "preinflated PNG tree proof archive SHA mismatch: "
            f"{proof_archive_sha!r} != {archive_sha256!r}"
        )
    proof_passed = any(
        payload.get(key) is True
        for key in (
            "runtime_consumption_proof_passed",
            "receiver_contract_satisfied",
            "preinflated_output_proof_passed",
        )
    )
    if not proof_passed:
        raise SystemExit(
            "preinflated PNG tree proof must record a passing receiver/runtime proof"
        )
    proof_file_list = payload.get("file_list_path")
    if proof_file_list is not None:
        try:
            if Path(str(proof_file_list)).resolve() != video_names_file.resolve():
                raise SystemExit(
                    "preinflated PNG tree proof file_list_path does not match "
                    f"--video-names-file: {proof_file_list}"
                )
        except OSError as exc:
            raise SystemExit(f"invalid proof file_list_path: {proof_file_list}") from exc
    proof_output_path = payload.get("receiver_output_path") or payload.get("output_path")
    if proof_output_path is not None:
        try:
            if Path(str(proof_output_path)).resolve() != frame_tree.resolve():
                raise SystemExit(
                    "preinflated PNG tree proof receiver_output_path does not "
                    f"match frame tree: {proof_output_path}"
                )
        except OSError as exc:
            raise SystemExit(f"invalid proof receiver_output_path: {proof_output_path}") from exc


def _png_frame_paths(frame_tree: Path) -> list[Path]:
    paths_by_index: dict[int, Path] = {}
    unexpected: list[str] = []
    for path in frame_tree.iterdir():
        if not path.is_file() or path.suffix.lower() != ".png":
            unexpected.append(path.name)
            continue
        try:
            index = int(path.stem)
        except ValueError as exc:
            raise SystemExit(f"PNG frame name must be an integer stem: {path}") from exc
        if index in paths_by_index:
            raise SystemExit(f"duplicate PNG frame index {index} under {frame_tree}")
        paths_by_index[index] = path
    if unexpected:
        preview = sorted(unexpected)[:8]
        suffix = "" if len(unexpected) <= 8 else f" ... +{len(unexpected) - 8} more"
        raise SystemExit(
            f"PNG frame tree contains unexpected non-frame entries: {preview}{suffix}"
        )
    if not paths_by_index:
        raise SystemExit(f"PNG frame tree contains no .png frames: {frame_tree}")
    indices = sorted(paths_by_index)
    expected = list(range(indices[-1] + 1))
    if indices != expected:
        missing = sorted(set(expected) - set(indices))
        raise SystemExit(
            f"PNG frame tree must be contiguous from 0; missing {missing[:8]}"
        )
    if len(indices) % 2:
        raise SystemExit(
            f"PNG frame tree has odd frame count {len(indices)}; cannot form full pairs"
        )
    return [paths_by_index[idx] for idx in indices]


def _png_frame_shape_hwc(path: Path) -> tuple[int, int, int]:
    from PIL import Image  # type: ignore[import-not-found]

    with Image.open(path) as im:
        if im.mode != "RGB":
            raise SystemExit(f"PNG frame must be RGB, got mode={im.mode!r}: {path}")
        width, height = im.size
    return (int(height), int(width), 3)


def _read_png_rgb(path: Path, *, frame_shape_hwc: tuple[int, int, int]) -> np.ndarray:
    from PIL import Image  # type: ignore[import-not-found]

    with Image.open(path) as im:
        if im.mode != "RGB":
            raise SystemExit(f"PNG frame must be RGB, got mode={im.mode!r}: {path}")
        arr = np.asarray(im, dtype=np.uint8)
    if tuple(int(v) for v in arr.shape) != frame_shape_hwc:
        raise ValueError(
            f"PNG frame shape mismatch for {path}: got {arr.shape}, "
            f"expected {frame_shape_hwc}"
        )
    return arr


def _iter_png_pair_batches(
    frame_paths: list[Path],
    *,
    pair_count: int,
    batch_pairs: int,
    frame_shape_hwc: tuple[int, int, int],
):
    if len(frame_paths) != pair_count * 2:
        raise ValueError(
            f"expected {pair_count * 2} PNG frames for {pair_count} pairs, "
            f"got {len(frame_paths)}"
        )
    for pair_start in range(0, pair_count, batch_pairs):
        pair_end = min(pair_count, pair_start + batch_pairs)
        pairs = []
        for pair_idx in range(pair_start, pair_end):
            first = _read_png_rgb(
                frame_paths[2 * pair_idx],
                frame_shape_hwc=frame_shape_hwc,
            )
            second = _read_png_rgb(
                frame_paths[2 * pair_idx + 1],
                frame_shape_hwc=frame_shape_hwc,
            )
            pairs.append(np.stack([first, second], axis=0))
        yield np.stack(pairs, axis=0)


def _png_tree_cache_blockers(
    *,
    frame_shape_hwc: tuple[int, int, int],
    png_frame_count: int,
    cached_pair_count: int,
    expected_full_frames: int,
) -> list[str]:
    blockers: list[str] = []
    if frame_shape_hwc != (CAMERA_HW[0], CAMERA_HW[1], 3):
        blockers.append(
            "png_frame_tree_noncontest_raw_geometry_"
            f"{frame_shape_hwc[0]}x{frame_shape_hwc[1]}"
        )
    if int(png_frame_count) != int(expected_full_frames):
        blockers.append(
            f"png_frame_tree_frame_count_{png_frame_count}_not_{expected_full_frames}"
        )
    if int(cached_pair_count) * 2 != int(png_frame_count):
        blockers.append("png_frame_tree_cache_prefix_subset")
    return sorted(set(blockers))


def _record_png_tree_inflated_output_artifacts(
    prov: dict[str, Any],
    work_dir: Path,
    inflated_dir: Path,
    video_names_file: Path,
    frame_tree: Path,
    frame_paths: list[Path],
) -> dict[str, Any]:
    names = [n.strip() for n in video_names_file.read_text().splitlines() if n.strip()]
    rel_tree = frame_tree.relative_to(inflated_dir).as_posix()
    files = []
    for path in frame_paths:
        h, w, c = _png_frame_shape_hwc(path)
        files.append(
            {
                "relative_path": path.relative_to(inflated_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path, prefix=0),
                "mode": "RGB",
                "shape_hwc": [h, w, c],
            }
        )
    aggregate_payload: dict[str, Any] = {
        "schema": "mlx_advisory_png_frame_tree_inflated_output_manifest_v1",
        "inflated_dir": str(inflated_dir),
        "video_names_file": str(video_names_file),
        "surface_kind": "png_frame_tree",
        "video_count": len(names),
        "frame_tree_relative_path": rel_tree,
        "png_frame_count": len(frame_paths),
        "raw_file_count": 0,
        "total_bytes": sum(int(f["bytes"]) for f in files),
        "files": files,
        **FALSE_AUTHORITY,
    }
    aggregate_payload["aggregate_sha256"] = hashlib.sha256(
        json.dumps(
            {
                "surface_kind": "png_frame_tree",
                "frame_tree_relative_path": rel_tree,
                "files": [
                    {
                        "relative_path": f["relative_path"],
                        "bytes": f["bytes"],
                        "sha256": f["sha256"],
                    }
                    for f in files
                ],
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    manifest_path = work_dir / "inflated_outputs_manifest.json"
    manifest_path.write_text(json.dumps(aggregate_payload, indent=2, sort_keys=True) + "\n")
    prov["inflated_output_manifest"] = {
        "path": str(manifest_path),
        "sha256": _sha256(manifest_path, prefix=0),
        "payload": aggregate_payload,
    }
    (work_dir / "provenance.json").write_text(
        json.dumps(prov, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return aggregate_payload


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
