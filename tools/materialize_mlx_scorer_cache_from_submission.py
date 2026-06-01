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
import os
import shutil
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
            "matching --video-names-file and must not live inside --work-dir."
        ),
    )
    parser.add_argument("--inflate-timeout", type=int, default=1800)
    parser.add_argument("--batch-pairs", type=int, default=1)
    parser.add_argument("--max-pairs", type=int)
    parser.add_argument(
        "--pair-ranges",
        help=(
            "For --hprc-direct-cache, render only explicit pair indices/ranges "
            "such as 1-2,4,9-12. The manifest preserves source pair indices so "
            "MLX response can align the subset against the full reference cache."
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
    if args.pair_ranges and not args.hprc_direct_cache:
        raise SystemExit("--pair-ranges currently requires --hprc-direct-cache")
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
    if preinflated_output_dir is not None:
        if args.hprc_direct_cache:
            raise SystemExit("--hprc-direct-cache cannot be combined with --preinflated-output-dir")
        if not preinflated_output_dir.is_dir():
            raise SystemExit(
                f"--preinflated-output-dir is not a directory: {preinflated_output_dir}"
            )
        if _is_relative_to(preinflated_output_dir, work_dir):
            raise SystemExit(
                "--preinflated-output-dir must not be inside --work-dir; "
                f"work-dir cleanup would destroy the source: {preinflated_output_dir}"
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
    inflate_executed = preinflated_output_dir is None and not args.hprc_direct_cache
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
    direct_cache_report: dict[str, Any] | None = None
    if args.hprc_direct_cache:
        _validate_archive_without_extracting(archive)
        inflate_elapsed = 0.0
    elif inflate_executed:
        members = _extract_archive(archive, extracted_dir)
        _validate_archive_members(members)
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
        "archive_sha256": _sha256(archive, prefix=0),
        "archive_size_bytes": archive.stat().st_size,
        "archive_path": str(archive),
        "submission_dir": str(submission_dir),
        "inflate_sh": str(inflate_sh),
        "video_names_file": str(video_names_file),
        "inflate_executed": inflate_executed,
        "preinflated_output_dir": (
            str(preinflated_output_dir) if preinflated_output_dir is not None else None
        ),
        **FALSE_AUTHORITY,
    }
    if args.hprc_direct_cache:
        direct_cache_report, manifest = _write_hprc_direct_cache(
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
        inflated_manifest = _record_inflated_output_artifacts(
            provenance,
            work_dir,
            inflated_dir,
            video_names_file,
        )
        raw_path = _single_raw_path(inflated_dir, video_names_file)
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
    if not args.hprc_direct_cache:
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
            None if args.hprc_direct_cache else str(work_dir / "inflated_outputs_manifest.json")
        ),
        "inflated_outputs_aggregate_sha256": inflated_manifest["aggregate_sha256"],
        "raw_path": str(raw_path) if raw_path is not None else None,
        "raw_sha256": manifest.get("raw_sha256"),
        "raw_pair_count": int(raw_pair_count),
        "cached_pair_count": int(manifest["pair_count"]),
        "pair_count": int(manifest["pair_count"]),
        "max_pairs": args.max_pairs,
        "local_acquisition_max_pairs": args.local_acquisition_max_pairs,
        "local_acquisition_partial_raw": local_acquisition_partial_raw,
        "local_acquisition_full_raw_pair_floor": 600,
        "inflate_executed": inflate_executed,
        "hprc_direct_cache": bool(args.hprc_direct_cache),
        "hprc_direct_cache_report": direct_cache_report,
        "preinflated_output_dir": (
            str(preinflated_output_dir) if preinflated_output_dir is not None else None
        ),
        "inflate_elapsed_seconds": float(inflate_elapsed),
        "elapsed_seconds": time.time() - started,
        "candidate_cache_identity_mode": "unaudited_candidate_generation_prior",
        "cpu_score_computed": False,
        "mlx_first_acquisition_only": True,
        "requires_local_cpu_gate_before_exact_auth": True,
        **FALSE_AUTHORITY,
    }
    write_json(report_output, report)
    if not args.keep_raw and inflate_executed and not args.hprc_direct_cache:
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


def _write_hprc_direct_cache(
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
    write_json(output_cache / "manifest.json", manifest)
    report = {
        "schema": "hprc_direct_mlx_scorer_cache_render.v1",
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
        "candidate_cache_identity_mode": "hprc_direct_receiver_render_advisory",
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
                "--hprc-direct-cache requires exactly one archive member named "
                f"0.bin, got {[info.filename for info in infos]}"
            )
        info = payload_infos[0]
        _validate_archive_members([item.filename for item in infos])
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


def _validate_archive_without_extracting(archive: Path) -> None:
    with zipfile.ZipFile(archive, "r") as zf:
        infos = zf.infolist()
        _validate_zip_container_integrity(archive, infos)
        _validate_archive_members([info.filename for info in infos])


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
