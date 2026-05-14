#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# ruff: noqa: I001
"""Profile PR79/PR77 local lossless-packer candidates and emit one S3 recommendation.

This tool performs local archive-byte screening only. It does not submit
remote jobs, does not run GPU eval, and does not make score claims. The
recommended candidate must preserve decoded PR79 action records and all
non-action runtime streams under the current robust-current payload parser.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.submission_archive import validate_archive_seg_tile_actions_payloads


PR79_S1_BUILDER = REPO_ROOT / "experiments/build_pr79_action_lossless_repack_candidates.py"
PR79_S2_BUILDER = REPO_ROOT / "experiments/build_pr79_action_dictionary_repack_candidates_v2.py"
DEFAULT_PR79_ARCHIVE = (
    REPO_ROOT / "experiments/results/top_submission_reverse_engineering_20260503_pr79/archive.zip"
)
DEFAULT_PR77_MIXED_MATRIX = (
    REPO_ROOT / "experiments/results/pr77_action_pose_mixed_container_20260503_codex/candidate_matrix.json"
)
DEFAULT_PR77_TRANSPLANT_MATRIX = (
    REPO_ROOT
    / "experiments/results/pr77_tile_action_transplant_stream_mix_20260503_worker/candidate_matrix.json"
)
DEFAULT_FLATPACK_MATRIX = (
    REPO_ROOT / "experiments/results/pr79_flatpack_transfer_worker_20260503/candidate_matrix.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "experiments/results/pr79_pr77_lossless_s3_profile_20260503_codex"
)

TOOL = "experiments/profile_pr79_pr77_lossless_s3_candidates.py"
SCHEMA = "pr79_pr77_lossless_s3_profile_v1"
RECOMMENDATION_SCHEMA = "pr79_pr77_lossless_s3_recommendation_v1"
CUDA_AUTH_EVAL_REQUIRED = (
    "No dispatch from this profiler. Before exact eval, claim a non-conflicting "
    "lane with tools/claim_lane_dispatch.py claim, then run exact CUDA auth eval "
    "on the identical archive bytes through archive.zip -> inflate.sh -> "
    "upstream/evaluate.py via experiments/contest_auth_eval.py --device cuda."
)


def _load_module(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def _repo_rel(path: Path | str | None) -> str | None:
    if path is None:
        return None
    path = Path(path)
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _zip_profile(path: Path) -> dict[str, Any]:
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        duplicate_names = sorted(
            name for name in {info.filename for info in infos} if sum(i.filename == name for i in infos) > 1
        )
        members = [
            {
                "compress_size": info.compress_size,
                "compress_type": info.compress_type,
                "date_time": list(info.date_time),
                "external_attr": info.external_attr,
                "filename": info.filename,
                "file_size": info.file_size,
            }
            for info in infos
        ]
    single_stored_p = (
        len(members) == 1
        and members[0]["filename"] == "p"
        and int(members[0]["compress_type"]) == zipfile.ZIP_STORED
    )
    return {
        "archive_bytes": path.stat().st_size,
        "archive_sha256": _sha256_file(path),
        "duplicate_member_names": duplicate_names,
        "member_count": len(members),
        "members": members,
        "single_stored_p": single_stored_p,
        "strict_zip_overhead_bytes": (
            path.stat().st_size - int(members[0]["file_size"])
            if single_stored_p
            else None
        ),
    }


def _manifest_path(row: Mapping[str, Any]) -> Path:
    raw = row.get("manifest_path")
    if not raw:
        raise ValueError(f"candidate row has no manifest_path: {row}")
    path = Path(str(raw))
    return path if path.is_absolute() else REPO_ROOT / path


def _archive_path(row: Mapping[str, Any]) -> Path:
    raw = row.get("archive_path")
    if not raw:
        raise ValueError(f"candidate row has no archive_path: {row}")
    path = Path(str(raw))
    return path if path.is_absolute() else REPO_ROOT / path


def _normalise_pr79_candidate(row: Mapping[str, Any]) -> dict[str, Any]:
    manifest = _read_json(_manifest_path(row))
    archive_path = _archive_path(row)
    runtime_validation = manifest.get("runtime_parse_validation", {})
    no_op = manifest.get("no_op_detection", {})
    zip_profile = _zip_profile(archive_path)
    archive_validation_errors = validate_archive_seg_tile_actions_payloads(archive_path)
    action_parity = bool(runtime_validation.get("action_record_parity"))
    non_action_preserved = bool(runtime_validation.get("non_action_streams_preserved"))
    payload_changed = not bool(no_op.get("payload_sha_equal_to_source"))
    archive_changed = not bool(no_op.get("archive_sha_equal_to_source"))
    byte_improvement = int(row["delta_bytes_vs_pr79"]) < 0
    exact_eval_ready_after_lane_claim = (
        byte_improvement
        and action_parity
        and non_action_preserved
        and payload_changed
        and archive_changed
        and not archive_validation_errors
        and zip_profile["single_stored_p"]
        and not zip_profile["duplicate_member_names"]
    )
    return {
        "archive_bytes": int(row["archive_bytes"]),
        "archive_path": _repo_rel(archive_path),
        "archive_sha256": str(row["archive_sha256"]),
        "archive_validation_errors": archive_validation_errors,
        "candidate_id": str(row["candidate_id"]),
        "decoded_semantics_proof": {
            "action_record_parity": action_parity,
            "decoded_action_sha256": runtime_validation.get("decoded_action_sha256"),
            "non_action_streams_preserved": non_action_preserved,
            "runtime_parser": runtime_validation.get("runtime_parser"),
        },
        "delta_bytes_vs_pr79": int(row["delta_bytes_vs_pr79"]),
        "exact_eval_ready_after_lane_claim": exact_eval_ready_after_lane_claim,
        "manifest_path": _repo_rel(_manifest_path(row)),
        "no_op_status": str(row.get("no_op_status", no_op.get("status", "unknown"))),
        "payload_bytes": int(row.get("payload_bytes", manifest.get("payload", {}).get("bytes", 0))),
        "score_claim": False,
        "stream_packing": manifest.get("stream_packing", {}),
        "zip_profile": zip_profile,
    }


def _pr79_source_row(matrix: Mapping[str, Any]) -> dict[str, Any]:
    source = dict(matrix["source_archive"])
    return {
        "archive_bytes": int(source["bytes"]),
        "archive_path": source.get("path"),
        "archive_sha256": str(source["sha256"]),
        "candidate_id": "source_pr79_noop_control",
        "delta_bytes_vs_pr79": 0,
        "score_claim": False,
        "status": "source_reference",
    }


def _run_pr79_action_builders(
    *,
    pr79_archive: Path,
    output_dir: Path,
    force: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    s1_builder = _load_module(PR79_S1_BUILDER, "pr79_s3_s1_builder")
    s2_builder = _load_module(PR79_S2_BUILDER, "pr79_s3_s2_builder")
    s1_matrix = s1_builder.build_candidates(
        pr79_archive=pr79_archive,
        output_dir=output_dir / "pr79_s1",
        force=force,
    )
    s2_matrix = s2_builder.build_candidates(
        pr79_archive=pr79_archive,
        output_dir=output_dir / "pr79_s2",
        force=force,
    )
    rows = []
    for matrix in (s1_matrix, s2_matrix):
        for row in matrix["byte_matrix"]:
            if str(row.get("candidate_id", "")).startswith("source_") or "manifest_path" not in row:
                continue
            rows.append(_normalise_pr79_candidate(row))
    return rows, {
        "s1": s1_matrix,
        "s2": s2_matrix,
        "source_pr79": _pr79_source_row(s2_matrix),
    }


def _summarise_pr77_matrix(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "matrix_path": _repo_rel(path),
            "status": "missing",
        }
    matrix = _read_json(path)
    rows = list(matrix.get("candidates", []))
    if not rows:
        return {
            "candidate_count": 0,
            "matrix_path": _repo_rel(path),
            "status": "empty",
        }
    best = min(rows, key=lambda row: int(row.get("archive_bytes", 10**12)))
    semantic_contract = str(best.get("semantic_contract", best.get("semantic_status", "")))
    strict_lossless = (
        semantic_contract == "pr77_non_action_streams_identical"
        and int(best.get("delta_bytes_vs_pr77", 0)) < 0
        and "noop" not in str(best.get("noop_status", ""))
    )
    return {
        "best_archive_bytes": int(best.get("archive_bytes", 0)),
        "best_candidate_id": str(best.get("candidate_id", "")),
        "best_delta_bytes_vs_pr77": best.get("delta_bytes_vs_pr77"),
        "best_manifest_path": _repo_rel(best.get("manifest_path")),
        "candidate_count": len(rows),
        "lossless_profile_note": (
            "PR77 context only: candidates may be exact-eval-ready mixed-stream rows, "
            "but they are not selected as PR79 decoded-action-preserving lossless repacks."
        ),
        "matrix_path": _repo_rel(path),
        "score_claim": False,
        "status": "profiled",
        "strict_lossless_selected": strict_lossless,
    }


def _summarise_flatpack_matrix(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"matrix_path": _repo_rel(path), "status": "missing"}
    matrix = _read_json(path)
    rows = list(matrix.get("candidates", []))
    if not rows:
        return {"candidate_count": 0, "matrix_path": _repo_rel(path), "status": "empty"}
    best = min(rows, key=lambda row: int(row.get("archive_bytes", 10**12)))
    return {
        "best_archive_bytes": int(best.get("archive_bytes", 0)),
        "best_candidate_id": str(best.get("candidate_id", "")),
        "best_delta_bytes_vs_source": best.get("delta_bytes_vs_source"),
        "best_manifest_path": _repo_rel(best.get("manifest_path")),
        "candidate_count": len(rows),
        "matrix_path": _repo_rel(path),
        "status": "profiled",
        "takeaway": (
            "runtime-closed RPK1 flatpack transfer was byte-regressive in the "
            "available matrix; not recommended as a lossless S3 candidate"
        ),
    }


def _select_recommendation(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    ready = [
        dict(candidate)
        for candidate in candidates
        if bool(candidate.get("exact_eval_ready_after_lane_claim"))
    ]
    if not ready:
        return {
            "candidate": None,
            "decision": "no_lossless_byte_reduction_candidate",
            "reason": "no candidate passed byte, no-op, archive, and decoded-semantics gates",
        }
    best = min(ready, key=lambda row: (int(row["archive_bytes"]), str(row["candidate_id"])))
    return {
        "candidate": best,
        "decision": "recommend_exact_cuda_eval_after_lane_claim",
        "reason": (
            "best local PR79 lossless action repack: archive bytes decrease, "
            "decoded action records and non-action streams are preserved, and "
            "archive validation is clean"
        ),
    }


def build_profile(
    *,
    pr79_archive: Path = DEFAULT_PR79_ARCHIVE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    pr77_mixed_matrix: Path = DEFAULT_PR77_MIXED_MATRIX,
    pr77_transplant_matrix: Path = DEFAULT_PR77_TRANSPLANT_MATRIX,
    flatpack_matrix: Path = DEFAULT_FLATPACK_MATRIX,
    force: bool = False,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates, source_context = _run_pr79_action_builders(
        pr79_archive=pr79_archive,
        output_dir=output_dir,
        force=force,
    )
    recommendation = _select_recommendation(candidates)
    profile = {
        "canonical_score_source_required": CUDA_AUTH_EVAL_REQUIRED,
        "candidates": sorted(
            candidates,
            key=lambda row: (int(row["archive_bytes"]), str(row["candidate_id"])),
        ),
        "evidence_grade": "empirical_lossless_byte_screen",
        "flatpack_transfer_context": _summarise_flatpack_matrix(flatpack_matrix),
        "no_remote_dispatch_performed": True,
        "pr77_mixed_context": _summarise_pr77_matrix(pr77_mixed_matrix),
        "pr77_tile_delta_context": _summarise_pr77_matrix(pr77_transplant_matrix),
        "recommendation": recommendation,
        "schema": SCHEMA,
        "score_claim": False,
        "source_context": source_context,
        "s3_checks": {
            "action_entropy_coding": "PR79 S2 adaptive arithmetic action stream profiled",
            "deterministic_compression_params": "captured per candidate manifest",
            "flatpack_transfer": "profiled from existing matrix when present",
            "joint_brotli_order": "profiled by flatpack transfer matrix when present",
            "pr77_tile_delta_lossless_repack": "profiled from PR77 matrices when present",
            "zip_container_overhead": "strict single stored p archives retain a 100-byte ZIP floor",
        },
        "tool": TOOL,
    }
    _write_json(output_dir / "candidate_matrix.json", profile)
    _write_json(
        output_dir / "recommendation.json",
        {
            "canonical_score_source_required": CUDA_AUTH_EVAL_REQUIRED,
            "evidence_grade": profile["evidence_grade"],
            "no_remote_dispatch_performed": True,
            "recommendation": recommendation,
            "schema": RECOMMENDATION_SCHEMA,
            "score_claim": False,
            "tool": TOOL,
        },
    )
    return profile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr79-archive", type=Path, default=DEFAULT_PR79_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pr77-mixed-matrix", type=Path, default=DEFAULT_PR77_MIXED_MATRIX)
    parser.add_argument("--pr77-transplant-matrix", type=Path, default=DEFAULT_PR77_TRANSPLANT_MATRIX)
    parser.add_argument("--flatpack-matrix", type=Path, default=DEFAULT_FLATPACK_MATRIX)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    profile = build_profile(
        pr79_archive=args.pr79_archive,
        output_dir=args.output_dir,
        pr77_mixed_matrix=args.pr77_mixed_matrix,
        pr77_transplant_matrix=args.pr77_transplant_matrix,
        flatpack_matrix=args.flatpack_matrix,
        force=bool(args.force),
    )
    print(json.dumps(profile["recommendation"], indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
