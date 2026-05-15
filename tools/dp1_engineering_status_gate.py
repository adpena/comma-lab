#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Emit the current DP1 engineering status and next executable gate.

The gate is intentionally stricter than a narrative ledger.  It reads the
planning, smoke, Tier-C, and tiny-full-run artifacts that currently define DP1
and fails closed if any artifact tries to claim score or promotion authority
without a real contest result.  Its default output is a machine-readable status
packet that names the next runnable gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_READINESS = Path("reports/cooperative_receiver/driving_prior_readiness.json")
DEFAULT_SMOKE_MANIFEST = Path("experiments/results/dp1_smoke_v2_hardening/manifest.json")
DEFAULT_TINY_FULL_MANIFEST = Path(
    "experiments/results/dp1_tiny_full_cpu_advisory_20260515_codex/manifest.json"
)
DEFAULT_TINY_FULL_PROVENANCE = Path(
    "experiments/results/dp1_tiny_full_cpu_advisory_20260515_codex/provenance.json"
)
DEFAULT_TIER_C = Path(
    "experiments/results/tier_c_real_scorer_fourway_codex_execute1_dp1fix_20260514/"
    "dp1_smoke_tier_c_real_scorer.json"
)

REAL_DP1_DATASET = "comma2k19"
ALLOWED_REAL_SOURCE_MODES = {
    "local_chunks",
    "local_cache",
    "stream_log",
    "prebuilt_codebook",
}
SOURCE_ROOT_TOKEN = "$DPP_COMMA2K19_CHUNKS_DIR"
DEFAULT_SOURCE_MAX_CHUNKS = 1
_PRIVATE_SOURCE_PATH_FIELDS = (
    "chunks_dir",
    "cache_dir",
    "stream_log_dir",
    "codebook_path",
)


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _stable_sha256_json(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sanitize_dataset_source_manifest(
    manifest: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Return a public-safe dataset-source manifest.

    The trainer provenance can legitimately carry local absolute source paths.
    The status gate output is meant to be copied into public ledgers, so path
    fields are reduced to set/unset evidence while relative chunk IDs and SHA
    coverage remain exact.
    """
    if manifest is None:
        return None
    public = json.loads(json.dumps(manifest))
    for field in _PRIVATE_SOURCE_PATH_FIELDS:
        value = public.get(field)
        if isinstance(value, str) and value and not value.startswith("$"):
            public[field] = f"<redacted:{field}:set>"
    return public


def _source_custody_preflight(
    source_chunks_dir: Path | None,
    *,
    source_max_chunks: int = DEFAULT_SOURCE_MAX_CHUNKS,
) -> dict[str, Any]:
    """Hash the real local chunks that the next DP1 timing smoke would consume.

    Public output records only paths relative to ``DPP_COMMA2K19_CHUNKS_DIR``.
    The private root is intentionally never serialized.
    """
    max_chunks = max(1, int(source_max_chunks))
    command = [
        "PYTHONPATH=src:upstream:$PWD",
        ".venv/bin/python",
        "experiments/train_substrate_pretrained_driving_prior.py",
        "--device",
        "cpu",
        "--full-cpu",
        "--advisory-cpu-explicitly-waived",
        "--dataset-name",
        "comma2k19",
        "--comma2k19-chunks-dir",
        SOURCE_ROOT_TOKEN,
        "--epochs",
        "1",
        "--batch-size",
        "1",
        "--max-pairs",
        "4",
        "--val-pair-count",
        "1",
        "--max-distillation-frames",
        "128",
        "--max-distillation-chunks",
        str(max_chunks),
        "--skip-auth-eval",
        "--output-dir",
        "experiments/results/dp1_comma2k19_onechunk_cpu_advisory_<UTC>",
    ]
    base: dict[str, Any] = {
        "schema": "dp1_source_custody_timing_smoke_preflight_v1",
        "dataset_name": REAL_DP1_DATASET,
        "source_mode": "local_chunks",
        "accepted_chunk_glob": "**/video.hevc",
        "path_kind": "relative_to_DPP_COMMA2K19_CHUNKS_DIR",
        "source_root_env": SOURCE_ROOT_TOKEN,
        "private_source_root_recorded": False,
        "source_max_chunks": max_chunks,
        "chunk_count": 0,
        "total_size_bytes": 0,
        "chunk_manifest": [],
        "chunk_manifest_sha256": None,
        "blockers": [],
        "next_run_command": command,
        "dataset_source_manifest": None,
    }
    if source_chunks_dir is None:
        base["status"] = "blocked_source_dir_not_configured"
        base["blockers"] = ["dp1_source_chunks_dir_not_configured"]
        return base

    blockers: list[str] = []
    try:
        from tac.substrates.pretrained_driving_prior.dataset_source import (
            collect_local_video_manifest,
        )

        rows = collect_local_video_manifest(
            Path(source_chunks_dir),
            max_files=max_chunks,
        )
    except FileNotFoundError:
        rows = []
        blockers.append("dp1_source_chunks_dir_missing")
    except OSError:
        rows = []
        blockers.append("dp1_source_chunks_dir_unreadable")

    public_rows = [
        {
            "relative_path": str(row["relpath"]),
            "size_bytes": int(row["bytes"]),
            "sha256": str(row["sha256"]),
        }
        for row in rows
    ]
    zero_byte_chunks = [
        row["relative_path"] for row in public_rows if int(row["size_bytes"]) <= 0
    ]
    if not public_rows and not blockers:
        blockers.append("dp1_real_chunks_missing_video_hevc")
    for relpath in zero_byte_chunks:
        blockers.append(f"dp1_real_chunk_zero_bytes:{relpath}")

    manifest_sha = _stable_sha256_json(public_rows) if public_rows else None
    base.update(
        {
            "status": "passed" if not blockers else "blocked",
            "chunk_count": len(public_rows),
            "total_size_bytes": sum(int(row["size_bytes"]) for row in public_rows),
            "chunk_manifest": public_rows,
            "chunk_manifest_sha256": manifest_sha,
            "blockers": blockers,
        }
    )
    if not blockers:
        chunk_ids = [row["relative_path"] for row in public_rows]
        chunk_sha256s = {
            row["relative_path"]: row["sha256"] for row in public_rows
        }
        base["dataset_source_manifest"] = {
            "schema": "dp1_dataset_source_manifest.v1",
            "dataset_name": REAL_DP1_DATASET,
            "source_mode": "local_chunks",
            "distillation_mode": "single_pass",
            "synthetic": False,
            "chunks_dir": SOURCE_ROOT_TOKEN,
            "chunk_ids": chunk_ids,
            "chunk_sha256_manifest": chunk_sha256s,
            "chunk_sha256_coverage": {
                "scope": "selected_chunks_only",
                "chunk_count": len(chunk_ids),
                "covered_count": len(chunk_ids),
                "complete_for_selected_chunks": bool(chunk_ids),
                "full_dataset_complete": False,
            },
            "local_video_manifest": [
                {
                    "relpath": row["relative_path"],
                    "bytes": row["size_bytes"],
                    "sha256": row["sha256"],
                }
                for row in public_rows
            ],
            "license_tags": ["comma2k19:MIT"],
            "dataset_provenance": "comma2k19_local_chunks_preflight",
            "reproducibility_blockers": [],
            "score_claim_allowed": False,
        }
    return base


def _artifact(path: Path, payload: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "path": str(path),
        "exists": payload is not None,
        "schema": payload.get("schema") if payload else None,
        "score_claim": payload.get("score_claim") if payload else None,
        "promotion_eligible": payload.get("promotion_eligible") if payload else None,
        "ready_for_exact_eval_dispatch": (
            payload.get("ready_for_exact_eval_dispatch") if payload else None
        ),
        "evidence_grade": payload.get("evidence_grade") if payload else None,
        "archive_bytes": payload.get("archive_bytes") if payload else None,
        "archive_sha256": payload.get("archive_sha256") if payload else None,
    }


def _finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _has_contest_score(payload: dict[str, Any]) -> bool:
    for key in (
        "contest_cuda_score",
        "contest_cpu_score",
        "score_recomputed",
        "canonical_score",
    ):
        if _finite_number(payload.get(key)):
            return True
    result = payload.get("result")
    if isinstance(result, dict):
        for key in ("contest_cuda_score", "contest_cpu_score", "score_recomputed"):
            if _finite_number(result.get(key)):
                return True
    archive_result = payload.get("archive_result")
    if isinstance(archive_result, dict):
        for key in ("score_recomputed", "canonical_score"):
            if _finite_number(archive_result.get(key)):
                return True
    return False


def _score_axis(payload: dict[str, Any]) -> str | None:
    for key in ("score_axis_tag", "axis", "hardware_axis", "evidence_axis"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return None


def _dataset_source_manifest(payloads: list[dict[str, Any] | None]) -> dict[str, Any] | None:
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        direct = payload.get("dataset_source_manifest")
        if isinstance(direct, dict):
            return direct
        codebook = payload.get("codebook_provenance")
        if isinstance(codebook, dict):
            nested = codebook.get("dataset_source_manifest")
            if isinstance(nested, dict):
                return nested
    return None


def _dataset_source_status(
    manifest: dict[str, Any] | None,
) -> tuple[bool, list[str], dict[str, Any] | None]:
    if manifest is None:
        return False, ["dp1_real_dataset_source_manifest_missing"], None
    dataset_name = str(manifest.get("dataset_name", ""))
    source_mode = str(manifest.get("source_mode", ""))
    blockers = [str(item) for item in manifest.get("reproducibility_blockers", [])]
    if dataset_name != REAL_DP1_DATASET:
        blockers.append(f"dp1_real_dataset_required:{REAL_DP1_DATASET}")
    if source_mode not in ALLOWED_REAL_SOURCE_MODES:
        blockers.append(f"dp1_real_source_mode_invalid:{source_mode or '<missing>'}")
    coverage = manifest.get("chunk_sha256_coverage")
    if source_mode != "prebuilt_codebook":
        selected_complete = (
            isinstance(coverage, dict)
            and coverage.get("complete_for_selected_chunks") is True
        )
        if not selected_complete:
            blockers.append(
                "dp1_real_dataset_selected_chunk_sha256_coverage_incomplete"
            )
    return not blockers, blockers, manifest


def _false_claims(artifacts: dict[str, dict[str, Any] | None]) -> list[str]:
    problems: list[str] = []
    for label, payload in artifacts.items():
        if not isinstance(payload, dict):
            continue
        path = payload.get("_artifact_path", label)
        has_score = _has_contest_score(payload)
        if payload.get("score_claim") is True and not has_score:
            problems.append(f"{path}: score_claim=true without contest score field")
        if payload.get("score_claim_valid") is True and not has_score:
            problems.append(f"{path}: score_claim_valid=true without contest score field")
        if payload.get("promotion_eligible") is True:
            problems.append(f"{path}: DP1 artifact claims promotion_eligible=true")
        if payload.get("rank_or_kill_eligible") is True and not has_score:
            problems.append(f"{path}: rank_or_kill_eligible=true without contest score")
        blockers = payload.get("dispatch_blockers")
        if (
            payload.get("ready_for_exact_eval_dispatch") is True
            and isinstance(blockers, list)
            and blockers
        ):
            problems.append(f"{path}: ready_for_exact_eval_dispatch=true with blockers")
    return problems


def _next_gate(
    *,
    real_source_ready: bool,
    real_source_blockers: list[str],
    exact_scores_present: bool,
    source_custody_preflight: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if exact_scores_present:
        return {
            "id": "dp1_result_review_packet",
            "status": "ready_after_exact_score_artifact",
            "purpose": "Build a DP1 result-review packet before status changes.",
            "commands": [
                [
                    "PYTHONPATH=src",
                    ".venv/bin/python",
                    "tools/build_result_review_packet.py",
                    "--help",
                ]
            ],
        }
    source_manifest_sha = None
    if isinstance(source_custody_preflight, dict):
        source_manifest_sha = source_custody_preflight.get("chunk_manifest_sha256")
    command = [
        "PYTHONPATH=src:upstream:$PWD",
        ".venv/bin/python",
        "experiments/train_substrate_pretrained_driving_prior.py",
        "--device",
        "cpu",
        "--full-cpu",
        "--advisory-cpu-explicitly-waived",
        "--dataset-name",
        "comma2k19",
        "--comma2k19-chunks-dir",
        "$DPP_COMMA2K19_CHUNKS_DIR",
        "--epochs",
        "1",
        "--batch-size",
        "1",
        "--max-pairs",
        "4",
        "--val-pair-count",
        "1",
        "--max-distillation-frames",
        "128",
        "--max-distillation-chunks",
        "1",
        "--skip-auth-eval",
        "--output-dir",
        "experiments/results/dp1_comma2k19_onechunk_cpu_advisory_<UTC>",
    ]
    return {
        "id": "dp1_comma2k19_onechunk_cpu_advisory_source_custody",
        "status": "blocked_until_source_supplied"
        if not real_source_ready
        else "executable_real_source_probe",
        "purpose": (
            "Produce the first real-source DP1 training/runtime custody artifact "
            "without a score claim."
        ),
        "blocked_by": real_source_blockers,
        "commands": [command],
        "success_criteria": [
            "dataset_source_manifest.schema=dp1_dataset_source_manifest.v1",
            "dataset_name=comma2k19",
            "chunk_sha256_coverage.complete_for_selected_chunks=true or prebuilt_codebook source mode",
            "source_custody_preflight.status=passed for local chunks",
            "score_claim=false",
            "promotion_eligible=false",
            "archive.zip and provenance.json written",
        ],
        "source_root_env": SOURCE_ROOT_TOKEN,
        "verified_source_chunk_manifest_sha256": source_manifest_sha,
    }


def build_status(
    *,
    readiness_path: Path = DEFAULT_READINESS,
    smoke_manifest_path: Path = DEFAULT_SMOKE_MANIFEST,
    tiny_full_manifest_path: Path = DEFAULT_TINY_FULL_MANIFEST,
    tiny_full_provenance_path: Path = DEFAULT_TINY_FULL_PROVENANCE,
    tier_c_path: Path = DEFAULT_TIER_C,
    source_chunks_dir: Path | None = None,
    source_max_chunks: int = DEFAULT_SOURCE_MAX_CHUNKS,
) -> dict[str, Any]:
    loaded: dict[str, dict[str, Any] | None] = {
        "readiness": _read_json(readiness_path),
        "smoke_manifest": _read_json(smoke_manifest_path),
        "tiny_full_manifest": _read_json(tiny_full_manifest_path),
        "tiny_full_provenance": _read_json(tiny_full_provenance_path),
        "tier_c": _read_json(tier_c_path),
    }
    paths = {
        "readiness": readiness_path,
        "smoke_manifest": smoke_manifest_path,
        "tiny_full_manifest": tiny_full_manifest_path,
        "tiny_full_provenance": tiny_full_provenance_path,
        "tier_c": tier_c_path,
    }
    for label, payload in loaded.items():
        if isinstance(payload, dict):
            payload["_artifact_path"] = str(paths[label])

    false_claims = _false_claims(loaded)
    source_custody = _source_custody_preflight(
        source_chunks_dir,
        source_max_chunks=source_max_chunks,
    )
    dataset_manifest = (
        source_custody.get("dataset_source_manifest")
        if source_chunks_dir is not None
        else _dataset_source_manifest(list(loaded.values()))
    )
    real_source_ready, source_blockers, normalized_source = _dataset_source_status(
        dataset_manifest
    )
    if source_chunks_dir is not None and source_custody.get("blockers"):
        source_blockers = list(
            dict.fromkeys(
                [
                    *[str(item) for item in source_custody.get("blockers", [])],
                    *source_blockers,
                ]
            )
        )
    exact_scores_present = any(
        _has_contest_score(payload)
        for payload in loaded.values()
        if isinstance(payload, dict)
    )
    smoke_ready = bool(
        loaded["smoke_manifest"] and loaded["smoke_manifest"].get("archive_path")
    )
    tier_c_ready = bool(
        loaded["tier_c"]
        and loaded["tier_c"].get("score_claim") is False
        and loaded["tier_c"].get("promotion_eligible") is False
    )
    tiny_full_ran = bool(
        loaded["tiny_full_provenance"]
        and loaded["tiny_full_provenance"].get("trainer")
        == "experiments/train_substrate_pretrained_driving_prior.py"
    )
    if false_claims:
        status = "blocked_false_dp1_score_or_promotion_claim"
    elif not real_source_ready:
        status = "implemented_proxy_ready_untrained_real_prior_missing"
    elif not exact_scores_present:
        status = "real_source_probe_ready_no_exact_score"
    else:
        status = "exact_score_artifact_present_needs_result_review"

    return {
        "schema": "dp1_engineering_status_gate_v1",
        "generated_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "lane_family": "pretrained_driving_prior_dp1",
        "engineering_status": status,
        "classification": "untrained_unpromoted_promising_substrate",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
        "capabilities_confirmed": {
            "planning_manifest_present": loaded["readiness"] is not None,
            "smoke_archive_materialized": smoke_ready,
            "tier_c_advisory_present": tier_c_ready,
            "tiny_full_cpu_advisory_ran": tiny_full_ran,
            "real_dataset_source_ready": real_source_ready,
            "real_source_custody_preflight_passed": (
                source_custody.get("status") == "passed"
            ),
            "exact_score_artifact_present": exact_scores_present,
        },
        "blockers": [
            *source_blockers,
            "dp1_trained_real_prior_missing",
            "dp1_paired_contest_cpu_cuda_exact_eval_missing",
        ],
        "false_claims": false_claims,
        "artifact_summary": {
            label: _artifact(paths[label], payload)
            for label, payload in loaded.items()
        },
        "source_custody_preflight": {
            key: value
            for key, value in source_custody.items()
            if key != "dataset_source_manifest"
        },
        "score_axes_observed": [
            axis
            for axis in (
                _score_axis(payload)
                for payload in loaded.values()
                if isinstance(payload, dict)
            )
            if axis
        ],
        "dataset_source_manifest": _sanitize_dataset_source_manifest(normalized_source),
        "next_gate": _next_gate(
            real_source_ready=real_source_ready,
            real_source_blockers=source_blockers,
            exact_scores_present=exact_scores_present,
            source_custody_preflight=source_custody,
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--readiness-json", type=Path, default=DEFAULT_READINESS)
    parser.add_argument("--smoke-manifest", type=Path, default=DEFAULT_SMOKE_MANIFEST)
    parser.add_argument(
        "--tiny-full-manifest",
        type=Path,
        default=DEFAULT_TINY_FULL_MANIFEST,
    )
    parser.add_argument(
        "--tiny-full-provenance",
        type=Path,
        default=DEFAULT_TINY_FULL_PROVENANCE,
    )
    parser.add_argument("--tier-c-json", type=Path, default=DEFAULT_TIER_C)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit nonzero when a false score/promotion claim is found.",
    )
    parser.add_argument(
        "--require-real-source",
        action="store_true",
        help="Exit nonzero until the DP1 real Comma2k19 source manifest is ready.",
    )
    parser.add_argument(
        "--comma2k19-chunks-dir",
        type=Path,
        help=(
            "Private local Comma2k19/test_videos chunks root. The public JSON "
            "records only paths relative to $DPP_COMMA2K19_CHUNKS_DIR."
        ),
    )
    parser.add_argument(
        "--source-max-chunks",
        type=int,
        default=DEFAULT_SOURCE_MAX_CHUNKS,
        help="Number of deterministic video.hevc chunks to hash for the timing smoke.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    source_chunks_dir = args.comma2k19_chunks_dir
    if source_chunks_dir is None:
        env_chunks_dir = os.environ.get("DPP_COMMA2K19_CHUNKS_DIR", "").strip()
        if env_chunks_dir:
            source_chunks_dir = Path(env_chunks_dir)
    status = build_status(
        readiness_path=args.readiness_json,
        smoke_manifest_path=args.smoke_manifest,
        tiny_full_manifest_path=args.tiny_full_manifest,
        tiny_full_provenance_path=args.tiny_full_provenance,
        tier_c_path=args.tier_c_json,
        source_chunks_dir=source_chunks_dir,
        source_max_chunks=args.source_max_chunks,
    )
    rendered = json.dumps(status, indent=2, sort_keys=True) + "\n"
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered, encoding="utf-8")
    print(
        json.dumps(
            {
                "schema": status["schema"],
                "engineering_status": status["engineering_status"],
                "classification": status["classification"],
                "false_claims": len(status["false_claims"]),
                "source_custody_preflight": status["source_custody_preflight"][
                    "status"
                ],
                "source_chunk_count": status["source_custody_preflight"][
                    "chunk_count"
                ],
                "next_gate": status["next_gate"]["id"],
                "next_gate_status": status["next_gate"]["status"],
                "output_json": str(args.output_json) if args.output_json else None,
            },
            sort_keys=True,
        )
    )
    if args.strict and status["false_claims"]:
        return 2
    if args.require_real_source and not status["capabilities_confirmed"][
        "real_dataset_source_ready"
    ]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
