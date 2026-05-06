#!/usr/bin/env python3
# pyc-recovery pass2: rehydrated from git blob a828277b8e87cfeb24d086eb73c0d01f82aef455 via `git fsck --lost-found`
# original path: experiments/results/renderer_selfcompression_nextwave_worker_20260503/c101_combined_zero_search.py
# OUR source dropped during commit 66c59aae filter-repo cleanup; .pyc was sole orphan left.
# Blob verified intact + parses cleanly with python ast.
# Recovered: 2026-05-05 by Sherlock pass2
"""C-101 local combined QZS3 threshold-zero renderer shrink screen.

This worker-local helper preserves every non-renderer logical member from the
C-101 source archive, applies small combined FP4 zeroing transforms to the
QZS3 renderer, and runs the existing renderer transplant pose-safety preflight.
It does not dispatch remote work and makes no score claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
for _path in (REPO_ROOT / "src", REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from experiments import build_renderer_shrink_candidate as pr75_builder  # noqa: E402
from experiments import preflight_renderer_transplant_pose_safety as pose_safety  # noqa: E402
from experiments import search_renderer_parity_shrink_candidate as shrink_search  # noqa: E402


SCHEMA = "c101_combined_zero_renderer_shrink_screen_v1"
RATE_SCORE_PER_BYTE = 25.0 / 37_545_489.0
C101_SCORE = 0.3151520345392486
C101_BYTES = 276_489
C101_SHA256 = "1c9be2dd14b2607b9a86ecc071d6a37842896d18d62449885dac58d55afbbd64"
DEFAULT_SOURCE_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/lightning_batch/"
    "exact_eval_c091_native_pose_manifold_top128_s025_t4_20260503T122013Z/archive.zip"
)
DEFAULT_SOURCE_EVIDENCE = DEFAULT_SOURCE_ARCHIVE.with_name(
    "contest_auth_eval.adjudicated.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "experiments/results/renderer_selfcompression_nextwave_worker_20260503/"
    "c101_combined_zero_screen"
)


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slug(value: str) -> str:
    return (
        value.replace(":", "_")
        .replace(",", "_")
        .replace(".", "p")
        .replace("-", "_")
    )


def _parse_transform(raw: str) -> tuple[str, float]:
    prefix, threshold = raw.split(":", 1)
    parsed = float(threshold)
    if not prefix or parsed < 0.0 or parsed > 1.0 or not math.isfinite(parsed):
        raise argparse.ArgumentTypeError(f"invalid transform {raw!r}")
    return prefix, parsed


def _parse_candidate(raw: str) -> tuple[str, tuple[tuple[str, float], ...]]:
    if "=" in raw:
        name, spec = raw.split("=", 1)
    else:
        name, spec = _slug(raw), raw
    transforms = tuple(_parse_transform(item) for item in spec.split(",") if item)
    if not transforms:
        raise argparse.ArgumentTypeError(f"empty candidate spec {raw!r}")
    return name, transforms


def _default_candidates() -> tuple[tuple[str, tuple[tuple[str, float], ...]], ...]:
    raw = (
        "f1_0135_all002=frame1_head:0.135,all_fp4:0.02",
        "f1_0135_all003=frame1_head:0.135,all_fp4:0.03",
        "f1_0135_f2h003=frame1_head:0.135,frame2_head:0.03",
        "f1_0135_f2h005=frame1_head:0.135,frame2_head:0.05",
        "f1_0135_f2pre005=frame1_head:0.135,frame2_head.pre:0.05",
        "f1_0135_f2block006=frame1_head:0.135,frame2_head.block2:0.06",
        "f1_0135_shared004=frame1_head:0.135,shared_trunk:0.04",
        "f1_013_f2h005=frame1_head:0.13,frame2_head:0.05",
        "f1_013_all004=frame1_head:0.13,all_fp4:0.04",
        "f1_0125_f2h005=frame1_head:0.125,frame2_head:0.05",
        "f1_0125_shared004=frame1_head:0.125,shared_trunk:0.04",
    )
    return tuple(_parse_candidate(item) for item in raw)


def _changed_tensor_summary(source_state: dict[str, Any], target_state: dict[str, Any]) -> dict[str, Any]:
    changed = []
    total_changed = 0
    for name, source_tensor in source_state.items():
        target_tensor = target_state[name]
        diff_count = int((source_tensor != target_tensor).sum().item())
        if diff_count:
            total_changed += diff_count
            changed.append(
                {
                    "name": name,
                    "changed_values": diff_count,
                    "numel": int(source_tensor.numel()),
                    "changed_fraction": diff_count / float(source_tensor.numel()),
                }
            )
    return {
        "changed_tensor_count": len(changed),
        "changed_value_count": total_changed,
        "changed_tensors": changed,
    }


def _build_candidate(
    *,
    context: dict[str, Any],
    output_dir: Path,
    candidate_id: str,
    transforms: tuple[tuple[str, float], ...],
    brotli_quality: int,
    source_evidence_path: Path,
    run_preflight: bool,
    preflight_max_pairs: int,
    max_mean_abs_delta: float,
    max_rms_delta: float,
    max_max_abs_delta: float,
) -> dict[str, Any]:
    state = shrink_search._clone_state(context["source_state"])  # noqa: SLF001
    step_metas = []
    for prefix, threshold in transforms:
        state, meta = shrink_search._apply_zero_fp4_prefix(  # noqa: SLF001
            state,
            prefix=prefix,
            threshold_fraction=threshold,
        )
        step_metas.append({"prefix": prefix, "threshold": threshold, **meta})
    changed_summary = _changed_tensor_summary(context["source_state"], state)
    renderer = shrink_search.encode_qzs3_state_dict(
        state,
        block_size=context["source_block_size"],
    )
    shrink_search.decode_qzs3_state_dict(renderer, device="cpu")
    payload, payload_meta = pr75_builder._build_pr75_payload(  # noqa: SLF001
        context["pr75_slices"],
        renderer_bytes=renderer,
        brotli_quality=brotli_quality,
    )
    candidate_dir = output_dir / candidate_id
    candidate_dir.mkdir(parents=True, exist_ok=True)
    archive_path = candidate_dir / "archive.zip"
    pr75_builder._write_single_member_archive(archive_path, payload)  # noqa: SLF001
    runtime_unpack = pr75_builder._verify_archive(  # noqa: SLF001
        archive_path,
        expected_renderer=renderer,
        expected_non_renderer_members=context["non_renderer_members"],
    )
    archive_bytes = archive_path.stat().st_size
    archive_sha = _sha256_file(archive_path)
    renderer_changed = renderer != context["source_renderer"]
    no_op_check = {
        "renderer_changed": renderer_changed,
        "archive_sha_changed": archive_sha != context["source_sha256"],
        "archive_bytes_changed": archive_bytes != len(context["source_bytes"]),
        "changed_value_count": changed_summary["changed_value_count"],
        "transform_is_noop": (
            not renderer_changed or changed_summary["changed_value_count"] <= 0
        ),
        "non_renderer_members_preserved": True,
        "runtime_unpack_verified": True,
    }
    manifest = {
        "schema": SCHEMA,
        "candidate_id": candidate_id,
        "score_claim": False,
        "promotion_eligible": False,
        "remote_gpu_dispatch_performed": False,
        "source_archive": {
            "path": str(context["source_archive"]),
            "bytes": len(context["source_bytes"]),
            "sha256": context["source_sha256"],
        },
        "source_evidence": str(source_evidence_path),
        "output_archive": {
            "path": str(archive_path),
            "bytes": archive_bytes,
            "sha256": archive_sha,
            "delta_bytes_vs_source_archive": archive_bytes - len(context["source_bytes"]),
            "score_if_components_unchanged_vs_c101": (
                C101_SCORE
                + RATE_SCORE_PER_BYTE * (archive_bytes - len(context["source_bytes"]))
            ),
            "byte_sufficient_for_sub314_if_components_unchanged": (
                C101_SCORE
                + RATE_SCORE_PER_BYTE * (archive_bytes - len(context["source_bytes"]))
            )
            < 0.314,
        },
        "renderer_transform": {
            "wire_format": "QZS3",
            "source_block_size": context["source_block_size"],
            "output_block_size": context["source_block_size"],
            "transforms": [
                {"prefix": prefix, "threshold": threshold}
                for prefix, threshold in transforms
            ],
            "source_bytes": len(context["source_renderer"]),
            "source_sha256": _sha256_bytes(context["source_renderer"]),
            "output_bytes": len(renderer),
            "output_sha256": _sha256_bytes(renderer),
            "step_metas": step_metas,
            **changed_summary,
        },
        "payload": {
            "member_name": pr75_builder.PR75_PAYLOAD_MEMBER,
            "bytes": len(payload),
            "sha256": _sha256_bytes(payload),
            **payload_meta,
        },
        "non_renderer_preservation": {
            "all_non_renderer_members_preserved": True,
            "members": pr75_builder._member_meta(context["non_renderer_members"]),  # noqa: SLF001
        },
        "runtime_contract": {
            "byte_closed": True,
            "single_payload_member": True,
            "runtime_unpack_verified": True,
            "runtime_unpack_summary": runtime_unpack,
            "renderer_only_transplant": True,
            "pose_safety_preflight_required_before_dispatch": True,
        },
        "no_op_check": no_op_check,
    }
    (candidate_dir / "build_manifest.json").write_bytes(_json_bytes(manifest))
    pose_report_meta = {
        "preflight_ran": False,
        "safe_for_exact_eval_dispatch": False,
        "failure_class": "pose_safety_preflight_not_run",
    }
    if run_preflight:
        output_json = candidate_dir / "pose_safety_preflight.json"
        report = pose_safety.build_pose_safety_preflight(
            source_archive=context["source_archive"],
            candidate_archive=archive_path,
            output_json=output_json,
            max_pairs=preflight_max_pairs,
            max_mean_abs_delta=max_mean_abs_delta,
            max_rms_delta=max_rms_delta,
            max_max_abs_delta=max_max_abs_delta,
        )
        pose_report_meta = {
            "preflight_ran": True,
            "path": str(output_json),
            "safe_for_exact_eval_dispatch": bool(
                report.get("safe_for_exact_eval_dispatch")
            ),
            "failure_class": report.get("failure_class"),
            "fail_closed_reasons": report.get("fail_closed_reasons", []),
            "output_parity": report.get("output_parity"),
        }
    return {
        "candidate_id": candidate_id,
        "archive": str(archive_path),
        "archive_bytes": archive_bytes,
        "archive_sha256": archive_sha,
        "delta_bytes_vs_source_archive": archive_bytes - len(context["source_bytes"]),
        "score_if_components_unchanged_vs_c101": manifest["output_archive"][
            "score_if_components_unchanged_vs_c101"
        ],
        "byte_sufficient_for_sub314_if_components_unchanged": manifest["output_archive"][
            "byte_sufficient_for_sub314_if_components_unchanged"
        ],
        "manifest": str(candidate_dir / "build_manifest.json"),
        "transforms": manifest["renderer_transform"]["transforms"],
        "no_op_check": no_op_check,
        "pose_safety": pose_report_meta,
        "score_claim": False,
        "promotion_eligible": False,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    source_archive = args.source_archive.resolve()
    source_evidence_path = args.source_evidence_path.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not args.force:
        raise FileExistsError(f"output directory is non-empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    actual_source_sha = _sha256_file(source_archive)
    source_custody = {
        "path": str(source_archive),
        "bytes": source_archive.stat().st_size,
        "sha256": actual_source_sha,
        "expected_bytes": C101_BYTES,
        "expected_sha256": C101_SHA256,
        "verified": (
            source_archive.stat().st_size == C101_BYTES
            and actual_source_sha == C101_SHA256
        ),
    }
    context = shrink_search._source_context(source_archive)  # noqa: SLF001
    candidates = []
    for candidate_id, transforms in args.candidate:
        candidates.append(
            _build_candidate(
                context=context,
                output_dir=output_dir,
                candidate_id=candidate_id,
                transforms=transforms,
                brotli_quality=args.brotli_quality,
                source_evidence_path=source_evidence_path,
                run_preflight=not args.skip_preflight,
                preflight_max_pairs=args.preflight_max_pairs,
                max_mean_abs_delta=args.max_mean_abs_delta,
                max_rms_delta=args.max_rms_delta,
                max_max_abs_delta=args.max_max_abs_delta,
            )
        )
    candidates.sort(key=lambda row: (row["archive_bytes"], row["candidate_id"]))
    safe_candidates = [
        row
        for row in candidates
        if row.get("pose_safety", {}).get("safe_for_exact_eval_dispatch")
    ]
    recommendation = {
        "recommendation": "do_not_dispatch",
        "reason": "no combined candidate passed local pose-safety",
        "remote_gpu_dispatch_performed": False,
        "claim_required_before_dispatch": True,
    }
    if safe_candidates:
        best = min(safe_candidates, key=lambda row: (row["archive_bytes"], row["candidate_id"]))
        recommendation = {
            "recommendation": (
                "claim_lane_then_remote_exact_eval"
                if best["byte_sufficient_for_sub314_if_components_unchanged"]
                else "do_not_dispatch_yet_safe_but_too_small"
            ),
            "reason": (
                "best local-safe combined candidate is byte-sufficient for sub-0.314"
                if best["byte_sufficient_for_sub314_if_components_unchanged"]
                else "best local-safe combined candidate does not reach the C-101 byte target"
            ),
            "candidate": best,
            "remote_gpu_dispatch_performed": False,
            "claim_required_before_dispatch": True,
        }
    summary = {
        "schema": SCHEMA,
        "score_claim": False,
        "promotion_eligible": False,
        "remote_gpu_dispatch_performed": False,
        "source_custody": source_custody,
        "source_evidence": str(source_evidence_path),
        "c101_score": C101_SCORE,
        "target_score": 0.314,
        "bytes_needed_for_sub314_if_components_unchanged": (
            math.floor((C101_SCORE - 0.314) / RATE_SCORE_PER_BYTE) + 1
        ),
        "preflight_thresholds": {
            "max_pairs": args.preflight_max_pairs,
            "max_mean_abs_delta": args.max_mean_abs_delta,
            "max_rms_delta": args.max_rms_delta,
            "max_max_abs_delta": args.max_max_abs_delta,
        },
        "candidate_count": len(candidates),
        "candidates": candidates,
        "safe_candidates": safe_candidates,
        "best_by_archive_bytes": candidates[0] if candidates else None,
        "dispatch_recommendation": recommendation,
    }
    (output_dir / "summary.json").write_bytes(_json_bytes(summary))
    (output_dir / "dispatch_recommendation.json").write_bytes(_json_bytes(recommendation))
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-archive", type=Path, default=DEFAULT_SOURCE_ARCHIVE)
    parser.add_argument("--source-evidence-path", type=Path, default=DEFAULT_SOURCE_EVIDENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--candidate",
        action="append",
        type=_parse_candidate,
        default=None,
        help="Candidate spec: name=prefix:threshold,prefix:threshold",
    )
    parser.add_argument("--brotli-quality", type=int, default=11)
    parser.add_argument("--preflight-max-pairs", type=int, default=5)
    parser.add_argument("--max-mean-abs-delta", type=float, default=3.0)
    parser.add_argument("--max-rms-delta", type=float, default=8.0)
    parser.add_argument("--max-max-abs-delta", type=float, default=80.0)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.candidate is None:
        args.candidate = list(_default_candidates())
    return args


def main(argv: list[str] | None = None) -> int:
    summary = run(parse_args(argv))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
