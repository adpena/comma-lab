#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Plan C091-native renderer/QZS3 self-compression big moves locally.

This is an orchestration helper around the existing PR75 renderer-shrink
builder and renderer-transplant pose-safety preflight. It validates the C091
frontier custody record, builds deterministic local renderer-only candidates
when requested, and refuses exact-eval recommendations unless a candidate is
both byte-sufficient and locally pose-safe. It never dispatches remote GPU work
or edits dispatch state.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "src", REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from experiments import build_renderer_shrink_candidate as renderer_builder  # noqa: E402
from experiments import preflight_renderer_transplant_pose_safety as pose_safety  # noqa: E402


SCHEMA = "c091_renderer_self_compression_bigmove_plan_v1"
TOOL = "experiments/plan_c091_renderer_self_compression_bigmove.py"
RATE_SCORE_PER_BYTE = 25.0 / 37_545_489.0
TARGET_SCORE = 0.314
MIN_PLAUSIBLE_SAVINGS_BYTES = 1_800
FRONTIER_SCORE = 0.31516575028285976
FRONTIER_BYTES = 276_481
FRONTIER_SHA256 = "03a2afd5fe92c93a9b7b7e43625158a73b455f0cfbca82d278008a728db78746"
DEFAULT_FRONTIER_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/lightning_batch/"
    "exact_eval_pr75_minp_public_replay_t4_20260503T1049Z/archive.zip"
)
DEFAULT_FRONTIER_EVAL = DEFAULT_FRONTIER_ARCHIVE.with_name(
    "contest_auth_eval.adjudicated.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "experiments/results/c091_renderer_self_compression_bigmove_20260503_worker"
)
DEFAULT_BLOCK_SIZES = (32, 48, 64, 96, 128, 192, 256, 512)
C091_FIXED_MASK_BR_BYTES = 219_472
C091_FIXED_RENDERER_BR_BYTES = 55_756
C091_FIXED_ACTIONS_BR_BYTES = 255


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def _repo_rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def parse_int_tuple(value: str) -> tuple[int, ...]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("integer list must not be empty")
    out: list[int] = []
    for item in items:
        try:
            parsed = int(item)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid integer {item!r}") from exc
        if parsed <= 0 or parsed > 4096:
            raise argparse.ArgumentTypeError(
                f"integer entries must be in [1, 4096], got {parsed}"
            )
        if parsed not in out:
            out.append(parsed)
    return tuple(out)


def byte_gap_to_target(
    *,
    frontier_score: float,
    target_score: float = TARGET_SCORE,
    rate_score_per_byte: float = RATE_SCORE_PER_BYTE,
) -> int:
    """Return bytes that must be saved to become strictly below target."""

    gap = float(frontier_score) - float(target_score)
    if gap < 0:
        return 0
    return int(math.floor(gap / float(rate_score_per_byte))) + 1


def score_if_components_unchanged(
    *,
    frontier_score: float,
    frontier_bytes: int,
    candidate_bytes: int,
) -> float:
    return float(frontier_score) + RATE_SCORE_PER_BYTE * (
        int(candidate_bytes) - int(frontier_bytes)
    )


def _load_json(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def validate_frontier_custody(
    *,
    archive: Path,
    eval_json: Path,
    expected_bytes: int = FRONTIER_BYTES,
    expected_sha256: str = FRONTIER_SHA256,
    expected_score: float = FRONTIER_SCORE,
) -> dict[str, Any]:
    """Validate the exact C091 archive/eval pair used for planning."""

    archive = archive.resolve()
    eval_json = eval_json.resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"frontier archive missing: {archive}")
    if not eval_json.is_file():
        raise FileNotFoundError(f"frontier eval JSON missing: {eval_json}")
    actual_bytes = archive.stat().st_size
    actual_sha = _sha256_file(archive)
    eval_payload = _load_json(eval_json)
    provenance = eval_payload.get("provenance")
    if not isinstance(provenance, Mapping):
        provenance = {}
    failures: list[str] = []
    if actual_bytes != int(expected_bytes):
        failures.append("archive_bytes_do_not_match_expected_c091")
    if actual_sha != expected_sha256:
        failures.append("archive_sha256_does_not_match_expected_c091")
    if int(eval_payload.get("archive_size_bytes", -1)) != actual_bytes:
        failures.append("eval_archive_size_mismatch")
    if provenance.get("archive_sha256") != actual_sha:
        failures.append("eval_archive_sha256_mismatch")
    if int(eval_payload.get("n_samples", -1)) != 600:
        failures.append("eval_not_full_600_samples")
    if provenance.get("device") != "cuda":
        failures.append("eval_not_cuda")
    score = float(
        eval_payload.get(
            "canonical_score",
            eval_payload.get("score_recomputed_from_components", math.nan),
        )
    )
    if not math.isfinite(score) or abs(score - expected_score) > 1e-12:
        failures.append("eval_score_does_not_match_expected_c091")
    return {
        "ok": not failures,
        "failures": failures,
        "archive": {
            "path": str(archive),
            "repo_relative_path": _repo_rel(archive),
            "bytes": actual_bytes,
            "sha256": actual_sha,
        },
        "eval_json": {
            "path": str(eval_json),
            "repo_relative_path": _repo_rel(eval_json),
            "score": score,
            "avg_segnet_dist": eval_payload.get("avg_segnet_dist"),
            "avg_posenet_dist": eval_payload.get("avg_posenet_dist"),
            "n_samples": eval_payload.get("n_samples"),
            "device": provenance.get("device"),
            "gpu_model": provenance.get("gpu_model"),
            "runtime_tree_sha256": (
                (provenance.get("inflate_runtime_manifest") or {}).get(
                    "runtime_tree_sha256"
                )
                if isinstance(provenance.get("inflate_runtime_manifest"), Mapping)
                else None
            ),
        },
    }


def _runtime_source_profile(archive: Path) -> dict[str, Any]:
    payload = renderer_builder._read_single_payload(archive)  # noqa: SLF001
    slices = _parse_c091_fixed_or_self_describing_slices(payload) if payload else None
    members, packaging = renderer_builder.blockfp.extract_runtime_members(archive)
    renderer = members[renderer_builder.RENDERER_MEMBER]
    return {
        "payload_layout": slices.get("format") if slices else None,
        "source_packaging": packaging,
        "logical_members": renderer_builder._member_meta(members),  # noqa: SLF001
        "renderer": {
            "bytes": len(renderer),
            "sha256": hashlib.sha256(renderer).hexdigest(),
            "magic4": renderer[:4].decode("latin1", errors="replace"),
            "qzs3_block_size": (
                int.from_bytes(renderer[4:6], "little")
                if renderer.startswith(b"QZS3") and len(renderer) >= 6
                else None
            ),
        },
    }


def _parse_c091_fixed_or_self_describing_slices(payload: bytes) -> dict[str, Any]:
    """Parse C091/minp PR75 slices without relying on older PR75 constants."""

    if payload.startswith(b"P3"):
        cursor = 2 + struct.calcsize("<IHH")
        mask_len, model_len, actions_len = struct.unpack_from("<IHH", payload, 2)
        format_name = "P3"
    elif payload.startswith(b"P6"):
        cursor = 2 + struct.calcsize("<IHHH")
        mask_len, model_len, actions_len, record_count = struct.unpack_from("<IHHH", payload, 2)
        format_name = "P6"
    elif len(payload) == 276_381:
        cursor = 0
        mask_len = C091_FIXED_MASK_BR_BYTES
        model_len = C091_FIXED_RENDERER_BR_BYTES
        actions_len = C091_FIXED_ACTIONS_BR_BYTES
        record_count = None
        format_name = "c091_pr75_minp_fixed_slices"
    else:
        parsed = renderer_builder._parse_pr75_slices(payload)  # noqa: SLF001
        if parsed is None:
            raise ValueError(f"unsupported C091 renderer payload length: {len(payload)}")
        return parsed
    if min(mask_len, model_len, actions_len) <= 0:
        raise ValueError("C091 payload has empty required slice")
    mask_start = cursor
    mask_end = mask_start + mask_len
    model_end = mask_end + model_len
    actions_end = model_end + actions_len
    if actions_end >= len(payload):
        raise ValueError("C091 payload slices leave no pose stream")
    return {
        "format": format_name,
        "mask": payload[mask_start:mask_end],
        "model": payload[mask_end:model_end],
        "action_dict": b"",
        "actions": payload[model_end:actions_end],
        "pose": payload[actions_end:],
        "record_count": locals().get("record_count"),
    }


def _build_c091_p3_payload(
    source_slices: Mapping[str, Any],
    *,
    renderer_bytes: bytes,
    brotli_quality: int = 11,
) -> tuple[bytes, dict[str, Any]]:
    import brotli

    model = brotli.compress(renderer_bytes, quality=brotli_quality, lgwin=24)
    if len(model) > 65_535:
        raise ValueError(f"compressed renderer does not fit PR75 u16 model_len: {len(model)}")
    mask = bytes(source_slices["mask"])
    actions = bytes(source_slices["actions"])
    pose = bytes(source_slices["pose"])
    header = b"P3" + struct.pack("<IHH", len(mask), len(model), len(actions))
    payload = header + mask + model + actions + pose
    return payload, {
        "payload_format": "c091_p3_preserve_minp_slices",
        "source_payload_format": source_slices.get("format"),
        "mask_slice_bytes": len(mask),
        "renderer_slice_bytes": len(model),
        "actions_slice_bytes": len(actions),
        "pose_slice_bytes": len(pose),
        "renderer_slice_sha256": hashlib.sha256(model).hexdigest(),
        "brotli_quality": brotli_quality,
    }


def _build_c091_qzs3_candidates(
    *,
    source_archive: Path,
    output_dir: Path,
    qzs3_block_sizes: tuple[int, ...],
    force: bool,
) -> dict[str, Any]:
    source_bytes = source_archive.read_bytes()
    source_members, source_packaging = renderer_builder.blockfp.extract_runtime_members(source_archive)
    source_renderer = source_members[renderer_builder.RENDERER_MEMBER]
    if not source_renderer.startswith(b"QZS3"):
        raise ValueError(f"C091 source renderer must be QZS3, got {source_renderer[:4]!r}")
    source_payload = renderer_builder._read_single_payload(source_archive)  # noqa: SLF001
    source_slices = _parse_c091_fixed_or_self_describing_slices(source_payload)
    non_renderer_members = {
        name: data
        for name, data in source_members.items()
        if name != renderer_builder.RENDERER_MEMBER
    }
    state = renderer_builder.decode_qzs3_state_dict(source_renderer, device="cpu")
    output_dir.mkdir(parents=True, exist_ok=True)
    candidates: list[dict[str, Any]] = []
    for block_size in qzs3_block_sizes:
        renderer = renderer_builder.encode_qzs3_state_dict(state, block_size=block_size)
        payload, payload_meta = _build_c091_p3_payload(
            source_slices,
            renderer_bytes=renderer,
            brotli_quality=11,
        )
        candidate_id = f"qzs3_b{block_size:04d}_c091_p3_preserved_minp_slices"
        candidate_dir = output_dir / candidate_id
        if candidate_dir.exists() and any(candidate_dir.iterdir()) and not force:
            raise FileExistsError(f"candidate directory is non-empty; pass --force: {candidate_dir}")
        candidate_dir.mkdir(parents=True, exist_ok=True)
        archive_path = candidate_dir / "archive.zip"
        renderer_builder._write_single_member_archive(archive_path, payload)  # noqa: SLF001
        runtime_unpack = renderer_builder._verify_archive(  # noqa: SLF001
            archive_path,
            expected_renderer=renderer,
            expected_non_renderer_members=non_renderer_members,
        )
        archive_bytes = archive_path.stat().st_size
        archive_sha = _sha256_file(archive_path)
        manifest = {
            "schema": "c091_renderer_self_compression_candidate_v1",
            "tool": TOOL,
            "candidate_id": candidate_id,
            "score_claim": False,
            "promotion_eligible": False,
            "remote_gpu_dispatch_performed": False,
            "source_archive": {
                "path": str(source_archive),
                "bytes": len(source_bytes),
                "sha256": hashlib.sha256(source_bytes).hexdigest(),
                **source_packaging,
            },
            "output_archive": {
                "path": str(archive_path),
                "bytes": archive_bytes,
                "sha256": archive_sha,
                "delta_bytes_vs_source_archive": archive_bytes - len(source_bytes),
                "formula_only_rate_delta_vs_source_archive": (
                    RATE_SCORE_PER_BYTE * (archive_bytes - len(source_bytes))
                ),
            },
            "renderer_transform": {
                "codec": "QZS3",
                "source_block_size": int.from_bytes(source_renderer[4:6], "little"),
                "output_block_size": int(block_size),
                "source_bytes": len(source_renderer),
                "source_sha256": hashlib.sha256(source_renderer).hexdigest(),
                "output_bytes": len(renderer),
                "output_sha256": hashlib.sha256(renderer).hexdigest(),
                "output_same_as_source_renderer": renderer == source_renderer,
            },
            "payload": {
                "member_name": renderer_builder.PR75_PAYLOAD_MEMBER,
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
                **payload_meta,
            },
            "non_renderer_preservation": {
                "all_non_renderer_members_preserved": True,
                "members": renderer_builder._member_meta(non_renderer_members),  # noqa: SLF001
            },
            "runtime_contract": {
                "byte_closed": True,
                "single_payload_member": True,
                "runtime_unpack_verified": True,
                "runtime_unpack_summary": runtime_unpack,
                "renderer_only_transplant": True,
                "pose_safety_preflight_required_before_dispatch": True,
            },
        }
        manifest_path = candidate_dir / "build_manifest.json"
        manifest_path.write_bytes(_json_bytes(manifest))
        candidates.append(
            {
                "candidate_id": candidate_id,
                "archive": str(archive_path),
                "archive_bytes": archive_bytes,
                "archive_sha256": archive_sha,
                "manifest": str(manifest_path),
                "qzs3_block_size": int(block_size),
                "delta_bytes_vs_source_archive": archive_bytes - len(source_bytes),
                "renderer_bytes": len(renderer),
                "renderer_sha256": hashlib.sha256(renderer).hexdigest(),
                "payload_format": payload_meta["payload_format"],
                "renderer_transform_kind": "qzs3_reencoded_state",
                "score_claim": False,
                "promotion_eligible": False,
            }
        )
    candidates.sort(key=lambda item: (int(item["archive_bytes"]), str(item["candidate_id"])))
    summary = {
        "schema": "c091_renderer_self_compression_candidate_summary_v1",
        "score_claim": False,
        "promotion_eligible": False,
        "remote_gpu_dispatch_performed": False,
        "source_archive": {
            "path": str(source_archive),
            "bytes": len(source_bytes),
            "sha256": hashlib.sha256(source_bytes).hexdigest(),
        },
        "source_payload_layout": source_slices.get("format"),
        "source_members": renderer_builder._member_meta(source_members),  # noqa: SLF001
        "qzs3_block_sizes": list(qzs3_block_sizes),
        "candidate_count": len(candidates),
        "best_by_archive_bytes": candidates[0] if candidates else None,
        "candidates": candidates,
    }
    (output_dir / "summary.json").write_bytes(_json_bytes(summary))
    return summary


def _candidate_row(
    item: Mapping[str, Any],
    *,
    frontier_score: float,
    frontier_bytes: int,
    target_score: float,
    min_plausible_savings_bytes: int,
) -> dict[str, Any]:
    archive_bytes = int(item["archive_bytes"])
    delta = archive_bytes - int(frontier_bytes)
    unchanged_score = score_if_components_unchanged(
        frontier_score=frontier_score,
        frontier_bytes=frontier_bytes,
        candidate_bytes=archive_bytes,
    )
    savings = -delta
    return {
        "candidate_id": str(item["candidate_id"]),
        "archive": str(item["archive"]),
        "archive_bytes": archive_bytes,
        "archive_sha256": str(item["archive_sha256"]),
        "manifest": str(item["manifest"]),
        "delta_bytes_vs_frontier": delta,
        "savings_bytes_vs_frontier": savings,
        "score_if_components_unchanged": unchanged_score,
        "byte_sufficient_for_sub314_if_components_unchanged": unchanged_score < target_score,
        "plausible_bigmove_byte_savings": savings >= int(min_plausible_savings_bytes),
        "qzs3_block_size": item.get("qzs3_block_size"),
        "renderer_bytes": item.get("renderer_bytes"),
        "renderer_sha256": item.get("renderer_sha256"),
        "payload_format": item.get("payload_format"),
        "pose_safety": {
            "preflight_ran": False,
            "safe_for_exact_eval_dispatch": False,
            "failure_class": "pose_safety_preflight_not_run",
        },
    }


def _run_pose_safety(
    *,
    source_archive: Path,
    row: Mapping[str, Any],
    max_pairs: int,
    max_mean_abs_delta: float,
    max_rms_delta: float,
    max_max_abs_delta: float,
) -> dict[str, Any]:
    archive = Path(str(row["archive"]))
    output_json = archive.with_name("pose_safety_preflight.json")
    try:
        report = pose_safety.build_pose_safety_preflight(
            source_archive=source_archive,
            candidate_archive=archive,
            output_json=output_json,
            max_pairs=max_pairs,
            max_mean_abs_delta=max_mean_abs_delta,
            max_rms_delta=max_rms_delta,
            max_max_abs_delta=max_max_abs_delta,
        )
    except Exception as exc:
        report = {
            "schema": pose_safety.SCHEMA,
            "score_claim": False,
            "promotion_eligible": False,
            "remote_gpu_dispatch_performed": False,
            "safe_for_exact_eval_dispatch": False,
            "failure_class": "renderer_transplant_pose_safety_exception",
            "fail_closed_reasons": [type(exc).__name__],
            "exception": str(exc),
        }
        output_json.write_bytes(_json_bytes(report))
    return {
        "preflight_ran": True,
        "path": str(output_json),
        "safe_for_exact_eval_dispatch": bool(report.get("safe_for_exact_eval_dispatch")),
        "failure_class": report.get("failure_class"),
        "fail_closed_reasons": report.get("fail_closed_reasons", []),
        "output_parity": report.get("output_parity"),
    }


def dispatch_recommendation(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    safe_big = [
        row
        for row in rows
        if row.get("byte_sufficient_for_sub314_if_components_unchanged")
        and row.get("plausible_bigmove_byte_savings")
        and row.get("pose_safety", {}).get("safe_for_exact_eval_dispatch")
    ]
    if safe_big:
        best = min(safe_big, key=lambda row: (int(row["archive_bytes"]), str(row["candidate_id"])))
        return {
            "recommendation": "claim_lane_then_remote_exact_eval",
            "reason": (
                "candidate is C091-native, byte-sufficient for sub-0.314 under "
                "unchanged components, and passed local pose-safety"
            ),
            "candidate": dict(best),
            "claim_required_before_dispatch": True,
            "claim_tool": "tools/claim_lane_dispatch.py claim ...",
            "remote_gpu_dispatch_performed": False,
        }
    safe = [
        row for row in rows if row.get("pose_safety", {}).get("safe_for_exact_eval_dispatch")
    ]
    if safe:
        best = min(safe, key=lambda row: (int(row["archive_bytes"]), str(row["candidate_id"])))
        return {
            "recommendation": "do_not_dispatch_yet_safe_but_too_small",
            "reason": "local-safe renderer candidate did not meet the C091 byte target",
            "candidate": dict(best),
            "claim_required_before_dispatch": True,
            "remote_gpu_dispatch_performed": False,
        }
    return {
        "recommendation": "do_not_dispatch",
        "reason": "no byte-sufficient C091 renderer candidate passed local pose-safety",
        "claim_required_before_dispatch": True,
        "remote_gpu_dispatch_performed": False,
    }


def build_plan(
    *,
    frontier_archive: Path = DEFAULT_FRONTIER_ARCHIVE,
    frontier_eval_json: Path = DEFAULT_FRONTIER_EVAL,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    qzs3_block_sizes: tuple[int, ...] = DEFAULT_BLOCK_SIZES,
    target_score: float = TARGET_SCORE,
    min_plausible_savings_bytes: int = MIN_PLAUSIBLE_SAVINGS_BYTES,
    max_preflight_candidates: int = 4,
    preflight_max_pairs: int = 5,
    max_mean_abs_delta: float = pose_safety.DEFAULT_MAX_MEAN_ABS_DELTA,
    max_rms_delta: float = pose_safety.DEFAULT_MAX_RMS_DELTA,
    max_max_abs_delta: float = pose_safety.DEFAULT_MAX_MAX_ABS_DELTA,
    skip_build: bool = False,
    skip_preflight: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Build a local C091 renderer shrink plan and write plan.json."""

    frontier_archive = frontier_archive.resolve()
    frontier_eval_json = frontier_eval_json.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()) and not force:
        raise FileExistsError(f"output directory is non-empty; pass --force: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    custody = validate_frontier_custody(
        archive=frontier_archive,
        eval_json=frontier_eval_json,
    )
    source_profile = _runtime_source_profile(frontier_archive)
    frontier_score = float(custody["eval_json"]["score"])
    frontier_bytes = int(custody["archive"]["bytes"])
    bytes_needed = byte_gap_to_target(
        frontier_score=frontier_score,
        target_score=target_score,
    )

    builder_summary: dict[str, Any] | None = None
    rows: list[dict[str, Any]] = []
    if custody["ok"] and not skip_build:
        builder_dir = output_dir / "qzs3_reblock_candidates"
        builder_summary = _build_c091_qzs3_candidates(
            source_archive=frontier_archive,
            output_dir=builder_dir,
            qzs3_block_sizes=qzs3_block_sizes,
            force=force,
        )
        rows = [
            _candidate_row(
                item,
                frontier_score=frontier_score,
                frontier_bytes=frontier_bytes,
                target_score=target_score,
                min_plausible_savings_bytes=min_plausible_savings_bytes,
            )
            for item in builder_summary.get("candidates", [])
        ]
    elif not custody["ok"]:
        rows = []

    rows.sort(key=lambda row: (row["archive_bytes"], row["candidate_id"]))
    preflight_targets = [
        row
        for row in rows
        if row["delta_bytes_vs_frontier"] < 0
        and (
            row["plausible_bigmove_byte_savings"]
            or row["byte_sufficient_for_sub314_if_components_unchanged"]
        )
    ][:max_preflight_candidates]
    if not skip_preflight:
        target_ids = {row["candidate_id"] for row in preflight_targets}
        for row in rows:
            if row["candidate_id"] not in target_ids:
                continue
            row["pose_safety"] = _run_pose_safety(
                source_archive=frontier_archive,
                row=row,
                max_pairs=preflight_max_pairs,
                max_mean_abs_delta=max_mean_abs_delta,
                max_rms_delta=max_rms_delta,
                max_max_abs_delta=max_max_abs_delta,
            )

    recommendation = (
        dispatch_recommendation(rows)
        if custody["ok"]
        else {
            "recommendation": "do_not_dispatch",
            "reason": "frontier custody validation failed",
            "custody_failures": custody["failures"],
            "claim_required_before_dispatch": True,
            "remote_gpu_dispatch_performed": False,
        }
    )
    plan = {
        "schema": SCHEMA,
        "tool": TOOL,
        "score_claim": False,
        "promotion_eligible": False,
        "remote_gpu_dispatch_performed": False,
        "output_dir": str(output_dir),
        "frontier_custody": custody,
        "source_runtime_profile": source_profile,
        "target_score": target_score,
        "rate_score_per_byte": RATE_SCORE_PER_BYTE,
        "bytes_needed_for_strict_sub314_if_components_unchanged": bytes_needed,
        "min_plausible_savings_bytes": min_plausible_savings_bytes,
        "qzs3_block_sizes": list(qzs3_block_sizes),
        "builder_summary_path": (
            str((output_dir / "qzs3_reblock_candidates" / "summary.json"))
            if builder_summary is not None
            else None
        ),
        "candidate_count": len(rows),
        "preflight_target_count": len(preflight_targets),
        "preflight_thresholds": {
            "max_pairs": preflight_max_pairs,
            "max_mean_abs_delta": max_mean_abs_delta,
            "max_rms_delta": max_rms_delta,
            "max_max_abs_delta": max_max_abs_delta,
        },
        "candidates": rows,
        "best_by_archive_bytes": rows[0] if rows else None,
        "safe_candidates": [
            row for row in rows if row.get("pose_safety", {}).get("safe_for_exact_eval_dispatch")
        ],
        "dispatch_recommendation": recommendation,
        "exact_eval_recommendation": recommendation,
    }
    (output_dir / "plan.json").write_bytes(_json_bytes(plan))
    (output_dir / "dispatch_recommendation.json").write_bytes(_json_bytes(recommendation))
    return plan


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frontier-archive", type=Path, default=DEFAULT_FRONTIER_ARCHIVE)
    parser.add_argument("--frontier-eval-json", type=Path, default=DEFAULT_FRONTIER_EVAL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--qzs3-block-sizes", type=parse_int_tuple, default=DEFAULT_BLOCK_SIZES)
    parser.add_argument("--target-score", type=float, default=TARGET_SCORE)
    parser.add_argument("--min-plausible-savings-bytes", type=int, default=MIN_PLAUSIBLE_SAVINGS_BYTES)
    parser.add_argument("--max-preflight-candidates", type=int, default=4)
    parser.add_argument("--preflight-max-pairs", type=int, default=5)
    parser.add_argument(
        "--max-mean-abs-delta",
        type=float,
        default=pose_safety.DEFAULT_MAX_MEAN_ABS_DELTA,
    )
    parser.add_argument("--max-rms-delta", type=float, default=pose_safety.DEFAULT_MAX_RMS_DELTA)
    parser.add_argument(
        "--max-max-abs-delta",
        type=float,
        default=pose_safety.DEFAULT_MAX_MAX_ABS_DELTA,
    )
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    plan = build_plan(
        frontier_archive=args.frontier_archive,
        frontier_eval_json=args.frontier_eval_json,
        output_dir=args.output_dir,
        qzs3_block_sizes=args.qzs3_block_sizes,
        target_score=args.target_score,
        min_plausible_savings_bytes=args.min_plausible_savings_bytes,
        max_preflight_candidates=args.max_preflight_candidates,
        preflight_max_pairs=args.preflight_max_pairs,
        max_mean_abs_delta=args.max_mean_abs_delta,
        max_rms_delta=args.max_rms_delta,
        max_max_abs_delta=args.max_max_abs_delta,
        skip_build=args.skip_build,
        skip_preflight=args.skip_preflight,
        force=args.force,
    )
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
