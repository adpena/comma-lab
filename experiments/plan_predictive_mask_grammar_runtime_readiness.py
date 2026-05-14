#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Plan local runtime readiness for predictive mask grammar archives.

This is a local-only readiness planner for the C067 sub-0.24 path. It reads the
byte-probe manifest, inspects already-built CMG3/PMG runtime surfaces and exact
negative artifacts, then emits a fail-closed next-action plan. It does not build
archives, load scorer networks, claim scores, or dispatch remote/GPU jobs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
TOOL = "experiments/plan_predictive_mask_grammar_runtime_readiness.py"
SCHEMA = "predictive_mask_grammar_runtime_readiness_plan_v1"
REPORT_NAME = "predictive_mask_grammar_runtime_readiness_plan.json"
EVIDENCE_GRADE = "local_planning_only_non_score"
CUDA_AUTH_EVAL_REQUIRED = (
    "archive.zip -> inflate.sh -> upstream/evaluate.py via "
    "experiments/contest_auth_eval.py --device cuda"
)

CURRENT_C067_SCORE = 0.31561703078448233
CURRENT_C067_ARCHIVE_BYTES = 276_214
SUB024_TARGET_SCORE = 0.24
SUB024_REQUIRED_SAVINGS_AT_UNCHANGED_DISTORTION = 113_564
CURRENT_MASK_STREAM_BYTES = 219_472
POSE_COLLAPSE_THRESHOLD = 0.01
SEG_COLLAPSE_THRESHOLD = 0.005

DEFAULT_PROBE_MANIFEST = (
    REPO_ROOT
    / "experiments/results/c067_predictive_mask_grammar_probe_20260502T1040Z/"
    "predictive_mask_grammar_probe_manifest.json"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "experiments/results/predictive_mask_grammar_runtime_readiness_20260502"
)


@dataclass(frozen=True)
class ExactNegativeSpec:
    artifact_id: str
    family: str
    path: Path


DEFAULT_EXACT_NEGATIVE_SPECS: tuple[ExactNegativeSpec, ...] = (
    ExactNegativeSpec(
        "c067_cmg3_nonzero_top1_t4",
        "cmg3_nonzero_row_runs",
        REPO_ROOT
        / "experiments/results/lightning_batch/"
        "exact_eval_c067_cmg3_nonzero_top1_t4_20260502T1100Z/"
        "contest_auth_eval.json",
    ),
    ExactNegativeSpec(
        "c067_cmg3_nonzero_top2_t4",
        "cmg3_nonzero_row_runs",
        REPO_ROOT
        / "experiments/results/lightning_batch/"
        "exact_eval_c067_cmg3_nonzero_top2_t4_20260502T1100Z/"
        "contest_auth_eval.json",
    ),
    ExactNegativeSpec(
        "c067_cmg3_rowspan_stride1_t4",
        "cmg3_row_span",
        REPO_ROOT
        / "experiments/results/lightning_batch/"
        "exact_eval_c067_cmg3_rowspan_stride1_t4_20260502T1225Z/"
        "contest_auth_eval.json",
    ),
    ExactNegativeSpec(
        "c067_cmg3_rowspan_stride2_l40sdiag",
        "cmg3_row_span",
        REPO_ROOT
        / "experiments/results/lightning_batch/"
        "exact_eval_c067_cmg3_rowspan_stride2_l40sdiag_20260502T1225Z/"
        "contest_auth_eval.json",
    ),
    ExactNegativeSpec(
        "c067_cmg3a_body200_l40s",
        "cmg3a_adaptive_runs",
        REPO_ROOT
        / "experiments/results/lightning_batch/"
        "exact_eval_c067_cmg3a_body200_l40s_20260502T114231Z/"
        "contest_auth_eval.json",
    ),
    ExactNegativeSpec(
        "pmg_hotspot_c067_t4",
        "pmg_hotspot_row_span_residual",
        REPO_ROOT
        / "experiments/results/lightning_batch/"
        "exact_eval_pmg_hotspot_c067_t4_20260502T1402Z/"
        "contest_auth_eval.json",
    ),
    ExactNegativeSpec(
        "pmg_hotspot_atomtop4068_l40sdiag",
        "pmg_hotspot_row_span_residual",
        REPO_ROOT
        / "experiments/results/lightning_batch/"
        "exact_eval_pmg_hotspot_atomtop4068_l40sdiag_20260502T1445Z/"
        "contest_auth_eval.json",
    ),
)


class PlannerError(ValueError):
    """Raised when readiness inputs are missing or unsafe."""


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


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise PlannerError(f"required probe manifest is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PlannerError(f"{path} is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise PlannerError(f"{path} must contain a JSON object")
    return payload


def _optional_json_object(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _require_dict(payload: dict[str, Any], key: str, path: Path) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise PlannerError(f"{path} malformed: missing object field {key!r}")
    return value


def _require_int(payload: dict[str, Any], key: str, path: Path, *, positive: bool = False) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise PlannerError(f"{path} malformed: field {key!r} must be an integer")
    if positive and value <= 0:
        raise PlannerError(f"{path} malformed: field {key!r} must be positive")
    return int(value)


def _require_bool(payload: dict[str, Any], key: str, path: Path) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        raise PlannerError(f"{path} malformed: field {key!r} must be boolean")
    return bool(value)


def _require_str(payload: dict[str, Any], key: str, path: Path) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise PlannerError(f"{path} malformed: field {key!r} must be a nonempty string")
    return value


def _finite_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    out = float(value)
    return out if math.isfinite(out) else None


def _probe_record(path: Path) -> dict[str, Any]:
    payload = _read_json_object(path)
    baseline = _require_dict(payload, "baseline", path)
    baseline_bytes = _require_int(baseline, "bytes", path, positive=True)
    best = _require_dict(payload, "best_candidate_by_compressed_size", path)
    best_compression = _require_dict(best, "best_compression", path)
    compressed_size = _require_int(best_compression, "compressed_size_bytes", path, positive=True)
    delta_vs_baseline = _require_int(best_compression, "delta_bytes_vs_baseline", path)
    candidate_id = _require_str(best, "candidate_id", path)
    family = _require_str(best, "family", path)
    evidence_grade = _require_str(best, "evidence_grade", path)
    scope = _require_str(best, "charged_payload_estimate_scope", path)
    exact_evaluable_now = _require_bool(best, "exact_evaluable_now", path)
    promotion_eligible = _require_bool(best, "promotion_eligible", path)
    score_claim = _require_bool(best, "score_claim", path)
    compressor = _require_str(best_compression, "compressor", path)
    transform_stats = best.get("transform_stats")
    if transform_stats is not None and not isinstance(transform_stats, dict):
        raise PlannerError(f"{path} malformed: transform_stats must be an object when present")
    span_shape = transform_stats.get("span_shape") if isinstance(transform_stats, dict) else None
    if span_shape is not None:
        if not (
            isinstance(span_shape, list)
            and len(span_shape) == 4
            and all(isinstance(v, int) and v > 0 for v in span_shape)
        ):
            raise PlannerError(f"{path} malformed: transform_stats.span_shape must be four positive integers")
    return {
        "path": str(path),
        "sha256": _sha256_file(path),
        "baseline_bytes": baseline_bytes,
        "baseline_source": baseline.get("source"),
        "candidate_id": candidate_id,
        "family": family,
        "evidence_grade": evidence_grade,
        "compressed_size_bytes": compressed_size,
        "compressor": compressor,
        "delta_bytes_vs_baseline": delta_vs_baseline,
        "charged_payload_estimate_scope": scope,
        "exact_evaluable_now": exact_evaluable_now,
        "exact_evaluable_reason": best.get("exact_evaluable_reason"),
        "promotion_eligible": promotion_eligible,
        "score_claim": score_claim,
        "span_shape": span_shape,
        "transform_stats": transform_stats or {},
    }


def _source_contains(path: Path, needles: tuple[str, ...]) -> bool:
    if not path.exists():
        return False
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        text = path.read_text(errors="ignore")
    return all(needle in text for needle in needles)


def inspect_runtime_support(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    inflate_renderer = repo_root / "submissions/robust_current/inflate_renderer.py"
    unpacker = repo_root / "submissions/robust_current/unpack_renderer_payload.py"
    packed_builder = repo_root / "experiments/build_renderer_packed_payload_archive.py"
    cmg3_rowspan_builder = repo_root / "experiments/build_cmg3_rowspan_candidate.py"
    pmg_builder = repo_root / "experiments/build_pmg_hotspot_candidate.py"
    return {
        "runtime_decoder_exists": _source_contains(
            inflate_renderer,
            ("_load_masks_from_cmg3", "row_span_stride_class_predictor_v1"),
        ),
        "hotspot_residual_decoder_exists": _source_contains(
            inflate_renderer,
            ("row_span_stride_class_predictor_hotspot_residual_v1", "_decode_cmg3_row_span_hotspot_residual"),
        ),
        "nonzero_run_decoder_exists": _source_contains(
            inflate_renderer,
            ("nonzero_row_runs_topk_v1", "_decode_cmg3_nonzero_row_runs"),
        ),
        "packed_payload_unpacker_accepts_cmg3": _source_contains(unpacker, ("masks.cmg3",)),
        "packed_payload_builder_accepts_cmg3": _source_contains(packed_builder, ("masks.cmg3",)),
        "local_rowspan_builder_exists": cmg3_rowspan_builder.exists(),
        "local_pmg_hotspot_builder_exists": pmg_builder.exists(),
        "supporting_paths": {
            "inflate_renderer": str(inflate_renderer),
            "unpack_renderer_payload": str(unpacker),
            "packed_payload_builder": str(packed_builder),
            "cmg3_rowspan_builder": str(cmg3_rowspan_builder),
            "pmg_hotspot_builder": str(pmg_builder),
        },
    }


def _exact_eval_record(spec: ExactNegativeSpec) -> dict[str, Any]:
    payload = _optional_json_object(spec.path)
    if payload is None:
        return {
            "artifact_id": spec.artifact_id,
            "family": spec.family,
            "path": str(spec.path),
            "present": False,
            "usable": False,
            "same_family_blocker": False,
        }
    archive_bytes = payload.get("archive_size_bytes")
    if isinstance(archive_bytes, bool) or not isinstance(archive_bytes, int):
        archive_bytes = None
    score = _finite_float(payload.get("score_recomputed_from_components", payload.get("final_score")))
    pose = _finite_float(payload.get("avg_posenet_dist"))
    seg = _finite_float(payload.get("avg_segnet_dist"))
    samples = payload.get("n_samples")
    if isinstance(samples, bool) or not isinstance(samples, int):
        samples = None
    pose_collapse = pose is not None and pose > POSE_COLLAPSE_THRESHOLD
    seg_collapse = seg is not None and seg > SEG_COLLAPSE_THRESHOLD
    byte_sufficient_for_sub024 = (
        archive_bytes is not None
        and archive_bytes <= CURRENT_C067_ARCHIVE_BYTES - SUB024_REQUIRED_SAVINGS_AT_UNCHANGED_DISTORTION
    )
    return {
        "artifact_id": spec.artifact_id,
        "family": spec.family,
        "path": str(spec.path),
        "sha256": _sha256_file(spec.path),
        "present": True,
        "usable": score is not None and pose is not None and seg is not None and samples == 600,
        "archive_bytes": archive_bytes,
        "savings_vs_c067_bytes": (
            CURRENT_C067_ARCHIVE_BYTES - archive_bytes if archive_bytes is not None else None
        ),
        "byte_sufficient_for_sub024_at_unchanged_distortion": byte_sufficient_for_sub024,
        "score_recomputed_from_components": score,
        "avg_posenet_dist": pose,
        "avg_segnet_dist": seg,
        "n_samples": samples,
        "pose_collapse": pose_collapse,
        "seg_collapse": seg_collapse,
        "same_family_blocker": pose_collapse or seg_collapse,
    }


def inspect_exact_negatives(specs: tuple[ExactNegativeSpec, ...]) -> dict[str, Any]:
    records = [_exact_eval_record(spec) for spec in specs]
    present = [record for record in records if record["present"]]
    blockers = [record for record in present if record["same_family_blocker"]]
    byte_sufficient_collapsed = [
        record
        for record in blockers
        if record.get("byte_sufficient_for_sub024_at_unchanged_distortion") is True
    ]
    return {
        "records": records,
        "present_count": len(present),
        "missing_count": len(records) - len(present),
        "same_family_blocker_count": len(blockers),
        "same_family_blockers_present": bool(blockers),
        "byte_sufficient_but_collapsed_count": len(byte_sufficient_collapsed),
        "blocking_scope": (
            "blocks immediate promotion or remote dispatch of measured CMG3/PMG "
            "row-span/row-run global replacement variants; does not kill all "
            "learned or pose-conditioned mask grammar research"
            if blockers
            else "no same-family exact-negative blocker found in configured artifacts"
        ),
    }


def compute_byte_headroom(probe: dict[str, Any]) -> dict[str, Any]:
    target_archive_bytes = (
        CURRENT_C067_ARCHIVE_BYTES - SUB024_REQUIRED_SAVINGS_AT_UNCHANGED_DISTORTION
    )
    probe_savings_vs_baseline = -int(probe["delta_bytes_vs_baseline"])
    lower_bound_archive_bytes = CURRENT_C067_ARCHIVE_BYTES - probe_savings_vs_baseline
    headroom_before_overhead = target_archive_bytes - lower_bound_archive_bytes
    return {
        "current_c067_score": CURRENT_C067_SCORE,
        "current_c067_archive_bytes": CURRENT_C067_ARCHIVE_BYTES,
        "target_score": SUB024_TARGET_SCORE,
        "target_archive_bytes_at_unchanged_distortion": target_archive_bytes,
        "required_savings_at_unchanged_distortion_bytes": (
            SUB024_REQUIRED_SAVINGS_AT_UNCHANGED_DISTORTION
        ),
        "probe_baseline_mask_stream_bytes": int(probe["baseline_bytes"]),
        "best_probe_compressed_payload_bytes": int(probe["compressed_size_bytes"]),
        "best_probe_savings_vs_probe_baseline_bytes": probe_savings_vs_baseline,
        "formula_only_archive_lower_bound_if_probe_replaced_mask_stream_bytes": (
            lower_bound_archive_bytes
        ),
        "formula_only_headroom_before_decoder_packer_runtime_overhead_bytes": (
            headroom_before_overhead
        ),
        "headroom_verdict": (
            "byte_probe_has_apparent_sub024_headroom_before charged decoder, packer, "
            "runtime, validator, and distortion costs"
            if headroom_before_overhead >= 0
            else "byte_probe_is_short_even_before runtime and distortion costs"
        ),
    }


def implementation_blockers(
    probe: dict[str, Any],
    runtime: dict[str, Any],
    negatives: dict[str, Any],
    byte_headroom: dict[str, Any],
) -> list[dict[str, Any]]:
    blockers: list[dict[str, Any]] = []
    if probe["score_claim"] or probe["promotion_eligible"] or probe["exact_evaluable_now"]:
        blockers.append(
            {
                "blocker_id": "probe_manifest_unsafe_promotable_state",
                "severity": "hard",
                "detail": "Probe manifest must remain non-score and non-promotable for readiness planning.",
            }
        )
    blockers.append(
        {
            "blocker_id": "byte_probe_not_archive_member",
            "severity": "hard",
            "detail": (
                "Best candidate is compressed probe payload only; it excludes a reviewed "
                "archive member, deterministic header, validator closure, packed-submission "
                "overhead, and exact CUDA artifact."
            ),
        }
    )
    if probe["compressor"] not in {"lzma_xz", "bz2", "zlib", "raw", "brotli"}:
        blockers.append(
            {
                "blocker_id": "probe_compressor_label_not_runtime_enum",
                "severity": "hard",
                "detail": (
                    f"Probe compressor {probe['compressor']!r} must be re-emitted through "
                    "the CMG3 runtime compressor enum before archive construction."
                ),
            }
        )
    if not runtime["runtime_decoder_exists"]:
        blockers.append(
            {
                "blocker_id": "missing_cmg3_runtime_decoder",
                "severity": "hard",
                "detail": "No local inflate CMG3 row-span decoder was detected.",
            }
        )
    if not runtime["packed_payload_builder_accepts_cmg3"]:
        blockers.append(
            {
                "blocker_id": "missing_cmg3_packed_payload_builder_route",
                "severity": "hard",
                "detail": "Packed-payload builder does not advertise masks.cmg3 support.",
            }
        )
    if negatives["same_family_blockers_present"]:
        blockers.append(
            {
                "blocker_id": "same_family_exact_negatives_pose_seg_collapse",
                "severity": "hard",
                "detail": (
                    "Existing exact CUDA CMG3/PMG variants are same-family blockers for "
                    "remote dispatch until a local decoded-geometry guard explains why "
                    "this candidate escapes the measured collapse mode."
                ),
            }
        )
    if byte_headroom["formula_only_headroom_before_decoder_packer_runtime_overhead_bytes"] < 0:
        blockers.append(
            {
                "blocker_id": "insufficient_formula_only_byte_headroom",
                "severity": "hard",
                "detail": "Probe payload is byte-insufficient before runtime and distortion costs.",
            }
        )
    return blockers


def ranked_next_actions(
    *,
    runtime: dict[str, Any],
    negatives: dict[str, Any],
    blockers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    actions = [
        {
            "rank": 1,
            "action_id": "local_decode_geometry_parity_gate",
            "status": "required_before_dispatch",
            "reason": (
                "Exact negatives show small-looking mask grammar changes can collapse "
                "PoseNet/SegNet, so the next step is a local decoded-mask parity and "
                "geometry-risk gate, not another GPU job."
            ),
        },
        {
            "rank": 2,
            "action_id": "reemit_probe_as_canonical_cmg3_candidate",
            "status": "local_only",
            "reason": (
                "The byte probe must be converted into canonical CMG3 bytes with the "
                "runtime compressor names, manifest SHAs, deterministic policy fields, "
                "and packed-payload accounting."
            ),
        },
        {
            "rank": 3,
            "action_id": "explain_escape_from_same_family_negatives",
            "status": "required_before_exact_eval",
            "reason": negatives["blocking_scope"],
        },
        {
            "rank": 4,
            "action_id": "add_validator_and_local_inflate_smoke",
            "status": "required_before_exact_eval",
            "reason": (
                "Archive readiness needs zip/payload closure and local inflate routing "
                "for masks.cmg3 before any dispatch claim."
            ),
        },
        {
            "rank": 5,
            "action_id": "remote_dispatch",
            "status": "blocked",
            "reason": "No remote dispatch in this task; exact eval is blocked by local readiness findings.",
        },
    ]
    if not runtime["runtime_decoder_exists"]:
        actions.insert(
            0,
            {
                "rank": 0,
                "action_id": "implement_cmg3_runtime_decoder",
                "status": "required_first",
                "reason": "A CMG3 runtime decoder was not detected.",
            },
        )
    for index, action in enumerate(actions, start=1):
        action["rank"] = index
    if not blockers:
        actions[-1]["status"] = "still_no_dispatch_without_user_claim_and_cuda_gate"
    return actions


def build_plan(
    *,
    probe_manifest: Path = DEFAULT_PROBE_MANIFEST,
    output_json: Path | None = None,
    exact_negative_specs: tuple[ExactNegativeSpec, ...] = DEFAULT_EXACT_NEGATIVE_SPECS,
    repo_root: Path = REPO_ROOT,
    command: list[str] | None = None,
) -> dict[str, Any]:
    output_json = output_json or DEFAULT_OUTPUT_DIR / REPORT_NAME
    probe = _probe_record(probe_manifest)
    runtime = inspect_runtime_support(repo_root=repo_root)
    negatives = inspect_exact_negatives(exact_negative_specs)
    byte_headroom = compute_byte_headroom(probe)
    blockers = implementation_blockers(probe, runtime, negatives, byte_headroom)
    dispatchable_now = False
    verdict = (
        "blocked_local_only_archive_readiness"
        if blockers
        else "local_readiness_plausible_but_still_non_score"
    )
    manifest = {
        "schema": SCHEMA,
        "tool": TOOL,
        "evidence_grade": EVIDENCE_GRADE,
        "score_claim": False,
        "promotion_eligible": False,
        "gpu_required": False,
        "cuda_jobs_launched": False,
        "cloud_jobs_dispatched": False,
        "remote_dispatch_allowed": False,
        "canonical_score_source_required": CUDA_AUTH_EVAL_REQUIRED,
        "created_by_command": command or [TOOL],
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
        "probe": probe,
        "byte_headroom": byte_headroom,
        "runtime_decoder_packer_readiness": runtime,
        "same_family_exact_negative_review": negatives,
        "implementation_blockers": blockers,
        "ranked_next_actions": ranked_next_actions(
            runtime=runtime,
            negatives=negatives,
            blockers=blockers,
        ),
        "dispatch_readiness": {
            "dispatchable_now": dispatchable_now,
            "verdict": verdict,
            "reason": (
                "The row-span probe has formula-only byte headroom, but existing "
                "CMG3/PMG exact negatives and probe-only payload scope block archive "
                "promotion until local runtime and geometry-parity gates pass."
            ),
        },
        "artifacts": {
            "manifest": {
                "path": str(output_json),
            }
        },
    }
    _write_json(output_json, manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-manifest", type=Path, default=DEFAULT_PROBE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_json = args.output_json or args.output_dir / REPORT_NAME
    manifest = build_plan(
        probe_manifest=args.probe_manifest,
        output_json=output_json,
        command=[TOOL, *sys.argv[1:]],
    )
    print(json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
