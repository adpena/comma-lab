#!/usr/bin/env python3
"""Compare CUDA component traces against a reference archive.

This is the offline bridge from exact per-pair scorer traces to repair-atom
planning. It does not create a score claim. It ranks where a candidate archive
exceeds a stronger reference in PoseNet/SegNet contribution, while separately
accounting for the rate term.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any


SCHEMA_VERSION = 1
CONTEST_UNCOMPRESSED_BYTES = 37_545_489


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _finite_float(value: Any, *, field: str) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"{field} must be finite, got {value!r}")
    return out


def _load_trace(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get("score_claim") is not False:
        raise ValueError(f"{path} is not a diagnostic component trace: score_claim={payload.get('score_claim')!r}")
    if payload.get("evidence_grade") != "diagnostic_component_trace":
        raise ValueError(f"{path} has unexpected evidence_grade={payload.get('evidence_grade')!r}")
    samples = payload.get("samples")
    if not isinstance(samples, list) or not samples:
        raise ValueError(f"{path} has no samples list")
    if int(payload.get("n_samples", -1)) != len(samples):
        raise ValueError(f"{path} n_samples does not match samples length")
    by_pair: dict[int, dict[str, Any]] = {}
    for raw in samples:
        pair = int(raw["pair_index"])
        if pair in by_pair:
            raise ValueError(f"{path} duplicate pair_index={pair}")
        by_pair[pair] = raw
    expected = set(range(len(samples)))
    if set(by_pair) != expected:
        missing = sorted(expected - set(by_pair))[:10]
        extra = sorted(set(by_pair) - expected)[:10]
        raise ValueError(f"{path} pair indices are not contiguous; missing={missing} extra={extra}")
    payload["_samples_by_pair"] = by_pair
    payload["_path"] = str(path)
    payload["_sha256"] = sha256_file(path)
    return payload


def _score_from_trace(trace: dict[str, Any], *, uncompressed_bytes: int) -> dict[str, float]:
    avg_pose = _finite_float(trace["avg_posenet_dist"], field="avg_posenet_dist")
    avg_seg = _finite_float(trace["avg_segnet_dist"], field="avg_segnet_dist")
    archive_bytes = int(trace["archive_size_bytes"])
    rate = archive_bytes / uncompressed_bytes
    pose = math.sqrt(10.0 * avg_pose)
    seg = 100.0 * avg_seg
    rate_score = 25.0 * rate
    return {
        "avg_posenet_dist": avg_pose,
        "avg_segnet_dist": avg_seg,
        "archive_size_bytes": float(archive_bytes),
        "score_pose_contribution": pose,
        "score_seg_contribution": seg,
        "score_rate_contribution": rate_score,
        "score_recomputed_from_components": pose + seg + rate_score,
    }


def _trace_identity(label: str, trace: dict[str, Any], *, uncompressed_bytes: int) -> dict[str, Any]:
    score = _score_from_trace(trace, uncompressed_bytes=uncompressed_bytes)
    hardware = _trace_hardware(trace)
    return {
        "label": label,
        "trace_json": trace["_path"],
        "trace_json_sha256": trace["_sha256"],
        "n_samples": int(trace["n_samples"]),
        "archive_size_bytes": int(trace["archive_size_bytes"]),
        "score_recomputed_from_components": score["score_recomputed_from_components"],
        "score_pose_contribution": score["score_pose_contribution"],
        "score_seg_contribution": score["score_seg_contribution"],
        "score_rate_contribution": score["score_rate_contribution"],
        "avg_posenet_dist": score["avg_posenet_dist"],
        "avg_segnet_dist": score["avg_segnet_dist"],
        "hardware": hardware,
    }


def _trace_hardware(trace: dict[str, Any]) -> dict[str, Any]:
    inputs = trace.get("trace_inputs") or {}
    device = inputs.get("device")
    device_type = str(device).split(":", 1)[0] if device is not None else None
    return {
        "device": device,
        "device_type": device_type,
        "cuda_device_name": inputs.get("cuda_device_name"),
        "cuda_device_index": inputs.get("cuda_device_index"),
        "cuda_device_capability": inputs.get("cuda_device_capability"),
        "torch_version": inputs.get("torch_version"),
        "torch_cuda_version": inputs.get("torch_cuda_version"),
    }


def _hardware_comparison(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    cand = _trace_hardware(candidate)
    ref = _trace_hardware(reference)
    same_device_type = (
        cand["device_type"] is not None
        and ref["device_type"] is not None
        and cand["device_type"] == ref["device_type"]
    )
    cand_gpu = cand.get("cuda_device_name")
    ref_gpu = ref.get("cuda_device_name")
    same_cuda_device_name = (
        cand_gpu is not None
        and ref_gpu is not None
        and cand_gpu == ref_gpu
    )
    unknown = cand["device_type"] is None or ref["device_type"] is None
    if cand["device_type"] == "cuda" or ref["device_type"] == "cuda":
        unknown = unknown or cand_gpu is None or ref_gpu is None
    if unknown:
        status = "unknown_hardware_identity"
    elif same_device_type and (cand["device_type"] != "cuda" or same_cuda_device_name):
        status = "same_hardware_identity"
    else:
        status = "hardware_mismatch"
    return {
        "status": status,
        "candidate": cand,
        "reference": ref,
        "same_device_type": same_device_type,
        "same_cuda_device_name": same_cuda_device_name,
        "allocator_use_allowed": status == "same_hardware_identity",
        "note": (
            "Use ranked pair atoms for Lagrangian allocation only when hardware "
            "identity matches. Cross-hardware or unknown-hardware comparisons "
            "are calibration/debug signal, not dispatch-grade atom evidence."
        ),
    }


def _pose_first_order_excess(delta_pose_dist: float, *, candidate_avg_pose: float, n: int) -> float:
    if candidate_avg_pose <= 0.0:
        return 0.0
    return (5.0 / math.sqrt(10.0 * candidate_avg_pose)) * (delta_pose_dist / n)


def compare_trace_pair(
    *,
    candidate_label: str,
    candidate: dict[str, Any],
    reference_label: str,
    reference: dict[str, Any],
    top_k: int,
    uncompressed_bytes: int,
) -> dict[str, Any]:
    n = int(candidate["n_samples"])
    if n != int(reference["n_samples"]):
        raise ValueError(f"sample count mismatch: candidate={n} reference={reference['n_samples']}")
    candidate_score = _score_from_trace(candidate, uncompressed_bytes=uncompressed_bytes)
    reference_score = _score_from_trace(reference, uncompressed_bytes=uncompressed_bytes)
    hardware_comparison = _hardware_comparison(candidate, reference)

    pair_deltas: list[dict[str, Any]] = []
    for pair, cand_sample in candidate["_samples_by_pair"].items():
        ref_sample = reference["_samples_by_pair"].get(pair)
        if ref_sample is None:
            raise ValueError(f"reference {reference_label} missing pair_index={pair}")
        cand_pose = _finite_float(cand_sample["posenet_dist"], field="candidate posenet_dist")
        ref_pose = _finite_float(ref_sample["posenet_dist"], field="reference posenet_dist")
        cand_seg = _finite_float(cand_sample["segnet_dist"], field="candidate segnet_dist")
        ref_seg = _finite_float(ref_sample["segnet_dist"], field="reference segnet_dist")
        delta_pose = cand_pose - ref_pose
        delta_seg = cand_seg - ref_seg
        seg_excess = 100.0 * delta_seg / n
        pose_excess = _pose_first_order_excess(
            delta_pose,
            candidate_avg_pose=candidate_score["avg_posenet_dist"],
            n=n,
        )
        pair_deltas.append(
            {
                "pair_index": pair,
                "video_name": cand_sample.get("video_name"),
                "frame_indices": cand_sample.get("frame_indices"),
                "delta_posenet_dist": delta_pose,
                "delta_segnet_dist": delta_seg,
                "score_pose_excess_first_order_at_candidate": pose_excess,
                "score_seg_excess_exact": seg_excess,
                "score_combined_excess_first_order": pose_excess + seg_excess,
                "candidate_posenet_dist": cand_pose,
                "candidate_segnet_dist": cand_seg,
                "reference_posenet_dist": ref_pose,
                "reference_segnet_dist": ref_seg,
            }
        )

    k = max(0, min(int(top_k), len(pair_deltas)))
    component_delta = {
        "score_delta_total": (
            candidate_score["score_recomputed_from_components"]
            - reference_score["score_recomputed_from_components"]
        ),
        "score_delta_pose_exact": (
            candidate_score["score_pose_contribution"]
            - reference_score["score_pose_contribution"]
        ),
        "score_delta_seg_exact": (
            candidate_score["score_seg_contribution"]
            - reference_score["score_seg_contribution"]
        ),
        "score_delta_rate_exact": (
            candidate_score["score_rate_contribution"]
            - reference_score["score_rate_contribution"]
        ),
        "archive_delta_bytes": int(candidate["archive_size_bytes"]) - int(reference["archive_size_bytes"]),
        "avg_posenet_delta": candidate_score["avg_posenet_dist"] - reference_score["avg_posenet_dist"],
        "avg_segnet_delta": candidate_score["avg_segnet_dist"] - reference_score["avg_segnet_dist"],
        "repair_break_even_bytes_per_score_point": uncompressed_bytes / 25.0,
        "one_score_millipoint_bytes": uncompressed_bytes * 0.001 / 25.0,
    }
    return {
        "reference": _trace_identity(reference_label, reference, uncompressed_bytes=uncompressed_bytes),
        "hardware_comparison": hardware_comparison,
        "component_delta_vs_reference": component_delta,
        "ranking_note": (
            "Seg excess is exact additive contribution. Pose excess is first-order "
            "at the candidate average because sqrt(10*mean_pose) is nonlinear."
        ),
        "top_excess_combined_samples": sorted(
            pair_deltas,
            key=lambda item: item["score_combined_excess_first_order"],
            reverse=True,
        )[:k],
        "top_excess_pose_samples": sorted(
            pair_deltas,
            key=lambda item: item["score_pose_excess_first_order_at_candidate"],
            reverse=True,
        )[:k],
        "top_excess_seg_samples": sorted(
            pair_deltas,
            key=lambda item: item["score_seg_excess_exact"],
            reverse=True,
        )[:k],
        "pair_deltas": pair_deltas,
    }


def build_comparison(
    *,
    candidate_label: str,
    candidate_trace: Path,
    reference_specs: list[tuple[str, Path]],
    top_k: int,
    uncompressed_bytes: int,
) -> dict[str, Any]:
    candidate = _load_trace(candidate_trace)
    references = [(label, _load_trace(path)) for label, path in reference_specs]
    comparisons = [
        compare_trace_pair(
            candidate_label=candidate_label,
            candidate=candidate,
            reference_label=label,
            reference=reference,
            top_k=top_k,
            uncompressed_bytes=uncompressed_bytes,
        )
        for label, reference in references
    ]
    best = min(
        comparisons,
        key=lambda item: item["reference"]["score_recomputed_from_components"],
    )
    hardware_statuses = [item["hardware_comparison"]["status"] for item in comparisons]
    allocator_use_allowed = all(
        item["hardware_comparison"]["allocator_use_allowed"] for item in comparisons
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": "experiments/compare_component_traces.py",
        "recorded_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "score_claim": False,
        "evidence_grade": (
            "diagnostic_trace_comparison_same_hardware"
            if allocator_use_allowed
            else "diagnostic_trace_comparison_hardware_untrusted"
        ),
        "uncompressed_size_bytes": uncompressed_bytes,
        "candidate": _trace_identity(candidate_label, candidate, uncompressed_bytes=uncompressed_bytes),
        "references": comparisons,
        "hardware_statuses": hardware_statuses,
        "allocator_use_allowed": allocator_use_allowed,
        "best_reference_label": best["reference"]["label"],
        "best_reference_component_delta": best["component_delta_vs_reference"],
        "recommended_next_use": (
            "Use top_excess_* samples as priors for repair atoms or renderer-basin "
            "diagnostics, then require exact CUDA archive eval for any archive built "
            "from those atoms."
        ),
    }


def _parse_reference(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("reference must be LABEL=PATH")
    label, path = raw.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("reference label must be non-empty")
    return label, Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--candidate-trace", type=Path, required=True)
    parser.add_argument(
        "--reference",
        action="append",
        type=_parse_reference,
        required=True,
        help="Reference trace as LABEL=PATH. May be repeated.",
    )
    parser.add_argument("--top-k", type=int, default=80)
    parser.add_argument("--uncompressed-size-bytes", type=int, default=CONTEST_UNCOMPRESSED_BYTES)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = build_comparison(
        candidate_label=args.candidate_label,
        candidate_trace=args.candidate_trace,
        reference_specs=args.reference,
        top_k=args.top_k,
        uncompressed_bytes=args.uncompressed_size_bytes,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output_json": str(args.output_json),
        "candidate": payload["candidate"]["label"],
        "best_reference_label": payload["best_reference_label"],
        "score_delta_total": payload["best_reference_component_delta"]["score_delta_total"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
