#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare exact CPU and CUDA auth-eval results for the same archive.

This xray closes the loop that the one-way CPU/CUDA drift predictor cannot:
when both axes already exist for a byte-identical packet, compute component
deltas, byte-equivalent gaps, and the precise reason a CPU-positive selector
does or does not transfer to CUDA.

Diagnostic only. It never promotes, ranks, dispatches, or claims a score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from tools.auth_eval_records import (  # noqa: E402
    AuthEvalRecord,
    parse_auth_eval_payload,
    runtime_tree_sha256,
)

SCHEMA = "xray_paired_cpu_cuda_axis_delta_v1"
TOOL = "tools/xray_paired_cpu_cuda_axis_delta.py"
DEFAULT_ORIGINAL_UNCOMPRESSED_SIZE_BYTES = 37_545_489


@dataclass(frozen=True)
class AxisInput:
    path: str
    payload_sha256: str
    record: AuthEvalRecord
    runtime_tree_sha256: str | None
    inflated_outputs_manifest: dict[str, Any] | None


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not load JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _load_axis_input(
    *,
    path: Path,
    required_axis: str,
    inflated_outputs_manifest_path: Path | None = None,
) -> AxisInput:
    payload = _load_json(path)
    record = parse_auth_eval_payload(payload)
    if record is None or record.score is None:
        raise ValueError(f"{path} is not a parseable auth-eval result")
    if record.score_axis != required_axis:
        raise ValueError(f"{path} axis={record.score_axis!r}; expected {required_axis!r}")

    manifest_payload = None
    if inflated_outputs_manifest_path is not None:
        manifest_payload = _load_json(inflated_outputs_manifest_path)
        aggregate = manifest_payload.get("aggregate_sha256")
        if not isinstance(aggregate, str) or not aggregate:
            raise ValueError(
                f"{inflated_outputs_manifest_path} missing aggregate_sha256"
            )

    return AxisInput(
        path=str(path),
        payload_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        record=record,
        runtime_tree_sha256=runtime_tree_sha256(payload),
        inflated_outputs_manifest=_summarize_inflated_manifest(
            manifest_payload,
            path=inflated_outputs_manifest_path,
        )
        if manifest_payload is not None
        else None,
    )


def _summarize_inflated_manifest(
    payload: dict[str, Any],
    *,
    path: Path | None,
) -> dict[str, Any]:
    files = payload.get("files")
    first_file = None
    if isinstance(files, list) and files and isinstance(files[0], dict):
        first = files[0]
        first_file = {
            key: first.get(key)
            for key in ("relative_path", "video_name", "bytes", "sha256")
            if key in first
        }
    return {
        "path": str(path) if path else None,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path else None,
        "aggregate_sha256": payload.get("aggregate_sha256"),
        "raw_file_count": payload.get("raw_file_count"),
        "total_bytes": payload.get("total_bytes"),
        "first_file": first_file,
    }


def _require_number(value: float | int | None, *, name: str) -> float:
    if value is None:
        raise ValueError(f"missing numeric field: {name}")
    return float(value)


def _score_terms(record: AuthEvalRecord) -> dict[str, float]:
    seg = _require_number(record.avg_segnet_dist, name="avg_segnet_dist")
    pose = _require_number(record.avg_posenet_dist, name="avg_posenet_dist")
    rate = _require_number(record.rate_unscaled, name="rate_unscaled")
    return {
        "score": _require_number(record.score, name="score"),
        "seg_dist": seg,
        "pose_dist": pose,
        "rate_unscaled": rate,
        "seg_score_contribution": 100.0 * seg,
        "pose_score_contribution": math.sqrt(10.0 * pose),
        "rate_score_contribution": 25.0 * rate,
    }


def _byte_equivalent(score_delta: float, original_uncompressed_size_bytes: int) -> float:
    """Return how many archive bytes equal ``score_delta`` in rate term."""
    return score_delta * float(original_uncompressed_size_bytes) / 25.0


def _gap_to_target(score: float, target_score: float, original_bytes: int) -> dict[str, Any]:
    score_gap = score - target_score
    return {
        "target_score": target_score,
        "score_gap": score_gap,
        "byte_gap_if_components_unchanged": math.ceil(
            _byte_equivalent(score_gap, original_bytes)
        )
        if score_gap > 0.0
        else 0,
        "already_below_target": score < target_score,
    }


def _dominant_delta(delta_terms: dict[str, float]) -> str:
    candidates = {
        "seg": abs(delta_terms["seg_score_contribution_delta"]),
        "pose": abs(delta_terms["pose_score_contribution_delta"]),
        "rate": abs(delta_terms["rate_score_contribution_delta"]),
    }
    return max(candidates.items(), key=lambda kv: kv[1])[0]


def build_report(
    *,
    cpu_axis: AxisInput,
    cuda_axis: AxisInput,
    label: str,
    target_score: float,
    original_uncompressed_size_bytes: int = DEFAULT_ORIGINAL_UNCOMPRESSED_SIZE_BYTES,
) -> dict[str, Any]:
    cpu = cpu_axis.record
    cuda = cuda_axis.record
    blockers: list[str] = []

    if cpu.archive_sha256 and cuda.archive_sha256 and cpu.archive_sha256 != cuda.archive_sha256:
        raise ValueError(
            "CPU and CUDA records point at different archive SHA-256 values: "
            f"{cpu.archive_sha256} != {cuda.archive_sha256}"
        )
    if cpu.archive_bytes is not None and cuda.archive_bytes is not None and cpu.archive_bytes != cuda.archive_bytes:
        raise ValueError(
            "CPU and CUDA records point at different archive byte counts: "
            f"{cpu.archive_bytes} != {cuda.archive_bytes}"
        )

    archive_sha = cpu.archive_sha256 or cuda.archive_sha256
    archive_bytes = cpu.archive_bytes if cpu.archive_bytes is not None else cuda.archive_bytes
    if archive_sha is None:
        blockers.append("archive_sha256_missing_from_one_or_both_records")
    if archive_bytes is None:
        blockers.append("archive_bytes_missing_from_one_or_both_records")
    if cpu.samples != 600 or cuda.samples != 600:
        blockers.append("paired_axis_requires_600_samples_on_both_axes")
    if cpu_axis.runtime_tree_sha256 != cuda_axis.runtime_tree_sha256:
        # Runtime tree SHA can differ by root path; content tree parity is usually
        # checked elsewhere. This is diagnostic, not fail-closed.
        blockers.append("runtime_tree_sha256_differs_check_content_tree_before_promotion")

    cpu_terms = _score_terms(cpu)
    cuda_terms = _score_terms(cuda)
    delta_terms = {
        "score_delta_cuda_minus_cpu": cuda_terms["score"] - cpu_terms["score"],
        "seg_dist_delta_cuda_minus_cpu": cuda_terms["seg_dist"] - cpu_terms["seg_dist"],
        "pose_dist_delta_cuda_minus_cpu": cuda_terms["pose_dist"] - cpu_terms["pose_dist"],
        "rate_unscaled_delta_cuda_minus_cpu": cuda_terms["rate_unscaled"] - cpu_terms["rate_unscaled"],
        "seg_score_contribution_delta": (
            cuda_terms["seg_score_contribution"] - cpu_terms["seg_score_contribution"]
        ),
        "pose_score_contribution_delta": (
            cuda_terms["pose_score_contribution"] - cpu_terms["pose_score_contribution"]
        ),
        "rate_score_contribution_delta": (
            cuda_terms["rate_score_contribution"] - cpu_terms["rate_score_contribution"]
        ),
    }
    dominant = _dominant_delta(delta_terms)
    score_delta = delta_terms["score_delta_cuda_minus_cpu"]
    byte_equiv = _byte_equivalent(score_delta, original_uncompressed_size_bytes)

    raw_output_comparison = None
    if cpu_axis.inflated_outputs_manifest and cuda_axis.inflated_outputs_manifest:
        cpu_hash = cpu_axis.inflated_outputs_manifest.get("aggregate_sha256")
        cuda_hash = cuda_axis.inflated_outputs_manifest.get("aggregate_sha256")
        raw_output_comparison = {
            "cpu_aggregate_sha256": cpu_hash,
            "cuda_aggregate_sha256": cuda_hash,
            "aggregate_sha256_match": bool(cpu_hash and cuda_hash and cpu_hash == cuda_hash),
        }

    if score_delta > 0.0 and abs(delta_terms["rate_score_contribution_delta"]) < 1e-12:
        verdict = "cpu_positive_cuda_miss_due_to_component_drift"
    elif score_delta <= 0.0:
        verdict = "cuda_matches_or_beats_cpu"
    else:
        verdict = "mixed_axis_gap"

    state_hash = hashlib.sha256(
        json.dumps(
            {
                "cpu": cpu_axis.payload_sha256,
                "cuda": cuda_axis.payload_sha256,
                "target_score": target_score,
                "label": label,
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:16]

    return {
        "schema_version": SCHEMA,
        "tool": TOOL,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "from_state_hash": state_hash,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": "diagnostic_only_paired_exact_axis_artifacts",
        "label": label,
        "archive": {
            "sha256": archive_sha,
            "bytes": archive_bytes,
        },
        "inputs": {
            "contest_cpu": {
                "path": cpu_axis.path,
                "payload_sha256": cpu_axis.payload_sha256,
                "record": asdict(cpu),
                "runtime_tree_sha256": cpu_axis.runtime_tree_sha256,
                "inflated_outputs_manifest": cpu_axis.inflated_outputs_manifest,
            },
            "contest_cuda": {
                "path": cuda_axis.path,
                "payload_sha256": cuda_axis.payload_sha256,
                "record": asdict(cuda),
                "runtime_tree_sha256": cuda_axis.runtime_tree_sha256,
                "inflated_outputs_manifest": cuda_axis.inflated_outputs_manifest,
            },
        },
        "components": {
            "contest_cpu": cpu_terms,
            "contest_cuda": cuda_terms,
            "delta_cuda_minus_cpu": delta_terms,
            "dominant_score_delta_component": dominant,
            "score_delta_byte_equivalent": byte_equiv,
        },
        "target_gaps": {
            "contest_cpu": _gap_to_target(
                cpu_terms["score"],
                target_score,
                original_uncompressed_size_bytes,
            ),
            "contest_cuda": _gap_to_target(
                cuda_terms["score"],
                target_score,
                original_uncompressed_size_bytes,
            ),
        },
        "raw_output_comparison": raw_output_comparison,
        "classification": verdict,
        "dispatch_blockers": [
            "diagnostic_tool_no_score_or_dispatch_authority",
            "paired_axis_delta_is_not_a_new_archive",
            *blockers,
        ],
        "recommended_next_step": _recommended_next_step(
            verdict=verdict,
            dominant=dominant,
            target_score=target_score,
            cpu_gap_bytes=_gap_to_target(
                cpu_terms["score"],
                target_score,
                original_uncompressed_size_bytes,
            )["byte_gap_if_components_unchanged"],
            cuda_gap_bytes=_gap_to_target(
                cuda_terms["score"],
                target_score,
                original_uncompressed_size_bytes,
            )["byte_gap_if_components_unchanged"],
        ),
    }


def _recommended_next_step(
    *,
    verdict: str,
    dominant: str,
    target_score: float,
    cpu_gap_bytes: int,
    cuda_gap_bytes: int,
) -> str:
    if verdict == "cpu_positive_cuda_miss_due_to_component_drift":
        return (
            f"Do not spend on rate-only polishing for CUDA: the paired-axis gap is "
            f"component-dominated by {dominant}. Use this packet as calibration for "
            "CUDA-in-loop selector rows, or rebuild the selector with a charged "
            "component objective before exact eval."
        )
    if verdict == "cuda_matches_or_beats_cpu":
        if cpu_gap_bytes or cuda_gap_bytes:
            return (
                f"Both axes still need byte/component movement to beat target "
                f"{target_score:.6f}; rate-only byte gaps are CPU={cpu_gap_bytes}, "
                f"CUDA={cuda_gap_bytes} if components are unchanged."
            )
        return "Both axes are already below target; run promotion/compliance gates before any claim."
    return (
        f"Mixed CPU/CUDA gap; inspect {dominant} component first. Byte-only gaps "
        f"to target are CPU={cpu_gap_bytes}, CUDA={cuda_gap_bytes} if components "
        "are unchanged."
    )


def render_markdown(report: dict[str, Any], regen_header: str) -> str:
    cpu = report["components"]["contest_cpu"]
    cuda = report["components"]["contest_cuda"]
    delta = report["components"]["delta_cuda_minus_cpu"]
    gaps = report["target_gaps"]
    lines = [regen_header, ""]
    lines.append("# Paired CPU/CUDA axis delta xray")
    lines.append("")
    lines.append(
        f"_Schema_: `{report['schema_version']}` · _Generated_: "
        f"`{report['generated_at_utc']}`"
    )
    lines.append("")
    lines.append("## Classification")
    lines.append("")
    lines.append(f"- verdict: `{report['classification']}`")
    lines.append(f"- dominant score-delta component: `{report['components']['dominant_score_delta_component']}`")
    lines.append(f"- score-delta byte equivalent: `{report['components']['score_delta_byte_equivalent']:.1f}` bytes")
    lines.append(f"- recommended next step: {report['recommended_next_step']}")
    lines.append("")
    lines.append("## Archive")
    lines.append("")
    lines.append(f"- sha256: `{report['archive']['sha256']}`")
    lines.append(f"- bytes: `{report['archive']['bytes']}`")
    lines.append("")
    lines.append("## Axis scores")
    lines.append("")
    lines.append("| axis | score | seg contrib | pose contrib | rate contrib |")
    lines.append("|---|---:|---:|---:|---:|")
    lines.append(
        f"| contest-CPU | {cpu['score']:.12f} | {cpu['seg_score_contribution']:.12f} | "
        f"{cpu['pose_score_contribution']:.12f} | {cpu['rate_score_contribution']:.12f} |"
    )
    lines.append(
        f"| contest-CUDA | {cuda['score']:.12f} | {cuda['seg_score_contribution']:.12f} | "
        f"{cuda['pose_score_contribution']:.12f} | {cuda['rate_score_contribution']:.12f} |"
    )
    lines.append("")
    lines.append("## CUDA minus CPU")
    lines.append("")
    lines.append(f"- total score delta: `{delta['score_delta_cuda_minus_cpu']:.12f}`")
    lines.append(f"- seg contribution delta: `{delta['seg_score_contribution_delta']:.12f}`")
    lines.append(f"- pose contribution delta: `{delta['pose_score_contribution_delta']:.12f}`")
    lines.append(f"- rate contribution delta: `{delta['rate_score_contribution_delta']:.12f}`")
    lines.append("")
    lines.append("## Target gaps")
    lines.append("")
    lines.append("| axis | target | score_gap | byte_gap_if_components_unchanged |")
    lines.append("|---|---:|---:|---:|")
    for axis_key, label in (("contest_cpu", "contest-CPU"), ("contest_cuda", "contest-CUDA")):
        g = gaps[axis_key]
        lines.append(
            f"| {label} | {g['target_score']:.6f} | {g['score_gap']:.12f} | "
            f"{g['byte_gap_if_components_unchanged']} |"
        )
    raw = report.get("raw_output_comparison")
    if raw:
        lines.append("")
        lines.append("## Inflated output aggregate")
        lines.append("")
        lines.append(f"- CPU aggregate: `{raw['cpu_aggregate_sha256']}`")
        lines.append(f"- CUDA aggregate: `{raw['cuda_aggregate_sha256']}`")
        lines.append(f"- aggregate match: `{raw['aggregate_sha256_match']}`")
    lines.append("")
    lines.append(
        "_Tag_: `[diagnostic: paired CPU/CUDA axis delta]`. This report is not "
        "a score claim, not a promotion artifact, and not dispatch authority."
    )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare paired contest-CPU and contest-CUDA auth-eval artifacts."
    )
    parser.add_argument("--cpu-auth-eval-json", type=Path, required=True)
    parser.add_argument("--cuda-auth-eval-json", type=Path, required=True)
    parser.add_argument("--cpu-inflated-outputs-manifest", type=Path, default=None)
    parser.add_argument("--cuda-inflated-outputs-manifest", type=Path, default=None)
    parser.add_argument("--label", default="paired_axis")
    parser.add_argument("--target-score", type=float, default=0.192)
    parser.add_argument(
        "--original-uncompressed-size-bytes",
        type=int,
        default=DEFAULT_ORIGINAL_UNCOMPRESSED_SIZE_BYTES,
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        cpu_axis = _load_axis_input(
            path=args.cpu_auth_eval_json,
            required_axis="contest_cpu",
            inflated_outputs_manifest_path=args.cpu_inflated_outputs_manifest,
        )
        cuda_axis = _load_axis_input(
            path=args.cuda_auth_eval_json,
            required_axis="contest_cuda",
            inflated_outputs_manifest_path=args.cuda_inflated_outputs_manifest,
        )
        report = build_report(
            cpu_axis=cpu_axis,
            cuda_axis=cuda_axis,
            label=args.label,
            target_score=args.target_score,
            original_uncompressed_size_bytes=args.original_uncompressed_size_bytes,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.output_dir or (
        REPO_ROOT / "experiments" / "results" / f"xray_paired_cpu_cuda_axis_delta_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / "paired_axis_delta.json"
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True))

    regen = (
        f"<!-- generated_at: {report['generated_at_utc']}, "
        f"from_state_hash: {report['from_state_hash']} -->"
    )
    out_md = out_dir / "paired_axis_delta.md"
    out_md.write_text(render_markdown(report, regen))

    parts = [
        ".venv/bin/python tools/xray_paired_cpu_cuda_axis_delta.py",
        f"--cpu-auth-eval-json {shlex.quote(str(args.cpu_auth_eval_json))}",
        f"--cuda-auth-eval-json {shlex.quote(str(args.cuda_auth_eval_json))}",
    ]
    if args.cpu_inflated_outputs_manifest:
        parts.append(
            f"--cpu-inflated-outputs-manifest {shlex.quote(str(args.cpu_inflated_outputs_manifest))}"
        )
    if args.cuda_inflated_outputs_manifest:
        parts.append(
            f"--cuda-inflated-outputs-manifest {shlex.quote(str(args.cuda_inflated_outputs_manifest))}"
        )
    parts.extend([
        f"--label {shlex.quote(args.label)}",
        f"--target-score {args.target_score}",
        f"--original-uncompressed-size-bytes {args.original_uncompressed_size_bytes}",
    ])
    (out_dir / "rebuild_command.txt").write_text(" \\\n  ".join(parts) + "\n")

    delta = report["components"]["delta_cuda_minus_cpu"]
    print(f"[xray-paired-axis] wrote {out_json}")
    print(f"[xray-paired-axis] wrote {out_md}")
    print(
        "[xray-paired-axis] "
        f"classification={report['classification']} "
        f"delta={delta['score_delta_cuda_minus_cpu']:.12f} "
        f"dominant={report['components']['dominant_score_delta_component']} "
        f"byte_equiv={report['components']['score_delta_byte_equivalent']:.1f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
