#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Summarize scorer-device x inflate-device auth-eval artifacts.

This is a diagnostic/xray tool. It never promotes, ranks, or kills lanes. It
exists to keep CPU/CUDA/inflate-device comparisons apples-to-apples by showing
archive SHA, runtime tree SHA, raw-output aggregate SHA, score axis, and
baseline deltas in one canonical shape.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

try:
    from tools.auth_eval_records import (
        inflated_output_manifest_summary,
        parse_auth_eval_payload,
        runtime_tree_sha256,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    from auth_eval_records import (
        inflated_output_manifest_summary,
        parse_auth_eval_payload,
        runtime_tree_sha256,
    )
from tac.auth_eval_schema import contest_formula_score  # noqa: E402
from tac.repo_io import json_text, read_json, write_json  # noqa: E402


def _parse_labeled_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("artifact must be LABEL=PATH")
    label, raw_path = value.split("=", 1)
    label = label.strip()
    if not label:
        raise argparse.ArgumentTypeError("artifact label must be non-empty")
    return label, Path(raw_path)


def _provenance(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("provenance")
    return value if isinstance(value, dict) else {}


def _artifact_summary(label: str, path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        return {
            "label": label,
            "path": str(path),
            "parse_ok": False,
            "blockers": ["artifact_not_json_object"],
        }
    record = parse_auth_eval_payload(payload)
    if record is None:
        return {
            "label": label,
            "path": str(path),
            "parse_ok": False,
            "blockers": ["auth_eval_unparseable"],
        }

    raw = inflated_output_manifest_summary(payload)
    prov = _provenance(payload)
    recomputed = None
    if (
        record.avg_segnet_dist is not None
        and record.avg_posenet_dist is not None
        and record.archive_bytes is not None
    ):
        recomputed = contest_formula_score(
            seg_dist=float(record.avg_segnet_dist),
            pose_dist=float(record.avg_posenet_dist),
            archive_bytes=int(record.archive_bytes),
        )
    return {
        "label": label,
        "path": str(path),
        "parse_ok": True,
        "blockers": [],
        "score": record.score,
        "score_recomputed": recomputed,
        "score_axis": record.score_axis,
        "scorer_device": (
            payload.get("scorer_device")
            or prov.get("scorer_device")
            or record.device
        ),
        "device": record.device,
        "evidence_grade": record.evidence_grade,
        "score_claim": record.score_claim_valid,
        "promotion_eligible": record.promotion_eligible,
        "rank_or_kill_eligible": record.rank_or_kill_eligible,
        "samples": record.samples,
        "avg_posenet_dist": record.avg_posenet_dist,
        "avg_segnet_dist": record.avg_segnet_dist,
        "archive_sha256": record.archive_sha256,
        "archive_bytes": record.archive_bytes,
        "runtime_tree_sha256": runtime_tree_sha256(payload),
        "raw_output_aggregate_sha256": raw.get("aggregate_sha256") if raw else None,
        "raw_output_manifest": raw,
        "inflate_device_policy": (
            payload.get("inflate_device_policy")
            or prov.get("inflate_device_policy")
            or "auto"
        ),
        "inflate_env_override_mode": prov.get("inflate_env_override_mode"),
        "diagnostic_blockers": payload.get("diagnostic_blockers") or [],
        "hardware": payload.get("hardware") or prov.get("hardware"),
    }


def _delta_row(entry: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    def delta(key: str) -> float | None:
        if entry.get(key) is None or baseline.get(key) is None:
            return None
        return float(entry[key]) - float(baseline[key])

    return {
        "label": entry["label"],
        "baseline_label": baseline["label"],
        "score_delta": delta("score"),
        "pose_delta": delta("avg_posenet_dist"),
        "seg_delta": delta("avg_segnet_dist"),
        "same_archive_sha256": (
            entry.get("archive_sha256") == baseline.get("archive_sha256")
            if entry.get("archive_sha256") and baseline.get("archive_sha256")
            else None
        ),
        "same_runtime_tree_sha256": (
            entry.get("runtime_tree_sha256") == baseline.get("runtime_tree_sha256")
            if entry.get("runtime_tree_sha256") and baseline.get("runtime_tree_sha256")
            else None
        ),
        "same_raw_output_aggregate_sha256": (
            entry.get("raw_output_aggregate_sha256")
            == baseline.get("raw_output_aggregate_sha256")
            if entry.get("raw_output_aggregate_sha256")
            and baseline.get("raw_output_aggregate_sha256")
            else None
        ),
    }


def analyze_matrix(
    artifacts: list[tuple[str, Path]],
    *,
    baseline_label: str | None = None,
) -> dict[str, Any]:
    entries = [_artifact_summary(label, path) for label, path in artifacts]
    by_label = {entry["label"]: entry for entry in entries}
    baseline = by_label.get(baseline_label or "") if baseline_label else None
    raw_groups: dict[str, list[str]] = {}
    runtime_groups: dict[str, list[str]] = {}
    for entry in entries:
        raw_sha = entry.get("raw_output_aggregate_sha256")
        runtime_sha = entry.get("runtime_tree_sha256")
        if isinstance(raw_sha, str) and raw_sha:
            raw_groups.setdefault(raw_sha, []).append(entry["label"])
        if isinstance(runtime_sha, str) and runtime_sha:
            runtime_groups.setdefault(runtime_sha, []).append(entry["label"])
    return {
        "schema": "device_axis_eval_matrix_analysis.v1",
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "entries": entries,
        "baseline_label": baseline.get("label") if baseline else None,
        "deltas_vs_baseline": (
            [_delta_row(entry, baseline) for entry in entries if entry is not baseline]
            if baseline
            else []
        ),
        "raw_output_groups": raw_groups,
        "runtime_tree_groups": runtime_groups,
        "blockers": [
            f"{entry['label']}:{blocker}"
            for entry in entries
            for blocker in entry.get("blockers", [])
        ],
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "notes": [
            "This analysis is diagnostic only; use canonical auth-eval artifacts for score claims.",
            "A non-auto inflate_device_policy is never promotion-eligible.",
            "Raw-output SHA equality localizes drift after inflate; raw-output SHA mismatch localizes drift at or before inflate.",
            "Do not infer CPU or CUDA is globally better from this matrix; keep the conclusion per archive/runtime.",
        ],
    }


def format_markdown(analysis: dict[str, Any]) -> str:
    lines = [
        "# Device-axis auth-eval matrix analysis",
        "",
        f"generated_at_utc: `{analysis['created_at_utc']}`",
        "score_claim: `false`",
        "promotion_eligible: `false`",
        "",
        "| Label | Axis | Scorer | Inflate | Score | Pose | Seg | Runtime SHA | Raw SHA |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for entry in analysis["entries"]:
        lines.append(
            "| {label} | {axis} | {scorer} | {inflate} | {score} | {pose} | {seg} | {runtime} | {raw} |".format(
                label=entry["label"],
                axis=entry.get("score_axis"),
                scorer=entry.get("scorer_device"),
                inflate=entry.get("inflate_device_policy"),
                score=entry.get("score"),
                pose=entry.get("avg_posenet_dist"),
                seg=entry.get("avg_segnet_dist"),
                runtime=str(entry.get("runtime_tree_sha256") or "")[:12],
                raw=str(entry.get("raw_output_aggregate_sha256") or "")[:12],
            )
        )
    if analysis["deltas_vs_baseline"]:
        lines.extend(["", "## Deltas vs baseline", ""])
        for row in analysis["deltas_vs_baseline"]:
            lines.append(
                "- {label}: score_delta={score_delta}, pose_delta={pose_delta}, "
                "seg_delta={seg_delta}, same_runtime={same_runtime}, same_raw={same_raw}".format(
                    label=row["label"],
                    score_delta=row["score_delta"],
                    pose_delta=row["pose_delta"],
                    seg_delta=row["seg_delta"],
                    same_runtime=row["same_runtime_tree_sha256"],
                    same_raw=row["same_raw_output_aggregate_sha256"],
                )
            )
    lines.extend(["", "## Notes", ""])
    lines.extend(f"- {note}" for note in analysis["notes"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        action="append",
        type=_parse_labeled_path,
        default=[],
        metavar="LABEL=PATH",
        help="Auth-eval JSON artifact to include in the matrix.",
    )
    parser.add_argument("--baseline-label")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    args = parser.parse_args()
    if not args.artifact:
        raise SystemExit("at least one --artifact LABEL=PATH is required")

    analysis = analyze_matrix(args.artifact, baseline_label=args.baseline_label)
    if args.json_out:
        write_json(args.json_out, analysis)
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(format_markdown(analysis), encoding="utf-8")
    print(json_text(analysis))
    return 0 if not analysis["blockers"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
