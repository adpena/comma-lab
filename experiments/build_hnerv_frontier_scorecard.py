#!/usr/bin/env python3
"""Build a compact scorecard for public HNeRV frontier intake.

The scorecard joins exact CUDA replay artifacts with forensic payload profiles.
It does not evaluate archives and does not promote prediction-only claims; it is
only a deterministic decision table for the next optimization action.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def maybe_load_json(path: Path | None) -> Any | None:
    if path is None or not path.is_file():
        return None
    return load_json(path)


def profile_by_sha(path: Path | None) -> dict[str, dict[str, Any]]:
    payload = maybe_load_json(path)
    if not payload:
        return {}
    return {item["archive_sha256"]: item for item in payload}


def contribution(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get(key)
    return float(value) if isinstance(value, int | float) else None


def row_from_eval(label: str, path: Path, profiles: dict[str, dict[str, Any]]) -> dict[str, Any]:
    payload = load_json(path)
    prov = payload.get("provenance") or {}
    sha = prov.get("archive_sha256")
    profile = profiles.get(sha, {})
    sections = profile.get("sections") or []
    largest_section = max(sections, key=lambda item: item.get("bytes", 0), default=None)
    return {
        "label": label,
        "evidence_grade": "A++"
        if prov.get("device") == "cuda" and prov.get("gpu_t4_match") and payload.get("n_samples") == 600
        else "A",
        "score": contribution(payload, "score_recomputed_from_components"),
        "archive_bytes": payload.get("archive_size_bytes"),
        "archive_sha256": sha,
        "avg_segnet_dist": contribution(payload, "avg_segnet_dist"),
        "avg_posenet_dist": contribution(payload, "avg_posenet_dist"),
        "score_seg_contribution": contribution(payload, "score_seg_contribution"),
        "score_pose_contribution": contribution(payload, "score_pose_contribution"),
        "score_rate_contribution": contribution(payload, "score_rate_contribution"),
        "gpu_model": prov.get("gpu_model"),
        "runtime_tree_sha256": (prov.get("inflate_runtime_manifest") or {}).get(
            "runtime_tree_sha256"
        ),
        "profile_kind": profile.get("kind"),
        "zip_member": profile.get("member_name"),
        "zip_overhead_bytes": profile.get("zip_overhead_bytes"),
        "largest_payload_section": largest_section,
        "eval_artifact": str(path),
    }


def render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# HNeRV Frontier Scorecard",
        "",
        "| label | grade | score | bytes | seg | pose | rate | largest section | archive sha |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in sorted(rows, key=lambda item: item["score"] if item["score"] is not None else 999):
        largest = row.get("largest_payload_section") or {}
        largest_text = (
            f"{largest.get('name')}:{largest.get('bytes')}"
            if largest
            else "n/a"
        )
        lines.append(
            "| {label} | {evidence_grade} | {score:.12f} | {archive_bytes} | {seg:.9f} | {pose:.9f} | {rate:.9f} | `{largest}` | `{sha}` |".format(
                label=row["label"],
                evidence_grade=row["evidence_grade"],
                score=row["score"],
                archive_bytes=row["archive_bytes"],
                seg=row["score_seg_contribution"],
                pose=row["score_pose_contribution"],
                rate=row["score_rate_contribution"],
                largest=largest_text,
                sha=(row["archive_sha256"] or "")[:16],
            )
        )
    lines.extend(
        [
            "",
            "Interpretation: score truth remains the exact CUDA replay JSON. Payload",
            "sections are forensic signals for the next compression action; they do",
            "not imply score deltas without a new exact archive eval.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile-json",
        type=Path,
        help="Payload profile JSON emitted by profile_hnerv_frontier_payloads.py.",
    )
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--md-out", type=Path, required=True)
    parser.add_argument(
        "evals",
        nargs="+",
        help="LABEL=path/to/contest_auth_eval.adjudicated.json",
    )
    args = parser.parse_args()

    profiles = profile_by_sha(args.profile_json)
    rows: list[dict[str, Any]] = []
    for item in args.evals:
        if "=" not in item:
            raise SystemExit(f"expected LABEL=PATH, got {item!r}")
        label, raw_path = item.split("=", 1)
        rows.append(row_from_eval(label, Path(raw_path), profiles))

    payload = {
        "schema_version": 1,
        "tool": "build_hnerv_frontier_scorecard",
        "score_truth": "exact_cuda_auth_eval_json",
        "rows": sorted(rows, key=lambda item: item["score"] if item["score"] is not None else 999),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    args.md_out.parent.mkdir(parents=True, exist_ok=True)
    args.md_out.write_text(render_markdown(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
