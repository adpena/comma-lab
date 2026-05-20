#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Select the next decoder-q candidate queue from feasibility/advisory evidence."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"expected JSON object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _mutation_key(row: dict[str, Any]) -> tuple[str, int, int]:
    mutation = row["mutation"]
    return (
        str(mutation["tensor_name"]),
        int(mutation["q_offset"]),
        int(mutation["delta"]),
    )


def _axis_target_mass(row: dict[str, Any]) -> dict[str, float]:
    evidence = row.get("op3v3_target_evidence")
    if not isinstance(evidence, dict):
        evidence = {}
    axis = evidence.get("axis_score_impact_abs_sum")
    if not isinstance(axis, dict):
        axis = {}
    return {
        "seg": float(axis.get("seg", 0.0)),
        "pose": float(axis.get("pose", 0.0)),
        "rate": float(axis.get("rate", 0.0)),
    }


def _target_mass(row: dict[str, Any]) -> float:
    evidence = row.get("op3v3_target_evidence")
    if isinstance(evidence, dict):
        return float(evidence.get("score_impact_abs_sum", 0.0))
    return 0.0


def _advisory_rows(summary: dict[str, Any] | None) -> dict[str, dict[str, Any]]:
    if not summary:
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for row in summary.get("candidates", []):
        if not isinstance(row, dict):
            continue
        candidate_id = row.get("candidate_id")
        advisory = row.get("advisory_eval")
        manifest = row.get("mutation_manifest")
        if not candidate_id or not isinstance(advisory, dict) or not isinstance(manifest, dict):
            continue
        mutation_row = manifest.get("mutation_row")
        if not isinstance(mutation_row, dict):
            continue
        rows[str(candidate_id)] = {
            "candidate_id": str(candidate_id),
            "mutation_key": _mutation_key(mutation_row),
            "mutation_row": mutation_row,
            "returncode": advisory.get("returncode"),
            "canonical_score": advisory.get("canonical_score"),
            "avg_posenet_dist": advisory.get("avg_posenet_dist"),
            "avg_segnet_dist": advisory.get("avg_segnet_dist"),
            "delta_vs_baseline_score": row.get("delta_vs_baseline_score"),
            "raw_comparison": row.get("raw_comparison"),
        }
    return rows


def _attach_advisory(
    fixed_rows: list[dict[str, Any]],
    advisory_by_id: dict[str, dict[str, Any]],
) -> dict[tuple[str, int, int], dict[str, Any]]:
    by_key: dict[tuple[str, int, int], dict[str, Any]] = {}
    mutation_to_id = {row["mutation_id"]: row for row in fixed_rows}
    for candidate_id, advisory in advisory_by_id.items():
        source = mutation_to_id.get(candidate_id)
        if source is None:
            continue
        enriched = dict(source)
        enriched["advisory"] = advisory
        by_key[_mutation_key(source)] = enriched
    return by_key


def _outcome_models(
    fixed_rows: list[dict[str, Any]],
    advisory_by_key: dict[tuple[str, int, int], dict[str, Any]],
    baseline_score: float | None,
) -> list[dict[str, Any]]:
    by_offset: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in fixed_rows:
        tensor_name, q_offset, _delta = _mutation_key(row)
        by_offset[(tensor_name, q_offset)].append(row)

    models = []
    for (tensor_name, q_offset), rows in sorted(by_offset.items()):
        measured = []
        for row in rows:
            key = _mutation_key(row)
            advisory_row = advisory_by_key.get(key)
            if not advisory_row:
                continue
            advisory = advisory_row["advisory"]
            if advisory.get("returncode") != 0 or advisory.get("canonical_score") is None:
                continue
            score = float(advisory["canonical_score"])
            measured.append(
                {
                    "delta": int(row["mutation"]["delta"]),
                    "candidate_id": row["mutation_id"],
                    "score": score,
                    "delta_vs_baseline_score": (
                        score - baseline_score if baseline_score is not None else advisory.get("delta_vs_baseline_score")
                    ),
                    "avg_posenet_dist": advisory.get("avg_posenet_dist"),
                    "avg_segnet_dist": advisory.get("avg_segnet_dist"),
                    "raw_comparison": advisory.get("raw_comparison"),
                }
            )
        measured.sort(key=lambda item: int(item["delta"]))
        best = min(measured, key=lambda item: float(item["score"])) if measured else None
        signed_slope = None
        measured_by_delta = {int(item["delta"]): item for item in measured}
        if -1 in measured_by_delta and 1 in measured_by_delta:
            signed_slope = (
                float(measured_by_delta[1]["score"])
                - float(measured_by_delta[-1]["score"])
            ) / 2.0
        elif -2 in measured_by_delta and 2 in measured_by_delta:
            signed_slope = (
                float(measured_by_delta[2]["score"])
                - float(measured_by_delta[-2]["score"])
            ) / 4.0
        target_mass = max((_target_mass(row) for row in rows), default=0.0)
        models.append(
            {
                "tensor_name": tensor_name,
                "q_offset": q_offset,
                "target_mass": target_mass,
                "axis_target_mass": max(
                    (_axis_target_mass(row) for row in rows),
                    key=lambda axis: axis["seg"] + axis["pose"] + axis["rate"],
                    default={"seg": 0.0, "pose": 0.0, "rate": 0.0},
                ),
                "measured": measured,
                "best_measured": best,
                "signed_score_slope_per_q_step": signed_slope,
            }
        )
    return models


def _rank_unmeasured(
    fixed_rows: list[dict[str, Any]],
    advisory_by_key: dict[tuple[str, int, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    out = []
    for row in fixed_rows:
        key = _mutation_key(row)
        if key in advisory_by_key:
            continue
        mutation = row["mutation"]
        out.append(
            {
                "candidate_id": row["mutation_id"],
                "mutation": mutation,
                "q_before": row.get("q_before"),
                "q_after": row.get("q_after"),
                "target_mass": _target_mass(row),
                "axis_target_mass": _axis_target_mass(row),
                "reason": "unmeasured_fixed_length_candidate",
                "priority_tuple": [
                    -_target_mass(row),
                    abs(int(mutation["delta"])),
                    str(mutation["tensor_name"]),
                    int(mutation["q_offset"]),
                    int(mutation["delta"]),
                ],
            }
        )
    out.sort(key=lambda item: tuple(item["priority_tuple"]))
    return out


def _rank_exploit_candidates(
    fixed_rows: list[dict[str, Any]],
    models: list[dict[str, Any]],
    advisory_by_key: dict[tuple[str, int, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    fixed_by_offset_delta: dict[tuple[str, int, int], dict[str, Any]] = {
        _mutation_key(row): row for row in fixed_rows
    }
    rows = []
    for model in models:
        slope = model.get("signed_score_slope_per_q_step")
        if slope is None:
            continue
        # If score increases with +q, try negative deltas; if it decreases, try positive deltas.
        preferred_sign = -1 if float(slope) > 0.0 else 1
        for magnitude in (2, 1):
            key = (model["tensor_name"], int(model["q_offset"]), preferred_sign * magnitude)
            source = fixed_by_offset_delta.get(key)
            if source is None or key in advisory_by_key:
                continue
            expected = abs(float(slope)) * magnitude
            rows.append(
                {
                    "candidate_id": source["mutation_id"],
                    "mutation": source["mutation"],
                    "q_before": source.get("q_before"),
                    "q_after": source.get("q_after"),
                    "target_mass": _target_mass(source),
                    "axis_target_mass": _axis_target_mass(source),
                    "reason": "signed_slope_preferred_direction",
                    "source_model": {
                        "tensor_name": model["tensor_name"],
                        "q_offset": model["q_offset"],
                        "signed_score_slope_per_q_step": slope,
                        "best_measured": model.get("best_measured"),
                    },
                    "expected_score_reduction_proxy": expected,
                }
            )
    rows.sort(
        key=lambda item: (
            -float(item["expected_score_reduction_proxy"]),
            -float(item["target_mass"]),
            str(item["mutation"]["tensor_name"]),
            int(item["mutation"]["q_offset"]),
        )
    )
    return rows


def build_selection(args: argparse.Namespace) -> dict[str, Any]:
    feasibility = _read_json(args.feasibility.resolve())
    advisory = _read_json(args.advisory_summary.resolve()) if args.advisory_summary else None
    fixed_rows = [
        row
        for row in feasibility.get("fixed_length_runtime_compatible_rows", [])
        if isinstance(row, dict) and row.get("fixed_length_runtime_compatible")
    ]
    advisory_by_id = _advisory_rows(advisory)
    advisory_by_key = _attach_advisory(fixed_rows, advisory_by_id)
    models = _outcome_models(fixed_rows, advisory_by_key, args.baseline_score)
    exploit = _rank_exploit_candidates(fixed_rows, models, advisory_by_key)
    unmeasured = _rank_unmeasured(fixed_rows, advisory_by_key)
    queue = []
    seen = set()
    for row in [*exploit, *unmeasured]:
        candidate_id = row["candidate_id"]
        if candidate_id in seen:
            continue
        seen.add(candidate_id)
        queue.append(row)
        if len(queue) >= args.limit:
            break

    return {
        "schema": "decoder_q_next_candidate_selection_v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "producer": "tools/select_decoder_q_next_candidates.py",
        "inputs": {
            "feasibility": str(args.feasibility.resolve()),
            "advisory_summary": str(args.advisory_summary.resolve()) if args.advisory_summary else None,
            "baseline_score": args.baseline_score,
            "limit": int(args.limit),
        },
        "summary": {
            "fixed_length_candidate_count": len(fixed_rows),
            "advisory_candidate_count": len(advisory_by_id),
            "measured_key_count": len(advisory_by_key),
            "outcome_model_count": len(models),
            "signed_slope_model_count": sum(1 for model in models if model.get("signed_score_slope_per_q_step") is not None),
            "queue_count": len(queue),
        },
        "outcome_models": models,
        "queue": queue,
        "authority": {
            "score_claim": False,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
            "notes": "Selection queue only; candidates require archive materialization, official inflate, advisory scoring, and exact eval before promotion.",
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feasibility", type=Path, required=True)
    parser.add_argument("--advisory-summary", type=Path)
    parser.add_argument("--baseline-score", type=float)
    parser.add_argument("--limit", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = build_selection(args)
    _write_json(args.output, payload)
    top = payload["queue"][0] if payload["queue"] else None
    top_brief = None
    if top is not None:
        mutation = top.get("mutation", {})
        top_brief = {
            "candidate_id": top.get("candidate_id"),
            "reason": top.get("selection_reason"),
            "tensor_name": mutation.get("tensor_name"),
            "q_offset": mutation.get("q_offset"),
            "delta": mutation.get("delta"),
            "proxy_score": top.get("proxy_score"),
            "advisory_delta_score": top.get("advisory_delta_score"),
        }
    print(
        json.dumps(
            {
                "output": str(args.output),
                "fixed_length_candidate_count": payload["summary"]["fixed_length_candidate_count"],
                "advisory_candidate_count": payload["summary"]["advisory_candidate_count"],
                "signed_slope_model_count": payload["summary"]["signed_slope_model_count"],
                "queue_count": payload["summary"]["queue_count"],
                "top_candidate": top_brief,
                "score_claim": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
