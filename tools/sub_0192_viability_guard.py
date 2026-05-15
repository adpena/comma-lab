#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Classify whether reviewed tactics can plausibly cross a sub-score threshold.

This is a read-only research guard. It reads result-review JSON or profile
ledger JSON/JSONL artifacts, computes the rate-term byte-equivalent gap to a
score threshold, and reports whether the observed tactic is frontier-eligible.
It does not claim scores, dispatch jobs, or mutate live state.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTEST_ARCHIVE_DENOMINATOR_BYTES = 37_545_489
RATE_SCORE_WEIGHT = 25.0
DEFAULT_THRESHOLD = 0.192
SCHEMA = "sub_0192_viability_guard_v1"


JsonObject = dict[str, Any]


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float) and math.isfinite(float(value)):
        return float(value)
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            return None
        if math.isfinite(parsed):
            return parsed
    return None


def _as_int(value: Any) -> int | None:
    number = _as_float(value)
    if number is None:
        return None
    if number < 0:
        return None
    return math.ceil(number)


def _get_path(data: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = data
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _first_number(
    data: Mapping[str, Any],
    paths: Iterable[tuple[str, ...]],
) -> tuple[float, str] | tuple[None, None]:
    for path in paths:
        number = _as_float(_get_path(data, path))
        if number is not None:
            return number, ".".join(path)
    return None, None


def _first_int(
    data: Mapping[str, Any],
    paths: Iterable[tuple[str, ...]],
) -> tuple[int, str] | tuple[None, None]:
    for path in paths:
        number = _as_int(_get_path(data, path))
        if number is not None:
            return number, ".".join(path)
    return None, None


def _record_metadata_matches_axis(data: Mapping[str, Any], axis: str) -> bool:
    expected = "cuda" if axis == "contest_cuda" else "cpu"
    fields: list[str] = []
    for path in (
        ("score_axis",),
        ("evidence_grade",),
        ("lane_tag",),
        ("device_axis_label",),
        ("auth_eval", "record", "score_axis"),
        ("auth_eval", "record", "evidence_grade"),
        ("exact_results", "score_axis"),
        ("score_recomputation", "score_axis"),
    ):
        value = _get_path(data, path)
        if isinstance(value, str):
            fields.append(value.lower())
    return any(expected in value for value in fields)


def _score_paths(axis: str, data: Mapping[str, Any]) -> list[tuple[str, ...]]:
    score_axis = str(data.get("score_axis", "")).lower()
    paths: list[tuple[str, ...]] = []
    if axis == "contest_cpu":
        paths.extend(
            [
                ("exact_results", "contest_cpu_score"),
                ("anchor", "cpu_score"),
                ("deltas", "paired_cpu_score_contest_cpu"),
            ]
        )
    elif axis == "contest_cuda":
        paths.extend(
            [
                ("exact_results", "contest_cuda_score"),
                ("anchor", "cuda_score"),
            ]
        )
    elif "cpu" in score_axis:
        paths.extend(_score_paths("contest_cpu", data))
    else:
        paths.extend(_score_paths("contest_cuda", data))

    if axis in {"contest_cpu", "contest_cuda"} and _record_metadata_matches_axis(data, axis):
        paths.extend(
            [
                ("score_recomputation", "recomputed_score"),
                ("components", "score_recomputed_from_components"),
                ("canonical_score",),
                ("score_recomputed_from_components",),
                ("recomputed_score",),
                ("reported_score",),
                ("score",),
                ("final_score",),
                ("display_final_score",),
                ("baseline_score",),
            ]
        )
    seen: set[tuple[str, ...]] = set()
    unique: list[tuple[str, ...]] = []
    for path in paths:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def _axis_for_record(axis: str, data: Mapping[str, Any]) -> str:
    if axis != "auto":
        return axis
    score_axis = str(data.get("score_axis", "")).lower()
    if "cpu" in score_axis:
        return "contest_cpu"
    if "cuda" in score_axis:
        return "contest_cuda"
    if _as_float(_get_path(data, ("exact_results", "contest_cuda_score"))) is not None:
        return "contest_cuda"
    if _as_float(_get_path(data, ("anchor", "cuda_score"))) is not None:
        return "contest_cuda"
    if _as_float(_get_path(data, ("exact_results", "contest_cpu_score"))) is not None:
        return "contest_cpu"
    if _as_float(_get_path(data, ("anchor", "cpu_score"))) is not None:
        return "contest_cpu"
    return "unknown"


def _walk_numbers(
    data: Any,
    prefix: tuple[str, ...] = (),
) -> Iterable[tuple[tuple[str, ...], float]]:
    if isinstance(data, Mapping):
        for key, value in data.items():
            yield from _walk_numbers(value, (*prefix, str(key)))
    elif isinstance(data, list):
        for idx, value in enumerate(data):
            yield from _walk_numbers(value, (*prefix, str(idx)))
    else:
        number = _as_float(data)
        if number is not None:
            yield prefix, number


def _best_measured_delta(data: Mapping[str, Any]) -> tuple[float, str] | tuple[None, None]:
    candidates: list[tuple[float, str]] = []
    key_names = {
        "delta_score_vs_baseline",
        "score_delta",
        "measured_score_delta",
        "delta_score",
    }
    for path, number in _walk_numbers(data):
        if path and path[-1] in key_names:
            candidates.append((number, ".".join(path)))

    baseline = _as_float(data.get("baseline_score"))
    canonical = _as_float(data.get("canonical_score"))
    if baseline is not None and canonical is not None:
        candidates.append((canonical - baseline, "canonical_score-minus-baseline_score"))

    if not candidates:
        return None, None
    return min(candidates, key=lambda item: item[0])


def _explicit_remaining_byte_mass(data: Mapping[str, Any]) -> tuple[int, str] | tuple[None, None]:
    return _first_int(
        data,
        [
            ("conclusion", "same_frame_realistic_saving_upper_bound_bytes"),
            ("conclusion", "remaining_byte_mass_bytes"),
            ("remaining_byte_mass_bytes",),
            ("max_remaining_saving_bytes",),
            ("byte_only_findings", "remaining_byte_mass_bytes"),
        ],
    )


def _summed_remaining_byte_mass(
    data: Mapping[str, Any],
) -> tuple[int, list[str]] | tuple[None, list[str]]:
    byte_keys = {
        "realistic_saving_bytes",
        "hardcode_saving_bytes",
        "remaining_saving_bytes",
        "available_byte_mass_bytes",
    }
    total = 0
    paths: list[str] = []
    for path, number in _walk_numbers(data):
        if not path or path[-1] not in byte_keys:
            continue
        if number <= 0:
            continue
        total += math.ceil(number)
        paths.append(".".join(path))
    if not paths:
        return None, []
    return total, paths


def _remaining_byte_mass(data: Mapping[str, Any]) -> tuple[int, str] | tuple[None, None]:
    explicit, source = _explicit_remaining_byte_mass(data)
    if explicit is not None and source is not None:
        return explicit, source
    summed, sources = _summed_remaining_byte_mass(data)
    if summed is None:
        return None, None
    return summed, "sum(" + ",".join(sources) + ")"


def _archive_bytes(data: Mapping[str, Any]) -> tuple[int, str] | tuple[None, None]:
    return _first_int(
        data,
        [
            ("archive", "bytes"),
            ("custody", "archive_bytes"),
            ("score_recomputation", "archive_bytes"),
            ("archive_anatomy", "archive_bytes"),
            ("anchor", "archive_bytes"),
            ("archive_bytes",),
            ("archive_size_bytes",),
        ],
    )


def _tactic_id(source: Path, data: Mapping[str, Any], index: int | None) -> str:
    for key in ("lane_id", "technique", "candidate_id", "schema"):
        value = data.get(key)
        if isinstance(value, str) and value:
            return value
    suffix = f"#{index}" if index is not None else ""
    return f"{source.stem}{suffix}"


def score_gap_to_bytes(score_gap: float) -> float:
    return score_gap * CONTEST_ARCHIVE_DENOMINATOR_BYTES / RATE_SCORE_WEIGHT


def bytes_to_score_delta(byte_count: int | float) -> float:
    return RATE_SCORE_WEIGHT * float(byte_count) / CONTEST_ARCHIVE_DENOMINATOR_BYTES


def classify_record(
    data: Mapping[str, Any],
    *,
    source_path: Path,
    threshold: float = DEFAULT_THRESHOLD,
    axis: str = "auto",
    index: int | None = None,
) -> JsonObject:
    effective_axis = _axis_for_record(axis, data)
    current_score, score_source = _first_number(data, _score_paths(effective_axis, data))
    archive_bytes, archive_source = _archive_bytes(data)
    remaining_bytes, remaining_source = _remaining_byte_mass(data)
    measured_delta, measured_delta_source = _best_measured_delta(data)

    tactic_id = _tactic_id(source_path, data, index)
    input_score_claim = data.get("score_claim")
    reasons: list[str] = []
    blockers: list[str] = []
    frontier_eligible = False
    classification = "insufficient_evidence"

    score_gap = None
    byte_equivalent_gap = None
    bytes_to_threshold = None
    remaining_score_equivalent = (
        bytes_to_score_delta(remaining_bytes) if remaining_bytes is not None else None
    )
    measured_improvement = max(0.0, -measured_delta) if measured_delta is not None else None
    measured_delta_byte_equivalent = (
        score_gap_to_bytes(measured_improvement) if measured_improvement is not None else None
    )
    if current_score is None:
        blockers.append("missing_current_score")
    else:
        score_gap = max(0.0, current_score - threshold)
        byte_equivalent_gap = score_gap_to_bytes(score_gap)
        bytes_to_threshold = math.ceil(byte_equivalent_gap)
        if score_gap <= 0:
            frontier_eligible = True
            classification = "already_below_or_equal_threshold"
            reasons.append("current_score_at_or_below_threshold")
        else:
            if remaining_bytes is not None:
                if remaining_bytes < bytes_to_threshold:
                    blockers.append("insufficient_remaining_byte_mass")
                    reasons.append(
                        "remaining_byte_mass "
                        f"{remaining_bytes} < bytes_to_threshold {bytes_to_threshold}"
                    )
                else:
                    reasons.append(
                        "remaining_byte_mass "
                        f"{remaining_bytes} >= bytes_to_threshold {bytes_to_threshold}"
                    )
                    frontier_eligible = True
                    classification = "frontier_plausible_by_remaining_byte_mass"
            else:
                reasons.append("remaining_byte_mass_missing")

            if measured_delta is not None:
                if measured_improvement <= 0:
                    blockers.append("measured_delta_not_improving")
                    reasons.append(f"best_measured_delta {measured_delta} is not an improvement")
                    frontier_eligible = False
                elif measured_improvement < score_gap:
                    blockers.append("insufficient_measured_score_delta")
                    reasons.append(
                        "measured_improvement "
                        f"{measured_improvement:.12g} < score_gap {score_gap:.12g}"
                    )
                    frontier_eligible = False
                else:
                    reasons.append(
                        "measured_improvement "
                        f"{measured_improvement:.12g} >= score_gap {score_gap:.12g}"
                    )
                    if not blockers:
                        frontier_eligible = True
                        classification = "frontier_plausible_by_measured_delta"
            else:
                reasons.append("measured_score_delta_missing")

            if blockers:
                classification = "not_frontier_eligible"
            elif not frontier_eligible:
                blockers.append("missing_remaining_mass_or_measured_delta_evidence")
                classification = "not_frontier_eligible"
    return {
        "tactic_id": tactic_id,
        "source_path": str(source_path),
        "source_index": index,
        "axis": effective_axis,
        "classification": classification,
        "frontier_eligible": frontier_eligible,
        "threshold": threshold,
        "current_score": current_score,
        "current_score_source": score_source,
        "score_gap_to_threshold": score_gap,
        "byte_equivalent_gap": byte_equivalent_gap,
        "bytes_to_threshold_if_components_unchanged": bytes_to_threshold,
        "archive_bytes": archive_bytes,
        "archive_bytes_source": archive_source,
        "remaining_byte_mass_bytes": remaining_bytes,
        "remaining_byte_mass_source": remaining_source,
        "remaining_byte_mass_score_equivalent": remaining_score_equivalent,
        "best_measured_delta": measured_delta,
        "best_measured_delta_source": measured_delta_source,
        "best_measured_improvement": measured_improvement,
        "best_measured_delta_byte_equivalent": measured_delta_byte_equivalent,
        "blockers": blockers,
        "reasons": reasons,
        "input_score_claim_field": input_score_claim,
        "score_claim": False,
        "dispatch_attempted": False,
        "promotion_eligible": False,
    }


def load_records(path: Path) -> list[tuple[JsonObject, int | None]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        records: list[tuple[JsonObject, int | None]] = []
        for index, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            parsed = json.loads(line)
            if not isinstance(parsed, dict):
                raise ValueError(f"{path}:{index} is not a JSON object")
            records.append((parsed, index))
        return records
    parsed = json.loads(text)
    if isinstance(parsed, dict):
        return [(parsed, None)]
    if isinstance(parsed, list):
        records = []
        for index, item in enumerate(parsed):
            if not isinstance(item, dict):
                raise ValueError(f"{path}[{index}] is not a JSON object")
            records.append((item, index))
        return records
    raise ValueError(f"{path} does not contain a JSON object/list")


def discover_default_inputs(repo_root: Path) -> list[Path]:
    research = repo_root / ".omx" / "research"
    if not research.is_dir():
        return []
    patterns = (
        "*result_review*.json",
        "*profile*.json",
        "*profile*.jsonl",
        "*ledger*.json",
        "*ledger*.jsonl",
    )
    paths: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(research.glob(pattern)):
            if path not in seen:
                seen.add(path)
                paths.append(path)
    return paths


def build_report(
    inputs: Iterable[Path],
    *,
    threshold: float = DEFAULT_THRESHOLD,
    axis: str = "auto",
) -> JsonObject:
    input_paths = list(inputs)
    reviews: list[JsonObject] = []
    errors: list[JsonObject] = []
    for raw_path in input_paths:
        path = raw_path if raw_path.is_absolute() else REPO_ROOT / raw_path
        try:
            records = load_records(path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append({"source_path": str(raw_path), "error": str(exc)})
            continue
        for data, index in records:
            reviews.append(
                classify_record(
                    data,
                    source_path=raw_path,
                    threshold=threshold,
                    axis=axis,
                    index=index,
                )
            )

    not_eligible = [row for row in reviews if not row["frontier_eligible"]]
    eligible = [row for row in reviews if row["frontier_eligible"]]
    return {
        "schema": SCHEMA,
        "generated_at_utc": datetime.now(UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "threshold": threshold,
        "score_formula_rate_term": "25 * archive_bytes / 37545489",
        "score_claim": False,
        "dispatch_attempted": False,
        "research_only": True,
        "input_count": len(input_paths),
        "review_count": len(reviews),
        "frontier_eligible_count": len(eligible),
        "not_frontier_eligible_count": len(not_eligible),
        "parse_error_count": len(errors),
        "reviews": reviews,
        "errors": errors,
    }


def render_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Sub-0.192 Viability Guard",
        "",
        f"- schema: `{report['schema']}`",
        f"- generated_at_utc: `{report['generated_at_utc']}`",
        f"- threshold: `{report['threshold']}`",
        "- score_claim: `false`",
        "- dispatch_attempted: `false`",
        "- research_only: `true`",
        f"- review_count: `{report['review_count']}`",
        f"- frontier_eligible_count: `{report['frontier_eligible_count']}`",
        f"- not_frontier_eligible_count: `{report['not_frontier_eligible_count']}`",
        "",
        "## Reviews",
        "",
    ]
    for row in report["reviews"]:
        verdict = "frontier-eligible" if row["frontier_eligible"] else "not-frontier-eligible"
        lines.extend(
            [
                f"### {row['tactic_id']}",
                "",
                f"- verdict: `{verdict}`",
                f"- classification: `{row['classification']}`",
                f"- source: `{row['source_path']}`",
                f"- axis: `{row['axis']}`",
                f"- current_score: `{row['current_score']}`",
                f"- score_gap_to_threshold: `{row['score_gap_to_threshold']}`",
                f"- byte_equivalent_gap: `{row['byte_equivalent_gap']}`",
                (
                    "- bytes_to_threshold_if_components_unchanged: "
                    f"`{row['bytes_to_threshold_if_components_unchanged']}`"
                ),
                f"- remaining_byte_mass_bytes: `{row['remaining_byte_mass_bytes']}`",
                f"- best_measured_delta: `{row['best_measured_delta']}`",
                "- blockers: "
                + (
                    ", ".join(f"`{blocker}`" for blocker in row["blockers"])
                    if row["blockers"]
                    else "`none`"
                ),
                "",
            ]
        )
    if report["errors"]:
        lines.extend(["## Parse Errors", ""])
        for error in report["errors"]:
            lines.append(f"- `{error['source_path']}`: {error['error']}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        default=[],
        help="Result-review/profile JSON or JSONL artifact. May be repeated.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Score threshold. Default: 0.192.",
    )
    parser.add_argument(
        "--axis",
        choices=("auto", "contest_cuda", "contest_cpu"),
        default="auto",
        help="Score axis to evaluate when artifacts carry multiple exact-eval axes.",
    )
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    parser.add_argument("--output", type=Path, help="Write report to this path.")
    parser.add_argument(
        "--fail-if-not-frontier-eligible",
        action="store_true",
        help="Exit 2 when any parsed artifact is classified not-frontier-eligible.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    inputs = args.input or discover_default_inputs(REPO_ROOT)
    if not inputs:
        raise SystemExit("no input artifacts found; pass --input")
    report = build_report(inputs, threshold=args.threshold, axis=args.axis)
    text = (
        json.dumps(report, indent=2, sort_keys=True) + "\n"
        if args.format == "json"
        else render_markdown(report)
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    if args.fail_if_not_frontier_eligible and report["not_frontier_eligible_count"]:
        return 2
    if report["parse_error_count"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
