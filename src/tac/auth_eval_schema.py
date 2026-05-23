# SPDX-License-Identifier: MIT
"""Normalize contest auth-eval JSON schemas without weakening custody.

The canonical evaluator emits ``canonical_score``, component distances, and
archive custody fields. Some older wrappers emitted a nested
``score_components`` object. Dispatch scripts should not silently lose score
signal when one schema is absent, and they must not mark a result claimable
when canonical fields are missing.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

FULL_CONTEST_SAMPLE_COUNT = 600
CONTEST_CUDA_EVIDENCE_TAG = "[contest-CUDA]"
CONTEST_AUTH_AXIS_BY_EVIDENCE_GRADE = {
    "contest-CPU": "contest_cpu",
    "contest-CUDA": "contest_cuda",
}
ORIGINAL_VIDEO_BYTES = 37_545_489
DEFAULT_FORMULA_ABS_TOL = 1e-6


def numeric_or_none(value: Any) -> float | None:
    """Return ``value`` as a finite float, excluding bool/null."""

    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None
    if isinstance(value, str):
        try:
            parsed = float(value)
        except ValueError:
            return None
        return parsed if math.isfinite(parsed) else None
    return None


def int_or_none(value: Any) -> int | None:
    """Return ``value`` as an int when it is an integer-like scalar."""

    parsed = numeric_or_none(value)
    if parsed is None or int(parsed) != parsed:
        return None
    return int(parsed)


def contest_formula_score(
    *,
    seg_dist: float,
    pose_dist: float,
    archive_bytes: int,
) -> float:
    """Return the official contest score from component distances and bytes."""

    return (
        100.0 * seg_dist
        + math.sqrt(10.0 * pose_dist)
        + 25.0 * archive_bytes / ORIGINAL_VIDEO_BYTES
    )


def first_numeric(*values: Any) -> float | None:
    """Return the first numeric value from ``values``."""

    for value in values:
        parsed = numeric_or_none(value)
        if parsed is not None:
            return parsed
    return None


def eval_metric_summary(eval_data: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize canonical and legacy auth-eval result keys.

    ``score`` always prefers the recomputed canonical score over rounded
    display scores. ``rate`` is the score contribution; ``rate_unscaled`` is the
    raw ``archive_bytes / contest_denominator`` term.
    """

    if not eval_data:
        return {
            "score": None,
            "pose_avg": None,
            "seg_avg": None,
            "rate": None,
            "rate_unscaled": None,
            "archive_size_bytes": None,
            "n_samples": None,
            "canonical_score_source": None,
        }
    sc = eval_data.get("score_components") or {}
    if not isinstance(sc, dict):
        sc = {}
    return {
        "score": first_numeric(
            eval_data.get("canonical_score"),
            eval_data.get("score_recomputed_from_components"),
            eval_data.get("score"),
            eval_data.get("total_score"),
            eval_data.get("final_score"),
        ),
        "pose_avg": first_numeric(
            eval_data.get("avg_posenet_dist"),
            eval_data.get("pose_avg"),
            sc.get("pose"),
            sc.get("pose_avg"),
            sc.get("posenet"),
        ),
        "seg_avg": first_numeric(
            eval_data.get("avg_segnet_dist"),
            eval_data.get("seg_avg"),
            sc.get("seg"),
            sc.get("seg_avg"),
            sc.get("segnet"),
        ),
        "rate": first_numeric(
            eval_data.get("score_rate_contribution"),
            eval_data.get("rate"),
            sc.get("rate"),
            sc.get("rate_term"),
        ),
        "rate_unscaled": first_numeric(
            eval_data.get("rate_unscaled"),
            sc.get("rate_unscaled"),
        ),
        "archive_size_bytes": int_or_none(
            eval_data.get("archive_size_bytes")
            if eval_data.get("archive_size_bytes") is not None
            else eval_data.get("archive_bytes")
        ),
        "n_samples": int_or_none(eval_data.get("n_samples")),
        "canonical_score_source": eval_data.get("canonical_score_source"),
    }


def auth_eval_completion_summary(eval_data: dict[str, Any] | None) -> dict[str, Any]:
    """Return compact authenticated fields for completion logs.

    Remote wrappers should quote this summary instead of inventing labels such
    as ``contest_cuda_score`` from the requested device alone. The evaluator's
    own evidence fields decide whether a result is a score claim.
    """

    metrics = eval_metric_summary(eval_data)
    prov = _provenance(eval_data)
    payload = eval_data if isinstance(eval_data, dict) else {}
    return {
        "score": metrics.get("score"),
        "score_source": metrics.get("canonical_score_source"),
        "archive_size_bytes": metrics.get("archive_size_bytes"),
        "n_samples": metrics.get("n_samples"),
        "evidence_grade": payload.get("evidence_grade"),
        "lane_tag": payload.get("lane_tag"),
        "score_axis": payload.get("score_axis"),
        "evidence_semantics": payload.get("evidence_semantics"),
        "score_claim": payload.get("score_claim") is True,
        "score_claim_valid": payload.get("score_claim_valid") is True,
        "promotion_eligible": payload.get("promotion_eligible") is True,
        "rank_or_kill_eligible": payload.get("rank_or_kill_eligible") is True,
        "device": eval_device(eval_data),
        "gpu_model": prov.get("gpu_model"),
        "gpu_t4_match": prov.get("gpu_t4_match"),
    }


def load_auth_eval_json(path: Path) -> dict[str, Any]:
    """Load an auth-eval JSON object from disk, failing closed on bad shape."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"auth-eval JSON must contain an object, got {type(payload).__name__}")
    return payload


def _print_auth_eval_completion_summary(args: argparse.Namespace) -> int:
    summary = auth_eval_completion_summary(load_auth_eval_json(args.path))
    if args.field:
        value = summary.get(args.field)
        if value is None:
            return 2
        if isinstance(value, str):
            print(value)
        else:
            print(json.dumps(value, sort_keys=True, separators=(",", ":")))
        return 0
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the small CLI used by remote wrappers."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    completion = subparsers.add_parser(
        "completion-summary",
        help="Print compact authenticated completion fields from contest_auth_eval.json.",
    )
    completion.add_argument("path", type=Path)
    completion.add_argument(
        "--field",
        choices=tuple(auth_eval_completion_summary({}).keys()),
        help="Print one summary field instead of the full JSON object.",
    )
    completion.set_defaults(func=_print_auth_eval_completion_summary)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    return int(args.func(args))


def required_exact_eval_metric_blockers(
    metrics: dict[str, Any],
    *,
    expected_archive_bytes: int | None = None,
    expected_n_samples: int | None = None,
    formula_abs_tol: float = DEFAULT_FORMULA_ABS_TOL,
) -> list[str]:
    """Return blockers that make an eval JSON non-claimable."""

    blockers: list[str] = []
    for key in ("score", "pose_avg", "seg_avg", "rate_unscaled", "archive_size_bytes"):
        if metrics.get(key) is None:
            blockers.append(f"{key}_missing")
    if metrics.get("canonical_score_source") != "score_recomputed_from_components":
        blockers.append("canonical_score_source_not_recomputed_from_components")
    score = metrics.get("score")
    pose_avg = metrics.get("pose_avg")
    seg_avg = metrics.get("seg_avg")
    archive_size_bytes = metrics.get("archive_size_bytes")
    if (
        score is not None
        and pose_avg is not None
        and seg_avg is not None
        and archive_size_bytes is not None
    ):
        if pose_avg < 0 or seg_avg < 0 or archive_size_bytes < 0:
            blockers.append("contest_formula_inputs_negative")
        else:
            recomputed = contest_formula_score(
                seg_dist=float(seg_avg),
                pose_dist=float(pose_avg),
                archive_bytes=int(archive_size_bytes),
            )
            if abs(float(score) - recomputed) > formula_abs_tol:
                blockers.append(
                    "score_component_formula_mismatch:"
                    f"score={float(score):.12g}:recomputed={recomputed:.12g}"
                )
    rate_unscaled = metrics.get("rate_unscaled")
    if rate_unscaled is not None and archive_size_bytes is not None:
        expected_rate = int(archive_size_bytes) / ORIGINAL_VIDEO_BYTES
        if abs(float(rate_unscaled) - expected_rate) > formula_abs_tol:
            blockers.append(
                "rate_unscaled_archive_bytes_mismatch:"
                f"rate={float(rate_unscaled):.12g}:expected={expected_rate:.12g}"
            )
    if (
        expected_archive_bytes is not None
        and metrics.get("archive_size_bytes") is not None
        and metrics["archive_size_bytes"] != expected_archive_bytes
    ):
        blockers.append(
            "archive_size_bytes_mismatch:"
            f"manifest={metrics['archive_size_bytes']}:actual={expected_archive_bytes}"
        )
    if (
        expected_n_samples is not None
        and metrics.get("n_samples") is None
    ):
        blockers.append("n_samples_missing")
    elif (
        expected_n_samples is not None
        and metrics["n_samples"] != expected_n_samples
    ):
        blockers.append(f"n_samples_mismatch:manifest={metrics['n_samples']}:expected={expected_n_samples}")
    return blockers


def _truthy(value: Any) -> bool:
    return value is True or (
        isinstance(value, str) and value.strip().lower() in {"1", "true", "yes"}
    )


def _provenance(eval_data: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(eval_data, dict):
        return {}
    value = eval_data.get("provenance")
    return value if isinstance(value, dict) else {}


def eval_device(eval_data: dict[str, Any] | None) -> str | None:
    """Return the normalized auth-eval device, preferring provenance custody."""

    if not isinstance(eval_data, dict):
        return None
    prov = _provenance(eval_data)
    device = (
        prov.get("actual_device")
        if prov.get("actual_device") is not None
        else eval_data.get("actual_device")
        if eval_data.get("actual_device") is not None
        else prov.get("device")
        if prov.get("device") is not None
        else eval_data.get("device")
    )
    return str(device).strip().lower() if device is not None else None


def contest_cuda_hardware_is_documented(eval_data: dict[str, Any] | None) -> bool:
    """Return True only for T4 or explicitly documented equivalent CUDA hardware."""

    if not isinstance(eval_data, dict):
        return False
    prov = _provenance(eval_data)
    if _truthy(prov.get("gpu_t4_match")) or _truthy(eval_data.get("gpu_t4_match")):
        return True
    equivalent = _truthy(eval_data.get("contest_equivalent_hardware")) or _truthy(
        prov.get("contest_equivalent_hardware")
    )
    note = (
        eval_data.get("contest_equivalent_hardware_note")
        or eval_data.get("hardware_equivalence_note")
        or eval_data.get("contest_equivalent_hardware_source")
        or eval_data.get("hardware_equivalence_source")
        or prov.get("contest_equivalent_hardware_note")
        or prov.get("hardware_equivalence_note")
        or prov.get("contest_equivalent_hardware_source")
        or prov.get("hardware_equivalence_source")
    )
    return equivalent and isinstance(note, str) and bool(note.strip())


def required_contest_cuda_evidence_blockers(
    eval_data: dict[str, Any] | None,
    metrics: dict[str, Any],
    *,
    expected_archive_bytes: int | None = None,
    expected_n_samples: int = FULL_CONTEST_SAMPLE_COUNT,
) -> list[str]:
    """Return blockers for a claimable ``[contest-CUDA]`` exact-eval result."""

    blockers = required_exact_eval_metric_blockers(
        metrics,
        expected_archive_bytes=expected_archive_bytes,
        expected_n_samples=expected_n_samples,
    )
    if eval_device(eval_data) != "cuda":
        blockers.append("device_not_cuda")
    if not contest_cuda_hardware_is_documented(eval_data):
        blockers.append("contest_cuda_hardware_not_t4_or_documented_equivalent")
    if isinstance(eval_data, dict):
        lane_tag = eval_data.get("lane_tag")
        if lane_tag is not None and lane_tag != CONTEST_CUDA_EVIDENCE_TAG:
            blockers.append("evidence_tag_not_contest_cuda")
        score_axis = eval_data.get("score_axis")
        if score_axis is not None and score_axis != "contest_cuda":
            blockers.append("score_axis_not_contest_cuda")
        semantics = eval_data.get("evidence_semantics")
        if semantics is not None and semantics != "contest_cuda_exact_auth_eval":
            blockers.append("evidence_semantics_not_contest_cuda_exact_auth_eval")
        if "score_claim_valid" in eval_data and eval_data.get("score_claim_valid") is not True:
            blockers.append("score_claim_valid_not_true")
    return blockers


def required_contest_cpu_evidence_blockers(
    eval_data: dict[str, Any] | None,
    metrics: dict[str, Any],
    *,
    expected_archive_bytes: int | None = None,
    expected_n_samples: int = FULL_CONTEST_SAMPLE_COUNT,
) -> list[str]:
    """Return blockers for a claimable ``[contest-CPU]`` exact-eval result."""

    blockers = required_exact_eval_metric_blockers(
        metrics,
        expected_archive_bytes=expected_archive_bytes,
        expected_n_samples=expected_n_samples,
    )
    if eval_device(eval_data) != "cpu":
        blockers.append("device_not_cpu")
    prov = _provenance(eval_data)
    if prov.get("platform_system") != "Linux":
        blockers.append("contest_cpu_platform_system_not_linux")
    machine = str(prov.get("platform_machine") or "").lower()
    if machine not in {"x86_64", "amd64"}:
        blockers.append("contest_cpu_platform_machine_not_x86_64")
    if isinstance(eval_data, dict):
        lane_tag = eval_data.get("lane_tag")
        if lane_tag is not None and lane_tag != "[contest-CPU]":
            blockers.append("evidence_tag_not_contest_cpu")
        score_axis = eval_data.get("score_axis")
        if score_axis is not None and score_axis != "contest_cpu":
            blockers.append("score_axis_not_contest_cpu")
        semantics = eval_data.get("evidence_semantics")
        if semantics is not None and semantics != "public_leaderboard_cpu_reproduction":
            blockers.append("evidence_semantics_not_public_leaderboard_cpu_reproduction")
        if "score_claim_valid" in eval_data and eval_data.get("score_claim_valid") is not True:
            blockers.append("score_claim_valid_not_true")
    return blockers


def required_contest_auth_axis_payload_blockers(
    eval_data: dict[str, Any] | None,
    metrics: dict[str, Any],
    *,
    expected_archive_bytes: int | None = None,
    expected_n_samples: int = FULL_CONTEST_SAMPLE_COUNT,
) -> list[str]:
    """Return blockers when a payload is not a strict contest auth-eval axis.

    This is stricter than the older CUDA-only helper because transfer bridges
    use it as the right-hand authority surface. A diagnostic/advisory payload
    with matching numbers must not become a local-acceleration calibration
    target merely by carrying plausible score fields.
    """

    if not isinstance(eval_data, dict):
        return ["auth_eval_payload_missing_or_not_object"]

    blockers: list[str] = []
    diagnostic_blockers = eval_data.get("diagnostic_blockers")
    if isinstance(diagnostic_blockers, list) and diagnostic_blockers:
        blockers.append("diagnostic_blockers_present")
    elif diagnostic_blockers is not None and diagnostic_blockers != []:
        blockers.append("diagnostic_blockers_malformed_or_present")

    grade = eval_data.get("evidence_grade")
    if grade == "contest-CUDA":
        _extend_unique(
            blockers,
            required_contest_cuda_evidence_blockers(
                eval_data,
                metrics,
                expected_archive_bytes=expected_archive_bytes,
                expected_n_samples=expected_n_samples,
            ),
        )
        _require_exact_field(blockers, eval_data, "lane_tag", CONTEST_CUDA_EVIDENCE_TAG)
        _require_exact_field(blockers, eval_data, "score_axis", "contest_cuda")
        _require_exact_field(
            blockers,
            eval_data,
            "evidence_semantics",
            "contest_cuda_exact_auth_eval",
        )
        if eval_data.get("exact_cuda_eval_complete") is not True:
            blockers.append("exact_cuda_eval_complete_not_true")
    elif grade == "contest-CPU":
        _extend_unique(
            blockers,
            required_contest_cpu_evidence_blockers(
                eval_data,
                metrics,
                expected_archive_bytes=expected_archive_bytes,
                expected_n_samples=expected_n_samples,
            ),
        )
        _require_exact_field(blockers, eval_data, "lane_tag", "[contest-CPU]")
        _require_exact_field(blockers, eval_data, "score_axis", "contest_cpu")
        _require_exact_field(
            blockers,
            eval_data,
            "evidence_semantics",
            "public_leaderboard_cpu_reproduction",
        )
        if eval_data.get("cpu_leaderboard_reproduction_eligible") is not True:
            blockers.append("cpu_leaderboard_reproduction_eligible_not_true")
    else:
        blockers.append("auth_eval_evidence_grade_not_contest_cpu_or_cuda")

    if eval_data.get("score_claim") is not True:
        blockers.append("score_claim_not_true")
    if eval_data.get("score_claim_valid") is not True:
        blockers.append("score_claim_valid_not_true")
    if eval_data.get("promotion_eligible") is not False:
        blockers.append("promotion_eligible_missing_or_not_false")
    if eval_data.get("rank_or_kill_eligible") is not False:
        blockers.append("rank_or_kill_eligible_missing_or_not_false")
    return blockers


def _extend_unique(blockers: list[str], additions: list[str]) -> None:
    for blocker in additions:
        if blocker not in blockers:
            blockers.append(blocker)


def _require_exact_field(
    blockers: list[str],
    payload: dict[str, Any],
    key: str,
    expected: str,
) -> None:
    if payload.get(key) != expected:
        blockers.append(f"{key}_not_{expected}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
