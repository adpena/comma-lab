#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Test 600-pair per-pair vectors for serial-independence assumptions.

This is a local diagnostic only.  It consumes existing JSON artifacts and emits
statistical evidence about whether per-pair loss/sensitivity vectors look
independent enough for downstream planning assumptions.  It never runs the
scorer, dispatches work, or creates a score claim.
"""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    import importlib.util

    _bootstrap_path = Path(__file__).resolve().parent / "tool_bootstrap.py"
    _spec = importlib.util.spec_from_file_location("tool_bootstrap", _bootstrap_path)
    if _spec is None or _spec.loader is None:
        raise
    _tool_bootstrap = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_tool_bootstrap)
    ensure_repo_imports = _tool_bootstrap.ensure_repo_imports
    repo_root_from_tool = _tool_bootstrap.repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.repo_io import read_json, repo_relative, sha256_file, write_json  # noqa: E402

EVIDENCE_GRADE = "local_independence_diagnostic_only"
EVIDENCE_SEMANTICS = (
    "statistical dependence check over existing advisory per-pair artifacts; "
    "not score evidence"
)
DEFAULT_MIN_LENGTH = 600
DEFAULT_MAX_LAG = 24
INDEPENDENCE_AUTOCORR_MAX = 0.15
BLOCK_AUTOCORR_MIN = 0.35
INDEPENDENCE_CROSS_CORR_MAX = 0.15
BLOCK_CROSS_CORR_MIN = 0.30
INDEPENDENCE_EFFECTIVE_N_MIN = 300.0
BLOCK_EFFECTIVE_N_MAX = 150.0
INDEPENDENCE_TOP50_SHARE_MAX = 0.25
BLOCK_TOP50_SHARE_MIN = 0.40
INDEX_LIKE_SERIES_NAMES = frozenset(
    {
        "idx",
        "index",
        "pair_idx",
        "pair_index",
        "frame_idx",
        "frame_index",
    }
)


@dataclass(frozen=True)
class SeriesCandidate:
    """One extracted numeric 600-pair sequence."""

    source_path: str
    series_path: str
    values: tuple[float, ...]


def _utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(float(value))


def _numeric_vector(value: Any, *, min_length: int) -> tuple[float, ...] | None:
    if not isinstance(value, list) or len(value) < min_length:
        return None
    if not all(_is_finite_number(item) for item in value):
        return None
    return tuple(float(item) for item in value)


def _series_path_is_index_like(series_path: str) -> bool:
    leaf = series_path.rsplit(".", maxsplit=1)[-1].replace("[*]", "").strip().lower()
    return leaf in INDEX_LIKE_SERIES_NAMES


def _walk_numeric_vectors(
    value: Any,
    *,
    prefix: str,
    min_length: int,
    max_depth: int,
) -> list[tuple[str, tuple[float, ...]]]:
    """Find numeric vectors without recursing through huge scalar payloads."""

    direct = _numeric_vector(value, min_length=min_length)
    if direct is not None:
        return [] if _series_path_is_index_like(prefix) else [(prefix, direct)]
    if max_depth <= 0:
        return []

    vectors: list[tuple[str, tuple[float, ...]]] = []
    if isinstance(value, dict):
        for key, child in sorted(value.items()):
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            vectors.extend(
                _walk_numeric_vectors(
                    child,
                    prefix=child_prefix,
                    min_length=min_length,
                    max_depth=max_depth - 1,
                )
            )
    elif isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
        rows = value
        candidate_keys = sorted(set().union(*(row.keys() for row in rows)))
        for key in candidate_keys:
            column = [row.get(key) for row in rows]
            numeric = _numeric_vector(column, min_length=min_length)
            child_path = f"{prefix}[*].{key}"
            if numeric is not None and not _series_path_is_index_like(child_path):
                vectors.append((child_path, numeric))
    return vectors


def _extract_path(payload: Any, series_path: str, *, min_length: int) -> tuple[float, ...]:
    """Extract a simple dotted path, with ``[]`` for list-of-dict columns."""

    current = payload
    for part in [piece for piece in series_path.split(".") if piece]:
        if part.endswith("[]"):
            key = part[:-2]
            if not isinstance(current, dict) or key not in current:
                raise KeyError(f"path segment {part!r} not found in {series_path!r}")
            rows = current[key]
            if not isinstance(rows, list):
                raise TypeError(f"path segment {part!r} is not a list in {series_path!r}")
            current = rows
            continue
        if isinstance(current, list):
            if not all(isinstance(row, dict) and part in row for row in current):
                raise KeyError(f"column {part!r} not found for every row in {series_path!r}")
            current = [row[part] for row in current]
            continue
        if not isinstance(current, dict) or part not in current:
            raise KeyError(f"path segment {part!r} not found in {series_path!r}")
        current = current[part]

    vector = _numeric_vector(current, min_length=min_length)
    if vector is None:
        raise TypeError(f"path {series_path!r} did not resolve to {min_length}+ finite numbers")
    return vector


def load_series_candidates(
    input_jsons: list[Path],
    *,
    explicit_paths: list[str],
    min_length: int,
    max_depth: int = 4,
) -> list[SeriesCandidate]:
    """Load explicit or auto-detected numeric vectors from JSON artifacts."""

    candidates: list[SeriesCandidate] = []
    for path in input_jsons:
        payload = read_json(path)
        rel = repo_relative(path, REPO_ROOT)
        if explicit_paths:
            for series_path in explicit_paths:
                values = _extract_path(payload, series_path, min_length=min_length)
                candidates.append(SeriesCandidate(rel, series_path, values))
        else:
            for series_path, values in _walk_numeric_vectors(
                payload,
                prefix="",
                min_length=min_length,
                max_depth=max_depth,
            ):
                candidates.append(SeriesCandidate(rel, series_path, values))
    return candidates


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _pearson(x_values: list[float], y_values: list[float]) -> float:
    if len(x_values) != len(y_values) or len(x_values) < 2:
        return 0.0
    x_mean = _mean(x_values)
    y_mean = _mean(y_values)
    num = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values, strict=True))
    x_den = sum((x - x_mean) ** 2 for x in x_values)
    y_den = sum((y - y_mean) ** 2 for y in y_values)
    if x_den <= 0.0 or y_den <= 0.0:
        return 0.0
    return num / math.sqrt(x_den * y_den)


def _ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        avg_rank = (cursor + 1 + end) / 2.0
        for original_index, _ in indexed[cursor:end]:
            ranks[original_index] = avg_rank
        cursor = end
    return ranks


def _normal_two_sided_p(z_value: float) -> float:
    return math.erfc(abs(z_value) / math.sqrt(2.0))


def _absolute_concentration(values: list[float]) -> dict[str, float]:
    magnitudes = sorted((abs(value) for value in values), reverse=True)
    total = sum(magnitudes)
    if total <= 0.0:
        return {
            "top_1_share": 0.0,
            "top_10_share": 0.0,
            "top_50_share": 0.0,
            "participation_ratio": 0.0,
        }
    square_total = sum(value * value for value in magnitudes)
    participation_ratio = (total * total / square_total) if square_total > 0.0 else 0.0
    return {
        "top_1_share": sum(magnitudes[:1]) / total,
        "top_10_share": sum(magnitudes[:10]) / total,
        "top_50_share": sum(magnitudes[:50]) / total,
        "participation_ratio": participation_ratio,
    }


def _chi_square_survival_wilson_hilferty(statistic: float, dof: int) -> float:
    """Dependency-free chi-square survival approximation."""

    if dof <= 0:
        return 1.0
    if statistic <= 0.0:
        return 1.0
    z_value = ((statistic / dof) ** (1.0 / 3.0) - (1.0 - 2.0 / (9.0 * dof))) / math.sqrt(
        2.0 / (9.0 * dof)
    )
    return 0.5 * math.erfc(z_value / math.sqrt(2.0))


def _autocorrelations(values: list[float], max_lag: int) -> list[float]:
    n = len(values)
    mean_value = _mean(values)
    denom = sum((value - mean_value) ** 2 for value in values)
    if denom <= 0.0:
        return [0.0 for _ in range(max_lag)]
    acf: list[float] = []
    for lag in range(1, max_lag + 1):
        if lag >= n:
            acf.append(0.0)
            continue
        num = sum((values[i] - mean_value) * (values[i - lag] - mean_value) for i in range(lag, n))
        acf.append(num / denom)
    return acf


def _ljung_box(values: list[float], acf: list[float]) -> tuple[float, float]:
    n = len(values)
    statistic = 0.0
    for lag, rho in enumerate(acf, start=1):
        statistic += (rho * rho) / max(n - lag, 1)
    statistic *= n * (n + 2)
    return statistic, _chi_square_survival_wilson_hilferty(statistic, len(acf))


def _runs_test(values: list[float]) -> dict[str, Any]:
    median = statistics.median(values)
    signs = [value > median for value in values if value != median]
    n1 = sum(signs)
    n2 = len(signs) - n1
    if n1 == 0 or n2 == 0:
        return {
            "median": median,
            "runs": 1 if signs else 0,
            "z": None,
            "p_two_sided": None,
            "reason": "constant_or_median_tied_sequence",
        }
    runs = 1 + sum(signs[idx] != signs[idx - 1] for idx in range(1, len(signs)))
    expected = 1.0 + (2.0 * n1 * n2) / (n1 + n2)
    variance = (
        2.0
        * n1
        * n2
        * (2.0 * n1 * n2 - n1 - n2)
        / (((n1 + n2) ** 2) * (n1 + n2 - 1))
    )
    if variance <= 0.0:
        return {
            "median": median,
            "runs": runs,
            "z": None,
            "p_two_sided": None,
            "reason": "zero_runs_variance",
        }
    z_value = (runs - expected) / math.sqrt(variance)
    return {
        "median": median,
        "runs": runs,
        "z": z_value,
        "p_two_sided": _normal_two_sided_p(z_value),
        "n_above_median": n1,
        "n_below_median": n2,
    }


def _effective_sample_size(n: int, acf: list[float]) -> float:
    """Estimate ESS from positive serial correlation until the first nonpositive lag."""

    positive_sum = 0.0
    for rho in acf:
        if rho <= 0.0:
            break
        positive_sum += rho
    return n / max(1.0 + 2.0 * positive_sum, 1e-12)


def _series_verdict(
    *,
    max_abs_autocorr: float,
    serial_effective_n: float,
    top_50_share: float,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if max_abs_autocorr > BLOCK_AUTOCORR_MIN:
        reasons.append("max_lag_autocorrelation_above_0p35")
    if serial_effective_n < BLOCK_EFFECTIVE_N_MAX:
        reasons.append("serial_effective_n_below_150")
    if top_50_share > BLOCK_TOP50_SHARE_MIN:
        reasons.append("top50_absolute_share_above_0p40")
    if reasons:
        return "independence_assumption_blocked", reasons

    warnings: list[str] = []
    if max_abs_autocorr > INDEPENDENCE_AUTOCORR_MAX:
        warnings.append("max_lag_autocorrelation_above_0p15")
    if serial_effective_n < INDEPENDENCE_EFFECTIVE_N_MIN:
        warnings.append("serial_effective_n_below_300")
    if top_50_share > INDEPENDENCE_TOP50_SHARE_MAX:
        warnings.append("top50_absolute_share_above_0p25")
    if warnings:
        return "block_bootstrap_required", warnings
    return "independence_ok_for_factorized_assumption", ["effect_size_gates_passed"]


def analyze_series(candidate: SeriesCandidate, *, max_lag: int) -> dict[str, Any]:
    values = list(candidate.values)
    n = len(values)
    acf = _autocorrelations(values, max_lag)
    ljung_stat, ljung_p = _ljung_box(values, acf)
    run_stats = _runs_test(values)
    n_eff = _effective_sample_size(n, acf)
    concentration = _absolute_concentration(values)
    adjacent_spearman = _pearson(_ranks(values[1:]), _ranks(values[:-1])) if n > 2 else 0.0
    max_abs_autocorr = max((abs(value) for value in acf), default=0.0)
    verdict, reasons = _series_verdict(
        max_abs_autocorr=max_abs_autocorr,
        serial_effective_n=n_eff,
        top_50_share=concentration["top_50_share"],
    )
    return {
        "source_path": candidate.source_path,
        "series_path": candidate.series_path,
        "n": n,
        "mean": _mean(values),
        "std": statistics.pstdev(values) if n > 1 else 0.0,
        "min": min(values),
        "max": max(values),
        "lag_autocorrelation": {str(idx): value for idx, value in enumerate(acf, start=1)},
        "max_abs_autocorrelation": max_abs_autocorr,
        "adjacent_pearson": acf[0] if acf else 0.0,
        "adjacent_spearman": adjacent_spearman,
        "ljung_box": {
            "lags": len(acf),
            "statistic": ljung_stat,
            "p_value_wilson_hilferty_approx": ljung_p,
        },
        "runs_test_median_split": run_stats,
        "serial_effective_sample_size": n_eff,
        "serial_effective_sample_size_fraction": n_eff / n,
        "absolute_concentration": concentration,
        "verdict": verdict,
        "verdict_reasons": reasons,
    }


def _aggregate_verdict(series_reports: list[dict[str, Any]]) -> str:
    if not series_reports:
        return "INCONCLUSIVE_MISSING_600_PAIR_SERIES"
    verdicts = {str(report["verdict"]) for report in series_reports}
    if "independence_assumption_blocked" in verdicts:
        return "independence_assumption_blocked"
    if "block_bootstrap_required" in verdicts:
        return "block_bootstrap_required"
    return "independence_ok_for_factorized_assumption"


def _cross_series_dependence(candidates: list[SeriesCandidate]) -> dict[str, Any]:
    pairs: list[dict[str, Any]] = []
    for left_idx, left in enumerate(candidates):
        for right in candidates[left_idx + 1 :]:
            if len(left.values) != len(right.values):
                continue
            x_values = list(left.values)
            y_values = list(right.values)
            pearson = _pearson(x_values, y_values)
            spearman = _pearson(_ranks(x_values), _ranks(y_values))
            max_abs = max(abs(pearson), abs(spearman))
            if max_abs <= INDEPENDENCE_CROSS_CORR_MAX:
                verdict = "cross_vector_independence_plausible"
                reasons = ["cross_correlation_effect_size_gate_passed"]
            elif max_abs > BLOCK_CROSS_CORR_MIN:
                verdict = "cross_vector_dependence_blocked"
                reasons = ["absolute_cross_correlation_above_0p30"]
            else:
                verdict = "cross_vector_block_bootstrap_required"
                reasons = ["absolute_cross_correlation_above_0p15"]
            pairs.append(
                {
                    "left": f"{left.source_path}:{left.series_path}",
                    "right": f"{right.source_path}:{right.series_path}",
                    "pearson": pearson,
                    "spearman": spearman,
                    "max_abs_correlation": max_abs,
                    "verdict": verdict,
                    "verdict_reasons": reasons,
                }
            )
    if not pairs:
        return {
            "pair_count": 0,
            "max_abs_correlation": 0.0,
            "verdict": "cross_vector_not_tested",
            "top_pairs": [],
        }
    top_pairs = sorted(pairs, key=lambda row: row["max_abs_correlation"], reverse=True)[:20]
    pair_verdicts = {row["verdict"] for row in pairs}
    if "cross_vector_dependence_blocked" in pair_verdicts:
        verdict = "cross_vector_dependence_blocked"
    elif "cross_vector_block_bootstrap_required" in pair_verdicts:
        verdict = "cross_vector_block_bootstrap_required"
    else:
        verdict = "cross_vector_independence_plausible"
    return {
        "pair_count": len(pairs),
        "max_abs_correlation": top_pairs[0]["max_abs_correlation"],
        "verdict": verdict,
        "top_pairs": top_pairs,
    }


def build_report(
    *,
    input_jsons: list[Path],
    explicit_paths: list[str],
    min_length: int,
    max_lag: int,
    max_depth: int,
) -> dict[str, Any]:
    candidates = load_series_candidates(
        input_jsons,
        explicit_paths=explicit_paths,
        min_length=min_length,
        max_depth=max_depth,
    )
    series_reports = [analyze_series(candidate, max_lag=max_lag) for candidate in candidates]
    cross_series = _cross_series_dependence(candidates)
    aggregate = _aggregate_verdict(series_reports)
    return {
        "schema": "pair_independence_diagnostic_v1",
        "generated_at_utc": _utc_now(),
        "tool": "tools/test_600_pair_independence.py",
        "evidence_grade": EVIDENCE_GRADE,
        "evidence_semantics": EVIDENCE_SEMANTICS,
        "axis_tag": "[independence-diagnostic]",
        "contest_axis": None,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatchable": False,
        "auth_eval_skipped": True,
        "result_json": None,
        "scorer_invoked": False,
        "provider_invoked": False,
        "dispatch_attempted": False,
        "predicted_delta_adjustment": 0.0,
        "blockers": [
            "not_contest_cuda",
            "not_contest_cpu",
            "existing_artifact_reanalysis_only",
            "no_archive_runtime_eval_performed",
            "requires_exact_cuda_auth_eval_before_score_use",
        ],
        "allowed_uses": [
            "assumption_quality_assurance",
            "solver_covariance_or_blocking_choice",
            "diagnostic_ranker_hygiene",
        ],
        "denied_uses": [
            "score_claim",
            "promotion",
            "rank_or_kill",
            "dispatch_readiness",
            "paper_empirical_claim",
        ],
        "consumer_contract": {
            "may_update_solver_correlation_priors": True,
            "may_gate_assumptions_about_pairwise_iid": True,
            "must_not_promote_score_or_candidate": True,
            "must_not_rank_or_kill_lanes_from_this_alone": True,
        },
        "parameters": {
            "min_length": min_length,
            "max_lag": max_lag,
            "max_depth": max_depth,
            "independence_autocorr_max": INDEPENDENCE_AUTOCORR_MAX,
            "block_autocorr_min": BLOCK_AUTOCORR_MIN,
            "independence_cross_corr_max": INDEPENDENCE_CROSS_CORR_MAX,
            "block_cross_corr_min": BLOCK_CROSS_CORR_MIN,
            "independence_effective_n_min": INDEPENDENCE_EFFECTIVE_N_MIN,
            "block_effective_n_max": BLOCK_EFFECTIVE_N_MAX,
            "independence_top50_share_max": INDEPENDENCE_TOP50_SHARE_MAX,
            "block_top50_share_min": BLOCK_TOP50_SHARE_MIN,
        },
        "inputs": [
            {
                "path": repo_relative(path, REPO_ROOT),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "custody": _input_custody(read_json(path)),
            }
            for path in input_jsons
        ],
        "series_count": len(series_reports),
        "aggregate_verdict": aggregate,
        "cross_series_dependence": cross_series,
        "series_reports": series_reports,
    }


def _input_custody(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    rows = payload.get("rows")
    pair_count = payload.get("pair_count", payload.get("n_pairs"))
    pair_index_basis = "unknown"
    if isinstance(rows, list) and len(rows) == DEFAULT_MIN_LENGTH:
        pair_indices = [
            row.get("pair_idx", row.get("pair_index"))
            for row in rows
            if isinstance(row, dict) and (row.get("pair_idx") is not None or row.get("pair_index") is not None)
        ]
        if (
            len(pair_indices) == DEFAULT_MIN_LENGTH
            and sorted(pair_indices) == list(range(DEFAULT_MIN_LENGTH))
        ):
            pair_index_basis = "absolute_0_to_599"
        else:
            pair_index_basis = "row_order_600_without_complete_absolute_pair_idx"
    elif pair_count == DEFAULT_MIN_LENGTH or any(
        len(value) == DEFAULT_MIN_LENGTH
        for value in payload.values()
        if isinstance(value, list)
    ):
        pair_index_basis = "implicit_ordered_600_no_pair_idx"
    keys = (
        "schema",
        "evidence_grade",
        "evidence_tag",
        "score_claim",
        "promotion_eligible",
        "rank_or_kill_eligible",
        "ready_for_exact_eval_dispatch",
        "ranking_only",
        "dispatch_attempted",
        "device",
        "platform",
        "n_pairs",
        "pair_count",
        "archive_sha256",
        "archive_bytes",
        "archive_id",
        "archive_path",
    )
    custody = {key: payload[key] for key in keys if key in payload}
    custody["pair_index_basis"] = pair_index_basis
    return custody


def _default_output_path() -> Path:
    stamp = _utc_now().replace(":", "").replace("-", "")
    return REPO_ROOT / "experiments" / "results" / f"600_pair_independence_test_{stamp}" / "report.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-json", type=Path, action="append", required=True)
    parser.add_argument(
        "--series-path",
        action="append",
        default=[],
        help=(
            "Optional dotted path to a numeric series. Use rows[].field or "
            "rows.field for list-of-dict columns. Repeatable. If omitted, "
            "the tool auto-discovers numeric vectors of --min-length or more."
        ),
    )
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--min-length", type=int, default=DEFAULT_MIN_LENGTH)
    parser.add_argument("--max-lag", type=int, default=DEFAULT_MAX_LAG)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args(argv)

    report = build_report(
        input_jsons=args.input_json,
        explicit_paths=args.series_path,
        min_length=args.min_length,
        max_lag=args.max_lag,
        max_depth=args.max_depth,
    )
    output_json = args.output_json or _default_output_path()
    write_json(output_json, report)

    if args.summary:
        print(
            "[600-pair-independence] "
            f"verdict={report['aggregate_verdict']} "
            f"series={report['series_count']} "
            f"output={repo_relative(output_json, REPO_ROOT)}"
        )
    return 0 if report["aggregate_verdict"] != "INCONCLUSIVE_MISSING_600_PAIR_SERIES" else 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
