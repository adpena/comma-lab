# SPDX-License-Identifier: MIT
"""Compare singleton and batched HPRC MLX scorer-response profiles."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY

HPRC_MLX_BATCH_PROFILE_COMPARISON_SCHEMA = "hprc_mlx_batch_profile_comparison.v1"
HPRC_MLX_COMPONENT_PROFILE_SCHEMA = "hprc_mlx_component_neutralization_profile.v1"


def compare_hprc_mlx_batch_profiles(
    *,
    singleton_profile_path: str | Path,
    batched_profile_path: str | Path,
    repo_root: str | Path = ".",
) -> dict[str, Any]:
    """Return a false-authority comparison of singleton versus batched MLX profiles."""

    root = Path(repo_root).expanduser().resolve(strict=False)
    singleton_path = _resolve(singleton_profile_path, base=root)
    batched_path = _resolve(batched_profile_path, base=root)
    singleton = _load_profile(singleton_path)
    batched = _load_profile(batched_path)
    _validate_profile_shapes(singleton=singleton, batched=batched)

    singleton_variants = _variant_rows(singleton)
    batched_variants = _variant_rows(batched)
    common_variant_ids = sorted(set(singleton_variants) & set(batched_variants))
    if not common_variant_ids:
        raise ValueError("profiles share no variant ids")

    response_rows = [
        _response_drift_row(
            variant_id=variant_id,
            singleton_variant=singleton_variants[variant_id],
            batched_variant=batched_variants[variant_id],
        )
        for variant_id in common_variant_ids
    ]
    delta_rows = [
        _section_delta_drift_row(
            variant_id=variant_id,
            singleton_rows=_section_rows(singleton),
            batched_rows=_section_rows(batched),
        )
        for variant_id in common_variant_ids
        if variant_id in _section_rows(singleton) and variant_id in _section_rows(batched)
    ]
    return {
        "schema": HPRC_MLX_BATCH_PROFILE_COMPARISON_SCHEMA,
        "generated_at_utc": _utc_stamp(),
        "repo_root": root.as_posix(),
        "singleton_profile_path": singleton_path.as_posix(),
        "batched_profile_path": batched_path.as_posix(),
        "comparison_context": {
            "singleton_scorer_batch_pairs": int(singleton.get("scorer_batch_pairs") or 1),
            "batched_scorer_batch_pairs": int(batched.get("scorer_batch_pairs") or 1),
            "singleton_baseline_reuse_enabled": bool(
                singleton.get("baseline_reuse", {}).get("enabled")
            ),
            "batched_baseline_reuse_enabled": bool(
                batched.get("baseline_reuse", {}).get("enabled")
            ),
            "note": (
                "Raw wall-clock is not promotion evidence when baseline-reuse differs; "
                "use per-scored-variant seconds and singleton exact replay before promotion."
            ),
        },
        "wall_clock": _wall_clock(singleton=singleton, batched=batched),
        "variant_response_drift_rows": response_rows,
        "section_delta_drift_rows": delta_rows,
        "max_abs_response_drift": _max_abs(response_rows, "max_abs_response_drift"),
        "max_abs_delta_drift": _max_abs(delta_rows, "max_abs_delta_drift"),
        "blockers": [
            "batched_mlx_shape_is_research_signal_not_score_authority",
            "singleton_full_replay_required_before_promotion",
            "contest_cpu_cuda_exact_eval_not_executed",
        ],
        **FALSE_AUTHORITY,
    }


def write_hprc_mlx_batch_profile_comparison(
    *,
    output_path: str | Path,
    comparison: dict[str, Any],
    allow_overwrite: bool = False,
) -> Path:
    """Write a deterministic profile comparison JSON file."""

    path = Path(output_path)
    if path.exists() and not allow_overwrite:
        raise FileExistsError(f"output exists; pass allow_overwrite=True: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(comparison, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _validate_profile_shapes(*, singleton: dict[str, Any], batched: dict[str, Any]) -> None:
    if int(singleton.get("scorer_batch_pairs") or 1) != 1:
        raise ValueError("singleton profile must have scorer_batch_pairs=1")
    if int(batched.get("scorer_batch_pairs") or 1) <= 1:
        raise ValueError("batched profile must have scorer_batch_pairs>1")
    if not bool(batched.get("batch_shape_research_signal")):
        raise ValueError("batched profile must declare batch_shape_research_signal")
    for key in ("max_pairs", "reference_cache_dir"):
        if singleton.get(key) != batched.get(key):
            raise ValueError(
                f"profile mismatch for {key}: {singleton.get(key)!r} != {batched.get(key)!r}"
            )


def _wall_clock(*, singleton: dict[str, Any], batched: dict[str, Any]) -> dict[str, Any]:
    singleton_seconds = float(singleton.get("elapsed_seconds") or 0.0)
    batched_seconds = float(batched.get("elapsed_seconds") or 0.0)
    singleton_jobs = _scored_job_count(singleton)
    batched_jobs = _scored_job_count(batched)
    return {
        "singleton_elapsed_seconds": singleton_seconds,
        "batched_elapsed_seconds": batched_seconds,
        "raw_speedup_singleton_over_batched": (
            singleton_seconds / batched_seconds if batched_seconds > 0 else None
        ),
        "singleton_scored_variant_count": singleton_jobs,
        "batched_scored_variant_count": batched_jobs,
        "singleton_seconds_per_scored_variant": (
            singleton_seconds / singleton_jobs if singleton_jobs else None
        ),
        "batched_seconds_per_scored_variant": (
            batched_seconds / batched_jobs if batched_jobs else None
        ),
        "per_scored_variant_speedup_singleton_over_batched": (
            (singleton_seconds / singleton_jobs) / (batched_seconds / batched_jobs)
            if singleton_jobs and batched_jobs and batched_seconds > 0
            else None
        ),
    }


def _scored_job_count(profile: dict[str, Any]) -> int:
    variant_count = len(_variant_rows(profile))
    baseline_reused = bool(profile.get("baseline_reuse", {}).get("enabled"))
    return max(variant_count - (1 if baseline_reused else 0), 0)


def _response_drift_row(
    *,
    variant_id: str,
    singleton_variant: dict[str, Any],
    batched_variant: dict[str, Any],
) -> dict[str, Any]:
    singleton = _load_json_object(Path(str(singleton_variant["mlx_response"])))
    batched = _load_json_object(Path(str(batched_variant["mlx_response"])))
    fields = (
        "avg_segnet_dist",
        "avg_posenet_dist",
        "canonical_score",
        "score_rate_contribution",
    )
    drifts = {
        f"delta_{field}": float(batched.get(field) or 0.0)
        - float(singleton.get(field) or 0.0)
        for field in fields
    }
    return {
        "variant_id": variant_id,
        "archive_sha256_match": singleton_variant.get("archive_zip_sha256")
        == batched_variant.get("archive_zip_sha256"),
        "hprc_0bin_sha256_match": singleton_variant.get("hprc_0bin_sha256")
        == batched_variant.get("hprc_0bin_sha256"),
        **drifts,
        "max_abs_response_drift": max(abs(value) for value in drifts.values()),
        **FALSE_AUTHORITY,
    }


def _section_delta_drift_row(
    *,
    variant_id: str,
    singleton_rows: dict[str, dict[str, Any]],
    batched_rows: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    singleton = singleton_rows[variant_id]
    batched = batched_rows[variant_id]
    fields = (
        "delta_nonrate_score",
        "delta_rate_score",
        "delta_total_mlx_score_advisory",
        "delta_avg_posenet_dist",
        "delta_avg_segnet_dist",
    )
    drifts = {
        f"delta_{field}": float(batched.get(field) or 0.0)
        - float(singleton.get(field) or 0.0)
        for field in fields
    }
    return {
        "variant_id": variant_id,
        "archive_bytes_removed_vs_baseline_match": int(
            singleton.get("archive_bytes_removed_vs_baseline") or 0
        )
        == int(batched.get("archive_bytes_removed_vs_baseline") or 0),
        **drifts,
        "max_abs_delta_drift": max(abs(value) for value in drifts.values()),
        **FALSE_AUTHORITY,
    }


def _variant_rows(profile: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = profile.get("variant_rows")
    if not isinstance(rows, list):
        raise ValueError("profile missing variant_rows")
    return {
        str(row["variant_id"]): row
        for row in rows
        if isinstance(row, dict) and row.get("variant_id")
    }


def _section_rows(profile: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = profile.get("section_value_rows")
    if not isinstance(rows, list):
        raise ValueError("profile missing section_value_rows")
    return {
        str(row["variant_id"]): row
        for row in rows
        if isinstance(row, dict) and row.get("variant_id")
    }


def _load_profile(path: Path) -> dict[str, Any]:
    payload = _load_json_object(path)
    if payload.get("schema") != HPRC_MLX_COMPONENT_PROFILE_SCHEMA:
        raise ValueError(f"profile has wrong schema: {path}")
    return payload


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be object: {path}")
    return payload


def _resolve(path: str | Path, *, base: Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (base / p).resolve(strict=False)


def _max_abs(rows: list[dict[str, Any]], key: str) -> float | None:
    if not rows:
        return None
    return max(float(row.get(key) or 0.0) for row in rows)


def _utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


__all__ = [
    "HPRC_MLX_BATCH_PROFILE_COMPARISON_SCHEMA",
    "compare_hprc_mlx_batch_profiles",
    "write_hprc_mlx_batch_profile_comparison",
]
