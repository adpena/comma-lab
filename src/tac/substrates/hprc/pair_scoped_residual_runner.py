# SPDX-License-Identifier: MIT
"""Bounded-runner planning for HPRC pair-scoped residual candidates."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY

HPRC_PAIR_SCOPED_RESIDUAL_RUNNER_PLAN_SCHEMA = (
    "hprc_pair_scoped_residual_bounded_runner_plan.v1"
)
HPRC_PAIR_SCOPED_RESIDUAL_RUNNER_ROW_SCHEMA = (
    "hprc_pair_scoped_residual_bounded_runner_row.v1"
)


def build_pair_scoped_residual_bounded_runner_plan(
    *,
    pair_plan_path: str | Path,
    reuse_baseline_profile_path: str | Path,
    candidate_dir: str | Path,
    output_dir: str | Path,
    repo_root: str | Path,
    max_candidates: int = 3,
    max_pairs: int = 600,
    window_pairs: int = 50,
    scorer_batch_pairs: int = 1,
    allow_batch_shape_research_signal: bool = False,
    device: str = "cpu",
    allow_large_tensor_cache: bool = True,
    profile_tool_path: str | Path = "tools/profile_hprc_mlx_component_neutralization.py",
    incremental_tool_path: str | Path = "tools/profile_hprc_incremental_pair_response.py",
) -> dict[str, Any]:
    """Return executable rows for measured HPRC pair-scoped residual candidates.

    The input pair-plan is advisory signal, but each emitted row is an executable
    archive-bound materialization/profile command.  Baseline reuse is mandatory
    because these candidates mutate only residual tokens; recomputing the same
    baseline scorer response would waste wall clock and disk.
    """

    root = Path(repo_root).expanduser().resolve(strict=False)
    pair_plan = _load_json_object(_resolve(pair_plan_path, base=root))
    reuse_profile = _resolve(reuse_baseline_profile_path, base=root)
    if not reuse_profile.is_file():
        raise FileNotFoundError(f"missing reuse baseline profile: {reuse_profile}")
    candidate_root = _resolve(candidate_dir, base=root)
    if not candidate_root.is_dir():
        raise FileNotFoundError(f"missing candidate dir: {candidate_root}")
    selected_output_dir = _resolve(output_dir, base=root)
    if max_candidates < 1:
        raise ValueError("max_candidates must be >= 1")
    if max_pairs < 1:
        raise ValueError("max_pairs must be >= 1")
    if window_pairs < 1:
        raise ValueError("window_pairs must be >= 1")
    if scorer_batch_pairs < 1:
        raise ValueError("scorer_batch_pairs must be >= 1")
    if int(scorer_batch_pairs) != 1 and not allow_batch_shape_research_signal:
        raise ValueError(
            "scorer_batch_pairs > 1 requires allow_batch_shape_research_signal=True"
        )
    if device not in {"cpu", "gpu"}:
        raise ValueError("device must be 'cpu' or 'gpu'")

    rows = _candidate_rows_from_pair_plan(pair_plan)[: int(max_candidates)]
    runner_rows = [
        _runner_row(
            source_row=row,
            rank=index,
            root=root,
            candidate_dir=candidate_root,
            output_dir=selected_output_dir,
            reuse_baseline_profile_path=reuse_profile,
            profile_tool_path=Path(profile_tool_path),
            max_pairs=int(max_pairs),
            window_pairs=int(window_pairs),
            scorer_batch_pairs=int(scorer_batch_pairs),
            allow_batch_shape_research_signal=bool(allow_batch_shape_research_signal),
            device=device,
            allow_large_tensor_cache=bool(allow_large_tensor_cache),
            incremental_tool_path=Path(incremental_tool_path),
        )
        for index, row in enumerate(rows, start=1)
    ]
    return {
        "schema": HPRC_PAIR_SCOPED_RESIDUAL_RUNNER_PLAN_SCHEMA,
        "generated_at_utc": _utc_stamp(),
        "repo_root": root.as_posix(),
        "pair_plan_path": _resolve(pair_plan_path, base=root).as_posix(),
        "pair_plan_sha256": _sha256_file(_resolve(pair_plan_path, base=root)),
        "reuse_baseline_profile_path": reuse_profile.as_posix(),
        "reuse_baseline_profile_sha256": _sha256_file(reuse_profile),
        "candidate_dir": candidate_root.as_posix(),
        "output_dir": selected_output_dir.as_posix(),
        "max_candidates": int(max_candidates),
        "max_pairs": int(max_pairs),
        "window_pairs": int(window_pairs),
        "scorer_batch_pairs": int(scorer_batch_pairs),
        "allow_batch_shape_research_signal": bool(allow_batch_shape_research_signal),
        "device": device,
        "baseline_reuse_required": True,
        "runner_rows": runner_rows,
        "runner_policy": {
            "schema": "hprc_pair_scoped_residual_runner_policy.v1",
            "selection": (
                "execute candidate rows in order of estimated archive-byte removal, "
                "then compare MLX advisory deltas under the same baseline profile"
            ),
            "promotion": (
                "receiver proof plus exact CPU/CUDA auth eval are required before "
                "any score or promotion claim"
            ),
            "stop_conditions": [
                "better_receiver_proven_archive_bound_candidate",
                "precise_exact_axis_blocker",
                "durable_negative_evidence_for_candidate_family",
            ],
        },
        "blockers": [
            "runner_plan_not_executed",
            "mlx_local_response_is_advisory_not_score_authority",
            "contest_cpu_cuda_exact_eval_not_executed",
        ],
        **FALSE_AUTHORITY,
    }


def write_pair_scoped_residual_bounded_runner_plan(
    *,
    output_path: str | Path,
    plan: dict[str, Any],
    allow_overwrite: bool = False,
) -> Path:
    """Write a deterministic HPRC pair-scoped runner plan."""

    path = Path(output_path)
    if path.exists() and not allow_overwrite:
        raise FileExistsError(f"output exists; pass allow_overwrite=True: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(plan, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return path


def _runner_row(
    *,
    source_row: dict[str, Any],
    rank: int,
    root: Path,
    candidate_dir: Path,
    output_dir: Path,
    reuse_baseline_profile_path: Path,
    profile_tool_path: Path,
    max_pairs: int,
    window_pairs: int,
    scorer_batch_pairs: int,
    allow_batch_shape_research_signal: bool,
    device: str,
    allow_large_tensor_cache: bool,
    incremental_tool_path: Path,
) -> dict[str, Any]:
    transform = _required_string(source_row, "residual_transform")
    candidate_id = _candidate_id(transform)
    candidate_output_dir = output_dir / candidate_id
    tool = _resolve(profile_tool_path, base=root)
    incremental_tool = _resolve(incremental_tool_path, base=root)
    profile_variant_id = f"residual_transform_{_profile_variant_slug(transform)}"
    pair_ranges_arg = _format_pair_ranges(source_row.get("pair_ranges", []))
    argv = [
        sys.executable,
        str(tool),
        "--candidate-dir",
        candidate_dir.as_posix(),
        "--output-dir",
        candidate_output_dir.as_posix(),
        "--sections",
        "--residual-transforms",
        transform,
        "--reuse-baseline-profile",
        reuse_baseline_profile_path.as_posix(),
        "--max-pairs",
        str(max_pairs),
        "--window-pairs",
        str(window_pairs),
        "--scorer-batch-pairs",
        str(scorer_batch_pairs),
        "--device",
        device,
        "--force",
    ]
    if allow_large_tensor_cache:
        argv.append("--allow-large-tensor-cache")
    if allow_batch_shape_research_signal:
        argv.append("--allow-batch-shape-research-signal")
    incremental_argv = [
        sys.executable,
        str(incremental_tool),
        "--profile",
        (candidate_output_dir / "hprc_mlx_component_neutralization_profile.json").as_posix(),
        "--candidate-variant-id",
        profile_variant_id,
        "--pair-ranges",
        pair_ranges_arg,
        "--output-dir",
        (candidate_output_dir / "incremental_pair_response").as_posix(),
        "--device",
        device,
        "--scorer-batch-pairs",
        str(scorer_batch_pairs),
        "--force",
    ]
    if allow_large_tensor_cache:
        incremental_argv.append("--allow-large-tensor-cache")
    if allow_batch_shape_research_signal:
        incremental_argv.append("--allow-batch-shape-research-signal")
    return {
        "schema": HPRC_PAIR_SCOPED_RESIDUAL_RUNNER_ROW_SCHEMA,
        "rank": int(rank),
        "candidate_id": candidate_id,
        "family": "hprc_compact_receiver",
        "stage": "pre_entropy_residual_tokens",
        "scope": "pair",
        "source_variant_id": source_row.get("source_variant_id"),
        "residual_transform": transform,
        "threshold_abs_le": source_row.get("threshold_abs_le"),
        "selected_pair_count": source_row.get("selected_pair_count"),
        "protected_pair_count": source_row.get("protected_pair_count"),
        "estimated_archive_bytes_removed_vs_baseline": source_row.get(
            "estimated_archive_bytes_removed_vs_baseline"
        ),
        "estimated_delta_nonrate_pair_local_sum": source_row.get(
            "estimated_delta_nonrate_pair_local_sum"
        ),
        "estimated_delta_rate_score": source_row.get("estimated_delta_rate_score"),
        "pair_ranges": source_row.get("pair_ranges", []),
        "profile_output_dir": candidate_output_dir.as_posix(),
        "profile_command_argv": argv,
        "incremental_response_command_argv": incremental_argv,
        "expected_incremental_response_report": (
            candidate_output_dir
            / "incremental_pair_response"
            / "hprc_incremental_pair_response_report.json"
        ).as_posix(),
        "expected_profile_report": (
            candidate_output_dir / "hprc_mlx_component_neutralization_profile.json"
        ).as_posix(),
        "expected_profile_backlog": (
            candidate_output_dir / "hprc_scorer_ranked_residual_shrink_backlog.json"
        ).as_posix(),
        "scorer_batch_pairs": int(scorer_batch_pairs),
        "batch_shape_research_signal": int(scorer_batch_pairs) != 1,
        "receiver_proof_followup": {
            "schema": "hprc_pair_scoped_receiver_proof_followup.v1",
            "required": True,
            "input_after_profile": "best residual_transform_* variant 0.bin",
            "output": "receiver-proven archive-bound candidate plus exact-axis blocker",
        },
        "baseline_reuse_required": True,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
    }


def _candidate_rows_from_pair_plan(pair_plan: dict[str, Any]) -> list[dict[str, Any]]:
    rows = pair_plan.get("pair_scoped_residual_candidate_rows")
    if not isinstance(rows, list):
        raise ValueError("pair plan missing pair_scoped_residual_candidate_rows")
    candidates = [row for row in rows if isinstance(row, dict)]
    candidates.sort(
        key=lambda row: (
            -int(row.get("estimated_archive_bytes_removed_vs_baseline") or 0),
            int(row.get("protected_pair_count") or 0),
            str(row.get("residual_transform") or ""),
        )
    )
    return candidates


def _candidate_id(transform: str) -> str:
    digest = hashlib.sha256(transform.encode("utf-8")).hexdigest()[:16]
    kind = transform.split("=", 1)[0].strip().lower().replace("_", "-")
    return f"hprc-{kind}-{digest}"


def _profile_variant_slug(value: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")
    if len(slug) <= 80:
        return slug
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    return f"{slug[:63].rstrip('_')}_{digest}"


def _format_pair_ranges(value: Any) -> str:
    if not isinstance(value, list):
        raise ValueError("pair_ranges must be a list")
    parts = []
    for row in value:
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError(f"invalid pair range row: {row!r}")
        start, end = int(row[0]), int(row[1])
        parts.append(str(start) if start == end else f"{start}-{end}")
    return ",".join(parts)


def _required_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"missing non-empty {key}")
    return value


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _resolve(path: str | Path, *, base: Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (base / p).resolve(strict=False)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


__all__ = [
    "HPRC_PAIR_SCOPED_RESIDUAL_RUNNER_PLAN_SCHEMA",
    "HPRC_PAIR_SCOPED_RESIDUAL_RUNNER_ROW_SCHEMA",
    "build_pair_scoped_residual_bounded_runner_plan",
    "write_pair_scoped_residual_bounded_runner_plan",
]
