#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build a false-authority compact-renderer MLX value profile.

This tool joins an existing compact renderer runner report, an existing MLX
scorer response payload, and an existing cache materialization report. It does
not inflate, score, mutate archives, or infer section value from a baseline
alone. Section byte attribution is recorded when spine manifests expose it;
measured per-section value remains blocked until a neutralization/ablation
profile exists.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.archive_byte_profile import contest_rate_term  # noqa: E402
from tac.auth_eval_schema import contest_formula_score  # noqa: E402
from tac.substrates.hprc.archive_candidate import FALSE_AUTHORITY  # noqa: E402
from tac.substrates.hprc.spine_bounded_runner import (  # noqa: E402
    HPRC_MLX_COMPONENT_PROFILE_SCHEMA,
)

COMPACT_RENDERER_MLX_SECTION_VALUE_SOURCE_SCHEMA = (
    "compact_renderer_mlx_baseline_section_value_profile.v1"
)
_SECTION_ATTRIBUTION_BLOCKER = "section_neutralization_or_ablation_replay_missing"
_METADATA_SECTION_NAMES = frozenset({"rdo_plan", "manifest_json", "receiver_state"})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-runner-report", required=True, type=Path)
    parser.add_argument("--mlx-response", required=True, type=Path)
    parser.add_argument("--cache-report", required=True, type=Path)
    parser.add_argument(
        "--projection-manifest",
        action="append",
        type=Path,
        default=[],
        help="Optional HPRC projection manifest; defaults to paths in runner report.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--family", default=None)
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    repo_root = _resolve(args.repo_root, base=REPO_ROOT)
    output = _resolve(args.output, base=repo_root)
    if output.exists() and not args.force:
        raise SystemExit(f"output exists; pass --force: {output}")
    profile = build_compact_renderer_mlx_section_value_profile(
        compact_runner_report_path=_resolve(args.compact_runner_report, base=repo_root),
        mlx_response_path=_resolve(args.mlx_response, base=repo_root),
        cache_report_path=_resolve(args.cache_report, base=repo_root),
        projection_manifest_paths=[
            _resolve(path, base=repo_root) for path in args.projection_manifest
        ],
        repo_root=repo_root,
        family_override=args.family,
        tool_argv=[sys.executable, str(Path(__file__).resolve()), *raw_argv],
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(profile, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "schema": HPRC_MLX_COMPONENT_PROFILE_SCHEMA,
                "profile": output.as_posix(),
                "section_byte_record_count": len(profile["section_byte_records"]),
                "score_claim": False,
                "ready_for_exact_eval_dispatch": False,
            },
            sort_keys=True,
        )
    )
    return 0


def build_compact_renderer_mlx_section_value_profile(
    *,
    compact_runner_report_path: str | Path,
    mlx_response_path: str | Path,
    cache_report_path: str | Path,
    projection_manifest_paths: list[str | Path] | tuple[str | Path, ...] = (),
    repo_root: str | Path = REPO_ROOT,
    family_override: str | None = None,
    tool_argv: list[str] | None = None,
) -> dict[str, Any]:
    """Return a bounded-runner-compatible baseline MLX value profile."""

    root = Path(repo_root).expanduser().resolve(strict=False)
    runner_path = _resolve(compact_runner_report_path, base=root)
    response_path = _resolve(mlx_response_path, base=root)
    cache_path = _resolve(cache_report_path, base=root)
    runner = _load_json_object(runner_path)
    response = _load_json_object(response_path)
    cache = _load_json_object(cache_path)
    projection_paths = _projection_paths(
        explicit_paths=projection_manifest_paths,
        runner=runner,
        base=root,
    )
    projection_records = [
        _projection_record(path=path, payload=_load_json_object(path))
        for path in projection_paths
        if path.is_file()
    ]
    section_records = [
        row
        for projection in projection_records
        for row in projection["section_byte_records"]
    ]
    archive = _candidate_archive_record(
        runner=runner,
        response=response,
        cache=cache,
        base=root,
    )
    archive_bytes = _first_int(
        archive.get("bytes"),
        response.get("archive_size_bytes"),
        runner.get("archive_bytes"),
    )
    components, component_blockers = _nonrate_components(
        response=response,
        archive_bytes=archive_bytes,
    )
    family = (
        family_override
        or _first_str(
            *(projection.get("family") for projection in projection_records),
            runner.get("execute_family"),
            runner.get("family"),
            response.get("response_family"),
        )
        or "unknown"
    )
    projection_manifest_path = (
        projection_records[0]["path"] if projection_records else None
    )
    max_pairs = _first_int(
        response.get("max_pairs"),
        response.get("n_samples"),
        cache.get("cached_pair_count"),
        cache.get("pair_count"),
        runner.get("num_pairs"),
    )
    training_context = _training_context(runner)
    blockers = _profile_blockers(
        base=root,
        archive=archive,
        archive_bytes=archive_bytes,
        component_blockers=component_blockers,
        projection_paths=projection_paths,
        projection_records=projection_records,
        section_records=section_records,
        max_pairs=max_pairs,
        cache=cache,
        response=response,
        training_context=training_context,
    )
    section_value_rows = [
        {
            "variant_id": "baseline",
            "neutralized_section": "none",
            "family": family,
            "projection_manifest_path": projection_manifest_path,
            "archive_zip_bytes": archive_bytes,
            "archive_bytes_removed_vs_baseline": 0,
            "avg_posenet_dist": components.get("avg_posenet_dist"),
            "avg_segnet_dist": components.get("avg_segnet_dist"),
            "delta_avg_posenet_dist": 0.0,
            "delta_avg_segnet_dist": 0.0,
            "delta_nonrate_score": 0.0,
            "delta_rate_score": 0.0,
            "delta_total_mlx_score_advisory": 0.0,
            "nonrate_score": components.get("nonrate_score"),
            "rate_score": components.get("rate_term"),
            "canonical_score": components.get("canonical_score"),
            "marginal_status": "baseline_only_section_value_not_measured",
            **FALSE_AUTHORITY,
        }
    ]
    return {
        "schema": HPRC_MLX_COMPONENT_PROFILE_SCHEMA,
        "source_schema": COMPACT_RENDERER_MLX_SECTION_VALUE_SOURCE_SCHEMA,
        "created_at_unix": time.time(),
        "repo_root": root.as_posix(),
        "tool_argv": list(tool_argv or []),
        "family": family,
        "projection_manifest_path": projection_manifest_path,
        "source_reports": {
            "compact_runner_report": _file_record(runner_path),
            "mlx_response": _file_record(response_path),
            "cache_report": _file_record(cache_path),
        },
        "candidate_archive": archive,
        "cache_report_manifests": _cache_manifest_records(cache=cache, base=root),
        "mlx_response_summary": {
            "schema_version": response.get("schema_version"),
            "response_family": response.get("response_family"),
            "score_axis": response.get("score_axis"),
            "evidence_grade": response.get("evidence_grade"),
            "hardware_substrate": response.get("hardware_substrate"),
            "batch_pairs": response.get("batch_pairs"),
            "batch_shape_research_signal_allowed": response.get(
                "batch_shape_research_signal_allowed"
            ),
            "max_pairs": response.get("max_pairs"),
            "n_samples": response.get("n_samples"),
            "candidate_cache_pairs": response.get("candidate_cache_pairs"),
            "reference_cache_pairs": response.get("reference_cache_pairs"),
            "raw_sha256": response.get("raw_sha256"),
            "inflated_outputs_aggregate_sha256": response.get(
                "inflated_outputs_aggregate_sha256"
            ),
        },
        "training_context": training_context,
        "score_components": components,
        "archive_byte_records": _archive_byte_records(
            runner=runner,
            response=response,
            cache=cache,
            archive=archive,
        ),
        "projection_records": projection_records,
        "section_byte_records": section_records,
        "section_value_rows": section_value_rows,
        "section_attribution_blockers": _section_attribution_blockers(section_records),
        "scope_status": {
            "base_mlx_response": "executed" if not component_blockers else "blocked",
            "full_video": (
                "executed"
                if max_pairs is not None and max_pairs >= 600
                else "sampled_prefix_requires_full_video_rerun"
            ),
            "section": (
                "blocked_missing_section_neutralization_or_ablation_evidence"
                if section_records
                else "blocked_missing_projection_section_manifest"
            ),
            "cache_manifest": (
                "present"
                if _manifest_path(cache, "cache_manifest", base=root) is not None
                else "missing"
            ),
        },
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def _nonrate_components(
    *,
    response: dict[str, Any],
    archive_bytes: int | None,
) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    seg = _float_or_none(response.get("avg_segnet_dist"))
    pose = _float_or_none(response.get("avg_posenet_dist"))
    if seg is None:
        blockers.append("mlx_response_avg_segnet_dist_missing")
    if pose is None:
        blockers.append("mlx_response_avg_posenet_dist_missing")
    rate_term = contest_rate_term(archive_bytes) if archive_bytes is not None else None
    nonrate_score = (
        contest_formula_score(seg_dist=seg, pose_dist=pose, archive_bytes=0)
        if seg is not None and pose is not None
        else None
    )
    canonical = _float_or_none(response.get("canonical_score"))
    recomputed = (
        contest_formula_score(
            seg_dist=seg,
            pose_dist=pose,
            archive_bytes=int(archive_bytes),
        )
        if seg is not None and pose is not None and archive_bytes is not None
        else None
    )
    if canonical is not None and recomputed is not None and abs(canonical - recomputed) > 1e-5:
        blockers.append("mlx_response_canonical_score_mismatch")
    return (
        {
            "avg_segnet_dist": seg,
            "avg_posenet_dist": pose,
            "segnet_score_component": None if seg is None else 100.0 * seg,
            "posenet_score_component": None if pose is None else math.sqrt(10.0 * pose),
            "nonrate_score": nonrate_score,
            "rate_term": rate_term,
            "canonical_score": canonical,
            "recomputed_total_score": recomputed,
            "response_score_rate_contribution": _float_or_none(
                response.get("score_rate_contribution")
            ),
        },
        blockers,
    )


def _candidate_archive_record(
    *,
    runner: dict[str, Any],
    response: dict[str, Any],
    cache: dict[str, Any],
    base: Path,
) -> dict[str, Any]:
    cache_archive = cache.get("archive") if isinstance(cache.get("archive"), dict) else {}
    path = _first_str(
        runner.get("archive_path"),
        (runner.get("training_artifact") or {}).get("archive_path")
        if isinstance(runner.get("training_artifact"), dict)
        else None,
        cache_archive.get("path"),
    )
    resolved = _resolve(path, base=base) if path else None
    bytes_value = _first_int(
        runner.get("archive_bytes"),
        (runner.get("training_artifact") or {}).get("archive_bytes")
        if isinstance(runner.get("training_artifact"), dict)
        else None,
        cache_archive.get("bytes"),
        response.get("archive_size_bytes"),
    )
    sha = _first_str(
        runner.get("archive_sha256"),
        (runner.get("training_artifact") or {}).get("archive_sha256")
        if isinstance(runner.get("training_artifact"), dict)
        else None,
        cache_archive.get("sha256"),
        response.get("archive_sha256"),
    )
    record = {
        "path": None if resolved is None else resolved.as_posix(),
        "bytes": bytes_value,
        "sha256": sha,
    }
    if resolved is not None and resolved.is_file():
        record["observed_bytes"] = resolved.stat().st_size
        record["observed_sha256"] = _sha256_file(resolved)
    return record


def _projection_paths(
    *,
    explicit_paths: list[str | Path] | tuple[str | Path, ...],
    runner: dict[str, Any],
    base: Path,
) -> list[Path]:
    paths = [_resolve(path, base=base) for path in explicit_paths]
    for raw in runner.get("projection_manifest_paths") or []:
        if isinstance(raw, str):
            paths.append(_resolve(raw, base=base))
    single = runner.get("projection_manifest_path")
    if isinstance(single, str):
        paths.append(_resolve(single, base=base))
    return _dedupe_paths(paths)


def _projection_record(*, path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    spine = _representation_spine(payload)
    sections = spine.get("sections") if isinstance(spine.get("sections"), list) else []
    family = str(spine.get("family") or payload.get("family") or "unknown")
    return {
        "path": path.as_posix(),
        "sha256": _sha256_file(path),
        "family": family,
        "hprc_bin_bytes": payload.get("hprc_bin_bytes") or spine.get("hprc_bin_bytes"),
        "source": spine.get("source") if isinstance(spine.get("source"), dict) else {},
        "manifest_extra": (
            spine.get("manifest_extra")
            if isinstance(spine.get("manifest_extra"), dict)
            else {}
        ),
        "section_byte_records": [
            _section_byte_record(path=path, family=family, raw=row)
            for row in sections
            if isinstance(row, dict)
        ],
    }


def _section_byte_record(*, path: Path, family: str, raw: dict[str, Any]) -> dict[str, Any]:
    byte_count = _first_int(raw.get("bytes"), raw.get("length")) or 0
    section_name = str(raw.get("name") or "")
    is_metadata = section_name in _METADATA_SECTION_NAMES
    return {
        "schema": "compact_renderer_section_byte_record.v1",
        "projection_manifest_path": path.as_posix(),
        "family": family,
        "section_name": section_name,
        "section_role": str(raw.get("role") or ""),
        "section_id": raw.get("id"),
        "section_bytes": byte_count,
        "section_sha256": raw.get("sha256"),
        "rate_cost": contest_rate_term(byte_count),
        "value_status": (
            "projection_contract_metadata_not_candidate_runtime_spend"
            if is_metadata
            else "blocked_missing_section_neutralization_or_ablation_evidence"
        ),
        "blockers": (
            ["contest_cpu_cuda_exact_eval_not_executed"]
            if is_metadata
            else [_SECTION_ATTRIBUTION_BLOCKER]
        ),
        **FALSE_AUTHORITY,
    }


def _representation_spine(payload: dict[str, Any]) -> dict[str, Any]:
    manifest = payload.get("manifest")
    if isinstance(manifest, dict):
        spine = manifest.get("representation_spine")
        if isinstance(spine, dict):
            return spine
    projection = payload.get("projection")
    if isinstance(projection, dict):
        manifest = projection.get("manifest")
        if isinstance(manifest, dict) and isinstance(manifest.get("representation_spine"), dict):
            return manifest["representation_spine"]
    if isinstance(payload.get("representation_spine"), dict):
        return payload["representation_spine"]
    return {}


def _profile_blockers(
    *,
    base: Path,
    archive: dict[str, Any],
    archive_bytes: int | None,
    component_blockers: list[str],
    projection_paths: list[Path],
    projection_records: list[dict[str, Any]],
    section_records: list[dict[str, Any]],
    max_pairs: int | None,
    cache: dict[str, Any],
    response: dict[str, Any],
    training_context: dict[str, Any],
) -> list[str]:
    blockers = ["mlx_local_response_is_advisory_not_score_authority"]
    blockers.extend(component_blockers)
    if archive_bytes is None:
        blockers.append("candidate_archive_bytes_missing")
    if (
        archive.get("path")
        and archive.get("observed_sha256")
        and archive.get("sha256")
        and archive["observed_sha256"] != archive["sha256"]
    ):
        blockers.append("candidate_archive_sha256_mismatch")
    if response.get("score_claim") is not False:
        blockers.append("mlx_response_false_authority_flags_missing")
    if response.get("ready_for_exact_eval_dispatch") is not False:
        blockers.append("mlx_response_exact_dispatch_flag_not_false")
    if not projection_paths:
        blockers.append("projection_manifest_path_missing")
    elif not projection_records:
        blockers.append("projection_manifest_unreadable_or_missing")
    if not section_records:
        blockers.append("projection_section_byte_attribution_missing")
    elif any(_section_requires_value_replay(row) for row in section_records):
        blockers.append(_SECTION_ATTRIBUTION_BLOCKER)
    if max_pairs is None or max_pairs < 600:
        blockers.append("full_video_mlx_response_not_executed")
    if training_context.get("family_demote_eligible") is False:
        blockers.append("compact_base_long_training_required_before_family_demote")
    if _path_from(cache.get("cache_manifest")) is not None and _manifest_path(cache, "cache_manifest", base=base) is None:
        blockers.append("cache_manifest_path_missing")
    if _path_from(cache.get("inflated_outputs_manifest")) is not None and _manifest_path(cache, "inflated_outputs_manifest", base=base) is None:
        blockers.append("inflated_outputs_manifest_path_missing")
    blockers.append("contest_cpu_cuda_exact_eval_not_executed")
    return _dedupe_values(blockers)


def _training_context(runner: dict[str, Any]) -> dict[str, Any]:
    artifact = runner.get("training_artifact")
    artifact = artifact if isinstance(artifact, dict) else {}
    config = artifact.get("config_snapshot")
    config = config if isinstance(config, dict) else {}
    total_epochs = _first_int(
        artifact.get("total_epochs_completed"),
        artifact.get("per_epoch_metrics_count"),
        config.get("epochs"),
    )
    requested_epochs = _first_int(config.get("epochs"))
    smoke = total_epochs is not None and total_epochs <= 1
    return {
        "schema": "compact_renderer_training_context.v1",
        "mode": _first_str(runner.get("mode"), artifact.get("schema_version")),
        "requested_epochs": requested_epochs,
        "total_epochs_completed": total_epochs,
        "per_epoch_metrics_count": _first_int(artifact.get("per_epoch_metrics_count")),
        "total_wall_clock_seconds": _float_or_none(artifact.get("total_wall_clock_seconds")),
        "evidence_role": (
            "custody_rate_replay_smoke" if smoke else "trained_candidate_replay"
        ),
        "artifact_demote_eligible": True,
        "family_demote_eligible": False if smoke else None,
        "next_training_action": (
            "continue_many_epoch_training_or_import_long_checkpoint_before_family_demotion"
            if smoke
            else "compare_against_same_byte_ceiling_and_section_value_profile"
        ),
        "interpretation": (
            "This row can demote the one-epoch artifact but not the family; "
            "compact NeRV-style bases may require many more epochs."
            if smoke
            else "Training depth is not marked as one-epoch smoke by this profile."
        ),
        **FALSE_AUTHORITY,
    }


def _section_attribution_blockers(section_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "section_name": row["section_name"],
            "section_bytes": row["section_bytes"],
            "blocker": _SECTION_ATTRIBUTION_BLOCKER,
            "required_evidence": (
                "full-video MLX replay of a neutralized or candidate-added section "
                "with delta_nonrate plus charged rate delta"
            ),
            **FALSE_AUTHORITY,
        }
        for row in section_records
        if _section_requires_value_replay(row)
    ]


def _section_requires_value_replay(row: dict[str, Any]) -> bool:
    return str(row.get("section_name") or "") not in _METADATA_SECTION_NAMES


def _archive_byte_records(
    *,
    runner: dict[str, Any],
    response: dict[str, Any],
    cache: dict[str, Any],
    archive: dict[str, Any],
) -> list[dict[str, Any]]:
    cache_archive = cache.get("archive") if isinstance(cache.get("archive"), dict) else {}
    rows = [
        ("compact_runner_report", runner.get("archive_bytes"), runner.get("archive_sha256")),
        ("mlx_response", response.get("archive_size_bytes"), response.get("archive_sha256")),
        ("cache_report", cache_archive.get("bytes"), cache_archive.get("sha256")),
        ("selected_candidate_archive", archive.get("bytes"), archive.get("sha256")),
    ]
    return [
        {"source": source, "bytes": _first_int(bytes_value), "sha256": sha}
        for source, bytes_value, sha in rows
        if _first_int(bytes_value) is not None or sha is not None
    ]


def _cache_manifest_records(*, cache: dict[str, Any], base: Path) -> dict[str, Any]:
    keys = (
        "cache_manifest",
        "inflated_outputs_manifest",
        "hprc_direct_cache_report",
    )
    return {key: _optional_file_record(_manifest_path(cache, key, base=base)) for key in keys}


def _manifest_path(cache: dict[str, Any], key: str, *, base: Path) -> Path | None:
    raw = cache.get(key)
    path = _path_from(raw)
    if path is None:
        return None
    resolved = _resolve(path, base=base)
    return resolved if resolved.is_file() else None


def _path_from(value: Any) -> str | Path | None:
    if isinstance(value, (str, Path)) and str(value):
        return value
    return None


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _optional_file_record(path: Path | None) -> dict[str, Any] | None:
    return None if path is None else _file_record(path)


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _resolve(path: str | Path | None, *, base: Path) -> Path:
    if path is None:
        raise ValueError("path is required")
    p = Path(path).expanduser()
    return p if p.is_absolute() else (base / p).resolve(strict=False)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _first_int(*values: Any) -> int | None:
    for value in values:
        if isinstance(value, bool) or value is None:
            continue
        if isinstance(value, str) and value.strip() == "":
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        return parsed
    return None


def _float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _first_str(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value:
            return value
    return None


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    out: list[Path] = []
    for path in paths:
        key = path.as_posix()
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    return out


def _dedupe_values(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


if __name__ == "__main__":  # pragma: no cover
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"profile_compact_renderer_mlx_section_value failed: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
