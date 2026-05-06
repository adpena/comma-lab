#!/usr/bin/env python3
"""Merge harvested direct finite-difference component-sensitivity shards.

This is the lightweight post-harvest path for Lightning shard outputs. It
does not load renderers, scorers, videos, masks, or poses; it validates shard
metadata and delegates tensor merging to
``profile_component_sensitivity._merge_finite_difference_shard_maps``.
"""
from __future__ import annotations

import argparse
import importlib.util
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    _bootstrap_path = Path(__file__).resolve().parent.parent / "tools" / "tool_bootstrap.py"
    _spec = importlib.util.spec_from_file_location("tool_bootstrap", _bootstrap_path)
    if _spec is None or _spec.loader is None:
        raise
    _tool_bootstrap = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_tool_bootstrap)
    ensure_repo_imports = _tool_bootstrap.ensure_repo_imports
    repo_root_from_tool = _tool_bootstrap.repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from experiments.profile_component_sensitivity import (
    COMPONENT_OUTPUTS,
    FINITE_DIFFERENCE_MERGE_SCHEMA,
    FINITE_DIFFERENCE_SHARD_SCHEMA,
    ComponentSensitivityProfileError,
    _channel_ref_payload,
    _channel_ref_sha256,
    _load_profile_summary,
    _merge_finite_difference_shard_maps,
    _refs_from_payload,
)
from tac.repo_io import json_line, json_text, write_json
from tac.sensitivity_map import load_sensitivity_map, save_sensitivity_map

PRODUCER = "experiments/merge_component_sensitivity_shards.py"
SUMMARY_FILENAME = "component_sensitivity_profile_summary.json"
VALIDATION_FILENAME = "component_sensitivity_shard_merge_validation.json"
VALIDATION_FORMAT = "component_sensitivity_shard_merge_validation_v1"
EXPECTED_SOURCE = "direct_renderer_cuda_finite_difference_component_response"


class ComponentSensitivityShardMergeError(ValueError):
    """Raised when finite-difference shards cannot be safely merged."""


@dataclass(frozen=True)
class ShardInput:
    root: Path
    summary: dict[str, Any]
    shard: dict[str, Any]
    shard_index: int
    shard_count: int
    assigned_refs: list[tuple[str, int]]
    all_refs: list[tuple[str, int]]


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json_text(payload).encode("utf-8")


_write_json = write_json


def _positive_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ComponentSensitivityShardMergeError(f"{field} must be an integer")
    if value <= 0:
        raise ComponentSensitivityShardMergeError(f"{field} must be positive")
    return int(value)


def _nonnegative_int(value: Any, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ComponentSensitivityShardMergeError(f"{field} must be an integer")
    if value < 0:
        raise ComponentSensitivityShardMergeError(f"{field} must be nonnegative")
    return int(value)


def _load_shard_input(root: Path) -> ShardInput:
    if not root.is_dir():
        raise ComponentSensitivityShardMergeError(f"shard dir not found: {root}")
    summary = _load_profile_summary(root / SUMMARY_FILENAME)
    shard = summary.get("finite_difference_shard")
    if not isinstance(shard, dict) or shard.get("schema") != FINITE_DIFFERENCE_SHARD_SCHEMA:
        raise ComponentSensitivityShardMergeError(f"{root}: missing finite-difference shard metadata")
    if shard.get("is_shard") is not True:
        raise ComponentSensitivityShardMergeError(f"{root}: merge input must be a partial shard")
    if summary.get("sensitivity_source") != EXPECTED_SOURCE:
        raise ComponentSensitivityShardMergeError(f"{root}: shard source is not direct finite difference")

    shard_index = _nonnegative_int(shard.get("shard_index"), field=f"{root}.shard_index")
    shard_count = _positive_int(shard.get("shard_count"), field=f"{root}.shard_count")
    if shard_index >= shard_count:
        raise ComponentSensitivityShardMergeError(
            f"{root}: shard_index={shard_index} outside shard_count={shard_count}"
        )

    assigned_refs = _refs_from_payload(
        shard.get("assigned_channel_refs"),
        label=f"{root}.assigned_channel_refs",
    )
    all_refs = _refs_from_payload(
        shard.get("all_channel_refs"),
        label=f"{root}.all_channel_refs",
    )
    if _channel_ref_sha256(assigned_refs) != shard.get("assigned_channel_sha256"):
        raise ComponentSensitivityShardMergeError(f"{root}: assigned channel SHA mismatch")
    if _channel_ref_sha256(all_refs) != shard.get("all_channel_sha256"):
        raise ComponentSensitivityShardMergeError(f"{root}: all-channel SHA mismatch")
    assigned_count = _nonnegative_int(
        shard.get("assigned_channel_count", len(assigned_refs)),
        field=f"{root}.assigned_channel_count",
    )
    all_count = _nonnegative_int(
        shard.get("all_channel_count", len(all_refs)),
        field=f"{root}.all_channel_count",
    )
    if assigned_count != len(assigned_refs):
        raise ComponentSensitivityShardMergeError(f"{root}: assigned channel count mismatch")
    if all_count != len(all_refs):
        raise ComponentSensitivityShardMergeError(f"{root}: all-channel count mismatch")

    return ShardInput(
        root=root,
        summary=summary,
        shard=dict(shard),
        shard_index=shard_index,
        shard_count=shard_count,
        assigned_refs=assigned_refs,
        all_refs=all_refs,
    )


def _load_and_validate_shards(
    shard_dirs: Sequence[str | Path],
    *,
    expected_shard_count: int | None,
    allow_incomplete: bool,
) -> tuple[list[ShardInput], dict[str, Any]]:
    if not shard_dirs:
        raise ComponentSensitivityShardMergeError("at least one --shard-dir is required")
    if expected_shard_count is not None:
        _positive_int(expected_shard_count, field="--expected-shard-count")

    inputs = [_load_shard_input(Path(path)) for path in shard_dirs]
    inputs.sort(key=lambda item: (item.shard_index, str(item.root)))

    declared_count = inputs[0].shard_count
    all_refs = inputs[0].all_refs
    invariant = {
        "n_pairs_total": inputs[0].summary.get("n_pairs_total"),
        "n_pairs_selected": inputs[0].summary.get("n_pairs_selected"),
        "n_pairs_calibration": inputs[0].summary.get("n_pairs_calibration"),
        "n_pairs_holdout": inputs[0].summary.get("n_pairs_holdout"),
        "split_seed": inputs[0].summary.get("split_seed"),
        "finite_difference_epsilon": inputs[0].summary.get("finite_difference_epsilon"),
        "device": inputs[0].summary.get("device"),
        "component_response_path": inputs[0].summary.get("component_response_path"),
        "all_channel_sha256": inputs[0].shard.get("all_channel_sha256"),
        "shard_count": declared_count,
    }

    seen_indices: dict[int, Path] = {}
    seen_refs: set[tuple[str, int]] = set()
    duplicate_refs: list[tuple[str, int]] = []
    covered_refs: list[tuple[str, int]] = []

    for item in inputs:
        if item.shard_count != declared_count:
            raise ComponentSensitivityShardMergeError(
                f"{item.root}: shard_count={item.shard_count} differs from first shard={declared_count}"
            )
        if item.all_refs != all_refs:
            raise ComponentSensitivityShardMergeError(f"{item.root}: all-channel refs differ from first shard")
        current_invariant = {
            "n_pairs_total": item.summary.get("n_pairs_total"),
            "n_pairs_selected": item.summary.get("n_pairs_selected"),
            "n_pairs_calibration": item.summary.get("n_pairs_calibration"),
            "n_pairs_holdout": item.summary.get("n_pairs_holdout"),
            "split_seed": item.summary.get("split_seed"),
            "finite_difference_epsilon": item.summary.get("finite_difference_epsilon"),
            "device": item.summary.get("device"),
            "component_response_path": item.summary.get("component_response_path"),
            "all_channel_sha256": item.shard.get("all_channel_sha256"),
            "shard_count": item.shard_count,
        }
        if current_invariant != invariant:
            raise ComponentSensitivityShardMergeError(f"{item.root}: shard invariant mismatch")
        if item.shard_index in seen_indices:
            raise ComponentSensitivityShardMergeError(
                "duplicate finite-difference shard index "
                f"{item.shard_index}: {seen_indices[item.shard_index]} and {item.root}"
            )
        seen_indices[item.shard_index] = item.root
        for ref in item.assigned_refs:
            if ref in seen_refs:
                duplicate_refs.append(ref)
            else:
                seen_refs.add(ref)
                covered_refs.append(ref)

    if duplicate_refs:
        raise ComponentSensitivityShardMergeError(
            f"duplicate finite-difference shard channel: {duplicate_refs[0]}"
        )

    expected_count = int(expected_shard_count) if expected_shard_count is not None else declared_count
    if expected_count != declared_count:
        raise ComponentSensitivityShardMergeError(
            f"expected shard count {expected_count} does not match declared shard_count {declared_count}"
        )

    provided_indices = set(seen_indices)
    missing_indices = [idx for idx in range(expected_count) if idx not in provided_indices]
    missing_refs = [ref for ref in all_refs if ref not in seen_refs]
    extra_refs = sorted(seen_refs - set(all_refs))
    if extra_refs:
        raise ComponentSensitivityShardMergeError(
            f"finite-difference shard coverage has refs outside all-channel refs: {extra_refs[:5]}"
        )
    if (missing_indices or missing_refs) and not allow_incomplete:
        raise ComponentSensitivityShardMergeError(
            "missing finite-difference shards/channels: "
            f"missing_shard_indices={missing_indices[:16]} missing_channel_count={len(missing_refs)}"
        )

    validation_base = {
        "declared_shard_count": declared_count,
        "expected_shard_count": expected_count,
        "provided_shard_count": len(inputs),
        "provided_shard_indices": sorted(provided_indices),
        "missing_shard_indices": missing_indices,
        "all_channel_count": len(all_refs),
        "all_channel_sha256": _channel_ref_sha256(all_refs),
        "covered_channel_count": len(covered_refs),
        "covered_channel_sha256": _channel_ref_sha256(covered_refs),
        "missing_channel_count": len(missing_refs),
        "missing_channel_sha256": _channel_ref_sha256(missing_refs),
        "coverage": "incomplete" if missing_indices or missing_refs else "exactly_once",
        "allow_incomplete": bool(allow_incomplete),
        "source_shards": [
            {
                "path": str(item.root),
                "shard_index": item.shard_index,
                "assigned_channel_count": len(item.assigned_refs),
                "assigned_channel_sha256": _channel_ref_sha256(item.assigned_refs),
            }
            for item in inputs
        ],
    }
    return inputs, validation_base


def _zero_like_maps(template: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        str(key): torch.zeros_like(value.detach().to(torch.float32).cpu())
        for key, value in template.items()
    }


def _normalize_shard_for_subset(
    *,
    source: ShardInput,
    normalized_index: int,
    normalized_count: int,
    covered_refs: list[tuple[str, int]],
) -> dict[str, Any]:
    shard = dict(source.shard)
    shard.update(
        {
            "is_shard": True,
            "shard_index": int(normalized_index),
            "shard_count": int(normalized_count),
            "all_channel_count": len(covered_refs),
            "all_channel_refs": _channel_ref_payload(covered_refs),
            "all_channel_sha256": _channel_ref_sha256(covered_refs),
            "source_shard_index": source.shard_index,
            "source_declared_shard_count": source.shard_count,
        }
    )
    return shard


def _write_normalized_shard(
    *,
    dest: Path,
    source: ShardInput,
    normalized_shard: Mapping[str, Any],
) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    summary = dict(source.summary)
    summary["finite_difference_shard"] = dict(normalized_shard)
    _write_json(dest / SUMMARY_FILENAME, summary)
    for component in COMPONENT_OUTPUTS:
        values, metadata = load_sensitivity_map(source.root / f"{component}_sensitivity_map.pt")
        save_sensitivity_map(
            dest / f"{component}_sensitivity_map.pt",
            values,
            metadata={**metadata, "finite_difference_shard": dict(normalized_shard)},
        )
        holdout_values, holdout_metadata = load_sensitivity_map(
            source.root / f"{component}_holdout_sensitivity_map.pt"
        )
        save_sensitivity_map(
            dest / f"{component}_holdout_sensitivity_map.pt",
            holdout_values,
            metadata={**holdout_metadata, "finite_difference_shard": dict(normalized_shard)},
        )


def _write_empty_normalized_shard(
    *,
    dest: Path,
    template: ShardInput,
    normalized_index: int,
    normalized_count: int,
    covered_refs: list[tuple[str, int]],
) -> None:
    empty_shard = {
        **template.shard,
        "is_shard": True,
        "shard_index": int(normalized_index),
        "shard_count": int(normalized_count),
        "assigned_channel_count": 0,
        "assigned_channel_refs": [],
        "assigned_channel_sha256": _channel_ref_sha256([]),
        "all_channel_count": len(covered_refs),
        "all_channel_refs": _channel_ref_payload(covered_refs),
        "all_channel_sha256": _channel_ref_sha256(covered_refs),
        "synthetic_empty_for_incomplete_merge": True,
        "source_declared_shard_count": template.shard_count,
    }
    dest.mkdir(parents=True, exist_ok=True)
    summary = dict(template.summary)
    summary["finite_difference_shard"] = empty_shard
    _write_json(dest / SUMMARY_FILENAME, summary)
    for component in COMPONENT_OUTPUTS:
        values, _metadata = load_sensitivity_map(template.root / f"{component}_sensitivity_map.pt")
        zero_values = _zero_like_maps(values)
        save_sensitivity_map(
            dest / f"{component}_sensitivity_map.pt",
            zero_values,
            metadata={
                **summary,
                "component": component,
                "scorer_target": component,
                "finite_difference_shard": empty_shard,
            },
        )
        save_sensitivity_map(
            dest / f"{component}_holdout_sensitivity_map.pt",
            zero_values,
            metadata={
                **summary,
                "component": component,
                "scorer_target": component,
                "split": "holdout",
                "finite_difference_shard": empty_shard,
            },
        )


def _merge_allow_incomplete(
    *,
    inputs: list[ShardInput],
    output_dir: Path,
    covered_refs: list[tuple[str, int]],
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    if not covered_refs:
        raise ComponentSensitivityShardMergeError("allow-incomplete merge has no covered channels")
    scratch = output_dir / ".merge_component_sensitivity_shards_tmp"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    try:
        normalized_count = max(2, len(inputs))
        normalized_dirs: list[Path] = []
        for normalized_index, source in enumerate(inputs):
            dest = scratch / f"shard_{normalized_index:04d}"
            normalized_shard = _normalize_shard_for_subset(
                source=source,
                normalized_index=normalized_index,
                normalized_count=normalized_count,
                covered_refs=covered_refs,
            )
            _write_normalized_shard(
                dest=dest,
                source=source,
                normalized_shard=normalized_shard,
            )
            normalized_dirs.append(dest)
        while len(normalized_dirs) < normalized_count:
            dest = scratch / f"shard_{len(normalized_dirs):04d}"
            _write_empty_normalized_shard(
                dest=dest,
                template=inputs[0],
                normalized_index=len(normalized_dirs),
                normalized_count=normalized_count,
                covered_refs=covered_refs,
            )
            normalized_dirs.append(dest)
        return _merge_finite_difference_shard_maps(normalized_dirs)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _merged_fd_shard(
    *,
    template: ShardInput,
    covered_refs: list[tuple[str, int]],
    validation_base: Mapping[str, Any],
) -> dict[str, Any]:
    exact = validation_base["coverage"] == "exactly_once"
    assigned_refs = template.all_refs if exact else covered_refs
    shard = dict(template.shard)
    shard.update(
        {
            "is_shard": False,
            "shard_index": 0,
            "shard_count": 1,
            "assigned_channel_count": len(assigned_refs),
            "assigned_channel_refs": _channel_ref_payload(assigned_refs),
            "assigned_channel_sha256": _channel_ref_sha256(assigned_refs),
            "all_channel_count": len(template.all_refs),
            "all_channel_refs": _channel_ref_payload(template.all_refs),
            "all_channel_sha256": _channel_ref_sha256(template.all_refs),
            "merge_required_for_certification_handoff": not exact,
            "merged_from_shards": True,
            "source_declared_shard_count": validation_base["declared_shard_count"],
            "source_shard_indices": validation_base["provided_shard_indices"],
            "missing_shard_indices": validation_base["missing_shard_indices"],
            "missing_channel_count": validation_base["missing_channel_count"],
        }
    )
    return shard


def _summary_base(
    *,
    template: ShardInput,
    merged_shard: Mapping[str, Any],
    merge_metadata: Mapping[str, Any],
    validation_base: Mapping[str, Any],
) -> dict[str, Any]:
    exact = validation_base["coverage"] == "exactly_once"
    handoff_eligible = bool(
        exact
        and merge_metadata.get("schema") == FINITE_DIFFERENCE_MERGE_SCHEMA
        and merge_metadata.get("coverage") == "exactly_once"
        and merge_metadata.get("certification_handoff_eligible") is True
    )
    summary = dict(template.summary)
    summary.update(
        {
            "merge_tool": PRODUCER,
            "merge_without_model_or_data_setup": True,
            "score_claim": False,
            "promotion_eligible": False,
            "official_component_response": False,
            "canonical_scorer_path": False,
            "finite_difference_shard": dict(merged_shard),
            "finite_difference_merge": dict(merge_metadata),
            "certification_handoff_eligible": handoff_eligible,
            "elapsed_s": 0.0,
        }
    )
    return summary


def _write_merged_outputs(
    *,
    output_dir: Path,
    calibration_maps: Mapping[str, Mapping[str, torch.Tensor]],
    holdout_maps: Mapping[str, Mapping[str, torch.Tensor]],
    summary_base: Mapping[str, Any],
) -> tuple[dict[str, str], dict[str, str]]:
    map_paths: dict[str, str] = {}
    holdout_map_paths: dict[str, str] = {}
    for component in COMPONENT_OUTPUTS:
        map_path = output_dir / f"{component}_sensitivity_map.pt"
        save_sensitivity_map(
            map_path,
            calibration_maps[component],
            metadata={
                **summary_base,
                "component": component,
                "scorer_target": component,
            },
        )
        map_paths[component] = str(map_path)
        holdout_path = output_dir / f"{component}_holdout_sensitivity_map.pt"
        save_sensitivity_map(
            holdout_path,
            holdout_maps[component],
            metadata={
                **summary_base,
                "component": component,
                "scorer_target": component,
                "split": "holdout",
            },
        )
        holdout_map_paths[component] = str(holdout_path)
    return map_paths, holdout_map_paths


def merge_component_sensitivity_shards(
    *,
    shard_dirs: Sequence[str | Path],
    output_dir: str | Path,
    expected_shard_count: int | None = None,
    allow_incomplete: bool = False,
) -> dict[str, Any]:
    """Merge direct finite-difference shard maps and write summary artifacts."""

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs, validation_base = _load_and_validate_shards(
        shard_dirs,
        expected_shard_count=expected_shard_count,
        allow_incomplete=allow_incomplete,
    )
    out_resolved = out_dir.resolve()
    for item in inputs:
        if item.root.resolve() == out_resolved:
            raise ComponentSensitivityShardMergeError(
                f"output dir must not overwrite source shard dir: {item.root}"
            )
    covered_ref_set = {
        ref
        for item in inputs
        for ref in item.assigned_refs
    }
    covered_refs = [ref for ref in inputs[0].all_refs if ref in covered_ref_set]

    if validation_base["coverage"] == "exactly_once":
        calibration_maps, holdout_maps, merge_metadata = _merge_finite_difference_shard_maps(
            [item.root for item in inputs]
        )
    else:
        calibration_maps, holdout_maps, merge_metadata = _merge_allow_incomplete(
            inputs=inputs,
            output_dir=out_dir,
            covered_refs=covered_refs,
        )

    merge_metadata = {
        **merge_metadata,
        "source_shard_count": validation_base["provided_shard_count"],
        "declared_shard_count": validation_base["declared_shard_count"],
        "expected_shard_count": validation_base["expected_shard_count"],
        "source_shard_dirs": [str(item.root) for item in inputs],
        "source_shard_indices": validation_base["provided_shard_indices"],
        "all_channel_count": validation_base["all_channel_count"],
        "all_channel_sha256": validation_base["all_channel_sha256"],
        "covered_channel_count": validation_base["covered_channel_count"],
        "covered_channel_sha256": validation_base["covered_channel_sha256"],
        "missing_shard_indices": validation_base["missing_shard_indices"],
        "missing_channel_count": validation_base["missing_channel_count"],
        "missing_channel_sha256": validation_base["missing_channel_sha256"],
        "coverage": validation_base["coverage"],
        "allow_incomplete": bool(allow_incomplete),
        "certification_handoff_eligible": validation_base["coverage"] == "exactly_once",
        "promotion_eligible": False,
        "score_claim": False,
    }
    merged_shard = _merged_fd_shard(
        template=inputs[0],
        covered_refs=covered_refs,
        validation_base=validation_base,
    )
    summary_base = _summary_base(
        template=inputs[0],
        merged_shard=merged_shard,
        merge_metadata=merge_metadata,
        validation_base=validation_base,
    )
    map_paths, holdout_map_paths = _write_merged_outputs(
        output_dir=out_dir,
        calibration_maps=calibration_maps,
        holdout_maps=holdout_maps,
        summary_base=summary_base,
    )

    validation_path = out_dir / VALIDATION_FILENAME
    summary = {
        **summary_base,
        "map_paths": map_paths,
        "holdout_map_paths": holdout_map_paths,
        "merge_validation_json": str(validation_path),
    }
    summary_path = out_dir / SUMMARY_FILENAME
    _write_json(summary_path, summary)

    validation = {
        "format": VALIDATION_FORMAT,
        "tool": PRODUCER,
        "artifact_dir": str(out_dir),
        "summary_path": str(summary_path),
        "map_paths": map_paths,
        "holdout_map_paths": holdout_map_paths,
        "score_claim": False,
        "promotion_eligible": False,
        "certification_handoff_eligible": summary["certification_handoff_eligible"],
        "finite_difference_merge": merge_metadata,
        **validation_base,
    }
    _write_json(validation_path, validation)
    return validation


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--shard-dir",
        action="append",
        type=Path,
        required=True,
        help="Directory containing one harvested direct finite-difference shard. Repeat for each shard.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-shard-count", type=int, default=None)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Allow a non-overlapping subset of shards to merge as incomplete/non-handoff.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        validation = merge_component_sensitivity_shards(
            shard_dirs=args.shard_dir,
            output_dir=args.output_dir,
            expected_shard_count=args.expected_shard_count,
            allow_incomplete=args.allow_incomplete,
        )
    except (
        ComponentSensitivityShardMergeError,
        ComponentSensitivityProfileError,
        FileNotFoundError,
        RuntimeError,
    ) as exc:
        raise SystemExit(f"FATAL: {exc}") from exc
    print(
        json_line(
            {
                "output_dir": validation["artifact_dir"],
                "validation_json": str(Path(validation["artifact_dir"]) / VALIDATION_FILENAME),
                "coverage": validation["coverage"],
                "promotion_eligible": False,
                "score_claim": False,
            }
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
