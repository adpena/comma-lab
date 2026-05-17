# SPDX-License-Identifier: MIT
"""Harvest TT5L side-info effect-curve cells from a paired-axis plan.

The Lightning paired-axis plan is the identity source of truth for TT5L
side-info effect-curve cells. This module converts that plan plus local
``contest_auth_eval`` artifacts into builder-ready cells without allowing a
manual harvest step to drop or remix ``pair_group_id`` / ``run_id``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from tac.optimization.l5_v2_measurement_schedule import (
    L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES,
    L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS,
)
from tac.optimization.l5_v2_probe_intake import (
    exact_eval_evidence_from_auth_eval_artifact,
)

L5V2_TT5L_SIDEINFO_EFFECT_CURVE_HARVEST_CELLS_SCHEMA = (
    "l5_v2_tt5l_sideinfo_effect_curve_harvest_cells_v1"
)
L5V2_TT5L_SIDEINFO_EFFECT_CURVE_HARVEST_CELLS_TOOL_PATH = (
    "tools/build_l5_v2_tt5l_sideinfo_effect_curve_cells_from_lightning_plan.py"
)

_FALSE_AUTHORITY_FLAGS = {
    "score_claim": False,
    "promotion_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "rank_or_kill_eligible": False,
}
_AUTH_EVAL_FILENAMES = (
    "contest_auth_eval.adjudicated.json",
    "contest_auth_eval.json",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_repo_path(path: str | Path, repo_root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repo_root / candidate


def _repo_relative(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _read_json_mapping(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must be an object: {path}")
    return payload


def _mapping_rows(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, Iterable) or isinstance(value, (str, bytes, Mapping)):
        return []
    return [row for row in value if isinstance(row, Mapping)]


def _variant_rows_by_name(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {
        str(row.get("variant") or ""): row
        for row in _mapping_rows(manifest.get("variants"))
        if str(row.get("variant") or "")
    }


def _variant_liveness_by_name(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for variant, row in _variant_rows_by_name(manifest).items():
        liveness = row.get("sideinfo_liveness")
        if isinstance(liveness, Mapping):
            out[variant] = dict(liveness)
    return out


def _source_variant_manifest(
    *,
    plan: Mapping[str, Any],
    plan_path: Path,
    repo_root: Path,
) -> tuple[dict[str, Any], str, str]:
    raw_path = str(plan.get("source_variant_manifest") or "").strip()
    if not raw_path:
        return {}, "", ""
    resolved = _resolve_repo_path(raw_path, repo_root)
    if not resolved.is_file():
        return {}, _repo_relative(resolved, repo_root), ""
    manifest = _read_json_mapping(resolved)
    recorded_sha = str(plan.get("source_variant_manifest_sha256") or "").strip()
    actual_sha = _sha256_file(resolved)
    if recorded_sha and recorded_sha != actual_sha:
        manifest = {
            **manifest,
            "_harvest_blocker": "source_variant_manifest_sha_mismatch",
        }
    return manifest, _repo_relative(resolved, repo_root), actual_sha


def _planned_cells(plan: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    cells = _mapping_rows(plan.get("cells"))
    return [
        cell
        for cell in cells
        if str(cell.get("axis") or "") in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES
        and str(cell.get("variant") or "") in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS
    ]


def _auth_eval_artifact_path(cell: Mapping[str, Any], *, repo_root: Path) -> Path | None:
    raw_dir = str(cell.get("local_artifact_dir") or "").strip()
    if not raw_dir:
        return None
    local_dir = _resolve_repo_path(raw_dir, repo_root)
    for filename in _AUTH_EVAL_FILENAMES:
        candidate = local_dir / filename
        if candidate.is_file():
            return candidate
    return None


def _plan_cell_identity(cell: Mapping[str, Any]) -> dict[str, str]:
    queue = cell.get("spec")
    queue_metadata = {}
    if isinstance(queue, Mapping) and isinstance(queue.get("queue_metadata"), Mapping):
        queue_metadata = dict(queue["queue_metadata"])
    return {
        "axis": str(cell.get("axis") or queue_metadata.get("axis") or "").strip(),
        "variant": str(cell.get("variant") or queue_metadata.get("variant") or "").strip(),
        "archive_sha256": str(
            cell.get("archive_sha256") or queue_metadata.get("archive_sha256") or ""
        ).strip(),
        "pair_group_id": str(
            cell.get("pair_group_id") or queue_metadata.get("pair_group_id") or ""
        ).strip(),
        "run_id": str(cell.get("run_id") or queue_metadata.get("run_id") or "").strip(),
    }


def _missing_evidence(
    *,
    identity: Mapping[str, str],
    cell: Mapping[str, Any],
) -> dict[str, Any]:
    local_dir = str(cell.get("local_artifact_dir") or "").strip()
    artifact_path = str(Path(local_dir) / "contest_auth_eval.json") if local_dir else ""
    return {
        "axis": identity["axis"],
        "archive_sha256": identity["archive_sha256"],
        "pair_group_id": identity["pair_group_id"],
        "run_id": identity["run_id"],
        "artifact_path": artifact_path,
    }


def _cell_blockers(
    *,
    identity: Mapping[str, str],
    cell: Mapping[str, Any],
    artifact_path: Path | None,
    liveness: Mapping[str, Any] | None,
    manifest_blocker: str,
) -> list[str]:
    blockers: list[str] = []
    for key in ("axis", "variant", "archive_sha256", "pair_group_id", "run_id"):
        if not identity.get(key):
            blockers.append(f"plan_cell_{key}_missing")
    spec = cell.get("spec")
    queue_metadata = (
        spec.get("queue_metadata")
        if isinstance(spec, Mapping) and isinstance(spec.get("queue_metadata"), Mapping)
        else {}
    )
    for key in ("axis", "variant", "archive_sha256", "pair_group_id", "run_id"):
        expected = identity.get(key, "")
        observed = str(queue_metadata.get(key) or "").strip()
        if observed and expected and observed != expected:
            blockers.append(f"plan_cell_queue_metadata_{key}_mismatch")
    if artifact_path is None:
        blockers.append(
            "harvested_exact_eval_artifact_missing:"
            f"{identity.get('variant') or '<missing>'}:"
            f"{identity.get('axis') or '<missing>'}"
        )
    if liveness is None:
        blockers.append(
            "source_variant_sideinfo_liveness_missing:"
            f"{identity.get('variant') or '<missing>'}"
        )
    if manifest_blocker:
        blockers.append(manifest_blocker)
    return list(dict.fromkeys(blockers))


def build_l5_v2_tt5l_sideinfo_effect_curve_cells_from_lightning_plan(
    *,
    plan: Mapping[str, Any],
    plan_path: str | Path,
    repo_root: str | Path,
) -> dict[str, Any]:
    """Return builder-ready effect-curve cells from a Lightning plan.

    Missing harvested artifacts are preserved as blocked cells, not dropped,
    so the aggregate effect-curve builder can report the exact missing
    ``variant``/``axis`` cells without weakening the 5x2 paired requirement.
    """

    root = Path(repo_root).resolve()
    resolved_plan = _resolve_repo_path(plan_path, root)
    plan_sha = _sha256_file(resolved_plan) if resolved_plan.is_file() else ""
    variant_manifest, variant_manifest_path, variant_manifest_sha = (
        _source_variant_manifest(plan=plan, plan_path=resolved_plan, repo_root=root)
    )
    manifest_blocker = str(variant_manifest.get("_harvest_blocker") or "")
    liveness_by_variant = _variant_liveness_by_name(variant_manifest)
    cells: list[dict[str, Any]] = []
    blockers: list[str] = []
    artifact_count = 0

    for cell in _planned_cells(plan):
        identity = _plan_cell_identity(cell)
        variant = identity["variant"]
        axis = identity["axis"]
        artifact_path = _auth_eval_artifact_path(cell, repo_root=root)
        liveness = liveness_by_variant.get(variant)
        cell_blockers = _cell_blockers(
            identity=identity,
            cell=cell,
            artifact_path=artifact_path,
            liveness=liveness,
            manifest_blocker=manifest_blocker,
        )
        if artifact_path is not None:
            evidence = exact_eval_evidence_from_auth_eval_artifact(
                artifact_path,
                axis=axis,
                repo_root=root,
                source_metadata={
                    "pair_group_id": identity["pair_group_id"],
                    "run_id": identity["run_id"],
                    "source_lightning_paired_axis_plan_path": _repo_relative(
                        resolved_plan,
                        root,
                    ),
                },
            )
            artifact_count += 1
            if evidence is None:
                evidence = _missing_evidence(identity=identity, cell=cell)
                cell_blockers.append(
                    f"harvested_exact_eval_artifact_unreadable:{variant}:{axis}"
                )
        else:
            evidence = _missing_evidence(identity=identity, cell=cell)

        cells.append(
            {
                **_FALSE_AUTHORITY_FLAGS,
                "axis": axis,
                "variant": variant,
                "archive_sha256": identity["archive_sha256"],
                "pair_group_id": identity["pair_group_id"],
                "run_id": identity["run_id"],
                "source_plan_cell": {
                    "local_artifact_dir": str(cell.get("local_artifact_dir") or ""),
                    "state_path": str(cell.get("state_path") or ""),
                    "job_name": str(cell.get("job_name") or ""),
                    "command_sha256": str(cell.get("command_sha256") or ""),
                },
                "source_variant_manifest": variant_manifest_path,
                "sideinfo_liveness": dict(liveness or {}),
                "evidence": evidence,
                "blockers": list(dict.fromkeys(cell_blockers)),
            }
        )
        blockers.extend(cell_blockers)

    missing_expected = sorted(
        {
            f"{axis}:{variant}"
            for axis in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES
            for variant in L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS
        }
        - {f"{row['axis']}:{row['variant']}" for row in cells}
    )
    blockers.extend(f"planned_cell_missing:{cell}" for cell in missing_expected)

    return {
        **_FALSE_AUTHORITY_FLAGS,
        "schema": L5V2_TT5L_SIDEINFO_EFFECT_CURVE_HARVEST_CELLS_SCHEMA,
        "tool": L5V2_TT5L_SIDEINFO_EFFECT_CURVE_HARVEST_CELLS_TOOL_PATH,
        "source_plan": _repo_relative(resolved_plan, root),
        "source_plan_sha256": plan_sha,
        "source_variant_manifest": variant_manifest_path,
        "source_variant_manifest_sha256": variant_manifest_sha,
        "source_plan_schema": str(plan.get("schema") or ""),
        "required_axes": list(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_AXES),
        "required_variants": list(L5V2_SIDEINFO_EFFECT_CURVE_REQUIRED_VARIANTS),
        "cell_count": len(cells),
        "harvested_exact_eval_artifact_count": artifact_count,
        "missing_exact_eval_artifact_count": max(0, len(cells) - artifact_count),
        "cells": cells,
        "observed_cells": cells,
        "sideinfo_effect_curve_builder_tool": "tools/build_l5_v2_sideinfo_effect_curve.py",
        "blockers": list(dict.fromkeys(blockers)),
    }


def l5_v2_tt5l_sideinfo_effect_curve_harvest_cells_json(
    payload: Mapping[str, Any],
) -> str:
    """Return deterministic harvest-cell JSON text."""

    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


__all__ = [
    "L5V2_TT5L_SIDEINFO_EFFECT_CURVE_HARVEST_CELLS_SCHEMA",
    "L5V2_TT5L_SIDEINFO_EFFECT_CURVE_HARVEST_CELLS_TOOL_PATH",
    "build_l5_v2_tt5l_sideinfo_effect_curve_cells_from_lightning_plan",
    "l5_v2_tt5l_sideinfo_effect_curve_harvest_cells_json",
]
