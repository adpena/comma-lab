# SPDX-License-Identifier: MIT
"""Plan paired CPU/CUDA auth-eval harvests for DP1 procedural-codebook arms.

This module is deliberately planning-only. DP1's three WAVE-3 recipes train
and export byte-closed archive/runtime packets while forcing
``DPP_SKIP_AUTH_EVAL=1``; the score-bearing step is a separate paired Modal
auth-eval dispatch through ``tools/dispatch_modal_paired_auth_eval.py``.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable

try:
    import yaml  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover - exercised by environments, not tests
    yaml = None  # type: ignore[assignment]

from tac.deploy.modal.paired_dispatch import (
    PAIRED_AUTH_EVAL_DISPATCH_TOOL,
    paired_auth_eval_dispatch_command_template,
)
from tac.optimization.dp1_live_modal_status import build_dp1_modal_call_status

SCHEMA = "dp1_procedural_paired_harvest_plan_v1"
TOOL_PATH = "tools/plan_dp1_procedural_paired_harvest.py"
DEFAULT_OUTPUT_ROOT = "experiments/results/dp1_procedural_paired_harvest"
ADJUDICATION_TOOL_PATH = "tools/adjudicate_dp1_procedural_paired_harvest.py"

DP1_VARIANT_ORDER = ("baseline", "procedural", "null_control")
DP1_RECIPE_BASENAMES = {
    "baseline": (
        "substrate_pretrained_driving_prior_original_baseline_modal_t4_paired_dispatch"
    ),
    "procedural": (
        "substrate_pretrained_driving_prior_procedural_codebook_modal_t4_paired_dispatch"
    ),
    "null_control": (
        "substrate_pretrained_driving_prior_null_exploit_codebook_modal_t4_paired_dispatch"
    ),
}
DP1_RECIPE_PATHS = {
    key: f".omx/operator_authorize_recipes/{name}.yaml"
    for key, name in DP1_RECIPE_BASENAMES.items()
}

FALSE_AUTHORITY_FLAGS = (
    "score_claim",
    "score_claim_valid",
    "promotion_eligible",
    "rank_or_kill_eligible",
    "ready_for_exact_eval_dispatch",
)

FALSE_AUTHORITY_NUMERIC_SCORE_FIELDS = (
    "contest_cuda_score",
    "contest_cpu_score",
    "auth_eval_cuda_score",
    "auth_eval_cpu_score",
    "score",
    "score_recomputed_from_components",
)

_SAFE_TOKEN_RE = re.compile(r"[^A-Za-z0-9_.:-]+")


def _repo_rel(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _safe_slug(value: object) -> str:
    text = str(value or "").strip().lower().replace("/", "_")
    text = _SAFE_TOKEN_RE.sub("_", text).strip("_.:-")
    return text or "unknown"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _read_json(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def _read_recipe(path: Path) -> Mapping[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read operator-authorize recipes")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} did not contain a YAML mapping")
    return payload


def _json_false_authority_blockers(payload: Mapping[str, Any], *, source: str) -> list[str]:
    blockers: list[str] = []
    for key in FALSE_AUTHORITY_FLAGS:
        if payload.get(key) is True:
            blockers.append(f"{source}_{key}_true")
    for key in FALSE_AUTHORITY_NUMERIC_SCORE_FIELDS:
        value = payload.get(key)
        if isinstance(value, int | float) and not isinstance(value, bool):
            blockers.append(f"{source}_{key}_present")
    return blockers


def _modal_metadata_path(output_dir: Path) -> Path | None:
    for candidate in (
        output_dir / "modal_metadata.json",
        output_dir.parent / "modal_metadata.json",
    ):
        if candidate.is_file():
            return candidate
    return None


def _modal_metadata_summary(
    *,
    output_dir: Path,
    repo_root: Path,
) -> tuple[dict[str, Any], list[str], list[str]]:
    blockers: list[str] = []
    warnings: list[str] = []
    metadata_path = _modal_metadata_path(output_dir)
    summary: dict[str, Any] = {
        "path": str(metadata_path) if metadata_path else None,
        "exists": metadata_path is not None,
    }
    if metadata_path is None:
        blockers.append("modal_training_metadata_missing")
        return summary, blockers, warnings

    payload = _read_json(metadata_path)
    sentinels = payload.get("sentinel_files_local_sha256")
    if not isinstance(sentinels, Mapping) or not sentinels:
        blockers.append("modal_training_metadata_sentinel_hashes_missing")
        sentinels = {}
    summary.update(
        {
            "path": _repo_rel(metadata_path, repo_root),
            "metadata_schema": payload.get("metadata_schema"),
            "lane_id": payload.get("lane_id"),
            "label": payload.get("label"),
            "call_id": payload.get("call_id"),
            "mounted_code_git_head": payload.get("mounted_code_git_head"),
            "mounted_code_git_branch": payload.get("mounted_code_git_branch"),
            "require_clean_head": payload.get("require_clean_head"),
            "sentinel_files_local_sha256": {
                str(key): str(value) for key, value in sentinels.items()
            },
        }
    )
    if payload.get("working_tree_dirty") is True:
        warnings.append("modal_training_metadata_working_tree_dirty")
    return summary, blockers, warnings


def _paired_source_equivalence(
    candidates: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str], list[str]]:
    rows = {row.get("variant"): row for row in candidates}
    baseline = rows.get("baseline")
    procedural = rows.get("procedural")
    blockers: list[str] = []
    warnings: list[str] = []
    summary: dict[str, Any] = {
        "checked": False,
        "shared_sentinel_count": 0,
        "mismatched_sentinel_files": [],
    }
    if not isinstance(baseline, Mapping) or not isinstance(procedural, Mapping):
        blockers.append("paired_candidate_rows_missing")
        return summary, blockers, warnings

    baseline_meta = baseline.get("modal_training_metadata")
    procedural_meta = procedural.get("modal_training_metadata")
    if not isinstance(baseline_meta, Mapping) or not isinstance(procedural_meta, Mapping):
        blockers.append("paired_candidate_modal_metadata_missing")
        return summary, blockers, warnings

    baseline_sentinels = baseline_meta.get("sentinel_files_local_sha256")
    procedural_sentinels = procedural_meta.get("sentinel_files_local_sha256")
    if not isinstance(baseline_sentinels, Mapping) or not isinstance(
        procedural_sentinels, Mapping
    ):
        blockers.append("paired_candidate_sentinel_hash_maps_missing")
        return summary, blockers, warnings

    shared = sorted(set(baseline_sentinels) & set(procedural_sentinels))
    baseline_only = sorted(set(baseline_sentinels) - set(procedural_sentinels))
    procedural_only = sorted(set(procedural_sentinels) - set(baseline_sentinels))
    mismatches = [
        rel
        for rel in shared
        if baseline_sentinels.get(rel) != procedural_sentinels.get(rel)
    ]
    summary.update(
        {
            "checked": True,
            "shared_sentinel_count": len(shared),
            "baseline_only_sentinel_files": baseline_only,
            "procedural_only_sentinel_files": procedural_only,
            "mismatched_sentinel_files": mismatches,
            "baseline_mounted_code_git_head": baseline_meta.get(
                "mounted_code_git_head"
            ),
            "procedural_mounted_code_git_head": procedural_meta.get(
                "mounted_code_git_head"
            ),
        }
    )
    if not shared:
        blockers.append("paired_candidate_sentinel_files_no_overlap")
    if mismatches:
        blockers.append("paired_candidate_sentinel_sha256_mismatch")
    if baseline_only or procedural_only:
        warnings.append("paired_candidate_sentinel_file_sets_differ")
    if (
        baseline_meta.get("mounted_code_git_head")
        and procedural_meta.get("mounted_code_git_head")
        and baseline_meta.get("mounted_code_git_head")
        != procedural_meta.get("mounted_code_git_head")
        and not mismatches
    ):
        warnings.append("paired_candidate_git_heads_differ_sentinel_hashes_match")
    return summary, blockers, warnings


def _recipe_blockers(recipe: Mapping[str, Any], *, expected_name: str) -> list[str]:
    blockers: list[str] = []
    if recipe.get("name") != expected_name:
        blockers.append("recipe_name_mismatch")
    if recipe.get("score_claim") is not False:
        blockers.append("recipe_score_claim_not_false")
    if recipe.get("promotion_eligible") is not False:
        blockers.append("recipe_promotion_eligible_not_false")
    if recipe.get("ready_for_exact_eval_dispatch") is not False:
        blockers.append("recipe_ready_for_exact_eval_dispatch_not_false")
    paired_axis = recipe.get("paired_axis")
    if not isinstance(paired_axis, Mapping) or paired_axis.get("enabled") is not True:
        blockers.append("recipe_paired_axis_not_enabled")
    env = recipe.get("env_overrides")
    if not isinstance(env, Mapping):
        blockers.append("recipe_env_overrides_missing")
    elif str(env.get("DPP_SKIP_AUTH_EVAL")) != "1":
        blockers.append("recipe_DPP_SKIP_AUTH_EVAL_not_1")
    if str(recipe.get("remote_driver") or "") != (
        "scripts/remote_lane_substrate_pretrained_driving_prior.sh"
    ):
        blockers.append("recipe_remote_driver_unexpected")
    return blockers


def _archive_summary(archive_path: Path) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    summary: dict[str, Any] = {
        "path": str(archive_path),
        "exists": archive_path.is_file(),
    }
    if not archive_path.is_file():
        blockers.append("archive_zip_missing")
        return summary, blockers

    summary["bytes"] = archive_path.stat().st_size
    summary["sha256"] = _sha256_file(archive_path)
    try:
        with zipfile.ZipFile(archive_path) as zf:
            names = zf.namelist()
            summary["zip_members"] = names
            if not ({"0.bin", "x"} & set(names)):
                blockers.append("archive_zip_missing_0bin_or_x_member")
    except zipfile.BadZipFile:
        blockers.append("archive_zip_bad_zip")
    return summary, blockers


def _runtime_summary(submission_dir: Path) -> tuple[dict[str, Any], list[str]]:
    blockers: list[str] = []
    inflate_sh = submission_dir / "inflate.sh"
    summary: dict[str, Any] = {
        "submission_dir": str(submission_dir),
        "submission_dir_exists": submission_dir.is_dir(),
        "inflate_sh": str(inflate_sh),
        "inflate_sh_exists": inflate_sh.is_file(),
    }
    if not submission_dir.is_dir():
        blockers.append("submission_dir_missing")
        return summary, blockers
    if not inflate_sh.is_file():
        blockers.append("submission_inflate_sh_missing")
    else:
        mode = inflate_sh.stat().st_mode
        summary["inflate_sh_executable"] = bool(mode & stat.S_IXUSR)
        if not os.access(inflate_sh, os.X_OK):
            blockers.append("submission_inflate_sh_not_executable")
    if not (submission_dir / "inflate.py").is_file():
        blockers.append("submission_inflate_py_missing")
    return summary, blockers


def _candidate_output_summary(
    *,
    variant: str,
    recipe: Mapping[str, Any],
    output_dir: Path | None,
    output_root: str,
    repo_root: Path,
) -> dict[str, Any]:
    recipe_lane_id = str(recipe.get("lane_id") or "")
    recipe_name = str(recipe.get("name") or "")
    blockers: list[str] = []
    warnings: list[str] = []

    if output_dir is None:
        blockers.append("candidate_output_dir_not_supplied")
        return {
            "variant": variant,
            "recipe_name": recipe_name,
            "recipe_lane_id": recipe_lane_id,
            "recipe_dispatch_enabled": recipe.get("dispatch_enabled"),
            "output_dir": None,
            "paired_dispatch_tool": PAIRED_AUTH_EVAL_DISPATCH_TOOL,
            "paired_dispatch_plan_command": None,
            "paired_dispatch_execute_command": None,
            "paired_dispatch_output_dirs": {},
            "harvest_commands": {},
            "status": "blocked",
            "blockers": blockers,
            "warnings": warnings,
            "score_claim": False,
            "promotion_eligible": False,
            "dispatch_attempted": False,
        }

    output_dir = output_dir.resolve()
    if not output_dir.is_dir():
        blockers.append("candidate_output_dir_missing")

    archive, archive_blockers = _archive_summary(output_dir / "archive.zip")
    blockers.extend(archive_blockers)
    runtime, runtime_blockers = _runtime_summary(output_dir / "submission")
    blockers.extend(runtime_blockers)
    metadata, metadata_blockers, metadata_warnings = _modal_metadata_summary(
        output_dir=output_dir,
        repo_root=repo_root,
    )
    blockers.extend(metadata_blockers)
    warnings.extend(metadata_warnings)

    manifest_path = output_dir / "manifest.json"
    provenance_path = output_dir / "provenance.json"
    procedural_variant_path = output_dir / "procedural_variant_provenance.json"
    if not manifest_path.is_file():
        blockers.append("manifest_json_missing")
        manifest: Mapping[str, Any] = {}
    else:
        manifest = _read_json(manifest_path)
        blockers.extend(_json_false_authority_blockers(manifest, source="manifest"))
    if not provenance_path.is_file():
        blockers.append("provenance_json_missing")
        provenance: Mapping[str, Any] = {}
    else:
        provenance = _read_json(provenance_path)
        blockers.extend(_json_false_authority_blockers(provenance, source="provenance"))

    provenance_lane = str(provenance.get("lane_id") or "")
    if recipe_lane_id and provenance_lane and recipe_lane_id != provenance_lane:
        blockers.append("provenance_lane_id_mismatch")

    if variant in {"procedural", "null_control"}:
        if not procedural_variant_path.is_file():
            blockers.append("procedural_variant_provenance_missing")
        else:
            procedural_variant = _read_json(procedural_variant_path)
            blockers.extend(
                _json_false_authority_blockers(
                    procedural_variant, source="procedural_variant"
                )
            )
            null_control = bool(procedural_variant.get("null_exploit_control"))
            if variant == "procedural" and null_control:
                blockers.append("procedural_variant_marked_null_control")
            if variant == "null_control" and not null_control:
                blockers.append("null_control_variant_not_marked_null_control")
    else:
        if procedural_variant_path.exists():
            warnings.append("baseline_has_unexpected_procedural_variant_provenance")

    command_plan: list[str] | None = None
    command_execute: list[str] | None = None
    output_dirs: dict[str, str] = {}
    if not blockers:
        archive_sha256 = str(archive["sha256"])
        lane_id_base = f"{recipe_lane_id}_exact_eval"
        run_id = f"dp1_{_safe_slug(variant)}_paired_auth_eval_{archive_sha256[:12]}"
        variant_output_root = f"{output_root}/{_safe_slug(variant)}"
        command_plan = paired_auth_eval_dispatch_command_template(
            archive_path=_repo_rel(output_dir / "archive.zip", repo_root),
            submission_dir=_repo_rel(output_dir / "submission", repo_root),
            lane_id_base=lane_id_base,
            archive_sha256=archive_sha256,
            execute=False,
            label=f"dp1_{variant}",
            run_id=run_id,
            inflate_sh="inflate.sh",
            output_root=variant_output_root,
            gpu="T4",
            claim_agent="codex:dp1_procedural_paired_harvest",
            claim_notes=(
                "dp1 procedural-codebook paired harvest; "
                f"variant={variant}; recipe={recipe_name}"
            ),
        )
        command_plan.extend(
            ["--json-out", f"{variant_output_root}/paired_dispatch_plan.json"]
        )
        command_execute = paired_auth_eval_dispatch_command_template(
            archive_path=_repo_rel(output_dir / "archive.zip", repo_root),
            submission_dir=_repo_rel(output_dir / "submission", repo_root),
            lane_id_base=lane_id_base,
            archive_sha256=archive_sha256,
            execute=True,
            label=f"dp1_{variant}",
            run_id=run_id,
            inflate_sh="inflate.sh",
            output_root=variant_output_root,
            gpu="T4",
            claim_agent="codex:dp1_procedural_paired_harvest",
            claim_notes=(
                "dp1 procedural-codebook paired harvest; "
                f"variant={variant}; recipe={recipe_name}"
            ),
        )
        command_execute.extend(
            ["--json-out", f"{variant_output_root}/paired_dispatch_plan.json"]
        )
        output_dirs = {
            "contest_cuda": (
                f"{variant_output_root}/modal_auth_eval/{run_id}_cuda"
            ),
            "contest_cpu": (
                f"{variant_output_root}/modal_auth_eval_cpu/{run_id}_cpu"
            ),
        }

    return {
        "variant": variant,
        "recipe_name": recipe_name,
        "recipe_lane_id": recipe_lane_id,
        "recipe_dispatch_enabled": recipe.get("dispatch_enabled"),
        "output_dir": str(output_dir),
        "archive": archive,
        "runtime": runtime,
        "modal_training_metadata": metadata,
        "manifest": {
            "path": str(manifest_path),
            "exists": manifest_path.is_file(),
            "score_claim": manifest.get("score_claim"),
            "score_claim_valid": manifest.get("score_claim_valid"),
            "promotion_eligible": manifest.get("promotion_eligible"),
            "ready_for_exact_eval_dispatch": manifest.get(
                "ready_for_exact_eval_dispatch"
            ),
        },
        "provenance": {
            "path": str(provenance_path),
            "exists": provenance_path.is_file(),
            "score_claim": provenance.get("score_claim"),
            "promotion_eligible": provenance.get("promotion_eligible"),
            "ready_for_exact_eval_dispatch": provenance.get(
                "ready_for_exact_eval_dispatch"
            ),
            "lane_id": provenance.get("lane_id"),
        },
        "paired_dispatch_tool": PAIRED_AUTH_EVAL_DISPATCH_TOOL,
        "paired_dispatch_plan_command": command_plan,
        "paired_dispatch_execute_command": command_execute,
        "paired_dispatch_output_dirs": output_dirs,
        "harvest_commands": {
            axis: f".venv/bin/python tools/recover_modal_auth_eval.py --output-dir {path}"
            for axis, path in output_dirs.items()
        },
        "status": "ready_for_paired_dispatch_plan" if not blockers else "blocked",
        "blockers": sorted(set(blockers)),
        "warnings": sorted(set(warnings)),
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "dispatch_attempted": False,
    }


def build_dp1_procedural_paired_harvest_plan(
    *,
    output_dirs: Mapping[str, str | Path | None],
    recipe_paths: Mapping[str, str | Path] | None = None,
    training_metadata_paths: Mapping[str, str | Path | None] | None = None,
    poll_training_calls: bool = False,
    modal_poll_timeout_seconds: float = 2.0,
    function_call_from_id: Callable[[str], Any] | None = None,
    repo_root: str | Path = ".",
    output_root: str = DEFAULT_OUTPUT_ROOT,
    include_null_control: bool = False,
) -> dict[str, Any]:
    """Return a false-authority-safe paired auth-eval harvest plan."""

    root = Path(repo_root).resolve()
    recipe_paths = recipe_paths or DP1_RECIPE_PATHS
    training_metadata_paths = training_metadata_paths or {}
    variants = ["baseline", "procedural"]
    if include_null_control or output_dirs.get("null_control"):
        variants.append("null_control")

    candidates: list[dict[str, Any]] = []
    top_blockers: list[str] = []
    training_call_status: dict[str, Any] | None = None
    if poll_training_calls:
        baseline_training_metadata = training_metadata_paths.get("baseline")
        procedural_training_metadata = training_metadata_paths.get("procedural")
        if not baseline_training_metadata or not procedural_training_metadata:
            top_blockers.append("dp1_training_metadata_paths_missing")
        else:
            training_call_status = build_dp1_modal_call_status(
                baseline_metadata=root / str(baseline_training_metadata),
                procedural_metadata=root / str(procedural_training_metadata),
                repo_root=root,
                timeout_seconds=modal_poll_timeout_seconds,
                function_call_from_id=function_call_from_id,
            )
            if training_call_status.get("status") == "needs_attention":
                top_blockers.append("dp1_training_calls_need_attention")
                top_blockers.extend(
                    f"training_call_{blocker}"
                    for blocker in training_call_status.get("blockers", [])
                )
            elif not training_call_status.get("ready_for_training_harvest"):
                top_blockers.append("dp1_training_calls_not_ready_for_harvest")
    for variant in variants:
        recipe_path = root / str(recipe_paths[variant])
        expected_name = DP1_RECIPE_BASENAMES[variant]
        if not recipe_path.is_file():
            top_blockers.append(f"{variant}_recipe_missing")
            recipe: Mapping[str, Any] = {
                "name": expected_name,
                "lane_id": "",
            }
            recipe_blockers = ["recipe_missing"]
        else:
            recipe = _read_recipe(recipe_path)
            recipe_blockers = _recipe_blockers(recipe, expected_name=expected_name)

        output_value = output_dirs.get(variant)
        output_path = Path(output_value) if output_value else None
        row = _candidate_output_summary(
            variant=variant,
            recipe=recipe,
            output_dir=output_path,
            output_root=output_root,
            repo_root=root,
        )
        if recipe_blockers:
            row["blockers"] = sorted(set(row["blockers"]) | set(recipe_blockers))
            row["status"] = "blocked"
        row["recipe_path"] = str(recipe_path)
        candidates.append(row)

    required_ready = {
        row["variant"]: row["status"] == "ready_for_paired_dispatch_plan"
        for row in candidates
        if row["variant"] in {"baseline", "procedural"}
    }
    paired_source_equivalence, paired_source_blockers, paired_source_warnings = (
        _paired_source_equivalence(candidates)
    )
    top_blockers.extend(paired_source_blockers)
    for row in candidates:
        if row["variant"] in {"baseline", "procedural"}:
            row["warnings"] = sorted(
                set(row.get("warnings") or []) | set(paired_source_warnings)
            )
    all_required_ready = all(
        required_ready.get(v, False) for v in ("baseline", "procedural")
    ) and not paired_source_blockers
    if not all_required_ready:
        top_blockers.append("baseline_and_procedural_paired_harvest_not_ready")

    rows_by_variant = {row["variant"]: row for row in candidates}
    post_harvest_adjudication_command: list[str] | None = None
    if all_required_ready:
        baseline = rows_by_variant["baseline"]
        procedural = rows_by_variant["procedural"]
        baseline_axes = baseline["paired_dispatch_output_dirs"]
        procedural_axes = procedural["paired_dispatch_output_dirs"]
        post_harvest_adjudication_command = [
            ".venv/bin/python",
            ADJUDICATION_TOOL_PATH,
            "--baseline-output-dir",
            str(baseline["output_dir"]),
            "--procedural-output-dir",
            str(procedural["output_dir"]),
            "--baseline-cpu-dir",
            str(baseline_axes["contest_cpu"]),
            "--baseline-cuda-dir",
            str(baseline_axes["contest_cuda"]),
            "--procedural-cpu-dir",
            str(procedural_axes["contest_cpu"]),
            "--procedural-cuda-dir",
            str(procedural_axes["contest_cuda"]),
            "--json-out",
            f"{output_root}/adjudication/dp1_procedural_paired_adjudication.json",
            "--md-out",
            f"{output_root}/adjudication/dp1_procedural_paired_adjudication.md",
        ]

    return {
        "schema": SCHEMA,
        "tool": TOOL_PATH,
        "adjudication_tool": ADJUDICATION_TOOL_PATH,
        "repo_root": str(root),
        "paired_dispatch_tool": PAIRED_AUTH_EVAL_DISPATCH_TOOL,
        "required_axes": ["contest_cpu", "contest_cuda"],
        "planning_only": True,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "adjudication_required": True,
        "all_required_candidates_ready": all_required_ready,
        "top_blockers": sorted(set(top_blockers)),
        "training_call_status": training_call_status,
        "candidates": candidates,
        "paired_source_equivalence": paired_source_equivalence,
        "post_harvest_adjudication_command": post_harvest_adjudication_command,
        "notes": [
            "DP1 training recipes force DPP_SKIP_AUTH_EVAL=1; score-bearing evidence starts only after this paired Modal auth-eval plan is executed and harvested.",
            "The optional null_control arm is a disambiguator; baseline and procedural are the required first-anchor pair.",
            "Per-axis commands are intentionally routed through tools/dispatch_modal_paired_auth_eval.py, never through single-axis Modal wrappers.",
            "Run the post-harvest adjudication command only after both variants have recovered CPU and CUDA auth-eval JSON.",
            "Baseline/procedural paired readiness requires matching shared Modal sentinel-file hashes; git heads may differ only when those mounted bytes match.",
            "If training_call_status is present, paired auth eval remains blocked until both training arms are ready for training harvest.",
        ],
    }


def render_markdown(plan: Mapping[str, Any]) -> str:
    rows = [
        "# DP1 Procedural Paired-Harvest Plan",
        "",
        f"- Schema: `{plan.get('schema')}`",
        f"- Paired dispatch tool: `{plan.get('paired_dispatch_tool')}`",
        f"- Required candidates ready: `{plan.get('all_required_candidates_ready')}`",
        f"- Top blockers: `{', '.join(plan.get('top_blockers') or []) or 'none'}`",
    ]
    training_call_status = plan.get("training_call_status")
    if isinstance(training_call_status, Mapping):
        baseline = (
            training_call_status.get("baseline")
            if isinstance(training_call_status.get("baseline"), Mapping)
            else {}
        )
        procedural = (
            training_call_status.get("procedural")
            if isinstance(training_call_status.get("procedural"), Mapping)
            else {}
        )
        baseline_poll = (
            baseline.get("poll") if isinstance(baseline.get("poll"), Mapping) else {}
        )
        procedural_poll = (
            procedural.get("poll")
            if isinstance(procedural.get("poll"), Mapping)
            else {}
        )
        rows.extend(
            [
                f"- Training-call status: `{training_call_status.get('status')}`",
                f"- Training harvest ready: `{training_call_status.get('ready_for_training_harvest')}`",
                f"- Baseline training call: `{baseline.get('call_id')}` / `{baseline_poll.get('status')}`",
                f"- Procedural training call: `{procedural.get('call_id')}` / `{procedural_poll.get('status')}`",
            ]
        )
    rows.extend(
        [
            "",
            "| variant | status | archive bytes | archive sha256 | blockers |",
            "|---|---|---:|---|---|",
        ]
    )
    for row in plan.get("candidates", []):
        if not isinstance(row, Mapping):
            continue
        archive = row.get("archive") if isinstance(row.get("archive"), Mapping) else {}
        blockers = ", ".join(row.get("blockers") or []) or "none"
        rows.append(
            "| {variant} | {status} | {bytes} | {sha} | {blockers} |".format(
                variant=row.get("variant"),
                status=row.get("status"),
                bytes=archive.get("bytes", ""),
                sha=str(archive.get("sha256") or "")[:16],
                blockers=blockers,
            )
        )
    rows.extend(
        [
            "",
            "## Commands",
            "",
            "Commands below are generated only for rows with complete byte custody.",
        ]
    )
    for row in plan.get("candidates", []):
        if not isinstance(row, Mapping):
            continue
        cmd = row.get("paired_dispatch_plan_command")
        if not cmd:
            continue
        rows.extend(
            [
                "",
                f"### {row.get('variant')}",
                "",
                "```bash",
                " ".join(str(part) for part in cmd),
                "```",
            ]
        )
    adjudication_cmd = plan.get("post_harvest_adjudication_command")
    if adjudication_cmd:
        rows.extend(
            [
                "",
                "### post-harvest adjudication",
                "",
                "```bash",
                " ".join(str(part) for part in adjudication_cmd),
                "```",
            ]
        )
    return "\n".join(rows) + "\n"


__all__ = [
    "DEFAULT_OUTPUT_ROOT",
    "DP1_RECIPE_PATHS",
    "SCHEMA",
    "TOOL_PATH",
    "ADJUDICATION_TOOL_PATH",
    "build_dp1_procedural_paired_harvest_plan",
    "render_markdown",
]
