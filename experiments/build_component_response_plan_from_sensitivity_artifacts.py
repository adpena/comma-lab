#!/usr/bin/env python3
"""Build an official component-response plan from CUDA sensitivity artifacts.

This is a deterministic glue tool for the post-harvest path:

1. Validate a harvested Lightning diagnostic component-sensitivity artifact
   directory.
2. Project the CUDA sensitivity maps onto the perturbation basis before any
   official response evaluation is run.
3. Build deterministic archive perturbation variants and a response plan with
   structured pre-response predictions attached.

The output is still not score evidence. It prepares the next official CUDA
``archive.zip -> inflate.sh -> upstream/evaluate.py`` response job.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    _bootstrap_path = Path(__file__).resolve().parents[1] / "tools" / "tool_bootstrap.py"
    _spec = importlib.util.spec_from_file_location("tool_bootstrap", _bootstrap_path)
    if _spec is None or _spec.loader is None:
        raise
    _tool_bootstrap = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_tool_bootstrap)
    ensure_repo_imports = _tool_bootstrap.ensure_repo_imports
    repo_root_from_tool = _tool_bootstrap.repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)
EXPERIMENTS_DIR = Path(__file__).resolve().parent
if str(EXPERIMENTS_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_DIR))

from build_component_response_perturbation_plan import (
    _load_basis_atoms,
    build_component_response_perturbation_plan,
)
from build_component_response_prediction_deltas import (
    build_component_response_prediction_deltas,
)

from tac.deploy.lightning.batch_jobs import (
    validate_local_component_sensitivity_artifact_dir,
)
from tac.repo_io import json_text, read_json, sha256_file, write_json

PRODUCER = "experiments/build_component_response_plan_from_sensitivity_artifacts.py"
SUMMARY_FORMAT = "official_component_response_plan_from_sensitivity_artifacts_summary_v1"
DEFAULT_OFFICIAL_EPSILONS = (-2.0, -1.0, 0.0, 1.0, 2.0)
COMPONENTS = ("posenet", "segnet", "combined")
MERGED_SHARD_VALIDATION_FORMAT = "component_sensitivity_shard_merge_validation_v1"
DIRECT_FD_SOURCE = "direct_renderer_cuda_finite_difference_component_response"


class ComponentResponsePlanFromSensitivityError(ValueError):
    """Raised when harvested sensitivity artifacts cannot build a safe plan."""


def _file_meta(path: Path) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise ComponentResponsePlanFromSensitivityError(f"required file not found: {path}")
    return {
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    write_json(path, payload)


def _finite_epsilons(values: list[float]) -> list[float]:
    if not values:
        raise ComponentResponsePlanFromSensitivityError("epsilon ladder must not be empty")
    out = sorted({float(value) for value in values})
    for index, value in enumerate(out):
        if not math.isfinite(value):
            raise ComponentResponsePlanFromSensitivityError(
                f"epsilon value {index} must be finite"
            )
    if not any(abs(value) > 1e-12 for value in out):
        raise ComponentResponsePlanFromSensitivityError(
            "epsilon ladder needs at least one nonzero point"
        )
    if not any(abs(value) <= 1e-12 for value in out):
        out.insert(0, 0.0)
        out = sorted(set(out))
    return out


def _required_artifact_file(artifact_dir: Path, name: str) -> Path:
    path = (artifact_dir / name).resolve()
    if not path.is_file():
        raise ComponentResponsePlanFromSensitivityError(
            f"sensitivity artifact missing required file: {path}"
        )
    return path


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = read_json(path)
    except json.JSONDecodeError as exc:
        raise ComponentResponsePlanFromSensitivityError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ComponentResponsePlanFromSensitivityError(f"{label} must be a JSON object: {path}")
    return payload


def _source_shard_dir(path_value: Any) -> Path:
    if not isinstance(path_value, str) or not path_value:
        raise ComponentResponsePlanFromSensitivityError(
            f"merged sensitivity source shard path is invalid: {path_value!r}"
        )
    path = Path(path_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _require_false(payload: Mapping[str, Any], *, field: str, label: str) -> None:
    if payload.get(field) is not False:
        raise ComponentResponsePlanFromSensitivityError(
            f"{label} must record {field}=false"
        )


def _validate_source_shard_custody(
    source_dir: Path,
    *,
    expected_baseline_archive_sha256: str,
    expected_baseline_archive_size_bytes: int,
) -> dict[str, Any]:
    inputs = _load_json_object(
        source_dir / "diagnostic_component_sensitivity_inputs.json",
        label=f"{source_dir} diagnostic component-sensitivity inputs",
    )
    baseline = inputs.get("baseline_archive")
    if not isinstance(baseline, dict):
        raise ComponentResponsePlanFromSensitivityError(
            f"{source_dir}: diagnostic inputs missing baseline_archive"
        )
    if baseline.get("sha256") != expected_baseline_archive_sha256:
        raise ComponentResponsePlanFromSensitivityError(
            f"{source_dir}: baseline archive sha256 mismatch: "
            f"{baseline.get('sha256')!r} != {expected_baseline_archive_sha256!r}"
        )
    if baseline.get("bytes") != expected_baseline_archive_size_bytes:
        raise ComponentResponsePlanFromSensitivityError(
            f"{source_dir}: baseline archive bytes mismatch: "
            f"{baseline.get('bytes')!r} != {expected_baseline_archive_size_bytes!r}"
        )
    _require_false(inputs, field="score_claim", label=f"{source_dir} inputs")
    _require_false(inputs, field="promotion_eligible", label=f"{source_dir} inputs")

    summary = _load_json_object(
        source_dir / "component_sensitivity_profile_summary.json",
        label=f"{source_dir} component sensitivity summary",
    )
    if summary.get("device") != "cuda":
        raise ComponentResponsePlanFromSensitivityError(
            f"{source_dir}: source shard device={summary.get('device')!r}, expected 'cuda'"
        )
    if summary.get("sensitivity_source") != DIRECT_FD_SOURCE:
        raise ComponentResponsePlanFromSensitivityError(
            f"{source_dir}: source shard sensitivity_source={summary.get('sensitivity_source')!r}"
        )
    for field in ("score_claim", "promotion_eligible", "official_component_response", "canonical_scorer_path"):
        _require_false(summary, field=field, label=f"{source_dir} summary")
    if summary.get("n_pairs_total") != 600:
        raise ComponentResponsePlanFromSensitivityError(
            f"{source_dir}: source shard n_pairs_total={summary.get('n_pairs_total')!r}, expected 600"
        )
    shard = summary.get("finite_difference_shard")
    if not isinstance(shard, dict) or shard.get("is_shard") is not True:
        raise ComponentResponsePlanFromSensitivityError(
            f"{source_dir}: source summary must be a finite-difference shard"
        )
    return {
        "path": str(source_dir),
        "shard_index": shard.get("shard_index"),
        "assigned_channel_count": shard.get("assigned_channel_count"),
        "assigned_channel_sha256": shard.get("assigned_channel_sha256"),
    }


def _validate_merged_component_sensitivity_artifact_dir(
    artifact_dir: Path,
    *,
    expected_baseline_archive_sha256: str,
    expected_baseline_archive_size_bytes: int,
) -> dict[str, Any]:
    summary = _load_json_object(
        artifact_dir / "component_sensitivity_profile_summary.json",
        label="merged component-sensitivity profile summary",
    )
    validation = _load_json_object(
        artifact_dir / "component_sensitivity_shard_merge_validation.json",
        label="component sensitivity shard merge validation",
    )
    if validation.get("format") != MERGED_SHARD_VALIDATION_FORMAT:
        raise ComponentResponsePlanFromSensitivityError(
            "merged sensitivity validation has unexpected format: "
            f"{validation.get('format')!r}"
        )
    if summary.get("merge_tool") != "experiments/merge_component_sensitivity_shards.py":
        raise ComponentResponsePlanFromSensitivityError("merged sensitivity summary missing merge_tool")
    if summary.get("device") != "cuda":
        raise ComponentResponsePlanFromSensitivityError(
            f"merged sensitivity device={summary.get('device')!r}, expected 'cuda'"
        )
    if summary.get("sensitivity_source") != DIRECT_FD_SOURCE:
        raise ComponentResponsePlanFromSensitivityError(
            "merged sensitivity artifact is not direct finite difference"
        )
    for field in ("score_claim", "promotion_eligible", "official_component_response", "canonical_scorer_path"):
        _require_false(summary, field=field, label="merged sensitivity summary")
    for field in ("score_claim", "promotion_eligible"):
        _require_false(validation, field=field, label="merged sensitivity validation")
    merge = summary.get("finite_difference_merge")
    if not isinstance(merge, dict):
        raise ComponentResponsePlanFromSensitivityError("merged sensitivity summary missing finite_difference_merge")
    validation_merge = validation.get("finite_difference_merge")
    if not isinstance(validation_merge, Mapping):
        raise ComponentResponsePlanFromSensitivityError(
            "merged sensitivity validation missing finite_difference_merge"
        )
    if dict(validation_merge) != dict(merge):
        raise ComponentResponsePlanFromSensitivityError(
            "merged sensitivity summary finite_difference_merge differs from validation"
        )
    for field in (
        "all_channel_count",
        "all_channel_sha256",
        "covered_channel_count",
        "covered_channel_sha256",
        "declared_shard_count",
        "expected_shard_count",
        "missing_channel_count",
        "missing_channel_sha256",
        "missing_shard_indices",
        "provided_shard_indices",
        "coverage",
    ):
        if field in validation and field in merge and validation.get(field) != merge.get(field):
            raise ComponentResponsePlanFromSensitivityError(
                f"merged sensitivity validation {field} differs from finite_difference_merge"
            )
    for payload, label in ((merge, "finite_difference_merge"), (validation, "merge validation")):
        if payload.get("coverage") != "exactly_once":
            raise ComponentResponsePlanFromSensitivityError(
                f"{label} coverage={payload.get('coverage')!r}; complete exactly-once coverage is required"
            )
        if payload.get("missing_channel_count") != 0 or payload.get("missing_shard_indices") not in ([], None):
            raise ComponentResponsePlanFromSensitivityError(f"{label} records missing shard/channel coverage")
        if payload.get("all_channel_count") != payload.get("covered_channel_count"):
            raise ComponentResponsePlanFromSensitivityError(f"{label} channel coverage count mismatch")
        if payload.get("certification_handoff_eligible") is not True:
            raise ComponentResponsePlanFromSensitivityError(
                f"{label} must be certification_handoff_eligible=true for planning"
            )
    fd_shard = summary.get("finite_difference_shard")
    if not isinstance(fd_shard, dict) or fd_shard.get("is_shard") is not False:
        raise ComponentResponsePlanFromSensitivityError("merged sensitivity summary must record a merged shard")
    if fd_shard.get("merge_required_for_certification_handoff") is not False:
        raise ComponentResponsePlanFromSensitivityError(
            "merged sensitivity shard still requires merge before certification handoff"
        )
    if summary.get("n_pairs_total") != 600:
        raise ComponentResponsePlanFromSensitivityError(
            f"merged sensitivity n_pairs_total={summary.get('n_pairs_total')!r}, expected 600"
        )

    from tac.sensitivity_map import load_sensitivity_map

    map_metadata: dict[str, Any] = {}
    for component in COMPONENTS:
        for suffix, split in (
            ("sensitivity_map.pt", None),
            ("holdout_sensitivity_map.pt", "holdout"),
        ):
            map_path = _required_artifact_file(artifact_dir, f"{component}_{suffix}")
            values, metadata = load_sensitivity_map(map_path)
            if not values:
                raise ComponentResponsePlanFromSensitivityError(
                    f"{component} {suffix} contains no tensor entries"
                )
            if metadata.get("device") != "cuda":
                raise ComponentResponsePlanFromSensitivityError(
                    f"{component} {suffix} device={metadata.get('device')!r}, expected 'cuda'"
                )
            if metadata.get("component") != component and metadata.get("scorer_target") != component:
                raise ComponentResponsePlanFromSensitivityError(
                    f"{component} {suffix} metadata does not identify the component"
                )
            if split is not None and metadata.get("split") != split:
                raise ComponentResponsePlanFromSensitivityError(
                    f"{component} {suffix} metadata must record split={split!r}"
                )
            for field in ("score_claim", "promotion_eligible", "official_component_response", "canonical_scorer_path"):
                _require_false(metadata, field=field, label=f"{component} {suffix} metadata")
            if metadata.get("sensitivity_source") != DIRECT_FD_SOURCE:
                raise ComponentResponsePlanFromSensitivityError(
                    f"{component} {suffix} sensitivity_source mismatch"
                )
            if metadata.get("finite_difference_shard") != fd_shard:
                raise ComponentResponsePlanFromSensitivityError(
                    f"{component} {suffix} finite_difference_shard metadata mismatch"
                )
            if split is None:
                map_metadata[component] = metadata

    source_dirs = merge.get("source_shard_dirs")
    if not isinstance(source_dirs, list) or not source_dirs:
        raise ComponentResponsePlanFromSensitivityError("merged artifact missing source_shard_dirs")
    source_shards = [
        _validate_source_shard_custody(
            _source_shard_dir(raw),
            expected_baseline_archive_sha256=expected_baseline_archive_sha256,
            expected_baseline_archive_size_bytes=expected_baseline_archive_size_bytes,
        )
        for raw in source_dirs
    ]
    source_indices = sorted(item["shard_index"] for item in source_shards)
    expected_indices = list(range(int(merge.get("declared_shard_count", len(source_shards)))))
    if source_indices != expected_indices:
        raise ComponentResponsePlanFromSensitivityError(
            f"merged source shard indices {source_indices} do not match expected {expected_indices}"
        )
    validation_source_shards = validation.get("source_shards")
    if not isinstance(validation_source_shards, list) or not validation_source_shards:
        raise ComponentResponsePlanFromSensitivityError(
            "merged sensitivity validation missing source_shards"
        )
    normalized_validation_shards = []
    for item in validation_source_shards:
        if not isinstance(item, Mapping):
            raise ComponentResponsePlanFromSensitivityError(
                "merged sensitivity validation source_shards entries must be objects"
            )
        normalized_validation_shards.append(
            {
                "path": str(_source_shard_dir(item.get("path"))),
                "shard_index": item.get("shard_index"),
                "assigned_channel_count": item.get("assigned_channel_count"),
                "assigned_channel_sha256": item.get("assigned_channel_sha256"),
            }
        )
    normalized_source_shards = [
        {
            "path": item["path"],
            "shard_index": item["shard_index"],
            "assigned_channel_count": item["assigned_channel_count"],
            "assigned_channel_sha256": item["assigned_channel_sha256"],
        }
        for item in source_shards
    ]
    if sorted(normalized_validation_shards, key=lambda item: int(item["shard_index"])) != sorted(
        normalized_source_shards,
        key=lambda item: int(item["shard_index"]),
    ):
        raise ComponentResponsePlanFromSensitivityError(
            "merged sensitivity validation source_shards do not match source shard custody"
        )
    assigned_sha_by_shard = merge.get("assigned_channel_sha256_by_shard")
    if isinstance(assigned_sha_by_shard, list):
        expected_assigned = [
            item["assigned_channel_sha256"]
            for item in sorted(normalized_source_shards, key=lambda row: int(row["shard_index"]))
        ]
        if assigned_sha_by_shard != expected_assigned:
            raise ComponentResponsePlanFromSensitivityError(
                "merged sensitivity assigned_channel_sha256_by_shard does not match source shards"
            )

    return {
        "schema_version": 1,
        "artifact_dir": str(artifact_dir),
        "role": "diagnostic_component_sensitivity_merged_shards",
        "baseline_archive_sha256": expected_baseline_archive_sha256,
        "baseline_archive_size_bytes": expected_baseline_archive_size_bytes,
        "device": "cuda",
        "gpu_model": "mixed_shards",
        "metadata": {},
        "summary": summary,
        "merge_validation": validation,
        "source_shards": source_shards,
        "sensitivity_source": DIRECT_FD_SOURCE,
        "planning_eligible": True,
        "certification_handoff_eligible": True,
        "certification_candidate": True,
        "map_metadata": map_metadata,
        "promotion_eligible": False,
        "score_claim": False,
        "score_source": "none:diagnostic_component_sensitivity_merged_shards_non_promotable",
    }


def _validate_perturbation_basis_against_baseline(
    basis_json: Path,
    *,
    baseline_archive: Path,
    baseline_meta: Mapping[str, Any],
) -> None:
    basis = _load_json_object(basis_json, label="perturbation basis")
    source_archive = basis.get("source_archive")
    if not isinstance(source_archive, Mapping):
        raise ComponentResponsePlanFromSensitivityError(
            f"{basis_json}: basis must record source_archive custody"
        )
    if source_archive.get("sha256") != baseline_meta["sha256"]:
        raise ComponentResponsePlanFromSensitivityError(
            f"{basis_json}: source_archive.sha256={source_archive.get('sha256')!r} "
            f"does not match baseline {baseline_meta['sha256']!r}"
        )
    if source_archive.get("bytes") != baseline_meta["bytes"]:
        raise ComponentResponsePlanFromSensitivityError(
            f"{basis_json}: source_archive.bytes={source_archive.get('bytes')!r} "
            f"does not match baseline {baseline_meta['bytes']!r}"
        )

    import zipfile

    members: dict[str, bytes] = {}
    with zipfile.ZipFile(baseline_archive, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            if info.filename in members:
                raise ComponentResponsePlanFromSensitivityError(
                    f"baseline archive has duplicate member {info.filename!r}"
                )
            members[info.filename] = zf.read(info)
    raw_atoms = basis.get("atoms")
    if not isinstance(raw_atoms, list) or not raw_atoms:
        raise ComponentResponsePlanFromSensitivityError(f"{basis_json}: atoms must be non-empty")
    for index, raw in enumerate(raw_atoms):
        if not isinstance(raw, Mapping):
            raise ComponentResponsePlanFromSensitivityError(
                f"{basis_json}: atoms[{index}] must be an object"
            )
        original_byte = raw.get("original_byte")
        if original_byte is None:
            metadata = raw.get("metadata")
            if isinstance(metadata, Mapping):
                original_byte = metadata.get("original_byte")
        if original_byte is None:
            continue
        member = raw.get("member")
        offset = raw.get("offset")
        if not isinstance(member, str) or member not in members:
            raise ComponentResponsePlanFromSensitivityError(
                f"{basis_json}: atoms[{index}] member={member!r} not in baseline"
            )
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ComponentResponsePlanFromSensitivityError(
                f"{basis_json}: atoms[{index}].offset invalid"
            )
        if offset >= len(members[member]):
            raise ComponentResponsePlanFromSensitivityError(
                f"{basis_json}: atoms[{index}] offset outside {member}"
            )
        if isinstance(original_byte, bool) or not isinstance(original_byte, int):
            raise ComponentResponsePlanFromSensitivityError(
                f"{basis_json}: atoms[{index}].original_byte must be an integer"
            )
        actual = int(members[member][offset])
        if int(original_byte) != actual:
            raise ComponentResponsePlanFromSensitivityError(
                f"{basis_json}: atoms[{index}] original_byte={original_byte} "
                f"does not match baseline {member}@{offset}={actual}"
            )


def build_component_response_plan_from_sensitivity_artifacts(
    *,
    sensitivity_artifact_dir: Path,
    baseline_archive: Path,
    baseline_contest_auth_eval_json: Path,
    perturbation_basis_json: Path | None = None,
    output_dir: Path,
    epsilons: list[float],
    max_mutated_bytes: int = 1024,
    max_abs_byte_delta: int = 32,
    max_raw_l1_delta: int = 4096,
    max_archive_bytes: int | None = None,
    max_archive_byte_delta: int | None = None,
    allow_merged_shard_artifact: bool = False,
    response_only_no_prediction_deltas: bool = False,
) -> dict[str, Any]:
    """Build prediction deltas and official response plan from harvested maps."""

    sensitivity_artifact_dir = sensitivity_artifact_dir.resolve()
    baseline_archive = baseline_archive.resolve()
    baseline_contest_auth_eval_json = baseline_contest_auth_eval_json.resolve()
    output_dir = output_dir.resolve()
    epsilon_values = _finite_epsilons(epsilons)
    baseline_meta = _file_meta(baseline_archive)
    baseline_eval_meta = _file_meta(baseline_contest_auth_eval_json)

    if allow_merged_shard_artifact:
        validation = _validate_merged_component_sensitivity_artifact_dir(
            sensitivity_artifact_dir,
            expected_baseline_archive_sha256=baseline_meta["sha256"],
            expected_baseline_archive_size_bytes=baseline_meta["bytes"],
        )
    else:
        validation = validate_local_component_sensitivity_artifact_dir(
            sensitivity_artifact_dir,
            expected_baseline_archive_sha256=baseline_meta["sha256"],
            expected_baseline_archive_size_bytes=baseline_meta["bytes"],
        )
    if validation.get("device") != "cuda":
        raise ComponentResponsePlanFromSensitivityError(
            f"sensitivity artifact device={validation.get('device')!r}, expected 'cuda'"
        )
    if validation.get("promotion_eligible") is not False:
        raise ComponentResponsePlanFromSensitivityError(
            "diagnostic sensitivity artifact must remain promotion_eligible=false"
        )

    if perturbation_basis_json is None:
        basis_json = _required_artifact_file(sensitivity_artifact_dir, "perturbation_basis_v1.json")
        basis_source = "sensitivity_artifact_dir"
    else:
        basis_json = perturbation_basis_json.resolve()
        _file_meta(basis_json)
        basis_source = "explicit_perturbation_basis_json"
    _validate_perturbation_basis_against_baseline(
        basis_json,
        baseline_archive=baseline_archive,
        baseline_meta=baseline_meta,
    )
    map_paths = {
        component: _required_artifact_file(
            sensitivity_artifact_dir,
            f"{component}_sensitivity_map.pt",
        )
        for component in COMPONENTS
    }

    prediction_json: Path | None = None
    if response_only_no_prediction_deltas:
        prediction_summary: dict[str, Any] = {
            "prediction_deltas": None,
            "omitted": True,
            "reason": (
                "response_only_no_prediction_deltas: source sensitivity maps are "
                "ranking/planning artifacts unless a byte-basis calibration is present"
            ),
        }
    else:
        prediction_json = output_dir / "official_component_response_prediction_deltas.json"
        prediction_summary = build_component_response_prediction_deltas(
            baseline_archive=baseline_archive,
            perturbation_basis_json=basis_json,
            output_json=prediction_json,
            component_maps=map_paths,
            epsilons=epsilon_values,
        )

    plan_dir = output_dir / "official_response_plan"
    plan_summary = build_component_response_perturbation_plan(
        baseline_archive=baseline_archive,
        output_dir=plan_dir,
        atoms=_load_basis_atoms(basis_json),
        epsilons=epsilon_values,
        plan_output=plan_dir / "official_component_response_plan.json",
        basis_output=plan_dir / "perturbation_basis_v1.json",
        variants_manifest_output=plan_dir / "archive_variants_manifest.json",
        baseline_contest_auth_eval_json=baseline_contest_auth_eval_json,
        predicted_deltas_json=prediction_json,
        require_predicted_deltas=not response_only_no_prediction_deltas,
        max_mutated_bytes=max_mutated_bytes,
        max_abs_byte_delta=max_abs_byte_delta,
        max_raw_l1_delta=max_raw_l1_delta,
        max_archive_bytes=max_archive_bytes,
        max_archive_byte_delta=max_archive_byte_delta,
    )

    summary_path = output_dir / "component_response_plan_from_sensitivity_artifacts_summary.json"
    summary = {
        "schema_version": 1,
        "format": SUMMARY_FORMAT,
        "producer": PRODUCER,
        "score_claim": False,
        "promotion_eligible": False,
        "response_only_no_prediction_deltas": bool(response_only_no_prediction_deltas),
        "evidence_grade": "planning:pre_response_prediction_and_archive_variants",
        "baseline_archive": baseline_meta,
        "baseline_contest_auth_eval_json": baseline_eval_meta,
        "sensitivity_artifact": {
            "path": str(sensitivity_artifact_dir),
            "role": validation.get("role"),
            "device": validation.get("device"),
            "gpu_model": validation.get("gpu_model"),
            "baseline_archive_sha256": validation.get("baseline_archive_sha256"),
            "baseline_archive_size_bytes": validation.get("baseline_archive_size_bytes"),
            "promotion_eligible": validation.get("promotion_eligible"),
            "score_claim": validation.get("score_claim"),
            "score_source": validation.get("score_source"),
        },
        "epsilon_ladder": [float(value) for value in epsilon_values],
        "component_maps": {component: _file_meta(path) for component, path in map_paths.items()},
        "perturbation_basis": {
            **_file_meta(basis_json),
            "basis_source": basis_source,
        },
        "prediction_deltas": prediction_summary["prediction_deltas"],
        "official_response_plan": plan_summary["plan"],
        "archive_variants_manifest": plan_summary["archive_variants_manifest"],
        "official_response_next_step": {
            "script": "experiments/profile_component_sensitivity_official.py",
            "requires_device": "cuda",
            "requires_same_run_zero": True,
            "requires_require_passed": not response_only_no_prediction_deltas,
        },
        "summary_path": str(summary_path),
    }
    _write_json(summary_path, summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sensitivity-artifact-dir", type=Path, required=True)
    parser.add_argument("--baseline-archive", type=Path, required=True)
    parser.add_argument("--baseline-contest-auth-eval-json", type=Path, required=True)
    parser.add_argument(
        "--perturbation-basis-json",
        type=Path,
        default=None,
        help=(
            "Optional fresh perturbation_basis_v1 JSON. If omitted, uses "
            "<sensitivity-artifact-dir>/perturbation_basis_v1.json."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epsilon", action="append", type=float, default=None)
    parser.add_argument("--max-mutated-bytes", type=int, default=1024)
    parser.add_argument("--max-abs-byte-delta", type=int, default=32)
    parser.add_argument("--max-raw-l1-delta", type=int, default=4096)
    parser.add_argument("--max-archive-bytes", type=int, default=None)
    parser.add_argument("--max-archive-byte-delta", type=int, default=None)
    parser.add_argument(
        "--allow-merged-shard-artifact",
        action="store_true",
        help=(
            "Accept a complete exactly-once output from "
            "experiments/merge_component_sensitivity_shards.py. This preserves "
            "non-promotable planning status and validates every source shard "
            "against the expected baseline archive."
        ),
    )
    parser.add_argument(
        "--response-only-no-prediction-deltas",
        action="store_true",
        help=(
            "Build archive variants and an official-response plan without "
            "predicted_delta fields. Use for rank-only sensitivity maps whose "
            "units are not reviewed archive-byte response units; resulting "
            "response runs are diagnostic/calibration, not promotion-passed."
        ),
    )
    args = parser.parse_args(argv)
    epsilons = args.epsilon if args.epsilon is not None else list(DEFAULT_OFFICIAL_EPSILONS)
    args.epsilons = _finite_epsilons([float(value) for value in epsilons])
    for field in ("max_mutated_bytes", "max_abs_byte_delta", "max_raw_l1_delta"):
        if getattr(args, field) < 0:
            parser.error(f"--{field.replace('_', '-')} must be nonnegative")
    for field in ("max_archive_bytes", "max_archive_byte_delta"):
        value = getattr(args, field)
        if value is not None and value < 0:
            parser.error(f"--{field.replace('_', '-')} must be nonnegative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = build_component_response_plan_from_sensitivity_artifacts(
            sensitivity_artifact_dir=args.sensitivity_artifact_dir,
            baseline_archive=args.baseline_archive,
            baseline_contest_auth_eval_json=args.baseline_contest_auth_eval_json,
            perturbation_basis_json=args.perturbation_basis_json,
            output_dir=args.output_dir,
            epsilons=args.epsilons,
            max_mutated_bytes=args.max_mutated_bytes,
            max_abs_byte_delta=args.max_abs_byte_delta,
            max_raw_l1_delta=args.max_raw_l1_delta,
            max_archive_bytes=args.max_archive_bytes,
            max_archive_byte_delta=args.max_archive_byte_delta,
            allow_merged_shard_artifact=args.allow_merged_shard_artifact,
            response_only_no_prediction_deltas=args.response_only_no_prediction_deltas,
        )
    except (ComponentResponsePlanFromSensitivityError, ValueError, FileNotFoundError) as exc:
        raise SystemExit(f"FATAL: {exc}") from exc
    print(json_text(summary), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
