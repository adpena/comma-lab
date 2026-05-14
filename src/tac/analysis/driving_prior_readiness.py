# SPDX-License-Identifier: MIT
"""Readiness manifest for the 2032 driving-prior saliency recommendation.

This module operationalizes the fdfc347f 2032 driving-prior recommendation as
a proxy-safe readiness probe. It does not download external datasets, load a
scorer at inflate time, train a renderer, build an archive, dispatch jobs, or
claim score movement.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tac.optimization.candidate_evidence_contract import CONTEST_UNCOMPRESSED_BYTES
from tac.optimization.proxy_candidate_contract import (
    apply_proxy_evidence_boundary,
    ordered_unique,
    validate_proxy_candidate,
)

SCHEMA = "tac_2032_driving_prior_readiness.v1"
TOOL = "tac.analysis.driving_prior_readiness"
LANE_ID = "driving_prior_pretrained_renderer_2032"
SOURCE_COMMIT = "fdfc347f"
SOURCE_MEMO = ".omx/research/expert_team_hardware_physics_future_alien_tech_20260513.md"
SOURCE_LEDGER = (
    ".omx/research/expert_team_hardware_physics_future_ledgers/"
    "07_time_traveler_2032_l5_autonomy_secret.md"
)
RATE_SCORE_PER_BYTE = 25.0 / CONTEST_UNCOMPRESSED_BYTES


class DrivingPriorReadinessError(ValueError):
    """Raised when a readiness manifest or hook probe is malformed."""


@dataclass(frozen=True)
class DatasetPretrainingSource:
    """One optional external dataset source for future pretraining."""

    dataset_id: str
    display_name: str
    role: str
    local_path_env: str
    license_gate: str
    expected_signal: str

    def inspect(self, env: Mapping[str, str] | None = None) -> dict[str, Any]:
        env_map = env if env is not None else os.environ
        configured = str(env_map.get(self.local_path_env, "")).strip()
        local_path = str(Path(configured).expanduser()) if configured else ""
        local_status = "external_optional_not_configured"
        if configured:
            local_status = (
                "local_path_present"
                if Path(local_path).exists()
                else "local_path_env_set_but_missing"
            )
        return {
            **asdict(self),
            "optional_external_source": True,
            "download_attempted": False,
            "network_access_attempted": False,
            "local_path": local_path,
            "local_status": local_status,
            "training_use": "pretraining_only_after_operator_supplies_local_copy",
        }


@dataclass(frozen=True)
class PenultimateHookTarget:
    """A scorer module whose forward output can seed saliency features."""

    target_id: str
    scorer_id: str
    module_path: str
    required: bool
    saliency_role: str
    expected_feature: str


@dataclass(frozen=True)
class ArchiveBudgetComponent:
    """Speculative byte budget component, still unmaterialized."""

    component_id: str
    purpose: str
    estimated_bytes_low: int
    estimated_bytes_high: int
    charged_to_archive: bool
    promotion_dependency: str

    def __post_init__(self) -> None:
        if self.estimated_bytes_low < 0 or self.estimated_bytes_high < 0:
            raise DrivingPriorReadinessError(f"{self.component_id}: byte estimates must be non-negative")
        if self.estimated_bytes_low > self.estimated_bytes_high:
            raise DrivingPriorReadinessError(f"{self.component_id}: low estimate exceeds high estimate")

    def as_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "rate_score_low": self.estimated_bytes_low * RATE_SCORE_PER_BYTE,
            "rate_score_high": self.estimated_bytes_high * RATE_SCORE_PER_BYTE,
        }


DEFAULT_DATASET_SOURCES: tuple[DatasetPretrainingSource, ...] = (
    DatasetPretrainingSource(
        dataset_id="comma2k19",
        display_name="Comma2k19",
        role="dashcam_temporal_prior",
        local_path_env="COMMA2K19_ROOT",
        license_gate="operator_must_verify_dataset_license_and_terms_before_training",
        expected_signal="forward-facing driving video, ego-motion, lane/scene priors",
    ),
    DatasetPretrainingSource(
        dataset_id="bdd100k",
        display_name="BDD100K",
        role="semantic_scene_prior",
        local_path_env="BDD100K_ROOT",
        license_gate="operator_must_verify_dataset_license_and_terms_before_training",
        expected_signal="road-object-semantic diversity for scorer-feature inversion",
    ),
    DatasetPretrainingSource(
        dataset_id="waymo_open_dataset",
        display_name="Waymo Open Dataset",
        role="motion_and_scene_geometry_prior",
        local_path_env="WAYMO_OPEN_DATASET_ROOT",
        license_gate="operator_must_verify_dataset_license_and_terms_before_training",
        expected_signal="vehicle dynamics, object motion, wide-scene geometry",
    ),
)

DEFAULT_HOOK_TARGETS: tuple[PenultimateHookTarget, ...] = (
    PenultimateHookTarget(
        target_id="posenet_vision_embedding",
        scorer_id="posenet",
        module_path="vision",
        required=False,
        saliency_role="raw_fastvit_world_feature_candidate",
        expected_feature="2048-d PoseNet vision embedding before summary projection",
    ),
    PenultimateHookTarget(
        target_id="posenet_summary_embedding",
        scorer_id="posenet",
        module_path="summarizer",
        required=True,
        saliency_role="pose_penultimate_feature_saliency",
        expected_feature="512-d PoseNet summary before Hydra heads",
    ),
    PenultimateHookTarget(
        target_id="posenet_hydra_resblock",
        scorer_id="posenet",
        module_path="hydra.resblock",
        required=False,
        saliency_role="pose_head_prelinear_candidate",
        expected_feature="512-d Hydra residual block before pose head",
    ),
    PenultimateHookTarget(
        target_id="segnet_encoder_pyramid",
        scorer_id="segnet",
        module_path="encoder",
        required=True,
        saliency_role="segmentation_world_model_feature_saliency",
        expected_feature="EfficientNet-B2 encoder feature pyramid",
    ),
    PenultimateHookTarget(
        target_id="segnet_decoder_map",
        scorer_id="segnet",
        module_path="decoder",
        required=False,
        saliency_role="segmentation_prelogit_feature_candidate",
        expected_feature="UNet decoder map before segmentation head",
    ),
)

DEFAULT_ARCHIVE_BUDGET: tuple[ArchiveBudgetComponent, ...] = (
    ArchiveBudgetComponent(
        component_id="world_model_prior_weights",
        purpose="sub-100K driving renderer prior, quantized and charged",
        estimated_bytes_low=24_000,
        estimated_bytes_high=35_000,
        charged_to_archive=True,
        promotion_dependency="train_and_export_deterministic_renderer_prior",
    ),
    ArchiveBudgetComponent(
        component_id="contest_residual_delta",
        purpose="contest-video-specific renderer/feature residual",
        estimated_bytes_low=5_000,
        estimated_bytes_high=15_000,
        charged_to_archive=True,
        promotion_dependency="score-feature saliency allocation and no-op proof",
    ),
    ArchiveBudgetComponent(
        component_id="ego_motion_lpc_innovations",
        purpose="pose-axis LPC innovations and initial conditions",
        estimated_bytes_low=6_000,
        estimated_bytes_high=10_000,
        charged_to_archive=True,
        promotion_dependency="pose residual grammar consumed by inflate",
    ),
    ArchiveBudgetComponent(
        component_id="boundary_inpaint_edges",
        purpose="high-frequency boundary content for scorer-sensitive edges",
        estimated_bytes_low=8_000,
        estimated_bytes_high=12_000,
        charged_to_archive=True,
        promotion_dependency="edge stream parser and pose/seg non-collapse gate",
    ),
    ArchiveBudgetComponent(
        component_id="scene_skeleton",
        purpose="small inverse-rendered object/geometry control stream",
        estimated_bytes_low=2_000,
        estimated_bytes_high=4_000,
        charged_to_archive=True,
        promotion_dependency="deterministic scene grammar and parser-section manifest",
    ),
    ArchiveBudgetComponent(
        component_id="inflate_runtime_source",
        purpose="scorer-free deterministic runtime source and tiny decoder",
        estimated_bytes_low=2_000,
        estimated_bytes_high=4_000,
        charged_to_archive=True,
        promotion_dependency="inflate.py runtime closure without scorer imports",
    ),
)


def default_repo_root() -> Path:
    """Return the repository root inferred from this module path."""

    return Path(__file__).resolve().parents[3]


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _artifact_row(repo_root: Path, relative_path: str, *, required: bool, hash_file: bool) -> dict[str, Any]:
    path = repo_root / relative_path
    exists = path.is_file()
    row: dict[str, Any] = {
        "relative_path": relative_path,
        "required": required,
        "exists": exists,
        "readable": exists and os.access(path, os.R_OK),
        "bytes": path.stat().st_size if exists else 0,
        "sha256": "",
    }
    if exists and hash_file:
        row["sha256"] = _sha256_file(path)
    return row


def _dependency_status() -> list[dict[str, Any]]:
    deps = (
        ("torch", "torch"),
        ("timm", "timm"),
        ("einops", "einops"),
        ("safetensors", "safetensors"),
        ("segmentation_models_pytorch", "segmentation_models_pytorch"),
    )
    return [
        {
            "package": package,
            "import_name": import_name,
            "available": importlib.util.find_spec(import_name) is not None,
        }
        for package, import_name in deps
    ]


def _named_modules(model: Any) -> dict[str, Any]:
    if hasattr(model, "named_modules"):
        return {str(name): module for name, module in model.named_modules()}
    return {}


def _resolve_module(model: Any, module_path: str) -> Any | None:
    named = _named_modules(model)
    if module_path in named:
        return named[module_path]
    cur = model
    for part in module_path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def _probe_one_hook(module: Any) -> tuple[bool, str]:
    register = getattr(module, "register_forward_hook", None)
    if register is None:
        return False, "module_lacks_register_forward_hook"
    handle = None
    try:
        handle = register(lambda *_args: None)
    except Exception as exc:  # pragma: no cover - exact message is dependency-specific
        return False, f"register_forward_hook_failed:{type(exc).__name__}:{exc}"
    finally:
        if handle is not None and hasattr(handle, "remove"):
            handle.remove()
    return True, "hook_registerable"


def check_penultimate_hook_targets(
    models: Mapping[str, Any],
    *,
    targets: Sequence[PenultimateHookTarget] = DEFAULT_HOOK_TARGETS,
) -> list[dict[str, Any]]:
    """Check that candidate scorer feature modules can accept forward hooks."""

    rows: list[dict[str, Any]] = []
    for target in targets:
        model = models.get(target.scorer_id)
        row = {
            **asdict(target),
            "model_present": model is not None,
            "module_present": False,
            "hook_registerable": False,
            "status": "model_missing",
        }
        if model is None:
            rows.append(row)
            continue
        module = _resolve_module(model, target.module_path)
        row["module_present"] = module is not None
        if module is None:
            row["status"] = "module_missing"
            rows.append(row)
            continue
        ok, status = _probe_one_hook(module)
        row["hook_registerable"] = ok
        row["status"] = status
        rows.append(row)
    return rows


def inspect_scorer_readiness(
    repo_root: str | Path | None = None,
    *,
    probe_structure: bool = True,
    load_weights: bool = False,
    hash_weights: bool = False,
) -> dict[str, Any]:
    """Inspect local scorer files/imports and, when possible, hook readiness.

    Weight loading is opt-in because the default readiness probe should remain a
    cheap CPU planning step. Even when weights are loaded, the context is
    analysis/compress-time only and never inflate-time.
    """

    root = Path(repo_root) if repo_root is not None else default_repo_root()
    root = root.resolve()
    artifacts = [
        _artifact_row(root, "upstream/modules.py", required=True, hash_file=False),
        _artifact_row(root, "upstream/models/posenet.safetensors", required=True, hash_file=hash_weights),
        _artifact_row(root, "upstream/models/segnet.safetensors", required=True, hash_file=hash_weights),
    ]
    deps = _dependency_status()
    missing_deps = [row["package"] for row in deps if not row["available"]]
    missing_artifacts = [row["relative_path"] for row in artifacts if row["required"] and not row["exists"]]
    hook_rows: list[dict[str, Any]] = []
    hook_probe = {
        "attempted": False,
        "load_weights": load_weights,
        "status": "not_attempted",
        "targets": hook_rows,
    }
    blockers: list[str] = []
    if missing_artifacts:
        blockers.extend(f"missing_scorer_artifact:{path}" for path in missing_artifacts)
    if missing_deps:
        blockers.extend(f"missing_python_dependency:{name}" for name in missing_deps)
    if not probe_structure:
        hook_probe["status"] = "skipped_by_operator_flag"
    elif missing_deps:
        hook_probe["status"] = "blocked_missing_python_deps"
    elif not (root / "upstream/modules.py").is_file():
        hook_probe["status"] = "blocked_missing_upstream_modules"
    else:
        upstream = str(root / "upstream")
        if upstream not in sys.path:
            sys.path.insert(0, upstream)
        try:
            from modules import PoseNet, SegNet  # type: ignore

            if load_weights:
                from safetensors.torch import load_file  # type: ignore

            posenet = PoseNet().eval()
            segnet = SegNet().eval()
            if load_weights:
                posenet.load_state_dict(
                    load_file(str(root / "upstream/models/posenet.safetensors"), device="cpu")
                )
                segnet.load_state_dict(
                    load_file(str(root / "upstream/models/segnet.safetensors"), device="cpu")
                )
            hook_rows = check_penultimate_hook_targets({"posenet": posenet, "segnet": segnet})
            required_failures = [
                row["target_id"]
                for row in hook_rows
                if row["required"] and not row["hook_registerable"]
            ]
            hook_probe = {
                "attempted": True,
                "load_weights": load_weights,
                "status": "passed" if not required_failures else "failed_required_targets",
                "targets": hook_rows,
            }
            blockers.extend(f"required_hook_target_not_ready:{target}" for target in required_failures)
        except Exception as exc:  # pragma: no cover - dependency-specific
            hook_probe = {
                "attempted": True,
                "load_weights": load_weights,
                "status": "failed_exception",
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "targets": hook_rows,
            }
            blockers.append(f"scorer_hook_probe_exception:{type(exc).__name__}")

    return {
        "schema": "scorer_penultimate_saliency_probe.v1",
        "analysis_time_only": True,
        "inflate_time_scorer_load_allowed": False,
        "device_policy": "cpu_only_no_cuda_no_mps",
        "artifacts": artifacts,
        "python_dependencies": deps,
        "hook_probe": hook_probe,
        "scorer_model_accessible": not missing_artifacts,
        "scorer_python_deps_available": not missing_deps,
        "scorer_penultimate_hook_ready": hook_probe.get("status") == "passed",
        "blockers": ordered_unique(blockers),
    }


def build_dataset_pretraining_plan(
    *,
    env: Mapping[str, str] | None = None,
    sources: Sequence[DatasetPretrainingSource] = DEFAULT_DATASET_SOURCES,
) -> dict[str, Any]:
    """Return a no-download plan for optional public-driving pretraining."""

    rows = [source.inspect(env) for source in sources]
    return {
        "schema": "driving_prior_dataset_pretraining_plan.v1",
        "downloads_attempted": False,
        "network_access_attempted": False,
        "sources": rows,
        "policy": {
            "external_sources_optional": True,
            "operator_supplies_local_copies": True,
            "license_review_required_before_training": True,
            "no_dataset_bytes_committed": True,
            "no_training_started": True,
        },
    }


def estimate_archive_budget(
    components: Sequence[ArchiveBudgetComponent] = DEFAULT_ARCHIVE_BUDGET,
) -> dict[str, Any]:
    """Return the speculative byte budget and score-rate term estimate."""

    rows = [component.as_dict() for component in components]
    low = sum(component.estimated_bytes_low for component in components)
    high = sum(component.estimated_bytes_high for component in components)
    return {
        "schema": "driving_prior_archive_budget_estimate.v1",
        "estimate_only_no_archive_materialized": True,
        "components": rows,
        "total_estimated_bytes_low": low,
        "total_estimated_bytes_high": high,
        "rate_score_low": low * RATE_SCORE_PER_BYTE,
        "rate_score_high": high * RATE_SCORE_PER_BYTE,
        "rate_formula": "25 * archive_bytes / 37545489",
        "byte_closed_packet_required_before_promotion": True,
    }


def _solver_wire_in() -> dict[str, Any]:
    return {
        "research_only": True,
        "sensitivity_map_contribution": "planned: scorer_penultimate_feature_saliency_map",
        "pareto_constraint": "non_binding_until_byte_closed_archive_and_exact_eval",
        "bit_allocator_hook": "planned: convert hook saliency to protection weights after trained prior exists",
        "cathedral_autopilot_dispatch_hook": "blocked: proxy-only row, no exact-ready dispatch",
        "continual_learning_posterior_update": "disabled: no empirical anchor",
        "probe_disambiguator": (
            "emit multiple hook targets now; future probe arbitrates "
            "SegNet encoder vs decoder and PoseNet summarizer vs Hydra features"
        ),
    }


def build_driving_prior_readiness_manifest(
    repo_root: str | Path | None = None,
    *,
    probe_scorer: bool = True,
    load_weights: bool = False,
    hash_weights: bool = False,
    env: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build the typed proxy-safe readiness manifest."""

    root = Path(repo_root) if repo_root is not None else default_repo_root()
    root = root.resolve()
    scorer = inspect_scorer_readiness(
        root,
        probe_structure=probe_scorer,
        load_weights=load_weights,
        hash_weights=hash_weights,
    )
    datasets = build_dataset_pretraining_plan(env=env)
    budget = estimate_archive_budget()
    training_blockers = [
        "public_driving_prior_not_trained",
        "deterministic_renderer_export_missing",
        "byte_closed_archive_not_materialized",
        "no_op_consumption_proof_missing",
        "paired_contest_cuda_and_contest_cpu_exact_eval_missing",
        "scorer_hooks_are_analysis_time_only_not_inflate_time",
    ]
    manifest: dict[str, Any] = {
        "schema": SCHEMA,
        "schema_version": 1,
        "tool": TOOL,
        "generated_at_utc": _utc_now(),
        "repo_root": str(root),
        "lane_id": LANE_ID,
        "campaign_id": LANE_ID,
        "source_commit": SOURCE_COMMIT,
        "source_memo": SOURCE_MEMO,
        "source_ledger": SOURCE_LEDGER,
        "recommendation": (
            "Pretrain a small driving renderer prior on operator-supplied "
            "public driving datasets, then use analysis-time scorer "
            "penultimate-feature saliency to allocate contest residual bytes."
        ),
        "research_only": True,
        "gpu_required": False,
        "gpu_used": False,
        "network_access_attempted": False,
        "downloads_attempted": False,
        "training_started": False,
        "archive_materialized": False,
        "score_evidence_grade": "invalid_no_score_planning_only",
        "scorer_readiness": scorer,
        "dataset_pretraining_plan": datasets,
        "archive_budget_estimate": budget,
        "saliency_probe_plan": {
            "schema": "scorer_penultimate_saliency_plan.v1",
            "analysis_time_only": True,
            "inflate_time_scorer_load_allowed": False,
            "hook_targets": [asdict(target) for target in DEFAULT_HOOK_TARGETS],
            "saliency_output": (
                "future tensor rows keyed by scorer_id, target_id, frame_pair, "
                "feature_channel, and byte-allocation protection weight"
            ),
            "arbitration_needed": True,
        },
        "training_plan": {
            "schema": "driving_prior_training_plan.v1",
            "status": "not_started",
            "pretraining_sources": [row["dataset_id"] for row in datasets["sources"]],
            "fine_tune_target": "contest_video_residual_only_after_prior_pretraining",
            "export_contract": "future scorer-free deterministic renderer/codebook archive section",
            "inflate_contract": "no PoseNet/SegNet imports or scorer weights at inflate time",
            "blockers": training_blockers,
        },
        "solver_wire_in": _solver_wire_in(),
    }
    manifest = apply_proxy_evidence_boundary(
        manifest,
        dispatch_blockers=[
            *scorer["blockers"],
            *training_blockers,
            "operator_must_supply_datasets_and_license_review_before_pretraining",
        ],
    )
    return manifest


def validate_readiness_manifest(manifest: Mapping[str, Any]) -> list[str]:
    """Return fail-closed validation errors for a readiness manifest."""

    violations: list[str] = []
    if manifest.get("schema") != SCHEMA:
        violations.append("schema_mismatch")
    if manifest.get("source_commit") != SOURCE_COMMIT:
        violations.append("source_commit_mismatch")
    if manifest.get("lane_id") != LANE_ID:
        violations.append("lane_id_mismatch")
    violations.extend(validate_proxy_candidate(manifest))
    for key in (
        "research_only",
        "gpu_required",
        "gpu_used",
        "network_access_attempted",
        "downloads_attempted",
        "training_started",
        "archive_materialized",
    ):
        value = manifest.get(key)
        if key == "research_only":
            if value is not True:
                violations.append("research_only_must_be_true")
        elif value is not False:
            violations.append(f"{key}_must_be_false")
    scorer = manifest.get("scorer_readiness")
    if not isinstance(scorer, Mapping):
        violations.append("scorer_readiness_missing")
    elif scorer.get("inflate_time_scorer_load_allowed") is not False:
        violations.append("inflate_time_scorer_load_allowed_must_be_false")
    datasets = manifest.get("dataset_pretraining_plan")
    if not isinstance(datasets, Mapping):
        violations.append("dataset_pretraining_plan_missing")
    else:
        if datasets.get("downloads_attempted") is not False:
            violations.append("dataset_plan_downloads_attempted_must_be_false")
        if datasets.get("network_access_attempted") is not False:
            violations.append("dataset_plan_network_access_attempted_must_be_false")
        for row in datasets.get("sources", []):
            if not isinstance(row, Mapping):
                violations.append("dataset_source_row_not_mapping")
                continue
            if row.get("download_attempted") is not False:
                violations.append(f"dataset_{row.get('dataset_id')}_download_attempted_must_be_false")
            if row.get("network_access_attempted") is not False:
                violations.append(f"dataset_{row.get('dataset_id')}_network_attempted_must_be_false")
    budget = manifest.get("archive_budget_estimate")
    if not isinstance(budget, Mapping):
        violations.append("archive_budget_estimate_missing")
    elif budget.get("estimate_only_no_archive_materialized") is not True:
        violations.append("archive_budget_must_be_estimate_only")
    saliency = manifest.get("saliency_probe_plan")
    if not isinstance(saliency, Mapping):
        violations.append("saliency_probe_plan_missing")
    elif saliency.get("inflate_time_scorer_load_allowed") is not False:
        violations.append("saliency_inflate_time_scorer_load_allowed_must_be_false")
    return ordered_unique(violations)


__all__ = [
    "DEFAULT_ARCHIVE_BUDGET",
    "DEFAULT_DATASET_SOURCES",
    "DEFAULT_HOOK_TARGETS",
    "LANE_ID",
    "SCHEMA",
    "SOURCE_COMMIT",
    "TOOL",
    "ArchiveBudgetComponent",
    "DatasetPretrainingSource",
    "DrivingPriorReadinessError",
    "PenultimateHookTarget",
    "build_dataset_pretraining_plan",
    "build_driving_prior_readiness_manifest",
    "check_penultimate_hook_targets",
    "estimate_archive_budget",
    "inspect_scorer_readiness",
    "validate_readiness_manifest",
]
