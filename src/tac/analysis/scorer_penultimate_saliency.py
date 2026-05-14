# SPDX-License-Identifier: MIT
"""Analysis-time scorer penultimate-feature saliency probes.

This module is intentionally proxy-only. It may inspect scorer architectures
and run CPU smoke forwards during analysis/compress-time, but it must never be
imported by a submission inflate runtime and never creates score authority.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from tac.optimization.proxy_candidate_contract import (
    apply_proxy_evidence_boundary,
    ordered_unique,
    validate_proxy_candidate,
)

SCHEMA = "tac_scorer_penultimate_saliency.v1"
TOOL = "tac.analysis.scorer_penultimate_saliency"
INFLATE_TIME_USE_ALLOWED = False
DEFAULT_LANE_ID = "cooperative_receiver_scorer_penultimate_saliency"


@dataclass(frozen=True)
class PenultimateHookTarget:
    """One known scorer module to inspect with an analysis-time hook."""

    target_id: str
    scorer_id: str
    module_path: str
    required: bool
    saliency_role: str
    expected_feature: str


DEFAULT_HOOK_TARGETS: tuple[PenultimateHookTarget, ...] = (
    PenultimateHookTarget(
        target_id="posenet_summarizer",
        scorer_id="posenet",
        module_path="summarizer",
        required=True,
        saliency_role="pose_penultimate_feature_saliency",
        expected_feature="PoseNet 512-d summary before Hydra pose heads",
    ),
    PenultimateHookTarget(
        target_id="posenet_hydra_resblock",
        scorer_id="posenet",
        module_path="hydra.resblock",
        required=False,
        saliency_role="pose_head_prelinear_feature_saliency",
        expected_feature="PoseNet Hydra residual block before per-head linear layers",
    ),
    PenultimateHookTarget(
        target_id="segnet_decoder",
        scorer_id="segnet",
        module_path="decoder",
        required=True,
        saliency_role="segmentation_penultimate_decoder_saliency",
        expected_feature="SegNet UNet decoder map before segmentation head logits",
    ),
    PenultimateHookTarget(
        target_id="segnet_encoder",
        scorer_id="segnet",
        module_path="encoder",
        required=False,
        saliency_role="segmentation_encoder_world_feature_saliency",
        expected_feature="SegNet EfficientNet encoder pyramid before UNet decoder",
    ),
)


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def flatten_tensors(value: Any, prefix: str = "out") -> list[tuple[str, torch.Tensor]]:
    """Flatten tensor leaves from common scorer output containers."""

    if isinstance(value, torch.Tensor):
        return [(prefix, value)]
    if isinstance(value, Mapping):
        rows: list[tuple[str, torch.Tensor]] = []
        for key, child in sorted(value.items(), key=lambda item: str(item[0])):
            rows.extend(flatten_tensors(child, f"{prefix}.{key}"))
        return rows
    if isinstance(value, (list, tuple)):
        rows = []
        for idx, child in enumerate(value):
            rows.extend(flatten_tensors(child, f"{prefix}.{idx}"))
        return rows
    return []


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    """Return deterministic JSON-safe norm stats for a tensor."""

    detached = tensor.detach().to(device="cpu", dtype=torch.float64)
    numel = int(detached.numel())
    finite = bool(torch.isfinite(detached).all().item()) if numel else True
    row: dict[str, Any] = {
        "shape": list(detached.shape),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "numel": numel,
        "finite": finite,
    }
    if numel == 0:
        row.update(
            {
                "mean": 0.0,
                "mean_abs": 0.0,
                "max_abs": 0.0,
                "l2_norm": 0.0,
                "rms": 0.0,
            }
        )
        return row
    abs_value = detached.abs()
    row.update(
        {
            "mean": float(detached.mean().item()),
            "mean_abs": float(abs_value.mean().item()),
            "max_abs": float(abs_value.max().item()),
            "l2_norm": float(torch.linalg.vector_norm(detached).item()),
            "rms": float(torch.sqrt(torch.mean(detached.square())).item()),
        }
    )
    return row


def _named_modules(model: Any) -> dict[str, Any]:
    if hasattr(model, "named_modules"):
        return {str(name): module for name, module in model.named_modules()}
    return {}


def resolve_module(model: Any, module_path: str) -> Any | None:
    """Resolve a module path via ``named_modules`` first, then attributes."""

    named = _named_modules(model)
    if module_path in named:
        return named[module_path]
    cur = model
    for part in module_path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


@contextmanager
def _temporarily_frozen(models: Mapping[str, nn.Module]):
    original: list[tuple[torch.nn.Parameter, bool]] = []
    for model in models.values():
        for param in model.parameters(recurse=True):
            original.append((param, bool(param.requires_grad)))
            param.requires_grad_(False)
    try:
        yield
    finally:
        for param, requires_grad in original:
            param.requires_grad_(requires_grad)


def _default_loss_from_outputs(outputs: Mapping[str, Any]) -> torch.Tensor | None:
    terms: list[torch.Tensor] = []
    for output in outputs.values():
        for _key, tensor in flatten_tensors(output):
            if tensor.is_floating_point():
                terms.append(tensor.float().square().mean())
    if not terms:
        return None
    total = terms[0]
    for term in terms[1:]:
        total = total + term
    return total


def _stats_l2_total(rows: Sequence[dict[str, Any]]) -> float:
    return math.sqrt(sum(float(row.get("l2_norm", 0.0)) ** 2 for row in rows))


def _proxy_wrap_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return apply_proxy_evidence_boundary(
        {
            **dict(row),
            "analysis_time_only": True,
            "inflate_time_use_allowed": INFLATE_TIME_USE_ALLOWED,
            "inflate_time_scorer_load_allowed": False,
            "score_path": "not_run",
            "score_claim_axis": "none",
        },
        dispatch_blockers=(
            "analysis_time_scorer_hook_only",
            "byte_closed_archive_missing",
            "exact_eval_not_run",
            "inflate_time_scorer_load_forbidden",
        ),
    )


def run_penultimate_saliency_probe(
    models: Mapping[str, nn.Module],
    inputs: Mapping[str, torch.Tensor],
    *,
    targets: Sequence[PenultimateHookTarget] = DEFAULT_HOOK_TARGETS,
    compute_gradients: bool = True,
    loss_fn: Callable[[Mapping[str, Any]], torch.Tensor | None] | None = None,
) -> dict[str, Any]:
    """Attach hooks, run one forward/backward smoke, and summarize saliency.

    The probe computes gradients with respect to analysis inputs and captured
    activations only. Parameters are temporarily frozen and restored.
    """

    hook_rows: list[dict[str, Any]] = []
    handles: list[Any] = []
    captures: dict[str, list[tuple[str, torch.Tensor]]] = {
        target.target_id: [] for target in targets
    }

    for target in targets:
        model = models.get(target.scorer_id)
        row: dict[str, Any] = {
            **asdict(target),
            "model_present": model is not None,
            "module_present": False,
            "hook_registerable": False,
            "activation_seen": False,
            "gradient_requested": compute_gradients,
            "gradient_available": False,
            "status": "model_missing",
        }
        if model is None:
            hook_rows.append(row)
            continue
        module = resolve_module(model, target.module_path)
        row["module_present"] = module is not None
        if module is None:
            row["status"] = "module_missing"
            hook_rows.append(row)
            continue

        def _capture_hook(
            _module: nn.Module,
            _inputs: tuple[Any, ...],
            output: Any,
            *,
            target_id: str = target.target_id,
        ) -> None:
            for key, tensor in flatten_tensors(output):
                if compute_gradients and tensor.requires_grad:
                    tensor.retain_grad()
                captures[target_id].append((key, tensor))

        try:
            handle = module.register_forward_hook(_capture_hook)
        except Exception as exc:  # pragma: no cover - dependency-specific
            row["status"] = "hook_register_failed"
            row["hook_error"] = f"{type(exc).__name__}: {exc}"
        else:
            row["hook_registerable"] = True
            row["status"] = "hook_registered"
            handles.append(handle)
        hook_rows.append(row)

    outputs: dict[str, Any] = {}
    forward_errors: list[str] = []
    backward_error = ""
    backward_ran = False
    loss_value: float | None = None

    try:
        with _temporarily_frozen(models):
            for scorer_id in sorted(models):
                model = models[scorer_id]
                model.eval()
                source = inputs.get(scorer_id)
                if source is None:
                    forward_errors.append(f"{scorer_id}:missing_input")
                    continue
                x = source.detach().clone()
                if not x.is_floating_point():
                    x = x.float()
                x.requires_grad_(compute_gradients)
                try:
                    outputs[scorer_id] = model(x)
                except Exception as exc:  # pragma: no cover - caller/model-specific
                    forward_errors.append(f"{scorer_id}:{type(exc).__name__}:{exc}")

            loss = loss_fn(outputs) if loss_fn is not None else _default_loss_from_outputs(outputs)
            if loss is not None:
                loss_value = float(loss.detach().to("cpu").item())
            if compute_gradients and loss is not None and loss.requires_grad:
                loss.backward()
                backward_ran = True
            elif compute_gradients:
                backward_error = "loss_missing_or_not_differentiable"
    finally:
        for handle in handles:
            if hasattr(handle, "remove"):
                handle.remove()

    finalized_rows: list[dict[str, Any]] = []
    for row in hook_rows:
        captured = captures.get(str(row["target_id"]), [])
        activation_stats = [
            {
                "tensor_key": key,
                **tensor_stats(tensor),
                "requires_grad": bool(tensor.requires_grad),
            }
            for key, tensor in captured
        ]
        grad_stats = []
        for key, tensor in captured:
            grad = getattr(tensor, "grad", None)
            if grad is not None:
                grad_stats.append({"tensor_key": key, **tensor_stats(grad)})
        activation_l2 = _stats_l2_total(activation_stats)
        grad_l2 = _stats_l2_total(grad_stats)
        row.update(
            {
                "activation_seen": bool(captured),
                "activation_tensors": activation_stats,
                "activation_l2_total": activation_l2,
                "activation_rms_max": max(
                    (float(item.get("rms", 0.0)) for item in activation_stats),
                    default=0.0,
                ),
                "backward_ran": backward_ran,
                "backward_error": backward_error,
                "gradient_available": bool(grad_stats),
                "gradient_tensors": grad_stats,
                "gradient_l2_total": grad_l2,
                "gradient_nonzero": grad_l2 > 0.0,
            }
        )
        if row["hook_registerable"] and row["activation_seen"] and row["gradient_available"]:
            row["status"] = "activation_and_gradient_captured"
        elif row["hook_registerable"] and row["activation_seen"]:
            row["status"] = "activation_captured_no_gradient"
        elif row["hook_registerable"]:
            row["status"] = "hook_registered_no_activation"
        finalized_rows.append(_proxy_wrap_row(row))

    required_rows = [row for row in finalized_rows if row.get("required")]
    return {
        "hook_rows": finalized_rows,
        "required_hook_modules_available": all(
            bool(row.get("module_present")) and bool(row.get("hook_registerable"))
            for row in required_rows
        ),
        "required_activations_seen": all(bool(row.get("activation_seen")) for row in required_rows),
        "gradient_support_available": any(bool(row.get("gradient_available")) for row in finalized_rows),
        "backward_ran": backward_ran,
        "backward_error": backward_error,
        "loss_value": loss_value,
        "forward_errors": forward_errors,
    }


class _TinyHydra(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.resblock = nn.Sequential(nn.Linear(8, 8), nn.ReLU(), nn.Linear(8, 8), nn.ReLU())
        self.final_layer = nn.Linear(8, 6)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.resblock(x)
        return {"pose": self.final_layer(hidden)}


class _TinyPoseNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.summarizer = nn.Sequential(nn.Linear(8, 16), nn.Tanh(), nn.Linear(16, 8), nn.ReLU())
        self.hydra = _TinyHydra()

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.hydra(self.summarizer(x))


class _TinySegNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.ReLU())
        self.decoder = nn.Sequential(nn.Conv2d(4, 4, 3, padding=1), nn.ReLU())
        self.segmentation_head = nn.Conv2d(4, 5, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.segmentation_head(self.decoder(self.encoder(x)))


def build_synthetic_scorer_models(seed: int = 20260513) -> dict[str, nn.Module]:
    """Build tiny CPU scorers with the same hook paths as upstream scorers."""

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        models: dict[str, nn.Module] = {
            "posenet": _TinyPoseNet().eval(),
            "segnet": _TinySegNet().eval(),
        }
    return models


def build_synthetic_inputs(seed: int = 20260513, batch_size: int = 2) -> dict[str, torch.Tensor]:
    """Return deterministic CPU inputs for the tiny scorer smoke."""

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1)
    return {
        "posenet": torch.randn((batch_size, 8), generator=generator),
        "segnet": torch.randn((batch_size, 3, 8, 8), generator=generator),
    }


def build_proxy_safe_manifest(
    probe: Mapping[str, Any],
    *,
    repo_root: str | Path | None = None,
    mode: str,
    seed: int,
    lane_id: str = DEFAULT_LANE_ID,
) -> dict[str, Any]:
    """Wrap a probe result in the canonical proxy false-authority contract."""

    hook_rows = list(probe.get("hook_rows", []))
    required_modules = bool(probe.get("required_hook_modules_available"))
    required_activations = bool(probe.get("required_activations_seen"))
    gradient_support = bool(probe.get("gradient_support_available"))
    manifest = {
        "schema": SCHEMA,
        "schema_version": 1,
        "tool": TOOL,
        "generated_at_utc": _utc_now(),
        "repo_root": str(Path(repo_root).resolve()) if repo_root is not None else "",
        "lane_id": lane_id,
        "campaign_id": "cooperative_receiver_score_lowering",
        "mode": mode,
        "seed": seed,
        "device_policy": "cpu_only_no_cuda_no_mps",
        "analysis_time_only": True,
        "compress_time_allowed": True,
        "inflate_time_use_allowed": INFLATE_TIME_USE_ALLOWED,
        "inflate_time_scorer_load_allowed": False,
        "inflate_time_policy": "forbidden: scorer hooks are analysis-time saliency only",
        "research_only": True,
        "network_access_attempted": False,
        "gpu_used": False,
        "dispatch_attempted": False,
        "score_path": "not_run",
        "score_claim_axis": "none",
        "byte_closed_archive_materialized": False,
        "exact_eval_run": False,
        "hook_rows": hook_rows,
        "required_hook_modules_available": required_modules,
        "required_activations_seen": required_activations,
        "gradient_support_available": gradient_support,
        "saliency_ready_for_proxy_planning": required_activations and gradient_support,
        "backward_ran": bool(probe.get("backward_ran")),
        "backward_error": str(probe.get("backward_error", "")),
        "loss_value": probe.get("loss_value"),
        "forward_errors": list(probe.get("forward_errors", [])),
        "solver_wire_in": {
            "research_only": True,
            "sensitivity_map_contribution": "scorer_penultimate_feature_saliency_proxy",
            "pareto_constraint": "non_binding_until_byte_closed_archive_and_exact_eval",
            "bit_allocator_hook": "proxy saliency weights only; archive grammar still missing",
            "cathedral_autopilot_dispatch_hook": "blocked: proxy row is not exact-ready",
            "continual_learning_posterior_update": "disabled: no empirical anchor",
            "probe_disambiguator": (
                "emits PoseNet summarizer/Hydra and SegNet decoder/encoder "
                "rows so later byte-closed probes can arbitrate target choice"
            ),
        },
    }
    blockers = [
        "byte_closed_archive_missing",
        "exact_eval_not_run",
        "analysis_time_scorer_hook_only",
        "inflate_time_scorer_load_forbidden",
    ]
    if not required_modules:
        blockers.append("required_penultimate_hook_module_missing")
    if not required_activations:
        blockers.append("required_penultimate_activation_missing")
    if not gradient_support:
        blockers.append("penultimate_gradient_support_missing")
    if manifest["forward_errors"]:
        blockers.append("forward_errors_present")
    return apply_proxy_evidence_boundary(manifest, dispatch_blockers=ordered_unique(blockers))


def build_synthetic_smoke_manifest(
    *,
    seed: int = 20260513,
    batch_size: int = 2,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    """Run the deterministic tiny CPU smoke and return a proxy-safe manifest."""

    models = build_synthetic_scorer_models(seed=seed)
    inputs = build_synthetic_inputs(seed=seed, batch_size=batch_size)
    probe = run_penultimate_saliency_probe(models, inputs)
    return build_proxy_safe_manifest(
        probe,
        repo_root=repo_root,
        mode="synthetic_cpu_smoke",
        seed=seed,
    )


def validate_penultimate_saliency_manifest(manifest: Mapping[str, Any]) -> list[str]:
    """Validate proxy-safety and analysis-only guardrails."""

    violations = validate_proxy_candidate(manifest)
    if manifest.get("analysis_time_only") is not True:
        violations.append("analysis_time_only_must_be_true")
    if manifest.get("inflate_time_use_allowed") is not False:
        violations.append("inflate_time_use_allowed_must_be_false")
    if manifest.get("inflate_time_scorer_load_allowed") is not False:
        violations.append("inflate_time_scorer_load_allowed_must_be_false")
    if manifest.get("exact_eval_run") is not False:
        violations.append("exact_eval_run_must_be_false")
    for idx, row in enumerate(manifest.get("hook_rows", [])):
        for violation in validate_proxy_candidate(row):
            violations.append(f"hook_rows[{idx}].{violation}")
        if row.get("analysis_time_only") is not True:
            violations.append(f"hook_rows[{idx}].analysis_time_only_must_be_true")
        if row.get("inflate_time_use_allowed") is not False:
            violations.append(f"hook_rows[{idx}].inflate_time_use_allowed_must_be_false")
    return violations


__all__ = [
    "DEFAULT_HOOK_TARGETS",
    "DEFAULT_LANE_ID",
    "INFLATE_TIME_USE_ALLOWED",
    "SCHEMA",
    "TOOL",
    "PenultimateHookTarget",
    "build_proxy_safe_manifest",
    "build_synthetic_inputs",
    "build_synthetic_scorer_models",
    "build_synthetic_smoke_manifest",
    "flatten_tensors",
    "resolve_module",
    "run_penultimate_saliency_probe",
    "tensor_stats",
    "validate_penultimate_saliency_manifest",
]
