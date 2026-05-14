#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""CUDA component sensitivity producer for beta/OWV3/NWCS promotion gates.

This profiler emits the artifacts consumed by
``experiments/build_component_sensitivity_manifest.py``:

- PoseNet, SegNet, and combined per-channel sensitivity maps.
- PoseNet, SegNet, and combined response-curve JSON files.
- A stability JSON file and deterministic sample-plan JSON.

The maps are empirical Fisher proxies or direct-renderer finite-difference
diagnostics, not score evidence. Current outputs are diagnostic-only and
deliberately cannot assemble a promotable ``component_sensitivity_v1``
manifest until finite-difference response validation runs through the
canonical ``archive.zip -> inflate.sh -> upstream/evaluate.py`` path.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

try:
    from tools.tool_bootstrap import ensure_repo_imports, prepend_paths, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover - direct script execution
    _bootstrap_path = Path(__file__).resolve().parent.parent / "tools" / "tool_bootstrap.py"
    _spec = importlib.util.spec_from_file_location("tool_bootstrap", _bootstrap_path)
    if _spec is None or _spec.loader is None:
        raise
    _tool_bootstrap = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_tool_bootstrap)
    ensure_repo_imports = _tool_bootstrap.ensure_repo_imports
    prepend_paths = _tool_bootstrap.prepend_paths
    repo_root_from_tool = _tool_bootstrap.repo_root_from_tool

REPO = repo_root_from_tool(__file__)
ensure_repo_imports(REPO)
prepend_paths(REPO / "upstream")

from experiments.convert_fisher_to_owv3_sensitivity_map import (
    convert_importance_to_channel_sensitivity,
)
from experiments.profile_hessian_per_weight import (
    _decode_gt_video,
    _load_masks_video,
    _load_pair_weights,
    _load_poses,
    _load_renderer,
    _select_eligible_params_with_exclusions,
)
from tac.component_sensitivity_artifact import (
    CONTEST_SAMPLE_COUNT,
    write_component_sensitivity_manifest,
)
from tac.repo_io import json_text, read_json, write_json
from tac.sensitivity_map import (
    load_sensitivity_map,
    save_sensitivity_map,
    sensitivity_cv_distance,
)

COMPONENT_OUTPUTS = ("posenet", "segnet", "combined")
SCORE_EPS = 1e-12
DEFAULT_RESPONSE_EPSILONS = [-0.002, -0.001, -0.0005, 0.0, 0.0005, 0.001, 0.002]
DEFAULT_FINITE_DIFFERENCE_EPSILON = 0.001
PERTURBATION_BASIS_FORMAT = "perturbation_basis_v1"
FINITE_DIFFERENCE_SHARD_SCHEMA = "component_sensitivity_direct_fd_shard_v1"
FINITE_DIFFERENCE_MERGE_SCHEMA = "component_sensitivity_direct_fd_merge_v1"
CONTEST_FRAME_COUNT = CONTEST_SAMPLE_COUNT * 2
OFFICIAL_COMPONENT_READOUTS = {
    "posenet": "official_pose_mse",
    "segnet": "official_argmax_disagreement",
    "combined": "official_component_formula",
}
RESPONSE_GATE_SPEC = {
    "zero_repro_tolerance": 1e-7,
    "observed_delta_min": 1e-12,
    "holdout_error_max": 0.35,
    "spearman_min": 0.30,
    "top_decile_overlap_min": 0.50,
}
STABILITY_THRESHOLDS = {
    "cv_max": 0.35,
    "spearman_min": 0.30,
    "pearson_min": 0.0,
    "top_decile_overlap_min": 0.50,
}
FISHER_PROXY_PROMOTION_BLOCKER = {
    "code": "fisher_proxy_not_official_component_response",
    "mathematical_explanation": (
        "This profiler estimates component sensitivity with empirical Fisher "
        "proxies and holdout perturbation diagnostics. It is not yet an "
        "official finite-difference component-response artifact over the exact "
        "contest scorer, so it cannot be used as promotion evidence."
    ),
}
CANONICAL_SCORER_PATH_PROMOTION_BLOCKER = {
    "code": "not_canonical_inflate_eval_path",
    "mathematical_explanation": (
        "This profiler measures finite differences by rendering float tensors "
        "in process and calling scorer modules directly. Promotion-grade "
        "component sensitivity must perturb exact archive bytes and measure "
        "component response through archive.zip -> inflate.sh -> "
        "upstream/evaluate.py."
    ),
}


class ComponentSensitivityProfileError(ValueError):
    """Raised when a component sensitivity profile cannot be produced."""


def _require_device(device: str, *, allow_diagnostic_cpu: bool) -> torch.device:
    if device == "mps":
        raise SystemExit(
            "FATAL: --device mps forbidden. MPS scorer/sensitivity drift is "
            "not contest-grade."
        )
    if device == "cpu" and not allow_diagnostic_cpu:
        raise SystemExit(
            "FATAL: --device cpu is diagnostic-only. Pass "
            "--allow-diagnostic-cpu to make that non-promotable status "
            "explicit, or run --device cuda."
        )
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("FATAL: --device cuda requested but CUDA is unavailable")
    return torch.device(device)


def _pair_record(pair_index: int) -> dict[str, int]:
    return {
        "video": 0,
        "pair_index": int(pair_index),
        "t": int(2 * pair_index),
        "t1": int(2 * pair_index + 1),
    }


def make_sample_plan(
    *,
    n_pairs: int,
    split_seed: int,
    holdout_fraction: float,
) -> dict[str, Any]:
    return make_sample_plan_for_indices(
        pair_indices=range(n_pairs),
        split_seed=split_seed,
        holdout_fraction=holdout_fraction,
    )


def make_sample_plan_for_indices(
    *,
    pair_indices: list[int] | range,
    split_seed: int,
    holdout_fraction: float,
) -> dict[str, Any]:
    indices = sorted(int(i) for i in pair_indices)
    n_pairs = len(indices)
    if n_pairs <= 1:
        raise ComponentSensitivityProfileError("pair_indices must contain > 1 pair")
    if not (0.0 < holdout_fraction < 1.0):
        raise ComponentSensitivityProfileError("holdout_fraction must be in (0, 1)")
    if len(set(indices)) != len(indices):
        raise ComponentSensitivityProfileError("pair_indices must be unique")
    indices.sort(key=lambda idx: hashlib.sha256(f"{split_seed}:{idx}".encode("ascii")).hexdigest())
    n_holdout = max(1, min(n_pairs - 1, round(n_pairs * holdout_fraction)))
    holdout = sorted(indices[:n_holdout])
    calibration = sorted(indices[n_holdout:])
    calibration_records = [_pair_record(i) for i in calibration]
    holdout_records = [_pair_record(i) for i in holdout]
    split_hash = hashlib.sha256(
        json.dumps(
            {
                "calibration_pairs": calibration_records,
                "holdout_pairs": holdout_records,
                "split_seed": int(split_seed),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return {
        "calibration_pairs": calibration_records,
        "holdout_pairs": holdout_records,
        "split_seed": int(split_seed),
        "split_hash": split_hash,
    }


def component_score_terms(
    pose_dist_per_pair: torch.Tensor,
    seg_proxy_per_pair: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return per-pair score-contribution tensors for each component."""
    if pose_dist_per_pair.shape != seg_proxy_per_pair.shape:
        raise ComponentSensitivityProfileError(
            "pose and SegNet proxy tensors must have the same per-pair shape"
        )
    posenet = torch.sqrt(10.0 * pose_dist_per_pair.clamp_min(0.0) + SCORE_EPS)
    segnet = 100.0 * seg_proxy_per_pair
    combined = posenet + segnet
    for name, value in (("posenet", posenet), ("segnet", segnet), ("combined", combined)):
        if not torch.isfinite(value).all():
            raise ComponentSensitivityProfileError(f"{name} score term is non-finite")
    return {"posenet": posenet, "segnet": segnet, "combined": combined}


def component_formula_from_mean_distortions(
    *,
    pose_dist: float,
    seg_dist: float,
) -> dict[str, float]:
    """Return official component readouts and combined no-rate contribution."""
    pose = float(pose_dist)
    seg = float(seg_dist)
    if not math.isfinite(pose) or pose < 0.0:
        raise ComponentSensitivityProfileError(
            f"pose_dist must be finite/nonnegative, got {pose!r}"
        )
    if not math.isfinite(seg) or seg < 0.0:
        raise ComponentSensitivityProfileError(
            f"seg_dist must be finite/nonnegative, got {seg!r}"
        )
    return {
        "posenet": pose,
        "segnet": seg,
        "combined": 100.0 * seg + math.sqrt(10.0 * pose),
    }


def _require_full_contest_frame_count(
    *,
    n_frames: int,
    n_mask_frames: int,
    promotion_finite_difference: bool,
) -> None:
    if int(n_mask_frames) != int(n_frames):
        raise SystemExit("FATAL: masks/video frame counts do not match")
    if not promotion_finite_difference:
        return
    if int(n_frames) != CONTEST_FRAME_COUNT:
        raise SystemExit(
            "FATAL: promotion finite-difference sensitivity requires exact "
            f"{CONTEST_FRAME_COUNT} contest frames "
            f"({CONTEST_SAMPLE_COUNT} pairs), got {int(n_frames)}"
        )


def _weighted_component_losses(
    terms: Mapping[str, torch.Tensor],
    weights: torch.Tensor,
) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name in COMPONENT_OUTPUTS:
        out[name] = (terms[name] * weights).sum()
    return out


def accumulate_component_fisher(
    losses: Mapping[str, torch.Tensor],
    params: Mapping[str, torch.nn.Parameter],
    accumulators: dict[str, dict[str, torch.Tensor]],
) -> None:
    """Accumulate squared gradients for each component loss."""
    param_items = list(params.items())
    names = [name for name, _param in param_items]
    values = [param for _name, param in param_items]
    for index, component in enumerate(COMPONENT_OUTPUTS):
        grads = torch.autograd.grad(
            losses[component],
            values,
            retain_graph=index < len(COMPONENT_OUTPUTS) - 1,
            allow_unused=True,
        )
        for name, grad in zip(names, grads):
            if grad is None:
                continue
            accumulators[component][name] += grad.detach().pow(2).to(torch.float64).cpu()


def _aggregate_maps(
    *,
    model: torch.nn.Module,
    importance_by_component: Mapping[str, Mapping[str, torch.Tensor]],
    aggregate: str,
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        component: convert_importance_to_channel_sensitivity(
            model=model,
            importance=importance_by_component[component],
            aggregate=aggregate,
            missing_policy="error",
            protected_missing_policy="error",
        )
        for component in COMPONENT_OUTPUTS
    }


def _conv_channel_sensitivity_skeleton(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    out: dict[str, torch.Tensor] = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            out[f"{name}.weight"] = torch.zeros(
                int(module.weight.shape[0]),
                dtype=torch.float32,
            )
    if not out:
        raise ComponentSensitivityProfileError("model has no Conv2d weights")
    return out


def _channel_ref_payload(refs: list[tuple[str, int]]) -> list[dict[str, Any]]:
    return [{"key": key, "channel": int(channel)} for key, channel in refs]


def _channel_ref_sha256(refs: list[tuple[str, int]]) -> str:
    return hashlib.sha256(
        json.dumps(
            _channel_ref_payload(refs),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _all_conv_channel_refs(model: torch.nn.Module) -> list[tuple[str, int]]:
    refs: list[tuple[str, int]] = []
    for key, tensor in sorted(_conv_channel_sensitivity_skeleton(model).items()):
        refs.extend((key, channel) for channel in range(int(tensor.numel())))
    if not refs:
        raise ComponentSensitivityProfileError("no Conv2d channels available for sharding")
    return refs


def _finite_difference_shard_plan(
    model: torch.nn.Module,
    *,
    shard_index: int,
    shard_count: int,
) -> dict[str, Any]:
    count = int(shard_count)
    index = int(shard_index)
    if count <= 0:
        raise ComponentSensitivityProfileError("finite_difference_shard_count must be positive")
    if index < 0 or index >= count:
        raise ComponentSensitivityProfileError(
            f"finite_difference_shard_index must be in [0, {count}), got {index}"
        )
    all_refs = _all_conv_channel_refs(model)
    start = len(all_refs) * index // count
    end = len(all_refs) * (index + 1) // count
    assigned = all_refs[start:end]
    if not assigned:
        raise ComponentSensitivityProfileError(
            f"finite-difference shard {index}/{count} has no assigned channels"
        )
    return {
        "schema": FINITE_DIFFERENCE_SHARD_SCHEMA,
        "is_shard": count > 1,
        "shard_index": index,
        "shard_count": count,
        "assigned_channel_count": len(assigned),
        "all_channel_count": len(all_refs),
        "assigned_channel_refs": _channel_ref_payload(assigned),
        "all_channel_refs": _channel_ref_payload(all_refs),
        "all_channel_sha256": _channel_ref_sha256(all_refs),
        "assigned_channel_sha256": _channel_ref_sha256(assigned),
        "partition": "contiguous_sorted_conv_channel_refs_v1",
        "merge_required_for_certification_handoff": count > 1,
    }


def _refs_from_payload(items: Any, *, label: str) -> list[tuple[str, int]]:
    if not isinstance(items, list):
        raise ComponentSensitivityProfileError(f"{label} must be a list")
    refs: list[tuple[str, int]] = []
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            raise ComponentSensitivityProfileError(f"{label}[{index}] must be an object")
        key = item.get("key")
        channel = item.get("channel")
        if not isinstance(key, str) or not key.endswith(".weight"):
            raise ComponentSensitivityProfileError(f"{label}[{index}].key is not canonical")
        if isinstance(channel, bool) or not isinstance(channel, int):
            raise ComponentSensitivityProfileError(f"{label}[{index}].channel must be an integer")
        refs.append((key, int(channel)))
    return refs


def _validate_channel_refs_for_model(
    model: torch.nn.Module,
    refs: list[tuple[str, int]],
) -> list[tuple[str, int]]:
    all_refs = set(_all_conv_channel_refs(model))
    out = sorted((str(key), int(channel)) for key, channel in refs)
    if len(out) != len(set(out)):
        raise ComponentSensitivityProfileError("finite-difference channel refs contain duplicates")
    missing = [ref for ref in out if ref not in all_refs]
    if missing:
        raise ComponentSensitivityProfileError(f"finite-difference channel refs not in model: {missing[:5]}")
    return out


def _finite_difference_component_channel_maps(
    *,
    model: torch.nn.Module,
    pair_indices: list[int],
    baseline: Mapping[str, float],
    masks_cpu: torch.Tensor,
    gt_frames_cpu: torch.Tensor,
    poses: torch.Tensor | None,
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    device: torch.device,
    zoom_warp: torch.nn.Module | None,
    pair_batch: int,
    epsilon: float,
    channel_refs: list[tuple[str, int]] | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Measure direct-renderer component response by central perturbation.

    The output units match the response model used by the perturbation curves:
    mean absolute component delta per ``epsilon**2`` for a signed per-channel
    RMS perturbation. This is intentionally slow and explicit, but still does
    not materialize archive bytes or run the canonical inflate/evaluate path.
    """

    eps = float(epsilon)
    if not math.isfinite(eps) or eps <= 0.0:
        raise ComponentSensitivityProfileError("finite-difference epsilon must be positive")
    if not pair_indices:
        raise ComponentSensitivityProfileError("finite-difference pair split is empty")

    maps = {
        component: _conv_channel_sensitivity_skeleton(model)
        for component in COMPONENT_OUTPUTS
    }
    refs = (
        _validate_channel_refs_for_model(model, channel_refs)
        if channel_refs is not None
        else _all_conv_channel_refs(model)
    )
    for key, channel in refs:
        values_by_eps: dict[float, dict[str, float]] = {}
        for signed_eps in (-eps, eps):
            originals = apply_channel_perturbation(
                model,
                [(key, channel, 1.0)],
                epsilon=signed_eps,
            )
            try:
                values_by_eps[signed_eps] = _evaluate_component_means(
                    model=model,
                    pair_indices=pair_indices,
                    masks_cpu=masks_cpu,
                    gt_frames_cpu=gt_frames_cpu,
                    poses=poses,
                    posenet=posenet,
                    segnet=segnet,
                    device=device,
                    zoom_warp=zoom_warp,
                    pair_batch=pair_batch,
                )
            finally:
                restore_perturbation(originals)
        for component in COMPONENT_OUTPUTS:
            delta_plus = float(values_by_eps[eps][component]) - float(baseline[component])
            delta_minus = float(values_by_eps[-eps][component]) - float(baseline[component])
            value = (abs(delta_plus) + abs(delta_minus)) / (2.0 * eps * eps)
            if not math.isfinite(value):
                raise ComponentSensitivityProfileError(
                    f"non-finite finite-difference sensitivity for {component} {key}[{channel}]"
                )
            maps[component][key][channel] = max(0.0, float(value))
    return maps


def select_top_channels(
    sensitivities: Mapping[str, torch.Tensor],
    *,
    top_k: int,
) -> list[tuple[str, int, float]]:
    if top_k <= 0:
        raise ComponentSensitivityProfileError("top_k must be positive")
    rows: list[tuple[str, int, float]] = []
    for key, tensor in sensitivities.items():
        flat = tensor.detach().to(torch.float32).cpu().reshape(-1)
        for channel, value in enumerate(flat.tolist()):
            rows.append((str(key), int(channel), float(value)))
    if not rows:
        raise ComponentSensitivityProfileError("no sensitivity channels available")
    rows.sort(key=lambda item: (-item[2], item[0], item[1]))
    return rows[: min(top_k, len(rows))]


def _atom_id(component: str, order: int, key: str, channel: int) -> str:
    digest = hashlib.sha256(
        f"{component}:{order}:{key}:{channel}".encode("utf-8")
    ).hexdigest()[:12]
    return f"{component}:{order:06d}:{digest}"


def build_perturbation_basis_payload(
    *,
    selected_by_component: Mapping[str, list[tuple[str, int, float]]],
    sample_plan: Mapping[str, Any],
    response_epsilons: list[float],
    response_top_k: int,
    input_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Materialize the deterministic perturbation basis used by response curves."""
    components: dict[str, Any] = {}
    for component in COMPONENT_OUTPUTS:
        atoms = []
        for order, (key, channel, sensitivity) in enumerate(
            selected_by_component.get(component, [])
        ):
            atoms.append(
                {
                    "atom_id": _atom_id(component, order, key, int(channel)),
                    "order": int(order),
                    "component": component,
                    "key": str(key),
                    "channel": int(channel),
                    "sensitivity": float(sensitivity),
                    "normalization": "weight_channel_rms",
                    "direction": "sign(weight), zeros treated as +1",
                }
            )
        components[component] = {
            "atom_count": len(atoms),
            "response_top_k": int(response_top_k),
            "sensitivity_sum": float(
                sum(max(0.0, float(atom["sensitivity"])) for atom in atoms)
            ),
            "atoms": atoms,
        }

    pair_index_payload = {
        "calibration_pair_indices": [
            int(item["pair_index"])
            for item in sample_plan.get("calibration_pairs", [])
        ],
        "holdout_pair_indices": [
            int(item["pair_index"])
            for item in sample_plan.get("holdout_pairs", [])
        ],
    }
    return {
        "schema_version": 1,
        "format": PERTURBATION_BASIS_FORMAT,
        "tool": "experiments/profile_component_sensitivity.py",
        "perturbation": "signed_weight_rms_topk_channels_v1",
        "epsilon_units": (
            "dimensionless multiplier on each selected channel's RMS weight"
        ),
        "epsilon_ladder": sorted(set(float(eps) for eps in response_epsilons)),
        "sign_convention": (
            "epsilon * rms(weight[channel]) * sign(weight[channel]); "
            "zero weights use +1 sign"
        ),
        "normalization": "per-channel RMS over the selected weight channel",
        "split_seed": int(sample_plan.get("split_seed", 0)),
        "split_hash": sample_plan.get("split_hash"),
        **pair_index_payload,
        "input_custody": {
            "checkpoint": input_metadata.get("checkpoint"),
            "video_mkv": input_metadata.get("video_mkv"),
            "masks_mkv": input_metadata.get("masks_mkv"),
            "poses": input_metadata.get("poses"),
            "upstream_dir": input_metadata.get("upstream_dir"),
            "pair_weights": input_metadata.get("pair_weights"),
            "n_pairs_total": input_metadata.get("n_pairs_total"),
            "n_pairs_selected": input_metadata.get("n_pairs_selected"),
        },
        "components": components,
    }


def _write_perturbation_basis_json(
    path: Path,
    *,
    selected_by_component: Mapping[str, list[tuple[str, int, float]]],
    sample_plan: Mapping[str, Any],
    response_epsilons: list[float],
    response_top_k: int,
    input_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    payload = build_perturbation_basis_payload(
        selected_by_component=selected_by_component,
        sample_plan=sample_plan,
        response_epsilons=response_epsilons,
        response_top_k=response_top_k,
        input_metadata=input_metadata,
    )
    write_json(path, payload)
    return payload


def apply_channel_perturbation(
    model: torch.nn.Module,
    selected_channels: list[tuple[str, int, float]],
    *,
    epsilon: float,
) -> list[tuple[torch.nn.Parameter, torch.Tensor]]:
    """Apply deterministic signed perturbations to selected Conv2d channels."""
    if not torch.isfinite(torch.tensor(float(epsilon))):
        raise ComponentSensitivityProfileError("epsilon must be finite")
    modules = dict(model.named_modules())
    originals: list[tuple[torch.nn.Parameter, torch.Tensor]] = []
    grouped: dict[str, set[int]] = {}
    for key, channel, _score in selected_channels:
        if not key.endswith(".weight"):
            raise ComponentSensitivityProfileError(f"non-canonical sensitivity key: {key}")
        grouped.setdefault(key[:-len(".weight")], set()).add(int(channel))

    with torch.no_grad():
        for module_name, channels in sorted(grouped.items()):
            module = modules.get(module_name)
            if module is None or not hasattr(module, "weight"):
                raise ComponentSensitivityProfileError(f"module not found for {module_name}")
            weight = module.weight
            originals.append((weight, weight.detach().clone()))
            for channel in sorted(channels):
                if channel < 0 or channel >= int(weight.shape[0]):
                    raise ComponentSensitivityProfileError(
                        f"{module_name}.weight channel {channel} out of range"
                    )
                view = weight[channel]
                scale = view.detach().float().pow(2).mean().sqrt().clamp_min(1e-8)
                sign = torch.sign(view)
                sign = torch.where(sign == 0, torch.ones_like(sign), sign)
                view.add_(float(epsilon) * scale.to(view.device, view.dtype) * sign)
    return originals


def restore_perturbation(originals: list[tuple[torch.nn.Parameter, torch.Tensor]]) -> None:
    with torch.no_grad():
        for param, original in originals:
            param.copy_(original.to(param.device, param.dtype))


def _render_official_scorer_pairs(
    *,
    model: torch.nn.Module,
    batch: list[int],
    masks_cpu: torch.Tensor,
    gt_frames_cpu: torch.Tensor,
    poses: torch.Tensor | None,
    device: torch.device,
    zoom_warp: torch.nn.Module | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not batch:
        raise ComponentSensitivityProfileError("empty pair batch")

    m_t_list = []
    m_t1_list = []
    gt_t_list = []
    gt_t1_list = []
    pose_list = [] if poses is not None else None
    for pair_index in batch:
        frame_index = pair_index * 2
        m_t_list.append(masks_cpu[frame_index])
        m_t1_list.append(masks_cpu[frame_index + 1])
        gt_t_list.append(gt_frames_cpu[frame_index])
        gt_t1_list.append(gt_frames_cpu[frame_index + 1])
        if poses is not None and pose_list is not None:
            pose_list.append(poses[pair_index])

    masks_t = torch.stack(m_t_list).to(device, dtype=torch.long)
    masks_t1 = torch.stack(m_t1_list).to(device, dtype=torch.long)
    gt_t = torch.stack(gt_t_list).to(device).float()
    gt_t1 = torch.stack(gt_t1_list).to(device).float()
    kwargs: dict[str, torch.Tensor] = {}
    if poses is not None and pose_list is not None:
        kwargs["pose"] = torch.stack(pose_list).to(device)
    if zoom_warp is not None:
        pair_indices = torch.tensor(batch, device=device)
        kwargs["ego_flow"] = zoom_warp(pair_indices, masks_t.shape[1], masks_t.shape[2])

    pairs_pred = model(masks_t, masks_t1, **kwargs).permute(0, 1, 4, 2, 3)
    bsz, seq, channels, height, width = pairs_pred.shape
    pred_flat = pairs_pred.reshape(bsz * seq, channels, height, width)
    pred_resized = F.interpolate(
        pred_flat,
        size=(384, 512),
        mode="bilinear",
        align_corners=False,
    ).reshape(bsz, seq, channels, 384, 512)
    gt_t_r = F.interpolate(gt_t, size=(384, 512), mode="bilinear", align_corners=False)
    gt_t1_r = F.interpolate(gt_t1, size=(384, 512), mode="bilinear", align_corners=False)
    gt_pair = torch.stack([gt_t_r, gt_t1_r], dim=1)
    return pred_resized, gt_pair


def _official_component_distortions_from_pairs(
    *,
    model: torch.nn.Module,
    batch: list[int],
    masks_cpu: torch.Tensor,
    gt_frames_cpu: torch.Tensor,
    poses: torch.Tensor | None,
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    device: torch.device,
    zoom_warp: torch.nn.Module | None,
) -> dict[str, torch.Tensor]:
    """Return official-style PoseNet MSE and SegNet argmax disagreement."""
    with torch.inference_mode():
        pred_resized, gt_pair = _render_official_scorer_pairs(
            model=model,
            batch=batch,
            masks_cpu=masks_cpu,
            gt_frames_cpu=gt_frames_cpu,
            poses=poses,
            device=device,
            zoom_warp=zoom_warp,
        )
        pose_out_pred = posenet(posenet.preprocess_input(pred_resized))
        pose_out_gt = posenet(posenet.preprocess_input(gt_pair))
        pose_dim = pose_out_pred["pose"].shape[-1] // 2
        pose_dist_per_pair = (
            pose_out_pred["pose"][..., :pose_dim]
            - pose_out_gt["pose"][..., :pose_dim]
        ).pow(2).mean(dim=tuple(range(1, pose_out_pred["pose"].ndim)))

        seg_logits_pred = segnet(segnet.preprocess_input(pred_resized))
        seg_logits_gt = segnet(segnet.preprocess_input(gt_pair))
        seg_diff = (seg_logits_pred.argmax(dim=1) != seg_logits_gt.argmax(dim=1)).float()
        seg_dist_per_pair = seg_diff.mean(dim=tuple(range(1, seg_diff.ndim)))

    for name, value in (("posenet", pose_dist_per_pair), ("segnet", seg_dist_per_pair)):
        if not torch.isfinite(value).all():
            raise ComponentSensitivityProfileError(
                f"official {name} component response contains non-finite values"
            )
        if (value < 0).any():
            raise ComponentSensitivityProfileError(
                f"official {name} component response contains negative values"
            )
    return {"posenet": pose_dist_per_pair, "segnet": seg_dist_per_pair}


def _component_metrics_from_pairs(
    *,
    model: torch.nn.Module,
    batch: list[int],
    masks_cpu: torch.Tensor,
    gt_frames_cpu: torch.Tensor,
    poses: torch.Tensor | None,
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    device: torch.device,
    zoom_warp: torch.nn.Module | None,
    require_grad: bool,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    if not batch:
        raise ComponentSensitivityProfileError("empty pair batch")

    m_t_list = []
    m_t1_list = []
    gt_t_list = []
    gt_t1_list = []
    pose_list = [] if poses is not None else None
    for pair_index in batch:
        frame_index = pair_index * 2
        m_t_list.append(masks_cpu[frame_index])
        m_t1_list.append(masks_cpu[frame_index + 1])
        gt_t_list.append(gt_frames_cpu[frame_index])
        gt_t1_list.append(gt_frames_cpu[frame_index + 1])
        if poses is not None and pose_list is not None:
            pose_list.append(poses[pair_index])

    masks_t = torch.stack(m_t_list).to(device, dtype=torch.long)
    masks_t1 = torch.stack(m_t1_list).to(device, dtype=torch.long)
    gt_t = torch.stack(gt_t_list).to(device).float()
    gt_t1 = torch.stack(gt_t1_list).to(device).float()
    kwargs: dict[str, torch.Tensor] = {}
    if poses is not None and pose_list is not None:
        kwargs["pose"] = torch.stack(pose_list).to(device)
    if zoom_warp is not None:
        pair_indices = torch.tensor(batch, device=device)
        kwargs["ego_flow"] = zoom_warp(pair_indices, masks_t.shape[1], masks_t.shape[2])

    context = torch.enable_grad() if require_grad else torch.inference_mode()
    with context:
        pairs_pred = model(masks_t, masks_t1, **kwargs).permute(0, 1, 4, 2, 3)
        bsz, seq, channels, height, width = pairs_pred.shape
        pred_flat = pairs_pred.reshape(bsz * seq, channels, height, width)
        pred_resized = F.interpolate(
            pred_flat,
            size=(384, 512),
            mode="bilinear",
            align_corners=False,
        ).reshape(bsz, seq, channels, 384, 512)
        gt_t_r = F.interpolate(gt_t, size=(384, 512), mode="bilinear", align_corners=False)
        gt_t1_r = F.interpolate(gt_t1, size=(384, 512), mode="bilinear", align_corners=False)
        gt_pair = torch.stack([gt_t_r, gt_t1_r], dim=1)

        pose_out_pred = posenet(posenet.preprocess_input(pred_resized))
        with torch.no_grad():
            pose_out_gt = posenet(posenet.preprocess_input(gt_pair))
            seg_target = segnet(segnet.preprocess_input(gt_pair)).argmax(dim=1)
        pose_dim = pose_out_pred["pose"].shape[-1] // 2
        pose_dist_per_pair = (
            pose_out_pred["pose"][..., :pose_dim]
            - pose_out_gt["pose"][..., :pose_dim]
        ).pow(2).mean(dim=tuple(range(1, pose_out_pred["pose"].ndim)))
        seg_logits_pred = segnet(segnet.preprocess_input(pred_resized))
        seg_proxy_per_pair = F.cross_entropy(
            seg_logits_pred,
            seg_target,
            reduction="none",
        ).mean(dim=(1, 2))
        terms = component_score_terms(pose_dist_per_pair, seg_proxy_per_pair)
    return terms, torch.ones(len(batch), device=device, dtype=torch.float32)


def _profile_split(
    *,
    model: torch.nn.Module,
    pair_indices: list[int],
    pair_weights: torch.Tensor,
    masks_cpu: torch.Tensor,
    gt_frames_cpu: torch.Tensor,
    poses: torch.Tensor | None,
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    device: torch.device,
    zoom_warp: torch.nn.Module | None,
    eligible: Mapping[str, torch.nn.Parameter],
    pair_batch: int,
) -> dict[str, dict[str, torch.Tensor]]:
    importance: dict[str, dict[str, torch.Tensor]] = {
        component: {
            name: torch.zeros_like(param, device="cpu", dtype=torch.float64)
            for name, param in eligible.items()
        }
        for component in COMPONENT_OUTPUTS
    }
    for param in model.parameters():
        param.requires_grad_(False)
    for param in eligible.values():
        param.requires_grad_(True)

    contributed = 0.0
    for start in range(0, len(pair_indices), pair_batch):
        batch = pair_indices[start:start + pair_batch]
        if not batch:
            continue
        terms, _unit = _component_metrics_from_pairs(
            model=model,
            batch=batch,
            masks_cpu=masks_cpu,
            gt_frames_cpu=gt_frames_cpu,
            poses=poses,
            posenet=posenet,
            segnet=segnet,
            device=device,
            zoom_warp=zoom_warp,
            require_grad=True,
        )
        weights = torch.tensor(
            [float(pair_weights[idx].item()) for idx in batch],
            device=device,
            dtype=torch.float32,
        )
        contributed += float(weights.sum().item())
        losses = _weighted_component_losses(terms, weights)
        accumulate_component_fisher(losses, eligible, importance)

    if contributed <= 0:
        raise ComponentSensitivityProfileError("selected split has zero pair weight")
    for component in COMPONENT_OUTPUTS:
        for name in importance[component]:
            importance[component][name] = importance[component][name] / contributed
    return importance


def _evaluate_component_means(
    *,
    model: torch.nn.Module,
    pair_indices: list[int],
    masks_cpu: torch.Tensor,
    gt_frames_cpu: torch.Tensor,
    poses: torch.Tensor | None,
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    device: torch.device,
    zoom_warp: torch.nn.Module | None,
    pair_batch: int,
) -> dict[str, float]:
    sums = {"posenet": 0.0, "segnet": 0.0}
    count = 0
    for start in range(0, len(pair_indices), pair_batch):
        batch = pair_indices[start:start + pair_batch]
        terms = _official_component_distortions_from_pairs(
            model=model,
            batch=batch,
            masks_cpu=masks_cpu,
            gt_frames_cpu=gt_frames_cpu,
            poses=poses,
            posenet=posenet,
            segnet=segnet,
            device=device,
            zoom_warp=zoom_warp,
        )
        for component in ("posenet", "segnet"):
            sums[component] += float(terms[component].detach().sum().cpu().item())
        count += len(batch)
    if count <= 0:
        raise ComponentSensitivityProfileError("cannot evaluate empty response split")
    return component_formula_from_mean_distortions(
        pose_dist=sums["posenet"] / count,
        seg_dist=sums["segnet"] / count,
    )


def _close_to_zero(value: float) -> bool:
    return abs(float(value)) <= 1e-12


def _close(a: float, b: float) -> bool:
    scale = max(1.0, abs(float(a)), abs(float(b)))
    return abs(float(a) - float(b)) <= 1e-9 * scale


def _response_epsilon_metadata(epsilons: list[float]) -> dict[str, Any]:
    values = sorted(set(float(eps) for eps in epsilons))
    if not values:
        raise ComponentSensitivityProfileError("response curve has no epsilons")
    if not all(math.isfinite(value) for value in values):
        raise ComponentSensitivityProfileError("response curve epsilons must be finite")
    positives = [value for value in values if value > 0.0 and not _close_to_zero(value)]
    negatives = [value for value in values if value < 0.0 and not _close_to_zero(value)]
    pair_count = sum(
        1
        for positive in positives
        if any(_close(negative, -positive) for negative in negatives)
    )
    if any(_close_to_zero(value) for value in values) and pair_count > 0:
        return {
            "response_kind": "symmetric",
            "epsilon_ladder": values,
            "symmetric_epsilon_pairs": int(pair_count),
        }
    return {
        "response_kind": "directional",
        "epsilon_ladder": values,
        "directional_action": {
            "action": "signed_weight_rms_topk_channel_perturbation",
            "epsilon_values": values,
        },
    }


def _finite_numeric_tree(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_finite_numeric_tree(child) for child in value.values())
    if isinstance(value, list):
        return all(_finite_numeric_tree(child) for child in value)
    return True


def _raw_prediction_strength(
    selected_channels: list[tuple[str, int, float]],
) -> float:
    return float(sum(max(0.0, float(score)) for _key, _channel, score in selected_channels))


def _prediction_enriched_points(
    points: list[dict[str, Any]],
    *,
    selected_channels: list[tuple[str, int, float]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach map-predicted response deltas and fitted holdout errors.

    The raw prediction is intentionally simple and explicit:
    ``epsilon^2 * sum(selected channel sensitivity)``. A scalar least-squares
    fit is recorded separately because Fisher proxy units are not official
    component units. The fitted metric is diagnostic, not promotion evidence.
    """
    strength = _raw_prediction_strength(selected_channels)
    enriched: list[dict[str, Any]] = []
    raw_predictions: list[float] = []
    observed_abs: list[float] = []
    nonzero_indices: list[int] = []
    for point in points:
        item = dict(point)
        epsilon = float(item.get("epsilon", 0.0))
        raw_delta = float(epsilon * epsilon * strength)
        item["prediction"] = {
            "raw_delta": raw_delta,
            "raw_model": "epsilon^2 * sum_selected_channel_sensitivity",
        }
        enriched.append(item)
        if _close_to_zero(epsilon):
            continue
        observed = abs(float(item.get("delta", 0.0)))
        if not (math.isfinite(raw_delta) and math.isfinite(observed)):
            continue
        raw_predictions.append(raw_delta)
        observed_abs.append(observed)
        nonzero_indices.append(len(enriched) - 1)

    if not raw_predictions:
        return enriched, {
            "implemented": True,
            "basis_strength": strength,
            "fit_status": "no_nonzero_prediction_points",
            "passed": False,
        }

    pred_t = torch.tensor(raw_predictions, dtype=torch.float64)
    obs_t = torch.tensor(observed_abs, dtype=torch.float64)
    denom = float((pred_t * pred_t).sum().item())
    fit_scale = float((pred_t * obs_t).sum().item() / denom) if denom > 1e-24 else 0.0
    fitted_t = pred_t * fit_scale
    abs_error_t = (fitted_t - obs_t).abs()
    rel_error_t = abs_error_t / obs_t.abs().clamp_min(1e-12)
    for idx, fitted, abs_err, rel_err in zip(
        nonzero_indices,
        fitted_t.tolist(),
        abs_error_t.tolist(),
        rel_error_t.tolist(),
    ):
        enriched[idx]["prediction"].update({
            "fit_scale": fit_scale,
            "fitted_abs_delta": float(fitted),
            "abs_error": float(abs_err),
            "relative_error": float(rel_err),
        })

    signs: list[bool] = []
    for idx, fitted in zip(nonzero_indices, fitted_t.tolist()):
        observed_delta = float(enriched[idx].get("delta", 0.0))
        predicted_sign = 0 if abs(fitted) <= 1e-12 else 1
        observed_sign = 0 if abs(observed_delta) <= 1e-12 else (1 if observed_delta > 0 else -1)
        signs.append(predicted_sign == observed_sign)

    pearson = _pearson_corr(pred_t, obs_t)
    spearman = _spearman_corr(pred_t, obs_t)
    max_relative_error = float(rel_error_t.max().item())
    metrics = {
        "implemented": True,
        "basis_strength": strength,
        "fit_status": "least_squares_scale_fit",
        "fit_scale": fit_scale,
        "point_count": len(raw_predictions),
        "max_abs_error": float(abs_error_t.max().item()),
        "mean_abs_error": float(abs_error_t.mean().item()),
        "max_relative_error": max_relative_error,
        "mean_relative_error": float(rel_error_t.mean().item()),
        "pearson": pearson,
        "spearman": spearman,
        "sign_accuracy": float(sum(1 for ok in signs if ok) / max(1, len(signs))),
        "passed": (
            max_relative_error <= RESPONSE_GATE_SPEC["holdout_error_max"]
            and spearman >= RESPONSE_GATE_SPEC["spearman_min"]
        ),
        "gate_spec": {
            "holdout_error_max": RESPONSE_GATE_SPEC["holdout_error_max"],
            "spearman_min": RESPONSE_GATE_SPEC["spearman_min"],
        },
    }
    return enriched, metrics


def _response_gate_results(
    points: list[dict[str, Any]],
    *,
    prediction_metrics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    finite_values = _finite_numeric_tree(points)
    observed_delta_max = max(
        (abs(float(point.get("delta", 0.0))) for point in points),
        default=0.0,
    )
    signal_present = (
        math.isfinite(observed_delta_max)
        and observed_delta_max >= RESPONSE_GATE_SPEC["observed_delta_min"]
    )
    zero_deltas = [
        abs(float(point.get("delta", 0.0)))
        for point in points
        if _close_to_zero(float(point.get("epsilon", float("nan"))))
    ]
    zero_repro_error = max(zero_deltas) if zero_deltas else None
    zero_repro = (
        zero_repro_error is not None
        and zero_repro_error <= RESPONSE_GATE_SPEC["zero_repro_tolerance"]
    )
    prediction_implemented = bool(
        prediction_metrics and prediction_metrics.get("implemented") is True
    )
    prediction_passed = bool(
        prediction_implemented and prediction_metrics.get("passed") is True
    )
    return {
        "finite_values": bool(finite_values),
        "observed_delta_max": observed_delta_max,
        "signal_present": bool(signal_present),
        "zero_repro": bool(zero_repro),
        "zero_repro_error": zero_repro_error,
        "prediction_error_gate_implemented": prediction_implemented,
        "prediction_error_passed": prediction_passed,
        "promotion_gate_passed": bool(
            finite_values
            and signal_present
            and zero_repro
            and prediction_passed
        ),
    }


def _write_response_curve(
    path: Path,
    *,
    component: str,
    baseline: Mapping[str, float],
    points: list[dict[str, Any]],
    selected_channels: list[tuple[str, int, float]],
    device: str,
    perturbation_basis_path: str | None = None,
    promotion_eligible: bool = False,
    sensitivity_source: str = "fisher_proxy",
    canonical_scorer_path: bool = False,
) -> None:
    enriched_points, prediction_metrics = _prediction_enriched_points(
        points,
        selected_channels=selected_channels,
    )
    observed_delta_max = max(
        (abs(float(point["delta"])) for point in enriched_points),
        default=0.0,
    )
    holdout_error = (
        float(prediction_metrics["max_relative_error"])
        if prediction_metrics.get("fit_status") == "least_squares_scale_fit"
        else observed_delta_max
    )
    holdout_error_kind = (
        "max_relative_prediction_error_diagnostic"
        if prediction_metrics.get("fit_status") == "least_squares_scale_fit"
        else "max_abs_observed_component_delta_diagnostic"
    )
    epsilon_metadata = _response_epsilon_metadata(
        [float(point["epsilon"]) for point in enriched_points]
    )
    official_response = bool(device == "cuda" and canonical_scorer_path)
    gate_results = _response_gate_results(
        enriched_points,
        prediction_metrics=prediction_metrics,
    )
    blockers = [] if promotion_eligible else [FISHER_PROXY_PROMOTION_BLOCKER]
    if promotion_eligible and not canonical_scorer_path:
        blockers.append(CANONICAL_SCORER_PATH_PROMOTION_BLOCKER)
    if not official_response:
        blockers.append(
            {
                "code": (
                    "non_canonical_component_response"
                    if device == "cuda"
                    else "non_cuda_response_curve"
                ),
                "mathematical_explanation": (
                    "Component-response curves are promotion-grade only when "
                    "they are CUDA-authored and measured through the canonical "
                    "archive.zip -> inflate.sh -> upstream/evaluate.py path."
                ),
            }
        )
    if promotion_eligible and not gate_results["promotion_gate_passed"]:
        blockers.append(
            {
                "code": "finite_difference_response_gate_failed",
                "mathematical_explanation": (
                    "The official CUDA finite-difference response curve did "
                    "not pass zero-reproduction and prediction-error gates."
                ),
            }
        )
    promotion_passed = bool(
        promotion_eligible
        and official_response
        and gate_results["promotion_gate_passed"]
        and not blockers
    )
    payload = {
        "schema_version": 1,
        "component": component,
        "device": device,
        "score_claim": False,
        "promotion_eligible": promotion_passed,
        "evidence_grade": (
            "A"
            if promotion_passed
            else "diagnostic_cuda_direct_renderer_component_response"
            if device == "cuda"
            else "diagnostic_cpu"
        ),
        "official_component_response": official_response,
        "canonical_scorer_path": bool(canonical_scorer_path),
        "component_response_path": (
            "archive_zip_inflate_sh_upstream_evaluate_py"
            if canonical_scorer_path
            else "direct_renderer_tensor_inprocess_scorer"
        ),
        "passed": promotion_passed,
        "gate_spec": dict(RESPONSE_GATE_SPEC),
        "gate_results": gate_results,
        "component_readout": OFFICIAL_COMPONENT_READOUTS[component],
        "component_units": {
            "posenet": "mean_pose_mse",
            "segnet": "mean_argmax_disagreement",
            "combined": "100*segnet + sqrt(10*posenet), no rate term",
        },
        "promotion_blockers": blockers,
        "sensitivity_source": sensitivity_source,
        "perturbation": "signed_weight_rms_topk_channels_v1",
        "perturbation_basis": {
            "format": PERTURBATION_BASIS_FORMAT,
            "path": perturbation_basis_path,
            "component_atom_ids": [
                _atom_id(component, order, key, channel)
                for order, (key, channel, _score) in enumerate(selected_channels)
            ],
        },
        "selected_channels": [
            {"key": key, "channel": channel, "sensitivity": score}
            for key, channel, score in selected_channels
        ],
        "baseline": dict(baseline),
        "count": len(enriched_points),
        "holdout_error": holdout_error,
        "holdout_error_kind": holdout_error_kind,
        "observed_holdout_delta_max": observed_delta_max,
        "prediction_calibration": prediction_metrics,
        **epsilon_metadata,
        "points": enriched_points,
    }
    write_json(path, payload)


def _paired_flat_values(
    a: Mapping[str, torch.Tensor],
    b: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    left: list[torch.Tensor] = []
    right: list[torch.Tensor] = []
    for key in sorted(set(a) & set(b)):
        if a[key].shape != b[key].shape:
            continue
        left.append(a[key].detach().to(torch.float64).cpu().reshape(-1))
        right.append(b[key].detach().to(torch.float64).cpu().reshape(-1))
    if not left:
        raise ComponentSensitivityProfileError("no shared sensitivity values for rank metrics")
    return torch.cat(left), torch.cat(right)


def _pearson_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() < 2:
        return 0.0
    a0 = a - a.mean()
    b0 = b - b.mean()
    denom = torch.linalg.vector_norm(a0) * torch.linalg.vector_norm(b0)
    if float(denom.item()) <= 1e-12:
        return 0.0
    return float((a0 * b0).sum().div(denom).clamp(-1.0, 1.0).item())


def _ordinal_ranks(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values, stable=True)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(values.numel(), dtype=torch.float64)
    return ranks


def _spearman_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() < 2:
        return 0.0
    return _pearson_corr(_ordinal_ranks(a), _ordinal_ranks(b))


def _rank_overlap(a: Mapping[str, torch.Tensor], b: Mapping[str, torch.Tensor], *, top_k: int) -> float:
    top_a = {(key, channel) for key, channel, _score in select_top_channels(a, top_k=top_k)}
    top_b = {(key, channel) for key, channel, _score in select_top_channels(b, top_k=top_k)}
    if not top_a and not top_b:
        return 1.0
    return len(top_a & top_b) / max(1, len(top_a | top_b))


def _top_fraction_overlap(
    a: Mapping[str, torch.Tensor],
    b: Mapping[str, torch.Tensor],
    *,
    fraction: float,
) -> dict[str, Any]:
    total = sum(int(tensor.numel()) for tensor in a.values())
    k = max(1, min(total, math.ceil(total * float(fraction))))
    return {"fraction": float(fraction), "k": int(k), "overlap": _rank_overlap(a, b, top_k=k)}


def _sensitivity_counts(maps: Mapping[str, torch.Tensor]) -> dict[str, int]:
    n_layers = len(maps)
    n_channels = 0
    zeros = 0
    nonfinite = 0
    negative = 0
    for tensor in maps.values():
        flat = tensor.detach().to(torch.float32).cpu().reshape(-1)
        n_channels += int(flat.numel())
        finite = torch.isfinite(flat)
        nonfinite += int((~finite).sum().item())
        zeros += int((flat[finite] == 0).sum().item())
        negative += int((flat[finite] < 0).sum().item())
    return {
        "layers": int(n_layers),
        "channels": int(n_channels),
        "zero": int(zeros),
        "nonfinite": int(nonfinite),
        "negative": int(negative),
    }


def _zero_like_maps(template: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        str(key): torch.zeros_like(value.detach().to(torch.float32).cpu())
        for key, value in template.items()
    }


def _copy_ref_value(
    target: dict[str, torch.Tensor],
    source: Mapping[str, torch.Tensor],
    *,
    key: str,
    channel: int,
    component: str,
    source_dir: Path,
) -> None:
    tensor = source.get(key)
    if tensor is None:
        raise ComponentSensitivityProfileError(
            f"{source_dir}: missing {component} sensitivity key {key!r}"
        )
    if channel < 0 or channel >= int(tensor.numel()):
        raise ComponentSensitivityProfileError(
            f"{source_dir}: {component} {key}[{channel}] out of range"
        )
    value = tensor.detach().to(torch.float32).cpu().reshape(-1)[channel]
    if not torch.isfinite(value) or float(value.item()) < 0.0:
        raise ComponentSensitivityProfileError(
            f"{source_dir}: {component} {key}[{channel}] must be finite/nonnegative"
        )
    target[key][channel] = value


def _load_profile_summary(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ComponentSensitivityProfileError(f"{path}: summary must be a JSON object")
    return payload


def _merge_finite_difference_shard_maps(
    shard_dirs: list[str | Path],
) -> tuple[dict[str, dict[str, torch.Tensor]], dict[str, dict[str, torch.Tensor]], dict[str, Any]]:
    """Merge direct-FD shard calibration and holdout maps with exact coverage."""

    roots = [Path(path) for path in shard_dirs]
    if len(roots) <= 1:
        raise ComponentSensitivityProfileError("merge requires at least two shard directories")
    summaries = []
    assigned_by_shard: list[list[tuple[str, int]]] = []
    all_refs: list[tuple[str, int]] | None = None
    expected: dict[str, Any] | None = None
    calibration_out: dict[str, dict[str, torch.Tensor]] | None = None
    holdout_out: dict[str, dict[str, torch.Tensor]] | None = None
    seen: set[tuple[str, int]] = set()

    for root in roots:
        if not root.is_dir():
            raise ComponentSensitivityProfileError(f"shard dir not found: {root}")
        summary = _load_profile_summary(root / "component_sensitivity_profile_summary.json")
        summaries.append(summary)
        shard = summary.get("finite_difference_shard")
        if not isinstance(shard, dict) or shard.get("schema") != FINITE_DIFFERENCE_SHARD_SCHEMA:
            raise ComponentSensitivityProfileError(f"{root}: missing finite-difference shard metadata")
        if shard.get("is_shard") is not True:
            raise ComponentSensitivityProfileError(f"{root}: merge input must be a partial shard")
        if summary.get("sensitivity_source") != "direct_renderer_cuda_finite_difference_component_response":
            raise ComponentSensitivityProfileError(f"{root}: shard source is not direct finite difference")
        refs = _refs_from_payload(shard.get("assigned_channel_refs"), label=f"{root}.assigned_channel_refs")
        declared_all_sha = shard.get("all_channel_sha256")
        if _channel_ref_sha256(refs) != shard.get("assigned_channel_sha256"):
            raise ComponentSensitivityProfileError(f"{root}: assigned channel SHA mismatch")

        all_payload = shard.get("all_channel_refs")
        if all_payload is not None:
            root_all_refs = _refs_from_payload(all_payload, label=f"{root}.all_channel_refs")
            if _channel_ref_sha256(root_all_refs) != declared_all_sha:
                raise ComponentSensitivityProfileError(f"{root}: all-channel SHA mismatch")
            if all_refs is None:
                all_refs = root_all_refs
            elif root_all_refs != all_refs:
                raise ComponentSensitivityProfileError(f"{root}: all-channel refs differ from first shard")

        invariant = {
            "n_pairs_total": summary.get("n_pairs_total"),
            "n_pairs_selected": summary.get("n_pairs_selected"),
            "n_pairs_calibration": summary.get("n_pairs_calibration"),
            "n_pairs_holdout": summary.get("n_pairs_holdout"),
            "split_seed": summary.get("split_seed"),
            "finite_difference_epsilon": summary.get("finite_difference_epsilon"),
            "device": summary.get("device"),
            "component_response_path": summary.get("component_response_path"),
            "all_channel_sha256": declared_all_sha,
            "shard_count": shard.get("shard_count"),
        }
        if expected is None:
            expected = invariant
        elif invariant != expected:
            raise ComponentSensitivityProfileError(f"{root}: shard invariant mismatch")

        assigned_by_shard.append(refs)
        for ref in refs:
            if ref in seen:
                raise ComponentSensitivityProfileError(f"duplicate finite-difference shard channel: {ref}")
            seen.add(ref)

        loaded_calibration: dict[str, dict[str, torch.Tensor]] = {}
        loaded_holdout: dict[str, dict[str, torch.Tensor]] = {}
        for component in COMPONENT_OUTPUTS:
            cal, cal_meta = load_sensitivity_map(root / f"{component}_sensitivity_map.pt")
            hold, hold_meta = load_sensitivity_map(root / f"{component}_holdout_sensitivity_map.pt")
            if cal_meta.get("component") != component or hold_meta.get("component") != component:
                raise ComponentSensitivityProfileError(f"{root}: {component} map metadata mismatch")
            if cal_meta.get("finite_difference_shard") != shard or hold_meta.get("finite_difference_shard") != shard:
                raise ComponentSensitivityProfileError(f"{root}: {component} map shard metadata mismatch")
            loaded_calibration[component] = cal
            loaded_holdout[component] = hold
        if calibration_out is None:
            calibration_out = {
                name: _zero_like_maps(loaded_calibration[name])
                for name in COMPONENT_OUTPUTS
            }
            holdout_out = {
                name: _zero_like_maps(loaded_holdout[name])
                for name in COMPONENT_OUTPUTS
            }
        assert calibration_out is not None and holdout_out is not None
        for component in COMPONENT_OUTPUTS:
            for key, channel in refs:
                _copy_ref_value(
                    calibration_out[component],
                    loaded_calibration[component],
                    key=key,
                    channel=channel,
                    component=component,
                    source_dir=root,
                )
                _copy_ref_value(
                    holdout_out[component],
                    loaded_holdout[component],
                    key=key,
                    channel=channel,
                    component=component,
                    source_dir=root,
                )

    if all_refs is None:
        raise ComponentSensitivityProfileError("shards did not record all-channel refs")
    if sorted(seen) != sorted(all_refs):
        missing = sorted(set(all_refs) - seen)
        extra = sorted(seen - set(all_refs))
        raise ComponentSensitivityProfileError(
            f"finite-difference shard coverage mismatch: missing={missing[:5]} extra={extra[:5]}"
        )
    assert calibration_out is not None and holdout_out is not None and expected is not None
    merge = {
        "schema": FINITE_DIFFERENCE_MERGE_SCHEMA,
        "source_shard_count": len(roots),
        "declared_shard_count": int(expected["shard_count"]),
        "source_shard_dirs": [str(root) for root in roots],
        "all_channel_count": len(all_refs),
        "all_channel_sha256": _channel_ref_sha256(all_refs),
        "assigned_channel_sha256_by_shard": [
            _channel_ref_sha256(refs) for refs in assigned_by_shard
        ],
        "coverage": "exactly_once",
        "promotion_eligible": False,
        "score_claim": False,
    }
    if merge["declared_shard_count"] != merge["source_shard_count"]:
        raise ComponentSensitivityProfileError(
            "finite-difference shard count mismatch: "
            f"declared={merge['declared_shard_count']} provided={merge['source_shard_count']}"
        )
    return calibration_out, holdout_out, merge


def _write_stability_json(
    path: Path,
    *,
    calibration_maps: Mapping[str, Mapping[str, torch.Tensor]],
    holdout_maps: Mapping[str, Mapping[str, torch.Tensor]],
    top_k: int,
) -> dict[str, Any]:
    cv = {
        component: max(sensitivity_cv_distance(calibration_maps[component], holdout_maps[component]).values())
        for component in COMPONENT_OUTPUTS
    }
    cv_by_layer = {
        component: sensitivity_cv_distance(calibration_maps[component], holdout_maps[component])
        for component in COMPONENT_OUTPUTS
    }
    rank = {
        component: _rank_overlap(calibration_maps[component], holdout_maps[component], top_k=top_k)
        for component in COMPONENT_OUTPUTS
    }
    topk = {
        component: {"k": int(top_k), "overlap": rank[component]}
        for component in COMPONENT_OUTPUTS
    }
    correlation: dict[str, dict[str, float]] = {}
    for component in COMPONENT_OUTPUTS:
        left, right = _paired_flat_values(calibration_maps[component], holdout_maps[component])
        correlation[component] = {
            "pearson": _pearson_corr(left, right),
            "spearman": _spearman_corr(left, right),
        }
    top_fraction = {
        component: {
            "top_1pct": _top_fraction_overlap(
                calibration_maps[component],
                holdout_maps[component],
                fraction=0.01,
            ),
            "top_5pct": _top_fraction_overlap(
                calibration_maps[component],
                holdout_maps[component],
                fraction=0.05,
            ),
            "top_10pct": _top_fraction_overlap(
                calibration_maps[component],
                holdout_maps[component],
                fraction=0.10,
            ),
        }
        for component in COMPONENT_OUTPUTS
    }
    counts = {
        component: {
            "calibration": _sensitivity_counts(calibration_maps[component]),
            "holdout": _sensitivity_counts(holdout_maps[component]),
        }
        for component in COMPONENT_OUTPUTS
    }
    component_passed = {
        component: (
            cv[component] <= STABILITY_THRESHOLDS["cv_max"]
            and correlation[component]["spearman"] >= STABILITY_THRESHOLDS["spearman_min"]
            and correlation[component]["pearson"] >= STABILITY_THRESHOLDS["pearson_min"]
            and top_fraction[component]["top_10pct"]["overlap"]
            >= STABILITY_THRESHOLDS["top_decile_overlap_min"]
        )
        for component in COMPONENT_OUTPUTS
    }
    payload = {
        "cv": cv,
        "rank": rank,
        "top_k": topk,
        "correlation": correlation,
        "top_fraction": top_fraction,
        "counts": counts,
        "cv_by_layer": cv_by_layer,
        "thresholds": dict(STABILITY_THRESHOLDS),
        "component_passed": component_passed,
        "passed": all(component_passed.values()),
    }
    write_json(path, payload)
    return payload


def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "no-git"


def profile_component_sensitivity(
    *,
    checkpoint: str,
    video_mkv: str,
    masks_mkv: str,
    poses_path: str | None,
    upstream_dir: str,
    output_dir: str,
    pair_weights_path: str | None,
    all_pairs: bool,
    top_k_pairs: int,
    pair_batch: int,
    response_top_k: int,
    response_epsilons: list[float],
    split_seed: int,
    holdout_fraction: float,
    aggregate: str,
    device: str,
    allow_diagnostic_cpu: bool = False,
    promotion_finite_difference: bool = False,
    finite_difference_epsilon: float = DEFAULT_FINITE_DIFFERENCE_EPSILON,
    finite_difference_shard_index: int = 0,
    finite_difference_shard_count: int = 1,
    merge_shard_dirs: list[str] | None = None,
) -> dict[str, Any]:
    t0 = time.monotonic()
    if promotion_finite_difference and device != "cuda":
        raise SystemExit("FATAL: promotion finite-difference sensitivity requires --device cuda")
    if promotion_finite_difference and not all_pairs:
        raise SystemExit(
            "FATAL: promotion finite-difference sensitivity requires --all-pairs "
            "so sample_plan covers the full contest set"
        )
    if merge_shard_dirs and not promotion_finite_difference:
        raise SystemExit("FATAL: --merge-shard-dir requires --promotion-finite-difference")
    if merge_shard_dirs and int(finite_difference_shard_count) != 1:
        raise SystemExit("FATAL: merged direct-FD output must use --finite-difference-shard-count=1")
    if int(finite_difference_shard_count) <= 0:
        raise SystemExit("FATAL: --finite-difference-shard-count must be positive")
    if int(finite_difference_shard_index) < 0 or int(finite_difference_shard_index) >= int(finite_difference_shard_count):
        raise SystemExit("FATAL: --finite-difference-shard-index must be within shard count")
    torch_device = _require_device(device, allow_diagnostic_cpu=allow_diagnostic_cpu)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = _load_renderer(checkpoint, device)
    has_film = hasattr(model, "pose_dim") and int(getattr(model, "pose_dim", 0) or 0) > 0
    has_zoom = bool(getattr(model, "use_zoom_flow", False))
    poses = None
    if has_film:
        if poses_path is None:
            raise SystemExit("FATAL: model has FiLM but --poses was not provided")
        poses = _load_poses(poses_path, int(getattr(model, "pose_dim", 6) or 6)).to(torch_device)

    gt_frames_cpu = _decode_gt_video(video_mkv)
    masks_cpu = _load_masks_video(masks_mkv)
    n_frames = int(gt_frames_cpu.shape[0])
    n_pairs = n_frames // 2
    _require_full_contest_frame_count(
        n_frames=n_frames,
        n_mask_frames=int(masks_cpu.shape[0]),
        promotion_finite_difference=promotion_finite_difference,
    )
    if promotion_finite_difference and n_pairs != CONTEST_SAMPLE_COUNT:
        raise SystemExit(
            "FATAL: promotion finite-difference sensitivity requires "
            f"{CONTEST_SAMPLE_COUNT} contest pairs, got {n_pairs}"
        )
    pair_weights = _load_pair_weights(None if all_pairs else pair_weights_path, n_pairs)
    if pair_weights_path is not None:
        selected = torch.topk(pair_weights, k=min(top_k_pairs, n_pairs)).indices.tolist()
    else:
        selected = list(range(n_pairs))
    selected = sorted(int(i) for i in selected)
    plan = make_sample_plan_for_indices(
        pair_indices=selected,
        split_seed=split_seed,
        holdout_fraction=holdout_fraction,
    )
    calibration_indices = [item["pair_index"] for item in plan["calibration_pairs"]]
    holdout_indices = [item["pair_index"] for item in plan["holdout_pairs"]]

    from tac.scorer import load_differentiable_scorers

    posenet, segnet = load_differentiable_scorers(upstream_dir, device=device)
    for scorer in (posenet, segnet):
        scorer.eval()
        for param in scorer.parameters():
            param.requires_grad_(False)

    zoom_warp = None
    if has_zoom:
        from tac.radial_zoom import RadialZoomWarp

        zoom_warp = RadialZoomWarp().to(torch_device)

    eligible, excluded = _select_eligible_params_with_exclusions(
        model,
        include_protected_conv2d=True,
    )
    if not eligible:
        raise ComponentSensitivityProfileError("no eligible Conv2d/Linear weights")
    fd_shard: dict[str, Any] | None = None
    fd_merge: dict[str, Any] | None = None
    fd_channel_refs: list[tuple[str, int]] | None = None
    if promotion_finite_difference:
        fd_shard = _finite_difference_shard_plan(
            model,
            shard_index=int(finite_difference_shard_index),
            shard_count=int(finite_difference_shard_count),
        )
        fd_channel_refs = _refs_from_payload(
            fd_shard["assigned_channel_refs"],
            label="finite_difference_shard.assigned_channel_refs",
        )

    calibration_baseline = _evaluate_component_means(
        model=model,
        pair_indices=calibration_indices,
        masks_cpu=masks_cpu,
        gt_frames_cpu=gt_frames_cpu,
        poses=poses,
        posenet=posenet,
        segnet=segnet,
        device=torch_device,
        zoom_warp=zoom_warp,
        pair_batch=pair_batch,
    )
    baseline = _evaluate_component_means(
        model=model,
        pair_indices=holdout_indices,
        masks_cpu=masks_cpu,
        gt_frames_cpu=gt_frames_cpu,
        poses=poses,
        posenet=posenet,
        segnet=segnet,
        device=torch_device,
        zoom_warp=zoom_warp,
        pair_batch=pair_batch,
    )

    if merge_shard_dirs:
        calibration_maps, holdout_maps, fd_merge = _merge_finite_difference_shard_maps(
            merge_shard_dirs
        )
        fd_shard = {
            **(fd_shard or {}),
            "is_shard": False,
            "merge_required_for_certification_handoff": False,
            "merged_from_shards": True,
            "assigned_channel_count": fd_merge["all_channel_count"],
            "all_channel_count": fd_merge["all_channel_count"],
            "all_channel_sha256": fd_merge["all_channel_sha256"],
            "assigned_channel_sha256": fd_merge["all_channel_sha256"],
        }
    elif promotion_finite_difference:
        calibration_maps = _finite_difference_component_channel_maps(
            model=model,
            pair_indices=calibration_indices,
            baseline=calibration_baseline,
            masks_cpu=masks_cpu,
            gt_frames_cpu=gt_frames_cpu,
            poses=poses,
            posenet=posenet,
            segnet=segnet,
            device=torch_device,
            zoom_warp=zoom_warp,
            pair_batch=pair_batch,
            epsilon=finite_difference_epsilon,
            channel_refs=fd_channel_refs,
        )
        holdout_maps = _finite_difference_component_channel_maps(
            model=model,
            pair_indices=holdout_indices,
            baseline=baseline,
            masks_cpu=masks_cpu,
            gt_frames_cpu=gt_frames_cpu,
            poses=poses,
            posenet=posenet,
            segnet=segnet,
            device=torch_device,
            zoom_warp=zoom_warp,
            pair_batch=pair_batch,
            epsilon=finite_difference_epsilon,
            channel_refs=fd_channel_refs,
        )
    else:
        calibration_importance = _profile_split(
            model=model,
            pair_indices=calibration_indices,
            pair_weights=pair_weights,
            masks_cpu=masks_cpu,
            gt_frames_cpu=gt_frames_cpu,
            poses=poses,
            posenet=posenet,
            segnet=segnet,
            device=torch_device,
            zoom_warp=zoom_warp,
            eligible=eligible,
            pair_batch=pair_batch,
        )
        holdout_importance = _profile_split(
            model=model,
            pair_indices=holdout_indices,
            pair_weights=pair_weights,
            masks_cpu=masks_cpu,
            gt_frames_cpu=gt_frames_cpu,
            poses=poses,
            posenet=posenet,
            segnet=segnet,
            device=torch_device,
            zoom_warp=zoom_warp,
            eligible=eligible,
            pair_batch=pair_batch,
        )

        calibration_maps = _aggregate_maps(
            model=model,
            importance_by_component=calibration_importance,
            aggregate=aggregate,
        )
        holdout_maps = _aggregate_maps(
            model=model,
            importance_by_component=holdout_importance,
            aggregate=aggregate,
        )

    metadata_base = {
        "schema_version": 1,
        "tool": "experiments/profile_component_sensitivity.py",
        "checkpoint": checkpoint,
        "video_mkv": video_mkv,
        "masks_mkv": masks_mkv,
        "poses": poses_path,
        "upstream_dir": upstream_dir,
        "pair_weights": pair_weights_path,
        "n_pairs_total": n_pairs,
        "n_pairs_selected": len(selected),
        "n_pairs_calibration": len(calibration_indices),
        "n_pairs_holdout": len(holdout_indices),
        "include_protected_conv2d": True,
        "excluded_layer_reasons": dict(sorted(excluded.items())),
        "aggregate": aggregate,
        "torch_version": torch.__version__,
        "device": device,
        "promotion_requested": bool(promotion_finite_difference),
        "score_claim": False,
        "promotion_eligible": False,
        "evidence_grade": (
            "diagnostic_cuda_direct_renderer_finite_difference"
            if promotion_finite_difference and device == "cuda"
            else "diagnostic_cuda_fisher_proxy"
            if device == "cuda"
            else "diagnostic_cpu"
        ),
        "official_component_response": False,
        "canonical_scorer_path": False,
        "component_response_path": "direct_renderer_tensor_inprocess_scorer",
        "sensitivity_source": (
            "direct_renderer_cuda_finite_difference_component_response"
            if promotion_finite_difference
            else "fisher_proxy"
        ),
        "finite_difference_epsilon": (
            float(finite_difference_epsilon)
            if promotion_finite_difference
            else None
        ),
        "finite_difference_shard": fd_shard,
        "finite_difference_merge": fd_merge,
        "certification_handoff_eligible": bool(
            promotion_finite_difference
            and fd_shard is not None
            and fd_shard.get("is_shard") is False
            and fd_shard.get("merge_required_for_certification_handoff") is False
        ),
        "promotion_blockers": (
            [CANONICAL_SCORER_PATH_PROMOTION_BLOCKER]
            if promotion_finite_difference
            else [FISHER_PROXY_PROMOTION_BLOCKER]
        ),
        "git_hash": _git_hash(),
    }

    map_paths: dict[str, str] = {}
    holdout_map_paths: dict[str, str] = {}
    for component in COMPONENT_OUTPUTS:
        path = out_dir / f"{component}_sensitivity_map.pt"
        save_sensitivity_map(
            path,
            calibration_maps[component],
            metadata={**metadata_base, "component": component, "scorer_target": component},
        )
        map_paths[component] = str(path)
        holdout_path = out_dir / f"{component}_holdout_sensitivity_map.pt"
        save_sensitivity_map(
            holdout_path,
            holdout_maps[component],
            metadata={
                **metadata_base,
                "component": component,
                "scorer_target": component,
                "split": "holdout",
            },
        )
        holdout_map_paths[component] = str(holdout_path)

    sample_plan_path = out_dir / "sample_plan.json"
    write_json(sample_plan_path, plan)
    stability_path = out_dir / "stability.json"
    _write_stability_json(
        stability_path,
        calibration_maps=calibration_maps,
        holdout_maps=holdout_maps,
        top_k=response_top_k,
    )

    selected_by_component = {
        component: select_top_channels(calibration_maps[component], top_k=response_top_k)
        for component in COMPONENT_OUTPUTS
    }
    perturbation_basis_path = out_dir / "perturbation_basis_v1.json"
    _write_perturbation_basis_json(
        perturbation_basis_path,
        selected_by_component=selected_by_component,
        sample_plan=plan,
        response_epsilons=response_epsilons,
        response_top_k=response_top_k,
        input_metadata=metadata_base,
    )
    response_paths: dict[str, str] = {}
    for component in COMPONENT_OUTPUTS:
        selected_channels = selected_by_component[component]
        points: list[dict[str, Any]] = []
        for epsilon in response_epsilons:
            originals = apply_channel_perturbation(
                model,
                selected_channels,
                epsilon=float(epsilon),
            )
            try:
                values = _evaluate_component_means(
                    model=model,
                    pair_indices=holdout_indices,
                    masks_cpu=masks_cpu,
                    gt_frames_cpu=gt_frames_cpu,
                    poses=poses,
                    posenet=posenet,
                    segnet=segnet,
                    device=torch_device,
                    zoom_warp=zoom_warp,
                    pair_batch=pair_batch,
                )
            finally:
                restore_perturbation(originals)
            points.append(
                {
                    "epsilon": float(epsilon),
                    "value": float(values[component]),
                    "baseline": float(baseline[component]),
                    "delta": float(values[component] - baseline[component]),
                    "all_components": values,
                }
            )
        path = out_dir / f"{component}_response_curve.json"
        _write_response_curve(
            path,
            component=component,
            baseline=baseline,
            points=points,
            selected_channels=selected_channels,
            device=device,
            perturbation_basis_path=str(perturbation_basis_path),
            promotion_eligible=bool(promotion_finite_difference),
            sensitivity_source=(
                "direct_renderer_cuda_finite_difference_component_response"
                if promotion_finite_difference
                else "fisher_proxy"
            ),
            canonical_scorer_path=False,
        )
        response_paths[component] = str(path)

    summary = {
        **metadata_base,
        "elapsed_s": time.monotonic() - t0,
        "map_paths": map_paths,
        "holdout_map_paths": holdout_map_paths,
        "response_curve_paths": response_paths,
        "perturbation_basis_json": str(perturbation_basis_path),
        "sample_plan_json": str(sample_plan_path),
        "stability_json": str(stability_path),
    }
    write_json(out_dir / "component_sensitivity_profile_summary.json", summary)
    return summary


def build_manifest_from_profile_outputs(
    *,
    summary: Mapping[str, Any],
    checkpoint: str,
    video_mkv: str,
    upstream_dir: str,
    archive: str,
    contest_auth_eval_json: str,
    output: str,
    evidence_grade: str = "A",
) -> dict[str, Any]:
    """Assemble ``component_sensitivity_v1`` from this profiler's outputs."""
    if summary.get("device") != "cuda":
        raise ComponentSensitivityProfileError(
            "component sensitivity manifest assembly requires a CUDA-authored profile"
        )
    if summary.get("promotion_eligible") is not True:
        raise ComponentSensitivityProfileError(
            "profile_component_sensitivity.py outputs are not promotion-eligible. "
            "Do not assemble a promotable component_sensitivity_v1 manifest "
            "from diagnostic Fisher-proxy or direct-renderer finite-difference "
            "artifacts."
        )
    if summary.get("canonical_scorer_path") is not True:
        raise ComponentSensitivityProfileError(
            "component sensitivity manifest assembly requires artifacts "
            "measured through the canonical archive.zip -> inflate.sh -> "
            "upstream/evaluate.py path; profile_component_sensitivity.py "
            "currently emits direct-renderer diagnostics."
        )
    from experiments.build_component_sensitivity_manifest import build_manifest

    ns = argparse.Namespace(
        checkpoint=Path(checkpoint),
        video=Path(video_mkv),
        upstream=Path(upstream_dir),
        archive=Path(archive),
        contest_auth_eval_json=Path(contest_auth_eval_json),
        component_maps={
            component: Path(summary["map_paths"][component])
            for component in COMPONENT_OUTPUTS
        },
        response_curves={
            component: Path(summary["response_curve_paths"][component])
            for component in COMPONENT_OUTPUTS
        },
        stability_json=Path(summary["stability_json"]),
        sample_plan_json=Path(summary["sample_plan_json"]),
        root=None,
        device="cuda",
        evidence_grade=evidence_grade,
        n_samples=CONTEST_SAMPLE_COUNT,
        n_pairs=CONTEST_SAMPLE_COUNT,
        split_seed=0,
        holdout_fraction=0.2,
        output=Path(output),
    )
    manifest = build_manifest(ns)
    write_component_sensitivity_manifest(output, manifest)
    return manifest


def _parse_epsilons(value: str) -> list[float]:
    out = [float(item) for item in value.split(",") if item.strip()]
    if not out:
        raise argparse.ArgumentTypeError("must provide at least one epsilon")
    if any(not torch.isfinite(torch.tensor(eps)) for eps in out):
        raise argparse.ArgumentTypeError("epsilons must be finite")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--masks-mkv", required=True)
    parser.add_argument("--poses", default=None)
    parser.add_argument("--upstream", default="upstream")
    parser.add_argument("--output-dir", required=True)
    weight_group = parser.add_mutually_exclusive_group(required=True)
    weight_group.add_argument("--pair-weights", default=None)
    weight_group.add_argument("--all-pairs", action="store_true")
    parser.add_argument("--top-k-pairs", type=int, default=64)
    parser.add_argument("--pair-batch", type=int, default=2)
    parser.add_argument("--response-top-k", type=int, default=16)
    parser.add_argument(
        "--response-epsilons",
        type=_parse_epsilons,
        default=DEFAULT_RESPONSE_EPSILONS,
    )
    parser.add_argument("--split-seed", type=int, default=20260430)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--aggregate", choices=["sum", "mean", "max"], default="sum")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--allow-diagnostic-cpu", action="store_true")
    parser.add_argument(
        "--promotion-finite-difference",
        action="store_true",
        help=(
            "Compute CUDA direct-renderer finite-difference component maps "
            "instead of Fisher-proxy maps. This is still non-promotable until "
            "component response is measured through archive.zip -> inflate.sh "
            "-> upstream/evaluate.py. Requires --device cuda, --all-pairs, "
            "and the full 600-pair contest video."
        ),
    )
    parser.add_argument(
        "--finite-difference-epsilon",
        type=float,
        default=DEFAULT_FINITE_DIFFERENCE_EPSILON,
        help=(
            "Positive signed-RMS perturbation epsilon for promotion finite "
            "differences."
        ),
    )
    parser.add_argument("--finite-difference-shard-index", type=int, default=0)
    parser.add_argument("--finite-difference-shard-count", type=int, default=1)
    parser.add_argument(
        "--merge-shard-dir",
        action="append",
        default=[],
        help=(
            "Directory containing a partial direct-FD shard to merge. Repeat for "
            "all shards. The merged output remains diagnostic and non-score."
        ),
    )
    parser.add_argument("--archive", default=None, help="Optional exact archive for manifest assembly.")
    parser.add_argument(
        "--contest-auth-eval-json",
        default=None,
        help="Optional exact CUDA contest_auth_eval.json for manifest assembly.",
    )
    parser.add_argument(
        "--manifest-output",
        default=None,
        help=(
            "Blocked for current diagnostic profiler outputs; use "
            "experiments/build_component_sensitivity_manifest.py only with "
            "canonical inflate/evaluate component-response artifacts."
        ),
    )
    parser.add_argument("--evidence-grade", choices=["A", "A++"], default="A")
    args = parser.parse_args(argv)
    manifest_args = [args.archive, args.contest_auth_eval_json, args.manifest_output]
    if any(value is not None for value in manifest_args) and not all(
        value is not None for value in manifest_args
    ):
        parser.error(
            "--archive, --contest-auth-eval-json, and --manifest-output "
            "must be provided together"
        )
    if args.manifest_output is not None:
        parser.error(
            "--manifest-output is disabled for profile_component_sensitivity.py "
            "because current outputs are diagnostic direct-renderer artifacts "
            "with promotion_eligible=false. Promotion manifest assembly "
            "requires canonical archive.zip -> inflate.sh -> "
            "upstream/evaluate.py component-response artifacts."
        )

    summary = profile_component_sensitivity(
        checkpoint=args.checkpoint,
        video_mkv=args.video,
        masks_mkv=args.masks_mkv,
        poses_path=args.poses,
        upstream_dir=args.upstream,
        output_dir=args.output_dir,
        pair_weights_path=args.pair_weights,
        all_pairs=args.all_pairs,
        top_k_pairs=args.top_k_pairs,
        pair_batch=args.pair_batch,
        response_top_k=args.response_top_k,
        response_epsilons=args.response_epsilons,
        split_seed=args.split_seed,
        holdout_fraction=args.holdout_fraction,
        aggregate=args.aggregate,
        device=args.device,
        allow_diagnostic_cpu=args.allow_diagnostic_cpu,
        promotion_finite_difference=args.promotion_finite_difference,
        finite_difference_epsilon=args.finite_difference_epsilon,
        finite_difference_shard_index=args.finite_difference_shard_index,
        finite_difference_shard_count=args.finite_difference_shard_count,
        merge_shard_dirs=args.merge_shard_dir,
    )
    if args.manifest_output is not None:
        manifest = build_manifest_from_profile_outputs(
            summary=summary,
            checkpoint=args.checkpoint,
            video_mkv=args.video,
            upstream_dir=args.upstream,
            archive=args.archive,
            contest_auth_eval_json=args.contest_auth_eval_json,
            output=args.manifest_output,
            evidence_grade=args.evidence_grade,
        )
        summary = {
            **summary,
            "component_sensitivity_manifest": args.manifest_output,
            "manifest_archive_sha256": manifest["contest_eval"]["archive"]["sha256"],
        }
    print(json_text(summary), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
