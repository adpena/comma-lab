"""Score-sensitivity map contract for β / Ω-W-V3.

This module intentionally does not compute scorer gradients itself. It defines
the narrow artifact contract consumed by sensitivity-aware codecs:

    { "<module>.weight" -> Tensor[O] }

where O is the Conv2d output-channel count and each value is a non-negative
compress-time score sensitivity. Authoritative maps must be produced on CUDA;
CPU maps are allowed only for local unit/smoke tests and must be tagged as
non-authoritative by callers.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import torch
import torch.nn as nn


SENSITIVITY_MAP_FORMAT = "tac_score_sensitivity_map_v1"
CERTIFIED_SENSITIVITY_MAP_CERTIFICATION_FORMAT = "component_sensitivity_map_certification_v1"

_CERTIFICATION_SHA_FIELDS = (
    "source_map_sha256",
    "official_response_curve_sha256",
    "stability_sha256",
    "sample_plan_sha256",
    "baseline_archive_sha256",
    "contest_auth_eval_json_sha256",
)
_CERTIFICATION_TRUE_FIELDS = (
    "official_component_response",
    "canonical_scorer_path",
    "promotion_eligible",
)
_CERTIFICATION_COMPONENTS = {"posenet", "segnet", "combined"}


class SensitivityMapError(ValueError):
    """Raised when a sensitivity map is malformed or non-authoritative."""


@dataclass(frozen=True)
class SensitivityMapStats:
    """Summary of a validated sensitivity map."""

    n_layers: int
    n_channels: int
    min_value: float
    max_value: float


def require_authoritative_device(device: str | torch.device | None) -> None:
    """Reject non-CUDA devices for promotion-grade sensitivity maps."""
    if device is None:
        raise SensitivityMapError(
            "authoritative sensitivity maps must record their device; got None"
        )
    device_type = torch.device(device).type
    if device_type != "cuda":
        raise SensitivityMapError(
            f"authoritative sensitivity maps require CUDA; got device={device_type!r}. "
            "CPU/MPS maps are smoke-only per CLAUDE.md non-negotiable "
            '"MPS auth eval is NOISE" — cannot promote/kill a lane.'
        )


def validate_sensitivity_vector(
    value: torch.Tensor,
    *,
    expected_channels: int,
    name: str,
) -> torch.Tensor:
    """Return a CPU float32 1-D non-negative vector or raise loudly."""
    if not torch.is_tensor(value):
        raise SensitivityMapError(
            f"{name}: sensitivity must be a torch.Tensor, got {type(value).__name__}"
        )
    if value.dim() != 1 or int(value.shape[0]) != int(expected_channels):
        raise SensitivityMapError(
            f"{name}: sensitivity shape {tuple(value.shape)} does not match "
            f"expected ({expected_channels},)"
        )
    out = value.detach().to(torch.float32).cpu()
    if not torch.isfinite(out).all():
        n_bad = int((~torch.isfinite(out)).sum().item())
        raise SensitivityMapError(
            f"{name}: sensitivity contains {n_bad} NaN/Inf value(s)"
        )
    if (out < 0).any():
        raise SensitivityMapError(f"{name}: sensitivity must be non-negative")
    return out


def conv_weight_shapes(model: nn.Module) -> dict[str, int]:
    """Return `{ '<module>.weight': out_channels }` for all Conv2d modules."""
    out: dict[str, int] = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            out[f"{name}.weight"] = int(module.weight.shape[0])
    return out


def resolve_layer_sensitivity(
    sensitivities: Mapping[str, torch.Tensor],
    *,
    module_name: str,
    weight: torch.Tensor,
    required: bool = True,
) -> torch.Tensor | None:
    """Resolve and validate a layer vector.

    The canonical key is `"<module>.weight"`. A bare module-name fallback is
    accepted to ease migration from older profiling scripts, but new artifacts
    should use the canonical key.
    """
    key = f"{module_name}.weight"
    value = sensitivities.get(key)
    if value is None:
        value = sensitivities.get(module_name)
    if value is None:
        if required:
            raise SensitivityMapError(
                f"missing sensitivity for {key}; OWV3 cannot protect channels "
                "without a compress-time score-sensitivity artifact"
            )
        return None
    return validate_sensitivity_vector(
        value,
        expected_channels=int(weight.shape[0]),
        name=key,
    )


def validate_sensitivity_map_for_model(
    sensitivities: Mapping[str, torch.Tensor],
    model: nn.Module,
    *,
    require_all_conv: bool = False,
) -> SensitivityMapStats:
    """Validate all provided entries and optionally require every Conv2d."""
    shapes = conv_weight_shapes(model)
    n_channels = 0
    mins: list[float] = []
    maxs: list[float] = []

    if require_all_conv:
        missing = [key for key in shapes if key not in sensitivities]
        if missing:
            raise SensitivityMapError(
                f"sensitivity map missing {len(missing)} Conv2d layer(s): "
                f"{missing[:5]}{'...' if len(missing) > 5 else ''}"
            )

    for key, value in sensitivities.items():
        canonical = key if key.endswith(".weight") else f"{key}.weight"
        if canonical not in shapes:
            continue
        vec = validate_sensitivity_vector(
            value,
            expected_channels=shapes[canonical],
            name=canonical,
        )
        n_channels += int(vec.numel())
        if vec.numel():
            mins.append(float(vec.min().item()))
            maxs.append(float(vec.max().item()))

    if n_channels == 0:
        raise SensitivityMapError(
            "sensitivity map contains no entries matching model Conv2d weights"
        )
    return SensitivityMapStats(
        n_layers=len(mins),
        n_channels=n_channels,
        min_value=min(mins) if mins else 0.0,
        max_value=max(maxs) if maxs else 0.0,
    )


def save_sensitivity_map(
    path: str | Path,
    sensitivities: Mapping[str, torch.Tensor],
    *,
    metadata: Mapping[str, object] | None = None,
) -> None:
    """Persist a tensor-only sensitivity artifact."""
    clean: MutableMapping[str, torch.Tensor] = {}
    for key, value in sensitivities.items():
        if not torch.is_tensor(value) or value.dim() != 1:
            raise SensitivityMapError(
                f"{key}: sensitivity must be a 1-D tensor before save"
            )
        clean[str(key)] = validate_sensitivity_vector(
            value,
            expected_channels=int(value.shape[0]),
            name=str(key),
        )
    payload = {
        "format": SENSITIVITY_MAP_FORMAT,
        "sensitivities": dict(clean),
        "metadata": dict(metadata or {}),
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, p)


def load_sensitivity_map(path: str | Path) -> tuple[dict[str, torch.Tensor], dict]:
    """Load a saved sensitivity artifact.

    Round 2B B3 fix (2026-05-06, 82% confidence): use `weights_only=True`.
    Sensitivity maps are produced remotely (Vast.ai / Lightning) and shipped
    back as artifacts — they are NOT fully trusted. The payload is a plain
    dict of tensors + a format-string field, all of which are supported under
    `weights_only=True` in PyTorch >= 2.0. This closes an arbitrary-code-
    execution path via pickle and aligns with the
    `preflight_loader_format_safety` allowlist gate.
    """
    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("format") != SENSITIVITY_MAP_FORMAT:
        raise SensitivityMapError(
            f"{path}: expected format {SENSITIVITY_MAP_FORMAT!r}"
        )
    raw = payload.get("sensitivities")
    if not isinstance(raw, dict):
        raise SensitivityMapError(f"{path}: missing sensitivities dict")
    out: dict[str, torch.Tensor] = {}
    for key, value in raw.items():
        if not torch.is_tensor(value):
            raise SensitivityMapError(f"{path}: {key} is not a tensor")
        out[str(key)] = validate_sensitivity_vector(
            value,
            expected_channels=int(value.shape[0]),
            name=str(key),
        )
    metadata = payload.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise SensitivityMapError(f"{path}: metadata must be a dict")
    return out, metadata


def validate_certified_sensitivity_map_metadata(
    metadata: Mapping[str, Any],
    *,
    component: str | None = None,
    require_review_clean_passes: int = 3,
) -> dict[str, Any]:
    """Return a checked certification packet for a promotable sensitivity map."""
    if not isinstance(metadata, Mapping):
        raise SensitivityMapError("certified sensitivity metadata must be a mapping")
    if metadata.get("promotion_eligible") is not True:
        raise SensitivityMapError("certified sensitivity metadata requires promotion_eligible=true")
    if metadata.get("official_component_response") is not True:
        raise SensitivityMapError(
            "certified sensitivity metadata requires official_component_response=true"
        )
    if metadata.get("canonical_scorer_path") is not True:
        raise SensitivityMapError("certified sensitivity metadata requires canonical_scorer_path=true")
    require_authoritative_device(metadata.get("device"))

    certification = metadata.get("certification")
    if not isinstance(certification, Mapping):
        raise SensitivityMapError("certified sensitivity metadata requires certification mapping")
    out = {str(k): v for k, v in certification.items()}
    if out.get("format") != CERTIFIED_SENSITIVITY_MAP_CERTIFICATION_FORMAT:
        raise SensitivityMapError(
            "certification.format must be "
            f"{CERTIFIED_SENSITIVITY_MAP_CERTIFICATION_FORMAT!r}"
        )
    cert_component = out.get("component")
    if cert_component not in _CERTIFICATION_COMPONENTS:
        raise SensitivityMapError(f"certification.component is invalid: {cert_component!r}")
    if component is not None and cert_component != component:
        raise SensitivityMapError(
            f"certification.component={cert_component!r} does not match {component!r}"
        )
    require_authoritative_device(out.get("device"))
    for key in _CERTIFICATION_TRUE_FIELDS:
        if out.get(key) is not True:
            raise SensitivityMapError(f"certification.{key} must be exactly true")
    for key in _CERTIFICATION_SHA_FIELDS:
        value = out.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
            raise SensitivityMapError(f"certification.{key} must be a lowercase SHA-256 hex digest")
    archive_bytes = out.get("baseline_archive_bytes")
    if not isinstance(archive_bytes, int) or isinstance(archive_bytes, bool) or archive_bytes <= 0:
        raise SensitivityMapError("certification.baseline_archive_bytes must be a positive integer")
    clean_passes = out.get("review_clean_passes")
    if (
        not isinstance(clean_passes, int)
        or isinstance(clean_passes, bool)
        or clean_passes < require_review_clean_passes
    ):
        raise SensitivityMapError(
            "certification.review_clean_passes must be at least "
            f"{require_review_clean_passes}"
        )
    # IMPORTANT (audit 2026-05-06): require key present AND empty list.
    # Previously ``unresolved not in ([], None)`` accepted ``None`` as a silent
    # pass — a missing/null key would skip the gate. Tighten to require an
    # explicit empty list so absent or non-list values fail loud.
    if "review_unresolved_blockers" not in out:
        raise SensitivityMapError(
            "certification.review_unresolved_blockers is required (must be an empty list)"
        )
    unresolved = out.get("review_unresolved_blockers")
    if not isinstance(unresolved, list) or unresolved != []:
        raise SensitivityMapError(
            "certification.review_unresolved_blockers must be an empty list"
        )
    return out


def save_certified_sensitivity_map(
    path: str | Path,
    sensitivities: Mapping[str, torch.Tensor],
    *,
    component: str,
    certification: Mapping[str, Any],
) -> None:
    """Persist a promotion-capable map after validating its certification."""
    metadata = {
        "component": component,
        "device": "cuda",
        "promotion_eligible": True,
        "official_component_response": True,
        "canonical_scorer_path": True,
        "sensitivity_source": "certified_official_component_sensitivity",
        "certification": dict(certification),
    }
    validate_certified_sensitivity_map_metadata(metadata, component=component)
    save_sensitivity_map(path, sensitivities, metadata=metadata)


def sensitivity_cv_distance(
    train: Mapping[str, torch.Tensor],
    holdout: Mapping[str, torch.Tensor],
    *,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Return per-layer normalized L1 distance between train and holdout maps."""
    out: dict[str, float] = {}
    shared = sorted(set(train) & set(holdout))
    if not shared:
        raise SensitivityMapError("no shared sensitivity keys for CV comparison")
    for key in shared:
        if train[key].shape != holdout[key].shape:
            raise SensitivityMapError(
                f"{key}: train shape {tuple(train[key].shape)} != holdout shape "
                f"{tuple(holdout[key].shape)}"
            )
        a = validate_sensitivity_vector(
            train[key],
            expected_channels=int(train[key].shape[0]),
            name=f"{key}:train",
        )
        b = validate_sensitivity_vector(
            holdout[key],
            expected_channels=int(holdout[key].shape[0]),
            name=f"{key}:holdout",
        )
        a = a / (a.sum() + eps)
        b = b / (b.sum() + eps)
        out[key] = float((a - b).abs().sum().item())
    return out


__all__ = [
    "SENSITIVITY_MAP_FORMAT",
    "CERTIFIED_SENSITIVITY_MAP_CERTIFICATION_FORMAT",
    "SensitivityMapError",
    "SensitivityMapStats",
    "require_authoritative_device",
    "validate_sensitivity_vector",
    "conv_weight_shapes",
    "resolve_layer_sensitivity",
    "validate_sensitivity_map_for_model",
    "save_sensitivity_map",
    "load_sensitivity_map",
    "save_certified_sensitivity_map",
    "validate_certified_sensitivity_map_metadata",
    "sensitivity_cv_distance",
]
