# SPDX-License-Identifier: MIT
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

import hashlib
import json
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

SENSITIVITY_MAP_FORMAT = "tac_score_sensitivity_map_v1"
SENSITIVITY_PAIR_MANIFEST_FORMAT = "tac_sensitivity_pair_manifest_v1"
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
_REAL_SENSITIVITY_BAD_BOOL_KEYS = frozenset(
    {
        "advisory_only",
        "debug",
        "design_mode",
        "dummy",
        "dummy_sensitivity",
        "fake",
        "fake_sensitivity",
        "is_debug",
        "is_planning",
        "is_planning_only",
        "is_smoke",
        "is_stub",
        "local_proxy",
        "non_authoritative",
        "non_promotable",
        "planning",
        "planning_only",
        "proxy",
        "proxy_only",
        "random",
        "random_sensitivity",
        "smoke",
        "smoke_test",
        "source_archive_stale",
        "stale",
        "stub",
        "synthetic",
        "synthetic_sensitivity",
        "uniform",
        "uniform_sensitivity",
    }
)
_REAL_SENSITIVITY_TEXT_KEYS = frozenset(
    {
        "evidence_grade",
        "kind",
        "mode",
        "notes",
        "provenance",
        "source",
        "source_kind",
        "status",
        "tag",
    }
)
_REAL_SENSITIVITY_BAD_TEXT_MARKERS = (
    "stub",
    "planning-only",
    "planning_only",
    "design-mode",
    "design_mode",
    "advisory-only",
    "advisory_only",
    "advisory only",
    "debug",
    "diagnostic",
    "dummy",
    "fake",
    "local_proxy",
    "non_authoritative",
    "non_promotable",
    "placeholder",
    "proxy_only",
    "random_sensitivity",
    "smoke",
    "stale",
    "superseded",
    "synthetic",
    "uniform_sensitivity",
)
_REAL_SENSITIVITY_SOURCE_SHA_KEYS = (
    "source_archive_sha256",
    "source_archive_sha",
    "baseline_archive_sha256",
)
_REAL_SENSITIVITY_SOURCE_BYTES_KEYS = (
    "source_archive_bytes",
    "baseline_archive_bytes",
)
_REAL_SENSITIVITY_MODEL_SHA_KEYS = (
    "model_sha256",
    "checkpoint_sha256",
    "state_dict_sha256",
    "state_dict_source_sha256",
    "decoder_state_dict_sha256",
    "model_state_dict_sha256",
    "baseline_model_sha256",
    "baseline_checkpoint_sha256",
)
_REAL_SENSITIVITY_MODEL_ID_KEYS = (
    "model_id",
    "checkpoint_id",
    "state_dict_id",
    "decoder_model_id",
    "baseline_model_id",
)
_REAL_SENSITIVITY_CERTIFICATION_SUMMARY_SHA_KEYS = (
    "certification_summary_sha256",
    "component_sensitivity_summary_sha256",
    "component_sensitivity_manifest_sha256",
    "certified_component_sensitivity_manifest_sha256",
)
_REAL_SENSITIVITY_COMPONENT_KEYS = (
    "component",
    "component_scope",
    "scorer_target",
)


class SensitivityMapError(ValueError):
    """Raised when a sensitivity map is malformed or non-authoritative."""


@dataclass(frozen=True)
class SensitivityMapStats:
    """Summary of a validated sensitivity map."""

    n_layers: int
    n_channels: int
    min_value: float
    max_value: float


def canonical_sensitivity_json_bytes(payload: Mapping[str, Any]) -> bytes:
    """Return canonical JSON bytes for sensitivity manifests and SHA binding."""
    return (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")


def sensitivity_manifest_sha256(payload: Mapping[str, Any]) -> str:
    """Return SHA-256 over the canonical sensitivity-manifest JSON contract."""
    return hashlib.sha256(canonical_sensitivity_json_bytes(payload)).hexdigest()


def build_contiguous_pair_manifest(
    n_pairs: int,
    *,
    latent_rows: int,
    start_pair_index: int = 0,
    seq_len: int = 2,
    source_bindings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the deterministic pair/sample plan used by β-Fisher producers.

    The upstream evaluator consumes non-overlapping frame sequences of length
    two. PR106's HNeRV latents are row-aligned to those evaluator pair indices,
    so row ``i`` protects video frames ``2*i`` and ``2*i+1`` by default.
    """
    for label, value in {
        "n_pairs": n_pairs,
        "latent_rows": latent_rows,
        "start_pair_index": start_pair_index,
        "seq_len": seq_len,
    }.items():
        if not isinstance(value, int) or isinstance(value, bool):
            raise SensitivityMapError(f"{label} must be an integer")
    if n_pairs <= 0:
        raise SensitivityMapError("n_pairs must be positive")
    if latent_rows <= 0:
        raise SensitivityMapError("latent_rows must be positive")
    if start_pair_index < 0:
        raise SensitivityMapError("start_pair_index must be non-negative")
    if seq_len <= 0:
        raise SensitivityMapError("seq_len must be positive")
    if start_pair_index + n_pairs > latent_rows:
        raise SensitivityMapError(
            "pair manifest exceeds latent rows: "
            f"start_pair_index={start_pair_index}, n_pairs={n_pairs}, "
            f"latent_rows={latent_rows}"
        )

    pairs: list[dict[str, Any]] = []
    for pair_index in range(start_pair_index, start_pair_index + n_pairs):
        frame_start = pair_index * seq_len
        pairs.append(
            {
                "pair_index": pair_index,
                "latent_index": pair_index,
                "frame_start": frame_start,
                "frame_indices": [frame_start + offset for offset in range(seq_len)],
            }
        )

    manifest: dict[str, Any] = {
        "format": SENSITIVITY_PAIR_MANIFEST_FORMAT,
        "selection": "contiguous_pr106_latent_rows",
        "pair_index_base": 0,
        "start_pair_index": start_pair_index,
        "n_pairs": n_pairs,
        "latent_rows": latent_rows,
        "seq_len": seq_len,
        "pairs": pairs,
    }
    if source_bindings:
        manifest["source_bindings"] = dict(source_bindings)
    return manifest


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


def _flatten_metadata(metadata: Mapping[str, Any], prefix: str = ""):
    for key, value in metadata.items():
        full_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            yield from _flatten_metadata(value, full_key)
        else:
            yield full_key, value


def _metadata_value(metadata: Mapping[str, Any], keys: tuple[str, ...]) -> Any | None:
    for key in keys:
        if key in metadata:
            return metadata[key]
    for container in ("source_archive", "baseline_archive", "source", "certification"):
        nested = metadata.get(container)
        if not isinstance(nested, Mapping):
            continue
        for key in keys:
            if key in nested:
                return nested[key]
        if any(candidate.endswith("sha256") for candidate in keys):
            for key in ("sha256", "archive_sha256"):
                if key in nested:
                    return nested[key]
        if any(candidate.endswith("bytes") for candidate in keys) and "bytes" in nested:
            return nested["bytes"]
    return None


def _metadata_values(metadata: Mapping[str, Any], keys: tuple[str, ...]):
    for key in keys:
        if key in metadata:
            yield key, metadata[key]
    for container in ("source_archive", "baseline_archive", "source", "model", "checkpoint", "certification"):
        nested = metadata.get(container)
        if not isinstance(nested, Mapping):
            continue
        for key in keys:
            if key in nested:
                yield f"{container}.{key}", nested[key]
        if any(candidate.endswith("sha256") for candidate in keys):
            for key in ("sha256", "archive_sha256", "model_sha256", "checkpoint_sha256"):
                if key in nested:
                    yield f"{container}.{key}", nested[key]
        if any(candidate.endswith("bytes") for candidate in keys) and "bytes" in nested:
            yield f"{container}.bytes", nested["bytes"]


def _normalise_expected_sha256(value: str, *, label: str) -> str:
    digest = str(value).strip().lower()
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise SensitivityMapError(f"expected {label} SHA must be a 64-hex digest")
    return digest


def _compare_expected_sha256(
    metadata: Mapping[str, Any],
    keys: tuple[str, ...],
    expected: str,
    *,
    label: str,
) -> list[str]:
    failures: list[str] = []
    try:
        actual = _normalise_expected_sha256(expected, label=label)
    except SensitivityMapError as exc:
        return [str(exc)]
    observed = list(_metadata_values(metadata, keys))
    if not observed:
        return [f"metadata missing 64-hex {label} SHA"]
    for source, recorded in observed:
        if not isinstance(recorded, str):
            failures.append(f"metadata {source} {label} SHA is not a string: {recorded!r}")
            continue
        digest = recorded.strip().lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            failures.append(f"metadata {source} {label} SHA is not a 64-hex digest: {recorded!r}")
            continue
        if digest != actual:
            failures.append(
                f"metadata {source} {label} SHA is stale or mismatched: "
                f"metadata={digest} actual={actual}"
            )
    return failures


def _compare_expected_string(
    metadata: Mapping[str, Any],
    keys: tuple[str, ...],
    expected: str,
    *,
    label: str,
) -> list[str]:
    failures: list[str] = []
    actual = str(expected)
    observed = list(_metadata_values(metadata, keys))
    if not observed:
        return [f"metadata missing {label} binding"]
    for source, recorded in observed:
        if not isinstance(recorded, str):
            failures.append(f"metadata {source} {label} is not a string: {recorded!r}")
        elif recorded != actual:
            failures.append(
                f"metadata {source} {label} is stale or mismatched: "
                f"metadata={recorded!r} actual={actual!r}"
            )
    return failures


def _component_scope_failures(metadata: Mapping[str, Any], component: str) -> list[str]:
    failures: list[str] = []
    for key in _REAL_SENSITIVITY_COMPONENT_KEYS:
        if key not in metadata:
            continue
        value = metadata[key]
        if isinstance(value, str):
            if value != component:
                failures.append(
                    f"metadata {key}={value!r} does not match component={component!r}"
                )
        elif isinstance(value, (list, tuple)):
            if component not in value:
                failures.append(
                    f"metadata {key} does not include component={component!r}: {value!r}"
                )
        elif value is not None:
            failures.append(f"metadata {key} has unsupported component binding: {value!r}")
    return failures


def real_sensitivity_metadata_blockers(metadata: Mapping[str, Any]) -> list[str]:
    """Return stub/proxy/planning markers that disqualify a real map."""
    if not isinstance(metadata, Mapping):
        return ["metadata is not a mapping"]
    blockers: list[str] = []
    for full_key, value in _flatten_metadata(metadata):
        key = full_key.rsplit(".", 1)[-1]
        key_norm = key.lower().replace("-", "_")
        if key_norm in _REAL_SENSITIVITY_BAD_BOOL_KEYS and value is True:
            blockers.append(f"{full_key}=true")
        if key_norm in _REAL_SENSITIVITY_TEXT_KEYS and isinstance(value, str):
            value_norm = value.lower()
            for marker in _REAL_SENSITIVITY_BAD_TEXT_MARKERS:
                if marker in value_norm:
                    blockers.append(f"{full_key} contains {marker!r}")
                    break
        if key_norm in {"status", "mode", "kind"} and isinstance(value, str):
            value_exact = value.strip().lower().replace("_", "-")
            if value_exact in {"planning", "design", "debug", "proxy", "prototype"}:
                blockers.append(f"{full_key}={value!r}")
    return blockers


def validate_real_sensitivity_artifact(
    sensitivities: Mapping[str, torch.Tensor],
    metadata: Mapping[str, Any],
    *,
    source_archive_sha256: str | None = None,
    source_archive_bytes: int | None = None,
    model_sha256: str | None = None,
    checkpoint_sha256: str | None = None,
    model_id: str | None = None,
    certification_summary_sha256: str | None = None,
    component: str | None = None,
    reject_uniform: bool = True,
) -> dict[str, Any]:
    """Fail closed unless a sensitivity artifact is promotion-capable.

    This is the integration gate for archive builders and dispatch dry-runs.
    It rejects diagnostic/stub/proxy metadata, requires certified CUDA
    component-response provenance, checks optional source archive / model /
    certification-summary custody, and refuses empty, all-zero, or uniform
    tensor payloads before candidate bytes are emitted.
    """
    if not isinstance(sensitivities, Mapping) or not sensitivities:
        raise SensitivityMapError("real sensitivity artifact requires non-empty sensitivities")
    if not isinstance(metadata, Mapping):
        raise SensitivityMapError("real sensitivity artifact requires metadata mapping")

    failures: list[str] = []
    failures.extend(real_sensitivity_metadata_blockers(metadata))

    try:
        certification = validate_certified_sensitivity_map_metadata(
            metadata,
            component=component,
        )
    except SensitivityMapError as exc:
        failures.append(f"certification rejected: {exc}")
        certification = {}

    if component is not None:
        failures.extend(_component_scope_failures(metadata, component))

    flat_values: list[torch.Tensor] = []
    for key, value in sensitivities.items():
        if not torch.is_tensor(value):
            failures.append(f"{key}: sensitivity is not a tensor")
            continue
        try:
            vec = validate_sensitivity_vector(
                value.reshape(-1),
                expected_channels=int(value.numel()),
                name=str(key),
            )
        except SensitivityMapError as exc:
            failures.append(str(exc))
            continue
        if vec.numel() == 0:
            failures.append(f"{key}: sensitivity is empty")
            continue
        flat_values.append(vec)

    if not flat_values:
        failures.append("sensitivity artifact has no tensor values")
    else:
        all_values = torch.cat(flat_values).float()
        if float(all_values.clamp_min(0).sum().item()) <= 0.0:
            failures.append("sensitivity artifact has no positive scorer signal")
        if reject_uniform and all_values.numel() > 1 and torch.allclose(
            all_values,
            all_values[:1].expand_as(all_values),
            rtol=0.0,
            atol=1e-12,
        ):
            failures.append(
                "sensitivity artifact is uniform; dummy/stub sensitivity is non-promotable"
            )

    if source_archive_sha256 is not None:
        failures.extend(
            _compare_expected_sha256(
                metadata,
                _REAL_SENSITIVITY_SOURCE_SHA_KEYS,
                source_archive_sha256,
                label="source archive",
            )
        )

    if source_archive_bytes is not None:
        actual_bytes = int(source_archive_bytes)
        recorded_bytes = _metadata_value(metadata, _REAL_SENSITIVITY_SOURCE_BYTES_KEYS)
        if recorded_bytes is not None:
            try:
                recorded_bytes_int = int(recorded_bytes)
            except (TypeError, ValueError):
                failures.append(
                    f"metadata source archive bytes is not an integer: {recorded_bytes!r}"
                )
            else:
                if recorded_bytes_int != actual_bytes:
                    failures.append(
                        "metadata source archive bytes is stale or mismatched: "
                        f"metadata={recorded_bytes_int} actual={actual_bytes}"
                    )
        certified_bytes = certification.get("baseline_archive_bytes")
        if (
            isinstance(certified_bytes, int)
            and not isinstance(certified_bytes, bool)
            and certified_bytes != actual_bytes
        ):
            failures.append(
                "certification baseline_archive_bytes is stale or mismatched: "
                f"metadata={certified_bytes} actual={actual_bytes}"
            )

    expected_model_sha256 = model_sha256
    if checkpoint_sha256 is not None:
        if expected_model_sha256 is not None:
            try:
                model_digest = _normalise_expected_sha256(expected_model_sha256, label="model")
                checkpoint_digest = _normalise_expected_sha256(checkpoint_sha256, label="checkpoint")
            except SensitivityMapError as exc:
                failures.append(str(exc))
                checkpoint_digest = model_digest = ""
            if model_digest and checkpoint_digest and model_digest != checkpoint_digest:
                failures.append(
                    "model_sha256 and checkpoint_sha256 expectations disagree: "
                    f"model={model_digest} checkpoint={checkpoint_digest}"
                )
        else:
            expected_model_sha256 = checkpoint_sha256
    if expected_model_sha256 is not None:
        failures.extend(
            _compare_expected_sha256(
                metadata,
                _REAL_SENSITIVITY_MODEL_SHA_KEYS,
                expected_model_sha256,
                label="model/checkpoint",
            )
        )
    if model_id is not None:
        failures.extend(
            _compare_expected_string(
                metadata,
                _REAL_SENSITIVITY_MODEL_ID_KEYS,
                model_id,
                label="model id",
            )
        )
    if certification_summary_sha256 is not None:
        failures.extend(
            _compare_expected_sha256(
                metadata,
                _REAL_SENSITIVITY_CERTIFICATION_SUMMARY_SHA_KEYS,
                certification_summary_sha256,
                label="certification summary",
            )
        )

    if failures:
        raise SensitivityMapError("; ".join(failures))

    n_values = int(sum(vec.numel() for vec in flat_values))
    return {
        "n_tensors": len(sensitivities),
        "n_values": n_values,
        "certification": certification,
        "source_archive_sha256": source_archive_sha256,
        "source_archive_bytes": source_archive_bytes,
        "model_sha256": expected_model_sha256,
        "model_id": model_id,
        "certification_summary_sha256": certification_summary_sha256,
    }


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
    "CERTIFIED_SENSITIVITY_MAP_CERTIFICATION_FORMAT",
    "SENSITIVITY_MAP_FORMAT",
    "SENSITIVITY_PAIR_MANIFEST_FORMAT",
    "SensitivityMapError",
    "SensitivityMapStats",
    "build_contiguous_pair_manifest",
    "canonical_sensitivity_json_bytes",
    "conv_weight_shapes",
    "load_sensitivity_map",
    "real_sensitivity_metadata_blockers",
    "require_authoritative_device",
    "resolve_layer_sensitivity",
    "save_certified_sensitivity_map",
    "save_sensitivity_map",
    "sensitivity_cv_distance",
    "sensitivity_manifest_sha256",
    "validate_certified_sensitivity_map_metadata",
    "validate_real_sensitivity_artifact",
    "validate_sensitivity_map_for_model",
    "validate_sensitivity_vector",
]


# COUNCIL-A1 axis-level reweighting API (2026-05-12).
#
# Re-export the operating-point-aware per-axis EV multipliers from
# the canonical :mod:`tac.sensitivity_map.axis_weights` submodule so
# downstream consumers (the FIX-C composition bridge + the GGGG A-1
# probe-disambiguator) can ``from tac.sensitivity_map import AxisWeights``.
# Per CLAUDE.md "Subagent coherence-by-default": ONE canonical surface,
# every consumer plugs into the same module. Memory:
# feedback_council_a1_sensitivity_map_axis_api_landed_20260512.md.
from tac.sensitivity_map import axis_weights as _axis_weights  # noqa: E402
from tac.sensitivity_map.axis_weights import (  # noqa: E402
    AXIS_NAMES,
    OLD_1X_OPERATING_POINT_AXIS_WEIGHTS,
    OPERATING_POINT_ANCHORS,
    PR102_CUDA_AXIS_WEIGHTS,
    PR106_R2_FRONTIER_AXIS_WEIGHTS,
    PR106_R2_POSE_PER_SEG_MARGINAL_RATIO,
    AxisWeights,
    AxisWeightsError,
    axis_weights_for_named_operating_point,
    compute_axis_weights,
    default_axis_weights,
    validate_axis_weights_mapping,
)

__all__ += [
    "AXIS_NAMES",
    "OLD_1X_OPERATING_POINT_AXIS_WEIGHTS",
    "OPERATING_POINT_ANCHORS",
    "PR102_CUDA_AXIS_WEIGHTS",
    "PR106_R2_FRONTIER_AXIS_WEIGHTS",
    "PR106_R2_POSE_PER_SEG_MARGINAL_RATIO",
    "AxisWeights",
    "AxisWeightsError",
    "axis_weights",
    "axis_weights_for_named_operating_point",
    "compute_axis_weights",
    "default_axis_weights",
    "validate_axis_weights_mapping",
]

# Expose submodule under its package attribute name for
# ``tac.sensitivity_map.axis_weights`` dotted access without explicit
# import in the consumer.
axis_weights = _axis_weights


# Wire-in 2 — Wyner-Ziv side-info covariance (Catalog #125 hook #1).
#
# Re-export the canonical per-byte reweighting API from the sister
# :mod:`tac.sensitivity_map.wyner_ziv_reweight` submodule so downstream
# consumers can ``from tac.sensitivity_map import axis_level_reweight``.
# The producer side is :func:`tac.master_gradient_consumers.wyner_ziv_side_info_covariance`
# (consumer 4). Memory: feedback_wire_in_2_wyner_ziv_to_sensitivity_map_landed_20260517.md.
# Lane: lane_wire_in_2_wyner_ziv_covariance_to_sensitivity_map_20260517.
from tac.sensitivity_map import wyner_ziv_reweight as _wyner_ziv_reweight  # noqa: E402
from tac.sensitivity_map.wyner_ziv_reweight import (  # noqa: E402
    MIXED_SENSITIVITY_BASELINE,
    PAIR_SPECIFIC_SENSITIVITY_UPWEIGHT,
    SHARED_PRIOR_SENSITIVITY_DOWNWEIGHT,
    WYNER_ZIV_REWEIGHT_OPERATING_POINT_TAG,
    WynerZivAxisLevelReweightError,
    axis_level_reweight,
    update_sensitivity_map_from_master_gradient_anchor,
)

__all__ += [
    "MIXED_SENSITIVITY_BASELINE",
    "PAIR_SPECIFIC_SENSITIVITY_UPWEIGHT",
    "SHARED_PRIOR_SENSITIVITY_DOWNWEIGHT",
    "WYNER_ZIV_REWEIGHT_OPERATING_POINT_TAG",
    "WynerZivAxisLevelReweightError",
    "axis_level_reweight",
    "update_sensitivity_map_from_master_gradient_anchor",
    "wyner_ziv_reweight",
]

# Expose submodule under its package attribute name for
# ``tac.sensitivity_map.wyner_ziv_reweight`` dotted access without explicit
# import in the consumer.
wyner_ziv_reweight = _wyner_ziv_reweight
