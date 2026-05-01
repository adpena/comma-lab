#!/usr/bin/env python3
"""Build promotable J-NWC/J-NWCS sensitivity input artifacts.

This script converts a validated ``component_sensitivity_v1`` manifest into
the two artifacts consumed by the J-NWCS remote scripts:

* ``ANCHOR_SENSITIVITY_PT``: per-anchor-parameter block sensitivities with
  component-manifest, archive, renderer, and parameter custody metadata.
* ``CORPUS_SENSITIVITY_PT``: a corpus-manifest-aligned 1-D block vector with
  component-manifest and corpus-manifest custody metadata.

It is intentionally bounded and fail-closed. Conv channel maps may be expanded
to Conv/ConvTranspose weights, but every float anchor parameter with full
NWCS blocks and every selected corpus tensor must be covered by real map data.
No uniform/fake/debug/proxy fallback values are generated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.component_sensitivity_artifact import (  # noqa: E402
    ComponentSensitivityArtifactError,
    validate_component_sensitivity_manifest,
)
from tac.neural_weight_corpus import (  # noqa: E402
    CorpusManifestError,
    _classify_tensor,
    _extract_state_dict,
    _manifest_checkpoint_path,
    load_corpus_manifest,
)
from tac.renderer_export import load_any_renderer_checkpoint  # noqa: E402


ANCHOR_OUTPUT_FORMAT = "tac.nwcs_anchor_sensitivity_inputs.v1"
CORPUS_OUTPUT_FORMAT = "tac.nwcs_corpus_sensitivity_inputs.v1"
CHANNEL_MAP_FORMAT = "tac_score_sensitivity_map_v1"

_DIAGNOSTIC_TRUE_KEYS = {
    "debug",
    "debug_mode",
    "dummy",
    "dummy_sensitivity",
    "fake",
    "fake_sensitivity",
    "is_debug",
    "is_smoke",
    "local_proxy",
    "non_authoritative",
    "non_promotable",
    "proxy",
    "proxy_only",
    "random",
    "random_sensitivity",
    "score_proxy",
    "scorer_proxy",
    "smoke",
    "smoke_test",
    "synthetic_sensitivity",
    "uniform",
    "uniform_sensitivity",
    "use_proxy",
    "uses_proxy",
}
_DIAGNOSTIC_STRING_MARKERS = (
    "diagnostic",
    "debug",
    "dummy",
    "from_fisher",
    "fisher",
    "fake_sensitivity",
    "fisher_proxy",
    "local_proxy",
    "non_authoritative",
    "non_promotable",
    "profile_component_sensitivity.py",
    "proxy_only",
    "random_sensitivity",
    "smoke",
    "synthetic_sensitivity",
    "uniform_sensitivity",
)
_PER_BLOCK_MARKERS = {
    "block",
    "blocks",
    "per_block",
    "per_parameter_block",
    "parameter_block",
    "nwcs_block",
}
_PER_ELEMENT_MARKERS = {
    "element",
    "per_element",
    "per_weight",
    "parameter_tensor",
    "full_tensor",
}
_CHANNEL_MARKERS = {
    "channel",
    "channels",
    "per_channel",
    "output_channel",
    "conv_output_channel",
}


class NWCSSensitivityInputBuildError(ValueError):
    """Raised when promotable NWCS sensitivity inputs cannot be built."""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise NWCSSensitivityInputBuildError(f"{path}: invalid JSON") from exc


def _resolve_referenced_path(value: Any, *, base_dir: Path) -> Path:
    if not isinstance(value, str) or not value:
        raise NWCSSensitivityInputBuildError(f"referenced path must be a string, got {value!r}")
    path = Path(value)
    if path.is_absolute() or path.exists():
        return path
    candidate = base_dir / path
    if candidate.exists():
        return candidate
    return path


def _assert_file_custody(entry: Mapping[str, Any], path: Path, *, label: str) -> None:
    if not path.exists():
        raise NWCSSensitivityInputBuildError(f"{label} does not exist: {path}")
    actual_bytes = path.stat().st_size
    expected_bytes = entry.get("bytes", entry.get("size_bytes"))
    if expected_bytes is not None and int(expected_bytes) != actual_bytes:
        raise NWCSSensitivityInputBuildError(
            f"{label} bytes mismatch: manifest={expected_bytes} actual={actual_bytes}"
        )
    expected_sha = entry.get("sha256")
    if expected_sha is not None:
        actual_sha = _sha256_file(path)
        if str(expected_sha) != actual_sha:
            raise NWCSSensitivityInputBuildError(
                f"{label} sha256 mismatch: manifest={expected_sha} actual={actual_sha}"
            )


def _load_component_manifest(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    if not isinstance(payload, dict):
        raise NWCSSensitivityInputBuildError(
            f"{path}: component_sensitivity_v1 manifest must be a JSON object"
        )
    try:
        validate_component_sensitivity_manifest(payload, promotion=True)
    except ComponentSensitivityArtifactError as exc:
        raise NWCSSensitivityInputBuildError(
            f"{path}: not a promotable component_sensitivity_v1 manifest: {exc}"
        ) from exc
    return payload


def _assert_component_manifest_matches_inputs(
    manifest: Mapping[str, Any],
    *,
    component_manifest_path: Path,
    anchor_renderer: Path,
    anchor_archive: Path,
) -> None:
    inputs = manifest.get("inputs")
    if not isinstance(inputs, Mapping):
        raise NWCSSensitivityInputBuildError("component manifest missing inputs")
    checkpoint_entry = inputs.get("checkpoint")
    if not isinstance(checkpoint_entry, Mapping):
        raise NWCSSensitivityInputBuildError("component manifest missing inputs.checkpoint")
    _assert_file_custody(
        checkpoint_entry,
        anchor_renderer,
        label="anchor renderer vs component manifest checkpoint",
    )

    contest_eval = manifest.get("contest_eval")
    if not isinstance(contest_eval, Mapping):
        raise NWCSSensitivityInputBuildError("component manifest missing contest_eval")
    archive_entry = contest_eval.get("archive")
    if not isinstance(archive_entry, Mapping):
        raise NWCSSensitivityInputBuildError("component manifest missing contest_eval.archive")
    _assert_file_custody(
        archive_entry,
        anchor_archive,
        label="anchor archive vs component manifest archive",
    )

    component_maps = manifest.get("component_maps")
    if not isinstance(component_maps, Mapping):
        raise NWCSSensitivityInputBuildError("component manifest missing component_maps")
    combined_entry = component_maps.get("combined")
    if not isinstance(combined_entry, Mapping):
        raise NWCSSensitivityInputBuildError("component manifest missing component_maps.combined")
    combined_path = _resolve_referenced_path(
        combined_entry.get("path"),
        base_dir=component_manifest_path.parent,
    )
    _assert_file_custody(
        combined_entry,
        combined_path,
        label="combined component sensitivity map",
    )


def _reject_diagnostic_markers(value: Any, *, context: str) -> None:
    def walk(item: Any, path: str) -> None:
        if torch.is_tensor(item):
            return
        if isinstance(item, Mapping):
            for raw_key, child in item.items():
                key = str(raw_key)
                norm = key.lower().replace("-", "_")
                child_path = f"{path}.{key}"
                if norm in _DIAGNOSTIC_TRUE_KEYS and child is True:
                    raise NWCSSensitivityInputBuildError(
                        f"{context}: diagnostic/non-promotable marker {child_path}=true"
                    )
                if norm == "promotion_eligible" and child is not True:
                    raise NWCSSensitivityInputBuildError(
                        f"{context}: promotion_eligible must be true, got {child!r}"
                    )
                walk(child, child_path)
            return
        if isinstance(item, (list, tuple)):
            for index, child in enumerate(item):
                walk(child, f"{path}[{index}]")
            return
        if isinstance(item, str):
            lowered = item.lower()
            for marker in _DIAGNOSTIC_STRING_MARKERS:
                if marker in lowered:
                    raise NWCSSensitivityInputBuildError(
                        f"{context}: diagnostic/non-promotable marker at {path}: {item!r}"
                    )

    walk(value, context)


def _load_combined_component_map(
    manifest: Mapping[str, Any],
    *,
    component_manifest_path: Path,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    component_maps = manifest["component_maps"]
    combined_entry = component_maps["combined"]
    path = _resolve_referenced_path(
        combined_entry["path"],
        base_dir=component_manifest_path.parent,
    )
    payload = torch.load(str(path), map_location="cpu", weights_only=False)
    _reject_diagnostic_markers(payload, context=str(path))

    fmt = None
    metadata: dict[str, Any] = {}
    raw: Any = payload
    if isinstance(payload, Mapping):
        fmt_value = payload.get("format")
        fmt = str(fmt_value) if isinstance(fmt_value, str) else None
        meta_value = payload.get("metadata")
        if isinstance(meta_value, Mapping):
            metadata = {str(k): v for k, v in meta_value.items()}
        for key in ("sensitivities", "component_sensitivities", "maps", "values"):
            if key in payload:
                raw = payload[key]
                break

    if not isinstance(raw, Mapping):
        raise NWCSSensitivityInputBuildError(
            f"{path}: combined component map must contain a sensitivity mapping"
        )

    out: dict[str, torch.Tensor] = {}
    for raw_key, value in raw.items():
        if not torch.is_tensor(value):
            raise NWCSSensitivityInputBuildError(
                f"{path}: sensitivity {raw_key!r} is not a tensor"
            )
        tensor = value.detach().cpu().float()
        if tensor.numel() == 0:
            raise NWCSSensitivityInputBuildError(
                f"{path}: sensitivity {raw_key!r} is empty"
            )
        if not torch.isfinite(tensor).all():
            raise NWCSSensitivityInputBuildError(
                f"{path}: sensitivity {raw_key!r} contains NaN/Inf"
            )
        if (tensor < 0).any():
            raise NWCSSensitivityInputBuildError(
                f"{path}: sensitivity {raw_key!r} contains negative values"
            )
        out[str(raw_key)] = tensor

    if not out:
        raise NWCSSensitivityInputBuildError(f"{path}: combined component map is empty")
    _reject_uniform_tensor_collection(out, label="combined component map")
    return out, {"path": str(path), "format": fmt, "metadata": metadata}


def _reject_uniform_tensor_collection(
    values: Mapping[str, torch.Tensor],
    *,
    label: str,
) -> None:
    flat = [
        tensor.detach().cpu().float().reshape(-1)
        for tensor in values.values()
        if tensor.numel() > 0
    ]
    if not flat:
        raise NWCSSensitivityInputBuildError(f"{label} contains no sensitivity values")
    all_values = torch.cat(flat)
    total_signal = float(all_values.clamp_min(0).sum().item())
    if total_signal <= 0.0:
        raise NWCSSensitivityInputBuildError(f"{label} contains no positive scorer signal")
    if all_values.numel() > 1 and torch.allclose(
        all_values,
        all_values[:1].expand_as(all_values),
        rtol=0.0,
        atol=1e-12,
    ):
        raise NWCSSensitivityInputBuildError(
            f"{label} is uniform; uniform/fake sensitivity is non-promotable"
        )


def _granularity(metadata: Mapping[str, Any]) -> str | None:
    for key in (
        "sensitivity_granularity",
        "granularity",
        "map_granularity",
        "sensitivity_kind",
    ):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            return value.lower().replace("-", "_")
    return None


def _block_reduce_per_element(
    tensor: torch.Tensor,
    *,
    block_size: int,
    name: str,
) -> torch.Tensor:
    flat = tensor.detach().cpu().float().reshape(-1)
    n_blocks = flat.numel() // block_size
    if n_blocks <= 0:
        raise NWCSSensitivityInputBuildError(f"{name}: tensor has no full NWCS blocks")
    out = flat[: n_blocks * block_size].reshape(n_blocks, block_size).mean(dim=1)
    return _validate_block_vector(out, name=name, expected_len=n_blocks)


def _validate_block_vector(
    tensor: torch.Tensor,
    *,
    name: str,
    expected_len: int,
) -> torch.Tensor:
    out = tensor.detach().cpu().float().reshape(-1)
    if int(out.numel()) != int(expected_len):
        raise NWCSSensitivityInputBuildError(
            f"{name}: sensitivity block length {out.numel()} != expected {expected_len}"
        )
    if not torch.isfinite(out).all():
        raise NWCSSensitivityInputBuildError(f"{name}: block sensitivity contains NaN/Inf")
    if (out < 0).any():
        raise NWCSSensitivityInputBuildError(f"{name}: block sensitivity contains negative values")
    if float(out.clamp_min(0).sum().item()) <= 0.0:
        raise NWCSSensitivityInputBuildError(f"{name}: block sensitivity has no positive scorer signal")
    return out


def _channel_count_for_param(
    name: str,
    param: torch.Tensor,
    module: nn.Module | None,
) -> tuple[int, int] | None:
    if not name.endswith(".weight") or module is None:
        return None
    if isinstance(module, nn.Conv2d):
        return int(param.shape[0]), 0
    if isinstance(module, nn.ConvTranspose2d):
        return int(param.shape[1]), 1
    return None


def _expand_channel_map(
    value: torch.Tensor,
    *,
    param: torch.Tensor,
    channel_dim: int,
) -> torch.Tensor:
    shape = [1] * int(param.dim())
    shape[channel_dim] = int(value.numel())
    return value.reshape(shape).expand(tuple(int(s) for s in param.shape))


def _parameter_block_sensitivity(
    *,
    name: str,
    param: torch.Tensor,
    map_value: torch.Tensor,
    map_metadata: Mapping[str, Any],
    map_format: str | None,
    module: nn.Module | None,
    block_size: int,
) -> tuple[torch.Tensor, str]:
    n_blocks = int(param.numel() // block_size)
    if n_blocks <= 0:
        raise NWCSSensitivityInputBuildError(f"{name}: parameter has no full blocks")

    granularity = _granularity(map_metadata)
    value = map_value.detach().cpu().float()
    if tuple(value.shape) == tuple(int(s) for s in param.shape):
        return (
            _block_reduce_per_element(value, block_size=block_size, name=name),
            "per_element",
        )

    if (
        value.reshape(-1).numel() == int(param.numel())
        and granularity in _PER_ELEMENT_MARKERS
    ):
        return (
            _block_reduce_per_element(
                value.reshape(tuple(int(s) for s in param.shape)),
                block_size=block_size,
                name=name,
            ),
            "per_element_flat",
        )

    flat = value.reshape(-1)
    channel_info = _channel_count_for_param(name, param, module)
    is_channel = channel_info is not None and int(flat.numel()) == channel_info[0]
    is_block = int(flat.numel()) == n_blocks

    if is_channel and is_block and granularity is None and map_format != CHANNEL_MAP_FORMAT:
        raise NWCSSensitivityInputBuildError(
            f"{name}: ambiguous 1-D sensitivity length {flat.numel()} could be "
            "per-block or per-channel; record sensitivity_granularity"
        )

    if is_channel and (
        map_format == CHANNEL_MAP_FORMAT
        or granularity in _CHANNEL_MARKERS
        or (granularity is None and not is_block)
    ):
        assert channel_info is not None
        expanded = _expand_channel_map(
            flat,
            param=param,
            channel_dim=channel_info[1],
        )
        return (
            _block_reduce_per_element(expanded, block_size=block_size, name=name),
            "per_output_channel",
        )

    if is_block:
        if granularity is None and map_format == CHANNEL_MAP_FORMAT:
            raise NWCSSensitivityInputBuildError(
                f"{name}: channel-map artifact cannot be reinterpreted as block sensitivity"
            )
        if granularity is not None and granularity not in _PER_BLOCK_MARKERS:
            raise NWCSSensitivityInputBuildError(
                f"{name}: sensitivity_granularity={granularity!r} is incompatible "
                "with a per-block vector"
            )
        return (
            _validate_block_vector(flat, name=name, expected_len=n_blocks),
            "per_block",
        )

    expected = [f"shape={tuple(int(s) for s in param.shape)}", f"blocks={n_blocks}"]
    if channel_info is not None:
        expected.append(f"channels={channel_info[0]}")
    raise NWCSSensitivityInputBuildError(
        f"{name}: sensitivity shape {tuple(value.shape)} does not match "
        f"expected {' or '.join(expected)}"
    )


def _build_anchor_sensitivities(
    model: nn.Module,
    component_map: Mapping[str, torch.Tensor],
    *,
    map_info: Mapping[str, Any],
    block_size: int,
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, Any]]]:
    modules = dict(model.named_modules())
    map_metadata = map_info.get("metadata") if isinstance(map_info.get("metadata"), Mapping) else {}
    map_format_value = map_info.get("format")
    map_format = str(map_format_value) if isinstance(map_format_value, str) else None

    sensitivities: dict[str, torch.Tensor] = {}
    metadata: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    failures: list[str] = []

    for name, param in model.named_parameters():
        if not torch.is_floating_point(param):
            continue
        n_blocks = int(param.numel() // block_size)
        if n_blocks <= 0:
            continue
        map_value = component_map.get(name)
        if map_value is None:
            missing.append(name)
            continue
        module_name = name.rsplit(".", 1)[0] if "." in name else ""
        module = modules.get(module_name)
        try:
            block_sens, source_kind = _parameter_block_sensitivity(
                name=name,
                param=param.detach().cpu(),
                map_value=map_value,
                map_metadata=map_metadata,
                map_format=map_format,
                module=module,
                block_size=block_size,
            )
        except NWCSSensitivityInputBuildError as exc:
            failures.append(str(exc))
            continue
        sensitivities[name] = block_sens
        metadata[name] = {
            "shape": [int(s) for s in param.shape],
            "dtype": str(param.dtype),
            "numel": int(param.numel()),
            "block_count": int(block_sens.numel()),
            "tail_elements": int(param.numel() - block_sens.numel() * block_size),
            "source_kind": source_kind,
        }

    if missing:
        preview = ", ".join(missing[:8])
        suffix = "..." if len(missing) > 8 else ""
        raise NWCSSensitivityInputBuildError(
            "combined component map does not cover "
            f"{len(missing)} float anchor parameter(s): {preview}{suffix}"
        )
    if failures:
        preview = "; ".join(failures[:5])
        suffix = " ..." if len(failures) > 5 else ""
        raise NWCSSensitivityInputBuildError(
            f"could not derive anchor block sensitivities: {preview}{suffix}"
        )
    if not sensitivities:
        raise NWCSSensitivityInputBuildError("anchor model has no covered float parameters")
    _reject_uniform_tensor_collection(sensitivities, label="anchor block sensitivities")
    return sensitivities, metadata


def _build_corpus_sensitivities(
    corpus_manifest_path: Path,
    *,
    anchor_sensitivities: Mapping[str, torch.Tensor],
    anchor_parameter_metadata: Mapping[str, Mapping[str, Any]],
    replay_root: Path | None,
    block_size: int,
) -> tuple[torch.Tensor, list[dict[str, Any]], dict[str, Any]]:
    try:
        manifest = load_corpus_manifest(corpus_manifest_path)
    except CorpusManifestError as exc:
        raise NWCSSensitivityInputBuildError(
            f"{corpus_manifest_path}: invalid corpus manifest: {exc}"
        ) from exc

    selection = manifest.get("selection")
    if not isinstance(selection, Mapping):
        raise NWCSSensitivityInputBuildError("corpus manifest missing selection")
    corpus_block_size = int(selection.get("block_size", 0))
    if corpus_block_size != int(block_size):
        raise NWCSSensitivityInputBuildError(
            f"corpus manifest block_size={corpus_block_size} != requested {block_size}"
        )

    out_parts: list[torch.Tensor] = []
    tensor_records: list[dict[str, Any]] = []
    expected_next = 0
    replay_root_path = replay_root.resolve(strict=False) if replay_root is not None else None

    for file_entry in manifest.get("files", []):
        if not isinstance(file_entry, Mapping) or not file_entry.get("selected"):
            continue
        try:
            checkpoint_path = _manifest_checkpoint_path(
                file_entry,
                replay_root=replay_root_path,
            )
        except CorpusManifestError as exc:
            raise NWCSSensitivityInputBuildError(str(exc)) from exc
        actual_size = checkpoint_path.stat().st_size
        expected_size = file_entry.get("size_bytes")
        if expected_size is None or int(expected_size) != actual_size:
            raise NWCSSensitivityInputBuildError(
                f"corpus manifest size mismatch for {checkpoint_path}: "
                f"manifest={expected_size} actual={actual_size}"
            )
        expected_sha = file_entry.get("sha256")
        if not isinstance(expected_sha, str) or expected_sha != _sha256_file(checkpoint_path):
            raise NWCSSensitivityInputBuildError(
                f"corpus manifest sha256 mismatch for {checkpoint_path}"
            )
        obj = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
        state_dict, _state_dict_key = _extract_state_dict(obj)
        if state_dict is None:
            raise NWCSSensitivityInputBuildError(
                f"corpus selected file has no state dict: {checkpoint_path}"
            )

        for tensor_entry in file_entry.get("tensors", []):
            if not isinstance(tensor_entry, Mapping) or not tensor_entry.get("selected"):
                continue
            name = str(tensor_entry["name"])
            if name not in state_dict:
                raise NWCSSensitivityInputBuildError(
                    f"corpus selected tensor {name!r} missing in {checkpoint_path}"
                )
            value = state_dict[name]
            reason = _classify_tensor(name, value, block_size)
            if reason is not None:
                raise NWCSSensitivityInputBuildError(
                    f"corpus selected tensor {name!r} is no longer selectable: {reason}"
                )
            shape = [int(s) for s in value.shape]
            if shape != list(tensor_entry.get("shape", [])):
                raise NWCSSensitivityInputBuildError(
                    f"corpus manifest shape mismatch for {checkpoint_path}:{name}: "
                    f"manifest={tensor_entry.get('shape')} actual={shape}"
                )
            if str(value.dtype) != str(tensor_entry.get("dtype")):
                raise NWCSSensitivityInputBuildError(
                    f"corpus manifest dtype mismatch for {checkpoint_path}:{name}: "
                    f"manifest={tensor_entry.get('dtype')} actual={value.dtype}"
                )
            block_count = int(value.numel() // block_size)
            if block_count != int(tensor_entry.get("block_count", -1)):
                raise NWCSSensitivityInputBuildError(
                    f"corpus manifest block_count mismatch for {checkpoint_path}:{name}: "
                    f"manifest={tensor_entry.get('block_count')} actual={block_count}"
                )
            anchor_meta = anchor_parameter_metadata.get(name)
            if anchor_meta is None or name not in anchor_sensitivities:
                raise NWCSSensitivityInputBuildError(
                    f"corpus selected tensor {name!r} has no anchor sensitivity"
                )
            if list(anchor_meta.get("shape", [])) != shape:
                raise NWCSSensitivityInputBuildError(
                    f"corpus selected tensor {name!r} shape {shape} does not "
                    f"match anchor sensitivity shape {anchor_meta.get('shape')}"
                )

            used = int(tensor_entry["used_block_count"])
            if used <= 0 or used > block_count:
                raise NWCSSensitivityInputBuildError(
                    f"corpus manifest used_block_count invalid for "
                    f"{checkpoint_path}:{name}: {used}"
                )
            start = int(tensor_entry["corpus_block_start"])
            end = int(tensor_entry["corpus_block_end"])
            if start != expected_next or end != start + used:
                raise NWCSSensitivityInputBuildError(
                    f"corpus manifest block ordering gap at {checkpoint_path}:{name}: "
                    f"start={start} end={end} expected_start={expected_next}"
                )
            source = anchor_sensitivities[name].detach().cpu().float().reshape(-1)
            if source.numel() < used:
                raise NWCSSensitivityInputBuildError(
                    f"anchor sensitivity for {name!r} has {source.numel()} blocks, "
                    f"corpus manifest needs {used}"
                )
            selected = _validate_block_vector(
                source[:used],
                name=f"corpus:{name}",
                expected_len=used,
            )
            out_parts.append(selected)
            tensor_records.append(
                {
                    "checkpoint": str(checkpoint_path),
                    "tensor": name,
                    "shape": shape,
                    "used_block_count": used,
                    "corpus_block_start": start,
                    "corpus_block_end": end,
                    "anchor_source_blocks": int(source.numel()),
                }
            )
            expected_next = end

    expected_total = int(manifest["totals"]["selected_blocks"])
    if expected_next != expected_total:
        raise NWCSSensitivityInputBuildError(
            f"corpus sensitivity block mismatch: built={expected_next} "
            f"manifest={expected_total}"
        )
    if not out_parts:
        raise NWCSSensitivityInputBuildError("corpus manifest produced no selected tensors")
    out = torch.cat(out_parts).float()
    _validate_block_vector(out, name="corpus sensitivities", expected_len=expected_total)
    if out.numel() > 1 and float(out.max().item()) == float(out.min().item()):
        raise NWCSSensitivityInputBuildError(
            "corpus sensitivities are uniform; uniform/fake sensitivity is non-promotable"
        )
    return out, tensor_records, manifest


def build_inputs(args: argparse.Namespace) -> dict[str, Any]:
    component_manifest_path = Path(args.component_sensitivity_manifest)
    anchor_renderer = Path(args.anchor_renderer)
    anchor_archive = Path(args.anchor_archive)
    corpus_manifest_path = Path(args.corpus_manifest)
    block_size = int(args.block_size)
    if block_size <= 0:
        raise NWCSSensitivityInputBuildError("block_size must be positive")

    component_manifest = _load_component_manifest(component_manifest_path)
    _assert_component_manifest_matches_inputs(
        component_manifest,
        component_manifest_path=component_manifest_path,
        anchor_renderer=anchor_renderer,
        anchor_archive=anchor_archive,
    )
    component_manifest_sha = _sha256_file(component_manifest_path)
    anchor_renderer_sha = _sha256_file(anchor_renderer)
    anchor_archive_sha = _sha256_file(anchor_archive)
    corpus_manifest_sha = _sha256_file(corpus_manifest_path)

    component_map, map_info = _load_combined_component_map(
        component_manifest,
        component_manifest_path=component_manifest_path,
    )
    model = load_any_renderer_checkpoint(anchor_renderer, device="cpu")
    model.eval()
    anchor_sensitivities, parameter_metadata = _build_anchor_sensitivities(
        model,
        component_map,
        map_info=map_info,
        block_size=block_size,
    )

    corpus_sensitivities, corpus_records, corpus_manifest = _build_corpus_sensitivities(
        corpus_manifest_path,
        anchor_sensitivities=anchor_sensitivities,
        anchor_parameter_metadata=parameter_metadata,
        replay_root=args.corpus_replay_root,
        block_size=block_size,
    )

    anchor_payload = {
        "format": ANCHOR_OUTPUT_FORMAT,
        "sensitivities": dict(anchor_sensitivities),
        "metadata": {
            "schema_version": 1,
            "source": "component_sensitivity_v1.combined",
            "promotion_eligible": True,
            "component_sensitivity_manifest": str(component_manifest_path),
            "component_sensitivity_manifest_sha256": component_manifest_sha,
            "combined_component_map": map_info,
            "anchor_archive": str(anchor_archive),
            "anchor_archive_sha256": anchor_archive_sha,
            "anchor_renderer": str(anchor_renderer),
            "anchor_renderer_sha256": anchor_renderer_sha,
            "block_size": block_size,
            "parameters": parameter_metadata,
        },
    }
    corpus_payload = {
        "format": CORPUS_OUTPUT_FORMAT,
        "sensitivities": corpus_sensitivities,
        "metadata": {
            "schema_version": 1,
            "source": "anchor_parameter_sensitivity_projected_to_corpus_manifest",
            "promotion_eligible": True,
            "component_sensitivity_manifest": str(component_manifest_path),
            "component_sensitivity_manifest_sha256": component_manifest_sha,
            "corpus_manifest": str(corpus_manifest_path),
            "corpus_manifest_sha256": corpus_manifest_sha,
            "anchor_renderer_sha256": anchor_renderer_sha,
            "anchor_archive_sha256": anchor_archive_sha,
            "block_size": block_size,
            "num_blocks": int(corpus_sensitivities.numel()),
            "selected_tensor_count": len(corpus_records),
            "selected_tensors": corpus_records,
        },
    }

    anchor_output = Path(args.anchor_output)
    corpus_output = Path(args.corpus_output)
    anchor_output.parent.mkdir(parents=True, exist_ok=True)
    corpus_output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(anchor_payload, anchor_output)
    torch.save(corpus_payload, corpus_output)

    summary = {
        "anchor_output": str(anchor_output),
        "anchor_parameter_count": len(anchor_sensitivities),
        "anchor_block_count": int(
            sum(int(t.numel()) for t in anchor_sensitivities.values())
        ),
        "anchor_renderer_sha256": anchor_renderer_sha,
        "anchor_archive_sha256": anchor_archive_sha,
        "component_sensitivity_manifest_sha256": component_manifest_sha,
        "corpus_output": str(corpus_output),
        "corpus_manifest_sha256": corpus_manifest_sha,
        "corpus_block_count": int(corpus_sensitivities.numel()),
        "corpus_manifest_selected_blocks": int(corpus_manifest["totals"]["selected_blocks"]),
        "promotion_eligible": True,
    }
    if args.summary_output is not None:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return summary


def _existing_path(value: str) -> Path:
    path = Path(value)
    if not path.exists():
        raise argparse.ArgumentTypeError(f"not found: {value}")
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--component-sensitivity-manifest",
        type=_existing_path,
        required=True,
        help="Validated promotion-grade component_sensitivity_v1 JSON.",
    )
    parser.add_argument(
        "--anchor-renderer",
        type=_existing_path,
        required=True,
        help="Anchor renderer checkpoint/bin used by the component sensitivity manifest.",
    )
    parser.add_argument(
        "--anchor-archive",
        type=_existing_path,
        required=True,
        help="Anchor archive.zip whose SHA must match the component manifest contest archive.",
    )
    parser.add_argument(
        "--corpus-manifest",
        type=_existing_path,
        required=True,
        help="Deterministic tac.neural_weight_corpus.v1 manifest.",
    )
    parser.add_argument("--corpus-replay-root", type=Path, default=None)
    parser.add_argument("--anchor-output", type=Path, required=True)
    parser.add_argument("--corpus-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, default=None)
    parser.add_argument("--block-size", type=int, default=16)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        summary = build_inputs(args)
    except (CorpusManifestError, ComponentSensitivityArtifactError, NWCSSensitivityInputBuildError) as exc:
        raise SystemExit(f"FATAL: {exc}") from exc
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
