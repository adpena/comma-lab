# SPDX-License-Identifier: MIT
"""Generated HNeRV architecture schemas for architecture-shrink lanes.

The public PR100-PR106 HNeRV family hardcodes a 28-tensor state schema for
``latent_dim=28`` and ``base_channels=36``. Architecture-shrink work needs the
same schema contract generated from an explicit architecture config so training
drivers can produce checkpoints for smaller widths without hand-editing tensor
shapes.

This module is pure schema/state-dict plumbing. It does not load scorers, does
not claim scores, and does not imply a generated checkpoint is valid for exact
eval until a matching runtime loader, archive packet, and CUDA auth eval exist.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import torch

DEFAULT_CHANNEL_MULTIPLIERS: tuple[float, ...] = (
    1.0,
    1.0,
    1.0,
    0.75,
    0.58,
    0.5,
    0.5,
)
DEFAULT_BASE_GRID: tuple[int, int] = (6, 8)
DEFAULT_EVAL_SIZE: tuple[int, int] = (384, 512)


Schema = tuple[tuple[str, tuple[int, ...]], ...]


class HNeRVArchSchemaError(ValueError):
    """Raised when a generated HNeRV schema/config is malformed."""


@dataclass(frozen=True)
class HNeRVArchConfig:
    """Small HNeRV decoder architecture contract.

    The formulas mirror the public HNeRVDecoder implementation:

    ``channels = [C, C, C, int(.75C), int(.58C), int(.5C), int(.5C)]``

    Args:
        latent_dim: per-frame-pair latent dimension.
        base_channels: HNeRV base width ``C``.
        eval_size: decoded pair size. Kept in the manifest even though it does
            not change state tensor shapes.
        base_grid: stem grid before the six PixelShuffle x2 stages.
        channel_multipliers: seven taper multipliers, one per stage channel.
    """

    latent_dim: int = 28
    base_channels: int = 36
    eval_size: tuple[int, int] = DEFAULT_EVAL_SIZE
    base_grid: tuple[int, int] = DEFAULT_BASE_GRID
    channel_multipliers: tuple[float, ...] = DEFAULT_CHANNEL_MULTIPLIERS

    def __post_init__(self) -> None:
        if self.latent_dim <= 0:
            raise HNeRVArchSchemaError(f"latent_dim must be > 0, got {self.latent_dim}")
        if self.base_channels <= 0:
            raise HNeRVArchSchemaError(
                f"base_channels must be > 0, got {self.base_channels}"
            )
        if len(self.eval_size) != 2 or any(int(v) <= 0 for v in self.eval_size):
            raise HNeRVArchSchemaError(f"bad eval_size: {self.eval_size!r}")
        if len(self.base_grid) != 2 or any(int(v) <= 0 for v in self.base_grid):
            raise HNeRVArchSchemaError(f"bad base_grid: {self.base_grid!r}")
        if len(self.channel_multipliers) != 7:
            raise HNeRVArchSchemaError(
                "channel_multipliers must have exactly 7 entries"
            )
        if any(float(v) <= 0.0 for v in self.channel_multipliers):
            raise HNeRVArchSchemaError(
                f"channel_multipliers must be positive: {self.channel_multipliers!r}"
            )

        channels = self.channels
        if min(channels) <= 0:
            raise HNeRVArchSchemaError(
                f"generated channel taper contains nonpositive channels: {channels!r}"
            )
        if channels[-1] // 2 <= 0:
            raise HNeRVArchSchemaError(
                f"final channel count {channels[-1]} too small for refine.0"
            )

    @property
    def channels(self) -> tuple[int, ...]:
        return tuple(
            int(self.base_channels * multiplier)
            for multiplier in self.channel_multipliers
        )

    @property
    def stem_features(self) -> int:
        return self.channels[0] * self.base_grid[0] * self.base_grid[1]

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "latent_dim": self.latent_dim,
            "base_channels": self.base_channels,
            "eval_size": list(self.eval_size),
            "base_grid": list(self.base_grid),
            "channel_multipliers": list(self.channel_multipliers),
            "channels": list(self.channels),
            "schema_fingerprint": schema_fingerprint(generate_hnerv_state_schema(self)),
            "n_state_tensors": len(generate_hnerv_state_schema(self)),
            "n_state_elements": schema_numel(generate_hnerv_state_schema(self)),
        }


def generate_hnerv_state_schema(config: HNeRVArchConfig) -> Schema:
    """Generate the HNeRVDecoder state schema for ``config``."""

    c = config.channels
    stem_features = config.stem_features
    rows: list[tuple[str, tuple[int, ...]]] = [
        ("stem.weight", (stem_features, config.latent_dim)),
        ("stem.bias", (stem_features,)),
    ]
    for idx in range(6):
        in_ch = c[idx]
        out_ch = c[idx + 1]
        rows.append((f"blocks.{idx}.weight", (out_ch * 4, in_ch, 3, 3)))
        rows.append((f"blocks.{idx}.bias", (out_ch * 4,)))

    for idx in range(6):
        in_ch = c[idx]
        out_ch = c[idx + 1]
        if in_ch != out_ch:
            rows.append((f"skips.{idx}.weight", (out_ch, in_ch, 1, 1)))
            rows.append((f"skips.{idx}.bias", (out_ch,)))

    final_ch = c[-1]
    refine_mid = final_ch // 2
    rows.extend(
        [
            ("refine.0.weight", (refine_mid, final_ch, 3, 3)),
            ("refine.0.bias", (refine_mid,)),
            ("refine.1.weight", (final_ch, refine_mid, 3, 3)),
            ("refine.1.bias", (final_ch,)),
            ("rgb_0.weight", (3, final_ch, 3, 3)),
            ("rgb_0.bias", (3,)),
            ("rgb_1.weight", (3, final_ch, 3, 3)),
            ("rgb_1.bias", (3,)),
        ]
    )
    return tuple(rows)


def schema_to_jsonable(schema: Schema) -> list[dict[str, Any]]:
    return [{"name": name, "shape": list(shape)} for name, shape in schema]


def schema_numel(schema: Schema) -> int:
    return int(sum(math.prod(shape) for _name, shape in schema))


def schema_fingerprint(schema: Schema) -> str:
    payload = json.dumps(schema_to_jsonable(schema), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def select_base_channels_for_element_retention(
    *,
    element_retention: float,
    latent_dim: int = 28,
    baseline_base_channels: int = 36,
    eval_size: tuple[int, int] = DEFAULT_EVAL_SIZE,
    floor: bool = True,
) -> HNeRVArchConfig:
    """Select a base width whose generated schema matches an element budget.

    ``floor=True`` chooses the closest width whose element count is not larger
    than the target. If every candidate exceeds the target, the smallest legal
    width is returned. This is a planning rule, not an optimality proof.
    """

    if not (0.0 < element_retention <= 1.0):
        raise HNeRVArchSchemaError(
            f"element_retention must be in (0, 1], got {element_retention}"
        )
    baseline = HNeRVArchConfig(
        latent_dim=latent_dim,
        base_channels=baseline_base_channels,
        eval_size=eval_size,
    )
    target = schema_numel(generate_hnerv_state_schema(baseline)) * element_retention
    candidates: list[tuple[float, int, HNeRVArchConfig]] = []
    for base_channels in range(2, baseline_base_channels + 1):
        try:
            cfg = HNeRVArchConfig(
                latent_dim=latent_dim,
                base_channels=base_channels,
                eval_size=eval_size,
            )
        except HNeRVArchSchemaError:
            continue
        numel = schema_numel(generate_hnerv_state_schema(cfg))
        if floor and numel > target:
            continue
        candidates.append((abs(numel - target), numel, cfg))
    if not candidates:
        for base_channels in range(2, baseline_base_channels + 1):
            try:
                cfg = HNeRVArchConfig(
                    latent_dim=latent_dim,
                    base_channels=base_channels,
                    eval_size=eval_size,
                )
            except HNeRVArchSchemaError:
                continue
            numel = schema_numel(generate_hnerv_state_schema(cfg))
            candidates.append((abs(numel - target), numel, cfg))
    if not candidates:
        raise HNeRVArchSchemaError("no legal base_channels candidates")
    # Tie-break toward smaller width to keep rate-side planning conservative.
    return sorted(candidates, key=lambda row: (row[0], row[1], row[2].base_channels))[0][2]


def initialize_state_dict_by_overlap(
    source_state_dict: dict[str, torch.Tensor],
    *,
    target_schema: Schema,
) -> dict[str, torch.Tensor]:
    """Create a target-shape state_dict by copying overlapping slices.

    Missing source tensors are initialized to zeros. Extra source tensors are
    ignored. This provides a deterministic bootstrap checkpoint for training
    generated schemas; it is not a claim that the checkpoint is performant.
    """

    out: dict[str, torch.Tensor] = {}
    for name, shape in target_schema:
        source = source_state_dict.get(name)
        dtype = source.dtype if isinstance(source, torch.Tensor) else torch.float32
        target = torch.zeros(*shape, dtype=dtype)
        if isinstance(source, torch.Tensor):
            slices = tuple(
                slice(0, min(int(a), int(b)))
                for a, b in zip(shape, source.shape, strict=False)
            )
            if len(slices) == len(shape) == source.ndim:
                target[slices] = source[slices].detach().cpu().to(dtype)
        out[name] = target
    return out


def state_dict_schema_rows(state_dict: dict[str, torch.Tensor]) -> Schema:
    rows: list[tuple[str, tuple[int, ...]]] = []
    for name in sorted(state_dict):
        tensor = state_dict[name]
        if not isinstance(tensor, torch.Tensor):
            raise HNeRVArchSchemaError(f"state_dict entry {name!r} is not a tensor")
        rows.append((name, tuple(int(v) for v in tensor.shape)))
    return tuple(rows)


def compare_schema_shapes(expected: Schema, actual: Schema) -> list[str]:
    """Return human-readable schema mismatches."""

    findings: list[str] = []
    actual_by_name = dict(actual)
    expected_names = {name for name, _shape in expected}
    for name, shape in expected:
        if name not in actual_by_name:
            findings.append(f"missing:{name}")
        elif tuple(actual_by_name[name]) != tuple(shape):
            findings.append(f"shape:{name}:{actual_by_name[name]}!={shape}")
    for name, _shape in actual:
        if name not in expected_names:
            findings.append(f"extra:{name}")
    return findings


def config_from_mapping(raw: dict[str, Any]) -> HNeRVArchConfig:
    allowed = {field.name for field in dataclasses.fields(HNeRVArchConfig)}
    extra = sorted(set(raw) - allowed)
    if extra:
        raise HNeRVArchSchemaError(f"unknown HNeRVArchConfig keys: {extra}")
    data = dict(raw)
    if "eval_size" in data:
        data["eval_size"] = tuple(data["eval_size"])
    if "base_grid" in data:
        data["base_grid"] = tuple(data["base_grid"])
    if "channel_multipliers" in data:
        data["channel_multipliers"] = tuple(data["channel_multipliers"])
    return HNeRVArchConfig(**data)
