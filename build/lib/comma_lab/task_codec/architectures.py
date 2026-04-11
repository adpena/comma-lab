"""Architecture registry for known post-filter variants and entrypoints."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


_INTEGER_FIELDS = {"hidden", "kernel", "block"}
_FLOAT_FIELDS = {"alpha", "dropout"}


def _coerce_parameter(name: str, value: object) -> object:
    if value is None:
        return None
    if name in _INTEGER_FIELDS:
        return int(value)
    if name in _FLOAT_FIELDS:
        return float(value)
    return value


@dataclass(frozen=True)
class ArchitectureSpec:
    """Serializable architecture description backed by an existing entrypoint."""

    variant: str
    entrypoint: str
    description: str
    default_parameters: dict[str, object] = field(default_factory=dict)
    aliases: tuple[str, ...] = ()

    def normalize_parameters(self, raw: Mapping[str, object] | None = None) -> dict[str, object]:
        normalized = {
            key: _coerce_parameter(key, value)
            for key, value in self.default_parameters.items()
        }
        normalized["variant"] = self.variant
        if raw is None:
            return normalized
        for key, value in raw.items():
            if key == "variant":
                continue
            normalized[key] = _coerce_parameter(key, value)
        return normalized


@dataclass(frozen=True)
class ArchitectureConfig:
    """Resolved architecture config suitable for artifact inspection or handoff."""

    variant: str
    entrypoint: str
    parameters: dict[str, object]
    description: str


class ArchitectureRegistry:
    """Registry of known post-filter architecture variants."""

    def __init__(self) -> None:
        self._specs: dict[str, ArchitectureSpec] = {}
        self._aliases: dict[str, str] = {}

    def register(self, spec: ArchitectureSpec) -> ArchitectureSpec:
        if spec.variant in self._specs:
            raise ValueError(f"Architecture already registered: {spec.variant}")
        self._specs[spec.variant] = spec
        for alias in spec.aliases:
            if alias in self._aliases:
                raise ValueError(f"Architecture alias already registered: {alias}")
            self._aliases[alias] = spec.variant
        return spec

    def get(self, variant: str) -> ArchitectureSpec:
        key = self._aliases.get(variant, variant)
        try:
            return self._specs[key]
        except KeyError as exc:
            raise KeyError(f"Unknown architecture variant: {variant}") from exc

    def variants(self) -> tuple[str, ...]:
        return tuple(sorted(self._specs))

    def resolve_config(self, payload: Mapping[str, object]) -> ArchitectureConfig:
        if "variant" not in payload:
            raise KeyError("Architecture payload must include 'variant'")
        spec = self.get(str(payload["variant"]))
        parameters = spec.normalize_parameters(payload)
        return ArchitectureConfig(
            variant=spec.variant,
            entrypoint=spec.entrypoint,
            parameters=parameters,
            description=spec.description,
        )


def register_default_architectures(registry: ArchitectureRegistry) -> ArchitectureRegistry:
    """Populate a registry with the post-filter variants currently used in-tree."""

    defaults = (
        ArchitectureSpec(
            variant="residual",
            entrypoint="experiments/train_postfilter_canonical.py",
            description="Baseline residual RGB CNN post-filter.",
            default_parameters={"hidden": 16, "kernel": 3},
            aliases=("canonical",),
        ),
        ArchitectureSpec(
            variant="depthwise",
            entrypoint="experiments/train_postfilter_canonical.py",
            description="Depthwise-separable canonical post-filter.",
            default_parameters={"hidden": 16, "kernel": 3},
        ),
        ArchitectureSpec(
            variant="luma",
            entrypoint="experiments/train_postfilter_canonical.py",
            description="Luminance-only residual post-filter.",
            default_parameters={"hidden": 16, "kernel": 3},
        ),
        ArchitectureSpec(
            variant="saliency_weighted",
            entrypoint="experiments/train_postfilter_saliency.py",
            description="Residual post-filter trained with saliency-weighted scorer loss.",
            default_parameters={"hidden": 16, "kernel": 3, "alpha": 20.0},
            aliases=("qat_ema",),
        ),
        ArchitectureSpec(
            variant="segaware",
            entrypoint="experiments/train_postfilter_segaware.py",
            description="Residual post-filter with segmentation-aware emphasis.",
            default_parameters={"hidden": 16, "kernel": 3},
        ),
        ArchitectureSpec(
            variant="dilated",
            entrypoint="experiments/train_postfilter_dilated.py",
            description="Residual post-filter with a dilated middle layer.",
            default_parameters={"hidden": 16, "kernel": 3},
        ),
        ArchitectureSpec(
            variant="pixelshuffle",
            entrypoint="experiments/train_postfilter_pixelshuffle_dilated.py",
            description="Half-resolution pixelshuffle/dilated residual post-filter.",
            default_parameters={"hidden": 64, "kernel": 3, "alpha": 20.0},
        ),
        ArchitectureSpec(
            variant="film_conditioned",
            entrypoint="experiments/train_postfilter_film_conditioned.py",
            description="Residual post-filter modulated by lightweight FiLM conditioning.",
            default_parameters={"hidden": 16, "kernel": 3, "alpha": 20.0},
        ),
        ArchitectureSpec(
            variant="pairaware",
            entrypoint="experiments/train_postfilter_pairaware.py",
            description="Pair-aware residual post-filter experiment.",
            default_parameters={"hidden": 16, "kernel": 3, "alpha": 20.0},
        ),
        ArchitectureSpec(
            variant="counterpoint",
            entrypoint="experiments/train_postfilter_counterpoint.py",
            description="Two-voice residual post-filter ensemble.",
            default_parameters={"hidden": 16, "kernel": 3, "alpha": 20.0},
        ),
        ArchitectureSpec(
            variant="dct_midband",
            entrypoint="experiments/train_postfilter_dct.py",
            description="Block-domain post-filter driven by DCT mid-band corrections.",
            default_parameters={"block": 8, "alpha": 20.0},
        ),
    )
    known = set(registry.variants())
    for spec in defaults:
        if spec.variant not in known:
            registry.register(spec)
    return registry
