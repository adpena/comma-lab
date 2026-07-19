"""Typed, fail-closed configuration for the local PDW1 palette tournament.

This is deliberately a small standalone DSL.  It is a research-only receipt
configuration, not an archive configuration or a score authority.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import MISSING, asdict, dataclass
from pathlib import Path
from typing import Any, Literal

FAMILY_NAMES = frozenset(
    {
        "baseline",
        "coordinate",
        "coordinate_warm_start",
        "dspsa",
        "global",
        "label_run_simplify",
        "zero",
    }
)
SCHEMA = "einstein_kolmogorov_crux_config.v1"


class EinsteinKolmogorovConfigError(ValueError):
    """Raised when a probe configuration is not an explicit safe contract."""


def _absolute_path(value: str, name: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        raise EinsteinKolmogorovConfigError(f"{name} must be an absolute explicit path")
    return str(path)


def _sha256(value: str, name: str) -> str:
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise EinsteinKolmogorovConfigError(f"{name} must be a lowercase SHA-256 hex digest")
    return value


@dataclass(frozen=True)
class EinsteinKolmogorovCruxConfig:
    """All inputs required for a deterministic, receiver-closed local probe."""

    packet_path: str
    packet_sha256: str
    gt_path: str
    gt_sha256: str
    segnet_path: str
    segnet_sha256: str
    upstream_path: str
    output_dir: str
    pair_indices: tuple[int, ...]
    family: Literal[
        "baseline",
        "zero",
        "global",
        "coordinate",
        "coordinate_warm_start",
        "dspsa",
        "label_run_simplify",
    ]
    seed: int
    iterations: int = 0
    label_min_run: int = 0
    target_first_displacement: float = 1.0
    gain_alpha: float = 0.602
    lower_bound: int = 0
    upper_bound: int = 255
    checkpoint_every_pairs: int = 1
    scorer_authority: str = "singleton_cpu_torch_segnet"
    scorer_height: int = 384
    scorer_width: int = 512
    camera_height: int = 874
    camera_width: int = 1164
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        if self.schema != SCHEMA:
            raise EinsteinKolmogorovConfigError(f"unsupported schema: {self.schema}")
        object.__setattr__(self, "packet_path", _absolute_path(self.packet_path, "packet_path"))
        object.__setattr__(self, "gt_path", _absolute_path(self.gt_path, "gt_path"))
        object.__setattr__(self, "segnet_path", _absolute_path(self.segnet_path, "segnet_path"))
        object.__setattr__(self, "upstream_path", _absolute_path(self.upstream_path, "upstream_path"))
        object.__setattr__(self, "output_dir", _absolute_path(self.output_dir, "output_dir"))
        for field in ("packet_sha256", "gt_sha256", "segnet_sha256"):
            object.__setattr__(self, field, _sha256(str(getattr(self, field)), field))
        if Path(self.segnet_path) != Path(self.upstream_path) / "models" / "segnet.safetensors":
            raise EinsteinKolmogorovConfigError("segnet_path must be upstream_path/models/segnet.safetensors")
        if self.family not in FAMILY_NAMES:
            raise EinsteinKolmogorovConfigError(f"unknown family: {self.family}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise EinsteinKolmogorovConfigError("seed must be a non-negative deterministic integer")
        if not self.pair_indices or tuple(sorted(set(self.pair_indices))) != self.pair_indices:
            raise EinsteinKolmogorovConfigError("pair_indices must be a non-empty sorted unique tuple")
        if any(isinstance(pair, bool) or not isinstance(pair, int) or pair < 0 for pair in self.pair_indices):
            raise EinsteinKolmogorovConfigError("pair_indices must contain non-negative integers")
        if self.iterations < 0 or self.checkpoint_every_pairs != 1:
            raise EinsteinKolmogorovConfigError("iterations must be >=0 and checkpoint_every_pairs must be exactly 1")
        if self.family in {"coordinate", "coordinate_warm_start", "dspsa"} and self.iterations == 0:
            raise EinsteinKolmogorovConfigError("coordinate search families and dspsa require iterations > 0")
        if self.family == "label_run_simplify":
            if self.iterations != 0 or not 2 <= self.label_min_run <= 64:
                raise EinsteinKolmogorovConfigError(
                    "label_run_simplify requires iterations=0 and label_min_run in [2,64]"
                )
        elif self.label_min_run != 0:
            raise EinsteinKolmogorovConfigError("label_min_run is active only for label_run_simplify")
        if not (0 <= self.lower_bound < self.upper_bound <= 255):
            raise EinsteinKolmogorovConfigError("bounds must be uint8 and lower_bound < upper_bound")
        if self.target_first_displacement != 1.0 or not (0.5 < self.gain_alpha <= 1.0):
            raise EinsteinKolmogorovConfigError(
                "target_first_displacement is frozen at exactly 1.0 and gain_alpha must be in (0.5, 1]"
            )
        if self.scorer_authority != "singleton_cpu_torch_segnet":
            raise EinsteinKolmogorovConfigError("only singleton_cpu_torch_segnet is an authority surface")
        if (self.scorer_height, self.scorer_width, self.camera_height, self.camera_width) != (384, 512, 874, 1164):
            raise EinsteinKolmogorovConfigError("geometry must be frozen 384x512 scorer / 874x1164 camera")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["pair_indices"] = list(self.pair_indices)
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.to_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EinsteinKolmogorovCruxConfig:
        expected = set(cls.__dataclass_fields__)
        unknown = set(payload) - expected
        missing = {
            name
            for name, field in cls.__dataclass_fields__.items()
            if name not in payload and field.default is MISSING and field.default_factory is MISSING
        }
        if unknown or missing:
            raise EinsteinKolmogorovConfigError(
                f"config keys differ (unknown={sorted(unknown)}, missing={sorted(missing)})"
            )
        values = dict(payload)
        values["pair_indices"] = tuple(values["pair_indices"])
        return cls(**values)

    @classmethod
    def from_json(cls, raw: str) -> EinsteinKolmogorovCruxConfig:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise EinsteinKolmogorovConfigError("config is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise EinsteinKolmogorovConfigError("config JSON must be an object")
        return cls.from_dict(payload)

    @classmethod
    def load(cls, path: str | Path) -> EinsteinKolmogorovCruxConfig:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))
