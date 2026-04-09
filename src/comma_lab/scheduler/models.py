from __future__ import annotations

from dataclasses import asdict, dataclass, field


class SchedulerValidationError(ValueError):
    """Raised when scheduler inputs are malformed."""


def _require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SchedulerValidationError(f"{field_name} must be a non-empty string")
    return value.strip()


def _optional_text(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_text(value, field_name)


def _non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SchedulerValidationError(f"{field_name} must be a non-negative integer")
    return value


def _optional_non_negative_int(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    return _non_negative_int(value, field_name)


def _optional_number(value: object, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SchedulerValidationError(f"{field_name} must be numeric")
    return float(value)


def _string_tuple(values: object, field_name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if not isinstance(values, list):
        raise SchedulerValidationError(f"{field_name} must be a list of strings")
    items: list[str] = []
    for index, value in enumerate(values):
        items.append(_require_text(value, f"{field_name}[{index}]"))
    return tuple(items)


def _string_dict(values: object, field_name: str) -> dict[str, object]:
    if values is None:
        return {}
    if not isinstance(values, dict):
        raise SchedulerValidationError(f"{field_name} must be an object")
    normalized: dict[str, object] = {}
    for key, value in values.items():
        normalized[_require_text(key, f"{field_name} key")] = value
    return normalized


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    track: str
    platform: str | None = None
    packaging_view: str | None = None
    config: dict[str, object] = field(default_factory=dict)
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_text(self.name, "experiment.name"))
        object.__setattr__(self, "track", _require_text(self.track, "experiment.track"))
        object.__setattr__(self, "platform", _optional_text(self.platform, "experiment.platform"))
        object.__setattr__(
            self,
            "packaging_view",
            _optional_text(self.packaging_view, "experiment.packaging_view"),
        )
        object.__setattr__(self, "config", _string_dict(self.config, "experiment.config"))
        object.__setattr__(self, "tags", _string_tuple(list(self.tags), "experiment.tags"))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BudgetSpec:
    max_runs: int | None = None
    max_active_runs: int | None = None
    max_failed_runs: int | None = None
    max_archive_bytes: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "max_runs", _optional_non_negative_int(self.max_runs, "budget.max_runs"))
        object.__setattr__(
            self,
            "max_active_runs",
            _optional_non_negative_int(self.max_active_runs, "budget.max_active_runs"),
        )
        object.__setattr__(
            self,
            "max_failed_runs",
            _optional_non_negative_int(self.max_failed_runs, "budget.max_failed_runs"),
        )
        object.__setattr__(
            self,
            "max_archive_bytes",
            _optional_non_negative_int(self.max_archive_bytes, "budget.max_archive_bytes"),
        )

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PlatformSpec:
    name: str
    kind: str
    result_devices: tuple[str, ...] = ()
    manifest_globs: tuple[str, ...] = ()
    status_globs: tuple[str, ...] = ()
    ledger_paths: tuple[str, ...] = ()
    budget: BudgetSpec = field(default_factory=BudgetSpec)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_text(self.name, "platform.name"))
        object.__setattr__(self, "kind", _require_text(self.kind, "platform.kind"))
        object.__setattr__(
            self,
            "result_devices",
            _string_tuple(list(self.result_devices), "platform.result_devices"),
        )
        object.__setattr__(
            self,
            "manifest_globs",
            _string_tuple(list(self.manifest_globs), "platform.manifest_globs"),
        )
        object.__setattr__(
            self,
            "status_globs",
            _string_tuple(list(self.status_globs), "platform.status_globs"),
        )
        object.__setattr__(
            self,
            "ledger_paths",
            _string_tuple(list(self.ledger_paths), "platform.ledger_paths"),
        )
        if not isinstance(self.budget, BudgetSpec):
            raise SchedulerValidationError("platform.budget must be a BudgetSpec")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "result_devices": list(self.result_devices),
            "manifest_globs": list(self.manifest_globs),
            "status_globs": list(self.status_globs),
            "ledger_paths": list(self.ledger_paths),
            "budget": self.budget.to_dict(),
        }


@dataclass(frozen=True)
class PlatformRegistry:
    version: int
    platforms: dict[str, PlatformSpec]

    def __post_init__(self) -> None:
        object.__setattr__(self, "version", _non_negative_int(self.version, "registry.version"))
        if not isinstance(self.platforms, dict):
            raise SchedulerValidationError("registry.platforms must be a dict")
        for name, platform in self.platforms.items():
            if name != platform.name:
                raise SchedulerValidationError(
                    f"registry.platforms key mismatch: expected {platform.name!r}, got {name!r}"
                )

    def platform_for_device(self, device: str | None) -> str | None:
        if not device:
            return None
        for platform in self.platforms.values():
            if device in platform.result_devices:
                return platform.name
        return None

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "platforms": [self.platforms[name].to_dict() for name in sorted(self.platforms)],
        }


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    source: str
    platform: str
    status: str
    experiment: str | None = None
    track: str | None = None
    device: str | None = None
    packaging_view: str | None = None
    score: float | None = None
    archive_bytes: int | None = None
    started_at: str | None = None
    finished_at: str | None = None
    path: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", _require_text(self.run_id, "run.run_id"))
        object.__setattr__(self, "source", _require_text(self.source, "run.source"))
        object.__setattr__(self, "platform", _require_text(self.platform, "run.platform"))
        object.__setattr__(self, "status", _require_text(self.status, "run.status"))
        object.__setattr__(self, "experiment", _optional_text(self.experiment, "run.experiment"))
        object.__setattr__(self, "track", _optional_text(self.track, "run.track"))
        object.__setattr__(self, "device", _optional_text(self.device, "run.device"))
        object.__setattr__(self, "packaging_view", _optional_text(self.packaging_view, "run.packaging_view"))
        object.__setattr__(self, "score", _optional_number(self.score, "run.score"))
        object.__setattr__(
            self,
            "archive_bytes",
            _optional_non_negative_int(self.archive_bytes, "run.archive_bytes"),
        )
        object.__setattr__(self, "started_at", _optional_text(self.started_at, "run.started_at"))
        object.__setattr__(self, "finished_at", _optional_text(self.finished_at, "run.finished_at"))
        object.__setattr__(self, "path", _optional_text(self.path, "run.path"))
        object.__setattr__(self, "metadata", _string_dict(self.metadata, "run.metadata"))

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BudgetUsage:
    total_runs: int = 0
    active_runs: int = 0
    failed_runs: int = 0
    archive_bytes: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "total_runs", _non_negative_int(self.total_runs, "usage.total_runs"))
        object.__setattr__(self, "active_runs", _non_negative_int(self.active_runs, "usage.active_runs"))
        object.__setattr__(self, "failed_runs", _non_negative_int(self.failed_runs, "usage.failed_runs"))
        object.__setattr__(self, "archive_bytes", _non_negative_int(self.archive_bytes, "usage.archive_bytes"))

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class PlatformBudgetReport:
    platform: PlatformSpec
    usage: BudgetUsage
    over_budget: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.platform.name,
            "kind": self.platform.kind,
            "budget": self.platform.budget.to_dict(),
            "usage": self.usage.to_dict(),
            "over_budget": self.over_budget,
        }


@dataclass(frozen=True)
class BudgetReport:
    platforms: dict[str, PlatformBudgetReport]

    def to_dict(self) -> dict[str, object]:
        return {
            "platforms": [self.platforms[name].to_dict() for name in sorted(self.platforms)],
        }


@dataclass(frozen=True)
class TrackStatus:
    track: str
    latest_result: RunRecord

    def to_dict(self) -> dict[str, object]:
        return {
            "track": self.track,
            "latest_result": self.latest_result.to_dict(),
        }


@dataclass(frozen=True)
class SchedulerStatusReport:
    repo_root: str
    result_count: int
    run_record_count: int
    tracks: tuple[TrackStatus, ...]
    active_runs: tuple[RunRecord, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "repo_root": self.repo_root,
            "result_count": self.result_count,
            "run_record_count": self.run_record_count,
            "tracks": [track.to_dict() for track in self.tracks],
            "active_runs": [run.to_dict() for run in self.active_runs],
        }
