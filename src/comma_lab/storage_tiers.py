# SPDX-License-Identifier: MIT
"""Tiered experiment-storage placement for local/autopilot runs.

The contract is intentionally substrate-agnostic: callers provide the workload
subdirectory they need, and the planner chooses the first writable tier with
enough free space. Durable score/eval authority stays elsewhere; this module
only decides where bulky rebuildable work should land.
"""

from __future__ import annotations

import os
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA = "comma_lab.experiment_storage_tier_plan.v1"
DEFAULT_WORKLOAD_SUBDIR = "experiments/results"
DEFAULT_RESERVE_FREE_GB = 40.0
DEFAULT_TIERS = (
    ("vertigo", "/Volumes/VertigoDataTier/pact"),
    ("apdatastore", "/Volumes/APDataStore/pact"),
)


class StorageTierError(ValueError):
    """Raised when no storage tier can safely accept a workload."""


@dataclass(frozen=True)
class StorageTierSpec:
    """A candidate root for bulky experiment outputs."""

    name: str
    root: Path
    priority: int
    reserve_free_bytes: int
    allow_create: bool = True
    allow_local_disk: bool = False


@dataclass(frozen=True)
class StorageTierStatus:
    """Machine-readable placement status for one tier."""

    name: str
    root: str
    workload_root: str
    workload_root_exists: bool
    priority: int
    exists: bool
    parent_exists: bool
    writable: bool
    total_bytes: int | None
    free_bytes: int | None
    reserve_free_bytes: int
    requested_bytes: int
    eligible: bool
    blockers: tuple[str, ...] = field(default_factory=tuple)
    allow_local_disk: bool = False

    @property
    def usable_bytes(self) -> int:
        if self.free_bytes is None:
            return 0
        return max(0, int(self.free_bytes) - int(self.reserve_free_bytes))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["usable_bytes"] = self.usable_bytes
        return payload


@dataclass(frozen=True)
class ExperimentStoragePlan:
    """A false-authority plan for placing bulky experiment artifacts."""

    schema: str
    generated_at_utc: str
    workload_subdir: str
    requested_bytes: int
    min_free_bytes: int
    selected_tier: str | None
    selected_root: str | None
    selected_workload_root: str | None
    tiers: tuple[StorageTierStatus, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "generated_at_utc": self.generated_at_utc,
            "workload_subdir": self.workload_subdir,
            "requested_bytes": self.requested_bytes,
            "min_free_bytes": self.min_free_bytes,
            "selected_tier": self.selected_tier,
            "selected_root": self.selected_root,
            "selected_workload_root": self.selected_workload_root,
            "tiers": [tier.to_dict() for tier in self.tiers],
            "score_claim": False,
            "promotion_eligible": False,
            "rank_or_kill_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }


def bytes_from_gib(value: float) -> int:
    if value < 0:
        raise StorageTierError("GiB values must be non-negative")
    return int(float(value) * (1024**3))


def default_storage_tiers(
    *,
    repo_root: Path,
    reserve_free_gb: float = DEFAULT_RESERVE_FREE_GB,
    allow_local_disk: bool = False,
) -> tuple[StorageTierSpec, ...]:
    """Return the standard local spill order for experiment work."""

    reserve = bytes_from_gib(reserve_free_gb)
    specs = [
        StorageTierSpec(
            name=name,
            root=Path(root),
            priority=index,
            reserve_free_bytes=reserve,
            allow_local_disk=False,
        )
        for index, (name, root) in enumerate(DEFAULT_TIERS)
    ]
    specs.append(
        StorageTierSpec(
            name="local",
            root=repo_root,
            priority=len(specs),
            reserve_free_bytes=reserve,
            allow_create=False,
            allow_local_disk=allow_local_disk,
        )
    )
    return tuple(specs)


def parse_storage_tier_specs(
    values: list[str],
    *,
    repo_root: Path,
    reserve_free_gb: float = DEFAULT_RESERVE_FREE_GB,
    allow_local_disk: bool = False,
) -> tuple[StorageTierSpec, ...]:
    """Parse ``name=/path`` overrides, or return the default waterfall."""

    if not values:
        return default_storage_tiers(
            repo_root=repo_root,
            reserve_free_gb=reserve_free_gb,
            allow_local_disk=allow_local_disk,
        )
    reserve = bytes_from_gib(reserve_free_gb)
    specs: list[StorageTierSpec] = []
    for index, value in enumerate(values):
        if "=" not in value:
            raise StorageTierError(f"storage tier must be name=/path, got {value!r}")
        name, raw_root = value.split("=", 1)
        if not name.strip() or not raw_root.strip():
            raise StorageTierError(f"storage tier must be name=/path, got {value!r}")
        root = Path(raw_root).expanduser()
        if not root.is_absolute():
            root = repo_root / root
        specs.append(
            StorageTierSpec(
                name=name.strip(),
                root=root,
                priority=index,
                reserve_free_bytes=reserve,
                allow_local_disk=allow_local_disk,
            )
        )
    return tuple(specs)


def plan_experiment_storage(
    tiers: tuple[StorageTierSpec, ...],
    *,
    workload_subdir: str = DEFAULT_WORKLOAD_SUBDIR,
    requested_bytes: int = 0,
    min_free_bytes: int = 0,
    create: bool = False,
    probe_writable: bool = True,
) -> ExperimentStoragePlan:
    """Choose the first eligible storage tier and return a durable plan."""

    if requested_bytes < 0 or min_free_bytes < 0:
        raise StorageTierError("requested_bytes and min_free_bytes must be non-negative")
    clean_workload = _clean_relative_subdir(workload_subdir)
    statuses = tuple(
        _status_for_tier(
            spec,
            workload_subdir=clean_workload,
            requested_bytes=requested_bytes,
            min_free_bytes=min_free_bytes,
            create=create,
            probe_writable=probe_writable,
        )
        for spec in sorted(tiers, key=lambda item: item.priority)
    )
    selected = next((tier for tier in statuses if tier.eligible), None)
    return ExperimentStoragePlan(
        schema=SCHEMA,
        generated_at_utc=_utc_stamp(),
        workload_subdir=clean_workload,
        requested_bytes=requested_bytes,
        min_free_bytes=min_free_bytes,
        selected_tier=None if selected is None else selected.name,
        selected_root=None if selected is None else selected.root,
        selected_workload_root=None if selected is None else selected.workload_root,
        tiers=statuses,
    )


def require_selected_storage(plan: ExperimentStoragePlan) -> Path:
    """Return the selected workload root or fail closed with tier blockers."""

    if plan.selected_workload_root is not None:
        return Path(plan.selected_workload_root)
    blockers = []
    for tier in plan.tiers:
        if tier.blockers:
            blockers.append(f"{tier.name}:{','.join(tier.blockers)}")
        else:
            blockers.append(f"{tier.name}:not_eligible")
    raise StorageTierError("no eligible storage tier: " + "; ".join(blockers))


def _utc_stamp() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _clean_relative_subdir(value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        raise StorageTierError("workload_subdir must be relative")
    parts = [part for part in path.parts if part not in {"", "."}]
    if any(part == ".." for part in parts):
        raise StorageTierError("workload_subdir may not contain '..'")
    return Path(*parts).as_posix() if parts else "."


def _status_for_tier(
    spec: StorageTierSpec,
    *,
    workload_subdir: str,
    requested_bytes: int,
    min_free_bytes: int,
    create: bool,
    probe_writable: bool,
) -> StorageTierStatus:
    root = spec.root.expanduser()
    workload_root = root / workload_subdir
    blockers: list[str] = []
    if _looks_like_local_disk(root) and not spec.allow_local_disk:
        blockers.append("local_disk_tier_disabled")
    parent = _nearest_existing_parent(root)
    if parent is None:
        blockers.append("no_existing_parent")
    if create and spec.allow_create:
        try:
            workload_root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            blockers.append(f"mkdir_failed:{type(exc).__name__}")
    exists = root.exists()
    workload_root_exists = workload_root.exists()
    if not workload_root_exists:
        blockers.append("workload_root_missing")
    parent_exists = parent is not None
    usage_path = workload_root if workload_root_exists else parent
    total_bytes: int | None = None
    free_bytes: int | None = None
    if usage_path is not None:
        try:
            usage = shutil.disk_usage(usage_path)
            total_bytes = int(usage.total)
            free_bytes = int(usage.free)
        except OSError as exc:
            blockers.append(f"disk_usage_failed:{type(exc).__name__}")
    writable = False
    if probe_writable and usage_path is not None:
        writable = _probe_write(workload_root if workload_root_exists else usage_path)
        if not writable:
            blockers.append("write_probe_failed")
    elif usage_path is not None:
        writable = os.access(usage_path, os.W_OK)
        if not writable:
            blockers.append("not_writable")
    if free_bytes is None:
        blockers.append("free_bytes_unavailable")
    else:
        required = int(requested_bytes) + int(min_free_bytes) + int(spec.reserve_free_bytes)
        if free_bytes < required:
            blockers.append(f"insufficient_free_bytes:{free_bytes}<{required}")
    eligible = not blockers
    return StorageTierStatus(
        name=spec.name,
        root=str(root),
        workload_root=str(workload_root),
        workload_root_exists=workload_root_exists,
        priority=spec.priority,
        exists=exists,
        parent_exists=parent_exists,
        writable=writable,
        total_bytes=total_bytes,
        free_bytes=free_bytes,
        reserve_free_bytes=spec.reserve_free_bytes,
        requested_bytes=requested_bytes,
        eligible=eligible,
        blockers=tuple(blockers),
        allow_local_disk=spec.allow_local_disk,
    )


def _nearest_existing_parent(path: Path) -> Path | None:
    candidate = path
    while True:
        if candidate.exists():
            return candidate
        if candidate.parent == candidate:
            return None
        candidate = candidate.parent


def _probe_write(path: Path) -> bool:
    probe_root = path if path.exists() and path.is_dir() else _nearest_existing_parent(path)
    if probe_root is None:
        return False
    probe = probe_root / f".storage_tier_probe_{os.getpid()}_{time.time_ns()}"
    try:
        probe.write_text("comma_lab_storage_tier_probe_v1\n", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        try:
            probe.unlink()
        except OSError:
            pass
        return False


def _looks_like_local_disk(path: Path) -> bool:
    try:
        resolved = path.expanduser().resolve(strict=False)
    except OSError:
        resolved = path.expanduser()
    return not str(resolved).startswith("/Volumes/")
