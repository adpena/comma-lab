from __future__ import annotations

import json
from pathlib import Path

from .models import BudgetSpec, PlatformRegistry, PlatformSpec, SchedulerValidationError


def _load_registry_payload(path: Path) -> dict[str, object]:
    raw = path.read_text()
    suffix = path.suffix.lower()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        if suffix in {".yaml", ".yml"}:
            raise SchedulerValidationError(
                f"YAML support is limited to JSON-compatible YAML without extra dependencies: {path}"
            ) from exc
        raise SchedulerValidationError(f"Invalid JSON registry: {path}: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise SchedulerValidationError(f"Registry root must be an object: {path}")
    return payload


def load_platform_registry(path: str | Path) -> PlatformRegistry:
    registry_path = Path(path)
    if not registry_path.exists():
        raise FileNotFoundError(f"Platform registry not found: {registry_path}")

    payload = _load_registry_payload(registry_path)
    version = payload.get("version")
    platforms_payload = payload.get("platforms")
    if not isinstance(platforms_payload, list):
        raise SchedulerValidationError(f"registry.platforms must be a list in {registry_path}")

    platforms: dict[str, PlatformSpec] = {}
    for index, item in enumerate(platforms_payload):
        if not isinstance(item, dict):
            raise SchedulerValidationError(f"registry.platforms[{index}] must be an object")
        budget_payload = item.get("budget", item.get("budgets", {}))
        if budget_payload is None:
            budget_payload = {}
        if not isinstance(budget_payload, dict):
            raise SchedulerValidationError(f"platform budget must be an object for entry {index}")
        platform = PlatformSpec(
            name=item.get("name"),
            kind=item.get("kind"),
            result_devices=tuple(item.get("result_devices", [])),
            manifest_globs=tuple(item.get("manifest_globs", [])),
            status_globs=tuple(item.get("status_globs", [])),
            ledger_paths=tuple(item.get("ledger_paths", [])),
            budget=BudgetSpec(
                max_runs=budget_payload.get("max_runs"),
                max_active_runs=budget_payload.get("max_active_runs"),
                max_failed_runs=budget_payload.get("max_failed_runs"),
                max_archive_bytes=budget_payload.get("max_archive_bytes"),
            ),
        )
        if platform.name in platforms:
            raise SchedulerValidationError(f"Duplicate platform name in registry: {platform.name}")
        platforms[platform.name] = platform

    return PlatformRegistry(version=version, platforms=platforms)
