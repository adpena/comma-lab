from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from .models import PlatformRegistry, RunRecord, SchedulerValidationError


RESULT_STATUSES = {"measured"}


def _read_json_file(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise SchedulerValidationError(f"Invalid JSON in {path}: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise SchedulerValidationError(f"Expected object in {path}")
    return payload


def _score_from_result(payload: dict[str, object]) -> float | None:
    for key in ("current_workflow_score", "rule_faithful_score", "score"):
        value = payload.get(key)
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise SchedulerValidationError(f"{key} must be numeric in results.jsonl")
            return float(value)
    return None


def _archive_bytes_from_payload(payload: dict[str, object]) -> int | None:
    for key in ("archive_bytes", "current_workflow_archive_bytes"):
        value = payload.get(key)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SchedulerValidationError(f"{key} must be a non-negative integer")
        return value
    return None


def _run_id_from_payload_or_path(payload: dict[str, object], path: Path, *, kind: str) -> str:
    run_id = payload.get("run_id")
    if isinstance(run_id, str) and run_id.strip():
        return run_id
    slug = payload.get("slug")
    if isinstance(slug, str) and slug.strip():
        return slug.strip()
    if path.stem:
        return path.stem
    raise SchedulerValidationError(f"{kind} missing run_id and slug: {path}")


def _merge_record(existing: RunRecord | None, incoming: RunRecord) -> RunRecord:
    if existing is None:
        return incoming
    return replace(
        existing,
        status=incoming.status or existing.status,
        experiment=incoming.experiment or existing.experiment,
        track=incoming.track or existing.track,
        device=incoming.device or existing.device,
        packaging_view=incoming.packaging_view or existing.packaging_view,
        score=incoming.score if incoming.score is not None else existing.score,
        archive_bytes=incoming.archive_bytes if incoming.archive_bytes is not None else existing.archive_bytes,
        started_at=incoming.started_at or existing.started_at,
        finished_at=incoming.finished_at or existing.finished_at,
        path=incoming.path or existing.path,
        metadata={**existing.metadata, **incoming.metadata},
    )


def _collect_result_records(
    repo_root: Path,
    registry: PlatformRegistry | None,
    records: dict[tuple[str, str], RunRecord],
) -> None:
    results_path = repo_root / "reports" / "results.jsonl"
    if not results_path.exists():
        return

    for line_number, raw_line in enumerate(results_path.read_text().splitlines(), start=1):
        if not raw_line.strip():
            continue
        try:
            payload = json.loads(raw_line)
        except json.JSONDecodeError as exc:
            raise SchedulerValidationError(
                f"Invalid JSON in {results_path}:{line_number}: {exc.msg}"
            ) from exc
        if not isinstance(payload, dict):
            raise SchedulerValidationError(f"Expected object in {results_path}:{line_number}")
        run_id = payload.get("run_id")
        track = payload.get("track")
        if not isinstance(run_id, str) or not run_id.strip():
            raise SchedulerValidationError(f"Missing run_id in {results_path}:{line_number}")
        if not isinstance(track, str) or not track.strip():
            raise SchedulerValidationError(f"Missing track in {results_path}:{line_number}")
        device = payload.get("device")
        if device is not None and (not isinstance(device, str) or not device.strip()):
            raise SchedulerValidationError(f"Invalid device in {results_path}:{line_number}")
        platform = registry.platform_for_device(device) if registry else None
        record = RunRecord(
            run_id=run_id,
            source="results.jsonl",
            platform=platform or device or "results",
            status="measured",
            experiment=run_id,
            track=track,
            device=device,
            packaging_view=payload.get("packaging_view"),
            score=_score_from_result(payload),
            archive_bytes=_archive_bytes_from_payload(payload),
            finished_at=payload.get("ts_utc"),
            path=str(results_path.relative_to(repo_root)),
            metadata={
                "artifacts": payload.get("artifacts", {}),
            },
        )
        records[(record.platform, record.run_id)] = _merge_record(records.get((record.platform, record.run_id)), record)


def _record_from_manifest(platform_name: str, repo_root: Path, path: Path, payload: dict[str, object]) -> RunRecord:
    run_id = _run_id_from_payload_or_path(payload, path, kind="manifest")
    return RunRecord(
        run_id=run_id,
        source="manifest.json",
        platform=platform_name,
        status=str(payload.get("status", "unknown")),
        experiment=payload.get("slug"),
        device=payload.get("device"),
        started_at=payload.get("started_at_utc"),
        finished_at=payload.get("finished_at_utc"),
        archive_bytes=_archive_bytes_from_payload(payload),
        path=str(path.relative_to(repo_root)),
        metadata={key: value for key, value in payload.items() if key not in {"run_id", "status", "slug"}},
    )


def _record_from_status(platform_name: str, repo_root: Path, path: Path, payload: dict[str, object]) -> RunRecord:
    run_id = _run_id_from_payload_or_path(payload, path, kind="status")
    return RunRecord(
        run_id=run_id,
        source="status.json",
        platform=platform_name,
        status=str(payload.get("status", "unknown")),
        experiment=payload.get("slug"),
        archive_bytes=_archive_bytes_from_payload(payload),
        finished_at=payload.get("finished_at_utc"),
        path=str(path.relative_to(repo_root)),
        metadata={key: value for key, value in payload.items() if key not in {"run_id", "status", "slug"}},
    )


def _record_from_ledger_line(
    platform_name: str,
    repo_root: Path,
    path: Path,
    payload: dict[str, object],
    line_number: int,
) -> RunRecord:
    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise SchedulerValidationError(f"ledger missing run_id: {path}:{line_number}")
    return RunRecord(
        run_id=run_id,
        source="_ledger.jsonl",
        platform=platform_name,
        status=str(payload.get("status", "unknown")),
        experiment=payload.get("slug"),
        archive_bytes=_archive_bytes_from_payload(payload),
        started_at=payload.get("started_at_utc"),
        finished_at=payload.get("finished_at_utc"),
        path=str(path.relative_to(repo_root)),
        metadata={key: value for key, value in payload.items() if key not in {"run_id", "status", "slug"}},
    )


def _collect_platform_records(
    repo_root: Path,
    registry: PlatformRegistry,
    records: dict[tuple[str, str], RunRecord],
) -> None:
    for platform in registry.platforms.values():
        for pattern in platform.manifest_globs:
            for path in sorted(repo_root.glob(pattern)):
                payload = _read_json_file(path)
                record = _record_from_manifest(platform.name, repo_root, path, payload)
                records[(record.platform, record.run_id)] = _merge_record(
                    records.get((record.platform, record.run_id)),
                    record,
                )
        for pattern in platform.status_globs:
            for path in sorted(repo_root.glob(pattern)):
                payload = _read_json_file(path)
                record = _record_from_status(platform.name, repo_root, path, payload)
                records[(record.platform, record.run_id)] = _merge_record(
                    records.get((record.platform, record.run_id)),
                    record,
                )
        for rel_path in platform.ledger_paths:
            path = repo_root / rel_path
            if not path.exists():
                continue
            for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
                if not raw_line.strip():
                    continue
                try:
                    payload = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise SchedulerValidationError(
                        f"Invalid JSON in {path}:{line_number}: {exc.msg}"
                    ) from exc
                if not isinstance(payload, dict):
                    raise SchedulerValidationError(f"Expected object in {path}:{line_number}")
                record = _record_from_ledger_line(platform.name, repo_root, path, payload, line_number)
                records[(record.platform, record.run_id)] = _merge_record(
                    records.get((record.platform, record.run_id)),
                    record,
                )


def collect_run_records(repo_root: str | Path, registry: PlatformRegistry | None = None) -> list[RunRecord]:
    root = Path(repo_root)
    if not root.exists():
        raise FileNotFoundError(f"Repo root does not exist: {root}")

    records: dict[tuple[str, str], RunRecord] = {}
    _collect_result_records(root, registry, records)
    if registry is not None:
        _collect_platform_records(root, registry, records)
    return sorted(
        records.values(),
        key=lambda record: (
            record.finished_at or "",
            record.started_at or "",
            record.platform,
            record.run_id,
        ),
        reverse=True,
    )
