from __future__ import annotations

from pathlib import Path

from .models import (
    BudgetReport,
    BudgetUsage,
    PlatformBudgetReport,
    PlatformRegistry,
    RunRecord,
    SchedulerStatusReport,
    TrackStatus,
)


ACTIVE_STATUSES = {
    "queued",
    "starting",
    "launching",
    "running",
    "running_managed_session",
}
FAILED_STATUSES = {"failed", "error"}


def select_result_records(records: list[RunRecord], *, track: str | None = None, limit: int | None = None) -> list[RunRecord]:
    selected = [record for record in records if record.status == "measured" and record.score is not None]
    if track is not None:
        selected = [record for record in selected if record.track == track]
    selected.sort(key=lambda record: (record.finished_at or "", record.run_id), reverse=True)
    if limit is not None:
        selected = selected[:limit]
    return selected


def build_status_report(repo_root: str | Path, records: list[RunRecord]) -> SchedulerStatusReport:
    result_records = select_result_records(records)
    latest_by_track: dict[str, RunRecord] = {}
    for record in result_records:
        if record.track and record.track not in latest_by_track:
            latest_by_track[record.track] = record

    active_runs = [record for record in records if record.status in ACTIVE_STATUSES]
    active_runs.sort(key=lambda record: (record.started_at or "", record.run_id), reverse=True)

    return SchedulerStatusReport(
        repo_root=str(Path(repo_root)),
        result_count=len(result_records),
        run_record_count=len(records),
        tracks=tuple(
            TrackStatus(track=track, latest_result=latest_by_track[track])
            for track in sorted(latest_by_track)
        ),
        active_runs=tuple(active_runs),
    )


def build_budget_report(registry: PlatformRegistry, records: list[RunRecord]) -> BudgetReport:
    platforms: dict[str, PlatformBudgetReport] = {}
    for name, platform in registry.platforms.items():
        platform_records = [record for record in records if record.platform == name]
        usage = BudgetUsage(
            total_runs=len(platform_records),
            active_runs=sum(1 for record in platform_records if record.status in ACTIVE_STATUSES),
            failed_runs=sum(1 for record in platform_records if record.status in FAILED_STATUSES),
            archive_bytes=sum(record.archive_bytes or 0 for record in platform_records),
        )
        budget = platform.budget
        over_budget = False
        if budget.max_runs is not None and usage.total_runs > budget.max_runs:
            over_budget = True
        if budget.max_active_runs is not None and usage.active_runs > budget.max_active_runs:
            over_budget = True
        if budget.max_failed_runs is not None and usage.failed_runs > budget.max_failed_runs:
            over_budget = True
        if budget.max_archive_bytes is not None and usage.archive_bytes > budget.max_archive_bytes:
            over_budget = True
        platforms[name] = PlatformBudgetReport(platform=platform, usage=usage, over_budget=over_budget)

    return BudgetReport(platforms=platforms)
