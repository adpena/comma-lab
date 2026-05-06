from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

from .state_models import DoctorReport, DriftFinding, PromotedResult, SyncResult, canonical_record_path


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON: {path}")
    return payload


def load_promoted_result(repo_root: Path) -> PromotedResult:
    return PromotedResult.from_dict(_load_json(canonical_record_path(repo_root)))


def promote_record(repo_root: str | Path, *, record_path: str | Path | None = None) -> SyncResult:
    root = Path(repo_root)
    source_path = Path(record_path) if record_path is not None else canonical_record_path(root)
    payload = _load_json(source_path)
    result = PromotedResult.from_dict(payload)

    authoritative_report = root / result.authoritative_report_path
    artifact_path = root / result.artifact_path
    if not authoritative_report.exists():
        raise FileNotFoundError(f"Authoritative report not found: {authoritative_report}")
    if not artifact_path.exists():
        raise FileNotFoundError(f"Promoted artifact not found: {artifact_path}")

    if source_path.resolve() != canonical_record_path(root).resolve():
        _atomic_write_json(canonical_record_path(root), result.to_dict())
    return sync_repo(root)


def _atomic_write_text(path: Path, content: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text() if path.exists() else None
    if existing == content:
        return False
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8") as handle:
        handle.write(content)
        temp_path = Path(handle.name)
    temp_path.replace(path)
    return True


def _atomic_write_json(path: Path, payload: object) -> bool:
    return _atomic_write_text(path, json.dumps(payload, indent=2) + "\n")


def _read_text(path: Path) -> str:
    return path.read_text() if path.exists() else ""


def _read_jsonl_for_doctor(root: Path, rel_path: str) -> tuple[list[dict[str, object]], list[DriftFinding]]:
    path = root / rel_path
    rows: list[dict[str, object]] = []
    findings: list[DriftFinding] = []
    if not path.exists():
        return rows, findings
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            findings.append(
                DriftFinding(
                    code="malformed_jsonl",
                    severity="error",
                    path=f"{rel_path}:{lineno}",
                    message=f"ledger row is not valid JSON: {exc.msg}",
                )
            )
            continue
        if not isinstance(payload, dict):
            findings.append(
                DriftFinding(
                    code="malformed_jsonl",
                    severity="error",
                    path=f"{rel_path}:{lineno}",
                    message="ledger row must be a JSON object",
                )
            )
            continue
        rows.append(payload)
    return rows, findings


def _render_latest_md(result: PromotedResult) -> str:
    return (
        "# latest report\n\n"
        f"## current state - {result.promoted_at[:10]}\n\n"
        f"Track B's promoted honest floor is now **`{result.score:.2f}`** via `{result.variant}`.\n\n"
        "## authoritative promoted floor\n\n"
        f"- Track: `{result.track}`\n"
        f"- Variant: `{result.variant}`\n"
        f"- Platform: `{result.platform}`\n"
        f"- Current-workflow score: **`{result.score:.2f}`** at `{result.archive_bytes:,}` bytes\n"
        f"- Distortions: PoseNet `{result.pose_distortion:.8f}`, SegNet `{result.seg_distortion:.8f}`\n"
        f"- Rate: `{result.rate:.8f}`\n"
        f"- Evidence: `{result.authoritative_report_path}`\n"
    )


def _render_current_focus_md(result: PromotedResult) -> str:
    return (
        f"# Current Focus — {result.promoted_at}\n\n"
        "## Floor\n"
        f"- **Official score**: {result.score:.2f}\n"
        f"- **Variant**: {result.variant}\n"
        f"- **Platform**: {result.platform}\n"
        f"- **Epoch**: {result.epoch}\n"
    )


def _render_next_experiments_md(result: PromotedResult) -> str:
    return (
        "# next experiments\n\n"
        f"## promoted floor: {result.score:.2f} ({result.variant})\n\n"
        "## next steps\n\n"
        "1. Preserve the promoted artifact and keep the canonical state synced from one record.\n"
        "2. Evaluate live Modal h96 artifacts against the promoted floor.\n"
        "3. Treat Kaggle as secondary until a deliberate tac-first retry is ready.\n"
    )


def _render_findings_md(result: PromotedResult) -> str:
    return (
        f"# Findings\n\n## {result.promoted_at[:10]} promoted floor\n\n"
        f"- Track B promoted floor is **{result.score:.2f}**.\n"
        f"- Variant: `{result.variant}`.\n"
        f"- Platform: `{result.platform}`.\n"
        f"- PoseNet `{result.pose_distortion:.8f}`, SegNet `{result.seg_distortion:.8f}`, rate `{result.rate:.8f}`.\n"
        "- Canonical score/report mirrors are generated from `.omx/state/promoted_result.json`.\n"
    )


def _render_run_log_md(result: PromotedResult) -> str:
    return (
        "# run log\n\n"
        f"## {result.promoted_at} - promoted floor synchronized\n\n"
        f"- authoritative promoted floor: **{result.score:.2f}**\n"
        f"- variant: `{result.variant}`\n"
        f"- platform: `{result.platform}`\n"
        f"- evidence: `{result.authoritative_report_path}`\n"
        "- mirrors are now expected to be derived from canonical promoted_result.json\n"
    )


def _ps_exists(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        completed = subprocess.run(
            ["ps", "-p", str(pid)],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    return completed.returncode == 0 and str(pid) in completed.stdout


def _doctor_managed_sessions(repo_root: Path) -> list[DriftFinding]:
    findings: list[DriftFinding] = []
    manifest_root = repo_root / ".omx" / "logs" / "remote_jobs"
    if not manifest_root.exists():
        return findings
    for path in sorted(manifest_root.glob("*.json")):
        payload = _load_json(path)
        if payload.get("status") != "running_managed_session":
            continue
        session_id = payload.get("session_id")
        if isinstance(session_id, int) and not _ps_exists(session_id):
            findings.append(
                DriftFinding(
                    code="stale_managed_session",
                    severity="warning",
                    path=str(path.relative_to(repo_root)),
                    message=f"managed session {session_id} is no longer live",
                )
            )
    return findings


def doctor_repo(repo_root: str | Path) -> DoctorReport:
    root = Path(repo_root)
    result = load_promoted_result(root)
    findings: list[DriftFinding] = []

    summary_path = root / result.summary_path
    expected_summary = result.summary_payload()
    if not summary_path.exists() or _load_json(summary_path) != expected_summary:
        findings.append(
            DriftFinding(
                code="canonical_summary_mismatch",
                severity="error",
                path=str(summary_path.relative_to(root)),
                message="canonical summary does not match promoted_result.json",
            )
        )

    canonical_report_path = root / result.authoritative_report_copy_path
    authoritative_report_path = root / result.authoritative_report_path
    if (
        not canonical_report_path.exists()
        or not authoritative_report_path.exists()
        or _read_text(canonical_report_path) != _read_text(authoritative_report_path)
    ):
        findings.append(
            DriftFinding(
                code="canonical_report_mismatch",
                severity="error",
                path=str(canonical_report_path.relative_to(root)),
                message="canonical copied report does not match authoritative evidence",
            )
        )

    latest_path = root / "reports" / "latest.md"
    expected_latest = _render_latest_md(result)
    if not latest_path.exists() or _read_text(latest_path) != expected_latest:
        findings.append(
            DriftFinding(
                code="latest_report_stale",
                severity="error",
                path=str(latest_path.relative_to(root)),
                message="latest.md is stale relative to promoted_result.json",
            )
        )

    surface_expectations = {
        ".omx/state/current_focus.md": _render_current_focus_md(result),
        ".omx/state/next_experiments.md": _render_next_experiments_md(result),
        ".omx/research/findings.md": _render_findings_md(result),
        ".ralph/run_log.md": _render_run_log_md(result),
    }
    for rel_path, expected_text in surface_expectations.items():
        path = root / rel_path
        if not path.exists() or _read_text(path) != expected_text:
            findings.append(
                DriftFinding(
                    code="projected_state_stale",
                    severity="error",
                    path=rel_path,
                    message="projected state surface is stale relative to promoted_result.json",
                )
            )

    results_rows, jsonl_findings = _read_jsonl_for_doctor(root, "reports/results.jsonl")
    findings.extend(jsonl_findings)
    if not any(row.get("run_id") == result.run_id and float(row.get("current_workflow_score")) == result.score for row in results_rows):
        findings.append(
            DriftFinding(
                code="results_ledger_stale",
                severity="error",
                path="reports/results.jsonl",
                message="results ledger is missing the canonical promoted result row",
            )
        )

    timeline_rows, jsonl_findings = _read_jsonl_for_doctor(root, "reports/timeline.jsonl")
    findings.extend(jsonl_findings)
    if not any(float(row.get("score", -1)) == result.score and row.get("variant") == result.variant for row in timeline_rows):
        findings.append(
            DriftFinding(
                code="timeline_ledger_stale",
                severity="error",
                path="reports/timeline.jsonl",
                message="timeline ledger is missing the canonical promoted event",
            )
        )

    findings.extend(_doctor_managed_sessions(root))
    return DoctorReport(findings=tuple(findings))


def _sync_ledgers(root: Path, result: PromotedResult) -> list[str]:
    changed: list[str] = []

    results_path = root / "reports" / "results.jsonl"
    rows: list[dict[str, object]] = []
    if results_path.exists():
        rows = [json.loads(line) for line in results_path.read_text().splitlines() if line.strip()]
    row = result.results_row()
    rows = [existing for existing in rows if existing.get("run_id") != result.run_id]
    rows.append(row)
    rows.sort(key=lambda item: str(item.get("ts_utc", "")))
    content = "".join(json.dumps(item) + "\n" for item in rows)
    if _atomic_write_text(results_path, content):
        changed.append(str(results_path.relative_to(root)))

    timeline_path = root / "reports" / "timeline.jsonl"
    timeline_rows: list[dict[str, object]] = []
    if timeline_path.exists():
        timeline_rows = [json.loads(line) for line in timeline_path.read_text().splitlines() if line.strip()]
    timeline_event = result.timeline_event()
    timeline_rows = [existing for existing in timeline_rows if existing.get("ts") != timeline_event["ts"]]
    timeline_rows.append(timeline_event)
    timeline_rows.sort(key=lambda item: str(item.get("ts", item.get("ts_utc", ""))))
    timeline_content = "".join(json.dumps(item) + "\n" for item in timeline_rows)
    if _atomic_write_text(timeline_path, timeline_content):
        changed.append(str(timeline_path.relative_to(root)))

    return changed


def _sync_managed_sessions(root: Path) -> list[str]:
    changed: list[str] = []
    manifest_root = root / ".omx" / "logs" / "remote_jobs"
    if not manifest_root.exists():
        return changed
    for path in sorted(manifest_root.glob("*.json")):
        payload = _load_json(path)
        if payload.get("status") != "running_managed_session":
            continue
        session_id = payload.get("session_id")
        if not isinstance(session_id, int) or _ps_exists(session_id):
            continue
        payload["status"] = "stale"
        notes = str(payload.get("notes", "")).strip()
        extra = f"Managed process/session not found during state sync (session_id={session_id})."
        payload["notes"] = f"{notes}\n{extra}".strip() if notes else extra
        if _atomic_write_json(path, payload):
            changed.append(str(path.relative_to(root)))
    return changed


def sync_repo(repo_root: str | Path) -> SyncResult:
    root = Path(repo_root)
    result = load_promoted_result(root)
    findings = doctor_repo(root).findings
    changed: list[str] = []

    authoritative_report = root / result.authoritative_report_path
    canonical_report = root / result.authoritative_report_copy_path
    if not authoritative_report.exists():
        raise FileNotFoundError(f"Authoritative report not found: {authoritative_report}")
    if _atomic_write_text(canonical_report, authoritative_report.read_text()):
        changed.append(str(canonical_report.relative_to(root)))

    if _atomic_write_json(root / result.summary_path, result.summary_payload()):
        changed.append(result.summary_path)
    if _atomic_write_text(root / "reports" / "latest.md", _render_latest_md(result)):
        changed.append("reports/latest.md")
    if _atomic_write_text(root / ".omx" / "state" / "current_focus.md", _render_current_focus_md(result)):
        changed.append(".omx/state/current_focus.md")
    if _atomic_write_text(root / ".omx" / "state" / "next_experiments.md", _render_next_experiments_md(result)):
        changed.append(".omx/state/next_experiments.md")
    if _atomic_write_text(root / ".omx" / "research" / "findings.md", _render_findings_md(result)):
        changed.append(".omx/research/findings.md")
    if _atomic_write_text(root / ".ralph" / "run_log.md", _render_run_log_md(result)):
        changed.append(".ralph/run_log.md")

    changed.extend(_sync_ledgers(root, result))
    changed.extend(_sync_managed_sessions(root))
    return SyncResult(changed_paths=tuple(dict.fromkeys(changed)), findings=findings)
