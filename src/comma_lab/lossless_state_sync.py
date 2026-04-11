from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from .state_models import DoctorReport, DriftFinding, SyncResult

if TYPE_CHECKING:
    from src.tac.lossless.contracts import LosslessCompressionResult


def _lossless_result_type():
    from src.tac.lossless.contracts import LosslessCompressionResult

    return LosslessCompressionResult


def _render_lossless_latest(result) -> str:
    from src.tac.lossless.state import render_lossless_latest

    return render_lossless_latest(result)


def canonical_record_path(repo_root: Path) -> Path:
    return repo_root / ".omx" / "state" / "lossless_promoted_result.json"


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON: {path}")
    return payload


def load_promoted_result(repo_root: str | Path) -> "LosslessCompressionResult":
    root = Path(repo_root)
    return _lossless_result_type()(**_load_json(canonical_record_path(root)))


def _read_text(path: Path) -> str:
    return path.read_text() if path.exists() else ""


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


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> bool:
    rendered = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
    return _atomic_write_text(path, rendered)


def _dedupe_rows(rows: list[dict[str, object]], canonical_row: dict[str, object]) -> list[dict[str, object]]:
    def same_promoted_result(left: dict[str, object], right: dict[str, object]) -> bool:
        return (
            left.get("event") == right.get("event") == "promotion"
            and left.get("profile") == right.get("profile")
            and left.get("method") == right.get("method")
            and left.get("compression_rate") == right.get("compression_rate")
            and left.get("original_bytes") == right.get("original_bytes")
        )

    filtered = [row for row in rows if row != canonical_row and not same_promoted_result(row, canonical_row)]
    filtered.append(canonical_row)
    return filtered


def _render_lossless_focus(result: "LosslessCompressionResult") -> str:
    return (
        "# Lossless Focus\n\n"
        "## promoted floor\n"
        f"- Profile: `{result.profile}`\n"
        f"- Method: `{result.method}`\n"
        f"- Compression rate: `{result.compression_rate:.4f}`\n"
        f"- Archive bytes: `{result.archive_bytes}`\n"
    )


def _render_lossless_findings(result: "LosslessCompressionResult") -> str:
    return (
        "# Lossless Findings\n\n"
        "## current promoted result\n\n"
        f"- Current promoted rate: `{result.compression_rate:.4f}`.\n"
        f"- Profile: `{result.profile}`.\n"
        f"- Method: `{result.method}`.\n"
        "- The lossless promotion flow is separate from the lossy promotion flow.\n"
        "- Lossless report surfaces are derived from the promoted lossless result record.\n"
    )


def _render_lossless_next_experiments(result: "LosslessCompressionResult") -> str:
    return (
        "# Lossless Next Experiments\n\n"
        f"## promoted baseline: `{result.profile}` at `{result.compression_rate:.4f}`\n\n"
        "1. Preserve the first measured lossless baseline and decompressor contract.\n"
        "2. Beat the promoted rate with another measured exact round-trip.\n"
        "3. Keep lossless promotion state separate from lossy promotion state.\n"
    )


def _results_row(result: "LosslessCompressionResult") -> dict[str, object]:
    return {
        "profile": result.profile,
        "method": result.method,
        "archive_path": result.archive_path,
        "archive_bytes": result.archive_bytes,
        "original_bytes": result.original_bytes,
        "compression_rate": result.compression_rate,
        "event": "promotion",
    }


def _timeline_row(result: "LosslessCompressionResult") -> dict[str, object]:
    return {
        "event": "promotion",
        "profile": result.profile,
        "method": result.method,
        "compression_rate": result.compression_rate,
        "archive_bytes": result.archive_bytes,
        "original_bytes": result.original_bytes,
        "archive_path": result.archive_path,
    }


def promote_record(repo_root: str | Path, *, record_path: str | Path | None = None) -> SyncResult:
    root = Path(repo_root)
    canonical_path = canonical_record_path(root)
    source_path = Path(record_path) if record_path is not None else canonical_path
    result = _lossless_result_type()(**_load_json(source_path))
    archive_path = Path(result.archive_path)
    if not archive_path.exists():
        raise FileNotFoundError(f"Lossless archive not found: {archive_path}")

    if source_path.resolve() != canonical_path.resolve():
        _atomic_write_json(canonical_path, result.__dict__)
    return sync_repo(root)


def doctor_repo(repo_root: str | Path) -> DoctorReport:
    root = Path(repo_root)
    result = load_promoted_result(root)
    findings: list[DriftFinding] = []

    surface_expectations = {
        "reports/lossless_latest.md": ("lossless_latest_stale", _render_lossless_latest(result)),
        "reports/lossless_results.jsonl": ("lossless_results_stale", None),
        "reports/lossless_timeline.jsonl": ("lossless_timeline_stale", None),
        ".omx/state/lossless_focus.md": ("lossless_focus_stale", _render_lossless_focus(result)),
        ".omx/state/lossless_next_experiments.md": (
            "lossless_next_experiments_stale",
            _render_lossless_next_experiments(result),
        ),
        ".omx/research/lossless_findings.md": ("lossless_findings_stale", _render_lossless_findings(result)),
    }

    for rel_path, (code, expected_text) in surface_expectations.items():
        path = root / rel_path
        if expected_text is not None:
            if not path.exists() or _read_text(path) != expected_text:
                findings.append(
                    DriftFinding(
                        code=code,
                        severity="error",
                        path=rel_path,
                        message="lossless projected state surface is stale relative to lossless_promoted_result.json",
                    )
                )

    results_path = root / "reports" / "lossless_results.jsonl"
    results_rows = _read_jsonl(results_path)
    if _results_row(result) not in results_rows:
        findings.append(
            DriftFinding(
                code="lossless_results_stale",
                severity="error",
                path="reports/lossless_results.jsonl",
                message="lossless results ledger is missing the canonical promoted row",
            )
        )

    timeline_path = root / "reports" / "lossless_timeline.jsonl"
    timeline_rows = _read_jsonl(timeline_path)
    if _timeline_row(result) not in timeline_rows:
        findings.append(
            DriftFinding(
                code="lossless_timeline_stale",
                severity="error",
                path="reports/lossless_timeline.jsonl",
                message="lossless timeline ledger is missing the canonical promoted row",
            )
        )

    return DoctorReport(tuple(findings))


def sync_repo(repo_root: str | Path) -> SyncResult:
    root = Path(repo_root)
    result = load_promoted_result(root)
    findings = doctor_repo(root).findings
    changed_paths: list[str] = []

    if _atomic_write_text(root / "reports" / "lossless_latest.md", _render_lossless_latest(result)):
        changed_paths.append("reports/lossless_latest.md")
    if _write_jsonl(
        root / "reports" / "lossless_results.jsonl",
        _dedupe_rows(_read_jsonl(root / "reports" / "lossless_results.jsonl"), _results_row(result)),
    ):
        changed_paths.append("reports/lossless_results.jsonl")
    if _write_jsonl(
        root / "reports" / "lossless_timeline.jsonl",
        _dedupe_rows(_read_jsonl(root / "reports" / "lossless_timeline.jsonl"), _timeline_row(result)),
    ):
        changed_paths.append("reports/lossless_timeline.jsonl")
    if _atomic_write_text(root / ".omx" / "state" / "lossless_focus.md", _render_lossless_focus(result)):
        changed_paths.append(".omx/state/lossless_focus.md")
    if _atomic_write_text(
        root / ".omx" / "state" / "lossless_next_experiments.md",
        _render_lossless_next_experiments(result),
    ):
        changed_paths.append(".omx/state/lossless_next_experiments.md")
    if _atomic_write_text(root / ".omx" / "research" / "lossless_findings.md", _render_lossless_findings(result)):
        changed_paths.append(".omx/research/lossless_findings.md")

    return SyncResult(tuple(changed_paths), findings)
