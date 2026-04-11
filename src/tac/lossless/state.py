from __future__ import annotations

import json
import tempfile
from dataclasses import asdict
from pathlib import Path

from .contracts import LosslessCompressionResult


def load_lossless_result(path: str | Path) -> LosslessCompressionResult:
    payload = json.loads(Path(path).read_text())
    return LosslessCompressionResult(**payload)


def render_lossless_latest(result: LosslessCompressionResult) -> str:
    return (
        "# Lossless Latest\n\n"
        f"Current promoted lossless baseline is **`{result.compression_rate:.4f}`** via `{result.profile}` using `{result.method}`.\n\n"
        "## promoted result\n\n"
        "- Status: exact round-trip confirmed\n"
        f"- Profile: `{result.profile}`\n"
        f"- Method: `{result.method}`\n"
        f"- Compression rate: **`{result.compression_rate:.4f}`**\n"
        f"- Archive bytes: `{result.archive_bytes}`\n"
        f"- Original bytes: `{result.original_bytes}`\n"
        f"- Archive path: `{result.archive_path}`\n"
        "- Ledgers: `reports/lossless_results.jsonl`, `reports/lossless_timeline.jsonl`\n"
    )


def _render_lossless_focus(result: LosslessCompressionResult) -> str:
    return (
        "# Lossless Focus\n\n"
        "## current promoted baseline\n"
        f"- Profile: `{result.profile}`\n"
        f"- Method: `{result.method}`\n"
        f"- Compression rate: `{result.compression_rate:.4f}`\n"
        f"- Archive bytes: `{result.archive_bytes}`\n"
    )


def _render_lossless_next_experiments(result: LosslessCompressionResult) -> str:
    return (
        "# Lossless Next Experiments\n\n"
        f"## promoted baseline: `{result.profile}` at `{result.compression_rate:.4f}`\n\n"
        "1. Preserve the first measured lossless baseline and decompressor contract.\n"
        "2. Beat the promoted rate with another measured exact round-trip.\n"
        "3. Keep lossless promotion state separate from lossy promotion state.\n"
    )


def _render_lossless_findings(result: LosslessCompressionResult) -> str:
    return (
        "# Lossless Findings\n\n"
        "## current promoted result\n\n"
        f"- The first measured lossless baseline is now **`{result.compression_rate:.4f}`**.\n"
        f"- Current promoted rate: `{result.compression_rate:.4f}`.\n"
        f"- Profile: `{result.profile}`.\n"
        f"- Method: `{result.method}`.\n"
        "- The lossless promotion flow is separate from the lossy promotion flow.\n"
        "- Lossless report surfaces are derived from the promoted lossless result record.\n"
    )


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


def _results_row(result: LosslessCompressionResult) -> dict[str, object]:
    return {
        "profile": result.profile,
        "method": result.method,
        "archive_path": result.archive_path,
        "archive_bytes": result.archive_bytes,
        "original_bytes": result.original_bytes,
        "compression_rate": result.compression_rate,
        "event": "promotion",
    }


def _timeline_row(result: LosslessCompressionResult) -> dict[str, object]:
    return {
        "event": "promotion",
        "profile": result.profile,
        "method": result.method,
        "compression_rate": result.compression_rate,
        "archive_bytes": result.archive_bytes,
        "original_bytes": result.original_bytes,
        "archive_path": result.archive_path,
    }


def promote_lossless_result(*, repo_root: str | Path, result_path: str | Path) -> dict[str, object]:
    root = Path(repo_root)
    result = load_lossless_result(result_path)
    archive_path = Path(result.archive_path)
    if not archive_path.exists():
        raise FileNotFoundError(f"Lossless archive not found: {archive_path}")

    results_row = _results_row(result)
    timeline_row = _timeline_row(result)

    reports_path = root / "reports" / "lossless_results.jsonl"
    timeline_path = root / "reports" / "lossless_timeline.jsonl"
    latest_path = root / "reports" / "lossless_latest.md"
    focus_path = root / ".omx" / "state" / "lossless_focus.md"
    next_experiments_path = root / ".omx" / "state" / "lossless_next_experiments.md"
    findings_path = root / ".omx" / "research" / "lossless_findings.md"

    changed_paths: list[str] = []
    if _write_jsonl(reports_path, _dedupe_rows(_read_jsonl(reports_path), results_row)):
        changed_paths.append("reports/lossless_results.jsonl")
    if _write_jsonl(timeline_path, _dedupe_rows(_read_jsonl(timeline_path), timeline_row)):
        changed_paths.append("reports/lossless_timeline.jsonl")
    if _atomic_write_text(latest_path, render_lossless_latest(result)):
        changed_paths.append("reports/lossless_latest.md")
    if _atomic_write_text(focus_path, _render_lossless_focus(result)):
        changed_paths.append(".omx/state/lossless_focus.md")
    if _atomic_write_text(next_experiments_path, _render_lossless_next_experiments(result)):
        changed_paths.append(".omx/state/lossless_next_experiments.md")
    if _atomic_write_text(findings_path, _render_lossless_findings(result)):
        changed_paths.append(".omx/research/lossless_findings.md")

    return {
        "command": "lossless_promote",
        "result": asdict(result),
        "repo_root": str(root),
        "changed_paths": changed_paths,
    }
