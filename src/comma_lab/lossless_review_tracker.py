from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from .state_models import DoctorReport, DriftFinding, SyncResult

GLOBAL_TRACKER_RELATIVE_PATH = Path(".omx/state/review_tracker.json")
LOSSLESS_TRACKER_RELATIVE_PATH = Path(".omx/state/lossless_review_tracker.json")


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON: {path}")
    return payload


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


def _lossless_entity_key(entity_key: str, entity: dict[str, object]) -> bool:
    file_path = str(entity.get("file_path", "")).replace("\\", "/")
    name = str(entity.get("name", ""))

    if file_path.startswith("src/tac/lossless/"):
        return True
    if file_path.startswith("experiments/test_tac_lossless"):
        return True
    if file_path == "src/tac/cli.py" and name in {"build_parser", "_run_lossless", "main"}:
        return True
    if file_path == "src/comma_lab/lossless_state_sync.py":
        return True
    if file_path == "src/comma_lab/cli.py" and name.startswith("cmd_lossless_"):
        return True
    return False


def _project_payload(payload: dict[str, object]) -> dict[str, object]:
    entities_payload = payload.get("entities", {})
    if not isinstance(entities_payload, dict):
        raise ValueError("review tracker payload must contain an entities object")
    filtered_entities = {
        key: value
        for key, value in entities_payload.items()
        if isinstance(value, dict) and _lossless_entity_key(key, value)
    }
    review_count = sum(1 for value in filtered_entities.values() if value.get("review_status") != "unreviewed")
    return {
        "version": payload.get("version", 0),
        "last_scan": payload.get("last_scan", ""),
        "entity_count": len(filtered_entities),
        "review_count": review_count,
        "entities": filtered_entities,
    }


def _render_payload(payload: dict[str, object]) -> str:
    return json.dumps(payload, indent=2) + "\n"


def canonical_tracker_path(repo_root: str | Path) -> Path:
    return Path(repo_root) / LOSSLESS_TRACKER_RELATIVE_PATH


def load_global_tracker(repo_root: str | Path) -> dict[str, object]:
    root = Path(repo_root)
    path = root / GLOBAL_TRACKER_RELATIVE_PATH
    if not path.exists():
        raise FileNotFoundError(f"missing global review tracker: {path}")
    return _load_json(path)


def project_tracker(repo_root: str | Path) -> dict[str, object]:
    return _project_payload(load_global_tracker(repo_root))


def doctor_repo(repo_root: str | Path) -> DoctorReport:
    root = Path(repo_root)
    expected = _render_payload(project_tracker(root))
    tracker_path = canonical_tracker_path(root)
    findings: list[DriftFinding] = []
    if not tracker_path.exists() or tracker_path.read_text() != expected:
        findings.append(
            DriftFinding(
                code="lossless_review_tracker_stale",
                severity="error",
                path=str(LOSSLESS_TRACKER_RELATIVE_PATH),
                message="lossless review tracker is stale relative to the global review tracker projection",
            )
        )
    return DoctorReport(tuple(findings))


def sync_repo(repo_root: str | Path) -> SyncResult:
    root = Path(repo_root)
    payload = project_tracker(root)
    changed_paths: list[str] = []
    if _atomic_write_text(canonical_tracker_path(root), _render_payload(payload)):
        changed_paths.append(str(LOSSLESS_TRACKER_RELATIVE_PATH))
    return SyncResult(tuple(changed_paths), doctor_repo(root).findings)


def scan_repo(repo_root: str | Path) -> SyncResult:
    root = Path(repo_root)
    subprocess.run([sys.executable, "tools/review_tracker.py", "scan"], cwd=root, check=True)
    return sync_repo(root)


def status_payload(repo_root: str | Path) -> dict[str, object]:
    payload = project_tracker(repo_root)
    entities = payload.get("entities", {})
    if not isinstance(entities, dict):
        raise ValueError("lossless review tracker projection expected entities mapping")
    counts = {"reviewed": 0, "unreviewed": 0, "stale": 0, "needs_fix": 0}
    for entity in entities.values():
        if not isinstance(entity, dict):
            continue
        status = str(entity.get("review_status", "unreviewed"))
        counts[status] = counts.get(status, 0) + 1
    counts["total"] = len(entities)
    return {
        "tracker_path": str(LOSSLESS_TRACKER_RELATIVE_PATH),
        "last_scan": payload.get("last_scan", ""),
        "counts": counts,
    }
