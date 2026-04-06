#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "reports" / "results.jsonl"
TIMELINE = ROOT / "reports" / "timeline.jsonl"
CANONICAL = ROOT / "reports" / "raw" / "robust_current-current_workflow-cpu-summary.json"
OUT = ROOT / "reports" / "graphs" / "experiment_manifest.json"
LOCAL_TZ = ZoneInfo("America/Chicago")
CHALLENGE_URL = "https://github.com/commaai/comma_video_compression_challenge"
GITHUB_URL = "https://github.com/adpena/comma-lab"


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def format_local_datetime(dt: datetime) -> str:
    local = dt.astimezone(LOCAL_TZ)
    hour = local.strftime("%I").lstrip("0") or "12"
    return f"{local.strftime('%b')} {local.day}, {local.year}, {hour}:{local.strftime('%M %p %Z')}"


def main() -> int:
    results = load_jsonl(RESULTS)
    timeline = load_jsonl(TIMELINE)
    canonical = json.loads(CANONICAL.read_text())
    robust = [r for r in results if r.get("track") == "robust_current"]
    exact = [r for r in results if r.get("track") == "exact_current"]
    best = min(robust, key=lambda r: r["current_workflow_score"])
    build_time_utc = datetime.now(timezone.utc)

    manifest = {
        "site_meta": {
            "maintainer": "Alejandro Pena",
            "repo_slug": "adpena/comma-lab",
            "github_url": GITHUB_URL,
            "challenge_url": CHALLENGE_URL,
            "updated_at_utc": build_time_utc.isoformat().replace("+00:00", "Z"),
            "updated_at_local": format_local_datetime(build_time_utc),
        },
        "tracks": {
            "exact_current": {
                "latest_current_workflow": exact[-1] if exact else None,
            },
            "robust_current": {
                "canonical_summary_path": str(CANONICAL.relative_to(ROOT)),
                "canonical_current_workflow": canonical,
                "best_run_id": best["run_id"],
            },
        },
        "promotion_runs": [r for r in robust if "promoted" in r["run_id"]],
        "timeline": timeline,
        "site_artifacts": {
            "dashboard_data": "reports/graphs/dashboard_data.json",
            "experiment_graph": "reports/graphs/experiment_graph.json",
            "promotion_accounting": "reports/graphs/promotion_accounting.md",
            "judges_one_pager": "reports/graphs/judges_one_pager.md",
            "submission_packet": "reports/graphs/submission_packet.md",
            "evidence_index": "reports/graphs/evidence_index.md",
        },
    }
    OUT.write_text(json.dumps(manifest, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
