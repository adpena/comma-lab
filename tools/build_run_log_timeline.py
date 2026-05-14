#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Extract run log entries into a structured JSON timeline for the site.

Parses .ralph/run_log.md into timestamped entries, extracts scores from
reports/raw/*-summary.json, and outputs a single timeline JSON.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path


def parse_run_log(path: str) -> list[dict]:
    """Parse run_log.md into a list of entries."""
    with open(path) as f:
        lines = f.readlines()

    entries = []
    current = None

    for i, line in enumerate(lines):
        if line.startswith("## 2026-"):
            if current:
                current["body"] = "".join(current["_lines"]).strip()
                del current["_lines"]
                entries.append(current)

            title = line.strip().lstrip("# ").strip()
            # Extract date from title
            date_match = re.match(
                r"(2026-\d{2}-\d{2})(?:[T ](\d{2}:\d{2}(?::\d{2})?))?", title
            )
            date_str = date_match.group(1) if date_match else ""
            time_str = date_match.group(2) if date_match and date_match.group(2) else ""

            # Clean title: remove date/time prefix
            clean_title = re.sub(
                r"^2026-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?(?:\s*-\d{4})?)?[\s\-—]+",
                "",
                title,
            ).strip()

            # Extract score mentions
            score_match = re.search(r"(\d+\.\d{2,})", clean_title)
            score = float(score_match.group(1)) if score_match else None

            # Detect entry type
            entry_type = "cycle"
            lower = clean_title.lower()
            if "breakthrough" in lower:
                entry_type = "breakthrough"
            elif "promotion" in lower or "promoted" in lower:
                entry_type = "promotion"
            elif "critical" in lower or "bug" in lower or "mismatch" in lower:
                entry_type = "fix"
            elif "hardening" in lower or "rigor" in lower:
                entry_type = "hardening"
            elif "diagnostic" in lower or "poll" in lower or "snapshot" in lower:
                entry_type = "diagnostic"

            current = {
                "line": i + 1,
                "date": date_str,
                "time": time_str,
                "title": clean_title,
                "type": entry_type,
                "score": score,
                "_lines": [],
            }
        elif current:
            current["_lines"].append(line)

    if current:
        current["body"] = "".join(current["_lines"]).strip()
        del current["_lines"]
        entries.append(current)

    return entries


def collect_scores(reports_dir: str) -> list[dict]:
    """Collect all scored runs from report JSON files."""
    scores = []
    raw_dir = Path(reports_dir) / "raw"
    if not raw_dir.exists():
        return scores

    for json_file in sorted(raw_dir.rglob("*summary*.json")):
        try:
            data = json.loads(json_file.read_text())
            if "current_workflow_score" in data or "rule_faithful_score" in data:
                # Extract date from parent directory name
                parent = json_file.parent.name
                date_match = re.match(r"(2026-\d{2}-\d{2})", parent)
                date = date_match.group(1) if date_match else ""

                scores.append({
                    "file": str(json_file.relative_to(raw_dir)),
                    "date": date,
                    "cw_score": data.get("current_workflow_score"),
                    "rf_score": data.get("rule_faithful_score"),
                    "pose": data.get("pose_distortion"),
                    "seg": data.get("seg_distortion"),
                    "rate": data.get("current_workflow_rate"),
                    "track": data.get("track", ""),
                })
        except (json.JSONDecodeError, KeyError):
            continue

    return scores


def build_timeline(repo_dir: str) -> dict:
    """Build the complete timeline data."""
    run_log_path = os.path.join(repo_dir, ".ralph/run_log.md")
    reports_dir = os.path.join(repo_dir, "reports")

    entries = parse_run_log(run_log_path) if os.path.exists(run_log_path) else []
    scores = collect_scores(reports_dir)

    # Extract milestone scores from entries
    milestones = []
    for e in entries:
        if e["type"] == "breakthrough" and e["score"]:
            milestones.append({
                "date": e["date"],
                "time": e["time"],
                "score": e["score"],
                "label": e["title"],
            })

    # Count report directories by date
    report_dirs = []
    raw_dir = Path(reports_dir) / "raw"
    if raw_dir.exists():
        for d in sorted(raw_dir.iterdir()):
            if d.is_dir() and d.name.startswith("2026-"):
                report_dirs.append(d.name)

    return {
        "generated_at": __import__("datetime").datetime.now().isoformat(),
        "total_entries": len(entries),
        "total_scores": len(scores),
        "total_report_dirs": len(report_dirs),
        "entries": entries,
        "milestones": milestones,
        "scores": scores,
        "report_dirs": report_dirs,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build run log timeline JSON")
    parser.add_argument("--repo", default=".", help="Repo root")
    parser.add_argument(
        "--out",
        default="reports/graphs/site/run_log_timeline.json",
        help="Output path",
    )
    args = parser.parse_args()

    timeline = build_timeline(os.path.abspath(args.repo))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(timeline, indent=2))

    size = out_path.stat().st_size / 1024
    print(f"Timeline: {timeline['total_entries']} entries, {timeline['total_scores']} scores, {timeline['total_report_dirs']} report dirs")
    print(f"Written to {out_path} ({size:.1f} KB)")


if __name__ == "__main__":
    main()
