#!/usr/bin/env python3
"""Build a scorecard from GitHub-hosted PR auth-eval comments.

The contest bot can leave multiple eval comments on the same public PR, often
with different devices. This tool records those comment rows explicitly so a
CUDA replay is not accidentally compared with a CPU leaderboard row.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any

DEFAULT_REPO = "commaai/comma_video_compression_challenge"
CONTEST_N_BYTES = 37_545_489


def recompute_score(*, pose: float, seg: float, archive_bytes: int) -> float:
    return 100.0 * seg + math.sqrt(10.0 * pose) + 25.0 * archive_bytes / CONTEST_N_BYTES


def _component(label: str, body: str) -> str | None:
    match = re.search(rf"{re.escape(label)}\s*:\s*([0-9.,]+)", body, re.IGNORECASE)
    return match.group(1) if match else None


def parse_eval_comment(body: str) -> dict[str, Any] | None:
    """Parse one GitHub Actions eval comment body.

    Returns None when the body is not an eval comment. Raises ValueError for an
    eval-shaped comment whose required numeric lines are missing, because that
    should be reviewed before the scorecard is trusted.
    """

    if "Eval Results" not in body:
        return None
    device_match = re.search(r"^\s*device:\s*([^\n]+)", body, re.MULTILINE)
    final_match = re.search(
        r"Final score:.*?=\s*([0-9.]+)",
        body,
        re.IGNORECASE | re.DOTALL,
    )
    pose = _component("Average PoseNet Distortion", body)
    seg = _component("Average SegNet Distortion", body)
    size = _component("Submission file size", body)
    missing = [
        name
        for name, value in (
            ("device", device_match.group(1) if device_match else None),
            ("Average PoseNet Distortion", pose),
            ("Average SegNet Distortion", seg),
            ("Submission file size", size),
            ("Final score", final_match.group(1) if final_match else None),
        )
        if value is None
    ]
    if missing:
        raise ValueError(f"eval comment missing required fields: {', '.join(missing)}")

    pose_f = float(str(pose).replace(",", ""))
    seg_f = float(str(seg).replace(",", ""))
    archive_bytes = int(str(size).replace(",", ""))
    printed_score = float(final_match.group(1))  # type: ignore[union-attr]
    recomputed = recompute_score(pose=pose_f, seg=seg_f, archive_bytes=archive_bytes)
    return {
        "device": device_match.group(1).strip(),  # type: ignore[union-attr]
        "pose": pose_f,
        "seg": seg_f,
        "archive_bytes": archive_bytes,
        "printed_score": printed_score,
        "recomputed_score_from_rounded_comment_components": recomputed,
        "printed_minus_recomputed": printed_score - recomputed,
    }


def parse_pr_json(pr_json: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for comment in pr_json.get("comments", []):
        body = comment.get("body") or ""
        parsed = parse_eval_comment(body)
        if parsed is None:
            continue
        parsed["author"] = (comment.get("author") or {}).get("login")
        parsed["created_at"] = comment.get("createdAt")
        parsed["updated_at"] = comment.get("updatedAt")
        rows.append(parsed)
    return {
        "pr": pr_json.get("number"),
        "title": pr_json.get("title"),
        "url": pr_json.get("url"),
        "author": (pr_json.get("author") or {}).get("login"),
        "head_sha": pr_json.get("headRefOid"),
        "eval_comments": rows,
    }


def fetch_pr_json(repo: str, pr: int) -> dict[str, Any]:
    proc = subprocess.run(
        [
            "gh",
            "pr",
            "view",
            str(pr),
            "--repo",
            repo,
            "--json",
            "number,title,url,author,headRefOid,comments",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(proc.stdout)


def build_scorecard(repo: str, prs: list[int]) -> dict[str, Any]:
    return {
        "schema": "public_pr_eval_comment_scorecard.v1",
        "repo": repo,
        "evidence_grade": "external_github_pr_comment",
        "score_claim": False,
        "promotion_eligible": False,
        "rows": [parse_pr_json(fetch_pr_json(repo, pr)) for pr in prs],
    }


def _parse_prs(args: argparse.Namespace) -> list[int]:
    prs: list[int] = []
    for value in args.pr or []:
        prs.append(int(value))
    if args.pr_range:
        start, end = args.pr_range
        prs.extend(range(start, end + 1))
    return sorted(set(prs))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO)
    parser.add_argument("--pr", action="append", type=int, help="PR number to inspect.")
    parser.add_argument(
        "--pr-range",
        nargs=2,
        type=int,
        metavar=("START", "END"),
        help="Inclusive PR range to inspect.",
    )
    parser.add_argument("--json-out", type=Path, help="Write scorecard JSON.")
    args = parser.parse_args()

    prs = _parse_prs(args)
    if not prs:
        raise SystemExit("provide at least one --pr or --pr-range START END")

    scorecard = build_scorecard(args.repo, prs)
    text = json.dumps(scorecard, indent=2, sort_keys=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
