#!/usr/bin/env python3
"""Leaderboard-move detector for the comma video compression challenge.

Polls the upstream challenge README, extracts the canonical TABLE-START/TABLE-END
leaderboard block, computes a stable hash of the score column, and emits a
RACE_MODE_DETECTED signal when the score column changes.

Outputs:
    - .omx/state/leaderboard_state.json — last hash + timestamp + top-3 PRs
    - .omx/state/RACE_MODE_ACTIVE.flag  — touched on every detected change
    - .omx/state/leaderboard_changes.jsonl — appended on every detected change

Modes:
    --initialize-baseline : capture current state without alerting (first run)
    --cron                : exit 0 silently when hash unchanged (cron-friendly);
                            exit 0 with RACE_MODE_DETECTED=1 on stdout when changed

Default mode (no flag) prints a human-readable summary every invocation and
emits RACE_MODE_DETECTED=1 on stdout iff the score column changed.

The README is fetched via `gh api repos/commaai/comma_video_compression_challenge/readme`.
The canonical block is delimited by HTML comments::

    <!-- TABLE-START -->
    <table class="ranked"> ... </table>
    <!-- TABLE-END -->

Stability: the hash covers ONLY the score column (not names, not PR links),
because cosmetic README edits (typo fixes, emoji additions) must NOT trigger a
race-mode alert. Score-column changes are the only signal that matters.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STATE_DIR = REPO / ".omx" / "state"
STATE_PATH = STATE_DIR / "leaderboard_state.json"
RACE_FLAG_PATH = STATE_DIR / "RACE_MODE_ACTIVE.flag"
CHANGES_JSONL_PATH = STATE_DIR / "leaderboard_changes.jsonl"

TABLE_START = "<!-- TABLE-START -->"
TABLE_END = "<!-- TABLE-END -->"

UPSTREAM_REPO = "commaai/comma_video_compression_challenge"


@dataclass
class LeaderboardEntry:
    rank: int
    score: float
    name: str
    pr_url: str | None
    pr_number: int | None


@dataclass
class LeaderboardState:
    score_column_hash: str
    captured_utc: str
    n_entries: int
    top_3: list[dict]
    upstream_repo: str = UPSTREAM_REPO


def _gh_fetch_readme(repo: str = UPSTREAM_REPO) -> str:
    """Return the decoded README content via `gh api`. Raises on any failure."""
    proc = subprocess.run(  # nosec — gh CLI invocation, no shell
        ["gh", "api", f"repos/{repo}/readme"],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"`gh api repos/{repo}/readme` failed rc={proc.returncode}: "
            f"{(proc.stderr or '').strip()[:500]}"
        )
    payload = json.loads(proc.stdout)
    if "content" not in payload:
        raise RuntimeError(f"unexpected gh api payload (no 'content' key): keys={list(payload)[:10]}")
    return base64.b64decode(payload["content"]).decode("utf-8", errors="replace")


def extract_leaderboard_block(readme: str) -> str:
    """Slice out the canonical TABLE-START..TABLE-END region. Raises if absent."""
    start_idx = readme.find(TABLE_START)
    end_idx = readme.find(TABLE_END)
    if start_idx < 0 or end_idx < 0 or end_idx <= start_idx:
        raise ValueError(
            f"leaderboard markers missing: TABLE-START={start_idx} TABLE-END={end_idx}"
        )
    # include both markers so the block is itself self-delimiting
    return readme[start_idx : end_idx + len(TABLE_END)]


_ROW_PATTERN = re.compile(r"<tr>(.*?)</tr>", re.DOTALL)
_CELL_PATTERN = re.compile(r"<td>\s*(.*?)\s*</td>", re.DOTALL)
_PR_HREF_PATTERN = re.compile(
    r'href="https://github\.com/[^/]+/[^/]+/pull/(\d+)"', re.IGNORECASE
)
_NUMERIC_PATTERN = re.compile(r"^-?\d+(\.\d+)?$")


def parse_leaderboard_entries(block: str) -> list[LeaderboardEntry]:
    """Parse rows from the TABLE-START block.

    Schema observed (2026-05-05): each <tr> has 4 cells:
      [0] rank (often empty)
      [1] score (numeric, e.g. "0.193")
      [2] name (e.g. "hnerv_ft_microcodec  👑")
      [3] link (anchor with PR href)
    """
    entries: list[LeaderboardEntry] = []
    rank = 0
    for row_html in _ROW_PATTERN.findall(block):
        cells = _CELL_PATTERN.findall(row_html)
        if len(cells) < 4:
            continue
        score_raw = re.sub(r"\s+", "", cells[1])
        if not _NUMERIC_PATTERN.match(score_raw):
            continue  # header row or malformed
        try:
            score = float(score_raw)
        except ValueError:
            continue
        name = re.sub(r"\s+", " ", _strip_html(cells[2])).strip()
        pr_match = _PR_HREF_PATTERN.search(cells[3])
        pr_number = int(pr_match.group(1)) if pr_match else None
        pr_url = (
            f"https://github.com/{UPSTREAM_REPO}/pull/{pr_number}"
            if pr_number is not None
            else None
        )
        rank += 1
        entries.append(
            LeaderboardEntry(
                rank=rank, score=score, name=name, pr_url=pr_url, pr_number=pr_number
            )
        )
    return entries


def _strip_html(s: str) -> str:
    return re.sub(r"<[^>]+>", "", s)


def hash_score_column(entries: list[LeaderboardEntry]) -> str:
    """Hash ONLY the ordered score column. Cosmetic edits do not flip the hash."""
    canonical = "\n".join(f"{e.rank}:{e.score:.6f}" for e in entries)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_state(path: Path = STATE_PATH) -> LeaderboardState | None:
    if not path.is_file():
        return None
    raw = json.loads(path.read_text())
    return LeaderboardState(**raw)


def save_state(state: LeaderboardState, path: Path = STATE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(state), indent=2, sort_keys=True))


def append_change(prev: LeaderboardState | None, curr: LeaderboardState,
                  jsonl_path: Path = CHANGES_JSONL_PATH) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "detected_utc": curr.captured_utc,
        "prev_hash": prev.score_column_hash if prev else None,
        "curr_hash": curr.score_column_hash,
        "prev_top_3": prev.top_3 if prev else [],
        "curr_top_3": curr.top_3,
        "prev_n_entries": prev.n_entries if prev else 0,
        "curr_n_entries": curr.n_entries,
    }
    with jsonl_path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def touch_race_flag(path: Path = RACE_FLAG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch(exist_ok=True)
    now = time.time()
    import os
    os.utime(path, (now, now))


def build_state_from_readme(readme: str) -> LeaderboardState:
    block = extract_leaderboard_block(readme)
    entries = parse_leaderboard_entries(block)
    if not entries:
        raise ValueError("parsed zero leaderboard entries — extraction regex drifted")
    return LeaderboardState(
        score_column_hash=hash_score_column(entries),
        captured_utc=datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        n_entries=len(entries),
        top_3=[
            {"rank": e.rank, "score": e.score, "name": e.name, "pr_url": e.pr_url}
            for e in entries[:3]
        ],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cron", action="store_true",
                        help="silent on no-change, suitable for cron")
    parser.add_argument("--initialize-baseline", action="store_true",
                        help="capture state without alerting (first run)")
    parser.add_argument("--state-path", type=Path, default=STATE_PATH)
    parser.add_argument("--race-flag-path", type=Path, default=RACE_FLAG_PATH)
    parser.add_argument("--changes-jsonl-path", type=Path, default=CHANGES_JSONL_PATH)
    parser.add_argument("--repo", default=UPSTREAM_REPO,
                        help=f"upstream repo (default: {UPSTREAM_REPO})")
    args = parser.parse_args(argv)

    try:
        readme = _gh_fetch_readme(args.repo)
    except (RuntimeError, json.JSONDecodeError) as exc:
        print(f"FATAL: README fetch failed: {exc}", file=sys.stderr)
        return 2

    try:
        curr = build_state_from_readme(readme)
    except ValueError as exc:
        print(f"FATAL: leaderboard extraction failed: {exc}", file=sys.stderr)
        return 3

    prev = load_state(args.state_path)

    if args.initialize_baseline:
        save_state(curr, args.state_path)
        if not args.cron:
            print(f"[leaderboard-poll] BASELINE captured @ {curr.captured_utc}")
            print(f"[leaderboard-poll]   hash    : {curr.score_column_hash[:16]}…")
            print(f"[leaderboard-poll]   entries : {curr.n_entries}")
            for e in curr.top_3:
                print(f"     #{e['rank']}  {e['score']:.4f}  {e['name']}")
        return 0

    unchanged = prev is not None and prev.score_column_hash == curr.score_column_hash

    if unchanged:
        # update timestamp but not the canonical state file? we keep the previous
        # captured_utc to reflect "unchanged since"; do not overwrite.
        if not args.cron:
            print(f"[leaderboard-poll] unchanged since {prev.captured_utc} (hash {prev.score_column_hash[:16]}…)")
        return 0

    # Hash differs — race mode triggered
    save_state(curr, args.state_path)
    append_change(prev, curr, args.changes_jsonl_path)
    touch_race_flag(args.race_flag_path)
    print("RACE_MODE_DETECTED=1")
    if not args.cron:
        print(f"[leaderboard-poll] CHANGE DETECTED @ {curr.captured_utc}")
        print(f"[leaderboard-poll]   prev_hash : "
              f"{(prev.score_column_hash[:16] + '…') if prev else '<none>'}")
        print(f"[leaderboard-poll]   curr_hash : {curr.score_column_hash[:16]}…")
        print("[leaderboard-poll]   top-3 now :")
        for e in curr.top_3:
            print(f"     #{e['rank']}  {e['score']:.4f}  {e['name']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
