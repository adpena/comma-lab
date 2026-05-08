#!/usr/bin/env python3
"""Plan or run public-PR CPU auth eval replay from the reproduction ledger.

This is intentionally separate from CUDA promotion. CPU replay is useful for
reproducing public leaderboard/comment rows, but it must not be mistaken for a
CUDA A++ promotion artifact inside this repo.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_LEDGER = Path("experiments/results/pr100_107_reproduction_ledger_20260507_codex/ledger.json")


def _utc_now_compact() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _safe_slug(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return slug.strip("._-") or "public_pr"


def load_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("rows"), list):
        return data["rows"]
    if isinstance(data, list):
        return data
    raise ValueError(f"unsupported ledger shape: {path}")


def _input_closure(paths: dict[str, Path]) -> dict[str, Any]:
    entries = {}
    missing = []
    for name, path in paths.items():
        exists = path.exists()
        entries[name] = {
            "path": str(path),
            "exists": exists,
            "is_file": path.is_file() if exists else False,
        }
        if not exists:
            missing.append(name)
    return {
        "required_inputs": entries,
        "missing_inputs": missing,
        "ready_to_execute": not missing,
    }


def find_pr_row(rows: list[dict[str, Any]], pr: int) -> dict[str, Any]:
    for row in rows:
        if row.get("pr") == pr:
            return row
    raise KeyError(f"PR {pr} not found in ledger")


def build_plan(
    *,
    row: dict[str, Any],
    repo_root: Path,
    run_id: str | None,
    work_dir: Path | None,
    upstream_dir: Path,
    video_names_file: Path,
    inflate_timeout: int,
    evaluate_timeout: int,
) -> dict[str, Any]:
    pr = int(row["pr"])
    name = row.get("leaderboard_name") or row.get("title") or f"pr{pr}"
    archive = row.get("archive") or {}
    archive_path = archive.get("path")
    inflate = (((row.get("source") or {}).get("key_files") or {}).get("inflate_sh") or {})
    inflate_path = inflate.get("path")
    if not archive_path:
        raise ValueError(f"PR {pr} row has no archive path")
    if not inflate_path:
        raise ValueError(f"PR {pr} row has no inflate.sh path")
    archive_path_obj = Path(archive_path)
    inflate_path_obj = Path(inflate_path)

    run_id = run_id or f"public-pr{pr}-{_safe_slug(str(name))}-cpu-auth-{_utc_now_compact()}"
    work_dir = work_dir or Path("experiments/results/public_cpu_auth_eval") / run_id
    command = [
        ".venv/bin/python",
        "experiments/contest_auth_eval.py",
        "--archive",
        str(archive_path),
        "--inflate-sh",
        str(inflate_path),
        "--upstream-dir",
        str(upstream_dir),
        "--video-names-file",
        str(video_names_file),
        "--device",
        "cpu",
        "--work-dir",
        str(work_dir),
        "--inflate-timeout",
        str(inflate_timeout),
        "--evaluate-timeout",
        str(evaluate_timeout),
        "--keep-work-dir",
    ]
    return {
        "schema": "public_pr_cpu_auth_eval_plan.v1",
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "repo_root": str(repo_root),
        "run_id": run_id,
        "pr": pr,
        "leaderboard_name": name,
        "archive": {
            "path": archive_path,
            "bytes": archive.get("bytes"),
            "sha256": archive.get("sha256"),
        },
        "inflate_sh": inflate_path,
        "upstream_dir": str(upstream_dir),
        "video_names_file": str(video_names_file),
        "input_closure": _input_closure(
            {
                "archive": archive_path_obj,
                "inflate_sh": inflate_path_obj,
                "upstream_dir": upstream_dir,
                "video_names_file": video_names_file,
            }
        ),
        "work_dir": str(work_dir),
        "device": "cpu",
        "evidence_semantics": "public_cpu_leaderboard_reproduction_not_cuda_promotion",
        "evidence_grade": "cpu_public_replay",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "command": command,
        "notes": [
            "CPU replay is for reproducing public PR/leaderboard comment rows.",
            "Do not use CPU replay to promote, rank, or kill internal lanes under the CUDA evidence gate.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pr", type=int, required=True)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--run-id")
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--upstream-dir", type=Path, default=Path("upstream"))
    parser.add_argument(
        "--video-names-file",
        type=Path,
        default=Path("upstream/public_test_video_names.txt"),
    )
    parser.add_argument("--inflate-timeout", type=int, default=1800)
    parser.add_argument("--evaluate-timeout", type=int, default=1800)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    rows = load_rows(args.ledger)
    plan = build_plan(
        row=find_pr_row(rows, args.pr),
        repo_root=args.repo_root,
        run_id=args.run_id,
        work_dir=args.work_dir,
        upstream_dir=args.upstream_dir,
        video_names_file=args.video_names_file,
        inflate_timeout=args.inflate_timeout,
        evaluate_timeout=args.evaluate_timeout,
    )
    text = json.dumps(plan, indent=2, sort_keys=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    print(text)

    if args.execute:
        if not plan["input_closure"]["ready_to_execute"]:
            missing = ", ".join(plan["input_closure"]["missing_inputs"])
            raise SystemExit(f"refusing to execute public CPU auth eval plan with missing inputs: {missing}")
        return subprocess.run(plan["command"], cwd=args.repo_root).returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
