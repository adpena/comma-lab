#!/usr/bin/env python3
"""Plan paired CPU and CUDA auth evals for the same archive/runtime.

CUDA remains the internal promotion/ranking gate. CPU replay is a separate
public-leaderboard reproduction axis. This planner keeps both commands tied to
the same archive bytes so CUDA-vs-CPU drift is measured rather than inferred.
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
    return slug.strip("._-") or "auth_eval"


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _load_ledger_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("rows"), list):
        return data["rows"]
    if isinstance(data, list):
        return data
    raise ValueError(f"unsupported ledger shape: {path}")


def _public_pr_inputs(ledger: Path, pr: int) -> tuple[Path, Path, str, dict[str, Any]]:
    for row in _load_ledger_rows(ledger):
        if row.get("pr") != pr:
            continue
        archive_path = (row.get("archive") or {}).get("path")
        inflate_path = (((row.get("source") or {}).get("key_files") or {}).get("inflate_sh") or {}).get("path")
        if not archive_path:
            raise ValueError(f"PR {pr} row has no archive path")
        if not inflate_path:
            raise ValueError(f"PR {pr} row has no inflate.sh path")
        label = f"public-pr{pr}-{row.get('leaderboard_name') or row.get('title') or 'submission'}"
        return Path(archive_path), Path(inflate_path), label, row
    raise KeyError(f"PR {pr} not found in {ledger}")


def _command(
    *,
    archive: Path,
    inflate_sh: Path,
    upstream_dir: Path,
    video_names_file: Path,
    device: str,
    work_dir: Path,
    inflate_timeout: int,
    evaluate_timeout: int,
) -> list[str]:
    return [
        ".venv/bin/python",
        "experiments/contest_auth_eval.py",
        "--archive",
        str(archive),
        "--inflate-sh",
        str(inflate_sh),
        "--upstream-dir",
        str(upstream_dir),
        "--video-names-file",
        str(video_names_file),
        "--device",
        device,
        "--work-dir",
        str(work_dir),
        "--inflate-timeout",
        str(inflate_timeout),
        "--evaluate-timeout",
        str(evaluate_timeout),
        "--keep-work-dir",
    ]


def build_plan(
    *,
    archive: Path,
    inflate_sh: Path,
    label: str,
    repo_root: Path,
    run_id: str | None,
    output_root: Path,
    upstream_dir: Path,
    video_names_file: Path,
    inflate_timeout: int,
    evaluate_timeout: int,
    public_pr_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_id = run_id or f"{_safe_slug(label)}-dual-auth-{_utc_now_compact()}"
    work_root = output_root / run_id
    archive_meta: dict[str, Any] = {"path": str(archive)}
    if archive.exists():
        archive_meta.update({"bytes": archive.stat().st_size, "sha256": _sha256(archive)})

    evals: dict[str, dict[str, Any]] = {}
    for device in ("cuda", "cpu"):
        semantics = (
            "contest_cuda_exact_auth_eval_promotion_axis"
            if device == "cuda"
            else "public_leaderboard_cpu_reproduction_axis"
        )
        evals[device] = {
            "device": device,
            "work_dir": str(work_root / device),
            "command": _command(
                archive=archive,
                inflate_sh=inflate_sh,
                upstream_dir=upstream_dir,
                video_names_file=video_names_file,
                device=device,
                work_dir=work_root / device,
                inflate_timeout=inflate_timeout,
                evaluate_timeout=evaluate_timeout,
            ),
            "evidence_semantics": semantics,
            "promotion_eligible_from_this_axis": device == "cuda",
        }
    return {
        "schema": "dual_device_auth_eval_plan.v1",
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "repo_root": str(repo_root),
        "run_id": run_id,
        "label": label,
        "archive": archive_meta,
        "inflate_sh": str(inflate_sh),
        "upstream_dir": str(upstream_dir),
        "video_names_file": str(video_names_file),
        "evals": evals,
        "public_pr_row": public_pr_row,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "notes": [
            "Run both commands on the same archive bytes before PR/frontier claims.",
            "CUDA is the internal promotion/ranking axis; CPU is the public leaderboard reproduction axis.",
            "Do not extrapolate CPU from CUDA or CUDA from CPU; compare paired JSON artifacts only.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--public-pr", type=int, help="Resolve archive/runtime from the PR100-107 ledger.")
    source.add_argument("--archive", type=Path, help="Archive path for an explicit submission.")
    parser.add_argument("--inflate-sh", type=Path, help="inflate.sh for --archive mode.")
    parser.add_argument("--label", help="Human-readable run label.")
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--run-id")
    parser.add_argument("--output-root", type=Path, default=Path("experiments/results/dual_device_auth_eval"))
    parser.add_argument("--upstream-dir", type=Path, default=Path("upstream"))
    parser.add_argument("--video-names-file", type=Path, default=Path("upstream/public_test_video_names.txt"))
    parser.add_argument("--inflate-timeout", type=int, default=1800)
    parser.add_argument("--evaluate-timeout", type=int, default=1800)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--execute", choices=("cpu", "cuda", "both"), help="Run planned local command(s).")
    args = parser.parse_args()

    public_pr_row = None
    if args.public_pr is not None:
        archive, inflate_sh, label, public_pr_row = _public_pr_inputs(args.ledger, args.public_pr)
    else:
        if args.inflate_sh is None:
            raise SystemExit("--inflate-sh is required with --archive")
        archive = args.archive
        inflate_sh = args.inflate_sh
        label = args.label or archive.stem

    plan = build_plan(
        archive=archive,
        inflate_sh=inflate_sh,
        label=label,
        repo_root=args.repo_root,
        run_id=args.run_id,
        output_root=args.output_root,
        upstream_dir=args.upstream_dir,
        video_names_file=args.video_names_file,
        inflate_timeout=args.inflate_timeout,
        evaluate_timeout=args.evaluate_timeout,
        public_pr_row=public_pr_row,
    )
    text = json.dumps(plan, indent=2, sort_keys=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    print(text)

    if args.execute:
        devices = ("cpu", "cuda") if args.execute == "both" else (args.execute,)
        for device in devices:
            result = subprocess.run(plan["evals"][device]["command"], cwd=args.repo_root)
            if result.returncode:
                return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
