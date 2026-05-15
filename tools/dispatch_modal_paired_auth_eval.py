#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Dispatch paired Modal CPU+CUDA auth evals for one archive/runtime.

Plan-only by default. Pass ``--execute`` to spawn both detached Modal jobs.
The individual Modal wrappers still own lane claims and artifact recovery; this
tool is the canonical operator entry point so CPU/CUDA pairing is the default
instead of an afterthought.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]


def _utc_now_compact() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _safe_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._-") or "modal_pair"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


def _optional_arg(cmd: list[str], flag: str, value: str) -> None:
    if value:
        cmd.extend([flag, value])


def build_plan(
    *,
    archive: Path,
    submission_dir: str,
    inflate_sh: str,
    run_id: str,
    pair_group_id: str,
    lane_id_base: str,
    output_root: Path,
    modal_bin: str,
    gpu: str,
    claim_agent: str,
    claim_notes: str,
    expected_runtime_tree_sha256: str = "",
) -> dict[str, Any]:
    archive = archive.resolve()
    if not archive.is_file():
        raise FileNotFoundError(f"archive not found: {archive}")
    archive_sha = _sha256(archive)
    archive_bytes = archive.stat().st_size
    notes = (
        claim_notes
        or (
            "paired Modal auth eval; same archive/runtime required on "
            "contest_cuda and contest_cpu axes"
        )
    )
    cuda_output = output_root / "modal_auth_eval" / f"{run_id}_cuda"
    cpu_output = output_root / "modal_auth_eval_cpu" / f"{run_id}_cpu"
    cuda_lane = f"{lane_id_base}_contest_cuda"
    cpu_lane = f"{lane_id_base}_contest_cpu"

    cuda_cmd = [
        modal_bin,
        "run",
        "--detach",
        "experiments/modal_auth_eval.py",
        "--archive",
        str(archive),
        "--inflate-sh",
        inflate_sh,
        "--output-dir",
        str(cuda_output),
        "--gpu",
        gpu,
        "--detach",
        "--provider-detach-ack",
        "--pair-group-id",
        pair_group_id,
        "--lane-id",
        cuda_lane,
        "--instance-job-id",
        f"{run_id}_cuda",
        "--claim-agent",
        claim_agent,
        "--claim-notes",
        f"{notes}; pair_group_id={pair_group_id}; axis=contest_cuda; archive_sha={archive_sha}; bytes={archive_bytes}",
    ]
    cpu_cmd = [
        modal_bin,
        "run",
        "--detach",
        "experiments/modal_auth_eval_cpu.py",
        "--archive",
        str(archive),
        "--inflate-sh",
        inflate_sh,
        "--output-dir",
        str(cpu_output),
        "--detach",
        "--provider-detach-ack",
        "--pair-group-id",
        pair_group_id,
        "--lane-id",
        cpu_lane,
        "--instance-job-id",
        f"{run_id}_cpu",
        "--claim-agent",
        claim_agent,
        "--claim-notes",
        f"{notes}; pair_group_id={pair_group_id}; axis=contest_cpu; archive_sha={archive_sha}; bytes={archive_bytes}",
    ]
    _optional_arg(cuda_cmd, "--submission-dir", submission_dir)
    _optional_arg(cpu_cmd, "--submission-dir", submission_dir)
    _optional_arg(cuda_cmd, "--expected-runtime-tree-sha256", expected_runtime_tree_sha256)
    _optional_arg(cpu_cmd, "--expected-runtime-tree-sha256", expected_runtime_tree_sha256)
    return {
        "schema": "modal_paired_auth_eval_dispatch_plan_v1",
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "pair_group_id": pair_group_id,
        "required_axes": ["contest_cuda", "contest_cpu"],
        "archive": {
            "path": str(archive),
            "bytes": archive_bytes,
            "sha256": archive_sha,
        },
        "runtime": {
            "submission_dir": submission_dir or None,
            "inflate_sh": inflate_sh,
            "expected_runtime_tree_sha256": expected_runtime_tree_sha256 or None,
        },
        "outputs": {
            "contest_cuda": str(cuda_output),
            "contest_cpu": str(cpu_output),
        },
        "lanes": {
            "contest_cuda": cuda_lane,
            "contest_cpu": cpu_lane,
        },
        "commands": {
            "contest_cuda": cuda_cmd,
            "contest_cpu": cpu_cmd,
        },
        "score_claim": False,
        "promotion_eligible": False,
        "notes": [
            "This tool is the default Modal auth-eval entry point for score-bearing archives.",
            "Both commands carry the same pair_group_id and exact archive SHA.",
            "Single-axis Modal wrapper use requires an explicit waiver reason.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--submission-dir", default="")
    parser.add_argument("--inflate-sh", default="submissions/robust_current/inflate.sh")
    parser.add_argument("--label", default="")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--pair-group-id", default="")
    parser.add_argument("--lane-id-base", default="")
    parser.add_argument("--output-root", type=Path, default=Path("experiments/results"))
    parser.add_argument("--modal-bin", default=".venv/bin/modal")
    parser.add_argument("--gpu", default="T4")
    parser.add_argument("--claim-agent", default="codex:modal_paired_auth_eval")
    parser.add_argument("--claim-notes", default="")
    parser.add_argument("--expected-runtime-tree-sha256", default="")
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    label = _safe_slug(args.label or args.archive.stem)
    run_id = _safe_slug(args.run_id or f"{label}_paired_modal_auth_{_utc_now_compact()}")
    pair_group_id = _safe_slug(args.pair_group_id or run_id)
    lane_id_base = _safe_slug(args.lane_id_base or f"lane_{pair_group_id}")
    plan = build_plan(
        archive=args.archive,
        submission_dir=args.submission_dir,
        inflate_sh=args.inflate_sh,
        run_id=run_id,
        pair_group_id=pair_group_id,
        lane_id_base=lane_id_base,
        output_root=args.output_root,
        modal_bin=args.modal_bin,
        gpu=args.gpu,
        claim_agent=args.claim_agent,
        claim_notes=args.claim_notes,
        expected_runtime_tree_sha256=args.expected_runtime_tree_sha256,
    )
    text = json.dumps(plan, indent=2, sort_keys=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    print(text)
    if not args.execute:
        return 0
    for axis in ("contest_cuda", "contest_cpu"):
        proc = subprocess.run(plan["commands"][axis], cwd=REPO_ROOT)
        if proc.returncode:
            return proc.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
