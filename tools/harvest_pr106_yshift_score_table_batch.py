#!/usr/bin/env python3
"""Harvest and validate a PR106 yshift score-table Lightning Batch run."""
from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.repo_io import json_text, write_json  # noqa: E402

DEFAULT_STATE_PATH = REPO_ROOT / ".omx/state/lightning_batch_jobs.json"
CLAIMS_PATH = REPO_ROOT / ".omx/state/active_lane_dispatch_claims.md"
VALIDATION_NAME = "pr106_yshift_batch_harvest_validation.json"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _latest_record(state_path: Path, job_name: str) -> dict[str, Any]:
    records = _load_json(state_path)
    if not isinstance(records, list):
        raise ValueError(f"Lightning state must be a JSON array: {state_path}")
    for record in reversed(records):
        spec = record.get("spec") if isinstance(record, dict) and isinstance(record.get("spec"), dict) else {}
        queue = record.get("queue") if isinstance(record, dict) and isinstance(record.get("queue"), dict) else {}
        job = record.get("job") if isinstance(record, dict) and isinstance(record.get("job"), dict) else {}
        if job_name in {spec.get("name"), queue.get("job_name"), job.get("name")}:
            return record
    raise KeyError(f"job not found in {state_path}: {job_name}")


def _remote_and_local(args: argparse.Namespace) -> tuple[str, Path]:
    record = _latest_record(args.state_path, args.job_name)
    spec = record.get("spec") if isinstance(record.get("spec"), dict) else {}
    queue = record.get("queue") if isinstance(record.get("queue"), dict) else {}
    remote = args.remote_artifact_dir or spec.get("remote_output_dir") or queue.get("remote_output_dir")
    local = args.mirror_dir or spec.get("local_artifact_dir") or queue.get("local_artifact_dir")
    if not isinstance(remote, str) or not remote:
        raise ValueError("remote artifact dir not recorded; pass --remote-artifact-dir")
    if not isinstance(local, str) or not local:
        raise ValueError("local mirror dir not recorded; pass --mirror-dir")
    return remote, REPO_ROOT / local


def _ssh_options(timeout: int) -> list[str]:
    return [
        "-o",
        "BatchMode=yes",
        "-o",
        "PasswordAuthentication=no",
        "-o",
        "KbdInteractiveAuthentication=no",
        "-o",
        "ServerAliveInterval=15",
        "-o",
        "ServerAliveCountMax=4",
        "-o",
        "TCPKeepAlive=yes",
        "-o",
        "ConnectionAttempts=3",
        "-o",
        f"ConnectTimeout={timeout}",
    ]


def harvest_remote(args: argparse.Namespace, *, remote_dir: str, mirror_dir: Path) -> None:
    if mirror_dir.exists() and not args.overwrite:
        raise SystemExit(f"FATAL: mirror dir exists; pass --overwrite: {mirror_dir}")
    mirror_dir.mkdir(parents=True, exist_ok=True)
    ssh = ["ssh", *_ssh_options(args.ssh_connect_timeout), args.ssh_target]
    probe = subprocess.run(
        [*ssh, "test -d " + shlex.quote(remote_dir)],
        cwd=REPO_ROOT,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        raise SystemExit(f"ARTIFACT_NOT_READY: remote dir missing: {remote_dir}")
    rsync = [
        "rsync",
        "-a",
        "-e",
        " ".join(shlex.quote(part) for part in ["ssh", *_ssh_options(args.ssh_connect_timeout)]),
        f"{args.ssh_target}:{remote_dir.rstrip('/')}/",
        str(mirror_dir) + "/",
    ]
    subprocess.run(rsync, cwd=REPO_ROOT, check=True)


def validate_mirror(mirror_dir: Path) -> dict[str, Any]:
    runner_preflight_path = mirror_dir / "lightning_runner_preflight.json"
    summary_path = mirror_dir / "pr106_yshift_score_table_batch_summary.json"
    score_json_path = mirror_dir / "contest_auth_eval.json"
    run_log_path = mirror_dir / "batch_run.log"
    yshift_dir = mirror_dir / "yshift_run"
    built_archive = yshift_dir / "build/pr106_yshift_sidechannel_archive.zip"
    score_table_manifest = yshift_dir / "score_table/score_table_manifest.json"
    required = [
        runner_preflight_path,
        summary_path,
        score_json_path,
        run_log_path,
        built_archive,
        score_table_manifest,
    ]
    missing = [str(path.relative_to(mirror_dir)) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing PR106 yshift batch artifacts: {missing}")

    runner = _load_json(runner_preflight_path)
    if runner.get("LIGHTNING_RUNNER_CUDA_PREFLIGHT_OK") is not True:
        raise ValueError("runner preflight did not record CUDA OK")
    summary = _load_json(summary_path)
    if summary.get("contest_auth_eval_json_exists") is not True:
        raise ValueError("batch summary did not observe contest_auth_eval.json")
    score = _load_json(score_json_path)
    final_score = score.get("final_score", score.get("score_recomputed_from_components"))
    if not isinstance(final_score, int | float) or not math.isfinite(float(final_score)):
        raise ValueError("contest_auth_eval.json missing finite final score")
    score_table = _load_json(score_table_manifest)
    if score_table.get("score_claim") is not False:
        raise ValueError("score-table manifest must not claim a score")

    validation = {
        "schema_version": 1,
        "status": "validated",
        "evidence_grade": "A_pending_adjudication",
        "score_claim": False,
        "final_score": float(final_score),
        "runner_gpu_names": runner.get("gpu_names"),
        "torch_version": runner.get("torch_version"),
        "torch_cuda": runner.get("torch_cuda"),
        "built_archive_bytes": built_archive.stat().st_size,
        "mirror_dir": str(mirror_dir),
        "contest_auth_eval_json": str(score_json_path),
        "score_table_manifest": str(score_table_manifest),
        "promotion_gate": "requires adjudication and custody review before score claim",
    }
    write_json(mirror_dir / VALIDATION_NAME, validation)
    return validation


def close_claim(args: argparse.Namespace, *, validation: dict[str, Any] | None, failed: bool) -> None:
    if not args.close_claim:
        return
    if failed:
        status = "failed_harvest_validation"
        notes = "PR106 yshift score-table Batch harvest/validation failed"
    else:
        status = f"completed_score={validation['final_score']:.12g}"
        notes = "PR106 yshift score-table Batch produced exact CUDA contest_auth_eval.json; adjudication still required"
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools/claim_lane_dispatch.py"),
        "claim",
        "--lane-id",
        "lane_pr106_yshift_score_table",
        "--platform",
        "lightning",
        "--instance-job-id",
        args.job_name,
        "--agent",
        args.agent,
        "--status",
        status,
        "--notes",
        notes,
        "--force",
    ]
    subprocess.run(cmd, cwd=REPO_ROOT, check=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-name", required=True)
    parser.add_argument("--ssh-target", default="")
    parser.add_argument("--state-path", type=Path, default=DEFAULT_STATE_PATH)
    parser.add_argument("--remote-artifact-dir", default=None)
    parser.add_argument("--mirror-dir", default=None)
    parser.add_argument("--ssh-connect-timeout", type=int, default=30)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-copy", action="store_true", help="Validate the local mirror without SSH copy.")
    parser.add_argument("--close-claim", action="store_true")
    parser.add_argument("--agent", default="codex:gpt-5.5")
    args = parser.parse_args(argv)

    remote_dir, mirror_dir = _remote_and_local(args)
    validation = None
    try:
        if not args.no_copy:
            if not args.ssh_target:
                raise SystemExit("FATAL: --ssh-target required unless --no-copy")
            harvest_remote(args, remote_dir=remote_dir, mirror_dir=mirror_dir)
        validation = validate_mirror(mirror_dir)
    except Exception:
        close_claim(args, validation=None, failed=True)
        raise
    close_claim(args, validation=validation, failed=False)
    print(json_text(validation), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
