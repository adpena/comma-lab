"""Modal actuator for T1 Ballé hyperprior end-to-end.

This file graduates T1 from a remote-command plan to a real Modal job path
without weakening custody:

* default path is local plan-only;
* real dispatch requires ``--execute`` through ``modal run``;
* lane claim is opened before ``.spawn()``;
* the Modal worker runs the existing T1 remote script with a copied claim
  ledger, so the remote script can fail closed on missing dispatch custody;
* recover closes the local claim and only treats the result as contest-CUDA
  evidence when ``auth_eval_schema`` reports zero blockers for 600 samples.

Usage:

    .venv/bin/python experiments/modal_t1_balle_endtoend.py plan \
        --json-out experiments/results/t1_modal_plan.json

    PYTHONPATH=src:upstream:$PWD .venv/bin/modal run --detach \
        experiments/modal_t1_balle_endtoend.py --execute \
        --label t1_balle_modal_20260510T000000Z

    .venv/bin/python experiments/modal_t1_balle_endtoend.py recover \
        --label t1_balle_modal_20260510T000000Z
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import modal

REPO_ROOT = Path(__file__).resolve().parent.parent
REMOTE_REPO = Path("/workspace/pact")
RESULT_ROOT = REPO_ROOT / "experiments" / "results"
REMOTE_OUT_ROOT = REMOTE_REPO / "experiments/results/modal_t1_balle_remote"
APP_NAME = "comma-t1-balle-endtoend"
LANE_ID = "t1_balle_128k_endtoend"
CLAIM_AGENT = "codex:modal_t1_balle_endtoend"
MODAL_SPAWN_SUBMISSION_UNKNOWN_STATUS = "ambiguous_modal_spawn_submission_recovery_required"
HOURLY_RATE_T4_USD = 0.59
DEFAULT_TIMEOUT_HOURS = 24.0
DEFAULT_TRAIN_TIMEOUT_HOURS = 22.5
REMOTE_POST_TRAIN_EVAL_BUFFER_HOURS = 1.0
REMOTE_ARTIFACT_COLLECTION_BUFFER_HOURS = 0.25
DEFAULT_COST_CAP_USD = 80.0
DEFAULT_SINKHORN_MAX_POSITIONS_PER_CHUNK = 2048
REMOTE_PYTHONPATH = f"{REMOTE_REPO / 'src'}:{REMOTE_REPO / 'upstream'}:{REMOTE_REPO}"
MOUNTED_CODE_PATHS = (
    "src",
    "pyproject.toml",
    "uv.lock",
    "upstream/evaluate.py",
    "upstream/frame_utils.py",
    "upstream/modules.py",
    "upstream/public_test_video_names.txt",
    "upstream/pyproject.toml",
    "upstream/uv.lock",
    "experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py",
    "experiments/contest_auth_eval.py",
    "tools/build_phase1_packet_compiler.py",
    "tools/tool_bootstrap.py",
    "tools/claim_lane_dispatch.py",
    "scripts/remote_lane_t1_balle_endtoend.sh",
    "scripts/remote_archive_only_eval.sh",
    "scripts/probe_nvdec.sh",
)
A1_CANONICAL_LOCAL_PATH = REPO_ROOT / "experiments/results/A1_canonical"
A1_DESIGNATION_LOCAL_PATH = REPO_ROOT / ".omx/state/canonical_a1_designation.md"
A1_CANONICAL_REMOTE_PATH = REMOTE_REPO / "experiments/results/A1_canonical"
A1_DESIGNATION_REMOTE_PATH = REMOTE_REPO / ".omx/state/canonical_a1_designation.md"


def _ensure_repo_import_paths() -> None:
    for path in (
        REPO_ROOT / "src",
        REPO_ROOT / "upstream",
        REPO_ROOT,
        REMOTE_REPO / "src",
        REMOTE_REPO / "upstream",
        REMOTE_REPO,
    ):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


_ensure_repo_import_paths()

from tac.deploy.modal.runtime import (  # noqa: E402
    CONTEST_SCORER_IMPORT_PROBE_MODULES,
    DALI_DISABLE_NVML_VALUE,
    PYTORCH_CUDA_ALLOC_CONF_VALUE,
    build_contest_cuda_base_image,
)


app = modal.App(APP_NAME)

base_image = build_contest_cuda_base_image(
    modal,
    python_version="3.11",
    extra_pip_packages=("compressai==1.2.8",),
)

run_image = (
    base_image.env(
        {
            "PYTHONPATH": REMOTE_PYTHONPATH,
            "DALI_DISABLE_NVML": DALI_DISABLE_NVML_VALUE,
            "PYTORCH_CUDA_ALLOC_CONF": PYTORCH_CUDA_ALLOC_CONF_VALUE,
        }
    )
    .add_local_dir("src", remote_path=str(REMOTE_REPO / "src"))
    .add_local_file("pyproject.toml", remote_path=str(REMOTE_REPO / "pyproject.toml"))
    .add_local_file("uv.lock", remote_path=str(REMOTE_REPO / "uv.lock"))
    .add_local_dir("upstream/models", remote_path=str(REMOTE_REPO / "upstream/models"))
    .add_local_dir("upstream/videos", remote_path=str(REMOTE_REPO / "upstream/videos"))
    .add_local_file("upstream/evaluate.py", remote_path=str(REMOTE_REPO / "upstream/evaluate.py"))
    .add_local_file("upstream/frame_utils.py", remote_path=str(REMOTE_REPO / "upstream/frame_utils.py"))
    .add_local_file("upstream/modules.py", remote_path=str(REMOTE_REPO / "upstream/modules.py"))
    .add_local_file(
        "upstream/public_test_video_names.txt",
        remote_path=str(REMOTE_REPO / "upstream/public_test_video_names.txt"),
    )
    .add_local_file("upstream/pyproject.toml", remote_path=str(REMOTE_REPO / "upstream/pyproject.toml"))
    .add_local_file("upstream/uv.lock", remote_path=str(REMOTE_REPO / "upstream/uv.lock"))
    .add_local_file("experiments/__init__.py", remote_path=str(REMOTE_REPO / "experiments/__init__.py"))
    .add_local_file(
        "experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py",
        remote_path=str(REMOTE_REPO / "experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py"),
    )
    .add_local_file(
        "experiments/contest_auth_eval.py",
        remote_path=str(REMOTE_REPO / "experiments/contest_auth_eval.py"),
    )
    .add_local_file(
        "tools/build_phase1_packet_compiler.py",
        remote_path=str(REMOTE_REPO / "tools/build_phase1_packet_compiler.py"),
    )
    .add_local_file(
        "tools/tool_bootstrap.py",
        remote_path=str(REMOTE_REPO / "tools/tool_bootstrap.py"),
    )
    .add_local_file(
        "tools/claim_lane_dispatch.py",
        remote_path=str(REMOTE_REPO / "tools/claim_lane_dispatch.py"),
    )
    .add_local_file(
        "scripts/remote_lane_t1_balle_endtoend.sh",
        remote_path=str(REMOTE_REPO / "scripts/remote_lane_t1_balle_endtoend.sh"),
    )
    .add_local_file(
        "scripts/remote_archive_only_eval.sh",
        remote_path=str(REMOTE_REPO / "scripts/remote_archive_only_eval.sh"),
    )
    .add_local_file(
        "scripts/probe_nvdec.sh",
        remote_path=str(REMOTE_REPO / "scripts/probe_nvdec.sh"),
    )
)

if A1_CANONICAL_LOCAL_PATH.exists():
    run_image = run_image.add_local_dir(
        str(A1_CANONICAL_LOCAL_PATH.resolve()),
        remote_path=str(A1_CANONICAL_REMOTE_PATH),
    )
if A1_DESIGNATION_LOCAL_PATH.is_file():
    run_image = run_image.add_local_file(
        str(A1_DESIGNATION_LOCAL_PATH),
        remote_path=str(A1_DESIGNATION_REMOTE_PATH),
    )


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(payload))  # BARE_WRITE_OK: artifact writer only; never used for .omx/state


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _first_existing(base: Path, candidates: tuple[str, ...]) -> Path | None:
    for rel in candidates:
        path = base / rel
        if path.exists():
            return path
    return None


def _canonical_a1_payload_snapshot() -> dict[str, Any]:
    """Return local canonical A1 payload readiness for Modal mounting."""
    canonical_dir = A1_CANONICAL_LOCAL_PATH
    memo_path = A1_DESIGNATION_LOCAL_PATH
    errors: list[str] = []
    files: dict[str, dict[str, Any]] = {}

    if not canonical_dir.exists():
        errors.append(f"canonical_dir_missing:{canonical_dir}")
    if not memo_path.is_file():
        errors.append(f"designation_memo_missing:{memo_path}")

    archive_path = (
        _first_existing(
            canonical_dir,
            (
                "finetuned_archive/archive.zip",
                "harvested_artifacts/finetuned_archive/archive.zip",
                "archive.zip",
            ),
        )
        if canonical_dir.exists()
        else None
    )
    checkpoint_path = (
        _first_existing(
            canonical_dir,
            (
                "train/checkpoint_best_proxy.pt",
                "harvested_artifacts/train/checkpoint_best_proxy.pt",
                "checkpoint_best_proxy.pt",
                "train/checkpoint_ema.pt",
                "harvested_artifacts/train/checkpoint_ema.pt",
            ),
        )
        if canonical_dir.exists()
        else None
    )
    latents_path = (
        _first_existing(
            canonical_dir,
            (
                "train/extracted_frozen_latents.pt",
                "harvested_artifacts/train/extracted_frozen_latents.pt",
                "harvested_artifacts/extracted_frozen_latents.pt",
                "extracted_frozen_latents.pt",
            ),
        )
        if canonical_dir.exists()
        else None
    )
    required = {
        "archive": archive_path,
        "checkpoint": checkpoint_path,
        "extracted_latents": latents_path,
        "designation_memo": memo_path if memo_path.is_file() else None,
    }
    for role, path in required.items():
        if path is None:
            errors.append(f"{role}_missing")
            continue
        rel = str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)
        files[role] = {
            "path": rel,
            "bytes": path.stat().st_size,
            "sha256": _sha256_path(path),
        }

    return {
        "schema_version": "canonical_a1_payload_snapshot_v1",
        "ready_for_modal_mount": not errors,
        "canonical_dir": str(canonical_dir),
        "canonical_dir_is_symlink": canonical_dir.is_symlink(),
        "canonical_dir_resolved": str(canonical_dir.resolve()) if canonical_dir.exists() else None,
        "remote_canonical_dir": str(A1_CANONICAL_REMOTE_PATH),
        "remote_designation_memo": str(A1_DESIGNATION_REMOTE_PATH),
        "files": files,
        "validation_errors": errors,
    }


def _run_git_text(args: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def _mounted_code_snapshot(out_dir: Path) -> dict[str, Any]:
    """Record the exact local code state Modal mounts for this score-bearing run."""
    snapshot_dir = out_dir / "code_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    mounted_paths = list(MOUNTED_CODE_PATHS)
    head_rc, head_stdout, head_stderr = _run_git_text(["rev-parse", "HEAD"])
    status_rc, status_stdout, status_stderr = _run_git_text(
        ["status", "--short", "--", *mounted_paths]
    )
    diff_rc, diff_stdout, diff_stderr = _run_git_text(["diff", "--binary", "--", *mounted_paths])
    cached_rc, cached_stdout, cached_stderr = _run_git_text(
        ["diff", "--cached", "--binary", "--", *mounted_paths]
    )
    status_path = snapshot_dir / "mounted_code_status.txt"
    diff_path = snapshot_dir / "mounted_code_worktree.patch"
    cached_diff_path = snapshot_dir / "mounted_code_index.patch"
    status_path.write_text(status_stdout)
    diff_path.write_text(diff_stdout)
    cached_diff_path.write_text(cached_stdout)
    return {
        "schema_version": "mounted_modal_code_snapshot_v1",
        "git_head": head_stdout.strip() if head_rc == 0 else None,
        "git_head_error": head_stderr.strip() if head_rc != 0 else None,
        "mounted_code_paths": mounted_paths,
        "dirty": bool(status_stdout.strip()),
        "status_short": status_stdout.splitlines(),
        "status_rc": status_rc,
        "status_error": status_stderr.strip() if status_rc != 0 else None,
        "worktree_diff_bytes": len(diff_stdout.encode("utf-8")),
        "worktree_diff_sha256": _sha256_bytes(diff_stdout.encode("utf-8")),
        "worktree_diff_path": str(diff_path),
        "worktree_diff_rc": diff_rc,
        "worktree_diff_error": diff_stderr.strip() if diff_rc != 0 else None,
        "index_diff_bytes": len(cached_stdout.encode("utf-8")),
        "index_diff_sha256": _sha256_bytes(cached_stdout.encode("utf-8")),
        "index_diff_path": str(cached_diff_path),
        "index_diff_rc": cached_rc,
        "index_diff_error": cached_stderr.strip() if cached_rc != 0 else None,
        "status_path": str(status_path),
        "score_promotion_note": (
            "If dirty=true, this dispatch remains reproducible only with the "
            "recorded patch artifacts or by rerunning from a clean commit."
        ),
    }


def _safe_label(value: str) -> str:
    label = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return label or "t1_balle_modal"


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


def _compact_stamp() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")


def _tail(value: str | bytes | None, limit: int = 4096) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    return value[-limit:]


def _result_dir(instance_job_id: str) -> Path:
    return RESULT_ROOT / _safe_label(instance_job_id)


def _estimated_cost(timeout_hours: float) -> float:
    return HOURLY_RATE_T4_USD * float(timeout_hours)


def _local_git_head() -> str | None:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.stdout.strip() if proc.returncode == 0 else None


def _local_git_branch() -> str | None:
    proc = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    branch = proc.stdout.strip()
    return branch if proc.returncode == 0 and branch else None


def _read_claims_ledger_bytes() -> bytes:
    path = REPO_ROOT / ".omx/state/active_lane_dispatch_claims.md"
    if not path.is_file():
        raise SystemExit(f"FATAL: missing dispatch claims ledger: {path}")
    return path.read_bytes()


def _claim_lane(
    *,
    instance_job_id: str,
    predicted_eta_utc: str,
    notes: str,
    status: str = "active_dispatching",
    force: bool = False,
) -> int:
    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools/claim_lane_dispatch.py"),
        "claim",
        "--lane-id",
        LANE_ID,
        "--platform",
        "modal",
        "--instance-job-id",
        instance_job_id,
        "--agent",
        CLAIM_AGENT,
        "--predicted-eta-utc",
        predicted_eta_utc,
        "--status",
        status,
    ]
    if notes:
        cmd.extend(["--notes", notes])
    if force:
        cmd.append("--force")
    print(f"[claim] {' '.join(shlex.quote(c) for c in cmd)}")
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    sys.stdout.write(proc.stdout)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
    return proc.returncode


def _dispatch_command(
    *,
    label: str,
    epochs: int,
    batch_size: int,
    timeout_hours: float,
    cost_cap_usd: float,
    train_timeout_hours: float,
    max_target_pairs: int | None,
    sinkhorn_max_positions_per_chunk: int,
) -> list[str]:
    cmd = [
        "PYTHONPATH=src:upstream:$PWD",
        ".venv/bin/modal",
        "run",
        "--detach",
        "experiments/modal_t1_balle_endtoend.py",
        "--execute",
        "--label",
        label,
        "--epochs",
        str(int(epochs)),
        "--batch-size",
        str(int(batch_size)),
        "--timeout-hours",
        str(float(timeout_hours)),
        "--cost-cap-usd",
        str(float(cost_cap_usd)),
        "--train-timeout-hours",
        str(float(train_timeout_hours)),
        "--sinkhorn-max-positions-per-chunk",
        str(int(sinkhorn_max_positions_per_chunk)),
    ]
    if max_target_pairs is not None:
        cmd.extend(["--max-target-pairs", str(int(max_target_pairs))])
    return cmd


def build_local_plan(
    *,
    label: str | None,
    epochs: int,
    batch_size: int,
    timeout_hours: float,
    cost_cap_usd: float,
    train_timeout_hours: float,
    max_target_pairs: int | None,
    sinkhorn_max_positions_per_chunk: int = DEFAULT_SINKHORN_MAX_POSITIONS_PER_CHUNK,
) -> tuple[dict[str, Any], int]:
    instance_job_id = _safe_label(label or f"t1_balle_modal_{_compact_stamp()}")
    modal_function_timeout_hours = DEFAULT_TIMEOUT_HOURS
    estimated_cost = _estimated_cost(modal_function_timeout_hours)
    predicted_eta_utc = (
        dt.datetime.now(dt.UTC) + dt.timedelta(hours=modal_function_timeout_hours)
    ).isoformat(timespec="seconds").replace("+00:00", "Z")
    validation_errors: list[str] = []
    if abs(float(timeout_hours) - modal_function_timeout_hours) > 1e-9:
        validation_errors.append(
            "timeout_hours_must_match_modal_function_timeout:"
            f"requested={float(timeout_hours):.2f}:actual={modal_function_timeout_hours:.2f}"
        )
    if float(train_timeout_hours) + 1.0 > modal_function_timeout_hours:
        validation_errors.append(
            "train_timeout_exceeds_modal_budget:"
            f"train_plus_eval_buffer={float(train_timeout_hours) + 1.0:.2f}>"
            f"{modal_function_timeout_hours:.2f}"
        )
    remote_finish_budget = (
        float(train_timeout_hours)
        + REMOTE_POST_TRAIN_EVAL_BUFFER_HOURS
        + REMOTE_ARTIFACT_COLLECTION_BUFFER_HOURS
    )
    if remote_finish_budget > modal_function_timeout_hours:
        validation_errors.append(
            "train_timeout_leaves_no_modal_artifact_buffer:"
            f"train_plus_eval_plus_artifact_buffer={remote_finish_budget:.2f}>"
            f"{modal_function_timeout_hours:.2f}"
        )
    if "guard" in instance_job_id.lower():
        if epochs > 100:
            validation_errors.append(
                f"guard_label_requires_epochs_lte_100:epochs={int(epochs)}"
            )
        if batch_size > 1:
            validation_errors.append(
                f"guard_label_requires_batch_size_lte_1:batch_size={int(batch_size)}"
            )
        if max_target_pairs is None or int(max_target_pairs) > 8:
            validation_errors.append(
                "guard_label_requires_max_target_pairs_lte_8:"
                f"max_target_pairs={max_target_pairs}"
            )
        if int(sinkhorn_max_positions_per_chunk) > DEFAULT_SINKHORN_MAX_POSITIONS_PER_CHUNK:
            validation_errors.append(
                "guard_label_requires_sinkhorn_chunk_lte_2048:"
                f"sinkhorn_max_positions_per_chunk={int(sinkhorn_max_positions_per_chunk)}"
            )
        if float(train_timeout_hours) > 3.0:
            validation_errors.append(
                "guard_label_requires_train_timeout_lte_3h:"
                f"train_timeout_hours={float(train_timeout_hours):.2f}"
            )
    if estimated_cost > cost_cap_usd:
        validation_errors.append(
            f"estimated_cost_exceeds_cap:{estimated_cost:.2f}>{float(cost_cap_usd):.2f}"
        )
    if epochs <= 0:
        validation_errors.append("epochs_must_be_positive")
    if batch_size <= 0:
        validation_errors.append("batch_size_must_be_positive")
    if max_target_pairs is not None and max_target_pairs <= 0:
        validation_errors.append("max_target_pairs_must_be_positive")
    if int(sinkhorn_max_positions_per_chunk) <= 0:
        validation_errors.append("sinkhorn_max_positions_per_chunk_must_be_positive")
    canonical_a1_payload = _canonical_a1_payload_snapshot()
    validation_errors.extend(
        f"canonical_a1_payload_{err}"
        for err in canonical_a1_payload["validation_errors"]
    )

    payload = {
        "schema_version": "t1_modal_local_plan_v1",
        "tool": "experiments/modal_t1_balle_endtoend.py",
        "app": APP_NAME,
        "lane_id": LANE_ID,
        "instance_job_id": instance_job_id,
        "created_at_utc": _utc_now(),
        "predicted_eta_utc": predicted_eta_utc,
        "estimated_cost_usd": estimated_cost,
        "modal_function_timeout_hours": modal_function_timeout_hours,
        "requested_timeout_hours": float(timeout_hours),
        "cost_cap_usd": float(cost_cap_usd),
        "params": {
            "epochs": int(epochs),
            "batch_size": int(batch_size),
            "timeout_hours": float(timeout_hours),
            "train_timeout_hours": float(train_timeout_hours),
            "max_target_pairs": max_target_pairs,
            "sinkhorn_max_positions_per_chunk": int(sinkhorn_max_positions_per_chunk),
            "device": "cuda",
            "segmentation_surrogate": "sinkhorn",
            "enable_t13_sqrt_n_budget": True,
            "enable_t19_adaptive_rho": True,
        },
        "canonical_a1_payload": canonical_a1_payload,
        "canonical_path": "archive.zip -> inflate.sh -> upstream/evaluate.py --device cuda",
        "auth_eval_device": "cuda",
        "expected_n_samples": 600,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "dispatch_attempted": False,
        "remote_or_gpu_eval_started": False,
        "lane_claim_opened": False,
        "modal_app_creation_not_attempted": True,
        "dry_run_default": True,
        "requires_execute_for_dispatch": True,
        "ready_for_modal_dispatch_command": not validation_errors,
        "validation_errors": validation_errors,
        "dispatch_command": _dispatch_command(
            label=instance_job_id,
            epochs=epochs,
            batch_size=batch_size,
            timeout_hours=timeout_hours,
            cost_cap_usd=cost_cap_usd,
            train_timeout_hours=train_timeout_hours,
            max_target_pairs=max_target_pairs,
            sinkhorn_max_positions_per_chunk=sinkhorn_max_positions_per_chunk,
        ),
        "recover_command_after_dispatch": (
            ".venv/bin/python experiments/modal_t1_balle_endtoend.py recover "
            f"--label {instance_job_id}"
        ),
        "notes": [
            "Plan-only path opens no lane claim and creates no Modal app.",
            "Real dispatch must use `modal run ... --execute`.",
            "Training-only output is not score evidence; recover accepts only exact contest-CUDA auth eval with zero blockers.",
        ],
    }
    return payload, 0 if not validation_errors else 2


def plan_cli(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="modal_t1_balle_endtoend.py plan",
        description="Write a T1 Modal dispatch plan without app creation, claim, or GPU spend.",
    )
    parser.add_argument("--label", default=None)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--timeout-hours", type=float, default=DEFAULT_TIMEOUT_HOURS)
    parser.add_argument("--cost-cap-usd", type=float, default=DEFAULT_COST_CAP_USD)
    parser.add_argument("--train-timeout-hours", type=float, default=DEFAULT_TRAIN_TIMEOUT_HOURS)
    parser.add_argument("--max-target-pairs", type=int, default=None)
    parser.add_argument(
        "--sinkhorn-max-positions-per-chunk",
        type=int,
        default=DEFAULT_SINKHORN_MAX_POSITIONS_PER_CHUNK,
    )
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args(argv)
    payload, rc = build_local_plan(
        label=args.label,
        epochs=args.epochs,
        batch_size=args.batch_size,
        timeout_hours=args.timeout_hours,
        cost_cap_usd=args.cost_cap_usd,
        train_timeout_hours=args.train_timeout_hours,
        max_target_pairs=args.max_target_pairs,
        sinkhorn_max_positions_per_chunk=args.sinkhorn_max_positions_per_chunk,
    )
    if args.json_out:
        _write_json(Path(args.json_out), payload)
    sys.stdout.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return rc


def _run_logged(
    name: str,
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    log_dir: Path,
    timeout: int,
) -> dict[str, Any]:
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{name}.stdout.log"
    stderr_path = log_dir / f"{name}.stderr.log"
    t0 = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(cwd),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
            check=False,
        )
        rc = int(proc.returncode)
        stdout = proc.stdout.decode("utf-8", errors="replace") if proc.stdout else ""
        stderr = proc.stderr.decode("utf-8", errors="replace") if proc.stderr else ""
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        rc = 124
        stdout = (exc.stdout or b"").decode("utf-8", errors="replace") if exc.stdout else ""
        stderr = (exc.stderr or b"").decode("utf-8", errors="replace") if exc.stderr else ""
        timed_out = True
    stdout_path.write_text(stdout)
    stderr_path.write_text(stderr)
    return {
        "name": name,
        "cmd": cmd,
        "returncode": rc,
        "elapsed_seconds": time.monotonic() - t0,
        "timed_out": timed_out,
        "stdout_tail": _tail(stdout),
        "stderr_tail": _tail(stderr),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }


def _collect_artifacts(out_dir: Path, max_bytes: int = 500 * 1024 * 1024) -> dict[str, bytes]:
    artifacts: dict[str, bytes] = {}
    extensions = (".bin", ".zip", ".pt", ".json", ".log", ".txt", ".sh", ".py")
    for path in sorted(out_dir.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path.suffix.lower() not in extensions:
            continue
        size = path.stat().st_size
        if size > max_bytes:
            continue
        artifacts[str(path.relative_to(out_dir))] = path.read_bytes()
    return artifacts


def _write_mounted_code_custody_marker(
    *,
    mounted_code_git_head: str,
    mounted_code_git_branch: str,
) -> None:
    """Record the local source commit Modal mounted without synthesizing git state."""
    payload = {
        "schema_version": "modal_mounted_code_custody_marker_v1",
        "mounted_code_git_head": mounted_code_git_head,
        "mounted_code_git_branch": mounted_code_git_branch,
        "note": (
            "Modal mounts selected files, not the local .git directory. "
            "The remote script must treat this marker and matching env vars "
            "as custody metadata, not as a synthetic repository HEAD."
        ),
    }
    _write_json(REMOTE_REPO / ".modal_mounted_code_custody.json", payload)


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _finish_remote(
    out_dir: Path,
    *,
    returncode: int,
    stage: str,
    commands: list[dict[str, Any]],
    validation_errors: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    auth_eval = _load_json_if_exists(out_dir / "contest_auth_eval.json")
    adjudication = _load_json_if_exists(out_dir / "auth_eval_adjudication.json")
    score_claim = bool(adjudication and adjudication.get("score_claim") is True)
    summary: dict[str, Any] = {
        "schema_version": "t1_modal_remote_summary_v1",
        "app": APP_NAME,
        "lane_id": LANE_ID,
        "returncode": int(returncode),
        "passed": returncode == 0,
        "stage": stage,
        "completed_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "canonical_path": "archive.zip -> inflate.sh -> upstream/evaluate.py --device cuda",
        "expected_n_samples": 600,
        "commands": commands,
        "validation_errors": list(validation_errors or []),
        "score_claim": score_claim,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "auth_eval_json": str(out_dir / "contest_auth_eval.json") if auth_eval else None,
        "auth_eval_adjudication_json": str(out_dir / "auth_eval_adjudication.json") if adjudication else None,
        "auth_eval": auth_eval,
        "auth_eval_adjudication": adjudication,
    }
    if extra:
        summary.update(extra)
    _write_json(out_dir / "modal_t1_summary.json", summary)
    return {
        "returncode": int(returncode),
        "stage": stage,
        "passed": returncode == 0,
        "summary": summary,
        "eval_data": auth_eval,
        "auth_eval_adjudication": adjudication,
        "artifacts": _collect_artifacts(out_dir),
    }


@app.function(image=run_image, gpu="T4", timeout=int(DEFAULT_TIMEOUT_HOURS * 3600))
def run_t1_balle_modal(
    *,
    instance_job_id: str,
    claim_ledger_bytes: bytes,
    epochs: int,
    batch_size: int,
    train_timeout_seconds: int,
    max_target_pairs: int | None,
    sinkhorn_max_positions_per_chunk: int,
    mounted_code_git_head: str,
    mounted_code_git_branch: str,
) -> dict[str, Any]:
    out_dir = REMOTE_OUT_ROOT / _safe_label(instance_job_id)
    if out_dir.exists():
        subprocess.run(["rm", "-rf", str(out_dir)], check=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    (REMOTE_REPO / ".omx/state").mkdir(parents=True, exist_ok=True)
    claim_path = REMOTE_REPO / ".omx/state/active_lane_dispatch_claims.md"
    claim_path.write_bytes(claim_ledger_bytes)  # BARE_WRITE_OK: single-writer Modal worker copies immutable local claim snapshot
    _write_mounted_code_custody_marker(
        mounted_code_git_head=mounted_code_git_head,
        mounted_code_git_branch=mounted_code_git_branch,
    )
    start = time.monotonic()
    missing_payload = [
        str(path)
        for path in (A1_CANONICAL_REMOTE_PATH, A1_DESIGNATION_REMOTE_PATH)
        if not path.exists()
    ]
    if missing_payload:
        return _finish_remote(
            out_dir,
            returncode=12,
            stage="missing_canonical_a1_payload",
            commands=[],
            validation_errors=[
                "missing_canonical_a1_payload:" + ",".join(missing_payload)
            ],
            extra={
                "elapsed_seconds": time.monotonic() - start,
                "instance_job_id": instance_job_id,
                "remote_claims_path": str(claim_path),
                "mounted_code_git_head": mounted_code_git_head,
                "mounted_code_git_branch": mounted_code_git_branch,
            },
        )

    env = {
        **os.environ,
        "PYTHONPATH": REMOTE_PYTHONPATH,
        "TAC_UPSTREAM_DIR": str(REMOTE_REPO / "upstream"),
        "DALI_DISABLE_NVML": DALI_DISABLE_NVML_VALUE,
        "PYTORCH_CUDA_ALLOC_CONF": PYTORCH_CUDA_ALLOC_CONF_VALUE,
        "PYTHONHASHSEED": "20",
        "WORKSPACE": str(REMOTE_REPO),
        "LOG_DIR": str(out_dir),
        "OUTPUT_DIR": str(out_dir / "output"),
        "T1_ALLOW_SCORE_DOMAIN_TRAINING": "1",
        "T1_RUN_CONTEST_CUDA_AUTH_EVAL": "1",
        "LOCAL_CUDA_WORKER": "1",
        "DISPATCH_PLATFORM": "modal",
        "T1_DISPATCH_INSTANCE_JOB_ID": instance_job_id,
        "T1_DISPATCH_CLAIMS_PATH": str(claim_path),
        "T1_MOUNTED_CODE_GIT_HEAD": mounted_code_git_head,
        "T1_MOUNTED_CODE_GIT_BRANCH": mounted_code_git_branch,
        "EPOCHS": str(int(epochs)),
        "BATCH_SIZE": str(int(batch_size)),
        "SINKHORN_MAX_POSITIONS_PER_CHUNK": str(int(sinkhorn_max_positions_per_chunk)),
        "SEGMENTATION_SURROGATE": "sinkhorn",
        "GRAD_CLIP_NORM": "1.0",
    }
    if max_target_pairs is not None:
        env["MAX_TARGET_PAIRS"] = str(int(max_target_pairs))

    try:
        import_probe = _run_logged(
            "t1_modal_import_probe",
            [
                sys.executable,
                "-c",
                (
                    "; ".join(
                        f"import {module}"
                        for module in CONTEST_SCORER_IMPORT_PROBE_MODULES
                    )
                    + "; "
                    "from tac.scorer import load_differentiable_scorers; "
                    "print('t1 modal scorer import probe OK')"
                ),
            ],
            cwd=REMOTE_REPO,
            env=env,
            log_dir=out_dir / "modal_logs",
            timeout=180,
        )
        if import_probe["returncode"] != 0:
            return _finish_remote(
                out_dir,
                returncode=import_probe["returncode"],
                stage="remote_import_probe_failed",
                commands=[import_probe],
                validation_errors=[
                    f"remote_import_probe_rc={import_probe['returncode']}"
                ],
                extra={
                    "elapsed_seconds": time.monotonic() - start,
                    "instance_job_id": instance_job_id,
                    "remote_claims_path": str(claim_path),
                    "mounted_code_git_head": mounted_code_git_head,
                    "mounted_code_git_branch": mounted_code_git_branch,
                },
            )
        run = _run_logged(
            "remote_lane_t1_balle_endtoend",
            ["bash", str(REMOTE_REPO / "scripts/remote_lane_t1_balle_endtoend.sh")],
            cwd=REMOTE_REPO,
            env=env,
            log_dir=out_dir / "modal_logs",
            timeout=max(300, int(train_timeout_seconds) + 3600),
        )
        result = _finish_remote(
            out_dir,
            returncode=run["returncode"],
            stage="completed" if run["returncode"] == 0 else "remote_script_failed",
            commands=[import_probe, run],
            validation_errors=[] if run["returncode"] == 0 else [f"remote_script_rc={run['returncode']}"],
            extra={
                "elapsed_seconds": time.monotonic() - start,
                "instance_job_id": instance_job_id,
                "remote_claims_path": str(claim_path),
                "mounted_code_git_head": mounted_code_git_head,
                "mounted_code_git_branch": mounted_code_git_branch,
            },
        )
        result["elapsed_seconds"] = time.monotonic() - start
        return result
    except Exception as exc:  # pragma: no cover
        return _finish_remote(
            out_dir,
            returncode=99,
            stage="modal_exception",
            commands=[],
            validation_errors=[f"{type(exc).__name__}: {exc}"],
            extra={"traceback": traceback.format_exc(), "elapsed_seconds": time.monotonic() - start},
        )


def _write_dispatch_metadata(
    *,
    instance_job_id: str,
    call_id: str,
    params: dict[str, Any],
    estimated_cost_usd: float,
    predicted_eta_utc: str,
) -> Path:
    out_dir = _result_dir(instance_job_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    code_snapshot = _mounted_code_snapshot(out_dir)
    canonical_a1_payload = _canonical_a1_payload_snapshot()
    payload = {
        "schema_version": "t1_modal_metadata_v1",
        "app": APP_NAME,
        "lane_id": LANE_ID,
        "instance_job_id": instance_job_id,
        "label": instance_job_id,
        "call_id": call_id,
        "gpu": "T4",
        "canonical_path": "archive.zip -> inflate.sh -> upstream/evaluate.py --device cuda",
        "expected_n_samples": 600,
        "mounted_code_snapshot": code_snapshot,
        "canonical_a1_payload": canonical_a1_payload,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "evidence_grade": "[dispatch in flight; no score claim]",
        "params": params,
        "estimated_cost_usd": float(estimated_cost_usd),
        "predicted_eta_utc": predicted_eta_utc,
        "dispatched_at_utc": _utc_now(),
        "recover_command": (
            ".venv/bin/python experiments/modal_t1_balle_endtoend.py recover "
            f"--label {instance_job_id}"
        ),
    }
    path = out_dir / "modal_metadata.json"
    _write_json(path, payload)
    (out_dir / "modal_call_id.txt").write_text(call_id + "\n")

    legacy_dir = RESULT_ROOT / f"lane_{instance_job_id}_modal"
    if legacy_dir.exists() or legacy_dir.is_symlink():
        try:
            legacy_dir.unlink()
        except IsADirectoryError:
            import shutil

            shutil.rmtree(legacy_dir)
    try:
        legacy_dir.symlink_to(out_dir, target_is_directory=True)
    except OSError:
        legacy_dir.mkdir(parents=True, exist_ok=True)
        _write_json(legacy_dir / "modal_metadata.json", payload)
        (legacy_dir / "modal_call_id.txt").write_text(call_id + "\n")
    return path


def _mark_modal_spawn_submission_unknown(
    *,
    instance_job_id: str,
    predicted_eta_utc: str,
    exc: Exception,
) -> Path:
    """Preserve custody when Modal spawn status is ambiguous.

    Once the code enters ``.spawn()``, an exception may occur after Modal has
    created a server-side call but before the SDK returns the call id. Keep the
    lane claim nonterminal so paid work remains visible until an operator
    reconciles Modal state.
    """
    out_dir = _result_dir(instance_job_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    record_path = out_dir / "modal_spawn_submission_unknown.json"
    payload = {
        "schema_version": "t1_modal_spawn_submission_unknown_v1",
        "app": APP_NAME,
        "lane_id": LANE_ID,
        "instance_job_id": instance_job_id,
        "call_id": None,
        "status": MODAL_SPAWN_SUBMISSION_UNKNOWN_STATUS,
        "terminal_claim_closed": False,
        "server_side_call_existence": "unknown",
        "recovery_required": True,
        "recovery_note": (
            "Modal .spawn() raised after entering the submission boundary. "
            "A server-side call may exist even though the SDK did not return a "
            "call id; reconcile Modal dashboard/API state before closing this claim."
        ),
        "exception_type": type(exc).__name__,
        "exception_repr": repr(exc),
        "traceback_tail": _tail(traceback.format_exc(), limit=8192),
        "created_at_utc": _utc_now(),
    }
    _write_json(record_path, payload)
    claim_rc = _claim_lane(
        instance_job_id=instance_job_id,
        predicted_eta_utc=predicted_eta_utc,
        notes=(
            "T1 Modal .spawn() status unknown after submission boundary; "
            "claim intentionally left nonterminal for manual Modal call "
            f"reconciliation; exception={type(exc).__name__}; record={record_path}"
        ),
        status=MODAL_SPAWN_SUBMISSION_UNKNOWN_STATUS,
        force=True,
    )
    payload["claim_update_rc"] = claim_rc
    _write_json(record_path, payload)
    return record_path


@app.local_entrypoint()
def main(
    label: str | None = None,
    epochs: int = 3000,
    batch_size: int = 16,
    timeout_hours: float = DEFAULT_TIMEOUT_HOURS,
    train_timeout_hours: float = DEFAULT_TRAIN_TIMEOUT_HOURS,
    cost_cap_usd: float = DEFAULT_COST_CAP_USD,
    max_target_pairs: int | None = None,
    sinkhorn_max_positions_per_chunk: int = DEFAULT_SINKHORN_MAX_POSITIONS_PER_CHUNK,
    execute: bool = False,
    force_claim: bool = False,
) -> None:
    instance_job_id = _safe_label(label or f"t1_balle_modal_{_compact_stamp()}")
    plan, rc = build_local_plan(
        label=instance_job_id,
        epochs=epochs,
        batch_size=batch_size,
        timeout_hours=timeout_hours,
        cost_cap_usd=cost_cap_usd,
        train_timeout_hours=train_timeout_hours,
        max_target_pairs=max_target_pairs,
        sinkhorn_max_positions_per_chunk=sinkhorn_max_positions_per_chunk,
    )
    if not execute:
        print(json.dumps(plan, indent=2, sort_keys=True))
        print("\nPLAN ONLY: pass --execute to create a claimed Modal GPU job.")
        return
    if rc != 0:
        raise SystemExit(f"FATAL: plan validation failed: {plan['validation_errors']}")

    estimated_cost = float(plan["estimated_cost_usd"])
    predicted_eta_utc = str(plan["predicted_eta_utc"])
    notes = (
        "T1 Ballé end-to-end Modal T4 dispatch; runs remote_lane_t1_balle_endtoend.sh; "
        "score_claim=false until recover verifies contest-CUDA auth_eval_schema blockers=0; "
        f"cost=${estimated_cost:.2f}"
    )
    claim_rc = _claim_lane(
        instance_job_id=instance_job_id,
        predicted_eta_utc=predicted_eta_utc,
        notes=notes,
        force=force_claim,
    )
    if claim_rc != 0:
        raise SystemExit(
            f"FATAL: lane claim failed rc={claim_rc}; aborting before GPU spend."
        )
    claim_ledger_bytes = _read_claims_ledger_bytes()
    mounted_code_git_head = _local_git_head()
    mounted_code_git_branch = _local_git_branch()
    if not mounted_code_git_head or mounted_code_git_branch != "main":
        _claim_lane(
            instance_job_id=instance_job_id,
            predicted_eta_utc=_utc_now(),
            notes=(
                "T1 Modal dispatch refused before spawn: local mounted code "
                f"custody invalid branch={mounted_code_git_branch!r} "
                f"head={mounted_code_git_head!r}"
            ),
            status="refused_dispatch_invalid_local_code_custody",
            force=True,
        )
        raise SystemExit(
            "FATAL: refusing Modal dispatch without local main-branch git custody."
        )

    try:
        call = run_t1_balle_modal.spawn(
            instance_job_id=instance_job_id,
            claim_ledger_bytes=claim_ledger_bytes,
            epochs=int(epochs),
            batch_size=int(batch_size),
            train_timeout_seconds=max(300, int(float(train_timeout_hours) * 3600)),
            max_target_pairs=max_target_pairs,
            sinkhorn_max_positions_per_chunk=int(sinkhorn_max_positions_per_chunk),
            mounted_code_git_head=mounted_code_git_head,
            mounted_code_git_branch=mounted_code_git_branch,
        )
    except Exception as exc:
        record_path = _mark_modal_spawn_submission_unknown(
            instance_job_id=instance_job_id,
            predicted_eta_utc=predicted_eta_utc,
            exc=exc,
        )
        raise SystemExit(
            "FATAL: Modal .spawn() status is ambiguous after submission boundary: "
            f"{type(exc).__name__}: {exc}. Lane claim left open as "
            f"{MODAL_SPAWN_SUBMISSION_UNKNOWN_STATUS}; recovery record={record_path}"
        ) from exc

    metadata_path = _write_dispatch_metadata(
        instance_job_id=instance_job_id,
        call_id=call.object_id,
        params=plan["params"],
        estimated_cost_usd=estimated_cost,
        predicted_eta_utc=predicted_eta_utc,
    )
    print(f"DISPATCHED T1 Modal T4 call_id={call.object_id}")
    print(f"instance_job_id={instance_job_id}")
    print(f"metadata={metadata_path}")
    print(f"recover=.venv/bin/python experiments/modal_t1_balle_endtoend.py recover --label {instance_job_id}")


def _returncode_is_zero(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return value == 0
    if isinstance(value, str):
        return value.strip() == "0"
    return False


def _close_recovery_claim(
    *,
    instance_job_id: str,
    call_id: str,
    status: str,
    notes: str,
) -> int:
    return _claim_lane(
        instance_job_id=instance_job_id,
        predicted_eta_utc=_utc_now(),
        notes=f"T1 Modal recover call_id={call_id}: {notes}",
        status=status,
        force=True,
    )


def _contest_cuda_score_claim_from_result(result: dict[str, Any]) -> tuple[bool, list[str], dict[str, Any]]:
    from tac.auth_eval_schema import eval_metric_summary, required_contest_cuda_evidence_blockers

    eval_data = result.get("eval_data")
    metrics = eval_metric_summary(eval_data if isinstance(eval_data, dict) else None)
    adjudication = result.get("auth_eval_adjudication")
    expected_archive_bytes = None
    if isinstance(adjudication, dict):
        if not isinstance(adjudication.get("packet_archive_size_bytes"), int):
            blockers = ["t1_remote_adjudication_packet_archive_size_bytes_missing"]
        else:
            blockers = []
            expected_archive_bytes = adjudication["packet_archive_size_bytes"]
        if not isinstance(adjudication.get("packet_archive_sha256"), str) or not adjudication.get("packet_archive_sha256"):
            blockers.append("t1_remote_adjudication_packet_archive_sha256_missing")
    else:
        blockers = []
    blockers = required_contest_cuda_evidence_blockers(
        eval_data if isinstance(eval_data, dict) else None,
        metrics,
        expected_archive_bytes=expected_archive_bytes,
        expected_n_samples=600,
    ) + blockers
    if not isinstance(adjudication, dict) or adjudication.get("score_claim") is not True:
        blockers.append("t1_remote_adjudication_score_claim_not_true")
    elif adjudication.get("blockers"):
        blockers.append("t1_remote_adjudication_has_blockers")
    if isinstance(adjudication, dict) and isinstance(eval_data, dict):
        packet_sha = adjudication.get("packet_archive_sha256")
        eval_sha = eval_data.get("provenance", {}).get("archive_sha256")
        if packet_sha and eval_sha and packet_sha != eval_sha:
            blockers.append("t1_recover_archive_sha_mismatch_adjudication_vs_eval")
        packet_size = adjudication.get("packet_archive_size_bytes")
        eval_size = metrics.get("archive_size_bytes")
        if isinstance(packet_size, int) and isinstance(eval_size, int) and packet_size != eval_size:
            blockers.append("t1_recover_archive_size_mismatch_adjudication_vs_eval")
    if result.get("summary", {}).get("score_claim") is not True:
        blockers.append("t1_remote_summary_score_claim_not_true")
    score_claim = _returncode_is_zero(result.get("returncode")) and not blockers
    return score_claim, blockers, metrics


def recover(label: str) -> int:
    instance_job_id = _safe_label(label)
    out_dir = _result_dir(instance_job_id)
    metadata_path = out_dir / "modal_metadata.json"
    if not metadata_path.is_file():
        print(f"FATAL: missing {metadata_path}", file=sys.stderr)
        return 2
    metadata = json.loads(metadata_path.read_text())
    call_id = metadata.get("call_id")
    if not isinstance(call_id, str) or not call_id:
        print(f"FATAL: {metadata_path} has no call_id", file=sys.stderr)
        return 2
    try:
        fc = modal.functions.FunctionCall.from_id(call_id)
        result = fc.get(timeout=2)
    except modal.exception.OutputExpiredError:
        _close_recovery_claim(
            instance_job_id=instance_job_id,
            call_id=call_id,
            status="failed_modal_result_cache_expired",
            notes="result cache expired before artifact harvest; no score claim",
        )
        return 3
    except TimeoutError:
        print(f"NOT READY: call_id={call_id} still queued or running. Re-run later.")
        return 4
    except Exception as exc:
        _close_recovery_claim(
            instance_job_id=instance_job_id,
            call_id=call_id,
            status="failed_modal_recover_exception",
            notes=f"recover exception {type(exc).__name__}: {exc}; no score claim",
        )
        return 5
    if not isinstance(result, dict):
        _close_recovery_claim(
            instance_job_id=instance_job_id,
            call_id=call_id,
            status="failed_modal_recover_invalid_result",
            notes="Modal result was not a dict; no score claim",
        )
        return 5

    artifacts_dir = out_dir / "harvested_artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for relpath, data in (result.get("artifacts") or {}).items():
        target = artifacts_dir / str(relpath)
        target.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, bytes):
            target.write_bytes(data)

    score_claim, blockers, metrics = _contest_cuda_score_claim_from_result(result)
    adjudication = result.get("auth_eval_adjudication")
    promotion_eligible = (
        bool(adjudication.get("promotion_eligible"))
        if isinstance(adjudication, dict)
        else False
    )
    promotion_blockers = (
        adjudication.get("promotion_blockers")
        if isinstance(adjudication, dict)
        and isinstance(adjudication.get("promotion_blockers"), list)
        else []
    )
    summary_path = out_dir / "harvest_summary.json"
    _write_json(
        summary_path,
        {
            "schema_version": "t1_modal_harvest_summary_v1",
            "instance_job_id": instance_job_id,
            "call_id": call_id,
            "returncode": result.get("returncode"),
            "stage": result.get("stage"),
            "passed": result.get("passed"),
            "metrics": metrics,
            "score_claim": score_claim,
            "promotion_eligible": promotion_eligible,
            "rank_or_kill_eligible": False,
            "promotion_blockers": promotion_blockers,
            "blockers": blockers,
            "artifacts_dir": str(artifacts_dir),
        },
    )
    status = "completed_t1_contest_cuda_recovered" if score_claim else "failed_t1_modal_recovered_no_score_claim"
    notes = (
        f"rc={result.get('returncode')!r}; stage={result.get('stage')!r}; "
        f"score_claim={score_claim}; blockers={blockers}; summary={summary_path}"
    )
    close_rc = _close_recovery_claim(
        instance_job_id=instance_job_id,
        call_id=call_id,
        status=status,
        notes=notes,
    )
    if close_rc != 0:
        return 6
    print(json.dumps(json.loads(summary_path.read_text()), indent=2, sort_keys=True))
    return 0 if score_claim else 1


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "plan":
        raise SystemExit(plan_cli(sys.argv[2:]))
    if len(sys.argv) >= 2 and sys.argv[1] == "recover":
        parser = argparse.ArgumentParser(prog="modal_t1_balle_endtoend.py recover")
        parser.add_argument("--label", required=True)
        ns = parser.parse_args(sys.argv[2:])
        raise SystemExit(recover(ns.label))
    print(
        "Use `plan`, `recover`, or `modal run experiments/modal_t1_balle_endtoend.py --execute ...`.",
        file=sys.stderr,
    )
    raise SystemExit(2)
