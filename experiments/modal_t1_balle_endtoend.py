# SPDX-License-Identifier: MIT
"""Modal actuator for T1 Ballé hyperprior end-to-end.

This file graduates T1 from a remote-command plan to a real Modal job path
without weakening custody:

* default path is local plan-only;
* real dispatch requires ``--execute`` through ``modal run``;
* lane claim is opened before ``.spawn()``;
* the Modal worker runs the existing T1 remote script with a copied claim
  ledger, so the remote script can fail closed on missing dispatch custody;
* the default training window is a short fail-fast tranche, not the 24h Modal
  function cap;
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

from tac.deploy.claims import DispatchClaimSpec, dispatch_claim_command
from tac.deploy.modal.static_manifest import (
    validate_static_manifest_covers_trainer_metadata,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
REMOTE_REPO = Path("/workspace/pact")
RESULT_ROOT = REPO_ROOT / "experiments" / "results"
REMOTE_OUT_ROOT = REMOTE_REPO / "experiments/results/modal_t1_balle_remote"
APP_NAME = "comma-t1-balle-endtoend"
LANE_ID = "t1_balle_128k_endtoend"
CLAIM_AGENT = "codex:modal_t1_balle_endtoend"
DEFAULT_CLAIMS_PATH = REPO_ROOT / ".omx/state/active_lane_dispatch_claims.md"
MODAL_SPAWN_SUBMISSION_UNKNOWN_STATUS = "ambiguous_modal_spawn_submission_recovery_required"
HOURLY_RATE_T4_USD = 0.59
DEFAULT_TIMEOUT_HOURS = 24.0
DEFAULT_TRAIN_TIMEOUT_HOURS = 2.0
MAX_TRAIN_TIMEOUT_HOURS_WITHOUT_LONG_OVERRIDE = 3.0
DEFAULT_T4_SCORE_DOMAIN_BATCH_SIZE = 1
REMOTE_POST_TRAIN_EVAL_BUFFER_HOURS = 1.0
REMOTE_ARTIFACT_COLLECTION_BUFFER_HOURS = 0.25
DEFAULT_COST_CAP_USD = 80.0
DEFAULT_SINKHORN_MAX_POSITIONS_PER_CHUNK = 2048
DEFAULT_T1_ARCHIVE_EVERY_EPOCHS = 100
DEFAULT_T1_MAX_CANDIDATE_ARCHIVE_BYTES = 190_000
DEFAULT_T1_MAX_EXACT_CUDA_CANDIDATES = 1
DEFAULT_T1_SEGMENTATION_SURROGATE = "sinkhorn"
DEFAULT_T1_SCORE_DOMAIN_OBJECTIVE = "direct_score"
DEFAULT_T1_MAX_STABLE_TRAIN_LOSS_ABS = 1e9
CONTEST_EXPECTED_PAIRS = 600
REMOTE_PYTHONPATH = f"{REMOTE_REPO / 'src'}:{REMOTE_REPO / 'upstream'}:{REMOTE_REPO}"
T1_EXTRACTED_ARCHIVE_RUNTIME_CUSTODY_MIN_GIT_HEAD = (
    "0be54cbf9aa407a551a7371a5c438c2df4e3822f"
)
MODAL_MOUNT_MANIFEST: tuple[dict[str, str | bool], ...] = (
    {"kind": "dir", "local_path": "src", "remote_path": str(REMOTE_REPO / "src"), "snapshot_policy": "git_diff"},
    {"kind": "file", "local_path": "pyproject.toml", "remote_path": str(REMOTE_REPO / "pyproject.toml"), "snapshot_policy": "git_diff"},
    {"kind": "file", "local_path": "uv.lock", "remote_path": str(REMOTE_REPO / "uv.lock"), "snapshot_policy": "git_diff"},
    {"kind": "dir", "local_path": "upstream/models", "remote_path": str(REMOTE_REPO / "upstream/models"), "snapshot_policy": "identity_only"},
    {"kind": "dir", "local_path": "upstream/videos", "remote_path": str(REMOTE_REPO / "upstream/videos"), "snapshot_policy": "identity_only"},
    {"kind": "file", "local_path": "upstream/evaluate.py", "remote_path": str(REMOTE_REPO / "upstream/evaluate.py"), "snapshot_policy": "git_diff"},
    {"kind": "file", "local_path": "upstream/frame_utils.py", "remote_path": str(REMOTE_REPO / "upstream/frame_utils.py"), "snapshot_policy": "git_diff"},
    {"kind": "file", "local_path": "upstream/modules.py", "remote_path": str(REMOTE_REPO / "upstream/modules.py"), "snapshot_policy": "git_diff"},
    {"kind": "file", "local_path": "upstream/public_test_video_names.txt", "remote_path": str(REMOTE_REPO / "upstream/public_test_video_names.txt"), "snapshot_policy": "git_diff"},
    {"kind": "file", "local_path": "upstream/pyproject.toml", "remote_path": str(REMOTE_REPO / "upstream/pyproject.toml"), "snapshot_policy": "git_diff"},
    {"kind": "file", "local_path": "upstream/uv.lock", "remote_path": str(REMOTE_REPO / "upstream/uv.lock"), "snapshot_policy": "git_diff"},
    {"kind": "file", "local_path": "experiments/modal_t1_balle_endtoend.py", "remote_path": str(REMOTE_REPO / "experiments/modal_t1_balle_endtoend.py"), "snapshot_policy": "git_diff"},
    {"kind": "file", "local_path": "experiments/__init__.py", "remote_path": str(REMOTE_REPO / "experiments/__init__.py"), "snapshot_policy": "git_diff"},
    {"kind": "file", "local_path": "experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py", "remote_path": str(REMOTE_REPO / "experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py"), "snapshot_policy": "git_diff"},
    {"kind": "file", "local_path": "experiments/contest_auth_eval.py", "remote_path": str(REMOTE_REPO / "experiments/contest_auth_eval.py"), "snapshot_policy": "git_diff"},
    {"kind": "file", "local_path": "tools/build_phase1_packet_compiler.py", "remote_path": str(REMOTE_REPO / "tools/build_phase1_packet_compiler.py"), "snapshot_policy": "git_diff"},
    {"kind": "file", "local_path": "tools/tool_bootstrap.py", "remote_path": str(REMOTE_REPO / "tools/tool_bootstrap.py"), "snapshot_policy": "git_diff"},
    {"kind": "file", "local_path": "tools/claim_lane_dispatch.py", "remote_path": str(REMOTE_REPO / "tools/claim_lane_dispatch.py"), "snapshot_policy": "git_diff"},
    {"kind": "file", "local_path": "scripts/remote_lane_t1_balle_endtoend.sh", "remote_path": str(REMOTE_REPO / "scripts/remote_lane_t1_balle_endtoend.sh"), "snapshot_policy": "git_diff"},
    {"kind": "file", "local_path": "scripts/remote_archive_only_eval.sh", "remote_path": str(REMOTE_REPO / "scripts/remote_archive_only_eval.sh"), "snapshot_policy": "git_diff"},
    {"kind": "file", "local_path": "scripts/probe_nvdec.sh", "remote_path": str(REMOTE_REPO / "scripts/probe_nvdec.sh"), "snapshot_policy": "git_diff"},
    {"kind": "file", "local_path": ".omx/research/pr95_hnerv_muon_trainer_parity_profile_20260510.json", "remote_path": str(REMOTE_REPO / ".omx/research/pr95_hnerv_muon_trainer_parity_profile_20260510.json"), "snapshot_policy": "git_diff", "optional": True},
)
MOUNTED_CODE_PATHS = tuple(
    str(spec["local_path"])
    for spec in MODAL_MOUNT_MANIFEST
    if spec.get("snapshot_policy") == "git_diff"
)
MOUNTED_DATA_PATHS = tuple(
    str(spec["local_path"])
    for spec in MODAL_MOUNT_MANIFEST
    if spec.get("snapshot_policy") == "identity_only"
)
A1_CANONICAL_LOCAL_PATH = REPO_ROOT / "experiments/results/A1_canonical"
A1_DESIGNATION_LOCAL_PATH = REPO_ROOT / ".omx/state/canonical_a1_designation.md"
A1_CANONICAL_REMOTE_PATH = REMOTE_REPO / "experiments/results/A1_canonical"
A1_DESIGNATION_REMOTE_PATH = REMOTE_REPO / ".omx/state/canonical_a1_designation.md"
PR95_PARITY_PROFILE_LOCAL_PATH = (
    REPO_ROOT / ".omx/research/pr95_hnerv_muon_trainer_parity_profile_20260510.json"
)
PR95_PARITY_PROFILE_REMOTE_PATH = (
    REMOTE_REPO / ".omx/research/pr95_hnerv_muon_trainer_parity_profile_20260510.json"
)
T1_TRAINER_PATH = (
    REPO_ROOT / "experiments/train_paradigm_delta_epsilon_zeta_track1_balle_endtoend.py"
)


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


def _assert_static_mount_manifest_covers_t1_trainer_metadata() -> None:
    """Fail closed if the static Modal manifest drifts from trainer metadata."""

    violations = validate_static_manifest_covers_trainer_metadata(
        MODAL_MOUNT_MANIFEST,
        trainer_path=T1_TRAINER_PATH,
        repo_root=REPO_ROOT,
    )
    if violations:
        raise RuntimeError(
            "modal_t1_balle_endtoend static mount manifest no longer covers "
            "trainer-declared required inputs / extra mounts:\n  "
            + "\n  ".join(violations[:8])
        )


_assert_static_mount_manifest_covers_t1_trainer_metadata()

from tac.deploy.modal.runtime import (  # noqa: E402
    CONTEST_SCORER_IMPORT_PROBE_MODULES,
    DALI_DISABLE_NVML_VALUE,
    PYTORCH_CUDA_ALLOC_CONF_VALUE,
    build_contest_cuda_base_image,
)
from tac.deploy.modal.harvest_outcomes import append_terminal_call_id_ledger_event  # noqa: E402


app = modal.App(APP_NAME)


def _apply_modal_mount_manifest(image: Any) -> Any:
    """Apply the same mount manifest that custody snapshots report."""

    for spec in MODAL_MOUNT_MANIFEST:
        local_path = str(spec["local_path"])
        remote_path = str(spec["remote_path"])
        local = REPO_ROOT / local_path
        if bool(spec.get("optional")) and not local.exists():
            continue
        if spec["kind"] == "dir":
            image = image.add_local_dir(local_path, remote_path=remote_path)  # MODAL_MANUAL_MOUNT_OK:uses static MODAL_MOUNT_MANIFEST tuple as discovery primitive; canonical pattern exempt
        elif spec["kind"] == "file":
            image = image.add_local_file(local_path, remote_path=remote_path)  # MODAL_MANUAL_MOUNT_OK:uses static MODAL_MOUNT_MANIFEST tuple as discovery primitive; canonical pattern exempt
        else:  # pragma: no cover - static manifest is test-covered.
            raise ValueError(f"unsupported Modal mount kind: {spec['kind']!r}")
    return image

base_image = build_contest_cuda_base_image(
    modal,
    python_version="3.11",
    extra_pip_packages=("compressai==1.2.8",),
)

run_image = _apply_modal_mount_manifest(
    base_image.env(
        {
            "PYTHONPATH": REMOTE_PYTHONPATH,
            "DALI_DISABLE_NVML": DALI_DISABLE_NVML_VALUE,
            "PYTORCH_CUDA_ALLOC_CONF": PYTORCH_CUDA_ALLOC_CONF_VALUE,
        }
    )
)

if A1_CANONICAL_LOCAL_PATH.exists():
    run_image = run_image.add_local_dir(  # MODAL_MANUAL_MOUNT_OK:uses static MODAL_MOUNT_MANIFEST tuple as discovery primitive; canonical pattern exempt
        str(A1_CANONICAL_LOCAL_PATH.resolve()),
        remote_path=str(A1_CANONICAL_REMOTE_PATH),
    )
if A1_DESIGNATION_LOCAL_PATH.is_file():
    run_image = run_image.add_local_file(  # MODAL_MANUAL_MOUNT_OK:uses static MODAL_MOUNT_MANIFEST tuple as discovery primitive; canonical pattern exempt
        str(A1_DESIGNATION_LOCAL_PATH),
        remote_path=str(A1_DESIGNATION_REMOTE_PATH),
    )
if PR95_PARITY_PROFILE_LOCAL_PATH.is_file():
    run_image = run_image.add_local_file(  # MODAL_MANUAL_MOUNT_OK:uses static MODAL_MOUNT_MANIFEST tuple as discovery primitive; canonical pattern exempt
        str(PR95_PARITY_PROFILE_LOCAL_PATH),
        remote_path=str(PR95_PARITY_PROFILE_REMOTE_PATH),
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
        "modal_mount_manifest": list(MODAL_MOUNT_MANIFEST),
        "mounted_code_paths": mounted_paths,
        "mounted_data_paths": list(MOUNTED_DATA_PATHS),
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


def _recover_result_dir(instance_job_id: str) -> Path:
    """Return the result directory containing Modal recovery metadata.

    New T1 dispatches write :func:`_result_dir` plus a
    ``lane_<label>_modal`` compatibility symlink for generic Modal harvesters.
    Historical/partner dispatches sometimes left only the compatibility
    directory. Recovery must accept the label printed at dispatch time without
    requiring an operator to know which convention produced it.
    """

    safe_id = _safe_label(instance_job_id)
    candidates: list[Path] = []
    if safe_id.startswith("lane_") and safe_id.endswith("_modal"):
        candidates.append(RESULT_ROOT / safe_id)
    candidates.extend(
        [
            _result_dir(safe_id),
            RESULT_ROOT / f"lane_{safe_id}_modal",
        ]
    )
    seen: set[Path] = set()
    for candidate in candidates:
        try:
            key = candidate.resolve()
        except OSError:
            key = candidate
        if key in seen:
            continue
        seen.add(key)
        if (candidate / "modal_metadata.json").is_file():
            return candidate
    return _result_dir(safe_id)


def _metadata_is_t1_recoverable(metadata: dict[str, Any]) -> bool:
    """Return whether ``metadata`` belongs to this T1 recovery tool."""

    if metadata.get("lane_id") == LANE_ID:
        return True
    lane_script = metadata.get("lane_script")
    return (
        isinstance(lane_script, str)
        and Path(lane_script).as_posix() == "scripts/remote_lane_t1_balle_endtoend.sh"
    )


def _estimated_cost(timeout_hours: float) -> float:
    return HOURLY_RATE_T4_USD * float(timeout_hours)


def _estimated_billable_timeout_hours(train_timeout_hours: float) -> float:
    """Estimate the billed wall-clock for the fail-fast training tranche."""

    remote_finish_budget = (
        float(train_timeout_hours)
        + REMOTE_POST_TRAIN_EVAL_BUFFER_HOURS
        + REMOTE_ARTIFACT_COLLECTION_BUFFER_HOURS
    )
    return min(DEFAULT_TIMEOUT_HOURS, remote_finish_budget)


def _contest_auth_eval_requested(max_target_pairs: int | None) -> bool:
    """Return whether this run can emit the full contest video for auth eval."""

    return max_target_pairs is None or int(max_target_pairs) >= CONTEST_EXPECTED_PAIRS


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
    path = DEFAULT_CLAIMS_PATH
    if not path.is_file():
        raise SystemExit(f"FATAL: missing dispatch claims ledger: {path}")
    return path.read_bytes()


def _active_lane_dispatch_conflicts(
    claims_path: Path = DEFAULT_CLAIMS_PATH,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Return unresolved same-lane dispatch claims using the canonical helper."""

    cmd = [
        sys.executable,
        str(REPO_ROOT / "tools" / "claim_lane_dispatch.py"),
        "summary",
        "--claims-path",
        str(claims_path),
        "--format",
        "json",
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return [], [], ["active_claim_summary_timeout"]
    if proc.returncode != 0:
        detail = _tail((proc.stderr or proc.stdout).strip(), limit=500).replace("\n", " ")
        return [], [], [f"active_claim_summary_failed:rc={proc.returncode}:{detail}"]
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return [], [], [f"active_claim_summary_invalid_json:{exc}"]
    active = payload.get("active")
    stale = payload.get("stale_nonterminal")
    if not isinstance(active, list) or not isinstance(stale, list):
        return [], [], ["active_claim_summary_invalid_schema:missing_unresolved_lists"]
    conflicts: list[dict[str, Any]] = []
    stale_conflicts: list[dict[str, Any]] = []

    def _project(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "timestamp_utc": row.get("timestamp_utc"),
            "agent": row.get("agent"),
            "lane_id": row.get("lane_id"),
            "platform": row.get("platform"),
            "instance_job_id": row.get("instance_job_id"),
            "status": row.get("status"),
            "predicted_eta_utc": row.get("predicted_eta_utc"),
        }

    for row in active:
        if not isinstance(row, dict) or row.get("lane_id") != LANE_ID:
            continue
        conflicts.append(_project(row))
    for row in stale:
        if not isinstance(row, dict) or row.get("lane_id") != LANE_ID:
            continue
        stale_conflicts.append(_project(row))
    return conflicts, stale_conflicts, []


def _claim_lane(
    *,
    instance_job_id: str,
    predicted_eta_utc: str,
    notes: str,
    status: str = "active_dispatching",
    force: bool = False,
) -> int:
    cmd = dispatch_claim_command(
        spec=DispatchClaimSpec(
            lane_id=LANE_ID,
            platform="modal",
            instance_job_id=instance_job_id,
            agent=CLAIM_AGENT,
            predicted_eta_utc=predicted_eta_utc,
            force=force,
            notes=notes,
        ),
        status=status,
        python_executable=sys.executable,
        claim_tool=REPO_ROOT / "tools/claim_lane_dispatch.py",
    )
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
    archive_every_epochs: int,
    max_candidate_archive_bytes: int,
    max_exact_cuda_candidates: int,
    allow_long_train_timeout: bool,
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
        "--archive-every-epochs",
        str(int(archive_every_epochs)),
        "--max-candidate-archive-bytes",
        str(int(max_candidate_archive_bytes)),
        "--max-exact-cuda-candidates",
        str(int(max_exact_cuda_candidates)),
    ]
    if max_target_pairs is not None:
        cmd.extend(["--max-target-pairs", str(int(max_target_pairs))])
    if allow_long_train_timeout:
        cmd.append("--allow-long-train-timeout")
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
    archive_every_epochs: int = DEFAULT_T1_ARCHIVE_EVERY_EPOCHS,
    max_candidate_archive_bytes: int = DEFAULT_T1_MAX_CANDIDATE_ARCHIVE_BYTES,
    max_exact_cuda_candidates: int = DEFAULT_T1_MAX_EXACT_CUDA_CANDIDATES,
    allow_long_train_timeout: bool = False,
    claims_path: Path = DEFAULT_CLAIMS_PATH,
) -> tuple[dict[str, Any], int]:
    instance_job_id = _safe_label(label or f"t1_balle_modal_{_compact_stamp()}")
    modal_function_timeout_hours = DEFAULT_TIMEOUT_HOURS
    estimated_billable_timeout_hours = _estimated_billable_timeout_hours(
        train_timeout_hours
    )
    estimated_cost = _estimated_cost(estimated_billable_timeout_hours)
    now_utc = dt.datetime.now(dt.UTC)
    predicted_eta_utc = (
        now_utc + dt.timedelta(hours=estimated_billable_timeout_hours)
    ).isoformat(timespec="seconds").replace("+00:00", "Z")
    modal_function_deadline_utc = (
        now_utc + dt.timedelta(hours=modal_function_timeout_hours)
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
    if (
        float(train_timeout_hours) > MAX_TRAIN_TIMEOUT_HOURS_WITHOUT_LONG_OVERRIDE
        and not allow_long_train_timeout
    ):
        validation_errors.append(
            "train_timeout_requires_explicit_long_override:"
            f"train_timeout_hours={float(train_timeout_hours):.2f}>"
            f"{MAX_TRAIN_TIMEOUT_HOURS_WITHOUT_LONG_OVERRIDE:.2f}"
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
    if int(archive_every_epochs) <= 0:
        validation_errors.append("archive_every_epochs_must_be_positive")
    if int(max_candidate_archive_bytes) <= 0:
        validation_errors.append("max_candidate_archive_bytes_must_be_positive")
    if int(max_exact_cuda_candidates) <= 0:
        validation_errors.append("max_exact_cuda_candidates_must_be_positive")
    if int(max_exact_cuda_candidates) > 2:
        validation_errors.append(
            "max_exact_cuda_candidates_cost_cap_lte_2:"
            f"max_exact_cuda_candidates={int(max_exact_cuda_candidates)}"
        )
    contest_auth_eval_requested = _contest_auth_eval_requested(max_target_pairs)
    if contest_auth_eval_requested and int(batch_size) > DEFAULT_T4_SCORE_DOMAIN_BATCH_SIZE:
        validation_errors.append(
            "full_600_pair_t4_scorer_domain_requires_batch_size_lte_1:"
            f"batch_size={int(batch_size)}"
        )
    canonical_a1_payload = _canonical_a1_payload_snapshot()
    validation_errors.extend(
        f"canonical_a1_payload_{err}"
        for err in canonical_a1_payload["validation_errors"]
    )
    (
        active_lane_dispatch_conflicts,
        stale_lane_dispatch_conflicts,
        active_claim_errors,
    ) = _active_lane_dispatch_conflicts(Path(claims_path))
    validation_errors.extend(active_claim_errors)
    for conflict in active_lane_dispatch_conflicts:
        validation_errors.append(
            "active_lane_dispatch_conflict:"
            f"{conflict.get('instance_job_id')}:{conflict.get('status')}"
        )
    for conflict in stale_lane_dispatch_conflicts:
        validation_errors.append(
            "stale_lane_dispatch_conflict:"
            f"{conflict.get('instance_job_id')}:{conflict.get('status')}"
        )

    payload = {
        "schema_version": "t1_modal_local_plan_v1",
        "tool": "experiments/modal_t1_balle_endtoend.py",
        "app": APP_NAME,
        "lane_id": LANE_ID,
        "instance_job_id": instance_job_id,
        "created_at_utc": _utc_now(),
        "predicted_eta_utc": predicted_eta_utc,
        "modal_function_deadline_utc": modal_function_deadline_utc,
        "estimated_cost_usd": estimated_cost,
        "estimated_billable_timeout_hours": estimated_billable_timeout_hours,
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
            "archive_in_loop": True,
            "archive_every_epochs": int(archive_every_epochs),
            "max_candidate_archive_bytes": int(max_candidate_archive_bytes),
            "max_exact_cuda_candidates": int(max_exact_cuda_candidates),
            "allow_long_train_timeout": bool(allow_long_train_timeout),
            "device": "cuda",
            "segmentation_surrogate": DEFAULT_T1_SEGMENTATION_SURROGATE,
            "score_domain_objective": DEFAULT_T1_SCORE_DOMAIN_OBJECTIVE,
            "max_stable_train_loss_abs": DEFAULT_T1_MAX_STABLE_TRAIN_LOSS_ABS,
            "enable_t13_sqrt_n_budget": True,
            "enable_t19_adaptive_rho": True,
            "contest_cuda_auth_eval_requested": contest_auth_eval_requested,
        },
        "canonical_a1_payload": canonical_a1_payload,
        "canonical_path": "archive.zip -> inflate.sh -> upstream/evaluate.py --device cuda",
        "auth_eval_device": "cuda",
        "expected_n_samples": 600,
        "contest_cuda_auth_eval_requested": contest_auth_eval_requested,
        "contest_cuda_auth_eval_request_reason": (
            "full_600_pair_export"
            if contest_auth_eval_requested
            else "subset_guard_training_only_max_target_pairs_lt_600"
        ),
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "claims_path": str(claims_path),
        "active_lane_dispatch_conflicts": active_lane_dispatch_conflicts,
        "stale_lane_dispatch_conflicts": stale_lane_dispatch_conflicts,
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
            archive_every_epochs=archive_every_epochs,
            max_candidate_archive_bytes=max_candidate_archive_bytes,
            max_exact_cuda_candidates=max_exact_cuda_candidates,
            allow_long_train_timeout=allow_long_train_timeout,
        ),
        "recover_command_after_dispatch": (
            ".venv/bin/python experiments/modal_t1_balle_endtoend.py recover "
            f"--label {instance_job_id}"
        ),
        "notes": [
            "Plan-only path opens no lane claim and creates no Modal app.",
            "Real dispatch must use `modal run ... --execute`.",
            (
                "Subset guard runs train and export local artifacts only; exact "
                "contest-CUDA auth eval is requested only for full 600-pair exports."
            ),
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
    parser.add_argument("--batch-size", type=int, default=DEFAULT_T4_SCORE_DOMAIN_BATCH_SIZE)
    parser.add_argument("--timeout-hours", type=float, default=DEFAULT_TIMEOUT_HOURS)
    parser.add_argument("--cost-cap-usd", type=float, default=DEFAULT_COST_CAP_USD)
    parser.add_argument("--train-timeout-hours", type=float, default=DEFAULT_TRAIN_TIMEOUT_HOURS)
    parser.add_argument("--max-target-pairs", type=int, default=None)
    parser.add_argument(
        "--sinkhorn-max-positions-per-chunk",
        type=int,
        default=DEFAULT_SINKHORN_MAX_POSITIONS_PER_CHUNK,
    )
    parser.add_argument("--archive-every-epochs", type=int, default=DEFAULT_T1_ARCHIVE_EVERY_EPOCHS)
    parser.add_argument(
        "--max-candidate-archive-bytes",
        type=int,
        default=DEFAULT_T1_MAX_CANDIDATE_ARCHIVE_BYTES,
    )
    parser.add_argument(
        "--max-exact-cuda-candidates",
        type=int,
        default=DEFAULT_T1_MAX_EXACT_CUDA_CANDIDATES,
    )
    parser.add_argument(
        "--allow-long-train-timeout",
        action="store_true",
        help=(
            "Allow train-timeout-hours above the default fail-fast cap. "
            "Use only with an explicit throughput justification."
        ),
    )
    parser.add_argument("--claims-path", type=Path, default=DEFAULT_CLAIMS_PATH)
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
        archive_every_epochs=args.archive_every_epochs,
        max_candidate_archive_bytes=args.max_candidate_archive_bytes,
        max_exact_cuda_candidates=args.max_exact_cuda_candidates,
        allow_long_train_timeout=args.allow_long_train_timeout,
        claims_path=args.claims_path,
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
    remote_adjudication_score_claim = bool(
        adjudication and adjudication.get("score_claim") is True
    )
    remote_adjudication_promotion_eligible = bool(
        adjudication and adjudication.get("promotion_eligible") is True
    )
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
        "score_claim": False,
        "score_claim_authority": "local_recover_only_after_exact_cuda_custody_gate",
        "remote_adjudication_score_claim": remote_adjudication_score_claim,
        "remote_adjudication_promotion_eligible": remote_adjudication_promotion_eligible,
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
    archive_every_epochs: int,
    max_candidate_archive_bytes: int,
    max_exact_cuda_candidates: int,
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

    contest_auth_eval_requested = _contest_auth_eval_requested(max_target_pairs)
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
        "T1_RUN_CONTEST_CUDA_AUTH_EVAL": "1" if contest_auth_eval_requested else "0",
        "LOCAL_CUDA_WORKER": "1",
        "DISPATCH_PLATFORM": "modal",
        "T1_DISPATCH_INSTANCE_JOB_ID": instance_job_id,
        "T1_DISPATCH_CLAIMS_PATH": str(claim_path),
        "T1_MOUNTED_CODE_GIT_HEAD": mounted_code_git_head,
        "T1_MOUNTED_CODE_GIT_BRANCH": mounted_code_git_branch,
        "EPOCHS": str(int(epochs)),
        "BATCH_SIZE": str(int(batch_size)),
        "SINKHORN_MAX_POSITIONS_PER_CHUNK": str(int(sinkhorn_max_positions_per_chunk)),
        "T1_ARCHIVE_IN_LOOP": "1",
        "T1_ARCHIVE_EVERY_EPOCHS": str(int(archive_every_epochs)),
        "T1_MAX_CANDIDATE_ARCHIVE_BYTES": str(int(max_candidate_archive_bytes)),
        "T1_MAX_EXACT_CUDA_CANDIDATES": str(int(max_exact_cuda_candidates)),
        "SEGMENTATION_SURROGATE": DEFAULT_T1_SEGMENTATION_SURROGATE,
        "T1_SCORE_DOMAIN_OBJECTIVE": DEFAULT_T1_SCORE_DOMAIN_OBJECTIVE,
        "T1_MAX_STABLE_TRAIN_LOSS_ABS": str(DEFAULT_T1_MAX_STABLE_TRAIN_LOSS_ABS),
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
                "contest_cuda_auth_eval_requested": contest_auth_eval_requested,
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
    batch_size: int = DEFAULT_T4_SCORE_DOMAIN_BATCH_SIZE,
    timeout_hours: float = DEFAULT_TIMEOUT_HOURS,
    train_timeout_hours: float = DEFAULT_TRAIN_TIMEOUT_HOURS,
    cost_cap_usd: float = DEFAULT_COST_CAP_USD,
    max_target_pairs: int | None = None,
    sinkhorn_max_positions_per_chunk: int = DEFAULT_SINKHORN_MAX_POSITIONS_PER_CHUNK,
    archive_every_epochs: int = DEFAULT_T1_ARCHIVE_EVERY_EPOCHS,
    max_candidate_archive_bytes: int = DEFAULT_T1_MAX_CANDIDATE_ARCHIVE_BYTES,
    max_exact_cuda_candidates: int = DEFAULT_T1_MAX_EXACT_CUDA_CANDIDATES,
    allow_long_train_timeout: bool = False,
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
        archive_every_epochs=archive_every_epochs,
        max_candidate_archive_bytes=max_candidate_archive_bytes,
        max_exact_cuda_candidates=max_exact_cuda_candidates,
        allow_long_train_timeout=allow_long_train_timeout,
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
            archive_every_epochs=int(archive_every_epochs),
            max_candidate_archive_bytes=int(max_candidate_archive_bytes),
            max_exact_cuda_candidates=int(max_exact_cuda_candidates),
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


def _mounted_code_git_head(metadata: dict[str, Any] | None) -> str | None:
    if not isinstance(metadata, dict):
        return None
    snapshot = metadata.get("mounted_code_snapshot")
    if not isinstance(snapshot, dict):
        return None
    head = snapshot.get("git_head")
    return head if isinstance(head, str) and head else None


def _git_head_contains_required_commit(head: str, required_commit: str) -> bool:
    if not re.fullmatch(r"[0-9a-fA-F]{40}", head):
        return False
    if not re.fullmatch(r"[0-9a-fA-F]{40}", required_commit):
        return False
    proc = subprocess.run(
        ["git", "merge-base", "--is-ancestor", required_commit, head],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode == 0


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


def _contest_cuda_score_claim_from_result(
    result: dict[str, Any],
    *,
    metadata: dict[str, Any] | None = None,
) -> tuple[bool, list[str], dict[str, Any]]:
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
    summary = result.get("summary")
    if isinstance(summary, dict):
        if summary.get("score_claim") is True:
            blockers.append("t1_remote_summary_score_claim_must_be_false_until_recover")
        if summary.get("remote_adjudication_score_claim") is not True:
            blockers.append("t1_remote_summary_remote_adjudication_score_claim_not_true")
    else:
        blockers.append("t1_remote_summary_missing")
    if metadata is not None:
        mounted_head = _mounted_code_git_head(metadata)
        if not mounted_head:
            blockers.append("t1_mounted_code_git_head_missing")
        elif not _git_head_contains_required_commit(
            mounted_head,
            T1_EXTRACTED_ARCHIVE_RUNTIME_CUSTODY_MIN_GIT_HEAD,
        ):
            blockers.append(
                "t1_mounted_code_missing_extracted_archive_runtime_hardening"
            )
    score_claim = _returncode_is_zero(result.get("returncode")) and not blockers
    return score_claim, blockers, metrics


def _eval_data_terminal_claim_facts(eval_data: Any, metrics: dict[str, Any]) -> str:
    if not isinstance(eval_data, dict):
        return ""
    provenance = eval_data.get("provenance")
    if not isinstance(provenance, dict):
        provenance = {}
    runtime_manifest = eval_data.get("inflate_runtime_manifest")
    if not isinstance(runtime_manifest, dict):
        runtime_manifest = {}
    archive_sha = (
        provenance.get("archive_sha256")
        or eval_data.get("archive_sha256")
        or eval_data.get("expected_archive_sha256")
    )
    runtime_sha = (
        eval_data.get("runtime_tree_sha256")
        or eval_data.get("inflate_runtime_tree_sha256")
        or eval_data.get("runtime_content_tree_sha256")
        or eval_data.get("inflate_runtime_content_tree_sha256")
        or runtime_manifest.get("runtime_tree_sha256")
        or runtime_manifest.get("runtime_content_tree_sha256")
    )
    facts: list[str] = []
    if isinstance(archive_sha, str) and archive_sha:
        facts.append(f"archive_sha={archive_sha}")
    archive_bytes = metrics.get("archive_size_bytes")
    if isinstance(archive_bytes, int):
        facts.append(f"archive_bytes={archive_bytes}")
    score = metrics.get("score")
    if isinstance(score, (int, float)):
        facts.append(f"score_recomputed={float(score):.17g}")
    if isinstance(runtime_sha, str) and runtime_sha:
        facts.append(f"runtime_tree_sha256={runtime_sha}")
    return "; ".join(facts)


def recover(label: str) -> int:
    instance_job_id = _safe_label(label)
    out_dir = _recover_result_dir(instance_job_id)
    metadata_path = out_dir / "modal_metadata.json"
    if not metadata_path.is_file():
        print(f"FATAL: missing {metadata_path}", file=sys.stderr)
        return 2
    metadata = json.loads(metadata_path.read_text())
    if not isinstance(metadata, dict) or not _metadata_is_t1_recoverable(metadata):
        print(
            "FATAL: refusing T1 recovery for non-T1 Modal metadata at "
            f"{metadata_path}; use the lane-specific or generic Modal recovery "
            "tool instead.",
            file=sys.stderr,
        )
        return 2
    call_id = metadata.get("call_id")
    if not isinstance(call_id, str) or not call_id:
        print(f"FATAL: {metadata_path} has no call_id", file=sys.stderr)
        return 2
    try:
        fc = modal.functions.FunctionCall.from_id(call_id)
        result = fc.get(timeout=2)
    except modal.exception.OutputExpiredError:
        append_terminal_call_id_ledger_event(
            repo_root=REPO_ROOT,
            metadata={**metadata, "call_id": call_id, "platform": "modal"},
            harvested={"status": "expired", "crash_kind": "RESULT_CACHE_EXPIRED"},
            terminal_claim=None,
            agent="codex:modal_t1_balle_endtoend",
        )
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
        append_terminal_call_id_ledger_event(
            repo_root=REPO_ROOT,
            metadata={**metadata, "call_id": call_id, "platform": "modal"},
            harvested={
                "status": "error_recover_exception",
                "error_type": type(exc).__name__,
                "error_message": str(exc)[:1000],
            },
            terminal_claim=None,
            agent="codex:modal_t1_balle_endtoend",
        )
        _close_recovery_claim(
            instance_job_id=instance_job_id,
            call_id=call_id,
            status="failed_modal_recover_exception",
            notes=f"recover exception {type(exc).__name__}: {exc}; no score claim",
        )
        return 5
    if not isinstance(result, dict):
        append_terminal_call_id_ledger_event(
            repo_root=REPO_ROOT,
            metadata={**metadata, "call_id": call_id, "platform": "modal"},
            harvested={
                "status": "error_unexpected_result_type",
                "error_type": type(result).__name__,
            },
            terminal_claim=None,
            agent="codex:modal_t1_balle_endtoend",
        )
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

    score_claim, blockers, metrics = _contest_cuda_score_claim_from_result(
        result,
        metadata=metadata,
    )
    adjudication = result.get("auth_eval_adjudication")
    remote_adjudication_promotion_eligible = (
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
    promotion_eligible = bool(
        score_claim
        and remote_adjudication_promotion_eligible
        and not promotion_blockers
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
            "remote_adjudication_promotion_eligible": remote_adjudication_promotion_eligible,
            "rank_or_kill_eligible": False,
            "promotion_blockers": promotion_blockers,
            "blockers": blockers,
            "artifacts_dir": str(artifacts_dir),
        },
    )
    if score_claim:
        status = "completed_contest_cuda_t1_recovered"
    elif _returncode_is_zero(result.get("returncode")):
        status = "completed_t1_training_only_recovered_no_score_claim"
    else:
        status = "failed_t1_modal_recovered_no_score_claim"
    terminal_facts = _eval_data_terminal_claim_facts(result.get("eval_data"), metrics)
    notes = (
        f"rc={result.get('returncode')!r}; stage={result.get('stage')!r}; "
        f"score_claim={score_claim}; blockers={blockers}; summary={summary_path}"
    )
    if terminal_facts:
        notes = f"{notes}; {terminal_facts}"
    close_rc = _close_recovery_claim(
        instance_job_id=instance_job_id,
        call_id=call_id,
        status=status,
        notes=notes,
    )
    append_terminal_call_id_ledger_event(
        repo_root=REPO_ROOT,
        metadata={**metadata, "call_id": call_id, "platform": "modal"},
        harvested={
            key: value
            for key, value in result.items()
            if key != "artifacts"
        },
        terminal_claim=None,
        agent="codex:modal_t1_balle_endtoend",
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
