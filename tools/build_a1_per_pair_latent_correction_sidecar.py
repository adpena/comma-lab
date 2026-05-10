r"""Re-search the per-pair latent correction sidecar against A1's substrate.

Background (from forensics dossier
`.omx/research/hnerv_leaderboard_binary_forensics_dossier_20260509.md`):

  PR99 introduced the per-pair latent correction sidecar mechanism: a
  single-dim perturbation per latent pair, grid-searched at compression time
  against joint SegNet+PoseNet distortion. PR99-103 all inherit this. The
  fixed delta vocabulary is ``[-10, -8, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4,
  5, 6, 8, 10] × 0.01``. Each pair gets one of 1+28*16=449 choices: no-op or
  one of 16 deltas applied to one of 28 latent dims.

  A1's existing 607-byte sidecar is INHERITED from PR101 (its source archive).
  PR101's sidecar was searched against PR98's substrate (which is the same
  decoder weights). A1 then fine-tuned the latents/decoder away from PR98's,
  so the optimal per-pair sidecar for A1 may differ.

This tool re-runs the per-pair greedy search against A1's frozen decoder and
A1's frozen latents, OPTIONALLY using:
  - proxy MSE between A1's decoded frames and ground-truth uint8 frames as
    the search signal (fast; ~CPU-minutes on M-series ARM)
  - SegNet+PoseNet joint distortion as the search signal (slow; ~CPU-hours
    even on M-series ARM)

The search is greedy, single-pair-at-a-time, single-dim only (matching PR101's
mechanism). For each pair, evaluates 1+28*16=449 candidate perturbations and
picks the one that minimizes the chosen objective.

Output:
  experiments/results/a1_per_pair_latent_sidecar_resampled_<timestamp>/
    submission_dir/                         (variant submission_dir)
      archive.zip                           (NEW archive; same decoder/latents as A1
                                             but new sidecar bytes — score-affecting)
      inflate.py                            (A1's existing inflate.py with PR101 bias)
      inflate.sh                            (A1's existing)
      src/{codec,model}.py                  (A1's existing)
    sidecar_search.log                      (per-pair best deltas, search timings)
    sidecar_manifest.json                   (search config, search signal,
                                             old/new archive SHA, expected delta)

Per CLAUDE.md:
  - All claims tagged ``[predicted; per-pair latent sidecar resampled on A1
    substrate via <signal>]`` until GHA result returns
    ``[contest-CPU GHA Linux x86_64]``.
  - Per HNeRV-parity discipline lesson 11: no-op detector — sidecar bytes
    DO change archive bytes (~600 bytes); new score MUST be re-measured.
  - Per lesson 13: any "didn't beat baseline" finding is
    DEFERRED-pending-research with reactivation criteria.

NOTE on operational scope:
  The PROXY-MSE search is CPU-feasible (~10-30 min on M-series ARM for 600
  pairs × 449 candidates). The SegNet+PoseNet search is too expensive without
  GPU (~30+ hours on CPU for full 600×449 evaluation). This tool defaults to
  ``--search-signal proxy_mse``; ``--search-signal joint_seg_pose`` is gated
  behind ``--accept-cpu-budget`` and a budget ceiling.
"""
from __future__ import annotations

import argparse
import atexit
import datetime as dt
import hashlib
import json
import os
import shutil
import stat
import struct
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent

A1_SUBMISSION_DIR = (
    REPO_ROOT
    / "experiments/results/track1_phase_a1_score_gradient_latentalign_importpathfix_lr2e6_20260509T012628Z_modal/harvested_artifacts/finetuned_archive/submission_dir"
)
A1_ARCHIVE_PATH = A1_SUBMISSION_DIR / "archive.zip"
A1_EXPECTED_ARCHIVE_SHA = (
    "87ec7ca5f2f328a8acdfc65f5cce0ab08a3a558eae88f36d4140870f141492b5"
)
A1_EXPECTED_ARCHIVE_BYTES = 178262
SIDECAR_LANE_ID = "lane_a1_per_pair_latent_sidecar_resampled"
LOCAL_RUNTIME_CUSTODY_SCHEMA = "a1_sidecar_local_runtime_custody_v1"
MEMBER_SECTION_PROOF_SCHEMA = "a1_sidecar_member_section_proof_v1"
RUNTIME_SMOKE_SCHEMA = "a1_sidecar_runtime_smoke_v1"
SIDECAR_CHOICE_STATE_SCHEMA = "a1_sidecar_choice_state_v1"
SIDECAR_PAIR_SEARCH_RECORDS_SCHEMA = "a1_sidecar_pair_search_records_v1"
SIDECAR_DISPATCH_CLAIM_SCHEMA = "a1_sidecar_dispatch_claim_v1"
SIDECAR_EXACT_EVAL_PREFLIGHT_SCHEMA = "a1_sidecar_exact_eval_preflight_v1"
DEFAULT_DISPATCH_CLAIMS_PATH = REPO_ROOT / ".omx/state/active_lane_dispatch_claims.md"
REQUIRED_RUNTIME_FILES = ("inflate.py", "inflate.sh", "src/codec.py", "src/model.py")
RUNTIME_TREE_EXCLUDED_FILES = frozenset(
    {"archive.zip", "archive_manifest.json", "contest_auth_eval.json", "report.txt"}
)
RUNTIME_TREE_EXCLUDED_PARTS = frozenset({"__pycache__"})
TERMINAL_DISPATCH_STATUS_PREFIXES = (
    "completed_",
    "failed_",
    "preempted",
    "cancelled",
    "refused_dispatch",
    "stale_assumed_dead",
    "stale_superseded",
    "stopped_",
    "falsified_",
    "retired_",
    "config_retired_",
    "measured_implementation_retired_",
)

# Sidecar fixed delta vocabulary (PR99-PR103 lineage)
SIDECAR_DELTAS_X100 = np.array(
    [-10, -8, -6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 8, 10], dtype=np.int8
)
SIDECAR_BASE = 1 + 28 * len(SIDECAR_DELTAS_X100)  # 449

N_PAIRS = 600
LATENT_DIM = 28
A1_LATENT_BLOB_LEN = 15_387
EVAL_H, EVAL_W = 384, 512
CAMERA_H, CAMERA_W = 874, 1164


class SidecarOutputLock:
    """Fail-closed single-writer sentinel for resumable sidecar state."""

    def __init__(self, output_dir: Path) -> None:
        self.path = output_dir / ".sidecar_choice_state.lock"
        self._acquired = False

    def acquire(self) -> None:
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError as exc:
            detail = ""
            try:
                detail = self.path.read_text(encoding="utf-8", errors="replace").strip()
            except OSError:
                pass
            raise RuntimeError(
                f"sidecar output directory is already locked: {self.path}"
                + (f" ({detail})" if detail else "")
                + ". Do not run duplicate writers into one output dir; remove the "
                "lock only after verifying no builder process is active."
            ) from exc
        payload = {
            "pid": os.getpid(),
            "started_at_utc": dt.datetime.now(dt.UTC).isoformat(),
            "path": str(self.path),
        }
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload, sort_keys=True) + "\n")
        self._acquired = True
        atexit.register(self.release)

    def release(self) -> None:
        if not self._acquired:
            return
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            return
        self._acquired = False


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest_path(path: Path) -> str:
    """Return a repo-relative path when possible, else an absolute path."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _is_sha256(value: object) -> bool:
    if not isinstance(value, str):
        return False
    text = value.strip().lower()
    return len(text) == 64 and all(ch in "0123456789abcdef" for ch in text)


def _mode_string(path: Path) -> str:
    return oct(path.stat().st_mode & 0o777)


def _canonical_json_sha256(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _runtime_file_record(submission_dir: Path, relpath: str) -> dict[str, Any]:
    path = submission_dir / relpath
    if not path.is_file():
        raise FileNotFoundError(f"required runtime file missing: {relpath}")
    if path.is_symlink():
        raise FileNotFoundError(f"runtime file is a symlink: {relpath}")
    st = path.stat()
    return {
        "relative_path": relpath,
        "path": manifest_path(path),
        "bytes": st.st_size,
        "sha256": sha256_of(path),
        "mode": _mode_string(path),
        "executable": bool(st.st_mode & stat.S_IXUSR),
    }


def _runtime_tree_relpaths(submission_dir: Path) -> list[str]:
    relpaths: list[str] = []
    for path in sorted(submission_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(submission_dir).as_posix()
        if rel in RUNTIME_TREE_EXCLUDED_FILES:
            continue
        if any(part in RUNTIME_TREE_EXCLUDED_PARTS for part in path.relative_to(submission_dir).parts):
            continue
        if path.name.endswith((".pyc", ".pyo")):
            continue
        relpaths.append(rel)
    return relpaths


def _runtime_tree_sha256(files: list[dict[str, Any]]) -> str:
    basis = [
        {
            "relative_path": row["relative_path"],
            "bytes": row["bytes"],
            "sha256": row["sha256"],
            "mode": row["mode"],
            "executable": row["executable"],
        }
        for row in sorted(files, key=lambda item: str(item["relative_path"]))
    ]
    return _canonical_json_sha256(basis)


def _bytes_record(name: str, payload: bytes, *, offset: int | None = None) -> dict[str, Any]:
    record: dict[str, Any] = {
        "name": name,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    if offset is not None:
        record["offset"] = offset
    return record


def collect_member_section_proof(archive_path: Path) -> dict[str, Any]:
    """Collect ZIP-member and inner-section proof for an A1 sidecar archive."""

    if not archive_path.is_file():
        raise FileNotFoundError(f"archive missing: {archive_path}")
    archive_sha = sha256_of(archive_path)
    with zipfile.ZipFile(archive_path, "r") as zf:
        infos = zf.infolist()
        if len(infos) != 1:
            raise ValueError(f"A1 sidecar archive must have one member, got {len(infos)}")
        info = infos[0]
        if info.filename != "x":
            raise ValueError(f"A1 sidecar archive member must be x, got {info.filename!r}")
        inner = zf.read(info.filename)

    if len(inner) < 4:
        raise ValueError("A1 sidecar inner member too short for decoder section header")
    section_total = struct.unpack_from("<I", inner, 0)[0]
    latent_start = section_total
    latent_end = latent_start + A1_LATENT_BLOB_LEN
    if section_total < 4 or section_total > len(inner):
        raise ValueError(f"bad decoder section total: {section_total}")
    if latent_end > len(inner):
        raise ValueError(
            f"bad latent section: latent_end={latent_end} inner_bytes={len(inner)}"
        )

    decoder_blob = inner[4:section_total]
    latent_blob = inner[latent_start:latent_end]
    sidecar_blob = inner[latent_end:]
    if not decoder_blob:
        raise ValueError("empty decoder blob in A1 sidecar archive")
    if len(latent_blob) != A1_LATENT_BLOB_LEN:
        raise ValueError(
            f"latent blob length mismatch: {len(latent_blob)} != {A1_LATENT_BLOB_LEN}"
        )

    return {
        "schema_version": MEMBER_SECTION_PROOF_SCHEMA,
        "archive": {
            "path": manifest_path(archive_path),
            "bytes": archive_path.stat().st_size,
            "sha256": archive_sha,
        },
        "member_count": len(infos),
        "single_member_name": info.filename,
        "members": [
            {
                "filename": info.filename,
                "file_size": info.file_size,
                "compress_size": info.compress_size,
                "compress_type": info.compress_type,
                "crc": f"{info.CRC:08x}",
                "header_offset": info.header_offset,
                "date_time": list(info.date_time),
                "sha256": hashlib.sha256(inner).hexdigest(),
            }
        ],
        "inner_sections": {
            "total_inner_bytes": len(inner),
            "total_inner_sha256": hashlib.sha256(inner).hexdigest(),
            "decoder_section_total": section_total,
            "decoder_blob": _bytes_record("decoder_blob", decoder_blob, offset=4),
            "latent_blob": _bytes_record("latent_blob", latent_blob, offset=latent_start),
            "sidecar_blob": _bytes_record("sidecar_blob", sidecar_blob, offset=latent_end),
        },
    }


def collect_local_runtime_custody(
    submission_dir: Path,
    *,
    archive_path: Path,
) -> dict[str, Any]:
    """Collect local runtime custody for the submission tree.

    The archive bytes/SHA are recorded separately from the runtime tree hash
    because archive bytes are the scored payload, while the runtime tree is the
    local inflate surface that consumes those bytes.
    """

    if not archive_path.is_file():
        raise FileNotFoundError(f"archive missing: {archive_path}")
    missing_required = [
        rel for rel in REQUIRED_RUNTIME_FILES if not (submission_dir / rel).is_file()
    ]
    if missing_required:
        raise FileNotFoundError(
            "required runtime file missing: " + ", ".join(missing_required)
        )
    relpaths = _runtime_tree_relpaths(submission_dir)
    files = [_runtime_file_record(submission_dir, rel) for rel in relpaths]
    archive_record = {
        "relative_path": "archive.zip",
        "path": manifest_path(archive_path),
        "bytes": archive_path.stat().st_size,
        "sha256": sha256_of(archive_path),
    }
    return {
        "schema_version": LOCAL_RUNTIME_CUSTODY_SCHEMA,
        "submission_dir": manifest_path(submission_dir),
        "archive": archive_record,
        "files": files,
        "file_count": len(files),
        "required_runtime_files": list(REQUIRED_RUNTIME_FILES),
        "excluded_files": sorted(RUNTIME_TREE_EXCLUDED_FILES),
        "excluded_parts": sorted(RUNTIME_TREE_EXCLUDED_PARTS),
        "runtime_tree_sha256": _runtime_tree_sha256(files),
    }


def _manifest_archive_path_value(manifest: dict[str, Any]) -> object:
    for key in ("archive_path", "candidate_archive_path", "new_archive_path"):
        value = manifest.get(key)
        if value:
            return value
    return None


def _resolve_manifest_path(value: object) -> Path | None:
    if isinstance(value, Path):
        path = value
    elif isinstance(value, str) and value.strip():
        path = Path(value)
    else:
        return None
    return path if path.is_absolute() else REPO_ROOT / path


def _runtime_custody_blockers(
    manifest: dict[str, Any],
    *,
    submission_dir: Path,
    archive_path: Path,
) -> list[str]:
    blockers: list[str] = []
    runtime_manifest = manifest.get("local_runtime_custody")
    if runtime_manifest is None:
        runtime_manifest = manifest.get("runtime_manifest")
    if not isinstance(runtime_manifest, dict):
        return ["local_runtime_custody_missing"]
    if runtime_manifest.get("schema_version") != LOCAL_RUNTIME_CUSTODY_SCHEMA:
        blockers.append("local_runtime_custody_schema_mismatch")
    if runtime_manifest.get("submission_dir") != manifest_path(submission_dir):
        blockers.append("local_runtime_custody_submission_dir_mismatch")

    archive_record = runtime_manifest.get("archive")
    if not isinstance(archive_record, dict):
        blockers.append("local_runtime_custody_archive_record_missing")
    else:
        expected_archive = {
            "path": manifest_path(archive_path),
            "bytes": archive_path.stat().st_size if archive_path.is_file() else None,
            "sha256": sha256_of(archive_path) if archive_path.is_file() else None,
        }
        for key, expected in expected_archive.items():
            if archive_record.get(key) != expected:
                blockers.append(f"local_runtime_custody_archive_{key}_mismatch")

    try:
        actual = collect_local_runtime_custody(submission_dir, archive_path=archive_path)
    except FileNotFoundError as exc:
        blockers.append(f"local_runtime_custody_file_missing:{exc}")
        return blockers

    actual_files = {row["relative_path"]: row for row in actual["files"]}
    manifest_files = runtime_manifest.get("files")
    if not isinstance(manifest_files, list):
        blockers.append("local_runtime_custody_files_missing")
        manifest_files_by_rel: dict[str, dict[str, Any]] = {}
    else:
        manifest_files_by_rel = {
            str(row.get("relative_path")): row
            for row in manifest_files
            if isinstance(row, dict)
        }

    actual_relpaths = set(actual_files)
    manifest_relpaths = set(manifest_files_by_rel)
    for rel in sorted(actual_relpaths - manifest_relpaths):
        blockers.append(f"local_runtime_custody_file_record_missing:{rel}")
    for rel in sorted(manifest_relpaths - actual_relpaths):
        blockers.append(f"local_runtime_custody_file_record_stale:{rel}")
    if runtime_manifest.get("file_count") != len(actual_files):
        blockers.append("local_runtime_custody_file_count_mismatch")

    for rel in sorted(actual_relpaths):
        expected = actual_files[rel]
        observed = manifest_files_by_rel.get(rel)
        if observed is None:
            continue
        for key in ("path", "bytes", "sha256", "mode", "executable"):
            if observed.get(key) != expected[key]:
                blockers.append(f"local_runtime_custody_{rel}_{key}_mismatch")
    for rel in REQUIRED_RUNTIME_FILES:
        if rel not in actual_files:
            blockers.append(f"local_runtime_custody_required_file_missing:{rel}")
    inflate_sh_record = manifest_files_by_rel.get("inflate.sh")
    if isinstance(inflate_sh_record, dict) and inflate_sh_record.get("executable") is not True:
        blockers.append("local_runtime_custody_inflate_sh_not_executable")

    expected_tree = actual["runtime_tree_sha256"]
    manifest_tree = runtime_manifest.get("runtime_tree_sha256")
    if manifest_tree != expected_tree:
        blockers.append("local_runtime_custody_runtime_tree_sha256_mismatch")
    top_level_tree = manifest.get("runtime_tree_sha256")
    if top_level_tree != expected_tree:
        blockers.append("runtime_tree_sha256_missing_or_mismatch")
    if not _is_sha256(manifest_tree):
        blockers.append("local_runtime_custody_runtime_tree_sha256_invalid")
    return blockers


def _no_op_detector_blockers(manifest: dict[str, Any]) -> list[str]:
    no_op = manifest.get("no_op_detector")
    if not isinstance(no_op, dict):
        return ["sidecar_no_op_detector_missing"]
    blockers: list[str] = []
    old_sha = no_op.get("old_inner_sidecar_sha256")
    new_sha = no_op.get("new_inner_sidecar_sha256")
    if not _is_sha256(old_sha):
        blockers.append("sidecar_no_op_old_sha256_missing_or_invalid")
    if not _is_sha256(new_sha):
        blockers.append("sidecar_no_op_new_sha256_missing_or_invalid")
    if _is_sha256(old_sha) and _is_sha256(new_sha) and old_sha == new_sha:
        blockers.append("sidecar_no_op_hashes_equal")
    if no_op.get("sidecar_changed") is not True:
        blockers.append("sidecar_no_op_detector_not_changed")
    baseline_output_sha = no_op.get("baseline_output_sha256")
    candidate_output_sha = no_op.get("candidate_output_sha256")
    if not _is_sha256(baseline_output_sha):
        blockers.append("sidecar_no_op_baseline_output_sha256_missing_or_invalid")
    if not _is_sha256(candidate_output_sha):
        blockers.append("sidecar_no_op_candidate_output_sha256_missing_or_invalid")
    if (
        _is_sha256(baseline_output_sha)
        and _is_sha256(candidate_output_sha)
        and baseline_output_sha == candidate_output_sha
    ):
        blockers.append("sidecar_no_op_runtime_outputs_equal")
    if no_op.get("runtime_output_changed") is not True:
        blockers.append("sidecar_no_op_runtime_output_not_proven_changed")
    return blockers


def _dispatch_custody_record_blockers(
    manifest: dict[str, Any],
    *,
    archive_path: Path,
    runtime_tree_sha256: str | None,
) -> list[str]:
    """Require structured pre-dispatch custody before readiness can flip true."""

    archive_sha = sha256_of(archive_path) if archive_path.is_file() else None
    blockers: list[str] = []
    dispatch_claim = manifest.get("dispatch_claim")
    if not isinstance(dispatch_claim, dict):
        blockers.append("dispatch_claim_record_missing")
    else:
        if dispatch_claim.get("schema_version") != SIDECAR_DISPATCH_CLAIM_SCHEMA:
            blockers.append("dispatch_claim_schema_mismatch")
        if dispatch_claim.get("lane_id") != SIDECAR_LANE_ID:
            blockers.append("dispatch_claim_lane_id_mismatch")
        if archive_sha is not None and dispatch_claim.get("archive_sha256") != archive_sha:
            blockers.append("dispatch_claim_archive_sha256_mismatch")
        if runtime_tree_sha256 and dispatch_claim.get("runtime_tree_sha256") != runtime_tree_sha256:
            blockers.append("dispatch_claim_runtime_tree_sha256_mismatch")
        status = dispatch_claim.get("status")
        if not isinstance(status, str) or _is_terminal_dispatch_status(status):
            blockers.append("dispatch_claim_status_not_active")
        blockers.extend(_dispatch_claim_live_ledger_blockers(dispatch_claim))

    preflight = manifest.get("exact_eval_preflight")
    if not isinstance(preflight, dict):
        blockers.append("exact_eval_preflight_record_missing")
    else:
        if preflight.get("schema_version") != SIDECAR_EXACT_EVAL_PREFLIGHT_SCHEMA:
            blockers.append("exact_eval_preflight_schema_mismatch")
        if preflight.get("passed") is not True:
            blockers.append("exact_eval_preflight_not_passed")
        if archive_sha is not None and preflight.get("archive_sha256") != archive_sha:
            blockers.append("exact_eval_preflight_archive_sha256_mismatch")
        if runtime_tree_sha256 and preflight.get("runtime_tree_sha256") != runtime_tree_sha256:
            blockers.append("exact_eval_preflight_runtime_tree_sha256_mismatch")
        if not preflight.get("command"):
            blockers.append("exact_eval_preflight_command_missing")
        if not _is_sha256(preflight.get("report_sha256")):
            blockers.append("exact_eval_preflight_report_sha256_missing_or_invalid")

    return blockers


def _is_terminal_dispatch_status(status: str) -> bool:
    return any(status.startswith(prefix) for prefix in TERMINAL_DISPATCH_STATUS_PREFIXES)


def _parse_claim_timestamp(value: object) -> dt.datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.UTC)
    return parsed.astimezone(dt.UTC)


def _parse_dispatch_claim_rows(path: Path) -> list[dict[str, str]]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return []
    rows: list[dict[str, str]] = []
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        if "timestamp_utc" in line and "lane_id" in line:
            continue
        if set(line.replace("|", "").strip()) <= {"-"}:
            continue
        cells = [cell.strip().replace("\\|", "|") for cell in line.strip("|").split("|")]
        if len(cells) < 8:
            continue
        rows.append(
            {
                "timestamp_utc": cells[0],
                "agent": cells[1],
                "lane_id": cells[2],
                "platform": cells[3],
                "instance_job_id": cells[4],
                "predicted_eta_utc": cells[5],
                "status": cells[6],
                "notes": cells[7],
            }
        )
    return rows


def _latest_dispatch_claim_row(
    *,
    claims_path: Path,
    lane_id: str,
    instance_job_id: str,
) -> dict[str, str] | None:
    latest: dict[str, str] | None = None
    latest_ts: dt.datetime | None = None
    for row in _parse_dispatch_claim_rows(claims_path):
        if row.get("lane_id") != lane_id:
            continue
        if row.get("instance_job_id") != instance_job_id:
            continue
        ts = _parse_claim_timestamp(row.get("timestamp_utc"))
        if latest is None or latest_ts is None or (ts is not None and ts > latest_ts):
            latest = row
            latest_ts = ts
    return latest


def _dispatch_claim_live_ledger_blockers(dispatch_claim: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    instance_job_id = dispatch_claim.get("instance_job_id")
    platform = dispatch_claim.get("platform")
    timestamp_utc = dispatch_claim.get("timestamp_utc")
    claims_path = _resolve_manifest_path(
        dispatch_claim.get("claims_path") or manifest_path(DEFAULT_DISPATCH_CLAIMS_PATH)
    )
    if not isinstance(instance_job_id, str) or not instance_job_id:
        blockers.append("dispatch_claim_instance_job_id_missing")
    if not isinstance(platform, str) or not platform:
        blockers.append("dispatch_claim_platform_missing")
    if _parse_claim_timestamp(timestamp_utc) is None:
        blockers.append("dispatch_claim_timestamp_utc_missing_or_invalid")
    if claims_path is None or not claims_path.is_file():
        blockers.append("dispatch_claim_ledger_missing")
        return blockers
    if blockers:
        return blockers

    latest = _latest_dispatch_claim_row(
        claims_path=claims_path,
        lane_id=SIDECAR_LANE_ID,
        instance_job_id=instance_job_id,
    )
    if latest is None:
        blockers.append("dispatch_claim_ledger_matching_row_missing")
        return blockers
    latest_status = latest.get("status", "")
    if _is_terminal_dispatch_status(latest_status):
        blockers.append("dispatch_claim_ledger_latest_row_terminal")
    if latest.get("timestamp_utc") != timestamp_utc:
        blockers.append("dispatch_claim_ledger_timestamp_mismatch")
    if latest.get("platform") != platform:
        blockers.append("dispatch_claim_ledger_platform_mismatch")
    if latest_status != dispatch_claim.get("status"):
        blockers.append("dispatch_claim_ledger_status_mismatch")
    latest_ts = _parse_claim_timestamp(latest.get("timestamp_utc"))
    if latest_ts is None:
        blockers.append("dispatch_claim_ledger_timestamp_invalid")
    else:
        ttl_hours = float(dispatch_claim.get("ttl_hours", 24.0))
        age_hours = (dt.datetime.now(dt.UTC) - latest_ts).total_seconds() / 3600.0
        if age_hours > ttl_hours:
            blockers.append("dispatch_claim_ledger_row_stale")
    return blockers


def _runtime_smoke_evidence_blockers(
    manifest: dict[str, Any],
    *,
    archive_path: Path,
    runtime_tree_sha256: str,
) -> list[str]:
    if manifest.get("runtime_smoke_checked") is not True:
        return ["runtime_smoke_not_checked"]
    evidence = manifest.get("runtime_smoke_evidence")
    if not isinstance(evidence, dict):
        return ["runtime_smoke_evidence_missing"]
    blockers: list[str] = []
    if evidence.get("runtime_surface") != "inflate_sh_exact_signature":
        blockers.append("runtime_smoke_evidence_not_inflate_sh_exact_signature")
    command = evidence.get("command")
    if not isinstance(command, (str, list)) or not command:
        blockers.append("runtime_smoke_evidence_command_missing")
    if evidence.get("exit_code") != 0:
        blockers.append("runtime_smoke_evidence_exit_code_nonzero")
    if evidence.get("archive_sha256") != sha256_of(archive_path):
        blockers.append("runtime_smoke_evidence_archive_sha256_mismatch")
    if evidence.get("runtime_tree_sha256") != runtime_tree_sha256:
        blockers.append("runtime_smoke_evidence_runtime_tree_sha256_mismatch")
    output_digest = evidence.get("output_digest_sha256")
    if not _is_sha256(output_digest):
        blockers.append("runtime_smoke_evidence_output_digest_missing_or_invalid")
    output_bytes = evidence.get("output_bytes")
    expected_output_bytes = evidence.get("expected_output_bytes")
    if not isinstance(expected_output_bytes, int) or expected_output_bytes <= 0:
        blockers.append("runtime_smoke_evidence_expected_output_bytes_missing")
    elif output_bytes != expected_output_bytes:
        blockers.append("runtime_smoke_evidence_output_size_mismatch")
    evidence_blockers = evidence.get("blockers")
    if isinstance(evidence_blockers, list) and evidence_blockers:
        blockers.append("runtime_smoke_evidence_has_blockers")
    return blockers


def _member_section_proof_blockers(
    manifest: dict[str, Any],
    *,
    archive_path: Path,
) -> list[str]:
    proof = manifest.get("member_section_proof")
    if not isinstance(proof, dict):
        return ["member_section_proof_missing"]
    blockers: list[str] = []
    if proof.get("schema_version") != MEMBER_SECTION_PROOF_SCHEMA:
        blockers.append("member_section_proof_schema_mismatch")
    try:
        actual = collect_member_section_proof(archive_path)
    except (FileNotFoundError, ValueError, zipfile.BadZipFile) as exc:
        return [f"member_section_proof_unreadable:{exc}"]

    if proof.get("archive") != actual["archive"]:
        blockers.append("member_section_proof_archive_mismatch")
    if proof.get("member_count") != 1:
        blockers.append("member_section_proof_member_count_mismatch")
    if proof.get("single_member_name") != "x":
        blockers.append("member_section_proof_single_member_name_mismatch")
    if proof.get("members") != actual["members"]:
        blockers.append("member_section_proof_members_mismatch")
    if proof.get("inner_sections") != actual["inner_sections"]:
        blockers.append("member_section_proof_inner_sections_mismatch")

    no_op = manifest.get("no_op_detector")
    inner_sections = proof.get("inner_sections")
    if isinstance(no_op, dict) and isinstance(inner_sections, dict):
        sidecar_blob = inner_sections.get("sidecar_blob")
        if isinstance(sidecar_blob, dict):
            if sidecar_blob.get("sha256") != no_op.get("new_inner_sidecar_sha256"):
                blockers.append("member_section_proof_sidecar_sha256_mismatch")
            if (
                "new_sidecar_bytes" in manifest
                and sidecar_blob.get("bytes") != manifest.get("new_sidecar_bytes")
            ):
                blockers.append("member_section_proof_sidecar_bytes_mismatch")
    return blockers


def _encode_sidecar_from_state_payload(
    payload: dict[str, Any],
    *,
    encode_format: str,
) -> bytes:
    dims, delta_idx, _ = sidecar_choice_arrays_from_payload(payload)
    if encode_format == "n_pairs_600":
        return encode_sidecar_n_pairs(dims, delta_idx)
    if encode_format == "packed_661":
        return encode_sidecar_huff_enum(dims, delta_idx)
    raise ValueError(f"unknown encode_format: {encode_format!r}")


def _sidecar_choice_state_blockers(manifest: dict[str, Any]) -> list[str]:
    record = manifest.get("sidecar_choice_state")
    full_search = manifest.get("full_non_smoke_search") is True
    if not isinstance(record, dict):
        return ["sidecar_choice_state_missing_for_full_search"] if full_search else []

    blockers: list[str] = []
    if record.get("schema_version") != SIDECAR_CHOICE_STATE_SCHEMA:
        blockers.append("sidecar_choice_state_schema_mismatch")
    state_path = _resolve_manifest_path(record.get("path"))
    if state_path is None:
        blockers.append("sidecar_choice_state_path_missing")
        return blockers
    if not state_path.is_file():
        blockers.append(f"sidecar_choice_state_missing:{manifest_path(state_path)}")
        return blockers
    actual_sha = sha256_of(state_path)
    if record.get("sha256") != actual_sha:
        blockers.append("sidecar_choice_state_sha256_mismatch")
    try:
        payload = _load_sidecar_choice_state_payload(state_path)
        dims, delta_idx, searched_mask = sidecar_choice_arrays_from_payload(payload)
    except (FileNotFoundError, ValueError) as exc:
        blockers.append(f"sidecar_choice_state_unreadable:{exc}")
        return blockers

    total_pairs = int(manifest.get("total_pairs", N_PAIRS))
    completed = int(searched_mask.sum())
    if payload.get("schema_version") != SIDECAR_CHOICE_STATE_SCHEMA:
        blockers.append("sidecar_choice_state_payload_schema_mismatch")
    if payload.get("lane_id") != SIDECAR_LANE_ID:
        blockers.append("sidecar_choice_state_lane_id_mismatch")
    if payload.get("old_archive_sha256") != manifest.get("old_archive_sha256"):
        blockers.append("sidecar_choice_state_old_archive_sha256_mismatch")
    no_op = manifest.get("no_op_detector")
    if isinstance(no_op, dict) and payload.get("old_sidecar_sha256") != no_op.get(
        "old_inner_sidecar_sha256"
    ):
        blockers.append("sidecar_choice_state_old_sidecar_sha256_mismatch")
    if payload.get("search_signal") != manifest.get("search_signal"):
        blockers.append("sidecar_choice_state_search_signal_mismatch")
    if payload.get("search_device") != manifest.get("search_device"):
        blockers.append("sidecar_choice_state_search_device_mismatch")
    if payload.get("encode_format") != manifest.get("encode_format"):
        blockers.append("sidecar_choice_state_encode_format_mismatch")
    if payload.get("total_pairs") != total_pairs:
        blockers.append("sidecar_choice_state_total_pairs_mismatch")
    if record.get("total_pairs") != payload.get("total_pairs"):
        blockers.append("sidecar_choice_state_record_total_pairs_mismatch")
    if record.get("n_pairs_completed_total") != completed:
        blockers.append("sidecar_choice_state_record_completed_count_mismatch")
    if payload.get("n_pairs_completed_total") != completed:
        blockers.append("sidecar_choice_state_payload_completed_count_mismatch")
    full_coverage = completed == total_pairs
    if record.get("full_coverage") is not full_coverage:
        blockers.append("sidecar_choice_state_record_full_coverage_mismatch")
    if payload.get("full_coverage") is not full_coverage:
        blockers.append("sidecar_choice_state_payload_full_coverage_mismatch")
    if full_search and not full_coverage:
        blockers.append("sidecar_choice_state_incomplete_for_full_search")

    try:
        records_by_pair = pair_search_records_from_payload(payload)
    except ValueError as exc:
        blockers.append(f"sidecar_pair_search_records_unreadable:{exc}")
        records_by_pair = {}
    searched_pairs = [int(i) for i in np.flatnonzero(searched_mask)]
    if completed and payload.get("pair_search_records_schema") != SIDECAR_PAIR_SEARCH_RECORDS_SCHEMA:
        blockers.append("sidecar_pair_search_records_schema_missing_or_mismatch")
    if record.get("pair_search_records_schema") != payload.get("pair_search_records_schema"):
        blockers.append("sidecar_choice_state_record_pair_search_schema_mismatch")
    missing_records = [i for i in searched_pairs if i not in records_by_pair]
    if missing_records:
        blockers.append(
            "sidecar_pair_search_records_missing_for_completed_pairs:"
            f"{len(missing_records)}"
        )
    unsafe_records: list[int] = []
    mismatched_records: list[int] = []
    for pair_index, pair_record in sorted(records_by_pair.items()):
        if pair_index < 0 or pair_index >= total_pairs:
            blockers.append(f"sidecar_pair_search_record_pair_index_out_of_range:{pair_index}")
            continue
        if pair_index not in searched_pairs:
            blockers.append(f"sidecar_pair_search_record_for_unsearched_pair:{pair_index}")
            continue
        if pair_record.get("schema_version") != SIDECAR_PAIR_SEARCH_RECORDS_SCHEMA:
            mismatched_records.append(pair_index)
        if pair_record.get("search_signal") != payload.get("search_signal"):
            mismatched_records.append(pair_index)
        if pair_record.get("search_device") != payload.get("search_device"):
            mismatched_records.append(pair_index)
        if pair_record.get("best_dim") != int(dims[pair_index]):
            mismatched_records.append(pair_index)
        if pair_record.get("best_delta_idx") != int(delta_idx[pair_index]):
            mismatched_records.append(pair_index)
        if pair_record.get("dispatch_safe_scalar_equivalent") is not True:
            unsafe_records.append(pair_index)
    if mismatched_records:
        blockers.append(
            "sidecar_pair_search_records_context_or_choice_mismatch:"
            f"{len(set(mismatched_records))}"
        )
    if unsafe_records:
        blockers.append(
            "sidecar_pair_search_records_not_scalar_equivalent:"
            f"{len(unsafe_records)}"
        )
    safe_count = sum(
        1
        for i in searched_pairs
        if records_by_pair.get(i, {}).get("dispatch_safe_scalar_equivalent") is True
    )
    if record.get("n_pairs_with_search_records") not in (None, len(records_by_pair)):
        blockers.append("sidecar_choice_state_record_pair_search_count_mismatch")
    if payload.get("n_pairs_with_search_records") != len(records_by_pair):
        blockers.append("sidecar_choice_state_payload_pair_search_count_mismatch")
    if payload.get("n_pairs_dispatch_safe_scalar_equivalent") != safe_count:
        blockers.append("sidecar_choice_state_payload_dispatch_safe_count_mismatch")
    if record.get("n_pairs_dispatch_safe_scalar_equivalent") not in (None, safe_count):
        blockers.append("sidecar_choice_state_record_dispatch_safe_count_mismatch")
    all_safe = safe_count == completed and not unsafe_records and not missing_records
    if payload.get("all_searched_pairs_dispatch_safe_scalar_equivalent") is not all_safe:
        blockers.append("sidecar_choice_state_payload_dispatch_safe_flag_mismatch")
    if record.get("all_searched_pairs_dispatch_safe_scalar_equivalent") not in (None, all_safe):
        blockers.append("sidecar_choice_state_record_dispatch_safe_flag_mismatch")
    if payload.get("search_device") == "mps":
        blockers.append("sidecar_choice_state_mps_proxy_search_advisory_only")

    try:
        encoded_sidecar = _encode_sidecar_from_state_payload(
            payload,
            encode_format=str(manifest.get("encode_format")),
        )
    except ValueError as exc:
        blockers.append(f"sidecar_choice_state_encode_failed:{exc}")
    else:
        encoded_sha = hashlib.sha256(encoded_sidecar).hexdigest()
        encoded_bytes = len(encoded_sidecar)
        if isinstance(no_op, dict) and encoded_sha != no_op.get(
            "new_inner_sidecar_sha256"
        ):
            blockers.append("sidecar_choice_state_encoded_sidecar_sha256_mismatch")
        if (
            "new_sidecar_bytes" in manifest
            and encoded_bytes != manifest.get("new_sidecar_bytes")
        ):
            blockers.append("sidecar_choice_state_encoded_sidecar_bytes_mismatch")
    # Keep locals live for validation readability and to make static checkers
    # notice that both arrays are parsed even when only the mask drives coverage.
    _ = (dims, delta_idx)
    return blockers


def _manifest_top_level_dispatch_blockers(manifest: dict[str, Any]) -> list[str]:
    """Top-level proxy/evidence blockers that do not require file IO."""

    blockers: list[str] = []
    if manifest.get("search_device") == "mps":
        blockers.append("mps_search_device_advisory_only_not_exact_eval_ready")
    return blockers


def run_local_runtime_smoke(
    submission_dir: Path,
    *,
    archive_path: Path,
    smoke_dir: Path,
    smoke_pairs: int = 1,
    runtime_tree_sha256: str,
) -> dict[str, Any]:
    """Run a bounded local runtime smoke without emitting a full 600-pair raw.

    The smoke imports the candidate ``inflate.py` and overrides ``N_PAIRS`` in
    that process only. Exact eval still uses the unmodified default runtime.
    """

    if smoke_pairs < 1:
        raise ValueError("smoke_pairs must be >= 1")
    smoke_dir.mkdir(parents=True, exist_ok=True)
    data_dir = smoke_dir / "data"
    out_dir = smoke_dir / "out"
    data_dir.mkdir(exist_ok=True)
    out_dir.mkdir(exist_ok=True)
    src_x = data_dir / "x"
    raw_out = out_dir / "0.raw"
    evidence_path = smoke_dir / "runtime_smoke_evidence.json"

    with zipfile.ZipFile(archive_path, "r") as zf:
        names = zf.namelist()
        if names != ["x"]:
            raise ValueError(f"runtime smoke expected single member ['x'], got {names}")
        src_x.write_bytes(zf.read("x"))

    runner = (
        "import importlib.util, sys\n"
        "from pathlib import Path\n"
        "inflate_py = Path(sys.argv[1]).resolve()\n"
        "src = sys.argv[2]\n"
        "dst = sys.argv[3]\n"
        "n_pairs = int(sys.argv[4])\n"
        "spec = importlib.util.spec_from_file_location('a1_smoke_inflate', inflate_py)\n"
        "assert spec is not None and spec.loader is not None\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        "mod.N_PAIRS = n_pairs\n"
        "frames = mod.inflate(src, dst)\n"
        "expected = n_pairs * 2\n"
        "if frames != expected:\n"
        "    raise SystemExit(f'expected {expected} frames, got {frames}')\n"
    )
    python_path = Path(sys.executable)
    command = [
        manifest_path(python_path),
        "-c",
        runner,
        manifest_path(submission_dir / "inflate.py"),
        manifest_path(src_x),
        manifest_path(raw_out),
        str(smoke_pairs),
    ]
    executed = [str(python_path), "-c", runner, str(submission_dir / "inflate.py"), str(src_x), str(raw_out), str(smoke_pairs)]
    started = dt.datetime.now(dt.UTC)
    proc = subprocess.run(
        executed,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    completed = dt.datetime.now(dt.UTC)
    output_digest = sha256_of(raw_out) if raw_out.is_file() else None
    output_bytes = raw_out.stat().st_size if raw_out.is_file() else 0
    expected_bytes = smoke_pairs * 2 * CAMERA_H * CAMERA_W * 3
    exit_code = proc.returncode
    blockers: list[str] = []
    if exit_code != 0:
        blockers.append("runtime_smoke_subprocess_failed")
    if output_bytes != expected_bytes:
        blockers.append("runtime_smoke_output_size_mismatch")
        exit_code = exit_code or 1
    evidence = {
        "schema_version": RUNTIME_SMOKE_SCHEMA,
        "runtime_surface": "inflate_py_import_smoke",
        "command": command,
        "exit_code": exit_code,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
        "started_at_utc": started.isoformat(),
        "completed_at_utc": completed.isoformat(),
        "elapsed_seconds": (completed - started).total_seconds(),
        "smoke_pairs": smoke_pairs,
        "expected_frames": smoke_pairs * 2,
        "archive_sha256": sha256_of(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "runtime_tree_sha256": runtime_tree_sha256,
        "input_member_path": manifest_path(src_x),
        "output_raw_path": manifest_path(raw_out),
        "output_bytes": output_bytes,
        "expected_output_bytes": expected_bytes,
        "output_digest_sha256": output_digest,
        "blockers": blockers,
        "contract": (
            "imports candidate inflate.py and overrides N_PAIRS only in the "
            "smoke process; exact eval uses inflate.sh with default full N_PAIRS"
        ),
    }
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    evidence["evidence_path"] = manifest_path(evidence_path)
    return evidence


def run_exact_inflate_sh_smoke(
    submission_dir: Path,
    *,
    archive_path: Path,
    smoke_dir: Path,
    runtime_tree_sha256: str,
    video_name: str = "0.mkv",
    expected_output_bytes: int | None = N_PAIRS * 2 * CAMERA_H * CAMERA_W * 3,
    timeout_seconds: int | None = None,
) -> dict[str, Any]:
    """Run the candidate runtime through the contest ``inflate.sh`` signature.

    This is intentionally separate from :func:`run_local_runtime_smoke`: the
    local smoke bounds ``N_PAIRS`` by importing ``inflate.py`` directly, while
    this path invokes ``inflate.sh <data_dir> <output_dir> <file_list>`` exactly.
    It can be expensive for the real A1 runtime because it decodes the full
    packet, so callers must opt into it explicitly.
    """

    smoke_dir.mkdir(parents=True, exist_ok=True)
    data_dir = smoke_dir / "data"
    out_dir = smoke_dir / "out"
    data_dir.mkdir(exist_ok=True)
    out_dir.mkdir(exist_ok=True)
    src_x = data_dir / "x"
    file_list = smoke_dir / "file_list.txt"
    evidence_path = smoke_dir / "exact_inflate_sh_smoke_evidence.json"

    with zipfile.ZipFile(archive_path, "r") as zf:
        names = zf.namelist()
        if names != ["x"]:
            raise ValueError(f"exact inflate smoke expected single member ['x'], got {names}")
        src_x.write_bytes(zf.read("x"))
    file_list.write_text(f"{video_name}\n")

    inflate_sh = submission_dir / "inflate.sh"
    command = [
        manifest_path(inflate_sh),
        manifest_path(data_dir),
        manifest_path(out_dir),
        manifest_path(file_list),
    ]
    executed = [
        str(inflate_sh),
        str(data_dir),
        str(out_dir),
        str(file_list),
    ]
    env = os.environ.copy()
    env["PYTHON"] = str(Path(sys.executable))
    started = dt.datetime.now(dt.UTC)
    try:
        proc = subprocess.run(
            executed,
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout_seconds,
            env=env,
        )
        timed_out = False
        exit_code = proc.returncode
        stdout = proc.stdout
        stderr = proc.stderr
    except subprocess.TimeoutExpired as exc:
        timed_out = True
        exit_code = 124
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
    completed = dt.datetime.now(dt.UTC)

    base = Path(video_name).stem
    raw_out = out_dir / f"{base}.raw"
    output_digest = sha256_of(raw_out) if raw_out.is_file() else None
    output_bytes = raw_out.stat().st_size if raw_out.is_file() else 0
    blockers: list[str] = []
    if timed_out:
        blockers.append("runtime_smoke_exact_inflate_sh_timed_out")
    if exit_code != 0:
        blockers.append("runtime_smoke_exact_inflate_sh_failed")
    if not raw_out.is_file():
        blockers.append("runtime_smoke_exact_inflate_sh_output_missing")
        exit_code = exit_code or 1
    if expected_output_bytes is not None and output_bytes != expected_output_bytes:
        blockers.append("runtime_smoke_exact_inflate_sh_output_size_mismatch")
        exit_code = exit_code or 1

    evidence = {
        "schema_version": RUNTIME_SMOKE_SCHEMA,
        "runtime_surface": "inflate_sh_exact_signature",
        "command": command,
        "exit_code": exit_code,
        "stdout": stdout[-4000:],
        "stderr": stderr[-4000:],
        "started_at_utc": started.isoformat(),
        "completed_at_utc": completed.isoformat(),
        "elapsed_seconds": (completed - started).total_seconds(),
        "timed_out": timed_out,
        "timeout_seconds": timeout_seconds,
        "video_name": video_name,
        "archive_sha256": sha256_of(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "runtime_tree_sha256": runtime_tree_sha256,
        "input_member_path": manifest_path(src_x),
        "file_list_path": manifest_path(file_list),
        "output_raw_path": manifest_path(raw_out),
        "output_bytes": output_bytes,
        "expected_output_bytes": expected_output_bytes,
        "output_digest_sha256": output_digest,
        "blockers": blockers,
        "contract": (
            "invokes candidate inflate.sh with archive_dir, output_dir, "
            "and file_list positional arguments; no N_PAIRS override"
        ),
    }
    evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    evidence["evidence_path"] = manifest_path(evidence_path)
    return evidence


def collect_runtime_output_change_probe(
    *,
    baseline_submission_dir: Path,
    candidate_submission_dir: Path,
    baseline_archive_path: Path,
    candidate_archive_path: Path,
    probe_dir: Path,
    smoke_pairs: int,
    candidate_runtime_tree_sha256: str,
) -> dict[str, Any]:
    """Compare bounded baseline/candidate decoded outputs for no-op proof."""

    baseline_evidence = run_local_runtime_smoke(
        baseline_submission_dir,
        archive_path=baseline_archive_path,
        smoke_dir=probe_dir / "baseline",
        smoke_pairs=smoke_pairs,
        runtime_tree_sha256=candidate_runtime_tree_sha256,
    )
    candidate_evidence = run_local_runtime_smoke(
        candidate_submission_dir,
        archive_path=candidate_archive_path,
        smoke_dir=probe_dir / "candidate",
        smoke_pairs=smoke_pairs,
        runtime_tree_sha256=candidate_runtime_tree_sha256,
    )
    baseline_sha = baseline_evidence.get("output_digest_sha256")
    candidate_sha = candidate_evidence.get("output_digest_sha256")
    output_changed = (
        _is_sha256(baseline_sha)
        and _is_sha256(candidate_sha)
        and baseline_sha != candidate_sha
    )
    return {
        "schema_version": "a1_sidecar_runtime_output_change_probe_v1",
        "runtime_surface": "inflate_py_import_smoke",
        "smoke_pairs": smoke_pairs,
        "baseline_archive_sha256": sha256_of(baseline_archive_path),
        "candidate_archive_sha256": sha256_of(candidate_archive_path),
        "runtime_tree_sha256": candidate_runtime_tree_sha256,
        "baseline_output_sha256": baseline_sha,
        "candidate_output_sha256": candidate_sha,
        "runtime_output_changed": output_changed,
        "baseline_evidence": baseline_evidence,
        "candidate_evidence": candidate_evidence,
    }


def load_upstream_yuv420_to_rgb():
    """Load the upstream CPU-eval RGB conversion helper without patching it."""
    import importlib.util

    frame_utils_path = REPO_ROOT / "upstream" / "frame_utils.py"
    spec = importlib.util.spec_from_file_location(
        "pact_sidecar_upstream_frame_utils",
        frame_utils_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load upstream frame_utils.py from {frame_utils_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.yuv420_to_rgb


def load_a1_archive_components(archive_path: Path) -> dict[str, Any]:
    """Parse A1's archive into decoder_sd, raw latents (pre-sidecar), and
    sidecar_blob — using A1's own codec module."""
    sys.path.insert(0, str(A1_SUBMISSION_DIR / "src"))
    # Import A1's codec module
    import importlib
    import zipfile

    if "codec" in sys.modules:
        importlib.reload(sys.modules["codec"])
    if "model" in sys.modules:
        importlib.reload(sys.modules["model"])
    import codec  # type: ignore

    with zipfile.ZipFile(archive_path, "r") as zf:
        member = zf.namelist()[0]
        archive_bytes = zf.read(member)
    section_total = struct.unpack_from("<I", archive_bytes, 0)[0]
    decoder_blob = archive_bytes[4:section_total]
    latent_blob = archive_bytes[section_total : section_total + codec.LATENT_BLOB_LEN]
    sidecar_blob = archive_bytes[section_total + codec.LATENT_BLOB_LEN :]
    decoder_sd = codec.decode_decoder_compact(decoder_blob)
    latents_pre_sidecar = codec.decode_latents_compact(latent_blob)
    latents_with_sidecar = codec.apply_latent_sidecar(latents_pre_sidecar, sidecar_blob)
    return {
        "archive_bytes": archive_bytes,
        "section_total": section_total,
        "decoder_blob": decoder_blob,
        "latent_blob": latent_blob,
        "sidecar_blob_old": sidecar_blob,
        "decoder_sd": decoder_sd,
        "latents_pre_sidecar": latents_pre_sidecar,
        "latents_with_sidecar_old": latents_with_sidecar,
        "n_pairs": codec.N_PAIRS,
        "latent_dim": codec.LATENT_DIM,
        "base_channels": codec.BASE_CHANNELS,
        "eval_size": codec.EVAL_SIZE,
        "codec_module": codec,
    }


def infer_sidecar_choices_from_latents(components: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Infer ``(dim, delta_idx)`` sidecar choices from old decoded latents.

    This preserves the inherited PR101 sidecar for any pair a partial/smoke
    search does not revisit. The codec supports several compact sidecar wire
    layouts, so inferring from ``latents_with_sidecar_old - latents_pre_sidecar``
    is simpler and harder to desynchronize from the runtime decoder.
    """
    diff = (
        components["latents_with_sidecar_old"]
        - components["latents_pre_sidecar"]
    ).detach().cpu().numpy()
    dims = np.full(N_PAIRS, 255, dtype=np.int64)
    delta_idx = np.full(N_PAIRS, -1, dtype=np.int64)
    deltas = SIDECAR_DELTAS_X100.astype(np.float32) / 100.0
    for k in range(N_PAIRS):
        nz = np.flatnonzero(np.abs(diff[k]) > 1e-6)
        if nz.size == 0:
            continue
        if nz.size != 1:
            raise ValueError(
                f"old sidecar inference expected <=1 changed latent dim per pair; "
                f"pair={k} changed_dims={nz.tolist()}"
            )
        dim = int(nz[0])
        delta = float(diff[k, dim])
        matches = np.flatnonzero(np.isclose(deltas, delta, rtol=0.0, atol=1e-4))
        if matches.size != 1:
            raise ValueError(
                f"old sidecar delta {delta:.6f} for pair={k} dim={dim} "
                "does not match fixed PR99-PR103 delta vocabulary"
            )
        dims[k] = dim
        delta_idx[k] = int(matches[0])
    return dims, delta_idx


def _validate_sidecar_choice_arrays(
    dims: np.ndarray,
    delta_idx: np.ndarray,
    searched_mask: np.ndarray | None = None,
    *,
    n_pairs: int = N_PAIRS,
    latent_dim: int = LATENT_DIM,
) -> None:
    if dims.shape != (n_pairs,):
        raise ValueError(f"dims shape mismatch: {dims.shape} != {(n_pairs,)}")
    if delta_idx.shape != (n_pairs,):
        raise ValueError(f"delta_idx shape mismatch: {delta_idx.shape} != {(n_pairs,)}")
    if searched_mask is not None and searched_mask.shape != (n_pairs,):
        raise ValueError(
            f"searched_mask shape mismatch: {searched_mask.shape} != {(n_pairs,)}"
        )
    for k, (dim, didx) in enumerate(zip(dims, delta_idx, strict=True)):
        dim_i = int(dim)
        didx_i = int(didx)
        if dim_i == 255:
            if didx_i != -1:
                raise ValueError(f"pair={k} has no-op dim but delta_idx={didx_i}")
            continue
        if not (0 <= dim_i < latent_dim):
            raise ValueError(f"pair={k} dim out of range: {dim_i}")
        if not (0 <= didx_i < len(SIDECAR_DELTAS_X100)):
            raise ValueError(f"pair={k} delta_idx out of range: {didx_i}")


def _sidecar_choice_state_payload(
    *,
    dims: np.ndarray,
    delta_idx: np.ndarray,
    searched_mask: np.ndarray,
    pair_search_records: dict[int, dict[str, Any]] | None = None,
    old_archive_sha256: str,
    old_sidecar_sha256: str,
    search_signal: str,
    search_device: str,
    encode_format: str,
    total_pairs: int,
    latent_dim: int,
) -> dict[str, Any]:
    _validate_sidecar_choice_arrays(
        dims,
        delta_idx,
        searched_mask,
        n_pairs=total_pairs,
        latent_dim=latent_dim,
    )
    searched_pair_indices = [int(i) for i in np.flatnonzero(searched_mask)]
    records_by_pair = pair_search_records or {}
    records = [
        dict(records_by_pair[int(i)])
        for i in sorted(records_by_pair)
        if 0 <= int(i) < int(total_pairs)
    ]
    dispatch_safe_pairs = {
        int(record["pair_index"])
        for record in records
        if record.get("dispatch_safe_scalar_equivalent") is True
    }
    return {
        "schema_version": SIDECAR_CHOICE_STATE_SCHEMA,
        "lane_id": SIDECAR_LANE_ID,
        "old_archive_sha256": old_archive_sha256,
        "old_sidecar_sha256": old_sidecar_sha256,
        "search_signal": search_signal,
        "search_device": search_device,
        "encode_format": encode_format,
        "total_pairs": int(total_pairs),
        "latent_dim": int(latent_dim),
        "delta_vocabulary_x100": [int(x) for x in SIDECAR_DELTAS_X100.tolist()],
        "dims": [int(x) for x in dims.tolist()],
        "delta_idx": [int(x) for x in delta_idx.tolist()],
        "searched_mask": [bool(x) for x in searched_mask.tolist()],
        "searched_pair_indices": searched_pair_indices,
        "n_pairs_completed_total": len(searched_pair_indices),
        "full_coverage": len(searched_pair_indices) == int(total_pairs),
        "unsearched_pairs_preserve_old_sidecar": True,
        "pair_search_records_schema": SIDECAR_PAIR_SEARCH_RECORDS_SCHEMA,
        "pair_search_records": records,
        "n_pairs_with_search_records": len(records),
        "n_pairs_dispatch_safe_scalar_equivalent": len(
            dispatch_safe_pairs.intersection(searched_pair_indices)
        ),
        "all_searched_pairs_dispatch_safe_scalar_equivalent": all(
            int(i) in dispatch_safe_pairs for i in searched_pair_indices
        ),
    }


def write_sidecar_choice_state(
    path: Path,
    *,
    dims: np.ndarray,
    delta_idx: np.ndarray,
    searched_mask: np.ndarray,
    pair_search_records: dict[int, dict[str, Any]] | None = None,
    old_archive_sha256: str,
    old_sidecar_sha256: str,
    search_signal: str,
    search_device: str,
    encode_format: str,
    total_pairs: int,
    latent_dim: int,
) -> dict[str, Any]:
    """Write deterministic per-pair sidecar choices for resume and custody."""

    payload = _sidecar_choice_state_payload(
        dims=dims,
        delta_idx=delta_idx,
        searched_mask=searched_mask,
        pair_search_records=pair_search_records,
        old_archive_sha256=old_archive_sha256,
        old_sidecar_sha256=old_sidecar_sha256,
        search_signal=search_signal,
        search_device=search_device,
        encode_format=encode_format,
        total_pairs=total_pairs,
        latent_dim=latent_dim,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    tmp_path.write_bytes(_canonical_json_bytes(payload) + b"\n")
    tmp_path.replace(path)
    return payload


def _load_sidecar_choice_state_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise ValueError(f"sidecar choice state is not valid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"sidecar choice state must be a JSON object: {path}")
    return payload


def sidecar_choice_arrays_from_payload(
    payload: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    for key in ("dims", "delta_idx", "searched_mask"):
        if not isinstance(payload.get(key), list):
            raise ValueError(f"sidecar choice state {key} must be a JSON list")
    dims = np.asarray(payload["dims"], dtype=np.int64)
    delta_idx = np.asarray(payload["delta_idx"], dtype=np.int64)
    searched_mask = np.asarray(payload["searched_mask"], dtype=bool)
    total_pairs = int(payload.get("total_pairs", N_PAIRS))
    latent_dim = int(payload.get("latent_dim", LATENT_DIM))
    _validate_sidecar_choice_arrays(
        dims,
        delta_idx,
        searched_mask,
        n_pairs=total_pairs,
        latent_dim=latent_dim,
    )
    return dims, delta_idx, searched_mask


def pair_search_records_from_payload(payload: dict[str, Any]) -> dict[int, dict[str, Any]]:
    """Return validated per-pair search-provenance records from saved state."""

    raw_records = payload.get("pair_search_records")
    if raw_records is None:
        return {}
    if not isinstance(raw_records, list):
        raise ValueError("sidecar choice state pair_search_records must be a list")
    out: dict[int, dict[str, Any]] = {}
    for raw in raw_records:
        if not isinstance(raw, dict):
            raise ValueError("sidecar choice state pair_search_records entries must be objects")
        pair_index = raw.get("pair_index")
        if not isinstance(pair_index, int):
            raise ValueError("sidecar pair_search_record missing integer pair_index")
        if pair_index in out:
            raise ValueError(f"duplicate sidecar pair_search_record for pair {pair_index}")
        out[pair_index] = dict(raw)
    return out


def _sidecar_pair_search_record(
    *,
    pair_index: int,
    search_signal: str,
    search_device: str,
    requested_candidate_batch_size: int,
    candidate_batch_size: int,
    base_mse: float,
    best_mse: float,
    best_dim: int,
    best_delta_idx: int,
    scalar_reference_status: str,
) -> dict[str, Any]:
    dispatch_safe = scalar_reference_status in {
        "scalar_direct",
        "scalar_equivalent_profiled_this_pair",
        "scalar_equivalent_recheck",
    }
    return {
        "schema_version": SIDECAR_PAIR_SEARCH_RECORDS_SCHEMA,
        "pair_index": int(pair_index),
        "search_signal": search_signal,
        "search_device": search_device,
        "requested_candidate_batch_size": int(requested_candidate_batch_size),
        "candidate_batch_size": int(candidate_batch_size),
        "base_mse": float(base_mse),
        "best_mse": float(best_mse),
        "best_dim": int(best_dim),
        "best_delta_idx": int(best_delta_idx),
        "scalar_reference_status": scalar_reference_status,
        "dispatch_safe_scalar_equivalent": bool(dispatch_safe),
    }


def load_sidecar_choice_state(
    path: Path,
    *,
    old_archive_sha256: str,
    old_sidecar_sha256: str,
    search_signal: str,
    search_device: str,
    encode_format: str,
    total_pairs: int,
    latent_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Load a prior search state, refusing mixed custody/search contexts."""

    payload = _load_sidecar_choice_state_payload(path)
    expected = {
        "schema_version": SIDECAR_CHOICE_STATE_SCHEMA,
        "lane_id": SIDECAR_LANE_ID,
        "old_archive_sha256": old_archive_sha256,
        "old_sidecar_sha256": old_sidecar_sha256,
        "search_signal": search_signal,
        "search_device": search_device,
        "encode_format": encode_format,
        "total_pairs": int(total_pairs),
        "latent_dim": int(latent_dim),
        "delta_vocabulary_x100": [int(x) for x in SIDECAR_DELTAS_X100.tolist()],
    }
    mismatches = [
        key
        for key, expected_value in expected.items()
        if payload.get(key) != expected_value
    ]
    if mismatches:
        raise ValueError(
            "sidecar choice state context mismatch: " + ", ".join(sorted(mismatches))
        )
    dims, delta_idx, searched_mask = sidecar_choice_arrays_from_payload(payload)
    completed = [int(i) for i in np.flatnonzero(searched_mask)]
    if payload.get("searched_pair_indices") != completed:
        raise ValueError("sidecar choice state searched_pair_indices mismatch")
    if payload.get("n_pairs_completed_total") != len(completed):
        raise ValueError("sidecar choice state completed-count mismatch")
    if payload.get("full_coverage") is not (len(completed) == int(total_pairs)):
        raise ValueError("sidecar choice state full_coverage mismatch")
    pair_search_records_from_payload(payload)
    return dims, delta_idx, searched_mask, payload


def sidecar_choice_state_manifest_record(path: Path) -> dict[str, Any]:
    payload = _load_sidecar_choice_state_payload(path)
    records = pair_search_records_from_payload(payload)
    return {
        "schema_version": payload.get("schema_version"),
        "path": manifest_path(path),
        "sha256": sha256_of(path),
        "n_pairs_completed_total": payload.get("n_pairs_completed_total"),
        "total_pairs": payload.get("total_pairs"),
        "full_coverage": payload.get("full_coverage"),
        "searched_pair_indices": payload.get("searched_pair_indices"),
        "pair_search_records_schema": payload.get("pair_search_records_schema"),
        "n_pairs_with_search_records": payload.get("n_pairs_with_search_records", len(records)),
        "n_pairs_dispatch_safe_scalar_equivalent": payload.get(
            "n_pairs_dispatch_safe_scalar_equivalent"
        ),
        "all_searched_pairs_dispatch_safe_scalar_equivalent": payload.get(
            "all_searched_pairs_dispatch_safe_scalar_equivalent"
        ),
    }


def encode_sidecar_huff_enum(dims: np.ndarray, delta_idx: np.ndarray) -> bytes:
    """Encode sidecar in PR101's HUFF_ENUM format (607 bytes for typical mix).

    For simplicity in this re-search tool we emit the SIDECAR_PACKED_LEN (661 B)
    layout instead of HUFF_ENUM (607 B) — encoding HUFF_ENUM requires the full
    Huffman+combinatorial machinery. PACKED_LEN is universally decodable by
    A1's codec.py (line 428-440) and only adds ~54 bytes vs HUFF_ENUM. This
    keeps the search results actionable while accepting a small archive bloat.

    A more aggressive future version could use the canonical Huffman path.
    """
    # Build choices array: 0=no-op, 1+i*16+d=dim i with delta d
    choices = np.zeros(N_PAIRS, dtype=np.int64)
    for k in range(N_PAIRS):
        if dims[k] != 255 and delta_idx[k] >= 0:
            choices[k] = 1 + dims[k] * len(SIDECAR_DELTAS_X100) + delta_idx[k]
    # Pack as base-449 mixed-radix integer little-endian (matches codec.py:428)
    value = 0
    for k in reversed(range(N_PAIRS)):
        value = value * SIDECAR_BASE + int(choices[k])
    n_bytes = (value.bit_length() + 7) // 8
    raw = value.to_bytes(n_bytes, "little")
    # Pad to SIDECAR_PACKED_LEN (661) — actually codec.py expects EXACTLY
    # SIDECAR_PACKED_LEN=661. Pad with zeros if smaller.
    SIDECAR_PACKED_LEN = 661
    if len(raw) < SIDECAR_PACKED_LEN:
        raw = raw + b"\x00" * (SIDECAR_PACKED_LEN - len(raw))
    elif len(raw) > SIDECAR_PACKED_LEN:
        raise ValueError(
            f"encoded packed sidecar overflow: {len(raw)} > {SIDECAR_PACKED_LEN}"
        )
    return raw


def encode_sidecar_n_pairs(dims: np.ndarray, delta_idx: np.ndarray) -> bytes:
    """Encode sidecar in the simplest 600-byte uint8-per-pair format
    (codec.py:441 path). Each pair gets one byte = choices[k]. Total=600 B."""
    choices = np.zeros(N_PAIRS, dtype=np.uint8)
    for k in range(N_PAIRS):
        if dims[k] != 255 and delta_idx[k] >= 0:
            value = 1 + dims[k] * len(SIDECAR_DELTAS_X100) + delta_idx[k]
            if value > 255:
                # 1+27*16+15 = 1+432+15 = 448 — cannot fit in uint8 (max 255).
                # The N_PAIRS layout (600 B) has limited reach.
                raise ValueError(
                    f"sidecar value {value} > 255; N_PAIRS layout cannot encode "
                    f"dim={dims[k]} delta_idx={delta_idx[k]}; switch to PACKED_LEN"
                )
            choices[k] = value
    return choices.tobytes()


def _sidecar_pair_work_plan(
    *,
    pair_indices: list[int],
    searched_mask: np.ndarray,
    pair_search_records: dict[int, dict[str, Any]],
    recheck_unproven_pairs: bool,
) -> dict[str, Any]:
    """Plan resumable sidecar work without touching decoder state.

    Legacy partial states created before per-pair scalar provenance existed
    have ``searched_mask=True`` but no dispatch-safe record. Exact-eval custody
    requires rechecking those pairs before promotion.
    """

    def _is_pair_dispatch_safe(pair_index: int) -> bool:
        return (
            pair_search_records.get(int(pair_index), {}).get(
                "dispatch_safe_scalar_equivalent"
            )
            is True
        )

    recheck_pairs = {
        int(k)
        for k in pair_indices
        if searched_mask[int(k)] and not _is_pair_dispatch_safe(int(k))
    } if recheck_unproven_pairs else set()
    skipped_already_completed = [
        int(k)
        for k in pair_indices
        if searched_mask[int(k)] and int(k) not in recheck_pairs
    ]
    work_pair_indices = [
        int(k)
        for k in pair_indices
        if (not searched_mask[int(k)]) or int(k) in recheck_pairs
    ]
    return {
        "recheck_pairs": recheck_pairs,
        "skipped_already_completed": skipped_already_completed,
        "work_pair_indices": work_pair_indices,
    }


def search_per_pair_proxy_mse(
    components: dict[str, Any],
    ground_truth_frames: np.ndarray,
    *,
    pair_indices: list[int] | None = None,
    log_path: Path | None = None,
    candidate_batch_size: int = 1,
    search_device: str = "cpu",
    initial_dims: np.ndarray | None = None,
    initial_delta_idx: np.ndarray | None = None,
    initial_searched_mask: np.ndarray | None = None,
    initial_pair_search_records: dict[int, dict[str, Any]] | None = None,
    state_path: Path | None = None,
    state_context: dict[str, Any] | None = None,
    max_search_seconds: float | None = None,
    candidate_batch_profile_sizes: list[int] | None = None,
    auto_candidate_batch_size: bool = False,
    recheck_unproven_pairs: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Greedy per-pair single-dim search using pixel MSE (fast proxy)."""
    import torch

    if max_search_seconds is not None and max_search_seconds <= 0:
        raise ValueError("max_search_seconds must be positive when supplied")
    if auto_candidate_batch_size and not candidate_batch_profile_sizes:
        raise ValueError(
            "auto_candidate_batch_size requires candidate_batch_profile_sizes"
        )
    decoder_sd = components["decoder_sd"]
    latents_base = components["latents_pre_sidecar"].clone()
    eval_h, eval_w = components["eval_size"]
    sys.path.insert(0, str(A1_SUBMISSION_DIR / "src"))
    if "model" not in sys.modules:
        import model as model_mod  # type: ignore
    else:
        model_mod = sys.modules["model"]
    decoder = model_mod.HNeRVDecoder(
        latent_dim=components["latent_dim"],
        base_channels=components["base_channels"],
        eval_size=(eval_h, eval_w),
    )
    device = torch.device(search_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--search-device cuda requested but CUDA is not available")
    if device.type == "mps" and (
        not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()
    ):
        raise RuntimeError("--search-device mps requested but MPS is not available")
    decoder.load_state_dict(decoder_sd)
    decoder.to(device)
    decoder.eval()
    latents_base = latents_base.to(device)

    n_pairs = components["n_pairs"]
    pair_indices = pair_indices if pair_indices is not None else list(range(n_pairs))
    if any(k < 0 or k >= n_pairs for k in pair_indices):
        raise ValueError(f"pair_indices outside n_pairs={n_pairs}: {pair_indices}")

    if initial_dims is None or initial_delta_idx is None:
        dims_out, delta_idx_out = infer_sidecar_choices_from_latents(components)
    else:
        dims_out = initial_dims.astype(np.int64, copy=True)
        delta_idx_out = initial_delta_idx.astype(np.int64, copy=True)
    if initial_searched_mask is None:
        searched_mask = np.zeros(n_pairs, dtype=bool)
    else:
        searched_mask = initial_searched_mask.astype(bool, copy=True)
    if dims_out.shape[0] != n_pairs or delta_idx_out.shape[0] != n_pairs:
        raise ValueError(
            f"old sidecar inference shape mismatch: dims={dims_out.shape} "
            f"delta_idx={delta_idx_out.shape} expected={n_pairs}"
        )
    _validate_sidecar_choice_arrays(
        dims_out,
        delta_idx_out,
        searched_mask,
        n_pairs=n_pairs,
        latent_dim=components["latent_dim"],
    )
    deltas = SIDECAR_DELTAS_X100.astype(np.float32) / 100.0

    log_lines = []
    t0 = time.time()
    pair_search_records = dict(initial_pair_search_records or {})

    def _is_pair_dispatch_safe(pair_index: int) -> bool:
        return (
            pair_search_records.get(int(pair_index), {}).get(
                "dispatch_safe_scalar_equivalent"
            )
            is True
        )

    work_plan = _sidecar_pair_work_plan(
        pair_indices=pair_indices,
        searched_mask=searched_mask,
        pair_search_records=pair_search_records,
        recheck_unproven_pairs=recheck_unproven_pairs,
    )
    recheck_pairs = work_plan["recheck_pairs"]
    recheck_pair_set = {int(i) for i in recheck_pairs}
    skipped_already_completed = work_plan["skipped_already_completed"]
    work_pair_indices = work_plan["work_pair_indices"]
    progress_every = max(1, min(25, len(work_pair_indices) or 1))
    completed_this_run = 0
    rechecked_unproven_completed_this_run = 0
    stopped_for_wall_clock = False
    active_candidate_batch_size = candidate_batch_size
    candidate_batch_profile: dict[str, Any] | None = None
    for n_done, k in enumerate(work_pair_indices, start=1):
        if (
            max_search_seconds is not None
            and completed_this_run > 0
            and (time.time() - t0) >= max_search_seconds
        ):
            stopped_for_wall_clock = True
            break
        # Ground truth uint8 frames for this pair
        gt = ground_truth_frames[k * 2 : k * 2 + 2]  # (2, CAMERA_H, CAMERA_W, 3)
        gt_eval = _resize_uint8_to_eval(
            gt,
            eval_h=eval_h,
            eval_w=eval_w,
        )  # (2, EVAL_H, EVAL_W, 3) float32
        base_lat = latents_base[k : k + 1].clone()
        if candidate_batch_profile_sizes is not None and candidate_batch_profile is None:
            candidate_batch_profile = profile_pair_proxy_mse_candidate_batches(
                decoder=decoder,
                base_lat=base_lat,
                gt_eval=gt_eval,
                eval_h=eval_h,
                eval_w=eval_w,
                deltas=deltas,
                candidate_batch_sizes=candidate_batch_profile_sizes,
            )
            if auto_candidate_batch_size:
                active_candidate_batch_size = int(
                    candidate_batch_profile["selected_candidate_batch_size"]
                )
                candidate_batch_profile["profiled_pair_index"] = int(k)
                print(
                    "[profile] candidate_batch_size "
                    f"selected={active_candidate_batch_size} "
                    f"reference=1 "
                    f"reason={candidate_batch_profile['selection_reason']}",
                    flush=True,
                )
        base_mse, best_mse, best_dim, best_didx = _best_pair_proxy_mse_candidate(
            decoder=decoder,
            base_lat=base_lat,
            gt_eval=gt_eval,
            eval_h=eval_h,
            eval_w=eval_w,
            deltas=deltas,
            candidate_batch_size=active_candidate_batch_size,
        )
        dims_out[k] = best_dim
        delta_idx_out[k] = best_didx
        searched_mask[k] = True
        if active_candidate_batch_size == 1:
            scalar_reference_status = "scalar_direct"
        elif (
            candidate_batch_profile is not None
            and candidate_batch_profile.get("profiled_pair_index") == int(k)
            and int(candidate_batch_profile.get("selected_candidate_batch_size", -1))
            == int(active_candidate_batch_size)
        ):
            scalar_reference_status = "scalar_equivalent_profiled_this_pair"
        else:
            scalar_reference_status = "non_scalar_unproven_for_this_pair"
        pair_search_records[int(k)] = _sidecar_pair_search_record(
            pair_index=int(k),
            search_signal="proxy_mse",
            search_device=device.type,
            requested_candidate_batch_size=candidate_batch_size,
            candidate_batch_size=active_candidate_batch_size,
            base_mse=base_mse,
            best_mse=best_mse,
            best_dim=best_dim,
            best_delta_idx=best_didx,
            scalar_reference_status=scalar_reference_status,
        )
        completed_this_run += 1
        if int(k) in recheck_pair_set:
            rechecked_unproven_completed_this_run += 1
        if state_path is not None:
            if state_context is None:
                raise ValueError("state_context is required when state_path is supplied")
            write_sidecar_choice_state(
                state_path,
                dims=dims_out,
                delta_idx=delta_idx_out,
                searched_mask=searched_mask,
                pair_search_records=pair_search_records,
                **state_context,
            )
        if n_done % progress_every == 0 or n_done == len(work_pair_indices):
            elapsed = time.time() - t0
            rate = completed_this_run / elapsed if elapsed > 0 else 0.0
            eta = (
                (len(work_pair_indices) - completed_this_run) / rate
                if rate > 0
                else 0.0
            )
            line = (
                f"[{completed_this_run}/{len(work_pair_indices)}] pair={k} "
                f"best_dim={best_dim} "
                f"best_didx={best_didx} mse_red={base_mse - best_mse:.4e} "
                f"rate={rate:.2f}/s eta={eta/60:.1f}min"
            )
            print(line, flush=True)
            log_lines.append(line)
    if state_path is not None:
        if state_context is None:
            raise ValueError("state_context is required when state_path is supplied")
        write_sidecar_choice_state(
            state_path,
            dims=dims_out,
            delta_idx=delta_idx_out,
            searched_mask=searched_mask,
            pair_search_records=pair_search_records,
            **state_context,
        )
    elapsed = time.time() - t0
    n_perturbed = int((dims_out != 255).sum())
    n_perturbed_searched = int(((dims_out != 255) & searched_mask).sum())
    print(
        f"[done] proxy-mse search: {completed_this_run} new pairs in {elapsed:.1f}s; "
        f"{n_perturbed_searched} searched-pair perturbations; "
        f"{n_perturbed} total perturbations after preserving old sidecar",
        flush=True,
    )
    if log_path:
        log_path.write_text("\n".join(log_lines) + "\n")
    remaining_unproven_after_run = sum(
        1 for i in np.flatnonzero(searched_mask) if not _is_pair_dispatch_safe(int(i))
    )
    return dims_out, delta_idx_out, {
        "search_signal": "proxy_mse",
        "n_pair_indices_requested": len(pair_indices),
        "n_pairs_skipped_already_completed": len(skipped_already_completed),
        "n_pairs_completed_this_run": completed_this_run,
        "n_pairs_completed_total": int(searched_mask.sum()),
        "n_pairs_perturbed": n_perturbed,
        "n_pairs_perturbed_searched": n_perturbed_searched,
        "unsearched_pairs_preserve_old_sidecar": True,
        "requested_candidate_batch_size": candidate_batch_size,
        "candidate_batch_size": active_candidate_batch_size,
        "auto_candidate_batch_size": auto_candidate_batch_size,
        "candidate_batch_profile": candidate_batch_profile,
        "recheck_unproven_pairs": recheck_unproven_pairs,
        "n_pairs_recheck_unproven_planned": len(recheck_pairs),
        "n_pairs_rechecked_unproven_completed_this_run": (
            rechecked_unproven_completed_this_run
        ),
        "remaining_unproven_records_after_run": remaining_unproven_after_run,
        "n_pairs_dispatch_safe_scalar_equivalent": sum(
            1 for i in np.flatnonzero(searched_mask) if _is_pair_dispatch_safe(int(i))
        ),
        "all_searched_pairs_dispatch_safe_scalar_equivalent": all(
            _is_pair_dispatch_safe(int(i)) for i in np.flatnonzero(searched_mask)
        ),
        "search_device": device.type,
        "elapsed_seconds": elapsed,
        "search_stopped_for_wall_clock": stopped_for_wall_clock,
        "searched_pair_indices": [int(i) for i in np.flatnonzero(searched_mask)],
    }


def _best_pair_proxy_mse_candidate(
    *,
    decoder: Any,
    base_lat: Any,
    gt_eval: np.ndarray,
    eval_h: int,
    eval_w: int,
    deltas: np.ndarray,
    candidate_batch_size: int,
) -> tuple[float, float, int, int]:
    """Return the best single-dim sidecar choice for one pair.

    The old implementation ran one decoder forward per candidate. This keeps
    identical greedy semantics while optionally evaluating perturbations in
    chunks. Large CPU chunks regressed the A1 smoke benchmark on 2026-05-09, so
    callers must opt into batching explicitly.
    """

    import torch

    if candidate_batch_size < 1:
        raise ValueError("candidate_batch_size must be >= 1")

    with torch.inference_mode():
        base_dec = decoder(base_lat).reshape(2, 3, eval_h, eval_w).detach().cpu().numpy()
    base_mse = _mse_uint8_after_clamp(base_dec, gt_eval)
    best_mse = base_mse
    best_dim = 255
    best_didx = -1

    if candidate_batch_size == 1:
        latent_dim = int(base_lat.shape[1])
        for d in range(latent_dim):
            for di, delta in enumerate(deltas):
                cand = base_lat.clone()
                cand[0, d] += float(delta)
                with torch.inference_mode():
                    cand_dec = decoder(cand).reshape(2, 3, eval_h, eval_w).detach().cpu().numpy()
                mse = _mse_uint8_after_clamp(cand_dec, gt_eval)
                if mse < best_mse:
                    best_mse = mse
                    best_dim = d
                    best_didx = di
        return base_mse, best_mse, best_dim, best_didx

    chunk: list[Any] = []
    metas: list[tuple[int, int]] = []

    def flush() -> None:
        nonlocal best_mse, best_dim, best_didx
        if not chunk:
            return
        batch = torch.cat(chunk, dim=0)
        with torch.inference_mode():
            decoded = decoder(batch)
        decoded_np = (
            decoded.reshape(len(chunk), 2, 3, eval_h, eval_w)
            .detach()
            .cpu()
            .numpy()
        )
        mses = _mse_uint8_batch_after_clamp(decoded_np, gt_eval)
        for mse, (dim, didx) in zip(mses, metas, strict=True):
            if float(mse) < best_mse:
                best_mse = float(mse)
                best_dim = dim
                best_didx = didx
        chunk.clear()
        metas.clear()

    latent_dim = int(base_lat.shape[1])
    for d in range(latent_dim):
        for di, delta in enumerate(deltas):
            cand = base_lat.clone()
            cand[0, d] += float(delta)
            chunk.append(cand)
            metas.append((d, di))
            if len(chunk) >= candidate_batch_size:
                flush()
    flush()
    return base_mse, best_mse, best_dim, best_didx


def profile_pair_proxy_mse_candidate_batches(
    *,
    decoder: Any,
    base_lat: Any,
    gt_eval: np.ndarray,
    eval_h: int,
    eval_w: int,
    deltas: np.ndarray,
    candidate_batch_sizes: list[int],
) -> dict[str, Any]:
    """Benchmark candidate chunk sizes against scalar search on one pair.

    The scalar ``candidate_batch_size=1`` result is the semantic reference. A
    larger chunk is selectable only if it returns the same best choice and
    numerically equivalent objective values on the profiled pair. This keeps
    batching opt-in and fail-closed: disagreement is recorded and scalar search
    remains selected.
    """

    unique_sizes: list[int] = []
    for raw_size in candidate_batch_sizes:
        size = int(raw_size)
        if size < 1:
            raise ValueError("candidate batch profile sizes must be >= 1")
        if size not in unique_sizes:
            unique_sizes.append(size)
    if 1 not in unique_sizes:
        unique_sizes.insert(0, 1)
    else:
        unique_sizes = [1] + [size for size in unique_sizes if size != 1]

    records: list[dict[str, Any]] = []
    reference: tuple[float, float, int, int] | None = None
    for size in unique_sizes:
        started = time.perf_counter()
        base_mse, best_mse, best_dim, best_didx = _best_pair_proxy_mse_candidate(
            decoder=decoder,
            base_lat=base_lat,
            gt_eval=gt_eval,
            eval_h=eval_h,
            eval_w=eval_w,
            deltas=deltas,
            candidate_batch_size=size,
        )
        elapsed = time.perf_counter() - started
        if reference is None:
            reference = (base_mse, best_mse, best_dim, best_didx)
        semantic_match = (
            best_dim == reference[2]
            and best_didx == reference[3]
            and np.isclose(base_mse, reference[0], rtol=0.0, atol=1e-6)
            and np.isclose(best_mse, reference[1], rtol=0.0, atol=1e-6)
        )
        records.append(
            {
                "candidate_batch_size": size,
                "elapsed_seconds": elapsed,
                "base_mse": base_mse,
                "best_mse": best_mse,
                "best_dim": int(best_dim),
                "best_delta_idx": int(best_didx),
                "semantic_match_scalar_reference": bool(semantic_match),
            }
        )

    selectable = [
        record for record in records if record["semantic_match_scalar_reference"]
    ]
    selected = min(
        selectable,
        key=lambda record: (
            float(record["elapsed_seconds"]),
            int(record["candidate_batch_size"]),
        ),
    )
    selected_size = int(selected["candidate_batch_size"])
    if selected_size == 1:
        mismatched = [
            int(record["candidate_batch_size"])
            for record in records
            if not record["semantic_match_scalar_reference"]
        ]
        reason = (
            "scalar_reference_fastest"
            if not mismatched
            else "non_scalar_batches_semantic_mismatch"
        )
    else:
        reason = "fastest_semantic_match"

    return {
        "schema_version": "a1_sidecar_candidate_batch_profile_v1",
        "reference_candidate_batch_size": 1,
        "profiled_candidate_batch_sizes": unique_sizes,
        "selected_candidate_batch_size": selected_size,
        "selection_reason": reason,
        "records": records,
    }


def _resize_uint8_to_eval(
    frames_uint8: np.ndarray,
    *,
    eval_h: int = EVAL_H,
    eval_w: int = EVAL_W,
) -> np.ndarray:
    """Resize uint8 (B, H, W, 3) ground-truth frames to (B, EVAL_H, EVAL_W, 3)
    float32 using bilinear, matching the *inverse* of the inflate-time bicubic
    upscale. Returns float32 in 0..255."""
    import torch
    import torch.nn.functional as F

    arr = (
        torch.from_numpy(frames_uint8.astype(np.float32))
        .permute(0, 3, 1, 2)  # B,C,H,W
    )
    arr = F.interpolate(arr, size=(eval_h, eval_w), mode="bilinear", align_corners=False)
    return arr.permute(0, 2, 3, 1).numpy()  # B,H,W,C


def _mse_uint8_after_clamp(decoded: np.ndarray, gt: np.ndarray) -> float:
    """decoded is (2, 3, EVAL_H, EVAL_W); gt is (2, EVAL_H, EVAL_W, 3) float32 0..255."""
    dec = decoded.transpose(0, 2, 3, 1).clip(0, 255)
    return float(np.mean((dec - gt) ** 2))


def _mse_uint8_batch_after_clamp(decoded: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Vectorized MSE for decoded shape (B, 2, 3, H, W)."""

    dec = decoded.transpose(0, 1, 3, 4, 2).clip(0, 255)
    return np.mean((dec - gt[None, ...]) ** 2, axis=(1, 2, 3, 4))


def ground_truth_pairs_needed(pair_indices: list[int], n_pairs: int) -> int:
    """Return how many leading pairs must be decoded for ``pair_indices``."""

    if not pair_indices:
        return 0
    max_pair = max(pair_indices)
    if max_pair < 0:
        raise ValueError("pair indices must be nonnegative")
    if max_pair >= n_pairs:
        raise ValueError(f"pair index {max_pair} outside n_pairs={n_pairs}")
    return max_pair + 1


def load_ground_truth_pairs(video_path: Path, n_pairs: int = 600) -> np.ndarray:
    """Decode video into pair frames (n_pairs * 2, CAMERA_H, CAMERA_W, 3) uint8.

    Uses pyav (already a project dependency). Each "pair" is two consecutive
    frames at the seq_len=2 non-overlapping batching.
    """
    import av  # type: ignore

    yuv420_to_rgb = load_upstream_yuv420_to_rgb()
    container = av.open(str(video_path))
    stream = container.streams.video[0]
    frames = []
    for f in container.decode(stream):
        # Use the exact upstream CPU-eval conversion helper. Raw PyAV rgb24
        # takes a different colorspace path and would optimize the sidecar
        # against the wrong byte substrate.
        img = yuv420_to_rgb(f)  # torch (H, W, 3) uint8
        frames.append(img.cpu().numpy())
        if len(frames) >= n_pairs * 2:
            break
    container.close()
    arr = np.stack(frames[: n_pairs * 2], axis=0)
    assert arr.shape[1:] == (CAMERA_H, CAMERA_W, 3), f"unexpected shape {arr.shape}"
    return arr


def write_resampled_archive(
    components: dict[str, Any],
    new_sidecar_blob: bytes,
    out_archive_path: Path,
) -> None:
    """Build the new archive.zip = decoder_section || latent_blob || new_sidecar."""
    import zipfile

    archive_bytes = components["archive_bytes"]
    section_total = components["section_total"]
    new_inner = (
        archive_bytes[:section_total]
        + components["latent_blob"]
        + new_sidecar_blob
    )
    out_archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_archive_path, "w", compression=zipfile.ZIP_STORED) as zf:
        zinfo = zipfile.ZipInfo(filename="x", date_time=(2024, 1, 1, 0, 0, 0))
        zinfo.external_attr = 0o644 << 16
        zf.writestr(zinfo, new_inner)


def enforce_manifest_dispatch_readiness(
    manifest: dict[str, Any],
    *,
    archive_path: Path,
    submission_dir: Path | None = None,
) -> dict[str, Any]:
    """Fail-closed on exact-eval readiness unless custody invariants hold."""

    prior_blockers = list(manifest.get("dispatch_blockers") or [])
    blockers: list[str] = []
    submission_dir = submission_dir or archive_path.parent
    if prior_blockers:
        manifest["superseded_dispatch_blockers"] = prior_blockers

    def block(reason: str) -> None:
        if reason not in blockers:
            blockers.append(reason)

    if manifest.get("lane_id") != SIDECAR_LANE_ID:
        block("sidecar_lane_id_missing_or_mismatch")
    for reason in _manifest_top_level_dispatch_blockers(manifest):
        block(reason)

    manifest_archive_value = _manifest_archive_path_value(manifest)
    manifest_archive_path = _resolve_manifest_path(manifest_archive_value)
    if manifest_archive_path is None:
        block("archive_path_missing")
    elif manifest_archive_path.resolve() != archive_path.resolve():
        block("archive_path_mismatch")

    if not archive_path.is_file():
        block(f"materialized_archive_missing:{manifest_path(archive_path)}")
    else:
        actual_sha = sha256_of(archive_path)
        actual_bytes = archive_path.stat().st_size
        for key in ("new_archive_sha256", "archive_sha256", "candidate_archive_sha256"):
            value = manifest.get(key)
            if value is not None and value != actual_sha:
                block(f"{key}_mismatch")
        for key in ("new_archive_bytes", "archive_size_bytes", "candidate_archive_bytes"):
            value = manifest.get(key)
            if value is not None and value != actual_bytes:
                block(f"{key}_mismatch")
        if not any(manifest.get(key) == actual_sha for key in ("new_archive_sha256", "archive_sha256", "candidate_archive_sha256")):
            block("archive_sha256_missing_or_mismatch")
        if not any(manifest.get(key) == actual_bytes for key in ("new_archive_bytes", "archive_size_bytes", "candidate_archive_bytes")):
            block("archive_bytes_missing_or_mismatch")
        if manifest.get("new_archive_sha256") != actual_sha:
            block("materialized_archive_sha256_mismatch")
        if manifest.get("new_archive_bytes") != actual_bytes:
            block("materialized_archive_size_mismatch")
        for reason in _runtime_custody_blockers(
            manifest,
            submission_dir=submission_dir,
            archive_path=archive_path,
        ):
            block(reason)
        for reason in _member_section_proof_blockers(manifest, archive_path=archive_path):
            block(reason)
        for reason in _sidecar_choice_state_blockers(manifest):
            block(reason)

    if manifest.get("smoke_only") is True:
        block("smoke_only_not_exact_eval_ready")
    if manifest.get("full_non_smoke_search") is not True:
        block("non_full_sidecar_search_not_exact_eval_ready")
    runtime_tree = manifest.get("runtime_tree_sha256")
    if isinstance(runtime_tree, str):
        for reason in _runtime_smoke_evidence_blockers(
            manifest,
            archive_path=archive_path,
            runtime_tree_sha256=runtime_tree,
        ):
            block(reason)
    else:
        block("runtime_tree_sha256_missing_or_mismatch")
        if manifest.get("runtime_smoke_checked") is True:
            block("runtime_smoke_evidence_runtime_tree_sha256_mismatch")
        else:
            block("runtime_smoke_not_checked")
    for reason in _dispatch_custody_record_blockers(
        manifest,
        archive_path=archive_path,
        runtime_tree_sha256=runtime_tree if isinstance(runtime_tree, str) else None,
    ):
        block(reason)
    for reason in _no_op_detector_blockers(manifest):
        block(reason)

    manifest["dispatch_blockers"] = blockers
    manifest["ready_for_exact_eval_dispatch"] = not blockers
    return manifest


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="output directory for the resampled submission_dir + manifest",
    )
    p.add_argument(
        "--search-signal",
        choices=["proxy_mse", "joint_seg_pose"],
        default="proxy_mse",
        help="search objective (proxy_mse is fast; joint_seg_pose requires --accept-cpu-budget)",
    )
    p.add_argument(
        "--video-path",
        type=Path,
        default=REPO_ROOT / "upstream/videos/0.mkv",
        help="ground-truth video for proxy MSE",
    )
    p.add_argument(
        "--n-pairs",
        type=int,
        default=N_PAIRS,
        help="how many pairs to search (smoke=10; full=600)",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="run a 10-pair smoke search (verifies plumbing in ~30 sec)",
    )
    p.add_argument(
        "--accept-cpu-budget",
        action="store_true",
        help="acknowledge the search will take CPU-hours (required for joint_seg_pose)",
    )
    p.add_argument(
        "--encode-format",
        choices=["packed_661", "n_pairs_600"],
        default="packed_661",
        help="sidecar wire format (PACKED is more general; N_PAIRS limited to dim<16)",
    )
    p.add_argument(
        "--candidate-batch-size",
        type=int,
        default=1,
        help=(
            "candidate latent perturbations per decoder forward chunk; default 1 "
            "preserves fastest measured scalar CPU path, larger values are experimental"
        ),
    )
    p.add_argument(
        "--profile-candidate-batches",
        type=int,
        nargs="+",
        default=None,
        metavar="N",
        help=(
            "benchmark candidate batch sizes on the first searched pair against "
            "the scalar reference and record the profile in sidecar_manifest.json"
        ),
    )
    p.add_argument(
        "--auto-candidate-batch-size",
        action="store_true",
        help=(
            "after --profile-candidate-batches, use the fastest profiled batch "
            "size that matches scalar search semantics on the profiled pair"
        ),
    )
    p.add_argument(
        "--search-device",
        choices=["cpu", "mps", "cuda"],
        default="cpu",
        help=(
            "device for proxy candidate generation only; default cpu. "
            "mps/cuda proxy choices are not score evidence"
        ),
    )
    p.add_argument(
        "--runtime-smoke",
        action="store_true",
        help="run a bounded local inflate.py smoke and record output digest evidence",
    )
    p.add_argument(
        "--runtime-smoke-pairs",
        type=int,
        default=1,
        help="pair count for --runtime-smoke; default 1 to avoid full raw output",
    )
    p.add_argument(
        "--exact-inflate-sh-smoke",
        action="store_true",
        help=(
            "run the contest inflate.sh archive_dir/output_dir/file_list signature; "
            "full A1 decode can be expensive and writes a full raw output"
        ),
    )
    p.add_argument(
        "--exact-inflate-sh-smoke-timeout",
        type=int,
        default=None,
        help="optional timeout in seconds for --exact-inflate-sh-smoke",
    )
    p.add_argument(
        "--resume-search-state",
        action="store_true",
        help=(
            "resume from output-dir/sidecar_choice_state.json if present; "
            "the saved context must match archive/search/device/encoding"
        ),
    )
    p.add_argument(
        "--recheck-unproven-pairs",
        action="store_true",
        help=(
            "when resuming, recompute already searched pairs that lack "
            "per-pair scalar-equivalence provenance"
        ),
    )
    p.add_argument(
        "--max-search-seconds",
        type=float,
        default=None,
        help=(
            "local wall-clock guard for proxy search; after at least one new pair "
            "is searched, stop cleanly and emit a fail-closed partial packet"
        ),
    )
    args = p.parse_args()
    if args.auto_candidate_batch_size and not args.profile_candidate_batches:
        args.profile_candidate_batches = [1, 2, 4, 8, 16, 32, 64]

    timestamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.output_dir or (
        REPO_ROOT
        / f"experiments/results/a1_per_pair_latent_sidecar_resampled_{timestamp}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    output_lock = SidecarOutputLock(out_dir)
    try:
        output_lock.acquire()
    except RuntimeError as exc:
        sys.stderr.write(f"[fatal] {exc}\n")
        return 2

    if args.search_signal == "joint_seg_pose" and not args.accept_cpu_budget:
        sys.stderr.write(
            "[fatal] joint_seg_pose search requires --accept-cpu-budget acknowledgement "
            "(estimated ~30+ CPU hours on M-series ARM for full 600 pairs)\n"
        )
        return 2

    actual_sha = sha256_of(A1_ARCHIVE_PATH)
    if actual_sha != A1_EXPECTED_ARCHIVE_SHA:
        sys.stderr.write(
            f"[fatal] A1 archive SHA mismatch: expected={A1_EXPECTED_ARCHIVE_SHA} "
            f"actual={actual_sha}\n"
        )
        return 2

    print(f"[ok] loading A1 archive components from {A1_ARCHIVE_PATH}", flush=True)
    components = load_a1_archive_components(A1_ARCHIVE_PATH)
    print(
        f"[ok] decoder tensors={len(components['decoder_sd'])} "
        f"latents shape={tuple(components['latents_pre_sidecar'].shape)} "
        f"old_sidecar_bytes={len(components['sidecar_blob_old'])}",
        flush=True,
    )
    old_sidecar_sha = hashlib.sha256(components["sidecar_blob_old"]).hexdigest()
    state_path = out_dir / "sidecar_choice_state.json"
    state_context = {
        "old_archive_sha256": A1_EXPECTED_ARCHIVE_SHA,
        "old_sidecar_sha256": old_sidecar_sha,
        "search_signal": args.search_signal,
        "search_device": args.search_device,
        "encode_format": args.encode_format,
        "total_pairs": int(components["n_pairs"]),
        "latent_dim": int(components["latent_dim"]),
    }
    initial_dims = None
    initial_delta_idx = None
    initial_searched_mask = None
    initial_pair_search_records: dict[int, dict[str, Any]] | None = None
    resumed_state = False
    if args.resume_search_state and state_path.exists():
        print(f"[ok] resuming sidecar choice state from {state_path}", flush=True)
        (
            initial_dims,
            initial_delta_idx,
            initial_searched_mask,
            _state_payload,
        ) = load_sidecar_choice_state(state_path, **state_context)
        initial_pair_search_records = pair_search_records_from_payload(_state_payload)
        resumed_state = True
    elif args.resume_search_state:
        print(f"[ok] no prior state at {state_path}; starting fresh", flush=True)

    n_pairs_to_search = 10 if args.smoke else args.n_pairs
    pair_indices = list(range(min(n_pairs_to_search, components["n_pairs"])))

    if args.search_signal == "proxy_mse":
        print(f"[start] proxy-mse search over {len(pair_indices)} pairs", flush=True)
        print(f"[ok] decoding ground-truth video {args.video_path}", flush=True)
        gt_pair_count = ground_truth_pairs_needed(pair_indices, components["n_pairs"])
        gt_frames = load_ground_truth_pairs(args.video_path, n_pairs=gt_pair_count)
        print(f"[ok] gt_frames shape={gt_frames.shape}", flush=True)
        log_path = out_dir / "sidecar_search.log"
        dims, delta_idx, search_meta = search_per_pair_proxy_mse(
            components,
            gt_frames,
            pair_indices=pair_indices,
            log_path=log_path,
            candidate_batch_size=args.candidate_batch_size,
            search_device=args.search_device,
            initial_dims=initial_dims,
            initial_delta_idx=initial_delta_idx,
            initial_searched_mask=initial_searched_mask,
            initial_pair_search_records=initial_pair_search_records,
            state_path=state_path,
            state_context=state_context,
            max_search_seconds=args.max_search_seconds,
            candidate_batch_profile_sizes=args.profile_candidate_batches,
            auto_candidate_batch_size=args.auto_candidate_batch_size,
            recheck_unproven_pairs=args.recheck_unproven_pairs,
        )
    else:
        sys.stderr.write(
            "[fatal] joint_seg_pose path not yet implemented in this tool — "
            "use proxy_mse and let GHA confirm. (Reactivation criterion: GPU "
            "available + budget approved.)\n"
        )
        return 2

    # Encode the new sidecar
    if args.encode_format == "n_pairs_600":
        new_sidecar = encode_sidecar_n_pairs(dims, delta_idx)
    else:
        new_sidecar = encode_sidecar_huff_enum(dims, delta_idx)
    print(
        f"[ok] new sidecar encoded: {len(new_sidecar)} bytes "
        f"(old was {len(components['sidecar_blob_old'])})",
        flush=True,
    )

    # Build the new submission_dir
    sub_dir = out_dir / "submission_dir"
    sub_dir.mkdir(exist_ok=True)
    new_archive_path = sub_dir / "archive.zip"
    write_resampled_archive(components, new_sidecar, new_archive_path)
    new_archive_sha = sha256_of(new_archive_path)
    new_archive_bytes = new_archive_path.stat().st_size
    # Copy A1's existing inflate.py, inflate.sh, src/* (the bias correction is
    # PRESERVED — we're only re-searching the latent sidecar, not the bias)
    shutil.copy2(A1_SUBMISSION_DIR / "inflate.py", sub_dir / "inflate.py")
    inflate_sh_path = sub_dir / "inflate.sh"
    shutil.copy2(A1_SUBMISSION_DIR / "inflate.sh", inflate_sh_path)
    inflate_sh_path.chmod(0o755)
    src_target = sub_dir / "src"
    src_target.mkdir(exist_ok=True)
    for fname in ("model.py", "codec.py"):
        shutil.copy2(A1_SUBMISSION_DIR / "src" / fname, src_target / fname)

    manifest = {
        "lane_id": SIDECAR_LANE_ID,
        "schema_version": "a1_per_pair_latent_sidecar_resampled_v1",
        "build_timestamp_utc": dt.datetime.now(dt.UTC).isoformat(),
        "submission_name": f"a1_sidecar_resampled_{args.search_signal}_{timestamp}",
        "search_signal": args.search_signal,
        "search_device": args.search_device,
        "search_meta": search_meta,
        "encode_format": args.encode_format,
        "old_archive_sha256": A1_EXPECTED_ARCHIVE_SHA,
        "old_archive_bytes": A1_EXPECTED_ARCHIVE_BYTES,
        "old_sidecar_bytes": len(components["sidecar_blob_old"]),
        "archive_path": manifest_path(new_archive_path),
        "archive_sha256": new_archive_sha,
        "archive_size_bytes": new_archive_bytes,
        "candidate_archive_path": manifest_path(new_archive_path),
        "candidate_archive_sha256": new_archive_sha,
        "candidate_archive_bytes": new_archive_bytes,
        "new_archive_path": manifest_path(new_archive_path),
        "new_archive_sha256": new_archive_sha,
        "new_archive_bytes": new_archive_bytes,
        "new_sidecar_bytes": len(new_sidecar),
        "delta_archive_bytes": new_archive_bytes - A1_EXPECTED_ARCHIVE_BYTES,
        "score_claim": False,
        "byte_proxy_only": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": (
            f"[predicted; per-pair latent sidecar resampled on A1 substrate via "
            f"{args.search_signal}; pre-GHA-dispatch]"
        ),
        "pending_dispatch_actions": [
            "claim lane before any GHA/remote eval dispatch",
            "run exact-eval dispatcher preflight against submission_dir",
        ],
        "post_eval_required_actions": [
            "record runtime tree SHA and terminal dispatch claim row",
        ],
        "tag_discipline": {
            "before_eval": (
                f"[predicted; per-pair latent sidecar resampled via {args.search_signal}]"
            ),
            "after_eval": "[contest-CPU GHA Linux x86_64] iff GHA dispatch succeeds",
        },
        "smoke_only": args.smoke,
        "full_non_smoke_search": (
            (not args.smoke) and len(pair_indices) == int(components["n_pairs"])
        ),
        "requested_n_pairs": args.n_pairs,
        "total_pairs": int(components["n_pairs"]),
        "runtime_smoke_checked": False,
        "n_pairs_searched": len(pair_indices),
        "n_pairs_perturbed": int((dims != 255).sum()),
        "a1_canonical_baseline": {
            "score": 0.19284757743677347,
            "tag": "[contest-CPU GHA Linux x86_64]",
            "evidence_path": "experiments/results/a1_latentalign_importpathfix_cpu_eval_gha_20260509/contest_auth_eval.adjudicated.json",
        },
        "no_op_detector": {
            "old_inner_sidecar_sha256": old_sidecar_sha,
            "new_inner_sidecar_sha256": hashlib.sha256(new_sidecar).hexdigest(),
            "sidecar_changed": components["sidecar_blob_old"] != new_sidecar,
        },
    }
    choice_state_record = sidecar_choice_state_manifest_record(state_path)
    manifest["sidecar_choice_state"] = choice_state_record
    manifest["choice_state_resumed"] = resumed_state
    manifest["n_pairs_searched"] = choice_state_record["n_pairs_completed_total"]
    manifest["n_pairs_completed_this_run"] = search_meta.get("n_pairs_completed_this_run")
    manifest["n_pairs_skipped_already_completed"] = search_meta.get(
        "n_pairs_skipped_already_completed"
    )
    manifest["n_pairs_recheck_unproven_planned"] = search_meta.get(
        "n_pairs_recheck_unproven_planned"
    )
    manifest["n_pairs_rechecked_unproven_completed_this_run"] = search_meta.get(
        "n_pairs_rechecked_unproven_completed_this_run"
    )
    manifest["remaining_unproven_records_after_run"] = search_meta.get(
        "remaining_unproven_records_after_run"
    )
    manifest["search_stopped_for_wall_clock"] = search_meta.get(
        "search_stopped_for_wall_clock"
    )
    manifest["full_non_smoke_search"] = (
        (not args.smoke) and choice_state_record["full_coverage"] is True
    )
    local_runtime_custody = collect_local_runtime_custody(
        sub_dir,
        archive_path=new_archive_path,
    )
    member_section_proof = collect_member_section_proof(new_archive_path)
    manifest["local_runtime_custody"] = local_runtime_custody
    manifest["runtime_manifest"] = local_runtime_custody
    manifest["runtime_tree_sha256"] = local_runtime_custody["runtime_tree_sha256"]
    manifest["member_section_proof"] = member_section_proof
    if args.runtime_smoke:
        smoke_evidence = run_local_runtime_smoke(
            sub_dir,
            archive_path=new_archive_path,
            smoke_dir=out_dir / "runtime_smoke",
            smoke_pairs=args.runtime_smoke_pairs,
            runtime_tree_sha256=local_runtime_custody["runtime_tree_sha256"],
        )
        manifest["runtime_smoke_checked"] = smoke_evidence.get("exit_code") == 0
        manifest["runtime_smoke_evidence"] = smoke_evidence
        output_change_probe = collect_runtime_output_change_probe(
            baseline_submission_dir=A1_SUBMISSION_DIR,
            candidate_submission_dir=sub_dir,
            baseline_archive_path=A1_ARCHIVE_PATH,
            candidate_archive_path=new_archive_path,
            probe_dir=out_dir / "runtime_output_change_probe",
            smoke_pairs=args.runtime_smoke_pairs,
            candidate_runtime_tree_sha256=local_runtime_custody["runtime_tree_sha256"],
        )
        manifest["runtime_output_change_probe"] = output_change_probe
        manifest["no_op_detector"].update(
            {
                "baseline_output_sha256": output_change_probe.get(
                    "baseline_output_sha256"
                ),
                "candidate_output_sha256": output_change_probe.get(
                    "candidate_output_sha256"
                ),
                "runtime_output_changed": output_change_probe.get(
                    "runtime_output_changed"
                ),
                "runtime_output_probe_schema": output_change_probe.get(
                    "schema_version"
                ),
            }
        )
    if args.exact_inflate_sh_smoke:
        exact_smoke_evidence = run_exact_inflate_sh_smoke(
            sub_dir,
            archive_path=new_archive_path,
            smoke_dir=out_dir / "exact_inflate_sh_smoke",
            runtime_tree_sha256=local_runtime_custody["runtime_tree_sha256"],
            timeout_seconds=args.exact_inflate_sh_smoke_timeout,
        )
        if "runtime_smoke_evidence" in manifest:
            manifest["local_import_runtime_smoke_evidence"] = manifest[
                "runtime_smoke_evidence"
            ]
        manifest["runtime_smoke_checked"] = exact_smoke_evidence.get("exit_code") == 0
        manifest["runtime_smoke_evidence"] = exact_smoke_evidence
    manifest = enforce_manifest_dispatch_readiness(
        manifest,
        archive_path=new_archive_path,
        submission_dir=sub_dir,
    )
    (out_dir / "sidecar_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(
        f"\n[done] sidecar_manifest written to "
        f"{manifest_path(out_dir / 'sidecar_manifest.json')}",
        flush=True,
    )
    print(
        f"  new_archive_bytes = {new_archive_bytes} "
        f"(Δ = {new_archive_bytes - A1_EXPECTED_ARCHIVE_BYTES:+d} from A1 baseline)",
        flush=True,
    )
    print(f"  new_archive_sha256 = {new_archive_sha}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
