#!/usr/bin/env python
"""Submission pipeline runner — one command to compress, inflate, score.

State-machine design: every stage transition is persisted to state.json.
If the process dies, `runner.py resume <run>` picks up from the last
completed stage. No nohup, no PID files, no /tmp scatter.

Usage:
    python runner.py evaluate --upstream-dir /path/to/challenge
    python runner.py status my_run
    python runner.py resume my_run --upstream-dir /path/to/challenge
"""
from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import zipfile
from enum import Enum
from pathlib import Path
from typing import Any

import click


# ============================================================
# PoseNet calibration — local scorer vs auth scorer
# ============================================================
# DIAGNOSIS (2026-04-11): Both torch 2.10.0 and 2.11.0 produce
# IDENTICAL per-pair PoseNet outputs on CPU. The 29x PoseNet inflation
# is NOT a torch/timm version issue. Root cause: auth scorer uses
# CUDA + DALI for video decode, which produces different ground truth
# pixel values than PyAV's CPU decode path. SegNet and Rate are
# unaffected (SegNet uses argmax which is robust to small pixel diffs;
# Rate is file-size-based with no model involvement).
#
# Calibration data (archive md5 463b6fdb, 864167 bytes):
#   Auth:  pose=0.00218, seg=0.00610, rate=0.02302, score=1.33
#   Local: pose=0.06256, seg=0.00565, rate=0.02302, score=1.93
#
# PoseNet ratio: 0.00218 / 0.06256 = 0.03484
# SegNet ratio:  0.00610 / 0.00565 = 1.0796 (close to 1.0)
# Rate ratio:    identical (no model)

POSE_CALIBRATION_FACTOR = 0.00218 / 0.06256  # = 0.03484
POSE_CALIBRATION_ARCHIVE_MD5 = "463b6fdb"
POSE_CALIBRATION_N_POINTS = 1  # PROVISIONAL — single data point


def calibrate_score(
    local_pose: float, local_seg: float, local_rate: float
) -> dict[str, float]:
    """Translate local scorer output to estimated auth score.

    Calibration points (same archive, different scorers):
      Auth:  pose=0.00218, seg=0.00610, rate=0.02302, score=1.33
      Local: pose=0.06256, seg=0.00565, rate=0.02302, score=1.93

    Root cause: DALI GPU decode (auth) vs PyAV CPU decode (local)
    produce different ground truth pixels. PoseNet MSE is sensitive
    to these sub-pixel differences; SegNet argmax is robust.

    SegNet: local is reliable (ratio ~1.08, nearly 1:1)
    Rate: identical (no model involved)
    PoseNet: local is 29x inflated (ratio = 0.00218/0.06256 = 0.0349)

    CONTRARIAN NOTE: The calibration factor is a MEAN correction.
    Per-pair PoseNet values vary by 1000x (CV=1.15). If the DALI-vs-PyAV
    divergence is content-dependent (e.g., high-motion frames diverge more),
    the linear correction would be wrong for different checkpoints that
    shift the distortion distribution across content types. However, the
    MEAN correction is still the best 1-point estimator for the AGGREGATE
    score, which is what the leaderboard uses.

    PROVISIONAL: Based on 1 calibration point. Submit for definitive score.

    Returns dict with calibrated components and estimated auth score.
    """
    calibrated_pose = local_pose * POSE_CALIBRATION_FACTOR
    # SegNet and Rate pass through unchanged
    calibrated_seg = local_seg
    calibrated_rate = local_rate

    calibrated_score = (
        100 * calibrated_seg
        + math.sqrt(10 * calibrated_pose)
        + 25 * calibrated_rate
    )

    return {
        "calibrated_pose": calibrated_pose,
        "calibrated_seg": calibrated_seg,
        "calibrated_rate": calibrated_rate,
        "calibrated_score": calibrated_score,
        "pose_calibration_factor": POSE_CALIBRATION_FACTOR,
        "calibration_n_points": POSE_CALIBRATION_N_POINTS,
        "calibration_archive_md5": POSE_CALIBRATION_ARCHIVE_MD5,
    }


# ============================================================
# File lock — prevents concurrent runner instances on same work-dir
# ============================================================
def _acquire_lock(lock_path: Path) -> None:
    """Write PID to lock file. Raise if another runner is alive."""
    if lock_path.exists():
        try:
            old_pid = int(lock_path.read_text().strip())
            os.kill(old_pid, 0)  # check if alive (signal 0 = no-op)
            raise click.ClickException(
                f"Another runner (PID {old_pid}) is already running.\n"
                f"Lock file: {lock_path}\n"
                f"If the process is dead, delete the lock file and retry."
            )
        except (ValueError, ProcessLookupError, PermissionError):
            pass  # stale lock — previous process is dead
    lock_path.write_text(str(os.getpid()))


def _release_lock(lock_path: Path) -> None:
    """Remove lock file if it belongs to this process."""
    if lock_path.exists():
        try:
            if int(lock_path.read_text().strip()) == os.getpid():
                lock_path.unlink()
        except (ValueError, OSError):
            pass


# ============================================================
# State machine
# ============================================================
class RunState(str, Enum):
    CREATED = "created"
    PREFLIGHT_OK = "preflight_ok"
    COMPRESSING = "compressing"
    COMPRESSED = "compressed"
    PACKAGING = "packaging"
    PACKAGED = "packaged"
    INFLATING = "inflating"
    INFLATED = "inflated"
    SCORING = "scoring"
    SCORED = "scored"
    FAILED = "failed"


# Ordered list of resumable stages — resume skips forward to first incomplete.
STAGE_ORDER = [
    RunState.PREFLIGHT_OK,
    RunState.COMPRESSED,
    RunState.PACKAGED,
    RunState.INFLATED,
    RunState.SCORED,
]


class RunManager:
    """Manages a single eval run directory and its state.json."""

    def __init__(self, run_dir: Path):
        self.run_dir = run_dir.resolve()
        self.state_file = self.run_dir / "state.json"
        self.log_dir = self.run_dir / "logs"
        self.submission_dir = self.run_dir / "submission"
        self.archive_unzip_dir = self.submission_dir / "archive"
        self.inflated_dir = self.submission_dir / "inflated"
        self.archive_zip = self.submission_dir / "archive.zip"

    # -- Directory bootstrap ------------------------------------------
    def ensure_dirs(self) -> None:
        for d in [
            self.run_dir,
            self.log_dir,
            self.submission_dir,
            self.archive_unzip_dir,
            self.inflated_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)

    # -- State persistence --------------------------------------------
    def load_state(self) -> dict[str, Any]:
        if self.state_file.exists():
            return json.loads(self.state_file.read_text())
        return {}

    def save_state(self, data: dict[str, Any]) -> None:
        self.state_file.write_text(json.dumps(data, indent=2, default=str) + "\n")

    def get_stage(self) -> RunState:
        data = self.load_state()
        raw = data.get("stage", RunState.CREATED.value)
        try:
            return RunState(raw)
        except ValueError:
            return RunState.CREATED

    def set_stage(self, stage: RunState, **extra: Any) -> None:
        data = self.load_state()
        data["stage"] = stage.value
        data["stage_updated"] = datetime.datetime.now().isoformat()
        data.update(extra)
        self.save_state(data)

    def record_failure(self, stage: RunState, error: str) -> None:
        data = self.load_state()
        data["stage"] = RunState.FAILED.value
        data["failed_at"] = stage.value
        data["error"] = error
        data["failed_time"] = datetime.datetime.now().isoformat()
        self.save_state(data)

    # -- Resume logic -------------------------------------------------
    def next_stage_after(self, completed: RunState) -> RunState | None:
        """Return the stage to execute after *completed*, or None if done."""
        for i, s in enumerate(STAGE_ORDER):
            if s == completed and i + 1 < len(STAGE_ORDER):
                return STAGE_ORDER[i + 1]
        return None

    def first_incomplete_stage(self) -> RunState:
        """Return the first stage that has not been completed."""
        current = self.get_stage()
        if current == RunState.FAILED:
            # Resume from the stage that failed
            data = self.load_state()
            failed_at = data.get("failed_at", RunState.CREATED.value)
            try:
                return RunState(failed_at)
            except ValueError:
                return RunState.PREFLIGHT_OK
        if current == RunState.SCORED:
            return RunState.SCORED  # nothing to do
        # Find current in order and return the next one
        for i, s in enumerate(STAGE_ORDER):
            if s.value == current.value:
                return STAGE_ORDER[i]  # re-run this stage (it was set as in-progress)
            # If current is the "doing" version (e.g., COMPRESSING -> next is COMPRESSED)
        # Map in-progress stages to their completion target
        in_progress_map = {
            RunState.CREATED: RunState.PREFLIGHT_OK,
            RunState.PREFLIGHT_OK: RunState.COMPRESSED,
            RunState.COMPRESSING: RunState.COMPRESSED,
            RunState.COMPRESSED: RunState.PACKAGED,
            RunState.PACKAGING: RunState.PACKAGED,
            RunState.PACKAGED: RunState.INFLATED,
            RunState.INFLATING: RunState.INFLATED,
            RunState.INFLATED: RunState.SCORED,
            RunState.SCORING: RunState.SCORED,
        }
        return in_progress_map.get(current, RunState.PREFLIGHT_OK)


# ============================================================
# Pre-flight checks
# ============================================================
def preflight_upstream(upstream_dir: Path) -> list[str]:
    """Validate upstream directory has everything the scorer needs."""
    errors = []
    required_files = [
        "evaluate.py",
        "frame_utils.py",
        "modules.py",
        "models/posenet.safetensors",
        "models/segnet.safetensors",
        "public_test_video_names.txt",
    ]
    # Check for at least one video
    video_names_file = upstream_dir / "public_test_video_names.txt"
    if video_names_file.exists():
        names = [
            ln.strip()
            for ln in video_names_file.read_text().splitlines()
            if ln.strip()
        ]
        for name in names:
            vpath = upstream_dir / "videos" / name
            if not vpath.exists():
                errors.append(f"Missing test video: {vpath}")
    for f in required_files:
        if not (upstream_dir / f).exists():
            errors.append(f"Missing upstream file: {f}")
    return errors


def preflight_tools() -> list[str]:
    """Check external tool availability."""
    errors = []
    if shutil.which("ffmpeg") is None:
        errors.append("ffmpeg not found in PATH")
    if shutil.which("uv") is None:
        errors.append("uv not found in PATH (required for package management)")
    # Python imports
    for mod in ["torch", "av", "numpy"]:
        try:
            __import__(mod)
        except ImportError:
            errors.append(f"Missing Python dependency: {mod}")
    return errors


def preflight_submission(submission_src: Path) -> list[str]:
    """Check submission source directory has required files."""
    errors = []
    required = ["config.env", "inflate.sh", "inflate_postfilter.py"]
    for f in required:
        if not (submission_src / f).exists():
            errors.append(f"Missing submission file: {submission_src / f}")
    # Postfilter weights
    if not (submission_src / "postfilter_int8.pt").exists():
        errors.append(f"Missing postfilter weights: {submission_src / 'postfilter_int8.pt'}")
    return errors


def preflight_checkpoint_identity(submission_src: Path) -> list[str]:
    """Verify the postfilter checkpoint matches promoted_result.json.

    This catches the silent-overwrite bug: if postfilter_int8.pt was replaced
    with a different checkpoint (different training run, different architecture),
    the eval will silently produce wrong scores. We lost hours debugging this
    exact scenario on 2026-04-12.
    """
    warnings = []
    ckpt_path = submission_src / "postfilter_int8.pt"
    promoted_path = submission_src.parent.parent / ".omx" / "state" / "promoted_result.json"

    if not ckpt_path.exists():
        return warnings

    # Compute checkpoint hash
    h = hashlib.md5()
    with open(ckpt_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    ckpt_md5 = h.hexdigest()

    # Compare with promoted result if available
    if promoted_path.exists():
        try:
            promoted = json.loads(promoted_path.read_text())
            promoted_artifact = promoted.get("artifact_path", "")
            if promoted_artifact:
                promoted_ckpt = submission_src.parent.parent / promoted_artifact
                if promoted_ckpt.exists():
                    h2 = hashlib.md5()
                    with open(promoted_ckpt, "rb") as f:
                        for chunk in iter(lambda: f.read(8192), b""):
                            h2.update(chunk)
                    promoted_md5 = h2.hexdigest()
                    if ckpt_md5 != promoted_md5:
                        warnings.append(
                            f"CHECKPOINT MISMATCH: postfilter_int8.pt (md5 {ckpt_md5[:8]}...) "
                            f"does not match promoted result artifact (md5 {promoted_md5[:8]}...). "
                            f"Promoted: {promoted_artifact} (score {promoted.get('score')}). "
                            f"This will produce WRONG scores. Restore the correct checkpoint."
                        )
                    else:
                        click.echo(
                            f"  Checkpoint verified: md5 {ckpt_md5[:12]}... matches promoted result",
                            err=True,
                        )
        except (json.JSONDecodeError, OSError):
            pass

    # Always log the hash for experiment records
    click.echo(f"  Checkpoint: {ckpt_path.name} (md5 {ckpt_md5})", err=True)
    return warnings


def preflight_config_match(submission_src: Path) -> list[str]:
    """Check if postfilter checkpoint was trained with matching encode config."""
    warnings = []
    ckpt_path = submission_src / "postfilter_int8.pt"
    config_path = submission_src / "config.env"
    if not ckpt_path.exists() or not config_path.exists():
        return warnings

    try:
        import torch

        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        meta = state.get("__meta__")
        if not isinstance(meta, dict):
            warnings.append(
                "Checkpoint has no __meta__ — cannot verify encode config match"
            )
            return warnings

        # config_fingerprint may be inside __meta__ or at top level (depends on
        # which version of training.py saved the checkpoint). Check both.
        fp = meta.get("config_fingerprint") or state.get("config_fingerprint")
        if fp is None:
            warnings.append(
                "Checkpoint has no config_fingerprint — cannot verify encode config match"
            )
            return warnings

        # Parse config.env
        current = _parse_config_env(config_path)
        # Map fingerprint keys to config.env keys
        key_map = {
            "crf": "SVT_AV1_CRF",
            "scale_w": "SCALE_W",
            "scale_h": "SCALE_H",
            "codec": "VIDEO_CODEC",
            "color_matrix": "SOURCE_COLOR_MATRIX",
        }
        mismatches = []
        for fk, ck in key_map.items():
            if fk in fp and ck in current:
                if str(fp[fk]) != str(current[ck]):
                    mismatches.append(
                        f"{fk}: checkpoint={fp[fk]}, config={current[ck]}"
                    )
        if mismatches:
            warnings.append(
                "CONFIG MISMATCH — postfilter may have been trained with different encode settings:\n"
                + "\n".join(f"  {m}" for m in mismatches)
                + "\nThis may cause distribution shift and score regression."
            )
    except Exception as e:
        warnings.append(f"Could not validate config match: {e}")

    return warnings


def _parse_config_env(path: Path) -> dict[str, str]:
    """Parse a shell-style KEY=VALUE config file."""
    result = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            k, v = line.split("=", 1)
            result[k.strip()] = v.strip()
    return result


def config_fingerprint(config_path: Path) -> str:
    """SHA-256 of the config.env contents for reproducibility tagging."""
    if not config_path.exists():
        return "missing"
    return hashlib.sha256(config_path.read_bytes()).hexdigest()[:16]


# ============================================================
# Subprocess runner with logging
# ============================================================
def run_stage(
    cmd: list[str],
    log_file: Path,
    timeout: int = 3600,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
    label: str = "",
) -> int:
    """Run a pipeline stage with proper logging and error handling.

    Returns exit code (always 0 on success, raises on failure).
    """
    merged_env = {**os.environ, **(env or {})}
    label_str = f" [{label}]" if label else ""
    click.echo(f"  Running{label_str}: {' '.join(str(c) for c in cmd[:6])}{'...' if len(cmd) > 6 else ''}")
    click.echo(f"  Log: {log_file}")

    with open(log_file, "w") as log:
        log.write(f"# Command: {' '.join(str(c) for c in cmd)}\n")
        log.write(f"# Started: {datetime.datetime.now().isoformat()}\n")
        log.write(f"# CWD: {cwd or os.getcwd()}\n\n")
        log.flush()

        proc = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=merged_env,
            cwd=str(cwd) if cwd else None,
        )
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            raise click.ClickException(
                f"Stage timed out after {timeout}s.{label_str}\n"
                f"Log: {log_file}\n"
                f"The process was killed. Use `runner.py resume` to retry."
            )

        if proc.returncode != 0:
            lines = log_file.read_text().splitlines()[-20:]
            raise click.ClickException(
                f"Stage failed (exit {proc.returncode}).{label_str}\n"
                f"Last 20 lines of {log_file}:\n" + "\n".join(lines)
            )

    return proc.returncode


# ============================================================
# Pipeline stages
# ============================================================
def stage_compress(
    mgr: RunManager,
    upstream_dir: Path,
    submission_src: Path,
) -> None:
    """Compress: call compress.sh as a subprocess.

    NEVER reimplement compress.sh inline — it has ROI, even-frame QP,
    sky degradation, pre-denoise, metadata stripping, and color params
    that are too complex to duplicate correctly. Call the real script.
    """
    mgr.set_stage(RunState.COMPRESSING)

    compress_sh = submission_src / "compress.sh"
    if not compress_sh.exists():
        raise click.ClickException(f"compress.sh not found: {compress_sh}")

    cmd = [
        "bash", str(compress_sh),
    ]
    env = {
        **os.environ,
        "COMMA_CHALLENGE_ROOT": str(upstream_dir),
        "CONFIG_ENV_PATH": str(submission_src / "config.env"),
    }

    run_stage(
        cmd,
        mgr.log_dir / "compress.log",
        timeout=600,
        label="compress (compress.sh)",
        env=env,
    )

    # compress.sh writes archive.zip to submission_src/archive.zip
    # Copy it into our run directory
    src_archive = submission_src / "archive.zip"
    if src_archive.exists():
        shutil.copy2(str(src_archive), str(mgr.run_dir / "submission" / "archive.zip"))

    mgr.set_stage(RunState.COMPRESSED)


def stage_package(
    mgr: RunManager,
    submission_src: Path,
) -> None:
    """Package: verify compress.sh output and unzip into archive_unzip_dir.

    compress.sh already bundles video + postfilter into archive.zip.
    We do NOT rebuild the zip — that would risk losing compress.sh features
    (ROI, metadata stripping, etc.). Instead we just unzip for inflate.
    """
    mgr.set_stage(RunState.PACKAGING)

    # compress.sh writes to submission_src/archive.zip. stage_compress
    # copies it to mgr.submission_dir/archive.zip. Verify it's there.
    if not mgr.archive_zip.exists():
        raise click.ClickException(
            f"archive.zip not found at {mgr.archive_zip}. "
            "Did stage_compress complete successfully?"
        )

    # Unzip into archive_unzip_dir so inflate can find the files
    mgr.archive_unzip_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(mgr.archive_zip, "r") as zf:
        zf.extractall(mgr.archive_unzip_dir)

    # Verify postfilter is present (contest rules: neural artifacts in archive)
    pf_in_archive = mgr.archive_unzip_dir / "postfilter_int8.pt"
    if not pf_in_archive.exists():
        click.echo(
            "postfilter_int8.pt not found in archive.zip. "
            "Contest rules require neural artifacts inside the archive. "
            "Fix compress.sh to bundle the postfilter.",
            err=True,
        )
        raise click.ClickException(
            "Archive missing postfilter_int8.pt — cannot proceed. "
            "This is a compress.sh packaging bug, not a fallback situation."
        )

    size = mgr.archive_zip.stat().st_size
    n_files = len(list(mgr.archive_unzip_dir.rglob("*")))
    click.echo(f"  archive.zip: {size:,} bytes, {n_files} files")

    mgr.set_stage(RunState.PACKAGED, archive_bytes=size)


def stage_inflate(
    mgr: RunManager,
    submission_src: Path,
    upstream_dir: Path,
    device: str = "cpu",
) -> None:
    """Inflate: decompress + upscale + postfilter."""
    mgr.set_stage(RunState.INFLATING)

    video_names_file = upstream_dir / "public_test_video_names.txt"

    # ALWAYS call inflate.sh — this matches the exact contest evaluation path.
    # inflate.sh sources config.env (getting ALL INFLATE_* vars), then
    # delegates to inflate_postfilter.py or ffmpeg depending on PYTHON_INFLATE.
    # This eliminates env var propagation bugs and ensures test == contest.
    inflate_sh = submission_src / "inflate.sh"
    if not inflate_sh.exists():
        raise click.ClickException(f"inflate.sh not found: {inflate_sh}")

    cmd = [
        "bash", str(inflate_sh),
        str(mgr.archive_unzip_dir),
        str(mgr.inflated_dir),
        str(video_names_file),
    ]
    inflate_env = {
        **os.environ,
        "PYTHONPATH": f"{submission_src.parent.parent / 'src'}:{upstream_dir}",
        "COMMA_CHALLENGE_ROOT": str(upstream_dir),
        "CONFIG_ENV_PATH": str(submission_src / "config.env"),
    }

    run_stage(
        cmd,
        mgr.log_dir / "inflate.log",
        timeout=1800,  # 30 min — matches contest time limit
        label="inflate (inflate.sh)",
        env=inflate_env,
    )

    # Validate inflated output
    names = [
        ln.strip()
        for ln in video_names_file.read_text().splitlines()
        if ln.strip()
    ]
    missing = []
    for name in names:
        stem = name.rsplit(".", 1)[0]
        raw = mgr.inflated_dir / f"{stem}.raw"
        if not raw.exists():
            missing.append(str(raw))

    if missing:
        raise click.ClickException(
            f"Inflate completed but {len(missing)} raw file(s) missing:\n"
            + "\n".join(f"  {m}" for m in missing)
        )

    # Validate frame count / size using SOURCE_W x SOURCE_H from config.env
    config = _parse_config_env(submission_src / "config.env")
    source_w = int(config.get("SOURCE_W", "1164"))
    source_h = int(config.get("SOURCE_H", "874"))
    for name in names:
        stem = name.rsplit(".", 1)[0]
        raw = mgr.inflated_dir / f"{stem}.raw"
        raw_size = raw.stat().st_size
        frame_bytes = source_w * source_h * 3
        if raw_size % frame_bytes != 0:
            raise click.ClickException(
                f"Inflated file {raw} has size {raw_size:,} bytes, "
                f"not a multiple of {frame_bytes:,} ({source_w}x{source_h}x3). "
                f"Frame count would be {raw_size / frame_bytes:.2f}."
            )
        n_frames = raw_size // frame_bytes
        click.echo(f"  {raw.name}: {n_frames} frames, {raw_size:,} bytes")

    mgr.set_stage(RunState.INFLATED)


def _find_scorer_python(upstream_dir: Path) -> str:
    """Find the best Python to run the scorer with.

    Preference order:
    1. Upstream venv Python (matches auth environment exactly)
    2. Current sys.executable (fallback)

    Using the upstream venv eliminates ANY version discrepancies in
    torch, timm, einops, etc. The upstream venv has torch 2.10.0 +
    timm 1.0.22 which is what auth uses.
    """
    upstream_python = upstream_dir / ".venv" / "bin" / "python"
    if upstream_python.exists():
        # Verify it actually works
        try:
            result = subprocess.run(
                [str(upstream_python), "-c", "import torch; print(torch.__version__)"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                click.echo(f"  Using upstream venv Python: {upstream_python}")
                click.echo(f"  Upstream torch: {result.stdout.strip()}")
                return str(upstream_python)
        except (subprocess.TimeoutExpired, OSError):
            pass
    click.echo(f"  Using current Python: {sys.executable}")
    return sys.executable


def stage_score(
    mgr: RunManager,
    upstream_dir: Path,
    submission_src: Path,
    device: str = "cpu",
) -> None:
    """Score: run upstream evaluate.py against inflated output.

    Prefers the upstream venv Python for scoring to minimize
    divergence from the auth scorer environment.
    """
    mgr.set_stage(RunState.SCORING)

    report_path = mgr.run_dir / "report.json"
    report_txt = mgr.run_dir / "report.txt"

    scorer_python = _find_scorer_python(upstream_dir)

    # The scorer expects:
    #   --submission-dir pointing to a dir with archive.zip and inflated/
    #   --uncompressed-dir pointing to videos/
    cmd = [
        scorer_python,
        str(upstream_dir / "evaluate.py"),
        "--submission-dir", str(mgr.submission_dir),
        "--uncompressed-dir", str(upstream_dir / "videos"),
        "--report", str(report_txt),
        "--video-names-file", str(upstream_dir / "public_test_video_names.txt"),
        "--device", device,
    ]

    run_stage(
        cmd,
        mgr.log_dir / "score.log",
        timeout=3600,
        label="score",
        cwd=upstream_dir,
    )

    # Parse report.txt into structured JSON
    _parse_and_save_report(report_txt, report_path, mgr, submission_src, device)


def _parse_and_save_report(
    report_txt: Path,
    report_json: Path,
    mgr: RunManager,
    submission_src: Path,
    device: str,
) -> None:
    """Parse the upstream report.txt into structured JSON and print summary."""
    text = report_txt.read_text()
    result: dict[str, Any] = {"raw": text}

    for line in text.splitlines():
        line = line.strip()
        if "PoseNet Distortion" in line:
            result["posenet_distortion"] = float(line.split(":")[-1].strip())
        elif "SegNet Distortion" in line:
            result["segnet_distortion"] = float(line.split(":")[-1].strip())
        elif "Compression Rate" in line:
            result["rate"] = float(line.split(":")[-1].strip())
        elif "Submission file size" in line:
            raw = line.split(":")[-1].strip().replace(",", "").replace("bytes", "").strip()
            result["submission_bytes"] = int(raw)
        elif "Final score" in line:
            # "100*segnet_dist + √(10*posenet_dist) + 25*rate = 1.52"
            parts = line.split("=")
            if len(parts) >= 2:
                try:
                    result["score"] = float(parts[-1].strip())
                except ValueError:
                    pass

    report_json.write_text(json.dumps(result, indent=2) + "\n")

    # Print summary
    seg = result.get("segnet_distortion", 0)
    pose = result.get("posenet_distortion", 0)
    rate = result.get("rate", 0)
    score = result.get("score")
    if score is None:
        score = 100 * seg + math.sqrt(10 * pose) + 25 * rate

    click.echo("")
    click.echo("=" * 54)
    click.echo(f"  LOCAL SCORE: {score:.4f}")
    click.echo(f"  SegNet:  {seg:.6f}  (100x  = {100 * seg:.4f})")
    click.echo(f"  PoseNet: {pose:.6f}  (sqrt(10x) = {math.sqrt(10 * pose):.4f})")
    click.echo(f"  Rate:    {rate:.6f}  (25x   = {25 * rate:.4f})")
    click.echo("=" * 54)

    # Calibrated auth estimate
    cal = calibrate_score(pose, seg, rate)
    click.echo(f"  CALIBRATED AUTH ESTIMATE: ~{cal['calibrated_score']:.2f}")
    click.echo(f"  Calibrated PoseNet: {cal['calibrated_pose']:.8f}  "
               f"(factor: {cal['pose_calibration_factor']:.4f}x)")
    click.echo(f"  (Based on archive md5 {cal['calibration_archive_md5']}, "
               f"n={cal['calibration_n_points']} calibration point(s))")
    click.echo(f"  WARNING: Calibration is PROVISIONAL. Submit for definitive score.")
    click.echo("=" * 54)

    mgr.set_stage(
        RunState.SCORED,
        score=score,
        calibrated_score=cal["calibrated_score"],
        posenet=pose,
        calibrated_posenet=cal["calibrated_pose"],
        segnet=seg,
        rate=rate,
    )

    # ── Complete experiment record ──────────────────────────────
    # Capture EVERYTHING needed to reproduce this result.
    _save_experiment_record(mgr, submission_src, result, device)


def _save_experiment_record(
    mgr: RunManager,
    submission_src: Path,
    scorer_result: dict[str, Any],
    device: str,
) -> None:
    """Save a complete, self-contained experiment record.

    Captures all config, hashes, git state, and results so this exact
    score can be reproduced and audited later. Writes to both
    run_dir/experiment_record.json AND appends to reports/results.jsonl.
    """
    config = _parse_config_env(submission_src / "config.env")

    # Archive hash
    archive_path = mgr.run_dir / "submission" / "archive.zip"
    archive_md5 = ""
    archive_bytes = 0
    if archive_path.exists():
        archive_bytes = archive_path.stat().st_size
        h = hashlib.md5()
        with open(archive_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        archive_md5 = h.hexdigest()

    # Checkpoint hash
    # Checkpoint is inside the unzipped archive (archive/ dir), not submission/ root
    ckpt_path = mgr.run_dir / "submission" / "archive" / "postfilter_int8.pt"
    if not ckpt_path.exists():
        # Fallback: check submission root and submission_src
        for fallback in [
            mgr.run_dir / "submission" / "postfilter_int8.pt",
            submission_src / "postfilter_int8.pt",
        ]:
            if fallback.exists():
                ckpt_path = fallback
                break
    ckpt_md5 = ""
    if ckpt_path.exists():
        h = hashlib.md5()
        with open(ckpt_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        ckpt_md5 = h.hexdigest()

    # Git commit
    git_commit = ""
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(submission_src),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass

    # Machine identifier
    import platform
    machine_id = f"{platform.node()}_{platform.system()}_{platform.machine()}"

    record = {
        "run_id": mgr.run_dir.name,
        "ts_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "score": scorer_result.get("score"),
        "posenet_distortion": scorer_result.get("posenet_distortion"),
        "segnet_distortion": scorer_result.get("segnet_distortion"),
        "rate": scorer_result.get("rate"),
        "archive_bytes": archive_bytes,
        "archive_md5": archive_md5,
        "checkpoint_md5": ckpt_md5,
        "git_commit": git_commit,
        "device": device,
        "machine": machine_id,
        "config_env": dict(config),
        "inflate_flags": {
            "brightness_shift": config.get("INFLATE_BRIGHTNESS_SHIFT", "0"),
            "chroma_smooth": config.get("INFLATE_CHROMA_SMOOTH", "0"),
            "deblock": config.get("INFLATE_DEBLOCK", "0"),
            "multi_pass": config.get("INFLATE_MULTI_PASS", "1"),
        },
        "run_dir": str(mgr.run_dir),
    }

    # Save to run directory
    record_path = mgr.run_dir / "experiment_record.json"
    record_path.write_text(json.dumps(record, indent=2) + "\n")

    # Append to results.jsonl (project-level)
    results_jsonl = submission_src.parent.parent / "reports" / "results.jsonl"
    if results_jsonl.parent.exists():
        with open(results_jsonl, "a") as f:
            f.write(json.dumps(record) + "\n")
        click.echo(f"  Record saved: {record_path}", err=True)
        click.echo(f"  Appended to: {results_jsonl}", err=True)
    else:
        click.echo(f"  Record saved: {record_path}", err=True)
        click.echo(f"  WARNING: reports/ dir not found, results.jsonl not updated", err=True)


# ============================================================
# CLI
# ============================================================
@click.group()
def cli():
    """Submission pipeline runner for robust_current."""


@cli.command()
@click.option(
    "--upstream-dir",
    envvar="COMMA_CHALLENGE_ROOT",
    required=True,
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
    help="Path to the upstream challenge repo (or set COMMA_CHALLENGE_ROOT).",
)
@click.option(
    "--work-dir",
    default=None,
    type=click.Path(resolve_path=True),
    help="Persistent working directory for runs. Default: ./eval_runs",
)
@click.option(
    "--run-name",
    default=None,
    help="Name for this run (default: timestamp-based).",
)
@click.option(
    "--device",
    default="cpu",
    help="Torch device for scoring (cpu, cuda, mps).",
)
@click.option(
    "--skip-compress",
    is_flag=True,
    default=False,
    help="Skip compress stage, use existing archive.zip from submission dir.",
)
def evaluate(
    upstream_dir: str,
    work_dir: str | None,
    run_name: str | None,
    device: str,
    skip_compress: bool,
) -> None:
    """Full pipeline: compress -> inflate -> score.

    Manages state, checkpoints progress, survives restarts.
    """
    upstream = Path(upstream_dir).resolve()
    submission_src = Path(__file__).resolve().parent

    if work_dir is None:
        work_dir_path = submission_src / "eval_runs"
    else:
        work_dir_path = Path(work_dir)

    if run_name is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        cfg_hash = config_fingerprint(submission_src / "config.env")
        run_name = f"{ts}_{cfg_hash}"

    mgr = RunManager(work_dir_path / run_name)
    mgr.ensure_dirs()

    click.echo(f"Run directory: {mgr.run_dir}")
    click.echo(f"Run name:      {run_name}")

    # -- Pre-flight ------------------------------------------------
    click.echo("\n--- Pre-flight checks ---")
    all_errors: list[str] = []
    all_errors.extend(preflight_upstream(upstream))
    all_errors.extend(preflight_tools())
    all_errors.extend(preflight_submission(submission_src))

    if all_errors:
        click.echo(click.style("PREFLIGHT FAILED:", fg="red", bold=True))
        for e in all_errors:
            click.echo(f"  - {e}")
        mgr.record_failure(RunState.PREFLIGHT_OK, "\n".join(all_errors))
        raise click.ClickException(f"Preflight failed with {len(all_errors)} error(s)")

    # Checkpoint identity verification (CRITICAL — catches wrong-checkpoint bugs)
    ckpt_warnings = preflight_checkpoint_identity(submission_src)
    config_warnings = preflight_config_match(submission_src)
    all_warnings = ckpt_warnings + config_warnings
    if all_warnings:
        click.echo(click.style("WARNINGS:", fg="yellow", bold=True))
        for w in all_warnings:
            click.echo(f"  - {w}")

    click.echo(click.style("All pre-flight checks passed.", fg="green"))

    # Snapshot config.env
    shutil.copy2(submission_src / "config.env", mgr.run_dir / "config.env")
    mgr.set_stage(
        RunState.PREFLIGHT_OK,
        config_hash=config_fingerprint(submission_src / "config.env"),
        upstream_dir=str(upstream),
        device=device,
        submission_src=str(submission_src),
        started=datetime.datetime.now().isoformat(),
    )

    _run_pipeline(mgr, upstream, submission_src, device, skip_compress)


def _run_pipeline(
    mgr: RunManager,
    upstream: Path,
    submission_src: Path,
    device: str,
    skip_compress: bool,
) -> None:
    """Execute pipeline stages, skipping already-completed ones."""
    lock_path = mgr.run_dir / ".lock"
    _acquire_lock(lock_path)
    current = mgr.get_stage()

    try:
        # -- Compress -----------------------------------------------
        if current in (RunState.PREFLIGHT_OK, RunState.COMPRESSING):
            if skip_compress:
                click.echo("\n--- Compress: SKIPPED (--skip-compress) ---")
                # Copy existing archive.zip
                src_zip = submission_src / "archive.zip"
                if not src_zip.exists():
                    raise click.ClickException(
                        f"--skip-compress but no archive.zip at {src_zip}"
                    )
                shutil.copy2(src_zip, mgr.archive_zip)
                # Unzip it
                _unzip_archive(mgr)
                mgr.set_stage(RunState.PACKAGED, archive_bytes=mgr.archive_zip.stat().st_size)
            else:
                click.echo("\n--- Compress ---")
                stage_compress(mgr, upstream, submission_src)
                click.echo("\n--- Package ---")
                stage_package(mgr, submission_src)
            current = mgr.get_stage()

        # If we already compressed but not packaged
        if current == RunState.COMPRESSED:
            click.echo("\n--- Package ---")
            stage_package(mgr, submission_src)
            current = mgr.get_stage()

        # -- Inflate -----------------------------------------------
        if current in (RunState.PACKAGED, RunState.INFLATING):
            click.echo("\n--- Inflate ---")
            stage_inflate(mgr, submission_src, upstream, device)
            current = mgr.get_stage()

        # -- Score -------------------------------------------------
        if current in (RunState.INFLATED, RunState.SCORING):
            click.echo("\n--- Score ---")
            stage_score(mgr, upstream, submission_src, device)
            current = mgr.get_stage()

        if current == RunState.SCORED:
            click.echo(click.style("\nPipeline complete.", fg="green", bold=True))

    except (click.ClickException, click.Abort) as exc:
        msg = str(exc)
        mgr.record_failure(mgr.get_stage(), msg)
        raise
    except Exception as exc:
        mgr.record_failure(mgr.get_stage(), str(exc))
        raise click.ClickException(
            f"Unexpected error: {exc}\nRun dir: {mgr.run_dir}\nUse `runner.py resume {mgr.run_dir.name}` to retry."
        ) from exc
    finally:
        _release_lock(lock_path)


def _unzip_archive(mgr: RunManager) -> None:
    """Unzip archive.zip into the archive directory."""
    if mgr.archive_unzip_dir.exists():
        shutil.rmtree(mgr.archive_unzip_dir)
    mgr.archive_unzip_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(mgr.archive_zip, "r") as zf:
        zf.extractall(mgr.archive_unzip_dir)


@cli.command()
@click.argument("run_name")
@click.option(
    "--work-dir",
    default=None,
    type=click.Path(resolve_path=True),
    help="Working directory. Default: ./eval_runs",
)
def status(run_name: str, work_dir: str | None) -> None:
    """Check status of a run."""
    submission_src = Path(__file__).resolve().parent
    if work_dir is None:
        work_dir_path = submission_src / "eval_runs"
    else:
        work_dir_path = Path(work_dir)

    run_dir = work_dir_path / run_name
    if not run_dir.exists():
        # Try listing available runs
        if work_dir_path.exists():
            runs = sorted(
                [d.name for d in work_dir_path.iterdir() if d.is_dir()],
                reverse=True,
            )
            if runs:
                click.echo(f"Run '{run_name}' not found. Available runs:")
                for r in runs[:10]:
                    click.echo(f"  {r}")
            else:
                click.echo(f"No runs found in {work_dir_path}")
        else:
            click.echo(f"Work directory does not exist: {work_dir_path}")
        raise click.ClickException(f"Run '{run_name}' not found")

    mgr = RunManager(run_dir)
    data = mgr.load_state()

    click.echo(f"Run:       {run_name}")
    click.echo(f"Directory: {mgr.run_dir}")
    click.echo(f"Stage:     {data.get('stage', 'unknown')}")

    if data.get("stage") == RunState.FAILED.value:
        click.echo(f"Failed at: {data.get('failed_at', 'unknown')}")
        click.echo(f"Error:     {data.get('error', 'unknown')}")
        click.echo(f"Time:      {data.get('failed_time', 'unknown')}")
    elif data.get("stage") == RunState.SCORED.value:
        click.echo(f"Score:     {data.get('score', 'N/A')}")
        click.echo(f"PoseNet:   {data.get('posenet', 'N/A')}")
        click.echo(f"SegNet:    {data.get('segnet', 'N/A')}")
        click.echo(f"Rate:      {data.get('rate', 'N/A')}")

    if "config_hash" in data:
        click.echo(f"Config:    {data['config_hash']}")
    if "started" in data:
        click.echo(f"Started:   {data['started']}")
    if "stage_updated" in data:
        click.echo(f"Updated:   {data['stage_updated']}")

    # Show archive size if available
    if mgr.archive_zip.exists():
        click.echo(f"Archive:   {mgr.archive_zip.stat().st_size:,} bytes")

    # Show log files
    if mgr.log_dir.exists():
        logs = sorted(mgr.log_dir.glob("*.log"))
        if logs:
            click.echo("Logs:")
            for log in logs:
                size = log.stat().st_size
                click.echo(f"  {log.name} ({size:,} bytes)")


@cli.command()
@click.argument("run_name")
@click.option(
    "--upstream-dir",
    envvar="COMMA_CHALLENGE_ROOT",
    required=True,
    type=click.Path(exists=True, file_okay=False, resolve_path=True),
)
@click.option(
    "--work-dir",
    default=None,
    type=click.Path(resolve_path=True),
)
@click.option("--device", default="cpu")
def resume(
    run_name: str,
    upstream_dir: str,
    work_dir: str | None,
    device: str,
) -> None:
    """Resume a failed/killed run from last checkpoint."""
    upstream = Path(upstream_dir).resolve()
    submission_src = Path(__file__).resolve().parent

    if work_dir is None:
        work_dir_path = submission_src / "eval_runs"
    else:
        work_dir_path = Path(work_dir)

    run_dir = work_dir_path / run_name
    if not run_dir.exists():
        raise click.ClickException(f"Run directory not found: {run_dir}")

    mgr = RunManager(run_dir)
    data = mgr.load_state()
    current = mgr.get_stage()

    click.echo(f"Resuming run: {run_name}")
    click.echo(f"Current stage: {current.value}")

    if current == RunState.SCORED:
        click.echo("Run already completed. Nothing to resume.")
        return

    # Determine what to resume
    target = mgr.first_incomplete_stage()
    click.echo(f"Will resume from: {target.value}")

    # Override device if provided
    if device:
        data["device"] = device
        mgr.save_state(data)

    # For resume, we need to figure out if compress was skipped
    skip_compress = current in (
        RunState.PACKAGED,
        RunState.INFLATING,
        RunState.INFLATED,
        RunState.SCORING,
    ) or (
        current == RunState.FAILED
        and data.get("failed_at") in (
            RunState.INFLATING.value,
            RunState.INFLATED.value,
            RunState.SCORING.value,
        )
    )

    # Reset from FAILED to the stage we want to resume from
    if current == RunState.FAILED:
        failed_at = data.get("failed_at", RunState.PREFLIGHT_OK.value)
        # Map failed_at to the previous completed stage
        resume_map = {
            RunState.PREFLIGHT_OK.value: RunState.PREFLIGHT_OK,
            RunState.COMPRESSING.value: RunState.PREFLIGHT_OK,
            RunState.COMPRESSED.value: RunState.COMPRESSED,
            RunState.PACKAGING.value: RunState.COMPRESSED,
            RunState.PACKAGED.value: RunState.PACKAGED,
            RunState.INFLATING.value: RunState.PACKAGED,
            RunState.INFLATED.value: RunState.INFLATED,
            RunState.SCORING.value: RunState.INFLATED,
        }
        resume_from = resume_map.get(failed_at, RunState.PREFLIGHT_OK)
        mgr.set_stage(resume_from)

    _run_pipeline(mgr, upstream, submission_src, device, skip_compress)


@cli.command("list")
@click.option(
    "--work-dir",
    default=None,
    type=click.Path(resolve_path=True),
)
def list_runs(work_dir: str | None) -> None:
    """List all runs."""
    submission_src = Path(__file__).resolve().parent
    if work_dir is None:
        work_dir_path = submission_src / "eval_runs"
    else:
        work_dir_path = Path(work_dir)

    if not work_dir_path.exists():
        click.echo(f"No runs directory: {work_dir_path}")
        return

    runs = sorted(
        [d for d in work_dir_path.iterdir() if d.is_dir()],
        key=lambda d: d.stat().st_mtime,
        reverse=True,
    )
    if not runs:
        click.echo("No runs found.")
        return

    click.echo(f"{'NAME':<45} {'STAGE':<15} {'SCORE':<10}")
    click.echo("-" * 70)
    for run_dir in runs:
        mgr = RunManager(run_dir)
        data = mgr.load_state()
        stage = data.get("stage", "?")
        score = data.get("score", "")
        if isinstance(score, float):
            score = f"{score:.4f}"
        click.echo(f"{run_dir.name:<45} {stage:<15} {score:<10}")


if __name__ == "__main__":
    cli()
