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
    """Package: bundle compressed video(s) + postfilter into archive.zip."""
    mgr.set_stage(RunState.PACKAGING)

    # Copy postfilter into archive dir
    pf_src = submission_src / "postfilter_int8.pt"
    pf_dst = mgr.archive_unzip_dir / "postfilter_int8.pt"
    if pf_src.exists():
        shutil.copy2(pf_src, pf_dst)

    # Create archive.zip
    with zipfile.ZipFile(mgr.archive_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for f in sorted(mgr.archive_unzip_dir.rglob("*")):
            if f.is_file():
                zf.write(f, f.relative_to(mgr.archive_unzip_dir))

    size = mgr.archive_zip.stat().st_size
    click.echo(f"  archive.zip: {size:,} bytes")

    # Also copy archive.zip to submission_dir root for scorer
    # (scorer expects submission_dir/archive.zip)
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

    # Use the Python postfilter inflate path (matches config PYTHON_INFLATE=postfilter)
    config = _parse_config_env(submission_src / "config.env")
    python_inflate = config.get("PYTHON_INFLATE", "0")

    if python_inflate == "postfilter":
        postfilter_path = mgr.archive_unzip_dir / "postfilter_int8.pt"
        if not postfilter_path.exists():
            postfilter_path = submission_src / "postfilter_int8.pt"

        inflate_script = submission_src / "inflate_postfilter.py"
        cmd = [
            sys.executable,
            str(inflate_script),
            str(mgr.archive_unzip_dir),
            str(mgr.inflated_dir),
            str(video_names_file),
            str(postfilter_path),
            "--device", device,
        ]
        # Propagate config.env settings as env vars so inflate_postfilter.py
        # picks them up via Click's envvar= bindings
        inflate_env = {
            **os.environ,
            "PYTHONPATH": f"{submission_src.parent.parent / 'src'}:{upstream_dir}",
            "COMMA_CHALLENGE_ROOT": str(upstream_dir),
            "INFLATE_BRIGHTNESS_SHIFT": config.get("INFLATE_BRIGHTNESS_SHIFT", "0"),
            "INFLATE_CHROMA_SMOOTH": config.get("INFLATE_CHROMA_SMOOTH", "0"),
            "INFLATE_DEBLOCK": config.get("INFLATE_DEBLOCK", "0"),
            "INFLATE_MULTI_PASS": config.get("INFLATE_MULTI_PASS", "1"),
        }
        run_stage(
            cmd,
            mgr.log_dir / "inflate.log",
            timeout=900,
            label="inflate (postfilter)",
            env=inflate_env,
        )
    else:
        # Use inflate.sh for ffmpeg-based inflation
        inflate_sh = submission_src / "inflate.sh"
        cmd = [
            "bash", str(inflate_sh),
            str(mgr.archive_unzip_dir),
            str(mgr.inflated_dir),
            str(video_names_file),
        ]
        env = {"CONFIG_ENV_PATH": str(submission_src / "config.env")}
        run_stage(
            cmd,
            mgr.log_dir / "inflate.log",
            timeout=900,
            label="inflate (ffmpeg)",
            env=env,
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


def stage_score(
    mgr: RunManager,
    upstream_dir: Path,
    submission_src: Path,
    device: str = "cpu",
) -> None:
    """Score: run upstream evaluate.py against inflated output."""
    mgr.set_stage(RunState.SCORING)

    report_path = mgr.run_dir / "report.json"
    report_txt = mgr.run_dir / "report.txt"

    # The scorer expects:
    #   --submission-dir pointing to a dir with archive.zip and inflated/
    #   --uncompressed-dir pointing to videos/
    cmd = [
        sys.executable,
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
    click.echo(f"  SCORE: {score:.4f}")
    click.echo(f"  SegNet:  {seg:.6f}  (100x  = {100 * seg:.4f})")
    click.echo(f"  PoseNet: {pose:.6f}  (sqrt(10x) = {math.sqrt(10 * pose):.4f})")
    click.echo(f"  Rate:    {rate:.6f}  (25x   = {25 * rate:.4f})")
    click.echo("=" * 54)

    mgr.set_stage(RunState.SCORED, score=score, posenet=pose, segnet=seg, rate=rate)

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

    # Warnings (non-blocking)
    config_warnings = preflight_config_match(submission_src)
    if config_warnings:
        click.echo(click.style("WARNINGS:", fg="yellow", bold=True))
        for w in config_warnings:
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
