#!/usr/bin/env python
"""One-command evaluation pipeline: compress -> inflate -> score.

Instead of running three separate scripts and juggling env vars, run:

    python eval.py --upstream-dir workspace/upstream/comma_video_compression_challenge

This handles the full pipeline with sane defaults that match config.env.
All paths are resolved to absolute immediately.  Pre-flight checks run
before any expensive computation so you don't discover a missing file
25 minutes into a scoring run.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import click

from config_env import load_config_env


# ── Helpers ──────────────────────────────────────────────────────────


def _fmt_elapsed(seconds: float) -> str:
    """Format seconds into '2m 15s' or '45s'."""
    m, s = divmod(int(seconds), 60)
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _run(
    cmd: list[str],
    label: str,
    *,
    env: dict[str, str] | None = None,
    cwd: str | Path | None = None,
    verbose: bool = True,
    timeout: int | None = None,
) -> subprocess.CompletedProcess:
    """Run a subprocess with timing, error handling, and stderr capture."""
    if verbose:
        click.echo(f"\n{'='*60}", err=True)
        click.echo(f"  [{label}] starting ...", err=True)
        click.echo(f"{'='*60}", err=True)
        click.echo(f"  $ {' '.join(cmd)}", err=True)

    t0 = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            env=env,
            cwd=cwd,
            capture_output=not verbose,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - t0
        raise click.ClickException(
            f"[{label}] timed out after {_fmt_elapsed(elapsed)}"
        )

    elapsed = time.monotonic() - t0
    if verbose:
        click.echo(
            f"  [{label}] {_fmt_elapsed(elapsed)} (exit {result.returncode})",
            err=True,
        )

    if result.returncode != 0:
        stderr_text = ""
        if not verbose and result.stderr:
            stderr_text = result.stderr.decode(errors="replace")
            click.echo(stderr_text, err=True)
        raise click.ClickException(
            f"[{label}] failed with exit code {result.returncode}"
        )
    return result


# ── Pre-flight ───────────────────────────────────────────────────────


def _preflight(
    upstream: Path,
    archive_zip: Path | None,
    skip_compress: bool,
    skip_inflate: bool,
    skip_score: bool,
    self_dir: Path,
) -> list[str]:
    """Validate everything exists before starting the 30-min pipeline.

    Returns a list of error strings (empty = all good).
    """
    errors: list[str] = []

    # Upstream directory contents
    required_upstream = [
        "evaluate.py",
        "frame_utils.py",
        "modules.py",
        "models/posenet.safetensors",
        "models/segnet.safetensors",
        "public_test_video_names.txt",
    ]
    # Only require videos when we'll actually score
    if not skip_score:
        required_upstream.append("videos/0.mkv")

    for rel in required_upstream:
        if not (upstream / rel).exists():
            errors.append(f"Missing upstream file: {upstream / rel}")

    # archive.zip (needed when skipping compress or for scoring)
    if skip_compress and not skip_inflate:
        if archive_zip is None:
            candidate = self_dir / "archive.zip"
            if not candidate.exists():
                errors.append(
                    f"--skip-compress requires archive.zip; not found at {candidate}"
                )
        elif not archive_zip.exists():
            errors.append(f"archive.zip not found: {archive_zip}")

    # inflate_postfilter.py
    if not skip_inflate:
        inflate_script = self_dir / "inflate_postfilter.py"
        if not inflate_script.exists():
            errors.append(f"inflate_postfilter.py not found at {inflate_script}")

    # compress.sh
    if not skip_compress:
        compress_sh = self_dir / "compress.sh"
        if not compress_sh.exists():
            errors.append(f"compress.sh not found at {compress_sh}")

    # ffmpeg
    if shutil.which("ffmpeg") is None:
        errors.append("ffmpeg not found in PATH")

    return errors


# ── Parse upstream report.txt into structured data ───────────────────


def _parse_report_txt(report_path: Path) -> dict:
    """Extract numbers from the upstream evaluator's report.txt."""
    text = report_path.read_text()
    result: dict = {}

    patterns = {
        "posenet_distortion": r"Average PoseNet Distortion:\s+([\d.eE+-]+)",
        "segnet_distortion": r"Average SegNet Distortion:\s+([\d.eE+-]+)",
        "compressed_size": r"Submission file size:\s+([\d,]+)",
        "uncompressed_size": r"Original uncompressed size:\s+([\d,]+)",
        "rate": r"Compression Rate:\s+([\d.eE+-]+)",
        "total_score": r"Final score:.*?=\s+([\d.]+)",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            val = m.group(1).replace(",", "")
            result[key] = float(val)

    return result


# ── Main command ─────────────────────────────────────────────────────


@click.command()
@click.option(
    "--config",
    "config_path",
    type=click.Path(),
    default="config.env",
    help="Config env file to load before running.",
)
@click.option(
    "--upstream-dir",
    envvar="COMMA_CHALLENGE_ROOT",
    required=True,
    type=click.Path(exists=True),
    help="Path to comma_video_compression_challenge repo.",
)
@click.option(
    "--archive-zip",
    default=None,
    type=click.Path(),
    help="Path to archive.zip (default: ./archive.zip in submission dir).",
)
@click.option(
    "--work-dir",
    default=None,
    help="Working directory (default: a fresh tac_eval_XXXXX dir under the system temp root).",
)
@click.option("--skip-compress", is_flag=True, help="Skip compression.")
@click.option("--skip-inflate", is_flag=True, help="Skip inflation.")
@click.option("--skip-score", is_flag=True, help="Skip scoring.")
@click.option(
    "--brightness-shift/--no-brightness-shift",
    envvar="INFLATE_BRIGHTNESS_SHIFT",
    default=True,
    help="Shift luminance toward midpoint.",
)
@click.option(
    "--chroma-smooth/--no-chroma-smooth",
    envvar="INFLATE_CHROMA_SMOOTH",
    default=True,
    help="Smooth chroma channels.",
)
@click.option(
    "--deblock/--no-deblock",
    envvar="INFLATE_DEBLOCK",
    default=False,
    help="Apply NLM deblocking.",
)
@click.option(
    "--multi-pass",
    type=int,
    envvar="INFLATE_MULTI_PASS",
    default=1,
    help="Post-filter passes (2=double pass).",
)
@click.option(
    "--device",
    default="cpu",
    help="Inference device: cpu, cuda, or mps.",
)
@click.option("--verbose/--quiet", default=True, help="Print progress to stderr.")
def evaluate(
    config_path,
    upstream_dir,
    archive_zip,
    work_dir,
    skip_compress,
    skip_inflate,
    skip_score,
    brightness_shift,
    chroma_smooth,
    deblock,
    multi_pass,
    device,
    verbose,
):
    """Full pipeline: compress -> inflate -> score.  One command.

    \b
    All paths are resolved to absolute immediately.
    Pre-flight checks validate inputs before any computation.

    \b
    Examples:
      # Full eval with defaults from config.env:
      python eval.py --upstream-dir path/to/upstream

      # Re-score without re-compressing (iterate on inflate settings):
      python eval.py --skip-compress --multi-pass 2 --upstream-dir ...

      # Just compress + inflate, skip scoring:
      python eval.py --skip-score --upstream-dir ...
    """
    t_pipeline = time.monotonic()

    # ── Resolve core paths ───────────────────────────────────────────
    self_dir = Path(__file__).resolve().parent
    upstream = Path(upstream_dir).resolve()
    src_dir = (self_dir.parent.parent / "src").resolve()

    # Load config.env into os.environ so sub-scripts pick up settings
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = self_dir / cfg_path
    config = load_config_env(cfg_path, into_environ=True)
    if verbose and config:
        click.echo(f"Loaded {len(config)} settings from {cfg_path}", err=True)

    # Propagate Click flags into env
    os.environ["INFLATE_BRIGHTNESS_SHIFT"] = "1" if brightness_shift else "0"
    os.environ["INFLATE_CHROMA_SMOOTH"] = "1" if chroma_smooth else "0"
    os.environ["INFLATE_DEBLOCK"] = "1" if deblock else "0"
    os.environ["INFLATE_MULTI_PASS"] = str(multi_pass)
    os.environ["COMMA_CHALLENGE_ROOT"] = str(upstream)

    # Resolve archive.zip
    if archive_zip is not None:
        archive_zip = Path(archive_zip).resolve()
    else:
        archive_zip = self_dir / "archive.zip"

    video_names_file = (upstream / "public_test_video_names.txt").resolve()

    # ── Pre-flight checks ────────────────────────────────────────────
    if verbose:
        click.echo(f"\nUpstream:    {upstream}", err=True)
        click.echo(f"Submission:  {self_dir}", err=True)
        click.echo(f"Archive:     {archive_zip}", err=True)
        click.echo(f"Device:      {device}", err=True)

    errors = _preflight(upstream, archive_zip, skip_compress, skip_inflate, skip_score, self_dir)
    if errors:
        click.echo(f"\n  PRE-FLIGHT FAILED ({len(errors)} errors):", err=True)
        for e in errors:
            click.echo(f"    ERROR: {e}", err=True)
        raise click.ClickException(f"{len(errors)} pre-flight checks failed")

    if verbose:
        click.echo("  Pre-flight: all checks passed", err=True)

    # ── Work directory setup ─────────────────────────────────────────
    work = Path(work_dir or tempfile.mkdtemp(prefix="tac_eval_")).resolve()
    submission_dir = work / "submission"
    inflated_dir = submission_dir / "inflated"
    submission_dir.mkdir(parents=True, exist_ok=True)
    inflated_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        click.echo(f"Work dir:    {work}", err=True)

    stage_times: dict[str, float] = {}

    # ── Step 1: Compress ─────────────────────────────────────────────
    if not skip_compress:
        t0 = time.monotonic()
        compress_sh = self_dir / "compress.sh"
        _run(["bash", str(compress_sh)], "compress", verbose=verbose, cwd=str(self_dir))
        stage_times["compress"] = time.monotonic() - t0
        # After compression, archive.zip should exist in self_dir
        archive_zip = self_dir / "archive.zip"
        if not archive_zip.exists():
            raise click.ClickException(
                f"compress.sh ran but archive.zip not found at {archive_zip}"
            )
    else:
        if verbose:
            click.echo("\n  Skipping compression (--skip-compress)", err=True)
        if not archive_zip.exists():
            raise click.ClickException(
                f"--skip-compress requires existing {archive_zip}"
            )

    # ── Step 2: Unpack archive + copy zip into submission_dir ────────
    if not skip_inflate or not skip_score:
        # Unzip into submission_dir (evaluator reads inflated/ from here)
        archive_unpack_dir = submission_dir
        _run(
            ["unzip", "-o", str(archive_zip), "-d", str(archive_unpack_dir)],
            "unpack-archive",
            verbose=verbose,
        )
        # Copy archive.zip itself — evaluator needs it for rate calculation
        dest_zip = submission_dir / "archive.zip"
        if dest_zip.resolve() != archive_zip.resolve():
            shutil.copy2(str(archive_zip), str(dest_zip))
        if verbose:
            click.echo(f"  archive.zip copied to {dest_zip}", err=True)

    # ── Step 3: Inflate ──────────────────────────────────────────────
    if not skip_inflate:
        t0 = time.monotonic()
        inflate_script = self_dir / "inflate_postfilter.py"

        # Discover postfilter weights inside the unpacked archive
        postfilter_path = None
        for candidate in [
            submission_dir / "postfilter_int8.pt",
            self_dir / "postfilter_int8.pt",
        ]:
            if candidate.exists():
                postfilter_path = candidate.resolve()
                break

        inflate_cmd = [
            sys.executable,
            str(inflate_script),
            str(submission_dir),     # archive_dir (contains .mkv files)
            str(inflated_dir),       # output_dir
            str(video_names_file),   # video_names_file
        ]
        if postfilter_path is not None:
            inflate_cmd.append(str(postfilter_path))

        # Pass Click options through
        if brightness_shift:
            inflate_cmd.append("--brightness-shift")
        else:
            inflate_cmd.append("--no-brightness-shift")
        if chroma_smooth:
            inflate_cmd.append("--chroma-smooth")
        else:
            inflate_cmd.append("--no-chroma-smooth")
        if deblock:
            inflate_cmd.append("--deblock")
        inflate_cmd.extend(["--multi-pass", str(multi_pass)])
        inflate_cmd.extend(["--device", device])
        inflate_cmd.extend(["--upstream-dir", str(upstream)])

        inflate_env = {
            **os.environ,
            "PYTHONPATH": f"{src_dir}:{upstream}",
            "INFLATE_BRIGHTNESS_SHIFT": "1" if brightness_shift else "0",
            "INFLATE_CHROMA_SMOOTH": "1" if chroma_smooth else "0",
            "INFLATE_DEBLOCK": "1" if deblock else "0",
            "COMMA_CHALLENGE_ROOT": str(upstream),
            "TAC_MODELS_DIR": str(upstream / "models"),
        }
        _run(
            inflate_cmd,
            "inflate",
            env=inflate_env,
            verbose=verbose,
            timeout=1800,  # 30 min
        )
        stage_times["inflate"] = time.monotonic() - t0
    else:
        if verbose:
            click.echo("\n  Skipping inflation (--skip-inflate)", err=True)
        # If skipping inflate, copy any existing inflated dir into work tree
        existing_inflated = self_dir / "inflated"
        if existing_inflated.exists() and not any(inflated_dir.iterdir()):
            if verbose:
                click.echo(
                    f"  Copying existing {existing_inflated} -> {inflated_dir}",
                    err=True,
                )
            shutil.copytree(str(existing_inflated), str(inflated_dir), dirs_exist_ok=True)

    # ── Step 4: Score ────────────────────────────────────────────────
    if not skip_score:
        t0 = time.monotonic()

        # Verify inflated dir has files
        inflated_files = list(inflated_dir.glob("*"))
        if not inflated_files:
            raise click.ClickException(
                f"inflated dir is empty: {inflated_dir}  — inflate step may have failed silently"
            )
        if verbose:
            click.echo(f"  inflated/ contains {len(inflated_files)} items", err=True)

        # Verify archive.zip is in submission_dir
        if not (submission_dir / "archive.zip").exists():
            raise click.ClickException(
                f"archive.zip missing from {submission_dir} — evaluator needs it for rate"
            )

        report_path = work / "report.txt"
        score_cmd = [
            sys.executable,
            str(upstream / "evaluate.py"),
            "--submission-dir", str(submission_dir),
            "--uncompressed-dir", str(upstream / "videos"),
            "--video-names-file", str(video_names_file),
            "--device", device,
            "--report", str(report_path),
        ]

        # Run scorer from upstream cwd so relative imports work
        score_env = {
            **os.environ,
            "PYTHONPATH": str(upstream),
        }
        _run(
            score_cmd,
            "score",
            env=score_env,
            cwd=str(upstream),
            verbose=verbose,
            timeout=3600,  # 60 min — scoring on CPU is slow
        )
        stage_times["score"] = time.monotonic() - t0

        # ── Parse and display results ────────────────────────────────
        if report_path.exists():
            parsed = _parse_report_txt(report_path)
            click.echo(f"\n{'='*60}", err=True)
            click.echo("  EVALUATION COMPLETE", err=True)
            click.echo(f"{'='*60}", err=True)
            if "total_score" in parsed:
                click.echo(
                    f"  SCORE:   {parsed['total_score']:.4f}", err=True
                )
            if "segnet_distortion" in parsed:
                click.echo(
                    f"  SegNet:  {parsed['segnet_distortion']:.8f}", err=True
                )
            if "posenet_distortion" in parsed:
                click.echo(
                    f"  PoseNet: {parsed['posenet_distortion']:.8f}", err=True
                )
            if "rate" in parsed:
                click.echo(
                    f"  Rate:    {parsed['rate']:.8f}", err=True
                )
            click.echo(f"{'='*60}", err=True)

            # Also write structured JSON for downstream tooling
            json_report = work / "report.json"
            json_report.write_text(json.dumps(parsed, indent=2) + "\n")
            if verbose:
                click.echo(f"  JSON report: {json_report}", err=True)
                click.echo(f"  Text report: {report_path}", err=True)
        else:
            click.echo("  WARNING: report.txt not found after scoring", err=True)
    else:
        if verbose:
            click.echo("\n  Skipping scoring (--skip-score)", err=True)

    # ── Summary ──────────────────────────────────────────────────────
    total_elapsed = time.monotonic() - t_pipeline
    click.echo(f"\n  Pipeline timing:", err=True)
    for stage, secs in stage_times.items():
        click.echo(f"    [{stage}] {_fmt_elapsed(secs)}", err=True)
    click.echo(f"    [total]  {_fmt_elapsed(total_elapsed)}", err=True)
    click.echo(f"  Work dir:  {work}", err=True)


if __name__ == "__main__":
    evaluate()
