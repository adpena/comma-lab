#!/usr/bin/env python
"""One-command evaluation pipeline: compress -> inflate -> score.

Instead of running three separate scripts and juggling env vars, run:

    python eval.py --upstream-dir workspace/upstream/comma_video_compression_challenge

This handles the full pipeline with sane defaults that match config.env.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import click

from config_env import load_config_env


def _find_upstream(explicit: str | None) -> Path:
    """Resolve upstream challenge root with multiple fallback strategies."""
    if explicit:
        p = Path(explicit)
        if p.exists() and (p / "evaluate.sh").exists():
            return p
        raise click.ClickException(
            f"Upstream dir {explicit} does not contain evaluate.sh"
        )
    # Default: relative to this script
    default = Path(__file__).resolve().parent.parent.parent / "workspace" / "upstream" / "comma_video_compression_challenge"
    if default.exists() and (default / "evaluate.sh").exists():
        return default
    raise click.ClickException(
        "Could not find upstream challenge root. "
        "Set --upstream-dir or COMMA_CHALLENGE_ROOT."
    )


def _run(cmd: list[str], label: str, verbose: bool = True) -> subprocess.CompletedProcess:
    """Run a subprocess with timing and error handling."""
    if verbose:
        click.echo(f"\n{'='*60}", err=True)
        click.echo(f"  {label}", err=True)
        click.echo(f"{'='*60}", err=True)
        click.echo(f"  $ {' '.join(cmd)}", err=True)
    t0 = time.monotonic()
    result = subprocess.run(cmd, capture_output=not verbose)
    elapsed = time.monotonic() - t0
    if verbose:
        click.echo(f"  [{label}] completed in {elapsed:.1f}s (exit {result.returncode})", err=True)
    if result.returncode != 0:
        if not verbose and result.stderr:
            click.echo(result.stderr.decode(errors="replace"), err=True)
        raise click.ClickException(f"{label} failed with exit code {result.returncode}")
    return result


@click.command()
@click.option("--config", "config_path", type=click.Path(),
              default="config.env", help="Config env file to load before running.")
@click.option("--upstream-dir", envvar="COMMA_CHALLENGE_ROOT",
              default=None, help="Upstream challenge root directory.")
@click.option("--skip-compress", is_flag=True,
              help="Skip compression, use existing archive.zip.")
@click.option("--skip-inflate", is_flag=True,
              help="Skip inflation, use existing inflated/ directory.")
@click.option("--skip-score", is_flag=True,
              help="Only compress + inflate, do not run scorer.")
@click.option("--brightness-shift/--no-brightness-shift", envvar="INFLATE_BRIGHTNESS_SHIFT",
              default=True, help="Shift luminance toward midpoint.")
@click.option("--chroma-smooth/--no-chroma-smooth", envvar="INFLATE_CHROMA_SMOOTH",
              default=True, help="Smooth chroma channels.")
@click.option("--deblock/--no-deblock", envvar="INFLATE_DEBLOCK",
              default=False, help="Apply NLM deblocking.")
@click.option("--multi-pass", type=int, envvar="INFLATE_MULTI_PASS",
              default=1, help="Post-filter passes (2=double pass).")
@click.option("--device", default="cpu",
              help="Inference device: cpu, cuda, or mps.")
@click.option("--verbose/--quiet", default=True,
              help="Print progress to stderr.")
def evaluate(config_path, upstream_dir, skip_compress, skip_inflate, skip_score,
             brightness_shift, chroma_smooth, deblock, multi_pass, device, verbose):
    """Full pipeline: compress -> inflate -> score. One command.

    \b
    Examples:
      # Full eval with defaults from config.env:
      python eval.py --upstream-dir workspace/upstream/comma_video_compression_challenge

      # Re-score without re-compressing (iterate on inflate settings):
      python eval.py --skip-compress --multi-pass 2 --upstream-dir ...

      # Just compress + inflate, skip scoring:
      python eval.py --skip-score --upstream-dir ...
    """
    t_start = time.monotonic()

    self_dir = Path(__file__).resolve().parent

    # Load config.env BEFORE Click processes env vars -- this populates
    # os.environ so that inflate.sh and compress.sh pick up settings.
    cfg_path = Path(config_path)
    if not cfg_path.is_absolute():
        cfg_path = self_dir / cfg_path
    config = load_config_env(cfg_path, into_environ=True)
    if verbose and config:
        click.echo(f"Loaded {len(config)} settings from {cfg_path}", err=True)

    # Propagate Click flags into env so inflate.sh / inflate_postfilter.py
    # pick them up via their own env var reads.
    os.environ["INFLATE_BRIGHTNESS_SHIFT"] = "1" if brightness_shift else "0"
    os.environ["INFLATE_CHROMA_SMOOTH"] = "1" if chroma_smooth else "0"
    os.environ["INFLATE_DEBLOCK"] = "1" if deblock else "0"
    os.environ["INFLATE_MULTI_PASS"] = str(multi_pass)

    upstream = _find_upstream(upstream_dir)
    os.environ["COMMA_CHALLENGE_ROOT"] = str(upstream)
    if verbose:
        click.echo(f"Upstream: {upstream}", err=True)

    video_names_file = upstream / "public_test_video_names.txt"
    if not video_names_file.exists():
        raise click.ClickException(f"Video names file not found: {video_names_file}")

    archive_zip = self_dir / "archive.zip"
    archive_dir = self_dir / "archive"
    inflated_dir = self_dir / "inflated"

    # ── Step 1: Compress ──
    if not skip_compress:
        compress_sh = self_dir / "compress.sh"
        if not compress_sh.exists():
            raise click.ClickException(f"compress.sh not found at {compress_sh}")
        _run(["bash", str(compress_sh)], "Compress", verbose=verbose)
    else:
        if verbose:
            click.echo("Skipping compression (--skip-compress)", err=True)
        if not archive_zip.exists():
            raise click.ClickException(
                f"--skip-compress requires existing {archive_zip}"
            )

    # ── Step 2: Unpack archive.zip (scorer expects a directory) ──
    if not skip_inflate:
        if archive_zip.exists():
            archive_dir.mkdir(parents=True, exist_ok=True)
            _run(
                ["unzip", "-o", str(archive_zip), "-d", str(archive_dir)],
                "Unpack archive.zip",
                verbose=verbose,
            )

    # ── Step 3: Inflate ──
    if not skip_inflate:
        inflate_cmd = [
            sys.executable, str(self_dir / "inflate_postfilter.py"),
            str(archive_dir), str(inflated_dir), str(video_names_file),
        ]
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

        _run(inflate_cmd, "Inflate (post-filter)", verbose=verbose)
    else:
        if verbose:
            click.echo("Skipping inflation (--skip-inflate)", err=True)
        if not inflated_dir.exists():
            raise click.ClickException(
                f"--skip-inflate requires existing {inflated_dir}"
            )

    # ── Step 4: Score ──
    if not skip_score:
        evaluate_py = upstream / "evaluate.py"
        if not evaluate_py.exists():
            raise click.ClickException(f"evaluate.py not found at {evaluate_py}")

        score_cmd = [
            sys.executable, str(evaluate_py),
            str(archive_zip),
            str(inflated_dir),
            str(video_names_file),
        ]
        result = _run(score_cmd, "Score (evaluate.py)", verbose=verbose)

        # Print final score prominently
        if verbose:
            click.echo(f"\n{'='*60}", err=True)
            click.echo("  EVALUATION COMPLETE", err=True)
            click.echo(f"{'='*60}", err=True)
    else:
        if verbose:
            click.echo("Skipping scoring (--skip-score)", err=True)

    t_total = time.monotonic() - t_start
    if verbose:
        click.echo(f"\nTotal pipeline time: {t_total:.1f}s", err=True)


if __name__ == "__main__":
    evaluate()
