#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Profile the exact contest evaluation pipeline stage by stage.

Replicates the EXACT pipeline from upstream evaluate.sh and eval.yml:
    1. unzip -o archive.zip -d archive/
    2. bash inflate.sh archive/ inflated/ video_names.txt
    3. verify .raw files exist
    4. python evaluate.py --submission-dir ... --device {cuda|mps|cpu}

Total contest budget: 30 minutes (1800 seconds).

Produces a clean time budget table showing how much wall-clock time
each stage consumes, and how much remains for TTO at inflate time.

Usage:
    # Local (MPS):
    python scripts/profile_contest_pipeline.py \
        --submission-dir submissions/robust_current \
        --device mps

    # On Modal T4:
    python scripts/profile_contest_pipeline.py \
        --submission-dir submissions/robust_current \
        --device cuda

    # Skip scoring (just profile unzip + inflate):
    python scripts/profile_contest_pipeline.py \
        --submission-dir submissions/robust_current \
        --skip-scoring

    # Use specific upstream root:
    python scripts/profile_contest_pipeline.py \
        --submission-dir submissions/robust_current \
        --upstream upstream/
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---- Constants ---- #

CONTEST_TIMEOUT_SECONDS: float = 1800.0  # 30 minutes (eval.yml timeout-minutes: 30)
FRAME_BYTES: int = 1164 * 874 * 3  # single frame: 3,052,008 bytes
EXPECTED_FRAMES: int = 1200
EXPECTED_RAW_BYTES: int = FRAME_BYTES * EXPECTED_FRAMES  # 3,662,409,600


@dataclass
class StageResult:
    """Timing result for a single pipeline stage."""

    name: str
    elapsed_seconds: float
    success: bool
    detail: str = ""


@dataclass
class PipelineProfile:
    """Complete pipeline profiling report."""

    stages: list[StageResult] = field(default_factory=list)
    device: str = "cpu"
    submission_dir: str = ""
    archive_size_bytes: int = 0
    raw_size_bytes: int = 0
    total_elapsed: float = 0.0

    @property
    def inflate_elapsed(self) -> float:
        """Wall-clock time consumed by the inflate stage."""
        for s in self.stages:
            if s.name == "inflate":
                return s.elapsed_seconds
        return 0.0

    @property
    def scoring_elapsed(self) -> float:
        """Wall-clock time consumed by scoring (evaluate.py)."""
        for s in self.stages:
            if s.name == "scoring":
                return s.elapsed_seconds
        return 0.0

    @property
    def overhead_elapsed(self) -> float:
        """Wall-clock time consumed by everything except inflate and scoring."""
        return self.total_elapsed - self.inflate_elapsed - self.scoring_elapsed

    @property
    def budget_remaining(self) -> float:
        """Seconds remaining within the 30-min contest budget after all stages."""
        return max(0.0, CONTEST_TIMEOUT_SECONDS - self.total_elapsed)

    @property
    def tto_budget_estimate(self) -> float:
        """Estimated seconds available for TTO at inflate time.

        This is: contest_timeout - scoring_time - overhead - safety_margin.
        The safety margin (60s) accounts for CI variance and model loading.
        """
        safety_margin = 60.0
        non_inflate = self.scoring_elapsed + self.overhead_elapsed + safety_margin
        return max(0.0, CONTEST_TIMEOUT_SECONDS - non_inflate)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the profile to a JSON-friendly dict."""
        return {
            "device": self.device,
            "submission_dir": self.submission_dir,
            "archive_size_bytes": self.archive_size_bytes,
            "raw_size_bytes": self.raw_size_bytes,
            "contest_timeout_seconds": CONTEST_TIMEOUT_SECONDS,
            "total_elapsed_seconds": round(self.total_elapsed, 3),
            "tto_budget_estimate_seconds": round(self.tto_budget_estimate, 3),
            "stages": [
                {
                    "name": s.name,
                    "elapsed_seconds": round(s.elapsed_seconds, 3),
                    "success": s.success,
                    "detail": s.detail,
                }
                for s in self.stages
            ],
        }


# ---- Stage runners ---- #


def _run_stage(name: str, fn: Any) -> StageResult:
    """Run a callable and time it with monotonic clock."""
    t0 = time.monotonic()
    try:
        detail = fn()
        elapsed = time.monotonic() - t0
        return StageResult(name=name, elapsed_seconds=elapsed, success=True,
                           detail=str(detail or ""))
    except Exception as e:
        elapsed = time.monotonic() - t0
        return StageResult(name=name, elapsed_seconds=elapsed, success=False,
                           detail=str(e))


def stage_unzip(archive_zip: Path, archive_dir: Path) -> str:
    """Stage 1: unzip -o archive.zip -d archive/"""
    if archive_dir.exists():
        shutil.rmtree(archive_dir)
    archive_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["unzip", "-o", str(archive_zip), "-d", str(archive_dir)],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"unzip failed (rc={result.returncode}): {result.stderr}")
    # Count extracted files
    n_files = sum(1 for _ in archive_dir.rglob("*") if _.is_file())
    total_size = sum(f.stat().st_size for f in archive_dir.rglob("*") if f.is_file())
    return f"{n_files} files, {total_size:,} bytes extracted"


def stage_inflate(
    inflate_sh: Path,
    archive_dir: Path,
    inflated_dir: Path,
    video_names_file: Path,
) -> str:
    """Stage 2: bash inflate.sh archive/ inflated/ video_names.txt"""
    if inflated_dir.exists():
        shutil.rmtree(inflated_dir)
    inflated_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    result = subprocess.run(
        ["bash", str(inflate_sh), str(archive_dir), str(inflated_dir),
         str(video_names_file)],
        capture_output=True, text=True, timeout=1500,
        env=env, cwd=str(inflate_sh.parent),
    )
    if result.returncode != 0:
        stderr_tail = result.stderr[-2000:] if result.stderr else "(no stderr)"
        raise RuntimeError(f"inflate.sh failed (rc={result.returncode}): {stderr_tail}")

    # Measure output
    raw_files = list(inflated_dir.rglob("*.raw"))
    total_bytes = sum(f.stat().st_size for f in raw_files)
    return f"{len(raw_files)} raw file(s), {total_bytes:,} bytes"


def stage_verify(inflated_dir: Path, video_names_file: Path) -> str:
    """Stage 3: verify .raw files exist and have correct size."""
    video_names = [
        line.strip() for line in video_names_file.read_text().splitlines()
        if line.strip()
    ]
    missing = []
    size_errors = []
    for name in video_names:
        stem = name.rsplit(".", 1)[0]
        raw_path = inflated_dir / f"{stem}.raw"
        if not raw_path.exists():
            missing.append(str(raw_path))
        elif raw_path.stat().st_size != EXPECTED_RAW_BYTES:
            actual = raw_path.stat().st_size
            size_errors.append(
                f"{raw_path}: {actual:,} bytes (expected {EXPECTED_RAW_BYTES:,})"
            )

    if missing:
        raise RuntimeError(f"Missing raw files: {missing}")
    if size_errors:
        return f"SIZE WARNINGS: {'; '.join(size_errors)}"
    return f"All {len(video_names)} raw files verified ({EXPECTED_RAW_BYTES:,} bytes each)"


def stage_scoring(
    submission_dir: Path,
    upstream_root: Path,
    video_names_file: Path,
    device: str,
) -> str:
    """Stage 4: python evaluate.py --submission-dir ... --device {device}"""
    evaluate_py = upstream_root / "evaluate.py"
    if not evaluate_py.exists():
        raise FileNotFoundError(f"evaluate.py not found at {evaluate_py}")

    videos_dir = upstream_root / "videos"
    report_path = submission_dir / "profile_report.txt"

    cmd = [
        sys.executable, str(evaluate_py),
        "--submission-dir", str(submission_dir),
        "--uncompressed-dir", str(videos_dir),
        "--report", str(report_path),
        "--video-names-file", str(video_names_file),
        "--device", device,
    ]

    env = os.environ.copy()
    env["PYTHONPATH"] = str(upstream_root) + ":" + env.get("PYTHONPATH", "")

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=1200, env=env,
        cwd=str(upstream_root),
    )
    if result.returncode != 0:
        stderr_tail = result.stderr[-2000:] if result.stderr else "(no stderr)"
        raise RuntimeError(f"evaluate.py failed (rc={result.returncode}): {stderr_tail}")

    # Extract score from report
    if report_path.exists():
        report_text = report_path.read_text()
        for line in report_text.splitlines():
            if "Final score" in line:
                return line.strip()
    return result.stdout[-500:] if result.stdout else "(no output)"


# ---- Main profiler ---- #


def profile_pipeline(
    submission_dir: Path,
    upstream_root: Path,
    video_names_file: Path,
    device: str,
    skip_scoring: bool = False,
) -> PipelineProfile:
    """Run all contest pipeline stages with timing.

    Args:
        submission_dir: path to submission (contains archive.zip, inflate.sh)
        upstream_root: path to upstream repo (contains evaluate.py, videos/)
        video_names_file: path to video_names.txt
        device: 'cuda', 'mps', or 'cpu'
        skip_scoring: if True, skip the evaluate.py stage

    Returns:
        PipelineProfile with timing data for all stages.
    """
    profile = PipelineProfile(device=device, submission_dir=str(submission_dir))

    archive_zip = submission_dir / "archive.zip"
    archive_dir = submission_dir / "archive"
    inflated_dir = submission_dir / "inflated"
    inflate_sh = submission_dir / "inflate.sh"

    if not archive_zip.exists():
        raise FileNotFoundError(f"archive.zip not found: {archive_zip}")
    if not inflate_sh.exists():
        raise FileNotFoundError(f"inflate.sh not found: {inflate_sh}")

    profile.archive_size_bytes = archive_zip.stat().st_size

    t_pipeline_start = time.monotonic()

    # Stage 1: Unzip
    print("=" * 72, file=sys.stderr)
    print("Stage 1/4: UNZIP", file=sys.stderr)
    print("=" * 72, file=sys.stderr)
    result = _run_stage(
        "unzip",
        lambda: stage_unzip(archive_zip, archive_dir),
    )
    profile.stages.append(result)
    _print_stage_result(result)

    # Stage 2: Inflate
    print("\n" + "=" * 72, file=sys.stderr)
    print("Stage 2/4: INFLATE", file=sys.stderr)
    print("=" * 72, file=sys.stderr)
    result = _run_stage(
        "inflate",
        lambda: stage_inflate(inflate_sh, archive_dir, inflated_dir, video_names_file),
    )
    profile.stages.append(result)
    _print_stage_result(result)

    # Measure raw output
    raw_files = list(inflated_dir.rglob("*.raw"))
    if raw_files:
        profile.raw_size_bytes = sum(f.stat().st_size for f in raw_files)

    # Stage 3: Verify
    print("\n" + "=" * 72, file=sys.stderr)
    print("Stage 3/4: VERIFY", file=sys.stderr)
    print("=" * 72, file=sys.stderr)
    result = _run_stage(
        "verify",
        lambda: stage_verify(inflated_dir, video_names_file),
    )
    profile.stages.append(result)
    _print_stage_result(result)

    # Stage 4: Scoring
    if skip_scoring:
        print("\n" + "=" * 72, file=sys.stderr)
        print("Stage 4/4: SCORING (SKIPPED)", file=sys.stderr)
        print("=" * 72, file=sys.stderr)
        result = StageResult(
            name="scoring", elapsed_seconds=0.0, success=True,
            detail="Skipped (--skip-scoring)",
        )
        profile.stages.append(result)
    else:
        print("\n" + "=" * 72, file=sys.stderr)
        print("Stage 4/4: SCORING", file=sys.stderr)
        print("=" * 72, file=sys.stderr)
        result = _run_stage(
            "scoring",
            lambda: stage_scoring(submission_dir, upstream_root, video_names_file, device),
        )
        profile.stages.append(result)
        _print_stage_result(result)

    profile.total_elapsed = time.monotonic() - t_pipeline_start
    return profile


def _print_stage_result(result: StageResult) -> None:
    """Print a single stage result to stderr."""
    status = "OK" if result.success else "FAILED"
    print(f"  [{status}] {result.elapsed_seconds:.2f}s -- {result.detail}",
          file=sys.stderr)


def print_budget_table(profile: PipelineProfile) -> None:
    """Print a clean time budget table to stdout."""
    print()
    print("=" * 72)
    print("CONTEST PIPELINE TIME BUDGET")
    print("=" * 72)
    print(f"  Device: {profile.device}")
    print(f"  Submission: {profile.submission_dir}")
    print(f"  Archive size: {profile.archive_size_bytes:,} bytes")
    print(f"  Raw output: {profile.raw_size_bytes:,} bytes")
    print()
    print(f"  {'Stage':<20s}  {'Time (s)':>10s}  {'% Budget':>10s}  {'Status':<8s}")
    print(f"  {'-' * 20}  {'-' * 10}  {'-' * 10}  {'-' * 8}")

    for s in profile.stages:
        pct = 100.0 * s.elapsed_seconds / CONTEST_TIMEOUT_SECONDS
        status = "OK" if s.success else "FAILED"
        print(f"  {s.name:<20s}  {s.elapsed_seconds:>10.2f}  {pct:>9.1f}%  {status:<8s}")

    print(f"  {'-' * 20}  {'-' * 10}  {'-' * 10}")
    pct_total = 100.0 * profile.total_elapsed / CONTEST_TIMEOUT_SECONDS
    print(f"  {'TOTAL':<20s}  {profile.total_elapsed:>10.2f}  {pct_total:>9.1f}%")
    print()
    print(f"  Contest budget:        {CONTEST_TIMEOUT_SECONDS:>8.0f}s  (30 min)")
    print(f"  Pipeline consumed:     {profile.total_elapsed:>8.2f}s")
    print(f"  Budget remaining:      {profile.budget_remaining:>8.2f}s")
    print()
    print(f"  --- TTO Budget Estimate ---")
    print(f"  Non-inflate overhead:  {profile.overhead_elapsed:>8.2f}s  (unzip + verify)")
    print(f"  Scoring time:          {profile.scoring_elapsed:>8.2f}s  (evaluate.py)")
    print(f"  Safety margin:         {60.0:>8.0f}s  (CI variance)")
    print(f"  Available for inflate: {profile.tto_budget_estimate:>8.2f}s")
    print(f"  Current inflate time:  {profile.inflate_elapsed:>8.2f}s")
    print(f"  TTO headroom:          {profile.tto_budget_estimate - profile.inflate_elapsed:>8.2f}s")
    print("=" * 72)


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Profile the exact contest evaluation pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Replicates the pipeline from upstream evaluate.sh:\n"
            "  1. unzip archive.zip\n"
            "  2. bash inflate.sh\n"
            "  3. verify .raw files\n"
            "  4. python evaluate.py\n\n"
            "Contest budget: 30 minutes total (eval.yml timeout-minutes: 30)"
        ),
    )
    parser.add_argument(
        "--submission-dir", type=Path,
        default=Path("submissions/robust_current"),
        help="Path to submission directory (contains archive.zip + inflate.sh)",
    )
    parser.add_argument(
        "--upstream", type=Path, default=Path("upstream"),
        help="Path to upstream repo (contains evaluate.py, videos/)",
    )
    parser.add_argument(
        "--video-names-file", type=Path, default=None,
        help="Path to video names file (default: upstream/public_test_video_names.txt)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        choices=["cuda", "mps", "cpu"],
        help="Device for scoring (default: auto-detect)",
    )
    parser.add_argument(
        "--skip-scoring", action="store_true",
        help="Skip the evaluate.py scoring stage (useful for quick inflate profiling)",
    )
    parser.add_argument(
        "--output-json", type=Path, default=None,
        help="Write JSON profile report to this path",
    )
    args = parser.parse_args()

    # Resolve paths
    submission_dir = args.submission_dir.resolve()
    upstream_root = args.upstream.resolve()

    if args.video_names_file is None:
        video_names_file = upstream_root / "public_test_video_names.txt"
    else:
        video_names_file = args.video_names_file.resolve()

    if not video_names_file.exists():
        print(f"ERROR: video names file not found: {video_names_file}", file=sys.stderr)
        sys.exit(1)

    # Auto-detect device
    if args.device is None:
        try:
            import torch
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        except ImportError:
            device = "cpu"
    else:
        device = args.device

    print(f"Profiling contest pipeline on device={device}", file=sys.stderr)
    print(f"  submission_dir: {submission_dir}", file=sys.stderr)
    print(f"  upstream_root:  {upstream_root}", file=sys.stderr)
    print(f"  video_names:    {video_names_file}", file=sys.stderr)
    print(file=sys.stderr)

    profile = profile_pipeline(
        submission_dir=submission_dir,
        upstream_root=upstream_root,
        video_names_file=video_names_file,
        device=device,
        skip_scoring=args.skip_scoring,
    )

    print_budget_table(profile)

    # Write JSON report
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(profile.to_dict(), f, indent=2)
        print(f"\nJSON report: {args.output_json}")


if __name__ == "__main__":
    main()
