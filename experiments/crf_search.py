#!/usr/bin/env python3
"""Per-video CRF optimizer for the comma video compression challenge.

Binary-searches CRF values using the smoke pipeline (semantic MAE) as a fast
proxy, then optionally runs the full scorer on the top candidates.

Usage:
    # Coarse sweep (CRF 32-36, step 1)
    uv run python experiments/crf_search.py --crf-min 32 --crf-max 36 --crf-step 1

    # Fine sweep around a known optimum
    uv run python experiments/crf_search.py --crf-min 33.0 --crf-max 35.0 --crf-step 0.5

    # Run full scorer on top N candidates after proxy sweep
    uv run python experiments/crf_search.py --crf-min 32 --crf-max 36 --crf-step 1 --full-eval-top 2
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SUBMISSION_DIR = REPO_ROOT / "submissions" / "robust_current"
CONFIG_TEMPLATE = SUBMISSION_DIR / "config.env"
UPSTREAM_ROOT = REPO_ROOT / "workspace" / "upstream" / "comma_video_compression_challenge"


@dataclass
class CRFResult:
    crf: float
    archive_bytes: int | None
    smoke_mae_mean: float | None
    smoke_mae_max: float | None
    smoke_passed: bool
    encode_secs: float
    smoke_secs: float
    full_score: float | None = None
    error: str | None = None


def make_config(crf: float, dest: Path) -> None:
    """Write a config.env with the given CRF, keeping everything else from the template."""
    lines = CONFIG_TEMPLATE.read_text().splitlines()
    out_lines: list[str] = []
    for line in lines:
        if line.strip().startswith("SVT_AV1_CRF="):
            out_lines.append(f"SVT_AV1_CRF={crf}")
        else:
            out_lines.append(line)
    dest.write_text("\n".join(out_lines) + "\n")


def encode_at_crf(crf: float, work_dir: Path) -> tuple[Path, float]:
    """Encode using compress.sh with a temp config. Returns (archive_path, elapsed_secs)."""
    config_path = work_dir / "config.env"
    make_config(crf, config_path)

    env = os.environ.copy()
    env["COMMA_CHALLENGE_ROOT"] = str(UPSTREAM_ROOT)
    env["CONFIG_ENV_PATH"] = str(config_path)

    t0 = time.monotonic()
    subprocess.run(
        ["bash", str(SUBMISSION_DIR / "compress.sh")],
        cwd=REPO_ROOT,
        env=env,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    elapsed = time.monotonic() - t0

    archive = SUBMISSION_DIR / "archive.zip"
    return archive, elapsed


def run_smoke() -> dict:
    """Run the smoke pipeline and return the parsed JSON summary.

    Uses sync mode (default) so that the freshly-encoded archive.zip in
    submissions/robust_current/ is copied to the upstream dir before inflation.
    """
    env = os.environ.copy()
    env["COMMA_CHALLENGE_ROOT"] = str(UPSTREAM_ROOT)

    t0 = time.monotonic()
    cp = subprocess.run(
        [
            sys.executable, "-m", "comma_lab.cli",
            "smoke-submission", "robust_current",
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    elapsed = time.monotonic() - t0

    if cp.returncode != 0:
        raise RuntimeError(f"Smoke failed:\n{cp.stderr}")

    # The JSON is in stdout; stderr has progress output
    return json.loads(cp.stdout), elapsed


def run_full_eval(device: str = "cpu") -> dict:
    """Run full evaluation and return parsed JSON summary."""
    env = os.environ.copy()
    env["COMMA_CHALLENGE_ROOT"] = str(UPSTREAM_ROOT)

    cp = subprocess.run(
        [
            sys.executable, "-m", "comma_lab.cli",
            "eval-submission", "robust_current",
            "--device", device,
        ],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    if cp.returncode != 0:
        raise RuntimeError(f"Full eval failed:\n{cp.stderr}")
    return json.loads(cp.stdout)


def sweep(
    crf_min: float,
    crf_max: float,
    crf_step: float,
    full_eval_top: int = 0,
    device: str = "cpu",
) -> list[CRFResult]:
    """Run the CRF sweep and return results sorted by proxy MAE."""

    # Generate CRF values
    crfs: list[float] = []
    c = crf_min
    while c <= crf_max + 1e-9:
        crfs.append(round(c, 2))
        c += crf_step

    print(f"=== CRF Search: testing {len(crfs)} values: {crfs} ===\n")

    results: list[CRFResult] = []

    for i, crf in enumerate(crfs):
        print(f"--- [{i+1}/{len(crfs)}] CRF={crf} ---")

        with tempfile.TemporaryDirectory(prefix="crf_search_") as work_dir:
            try:
                # Step 1: Encode
                print(f"  Encoding at CRF {crf}...")
                archive_path, encode_secs = encode_at_crf(crf, Path(work_dir))
                archive_bytes = archive_path.stat().st_size
                print(f"  Encoded in {encode_secs:.1f}s, archive={archive_bytes:,} bytes")

                # Step 2: Run smoke (proxy eval)
                print(f"  Running smoke proxy...")
                smoke_data, smoke_secs = run_smoke()
                video_result = smoke_data["results"][0]  # single video
                mae_mean = video_result["semantic_mae_mean"]
                mae_max = video_result["semantic_mae_max"]
                passed = video_result["semantic_check_passed"]
                print(f"  Smoke: MAE mean={mae_mean:.4f}, max={mae_max:.4f}, passed={passed} ({smoke_secs:.1f}s)")

                results.append(CRFResult(
                    crf=crf,
                    archive_bytes=archive_bytes,
                    smoke_mae_mean=mae_mean,
                    smoke_mae_max=mae_max,
                    smoke_passed=passed,
                    encode_secs=encode_secs,
                    smoke_secs=smoke_secs,
                ))

            except Exception as e:
                print(f"  ERROR: {e}")
                results.append(CRFResult(
                    crf=crf,
                    archive_bytes=None,
                    smoke_mae_mean=None,
                    smoke_mae_max=None,
                    smoke_passed=False,
                    encode_secs=0,
                    smoke_secs=0,
                    error=str(e),
                ))

    # Sort by proxy MAE (lower is better quality, but we want the sweet spot)
    valid = [r for r in results if r.smoke_mae_mean is not None]
    valid.sort(key=lambda r: r.smoke_mae_mean)

    # Print summary table
    print("\n=== PROXY RESULTS (sorted by MAE mean) ===")
    print(f"{'CRF':>6} {'Archive':>12} {'MAE mean':>10} {'MAE max':>10} {'Pass':>6} {'Encode':>8} {'Smoke':>8}")
    print("-" * 70)
    for r in valid:
        print(
            f"{r.crf:>6.1f} "
            f"{r.archive_bytes:>12,} "
            f"{r.smoke_mae_mean:>10.4f} "
            f"{r.smoke_mae_max:>10.4f} "
            f"{'Y' if r.smoke_passed else 'N':>6} "
            f"{r.encode_secs:>7.1f}s "
            f"{r.smoke_secs:>7.1f}s"
        )

    if valid:
        best = valid[0]
        print(f"\nBest proxy CRF: {best.crf} (MAE mean={best.smoke_mae_mean:.4f}, archive={best.archive_bytes:,} bytes)")

    # Optionally run full scorer on top N
    if full_eval_top > 0 and valid:
        top_n = valid[:full_eval_top]
        print(f"\n=== Running full scorer on top {len(top_n)} candidates ===")
        for r in top_n:
            print(f"\n--- Full eval CRF={r.crf} ---")
            try:
                # Re-encode at this CRF (archive may have been overwritten)
                with tempfile.TemporaryDirectory(prefix="crf_full_") as wd:
                    encode_at_crf(r.crf, Path(wd))
                eval_data = run_full_eval(device)
                r.full_score = eval_data["current_workflow_score"]
                print(f"  Full score: {r.full_score}")
            except Exception as e:
                print(f"  ERROR: {e}")
                r.error = str(e)

        scored = [r for r in top_n if r.full_score is not None]
        if scored:
            scored.sort(key=lambda r: r.full_score)
            print(f"\n=== FULL SCORER RESULTS ===")
            for r in scored:
                print(f"  CRF {r.crf}: score={r.full_score:.4f}")
            print(f"\nBest scored CRF: {scored[0].crf} (score={scored[0].full_score:.4f})")

    # Save results to JSON
    out_path = REPO_ROOT / "experiments" / "crf_search_results.json"
    out_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "crf_range": {"min": crf_min, "max": crf_max, "step": crf_step},
        "results": [asdict(r) for r in results],
        "best_proxy_crf": valid[0].crf if valid else None,
    }
    out_path.write_text(json.dumps(out_data, indent=2) + "\n")
    print(f"\nResults saved to: {out_path}")

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Per-video CRF optimizer")
    parser.add_argument("--crf-min", type=float, default=32, help="Minimum CRF to test")
    parser.add_argument("--crf-max", type=float, default=36, help="Maximum CRF to test")
    parser.add_argument("--crf-step", type=float, default=1, help="CRF step size")
    parser.add_argument("--full-eval-top", type=int, default=0, help="Run full scorer on top N proxy results")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"])
    args = parser.parse_args()

    sweep(
        crf_min=args.crf_min,
        crf_max=args.crf_max,
        crf_step=args.crf_step,
        full_eval_top=args.full_eval_top,
        device=args.device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
