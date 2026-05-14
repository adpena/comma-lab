# SPDX-License-Identifier: MIT
"""CRF optimization sweep for the comma video compression challenge.

Core logic extracted from experiments/crf_search.py for use via ``tac crf-search``.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from tac.versioned_output import versioned_write

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_SUBMISSION_DIR = _PROJECT_ROOT / "submissions" / "robust_current"
_CONFIG_TEMPLATE = _SUBMISSION_DIR / "config.env"
_UPSTREAM_ROOT = _PROJECT_ROOT / "workspace" / "upstream" / "comma_video_compression_challenge"


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
    lines = _CONFIG_TEMPLATE.read_text().splitlines()
    out_lines: list[str] = []
    for line in lines:
        if line.strip().startswith("SVT_AV1_CRF="):
            out_lines.append(f"SVT_AV1_CRF={crf}")
        else:
            out_lines.append(line)
    dest.write_text("\n".join(out_lines) + "\n")


def encode_at_crf(crf: float, work_dir: Path) -> tuple[Path, float]:
    """Encode using compress.sh with a temp config.  Returns (archive_path, elapsed_secs)."""
    config_path = work_dir / "config.env"
    make_config(crf, config_path)

    env = os.environ.copy()
    env["COMMA_CHALLENGE_ROOT"] = str(_UPSTREAM_ROOT)
    env["CONFIG_ENV_PATH"] = str(config_path)

    t0 = time.monotonic()
    subprocess.run(
        ["bash", str(_SUBMISSION_DIR / "compress.sh")],
        cwd=_PROJECT_ROOT,
        env=env,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    elapsed = time.monotonic() - t0

    archive = _SUBMISSION_DIR / "archive.zip"
    return archive, elapsed


def run_smoke() -> tuple[dict, float]:
    """Run the smoke pipeline and return the parsed JSON summary."""
    env = os.environ.copy()
    env["COMMA_CHALLENGE_ROOT"] = str(_UPSTREAM_ROOT)

    t0 = time.monotonic()
    cp = subprocess.run(
        [sys.executable, "-m", "comma_lab.cli", "smoke-submission", "robust_current"],
        cwd=_PROJECT_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )
    elapsed = time.monotonic() - t0

    if cp.returncode != 0:
        raise RuntimeError(f"Smoke failed:\n{cp.stderr}")
    return json.loads(cp.stdout), elapsed


def run_full_eval(device: str = "cpu") -> dict:
    """Run full evaluation and return parsed JSON summary."""
    env = os.environ.copy()
    env["COMMA_CHALLENGE_ROOT"] = str(_UPSTREAM_ROOT)

    cp = subprocess.run(
        [sys.executable, "-m", "comma_lab.cli", "eval-submission", "robust_current", "--device", device],
        cwd=_PROJECT_ROOT,
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
                print(f"  Encoding at CRF {crf}...")
                archive_path, encode_secs = encode_at_crf(crf, Path(work_dir))
                archive_bytes = archive_path.stat().st_size
                print(f"  Encoded in {encode_secs:.1f}s, archive={archive_bytes:,} bytes")

                print("  Running smoke proxy...")
                smoke_data, smoke_secs = run_smoke()
                video_result = smoke_data["results"][0]
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
                    crf=crf, archive_bytes=None, smoke_mae_mean=None,
                    smoke_mae_max=None, smoke_passed=False,
                    encode_secs=0, smoke_secs=0, error=str(e),
                ))

    valid = [r for r in results if r.smoke_mae_mean is not None]
    valid.sort(key=lambda r: r.smoke_mae_mean)

    print("\n=== PROXY RESULTS (sorted by MAE mean) ===")
    print(f"{'CRF':>6} {'Archive':>12} {'MAE mean':>10} {'MAE max':>10} {'Pass':>6} {'Encode':>8} {'Smoke':>8}")
    print("-" * 70)
    for r in valid:
        print(
            f"{r.crf:>6.1f} {r.archive_bytes:>12,} {r.smoke_mae_mean:>10.4f} "
            f"{r.smoke_mae_max:>10.4f} {'Y' if r.smoke_passed else 'N':>6} "
            f"{r.encode_secs:>7.1f}s {r.smoke_secs:>7.1f}s"
        )

    if valid:
        best = valid[0]
        print(f"\nBest proxy CRF: {best.crf} (MAE mean={best.smoke_mae_mean:.4f}, archive={best.archive_bytes:,} bytes)")

    if full_eval_top > 0 and valid:
        top_n = valid[:full_eval_top]
        print(f"\n=== Running full scorer on top {len(top_n)} candidates ===")
        for r in top_n:
            print(f"\n--- Full eval CRF={r.crf} ---")
            try:
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
            print("\n=== FULL SCORER RESULTS ===")
            for r in scored:
                print(f"  CRF {r.crf}: score={r.full_score:.4f}")
            print(f"\nBest scored CRF: {scored[0].crf} (score={scored[0].full_score:.4f})")

    out_path = _PROJECT_ROOT / "experiments" / "crf_search_results.json"
    out_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "crf_range": {"min": crf_min, "max": crf_max, "step": crf_step},
        "results": [asdict(r) for r in results],
        "best_proxy_crf": valid[0].crf if valid else None,
    }
    crf_tag = f"crf{crf_min}-{crf_max}"
    versioned_path = versioned_write(out_path, json.dumps(out_data, indent=2) + "\n", config_tag=crf_tag)
    print(f"\nResults saved to: {versioned_path}")

    return results


def find_optimal_crf(
    checkpoint_path: str | Path | None = None,
    crf_range: list[float] | None = None,
    eval_pairs: int = 50,
    device: str = "cpu",
) -> dict:
    """Find the CRF value that minimizes the composite proxy score.

    This is the single highest-ROI CPU experiment: zero retraining, just
    sweep the encoder quality parameter and pick the best.

    The composite score is ``100 * seg_distortion + sqrt(10 * pose_distortion)``.
    Lower CRF = better quality but bigger archive (higher rate penalty).
    Higher CRF = worse quality but smaller archive (lower rate penalty).
    The optimal point balances rate vs distortion.

    Args:
        checkpoint_path: path to postfilter int8 checkpoint. If None, uses
            the current submission checkpoint.
        crf_range: list of CRF values to test. Default [31, 32, 33, 34, 35, 36].
        eval_pairs: number of frame pairs for proxy evaluation (default 50).
        device: device for proxy eval (default "cpu").

    Returns:
        dict with keys:
            best_crf: the optimal CRF value
            best_score: the composite score at optimal CRF
            results: list of per-CRF result dicts
    """
    if crf_range is None:
        crf_range = [31, 32, 33, 34, 35, 36]

    if checkpoint_path is None:
        checkpoint_path = _SUBMISSION_DIR / "postfilter_int8.pt"
    checkpoint_path = Path(checkpoint_path)

    print(f"=== find_optimal_crf: testing CRFs {crf_range} ===")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Eval pairs: {eval_pairs}")

    scored_results: list[dict] = []

    for crf in crf_range:
        print(f"\n--- CRF={crf} ---")
        try:
            with tempfile.TemporaryDirectory(prefix="crf_opt_") as work_dir:
                # Encode at this CRF
                archive_path, encode_secs = encode_at_crf(crf, Path(work_dir))
                archive_bytes = archive_path.stat().st_size

                # Run smoke proxy to get MAE scores
                smoke_data, smoke_secs = run_smoke()
                video_result = smoke_data["results"][0]
                mae_mean = video_result["semantic_mae_mean"]
                mae_max = video_result["semantic_mae_max"]
                passed = video_result["semantic_check_passed"]

                # Compute rate component: archive_bytes / (total_frames * target_pixels * 3)
                # The rate term in the scoring formula is roughly proportional to archive size.
                # Use archive bytes as the rate proxy.
                rate_proxy = archive_bytes / 1_000_000  # MB

                # Composite = distortion_proxy + rate_proxy_scaled
                # The actual score formula: score = 100*seg + sqrt(10*pose) + rate
                # We approximate distortion via MAE mean (which correlates with
                # both PoseNet and SegNet distortion from proxy evaluation).
                # Rate is the archive size normalized.
                composite = mae_mean + 0.001 * rate_proxy  # MAE + small rate penalty

                result = {
                    "crf": crf,
                    "archive_bytes": archive_bytes,
                    "mae_mean": mae_mean,
                    "mae_max": mae_max,
                    "passed": passed,
                    "encode_secs": encode_secs,
                    "smoke_secs": smoke_secs,
                    "composite": composite,
                }
                scored_results.append(result)
                print(f"  Archive={archive_bytes:,}B  MAE={mae_mean:.4f}  "
                      f"Composite={composite:.6f}  ({encode_secs:.1f}s encode, {smoke_secs:.1f}s eval)")

        except Exception as e:
            print(f"  ERROR: {e}")
            scored_results.append({
                "crf": crf, "error": str(e), "composite": float("inf"),
            })

    # Sort by composite score
    valid = [r for r in scored_results if "error" not in r]
    if not valid:
        return {"best_crf": None, "best_score": None, "results": scored_results}

    valid.sort(key=lambda r: r["composite"])
    best = valid[0]

    print(f"\n=== OPTIMAL CRF: {best['crf']} ===")
    print(f"  Composite score: {best['composite']:.6f}")
    print(f"  MAE mean: {best['mae_mean']:.4f}")
    print(f"  Archive size: {best['archive_bytes']:,} bytes")

    # Save results
    out_path = _PROJECT_ROOT / "experiments" / "crf_optimal_results.json"
    out_data = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "crf_range": crf_range,
        "best_crf": best["crf"],
        "best_composite": best["composite"],
        "results": scored_results,
    }
    optimal_tag = f"crf{best['crf']}"
    versioned_path = versioned_write(out_path, json.dumps(out_data, indent=2) + "\n", config_tag=optimal_tag)
    print(f"Results saved to: {versioned_path}")

    return {
        "best_crf": best["crf"],
        "best_score": best["composite"],
        "results": scored_results,
    }
