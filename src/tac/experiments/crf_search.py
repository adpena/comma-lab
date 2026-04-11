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

                print(f"  Running smoke proxy...")
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
            print(f"\n=== FULL SCORER RESULTS ===")
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
    out_path.write_text(json.dumps(out_data, indent=2) + "\n")
    print(f"\nResults saved to: {out_path}")

    return results
