#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""Contest-compliant end-to-end evaluation.

Single command: archive.zip → extract → inflate → upstream evaluate.py → score.

Directory layout matches upstream evaluate.py expectations exactly:
    work_dir/
        archive.zip          ← rate calculation
        inflated/0.raw       ← TensorVideoDataset

Usage:
    python experiments/contest_eval.py --archive submissions/robust_current/archive.zip
    python experiments/contest_eval.py --archive submissions/robust_current/archive.zip --device cuda
    python experiments/contest_eval.py --archive submissions/robust_current/archive.zip --keep-work-dir

This is the ONLY valid way to measure a contest-compliant score. All other
eval scripts (auth_eval_renderer.py, modal_auth_eval.py) are proxies.
"""
import argparse
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


# Upstream constants (from frame_utils.py)
OUT_W, OUT_H = 1164, 874
NUM_FRAMES = 1200
EXPECTED_RAW_BYTES = OUT_W * OUT_H * 3 * NUM_FRAMES  # 3,662,409,600


def find_repo_root() -> Path:
    """Walk up from this file to find the repo root (contains upstream/)."""
    p = Path(__file__).resolve().parent
    while p != p.parent:
        if (p / "upstream" / "evaluate.py").exists():
            return p
        p = p.parent
    raise FileNotFoundError("Cannot find repo root (no upstream/evaluate.py)")


def validate_archive(archive_path: Path) -> int:
    """Validate archive exists and return size in bytes."""
    if not archive_path.exists():
        print(f"FATAL: Archive not found: {archive_path}", file=sys.stderr)
        sys.exit(1)
    size = archive_path.stat().st_size
    if size == 0:
        print(f"FATAL: Archive is empty: {archive_path}", file=sys.stderr)
        sys.exit(1)
    return size


def extract_archive(archive_path: Path, extract_dir: Path) -> None:
    """Extract archive.zip to extract_dir."""
    repo_root = find_repo_root()
    src_path = str(repo_root / "src")
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    from tac.submission_archive import safe_extract_zip

    extract_dir.mkdir(parents=True, exist_ok=True)
    safe_extract_zip(archive_path, extract_dir)


def run_inflate(
    repo_root: Path,
    archive_dir: Path,
    inflated_dir: Path,
    video_names_file: Path,
) -> float:
    """Run inflate_renderer.py and return elapsed time."""
    inflate_script = repo_root / "submissions" / "robust_current" / "inflate_renderer.py"
    if not inflate_script.exists():
        print(f"FATAL: inflate script not found: {inflate_script}", file=sys.stderr)
        sys.exit(1)

    inflated_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root / 'src'}:{repo_root / 'upstream'}:{repo_root}"
    # Remove env vars that would make inflate non-contest-compliant
    for unsafe_var in ("INFLATE_MASK_SOURCE", "INFLATE_TTO", "INFLATE_MINI_TTO",
                       "INFLATE_SKIP_MASKS"):
        env.pop(unsafe_var, None)

    t0 = time.monotonic()
    result = subprocess.run(
        [
            sys.executable, str(inflate_script),
            str(archive_dir),
            str(inflated_dir),
            str(video_names_file),
        ],
        env=env,
        capture_output=False,
    )
    elapsed = time.monotonic() - t0

    if result.returncode != 0:
        print(f"FATAL: inflate_renderer.py failed (exit {result.returncode})", file=sys.stderr)
        sys.exit(1)

    return elapsed


def validate_inflated(inflated_dir: Path, video_names: list[str]) -> None:
    """Validate inflated output matches upstream expectations."""
    for name in video_names:
        raw_name = Path(name).with_suffix(".raw").name
        raw_path = inflated_dir / raw_name
        if not raw_path.exists():
            print(f"FATAL: Inflated file missing: {raw_path}", file=sys.stderr)
            sys.exit(1)
        size = raw_path.stat().st_size
        if size != EXPECTED_RAW_BYTES:
            print(
                f"FATAL: {raw_path} is {size:,} bytes, expected {EXPECTED_RAW_BYTES:,} "
                f"({NUM_FRAMES} frames × {OUT_H}×{OUT_W}×3)",
                file=sys.stderr,
            )
            sys.exit(1)


def run_upstream_evaluate(
    repo_root: Path,
    submission_dir: Path,
    video_names_file: Path,
    device: str,
    batch_size: int,
    report_path: Path,
) -> float:
    """Run upstream evaluate.py and return elapsed time."""
    evaluate_script = repo_root / "upstream" / "evaluate.py"
    uncompressed_dir = repo_root / "upstream" / "videos"

    if not evaluate_script.exists():
        print(f"FATAL: upstream evaluate.py not found: {evaluate_script}", file=sys.stderr)
        sys.exit(1)
    if not uncompressed_dir.exists():
        print(f"FATAL: upstream videos not found: {uncompressed_dir}", file=sys.stderr)
        sys.exit(1)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{repo_root / 'upstream'}:{repo_root}"

    t0 = time.monotonic()
    result = subprocess.run(
        [
            sys.executable, str(evaluate_script),
            "--submission-dir", str(submission_dir),
            "--uncompressed-dir", str(uncompressed_dir),
            "--video-names-file", str(video_names_file),
            "--device", device,
            "--batch-size", str(batch_size),
            "--report", str(report_path),
        ],
        env=env,
        capture_output=False,
    )
    elapsed = time.monotonic() - t0

    if result.returncode != 0:
        print(f"FATAL: upstream evaluate.py failed (exit {result.returncode})", file=sys.stderr)
        sys.exit(1)

    return elapsed


def parse_report(report_path: Path) -> dict:
    """Parse upstream report.txt into structured data.

    Raises ValueError if any required field is missing — prevents silent
    fallback to zero which would produce unrealistically good scores.
    """
    text = report_path.read_text()
    result = {}
    for line in text.splitlines():
        line = line.strip()
        if "PoseNet Distortion" in line:
            result["posenet_dist"] = float(line.split(":")[-1].strip())
        elif "SegNet Distortion" in line:
            result["segnet_dist"] = float(line.split(":")[-1].strip())
        elif "Submission file size" in line:
            result["archive_bytes"] = int(line.split(":")[-1].strip().replace(",", "").split()[0])
        elif "Original uncompressed size" in line:
            result["gt_bytes"] = int(line.split(":")[-1].strip().replace(",", "").split()[0])
        elif "Compression Rate" in line:
            result["rate"] = float(line.split(":")[-1].strip())
        elif "Final score" in line:
            result["score"] = float(line.split("=")[-1].strip())

    required = ["posenet_dist", "segnet_dist", "rate", "score"]
    missing = [k for k in required if k not in result]
    if missing:
        raise ValueError(
            f"FATAL: Failed to parse upstream report — missing fields: {missing}\n"
            f"Report contents:\n{text}"
        )
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Contest-compliant e2e evaluation. One command, authoritative score.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--archive", required=True, type=Path,
        help="Path to archive.zip",
    )
    parser.add_argument(
        "--device", default=None,
        help="Torch device (default: auto-detect)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Scoring batch size (default: 4)",
    )
    parser.add_argument(
        "--keep-work-dir", action="store_true",
        help="Keep temporary work directory after eval (for debugging)",
    )
    parser.add_argument(
        "--work-dir", type=Path, default=None,
        help="Use specific work directory instead of temp (implies --keep-work-dir)",
    )
    parser.add_argument(
        "--output-json", type=Path, default=None,
        help="Write structured results to JSON file",
    )
    args = parser.parse_args()

    repo_root = find_repo_root()
    archive_path = args.archive.resolve()

    # Auto-detect device
    if args.device is None:
        import torch
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    # ── Step 0: Validate inputs ──────────────────────────────────
    archive_bytes = validate_archive(archive_path)
    gt_bytes = sum(
        f.stat().st_size
        for f in (repo_root / "upstream" / "videos").rglob("*")
        if f.is_file()
    )

    print("=" * 60)
    print("  Contest-Compliant E2E Evaluation")
    print(f"  Archive: {archive_path} ({archive_bytes:,} bytes)")
    print(f"  Device: {args.device}")
    print(f"  Rate (pre-computed): {archive_bytes / gt_bytes:.6f}")
    print("=" * 60)

    # ── Step 1: Set up work directory (matches upstream layout) ──
    if args.work_dir:
        work_dir = args.work_dir.resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        keep = True
    else:
        work_dir = Path(tempfile.mkdtemp(prefix="pact_eval_"))
        keep = args.keep_work_dir

    submission_dir = work_dir / "submission"
    archive_extract_dir = work_dir / "archive"
    inflated_dir = submission_dir / "inflated"
    report_path = work_dir / "report.txt"

    # video_names.txt
    video_names_file = work_dir / "video_names.txt"
    video_names_file.write_text("0.mkv\n")
    video_names = ["0.mkv"]

    try:
        # Copy archive.zip to submission_dir (upstream reads it for rate)
        submission_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(archive_path, submission_dir / "archive.zip")

        # ── Step 2: Extract archive ──────────────────────────────
        print("\n[1/4] Extracting archive...")
        extract_archive(archive_path, archive_extract_dir)
        print(f"  Extracted to {archive_extract_dir}")
        for f in sorted(archive_extract_dir.iterdir()):
            print(f"    {f.name}: {f.stat().st_size:,} bytes")

        # ── Step 3: Inflate ──────────────────────────────────────
        print("\n[2/4] Inflating (contest-compliant, no scorers)...")
        inflate_time = run_inflate(
            repo_root, archive_extract_dir, inflated_dir, video_names_file,
        )
        print(f"  Inflate time: {inflate_time:.1f}s")

        # ── Step 4: Validate inflated output ─────────────────────
        print("\n[3/4] Validating inflated output...")
        validate_inflated(inflated_dir, video_names)
        raw_size = (inflated_dir / "0.raw").stat().st_size
        print(f"  0.raw: {raw_size:,} bytes (expected {EXPECTED_RAW_BYTES:,}) OK")

        # ── Step 5: Score via upstream evaluate.py ────────────────
        print("\n[4/4] Scoring via upstream evaluate.py...")
        score_time = run_upstream_evaluate(
            repo_root, submission_dir, video_names_file,
            args.device, args.batch_size, report_path,
        )
        print(f"  Score time: {score_time:.1f}s")

        # ── Parse and display results ─────────────────────────────
        results = parse_report(report_path)
        total_time = inflate_time + score_time

        print("\n" + "=" * 60)
        print("  AUTHORITATIVE CONTEST-COMPLIANT RESULTS")
        print("=" * 60)
        posenet_d = results.get("posenet_dist", 0)
        segnet_d = results.get("segnet_dist", 0)
        rate = results.get("rate", archive_bytes / gt_bytes)
        score = results.get("score", 100 * segnet_d + math.sqrt(10 * posenet_d) + 25 * rate)

        print(f"  PoseNet dist:  {posenet_d:.8f}  →  √(10×d) = {math.sqrt(10 * posenet_d):.4f}")
        print(f"  SegNet dist:   {segnet_d:.8f}  →  100×d   = {100 * segnet_d:.4f}")
        print(f"  Archive:       {archive_bytes:,} bytes")
        print(f"  Rate:          {rate:.8f}  →  25×r    = {25 * rate:.4f}")
        print("  ──────────────────────────────────────")
        print(f"  SCORE:         {score:.2f}")
        print("  ──────────────────────────────────────")
        print(f"  Inflate time:  {inflate_time:.1f}s")
        print(f"  Score time:    {score_time:.1f}s")
        print(f"  Total time:    {total_time:.1f}s")
        print(f"  T4 budget:     {'PASS' if total_time < 1800 else 'FAIL'} ({total_time:.0f}s / 1800s)")
        print("=" * 60)

        # Write structured JSON
        structured = {
            "score": score,
            "posenet_dist": posenet_d,
            "segnet_dist": segnet_d,
            "archive_bytes": archive_bytes,
            "gt_bytes": gt_bytes,
            "rate": rate,
            "posenet_contribution": math.sqrt(10 * posenet_d),
            "segnet_contribution": 100 * segnet_d,
            "rate_contribution": 25 * rate,
            "inflate_time_s": inflate_time,
            "score_time_s": score_time,
            "total_time_s": total_time,
            "device": args.device,
            "archive_path": str(archive_path),
            "lane": "contest-compliant",
        }

        if args.output_json:
            from tac.repo_io import json_text

            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(json_text(structured), encoding="utf-8")
            print(f"\n  Results saved: {args.output_json}")

        # Also save alongside archive (include stem to avoid race with concurrent evals)
        result_name = f"contest_eval_{archive_path.stem}.json"
        default_json = archive_path.parent / result_name
        default_json.write_text(json_text(structured), encoding="utf-8")

    finally:
        if not keep:
            shutil.rmtree(work_dir, ignore_errors=True)
        else:
            print(f"\n  Work directory kept: {work_dir}")


if __name__ == "__main__":
    main()
