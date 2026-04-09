#!/usr/bin/env python
"""Scorer-faithful local proxy for post-filter evaluation.

This script mimics the real evaluator's pipeline as closely as possible:

1. Loads int8 weights through ``inflate_postfilter.load_postfilter_int8``
   (same loader the submission uses)
2. Applies the post-filter to ALL frames (no subsampling)
3. Batches frame pairs in groups of 16 (same as evaluator's batch_size=16)
4. Runs PoseNet + SegNet on CPU (same device as the authoritative scorer)
5. Computes the exact same distortion metrics and score formula

This closes the proxy→scorer gap by eliminating:
- Subsample bias (evaluates all 600 pairs)
- Device mismatch (CPU, not MPS)
- Loader mismatch (int8 path, not fp32 state_dict)
- Batch size mismatch (16, not 1)

Usage::

    uv run --with av --with torch --with safetensors --with timm --with einops \\
           --with segmentation-models-pytorch --with numpy \\
           python experiments/proxy_score_faithful.py \\
           experiments/postfilter_weights/postfilter_saliency_alpha20_int8.pt

Prints a JSON summary compatible with the real scorer output.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parent
UPSTREAM = PROJECT / "workspace" / "upstream" / "comma_video_compression_challenge"
VIDEOS_DIR = UPSTREAM / "videos"
LIVE_ARCHIVE_ZIP = PROJECT / "submissions" / "robust_current" / "archive.zip"
LEGACY_ARCHIVE_ZIP = PROJECT / "reports" / "raw" / "2026-04-06-av1-roi-experiments" / "decode_base_archive.zip"

sys.path.insert(0, str(UPSTREAM))
sys.path.insert(0, str(PROJECT / "submissions" / "robust_current"))

from inflate_postfilter import inflate_with_postfilter


REPORT_PATTERNS = {
    "pose_distortion": re.compile(r"Average PoseNet Distortion:\s*([0-9.]+)"),
    "seg_distortion": re.compile(r"Average SegNet Distortion:\s*([0-9.]+)"),
    "archive_bytes": re.compile(r"Submission file size:\s*([0-9,]+) bytes"),
    "original_bytes": re.compile(r"Original uncompressed size:\s*([0-9,]+) bytes"),
    "rate": re.compile(r"Compression Rate:\s*([0-9.]+)"),
    "score": re.compile(r"Final score: .* =\s*([0-9.]+)"),
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scorer-faithful post-filter proxy")
    parser.add_argument("weights_path", help="Path to int8 post-filter weights")
    parser.add_argument(
        "--archive-zip",
        type=Path,
        default=None,
        help="Archive zip to evaluate against. Defaults to submissions/robust_current/archive.zip when present.",
    )
    return parser


def resolve_archive_zip(
    archive_zip: str | os.PathLike[str] | Path | None,
    *,
    project_root: Path = PROJECT,
) -> Path:
    if archive_zip is not None:
        return Path(archive_zip)

    live = project_root / "submissions" / "robust_current" / "archive.zip"
    if live.exists():
        return live

    legacy = project_root / "reports" / "raw" / "2026-04-06-av1-roi-experiments" / "decode_base_archive.zip"
    if legacy.exists():
        return legacy

    raise FileNotFoundError(
        "No archive zip found. Pass --archive-zip explicitly or package submissions/robust_current/archive.zip first."
    )
def prepare_submission_dir(work_root: Path, archive_zip: Path, raw_path: Path, *, video_stem: str = "0") -> Path:
    submission_dir = work_root / "submission"
    inflated_dir = submission_dir / "inflated"
    inflated_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(archive_zip, submission_dir / "archive.zip")
    shutil.copy2(raw_path, inflated_dir / f"{video_stem}.raw")
    return submission_dir


def run_upstream_evaluate(submission_dir: Path, *, device: str = "cpu") -> Path:
    report_path = submission_dir / "report.txt"
    cmd = [
        sys.executable,
        str(UPSTREAM / "evaluate.py"),
        "--submission-dir",
        str(submission_dir),
        "--uncompressed-dir",
        str(VIDEOS_DIR),
        "--report",
        str(report_path),
        "--video-names-file",
        str(UPSTREAM / "public_test_video_names.txt"),
        "--device",
        device,
    ]
    subprocess.run(cmd, check=True, cwd=UPSTREAM)
    return report_path


def parse_upstream_report(report_path: Path) -> dict[str, float | int]:
    text = report_path.read_text()
    parsed: dict[str, float | int] = {}
    for key, pattern in REPORT_PATTERNS.items():
        match = pattern.search(text)
        if not match:
            raise ValueError(f"Could not parse {key} from report {report_path}")
        raw = match.group(1).replace(",", "")
        parsed[key] = int(raw) if key.endswith("bytes") else float(raw)
    return parsed


def main():
    args = build_arg_parser().parse_args()
    weights_path = args.weights_path
    archive_zip = resolve_archive_zip(args.archive_zip)
    device = "cpu"

    print(f"[proxy-faithful] weights: {weights_path}")
    print(f"[proxy-faithful] archive: {archive_zip}")
    print(f"[proxy-faithful] device: {device}")

    # Inflate via the same path the submission uses
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        # Extract archive
        extract_dir = tmp_root / "archive"
        with zipfile.ZipFile(str(archive_zip)) as zf:
            zf.extractall(extract_dir)
        mkv_candidates = sorted(extract_dir.rglob("*.mkv"))
        if not mkv_candidates:
            raise FileNotFoundError(
                f"No .mkv found inside archive zip {archive_zip}"
            )
        mkv = mkv_candidates[0]

        raw_path = tmp_root / "inflated.raw"
        print(f"[proxy-faithful] Inflating {mkv} with post-filter...")
        n_frames = inflate_with_postfilter(
            str(mkv), str(raw_path), weights_path,
            target_w=1164, target_h=874, device=device,
        )
        print(f"[proxy-faithful] Inflated {n_frames} frames")

        submission_dir = prepare_submission_dir(tmp_root, archive_zip, raw_path, video_stem=mkv.stem)
        print(f"[proxy-faithful] Running upstream evaluate.py...")
        report_path = run_upstream_evaluate(submission_dir, device=device)
        parsed = parse_upstream_report(report_path)

    print(f"\n[proxy-faithful] Results:")
    print(f"  PoseNet distortion: {parsed['pose_distortion']:.8f}")
    print(f"  SegNet distortion:  {parsed['seg_distortion']:.8f}")
    print(f"  Compression rate:   {parsed['rate']:.8f}")
    print(f"  Final score:        {parsed['score']:.4f}")

    result = {
        "pose_distortion": parsed["pose_distortion"],
        "seg_distortion": parsed["seg_distortion"],
        "current_workflow_rate": parsed["rate"],
        "current_workflow_score": parsed["score"],
        "current_workflow_archive_bytes": parsed["archive_bytes"],
        "weights": weights_path,
        "archive": str(archive_zip),
        "device": device,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
