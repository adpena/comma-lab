"""Scorer-faithful proxy evaluation logic.

Mimics the real evaluator's pipeline as closely as possible:

1. Loads int8 weights through ``inflate_postfilter.load_postfilter_int8``
   (same loader the submission uses)
2. Applies the post-filter to ALL frames (no subsampling)
3. Batches frame pairs in groups of 16 (same as evaluator's batch_size=16)
4. Runs PoseNet + SegNet on CPU (same device as the authoritative scorer)
5. Computes the exact same distortion metrics and score formula

This closes the proxy->scorer gap by eliminating:
- Subsample bias (evaluates all 600 pairs)
- Device mismatch (CPU, not MPS)
- Loader mismatch (int8 path, not fp32 state_dict)
- Batch size mismatch (16, not 1)
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import argparse

REPORT_PATTERNS = {
    "pose_distortion": re.compile(r"Average PoseNet Distortion:\s*([0-9.]+)"),
    "seg_distortion": re.compile(r"Average SegNet Distortion:\s*([0-9.]+)"),
    "archive_bytes": re.compile(r"Submission file size:\s*([0-9,]+) bytes"),
    "original_bytes": re.compile(r"Original uncompressed size:\s*([0-9,]+) bytes"),
    "rate": re.compile(r"Compression Rate:\s*([0-9.]+)"),
    "score": re.compile(r"Final score: .* =\s*([0-9.]+)"),
}


def _default_paths():
    """Compute default project-relative paths.

    Returns (PROJECT, UPSTREAM, VIDEOS_DIR, LIVE_ARCHIVE_ZIP, LEGACY_ARCHIVE_ZIP).
    """
    # Walk up from this file: src/tac/proxy_eval.py -> src/tac -> src -> project root
    project = Path(__file__).resolve().parent.parent.parent
    upstream = project / "workspace" / "upstream" / "comma_video_compression_challenge"
    videos_dir = upstream / "videos"
    live_archive = project / "submissions" / "robust_current" / "archive.zip"
    legacy_archive = project / "reports" / "raw" / "2026-04-06-av1-roi-experiments" / "decode_base_archive.zip"
    return project, upstream, videos_dir, live_archive, legacy_archive


def resolve_archive_zip(
    archive_zip: str | os.PathLike[str] | Path | None,
    *,
    project_root: Path | None = None,
) -> Path:
    """Find the archive zip to evaluate, with sensible fallback logic.

    Parameters
    ----------
    archive_zip : path or None
        Explicit path.  When *None*, falls back to the live submission archive
        and then the legacy decode-base archive.
    project_root : Path or None
        Override project root (defaults to auto-detected).
    """
    if archive_zip is not None:
        return Path(archive_zip)

    if project_root is None:
        project_root = _default_paths()[0]

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
    """Set up a submission directory matching upstream evaluate.py expectations."""
    submission_dir = work_root / "submission"
    inflated_dir = submission_dir / "inflated"
    inflated_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(archive_zip, submission_dir / "archive.zip")
    shutil.copy2(raw_path, inflated_dir / f"{video_stem}.raw")
    return submission_dir


def run_upstream_evaluate(submission_dir: Path, *, device: str = "cpu",
                          upstream: Path | None = None,
                          videos_dir: Path | None = None) -> Path:
    """Shell out to upstream ``evaluate.py``.

    Parameters
    ----------
    submission_dir : Path
        Directory prepared by :func:`prepare_submission_dir`.
    device : str
        Torch device for the scorer.
    upstream : Path or None
        Upstream repo root (auto-detected when *None*).
    videos_dir : Path or None
        Directory with uncompressed reference videos.
    """
    if upstream is None:
        _, upstream, _videos, _, _ = _default_paths()
    if videos_dir is None:
        videos_dir = upstream / "videos"

    report_path = submission_dir / "report.txt"
    cmd = [
        sys.executable,
        str(upstream / "evaluate.py"),
        "--submission-dir",
        str(submission_dir),
        "--uncompressed-dir",
        str(videos_dir),
        "--report",
        str(report_path),
        "--video-names-file",
        str(upstream / "public_test_video_names.txt"),
        "--device",
        device,
    ]
    subprocess.run(cmd, check=True, cwd=upstream)
    return report_path


def parse_upstream_report(report_path: Path) -> dict[str, float | int]:
    """Parse a report.txt produced by upstream ``evaluate.py``.

    Returns a dict with keys: pose_distortion, seg_distortion,
    archive_bytes, original_bytes, rate, score.
    """
    text = report_path.read_text()
    parsed: dict[str, float | int] = {}
    for key, pattern in REPORT_PATTERNS.items():
        match = pattern.search(text)
        if not match:
            raise ValueError(f"Could not parse {key} from report {report_path}")
        raw = match.group(1).replace(",", "")
        parsed[key] = int(raw) if key.endswith("bytes") else float(raw)
    return parsed


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for the proxy-faithful evaluator."""
    parser = argparse.ArgumentParser(description="Scorer-faithful post-filter proxy")
    parser.add_argument("weights_path", help="Path to int8 post-filter weights")
    parser.add_argument(
        "--archive-zip",
        type=Path,
        default=None,
        help="Archive zip to evaluate against. Defaults to submissions/robust_current/archive.zip when present.",
    )
    return parser


def run_faithful_proxy(weights_path: str, archive_zip: Path | None = None,
                       device: str = "cpu") -> dict:
    """End-to-end scorer-faithful proxy evaluation.

    Parameters
    ----------
    weights_path : str
        Path to int8 post-filter weights.
    archive_zip : Path or None
        Archive to evaluate (auto-resolved when *None*).
    device : str
        Torch device for the scorer.

    Returns
    -------
    dict
        Result dict with pose_distortion, seg_distortion, rate, score, etc.
    """
    project, upstream, videos_dir, _, _ = _default_paths()

    # Ensure upstream and submission code are importable
    if str(upstream) not in sys.path:
        sys.path.insert(0, str(upstream))
    robust_dir = str(project / "submissions" / "robust_current")
    if robust_dir not in sys.path:
        sys.path.insert(0, robust_dir)

    from inflate_postfilter import inflate_with_postfilter

    archive_zip = resolve_archive_zip(archive_zip, project_root=project)

    print(f"[proxy-faithful] weights: {weights_path}")
    print(f"[proxy-faithful] archive: {archive_zip}")
    print(f"[proxy-faithful] device: {device}")

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
        report_path = run_upstream_evaluate(submission_dir, device=device,
                                            upstream=upstream, videos_dir=videos_dir)
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
    return result


def main():
    """CLI entry point."""
    args = build_arg_parser().parse_args()
    result = run_faithful_proxy(
        args.weights_path,
        archive_zip=args.archive_zip,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
