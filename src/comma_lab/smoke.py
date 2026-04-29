from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path

from .install import install_submission
from .lock import submission_lock
from .paths import default_upstream_root, repo_root
from .tracks.exact_current import create_minimal_archive

SOURCE_COLOR_RANGE = "tv"
# BT.601 (smpte170m) matches the scorer's preprocess_input which uses
# BT.601 coefficients (kYR=0.299, kYG=0.587, kYB=0.114). Using BT.709
# here causes a systematic 3-5% color error in every frame.
SOURCE_COLOR_MATRIX = "smpte170m"
SOURCE_COLOR_PRIMARIES = "smpte170m"
SOURCE_COLOR_TRC = "smpte170m"


def _ffmpeg_bin() -> str:
    return os.environ.get("FFMPEG_BIN", "ffmpeg")


def _ffprobe_bin() -> str:
    return os.environ.get("FFPROBE_BIN", "ffprobe")


@dataclass
class VideoSmokeResult:
    video: str
    width: int
    height: int
    fps: float
    expected_frames: int
    expected_frame_bytes: int
    expected_total_bytes: int
    actual_total_bytes: int | None
    actual_frames: int | None
    raw_exists: bool
    size_matches: bool
    frame_count_matches: bool
    semantic_sample_indices: list[int]
    semantic_mae_mean: float | None
    semantic_mae_max: float | None
    semantic_channel_mean_abs_diff: list[float] | None
    semantic_check_passed: bool


@dataclass
class SmokeSummary:
    track: str
    upstream_root: str
    video_names_file: str
    archive_path: str
    inflated_dir: str
    all_passed: bool
    file_count_matches: bool
    results: list[VideoSmokeResult]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)


def _run(
    cmd: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    capture: bool = False,
) -> subprocess.CompletedProcess[str]:
    if capture:
        return subprocess.run(
            cmd,
            cwd=cwd,
            env=env,
            check=True,
            text=True,
            capture_output=True,
        )
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        check=True,
        text=True,
        stdout=sys.stderr,
        stderr=sys.stderr,
    )


def _fps_from_rate(rate: str) -> float:
    frac = Fraction(rate)
    return frac.numerator / frac.denominator if frac.denominator else 0.0


def _video_meta(path: Path) -> dict[str, int | float]:
    cp = _run(
        [
            _ffprobe_bin(),
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-count_frames",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames,nb_read_frames,duration",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        cwd=repo_root(),
        capture=True,
    )
    data = json.loads(cp.stdout)
    stream = data["streams"][0]
    fps = _fps_from_rate(stream.get("avg_frame_rate", "0/1"))
    duration = float(stream.get("duration") or data.get("format", {}).get("duration") or 0.0)
    total_frames_raw = stream.get("nb_read_frames") or stream.get("nb_frames")
    if total_frames_raw in (None, "N/A"):
        total_frames = int(round(duration * fps))
    else:
        total_frames = int(total_frames_raw)
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": fps,
        "frames": total_frames,
    }


def _semantic_sample_indices(total_frames: int) -> list[int]:
    if total_frames <= 0:
        return []
    points = {0, max(total_frames // 2, 0), max(total_frames - 1, 0)}
    return sorted(points)


def _extract_reference_rgb_frames(video: Path, frame_indices: list[int], frame_size: int) -> list[bytes]:
    if not frame_indices:
        return []
    expr = "+".join(f"eq(n\\,{idx})" for idx in frame_indices)
    cp = subprocess.run(  # subprocess-no-check-OK: check=True is set on line 175 below — preflight 8-line window misses it
        [
            _ffmpeg_bin(),
            "-v",
            "error",
            "-i",
            str(video),
            "-vf",
            (
                f"select='{expr}',"
                f"scale=iw:ih:flags=bilinear:"
                f"in_range={SOURCE_COLOR_RANGE}:out_range=pc:"
                f"in_color_matrix={SOURCE_COLOR_MATRIX}:"
                f"in_primaries={SOURCE_COLOR_PRIMARIES}:"
                f"in_transfer={SOURCE_COLOR_TRC},"
                f"format=rgb24"
            ),
            "-vsync",
            "0",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-",
        ],
        cwd=repo_root(),
        check=True,
        text=False,
        capture_output=True,
    )
    raw = cp.stdout
    if len(raw) % frame_size != 0:
        raise ValueError(f"Unexpected reference RGB buffer size {len(raw)} for frame size {frame_size}")
    return [raw[i : i + frame_size] for i in range(0, len(raw), frame_size)]


def _read_candidate_rgb_frames(raw_path: Path, frame_indices: list[int], frame_size: int) -> list[bytes]:
    frames: list[bytes] = []
    with raw_path.open("rb") as fh:
        for idx in frame_indices:
            fh.seek(idx * frame_size)
            chunk = fh.read(frame_size)
            if len(chunk) != frame_size:
                raise ValueError(f"Unexpected candidate RGB chunk size {len(chunk)} for frame {idx}")
            frames.append(chunk)
    return frames


def _semantic_metrics(reference_frames: list[bytes], candidate_frames: list[bytes]) -> tuple[float, float, list[float]]:
    if len(reference_frames) != len(candidate_frames):
        raise ValueError("Reference/candidate frame count mismatch in semantic metrics")
    frame_maes: list[float] = []
    channel_diffs = [0.0, 0.0, 0.0]
    channel_counts = [0, 0, 0]

    for ref, cand in zip(reference_frames, candidate_frames):
        abs_sum = 0
        for i, (a, b) in enumerate(zip(ref, cand)):
            diff = abs(a - b)
            abs_sum += diff
            channel = i % 3
            channel_diffs[channel] += diff
            channel_counts[channel] += 1
        frame_maes.append(abs_sum / max(1, len(ref)))

    mae_mean = sum(frame_maes) / max(1, len(frame_maes))
    mae_max = max(frame_maes) if frame_maes else 0.0
    channel_mean_abs_diff = [
        channel_diffs[i] / max(1, channel_counts[i])
        for i in range(3)
    ]
    return mae_mean, mae_max, channel_mean_abs_diff


def smoke_submission(
    name: str,
    *,
    upstream_root: Path | None = None,
    sync: bool = True,
    package: bool = False,
) -> SmokeSummary:
    root = repo_root()
    upstream_root = upstream_root or default_upstream_root()
    source_submission_dir = root / "submissions" / name
    with submission_lock(name, upstream_root):
        if package and not sync:
            raise ValueError("Packaging without sync is unsupported because the packaged artifact would not be the one under test.")

        if package:
            if name == "exact_current":
                create_minimal_archive(source_submission_dir / "archive.zip")
            elif name == "robust_current":
                package_env = os.environ.copy()
                package_env["COMMA_CHALLENGE_ROOT"] = str(upstream_root)
                _run(["bash", str(source_submission_dir / "compress.sh")], cwd=root, env=package_env)
            else:
                raise ValueError(f"Unsupported submission for packaging: {name}")

        if sync:
            install_submission(name, upstream_root=upstream_root, force=True)

        submission_dir = upstream_root / "submissions" / name
        archive_path = submission_dir / "archive.zip"
        archive_dir = submission_dir / "archive"
        inflated_dir = submission_dir / "inflated"
        video_names_file = upstream_root / "public_test_video_names.txt"

        if archive_dir.exists():
            subprocess.run(["rm", "-rf", str(archive_dir)], check=True)
        if inflated_dir.exists():
            subprocess.run(["rm", "-rf", str(inflated_dir)], check=True)
        archive_dir.mkdir(parents=True, exist_ok=True)
        inflated_dir.mkdir(parents=True, exist_ok=True)

        _run(["unzip", "-q", str(archive_path), "-d", str(archive_dir)], cwd=submission_dir)
        env = os.environ.copy()
        env["COMMA_CHALLENGE_ROOT"] = str(upstream_root)
        _run(
            [
                "bash",
                str(submission_dir / "inflate.sh"),
                str(archive_dir),
                str(inflated_dir),
                str(video_names_file),
            ],
            cwd=submission_dir,
            env=env,
        )

        results: list[VideoSmokeResult] = []
        listed = [line.strip() for line in video_names_file.read_text().splitlines() if line.strip()]
        for rel in listed:
            source_video = upstream_root / "videos" / rel
            meta = _video_meta(source_video)
            stem = rel.rsplit(".", 1)[0]
            raw_path = inflated_dir / f"{stem}.raw"
            frame_bytes = int(meta["width"]) * int(meta["height"]) * 3
            expected_total_bytes = int(meta["frames"]) * frame_bytes
            raw_exists = raw_path.exists()
            actual_total_bytes = raw_path.stat().st_size if raw_exists else None
            actual_frames = (
                actual_total_bytes // frame_bytes
                if raw_exists and actual_total_bytes is not None and actual_total_bytes % frame_bytes == 0
                else None
            )
            size_matches = actual_total_bytes == expected_total_bytes
            frame_count_matches = actual_frames == int(meta["frames"])
            semantic_indices = _semantic_sample_indices(int(meta["frames"]))
            semantic_mae_mean: float | None = None
            semantic_mae_max: float | None = None
            semantic_channel_mean_abs_diff: list[float] | None = None
            semantic_check_passed = False
            if raw_exists and frame_count_matches:
                try:
                    reference_frames = _extract_reference_rgb_frames(source_video, semantic_indices, frame_bytes)
                    candidate_frames = _read_candidate_rgb_frames(raw_path, semantic_indices, frame_bytes)
                    semantic_mae_mean, semantic_mae_max, semantic_channel_mean_abs_diff = _semantic_metrics(reference_frames, candidate_frames)
                    semantic_check_passed = (
                        semantic_mae_mean <= 80.0
                        and semantic_mae_max <= 120.0
                        and max(semantic_channel_mean_abs_diff) <= 100.0
                    )
                except Exception:
                    semantic_check_passed = False
            results.append(
                VideoSmokeResult(
                    video=rel,
                    width=int(meta["width"]),
                    height=int(meta["height"]),
                    fps=float(meta["fps"]),
                    expected_frames=int(meta["frames"]),
                    expected_frame_bytes=frame_bytes,
                    expected_total_bytes=expected_total_bytes,
                    actual_total_bytes=actual_total_bytes,
                    actual_frames=actual_frames,
                    raw_exists=raw_exists,
                    size_matches=size_matches,
                    frame_count_matches=frame_count_matches,
                    semantic_sample_indices=semantic_indices,
                    semantic_mae_mean=semantic_mae_mean,
                    semantic_mae_max=semantic_mae_max,
                    semantic_channel_mean_abs_diff=semantic_channel_mean_abs_diff,
                    semantic_check_passed=semantic_check_passed,
                )
            )

        actual_raws = sorted(p.relative_to(inflated_dir).as_posix() for p in inflated_dir.rglob("*.raw"))
        expected_raws = sorted(f"{rel.rsplit('.', 1)[0]}.raw" for rel in listed)
        file_count_matches = actual_raws == expected_raws
        all_passed = file_count_matches and all(
            r.raw_exists and r.size_matches and r.frame_count_matches and r.semantic_check_passed
            for r in results
        )
        return SmokeSummary(
            track=name,
            upstream_root=str(upstream_root),
            video_names_file=str(video_names_file),
            archive_path=str(archive_path),
            inflated_dir=str(inflated_dir),
            all_passed=all_passed,
            file_count_matches=file_count_matches,
            results=results,
        )
