#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GRAPHS = ROOT / "reports" / "graphs"
MEDIA = GRAPHS / "media"
SOURCE_VIDEO = ROOT / "workspace" / "upstream" / "comma_video_compression_challenge" / "videos" / "0.mkv"
SUBMISSION_DIR = ROOT / "submissions" / "robust_current"
MEDIA_META = MEDIA / "comparison_manifest.json"

CLIP_START_SEC = 12
CLIP_DURATION_SEC = 6
OUTPUT_W = 960
OUTPUT_H = 720
ZOOM_W = 420
ZOOM_H = 316
ZOOM_X = 372
ZOOM_Y = 286
POSTER_OFFSET_SEC = 2.0


def run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def ensure_inflated_raw(tmp_dir: Path) -> Path:
    archive_dir = tmp_dir / "archive"
    inflated_dir = tmp_dir / "inflated"
    one_video_file = tmp_dir / "one-video.txt"
    archive_dir.mkdir(parents=True, exist_ok=True)
    inflated_dir.mkdir(parents=True, exist_ok=True)
    run(["unzip", "-q", str(SUBMISSION_DIR / "archive.zip"), "-d", str(archive_dir)], cwd=ROOT)
    one_video_file.write_text("0.mkv\n")
    env = os.environ.copy()
    env["CONFIG_ENV_PATH"] = str(SUBMISSION_DIR / "config.env")
    run(
        [
            "bash",
            str(SUBMISSION_DIR / "inflate.sh"),
            str(archive_dir),
            str(inflated_dir),
            str(one_video_file),
        ],
        cwd=ROOT,
        env=env,
    )
    return inflated_dir / "0.raw"


def build() -> None:
    MEDIA.mkdir(parents=True, exist_ok=True)
    stale_helper = MEDIA / "one-video.txt"
    if stale_helper.exists():
        stale_helper.unlink()
    with tempfile.TemporaryDirectory(prefix="comma_lab_media_") as tmp:
        tmp_dir = Path(tmp)
        raw_path = ensure_inflated_raw(tmp_dir)
        original_mp4 = MEDIA / "original_preview.mp4"
        inflated_mp4 = MEDIA / "inflated_preview.mp4"
        original_zoom_mp4 = MEDIA / "original_zoom_preview.mp4"
        inflated_zoom_mp4 = MEDIA / "inflated_zoom_preview.mp4"
        original_poster = MEDIA / "original_poster.jpg"
        inflated_poster = MEDIA / "inflated_poster.jpg"
        original_zoom_poster = MEDIA / "original_zoom_poster.jpg"
        inflated_zoom_poster = MEDIA / "inflated_zoom_poster.jpg"

        run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                str(CLIP_START_SEC),
                "-t",
                str(CLIP_DURATION_SEC),
                "-i",
                str(SOURCE_VIDEO),
                "-vf",
                f"scale={OUTPUT_W}:{OUTPUT_H}:flags=lanczos",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                str(original_mp4),
            ],
            cwd=ROOT,
        )
        run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                str(CLIP_START_SEC + POSTER_OFFSET_SEC),
                "-i",
                str(SOURCE_VIDEO),
                "-frames:v",
                "1",
                "-update",
                "1",
                "-vf",
                f"scale={OUTPUT_W}:{OUTPUT_H}:flags=lanczos",
                str(original_poster),
            ],
            cwd=ROOT,
        )

        run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-video_size",
                "1164x874",
                "-framerate",
                "20",
                "-ss",
                str(CLIP_START_SEC),
                "-t",
                str(CLIP_DURATION_SEC),
                "-i",
                str(raw_path),
                "-vf",
                f"scale={OUTPUT_W}:{OUTPUT_H}:flags=lanczos",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                str(inflated_mp4),
            ],
            cwd=ROOT,
        )
        run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-video_size",
                "1164x874",
                "-framerate",
                "20",
                "-ss",
                str(CLIP_START_SEC + POSTER_OFFSET_SEC),
                "-i",
                str(raw_path),
                "-frames:v",
                "1",
                "-update",
                "1",
                "-vf",
                f"scale={OUTPUT_W}:{OUTPUT_H}:flags=lanczos",
                str(inflated_poster),
            ],
            cwd=ROOT,
        )

        zoom_filter = f"crop={ZOOM_W}:{ZOOM_H}:{ZOOM_X}:{ZOOM_Y},scale={OUTPUT_W}:{OUTPUT_H}:flags=lanczos"
        run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                str(CLIP_START_SEC),
                "-t",
                str(CLIP_DURATION_SEC),
                "-i",
                str(SOURCE_VIDEO),
                "-vf",
                zoom_filter,
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                str(original_zoom_mp4),
            ],
            cwd=ROOT,
        )
        run(
            [
                "ffmpeg",
                "-y",
                "-ss",
                str(CLIP_START_SEC + POSTER_OFFSET_SEC),
                "-i",
                str(SOURCE_VIDEO),
                "-frames:v",
                "1",
                "-update",
                "1",
                "-vf",
                zoom_filter,
                str(original_zoom_poster),
            ],
            cwd=ROOT,
        )

        run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-video_size",
                "1164x874",
                "-framerate",
                "20",
                "-ss",
                str(CLIP_START_SEC),
                "-t",
                str(CLIP_DURATION_SEC),
                "-i",
                str(raw_path),
                "-vf",
                zoom_filter,
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "18",
                str(inflated_zoom_mp4),
            ],
            cwd=ROOT,
        )
        run(
            [
                "ffmpeg",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-video_size",
                "1164x874",
                "-framerate",
                "20",
                "-ss",
                str(CLIP_START_SEC + POSTER_OFFSET_SEC),
                "-i",
                str(raw_path),
                "-frames:v",
                "1",
                "-update",
                "1",
                "-vf",
                zoom_filter,
                str(inflated_zoom_poster),
            ],
            cwd=ROOT,
        )

    MEDIA_META.write_text(
        json.dumps(
            {
                "source_video": str(SOURCE_VIDEO.relative_to(ROOT)),
                "submission_archive": str((SUBMISSION_DIR / "archive.zip").relative_to(ROOT)),
                "clip_start_sec": CLIP_START_SEC,
                "clip_duration_sec": CLIP_DURATION_SEC,
                "zoom_crop": {
                    "x": ZOOM_X,
                    "y": ZOOM_Y,
                    "w": ZOOM_W,
                    "h": ZOOM_H,
                },
                "assets": [
                    "original_preview.mp4",
                    "inflated_preview.mp4",
                    "original_zoom_preview.mp4",
                    "inflated_zoom_preview.mp4",
                    "original_poster.jpg",
                    "inflated_poster.jpg",
                    "original_zoom_poster.jpg",
                    "inflated_zoom_poster.jpg",
                ],
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    build()
