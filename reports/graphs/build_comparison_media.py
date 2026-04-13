#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

_GRAPHS_DIR = Path(__file__).resolve().parent
if str(_GRAPHS_DIR) not in sys.path:
    sys.path.insert(0, str(_GRAPHS_DIR))
from _versioned_output import versioned_write

ROOT = Path(__file__).resolve().parents[2]
GRAPHS = ROOT / "reports" / "graphs"
MEDIA = GRAPHS / "media"
SOURCE_VIDEO = ROOT / "workspace" / "upstream" / "comma_video_compression_challenge" / "videos" / "0.mkv"
SUBMISSION_DIR = ROOT / "submissions" / "robust_current"
MEDIA_META = MEDIA / "comparison_manifest.json"
WEIGHTS_DIR = ROOT / "experiments" / "postfilter_weights"

CLIP_START_SEC = 12
CLIP_DURATION_SEC = 6
OUTPUT_W = 960
OUTPUT_H = 720
ZOOM_W = 420
ZOOM_H = 316
ZOOM_X = 372
ZOOM_Y = 286
POSTER_OFFSET_SEC = 2.0

TOP_VARIANTS = [
    {
        "id": "floor_173",
        "label": "1.73 current floor",
        "note": "long1000 h64",
        "score": 1.73,
        "weight_path": SUBMISSION_DIR / "postfilter_int8.pt",
        "preview": "inflated_preview.mp4",
        "poster": "inflated_poster.jpg",
    },
    {
        "id": "floor_184",
        "label": "1.84 prior floor",
        "note": "ensemble h32 + MC 75/25",
        "score": 1.84,
        "weight_path": WEIGHTS_DIR / "postfilter_ensemble_h32_qat_mc75_25_int8.pt",
        "preview": "inflated_184_preview.mp4",
        "poster": "inflated_184_poster.jpg",
    },
    {
        "id": "floor_185",
        "label": "1.85 older floor",
        "note": "long1000 h32",
        "score": 1.85,
        "weight_path": WEIGHTS_DIR / "postfilter_long1000_qat_ema_alpha20_h32_int8.pt",
        "preview": "inflated_185_preview.mp4",
        "poster": "inflated_185_poster.jpg",
    },
]
CURRENT_VARIANT_ID = TOP_VARIANTS[0]["id"]

LEADERBOARD_REFS = [
    {"label": "Public #1", "score": 1.89, "note": "neural_inflate"},
    {"label": "Public #2", "score": 1.94, "note": "roi_v2"},
    {"label": "Public #3", "score": 1.95, "note": "av1_roi_lanczos_unsharp"},
]


def run(cmd: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def needs_rebuild(outputs: list[Path], inputs: list[Path]) -> bool:
    if not all(p.exists() for p in outputs):
        return True
    newest_input = max(p.stat().st_mtime for p in inputs if p.exists())
    oldest_output = min(p.stat().st_mtime for p in outputs)
    return oldest_output < newest_input


def ensure_inflated_raw(tmp_dir: Path, *, postfilter_path: Path | None = None) -> Path:
    archive_dir = tmp_dir / "archive"
    inflated_dir = tmp_dir / "inflated"
    one_video_file = tmp_dir / "one-video.txt"
    archive_dir.mkdir(parents=True, exist_ok=True)
    inflated_dir.mkdir(parents=True, exist_ok=True)
    run(["unzip", "-q", str(SUBMISSION_DIR / "archive.zip"), "-d", str(archive_dir)], cwd=ROOT)
    one_video_file.write_text("0.mkv\n")
    env = os.environ.copy()
    env["CONFIG_ENV_PATH"] = str(SUBMISSION_DIR / "config.env")
    if postfilter_path is not None:
        env["POSTFILTER_PATH"] = str(postfilter_path)
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


def build_preview_from_raw(raw_path: Path, *, preview_path: Path, poster_path: Path) -> None:
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
            str(preview_path),
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
            str(poster_path),
        ],
        cwd=ROOT,
    )


def build() -> None:
    MEDIA.mkdir(parents=True, exist_ok=True)
    stale_helper = MEDIA / "one-video.txt"
    if stale_helper.exists():
        stale_helper.unlink()
    if not SOURCE_VIDEO.exists():
        raise FileNotFoundError(f"Source video not found: {SOURCE_VIDEO}")
    if not (SUBMISSION_DIR / "archive.zip").exists():
        raise FileNotFoundError(f"Submission archive not found: {SUBMISSION_DIR / 'archive.zip'}")
    for variant in TOP_VARIANTS:
        weight_path = Path(variant["weight_path"])
        if not weight_path.exists():
            raise FileNotFoundError(
                f"Required explorer weight not found for {variant['id']}: {weight_path}"
            )

    with tempfile.TemporaryDirectory(prefix="comma_lab_media_") as tmp:
        tmp_dir = Path(tmp)
        original_mp4 = MEDIA / "original_preview.mp4"
        original_zoom_mp4 = MEDIA / "original_zoom_preview.mp4"
        original_poster = MEDIA / "original_poster.jpg"
        original_zoom_poster = MEDIA / "original_zoom_poster.jpg"
        inflated_zoom_mp4 = MEDIA / "inflated_zoom_preview.mp4"
        inflated_zoom_poster = MEDIA / "inflated_zoom_poster.jpg"

        if needs_rebuild([original_mp4, original_poster], [SOURCE_VIDEO]):
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

        variant_manifest = []
        current_variant_raw: Path | None = None
        for variant in TOP_VARIANTS:
            weight_path = Path(variant["weight_path"])
            preview_path = MEDIA / variant["preview"]
            poster_path = MEDIA / variant["poster"]
            outputs = [preview_path, poster_path]
            inputs = [SOURCE_VIDEO, SUBMISSION_DIR / "archive.zip", weight_path]
            if needs_rebuild(outputs, inputs):
                variant_tmp = tmp_dir / variant["id"]
                raw_path = ensure_inflated_raw(variant_tmp, postfilter_path=weight_path)
                build_preview_from_raw(raw_path, preview_path=preview_path, poster_path=poster_path)
                if variant["id"] == CURRENT_VARIANT_ID:
                    current_variant_raw = raw_path
            elif variant["id"] == CURRENT_VARIANT_ID:
                current_variant_raw = ensure_inflated_raw(tmp_dir / "current_zoom", postfilter_path=weight_path)
            variant_manifest.append(
                {
                    "id": variant["id"],
                    "label": variant["label"],
                    "note": variant["note"],
                    "score": variant["score"],
                    "preview": variant["preview"],
                    "poster": variant["poster"],
                }
            )

        if current_variant_raw is None:
            current_variant_raw = ensure_inflated_raw(tmp_dir / "current_zoom_fallback", postfilter_path=Path(TOP_VARIANTS[0]["weight_path"]))

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

        if needs_rebuild([inflated_zoom_mp4, inflated_zoom_poster], [SOURCE_VIDEO, SUBMISSION_DIR / "archive.zip", Path(TOP_VARIANTS[0]["weight_path"])]):
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
                    str(current_variant_raw),
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
                    str(current_variant_raw),
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

    versioned_write(
        MEDIA_META,
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
                "source_size": {"w": 1164, "h": 874},
                "preview_size": {"w": OUTPUT_W, "h": OUTPUT_H},
                "variants": variant_manifest,
                "leaderboard_refs": LEADERBOARD_REFS,
                "updated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "assets": [
                    "original_preview.mp4",
                    *(variant["preview"] for variant in TOP_VARIANTS),
                    "original_zoom_preview.mp4",
                    "inflated_zoom_preview.mp4",
                    "original_poster.jpg",
                    *(variant["poster"] for variant in TOP_VARIANTS),
                    "original_zoom_poster.jpg",
                    "inflated_zoom_poster.jpg",
                ],
            },
            indent=2,
        )
        + "\n",
        config_tag="robust_current",
    )


if __name__ == "__main__":
    build()
