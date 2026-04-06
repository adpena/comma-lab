#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
from pathlib import Path


def find_upstream_root(start: Path) -> Path:
    env = os.getenv("COMMA_CHALLENGE_ROOT")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env).resolve())

    current = start.resolve()
    candidates.extend([current, *current.parents])

    for cand in candidates:
        if (cand / "evaluate.sh").exists() and (cand / "videos").exists() and (cand / "frame_utils.py").exists():
            return cand
    raise FileNotFoundError(
        "Could not find upstream challenge root. "
        "Set COMMA_CHALLENGE_ROOT or place this submission inside the upstream repo."
    )


def load_upstream_converter(upstream_root: Path):
    frame_utils_path = upstream_root / "frame_utils.py"
    if not frame_utils_path.exists():
        return None

    spec = importlib.util.spec_from_file_location("upstream_frame_utils", frame_utils_path)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, "yuv420_to_rgb", None)


def decode_with_pyav(video_path: Path, raw_path: Path, converter) -> None:
    import av  # type: ignore

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("wb") as fout:
        container = av.open(str(video_path))
        try:
            stream = container.streams.video[0]
            for frame in container.decode(stream):
                if converter is not None:
                    arr = converter(frame)
                    if hasattr(arr, "cpu"):
                        arr = arr.cpu().numpy()
                else:
                    arr = frame.to_ndarray(format="rgb24")
                fout.write(arr.tobytes())
        finally:
            container.close()


def decode_with_ffmpeg(video_path: Path, raw_path: Path) -> None:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        os.getenv("FFMPEG_BIN", "ffmpeg"),
        "-y",
        "-i",
        str(video_path),
        "-an",
        "-sn",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        str(raw_path),
    ]
    subprocess.run(cmd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inflate exact-current submission by reconstructing raw frames from repo-side videos.")
    parser.add_argument("archive_dir", type=Path)
    parser.add_argument("inflated_dir", type=Path)
    parser.add_argument("video_names_file", type=Path)
    parser.add_argument("--upstream-root", type=Path, default=None)
    args = parser.parse_args()

    del args.archive_dir  # intentionally unused

    script_dir = Path(__file__).resolve().parent
    upstream_root = args.upstream_root or find_upstream_root(script_dir)
    converter = None
    try:
        converter = load_upstream_converter(upstream_root)
    except Exception:
        converter = None

    names = [line.strip() for line in args.video_names_file.read_text().splitlines() if line.strip()]

    for rel in names:
        rel_path = Path(rel)
        video_path = upstream_root / "videos" / rel_path
        raw_path = args.inflated_dir / rel_path.with_suffix(".raw")
        raw_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            decode_with_pyav(video_path, raw_path, converter)
        except Exception:
            decode_with_ffmpeg(video_path, raw_path)

        print(f"Inflated: {video_path} -> {raw_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
