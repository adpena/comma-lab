#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Decode a video to raw RGB using upstream evaluator semantics."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def decode(args: argparse.Namespace) -> dict[str, Any]:
    upstream_dir = args.upstream_dir.resolve()
    if str(upstream_dir) not in sys.path:
        sys.path.insert(0, str(upstream_dir))
    import av
    from frame_utils import camera_size, yuv420_to_rgb

    video = args.video.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not args.force:
        raise FileExistsError(f"output exists; pass --force to overwrite: {output}")
    width, height = camera_size
    frame_bytes = int(width) * int(height) * 3
    start = time.monotonic()
    frame_count = 0
    tmp = output.with_suffix(output.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    with tmp.open("wb") as handle:
        container = av.open(str(video))
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            arr = yuv420_to_rgb(frame).numpy()
            if arr.shape != (height, width, 3):
                raise ValueError(f"unexpected decoded frame shape {arr.shape}; expected {(height, width, 3)}")
            handle.write(arr.astype("uint8", copy=False).tobytes())
            frame_count += 1
        container.close()
    tmp.replace(output)
    elapsed = time.monotonic() - start
    payload = {
        "schema": "upstream_video_raw_decode.v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "producer": "tools/decode_upstream_video_to_raw.py",
        "video": str(video),
        "video_sha256": _sha256_file(video),
        "output": str(output),
        "output_bytes": output.stat().st_size,
        "output_sha256": _sha256_file(output),
        "frame_count": frame_count,
        "frame_shape": {"height": height, "width": width, "channels": 3},
        "frame_bytes": frame_bytes,
        "expected_total_bytes": frame_count * frame_bytes,
        "elapsed_seconds": elapsed,
        "decode_semantics": "upstream.frame_utils.yuv420_to_rgb via PyAV, matching upstream/evaluate.py CPU AVVideoDataset",
        "score_claim": False,
    }
    if output.stat().st_size != frame_count * frame_bytes:
        raise ValueError(
            f"raw size mismatch after decode: got {output.stat().st_size}, expected {frame_count * frame_bytes}"
        )
    if args.manifest:
        _write_json(args.manifest.resolve(), payload)
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--upstream-dir", type=Path, default=REPO_ROOT / "upstream")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    payload = decode(parse_args(argv))
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
