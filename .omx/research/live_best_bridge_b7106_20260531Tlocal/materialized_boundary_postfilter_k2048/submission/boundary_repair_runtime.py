#!/usr/bin/env python
# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def _array(payload: dict, key: str, dtype: object) -> np.ndarray:
    return np.asarray(payload.get(key, []), dtype=dtype)


def apply_overlay(raw_path: Path, overlay_path: Path, video_stem: str) -> dict:
    overlay = json.loads(overlay_path.read_text(encoding="utf-8"))
    if overlay.get("video_stem") not in (None, "", video_stem):
        return {"applied": False, "reason": "video_stem_mismatch"}
    raw_shape = tuple(int(v) for v in overlay["raw_shape"])
    raw = np.memmap(raw_path, dtype=np.uint8, mode="r+", shape=raw_shape)
    frame = _array(overlay, "frame_indices", np.int64)
    y = _array(overlay, "y", np.int64)
    x = _array(overlay, "x", np.int64)
    strategy = str(overlay.get("strategy") or "")
    if strategy == "source_pixel_patch":
        rgb = _array(overlay, "rgb", np.uint8).reshape((-1, 3))
        raw[frame, y, x, :] = rgb
    elif strategy == "masked_local_median":
        radius = max(1, int(overlay.get("radius", 1)))
        for f, yy, xx in zip(frame.tolist(), y.tolist(), x.tolist(), strict=True):
            y0 = max(0, yy - radius)
            y1 = min(raw_shape[1], yy + radius + 1)
            x0 = max(0, xx - radius)
            x1 = min(raw_shape[2], xx + radius + 1)
            raw[f, yy, xx, :] = np.median(raw[f, y0:y1, x0:x1, :], axis=(0, 1)).astype(np.uint8)
    else:
        raise ValueError(f"unsupported boundary repair strategy: {strategy!r}")
    raw.flush()
    return {"applied": True, "strategy": strategy, "points": int(len(frame))}


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 3:
        print("usage: boundary_repair_runtime.py RAW_PATH OVERLAY_JSON VIDEO_STEM", file=sys.stderr)
        return 2
    result = apply_overlay(Path(args[0]), Path(args[1]), args[2])
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
