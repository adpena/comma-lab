#!/usr/bin/env python3
"""Generate SVT-AV1 ROI map file from Falcon Perception masks.

Converts per-frame importance masks (float32, 0-1) to QP offset maps
for SvtAv1EncApp's --roi-map-file parameter.

The ROI map is purely encoder-side — it doesn't modify pixels,
so PoseNet cannot be damaged.

Format: one line per frame change.
Each line: frame_number offset0 offset1 ... offset62
where there are ceil(W/64) * ceil(H/64) = 9*7 = 63 values.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np


def masks_to_roi_map(
    masks: np.ndarray,
    encode_w: int,
    encode_h: int,
    total_frames: int,
    road_qp_delta: int = -10,
    sky_qp_delta: int = 10,
    neutral_qp_delta: int = 0,
    threshold_high: float = 0.6,
    threshold_low: float = 0.3,
) -> list[str]:
    """Convert importance masks to ROI map lines.

    Args:
        masks: (N, H, W) float32 importance masks (1=important, 0=unimportant)
        encode_w: encode resolution width (e.g., 524)
        encode_h: encode resolution height (e.g., 394)
        total_frames: total video frames
        road_qp_delta: QP offset for important regions (negative = more bits)
        sky_qp_delta: QP offset for unimportant regions (positive = fewer bits)
        neutral_qp_delta: QP offset for neutral regions
        threshold_high: mask value above which region is "important"
        threshold_low: mask value below which region is "unimportant"

    Returns:
        List of ROI map lines in SVT-AV1 format.
    """
    cols = math.ceil(encode_w / 64)
    rows = math.ceil(encode_h / 64)
    n_blocks = cols * rows

    # Scale masks to encode resolution
    mask_h, mask_w = masks.shape[1], masks.shape[2]
    scale_y = mask_h / encode_h
    scale_x = mask_w / encode_w

    lines = []
    prev_offsets = None

    for frame_idx in range(total_frames):
        # Interpolate mask for this frame
        if masks.shape[0] >= total_frames:
            mask = masks[frame_idx]
        else:
            ratio = frame_idx / max(total_frames - 1, 1)
            ml_idx = ratio * (masks.shape[0] - 1)
            lo = int(ml_idx)
            hi = min(lo + 1, masks.shape[0] - 1)
            t = ml_idx - lo
            mask = masks[lo] * (1.0 - t) + masks[hi] * t

        # Compute per-block average importance
        offsets = []
        for by in range(rows):
            for bx in range(cols):
                # Block bounds in mask coordinates
                y0 = int(by * 64 * scale_y)
                y1 = int(min((by + 1) * 64, encode_h) * scale_y)
                x0 = int(bx * 64 * scale_x)
                x1 = int(min((bx + 1) * 64, encode_w) * scale_x)

                block_mean = mask[y0:y1, x0:x1].mean()

                if block_mean >= threshold_high:
                    offsets.append(road_qp_delta)
                elif block_mean <= threshold_low:
                    offsets.append(sky_qp_delta)
                else:
                    offsets.append(neutral_qp_delta)

        # Only emit a line if offsets changed
        if offsets != prev_offsets:
            offset_str = " ".join(str(o) for o in offsets)
            lines.append(f"{frame_idx} {offset_str}")
            prev_offsets = offsets

    return lines


def main() -> int:
    p = argparse.ArgumentParser(description="Generate SVT-AV1 ROI map from importance masks")
    p.add_argument("--masks", required=True, type=Path, help="Path to .npy mask file")
    p.add_argument("--output", required=True, type=Path, help="Output ROI map file")
    p.add_argument("--encode-w", type=int, default=524)
    p.add_argument("--encode-h", type=int, default=394)
    p.add_argument("--total-frames", type=int, default=1200)
    p.add_argument("--road-delta", type=int, default=-10, help="QP delta for important regions (negative=more bits)")
    p.add_argument("--sky-delta", type=int, default=10, help="QP delta for unimportant regions (positive=fewer bits)")
    p.add_argument("--neutral-delta", type=int, default=0)
    p.add_argument("--threshold-high", type=float, default=0.6)
    p.add_argument("--threshold-low", type=float, default=0.3)
    args = p.parse_args()

    print(f"Loading masks from {args.masks}", file=sys.stderr)
    masks = np.load(str(args.masks))
    print(f"  Shape: {masks.shape}", file=sys.stderr)

    lines = masks_to_roi_map(
        masks, args.encode_w, args.encode_h, args.total_frames,
        road_qp_delta=args.road_delta, sky_qp_delta=args.sky_delta,
        neutral_qp_delta=args.neutral_delta,
        threshold_high=args.threshold_high, threshold_low=args.threshold_low,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} ROI map events to {args.output}", file=sys.stderr)

    # Print summary
    if lines:
        sample = lines[0].split()
        n_blocks = len(sample) - 1
        offsets = [int(x) for x in sample[1:]]
        road_pct = sum(1 for o in offsets if o < 0) / n_blocks * 100
        sky_pct = sum(1 for o in offsets if o > 0) / n_blocks * 100
        neutral_pct = sum(1 for o in offsets if o == 0) / n_blocks * 100
        print(f"  Blocks: {n_blocks} ({args.encode_w}x{args.encode_h} @ 64x64)", file=sys.stderr)
        print(f"  Frame 0: road={road_pct:.0f}% neutral={neutral_pct:.0f}% sky={sky_pct:.0f}%", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
