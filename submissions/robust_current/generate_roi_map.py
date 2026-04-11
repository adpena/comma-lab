#!/usr/bin/env python3
# ============================================================================
# LEGACY wrapper — core logic migrated to tac.roi_preprocessing.masks_to_roi_map
# ============================================================================
"""Generate SVT-AV1 ROI map file from Falcon Perception masks."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from tac.roi_preprocessing import masks_to_roi_map


def main() -> int:
    p = argparse.ArgumentParser(description="Generate SVT-AV1 ROI map from importance masks")
    p.add_argument("--masks", required=True, type=Path, help="Path to .npy mask file")
    p.add_argument("--output", required=True, type=Path, help="Output ROI map file")
    p.add_argument("--encode-w", type=int, default=524)
    p.add_argument("--encode-h", type=int, default=394)
    p.add_argument("--total-frames", type=int, default=1200)
    p.add_argument("--road-delta", type=int, default=-10)
    p.add_argument("--sky-delta", type=int, default=10)
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
    return 0


if __name__ == "__main__":
    sys.exit(main())
