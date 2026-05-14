# SPDX-License-Identifier: MIT
"""Build a diet variant of an existing submission archive.

Usage:
    python experiments/build_diet_archive.py \\
        --input experiments/results/lane_a_landed/archive_lane_a.zip \\
        --output experiments/results/lane_a_landed/archive_lane_a_diet.zip \\
        --techniques pose_delta,mkv_passthrough

Techniques (comma-separated):
    pose_delta            Re-encode optimized_poses.pt via Lane PD codec.
    mkv_passthrough       Store masks.mkv with ZIP_STORED instead of DEFLATE.
    arithmetic_renderer   If renderer.bin is a Selfcomp tar.xz payload,
                          repack to Lane SH SHv1 (no-op for ASYM/FP4A/etc.).
    zip_recompress        Use ZIP_LZMA for member compression.

The diet output is deterministic (fixed timestamp, fixed compresslevel).
Always verify the result with :func:`tac.archive_diet.verify_diet_archive`
before submitting.

CLAUDE.md compliance: this script is encoder-only, never loads scorers,
and never modifies the upstream pinned snapshot.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.archive_diet import diet_archive, verify_diet_archive  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--techniques",
        default="pose_delta,mkv_passthrough",
        help="Comma-separated technique names. Default: pose_delta,mkv_passthrough.",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip the post-build verification step (not recommended).",
    )
    parser.add_argument(
        "--pose-dim",
        type=int,
        default=6,
        help="Pose dimensionality for verification decode (default: 6).",
    )
    args = parser.parse_args()

    techniques = [t.strip() for t in args.techniques.split(",") if t.strip()]
    if not args.input.is_file():
        print(f"input archive not found: {args.input}", file=sys.stderr)
        return 1

    result = diet_archive(args.input, args.output, techniques)
    summary = result.as_dict()
    if not args.skip_verify:
        summary["verify"] = verify_diet_archive(
            args.input, args.output, pose_dim=args.pose_dim
        )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if (args.skip_verify or summary["verify"]["ok"]) else 2


if __name__ == "__main__":
    sys.exit(main())
