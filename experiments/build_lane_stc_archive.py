#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Lane STC archive builder — boundary-coded mask payload.

Takes a Lane A or Lane G anchor archive (renderer.bin + masks.mkv +
optimized_poses.pt), decodes the AV1 mask video back to argmax class IDs,
re-encodes the class IDs through ``encode_mask_video_stc`` (the
arithmetic-coded boundary codec from ``tac.stc_boundary_codec``), and
writes a deterministic zip with the new mask payload.

The renderer + poses are unchanged. The hypothesis (per
``docs/paper/lane_stc_boundary_coding_design_20260429.md``):
boundary-focused class-id coding cuts the mask payload from
~200 KB (AV1 monochrome) to 120-140 KB (rate -0.04 to -0.05).

Output archive layout:
    archive_lane_stc.zip:
        renderer.bin                 (unchanged)
        masks.stcb                   (new STC payload)
        optimized_poses.pt OR poses.pt (unchanged, whichever exists)

Usage:
    python experiments/build_lane_stc_archive.py \\
        --anchor-archive experiments/results/lane_a_landed/archive_lane_a.zip \\
        --output experiments/results/lane_stc/archive_lane_stc.zip \\
        --boundary-fraction 0.05
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import zipfile
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (_REPO_ROOT / "src",):
    if _p.is_dir() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from tac.mask_codec import decode_masks  # noqa: E402  (legacy AV1 -> class IDs)
from tac.repo_io import json_text  # noqa: E402
from tac.stc_boundary_codec import (  # noqa: E402
    decode_mask_video_stc,
    encode_mask_video_stc,
)
from tac.submission_archive import safe_extract_zip  # noqa: E402


def _det_write(zout: zipfile.ZipFile, src: Path, arcname: str) -> None:
    """Deterministic zip write (fixed mtime, ZIP_DEFLATED, level 9).

    Required by Codex R5-r6 #5 / check_archive_builders_use_deterministic_zip.
    """
    info = zipfile.ZipInfo(arcname)
    info.date_time = (1980, 1, 1, 0, 0, 0)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    zout.writestr(info, src.read_bytes(), compresslevel=9)


def build_lane_stc_archive(
    anchor_archive: Path,
    output: Path,
    boundary_fraction: float = 0.05,
    keep_legacy_masks: bool = False,
) -> dict:
    """Re-encode anchor masks.mkv -> masks.stcb (Lane STC).

    Returns:
        dict with byte-size statistics (anchor_masks_bytes, stc_bytes,
        output_bytes, rate_delta_bytes, predicted_score_delta).
    """
    anchor_archive = Path(anchor_archive)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        safe_extract_zip(anchor_archive, td_path)

        masks_mkv = td_path / "masks.mkv"
        if not masks_mkv.exists():
            raise FileNotFoundError(
                f"anchor archive missing masks.mkv: {anchor_archive}"
            )
        anchor_masks_bytes = masks_mkv.stat().st_size

        # Decode AV1 -> class IDs.
        masks = decode_masks(masks_mkv)
        n, h, w = masks.shape
        print(
            f"[lane-stc] decoded {n} mask frames @ {h}x{w} "
            f"from {anchor_masks_bytes:,} B AV1"
        )

        # Encode with STC.
        stcb_path = td_path / "masks.stcb"
        stc_bytes = encode_mask_video_stc(
            masks,
            stcb_path,
            boundary_fraction=boundary_fraction,
            verify_roundtrip=True,
        )
        print(
            f"[lane-stc] STC encoded to {stc_bytes:,} B "
            f"(boundary_fraction={boundary_fraction})"
        )

        # Sanity: independent decode-and-compare.
        decoded = decode_mask_video_stc(stcb_path)
        if not torch.equal(decoded, masks):
            raise RuntimeError(
                "STC archive build: independent re-decode disagreed with input"
            )

        renderer = td_path / "renderer.bin"
        if not renderer.exists():
            raise FileNotFoundError("anchor archive missing renderer.bin")

        with zipfile.ZipFile(
            output, "w", zipfile.ZIP_DEFLATED, compresslevel=9
        ) as zout:
            _det_write(zout, renderer, "renderer.bin")
            _det_write(zout, stcb_path, "masks.stcb")
            if keep_legacy_masks:
                _det_write(zout, masks_mkv, "masks.mkv")
            for poses_name in ("optimized_poses.pt", "poses.pt"):
                p = td_path / poses_name
                if p.exists():
                    _det_write(zout, p, poses_name)
                    break

    output_bytes = output.stat().st_size
    rate_delta_bytes = stc_bytes - anchor_masks_bytes
    # 25 * delta_bytes / 37545489 (per-byte score weight in the contest).
    predicted_score_delta = 25 * rate_delta_bytes / 37_545_489

    return {
        "anchor_masks_bytes": int(anchor_masks_bytes),
        "stc_bytes": int(stc_bytes),
        "rate_delta_bytes": int(rate_delta_bytes),
        "output_bytes": int(output_bytes),
        "boundary_fraction": float(boundary_fraction),
        "predicted_score_delta": float(predicted_score_delta),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--anchor-archive",
        type=Path,
        required=True,
        help="Path to a Lane A or Lane G archive (renderer.bin + masks.mkv).",
    )
    p.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the Lane STC archive.zip.",
    )
    p.add_argument(
        "--boundary-fraction",
        type=float,
        default=0.05,
        help="Per-frame fraction of pixels marked as boundary (default 0.05).",
    )
    p.add_argument(
        "--keep-legacy-masks",
        action="store_true",
        help="Also include masks.mkv in the output (A/B testing).",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Optional path to write a JSON manifest with byte stats.",
    )
    args = p.parse_args(argv)

    info = build_lane_stc_archive(
        anchor_archive=args.anchor_archive,
        output=args.output,
        boundary_fraction=args.boundary_fraction,
        keep_legacy_masks=args.keep_legacy_masks,
    )
    print(
        f"[lane-stc] anchor masks.mkv={info['anchor_masks_bytes']:,}B"
        f"  masks.stcb={info['stc_bytes']:,}B"
        f"  delta={info['rate_delta_bytes']:+,}B"
        f"  output_archive={info['output_bytes']:,}B"
        f"  predicted_score_delta={info['predicted_score_delta']:+.4f}"
    )
    if args.manifest is not None:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json_text(info), encoding="utf-8")
        print(f"[lane-stc] manifest -> {args.manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
