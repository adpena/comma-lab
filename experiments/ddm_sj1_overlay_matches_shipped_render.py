#!/usr/bin/env python3
"""Prove the pose leg was measured on the frames the ARCHIVE actually decodes to.

ddm_sj1 measures d_pose on an OVERLAY -- the base decode with the candidate's odd frames
spliced in -- because copying 3.66 GB per variant to score a few hundred changed frames is
not affordable.  The overlay renders those frames from the token field through the render
tree; the SHIPPED candidate renders them inside the receiver, from the tokens it decodes.
Those are two different code paths to the same claim, and the pose number is only a claim
about the shipped object if they produce the same pixels.

Pass 7 asserted that by construction and never measured it.  This module measures it: for
every ADMITTED pair, the overlay's odd frame against frame ``2p+1`` of the candidate's own
parse-back ``0.raw``.  Byte-identical or the pose leg is describing a different object.

MEASURED on pass 7's landed candidate before this module was used on pass 8: 42 of 42
admitted pairs byte-identical, zero differing.

Axis ``[macOS-CPU advisory / scorer-free EXACT byte comparison]``.  No scorer runs.  No
score is claimed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

CAMERA_H, CAMERA_W = 874, 1164
N_PAIRS = 600


class OverlayIdentityError(RuntimeError):
    """The overlay and the shipped render are not the same pixels."""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlay-manifest", type=Path, required=True,
                        help="the OVERLAY.json written beside the odd-frame memmap")
    parser.add_argument("--parseback-raw", type=Path, required=True,
                        help="the candidate's own parse-back 0.raw")
    parser.add_argument("--admitted-pairs", type=Path, required=True,
                        help="kept_pairs.json -- only the pairs the candidate actually ships")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    overlay = json.loads(args.overlay_manifest.read_text())
    pairs = [int(p) for p in overlay["pairs"]]
    slot = {pair: index for index, pair in enumerate(pairs)}
    odd = np.memmap(overlay["odd_frames_path"], dtype=np.uint8, mode="r",
                    shape=(len(pairs), CAMERA_H, CAMERA_W, 3))
    raw_bytes = args.parseback_raw.stat().st_size
    expected = 2 * N_PAIRS * CAMERA_H * CAMERA_W * 3
    if raw_bytes != expected:
        raise OverlayIdentityError(f"parse-back raw is {raw_bytes} B, not {expected}")
    raw = np.memmap(args.parseback_raw, dtype=np.uint8, mode="r",
                    shape=(2 * N_PAIRS, CAMERA_H, CAMERA_W, 3))

    admitted = sorted(int(p) for p in json.loads(args.admitted_pairs.read_text()))
    missing = [p for p in admitted if p not in slot]
    if missing:
        raise OverlayIdentityError(
            f"{len(missing)} admitted pairs are absent from the overlay (first: {missing[:5]})"
        )

    started = time.perf_counter()
    differing: list[dict] = []
    for pair in admitted:
        left = np.asarray(odd[slot[pair]])
        right = np.asarray(raw[2 * pair + 1])
        if not np.array_equal(left, right):
            delta = np.abs(left.astype(np.int16) - right.astype(np.int16))
            differing.append({"pair": pair, "max_abs_delta": int(delta.max()),
                              "cells_differing": int((delta > 0).sum())})

    document = {
        "schema": "ddm_sj1_overlay_matches_shipped_render.v1",
        "written_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "axis": "[macOS-CPU advisory / scorer-free EXACT byte comparison]",
        "question": ("were the frames the pose leg scored the SAME PIXELS the archive decodes "
                     "to, or only the same field rendered by a second code path?"),
        "overlay_manifest": str(args.overlay_manifest),
        "overlay_odd_frames_sha256": overlay.get("odd_frames_sha256"),
        "parseback_raw": str(args.parseback_raw),
        "admitted_pairs": len(admitted),
        "byte_identical_pairs": len(admitted) - len(differing),
        "differing_pairs": differing,
        "verdict": "PASS" if not differing else "FAIL",
        "consequence_if_it_fails": (
            "the d_pose the admission spent its multiplier on was measured on frames the "
            "candidate does not ship; the leg would have to be re-measured on the parse-back "
            "decode before any projection is quoted"
        ),
        "seconds": time.perf_counter() - started,
        "score_claim": False,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2, sort_keys=True))
    print(json.dumps({k: v for k, v in document.items()
                      if k not in ("consequence_if_it_fails", "question")}, indent=2, sort_keys=True))
    if differing:
        raise OverlayIdentityError(
            f"{len(differing)} admitted pairs differ between the overlay and the shipped render"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
