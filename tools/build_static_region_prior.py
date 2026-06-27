#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build the byte-closed FROZEN-SOURCE static-region d_seg prior (compress-time producer).

NON-TRAINER helper. Reads the deterministic frozen-SegNet GT argmax over the 600 last-frames
of the contest source (0.mkv) and emits a byte-closed CAPACITY-ROUTING CLAMP prior the witness /
inflate path can consume WITHOUT loading any scorer at decode:

  * parametric bands  : sky (top rows < R_sky -> class C_sky) + hood (bottom rows >= R_hood ->
                        class C_hood). ~8 bytes. Clamps the camera-geometry-constant regions.
  * full static map   : single 384x512 5-class map (sentinel 255 = dynamic), brotli-q11.
                        ~2 KB. Clamps every temporally-constant pixel (72% of frame).

The clamp is a PRIOR / mask: the coordinate-INR spends 0 basis capacity on clamped (constant,
high-margin) pixels and routes it to the binding ~28% (the boundary annulus). This is COUNTED
rate (video-derived stored table) but measured cheap (~8 B parametric / ~2 KB full = 0.0013
score-units). NO scorer at decode (read the stored table, not the GT SegNet).

NO-FAKE / byte-closeability: the static partition IS video-derived (specific to this source).
It is byte-closed as STORED data (counted), NOT smuggled into inflate.py as "code". The
directional/annulus targeting that actually MOVES d_seg is NOT producible here — it is GT-derived
per-frame and byte-closes only via the task-space vehicle (oriented basis + stored 8-dim coords).

Evidence: [macOS-CPU advisory]; promotion_eligible=false; score_claim=false. $0, CPU, no MPS.

Design memo: .omx/research/frozen_source_0byte_dseg_priors_design_20260626.md
Measurement : .omx/research/frozen_source_0byte_dseg_priors_20260626.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "src", REPO_ROOT / "upstream"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

_FORBIDDEN_TMP = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")
_DEFAULT_ARGMAX = (
    "/Volumes/VertigoDataTier/pact/lever_b_score_native_argmax_smoke_20260610/"
    "targets_n600/gt_segnet_argmax.u8"
)
_SSD_TIERS = ("/Volumes/VertigoDataTier/pact", "/Volumes/APDataStore/pact")
N, H, W = 600, 384, 512
NPX = H * W


def _brotli_compress(b: bytes) -> bytes:
    import brotli  # noqa: PLC0415

    return brotli.compress(b, quality=11)


def _detect_band(static2d: np.ndarray, cls_frac: np.ndarray, *, from_bottom: bool,
                 thresh: float) -> tuple[int, int] | None:
    """Largest contiguous edge band whose rows are > thresh static AND > thresh single-class."""
    row_static = static2d.mean(axis=1)
    r0 = None
    rows = range(H - 1, -1, -1) if from_bottom else range(H)
    for r in rows:
        if row_static[r] > thresh and cls_frac[r] > thresh:
            r0 = r
        else:
            break
    if r0 is None:
        return None
    return (r0, H) if from_bottom else (0, r0 + 1)


def _resolve_out_dir(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    for tier in _SSD_TIERS:
        if Path(tier).is_dir():
            return Path(tier) / "static_region_prior_20260626"
    return REPO_ROOT / ".omx" / "state" / "static_region_prior_20260626"


def build(argmax_path: Path, out_dir: Path, *, band_thresh: float) -> dict[str, Any]:
    t0 = time.time()
    raw = np.fromfile(argmax_path, dtype=np.uint8)
    if raw.size != N * H * W:
        raise SystemExit(f"argmax size {raw.size} != {N * H * W} (expected {N}x{H}x{W})")
    am = raw.reshape(N, H, W)
    amf = am.reshape(N, NPX)

    static = amf.min(0) == amf.max(0)
    static2d = static.reshape(H, W)
    static_frac = float(static.mean())

    # per-class per-row temporal fraction (for band detection)
    sky_cls, hood_cls = 2, 4
    sky_frac_row = (am == sky_cls).mean(0).mean(1)
    hood_frac_row = (am == hood_cls).mean(0).mean(1)
    sky_band = _detect_band(static2d, sky_frac_row, from_bottom=False, thresh=band_thresh)
    hood_band = _detect_band(static2d, hood_frac_row, from_bottom=True, thresh=band_thresh)

    # full static map: class where static, 255 (sentinel) where dynamic
    static_class = np.full(NPX, 255, np.uint8)
    static_class[static] = amf[0][static]
    map_raw = static_class.tobytes()
    map_brotli = _brotli_compress(map_raw)

    # roundtrip verify (decompress -> identical)
    import brotli  # noqa: PLC0415

    assert brotli.decompress(map_brotli) == map_raw, "brotli roundtrip mismatch"

    out_dir.mkdir(parents=True, exist_ok=True)
    if any(s in str(out_dir.resolve()) for s in _FORBIDDEN_TMP):
        raise SystemExit(f"refusing transient out_dir: {out_dir}")

    # byte-closed prior blob: header (parametric bands) + brotli static map
    # layout: magic 'SRP1' | u16 R_sky u8 C_sky | u16 R_hood u8 C_hood | u32 map_len | map_brotli
    r_sky = sky_band[1] if sky_band else 0
    r_hood = hood_band[0] if hood_band else H
    blob = (
        b"SRP1"
        + struct.pack("<HB", r_sky, sky_cls if sky_band else 255)
        + struct.pack("<HB", r_hood, hood_cls if hood_band else 255)
        + struct.pack("<I", len(map_brotli))
        + map_brotli
    )
    blob_path = out_dir / "static_region_prior.srp1"
    blob_path.write_bytes(blob)

    parametric_only = (
        b"SRP1"
        + struct.pack("<HB", r_sky, sky_cls if sky_band else 255)
        + struct.pack("<HB", r_hood, hood_cls if hood_band else 255)
        + struct.pack("<I", 0)
    )
    parametric_px_frac = (
        ((r_sky * W) + ((H - r_hood) * W)) / NPX if (sky_band or hood_band) else 0.0
    )

    manifest = {
        "subagent": "build_static_region_prior_20260626",
        "utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "evidence_grade": "[macOS-CPU advisory]",
        "promotion_eligible": False,
        "score_claim": False,
        "promotable": False,
        "source_argmax": str(argmax_path),
        "source_video": "upstream/videos/0.mkv (comma2k19 RAV4 segment, frozen)",
        "n_frames": N,
        "h": H,
        "w": W,
        "static_px_frac": static_frac,
        "parametric_bands": {
            "sky": {"rows": [0, r_sky], "class": sky_cls} if sky_band else None,
            "hood": {"rows": [r_hood, H], "class": hood_cls} if hood_band else None,
            "px_frac": parametric_px_frac,
            "bytes": len(parametric_only),
            "byte_closeable": "COUNTED-trivial (stored row thresholds; no scorer at decode)",
        },
        "full_static_map": {
            "px_frac": static_frac,
            "raw_bytes": len(map_raw),
            "brotli_q11_bytes": len(map_brotli),
            "score_units_full_blob": round(25 * len(blob) / 37_545_489, 6),
            "byte_closeable": "COUNTED-cheap (stored 384x512 table; no scorer at decode)",
        },
        "blob": {
            "path": str(blob_path),
            "bytes": len(blob),
            "sha256": hashlib.sha256(blob).hexdigest(),
            "format": "SRP1: magic|u16 R_sky,u8 C_sky|u16 R_hood,u8 C_hood|u32 map_len|brotli(map)",
        },
        "consumer_contract": (
            "inflate/witness reads SRP1, clamps rows<R_sky->C_sky and rows>=R_hood->C_hood; "
            "optionally brotli-decompress the 384x512 map and clamp static px (!=255). Use as a "
            "capacity-routing MASK: 0 INR basis on clamped px, all on the dynamic ~28% annulus."
        ),
        "NOT_producible_here": (
            "directional/annulus targeting (the -48% d_seg lever) is GT-derived per-frame; it "
            "byte-closes only via the task-space vehicle (oriented basis + stored 8-dim coords)."
        ),
        "disk_hygiene": {
            "rebuildable": True,
            "rebuild_command": (
                f".venv/bin/python tools/build_static_region_prior.py "
                f"--argmax {argmax_path} --out-dir {out_dir}"
            ),
            "reason_rebuildable": "deterministic function of the frozen GT SegNet argmax cache",
            "bytes_on_disk": len(blob),
        },
        "elapsed_sec": round(time.time() - t0, 1),
    }
    (out_dir / "static_region_prior_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--argmax", default=_DEFAULT_ARGMAX, help="path to gt_segnet_argmax.u8")
    ap.add_argument("--out-dir", default=None, help="output dir (default: SSD tier)")
    ap.add_argument("--band-thresh", type=float, default=0.99,
                    help="per-row static+single-class fraction to clamp a band (default 0.99)")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    argmax_path = Path(args.argmax)
    if not argmax_path.is_file():
        raise SystemExit(
            f"argmax cache not found: {argmax_path}\n"
            "rebuild it with tools/lever_b_build_score_native_targets.py (frozen SegNet on 0.mkv)."
        )
    manifest = build(argmax_path, _resolve_out_dir(args.out_dir), band_thresh=args.band_thresh)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
