#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""#401 BLIND-COORDINATE proof artifact — n600 bit-identity-through-R + real byte delta.

Produces the load-bearing PROOF for the blind-coordinate rate lever:

  1. blind-mask derivation report (exact geometry: 230,904 blind px/frame; 768x1024 retained).
  2. n600 BIT-IDENTITY THROUGH R: over ALL 600 gt pairs, fill the blind set with ARBITRARY
     content in BOTH frames and assert the real torch scorer-input tensors (posenet_in +
     segnet_in) are bit-for-bit identical -> d_seg/d_pose provably unchanged. n600 or it is
     not evidence (CLAUDE.md "ALLERGIC to non-n600-scale").
  3. real byte delta: lossless byte-close (brotli-q11) of the FULL camera frame vs the
     retained sub-grid, averaged over a sample of REAL gt camera frames.
  4. (optional) copies a real finished archive.zip in for the "real byte-closed archive"
     baseline context (with an explicit scope caveat: a pure-generator witness archive
     stores no camera pixels, so the direct saving applies to camera-res-storing sections).

CONTAINMENT: numpy resizes + torch preprocess (CPU-light — NO scorer forward, NO Modal).
Serial. Runs while the live run owns the GPU. All numbers ``[macOS-CPU advisory /
derivation]``; only the bit-identity claim is a PROOF (and only after n600 passes).

Usage:
  .venv/bin/python tools/blind_coordinate_proof.py \
      --gt-cache experiments/results/mlx_fleet_gt_cache/gt_n600.npz \
      --out-dir experiments/results/blind_coord_401_<utc> \
      --byte-delta-frames 32 --fill-mode random \
      [--copy-archive experiments/results/<finished>/archive.zip]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

from tac.through_r.blind_coordinate import (  # noqa: E402
    BLIND_COORD_LABEL,
    bit_identity_report,
    blind_fraction,
    build_blind_mask,
    measure_byte_delta,
)


def _git_sha() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gt-cache", default="experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-pairs", type=int, default=0, help="0 => all n600 (n600 or not evidence)")
    ap.add_argument("--byte-delta-frames", type=int, default=32)
    ap.add_argument("--fill-mode", default="random", choices=["random", "max", "zero"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--copy-archive", default=None,
                    help="a real finished archive.zip to copy in for byte-closed baseline context")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    gt = np.load(args.gt_cache)
    f0 = gt["gt_f0"]
    f1 = gt["gt_f1"]
    n_all = int(f0.shape[0])
    n = n_all if args.max_pairs in (0, None) else min(int(args.max_pairs), n_all)

    bm = build_blind_mask()
    frac = blind_fraction()
    print(f"[proof] blind mask: {frac['n_blind_px']} blind px/frame "
          f"({100*frac['blind_fraction']:.4f}%), retained {frac['retained_subgrid_hw']}", flush=True)

    print(f"[proof] n600 bit-identity-through-R over {n} pairs (fill={args.fill_mode}) ...", flush=True)
    bit = bit_identity_report(f0[:n], f1[:n], seed=args.seed, bm=bm, fill_mode=args.fill_mode)
    print(f"[proof]   -> all_bit_identical={bit.all_bit_identical} "
          f"max|Δpose|={bit.max_abs_diff_pose} max|Δseg|={bit.max_abs_diff_seg} "
          f"failures={bit.n_failures}", flush=True)

    nbd = min(int(args.byte_delta_frames), n_all)
    print(f"[proof] byte delta over {nbd} real gt frames ...", flush=True)
    bd = measure_byte_delta(f1[:nbd], bm=bm)
    print(f"[proof]   -> codec={bd.codec} full={bd.bytes_full_mean:.0f}B "
          f"retained={bd.bytes_retained_mean:.0f}B delta={bd.byte_delta_mean:.0f}B "
          f"({100*bd.delta_fraction_mean:.2f}%)", flush=True)

    archive_ctx: dict = {}
    if args.copy_archive:
        src = Path(args.copy_archive)
        if src.exists():
            dst = out / f"context_{src.parent.name}__archive.zip"
            shutil.copy2(src, dst)
            archive_ctx = {
                "copied_from": str(src),
                "copied_to": str(dst),
                "archive_zip_bytes": int(dst.stat().st_size),
                "scope_caveat": (
                    "This finished archive is the byte-closed baseline CONTEXT. A pure-generator "
                    "witness archive stores no camera pixels, so the blind-coordinate direct saving "
                    "is 0 until it carries a camera-res residual/sidecar section; the measured "
                    "delta_fraction is the saving AVAILABLE to any camera-res-storing representation."
                ),
            }
        else:
            archive_ctx = {"copied_from": str(src), "error": "archive not found"}

    artifact = {
        "schema": "blind_coordinate_proof.v1",
        "task": "#401",
        "utc": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
        "git_sha": _git_sha(),
        "gt_cache": args.gt_cache,
        "label": BLIND_COORD_LABEL,
        "blind_fraction": frac,
        "bit_identity_through_R": bit.to_json_dict(),
        "byte_delta": bd.to_json_dict(),
        "real_archive_context": archive_ctx,
        "elapsed_seconds": round(time.time() - t0, 1),
        "authority_note": (
            "blind_fraction + bit_identity are EXACT PROOFs (deterministic kernel property + "
            "torch-verified n600). byte_delta is a real lossless byte-close of real camera-res "
            "video content, [macOS-CPU advisory / derivation] NON-PROMOTABLE. Pointer UNMOVED."
        ),
    }
    art_path = out / "blind_coordinate_proof.json"
    art_path.write_text(json.dumps(artifact, indent=2))
    print(f"[proof] wrote {art_path} ({artifact['elapsed_seconds']}s)", flush=True)
    if not bit.all_bit_identical:
        print("[proof] WARNING: bit-identity FAILED on some pairs — lever NOT proven!", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
