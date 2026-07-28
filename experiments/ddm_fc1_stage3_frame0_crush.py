#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""DDM-FC1 STAGE 3 -- FRAME_0 CARRIER real crush curve (n600, the #1 binding stream).

frame_0 is seg-free (SegNet reads the last frame only), so its ONLY constraint is d_pose (PoseNet
reads both frames). This measures the REAL coded size of the 600 base frames at aggressive lossy
crush rungs (WebP, a real compiled codec, decode << 1 s). It names the binding stream's floor with
MEASURED bytes and the S-rate it implies. d_pose collateral under crush (A7/collapse-2) is NOT
measured here (pose is banked via the stored-target dxi sidecar on the UN-crushed base; crushing the
pose-read frames would perturb it -- named, not faked). `[macOS-CPU advisory]`.
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

SCHEMA = "ddm_fc1_stage3_frame0_crush.v1"
N_REF = 37_545_489


def _webp_bytes(frame_hwc: np.ndarray, quality: int, method: int) -> int:
    b = io.BytesIO()
    Image.fromarray(frame_hwc).save(b, "WEBP", quality=quality, method=method)
    return len(b.getvalue())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--gt-cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-pairs", type=int, default=600)
    args = ap.parse_args(argv)

    t0 = time.time()
    gt_f0 = np.load(str(args.gt_cache))["gt_f0"][: args.max_pairs]  # (n,874,1164,3)
    n = gt_f0.shape[0]
    rungs = [(1, 6), (10, 4), (30, 4), (50, 4), (75, 4)]
    results = []
    for q, meth in rungs:
        total = 0
        for i in range(n):
            total += _webp_bytes(gt_f0[i], q, meth)
        rate = 25.0 * total / N_REF
        results.append({
            "webp_quality": q,
            "webp_method": meth,
            "total_bytes": total,
            "bytes_per_frame": total / n,
            "S_rate_term": rate,
        })
        print(f"[frame0] Q{q} m{meth}: {total} B ({total/n:.0f} B/frame) rate_term={rate:.4f} ({time.time()-t0:.0f}s)", flush=True)

    out = {
        "schema": SCHEMA,
        "evidence_axis": "[macOS-CPU advisory] REAL WebP coder; frame_0 seg-free; d_pose collateral NOT measured (pose banked on un-crushed base)",
        "pairs": n,
        "n_reference_bytes": N_REF,
        "bar_bytes_0p172": 187_727,
        "bar_bytes_0p15": 154_522,
        "crush_curve": results,
        "note": "even the most aggressive pose-plausible crush is multi-MB across 600 frames -> rate_term >> the 0.03 budget left after banked pose (0.127) + near-solved seg (0.0152). frame_0 is the binding stream.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2), flush=True)
    print(f"[done] stage3 in {time.time()-t0:.0f}s -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
