#!/usr/bin/env python
"""ddm_js2 instrument control — local CPU SegNet forward on CUDA-custody tensors.

Purpose (js2 fire-order 1, charter law 1 "instrument validation FIRST"):
the js2 arm measured 44.13% flip disagreement between the LOCAL decode of the
CUDA-locked cp135 receiver and the promoted T4 row (rendered-raw hashes differ),
so every locally-scored seg proposal was inadmissible. This control removes the
decode axis entirely: it consumes the scorer-input tensors the REAL T4 lane
persisted to the Modal Volume (segnet_last_rgb.npy — the exact 384x512 float
planes SegNet consumed on the shipping axis, written by
tac.local_acceleration.mlx_preprocess.write_scorer_input_cache_from_raw_file),
runs the frozen CPU-torch SegNet on them via the canonical upstream distortion
net, and compares total argmax flips against GT with the promoted scalar.

Admission gate (from the arm's fire trigger): flip disagreement <= 1% of the
promoted flip count. Remaining divergence after this control is FORWARD-
instrument only (CPU-vs-CUDA SegNet drift, historically ~1.5e-5 scale).

Payload law (P0 DEF CON 1000): the custody .npy files persist on the SSD tier
and the Modal Volume with sha256 receipts; the argmax field this control
computes IS persisted (lstars_local_on_custody.npy) with its own sha receipt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

PROMOTED_D_SEG = 0.00029643  # cp135 floor row, contest-CUDA T4 n600 (sha 6eb1a3b7...)
PIXELS_PER_PAIR = 512 * 384
GATE_FRACTION = 0.01


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(
            "/Volumes/VertigoDataTier/pact/ddm_js2_20260812/"
            "instrument_validation_cuda/scorer_input_cache_tensors"
        ),
    )
    ap.add_argument(
        "--gt-cache",
        type=Path,
        default=Path("experiments/results/mlx_fleet_gt_cache/gt_n600.npz"),
    )
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out-json", type=Path, default=None)
    args = ap.parse_args()

    import torch

    torch.set_num_threads(int(args.threads))

    repo_root = Path(__file__).resolve().parents[1]
    from tac.local_acceleration.mlx_scorer_response import (
        _load_upstream_distortion_net,
        _resolve_upstream_dir,
        load_scorer_input_cache,
    )

    # Full integrity re-hash: this run IS the custody verification.
    cache = load_scorer_input_cache(args.cache_dir)
    seg = cache.segnet_last_rgb  # (P, 3, 384, 512) fp32, exact SegNet input planes
    n_pairs = int(seg.shape[0])

    gt = np.load(args.gt_cache)
    lstars = gt["lstars"]  # (600, 384, 512) GT argmax, CPU-torch lineage
    if lstars.shape[0] != n_pairs:
        raise SystemExit(
            f"FATAL pair-count mismatch: custody {n_pairs} vs GT {lstars.shape[0]}"
        )

    dist = _load_upstream_distortion_net(_resolve_upstream_dir(repo_root))

    flips_total = 0
    local_argmax = np.empty(
        (n_pairs, lstars.shape[1], lstars.shape[2]), dtype=np.uint8
    )
    t0 = time.time()
    with torch.inference_mode():
        for lo in range(0, n_pairs, max(1, int(args.batch))):
            hi = min(n_pairs, lo + max(1, int(args.batch)))
            x = torch.from_numpy(np.asarray(seg[lo:hi], dtype=np.float32).copy())
            logits = dist.segnet(x).float().numpy()
            pred = logits.argmax(axis=1).astype(np.uint8)
            local_argmax[lo:hi] = pred
            flips_total += int((pred != lstars[lo:hi]).sum())

    d_seg_local = flips_total / (n_pairs * PIXELS_PER_PAIR)
    promoted_flips = round(PROMOTED_D_SEG * n_pairs * PIXELS_PER_PAIR)
    disagreement = abs(flips_total - promoted_flips) / max(promoted_flips, 1)
    verdict = (
        "INSTRUMENT_VALIDATED" if disagreement <= GATE_FRACTION else "STILL_BLOCKED"
    )

    out_dir = args.cache_dir.parent
    argmax_path = out_dir / "lstars_local_on_custody.npy"
    np.save(argmax_path, local_argmax)

    result = {
        "schema": "ddm_js2_instrument_control_v1",
        "verdict": verdict,
        "flips_local_on_custody": flips_total,
        "flips_promoted_scalar_derived": promoted_flips,
        "disagreement_fraction": disagreement,
        "gate_fraction": GATE_FRACTION,
        "d_seg_local_on_custody": d_seg_local,
        "d_seg_promoted": PROMOTED_D_SEG,
        "n_pairs": n_pairs,
        "batch": int(args.batch),
        "cache_manifest_source": str(cache.manifest.get("source", "")),
        "cache_archive_sha256": str(cache.manifest.get("archive_sha256", "")),
        "local_argmax_field": {
            "path": str(argmax_path),
            "sha256": sha256_file(argmax_path),
            "bytes": argmax_path.stat().st_size,
        },
        "elapsed_seconds": time.time() - t0,
        "axis_note": (
            "[macOS-CPU forward on contest-CUDA-custody inputs] — control only; "
            "residual divergence is forward-instrument drift"
        ),
        "score_claim": False,
        "promotable": False,
    }
    out_json = args.out_json or (out_dir / "INSTRUMENT_VALIDATION_CUDA_CUSTODY.json")
    out_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "verdict",
                    "flips_local_on_custody",
                    "flips_promoted_scalar_derived",
                    "disagreement_fraction",
                    "d_seg_local_on_custody",
                    "elapsed_seconds",
                )
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
