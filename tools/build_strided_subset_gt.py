#!/usr/bin/env python3
"""Build a STRIDED subset + its complement (held-out) GT cache from a full fleet GT npz.

AMORTIZATION pivot (operator 2026-06-27, DAG FEED-ep). The level-set witness factors into a
SHARED decoder + per-(pair,frame) codes. To amortize the days-long n600 joint fit into hours we
train a decoder on a STRIDED subset of the 600 pairs (strided = every Nth pair so it SPANS the
whole continuous comma2k19 drive -> scene diversity, the prerequisite for the frozen decoder to
generalize via the code channel), then freeze it and fit only the codes.

This tool slices a full GT cache (the EXACT outputs of precompute_gt; schema:
n_pairs, gt_f0, gt_f1, lstars, margins, gt_poses) into TWO byte-faithful sub-caches with the
identical schema/dtypes the trainer's ``load_gt_from_cache`` expects:

  * SUBSET  = pairs at indices [0, stride, 2*stride, ...]  (the decoder-training set)
  * HELDOUT = the complement (pairs the decoder NEVER trains on; only their codes are later fit)
    -> the held-out d_seg of the frozen subset decoder is the decisive amortization gate.

NO surrogate / NO downsample: the sliced arrays are exact copies of the cached frozen-CPU-torch
authority. Memory-bounded: each npz member is materialized ONCE (NpzFile re-inflates per index).
Writes a provenance manifest (src sha256/bytes, stride, the exact index lists) next to the outputs
per the AGENTS.md provenance non-negotiable. Durable paths only (NO /tmp).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

_KEYS = ("gt_f0", "gt_f1", "lstars", "margins", "gt_poses")


def _sha256_head(path: Path, cap_bytes: int = 256 * 1024 * 1024) -> str:
    """sha256 over the first cap_bytes (full hash of a ~5GB npz is slow; head-hash is a stable
    provenance fingerprint sufficient to detect a different source cache)."""
    h = hashlib.sha256()
    read = 0
    with open(path, "rb") as f:
        while read < cap_bytes:
            chunk = f.read(min(8 * 1024 * 1024, cap_bytes - read))
            if not chunk:
                break
            h.update(chunk)
            read += len(chunk)
    return f"sha256-head{cap_bytes}:{h.hexdigest()}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", type=Path,
                    default=Path("experiments/results/mlx_fleet_gt_cache/gt_n600.npz"),
                    help="full fleet GT npz to slice.")
    ap.add_argument("--stride", type=int, default=3,
                    help="subset = pairs at indices [0::stride] (every Nth pair, spans the drive).")
    ap.add_argument("--out-subset", type=Path, default=None,
                    help="output strided-subset npz (default: gt_strided_n<K>.npz next to --src).")
    ap.add_argument("--out-heldout", type=Path, default=None,
                    help="output held-out (complement) npz (default: gt_heldout_n<M>.npz next to --src).")
    ap.add_argument("--no-heldout", action="store_true",
                    help="only write the subset (skip the complement).")
    args = ap.parse_args(argv)

    src = Path(args.src)
    if not src.exists():
        raise FileNotFoundError(f"--src {src} does not exist (NO-FAKE: refusing to fabricate GT).")
    if "/tmp/" in str(src) or str(src).startswith("/tmp"):
        raise ValueError("refusing a /tmp source path (durable evidence non-negotiable).")

    t0 = time.time()
    z = np.load(src, allow_pickle=False)
    cached_P = int(z["n_pairs"])
    idx_subset = np.arange(0, cached_P, max(1, int(args.stride)), dtype=np.int64)
    mask = np.ones(cached_P, dtype=bool)
    mask[idx_subset] = False
    idx_heldout = np.nonzero(mask)[0].astype(np.int64)
    K = int(idx_subset.size)
    M = int(idx_heldout.size)

    out_subset = Path(args.out_subset) if args.out_subset else src.parent / f"gt_strided_n{K}.npz"
    out_heldout = Path(args.out_heldout) if args.out_heldout else src.parent / f"gt_heldout_n{M}.npz"
    for o in (out_subset, out_heldout):
        if "/tmp/" in str(o) or str(o).startswith("/tmp"):
            raise ValueError("refusing a /tmp output path (durable evidence non-negotiable).")

    # Materialize each member ONCE (the NpzFile re-inflates the whole member on every index;
    # see load_gt_from_cache for the OOM-balloon this avoids).
    print(json.dumps({"stage": "load", "src": str(src), "cached_P": cached_P}), flush=True)
    members = {k: np.asarray(z[k]) for k in _KEYS}
    print(json.dumps({"stage": "loaded", "secs": round(time.time() - t0, 1),
                      "shapes": {k: list(members[k].shape) for k in _KEYS}}), flush=True)

    def _write(out_path: Path, idx: np.ndarray) -> dict:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"n_pairs": np.int64(idx.size)}
        for k in _KEYS:
            payload[k] = members[k][idx]  # advanced-index copy, dtype preserved
        # np.savez APPENDS ".npz" if the name doesn't end in it -> tmp MUST end in .npz.
        tmp = out_path.parent / (out_path.stem + ".tmp.npz")
        np.savez(tmp, **payload)  # uncompressed (matches the existing caches; fast load)
        tmp.replace(out_path)  # atomic
        return {"path": str(out_path), "n_pairs": int(idx.size),
                "bytes": out_path.stat().st_size,
                "dtypes": {k: str(payload[k].dtype) for k in _KEYS}}

    tw = time.time()
    sub_info = _write(out_subset, idx_subset)
    print(json.dumps({"stage": "wrote_subset", **sub_info, "secs": round(time.time() - tw, 1)}), flush=True)

    held_info = None
    if not args.no_heldout:
        tw = time.time()
        held_info = _write(out_heldout, idx_heldout)
        print(json.dumps({"stage": "wrote_heldout", **held_info, "secs": round(time.time() - tw, 1)}), flush=True)

    manifest = {
        "tool": "build_strided_subset_gt.py",
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "src": str(src), "src_bytes": src.stat().st_size, "src_sha256_head": _sha256_head(src),
        "cached_P": cached_P, "stride": int(args.stride),
        "subset": sub_info, "subset_indices": idx_subset.tolist(),
        "heldout": held_info, "heldout_indices": (None if held_info is None else idx_heldout.tolist()),
        "schema_keys": ["n_pairs", *list(_KEYS)],
        "note": "subset spans the drive at the given stride; heldout = complement; exact frozen-CPU-torch authority slices (NO surrogate).",
    }
    man_path = src.parent / f"gt_strided_stride{int(args.stride)}_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"stage": "manifest", "path": str(man_path), "K_subset": K, "M_heldout": M,
                      "total_secs": round(time.time() - t0, 1)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
