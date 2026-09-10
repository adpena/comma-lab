"""ddm_gdc4 stage-00d: the dense 2-D context floor of the same token field.

The run-native factorization pays an ADDRESSING tax: it must say which
transition maps to which before it can say where the boundary is. A dense
per-cell context coder never pays that tax. This probe prices the dense
factorization on the identical field with the identical honesty rules, so the
arm's verdict compares two real numbers instead of one number and a belief.

For each context family the probe reports:

* ``static_bytes``  -- the empirical conditional code length
  ``sum_ctx sum_sym n log2(n_ctx / n_ctx,sym)``, i.e. an ideal two-pass coder;
* ``model_penalty_bytes`` -- the MDL cost of the table an adaptive coder would
  learn online, ``used_contexts * (A - 1) / 2 * log2(N)`` bits;
* ``adaptive_equivalent_bytes`` -- their sum, the honest one-pass number.

``research_only=true`` · ``score_claim=false``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np

FIELD_SHAPE = (600, 384, 512)
DOOR_TOTAL_B = 94_010
SHIPPED_TAIL_B = 119_969
A = 5


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _shift_up(frame: np.ndarray, fill: int = 0) -> np.ndarray:
    out = np.empty_like(frame)
    out[0] = fill
    out[1:] = frame[:-1]
    return out


def _shift_left(frame: np.ndarray, fill: int = 0) -> np.ndarray:
    out = np.empty_like(frame)
    out[:, 0] = fill
    out[:, 1:] = frame[:, :-1]
    return out


def _shift_right(frame: np.ndarray, fill: int = 0) -> np.ndarray:
    out = np.empty_like(frame)
    out[:, -1] = fill
    out[:, :-1] = frame[:, 1:]
    return out


def context_families(frame: np.ndarray, prev: np.ndarray | None) -> dict[str, np.ndarray]:
    """Build causal context index planes for one frame.

    Every component is an already-decoded cell in raster order (left, up,
    up-left, up-right) or a cell of the fully decoded previous frame, so each
    family is realizable by a real sequential decoder.
    """
    left = _shift_left(frame).astype(np.int32)
    up = _shift_up(frame).astype(np.int32)
    upleft = _shift_left(_shift_up(frame)).astype(np.int32)
    upright = _shift_right(_shift_up(frame)).astype(np.int32)
    prev_same = (prev if prev is not None else np.zeros_like(frame)).astype(np.int32)
    prev_left = _shift_left(prev_same.astype(np.uint8)).astype(np.int32)
    prev_up = _shift_up(prev_same.astype(np.uint8)).astype(np.int32)

    fam: dict[str, np.ndarray] = {}
    fam["left_up"] = left * 5 + up
    fam["left_up_ul_ur"] = ((left * 5 + up) * 5 + upleft) * 5 + upright
    fam["left_up_ul_ur_prev"] = fam["left_up_ul_ur"] * 5 + prev_same
    fam["left_up_ul_ur_prev3"] = (fam["left_up_ul_ur_prev"] * 5 + prev_left) * 5 + prev_up
    return fam


FAMILY_SIZES = {
    "left_up": 25,
    "left_up_ul_ur": 625,
    "left_up_ul_ur_prev": 3_125,
    "left_up_ul_ur_prev3": 78_125,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--field", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--frames", type=int, default=FIELD_SHAPE[0])
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args(argv)

    field_path = Path(args.field)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    n_frames = int(args.frames)

    field = np.memmap(field_path, dtype=np.uint8, mode="r", shape=FIELD_SHAPE)
    joints = {name: np.zeros(size * A, dtype=np.int64) for name, size in FAMILY_SIZES.items()}

    started = time.time()
    prev = None
    for f in range(n_frames):
        frame = np.asarray(field[f])
        fams = context_families(frame, prev)
        sym = frame.reshape(-1).astype(np.int64)
        for name, plane in fams.items():
            idx = plane.reshape(-1).astype(np.int64) * A + sym
            joints[name] += np.bincount(idx, minlength=FAMILY_SIZES[name] * A)
        prev = frame
        if f % 100 == 0:
            print(f"[gdc4-dense] frame {f}/{n_frames} {time.time() - started:.1f}s", flush=True)

    total_cells = n_frames * FIELD_SHAPE[1] * FIELD_SHAPE[2]
    log2n = math.log2(total_cells)
    results = {}
    for name, joint in joints.items():
        table = joint.reshape(-1, A)
        ctx_total = table.sum(axis=1)
        used = int((ctx_total > 0).sum())
        nz = table > 0
        counts = table[nz].astype(np.float64)
        ctx_rep = np.repeat(ctx_total, A).reshape(-1, A)[nz].astype(np.float64)
        bits = float((counts * np.log2(ctx_rep / counts)).sum())
        penalty_bits = used * (A - 1) / 2.0 * log2n
        results[name] = {
            "contexts_declared": FAMILY_SIZES[name],
            "contexts_used": used,
            "static_bits": bits,
            "static_bytes": bits / 8.0,
            "model_penalty_bytes": penalty_bits / 8.0,
            "adaptive_equivalent_bytes": (bits + penalty_bits) / 8.0,
            "bits_per_cell": bits / total_cells,
            "vs_door_ratio": ((bits + penalty_bits) / 8.0) / DOOR_TOTAL_B,
        }

    best = min(results, key=lambda k: results[k]["adaptive_equivalent_bytes"])
    result = {
        "arm": "ddm_gdc4",
        "stage": "stage00d_dense_context_floor",
        "research_only": True,
        "score_claim": False,
        "promotable": False,
        "axis": "[macOS-CPU byte-only n600]",
        "seed": int(args.seed),
        "frames": n_frames,
        "cells": total_cells,
        "field": {
            "path": str(field_path),
            "sha256": sha256_file(field_path) if n_frames == FIELD_SHAPE[0] else None,
        },
        "families": results,
        "best_family": best,
        "best_adaptive_equivalent_bytes": results[best]["adaptive_equivalent_bytes"],
        "door_total_bytes": DOOR_TOTAL_B,
        "shipped_tail_bytes": SHIPPED_TAIL_B,
        "elapsed_s": time.time() - started,
        "note": (
            "These are code lengths for a free ONLINE model: nothing is "
            "transmitted beyond the coded symbols, and the penalty term charges "
            "the table an adaptive coder learns. Any offline-trained model must "
            "beat this AND pay for its own counted weights."
        ),
    }
    out_path = out_dir / "RESULT.json"
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(result, indent=2, sort_keys=True))
    os.replace(tmp, out_path)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
