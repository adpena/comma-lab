#!/usr/bin/env python3
"""ddm_rp1 -- seg identity between two decodes of the same body.

THE CONTRACT THIS CHECKS
------------------------
Every token change this arm admits was accepted only because the pair's rendered,
re-segmented argmax came back IDENTICAL on all 196,608 cells.  That is a per-pair,
per-proposal guarantee made during the search.  It is not the same statement as "the
composed field's seg leg equals the base's", because the composed field carries many pairs'
edits at once and reaches the scorer through a different path -- the receiver's own decode
rather than this arm's renderer.

So the guarantee is re-checked here on whole decodes: base against candidate, cell by cell
over all 600 pairs, plus each side's flip count against the DALI GT so the seg leg itself is
reported rather than inferred.  A single differing cell is a REFUSAL, because the whole
premise of a rate-directed pass is that its seg delta is exactly zero.

Reads either an overlay directory (this arm's own renders) or a receiver ``0.raw`` decode,
so the same check serves the render-time precursor and the parse-back gate.

``[macOS-CPU advisory, jg1 instrument, DALI GT lineage]``; ``score_claim=false``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO / "experiments"))

import ddm_jg1_seg_solve as jg1  # noqa: E402
import ddm_rp1_rate_rank as rp1  # noqa: E402
import ddm_up2_shipping_pose_solve as up2  # noqa: E402

N_PAIRS = jg1.N_PAIRS
CAMERA_H, CAMERA_W = jg1.CAMERA_H, jg1.CAMERA_W


def open_frames(spec: str) -> np.ndarray:
    """Return a (600, 874, 1164, 3) uint8 view of the ODD frames of one decode."""
    path = Path(spec)
    if path.is_dir():
        manifest = json.loads((path / "OVERLAY.json").read_text())
        pairs = [int(p) for p in manifest["pairs"]]
        if pairs != list(range(N_PAIRS)):
            raise rp1.Rp1Error(
                f"overlay {path} covers {len(pairs)} pairs, not all {N_PAIRS}; a partial "
                "overlay would silently compare this arm's renders against a decode"
            )
        return np.memmap(
            path / "odd_frames.u8", dtype=np.uint8, mode="r",
            shape=(N_PAIRS, CAMERA_H, CAMERA_W, 3),
        )
    expected = 2 * N_PAIRS * CAMERA_H * CAMERA_W * 3
    if path.stat().st_size != expected:
        raise rp1.Rp1Error(f"decode {path} is {path.stat().st_size} B, expected {expected}")
    raw = np.memmap(
        path, dtype=np.uint8, mode="r", shape=(2 * N_PAIRS, CAMERA_H, CAMERA_W, 3)
    )
    return raw[1::2]


def cmd_check(args: argparse.Namespace) -> int:
    up2.verify_gt_lineage(axis="contest_cuda", declared_lineage=up2.LINEAGE_DALI)
    rp1.set_threads(args.threads)
    net = jg1.load_segnet()
    gt = jg1.load_gt_seg_labels(up2.LINEAGE_DALI)
    base = open_frames(args.base)
    cand = open_frames(args.candidate)

    per_pair_base = np.zeros(N_PAIRS, dtype=np.int64)
    per_pair_cand = np.zeros(N_PAIRS, dtype=np.int64)
    disagreeing_cells = np.zeros(N_PAIRS, dtype=np.int64)
    started = time.time()
    for start in range(0, N_PAIRS, args.batch):
        stop = min(N_PAIRS, start + args.batch)
        idx = np.arange(start, stop)
        a = jg1.argmax_from_camera_frames(net, np.asarray(base[start:stop]))
        b = jg1.argmax_from_camera_frames(net, np.asarray(cand[start:stop]))
        g = np.asarray(gt[start:stop])
        per_pair_base[idx] = (a != g).reshape(len(idx), -1).sum(axis=1)
        per_pair_cand[idx] = (b != g).reshape(len(idx), -1).sum(axis=1)
        disagreeing_cells[idx] = (a != b).reshape(len(idx), -1).sum(axis=1)
        if stop % 100 == 0:
            print(
                json.dumps(
                    {
                        "pairs": stop,
                        "disagreeing_cells_so_far": int(disagreeing_cells.sum()),
                        "s_per_pair": (time.time() - started) / stop,
                    }
                ),
                flush=True,
            )

    cells = N_PAIRS * jg1.EVAL_H * jg1.EVAL_W
    verdict = {
        "schema": "ddm_rp1_segcheck.v1",
        "axis": "[macOS-CPU advisory, jg1 instrument, DALI GT lineage]",
        "score_claim": False,
        "base": str(args.base),
        "candidate": str(args.candidate),
        "cells_compared": cells,
        "cells_disagreeing": int(disagreeing_cells.sum()),
        "pairs_disagreeing": int((disagreeing_cells > 0).sum()),
        "identical": bool(disagreeing_cells.sum() == 0),
        "base_flipped_cells": int(per_pair_base.sum()),
        "candidate_flipped_cells": int(per_pair_cand.sum()),
        "base_d_seg": float(per_pair_base.sum()) / cells,
        "candidate_d_seg": float(per_pair_cand.sum()) / cells,
        "delta_d_seg": float(per_pair_cand.sum() - per_pair_base.sum()) / cells,
        "elapsed_seconds": time.time() - started,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(verdict, indent=2, sort_keys=True))
    np.save(out.with_suffix(".per_pair_base.npy"), per_pair_base)
    np.save(out.with_suffix(".per_pair_candidate.npy"), per_pair_cand)
    print(json.dumps(verdict, indent=2))
    if args.require_identical and not verdict["identical"]:
        raise rp1.Rp1Error(
            f"SEG IDENTITY FAILED: {verdict['cells_disagreeing']} cells over "
            f"{verdict['pairs_disagreeing']} pairs; a rate-directed pass whose seg leg "
            "moved is not the object this arm admitted"
        )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check")
    check.add_argument("--base", required=True, help="overlay dir or 0.raw")
    check.add_argument("--candidate", required=True, help="overlay dir or 0.raw")
    check.add_argument("--out", required=True)
    check.add_argument("--batch", type=int, default=4)
    check.add_argument("--threads", type=int, default=3)
    check.add_argument("--require-identical", action="store_true")
    check.set_defaults(func=cmd_check)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
