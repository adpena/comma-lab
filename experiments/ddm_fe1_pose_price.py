#!/usr/bin/env python3
"""ddm_fe1: price ONE frame-embedding move end to end -- seg cells, bits, pose.

A frame-embedding move changes the whole of frame ``2p+1`` for one pair, so the pose
leg is the arm's real gate, not an afterthought.  This prices a named move exactly the
way the shipped chain would carry it:

1. render frame ``2p+1`` under the moved codes (the receiver's own batch-1 forward);
2. splice that frame over the LIVE decode (odd frames only -- a frame-embedding move
   cannot reach frame ``2p``, which the carrier renders);
3. measure the pair's d_pose STALE (live carrier codes, moved render);
4. re-solve that pair's carrier from the LIVE coefficients with ``jg5.refine_pair``
   (damped GN plus the +-2 lattice polish, the shipped chain);
5. measure the pair's d_pose RESOLVED, and price the code change through the SHIPPED
   RC1 semantic coder by a REAL re-encode of the whole section.

Everything is per pair and every number is retained beside its pair index
(PER-PAIR RECEIPTS LAW).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO / "experiments"))

import ddm_br1_pose_basis_reorientation as br1
import ddm_fe1_frame_embedding_search as fe1
import ddm_jg1_seg_solve as jg1
import ddm_jg5_pose_resolve_on_edited_renders as jg5
import ddm_up2_shipping_pose_solve as up2

N_PAIRS = fe1.N_PAIRS
#: Encoder-only container choices.  Brotli streams are self-describing and the CK2
#: interleave rides in the RX1 ``reserved`` byte, so none of these is transmitted.
CONTAINER_QUALITIES = (9, 10, 11)
CONTAINER_LGWINS = (16, 18, 20, 22, 24)
#: The shape the live body ships.  Pinned by BYTE identity, not by length: (ck2, 11, 16)
#: and (ck2, 11, 24) both produce 30,246 B on the shipped codes and they are DIFFERENT
#: 30,246 bytes -- brotli records the window size in its own header -- so a tie-break that
#: reads only the length silently ships a different stream.  Only lgwin=24 reproduces the
#: archive's semantic section byte for byte.
SHIPPED_SHAPE = ("ck2", 11, 24)
CAMERA_H, CAMERA_W = jg1.CAMERA_H, jg1.CAMERA_W
CELL_COUNT = N_PAIRS * fe1.EVAL_H * fe1.EVAL_W


class _MemoryOverlayRaw:
    """The live decode with named odd frames replaced, served the way br1 reads it."""

    def __init__(self, base, frames: dict[int, np.ndarray]) -> None:
        self.base = base
        self.frames = frames
        self.shape = base.shape
        self.dtype = base.dtype

    def __getitem__(self, key):
        arr = np.asarray(key)
        if arr.ndim != 1:
            raise fe1.Fe1Error(
                "overlay only serves 1-D frame-index arrays; another access shape "
                "would silently mix base and candidate frames"
            )
        out = np.asarray(self.base[arr]).copy()
        for row, frame in enumerate(arr.tolist()):
            if frame % 2 == 1 and (frame // 2) in self.frames:
                out[row] = self.frames[frame // 2]
        return out


def build_pose_instrument(raw):
    state = up2.load_carrier_state(fe1.LIVE_TREE, verify_archive=False)
    targets, lineage = up2.load_gt_poses(up2.DEFAULT_DALI_GT)
    if lineage != up2.LINEAGE_DALI:
        raise fe1.Fe1Error(f"GT pose lineage is {lineage}, not {up2.LINEAGE_DALI}")
    up2.verify_gt_lineage(axis="contest_cuda", declared_lineage=lineage)
    posenet = up2.load_posenet()
    up2.enable_posenet_gradients()
    blow = br1.low_basis(state)
    gram, bmat = br1.span_gram(blow)
    return br1.Instrument(state, raw, targets, posenet, blow, gram, bmat)


def semantic_bytes_for(section, codes: np.ndarray) -> int:
    """Exact RC1 semantic stream length for a code table, through the shipped coder."""
    return len(section.stream_with_codes(codes))


def cmd_price(args) -> int:
    fe1._set_threads(args.threads)
    moves = []
    seen: set[int] = set()
    for spec in args.moves.split(";"):
        pair, dim, code = (int(x) for x in spec.split(","))
        if pair in seen and not args.allow_repeat_pairs:
            raise fe1.Fe1Error(
                f"pair {pair} appears twice: the overlay holds ONE frame per pair, so a "
                "second move on the same pair would silently be priced against the "
                "first move's render.  Price them in separate runs, or pass "
                "--allow-repeat-pairs and read every repeated pair's pose leg as "
                "belonging to the LAST move for that pair."
            )
        seen.add(pair)
        moves.append((pair, dim, code))

    body = fe1.load_body(with_raw=False, verify_shas=not args.no_sha)
    base_raw = fe1._open_raw(fe1.LIVE_RAW)
    section = body.section

    # ---- seg + rate legs, and the edited renders -------------------------------
    rows: list[dict[str, Any]] = []
    frames: dict[int, np.ndarray] = {}
    codes_all = section.codes.astype(np.int64).copy()
    base_stream_bytes = len(section.rc1_stream)
    started = time.time()
    for pair, dim, code in moves:
        fe1.restore_pair_codes(body, pair)
        base_flips = fe1.flips_pair(fe1.argmax_pair(body, pair), body, pair)
        row_codes = section.codes[pair].astype(np.int64).copy()
        old = int(row_codes[dim])
        row_codes[dim] = code
        fe1.set_pair_codes(body, pair, row_codes)
        frame = fe1.render_pair(body, pair)
        moved_flips = fe1.flips_pair(
            jg1.argmax_from_camera_frames(body.net, frame)[0], body, pair
        )
        fe1.restore_pair_codes(body, pair)
        frames[pair] = frame[0]
        alone = section.codes.astype(np.int64).copy()
        alone[pair, dim] = code
        alone_bytes = semantic_bytes_for(section, alone)
        codes_all[pair, dim] = code
        cumulative_bytes = semantic_bytes_for(section, codes_all)
        rows.append(
            {
                "pair": pair,
                "dim": dim,
                "old": old,
                "new": code,
                "base_flips": base_flips,
                "moved_flips": moved_flips,
                "d_cells": moved_flips - base_flips,
                # THIS move on the shipped table, nothing else changed.
                "semantic_bytes_alone": alone_bytes,
                "d_semantic_bytes_alone": alone_bytes - base_stream_bytes,
                # THIS move plus every move already applied in this run: the marginal
                # picture, which is what a multi-pair admission actually pays.
                "semantic_bytes_cumulative": cumulative_bytes,
                "d_semantic_bytes_cumulative": cumulative_bytes - base_stream_bytes,
            }
        )
        print(
            f"pair {pair} d{dim} {old:+d}->{code:+d}: flips {base_flips} -> "
            f"{moved_flips} ({moved_flips - base_flips:+d} cells), semantic stream "
            f"{alone_bytes - base_stream_bytes:+d} B alone / "
            f"{cumulative_bytes - base_stream_bytes:+d} B cumulative",
            flush=True,
        )

    combined_bytes = semantic_bytes_for(section, codes_all)

    # ---- pose legs -------------------------------------------------------------
    base_inst = build_pose_instrument(base_raw)
    live_codes = np.asarray(base_inst.state.codes, dtype=np.int32)
    dd_threshold = jg5.materiality_dd_threshold(args.base_mean_d_pose)
    overlay = _MemoryOverlayRaw(base_raw, frames)
    moved_inst = build_pose_instrument(overlay)
    for row in rows:
        pair = row["pair"]
        row["d_pose_base"] = float(
            br1.evaluate_codes(base_inst, pair, live_codes[pair][None])[0]
        )
        row["d_pose_stale"] = float(
            br1.evaluate_codes(moved_inst, pair, live_codes[pair][None])[0]
        )
        refined = jg5.refine_pair(
            moved_inst,
            pair,
            live_codes[pair],
            dd_threshold=dd_threshold,
            outer_rounds=args.outer_rounds,
            max_gn_iterations=args.max_gn_iterations,
        )
        row["d_pose_resolved"] = float(refined["final_d_pose"])
        row["resolved_codes"] = [int(c) for c in refined["codes"]]
        row["refine"] = {
            k: refined[k]
            for k in refined
            if k in ("rounds", "evaluations", "gn_iterations", "polish_steps", "stop_reason")
        }
        row["pose_recovery_x"] = (
            row["d_pose_stale"] / row["d_pose_resolved"]
            if row["d_pose_resolved"] > 0
            else math.inf
        )
        print(
            f"pair {pair}: d_pose base {row['d_pose_base']:.6e} -> stale "
            f"{row['d_pose_stale']:.6e} ({row['d_pose_stale'] / row['d_pose_base']:.2f}x) "
            f"-> resolved {row['d_pose_resolved']:.6e} "
            f"(recovery {row['pose_recovery_x']:.2f}x)",
            flush=True,
        )

    # ---- the per-pair arithmetic ------------------------------------------------
    base_pose_mean = args.base_mean_d_pose
    base_leg = math.sqrt(10.0 * base_pose_mean)
    for row in rows:
        d_pose_mean_new = base_pose_mean + (
            row["d_pose_resolved"] - row["d_pose_base"]
        ) / N_PAIRS
        row["dS_seg"] = 100.0 * row["d_cells"] / CELL_COUNT
        row["dS_pose_resolved"] = math.sqrt(10.0 * d_pose_mean_new) - base_leg
        row["dS_pose_stale"] = (
            math.sqrt(
                10.0
                * (
                    base_pose_mean
                    + (row["d_pose_stale"] - row["d_pose_base"]) / N_PAIRS
                )
            )
            - base_leg
        )
        row["dS_rate_semantic_alone"] = (
            row["d_semantic_bytes_alone"] * fe1.RATE_PER_BYTE
        )
        row["dS_total_semantic_alone"] = (
            row["dS_seg"] + row["dS_pose_resolved"] + row["dS_rate_semantic_alone"]
        )
        row["admits"] = bool(row["dS_total_semantic_alone"] < -2e-5)

    result = {
        "schema": "ddm_fe1_price.v1",
        "axis": (
            "d_seg [macOS-CPU advisory, jg1/sj1 instrument, DALI GT]; d_pose "
            "[cpu_torch fp32, DALI GT]; bytes exact through the shipped RC1 coder"
        ),
        "score_claim": False,
        "moves": [list(m) for m in moves],
        "base_semantic_stream_bytes": base_stream_bytes,
        "combined_semantic_stream_bytes": combined_bytes,
        "d_combined_semantic_bytes": combined_bytes - base_stream_bytes,
        "base_mean_d_pose": base_pose_mean,
        "dd_threshold": dd_threshold,
        "admit_bar": -2e-5,
        "rows": rows,
        "elapsed_s": round(time.time() - started, 1),
        "receipts": body.receipts,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({"out": str(out), "rows": len(rows)}))
    return 0


def archive_section_bytes(section, codes: np.ndarray) -> dict[str, int]:
    """The SEMANTIC SECTION as the archive stores it, both legal container shapes.

    The receiver reads the section as ``brotli -> (CK2 un-interleave if reserved bit
    0x2) -> RC1 rider`` (``runtime/residual_archive.py:193,237,248``).  Brotli streams
    are self-describing and the CK2 choice rides in ``reserved``, so quality, window and
    interleave are ENCODER-ONLY choices the receiver never has to be told about: all of
    them are searched and the smallest wins, the same container search
    ``up3.build_archive`` already runs for the carrier.  This matters far more here than
    it looks.  The RC1 payload is range-coded, so one changed symbol re-randomises every
    bit after it and the shipped container shape -- which was chosen for the SHIPPED
    bytes -- loses roughly 65 B of matches on any edit.  A search over the shapes
    recovers most of that, and skipping it would price the whole arm about 2x too high.
    """
    import brotli
    import ddm_up3_carrier_splice as up3

    stream = section.stream_with_codes(codes)
    interleaved = up3._ck2_interleave_planes(stream)
    shapes: dict[tuple[str, int, int], int] = {}
    for quality in CONTAINER_QUALITIES:
        for lgwin in CONTAINER_LGWINS:
            shapes[("ck2", quality, lgwin)] = len(
                brotli.compress(interleaved, quality=quality, lgwin=lgwin)
            )
            shapes[("plain", quality, lgwin)] = len(
                brotli.compress(stream, quality=quality, lgwin=lgwin)
            )
    best_shape = min(shapes, key=lambda key: (shapes[key], key))
    return {
        "rc1_stream": len(stream),
        "section_shipped_shape": shapes[SHIPPED_SHAPE],
        "section_best": shapes[best_shape],
        "best_shape": list(best_shape),
        "shapes": {"/".join(str(part) for part in k): v for k, v in shapes.items()},
    }


def cmd_rate_law(args) -> int:
    """What does the ARCHIVE charge for N changed frame_embed codes?

    The RC1 payload is a range-coded bitstream: changing one symbol re-randomises every
    bit after it, so the brotli layer above it loses the matches it had on the shipped
    bytes.  That makes the price of an edit almost independent of how many codes moved
    -- a fixed container break plus a small per-code term -- and it is the number that
    decides whether a one-cell-per-pair repair can ever pay.
    """
    fe1._set_threads(args.threads)
    section = fe1.load_semantic_section()
    shipped = archive_section_bytes(section, section.codes)
    if shipped["section_best"] != args.shipped_section_bytes:
        raise fe1.Fe1Error(
            f"container identity failed: the best rebuild of the SHIPPED codes is "
            f"{shipped['section_best']} B, archive header says "
            f"{args.shipped_section_bytes} B"
        )
    rng = np.random.default_rng(args.seed)
    steps = (-1, 1, 2)
    rows = []
    for count in [int(x) for x in args.counts.split(",")]:
        draws = []
        for _ in range(args.repeats):
            codes = section.codes.astype(np.int64).copy()
            pairs = rng.choice(N_PAIRS, size=count, replace=False)
            dims = rng.integers(0, fe1.FRAME_DIM, size=count)
            for pair, dim in zip(pairs, dims, strict=True):
                old = int(codes[pair, dim])
                options = [
                    old + step
                    for step in steps
                    if fe1.CODE_MIN <= old + step <= fe1.CODE_MAX
                ]
                codes[pair, dim] = int(rng.choice(options))
            measured = archive_section_bytes(section, codes)
            draws.append(
                {
                    "d_rc1_stream": measured["rc1_stream"] - shipped["rc1_stream"],
                    "d_section_shipped_shape": measured["section_shipped_shape"]
                    - shipped["section_best"],
                    "d_section_best": measured["section_best"]
                    - shipped["section_best"],
                    "best_shape": measured["best_shape"],
                }
            )
        best = np.array([d["d_section_best"] for d in draws], dtype=np.float64)
        fixed = np.array(
            [d["d_section_shipped_shape"] for d in draws], dtype=np.float64
        )
        rows.append(
            {
                "changed_codes": count,
                "repeats": args.repeats,
                "d_section_best_mean": float(best.mean()),
                "d_section_best_std": float(best.std()),
                "d_section_best_min": int(best.min()),
                "d_section_best_max": int(best.max()),
                "bytes_per_code": float(best.mean() / count),
                "d_section_shipped_shape_mean": float(fixed.mean()),
                "container_search_recovers_bytes": float(fixed.mean() - best.mean()),
                "dS_rate_mean": float(best.mean() * fe1.RATE_PER_BYTE),
                "cells_to_break_even": float(
                    best.mean() * fe1.RATE_PER_BYTE / (100.0 / CELL_COUNT)
                ),
                "draws": draws,
            }
        )
        print(
            f"  N={count:>4}: archive section {best.mean():+7.1f} +-{best.std():>5.1f} B "
            f"({best.mean() / count:+.3f} B/code) -> needs "
            f"{best.mean() * fe1.RATE_PER_BYTE / (100.0 / CELL_COUNT):.1f} repaired "
            f"cells to break even",
            flush=True,
        )
    result = {
        "schema": "ddm_fe1_rate_law.v1",
        "axis": "[exact bytes through the shipped container; scorer-free]",
        "score_claim": False,
        "shipped": shipped,
        "shipped_section_bytes_header": args.shipped_section_bytes,
        "seed": int(args.seed),
        "cell_value_S": 100.0 / CELL_COUNT,
        "rate_per_byte_S": fe1.RATE_PER_BYTE,
        "rows": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(f"wrote {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    price = sub.add_parser("price", help="price named moves end to end")
    price.add_argument(
        "--moves",
        required=True,
        help="semicolon-separated pair,dim,new_code triples",
    )
    price.add_argument("--threads", type=int, default=3)
    price.add_argument("--no-sha", action="store_true")
    price.add_argument("--outer-rounds", type=int, default=40)
    price.add_argument("--max-gn-iterations", type=int, default=400)
    price.add_argument(
        "--base-mean-d-pose", type=float, default=fe1.LIVE_D_POSE
    )
    price.add_argument("--allow-repeat-pairs", action="store_true")
    price.add_argument("--out", default=str(fe1.WORK / "admission/PRICE.json"))
    price.set_defaults(func=cmd_price)

    rate = sub.add_parser("rate-law", help="archive cost of N changed codes")
    rate.add_argument("--counts", default="1,2,5,10,25,50,72,100,150,200")
    rate.add_argument("--repeats", type=int, default=8)
    rate.add_argument("--seed", type=int, default=20260909)
    rate.add_argument("--threads", type=int, default=2)
    rate.add_argument("--shipped-section-bytes", type=int, default=30_246)
    rate.add_argument("--out", default=str(fe1.WORK / "admission/RATE_LAW.json"))
    rate.set_defaults(func=cmd_rate_law)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
