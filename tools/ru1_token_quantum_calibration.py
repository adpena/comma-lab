#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""ddm_ru1 Tier-B — token-quantum currency calibration on the t3 endpoint archive.

Measures, with REAL single-token edits through the committed TR1 receiver +
frozen CPU-torch SegNet, the conversion between the vehicle's actionable
quantum (one 4-bit token level, = 2/15 in renderer input units) and the logit
deficits recorded by the Tier-A atlas:

  per edit (pair, cell_r, cell_c, channel, delta in {+1,-1}):
    flips_before/after (whole pair), in-cell flips before/after,
    mean & median m_def improvement at previously-flipped in-cell pixels (κ),
    Δbytes of the re-encoded token section (brotli, exact).

This is INSTRUMENTATION (unit calibration for the residual typing), not a
solve — pb1 owns the QDBS/GN solve chain. Sampled edits only.

Axis: [macOS-CPU advisory]. score_claim=false. Consumer: ddm_ru1 deliverable
1/2 quantum histogram + pb1 P2 aim quantification.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path("/Users/adpena/projects/pact")
SCHEMA = "ddm_ru1_token_quantum_calibration.v1"
CELL = 16  # px per token cell at (384,512)

SLOT_TOKENS = ("pb1_receiver_realized_verdict", "train_levelset_witness",
               "train_witness_realized", "pb1_qdbs", "pb1_p2")


def slot_is_live() -> bool:
    out = subprocess.run(["ps", "-axo", "command"], capture_output=True,
                         text=True, check=False).stdout
    return any(tok in line for line in out.splitlines()
               for tok in SLOT_TOKENS if "ru1_token_quantum" not in line)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--archive", required=True, type=Path)
    ap.add_argument("--gt-cache", required=True, type=Path)
    ap.add_argument("--atlas-flat", required=True, type=Path,
                    help="atlas_flat.npz from the Tier-A pass")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--pairs", type=int, nargs="+",
                    default=[66, 138, 300, 450, 517, 560])
    ap.add_argument("--cells-per-pair", type=int, default=3)
    ap.add_argument("--bytes-sample-every", type=int, default=4,
                    help="measure Δbytes on every k-th edit (brotli cost)")
    ap.add_argument("--skip-slot-check", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not args.skip_slot_check and slot_is_live():
        raise SystemExit("[refuse] pb1/trainer scorer job live - ru1 top-ups "
                         "only when the slot is idle (charter)")
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO / "upstream"))
    import torch

    from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap
    from tac.boundary_math.seg_core import load_real_segnet
    from tac.optimization.ddm_tr1_runtime import (
        _encode_tokens,
        parse_archive,
        render_frame1_camera_uint8,
    )

    parsed = parse_archive(args.archive.read_bytes())
    packet = parsed.packet
    # token_codes is backed by a read-only frombuffer view; install a writable
    # in-memory copy for the edit loop (analysis only - never re-emitted).
    codes = np.array(packet.token_codes, dtype=np.uint8, copy=True)
    object.__setattr__(packet, "token_codes", codes)  # (P, gh, gw, C)
    levels = int(packet.selector["token_quant_levels"])
    lstars = open_stored_npy_memmap(args.gt_cache, "lstars")
    seg_cpu = load_real_segnet("cpu")

    flat = np.load(args.atlas_flat)
    fpair, fy, fx = flat["pair"], flat["y"], flat["x"]
    fmd = flat["m_def"]

    base_tokens_bytes = len(_encode_tokens(codes))

    def seg_forward(frame_u8: np.ndarray) -> np.ndarray:
        xp = (torch.from_numpy(np.asarray(frame_u8)[None, None])
              .permute(0, 1, 4, 2, 3).contiguous().float())
        with torch.inference_mode():
            z = seg_cpu(seg_cpu.preprocess_input(xp))
        return z[0].cpu().numpy().astype(np.float32)  # (5,h,w)

    rows: list[dict] = []
    t_start = time.time()
    edit_counter = 0
    for pi in args.pairs:
        gt = np.asarray(lstars[pi], dtype=np.int64)
        sel = fpair == pi
        if not sel.any():
            continue
        # top cells for this pair by flip count
        cy, cx = (fy[sel] // CELL).astype(int), (fx[sel] // CELL).astype(int)
        cell_ids = cy * 32 + cx
        uniq, cnt = np.unique(cell_ids, return_counts=True)
        top_cells = uniq[np.argsort(cnt)[::-1][:args.cells_per_pair]]

        z0 = seg_forward(render_frame1_camera_uint8(packet, pi))
        realized0 = z0.argmax(axis=0)
        flips0 = int((realized0 != gt).sum())

        for cell in top_cells:
            r, c = int(cell) // 32, int(cell) % 32
            ys, xs = np.nonzero((realized0 != gt)[r * CELL:(r + 1) * CELL,
                                                  c * CELL:(c + 1) * CELL])
            ys, xs = ys + r * CELL, xs + c * CELL
            if ys.size == 0:
                continue
            gtc = gt[ys, xs]
            rlc0 = realized0[ys, xs]
            idxk = np.arange(ys.size)
            md0 = z0[:, ys, xs][rlc0, idxk] - z0[:, ys, xs][gtc, idxk]
            for ch in range(codes.shape[3]):
                for delta in (+1, -1):
                    old = int(codes[pi, r, c, ch])
                    new = old + delta
                    if new < 0 or new >= levels:
                        continue
                    codes[pi, r, c, ch] = new
                    z1 = seg_forward(render_frame1_camera_uint8(packet, pi))
                    d_bytes = None
                    if edit_counter % args.bytes_sample_every == 0:
                        d_bytes = len(_encode_tokens(codes)) - base_tokens_bytes
                    codes[pi, r, c, ch] = old  # restore
                    realized1 = z1.argmax(axis=0)
                    flips1 = int((realized1 != gt).sum())
                    fixed_incell = int((realized1[ys, xs] == gtc).sum())
                    md1 = (z1[:, ys, xs][rlc0, idxk]
                           - z1[:, ys, xs][gtc, idxk])
                    dmd = md0 - md1  # >0 means deficit moved toward the fix
                    rows.append({
                        "pair": pi, "cell_r": r, "cell_c": c, "channel": ch, "delta": delta,
                        "flips_before": flips0, "flips_after": flips1,
                        "net_fixed": flips0 - flips1,
                        "incell_flips_before": int(ys.size),
                        "incell_fixed": fixed_incell,
                        "kappa_mean": float(np.mean(np.abs(dmd))),
                        "kappa_med": float(np.median(np.abs(dmd))),
                        "kappa_signed_mean": float(np.mean(dmd)),
                        "d_bytes": d_bytes,
                    })
                    edit_counter += 1
        print(f"[pair {pi}] edits so far {edit_counter} "
              f"({time.time() - t_start:.0f}s)", flush=True)

    kmed = [r["kappa_med"] for r in rows]
    kmean = [r["kappa_mean"] for r in rows]
    net = [r["net_fixed"] for r in rows]
    dbytes = [r["d_bytes"] for r in rows if r["d_bytes"] is not None]
    best_per_cell: dict[tuple, int] = {}
    for r in rows:
        key = (r["pair"], r["cell_r"], r["cell_c"])
        best_per_cell[key] = max(best_per_cell.get(key, -10**9), r["net_fixed"])
    kappa_med_all = float(np.median(kmed)) if kmed else float("nan")
    frac_below = (float((fmd < kappa_med_all).mean())
                  if kmed else float("nan"))
    receipt = {
        "schema": SCHEMA,
        "evidence_axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotion_eligible": False,
        "archive": str(args.archive),
        "n_edits": len(rows),
        "quantum_input_units": 2.0 / (levels - 1),
        "kappa_med_of_medians": kappa_med_all,
        "kappa_p25": float(np.percentile(kmed, 25)) if kmed else None,
        "kappa_p75": float(np.percentile(kmed, 75)) if kmed else None,
        "kappa_mean_of_means": float(np.mean(kmean)) if kmean else None,
        "net_fixed_best": int(max(net)) if net else None,
        "net_fixed_med": float(np.median(net)) if net else None,
        "net_fixed_frac_positive": float(np.mean([n > 0 for n in net])) if net else None,
        "best_single_edit_per_cell": [
            {"pair": k[0], "cell": [k[1], k[2]], "best_net_fixed": v}
            for k, v in sorted(best_per_cell.items())],
        "d_bytes_samples": dbytes,
        "d_bytes_med": float(np.median(dbytes)) if dbytes else None,
        "atlas_frac_m_def_below_kappa_med": frac_below,
        "wall_seconds": float(time.time() - t_start),
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(receipt, indent=1) + "\n")
    tmp.replace(args.out)
    print(json.dumps({k: receipt[k] for k in (
        "n_edits", "kappa_med_of_medians", "kappa_p25", "kappa_p75",
        "net_fixed_best", "net_fixed_med", "net_fixed_frac_positive",
        "d_bytes_med", "atlas_frac_m_def_below_kappa_med",
        "wall_seconds")}, indent=1))
    print(f"receipt: {args.out}")


if __name__ == "__main__":
    main()
