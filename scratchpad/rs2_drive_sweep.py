#!/usr/bin/env python
"""ddm_rs2 — the DRIVE sweep: n600, every live cell, exact, SCORER-FREE.

WHAT IS MEASURED
----------------
For every LIVE token-lattice cell c (384 of 768; the other 384 are already dead
at the live `cell_drop50` base), the exact perturbation the drop of c induces in
**the SegNet's own input plane**:

    DRIVE(c) = sum_{p<600} || D(cam_drop_c(p)) - D(cam_base(p)) ||_1

with `cam = clip(rint(bicubic_up(render)))` the DELIVERED camera frame_1 and `D`
the frozen scorer downsample (bilinear, align_corners=False, antialias=False =
disjoint 2x2 point sampling).  SegNet reads ONLY `D(f1)`, so:

    DRIVE(c) == 0  <=>  the drop of c causes EXACTLY ZERO argmax flips.

That is an exact scorer-free certificate, not an estimate.  DRIVE is the DRIVE
half of flip damage; the SUSCEPTIBILITY half (how close each perturbed pixel sits
to the argmax separatrix) is the queued scorer leg.

WHY IT IS AFFORDABLE
--------------------
A single cell's drop perturbs a bounded neighbourhood (MEASURED in the pilot:
84 x 82 scorer pixels for cell (13,17), 6,192 px).  Cells whose grid positions are
>= PITCH apart therefore have DISJOINT supports and can be dropped in ONE render.
PITCH=6 gives 36 groups.  Disjointness is VERIFIED, not assumed: the sum of the
per-cell box L1s must equal the whole-plane L1 for every group (leak_L1 == 0).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

SSD = Path("/Volumes/VertigoDataTier/pact")
RT = SSD / "ddm_v4d_20260731"
PF = SSD / "ddm_pfs1_20260729/d1/eval_root/submissions/pfs1"
WORK = SSD / "ddm_rs2_20260803"
CX1_DIR = WORK / "cx1_dir"

for _p in ("src", str(PF), str(RT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import inflate_runner_v4d as IR  # noqa: E402

from tac.optimization import ddm_ix2_archive_container as C  # noqa: E402
from tac.optimization.ddm_ll1_window_solve import window_geometry  # noqa: E402

LEVELS = 16
SEG_H, SEG_W = 384, 512
PITCH = 6
BOX_MARGIN = 40          # px each side of the cell's own 16x16 tile
THRESHOLDS = (0.0, 1.0, 2.0, 4.0, 8.0)


def _sections(dec):
    blob = CX1_DIR / IR.IX2_MEMBER
    bulk, sections = IR.parse_payload(blob.read_bytes())
    _config, renderer, selector, _pose = sections
    mask_bits, floats = IR.decode_renderer_frame(renderer)
    return C.decode_token_frame(bulk), {
        "lotto_renderer": IR._reframe_renderer(mask_bits, floats),
        "selector": selector,
        "pose_stub": IR.IX2_POSE_STUB,
    }


def retoken(dec, secs, codes):
    from ddm_tr1_runtime import build_packet, parse_packet

    dec.packet = parse_packet(
        build_packet(
            IR.IX2_TR1_METADATA,
            {
                "tokens": IR._encode_tokens(np.ascontiguousarray(codes, dtype=np.uint8)),
                **secs,
            },
        )
    )
    return dec


def make_D():
    ys, xs, wy, wx = window_geometry()
    ys, xs = np.asarray(ys), np.asarray(xs)
    wy = np.asarray(wy, dtype=np.float32)
    wx = np.asarray(wx, dtype=np.float32)

    def D(cam: np.ndarray) -> np.ndarray:
        v = np.asarray(cam, dtype=np.float32)
        r = v[ys[:, 0]] * wy[:, 0, None, None] + v[ys[:, 1]] * wy[:, 1, None, None]
        return r[:, xs[:, 0]] * wx[None, :, 0, None] + r[:, xs[:, 1]] * wx[None, :, 1, None]

    return D


def main() -> int:
    t_start = time.time()
    dec = IR.Decoder(CX1_DIR)
    codes, secs = _sections(dec)
    P, R, Cc, K = codes.shape
    base, delta = C._factor_mode_delta(codes, LEVELS)
    live_cell = (delta != 0).any(axis=0).any(axis=2)            # (R,Cc)
    live_idx = np.argwhere(live_cell)
    D = make_D()

    # ---- base pass -------------------------------------------------------
    dec = retoken(dec, secs, codes)
    base_seg = np.empty((P, SEG_H, SEG_W, 3), dtype=np.float32)
    t0 = time.time()
    for p in range(P):
        base_seg[p] = D(dec.f1(p))
    t_base = time.time() - t0
    print(f"base pass {t_base:.1f}s", flush=True)

    ncell = R * Cc
    acc = {
        "drive_L1": np.zeros(ncell),
        "drive_Linf": np.zeros(ncell),
        "px_over": np.zeros((ncell, len(THRESHOLDS))),
        "rf_r0": np.full(ncell, 10**6),
        "rf_r1": np.full(ncell, -1),
        "rf_c0": np.full(ncell, 10**6),
        "rf_c1": np.full(ncell, -1),
    }
    per_pair = np.zeros((ncell, P), dtype=np.float32)
    groups, leaks = [], []

    for dr in range(PITCH):
        for dc in range(PITCH):
            sel = live_idx[(live_idx[:, 0] % PITCH == dr) & (live_idx[:, 1] % PITCH == dc)]
            if len(sel) == 0:
                continue
            mod = codes.copy()
            mod[:, sel[:, 0], sel[:, 1], :] = base[sel[:, 0], sel[:, 1], :][None]
            dec = retoken(dec, secs, mod)

            boxes = []
            for r_, c_ in sel:
                r0 = max(0, r_ * 16 - BOX_MARGIN)
                r1 = min(SEG_H, (r_ + 1) * 16 + BOX_MARGIN)
                c0 = max(0, c_ * 16 - BOX_MARGIN)
                c1 = min(SEG_W, (c_ + 1) * 16 + BOX_MARGIN)
                boxes.append((int(r_) * Cc + int(c_), r0, r1, c0, c1))

            tg, leak = time.time(), 0.0
            for p in range(P):
                d = np.abs(D(dec.f1(p)) - base_seg[p])
                total = float(d.sum())
                inbox = 0.0
                for idx, r0, r1, c0, c1 in boxes:
                    sub = d[r0:r1, c0:c1]
                    s = float(sub.sum())
                    inbox += s
                    if s == 0.0:
                        continue
                    acc["drive_L1"][idx] += s
                    per_pair[idx, p] = s
                    m = sub.max(axis=2)
                    acc["drive_Linf"][idx] = max(acc["drive_Linf"][idx], float(m.max()))
                    for t_i, thr in enumerate(THRESHOLDS):
                        acc["px_over"][idx, t_i] += float((m > thr).sum())
                    nz = np.argwhere(m > 0)
                    if len(nz):
                        acc["rf_r0"][idx] = min(acc["rf_r0"][idx], r0 + int(nz[:, 0].min()))
                        acc["rf_r1"][idx] = max(acc["rf_r1"][idx], r0 + int(nz[:, 0].max()))
                        acc["rf_c0"][idx] = min(acc["rf_c0"][idx], c0 + int(nz[:, 1].min()))
                        acc["rf_c1"][idx] = max(acc["rf_c1"][idx], c0 + int(nz[:, 1].max()))
                leak += total - inbox
            leaks.append(leak)
            groups.append({"dr": dr, "dc": dc, "cells": len(sel), "leak_L1": leak,
                           "seconds": round(time.time() - tg, 1)})
            print(f"group ({dr},{dc}) n={len(sel):3d} leak={leak:.6g} "
                  f"{time.time() - tg:.1f}s  [{time.time() - t_start:.0f}s total]", flush=True)

    out = WORK / "rs2_drive_sweep"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / "cell_drive.npz",
        drive_L1=acc["drive_L1"], drive_Linf=acc["drive_Linf"],
        px_over=acc["px_over"], thresholds=np.array(THRESHOLDS),
        rf_r0=acc["rf_r0"], rf_r1=acc["rf_r1"], rf_c0=acc["rf_c0"], rf_c1=acc["rf_c1"],
        live_cell=live_cell.reshape(-1), per_pair=per_pair,
    )
    (out / "receipt.json").write_text(json.dumps({
        "axis": "[byte-closed, scorer-free]",
        "score_claim": False, "promotion_eligible": False,
        "n_pairs": int(P), "live_cells": int(live_cell.sum()),
        "pitch": PITCH, "box_margin": BOX_MARGIN,
        "thresholds": list(THRESHOLDS),
        "groups": groups,
        "max_abs_leak_L1": float(np.abs(leaks).max()) if leaks else 0.0,
        "disjointness_verified": bool(np.abs(leaks).max() == 0.0) if leaks else None,
        "base_pass_seconds": round(t_base, 1),
        "elapsed_s": round(time.time() - t_start, 1),
    }, indent=2, sort_keys=True))
    print("DONE", round(time.time() - t_start, 1), "s  max|leak|=",
          float(np.abs(leaks).max()) if leaks else 0.0, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
