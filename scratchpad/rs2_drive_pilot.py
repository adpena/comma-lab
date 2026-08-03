#!/usr/bin/env python
"""ddm_rs2 PILOT — is the drop's SCORER-PLANE DRIVE measurable, and how local is it?

Three things, all scorer-free (decoder + the frozen resamplers only; no SegNet,
no PoseNet):

  1. Reproduce the live base exactly (tokens, archive bytes, live-cell count).
  2. MEASURE the receptive field of a single cell drop in the (384,512) scorer
     plane -- so the disjoint-support batching in the n600 run is a MEASUREMENT,
     not an assumption.
  3. Pilot the DRIVE currency on a handful of cells on a few pairs, so the cost
     of the full n600 sweep is known before it is launched.

DRIVE(cell) := sum_pairs || D(cam_drop) - D(cam_base) ||_1  over (384,512,3),
where cam = clip(rint(bicubic_up(render))) is the DELIVERED frame_1 and D is the
frozen scorer downsample (bilinear, align_corners=False, antialias=False =
DISJOINT 2x2 point sampling).  DRIVE = 0 is an EXACT, scorer-free proof of zero
argmax flips: SegNet reads only D(f1), so an unchanged D(f1) cannot flip a label.
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
OUT = WORK / "rs2_drive_pilot.json"

for _p in ("src", str(PF), str(RT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import inflate_runner_v4d as IR  # noqa: E402

from tac.optimization import ddm_ix2_archive_container as C  # noqa: E402
from tac.optimization.ddm_ll1_window_solve import window_geometry  # noqa: E402

LEVELS = 16
SEG_H, SEG_W = 384, 512


def build_decoder(codes: np.ndarray) -> IR.Decoder:
    """A Decoder whose token lattice is `codes` and whose everything-else is live."""
    dec = IR.Decoder(CX1_DIR)
    return retoken(dec, codes)


def retoken(dec: IR.Decoder, codes: np.ndarray) -> IR.Decoder:
    from ddm_tr1_runtime import build_packet, parse_packet

    payload = IR._encode_tokens(np.ascontiguousarray(codes, dtype=np.uint8))
    dec.packet = parse_packet(
        build_packet(
            IR.IX2_TR1_METADATA,
            {
                "tokens": payload,
                "lotto_renderer": dec._sections["lotto_renderer"],
                "selector": dec._sections["selector"],
                "pose_stub": dec._sections["pose_stub"],
            },
        )
    )
    return dec


def make_D():
    ys, xs, wy, wx = window_geometry()
    ys = np.asarray(ys)
    xs = np.asarray(xs)
    wy = np.asarray(wy, dtype=np.float64)
    wx = np.asarray(wx, dtype=np.float64)

    def D(cam: np.ndarray) -> np.ndarray:
        v = np.asarray(cam, dtype=np.float64)
        # rows first: (SEG_H, 2, W, 3) -> (SEG_H, W, 3)
        r = (v[ys[:, 0], :, :] * wy[:, 0, None, None]) + (v[ys[:, 1], :, :] * wy[:, 1, None, None])
        out = (r[:, xs[:, 0], :] * wx[None, :, 0, None]) + (r[:, xs[:, 1], :] * wx[None, :, 1, None])
        return out

    return D, (ys, xs, wy, wx)


def main() -> int:
    t_start = time.time()
    rep: dict = {"axis": "[byte-closed, scorer-free]", "score_claim": False}

    dec = IR.Decoder(CX1_DIR)
    # stash the sections so retoken() can rebuild the packet
    blob = CX1_DIR / IR.IX2_MEMBER
    bulk, sections = IR.parse_payload(blob.read_bytes())
    _config, renderer, selector, _pose = sections
    mask_bits, floats = IR.decode_renderer_frame(renderer)
    dec._sections = {
        "lotto_renderer": IR._reframe_renderer(mask_bits, floats),
        "selector": selector,
        "pose_stub": IR.IX2_POSE_STUB,
    }
    codes = C.decode_token_frame(bulk)
    P, R, Cc, K = codes.shape
    rep["lattice"] = [int(P), int(R), int(Cc), int(K)]

    ref = np.load("/Volumes/VertigoDataTier/pact/ddm_br1_20260803/cx1_tokens.npy")
    rep["tokens_match_br1"] = bool(np.array_equal(ref, codes))
    rep["token_member_bytes"] = len(C.encode_token_frame(codes, levels=LEVELS))

    base, delta = C._factor_mode_delta(codes, LEVELS)
    live_unit = (delta != 0).any(axis=0)                     # (R,Cc,K)
    live_cell = live_unit.any(axis=2)                        # (R,Cc)
    rep["live_units"] = int(live_unit.sum())
    rep["live_cells"] = int(live_cell.sum())
    rep["dead_cells"] = int((~live_cell).sum())

    D, _geom = make_D()

    # ---- 2. receptive field of a single cell drop, measured on the scorer plane
    live_rc = np.argwhere(live_cell)
    probe_rc = live_rc[len(live_rc) // 2]
    pr, pc = int(probe_rc[0]), int(probe_rc[1])
    pair = int(np.argmax(np.abs(delta[:, pr, pc, :]).sum(axis=1)))

    dec = retoken(dec, codes)
    t0 = time.time()
    cam_base = dec.f1(pair)
    t_f1 = time.time() - t0
    d_base = D(cam_base)

    mod = codes.copy()
    mod[:, pr, pc, :] = base[pr, pc, :][None, :]
    dec = retoken(dec, mod)
    cam_drop = dec.f1(pair)
    d_drop = D(cam_drop)

    diff_cam = cam_drop.astype(np.int32) - cam_base.astype(np.int32)
    diff_seg = d_drop - d_base
    nz = np.argwhere(np.abs(diff_seg).sum(axis=2) > 0)
    rep["rf_probe"] = {
        "cell": [pr, pc],
        "pair": pair,
        "seg_plane_changed_px": len(nz),
        "camera_changed_px": int((np.abs(diff_cam).sum(axis=2) > 0).sum()),
        "seg_bbox_rows": [int(nz[:, 0].min()), int(nz[:, 0].max())] if len(nz) else None,
        "seg_bbox_cols": [int(nz[:, 1].min()), int(nz[:, 1].max())] if len(nz) else None,
        "cell_center_row_px": (pr + 0.5) * SEG_H / R,
        "cell_center_col_px": (pc + 0.5) * SEG_W / Cc,
        "drive_L1": float(np.abs(diff_seg).sum()),
        "drive_Linf": float(np.abs(diff_seg).max()),
        "seconds_per_f1": round(t_f1, 3),
    }

    # ---- 3. pilot the DRIVE spread over a stratified sample of live cells, 1 pair
    act_cell = (delta != 0).sum(axis=0).sum(axis=2)          # (R,Cc) activity
    live_flat = np.ravel_multi_index(live_rc.T, (R, Cc))
    act_live = act_cell.reshape(-1)[live_flat]
    order = np.argsort(-act_live, kind="stable")
    picks = order[np.linspace(0, len(order) - 1, 12).astype(int)]

    rows = []
    for j in picks:
        idx = int(live_flat[j])
        r_, c_ = divmod(idx, Cc)
        mod = codes.copy()
        mod[:, r_, c_, :] = base[r_, c_, :][None, :]
        dec = retoken(dec, mod)
        dd = D(dec.f1(pair)) - d_base
        rows.append(
            {
                "cell": [int(r_), int(c_)],
                "activity": int(act_cell[r_, c_]),
                "drive_L1_pair": float(np.abs(dd).sum()),
                "changed_px_pair": int((np.abs(dd).sum(axis=2) > 0).sum()),
                "drive_Linf_pair": float(np.abs(dd).max()),
            }
        )
    rep["pilot_rows"] = rows
    a = np.array([r["activity"] for r in rows], dtype=float)
    d = np.array([r["drive_L1_pair"] for r in rows], dtype=float)
    if a.std() > 0 and d.std() > 0:
        rep["pilot_spearman_activity_vs_drive"] = float(
            np.corrcoef(np.argsort(np.argsort(a)), np.argsort(np.argsort(d)))[0, 1]
        )
    rep["pilot_drive_spread"] = {
        "min": float(d.min()),
        "median": float(np.median(d)),
        "max": float(d.max()),
        "ratio_max_min": float(d.max() / d.min()) if d.min() > 0 else None,
        "n_zero": int((d == 0).sum()),
    }
    rep["elapsed_s"] = round(time.time() - t_start, 1)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=2, sort_keys=True))
    print(json.dumps(rep, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
