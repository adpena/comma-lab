# SPDX-License-Identifier: MIT
"""ddm_lr2 -- the LEGAL REALIZATION LADDER for the phase-field seg gain.

bz1 measured exactly ONE legal realizer of the block16 offset field (naive block-translation of
the DECODED camera RGB round-tripped through D) and found eta ~0.11 vs its 0.426 bar.  That
verdict is INSTANCE-scoped to the naive realizer.  This harness measures the ladder of BETTER
legal realizers on the same pairs, each against ITS OWN recomputed bar:

  RUNG A1  token-field translation + re-render: bilinearly resample the shipped TR1 token grid
           (24x32 == the block16 grid exactly, grid_downsample=16) by (dy/16, dx/16) per cell,
           re-render through the SHIPPED LOTTO renderer, bicubic-up, uint8, score.  Carrier =
           the same offset field.  Legal: renderer + offsets + free code, no scorer at decode.
  RUNG A2  pre-R native translate: translate the renderer's native 384x512 float output per
           block by the offsets, THEN bicubic-up + uint8.  Removes bz1's double-resample
           (camera -> D -> translate -> bilinear-up) from the loop.  Same carrier.
  RUNG A3  response-solved token offsets (subcommand `response`): for each candidate global
           offset in [-rmax,rmax]^2 render the globally-token-shifted frame, argmax it, and
           select PER BLOCK the offset whose REALIZED argmax agrees best with GT; compose the
           per-block token warp, final render, score.  This solves the offsets against the
           RENDER RESPONSE instead of the label-field translation -- the direct attack on the
           SegNet non-equivariance that killed the naive realizer.  Offsets re-priced.
  RUNG C   receiver-derivable context paint (subcommand `solve`): per SELECTED block one RGB
           shift (3 int8), applied over the block's camera pixels on top of the A2 transport.
           Address = block index (receiver knows the moved blocks from the offsets; selection
           indices are counted).  Solved encode-side with Adam vs the frozen head (legal:
           solve at ENCODE, ship constants).
  RUNG B   warp + solved sparse residual (subcommand `solve`): per-pixel delta on the
           moved-block band solved encode-side (sq1/js1 mechanics), then TOP-K sparsified and
           priced as (position,value) records.  eta(K) traces the knee.

Every rung: (i) same pairs as bz1 (the first 8 of sq1's stratified 32); (ii) LEGAL realized
eta with whole-frame accounting eta = (flips_before - flips_after)/n_described; (iii) actual
carrier bytes with a real coder (LZMA1); (iv) the rung's OWN bar = rate_total/gross; (v)
d_pose ratio per arm (the frame_0 k=4 repair is PROVEN to 124x damage -- bz1 G2 -- its 96
B/pair stream is included in every bar).

Axis: [macOS-CPU frozen-scorer advisory] NON-PROMOTABLE.  score_claim=False.
"""
from __future__ import annotations

import argparse
import json
import lzma
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))
SUB_PU2 = Path("/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/submission_pu2")
sys.path.insert(0, str(SUB_PU2))

from ddm_et1_ph1_block16_on_our_vehicle import solve_blocks, translate_blocks  # noqa: E402
from ddm_sq1_eta_seg_realization import (  # noqa: E402
    CAM_H,
    CAM_W,
    N_PAIRS_TOTAL,
    SEG_H,
    SEG_W,
    Scorer,
    decode_gt_frames,
    scorer_mask_to_camera,
    seq_len,
)
from ddm_sq1_stage_decomposition_and_solved_paint import (  # noqa: E402
    realize_scorer_paint_to_camera,
    resize_to_scorer,
)

S_PER_FLIP = 100.0 / (N_PAIRS_TOTAL * SEG_H * SEG_W)
RATE_PER_BYTE = 25.0 / 37_545_489.0
BLOCK = 16
DS = 16                       # TR1 grid_downsample (asserted at runtime)
GROSS_N600_S = 0.18039        # et1 re-solved block16 gross on our field (n600 label ceiling)
OFFSET_LZMA1_N600 = 57_809    # bz1-measured LZMA1 price of the label-solved n600 offset field
POSE_STREAM_N600 = 57_600     # 96 B/pair k=4 frame_0 repair (bz1 G1/G2 PASS)
OUT_DIR = Path("/Volumes/VertigoDataTier/pact/ddm_lr2_20260804")


def lzma1_raw(b: bytes) -> int:
    filt = [{"id": lzma.FILTER_LZMA1, "dict_size": 1 << 22, "lc": 0, "lp": 0, "pb": 0}]
    return len(lzma.compress(b, format=lzma.FORMAT_RAW, filters=filt))


# ================================================================================================
# shipped-vehicle render path (RUNG A substrate)
# ================================================================================================
def load_decoder():
    import inflate_runner as ir

    dec = ir.Decoder(SUB_PU2 / "archive")
    sel = dec.packet.selector
    if (sel["grid_downsample"], sel["grid_h"], sel["grid_w"]) != (DS, SEG_H // DS, SEG_W // DS):
        raise SystemExit("TR1 selector geometry differs from the block16 assumption")
    return dec


def render_from_tokens(packet, tok: np.ndarray) -> np.ndarray:
    """render_frame1_float with a CALLER-SUPPLIED token grid (same weights/path otherwise)."""
    import ddm_tr1_runtime as tr1

    x = np.ascontiguousarray(tok, dtype=np.float32)[None]
    weights = tr1._fixed_lotto_weights(packet)
    shapes = tr1._conv_shapes(packet.selector)
    x = tr1._conv2d_nhwc(x, weights[0]) + packet.biases[0]
    x = tr1._gelu_exact(x)
    for index, (name, _shape) in enumerate(shapes[1:-1], start=1):
        if not name.startswith("up"):
            raise SystemExit("TR1 renderer layer order differs")
        x = np.repeat(np.repeat(x, 2, axis=1), 2, axis=2)
        x = tr1._conv2d_nhwc(x, weights[index]) + packet.biases[index]
        x = tr1._gelu_exact(x)
    x = tr1._conv2d_nhwc(x, weights[-1]) + packet.biases[-1]
    return np.ascontiguousarray(tr1._sigmoid(x)[0] * np.float32(255), dtype=np.float32)


def to_camera_u8(native_float: np.ndarray) -> np.ndarray:
    import ddm_tr1_runtime as tr1

    up = tr1.bicubic_up_to_camera_float(native_float)
    return np.ascontiguousarray(np.clip(np.rint(up), 0, 255).astype(np.uint8))


def resample_tokens(tok: np.ndarray, off: np.ndarray) -> np.ndarray:
    """Per-cell fractional translation of the token grid: out[i,j] = T[i+dy/DS, j+dx/DS],
    bilinear, border-clamped.  Same sampling convention as translate_blocks (sample FROM the
    shifted location).  One block16 == one token cell (asserted in load_decoder)."""
    gh, gw, cw = tok.shape
    off2 = off.reshape(gh, gw, 2).astype(np.float32)
    ii = np.arange(gh, dtype=np.float32)[:, None] + off2[:, :, 0] / DS
    jj = np.arange(gw, dtype=np.float32)[None, :] + off2[:, :, 1] / DS
    ii = np.clip(ii, 0.0, gh - 1.0)
    jj = np.clip(jj, 0.0, gw - 1.0)
    i0 = np.floor(ii).astype(np.int64)
    j0 = np.floor(jj).astype(np.int64)
    i1 = np.minimum(i0 + 1, gh - 1)
    j1 = np.minimum(j0 + 1, gw - 1)
    fi = (ii - i0)[..., None].astype(np.float32)
    fj = (jj - j0)[..., None].astype(np.float32)
    out = (tok[i0, j0] * (1 - fi) * (1 - fj) + tok[i1, j0] * fi * (1 - fj)
           + tok[i0, j1] * (1 - fi) * fj + tok[i1, j1] * fi * fj)
    return np.ascontiguousarray(out, dtype=np.float32)


def translate_native_blocks(img: np.ndarray, off: np.ndarray) -> np.ndarray:
    """bz1's translate_rgb_blocks on the renderer's NATIVE (384,512,3) float output."""
    nby, nbx = SEG_H // BLOCK, SEG_W // BLOCK
    out = img.copy()
    for bi in range(nby):
        for bj in range(nbx):
            dy, dx = int(off[bi * nbx + bj][0]), int(off[bi * nbx + bj][1])
            if dy == 0 and dx == 0:
                continue
            ys, ye, xs, xe = bi * BLOCK, (bi + 1) * BLOCK, bj * BLOCK, (bj + 1) * BLOCK
            yy = np.clip(np.arange(ys, ye) + dy, 0, SEG_H - 1)
            xx = np.clip(np.arange(xs, xe) + dx, 0, SEG_W - 1)
            out[ys:ye, xs:xe] = img[np.ix_(yy, xx)]
    return out


def block_mask(off: np.ndarray) -> np.ndarray:
    """(384,512) bool mask of MOVED blocks (nonzero offset) -- receiver-derivable."""
    nby, nbx = SEG_H // BLOCK, SEG_W // BLOCK
    moved = (np.abs(off.reshape(nby, nbx, 2)).sum(axis=2) > 0)
    return np.repeat(np.repeat(moved, BLOCK, axis=0), BLOCK, axis=1)


# ================================================================================================
# shared per-pair context
# ================================================================================================
class PairCtx:
    def __init__(self, sc, dec_pkt, raw, gt_frames, p: int):
        self.p = p
        self.dec = np.stack([raw[seq_len * p], raw[seq_len * p + 1]]).astype(np.uint8)
        self.gt = np.stack([gt_frames[seq_len * p], gt_frames[seq_len * p + 1]])
        self.lstar = sc.seg_argmax(self.dec)
        self.lgt = sc.seg_argmax(self.gt)
        self.flips0 = int((self.lstar != self.lgt).sum())
        self.off = solve_blocks(self.lstar, self.lgt, BLOCK, 5).reshape(-1, 2)
        self.target = translate_blocks(self.lstar, self.off, BLOCK)
        self.nd = self.flips0 - int((self.target != self.lgt).sum())
        self.pose_gt = sc.pose_out(self.gt)
        self.d_pose_shipped = sc.d_pose(self.pose_gt, sc.pose_out(self.dec))
        self.dec_pkt = dec_pkt

    def score_f1(self, sc, cam_f1_u8: np.ndarray, region: np.ndarray | None = None) -> dict:
        pair = np.stack([self.dec[0], cam_f1_u8]).astype(np.uint8)
        lam = sc.seg_argmax(pair)
        fa = int((lam != self.lgt).sum())
        dp = sc.d_pose(self.pose_gt, sc.pose_out(pair))
        out = {
            "flips_after": fa,
            "eta": ((self.flips0 - fa) / self.nd) if self.nd else None,
            "d_pose": dp,
            "d_pose_ratio_vs_shipped": (dp / self.d_pose_shipped) if self.d_pose_shipped else None,
        }
        if region is not None:
            # sg3 crossover quantity: collateral flip rate on already-correct pixels.
            was_ok = self.lstar == self.lgt
            intro = was_ok & (lam != self.lgt)
            out["fixed_flips"] = int(((self.lstar != self.lgt) & (lam == self.lgt)).sum())
            out["introduced_flips_total"] = int(intro.sum())
            out["correct_px_in_region"] = int((was_ok & region).sum())
            out["introduced_in_region"] = int((intro & region).sum())
            out["collateral_rate_in_region"] = (
                int((intro & region).sum()) / int((was_ok & region).sum())
                if int((was_ok & region).sum()) else None)
        return out


def checkpoint(out_path: Path, payload: dict) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=1)
    tmp.replace(out_path)


def base_header(tag: str) -> dict:
    return {
        "schema": f"ddm_lr2_{tag}.v1",
        "axis": "[macOS-CPU frozen-scorer advisory] NON-PROMOTABLE",
        "score_claim": False,
        "promotion_eligible": False,
        "S_per_flip": S_PER_FLIP,
        "rate_per_byte": RATE_PER_BYTE,
        "gross_n600_S": GROSS_N600_S,
        "offset_lzma1_n600": OFFSET_LZMA1_N600,
        "pose_stream_n600": POSE_STREAM_N600,
    }


# ================================================================================================
# subcommand: transport  (RUNG A1 + A2, cheap deterministic realizers)
# ================================================================================================
def run_transport(args, sc, dec, raw, gt_frames, pairs) -> None:
    import ddm_tr1_runtime as tr1

    out = base_header("transport")
    out["rows"] = []
    for n, p in enumerate(pairs):
        t0 = time.time()
        ctx = PairCtx(sc, dec.packet, raw, gt_frames, p)

        # control: the shipped frame_1 IS the TR1 render (bit-exact) and reproduces flips0
        cam_ship = tr1.render_frame1_camera_uint8(dec.packet, p)
        control_render = bool(np.array_equal(cam_ship, ctx.dec[1]))

        tok = tr1.decode_token_grid(dec.packet, p)
        native = render_from_tokens(dec.packet, tok)

        # A1: token-field fractional translation + re-render
        a1 = ctx.score_f1(sc, to_camera_u8(render_from_tokens(dec.packet, resample_tokens(tok, ctx.off))))
        # A2: pre-R native-output block translation
        a2 = ctx.score_f1(sc, to_camera_u8(translate_native_blocks(native, ctx.off)))

        row = {"pair": p, "flips_before": ctx.flips0, "n_described": ctx.nd,
               "moved_blocks": int((np.abs(ctx.off).sum(axis=1) > 0).sum()),
               "control_render_matches_shipped": control_render,
               "d_pose_shipped": ctx.d_pose_shipped,
               "A1_token_resample": a1, "A2_native_translate": a2,
               "offsets_lzma1_this_pair": lzma1_raw(ctx.off.astype(np.int8).tobytes()),
               "wall_s": round(time.time() - t0, 1)}
        out["rows"].append(row)
        checkpoint(args.out, out)
        print(f"[transport] pair {p:3d} ({n+1}/{len(pairs)}) nd {ctx.nd:5d} "
              f"| A1 eta {a1['eta']:.4f} dpx {a1['d_pose_ratio_vs_shipped']:.2f} "
              f"| A2 eta {a2['eta']:.4f} dpx {a2['d_pose_ratio_vs_shipped']:.2f} "
              f"[{row['wall_s']}s]", flush=True)


# ================================================================================================
# subcommand: response  (RUNG A3 -- response-solved token offsets)
# ================================================================================================
def run_response(args, sc, dec, raw, gt_frames, pairs) -> None:
    import ddm_tr1_runtime as tr1

    rmax = args.rmax
    cand = [(dy, dx) for dy in range(-rmax, rmax + 1) for dx in range(-rmax, rmax + 1)]
    nby, nbx = SEG_H // BLOCK, SEG_W // BLOCK
    out = base_header("response")
    out["rmax"] = rmax
    out["rows"] = []
    for n, p in enumerate(pairs):
        t0 = time.time()
        ctx = PairCtx(sc, dec.packet, raw, gt_frames, p)
        tok = tr1.decode_token_grid(dec.packet, p)
        native = render_from_tokens(dec.packet, tok)

        # per-candidate GLOBAL shift -> realized argmax -> per-block agreement, on BOTH
        # actuation substrates: (tok) token-warp + re-render, (rgb) native-RGB translate.
        best = {s: np.full((nby, nbx), -1, dtype=np.int64) for s in ("tok", "rgb")}
        best_off = {s: np.zeros((nby, nbx, 2), dtype=np.int8) for s in ("tok", "rgb")}
        for ci, (dy, dx) in enumerate(cand):
            off_g = np.tile(np.array([dy, dx], dtype=np.int8), (nby * nbx, 1))
            cams = {
                "tok": to_camera_u8(render_from_tokens(dec.packet, resample_tokens(tok, off_g))),
                "rgb": to_camera_u8(translate_native_blocks(native, off_g)),
            }
            for s, cam in cams.items():
                lam = sc.seg_argmax(np.stack([ctx.dec[0], cam]).astype(np.uint8))
                ag = (lam == ctx.lgt).reshape(nby, BLOCK, nbx, BLOCK).sum(axis=(1, 3))
                if dy == 0 and dx == 0:
                    # seed the zero offset so ties resolve to (0,0) (entropy discipline, et1)
                    upd = ag >= best[s]
                else:
                    upd = ag > best[s]
                best[s][upd] = ag[upd]
                best_off[s][upd] = (dy, dx)
            if ci % 20 == 0:
                print(f"    cand {ci+1}/{len(cand)} t={time.time()-t0:.0f}s", flush=True)

        zero_ag = (ctx.lstar == ctx.lgt).reshape(nby, BLOCK, nbx, BLOCK).sum(axis=(1, 3))
        row = {"pair": p, "flips_before": ctx.flips0, "n_described_label": ctx.nd,
               "moved_blocks_label": int((np.abs(ctx.off).sum(axis=1) > 0).sum()),
               "offsets_label_lzma1": lzma1_raw(ctx.off.astype(np.int8).tobytes()),
               "d_pose_shipped": ctx.d_pose_shipped}
        for s in ("tok", "rgb"):
            off_r = best_off[s].reshape(-1, 2)
            cam = (to_camera_u8(render_from_tokens(dec.packet, resample_tokens(tok, off_r)))
                   if s == "tok" else to_camera_u8(translate_native_blocks(native, off_r)))
            comp = ctx.score_f1(sc, cam)      # composite realized (interference included)
            row[f"{s}_composite"] = comp
            row[f"{s}_moved_blocks_response"] = int((np.abs(off_r).sum(axis=1) > 0).sum())
            row[f"{s}_blockwise_bound_flips_fixed"] = int((best[s] - zero_ag).sum())
            row[f"{s}_offsets_response_lzma1"] = lzma1_raw(off_r.astype(np.int8).tobytes())
        row["wall_s"] = round(time.time() - t0, 1)
        out["rows"].append(row)
        checkpoint(args.out, out)
        print(f"[response] pair {p:3d} ({n+1}/{len(pairs)}) "
              f"| tok eta {row['tok_composite']['eta']:.4f} (bound {row['tok_blockwise_bound_flips_fixed']}) "
              f"| rgb eta {row['rgb_composite']['eta']:.4f} (bound {row['rgb_blockwise_bound_flips_fixed']}) "
              f"[{row['wall_s']}s]", flush=True)


# ================================================================================================
# subcommand: solve  (RUNG C context params + RUNG B sparse residual, encode-side solves)
# ================================================================================================
def run_solve(args, sc, dec, raw, gt_frames, pairs) -> None:
    import ddm_tr1_runtime as tr1

    out = base_header("solve")
    out["solver"] = {"steps": args.steps, "lr": args.lr, "eval_every": args.eval_every}
    out["rows"] = []
    segnet = sc.net.segnet

    def seg_forward(x_s: torch.Tensor) -> torch.Tensor:
        # x_s: (1,3,384,512) float scorer-lattice input (already resized) -> logits
        return segnet(x_s)

    for n, p in enumerate(pairs):
        t0 = time.time()
        ctx = PairCtx(sc, dec.packet, raw, gt_frames, p)
        tok = tr1.decode_token_grid(dec.packet, p)
        native = render_from_tokens(dec.packet, tok)
        cam_t = to_camera_u8(translate_native_blocks(native, ctx.off))     # A2 transport
        base_s = resize_to_scorer(cam_t)                                    # (1,3,384,512)
        mask_s = block_mask(ctx.off)                                        # moved blocks
        m = torch.from_numpy(mask_s)[None, None].float()
        tgt = torch.from_numpy(ctx.lgt.astype(np.int64))[None]
        nbx = SEG_W // BLOCK
        moved_idx = np.nonzero(np.abs(ctx.off).sum(axis=1) > 0)[0]
        n_moved = int(moved_idx.size)

        a2_score = ctx.score_f1(sc, cam_t)

        # ---------------- RUNG C: per-moved-block constant RGB shift ---------------------------
        blk_of_px = (np.arange(SEG_H)[:, None] // BLOCK) * nbx + (np.arange(SEG_W)[None, :] // BLOCK)
        moved_rank = {int(b): k for k, b in enumerate(moved_idx)}
        px_to_param = np.full((SEG_H, SEG_W), -1, dtype=np.int64)
        for b, k in moved_rank.items():
            px_to_param[blk_of_px == b] = k
        # camera-level param map, built ONCE per pair (vectorized realization)
        cam_param = np.full((CAM_H, CAM_W), -1, dtype=np.int64)
        for b, k in moved_rank.items():
            cam_param[scorer_mask_to_camera(blk_of_px == b)] = k
        cam_valid = cam_param >= 0

        def realize_shifts(shift_i8: np.ndarray, keep: np.ndarray) -> np.ndarray:
            """Apply per-block int8 RGB shifts (blocks in `keep`) to the transport at CAMERA res."""
            sh_full = np.zeros((n_moved, 3), dtype=np.int16)
            sh_full[keep] = shift_i8[keep].astype(np.int16)
            cam = cam_t.astype(np.int16).copy()
            cam[cam_valid] = cam[cam_valid] + sh_full[cam_param[cam_valid]]
            return np.clip(cam, 0, 255).astype(np.uint8)

        best_c = None
        with torch.enable_grad():
            sh = torch.zeros((n_moved, 3), requires_grad=True)
            opt = torch.optim.Adam([sh], lr=args.lr)
            pmap = torch.from_numpy(px_to_param)
            gather = torch.where(pmap[None, None] >= 0, pmap[None, None], 0)
            inband = (pmap >= 0)[None, None].float()
            for it in range(args.steps + 1):
                field = sh[gather[0, 0]].permute(2, 0, 1)[None] * inband
                cur = torch.clamp(base_s + field, 0.0, 255.0)
                if it % args.eval_every == 0 or it == args.steps:
                    s_i8 = np.clip(np.round(sh.detach().numpy()), -127, 127).astype(np.int8)
                    cam_e = realize_shifts(s_i8, np.arange(n_moved))
                    lam = sc.seg_argmax(np.stack([ctx.dec[0], cam_e]).astype(np.uint8))
                    fa = int((lam != ctx.lgt).sum())
                    if best_c is None or fa < best_c[0]:
                        best_c = (fa, s_i8.copy())
                if it == args.steps:
                    break
                loss = torch.nn.functional.cross_entropy(seg_forward(cur), tgt)
                opt.zero_grad()
                loss.backward()
                opt.step()

        fa_c, shifts = best_c
        # top-M selection by |shift| L1 (address+value counted per kept block)
        c_arms = {}
        order = np.argsort(-np.abs(shifts.astype(np.int32)).sum(axis=1))
        for M in (16, 32, n_moved):
            keep = order[:min(M, n_moved)]
            cam_e = realize_shifts(shifts, keep)
            scr = ctx.score_f1(sc, cam_e)
            payload = (np.sort(moved_idx[keep]).astype(np.uint16).tobytes()
                       + shifts[keep].astype(np.int8).tobytes())
            c_arms[f"M{M}"] = {**scr, "kept_blocks": int(len(keep)),
                               "payload_lzma1": lzma1_raw(payload),
                               "payload_raw": len(payload)}

        # ---------------- RUNG B: per-pixel sparse solved residual ----------------------------
        best_b = None
        with torch.enable_grad():
            delta = torch.zeros_like(base_s, requires_grad=True)
            opt = torch.optim.Adam([delta], lr=args.lr)
            for it in range(args.steps + 1):
                cur = torch.clamp(base_s + delta * m, 0.0, 255.0)
                if it % args.eval_every == 0 or it == args.steps:
                    q = torch.round(torch.clamp(base_s + delta.detach() * m, 0.0, 255.0))
                    paint = q[0].permute(1, 2, 0).numpy().astype(np.uint8)
                    cam_e = realize_scorer_paint_to_camera(cam_t, mask_s, paint)
                    lam = sc.seg_argmax(np.stack([ctx.dec[0], cam_e]).astype(np.uint8))
                    fa = int((lam != ctx.lgt).sum())
                    if best_b is None or fa < best_b[0]:
                        best_b = (fa, paint.copy())
                if it == args.steps:
                    break
                loss = torch.nn.functional.cross_entropy(seg_forward(cur), tgt)
                opt.zero_grad()
                loss.backward()
                opt.step()

        fa_b, paint = best_b
        base_round = np.clip(np.round(base_s[0].permute(1, 2, 0).numpy()), 0, 255).astype(np.int16)
        dmag = np.abs(paint.astype(np.int16) - base_round).sum(axis=2) * mask_s
        b_arms = {}
        for K in (32, 64, 128, 256):
            flat = dmag.ravel()
            keep_lin = np.argsort(-flat)[:K]
            keep_lin = keep_lin[flat[keep_lin] > 0]
            kmask = np.zeros(SEG_H * SEG_W, dtype=bool)
            kmask[keep_lin] = True
            kmask = kmask.reshape(SEG_H, SEG_W)
            cam_e = realize_scorer_paint_to_camera(cam_t, kmask, paint)
            scr = ctx.score_f1(sc, cam_e)
            ii, jj = np.nonzero(kmask)
            lin = (ii.astype(np.uint32) * SEG_W + jj.astype(np.uint32))
            payload = lin.astype("<u4").tobytes() + paint[kmask].astype(np.uint8).tobytes()
            b_arms[f"K{K}"] = {**scr, "kept_px": int(kmask.sum()),
                               "payload_lzma1": lzma1_raw(payload),
                               "payload_raw": len(payload)}
        # dense-band reference (unpayable, the eta ceiling of this solve)
        cam_e = realize_scorer_paint_to_camera(cam_t, mask_s, paint)
        scr_dense = ctx.score_f1(sc, cam_e)

        row = {"pair": p, "flips_before": ctx.flips0, "n_described": ctx.nd,
               "n_moved_blocks": n_moved,
               "A2_transport": a2_score,
               "C_arms": c_arms,
               "B_arms": b_arms,
               "B_dense_band": scr_dense,
               "d_pose_shipped": ctx.d_pose_shipped,
               "wall_s": round(time.time() - t0, 1)}
        out["rows"].append(row)
        checkpoint(args.out, out)
        cM = c_arms[f"M{n_moved}"]
        print(f"[solve] pair {p:3d} ({n+1}/{len(pairs)}) | A2 {a2_score['eta']:.3f} "
              f"| C(all) eta {cM['eta']:.3f} ({cM['payload_lzma1']}B) "
              f"| B dense eta {scr_dense['eta']:.3f} "
              f"| B K64 eta {b_arms['K64']['eta']:.3f} ({b_arms['K64']['payload_lzma1']}B) "
              f"[{row['wall_s']}s]", flush=True)


# ================================================================================================
# subcommand: solve0  (RUNG C0 -- per-block solved shifts WITHOUT the offset field)
# ================================================================================================
def run_solve0(args, sc, dec, raw, gt_frames, pairs) -> None:
    """C0: drop the transport AND the 57.8 KB offset field entirely.  Base = the SHIPPED render.
    Encode-side, select the top-`max M` blocks by per-block flip count (GT-informed selection is
    legal at encode; the block INDICES are counted payload), solve one RGB shift per block, and
    report nested top-M realizations.  Carrier per pair = M x (2 B index + 3 B shift) + pose
    stream.  This measures whether the phase field's target flips are carried cheaper by pure
    receiver-legal block paint than by transport+anything."""
    import ddm_tr1_runtime as tr1
    from ddm_sq1_eta_seg_realization import COL_SUP, ROW_SUP
    from ddm_sq1_pose_null_constrained_paint import pose_null_projector, project_null

    out = base_header("solve0")
    out["solver"] = {"steps": args.steps, "lr": args.lr, "eval_every": args.eval_every,
                     "m_max": args.m_max}
    out["rows"] = []
    segnet = sc.net.segnet
    nbx = SEG_W // BLOCK
    P12 = pose_null_projector()

    for n, p in enumerate(pairs):
        t0 = time.time()
        ctx = PairCtx(sc, dec.packet, raw, gt_frames, p)
        cam_base = tr1.render_frame1_camera_uint8(dec.packet, p)     # == shipped frame_1 (A0)
        base_s = resize_to_scorer(cam_base)
        tgt = torch.from_numpy(ctx.lgt.astype(np.int64))[None]
        nby = SEG_H // BLOCK

        flips_blk = (ctx.lstar != ctx.lgt).reshape(nby, BLOCK, nbx, BLOCK).sum(axis=(1, 3))
        sel_blocks = np.argsort(-flips_blk.ravel())[:args.m_max]
        n_sel = int(sel_blocks.size)
        blk_of_px = (np.arange(SEG_H)[:, None] // BLOCK) * nbx + (np.arange(SEG_W)[None, :] // BLOCK)
        px_to_param = np.full((SEG_H, SEG_W), -1, dtype=np.int64)
        for k, b in enumerate(sel_blocks):
            px_to_param[blk_of_px == b] = k
        cam_param = np.full((CAM_H, CAM_W), -1, dtype=np.int64)
        for k, b in enumerate(sel_blocks):
            cam_param[scorer_mask_to_camera(blk_of_px == b)] = k
        cam_valid = cam_param >= 0

        def realize_shifts(shift_i8: np.ndarray, keep: np.ndarray) -> np.ndarray:
            sh_full = np.zeros((n_sel, 3), dtype=np.int16)
            sh_full[keep] = shift_i8[keep].astype(np.int16)
            cam = cam_base.astype(np.int16).copy()
            cam[cam_valid] = cam[cam_valid] + sh_full[cam_param[cam_valid]]
            return np.clip(cam, 0, 255).astype(np.uint8)

        pmap = torch.from_numpy(px_to_param)
        gather = torch.where(pmap >= 0, pmap, 0)
        inband = (pmap >= 0)[None, None].float()

        def broadcast_field(sh_t: torch.Tensor) -> torch.Tensor:
            return sh_t[gather].permute(2, 0, 1)[None] * inband

        def realize_null(shift_i8: np.ndarray, keep: np.ndarray) -> np.ndarray:
            """Receiver path of the NULL arm: int8 params -> broadcast -> rank-6 projector ->
            float scorer-lattice delta -> ADDITIVE write on each scorer px's 4 private camera
            support px -> round once.  Deterministic, scorer-free, generic (rule 118)."""
            sh_full = np.zeros((n_sel, 3), dtype=np.float32)
            sh_full[keep] = shift_i8[keep].astype(np.float32)
            field = torch.from_numpy(sh_full)[gather].permute(2, 0, 1)[None] * inband
            d = project_null(field, P12)[0].permute(1, 2, 0).numpy()      # (384,512,3)
            cam = cam_base.astype(np.float32).copy()
            ii, jj = np.nonzero(px_to_param >= 0)
            vals = d[ii, jj]
            for a in range(2):
                r = ROW_SUP[ii, a]
                for b in range(2):
                    c = COL_SUP[jj, b]
                    cam[r, c] = cam[r, c] + vals
            return np.clip(np.rint(cam), 0, 255).astype(np.uint8)

        def solve_params(use_null: bool):
            best = None
            with torch.enable_grad():
                sh = torch.zeros((n_sel, 3), requires_grad=True)
                opt = torch.optim.Adam([sh], lr=args.lr)
                for it in range(args.steps + 1):
                    field = broadcast_field(sh)
                    if use_null:
                        field = project_null(field, P12)
                    cur = torch.clamp(base_s + field, 0.0, 255.0)
                    if it % args.eval_every == 0 or it == args.steps:
                        s_i8 = np.clip(np.round(sh.detach().numpy()), -127, 127).astype(np.int8)
                        cam_e = (realize_null if use_null else realize_shifts)(
                            s_i8, np.arange(n_sel))
                        lam = sc.seg_argmax(np.stack([ctx.dec[0], cam_e]).astype(np.uint8))
                        fa = int((lam != ctx.lgt).sum())
                        if best is None or fa < best[0]:
                            best = (fa, s_i8.copy())
                    if it == args.steps:
                        break
                    loss = torch.nn.functional.cross_entropy(segnet(cur), tgt)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            return best

        arms = {}
        for use_null, sfx in ((False, ""), (True, "_null")):
            _fa, shifts = solve_params(use_null)
            realize = realize_null if use_null else realize_shifts
            gain_proxy = (np.abs(shifts.astype(np.int32)).sum(axis=1)
                          * flips_blk.ravel()[sel_blocks])
            order = np.argsort(-gain_proxy)
            for M in (8, 16, 32, args.m_max):
                keep = order[:min(M, n_sel)]
                region = np.isin(blk_of_px, sel_blocks[keep])
                scr = ctx.score_f1(sc, realize(shifts, keep), region=region)
                payload = (np.sort(sel_blocks[keep]).astype(np.uint16).tobytes()
                           + shifts[keep].astype(np.int8).tobytes())
                arms[f"M{M}{sfx}"] = {**scr, "kept_blocks": int(len(keep)),
                                      "payload_lzma1": lzma1_raw(payload),
                                      "payload_raw": len(payload)}

        row = {"pair": p, "flips_before": ctx.flips0, "n_described": ctx.nd,
               "selected_blocks": n_sel, "arms": arms,
               "d_pose_shipped": ctx.d_pose_shipped,
               "wall_s": round(time.time() - t0, 1)}
        out["rows"].append(row)
        checkpoint(args.out, out)
        a32 = arms["M32"]
        print(f"[solve0] pair {p:3d} ({n+1}/{len(pairs)}) nd {ctx.nd:5d} "
              f"| M8 {arms['M8']['eta']:.3f} | M16 {arms['M16']['eta']:.3f} "
              f"| M32 {a32['eta']:.3f} ({a32['payload_lzma1']}B) "
              f"| M{args.m_max} {arms[f'M{args.m_max}']['eta']:.3f} "
              f"[{row['wall_s']}s]", flush=True)


# ================================================================================================
# subcommand: keys  (address-key A/B for rung C0 -- cg1's named unknown, coordinator fold-in)
# ================================================================================================
def run_keys(args, sc, dec, raw, gt_frames, pairs) -> None:
    """Same C0 solve at M=32 under THREE address keys:
      GT     per-pair top-32 blocks by flips (encode-side key; measured in solve0, re-run here
             for an exactly-matched A/B on the same solver instance),
      STATIC top-32 blocks by sg3's static risk mass (pixels that EVER flip across n600,
             reconstructed from the argmax caches; ONE fixed block list, ~64 B shipped once),
      PROXY  top-32 blocks by DECODER-DERIVABLE edge energy of the rendered frame (0 B address;
             receiver re-derives the ranking deterministically from its own render, rule 118).
    Reports per key: realized eta, collateral rate (sg3's 1.035% discriminator), flip-capture
    of the block set, d_pose ratio, and payload bytes."""
    import ddm_tr1_runtime as tr1

    nby, nbx = SEG_H // BLOCK, SEG_W // BLOCK
    argmax_dir = Path("/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache")
    gt_c = np.load(argmax_dir / "gt_argmax_n600.npy", mmap_mode="r")
    cx_c = np.load(argmax_dir / "cx1_argmax_n600.npy", mmap_mode="r")
    static_set = np.zeros((SEG_H, SEG_W), dtype=bool)
    for q in range(gt_c.shape[0]):
        static_set |= np.asarray(gt_c[q]) != np.asarray(cx_c[q])
    static_px = int(static_set.sum())
    static_mass = static_set.reshape(nby, BLOCK, nbx, BLOCK).sum(axis=(1, 3)).ravel()
    static_top = np.argsort(-static_mass)[: args.m_keys]

    out = base_header("keys")
    out["solver"] = {"steps": args.steps, "lr": args.lr, "eval_every": args.eval_every,
                     "m_keys": args.m_keys}
    out["static_set_px"] = static_px            # sg3 cross-receipt control: 43,798
    out["static_top_blocks"] = [int(b) for b in static_top]
    out["rows"] = []
    segnet = sc.net.segnet
    blk_of_px = (np.arange(SEG_H)[:, None] // BLOCK) * nbx + (np.arange(SEG_W)[None, :] // BLOCK)

    for n, p in enumerate(pairs):
        t0 = time.time()
        ctx = PairCtx(sc, dec.packet, raw, gt_frames, p)
        cam_base = tr1.render_frame1_camera_uint8(dec.packet, p)
        native = render_from_tokens(dec.packet, tr1.decode_token_grid(dec.packet, p))
        base_s = resize_to_scorer(cam_base)
        tgt = torch.from_numpy(ctx.lgt.astype(np.int64))[None]
        flips_map = ctx.lstar != ctx.lgt
        flips_blk = flips_map.reshape(nby, BLOCK, nbx, BLOCK).sum(axis=(1, 3)).ravel()

        # decoder-derivable proxy: edge energy of the rendered native luma (deterministic fp32)
        luma = (0.299 * native[..., 0] + 0.587 * native[..., 1]
                + 0.114 * native[..., 2]).astype(np.float32)
        en = np.zeros_like(luma)
        en[:-1, :] += np.abs(np.diff(luma, axis=0))
        en[:, :-1] += np.abs(np.diff(luma, axis=1))
        proxy_mass = en.reshape(nby, BLOCK, nbx, BLOCK).sum(axis=(1, 3)).ravel()

        keysets = {
            "gt": np.argsort(-flips_blk)[: args.m_keys],
            "static": static_top,
            "proxy": np.argsort(-proxy_mass)[: args.m_keys],
        }
        row = {"pair": p, "flips_before": ctx.flips0, "n_described": ctx.nd,
               "d_pose_shipped": ctx.d_pose_shipped, "arms": {}}
        for key, sel_blocks in keysets.items():
            sel_blocks = np.asarray(sel_blocks, dtype=np.int64)
            region = np.isin(blk_of_px, sel_blocks)
            capture = float(flips_map[region].sum() / max(ctx.flips0, 1))
            px_to_param = np.full((SEG_H, SEG_W), -1, dtype=np.int64)
            for k, b in enumerate(sel_blocks):
                px_to_param[blk_of_px == b] = k
            cam_param = np.full((CAM_H, CAM_W), -1, dtype=np.int64)
            for k, b in enumerate(sel_blocks):
                cam_param[scorer_mask_to_camera(blk_of_px == b)] = k
            cam_valid = cam_param >= 0
            n_sel = int(sel_blocks.size)

            def realize(shift_i8: np.ndarray) -> np.ndarray:
                cam = cam_base.astype(np.int16).copy()
                cam[cam_valid] = cam[cam_valid] + shift_i8.astype(np.int16)[cam_param[cam_valid]]
                return np.clip(cam, 0, 255).astype(np.uint8)

            pmap = torch.from_numpy(px_to_param)
            gather = torch.where(pmap >= 0, pmap, 0)
            inband = (pmap >= 0)[None, None].float()
            best = None
            with torch.enable_grad():
                sh = torch.zeros((n_sel, 3), requires_grad=True)
                opt = torch.optim.Adam([sh], lr=args.lr)
                for it in range(args.steps + 1):
                    cur = torch.clamp(base_s + sh[gather].permute(2, 0, 1)[None] * inband,
                                      0.0, 255.0)
                    if it % args.eval_every == 0 or it == args.steps:
                        s_i8 = np.clip(np.round(sh.detach().numpy()), -127, 127).astype(np.int8)
                        lam = sc.seg_argmax(np.stack([ctx.dec[0], realize(s_i8)]).astype(np.uint8))
                        fa = int((lam != ctx.lgt).sum())
                        if best is None or fa < best[0]:
                            best = (fa, s_i8.copy())
                    if it == args.steps:
                        break
                    loss = torch.nn.functional.cross_entropy(segnet(cur), tgt)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            _fa, shifts = best
            scr = ctx.score_f1(sc, realize(shifts), region=region)
            payload = shifts.astype(np.int8).tobytes()      # params ONLY, order = key rank
            row["arms"][key] = {**scr, "capture_of_pair_flips": capture,
                                "params_lzma1": lzma1_raw(payload),
                                "params_raw": len(payload)}
        row["wall_s"] = round(time.time() - t0, 1)
        out["rows"].append(row)
        checkpoint(args.out, out)
        a = row["arms"]
        print(f"[keys] pair {p:3d} ({n+1}/{len(pairs)}) "
              f"| gt eta {a['gt']['eta']:.3f} cap {a['gt']['capture_of_pair_flips']:.2f} "
              f"| static eta {a['static']['eta']:.3f} cap {a['static']['capture_of_pair_flips']:.2f} "
              f"| proxy eta {a['proxy']['eta']:.3f} cap {a['proxy']['capture_of_pair_flips']:.2f} "
              f"[{row['wall_s']}s]", flush=True)


# ================================================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("transport", "response", "solve", "solve0", "keys"))
    ap.add_argument("--pairs", type=str, default="0,20,48,115,154,170,179,180")
    ap.add_argument("--gt-mkv", type=Path, default=REPO / "upstream/videos/0.mkv")
    ap.add_argument("--raw", type=Path, default=SUB_PU2 / "inflated/0.raw")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--rmax", type=int, default=5)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--lr", type=float, default=4.0)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--m-max", type=int, default=64)
    ap.add_argument("--m-keys", type=int, default=32)
    args = ap.parse_args()

    t0 = time.time()
    pairs = [int(x) for x in args.pairs.split(",")]
    raw = np.memmap(args.raw, dtype=np.uint8, mode="r",
                    shape=(N_PAIRS_TOTAL * seq_len, CAM_H, CAM_W, 3))
    wanted = set()
    for p in pairs:
        wanted.update({seq_len * p, seq_len * p + 1})
    gt_frames = decode_gt_frames(args.gt_mkv, wanted)
    sc = Scorer(args.threads)
    dec = load_decoder()
    print(f"[lr2:{args.cmd}] ready t={time.time()-t0:.1f}s pairs={pairs}", flush=True)

    if args.cmd == "transport":
        run_transport(args, sc, dec, raw, gt_frames, pairs)
    elif args.cmd == "response":
        run_response(args, sc, dec, raw, gt_frames, pairs)
    elif args.cmd == "solve0":
        run_solve0(args, sc, dec, raw, gt_frames, pairs)
    elif args.cmd == "keys":
        run_keys(args, sc, dec, raw, gt_frames, pairs)
    else:
        run_solve(args, sc, dec, raw, gt_frames, pairs)
    print(f"[lr2:{args.cmd}] done t={time.time()-t0:.1f}s -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
