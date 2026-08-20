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
sys.path.insert(0, str(REPO / "src"))

from tac.payload_retention import (  # noqa: E402
    portable_retention_record,
    resolve_portable_path,
    retain_payload,
    retention_root,
)

SUB_PU2 = resolve_portable_path("$PACT_TIER1/ddm_pu2_20260803/submission_pu2")
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
OUT_DIR = resolve_portable_path("$PACT_TIER1/ddm_lr2_20260804")


def lzma1_raw(b: bytes) -> int:
    """Price ``b`` under the shipped LZMA1 filter chain. PURE ORACLE — prices, never owns.

    Retention is discharged by the CALLER: every arm routes its carrier through
    :func:`retain_arm_payload` before pricing it, so the bytes this function measures are
    already on the SSD tier with a recorded sha256.  Persisting inside the oracle would
    re-write the same payload once per ladder rung and per K/M sweep step, which is noise,
    not custody.
    """
    filt = [{"id": lzma.FILTER_LZMA1, "dict_size": 1 << 22, "lc": 0, "lp": 0, "pb": 0}]
    return len(lzma.compress(b, format=lzma.FORMAT_RAW, filters=filt))  # MEASURE_ONLY_OK: pure pricing oracle; every caller retains the carrier via retain_arm_payload before pricing it


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


def retain_arm_payload(arm: str, pair: int, payload: bytes) -> dict:
    """Persist ONE candidate carrier and return its custody row.

    ALWAYS KEEP THE PAYLOAD (P0, operator 2026-08-09).  Every arm of this ladder solves a
    real carrier -- block-selection indices plus int8 shifts -- prices it with LZMA1, and
    used to drop the bytes on the floor, leaving a JSON of lengths.  That is the anchor
    incident verbatim: ``ans_real_n600.py`` kept two coder LENGTHS and the discarded loser
    later measured -2,120 B BETTER than the shipped winner.  So this keeps EVERY arm's
    bytes, not the best one's, and records sha256 so the next consumer can prove
    byte-identity instead of re-solving a 15 s/pair Adam descent.

    Routing is delegated to :func:`retention_root`, never hardcoded: this module's
    ``OUT_DIR`` points at VertigoDataTier, which measured 893 MiB free (100% capacity) on
    2026-08-16, so a fixed first-tier write fails mid-run.
    """
    root = retention_root("ddm_lr2", need_bytes=len(payload)) / f"pair{pair:03d}"
    return portable_retention_record(
        retain_payload(root / f"{arm}.carrier.bin", payload)
    )


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
            custody = retain_arm_payload(f"rungC_M{M}", p, payload)
            c_arms[f"M{M}"] = {**scr, "kept_blocks": int(len(keep)),
                               "payload_lzma1": lzma1_raw(payload),
                               "payload_raw": len(payload),
                               "carrier_custody": custody}

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
            custody = retain_arm_payload(f"rungB_K{K}", p, payload)
            b_arms[f"K{K}"] = {**scr, "kept_px": int(kmask.sum()),
                               "payload_lzma1": lzma1_raw(payload),
                               "payload_raw": len(payload),
                               "carrier_custody": custody}
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
                arm = f"M{M}{sfx}"
                custody = retain_arm_payload(arm, p, payload)
                arms[arm] = {**scr, "kept_blocks": int(len(keep)),
                             "payload_lzma1": lzma1_raw(payload),
                             "payload_raw": len(payload),
                             "carrier_custody": custody}

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
    argmax_dir = resolve_portable_path("$PACT_TIER1/ddm_pu2_20260803/argmax_cache")
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
            custody = retain_arm_payload(f"keys_{key}", p, payload)
            row["arms"][key] = {**scr, "capture_of_pair_flips": capture,
                                "params_lzma1": lzma1_raw(payload),
                                "params_raw": len(payload),
                                "carrier_custody": custody}
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
# subcommand: fo1  (fire-order-1: budget x M sweep on the STATIC key; U vs AC arms)
# ================================================================================================
def run_fo1(args, sc, dec, raw, gt_frames, pairs) -> None:
    """FO-1 (coordinator-fired): the +0.0045 S static-cell gap on an UNCAPPED solver.

    Arms per (pair, M):
      U   unconstrained per-block DC shifts (3 int8/block), carrier includes the 57,600 B
          frame_0 pose stream (DC is 100% pose-visible -- the AC-only law).
      AC  per-block coefficients on a CONTENT-ADAPTIVE AC atom pushed through the rank-6
          frame_1 yuv6-null projector: delta = P( c_b * g_b ), g_b = the block's own rendered
          zero-mean unit-normalised luma pattern (receiver re-derives g_b deterministically
          from its own render; P is free code; only c_b ships -- 3 int8/block, SAME rate as U).
          Pose-neutral BY CONSTRUCTION up to uint8 rounding (m85 integer-actuator caveat --
          measured, not assumed); carrier EXCLUDES the pose stream.
    Solver: Adam, realized-argmax best-iterate, EARLY-STOP on realized-flip patience (never a
    step bound as the stop); the convergence curve ships in the receipt so 'floor' claims die.
    Entropy trim: params re-quantised to step-2 / step-4 int8 lattices, re-realized, re-scored,
    LZMA1-priced (quantization-toolbox 'aware' form: quantise -> re-eval realized).
    """
    import ddm_tr1_runtime as tr1
    from ddm_sq1_eta_seg_realization import COL_SUP, ROW_SUP
    from ddm_sq1_pose_null_constrained_paint import pose_null_projector, project_null

    nby, nbx = SEG_H // BLOCK, SEG_W // BLOCK
    argmax_dir = resolve_portable_path("$PACT_TIER1/ddm_pu2_20260803/argmax_cache")
    gt_c = np.load(argmax_dir / "gt_argmax_n600.npy", mmap_mode="r")
    cx_c = np.load(argmax_dir / "cx1_argmax_n600.npy", mmap_mode="r")
    static_set = np.zeros((SEG_H, SEG_W), dtype=bool)
    for q in range(gt_c.shape[0]):
        static_set |= np.asarray(gt_c[q]) != np.asarray(cx_c[q])
    static_mass = static_set.reshape(nby, BLOCK, nbx, BLOCK).sum(axis=(1, 3)).ravel()
    m_list = [int(x) for x in args.m_list.split(",")]
    P12 = pose_null_projector()

    out = base_header("fo1")
    out["solver"] = {"max_steps": args.max_steps, "lr": args.lr,
                     "eval_every": args.eval_every, "patience_evals": args.patience}
    out["static_set_px"] = int(static_set.sum())
    out["m_list"] = m_list
    out["rows"] = []
    segnet = sc.net.segnet
    blk_of_px = (np.arange(SEG_H)[:, None] // BLOCK) * nbx + (np.arange(SEG_W)[None, :] // BLOCK)

    for n, p in enumerate(pairs):
        t_pair = time.time()
        ctx = PairCtx(sc, dec.packet, raw, gt_frames, p)
        cam_base = tr1.render_frame1_camera_uint8(dec.packet, p)
        base_s = resize_to_scorer(cam_base)
        tgt = torch.from_numpy(ctx.lgt.astype(np.int64))[None]
        luma_s = (0.299 * base_s[0, 0] + 0.587 * base_s[0, 1] + 0.114 * base_s[0, 2]).numpy()

        row = {"pair": p, "flips_before": ctx.flips0, "n_described": ctx.nd,
               "d_pose_shipped": ctx.d_pose_shipped, "cells": {}}
        for M in m_list:
            sel_blocks = np.argsort(-static_mass)[:M].astype(np.int64)
            region = np.isin(blk_of_px, sel_blocks)
            px_to_param = np.full((SEG_H, SEG_W), -1, dtype=np.int64)
            for k, b in enumerate(sel_blocks):
                px_to_param[blk_of_px == b] = k
            pmap = torch.from_numpy(px_to_param)
            gather = torch.where(pmap >= 0, pmap, 0)
            inband = (pmap >= 0)[None, None].float()
            cam_param = np.full((CAM_H, CAM_W), -1, dtype=np.int64)
            for k, b in enumerate(sel_blocks):
                cam_param[scorer_mask_to_camera(blk_of_px == b)] = k
            cam_valid = cam_param >= 0
            # AC atom per block: zero-mean unit-max luma pattern of the block's own render
            g = np.zeros((SEG_H, SEG_W), dtype=np.float32)
            for b in sel_blocks:
                mask_b = blk_of_px == b
                v = luma_s[mask_b]
                v = v - v.mean()
                amp = np.abs(v).max()
                g[mask_b] = (v / amp) if amp > 1e-6 else 0.0
            g_t = torch.from_numpy(g)[None, None]

            def field_of(sh_t, ac: bool):
                f = sh_t[gather].permute(2, 0, 1)[None] * inband
                if ac:
                    return project_null(f * g_t, P12)
                return f

            def realize(shift_i8: np.ndarray, ac: bool) -> np.ndarray:
                if not ac:
                    cam = cam_base.astype(np.int16).copy()
                    cam[cam_valid] = (cam[cam_valid]
                                      + shift_i8.astype(np.int16)[cam_param[cam_valid]])
                    return np.clip(cam, 0, 255).astype(np.uint8)
                sh_t = torch.from_numpy(shift_i8.astype(np.float32))
                d = field_of(sh_t, True)[0].permute(1, 2, 0).numpy()
                cam = cam_base.astype(np.float32).copy()
                ii, jj = np.nonzero(px_to_param >= 0)
                vals = d[ii, jj]
                for a in range(2):
                    r = ROW_SUP[ii, a]
                    for b2 in range(2):
                        c2 = COL_SUP[jj, b2]
                        cam[r, c2] = cam[r, c2] + vals
                return np.clip(np.rint(cam), 0, 255).astype(np.uint8)

            arm_list = [a.strip() for a in args.arms.split(",")]
            for ac, arm in ((False, "U"), (True, "AC")):
                if arm not in arm_list:
                    continue
                best = None
                curve = []
                stop_reason = "cap"
                with torch.enable_grad():
                    sh = torch.zeros((M, 3), requires_grad=True)
                    opt = torch.optim.Adam([sh], lr=args.lr)
                    since_best = 0
                    for it in range(args.max_steps + 1):
                        cur = torch.clamp(base_s + field_of(sh, ac), 0.0, 255.0)
                        if it % args.eval_every == 0 or it == args.max_steps:
                            s_i8 = np.clip(np.round(sh.detach().numpy()), -127, 127
                                           ).astype(np.int8)
                            lam = sc.seg_argmax(np.stack([ctx.dec[0], realize(s_i8, ac)]
                                                         ).astype(np.uint8))
                            fa = int((lam != ctx.lgt).sum())
                            curve.append([it, fa])
                            if best is None or fa < best[0]:
                                best = (fa, s_i8.copy())
                                since_best = 0
                            else:
                                since_best += 1
                            if since_best >= args.patience:
                                stop_reason = "converged"
                                break
                        if it == args.max_steps:
                            break
                        loss = torch.nn.functional.cross_entropy(segnet(cur), tgt)
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                _fa, shifts = best
                depths = {}
                for dname, step_q in (("int8", 1), ("step2", 2), ("step4", 4)):
                    sq = (np.round(shifts.astype(np.float32) / step_q) * step_q
                          ).astype(np.int8)
                    scr = ctx.score_f1(sc, realize(sq, ac), region=region)
                    depths[dname] = {**scr,
                                     "params_lzma1": lzma1_raw(sq.astype(np.int8).tobytes()),
                                     "params_raw": int(sq.size)}
                row["cells"][f"M{M}_{arm}"] = {
                    "depths": depths, "convergence_curve": curve,
                    "stop_reason": stop_reason, "steps_run": curve[-1][0]}
        row["wall_s"] = round(time.time() - t_pair, 1)
        out["rows"].append(row)
        checkpoint(args.out, out)
        parts = []
        for cell, c8 in row["cells"].items():
            parts.append(f"{cell} {c8['depths']['int8']['eta']:.3f}"
                         f"@{c8['steps_run']}s/{c8['stop_reason'][:4]}")
        print(f"[fo1] pair {p:3d} ({n+1}/{len(pairs)}) | " + " | ".join(parts)
              + f" [{row['wall_s']}s]", flush=True)


# ================================================================================================
# subcommand: tx  (operator steer: FEATURE-BEARING paint arm + band-address arm + collateral
#                  spatial profile; sg3 corrected crossover 0.6285% measured alongside 1.035%)
# ================================================================================================
def _region_boundary_rings(region: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rings around the edit-region boundary: (within 1 px, within 2-4 px), both sides."""
    from ddm_sq1_eta_seg_realization import dilate

    near = dilate(region, 1) & dilate(~region, 1)              # <=1 px from the boundary
    mid = dilate(region, 4) & dilate(~region, 4) & ~near       # 2..4 px from the boundary
    return near, mid


def run_tx(args, sc, dec, raw, gt_frames, pairs) -> None:
    """Race, at matched accounting on the same pairs (each arm vs its OWN bar):
      U_flat   per-block DC shifts, static key M32 (3 int8/blk; + pose stream)   [re-solve]
      TX       FEATURE-BEARING pose-null paint: per-block coefficients on k=4 zero-mean AC
               atoms (the block's own luma pattern + DCT(0,1),(1,0),(1,1)) through the rank-6
               projector.  12 int8/blk; NO pose stream (AC-only law).  M16 direct + nested M8.
      BAND     Road<->Lane boundary-band arm (sg3's 4th address): region = r=1 dilation of the
               live field's Road<->Lane interface; FREE per-px solved paint (sq1 mechanics).
               Priced at sg3's 81,365 B band address + pose stream.
    Diagnostic per arm: collateral spatial profile -- introduced flips within 1 px / 2-4 px of
    the edit-region boundary vs farther (the operator's flat-paint-discontinuity mechanism).
    Collateral reported against BOTH crossover operating points (1.035% superseded; 0.6285%
    corrected, sg3 2583e0f155)."""
    import ddm_tr1_runtime as tr1
    from ddm_sq1_eta_seg_realization import dilate
    from ddm_sq1_pose_null_constrained_paint import pose_null_projector, project_null
    from ddm_sq1_stage_decomposition_and_solved_paint import solve_margin_optimal_paint

    nby, nbx = SEG_H // BLOCK, SEG_W // BLOCK
    argmax_dir = resolve_portable_path("$PACT_TIER1/ddm_pu2_20260803/argmax_cache")
    gt_c = np.load(argmax_dir / "gt_argmax_n600.npy", mmap_mode="r")
    cx_c = np.load(argmax_dir / "cx1_argmax_n600.npy", mmap_mode="r")
    static_set = np.zeros((SEG_H, SEG_W), dtype=bool)
    for q in range(gt_c.shape[0]):
        static_set |= np.asarray(gt_c[q]) != np.asarray(cx_c[q])
    static_mass = static_set.reshape(nby, BLOCK, nbx, BLOCK).sum(axis=(1, 3)).ravel()
    P12 = pose_null_projector()

    # fixed zero-mean unit-max DCT atoms over a 16x16 block (deterministic, rule-118 free)
    yy, xx = np.meshgrid(np.arange(BLOCK), np.arange(BLOCK), indexing="ij")
    dct_atoms = []
    for u, v in ((0, 1), (1, 0), (1, 1)):
        at = (np.cos(np.pi * (yy + 0.5) * u / BLOCK)
              * np.cos(np.pi * (xx + 0.5) * v / BLOCK)).astype(np.float32)
        at -= at.mean()
        dct_atoms.append(at / np.abs(at).max())

    out = base_header("tx")
    out["solver"] = {"max_steps": args.max_steps, "lr": args.lr,
                     "eval_every": args.eval_every, "patience_evals": args.patience}
    out["band_address_bytes_sg3"] = 81_365
    out["crossovers"] = {"superseded": 0.01035, "corrected_2583e0f155": 0.006285}
    out["rows"] = []
    segnet = sc.net.segnet
    blk_of_px = (np.arange(SEG_H)[:, None] // BLOCK) * nbx + (np.arange(SEG_W)[None, :] // BLOCK)

    def profile(scr_region: np.ndarray, lam: np.ndarray, ctx) -> dict:
        near, mid = _region_boundary_rings(scr_region)
        intro = (ctx.lstar == ctx.lgt) & (lam != ctx.lgt)
        tot = int(intro.sum())
        return {"introduced_total": tot,
                "introduced_within_1px_of_region_boundary": int((intro & near).sum()),
                "introduced_2_to_4px": int((intro & mid).sum()),
                "introduced_beyond_4px": int((intro & ~(near | mid)).sum())}

    for n, p in enumerate(pairs):
        t_pair = time.time()
        ctx = PairCtx(sc, dec.packet, raw, gt_frames, p)
        cam_base = tr1.render_frame1_camera_uint8(dec.packet, p)
        base_s = resize_to_scorer(cam_base)
        tgt = torch.from_numpy(ctx.lgt.astype(np.int64))[None]
        luma_s = (0.299 * base_s[0, 0] + 0.587 * base_s[0, 1] + 0.114 * base_s[0, 2]).numpy()
        row = {"pair": p, "flips_before": ctx.flips0, "n_described": ctx.nd,
               "d_pose_shipped": ctx.d_pose_shipped, "arms": {}}

        # ---------------- shared static-key block machinery (M = 32 superset) -----------------
        sel32 = np.argsort(-static_mass)[:32].astype(np.int64)
        region32 = np.isin(blk_of_px, sel32)
        px_to_param = np.full((SEG_H, SEG_W), -1, dtype=np.int64)
        for k, b in enumerate(sel32):
            px_to_param[blk_of_px == b] = k
        pmap = torch.from_numpy(px_to_param)
        gather = torch.where(pmap >= 0, pmap, 0)
        inband = (pmap >= 0)[None, None].float()
        cam_param = np.full((CAM_H, CAM_W), -1, dtype=np.int64)
        for k, b in enumerate(sel32):
            cam_param[scorer_mask_to_camera(blk_of_px == b)] = k
        cam_valid = cam_param >= 0

        # per-block atom stack: (32, 4, H, W) sparse via block masks; built as full-frame fields
        atom_fields = np.zeros((4, SEG_H, SEG_W), dtype=np.float32)
        for k, b in enumerate(sel32):
            mask_b = blk_of_px == b
            v = luma_s[mask_b].reshape(BLOCK, BLOCK)
            v = v - v.mean()
            amp = np.abs(v).max()
            atom_fields[0][mask_b] = (v / amp).ravel() if amp > 1e-6 else 0.0
            for a in range(3):
                atom_fields[1 + a][mask_b] = dct_atoms[a].ravel()
        atoms_t = torch.from_numpy(atom_fields)[None]            # (1,4,H,W)

        def solve_arm(make_field, n_par, realize, label):
            best, curve, stop = None, [], "cap"
            with torch.enable_grad():
                cc = torch.zeros(n_par, requires_grad=True)
                opt = torch.optim.Adam([cc], lr=args.lr)
                since = 0
                for it in range(args.max_steps + 1):
                    cur = torch.clamp(base_s + make_field(cc), 0.0, 255.0)
                    if it % args.eval_every == 0 or it == args.max_steps:
                        c_i8 = np.clip(np.round(cc.detach().numpy()), -127, 127).astype(np.int8)
                        lam = sc.seg_argmax(np.stack([ctx.dec[0], realize(c_i8)]
                                                     ).astype(np.uint8))
                        fa = int((lam != ctx.lgt).sum())
                        curve.append([it, fa])
                        if best is None or fa < best[0]:
                            best, since = (fa, c_i8.copy()), 0
                        else:
                            since += 1
                        if since >= args.patience:
                            stop = "converged"
                            break
                    if it == args.max_steps:
                        break
                    loss = torch.nn.functional.cross_entropy(segnet(cur), tgt)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
            return best, curve, stop

        # ---------------- U_flat M32 (re-solve; keeps params for the diagnostic) --------------
        def u_field(cc):
            return cc.reshape(32, 3)[gather].permute(2, 0, 1)[None] * inband

        def u_realize(c_i8):
            cam = cam_base.astype(np.int16).copy()
            cam[cam_valid] = (cam[cam_valid]
                              + c_i8.reshape(32, 3).astype(np.int16)[cam_param[cam_valid]])
            return np.clip(cam, 0, 255).astype(np.uint8)

        (fa_u, cu), curve_u, stop_u = solve_arm(u_field, (32, 3), u_realize, "U")
        cam_u = u_realize(cu)
        lam_u = sc.seg_argmax(np.stack([ctx.dec[0], cam_u]).astype(np.uint8))
        scr_u = ctx.score_f1(sc, cam_u, region=region32)
        row["arms"]["U_flat_M32"] = {
            **scr_u, "params_lzma1": lzma1_raw(cu.astype(np.int8).tobytes()),
            "profile": profile(region32, lam_u, ctx), "stop_reason": stop_u,
            "steps_run": curve_u[-1][0]}

        # ---------------- TX M16 direct (+ nested M8), 12 int8/blk, through P -----------------
        sel16 = np.argsort(-static_mass)[:16].astype(np.int64)
        keep16 = np.isin(sel32, sel16)
        keep16_t = torch.from_numpy(keep16.astype(np.float32))[:, None, None]

        def tx_field_np(c_np: np.ndarray) -> torch.Tensor:
            return tx_field_t(torch.from_numpy(c_np.reshape(-1).astype(np.float32)),
                              keep=None)

        def tx_field_t(cc: torch.Tensor, keep) -> torch.Tensor:
            c = cc.reshape(32, 4, 3)
            if keep is not None:
                c = c * keep
            f = torch.zeros(1, 3, SEG_H, SEG_W)
            for a in range(4):
                coef = c[:, a, :][gather].permute(2, 0, 1)[None] * inband
                f = f + coef * atoms_t[:, a:a + 1]
            return project_null(f, P12)

        def tx_mask(cc_np, keep_mask):
            c = cc_np.reshape(32, 4, 3).astype(np.float32).copy()
            c[~keep_mask] = 0.0
            return c

        def tx_realize_from(c_np):
            from ddm_sq1_eta_seg_realization import COL_SUP, ROW_SUP

            d = tx_field_np(c_np)[0].permute(1, 2, 0).numpy()
            cam = cam_base.astype(np.float32).copy()
            ii, jj = np.nonzero(px_to_param >= 0)
            vals = d[ii, jj]
            for a2 in range(2):
                r = ROW_SUP[ii, a2]
                for b2 in range(2):
                    c2 = COL_SUP[jj, b2]
                    cam[r, c2] = cam[r, c2] + vals
            return np.clip(np.rint(cam), 0, 255).astype(np.uint8)

        def tx_field_m16(cc):
            return tx_field_t(cc, keep16_t)

        def tx_realize_m16(c_i8):
            return tx_realize_from(tx_mask(c_i8, keep16))

        (fa_t, ct), curve_t, stop_t = solve_arm(tx_field_m16, (32, 4, 3), tx_realize_m16, "TX")
        ct_m16 = tx_mask(ct, keep16)
        region16 = np.isin(blk_of_px, sel16)
        cam_t = tx_realize_from(ct_m16)
        lam_t = sc.seg_argmax(np.stack([ctx.dec[0], cam_t]).astype(np.uint8))
        scr_t = ctx.score_f1(sc, cam_t, region=region16)
        p16 = ct_m16[keep16].astype(np.int8)                    # 16 blocks x 4 atoms x 3 ch
        row["arms"]["TX_ac4_M16"] = {
            **scr_t, "params_lzma1": lzma1_raw(p16.tobytes()),
            "profile": profile(region16, lam_t, ctx), "stop_reason": stop_t,
            "steps_run": curve_t[-1][0]}
        # nested M8 (labelled UNDER-measure per §3 instrument note)
        sel8 = np.argsort(-static_mass)[:8].astype(np.int64)
        keep8 = np.isin(sel32, sel8)
        ct_m8 = tx_mask(ct, keep8)
        cam_t8 = tx_realize_from(ct_m8)
        scr_t8 = ctx.score_f1(sc, cam_t8, region=np.isin(blk_of_px, sel8))
        row["arms"]["TX_ac4_M8_nested"] = {
            **scr_t8, "params_lzma1": lzma1_raw(ct_m8[keep8].astype(np.int8).tobytes())}

        # ---------------- BAND arm: Road<->Lane r=1 interface, free per-px solved paint -------
        road_lane_edge = np.zeros((SEG_H, SEG_W), dtype=bool)
        ls = ctx.lstar
        for dy, dx in ((0, 1), (1, 0)):
            a_sl = ls[dy:, dx:] if dy or dx else ls
            b_sl = ls[: SEG_H - dy, : SEG_W - dx]
            e = ((a_sl == 0) & (b_sl == 1)) | ((a_sl == 1) & (b_sl == 0))
            road_lane_edge[dy:, dx:] |= e
            road_lane_edge[: SEG_H - dy, : SEG_W - dx] |= e
        band = dilate(road_lane_edge, 1)
        _nb, paint, tag = solve_margin_optimal_paint(
            segnet, cam_base, ctx.gt[1], band, ctx.lgt,
            steps=args.max_steps // 2, lr=args.lr, eval_every=args.eval_every)
        cam_b = realize_scorer_paint_to_camera(cam_base, band, paint)
        lam_b = sc.seg_argmax(np.stack([ctx.dec[0], cam_b]).astype(np.uint8))
        scr_b = ctx.score_f1(sc, cam_b, region=band)
        rl_flips = int((road_lane_edge & (ctx.lstar != ctx.lgt)).sum())
        row["arms"]["BAND_road_lane"] = {
            **scr_b, "band_px": int(band.sum()),
            "capture_of_pair_flips": float(((ctx.lstar != ctx.lgt) & band).sum()
                                           / max(ctx.flips0, 1)),
            "road_lane_edge_flips": rl_flips,
            "profile": profile(band, lam_b, ctx), "solver_tag": tag}

        row["wall_s"] = round(time.time() - t_pair, 1)
        out["rows"].append(row)
        checkpoint(args.out, out)
        print(f"[tx] pair {p:3d} ({n+1}/{len(pairs)}) "
              f"| U {scr_u['eta']:.3f} | TX16 {scr_t['eta']:.3f} dpx "
              f"{scr_t['d_pose_ratio_vs_shipped']:.2f} | TX8n {scr_t8['eta']:.3f} "
              f"| BAND {scr_b['eta']:.3f} cap {row['arms']['BAND_road_lane']['capture_of_pair_flips']:.2f} "
              f"[{row['wall_s']}s]", flush=True)


# ================================================================================================
# subcommand: band  (queue head item 2: the PRICE-MATCHED LEGAL value-realizer for the
#                    Road<->Lane band -- class-anchor paint, NO solve, NO oracle values)
# ================================================================================================
def run_band(args, sc, dec, raw, gt_frames, pairs) -> None:
    """sg3 priced the Road<->Lane r=1 band ADDRESS (positions + target labels) at 81,365 B with
    break-even survival 39.56%.  ddm_lr2 §11.3 measured the VALUE-ORACLE ceiling (solved paint,
    eta 1.14-1.34).  This measures the missing number: the LEGAL deterministic value-realizer.

    Realizer: CLASS-ANCHOR paint.  Anchor colors are computed ENCODE-side (mean decoded camera
    RGB of each class's stable interior, per the decoder's own field) and SHIPPED as a counted
    table -- 15 B/pair (AP-pair) or one 15 B global table (AP-global).  The receiver paints
    each band pixel's 4 private camera px with the anchor of its STORED target label.  Zero
    scorer weights at decode, zero oracle pixel values (mirage-law clean).

    Variants per pair (1 SegNet + 1 PoseNet forward each -- no solve, so n=32 is direct):
      AP_pair    per-pair anchors (15 B/pair)
      AP_global  one anchor table for the video (sample-mean approximation, labelled)
      AP_null    the same AP_pair paint delta projected into the rank-6 pose-null subspace
                 (band snapped to 2x2 cells -- the projection/mask commute rule), additive
                 realization: the STREAM-FREE pose-neutral variant.
    Reported per variant: whole-frame realized gain, survival vs band-captured flips (sg3's
    39.56% bar), collateral at BOTH crossovers, d_pose (+ratio) -- the per-pair damage
    distribution feeds the per-base pose-repair gate."""
    import ddm_tr1_runtime as tr1
    from ddm_sq1_eta_seg_realization import COL_SUP, ROW_SUP, dilate
    from ddm_sq1_pose_null_constrained_paint import (
        pose_null_projector,
        project_null,
        snap_band_to_blocks,
    )

    P12 = pose_null_projector()
    out = base_header("band")
    out["band_address_bytes_sg3"] = 81_365
    out["crossovers"] = {"superseded": 0.01035, "corrected_2583e0f155": 0.006285}
    out["rows"] = []

    for n, p in enumerate(pairs):
        t_pair = time.time()
        ctx = PairCtx(sc, dec.packet, raw, gt_frames, p)
        cam_base = tr1.render_frame1_camera_uint8(dec.packet, p)
        base_s = resize_to_scorer(cam_base)[0].permute(1, 2, 0).numpy()   # (384,512,3) float
        ls = ctx.lstar

        road_lane_edge = np.zeros((SEG_H, SEG_W), dtype=bool)
        for dy, dx in ((0, 1), (1, 0)):
            a_sl = ls[dy:, dx:]
            b_sl = ls[: SEG_H - dy, : SEG_W - dx]
            e = ((a_sl == 0) & (b_sl == 1)) | ((a_sl == 1) & (b_sl == 0))
            road_lane_edge[dy:, dx:] |= e
            road_lane_edge[: SEG_H - dy, : SEG_W - dx] |= e
        band = dilate(road_lane_edge, 1)
        band_snapped = snap_band_to_blocks(band)
        captured = int(((ctx.lstar != ctx.lgt) & band).sum())

        # encode-side per-pair anchors from the decoder's own field (stable interior)
        anchors = np.zeros((5, 3), dtype=np.float32)
        for c in range(5):
            interior = (ls == c) & ~band
            if interior.sum() < 16:
                interior = ls == c
            if interior.sum() == 0:
                continue
            cam_mask = scorer_mask_to_camera(interior)
            anchors[c] = cam_base[cam_mask].reshape(-1, 3).mean(axis=0)
        anchors_u8 = np.clip(np.round(anchors), 0, 255).astype(np.uint8)

        def paint_scorer(anch: np.ndarray) -> np.ndarray:
            paint = base_s.copy()
            ii, jj = np.nonzero(band)
            paint[ii, jj] = anch[ctx.lgt[ii, jj]]
            return paint

        row = {"pair": p, "flips_before": ctx.flips0, "n_described": ctx.nd,
               "band_px": int(band.sum()), "band_captured_flips": captured,
               "anchors_pair_u8": anchors_u8.tolist(),
               "d_pose_shipped": ctx.d_pose_shipped, "arms": {}}

        # ---- AP_pair: flat 4-px paint with per-pair anchors --------------------------------
        pnt = paint_scorer(anchors_u8.astype(np.float32))
        cam_e = realize_scorer_paint_to_camera(
            cam_base, band, np.clip(np.round(pnt), 0, 255).astype(np.uint8))
        scr = ctx.score_f1(sc, cam_e, region=band)
        row["arms"]["AP_pair"] = {
            **scr,
            "survival_vs_captured": ((ctx.flips0 - scr["flips_after"]) / captured
                                     if captured else None),
            "anchor_table_bytes": 15}

        # ---- AP_null: the same paint delta through the rank-6 projector (stream-free) ------
        delta = (pnt - base_s) * band_snapped[..., None]
        d_t = torch.from_numpy(np.ascontiguousarray(
            delta.transpose(2, 0, 1)[None], dtype=np.float32))
        d_proj = project_null(d_t, P12)[0].permute(1, 2, 0).numpy()
        cam_f = cam_base.astype(np.float32).copy()
        ii, jj = np.nonzero(band_snapped)
        vals = d_proj[ii, jj]
        for a2 in range(2):
            r = ROW_SUP[ii, a2]
            for b2 in range(2):
                c2 = COL_SUP[jj, b2]
                cam_f[r, c2] = cam_f[r, c2] + vals
        cam_e = np.clip(np.rint(cam_f), 0, 255).astype(np.uint8)
        scr = ctx.score_f1(sc, cam_e, region=band_snapped)
        row["arms"]["AP_null"] = {
            **scr,
            "survival_vs_captured": ((ctx.flips0 - scr["flips_after"]) / captured
                                     if captured else None),
            "anchor_table_bytes": 15}

        row["wall_s"] = round(time.time() - t_pair, 1)
        out["rows"].append(row)
        checkpoint(args.out, out)
        ap_ = row["arms"]["AP_pair"]
        an_ = row["arms"]["AP_null"]
        print(f"[band] pair {p:3d} ({n+1}/{len(pairs)}) band {row['band_px']:5d}px "
              f"cap {captured:4d} | AP eta {ap_['eta']:.3f} surv "
              f"{ap_['survival_vs_captured']:.3f} dpx {ap_['d_pose_ratio_vs_shipped']:.1f} "
              f"| APnull eta {an_['eta']:.3f} dpx {an_['d_pose_ratio_vs_shipped']:.2f} "
              f"[{row['wall_s']}s]", flush=True)

    # AP_global: one anchor table = mean of the per-pair tables (sample approximation)
    glob = np.clip(np.round(np.mean(
        [r["anchors_pair_u8"] for r in out["rows"]], axis=0)), 0, 255).astype(np.uint8)
    out["anchors_global_u8"] = glob.tolist()
    checkpoint(args.out, out)
    print(f"[band] done; global anchor table (sample-mean, labelled) {glob.tolist()}",
          flush=True)


# ================================================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("transport", "response", "solve", "solve0", "keys", "fo1",
                                    "tx", "band"))
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
    ap.add_argument("--m-list", type=str, default="32,64")
    ap.add_argument("--max-steps", type=int, default=150)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--arms", type=str, default="U,AC")
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
    elif args.cmd == "fo1":
        run_fo1(args, sc, dec, raw, gt_frames, pairs)
    elif args.cmd == "tx":
        run_tx(args, sc, dec, raw, gt_frames, pairs)
    elif args.cmd == "band":
        run_band(args, sc, dec, raw, gt_frames, pairs)
    else:
        run_solve(args, sc, dec, raw, gt_frames, pairs)
    print(f"[lr2:{args.cmd}] done t={time.time()-t0:.1f}s -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
