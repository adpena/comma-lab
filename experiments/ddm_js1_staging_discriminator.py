#!/usr/bin/env python
"""ddm_js1 -- STAGING DISCRIMINATOR on the block16 phase field.

et1 measured the block16 regional phase field on our shipped vehicle: reach 41.84%, gross
0.18039 S, 46,247 B, break-even eta 0.1707, REALIZED eta ~0.48-0.52 (cap-pinned = a FLOOR).
That row is seg-LIVE and pose-BLOCKED: d_pose ratios [1.064, 1.190, 1.000, 3.652, 2.094,
1.452, 1.530, 1.120] over the first 8 pairs of its n=32.  et1's budget ladder additionally
measured that eta and pose damage are COUPLED monotonically (eta 0.3345/0.4085/0.4613 at
d_pose 1.003x/1.064x/1.368x for 10/25/50 steps) -- the seg gain is BOUGHT WITH pose.

This unit measures WHICH STAGING converts that seg-live/pose-blocked row into a net win.

    ARM C-PRIME  stage 1 = deep UNCONSTRAINED block16 seg solve on frame_1 (buy eta at full
                 strength, accept the photometric damage)
                 stage 2 = repair d_pose on FRAME_0.
                 SegNet reads ONLY the last frame (upstream/modules.py SegNet.preprocess_input
                 is literally `x = x[:, -1, ...]  # Use only last frame`), so a frame_0 edit is
                 seg-invisible BY CONSTRUCTION -- no cell/margin constraint machinery is needed
                 to hold seg exact.  d_pose is RELATIVE between the two DELIVERED frames (m87),
                 so restoring correspondence by moving frame_0 is a legitimate objective, not a
                 fidelity violation.  bo1 measured stem-level delta_0 cancellability at 99.96%
                 median -- that is the STRUCTURAL ceiling; what this script reports is the
                 REALIZED fraction through the full R -> uint8 -> PoseNet path.

    ARM C        same stage 1; stage 2 = pose repair on FRAME_1 constrained to the argmax
                 CELLS of the stage-1 result (seg-exact by CONSTRAINT rather than by
                 construction).  The control that tells us whether frame_0's structural
                 seg-freedom actually buys anything over constraining frame_1.

    ARM B        joint penalty: one solve on frame_1 with a d_pose penalty term, swept.

ACTUATOR / EXACTNESS.  Both scorers reach the camera plane through the SAME operator D
(pz1: PoseNet and SegNet make the identical interpolate call to segnet_model_input_size), and
D's 2x2 supports are PRIVATE (m86, asserted fail-closed here).  So the solve runs on the
SCORER lattice, and `realize_scorer_paint_to_camera` writes it back to the camera plane where
D reproduces it EXACTLY.  Every arm is then RE-SCORED FROM THE REAL CAMERA PAIR, so the
predicted-vs-verified gap is MEASURED, never assumed (caution (b): null-space membership does
not survive lattice resampling -- pz1 measured 1.662x attenuation for warp-shaped fields).

GRADIENT PATH.  upstream/frame_utils.py:50 decorates rgb_to_yuv6 with @torch.no_grad(), which
SEVERS the PoseNet gradient (the CLAUDE.md eval_roundtrip bug class).  The canonical
differentiable replacement is patched in and its forward equivalence to upstream is ASSERTED
before any gradient is trusted.

BYTES.  This script measures REALIZATION, not carriage.  All arms share stage 1's carriage, so
the DISCRIMINATION between them is valid under any carriage model; ARM C-PRIME additionally
needs a frame_0 stream whose price is an OPEN question recorded in the receipt, never assumed
free.  No row here is byte-closed.

Axis: [macOS-CPU frozen-scorer advisory] NON-PROMOTABLE.  score_claim=false.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))

from ddm_et1_ph1_block16_on_our_vehicle import solve_blocks, translate_blocks
from ddm_sq1_eta_seg_realization import (
    CAM_H,
    CAM_W,
    N_PAIRS_TOTAL,
    SEG_H,
    SEG_W,
    Scorer,
    _assert_private_support,
    decode_gt_frames,
    seq_len,
)
from ddm_sq1_stage_decomposition_and_solved_paint import (
    realize_scorer_paint_to_camera,
    resize_to_scorer,
    solve_margin_optimal_paint,
)

S_PER_FLIP = 100.0 / (N_PAIRS_TOTAL * SEG_H * SEG_W)
RATE_PER_BYTE = 25.0 / 37_545_489.0
GAP_S = 0.7910689 - 0.172141
# et1 §8: dS/dd_pose at the CURRENT operating point (K3 -- never a shelf price).
DS_DDPOSE = 31.3026


# ================================================================================================
# gradient path -- the @torch.no_grad() on upstream rgb_to_yuv6 severs the pose gradient
# ================================================================================================
def patch_yuv6_and_assert(sc: Scorer) -> dict:
    """Install the differentiable yuv6 and PROVE it matches upstream before trusting a gradient.

    m50 / vacuity discipline: a patch that silently no-ops would make every pose gradient below
    zero and the arm would read as 'pose is unrepairable'.  So the equivalence is asserted, and
    the gradient is additionally proven to FLOW (non-zero grad on a probe) before any solve.
    """
    from tac.differentiable_eval_roundtrip import (
        assert_yuv6_forward_equivalence_to_upstream,
        patch_upstream_yuv6_globally,
    )

    patch_upstream_yuv6_globally()
    eq = assert_yuv6_forward_equivalence_to_upstream(num_samples=5, atol=1e-6)

    # gradient-liveness probe: the patch must make d(pose)/d(pixels) actually non-zero.
    leaf = (torch.rand(1, 2, 3, 64, 64) * 255.0).requires_grad_(True)
    with torch.enable_grad():
        y = sc.net.posenet.preprocess_input(leaf)
        y.sum().backward()
    live = leaf.grad is not None and float(leaf.grad.abs().sum()) > 0.0
    if not live:
        raise RuntimeError(
            "yuv6 patch installed but the PoseNet preprocess gradient is still dead -- every "
            "pose solve below would silently return its starting point (m50 vacuity class)."
        )
    return {"patched": True,
            "yuv6_equivalence_passed": bool(eq.get("passed")),
            "yuv6_max_abs_error": float(eq.get("max_abs_error", float("nan"))),
            "gradient_live": bool(live),
            "gradient_probe_abs_sum": float(leaf.grad.abs().sum())}


def pose_forward_grad(posenet, f0_s: torch.Tensor, f1_s: torch.Tensor):
    """PoseNet forward on SCORER-lattice frames, with gradients, bypassing the camera resize.

    Legal because D(camera) == the scorer-lattice tensor EXACTLY once all four private camera
    pixels of each scorer pixel carry that pixel's value (m86 + realize_scorer_paint_to_camera).
    The camera round-trip is nonetheless re-scored at the end of every pair, so this shortcut is
    VERIFIED rather than assumed.
    """
    from frame_utils import rgb_to_yuv6

    x = torch.cat([f0_s, f1_s], dim=0)                     # (2,3,384,512)
    # yuv6 packs each 2x2 luma block into 4 channels + 2 subsampled chroma, so the spatial dims
    # HALVE: (2,6,192,256).  Derive them from the tensor rather than assuming (measured, not
    # asserted -- the first run of this script caught a hardcoded (384,512) here).
    y = rgb_to_yuv6(x)                                     # (2,6,h,w) -- patched, grad-live
    if y.shape[0] != 2 or y.shape[1] != 6:
        raise RuntimeError(f"unexpected yuv6 shape {tuple(y.shape)}; expected (2,6,h,w)")
    # contiguous reshape (2,6,h,w)->(1,12,h,w) reproduces upstream's 'b (t c) h w' ordering
    y = y.reshape(1, 12, y.shape[-2], y.shape[-1])
    return posenet(y)


def d_pose_t(posenet, out_gt, out_cmp) -> torch.Tensor:
    return posenet.compute_distortion(out_gt, out_cmp)[0]


# ================================================================================================
# ARM C-PRIME -- pose repair on the structurally seg-free frame_0
# ================================================================================================
def solve_pose_repair_frame0(sc: Scorer, dec_f0: np.ndarray, edited_f1: np.ndarray,
                             pose_gt, *, steps: int, lr: float, eval_every: int,
                             linf: float) -> tuple:
    """Solve a scorer-lattice delta on FRAME_0 that minimises d_pose against the DAMAGED frame_1.

    Best-iterate is retained on the REALIZED (rounded uint8) d_pose, never on the differentiable
    proxy -- the fd2/tb1 rider, and the reason et1's eta numbers are trustworthy.

    `linf` bounds the per-pixel excursion; linf<=0 means unconstrained (the ceiling arm).
    """
    posenet = sc.net.posenet
    base0 = resize_to_scorer(dec_f0)                       # (1,3,384,512) float
    f1_s = resize_to_scorer(edited_f1).detach()

    def realized_dpose(t0: torch.Tensor) -> float:
        q = torch.round(torch.clamp(t0, 0.0, 255.0)).detach()
        with torch.no_grad():
            out = pose_forward_grad(posenet, q, f1_s)
            return float(d_pose_t(posenet, pose_gt, out))

    best = (realized_dpose(base0), np.zeros((SEG_H, SEG_W, 3), np.float32), "identity@0")
    delta = torch.zeros_like(base0, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    with torch.enable_grad():
        for it in range(steps + 1):
            d = delta if linf <= 0 else torch.clamp(delta, -linf, linf)
            cur = torch.clamp(base0 + d, 0.0, 255.0)
            if it % eval_every == 0 or it == steps:
                dp = realized_dpose(cur)
                if dp < best[0]:
                    q = torch.round(torch.clamp(cur, 0.0, 255.0)).detach()
                    best = (dp, q[0].permute(1, 2, 0).numpy().astype(np.uint8), f"f0@{it}")
            if it == steps:
                break
            out = pose_forward_grad(posenet, cur, f1_s)
            loss = d_pose_t(posenet, pose_gt, out)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return best  # (realized_dpose_at_scorer_lattice, frame0_paint_u8 HWC, tag)


def solve_pose_repair_frame0_cheap_dct(sc: Scorer, dec_f0: np.ndarray, edited_f1: np.ndarray,
                                       pose_gt, *, k: int, steps: int, lr: float,
                                       eval_every: int) -> tuple:
    """ARM C-PRIME-CHEAP -- repair frame_0 using only a rank-k low-frequency DCT basis.

    WHY THIS ARM EXISTS AND WHY IT IS NOT REDUNDANT WITH delta_structure.  Measuring that the
    FREE solution's delta is expensive, or that a cheap basis captures little of its energy, does
    NOT establish that a cheap basis cannot solve the problem -- p3v2 named this exactly: "the
    free win is BASIS-ADVERSARIAL".  Projecting a free solution onto a cheap basis and SOLVING
    within that basis are different operations, and only the second answers the carriage
    question.  ph5o ran the second one on the sibling blind-set actuator (rank-6 separable DCT ->
    all-zero on 100% of pairs); this runs it on frame_0 against the staged objective.

    The basis is a GENERIC separable 2-D DCT, deterministically generated, so it is FREE in
    inflate.py under rule 118: only k*k*3 coefficients per pair are COUNTED.
    """
    from scipy.fft import idctn

    posenet = sc.net.posenet
    base0 = resize_to_scorer(dec_f0)                       # (1,3,384,512)
    f1_s = resize_to_scorer(edited_f1).detach()

    # precompute the k*k separable-DCT synthesis atoms on the scorer lattice (generic, free)
    atoms = np.zeros((k * k, SEG_H, SEG_W), np.float32)
    for a in range(k):
        for b in range(k):
            c = np.zeros((SEG_H, SEG_W))
            c[a, b] = 1.0
            atoms[a * k + b] = idctn(c, type=2, norm="ortho", axes=(-2, -1))
    A = torch.from_numpy(atoms).reshape(k * k, -1)          # (K, H*W)

    def realized_dpose(t0: torch.Tensor) -> float:
        q = torch.round(torch.clamp(t0, 0.0, 255.0)).detach()
        with torch.no_grad():
            return float(d_pose_t(posenet, pose_gt, pose_forward_grad(posenet, q, f1_s)))

    best = (realized_dpose(base0), None, "identity@0", 0.0)
    coef = torch.zeros(3, k * k, requires_grad=True)
    opt = torch.optim.Adam([coef], lr=lr)
    with torch.enable_grad():
        for it in range(steps + 1):
            d = (coef @ A).reshape(1, 3, SEG_H, SEG_W)
            cur = torch.clamp(base0 + d, 0.0, 255.0)
            if it % eval_every == 0 or it == steps:
                # EVALUATE THE PAYLOAD WE WOULD ACTUALLY SHIP.  The counted object is k*k*3
                # INT16 coefficients, so the best-iterate must be selected on the QUANTISED
                # synthesis -- otherwise the reported value belongs to a float payload nobody
                # can carry, and the price would be counted against a value never measured
                # through its own quantiser.  This makes the byte-close gate hold BY
                # CONSTRUCTION rather than as an owed follow-on.
                cq = torch.round(coef.detach()).clamp(-32768, 32767)
                dq = (cq @ A).reshape(1, 3, SEG_H, SEG_W)
                curq = torch.clamp(base0 + dq, 0.0, 255.0)
                dp = realized_dpose(curq)
                if dp < best[0]:
                    q = torch.round(curq).detach()
                    best = (dp, q[0].permute(1, 2, 0).numpy().astype(np.uint8), f"dct{k}q@{it}",
                            float(cq.abs().max()))
            if it == steps:
                break
            out = pose_forward_grad(posenet, cur, f1_s)
            loss = d_pose_t(posenet, pose_gt, out)
            opt.zero_grad()
            loss.backward()
            opt.step()
    # counted payload = k*k*3 int16 coefficients per pair (the basis itself is generic => free)
    return best[0], best[1], best[2], int(coef.detach().numel() * 2), best[3]


def solve_pose_repair_frame1_cellconstrained(sc: Scorer, edited_f1: np.ndarray,
                                             lam_locked: np.ndarray, pose_gt,
                                             dec_f0: np.ndarray, *, steps: int, lr: float,
                                             eval_every: int) -> tuple:
    """ARM C -- pose repair on FRAME_1, constrained to keep the stage-1 argmax EXACTLY.

    Seg-exactness is enforced by REJECTION on the realized argmax (never by a soft penalty and
    never assumed): an iterate whose SegNet argmax differs anywhere from `lam_locked` is not
    eligible to become the best-iterate.  That makes the constraint measured, not claimed.
    """
    posenet, segnet = sc.net.posenet, sc.net.segnet
    base1 = resize_to_scorer(edited_f1)
    f0_s = resize_to_scorer(dec_f0).detach()
    tgt = torch.from_numpy(lam_locked.astype(np.int64))[None]

    def eval_iter(t1: torch.Tensor):
        q = torch.round(torch.clamp(t1, 0.0, 255.0)).detach()
        with torch.no_grad():
            lam = segnet(q).argmax(dim=1)[0].numpy().astype(np.uint8)
            if not (lam == lam_locked).all():
                return None, q            # violates the cell constraint -> ineligible
            out = pose_forward_grad(posenet, f0_s, q)
            return float(d_pose_t(posenet, pose_gt, out)), q

    dp0, q0 = eval_iter(base1)
    best = (dp0 if dp0 is not None else float("inf"),
            q0[0].permute(1, 2, 0).numpy().astype(np.uint8), "identity@0", 0)
    n_reject = 0
    delta = torch.zeros_like(base1, requires_grad=True)
    opt = torch.optim.Adam([delta], lr=lr)
    with torch.enable_grad():
        for it in range(steps + 1):
            cur = torch.clamp(base1 + delta, 0.0, 255.0)
            if it % eval_every == 0 or it == steps:
                dp, q = eval_iter(cur)
                if dp is None:
                    n_reject += 1
                elif dp < best[0]:
                    best = (dp, q[0].permute(1, 2, 0).numpy().astype(np.uint8), f"f1cell@{it}",
                            n_reject)
            if it == steps:
                break
            out = pose_forward_grad(posenet, f0_s, cur)
            loss = d_pose_t(posenet, pose_gt, out)
            # keep the argmax: cross-entropy toward the locked labels holds the cell
            loss = loss + 0.05 * torch.nn.functional.cross_entropy(segnet(cur), tgt)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return best[0], best[1], best[2], n_reject


def realize_full_frame(dec_f: np.ndarray, paint_u8: np.ndarray) -> np.ndarray:
    """Write a FULL scorer-lattice frame into the camera plane via D's private 2x2 supports."""
    return realize_scorer_paint_to_camera(dec_f, np.ones((SEG_H, SEG_W), bool), paint_u8)


def delta_structure(base_s: np.ndarray, paint_u8: np.ndarray) -> dict:
    """Measure whether a repair delta is CHEAPLY CODEABLE or ADDRESS-LIMITED.

    ph5o measured the sibling seg-free pose actuator and found ALIGNMENT: YES / RATE: NO -- the
    descent direction is a handful of isolated, per-pair-private pixels whose ADDRESS costs
    about an order of magnitude more than the pose it buys, and a rank-6 separable-DCT basis
    solved to the all-zero integer vector on 100% of pairs ("the cheapness of a generic basis
    and the localisation of the descent are the same property with opposite signs").

    That is a PREDICTION about this arm, so it is measured rather than inherited: a repair that
    works but cannot be paid for is not a bankable row.  Reported: real-coder bytes for the raw
    delta, sparsity, and how much of the delta's energy a cheap low-frequency DCT basis captures
    (a generic DCT basis is FREE in inflate.py under rule 118, so high capture = cheap carriage).
    """
    import lzma

    import brotli

    d = paint_u8.astype(np.int16) - np.rint(base_s).astype(np.int16)
    nnz = int(np.count_nonzero(d))
    raw = np.ascontiguousarray(d.astype(np.int16)).tobytes()
    out: dict = {
        "nnz": nnz,
        "nnz_frac": nnz / d.size,
        "max_abs": int(np.abs(d).max()),
        "mean_abs": float(np.abs(d).mean()),
        "brotli_q11_bytes": len(brotli.compress(raw, quality=11)),
        "lzma_bytes": len(lzma.compress(raw, preset=9)),
        "raw_int16_bytes": len(raw),
    }
    # low-frequency DCT energy capture, per channel, on the scorer lattice
    try:
        from scipy.fftpack import dctn

        e_tot = float((d.astype(np.float64) ** 2).sum())
        if e_tot > 0:
            C = dctn(d.astype(np.float64), axes=(0, 1), norm="ortho")
            for k in (4, 8, 16, 32):
                out[f"dct_energy_capture_top{k}x{k}"] = float(
                    (C[:k, :k, :] ** 2).sum() / e_tot)
    except Exception as exc:                                   # scipy optional
        out["dct_error"] = str(exc)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub-dir", type=Path, required=True)
    ap.add_argument("--gt-mkv", type=Path, required=True)
    ap.add_argument("--pairs-npy", type=Path, required=True)
    ap.add_argument("--argmax-cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only-pairs", type=str, default="",
                    help="comma-separated pair ids (must be a SUBSET of --pairs-npy)")
    ap.add_argument("--block", type=int, default=16)
    ap.add_argument("--rmax", type=int, default=5)
    ap.add_argument("--seg-steps", type=int, default=25)
    ap.add_argument("--seg-lr", type=float, default=2.0)
    ap.add_argument("--pose-steps", type=int, default=40)
    ap.add_argument("--pose-lr", type=float, default=2.0)
    ap.add_argument("--pose-linf", type=float, default=0.0, help="<=0 = unconstrained ceiling")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--arms", type=str, default="cprime",
                    help="comma list of: cprime,cheapdct,poseonly,ccell")
    ap.add_argument("--dct-k", type=str, default="8,32",
                    help="comma list of DCT ranks k for the cheapdct arm")
    ap.add_argument("--dct-lr", type=float, default=20.0)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    arms = {a.strip() for a in args.arms.split(",") if a.strip()}
    t0 = time.time()
    all_pairs = np.load(args.pairs_npy).tolist()
    if args.only_pairs:
        want = [int(x) for x in args.only_pairs.split(",")]
        missing = [p for p in want if p not in all_pairs]
        if missing:
            raise SystemExit(f"--only-pairs {missing} not in --pairs-npy (pair-matching broken)")
        pairs = want
    else:
        pairs = all_pairs

    geom = _assert_private_support()
    raw = np.memmap(args.sub_dir / "inflated" / "0.raw", dtype=np.uint8, mode="r",
                    shape=(N_PAIRS_TOTAL * seq_len, CAM_H, CAM_W, 3))
    cx1 = np.load(args.argmax_cache / "cx1_argmax_n600.npy", mmap_mode="r")
    gtc = np.load(args.argmax_cache / "gt_argmax_n600.npy", mmap_mode="r")

    wanted = set()
    for p in pairs:
        wanted.update({seq_len * p, seq_len * p + 1})
    gt_frames = decode_gt_frames(args.gt_mkv, wanted)
    sc = Scorer(args.threads)
    patch = patch_yuv6_and_assert(sc)
    print(f"[js1] scorer+patch ready t={time.time()-t0:.1f}s  {patch}", flush=True)

    rows: list[dict] = []
    if args.resume and args.out.exists():
        rows = json.loads(args.out.read_text()).get("rows", [])
        done = {int(r["pair"]) for r in rows}
        pairs = [p for p in pairs if p not in done]
        print(f"[js1] resume: {len(rows)} on disk, {len(pairs)} left", flush=True)

    def flush() -> None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"schema": "ddm_js1_staging_discriminator.v1",
                       "axis": "[macOS-CPU frozen-scorer advisory] NON-PROMOTABLE",
                       "score_claim": False, "promotion_eligible": False,
                       "verdict_scope_default": "FORMULATION",
                       "pointer": "0.1910828242 [contest-CPU] UNMOVED",
                       "own_vehicle_frontier": "S = 0.7910689 @ 353,805 B [macOS-CPU advisory]",
                       "block": args.block, "rmax": args.rmax, "arms": sorted(arms),
                       "budget": {"seg_steps": args.seg_steps, "seg_lr": args.seg_lr,
                                  "pose_steps": args.pose_steps, "pose_lr": args.pose_lr,
                                  "pose_linf": args.pose_linf,
                                  "eval_every": args.eval_every},
                       "denominators": {"gap_S": GAP_S, "S_per_flip": S_PER_FLIP,
                                        "rate_per_byte": RATE_PER_BYTE,
                                        "dS_dd_pose_at_operating_point": DS_DDPOSE},
                       "yuv6_patch": patch, "D_geometry": geom,
                       "pairs_requested": pairs, "rows": rows}, f, indent=1)

    for n, p in enumerate(pairs):
        tp = time.time()
        dec = np.stack([raw[seq_len * p], raw[seq_len * p + 1]]).astype(np.uint8)
        gt = np.stack([gt_frames[seq_len * p], gt_frames[seq_len * p + 1]])
        lstar = sc.seg_argmax(dec)
        lgt = sc.seg_argmax(gt)
        rec: dict = {"pair": int(p),
                     "C2_lstar_matches_cache": bool((lstar == np.asarray(cx1[p])).all()),
                     "C3_lgt_matches_cache": bool((lgt == np.asarray(gtc[p])).all())}

        flips0_map = lstar != lgt
        flips0 = int(flips0_map.sum())
        rec["flips_before"] = flips0
        pose_gt = sc.pose_out(gt)
        dp_before = sc.d_pose(pose_gt, sc.pose_out(dec))
        rec["d_pose_before"] = dp_before

        # CONTROL C5: the scorer-lattice pose path used by the solver must reproduce the
        # canonical CAMERA path exactly.  If it does not, every pose number below is measured
        # on a different object than the one that is scored (the surrogate-is-not-authority
        # class).  Measured per pair, never assumed.
        with torch.no_grad():
            out_sl = pose_forward_grad(sc.net.posenet,
                                       resize_to_scorer(dec[0]), resize_to_scorer(dec[1]))
            dp_sl = float(d_pose_t(sc.net.posenet, pose_gt, out_sl))
        rec["C5_scorer_lattice_pose_matches_camera"] = {
            "camera": dp_before, "scorer_lattice": dp_sl,
            "abs_diff": abs(dp_sl - dp_before),
            "rel_diff": abs(dp_sl - dp_before) / max(abs(dp_before), 1e-30),
        }

        off = solve_blocks(lstar, lgt, args.block, args.rmax)
        target = translate_blocks(lstar, off.reshape(-1, 2), args.block)
        band = target != lstar
        rec["band_px"] = int(band.sum())
        nd = flips0 - int((target != lgt).sum())
        rec["n_described"] = nd

        # ---- STAGE 1: deep UNCONSTRAINED seg solve on frame_1 -------------------------------
        _, paint, tag, solve_diag = solve_margin_optimal_paint(
            sc.net.segnet, dec[1], gt[1], band, target,
            steps=args.seg_steps, lr=args.seg_lr, eval_every=args.eval_every)
        edited_f1 = realize_scorer_paint_to_camera(dec[1], band, paint)
        pair_s1 = np.stack([dec[0], edited_f1])
        lam_s1 = sc.seg_argmax(pair_s1)
        fa = int((lam_s1 != lgt).sum())
        dp_s1 = sc.d_pose(pose_gt, sc.pose_out(pair_s1))
        rec["stage1"] = {
            "solve_tag": tag,
            "cap_pinned": bool(str(tag).rsplit("@", 1)[-1] == str(args.seg_steps)),
            "stop_reason": solve_diag["selected"]["stop_reason"],
            "trajectory_stop": solve_diag["selected"].get("trajectory_stop"),
            "steps_run": solve_diag["selected"]["steps_run"],
            "best_step": solve_diag["selected"]["best_step"],
            "selected_curve": solve_diag["selected"].get("curve", []),
            "start_diagnostics": solve_diag.get("starts", []),
            "flips_after": fa,
            "eta_realized": ((flips0 - fa) / nd) if nd else None,
            "d_pose_after": dp_s1,
            "d_pose_ratio": dp_s1 / dp_before,
            "target_fidelity_in_band": float((lam_s1[band] == target[band]).mean())
            if int(band.sum()) else None,
        }

        # ---- CONTROL: frame_0 really is seg-invisible (measured, not cited) ------------------
        rng = np.random.default_rng(7 + p)
        f0_noise = np.clip(dec[0].astype(np.int16)
                           + rng.integers(-40, 41, dec[0].shape), 0, 255).astype(np.uint8)
        lam_ctrl = sc.seg_argmax(np.stack([f0_noise, edited_f1]))
        rec["control_frame0_is_seg_free"] = bool((lam_ctrl == lam_s1).all())

        # ---- STAGE 2 ------------------------------------------------------------------------
        if "cprime" in arms:
            dp_pred, paint0, tag0 = solve_pose_repair_frame0(
                sc, dec[0], edited_f1, pose_gt, steps=args.pose_steps, lr=args.pose_lr,
                eval_every=args.eval_every, linf=args.pose_linf)
            edited_f0 = realize_full_frame(dec[0], paint0) if tag0 != "identity@0" else dec[0]
            pair_cp = np.stack([edited_f0, edited_f1])
            lam_cp = sc.seg_argmax(pair_cp)            # MUST equal lam_s1 exactly
            dp_ver = sc.d_pose(pose_gt, sc.pose_out(pair_cp))
            fa_cp = int((lam_cp != lgt).sum())
            rec["arm_cprime"] = {
                "tag": tag0,
                "d_pose_predicted_scorer_lattice": dp_pred,
                "d_pose_verified_from_camera": dp_ver,
                "realization_gap": dp_ver - dp_pred,
                "d_pose_ratio_vs_before": dp_ver / dp_before,
                "repair_fraction_of_damage": (
                    (dp_s1 - dp_ver) / (dp_s1 - dp_before)
                    if abs(dp_s1 - dp_before) > 1e-18 else None),
                "seg_exactly_preserved": bool((lam_cp == lam_s1).all()),
                "flips_after": fa_cp,
                "eta_realized": ((flips0 - fa_cp) / nd) if nd else None,
                "max_abs_f0_camera_delta": int(
                    np.abs(edited_f0.astype(np.int16) - dec[0].astype(np.int16)).max()),
                # ph5o's prediction, measured on THIS actuator rather than inherited
                "delta_structure": (
                    delta_structure(
                        resize_to_scorer(dec[0])[0].permute(1, 2, 0).numpy(), paint0)
                    if tag0 != "identity@0" else None),
            }

        if "cheapdct" in arms:
            for k in (int(x) for x in str(args.dct_k).split(",")):
                dp_c, paint0c, tagc, cbytes, cmax = solve_pose_repair_frame0_cheap_dct(
                    sc, dec[0], edited_f1, pose_gt, k=k, steps=args.pose_steps,
                    lr=args.dct_lr, eval_every=args.eval_every)
                if paint0c is not None:
                    pair_ch = np.stack([realize_full_frame(dec[0], paint0c), edited_f1])
                    dp_cv = sc.d_pose(pose_gt, sc.pose_out(pair_ch))
                    seg_ok = bool((sc.seg_argmax(pair_ch) == lam_s1).all())
                else:
                    dp_cv, seg_ok = dp_s1, True
                rec[f"arm_cprime_cheap_dct{k}"] = {
                    "tag": tagc, "k": k,
                    "d_pose_verified_from_camera": dp_cv,
                    "d_pose_ratio_vs_before": dp_cv / dp_before,
                    "d_pose_ratio_vs_stage1_damage": dp_cv / dp_s1,
                    "solved_to_all_zero": bool(paint0c is None),
                    "seg_exactly_preserved": seg_ok,
                    "counted_bytes_per_pair": cbytes,
                    "counted_bytes_n600": cbytes * N_PAIRS_TOTAL,
                    "rate_cost_S_n600": cbytes * N_PAIRS_TOTAL * RATE_PER_BYTE,
                    # byte-close gate: the value above was measured ON the int16 payload
                    "value_measured_through_int16_quantiser": True,
                    "max_abs_int16_coefficient": cmax,
                    "int16_range_ok": bool(cmax <= 32767),
                }

        # ---- CONTROL ARM: pose-only, NO seg solve -------------------------------------------
        # DECISIVE ATTRIBUTION.  pu2 solved frame_0 on only 6 pairs, so an un-solved pair still
        # holds unharvested frame_0 pose headroom.  Without this control a repair ratio below
        # 1.0 would credit the STAGING for headroom that was simply never taken -- the same
        # unanchored-delta error this unit caught in et1's inherited dS/dd_pose.  This arm
        # solves frame_0 against the UNDAMAGED frame_1, so:
        #     staging cost of seg  =  cprime_ratio  -  poseonly_ratio
        if "poseonly" in arms:
            dp_p, paint0p, tag0p = solve_pose_repair_frame0(
                sc, dec[0], dec[1], pose_gt, steps=args.pose_steps, lr=args.pose_lr,
                eval_every=args.eval_every, linf=args.pose_linf)
            ed_f0p = realize_full_frame(dec[0], paint0p) if tag0p != "identity@0" else dec[0]
            pair_po = np.stack([ed_f0p, dec[1]])
            dp_pv = sc.d_pose(pose_gt, sc.pose_out(pair_po))
            lam_po = sc.seg_argmax(pair_po)
            rec["arm_poseonly_control"] = {
                "tag": tag0p,
                "d_pose_predicted_scorer_lattice": dp_p,
                "d_pose_verified_from_camera": dp_pv,
                "realization_gap": dp_pv - dp_p,
                "d_pose_ratio_vs_before": dp_pv / dp_before,
                "seg_unchanged_vs_shipped": bool((lam_po == lstar).all()),
                "flips_after": int((lam_po != lgt).sum()),
            }

        if "ccell" in arms:
            dp_predc, paint1, tagc, nrej = solve_pose_repair_frame1_cellconstrained(
                sc, edited_f1, lam_s1, pose_gt, dec[0], steps=args.pose_steps,
                lr=args.pose_lr, eval_every=args.eval_every)
            ed_f1b = realize_full_frame(edited_f1, paint1) if tagc != "identity@0" else edited_f1
            pair_cc = np.stack([dec[0], ed_f1b])
            lam_cc = sc.seg_argmax(pair_cc)
            dp_verc = sc.d_pose(pose_gt, sc.pose_out(pair_cc))
            fa_cc = int((lam_cc != lgt).sum())
            rec["arm_ccell"] = {
                "tag": tagc, "n_rejected_iterates": nrej,
                "d_pose_predicted_scorer_lattice": dp_predc,
                "d_pose_verified_from_camera": dp_verc,
                "realization_gap": dp_verc - dp_predc,
                "d_pose_ratio_vs_before": dp_verc / dp_before,
                "repair_fraction_of_damage": (
                    (dp_s1 - dp_verc) / (dp_s1 - dp_before)
                    if abs(dp_s1 - dp_before) > 1e-18 else None),
                "seg_exactly_preserved": bool((lam_cc == lam_s1).all()),
                "flips_after": fa_cc,
                "eta_realized": ((flips0 - fa_cc) / nd) if nd else None,
            }

        rows.append(rec)
        s1 = rec["stage1"]
        msg = (f"[js1] pair {p:3d} ({n+1}/{len(pairs)}) flips {flips0:5d} nd {nd:5d} | "
               f"S1 eta {s1['eta_realized']:+.4f} dpose {s1['d_pose_ratio']:.3f}x")
        if "arm_cprime" in rec:
            a = rec["arm_cprime"]
            rf = a["repair_fraction_of_damage"]
            msg += (f" | C' dpose {a['d_pose_ratio_vs_before']:.3f}x "
                    f"repair {f'{rf:.1%}' if rf is not None else 'n/a'} "
                    f"seg_exact={a['seg_exactly_preserved']} gap {a['realization_gap']:+.2e}")
        if "arm_poseonly_control" in rec:
            a = rec["arm_poseonly_control"]
            msg += f" | POSEONLY dpose {a['d_pose_ratio_vs_before']:.3f}x"
        if "arm_ccell" in rec:
            a = rec["arm_ccell"]
            rf = a["repair_fraction_of_damage"]
            msg += (f" | Ccell dpose {a['d_pose_ratio_vs_before']:.3f}x "
                    f"repair {f'{rf:.1%}' if rf is not None else 'n/a'} rej {a['n_rejected_iterates']}")
        print(msg + f" [{time.time()-tp:.1f}s]", flush=True)
        flush()

    flush()
    print(f"[js1] DONE {len(rows)} t={time.time()-t0:.1f}s -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
