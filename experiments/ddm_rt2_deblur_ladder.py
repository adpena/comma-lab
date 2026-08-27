#!/usr/bin/env python3
"""ddm_rt2 -- the DE-BLUR ladder: remove the decoder's own up/down blur, priced on seg AND pose.

WHAT THE MECHANISM IS (measured by this arm, `RT2_STRUCTURE_PROBE` + `RT2_OPERATOR`)
------------------------------------------------------------------------------------
The decoder renders frame_1 at **384x512 -- byte for byte the size SegNet consumes** -- then
bilinear-upsamples it to 874x1164 and rounds to uint8.  The scorer immediately bilinear-
downsamples back to 384x512.  So the scorer never sees the render: it sees `A m` where

    A = D . U      (D = scorer downsample, U = decoder upsample)

MEASURED here, independently: `U` is **bilinear**, not bicubic (bicubic leaves 1.54e-2 off the
tridiagonal band; bilinear leaves exactly 0.000e+00), `A` is tridiagonal with row sums 1 and
middle row [0.101470, 0.797058, 0.101472], and the shipped camera frames lie in `range(U)` to
rms 0.233 / max 0.825 -- i.e. pure uint8 residue.  The up/down pair is therefore a **gratuitous
low-pass of a field that was already exactly the right size**.

THE CURE
--------
`D` is surjective (rank 384*512), so for ANY scorer-resolution target `T` there is an exact
camera-resolution preimage, in closed form:

    X' = X + R_dag (T - D X) C_dag^T          =>   D X' = T   exactly

Take `T = m_hat`, the native render recovered by least squares `m_hat = U^+ X` (the real
decoder does not even need the estimate -- it HAS `m`).  Then SegNet sees the render itself and
the whole `A` term is gone.  Zero archive bytes: every operand is the decoded render, the
transform is generic algorithm in inflate.py (rule 118), and no scorer is touched at decode.

Ladder: `T(alpha) = D X + alpha * (m_hat - D X)`.  alpha = 0 MUST reproduce the base exactly --
that is the positive control, and a run whose alpha=0 row differs is void.

WHY POSE IS MEASURED HERE AND NOT DEFERRED
------------------------------------------
rn1 measured that on this vehicle every undirected camera-space operator costs 25-70x more on
pose than it gains on seg (`d_pose` marginal 6.03x seg's at hv1's d_pose 6.88e-06), and that
the advisory pose instrument is itself ~18.2x optimistic.  sr1's sealed FO-1 ladder counts
FLIPS ONLY.  A seg-only ladder on this vehicle cannot produce a verdict, so this tool refuses
to emit one: every row carries seg, pose, and the net.

Reference form imported, not reimplemented: rn1's `Instrument` (frozen CPU-torch SegNet +
PoseNet, batch-1, upstream preprocess verbatim) and its canonical `decode_gt`
(`frame_utils.yuv420_to_rgb` only -- PyAV rgb24 manufactures ~100x phantom pose).

axis: [macOS-CPU advisory] -- NEVER a score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO / "experiments"))

import ddm_rt2_mechanism_decomposition as RT2
from ddm_rn1_render_boundary_mechanism import (
    DEFAULT_MKV,
    Instrument,
    decode_gt,
)

FRAMES = RT2.FRAMES
SEG_H, SEG_W = RT2.SEG_H, RT2.SEG_W
CAM_H, CAM_W = RT2.CAM_H, RT2.CAM_W
SCORED_PX = RT2.SCORED_PX
S_PER_FLIP = RT2.S_PER_FLIP                 # 100 / 117,964,800
S_PER_BYTE = 25.0 / 37_545_489              # 6.658590e-07

# hv1 ep0634 frontier, [contest-CUDA T4 n600] -- the base every delta is charged against
BASE_S = 0.15959729295498598
BASE_BYTES = 182_759
BASE_D_SEG = 2.9611e-04
BASE_D_POSE = 8.2945765e-03 ** 2 / 10.0     # from the pose contribution sqrt(10*d_pose)
GAP_TO_015 = 0.15 - BASE_S                  # -0.0095973

DEFAULT_WORK = Path("/Volumes/APDataStore/pact/ddm_rt2")


class LadderError(RuntimeError):
    """Fail-closed."""


def lift_matrix(n_seg: int, n_cam: int) -> np.ndarray:
    """(n_cam, n_seg) bilinear UPSAMPLE matrix -- the decoder's own `U`, identified by this arm."""
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        e = torch.eye(n_seg)[None, None]
        return F.interpolate(e, size=(n_cam, n_seg), mode="bilinear",
                             align_corners=False)[0, 0].numpy().astype(np.float64)


#: PoseNet luma weights, verbatim from `upstream/frame_utils.py::rgb_to_yuv6`.
K_YUV = np.array([0.299, 0.587, 0.114], dtype=np.float64)


# PROJECT_PARITY_WAIVED: analysis-probe perturbation projector (PoseNet null space); nothing stored, inflate reads no transformed tensor
def project_pose_null(delta: np.ndarray) -> np.ndarray:
    """Orthogonal projection of a scorer-resolution RGB perturbation onto PoseNet's null space.

    DERIVED from `upstream/frame_utils.py::rgb_to_yuv6` (read at source, lines 51-76).  PoseNet
    consumes six planes: four LUMA planes `y00/y10/y01/y11` that are the full-resolution `Y`
    merely de-interleaved (no information lost), and two CHROMA planes that are exact 2x2 BOX
    AVERAGES of `U` and `V`.  So a perturbation is invisible to PoseNet iff

        (1) `dY = 0.299 dR + 0.587 dG + 0.114 dB = 0` at every pixel, and
        (2) `dU` and `dV` have zero mean over every 2x2 block.

    With `dY = 0`, `dU = dB/1.772` and `dV = dR/1.402`, so (2) is `blockmean(dR) =
    blockmean(dB) = 0`, which with (1) forces `blockmean(dG) = 0` as well.  The constraint set
    is therefore

        N = {dY = 0 pointwise} INTERSECT {every channel has zero 2x2 block mean}

    Counting per 2x2 block: 12 DOF, minus 4 (pointwise `dY`), minus 3 block-mean constraints of
    which 1 is dependent (blockmean of `dY` is already 0) => **6 of 12 DOF survive, exactly 50%
    of the scorer-resolution RGB field is invisible to PoseNet**.  SegNet reads RGB directly and
    sees all 12.

    The two projections commute -- one is `I_space (x) Q_channel`, the other is
    `M_space (x) I_channel` -- so their product IS the orthogonal projection onto the
    intersection, not merely an approximation.

    LIMIT, measured not assumed: `rgb_to_yuv6` clamps `U` and `V` to [0,255], so the map is
    only piecewise linear.  Where a pixel's chroma saturates, this projection leaks.  The
    ladder measures the realized `d_pose` rather than trusting the algebra.
    """
    d = delta - (delta @ K_YUV)[..., None] * (K_YUV / float(K_YUV @ K_YUV))
    h, w = d.shape[0] // 2 * 2, d.shape[1] // 2 * 2
    blk = d[:h, :w].reshape(h // 2, 2, w // 2, 2, d.shape[2])
    d[:h, :w] = (blk - blk.mean(axis=(1, 3), keepdims=True)).reshape(h, w, d.shape[2])
    return d


def snap_to_blocks(mask: np.ndarray) -> np.ndarray:
    """Grow a scorer-lattice mask to whole 2x2 blocks.

    REQUIRED for exact nullity when a support is combined with the pose-null projection, not
    cosmetic -- two of the six constraints are BLOCK-mean conditions (mean dR = 0, mean dB = 0),
    so masking a projected delta at pixel granularity re-introduces a block mean and the edit
    stops being pose-null.  Same convention as sq1's `snap_band_to_blocks`, which rt1's eta gate
    already uses for exactly this reason.
    """
    h, w = mask.shape[0] // 2 * 2, mask.shape[1] // 2 * 2
    out = np.zeros_like(mask, dtype=bool)
    blk = mask[:h, :w].astype(bool).reshape(h // 2, 2, w // 2, 2).any(axis=(1, 3))
    out[:h, :w] = np.repeat(np.repeat(blk, 2, axis=0), 2, axis=1)
    return out


class Deblur:
    """Recover the native render from the shipped camera frame, and realize it exactly."""

    def __init__(self) -> None:
        self.rs = RT2.Resampler()
        self.Ur = lift_matrix(SEG_H, CAM_H)
        self.Uc = lift_matrix(SEG_W, CAM_W)
        # least-squares recovery m_hat = U^+ X  (per axis; U has full column rank)
        self.Ur_pinv = np.linalg.solve(self.Ur.T @ self.Ur, self.Ur.T)   # (SEG_H, CAM_H)
        self.Uc_pinv = np.linalg.solve(self.Uc.T @ self.Uc, self.Uc.T)   # (SEG_W, CAM_W)
        with np.errstate(all="ignore"):
            self.A_row = self.rs.R @ self.Ur
            self.A_col = self.rs.C @ self.Uc
            for nm, a in (("row", self.A_row), ("col", self.A_col)):
                band = (np.diag(np.diag(a)) + np.diag(np.diag(a, 1), 1)
                        + np.diag(np.diag(a, -1), -1))
                off = float(np.abs(a - band).max())
                if off > 1e-12:
                    raise LadderError(f"A_{nm} is not tridiagonal (off-band {off:g}); "
                                      f"the decoder lift is not bilinear -- re-identify it")

    def native(self, frame_cam: np.ndarray) -> np.ndarray:
        """m_hat = U^+ X -- the 384x512 field the decoder rendered, up to uint8 residue."""
        return np.einsum("oi,ijc,pj->opc", self.Ur_pinv, frame_cam.astype(np.float64),
                         self.Uc_pinv, optimize=True)

    def realize(self, frame_cam: np.ndarray, alpha: float,
                *, pose_null: bool = False,
                support: np.ndarray | None = None) -> tuple[np.ndarray, dict]:
        x = frame_cam.astype(np.float64)
        seen = self.rs.down_scorer(x)
        delta = alpha * (self.native(x) - seen)
        extra: dict[str, float] = {}
        if support is not None:
            # snap FIRST, then mask, then project: every block is wholly in or wholly out, so
            # the blockwise projection maps out-blocks 0 -> 0 and leaves nullity exact.
            delta = delta * support[..., None]
            extra["support_frac_scorer"] = float(support.mean())
        if pose_null:
            delta = project_pose_null(delta)
            # nullity is asserted, never assumed (the whole pose claim rests on it)
            dy = float(np.abs(delta @ K_YUV).max())
            h, w = delta.shape[0] // 2 * 2, delta.shape[1] // 2 * 2
            bm = float(np.abs(delta[:h, :w].reshape(h // 2, 2, w // 2, 2, 3)
                              .mean(axis=(1, 3))).max())
            extra |= {"nullity_max_abs_dY": dy, "nullity_max_abs_blockmean": bm}
            if support is not None and max(dy, bm) > 1e-9:
                raise LadderError(
                    f"support+projection broke nullity (dY {dy:g}, blockmean {bm:g}); "
                    f"the support snap is wrong -- no pose row from this run is admissible")
        target = seen + delta
        pre = self.rs.precompensate(x, target)
        out = np.clip(np.rint(pre), 0, 255).astype(np.uint8)
        got = self.rs.down_scorer(out.astype(np.float64))
        return out, {
            "realized_max_abs_err_vs_target": float(np.abs(got - target).max()),
            "realized_rms_err_vs_target": float(np.sqrt(((got - target) ** 2).mean())),
            "camera_rms_delta": float(np.sqrt(((out.astype(np.float64) - x) ** 2).mean())),
            "camera_max_delta": float(np.abs(out.astype(np.float64) - x).max()),
            "frac_out_of_box_pre_clip": float(((pre < 0) | (pre > 255)).mean()),
            "scorer_rms_move": float(np.sqrt(((target - seen) ** 2).mean())),
            "camera_frac_touched": float((out != frame_cam).any(axis=2).mean()),
        } | extra


def sha256_file(path: Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while (b := fh.read(chunk)):
            h.update(b)
    return h.hexdigest()


def seeded_pairs(n: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    return sorted(int(v) for v in rng.choice(FRAMES, size=n, replace=False))


def run(args: argparse.Namespace) -> int:
    pairs = (list(range(FRAMES)) if args.n_pairs >= FRAMES
             else seeded_pairs(args.n_pairs, args.seed))
    raw = RT2.open_raw(args.raw)
    gt = np.load(args.gt, mmap_mode="r")
    db = Deblur()
    inst = Instrument(args.threads, with_pose=not args.no_pose)

    sup_all = sup_meta = None
    if args.support is None and args.edit_frames != "both":
        raise LadderError("--edit-frames only applies with --support; without a support the "
                          "ladder is a whole-frame operator and both frames are always edited")
    if args.support is not None:
        sup_all = np.load(args.support, mmap_mode="r")
        if sup_all.shape != (FRAMES, SEG_H, SEG_W):
            raise LadderError(f"--support has shape {sup_all.shape}, expected "
                              f"{(FRAMES, SEG_H, SEG_W)} (scorer-lattice, per pair)")
        sup_meta = {"path": str(args.support), "sha256": sha256_file(args.support),
                    "edit_frames": args.edit_frames,
                    "snapped_to_2x2_blocks": bool(args.pose_null)}

    gtf = {}
    if not args.no_pose:
        wanted = set()
        for p in pairs:
            wanted |= {2 * p, 2 * p + 1}
        print(f"decoding {len(wanted)} GT frames (canonical yuv420_to_rgb) ...", flush=True)
        gtf = decode_gt(args.mkv, wanted)

    base_am, base_dpose = {}, {}
    for p in pairs:
        base_am[p] = inst.seg_argmax(np.asarray(raw[2 * p + 1]))
        if not args.no_pose:
            g = np.stack([gtf[2 * p], gtf[2 * p + 1]])
            base_dpose[p] = (inst.pose_out(g),
                             inst.d_pose(inst.pose_out(g),
                                         inst.pose_out(np.asarray(raw[2 * p:2 * p + 2]))))

    npx = len(pairs) * SEG_H * SEG_W
    base_flips = int(sum((base_am[p] != np.asarray(gt[p])).sum() for p in pairs))
    rows = []
    for alpha in args.alphas:
        t0 = time.time()
        flips = fixed = broken = 0
        dp_after: list[float] = []
        per_pair: list[dict] = []
        diag_acc: dict[str, list[float]] = {}
        for p in pairs:
            pair = np.asarray(raw[2 * p:2 * p + 2])
            edited = np.empty_like(pair)
            sup = None
            if sup_all is not None:
                sup = np.asarray(sup_all[p]).astype(bool)
                if args.pose_null:
                    sup = snap_to_blocks(sup)
            for j in range(2):
                # `f1` leaves frame_0 byte-identical: SegNet reads only frame_1
                # (`modules.py:108` x[:, -1]), so a seg-correction channel has no reason to
                # touch frame_0 -- but PoseNet reads BOTH, so this changes the pose leg and is
                # measured as its own arm rather than assumed equivalent.
                if sup is not None and args.edit_frames == "f1" and j == 0:
                    edited[j] = pair[j]
                    continue
                edited[j], d = db.realize(pair[j], alpha,
                                          pose_null=args.pose_null, support=sup)
                for k, v in d.items():
                    diag_acc.setdefault(k, []).append(v)
            am = inst.seg_argmax(edited[1])
            gtp = np.asarray(gt[p])
            was, now = base_am[p] != gtp, am != gtp
            flips += int(now.sum())
            fixed += int((was & ~now).sum())
            broken += int((~was & now).sum())
            # PER-PAIR RETENTION (P0 always-keep-the-payload, and the scientific point): the
            # scorer-convention aggregate is a ratio of MEANS, so a handful of heavy-d_pose pairs
            # can dominate it and make the aggregate non-monotone in alpha.  Keeping the per-pair
            # rows is what lets a reader tell "the leak is small" from "the estimator is noisy".
            rec = {"pair": int(p), "flips": int(now.sum()),
                   "fixed": int((was & ~now).sum()), "broken": int((~was & now).sum())}
            if not args.no_pose:
                og, base_dp = base_dpose[p]
                dpa = inst.d_pose(og, inst.pose_out(edited))
                dp_after.append(dpa)
                rec |= {"d_pose_before": float(base_dp), "d_pose_after": float(dpa),
                        "d_pose_ratio_pair": float(dpa / base_dp) if base_dp else float("nan")}
            per_pair.append(rec)
        d_flips = flips - base_flips
        scale = SCORED_PX / npx
        dS_seg = d_flips * scale * S_PER_FLIP
        row = {
            "alpha": alpha,
            "pose_null_projected": args.pose_null,
            "flips": flips,
            "base_flips": base_flips,
            "delta_flips": d_flips,
            "delta_flips_n600_equivalent": d_flips * scale,
            "fixed": fixed,
            "broken": broken,
            "broken_per_fixed": (broken / fixed if fixed else float("inf")),
            "pct_of_base_recovered": -100.0 * d_flips / base_flips,
            "delta_S_seg": dS_seg,
            "per_pair": per_pair,
            "diagnostics_mean": {k: float(np.mean(v)) for k, v in diag_acc.items()},
            "diagnostics_max": {k: float(np.max(v)) for k, v in diag_acc.items()},
            "wall_s": time.time() - t0,
        }
        if not args.no_pose:
            # scorer convention: aggregate d_pose ITSELF, never a mean of per-pair ratios
            # (rt1 s6.2b -- the two disagree in SIGN).
            mb = float(np.mean([v for _, v in base_dpose.values()]))
            ma = float(np.mean(dp_after))
            dS_pose = np.sqrt(10.0 * ma) - np.sqrt(10.0 * mb)
            row |= {
                "d_pose_before": mb, "d_pose_after": ma,
                "d_pose_ratio_scorer_convention": (ma / mb if mb else float("nan")),
                "delta_S_pose": float(dS_pose),
                "delta_S_net": float(dS_seg + dS_pose),
                "pose_leg_over_seg_leg": (abs(dS_pose) / abs(dS_seg)
                                          if dS_seg else float("inf")),
            }
        rows.append(row)
        print(json.dumps({k: v for k, v in row.items()
                          if k not in ("diagnostics_mean", "diagnostics_max", "per_pair")},
                         sort_keys=True), flush=True)

    ctrl = next((r for r in rows if r["alpha"] == 0.0), None)
    control_ok = (ctrl is not None and ctrl["delta_flips"] == 0
                  and ctrl["fixed"] == 0 and ctrl["broken"] == 0)
    receipt = {
        "schema": "ddm_rt2_deblur_ladder.v1",
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotable": False,
        "pointer_moved": False,
        "n_pairs": len(pairs),
        "seed": args.seed,
        "pairs_are_seeded_random_not_prefix": True,
        "pairs": pairs,
        "pose_measured": not args.no_pose,
        "pose_null_projected": args.pose_null,
        "support": sup_meta,
        "positive_control_alpha0_reproduces_base": control_ok,
        "operator_identification": {
            "decoder_lift_U": "bilinear (bicubic leaves 1.54e-2 off the tridiagonal band)",
            "A_row_middle": db.A_row[SEG_H // 2, SEG_H // 2 - 1:SEG_H // 2 + 2].tolist(),
            "A_col_middle": db.A_col[SEG_W // 2, SEG_W // 2 - 1:SEG_W // 2 + 2].tolist(),
            "A_row_offband_max": 0.0,
            "A_col_offband_max": 0.0,
        },
        "base": {"S": BASE_S, "bytes": BASE_BYTES, "gap_to_0p15": GAP_TO_015},
        "exchange": {"S_per_flip": S_PER_FLIP, "S_per_byte": S_PER_BYTE},
        "rows": rows,
        "instrument": {"backend": "cpu-torch", "batch_pairs": 1, "threads": args.threads,
                       "reference_form": "ddm_rn1_render_boundary_mechanism.Instrument"},
    }
    if not control_ok and any(r["alpha"] == 0.0 for r in rows):
        receipt["VOID"] = ("alpha=0 did not reproduce the base exactly; every row in this "
                           "receipt is instrument-confounded and MUST NOT be cited")
    tag = (f"n{len(pairs)}_s{args.seed}" + ("_posenull" if args.pose_null else "")
           + ("_nopose" if args.no_pose else "")
           + (f"_sup{args.edit_frames}" if sup_all is not None else ""))
    out = args.work / f"RT2_DEBLUR_LADDER_{tag}.json"
    args.work.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True))
    print(f"\nwrote {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--raw", type=Path, default=RT2.DEFAULT_RAW)
    ap.add_argument("--gt", type=Path, default=RT2.DEFAULT_GT)
    ap.add_argument("--mkv", type=Path, default=DEFAULT_MKV)
    ap.add_argument("--work", type=Path, default=DEFAULT_WORK)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--n-pairs", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20260817)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.0, 0.25, 0.5, 0.75, 1.0])
    ap.add_argument("--pose-null", action="store_true",
                    help="project the perturbation onto PoseNet's exact null space")
    ap.add_argument("--no-pose", action="store_true",
                    help="seg-only -- DIAGNOSTIC ONLY; rn1 measured pose at 25-70x the seg leg")
    ap.add_argument("--support", type=Path, default=None,
                    help="(FRAMES,384,512) scorer-lattice per-pair mask restricting the edit; "
                         "snapped to 2x2 blocks when --pose-null so nullity stays exact")
    ap.add_argument("--edit-frames", choices=["both", "f1"], default="both",
                    help="with --support: 'both' matches the full-frame ladder's convention; "
                         "'f1' is the seg-channel shape (SegNet reads only frame_1)")
    args = ap.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
