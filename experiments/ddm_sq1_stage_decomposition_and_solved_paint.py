#!/usr/bin/env python
"""ddm_sq1 Job 1b -- WHERE realization dies, per stage / per class / per edge, and whether the
named v14 cure moves it.

Operator steer 2026-08-03 (binding): "Everything is realizable with the proper engineering."
A low eta falsifies the REALIZER, not the in-band-GT family (Catalog #307).  So this script
does NOT emit a row verdict.  It decomposes the realization chain

    description -> painted RGB -> through R/D -> uint8 -> frozen SegNet argmax

and reports where the loss actually sits, then tests the cure the playbook names for that
stage.  Realizer v0 (ddm_sq1_eta_seg_realization.py) skipped every cure; this is v1.

STAGE MAP and what is measured here
  S0 address    flips the free band does not even contain          -> capture, per class/edge
  S1 paint      the colour chosen for a described pixel            -> EXACTNESS CHECK
  S2 R / D      camera 874x1164 -> scorer 384x512 bilinear         -> EXACTNESS CHECK
  S3 uint8      camera-plane quantisation of the paint             -> EXACTNESS CHECK
  S4 argmax     the frozen net's REGIONAL response                 -> the residual

S1+S2+S3 are checked numerically rather than assumed: m86 measured D's 2x2 supports to be
PRIVATE, and bilinear weights sum to 1, so painting all 4 camera pixels of a scorer pixel with
one uint8 value should reproduce that value EXACTLY at the scorer lattice.  If that holds, the
whole realization debt is localised to S4 and the paint/AA/uint8 cures are NOT the binding ones.

CURE TESTED (v14 playbook, the S4 item): margin-optimal prototype colours SOLVED from the
frozen head, with the two named riders -- multi-start (pu2 mechanism) and in-loop REALIZED-flip
validation with best-iterate retention (fd2/tb1 lesson).  v0's paints were truth (L0) and class
mean (L1); neither is margin-optimal, so neither tests the physics.

Axis: [macOS-CPU frozen-scorer advisory] NON-PROMOTABLE.  score_claim=false.
Pointer 0.1910828242 [contest-CPU] UNMOVED.
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

from ddm_sq1_eta_seg_realization import (  # noqa: E402
    CAM_H,
    CAM_W,
    CLASS_NAMES,
    COL_SUP,
    N_CLASSES,
    N_PAIRS_TOTAL,
    ROW_SUP,
    SEG_H,
    SEG_W,
    Scorer,
    decode_gt_frames,
    label_boundary_band,
    seq_len,
)
from tac.optimization.trajectory_stopping import (  # noqa: E402
    TrajectoryPoint,
    TrajectoryStopConfig,
    byte_score_units,
    evaluate_trajectory_stop,
    seg_flip_score_units,
)

SQ1_TRAJECTORY_STOP_CONFIG = TrajectoryStopConfig(
    score_units_per_objective=seg_flip_score_units(
        n_pairs=N_PAIRS_TOTAL,
        height=SEG_H,
        width=SEG_W,
    ),
    # Derived bar: one counted archive byte per solver step in the exact score law.
    marginal_score_gain_per_compute=byte_score_units(),
)


def resize_to_scorer(cam_u8: np.ndarray) -> torch.Tensor:
    """The scorer's own D, applied to one camera frame -> (1,3,384,512) float."""
    x = torch.from_numpy(np.ascontiguousarray(cam_u8))[None].permute(0, 3, 1, 2).float()
    return torch.nn.functional.interpolate(x, size=(SEG_H, SEG_W), mode="bilinear")


def realize_scorer_paint_to_camera(dec_f1: np.ndarray, band: np.ndarray,
                                   paint_u8: np.ndarray) -> np.ndarray:
    """Write a scorer-lattice uint8 paint into the camera plane.

    Sets all four camera pixels of every band scorer pixel to that pixel's paint value.  Because
    D's supports are private and its 4 bilinear weights sum to 1, D then reproduces `paint_u8`
    EXACTLY on the band and leaves every off-band scorer pixel untouched.
    """
    out = dec_f1.copy()
    ii, jj = np.nonzero(band)
    vals = paint_u8[ii, jj]
    for a in range(2):
        r = ROW_SUP[ii, a]
        for b in range(2):
            c = COL_SUP[jj, b]
            out[r, c] = vals
    return out


def confusion(gt: np.ndarray, rd: np.ndarray) -> np.ndarray:
    """Directed C[gt_class][rendered_class] (pc2: decompose per EDGE, never per class)."""
    return np.bincount((gt.astype(np.int64) * N_CLASSES + rd.astype(np.int64)).ravel(),
                       minlength=N_CLASSES * N_CLASSES).reshape(N_CLASSES, N_CLASSES)


def solve_margin_optimal_paint(segnet, dec_f1: np.ndarray, gt_f1: np.ndarray,
                               band: np.ndarray, lgt: np.ndarray, *, steps: int, lr: float,
                               eval_every: int, convergence_patience_evals: int = 0,
                               convergence_min_improvement: int = 1) -> tuple:
    """S4 cure: solve the in-band scorer-lattice colours against the FROZEN head.

    Two starts (pu2 multi-start rider): the decoded colours, and the truth colours.
    In-loop REALIZED-argmax validation with best-iterate retention (fd2/tb1 rider) -- the
    proxy CE is never trusted to pick the iterate.
    """
    base = resize_to_scorer(dec_f1)                       # (1,3,384,512)
    truth = resize_to_scorer(gt_f1)
    tgt = torch.from_numpy(lgt.astype(np.int64))[None]    # (1,384,512)
    m = torch.from_numpy(band)[None, None].float()        # (1,1,384,512)

    best = None
    diagnostics = {
        "steps_requested": int(steps),
        "lr": float(lr),
        "eval_every": int(eval_every),
        "convergence_patience_evals": int(convergence_patience_evals),
        "convergence_min_improvement": int(convergence_min_improvement),
        "starts": [],
    }
    # Scorer.__init__ disables grad process-wide (inference harness); the solve needs it back.
    with torch.enable_grad():
        for start_name, start in (("dec", base), ("truth", truth)):
            x = start.clone().detach()
            delta = torch.zeros_like(base, requires_grad=True)
            opt = torch.optim.Adam([delta], lr=lr)
            start_diag = {
                "start": start_name,
                "curve": [],
                "stop_reason": "iteration_cap",
                "trajectory_stop": None,
                "steps_run": 0,
                "best_step": None,
                "best_proxy_flips": None,
            }
            start_best = None
            evals_since_best = 0
            for it in range(steps + 1):
                cur = torch.clamp(base * (1 - m) + (x + delta) * m, 0.0, 255.0)
                if it % eval_every == 0 or it == steps:
                    q = torch.round(cur).detach()
                    with torch.no_grad():
                        lam = segnet(q).argmax(dim=1)[0].numpy().astype(np.uint8)
                    n_bad = int((lam != lgt).sum())
                    start_diag["curve"].append({"step": int(it), "proxy_flips": n_bad})
                    start_diag["steps_run"] = int(it)
                    improved = (
                        start_best is None
                        or int(start_best[0]) - n_bad >= max(1, int(convergence_min_improvement))
                    )
                    if improved:
                        start_best = (n_bad, int(it))
                        start_diag["best_step"] = int(it)
                        start_diag["best_proxy_flips"] = n_bad
                        evals_since_best = 0
                    else:
                        evals_since_best += 1
                    if best is None or n_bad < best[0]:
                        best = (n_bad, q[0].permute(1, 2, 0).numpy().astype(np.uint8),
                                f"{start_name}@{it}", start_name)
                    if len(start_diag["curve"]) >= SQ1_TRAJECTORY_STOP_CONFIG.min_fit_points:
                        stop_decision = evaluate_trajectory_stop(
                            [
                                TrajectoryPoint(
                                    compute=float(point["step"]),
                                    objective=float(point["proxy_flips"]),
                                )
                                for point in start_diag["curve"]
                            ],
                            SQ1_TRAJECTORY_STOP_CONFIG,
                            safety_bound_compute=float(steps),
                        )
                        start_diag["trajectory_stop"] = stop_decision.to_payload()
                        if (
                            stop_decision.should_stop
                            and stop_decision.stop_reason
                            in {"converged_projected", "marginal_below_bar"}
                            and it < steps
                        ):
                            start_diag["stop_reason"] = stop_decision.stop_reason
                            break
                    if (
                        convergence_patience_evals > 0
                        and evals_since_best >= convergence_patience_evals
                        and it < steps
                    ):
                        start_diag["stop_reason"] = "plateau_no_proxy_improvement"
                        break
                if it == steps:
                    if len(start_diag["curve"]) >= SQ1_TRAJECTORY_STOP_CONFIG.min_fit_points:
                        stop_decision = evaluate_trajectory_stop(
                            [
                                TrajectoryPoint(
                                    compute=float(point["step"]),
                                    objective=float(point["proxy_flips"]),
                                )
                                for point in start_diag["curve"]
                            ],
                            SQ1_TRAJECTORY_STOP_CONFIG,
                            safety_bound_compute=float(steps),
                        )
                        start_diag["trajectory_stop"] = stop_decision.to_payload()
                    if start_diag["best_step"] == steps:
                        start_diag["stop_reason"] = "iteration_cap_best_at_cap"
                    elif convergence_patience_evals <= 0:
                        start_diag["stop_reason"] = "iteration_cap_no_convergence_test"
                    else:
                        start_diag["stop_reason"] = "iteration_cap_before_plateau"
                    break
                logits = segnet(cur)
                loss = torch.nn.functional.cross_entropy(logits, tgt)
                opt.zero_grad()
                loss.backward()
                opt.step()
            diagnostics["starts"].append(start_diag)
    selected_start = best[3]
    selected_diag = next(d for d in diagnostics["starts"] if d["start"] == selected_start)
    diagnostics["selected"] = {
        "start": selected_start,
        "tag": best[2],
        "best_step": int(best[2].rsplit("@", 1)[1]),
        "best_proxy_flips": int(best[0]),
        "stop_reason": selected_diag["stop_reason"],
        "trajectory_stop": selected_diag["trajectory_stop"],
        "steps_run": selected_diag["steps_run"],
        "curve": selected_diag["curve"],
    }
    return best[:3] + (diagnostics,)  # proxy flips, paint_u8 HWC, tag, diagnostics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub-dir", type=Path, required=True)
    ap.add_argument("--gt-mkv", type=Path, required=True)
    ap.add_argument("--pairs-npy", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--lr", type=float, default=4.0)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--convergence-patience-evals", type=int, default=0,
                    help="stop a start after this many eval points without proxy-flip improvement")
    ap.add_argument("--convergence-min-improvement", type=int, default=1,
                    help="minimum proxy-flip drop that resets the convergence patience")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    pairs = np.load(args.pairs_npy).tolist()
    if args.limit:
        pairs = pairs[: args.limit]

    raw = np.memmap(args.sub_dir / "inflated" / "0.raw", dtype=np.uint8, mode="r",
                    shape=(N_PAIRS_TOTAL * seq_len, CAM_H, CAM_W, 3))
    wanted = set()
    for p in pairs:
        wanted.update({seq_len * p, seq_len * p + 1})
    gt_frames = decode_gt_frames(args.gt_mkv, wanted)
    sc = Scorer(args.threads)
    segnet = sc.net.segnet
    print(f"[sq1b] ready t={time.time()-t0:.1f}s, {len(pairs)} pairs", flush=True)

    rows = []
    if args.resume and args.out.exists():          # resumable-from-disk (P0 non-negotiable)
        rows = json.loads(args.out.read_text()).get("rows", [])
        done = {int(r["pair"]) for r in rows}
        pairs = [p for p in pairs if p not in done]
        print(f"[sq1b] resume: {len(rows)} rows on disk, {len(pairs)} remaining", flush=True)

    for n, p in enumerate(pairs):
        tp = time.time()
        dec = np.stack([raw[seq_len * p], raw[seq_len * p + 1]]).astype(np.uint8)
        gt = np.stack([gt_frames[seq_len * p], gt_frames[seq_len * p + 1]])
        lstar = sc.seg_argmax(dec)
        lgt = sc.seg_argmax(gt)
        band = label_boundary_band(lstar, 1)
        flips0 = lstar != lgt
        C0 = confusion(lgt, lstar)

        rec = {"pair": int(p), "flips_before": int(flips0.sum()),
               "band_frac": float(band.mean()),
               "C_before": C0.tolist(),
               "described_in_band": int((flips0 & band).sum())}

        # ---- S1/S2/S3 EXACTNESS: is the paint delivered to the scorer lattice unchanged? ----
        # Paint the band with truth's scorer-lattice colours, realized through camera+uint8.
        truth_sc = torch.round(resize_to_scorer(gt[1]))[0].permute(1, 2, 0).numpy()
        truth_sc_u8 = np.clip(truth_sc, 0, 255).astype(np.uint8)
        cam_edit = realize_scorer_paint_to_camera(dec[1], band, truth_sc_u8)
        got = resize_to_scorer(cam_edit)[0].permute(1, 2, 0).numpy()
        base_sc = resize_to_scorer(dec[1])[0].permute(1, 2, 0).numpy()
        err_band = np.abs(got[band] - truth_sc_u8[band]).max() if band.any() else 0.0
        err_off = np.abs(got[~band] - base_sc[~band]).max() if (~band).any() else 0.0
        rec["S123_max_abs_err_on_band"] = float(err_band)
        rec["S123_max_abs_err_off_band"] = float(err_off)

        def score_camera(cam_f1, tag):
            lam = sc.seg_argmax(np.stack([dec[0], cam_f1]))
            fa = lam != lgt
            C = confusion(lgt, lam)
            pose_gt = sc.pose_out(gt)
            return {
                f"{tag}_flips_after": int(fa.sum()),
                f"{tag}_fixed": int((flips0 & ~fa).sum()),
                f"{tag}_introduced": int((~flips0 & fa).sum()),
                f"{tag}_C_after": C.tolist(),
                f"{tag}_d_pose_after": sc.d_pose(pose_gt, sc.pose_out(np.stack([dec[0], cam_f1]))),
            }

        # ---- S4 residual with an EXACT paint (truth) ----------------------------------------
        rec.update(score_camera(cam_edit, "S4_truthpaint"))

        # ---- the named S4 CURE: margin-optimal solved paint, multi-start, best-realized ----
        nbad, paint, tag, solve_diag = solve_margin_optimal_paint(
            segnet, dec[1], gt[1], band, lgt,
            steps=args.steps, lr=args.lr, eval_every=args.eval_every,
            convergence_patience_evals=args.convergence_patience_evals,
            convergence_min_improvement=args.convergence_min_improvement)
        rec["solved_start_tag"] = tag
        rec["solved_proxy_flips_scorer_lattice"] = int(nbad)
        rec["solved_stop_reason"] = solve_diag["selected"]["stop_reason"]
        if solve_diag["selected"].get("trajectory_stop") is not None:
            rec["solved_trajectory_stop"] = solve_diag["selected"]["trajectory_stop"]
            rec["solved_trajectory_stop_reason"] = solve_diag["selected"]["trajectory_stop"][
                "stop_reason"
            ]
        rec["solved_best_step"] = solve_diag["selected"]["best_step"]
        rec["solved_steps_run"] = solve_diag["selected"]["steps_run"]
        rec["solved_best_start"] = solve_diag["selected"]["start"]
        rec["solved_convergence_curve"] = solve_diag["selected"]["curve"]
        rec["solved_all_start_diagnostics"] = solve_diag["starts"]
        cam_solved = realize_scorer_paint_to_camera(dec[1], band, paint)
        rec.update(score_camera(cam_solved, "S4_solvedpaint"))

        rows.append(rec)
        d = rec["described_in_band"]
        print(f"[sq1b] pair {p:3d} ({n+1}/{len(pairs)}) desc {d:5d} | S123err band {err_band:.3g} "
              f"off {err_off:.3g} | truthpaint after {rec['S4_truthpaint_flips_after']:5d} "
              f"| SOLVED proxy {nbad:5d} realized {rec['S4_solvedpaint_flips_after']:5d} "
              f"(was {rec['flips_before']:5d}) {rec['solved_stop_reason']} "
              f"best@{rec['solved_best_step']} [{time.time()-tp:.1f}s]", flush=True)

        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"schema": "ddm_sq1_stage_decomposition.v1",
                       "axis": "[macOS-CPU frozen-scorer advisory] NON-PROMOTABLE",
                       "score_claim": False, "promotion_eligible": False,
                       "pointer": "0.1910828242 [contest-CPU] UNMOVED",
                       "class_names": CLASS_NAMES,
                       "solver": {"steps": args.steps, "lr": args.lr,
                                  "eval_every": args.eval_every,
                                  "convergence_patience_evals":
                                  args.convergence_patience_evals,
                                  "convergence_min_improvement":
                                  args.convergence_min_improvement,
                                  "starts": ["dec", "truth"],
                                  "riders": ["multi_start_pu2", "in_loop_realized_flip_fd2_tb1"]},
                       "pairs": pairs, "rows": rows}, f, indent=1)

    print(f"[sq1b] DONE t={time.time()-t0:.1f}s -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
