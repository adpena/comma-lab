#!/usr/bin/env python3
"""ddm_rt1 ETA GATE -- pose-constrained realization efficiency on the hv1 ep0634 base.

The decision this pays.  rt1 §5 measured the free-band seg-correction channel at 32,270 real
verified bytes (+965 B target class) and showed the coder was never the binding constraint:
the channel breaks even only if realization efficiency **eta > 0.753**.  sq1 measured
eta = 0.7895 free / 0.5406 pose-constrained ON THE v4d VEHICLE.  Nothing has measured eta on
hv1.  This module does, under the pose constraint, because any shippable version pays it.

PRE-REGISTERED BAR (written before any solve):
    eta > 0.753  -> family LIVE   -> route to a build fire-order for the full-context coder
    eta <= 0.753 -> family CLOSED on arithmetic, verdict_scope: this vehicle, this band,
                    pose-constrained realization.

OPTIMAL FORM.  The solver is sq1's own, imported not reimplemented:
`solve_null_constrained` (hard pose-null projection -- frame_1's yuv6 is left unchanged, so the
constraint is structural rather than a soft penalty), `realize_scorer_paint_to_camera`,
`Scorer`, `decode_gt_frames`.  Riders carried: multi-start (pu2), in-loop REALIZED-argmax
validation with best-iterate retention (fd2/tb1 -- the proxy CE never picks the iterate),
whole-frame accounting (a cure that fixes 900 band pixels while breaking 400 elsewhere has not
delivered 900), and d_pose RECOMPUTED against decoded GT (qs4's stale-compensation lesson).

SAMPLING (m96).  Pairs are drawn by a seeded RANDOM choice, never a [:n] prefix: pose-axis
prefix bias runs 2.5-4.2x anti-conservative.  A subset may REFUTE the bar; a subset PASS does
not license a LIVE verdict without the fuller population.

Axis `[macOS-CPU advisory]` -- NEVER a score.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))

from ddm_sq1_eta_seg_realization import (
    Scorer,
    decode_gt_frames,
    dilate,
)
from ddm_sq1_pose_null_constrained_paint import (
    pose_null_projector,
    project_null,
    snap_band_to_blocks,
    yuv6_shift,
)
from ddm_sq1_stage_decomposition_and_solved_paint import (
    realize_scorer_paint_to_camera,
    resize_to_scorer,
)

FRAMES, SEG_H, SEG_W, CAM_H, CAM_W = 600, 384, 512, 874, 1164
SCORED_PX = FRAMES * SEG_H * SEG_W
ETA_BAR = 0.753                       # rt1 §5.4, mask 32,270 B + target 965 B
SEG_DS_PER_FLIP = 100.0 / SCORED_PX
RATE_DS_PER_BYTE = 25.0 / 37_545_489
CHANNEL_BYTES = 32270 + 965
HV1_S = 0.15959729295498598

DEFAULT_WORK = Path("/Volumes/APDataStore/pact/ddm_rt1_seg_roundtrip_20260816")
DEFAULT_RAW = Path(
    "/Volumes/APDataStore/pact/ddm_wc1_advisory_decode_wallclock_20260815/runs/"
    "base_optimized_n600_r3/output/0.raw"
)
DEFAULT_TOKENS = Path(
    "/Volumes/APDataStore/pact/ddm_hv1_harvest_compose/ep0634/retained/coders/"
    "s1p25_c1p0/decoded_spatial_tokens.rc64.bin"
)
DEFAULT_GT_ARGMAX = Path(
    "/Volumes/VertigoDataTier/pact/ddm_qs3_20260813/retained/inputs/gt_argmax_n600.npy"
)
DEFAULT_GT_MKV = REPO / "upstream" / "videos" / "0.mkv"


class EtaGateError(RuntimeError):
    """Fail-closed error for custody violations."""


def solve_multistart(segnet, dec_f1: np.ndarray, gt_f1: np.ndarray, band: np.ndarray,
                     lgt: np.ndarray, described: np.ndarray, P, *, steps: int, lr: float,
                     eval_every: int, pose_null: bool, focus_weight: float) -> dict:
    """sq1's solve, with ONE declared operating-point adaptation.

    Riders carried verbatim from sq1: multi-start (pu2), in-loop REALIZED-argmax validation with
    best-iterate retention (fd2/tb1), hard pose-null projection when `pose_null` (frame_1's yuv6
    is left unchanged, so the pose constraint is structural, not a penalty).

    DECLARED ADAPTATION -- sq1's objective does NOT transfer to this operating point.  sq1 used
    a plain whole-frame cross-entropy.  On v4d that was ~848 wrong pixels per pair; on hv1 it is
    ~50 wrong out of 196,608, so the wrong pixels carry 0.025% of the loss and the gradient
    spends the band's capacity re-confirming pixels that are already right.  MEASURED signature
    of exactly that: with sq1's loss the best iterate is step 0 for every pair and both modes --
    the solver never moves.  So the CE is reweighted by `focus_weight` on the described set,
    which keeps the collateral term (every other pixel still contributes at weight 1) while
    making the target term visible.  This is an operating-point adaptation of the objective, not
    a different mechanism, and it is declared here rather than buried.
    """
    import torch

    base = resize_to_scorer(dec_f1)
    truth = resize_to_scorer(gt_f1)
    tgt = torch.from_numpy(lgt.astype(np.int64))[None]
    m = torch.from_numpy(snap_band_to_blocks(band) if pose_null else band)[None, None].float()
    w = torch.ones((1, SEG_H, SEG_W), dtype=torch.float32)
    w[0][torch.from_numpy(described)] = float(focus_weight)
    w_sum = w.sum()

    best = None
    starts = []
    with torch.enable_grad():
        for start_name, init in (("zero", None), ("truth_dir", truth - base)):
            raw = (torch.zeros_like(base) if init is None else init.clone()).requires_grad_(True)
            opt = torch.optim.Adam([raw], lr=lr)
            curve = []
            for it in range(steps + 1):
                d = project_null(raw, P) if pose_null else raw
                cur = torch.clamp(base + d * m, 0.0, 255.0)
                if it % eval_every == 0 or it == steps:
                    q = torch.round(cur).detach()
                    with torch.no_grad():
                        lam = segnet(q).argmax(dim=1)[0].numpy().astype(np.uint8)
                    n_bad = int((lam != lgt).sum())
                    curve.append({"step": it, "proxy_flips": n_bad})
                    if best is None or n_bad < best[0]:
                        best = (n_bad, q[0].permute(1, 2, 0).numpy().astype(np.uint8),
                                f"{start_name}@{it}")
                if it == steps:
                    break
                ce = torch.nn.functional.cross_entropy(segnet(cur), tgt, reduction="none")
                loss = (ce * w).sum() / w_sum
                opt.zero_grad()
                loss.backward()
                opt.step()
            starts.append({"start": start_name, "curve": curve})
    return {"proxy_flips": best[0], "paint_u8": best[1], "tag": best[2], "starts": starts}


def run(args: argparse.Namespace) -> int:
    frame_bytes = CAM_H * CAM_W * 3
    n_raw = args.raw.stat().st_size // frame_bytes
    if n_raw != 2 * FRAMES:
        raise EtaGateError(f"{args.raw} holds {n_raw} frames, expected {2 * FRAMES}")
    raw = np.memmap(args.raw, dtype=np.uint8, mode="r", shape=(n_raw, CAM_H, CAM_W, 3))
    tok = np.memmap(args.tokens, dtype=np.uint8, mode="r", shape=(FRAMES, SEG_H, SEG_W))
    gt_argmax = np.load(args.gt_argmax, mmap_mode="r")
    base_argmax = np.load(args.work / "argmax_base.npy", mmap_mode="r")  # rt1, do NOT re-derive

    rng = np.random.default_rng(args.seed)
    pairs = sorted(rng.choice(FRAMES, size=args.n_pairs, replace=False).tolist())

    wanted = set()
    for t in pairs:
        wanted.update({2 * t, 2 * t + 1})
    t_gt = time.time()
    gt_frames = decode_gt_frames(args.gt_mkv, wanted)
    gt_decode_s = time.time() - t_gt

    sc = Scorer(args.threads)
    segnet = sc.net.segnet
    P = pose_null_projector()

    outdir = args.out
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    t0 = time.time()
    for k, t in enumerate(pairs):
        lab = np.asarray(tok[t])
        lgt = np.asarray(gt_argmax[t])
        dec0 = np.asarray(raw[2 * t])
        dec1 = np.asarray(raw[2 * t + 1])
        gt0, gt1 = gt_frames[2 * t], gt_frames[2 * t + 1]

        a0 = np.asarray(base_argmax[t])
        flips0 = a0 != lgt
        described = flips0 & _boundary(lab)          # ring-0: exactly what §5 priced
        # ACTUATOR SUPPORT: the DESCRIBED set dilated, not sq1's full label-boundary band.
        # The decoder decodes the flip mask, so it knows the described set -- dilating it costs
        # zero bytes and is at least as legal as the band.  MEASURED reason for the change:
        # sq1's band is ~11,377 px to fix ~37, and editing all of it drives whole-frame flips
        # 35 -> 343 (collateral 8x the gain) so best-iterate retention correctly refuses every
        # iterate and eta pins at ~0.  On the described-set support eta is 0.51-0.69.  sq1's
        # support convention suits its ~848-flip/pair density, not this vehicle's ~50.
        support = described if args.radius == 0 else dilate(described, args.radius)
        n_desc = int(described.sum())
        if n_desc == 0:
            continue

        # mode "free" is the unconstrained POSITIVE CONTROL: if it cannot realize flips either,
        # the instrument is broken and no verdict is admissible (confound-hunt L3).
        pose_null = args.mode == "null"
        sol = solve_multistart(segnet, dec1, gt1, support, lgt, described, P,
                               steps=args.steps, lr=args.lr, eval_every=args.eval_every,
                               pose_null=pose_null, focus_weight=args.focus_weight)
        edit_mask = snap_band_to_blocks(support) if pose_null else support
        cam_edit = realize_scorer_paint_to_camera(dec1, edit_mask, sol["paint_u8"])
        # realization fidelity: D must reproduce the solved paint on the edited support
        got = resize_to_scorer(cam_edit)[0].permute(1, 2, 0).numpy()
        realize_err = float(np.abs(got[edit_mask] - sol["paint_u8"][edit_mask]).max()) \
            if edit_mask.any() else 0.0
        lam = sc.seg_argmax(np.stack([dec0, cam_edit]))
        flips_after = lam != lgt
        n_before, n_after = int(flips0.sum()), int(flips_after.sum())

        pose_gt = sc.pose_out(np.stack([gt0, gt1]))
        d_pose_before = sc.d_pose(pose_gt, sc.pose_out(np.stack([dec0, dec1])))
        d_pose_after = sc.d_pose(pose_gt, sc.pose_out(np.stack([dec0, cam_edit])))

        row = {
            "pair": int(t),
            "n_described_ring0": n_desc,
            "flips_before": n_before,
            "flips_after": n_after,
            "fixed": int((flips0 & ~flips_after).sum()),
            "introduced": int((~flips0 & flips_after).sum()),
            "eta_net": (n_before - n_after) / n_desc,
            "proxy_flips_scorer_lattice": sol["proxy_flips"],
            "solved_start_tag": sol["tag"],
            "realize_max_abs_err_on_support": realize_err,
            "d_pose_before": d_pose_before,
            "d_pose_after": d_pose_after,
            "d_pose_ratio": (d_pose_after / d_pose_before) if d_pose_before else None,
            "yuv6_shift": yuv6_shift(dec1, cam_edit),
            "support_px": int(support.sum()),
            "snap_tax": float(snap_band_to_blocks(support).sum() / max(support.sum(), 1)),
        }
        rows.append(row)
        if args.retain_frames:
            np.save(outdir / f"cam_edit_pair{t:04d}.npy", cam_edit)
        (outdir / "ETA_GATE_ROWS.jsonl").open("a").write(json.dumps(row) + "\n")
        print(f"  [{k + 1}/{len(pairs)}] pair {t}: eta={row['eta_net']:+.4f} "
              f"desc={n_desc} {n_before}->{n_after} dpose x{row['d_pose_ratio']:.3f}", flush=True)

    etas = np.array([r["eta_net"] for r in rows], dtype=np.float64)
    desc = np.array([r["n_described_ring0"] for r in rows], dtype=np.float64)
    before = np.array([r["flips_before"] for r in rows], dtype=np.float64)
    after = np.array([r["flips_after"] for r in rows], dtype=np.float64)
    eta_pooled = float((before.sum() - after.sum()) / desc.sum())
    ratios = np.array([r["d_pose_ratio"] for r in rows if r["d_pose_ratio"] is not None])

    gain_all = 34666 * SEG_DS_PER_FLIP
    cost = CHANNEL_BYTES * RATE_DS_PER_BYTE
    verdict = "LIVE" if eta_pooled > ETA_BAR else "CLOSED"
    rec = {
        "schema": "ddm_rt1_eta_gate.v1",
        "axis": "[macOS-CPU advisory] -- NEVER a score",
        "score_claim": False,
        "promotable": False,
        "pre_registered_bar": ETA_BAR,
        "bar_provenance": f"rt1 §5.4: channel = {CHANNEL_BYTES} B "
                          "(M7 mask 32,270 + target class 965)",
        "sampling": {"seed": args.seed, "method": "seeded random choice without replacement "
                                                  "(m96: never a [:n] prefix)",
                     "n_pairs_requested": args.n_pairs, "n_pairs_measured": len(rows),
                     "pairs": [r["pair"] for r in rows]},
        "solver": {"reference_form": "sq1 solve_null_constrained + pu2 multi-start rider",
                   "steps": args.steps, "lr": args.lr, "eval_every": args.eval_every,
                   "focus_weight": args.focus_weight,
                   "starts": 2, "pose_constraint": ("hard yuv6 null-space projection" if args.mode == "null"
                                       else "NONE -- unconstrained positive control"),
                   "mode": args.mode,
                   "support_radius": args.radius},
        "eta": {
            "pooled": eta_pooled,
            "per_pair_mean": float(etas.mean()), "per_pair_sd": float(etas.std(ddof=1))
            if etas.size > 1 else None,
            "min": float(etas.min()), "max": float(etas.max()),
            "pairs_above_bar": int((etas > ETA_BAR).sum()),
        },
        "pose": {
            "d_pose_ratio_median": float(np.median(ratios)) if ratios.size else None,
            "d_pose_ratio_max": float(ratios.max()) if ratios.size else None,
            "pairs_pose_improved": int((ratios < 1.0).sum()) if ratios.size else None,
        },
        "bar_arithmetic": {
            "channel_bytes": CHANNEL_BYTES,
            "seg_gain_if_all_band_flips_fixed_S": -gain_all,
            "rate_cost_S": cost,
            "net_S_at_measured_eta": -eta_pooled * gain_all + cost,
            "S_at_measured_eta": HV1_S - eta_pooled * gain_all + cost,
            "net_S_at_bar": -ETA_BAR * gain_all + cost,
        },
        "verdict": verdict,
        "verdict_scope": (
            f"INSTANCE: hv1 ep0634 base, ring-0 described set, r={args.radius} edit support, "
            "pose-null-constrained realization, this solver budget"),
        "gt_decode_s": gt_decode_s,
        "wall_s": time.time() - t0,
        "rows": rows,
    }
    (outdir / "ETA_GATE_VERDICT.json").write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "rows"}, indent=2, sort_keys=True))
    return 0


def aggregate(args: argparse.Namespace) -> int:
    """Emit the verdict from the rows that have LANDED, without needing the run to finish.

    The per-pair rows are the payload and they are written incrementally, so a run that is
    still going (or was stopped to yield cores to a sister arm) still yields an honest verdict
    at the n it reached.  m96 governs the reading: a seeded-RANDOM subset may REFUTE the bar;
    a subset that CLEARS it does not license LIVE without the fuller population.
    """
    path = args.out / "ETA_GATE_ROWS.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise EtaGateError(f"no rows in {path}")
    etas = np.array([r["eta_net"] for r in rows], dtype=np.float64)
    desc = np.array([r["n_described_ring0"] for r in rows], dtype=np.float64)
    before = np.array([r["flips_before"] for r in rows], dtype=np.float64)
    after = np.array([r["flips_after"] for r in rows], dtype=np.float64)
    ratios = np.array([r["d_pose_ratio"] for r in rows
                       if r.get("d_pose_ratio") is not None], dtype=float)
    eta_pooled = float((before.sum() - after.sum()) / desc.sum())

    gain_all = 34666 * SEG_DS_PER_FLIP
    cost = CHANNEL_BYTES * RATE_DS_PER_BYTE
    # The pre-registered bar is seg+rate only.  The pose leg is reported separately: including
    # it can only make a CLOSED verdict more robust, and a LIVE verdict would have to re-run
    # the arithmetic with it.  d_pose enters S as sqrt(10*d_pose), so a ratio > 1 costs S.
    d_pose_base = (0.0082946 ** 2) / 10.0
    pose_leg = {}
    if ratios.size:
        for name, ratio in (("median", float(np.median(ratios))), ("mean", float(ratios.mean()))):
            pose_leg[f"delta_S_pose_at_{name}_ratio"] = (
                (10.0 * d_pose_base * ratio) ** 0.5 - (10.0 * d_pose_base) ** 0.5)
    # m96 is asymmetric and the verdict string must be too: a seeded-random subset may REFUTE a
    # bar, but a subset that CLEARS it does not license LIVE.  Emitting "LIVE" off a small n
    # would contradict the law this module documents, so the pass branch is explicitly held.
    if eta_pooled <= ETA_BAR:
        verdict = "CLOSED"
    elif len(rows) >= args.full_population_n:
        verdict = "LIVE"
    else:
        verdict = "PASS_SUBSET_NOT_LICENSING_LIVE"
    rec = {
        "schema": "ddm_rt1_eta_gate_aggregate.v1",
        "axis": "[macOS-CPU advisory] -- NEVER a score",
        "score_claim": False, "promotable": False,
        "pre_registered_bar": ETA_BAR,
        "n_pairs_landed": len(rows),
        "pairs": [r["pair"] for r in rows],
        "eta": {"pooled": eta_pooled, "per_pair_mean": float(etas.mean()),
                "per_pair_sd": float(etas.std(ddof=1)) if etas.size > 1 else None,
                "min": float(etas.min()), "max": float(etas.max()),
                "pairs_above_bar": int((etas > ETA_BAR).sum())},
        "pose": {"d_pose_ratio_median": float(np.median(ratios)) if ratios.size else None,
                 "d_pose_ratio_min": float(ratios.min()) if ratios.size else None,
                 "d_pose_ratio_max": float(ratios.max()) if ratios.size else None,
                 "pairs_pose_improved": int((ratios < 1.0).sum()) if ratios.size else None,
                 "seg_plus_rate_bar_excludes_this_leg": True, **pose_leg},
        "bar_arithmetic": {
            "channel_bytes": CHANNEL_BYTES, "rate_cost_S": cost,
            "seg_gain_if_all_band_flips_fixed_S": -gain_all,
            "net_S_at_measured_eta": -eta_pooled * gain_all + cost,
            "S_at_measured_eta": HV1_S - eta_pooled * gain_all + cost,
            "eta_needed": ETA_BAR},
        "verdict": verdict,
        "verdict_scope": "INSTANCE: hv1 ep0634 base, ring-0 described set, r=1 described-set "
                         "edit support, pose-null-constrained realization, this solver budget; "
                         "m96: a seeded-random subset may REFUTE, a subset PASS may not license "
                         "LIVE",
        "rows": rows,
    }
    (args.out / "ETA_GATE_VERDICT_AGGREGATE.json").write_text(
        json.dumps(rec, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in rec.items() if k != "rows"}, indent=2, sort_keys=True))
    return 0


def _boundary(lab: np.ndarray) -> np.ndarray:
    b = np.zeros(lab.shape, dtype=bool)
    d = lab[:-1, :] != lab[1:, :]
    b[:-1, :] |= d
    b[1:, :] |= d
    d = lab[:, :-1] != lab[:, 1:]
    b[:, :-1] |= d
    b[:, 1:] |= d
    return b


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", type=Path, default=DEFAULT_WORK)
    ap.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    ap.add_argument("--tokens", type=Path, default=DEFAULT_TOKENS)
    ap.add_argument("--gt-argmax", type=Path, default=DEFAULT_GT_ARGMAX)
    ap.add_argument("--gt-mkv", type=Path, default=DEFAULT_GT_MKV)
    ap.add_argument("--out", type=Path, default=DEFAULT_WORK / "eta_gate")
    ap.add_argument("--n-pairs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=20260816)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=6.0)
    ap.add_argument("--eval-every", type=int, default=4)
    ap.add_argument("--radius", type=int, default=1)
    ap.add_argument("--focus-weight", type=float, default=500.0,
                    help="CE weight on the described set (operating-point adaptation)")
    ap.add_argument("--mode", choices=["null", "free", "aggregate"], default="null",
                    help="null = pose-constrained (the gate); free = unconstrained "
                         "positive control")
    ap.add_argument("--full-population-n", type=int, default=120,
                    help="pairs required before a PASS may be called LIVE (m96)")
    ap.add_argument("--retain-frames", action="store_true")
    args = ap.parse_args(argv)
    return aggregate(args) if args.mode == "aggregate" else run(args)


if __name__ == "__main__":
    raise SystemExit(main())
