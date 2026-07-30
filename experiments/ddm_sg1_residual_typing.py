"""ddm_sg1 — QA74 SegNet residual TYPING (the pose-collapse playbook applied to seg).

Decompose the LIVE renderer d_seg residual into a typed table so the QA24 re-burn can
AIM (protect high-value cells/classes) and QA75 solve-distillation can TARGET (the
amortization-gap regions only).

REFERENCE (operator correction 2026-07-31): the decisive column uses the EXACT C1 solve
(17,927 err = d_seg 1.52e-4; the r6cal SHA-bound settled control) as the "solve" side,
NOT the box-tolerance solve (136,839 err = 0.00116). The exact solve is a compress-time
TEACHER (never ships; strict-scorer rule bars only decode-time scorer loads). The exact
solve's per-CLASS concede floor is read from the ms2r_r3 scorer_measurement.json receipt
(q1_stratum_errors) — no inflate needed. The renderer's per-pixel argmax is MEASURED here
through gr1's exact render+R+SegNet path.

TYPING DIMENSIONS (all on the renderer ENDPOINT residual, pfs1 D1 archive codes = 0.00389):
  * per GT-class flip counts  -> vs exact-solve per-class concede -> per-class ATTACK mass.
  * per margin-depth (GT SegNet margin, the rank-4 head flip-distance proxy) buckets.
  * spatial stationarity: per-(row,col) recurrence (image-static) + row-band foveation.
  * per token-CELL flip mass (grid 24x32, downsample 16 -> cell=(r//16,c//16)) -> the
    QA24 protect-set, cross-referenced with gr1's |g|-sum cell ranking.
  * Lane verdict (Tao): exact-solve Lane concede (2,556) / renderer Lane errors -> is Lane's
    residual SegNet-stride-limited (concede/pre-R place) or renderer-reach (QA24 cures)?

AUTHORITY: argmax REALIZED through the real render+R+SegNet (bit-identical to gr1's verdict);
per-class concede from the ms2r_r3 receipt (MEASURED_EXACT). [macOS-CPU advisory];
score_claim=false; promotion_eligible=false; pointer 0.1910828242 [contest-CPU] UNMOVED.
NON-PROMOTABLE. No scorer promotion / paid dispatch / pointer mutation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
from ddm_gr1_granularity_rerace import (
    ARCHIVE,
    GT,
    LEVELS,
    decode_token_codes,
    load_model,
)

POINTER = "0.1910828242 [contest-CPU] UNMOVED"
TOTAL_PX = 600 * 384 * 512  # 117,964,800
CLASS_NAMES = {0: "Road", 1: "Lane", 2: "Undrivable", 3: "Movable", 4: "MyCar"}
GRID_DS = 16  # grid_downsample: SegNet pixel (r,c) -> token cell (r//16, c//16)
GRID_H, GRID_W = 384 // GRID_DS, 512 // GRID_DS  # 24 x 32
OUTDIR = Path("/Volumes/VertigoDataTier/pact/ddm_sg1_20260731")
# exact-solve (q1/C1) per-class concede floor, read from the ms2r_r3 receipt (MEASURED_EXACT)
SOLVE_RECEIPT = ("/Volumes/VertigoDataTier/pact/ddm_ms2r_r3_box_tolerance_solve_20260725T030551Z/"
                 "stage_checkpoints/02_scorers/scorer_measurement.json")


def exact_solve_concede() -> dict:
    with open(SOLVE_RECEIPT) as fh:
        d = json.load(fh)
    q1 = d["q1_exact_control"]
    per_class = dict.fromkeys(("Road", "Lane", "Undrivable", "Movable", "MyCar"), 0)
    per_class_by_pair = np.zeros((600, 5), np.int64)  # index -> class id via CLASS_NAMES
    name2id = {v: k for k, v in CLASS_NAMES.items()}
    for r in d["rows"]:
        pid = int(r["pair_id"])
        for cname, v in r.get("q1_stratum_errors", {}).items():
            per_class[cname] += int(v)
            per_class_by_pair[pid, name2id[cname]] += int(v)
    return {"total_errors": int(q1["errors"]), "d_seg": float(q1["d_seg"]),
            "per_class": per_class, "per_class_by_pair": per_class_by_pair}


# ------------------------------------------------------------------ render (resumable)
def render_argmax(n_pairs: int, chunk: int, outdir: Path) -> Path:
    """Render each pair through gr1's exact path -> per-pixel SegNet argmax; resumable per chunk.

    Saves realized argmax (n,384,512) uint8 to <outdir>/argmax/chunk_XXXX.npy and a per-pair
    mean d_seg for validation vs gr1's 0.00389. Returns the argmax dir."""
    import mlx.core as mx

    from experiments.train_witness_realized_through_R_mlx import (
        _torch_R_to_camera_uint8,
        cpu_verdict_d_seg_argmax_batch,
    )
    from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap
    from tac.boundary_math.seg_core import load_real_segnet

    amdir = outdir / "argmax"
    amdir.mkdir(parents=True, exist_ok=True)
    lstars = open_stored_npy_memmap(Path(GT), "lstars")

    model, cfg, meta = load_model()
    # inject pfs1 D1 archive codes (= the shipped renderer endpoint, 0.00389).
    frame = zipfile.ZipFile(ARCHIVE).read("state/tokens.dr7t")
    codes = np.asarray(decode_token_codes(frame), dtype=np.uint8)
    base_arr = np.asarray(model.tokens_base, dtype=np.float32)
    tvals = codes.astype(np.float32) / (LEVELS - 1) * 2.0 - 1.0
    inj = tvals - base_arr[None]
    model.tokens_delta = mx.array(inj.astype(np.float32))
    mx.eval(model.parameters())

    seg_cpu = load_real_segnet("cpu")
    dseg_pp = np.full(n_pairs, np.nan, np.float64)
    dpath = outdir / "dseg_per_pair.npy"
    if dpath.exists():
        prev = np.load(dpath)
        dseg_pp[: len(prev)] = prev
    t0 = time.monotonic()
    for c0 in range(0, n_pairs, chunk):
        cpath = amdir / f"chunk_{c0:04d}.npy"
        if cpath.exists():  # resume: skip completed chunks
            continue
        idxs = list(range(c0, min(c0 + chunk, n_pairs)))
        cams, gts = [], []
        with mx.stream(mx.cpu):
            for i in idxs:
                rgb = model.render_frame(i)
                mx.eval(rgb)
                cams.append(_torch_R_to_camera_uint8(np.asarray(rgb, dtype=np.float32)[0]))
                gts.append(np.asarray(lstars[i], dtype=np.int64))
        dsegs, realized = cpu_verdict_d_seg_argmax_batch(seg_cpu, cams, gts)
        np.save(cpath, realized.astype(np.uint8))
        for k, i in enumerate(idxs):
            dseg_pp[i] = dsegs[k]
        np.save(dpath, dseg_pp)
        done = int(np.count_nonzero(~np.isnan(dseg_pp)))
        print(json.dumps({"chunk": c0, "done_pairs": done,
                          "mean_dseg_sofar": float(np.nanmean(dseg_pp)),
                          "wall_s": round(time.monotonic() - t0, 1)}), flush=True)
    print(json.dumps({"render_complete": True, "n_pairs": n_pairs,
                      "mean_dseg": float(np.nanmean(dseg_pp)),
                      "wall_s": round(time.monotonic() - t0, 1)}), flush=True)
    return amdir


def load_realized(amdir: Path, n_pairs: int, chunk: int) -> np.ndarray:
    parts = []
    for c0 in range(0, n_pairs, chunk):
        parts.append(np.load(amdir / f"chunk_{c0:04d}.npy"))
    return np.concatenate(parts, axis=0)  # (n,384,512) uint8


# ------------------------------------------------------------------ typing ($0)
def type_residual(realized: np.ndarray, outdir: Path) -> dict:
    from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap

    lstars = np.asarray(open_stored_npy_memmap(Path(GT), "lstars"), dtype=np.int64)[: len(realized)]
    margins = np.asarray(open_stored_npy_memmap(Path(GT), "margins"), dtype=np.float32)[: len(realized)]
    n = len(realized)
    flip = realized.astype(np.int64) != lstars  # (n,384,512) bool
    total_flip = int(flip.sum())
    d_seg = total_flip / (n * 384 * 512)

    solve = exact_solve_concede()

    # --- A. per GT-class flip counts (indexed by GT class) vs exact-solve concede ---
    per_class = {}
    for cid, cname in CLASS_NAMES.items():
        clsmask = lstars == cid
        rflip = int((flip & clsmask).sum())
        concede = solve["per_class"][cname]
        gt_px = int(clsmask.sum())
        per_class[cname] = {
            "renderer_errors": rflip,
            "exact_solve_concede": concede,
            "amortization_gap": rflip - concede,  # attack mass (containment-caveated)
            "gap_frac_of_class": round((rflip - concede) / max(1, rflip), 4),
            "gt_pixels": gt_px,
            "renderer_err_rate": round(rflip / max(1, gt_px), 6),
        }

    # --- B. per margin-depth (GT SegNet margin at flip pixels) ---
    flat_flip = flip.reshape(-1)
    flat_marg = margins.reshape(-1)
    fm = flat_marg[flat_flip]
    marg_q = np.quantile(margins, np.linspace(0, 1, 11)).tolist()
    # bucket flips by GT-margin decile (of the full margin distribution)
    edges = np.quantile(margins.reshape(-1), np.linspace(0, 1, 11))
    fbuck = np.clip(np.digitize(fm, edges[1:-1]), 0, 9)
    marg_hist = [int(np.count_nonzero(fbuck == b)) for b in range(10)]
    margin_depth = {
        "flip_margin_p10_p50_p90": [float(np.quantile(fm, q)) for q in (0.1, 0.5, 0.9)],
        "all_margin_deciles": [round(float(x), 4) for x in marg_q],
        "flip_count_by_gt_margin_decile": marg_hist,
        "frac_flips_in_bottom_margin_decile": round(marg_hist[0] / max(1, total_flip), 4),
    }

    # --- C. spatial stationarity: per-(row,col) recurrence + row-band foveation ---
    flip_freq = flip.mean(axis=0)  # (384,512) fraction of pairs flipping each pixel
    # image-static = pixels flipping in >=50% of pairs; carry what share of total flip mass?
    static_mask = flip_freq >= 0.5
    static_flip_mass = int(flip[:, static_mask].sum())
    row_flip = flip.sum(axis=(0, 2))  # (384,) flip mass per SegNet row
    band_160_240 = int(row_flip[160:240].sum())
    band_hood = int(row_flip[290:].sum())  # hood/MyCar bottom band
    stationarity = {
        "image_static_pixel_count(freq>=0.5)": int(static_mask.sum()),
        "image_static_flip_mass": static_flip_mass,
        "image_static_flip_mass_frac": round(static_flip_mass / max(1, total_flip), 4),
        "flip_mass_rows_160_240": band_160_240,
        "flip_mass_rows_160_240_frac": round(band_160_240 / max(1, total_flip), 4),
        "flip_mass_rows_290plus_hood": band_hood,
        "flip_mass_rows_290plus_frac": round(band_hood / max(1, total_flip), 4),
        "flip_freq_p50_p90_p99_p100": [float(np.quantile(flip_freq[flip_freq > 0], q))
                                        for q in (0.5, 0.9, 0.99, 1.0)],
    }

    # --- D. per token-CELL flip mass (grid 24x32, downsample 16) -> QA24 protect-set ---
    # cell flip mass = flips summed over the 16x16 block over all pairs.
    cell_mass = np.zeros((GRID_H, GRID_W), np.int64)
    fs = flip.sum(axis=0)  # (384,512) per-pixel flip count over pairs
    for gr in range(GRID_H):
        for gc in range(GRID_W):
            cell_mass[gr, gc] = int(fs[gr * GRID_DS:(gr + 1) * GRID_DS,
                                       gc * GRID_DS:(gc + 1) * GRID_DS].sum())
    # cross-reference with gr1 |g|-sum cell ranking (which cells cell_drop50 keeps).
    gr1_cellrank = None
    gcache = Path("/Volumes/VertigoDataTier/pact/ddm_gr1_20260730/gr1_sensitivity_gabs.npy")
    if gcache.exists():
        g_abs = np.load(gcache)  # (600,24,32,4)
        cell_sens = g_abs.sum(axis=(0, 3))  # (24,32) |g|-sum per cell
        gr1_cellrank = cell_sens
    cell_out = {
        "grid_hw": [GRID_H, GRID_W],
        "n_cells": GRID_H * GRID_W,
        "cell_flip_mass_total": int(cell_mass.sum()),
        "cell_flip_mass_p50_p90_p99": [float(np.quantile(cell_mass, q)) for q in (0.5, 0.9, 0.99)],
    }
    # the top-K flip-mass cells (the amortization-gap concentration) and their |g| rank
    order = np.argsort(cell_mass.reshape(-1))[::-1]
    topk = []
    for ci in order[:40]:
        r, c = divmod(int(ci), GRID_W)
        row = {"cell": [r, c], "flip_mass": int(cell_mass[r, c])}
        if gr1_cellrank is not None:
            row["gr1_g_sum"] = float(gr1_cellrank[r, c])
        topk.append(row)
    cell_out["top40_flip_mass_cells"] = topk
    # save the full cell-mass map + gr1 rank for the grid-derivation step
    np.save(outdir / "cell_flip_mass.npy", cell_mass)
    if gr1_cellrank is not None:
        np.save(outdir / "gr1_cell_gsum.npy", gr1_cellrank)
        # agreement: of the 384 cells cell_drop50 KEEPS (top-|g| half), what flip-mass fraction?
        gflat = gr1_cellrank.reshape(-1)
        keep = np.argsort(gflat)[::-1][:GRID_H * GRID_W // 2]  # top-half |g| = kept by cell_drop50
        keep_mask = np.zeros(GRID_H * GRID_W, bool)
        keep_mask[keep] = True
        cell_out["cell_drop50_kept_flip_mass_frac"] = round(
            float(cell_mass.reshape(-1)[keep_mask].sum()) / max(1, cell_mass.sum()), 4)

    # --- Lane verdict (Tao) ---
    lane = per_class["Lane"]
    lane_verdict = {
        "renderer_Lane_errors": lane["renderer_errors"],
        "exact_solve_Lane_concede": lane["exact_solve_concede"],
        "Lane_gap_over_concede_ratio": round(lane["renderer_errors"] / max(1, lane["exact_solve_concede"]), 2),
        "reading": ("renderer-reach-limited (QA24/distill can cure) — gap >> concede"
                    if lane["renderer_errors"] > 3 * lane["exact_solve_concede"]
                    else "SegNet-stride-limited (concede or pre-R place) — gap ~ concede"),
    }

    result = {
        "schema": "ddm_sg1_residual_typing.v1", "pointer": POINTER,
        "score_claim": False, "promotion_eligible": False,
        "evidence_axis": "[macOS-CPU advisory] renderer argmax realized render+R+SegNet; concede from ms2r_r3 receipt",
        "n_pairs": n,
        "renderer_endpoint": {"total_flip": total_flip, "d_seg": round(d_seg, 7)},
        "exact_solve": {"total_errors": solve["total_errors"], "d_seg": solve["d_seg"],
                        "per_class": solve["per_class"]},
        "amortization_gap_x": round(d_seg / solve["d_seg"], 2),
        "per_class": per_class,
        "margin_depth": margin_depth,
        "stationarity": stationarity,
        "cells": cell_out,
        "lane_verdict": lane_verdict,
    }
    (outdir / "sg1_typing_receipt.json").write_text(json.dumps(result, indent=1))
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["render", "type", "both"], default="both")
    ap.add_argument("--pairs", type=int, default=600)
    ap.add_argument("--chunk", type=int, default=120)
    ap.add_argument("--outdir", type=Path, default=OUTDIR)
    args = ap.parse_args()
    if args.chunk > 120:
        raise SystemExit("chunk must be <= 120 (charter law)")
    args.outdir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("render", "both"):
        render_argmax(args.pairs, args.chunk, args.outdir)
    if args.mode in ("type", "both"):
        amdir = args.outdir / "argmax"
        realized = load_realized(amdir, args.pairs, args.chunk)
        res = type_residual(realized, args.outdir)
        print(json.dumps({k: res[k] for k in
                          ("renderer_endpoint", "amortization_gap_x", "lane_verdict")}, indent=1),
              flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
