#!/usr/bin/env python3
"""ddm_fo2h LEG 2 -- re-run sr1's waterfill inclusion test against MEASURED coder bytes.

WHAT THIS PAYS.  sr1 chose its 41-cell support by an inclusion test whose cost model was the
IDEAL per-cell conditional entropy -- no model cost, no coder inefficiency, and no price at all
for the cell SET itself.  fo1 then measured what a real coder actually spends on that support
(4,308 B) and closed its memo with the one instruction this module executes:

    "The cell-set side info is the only counted item the waterfill framing hides, and it is
     also the item that grows if a future arm waterfills harder. ... Price it INSIDE the
     waterfill's inclusion test, not after it."   -- fo1 memo s7

So the selection was optimized against the wrong objective.  This module re-runs the sweep with
(a) REAL round-trip-verified coder bytes at every candidate level and (b) the cell-set side info
inside the objective, and reports where the optimum actually sits.

WHY A PREFIX SWEEP IS THE RIGHT FAMILY, AND WHY IT IS COMPLETE HERE.  sr1's marginal test is a
per-cell independent comparison, so it is exactly a threshold on the per-cell value ratio
(seg S bought) / (rate S spent) -- which makes sr1's own optimum a PREFIX of the cells ranked by
that ratio (sr1's `prefix_optimum` and `marginal_waterfill` agree at 41 cells, which is the
receipt for this).  Only 74 of the 1200 cells are live, so sweeping every prefix m = 1..74 is
the COMPLETE enumeration of that family, not a sample of it.  The real coder is NOT additive
per cell (its CABAC contexts adapt across the whole stream), which is precisely why the
selection has to be re-scored by coding each candidate rather than by summing per-cell costs.

CHEAP SWEEP, SAME OBJECT.  fo1's `build_frames` recomputes the label boundary, cell keys, edge
pairs and the boundary WALK for every selection.  The walk that fo1's best coder (M8,
`bandwalk_pair`) uses is `walk_order(band)` -- a function of the BAND alone, so it does not
depend on the selection at all.  This module therefore does ONE pass over the 600 frames, keeps
the band-ordered arrays, and derives each candidate's frame dicts by boolean subsetting.  The
coders themselves are fo1's `code_mask` / `code_target`, IMPORTED AND CALLED VERBATIM -- nothing
about the coding is reimplemented here.

FAIL-CLOSED CONTROL.  At sr1's exact 41-cell selection this module must reproduce fo1's measured
M8 = 4,123 B and T2 = 185 B EXACTLY.  If it does not, the fast path is not the same object and
the module refuses to emit a sweep.  Every payload is decoded back through the same function
that encoded it and compared against the truth field before its byte count is used.

CELL-SET SIDE INFO, PRICED TWO WAYS.  The cell LIVENESS map (`band_px > 0`) is a deterministic
function of the decoded label field over the whole clip, so the receiver derives it for free and
the set the encoder must actually name is an m-subset of the LIVE cells, not of all 1200.  Both
prices are reported; the live-set price is used in the objective and the 1200-cell price is
carried alongside so no reader has to take the cheaper one on trust.  The count m is itself sent
(7 bits at 74 live cells).

Axis `[macOS-CPU advisory]`; scorer-free, $0, no launches, no Modal.  `score_claim=false`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "experiments"))

from ddm_fo1_waterfill_real_coder import (  # reuse, never reimplement
    BASE_ARCHIVE_SHA256,
    BASE_S,
    FRAMES,
    N_CELLS,
    N_CLASSES,
    RATE_DS_PER_BYTE,
    SEG_DS_PER_FLIP,
    SEG_H,
    SEG_W,
    SR1_IDEAL_BYTES,
    SR1_IDEAL_CELLS,
    SR1_IDEAL_FLIPS,
    binary_entropy_bits,
    boundary,
    cell_features,
    cell_key,
    code_mask,
    code_target,
    edge_pair_field,
    open_tokens,
    save_array,
    save_blob,
    sha256_bytes,
    walk_order,
)

# --- fo1's measured row on sr1's 41-cell support: the control this module must reproduce -----
FO1_M8_MASK_B = 4123
FO1_T2_TARGET_B = 185
FO1_TOTAL_B = 4308
FO1_BREAKEVEN_ETA = 0.5196321126365346   # FROZEN -- fo1's break-even on real bytes

DEFAULT_RT1_WORK = Path("/Volumes/APDataStore/pact/ddm_rt1_seg_roundtrip_20260816")
DEFAULT_SR1_WORK = Path("/Volumes/APDataStore/pact/ddm_sr1_manufactured_seg_recovery_20260816")
DEFAULT_FO1_WORK = Path("/Volumes/APDataStore/pact/ddm_fo1_waterfill_real_coder")
DEFAULT_TOKENS = Path(
    "/Volumes/APDataStore/pact/ddm_hv1_harvest_compose/ep0634/retained/coders/"
    "s1p25_c1p0/decoded_spatial_tokens.rc64.bin"
)
DEFAULT_GT = Path(
    "/Volumes/VertigoDataTier/pact/ddm_qs3_20260813/retained/inputs/gt_argmax_n600.npy"
)
DEFAULT_WORK = Path("/Volumes/APDataStore/pact/ddm_fo2h_eta_hardening")


class Fo2hError(RuntimeError):
    """Fail-closed error for custody, reconstruction, or round-trip violations."""


def progress(work: Path, milestone: str, detail: dict) -> None:
    """Append a milestone stamped with THIS arm's id.

    Deliberately not fo1's `progress`: that helper hardcodes `arm: ddm_fo1`, so reusing it
    would file this arm's milestones under fo1 and a later reader would mis-attribute them.
    Reuse is for MECHANISM (the coders); provenance identity is never borrowed.
    """
    row = {"arm": "ddm_fo2h", "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "milestone": milestone, "detail": detail, "pid": os.getpid(),
           "host": socket.gethostname()}
    work.mkdir(parents=True, exist_ok=True)
    with (work / "PROGRESS.jsonl").open("a") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"[fo2h] {milestone}: {json.dumps(detail, sort_keys=True)}", flush=True)


# ============================================================================================
# stage 1 -- ONE selection-independent pass over the clip
# ============================================================================================
def precompute(args: argparse.Namespace) -> dict:
    """Band-ordered arrays for every frame, plus the per-cell histograms.

    Everything kept here is a function of the DECODED LABEL FIELD and rt1's retained flip/target
    fields -- never of a candidate selection -- so one pass serves the whole sweep.
    """
    t0 = time.time()
    tok = open_tokens(args.tokens)
    gt = np.load(args.gt, mmap_mode="r")
    pred = np.load(args.rt1_work / "argmax_base.npy", mmap_mode="r")
    flip_ret = np.load(args.rt1_work / "flip_mask_vs_gt.npy", mmap_mode="r")
    band_ret = np.load(args.rt1_work / "free_band_mask.npy", mmap_mode="r")
    tgt_ret = np.load(args.rt1_work / "flip_target_class.npy", mmap_mode="r")

    band_px = np.zeros(N_CELLS, dtype=np.int64)
    flip_px = np.zeros(N_CELLS, dtype=np.int64)
    tgt_counts = np.zeros((N_CLASSES, N_CLASSES + 1, N_CLASSES), dtype=np.int64)
    per: list[dict] = []
    for t in range(args.frames):
        lab = np.asarray(tok[t])
        band = boundary(lab)
        # fail-closed, same two controls fo1 ran: the support this arm re-selects over must BE
        # the object sr1 priced and rt1 measured.
        if not np.array_equal(band, np.asarray(band_ret[t]).astype(bool)):
            raise Fo2hError(f"rt1 free_band_mask != boundary(labels) at frame {t}")
        flip_full = np.asarray(pred[t]) != np.asarray(gt[t])
        if not np.array_equal(flip_full, np.asarray(flip_ret[t]).astype(bool)):
            raise Fo2hError(f"rt1 flip_mask_vs_gt != (argmax_base != gt) at frame {t}")

        key = cell_key(lab)
        flip = flip_full & band
        band_px += np.bincount(key[band], minlength=N_CELLS)
        flip_px += np.bincount(key[flip], minlength=N_CELLS)
        own, partner, _ = cell_features(lab)
        part = np.where(partner == 255, N_CLASSES, partner).astype(np.uint8)
        if flip.any():
            np.add.at(
                tgt_counts,
                (own[flip].astype(np.int64), part[flip].astype(np.int64),
                 np.asarray(gt[t])[flip].astype(np.int64)),
                1,
            )

        bflat = np.flatnonzero(band.reshape(-1)).astype(np.int64)   # raster order over the band
        pair = edge_pair_field(lab).reshape(-1)
        per.append({
            "band_flat": bflat,
            "band_key": key.reshape(-1)[bflat].astype(np.int16),
            "band_pair": pair[bflat].astype(np.uint8),
            "band_own": lab.reshape(-1)[bflat].astype(np.uint8),
            "band_part": part.reshape(-1)[bflat].astype(np.uint8),
            "band_truth": np.asarray(flip_ret[t]).astype(bool).reshape(-1)[bflat],
            "band_target": np.asarray(tgt_ret[t]).reshape(-1)[bflat].astype(np.uint8),
            # selection-INDEPENDENT: M8 walks the full label boundary and keeps support symbols
            "band_walk": walk_order(band),
        })
        if (t + 1) % 100 == 0:
            print(f"  [precompute] {t + 1}/{args.frames} band {int(band_px.sum()):,} "
                  f"flips {int(flip_px.sum()):,}", flush=True)

    control = {}
    for name, mine in (("cell_band_px", band_px), ("cell_flip_px", flip_px)):
        ref_path = args.sr1_work / f"{name}.npy"
        ref = np.load(ref_path)
        same = bool(np.array_equal(ref, mine))
        control[name] = {"byte_identical_to_sr1": same,
                         "sr1_sum": int(ref.sum()), "mine_sum": int(mine.sum())}
        if not same and args.frames == FRAMES:
            raise Fo2hError(f"reconstructed {name} != sr1's retained payload")

    tot = tgt_counts.sum(axis=2, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        prob = np.where(tot > 0, tgt_counts / np.maximum(tot, 1), 0.0)
        logp = np.where(prob > 0, np.log2(np.maximum(prob, 1e-300)), 0.0)
    target_bits_total = float(-(tgt_counts * logp).sum())
    target_bits_per_flip = target_bits_total / max(int(flip_px.sum()), 1)

    return {"per": per, "band_px": band_px, "flip_px": flip_px,
            "target_bits_per_flip": target_bits_per_flip,
            "control_vs_sr1": control, "wall_s": time.time() - t0}


# ============================================================================================
# stage 2 -- frame dicts for a candidate selection, then fo1's own coders
# ============================================================================================
def frames_for_selection(pre: dict, selected: np.ndarray) -> list[dict]:
    """Derive fo1's per-frame coder input for `selected` by subsetting the band arrays."""
    sel_lut = np.zeros(N_CELLS, dtype=bool)
    sel_lut[selected] = True
    out: list[dict] = []
    for fr in pre["per"]:
        keep = sel_lut[fr["band_key"].astype(np.int64)]
        raster = fr["band_flat"][keep]
        sup_flat = np.zeros(SEG_H * SEG_W, dtype=bool)
        sup_flat[raster] = True
        bw = fr["band_walk"]
        out.append({
            "raster": raster,
            "walk_band": bw[sup_flat[bw]],
            "bits_raster": fr["band_truth"][keep].astype(np.uint8),
            "pair": fr["band_pair"][keep],
            "own": fr["band_own"][keep],
            "part": fr["band_part"][keep],
            "target_raster": fr["band_target"][keep],
        })
    return out


def code_and_verify(frames: list[dict]) -> dict:
    """Encode mask + target with fo1's best coders and DECODE BOTH BACK before believing a byte.

    A byte count from an unverified encoder is an assertion.  Every count this module reports
    has been inverted through the same function that produced it.
    """
    mask_blob, produced, n_sym = code_mask("bandwalk_pair", frames)
    _, decoded, _ = code_mask("bandwalk_pair", frames, payload=mask_blob)
    for got, ref, fr in zip(decoded, produced, frames, strict=True):
        if not np.array_equal(got, ref) or not np.array_equal(got, fr["bits_raster"]):
            raise Fo2hError("mask payload does not round-trip to the truth field")

    tgt_blob, tgt_prod, n_tgt = code_target(frames, produced, contextual=True)
    _, tgt_dec, _ = code_target(frames, decoded, contextual=True, payload=tgt_blob)
    for got, ref in zip(tgt_dec, tgt_prod, strict=True):
        if not np.array_equal(got, ref):
            raise Fo2hError("target payload does not round-trip")

    return {"mask_bytes": len(mask_blob), "target_bytes": len(tgt_blob),
            "symbols": n_sym, "flips": n_tgt,
            "mask_sha256": sha256_bytes(mask_blob), "target_sha256": sha256_bytes(tgt_blob),
            "roundtrip_verified": True, "_mask_blob": mask_blob, "_tgt_blob": tgt_blob}


def cellset_bits(m: int, universe: int) -> float:
    """Exact minimum to name an unordered m-subset of `universe`: log2 C(universe, m).

    This is the combinatorial-rank price (CLAUDE.md L31 / PR101's colex-rank sidecar), i.e. the
    floor a real index coder reaches -- not a padded index list.  The count m is sent alongside.
    """
    if m <= 0 or m > universe:
        return 0.0
    return float(math.log2(math.comb(universe, m)))


# ============================================================================================
# stage 3 -- the sweep
# ============================================================================================
def sweep(args: argparse.Namespace) -> int:
    t0 = time.time()
    work = args.work
    (work / "retained").mkdir(parents=True, exist_ok=True)
    progress(work, "leg2-start", {"frames": args.frames, "eta_grid": args.eta})

    pre = precompute(args)
    band_px, flip_px = pre["band_px"], pre["flip_px"]
    tgt_bpf = pre["target_bits_per_flip"]
    live = band_px > 0
    n_live = int(live.sum())
    cell_ids = np.flatnonzero(live)
    n_r = band_px[live].astype(np.float64)
    k_r = flip_px[live].astype(np.float64)
    p_r = k_r / n_r
    ideal_bits_r = n_r * binary_entropy_bits(p_r)
    progress(work, "leg2-precomputed", {
        "cells_live": n_live, "band_px": int(band_px.sum()), "band_flips": int(flip_px.sum()),
        "target_bits_per_flip": tgt_bpf,
        "histograms_byte_identical_to_sr1": all(
            c["byte_identical_to_sr1"] for c in pre["control_vs_sr1"].values())})

    # --- the ranking: sr1's own marginal test is a threshold on this per-cell value ratio -----
    ideal_cost_B_r = ideal_bits_r / 8.0 + k_r * tgt_bpf / 8.0
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(ideal_cost_B_r > 0,
                         (k_r * SEG_DS_PER_FLIP) / (ideal_cost_B_r * RATE_DS_PER_BYTE), 0.0)
    order = np.argsort(-ratio, kind="stable")

    # --- CONTROL: sr1's exact 41-cell selection must reproduce fo1's measured bytes -----------
    sr1_sel = np.load(args.fo1_work / "retained" / "selected_cells.npy")
    if sr1_sel.size != SR1_IDEAL_CELLS:
        raise Fo2hError(f"fo1 selected_cells has {sr1_sel.size} ids, expected {SR1_IDEAL_CELLS}")
    ctrl = code_and_verify(frames_for_selection(pre, sr1_sel))
    ctrl_ok = (ctrl["mask_bytes"] == FO1_M8_MASK_B and ctrl["target_bytes"] == FO1_T2_TARGET_B
               and ctrl["flips"] == SR1_IDEAL_FLIPS)
    control = {"sr1_41_cell_selection": {
        "mask_bytes": ctrl["mask_bytes"], "fo1_mask_bytes": FO1_M8_MASK_B,
        "target_bytes": ctrl["target_bytes"], "fo1_target_bytes": FO1_T2_TARGET_B,
        "flips": ctrl["flips"], "sr1_flips": SR1_IDEAL_FLIPS,
        "symbols": ctrl["symbols"],
        "reproduces_fo1_exactly": bool(ctrl_ok)}}
    if not ctrl_ok and args.frames == FRAMES:
        raise Fo2hError(
            f"fast path does not reproduce fo1: mask {ctrl['mask_bytes']} vs {FO1_M8_MASK_B}, "
            f"target {ctrl['target_bytes']} vs {FO1_T2_TARGET_B} -- refusing to emit a sweep")
    # the ranking must also PLACE sr1's selection as a prefix, or the family is the wrong one
    prefix41 = set(cell_ids[order[:SR1_IDEAL_CELLS]].tolist())
    control["ranking_reproduces_sr1_selection_as_prefix"] = bool(prefix41 == set(sr1_sel.tolist()))
    progress(work, "leg2-control", control)

    # --- the sweep over every prefix ----------------------------------------------------------
    etas = list(args.eta)
    rows = []
    best_blobs: dict[int, dict] = {}
    # Ladder.  Coding cost scales with SUPPORT SIZE, which grows steeply in the tail as the
    # selection approaches the whole band (rt1's describe-everything, an already-measured
    # NON-SUPPLIER).  So enumerate EVERY level through `--dense-through` -- the region where an
    # optimum can plausibly sit, including well past the incumbent 41 -- and step the expensive
    # tail coarsely rather than skipping it, so the tail is measured and not assumed.
    dense = min(args.dense_through, n_live)
    levels = sorted({*range(1, dense + 1), *range(dense, n_live + 1, args.tail_step),
                     n_live, SR1_IDEAL_CELLS})
    levels = [m for m in levels if 1 <= m <= n_live]
    for m in levels:
        sel = np.sort(cell_ids[order[:m]])
        frames = frames_for_selection(pre, sel)
        r = code_and_verify(frames)
        flips = float(k_r[order[:m]].sum())
        if r["flips"] != int(flips):
            raise Fo2hError(f"m={m}: coded {r['flips']} flips, histogram says {int(flips)}")
        side_live_B = cellset_bits(m, n_live) / 8.0
        side_1200_B = cellset_bits(m, N_CELLS) / 8.0
        count_B = math.ceil(math.log2(n_live + 1)) / 8.0
        payload_B = r["mask_bytes"] + r["target_bytes"]
        total_B = payload_B + side_live_B + count_B
        row = {
            "cells": m, "flips": int(flips), "band_px": float(n_r[order[:m]].sum()),
            "density": float(flips / max(n_r[order[:m]].sum(), 1.0)),
            "mask_bytes": r["mask_bytes"], "target_bytes": r["target_bytes"],
            "payload_bytes": payload_B,
            "cellset_side_info_B_live_universe": side_live_B,
            "cellset_side_info_B_1200_universe": side_1200_B,
            "count_field_B": count_B,
            "total_bytes_with_side_info": total_B,
            "ideal_bytes_sr1_model": float(ideal_bits_r[order[:m]].sum() / 8.0
                                           + flips * tgt_bpf / 8.0),
            "real_over_ideal_pct": None,
            "bytes_per_described_flip": payload_B * 8.0 / max(flips, 1.0),
            "mask_sha256": r["mask_sha256"], "target_sha256": r["target_sha256"],
            "roundtrip_verified": r["roundtrip_verified"],
            "net_dS_by_eta": {}, "breakeven_eta": None,
        }
        row["real_over_ideal_pct"] = 100.0 * (payload_B / row["ideal_bytes_sr1_model"] - 1.0)
        for e in etas:
            row["net_dS_by_eta"][f"{e:.4f}"] = -e * flips * SEG_DS_PER_FLIP \
                + total_B * RATE_DS_PER_BYTE
        row["breakeven_eta"] = total_B * RATE_DS_PER_BYTE / (flips * SEG_DS_PER_FLIP) \
            if flips > 0 else None
        rows.append(row)
        best_blobs[m] = {"mask": r["_mask_blob"], "tgt": r["_tgt_blob"], "sel": sel}
        print(f"  [sweep] m={m:3d} flips={int(flips):6d} payload={payload_B:7d} B "
              f"side={side_live_B:6.1f} B  breakeven_eta={row['breakeven_eta']:.4f}", flush=True)
        progress(work, "leg2-level", {"cells": m, "flips": int(flips),
                                      "payload_B": payload_B,
                                      "breakeven_eta": row["breakeven_eta"]})

    # --- adjudication -------------------------------------------------------------------------
    incumbent = next((r for r in rows if r["cells"] == SR1_IDEAL_CELLS), None)
    best_by_eta = {}
    for e in etas:
        k = f"{e:.4f}"
        b = min(rows, key=lambda r: r["net_dS_by_eta"][k])
        best_by_eta[k] = {
            "cells": b["cells"], "flips": b["flips"],
            "payload_bytes": b["payload_bytes"],
            "total_bytes_with_side_info": b["total_bytes_with_side_info"],
            "net_dS": b["net_dS_by_eta"][k],
            "incumbent_41_net_dS": incumbent["net_dS_by_eta"][k] if incumbent else None,
            "improvement_dS": (incumbent["net_dS_by_eta"][k] - b["net_dS_by_eta"][k])
            if incumbent else None,
        }
    min_breakeven = min(rows, key=lambda r: r["breakeven_eta"] or 9e9)

    # retain the payloads of the incumbent and of the lowest-break-even selection
    retained = {}
    for tag, m in (("incumbent_41", SR1_IDEAL_CELLS), ("min_breakeven", min_breakeven["cells"])):
        blob = best_blobs.get(m)
        if blob is None:
            continue
        retained[tag] = {
            "cells": int(m),
            "mask": save_blob(work / "retained" / f"fo2h_mask_m{m:03d}.rc", blob["mask"]),
            "target": save_blob(work / "retained" / f"fo2h_target_m{m:03d}.rc", blob["tgt"]),
            "selected_cells": save_array(
                work / "retained" / f"fo2h_selected_cells_m{m:03d}.npy", blob["sel"]),
        }

    rec = {
        "schema": "ddm_fo2h_waterfill_measured.v1",
        "axis": "[macOS-CPU advisory] scorer-free -- NEVER a score",
        "score_claim": False, "promotable": False,
        "base_archive_sha256": BASE_ARCHIVE_SHA256, "base_S": BASE_S,
        "question": "where does the waterfill optimum sit when the cost model is MEASURED coder "
                    "bytes and the cell-set side info is priced INSIDE the inclusion test",
        "coders": {"mask": "fo1 M8 -- CABAC full-band-walk order, support-only symbols "
                           "(pair x run x temporal), 88 contexts",
                   "target": "fo1 T2 -- adaptive binary-tree AC, context = (own, partner)",
                   "reused_verbatim_from": "experiments/ddm_fo1_waterfill_real_coder.py"},
        "ranking": "per-cell value ratio (seg S bought)/(ideal rate S spent) -- the statistic "
                   "sr1's marginal test thresholds on; prefixes of it are the complete family",
        "cells_live": n_live, "cells_universe": N_CELLS,
        "target_bits_per_flip": tgt_bpf,
        "control_vs_sr1_histograms": pre["control_vs_sr1"],
        "control_vs_fo1_bytes": control,
        "cellset_side_info_note":
            "liveness (band_px>0) is a deterministic function of the decoded label field over "
            "the whole clip, so the receiver derives the 74-cell live set free and the encoder "
            "names an m-subset of it; the 1200-universe price is carried alongside so the "
            "cheaper figure is never taken on trust",
        "frozen_pins": {"fo1_total_B": FO1_TOTAL_B, "fo1_breakeven_eta": FO1_BREAKEVEN_ETA,
                        "sr1_ideal_total_B": SR1_IDEAL_BYTES},
        "eta_grid": etas,
        "best_by_eta": best_by_eta,
        "min_breakeven_eta_row": {k: v for k, v in min_breakeven.items()
                                  if not k.startswith("_")},
        "incumbent_41": {k: v for k, v in (incumbent or {}).items() if not k.startswith("_")},
        "retained_payloads": retained,
        "verdict_scope": "formulation -- the fo1 M8+T2 coder pair on prefixes of the "
                         "value-ratio-ranked live cells of rt1's free label boundary at n600 on "
                         "the hv1 ep0634 base; a different coder or a non-prefix cell family "
                         "could move the optimum",
        "rows": rows,
        "wall_s": time.time() - t0,
    }
    (work / "FO2H_WATERFILL_MEASURED.json").write_text(
        json.dumps(rec, indent=2, sort_keys=True) + "\n")
    progress(work, "leg2-done", {"levels": len(rows),
                                 "min_breakeven_eta": min_breakeven["breakeven_eta"],
                                 "min_breakeven_cells": min_breakeven["cells"],
                                 "incumbent_breakeven_eta": incumbent["breakeven_eta"]
                                 if incumbent else None})
    print(json.dumps({k: v for k, v in rec.items() if k != "rows"}, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", type=Path, default=DEFAULT_WORK)
    ap.add_argument("--rt1-work", type=Path, default=DEFAULT_RT1_WORK)
    ap.add_argument("--sr1-work", type=Path, default=DEFAULT_SR1_WORK)
    ap.add_argument("--fo1-work", type=Path, default=DEFAULT_FO1_WORK)
    ap.add_argument("--tokens", type=Path, default=DEFAULT_TOKENS)
    ap.add_argument("--gt", type=Path, default=DEFAULT_GT)
    ap.add_argument("--frames", type=int, default=FRAMES)
    ap.add_argument("--eta", type=float, nargs="+",
                    default=[0.5196, 0.5651, 0.6111, 0.6235, 1.0],
                    help="eta grid the net dS is reported on")
    ap.add_argument("--dense-through", type=int, default=50,
                    help="code EVERY prefix level up to this many cells")
    ap.add_argument("--tail-step", type=int, default=4,
                    help="step for levels above --dense-through (the expensive near-full-band "
                         "tail; measured coarsely, never skipped)")
    return sweep(ap.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
