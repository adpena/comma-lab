#!/usr/bin/env python3
"""ddm_gt2 - grammar induction re-run against the REAL n600 SegNet GT argmax corpus.

Arm: ddm_gt2.  Research-only, NON-PROMOTABLE, [macOS-CPU advisory].
score_claim=false.  No scorer forward passes are run: the argmax corpus is read
from disk (kept by ddm_pu2), so every per-class / per-edge / per-pixel question
about GT and about the shipped cx1 receiver is answered without a scorer.

WHAT THIS MEASURES (premise corrected 2026-08-03, verified at source)
---------------------------------------------------------------------
The prior grammar-induction line was NOT corpus-poor: #620 g1 and #651 dv2
both induced from the REAL full-n600 GT argmax (g1's command line consumed
`experiments/results/mlx_fleet_gt_cache/gt_n600.npz`; dv2 consumed the same
cache with SHA-256 verification), and that cache's `lstars` is BIT-IDENTICAL
to this arm's corpus (0 of 117,964,800 px differ vs ddm_pu2's
gt_argmax_n600.npy, verified 2026-08-03).  What this arm adds that the prior
line did not have:
  (1) the SHIPPED-RECEIVER side (cx1_argmax_n600.npy) -- the tongue as
      SPOKEN vs as written, so flips/verbs are measurable per edge;
  (2) REAL-coder pricing (lzma/brotli/zlib actual outputs), never an entropy
      estimate alone -- the estimate-vs-coder gap is itself reported;
  (3) the VERB decomposition ddm_cg1r consumes as force-ledger columns.

TYPING DISCIPLINE (rule-118; this is a hard gate, see the memo's Symbol Table)
-----------------------------------------------------------------------------
Every symbol this script prices carries a type:
  GENERIC - a property of the FIXED operators (the frozen scorer's class
            semantics, the image lattice, a rasterizer, a dilation, a
            deterministic function of already-transmitted payload).  Such a
            symbol is free in inflate.py and is NOT counted.
  FITTED  - derived from THIS clip's content (control points, widths, a
            lexicon induced from this clip's frames).  COUNTED in archive.zip,
            and counted EXACTLY (no lossy re-rounding of a counted table).
A symbol with no type is inadmissible.  `stage_*` outputs carry `symbol_type`
on every priced row.

STAGES (each writes its own JSON; re-runnable independently)
  control - reproduce pu2's flip/d_seg row from the corpus (FAIL-CLOSED)
  depth   - per-class depth profile + the flip-vs-depth conjugation diagnosis
  mdl     - real-coder MDL of L* vs sx1's entropy estimate
  split   - static-lexicon / per-pair-sentence split, ADDRESS priced apart
            from PAYLOAD
  lane    - Lane production rules: region-paint vs curve-native, priced at
            matched fidelity

USAGE
  .venv/bin/python experiments/ddm_gt2_gt_tongue_induction.py --stage control
  .venv/bin/python experiments/ddm_gt2_gt_tongue_induction.py --stage all
"""

from __future__ import annotations

import argparse
import json
import lzma
import os
import time
import zlib
from pathlib import Path

import numpy as np

try:
    import brotli
except ImportError:  # pragma: no cover - brotli is present in this venv
    brotli = None

import cv2

# ---------------------------------------------------------------------------
# Canonical constants.  Every one carries its provenance.
# ---------------------------------------------------------------------------

# The corpus kept by ddm_pu2 (its memo section 10.4).  Not rebuildable scratch:
# it IS the measurement.
CORPUS_DIR = Path("/Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache")
GT_PATH = CORPUS_DIR / "gt_argmax_n600.npy"
CX_PATH = CORPUS_DIR / "cx1_argmax_n600.npy"

OUT_DIR = Path("/Volumes/VertigoDataTier/pact/ddm_gt2_20260803")

# ddm_pu2's fail-closed control row, measured through the shipped cx1 receiver.
# This script REPRODUCES it from the corpus; a mismatch is a hard stop.
PU2_FLIPS = 508640
PU2_D_SEG = 0.00431179
PU2_REL_TOL = 1e-5

# Canonical comma10k class order, MEASURED 2026-06-27 (CLAUDE.md; do NOT
# re-derive by luma-sorting -- that gives the WRONG order and has bitten 3x).
CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
N_CLASSES = 5

# The exchange rate, owned by ddm_wf2.  NOT a cost -- it is the rate at which a
# seg flip trades against a byte.  Cited, never re-derived here.
W_BYTES_PER_FLIP = 1.273108215332031

# Live-best anchor (the baseline every delta in this arm is stated against).
LIVE_BEST_S = 0.7910689
LIVE_BEST_BYTES = 353805
LIVE_BEST_SEG_LEG = 0.4311790
LIVE_BEST_GAP = 0.6189279

# PR130 floor legs (the TARGET), from the frontier pointer's official row.
PR130_SEG_LEG = 0.02966

# sx1's published label-field MDL row, for the estimate-vs-coder comparison.
# Source: .omx/research/ddm_sx1_label_field_mdl_n600.json
SX1_H1_BITS = 1986727.7199541146
SX1_H1_MODEL_BITS = 40000.0
SX1_LZMA9E_60F_BYTES = 42812
SX1_ZLIB9_60F_BYTES = 61372
SX1_BND_PX = 2551382


def _codec_bytes(buf: bytes, name: str) -> int:
    """Real-coder byte count.  No estimates: these are actual compressed sizes."""
    if name == "lzma9e":
        return len(lzma.compress(buf, preset=9 | lzma.PRESET_EXTREME))
    if name == "zlib9":
        return len(zlib.compress(buf, 9))
    if name == "brotli11":
        if brotli is None:
            raise RuntimeError("brotli unavailable")
        return len(brotli.compress(buf, quality=11))
    raise ValueError(f"unknown codec {name}")


def _load(mmap: bool = False):
    kw = {"mmap_mode": "r"} if mmap else {}
    gt = np.load(GT_PATH, **kw)
    cx = np.load(CX_PATH, **kw)
    return gt, cx


def _write(name: str, payload: dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    payload = {
        "arm": "ddm_gt2",
        "axis": "[macOS-CPU advisory]",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "scorer_forwards_run": 0,
        "substrate": "gt_argmax_n600.npy + cx1_argmax_n600.npy (ddm_pu2 corpus); no decode/scorer",
        "n_frames": 600,
        **payload,
    }
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    os.replace(tmp, path)
    print(f"[gt2] wrote {path}")
    return path


# ---------------------------------------------------------------------------
# STAGE control
# ---------------------------------------------------------------------------

def stage_control() -> dict:
    """Reproduce pu2's row from the corpus.  FAIL-CLOSED on mismatch."""
    gt, cx = _load()
    gt = np.asarray(gt)
    cx = np.asarray(cx)
    px = int(gt.size)
    flips = int(np.count_nonzero(gt != cx))
    d_seg = flips / px

    rel = abs(d_seg - PU2_D_SEG) / PU2_D_SEG
    ok_flips = flips == PU2_FLIPS
    ok_dseg = rel <= PU2_REL_TOL
    if not (ok_flips and ok_dseg):
        raise SystemExit(
            f"CONTROL FAILED: flips={flips} (want {PU2_FLIPS}) "
            f"d_seg={d_seg!r} rel={rel:.3e} (want <= {PU2_REL_TOL})"
        )

    # Full directed 5x5 confusion: rows = GT class, cols = cx1 class.
    conf = np.bincount(
        (gt.ravel().astype(np.int64) * N_CLASSES + cx.ravel().astype(np.int64)),
        minlength=N_CLASSES * N_CLASSES,
    ).reshape(N_CLASSES, N_CLASSES)

    gt_pop = conf.sum(axis=1)
    cx_pop = conf.sum(axis=0)

    # Undirected edge mass: an EDGE is an unordered class pair.  Per ddm_pc2,
    # per-CLASS tables hide the hub structure; decompose per EDGE.
    edges = []
    for a in range(N_CLASSES):
        for b in range(a + 1, N_CLASSES):
            fwd = int(conf[a, b])
            rev = int(conf[b, a])
            tot = fwd + rev
            if tot == 0:
                continue
            hi, lo = max(fwd, rev), min(fwd, rev)
            edges.append(
                {
                    "edge": f"{CLASS_NAMES[a]}<->{CLASS_NAMES[b]}",
                    "gt_to_cx": fwd,
                    "cx_to_gt": rev,
                    "total": tot,
                    "share_of_flips": tot / flips,
                    "asymmetry": (hi / lo) if lo else None,
                    "dominant": (
                        f"{CLASS_NAMES[a]}->{CLASS_NAMES[b]}"
                        if fwd >= rev
                        else f"{CLASS_NAMES[b]}->{CLASS_NAMES[a]}"
                    ),
                    # S-arithmetic: 100 * flips / px is this edge's seg contribution.
                    "seg_S_contribution": 100.0 * tot / px,
                    "share_of_gap_vs_live_best": (100.0 * tot / px) / LIVE_BEST_GAP,
                }
            )
    edges.sort(key=lambda e: -e["total"])

    # Per-class NET flow, with the denominator STATED (a bare percentage is
    # unanchored; ddm_pu2 quotes a different normalisation, so both are given).
    net = []
    for i, name in enumerate(CLASS_NAMES):
        lost = int(gt_pop[i] - conf[i, i])
        gained = int(cx_pop[i] - conf[i, i])
        net.append(
            {
                "class": name,
                "gt_px": int(gt_pop[i]),
                "cx_px": int(cx_pop[i]),
                "lost_px": lost,
                "gained_px": gained,
                "net_px": gained - lost,
                "net_pct_of_own_gt_population": 100.0 * (gained - lost) / int(gt_pop[i]),
                "net_pct_of_total_flips": 100.0 * (gained - lost) / flips,
                "touched_pct_of_own_gt_population": 100.0 * lost / int(gt_pop[i]),
            }
        )

    return {
        "stage": "control",
        "control_passed": True,
        "px_total": px,
        "flips": flips,
        "d_seg": d_seg,
        "pu2_flips": PU2_FLIPS,
        "pu2_d_seg": PU2_D_SEG,
        "rel_err_vs_pu2": rel,
        "confusion_gt_rows_cx_cols": conf.tolist(),
        "class_names": list(CLASS_NAMES),
        "edges_by_mass": edges,
        "net_flow": net,
        "baseline_named": {
            "live_best_S": LIVE_BEST_S,
            "live_best_bytes": LIVE_BEST_BYTES,
            "live_best_seg_leg": LIVE_BEST_SEG_LEG,
            "gap_to_pr130": LIVE_BEST_GAP,
            "target_seg_leg_pr130": PR130_SEG_LEG,
        },
    }


# ---------------------------------------------------------------------------
# STAGE depth - the conjugation diagnosis
# ---------------------------------------------------------------------------

def _depth_map(mask: np.ndarray) -> np.ndarray:
    """L1 depth of each True pixel below the mask boundary (0 outside).

    distanceTransform with DIST_L1 / mask 3 gives an exact integer chamfer
    distance to the nearest zero pixel; depth 1 == the pixel touches a
    non-class pixel in 4-connectivity.  GENERIC (a lattice operator).
    """
    return cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L1, 3)


# The VERB LEXICON.  These names are the column headers ddm_cg1r (task #809)
# consumes for its (class|edge) x force x VERB force ledger.  A force acts on a
# class THROUGH a production, so the verb is the channel by which any training
# force, loss term, guard or carrier reaches a class.  Each verb carries its
# OPERATIONAL measurement rule -- a verb that cannot be measured from the
# corpus is declared OUT_OF_SCOPE rather than asserted.
VERB_LEXICON = {
    "DISPLACE": {
        "gloss": "the separatrix moved; mass is locally conserved across the edge",
        "measured_by": "per frame, per unordered edge: 2*min(a->b, b->a). Both directions present in the same frame means the boundary wobbled rather than one side losing.",
        "sign": "symmetric - no class gains or loses net area",
        "in_scope": True,
    },
    "TRANSFER": {
        "gloss": "one side of the edge genuinely lost area to the other (erosion of the loser, growth of the winner)",
        "measured_by": "per frame, per unordered edge: abs(a->b minus b->a), signed toward the winner",
        "sign": "antisymmetric - this is the verb the net-flow number is made of",
        "in_scope": True,
    },
    "ERODE": {
        "gloss": "a surviving component lost a shallow rim (surface peel)",
        "measured_by": "flip patch whose parent GT component retains >=5% of its mass, and whose max GT-depth is <=1",
        "sign": "negative for the eroded class",
        "in_scope": True,
    },
    "GOUGE": {
        "gloss": "a surviving component lost interior, not just rim",
        "measured_by": "flip patch whose parent survives (>=5%) but whose max GT-depth is >1",
        "sign": "negative for the gouged class",
        "in_scope": True,
    },
    "ANNIHILATE": {
        "gloss": "an entire GT component is gone - a whole WORD of the tongue was dropped",
        "measured_by": "GT connected component retaining <5% of its pixels as its own class in cx1",
        "sign": "negative; the strongest form of the dropped-word-class failure",
        "in_scope": True,
    },
    "BIRTH": {
        "gloss": "a component exists in cx1 with no GT counterpart",
        "measured_by": "cx1 connected component with <5% overlap onto the same GT class",
        "sign": "positive, and usually harmful (a hallucinated word)",
        "in_scope": True,
    },
    "FRAGMENT": {
        "gloss": "component count rises without proportional area change (one word split into several)",
        "measured_by": "per frame, per class: components(cx1) minus components(GT), reported against the area delta",
        "sign": "structure-changing; can be area-neutral and therefore invisible to d_seg",
        "in_scope": True,
    },
    "AMPLITUDE": {
        "gloss": "the pre-argmax logit margin moved without crossing",
        "measured_by": "NOT MEASURABLE from an argmax corpus - it is by construction invisible after argmax",
        "sign": "unknown from this corpus",
        "in_scope": False,
        "out_of_scope_reason": "requires pre-argmax logits or a margin field; this arm ran zero scorer forwards. ddm_hg1's barrier/margin-recovery measurements are the surface that owns this verb.",
    },
    "PHASE": {
        "gloss": "sub-pixel position of the whole structure (the positional DOF ddm_pc2 names as phase-faithfulness)",
        "measured_by": "NOT SEPARABLE from DISPLACE at argmax resolution - a sub-pixel shift appears as an integer-pixel boundary move or as nothing",
        "sign": "unknown from this corpus",
        "in_scope": False,
        "out_of_scope_reason": "argmax quantises position to the lattice; phase lives below it. ddm_pc2 owns per-pair positional DOF.",
    },
}


def stage_verbs(frames: int = 600) -> dict:
    """EDGE-indexed verb decomposition of the whole seg gap.

    Per m91: decompose per EDGE, never per class -- charging by GT class splits
    ONE separatrix across two rows and hides the hub.  Per the operator's
    directive, the lexicon's VERBS are the channel through which any force
    reaches a class, so the mass is attributed to (edge x verb) and also
    reported PER FRAME so a force can be aimed at individual frames.

    The two primary verbs are measured with a per-frame local decomposition:
      DISPLACE = 2*min(fwd, rev)   (boundary wobble; mass conserved)
      TRANSFER = |fwd - rev|       (one side genuinely lost area)
    Computed PER FRAME then summed -- doing it globally would let one frame's
    forward flow cancel another frame's reverse flow and fake conservation.
    """
    gt, cx = _load(mmap=True)
    n = min(frames, gt.shape[0])
    h, w = gt.shape[1], gt.shape[2]

    pairs = [(a, b) for a in range(N_CLASSES) for b in range(a + 1, N_CLASSES)]
    disp = {p: 0 for p in pairs}
    xfer = {p: 0 for p in pairs}
    xfer_dir = {p: 0 for p in pairs}  # signed: + means a lost to b
    per_frame_edge = {p: [] for p in pairs}

    max_d = 12
    depth_hist = np.zeros((N_CLASSES, max_d + 1), dtype=np.int64)
    flip_hist = np.zeros((N_CLASSES, max_d + 1), dtype=np.int64)

    # component-level verbs, per class
    comp_stats = {
        c: {"annihilate": 0, "annihilate_px": 0, "birth": 0, "birth_px": 0,
            "survive": 0, "gt_comps": 0, "cx_comps": 0,
            "erode_px": 0, "gouge_px": 0}
        for c in range(N_CLASSES)
    }

    t0 = time.time()
    for t in range(n):
        g = np.asarray(gt[t])
        c = np.asarray(cx[t])
        diff = g != c

        for (a, b) in pairs:
            fwd = int(np.count_nonzero((g == a) & (c == b)))
            rev = int(np.count_nonzero((g == b) & (c == a)))
            d2 = 2 * min(fwd, rev)
            x = abs(fwd - rev)
            disp[(a, b)] += d2
            xfer[(a, b)] += x
            xfer_dir[(a, b)] += fwd - rev
            if fwd or rev:
                per_frame_edge[(a, b)].append(
                    {"frame": t, "fwd": fwd, "rev": rev, "displace": d2, "transfer": x}
                )

        for cls in range(N_CLASSES):
            m = g == cls
            if not m.any():
                continue
            d = _depth_map(m)
            di = np.clip(d[m].astype(np.int32), 0, max_d)
            depth_hist[cls] += np.bincount(di, minlength=max_d + 1)
            fm = m & diff
            if fm.any():
                dfi = np.clip(d[fm].astype(np.int32), 0, max_d)
                flip_hist[cls] += np.bincount(dfi, minlength=max_d + 1)

            # component-level: does each GT word survive?
            ncc, lab, stats, _ = cv2.connectedComponentsWithStats(
                m.astype(np.uint8), connectivity=4
            )
            comp_stats[cls]["gt_comps"] += ncc - 1
            surv = c == cls
            if ncc > 1:
                # per-component surviving pixel count
                keep = np.bincount(
                    lab[surv & m].ravel(), minlength=ncc
                ) if (surv & m).any() else np.zeros(ncc, dtype=np.int64)
                areas = stats[:, cv2.CC_STAT_AREA]
                for k in range(1, ncc):
                    frac = keep[k] / areas[k] if areas[k] else 0.0
                    if frac < 0.05:
                        comp_stats[cls]["annihilate"] += 1
                        comp_stats[cls]["annihilate_px"] += int(areas[k])
                    else:
                        comp_stats[cls]["survive"] += 1
                        lost = (lab == k) & fm
                        if lost.any():
                            shallow = int(np.count_nonzero(lost & (d <= 1.0)))
                            comp_stats[cls]["erode_px"] += shallow
                            comp_stats[cls]["gouge_px"] += int(lost.sum()) - shallow

            # BIRTH: cx1 components with no GT support
            mc = c == cls
            if mc.any():
                ncc2, lab2, stats2, _ = cv2.connectedComponentsWithStats(
                    mc.astype(np.uint8), connectivity=4
                )
                comp_stats[cls]["cx_comps"] += ncc2 - 1
                sup = m & mc
                keep2 = np.bincount(
                    lab2[sup].ravel(), minlength=ncc2
                ) if sup.any() else np.zeros(ncc2, dtype=np.int64)
                areas2 = stats2[:, cv2.CC_STAT_AREA]
                for k in range(1, ncc2):
                    if (keep2[k] / areas2[k] if areas2[k] else 0.0) < 0.05:
                        comp_stats[cls]["birth"] += 1
                        comp_stats[cls]["birth_px"] += int(areas2[k])

        if t % 100 == 0:
            print(f"[gt2:verbs] {t}/{n} {time.time()-t0:.0f}s", flush=True)

    px = n * h * w
    # flips = sum over edges of (fwd + rev) = sum(disp + xfer): DISPLACE counts
    # 2*min(fwd, rev) and TRANSFER |fwd - rev|, which partition fwd+rev exactly.
    total_flips = sum(disp[p] + xfer[p] for p in pairs)

    edge_rows = []
    for p in pairs:
        tot = disp[p] + xfer[p]
        if tot == 0:
            continue
        pf = per_frame_edge[p]
        pf_sorted = sorted(pf, key=lambda r: -r["transfer"])[:10]
        edge_rows.append(
            {
                "edge": f"{CLASS_NAMES[p[0]]}<->{CLASS_NAMES[p[1]]}",
                "total_flips": tot,
                "share_of_all_flips": tot / total_flips,
                "seg_S_contribution": 100.0 * tot / px,
                "share_of_gap_vs_live_best": (100.0 * tot / px) / LIVE_BEST_GAP,
                "VERB_DISPLACE_px": disp[p],
                "VERB_TRANSFER_px": xfer[p],
                "displace_share": disp[p] / tot,
                "transfer_share": xfer[p] / tot,
                "transfer_direction": (
                    f"{CLASS_NAMES[p[0]]}->{CLASS_NAMES[p[1]]}"
                    if xfer_dir[p] > 0
                    else f"{CLASS_NAMES[p[1]]}->{CLASS_NAMES[p[0]]}"
                ),
                "frames_touched": len(pf),
                "worst_10_frames_by_transfer": pf_sorted,
                "transfer_concentration_top10_frames": (
                    sum(r["transfer"] for r in pf_sorted) / xfer[p] if xfer[p] else None
                ),
            }
        )
    edge_rows.sort(key=lambda r: -r["total_flips"])

    class_rows = []
    for cls, name in enumerate(CLASS_NAMES):
        tot = int(depth_hist[cls].sum())
        ftot = int(flip_hist[cls].sum())
        shallow_px = int(depth_hist[cls][:2].sum())
        shallow_flips = int(flip_hist[cls][:2].sum())
        deep_px = tot - shallow_px
        deep_flips = ftot - shallow_flips
        cs = comp_stats[cls]
        class_rows.append(
            {
                "class": name,
                "gt_px": tot,
                "flips": ftot,
                "flip_rate_own_population": ftot / tot if tot else None,
                "frac_px_at_depth_le_1": shallow_px / tot if tot else None,
                "flip_rate_depth_le_1": shallow_flips / shallow_px if shallow_px else None,
                "flip_rate_depth_gt_1": deep_flips / deep_px if deep_px else None,
                "erasure_selectivity_shallow_over_deep": (
                    (shallow_flips / shallow_px) / (deep_flips / deep_px)
                    if shallow_px and deep_px and deep_flips
                    else None
                ),
                "depth_hist_gt_px": depth_hist[cls].tolist(),
                "depth_hist_flips": flip_hist[cls].tolist(),
                "VERB_ANNIHILATE_components": cs["annihilate"],
                "VERB_ANNIHILATE_px": cs["annihilate_px"],
                "VERB_BIRTH_components": cs["birth"],
                "VERB_BIRTH_px": cs["birth_px"],
                "VERB_ERODE_px": cs["erode_px"],
                "VERB_GOUGE_px": cs["gouge_px"],
                "gt_components": cs["gt_comps"],
                "cx_components": cs["cx_comps"],
                "VERB_FRAGMENT_component_delta": cs["cx_comps"] - cs["gt_comps"],
                "annihilation_rate_of_words": (
                    cs["annihilate"] / cs["gt_comps"] if cs["gt_comps"] else None
                ),
            }
        )

    return {
        "stage": "verbs",
        "frames": n,
        "px": px,
        "total_flips": total_flips,
        "verb_lexicon": VERB_LEXICON,
        "consumer": "ddm_cg1r (task #809) force ledger: (class|edge) x force x VERB -> helps/harms/neutral + protection. These VERB_* keys are its column headers.",
        "edge_indexed": edge_rows,
        "class_indexed_secondary": class_rows,
        "why_edge_first": "m91: charging by GT class splits ONE separatrix across two rows and hides that Road is the hub",
        "elapsed_s": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# STAGE mdl - real-coder MDL of the GT label field
# ---------------------------------------------------------------------------

def stage_mdl(frames: int = 600) -> dict:
    """Real-coder cost of L*, against sx1's H1 entropy ESTIMATE.

    sx1 published 253,341 B = (H1_bits + H1_model_bits)/8 -- a context-model
    entropy estimate, not a coder output.  Its only real-coder row was 60
    frames.  This measures the actual coder on the full corpus.
    """
    gt, _ = _load(mmap=True)
    g = np.asarray(gt[:frames])
    px = int(g.size)

    rows = []
    raw = g.tobytes()
    for codec in ("zlib9", "lzma9e", "brotli11"):
        t0 = time.time()
        nb = _codec_bytes(raw, codec)
        rows.append(
            {
                "grammar": "G_raster_whole_corpus",
                "symbol_type": "FITTED",  # the compressed field IS this clip's content
                "codec": codec,
                "bytes": nb,
                "bits_per_px": 8.0 * nb / px,
                "elapsed_s": time.time() - t0,
            }
        )
        print(f"[gt2:mdl] whole {codec} {nb} B", flush=True)

    # Per-frame independent coding: measures how much the sentence gains from
    # sharing context ACROSS pairs (i.e. whether the lexicon should be static).
    per_frame_total = 0
    for t in range(frames):
        per_frame_total += _codec_bytes(np.asarray(gt[t]).tobytes(), "lzma9e")
    rows.append(
        {
            "grammar": "G_raster_per_frame_independent",
            "symbol_type": "FITTED",
            "codec": "lzma9e",
            "bytes": per_frame_total,
            "bits_per_px": 8.0 * per_frame_total / px,
        }
    )
    print(f"[gt2:mdl] per-frame lzma total {per_frame_total} B", flush=True)

    sx1_estimate_bytes = (SX1_H1_BITS + SX1_H1_MODEL_BITS) / 8.0
    whole_lzma = next(r["bytes"] for r in rows if r["grammar"] == "G_raster_whole_corpus" and r["codec"] == "lzma9e")
    sx1_extrapolated = SX1_LZMA9E_60F_BYTES * (frames / 60.0)

    return {
        "stage": "mdl",
        "frames": frames,
        "px": px,
        "rows": rows,
        "sx1_comparison": {
            "sx1_h1_estimate_bytes": sx1_estimate_bytes,
            "sx1_estimate_is_a_real_coder": False,
            "sx1_lzma9e_60f_bytes": SX1_LZMA9E_60F_BYTES,
            "sx1_lzma9e_naive_extrapolation_to_n600": sx1_extrapolated,
            "measured_whole_corpus_lzma9e_bytes": whole_lzma,
            "coder_over_estimate_ratio": whole_lzma / sx1_estimate_bytes,
            "extrapolation_error_ratio": whole_lzma / sx1_extrapolated,
        },
    }


# ---------------------------------------------------------------------------
# STAGE split - static lexicon vs per-pair sentence, ADDRESS apart from PAYLOAD
# ---------------------------------------------------------------------------

def stage_split(frames: int = 600, band_radii=(1, 2, 3, 5, 8)) -> dict:
    """Split L* into a STATIC lexicon (sent once) and PER-PAIR sentences.

    ddm_hs1 measured that seg x CELL is concentrated AND static (Gini 0.8581)
    while seg x PAIR is flat (Gini 0.0858), and that a static top-128 address
    costs 62 B against a per-pair top-64 at 23,516 B -- 379x.  That finding was
    never applied to the grammar's design.  Here the static part is the
    per-pixel MODE over the corpus, and the per-pair part is the residual.

    ADDRESS is priced apart from PAYLOAD because they are not the same
    quantity.  The load-bearing question is whether the ADDRESS is a
    deterministic function of already-transmitted payload -- if it is, it is
    GENERIC and free (the receiver re-derives it), and only the PAYLOAD counts.
    """
    gt, _ = _load(mmap=True)
    h, w = gt.shape[1], gt.shape[2]
    px_frame = h * w

    # --- static lexicon: per-pixel modal class over the corpus -------------
    counts = np.zeros((N_CLASSES, h, w), dtype=np.int32)
    for t in range(frames):
        g = np.asarray(gt[t])
        for cls in range(N_CLASSES):
            counts[cls] += g == cls
    static = counts.argmax(axis=0).astype(np.uint8)
    static_bytes = _codec_bytes(static.tobytes(), "lzma9e")

    # --- the static field's own boundary band (GENERIC: a dilation of an
    #     already-transmitted symbol -- the receiver re-derives it for free) --
    sb = np.zeros((h, w), dtype=bool)
    sb[:-1, :] |= static[:-1, :] != static[1:, :]
    sb[1:, :] |= static[:-1, :] != static[1:, :]
    sb[:, :-1] |= static[:, :-1] != static[:, 1:]
    sb[:, 1:] |= static[:, :-1] != static[:, 1:]
    bands = {}
    for r in band_radii:
        k = 2 * r + 1
        bands[r] = cv2.dilate(sb.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)

    # --- per-pair residual --------------------------------------------------
    resid_px = 0
    addr_explicit_bytes = 0
    payload_bytes = 0
    in_band = {r: 0 for r in band_radii}
    resid_masks = []
    payload_syms = []
    t0 = time.time()
    for t in range(frames):
        g = np.asarray(gt[t])
        r_mask = g != static
        k = int(r_mask.sum())
        resid_px += k
        resid_masks.append(np.packbits(r_mask.ravel()))
        payload_syms.append(g[r_mask].copy())
        for r in band_radii:
            in_band[r] += int((r_mask & bands[r]).sum())
        if t % 150 == 0:
            print(f"[gt2:split] {t}/{frames} resid={k} {time.time()-t0:.0f}s", flush=True)

    addr_explicit_bytes = _codec_bytes(b"".join(m.tobytes() for m in resid_masks), "lzma9e")
    payload_bytes = _codec_bytes(b"".join(p.tobytes() for p in payload_syms), "lzma9e")

    band_rows = []
    for r in band_radii:
        band_px = int(bands[r].sum())
        cover = in_band[r] / resid_px if resid_px else None
        # Restricted address: one bit per BAND pixel per frame, entropy-coded.
        # The band itself is GENERIC (re-derived from the static field), so the
        # counted part is only the in-band occupancy bits.
        band_rows.append(
            {
                "radius": r,
                "band_px_per_frame": band_px,
                "band_frac_of_frame": band_px / px_frame,
                "residual_coverage": cover,
                "residual_px_outside_band": resid_px - in_band[r],
                "address_domain_reduction_vs_full_frame": px_frame / band_px if band_px else None,
                "symbol_type_of_band": "GENERIC",
                "why_generic": "band = dilate(boundary(static_lexicon)); a deterministic function of already-transmitted payload, so the receiver re-derives it and it is NOT counted",
            }
        )

    return {
        "stage": "split",
        "frames": frames,
        "px_per_frame": px_frame,
        "static_lexicon": {
            "symbol_type": "FITTED",
            "why_fitted": "the per-pixel modal class is induced from THIS clip; it is video-derived and must be COUNTED, exactly",
            "raw_px": px_frame,
            "lzma9e_bytes": static_bytes,
            "amortised_bytes_per_frame": static_bytes / frames,
        },
        "per_pair_residual": {
            "symbol_type": "FITTED",
            "total_residual_px": resid_px,
            "residual_px_per_frame": resid_px / frames,
            "residual_frac_of_field": resid_px / (frames * px_frame),
            "address_explicit_lzma9e_bytes": addr_explicit_bytes,
            "payload_lzma9e_bytes": payload_bytes,
            "address_plus_payload_bytes": addr_explicit_bytes + payload_bytes,
            "address_share": addr_explicit_bytes / (addr_explicit_bytes + payload_bytes),
        },
        "total_two_part_bytes": static_bytes + addr_explicit_bytes + payload_bytes,
        "band_analysis": band_rows,
        "elapsed_s": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# STAGE temporal - the GENERATE production's zero-model baseline
# ---------------------------------------------------------------------------

def stage_temporal(frames: int = 600) -> dict:
    """Price the GENERATE production's zero-model baseline: L*_t given L*_{t-1}.

    The mdl stage measured that IMPLICIT temporal context (whole-corpus coding)
    is worth 26.5% over per-frame-independent coding.  The split stage measured
    that EXPLICIT static factoring (mode field + residual) LOSES to implicit
    context.  This stage asks the symmetric temporal question: does an EXPLICIT
    frame-difference factoring (address = "where did the field change since the
    previous pair's frame_1", payload = the new classes there) beat the
    implicit whole-corpus coder?  All rows are REAL coder outputs.

    Corpus geometry note: frame t here is pair t's frame_1; consecutive pairs'
    frame_1 fields are two video frames apart, so the delta priced here is the
    2-frame ego-motion churn -- exactly the gap a per-pair sentence bridges.
    """
    gt, _ = _load(mmap=True)
    n = min(frames, gt.shape[0])

    prev = np.asarray(gt[0])
    frame0_bytes = _codec_bytes(prev.tobytes(), "lzma9e")

    churn_px = 0
    masks = []
    payloads = []
    per_frame_churn = []
    t0 = time.time()
    for t in range(1, n):
        cur = np.asarray(gt[t])
        m = cur != prev
        k = int(m.sum())
        churn_px += k
        per_frame_churn.append(k)
        masks.append(np.packbits(m.ravel()))
        payloads.append(cur[m].copy())
        prev = cur
    addr_bytes = _codec_bytes(b"".join(x.tobytes() for x in masks), "lzma9e")
    payload_bytes = _codec_bytes(b"".join(x.tobytes() for x in payloads), "lzma9e")
    total = frame0_bytes + addr_bytes + payload_bytes

    px_frame = int(np.asarray(gt[0]).size)
    return {
        "stage": "temporal",
        "frames": n,
        "symbol_type": "FITTED",
        "why_fitted": "every part (keyframe, churn masks, class payload) is THIS clip's content",
        "keyframe_lzma9e_bytes": frame0_bytes,
        "churn_px_total": churn_px,
        "churn_px_per_pair_step": churn_px / (n - 1),
        "churn_frac_of_frame_per_step": churn_px / ((n - 1) * px_frame),
        "address_lzma9e_bytes": addr_bytes,
        "payload_lzma9e_bytes": payload_bytes,
        "address_share": addr_bytes / (addr_bytes + payload_bytes),
        "total_temporal_two_part_bytes": total,
        "per_frame_churn_min_med_max": [
            int(np.min(per_frame_churn)),
            float(np.median(per_frame_churn)),
            int(np.max(per_frame_churn)),
        ],
        "comparisons": {
            "whole_corpus_implicit_lzma9e_bytes": 410584,
            "per_frame_independent_lzma9e_bytes": 558364,
            "temporal_two_part_over_implicit": total / 410584.0,
            "note": "comparison constants are the mdl stage's own measured rows (gt2_mdl.json)",
        },
        "elapsed_s": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# STAGE lane - region-paint vs curve-native, priced at matched fidelity
# ---------------------------------------------------------------------------

def _poly_program_bytes(polys: list[np.ndarray]) -> int:
    """Counted size of a polygon program string, entropy-coded for real.

    The vertex stream is delta-coded within each polygon (the deltas are small
    and highly skewed for a thin curved structure) and then run through a real
    coder.  The RASTERISER that expands this program is GENERIC (fillPoly is an
    algorithm, not clip content) and is NOT counted; only the program string is.
    """
    buf = bytearray()
    for p in polys:
        pts = p.reshape(-1, 2).astype(np.int32)
        buf += len(pts).to_bytes(2, "little")
        prev = np.zeros(2, dtype=np.int32)
        for xy in pts:
            d = xy - prev
            prev = xy
            for v in d:
                # zigzag + varint: a standard generic integer code
                z = (int(v) << 1) ^ (int(v) >> 31)
                while True:
                    b = z & 0x7F
                    z >>= 7
                    if z:
                        buf.append(b | 0x80)
                    else:
                        buf.append(b)
                        break
    return _codec_bytes(bytes(buf), "lzma9e")


def stage_lane(frames: int = 600, epsilons=(0.0, 0.5, 1.0, 1.5, 2.0, 3.0)) -> dict:
    """Price a Lane production three ways, at MATCHED fidelity.

    dd1's census (consumed, not re-measured) gives Lane area/perimeter = 1.407
    against Road 21.93, Undrivable 176.2, MyCar 97.4.  area/perimeter IS the
    efficiency of a REGION production: pixels delivered per unit of boundary
    description.  This stage tests that reading by pricing both grammars with a
    real coder, for Lane AND for a control class, and reporting the error each
    grammar introduces in the same unit the score charges (flips).
    """
    gt, _ = _load(mmap=True)
    h, w = gt.shape[1], gt.shape[2]

    results = {}
    for cls, name in ((1, "Lane"), (0, "Road"), (3, "Movable")):
        # G_raster: the class mask as a bitplane, real coder.
        packed = []
        area = 0
        for t in range(frames):
            m = np.asarray(gt[t]) == cls
            area += int(m.sum())
            packed.append(np.packbits(m.ravel()))
        raster_bytes = _codec_bytes(b"".join(p.tobytes() for p in packed), "lzma9e")

        # G_poly: contour polygons at several simplification tolerances.
        poly_rows = []
        for eps in epsilons:
            polys_all: list[np.ndarray] = []
            err_px = 0
            n_verts = 0
            n_comp = 0
            for t in range(frames):
                m = (np.asarray(gt[t]) == cls).astype(np.uint8)
                if not m.any():
                    continue
                contours, _ = cv2.findContours(m, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
                recon = np.zeros((h, w), np.uint8)
                simp = []
                for c in contours:
                    s = cv2.approxPolyDP(c, eps, True) if eps > 0 else c
                    simp.append(s)
                    n_verts += len(s)
                n_comp += len(contours)
                cv2.fillPoly(recon, simp, 1)
                err_px += int(np.count_nonzero(recon.astype(bool) ^ m.astype(bool)))
                polys_all.extend(simp)
            pbytes = _poly_program_bytes(polys_all)
            poly_rows.append(
                {
                    "epsilon_px": eps,
                    "symbol_type": "FITTED",
                    "why_fitted": "control points are THIS clip's geometry; the rasteriser that expands them is GENERIC and not counted",
                    "vertices": n_verts,
                    "contours": n_comp,
                    "program_lzma9e_bytes": pbytes,
                    "reconstruction_error_px": err_px,
                    "error_px_per_frame": err_px / frames,
                    # In the unit the score charges: an introduced error pixel is
                    # a potential flip.  Priced against W, the exchange rate.
                    "error_seg_S_if_all_became_flips": 100.0 * err_px / (frames * h * w),
                    "bytes_vs_raster": pbytes / raster_bytes if raster_bytes else None,
                }
            )
        results[name] = {
            "class_index": cls,
            "gt_area_px": area,
            "G_raster": {
                "symbol_type": "FITTED",
                "codec": "lzma9e",
                "bytes": raster_bytes,
                "bytes_per_area_px": raster_bytes / area if area else None,
            },
            "G_poly": poly_rows,
        }
        print(f"[gt2:lane] {name} raster={raster_bytes} B", flush=True)

    return {
        "stage": "lane",
        "frames": frames,
        "note": "dd1's area/perimeter census is CONSUMED here, not re-measured: Lane 1.407, Road 21.93, Undrivable 176.2, MyCar 97.4, Movable 11.07",
        "W_bytes_per_flip_exchange_rate": W_BYTES_PER_FLIP,
        "W_owner": "ddm_wf2 owns the per-mechanism price law; W is an exchange rate, not a cost",
        "per_class": results,
    }


STAGES = {
    "control": (stage_control, "gt2_control.json"),
    "verbs": (stage_verbs, "gt2_verbs.json"),
    "mdl": (stage_mdl, "gt2_mdl.json"),
    "split": (stage_split, "gt2_split.json"),
    "temporal": (stage_temporal, "gt2_temporal.json"),
    "lane": (stage_lane, "gt2_lane.json"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", required=True, choices=[*STAGES, "all"])
    ap.add_argument("--frames", type=int, default=600)
    args = ap.parse_args()

    if not GT_PATH.exists():
        raise SystemExit(f"corpus missing: {GT_PATH}")

    names = list(STAGES) if args.stage == "all" else [args.stage]
    for nm in names:
        fn, out = STAGES[nm]
        t0 = time.time()
        payload = fn() if nm == "control" else fn(frames=args.frames)
        payload["wall_clock_s"] = time.time() - t0
        _write(out, payload)


if __name__ == "__main__":
    main()
