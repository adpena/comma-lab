# SPDX-License-Identifier: MIT
"""Necessity solver — inverse of the flattened scorer factorization, per Morse-Smale stratum.

P0-of-all-P0 crown mechanism (operator 2026-07-15): "determine what is necessary for
realization by inverse of the flattened factorization" + "per edge and saddle" + the
Kolmogorov guiding meta-question ("shortest PROGRAM whose fixed point is the witness;
generators FREE in inflate.py per rule 118, count only the irreducible seed").

The flattened chain (``.omx/research/frozen_scorer_exact_factorization_20260715.md``)::

    d_seg = decision ∘ N_seg ∘ A          (A = shared bilinear resize (874,1164)->(384,512))

is inverted stratum by stratum on the REAL frozen weights / REAL cached GT:

  decision⁻¹  rank-4 argmax polytope; margins are the CACHED bit-exact top1-top2 field
              (gt_n600.npz 'margins'; parity vs the real forward == 0.0 per stage_b1).
  N⁻¹         first-order min-norm camera-res displacement m/||∂m/∂x_cam|| through the
              REAL frozen CPU-torch SegNet INCLUDING the resize (full-chain VJP),
              on a sampled pixel subset (scope labeled).
  A⁻¹         EXACT support pullback through the closed-form separable resize matrices
              (tac.optimization.evaluator_invisibility_basis, derivation-grade).
  ∩ uint8     realization tightness = per-coordinate amplitude of the min-norm
              displacement vs 1 LSB (sub-LSB ⟹ realization-limited, not gradient-limited).

Morse-Smale strata of the cached argmax partition (n600, exact):
  CELLS   = per-class interiors (2-cells)
  EDGES   = per-class-pair separatrix cracks (1-cells; crack = label change between
            4-adjacent pixels; arc length == crack count in the lattice metric)
  SADDLES = junction vertices where the 2x2 pixel block holds >=3 distinct labels
            (0-cells; == crack-graph vertices of degree >= 3)

Rate ladders (both emitted; models explicit):
  H-ladder (entropy of the necessary preimage): per-pair
      bits = cracks x H_turn + components x (log2(384*512) + 2)
      with H_turn = H0{straight, left, right} MEASURED from degree-2 vertex turn stats.
  K-ladder (Kolmogorov generator+seed, the superseding frame): generator = closed-polygon
      rasterizer + argmax-region fill (deterministic, FREE in inflate.py per rule 118);
      seed = Douglas-Peucker simplified boundary vertices at eps in {0.5, 1, 2} px,
      int16-delta-packed and brotli -q11 coded (MEASURED bytes). Geometric tolerance
      only — NOT a d_seg claim (the measured D-ladder rows are the d_seg-verified sibling).

Axes / honesty: all measurements ``[macOS-CPU advisory]`` on the frozen CPU-torch fp32
scorer / bit-exact cached GT; research_only; score_claim=false; promotable=false; no
dispatch. The n600 stages use ALL 600 scored pairs; subset stages label their stride.

Artifacts: ``experiments/results/necessity_solver_20260715/*.json``
Memo:      ``.omx/research/necessity_solver_inverse_factorization_20260715.md``
"""

from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

GT_N600 = "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
OUT_DIR = "experiments/results/necessity_solver_20260715"
CLASSES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
H_S, W_S = 384, 512  # scorer res
H_C, W_C = 874, 1164  # camera res
AXIS = "[macOS-CPU advisory] frozen CPU-torch fp32; bit-exact cached GT (gt_n600.npz)"

# margin histogram: fixed bins so quantiles are computable from streamed counts.
HIST_MAX = 40.0
HIST_BINS = 4000
_HIST_EDGES = np.linspace(0.0, HIST_MAX, HIST_BINS + 1)


def _pair_code(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    return lo * 5 + hi


def _pair_name(code: int) -> str:
    return f"{CLASSES[code // 5]}-{CLASSES[code % 5]}"


def _hist_quantile(counts: np.ndarray, q: float) -> float:
    c = np.cumsum(counts)
    if c[-1] == 0:
        return float("nan")
    k = np.searchsorted(c, q * c[-1])
    return float(_HIST_EDGES[min(k + 1, HIST_BINS)])


def _load_gt(keys: tuple[str, ...]) -> dict[str, np.ndarray]:
    z = np.load(GT_N600)
    return {k: z[k] for k in keys}


# ---------------------------------------------------------------------------
# Stage STRAT — n600 Morse-Smale stratification + cached-margin per-stratum stats
# ---------------------------------------------------------------------------
def stage_strat() -> dict:
    import cv2

    g = _load_gt(("lstars", "margins"))
    lstars, margins = g["lstars"], g["margins"]
    n = lstars.shape[0]

    pair_cracks = np.zeros(25, dtype=np.int64)
    pair_straight = np.zeros(25, dtype=np.int64)
    pair_corner = np.zeros(25, dtype=np.int64)
    pair_components = np.zeros(25, dtype=np.int64)
    pair_margin_hist = np.zeros((25, HIST_BINS), dtype=np.int64)
    cell_margin_hist = np.zeros((5, HIST_BINS), dtype=np.int64)
    saddle_margin_hist = np.zeros(HIST_BINS, dtype=np.int64)
    cell_area = np.zeros(5, dtype=np.int64)
    edge_px_total = 0
    saddle_vertex_total = 0
    triple_counts: dict[str, int] = {}
    saddle4_total = 0

    for i in range(n):
        lab = lstars[i]
        m = margins[i]
        dx = lab[:, :-1] != lab[:, 1:]  # crack between horizontally adjacent pixels
        dy = lab[:-1, :] != lab[1:, :]  # crack between vertically adjacent pixels

        # per-pair crack counts (arc length in lattice metric)
        cx = _pair_code(lab[:, :-1][dx], lab[:, 1:][dx])
        cy = _pair_code(lab[:-1, :][dy], lab[1:, :][dy])
        pair_cracks += np.bincount(cx, minlength=25) + np.bincount(cy, minlength=25)

        # per-pair margin samples: both pixels flanking every crack
        for code_arr, m_a, m_b in (
            (cx, m[:, :-1][dx], m[:, 1:][dx]),
            (cy, m[:-1, :][dy], m[1:, :][dy]),
        ):
            for samp in (m_a, m_b):
                bins = np.clip((samp / HIST_MAX * HIST_BINS).astype(np.int64), 0, HIST_BINS - 1)
                np.add.at(pair_margin_hist, (code_arr, bins), 1)

        # vertex degrees on the (H-1)x(W-1) dual grid
        up = dx[:-1, :]
        down = dx[1:, :]
        left = dy[:, :-1]
        right = dy[:, 1:]
        deg = up.astype(np.int8) + down + left + right
        straight = (up & down & ~left & ~right) | (left & right & ~up & ~down)
        deg2 = deg == 2
        corner = deg2 & ~straight

        # 2x2 block distinct-label count (saddles = >=3 distinct == deg>=3 vertices)
        blk = np.stack([lab[:-1, :-1], lab[:-1, 1:], lab[1:, :-1], lab[1:, 1:]])
        blk_sorted = np.sort(blk, axis=0)
        distinct = 1 + (blk_sorted[1] != blk_sorted[0]).astype(np.int8) \
            + (blk_sorted[2] != blk_sorted[1]) + (blk_sorted[3] != blk_sorted[2])
        saddle = distinct >= 3
        saddle_vertex_total += int(saddle.sum())
        saddle4_total += int((distinct == 4).sum())

        # per-triple counts (distinct==3): key by the 3 sorted labels present
        t_mask = distinct == 3
        if t_mask.any():
            s0 = blk_sorted[0][t_mask]
            s = blk_sorted[:, t_mask]
            # the 3 distinct values among 4 sorted entries
            uniq = np.where(s[1] != s[0], s[1], s[2])
            uniq2 = np.where(s[3] != s[2], s[3], s[2])
            code3 = s0 * 25 + uniq * 5 + uniq2
            for c3, cnt in zip(*np.unique(code3, return_counts=True), strict=True):
                key = f"{CLASSES[int(c3) // 25]}-{CLASSES[(int(c3) // 5) % 5]}-{CLASSES[int(c3) % 5]}"
                triple_counts[key] = triple_counts.get(key, 0) + int(cnt)

        # per-pair turn stats at 2-class degree-2 vertices
        two_class = distinct == 2
        for mask, tgt in ((straight & deg2 & two_class, pair_straight),
                          (corner & two_class, pair_corner)):
            if mask.any():
                pc = _pair_code(blk_sorted[0][mask], blk_sorted[3][mask])
                tgt += np.bincount(pc, minlength=25)

        # saddle-flank pixel margins (the 4 pixels around each saddle vertex)
        if saddle.any():
            ys, xs = np.where(saddle)
            for oy, ox in ((0, 0), (0, 1), (1, 0), (1, 1)):
                samp = m[ys + oy, xs + ox]
                bins = np.clip((samp / HIST_MAX * HIST_BINS).astype(np.int64), 0, HIST_BINS - 1)
                np.add.at(saddle_margin_hist, bins, 1)

        # cells: interior (non-boundary) pixels per class
        bmask = np.zeros_like(lab, dtype=bool)
        bmask[:, :-1] |= dx
        bmask[:, 1:] |= dx
        bmask[:-1, :] |= dy
        bmask[1:, :] |= dy
        edge_px_total += int(bmask.sum())
        interior = ~bmask
        for c in range(5):
            cm = interior & (lab == c)
            cell_area[c] += int(cm.sum())
            if cm.any():
                samp = m[cm]
                bins = np.clip((samp / HIST_MAX * HIST_BINS).astype(np.int64), 0, HIST_BINS - 1)
                np.add.at(cell_margin_hist, (np.full(len(bins), c), bins), 1)

        # per-pair connected components of the boundary-pixel mask (curve count)
        for code in np.unique(np.concatenate([cx, cy])):
            pm = np.zeros_like(lab, dtype=np.uint8)
            a, b = code // 5, code % 5
            adj_x = dx & (
                ((lab[:, :-1] == a) & (lab[:, 1:] == b)) | ((lab[:, :-1] == b) & (lab[:, 1:] == a))
            )
            adj_y = dy & (
                ((lab[:-1, :] == a) & (lab[1:, :] == b)) | ((lab[:-1, :] == b) & (lab[1:, :] == a))
            )
            pm[:, :-1] |= adj_x
            pm[:, 1:] |= adj_x
            pm[:-1, :] |= adj_y
            pm[1:, :] |= adj_y
            ncomp, _ = cv2.connectedComponents(pm, connectivity=8)
            pair_components[code] += ncomp - 1

    from tac.canonical_equations.segnet_head_rank4_flipdist_20260715 import HEAD_PAIR_NORMS

    pairs_out = {}
    for code in range(25):
        if pair_cracks[code] == 0:
            continue
        name = _pair_name(code)
        s, c = int(pair_straight[code]), int(pair_corner[code])
        p_corner = c / (s + c) if (s + c) else float("nan")
        # H0 of {straight, left, right}; corners split evenly by symmetry
        if 0.0 < p_corner < 1.0:
            h_turn = -(1 - p_corner) * np.log2(1 - p_corner) - p_corner * np.log2(p_corner / 2)
        else:
            h_turn = float("nan")
        med = _hist_quantile(pair_margin_hist[code], 0.5)
        pairs_out[name] = {
            "cracks_total_n600": int(pair_cracks[code]),
            "cracks_per_frame": pair_cracks[code] / n,
            "components_total_n600": int(pair_components[code]),
            "components_per_frame": pair_components[code] / n,
            "turn_straight": s,
            "turn_corner": c,
            "p_corner": p_corner,
            "H_turn_bits_per_step": float(h_turn),
            "edge_margin_med": med,
            "edge_margin_p10": _hist_quantile(pair_margin_hist[code], 0.1),
            "edge_margin_p90": _hist_quantile(pair_margin_hist[code], 0.9),
            "flipdist_feat_med": med / HEAD_PAIR_NORMS[name] if name in HEAD_PAIR_NORMS else None,
        }

    out = {
        "scope": f"n600 (ALL scored pairs) cached lstars+margins; {AXIS}",
        "n_frames": n,
        "cells": {
            CLASSES[c]: {
                "interior_px_total": int(cell_area[c]),
                "interior_frac": cell_area[c] / (n * H_S * W_S),
                "interior_margin_med": _hist_quantile(cell_margin_hist[c], 0.5),
                "interior_margin_p10": _hist_quantile(cell_margin_hist[c], 0.1),
            }
            for c in range(5)
        },
        "edges": pairs_out,
        "edge_px_total": edge_px_total,
        "edge_px_frac": edge_px_total / (n * H_S * W_S),
        "saddles": {
            "vertices_total_n600": saddle_vertex_total,
            "vertices_per_frame": saddle_vertex_total / n,
            "distinct4_total": saddle4_total,
            "per_triple": dict(sorted(triple_counts.items(), key=lambda kv: -kv[1])),
            "flank_margin_med": _hist_quantile(saddle_margin_hist, 0.5),
            "flank_margin_p10": _hist_quantile(saddle_margin_hist, 0.1),
            "flank_margin_p90": _hist_quantile(saddle_margin_hist, 0.9),
        },
    }
    return out


# ---------------------------------------------------------------------------
# Stage KLADDER — generator+seed reduction: DP-simplified boundary polygons,
# int16-delta packed, brotli -q11 (MEASURED seed bytes; generator FREE rule 118)
# ---------------------------------------------------------------------------
def stage_kladder(eps_list: tuple[float, ...] = (0.5, 1.0, 2.0)) -> dict:
    import brotli
    import cv2

    g = _load_gt(("lstars",))
    lstars = g["lstars"]
    n = lstars.shape[0]

    out: dict = {
        "scope": f"n600 cached lstars; {AXIS}",
        "model": (
            "generator (FREE, rule 118): closed-polygon rasterizer + per-class region fill; "
            "seed (COUNTED): DP-simplified per-class region contours, first vertex u16 pair + "
            "int16 vertex deltas, brotli -q11. Each physical inter-class edge appears in exactly "
            "2 class contours -> /2 shared-edge adjustment (image-border arcs single-counted; "
            "adjustment therefore slightly UNDER-corrects). Geometric tol=eps px @ (384,512); "
            "NOT a d_seg claim."
        ),
        "eps": {},
    }
    for eps in eps_list:
        vtx_total = 0
        comp_total = 0
        payload = bytearray()
        per_class_vtx = [0] * 5
        for i in range(n):
            lab = lstars[i].astype(np.uint8)
            for c in range(5):
                mask = (lab == c).astype(np.uint8)
                if not mask.any():
                    continue
                contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
                for cont in contours:
                    if len(cont) < 3:
                        continue
                    ap = cv2.approxPolyDP(cont, eps, True)[:, 0, :]
                    vtx_total += len(ap)
                    per_class_vtx[c] += len(ap)
                    comp_total += 1
                    v = ap.astype(np.int32)
                    first = v[0].astype(np.uint16)
                    deltas = np.diff(v, axis=0).astype(np.int16)
                    payload += first.tobytes() + np.int16(len(deltas)).tobytes() + deltas.tobytes()
        coded = brotli.compress(bytes(payload), quality=11)
        out["eps"][str(eps)] = {
            "vertices_total_n600": vtx_total,
            "vertices_per_frame": vtx_total / n,
            "components_total_n600": comp_total,
            "per_class_vertices": {CLASSES[c]: per_class_vtx[c] for c in range(5)},
            "raw_payload_bytes": len(payload),
            "brotli_q11_bytes": len(coded),
            "brotli_q11_bytes_shared_edge_adjusted": len(coded) / 2,
            "bytes_per_frame_adjusted": len(coded) / 2 / n,
        }
    return out


# ---------------------------------------------------------------------------
# Stage ASUPPORT — exact A^T support pullback per stratum (camera-res necessity)
# ---------------------------------------------------------------------------
def stage_asupport(stride: int = 25) -> dict:
    from tac.optimization.evaluator_invisibility_basis import _resize_1d_matrix

    g = _load_gt(("lstars",))
    lstars = g["lstars"]
    frames = list(range(0, lstars.shape[0], stride))

    r_h = _resize_1d_matrix(H_C, H_S)  # (384, 874)
    r_w = _resize_1d_matrix(W_C, W_S)  # (512, 1164)
    zero_rows = np.where(r_h.sum(axis=0) == 0.0)[0]
    zero_cols = np.where(r_w.sum(axis=0) == 0.0)[0]
    zw_mask = np.zeros((H_C, W_C), dtype=bool)
    zw_mask[zero_rows, :] = True
    zw_mask[:, zero_cols] = True
    zw_frac = float(zw_mask.mean())

    fracs = {"saddle": [], "edge": [], "cell_loose": []}
    for i in frames:
        lab = lstars[i]
        dx = lab[:, :-1] != lab[:, 1:]
        dy = lab[:-1, :] != lab[1:, :]
        bmask = np.zeros_like(lab, dtype=bool)
        bmask[:, :-1] |= dx
        bmask[:, 1:] |= dx
        bmask[:-1, :] |= dy
        bmask[1:, :] |= dy
        blk = np.stack([lab[:-1, :-1], lab[:-1, 1:], lab[1:, :-1], lab[1:, 1:]])
        bs = np.sort(blk, axis=0)
        distinct = 1 + (bs[1] != bs[0]).astype(np.int8) + (bs[2] != bs[1]) + (bs[3] != bs[2])
        saddle = distinct >= 3
        smask = np.zeros_like(lab, dtype=bool)
        ys, xs = np.where(saddle)
        for oy, ox in ((0, 0), (0, 1), (1, 0), (1, 1)):
            smask[ys + oy, xs + ox] = True

        def pull(mask: np.ndarray) -> np.ndarray:
            # macOS Accelerate dgemm raises a spurious divide-by-zero FP flag here;
            # output verified bit-exact vs einsum (max abs diff 0.0), so suppress.
            with np.errstate(divide="ignore", invalid="ignore"):
                return (r_h.T @ mask.astype(np.float64) @ r_w) > 0.0

        cam_saddle = pull(smask)
        cam_edge = pull(bmask & ~smask) & ~cam_saddle
        cam_cell = ~(cam_saddle | cam_edge) & ~zw_mask
        fracs["saddle"].append(float(cam_saddle.mean()))
        fracs["edge"].append(float(cam_edge.mean()))
        fracs["cell_loose"].append(float(cam_cell.mean()))

    return {
        "scope": f"stride-{stride} subset ({len(frames)} frames of n600); EXACT closed-form resize matrices",
        "camera_zero_weight_frac_certified": zw_frac,
        "camera_frac_mean": {k: float(np.mean(v)) for k, v in fracs.items()},
        "camera_frac_note": (
            "priority attribution saddle > edge > cell; 'cell_loose' is only loosely necessary "
            "(argmax-interior, B2 blind: membership not values); zero-weight is certified FREE"
        ),
    }


# ---------------------------------------------------------------------------
# Stage VJP — full-chain (camera-res -> resize -> SegNet) min-norm displacement
# ---------------------------------------------------------------------------
def stage_vjp(frame_stride: int = 75, seed: int = 0) -> dict:
    import sys

    import torch
    import torch.nn.functional as tfun

    torch.set_num_threads(4)
    sys.path.insert(0, "upstream")
    import segmentation_models_pytorch as smp
    from safetensors.torch import load_file

    model = smp.Unet("tu-efficientnet_b2", classes=5, activation=None, encoder_weights=None)
    model.load_state_dict(load_file("upstream/models/segnet.safetensors", device="cpu"), strict=True)
    model.eval()

    g = _load_gt(("lstars", "gt_f1"))
    lstars, gt_f1 = g["lstars"], g["gt_f1"]
    frames = list(range(0, lstars.shape[0], frame_stride))
    rng = np.random.default_rng(seed)
    rows = []
    runnerup_neighbor_agree = [0, 0]

    for i in frames:
        lab = lstars[i]
        dx = lab[:, :-1] != lab[:, 1:]
        dy = lab[:-1, :] != lab[1:, :]
        bmask = np.zeros_like(lab, dtype=bool)
        bmask[:, :-1] |= dx
        bmask[:, 1:] |= dx
        bmask[:-1, :] |= dy
        bmask[1:, :] |= dy
        blk = np.stack([lab[:-1, :-1], lab[:-1, 1:], lab[1:, :-1], lab[1:, 1:]])
        bs = np.sort(blk, axis=0)
        distinct = 1 + (bs[1] != bs[0]).astype(np.int8) + (bs[2] != bs[1]) + (bs[3] != bs[2])
        smask = np.zeros_like(lab, dtype=bool)
        ys, xs = np.where(distinct >= 3)
        for oy, ox in ((0, 0), (0, 1), (1, 0), (1, 1)):
            smask[ys + oy, xs + ox] = True

        picks: list[tuple[int, int, str]] = []
        sy, sx = np.where(smask)
        if len(sy):
            for j in rng.choice(len(sy), size=min(3, len(sy)), replace=False):
                picks.append((int(sy[j]), int(sx[j]), "saddle"))
        cx = _pair_code(lab[:, :-1][dx], lab[:, 1:][dx])
        cy_ = _pair_code(lab[:-1, :][dy], lab[1:, :][dy])
        counts = np.bincount(cx, minlength=25) + np.bincount(cy_, minlength=25)
        emask = bmask & ~smask
        for code in np.argsort(-counts)[:5]:
            if counts[code] == 0:
                continue
            a, b = code // 5, code % 5
            pmask = emask & ((lab == a) | (lab == b))
            # restrict to pixels actually adjacent to the other class of the pair
            py, px = np.where(pmask)
            if not len(py):
                continue
            keep = []
            for j in range(len(py)):
                y, x = py[j], px[j]
                nb = {lab[max(y - 1, 0), x], lab[min(y + 1, H_S - 1), x],
                      lab[y, max(x - 1, 0)], lab[y, min(x + 1, W_S - 1)]}
                if {a, b} <= (nb | {lab[y, x]}):
                    keep.append(j)
                if len(keep) >= 64:
                    break
            if keep:
                for j in rng.choice(keep, size=min(2, len(keep)), replace=False):
                    picks.append((int(py[j]), int(px[j]), f"edge:{_pair_name(int(code))}"))
        # interior contrast pixels (2 largest classes)
        interior = ~bmask
        for c in np.argsort(-np.bincount(lab.ravel(), minlength=5))[:2]:
            iy, ix = np.where(interior & (lab == c))
            if len(iy):
                j = rng.integers(len(iy))
                picks.append((int(iy[j]), int(ix[j]), f"interior:{CLASSES[c]}"))

        x_cam = torch.from_numpy(gt_f1[i]).permute(2, 0, 1).float().unsqueeze(0)
        x_cam.requires_grad_(True)
        x_s = tfun.interpolate(x_cam, size=(H_S, W_S), mode="bilinear")
        logits = model(x_s)[0]
        top2 = torch.topk(logits, 2, dim=0)
        marg = top2.values[0] - top2.values[1]
        runner = top2.indices[1].detach().numpy()

        for py_, px_, kind in picks:
            if x_cam.grad is not None:
                x_cam.grad = None
            marg[py_, px_].backward(retain_graph=True)
            gr = x_cam.grad[0].detach().numpy()
            gn = float(np.sqrt((gr ** 2).sum()))
            m = float(marg[py_, px_].item())
            if gn <= 0:
                continue
            gmax = float(np.abs(gr).max())
            a_max = m * gmax / (gn * gn)  # largest per-coordinate amplitude of min-norm δ*
            if kind.startswith("edge"):
                nb = {lab[max(py_ - 1, 0), px_], lab[min(py_ + 1, H_S - 1), px_],
                      lab[py_, max(px_ - 1, 0)], lab[py_, min(px_ + 1, W_S - 1)]}
                agree = int(runner[py_, px_]) in nb
                runnerup_neighbor_agree[0 if agree else 1] += 1
            rows.append({
                "frame": i, "y": py_, "x": px_, "kind": kind,
                "class": CLASSES[int(lab[py_, px_])],
                "runner_up": CLASSES[int(runner[py_, px_])],
                "margin": m, "gradnorm_cam": gn,
                "flipdist_L2_cam_px_units": m / gn,
                "min_norm_delta_max_coord": a_max,
                "sub_lsb": bool(a_max < 0.5),
            })
        del logits, marg, x_s, x_cam

    def _agg(pred):
        sel = [r["flipdist_L2_cam_px_units"] for r in rows if pred(r)]
        amp = [r["min_norm_delta_max_coord"] for r in rows if pred(r)]
        sub = [r["sub_lsb"] for r in rows if pred(r)]
        if not sel:
            return None
        return {
            "n": len(sel),
            "flipdist_med": float(np.median(sel)),
            "flipdist_p10": float(np.percentile(sel, 10)),
            "amp_max_coord_med": float(np.median(amp)),
            "frac_sub_lsb": float(np.mean(sub)),
        }

    return {
        "scope": (
            f"frame-stride {frame_stride} ({len(frames)} frames of n600), sampled pixels "
            "(edge candidates scanned row-major, first 64 per pair -> biased toward each "
            f"pair's topmost extent); full-chain VJP incl. exact resize; {AXIS}"
        ),
        "runnerup_equals_neighbor_label": {
            "agree": runnerup_neighbor_agree[0],
            "disagree": runnerup_neighbor_agree[1],
        },
        "agg": {
            "saddle": _agg(lambda r: r["kind"] == "saddle"),
            "edge": _agg(lambda r: r["kind"].startswith("edge")),
            "interior": _agg(lambda r: r["kind"].startswith("interior")),
        },
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Stage FLOORS — combine H-ladder + K-ladder into per-stratum necessity table
# ---------------------------------------------------------------------------
def stage_floors(strat: dict, kladder: dict) -> dict:
    n = strat["n_frames"]
    overhead_bits = np.log2(H_S * W_S) + 2.0  # start point + initial direction per curve
    per_pair = {}
    total_bits = 0.0
    for name, e in strat["edges"].items():
        h = e["H_turn_bits_per_step"]
        if not np.isfinite(h):
            continue
        bits = e["cracks_total_n600"] * h + e["components_total_n600"] * overhead_bits
        per_pair[name] = {
            "H_ladder_bits_n600": bits,
            "H_ladder_bytes_per_frame": bits / 8.0 / n,
        }
        total_bits += bits
    k1 = kladder["eps"]["1.0"]
    out = {
        "model_H": "bits = cracks x H_turn + components x (log2(HW)+2); spatial-only (no temporal prediction)",
        "model_K": kladder["model"],
        "edges_H_ladder_total_bytes_n600": total_bits / 8.0,
        "edges_H_ladder_bytes_per_frame": total_bits / 8.0 / n,
        "edges_K_ladder_bytes_n600_eps1": k1["brotli_q11_bytes_shared_edge_adjusted"],
        "edges_K_ladder_bytes_per_frame_eps1": k1["bytes_per_frame_adjusted"],
        "per_pair": per_pair,
        "cells_seed": "5 class palettes (~15 bytes/video) given edges; membership implied by the edge complex (B2 blind interior)",
        "saddles_seed": (
            "MARGINAL ~0 bytes given coded edges: saddles are crack-graph vertices of degree>=3, "
            "implied by curve intersections; their cost is PRECISION on edge curves near junctions "
            "(tie-locus #360), quantified by the saddle-vs-edge margin/flip-distance ratio, not bytes"
        ),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stages", default="strat,kladder,asupport,vjp,floors",
                    help="comma list of stages to run")
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    stages = args.stages.split(",")
    results: dict[str, dict] = {}

    def _emit(name: str, obj: dict) -> None:
        path = os.path.join(args.out_dir, f"{name}.json")
        with open(path, "w") as fh:
            json.dump(obj, fh, indent=1, default=float)
        print(f"[necessity] wrote {path}")

    for name, fn in (("strat", stage_strat), ("kladder", stage_kladder),
                     ("asupport", stage_asupport), ("vjp", stage_vjp)):
        if name in stages:
            t0 = time.time()
            results[name] = fn()
            results[name]["elapsed_sec"] = time.time() - t0
            _emit(name, results[name])
    if "floors" in stages:
        def _load(name: str) -> dict:
            with open(os.path.join(args.out_dir, f"{name}.json")) as fh:
                return json.load(fh)

        strat = results.get("strat") or _load("strat")
        klad = results.get("kladder") or _load("kladder")
        _emit("floors", stage_floors(strat, klad))


if __name__ == "__main__":
    main()
