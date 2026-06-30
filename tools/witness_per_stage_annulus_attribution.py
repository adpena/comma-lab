#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""PER-STAGE, PER-PIXEL/PER-CLASS d_seg ATTRIBUTION across the level-set witness run.

Diffs the REALIZED SegNet-argmax across the run's preserved per-stage checkpoints
(CE -> tau_softplus -> l7 -> Muon) to answer FOUR operator questions:

  (1) WHERE the d_seg improvements come from  (which pixels / classes / regions each stage fixes)
  (2) what is STILL INTRACTABLE              (residual no stage corrects)
  (3) what is being improved NOW             (the Muon stage)
  (4) whether it COHERES                     (does each stage match its expected role / prime later)

PLUS (observability -> manipulability bridge): for each persistent-wrong residual it emits a
recommended SURGICAL REPAIR (LEARN / STORE / DETERMINISTIC / UNIWARD), split by PRIMED (realized
SegNet margin-toward-GT MOVING across the stage diffs -> LEARN) vs STUCK (margin flat ->
STORE/DETERMINISTIC), ranked by Delta d_seg-per-byte under through-R survival.

AUTHORITY (NO-FAKE):
  * REAL render-through-R + REAL CPU-torch SegNet argmax -- the SAME functions the trainer's
    ``realized_verdict()`` uses (imported, NOT reimplemented):
      - ``levelset_rgb_forward_numpy`` / ``int8_dequant_params``  (tac.boundary_math.lever_b_levelset_generator)
      - ``self_orientation_directional_feats``                    (tac.boundary_math.lever_b_generator)
      - ``_torch_R_to_camera_uint8`` / ``_build_render_coords`` / ``load_gt_from_cache``
        (experiments/train_witness_realized_through_R_mlx)
    Each ckpt is rendered with ITS OWN cfg scalars (softmax_temp/hosc/activation/chroma/bank/...),
    int8-DEQUANTIZED (the EMA shadow ships as int8 -> the verdict renders the DEPLOY weights).
  * SELF-ORIENT directional feats are NOT stored in the npz (trainer trajectory state). They are
    reconstructed by the DEPLOY FIXED-POINT (dirf from zeros, ``--so-iters`` updates of
    ``self_orientation_directional_feats`` on the decoder's OWN argmax, early-stop on argmax
    stability) -- EXACTLY the byte-close/inflate authority (tools/levelset_byte_close_and_eval.py,
    so_iters default 4). This is reproducible-from-npz-alone AND identical-method across all stages
    (apples-to-apples). It may differ slightly from the trainer's live trajectory d_seg (the dir
    feats accumulated over training) -- a KNOWN, documented gap; the realized number here is the
    deploy TRUTH (per the byte-close docstring "any gap vs the trainer's number is itself a finding").
  * CANONICAL CLASS ORDER (CLAUDE.md, NON-NEGOTIABLE; NOT luma-sort):
      0=Road  1=Lane  2=Undrivable(incl sky)  3=Movable(cars)  4=MyCar(ego hood)
  * numpy-fp32 is the verdict authority. Everything here is tagged
      ``[macOS-numpy advisory . NON-PROMOTABLE]``
    -- the realized-through-R verdict surface, NOT a contest score. The frontier pointer is
    0.19110 and UNMOVED; this tool does not move it.

CPU/numpy only. No MPS, no GPU, no MLX. Read-only on checkpoints. Does not disturb live training.

Usage:
  .venv/bin/python tools/witness_per_stage_annulus_attribution.py \\
      --ckpt CE=experiments/results/levelset_openpilot_seeded_n200_DEPLOY/levelset_ckpt_stageCE_ep299.npz \\
      --ckpt Tau=experiments/results/levelset_openpilot_seeded_n200_DEPLOY/levelset_ckpt_stageTau_ep599.npz \\
      --ckpt l7=experiments/results/levelset_thetastar_l7_arm/levelset_ema_stageL7arm_final_ep725.npz \\
      --ckpt MuonStart=experiments/results/levelset_thetastar_muon_arm/levelset_ckpt_stageMuonStart_ep726.npz \\
      --ckpt MuonBest=experiments/results/levelset_thetastar_muon_arm/levelset_witness_ema_BEST.npz \\
      --gt-cache experiments/results/mlx_fleet_gt_cache/gt_strided_n200.npz \\
      --num-pairs 200 --verdict-pairs 96 --so-iters 4 \\
      --out-dir experiments/results/witness_per_stage_attribution
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src", REPO / "upstream", REPO / "experiments"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# CANONICAL comma10k class order (CLAUDE.md NON-NEGOTIABLE -- never luma-sort).
CLASS_NAMES = {0: "Road", 1: "Lane", 2: "Undrivable", 3: "Movable", 4: "MyCar"}
N_CLASSES = 5
ADVISORY = "[macOS-numpy advisory . NON-PROMOTABLE]"


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _utc_compact() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


# ---------------------------------------------------------------------------
# checkpoint loading (params = unprefixed keys = EMA shadow; cfg = __-prefixed).
# ---------------------------------------------------------------------------
def load_ckpt(path: Path) -> tuple[dict[str, np.ndarray], dict]:
    z = np.load(path, allow_pickle=False)
    params = {k: np.asarray(z[k], np.float32) for k in z.files if not k.startswith("__")}
    cfg = {k: (z[k].item() if z[k].size == 1 else z[k].tolist()) for k in z.files if k.startswith("__")}
    for req in ("code", "in_proj.weight", "out_sdf.weight", "out_tex.weight", "palette"):
        if req not in params:
            raise ValueError(f"{path} lacks required param {req!r} (NO-FAKE).")
    return params, cfg


def cfg_scalars(cfg: dict, params: dict) -> dict:
    """Parse the render cfg from the npz __cfg_*/__bank_*/__render_hw scalars (deploy-faithful)."""
    rh, rw = cfg.get("__render_hw", [384, 512])
    mbf = float(cfg.get("__cfg_max_bank_freq", -1.0))
    return {
        "render_h": int(rh), "render_w": int(rw),
        "softmax_temp": float(cfg["__cfg_softmax_temp"]),
        "n_hidden": int(cfg["__cfg_n_hidden"]),
        "hidden_dim": int(cfg["__cfg_hidden_dim"]),
        "activation": str(cfg["__cfg_activation"]),
        "chroma": bool(int(cfg["__cfg_chroma"])),
        "wire_w0": float(cfg["__cfg_wire_w0"]), "wire_s0": float(cfg["__cfg_wire_s0"]),
        "hosc_beta": float(cfg["__cfg_hosc_beta"]), "hosc_omega": float(cfg["__cfg_hosc_omega"]),
        "bank_n_scales": int(cfg["__bank_n_scales"]), "bank_n_orient0": int(cfg["__bank_n_orient0"]),
        "bank_f0": float(cfg["__bank_f0"]), "bank_base": float(cfg["__bank_base"]),
        "bank_n_iso": int(cfg["__bank_n_iso"]),
        "max_bank_freq": (None if mbf < 0 else mbf),
        "self_orient": bool(int(cfg.get("__cfg_self_orient", 0))),
        "n_dir_freqs": int(cfg.get("__cfg_n_dir_freqs", 0)),
        "freq_across": float(cfg.get("__cfg_freq_across", 32.0)),
        "freq_along": float(cfg.get("__cfg_freq_along", 4.0)),
        "so_tau": float(cfg.get("__cfg_so_tau", 4.0)),
        "epoch": int(cfg.get("__epoch", -1)),
    }


# ---------------------------------------------------------------------------
# REAL render path (imported canonical functions; deploy fixed-point for self-orient).
# ---------------------------------------------------------------------------
def build_render_context(sc: dict):
    from train_witness_realized_through_R_mlx import _build_render_coords
    from tac.boundary_math.lever_b_levelset_generator import (
        CurveletBankConfig,
        curvelet_directional_B,
        curvelet_feats,
    )

    coords = _build_render_coords(sc["render_h"], sc["render_w"])
    bank = CurveletBankConfig(
        n_scales=sc["bank_n_scales"], n_orient0=sc["bank_n_orient0"],
        f0=sc["bank_f0"], base=sc["bank_base"], n_iso=sc["bank_n_iso"],
    )
    B = curvelet_directional_B(bank, max_freq=sc["max_bank_freq"])
    curv = curvelet_feats(coords, B).astype(np.float32)
    return coords, curv


def render_frame1_argmax(deploy, sc, coords, curv, code_f1, so_iters: int):
    """Deploy-faithful frame1 render (camera uint8) + the per-iter fixed-point count.

    Mirrors the trainer ``recompute_self_orient`` + ``_render_numpy_deploy`` and the byte-close
    inflate self-orient loop EXACTLY: dirf starts at zeros, each iter renders the decoder's OWN
    argmax and recomputes dirf via the CANONICAL ``self_orientation_directional_feats``; early-stop
    when the argmax is unchanged. Returns ``(camera_uint8 (CAM_H,CAM_W,3), n_iters, converged)``.
    """
    from train_witness_realized_through_R_mlx import _torch_R_to_camera_uint8
    from tac.boundary_math.lever_b_generator import self_orientation_directional_feats
    from tac.boundary_math.lever_b_levelset_generator import levelset_rgb_forward_numpy

    rh, rw = sc["render_h"], sc["render_w"]
    fwd_kw = dict(
        n_hidden=sc["n_hidden"], hidden_dim=sc["hidden_dim"], n_classes=N_CLASSES,
        activation=sc["activation"], softmax_temp=sc["softmax_temp"],
        wire_w0=sc["wire_w0"], wire_s0=sc["wire_s0"],
        hosc_beta=sc["hosc_beta"], hosc_omega=sc["hosc_omega"], chroma=sc["chroma"],
    )
    n_iters, converged = 0, True
    if sc["self_orient"] and so_iters > 0:
        dir_w = 4 * sc["n_dir_freqs"]
        dirf = np.zeros((curv.shape[0], dir_w), np.float32)
        prev_am = None
        converged = False
        for it in range(so_iters):
            feats = np.concatenate([curv, dirf], axis=-1)
            _rgb, phi = levelset_rgb_forward_numpy(deploy, feats, code_f1, **fwd_kw)
            am = phi.argmax(-1).reshape(rh, rw).astype(np.int64)
            n_iters = it + 1
            if prev_am is not None and np.array_equal(am, prev_am):
                converged = True
                break
            dirf = self_orientation_directional_feats(
                coords, am, n_freqs=sc["n_dir_freqs"],
                freq_across=sc["freq_across"], freq_along=sc["freq_along"], tau=sc["so_tau"],
            ).astype(np.float32)
            prev_am = am
        feats = np.concatenate([curv, dirf], axis=-1)
    else:
        feats = curv
    rgb, _phi = levelset_rgb_forward_numpy(deploy, feats, code_f1, **fwd_kw)
    cam = _torch_R_to_camera_uint8(rgb.reshape(rh, rw, 3))
    return cam, n_iters, converged


# ---------------------------------------------------------------------------
# REAL frozen CPU-torch SegNet readout (bit-identical argmax to cpu_verdict_d_seg_batch) PLUS the
# per-pixel margin-toward-GT (logit[GT] - max_{k!=GT} logit[k]) used for PRIMED/STUCK detection.
# ---------------------------------------------------------------------------
def segnet_readout_batch(seg_cpu, frames1_uint8: list, gt_argmax_list: list):
    """Returns (argmax (N,h,w) int8, gt_margin (N,h,w) float32).

    argmax mirrors ``cpu_verdict_d_seg_batch`` EXACTLY (preprocess_input -> forward -> argmax(dim=1)
    over the (N,1,H,W,3) last-frame stack). gt_margin = realized SegNet ``logit[GT] - max other``:
    >0 <=> GT wins (correct); <0 magnitude = how far the realized readout is from flipping to GT.
    """
    import torch

    arr = np.stack([np.asarray(f)[None] for f in frames1_uint8], axis=0)  # (N,1,H,W,3)
    xp = torch.from_numpy(arr).permute(0, 1, 4, 2, 3).contiguous().float()  # (N,1,3,H,W)
    with torch.inference_mode():
        seg_in = seg_cpu.preprocess_input(xp)
        logits = seg_cpu(seg_in)  # (N,5,h,w)
    L = logits.cpu().numpy().astype(np.float32)
    am = L.argmax(axis=1).astype(np.int8)  # (N,h,w)
    n, _k, h, w = L.shape
    gt = np.stack([np.asarray(g) for g in gt_argmax_list], axis=0).astype(np.int64)  # (N,h,w)
    ii, jj = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    margins = np.empty((n, h, w), np.float32)
    for t in range(n):
        Lt = L[t]                                  # (5,h,w)
        gtt = gt[t]
        gt_logit = Lt[gtt, ii, jj]                 # (h,w)
        Lmask = Lt.copy()
        Lmask[gtt, ii, jj] = -np.inf
        other_max = Lmask.max(axis=0)
        margins[t] = gt_logit - other_max
    return am, margins


# ---------------------------------------------------------------------------
# per-ckpt pass: render the verdict subset -> realized argmax + gt_margin maps -> per-ckpt npz.
# ---------------------------------------------------------------------------
def process_ckpt(name, path, sc, coords, curv, vpairs, gt, seg_cpu, so_iters, seg_chunk, log):
    from tac.boundary_math.lever_b_levelset_generator import int8_dequant_params

    params, _cfg = load_ckpt(path)
    deploy = int8_dequant_params(params)
    code = deploy["code"]
    rh, rw = sc["render_h"], sc["render_w"]
    V = len(vpairs)
    h, w = gt.lstars[vpairs[0]].shape

    frames = []
    iters = []
    t0 = time.time()
    for k, pi in enumerate(vpairs):
        cam, nit, _conv = render_frame1_argmax(deploy, sc, coords, curv, code[2 * pi + 1], so_iters)
        frames.append(cam)
        iters.append(nit)
        if (k + 1) % 16 == 0:
            log(f"  [{name}] rendered {k + 1}/{V} pairs ({(time.time() - t0):.0f}s)")

    am = np.empty((V, h, w), np.int8)
    margin = np.empty((V, h, w), np.float32)
    for s in range(0, V, seg_chunk):
        e = min(s + seg_chunk, V)
        a, m = segnet_readout_batch(
            seg_cpu, frames[s:e], [gt.lstars[vpairs[i]] for i in range(s, e)]
        )
        am[s:e] = a
        margin[s:e] = m
    log(f"  [{name}] SegNet readout done; total {(time.time() - t0):.0f}s; "
        f"mean fixed-point iters {np.mean(iters):.2f}")
    return {
        "argmax": am, "gt_margin": margin, "iters": np.asarray(iters, np.int32),
        "softmax_temp": sc["softmax_temp"], "epoch": sc["epoch"], "render_hw": (rh, rw),
    }


# ---------------------------------------------------------------------------
# attribution math.
# ---------------------------------------------------------------------------
def gt_subset(gt, vpairs):
    g = np.stack([np.asarray(gt.lstars[pi]).astype(np.int8) for pi in vpairs], axis=0)  # (V,h,w)
    return g


def boundary_band_subset(g_sub: np.ndarray, radius: int = 2):
    from tac.boundary_math.lever_b_levelset_generator import _boundary_band

    return np.stack([_boundary_band(g_sub[i].astype(np.int64), radius=radius) for i in range(g_sub.shape[0])], axis=0)


def per_ckpt_stats(am: np.ndarray, g: np.ndarray, band: np.ndarray) -> dict:
    """am,g (V,h,w); band (V,h,w) bool. Returns realized d_seg + per-class + annulus stats."""
    wrong = am.astype(np.int64) != g.astype(np.int64)  # (V,h,w)
    total = wrong.size
    n_wrong = int(wrong.sum())
    d_seg = n_wrong / total
    # per-CLASS disagreement (fraction of each GT class's pixels mislabeled) + flip-mass share.
    per_class = {}
    flip_mass = {}
    for c in range(N_CLASSES):
        gc = g == c
        gcn = int(gc.sum())
        wc = int((wrong & gc).sum())
        per_class[c] = {"gt_px": gcn, "wrong_px": wc, "disagree": (wc / gcn if gcn else 0.0)}
        flip_mass[c] = (wc / n_wrong if n_wrong else 0.0)
    # annulus localization: fraction of wrong pixels inside the GT boundary band.
    n_wrong_band = int((wrong & band).sum())
    annulus_frac = (n_wrong_band / n_wrong if n_wrong else 0.0)
    return {
        "d_seg": d_seg, "n_wrong": n_wrong, "total_px": total,
        "per_class": per_class, "flip_mass": flip_mass,
        "annulus_frac_of_wrong": annulus_frac,
    }


def transition_stats(amA, mA, amB, mB, g, band, prime_eps: float) -> dict:
    """Per-pixel CORRECTED/REGRESSED/PERSISTENT-WRONG/PERSISTENT-RIGHT + PRIMED/STUCK split.

    PRIMED (subset of persistent-wrong): gt_margin moved toward GT (mB > mA + eps) -- realized
    SegNet readout closing on the GT class (will fall to a later stage). STUCK: margin flat/worse.
    """
    wrongA = amA.astype(np.int64) != g.astype(np.int64)
    wrongB = amB.astype(np.int64) != g.astype(np.int64)
    corrected = wrongA & ~wrongB
    regressed = ~wrongA & wrongB
    persist_wrong = wrongA & wrongB
    persist_right = ~wrongA & ~wrongB
    dmargin = mB - mA  # >0 => moving toward GT (less negative)
    primed = persist_wrong & (dmargin > prime_eps)
    stuck = persist_wrong & ~(dmargin > prime_eps)

    def _by_class(mask):
        return {c: int((mask & (g == c)).sum()) for c in range(N_CLASSES)}

    out = {
        "corrected": int(corrected.sum()), "regressed": int(regressed.sum()),
        "persist_wrong": int(persist_wrong.sum()), "persist_right": int(persist_right.sum()),
        "primed": int(primed.sum()), "stuck": int(stuck.sum()),
        "net_d_seg_delta": (int(wrongB.sum()) - int(wrongA.sum())) / g.size,
        "corrected_by_class": _by_class(corrected),
        "regressed_by_class": _by_class(regressed),
        "persist_wrong_by_class": _by_class(persist_wrong),
        "primed_by_class": _by_class(primed),
        "stuck_by_class": _by_class(stuck),
        "corrected_annulus_frac": (int((corrected & band).sum()) / max(int(corrected.sum()), 1)),
        "persist_wrong_annulus_frac": (int((persist_wrong & band).sum()) / max(int(persist_wrong.sum()), 1)),
        # mean margin movement on the persistent-wrong residual (PRIMED intensity), per class.
        "persist_wrong_mean_dmargin": (float(dmargin[persist_wrong].mean()) if int(persist_wrong.sum()) else 0.0),
        "persist_wrong_mean_dmargin_by_class": {
            c: (float(dmargin[persist_wrong & (g == c)].mean())
                if int((persist_wrong & (g == c)).sum()) else 0.0)
            for c in range(N_CLASSES)
        },
    }
    return out


# ---------------------------------------------------------------------------
# optional PNG overlays (headless; skipped if matplotlib unavailable).
# ---------------------------------------------------------------------------
def maybe_render_pngs(out_dir, names, ck_maps, g, band, log) -> list:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa: BLE001
        log(f"[png] matplotlib unavailable ({e}); skipping overlays.")
        return []
    paths = []
    # aggregate per-pixel error frequency per stage (mean over pairs), on a single representative grid.
    for name in names:
        am = ck_maps[name]["argmax"]
        wrong = (am.astype(np.int64) != g.astype(np.int64)).mean(axis=0)  # (h,w) freq
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].imshow(wrong, cmap="hot", vmin=0, vmax=max(wrong.max(), 1e-6))
        ax[0].set_title(f"{name}: per-pixel wrong-frequency (mean over pairs)")
        ax[0].axis("off")
        bandfreq = band.mean(axis=0)
        ax[1].imshow(bandfreq, cmap="gray")
        ax[1].imshow(wrong, cmap="hot", alpha=0.6, vmin=0, vmax=max(wrong.max(), 1e-6))
        ax[1].set_title(f"{name}: wrong overlaid on GT annulus")
        ax[1].axis("off")
        p = out_dir / f"errmap_{name}.png"
        fig.tight_layout()
        fig.savefig(p, dpi=90)
        plt.close(fig)
        paths.append(str(p))
    log(f"[png] wrote {len(paths)} stage error maps.")
    return paths


# ---------------------------------------------------------------------------
# report.
# ---------------------------------------------------------------------------
def write_report(out_md, names, ck_stats, ck_meta, transitions, png_paths, meta) -> None:
    L = []
    w = L.append
    w(f"# Witness per-stage / per-pixel / per-class d_seg attribution  {ADVISORY}")
    w("")
    w(f"- generated: `{_utc()}`")
    w(f"- verdict pairs: **{meta['verdict_pairs']}** of {meta['num_pairs']} "
      f"(strided `range(0,{meta['num_pairs']},{meta['stride']})[:{meta['verdict_pairs']}]`, "
      "identical across all ckpts)")
    w(f"- render: deploy-faithful fp32 ONE-CODEPATH, int8-dequantized EMA shadow, "
      f"self-orient fixed-point `so_iters={meta['so_iters']}` (byte-close/inflate authority)")
    w(f"- d_seg = realized through-R frozen CPU-torch SegNet argmax disagreement vs GT `lstars` "
      "(NOT a proxy). Pointer 0.19110 UNMOVED; this is advisory.")
    w("- class order (CLAUDE.md): 0=Road 1=Lane 2=Undrivable 3=Movable 4=MyCar")
    w("")
    w("## Stage chain + realized d_seg")
    w("")
    w("| stage | epoch | softmax_temp | mean fp-iters | realized d_seg | n_wrong | annulus% of wrong |")
    w("|---|--:|--:|--:|--:|--:|--:|")
    for n in names:
        s = ck_stats[n]
        m = ck_meta[n]
        w(f"| {n} | {m['epoch']} | {m['softmax_temp']:.4f} | {m['mean_iters']:.2f} | "
          f"**{s['d_seg']:.6f}** | {s['n_wrong']:,} | {100 * s['annulus_frac_of_wrong']:.1f}% |")
    w("")
    w("## Per-class disagreement per stage  (fraction of each GT class's pixels mislabeled)")
    w("")
    hdr = "| stage | " + " | ".join(CLASS_NAMES[c] for c in range(N_CLASSES)) + " |"
    w(hdr)
    w("|---" * (N_CLASSES + 1) + "|")
    for n in names:
        pc = ck_stats[n]["per_class"]
        row = f"| {n} | " + " | ".join(f"{100 * pc[c]['disagree']:.2f}%" for c in range(N_CLASSES)) + " |"
        w(row)
    w("")
    w("### Flip MASS share per stage  (of ALL wrong pixels, fraction whose GT class is c)")
    w("")
    w(hdr)
    w("|---" * (N_CLASSES + 1) + "|")
    for n in names:
        fm = ck_stats[n]["flip_mass"]
        row = f"| {n} | " + " | ".join(f"{100 * fm[c]:.1f}%" for c in range(N_CLASSES)) + " |"
        w(row)
    w("")
    w("## Stage transitions  (per-pixel, summed over the verdict pairs)")
    w("")
    w("| transition | net d_seg Delta | corrected | regressed | persist-wrong | PRIMED | STUCK | "
      "corrected annulus% | persist-wrong annulus% |")
    w("|---|--:|--:|--:|--:|--:|--:|--:|--:|")
    for tr in transitions:
        t = tr["stats"]
        w(f"| {tr['a']}->{tr['b']} | {t['net_d_seg_delta']:+.6f} | {t['corrected']:,} | "
          f"{t['regressed']:,} | {t['persist_wrong']:,} | {t['primed']:,} | {t['stuck']:,} | "
          f"{100 * t['corrected_annulus_frac']:.1f}% | {100 * t['persist_wrong_annulus_frac']:.1f}% |")
    w("")
    w("### Per-transition CORRECTED by class")
    w("")
    w(hdr.replace("stage", "transition"))
    w("|---" * (N_CLASSES + 1) + "|")
    for tr in transitions:
        cbc = tr["stats"]["corrected_by_class"]
        w(f"| {tr['a']}->{tr['b']} | " + " | ".join(f"{cbc[c]:,}" for c in range(N_CLASSES)) + " |")
    w("")
    w("### Per-transition PRIMED by class  (persist-wrong with realized SegNet margin moving toward GT)")
    w("")
    w(hdr.replace("stage", "transition"))
    w("|---" * (N_CLASSES + 1) + "|")
    for tr in transitions:
        pbc = tr["stats"]["primed_by_class"]
        w(f"| {tr['a']}->{tr['b']} | " + " | ".join(f"{pbc[c]:,}" for c in range(N_CLASSES)) + " |")
    w("")
    w("### Per-transition STUCK by class  (persist-wrong, margin flat/worse -> store/deterministic candidates)")
    w("")
    w(hdr.replace("stage", "transition"))
    w("|---" * (N_CLASSES + 1) + "|")
    for tr in transitions:
        sbc = tr["stats"]["stuck_by_class"]
        w(f"| {tr['a']}->{tr['b']} | " + " | ".join(f"{sbc[c]:,}" for c in range(N_CLASSES)) + " |")
    w("")
    if png_paths:
        w("## Error-map overlays")
        w("")
        for p in png_paths:
            w(f"- `{p}`")
        w("")
    # narrative answers + repair recommendations are appended by the caller (data-grounded).
    out_md.write_text("\n".join(L) + "\n" + meta.get("narrative", ""), encoding="utf-8")


# ---------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", action="append", required=True,
                    help="NAME=path/to.npz (order = stage chain). Repeat. e.g. CE=...stageCE_ep299.npz")
    ap.add_argument("--gt-cache", type=Path, required=True)
    ap.add_argument("--num-pairs", type=int, default=200)
    ap.add_argument("--verdict-pairs", type=int, default=96,
                    help="trainer verdict subset size (strided). default 96 (the trainer authority subset).")
    ap.add_argument("--so-iters", type=int, default=4, help="self-orient deploy fixed-point iters (byte-close default 4).")
    ap.add_argument("--max-pairs", type=int, default=0, help="cap verdict pairs (smoke); 0=use --verdict-pairs.")
    ap.add_argument("--seg-chunk", type=int, default=16, help="SegNet batch chunk (memory bound).")
    ap.add_argument("--prime-eps", type=float, default=0.05,
                    help="min realized-SegNet margin-toward-GT movement to call a persist-wrong px PRIMED.")
    ap.add_argument("--band-radius", type=int, default=2, help="GT boundary-band (annulus) dilation radius.")
    ap.add_argument("--threads", type=int, default=4, help="OMP/torch threads (keep modest; don't starve live run).")
    ap.add_argument("--png", action="store_true", help="emit per-stage error-map PNGs (headless matplotlib).")
    ap.add_argument("--out-dir", type=Path, default=REPO / "experiments/results/witness_per_stage_attribution")
    ap.add_argument("--report", type=Path, default=None, help="report .md path (default .omx/research/witness_per_stage_attribution_<UTC>.md)")
    args = ap.parse_args(argv)

    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
    import torch
    torch.set_num_threads(args.threads)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    def log(msg):
        print(f"[{_utc()}] {msg}", flush=True)

    pairs = []
    for spec in args.ckpt:
        if "=" not in spec:
            raise SystemExit(f"--ckpt must be NAME=path; got {spec!r}")
        name, p = spec.split("=", 1)
        pairs.append((name, Path(p)))
    names = [n for n, _ in pairs]
    log(f"stage chain: {names}")

    from train_witness_realized_through_R_mlx import load_gt_from_cache

    log(f"loading GT cache {args.gt_cache} (P={args.num_pairs}) ...")
    gt, seg_cpu, _posenet = load_gt_from_cache(args.gt_cache, args.num_pairs)
    P = gt.n_pairs
    vp_target = args.max_pairs if args.max_pairs else args.verdict_pairs
    stride = max(1, P // max(vp_target, 1))
    vpairs = list(range(0, P, stride))[:vp_target]
    log(f"verdict subset: {len(vpairs)} pairs (stride {stride})")

    g = gt_subset(gt, vpairs)             # (V,h,w) int8
    band = boundary_band_subset(g, radius=args.band_radius)  # (V,h,w) bool
    np.save(out_dir / "_gt_argmax_subset.npy", g)
    np.save(out_dir / "_gt_band_subset.npy", band)

    ck_maps = {}
    ck_meta = {}
    for name, path in pairs:
        cache = out_dir / f"maps_{name}.npz"
        if cache.exists():
            log(f"[{name}] resume: loading cached maps {cache}")
            z = np.load(cache, allow_pickle=False)
            ck_maps[name] = {"argmax": z["argmax"], "gt_margin": z["gt_margin"]}
            ck_meta[name] = {"softmax_temp": float(z["softmax_temp"]), "epoch": int(z["epoch"]),
                             "mean_iters": float(z["mean_iters"])}
            continue
        params, cfg = load_ckpt(path)
        sc = cfg_scalars(cfg, params)
        log(f"[{name}] ep{sc['epoch']} softmax_temp={sc['softmax_temp']:.4f} self_orient={sc['self_orient']} render={sc['render_h']}x{sc['render_w']}")
        coords, curv = build_render_context(sc)
        dir_w_expect = 4 * sc["n_dir_freqs"] if sc["self_orient"] else 0
        if params["in_proj.weight"].shape[1] != curv.shape[1] + dir_w_expect:
            raise ValueError(
                f"[{name}] in_feat {params['in_proj.weight'].shape[1]} != curv {curv.shape[1]} + dir_w "
                f"{dir_w_expect}: bank cfg does not reproduce the trained input width (NO-FAKE).")
        res = process_ckpt(name, path, sc, coords, curv, vpairs, gt, seg_cpu,
                           args.so_iters, args.seg_chunk, log)
        np.savez(out_dir / f".maps_{name}.tmp.npz",
                 argmax=res["argmax"], gt_margin=res["gt_margin"],
                 iters=res["iters"], softmax_temp=res["softmax_temp"], epoch=res["epoch"],
                 mean_iters=float(np.mean(res["iters"])))
        os.replace(out_dir / f".maps_{name}.tmp.npz", cache)
        ck_maps[name] = {"argmax": res["argmax"], "gt_margin": res["gt_margin"]}
        ck_meta[name] = {"softmax_temp": res["softmax_temp"], "epoch": res["epoch"],
                         "mean_iters": float(np.mean(res["iters"]))}

    # per-ckpt + transition stats.
    ck_stats = {n: per_ckpt_stats(ck_maps[n]["argmax"], g, band) for n in names}
    transitions = []
    for a, b in zip(names[:-1], names[1:]):
        t = transition_stats(ck_maps[a]["argmax"], ck_maps[a]["gt_margin"],
                             ck_maps[b]["argmax"], ck_maps[b]["gt_margin"], g, band, args.prime_eps)
        transitions.append({"a": a, "b": b, "stats": t})

    png_paths = maybe_render_pngs(out_dir, names, ck_maps, g, band, log) if args.png else []

    summary = {
        "advisory": ADVISORY, "generated": _utc(), "verdict_pairs": len(vpairs),
        "num_pairs": P, "stride": stride, "so_iters": args.so_iters,
        "ck_stats": {n: ck_stats[n] for n in names},
        "ck_meta": ck_meta,
        "transitions": transitions,
        "png_paths": png_paths,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    log(f"wrote {out_dir / 'summary.json'}")

    report = args.report or (REPO / ".omx/research" / f"witness_per_stage_attribution_{_utc_compact()}.md")
    report.parent.mkdir(parents=True, exist_ok=True)
    meta = {"verdict_pairs": len(vpairs), "num_pairs": P, "stride": stride, "so_iters": args.so_iters,
            "narrative": ""}
    write_report(report, names, ck_stats, ck_meta, transitions, png_paths, meta)
    log(f"wrote {report}")
    # echo key numbers to stdout for the caller.
    print(json.dumps({
        "d_seg_per_stage": {n: ck_stats[n]["d_seg"] for n in names},
        "transitions": [{"t": f"{tr['a']}->{tr['b']}", "net": tr["stats"]["net_d_seg_delta"],
                         "corrected": tr["stats"]["corrected"], "regressed": tr["stats"]["regressed"],
                         "primed": tr["stats"]["primed"], "stuck": tr["stats"]["stuck"]}
                        for tr in transitions],
    }, indent=2, default=float), flush=True)


if __name__ == "__main__":
    main()
