#!/usr/bin/env python3
"""Erasure-timing per-stage attribution probe (#253).

A $0 POST-HOC tool. Given a set of per-stage witness checkpoints (from a #205
level-set run), it turns them into an ERASURE-TIMING CURVE so the surgical-lever
A/B becomes a per-stage attribution DAG.

For EACH checkpoint it measures — REALIZED THROUGH THE REAL R OPERATOR + the
FROZEN CPU-torch SegNet argmax on the exact rendered bytes (NO proxy, NO direct
argmax mirage, NEVER MPS) — three long-tail metrics:

  1. PER-CLASS d_seg -- decompose the d_seg flip mass by the 5 canonical
     comma10k classes (which classes flip).
  2. LANE / MOVABLE RECALL -- for the rare classes Lane(1) + Movable(3):
     recall = (correctly-predicted GT-class pixels) / (GT-class pixels). Low
     recall = ERASED. This is the long-tail metric.
  3. FINEST-SCALE ISLAND BIRTH / SURVIVAL -- how many fine features (lane
     dashes, thin markings, hood outline) are ALIVE (argmax correct AND
     survives R) at this stage. Measured two complementary ways:
       * connected-component islands of the GT rare-class mask (scipy.label),
         sized + margin-weighted (the GT margin field = #141-style saliency);
       * #218 birth-death persistence (h0_superlevel_persistence) of the GT
         margin within the rare-class mask -> survival binned by persistence
         (low-persistence = finest = dashes; error ~ 1/persistence).

The 3-stage BIRTH / SURVIVE-TRAINING / SURVIVE-R pipeline (task #253):
  * BIRTH            = is the fine feature ever argmax-correct (early stage / ep0 seed)?
  * SURVIVE-TRAINING = the erasure-timing curve: survival vs stage/epoch. Alive
                       early then dead = erased DURING training.
  * SURVIVE-R        = realized (post-R SegNet) vs witness-internal (pre-R phi
                       argmax): the gap isolates R+SegNet-induced erasure.

CANONICAL CLASS ORDER (comma10k, MEASURED; a 3x-recurring bug):
  0=Road 1=Lane 2=Undrivable(incl sky) 3=Movable(cars) 4=MyCar(hood).
  It is FORBIDDEN to re-derive by luma-sorting class_values=[41,76,90,124,161].
  The witness out_sdf head + the GT SegNet argmax cache both use THIS order.

AUTHORITY: the numpy fp32 render (the ONE CODEPATH) + CPU-torch SegNet argmax on
the exact R-rendered bytes is the DETERMINISTIC authority (CPU-locked; MLX-GPU is
NOT bit-identical cross-process, so all measurement here is CPU-torch). Every row
is tagged advisory/non-promotable: this is a per-stage ATTRIBUTION probe, not a
frontier claim. The exact contest pointer moves only through upstream/evaluate.py.

REUSE (canonical, do-not-reinvent):
  * tac.boundary_math.lever_b_levelset_generator.levelset_rgb_forward_numpy  (the numpy ONE CODEPATH)
  * tac.boundary_math.lever_b_levelset_generator.{build_coords, curvelet_directional_B, curvelet_feats}
  * tac.local_acceleration.torch_levelset_inflate.dir_feats                  (self-orient fixed point)
  * tools.levelset_byte_close_and_eval.{_load_levelset_ckpt, detect_self_orient, _bank_cfg, CAMERA_H, CAMERA_W}
  * tac.boundary_math.seg_core.load_real_segnet + the batched CPU-torch SegNet argmax verdict
  * tools.birth_death_persistence_dseg.h0_superlevel_persistence            (#218 persistence)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# canonical sys.path setup (mirror tools/levelset_byte_close_and_eval.py) so
# ``import tools.X`` / ``import tac.X`` / upstream ``modules`` resolve when this
# file is run as a script (python tools/erasure_timing_attribution.py).
_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO, _REPO / "src", _REPO / "experiments", _REPO / "upstream"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# --- canonical constants ---------------------------------------------------
# comma10k MEASURED order (see CLAUDE.md "SegNet sees REGIONS"). DO NOT luma-sort.
CANONICAL_CLASS_NAMES: dict[int, str] = {
    0: "Road", 1: "Lane", 2: "Undrivable", 3: "Movable", 4: "MyCar",
}
N_CLASSES = 5
RARE_CLASSES = (1, 3)  # Lane, Movable -- the long-tail (dashes, cars)
SEG_H, SEG_W = 384, 512  # SegNet argmax resolution (evaluate.py contract)

try:  # camera (native) dims -- the R operator upsamples render-res -> camera.
    from tools.levelset_byte_close_and_eval import CAMERA_H, CAMERA_W
except Exception:  # pragma: no cover
    CAMERA_H, CAMERA_W = 874, 1164

_ADVISORY = (
    "advisory_non_promotable: realized-through-R + FROZEN CPU-torch SegNet argmax "
    "(numpy fp32 ONE CODEPATH; NO MPS; NO proxy). Per-stage ATTRIBUTION probe, not a "
    "frontier/score claim. The exact pointer moves only through upstream/evaluate.py."
)


# ---------------------------------------------------------------------------
# provenance
# ---------------------------------------------------------------------------
def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent.parent,
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _sha256_short(path: Path, n: int = 16) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def _parse_stage_epoch(ckpt_path: Path, raw_cfg_epoch: int | None) -> tuple[str, int | None]:
    """Stage tag + epoch from the PRESERVED per-stage filename
    ``levelset_ckpt_<stageTag>_ep<N>.npz`` (trainer _do_checkpoint), else the npz
    ``__epoch`` scalar / the file stem."""
    name = ckpt_path.name
    m = re.match(r"levelset_ckpt_(?P<tag>.+?)_ep(?P<ep>\d+)\.npz$", name)
    if m:
        return m.group("tag"), int(m.group("ep"))
    m2 = re.search(r"_ep(?P<ep>\d+)\.npz$", name)
    ep = int(m2.group("ep")) if m2 else raw_cfg_epoch
    # ema-best / live / other -> use the stem (minus common noise) as the stage label
    stem = ckpt_path.stem
    for pre in ("levelset_witness_", "levelset_"):
        if stem.startswith(pre):
            stem = stem[len(pre):]
    return stem, ep


# ---------------------------------------------------------------------------
# NO-MPS guard (authority discipline)
# ---------------------------------------------------------------------------
def _assert_segnet_cpu(seg_cpu: Any) -> None:
    import torch  # noqa

    for p in seg_cpu.parameters():
        if p.device.type != "cpu":
            raise RuntimeError(
                f"SegNet param on device {p.device!r} -- authority requires CPU-torch "
                "(MPS corrupts SegNet 2x / score 2.5x; NEVER a score/attribution authority)."
            )
        break


# ---------------------------------------------------------------------------
# R operator: render-grid RGB (rh,rw) -> camera uint8 (ch,cw). Op-for-op mirror
# of tools.levelset_byte_close_and_eval._R (the shipped inflate.py numpy oracle):
# bicubic upsample to camera, round (STE at deploy), clamp [0,255], uint8. The
# SegNet preprocess_input then bilinears camera->384 (contest-exact; the GT
# cache lstars were produced the same way from the camera-native GT frame1).
# ---------------------------------------------------------------------------
def _R_to_camera(rgb_pk3: np.ndarray, rh: int, rw: int, ch: int, cw: int) -> np.ndarray:
    import torch

    x = torch.from_numpy(
        np.ascontiguousarray(np.asarray(rgb_pk3, np.float32).reshape(rh, rw, 3))
    ).permute(2, 0, 1)[None].float()
    with torch.inference_mode():
        up = torch.nn.functional.interpolate(x, size=(ch, cw), mode="bicubic", align_corners=False)
        up = torch.clamp(torch.round(up), 0.0, 255.0)
    return up[0].permute(1, 2, 0).contiguous().numpy().astype(np.uint8)


# ---------------------------------------------------------------------------
# witness frame1 render (numpy ONE CODEPATH) -> (camera uint8 render, witness
# out_sdf argmax at render res). Reuses levelset_rgb_forward_numpy + the
# self-orient fixed point EXACTLY as the byte-close / inflate decode does.
# ---------------------------------------------------------------------------
def _fwd_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    return dict(
        n_hidden=int(cfg["n_hidden"]), hidden_dim=int(cfg["hidden_dim"]),
        n_classes=int(cfg["n_classes"]), activation=str(cfg["activation"]),
        softmax_temp=float(cfg["softmax_temp"]), wire_w0=float(cfg["wire_w0"]),
        wire_s0=float(cfg["wire_s0"]), hosc_beta=float(cfg["hosc_beta"]),
        hosc_omega=float(cfg["hosc_omega"]), chroma=bool(cfg["chroma"]),
    )


def _render_frame1(
    params: dict[str, np.ndarray], cfg: dict[str, Any], so: dict[str, Any],
    coords: np.ndarray, curv: np.ndarray, code_f1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (camera_uint8 frame1 render (ch,cw,3), render-res uint8 (rh,rw,3),
    witness out_sdf internal-partition argmax (rh,rw)).

    * camera render -> post-R realized d_seg (the authority path; SegNet.preprocess
      bilinears camera->384, contest-exact).
    * render-res uint8 -> the PRE-R frame (SURVIVE-R: SegNet on the raw render with NO
      camera bicubic+uint8 roundtrip; realized - pre_R isolates the R erasure).
    * witness out_sdf argmax -> the witness's INTERNAL SDF partition (a diagnostic; for
      a store_nothing witness the realized d_seg rides the RGB->SegNet path, so this
      internal argmax need NOT match GT -- it is NOT a pre-R SegNet pass)."""
    from tac.boundary_math.lever_b_levelset_generator import levelset_rgb_forward_numpy
    from tac.local_acceleration.torch_levelset_inflate import dir_feats

    rh, rw = int(cfg["render_h"]), int(cfg["render_w"])
    fkw = _fwd_kwargs(cfg)
    if so["self_orient"]:
        ndf = int(so["n_dir_freqs"]); fa = float(so["freq_along"])
        fc = float(so["freq_across"]); tau = float(so["tau"]); iters = int(so["iters"])
        dirf = np.zeros((curv.shape[0], 4 * ndf), np.float32)
        prev = None
        for _ in range(max(iters, 1)):
            feats = np.concatenate([curv, dirf], axis=-1)
            _rgb, phi = levelset_rgb_forward_numpy(params, feats, code_f1, **fkw)
            am = phi.argmax(-1).reshape(rh, rw).astype(np.int64)
            if prev is not None and np.array_equal(am, prev):
                break  # argmax fixed point -> remaining iters are no-ops
            dirf = dir_feats(coords, am, ndf, fa, fc, tau)
            prev = am
        feats = np.concatenate([curv, dirf], axis=-1)
    else:
        feats = curv
    rgb, phi = levelset_rgb_forward_numpy(params, feats, code_f1, **fkw)
    witness_argmax = phi.argmax(-1).reshape(rh, rw).astype(np.int64)
    cam = _R_to_camera(rgb, rh, rw, CAMERA_H, CAMERA_W)
    render_uint8 = np.clip(np.round(np.asarray(rgb, np.float32)), 0, 255).astype(np.uint8).reshape(rh, rw, 3)
    return cam, render_uint8, witness_argmax


# ---------------------------------------------------------------------------
# batched CPU-torch SegNet argmax on camera frames -> realized argmax (384x512).
# CHUNKED (#205 OOM law: any full-P batched scorer forward MUST chunk). Mirrors
# experiments/...cpu_verdict_d_seg_batch (eval-mode BatchNorm running stats ->
# batch-size independent -> bit-identical to frame-by-frame). NEVER MPS.
# ---------------------------------------------------------------------------
def _segnet_argmax_batch(seg_cpu: Any, camera_frames: list[np.ndarray], chunk: int) -> list[np.ndarray]:
    import torch

    out: list[np.ndarray] = []
    chunk = max(int(chunk), 1)
    for i in range(0, len(camera_frames), chunk):
        sub = camera_frames[i:i + chunk]
        arr = np.stack([np.asarray(f, np.uint8)[None] for f in sub], axis=0)  # (n,1,H,W,3)
        xp = torch.from_numpy(arr).permute(0, 1, 4, 2, 3).contiguous().float()  # (n,1,3,H,W)
        with torch.inference_mode():
            seg_in = seg_cpu.preprocess_input(xp)
            am = seg_cpu(seg_in).argmax(dim=1).cpu().numpy().astype(np.int64)  # (n,384,512)
        out.extend([am[j] for j in range(am.shape[0])])
    return out


# ---------------------------------------------------------------------------
# metric accumulators (over all pairs, realized argmax vs GT argmax at 384x512)
# ---------------------------------------------------------------------------
class _Accum:
    def __init__(self) -> None:
        self.total_px = 0
        self.total_flips = 0
        self.per_class_flip = np.zeros(N_CLASSES, np.int64)     # flips where gt==c
        self.recall_support = np.zeros(N_CLASSES, np.int64)     # count(gt==c)
        self.recall_correct = np.zeros(N_CLASSES, np.int64)     # count(gt==c & realized==c)
        self.witness_sdf_flips = 0                               # witness out_sdf internal-partition flips (diagnostic)
        self.pre_r_flips = 0                                     # SURVIVE-R: SegNet on raw render (no R roundtrip)
        self.pre_r_measured = False
        # cc islands (per rare class): [n_islands, n_fine, fine_alive, sum_fine_recall]
        self.cc: dict[int, np.ndarray] = {c: np.zeros(4, np.float64) for c in RARE_CLASSES}
        # persistence features: collected (pers, alive_px, npx) rows across pairs
        self.pers_rows: list[tuple[float, float, float]] = []

    def update_seg(self, realized: np.ndarray, witness_am: np.ndarray, gt: np.ndarray,
                   pre_r: np.ndarray | None = None) -> None:
        flips = realized != gt
        self.total_px += int(gt.size)
        self.total_flips += int(flips.sum())
        for c in range(N_CLASSES):
            gc = gt == c
            self.per_class_flip[c] += int(np.count_nonzero(flips & gc))
            self.recall_support[c] += int(np.count_nonzero(gc))
            self.recall_correct[c] += int(np.count_nonzero(gc & (realized == c)))
        # witness INTERNAL out_sdf partition (diagnostic, NOT a pre-R SegNet pass): compare if same res.
        if witness_am.shape == gt.shape:
            self.witness_sdf_flips += int(np.count_nonzero(witness_am != gt))
        # SURVIVE-R: SegNet argmax on the raw render (no camera bicubic+uint8 roundtrip).
        if pre_r is not None:
            self.pre_r_measured = True
            self.pre_r_flips += int(np.count_nonzero(pre_r != gt))

    def update_islands_cc(self, realized: np.ndarray, gt: np.ndarray, fine_thresh: int) -> None:
        from scipy import ndimage

        struct = np.ones((3, 3), np.int64)  # 8-connectivity
        for c in RARE_CLASSES:
            mask = gt == c
            if not mask.any():
                continue
            lbl, n = ndimage.label(mask, structure=struct)
            if n == 0:
                continue
            sizes = ndimage.sum(np.ones_like(lbl), lbl, index=np.arange(1, n + 1))
            correct = (realized == c) & mask
            corr_per = ndimage.sum(correct.astype(np.int64), lbl, index=np.arange(1, n + 1))
            recall_per = corr_per / np.maximum(sizes, 1.0)
            fine = sizes <= float(fine_thresh)
            acc = self.cc[c]
            acc[0] += float(n)
            acc[1] += float(np.count_nonzero(fine))
            acc[2] += float(np.count_nonzero(fine & (recall_per >= 0.5)))
            acc[3] += float(recall_per[fine].sum())

    def update_islands_persistence(self, realized: np.ndarray, gt: np.ndarray, gt_margin: np.ndarray) -> None:
        from tools.birth_death_persistence_dseg import h0_superlevel_persistence

        mask = np.isin(gt, RARE_CLASSES)  # long-tail rare-class support
        if not mask.any():
            return
        pix_feat, _birth, _death, pers = h0_superlevel_persistence(
            np.asarray(gt_margin, np.float64), mask
        )
        if pers.size == 0:
            return
        correct_flat = (realized == gt).ravel()
        pf = pix_feat  # (H*W,) feature id per masked pixel, -1 elsewhere
        # per feature: alive_px / npx
        for fid in range(pers.size):
            sel = pf == fid
            npx = int(np.count_nonzero(sel))
            if npx == 0:
                continue
            alive = float(np.count_nonzero(correct_flat[sel]))
            self.pers_rows.append((float(pers[fid]), alive, float(npx)))

    # --- finalization ---
    def finalize(self, n_pairs: int) -> dict[str, Any]:
        d_seg = self.total_flips / max(self.total_px, 1)
        d_seg_sdf = self.witness_sdf_flips / max(self.total_px, 1)
        d_seg_pre_r = (self.pre_r_flips / max(self.total_px, 1)) if self.pre_r_measured else None
        per_class = {}
        for c in range(N_CLASSES):
            cnt = int(self.per_class_flip[c])
            per_class[CANONICAL_CLASS_NAMES[c]] = {
                "class_idx": c,
                "flip_count": cnt,
                "frac_of_total_pixels": cnt / max(self.total_px, 1),
                "frac_of_dseg": (cnt / self.total_flips) if self.total_flips else 0.0,
            }
        recall = {}
        for c in range(N_CLASSES):
            sup = int(self.recall_support[c]); cor = int(self.recall_correct[c])
            recall[CANONICAL_CLASS_NAMES[c]] = {
                "class_idx": c, "support_px": sup, "correct_px": cor,
                "recall": (cor / sup) if sup else None,
            }
        cc = {}
        for c in RARE_CLASSES:
            n_isl, n_fine, fine_alive, sum_recall = self.cc[c]
            cc[CANONICAL_CLASS_NAMES[c]] = {
                "class_idx": c,
                "n_islands": int(n_isl),
                "n_fine_islands": int(n_fine),
                "fine_island_survival_rate": (fine_alive / n_fine) if n_fine else None,
                "mean_fine_island_recall": (sum_recall / n_fine) if n_fine else None,
            }
        pers = self._finalize_persistence()
        out: dict[str, Any] = {
            "n_pairs": int(n_pairs),
            "d_seg_realized": float(d_seg),
            "d_seg_witness_sdf_internal": float(d_seg_sdf),
            "witness_sdf_minus_realized": float(d_seg_sdf - d_seg),
            "d_seg_pre_R": (float(d_seg_pre_r) if d_seg_pre_r is not None else None),
            "r_roundtrip_erasure_gap": (float(d_seg - d_seg_pre_r) if d_seg_pre_r is not None else None),
            "per_class_flip_mass": per_class,
            "recall": recall,
            "island_cc": cc,
            "island_persistence": pers,
        }
        return out

    def _finalize_persistence(self) -> dict[str, Any] | None:
        if not self.pers_rows:
            return None
        rows = np.asarray(self.pers_rows, np.float64)  # (F, 3): pers, alive_px, npx
        p = rows[:, 0]
        # tercile bins by persistence (low=finest=dashes -> high=prominent)
        qs = np.quantile(p, [1.0 / 3.0, 2.0 / 3.0]) if p.size >= 3 else np.array([p.min(), p.max()])
        edges = [(-np.inf, qs[0]), (qs[0], qs[1]), (qs[1], np.inf)]
        labels = ["low_pers_finest", "mid_pers", "high_pers_prominent"]
        bins = []
        for (lo, hi), lab in zip(edges, labels):
            sel = (p > lo) & (p <= hi) if lo != -np.inf else (p <= hi)
            npx = rows[sel, 2].sum(); alive = rows[sel, 1].sum()
            bins.append({
                "bin": lab, "pers_lo": (None if lo == -np.inf else float(lo)),
                "pers_hi": (None if hi == np.inf else float(hi)),
                "n_features": int(np.count_nonzero(sel)),
                "survival": (float(alive / npx) if npx else None),
            })
        low = bins[0]
        return {
            "bins": bins,
            "low_pers_finest_survival": low["survival"],
            "n_features_total": int(rows.shape[0]),
            "note": "survival = P(realized argmax == GT argmax) on the feature's pixels; "
                    "low-persistence (finest = dashes) survival is the erasure headline.",
        }


# ---------------------------------------------------------------------------
# GT cache load (ONLY lstars + margins; skip pose/frames -> light + focused)
# ---------------------------------------------------------------------------
def _load_gt(gt_cache: Path, num_pairs: int) -> tuple[np.ndarray, np.ndarray, int]:
    z = np.load(gt_cache, allow_pickle=True)
    cached = int(z["n_pairs"])
    P = min(int(num_pairs), cached)
    lstars = np.asarray(z["lstars"])[:P]   # (P,384,512) int64  -- read member ONCE
    margins = np.asarray(z["margins"])[:P]  # (P,384,512) float32
    if lstars.shape[1:] != (SEG_H, SEG_W):
        raise ValueError(f"gt lstars shape {lstars.shape} != (P,{SEG_H},{SEG_W}) (NO-FAKE).")
    return lstars, margins, P


# ---------------------------------------------------------------------------
# one checkpoint -> metric row (the erasure-timing point)
# ---------------------------------------------------------------------------
def measure_checkpoint(
    ckpt_path: Path, lstars: np.ndarray, margins: np.ndarray, seg_cpu: Any,
    *, so_overrides: dict[str, Any], verdict_batch: int, fine_thresh: int,
    islands: str, num_pairs: int, measure_r_survival: bool = False,
) -> dict[str, Any]:
    from tac.boundary_math.lever_b_levelset_generator import (
        build_coords, curvelet_directional_B, curvelet_feats,
    )
    from tools.levelset_byte_close_and_eval import (
        _bank_cfg, _load_levelset_ckpt, detect_self_orient,
    )

    t0 = time.time()
    params, cfg = _load_levelset_ckpt(ckpt_path.parent, ckpt_path.name)
    so = detect_self_orient(cfg, so_overrides)
    P = min(int(num_pairs), int(cfg["n_pairs"]), int(lstars.shape[0]))

    rh, rw = int(cfg["render_h"]), int(cfg["render_w"])
    coords = build_coords(rh, rw)
    B = curvelet_directional_B(_bank_cfg(cfg), max_freq=cfg["max_bank_freq"])
    curv = curvelet_feats(coords, B)
    # NO-FAKE: the FREE bank must reproduce the trained in_proj input width.
    exp_in = int(so["curvelet_feat_width"]) + int(so.get("dir_w", 0))
    if int(cfg["in_feat"]) != exp_in:
        raise ValueError(
            f"in_feat {cfg['in_feat']} != curv({so['curvelet_feat_width']}) + dir({so.get('dir_w', 0)}) "
            f"= {exp_in}: the bank cfg does not regenerate the trained weights (NO-FAKE)."
        )
    if curv.shape[1] != int(so["curvelet_feat_width"]):
        raise ValueError(f"curvelet feat width {curv.shape[1]} != detected {so['curvelet_feat_width']} (NO-FAKE).")

    code = np.asarray(params["code"], np.float32)
    acc = _Accum()
    # pipeline in chunks of verdict_batch: render frame1 -> camera uint8, batch-SegNet, accumulate, discard.
    chunk = max(int(verdict_batch), 1)
    for i in range(0, P, chunk):
        idxs = list(range(i, min(i + chunk, P)))
        cams: list[np.ndarray] = []
        renders: list[np.ndarray] = []
        wit_ams: list[np.ndarray] = []
        for pi in idxs:
            cam, render_u8, wit = _render_frame1(params, cfg, so, coords, curv, code[2 * pi + 1])
            cams.append(cam)
            renders.append(render_u8)
            wit_ams.append(wit)
        realized = _segnet_argmax_batch(seg_cpu, cams, chunk)
        pre_r = _segnet_argmax_batch(seg_cpu, renders, chunk) if measure_r_survival else None
        for k, pi in enumerate(idxs):
            gt = lstars[pi]
            acc.update_seg(realized[k], wit_ams[k], gt, pre_r=(pre_r[k] if pre_r is not None else None))
            if islands in ("cc", "both"):
                acc.update_islands_cc(realized[k], gt, fine_thresh)
            if islands in ("persistence", "both"):
                acc.update_islands_persistence(realized[k], gt, margins[pi])
        del cams, renders, wit_ams, realized, pre_r

    row = acc.finalize(P)
    raw_epoch = None
    try:
        z = np.load(ckpt_path, allow_pickle=False)
        if "__epoch" in z.files:
            raw_epoch = int(z["__epoch"])
    except Exception:
        pass
    stage_tag, epoch = _parse_stage_epoch(ckpt_path, raw_epoch)
    row.update({
        "checkpoint": str(ckpt_path),
        "checkpoint_sha256_16": _sha256_short(ckpt_path),
        "stage_tag": stage_tag,
        "epoch": epoch,
        "self_orient": bool(so["self_orient"]),
        "render_hw": [rh, rw],
        "wall_s": round(time.time() - t0, 2),
    })
    return row


# ---------------------------------------------------------------------------
# text summary (the "simple text/plot summary")
# ---------------------------------------------------------------------------
def _fmt(x: Any, w: int = 9, p: int = 5) -> str:
    if x is None:
        return "-".rjust(w)
    return f"{x:.{p}f}".rjust(w)


def render_text_curve(rows: list[dict[str, Any]]) -> str:
    lines = []
    lines.append("ERASURE-TIMING CURVE (realized-through-R + CPU-torch SegNet; advisory/non-promotable)")
    lines.append("class order: 0=Road 1=Lane 2=Undrivable 3=Movable 4=MyCar  (comma10k MEASURED; NOT luma-sort)")
    hdr = f"{'stage':<18}{'epoch':>7}{'d_seg':>10}{'preR_dseg':>11}{'R_erase':>9}" \
          f"{'Lane_rec':>10}{'Mov_rec':>9}{'fineLaneSurv':>13}{'lowPersSurv':>12}"
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for r in rows:
        lane_rec = r["recall"]["Lane"]["recall"]
        mov_rec = r["recall"]["Movable"]["recall"]
        fine_lane = r["island_cc"].get("Lane", {}).get("fine_island_survival_rate") if r.get("island_cc") else None
        low_pers = (r["island_persistence"] or {}).get("low_pers_finest_survival") if r.get("island_persistence") else None
        lines.append(
            f"{str(r['stage_tag'])[:18]:<18}{(r['epoch'] if r['epoch'] is not None else -1):>7}"
            f"{_fmt(r['d_seg_realized'])}{_fmt(r['d_seg_pre_R'],11)}{_fmt(r['r_roundtrip_erasure_gap'])}"
            f"{_fmt(lane_rec,10)}{_fmt(mov_rec)}{_fmt(fine_lane,13)}{_fmt(low_pers,12)}"
        )
    lines.append("")
    lines.append("(preR_dseg / R_erase are '-' unless --measure-r-survival; d_seg_witness_sdf_internal in JSON is a diagnostic, NOT pre-R)")
    lines.append("per-class d_seg flip-mass fraction (of total d_seg) @ last checkpoint:")
    if rows:
        pc = rows[-1]["per_class_flip_mass"]
        lines.append("  " + "  ".join(f"{n}={pc[n]['frac_of_dseg']:.3f}" for n in CANONICAL_CLASS_NAMES.values()))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# compare mode: two runs -> per-lever attribution diff (which STAGE moves)
# ---------------------------------------------------------------------------
def compare_curves(run_a: list[dict[str, Any]], run_b: list[dict[str, Any]],
                   label_a: str, label_b: str) -> dict[str, Any]:
    def _key(r: dict[str, Any]) -> str:
        return str(r["stage_tag"])

    a_by = {_key(r): r for r in run_a}
    b_by = {_key(r): r for r in run_b}
    stages = [s for s in a_by if s in b_by]
    diffs = []
    for s in stages:
        ra, rb = a_by[s], b_by[s]
        def d(path):
            va, vb = ra, rb
            for p in path:
                va = (va or {}).get(p) if isinstance(va, dict) else None
                vb = (vb or {}).get(p) if isinstance(vb, dict) else None
            if va is None or vb is None:
                return None
            return float(vb - va)
        diffs.append({
            "stage_tag": s,
            "epoch_a": ra.get("epoch"), "epoch_b": rb.get("epoch"),
            "delta_d_seg_realized": d(["d_seg_realized"]),
            "delta_lane_recall": d(["recall", "Lane", "recall"]),
            "delta_movable_recall": d(["recall", "Movable", "recall"]),
            "delta_fine_lane_survival": d(["island_cc", "Lane", "fine_island_survival_rate"]),
            "delta_low_pers_survival": d(["island_persistence", "low_pers_finest_survival"]),
        })
    return {
        "mode": "compare",
        "run_a": label_a, "run_b": label_b,
        "note": "delta = B - A per matched stage_tag. Which STAGE a lever moves d_seg / "
                "lane-recall / island-survival is the per-lever attribution (BIRTH via ep0 "
                "seed / SURVIVE-TRAINING via persistence / SURVIVE-R via the R gap).",
        "per_stage": diffs,
    }


# ---------------------------------------------------------------------------
# checkpoint discovery
# ---------------------------------------------------------------------------
def _discover_checkpoints(args: argparse.Namespace, run_dir: Path | None,
                          explicit: list[str] | None) -> list[Path]:
    if explicit:
        return [Path(p) for p in explicit]
    if run_dir is None:
        return []
    found = sorted(run_dir.glob(args.stage_glob))
    if not found and args.fallback_ema:
        for c in ("levelset_witness_ema_BEST.npz", "levelset_witness_ema_mlx.npz"):
            p = run_dir / c
            if p.exists():
                found.append(p)
    return found


def _measure_run(ckpts: list[Path], lstars, margins, seg_cpu, args) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cp in ckpts:
        if not cp.exists():
            raise FileNotFoundError(f"checkpoint not found: {cp} (NO-FAKE: refuse to fabricate).")
        print(f"[erasure] measuring {cp.name} ...", flush=True)
        row = measure_checkpoint(
            cp, lstars, margins, seg_cpu,
            so_overrides={"freq_across": args.so_freq_across, "freq_along": args.so_freq_along,
                          "tau": args.so_tau, "iters": args.so_iters},
            verdict_batch=args.verdict_batch, fine_thresh=args.fine_size_thresh,
            islands=args.islands, num_pairs=args.num_pairs,
            measure_r_survival=args.measure_r_survival,
        )
        rows.append(row)
        print(f"[erasure]   d_seg={row['d_seg_realized']:.5f}  Lane_recall="
              f"{row['recall']['Lane']['recall']}  ({row['wall_s']}s)", flush=True)
    # sort by epoch if available, else keep discovery order
    if all(r["epoch"] is not None for r in rows):
        rows.sort(key=lambda r: r["epoch"])
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=str, default=None,
                    help="level-set run dir containing per-stage checkpoints.")
    ap.add_argument("--checkpoints", nargs="*", default=None,
                    help="explicit list of checkpoint .npz paths (overrides --run-dir glob).")
    ap.add_argument("--stage-glob", type=str, default="levelset_ckpt_*.npz",
                    help="glob for per-stage checkpoints inside --run-dir.")
    ap.add_argument("--fallback-ema", action="store_true",
                    help="if no stage ckpts match, fall back to the EMA BEST/mlx checkpoint.")
    ap.add_argument("--gt-cache", type=str,
                    default="experiments/results/mlx_fleet_gt_cache/gt_n96.npz")
    ap.add_argument("--num-pairs", type=int, default=48,
                    help="pairs to measure (SMALL for validation; full n600 later).")
    ap.add_argument("--verdict-batch", type=int, default=16,
                    help="SegNet forward chunk (#205 OOM law: chunk full-P scorer forwards).")
    ap.add_argument("--islands", choices=["cc", "persistence", "both", "none"], default="both")
    ap.add_argument("--fine-size-thresh", type=int, default=32,
                    help="connected-component island px-size <= this = finest-scale (dashes/thin).")
    ap.add_argument("--measure-r-survival", action="store_true",
                    help="SURVIVE-R: also SegNet the raw render (pre-R) -> r_roundtrip_erasure_gap "
                         "= realized - pre_R (2nd SegNet forward per pair; opt-in).")
    # self-orient decode params (NOT persisted by the trainer; #202 CLI defaults == trainer defaults)
    ap.add_argument("--so-freq-across", type=float, default=32.0)
    ap.add_argument("--so-freq-along", type=float, default=4.0)
    ap.add_argument("--so-tau", type=float, default=4.0)
    ap.add_argument("--so-iters", type=int, default=4)
    ap.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"), default=None,
                    help="two run dirs: emit the per-lever attribution DIFF of their erasure curves.")
    ap.add_argument("--out", type=str, default=None, help="output JSON path.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    import torch
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass

    gt_cache = Path(args.gt_cache)
    if not gt_cache.exists():
        raise FileNotFoundError(f"gt cache not found: {gt_cache}")
    lstars, margins, P = _load_gt(gt_cache, args.num_pairs)
    args.num_pairs = P

    from tac.boundary_math.seg_core import load_real_segnet
    seg_cpu = load_real_segnet("cpu")
    _assert_segnet_cpu(seg_cpu)

    prov = {
        "git_hash": _git_hash(),
        "gt_cache": str(gt_cache),
        "gt_cache_sha256_16": _sha256_short(gt_cache),
        "n_pairs_measured": P,
        "seed": args.seed,
        "class_order": CANONICAL_CLASS_NAMES,
        "so_overrides": {"freq_across": args.so_freq_across, "freq_along": args.so_freq_along,
                         "tau": args.so_tau, "iters": args.so_iters},
        "authority": _ADVISORY,
        "tool": "tools/erasure_timing_attribution.py (#253)",
    }

    if args.compare:
        run_a, run_b = Path(args.compare[0]), Path(args.compare[1])
        ck_a = _discover_checkpoints(args, run_a, None)
        ck_b = _discover_checkpoints(args, run_b, None)
        if not ck_a or not ck_b:
            raise FileNotFoundError("--compare needs stage checkpoints in BOTH run dirs.")
        rows_a = _measure_run(ck_a, lstars, margins, seg_cpu, args)
        rows_b = _measure_run(ck_b, lstars, margins, seg_cpu, args)
        diff = compare_curves(rows_a, rows_b, str(run_a), str(run_b))
        result = {"provenance": prov, "compare": diff,
                  "run_a_rows": rows_a, "run_b_rows": rows_b}
        print("\n" + json.dumps(diff, indent=2))
    else:
        run_dir = Path(args.run_dir) if args.run_dir else None
        ckpts = _discover_checkpoints(args, run_dir, args.checkpoints)
        if not ckpts:
            raise FileNotFoundError(
                "no checkpoints found; pass --checkpoints <a.npz ...> or --run-dir <dir> "
                "(with --stage-glob / --fallback-ema).")
        rows = _measure_run(ckpts, lstars, margins, seg_cpu, args)
        result = {"provenance": prov, "erasure_curve": rows}
        print("\n" + render_text_curve(rows))

    if args.out:
        outp = Path(args.out)
        if str(outp).startswith("/tmp") or "/tmp/" in str(outp):
            raise ValueError("refuse /tmp output (transient-evidence trap); use experiments/results or .omx/.")
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(result, indent=2))
        print(f"\n[erasure] wrote {outp}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
