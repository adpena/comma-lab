# SPDX-License-Identifier: MIT
"""ddm_qa92 — THE Lane-carrier RECEIVER DISCRIMINATOR (gc12 rung 0, task #801).

The one demanded measurement of the gc12 wall-branch birth-completion ladder
(`.omx/research/ddm_gc12_wall_branch_convocation_20260731.md`, §4).

QUESTION: does painting the MISSING (erased) super-nucleus Lane structure ONTO the
textured control_tail render actually flip the frozen SegNet back to Lane?  fp1
measured ONLY the ALL-FLAT paint receiver floor (0.008305).  The localized
paint-on-texture case (a real carrier compositing INTO the rendered RGB) is the
carrier family's live-or-die quantity and is GENUINELY UNPREDICTED (Tishby/Atick
predict HIGH recovery; Daubechies predicts partial + AA-sensitive; QA91 bounds the
pool <= ~0.134 S).  This is the genuine discriminator that earns the free scorer slot.

METHOD (n600; frozen CPU-torch SegNet = authority, NEVER MPS):
  BASE   : render control_tail ep499 frame -> deploy R (bicubic up to CAMERA -> uint8)
           -> SegNet -> base realized argmax + base d_seg.  Erased super-nucleus Lane
           components are labeled from THIS pass (scipy 8-conn on GT lstars vs base
           realized; folds fp1's deferred per-component erasure item).
  TIER-1 : composite GT camera RGB (gt_f1) at the erased-component support (+1px rim)
           onto the CAMERA-resolution base render with an ANTI-ALIASED alpha
           (Daubechies binding requirement: composite at camera res PRE-R with AA
           edges, else tier-1 measures aliasing not the receiver) -> uint8 -> SegNet.
           = the Wyner-Ziv ORACLE upper bound (what a perfect textured carrier recovers).
  TIER-2 : same support/AA, filled with fp1's SOLVED Lane prototype colour [77,87,119]
           (what a 1-2KB parametric flat-stroke carrier would paint) -> uint8 -> SegNet.

OUTPUTS (all [macOS-CPU advisory], score_claim=false; additive-S is the sole authority):
  P  = above-nucleus erased-Lane pool remaining, S-units (base flip-mass in target set)
  O  = oracle recovery FRACTION of that pool (target flip-mass) ; F = flat recovery fraction
  collateral (off-target flips the composite introduces) + JOINT dS per tier + per-class dS
  P*O = target-only recovered S-units (the Contrarian rung-2 skip bound).

FALSIFIERS (pre-registered, gc12 §4; route rung 2 mechanically):
  O < 0.25            -> carrier family (b) CLOSES, verdict_scope FORMULATION.
  P*O < 0.05 S        -> rung 2 SKIPS to burn-4 (Contrarian bound).
  F >= 0.7*O          -> rung 2 fires (b) flat 1-2KB carrier suffices.
  O>=0.25 ^ F<0.7*O   -> rung 2 fires (e1) textured / solve-seeded births.

Pointer honesty: 0.1910828242 [contest-CPU] UNMOVED.  Seeded + resumable + atomic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import numpy.lib.format as _npfmt

REPO = Path("/Users/adpena/Projects/pact")
DEFAULT_CKPT = (
    "/Volumes/VertigoDataTier/pact/ddm_pa1r_20260730/control_tail/"
    "checkpoints/stage_seg_trunk_tau_final.npz"
)
DEFAULT_GT_CACHE = "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
DEFAULT_OUT = "/Volumes/VertigoDataTier/pact/ddm_qa92_20260731"
SEG_H, SEG_W = 384, 512
CAMERA_H, CAMERA_W = 874, 1164
N_CLASSES = 5
LANE_CLASS = 1  # comma10k canonical order [Road, Lane, Undrivable, Movable, MyCar]
CLASS_ORDER = "[Road, Lane, Undrivable, Movable, MyCar]"
# fp1 SOLVED margin-optimal Lane prototype colour (prototypes.npz proto_solved[LANE]).
PROTO_LANE_SOLVED = np.array([77.43, 86.71, 118.53], dtype=np.float32)
NUCLEUS_PX = 5          # QA91 / #315 nucleus threshold (super-nucleus = >5px)
ERASED_RECOVER_THRESH = 0.5  # component erased in base iff <50% of its GT-Lane px classified Lane


# --------------------------------------------------------------------------- io
def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(payload)
    os.replace(tmp, path)


def _atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as f:
        np.savez(f, **arrays)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# -------------------------------------------------------------------- gt loader
def load_gt(gt_cache: Path, num_pairs: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (lstars (P,384,512) int64, gt_f1 (P,874,1164,3) uint8) from the gt cache."""
    z = zipfile.ZipFile(gt_cache)
    with z.open("lstars.npy") as f:
        lst = _npfmt.read_array(f)[:num_pairs].astype(np.int64)
    with z.open("gt_f1.npy") as f:
        f1 = _npfmt.read_array(f)[:num_pairs]
    return np.ascontiguousarray(lst), np.ascontiguousarray(f1)


# ------------------------------------------------------------- R + composite ops
def _torch_bicubic_up_float(f384: np.ndarray) -> np.ndarray:
    """Deploy-faithful R upscale WITHOUT the final round/clamp (kept float for
    compositing): render-res (384,512,3) float -> bicubic up to camera (874,1164,3)
    float.  Same torch bicubic as _torch_R_to_camera_uint8 (train_witness ...)."""
    import torch

    x = torch.from_numpy(np.ascontiguousarray(f384)).permute(2, 0, 1)[None].float()
    with torch.inference_mode():
        up = torch.nn.functional.interpolate(
            x, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False)
    return up[0].permute(1, 2, 0).contiguous().numpy()  # (874,1164,3) float


def _bilinear_up_alpha(mask384: np.ndarray) -> np.ndarray:
    """Anti-aliased alpha matte: binary/soft (384,512) mask -> bilinear up to camera
    (874,1164) in [0,1].  Bilinear (not bicubic) so the alpha never overshoots the
    [0,1] range; the soft transition IS the anti-aliasing Daubechies requires."""
    import torch

    x = torch.from_numpy(np.ascontiguousarray(mask384.astype(np.float32)))[None, None]
    with torch.inference_mode():
        up = torch.nn.functional.interpolate(
            x, size=(CAMERA_H, CAMERA_W), mode="bilinear", align_corners=False)
    return np.clip(up[0, 0].numpy(), 0.0, 1.0)  # (874,1164)


def _composite_uint8(base_camf: np.ndarray, alpha_cam: np.ndarray,
                     fill_cam: np.ndarray) -> np.ndarray:
    """comp = (1-alpha)*base + alpha*fill at CAMERA res -> round/clamp -> uint8.
    ``fill_cam`` is (874,1164,3) float (GT RGB) or a (3,) float broadcast (flat proto).
    When alpha==0 this is bit-identical to _torch_R_to_camera_uint8(base_384)."""
    a = alpha_cam[..., None]
    if fill_cam.ndim == 1:
        fill = fill_cam[None, None, :]
    else:
        fill = fill_cam
    comp = (1.0 - a) * base_camf + a * fill
    return np.ascontiguousarray(np.clip(np.rint(comp), 0, 255).astype(np.uint8))


# ------------------------------------------------------------- frozen TR1 render
def load_module(checkpoint: Path):
    sys.path.insert(0, str(REPO / "experiments"))
    from ddm_fp1_class_field_projection import load_frozen_module
    return load_frozen_module(checkpoint)


# ---------------------------------------------------------- erased-component set
def erased_super_nucleus_mask(lane_gt: np.ndarray, base_realized: np.ndarray,
                              structure: np.ndarray) -> tuple[np.ndarray, int, int, int]:
    """From GT Lane mask (H,W bool) + base realized argmax (H,W), return the union
    support mask of erased super-nucleus (>5px) Lane components, plus
    (n_super, n_erased_super, target_px_before_rim).  A component is 'erased' iff
    <50% of its GT-Lane pixels are classified Lane in the base pass."""
    import scipy.ndimage as ndi

    lab, n = ndi.label(lane_gt, structure=structure)
    target = np.zeros_like(lane_gt, dtype=bool)
    n_super = 0
    n_erased = 0
    rec_lane = base_realized == LANE_CLASS
    if n:
        # vectorized per-component size + recovered count
        sizes = np.bincount(lab.ravel(), minlength=n + 1)
        rec_counts = np.bincount(lab.ravel(), weights=rec_lane.ravel(), minlength=n + 1)
        for c in range(1, n + 1):
            sz = int(sizes[c])
            if sz <= NUCLEUS_PX:
                continue
            n_super += 1
            if (rec_counts[c] / sz) < ERASED_RECOVER_THRESH:
                n_erased += 1
                target |= lab == c
    return target, n_super, n_erased, int(target.sum())


# -------------------------------------------------------------- per-chunk worker
def _per_class_flip_counts(realized: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """(5,) count of GT-class-c pixels whose realized argmax != gt (flipped)."""
    flip = realized != gt
    out = np.zeros(N_CLASSES, dtype=np.int64)
    for c in range(N_CLASSES):
        out[c] = int(((gt == c) & flip).sum())
    return out


def run_chunk(args, c0: int, c1: int, model, cfg, seg, lstars, gt_f1,
              structure) -> dict:
    """Render + 3 SegNet passes (base/oracle/flat) over pairs [c0,c1). Returns the
    per-pair accumulator dict for this chunk (also saved atomically to disk)."""
    import mlx.core as mx
    import scipy.ndimage as ndi

    from train_witness_realized_through_R_mlx import (
        _torch_R_to_camera_uint8,
        cpu_verdict_d_seg_argmax_batch,
    )

    idxs = list(range(c0, c1))
    npc = len(idxs)

    # ---- render base 384 (deploy stream = mlx cpu, matches trainer realized_gate) ----
    base384: list[np.ndarray] = []
    with mx.stream(mx.cpu):
        for i in idxs:
            rgb = model.render_frame(int(i))
            mx.eval(rgb)
            base384.append(np.asarray(rgb, dtype=np.float32)[0])  # (384,512,3)

    base_cam = [_torch_R_to_camera_uint8(f) for f in base384]
    gts = [lstars[i] for i in idxs]

    # ---- BASE SegNet pass (batched, seg_batch) ----
    base_realized = np.zeros((npc, SEG_H, SEG_W), dtype=np.int64)
    base_dseg = np.zeros(npc)
    for b0 in range(0, npc, args.seg_batch):
        b1 = min(b0 + args.seg_batch, npc)
        ds, rz = cpu_verdict_d_seg_argmax_batch(seg, base_cam[b0:b1], gts[b0:b1])
        base_dseg[b0:b1] = ds
        base_realized[b0:b1] = np.asarray(rz)

    # ---- erased super-nucleus target masks + AA alpha, oracle+flat composites ----
    oracle_cam: list[np.ndarray] = []
    flat_cam: list[np.ndarray] = []
    target_masks = np.zeros((npc, SEG_H, SEG_W), dtype=bool)
    n_super = np.zeros(npc, dtype=np.int64)
    n_erased = np.zeros(npc, dtype=np.int64)
    target_px = np.zeros(npc, dtype=np.int64)
    for k, i in enumerate(idxs):
        lane_gt = gts[k] == LANE_CLASS
        tmask, ns, ne, tpx = erased_super_nucleus_mask(lane_gt, base_realized[k], structure)
        target_masks[k] = tmask
        n_super[k] = ns
        n_erased[k] = ne
        target_px[k] = tpx
        # +1px rim then AA alpha (bilinear up); composite at CAMERA res pre-R.
        rim = ndi.binary_dilation(tmask, structure=structure, iterations=1) if tpx else tmask
        base_camf = _torch_bicubic_up_float(base384[k])
        if tpx:
            alpha = _bilinear_up_alpha(rim)
            oracle_cam.append(_composite_uint8(base_camf, alpha, gt_f1[i].astype(np.float32)))
            flat_cam.append(_composite_uint8(base_camf, alpha, PROTO_LANE_SOLVED))
        else:
            # empty target -> composite == base (bit-identical); reuse base_cam
            oracle_cam.append(base_cam[k])
            flat_cam.append(base_cam[k])

    # ---- ORACLE + FLAT SegNet passes ----
    oracle_realized = np.zeros((npc, SEG_H, SEG_W), dtype=np.int64)
    flat_realized = np.zeros((npc, SEG_H, SEG_W), dtype=np.int64)
    oracle_dseg = np.zeros(npc)
    flat_dseg = np.zeros(npc)
    for b0 in range(0, npc, args.seg_batch):
        b1 = min(b0 + args.seg_batch, npc)
        ds, rz = cpu_verdict_d_seg_argmax_batch(seg, oracle_cam[b0:b1], gts[b0:b1])
        oracle_dseg[b0:b1] = ds
        oracle_realized[b0:b1] = np.asarray(rz)
        ds, rz = cpu_verdict_d_seg_argmax_batch(seg, flat_cam[b0:b1], gts[b0:b1])
        flat_dseg[b0:b1] = ds
        flat_realized[b0:b1] = np.asarray(rz)

    # ---- per-pair accumulators ----
    cls_gt = np.zeros((npc, N_CLASSES), dtype=np.int64)
    cls_base = np.zeros((npc, N_CLASSES), dtype=np.int64)
    cls_oracle = np.zeros((npc, N_CLASSES), dtype=np.int64)
    cls_flat = np.zeros((npc, N_CLASSES), dtype=np.int64)
    base_flip_T = np.zeros(npc, dtype=np.int64)
    oracle_flip_T = np.zeros(npc, dtype=np.int64)
    flat_flip_T = np.zeros(npc, dtype=np.int64)
    base_flip_off = np.zeros(npc, dtype=np.int64)
    oracle_flip_off = np.zeros(npc, dtype=np.int64)
    flat_flip_off = np.zeros(npc, dtype=np.int64)
    # component-level recovery (of erased super-nucleus components)
    comp_recovered_oracle = np.zeros(npc, dtype=np.int64)
    comp_recovered_flat = np.zeros(npc, dtype=np.int64)
    for k in range(npc):
        g = gts[k]
        for c in range(N_CLASSES):
            cls_gt[k, c] = int((g == c).sum())
        cls_base[k] = _per_class_flip_counts(base_realized[k], g)
        cls_oracle[k] = _per_class_flip_counts(oracle_realized[k], g)
        cls_flat[k] = _per_class_flip_counts(flat_realized[k], g)
        T = target_masks[k]
        offT = ~T
        base_flip = base_realized[k] != g
        oracle_flip = oracle_realized[k] != g
        flat_flip = flat_realized[k] != g
        base_flip_T[k] = int((base_flip & T).sum())
        oracle_flip_T[k] = int((oracle_flip & T).sum())
        flat_flip_T[k] = int((flat_flip & T).sum())
        base_flip_off[k] = int((base_flip & offT).sum())
        oracle_flip_off[k] = int((oracle_flip & offT).sum())
        flat_flip_off[k] = int((flat_flip & offT).sum())
        # component-level: recount erased super-nucleus components now recovered (>50% Lane)
        if target_px[k]:
            lane_gt = g == LANE_CLASS
            lab, n = ndi.label(lane_gt, structure=structure)
            for cc in range(1, n + 1):
                comp = lab == cc
                if comp.sum() <= NUCLEUS_PX:
                    continue
                if not (comp & T).any():  # only erased components are in T
                    continue
                if (oracle_realized[k][comp] == LANE_CLASS).mean() >= ERASED_RECOVER_THRESH:
                    comp_recovered_oracle[k] += 1
                if (flat_realized[k][comp] == LANE_CLASS).mean() >= ERASED_RECOVER_THRESH:
                    comp_recovered_flat[k] += 1

    acc = {
        "idxs": np.asarray(idxs, dtype=np.int64),
        "base_dseg": base_dseg, "oracle_dseg": oracle_dseg, "flat_dseg": flat_dseg,
        "cls_gt": cls_gt, "cls_base": cls_base, "cls_oracle": cls_oracle, "cls_flat": cls_flat,
        "base_flip_T": base_flip_T, "oracle_flip_T": oracle_flip_T, "flat_flip_T": flat_flip_T,
        "base_flip_off": base_flip_off, "oracle_flip_off": oracle_flip_off,
        "flat_flip_off": flat_flip_off,
        "n_super": n_super, "n_erased": n_erased, "target_px": target_px,
        "comp_recovered_oracle": comp_recovered_oracle,
        "comp_recovered_flat": comp_recovered_flat,
    }
    return acc


# ---------------------------------------------------------------- aggregate/verdict
def aggregate(out_dir: Path, num_pairs: int, ckpt_sha: str) -> dict:
    accs = []
    c0 = 0
    while c0 < num_pairs:
        # discover chunk files
        cand = sorted(out_dir.glob(f"chunk_{c0:04d}_*.npz"))
        if not cand:
            raise RuntimeError(f"missing chunk starting at {c0}")
        z = np.load(cand[0])
        accs.append({k: z[k] for k in z.files})
        c1 = int(z["idxs"][-1]) + 1
        c0 = c1
    cat = {k: np.concatenate([a[k] for a in accs], 0) for k in accs[0]}

    total_px = float(num_pairs * SEG_H * SEG_W)
    sum_base_T = float(cat["base_flip_T"].sum())
    sum_oracle_T = float(cat["oracle_flip_T"].sum())
    sum_flat_T = float(cat["flat_flip_T"].sum())

    P = 100.0 * sum_base_T / total_px  # erased-Lane pool remaining (S-units)
    O = (sum_base_T - sum_oracle_T) / sum_base_T if sum_base_T else 0.0  # noqa: E741 (gc12 canonical falsifier name)
    F = (sum_base_T - sum_flat_T) / sum_base_T if sum_base_T else 0.0
    P_times_O = 100.0 * (sum_base_T - sum_oracle_T) / total_px  # recovered pool S-units (target-only)
    P_times_F = 100.0 * (sum_base_T - sum_flat_T) / total_px

    # joint dS per tier (net over ALL pixels; negative = improvement)
    base_total = float(cat["base_dseg"].sum())
    dS_joint_oracle = 100.0 * (cat["oracle_dseg"].sum() - base_total) / num_pairs
    dS_joint_flat = 100.0 * (cat["flat_dseg"].sum() - base_total) / num_pairs
    # collateral (off-target added flips), S-units
    coll_oracle = 100.0 * (cat["oracle_flip_off"].sum() - cat["base_flip_off"].sum()) / total_px
    coll_flat = 100.0 * (cat["flat_flip_off"].sum() - cat["base_flip_off"].sum()) / total_px

    def per_class_dS(cls_tier):
        return [100.0 * (float(cls_tier[:, c].sum()) - float(cat["cls_base"][:, c].sum()))
                / total_px for c in range(N_CLASSES)]

    base_dseg_mean = base_total / num_pairs
    base_per_class = [100.0 * float(cat["cls_base"][:, c].sum()) / total_px
                      for c in range(N_CLASSES)]

    n_erased_total = int(cat["n_erased"].sum())
    comp_O = (float(cat["comp_recovered_oracle"].sum()) / n_erased_total) if n_erased_total else 0.0
    comp_F = (float(cat["comp_recovered_flat"].sum()) / n_erased_total) if n_erased_total else 0.0

    # ---- mechanical falsifier routing (gc12 §4) ----
    reasons = []
    if O < 0.25:
        route = "CARRIER_FAMILY_CLOSES"
        reasons.append(f"O={O:.4f} < 0.25 (pool receiver/ERF-limited even on texture); "
                       "carrier family (b) CLOSES verdict_scope=FORMULATION(paint-on-texture "
                       "Lane compositing @ control_tail); (d) closes with it; burn-4 birth arm "
                       "defaults to KD-from-birth")
    else:
        route = None
    skip_burn4 = P_times_O < 0.05
    if skip_burn4:
        reasons.append(f"P*O={P_times_O:.4f} < 0.05 S (Contrarian bound): rung 2 SKIPPED, "
                       "slot -> burn-4 charter directly")
    if route is None:  # O >= 0.25
        if F >= 0.7 * O:
            route = "RUNG2_FIRES_b_FLAT_CARRIER"
            reasons.append(f"F={F:.4f} >= 0.7*O={0.7*O:.4f}: flat 1-2KB parametric Lane stroke "
                           "carrier suffices (cheapest build wins; learned paint stays unbuilt)")
        else:
            route = "RUNG2_FIRES_e1_TEXTURED_OR_SEEDING"
            reasons.append(f"O>=0.25 ^ F={F:.4f} < 0.7*O={0.7*O:.4f}: recovery real but needs "
                           "textured content -> (e1) solve-seeded births / textured carrier")
    if skip_burn4 and route != "CARRIER_FAMILY_CLOSES":
        route = route + "__BUT_P*O<0.05_SKIP_TO_BURN4"

    verdict = {
        "schema": "ddm_qa92_carrier_discriminator.v1",
        "evidence_axis": "[macOS-CPU advisory]", "score_claim": False, "research_only": True,
        "pointer": "0.1910828242 [contest-CPU] UNMOVED",
        "n_pairs": num_pairs, "ckpt": DEFAULT_CKPT, "ckpt_sha256": ckpt_sha,
        "class_order": CLASS_ORDER, "lane_class_index": LANE_CLASS,
        "proto_lane_solved": PROTO_LANE_SOLVED.round(2).tolist(),
        "nucleus_px_threshold": NUCLEUS_PX, "erased_recover_thresh": ERASED_RECOVER_THRESH,
        # base state
        "base_dseg_mean": round(base_dseg_mean, 7),
        "base_per_class_S_units": [round(x, 5) for x in base_per_class],
        "base_lane_S_units": round(base_per_class[LANE_CLASS], 5),
        # pool + recovery
        "P_pool_S_units": round(P, 5),
        "O_oracle_recovery_frac": round(O, 5),
        "F_flat_recovery_frac": round(F, 5),
        "PxO_recovered_S_units": round(P_times_O, 5),
        "PxF_recovered_S_units": round(P_times_F, 5),
        "comp_level_O_frac": round(comp_O, 5),
        "comp_level_F_frac": round(comp_F, 5),
        "n_erased_super_nucleus_total": n_erased_total,
        "n_super_nucleus_total": int(cat["n_super"].sum()),
        "target_px_total": int(cat["target_px"].sum()),
        # collateral + joint (the honest net; additive-S sole authority)
        "collateral_oracle_S_units": round(coll_oracle, 5),
        "collateral_flat_S_units": round(coll_flat, 5),
        "dS_joint_oracle_S_units": round(dS_joint_oracle, 5),
        "dS_joint_flat_S_units": round(dS_joint_flat, 5),
        "per_class_dS_oracle_S_units": [round(x, 5) for x in per_class_dS(cat["cls_oracle"])],
        "per_class_dS_flat_S_units": [round(x, 5) for x in per_class_dS(cat["cls_flat"])],
        # falsifier verdict + routing
        "falsifier_route": route,
        "rung2_skip_to_burn4": bool(skip_burn4),
        "routing_reasons": reasons,
    }
    return verdict


# --------------------------------------------------------------------- driver
def cmd_run(args) -> int:
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    import scipy.ndimage as ndi  # noqa: F401 (import check)

    structure = np.ones((3, 3), dtype=bool)  # 8-connectivity
    ckpt_sha = _sha256_file(args.checkpoint)
    print(f"[qa92] ckpt sha256 {ckpt_sha[:16]}...  n_pairs {args.num_pairs}")

    lstars, gt_f1 = load_gt(Path(DEFAULT_GT_CACHE), args.num_pairs)
    print(f"[qa92] gt loaded lstars {lstars.shape} gt_f1 {gt_f1.shape}")
    cfg, model = load_module(args.checkpoint)
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO / "upstream"))
    from tac.boundary_math.seg_core import load_real_segnet
    seg = load_real_segnet("cpu")

    t0 = time.time()
    for c0 in range(0, args.num_pairs, args.chunk):
        c1 = min(c0 + args.chunk, args.num_pairs)
        cpath = out_dir / f"chunk_{c0:04d}_{c1:04d}.npz"
        if cpath.exists():
            print(f"[skip] {cpath.name}", flush=True)
            continue
        acc = run_chunk(args, c0, c1, model, cfg, seg, lstars, gt_f1, structure)
        _atomic_save_npz(cpath, acc)
        bm = float(acc["base_dseg"].mean())
        om = float(acc["oracle_dseg"].mean())
        fm = float(acc["flat_dseg"].mean())
        print(f"[{c0}:{c1}] base {bm:.6f} oracle {om:.6f} flat {fm:.6f} "
              f"erased_super {int(acc['n_erased'].sum())}  {time.time()-t0:.0f}s", flush=True)

    verdict = aggregate(out_dir, args.num_pairs, ckpt_sha)
    verdict["wall_seconds"] = round(time.time() - t0, 1)
    _atomic_write_bytes(out_dir / "qa92_verdict.json",
                        (json.dumps(verdict, indent=1, sort_keys=True) + "\n").encode())
    print("\n=== QA92 VERDICT ===")
    print(json.dumps(verdict, indent=1, sort_keys=True))
    return 0


def cmd_aggregate(args) -> int:
    ckpt_sha = _sha256_file(args.checkpoint)
    verdict = aggregate(args.out_dir, args.num_pairs, ckpt_sha)
    _atomic_write_bytes(args.out_dir / "qa92_verdict.json",
                        (json.dumps(verdict, indent=1, sort_keys=True) + "\n").encode())
    print(json.dumps(verdict, indent=1, sort_keys=True))
    return 0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["run", "aggregate"])
    ap.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_CKPT))
    ap.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT))
    ap.add_argument("--num-pairs", type=int, default=600)
    ap.add_argument("--chunk", type=int, default=120)
    ap.add_argument("--seg-batch", type=int, default=12)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "aggregate":
        return cmd_aggregate(args)
    return 1


if __name__ == "__main__":
    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO / "experiments"))
    sys.path.insert(0, str(REPO / "upstream"))
    raise SystemExit(main())
