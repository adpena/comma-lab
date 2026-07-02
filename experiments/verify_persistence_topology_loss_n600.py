# SPDX-License-Identifier: MIT
"""$0 n600 verification of the persistence/topology loss (soft-clDice + island recall).

[macOS-numpy advisory / NON-PROMOTABLE] — pointer UNMOVED 0.19110. CPU + local-MLX only, no
remote GPU, no exact-eval. Proves the GRADIENT-SIGNAL QUALITY of the term (NOT a d_seg drop —
that needs a trained run). Real n600 GT (frozen-SegNet argmax authority) + real n96 witness.

Six checks:
  1. class self-detection on real n600 argmax  -> confirm Lane+Movable are the thin/small tail.
  2. density-weight mass by component-size bin  -> concentrated on the small/thin tail for the
     target classes, ~0 in the bulk-class interiors (targets islands, NOT bulk).
  3. erosion-erasure SENSITIVITY sweep          -> as islands are erased (recall drops), the
     topology loss rises MONOTONICALLY while per-pixel CE barely moves (the missing signal).
  4. toy MLX gradient descent (real frames)     -> CE+topology recovers MORE erased thin/small
     island pixels than CE-only (the loss RAISES island recall).
  5. real n96 witness corroboration             -> the density-weight mass overlaps the ACTUAL
     witness-flipped island pixels (simulation matches reality).
  6. COMPUTE benchmark                          -> MLX-GPU vs numpy throughput at trainer scale.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import mlx.core as mx
from scipy import ndimage

from tac.boundary_math import persistence_topology_loss as P

GT_N600 = Path("experiments/results/mlx_fleet_gt_cache/gt_n600.npz")
GT_N96 = Path("experiments/results/mlx_fleet_gt_cache/gt_n96.npz")
WITNESS_L7 = Path("experiments/results/witness_per_stage_attribution/maps_l7.npz")
OUT = Path("experiments/results/persistence_topology_verification")


def _oh(lab: np.ndarray, C: int = 5) -> np.ndarray:
    return np.eye(C, dtype=np.float32)[lab]


def _logits_from_argmax(lab: np.ndarray, C: int = 5, conf: float = 8.0) -> np.ndarray:
    """Confident logits realizing argmax==lab (a stand-in 'perfect'/'erased' prediction field)."""
    return (_oh(lab, C) * conf).astype(np.float32)


def check1_class_detection(lstars: np.ndarray) -> dict:
    targets, ev = P.detect_persistence_tail_classes(lstars, top_k=2, max_frames=200)
    return {
        "detected_targets": list(targets),
        "expected_lane_movable": [1, 3],
        "match": sorted(targets) == [1, 3],
        "evidence": [
            {
                "cls": e.cls, "frac": round(e.frac_of_frame, 5),
                "mean_comp_frac": round(e.mean_component_frac, 7),
                "thinness": round(e.thinness, 4), "n_comp": round(e.n_components_per_frame, 1),
                "erasure_risk": round(e.erasure_risk, 2),
            }
            for e in ev
        ],
    }


def check2_weight_mass_by_size(lstars: np.ndarray, targets, n_frames: int = 100) -> dict:
    """density-weight mass by component-size bin, per class. Tail classes should put mass on
    small comps; bulk classes' huge interior gets ~0 weight."""
    struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    out = {}
    n = min(n_frames, lstars.shape[0])
    for c in range(5):
        small_mass = 0.0  # weight mass on components < 64 px (the erasure tail)
        big_mass = 0.0    # weight mass on components >= 1024 px (bulk)
        for i in range(n):
            mask = (lstars[i] == c).astype(np.float32)
            if mask.sum() == 0:
                continue
            w = P.persistence_recall_weight_np(mask)  # (H,W)
            lab, k = ndimage.label(mask, structure=struct)
            if k == 0:
                continue
            sizes = np.bincount(lab.ravel())
            for comp in range(1, k + 1):
                comp_mask = lab == comp
                m = float(w[comp_mask].sum())
                if sizes[comp] < 64:
                    small_mass += m
                elif sizes[comp] >= 1024:
                    big_mass += m
        tot = small_mass + big_mass + 1e-9
        out[str(c)] = {
            "is_target": c in targets,
            "small_comp_mass_frac": round(small_mass / tot, 4),
            "big_comp_mass_frac": round(big_mass / tot, 4),
        }
    return out


def check3_erasure_sensitivity(lstars: np.ndarray, targets, n_frames: int = 30) -> dict:
    """Erode the target-class structures progressively; measure island-recall vs topology-loss
    vs per-pixel CE. Topology must rise monotonically as recall falls; CE must stay ~flat."""
    struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    n = min(n_frames, lstars.shape[0])
    rows = []
    for erode_it in (0, 1, 2, 3):
        recalls, topos, ces = [], [], []
        for i in range(n):
            lab = lstars[i]
            oh = _oh(lab)
            # erase: erode each target class, reassign erased pixels to the road class 0 (bulk).
            lab_er = lab.copy()
            for c in targets:
                mask = lab == c
                if mask.sum() == 0 or erode_it == 0:
                    continue
                keep = ndimage.binary_erosion(mask, structure=struct, iterations=erode_it)
                lab_er[mask & ~keep] = 0
            logits_er = _logits_from_argmax(lab_er)
            probs_er = P._softmax_last_np(logits_er)
            # island-recall over the ORIGINAL target-class pixels
            tgt_px = np.isin(lab, list(targets))
            if tgt_px.sum() > 0:
                recalls.append(float((lab_er[tgt_px] == lab[tgt_px]).mean()))
            topos.append(P.persistence_topology_loss_np(logits_er, oh, targets))
            # per-pixel CE over ALL pixels (topology-blind baseline)
            ce = -np.log(np.clip(np.sum(probs_er * oh, axis=-1), 1e-6, 1.0))
            ces.append(float(ce.mean()))
        rows.append({
            "erode_iters": erode_it,
            "island_recall": round(float(np.mean(recalls)), 4),
            "topology_loss": round(float(np.mean(topos)), 4),
            "perpixel_ce": round(float(np.mean(ces)), 5),
        })
    # monotonic: topology strictly increasing as erosion increases (recall decreasing)?
    topo_seq = [r["topology_loss"] for r in rows]
    ce_seq = [r["perpixel_ce"] for r in rows]
    topo_span = topo_seq[-1] - topo_seq[0]
    ce_span = ce_seq[-1] - ce_seq[0]
    return {
        "rows": rows,
        "topology_monotone_increasing": all(topo_seq[j] < topo_seq[j + 1] for j in range(len(topo_seq) - 1)),
        "topology_span": round(topo_span, 4),
        "ce_span": round(ce_span, 5),
        "topology_vs_ce_sensitivity_ratio": round(topo_span / (abs(ce_span) + 1e-6), 1),
    }


def check4_toy_descent(lstars: np.ndarray, targets, n_frames: int = 6, steps: int = 150,
                       lr: float = 10.0) -> dict:
    """From an erased init, optimize the SegNet-logit field toward GT with CE-only vs
    CE+topology (MLX autograd). Island-recall after = fraction of erased thin/small pixels
    recovered. Plain per-pixel CE is a mean over ALL H*W pixels, so the per-island-pixel
    gradient is ~1/(H*W) — too diffuse to flip a whole erased dash. The topology term's recall
    is normalized by the (small) weighted CLASS-pixel count, so its gradient CONCENTRATES on the
    islands — that is the mechanism this check isolates."""
    struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    n = min(n_frames, lstars.shape[0])
    res = {"ce_only_recall": [], "ce_plus_topo_recall": []}
    for i in range(n):
        lab = lstars[i]
        oh_np = _oh(lab)
        # erased init: erode target classes 2x -> islands gone.
        lab_er = lab.copy()
        erased_px = np.zeros(lab.shape, dtype=bool)
        for c in targets:
            mask = lab == c
            if mask.sum() == 0:
                continue
            keep = ndimage.binary_erosion(mask, structure=struct, iterations=2)
            drop = mask & ~keep
            lab_er[drop] = 0
            erased_px |= drop
        if erased_px.sum() == 0:
            continue
        init = mx.array(_logits_from_argmax(lab_er, conf=4.0))
        oh = mx.array(oh_np)

        def run(use_topo: bool):
            logits = mx.array(np.asarray(init))
            for _ in range(steps):
                def loss_fn(lg):
                    probs = mx.softmax(lg, axis=-1)
                    ce = mx.mean(-mx.log(mx.clip(mx.sum(probs * oh, axis=-1), 1e-6, 1.0)))
                    if use_topo:
                        return ce + 3.0 * P.persistence_topology_loss_mlx(lg, oh, targets)
                    return ce
                g = mx.grad(loss_fn)(logits)
                logits = logits - lr * g
                mx.eval(logits)
            pred = np.asarray(mx.argmax(logits, axis=-1))
            return float((pred[erased_px] == lab[erased_px]).mean())

        res["ce_only_recall"].append(run(False))
        res["ce_plus_topo_recall"].append(run(True))
    ce_r = float(np.mean(res["ce_only_recall"])) if res["ce_only_recall"] else 0.0
    tp_r = float(np.mean(res["ce_plus_topo_recall"])) if res["ce_plus_topo_recall"] else 0.0
    return {
        "n_frames_with_islands": len(res["ce_only_recall"]),
        "ce_only_island_recall": round(ce_r, 4),
        "ce_plus_topology_island_recall": round(tp_r, 4),
        "topology_recall_gain": round(tp_r - ce_r, 4),
    }


def check5_real_witness(gt96: dict) -> dict:
    """density-weight mass on ACTUAL witness-flipped island pixels vs correct pixels (n96)."""
    if not WITNESS_L7.exists():
        return {"skipped": "no witness maps"}
    wd = np.load(WITNESS_L7)
    wit = wd["argmax"].astype(np.int64)  # (96,H,W)
    gt = gt96["lstars"][: wit.shape[0]]
    targets = (1, 3)
    mass_on_flip, mass_on_correct = 0.0, 0.0
    for i in range(wit.shape[0]):
        for c in targets:
            mask = (gt[i] == c).astype(np.float32)
            if mask.sum() == 0:
                continue
            w = P.persistence_recall_weight_np(mask)
            flip = (gt[i] == c) & (wit[i] != c)  # GT class-c pixel the witness erased
            correct = (gt[i] == c) & (wit[i] == c)
            mass_on_flip += float(w[flip].sum())
            mass_on_correct += float(w[correct].sum())
    tot = mass_on_flip + mass_on_correct + 1e-9
    return {
        "weight_mass_on_witness_flips_frac": round(mass_on_flip / tot, 4),
        "weight_mass_on_correct_frac": round(mass_on_correct / tot, 4),
        "note": "high flip-mass => the persistence weight concentrates on the ACTUAL erased islands",
    }


def check6_benchmark(H: int = 384, W: int = 512, C: int = 5, reps: int = 20) -> dict:
    """MLX-GPU vs numpy throughput at the trainer's per-step shape (1,H,W,C)."""
    rng = np.random.default_rng(0)
    logits = (rng.standard_normal((1, H, W, C)) * 3).astype(np.float32)
    oh = _oh(logits[0].argmax(-1))[None]
    lm, om = mx.array(logits), mx.array(oh)
    targets = (1, 3)

    # numpy timing
    P.persistence_topology_loss_np(logits, oh, targets)  # warm
    t0 = time.perf_counter()
    for _ in range(reps):
        P.persistence_topology_loss_np(logits, oh, targets)
    np_ms = (time.perf_counter() - t0) / reps * 1e3

    # MLX (loss+grad) uncompiled
    def loss(lg):
        return P.persistence_topology_loss_mlx(lg, om, targets)
    vg = mx.value_and_grad(loss)
    v, g = vg(lm); mx.eval(v, g)  # warm
    t0 = time.perf_counter()
    for _ in range(reps):
        v, g = vg(lm); mx.eval(v, g)
    mlx_ms = (time.perf_counter() - t0) / reps * 1e3

    # MLX compiled loss (fwd only, hot-path fusion)
    cfn = P.make_persistence_topology_loss_mlx_compiled(targets)
    r = cfn(lm, om); mx.eval(r)
    t0 = time.perf_counter()
    for _ in range(reps):
        r = cfn(lm, om); mx.eval(r)
    mlx_c_ms = (time.perf_counter() - t0) / reps * 1e3

    return {
        "shape": [1, H, W, C],
        "numpy_fwd_ms": round(np_ms, 3),
        "mlx_gpu_fwd_bwd_ms": round(mlx_ms, 3),
        "mlx_gpu_compiled_fwd_ms": round(mlx_c_ms, 3),
        "n600_est_epoch_overhead_s_fwdbwd": round(mlx_ms * 600 / 1e3, 2),
        "metal_kernel_flag": P.metal_pool_kernel_signature()["env_flag"],
        "note": "hot path = soft-skeleton 3x3 min/max pool; #212 fused Metal kernel candidate",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d600 = np.load(GT_N600)
    lstars = d600["lstars"]  # (600,H,W)
    d96 = {k: np.load(GT_N96)[k] for k in ("lstars",)}

    print("[1/6] class self-detection (n600) ...")
    c1 = check1_class_detection(lstars)
    print("      targets:", c1["detected_targets"], "match Lane+Movable:", c1["match"])

    targets = tuple(c1["detected_targets"]) or (1, 3)
    print("[2/6] density-weight mass by component size ...")
    c2 = check2_weight_mass_by_size(lstars, targets)
    print("[3/6] erosion-erasure sensitivity sweep ...")
    c3 = check3_erasure_sensitivity(lstars, targets)
    print("      topology span %.3f vs CE span %.5f (ratio %.0fx); monotone=%s"
          % (c3["topology_span"], c3["ce_span"], c3["topology_vs_ce_sensitivity_ratio"],
             c3["topology_monotone_increasing"]))
    print("[4/6] toy MLX gradient descent (real frames) ...")
    c4 = check4_toy_descent(lstars, targets)
    print("      CE-only recall %.3f vs CE+topology %.3f (gain %.3f)"
          % (c4["ce_only_island_recall"], c4["ce_plus_topology_island_recall"], c4["topology_recall_gain"]))
    print("[5/6] real n96 witness corroboration ...")
    c5 = check5_real_witness(d96)
    print("[6/6] MLX-GPU vs numpy benchmark ...")
    c6 = check6_benchmark()
    print("      numpy %.2fms | mlx fwd+bwd %.2fms | mlx compiled fwd %.2fms"
          % (c6["numpy_fwd_ms"], c6["mlx_gpu_fwd_bwd_ms"], c6["mlx_gpu_compiled_fwd_ms"]))

    report = {
        "axis": "[macOS-numpy advisory / NON-PROMOTABLE]",
        "pointer": "UNMOVED 0.19110 (gradient-signal quality, not exact-eval)",
        "check1_class_detection": c1,
        "check2_weight_mass_by_size": c2,
        "check3_erasure_sensitivity": c3,
        "check4_toy_descent": c4,
        "check5_real_witness": c5,
        "check6_benchmark": c6,
    }
    out_path = OUT / "verification.json"
    out_path.write_text(json.dumps(report, indent=2))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
