# SPDX-License-Identifier: MIT
"""Real-teacher A/B: boundary-weighted TCKD vs Hinton KL T=2.0 (the optimal seg teacher test).

Tests the falsifiable prediction from the teacher design memo
(`.omx/research/optimal_scorer_teacher_design_20260531T103350Z.md` §4): the
boundary-weighted TCKD seg-distillation objective reaches a lower ``d_seg``
(per-pixel argmax-disagreement RATE — the actual contest SegNet distortion) than
KL T=2.0 at MATCHED training steps, because it re-allocates gradient off the
``O(H*W)`` score-flat confident interior pixels onto the ~5-15% decision-boundary
band where ``∂d_seg/∂rendered != 0``.

CONTEST-FAITHFUL by construction:
  * Teacher = the REAL contest SegNet (``smp.Unet`` EfficientNet-B2) on REAL
    frames decoded from ``upstream/videos/0.mkv`` (NOT a synthetic fixture —
    Catalog forbidden class #3).
  * d_seg = ``(student.argmax != teacher.argmax).mean()`` — the EXACT
    ``upstream/modules.py`` SegNet.compute_distortion functional.
  * The student is a learnable per-pixel logit field initialized near the
    teacher (teacher + noise) — this models the SUBSTRATE setting: the rendered
    frame -> SegNet mostly agrees on confident interiors (large regions of
    road/sky) and disagrees at boundaries. The A/B asks which objective fixes
    the boundary disagreements faster at matched steps. This isolates the
    teacher OBJECTIVE from the substrate architecture.

Output is ``[macOS-MLX research-signal]`` — a training-dynamics proxy on the
SegNet decision functional, NOT a contest-score claim (per CLAUDE.md "MLX
portable-local-substrate authority" + Catalog #192/#341). The downstream
contest-faithful test is wiring boundary_tckd into a substrate trainer and
running it; this A/B validates the mechanism + the >=8% prediction direction
that justifies that wire-in.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np


def _decode_frames(video_path: str, n_frames: int) -> np.ndarray:
    import av

    cont = av.open(video_path)
    frames = []
    for i, fr in enumerate(cont.decode(video=0)):
        if i >= n_frames:
            break
        frames.append(fr.to_ndarray(format="rgb24"))
    cont.close()
    return np.stack(frames).astype(np.float32) / 255.0  # (N, H, W, 3) in [0,1]


def _real_segnet_teacher_logits(frames_nhwc: np.ndarray, *, downsample: int) -> np.ndarray:
    """REAL contest SegNet logits on real frames. Returns (N_pixels, num_classes)."""
    import torch

    from tac.scorer import load_default_scorers

    _, segnet = load_default_scorers("upstream", device="cpu")
    n, h, w, _ = frames_nhwc.shape
    # SegNet.preprocess_input expects (B, T, C, H, W) and slices x[:, -1] (last
    # frame) then interpolates to the canonical (512, 384). We pass T=1.
    x = (
        torch.from_numpy(frames_nhwc)
        .permute(0, 3, 1, 2)  # (N, 3, H, W)
        .unsqueeze(1)  # (N, 1, 3, H, W)  -> T=1
    )
    with torch.no_grad():
        pre = segnet.preprocess_input(x)  # (N, 3, 384, 512)
        logits = segnet(pre)  # (N, C, 384, 512)
    logits = logits.detach().float().numpy()  # (N, C, Hs, Ws)
    if downsample > 1:
        # average-pool the SPATIAL axes to shrink the pixel count for a fast A/B
        # This is a training-dynamics proxy: pooled logits can change argmax
        # decisions, so scorer-fidelity claims require --downsample 1.
        nC, c, hs, ws = logits.shape
        hs2, ws2 = hs // downsample, ws // downsample
        logits = logits[:, :, : hs2 * downsample, : ws2 * downsample]
        logits = logits.reshape(nC, c, hs2, downsample, ws2, downsample).mean(axis=(3, 5))
    nC, c, hs, ws = logits.shape
    # -> (N_pixels, C) with class axis last (the kernel convention).
    return np.transpose(logits, (0, 2, 3, 1)).reshape(-1, c).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="upstream/videos/0.mkv")
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--downsample", type=int, default=4)
    ap.add_argument("--noise-std", type=float, default=2.0)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--lr", type=float, default=0.3)
    ap.add_argument("--temperature", type=float, default=2.0)
    ap.add_argument("--tau-boundary", type=float, default=1.0)
    ap.add_argument("--hinge-margin", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import mlx.core as mx
    import mlx.optimizers as optim

    from tac.substrates.hinton_distilled_scorer_surrogate.mlx_loss import (
        boundary_argmax_hinge_loss,
        boundary_decision_tckd_loss,
        boundary_weighted_tckd_loss,
        hinton_distilled_kl_t2_loss,
        segnet_boundary_band_weights_mlx,
    )

    t0 = time.time()
    frames = _decode_frames(args.video, args.n_frames)
    teacher_np = _real_segnet_teacher_logits(frames, downsample=args.downsample)
    teacher = mx.array(teacher_np)  # (P, C)
    n_pixels, n_classes = teacher.shape
    teacher_argmax = mx.argmax(teacher, axis=-1)

    # boundary-band fraction on the REAL teacher (calibrates tau_b honesty).
    band_w = segnet_boundary_band_weights_mlx(teacher, tau_boundary=args.tau_boundary)
    band_frac = float((band_w > np.exp(-1.0)).mean())

    mx.random.seed(args.seed)
    init = teacher + mx.random.normal(teacher.shape) * args.noise_std  # near-correct start

    def d_seg(student: mx.array) -> float:
        return float((mx.argmax(student, axis=-1) != teacher_argmax).astype(mx.float32).mean())

    def run(objective: str) -> dict:
        student = mx.array(init)  # same init for ALL arms (apples-to-apples)
        opt = optim.Adam(learning_rate=args.lr)
        teacher_sg = mx.stop_gradient(teacher)
        traj = [d_seg(student)]
        if objective == "kl_t2":
            loss_fn = lambda s: hinton_distilled_kl_t2_loss(  # noqa: E731
                s, teacher_sg, temperature=args.temperature
            )
        elif objective == "boundary_tckd":  # target-vs-rest TCKD (Object 2, binarised)
            loss_fn = lambda s: boundary_weighted_tckd_loss(  # noqa: E731
                s, teacher_sg, temperature=args.temperature, tau_boundary=args.tau_boundary
            )
        elif objective == "boundary_decision_tckd":  # runner-up-AWARE top1<->top2
            loss_fn = lambda s: boundary_decision_tckd_loss(  # noqa: E731
                s, teacher_sg, temperature=args.temperature, tau_boundary=args.tau_boundary
            )
        elif objective == "boundary_argmax_hinge":  # impostor-COMPLETE Crammer-Singer hinge
            # margin REPLACES temperature: raw-logit surrogate whose zero-set is
            # EXACTLY {argmax-correct}; max_{j!=target} sweeps ALL impostor classes
            # (incl out-of-pair top3/4/5) -> nonzero whenever d_seg is wrong.
            loss_fn = lambda s: boundary_argmax_hinge_loss(  # noqa: E731
                s, teacher_sg, margin=args.hinge_margin, tau_boundary=args.tau_boundary
            )
        else:
            raise ValueError(f"unknown objective: {objective}")
        grad_fn = mx.value_and_grad(loss_fn)
        for _ in range(args.steps):
            loss, g = grad_fn(student)
            student = opt.apply_gradients({"s": g}, {"s": student})["s"]
            mx.eval(student)
            traj.append(d_seg(student))
        return {"final_d_seg": traj[-1], "init_d_seg": traj[0], "trajectory": traj}

    kl = run("kl_t2")
    tckd = run("boundary_tckd")
    dec = run("boundary_decision_tckd")
    hinge = run("boundary_argmax_hinge")

    init_dseg = kl["init_d_seg"]

    def _rel(arm: dict) -> float:
        # relative d_seg reduction vs KL T=2.0 at matched steps (the falsifiable metric).
        return (
            (kl["final_d_seg"] - arm["final_d_seg"]) / kl["final_d_seg"]
            if kl["final_d_seg"] > 0
            else 0.0
        )

    rel_reduction = _rel(tckd)  # target-vs-rest TCKD (the original metric; back-compat)
    rel_reduction_decision = _rel(dec)  # runner-up-aware decision-TCKD (the headline arm)
    rel_reduction_hinge = _rel(hinge)  # impostor-complete Crammer-Singer hinge (the loss-side fix)
    # does the runner-up identity help OVER target-vs-rest? (Object 2 marginal value)
    decision_over_tckd = (
        (tckd["final_d_seg"] - dec["final_d_seg"]) / tckd["final_d_seg"]
        if tckd["final_d_seg"] > 0
        else 0.0
    )
    # does the impostor-COMPLETE hinge help OVER the soft losses? (the operator's
    # "out-of-pair impostor class" insight: soft losses can be zero while d_seg is
    # wrong; the hinge cannot be -> it should win when impostors are out-of-pair).
    hinge_over_best_soft = (
        (min(tckd["final_d_seg"], dec["final_d_seg"]) - hinge["final_d_seg"])
        / min(tckd["final_d_seg"], dec["final_d_seg"])
        if min(tckd["final_d_seg"], dec["final_d_seg"]) > 0
        else 0.0
    )
    # winner among the four arms (lowest final d_seg).
    arms = {
        "kl_t2": kl["final_d_seg"],
        "boundary_tckd": tckd["final_d_seg"],
        "boundary_decision_tckd": dec["final_d_seg"],
        "boundary_argmax_hinge": hinge["final_d_seg"],
    }
    winner = min(arms, key=arms.get)
    result = {
        "evidence_grade": "[macOS-MLX research-signal]",
        "score_claim": False,
        "promotable": False,
        "axis_tag": "[macOS-MLX research-signal]",
        "real_teacher": "contest SegNet (smp.Unet EfficientNet-B2) on upstream/videos/0.mkv",
        "n_frames": args.n_frames,
        "downsample": args.downsample,
        "scorer_fidelity": (
            "full_scorer_grid"
            if args.downsample == 1
            else "pooled_logit_training_dynamics_proxy"
        ),
        "argmax_preservation_claim": args.downsample == 1,
        "full_grid_rerun_required_for_exact_scorer_fidelity": args.downsample != 1,
        "n_pixels": n_pixels,
        "n_classes": n_classes,
        "boundary_band_fraction": band_frac,
        "noise_std": args.noise_std,
        "steps": args.steps,
        "lr": args.lr,
        "temperature": args.temperature,
        "tau_boundary": args.tau_boundary,
        "hinge_margin": args.hinge_margin,
        "init_d_seg": init_dseg,
        "kl_t2_final_d_seg": kl["final_d_seg"],
        "boundary_tckd_final_d_seg": tckd["final_d_seg"],
        "boundary_decision_tckd_final_d_seg": dec["final_d_seg"],
        "boundary_argmax_hinge_final_d_seg": hinge["final_d_seg"],
        "relative_d_seg_reduction_vs_kl_t2": rel_reduction,
        "relative_d_seg_reduction_decision_vs_kl_t2": rel_reduction_decision,
        "relative_d_seg_reduction_hinge_vs_kl_t2": rel_reduction_hinge,
        "runner_up_identity_marginal_vs_target_vs_rest": decision_over_tckd,
        "impostor_complete_hinge_marginal_vs_best_soft": hinge_over_best_soft,
        "winner": winner,
        "prediction_threshold": 0.08,
        "prediction_holds": rel_reduction >= 0.08,
        "decision_prediction_holds": rel_reduction_decision >= 0.08,
        "hinge_prediction_holds": rel_reduction_hinge >= 0.08,
        "kl_t2_trajectory": kl["trajectory"],
        "boundary_tckd_trajectory": tckd["trajectory"],
        "boundary_decision_tckd_trajectory": dec["trajectory"],
        "boundary_argmax_hinge_trajectory": hinge["trajectory"],
        "wall_clock_s": round(time.time() - t0, 1),
    }
    print(json.dumps({k: v for k, v in result.items() if "trajectory" not in k}, indent=2))
    print(
        f"\nVERDICT: init d_seg={init_dseg:.4f} | "
        f"KL T=2.0 -> {kl['final_d_seg']:.4f} | "
        f"target-vs-rest TCKD -> {tckd['final_d_seg']:.4f} ({rel_reduction * 100:+.1f}% vs KL) | "
        f"runner-up-aware decision-TCKD -> {dec['final_d_seg']:.4f} "
        f"({rel_reduction_decision * 100:+.1f}% vs KL) | "
        f"impostor-complete hinge -> {hinge['final_d_seg']:.4f} "
        f"({rel_reduction_hinge * 100:+.1f}% vs KL) | "
        f"WINNER: {winner} | hinge worth {hinge_over_best_soft * 100:+.1f}% over best soft loss"
    )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(result, indent=2))
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
