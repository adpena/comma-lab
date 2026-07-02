#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""ACCELERATOR PROBE 3 — soft-logit distillation vs hard-argmax CE for d_seg descent.

The QUESTION (root-cause of Probe C's vanishing-gradient finding): the hard-argmax
cross-entropy gradient vanishes at the SegNet decision boundary — once the carrier's
reconstruction makes SegNet's argmax "mostly right", CE has little left to push, so
the exact argmax-flip d_seg descends as a slow power law (Probe C's ~50k-epoch
projection). The HYPOTHESIS: distilling the frozen SegNet's SOFT LOGITS (Hinton
T=2.0, the FULL 5-class inter-class structure — the "dark knowledge") instead of
the hard argmax carries a richer, non-vanishing gradient even where the argmax is
already correct, so the surrogate keeps descending exact d_seg after CE stalls.

WHY THIS IS A REAL, NON-TRIVIAL TEST (not a re-run of the soft_cosine lever):
the live pipeline (`tac.score_aware_loop.targets.build_gt_targets`) caches ONLY
``seg_targets_hard = so.argmax(dim=1)`` — it DISCARDS the teacher's full GT logits.
Every existing seg surrogate (CE, soft_cosine, lovasz, fisher_rao) therefore uses
a ONE-HOT GT target (no inter-class structure). Soft-logit distillation REQUIRES
the teacher's full soft distribution ``softmax(gt_logits / T)`` as the target. This
probe is the first to retain + use the teacher logits, so it tests a genuinely
distinct gradient.

THE DECISIVE SMOKE (apples-to-apples; the ONLY confound-free comparison):
* ONE shared init carrier (a learnable per-pair LAST-FRAME image — exactly the
  tensor SegNet sees; SegNet's gradient flows into it identically to how it flows
  into a decoder's output, so this isolates the seg-LOSS axis from any specific
  decoder architecture).
* N real GT pairs decoded via ``frame_utils.yuv420_to_rgb`` (the canonical GT path;
  PyAV rgb24 is FORBIDDEN — manufactures ~100x phantom pose).
* THREE arms, SAME init / SAME optimizer / SAME LR / SAME epochs / SAME seed:
    (A) CE         — cross-entropy to the HARD argmax target (the current default).
    (B) KL_T2      — KL on SegNet soft logits, T=2.0 (Hinton distillation), full
                     5-class teacher distribution as target.
    (C) HYBRID     — KL_T2 + a small hard-flip CE term (best-of-both).
* EXACT argmax-flip d_seg measured every epoch against the REAL frozen SegNet
  (``(argmax(pred_logits) != gt_argmax).float().mean()`` — the contest metric,
  upstream/modules.py:112), plus a fitted power-law descent exponent per arm.

POSE: held out of this probe ENTIRELY (the carrier is the SegNet last-frame; pose
is a 2-frame net handled separately). The CLAUDE.md risk "KL distill as PRIMARY
loss caused PoseNet collapse" is about pose; here KL is the SEG loss only and pose
is untouched, so this probe CANNOT harm pose by construction. We additionally
report a pose-proxy (the carrier's drift from the GT last-frame) so we can flag if
the soft-logit objective pushes the reconstruction visibly further from GT than CE
(which would be a pose-relevant fidelity regression to watch in a full run).

AUTHORITY: every number here is ``[macOS-CPU advisory]`` / ``[contest-CPU advisory]``
NON-PROMOTABLE (in-loop advisory d_seg on a synthetic carrier, NOT a byte-closed
archive). The frontier is UNMOVED. This probe GATES (it does not move) the score:
it answers whether soft-logit distillation is worth wiring as a real seg loss arm.

Run (CPU only — MPS is OFF here; the live basin owns the Apple GPU):
    .venv/bin/python experiments/probe_logit_distillation_dseg.py \
        --n-pairs 8 --epochs 300 --out experiments/results/probe3_logit_distill_<utc>
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F


# ── GT side: real frozen SegNet + GT pairs (teacher logits RETAINED) ─────────


def _build_teacher_targets(
    video_path: str, n_pairs: int, device: str
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode the first ``n_pairs`` GT pairs and run the REAL frozen SegNet.

    Returns:
      gt_last_chw  : (N, 3, 384, 512) float — GT LAST frame at SegNet OUTPUT
                     resolution (bilinear, the SegNet preprocess size) in [0,255].
                     This is the carrier's init/fidelity reference (the tensor the
                     SegNet preprocess actually consumes after interpolate).
      gt_logits    : (N, 5, 384, 512) float — frozen SegNet's FULL GT logits (the
                     teacher distribution; the soft-logit-distillation target).
      gt_argmax    : (N, 384, 512) int64 — argmax of gt_logits (the contest hard
                     target / the cached ``seg_targets_hard``).
    """
    import sys

    up = str(Path("upstream").resolve())
    if up not in sys.path:
        sys.path.insert(0, up)
    from tac.score_aware_loop.targets import load_frozen_distortion_net

    net = load_frozen_distortion_net(device=device)
    segnet = net.segnet  # frozen; preprocess_input takes last frame + interpolate

    import av
    from frame_utils import yuv420_to_rgb  # type: ignore

    container = av.open(str(video_path))
    last_frames: list[torch.Tensor] = []
    gt_logits_list: list[torch.Tensor] = []
    prev = None
    with torch.inference_mode():
        for frame in container.decode(container.streams.video[0]):
            f = yuv420_to_rgb(frame)  # (H, W, 3) float, camera resolution [0,255]
            if prev is None:
                prev = f
                continue
            f0, f1 = prev, f
            prev = None
            # The DistortionNet preprocess consumes (b, t, h, w, c); ``yuv420_to_rgb``
            # returns (H, W, C), so stack -> (2,H,W,C) -> unsqueeze -> (1,2,H,W,C).
            # SegNet sees ONLY the LAST frame, then bilinear-resizes it to its input
            # size. ``DistortionNet.preprocess_input`` does the (b,t,h,w,c)->(b,t,c,h,w)
            # rearrange + per-head preprocess; we take exactly its segnet_in so the
            # carrier == the tensor SegNet actually consumes.
            pair_bthwc = torch.stack([f0, f1]).unsqueeze(0).to(device).float()  # (1,2,H,W,3)
            _posenet_in, segnet_in = net.preprocess_input(pair_bthwc)  # segnet_in: (1,3,384,512)
            logits = segnet(segnet_in)  # (1,5,384,512)
            last_frames.append(segnet_in.squeeze(0).clone())  # (3,384,512)
            gt_logits_list.append(logits.squeeze(0).clone())  # (5,384,512)
            if len(last_frames) >= n_pairs:
                break
    container.close()

    gt_last = torch.stack(last_frames)  # (N,3,384,512)
    gt_logits = torch.stack(gt_logits_list)  # (N,5,384,512)
    gt_argmax = gt_logits.argmax(dim=1)  # (N,384,512)
    return gt_last, gt_logits, gt_argmax


# ── The carrier (the optimization variable SegNet's gradient flows into) ─────


def _make_carrier(gt_last: torch.Tensor, seed: int, init_noise: float) -> torch.Tensor:
    """A learnable per-pair last-frame image, init = GT last frame + small noise.

    Init at GT + noise (NOT at GT exactly) so there is a real, non-trivial d_seg to
    descend from a SHARED starting point (the same init for all 3 arms by seed). The
    carrier is the tensor SegNet's preprocess produces; SegNet's gradient flows into
    it exactly as it flows into a decoder's reconstruction, so optimizing it tests
    the seg-LOSS gradient quality directly, architecture-free.
    """
    g = torch.Generator().manual_seed(seed)
    noise = torch.empty_like(gt_last).normal_(0.0, init_noise, generator=g)
    carrier = (gt_last + noise).clone().detach()
    carrier.requires_grad_(True)
    return carrier


# ── The three seg-loss arms ──────────────────────────────────────────────────


def _seg_loss_ce(pred_logits: torch.Tensor, gt_argmax: torch.Tensor) -> torch.Tensor:
    """(A) CE to the HARD argmax target — the current default seg objective.

    Cross-entropy over the 5 classes against the teacher's argmax. The gradient
    w.r.t. a pixel vanishes as ``softmax(pred)[gt] -> 1`` (the boundary stall)."""
    return F.cross_entropy(pred_logits, gt_argmax)


def _seg_loss_kl_t2(
    pred_logits: torch.Tensor, gt_logits: torch.Tensor, temperature: float
) -> torch.Tensor:
    """(B) Soft-logit distillation — KL on the FULL teacher distribution, Hinton T.

    ``T*T * KL( softmax(pred/T) || softmax(teacher/T) )`` — the canonical Hinton-
    Vinyals-Dean 2014 distillation loss (mirrors
    ``tac.blocknerv_as_renderer.default_seg_surrogate``). The teacher's soft target
    carries the inter-class structure (dark knowledge), so the gradient does NOT
    vanish where the argmax is already correct — it keeps matching the relative
    logit gaps, which is what argmax-FLIP stability depends on at the boundary."""
    T = temperature
    log_p = F.log_softmax(pred_logits / T, dim=1)
    q = F.softmax(gt_logits / T, dim=1)
    return F.kl_div(log_p, q, reduction="none").sum(dim=1).mean() * (T * T)


def _seg_loss_hybrid(
    pred_logits: torch.Tensor,
    gt_logits: torch.Tensor,
    gt_argmax: torch.Tensor,
    temperature: float,
    ce_weight: float,
) -> torch.Tensor:
    """(C) HYBRID — soft-logit KL + a small hard-flip CE term.

    The KL term carries the rich boundary gradient; the small CE term keeps a direct
    pull on the actual argmax-flip pixels. ``ce_weight`` is small (the CE is the
    minority term) so the soft-logit signal dominates."""
    kl = _seg_loss_kl_t2(pred_logits, gt_logits, temperature)
    ce = _seg_loss_ce(pred_logits, gt_argmax)
    return kl + ce_weight * ce


# ── Exact contest d_seg (argmax-flip rate) ───────────────────────────────────


@torch.no_grad()
def _exact_d_seg(pred_logits: torch.Tensor, gt_argmax: torch.Tensor) -> float:
    """The EXACT contest metric: per-pixel argmax-disagreement rate, mean over the
    batch (upstream/modules.py:112 — ``(argmax(out1)!=argmax(out2)).float().mean()``)."""
    return (pred_logits.argmax(dim=1) != gt_argmax).float().mean().item()


# ── Power-law descent exponent fit ───────────────────────────────────────────


def _fit_descent_exponent(epochs: list[int], d_seg: list[float]) -> dict[str, float]:
    """Fit ``d_seg(t) ≈ a * t^(-p)`` over the post-warmup tail (log-log least
    squares); ``p`` is the descent exponent — larger ``p`` = faster power-law
    descent toward the floor. Returns the exponent + the implied epochs-to-target.

    Probe C projected ~50k epochs to a sub-0.15 d_seg under the CE exponent; a
    materially larger exponent for soft-logit distillation would overturn that
    projection. Fit on the tail (skip the first 25% — the transient) so the
    exponent reflects the asymptotic descent, not the init drop."""
    pts = [
        (e, d)
        for e, d in zip(epochs, d_seg)
        if e > 0 and d > 1e-9
    ]
    if len(pts) < 4:
        return {"exponent_p": float("nan"), "fit_n": len(pts)}
    tail_start = pts[len(pts) // 4][0]  # skip first 25%
    tail = [(e, d) for e, d in pts if e >= tail_start]
    if len(tail) < 4:
        tail = pts
    xs = [math.log(e) for e, _ in tail]
    ys = [math.log(d) for _, d in tail]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    slope = sxy / sxx if sxx > 0 else float("nan")  # = -p
    p = -slope
    return {"exponent_p": p, "fit_n": n, "tail_start_epoch": float(tail_start)}


# ── One arm ──────────────────────────────────────────────────────────────────


def _run_arm(
    arm: str,
    gt_last: torch.Tensor,
    gt_logits: torch.Tensor,
    gt_argmax: torch.Tensor,
    segnet: torch.nn.Module,
    *,
    epochs: int,
    lr: float,
    seed: int,
    init_noise: float,
    temperature: float,
    hybrid_ce_weight: float,
    eval_every: int,
) -> dict:
    """Train ONE seg-loss arm from the shared init; return its d_seg trajectory +
    fitted descent exponent + final pose-fidelity proxy."""
    carrier = _make_carrier(gt_last, seed=seed, init_noise=init_noise)
    opt = torch.optim.Adam([carrier], lr=lr)

    traj_epoch: list[int] = []
    traj_d_seg: list[float] = []
    traj_loss: list[float] = []
    t0 = time.time()
    for ep in range(epochs + 1):
        # SegNet forward on the carrier (the carrier IS the segnet_in tensor).
        logits = segnet(carrier)  # (N,5,384,512)
        if arm == "CE":
            loss = _seg_loss_ce(logits, gt_argmax)
        elif arm == "KL_T2":
            loss = _seg_loss_kl_t2(logits, gt_logits, temperature)
        elif arm == "HYBRID":
            loss = _seg_loss_hybrid(
                logits, gt_logits, gt_argmax, temperature, hybrid_ce_weight
            )
        else:
            raise ValueError(f"unknown arm {arm!r}")

        if ep % eval_every == 0 or ep == epochs:
            d_seg = _exact_d_seg(logits, gt_argmax)
            traj_epoch.append(ep)
            traj_d_seg.append(d_seg)
            traj_loss.append(float(loss.item()))

        if ep == epochs:
            break
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    # Pose-fidelity proxy: how far the final carrier drifted from the GT last frame
    # (RMSE in [0,255] pixel units). A soft-logit objective that pushed the carrier
    # much further from GT than CE would be a fidelity (pose-relevant) regression.
    with torch.no_grad():
        recon_rmse = math.sqrt(F.mse_loss(carrier, gt_last).item())

    fit = _fit_descent_exponent(traj_epoch, traj_d_seg)
    return {
        "arm": arm,
        "epochs": epochs,
        "lr": lr,
        "seed": seed,
        "init_noise": init_noise,
        "temperature": temperature,
        "hybrid_ce_weight": hybrid_ce_weight,
        "wall_sec": round(time.time() - t0, 2),
        "d_seg_init": traj_d_seg[0] if traj_d_seg else None,
        "d_seg_final": traj_d_seg[-1] if traj_d_seg else None,
        "recon_rmse_vs_gt": round(recon_rmse, 4),
        "descent_exponent_p": fit.get("exponent_p"),
        "fit_n": fit.get("fit_n"),
        "trajectory": [
            {"epoch": e, "d_seg": d, "loss": l}
            for e, d, l in zip(traj_epoch, traj_d_seg, traj_loss)
        ],
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--video", default="upstream/videos/0.mkv")
    p.add_argument("--n-pairs", type=int, default=8,
                   help="real GT pairs to train the carrier on (small-n decisive smoke).")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.5,
                   help="Adam LR on the carrier (image-domain; high is fine — it is "
                        "a direct-pixel variable).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--init-noise", type=float, default=12.0,
                   help="stddev (in [0,255] pixel units) of the shared-init noise on "
                        "the GT last frame — sets the starting d_seg the arms descend.")
    p.add_argument("--temperature", type=float, default=2.0,
                   help="Hinton distillation temperature for the soft-logit arms.")
    p.add_argument("--hybrid-ce-weight", type=float, default=0.1)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--out", type=Path,
                   default=Path("experiments/results/probe3_logit_distill"))
    args = p.parse_args(argv)

    if torch.cuda.is_available():
        device = "cpu"  # keep this probe deterministic + CPU-only (MPS/CUDA off)
    else:
        device = "cpu"
    torch.manual_seed(args.seed)

    print("=" * 80)
    print("PROBE 3 — soft-logit distillation (KL T=2.0) vs hard-argmax CE: d_seg descent")
    print(f"  n_pairs={args.n_pairs} epochs={args.epochs} lr={args.lr} "
          f"init_noise={args.init_noise} T={args.temperature} device={device}")
    print("  [macOS-CPU advisory] NON-PROMOTABLE — gates whether soft-logit distill is")
    print("  worth wiring as a real seg loss; the exact frontier is UNMOVED.")
    print("=" * 80)

    print("\n[1/3] decoding real GT pairs + retaining teacher SegNet logits ...")
    t_targets = time.time()
    gt_last, gt_logits, gt_argmax = _build_teacher_targets(
        args.video, args.n_pairs, device
    )
    print(f"      gt_last={tuple(gt_last.shape)} gt_logits={tuple(gt_logits.shape)} "
          f"gt_argmax={tuple(gt_argmax.shape)}  ({time.time()-t_targets:.1f}s)")

    # Re-load the frozen SegNet once for the training forwards (gradient-enabled
    # input, frozen weights).
    import sys
    up = str(Path("upstream").resolve())
    if up not in sys.path:
        sys.path.insert(0, up)
    from tac.score_aware_loop.targets import load_frozen_distortion_net
    segnet = load_frozen_distortion_net(device=device).segnet

    arms = ["CE", "KL_T2", "HYBRID"]
    results = []
    for i, arm in enumerate(arms, 1):
        print(f"\n[2/3] arm {i}/{len(arms)}: {arm} (shared init, seed={args.seed}) ...")
        r = _run_arm(
            arm, gt_last, gt_logits, gt_argmax, segnet,
            epochs=args.epochs, lr=args.lr, seed=args.seed,
            init_noise=args.init_noise, temperature=args.temperature,
            hybrid_ce_weight=args.hybrid_ce_weight, eval_every=args.eval_every,
        )
        results.append(r)
        print(f"      d_seg {r['d_seg_init']:.6f} -> {r['d_seg_final']:.6f}  "
              f"exponent p={r['descent_exponent_p']:.3f}  "
              f"recon_rmse={r['recon_rmse_vs_gt']}  ({r['wall_sec']}s)")

    # ── Verdict ──────────────────────────────────────────────────────────────
    print("\n[3/3] VERDICT")
    print("-" * 80)
    ce = next(r for r in results if r["arm"] == "CE")
    kl = next(r for r in results if r["arm"] == "KL_T2")
    hy = next(r for r in results if r["arm"] == "HYBRID")
    print(f"  {'arm':>8} {'d_seg_final':>12} {'exponent_p':>11} {'recon_rmse':>11} "
          f"{'Δd_seg_vs_CE':>13}")
    for r in (ce, kl, hy):
        dvs = r["d_seg_final"] - ce["d_seg_final"]
        print(f"  {r['arm']:>8} {r['d_seg_final']:>12.6f} "
              f"{r['descent_exponent_p']:>11.3f} {r['recon_rmse_vs_gt']:>11.4f} "
              f"{dvs:>+13.6f}")

    # Decision rules.
    faster_kl = (kl["descent_exponent_p"] or 0) > (ce["descent_exponent_p"] or 0) * 1.05
    lower_kl = kl["d_seg_final"] < ce["d_seg_final"] * 0.95
    faster_hy = (hy["descent_exponent_p"] or 0) > (ce["descent_exponent_p"] or 0) * 1.05
    lower_hy = hy["d_seg_final"] < ce["d_seg_final"] * 0.95
    # Pose-fidelity guard: flag if a soft-logit arm drifts >25% further from GT.
    kl_fidelity_regress = kl["recon_rmse_vs_gt"] > ce["recon_rmse_vs_gt"] * 1.25
    hy_fidelity_regress = hy["recon_rmse_vs_gt"] > ce["recon_rmse_vs_gt"] * 1.25

    if (faster_kl or lower_kl) and not kl_fidelity_regress:
        verdict = "ACCELERATES — soft-logit KL_T2 descends d_seg faster/lower than CE, no fidelity regress"
    elif (faster_hy or lower_hy) and not hy_fidelity_regress:
        verdict = "ACCELERATES (HYBRID) — KL+CE hybrid descends faster/lower than CE, no fidelity regress"
    elif faster_kl or lower_kl or faster_hy or lower_hy:
        verdict = "MIXED — a soft-logit arm wins on d_seg but a fidelity-regress flag fired (watch pose in a full run)"
    else:
        verdict = "NO ACCELERATION — soft-logit distillation does NOT beat hard CE on d_seg here"

    print("-" * 80)
    print(f"  VERDICT: {verdict}")
    print(f"  faster_kl={faster_kl} lower_kl={lower_kl} "
          f"faster_hy={faster_hy} lower_hy={lower_hy}")
    print(f"  fidelity-regress flags: kl={kl_fidelity_regress} hy={hy_fidelity_regress}")
    print("-" * 80)

    args.out.mkdir(parents=True, exist_ok=True)
    out_json = args.out / "probe3_logit_distill_result.json"
    payload = {
        "probe": "probe3_logit_distillation_dseg",
        "authority": "[macOS-CPU advisory] NON-PROMOTABLE",
        "config": vars(args) | {"device": device, "video": str(args.video),
                                "out": str(args.out)},
        "verdict": verdict,
        "decision_flags": {
            "faster_kl": faster_kl, "lower_kl": lower_kl,
            "faster_hy": faster_hy, "lower_hy": lower_hy,
            "kl_fidelity_regress": kl_fidelity_regress,
            "hy_fidelity_regress": hy_fidelity_regress,
        },
        "arms": results,
    }
    out_json.write_text(json.dumps(payload, indent=2))
    print(f"  wrote {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
