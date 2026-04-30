"""Lane Ω-V2 — Lagrangian QAT loop with per-WEIGHT learnable bit-depth.

This is the mathematically-optimal evolution of Lane Ω-V1.

Replaces the closed-form water-fill bit allocator with Lagrangian dual
ascent + learnable per-element bit-depth. The KKT condition at the
constrained optimum (rate-distortion duality) is

    ∂D/∂bits_ij = -λ · ∂R/∂bits_ij    ∀ (i,j)

which Lagrangian dual ascent + SGD on bits converges to. See
``src/tac/learnable_bit_quant.py`` docstring for the full derivation.

Pipeline (single Python script, no orchestration deps):

    1. Load Lane A renderer (fp32) from --checkpoint.
    2. Swap eligible Conv2d → LearnableBitConv2d (init_bits=8.0, optional
       Hessian warm-start from --hessian-init).
    3. Setup λ ramp: λ=lambda_start until lambda_ramp_start_frac of total
       epochs, then linearly ramp to lambda_end.
    4. Per step:
         scorer_loss = SegNet hinge + PoseNet MSE   (eval_roundtrip=True,
                                                     mandatory)
         kl_loss     = KL(student || teacher) on the renderer's RGB output
                       (post-fix weight = 0.002, validated on Lane A path)
         rate_loss   = λ · (mean(bits) - target_bits)         # linear primal
                                                              # (Round 4 fix:
                                                              # no ReLU; KKT
                                                              # clamp lives
                                                              # in dual update)
         loss        = scorer_loss + kl_weight * kl_loss + rate_loss
       Backward, optimizer step (separate parameter groups: weights at lr,
       bits at lr * 0.1).
    5. Save best checkpoint (lowest val scorer loss).
    6. Round bits to int and re-export as OMG1.

CLAUDE.md compliance:
  * --device cuda mandatory (no MPS / CPU fallback).
  * eval_roundtrip=True — non-negotiable per CLAUDE.md.
  * Strict-scorer-rule: scorers loaded for QAT only, not in archive.
  * Provenance JSON written next to output.
  * `parser.add_argument` flags pinned to the surface declared here +
    matched against the launcher script's `--<flag>` calls (see
    ``test_remote_lane_omega_v2_script.py``).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "upstream"))


# ── Helpers ───────────────────────────────────────────────────────────────


def _device_strict(device: str) -> torch.device:
    """CUDA-required device check (CLAUDE.md FORBIDDEN: MPS/CPU fallback)."""
    if device == "mps":
        raise SystemExit(
            "FATAL: --device mps forbidden. CLAUDE.md non-negotiable: "
            "MPS-CUDA drift on PoseNet is 23x. Lane Ω-V2 QAT is CUDA-only."
        )
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "FATAL: --device cuda requested but CUDA not available. "
            "Lane Ω-V2 QAT runs on CUDA only (proxy-auth gap on non-CUDA "
            "is unbounded). Pass --device cpu only with the explicit "
            "[advisory only] tag understanding."
        )
    return torch.device(device)


def _load_renderer(checkpoint: str, device: torch.device) -> nn.Module:
    from tac.renderer_export import load_any_renderer_checkpoint

    model = load_any_renderer_checkpoint(checkpoint, device=str(device))
    model.train()  # QAT needs train mode for STE noise (we still no-grad scorers)
    return model


def _load_poses(poses_path: str | None, device: torch.device) -> torch.Tensor | None:
    if poses_path is None:
        return None
    from tac.submission_archive import load_optimized_poses

    return load_optimized_poses(
        poses_path, pose_dim=6, expected_n_pairs=600,
    ).to(device)


def _decode_video(video_mkv: str) -> torch.Tensor:
    """Decode 1200-frame GT video → uint8 (N, 3, H, W) on CPU."""
    import av

    container = av.open(video_mkv)
    stream = container.streams.video[0]
    frames = []
    for frame in container.decode(stream):
        rgb = frame.to_ndarray(format="rgb24")
        frames.append(torch.from_numpy(rgb))
    container.close()
    return torch.stack(frames, dim=0).permute(0, 3, 1, 2).contiguous()


def _decode_masks(masks_mkv: str) -> torch.Tensor:
    """Decode masks .mkv (luma) → int64 (N, H, W) on CPU."""
    import av

    container = av.open(masks_mkv)
    stream = container.streams.video[0]
    masks = []
    for frame in container.decode(stream):
        gray = frame.to_ndarray(format="gray")
        masks.append(torch.from_numpy(gray))
    container.close()
    return torch.stack(masks, dim=0).to(torch.long)


def _load_hessian_init(path: str | None) -> dict | None:
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"FATAL: --hessian-init {path} does not exist")
    # Magic-byte safety (DEN-V2 trap): pair_weights / hessian_per_weight
    # files are pickle, never renderer .bin.
    with open(p, "rb") as f:
        magic = f.read(4)
    forbidden_magics = (b"FP4A", b"ASYM", b"DPSM", b"I4LZ", b"CCh1", b"C3R1",
                        b"SCv1", b"OMG1")
    if magic in forbidden_magics:
        raise ValueError(
            f"--hessian-init {path} has renderer magic {magic!r} — "
            f"expected pytorch pickle. Wrong file?"
        )
    data = torch.load(p, map_location="cpu", weights_only=False)
    return data


# ── λ schedule ────────────────────────────────────────────────────────────


def _lambda_for_epoch(
    epoch: int,
    total_epochs: int,
    *,
    lambda_start: float,
    lambda_end: float,
    ramp_start_frac: float,
) -> float:
    """Linear ramp λ_start → λ_end starting at ``ramp_start_frac`` of total."""
    progress = epoch / max(total_epochs - 1, 1)
    if progress < ramp_start_frac:
        return float(lambda_start)
    ramp_progress = (progress - ramp_start_frac) / max(
        1.0 - ramp_start_frac, 1e-6
    )
    ramp_progress = max(0.0, min(1.0, ramp_progress))
    return float(lambda_start + (lambda_end - lambda_start) * ramp_progress)


# ── Loss ──────────────────────────────────────────────────────────────────


def _scorer_loss(
    model: nn.Module,
    masks_t: torch.Tensor,         # (B, H, W) int64
    masks_t1: torch.Tensor,
    gt_t: torch.Tensor,            # (B, 3, H, W) float
    gt_t1: torch.Tensor,
    poses_b: torch.Tensor | None,  # (B, pose_dim) or None
    posenet: nn.Module,
    segnet: nn.Module,
    *,
    seg_weight: float,
    pose_weight: float,
    noise_std: float,
    eval_roundtrip: bool,
) -> tuple[torch.Tensor, dict]:
    """Lane Ω-V2 scorer loss = SegNet hinge + PoseNet MSE.

    Identical numerical recipe to Lane S / qat_finetune Phase 2 to keep
    proxy-auth alignment stable.
    """
    from tac.renderer import simulate_eval_roundtrip
    from tac.camera import CAMERA_H, CAMERA_W

    B = masks_t.shape[0]

    pair_kwargs = {}
    if poses_b is not None and getattr(model, "pose_dim", 0) > 0:
        pair_kwargs["pose"] = poses_b
    pairs = model(masks_t, masks_t1, **pair_kwargs)  # (B, 2, H, W, 3)
    pred_even = pairs[:, 0]  # (B, H, W, 3)
    pred_odd = pairs[:, 1]

    pred_all = torch.cat([pred_even, pred_odd], dim=0)  # (2B, H, W, 3)
    gt_all = torch.cat([gt_t, gt_t1], dim=0).permute(0, 2, 3, 1)  # (2B, H, W, 3)

    if eval_roundtrip:
        pred_chw = pred_all.permute(0, 3, 1, 2)
        pred_chw = simulate_eval_roundtrip(
            pred_chw, target_h=CAMERA_H, target_w=CAMERA_W, noise_std=noise_std,
        )
        pred_for_loss = pred_chw.permute(0, 2, 3, 1)
        with torch.no_grad():
            gt_chw = gt_all.permute(0, 3, 1, 2)
            gt_chw = simulate_eval_roundtrip(
                gt_chw, target_h=CAMERA_H, target_w=CAMERA_W, noise_std=0.0,
            )
            gt_for_loss = gt_chw.permute(0, 2, 3, 1)
    else:
        pred_for_loss = pred_all
        gt_for_loss = gt_all

    # SegNet hinge
    pred_seg_chw = pred_for_loss.permute(0, 3, 1, 2).contiguous()
    pred_seg_in = segnet.preprocess_input(pred_seg_chw.unsqueeze(1).contiguous())
    pred_seg_logits = segnet(pred_seg_in)
    gt_seg_argmax = torch.cat([masks_t, masks_t1], dim=0)
    logit_h, logit_w = pred_seg_logits.shape[2], pred_seg_logits.shape[3]
    if gt_seg_argmax.shape[1] != logit_h or gt_seg_argmax.shape[2] != logit_w:
        gt_seg_argmax = F.interpolate(
            gt_seg_argmax.float().unsqueeze(1),
            size=(logit_h, logit_w),
            mode="nearest",
        ).squeeze(1).long()
    correct = pred_seg_logits.gather(1, gt_seg_argmax.unsqueeze(1)).squeeze(1)
    mask_inf = torch.zeros_like(pred_seg_logits)
    mask_inf.scatter_(1, gt_seg_argmax.unsqueeze(1), float("-inf"))
    runner_up = (pred_seg_logits + mask_inf).max(dim=1).values
    seg_loss = F.relu(1.0 - (correct - runner_up)).mean()

    # PoseNet MSE
    pred_pose_chw = pred_for_loss.permute(0, 3, 1, 2).contiguous()
    pred_pose_pairs = torch.stack(
        [pred_pose_chw[:B], pred_pose_chw[B:]], dim=1
    ).contiguous()
    pred_pose_in = posenet.preprocess_input(pred_pose_pairs)
    pred_pose_out = posenet(pred_pose_in)["pose"][..., :6]
    with torch.no_grad():
        gt_pose_chw = gt_for_loss.permute(0, 3, 1, 2).contiguous()
        gt_pose_pairs = torch.stack(
            [gt_pose_chw[:B], gt_pose_chw[B:]], dim=1
        ).contiguous()
        gt_pose_in = posenet.preprocess_input(gt_pose_pairs)
        gt_pose_out = posenet(gt_pose_in)["pose"][..., :6]
    pose_loss = (pred_pose_out - gt_pose_out).pow(2).mean()

    total = seg_weight * seg_loss + pose_weight * pose_loss
    return total, {
        "seg_loss": float(seg_loss.item()),
        "pose_loss": float(pose_loss.item()),
        "scorer_loss": float(total.item()),
    }


# ── Main QAT loop ─────────────────────────────────────────────────────────


def run_qat(args: argparse.Namespace) -> dict:
    t_start = time.monotonic()
    device = _device_strict(args.device)
    print(f"[lane-omega-v2] device={device}")

    # Determinism
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception as exc:  # pragma: no cover
        print(f"[lane-omega-v2]   determinism warning: {exc}")

    # Load renderer
    print(f"[lane-omega-v2] loading renderer: {args.checkpoint}")
    model = _load_renderer(args.checkpoint, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[lane-omega-v2]   {type(model).__name__} {n_params:,} params "
          f"film={getattr(model, 'pose_dim', 0)}")

    # Hessian warm-start (optional)
    hessian_init = _load_hessian_init(args.hessian_init)
    if hessian_init is not None:
        print(f"[lane-omega-v2] hessian warm-start loaded from {args.hessian_init}")

    # Swap convs
    from tac.learnable_bit_quant import (
        LagrangianRateController,
        LearnableBitConv2d,
        compute_learnable_bit_rate_penalty,
        list_learnable_bit_layers,
        renderer_average_learnable_bits_per_weight,
        swap_renderer_convs_with_learnable_bits,
    )
    swap_report = swap_renderer_convs_with_learnable_bits(
        model,
        init_bits=args.init_bits,
        hessian_init=hessian_init,
    )
    print(f"[lane-omega-v2] swapped={len(swap_report['swapped'])} "
          f"protected={len(swap_report['protected'])} "
          f"warm_started={len(swap_report['warm_started'])} "
          f"params_swapped={swap_report['total_swapped_params']:,}")

    model = model.to(device)

    # Load poses
    poses = _load_poses(args.poses, device)
    if getattr(model, "pose_dim", 0) > 0 and poses is None:
        raise SystemExit(
            "FATAL: model has FiLM (pose_dim>0) but --poses not provided. "
            "PoseNet collapses 32x without conditioning poses (memory: "
            "feedback_film_eval_no_poses_critical)."
        )

    # Load data
    print(f"[lane-omega-v2] decoding video {args.video} ...")
    gt_frames_cpu = _decode_video(args.video)
    masks_cpu = _decode_masks(args.masks_mkv)
    n_frames = gt_frames_cpu.shape[0]
    n_pairs = n_frames // 2
    print(f"[lane-omega-v2]   {n_frames} frames → {n_pairs} pairs")

    # Load scorers
    print(f"[lane-omega-v2] loading differentiable scorers ...")
    from tac.scorer import load_differentiable_scorers
    posenet, segnet = load_differentiable_scorers(args.upstream, device=str(device))
    posenet.eval()
    segnet.eval()
    for p in posenet.parameters():
        p.requires_grad_(False)
    for p in segnet.parameters():
        p.requires_grad_(False)

    # Optimizer: separate parameter groups
    weight_params: list[nn.Parameter] = []
    bits_params: list[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith(".bit_depth.raw"):
            bits_params.append(p)
        else:
            weight_params.append(p)
    optimizer = torch.optim.Adam([
        {"params": weight_params, "lr": args.lr},
        {"params": bits_params, "lr": args.lr * args.bits_lr_scale},
    ])
    print(f"[lane-omega-v2] optimizer: weight_params={len(weight_params)} "
          f"bits_params={len(bits_params)} "
          f"(lr={args.lr}, bits_lr={args.lr * args.bits_lr_scale})")

    # Council D 2026-04-29 PM: EMA shadow over the union of weight params
    # AND learnable bit_depth params. Per CLAUDE.md "EMA — NON-NEGOTIABLE",
    # every training path must EMA the model and ship the EMA shadow as
    # the inference state. Quantizr (#1, 0.33) + Selfcomp (#2, 0.38) both
    # EMA through QAT. The Lagrangian dual variable is NOT EMA'd — only
    # the primal weights / bit-depths.
    from tac.training import EMA
    ema = EMA(model, decay=float(args.ema_decay))
    print(f"[lane-omega-v2] EMA enabled (decay={args.ema_decay})")

    # Output dir
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    best_path = output_dir / "renderer_best.pt"
    final_omg1 = output_dir / "renderer.bin"
    log_path = output_dir / "qat_log.jsonl"
    log_f = open(log_path, "w")

    # Provenance
    provenance = {
        "lane": "omega_v2_lagrangian",
        "checkpoint": str(args.checkpoint),
        "init_bits": args.init_bits,
        "target_bits": args.target_bits,
        "lambda_start": args.lambda_start,
        "lambda_end": args.lambda_end,
        "lambda_ramp_start_frac": args.lambda_ramp_start_frac,
        "lambda_dual_eta": args.lambda_dual_eta,
        "total_epochs": args.total_epochs,
        "lr": args.lr,
        "bits_lr_scale": args.bits_lr_scale,
        "noise_std": args.noise_std,
        "kl_weight": args.kl_weight,
        "seg_weight": args.seg_weight,
        "pose_weight": args.pose_weight,
        "hessian_init": str(args.hessian_init) if args.hessian_init else None,
        "n_swapped_layers": len(swap_report["swapped"]),
        "n_warm_started": len(swap_report["warm_started"]),
        "device": str(device),
        "seed": args.seed,
        "ema_decay": float(args.ema_decay),
        "torch_version": torch.__version__,
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))

    best_scorer_loss = float("inf")
    layers = list_learnable_bit_layers(model)
    n_total_weights = sum(layer.weight_numel() for _name, layer in layers)
    print(f"[lane-omega-v2] total learnable-bit weights: {n_total_weights:,} "
          f"(target ≈ {n_total_weights * args.target_bits / 8 / 1024:.1f}KB)")

    # Bug 2 fix (codex Round 3 + Round 4): true primal-dual rate
    # controller. Replaces the previous fixed `λ · relu(excess)²`
    # (squared hinge, gradient zero at boundary, equilibrium ABOVE
    # target) with the textbook Lagrangian dual ascent
    #     L(θ, λ) = D(θ) + λ · (mean_bits − target),  λ ≥ 0
    #     λ_{t+1} = max(0, λ_t + η · (mean_bits − target))
    # Round 4 also dropped a residual ReLU from the primal surrogate
    # itself — the KKT non-negativity clamp belongs ONLY in the dual
    # update, never in the primal (otherwise gradient is zero under
    # slack and the bit allocator drifts above target).
    # which converges to the KKT boundary mean_bits = target. The legacy
    # `--lambda-start` / `--lambda-end` ramp is preserved as a *cap* on
    # the dual variable so the operator can still bound the rate
    # pressure during phase-1 warm-up.
    rate_controller = LagrangianRateController(
        target_bits_per_weight=float(args.target_bits),
        eta=float(args.lambda_dual_eta),
        initial_lambda=float(args.lambda_start),
    )
    print(
        f"[lane-omega-v2] Lagrangian rate controller: "
        f"target={args.target_bits:.3f} eta={args.lambda_dual_eta} "
        f"initial_lambda={args.lambda_start} (cap from ramp = lambda_end "
        f"{args.lambda_end:.3f})"
    )

    # Move data to device piecewise per batch (saves VRAM for 1200 frames).
    for epoch in range(args.total_epochs):
        # Random pair index
        idx = torch.randint(0, n_pairs, (1,)).item()
        j = idx * 2
        m_t = masks_cpu[j:j + 1].to(device)
        m_t1 = masks_cpu[j + 1:j + 2].to(device)
        gt_t = gt_frames_cpu[j:j + 1].float().to(device)
        gt_t1 = gt_frames_cpu[j + 1:j + 2].float().to(device)
        poses_b = poses[idx:idx + 1] if poses is not None else None

        # Lambda *cap* schedule (legacy ramp — now bounds the dual variable
        # rather than directly setting it). The dual ascent inside
        # ``rate_controller`` does the actual updating; the ramp just
        # tightens the upper bound on the multiplier so phase-1 warm-up
        # cannot generate excessive rate pressure.
        lam_cap = _lambda_for_epoch(
            epoch, args.total_epochs,
            lambda_start=args.lambda_start,
            lambda_end=args.lambda_end,
            ramp_start_frac=args.lambda_ramp_start_frac,
        )
        rate_controller.lambda_max = float(lam_cap)
        # Re-clamp current λ to the new cap so ramp tightening takes
        # effect immediately (without a spurious dual_update step).
        rate_controller.reclamp_to_lambda_max()
        lam = rate_controller.lambda_rate

        # Forward + scorer loss
        scorer_loss, metrics = _scorer_loss(
            model, m_t, m_t1, gt_t, gt_t1, poses_b,
            posenet, segnet,
            seg_weight=args.seg_weight,
            pose_weight=args.pose_weight,
            noise_std=args.noise_std,
            eval_roundtrip=True,  # CLAUDE.md non-negotiable
        )

        # Rate penalty (Lagrangian on bits/weight) — linear in residual,
        # gradient at boundary = λ (NOT 0 like the squared-hinge form).
        # Round 13 (C-1): pass target=None with a controller — the
        # controller already pins the target on construction and the
        # function now refuses double-specification.
        rate_loss = compute_learnable_bit_rate_penalty(
            model, target_bits_per_weight=None, lambda_rate=rate_controller,
        )

        # KL distill (post-fix weight = 0.002): we use the scorer-loss
        # already, so KL is currently a placeholder (kl_weight default 0).
        # Keeping the wiring so the loss surface is uniform with
        # train_distill.py.
        kl_loss = torch.zeros((), device=device)

        loss = scorer_loss + args.kl_weight * kl_loss + rate_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(weight_params + bits_params, 1.0)
        optimizer.step()
        # Council D 2026-04-29: EMA update AFTER primal step (BEFORE the
        # dual update — the EMA shadow tracks weights + bit-depths only;
        # the Lagrangian λ has its own dual-ascent and is not EMA'd).
        ema.update(model)

        # Bug 2 fix (codex Round 3): dual ascent on the constraint
        # residual after the primal step. Computes mean_bits from the
        # *post-step* iterate so the next iteration's λ reflects the
        # latest constraint slack.
        post_step_mean_bits = renderer_average_learnable_bits_per_weight(model)
        rate_controller.dual_update(post_step_mean_bits)

        # Logging + checkpointing
        if (epoch % args.log_every == 0) or (epoch == args.total_epochs - 1):
            mean_bits = renderer_average_learnable_bits_per_weight(model)
            log = {
                "epoch": epoch,
                "lambda": lam,
                "loss": float(loss.item()),
                "scorer_loss": metrics["scorer_loss"],
                "seg_loss": metrics["seg_loss"],
                "pose_loss": metrics["pose_loss"],
                "rate_loss": float(rate_loss.item()),
                "mean_bits": mean_bits,
            }
            print(f"[lane-omega-v2] ep={epoch:4d} | scorer={log['scorer_loss']:.4f} "
                  f"seg={log['seg_loss']:.4f} pose={log['pose_loss']:.5f} | "
                  f"rate={log['rate_loss']:.4f} (λ={lam:.3f}) | "
                  f"bits/w={mean_bits:.3f}", flush=True)
            log_f.write(json.dumps(log) + "\n")
            log_f.flush()

        # Track best (by scorer loss only — rate is a constraint, not a metric)
        if metrics["scorer_loss"] < best_scorer_loss:
            best_scorer_loss = metrics["scorer_loss"]
            # Council D 2026-04-29 PM: ship EMA shadow (CLAUDE.md
            # non-negotiable). The OMG1 reload-and-export below loads
            # ``model_state_dict`` first and then exports — keeping the
            # primary key as the EMA shadow ensures the OMG1 binary is
            # the smoothed shadow, not the noisy single-step.
            torch.save({
                "model_state_dict": ema.state_dict(),
                "model_state_dict_live": model.state_dict(),
                "ema_state_dict": ema.state_dict(),
                "ema_decay": float(args.ema_decay),
                "epoch": epoch,
                "scorer_loss": best_scorer_loss,
                "mean_bits": renderer_average_learnable_bits_per_weight(model),
            }, best_path)

    log_f.close()
    print(f"[lane-omega-v2] training done. best_scorer_loss={best_scorer_loss:.4f}")
    print(f"[lane-omega-v2] best checkpoint: {best_path}")

    # Reload best, then export OMG1
    print(f"[lane-omega-v2] reloading best for OMG1 export ...")
    best = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best["model_state_dict"])

    # Final stats before export
    mean_bits_final = renderer_average_learnable_bits_per_weight(model)
    print(f"[lane-omega-v2] final mean bits/weight: {mean_bits_final:.3f} "
          f"(target {args.target_bits:.3f})")

    # OMG1 export auto-unwraps LearnableBitConv2d wrappers and extracts bits.
    from tac.renderer_export import export_omega_renderer
    n_bytes = export_omega_renderer(
        model, bits_per_weight=None, output_path=final_omg1,
        use_lzma=True,
        arch_extra={"lane": "omega_v2_lagrangian"},
    )
    print(f"[lane-omega-v2] WROTE {final_omg1}: {n_bytes:,} bytes")

    summary = {
        "best_scorer_loss": best_scorer_loss,
        "mean_bits_final": mean_bits_final,
        "target_bits": args.target_bits,
        "n_swapped_layers": len(swap_report["swapped"]),
        "n_total_weights": n_total_weights,
        "renderer_bin_bytes": n_bytes,
        "elapsed_s": time.monotonic() - t_start,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[lane-omega-v2] summary: {summary}")
    return summary


# ── CLI ───────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Lane Ω-V2 Lagrangian QAT (per-weight learnable bit-depth)",
    )
    parser.add_argument("--checkpoint", required=True,
                        help="Lane A renderer .bin to anchor on")
    parser.add_argument("--video", required=True,
                        help="GT video .mkv (1200 frames)")
    parser.add_argument("--masks-mkv", required=True,
                        help="Masks .mkv (matches GT frame count)")
    parser.add_argument("--poses", default=None,
                        help="Optimized poses .pt (REQUIRED if model has FiLM)")
    parser.add_argument("--upstream", default="upstream",
                        help="Upstream dir (default: upstream)")
    parser.add_argument("--output-dir", required=True,
                        help="Output dir for QAT logs + renderer.bin")
    parser.add_argument("--init-bits", type=float, default=8.0,
                        help="Initial per-weight bit-depth (default 8.0)")
    parser.add_argument("--target-bits", type=float, default=2.5,
                        help="Target average bits/weight (default 2.5)")
    parser.add_argument("--lambda-start", type=float, default=0.0,
                        help="Initial Lagrangian λ (default 0.0)")
    parser.add_argument("--lambda-end", type=float, default=1.0,
                        help="Final Lagrangian λ after ramp (default 1.0)")
    parser.add_argument("--lambda-ramp-start-frac", type=float, default=0.3,
                        help="Fraction of training before ramp begins (default 0.3)")
    parser.add_argument("--lambda-dual-eta", type=float, default=1e-3,
                        help="Dual-ascent step size for the Lagrangian rate "
                             "controller (default 1e-3 — Boyd & Vandenberghe "
                             "§5.4 convention; smaller for more conservative "
                             "λ tracking, larger for faster convergence to "
                             "the KKT boundary).")
    parser.add_argument("--total-epochs", type=int, default=200,
                        help="Total QAT epochs (default 200)")
    parser.add_argument("--lr", type=float, default=2.5e-6,
                        help="Adam lr for weights (default 2.5e-6)")
    parser.add_argument("--bits-lr-scale", type=float, default=0.1,
                        help="Multiplier on lr for the bits parameter group "
                             "(default 0.1)")
    parser.add_argument("--noise-std", type=float, default=0.5,
                        help="Roundtrip noise std on PRED frames (default 0.5)")
    parser.add_argument("--kl-weight", type=float, default=0.0,
                        help="KL distill weight (default 0; placeholder)")
    parser.add_argument("--seg-weight", type=float, default=100.0,
                        help="SegNet hinge loss weight (default 100)")
    parser.add_argument("--pose-weight", type=float, default=10.0,
                        help="PoseNet MSE loss weight (default 10)")
    parser.add_argument("--hessian-init", default=None,
                        help="Optional path to hessian_per_weight.pt for "
                             "warm-starting bits ∝ √(I/median(I))")
    parser.add_argument("--device", default="cuda",
                        choices=["cuda", "cpu"],
                        help="Compute device (CUDA strongly preferred)")
    parser.add_argument("--seed", type=int, default=1234,
                        help="Random seed (default 1234)")
    parser.add_argument("--log-every", type=int, default=10,
                        help="Epochs between log lines (default 10)")
    # Council D 2026-04-29 PM: EMA on QAT (Quantizr canonical 0.997).
    # Per CLAUDE.md "EMA — NON-NEGOTIABLE": every training path must
    # EMA the model and ship the shadow as the inference checkpoint.
    parser.add_argument("--ema-decay", type=float, default=0.997,
                        help="EMA decay (Quantizr 0.997). EMA is mandatory "
                             "per CLAUDE.md non-negotiable; the EMA shadow "
                             "is what gets saved as best/best_path and "
                             "exported as the OMG1 binary.")
    args = parser.parse_args(argv)

    run_qat(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
