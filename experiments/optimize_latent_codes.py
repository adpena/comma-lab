#!/usr/bin/env python3
"""Per-Pair Latent Code Optimization (Phase 4 of distillation).

After distillation produces a good renderer, freeze it. Add a learned embedding
table (600 x 16D). For each pair, optimize z_i via gradient descent through
the frozen renderer + frozen scorers. The latent modulates the renderer via
FiLM conditioning (additive bias injection after bottleneck).

The key insight: the renderer has a single set of weights for ALL 600 pairs.
A 16D per-pair latent code gives the renderer pair-specific information that
the global weights cannot capture — analogous to AdaIN in style transfer or
NeRF appearance embeddings.

Archive cost: 600 × 16 × 1 byte (int8) = 9.6KB — negligible rate impact.

Usage:
    # Smoke test (local MPS):
    PYTHONPATH=src:upstream python experiments/optimize_latent_codes.py \
        --device mps --smoke

    # Full run (Vast.ai 4090):
    PYTHONPATH=src:upstream python experiments/optimize_latent_codes.py \
        --device cuda --steps 500 \
        --checkpoint experiments/results/v5_lagrangian_renderer/renderer_best.pt

    # Resume from partial optimization:
    PYTHONPATH=src:upstream python experiments/optimize_latent_codes.py \
        --device cuda --resume experiments/results/latent_codes/latent_codes_partial.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_CANDIDATE_UPSTREAM = [
    Path(os.environ["TAC_UPSTREAM_DIR"]) if os.environ.get("TAC_UPSTREAM_DIR") else None,
    Path(os.environ["UPSTREAM_ROOT"]) if os.environ.get("UPSTREAM_ROOT") else None,
    Path("/kaggle/working/upstream"),
    Path(__file__).resolve().parent.parent / "upstream",
]
UPSTREAM_ROOT: Path | None = None
for _p in _CANDIDATE_UPSTREAM:
    if _p is not None and (_p / "modules.py").exists():
        UPSTREAM_ROOT = _p
        break
if UPSTREAM_ROOT is not None and str(UPSTREAM_ROOT) not in sys.path:
    sys.path.insert(0, str(UPSTREAM_ROOT))

RESULTS_DIR = (
    Path(os.environ["TAC_RESULTS_DIR"])
    if os.environ.get("TAC_RESULTS_DIR")
    else Path(__file__).resolve().parent / "results" / "latent_codes"
)


# ---------------------------------------------------------------------------
# Latent-conditioned renderer wrapper
# ---------------------------------------------------------------------------
class LatentConditionedRenderer(nn.Module):
    """Wraps a frozen MaskRenderer with per-pair latent code injection.

    The latent code is injected via a learned linear projection into the
    bottleneck feature space (additive bias, same mechanism as FiLM beta).
    This is lightweight: only the latent codes and projector are trainable.

    Args:
        renderer: frozen MaskRenderer
        n_pairs: number of pairs (600 for comma challenge)
        latent_dim: dimensionality of per-pair latent code
        inject_ch: channel dimension at injection point (renderer.mid_ch)
    """

    def __init__(
        self,
        renderer: nn.Module,
        n_pairs: int = 600,
        latent_dim: int = 16,
        inject_ch: int = 60,
    ):
        super().__init__()
        self.renderer = renderer
        self.latent_table = nn.Embedding(n_pairs, latent_dim)
        self.projector = nn.Linear(latent_dim, inject_ch)

        # Init: zero projection so renderer starts unchanged
        nn.init.zeros_(self.projector.weight)
        nn.init.zeros_(self.projector.bias)
        # Small random latent init for diversity
        nn.init.normal_(self.latent_table.weight, std=0.01)

        # Freeze renderer
        for p in self.renderer.parameters():
            p.requires_grad = False

    @property
    def latent_dim(self) -> int:
        return self.latent_table.embedding_dim

    @property
    def n_pairs(self) -> int:
        return self.latent_table.num_embeddings

    def forward(
        self,
        masks: torch.Tensor,
        pair_idx: int | torch.Tensor,
        pose: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Render with latent code modulation.

        Args:
            masks: (B, H, W) long tensor of class indices
            pair_idx: scalar or (B,) tensor of pair indices
            pose: optional (B, pose_dim) pose vectors for FiLM

        Returns:
            (B, 3, H, W) float tensor in [0, 255]
        """
        # Get latent code
        if isinstance(pair_idx, int):
            idx_t = torch.tensor([pair_idx], device=masks.device).expand(masks.shape[0])
        else:
            idx_t = pair_idx

        z = self.latent_table(idx_t)  # (B, latent_dim)
        bias = self.projector(z)  # (B, inject_ch)

        # Run renderer forward with hook injection
        # We intercept after bottleneck by temporarily modifying the bottleneck output
        return self._forward_with_latent(masks, bias, pose)

    def _forward_with_latent(
        self,
        masks: torch.Tensor,
        bias: torch.Tensor,
        pose: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with latent bias injected after bottleneck."""
        r = self.renderer

        # Embed masks
        x = r.embedding(masks)
        x = x.permute(0, 3, 1, 2).contiguous()

        # Coordinate grid
        if r.use_coord_grid:
            B, _, H, W = x.shape
            gy = torch.linspace(-1, 1, H, device=x.device, dtype=x.dtype)
            gx = torch.linspace(-1, 1, W, device=x.device, dtype=x.dtype)
            grid_y, grid_x = torch.meshgrid(gy, gx, indexing="ij")
            coords = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0).expand(B, -1, -1, -1)
            x = torch.cat([x, coords], dim=1)

        # Stem
        stem = r.stem_conv(x)
        stem = r.stem_res(stem, masks)

        # Down 1
        down1 = r.down_conv(stem)
        down1 = r.down_res(down1, masks)

        # Bottleneck (depth=1 path)
        if r.depth >= 2:
            down2 = r.down2_conv(down1)
            down2 = r.down2_res(down2, masks)
            mid = r.bottleneck(down2, masks)
            up2 = r.up2_conv(mid)
            if up2.shape[2:] != down1.shape[2:]:
                up2 = F.interpolate(up2, size=down1.shape[2:], mode="bilinear", align_corners=False)
            up2 = r.up2_res(up2, masks)
            fused2 = torch.cat([down1, up2], dim=1)
            half_res = r.fuse2_conv(fused2)
        else:
            half_res = r.bottleneck(down1, masks)

        # === INJECT LATENT BIAS HERE ===
        # bias: (B, inject_ch) → (B, inject_ch, 1, 1) for broadcasting
        half_res = half_res + bias.unsqueeze(-1).unsqueeze(-1)

        # FiLM on bottleneck (if pose available)
        if r.film_bottleneck is not None and pose is not None:
            half_res = r.film_bottleneck(half_res, pose)

        # Up 1
        up = r.up_conv(half_res)
        if up.shape[2:] != stem.shape[2:]:
            up = F.interpolate(up, size=stem.shape[2:], mode="bilinear", align_corners=False)
        up = r.up_res(up, masks)

        # Skip + fuse
        fused = torch.cat([stem, up], dim=1)
        fused = r.fuse_conv(fused)

        # FiLM on decoder (if pose available)
        if r.film_decoder is not None and pose is not None:
            fused = r.film_decoder(fused, pose)

        # Head
        rgb = 255.0 * torch.sigmoid(r.head(fused) / 50.0)
        return rgb

    def export_latent_table_int8(self) -> bytes:
        """Quantize latent table to int8 for archive storage.

        Returns:
            Raw bytes of int8 tensor (n_pairs * latent_dim bytes).
        """
        table = self.latent_table.weight.detach().cpu()
        # Per-row quantization: scale each latent to [-127, 127]
        scales = table.abs().max(dim=1, keepdim=True).values.clamp(min=1e-8)
        quantized = (table / scales * 127.0).round().clamp(-127, 127).to(torch.int8)
        # Store scales as fp16 for dequant
        return {
            "codes_int8": quantized,
            "scales_fp16": scales.squeeze(1).half(),
            "projector_state": {k: v.cpu() for k, v in self.projector.state_dict().items()},
        }


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------
def optimize_latent_codes(
    model: LatentConditionedRenderer,
    gt_frames: list[torch.Tensor],
    gt_masks: list[torch.Tensor],
    posenet: nn.Module,
    segnet: nn.Module,
    device: torch.device,
    steps: int = 500,
    lr: float = 0.01,
    seg_weight: float = 100.0,
    pose_weight: float = 10.0,
    rate_penalty: float = 0.001,
    eval_roundtrip: bool = True,
    output_dir: Path | None = None,
) -> dict:
    """Optimize latent codes for all pairs via gradient descent.

    Args:
        model: LatentConditionedRenderer (renderer frozen, latents trainable)
        gt_frames: list of (H, W, 3) uint8 tensors (1200 frames)
        gt_masks: list of (H, W) long tensors (1200 frames, argmax of SegNet)
        posenet: frozen PoseNet
        segnet: frozen SegNet
        device: compute device
        steps: optimization steps per pair
        lr: learning rate for latent codes
        seg_weight: SegNet loss weight
        pose_weight: PoseNet loss weight
        rate_penalty: L2 penalty on latent magnitude (discourages large codes)
        eval_roundtrip: simulate contest eval resize chain (384→874→uint8→384) before scoring
        output_dir: save intermediate results

    Returns:
        dict with per-pair metrics and final statistics
    """
    from tac.losses import scorer_forward_pair, _hwc_to_chw

    n_pairs = len(gt_frames) // 2
    assert n_pairs == model.n_pairs, f"Expected {model.n_pairs} pairs, got {n_pairs}"

    # Optimizer: only latent table and projector weights
    optimizer = torch.optim.Adam(
        [model.latent_table.weight, *model.projector.parameters()],
        lr=lr,
    )

    # Pre-cache GT scorer outputs (frozen, constant)
    print(f"[latent] Pre-caching GT scorer outputs for {n_pairs} pairs...")
    gt_pose_cache = []
    gt_seg_cache = []
    for pair_idx in range(n_pairs):
        f0 = gt_frames[pair_idx * 2].float().to(device)
        f1 = gt_frames[pair_idx * 2 + 1].float().to(device)
        pair_hwc = torch.stack([f0, f1], dim=0).unsqueeze(0)  # (1,2,H,W,3)
        pair_chw = pair_hwc.permute(0, 1, 4, 2, 3).contiguous()

        with torch.no_grad():
            gp_out, gs_out = scorer_forward_pair(pair_chw, posenet, segnet)
            gt_pose_cache.append(gp_out["pose"][..., :6].cpu())
            gt_seg_cache.append(F.softmax(gs_out, dim=1).cpu())
    print(f"[latent] GT cache complete.")

    # Optimization loop
    results = {"per_pair": [], "total_time": 0.0}
    t0 = time.time()

    for pair_idx in range(n_pairs):
        pair_t0 = time.time()

        # Get masks for this pair
        m0 = gt_masks[pair_idx * 2].to(device)
        m1 = gt_masks[pair_idx * 2 + 1].to(device)
        masks_batch = torch.stack([m0, m1], dim=0)  # (2, H, W)

        # GT targets: recompute through roundtrip when eval_roundtrip is on
        if eval_roundtrip:
            from tac.renderer import simulate_eval_roundtrip
            from tac.camera import CAMERA_H, CAMERA_W
            f0 = gt_frames[pair_idx * 2].float().to(device)
            f1 = gt_frames[pair_idx * 2 + 1].float().to(device)
            gt_chw = torch.stack([f0, f1], dim=0).permute(0, 3, 1, 2).contiguous()
            gt_chw = simulate_eval_roundtrip(gt_chw, target_h=CAMERA_H, target_w=CAMERA_W, noise_std=0.0)
            gt_pair_for_scorer = gt_chw.unsqueeze(0)
            with torch.no_grad():
                gp_gt, gs_gt = scorer_forward_pair(gt_pair_for_scorer, posenet, segnet)
            gt_pose_6 = gp_gt["pose"][..., :6]
            gt_seg_soft = F.softmax(gs_gt, dim=1)
        else:
            gt_pose_6 = gt_pose_cache[pair_idx].to(device)
            gt_seg_soft = gt_seg_cache[pair_idx].to(device)

        best_loss = float("inf")
        best_step = 0

        for step in range(steps):
            optimizer.zero_grad()

            # Render with latent modulation
            rgb = model(masks_batch, pair_idx)  # (2, 3, H, W)

            # eval_roundtrip: simulate contest eval resize chain (384→874→uint8→384)
            if eval_roundtrip:
                from tac.renderer import simulate_eval_roundtrip
                from tac.camera import CAMERA_H, CAMERA_W
                rgb_rt = simulate_eval_roundtrip(
                    rgb, target_h=CAMERA_H, target_w=CAMERA_W, noise_std=0.5,
                )
            else:
                rgb_rt = rgb

            # Build pair in BTCHW format for scorers
            pair_chw = rgb_rt.unsqueeze(0)  # (1, 2, C, H, W)

            # Score
            fp_out, fs_out = scorer_forward_pair(pair_chw, posenet, segnet)
            pose_dist = (fp_out["pose"][..., :6] - gt_pose_6).pow(2).mean()
            pred_soft = F.softmax(fs_out, dim=1)
            seg_dist = 1.0 - (pred_soft * gt_seg_soft).sum(dim=1).mean()

            # Loss: scorer formula + rate penalty
            loss = (
                seg_weight * seg_dist
                + pose_weight * torch.sqrt(pose_dist + 1e-8)
                + rate_penalty * model.latent_table.weight[pair_idx].pow(2).sum()
            )

            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_step = step

        pair_time = time.time() - pair_t0
        results["per_pair"].append({
            "pair_idx": pair_idx,
            "best_loss": best_loss,
            "best_step": best_step,
            "time_s": pair_time,
            "final_pose_dist": pose_dist.item(),
            "final_seg_dist": seg_dist.item(),
        })

        if pair_idx % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (pair_idx + 1) * (n_pairs - pair_idx - 1)
            print(
                f"[latent] Pair {pair_idx}/{n_pairs} | "
                f"loss={best_loss:.4f} pose={pose_dist.item():.6f} seg={seg_dist.item():.4f} | "
                f"ETA: {eta/60:.1f}min"
            )

    results["total_time"] = time.time() - t0

    # Save results
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        archive = model.export_latent_table_int8()
        torch.save(archive, output_dir / "latent_codes_int8.pt")
        torch.save(model.state_dict(), output_dir / "latent_model_full.pt")

        # Archive size accounting
        codes_bytes = archive["codes_int8"].numel()
        scales_bytes = archive["scales_fp16"].numel() * 2
        proj_bytes = sum(v.numel() * v.element_size() for v in archive["projector_state"].values())
        results["archive_bytes"] = codes_bytes + scales_bytes + proj_bytes
        results["archive_breakdown"] = {
            "codes_int8": codes_bytes,
            "scales_fp16": scales_bytes,
            "projector": proj_bytes,
        }

        with open(output_dir / "results.json", "w") as f:
            json.dump(results, f, indent=2)
        print(f"[latent] Saved to {output_dir}")
        print(f"[latent] Archive size: {results['archive_bytes']} bytes ({results['archive_bytes']/1024:.1f}KB)")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-pair latent code optimization (Phase 4 of distillation)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--device", type=str, default="cuda", choices=["cuda", "mps", "cpu"])
    p.add_argument("--checkpoint", type=str, default=None,
                   help="Path to distilled renderer checkpoint")
    p.add_argument("--steps", type=int, default=500, help="Optimization steps per pair")
    p.add_argument("--lr", type=float, default=0.01, help="Learning rate for latent codes")
    p.add_argument("--latent-dim", type=int, default=16, help="Latent code dimensionality")
    p.add_argument("--seg-weight", type=float, default=100.0, help="SegNet loss weight")
    p.add_argument("--pose-weight", type=float, default=10.0, help="PoseNet loss weight")
    p.add_argument("--rate-penalty", type=float, default=0.001,
                   help="L2 penalty on latent magnitude (controls archive size)")
    # CLAUDE.md non-negotiable: eval_roundtrip ALWAYS True. Removed
    # `--no-eval-roundtrip` flag; only escape hatch is TAC_ALLOW_NO_ROUNDTRIP=1.
    p.add_argument("--eval-roundtrip", action="store_true", default=True,
                   help="Simulate contest eval resize chain in scorer loss. "
                        "ALWAYS True; disabling requires TAC_ALLOW_NO_ROUNDTRIP=1.")
    p.add_argument("--upstream", type=str, default=None, help="Path to upstream repo")
    p.add_argument("--video", type=str, default=None, help="Path to GT video")
    p.add_argument("--output-dir", type=str, default=None, help="Output directory")
    p.add_argument("--resume", type=str, default=None, help="Resume from partial .pt file")
    p.add_argument("--smoke", action="store_true", help="Smoke test: 10 pairs, 50 steps")
    return p.parse_args()


def _enforce_eval_roundtrip(args) -> None:
    """CLAUDE.md non-negotiable: eval_roundtrip ALWAYS True; only escape hatch
    is TAC_ALLOW_NO_ROUNDTRIP=1 env var with loud banner.

    2026-04-27 codex R5-4 #4: delegated to the centralised
    `tac.eval_roundtrip_gate.enforce_eval_roundtrip` helper. The previous
    per-script copies were sticky — they only printed the warning when
    `args.eval_roundtrip` was already False, so a leftover env var in a
    shell / tmux session silently relaxed later runs without acknowledgement.
    The centralised helper warns whenever the env var is present and
    records it in run provenance.
    """
    from tac.eval_roundtrip_gate import enforce_eval_roundtrip
    output_dir = getattr(args, "output_dir", None)
    enforce_eval_roundtrip(args, output_dir=output_dir, write_provenance=output_dir is not None)


def main() -> None:
    args = parse_args()

    if args.smoke:
        args.steps = 50
        n_frames = 20
    else:
        n_frames = 1200

    device = torch.device(args.device)

    # Resolve paths
    from tac.utils import find_project_root
    root = find_project_root()
    upstream = Path(args.upstream) if args.upstream else root / "upstream"
    video_path = Path(args.video) if args.video else upstream / "videos" / "0.mkv"
    output_dir = Path(args.output_dir) if args.output_dir else RESULTS_DIR
    # codex R5-r6 #3: gate AFTER output_dir resolution so sidecar lands.
    if args.output_dir is None:
        args.output_dir = str(output_dir)
    _enforce_eval_roundtrip(args)
    checkpoint = Path(args.checkpoint) if args.checkpoint else (
        root / "experiments" / "results" / "v5_lagrangian_renderer" / "renderer_best.pt"
    )

    # Verify checkpoint identity
    from tac.checkpoint import verify_checkpoint_identity
    verify_checkpoint_identity(str(checkpoint))

    # Load renderer
    from tac.renderer import MaskRenderer
    print(f"[latent] Loading renderer from {checkpoint}")
    state = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
    # Detect architecture params from state dict
    if isinstance(state, dict) and "model_state_dict" in state:
        renderer_state = state["model_state_dict"]
        config = state.get("config", {})
    else:
        renderer_state = state
        config = {}

    renderer = MaskRenderer(
        embed_dim=config.get("embed_dim", 6),
        base_ch=config.get("base_ch", 36),
        mid_ch=config.get("mid_ch", 60),
        depth=config.get("depth", 1),
        pose_dim=config.get("pose_dim", 0),
    )
    renderer.load_state_dict(renderer_state, strict=False)
    renderer = renderer.to(device).eval()

    # Build latent-conditioned wrapper
    n_pairs = n_frames // 2
    model = LatentConditionedRenderer(
        renderer=renderer,
        n_pairs=n_pairs,
        latent_dim=args.latent_dim,
        inject_ch=config.get("mid_ch", 60),
    ).to(device)

    if args.resume:
        print(f"[latent] Resuming from {args.resume}")
        resume_state = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(resume_state, strict=False)

    # Load scorers
    from tac.scorer import load_differentiable_scorers
    posenet, segnet = load_differentiable_scorers(str(upstream), device=device)

    # Decode video and extract masks
    from tac.data import decode_video
    from tac.scorer import extract_gt_masks
    print(f"[latent] Decoding video: {video_path}")
    gt_frames = decode_video(str(video_path))[:n_frames]
    print(f"[latent] Extracting SegNet masks for {len(gt_frames)} frames...")
    gt_masks = extract_gt_masks(gt_frames, segnet, device=device)

    # Run optimization
    results = optimize_latent_codes(
        model=model,
        gt_frames=gt_frames,
        gt_masks=gt_masks,
        posenet=posenet,
        segnet=segnet,
        device=device,
        steps=args.steps,
        lr=args.lr,
        seg_weight=args.seg_weight,
        pose_weight=args.pose_weight,
        rate_penalty=args.rate_penalty,
        eval_roundtrip=args.eval_roundtrip,
        output_dir=output_dir,
    )

    print(f"\n{'='*60}")
    print(f"[latent] COMPLETE in {results['total_time']/60:.1f} min")
    if "archive_bytes" in results:
        print(f"[latent] Archive cost: {results['archive_bytes']/1024:.1f} KB")
    mean_pose = sum(r["final_pose_dist"] for r in results["per_pair"]) / len(results["per_pair"])
    mean_seg = sum(r["final_seg_dist"] for r in results["per_pair"]) / len(results["per_pair"])
    print(f"[latent] Mean PoseNet dist: {mean_pose:.6f}")
    print(f"[latent] Mean SegNet dist:  {mean_seg:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
