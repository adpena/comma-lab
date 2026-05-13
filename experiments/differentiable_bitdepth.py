#!/usr/bin/env python3
"""Differentiable bit-depth learning (Fridrich approach).

Learns per-layer optimal bit allocation via Gumbel-softmax over candidate
bit-depths {2, 3, 4, 6, 8}. Joint optimization of scorer loss + rate penalty.

This is the DIFFERENTIABLE approach — learns bit allocation jointly with
task loss, handling inter-layer dependencies that the greedy sweep misses.

Usage:
    PYTHONPATH=src:upstream:$PWD python experiments/differentiable_bitdepth.py \
        --checkpoint /tmp/distill_v2_ep850_best.pt \
        --device mps --steps 20 --n-pairs 10
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "upstream"))


CANDIDATE_BITS = torch.tensor([2.0, 3.0, 4.0, 6.0, 8.0])


def quantize_to_bits_differentiable(tensor: torch.Tensor, bits: float) -> torch.Tensor:
    """Simulate quantization to fractional bits via interpolation (STE)."""
    bits_val = bits.item() if isinstance(bits, torch.Tensor) else bits
    n_levels = 2 ** bits_val
    vmin = tensor.min()
    vmax = tensor.max()
    if (vmax - vmin).item() < 1e-10:
        return tensor
    scale = (vmax - vmin) / (n_levels - 1)
    quantized = ((tensor - vmin) / scale).round().clamp(0, n_levels - 1)
    # STE: forward uses quantized, backward passes through
    return (quantized * scale + vmin - tensor).detach() + tensor


class MixedPrecisionWrapper(nn.Module):
    """Wraps a model with learnable per-layer bit-depth allocation.

    Each parameter group gets a logit vector over candidate bit-depths.
    During forward, Gumbel-softmax selects a bit-depth per layer.
    The rate loss penalizes total bits used.
    """

    def __init__(self, model: nn.Module, tau: float = 1.0):
        super().__init__()
        self.model = model
        self.tau = tau

        # One logit vector per parameter group (initialized uniform)
        self.bit_logits = nn.ParameterDict()
        self.param_names = []
        for name, param in model.named_parameters():
            if param.ndim >= 2:  # skip biases
                safe_name = name.replace(".", "_")
                self.bit_logits[safe_name] = nn.Parameter(
                    torch.zeros(len(CANDIDATE_BITS))  # uniform init
                )
                self.param_names.append((name, safe_name, param.numel()))

    def get_effective_bits(self) -> dict[str, float]:
        """Get current effective bit-depth per layer (soft allocation)."""
        result = {}
        candidates = CANDIDATE_BITS.to(next(self.model.parameters()).device)
        for name, safe_name, _ in self.param_names:
            probs = F.softmax(self.bit_logits[safe_name], dim=0)
            effective_bits = (probs * candidates).sum().item()
            result[name] = effective_bits
        return result

    def get_hard_allocation(self) -> dict[str, int]:
        """Get discrete bit-depth allocation (argmax)."""
        result = {}
        for name, safe_name, _ in self.param_names:
            idx = self.bit_logits[safe_name].argmax().item()
            result[name] = int(CANDIDATE_BITS[idx].item())
        return result

    def compute_rate_loss(self) -> torch.Tensor:
        """Total bits used (differentiable via soft allocation)."""
        candidates = CANDIDATE_BITS.to(next(self.model.parameters()).device)
        total_bits = torch.tensor(0.0, device=candidates.device)
        for name, safe_name, n_params in self.param_names:
            probs = F.softmax(self.bit_logits[safe_name], dim=0)
            effective_bits = (probs * candidates).sum()
            total_bits = total_bits + n_params * effective_bits
        return total_bits

    def apply_quantization(self):
        """Apply quantization via nn.utils.parametrize (preserves autograd graph).

        Unlike .data mutation, parametrize intercepts weight access in forward
        and routes gradients through the Gumbel-softmax bit selection.
        """
        candidates = CANDIDATE_BITS.to(next(self.model.parameters()).device)

        class BitSelectParametrize(nn.Module):
            def __init__(self, logits, tau, training_mode):
                super().__init__()
                self.logits = logits
                self.tau = tau
                self.training_mode = training_mode

            def forward(self, weight):
                if self.training_mode:
                    probs = F.gumbel_softmax(self.logits, tau=self.tau, hard=True)
                else:
                    probs = torch.zeros_like(self.logits)
                    probs[self.logits.argmax()] = 1.0
                selected_bits = (probs * candidates).sum()
                return quantize_to_bits_differentiable(weight.contiguous(), selected_bits)

        for name, safe_name, _ in self.param_names:
            module = dict(self.model.named_modules())[".".join(name.split(".")[:-1])]
            if not nn.utils.parametrize.is_parametrized(module, "weight"):
                nn.utils.parametrize.register_parametrization(
                    module, "weight",
                    BitSelectParametrize(
                        self.bit_logits[safe_name], self.tau, self.training
                    ),
                )
        return {}  # no saved state needed — parametrize handles it

    def restore_weights(self, saved: dict):
        """Remove parametrizations to restore original weights."""
        for name, safe_name, _ in self.param_names:
            module = dict(self.model.named_modules())[".".join(name.split(".")[:-1])]
            if nn.utils.parametrize.is_parametrized(module, "weight"):
                nn.utils.parametrize.remove_parametrizations(module, "weight")


def main():
    parser = argparse.ArgumentParser(description="Differentiable bit-depth learning")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--n-pairs", type=int, default=10)
    parser.add_argument("--rate-lambda", type=float, default=0.001,
                        help="Rate penalty weight (higher = smaller archive)")
    parser.add_argument("--output", default="experiments/results/differentiable_bitdepth.json")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Load model
    from tac.renderer_export import load_any_renderer_checkpoint
    model = load_any_renderer_checkpoint(args.checkpoint, device=str(device))
    if isinstance(model, tuple):
        model = model[0]
    model.eval()

    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

    # Load data
    import av
    video_path = Path("upstream/videos/0.mkv")
    gt_frames = []
    with av.open(str(video_path)) as container:
        for frame in container.decode(container.streams.video[0]):
            gt_frames.append(frame.to_ndarray(format="rgb24"))
            if len(gt_frames) >= args.n_pairs * 2:
                break

    # Load masks
    from tac.mask_codec import decode_masks
    masks_path = Path("submissions/robust_current/masks_crf50.mkv")
    masks = decode_masks(str(masks_path))

    # Load poses
    poses_path = Path("experiments/results/gt_poses.pt")
    poses = torch.load(str(poses_path), map_location="cpu", weights_only=True).float() if poses_path.exists() else None

    # Load scorer
    from modules import DistortionNet
    dn = DistortionNet().eval().to(device)
    dn.load_state_dicts(
        Path("upstream/models/posenet.safetensors"),
        Path("upstream/models/segnet.safetensors"),
        device,
    )

    # Wrap model with mixed precision
    wrapper = MixedPrecisionWrapper(model, tau=1.0)
    wrapper.to(device)

    # Optimizer for bit logits only (not model weights)
    bit_optimizer = torch.optim.Adam(wrapper.bit_logits.parameters(), lr=0.1)

    print(f"\n{'='*60}")
    print("DIFFERENTIABLE BIT-DEPTH LEARNING")
    print(f"  Steps: {args.steps}")
    print(f"  Rate lambda: {args.rate_lambda}")
    print(f"  Candidates: {CANDIDATE_BITS.tolist()}")
    print(f"{'='*60}\n")

    # Initial allocation
    init_bits = wrapper.get_effective_bits()
    print("Initial (uniform):", {k.split(".")[-1]: f"{v:.1f}" for k, v in list(init_bits.items())[:5]}, "...")

    history = []
    for step in range(args.steps):
        bit_optimizer.zero_grad()

        # Apply quantization
        wrapper.train()
        saved = wrapper.apply_quantization()

        # Forward pass on a random pair
        i = step % min(args.n_pairs, len(gt_frames) // 2)
        m_t = masks[2 * i: 2 * i + 1].to(device, dtype=torch.long)
        m_t1 = masks[2 * i + 1: 2 * i + 2].to(device, dtype=torch.long)
        p = poses[i: i + 1].to(device) if poses is not None else None
        kwargs = {"pose": p} if p is not None else {}

        pairs = model(m_t, m_t1, **kwargs)
        chw = pairs[0].permute(0, 3, 1, 2).float().contiguous()
        cam = F.interpolate(chw, size=(874, 1164), mode="bilinear", align_corners=False)
        cam = cam.round().clamp(0, 255)

        gt_p = torch.stack([
            torch.from_numpy(gt_frames[2 * i]).float(),
            torch.from_numpy(gt_frames[2 * i + 1]).float(),
        ]).unsqueeze(0).to(device)
        comp_p = cam.permute(0, 2, 3, 1).unsqueeze(0).contiguous()

        # Scorer distortion
        pd, sd = dn.compute_distortion(gt_p, comp_p)
        distortion_loss = 100 * sd.mean() + torch.sqrt(torch.clamp(10 * pd.mean(), min=1e-8))

        # Rate loss (total bits)
        rate_loss = wrapper.compute_rate_loss()
        total_bits = rate_loss.item()
        total_kb = total_bits / 8 / 1024

        # Combined loss
        loss = distortion_loss + args.rate_lambda * rate_loss

        # Restore weights before backward (STE: gradient to logits only)
        wrapper.restore_weights(saved)

        # Backward through bit logits
        loss.backward()
        bit_optimizer.step()

        if step % 5 == 0 or step == args.steps - 1:
            bits_alloc = wrapper.get_effective_bits()
            min_bits = min(bits_alloc.values())
            max_bits = max(bits_alloc.values())
            print(f"  step {step:>3d} | dist={distortion_loss.item():.3f} | "
                  f"rate={total_kb:.1f}KB | bits=[{min_bits:.1f}, {max_bits:.1f}] | "
                  f"loss={loss.item():.3f}")
            history.append({
                "step": step,
                "distortion": distortion_loss.item(),
                "rate_kb": total_kb,
                "min_bits": min_bits,
                "max_bits": max_bits,
            })

    # Final allocation
    print(f"\n{'='*60}")
    print("LEARNED ALLOCATION (hard argmax):")
    hard = wrapper.get_hard_allocation()
    soft = wrapper.get_effective_bits()
    total_mixed = sum(
        dict(model.named_parameters())[name].numel() * hard[name]
        for name in hard
    )
    total_uniform = sum(
        dict(model.named_parameters())[name].numel() * 4
        for name in hard
    )

    for name in sorted(hard.keys()):
        n = dict(model.named_parameters())[name].numel()
        print(f"  {name}: {hard[name]}b (soft={soft[name]:.1f}b, {n} params)")

    mixed_kb = total_mixed / 8 / 1024
    uniform_kb = total_uniform / 8 / 1024
    print(f"\nUniform FP4: {uniform_kb:.1f} KB")
    print(f"Learned mixed: {mixed_kb:.1f} KB")
    print(f"Savings: {uniform_kb - mixed_kb:.1f} KB ({(1 - mixed_kb/uniform_kb)*100:.1f}%)")

    # Did bit-depths DIVERGE from uniform?
    bits_used = set(hard.values())
    diverged = len(bits_used) > 1
    print(f"\nDivergence: {'YES — approach works!' if diverged else 'NO — stayed uniform'}")
    print(f"Unique bit-depths used: {sorted(bits_used)}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "hard_allocation": hard,
        "soft_allocation": {k: round(v, 2) for k, v in soft.items()},
        "uniform_kb": uniform_kb,
        "mixed_kb": mixed_kb,
        "savings_pct": round((1 - mixed_kb / uniform_kb) * 100, 1),
        "diverged": diverged,
        "history": history,
    }
    output_path.write_text(json.dumps(output_data, indent=2))
    print(f"\nResults saved: {output_path}")


if __name__ == "__main__":
    main()
