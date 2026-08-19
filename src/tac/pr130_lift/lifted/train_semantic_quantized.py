#!/usr/bin/env python3
# borrowed_substrate_accounting:
#   source_repo: /Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo
#   source_repo_head: e34f31bc4969042c0051ac81aa3c56884419a231
#   lifted_at_head: 2f94596bb0136d342254022a5c9584756eae0468
#   source_path: code/train_semantic_quantized.py
#   source_sha256: 4bcaf8a5c581c1e5eb057ea0ef760f269e1eabfdd2fca926bcbf61f4163a248d
#   source_sha256_scope: exact source_path bytes at source_repo_head before adaptations
#   theirs: QAT loop, fake quantization, distillation option, packed-size accounting.
#   ours: governed-admission guard after CLI parsing; scientific mechanism remains PR130-lifted.
#   adaptations: governed_admission_guard_after_argparse
"""Quantization-aware fine-tuning for the compact semantic renderer."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch.func import functional_call

from semantic_renderer_oracle import (
    CAMERA_H,
    CAMERA_W,
    SemanticTokenRenderer,
    curriculum_loss,
    ste_uint8,
)

# fp16's smallest positive (subnormal) value, 2**-24 = 5.960464e-08.  A positive
# floor applied in fp32 BELOW this rounds to EXACTLY 0.0 when narrowed to fp16,
# silently re-opening the divide-by-zero / zero-scale the floor exists to close.
# The floor must therefore be re-applied AFTER the cast.
_FP16_MIN_POSITIVE = 5.960464477539063e-08

N = 600


def fake_quantize(value: torch.Tensor, bits: int, embedding: bool) -> torch.Tensor:
    source = value.float()
    if source.ndim < 2:
        rounded = source.to(torch.float16).float()
        return source + (rounded - source).detach()
    limit = (1 << (bits - 1)) - 1
    if embedding:
        reduce_dims = tuple(range(source.ndim - 1))
    else:
        reduce_dims = tuple(range(1, source.ndim))
    scale = source.detach().abs().amax(
        dim=reduce_dims, keepdim=True
    ).clamp_min(1e-8) / limit
    scale = scale.to(torch.float16).clamp(min=_FP16_MIN_POSITIVE).float()
    normalized = (source / scale).clamp(-limit, limit)
    codes = normalized + (normalized.round() - normalized).detach()
    return codes * scale


def quantized_forward(model, tokens, idx, bits):
    parameters = {
        name: fake_quantize(value, bits, name.endswith("embed.weight"))
        for name, value in model.named_parameters()
    }
    return functional_call(model, parameters, (tokens, idx))


def render_quantized(model, tokens, idx, bits, exact_path):
    frame = quantized_forward(model, tokens, idx, bits)
    if not exact_path:
        return ste_uint8(frame)
    camera = ste_uint8(F.interpolate(
        frame, size=(CAMERA_H, CAMERA_W), mode="bilinear", align_corners=False
    ))
    return F.interpolate(camera, size=(384, 512), mode="bilinear", align_corners=False)


def render_float(model, tokens, idx, exact_path):
    frame = model(tokens, idx)
    if not exact_path:
        return ste_uint8(frame)
    camera = ste_uint8(F.interpolate(
        frame, size=(CAMERA_H, CAMERA_W), mode="bilinear", align_corners=False
    ))
    return F.interpolate(camera, size=(384, 512), mode="bilinear", align_corners=False)


def gather_conditioning(
    conditioning_tokens: torch.Tensor, indices: torch.Tensor, temporal_radius: int
) -> torch.Tensor:
    indices = indices.detach().to(device="cpu", dtype=torch.long)
    if temporal_radius == 0:
        return conditioning_tokens[indices]
    offsets = torch.arange(-temporal_radius, temporal_radius + 1)
    temporal_indices = (indices[:, None] + offsets).clamp(
        0, conditioning_tokens.shape[0] - 1
    )
    return conditioning_tokens[temporal_indices]


@torch.no_grad()
def evaluate_all(
    model, segnet, conditioning_tokens, target_tokens, bits, batch_size, device
) -> float:
    model.eval()
    mismatches = 0
    pixels = 0
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        idx = torch.arange(start, end, device=device)
        conditioning = gather_conditioning(
            conditioning_tokens, idx, model.temporal_radius
        ).to(device)
        target = target_tokens[start:end].to(device)
        frame = render_quantized(model, conditioning, idx, bits, exact_path=True)
        pred = segnet(frame).argmax(1)
        mismatches += int((pred != target).sum())
        pixels += target.numel()
    return mismatches / pixels


@torch.no_grad()
def evaluate_rgb(
    model, conditioning_tokens, master_targets, bits, batch_size, device
) -> float:
    model.eval()
    squared_error = 0.0
    values = 0
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        idx = torch.arange(start, end, device=device)
        conditioning = gather_conditioning(
            conditioning_tokens, idx, model.temporal_radius
        ).to(device)
        frame = render_quantized(model, conditioning, idx, bits, exact_path=True)
        target = F.interpolate(
            master_targets[start:end].to(device=device, dtype=torch.float32),
            size=frame.shape[-2:], mode="bilinear", align_corners=False,
        )
        squared_error += float(((frame - target) / 255.0).square().sum())
        values += frame.numel()
    return squared_error / values


def packed_size(model, bits) -> int:
    size = 0
    for name, value in model.state_dict().items():
        if value.ndim < 2:
            size += value.numel() * 2
        else:
            scales = value.shape[-1] if name.endswith("embed.weight") else value.shape[0]
            size += (value.numel() * bits + 7) // 8 + scales * 2
    return size


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge-root", type=Path, required=True)
    parser.add_argument(
        "--cache", type=Path,
        help="Use one cache for both renderer conditioning and supervision",
    )
    parser.add_argument("--input-cache", type=Path)
    parser.add_argument("--target-cache", type=Path)
    parser.add_argument("--master-cache", type=Path)
    parser.add_argument("--distill-weight", type=float, default=0.0)
    parser.add_argument("--distill-max-seg", type=float, default=4e-4)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument(
        "--float-warmup-steps", type=int, default=0,
        help="adapt in float before switching to exact quantization-aware training",
    )
    parser.add_argument("--ce-fraction", type=float, default=0.50)
    parser.add_argument("--softplus-fraction", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--disable-tf32", action="store_true")
    parser.add_argument("--fixed-zero-mask", action="store_true")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--save", type=Path, required=True)
    args = parser.parse_args()
    from tac.admission_guard import assert_governed_admission
    assert_governed_admission("train_semantic_quantized")
    if not 2 <= args.bits <= 8:
        raise ValueError("--bits must be in [2,8]")
    if not 0 <= args.float_warmup_steps < args.steps:
        raise ValueError("--float-warmup-steps must be in [0, steps)")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if args.disable_tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    root = args.challenge_root.resolve()
    sys.path.insert(0, str(root))
    import modules  # pylint: disable=import-error,import-outside-toplevel

    input_cache = args.input_cache or args.cache
    target_cache = args.target_cache or args.cache
    if input_cache is None or target_cache is None:
        parser.error("provide --cache or both --input-cache and --target-cache")
    conditioning_tokens = torch.load(
        input_cache, map_location="cpu", weights_only=False
    )["seg"].long()
    target_tokens = torch.load(
        target_cache, map_location="cpu", weights_only=False
    )["seg"].long()
    master_targets = None
    if args.master_cache is not None:
        master_targets = torch.load(
            args.master_cache, map_location="cpu", weights_only=False
        )["masters"].to(torch.uint8)
    checkpoint = torch.load(args.init, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = SemanticTokenRenderer(
        width=int(config["width"]), blocks=int(config["blocks"]),
        frame_dim=int(config["frame_dim"]), num_pairs=N,
        phase_y=int(config.get("phase_y", 1)),
        phase_x=int(config.get("phase_x", 1)),
        temporal_radius=int(config.get("temporal_radius", 0)),
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    zero_masks = {
        name: (value == 0).to(device)
        for name, value in checkpoint["state_dict"].items()
        if args.fixed_zero_mask and value.ndim >= 2
    }
    segnet = modules.SegNet().eval().to(device)
    segnet.load_state_dict(load_file(modules.segnet_sd_path, device=str(device)))
    for parameter in segnet.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.lr * 0.01
    )
    generator = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(N, generator=generator)
    cursor = 0
    best_seg = evaluate_all(
        model, segnet, conditioning_tokens, target_tokens,
        args.bits, args.eval_batch_size, device,
    )
    best_rgb = (
        evaluate_rgb(
            model, conditioning_tokens, master_targets,
            args.bits, args.eval_batch_size, device,
        )
        if master_targets is not None else None
    )
    best_key = (
        (
            best_seg > args.distill_max_seg,
            best_seg if best_seg > args.distill_max_seg else best_rgb,
            best_rgb if best_seg > args.distill_max_seg else best_seg,
        )
        if best_rgb is not None else (best_seg,)
    )
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    history = [{
        "step": 0, "quantized_exact_seg": best_seg,
        "normalized_rgb_mse": best_rgb,
    }]
    print(json.dumps(history[-1]), flush=True)

    for step in range(1, args.steps + 1):
        if cursor + args.batch_size > N:
            order = torch.randperm(N, generator=generator)
            cursor = 0
        batch_ids_cpu = order[cursor:cursor + args.batch_size]
        cursor += args.batch_size
        idx = batch_ids_cpu.to(device)
        conditioning = gather_conditioning(
            conditioning_tokens, batch_ids_cpu, model.temporal_radius
        ).to(device)
        target = target_tokens[batch_ids_cpu].to(device)
        model.train()
        in_float_warmup = step <= args.float_warmup_steps
        frame = (
            render_float(model, conditioning, idx, exact_path=True)
            if in_float_warmup
            else render_quantized(
                model, conditioning, idx, args.bits, exact_path=True
            )
        )
        logits = segnet(frame)
        if in_float_warmup:
            segmentation_loss = F.cross_entropy(logits, target)
            phase = "float_ce"
        else:
            qat_steps = args.steps - args.float_warmup_steps
            segmentation_loss, phase = curriculum_loss(
                logits, target, step - args.float_warmup_steps - 1,
                qat_steps, args.ce_fraction, args.softplus_fraction,
            )
        distill_loss = torch.zeros((), device=device)
        if master_targets is not None:
            master = F.interpolate(
                master_targets[batch_ids_cpu].to(
                    device=device, dtype=torch.float32
                ),
                size=frame.shape[-2:], mode="bilinear", align_corners=False,
            )
            distill_loss = F.mse_loss(frame / 255.0, master / 255.0)
        loss = segmentation_loss + args.distill_weight * distill_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is not None and name in zero_masks:
                parameter.grad.masked_fill_(zero_masks[name], 0.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        if zero_masks:
            with torch.no_grad():
                for name, parameter in model.named_parameters():
                    if name in zero_masks:
                        parameter.masked_fill_(zero_masks[name], 0.0)
        scheduler.step()

        if step % args.eval_every == 0 or step == args.steps:
            exact = evaluate_all(
                model, segnet, conditioning_tokens, target_tokens,
                args.bits, args.eval_batch_size, device,
            )
            rgb_mse = (
                evaluate_rgb(
                    model, conditioning_tokens, master_targets,
                    args.bits, args.eval_batch_size, device,
                )
                if master_targets is not None else None
            )
            candidate_key = (
                (
                    exact > args.distill_max_seg,
                    exact if exact > args.distill_max_seg else rgb_mse,
                    rgb_mse if exact > args.distill_max_seg else exact,
                )
                if rgb_mse is not None else (exact,)
            )
            if candidate_key < best_key:
                best_key = candidate_key
                best_seg = exact
                best_rgb = rgb_mse
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
            record = {
                "step": step,
                "phase": phase,
                "loss": float(loss.detach()),
                "segmentation_loss": float(segmentation_loss.detach()),
                "distill_loss": float(distill_loss.detach()),
                "quantized_exact_seg": exact,
                "best_quantized_exact_seg": best_seg,
                "normalized_rgb_mse": rgb_mse,
                "best_normalized_rgb_mse": best_rgb,
                "lr": optimizer.param_groups[0]["lr"],
            }
            history.append(record)
            print(json.dumps(record), flush=True)

    model.load_state_dict(best_state)
    final_seg = evaluate_all(
        model, segnet, conditioning_tokens, target_tokens,
        args.bits, args.eval_batch_size, device,
    )
    result = {
        "verdict": "PASS" if final_seg < 4e-4 else "FAIL",
        "config": {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()},
        "quantized_exact_seg": final_seg,
        "normalized_rgb_mse": best_rgb,
        "packed_parameter_bytes": packed_size(model, args.bits),
        "history": history,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    args.save.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state, "config": config, "quant_bits": args.bits,
        "best_exact_seg": final_seg, "result": result,
    }, args.save)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
