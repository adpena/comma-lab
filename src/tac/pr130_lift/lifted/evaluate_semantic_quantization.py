#!/usr/bin/env python3
# borrowed_substrate_accounting:
#   source_repo: /Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo
#   source_repo_head: e34f31bc4969042c0051ac81aa3c56884419a231
#   lifted_at_head: 2f94596bb0136d342254022a5c9584756eae0468
#   source_path: code/evaluate_semantic_quantization.py
#   source_sha256: 5bbd2136174bfa2c99219d73f45103d4293f60e3f4eced5ee188e38053923962
#   source_sha256_scope: exact source_path bytes at source_repo_head before adaptations
#   theirs: Deployment-shaped semantic renderer quantization evaluator.
#   ours: accounting header only; mechanism is intentionally PR130-lifted.
#   adaptations: none
"""Measure exact semantic error after deployment-shaped weight quantization."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from semantic_renderer_oracle import SemanticTokenRenderer, render_for_seg



# fp16's smallest positive (subnormal) value, 2**-24 = 5.960464e-08.  A positive
# floor applied in fp32 BELOW this rounds to EXACTLY 0.0 when narrowed to fp16,
# silently re-opening the divide-by-zero / zero-scale the floor exists to close.
# The floor must therefore be re-applied AFTER the cast.
_FP16_MIN_POSITIVE = 5.960464477539063e-08

def quantize_tensor(value: torch.Tensor, bits: int, embedding: bool) -> tuple[torch.Tensor, int]:
    limit = (1 << (bits - 1)) - 1
    source = value.detach().float()
    if source.ndim < 2:
        stored = source.to(torch.float16)
        return stored.float(), stored.numel() * stored.element_size()
    if embedding:
        reduce_dims = tuple(range(source.ndim - 1))
        scale = source.abs().amax(dim=reduce_dims, keepdim=True).clamp_min(1e-8) / limit
    else:
        reduce_dims = tuple(range(1, source.ndim))
        scale = source.abs().amax(dim=reduce_dims, keepdim=True).clamp_min(1e-8) / limit
    scale = scale.to(torch.float16).clamp(min=_FP16_MIN_POSITIVE)
    q = (source / scale.float()).round().clamp(-limit, limit)
    restored = q * scale.float()
    code_bytes = (source.numel() * bits + 7) // 8
    return restored, code_bytes + scale.numel() * scale.element_size()


@torch.no_grad()
def evaluate(model, segnet, tokens, pair_ids, batch_size, device) -> float:
    mismatches = 0
    pixels = 0
    model.eval()
    for start in range(0, len(pair_ids), batch_size):
        selected = pair_ids[start:start + batch_size]
        idx = torch.tensor(selected, dtype=torch.long, device=device)
        target = tokens[selected].to(device)
        frame = render_for_seg(model, target, idx, exact_path=True)
        pred = segnet(frame).argmax(1)
        mismatches += int((pred != target).sum())
        pixels += target.numel()
    return mismatches / pixels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bits", type=int, nargs="+", default=[4, 5, 6, 8])
    parser.add_argument("--pairs", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device(args.device)
    root = args.challenge_root.resolve()
    sys.path.insert(0, str(root))
    import modules  # pylint: disable=import-error,import-outside-toplevel

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    tokens = torch.load(args.cache, map_location="cpu", weights_only=False)["seg"].long()
    pair_ids = np.linspace(0, 599, args.pairs, dtype=np.int64).tolist()
    segnet = modules.SegNet().eval().to(device)
    segnet.load_state_dict(load_file(modules.segnet_sd_path, device=str(device)))

    results = []
    for bits in args.bits:
        if not 2 <= bits <= 8:
            raise ValueError("quantization bits must be in [2,8]")
        model = SemanticTokenRenderer(
            width=int(config["width"]), blocks=int(config["blocks"]),
            frame_dim=int(config["frame_dim"]), num_pairs=600,
        ).to(device)
        quantized = {}
        payload_bytes = 0
        for name, value in checkpoint["state_dict"].items():
            restored, size = quantize_tensor(
                value, bits, embedding=name.endswith("embed.weight")
            )
            quantized[name] = restored
            payload_bytes += size
        model.load_state_dict(quantized)
        seg = evaluate(model, segnet, tokens, pair_ids, args.batch_size, device)
        record = {
            "bits": bits,
            "pairs": len(pair_ids),
            "exact_seg": seg,
            "packed_parameter_bytes": payload_bytes,
        }
        results.append(record)
        print(json.dumps(record), flush=True)

    result = {
        "checkpoint": str(args.checkpoint),
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
