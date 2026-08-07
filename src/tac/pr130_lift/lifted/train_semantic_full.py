#!/usr/bin/env python3
# borrowed_substrate_accounting:
#   source_repo: /Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo
#   source_head: 2f94596bb0136d342254022a5c9584756eae0468
#   source_path: code/train_semantic_full.py
#   source_sha256: 2d7a3575e422dc2b5823b97e52101ad5632e5fa04a98e0af8a83d85b7c2176b8
#   theirs: Full semantic renderer trainer loop, scheduler, selection, CLI.
#   ours: This accounting header only; mechanism is intentionally PR130-lifted.
"""Train the compact semantic renderer on all 600 token maps."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from semantic_renderer_oracle import (
    SemanticTokenRenderer,
    curriculum_loss,
    render_for_seg,
)


N_PAIRS = 600


@torch.no_grad()
def evaluate_all(model, segnet, tokens, batch_size, device) -> float:
    model.eval()
    mismatches = 0
    pixels = 0
    for start in range(0, N_PAIRS, batch_size):
        end = min(start + batch_size, N_PAIRS)
        idx = torch.arange(start, end, device=device)
        target = tokens[start:end].to(device)
        frame = render_for_seg(model, target, idx, exact_path=True)
        pred = segnet(frame).argmax(1)
        mismatches += int((pred != target).sum())
        pixels += target.numel()
    return mismatches / pixels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge-root", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--ce-fraction", type=float, default=0.75)
    parser.add_argument("--softplus-fraction", type=float, default=0.92)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--train-exact-path", action="store_true")
    parser.add_argument("--freeze-prefix-blocks", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--save", type=Path, required=True)
    args = parser.parse_args()
    from tac.admission_guard import assert_governed_admission
    assert_governed_admission("train_semantic_full")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    root = args.challenge_root.resolve()
    sys.path.insert(0, str(root))
    import modules  # pylint: disable=import-error,import-outside-toplevel

    tokens = torch.load(args.cache, map_location="cpu", weights_only=False)["seg"].long()
    checkpoint = torch.load(args.init, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    model = SemanticTokenRenderer(
        width=int(config["width"]), blocks=int(config["blocks"]),
        frame_dim=int(config["frame_dim"]), num_pairs=N_PAIRS,
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    if args.freeze_prefix_blocks:
        if not 0 <= args.freeze_prefix_blocks < len(model.blocks):
            raise ValueError("--freeze-prefix-blocks must leave at least one block trainable")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for block in model.blocks[args.freeze_prefix_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)
    segnet = modules.SegNet().eval().to(device)
    segnet.load_state_dict(load_file(modules.segnet_sd_path, device=str(device)))
    for parameter in segnet.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr, weight_decay=0.0,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.lr * 0.01
    )
    generator = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(N_PAIRS, generator=generator)
    cursor = 0
    best_seg = evaluate_all(model, segnet, tokens, args.eval_batch_size, device)
    best_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    history = [{"step": 0, "exact_seg": best_seg, "best_exact_seg": best_seg}]
    print(json.dumps(history[-1]), flush=True)

    for step in range(1, args.steps + 1):
        if cursor + args.batch_size > N_PAIRS:
            order = torch.randperm(N_PAIRS, generator=generator)
            cursor = 0
        batch_ids = order[cursor:cursor + args.batch_size]
        cursor += args.batch_size
        idx = batch_ids.to(device)
        target = tokens[batch_ids].to(device)
        progress = (step - 1) / max(args.steps - 1, 1)
        amp_enabled = args.amp and device.type == "cuda" and progress < args.ce_fraction
        model.train()
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=amp_enabled
        ):
            frame = render_for_seg(
                model, target, idx, exact_path=args.train_exact_path
            )
            logits = segnet(frame)
            loss, phase = curriculum_loss(
                logits, target, step - 1, args.steps,
                args.ce_fraction, args.softplus_fraction,
            )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        scheduler.step()

        if step % args.eval_every == 0 or step == args.steps:
            exact = evaluate_all(
                model, segnet, tokens, args.eval_batch_size, device
            )
            if exact < best_seg:
                best_seg = exact
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
            record = {
                "step": step,
                "phase": phase,
                "loss": float(loss.detach()),
                "exact_seg": exact,
                "best_exact_seg": best_seg,
                "lr": optimizer.param_groups[0]["lr"],
            }
            history.append(record)
            print(json.dumps(record), flush=True)

    model.load_state_dict(best_state)
    final_seg = evaluate_all(model, segnet, tokens, args.eval_batch_size, device)
    params = sum(parameter.numel() for parameter in model.parameters())
    result = {
        "verdict": "PASS" if final_seg < 4.0e-4 else "FAIL",
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "params": params,
        "projected_model_bytes_int4_excluding_scales": math.ceil(params / 2),
        "final_exact_seg": final_seg,
        "history": history,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    args.save.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state,
        "config": config,
        "best_exact_seg": final_seg,
        "result": result,
    }, args.save)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
