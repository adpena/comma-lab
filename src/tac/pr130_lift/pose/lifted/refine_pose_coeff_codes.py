#!/usr/bin/env python3
# borrowed_substrate_accounting:
#   source_repo: /Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo
#   source_head: 2f94596bb0136d342254022a5c9584756eae0468
#   source_path: code/refine_pose_coeff_codes.py
#   source_sha256: 8912464ba22f6bde09d4d53e4e16d1c6880b48d2358752219073fa2d4710a403
#   theirs: PR130 pose-leg mechanism, tensor shapes, search/codec semantics.
#   ours: vendored custody copy only; mx2 behavior changes live in sibling wrappers.
#   score_claim: false
#
"""Jointly refine deployed int12 pose coefficients for the worst pairs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from learned_pose_carrier_oracle import predict, quantize_basis
from pack_semantic_pose import quantize_coeff
from pose_basis_oracle import summarize


N = 600


def ste_round(value: torch.Tensor, limit: int) -> torch.Tensor:
    bounded = value.clamp(-limit, limit)
    return bounded + (bounded.round() - bounded).detach()


@torch.no_grad()
def evaluate_all(
    posenet, masters, targets, basis, coeff, batch_size, device, amplitude
):
    values = []
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        master = masters[start:end].to(device=device, dtype=torch.float32)
        target = targets[start:end].to(device=device)
        pred = predict(
            posenet, master, coeff[start:end], basis, amplitude, "gray"
        )
        values.append((pred - target).square().mean(1).cpu())
    return torch.cat(values, dim=0)


@torch.no_grad()
def evaluate_selected(
    posenet, masters, targets, basis, coeff, batch_size, amplitude
):
    values = []
    for start in range(0, len(coeff), batch_size):
        end = min(start + batch_size, len(coeff))
        valid = end - start
        master = masters[start:end]
        target = targets[start:end]
        selected = coeff[start:end]
        if valid < batch_size:
            pad = batch_size - valid
            master = torch.cat([
                master, master[-1:].expand(pad, -1, -1, -1)
            ])
            target = torch.cat([
                target, target[-1:].expand(pad, -1)
            ])
            selected = torch.cat([
                selected, selected[-1:].expand(pad, -1)
            ])
        pred = predict(
            posenet, master, selected, basis, amplitude, "gray"
        )
        values.append(
            (pred[:valid] - target[:valid]).square().mean(1).cpu()
        )
    return torch.cat(values, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge-root", type=Path, required=True)
    parser.add_argument("--target-cache", type=Path, required=True)
    parser.add_argument("--master-cache", type=Path, required=True)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--lr", type=float, default=8.0)
    parser.add_argument("--amplitude", type=float, default=64.0)
    parser.add_argument("--basis-bits", type=int, default=8)
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--disable-tf32", action="store_true")
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--save", type=Path, required=True)
    args = parser.parse_args()

    if args.disable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    sys.path.insert(0, str(args.challenge_root.resolve()))
    import modules  # pylint: disable=import-error,import-outside-toplevel

    cache = torch.load(
        args.target_cache, map_location="cpu", weights_only=False
    )
    targets = cache["pose"].float()
    master_payload = torch.load(
        args.master_cache, map_location="cpu", weights_only=False
    )
    masters = master_payload.get("masters", master_payload.get("frames"))
    initial = torch.load(args.init, map_location="cpu", weights_only=False)
    basis, _, _ = quantize_basis(
        initial["basis"].float().to(device), args.basis_bits
    )
    coeff, coeff_codes, coeff_scales = quantize_coeff(
        initial["coeff"].float().to(device), 12
    )
    coeff_codes = coeff_codes.to(torch.int16)

    posenet = modules.PoseNet().eval().to(device)
    posenet.load_state_dict(
        load_file(modules.posenet_sd_path, device=str(device))
    )
    for parameter in posenet.parameters():
        parameter.requires_grad_(False)

    baseline = evaluate_all(
        posenet, masters, targets, basis, coeff, args.eval_batch_size,
        device, args.amplitude,
    )
    if args.evaluate_only:
        result = {
            "verdict": (
                "PASS" if float(baseline.mean()) < 1.0e-5 else "FAIL"
            ),
            "selected_pair_ids": [],
            "quantized_basis_coeff": summarize(baseline, 3e-5),
            "per_pair_mse": baseline.tolist(),
            "history": [],
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2) + "\n")
        args.save.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "basis": basis.detach().cpu(),
            "coeff": coeff.detach().cpu(),
            "result": result,
        }, args.save)
        print(json.dumps({
            key: value for key, value in result.items()
            if key != "per_pair_mse"
        }, indent=2), flush=True)
        return
    protected_codes = torch.zeros_like(coeff_codes, dtype=torch.bool)
    for dimension in range(coeff_codes.shape[1]):
        anchors = torch.nonzero(
            coeff_codes[:, dimension].abs() == 2047,
            as_tuple=False,
        ).flatten()
        if len(anchors) == 0:
            raise RuntimeError(
                f"coefficient dimension {dimension} has no scale anchor"
            )
        protected_codes[anchors[0], dimension] = True
    ranking = baseline.argsort(descending=True)
    selected_ids = ranking[:args.top_k]
    selected_ids_device = selected_ids.to(device)
    selected_protected = protected_codes.index_select(
        0, selected_ids_device
    )
    selected_masters = masters[selected_ids].to(
        device=device, dtype=torch.float32
    )
    selected_targets = targets[selected_ids].to(device=device)

    latent = torch.nn.Parameter(
        coeff_codes.index_select(0, selected_ids_device).float()
    )
    optimizer = torch.optim.Adam([latent], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.lr * 0.01
    )
    best_codes = latent.detach().round().to(torch.int16).clone()
    best_mse = baseline[selected_ids].clone()
    history = []
    print(json.dumps({
        "step": 0, **summarize(baseline, 3e-5)
    }), flush=True)

    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        for start in range(0, args.top_k, args.train_batch_size):
            end = min(start + args.train_batch_size, args.top_k)
            selected_coeff = (
                ste_round(latent[start:end], 2047)
                * coeff_scales[None]
            )
            pred = predict(
                posenet,
                selected_masters[start:end],
                selected_coeff,
                basis,
                args.amplitude,
                "gray",
            )
            loss = (
                pred - selected_targets[start:end]
            ).square().mean()
            (loss * ((end - start) / args.top_k)).backward()
            total_loss += (
                float(loss.detach()) * (end - start) / args.top_k
            )
        torch.nn.utils.clip_grad_norm_([latent], 100.0)
        if latent.grad is not None:
            latent.grad.masked_fill_(selected_protected, 0.0)
        optimizer.step()
        with torch.no_grad():
            latent.clamp_(-2047, 2047)
            anchored = coeff_codes.index_select(
                0, selected_ids_device
            ).float()
            latent[selected_protected] = anchored[selected_protected]
        scheduler.step()

        if step == 1 or step % args.eval_every == 0 or step == args.steps:
            deployed_codes = latent.detach().round().clamp(-2047, 2047)
            deployed_coeff = deployed_codes * coeff_scales[None]
            current_mse = evaluate_selected(
                posenet,
                selected_masters,
                selected_targets,
                basis,
                deployed_coeff,
                args.eval_batch_size,
                args.amplitude,
            )
            improved = current_mse < best_mse
            if improved.any():
                best_mse[improved] = current_mse[improved]
                best_codes[improved.to(device)] = deployed_codes[
                    improved.to(device)
                ].to(torch.int16)
            combined = baseline.clone()
            combined[selected_ids] = best_mse
            record = {
                "step": step,
                "loss": total_loss,
                "rows_improved": int(improved.sum()),
                **summarize(combined, 3e-5),
            }
            history.append(record)
            print(json.dumps(record), flush=True)

    merged_codes = coeff_codes.clone()
    merged_codes.index_copy_(0, selected_ids_device, best_codes)
    merged_coeff = merged_codes.float() * coeff_scales[None]
    _, roundtrip_codes, _ = quantize_coeff(merged_coeff, 12)
    if not torch.equal(
        roundtrip_codes.to(torch.int16), merged_codes
    ):
        raise RuntimeError(
            "coefficient scale anchors changed during refinement"
        )

    final_mse = baseline.clone()
    final_mse[selected_ids] = best_mse
    result = {
        "verdict": (
            "PASS" if float(final_mse.mean()) < 1.0e-5 else "FAIL"
        ),
        "selected_pair_ids": selected_ids.tolist(),
        "quantized_basis_coeff": summarize(final_mse, 3e-5),
        "per_pair_mse": final_mse.tolist(),
        "history": history,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    args.save.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "basis": basis.detach().cpu(),
        "coeff": merged_coeff.detach().cpu(),
        "result": result,
    }, args.save)
    print(json.dumps({
        key: value for key, value in result.items()
        if key not in ("history", "per_pair_mse")
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
