#!/usr/bin/env python3
# borrowed_substrate_accounting:
#   source_repo: /Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo
#   source_head: 2f94596bb0136d342254022a5c9584756eae0468
#   source_path: code/search_pose_coeff_cpu.py
#   source_sha256: 42b0f57cad5170b6da2d7147ba90af147cda7baecc74b848b9cd2c4bed7c9285
#   theirs: PR130 pose-leg mechanism, tensor shapes, search/codec semantics.
#   ours: vendored custody copy only; mx2 behavior changes live in sibling wrappers.
#   score_claim: false
#
"""Greedy exact PoseNet search over deployed int12 coefficient codes."""

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
            posenet, master, coeff[start:end], basis, amplitude, "gray",
        )
        values.append((pred - target).square().mean(1).cpu())
    return torch.cat(values, dim=0)


@torch.no_grad()
def score_candidates(
    posenet, master, target, basis, codes, scales, batch_size, device, amplitude
):
    master = master.to(device=device, dtype=torch.float32)
    target = target.to(device=device)
    values = []
    for start in range(0, len(codes), batch_size):
        chunk = codes[start:start + batch_size]
        valid = len(chunk)
        if valid < batch_size:
            chunk = torch.cat([
                chunk, chunk[-1:].expand(batch_size - valid, -1)
            ])
        coeff = chunk.float() * scales[None]
        pred = predict(
            posenet,
            master.expand(batch_size, -1, -1, -1).float(),
            coeff,
            basis,
            amplitude,
            "gray",
        )
        mse = (pred - target.expand(batch_size, -1)).square().mean(1)
        values.append(mse[:valid].cpu())
    return torch.cat(values, dim=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge-root", type=Path, required=True)
    parser.add_argument("--target-cache", type=Path, required=True)
    parser.add_argument("--master-cache", type=Path, required=True)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument(
        "--steps", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32]
    )
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--candidate-batch-size", type=int, default=8)
    parser.add_argument("--amplitude", type=float, default=64.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--save", type=Path, required=True)
    args = parser.parse_args()

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
    device = torch.device(args.device)
    basis, basis_codes, basis_scales = quantize_basis(
        initial["basis"].float().to(device), 8
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

    history = []
    baseline = evaluate_all(
        posenet, masters, targets, basis, coeff, args.eval_batch_size,
        device, args.amplitude,
    )
    print(json.dumps({
        "pass": 0, **summarize(baseline, 3e-5)
    }), flush=True)

    for pass_index in range(1, args.passes + 1):
        coeff = coeff_codes.float() * coeff_scales[None]
        current_mse = evaluate_all(
            posenet, masters, targets, basis, coeff, args.eval_batch_size,
            device, args.amplitude,
        )
        selected = current_mse.argsort(
            descending=True
        )[:args.top_k].tolist()
        accepted = 0
        improvement = 0.0
        for rank, pair_id in enumerate(selected, 1):
            current = coeff_codes[pair_id].clone()
            candidates = [current]
            for dim in range(current.numel()):
                for step in args.steps:
                    for direction in (-1, 1):
                        candidate = current.clone()
                        candidate[dim] = torch.clamp(
                            candidate[dim] + direction * step, -2047, 2047
                        )
                        if candidate[dim] != current[dim]:
                            candidates.append(candidate)
            candidates = torch.unique(torch.stack(candidates), dim=0)
            current_index = int(torch.nonzero(
                (candidates == current).all(dim=1), as_tuple=False
            )[0])
            scores = score_candidates(
                posenet,
                masters[pair_id:pair_id + 1],
                targets[pair_id:pair_id + 1],
                basis,
                candidates,
                coeff_scales,
                args.candidate_batch_size,
                device,
                args.amplitude,
            )
            best_index = int(scores.argmin())
            old_value = float(scores[current_index])
            new_value = float(scores[best_index])
            if new_value < old_value:
                coeff_codes[pair_id] = candidates[best_index]
                accepted += 1
                improvement += old_value - new_value
            print(json.dumps({
                "pass": pass_index,
                "rank": rank,
                "pair": pair_id,
                "old": old_value,
                "new": new_value,
                "accepted": new_value < old_value,
                "cumulative_improvement": improvement,
            }), flush=True)

        coeff = coeff_codes.float() * coeff_scales[None]
        full_mse = evaluate_all(
            posenet, masters, targets, basis, coeff, args.eval_batch_size,
            device, args.amplitude,
        )
        record = {
            "pass": pass_index,
            "accepted": accepted,
            "measured_candidate_improvement": improvement,
            **summarize(full_mse, 3e-5),
        }
        history.append(record)
        print(json.dumps(record), flush=True)
        checkpoint = {
            "basis": (
                basis_codes.float()
                * basis_scales[:, None, None, None]
            ).cpu(),
            "coeff": coeff.cpu(),
            "result": {
                "pair_ids": list(range(N)),
                "quantized_basis_coeff": summarize(full_mse, 3e-5),
                "per_pair_mse": full_mse.tolist(),
                "history": history,
            },
        }
        args.save.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, args.save)

    result = {
        "verdict": "PASS" if history[-1]["mean"] < 6.4e-6 else "FAIL",
        "quantized_basis_coeff": {
            key: history[-1][key]
            for key in ("mean", "median", "max", "reached", "total")
        },
        "history": history,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
