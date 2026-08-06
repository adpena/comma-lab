#!/usr/bin/env python3
# borrowed_substrate_accounting:
#   source_repo: /Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo
#   source_head: 2f94596bb0136d342254022a5c9584756eae0468
#   source_path: code/train_pose_carrier_full.py
#   source_sha256: 684a4906edecb7653572db77c11a03a4e445eb256a8dc7b665e8fa0f78cab649
#   theirs: PR130 pose-leg mechanism, tensor shapes, search/codec semantics.
#   ours: vendored custody copy only; mx2 behavior changes live in sibling wrappers.
#   score_claim: false
#
"""Train the quantized low-rank PoseNet carrier on all 600 frame pairs."""

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

from learned_pose_carrier_oracle import (
    fake_quant_basis,
    predict,
    quantize_basis,
)
from pose_basis_oracle import CAMERA_H, CAMERA_W, N_TOTAL_PAIRS, ste_uint8, summarize
from semantic_renderer_oracle import SemanticTokenRenderer


@torch.no_grad()
def render_all_masters(model, tokens, batch_size, device) -> torch.Tensor:
    """Render and retain the exact camera-resolution uint8 second frames."""
    model.eval()
    masters = torch.empty(
        N_TOTAL_PAIRS, 3, CAMERA_H, CAMERA_W, dtype=torch.uint8, device="cpu"
    )
    for start in range(0, N_TOTAL_PAIRS, batch_size):
        end = min(start + batch_size, N_TOTAL_PAIRS)
        idx = torch.arange(start, end, device=device)
        target = tokens[start:end].to(device)
        frame = model(target, idx)
        camera = ste_uint8(F.interpolate(
            frame, size=(CAMERA_H, CAMERA_W), mode="bilinear", align_corners=False
        ))
        masters[start:end].copy_(camera.to(torch.uint8).cpu())
        print(json.dumps({"master_rendered": end, "master_total": N_TOTAL_PAIRS}), flush=True)
    return masters


def load_or_render_masters(args, model, tokens, device) -> torch.Tensor:
    if args.master_cache is not None and args.reuse_master_cache and args.master_cache.exists():
        payload = torch.load(args.master_cache, map_location="cpu", weights_only=False)
        if payload.get("source_checkpoint") != str(args.master_checkpoint.resolve()):
            raise ValueError("master cache was rendered from a different checkpoint")
        masters = payload["masters"]
    else:
        masters = render_all_masters(model, tokens, args.render_batch_size, device)
        if args.master_cache is not None:
            args.master_cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "source_checkpoint": str(args.master_checkpoint.resolve()),
                "masters": masters,
            }, args.master_cache)
    expected = (N_TOTAL_PAIRS, 3, CAMERA_H, CAMERA_W)
    if tuple(masters.shape) != expected or masters.dtype != torch.uint8:
        raise ValueError(f"master cache must be uint8 with shape {expected}")
    return masters


@torch.no_grad()
def evaluate_all(posenet, masters, targets, coeff, basis, args, device) -> torch.Tensor:
    posenet.eval()
    values = []
    for start in range(0, N_TOTAL_PAIRS, args.eval_batch_size):
        end = min(start + args.eval_batch_size, N_TOTAL_PAIRS)
        master = masters[start:end].to(device=device, dtype=torch.float32)
        pred = predict(
            posenet, master, coeff[start:end], basis,
            args.amplitude, args.carrier_base, args.master_carrier_amplitude,
        )
        values.append((pred - targets[start:end].to(device)).square().mean(1).cpu())
    return torch.cat(values)


def initialize_coefficients(checkpoint, targets, target_scale, basis_dim, device) -> torch.Tensor:
    source = checkpoint["coeff"].float()
    if source.shape[1] != basis_dim:
        raise ValueError("carrier checkpoint basis dimension does not match")
    pair_ids = checkpoint.get("result", {}).get("pair_ids")
    if pair_ids is None or len(pair_ids) != source.shape[0]:
        pair_ids = np.linspace(0, N_TOTAL_PAIRS - 1, source.shape[0], dtype=np.int64).tolist()
    selected = targets[pair_ids]
    distance = ((targets[:, None] - selected[None]) / target_scale).square().mean(2)
    nearest = distance.argmin(1)
    initial = source[nearest].clone()
    initial[pair_ids] = source
    return initial.to(device)


def fake_quant_selected_coeff(
    selected: torch.Tensor, full_weight: torch.Tensor, bits: int
) -> torch.Tensor:
    """Quantize selected rows using deployment's global per-dimension scales."""
    max_code = (1 << (bits - 1)) - 1
    scales = full_weight.detach().abs().amax(dim=0).clamp_min(1e-8) / max_code
    normalized = (selected / scales).clamp(-max_code, max_code)
    codes = normalized + (normalized.round() - normalized).detach()
    return codes * scales


def quantize_coeff(
    coeff: torch.Tensor, bits: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    max_code = (1 << (bits - 1)) - 1
    scales = coeff.abs().amax(dim=0).clamp_min(1e-8) / max_code
    dtype = torch.int8 if bits <= 8 else torch.int16
    codes = (coeff / scales).round().clamp(-max_code, max_code).to(dtype)
    return codes.float() * scales, codes, scales


def clip_sparse_gradient(parameter: torch.Tensor, max_norm: float) -> None:
    if parameter.grad is None:
        return
    gradient = parameter.grad.coalesce()
    values = gradient.values()
    norm = values.norm()
    if torch.isfinite(norm) and norm > max_norm:
        values.mul_(max_norm / norm)
    parameter.grad = gradient


class RowLocalSparseAdam(torch.optim.Optimizer):
    """Sparse Adam with independent bias-correction clocks for every row."""

    def __init__(
        self, params, lr: float, betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        super().__init__(params, {"lr": lr, "betas": betas, "eps": eps})

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad.coalesce()
                rows = gradient.indices()[0]
                values = gradient.values()
                state = self.state[parameter]
                if not state:
                    state["row_step"] = torch.zeros(
                        parameter.shape[0], dtype=torch.int64,
                        device=parameter.device,
                    )
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)

                row_step = state["row_step"].index_select(0, rows).add_(1)
                exp_avg = state["exp_avg"].index_select(0, rows)
                exp_avg_sq = state["exp_avg_sq"].index_select(0, rows)
                exp_avg.mul_(beta1).add_(values, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(
                    values, values, value=1.0 - beta2
                )
                state["row_step"].index_copy_(0, rows, row_step)
                state["exp_avg"].index_copy_(0, rows, exp_avg)
                state["exp_avg_sq"].index_copy_(0, rows, exp_avg_sq)

                step_float = row_step.to(values.dtype)
                bias_correction1 = 1.0 - beta1 ** step_float
                bias_correction2 = 1.0 - beta2 ** step_float
                denominator = (
                    exp_avg_sq.sqrt()
                    / bias_correction2.sqrt().unsqueeze(1)
                ).add_(group["eps"])
                update = exp_avg / bias_correction1.unsqueeze(1) / denominator
                parameter.index_add_(0, rows, update, alpha=-group["lr"])
        return loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge-root", type=Path, required=True)
    parser.add_argument("--target-cache", type=Path, required=True)
    parser.add_argument("--master-checkpoint", type=Path, required=True)
    parser.add_argument("--init-carrier", type=Path, required=True)
    parser.add_argument("--master-cache", type=Path, default=None)
    parser.add_argument("--reuse-master-cache", action="store_true")
    parser.add_argument("--cache-masters-on-device", action="store_true")
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument(
        "--stop-after-step",
        type=int,
        help=(
            "stop early while retaining --steps as the cosine-schedule horizon; "
            "used for historically selected intermediate checkpoints"
        ),
    )
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--eval-batch-size", type=int, default=12)
    parser.add_argument("--render-batch-size", type=int, default=4)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--lr-basis", type=float, default=0.003)
    parser.add_argument("--lr-coeff", type=float, default=0.03)
    parser.add_argument("--basis-freeze-fraction", type=float, default=0.30)
    parser.add_argument("--basis-train-until-fraction", type=float, default=1.0)
    parser.add_argument("--qat-fraction", type=float, default=0.65)
    parser.add_argument("--coeff-qat-fraction", type=float, default=None)
    parser.add_argument("--metric-loss-after-basis", action="store_true")
    parser.add_argument("--always-metric-loss", action="store_true")
    parser.add_argument("--metric-normalized-weight", type=float, default=0.0)
    parser.add_argument("--hard-mining-power", type=float, default=0.0)
    parser.add_argument("--hard-mining-max", type=float, default=8.0)
    parser.add_argument("--basis-bits", type=int, default=8)
    parser.add_argument("--coeff-bits", type=int, choices=(8, 10, 12, 16), default=8)
    parser.add_argument("--amplitude", type=float, default=32.0)
    parser.add_argument("--master-carrier-amplitude", type=float, default=0.0)
    parser.add_argument("--carrier-base", choices=("gray", "master"), default="gray")
    parser.add_argument("--zero-init-coeff", action="store_true")
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--save", type=Path, required=True)
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be positive")
    stop_after = args.stop_after_step or args.steps
    if not 1 <= stop_after <= args.steps:
        raise ValueError("--stop-after-step must be in [1, --steps]")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    root = args.challenge_root.resolve()
    sys.path.insert(0, str(root))
    import modules  # pylint: disable=import-error,import-outside-toplevel

    cache = torch.load(args.target_cache, map_location="cpu", weights_only=False)
    tokens = cache["seg"].long()
    targets = cache["pose"].float()
    if tokens.shape[0] != N_TOTAL_PAIRS or targets.shape != (N_TOTAL_PAIRS, 6):
        raise ValueError("target cache does not contain the expected 600 pairs")
    target_scale = (targets.amax(0) - targets.amin(0)).clamp_min(1e-4)

    master_checkpoint = torch.load(args.master_checkpoint, map_location="cpu", weights_only=False)
    master_config = master_checkpoint["config"]
    master_model = SemanticTokenRenderer(
        width=int(master_config["width"]), blocks=int(master_config["blocks"]),
        frame_dim=int(master_config["frame_dim"]), num_pairs=N_TOTAL_PAIRS,
    ).eval().to(device)
    master_state = master_checkpoint["state_dict"]
    if "quant_bits" in master_checkpoint:
        from evaluate_semantic_quantization import quantize_tensor

        bits = int(master_checkpoint["quant_bits"])
        master_state = {
            name: quantize_tensor(
                value, bits, embedding=name.endswith("embed.weight")
            )[0]
            for name, value in master_state.items()
        }
    master_model.load_state_dict(master_state)
    masters = load_or_render_masters(args, master_model, tokens, device)
    del master_model
    torch.cuda.empty_cache()
    if args.cache_masters_on_device:
        masters = masters.to(device=device, non_blocking=True)

    posenet = modules.PoseNet().eval().to(device)
    posenet.load_state_dict(load_file(modules.posenet_sd_path, device=str(device)))
    for parameter in posenet.parameters():
        parameter.requires_grad_(False)

    initial = torch.load(args.init_carrier, map_location="cpu", weights_only=False)
    raw_basis = torch.nn.Parameter(initial["basis"].float().to(device))
    basis_dim = raw_basis.shape[0]
    coeff = torch.nn.Embedding(N_TOTAL_PAIRS, basis_dim, sparse=True).to(device)
    with torch.no_grad():
        coeff.weight.copy_(initialize_coefficients(
            initial, targets, target_scale, basis_dim, device
        ))
        if args.zero_init_coeff:
            coeff.weight.zero_()
    basis_optimizer = torch.optim.Adam([raw_basis], lr=args.lr_basis)
    coeff_optimizer = RowLocalSparseAdam([coeff.weight], lr=args.lr_coeff)
    basis_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        basis_optimizer, T_max=args.steps, eta_min=args.lr_basis * 0.01
    )
    coeff_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        coeff_optimizer, T_max=args.steps, eta_min=args.lr_coeff * 0.01
    )
    generator = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(N_TOTAL_PAIRS, generator=generator)
    sampling_weights = None
    cursor = 0
    freeze_until = int(args.steps * args.basis_freeze_fraction)
    train_until = int(args.steps * args.basis_train_until_fraction)
    qat_start = int(args.steps * args.qat_fraction)
    coeff_qat_fraction = (
        args.qat_fraction if args.coeff_qat_fraction is None else args.coeff_qat_fraction
    )
    coeff_qat_start = int(args.steps * coeff_qat_fraction)
    history = []
    best = {"mean": float("inf"), "basis": None, "coeff": None}

    for step in range(1, stop_after + 1):
        if cursor + args.batch_size > N_TOTAL_PAIRS:
            if sampling_weights is None:
                order = torch.randperm(N_TOTAL_PAIRS, generator=generator)
            else:
                order = torch.multinomial(
                    sampling_weights,
                    N_TOTAL_PAIRS,
                    replacement=True,
                    generator=generator,
                )
            cursor = 0
        batch_ids_cpu = order[cursor:cursor + args.batch_size]
        cursor += args.batch_size
        batch_ids = batch_ids_cpu.to(device)
        if masters.device.type == device.type:
            master = masters.index_select(0, batch_ids).to(dtype=torch.float32)
        else:
            master = masters[batch_ids_cpu].to(device=device, dtype=torch.float32)
        target = targets[batch_ids_cpu].to(device)
        use_basis_qat = step > qat_start
        use_coeff_qat = step > coeff_qat_start
        train_basis = step > freeze_until and step <= train_until
        forward_basis = raw_basis if train_basis else raw_basis.detach()
        forward_coeff = coeff(batch_ids)
        if use_basis_qat:
            forward_basis = fake_quant_basis(forward_basis, args.basis_bits)
        if use_coeff_qat:
            forward_coeff = fake_quant_selected_coeff(
                forward_coeff, coeff.weight, args.coeff_bits
            )
        pred = predict(
            posenet, master, forward_coeff, forward_basis,
            args.amplitude, args.carrier_base, args.master_carrier_amplitude,
        )
        residual = pred - target
        normalized = residual / target_scale.to(device)
        use_metric_loss = args.always_metric_loss or (
            args.metric_loss_after_basis and step > train_until
        )
        if use_metric_loss:
            loss = (
                residual.square().mean()
                + args.metric_normalized_weight * normalized.square().mean()
            )
        else:
            loss = normalized.square().mean() + 0.02 * residual.square().mean()
        basis_optimizer.zero_grad(set_to_none=True)
        coeff_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([raw_basis], 10.0)
        clip_sparse_gradient(coeff.weight, 10.0)
        basis_optimizer.step()
        coeff_optimizer.step()
        basis_scheduler.step()
        coeff_scheduler.step()

        if step == 1 or step % 100 == 0:
            sample = residual.detach().square().mean(1)
            record = {
                "step": step,
                "phase": (
                    "full_qat" if use_basis_qat and use_coeff_qat
                    else "basis_qat" if use_basis_qat
                    else "float"
                ),
                "loss_mode": "raw_mse" if use_metric_loss else "range_normalized",
                "loss": float(loss.detach()),
                "sample": summarize(sample, 3e-5),
                "lr_basis": basis_optimizer.param_groups[0]["lr"],
                "lr_coeff": coeff_optimizer.param_groups[0]["lr"],
            }
            print(json.dumps(record), flush=True)

        if step % args.eval_every == 0 or step == stop_after:
            basis_q, _, _ = quantize_basis(raw_basis.detach(), args.basis_bits)
            coeff_q, _, _ = quantize_coeff(coeff.weight.detach(), args.coeff_bits)
            full_mse = evaluate_all(
                posenet, masters, targets, coeff_q, basis_q, args, device
            )
            summary = summarize(full_mse, 3e-5)
            record = {"step": step, "phase": "full_quantized", **summary}
            history.append(record)
            print(json.dumps(record), flush=True)
            if args.hard_mining_power > 0.0:
                median = full_mse.median().clamp_min(1e-12)
                relative = (full_mse / median).clamp(1.0, args.hard_mining_max)
                sampling_weights = relative.pow(args.hard_mining_power)
                print(json.dumps({
                    "step": step,
                    "event": "update_hard_mining_weights",
                    "weight_min": float(sampling_weights.min()),
                    "weight_max": float(sampling_weights.max()),
                    "weight_mean": float(sampling_weights.mean()),
                }), flush=True)
            latest_path = args.save.with_name(args.save.stem + ".latest.pt")
            latest_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_payload = {
                "basis": raw_basis.detach().cpu().clone(),
                "coeff": coeff.weight.detach().cpu().clone(),
                "result": {
                    "pair_ids": list(range(N_TOTAL_PAIRS)),
                    "step": step,
                    "quantized_basis_coeff": summary,
                    "per_pair_mse": full_mse.tolist(),
                },
            }
            torch.save(checkpoint_payload, latest_path)
            if use_basis_qat and use_coeff_qat and summary["mean"] < best["mean"]:
                best = {
                    "mean": summary["mean"],
                    "basis": raw_basis.detach().cpu().clone(),
                    "coeff": coeff.weight.detach().cpu().clone(),
                }
                best_path = args.save.with_name(args.save.stem + ".best.pt")
                torch.save(checkpoint_payload, best_path)

    if best["basis"] is None:
        best = {
            "mean": history[-1]["mean"],
            "basis": raw_basis.detach().cpu().clone(),
            "coeff": coeff.weight.detach().cpu().clone(),
        }
    basis_best = best["basis"].to(device)
    coeff_best = best["coeff"].to(device)
    with torch.no_grad():
        fp_mse = evaluate_all(posenet, masters, targets, coeff_best, basis_best, args, device)
        basis_q, basis_codes, basis_scales = quantize_basis(basis_best, args.basis_bits)
        coeff_q, coeff_codes, coeff_scales = quantize_coeff(
            coeff_best, args.coeff_bits
        )
        q_mse = evaluate_all(posenet, masters, targets, coeff_q, basis_q, args, device)

    basis_values = int(raw_basis.numel())
    projected_bytes = (basis_values * args.basis_bits + 7) // 8 + 4 * basis_dim
    projected_bytes += (
        N_TOTAL_PAIRS * basis_dim * args.coeff_bits + 7
    ) // 8 + 4 * basis_dim
    q_summary = summarize(q_mse, 3e-5)
    passed = q_summary["mean"] < 3e-6 and q_summary["reached"] == N_TOTAL_PAIRS
    result = {
        "verdict": "PASS" if passed else "FAIL",
        "config": {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()},
        "float": summarize(fp_mse, 3e-5),
        "quantized_basis_coeff": q_summary,
        "projected_600_payload_bytes": projected_bytes,
        "basis_code_range": [int(basis_codes.min()), int(basis_codes.max())],
        "basis_scale_range": [float(basis_scales.min()), float(basis_scales.max())],
        "coefficient_code_range": [int(coeff_codes.min()), int(coeff_codes.max())],
        "coefficient_scale_range": [float(coeff_scales.min()), float(coeff_scales.max())],
        "per_pair_mse": q_mse.tolist(),
        "history": history,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    args.save.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"basis": best["basis"], "coeff": best["coeff"], "result": result}, args.save)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
