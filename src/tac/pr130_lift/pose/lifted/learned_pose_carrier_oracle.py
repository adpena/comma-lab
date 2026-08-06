#!/usr/bin/env python3
# borrowed_substrate_accounting:
#   source_repo: /Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo
#   source_head: 2f94596bb0136d342254022a5c9584756eae0468
#   source_path: code/learned_pose_carrier_oracle.py
#   source_sha256: 59a574bf2ec52953ccdcf50edab94fff5d9c62158b993bd4da2e245f5fe310a4
#   theirs: PR130 pose-leg mechanism, tensor shapes, search/codec semantics.
#   ours: vendored custody copy only; mx2 behavior changes live in sibling wrappers.
#   score_claim: false
#
"""Learn a compact low-rank pixel carrier that controls PoseNet outputs.

Unlike pose_basis_oracle.py, both the spatial basis and per-pair coefficients
are optimized. The model-size projection assumes an int4 basis and int8
coefficients for all 600 pairs.
"""

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

from pose_basis_oracle import (
    CAMERA_H,
    CAMERA_W,
    EVAL_H,
    EVAL_W,
    N_TOTAL_PAIRS,
    load_selected_pairs,
    pose_output,
    ste_uint8,
    summarize,
)


HERE = Path(__file__).resolve().parent
DEFAULT_REPO = HERE.parents[1] / "repo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge-root", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--target-cache", type=Path, required=True)
    parser.add_argument("--master-checkpoint", type=Path, default=None)
    parser.add_argument("--pairs", type=int, default=4)
    parser.add_argument("--basis-dim", type=int, default=8)
    parser.add_argument("--basis-height", type=int, default=24)
    parser.add_argument("--basis-width", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--amplitude", type=float, default=32.0)
    parser.add_argument("--carrier-base", choices=("master", "gray"), default="master")
    parser.add_argument("--basis-bits", type=int, default=8)
    parser.add_argument("--qat-fraction", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", type=Path, default=HERE / "results" / "learned_pose_carrier.json")
    parser.add_argument("--save", type=Path, default=None)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--resume-basis", type=Path, default=None)
    return parser.parse_args()


def normalized_basis(raw: torch.Tensor) -> torch.Tensor:
    basis = F.interpolate(raw, size=(EVAL_H, EVAL_W), mode="bicubic", align_corners=False)
    basis = basis - basis.mean(dim=(1, 2, 3), keepdim=True)
    rms = basis.square().mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(1e-5)
    return basis / rms


def render_slave(master_camera: torch.Tensor, coeff: torch.Tensor, raw_basis: torch.Tensor,
                 amplitude: float, carrier_base: str = "master") -> torch.Tensor:
    master = F.interpolate(
        ste_uint8(master_camera), size=(EVAL_H, EVAL_W), mode="bilinear", align_corners=False
    )
    base = master if carrier_base == "master" else torch.full_like(master, 127.5)
    carrier = torch.einsum("bk,kchw->bchw", coeff, normalized_basis(raw_basis))
    carrier = carrier / math.sqrt(raw_basis.shape[0])
    slave = ste_uint8(base + amplitude * carrier)
    slave = F.interpolate(slave, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False)
    return ste_uint8(slave)


def predict(posenet, master: torch.Tensor, coeff: torch.Tensor, raw_basis: torch.Tensor,
            amplitude: float, carrier_base: str = "master",
            master_carrier_amplitude: float = 0.0) -> torch.Tensor:
    slave = render_slave(master, coeff, raw_basis, amplitude, carrier_base)
    master_output = ste_uint8(master)
    if master_carrier_amplitude != 0.0:
        master_eval = F.interpolate(
            master_output, size=(EVAL_H, EVAL_W), mode="bilinear",
            align_corners=False,
        )
        carrier = torch.einsum("bk,kchw->bchw", coeff, normalized_basis(raw_basis))
        carrier = carrier / math.sqrt(raw_basis.shape[0])
        master_eval = ste_uint8(master_eval + master_carrier_amplitude * carrier)
        master_output = F.interpolate(
            master_eval, size=(CAMERA_H, CAMERA_W), mode="bicubic",
            align_corners=False,
        )
        master_output = ste_uint8(master_output)
    return pose_output(posenet, torch.stack([slave, master_output], dim=1))


def quantize_basis(raw: torch.Tensor, bits: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not 2 <= bits <= 8:
        raise ValueError("--basis-bits must be in [2,8]")
    limit = (1 << (bits - 1)) - 1
    scales = raw.abs().flatten(1).amax(dim=1).clamp_min(1e-8) / limit
    q = (raw / scales[:, None, None, None]).round().clamp(-limit, limit).to(torch.int8)
    return q.float() * scales[:, None, None, None], q, scales


def quantize_coeff_int8(coeff: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scales = coeff.abs().amax(dim=0).clamp_min(1e-8) / 127.0
    q = (coeff / scales).round().clamp(-127, 127).to(torch.int8)
    return q.float() * scales, q, scales


def fake_quant_basis(raw: torch.Tensor, bits: int) -> torch.Tensor:
    limit = (1 << (bits - 1)) - 1
    scales = (raw.detach().abs().flatten(1).amax(dim=1).clamp_min(1e-8) / limit)
    normalized = (raw / scales[:, None, None, None]).clamp(-limit, limit)
    codes = normalized + (normalized.round() - normalized).detach()
    return codes * scales[:, None, None, None]


def fake_quant_coeff(coeff: torch.Tensor) -> torch.Tensor:
    scales = coeff.detach().abs().amax(dim=0).clamp_min(1e-8) / 127.0
    normalized = (coeff / scales).clamp(-127, 127)
    codes = normalized + (normalized.round() - normalized).detach()
    return codes * scales


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    root = args.challenge_root.resolve()
    video = (args.video or root / "videos" / "0.mkv").resolve()
    if not 1 <= args.pairs <= N_TOTAL_PAIRS:
        raise ValueError(f"--pairs must be in [1,{N_TOTAL_PAIRS}]")

    sys.path.insert(0, str(root))
    import frame_utils  # pylint: disable=import-error,import-outside-toplevel
    import modules  # pylint: disable=import-error,import-outside-toplevel

    pair_ids = np.linspace(0, N_TOTAL_PAIRS - 1, args.pairs, dtype=np.int64).tolist()
    cache = torch.load(args.target_cache, map_location="cpu", weights_only=False)
    targets_all = cache["pose"].float()
    target = targets_all[pair_ids].to(device)
    target_scale = (targets_all.amax(0) - targets_all.amin(0)).clamp_min(1e-4).to(device)

    if args.master_checkpoint is None:
        gt = load_selected_pairs(video, frame_utils, pair_ids).to(device=device, dtype=torch.float32)
        master = gt[:, 1].permute(0, 3, 1, 2)
    else:
        from semantic_renderer_oracle import SemanticTokenRenderer

        master_checkpoint = torch.load(args.master_checkpoint, map_location=device, weights_only=False)
        master_config = master_checkpoint["config"]
        master_model = SemanticTokenRenderer(
            width=int(master_config["width"]), blocks=int(master_config["blocks"]),
            frame_dim=int(master_config["frame_dim"]), num_pairs=N_TOTAL_PAIRS,
        ).eval().to(device)
        master_model.load_state_dict(master_checkpoint["state_dict"])
        pair_idx = torch.tensor(pair_ids, device=device, dtype=torch.long)
        tokens = cache["seg"][pair_ids].to(device=device, dtype=torch.long)
        with torch.inference_mode():
            master_eval = master_model(tokens, pair_idx)
            master = ste_uint8(F.interpolate(
                master_eval, size=(CAMERA_H, CAMERA_W), mode="bilinear", align_corners=False
            ))

    posenet = modules.PoseNet().eval().to(device)
    posenet.load_state_dict(load_file(modules.posenet_sd_path, device=str(device)))
    for parameter in posenet.parameters():
        parameter.requires_grad_(False)

    raw_basis = torch.nn.Parameter(
        torch.randn(args.basis_dim, 3, args.basis_height, args.basis_width, device=device) * 0.05
    )
    coeff = torch.nn.Parameter(torch.randn(args.pairs, args.basis_dim, device=device) * 0.01)
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location="cpu", weights_only=False)
        if checkpoint["basis"].shape != raw_basis.shape or checkpoint["coeff"].shape != coeff.shape:
            raise ValueError("resume checkpoint shape does not match requested basis/pair configuration")
        raw_basis.data.copy_(checkpoint["basis"].to(device))
        coeff.data.copy_(checkpoint["coeff"].to(device))
    elif args.resume_basis is not None:
        checkpoint = torch.load(args.resume_basis, map_location="cpu", weights_only=False)
        source_basis = checkpoint["basis"]
        if (source_basis.shape[1:] != raw_basis.shape[1:]
                or source_basis.shape[0] > raw_basis.shape[0]):
            raise ValueError("resume-basis checkpoint cannot be expanded to requested basis")
        raw_basis.data[:source_basis.shape[0]].copy_(source_basis.to(device))
    optimizer = torch.optim.Adam([raw_basis, coeff], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.steps, eta_min=args.lr * 0.01
    )

    history = []
    best = {"mean": float("inf"), "basis": None, "coeff": None}
    qat_start = int(args.steps * args.qat_fraction)
    for step in range(1, args.steps + 1):
        use_qat = step > qat_start
        if step == qat_start + 1:
            best = {"mean": float("inf"), "basis": None, "coeff": None}
        forward_basis = fake_quant_basis(raw_basis, args.basis_bits) if use_qat else raw_basis
        forward_coeff = fake_quant_coeff(coeff) if use_qat else coeff
        pred = predict(
            posenet, master, forward_coeff, forward_basis, args.amplitude, args.carrier_base
        )
        residual = pred - target
        loss = (residual / target_scale).square().mean() + 0.02 * residual.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([raw_basis, coeff], 10.0)

        exact = residual.detach().square().mean(dim=1)
        mean = float(exact.mean())
        if mean < best["mean"]:
            best = {
                "mean": mean,
                "basis": raw_basis.detach().cpu().clone(),
                "coeff": coeff.detach().cpu().clone(),
            }
        if step == 1 or step % 25 == 0 or step == args.steps:
            record = {"step": step, "phase": "qat" if use_qat else "float",
                      "loss": float(loss.detach()), **summarize(exact, 3e-5)}
            history.append(record)
            print(json.dumps(record), flush=True)
        optimizer.step()
        scheduler.step()

    basis_best = best["basis"].to(device)
    coeff_best = best["coeff"].to(device)
    with torch.no_grad():
        fp_mse = (predict(
            posenet, master, coeff_best, basis_best, args.amplitude, args.carrier_base
        ) - target).square().mean(1)
        basis_q, basis_codes, basis_scales = quantize_basis(basis_best, args.basis_bits)
        coeff_q, coeff_codes, coeff_scales = quantize_coeff_int8(coeff_best)
        q_mse = (predict(
            posenet, master, coeff_q, basis_q, args.amplitude, args.carrier_base
        ) - target).square().mean(1)

    basis_values = args.basis_dim * 3 * args.basis_height * args.basis_width
    projected_bytes = (basis_values * args.basis_bits + 7) // 8 + 2 * args.basis_dim
    projected_bytes += N_TOTAL_PAIRS * args.basis_dim + 2 * args.basis_dim
    q_summary = summarize(q_mse, 3e-5)
    passed = q_summary["mean"] < 1e-5 and q_summary["reached"] == args.pairs
    result = {
        "verdict": "PASS" if passed else "FAIL",
        "config": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "pair_ids": pair_ids,
        "float": summarize(fp_mse, 3e-5),
        "quantized_basis_int8_coeff": q_summary,
        "projected_600_payload_bytes": projected_bytes,
        "basis_code_range": [int(basis_codes.min()), int(basis_codes.max())],
        "basis_scale_range": [float(basis_scales.min()), float(basis_scales.max())],
        "coefficient_code_range": [int(coeff_codes.min()), int(coeff_codes.max())],
        "coefficient_scale_range": [float(coeff_scales.min()), float(coeff_scales.max())],
        "history": history,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"basis": best["basis"], "coeff": best["coeff"], "result": result}, args.save)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
