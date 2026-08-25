#!/usr/bin/env python3
# borrowed_substrate_accounting:
#   source_repo: /Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo
#   source_repo_head: e34f31bc4969042c0051ac81aa3c56884419a231
#   lifted_at_head: 2f94596bb0136d342254022a5c9584756eae0468
#   source_path: code/pose_basis_oracle.py
#   source_sha256: 90909bdd600cf8f518b8fc951b19fd6b66796a1ed963ece4c7638cb1b14057cb
#   source_sha256_scope: exact source_path bytes at source_repo_head before adaptations
#   theirs: PR130 pose-leg mechanism, tensor shapes, search/codec semantics.
#   ours: accounting header only; mx2 behavior changes live in sibling wrappers.
#   adaptations: none
#   score_claim: false
#
"""Test whether a tiny seeded basis can carry the six PoseNet outputs.

This is an oracle for the pose branch only. It intentionally uses the original
second frame as the fixed master so a failure cannot be blamed on the semantic
renderer. The next gate repeats the experiment with synthetic master frames.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import av
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file


HERE = Path(__file__).resolve().parent
DEFAULT_REPO = HERE.parents[1] / "repo"
EVAL_H, EVAL_W = 384, 512
CAMERA_H, CAMERA_W = 874, 1164
N_TOTAL_PAIRS = 600


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--challenge-root", type=Path, default=DEFAULT_REPO)
    p.add_argument("--video", type=Path, default=None)
    p.add_argument("--pairs", type=int, default=12)
    p.add_argument("--basis-dim", type=int, default=12)
    p.add_argument("--basis-height", type=int, default=24)
    p.add_argument("--basis-width", type=int, default=32)
    p.add_argument("--steps", type=int, default=250)
    p.add_argument("--lr", type=float, default=0.08)
    p.add_argument("--amplitude", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=20260715)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path, default=HERE / "results" / "pose_basis_oracle.json")
    return p.parse_args()


def load_selected_pairs(video: Path, frame_utils, pair_ids: list[int]) -> torch.Tensor:
    wanted = set(pair_ids)
    selected: dict[int, torch.Tensor] = {}
    container = av.open(str(video))
    prev = None
    pair_idx = 0
    for frame in container.decode(container.streams.video[0]):
        rgb = frame_utils.yuv420_to_rgb(frame)
        if prev is None:
            prev = rgb
            continue
        if pair_idx in wanted:
            selected[pair_idx] = torch.stack([prev, rgb])
        prev = None
        pair_idx += 1
        if len(selected) == len(wanted):
            break
    container.close()
    missing = wanted.difference(selected)
    if missing:
        raise RuntimeError(f"video ended before pairs {sorted(missing)}")
    return torch.stack([selected[i] for i in pair_ids])


def seeded_basis(k: int, h: int, w: int, seed: int, device: torch.device) -> torch.Tensor:
    """Generate a zero-payload basis from a reproducible integer PRNG seed."""
    rng = np.random.default_rng(seed)
    low = rng.integers(-127, 128, size=(k, 3, h, w), dtype=np.int16).astype(np.float32)

    # Reserve six well-conditioned global colour/gradient directions before
    # random directions. These are cheap directions PoseNet is likely to see.
    yy = np.linspace(-1.0, 1.0, h, dtype=np.float32)[None, :, None]
    xx = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, None, :]
    if k >= 1:
        low[0] = 0
        low[0, 0] = 127
    if k >= 2:
        low[1] = 0
        low[1, 1] = 127
    if k >= 3:
        low[2] = 0
        low[2, 2] = 127
    if k >= 4:
        low[3] = np.broadcast_to(127 * xx, (3, h, w))
    if k >= 5:
        low[4] = np.broadcast_to(127 * yy, (3, h, w))
    if k >= 6:
        low[5] = np.broadcast_to(63.5 * (xx + yy), (3, h, w))

    basis = torch.from_numpy(low).to(device=device) / 127.0
    basis = F.interpolate(basis, size=(EVAL_H, EVAL_W), mode="bicubic", align_corners=False)
    basis = basis - basis.mean(dim=(1, 2, 3), keepdim=True)
    rms = basis.square().mean(dim=(1, 2, 3), keepdim=True).sqrt().clamp_min(1e-6)
    return basis / rms


def ste_uint8(x: torch.Tensor) -> torch.Tensor:
    clipped = x.clamp(0.0, 255.0)
    rounded = clipped.round()
    return clipped + (rounded - clipped).detach()


def render_slave(master_eval: torch.Tensor, coeff: torch.Tensor, basis: torch.Tensor,
                 amplitude: float) -> torch.Tensor:
    eps = 1.0 / 255.0
    base = (master_eval / 255.0).clamp(eps, 1.0 - eps)
    base_logits = torch.log(base) - torch.log1p(-base)
    carrier = torch.einsum("bk,kchw->bchw", coeff, basis) / math.sqrt(basis.shape[0])
    slave_eval = torch.sigmoid(base_logits + amplitude * carrier) * 255.0

    # Match the submission path: 384x512 render -> bicubic camera frame ->
    # uint8 storage. PoseNet will bilinearly return this to 384x512.
    slave_camera = F.interpolate(
        slave_eval, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False
    )
    return ste_uint8(slave_camera)


def rgb_to_yuv6_differentiable(rgb: torch.Tensor) -> torch.Tensor:
    h, w = rgb.shape[-2:]
    rgb = rgb[..., : 2 * (h // 2), : 2 * (w // 2)]
    red, green, blue = rgb.unbind(dim=-3)
    y = (0.299 * red + 0.587 * green + 0.114 * blue).clamp(0.0, 255.0)
    u = ((blue - y) / 1.772 + 128.0).clamp(0.0, 255.0)
    v = ((red - y) / 1.402 + 128.0).clamp(0.0, 255.0)
    u_sub = (u[..., 0::2, 0::2] + u[..., 1::2, 0::2]
             + u[..., 0::2, 1::2] + u[..., 1::2, 1::2]) * 0.25
    v_sub = (v[..., 0::2, 0::2] + v[..., 1::2, 0::2]
             + v[..., 0::2, 1::2] + v[..., 1::2, 1::2]) * 0.25
    return torch.stack([
        y[..., 0::2, 0::2], y[..., 1::2, 0::2],
        y[..., 0::2, 1::2], y[..., 1::2, 1::2], u_sub, v_sub,
    ], dim=-3)


def pose_output(posenet, pair_camera: torch.Tensor) -> torch.Tensor:
    batch, seq = pair_camera.shape[:2]
    flat = pair_camera.reshape(batch * seq, 3, *pair_camera.shape[-2:])
    flat = F.interpolate(flat, size=(EVAL_H, EVAL_W), mode="bilinear", align_corners=False)
    pose_in = rgb_to_yuv6_differentiable(flat)
    pose_in = pose_in.reshape(batch, seq * 6, *pose_in.shape[-2:])
    return posenet(pose_in)["pose"][:, :6]


def evaluate_coeff(posenet, master_camera: torch.Tensor, target: torch.Tensor,
                   coeff: torch.Tensor, basis: torch.Tensor, amplitude: float) -> torch.Tensor:
    master_u8 = ste_uint8(master_camera)
    master_eval = F.interpolate(master_u8, size=(EVAL_H, EVAL_W), mode="bilinear", align_corners=False)
    slave_u8 = render_slave(master_eval, coeff, basis, amplitude)
    pair = torch.stack([slave_u8, master_u8], dim=1)
    pred = pose_output(posenet, pair)
    return (pred - target).square().mean(dim=1)


def quantize_coeff(coeff: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scales = coeff.abs().amax(dim=0).clamp_min(1e-8) / 127.0
    q = (coeff / scales).round().clamp(-127, 127).to(torch.int8)
    return q.float() * scales, q, scales


def summarize(values: torch.Tensor, threshold: float) -> dict[str, float | int]:
    v = values.detach().float().cpu()
    return {
        "mean": float(v.mean()),
        "median": float(v.median()),
        "max": float(v.max()),
        "reached": int((v < threshold).sum()),
        "total": int(v.numel()),
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    root = args.challenge_root.resolve()
    video = (args.video or root / "videos" / "0.mkv").resolve()
    sys.path.insert(0, str(root))
    import frame_utils  # pylint: disable=import-error,import-outside-toplevel
    import modules  # pylint: disable=import-error,import-outside-toplevel

    if not 1 <= args.pairs <= N_TOTAL_PAIRS:
        raise ValueError(f"--pairs must be in [1,{N_TOTAL_PAIRS}]")
    pair_ids = np.linspace(0, N_TOTAL_PAIRS - 1, args.pairs, dtype=np.int64).tolist()
    gt = load_selected_pairs(video, frame_utils, pair_ids).to(device=device, dtype=torch.float32)
    gt_chw = gt.permute(0, 1, 4, 2, 3)

    posenet = modules.PoseNet().eval().to(device)
    posenet.load_state_dict(load_file(modules.posenet_sd_path, device=str(device)))
    for p in posenet.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        target = pose_output(posenet, gt_chw)

    master = gt_chw[:, 1]
    basis = seeded_basis(
        args.basis_dim, args.basis_height, args.basis_width, args.seed, device
    )
    coeff = torch.nn.Parameter(torch.zeros(args.pairs, args.basis_dim, device=device))
    opt = torch.optim.Adam([coeff], lr=args.lr)

    history = []
    for step in range(args.steps):
        mse = evaluate_coeff(posenet, master, target, coeff, basis, args.amplitude)
        loss = mse.mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([coeff], 10.0)
        opt.step()
        if step == 0 or (step + 1) % 25 == 0 or step + 1 == args.steps:
            rec = {"step": step + 1, **summarize(mse, 3e-5)}
            history.append(rec)
            print(json.dumps(rec), flush=True)

    with torch.no_grad():
        mse_fp = evaluate_coeff(posenet, master, target, coeff, basis, args.amplitude)
        coeff_q, q, scales = quantize_coeff(coeff)
        mse_q = evaluate_coeff(posenet, master, target, coeff_q, basis, args.amplitude)

    projected_bytes = N_TOTAL_PAIRS * args.basis_dim + 2 * args.basis_dim
    q_summary = summarize(mse_q, 3e-5)
    verdict = (
        q_summary["reached"] >= math.ceil(0.95 * args.pairs)
        and q_summary["mean"] < 1e-5
        and projected_bytes <= 8_000
    )
    result = {
        "verdict": "PASS" if verdict else "FAIL",
        "config": {
            "pair_ids": pair_ids,
            "basis_dim": args.basis_dim,
            "basis_shape": [args.basis_height, args.basis_width],
            "steps": args.steps,
            "lr": args.lr,
            "amplitude": args.amplitude,
            "seed": args.seed,
            "device": str(device),
        },
        "float_coeff": summarize(mse_fp, 3e-5),  # SLIM_FLOAT_COEF_OK: diagnostic receipt field naming the float baseline of an int8 quantization probe (both sides of the float-vs-int8 gap are reported); no ranker consumes float coefficients
        "int8_coeff": q_summary,
        "projected_600_coeff_bytes": projected_bytes,
        "coefficient_range": [int(q.min()), int(q.max())],
        "scale_range": [float(scales.min()), float(scales.max())],
        "history": history,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
