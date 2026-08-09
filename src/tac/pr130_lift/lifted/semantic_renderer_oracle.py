#!/usr/bin/env python3
# borrowed_substrate_accounting:
#   source_repo: /Volumes/VertigoDataTier/pact/pr130_eureka_intake_20260806/repro_repo
#   source_repo_head: e34f31bc4969042c0051ac81aa3c56884419a231
#   lifted_at_head: 2f94596bb0136d342254022a5c9584756eae0468
#   source_path: code/semantic_renderer_oracle.py
#   source_sha256: 2bf3a6a8621334723fec1c3e596665d1f049a55f311c5348dd5a4c588873f25b
#   source_sha256_scope: exact source_path bytes at source_repo_head before adaptations
#   theirs: SemanticTokenRenderer architecture, curriculum, uint8/R render path, CLI.
#   ours: accounting header only; mechanism is intentionally PR130-lifted.
#   adaptations: none
"""Clean-room semantic-token renderer oracle.

The model reads the campaign's own five-class SegNet target maps and renders
only frame 2.  It intentionally uses a non-saturating CE -> softplus-margin ->
expected-flip curriculum; true uint8-path argmax disagreement selects the best
checkpoint.  The first oracle keeps the tokens exact.  A later stage adds a
sparse boundary-control alphabet only if this base reaches its gate.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
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
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--pairs", type=int, default=12)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--frame-dim", type=int, default=8)
    p.add_argument("--ce-fraction", type=float, default=0.80)
    p.add_argument("--softplus-fraction", type=float, default=0.95)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=20260715)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=Path,
                   default=HERE / "results" / "semantic_renderer_oracle.json")
    p.add_argument("--resume", type=Path, default=None)
    p.add_argument("--save", type=Path, default=None)
    return p.parse_args()


def ste_uint8(x: torch.Tensor) -> torch.Tensor:
    clipped = x.clamp(0.0, 255.0)
    return clipped + (clipped.round() - clipped).detach()


class TokenBlock(nn.Module):
    def __init__(self, width: int, frame_dim: int, dilation: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(
            width, width, 3, padding=dilation, dilation=dilation, groups=width
        )
        self.pw = nn.Conv2d(width, width, 1)
        self.norm = nn.GroupNorm(max(1, width // 8), width)
        self.film = nn.Linear(frame_dim, 2 * width)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x: torch.Tensor, frame: torch.Tensor) -> torch.Tensor:
        y = self.norm(self.pw(self.dw(x)))
        scale, shift = self.film(frame).chunk(2, dim=1)
        y = y * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        return x + F.gelu(y)


class SemanticTokenRenderer(nn.Module):
    def __init__(self, width: int = 64, blocks: int = 2, frame_dim: int = 8,
                 num_pairs: int = N_TOTAL_PAIRS, num_tokens: int = 5,
                 phase_y: int = 1, phase_x: int = 1,
                 temporal_radius: int = 0):
        super().__init__()
        self.width = width
        self.num_tokens = num_tokens
        self.phase_y = phase_y
        self.phase_x = phase_x
        self.temporal_radius = temporal_radius
        self.token_embed = nn.Embedding(num_tokens, width)
        self.frame_embed = nn.Embedding(num_pairs, frame_dim)
        phase_channels = (phase_y if phase_y > 1 else 0) + (
            phase_x if phase_x > 1 else 0
        )
        temporal_channels = 2 * temporal_radius * num_tokens
        self.coord_mix = nn.Conv2d(
            width + 4 + phase_channels + temporal_channels, width, 1
        )
        dilations = [1, 1] + [min(2 ** (index - 1), 4) for index in range(2, blocks)]
        self.blocks = nn.ModuleList([
            TokenBlock(width, frame_dim, dilation=dilations[index])
            for index in range(blocks)
        ])
        self.head = nn.Conv2d(width, 3, 3, padding=1)

    def coordinates(self, batch: int, h: int, w: int, device: torch.device,
                    dtype: torch.dtype) -> torch.Tensor:
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype),
            indexing="ij",
        )
        channels = [xx, yy, xx.square(), yy.square()]
        if self.phase_y > 1:
            phase_y = F.one_hot(
                torch.arange(h, device=device) % self.phase_y,
                num_classes=self.phase_y,
            ).to(dtype).T[:, :, None].expand(-1, -1, w)
            channels.extend(phase_y.unbind(0))
        if self.phase_x > 1:
            phase_x = F.one_hot(
                torch.arange(w, device=device) % self.phase_x,
                num_classes=self.phase_x,
            ).to(dtype).T[:, None, :].expand(-1, h, -1)
            channels.extend(phase_x.unbind(0))
        coord = torch.stack(channels, dim=0)
        return coord.unsqueeze(0).expand(batch, -1, -1, -1)

    def forward(self, tokens: torch.Tensor, pair_idx: torch.Tensor) -> torch.Tensor:
        temporal = []
        if self.temporal_radius:
            expected = 2 * self.temporal_radius + 1
            if tokens.ndim != 4 or tokens.shape[1] != expected:
                raise ValueError(
                    f"temporal renderer expects [B,{expected},H,W] tokens"
                )
            center = tokens[:, self.temporal_radius]
            for offset in range(expected):
                if offset == self.temporal_radius:
                    continue
                temporal.append(
                    F.one_hot(tokens[:, offset], self.num_tokens)
                    .movedim(-1, 1)
                )
        else:
            if tokens.ndim != 3:
                raise ValueError("renderer expects [B,H,W] tokens")
            center = tokens
        x = self.token_embed(center).permute(0, 3, 1, 2)
        features = [
            x,
            self.coordinates(
                x.shape[0], x.shape[-2], x.shape[-1], x.device, x.dtype
            ),
        ]
        features.extend(item.to(dtype=x.dtype) for item in temporal)
        x = self.coord_mix(torch.cat(features, dim=1))
        frame = self.frame_embed(pair_idx)
        for block in self.blocks:
            x = block(x, frame)
        return torch.sigmoid(self.head(F.gelu(x))) * 255.0


def target_margin(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target_logit = logits.gather(1, target[:, None])
    other = logits.clone()
    other.scatter_(1, target[:, None], -1e9)
    return target_logit - other.amax(dim=1, keepdim=True)


def curriculum_loss(logits: torch.Tensor, target: torch.Tensor, step: int,
                    total_steps: int, ce_fraction: float,
                    softplus_fraction: float) -> tuple[torch.Tensor, str]:
    progress = step / max(total_steps - 1, 1)
    if progress < ce_fraction:
        temp = 1.0 * (0.08 ** (progress / ce_fraction))
        return F.cross_entropy(logits / temp, target), "ce"
    margin = target_margin(logits, target)
    if progress < softplus_fraction:
        tau = 0.20
        return (F.softplus(-margin / tau) * tau).mean(), "softplus_margin"
    tail = (progress - softplus_fraction) / max(1.0 - softplus_fraction, 1e-6)
    tau = 0.15 - 0.10 * tail
    return torch.sigmoid(-margin / tau).mean(), "expected_flip"


def render_for_seg(model: nn.Module, tokens: torch.Tensor, idx: torch.Tensor,
                   exact_path: bool) -> torch.Tensor:
    frame_eval = model(tokens, idx)
    if not exact_path:
        return ste_uint8(frame_eval)
    camera = ste_uint8(F.interpolate(
        frame_eval, size=(CAMERA_H, CAMERA_W), mode="bilinear", align_corners=False
    ))
    return F.interpolate(camera, size=(EVAL_H, EVAL_W), mode="bilinear", align_corners=False)


@torch.no_grad()
def exact_seg(model: nn.Module, segnet: nn.Module, tokens: torch.Tensor,
              idx: torch.Tensor) -> float:
    model.eval()
    frame = render_for_seg(model, tokens, idx, exact_path=True)
    pred = segnet(frame).argmax(dim=1)
    return float((pred != tokens).float().mean())


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    root = args.challenge_root.resolve()
    sys.path.insert(0, str(root))
    import modules  # pylint: disable=import-error,import-outside-toplevel

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    all_tokens = cache["seg"].long()
    pair_ids = np.linspace(0, N_TOTAL_PAIRS - 1, args.pairs, dtype=np.int64).tolist()
    idx = torch.tensor(pair_ids, dtype=torch.long, device=device)
    tokens = all_tokens[idx.cpu()].to(device)

    segnet = modules.SegNet().eval().to(device)
    segnet.load_state_dict(load_file(modules.segnet_sd_path, device=str(device)))
    for param in segnet.parameters():
        param.requires_grad_(False)

    model = SemanticTokenRenderer(
        args.width, args.blocks, args.frame_dim, num_pairs=N_TOTAL_PAIRS
    ).to(device)
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"])
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(args.steps, 1), eta_min=args.lr * 0.01
    )
    best_exact = math.inf
    best_state = None
    history = []

    for step in range(args.steps):
        model.train()
        progress = step / max(args.steps - 1, 1)
        amp_enabled = args.amp and device.type == "cuda" and progress < args.ce_fraction
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=amp_enabled):
            frame = render_for_seg(model, tokens, idx, exact_path=False)
            logits = segnet(frame)
            loss, phase = curriculum_loss(
                logits, tokens, step, args.steps,
                args.ce_fraction, args.softplus_fraction,
            )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        sched.step()

        if step == 0 or (step + 1) % 25 == 0 or step + 1 == args.steps:
            exact = exact_seg(model, segnet, tokens, idx)
            if exact < best_exact:
                best_exact = exact
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            rec = {"step": step + 1, "phase": phase, "loss": float(loss),
                   "exact_seg": exact, "best_exact_seg": best_exact,
                   "lr": opt.param_groups[0]["lr"]}
            history.append(rec)
            print(json.dumps(rec), flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    final_exact = exact_seg(model, segnet, tokens, idx)
    projected_int4 = math.ceil(n_params / 2)
    verdict = final_exact < 4.5e-4
    result = {
        "verdict": "PASS" if verdict else "FAIL",
        "config": vars(args) | {"challenge_root": str(root), "cache": str(args.cache),
                                "out": str(args.out)},
        "pair_ids": pair_ids,
        "params": n_params,
        "projected_model_bytes_int4_excluding_scales": projected_int4,
        "best_exact_seg": best_exact,
        "final_exact_seg": final_exact,
        "history": history,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=str) + "\n")
    save_path = args.save or args.out.with_suffix(".pt")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(), "pair_ids": pair_ids,
        "config": result["config"], "best_exact_seg": best_exact,
    }, save_path)
    print(json.dumps(result, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
