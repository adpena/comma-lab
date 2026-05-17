#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Z6 real-video ego-proxy sweep for the L5-v2 staircase.

This is a no-scorer, non-promotional probe. It tests whether any cheap
real-video ego proxy makes Z6's full FiLM predictor beat the identity control
on a tiny real-video smoke batch before spending on a full scorer-bearing path.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import experiments.train_substrate_time_traveler_l5_z6 as z6_trainer  # noqa: E402
from tac.substrates.time_traveler_l5_z6 import (  # noqa: E402
    Z6PredictiveCodingConfig,
    Z6PredictiveCodingSubstrate,
    pack_archive,
)

SCHEMA = "z6_real_video_ego_proxy_sweep_v1"
PROBE_ID = "z6_real_video_ego_proxy_sweep"
LANE_ID = "lane_time_traveler_l5_z6_l1_scaffold_substrate_build_20260516"
DEFAULT_VIDEO_PATH = REPO_ROOT / "upstream" / "videos" / "0.mkv"
DEFAULT_OUTPUT_JSON = (
    ".omx/research/l5_v2_z6_real_video_ego_proxy_sweep_20260516_codex.json"
)
DEFAULT_OUTPUT_MD = (
    ".omx/research/l5_v2_z6_real_video_ego_proxy_sweep_20260516_codex.md"
)
FALSE_AUTHORITY_FLAGS = {
    "score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "ready_for_paid_dispatch": False,
    "paradigm_claim_allowed": False,
}
PAIRED_CONTROL_INITIALIZATION = "shared_modules_seed_order_matched_v2"
SEMANTIC_EGO_PROXY_IDS = frozenset(
    {
        "frame_delta",
        "moment_proxy",
        "quadrant_delta",
        "posenet_pose",
    }
)


def _tiny_cfg(*, identity_predictor: bool, num_pairs: int = 5) -> Z6PredictiveCodingConfig:
    return Z6PredictiveCodingConfig(
        latent_dim=8,
        decoder_embed_dim=16,
        decoder_channels=(12, 10, 8, 6),
        decoder_num_upsample_blocks=4,
        num_pairs=num_pairs,
        output_height=48,
        output_width=64,
        predictor_hidden_dim=16,
        predictor_film_mlp_hidden_dim=8,
        predictor_kernel_size=3,
        predictor_ego_motion_dim=4,
        identity_predictor=identity_predictor,
        lambda_residual_entropy=1.0,
    )


def _normalize_features(features: torch.Tensor, *, ego_motion_dim: int) -> torch.Tensor:
    if features.dim() != 2:
        raise ValueError(f"features must be 2-D; got {tuple(features.shape)}")
    centered = features - features.mean(dim=0, keepdim=True)
    scale = centered.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    normalized = centered / scale
    if normalized.shape[1] < ego_motion_dim:
        repeats = math.ceil(ego_motion_dim / normalized.shape[1])
        normalized = normalized.repeat(1, repeats)
    return normalized[:, :ego_motion_dim].contiguous()


def _moment_proxy(
    target0: torch.Tensor,
    target1: torch.Tensor,
    *,
    ego_motion_dim: int,
) -> torch.Tensor:
    if target0.shape != target1.shape:
        raise ValueError("target0 and target1 shapes must match")
    _, _, height, width = target0.shape
    y = torch.linspace(-1.0, 1.0, steps=height, device=target0.device).view(
        1, 1, height, 1
    )
    x = torch.linspace(-1.0, 1.0, steps=width, device=target0.device).view(
        1, 1, 1, width
    )
    luma0 = target0.mean(dim=1, keepdim=True)
    luma1 = target1.mean(dim=1, keepdim=True)
    eps = 1e-6
    mass0 = luma0.sum(dim=(2, 3), keepdim=True).clamp_min(eps)
    mass1 = luma1.sum(dim=(2, 3), keepdim=True).clamp_min(eps)
    cx0 = (luma0 * x).sum(dim=(2, 3), keepdim=True) / mass0
    cy0 = (luma0 * y).sum(dim=(2, 3), keepdim=True) / mass0
    cx1 = (luma1 * x).sum(dim=(2, 3), keepdim=True) / mass1
    cy1 = (luma1 * y).sum(dim=(2, 3), keepdim=True) / mass1
    diff = luma1 - luma0
    features = torch.cat(
        [
            (cx1 - cx0).flatten(1),
            (cy1 - cy0).flatten(1),
            diff.mean(dim=(2, 3)).flatten(1),
            diff.abs().mean(dim=(2, 3)).flatten(1),
        ],
        dim=1,
    )
    return _normalize_features(features, ego_motion_dim=ego_motion_dim)


def _quadrant_delta_proxy(
    target0: torch.Tensor,
    target1: torch.Tensor,
    *,
    ego_motion_dim: int,
) -> torch.Tensor:
    diff = (target1 - target0).mean(dim=1, keepdim=True)
    pooled = torch.nn.functional.adaptive_avg_pool2d(diff, (2, 2)).flatten(1)
    return _normalize_features(pooled, ego_motion_dim=ego_motion_dim)


def build_ego_proxy_candidates(
    target0: torch.Tensor,
    target1: torch.Tensor,
    *,
    ego_motion_dim: int,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    """Return deterministic candidate ego proxies for real-video smoke."""

    frame_delta = z6_trainer._ego_motion_from_smoke_targets(
        target0,
        target1,
        ego_motion_dim=ego_motion_dim,
    )
    ramp = torch.linspace(
        -1.0,
        1.0,
        steps=target0.shape[0] * ego_motion_dim,
        device=target0.device,
    ).view(target0.shape[0], ego_motion_dim)
    generator = torch.Generator(device="cpu").manual_seed(seed + 17)
    random_proxy = torch.randn(
        target0.shape[0],
        ego_motion_dim,
        generator=generator,
        device="cpu",
    ).to(target0.device)
    return {
        "zero": torch.zeros_like(frame_delta),
        "ramp": ramp,
        "frame_delta": frame_delta,
        "moment_proxy": _moment_proxy(
            target0,
            target1,
            ego_motion_dim=ego_motion_dim,
        ),
        "quadrant_delta": _quadrant_delta_proxy(
            target0,
            target1,
            ego_motion_dim=ego_motion_dim,
        ),
        "random_control": random_proxy,
    }


def build_posenet_pose_proxy(
    target0: torch.Tensor,
    target1: torch.Tensor,
    *,
    upstream_dir: Path,
    ego_motion_dim: int,
    device: str = "cpu",
    batch_size: int = 8,
) -> torch.Tensor:
    """Return a PoseNet-derived ego proxy for the tiny smoke targets.

    This is a compress-time/scientific probe only. It loads the official
    PoseNet scorer locally, extracts the first six pose coordinates from each
    target pair, normalizes them, and truncates/repeats to the Z6 ego-motion
    dimension. The resulting proxy is not score authority and is never used at
    inflate time.
    """

    from tac.scorer import load_default_scorers

    if target0.shape != target1.shape:
        raise ValueError("target0 and target1 shapes must match")
    if target0.dim() != 4 or target0.shape[1] != 3:
        raise ValueError("targets must be shaped (num_pairs, 3, height, width)")
    posenet, _segnet = load_default_scorers(upstream_dir, device=device)
    pairs = torch.stack([target0, target1], dim=1).to(
        device=device,
        dtype=torch.float32,
    ) * 255.0
    chunks: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, pairs.shape[0], max(1, int(batch_size))):
            batch = pairs[start : start + max(1, int(batch_size))]
            out = posenet(posenet.preprocess_input(batch))
            chunks.append(out["pose"][..., :6].detach().to("cpu", dtype=torch.float32))
    pose = torch.cat(chunks, dim=0)
    return _normalize_features(pose, ego_motion_dim=ego_motion_dim).to(target0.device)


def _train_one(
    *,
    target0: torch.Tensor,
    target1: torch.Tensor,
    ego_motion: torch.Tensor,
    proxy_id: str,
    identity_predictor: bool,
    epochs: int,
    seed: int,
    lr: float,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    cfg = _tiny_cfg(identity_predictor=identity_predictor, num_pairs=target0.shape[0])
    substrate = Z6PredictiveCodingSubstrate(cfg).to(target0.device)
    substrate.ego_motion_buffer.copy_(ego_motion.to(target0.device))
    opt = torch.optim.AdamW(substrate.parameters(), lr=lr)
    losses: list[dict[str, float]] = []
    for epoch in range(max(1, min(int(epochs), 3))):
        opt.zero_grad()
        idx = torch.arange(cfg.num_pairs, device=target0.device, dtype=torch.long)
        rgb0, rgb1, _z = substrate.reconstruct_pair(idx)
        recon_loss = (rgb0 - target0).pow(2).mean() + (rgb1 - target1).pow(2).mean()
        residual_loss = cfg.lambda_residual_entropy * substrate.residuals.pow(2).mean()
        loss = recon_loss + residual_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(substrate.parameters(), 1.0)
        opt.step()
        losses.append(
            {
                "epoch": float(epoch),
                "loss": float(loss.detach().item()),
                "recon": float(recon_loss.detach().item()),
                "residual": float(residual_loss.detach().item()),
            }
        )
    meta = {
        "encoder_input_channels": cfg.encoder_input_channels,
        "encoder_hidden_dim": cfg.encoder_hidden_dim,
        "decoder_embed_dim": cfg.decoder_embed_dim,
        "decoder_initial_grid_h": cfg.decoder_initial_grid_h,
        "decoder_initial_grid_w": cfg.decoder_initial_grid_w,
        "decoder_channels": list(cfg.decoder_channels),
        "decoder_num_upsample_blocks": cfg.decoder_num_upsample_blocks,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
        "predictor_hidden_dim": cfg.predictor_hidden_dim,
        "predictor_film_mlp_hidden_dim": cfg.predictor_film_mlp_hidden_dim,
        "latent_init_std": cfg.latent_init_std,
        "smoke": True,
        "smoke_target_mode": "real-video",
        "smoke_ego_proxy_id": proxy_id,
        "requested_epochs": epochs,
        "effective_epochs": len(losses),
    }
    archive_bytes = pack_archive(
        substrate.encoder.state_dict(),
        substrate.decoder.state_dict(),
        substrate.predictor.state_dict(),
        substrate.latent_init.detach().cpu(),
        substrate.residuals.detach().cpu(),
        substrate.ego_motion_buffer.detach().cpu(),
        meta,
        lambda_residual_entropy=cfg.lambda_residual_entropy,
        predictor_kernel_size=cfg.predictor_kernel_size,
        identity_predictor=identity_predictor,
    )
    final = losses[-1]
    return {
        "proxy_id": proxy_id,
        "identity_predictor": identity_predictor,
        "paired_control_initialization": PAIRED_CONTROL_INITIALIZATION,
        "paired_control_shared_modules": [
            "encoder",
            "decoder",
            "latent_init",
            "residuals",
            "ego_motion_buffer",
        ],
        "final_loss_proxy": final["loss"],
        "final_recon": final["recon"],
        "final_residual": final["residual"],
        "archive_bytes": len(archive_bytes),
        "ego_motion_nonzero_fraction": float((ego_motion.abs() > 0).float().mean().item()),
        "ego_motion_l2": float(ego_motion.pow(2).sum().sqrt().item()),
        "param_breakdown": substrate.num_parameters_breakdown(),
        "cfg": asdict(cfg),
        "losses": losses,
    }


def run_sweep_on_targets(
    *,
    target0: torch.Tensor,
    target1: torch.Tensor,
    extra_ego_proxies: dict[str, torch.Tensor] | None = None,
    epochs: int = 3,
    seed: int = 0,
    lr: float = 5e-4,
) -> dict[str, Any]:
    """Run the real-video ego-proxy sweep on already-decoded targets."""

    candidates = build_ego_proxy_candidates(
        target0,
        target1,
        ego_motion_dim=4,
        seed=seed,
    )
    if extra_ego_proxies:
        for proxy_id, ego_motion in extra_ego_proxies.items():
            if proxy_id in candidates:
                raise ValueError(f"duplicate ego proxy candidate: {proxy_id!r}")
            if tuple(ego_motion.shape) != tuple(next(iter(candidates.values())).shape):
                raise ValueError(
                    f"ego proxy {proxy_id!r} shape {tuple(ego_motion.shape)} "
                    f"!= expected {tuple(next(iter(candidates.values())).shape)}"
                )
            candidates[proxy_id] = ego_motion
    rows: list[dict[str, Any]] = []
    for proxy_id, ego_motion in candidates.items():
        full = _train_one(
            target0=target0,
            target1=target1,
            ego_motion=ego_motion,
            proxy_id=proxy_id,
            identity_predictor=False,
            epochs=epochs,
            seed=seed,
            lr=lr,
        )
        identity = _train_one(
            target0=target0,
            target1=target1,
            ego_motion=ego_motion,
            proxy_id=proxy_id,
            identity_predictor=True,
            epochs=epochs,
            seed=seed,
            lr=lr,
        )
        delta_loss = float(identity["final_loss_proxy"]) - float(full["final_loss_proxy"])
        rows.append(
            {
                "proxy_id": proxy_id,
                "full_film": full,
                "identity": identity,
                "identity_minus_full_loss_proxy": delta_loss,
                "identity_minus_full_recon": (
                    float(identity["final_recon"]) - float(full["final_recon"])
                ),
                "identity_minus_full_residual": (
                    float(identity["final_residual"]) - float(full["final_residual"])
                ),
                "full_minus_identity_archive_bytes": (
                    int(full["archive_bytes"]) - int(identity["archive_bytes"])
                ),
                "full_film_proxy_wins": delta_loss > 0.0,
            }
        )
    best_row = max(rows, key=lambda row: float(row["identity_minus_full_loss_proxy"]))
    verdict = (
        "full_film_proxy_found_real_video_smoke"
        if bool(best_row["full_film_proxy_wins"])
        else "identity_dominates_all_tested_ego_proxies_real_video_smoke"
    )
    semantic_proxy_supported = bool(best_row["full_film_proxy_wins"]) and str(
        best_row["proxy_id"]
    ) in SEMANTIC_EGO_PROXY_IDS
    posenet_proxy_tested = "posenet_pose" in candidates
    blockers = [
        "real_video_smoke_proxy_no_scorer",
        "no_contest_cpu_cuda_pair",
        "no_byte_closed_score_anchor",
        "not_paradigm_claim_authority",
    ]
    if not semantic_proxy_supported:
        blockers.append("ego_proxy_semantics_not_hard_earned")
    if posenet_proxy_tested and str(best_row["proxy_id"]) != "posenet_pose":
        blockers.append("posenet_pose_proxy_not_best")
    if bool(best_row["full_film_proxy_wins"]) and posenet_proxy_tested and not semantic_proxy_supported:
        recommended_next_actions = [
            "do not paid-dispatch Z6-v1 full-FiLM from this probe: PoseNet-derived ego did not beat random/zero controls",
            "either run a true scorer-bearing paired probe or redesign the ego-conditioning objective before full_main",
            "advance Z7/Z8 only as new measured configurations, not as an automatic Z6-v1 promotion",
        ]
    elif bool(best_row["full_film_proxy_wins"]):
        recommended_next_actions = [
            "treat the full-FiLM win as predictor-capacity liveness, not a score or paradigm claim",
            "run a scorer-bearing or PoseNet-derived ego proxy probe before paid dispatch",
            "if the best proxy remains zero/random, redesign the ego-conditioning objective before full_main",
        ]
    else:
        recommended_next_actions = [
            "do not spend on Z6 full-FiLM until a scorer-bearing or PoseNet-derived ego proxy beats identity in smoke",
            "try a PoseNet/SegNet-derived ego proxy or redesign predictor objective before paid dispatch",
            "if no proxy beats identity, retire Z6-v1 FiLM as measured configuration only, not the whole L5 staircase",
        ]
    return {
        "schema": SCHEMA,
        "probe_id": PROBE_ID,
        "lane_id": LANE_ID,
        "evidence_grade": "real_video_smoke_proxy_no_scorer",
        "verdict": verdict,
        "paired_control_initialization": PAIRED_CONTROL_INITIALIZATION,
        **FALSE_AUTHORITY_FLAGS,
        "epochs": max(1, min(int(epochs), 3)),
        "seed": seed,
        "candidate_count": len(rows),
        "posenet_proxy_tested": posenet_proxy_tested,
        "semantic_ego_proxy_ids": sorted(SEMANTIC_EGO_PROXY_IDS),
        "best_proxy_id": best_row["proxy_id"],
        "semantic_ego_proxy_supported": semantic_proxy_supported,
        "best_identity_minus_full_loss_proxy": best_row[
            "identity_minus_full_loss_proxy"
        ],
        "rows": rows,
        "blockers": blockers,
        "recommended_next_actions": recommended_next_actions,
    }


def run_sweep_from_video(
    *,
    video_path: Path,
    device: str = "cpu",
    include_posenet_proxy: bool = False,
    upstream_dir: Path = REPO_ROOT / "upstream",
    epochs: int = 3,
    seed: int = 0,
    lr: float = 5e-4,
) -> dict[str, Any]:
    video_path = video_path.expanduser().resolve()
    cfg = _tiny_cfg(identity_predictor=False)
    target0, target1, _ego = z6_trainer._decode_real_video_smoke_targets(
        video_path,
        cfg,
        device=device,
    )
    extra_ego_proxies: dict[str, torch.Tensor] = {}
    if include_posenet_proxy:
        extra_ego_proxies["posenet_pose"] = build_posenet_pose_proxy(
            target0,
            target1,
            upstream_dir=upstream_dir,
            ego_motion_dim=cfg.predictor_ego_motion_dim,
            device=device,
        )
    payload = run_sweep_on_targets(
        target0=target0,
        target1=target1,
        extra_ego_proxies=extra_ego_proxies,
        epochs=epochs,
        seed=seed,
        lr=lr,
    )
    try:
        payload["video_path"] = str(video_path.relative_to(REPO_ROOT))
    except ValueError:
        payload["video_path"] = video_path.name
    payload["device"] = device
    payload["include_posenet_proxy"] = include_posenet_proxy
    return payload


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# L5 v2 Z6 real-video ego-proxy sweep",
        "",
        f"- schema: `{payload.get('schema')}`",
        f"- probe_id: `{payload.get('probe_id')}`",
        f"- lane_id: `{payload.get('lane_id')}`",
        f"- evidence_grade: `{payload.get('evidence_grade')}`",
        f"- verdict: `{payload.get('verdict')}`",
        f"- paired_control_initialization: `{payload.get('paired_control_initialization')}`",
        f"- best_proxy_id: `{payload.get('best_proxy_id')}`",
        f"- posenet_proxy_tested: `{payload.get('posenet_proxy_tested')}`",
        f"- semantic_ego_proxy_supported: `{payload.get('semantic_ego_proxy_supported')}`",
        f"- semantic_ego_proxy_ids: `{payload.get('semantic_ego_proxy_ids')}`",
        f"- best_identity_minus_full_loss_proxy: `{payload.get('best_identity_minus_full_loss_proxy')}`",
        "- score_claim: `false`",
        "- promotion_eligible: `false`",
        "- rank_or_kill_eligible: `false`",
        "- ready_for_exact_eval_dispatch: `false`",
        "- ready_for_paid_dispatch: `false`",
        "- paradigm_claim_allowed: `false`",
        "",
        "This sweep is a no-scorer real-video smoke proxy. It tests whether "
        "any cheap ego proxy makes the Z6 FiLM predictor beat identity before "
        "spending on scorer-bearing training or exact eval.",
        "",
        "## Rows",
    ]
    for row in payload.get("rows", []):
        lines.extend(
            [
                "",
                f"### {row['proxy_id']}",
                "",
                f"- full_film_proxy_wins: `{row['full_film_proxy_wins']}`",
                f"- identity_minus_full_loss_proxy: `{row['identity_minus_full_loss_proxy']}`",
                f"- identity_minus_full_recon: `{row['identity_minus_full_recon']}`",
                f"- identity_minus_full_residual: `{row['identity_minus_full_residual']}`",
                f"- full_minus_identity_archive_bytes: `{row['full_minus_identity_archive_bytes']}`",
                f"- full_film_loss_proxy: `{row['full_film']['final_loss_proxy']}`",
                f"- identity_loss_proxy: `{row['identity']['final_loss_proxy']}`",
                f"- full_paired_control_initialization: `{row['full_film']['paired_control_initialization']}`",
                f"- identity_paired_control_initialization: `{row['identity']['paired_control_initialization']}`",
            ]
        )
    for section, key in (
        ("Blockers", "blockers"),
        ("Recommended Next Actions", "recommended_next_actions"),
    ):
        lines.extend(["", f"## {section}"])
        for value in payload.get(key, []):
            lines.append(f"- {value}")
    return "\n".join(lines) + "\n"


def _resolve_output(path: Path, *, repo_root: Path) -> Path:
    resolved = path if path.is_absolute() else repo_root / path
    resolved = resolved.expanduser().resolve()
    resolved.relative_to(repo_root)
    text = str(resolved)
    if (
        text.startswith("/tmp/")
        or "/private/tmp/" in text
        or "/var/tmp/" in text
    ):
        raise ValueError(f"refusing to write Z6 ego-proxy sweep to tmp: {text!r}")
    return resolved


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--video-path", type=Path, default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--upstream-dir", type=Path, default=REPO_ROOT / "upstream")
    parser.add_argument(
        "--include-posenet-proxy",
        action="store_true",
        help=(
            "Add a local PoseNet-derived ego proxy candidate. This loads the "
            "official PoseNet scorer for probe construction only; output remains "
            "no-score and non-promotional."
        ),
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--output-json", type=Path, default=Path(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-md", type=Path, default=Path(DEFAULT_OUTPUT_MD))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.expanduser().resolve()
    try:
        video_path = args.video_path
        if not video_path.is_absolute():
            video_path = repo_root / video_path
        payload = run_sweep_from_video(
            video_path=video_path,
            device=args.device,
            include_posenet_proxy=args.include_posenet_proxy,
            upstream_dir=(
                args.upstream_dir
                if args.upstream_dir.is_absolute()
                else repo_root / args.upstream_dir
            ),
            epochs=args.epochs,
            seed=args.seed,
            lr=args.lr,
        )
        output_json = _resolve_output(args.output_json, repo_root=repo_root)
        output_md = _resolve_output(args.output_md, repo_root=repo_root)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(render_markdown(payload), encoding="utf-8")
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"[z6-ego-proxy-sweep] FATAL: {exc}", file=sys.stderr)
        return 2
    print(
        "[z6-ego-proxy-sweep] "
        f"verdict={payload['verdict']} "
        f"best_proxy={payload['best_proxy_id']} "
        "score_claim=false promotion_eligible=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
