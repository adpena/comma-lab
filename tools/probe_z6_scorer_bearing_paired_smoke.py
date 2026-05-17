#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Z6 scorer-bearing paired smoke for the L5-v2 staircase.

This probe is intentionally tiny and non-promotional. It trains full-FiLM and
identity-predictor Z6 arms under the same shared-module initialization while the
loss path carries SegNet/PoseNet terms through
``Z6PredictiveCodingScoreAwareLoss``. The output is a dispatch gate artifact,
not a contest score.
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
from tac.scorer import load_differentiable_scorers  # noqa: E402
from tac.substrates.time_traveler_l5_z6 import (  # noqa: E402
    Z6PredictiveCodingLossWeights,
    Z6PredictiveCodingScoreAwareLoss,
    Z6PredictiveCodingSubstrate,
    pack_archive,
)
from tools.probe_z6_real_video_ego_proxy_sweep import (  # noqa: E402
    PAIRED_CONTROL_INITIALIZATION,
    SEMANTIC_EGO_PROXY_IDS,
    _normalize_features,
    _tiny_cfg,
    build_ego_proxy_candidates,
)

SCHEMA = "z6_scorer_bearing_paired_smoke_v1"
PROBE_ID = "z6_scorer_bearing_paired_smoke"
LANE_ID = "lane_time_traveler_l5_z6_l1_scaffold_substrate_build_20260516"
DEFAULT_VIDEO_PATH = REPO_ROOT / "upstream" / "videos" / "0.mkv"
DEFAULT_UPSTREAM_DIR = REPO_ROOT / "upstream"
DEFAULT_OUTPUT_JSON = (
    ".omx/research/l5_v2_z6_scorer_bearing_paired_smoke_20260517_codex.json"
)
DEFAULT_OUTPUT_MD = (
    ".omx/research/l5_v2_z6_scorer_bearing_paired_smoke_20260517_codex.md"
)
CONTEST_NORMALIZER_BYTES = 37_545_489.0
FALSE_AUTHORITY_FLAGS = {
    "score_claim": False,
    "promotion_eligible": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "ready_for_paid_dispatch": False,
    "paradigm_claim_allowed": False,
}


def _pack_arm_archive_bytes(substrate: Z6PredictiveCodingSubstrate) -> bytes:
    cfg = substrate.cfg
    meta = {
        "cfg": asdict(cfg),
        "probe_id": PROBE_ID,
        "identity_predictor": cfg.identity_predictor,
        "score_claim": False,
    }
    return pack_archive(
        substrate.encoder.state_dict(),
        substrate.decoder.state_dict(),
        substrate.predictor.state_dict(),
        substrate.latent_init.detach().cpu(),
        substrate.residuals.detach().cpu(),
        substrate.ego_motion_buffer.detach().cpu(),
        meta,
        lambda_residual_entropy=cfg.lambda_residual_entropy,
        predictor_kernel_size=cfg.predictor_kernel_size,
        identity_predictor=cfg.identity_predictor,
    )


def _contest_like_score_proxy(
    *,
    pose_term: float,
    seg_term: float,
    archive_bytes: int,
) -> dict[str, float]:
    rate = float(archive_bytes) / CONTEST_NORMALIZER_BYTES
    pose_contribution = math.sqrt(max(0.0, 10.0 * pose_term))
    seg_contribution = 100.0 * seg_term
    rate_contribution = 25.0 * rate
    return {
        "score_proxy": seg_contribution + pose_contribution + rate_contribution,
        "pose_term": pose_term,
        "seg_term": seg_term,
        "rate": rate,
        "pose_contribution": pose_contribution,
        "seg_contribution": seg_contribution,
        "rate_contribution": rate_contribution,
    }


def build_posenet_pose_proxy_from_loaded_scorer(
    target0: torch.Tensor,
    target1: torch.Tensor,
    *,
    posenet: torch.nn.Module,
    ego_motion_dim: int,
    device: str = "cpu",
    batch_size: int = 8,
) -> torch.Tensor:
    """Build the PoseNet-derived ego proxy without loading scorers twice."""

    if target0.shape != target1.shape:
        raise ValueError("target0 and target1 shapes must match")
    pairs = torch.stack([target0, target1], dim=1).to(
        device=device,
        dtype=torch.float32,
    ) * 255.0
    chunks: list[torch.Tensor] = []
    with torch.inference_mode():
        for start in range(0, pairs.shape[0], max(1, int(batch_size))):
            batch = pairs[start : start + max(1, int(batch_size))]
            out = posenet(posenet.preprocess_input(batch))
            chunks.append(out["pose"][..., :6].detach().cpu().float())
    pose = torch.cat(chunks, dim=0)
    return _normalize_features(pose, ego_motion_dim=ego_motion_dim).to(target0.device)


def train_scorer_bearing_arm(
    *,
    target0: torch.Tensor,
    target1: torch.Tensor,
    ego_motion: torch.Tensor,
    proxy_id: str,
    identity_predictor: bool,
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    epochs: int = 1,
    seed: int = 0,
    lr: float = 2e-4,
    lambda_residual_entropy: float = 1.0,
) -> dict[str, Any]:
    """Train one Z6 arm under the scorer-bearing loss path."""

    torch.manual_seed(int(seed))
    cfg = _tiny_cfg(
        identity_predictor=identity_predictor,
        num_pairs=target0.shape[0],
    )
    cfg = type(cfg)(
        **{
            **asdict(cfg),
            "lambda_residual_entropy": float(lambda_residual_entropy),
        }
    )
    substrate = Z6PredictiveCodingSubstrate(cfg).to(target0.device)
    substrate.ego_motion_buffer.copy_(ego_motion.to(target0.device))
    loss_fn = Z6PredictiveCodingScoreAwareLoss(
        seg_scorer=segnet,
        pose_scorer=posenet,
        weights=Z6PredictiveCodingLossWeights(
            lambda_residual_entropy=float(lambda_residual_entropy)
        ),
    ).to(target0.device)
    loss_fn.train()
    opt = torch.optim.AdamW(substrate.parameters(), lr=float(lr))

    target0_255 = target0 * 255.0
    target1_255 = target1 * 255.0
    losses: list[dict[str, float]] = []
    for epoch in range(max(1, min(int(epochs), 2))):
        opt.zero_grad()
        idx = torch.arange(cfg.num_pairs, device=target0.device, dtype=torch.long)
        rgb0, rgb1, _z = substrate.reconstruct_pair(idx)
        archive_bytes_proxy = torch.tensor(
            float(len(_pack_arm_archive_bytes(substrate))),
            device=target0.device,
        )
        loss, parts = loss_fn(
            reconstructed_rgb_0=rgb0 * 255.0,
            reconstructed_rgb_1=rgb1 * 255.0,
            gt_rgb_0=target0_255,
            gt_rgb_1=target1_255,
            archive_bytes_proxy=archive_bytes_proxy,
            residuals=substrate.residuals,
            apply_eval_roundtrip=True,
            noise_std=0.0,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(substrate.parameters(), 1.0)
        opt.step()
        losses.append(
            {
                "epoch": float(epoch),
                "loss_total": float(loss.detach().cpu().item()),
                "seg_term": float(parts["seg_term"].cpu().item()),
                "pose_term": float(parts["pose_term"].cpu().item()),
                "pc_term": float(parts["pc_term"].cpu().item()),
                "rate_term": float(parts["rate_term"].cpu().item()),
            }
        )

    loss_fn.eval()
    with torch.no_grad():
        idx = torch.arange(cfg.num_pairs, device=target0.device, dtype=torch.long)
        rgb0, rgb1, _z = substrate.reconstruct_pair(idx)
        archive_bytes = len(_pack_arm_archive_bytes(substrate))
        final_loss, final_parts = loss_fn(
            reconstructed_rgb_0=rgb0 * 255.0,
            reconstructed_rgb_1=rgb1 * 255.0,
            gt_rgb_0=target0_255,
            gt_rgb_1=target1_255,
            archive_bytes_proxy=torch.tensor(float(archive_bytes), device=target0.device),
            residuals=substrate.residuals,
            apply_eval_roundtrip=True,
            noise_std=0.0,
        )
    score_proxy = _contest_like_score_proxy(
        pose_term=float(final_parts["pose_term"].cpu().item()),
        seg_term=float(final_parts["seg_term"].cpu().item()),
        archive_bytes=archive_bytes,
    )
    return {
        "proxy_id": proxy_id,
        "identity_predictor": bool(identity_predictor),
        "paired_control_initialization": PAIRED_CONTROL_INITIALIZATION,
        "epochs": len(losses),
        "losses": losses,
        "final_loss_total": float(final_loss.cpu().item()),
        "final_pc_term": float(final_parts["pc_term"].cpu().item()),
        "archive_bytes": archive_bytes,
        "param_breakdown": substrate.num_parameters_breakdown(),
        **score_proxy,
    }


def run_scorer_bearing_probe_on_targets(
    *,
    target0: torch.Tensor,
    target1: torch.Tensor,
    posenet: torch.nn.Module,
    segnet: torch.nn.Module,
    candidate_ids: tuple[str, ...] = ("posenet_pose", "random_control", "zero"),
    epochs: int = 1,
    seed: int = 0,
    lr: float = 2e-4,
    lambda_residual_entropy: float = 1.0,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run full-FiLM vs identity under scorer-bearing loss for each proxy."""

    cfg = _tiny_cfg(identity_predictor=False, num_pairs=target0.shape[0])
    candidates = build_ego_proxy_candidates(
        target0,
        target1,
        ego_motion_dim=cfg.predictor_ego_motion_dim,
        seed=seed,
    )
    candidates["posenet_pose"] = build_posenet_pose_proxy_from_loaded_scorer(
        target0,
        target1,
        posenet=posenet,
        ego_motion_dim=cfg.predictor_ego_motion_dim,
        device=device,
    )
    missing = [candidate for candidate in candidate_ids if candidate not in candidates]
    if missing:
        raise ValueError(f"unknown candidate id(s): {missing}")

    rows: list[dict[str, Any]] = []
    for candidate_idx, proxy_id in enumerate(candidate_ids):
        ego_motion = candidates[proxy_id]
        full = train_scorer_bearing_arm(
            target0=target0,
            target1=target1,
            ego_motion=ego_motion,
            proxy_id=proxy_id,
            identity_predictor=False,
            posenet=posenet,
            segnet=segnet,
            epochs=epochs,
            seed=seed,
            lr=lr,
            lambda_residual_entropy=lambda_residual_entropy,
        )
        identity = train_scorer_bearing_arm(
            target0=target0,
            target1=target1,
            ego_motion=ego_motion,
            proxy_id=proxy_id,
            identity_predictor=True,
            posenet=posenet,
            segnet=segnet,
            epochs=epochs,
            seed=seed,
            lr=lr,
            lambda_residual_entropy=lambda_residual_entropy,
        )
        delta = float(identity["score_proxy"]) - float(full["score_proxy"])
        rows.append(
            {
                "proxy_id": proxy_id,
                "candidate_index": candidate_idx,
                "full_film": full,
                "identity": identity,
                "full_film_scorer_proxy_wins": delta > 0.0,
                "identity_minus_full_score_proxy": delta,
                "identity_minus_full_seg_term": float(identity["seg_term"])
                - float(full["seg_term"]),
                "identity_minus_full_pose_term": float(identity["pose_term"])
                - float(full["pose_term"]),
                "full_minus_identity_archive_bytes": int(full["archive_bytes"])
                - int(identity["archive_bytes"]),
            }
        )

    best_row = max(rows, key=lambda row: float(row["identity_minus_full_score_proxy"]))
    semantic_scorer_proxy_supported = bool(
        best_row["full_film_scorer_proxy_wins"]
    ) and str(best_row["proxy_id"]) in SEMANTIC_EGO_PROXY_IDS
    blockers = [
        "tiny_cpu_scorer_bearing_proxy_not_contest_cpu_or_cuda",
        "no_byte_closed_archive_eval",
        "no_contest_cpu_cuda_pair",
        "not_paradigm_claim_authority",
    ]
    if not semantic_scorer_proxy_supported:
        blockers.append("scorer_bearing_semantics_not_hard_earned")
    if str(best_row["proxy_id"]) != "posenet_pose":
        blockers.append("posenet_pose_scorer_proxy_not_best")

    if semantic_scorer_proxy_supported:
        verdict = "semantic_full_film_scorer_proxy_found"
        recommended_next_actions = [
            "run a paid scorer-bearing timing smoke only after lane dispatch claim",
            "keep score_claim=false until byte-closed CPU/CUDA exact eval exists",
        ]
    else:
        verdict = "semantic_full_film_scorer_proxy_not_found"
        recommended_next_actions = [
            "do not paid-dispatch Z6-v1 full-FiLM from this tiny scorer-bearing probe",
            "redesign ego-conditioning or advance Z7/Z8 as new measured configurations",
            "preserve this as a Z6 trust-region update, not a lane kill",
        ]

    return {
        "schema": SCHEMA,
        "probe_id": PROBE_ID,
        "lane_id": LANE_ID,
        "evidence_grade": "tiny_cpu_scorer_bearing_proxy_no_archive_eval",
        "hardware_axis": "local_cpu_proxy_not_contest_cpu",
        "paired_control_initialization": PAIRED_CONTROL_INITIALIZATION,
        "epochs": max(1, min(int(epochs), 2)),
        "seed": seed,
        "candidate_ids": list(candidate_ids),
        "candidate_count": len(rows),
        "best_proxy_id": best_row["proxy_id"],
        "best_identity_minus_full_score_proxy": best_row[
            "identity_minus_full_score_proxy"
        ],
        "semantic_scorer_proxy_supported": semantic_scorer_proxy_supported,
        "semantic_ego_proxy_ids": sorted(SEMANTIC_EGO_PROXY_IDS),
        "verdict": verdict,
        "blockers": blockers,
        "recommended_next_actions": recommended_next_actions,
        "rows": rows,
        **FALSE_AUTHORITY_FLAGS,
    }


def run_scorer_bearing_probe_from_video(
    *,
    video_path: Path,
    upstream_dir: Path,
    device: str = "cpu",
    num_pairs: int = 2,
    candidate_ids: tuple[str, ...] = ("posenet_pose", "random_control", "zero"),
    epochs: int = 1,
    seed: int = 0,
    lr: float = 2e-4,
    lambda_residual_entropy: float = 1.0,
) -> dict[str, Any]:
    """Load real-video targets and official scorers, then run the probe."""

    cfg = _tiny_cfg(identity_predictor=False, num_pairs=max(2, int(num_pairs)))
    target0, target1, _ego = z6_trainer._decode_real_video_smoke_targets(
        video_path,
        cfg,
        device=device,
    )
    posenet, segnet = load_differentiable_scorers(upstream_dir, device=device)
    payload = run_scorer_bearing_probe_on_targets(
        target0=target0,
        target1=target1,
        posenet=posenet,
        segnet=segnet,
        candidate_ids=candidate_ids,
        epochs=epochs,
        seed=seed,
        lr=lr,
        lambda_residual_entropy=lambda_residual_entropy,
        device=device,
    )
    try:
        payload["video_path"] = str(video_path.relative_to(REPO_ROOT))
    except ValueError:
        payload["video_path"] = str(video_path)
    payload["upstream_dir"] = str(upstream_dir)
    payload["device"] = device
    payload["num_pairs"] = cfg.num_pairs
    return payload


def render_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# L5-v2 Z6 Scorer-Bearing Paired Smoke",
        "",
        "This is a tiny local scorer-bearing proxy. It is not an exact contest",
        "CPU/CUDA eval and cannot promote or rank a submission.",
        "",
        "## Summary",
        f"- schema: `{payload.get('schema')}`",
        f"- verdict: `{payload.get('verdict')}`",
        f"- evidence_grade: `{payload.get('evidence_grade')}`",
        f"- hardware_axis: `{payload.get('hardware_axis')}`",
        f"- paired_control_initialization: `{payload.get('paired_control_initialization')}`",
        f"- best_proxy_id: `{payload.get('best_proxy_id')}`",
        f"- best_identity_minus_full_score_proxy: `{payload.get('best_identity_minus_full_score_proxy')}`",
        f"- semantic_scorer_proxy_supported: `{payload.get('semantic_scorer_proxy_supported')}`",
        "- score_claim: `false`",
        "- promotion_eligible: `false`",
        "- ready_for_paid_dispatch: `false`",
        "",
        "## Candidate Rows",
    ]
    for row in payload.get("rows", []):
        lines.extend(
            [
                "",
                f"### {row.get('proxy_id')}",
                f"- full_film_scorer_proxy_wins: `{row.get('full_film_scorer_proxy_wins')}`",
                f"- identity_minus_full_score_proxy: `{row.get('identity_minus_full_score_proxy')}`",
                f"- identity_minus_full_seg_term: `{row.get('identity_minus_full_seg_term')}`",
                f"- identity_minus_full_pose_term: `{row.get('identity_minus_full_pose_term')}`",
                f"- full_minus_identity_archive_bytes: `{row.get('full_minus_identity_archive_bytes')}`",
            ]
        )
    lines.extend(["", "## Blockers"])
    lines.extend(f"- {blocker}" for blocker in payload.get("blockers", []))
    lines.extend(["", "## Recommended Next Actions"])
    lines.extend(f"- {action}" for action in payload.get("recommended_next_actions", []))
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--video-path", type=Path, default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-pairs", type=int, default=2)
    parser.add_argument("--candidate", action="append", dest="candidates")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lambda-residual-entropy", type=float, default=1.0)
    parser.add_argument("--output-json", type=Path, default=Path(DEFAULT_OUTPUT_JSON))
    parser.add_argument("--output-md", type=Path, default=Path(DEFAULT_OUTPUT_MD))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    video_path = args.video_path if args.video_path.is_absolute() else repo_root / args.video_path
    upstream_dir = (
        args.upstream_dir if args.upstream_dir.is_absolute() else repo_root / args.upstream_dir
    )
    candidate_ids = tuple(args.candidates or ("posenet_pose", "random_control", "zero"))
    payload = run_scorer_bearing_probe_from_video(
        video_path=video_path,
        upstream_dir=upstream_dir,
        device=args.device,
        num_pairs=args.num_pairs,
        candidate_ids=candidate_ids,
        epochs=args.epochs,
        seed=args.seed,
        lr=args.lr,
        lambda_residual_entropy=args.lambda_residual_entropy,
    )
    output_json = args.output_json if args.output_json.is_absolute() else repo_root / args.output_json
    output_md = args.output_md if args.output_md.is_absolute() else repo_root / args.output_md
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    output_md.write_text(render_markdown(payload), encoding="utf-8")
    print(
        f"[z6-scorer-bearing-smoke] verdict={payload['verdict']} "
        f"best_proxy={payload['best_proxy_id']} score_claim=false "
        f"promotion_eligible=false"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
