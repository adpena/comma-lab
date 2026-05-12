#!/usr/bin/env python3
"""BlockNeRV-as-renderer — production trainer.

Per operator directive 2026-05-11 (NeRV-family expansion) + CLAUDE.md HNeRV
parity discipline. Mirrors the Lane 12-v2 production trainer pattern with
substrate-specific imports for BlockNeRV.

REPRESENTATION_ARCHIVE_GRAMMAR_BLUEPRINT (per CLAUDE.md Catalog #124)
---------------------------------------------------------------------
  archive_grammar: monolithic single-file 0.bin (16-byte fixed header +
    5 length-prefixed sections per src/tac/blocknerv_as_renderer.py
    ARCHIVE_GRAMMAR_BLOCKNERV)
  parser_section_manifest: ARCHIVE_GRAMMAR_BLOCKNERV in
    src/tac/blocknerv_as_renderer.py with schema_keys_in_order =
    BlockNeRVRenderer.SCHEMA (pinned)
  inflate_runtime_loc_budget: substrate_engineering — Phase B inflate ≤200 LOC
  runtime_dep_closure: torch + brotli (zero dep on tac.*)
  export_format: blocknerv_phase_a_monolithic_singlefile_0bin
  score_aware_loss: train_step_blocknerv with differentiable rgb_to_yuv6 +
    load_differentiable_scorers
  bolt_on_loc_budget: substrate_engineering (full tile-decomposed renderer)
  no_op_detector_planned: export_blocknerv_to_archive returns sha256

CLAUDE.md non-negotiables wired (same as Lane 12-v2):
  - eval_roundtrip=True; EMA decay=0.997 with snapshot+restore
  - differentiable rgb_to_yuv6 monkey-patch BEFORE scorer load
  - score-domain Lagrangian; real video data outside --smoke
  - auth eval gated behind --phase-b-auth-memo (Catalog #150)
  - CUDA-required default; NO MPS; NO /tmp paths

Cost when dispatched: ~$30-50 (Vast.ai T4 ~5-10h). Operator-gated.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


LANE_ID = "lane_blocknerv_as_renderer"
SCHEMA_VERSION = "1.0.0-blocknerv-as-renderer-production"
PREDICTED_DELTA_SCORE = (
    "[predicted; HNeRV parity discipline; tile-decomposed inductive bias gives "
    "~1.2× param efficiency vs single-decoder NeRV; pose-axis marginal 2.71× "
    "SegNet at PR106 r2 frontier]"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="BlockNeRV-as-renderer production trainer")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--ema-decay", type=float, default=0.997)
    p.add_argument("--latent-dim", type=int, default=16)
    p.add_argument("--tile-rows", type=int, default=6)
    p.add_argument("--tile-cols", type=int, default=8)
    p.add_argument("--tile-h", type=int, default=64)
    p.add_argument("--tile-w", type=int, default=64)
    p.add_argument("--coord-embed-dim", type=int, default=8)
    p.add_argument("--base-channels", type=int, default=28)
    p.add_argument("--n-pairs", type=int, default=600)
    p.add_argument("--lambda-seg", type=float, default=100.0)
    p.add_argument("--lambda-pose", type=float, default=288.6751345948129)
    p.add_argument("--grad-clip-norm", type=float, default=1.0)
    p.add_argument("--video-path", type=Path,
                   default=REPO_ROOT / "upstream" / "videos" / "0.mkv")
    p.add_argument("--max-pairs", type=int, default=None)
    p.add_argument("--enable-differentiable-yuv6", action="store_true", default=True)
    p.add_argument("--enable-score-aware-loss", action="store_true", default=True)
    p.add_argument("--seed", type=int, default=20260511)
    p.add_argument("--smoke", action="store_true", default=False)
    p.add_argument("--auth-eval", action="store_true", default=False)
    p.add_argument("--phase-b-auth-memo", type=str, default=None,
                   help="Catalog #150 — repo-relative committed memo path required for --auth-eval.")
    p.add_argument("--eval-every-epochs", type=int, default=25)
    return p.parse_args(argv)


def _resolve_device(device_str: str) -> "torch.device":
    if device_str == "mps":
        raise SystemExit("[blocknerv] --device mps REFUSED per CLAUDE.md")
    if device_str == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                "[blocknerv] --device cuda requested but CUDA unavailable. "
                "Pass --device cpu only with explicit deterministic-bytes-acceptable smoke intent."
            )
        return torch.device("cuda")
    return torch.device("cpu")


def _seed_everything(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _activate_differentiable_yuv6():
    from tac.differentiable_eval_roundtrip import patch_upstream_yuv6_globally
    return patch_upstream_yuv6_globally()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.auth_eval:
        if args.phase_b_auth_memo is None:
            raise SystemExit(
                "[blocknerv] --auth-eval refused: requires --phase-b-auth-memo "
                "<committed_repo_path> per Catalog #150."
            )
        from tac.lane_12_v2_nerv_as_renderer import phase_b_preconditions_status
        status = phase_b_preconditions_status(
            consult_session_state=True,
            auth_memo_path=Path(args.phase_b_auth_memo),
        )
        if status.get("operator_phase_b_authorization") != "MET":
            raise SystemExit(
                f"[blocknerv] --auth-eval refused: operator-phase-B-authorization gate "
                f"PENDING for {args.phase_b_auth_memo}. Status: {status}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _seed_everything(args.seed)
    device = _resolve_device(args.device)

    yuv6_token = None
    if args.enable_differentiable_yuv6 and args.enable_score_aware_loss:
        yuv6_token = _activate_differentiable_yuv6()
        print("[blocknerv] differentiable rgb_to_yuv6 monkey-patch active")

    from tac.blocknerv_as_renderer import (
        BlockNeRVConfig,
        BlockNeRVLatentTable,
        BlockNeRVRenderer,
        _make_synthetic_pair_batch_for_smoke,
        default_pose_surrogate,
        default_seg_surrogate,
        export_blocknerv_to_archive,
        train_step_blocknerv,
    )
    from tac.lane_12_v2_nerv_as_renderer import RealPairBatchSource
    from tac.training import EMA

    eval_h = args.tile_rows * args.tile_h
    eval_w = args.tile_cols * args.tile_w
    config = BlockNeRVConfig(
        latent_dim=args.latent_dim,
        tile_h=args.tile_h, tile_w=args.tile_w,
        tile_rows=args.tile_rows, tile_cols=args.tile_cols,
        coord_embed_dim=args.coord_embed_dim,
        base_channels=8 if args.smoke else args.base_channels,
        n_stages=3,
        n_pairs=4 if args.smoke else args.n_pairs,
        eval_size=(eval_h, eval_w),
        lambda_seg=args.lambda_seg,
        lambda_pose=args.lambda_pose,
        cuda_required=(args.device == "cuda" and not args.smoke),
    )

    renderer = BlockNeRVRenderer(config).to(device)
    latent_table = BlockNeRVLatentTable(config.n_pairs, config.latent_dim).to(device)
    n_params = sum(p.numel() for p in renderer.parameters()) + \
               sum(p.numel() for p in latent_table.parameters())
    print(f"[blocknerv] renderer={sum(p.numel() for p in renderer.parameters()):,} "
          f"latents={sum(p.numel() for p in latent_table.parameters()):,} "
          f"total={n_params:,} device={device}")

    ema_renderer = EMA(renderer, decay=args.ema_decay)
    ema_latents = EMA(latent_table, decay=args.ema_decay)
    optimizer = torch.optim.Adam(
        list(renderer.parameters()) + list(latent_table.parameters()),
        lr=args.learning_rate,
    )

    scorer_seg = scorer_pose = None
    if args.enable_score_aware_loss:
        from tac.scorer import load_differentiable_scorers
        scorer_pose, scorer_seg = load_differentiable_scorers(
            REPO_ROOT / "upstream", device=device,
        )
        scorer_pose.eval()
        scorer_seg.eval()
        for m in (scorer_pose, scorer_seg):
            for p in m.parameters():
                p.requires_grad_(False)

    use_synthetic = bool(args.smoke)
    if use_synthetic:
        print("[blocknerv] SMOKE MODE — # SYNTHETIC_NON_SMOKE_OK:blocknerv_smoke_via_argparse")
    else:
        if not args.video_path.exists():
            raise SystemExit(
                f"[blocknerv] non-smoke training requires {args.video_path}"
            )
        batch_source = RealPairBatchSource(
            video_path=args.video_path, n_pairs=config.n_pairs,
            eval_size=config.eval_size,
        )

    history: list[dict] = []
    epochs = 1 if args.smoke else args.epochs
    batch_size = 1 if args.smoke else args.batch_size

    for epoch in range(epochs):
        renderer.train()
        latent_table.train()
        epoch_loss = 0.0
        n_batches = 0

        if use_synthetic:
            pair_indices, gt_pairs_uint8 = _make_synthetic_pair_batch_for_smoke(
                batch_size=batch_size, latent_dim=config.latent_dim,
                eval_size=config.eval_size, n_pairs=config.n_pairs,
                seed=args.seed + epoch,
            )
            batches = [(pair_indices.to(device), gt_pairs_uint8.to(device))]
        else:
            batches = batch_source.iter_batches(
                batch_size=batch_size, max_pairs=args.max_pairs,
            )

        for pair_indices, gt_pairs_uint8 in batches:
            pair_indices = pair_indices.to(device)
            gt_pairs_uint8 = gt_pairs_uint8.to(device)

            if args.enable_score_aware_loss:
                result = train_step_blocknerv(
                    renderer=renderer, latent_table=latent_table,
                    pair_indices=pair_indices, gt_pairs_uint8=gt_pairs_uint8,
                    scorer_seg=scorer_seg, scorer_pose=scorer_pose,
                    seg_surrogate=default_seg_surrogate,
                    pose_surrogate=default_pose_surrogate,
                    lambda_seg=args.lambda_seg, lambda_pose=args.lambda_pose,
                    eval_roundtrip=True,
                )
            else:
                z = latent_table(pair_indices)
                decoded = renderer(z)
                target_native = torch.nn.functional.interpolate(
                    gt_pairs_uint8.float().reshape(-1, 3, *gt_pairs_uint8.shape[-2:]),
                    size=config.eval_size, mode="bicubic", align_corners=False,
                ).reshape(decoded.shape)
                loss = torch.nn.functional.l1_loss(decoded, target_native)
                result = {"loss": loss}

            if not torch.isfinite(result["loss"]).all():
                raise RuntimeError(f"[blocknerv] non-finite loss epoch={epoch} batch={n_batches}")

            optimizer.zero_grad()
            result["loss"].backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(renderer.parameters()) + list(latent_table.parameters()),
                    max_norm=args.grad_clip_norm,
                )
            optimizer.step()
            ema_renderer.update(renderer)
            ema_latents.update(latent_table)

            epoch_loss += float(result["loss"].detach())
            n_batches += 1

        avg = epoch_loss / max(n_batches, 1)
        if epoch == 0 or (epoch + 1) % args.eval_every_epochs == 0 or epoch == epochs - 1:
            history.append({"epoch": epoch + 1, "avg_loss": avg})
            print(f"[blocknerv] epoch {epoch+1}/{epochs} avg_loss={avg:.4f}")

    # ── EMA-shadow archive export ────────────────────────────────────
    orig_renderer = {k: v.detach().clone() for k, v in renderer.state_dict().items()}
    orig_latents = {k: v.detach().clone() for k, v in latent_table.state_dict().items()}
    try:
        ema_renderer.apply(renderer)
        ema_latents.apply(latent_table)
        archive_path = args.output_dir / "0.bin"
        archive_sha = export_blocknerv_to_archive(
            renderer=renderer, latent_table=latent_table, output_path=archive_path,
        )
    finally:
        renderer.load_state_dict(orig_renderer)
        latent_table.load_state_dict(orig_latents)
        renderer.train()
        latent_table.train()

    archive_bytes = archive_path.stat().st_size
    print(f"[blocknerv] archive {archive_path} sha256={archive_sha[:16]} bytes={archive_bytes}")

    auth_eval_result = None
    if args.auth_eval:
        print("[blocknerv] --auth-eval gated path: deferring to external dispatcher")
        auth_eval_result = {
            "status": "deferred_to_external_dispatcher",
            "note": "Operator runs experiments/contest_auth_eval.py with --archive-path + --phase-b-auth-memo",
        }

    provenance = {
        "schema": SCHEMA_VERSION,
        "lane_id": LANE_ID,
        "started_at_utc": started_at,
        "device": str(device),
        "smoke": bool(args.smoke),
        "epochs": int(epochs),
        "n_params": int(n_params),
        "config": {
            "latent_dim": config.latent_dim,
            "tile_rows": config.tile_rows, "tile_cols": config.tile_cols,
            "tile_h": config.tile_h, "tile_w": config.tile_w,
            "coord_embed_dim": config.coord_embed_dim,
            "base_channels": config.base_channels,
            "n_pairs": config.n_pairs,
            "eval_size": list(config.eval_size),
            "lambda_seg": config.lambda_seg, "lambda_pose": config.lambda_pose,
        },
        "archive_path": str(archive_path),
        "archive_sha256": archive_sha,
        "archive_bytes": int(archive_bytes),
        "ema_decay": float(args.ema_decay),
        "predicted_delta_score": PREDICTED_DELTA_SCORE,
        "score_claim": False, "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": "[predicted; BlockNeRV production trainer; no anchor]",
        "score_aware_loss_enabled": bool(args.enable_score_aware_loss),
        "differentiable_yuv6_enabled": bool(args.enable_differentiable_yuv6),
        "auth_eval_result": auth_eval_result, "history": history,
        "compliance_tags": [
            "ema_0p997_snapshot_restore", "eval_roundtrip_true",
            "no_mps_authoritative", "differentiable_yuv6",
            "score_aware_lagrangian", "no_synthetic_outside_smoke",
            "no_tmp_paths", "auth_eval_gated_phase_b_option_c",
            "cuda_required_default",
        ],
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(f"[blocknerv] done; provenance written to {args.output_dir}/provenance.json")

    if yuv6_token is not None:
        from tac.differentiable_eval_roundtrip import unpatch_upstream_yuv6
        try:
            unpatch_upstream_yuv6(yuv6_token)
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
