#!/usr/bin/env python3
"""MNeRV-as-renderer — production trainer.

Per operator directive 2026-05-11 + HNeRV parity discipline lesson 5 (full RGB
renderer). MNeRV is a 3-scale hierarchical NeRV variant (Mallat-scattering-
style bandpass cascade) that composes orthogonally with the numpy_inverse_dwt
inflate path. Architecture lives in :mod:`tac.mnerv_as_renderer`.

REPRESENTATION_ARCHIVE_GRAMMAR_BLUEPRINT (per CLAUDE.md Catalog #124)
---------------------------------------------------------------------
  archive_grammar: monolithic 0.bin (see ARCHIVE_GRAMMAR_MNERV in
    src/tac/mnerv_as_renderer.py) — 16-byte header + 4 length-prefixed
    sections covering decoder_blob (INT8+brotli, 3-scale cascade),
    scale_table (FP16), latent_blob (uint8 delta-zigzag+brotli), sidecar
  parser_section_manifest: ARCHIVE_GRAMMAR_MNERV declared at module level
  inflate_runtime_loc_budget: substrate_engineering — ≤200 LOC hermetic
  runtime_dep_closure: torch + brotli (zero dep on tac.*)
  export_format: mnerv_phase_a_monolithic_singlefile_0bin
  score_aware_loss: mnerv_train_step uses load_differentiable_scorers
    PoseNet + SegNet via Lagrangian λ_seg + λ_pose
  bolt_on_loc_budget: substrate_engineering (full multi-scale renderer)
  no_op_detector_planned: export_mnerv_to_archive returns sha256

CLAUDE.md non-negotiables wired through this trainer
----------------------------------------------------
  - eval_roundtrip=True (CLAUDE.md "eval_roundtrip — NON-NEGOTIABLE")
  - EMA decay=0.997 with snapshot+restore (CLAUDE.md)
  - differentiable rgb_to_yuv6 monkey-patch BEFORE scorer construction
  - score-domain Lagrangian (NOT weight-domain proxy)
  - real video data outside ``--smoke`` (no make_synthetic; Catalog #114)
  - auth eval gated behind ``--phase-b-auth-memo`` (Catalog #150)
  - CUDA-required default; NO MPS authoritative
  - No /tmp paths (CLAUDE.md FORBIDDEN_PATTERNS)
  - Co-Authored-By trailer auto-appended via subagent_commit_serializer

Cost when dispatched: ~$40-60 (Vast.ai T4 ~6-12h — multi-scale 3× work vs Lane 12-v2).
Operator-gated.

Predicted Δ score at PR106 r2 frontier:
  ``[predicted; Mallat scattering bandpass cascade; 3-scale hierarchical
  decomposition gives ~1.5× the parameter efficiency at fixed bytes vs
  single-scale NeRV per Mallat 2012 scattering theorem]``. NOT a score
  claim until [contest-CUDA] anchor on exact archive bytes.
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


LANE_ID = "lane_mnerv_as_renderer"
SCHEMA_VERSION = "1.0.0-mnerv-as-renderer-production"
PREDICTED_DELTA_SCORE = (
    "[predicted; Mallat scattering bandpass cascade; 3-scale hierarchical "
    "decomposition; ~1.5× param efficiency at fixed bytes vs single-scale]"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MNeRV (multi-scale NeRV) production trainer",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--ema-decay", type=float, default=0.997)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--base-channels", type=int, default=24,
                        help="Smaller than Lane 12-v2's 36 because 3 scales")
    parser.add_argument("--n-pairs", type=int, default=600)
    parser.add_argument("--lambda-seg", type=float, default=100.0)
    parser.add_argument("--lambda-pose", type=float, default=288.6751345948129)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--video-path", type=Path,
                        default=REPO_ROOT / "upstream" / "videos" / "0.mkv")
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--enable-differentiable-yuv6", action="store_true", default=True)
    parser.add_argument("--enable-score-aware-loss", action="store_true", default=True)
    parser.add_argument("--seed", type=int, default=20260511)
    parser.add_argument("--smoke", action="store_true", default=False)
    parser.add_argument("--auth-eval", action="store_true", default=False)
    parser.add_argument("--phase-b-auth-memo", type=str, default=None,
                        help="Repo-relative committed memo (Catalog #150)")
    parser.add_argument("--eval-every-epochs", type=int, default=25)
    return parser.parse_args(argv)


def _resolve_device(device_str: str) -> "torch.device":
    if device_str == "mps":
        raise SystemExit(
            "[mnerv] --device mps REFUSED per CLAUDE.md "
            "'MPS auth eval is NOISE' non-negotiable."
        )
    if device_str == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                "[mnerv] --device cuda requested but unavailable. "
                "CUDA-required default per CLAUDE.md MPS-fallback-trap "
                "FORBIDDEN_PATTERNS. Use --device cpu only for smoke."
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


def _refuse_auth_eval_without_authorization(args: argparse.Namespace) -> None:
    """Phase B Option C gate per Catalog #150."""
    if not args.auth_eval:
        return
    if args.phase_b_auth_memo is None:
        raise SystemExit(
            "[mnerv] --auth-eval refused: requires --phase-b-auth-memo "
            "<committed_repo_path> per Phase B Option C (Catalog #150)."
        )
    # Reuse Lane 12-v2's validator (Catalog #150 implementation):
    from tac.lane_12_v2_nerv_as_renderer import phase_b_preconditions_status
    status = phase_b_preconditions_status(
        consult_session_state=True,
        auth_memo_path=Path(args.phase_b_auth_memo),
    )
    if status.get("operator_phase_b_authorization") != "MET":
        raise SystemExit(
            f"[mnerv] --auth-eval refused: operator-Phase-B-authorization "
            f"PENDING for memo {args.phase_b_auth_memo}. Status: {status}"
        )


def _make_synthetic_mnerv_smoke_batch(
    *, batch_size: int, latent_dim: int, n_pairs: int, seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Smoke-only synthetic pair batch.

    # SYNTHETIC_NON_SMOKE_OK:mnerv_smoke_via_argparse_gated
    """
    g = torch.Generator().manual_seed(seed)
    pair_indices = torch.randint(0, n_pairs, (batch_size,), generator=g)
    H, W = 874, 1164
    gt_pairs = torch.randint(
        0, 256, (batch_size, 2, 3, H, W), generator=g, dtype=torch.uint8,
    )
    return pair_indices, gt_pairs


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _refuse_auth_eval_without_authorization(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _seed_everything(args.seed)
    device = _resolve_device(args.device)

    yuv6_token = None
    if args.enable_differentiable_yuv6 and args.enable_score_aware_loss:
        yuv6_token = _activate_differentiable_yuv6()
        print("[mnerv] differentiable rgb_to_yuv6 monkey-patch active")

    from tac.mnerv_as_renderer import (
        MNeRVConfig,
        MNeRVRenderer,
        MNeRVLatentTable,
        default_mnerv_pose_surrogate,
        default_mnerv_seg_surrogate,
        export_mnerv_to_archive,
        mnerv_train_step,
    )
    from tac.training import EMA

    config = MNeRVConfig(
        latent_dim=args.latent_dim,
        base_channels=8 if args.smoke else args.base_channels,
        n_pairs=4 if args.smoke else args.n_pairs,
        lambda_seg=args.lambda_seg,
        lambda_pose=args.lambda_pose,
        cuda_required=(args.device == "cuda" and not args.smoke),
    )

    renderer = MNeRVRenderer(config).to(device)
    latent_table = MNeRVLatentTable(config.n_pairs, config.latent_dim).to(device)
    n_params = sum(p.numel() for p in renderer.parameters()) + \
               sum(p.numel() for p in latent_table.parameters())
    print(f"[mnerv] renderer={sum(p.numel() for p in renderer.parameters()):,} "
          f"latents={sum(p.numel() for p in latent_table.parameters()):,} "
          f"total={n_params:,} scales={config.n_scales} device={device}")

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
        scorer_pose.eval(); scorer_seg.eval()
        for m in (scorer_pose, scorer_seg):
            for p in m.parameters():
                p.requires_grad_(False)

    use_synthetic = bool(args.smoke)
    if use_synthetic:
        print("[mnerv] SMOKE — synthetic pair batches "
              "(# SYNTHETIC_NON_SMOKE_OK:mnerv_smoke_via_argparse)")
    else:
        if not args.video_path.exists():
            raise SystemExit(
                f"[mnerv] non-smoke requires {args.video_path} (CLAUDE.md "
                "FORBIDDEN_PATTERNS: synthetic in non-smoke refused)"
            )
        # Reuse Lane 12-v2's RealPairBatchSource (it's substrate-agnostic).
        from tac.lane_12_v2_nerv_as_renderer import RealPairBatchSource
        batch_source = RealPairBatchSource(
            video_path=args.video_path,
            n_pairs=config.n_pairs,
            eval_size=config.eval_size,
        )

    history: list[dict] = []
    epochs = 1 if args.smoke else args.epochs
    batch_size = 1 if args.smoke else args.batch_size

    for epoch in range(epochs):
        renderer.train(); latent_table.train()
        epoch_loss = 0.0
        n_batches = 0

        if use_synthetic:
            pair_indices, gt_pairs_uint8 = _make_synthetic_mnerv_smoke_batch(
                batch_size=batch_size,
                latent_dim=config.latent_dim,
                n_pairs=config.n_pairs,
                seed=args.seed + epoch,
            )
            pair_indices = pair_indices.to(device)
            gt_pairs_uint8 = gt_pairs_uint8.to(device)
            batches = [(pair_indices, gt_pairs_uint8)]
        else:
            batches = batch_source.iter_batches(
                batch_size=batch_size, max_pairs=args.max_pairs,
            )

        for pair_indices, gt_pairs_uint8 in batches:
            pair_indices = pair_indices.to(device)
            gt_pairs_uint8 = gt_pairs_uint8.to(device)

            if args.enable_score_aware_loss:
                result = mnerv_train_step(
                    renderer=renderer,
                    latent_table=latent_table,
                    pair_indices=pair_indices,
                    gt_pairs_uint8=gt_pairs_uint8,
                    scorer_seg=scorer_seg,
                    scorer_pose=scorer_pose,
                    seg_surrogate=default_mnerv_seg_surrogate,
                    pose_surrogate=default_mnerv_pose_surrogate,
                    lambda_seg=args.lambda_seg,
                    lambda_pose=args.lambda_pose,
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
                raise RuntimeError(f"[mnerv] non-finite loss at epoch {epoch}")

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
            print(f"[mnerv] epoch {epoch+1}/{epochs} avg_loss={avg:.4f}")

    # Archive from EMA shadow (snapshot+restore).
    orig_r = {k: v.detach().clone() for k, v in renderer.state_dict().items()}
    orig_l = {k: v.detach().clone() for k, v in latent_table.state_dict().items()}
    try:
        ema_renderer.apply(renderer)
        ema_latents.apply(latent_table)
        archive_path = args.output_dir / "0.bin"
        archive_sha = export_mnerv_to_archive(
            renderer=renderer, latent_table=latent_table, output_path=archive_path,
        )
    finally:
        renderer.load_state_dict(orig_r)
        latent_table.load_state_dict(orig_l)
        renderer.train(); latent_table.train()

    archive_bytes = archive_path.stat().st_size
    print(f"[mnerv] archive {archive_path} sha={archive_sha[:16]} bytes={archive_bytes}")

    auth_eval_result = None
    if args.auth_eval:
        print("[mnerv] --auth-eval gated path: defer to external dispatcher")
        auth_eval_result = {
            "status": "deferred_to_external_dispatcher",
            "note": ("Operator runs experiments/contest_auth_eval.py with "
                     "--archive-path + --phase-b-auth-memo per CLAUDE.md "
                     "submission-auth-eval-BOTH-CPU-AND-CUDA."),
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
            "base_channels": config.base_channels,
            "n_pairs": config.n_pairs,
            "n_scales": config.n_scales,
            "eval_size": list(config.eval_size),
            "lambda_seg": config.lambda_seg,
            "lambda_pose": config.lambda_pose,
        },
        "archive_path": str(archive_path),
        "archive_sha256": archive_sha,
        "archive_bytes": int(archive_bytes),
        "ema_decay": float(args.ema_decay),
        "predicted_delta_score": PREDICTED_DELTA_SCORE,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": "[predicted; MNeRV production trainer; no anchor]",
        "score_aware_loss_enabled": bool(args.enable_score_aware_loss),
        "differentiable_yuv6_enabled": bool(args.enable_differentiable_yuv6),
        "auth_eval_result": auth_eval_result,
        "history": history,
        "compliance_tags": [
            "ema_0p997_snapshot_restore",
            "eval_roundtrip_true",
            "no_mps_authoritative",
            "differentiable_yuv6",
            "score_aware_lagrangian",
            "no_synthetic_outside_smoke",
            "no_tmp_paths",
            "auth_eval_gated_phase_b_option_c",
            "cuda_required_default",
            "multi_scale_3_level_mallat_scattering",
        ],
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(f"[mnerv] done; provenance → {args.output_dir}/provenance.json")

    if yuv6_token is not None:
        from tac.differentiable_eval_roundtrip import unpatch_upstream_yuv6
        try:
            unpatch_upstream_yuv6(yuv6_token)
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
