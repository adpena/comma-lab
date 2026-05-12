#!/usr/bin/env python3
"""VQ-VAE-as-full-renderer — production trainer.

Per operator directive 2026-05-11 + HNeRV parity discipline lesson 5 + van
den Oord 2017 council position. THIS trainer makes VQ-VAE the FULL renderer
(codebook indices → renderer → RGB), distinct from T17's BOLT-ON shared
codebook on T1's substrate.

Architecture (substrate in :mod:`tac.vqvae_as_full_renderer`):
  Encoder (latent → tokens_per_pair tokens) → Codebook (van den Oord
  persistent-EMA, decay 0.99) → Decoder (PixelShuffle to camera res).

REPRESENTATION_ARCHIVE_GRAMMAR_BLUEPRINT (per CLAUDE.md Catalog #124)
---------------------------------------------------------------------
  archive_grammar: monolithic 0.bin (ARCHIVE_GRAMMAR_VQVAE_FULL) — 16-byte
    header + 5 length-prefixed sections (codebook_blob FP16 raw,
    decoder_blob INT8+brotli, scale_table FP16, indices_blob brotli uint16,
    sidecar empty)
  parser_section_manifest: ARCHIVE_GRAMMAR_VQVAE_FULL declared in
    src/tac/vqvae_as_full_renderer.py
  inflate_runtime_loc_budget: substrate_engineering — ≤200 LOC hermetic
  runtime_dep_closure: torch + brotli (zero dep on tac.*)
  export_format: vqvae_full_renderer_phase_a_monolithic_singlefile_0bin
  score_aware_loss: vqvae_train_step uses load_differentiable_scorers with
    Lagrangian λ_seg + λ_pose + van den Oord commitment loss β=0.25
  bolt_on_loc_budget: substrate_engineering (full codebook+decoder renderer)
  no_op_detector_planned: export_vqvae_to_archive returns sha256

CLAUDE.md non-negotiables wired through this trainer
----------------------------------------------------
  - eval_roundtrip=True (CLAUDE.md "eval_roundtrip — NON-NEGOTIABLE")
  - Weight EMA decay=0.997 with snapshot+restore (CLAUDE.md)
  - Codebook EMA decay=0.99 (van den Oord §3.2 — CLAUDE.md EMA exception
    clause for VQ-VAE codebooks)
  - NN-2 perplexity gate ≥ 0.4·N per epoch (CLAUDE.md Phase 2 pre-design)
  - differentiable rgb_to_yuv6 monkey-patch BEFORE scorer construction
  - score-domain Lagrangian + van den Oord commitment loss
  - real video data outside ``--smoke`` (no make_synthetic; Catalog #114)
  - auth eval gated behind ``--phase-b-auth-memo`` (Catalog #150)
  - CUDA-required default; NO MPS authoritative
  - No /tmp paths (CLAUDE.md FORBIDDEN_PATTERNS)

Cost when dispatched: ~$30-60 (Vast.ai T4 ~6-12h). Operator-gated.

Predicted Δ score at PR106 r2 frontier:
  ``[predicted; van den Oord BD-rate; 256-entry × 64-dim FP16 codebook
  (~32 KB) replacing ~80 KB of per-pair latents enables ~50 KB total
  budget reduction at fixed distortion per VQ-VAE rate-distortion theory]``.
  NOT a score claim until [contest-CUDA] anchor.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

from tac.output_path_policy import assert_not_temporary_output_dir

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


LANE_ID = "lane_vqvae_as_full_renderer"
SCHEMA_VERSION = "1.0.0-vqvae-as-full-renderer-production"
PREDICTED_DELTA_SCORE = (
    "[predicted; van den Oord BD-rate; 256-entry × 64-dim FP16 codebook "
    "(~32 KB) replacing ~80 KB of per-pair latents]"
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VQ-VAE-as-full-renderer production trainer",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--ema-decay", type=float, default=0.997,
                        help="Weight EMA decay (CLAUDE.md canonical 0.997)")
    parser.add_argument("--codebook-ema-decay", type=float, default=0.99,
                        help="Codebook EMA (van den Oord §3.2; CLAUDE.md exception clause)")
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--num-entries", type=int, default=256,
                        help="Codebook size (van den Oord §4.2 canonical 256)")
    parser.add_argument("--entry-dim", type=int, default=64,
                        help="Codebook entry dim (Quantizr-class C=64)")
    parser.add_argument("--tokens-per-pair", type=int, default=8)
    parser.add_argument("--n-pairs", type=int, default=600)
    parser.add_argument("--lambda-seg", type=float, default=100.0)
    parser.add_argument("--lambda-pose", type=float, default=288.6751345948129)
    parser.add_argument("--commitment-weight", type=float, default=0.25,
                        help="van den Oord commitment loss β (canonical 0.25)")
    parser.add_argument("--perplexity-floor-ratio", type=float, default=0.4,
                        help="NN-2 floor / num_entries (CLAUDE.md Phase 2 pre-design)")
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
            "[vqvae] --device mps REFUSED per CLAUDE.md "
            "'MPS auth eval is NOISE' non-negotiable."
        )
    if device_str == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                "[vqvae] --device cuda requested but unavailable. "
                "CUDA-required default per CLAUDE.md MPS-fallback-trap."
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
            "[vqvae] --auth-eval refused: requires --phase-b-auth-memo "
            "<committed_repo_path> per Phase B Option C (Catalog #150)."
        )
    from tac.lane_12_v2_nerv_as_renderer import phase_b_preconditions_status
    status = phase_b_preconditions_status(
        consult_session_state=True,
        auth_memo_path=Path(args.phase_b_auth_memo),
    )
    if status.get("operator_phase_b_authorization") != "MET":
        raise SystemExit(
            f"[vqvae] --auth-eval refused: PENDING for memo "
            f"{args.phase_b_auth_memo}. Status: {status}"
        )


def _make_synthetic_vqvae_smoke_batch(
    *, batch_size: int, latent_dim: int, n_pairs: int, seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Smoke-only synthetic pair batch.

    # SYNTHETIC_NON_SMOKE_OK:vqvae_smoke_via_argparse_gated
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
    try:
        assert_not_temporary_output_dir(args.output_dir, tool_name="vqvae")
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _seed_everything(args.seed)
    device = _resolve_device(args.device)

    yuv6_token = None
    if args.enable_differentiable_yuv6 and args.enable_score_aware_loss:
        yuv6_token = _activate_differentiable_yuv6()
        print("[vqvae] differentiable rgb_to_yuv6 monkey-patch active")

    from tac.vqvae_as_full_renderer import (
        VQVAEFullConfig,
        VQVAEFullRenderer,
        VQVAEFullLatentTable,
        assert_codebook_perplexity_ok,
        default_vqvae_pose_surrogate,
        default_vqvae_seg_surrogate,
        export_vqvae_to_archive,
        vqvae_train_step,
        VQVAECodebookCollapseError,
    )
    from tac.training import EMA

    config = VQVAEFullConfig(
        latent_dim=args.latent_dim,
        num_entries=args.num_entries,
        entry_dim=args.entry_dim,
        tokens_per_pair=args.tokens_per_pair,
        n_pairs=4 if args.smoke else args.n_pairs,
        codebook_ema_decay=args.codebook_ema_decay,
        commitment_weight=args.commitment_weight,
        nn2_perplexity_floor_ratio=args.perplexity_floor_ratio,
        lambda_seg=args.lambda_seg,
        lambda_pose=args.lambda_pose,
        cuda_required=(args.device == "cuda" and not args.smoke),
        # Smaller base_channels in smoke for speed.
        base_channels=8 if args.smoke else 36,
    )

    renderer = VQVAEFullRenderer(config).to(device)
    latent_table = VQVAEFullLatentTable(config.n_pairs, config.latent_dim).to(device)
    n_decoder = sum(p.numel() for p in renderer.decoder.parameters())
    n_encoder = sum(p.numel() for p in renderer.encoder.parameters())
    n_latents = sum(p.numel() for p in latent_table.parameters())
    codebook_bytes = config.num_entries * config.entry_dim * 2  # FP16
    print(f"[vqvae] encoder={n_encoder:,} decoder={n_decoder:,} "
          f"latents={n_latents:,} codebook={config.num_entries}×{config.entry_dim} "
          f"(~{codebook_bytes}B) device={device}")

    # CLAUDE.md weight EMA 0.997 wraps decoder + encoder + latents (NOT codebook;
    # codebook has its own persistent EMA at decay 0.99 — van den Oord §3.2).
    ema_renderer = EMA(renderer, decay=args.ema_decay)
    ema_latents = EMA(latent_table, decay=args.ema_decay)

    # Optimizer trains gradient-bearing params: encoder + decoder + latents.
    # Codebook entries are BUFFERS, updated via explicit update_ema.
    grad_params = (
        list(renderer.encoder.parameters())
        + list(renderer.decoder.parameters())
        + list(latent_table.parameters())
    )
    optimizer = torch.optim.Adam(grad_params, lr=args.learning_rate)

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
        print("[vqvae] SMOKE — synthetic pair batches "
              "(# SYNTHETIC_NON_SMOKE_OK:vqvae_smoke_via_argparse)")
    else:
        if not args.video_path.exists():
            raise SystemExit(
                f"[vqvae] non-smoke requires {args.video_path} (CLAUDE.md "
                "FORBIDDEN_PATTERNS: synthetic in non-smoke refused)"
            )
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
        epoch_indices: list[torch.Tensor] = []

        if use_synthetic:
            pair_indices, gt_pairs_uint8 = _make_synthetic_vqvae_smoke_batch(
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
                result = vqvae_train_step(
                    renderer=renderer,
                    latent_table=latent_table,
                    pair_indices=pair_indices,
                    gt_pairs_uint8=gt_pairs_uint8,
                    scorer_seg=scorer_seg,
                    scorer_pose=scorer_pose,
                    seg_surrogate=default_vqvae_seg_surrogate,
                    pose_surrogate=default_vqvae_pose_surrogate,
                    lambda_seg=args.lambda_seg,
                    lambda_pose=args.lambda_pose,
                    commitment_weight=args.commitment_weight,
                    eval_roundtrip=True,
                )
            else:
                # Smoke / research-only: pixel-L1 + commitment.
                z = latent_table(pair_indices)
                decoded, indices, commitment_loss = renderer(z)
                target_native = torch.nn.functional.interpolate(
                    gt_pairs_uint8.float().reshape(-1, 3, *gt_pairs_uint8.shape[-2:]),
                    size=config.eval_size, mode="bicubic", align_corners=False,
                ).reshape(decoded.shape)
                loss = (
                    torch.nn.functional.l1_loss(decoded, target_native)
                    + args.commitment_weight * commitment_loss
                )
                result = {"loss": loss, "indices": indices.detach(),
                          "commitment_loss": commitment_loss}

            if not torch.isfinite(result["loss"]).all():
                raise RuntimeError(f"[vqvae] non-finite loss at epoch {epoch}")

            optimizer.zero_grad()
            result["loss"].backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(grad_params, max_norm=args.grad_clip_norm)
            optimizer.step()

            # Persistent EMA codebook update (van den Oord §3.2) AFTER weight step.
            with torch.no_grad():
                z_e_for_ema = renderer.encoder(latent_table(pair_indices))
                _, idx_for_ema, _ = renderer.codebook(z_e_for_ema)
                renderer.codebook.update_ema(z_e_for_ema, idx_for_ema)

            # Weight EMA (decoder + encoder + latent_table; codebook excluded via shadow keys).
            ema_renderer.update(renderer)
            ema_latents.update(latent_table)

            epoch_indices.append(result["indices"].detach().reshape(-1).cpu())
            epoch_loss += float(result["loss"].detach())
            n_batches += 1

        avg = epoch_loss / max(n_batches, 1)
        all_indices = torch.cat(epoch_indices) if epoch_indices else torch.tensor([0])

        # NN-2 perplexity gate (CLAUDE.md Phase 2 pre-design); raises on collapse.
        # SMOKE mode tolerates collapse because synthetic noise → degenerate indices;
        # the gate runs but the assertion is non-fatal in smoke for development speed.
        try:
            perp_diag = assert_codebook_perplexity_ok(
                all_indices,
                num_entries=config.num_entries,
                floor_ratio=args.perplexity_floor_ratio,
            )
        except VQVAECodebookCollapseError as exc:
            if args.smoke:
                perp_diag = {
                    "perplexity": -1.0, "floor": -1.0,
                    "floor_ratio": args.perplexity_floor_ratio,
                    "num_entries": config.num_entries,
                    "passed": False,
                    "smoke_tolerated": True,
                }
                print(f"[vqvae] smoke NN-2 collapse tolerated: {exc}")
            else:
                raise

        if epoch == 0 or (epoch + 1) % args.eval_every_epochs == 0 or epoch == epochs - 1:
            history.append({
                "epoch": epoch + 1,
                "avg_loss": avg,
                "perplexity": perp_diag["perplexity"],
                "perplexity_floor": perp_diag["floor"],
            })
            print(f"[vqvae] epoch {epoch+1}/{epochs} loss={avg:.4f} "
                  f"perp={perp_diag['perplexity']:.1f}/{perp_diag['floor']:.1f}")

    # Archive from EMA shadow.
    orig_r = {k: v.detach().clone() for k, v in renderer.state_dict().items()}
    orig_l = {k: v.detach().clone() for k, v in latent_table.state_dict().items()}
    try:
        ema_renderer.apply(renderer)
        ema_latents.apply(latent_table)
        archive_path = args.output_dir / "0.bin"
        archive_sha = export_vqvae_to_archive(
            renderer=renderer, latent_table=latent_table, output_path=archive_path,
        )
    finally:
        renderer.load_state_dict(orig_r)
        latent_table.load_state_dict(orig_l)
        renderer.train(); latent_table.train()

    archive_bytes = archive_path.stat().st_size
    print(f"[vqvae] archive {archive_path} sha={archive_sha[:16]} bytes={archive_bytes}")

    auth_eval_result = None
    if args.auth_eval:
        print("[vqvae] --auth-eval gated path: defer to external dispatcher")
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
        "n_decoder": int(n_decoder),
        "n_encoder": int(n_encoder),
        "n_latents": int(n_latents),
        "codebook_bytes": int(codebook_bytes),
        "config": {
            "latent_dim": config.latent_dim,
            "num_entries": config.num_entries,
            "entry_dim": config.entry_dim,
            "tokens_per_pair": config.tokens_per_pair,
            "n_pairs": config.n_pairs,
            "codebook_ema_decay": config.codebook_ema_decay,
            "commitment_weight": config.commitment_weight,
            "nn2_perplexity_floor_ratio": config.nn2_perplexity_floor_ratio,
            "eval_size": list(config.eval_size),
            "lambda_seg": config.lambda_seg,
            "lambda_pose": config.lambda_pose,
        },
        "archive_path": str(archive_path),
        "archive_sha256": archive_sha,
        "archive_bytes": int(archive_bytes),
        "ema_decay_weights": float(args.ema_decay),
        "ema_decay_codebook": float(args.codebook_ema_decay),
        "predicted_delta_score": PREDICTED_DELTA_SCORE,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": "[predicted; VQ-VAE full-renderer trainer; no anchor]",
        "score_aware_loss_enabled": bool(args.enable_score_aware_loss),
        "differentiable_yuv6_enabled": bool(args.enable_differentiable_yuv6),
        "auth_eval_result": auth_eval_result,
        "history": history,
        "compliance_tags": [
            "ema_0p997_weights_snapshot_restore",
            "codebook_ema_0p99_van_den_oord_canon",
            "eval_roundtrip_true",
            "no_mps_authoritative",
            "differentiable_yuv6",
            "score_aware_lagrangian",
            "vandenoord_commitment_loss",
            "nn2_perplexity_gate_per_epoch",
            "no_synthetic_outside_smoke",
            "no_tmp_paths",
            "auth_eval_gated_phase_b_option_c",
            "cuda_required_default",
            "vqvae_as_full_renderer_not_bolt_on",
        ],
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    print(f"[vqvae] done; provenance → {args.output_dir}/provenance.json")

    if yuv6_token is not None:
        from tac.differentiable_eval_roundtrip import unpatch_upstream_yuv6
        try:
            unpatch_upstream_yuv6(yuv6_token)
        except Exception:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
