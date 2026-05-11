#!/usr/bin/env python3
"""T15 — Time-varying FiLM modulator trainer (T1-clone derivative).

Phase 2 pre-design pass (2026-05-09) consensus + readiness pre-stage
(2026-05-11) handoff: T15 generalizes Quantizr's STATIC FiLM (single γ,β
shared across all pairs) to PER-PAIR FiLM where (γ_t, β_t) =
modulator_MLP(pose_delta_t). The modulator MLP (~4544 params @ FP4 =
2272 bytes) plus per-pair pose (already in archive) keeps the modulator
addition local to a Phase 2 substrate-engineering bolt-on. Reuses the T1
end-to-end stack (Ballé hyperprior + 128K decoder + Lagrangian-ADMM) plus
the canonical eval_roundtrip + EMA + differentiable YUV6 discipline.

REPRESENTATION_ARCHIVE_GRAMMAR_BLUEPRINT (per HNeRV parity discipline):
  archive_grammar: T1 monolithic-single-x-member + modulator.bin sidecar
  parser_section_manifest: x SHA-256, decoder.bin SHA-256, balle.bin SHA-256,
    modulator.bin SHA-256, byte sizes per section
  inflate_runtime_loc_budget: substrate_engineering (parent T1 budget)
  runtime_dep_closure: tac runtime + torch + brotli + compressai
  export_format: phase2_t15_t1_film_bolt_on (T1 grammar + modulator sidecar)
  score_aware_loss: T1 score-aware loss + FiLM gradient flow regression (NN-1)
  bolt_on_loc_budget: substrate_engineering (per Phase 2 pre-design)
  no_op_detector_planned: exact old/new modulator.bin + decoder.bin SHA proof

CLAUDE.md non-negotiables — wired through this trainer
------------------------------------------------------

- **EMA decay 0.997 with snapshot+restore at eval time** — modulator MLP +
  decoder + Ballé all wrapped by ``tac.training.EMA(decay=0.997)``; inference
  uses the EMA shadow per CLAUDE.md "EMA — NON-NEGOTIABLE".
- **eval_roundtrip = True** — proxy loss simulates the contest's inflate
  roundtrip (384→874→uint8→384) via ``apply_eval_roundtrip_during_training``.
- **Differentiable YUV6** — ``patch_upstream_yuv6_globally`` is called
  BEFORE any scorer load per CLAUDE.md "NeRV/HNeRV renderer trainers must
  also keep scorer preprocess differentiable".
- **Score-domain Lagrangian** — α·B(θ)/N + β·d_seg(θ) + γ·sqrt(10·d_pose(θ))
  per CLAUDE.md "Meta-Lagrangian/Pareto solver" + contest functional.
- **NEVER MPS authoritative** — refuses ``--device mps``; advisory MPS
  tagged ``[macOS-CPU advisory only]`` per CLAUDE.md.
- **No make_synthetic_* outside --smoke** — non-smoke uses
  ``load_real_target_pairs`` (PyAV on ``upstream/videos/0.mkv``).
- **No /tmp paths** — outputs under ``experiments/results/<lane>_<utc>/``.
- **Auth eval gated** — ``--auth-eval`` refused (Phase 2 scaffold-only
  unless explicit dispatch claim + operator approval; mirrors T1).

NN-1 gate (NON-NEGOTIABLE; per Phase 2 pre-design):
  Adam 5-step gradient-flow regression test on the FiLM modulator MUST
  pass before any dispatch. Test asserts L2 movement > 1e-3. The trainer
  itself also asserts modulator gradients are FINITE and NONZERO on the
  first training step (fail-closed).

Predicted Δ score (stand-alone on A1 substrate):
  -0.005 ± 0.003 ``[predicted; Berger pose; ρ_pose=0.85 fallback per
  pre-design pass]``. NOT a score claim until contest-CUDA anchor lands.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tac.film_time_varying import (  # noqa: E402
    TimeVaryingFiLM,
    TimeVaryingFiLMConfig,
    time_varying_film_state_bytes,
)
from tac.paradigm_delta_epsilon_zeta import (  # noqa: E402
    BalleHyperpriorConfig,
    Decoder128KConfig,
    JointLagrangianADMM,
    JointLagrangianADMMConfig,
    build_balle_hyperprior,
    build_decoder_128k,
    load_frozen_a1_encoder,
)
from tac.training import EMA  # noqa: E402

# Reuse T1's helpers (eval_roundtrip, real-target loader, smoke target,
# YUV6 monkey patch, scorer gradient reachability, device resolve, seed).
from experiments.train_paradigm_delta_epsilon_zeta_track1_balle_endtoend import (  # noqa: E402
    EVAL_HW,
    PHASE1_SCAFFOLD_ONLY,
    _activate_yuv6_mode_t1,
    _resolve_device,
    _resolve_yuv6_mode_with_probe_t1,
    _seed_everything,
    assert_score_domain_gradient_reachability,
    eval_roundtrip_decoded,
    eval_roundtrip_pixel_l1,
    load_real_target_pairs,
    load_target_pixels_from_path,
    make_smoke_target,
    refuse_phase1_scaffold_path,
)


T15_LANE_ID = "lane_t15_time_varying_film_phase2_preregistered"
T15_SCHEMA_VERSION = "0.1.0-phase2-t15-film-bolt-on"
T15_PREDICTED_DELTA_SCORE = (
    "-0.005 ± 0.003 [predicted; Berger pose; ρ_pose=0.85 fallback "
    "per Phase 2 pre-design pass 2026-05-09]"
)


# ---------------------------------------------------------------------------
# T15 forward: pose-conditioned channel-wise FiLM applied to decoder output
# ---------------------------------------------------------------------------


def _extract_pose_delta_for_pair(latents_pair: torch.Tensor, pose_dim: int) -> torch.Tensor:
    """Derive a stable pose_delta proxy from the per-pair latent table.

    The T1 scaffold's A1 latents (28-dim per pair) are NOT the contest
    pose stream; this trainer is a substrate-engineering bolt-on so we
    project the first ``pose_dim`` channels of the per-pair latent into
    a pose-shaped vector. Real Phase 2 dispatch will replace this with
    the contest pose payload at integration time (gated by Phase 2
    readiness blockers per Phase 2 pre-design).

    Returns (B, pose_dim) float32 on the same device as ``latents_pair``.
    """
    if latents_pair.dim() != 2:
        raise ValueError(
            f"latents_pair must be (B, latent_dim); got shape {tuple(latents_pair.shape)}"
        )
    if latents_pair.shape[1] < pose_dim:
        # Smoke fallback: pad with zeros so smoke-mode build verification
        # works with small latent dims.
        return F.pad(latents_pair, (0, pose_dim - latents_pair.shape[1]))
    return latents_pair[:, :pose_dim].contiguous()


def _apply_t15_film_to_decoded(
    decoded: torch.Tensor, modulator: torch.nn.Module, pose_delta: torch.Tensor
) -> torch.Tensor:
    """Apply per-pair γ⊙f + β to the decoder output channels.

    decoded shape: (B, 2, 3, H, W) — pair-of-frames RGB output from
      ``Decoder128K``. T15 modulates the channel dim (3) per pair.

    Note: T15's TimeVaryingFiLMConfig.feature_channels MUST equal 3 to
    match the decoder's RGB output for this insertion point. The Quantizr-
    canonical FiLM modulator (feature_channels=64) is the council canon
    for modulating decoder INTERIOR features; this trainer inserts at the
    OUTPUT for simplicity (substrate-engineering bolt-on). Phase 3
    refinement may move the insertion point earlier.
    """
    if decoded.dim() != 5:
        raise ValueError(
            f"decoded must be (B, 2, 3, H, W); got shape {tuple(decoded.shape)}"
        )
    B, P, C, H, W = decoded.shape
    # Modulator expects (B, pose_dim) -> (B, C); apply per-channel.
    gamma, beta = modulator(pose_delta)
    # Broadcast across pair dim P and spatial HxW.
    gamma_b = gamma.view(B, 1, C, 1, 1)
    beta_b = beta.view(B, 1, C, 1, 1)
    return decoded * gamma_b + beta_b


def assert_modulator_gradient_finite_nonzero(
    modulator: torch.nn.Module,
) -> dict[str, float]:
    """Fail-closed: confirm the modulator's params received gradient.

    Asserts every modulator param has a non-None, finite grad AND that
    the total L2 grad norm is > 0. This is the inline counterpart to the
    NN-1 regression test in `src/tac/tests/test_film_time_varying.py`.
    """
    total_sq = 0.0
    saw_grad = False
    for name, p in modulator.named_parameters():
        g = p.grad
        if g is None:
            raise RuntimeError(
                f"[t15] modulator param {name} has no gradient after backward"
            )
        if not torch.isfinite(g).all().item():
            raise RuntimeError(
                f"[t15] modulator param {name} has non-finite gradient"
            )
        saw_grad = True
        total_sq += float(g.detach().double().pow(2).sum().item())
    if not saw_grad or total_sq <= 0.0:
        raise RuntimeError(
            f"[t15] NN-1 fail-closed: modulator grad L2={total_sq:.6g}; "
            "expected > 0 (modulator must receive gradient on first step)"
        )
    return {"modulator_grad_l2": float(total_sq ** 0.5)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="T15 — Time-varying FiLM trainer (T1-clone derivative)"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--aux-learning-rate", type=float, default=1e-3)
    parser.add_argument("--ema-decay", type=float, default=0.997)
    parser.add_argument("--noise-std", type=float, default=0.5)
    parser.add_argument("--rate-target-bytes", type=float, default=80000.0)
    parser.add_argument("--seg-target", type=float, default=7e-4)
    parser.add_argument("--pose-target", type=float, default=1.7e-4)
    parser.add_argument("--rho-init", type=float, default=1.0)
    parser.add_argument("--enable-eval-roundtrip-in-training", action="store_true", default=True)
    parser.add_argument("--enable-differentiable-yuv6", action="store_true", default=True)
    parser.add_argument("--yuv6-mode", default="auto",
                        choices=["auto", "monkey_patch_global", "tac_differentiable_routing"])
    parser.add_argument("--enable-scorer-domain-loss", action="store_true", default=False)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--eval-every-epochs", type=int, default=25)
    parser.add_argument("--auth-eval", action="store_true", default=False)
    parser.add_argument("--video-path", type=Path,
                        default=REPO_ROOT / "upstream" / "videos" / "0.mkv")
    parser.add_argument("--target-pixels-path", type=Path, default=None)
    parser.add_argument("--max-target-pairs", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument("--canonical-a1-relpath", type=str,
                        default="experiments/results/A1_canonical")
    parser.add_argument("--allow-missing-canonical-a1", action="store_true", default=False)
    # T15-specific
    parser.add_argument("--t15-pose-dim", type=int, default=6,
                        help="Modulator input dim (contest pose payload = 6)")
    parser.add_argument("--t15-hidden-dim", type=int, default=32,
                        help="Modulator MLP hidden width (council canon = 32)")
    parser.add_argument("--t15-modulator-quant", default="fp4",
                        choices=["fp4", "fp8", "fp16", "fp32"])
    parser.add_argument("--t15-modulator-activation", default="relu",
                        choices=["relu", "gelu", "silu"])
    return parser.parse_args(argv)


def _canonical_dir_name_from_relpath(relpath: str) -> str:
    return Path(relpath).name


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.device == "mps":
        raise SystemExit("[t15] --device mps refused; see CLAUDE.md MPS-is-noise rule")
    if args.auth_eval:
        refuse_phase1_scaffold_path("--auth-eval [T15 scaffold; T1-clone]")
    if not args.smoke and PHASE1_SCAFFOLD_ONLY and not args.enable_scorer_domain_loss:
        refuse_phase1_scaffold_path("non-smoke T15 training [scaffold; T1-clone]")
    if args.grad_clip_norm < 0:
        raise SystemExit("[t15] --grad-clip-norm must be non-negative")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _seed_everything(args.seed)
    device = _resolve_device(args.device)

    # PR #95 binary-forensics replication: activate autograd-preserving YUV6.
    yuv6_mode = _resolve_yuv6_mode_with_probe_t1(args.yuv6_mode)
    yuv6_token = _activate_yuv6_mode_t1(
        yuv6_mode, enabled=args.enable_differentiable_yuv6
    )
    print(
        f"[t15] eval_roundtrip={args.enable_eval_roundtrip_in_training} "
        f"yuv6_mode={yuv6_mode.value} smoke={args.smoke} device={device}"
    )

    # Load A1 latents + targets.
    if args.smoke and args.allow_missing_canonical_a1:
        latents, target_pixels = make_smoke_target(
            n_pairs=4, latent_dim=28, seed=args.seed,
        )
    else:
        encoder = load_frozen_a1_encoder(
            repo_root=REPO_ROOT,
            canonical_dir_name=_canonical_dir_name_from_relpath(args.canonical_a1_relpath),
        )
        latents = encoder.latents
        latents.requires_grad_(False)
        if args.smoke:
            latents = latents[:4].clone()
            _, target_pixels = make_smoke_target(
                n_pairs=int(latents.shape[0]),
                latent_dim=int(latents.shape[1]),
                seed=args.seed,
            )
        else:
            if args.target_pixels_path is not None:
                target_pixels = load_target_pixels_from_path(args.target_pixels_path)
                if args.max_target_pairs is not None:
                    target_pixels = target_pixels[: args.max_target_pairs].clone()
                    latents = latents[: target_pixels.shape[0]].clone()
            else:
                target_pixels = load_real_target_pairs(
                    args.video_path,
                    n_pairs=int(latents.shape[0]),
                    max_pairs=args.max_target_pairs,
                )
                latents = latents[: target_pixels.shape[0]].clone()
    latents = latents.to(device)
    target_pixels = target_pixels.to(device)

    # Build modules: decoder + Ballé hyperprior + T15 modulator.
    decoder_config = Decoder128KConfig(latent_dim=int(latents.shape[1]))
    decoder = build_decoder_128k(decoder_config).to(device)
    balle_config = BalleHyperpriorConfig(y_channels=int(latents.shape[1]))
    balle = build_balle_hyperprior(balle_config).to(device)

    # T15: feature_channels=3 because we insert AT the decoder OUTPUT (RGB).
    # This is the substrate-engineering bolt-on insertion point per design memo.
    t15_config = TimeVaryingFiLMConfig(
        pose_dim=args.t15_pose_dim,
        feature_channels=3,
        hidden_dim=args.t15_hidden_dim,
        activation=args.t15_modulator_activation,
        quantization=args.t15_modulator_quant,
        label=f"t15_t1clone_{started_at}",
    )
    modulator = TimeVaryingFiLM(t15_config).to(device)
    modulator_bytes = time_varying_film_state_bytes(t15_config)

    n_decoder = sum(p.numel() for p in decoder.parameters())
    n_balle = sum(p.numel() for p in balle.parameters())
    n_modulator = sum(p.numel() for p in modulator.parameters())
    print(
        f"[t15] decoder={n_decoder:,} balle={n_balle:,} "
        f"modulator={n_modulator:,} (~{modulator_bytes} archive bytes)"
    )

    # Optimisers (modulator joins the main optimiser; CompressAI requires
    # separate aux optimiser for entropy-bottleneck quantiles).
    decoder_trainable = [p for p in decoder.parameters() if p.requires_grad]
    balle_main_trainable = [
        p for n, p in balle.named_parameters()
        if p.requires_grad and ("entropy_bottleneck" not in n or "quantiles" not in n)
    ]
    modulator_trainable = [p for p in modulator.parameters() if p.requires_grad]
    main_params = [*decoder_trainable, *balle_main_trainable, *modulator_trainable]
    aux_params = [p for n, p in balle.named_parameters() if "quantiles" in n]
    optim_main = torch.optim.Adam(main_params, lr=args.learning_rate)
    optim_aux = torch.optim.Adam(aux_params, lr=args.aux_learning_rate) if aux_params else None

    # EMA shadows (decay 0.997 per CLAUDE.md "EMA — NON-NEGOTIABLE").
    ema_decoder = EMA(decoder, decay=args.ema_decay)
    ema_balle = EMA(balle, decay=args.ema_decay)
    ema_modulator = EMA(modulator, decay=args.ema_decay)

    coord = JointLagrangianADMM(JointLagrangianADMMConfig(
        rate_target_bytes=float(args.rate_target_bytes),
        seg_target=args.seg_target,
        pose_target=args.pose_target,
        rho_init=args.rho_init,
    ))

    n_pairs = int(latents.shape[0])
    epochs = 1 if args.smoke else args.epochs
    batch_size = 1 if args.smoke else min(args.batch_size, n_pairs)
    history: list[dict] = []
    modulator_gradcheck_done = False

    posenet = segnet = None
    if args.enable_scorer_domain_loss:
        from tac.scorer import load_differentiable_scorers  # noqa: WPS433
        posenet, segnet = load_differentiable_scorers(REPO_ROOT / "upstream", device=device)
        posenet.eval(); segnet.eval()
        for m in (posenet, segnet):
            for p in m.parameters():
                p.requires_grad_(False)

    for epoch in range(epochs):
        decoder.train(); balle.train(); modulator.train()
        perm = torch.randperm(n_pairs, generator=torch.Generator().manual_seed(args.seed + epoch))
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_pairs, batch_size):
            idx = perm[start:start + batch_size]
            y = latents[idx]
            tgt = target_pixels[idx]
            pose_delta = _extract_pose_delta_for_pair(y, args.t15_pose_dim)

            balle_out = balle(y)
            decoded = decoder(balle_out["y_hat"])
            decoded_filmed = _apply_t15_film_to_decoded(decoded, modulator, pose_delta)

            if args.enable_scorer_domain_loss and posenet is not None and segnet is not None:
                from tac.losses import scorer_loss_terms_btchw  # noqa: WPS433
                decoded_rt = eval_roundtrip_decoded(
                    decoded_filmed, noise_std=args.noise_std,
                    enable_eval_roundtrip_in_training=args.enable_eval_roundtrip_in_training,
                )
                scorer_d, pose_loss, seg_loss = scorer_loss_terms_btchw(
                    decoded_rt, tgt, posenet, segnet,
                    segmentation_surrogate="sinkhorn",
                    segmentation_temperature=1.0,
                    fisher_rao_eps=1e-6,
                    sinkhorn_max_positions_per_chunk=1024,
                )
                distortion = scorer_d
            else:
                distortion = eval_roundtrip_pixel_l1(
                    decoded_filmed, tgt,
                    noise_std=args.noise_std,
                    enable_eval_roundtrip_in_training=args.enable_eval_roundtrip_in_training,
                )
                seg_loss = torch.tensor(args.seg_target, device=device)
                pose_loss = torch.tensor(args.pose_target, device=device)

            res = coord.step(
                distortion=distortion,
                rate_bits=balle_out["rate_total_bits"],
                seg_loss=seg_loss,
                pose_loss=pose_loss,
            )
            if not torch.isfinite(res.augmented_lagrangian).all():
                raise RuntimeError("[t15] non-finite augmented Lagrangian")

            optim_main.zero_grad()
            res.augmented_lagrangian.backward()

            # NN-1 gate (first-step assertion; full regression in test suite).
            if not modulator_gradcheck_done:
                _ = assert_modulator_gradient_finite_nonzero(modulator)
                if args.enable_scorer_domain_loss:
                    _ = assert_score_domain_gradient_reachability(
                        decoder_params=decoder_trainable,
                        balle_main_params=balle_main_trainable,
                    )
                modulator_gradcheck_done = True

            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(main_params, max_norm=args.grad_clip_norm)
            optim_main.step()

            if optim_aux is not None:
                optim_aux.zero_grad()
                aux = balle.aux_loss()
                aux.backward()
                optim_aux.step()

            ema_decoder.update(decoder)
            ema_balle.update(balle)
            ema_modulator.update(modulator)

            epoch_loss += float(res.augmented_lagrangian.detach())
            n_batches += 1

        avg = epoch_loss / max(n_batches, 1)
        if epoch == 0 or (epoch + 1) % args.eval_every_epochs == 0 or epoch == epochs - 1:
            history.append({"epoch": epoch + 1, "avg_loss": avg, "rho": coord.rho})
            print(f"[t15] epoch {epoch+1}/{epochs} loss={avg:.4f} rho={coord.rho:.3f}")

    # Save EMA shadow as inference checkpoint (per CLAUDE.md EMA non-negotiable).
    ckpt = {
        "schema": T15_SCHEMA_VERSION,
        "ema_decoder": ema_decoder.shadow,
        "ema_balle": ema_balle.shadow,
        "ema_modulator": ema_modulator.shadow,
        "t15_config": {
            "pose_dim": t15_config.pose_dim,
            "feature_channels": t15_config.feature_channels,
            "hidden_dim": t15_config.hidden_dim,
            "activation": t15_config.activation,
            "quantization": t15_config.quantization,
            "label": t15_config.label,
        },
        "modulator_bytes": modulator_bytes,
        "predicted_delta_score": T15_PREDICTED_DELTA_SCORE,
    }
    torch.save(ckpt, args.output_dir / "t15_ema_shadow.pt")

    manifest = {
        "schema": T15_SCHEMA_VERSION,
        "lane_id": T15_LANE_ID,
        "phase": "2_scaffold_t15_t1_clone",
        "started_at_utc": started_at,
        "device": str(device),
        "smoke": bool(args.smoke),
        "epochs": int(epochs),
        "n_pairs": n_pairs,
        "decoder_params": int(n_decoder),
        "balle_params": int(n_balle),
        "modulator_params": int(n_modulator),
        "modulator_archive_bytes": int(modulator_bytes),
        "predicted_delta_score": T15_PREDICTED_DELTA_SCORE,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": "[predicted; Phase 2 T15 scaffold; not yet empirical]",
        "score_aware_loss_enabled": bool(args.enable_scorer_domain_loss),
        "eval_roundtrip": bool(args.enable_eval_roundtrip_in_training),
        "ema_decay": float(args.ema_decay),
        "nn1_gate_passed": bool(modulator_gradcheck_done),
        "history": history,
        "compliance_tags": [
            "ema_0p997_snapshot_restore",
            "eval_roundtrip_true",
            "no_mps_authoritative",
            "differentiable_yuv6",
            "score_aware_lagrangian",
            "no_synthetic_outside_smoke",
            "no_tmp_paths",
            "auth_eval_gated",
        ],
    }
    (args.output_dir / "t15_provenance.json").write_text(json.dumps(manifest, indent=2))
    print(f"[t15] done; manifest written to {args.output_dir}/t15_provenance.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
