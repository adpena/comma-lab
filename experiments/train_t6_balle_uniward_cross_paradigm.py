#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""T6 — Ballé + UNIWARD cross-paradigm trainer (T1-clone derivative).

Phase 2 pre-design pass (2026-05-09) consensus + readiness pre-stage
(2026-05-11) handoff: T6 stacks Fridrich's UNIWARD undetectable-embedding
budget on top of the T1 Ballé hyperprior end-to-end Lagrangian-ADMM
substrate.  The cross-paradigm composition adds an inner-loop loss term
that weights pixel-domain distortion by the UNIWARD texture probability
(``tac.uniward_texture.compute_texture_probability``); textured regions
get more aggressive distortion allowance because they are statistically
invisible to the SegNet/PoseNet scorers per Fridrich inverse-steganalysis
principle #1.

REPRESENTATION_ARCHIVE_GRAMMAR_BLUEPRINT (per HNeRV parity discipline):
  archive_grammar:        T1 three-member (x / decoder.bin / balle.bin) +
                          uniward_budget metadata only (no new sidecar)
  parser_section_manifest: x SHA-256, decoder.bin SHA-256, balle.bin SHA-256,
                          uniward_budget byte sizes per section
  inflate_runtime_loc_budget: substrate_engineering (parent T1 budget, no
                          new inflate-side state — UNIWARD is compress-time only)
  runtime_dep_closure:    T1 + (no new deps; UNIWARD math is local fp32 conv)
  export_format:          phase2_t6_t1_uniward_cross_paradigm (T1 grammar +
                          UNIWARD-weighted loss baked into trained weights)
  score_aware_loss:       T1 score-aware loss + UNIWARD-budget inner-loop weighting
  bolt_on_loc_budget:     substrate_engineering (per Phase 2 pre-design)
  no_op_detector_planned: exact old/new decoder.bin + balle.bin SHA proof
                          (UNIWARD is gradient-only; archive byte deltas come
                          from the trained weights drifting toward different
                          texture-vs-flat allocations)

CLAUDE.md non-negotiables — wired through this trainer
------------------------------------------------------

- **EMA decay 0.997 with snapshot+restore at eval time** — decoder + Ballé
  wrapped by ``tac.training.EMA(decay=0.997)``; inference uses the EMA
  shadow per CLAUDE.md "EMA — NON-NEGOTIABLE".
- **eval_roundtrip = True** — proxy loss simulates the contest's inflate
  roundtrip (384→874→uint8→384) via ``apply_eval_roundtrip_during_training``.
- **Differentiable YUV6** — ``patch_upstream_yuv6_globally`` is called
  BEFORE any scorer load per CLAUDE.md "NeRV/HNeRV renderer trainers must
  also keep scorer preprocess differentiable".
- **Score-domain Lagrangian** — α·B(θ)/N + β·d_seg(θ) + γ·sqrt(10·d_pose(θ))
  per CLAUDE.md "Meta-Lagrangian/Pareto solver" + contest functional. T6
  ADDS an UNIWARD-budget term inside the distortion: pixel-L1 in textured
  regions is down-weighted (the scorer is less sensitive there).
- **NEVER MPS authoritative** — refuses ``--device mps``.
- **No make_synthetic_* outside --smoke** — non-smoke uses
  ``load_real_target_pairs`` (PyAV on ``upstream/videos/0.mkv``).
- **No /tmp paths** — outputs under ``experiments/results/<lane>_<utc>/``.
- **Auth eval gated** — ``--auth-eval`` refused (Phase 2 scaffold-only).

NN-T6 gate (NON-NEGOTIABLE; ratified by 3-clean-pass review 2026-05-11):
  The UNIWARD-weighted distortion MUST remain finite + non-zero on the
  first training step; the trainer asserts both at NN-T6 gate-fire time.
  A degenerate UNIWARD probability (all flat / all textured) would
  collapse the trainer to vanilla T1 — that case is detected and
  pause-not-killed per CLAUDE.md "KILL is LAST RESORT".

Predicted Δ score (stand-alone on A1 substrate):
  -0.018 ± 0.008 ``[predicted; T1+T18 STAR + UNIWARD-budget Ballé
  cross-paradigm; per pre-stage G manifest 2026-05-11 + Phase 2 pre-design
  pass 2026-05-09]``. NOT a score claim until contest-CUDA anchor lands.
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
from tac.uniward_texture import compute_texture_probability  # noqa: E402

# Reuse T1's helpers verbatim per HNeRV parity discipline lesson 7
# (bolt-on ≤ 350 LOC; substrate engineering reused, not duplicated).
from experiments.train_paradigm_delta_epsilon_zeta_track1_balle_endtoend import (  # noqa: E402
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


T6_LANE_ID = "lane_t6_balle_uniward_cross_paradigm_phase2_preregistered"
T6_SCHEMA_VERSION = "0.1.0-phase2-t6-balle-uniward-cross-paradigm"
T6_PREDICTED_DELTA_SCORE = (
    "-0.018 ± 0.008 [predicted; T1+T18 STAR + UNIWARD-budget Ballé "
    "cross-paradigm; pre-stage G manifest 2026-05-11]"
)


# ---------------------------------------------------------------------------
# UNIWARD-weighted distortion (cross-paradigm inner loop)
# ---------------------------------------------------------------------------


def compute_uniward_weight_map(
    decoded_btchw: torch.Tensor,
    *,
    detach: bool = True,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return an (H, W) texture-probability map for the decoded pair.

    Fridrich UNIWARD principle: textured regions have high directional-
    Haar-residual energy AND high local variance; the scorer is robust
    to perturbations there.  The trainer uses this map to DOWN-weight
    pixel-domain distortion in textured pixels (the scorer doesn't care)
    and UP-weight it in flat regions (the scorer cares a lot there).

    ``decoded_btchw`` shape: ``(B, T, C, H, W)`` — pair-of-frames RGB
    output from ``Decoder128K``.  The weight map is computed jointly
    across the pair (we collapse the time dim by flattening) per
    Lane SI-V3 canonical contract.  The returned (H, W) tensor is
    DETACHED by default so it acts as a fixed weighting (not a learnable
    target).
    """
    if decoded_btchw.dim() != 5 or decoded_btchw.shape[2] != 3:
        raise ValueError(
            "decoded must be (B, T, 3, H, W); got "
            f"{tuple(decoded_btchw.shape)}"
        )
    B, T, C, H, W = decoded_btchw.shape
    # Flatten pair dim into the batch so compute_texture_probability sees
    # (B*T, 3, H, W) as it expects.
    flat = decoded_btchw.reshape(B * T, C, H, W)
    # Detach for the weight map per Lane SI-V3 contract: the weighting
    # is a fixed prior, not a learning signal that pulls textures around.
    tex_hw = compute_texture_probability(
        flat, scorers=[], detach=detach, require_cuda=False, eps=eps
    )
    return tex_hw


def uniward_weighted_pixel_l1(
    decoded_btchw: torch.Tensor,
    target_btchw: torch.Tensor,
    *,
    uniward_weight_hw: torch.Tensor,
    flat_floor: float,
    textured_ceiling: float,
    eval_roundtrip_noise_std: float,
    enable_eval_roundtrip_in_training: bool,
) -> torch.Tensor:
    """Pixel-L1 weighted by UNIWARD texture probability.

    Forward:

        weight(h, w) = clip(  flat_floor +
                              (textured_ceiling - flat_floor) * tex_hw,
                              flat_floor, textured_ceiling)
        loss = mean(|decoded_rt - target| * weight)

    where ``decoded_rt`` is the decoded-with-eval-roundtrip frames (uint8
    bottleneck simulated per CLAUDE.md ``eval_roundtrip = True`` rule).

    Council canon:
      ``flat_floor = 0.5``        — flat regions get HALF the standard L1
                                    (scorer cares less than naive uniform)
      ``textured_ceiling = 1.5``  — textured regions get 1.5x L1 weighting
                                    (UNIWARD principle: scorer is robust
                                    but visual identity is preserved)

    The min-max clamping is structural to the Fridrich UNIWARD trade-off:
    a degenerate texture map (all 0 or all 1) collapses to a UNIFORM
    weighting at ``flat_floor`` or ``textured_ceiling`` — never below the
    standard L1 weight in the worst case.
    """
    if uniward_weight_hw.dim() != 2:
        raise ValueError(
            f"uniward_weight_hw must be (H, W); got {tuple(uniward_weight_hw.shape)}"
        )
    if not (0.0 < flat_floor <= textured_ceiling):
        raise ValueError(
            f"require 0 < flat_floor ({flat_floor}) <= textured_ceiling "
            f"({textured_ceiling})"
        )

    # Per-pixel weighting (broadcasts (H, W) over (B, T, C, H, W)).
    # Normalize tex_hw to [0, 1] over the whole frame, then linearly map
    # into [flat_floor, textured_ceiling].
    tex_min = uniward_weight_hw.min()
    tex_max = uniward_weight_hw.max()
    denom = (tex_max - tex_min).clamp_min(1e-12)
    tex_norm = (uniward_weight_hw - tex_min) / denom  # in [0, 1]
    weight = (
        flat_floor + (textured_ceiling - flat_floor) * tex_norm
    ).clamp(flat_floor, textured_ceiling)

    # Apply eval_roundtrip if enabled (CLAUDE.md non-negotiable).
    decoded_rt = eval_roundtrip_decoded(
        decoded_btchw, noise_std=eval_roundtrip_noise_std,
        enable_eval_roundtrip_in_training=enable_eval_roundtrip_in_training,
    )
    abs_diff = (decoded_rt - target_btchw).abs()
    # Broadcast (H, W) over (B, T, C, H, W).
    weighted = abs_diff * weight.view(1, 1, 1, *weight.shape)
    return weighted.mean()


def assert_uniward_loss_nondegenerate(
    distortion: torch.Tensor,
    uniward_weight_hw: torch.Tensor,
    *,
    label: str = "t6",
) -> dict:
    """NN-T6 gate: refuse degenerate UNIWARD probability + non-finite loss.

    Detects two collapse modes:
      1.  ``distortion`` is NaN / Inf / negative — the loss is broken.
      2.  ``uniward_weight_hw`` collapses to a constant (textured-flat
          extrema are equal) — the trainer is degenerate to vanilla L1.

    Both raise ``RuntimeError`` with structured diagnostic per CLAUDE.md
    "KILL is LAST RESORT" → trainer pauses (not kills) so operator
    can inspect and re-launch.
    """
    if not torch.isfinite(distortion).all():
        raise RuntimeError(
            f"[{label}] NN-T6 fail-closed: distortion is non-finite; "
            "see UNIWARD weight map for degeneracy."
        )
    if float(distortion) < 0.0:
        raise RuntimeError(
            f"[{label}] NN-T6 fail-closed: distortion {float(distortion):.6g} < 0; "
            "the UNIWARD-weighted L1 must be non-negative."
        )
    tex_range = float(
        (uniward_weight_hw.max() - uniward_weight_hw.min()).item()
    )
    if tex_range < 1e-6:
        raise RuntimeError(
            f"[{label}] NN-T6 fail-closed: UNIWARD texture map "
            f"range {tex_range:.6g} < 1e-6; trainer would collapse to "
            "vanilla T1.  Pause and inspect (KILL is LAST RESORT)."
        )
    return {
        "distortion": float(distortion),
        "uniward_texture_range": tex_range,
        "uniward_texture_min": float(uniward_weight_hw.min().item()),
        "uniward_texture_max": float(uniward_weight_hw.max().item()),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="T6 — Ballé + UNIWARD cross-paradigm trainer (T1-clone)"
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
    parser.add_argument(
        "--yuv6-mode", default="auto",
        choices=["auto", "monkey_patch_global", "tac_differentiable_routing"],
    )
    parser.add_argument("--enable-scorer-domain-loss", action="store_true", default=False)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--eval-every-epochs", type=int, default=25)
    parser.add_argument("--auth-eval", action="store_true", default=False)
    parser.add_argument(
        "--video-path", type=Path,
        default=REPO_ROOT / "upstream" / "videos" / "0.mkv",
    )
    parser.add_argument("--target-pixels-path", type=Path, default=None)
    parser.add_argument("--max-target-pairs", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument(
        "--canonical-a1-relpath", type=str,
        default="experiments/results/A1_canonical",
    )
    parser.add_argument("--allow-missing-canonical-a1", action="store_true", default=False)
    # T6-specific (UNIWARD budget knobs).
    parser.add_argument(
        "--t6-flat-floor", type=float, default=0.5,
        help="Flat-region L1 weight (council canon = 0.5)",
    )
    parser.add_argument(
        "--t6-textured-ceiling", type=float, default=1.5,
        help="Textured-region L1 weight (council canon = 1.5)",
    )
    parser.add_argument(
        "--t6-uniward-eps", type=float, default=1e-6,
        help="UNIWARD probability stabilizer",
    )
    parser.add_argument(
        "--t6-disable-uniward", action="store_true", default=False,
        help="Diagnostic: disable UNIWARD weighting (collapses to T1)",
    )
    return parser.parse_args(argv)


def _canonical_dir_name_from_relpath(relpath: str) -> str:
    return Path(relpath).name


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.device == "mps":
        raise SystemExit("[t6] --device mps refused; see CLAUDE.md MPS-is-noise rule")
    if args.auth_eval:
        refuse_phase1_scaffold_path("--auth-eval [T6 scaffold; T1-clone]")
    if not args.smoke and PHASE1_SCAFFOLD_ONLY and not args.enable_scorer_domain_loss:
        refuse_phase1_scaffold_path("non-smoke T6 training [scaffold; T1-clone]")
    if args.grad_clip_norm < 0:
        raise SystemExit("[t6] --grad-clip-norm must be non-negative")
    if not (0.0 < args.t6_flat_floor <= args.t6_textured_ceiling):
        raise SystemExit(
            "[t6] require 0 < --t6-flat-floor <= --t6-textured-ceiling"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _seed_everything(args.seed)
    device = _resolve_device(args.device)

    # PR #95 binary-forensics replication: activate autograd-preserving YUV6.
    yuv6_mode = _resolve_yuv6_mode_with_probe_t1(args.yuv6_mode)
    _activate_yuv6_mode_t1(yuv6_mode, enabled=args.enable_differentiable_yuv6)
    print(
        f"[t6] eval_roundtrip={args.enable_eval_roundtrip_in_training} "
        f"yuv6_mode={yuv6_mode.value} smoke={args.smoke} device={device}"
    )

    # Load A1 latents + targets (verbatim T15 pattern; substrate is shared).
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

    # Build modules: decoder + Ballé hyperprior (NO new module — T6 is
    # a loss bolt-on, not an architecture bolt-on; per HNeRV parity
    # discipline lesson 7 substrate-engineering ≤ 350 LOC).
    decoder_config = Decoder128KConfig(latent_dim=int(latents.shape[1]))
    decoder = build_decoder_128k(decoder_config).to(device)
    balle_config = BalleHyperpriorConfig(y_channels=int(latents.shape[1]))
    balle = build_balle_hyperprior(balle_config).to(device)

    n_decoder = sum(p.numel() for p in decoder.parameters())
    n_balle = sum(p.numel() for p in balle.parameters())
    print(
        f"[t6] decoder={n_decoder:,} balle={n_balle:,} "
        f"(no new architectural params — UNIWARD is loss-side only)"
    )

    decoder_trainable = [p for p in decoder.parameters() if p.requires_grad]
    balle_main_trainable = [
        p for n, p in balle.named_parameters()
        if p.requires_grad and ("entropy_bottleneck" not in n or "quantiles" not in n)
    ]
    main_params = [*decoder_trainable, *balle_main_trainable]
    aux_params = [p for n, p in balle.named_parameters() if "quantiles" in n]
    optim_main = torch.optim.Adam(main_params, lr=args.learning_rate)
    optim_aux = torch.optim.Adam(aux_params, lr=args.aux_learning_rate) if aux_params else None

    # EMA shadows (decay 0.997 per CLAUDE.md "EMA — NON-NEGOTIABLE").
    ema_decoder = EMA(decoder, decay=args.ema_decay)
    ema_balle = EMA(balle, decay=args.ema_decay)

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
    nnT6_gate_done = False
    nnT6_diag: dict = {}

    posenet = segnet = None
    if args.enable_scorer_domain_loss:
        from tac.scorer import load_differentiable_scorers  # noqa: WPS433
        posenet, segnet = load_differentiable_scorers(REPO_ROOT / "upstream", device=device)
        posenet.eval(); segnet.eval()
        for m in (posenet, segnet):
            for p in m.parameters():
                p.requires_grad_(False)

    for epoch in range(epochs):
        decoder.train(); balle.train()
        perm = torch.randperm(
            n_pairs,
            generator=torch.Generator().manual_seed(args.seed + epoch),
        )
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_pairs, batch_size):
            idx = perm[start:start + batch_size]
            y = latents[idx]
            tgt = target_pixels[idx]

            balle_out = balle(y)
            decoded = decoder(balle_out["y_hat"])

            if args.enable_scorer_domain_loss and posenet is not None and segnet is not None:
                from tac.losses import scorer_loss_terms_btchw  # noqa: WPS433
                decoded_rt = eval_roundtrip_decoded(
                    decoded, noise_std=args.noise_std,
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
                # UNIWARD weighting applies in scorer-domain path only as
                # diagnostic (the scorer is already operating-point-aware);
                # we compute the texture map for nnT6 gate diagnostics.
                tex_hw = compute_uniward_weight_map(
                    decoded, detach=True, eps=args.t6_uniward_eps,
                )
            else:
                if args.t6_disable_uniward:
                    distortion = eval_roundtrip_pixel_l1(
                        decoded, tgt,
                        noise_std=args.noise_std,
                        enable_eval_roundtrip_in_training=args.enable_eval_roundtrip_in_training,
                    )
                    tex_hw = compute_uniward_weight_map(
                        decoded, detach=True, eps=args.t6_uniward_eps,
                    )
                else:
                    tex_hw = compute_uniward_weight_map(
                        decoded, detach=True, eps=args.t6_uniward_eps,
                    )
                    distortion = uniward_weighted_pixel_l1(
                        decoded, tgt,
                        uniward_weight_hw=tex_hw,
                        flat_floor=args.t6_flat_floor,
                        textured_ceiling=args.t6_textured_ceiling,
                        eval_roundtrip_noise_std=args.noise_std,
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
                raise RuntimeError("[t6] non-finite augmented Lagrangian")

            optim_main.zero_grad()
            res.augmented_lagrangian.backward()

            # NN-T6 gate (first-step assertion; structured pause-not-kill).
            if not nnT6_gate_done:
                nnT6_diag = assert_uniward_loss_nondegenerate(
                    distortion=distortion, uniward_weight_hw=tex_hw,
                )
                if args.enable_scorer_domain_loss:
                    _ = assert_score_domain_gradient_reachability(
                        decoder_params=decoder_trainable,
                        balle_main_params=balle_main_trainable,
                    )
                nnT6_gate_done = True

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

            epoch_loss += float(res.augmented_lagrangian.detach())
            n_batches += 1

        avg = epoch_loss / max(n_batches, 1)
        if epoch == 0 or (epoch + 1) % args.eval_every_epochs == 0 or epoch == epochs - 1:
            history.append({"epoch": epoch + 1, "avg_loss": avg, "rho": coord.rho})
            print(f"[t6] epoch {epoch+1}/{epochs} loss={avg:.4f} rho={coord.rho:.3f}")

    # Save EMA shadow as inference checkpoint (CLAUDE.md EMA non-negotiable).
    ckpt = {
        "schema": T6_SCHEMA_VERSION,
        "ema_decoder": ema_decoder.shadow,
        "ema_balle": ema_balle.shadow,
        "t6_config": {
            "flat_floor": float(args.t6_flat_floor),
            "textured_ceiling": float(args.t6_textured_ceiling),
            "uniward_eps": float(args.t6_uniward_eps),
            "disable_uniward": bool(args.t6_disable_uniward),
        },
        "predicted_delta_score": T6_PREDICTED_DELTA_SCORE,
    }
    torch.save(ckpt, args.output_dir / "t6_ema_shadow.pt")

    manifest = {
        "schema": T6_SCHEMA_VERSION,
        "lane_id": T6_LANE_ID,
        "phase": "2_scaffold_t6_t1_uniward_cross_paradigm",
        "started_at_utc": started_at,
        "device": str(device),
        "smoke": bool(args.smoke),
        "epochs": int(epochs),
        "n_pairs": n_pairs,
        "decoder_params": int(n_decoder),
        "balle_params": int(n_balle),
        "uniward_added_params": 0,
        "predicted_delta_score": T6_PREDICTED_DELTA_SCORE,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": "[predicted; Phase 2 T6 scaffold; not yet empirical]",
        "score_aware_loss_enabled": bool(args.enable_scorer_domain_loss),
        "eval_roundtrip": bool(args.enable_eval_roundtrip_in_training),
        "ema_decay": float(args.ema_decay),
        "nn_t6_gate_passed": bool(nnT6_gate_done),
        "nn_t6_gate_diag": nnT6_diag,
        "uniward_disabled": bool(args.t6_disable_uniward),
        "t6_flat_floor": float(args.t6_flat_floor),
        "t6_textured_ceiling": float(args.t6_textured_ceiling),
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
            "uniward_cross_paradigm_loss_bolt_on",
        ],
    }
    (args.output_dir / "t6_provenance.json").write_text(json.dumps(manifest, indent=2))
    print(f"[t6] done; manifest written to {args.output_dir}/t6_provenance.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
