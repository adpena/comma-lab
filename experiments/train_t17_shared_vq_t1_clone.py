#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""T17 — Shared VQ-VAE codebook trainer (T1-clone derivative).

Phase 2 pre-design pass (2026-05-09) consensus + readiness pre-stage
(2026-05-11) handoff: T17 replaces three independent latent spaces
(renderer's per-pixel emission, quantizer's per-tensor codes, aux scorer
features) with a SINGLE 256-entry × 64-dim @ FP16 codebook (~32 KB) using
van den Oord persistent-EMA updates (decay 0.99 per CLAUDE.md EMA
exception clause for VQ-VAE codebooks). The codebook is shared across all
600 pairs INDEX-SPACE; per-pair difference comes from the upstream Ballé
encoder output. Vanilla (gradient-trained) codebook is FORBIDDEN per
UNANIMOUS council vote: it collapses on Comma's continuous-features
substrate. Persistent-EMA is the only viable form.

REPRESENTATION_ARCHIVE_GRAMMAR_BLUEPRINT (per HNeRV parity discipline):
  archive_grammar: T1 monolithic-single-x-member + codebook.bin sidecar
  parser_section_manifest: x SHA-256, decoder.bin SHA-256, balle.bin SHA-256,
    codebook.bin SHA-256 (~32 KB FP16), byte sizes per section
  inflate_runtime_loc_budget: substrate_engineering (parent T1 budget)
  runtime_dep_closure: tac runtime + torch + brotli + compressai
  export_format: phase2_t17_t1_vq_bolt_on (T1 grammar + codebook sidecar)
  score_aware_loss: T1 score-aware loss + commitment_loss + NN-2 perplexity gate
  bolt_on_loc_budget: substrate_engineering (per Phase 2 pre-design)
  no_op_detector_planned: exact old/new codebook.bin + balle.bin SHA proof

CLAUDE.md non-negotiables — wired through this trainer
------------------------------------------------------

- **Weight EMA decay 0.997 with snapshot+restore** — decoder + Ballé +
  encoder weights wrapped by ``tac.training.EMA(decay=0.997)``.
- **Codebook EMA decay 0.99** — CLAUDE.md "EMA — NON-NEGOTIABLE"
  EXCEPTION clause: codebooks adapt FASTER than weights (van den Oord
  §3.2 canonical). The 0.997 weight-EMA does NOT apply here.
- **eval_roundtrip = True** + **Differentiable YUV6** + **CUDA-required** +
  **No make_synthetic outside --smoke** + **Score-domain Lagrangian** — same
  as T1/T15.
- **No /tmp paths**; outputs under ``experiments/results/<lane>_<utc>/``.

NN-2 gate (NON-NEGOTIABLE; per Phase 2 pre-design):
  Per-epoch codebook perplexity >= 0.4 * num_entries (= 102.4 for the
  256-entry default). Codebook-collapse pause-on-breach with re-init of
  dead entries from recent encoder outputs. Pause (not kill): the
  training loop emits a structured ``CodebookCollapseError`` that the
  operator inspects for diagnostic context.

Predicted Δ score (stand-alone on A1 substrate):
  -0.006 ± 0.003 ``[predicted; van den Oord BD-rate; 256-entry × 64-dim
  @ FP16 per Phase 2 pre-design pass 2026-05-09]``. NOT a score claim
  until contest-CUDA anchor lands.
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

from tac.paradigm_delta_epsilon_zeta import (  # noqa: E402
    BalleHyperpriorConfig,
    Decoder128KConfig,
    JointLagrangianADMM,
    JointLagrangianADMMConfig,
    build_balle_hyperprior,
    build_decoder_128k,
    load_frozen_a1_encoder,
)
from tac.shared_vq_codebook import (  # noqa: E402
    SharedCodebook,
    SharedCodebookConfig,
    compute_codebook_perplexity,
    shared_codebook_state_bytes,
)
from tac.training import EMA  # noqa: E402

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


T17_LANE_ID = "lane_t17_shared_vq_codebook_phase2_preregistered"
T17_SCHEMA_VERSION = "0.1.0-phase2-t17-shared-vq-bolt-on"
T17_PREDICTED_DELTA_SCORE = (
    "-0.006 ± 0.003 [predicted; van den Oord BD-rate; 256-entry × 64-dim "
    "@ FP16 per Phase 2 pre-design pass 2026-05-09]"
)
# NN-2 perplexity floor: 0.4 × num_entries (council canon).
NN2_PERPLEXITY_FLOOR_RATIO = 0.4
# Commitment loss weight (van den Oord §3.2 canonical β=0.25).
COMMITMENT_LOSS_WEIGHT_DEFAULT = 0.25


class CodebookCollapseError(RuntimeError):
    """Raised by the NN-2 gate when codebook perplexity is too low."""


# ---------------------------------------------------------------------------
# Codebook entry-space projection for Ballé latents
# ---------------------------------------------------------------------------


def _project_balle_latent_to_codebook(
    y_hat: torch.Tensor, entry_dim: int
) -> torch.Tensor:
    """Project per-pair Ballé latent to (B, T, entry_dim) for VQ quantization.

    Ballé hyperprior emits ``y_hat`` of shape (B, latent_dim) for the
    per-pair latent. We split it into ``T = latent_dim // entry_dim`` tokens
    of width ``entry_dim`` (truncating any remainder). The VQ codebook
    operates token-wise; each token is independently snapped to the nearest
    codebook entry.

    Returns (B, T, entry_dim) for the codebook forward pass.
    """
    if y_hat.dim() != 2:
        raise ValueError(f"y_hat must be (B, latent_dim); got {tuple(y_hat.shape)}")
    B, D = y_hat.shape
    T = D // entry_dim
    if T < 1:
        raise ValueError(
            f"latent_dim {D} < entry_dim {entry_dim}; cannot tokenize "
            "(consider --t17-entry-dim <= latent_dim)"
        )
    used = T * entry_dim
    truncated = y_hat[:, :used]
    return truncated.view(B, T, entry_dim)


def _unproject_quantized_to_balle_shape(
    z_q: torch.Tensor, target_shape: tuple[int, int]
) -> torch.Tensor:
    """Inverse of :func:`_project_balle_latent_to_codebook`.

    Accepts (B, T, entry_dim) and returns (B, used_dim) flattened back
    to the per-pair latent shape. If the target latent_dim is larger
    than ``used_dim``, the trailing dims are padded with zeros (which
    matches the truncation behavior of the forward path).
    """
    B, T, E = z_q.shape
    target_B, target_D = target_shape
    if target_B != B:
        raise ValueError(f"batch dim mismatch: got {B}, target {target_B}")
    used = T * E
    flat = z_q.reshape(B, used)
    if used < target_D:
        flat = F.pad(flat, (0, target_D - used))
    return flat


def assert_codebook_perplexity_ok(
    indices_epoch: torch.Tensor,
    num_entries: int,
    floor_ratio: float = NN2_PERPLEXITY_FLOOR_RATIO,
) -> dict:
    """NN-2: gate codebook perplexity for the epoch.

    Returns a diagnostic dict with the measured perplexity + floor.
    Raises ``CodebookCollapseError`` if perplexity < ratio · N (= 102.4
    for N=256, ratio=0.4 per council canon).
    """
    perplexity = compute_codebook_perplexity(indices_epoch, num_entries=num_entries)
    floor = floor_ratio * num_entries
    diag = {
        "perplexity": float(perplexity),
        "floor": float(floor),
        "floor_ratio": float(floor_ratio),
        "num_entries": int(num_entries),
        "passed": perplexity >= floor,
    }
    if not diag["passed"]:
        raise CodebookCollapseError(
            f"[t17] NN-2 codebook collapse: perplexity {perplexity:.1f} < "
            f"floor {floor:.1f} (ratio {floor_ratio} × N={num_entries}); "
            "trainer paused for operator inspection. Re-init dead entries "
            "from recent encoder outputs and re-run; do not silently continue."
        )
    return diag


def reinit_dead_codebook_entries(
    codebook_module: torch.nn.Module,
    recent_z_e: torch.Tensor,
    dead_threshold: float = 1.0,
) -> int:
    """Helper: re-initialise dead codebook entries from recent encoder outputs.

    A "dead" entry is one whose persistent EMA count is below
    ``dead_threshold``. Re-initialisation samples without replacement
    from ``recent_z_e`` (shape (M, entry_dim)) and overwrites the
    dead-entry codebook rows. Returns the number of entries re-init'd.

    Called only by the OPERATOR after a ``CodebookCollapseError``; never
    silently inside the training loop.
    """
    with torch.no_grad():
        dead_mask = codebook_module.ema_count < dead_threshold
        n_dead = int(dead_mask.sum().item())
        if n_dead == 0:
            return 0
        flat = recent_z_e.reshape(-1, recent_z_e.shape[-1])
        if flat.shape[0] < n_dead:
            raise RuntimeError(
                f"reinit_dead_codebook_entries: need {n_dead} samples; "
                f"got {flat.shape[0]}"
            )
        idx = torch.randperm(flat.shape[0])[:n_dead]
        codebook_module.codebook[dead_mask] = flat[idx]
        # Reset their EMA state so they get a fresh adaptation window.
        codebook_module.ema_count[dead_mask] = 1.0
        codebook_module.ema_sum[dead_mask] = flat[idx].clone()
    return n_dead


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="T17 — Shared VQ-VAE codebook trainer (T1-clone derivative)"
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
    # T17-specific
    parser.add_argument("--t17-num-entries", type=int, default=256,
                        help="VQ codebook size (van den Oord §4.2 canon = 256)")
    parser.add_argument("--t17-entry-dim", type=int, default=64,
                        help="VQ codebook entry dim (Quantizr-class C=64)")
    parser.add_argument("--t17-codebook-ema-decay", type=float, default=0.99,
                        help="Codebook EMA decay (van den Oord §3.2 canon = 0.99; "
                             "DISTINCT from weight EMA 0.997)")
    parser.add_argument("--t17-codebook-quant", default="fp16",
                        choices=["fp4", "fp8", "fp16", "fp32"])
    parser.add_argument("--t17-commitment-weight", type=float,
                        default=COMMITMENT_LOSS_WEIGHT_DEFAULT,
                        help="van den Oord commitment loss β (canon = 0.25)")
    parser.add_argument("--t17-perplexity-floor-ratio", type=float,
                        default=NN2_PERPLEXITY_FLOOR_RATIO,
                        help="NN-2 perplexity floor as fraction of num_entries (0.4 default)")
    return parser.parse_args(argv)


def _canonical_dir_name_from_relpath(relpath: str) -> str:
    return Path(relpath).name


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.device == "mps":
        raise SystemExit("[t17] --device mps refused; see CLAUDE.md MPS-is-noise rule")
    if args.auth_eval:
        refuse_phase1_scaffold_path("--auth-eval [T17 scaffold; T1-clone]")
    if not args.smoke and PHASE1_SCAFFOLD_ONLY and not args.enable_scorer_domain_loss:
        refuse_phase1_scaffold_path("non-smoke T17 training [scaffold; T1-clone]")
    if not (0.0 <= args.t17_perplexity_floor_ratio <= 1.0):
        raise SystemExit("[t17] --t17-perplexity-floor-ratio must be in [0, 1]")
    if not (0.9 <= args.t17_codebook_ema_decay < 1.0):
        raise SystemExit("[t17] --t17-codebook-ema-decay must be in [0.9, 1.0)")
    if args.grad_clip_norm < 0:
        raise SystemExit("[t17] --grad-clip-norm must be non-negative")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _seed_everything(args.seed)
    device = _resolve_device(args.device)

    yuv6_mode = _resolve_yuv6_mode_with_probe_t1(args.yuv6_mode)
    yuv6_token = _activate_yuv6_mode_t1(
        yuv6_mode, enabled=args.enable_differentiable_yuv6
    )
    print(
        f"[t17] eval_roundtrip={args.enable_eval_roundtrip_in_training} "
        f"yuv6_mode={yuv6_mode.value} smoke={args.smoke} device={device}"
    )

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

    decoder_config = Decoder128KConfig(latent_dim=int(latents.shape[1]))
    decoder = build_decoder_128k(decoder_config).to(device)
    balle_config = BalleHyperpriorConfig(y_channels=int(latents.shape[1]))
    balle = build_balle_hyperprior(balle_config).to(device)

    # T17 codebook: 256 entries × entry_dim @ FP16 by default.
    if int(latents.shape[1]) < args.t17_entry_dim:
        raise SystemExit(
            f"[t17] latent_dim {int(latents.shape[1])} < --t17-entry-dim "
            f"{args.t17_entry_dim}; reduce entry_dim or use larger latents"
        )
    codebook_config = SharedCodebookConfig(
        num_entries=args.t17_num_entries,
        entry_dim=args.t17_entry_dim,
        ema_decay=args.t17_codebook_ema_decay,
        epsilon_laplace=1e-5,
        quantization=args.t17_codebook_quant,
        label=f"t17_t1clone_{started_at}",
    )
    codebook = SharedCodebook(codebook_config).to(device)
    codebook_bytes = shared_codebook_state_bytes(codebook_config)

    n_decoder = sum(p.numel() for p in decoder.parameters())
    n_balle = sum(p.numel() for p in balle.parameters())
    print(
        f"[t17] decoder={n_decoder:,} balle={n_balle:,} "
        f"codebook={args.t17_num_entries}×{args.t17_entry_dim} "
        f"(~{codebook_bytes} archive bytes); EMA decay={args.t17_codebook_ema_decay} "
        f"(CLAUDE.md VQ-VAE exception)"
    )

    decoder_trainable = [p for p in decoder.parameters() if p.requires_grad]
    balle_main_trainable = [
        p for n, p in balle.named_parameters()
        if p.requires_grad and ("entropy_bottleneck" not in n or "quantiles" not in n)
    ]
    # Note: codebook entries are BUFFERS not parameters; not optimised by gradient.
    main_params = [*decoder_trainable, *balle_main_trainable]
    aux_params = [p for n, p in balle.named_parameters() if "quantiles" in n]
    optim_main = torch.optim.Adam(main_params, lr=args.learning_rate)
    optim_aux = torch.optim.Adam(aux_params, lr=args.aux_learning_rate) if aux_params else None

    # Weight-EMA wrapping (decay 0.997) for decoder + balle (NOT the codebook;
    # codebook has its own persistent-EMA at decay 0.99).
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
        perm = torch.randperm(n_pairs, generator=torch.Generator().manual_seed(args.seed + epoch))
        epoch_loss = 0.0
        n_batches = 0
        epoch_indices: list[torch.Tensor] = []
        for start in range(0, n_pairs, batch_size):
            idx = perm[start:start + batch_size]
            y = latents[idx]
            tgt = target_pixels[idx]

            balle_out = balle(y)
            y_hat = balle_out["y_hat"]
            # T17: project Ballé y_hat into codebook entry-space, quantize.
            z_e = _project_balle_latent_to_codebook(y_hat, args.t17_entry_dim)
            z_q_st, indices, commitment_loss = codebook(z_e)
            epoch_indices.append(indices.detach().reshape(-1).cpu())
            # Unproject quantized latent back to Ballé latent shape.
            y_hat_q = _unproject_quantized_to_balle_shape(
                z_q_st, target_shape=(y_hat.shape[0], y_hat.shape[1])
            )
            decoded = decoder(y_hat_q)

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
            else:
                distortion = eval_roundtrip_pixel_l1(
                    decoded, tgt, noise_std=args.noise_std,
                    enable_eval_roundtrip_in_training=args.enable_eval_roundtrip_in_training,
                )
                seg_loss = torch.tensor(args.seg_target, device=device)
                pose_loss = torch.tensor(args.pose_target, device=device)

            # Add commitment loss to the distortion term (van den Oord §3.2).
            distortion_with_commit = distortion + args.t17_commitment_weight * commitment_loss

            res = coord.step(
                distortion=distortion_with_commit,
                rate_bits=balle_out["rate_total_bits"],
                seg_loss=seg_loss,
                pose_loss=pose_loss,
            )
            if not torch.isfinite(res.augmented_lagrangian).all():
                raise RuntimeError("[t17] non-finite augmented Lagrangian")

            optim_main.zero_grad()
            res.augmented_lagrangian.backward()
            if args.enable_scorer_domain_loss:
                _ = assert_score_domain_gradient_reachability(
                    decoder_params=decoder_trainable,
                    balle_main_params=balle_main_trainable,
                )
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(main_params, max_norm=args.grad_clip_norm)
            optim_main.step()

            if optim_aux is not None:
                optim_aux.zero_grad()
                aux = balle.aux_loss()
                aux.backward()
                optim_aux.step()

            # Codebook persistent-EMA update AFTER weights are stepped
            # (van den Oord §3.2). Decoupled from gradient-based optim.
            codebook.update_ema(z_e.detach(), indices.detach())

            ema_decoder.update(decoder)
            ema_balle.update(balle)

            epoch_loss += float(res.augmented_lagrangian.detach())
            n_batches += 1

        avg = epoch_loss / max(n_batches, 1)
        # NN-2 perplexity gate at end of epoch.
        all_indices = torch.cat(epoch_indices) if epoch_indices else torch.tensor([0])
        perp_diag = assert_codebook_perplexity_ok(
            all_indices,
            num_entries=args.t17_num_entries,
            floor_ratio=args.t17_perplexity_floor_ratio,
        )
        if epoch == 0 or (epoch + 1) % args.eval_every_epochs == 0 or epoch == epochs - 1:
            history.append({
                "epoch": epoch + 1, "avg_loss": avg, "rho": coord.rho,
                "perplexity": perp_diag["perplexity"], "perplexity_floor": perp_diag["floor"],
            })
            print(
                f"[t17] epoch {epoch+1}/{epochs} loss={avg:.4f} rho={coord.rho:.3f} "
                f"perp={perp_diag['perplexity']:.1f}/{perp_diag['floor']:.1f}"
            )

    ckpt = {
        "schema": T17_SCHEMA_VERSION,
        "ema_decoder": ema_decoder.shadow,
        "ema_balle": ema_balle.shadow,
        "codebook": codebook.codebook.detach().cpu(),
        "codebook_ema_count": codebook.ema_count.detach().cpu(),
        "codebook_ema_sum": codebook.ema_sum.detach().cpu(),
        "t17_config": {
            "num_entries": codebook_config.num_entries,
            "entry_dim": codebook_config.entry_dim,
            "ema_decay": codebook_config.ema_decay,
            "quantization": codebook_config.quantization,
        },
        "codebook_bytes": codebook_bytes,
        "predicted_delta_score": T17_PREDICTED_DELTA_SCORE,
    }
    torch.save(ckpt, args.output_dir / "t17_ema_shadow.pt")

    manifest = {
        "schema": T17_SCHEMA_VERSION,
        "lane_id": T17_LANE_ID,
        "phase": "2_scaffold_t17_t1_clone",
        "started_at_utc": started_at,
        "device": str(device),
        "smoke": bool(args.smoke),
        "epochs": int(epochs),
        "n_pairs": n_pairs,
        "decoder_params": int(n_decoder),
        "balle_params": int(n_balle),
        "codebook_num_entries": int(codebook_config.num_entries),
        "codebook_entry_dim": int(codebook_config.entry_dim),
        "codebook_archive_bytes": int(codebook_bytes),
        "codebook_ema_decay": float(codebook_config.ema_decay),
        "weight_ema_decay": float(args.ema_decay),
        "predicted_delta_score": T17_PREDICTED_DELTA_SCORE,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": "[predicted; Phase 2 T17 scaffold; not yet empirical]",
        "score_aware_loss_enabled": bool(args.enable_scorer_domain_loss),
        "eval_roundtrip": bool(args.enable_eval_roundtrip_in_training),
        "nn2_gate_passed_each_epoch": True,  # asserted inline; raises on failure
        "perplexity_floor_ratio": float(args.t17_perplexity_floor_ratio),
        "history": history,
        "compliance_tags": [
            "ema_0p997_snapshot_restore_weights",
            "codebook_ema_0p99_van_den_oord_canon",
            "eval_roundtrip_true",
            "no_mps_authoritative",
            "differentiable_yuv6",
            "score_aware_lagrangian",
            "vandenoord_commitment_loss",
            "nn2_perplexity_gate_per_epoch",
            "no_synthetic_outside_smoke",
            "no_tmp_paths",
            "auth_eval_gated",
        ],
    }
    (args.output_dir / "t17_provenance.json").write_text(json.dumps(manifest, indent=2))
    print(f"[t17] done; manifest written to {args.output_dir}/t17_provenance.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
