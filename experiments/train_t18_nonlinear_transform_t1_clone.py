#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""T18 — Ballé nonlinear transform coding trainer (T1-clone derivative).

Phase 2 pre-design pass (2026-05-09) consensus + readiness pre-stage
(2026-05-11) handoff: T18 inserts a learned 4-layer MLP (GELU, He-Zheng
2024 canonical) BETWEEN the Ballé encoder output and the entropy
bottleneck. Skip-connection identity-init keeps T18 graceful at startup
(degrades to vanilla Ballé). Forward and inverse MLPs are SEPARATE
parameter sets per He-Zheng 2024 §3.2 (weight-tying degrades RD-frontier).

REPRESENTATION_ARCHIVE_GRAMMAR_BLUEPRINT (per HNeRV parity discipline):
  archive_grammar: T1 monolithic-single-x-member + transform.bin sidecar
    (forward + inverse MLPs, mixed precision FP16 train / FP4 ship)
  parser_section_manifest: x SHA-256, decoder.bin SHA-256, balle.bin SHA-256,
    transform.bin SHA-256, byte sizes per section
  inflate_runtime_loc_budget: substrate_engineering (parent T1 budget)
  runtime_dep_closure: tac runtime + torch + brotli + compressai
  export_format: phase2_t18_t1_nonlinear_transform_bolt_on
  score_aware_loss: T1 score-aware loss + NN-3 invertibility gate
  bolt_on_loc_budget: substrate_engineering (per Phase 2 pre-design)
  no_op_detector_planned: exact old/new transform.bin + balle.bin SHA proof

CLAUDE.md non-negotiables — wired through this trainer
------------------------------------------------------

- **EMA decay 0.997 with snapshot+restore** — decoder + Ballé + forward
  MLP + inverse MLP all wrapped by ``tac.training.EMA(decay=0.997)``.
- **eval_roundtrip = True** + **Differentiable YUV6** + **CUDA-required** +
  **No make_synthetic outside --smoke** + **Score-domain Lagrangian** — as T1.
- **latent_dim shrunk 192→64** per MacKay MDL critique (pre-design Q1).
- **No /tmp paths**; outputs under ``experiments/results/<lane>_<utc>/``.

NN-3 gate (NON-NEGOTIABLE; per Phase 2 pre-design):
  Every 100 training steps, sample 100 random z_e via the encoder, run
  ``forward(z_e)`` then ``invert(forward(z_e))``, assert
  ``||z_e − invert(forward(z_e))||² < 0.5``. Pause (not kill) on breach
  via a structured ``InvertibilityBreachError`` for operator inspection.
  Per-step probe is monitored but the assertion fires periodically.

HARD GATE (Probe T18-B):
  Net byte savings > 0 vs T1 baseline. If FAILS, T18 is
  DEFERRED-PENDING-PHASE-3-DISTILLATION. The trainer itself does NOT
  enforce HARD GATE (that's a separate Probe T18-B dispatch); but the
  manifest carries the predicted net-byte-savings band so the operator
  can compare against empirical anchor on dispatch.

Predicted Δ score (stand-alone on A1 substrate):
  -0.003 ± 0.002 ``[predicted; He-Zheng 2024 BD-rate; conditional on
  T18-B HARD GATE per Phase 2 pre-design pass 2026-05-09]``. NOT a score
  claim until contest-CUDA anchor lands.
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

from tac.balle_nonlinear_transform import (  # noqa: E402
    NonlinearTransformBlock,
    NonlinearTransformConfig,
    transform_state_bytes,
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


T18_LANE_ID = "lane_t18_balle_nonlinear_transform_phase2_preregistered"
T18_SCHEMA_VERSION = "0.1.0-phase2-t18-nonlinear-transform-bolt-on"
T18_PREDICTED_DELTA_SCORE = (
    "-0.003 ± 0.002 [predicted; He-Zheng 2024 BD-rate; conditional on "
    "T18-B HARD GATE per Phase 2 pre-design pass 2026-05-09]"
)
NN3_INVERTIBILITY_FLOOR_DEFAULT = 0.5
NN3_PROBE_EVERY_STEPS_DEFAULT = 100
NN3_PROBE_SAMPLE_COUNT_DEFAULT = 100


class InvertibilityBreachError(RuntimeError):
    """Raised by the NN-3 gate when forward/inverse transforms drift apart."""


# ---------------------------------------------------------------------------
# NN-3 gate
# ---------------------------------------------------------------------------


def assert_invertibility_ok(
    transform_block: torch.nn.Module,
    z_e_sample: torch.Tensor,
    floor: float = NN3_INVERTIBILITY_FLOOR_DEFAULT,
    step: int | None = None,
) -> dict:
    """NN-3: gate sustained-training invertibility.

    Per Phase 2 pre-design pass §1 Contrarian challenge (iii): test 7 in
    ``test_balle_nonlinear_transform.py`` only verifies invertibility at
    init; after 1000+ training steps the forward and inverse MLPs may
    drift. This gate samples random z_e from the encoder, runs
    ``forward → invert → check L2``, and raises if drift exceeds the
    council-chosen floor (0.5 L2).
    """
    with torch.no_grad():
        z_t = transform_block(z_e_sample)
        z_e_recovered = transform_block.invert(z_t)
        invertibility_error = float(((z_e_sample - z_e_recovered) ** 2).mean().item())
    diag = {
        "invertibility_error": invertibility_error,
        "floor": float(floor),
        "passed": invertibility_error < floor,
        "step": step,
    }
    if not diag["passed"]:
        raise InvertibilityBreachError(
            f"[t18] NN-3 invertibility breach at step {step}: "
            f"||z_e − invert(forward(z_e))||² = {invertibility_error:.4f} "
            f">= floor {floor:.4f}; halting for operator inspection. "
            "Check (a) FP16-vs-FP4 quant drift, (b) skip-connection coefficient, "
            "(c) inverse MLP gradient norm. Do not silently continue."
        )
    return diag


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="T18 — Ballé nonlinear transform trainer (T1-clone derivative)"
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
    # T18-specific
    parser.add_argument("--t18-latent-dim", type=int, default=64,
                        help="Transform latent dim (council canon = 64 per MacKay MDL)")
    parser.add_argument("--t18-expansion-factor", type=int, default=4,
                        help="FFN expansion factor (He-Zheng 2024 canon = 4)")
    parser.add_argument("--t18-num-hidden-layers", type=int, default=3,
                        help="GELU hidden layers (He-Zheng 2024 §3.2 sweet spot = 3)")
    parser.add_argument("--t18-activation", default="gelu",
                        choices=["gelu", "relu", "silu"])
    parser.add_argument("--t18-transform-quant", default="fp16",
                        choices=["fp4", "fp8", "fp16", "fp32"],
                        help="Mixed precision: FP16 train, FP4 ship (re-quantize on save)")
    parser.add_argument("--t18-ship-quant", default="fp4",
                        choices=["fp4", "fp8", "fp16", "fp32"],
                        help="Final archive quant after training (FP4 default per pre-design)")
    parser.add_argument("--t18-nn3-floor", type=float, default=NN3_INVERTIBILITY_FLOOR_DEFAULT,
                        help="NN-3 invertibility L2 floor (council canon = 0.5)")
    parser.add_argument("--t18-nn3-probe-every-steps", type=int,
                        default=NN3_PROBE_EVERY_STEPS_DEFAULT)
    parser.add_argument("--t18-nn3-probe-sample-count", type=int,
                        default=NN3_PROBE_SAMPLE_COUNT_DEFAULT)
    return parser.parse_args(argv)


def _canonical_dir_name_from_relpath(relpath: str) -> str:
    return Path(relpath).name


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.device == "mps":
        raise SystemExit("[t18] --device mps refused; see CLAUDE.md MPS-is-noise rule")
    if args.auth_eval:
        refuse_phase1_scaffold_path("--auth-eval [T18 scaffold; T1-clone]")
    if not args.smoke and PHASE1_SCAFFOLD_ONLY and not args.enable_scorer_domain_loss:
        refuse_phase1_scaffold_path("non-smoke T18 training [scaffold; T1-clone]")
    if args.t18_nn3_floor <= 0:
        raise SystemExit("[t18] --t18-nn3-floor must be > 0")
    if args.t18_nn3_probe_every_steps <= 0:
        raise SystemExit("[t18] --t18-nn3-probe-every-steps must be > 0")
    if args.t18_nn3_probe_sample_count <= 0:
        raise SystemExit("[t18] --t18-nn3-probe-sample-count must be > 0")
    if args.grad_clip_norm < 0:
        raise SystemExit("[t18] --grad-clip-norm must be non-negative")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _seed_everything(args.seed)
    device = _resolve_device(args.device)

    yuv6_mode = _resolve_yuv6_mode_with_probe_t1(args.yuv6_mode)
    yuv6_token = _activate_yuv6_mode_t1(
        yuv6_mode, enabled=args.enable_differentiable_yuv6
    )
    print(
        f"[t18] eval_roundtrip={args.enable_eval_roundtrip_in_training} "
        f"yuv6_mode={yuv6_mode.value} smoke={args.smoke} device={device}"
    )

    if args.smoke and args.allow_missing_canonical_a1:
        latents, target_pixels = make_smoke_target(
            n_pairs=4, latent_dim=args.t18_latent_dim, seed=args.seed,
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

    # T18 nonlinear transform: inserts between encoder output and entropy
    # bottleneck. Operates on the per-pair latent_dim (which must match
    # the Ballé y_hat shape). We project to a smaller t18_latent_dim if
    # the contest latent is wider.
    actual_latent_dim = int(latents.shape[1])
    if actual_latent_dim != args.t18_latent_dim:
        print(
            f"[t18] note: actual latent_dim={actual_latent_dim} differs from "
            f"--t18-latent-dim={args.t18_latent_dim}; using actual dim "
            f"(no projection MLP added; council's --t18-latent-dim guidance "
            f"applies when caller can choose encoder dim)"
        )
    t18_dim = actual_latent_dim
    t18_config = NonlinearTransformConfig(
        latent_dim=t18_dim,
        expansion_factor=args.t18_expansion_factor,
        num_hidden_layers=args.t18_num_hidden_layers,
        activation=args.t18_activation,
        quantization=args.t18_transform_quant,
        label=f"t18_t1clone_{started_at}",
    )
    transform = NonlinearTransformBlock(t18_config).to(device)
    transform_bytes_train = transform_state_bytes(t18_config)
    # Predicted ship-quant byte cost.
    ship_config = NonlinearTransformConfig(
        latent_dim=t18_config.latent_dim,
        expansion_factor=t18_config.expansion_factor,
        num_hidden_layers=t18_config.num_hidden_layers,
        activation=t18_config.activation,
        quantization=args.t18_ship_quant,
        label=t18_config.label,
    )
    transform_bytes_ship = transform_state_bytes(ship_config)

    n_decoder = sum(p.numel() for p in decoder.parameters())
    n_balle = sum(p.numel() for p in balle.parameters())
    n_transform = sum(p.numel() for p in transform.parameters())
    print(
        f"[t18] decoder={n_decoder:,} balle={n_balle:,} "
        f"transform={n_transform:,} (~{transform_bytes_train} B {args.t18_transform_quant} train, "
        f"~{transform_bytes_ship} B {args.t18_ship_quant} ship)"
    )

    decoder_trainable = [p for p in decoder.parameters() if p.requires_grad]
    balle_main_trainable = [
        p for n, p in balle.named_parameters()
        if p.requires_grad and ("entropy_bottleneck" not in n or "quantiles" not in n)
    ]
    transform_trainable = [p for p in transform.parameters() if p.requires_grad]
    main_params = [*decoder_trainable, *balle_main_trainable, *transform_trainable]
    aux_params = [p for n, p in balle.named_parameters() if "quantiles" in n]
    optim_main = torch.optim.Adam(main_params, lr=args.learning_rate)
    optim_aux = torch.optim.Adam(aux_params, lr=args.aux_learning_rate) if aux_params else None

    ema_decoder = EMA(decoder, decay=args.ema_decay)
    ema_balle = EMA(balle, decay=args.ema_decay)
    ema_transform = EMA(transform, decay=args.ema_decay)

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
    nn3_probes: list[dict] = []
    global_step = 0

    posenet = segnet = None
    if args.enable_scorer_domain_loss:
        from tac.scorer import load_differentiable_scorers  # noqa: WPS433
        posenet, segnet = load_differentiable_scorers(REPO_ROOT / "upstream", device=device)
        posenet.eval(); segnet.eval()
        for m in (posenet, segnet):
            for p in m.parameters():
                p.requires_grad_(False)

    for epoch in range(epochs):
        decoder.train(); balle.train(); transform.train()
        perm = torch.randperm(n_pairs, generator=torch.Generator().manual_seed(args.seed + epoch))
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_pairs, batch_size):
            idx = perm[start:start + batch_size]
            y = latents[idx]
            tgt = target_pixels[idx]

            # T18: insert nonlinear transform BEFORE Ballé entropy bottleneck.
            # The Ballé wrapper takes ``y`` as input and runs y → y_hat
            # through its entropy bottleneck internally; we transform the
            # input latent prior to Ballé's internal quantization. The
            # inverse transform is applied to y_hat post-Ballé.
            y_t = transform(y)
            balle_out = balle(y_t)
            # Inverse transform on y_hat (decoder side).
            y_hat_inv = transform.invert(balle_out["y_hat"])
            decoded = decoder(y_hat_inv)

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

            res = coord.step(
                distortion=distortion,
                rate_bits=balle_out["rate_total_bits"],
                seg_loss=seg_loss,
                pose_loss=pose_loss,
            )
            if not torch.isfinite(res.augmented_lagrangian).all():
                raise RuntimeError("[t18] non-finite augmented Lagrangian")

            optim_main.zero_grad()
            res.augmented_lagrangian.backward()
            if args.enable_scorer_domain_loss and global_step == 0:
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

            ema_decoder.update(decoder)
            ema_balle.update(balle)
            ema_transform.update(transform)

            # NN-3 invertibility probe every N steps.
            if global_step % args.t18_nn3_probe_every_steps == 0:
                sample_n = min(args.t18_nn3_probe_sample_count, n_pairs)
                with torch.no_grad():
                    sample_idx = torch.randperm(n_pairs)[:sample_n].to(device)
                    z_e_probe = latents[sample_idx]
                diag = assert_invertibility_ok(
                    transform, z_e_probe,
                    floor=args.t18_nn3_floor,
                    step=global_step,
                )
                nn3_probes.append(diag)

            epoch_loss += float(res.augmented_lagrangian.detach())
            n_batches += 1
            global_step += 1

        avg = epoch_loss / max(n_batches, 1)
        if epoch == 0 or (epoch + 1) % args.eval_every_epochs == 0 or epoch == epochs - 1:
            history.append({
                "epoch": epoch + 1, "avg_loss": avg, "rho": coord.rho,
                "global_step": global_step,
                "last_nn3_invertibility_error": (
                    nn3_probes[-1]["invertibility_error"] if nn3_probes else None
                ),
            })
            print(
                f"[t18] epoch {epoch+1}/{epochs} loss={avg:.4f} rho={coord.rho:.3f} "
                f"step={global_step} nn3_probes={len(nn3_probes)}"
            )

    ckpt = {
        "schema": T18_SCHEMA_VERSION,
        "ema_decoder": ema_decoder.shadow,
        "ema_balle": ema_balle.shadow,
        "ema_transform": ema_transform.shadow,
        "t18_config": {
            "latent_dim": t18_config.latent_dim,
            "expansion_factor": t18_config.expansion_factor,
            "num_hidden_layers": t18_config.num_hidden_layers,
            "activation": t18_config.activation,
            "train_quantization": t18_config.quantization,
            "ship_quantization": args.t18_ship_quant,
            "label": t18_config.label,
        },
        "transform_bytes_train": transform_bytes_train,
        "transform_bytes_ship": transform_bytes_ship,
        "predicted_delta_score": T18_PREDICTED_DELTA_SCORE,
    }
    torch.save(ckpt, args.output_dir / "t18_ema_shadow.pt")

    manifest = {
        "schema": T18_SCHEMA_VERSION,
        "lane_id": T18_LANE_ID,
        "phase": "2_scaffold_t18_t1_clone",
        "started_at_utc": started_at,
        "device": str(device),
        "smoke": bool(args.smoke),
        "epochs": int(epochs),
        "n_pairs": n_pairs,
        "decoder_params": int(n_decoder),
        "balle_params": int(n_balle),
        "transform_params": int(n_transform),
        "transform_bytes_train": int(transform_bytes_train),
        "transform_bytes_ship": int(transform_bytes_ship),
        "transform_train_quantization": t18_config.quantization,
        "transform_ship_quantization": args.t18_ship_quant,
        "predicted_delta_score": T18_PREDICTED_DELTA_SCORE,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": "[predicted; Phase 2 T18 scaffold; not yet empirical]",
        "score_aware_loss_enabled": bool(args.enable_scorer_domain_loss),
        "eval_roundtrip": bool(args.enable_eval_roundtrip_in_training),
        "ema_decay": float(args.ema_decay),
        "nn3_probes_performed": len(nn3_probes),
        "nn3_invertibility_floor": float(args.t18_nn3_floor),
        "nn3_probe_every_steps": int(args.t18_nn3_probe_every_steps),
        "history": history,
        "hard_gate_t18_b_status": (
            "NOT_RUN_BY_THIS_TRAINER — Probe T18-B is a separate dispatch "
            "that compares this trainer's archive byte count vs T1 baseline; "
            "net byte savings > 0 required for Phase 2 inclusion."
        ),
        "compliance_tags": [
            "ema_0p997_snapshot_restore",
            "eval_roundtrip_true",
            "no_mps_authoritative",
            "differentiable_yuv6",
            "score_aware_lagrangian",
            "skip_connection_identity_init",
            "separate_forward_inverse_mlps",
            "nn3_invertibility_probe_periodic",
            "mixed_precision_fp16_train_fp4_ship",
            "no_synthetic_outside_smoke",
            "no_tmp_paths",
            "auth_eval_gated",
        ],
    }
    (args.output_dir / "t18_provenance.json").write_text(json.dumps(manifest, indent=2))
    print(f"[t18] done; manifest written to {args.output_dir}/t18_provenance.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
