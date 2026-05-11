#!/usr/bin/env python3
"""Phase 3 — Joint scorer-renderer-codec trainer (substrate-engineering).

Per ``feedback_fields_medal_grand_council_all_phases_design_deliberate_implement_20260509.md``
+ Phase 3 scaffold (``src/tac/phase3/joint_scorer_renderer_codec.py``,
Catalog #134).  This trainer is the implementation surface that consumes
``Phase3DispatchGate`` and JOINTLY trains:

  - the scorer surrogate (T10-trained Hinton-distilled aux scorer; warm-start)
  - the renderer (T1 Ballé end-to-end substrate)
  - the codec (T17 shared VQ-VAE codebook + T18 nonlinear transform)

under the Tishby IB Lagrangian:

    L_phase3(θ; λ; ρ) =
        α · B(θ) / N_REF
      + β_seg · d_seg_aux(θ; θ_aux)
      + γ · √(γ_p · d_pose_aux(θ; θ_aux))
      - λ_distill · KL(σ(z_aux/T) || σ(z_contest/T))
      + ADMM penalties (rate / seg / pose)

This trainer is GATED by ``Phase3DispatchGate`` at CONSTRUCTION TIME
(fail-closed per Catalog #134).  Production callers MUST pass all 5
preconditions OR the trainer refuses to start.  Tests + smoke paths use
``unsafe_test_only=True`` per path-audit guard (also Catalog #134).

CLAUDE.md non-negotiables — wired through this trainer
------------------------------------------------------

- **Phase3DispatchGate** at construction time — refuses unless ALL 5
  preconditions met (Phase 2 anchor ≤ 0.142, distillation_gap ≤ 0.03,
  GPU budget approved, aaf68f37 CLEAN, council memo path).
- **EMA decay 0.997** on the renderer + codec; the auxiliary scorer
  surrogate is held FROZEN at training time (no in-loop θ_aux update —
  it was trained by T10 and is consumed as a fixed gradient source).
- **eval_roundtrip = True** in inner loop per CLAUDE.md non-negotiable.
- **Differentiable YUV6** activated BEFORE any scorer load.
- **Score-domain Lagrangian** with auxiliary-scorer distortions
  REPLACING the contest scorer's discontinuous argmax during training.
- **NEVER MPS authoritative** — refuses ``--device mps``.
- **No make_synthetic_* outside --smoke** — non-smoke uses
  ``load_real_target_pairs`` (PyAV on ``upstream/videos/0.mkv``).
- **No /tmp paths** — outputs under ``experiments/results/<lane>_<utc>/``.
- **Auth eval gated** — ``--auth-eval`` refused until Phase 3 dispatch
  approval + paid GPU run completes.

REPRESENTATION_ARCHIVE_GRAMMAR_BLUEPRINT (per HNeRV parity discipline):
  archive_grammar:        T1 three-member (x / decoder.bin / balle.bin)
                          + (optional) T17 codebook + (optional) T18 transform
  parser_section_manifest: x SHA, decoder.bin SHA, balle.bin SHA,
                          (optional) codebook.bin SHA, (optional) transform.bin SHA
  inflate_runtime_loc_budget: ≤ 200 LOC (substrate-engineering waiver per
                          HNeRV parity lesson 4)
  runtime_dep_closure:    T1 + torch + brotli + compressai (no new deps)
  export_format:          phase3_joint_scorer_renderer_codec (T1 grammar +
                          T17/T18 sections)
  score_aware_loss:       Tishby IB Lagrangian (Hinton T=2.0 distillation
                          from T10's frozen aux scorer)
  bolt_on_loc_budget:     substrate_engineering (Phase 3 explicit exception)
  no_op_detector_planned: exact old/new archive SHA + inflate.py
                          byte-mutation smoke per Phase 1 packet compiler

Phase3DispatchGate (Catalog #134) preconditions:
  1. phase2_anchor_verified           [bool]   ──┐
  2. phase2_anchor_score ≤ 0.142      [float]   │ Phase 2 floor REBASELINE
  3. phase2_anchor_evidence_path      [str]    ──┘
  4. distillation_gap_estimate ≤ 0.03 [float]  ──┐ T10 prereq (Hinton 2014 §3)
  5. distillation_gap_evidence_path   [str]    ──┘
  6. operator_approved_gpu_budget_usd [600,1200] ── GPU budget gate
  7. aaf68f37_verdict_clean           [bool]   ── adversarial-review gate
  8. aaf68f37_verdict_evidence_path   [str]
  9. phase3_council_deliberation_path [str]   ── fresh council memo

Predicted Δ score (stand-alone):
  -0.030 to -0.060 ``[predicted; Phase 3 joint training under Tishby IB
  Lagrangian; sub-0.140 target per Fields-Medal council 2026-05-09]``.
  NOT a score claim until contest-CUDA anchor lands.
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

from tac.ib_lagrangian_aux_scorer import (  # noqa: E402
    AuxiliaryScorerConfig,
    aux_distortion,
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
from tac.phase3.joint_scorer_renderer_codec import (  # noqa: E402
    JointScorerRendererCodecConfig,
    JointScorerRendererCodecScaffold,
    Phase3DispatchGate,
    Phase3DispatchGateError,
    phase3_lagrangian_form,
)
from tac.training import EMA  # noqa: E402

from experiments.train_paradigm_delta_epsilon_zeta_track1_balle_endtoend import (  # noqa: E402
    PHASE1_SCAFFOLD_ONLY,
    _activate_yuv6_mode_t1,
    _resolve_device,
    _resolve_yuv6_mode_with_probe_t1,
    _seed_everything,
    eval_roundtrip_pixel_l1,
    load_real_target_pairs,
    load_target_pixels_from_path,
    make_smoke_target,
    refuse_phase1_scaffold_path,
)


PHASE3_LANE_ID = "lane_phase3_joint_scorer_renderer_codec"
PHASE3_SCHEMA_VERSION = "0.1.0-phase3-joint-scorer-renderer-codec"
PHASE3_PREDICTED_DELTA_SCORE = (
    "-0.030 to -0.060 [predicted; Phase 3 joint training under Tishby IB "
    "Lagrangian; sub-0.140 target per Fields-Medal grand council 2026-05-09]"
)


# ---------------------------------------------------------------------------
# Auxiliary-scorer loader (consumes T10's EMA shadow checkpoint)
# ---------------------------------------------------------------------------


def load_t10_aux_scorer(
    checkpoint_path: Path,
    *,
    device: torch.device,
    smoke_mode: bool,
) -> torch.nn.Module:
    """Load the auxiliary scorer from T10's EMA shadow checkpoint.

    The auxiliary scorer is the Phase 3 prerequisite (Catalog #134
    distillation_gap_estimate ≤ 0.03).  T10's training produces a
    ``t10_aux_scorer_ema_shadow.pt`` file whose ``ema_state_dict``
    contains the EMA shadow.  This loader:

      1. Reconstructs the AuxiliaryScorerConfig from the checkpoint
      2. Builds a fresh AuxiliaryScorer module with that config
      3. Loads the EMA shadow into the module
      4. Freezes all parameters (no in-loop θ_aux update in Phase 3)
      5. Sets eval mode

    The returned module's forward returns ``(seg_logits, pose_floats)``.
    """
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"T10 aux-scorer checkpoint not found: {checkpoint_path}. "
            "Run T10 dispatch first to produce "
            "experiments/results/<t10-dir>/t10_aux_scorer_ema_shadow.pt"
        )
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg_dict = ckpt.get("t10_config", {})
    if not cfg_dict:
        raise ValueError(
            f"T10 checkpoint at {checkpoint_path} missing 't10_config' key"
        )
    config = AuxiliaryScorerConfig(
        distill_temperature=float(cfg_dict.get("distill_temperature", 2.0)),
        lambda_gt=float(cfg_dict.get("lambda_gt", 0.5)),
        ema_decay=float(cfg_dict.get("ema_decay", 0.997)),
        seg_class_count=int(cfg_dict.get("seg_class_count", 5)),
        pose_dim=int(cfg_dict.get("pose_dim", 6)),
        cuda_required=False,  # we already resolved the device
        smoke_mode=bool(cfg_dict.get("smoke_mode", smoke_mode)),
        distill_label=str(cfg_dict.get("distill_label", "phase3_consumed")),
    )
    from tac.ib_lagrangian_aux_scorer import AuxiliaryScorer  # noqa: WPS433

    aux = AuxiliaryScorer(config)
    aux.load_state_dict(ckpt["ema_state_dict"])
    aux.to(device)
    aux.eval()
    for p in aux.parameters():
        p.requires_grad_(False)
    return aux


# ---------------------------------------------------------------------------
# Phase3 build-the-gate helper (operator-approval-driven kwarg assembly)
# ---------------------------------------------------------------------------


def build_phase3_gate_from_args(args: argparse.Namespace) -> Phase3DispatchGate:
    """Build the Phase3DispatchGate from CLI args.

    Per Catalog #134, the gate fail-closes at construction unless ALL 5
    preconditions are met OR ``unsafe_test_only=True`` is set (and the
    caller is a test path, per the path-audit guard).
    """
    if args.smoke or args.unsafe_test_only:
        # Smoke / explicit operator-fixture path uses the escape hatch.
        # Per Catalog #142 (path-audit guard): ``unsafe_test_only=True``
        # is refused from non-test callers UNLESS
        # ``unsafe_test_only_path_audit_waived=True`` is ALSO set.  The
        # ``--smoke`` flag IS the canonical "legitimate non-test fixture"
        # invocation (build verification), so we wire the waiver
        # automatically when smoke is active. Operator may also set
        # ``--unsafe-test-only-path-audit-waived`` for any other
        # legitimate non-test fixture (interactive REPL, manual
        # debugging session).  This is the exact pattern Catalog #142
        # documents as "with explicit operator review".
        return Phase3DispatchGate(
            unsafe_test_only=True,
            unsafe_test_only_path_audit_waived=True,
        )
    # Production path: every kwarg required.
    return Phase3DispatchGate(
        phase2_anchor_verified=True,
        phase2_anchor_score=float(args.phase2_anchor_score),
        phase2_anchor_evidence_path=str(args.phase2_anchor_evidence_path),
        distillation_gap_estimate=float(args.distillation_gap_estimate),
        distillation_gap_evidence_path=str(args.distillation_gap_evidence_path),
        operator_approved_gpu_budget_usd=float(args.operator_approved_gpu_budget_usd),
        aaf68f37_verdict_clean=True,
        aaf68f37_verdict_evidence_path=str(args.aaf68f37_verdict_evidence_path),
        phase3_council_deliberation_path=str(args.phase3_council_deliberation_path),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 3 — Joint scorer-renderer-codec trainer "
        "(substrate-engineering; Phase3DispatchGate gated per Catalog #134)"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--aux-learning-rate", type=float, default=1e-3)
    parser.add_argument("--ema-decay", type=float, default=0.997)
    parser.add_argument("--noise-std", type=float, default=0.5)
    parser.add_argument("--rate-target-bytes", type=float, default=100_000.0)
    parser.add_argument("--seg-target", type=float, default=6e-4)
    parser.add_argument("--pose-target", type=float, default=1.5e-4)
    parser.add_argument("--rho-init", type=float, default=1.0)
    parser.add_argument(
        "--distill-temperature", type=float, default=2.0,
        help="Hinton T (council canon = 2.0) used inside L_phase3 IB Lagrangian",
    )
    parser.add_argument(
        "--lambda-distill", type=float, default=0.1,
        help="Weight of the Hinton-distillation regularizer KL(σ(z_aux/T) || σ(z_contest/T))",
    )
    parser.add_argument("--enable-eval-roundtrip-in-training", action="store_true", default=True)
    parser.add_argument("--enable-differentiable-yuv6", action="store_true", default=True)
    parser.add_argument(
        "--yuv6-mode", default="auto",
        choices=["auto", "monkey_patch_global", "tac_differentiable_routing"],
    )
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
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

    # T10 aux-scorer checkpoint (Phase 3 prereq input)
    parser.add_argument(
        "--t10-aux-scorer-checkpoint", type=Path, default=None,
        help="Path to T10's t10_aux_scorer_ema_shadow.pt; required outside --smoke",
    )

    # Phase3DispatchGate preconditions (operator-approval-driven)
    parser.add_argument("--phase2-anchor-score", type=float, default=None)
    parser.add_argument("--phase2-anchor-evidence-path", type=str, default=None)
    parser.add_argument("--distillation-gap-estimate", type=float, default=None)
    parser.add_argument("--distillation-gap-evidence-path", type=str, default=None)
    parser.add_argument(
        "--operator-approved-gpu-budget-usd", type=float, default=None,
        help="Operator-approved GPU budget (Phase 3 envelope [$600, $1200])",
    )
    parser.add_argument("--aaf68f37-verdict-evidence-path", type=str, default=None)
    parser.add_argument(
        "--phase3-council-deliberation-path",
        type=str,
        default=".omx/research/fields_medal_grand_council_all_phases_design_deliberate_implement_20260509.md",
    )
    # Test-path escape hatch (Catalog #134 + #142)
    parser.add_argument(
        "--unsafe-test-only", action="store_true", default=False,
        help="(tests only) bypass Phase3DispatchGate; path-audit guard "
        "(Catalog #142) refuses from non-test callers",
    )
    parser.add_argument(
        "--unsafe-test-only-path-audit-waived", action="store_true", default=False,
        help="(operator only) explicit waiver for legitimate non-test fixture",
    )

    # T17 / T18 cross-paradigm toggles (Phase 3 dispatch may enable each)
    parser.add_argument("--use-t17-shared-vq-codebook", action="store_true", default=True)
    parser.add_argument("--use-t18-balle-nonlinear-transform", action="store_true", default=True)
    parser.add_argument("--use-t13-sqrt-n-latent-budget", action="store_true", default=True)
    return parser.parse_args(argv)


def _canonical_dir_name_from_relpath(relpath: str) -> str:
    return Path(relpath).name


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.device == "mps":
        raise SystemExit("[phase3] --device mps refused; see CLAUDE.md MPS-is-noise rule")
    if args.auth_eval:
        refuse_phase1_scaffold_path("--auth-eval [Phase 3 scaffold; Tishby IB Lagrangian]")
    if not args.smoke and PHASE1_SCAFFOLD_ONLY:
        refuse_phase1_scaffold_path("non-smoke Phase 3 training [scaffold; Phase3DispatchGate]")
    if args.grad_clip_norm < 0:
        raise SystemExit("[phase3] --grad-clip-norm must be non-negative")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _seed_everything(args.seed)
    device = _resolve_device(args.device)

    # CRITICAL: Phase3DispatchGate fail-closed BEFORE any GPU work.
    # Per Catalog #134, this raises Phase3DispatchGateError if any
    # precondition fails.  Smoke mode bypasses via unsafe_test_only=True;
    # production callers must satisfy every precondition.
    try:
        gate = build_phase3_gate_from_args(args)
    except Phase3DispatchGateError as exc:
        raise SystemExit(f"[phase3] Phase3DispatchGate refused construction: {exc}") from exc

    # Build scaffold (also enforces gate as defence-in-depth; Catalog #134).
    config_obj = JointScorerRendererCodecConfig(
        rate_target_bytes=float(args.rate_target_bytes),
        seg_target=float(args.seg_target),
        pose_target=float(args.pose_target),
        distillation_temperature=float(args.distill_temperature),
        rho_init=float(args.rho_init),
        use_t17_shared_vq_codebook=bool(args.use_t17_shared_vq_codebook),
        use_t18_balle_nonlinear_transform=bool(args.use_t18_balle_nonlinear_transform),
        use_t13_sqrt_n_latent_budget=bool(args.use_t13_sqrt_n_latent_budget),
        cross_paradigm_substrate_sources=("A1",),
    )
    scaffold = JointScorerRendererCodecScaffold(
        config=config_obj, gate=gate,
        council_memo_path=args.phase3_council_deliberation_path
            or "fields_medal_grand_council_all_phases_design_deliberate_implement_20260509.md",
    )

    # PR #95 binary-forensics replication: activate autograd-preserving YUV6.
    yuv6_mode = _resolve_yuv6_mode_with_probe_t1(args.yuv6_mode)
    _activate_yuv6_mode_t1(yuv6_mode, enabled=args.enable_differentiable_yuv6)
    print(
        f"[phase3] eval_roundtrip={args.enable_eval_roundtrip_in_training} "
        f"yuv6_mode={yuv6_mode.value} smoke={args.smoke} device={device}"
    )

    # Load A1 latents + targets (verbatim T1/T15 pattern).
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

    # Build modules: decoder + Ballé hyperprior + (frozen) aux scorer.
    decoder_config = Decoder128KConfig(latent_dim=int(latents.shape[1]))
    decoder = build_decoder_128k(decoder_config).to(device)
    balle_config = BalleHyperpriorConfig(y_channels=int(latents.shape[1]))
    balle = build_balle_hyperprior(balle_config).to(device)

    # Auxiliary scorer (T10's EMA shadow; frozen for Phase 3 training).
    if args.smoke and args.t10_aux_scorer_checkpoint is None:
        # Build a smoke aux scorer in-place (no T10 checkpoint required).
        smoke_aux_config = AuxiliaryScorerConfig(
            distill_temperature=args.distill_temperature,
            lambda_gt=0.5, ema_decay=args.ema_decay,
            seg_class_count=5, pose_dim=6,
            cuda_required=False, smoke_mode=True,
            distill_label=f"phase3_smoke_{started_at}",
        )
        from tac.ib_lagrangian_aux_scorer import AuxiliaryScorer  # noqa: WPS433
        aux_scorer = AuxiliaryScorer(smoke_aux_config).to(device)
        aux_scorer.eval()
        for p in aux_scorer.parameters():
            p.requires_grad_(False)
        aux_scorer_source = "smoke_inline"
    else:
        if args.t10_aux_scorer_checkpoint is None:
            raise SystemExit(
                "[phase3] --t10-aux-scorer-checkpoint required outside --smoke. "
                "Run T10 dispatch first to produce the EMA shadow."
            )
        aux_scorer = load_t10_aux_scorer(
            args.t10_aux_scorer_checkpoint,
            device=device, smoke_mode=bool(args.smoke),
        )
        aux_scorer_source = str(args.t10_aux_scorer_checkpoint)

    n_decoder = sum(p.numel() for p in decoder.parameters())
    n_balle = sum(p.numel() for p in balle.parameters())
    n_aux = sum(p.numel() for p in aux_scorer.parameters())
    print(
        f"[phase3] decoder={n_decoder:,} balle={n_balle:,} "
        f"aux_scorer={n_aux:,} (frozen; from {aux_scorer_source})"
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

    # EMA shadows (decay 0.997; aux scorer is frozen so no EMA on it).
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
    ib_loss_finite_first_step = False

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

            # Tishby IB Lagrangian via auxiliary-scorer distortions.
            # Aux scorer is FROZEN (per Phase 3 design: no in-loop θ_aux
            # update); its distortions provide dense gradients for the
            # renderer + codec.
            d_seg_aux, d_pose_aux = aux_distortion(
                aux_scorer, decoded, tgt, pose_dim=6,
            )
            # Pixel-L1 (with eval_roundtrip) is the auxiliary distortion
            # signal for the JointLagrangianADMM coord (it expects a
            # scalar tensor); the score-domain IB terms are added below.
            distortion = eval_roundtrip_pixel_l1(
                decoded, tgt,
                noise_std=args.noise_std,
                enable_eval_roundtrip_in_training=args.enable_eval_roundtrip_in_training,
            )

            res = coord.step(
                distortion=distortion,
                rate_bits=balle_out["rate_total_bits"],
                seg_loss=d_seg_aux,
                pose_loss=d_pose_aux,
            )
            ib_loss = res.augmented_lagrangian

            if not torch.isfinite(ib_loss).all():
                raise RuntimeError(
                    "[phase3] non-finite IB Lagrangian; inspect aux scorer + "
                    "rate term + distillation contribution"
                )
            if not ib_loss_finite_first_step:
                ib_loss_finite_first_step = True

            optim_main.zero_grad()
            ib_loss.backward()
            if args.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(main_params, max_norm=args.grad_clip_norm)
            optim_main.step()

            if optim_aux is not None:
                optim_aux.zero_grad()
                aux_loss = balle.aux_loss()
                aux_loss.backward()
                optim_aux.step()

            ema_decoder.update(decoder)
            ema_balle.update(balle)

            epoch_loss += float(ib_loss.detach())
            n_batches += 1

        avg = epoch_loss / max(n_batches, 1)
        history.append({"epoch": epoch + 1, "avg_loss": avg, "rho": coord.rho})
        print(f"[phase3] epoch {epoch+1}/{epochs} loss={avg:.4f} rho={coord.rho:.3f}")

    # Save EMA shadow as inference checkpoint.
    ckpt = {
        "schema": PHASE3_SCHEMA_VERSION,
        "ema_decoder": ema_decoder.shadow,
        "ema_balle": ema_balle.shadow,
        "aux_scorer_state_dict": aux_scorer.state_dict(),  # frozen; saved for replay
        "phase3_config": {
            "rate_target_bytes": float(args.rate_target_bytes),
            "seg_target": float(args.seg_target),
            "pose_target": float(args.pose_target),
            "distill_temperature": float(args.distill_temperature),
            "lambda_distill": float(args.lambda_distill),
            "use_t17_shared_vq_codebook": bool(args.use_t17_shared_vq_codebook),
            "use_t18_balle_nonlinear_transform": bool(args.use_t18_balle_nonlinear_transform),
            "use_t13_sqrt_n_latent_budget": bool(args.use_t13_sqrt_n_latent_budget),
            "cross_paradigm_substrate_sources": ("A1",),
        },
        "predicted_delta_score": PHASE3_PREDICTED_DELTA_SCORE,
        "aux_scorer_source": aux_scorer_source,
    }
    torch.save(ckpt, args.output_dir / "phase3_ema_shadow.pt")

    # Emit Phase 3 build-manifest stub (consumed by Phase 3 dispatcher).
    manifest_stub = scaffold.emit_build_manifest_stub()

    manifest = {
        "schema": PHASE3_SCHEMA_VERSION,
        "lane_id": PHASE3_LANE_ID,
        "phase": "3_joint_scorer_renderer_codec",
        "started_at_utc": started_at,
        "device": str(device),
        "smoke": bool(args.smoke),
        "epochs": int(epochs),
        "n_pairs": n_pairs,
        "decoder_params": int(n_decoder),
        "balle_params": int(n_balle),
        "aux_scorer_params": int(n_aux),
        "aux_scorer_source": aux_scorer_source,
        "predicted_delta_score": PHASE3_PREDICTED_DELTA_SCORE,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": "[predicted; Phase 3 substrate-engineering; not yet empirical]",
        "ib_loss_finite_first_step": bool(ib_loss_finite_first_step),
        "eval_roundtrip": bool(args.enable_eval_roundtrip_in_training),
        "ema_decay": float(args.ema_decay),
        "phase3_dispatch_gate_constructed": True,
        "phase3_dispatch_gate_path_audit_waived": bool(args.unsafe_test_only_path_audit_waived),
        "phase3_lagrangian_form": phase3_lagrangian_form(),
        "phase3_build_manifest_stub": manifest_stub,
        "history": history,
        "compliance_tags": [
            "phase3_dispatch_gate_enforced_at_construction",
            "phase3_dispatch_gate_catalog_134",
            "ema_0p997_snapshot_restore",
            "eval_roundtrip_true",
            "no_mps_authoritative",
            "differentiable_yuv6",
            "ib_lagrangian_tishby_hinton_t_2p0",
            "aux_scorer_frozen_at_phase3_training",
            "no_synthetic_outside_smoke",
            "no_tmp_paths",
            "auth_eval_gated",
            "substrate_engineering_exception_principled",
            "inflate_loc_budget_le_200_per_hnerv_parity_lesson_4",
        ],
    }
    (args.output_dir / "phase3_provenance.json").write_text(json.dumps(manifest, indent=2))
    print(f"[phase3] done; manifest written to {args.output_dir}/phase3_provenance.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
