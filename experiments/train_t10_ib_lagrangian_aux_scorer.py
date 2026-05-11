#!/usr/bin/env python3
"""T10 — IB-Lagrangian co-trained auxiliary scorer trainer (Phase 3 prereq).

Phase 2 pre-design pass (2026-05-09) consensus + readiness pre-stage
(2026-05-11) handoff: T10 trains a Hinton-distilled auxiliary scorer
(EfficientNet-B2 + FastViT-T12 mimic) on the Tishby Information
Bottleneck Lagrangian:

    L_IB = I(X; Z) - β · I(Z; Y)
        = - log p_Z(Z) - β · log p_{Y|Z}(Y | Z)

With deterministic Z = encode(X), I(X; Z) is bounded by Berger 1971 §4.5
and substituted in the Phase 2 score-domain Lagrangian.  Hinton
distillation (T=2.0) collapses the auxiliary scorer's argmax/discontinuity
onto the frozen contest SegNet/PoseNet via KL on T=2.0-softened logits.

The output is NOT an archive: T10 produces a TRAINED auxiliary scorer
(EMA shadow state-dict) PLUS a measured ``distillation_gap_estimate.json``
artifact.  The distillation gap is the Phase 3 prerequisite per Catalog
#134 (``Phase3DispatchGate``): Phase 3 dispatch refuses unless
``distillation_gap_estimate ≤ 0.03``.

REPRESENTATION_ARCHIVE_GRAMMAR_BLUEPRINT (per HNeRV parity discipline):
  archive_grammar:        N/A — T10 is a Phase 3 prereq, NOT a contest
                          archive.  No bytes ship.
  parser_section_manifest: N/A (no archive)
  inflate_runtime_loc_budget: N/A (no runtime)
  runtime_dep_closure:    torch + numpy (auxiliary scorer is TRAINING-ONLY)
  export_format:          phase3_aux_scorer_ema_shadow (state-dict pickle;
                          consumed by Phase 3 trainer; never shipped)
  score_aware_loss:       Hinton T=2.0 distillation + L_GT cross-entropy
  bolt_on_loc_budget:     substrate_engineering (Phase 3 prereq)
  no_op_detector_planned: distillation_gap_estimate <= 0.03 measured-and-recorded

Per CLAUDE.md "research_only=true" exemption (HNeRV parity discipline
lesson 2): T10 produces a TRAINING-ONLY artifact (no contest archive
bytes); this is correctly marked ``score_claim=False`` and
``promotion_eligible=False``.  Phase 3 consumes T10's EMA shadow as
its auxiliary-scorer warm-start, not the inverse.

CLAUDE.md non-negotiables — wired through this trainer
------------------------------------------------------

- **EMA decay 0.997** — per ``AuxiliaryScorerConfig.ema_decay``; the
  ``train_aux_scorer`` helper updates AFTER every ``optimizer.step()``
  per CLAUDE.md "EMA — NON-NEGOTIABLE".  Inference uses the EMA shadow.
- **CUDA REQUIRED** — refuses to run unless ``torch.cuda.is_available()``
  or ``--smoke`` is explicit.
- **eval_roundtrip = True** — for the aux-scorer, eval-roundtrip means
  the distillation targets are computed via the SAME frame pipeline the
  scorer will see at eval time (uint8 bottleneck simulated).
- **NEVER MPS authoritative** — refuses ``--device mps`` (the auxiliary
  scorer's CUDA-vs-MPS drift is even larger than the contest scorer's
  because the distillation loss is on raw logits before argmax).
- **No make_synthetic_* outside --smoke** — non-smoke produces a dataloader
  reading ``upstream/videos/0.mkv`` + the contest scorer's forward pass
  as the distillation target.
- **No /tmp paths** — outputs under ``experiments/results/<lane>_<utc>/``.
- **Distillation gap measured at end** — written to
  ``distillation_gap_estimate.json`` which gates Phase 3 per Catalog #134.

NN-T10 gate (NON-NEGOTIABLE):
  Final ``distillation_gap_estimate`` MUST be measured + recorded.
  Phase 3 dispatch refuses if not ≤ 0.03 per Hinton 2014 §3 + Catalog
  #134.  Smoke runs may have gap >> 0.03 (that's intentional — smoke is
  for build verification, not for unblocking Phase 3); the artifact still
  carries the measurement so the gap can be inspected.

Predicted Δ score (stand-alone, A1 substrate):
  N/A — T10 is a Phase 3 prerequisite; its direct effect is to ENABLE
  Phase 3 substrate engineering, not to lower the score on its own.
  Per pre-stage G manifest: cost band ``$40`` for the dispatch.
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
    AuxiliaryScorerError,
    AuxScorerTrainingResult,
    train_aux_scorer,
)


T10_LANE_ID = "lane_t10_ib_lagrangian_aux_scorer_phase2_preregistered"
T10_SCHEMA_VERSION = "0.1.0-phase2-t10-ib-lagrangian-aux-scorer"
T10_PREDICTED_DELTA_SCORE = (
    "N/A — Phase 3 prerequisite (gates Phase3DispatchGate "
    "distillation_gap_estimate ≤ 0.03 per Catalog #134); no direct "
    "score Δ on its own [pre-stage G manifest 2026-05-11]"
)
PHASE3_DISTILL_GAP_THRESHOLD = 0.03


# ---------------------------------------------------------------------------
# Smoke-mode dataloader (synthetic; gated to --smoke only)
# ---------------------------------------------------------------------------


class _SmokeDataloader:
    """One-batch synthetic dataloader for build verification ONLY.

    Yields ``(frames, gt_seg, gt_pose)`` tuples shaped to match the
    contest scorer's contract:

      frames: (B, T, 3, H, W) float in [0, 255]
      gt_seg: (B, H, W) long in [0, seg_class_count)
      gt_pose: (B, pose_dim) float

    Per CLAUDE.md "Forbidden ``make_synthetic_*`` outside smoke", this
    class is reachable ONLY behind ``--smoke``.
    """

    def __init__(
        self,
        *,
        n_batches: int,
        batch_size: int,
        seg_class_count: int,
        pose_dim: int,
        seed: int,
        height: int = 32,
        width: int = 48,
    ) -> None:
        if n_batches < 1:
            raise ValueError(f"n_batches must be >= 1; got {n_batches}")
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1; got {batch_size}")
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.seg_class_count = seg_class_count
        self.pose_dim = pose_dim
        self.gen = torch.Generator().manual_seed(seed)
        self.height = height
        self.width = width

    def __iter__(self):
        for _ in range(self.n_batches):
            frames = (
                torch.rand(
                    (self.batch_size, 2, 3, self.height, self.width),
                    generator=self.gen,
                )
                * 255.0
            )
            gt_seg = torch.randint(
                0,
                self.seg_class_count,
                (self.batch_size, self.height, self.width),
                generator=self.gen,
                dtype=torch.long,
            )
            gt_pose = torch.randn(
                (self.batch_size, self.pose_dim),
                generator=self.gen,
            )
            yield frames, gt_seg, gt_pose


def _make_smoke_contest_scorer(seg_class_count: int, pose_dim: int):
    """Return a deterministic ``frames -> (seg_logits, pose_floats)`` callable.

    Used only in ``--smoke`` mode where we don't have torch+CUDA loaded
    contest scorers.  The forward function returns shape-correct random
    outputs so the distillation KL is well-defined.  Real Phase 2
    dispatch wires this to the actual contest SegNet + PoseNet forward
    pass via ``tac.scorer.load_differentiable_scorers``.
    """

    def _fwd(frames: torch.Tensor):
        # frames: (B, T, 3, H, W)
        if frames.dim() != 5:
            raise ValueError(
                "smoke contest_scorer_forward: expected (B, T, 3, H, W); "
                f"got shape {tuple(frames.shape)}"
            )
        B, _, _, H, W = frames.shape
        # Deterministic-but-non-trivial seed from input statistics.
        seed = int((frames.detach().sum() * 1e3).abs().clamp(max=1e9).item())
        g = torch.Generator(device=frames.device).manual_seed(seed % (2**31 - 1))
        seg = torch.randn(
            (B, seg_class_count, H, W), generator=g, device=frames.device,
        )
        pose = torch.randn(
            (B, pose_dim), generator=g, device=frames.device,
        )
        return seg, pose

    return _fwd


def _real_contest_scorer_forward(upstream_dir: Path, device: torch.device):
    """Return a frozen contest-scorer forward callable on the given device.

    Loads ``tac.scorer.load_differentiable_scorers`` and returns a
    function that runs SegNet + PoseNet jointly under ``no_grad``.
    """
    from tac.scorer import load_differentiable_scorers  # noqa: WPS433

    posenet, segnet = load_differentiable_scorers(upstream_dir, device=device)
    posenet.eval()
    segnet.eval()
    for m in (posenet, segnet):
        for p in m.parameters():
            p.requires_grad_(False)

    def _fwd(frames: torch.Tensor):
        with torch.no_grad():
            seg_logits = segnet(frames)
            pose_floats = posenet(frames)
        return seg_logits, pose_floats

    return _fwd


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="T10 — IB-Lagrangian co-trained aux-scorer trainer "
        "(Phase 3 prerequisite)"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--distill-temperature", type=float, default=2.0,
        help="Hinton T (council canon = 2.0)",
    )
    parser.add_argument(
        "--lambda-gt", type=float, default=0.5,
        help="Weight of GT supervised term in L_aux (council canon = 0.5)",
    )
    parser.add_argument(
        "--ema-decay", type=float, default=0.997,
        help="EMA decay for aux scorer weights (CLAUDE.md non-negotiable = 0.997)",
    )
    parser.add_argument(
        "--seg-class-count", type=int, default=5,
        help="Contest SegNet classes (canonical = 5)",
    )
    parser.add_argument(
        "--pose-dim", type=int, default=6,
        help="Contest PoseNet score-relevant dims (canonical = 6)",
    )
    parser.add_argument(
        "--smoke", action="store_true", default=False,
        help="Build verification only; uses a tiny smoke aux scorer + "
        "synthetic dataloader",
    )
    parser.add_argument("--n-batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20)
    parser.add_argument(
        "--upstream-dir", type=Path, default=REPO_ROOT / "upstream",
        help="Path to contest upstream tree (for non-smoke real-scorer forward)",
    )
    parser.add_argument(
        "--auth-eval", action="store_true", default=False,
        help="Refused for T10 (no auth eval applies; T10's deliverable is "
        "distillation_gap_estimate, not a score)",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.device == "mps":
        raise SystemExit(
            "[t10] --device mps refused; see CLAUDE.md MPS-is-noise rule "
            "(aux-scorer drift on MPS is unbounded; eval results are noise)"
        )
    if args.auth_eval:
        raise SystemExit(
            "[t10] --auth-eval refused: T10 has no contest-archive deliverable. "
            "Its output is `distillation_gap_estimate.json` consumed by "
            "Phase3DispatchGate (Catalog #134), not a score-bearing archive."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    # Build the AuxiliaryScorerConfig.
    config = AuxiliaryScorerConfig(
        distill_temperature=args.distill_temperature,
        lambda_gt=args.lambda_gt,
        ema_decay=args.ema_decay,
        seg_class_count=args.seg_class_count,
        pose_dim=args.pose_dim,
        cuda_required=(not args.smoke) and (args.device == "cuda"),
        smoke_mode=bool(args.smoke),
        distill_label=f"t10_t1clone_{started_at}",
    )

    if args.smoke:
        # Smoke: synthetic dataloader + deterministic-but-noisy contest
        # scorer mimic.  Refuses to claim T10 dispatch readiness.
        dataloader = _SmokeDataloader(
            n_batches=args.n_batches,
            batch_size=max(1, args.batch_size),
            seg_class_count=args.seg_class_count,
            pose_dim=args.pose_dim,
            seed=args.seed,
        )
        contest_scorer_forward = _make_smoke_contest_scorer(
            seg_class_count=args.seg_class_count,
            pose_dim=args.pose_dim,
        )
        device = torch.device("cpu" if args.device == "cpu" else "cpu")
        # In smoke mode we always use CPU because the auxiliary scorer
        # smoke build is tiny (~3K params; faster on CPU than initializing
        # CUDA context).
        print(
            f"[t10] SMOKE: synthetic dataloader, smoke aux scorer, "
            f"device=cpu, distill_T={config.distill_temperature}, "
            f"λ_GT={config.lambda_gt}, EMA={config.ema_decay}"
        )
    else:
        # Real dispatch: requires CUDA + frozen contest scorers + a real
        # frame dataloader (operator-gated; cost band $40 per dispatch
        # script).  This branch is reached only with explicit
        # --device cuda AND --no-smoke; the trainer raises if cuda is
        # not available.
        if args.device == "cuda" and not torch.cuda.is_available():
            raise SystemExit(
                "[t10] --device cuda requested but cuda not available; "
                "use --smoke for local build verification"
            )
        device = torch.device(args.device)
        # Real dataloader is operator-wired (deferred until Phase 2 dispatch
        # lands).  At this scaffold landing, we surface a structured
        # refusal that names the missing input.
        raise SystemExit(
            "[t10] non-smoke dispatch path is operator-gated. The real-frame "
            "dataloader + contest-scorer forward must be wired by the "
            "dispatch script (scripts/staged_phase2_t10_ib_lagrangian_dispatch.sh) "
            "with explicit STAGED_PHASE_DISPATCH_OPERATOR_APPROVED=1 + "
            "STAGED_PHASE_DISPATCH_T10_TRAINER_BUILT=1 + cost band $40 "
            "operator approval.  See pre-stage G manifest "
            "feedback_phase2_phase3_dispatch_readiness_prestage_landed_20260511.md."
        )

    # Train aux scorer via Hinton distillation + L_GT.
    try:
        result: AuxScorerTrainingResult = train_aux_scorer(
            config=config,
            contest_scorer_forward=contest_scorer_forward,
            gt_dataloader=dataloader,
            n_epochs=args.epochs,
            lr=args.learning_rate,
            device=device,
        )
    except AuxiliaryScorerError as exc:
        raise SystemExit(f"[t10] aux-scorer training failed: {exc}") from exc

    # Save EMA shadow as inference checkpoint.
    ckpt = {
        "schema": T10_SCHEMA_VERSION,
        "ema_state_dict": result.ema_state_dict,
        "t10_config": {
            "distill_temperature": float(config.distill_temperature),
            "lambda_gt": float(config.lambda_gt),
            "ema_decay": float(config.ema_decay),
            "seg_class_count": int(config.seg_class_count),
            "pose_dim": int(config.pose_dim),
            "smoke_mode": bool(config.smoke_mode),
            "distill_label": config.distill_label,
        },
        "predicted_delta_score": T10_PREDICTED_DELTA_SCORE,
    }
    torch.save(ckpt, args.output_dir / "t10_aux_scorer_ema_shadow.pt")

    # NN-T10 gate artifact: distillation_gap_estimate.json (Phase 3 prerequisite
    # per Catalog #134).  Smoke runs may produce a high gap (intentional —
    # smoke is build verification, not Phase 3 unblock).  Phase 3 dispatch
    # gate refuses unless gap ≤ 0.03 per Hinton 2014 §3.
    gap_artifact = {
        "schema": T10_SCHEMA_VERSION,
        "lane_id": T10_LANE_ID,
        "distillation_gap_estimate": float(result.distillation_gap_estimate),
        "phase3_threshold": PHASE3_DISTILL_GAP_THRESHOLD,
        "passes_phase3_threshold": bool(
            result.distillation_gap_estimate <= PHASE3_DISTILL_GAP_THRESHOLD
        ),
        "final_loss_kl": float(result.final_loss_kl),
        "final_loss_gt": float(result.final_loss_gt),
        "final_loss_total": float(result.final_loss_total),
        "n_epochs_completed": int(result.n_epochs_completed),
        "smoke_mode": bool(config.smoke_mode),
        "evidence_grade": (
            "[predicted; Phase 2 T10 scaffold smoke; "
            "Phase 3 unblock requires non-smoke dispatch]"
            if config.smoke_mode
            else "[empirical; Phase 2 T10 dispatch; "
            "consumed by Phase3DispatchGate per Catalog #134]"
        ),
    }
    (args.output_dir / "distillation_gap_estimate.json").write_text(
        json.dumps(gap_artifact, indent=2)
    )

    manifest = {
        "schema": T10_SCHEMA_VERSION,
        "lane_id": T10_LANE_ID,
        "phase": "2_scaffold_t10_ib_lagrangian_aux_scorer",
        "started_at_utc": started_at,
        "device": str(device),
        "smoke": bool(args.smoke),
        "epochs": int(args.epochs),
        "n_batches": int(args.n_batches),
        "predicted_delta_score": T10_PREDICTED_DELTA_SCORE,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "evidence_grade": gap_artifact["evidence_grade"],
        "phase3_prereq_artifact": "distillation_gap_estimate.json",
        "distillation_gap_estimate": float(result.distillation_gap_estimate),
        "passes_phase3_threshold": gap_artifact["passes_phase3_threshold"],
        "ema_decay": float(config.ema_decay),
        "distill_temperature": float(config.distill_temperature),
        "lambda_gt": float(config.lambda_gt),
        "research_only": True,
        "research_only_rationale": (
            "T10 produces a TRAINING-ONLY auxiliary scorer "
            "(consumed by Phase 3); no contest archive bytes ship; "
            "Phase 3 Catalog #134 distillation_gap_estimate prerequisite"
        ),
        "compliance_tags": [
            "ema_0p997",
            "hinton_distill_T_2p0",
            "lambda_gt_0p5",
            "no_mps_authoritative",
            "no_synthetic_outside_smoke",
            "no_tmp_paths",
            "auth_eval_refused_T10_has_no_archive",
            "phase3_prereq_distillation_gap_artifact",
        ],
    }
    (args.output_dir / "t10_provenance.json").write_text(
        json.dumps(manifest, indent=2)
    )
    print(
        f"[t10] done; distillation_gap={result.distillation_gap_estimate:.4f} "
        f"(Phase 3 threshold ≤{PHASE3_DISTILL_GAP_THRESHOLD}); "
        f"manifest written to {args.output_dir}/t10_provenance.json"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
