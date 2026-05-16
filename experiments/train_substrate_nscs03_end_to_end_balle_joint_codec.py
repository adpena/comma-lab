# SPDX-License-Identifier: MIT
"""Train the NSCS03 end-to-end Ballé joint codec substrate.

Operator-callable training script for NSCS03 per the assumptions-challenge-audit
NSCS03 design memo and the operator NON-NEGOTIABLE *"UNIQUE-AND-COMPLETE-PER-METHOD"*
mode landed 2026-05-15. The substrate is **end-to-end Ballé 2018 joint codec**
— convolutional analysis g_a + entropy bottleneck + scale hyperprior +
convolutional synthesis g_s — joint-trained with score-aware loss
backpropagating THROUGH the bottleneck.

Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"
non-negotiable: this trainer's ``_full_main`` is **council-gated**
(raises ``NotImplementedError`` until Phase 2 design verdict adjudicates the
λ_R sweep and conditional-Gaussian σ-floor calibration). The recipe
declares ``research_only: true`` so no $1+ Modal dispatch can fire until
the operator approves the follow-up subagent.

Council-binding contract (CLAUDE.md non-negotiables) honored end-to-end
in the planned ``_full_main`` (currently scaffolded; full implementation
deferred per substrate-engineering exception):

- Train against ``upstream/videos/0.mkv`` decoded via pyav (NOT synthetic
  data per Catalog #114).
- Patch upstream ``rgb_to_yuv6`` via ``patch_upstream_yuv6_globally`` BEFORE
  scorer construction (PR #95/#106 contract).
- ``load_differentiable_scorers`` for SegNet/PoseNet (no scorer load at
  inflate; only at training).
- ``apply_eval_roundtrip_during_training`` inside the per-batch loop.
- ``tac.training.EMA(decay=0.997)`` update after every ``optimizer.step``;
  inference checkpoint = EMA shadow (CLAUDE.md "EMA — non-negotiable").
- Score-domain Lagrangian per HNeRV parity L6 PLUS NSCS03-specific rate term::

      L = α·B(θ)/N + β·d_seg(θ) + γ·sqrt(d_pose(θ)) + λ_R·(R_main + R_hyper)

  where R_main = -log2 N(y_hat; 0, σ²) (scale hyperprior) and
  R_hyper = -log2 p_z(z_hat) (factorized prior) are differentiable through
  the entropy model.
- AdamW + cosine annealing; gradient clip 1.0; NaN watchdog.
- End with paired CPU+CUDA auth eval per CLAUDE.md "Submission auth eval —
  BOTH CPU AND CUDA".
- Continual-learning posterior update via ``posterior_update_locked``.
- Cost-band anchor append.
- Contest-compliant runtime emission (inflate.sh / inflate.py with 3
  positional args + ``set -euo pipefail`` + ≤ 200 LOC inflate.py with the
  L4 NEEDS-WORK waiver + NO scorer imports) per Catalog #146.
- TIER_1_OPERATOR_REQUIRED_FLAGS declared per Catalog #151 for wire-up.

Usage (smoke; CPU; deterministic synthetic batches; no scorer load)::

    .venv/bin/python experiments/train_substrate_nscs03_end_to_end_balle_joint_codec.py \\
        --output-dir experiments/results/nscs03_smoke_<utc> \\
        --epochs 3 --device cpu --smoke

Usage (full; deferred per substrate-engineering exception until council
approves λ_R sweep)::

    # COUNCIL-GATED — raises NotImplementedError until Phase 2 verdict.
    .venv/bin/python experiments/train_substrate_nscs03_end_to_end_balle_joint_codec.py \\
        --video-path upstream/videos/0.mkv \\
        --output-dir experiments/results/nscs03_<utc> \\
        --epochs 2000 --batch-size 16 --lr 5e-4 --device cuda
"""
# AUTOCAST_FP16_WAIVED:entropy-bottleneck-numerical-instability-pending-fp32-eb-forward
# TORCH_COMPILE_WAIVED:defer-until-per-substrate-canary-validates-Inductor-graph-breaks
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tac.substrates.nscs03_end_to_end_balle_joint_codec.registered_substrate import (
    NSCS03_END_TO_END_BALLE_CONTRACT,  # noqa: F401  (forces contract validation)
)

# ---------------------------------------------------------------------------
# Module paths + constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO_PATH = REPO_ROOT / "upstream" / "videos" / "0.mkv"
DEFAULT_UPSTREAM_DIR = REPO_ROOT / "upstream"

# Eval-roundtrip target resolution (per upstream evaluate.py):
EVAL_HW = (384, 512)
CAMERA_HW = (874, 1164)
N_PAIRS_FULL = 600
CONTEST_NORMALIZER = 37_545_489.0


# ---------------------------------------------------------------------------
# Catalog #151 manifest — every flag below must be threaded by any operator
# wrapper that subprocess-invokes this trainer.
# ---------------------------------------------------------------------------
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--video-path": {
        "env": "NSCS03_VIDEO_PATH",
        "rationale": (
            "score-aware substrate MUST train against the contest video "
            "(upstream/videos/0.mkv); synthetic data is FORBIDDEN outside --smoke"
        ),
        "default": str(DEFAULT_VIDEO_PATH.relative_to(REPO_ROOT)),
        "required_input_file": True,
        "generator_command": "contest-pinned upstream snapshot — never regenerated locally",
        "rationale_audit": (
            ".omx/research/assumptions_challenge_audit_shared_assumptions_matrix_20260515.json"
            "#NSCS03"
        ),
    },
    "--output-dir": {
        "env": "NSCS03_OUTPUT_DIR",
        "rationale": "custody location for checkpoints + archive + provenance",
    },
    "--epochs": {
        "env": "NSCS03_EPOCHS",
        "rationale": (
            "end-to-end joint codec; under-training silently regresses (council "
            "target: 2000 for full; 100 for smoke)"
        ),
        "default": "2000",
    },
    "--upstream-dir": {
        "env": "NSCS03_UPSTREAM_DIR",
        "rationale": (
            "upstream/ root for scorer weights + evaluate.py; required for full "
            "training (non-smoke) and auth eval"
        ),
        "default": str(DEFAULT_UPSTREAM_DIR.relative_to(REPO_ROOT)),
    },
    "--device": {
        "env": "NSCS03_DEVICE",
        "rationale": (
            "compute device; cuda required for full training (MPS refused per "
            "CLAUDE.md MPS-NOISE rule); cpu permitted only with --smoke"
        ),
        "default": "cuda",
    },
    "--main-latent-channels": {
        "env": "NSCS03_MAIN_LATENT_CHANNELS",
        "rationale": (
            "NSCS03-specific: main latent y channels (Ballé 2018 reference 192; "
            "ours 64 to keep param/rate envelope compatible with the 0.19 "
            "frontier). Sweep [48, 64, 96, 128] in follow-up wave."
        ),
        "default": "64",
    },
    "--hyper-latent-channels": {
        "env": "NSCS03_HYPER_LATENT_CHANNELS",
        "rationale": (
            "NSCS03-specific: hyper latent z channels (Ballé 2018 reference 128; "
            "ours 32). Side-info stream determines the conditional Gaussian sigma "
            "for the main latent."
        ),
        "default": "32",
    },
    "--lambda-R": {
        "env": "NSCS03_LAMBDA_R",
        "rationale": (
            "NSCS03-specific: weight on the differentiable rate term "
            "(R_main + R_hyper). Default 0.5; sweep [0.1, 1.0] in follow-up."
        ),
        "default": "0.5",
    },
    "--gdn-eps": {
        "env": "NSCS03_GDN_EPS",
        "rationale": (
            "NSCS03-specific: GDN numerical floor; default 1e-6 (NOT 1e-12) for "
            "fp16-autocast hygiene per Catalog #172."
        ),
        "default": "1e-6",
    },
    "--sigma-floor": {
        "env": "NSCS03_SIGMA_FLOOR",
        "rationale": (
            "NSCS03-specific: minimum sigma for the conditional-Gaussian density "
            "on y. Prevents degenerate p_y collapse to delta-function."
        ),
        "default": "1e-4",
    },
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="train_substrate_nscs03_end_to_end_balle_joint_codec",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--video-path", type=Path, default=DEFAULT_VIDEO_PATH)
    p.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--main-latent-channels", type=int, default=64)
    p.add_argument("--hyper-latent-channels", type=int, default=32)
    p.add_argument("--lambda-R", type=float, default=0.5)
    p.add_argument("--gdn-eps", type=float, default=1e-6)
    p.add_argument("--sigma-floor", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args(argv)


def _smoke_main(args: argparse.Namespace) -> int:
    """Smoke main: build the substrate; do a tiny forward + backward sanity
    check; emit a smoke stats.json. Catalog #114 compliance: synthetic data
    is permitted ONLY in smoke mode."""
    import torch

    from tac.substrates.nscs03_end_to_end_balle_joint_codec import (
        NSCS03Config,
        NSCS03JointCodecSubstrate,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[NSCS03 smoke] device={args.device} epochs={args.epochs}")

    cfg = NSCS03Config(
        main_latent_channels=args.main_latent_channels,
        hyper_latent_channels=args.hyper_latent_channels,
        gdn_eps=args.gdn_eps,
        sigma_floor=args.sigma_floor,
    )
    torch.manual_seed(args.seed)
    model = NSCS03JointCodecSubstrate(cfg).to(args.device)
    n_params = model.num_parameters()
    print(f"[NSCS03 smoke] num_params={n_params}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    for epoch in range(args.epochs):
        # SYNTHETIC_NON_SMOKE_OK:smoke-mode-only-per-Catalog-114 — not used in
        # full path because _full_main raises NotImplementedError.
        x = torch.rand(2, cfg.in_channels, 384, 512, device=args.device)
        recon, parts = model(x)
        loss = (
            torch.nn.functional.mse_loss(recon, x) * 100.0
            + args.lambda_R * parts["total_rate"]
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        if (epoch + 1) % max(1, args.epochs // 4) == 0:
            print(
                f"[NSCS03 smoke] epoch {epoch + 1}/{args.epochs} "
                f"loss={loss.item():.4f} main_rate={parts['main_rate'].item():.4f} "
                f"hyper_rate={parts['hyper_rate'].item():.4f}"
            )

    stats = {
        "smoke": True,
        "epochs": args.epochs,
        "num_params": n_params,
        "final_loss": float(loss.item()),
        "final_main_rate": float(parts["main_rate"].item()),
        "final_hyper_rate": float(parts["hyper_rate"].item()),
        "config": asdict(cfg),
        # Per CLAUDE.md "Apples-to-apples evidence discipline": tag the smoke
        # output with explicit non-promotion fields so it cannot be mistaken
        # for a contest-axis anchor.
        "auth_eval_score": None,
        "auth_eval_score_axis": "smoke_no_auth_eval",
        "auth_eval_score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "result_review_blockers": [
            "smoke_path_synthetic_data_no_auth_eval",
            "research_only_substrate_engineering_scaffold",
        ],
    }
    out_path = args.output_dir / "stats.json"
    out_path.write_text(json.dumps(stats, indent=2, sort_keys=True))
    print(f"[NSCS03 smoke] DONE; stats written to {out_path}")
    return 0


def _full_main(args: argparse.Namespace) -> int:
    """Full training path — COUNCIL-GATED per CLAUDE.md substrate-engineering
    exception (Catalog #220 + #240).

    Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"
    non-negotiable: NSCS03 lands as a substrate-engineering scaffold with
    ``research_only=true`` declared in the recipe. The full _full_main
    implementation is deferred until the operator-approved Phase 2 follow-up
    subagent calibrates:

    1. λ_R sweep [0.1, 1.0] (4-config Lightning T4 smoke first)
    2. Conditional-Gaussian σ-floor sensitivity (current 1e-4 may be too tight
       at low rate operating points)
    3. EntropyBottleneck initial scale (Ballé 2018 reference uses 10; our
       canonical default may need recalibration for video latents)
    4. AdamW LR schedule (cosine vs Ballé 2018 staircase)
    5. EMA decay (canonical 0.997 vs Ballé 2018 reference 0.999 for hyperprior)

    Per CLAUDE.md "KILL is LAST RESORT": this is NOT a kill; it is a
    DEFERRED-pending-research substrate-engineering scaffold with explicit
    reactivation criteria. The recipe already declares this state via
    ``research_only: true`` and ``dispatch_blockers: [phase_2_council_approval_required]``.

    pre_build_substrate_engineering signal honored per Catalog #220 acceptance
    cascade (3): ``_full_main raises NotImplementedError`` AND
    ``lane_class=substrate_engineering`` declared in lane registry notes.
    """
    raise NotImplementedError(
        "NSCS03 _full_main is council-gated per CLAUDE.md substrate-engineering "
        "exception. The substrate ships as research_only=true scaffold. The full "
        "trainer (with paired CPU+CUDA auth eval, archive emission, runtime "
        "vendoring, cost-band anchor append, continual-learning posterior "
        "update) lands once the Phase 2 council adjudicates the lambda_R sweep + "
        "sigma-floor calibration. See "
        ".omx/research/assumptions_challenge_audit_shared_assumptions_matrix_20260515.json#NSCS03 "
        "reactivation_criteria_if_smoke_regresses for the gating empirical "
        "thresholds."
    )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.smoke:
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":
    sys.exit(main())
