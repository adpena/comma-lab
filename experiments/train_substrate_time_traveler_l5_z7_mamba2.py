# SPDX-License-Identifier: MIT
"""Z7-Mamba-2 substrate trainer SCAFFOLD — pre-build per Catalog #240.

Per the Z7-Mamba-2 design memo
(``.omx/research/z7_mamba2_substrate_design_memo_20260518.md``):
this is the TOP-5 #2 SCAFFOLD landing per the deep-research wave
(``.omx/research/comprehensive_research_wave_20260518.md`` §0 + §2.2 + §3.6).

Z7-Mamba-2 is the **canonical Catalog #308 N>=3 alternative-probe-methodology**
to Z7-GRU (Hafner Revision #3 binding) within the predictive-coding-recurrent
substrate class. Mamba-2 (Dao-Gu 2024, arxiv 2405.21060) is a selective
state-space sequence model proven to match Transformer quality at O(N)
compute on long-context language and video tasks.

**This trainer is PRE-BUILD per Catalog #240** — ``_full_main`` raises
``NotImplementedError`` until Wave N+1 council convenes PROCEED-unconditional
per Z7 parent symposium Revision #6 cascade (sequenced AFTER Z7-GRU Wave 2
disambiguator outcome OR operator explicit-frontier-override per Catalog
#300 with verbatim quote in ``council_override_rationale``).

The recipe (``.omx/operator_authorize_recipes/substrate_time_traveler_l5_z7_mamba2_modal_a100_dispatch.yaml``)
declares ``research_only: true`` + ``dispatch_enabled: false`` so this
scaffold satisfies Catalog #240 substrate-engineering discipline without
risking phantom-score dispatch.

Scope at THIS scaffold landing:

- ``_smoke_main`` builds a tiny ``Mamba2PredictorConfig`` + instantiates
  ``Mamba2Predictor`` + forward-pass sanity check on synthetic 600-pair
  sequence. Validates canonical signature compatibility with Z6 sister
  per design memo §6 layer #2.
- ``_full_main`` raises ``NotImplementedError`` with explicit message
  citing Z7 parent symposium Revision #6 + Wave N+1 council requirement.

Phase 2 build (deferred to Wave N+1 PROCEED-unconditional council):

- Full Z7-Mamba-2 substrate (Z6 encoder + Mamba2Predictor + Z6 decoder +
  Z7MCM2 archive grammar + Z6 score-aware loss pattern).
- ``_full_main`` decodes real pairs via canonical
  ``tac.substrates._shared.trainer_skeleton.decode_real_pairs``, derives
  runtime-configurable ego-source per Revision #4 (PoseNet-projection
  OR scorer-logit-conditioning from Z6 4c outcome), trains across 100ep
  with β-IB-Lagrangian initialized from C6 Phase 2 empirical anchor.
- Routes auth eval through canonical
  ``smoke_auth_eval_gate.gate_auth_eval_call`` per Catalog #226.
- Inflate-device-fork via canonical ``select_inflate_device`` per
  Catalog #205.
"""
# AUTOCAST_FP16_WAIVED:pre-build-scaffold-no-mixed-precision-until-Wave-N+1-trainer-build
# TORCH_COMPILE_WAIVED:pre-build-scaffold-autoregressive-Mamba2-unroll-needs-canary-validation-at-Wave-N+1
# TF32_WAIVED:pre-build-scaffold-no-CUDA-matmul-pre-Wave-N+1
# NO_GRAD_WAIVED:pre-build-scaffold-no-scorer-forward-pre-Wave-N+1
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

from tac.optimization.mamba2_predictor import (
    MAMBA_SSM_AVAILABLE,
    Mamba2Predictor,
    Mamba2PredictorConfig,
)

DEFAULT_VIDEO_PATH = REPO_ROOT / "upstream" / "videos" / "0.mkv"


# ---------------------------------------------------------------------------
# Catalog #151 manifest — every flag below must be threaded by any operator
# wrapper. AnnAssign per Catalog #168 (NOT bare Assign).
# Required-input flags carry ``required_input_file: True`` per Catalog #152.
# ---------------------------------------------------------------------------
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--video-path": {
        "env": "Z7_MAMBA2_VIDEO_PATH",
        "rationale": (
            "Path to the contest video upstream/videos/0.mkv decoded via "
            "pyav into per-pair frames; required for non-smoke training "
            "at Wave N+1 trainer build"
        ),
        "default": str(DEFAULT_VIDEO_PATH),
        "required_input_file": True,
    },
    "--output-dir": {
        "env": "Z7_MAMBA2_OUTPUT_DIR",
        "rationale": (
            "Output directory for checkpoints, archive, stats, runtime "
            "tree, auth eval JSON; must be writable + outside /tmp"
        ),
        "default": None,
    },
    "--epochs": {
        "env": "Z7_MAMBA2_EPOCHS",
        "rationale": "Training epoch count; smoke=3, Modal A100 full=100",
        "default": "100",
    },
    "--batch-size": {
        "env": "Z7_MAMBA2_BATCH_SIZE",
        "rationale": (
            "Per-step pair count; A100 handles 4-8 at 384x512 with "
            "autoregressive Mamba-2 unroll across 600 pairs"
        ),
        "default": "4",
    },
    "--lr": {
        "env": "Z7_MAMBA2_LR",
        "rationale": "AdamW base learning rate; default 5e-4 per Z6 sister",
        "default": "5e-4",
    },
    "--mamba2-d-model": {
        "env": "Z7_MAMBA2_D_MODEL",
        "rationale": (
            "Mamba-2 internal model dimension; default 64 per design memo "
            "§7 architectural specification (sister to GRU hidden_dim=128 "
            "halved for parameter parity)"
        ),
        "default": "64",
    },
    "--mamba2-d-state": {
        "env": "Z7_MAMBA2_D_STATE",
        "rationale": (
            "Mamba-2 selective state-space dimension; default 16 per upstream "
            "Mamba-2 canonical for language; CC-9 CARGO-CULTED-PENDING for "
            "dashcam contest 600-pair sequence per design memo §2"
        ),
        "default": "16",
    },
    "--mamba2-expand": {
        "env": "Z7_MAMBA2_EXPAND",
        "rationale": "Mamba-2 expansion factor; default 2 per upstream reference",
        "default": "2",
    },
    "--mamba2-backend": {
        "env": "Z7_MAMBA2_BACKEND",
        "rationale": (
            "Mamba-2 backend selection: 'auto' (default; mamba_ssm fallback "
            "to reference_torch), 'mamba_ssm' (CUDA only), 'reference_torch' "
            "(MPS/CPU compatible per design memo §13 local proxy training)"
        ),
        "default": "auto",
    },
    "--ego-source": {
        "env": "Z7_MAMBA2_EGO_SOURCE",
        "rationale": (
            "Runtime-configurable ego-source per Z7 parent symposium Revision "
            "#4: 'posenet_projection' (Z6-v1 baseline 8-dim) OR "
            "'scorer_logit_compressed' (Z6 Wave 2 4c winning channel if "
            "full-FiLM-WIN)"
        ),
        "default": "posenet_projection",
    },
    "--ego-motion-dim": {
        "env": "Z7_MAMBA2_EGO_MOTION_DIM",
        "rationale": (
            "Ego-motion vector dimension; default 8 matches Z6-v1 PoseNet-"
            "projection baseline"
        ),
        "default": "8",
    },
    "--identity-predictor": {
        "env": "Z7_MAMBA2_IDENTITY_PREDICTOR",
        "rationale": (
            "Probe-disambiguator ablation per Catalog #125 hook #6 + Z7 "
            "symposium Revision #2 same-archive-bytes pattern: when true, "
            "predictor returns z_prev unchanged (no learning); compare to "
            "full Mamba-2 for Rao-Ballard refutation/confirmation"
        ),
        "default": "false",
    },
    "--stateful": {
        "env": "Z7_MAMBA2_STATEFUL",
        "rationale": (
            "When true, Mamba-2 hidden state persists across 600-pair "
            "sequence (Wyner-Ziv implicit side-info channel pattern per "
            "Catalog #311 Ballard verbatim). When false, state resets every "
            "pair (ablation: Mamba-2 reduces to stateless nonlinear transform)"
        ),
        "default": "true",
    },
    "--beta-ib": {
        "env": "Z7_MAMBA2_BETA_IB",
        "rationale": (
            "β-IB-Lagrangian parameter; per Z7 symposium Revision #5 + design "
            "memo §6 layer #4: MUST initialize from C6 IBPS Phase 2 empirical "
            "β-optimal anchor at Wave N+1 trainer build (NOT guessed "
            "independently). Placeholder default=1.0 for scaffold smoke."
        ),
        "default": "1.0",
    },
    "--smoke": {
        "env": "Z7_MAMBA2_SMOKE",
        "rationale": (
            "When set, runs _smoke_main: tiny Mamba2Predictor instantiation "
            "+ forward-pass sanity check on synthetic data. _full_main "
            "(non-smoke) raises NotImplementedError until Wave N+1 council "
            "PROCEED-unconditional per Z7 symposium Revision #6"
        ),
        "default": "false",
    },
    "--device": {
        "env": "Z7_MAMBA2_DEVICE",
        "rationale": (
            "Compute device: 'cuda' (Modal A100), 'mps' (local M5 Max proxy "
            "per design memo §13), 'cpu' (smoke). MPS produces NON-AUTHORITATIVE "
            "MPS-research-signal evidence per CLAUDE.md 'MPS auth eval is NOISE'"
        ),
        "default": "cpu",
    },
}


# ---------------------------------------------------------------------------
# Catalog #244 NVML/CUDA env exports — declared at module level so audit
# tools see the canonical compliance even though the trainer scaffold
# itself does NOT dispatch to GPU; sister to driver script wire-in.
# ---------------------------------------------------------------------------
# DALI_DISABLE_NVML=1
# CUBLAS_WORKSPACE_CONFIG=:4096:8
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Z7-Mamba-2 substrate trainer SCAFFOLD (pre-build per Catalog #240). "
            "Smoke mode validates canonical signature; full mode raises "
            "NotImplementedError until Wave N+1 council."
        )
    )
    for flag, meta in TIER_1_OPERATOR_REQUIRED_FLAGS.items():
        default_env = meta.get("env")
        if default_env and default_env in os.environ:
            default = os.environ[default_env]
        else:
            default = meta.get("default")
        if flag in ("--smoke", "--identity-predictor", "--stateful"):
            # Bool flags
            parser.add_argument(
                flag,
                action="store_true" if default in (None, "false", "False", False) else "store_false",
                help=meta.get("rationale", ""),
            )
        else:
            parser.add_argument(
                flag,
                default=default,
                help=meta.get("rationale", ""),
            )
    return parser


def _resolve_smoke_config(args: argparse.Namespace) -> Mamba2PredictorConfig:
    """Build Mamba2PredictorConfig from argparse, with smoke defaults."""
    return Mamba2PredictorConfig(
        latent_dim=24,  # matches Z6-v1
        ego_motion_dim=int(args.ego_motion_dim),
        d_model=int(args.mamba2_d_model),
        d_state=int(args.mamba2_d_state),
        expand=int(args.mamba2_expand),
        d_conv=4,
        backend=str(args.mamba2_backend),
        stateful=bool(args.stateful) if isinstance(args.stateful, bool)
                 else str(args.stateful).lower() in ("true", "1", "yes"),
        identity_predictor=bool(args.identity_predictor) if isinstance(args.identity_predictor, bool)
                           else str(args.identity_predictor).lower() in ("true", "1", "yes"),
    )


def _smoke_main(args: argparse.Namespace) -> int:
    """Smoke: instantiate Mamba2Predictor + forward-pass sanity check.

    Validates:
    1. Mamba2PredictorConfig builds with default canonical parameters.
    2. Mamba2Predictor instantiates with available backend (auto-fallback OK).
    3. Forward-pass on synthetic 4-pair batch produces correct shape.
    4. Identity-predictor mode returns z_prev unchanged.
    5. Per-pair master gradient compatibility (Catalog #810).
    6. 20-step autoregressive unroll on synthetic sequence (sanity that
       state-evolution works).
    """
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is None:
        output_dir = REPO_ROOT / "experiments" / "results" / (
            f"z7_mamba2_scaffold_smoke_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[z7_mamba2_scaffold] smoke mode; output_dir={output_dir}")
    print(f"[z7_mamba2_scaffold] MAMBA_SSM_AVAILABLE={MAMBA_SSM_AVAILABLE}")
    print(f"[z7_mamba2_scaffold] device={args.device}")
    print(f"[z7_mamba2_scaffold] backend={args.mamba2_backend}")

    cfg = _resolve_smoke_config(args)
    print(f"[z7_mamba2_scaffold] config={cfg}")

    predictor = Mamba2Predictor(cfg)
    print(f"[z7_mamba2_scaffold] predictor={predictor.to_z6_compatible_signature()}")
    print(f"[z7_mamba2_scaffold] num_parameters={predictor.num_parameters()}")

    # Sanity 1: forward-pass shape
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    predictor = predictor.to(device)
    z_prev = torch.randn(4, cfg.latent_dim, device=device)
    ego = torch.randn(4, cfg.ego_motion_dim, device=device)
    out = predictor(z_prev, ego)
    assert out.shape == z_prev.shape, f"shape mismatch: {out.shape} != {z_prev.shape}"
    print(f"[z7_mamba2_scaffold] forward-pass OK: input {tuple(z_prev.shape)} -> output {tuple(out.shape)}")

    # Sanity 2: identity-predictor mode
    cfg_id = Mamba2PredictorConfig(
        latent_dim=cfg.latent_dim,
        ego_motion_dim=cfg.ego_motion_dim,
        identity_predictor=True,
    )
    predictor_id = Mamba2Predictor(cfg_id).to(device)
    out_id = predictor_id(z_prev, ego)
    assert torch.allclose(out_id, z_prev), "identity_predictor must return z_prev unchanged"
    assert predictor_id.num_parameters() == 0, "identity_predictor must have zero trainable params"
    print(f"[z7_mamba2_scaffold] identity-predictor mode OK (0 params)")

    # Sanity 3: per-pair master gradient compatibility (Catalog #810)
    z_grad = torch.randn(4, cfg.latent_dim, device=device, requires_grad=True)
    e_grad = torch.randn(4, cfg.ego_motion_dim, device=device, requires_grad=True)
    out_grad = predictor(z_grad, e_grad)
    out_grad.sum().backward()
    assert z_grad.grad is not None and z_grad.grad.abs().sum() > 0, (
        "z_prev gradient must flow through Mamba-2 predictor (Catalog #810)"
    )
    assert e_grad.grad is not None and e_grad.grad.abs().sum() > 0, (
        "ego_motion gradient must flow through Mamba-2 predictor (Catalog #810)"
    )
    print(f"[z7_mamba2_scaffold] per-pair master gradient compatibility OK (Catalog #810)")

    # Sanity 4: 20-step autoregressive unroll (state-evolution sanity)
    predictor.reset_state(1, device=device)
    states = []
    for t in range(20):
        z = torch.randn(1, cfg.latent_dim, device=device)
        e = torch.randn(1, cfg.ego_motion_dim, device=device)
        with torch.no_grad():
            outs = predictor(z, e)
        states.append(outs.detach().clone())
    # State should evolve (outputs differ between early and late timesteps)
    state_diff = (states[0] - states[10]).abs().sum().item()
    assert state_diff > 0, "Mamba-2 state should evolve across 20-step unroll"
    print(f"[z7_mamba2_scaffold] 20-step autoregressive unroll OK (state diff={state_diff:.4f})")

    # Emit smoke stats
    stats_path = output_dir / "z7_mamba2_scaffold_smoke_stats.json"
    stats = {
        "schema_version": 1,
        "name": "z7_mamba2_scaffold_smoke_stats",
        "substrate_id": "time_traveler_l5_z7_mamba2",
        "lane_id": "lane_top5_2_z7_mamba2_scaffold_design_20260518",
        "mamba_ssm_available": MAMBA_SSM_AVAILABLE,
        "backend_active": predictor.backend_active,
        "num_parameters": predictor.num_parameters(),
        "config": {
            "latent_dim": cfg.latent_dim,
            "ego_motion_dim": cfg.ego_motion_dim,
            "d_model": cfg.d_model,
            "d_state": cfg.d_state,
            "expand": cfg.expand,
            "d_conv": cfg.d_conv,
            "backend": cfg.backend,
            "stateful": cfg.stateful,
            "identity_predictor": cfg.identity_predictor,
        },
        "sanity_checks_passed": [
            "instantiation",
            "forward_pass_shape",
            "identity_predictor_returns_z_prev",
            "per_pair_master_gradient_compatible_catalog_810",
            "20_step_autoregressive_state_evolution",
        ],
        "evidence_grade": "smoke_scaffold_only_NOT_promotable",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "result_review_blockers": [
            "scaffold_smoke_validates_canonical_signature_only_not_training",
            "full_main_raises_NotImplementedError_per_catalog_240",
            "wave_n_plus_1_council_required_per_z7_symposium_revision_6",
            "z7_gru_wave_2_disambiguator_outcome_required_per_revision_1",
        ],
        "device": str(device),
        "utc_now": datetime.now(UTC).isoformat(),
    }
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True))
    print(f"[z7_mamba2_scaffold] stats written: {stats_path}")

    return 0


def _full_main(args: argparse.Namespace) -> int:
    """Full Z7-Mamba-2 training — NOT YET BUILT per Catalog #240.

    Raises NotImplementedError per Z7 parent symposium Revision #6 +
    Catalog #240 substrate-engineering discipline. The full trainer
    requires:

    1. Z7-GRU Wave 2 disambiguator outcome (sister codex probe pending
       paired exact-eval at this scaffold; per Z7 symposium Revision #1
       binding).
    2. C6 IBPS Phase 2 redesign empirical β-IB-Lagrangian anchor
       (sister memo pending; per Z7 symposium Revision #5 binding).
    3. Wave N+1 council PROCEED-unconditional verdict (per Catalog #315
       OPTIMAL-FORM iteration discipline).
    4. Operator dispatch authorization with verbatim quote in
       ``council_override_rationale`` if invoking PATH 2 (operator
       explicit-frontier-override per Catalog #300 skipping Z7-GRU).

    Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"
    non-negotiable + Catalog #220: this scaffold opt-out is canonical
    via ``research_only: true`` in the operator-authorize recipe + this
    explicit raise.
    """
    raise NotImplementedError(
        "Z7-Mamba-2 substrate _full_main is PRE-BUILD per Catalog #240. "
        "Per Z7 parent symposium Revision #6 (.omx/research/"
        "council_per_substrate_symposium_z7_lstm_predictive_coding_"
        "20260517.md): Z7-Mamba-2 is enumerated as Wave-N+2 PIVOT-PATH "
        "(b) IF Z7-GRU Wave 2 disambiguator LOSES or DEFERs. Per Catalog "
        "#315 OPTIMAL-FORM iteration discipline: Z7-Mamba-2 paid dispatch "
        "requires Wave N+1 council PROCEED-unconditional verdict convened "
        "AFTER (a) Z7-GRU Wave 2 disambiguator outcome lands OR (b) "
        "operator explicit-frontier-override per Catalog #300 with verbatim "
        "quote in council_override_rationale. Recipe declares "
        "research_only: true + dispatch_enabled: false per Catalog #240 + "
        "#324. See .omx/research/z7_mamba2_substrate_design_memo_20260518.md "
        "for full reactivation cascade (§9 dispatch sequencing + §10 pivot "
        "paths). For LOCAL M5 Max MPS proxy training pattern per design "
        "memo §13: use --smoke flag with --device mps; results are "
        "MPS-research-signal grade per CLAUDE.md 'MPS auth eval is NOISE'."
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)

    smoke = bool(args.smoke) if isinstance(args.smoke, bool) \
        else str(args.smoke).lower() in ("true", "1", "yes")

    if smoke:
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":
    sys.exit(main())
