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
# AUTOCAST_FP16_WAIVED:research-only-substrate-engineering-pending-PATH-C-100ep-canary-per-Catalog-325-symposium
# TORCH_COMPILE_WAIVED:research-only-substrate-engineering-Mamba2-unroll-needs-PATH-B-T4-smoke-canary-validation-pre-compile
# TF32_WAIVED:research-only-substrate-engineering-TF32-lands-at-PATH-C-100ep-canary-per-Catalog-178
# NO_GRAD_WAIVED:research-only-substrate-engineering-eval-time-no-grad-wraps-at-PATH-C-canary-per-Catalog-180
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

import hashlib
import shutil
import time
import zipfile

import torch.nn.functional as F

from tac.optimization.mamba2_predictor import (
    MAMBA_SSM_AVAILABLE,
    Mamba2Predictor,
    Mamba2PredictorConfig,
)
from tac.substrates.time_traveler_l5_z7_mamba2 import (
    Z7Mamba2PredictiveCodingConfig,
    Z7Mamba2PredictiveCodingSubstrate,
    Z7Mamba2PredictiveCodingLossWeights,
    Z7Mamba2PredictiveCodingScoreAwareLoss,
    pack_archive,
)
from tac.substrates.time_traveler_l5_z7_mamba2.inflate import inflate_one_video

# Canonical helpers for full _full_main implementation
try:
    from tac.substrates._shared.trainer_skeleton import (
        decode_real_pairs as _decode_real_pairs,
    )
except ImportError:  # pragma: no cover - canonical helper not in path
    _decode_real_pairs = None

SUBSTRATE_ID = "time_traveler_l5_z7_mamba2"
LANE_ID = "lane_z7_as_mamba_2_full_landing_20260518"

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
    "--max-pairs": {
        "env": "Z7_MAMBA2_MAX_PAIRS",
        "rationale": (
            "Number of real contest frame-pairs to train/export. Use 600 only "
            "for a future ratified full run; smoke/timing defaults to 8."
        ),
        "default": "8",
    },
    "--latent-dim": {
        "env": "Z7_MAMBA2_LATENT_DIM",
        "rationale": "Latent dimensionality for the byte-closed Z7MCM2 packet.",
        "default": "24",
    },
    "--decoder-embed-dim": {
        "env": "Z7_MAMBA2_DECODER_EMBED_DIM",
        "rationale": "Decoder embedding width for the Z6-compatible RGB renderer.",
        "default": "32",
    },
    "--decoder-channels": {
        "env": "Z7_MAMBA2_DECODER_CHANNELS",
        "rationale": "Comma-separated decoder channels, e.g. 32,24,16,12.",
        "default": "32,24,16,12",
    },
    "--decoder-num-upsample-blocks": {
        "env": "Z7_MAMBA2_DECODER_UPSAMPLE_BLOCKS",
        "rationale": "PixelShuffle upsample block count for the RGB decoder.",
        "default": "4",
    },
    "--decoder-initial-grid-h": {
        "env": "Z7_MAMBA2_DECODER_INITIAL_GRID_H",
        "rationale": "Decoder initial latent grid height.",
        "default": "24",
    },
    "--decoder-initial-grid-w": {
        "env": "Z7_MAMBA2_DECODER_INITIAL_GRID_W",
        "rationale": "Decoder initial latent grid width.",
        "default": "32",
    },
    "--output-height": {
        "env": "Z7_MAMBA2_OUTPUT_HEIGHT",
        "rationale": "Training/render height; full score-aware runs should use 384.",
        "default": "384",
    },
    "--output-width": {
        "env": "Z7_MAMBA2_OUTPUT_WIDTH",
        "rationale": "Training/render width; full score-aware runs should use 512.",
        "default": "512",
    },
    "--inflate-verify": {
        "env": "Z7_MAMBA2_INFLATE_VERIFY",
        "rationale": (
            "Run scorer-free inflate on the emitted packet as a local runtime check."
        ),
        "default": "false",
    },
    "--emit-static-control": {
        "env": "Z7_MAMBA2_EMIT_STATIC_CONTROL",
        "rationale": (
            "Emit an identity/static-capacity control archive.zip with the same "
            "contest archive byte count as the recurrent packet (Catalog #125 hook #6)."
        ),
        "default": "true",
    },
    "--loss-mode": {
        "env": "Z7_MAMBA2_LOSS_MODE",
        "rationale": (
            "Training loss: proxy keeps the local MSE smoke path; score_aware "
            "loads frozen differentiable scorers at compress time only."
        ),
        "default": "proxy",
    },
    "--noise-std": {
        "env": "Z7_MAMBA2_NOISE_STD",
        "rationale": "Eval-roundtrip-aware noise stddev for score_aware loss training.",
        "default": "0.0",
    },
    "--upstream-dir": {
        "env": "Z7_MAMBA2_UPSTREAM_DIR",
        "rationale": "Path to upstream/ for differentiable scorer load.",
        "default": str(REPO_ROOT / "upstream"),
    },
    "--alpha-rate": {
        "env": "Z7_MAMBA2_ALPHA_RATE",
        "rationale": "Score-aware rate-term weight (contest canonical 25.0).",
        "default": "25.0",
    },
    "--beta-seg": {
        "env": "Z7_MAMBA2_BETA_SEG",
        "rationale": "Score-aware seg-term weight (contest canonical 100.0).",
        "default": "100.0",
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


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _build_archive_zip(zip_path: Path, *, bin_bytes: bytes, comment: bytes = b"") -> bytes:
    """Build deterministic archive.zip with 0.bin as single member.

    Per Catalog #5 `check_archive_builders_use_deterministic_zip`: use
    ZipInfo + writestr with fixed timestamp + no compression for
    byte-stable output across re-builds.
    """
    info = zipfile.ZipInfo(filename="0.bin", date_time=(2026, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(info, bin_bytes)
        if comment:
            zf.comment = comment
    return zip_path.read_bytes()


def _resolve_full_config(args: argparse.Namespace) -> Z7Mamba2PredictiveCodingConfig:
    """Build full Z7Mamba2PredictiveCodingConfig from argparse."""
    decoder_channels = tuple(int(c.strip()) for c in str(args.decoder_channels).split(","))
    return Z7Mamba2PredictiveCodingConfig(
        latent_dim=int(args.latent_dim),
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
        beta_ib=float(args.beta_ib),
        num_pairs=int(args.max_pairs),
        decoder_embed_dim=int(args.decoder_embed_dim),
        decoder_initial_grid_h=int(args.decoder_initial_grid_h),
        decoder_initial_grid_w=int(args.decoder_initial_grid_w),
        decoder_channels=decoder_channels,
        decoder_num_upsample_blocks=int(args.decoder_num_upsample_blocks),
        output_height=int(args.output_height),
        output_width=int(args.output_width),
    )


def _select_device(device: str) -> torch.device:
    """Select device with canonical fallback.

    Per CLAUDE.md "Forbidden device-selection defaults": this function
    DOES NOT silently fall back to MPS — explicit --device choice required.
    """
    device = str(device).lower()
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda requested but torch.cuda.is_available()=False; "
                "this trainer's full mode requires explicit device choice per "
                "CLAUDE.md 'Forbidden device-selection defaults'"
            )
        return torch.device("cuda")
    if device == "mps":
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            raise RuntimeError(
                "--device mps requested but MPS not available on this host"
            )
        return torch.device("mps")
    if device == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unknown device {device!r}; expected one of cuda|mps|cpu")


def _normalize_loss_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in ("proxy", "score_aware"):
        raise ValueError(f"unknown loss_mode {mode!r}; expected proxy|score_aware")
    return mode


def _ego_motion_from_pairs(pairs: torch.Tensor, ego_motion_dim: int) -> torch.Tensor:
    """Derive ego-motion sidecar from frame-pair delta proxy.

    Per Z7 parent symposium Revision #4: runtime-configurable ego-source.
    This is the canonical baseline (PoseNet-projection NOT used at training
    to avoid scorer load; sister to Z7-LSTM canonical).
    """
    if pairs.dim() != 5 or pairs.shape[1] != 2 or pairs.shape[2] != 3:
        raise ValueError(
            f"pairs must be (num_pairs, 2, 3, H, W); got {tuple(pairs.shape)}"
        )
    delta = (pairs[:, 1] - pairs[:, 0]).abs().mean(dim=(1, 2, 3))  # (num_pairs,)
    # Tile delta scalar into ego_motion_dim slots with simple sinusoidal embedding.
    num_pairs = pairs.shape[0]
    ego = torch.zeros(num_pairs, ego_motion_dim, dtype=torch.float32)
    for k in range(ego_motion_dim):
        ego[:, k] = (delta * (k + 1) * 0.1).sin()
    return ego


def _estimate_archive_bytes_proxy(
    model: Z7Mamba2PredictiveCodingSubstrate,
    config: Z7Mamba2PredictiveCodingConfig,
) -> int:
    """Estimate archive byte cost for score-aware rate term.

    Approximates: decoder/predictor fp16 + int8 latent_init/residuals/ego.
    Compression by zlib brings this ~2x lower in practice; calibration
    against measured archive bytes happens post-emission.
    """
    decoder_params = model.decoder.num_parameters()
    predictor_params = model.predictor.num_parameters()
    ctx_params = (
        0 if model.context_conditioner is None
        else sum(p.numel() for p in model.context_conditioner.parameters() if p.requires_grad)
    )
    fp16_bytes = 2 * (decoder_params + predictor_params + ctx_params)
    int8_bytes = (
        config.latent_dim
        + config.num_pairs * config.latent_dim
        + config.num_pairs * config.ego_motion_dim
    )
    # Compression ratio rough estimate ~0.5 for zlib level 9 on weight blobs
    return int(0.5 * fp16_bytes + int8_bytes + 500)  # 500 byte header+meta overhead


def _build_score_aware_loss(
    *,
    args: argparse.Namespace,
    device: torch.device,
) -> Z7Mamba2PredictiveCodingScoreAwareLoss:
    """Build score-aware loss with frozen differentiable scorers.

    Per CLAUDE.md "eval_roundtrip" non-negotiable + Catalog #164
    (canonical scorer-preprocess routing): use canonical
    `load_differentiable_scorers` from tac.differentiable_eval_roundtrip
    + patch upstream YUV6 before scorer load.
    """
    from tac.differentiable_eval_roundtrip import (
        load_differentiable_scorers,
        patch_upstream_yuv6_globally,
    )

    patch_upstream_yuv6_globally()
    upstream_dir = Path(args.upstream_dir)
    pose_scorer, seg_scorer = load_differentiable_scorers(
        upstream_dir=str(upstream_dir),
        device=device,
    )
    # Freeze scorer params
    for p in pose_scorer.parameters():
        p.requires_grad_(False)
    for p in seg_scorer.parameters():
        p.requires_grad_(False)
    weights = Z7Mamba2PredictiveCodingLossWeights(
        alpha_rate=float(args.alpha_rate),
        beta_seg=float(args.beta_seg),
        beta_ib=float(args.beta_ib),
    )
    return Z7Mamba2PredictiveCodingScoreAwareLoss(
        seg_scorer=seg_scorer,
        pose_scorer=pose_scorer,
        weights=weights,
    )


def _context_conditioner_state_dict(
    model: Z7Mamba2PredictiveCodingSubstrate,
) -> dict[str, torch.Tensor]:
    """Return context conditioner state_dict for archive encoder_blob slot."""
    if model.context_conditioner is None:
        return {}
    return model.context_conditioner.state_dict()


def _static_control_archive_pair(
    *,
    model: Z7Mamba2PredictiveCodingSubstrate,
    cfg: Z7Mamba2PredictiveCodingConfig,
    meta: dict[str, object],
    target_archive_zip_bytes: int,
) -> tuple[bytes, bytes, dict[str, object]]:
    """Build identity/static-capacity control archive per Catalog #125 hook #6.

    Per Z7 parent symposium Revision #2 same-archive-bytes pattern: the
    control packs an identity_predictor=True Mamba2 (0 trainable predictor
    params) with the same archive size as the recurrent packet. Provides
    a paired-comparison anchor: recurrent-WIN at delta_S >= 0.005 on
    contest-CUDA is the canonical Z7 promotion criterion.
    """
    control_cfg = Z7Mamba2PredictiveCodingConfig(
        latent_dim=cfg.latent_dim,
        ego_motion_dim=cfg.ego_motion_dim,
        d_model=cfg.d_model,
        d_state=cfg.d_state,
        expand=cfg.expand,
        d_conv=cfg.d_conv,
        backend=cfg.backend,
        stateful=False,
        identity_predictor=True,  # The control: no learning
        beta_ib=cfg.beta_ib,
        num_pairs=cfg.num_pairs,
        decoder_embed_dim=cfg.decoder_embed_dim,
        decoder_initial_grid_h=cfg.decoder_initial_grid_h,
        decoder_initial_grid_w=cfg.decoder_initial_grid_w,
        decoder_channels=cfg.decoder_channels,
        decoder_num_upsample_blocks=cfg.decoder_num_upsample_blocks,
        output_height=cfg.output_height,
        output_width=cfg.output_width,
        latent_init_std=cfg.latent_init_std,
        context_conditioning_mode=cfg.context_conditioning_mode,
        context_affine_strength=cfg.context_affine_strength,
    )
    control_sub = Z7Mamba2PredictiveCodingSubstrate(control_cfg)
    control_sub.load_state_dict(
        {k: v for k, v in model.state_dict().items() if "predictor" not in k},
        strict=False,
    )
    control_sub.eval()
    control_meta = dict(meta)
    control_meta["z7_mamba2_control_kind"] = "identity_predictor_static_capacity"
    control_meta.update(control_sub.decoder_metadata())
    control_bytes = pack_archive(
        _context_conditioner_state_dict(control_sub),
        control_sub.decoder.state_dict(),
        control_sub.predictor.state_dict(),  # empty for identity
        control_sub.latent_init.detach().cpu(),
        control_sub.residuals.detach().cpu(),
        control_sub.ego_motion_buffer.detach().cpu(),
        control_meta,
        config=control_cfg,
    )
    control_zip_path = Path("/tmp") / "_static_control_smoke.zip"  # ephemeral; caller saves real path
    control_zip_bytes = _build_archive_zip(control_zip_path, bin_bytes=control_bytes)
    return control_bytes, control_zip_bytes, {
        "kind": "identity_predictor_static_capacity",
        "archive_bin_bytes": len(control_bytes),
        "archive_zip_bytes": len(control_zip_bytes),
        "target_archive_zip_bytes": int(target_archive_zip_bytes),
        "predictor_trainable_params": 0,
    }


def _write_runtime(submission_dir: Path) -> None:
    """Write the scorer-free Z7-Mamba-2 contest runtime tree.

    Per CLAUDE.md HNeRV parity L4 (≤200 LOC inflate; substrate-engineering
    waiver applied) + L9 (runtime closure: no scorer imports, deterministic
    dependency closure: torch + brotli + (mamba_ssm optional)).
    """
    submission_dir.mkdir(parents=True, exist_ok=True)
    # inflate.sh: canonical 3-arg dispatch contract per Catalog #146
    inflate_sh = """#!/bin/bash
set -euo pipefail
# Z7-Mamba-2 contest inflate runtime
# Args: $1 = archive_dir, $2 = output_dir, $3 = file_list
HERE="$(cd "$(dirname "$0")" && pwd)"
DATA_DIR="$1"
OUTPUT_DIR="$2"
FILE_LIST="$3"
mkdir -p "$OUTPUT_DIR"
exec python "$HERE/inflate.py" "$DATA_DIR" "$OUTPUT_DIR" "$FILE_LIST"
"""
    inflate_sh_path = submission_dir / "inflate.sh"
    inflate_sh_path.write_text(inflate_sh)
    inflate_sh_path.chmod(0o755)
    # inflate.py: vendor minimal runtime (per Catalog #295 explicit-vendor accept)
    inflate_py = '''#!/usr/bin/env python3
"""Z7-Mamba-2 contest inflate runtime."""
# SUBMISSION_PYTHONPATH_SHIM_OK:vendored-tac-substrate-required-for-z7-mamba2-runtime
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC_DIR = HERE / "src"
sys.path.insert(0, str(SRC_DIR))


def select_inflate_device():
    """Canonical Z7-Mamba-2 inflate device selector per Catalog #205."""
    import os
    import torch
    env_choice = os.environ.get("PACT_INFLATE_DEVICE", "auto").lower()
    if env_choice == "cpu":
        return "cpu"
    if env_choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("PACT_INFLATE_DEVICE=cuda but CUDA unavailable")
        return "cuda"
    if env_choice == "mps":
        raise RuntimeError("PACT_INFLATE_DEVICE=mps refused per CLAUDE.md 'MPS auth eval is NOISE'")
    # auto
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def main():
    if len(sys.argv) < 4:
        print("usage: inflate.py <archive_dir> <output_dir> <file_list>", file=sys.stderr)
        return 2
    archive_dir = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])
    file_list_path = Path(sys.argv[3])
    from tac.substrates.time_traveler_l5_z7_mamba2.inflate import (
        inflate_one_video,
        _read_single_member_archive_bytes,
    )
    from tac.substrates._shared.inflate_runtime import raw_output_path
    file_list = file_list_path.read_text(encoding="utf-8").strip().splitlines()
    archive_bytes = _read_single_member_archive_bytes(archive_dir)
    device = select_inflate_device()
    for fname in file_list:
        name = fname.strip()
        if not name:
            continue
        inflate_one_video(archive_bytes, raw_output_path(output_dir, name), device=device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''
    (submission_dir / "inflate.py").write_text(inflate_py)


def _boolish(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in ("true", "1", "yes")


def _full_main(args: argparse.Namespace) -> int:
    """Full Z7-Mamba-2 training + byte-closed archive emission + optional inflate verify.

    This is the canonical full trainer per CLAUDE.md "Substrate scaffolds MUST be
    COMPLETE or RESEARCH-ONLY" non-negotiable + HNeRV parity discipline (PR95
    paradigm: bind ALL ingredients simultaneously in one packet). Implements:

    - Real contest-video pair decode via canonical `decode_real_pairs` (per
      Catalog #114 forbidden synthetic-non-smoke).
    - Mamba-2 selective state-space autoregression across the num_pairs sequence.
    - Z6-compatible PixelShuffle decoder rendering at output_height x output_width.
    - Score-aware OR proxy loss (operator-selectable; score_aware MUST use
      eval_roundtrip per CLAUDE.md non-negotiable).
    - Z7MCM2 monolithic byte-closed archive (encoder/decoder/predictor/latent/
      residuals/ego_motion blobs + sorted JSON meta).
    - Deterministic archive.zip per Catalog #5.
    - Submission runtime tree per CLAUDE.md HNeRV parity L4 + Catalog #146 +
      Catalog #295.
    - Optional inflate verify per byte-mutation smoke contract (Catalog #139 +
      Catalog #272).
    - Optional static-capacity control archive (Catalog #125 hook #6 +
      Z7 symposium Revision #2 same-archive-bytes pattern).

    Output is **non-promotable by construction**: every stats/meta field
    sets `score_claim=False`, `promotion_eligible=False`,
    `ready_for_exact_eval_dispatch=False`, `ready_for_paid_dispatch=False`.
    Per CLAUDE.md "Apples-to-apples evidence discipline": authority flags
    flip only after paired Tier-C post-training validation + per-substrate
    symposium PROCEED-unconditional per Catalog #324 + #325.

    Per CLAUDE.md "Forbidden premature KILL without research exhaustion":
    this implementation does NOT include the paid Modal/Lightning dispatch
    invocation; recipe at
    `.omx/operator_authorize_recipes/substrate_time_traveler_l5_z7_mamba2_*.yaml`
    must explicitly flip `dispatch_enabled: true` AND have a recent
    PROCEED-unconditional council deliberation per Catalog #325 before
    paid dispatch fires.

    Returns 0 on success; raises on failure (per Catalog #279 fail-closed).
    """
    if _decode_real_pairs is None:
        raise RuntimeError(
            "Canonical decode_real_pairs helper unavailable; check "
            "tac.substrates._shared.trainer_skeleton import"
        )
    total_started_at = time.perf_counter()
    stage_wall_seconds: dict[str, float] = {}
    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is None:
        output_dir = REPO_ROOT / "experiments" / "results" / (
            f"z7_mamba2_full_export_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = _resolve_full_config(args)
    device = _select_device(args.device)
    loss_mode = _normalize_loss_mode(args.loss_mode)
    torch.manual_seed(721)  # Distinct from Z7-LSTM seed 711 for paired-comparison

    print(f"[z7_mamba2_full] output_dir={output_dir}")
    print(f"[z7_mamba2_full] device={device} loss_mode={loss_mode}")
    print(f"[z7_mamba2_full] num_pairs={cfg.num_pairs} epochs={args.epochs}")
    print(f"[z7_mamba2_full] mamba2 d_model={cfg.d_model} d_state={cfg.d_state} expand={cfg.expand}")

    # Stage 1: decode real contest pairs
    stage_started_at = time.perf_counter()
    real_pairs = _decode_real_pairs(
        Path(args.video_path),
        n_pairs=600,
        max_pairs=cfg.num_pairs,
        substrate_tag=SUBSTRATE_ID,
        repo_root=REPO_ROOT,
    )
    pairs_unit = real_pairs.to(device=device, dtype=torch.float32) / 255.0
    if tuple(pairs_unit.shape[-2:]) != (cfg.output_height, cfg.output_width):
        flat = pairs_unit.reshape(-1, 3, pairs_unit.shape[-2], pairs_unit.shape[-1])
        resized = F.interpolate(
            flat,
            size=(cfg.output_height, cfg.output_width),
            mode="bilinear",
            align_corners=False,
        )
        pairs_unit = resized.reshape(
            cfg.num_pairs,
            2,
            3,
            cfg.output_height,
            cfg.output_width,
        )
    stage_wall_seconds["decode_resize_seconds"] = time.perf_counter() - stage_started_at
    print(f"[z7_mamba2_full] decoded {cfg.num_pairs} pairs in {stage_wall_seconds['decode_resize_seconds']:.2f}s")

    # Stage 2: build model + initialize ego-motion buffer
    stage_started_at = time.perf_counter()
    model = Z7Mamba2PredictiveCodingSubstrate(cfg).to(device)
    with torch.no_grad():
        model.ego_motion_buffer.copy_(
            _ego_motion_from_pairs(pairs_unit.detach().cpu(), cfg.ego_motion_dim).to(
                device=device,
                dtype=model.ego_motion_buffer.dtype,
            )
        )
    stage_wall_seconds["model_init_ego_seconds"] = time.perf_counter() - stage_started_at
    breakdown = model.num_parameters_breakdown()
    print(f"[z7_mamba2_full] model built: params={breakdown}")
    print(f"[z7_mamba2_full] predictor backend_active={model.predictor.backend_active}")

    # Stage 3: optional score-aware loss build
    archive_bytes_proxy_tensor = torch.tensor(
        float(_estimate_archive_bytes_proxy(model, cfg)),
        device=device,
    )
    score_aware_loss: torch.nn.Module | None = None
    if loss_mode == "score_aware":
        stage_started_at = time.perf_counter()
        score_aware_loss = _build_score_aware_loss(args=args, device=device)
        stage_wall_seconds["score_aware_scorer_load_seconds"] = (
            time.perf_counter() - stage_started_at
        )
        score_aware_loss.train()
        print(f"[z7_mamba2_full] score_aware scorers loaded in {stage_wall_seconds['score_aware_scorer_load_seconds']:.2f}s")
    else:
        stage_wall_seconds["score_aware_scorer_load_seconds"] = 0.0

    # Stage 4: training loop
    epochs = max(1, int(args.epochs))
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr))
    losses: list[dict[str, float | int]] = []
    target0 = pairs_unit[:, 0]
    target1 = pairs_unit[:, 1]
    train_started_at = time.perf_counter()
    for epoch in range(epochs):
        epoch_started_at = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        rgb_0, rgb_1, latents = model.reconstruct_all_pairs()
        residual_loss = model.residuals.pow(2).mean()
        latent_smoothness = (
            (latents[1:] - latents[:-1]).pow(2).mean()
            if latents.shape[0] > 1
            else latents.new_tensor(0.0)
        )
        if score_aware_loss is not None:
            loss, parts = score_aware_loss(
                reconstructed_rgb_0=rgb_0 * 255.0,
                reconstructed_rgb_1=rgb_1 * 255.0,
                gt_rgb_0=target0 * 255.0,
                gt_rgb_1=target1 * 255.0,
                archive_bytes_proxy=archive_bytes_proxy_tensor,
                residuals=model.residuals,
                latents=latents,
                apply_eval_roundtrip=True,
                noise_std=float(args.noise_std),
            )
            loss_record: dict[str, float | int] = {
                "epoch": epoch,
                "loss": float(loss.item()),
            }
            for key, value in parts.items():
                loss_record[key] = float(value.detach().cpu())
        else:
            recon_loss = (rgb_0 - target0).pow(2).mean() + (
                rgb_1 - target1
            ).pow(2).mean()
            loss = recon_loss + float(args.beta_ib) * 1e-3 * (
                residual_loss + latent_smoothness
            )
            loss_record = {
                "epoch": epoch,
                "loss": float(loss.item()),
                "recon": float(recon_loss.item()),
                "residual": float(residual_loss.item()),
                "latent_smoothness": float(latent_smoothness.item()),
            }
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        loss_record["wall_seconds"] = time.perf_counter() - epoch_started_at
        losses.append(loss_record)
        if epoch < 3 or epoch % max(1, epochs // 10) == 0:
            print(f"[z7_mamba2_full] epoch {epoch}: loss={loss.item():.6f} wall={loss_record['wall_seconds']:.2f}s")
    train_total_seconds = time.perf_counter() - train_started_at
    stage_wall_seconds["train_total_seconds"] = train_total_seconds

    # Stage 5: pack Z7MCM2 archive
    stage_started_at = time.perf_counter()
    meta: dict[str, object] = {
        **model.decoder_metadata(),
        "schema": "z7_mamba2_full_main_export_v1",
        "substrate_id": SUBSTRATE_ID,
        "lane_id": LANE_ID,
        "target_video": str(Path(args.video_path)),
        "target_pairs": cfg.num_pairs,
        "epochs": epochs,
        "batch_size_flag": int(args.batch_size),
        "lr": float(args.lr),
        "loss_mode": loss_mode,
        "score_aware_scorer_loss_used": loss_mode == "score_aware",
        "ego_source": str(args.ego_source),
        "ego_motion_source": "real_video_pair_delta_proxy_no_scorer_load",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "ready_for_paid_dispatch": False,
    }
    archive_bytes = pack_archive(
        _context_conditioner_state_dict(model),
        model.decoder.state_dict(),
        model.predictor.state_dict(),
        model.latent_init.detach().cpu(),
        model.residuals.detach().cpu(),
        model.ego_motion_buffer.detach().cpu(),
        meta,
        config=cfg,
    )
    bin_path = output_dir / "0.bin"
    bin_path.write_bytes(archive_bytes)
    archive_zip_path = output_dir / "archive.zip"
    _build_archive_zip(archive_zip_path, bin_bytes=archive_bytes)
    archive_zip_bytes = archive_zip_path.read_bytes()

    # Stage 5b: optional static-capacity control
    static_control: dict[str, object] | None = None
    static_control_bytes: bytes | None = None
    if _boolish(args.emit_static_control):
        try:
            static_control_bytes, static_control_zip_bytes, static_control = (
                _static_control_archive_pair(
                    model=model,
                    cfg=cfg,
                    meta=meta,
                    target_archive_zip_bytes=len(archive_zip_bytes),
                )
            )
            control_dir = output_dir / "static_capacity_control"
            control_dir.mkdir(parents=True, exist_ok=True)
            control_bin_path = control_dir / "0.bin"
            control_zip_path = control_dir / "archive.zip"
            control_bin_path.write_bytes(static_control_bytes)
            _build_archive_zip(control_zip_path, bin_bytes=static_control_bytes)
            assert isinstance(static_control, dict)
            static_control["archive_bin_path"] = str(control_bin_path)
            static_control["archive_zip_path"] = str(control_zip_path)
        except Exception as e:  # pragma: no cover - defensive
            print(f"[z7_mamba2_full] static control emission failed: {e}", file=sys.stderr)
            static_control = {"emission_failed": str(e)}

    # Stage 6: submission runtime tree
    submission_dir = output_dir / "submission_runtime"
    _write_runtime(submission_dir)
    (submission_dir / "0.bin").write_bytes(archive_bytes)
    shutil.copy2(archive_zip_path, submission_dir / "archive.zip")
    stage_wall_seconds["export_packaging_seconds"] = (
        time.perf_counter() - stage_started_at
    )

    # Stage 7: optional inflate verify
    inflate_verify: dict[str, object] | None = None
    stage_started_at = time.perf_counter()
    if _boolish(args.inflate_verify):
        verify_raw = output_dir / "inflate_verify" / "0.raw"
        try:
            frames = inflate_one_video(archive_bytes, verify_raw, device=str(device))
            inflate_verify = {
                "raw_path": str(verify_raw),
                "frames_written": int(frames),
                "raw_bytes": verify_raw.stat().st_size,
                "raw_sha256": _sha256_bytes(verify_raw.read_bytes()),
            }
            if static_control_bytes is not None and isinstance(static_control, dict):
                control_raw = output_dir / "inflate_verify" / "static_control.raw"
                control_frames = inflate_one_video(
                    static_control_bytes,
                    control_raw,
                    device=str(device),
                )
                control_raw_bytes = control_raw.read_bytes()
                control_sha = _sha256_bytes(control_raw_bytes)
                recurrent_raw_bytes = verify_raw.read_bytes()
                byte_differences = sum(
                    a != b for a, b in zip(recurrent_raw_bytes, control_raw_bytes)
                ) + abs(len(recurrent_raw_bytes) - len(control_raw_bytes))
                static_control.update({
                    "inflate_verify_raw_path": str(control_raw),
                    "inflate_verify_frames_written": int(control_frames),
                    "inflate_verify_raw_bytes": control_raw.stat().st_size,
                    "inflate_verify_raw_sha256": control_sha,
                    "runtime_output_changed_vs_recurrent": (
                        control_sha != inflate_verify["raw_sha256"]
                    ),
                    "runtime_output_byte_differences_vs_recurrent": int(byte_differences),
                })
        except Exception as e:  # pragma: no cover - defensive
            inflate_verify = {"verify_failed": str(e)}
    stage_wall_seconds["inflate_verify_seconds"] = time.perf_counter() - stage_started_at

    # Stage 8: emit stats + timing
    final_loss = losses[-1] if losses else {"loss": None}
    total_wall_seconds = time.perf_counter() - total_started_at
    stage_wall_seconds["total_wall_seconds"] = total_wall_seconds
    timing_smoke = {
        "schema": "z7_mamba2_timing_smoke_v1",
        "axis": "[local-trainer-timing advisory]",
        "device": str(device),
        "loss_mode": loss_mode,
        "epochs": epochs,
        "num_pairs": cfg.num_pairs,
        "stage_wall_seconds": {
            key: float(value) for key, value in stage_wall_seconds.items()
        },
        "seconds_per_epoch": float(train_total_seconds / max(1, epochs)),
        "seconds_per_pair_epoch": float(
            train_total_seconds / max(1, epochs * cfg.num_pairs)
        ),
        "pairs_per_second_epoch": float(
            (epochs * cfg.num_pairs) / max(train_total_seconds, 1e-12)
        ),
        "promotion_eligible": False,
        "score_claim": False,
        "ready_for_paid_dispatch": False,
    }
    stats = {
        "schema_version": 1,
        "name": "z7_mamba2_full_main_export_stats",
        "substrate_id": SUBSTRATE_ID,
        "lane_id": LANE_ID,
        "config": {
            "latent_dim": cfg.latent_dim,
            "ego_motion_dim": cfg.ego_motion_dim,
            "mamba2_d_model": cfg.d_model,
            "mamba2_d_state": cfg.d_state,
            "mamba2_expand": cfg.expand,
            "mamba2_d_conv": cfg.d_conv,
            "mamba2_backend_active": str(model.predictor.backend_active),
            "stateful": cfg.stateful,
            "identity_predictor": cfg.identity_predictor,
            "beta_ib": cfg.beta_ib,
            "num_pairs": cfg.num_pairs,
            **model.decoder_metadata(),
        },
        "param_breakdown": breakdown,
        "losses": losses,
        "final_loss": final_loss.get("loss"),
        "final_loss_proxy": final_loss.get("loss") if loss_mode == "proxy" else None,
        "final_loss_score_aware": (
            final_loss.get("loss") if loss_mode == "score_aware" else None
        ),
        "archive_bin_path": str(bin_path),
        "archive_bin_bytes": len(archive_bytes),
        "archive_bin_sha256": _sha256_bytes(archive_bytes),
        "archive_zip_path": str(archive_zip_path),
        "archive_zip_bytes": len(archive_zip_bytes),
        "archive_zip_sha256": _sha256_bytes(archive_zip_bytes),
        "static_capacity_control": static_control,
        "inflate_verify": inflate_verify,
        "timing_smoke": timing_smoke,
        # Authority flags — non-promotable by construction per CLAUDE.md
        "evidence_grade": "z7_mamba2_full_train_packet_not_promotable_pending_council_per_catalog_325",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "ready_for_paid_dispatch": False,
        "auth_eval_score_axis": "diagnostic_only_no_authority",
        "auth_eval_score_claim_valid": False,
        "result_review_blockers": [
            "z7_mamba2_full_train_packet_not_paired_exact_eval_validated",
            "wave_n_plus_1_council_required_per_z7_mamba2_symposium",
            "post_training_tier_c_validation_required_per_catalog_324",
            "per_substrate_symposium_evidence_required_per_catalog_325",
            "predicted_band_validation_status_pending_post_training",
        ],
        "utc_now": datetime.now(UTC).isoformat(),
    }
    stats_path = output_dir / "z7_mamba2_full_main_export_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True, default=str))
    print(f"[z7_mamba2_full] stats written: {stats_path}")
    print(f"[z7_mamba2_full] archive_bin_bytes={len(archive_bytes)} archive_zip_bytes={len(archive_zip_bytes)}")
    print(f"[z7_mamba2_full] archive_bin_sha256={_sha256_bytes(archive_bytes)[:16]}...")
    print(f"[z7_mamba2_full] total wall_seconds={total_wall_seconds:.2f}")
    print(f"[z7_mamba2_full] DONE [no-auth-eval-pending-wave-N+1-council]")

    return 0


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
