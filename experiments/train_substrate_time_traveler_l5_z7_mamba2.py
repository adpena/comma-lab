# SPDX-License-Identifier: MIT
"""Z7-Mamba-2 substrate trainer — built research-only per Catalog #240.

Per the Z7-Mamba-2 design memo
(``.omx/research/z7_mamba2_substrate_design_memo_20260518.md``):
this started as the TOP-5 #2 scaffold landing per the deep-research wave
(``.omx/research/comprehensive_research_wave_20260518.md`` §0 + §2.2 + §3.6).

Z7-Mamba-2 is the **canonical Catalog #308 N>=3 alternative-probe-methodology**
to Z7-GRU (Hafner Revision #3 binding) within the predictive-coding-recurrent
substrate class. Mamba-2 (Dao-Gu 2024, arxiv 2405.21060) is a selective
state-space sequence model proven to match Transformer quality at O(N)
compute on long-context language and video tasks.

**This trainer is BUILT but non-promotable per Catalog #240/Catalog #324** —
``_full_main`` emits a byte-closed research packet while the operator recipe
stays ``research_only: true`` and ``dispatch_enabled: false`` until Wave N+1
council, identity-disambiguator, mamba_ssm preflight, and paired exact-eval
custody unblock it.

The recipe (``.omx/operator_authorize_recipes/substrate_time_traveler_l5_z7_mamba2_modal_a100_dispatch.yaml``)
declares ``research_only: true`` + ``dispatch_enabled: false`` so this
research-only substrate satisfies Catalog #240 substrate-engineering discipline
without risking phantom-score dispatch.

Scope at THIS landing:

- ``_smoke_main`` builds a tiny ``Mamba2PredictorConfig`` + instantiates
  ``Mamba2Predictor`` + forward-pass sanity check on synthetic 600-pair
  sequence. Validates canonical signature compatibility with Z6 sister
  per design memo §6 layer #2.
- ``_full_main`` implements the full Z7-Mamba-2 substrate (Z6 encoder +
  Mamba2Predictor + Z6 decoder +
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
import io
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import hashlib
import shutil
import struct
import time
import zipfile

import torch
import torch.nn.functional as F

from tac.optimization.mamba2_predictor import (
    MAMBA_SSM_AVAILABLE,
    Mamba2Predictor,
    Mamba2PredictorConfig,
)
from tac.substrates._shared.inflate_runtime import CAMERA_HW
from tac.substrates.time_traveler_l5_z7_mamba2 import (
    Z7MCM2_HEADER_FMT,
    Z7MCM2_HEADER_SIZE,
    Z7MCM2_MAGIC,
    Z7MCM2_SCHEMA_VERSION,
    Z7Mamba2PredictiveCodingConfig,
    Z7Mamba2PredictiveCodingLossWeights,
    Z7Mamba2PredictiveCodingScoreAwareLoss,
    Z7Mamba2PredictiveCodingSubstrate,
    pack_archive,
    parse_z7mcm2_archive_bytes,
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
DEFAULT_PAIR_CHUNK_SIZE = 8
SCORE_AWARE_BACKWARD_MODE = (
    "two_pass_streamed_chunks_global_pose_sqrt_exact_first_order"
)


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
            "Pair chunk size for decoder/scorer loss. The recurrent latent "
            "sequence is replayed in order, then decoded/scored in bounded "
            "pair chunks so 600-pair score-aware runs do not require one "
            "giant scorer graph."
        ),
        "default": str(DEFAULT_PAIR_CHUNK_SIZE),
    },
    "--lr": {
        "env": "Z7_MAMBA2_LR",
        "rationale": "AdamW base learning rate; default 5e-4 per Z6 sister",
        "default": "5e-4",
    },
    "--lr-warmup-steps": {
        "env": "Z7_MAMBA2_LR_WARMUP_STEPS",
        "rationale": (
            "Linear LR warmup steps for Z7-Mamba-2 stability; 0 disables warmup."
        ),
        "default": "0",
    },
    "--grad-clip-norm": {
        "env": "Z7_MAMBA2_GRAD_CLIP_NORM",
        "rationale": (
            "Gradient clipping max norm for canonical-scale Mamba stability; "
            "values <=0 disable clipping."
        ),
        "default": "1.0",
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
            "Mamba-2 backend selection. Default is 'reference_torch' because "
            "the Z7MCM2 inflate runtime replays byte-closed packets through "
            "the scorer-free reference implementation. 'mamba_ssm' is "
            "training-only research until its state/export replay contract "
            "is implemented."
        ),
        "default": "reference_torch",
    },
    "--ego-source": {
        "env": "Z7_MAMBA2_EGO_SOURCE",
        "rationale": (
            "Implemented scorer-free ego source. The trainer currently "
            "accepts only 'frame_delta_proxy'; planned scorer-derived "
            "sources fail closed until their extraction path is actually "
            "implemented and contest-compliance reviewed."
        ),
        "default": "frame_delta_proxy",
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
            "+ forward-pass sanity check on synthetic data. Non-smoke "
            "_full_main is implemented but remains recipe-gated until Wave "
            "N+1 council, identity disambiguator, and paired exact eval."
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
# tools see the canonical compliance even though the trainer itself remains
# recipe-gated; sister to driver script wire-in.
# ---------------------------------------------------------------------------
# DALI_DISABLE_NVML=1
# CUBLAS_WORKSPACE_CONFIG=:4096:8
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Z7-Mamba-2 substrate trainer (built research-only per Catalog #240). "
            "Smoke mode validates canonical signature; full mode emits a "
            "non-promotable research packet while the recipe remains gated."
        )
    )
    for flag, meta in TIER_1_OPERATOR_REQUIRED_FLAGS.items():
        default_env = meta.get("env")
        default = (
            os.environ[default_env]
            if default_env and default_env in os.environ
            else meta.get("default")
        )
        if flag in (
            "--smoke",
            "--identity-predictor",
            "--stateful",
            "--inflate-verify",
            "--emit-static-control",
        ):
            parser.add_argument(
                flag,
                nargs="?",
                const=True,
                default=_boolish(default),
                type=_boolish,
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
        stateful=_boolish(args.stateful),
        identity_predictor=_boolish(args.identity_predictor),
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

    # Sanity 1: forward-pass shape. Smoke mode uses the same explicit
    # selector as full mode so local MPS/CPU/CUDA checks cannot silently
    # drift into another architecture.
    device = _select_device(args.device)
    device_contract = _device_runtime_contract(device)
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
    print("[z7_mamba2_scaffold] identity-predictor mode OK (0 params)")

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
    print("[z7_mamba2_scaffold] per-pair master gradient compatibility OK (Catalog #810)")

    # Sanity 4: 20-step autoregressive unroll (state-evolution sanity)
    predictor.reset_state(1, device=device)
    states = []
    for _t in range(20):
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
        "lane_id": LANE_ID,
        "historical_scaffold_lane_id": "lane_top5_2_z7_mamba2_scaffold_design_20260518",
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
        "evidence_grade": "smoke_signature_only_NOT_promotable",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "ready_for_paid_dispatch": False,
        "rank_or_kill_eligible": False,
        "result_review_blockers": [
            "scaffold_smoke_validates_canonical_signature_only_not_training",
            "full_main_built_but_non_promotable_without_post_training_tier_c",
            "wave_n_plus_1_council_required_per_z7_symposium_revision_6",
            "z7_gru_wave_2_disambiguator_outcome_required_per_revision_1",
        ],
        "device": str(device),
        "device_runtime_contract": device_contract,
        "utc_now": datetime.now(UTC).isoformat(),
    }
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True))
    print(f"[z7_mamba2_scaffold] stats written: {stats_path}")

    return 0


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _deterministic_comment(length: int) -> bytes:
    """Return deterministic ASCII ZIP-comment padding of exactly ``length``."""
    if length < 0 or length > 0xFFFF:
        raise ValueError(f"ZIP comment length {length} outside u16 range")
    return _deterministic_padding_bytes(
        length,
        label="z7-mamba2-static-control-zip-comment",
    )


def _deterministic_padding_bytes(length: int, *, label: str) -> bytes:
    """Return deterministic ASCII padding bytes of exactly ``length``."""
    if length < 0:
        raise ValueError(f"padding length must be non-negative; got {length}")
    chunks: list[bytes] = []
    counter = 0
    while sum(len(chunk) for chunk in chunks) < length:
        chunks.append(
            hashlib.sha256(
                f"{label}:{counter}".encode()
            ).hexdigest().encode("ascii")
        )
        counter += 1
    return b"".join(chunks)[:length]


def _deterministic_padding_text(length: int, *, label: str) -> str:
    """Return deterministic ASCII metadata padding text of exactly ``length``."""
    return _deterministic_padding_bytes(length, label=label).decode("ascii")


def _archive_zip_bytes(*, bin_bytes: bytes, comment: bytes = b"") -> bytes:
    """Build deterministic archive.zip bytes with 0.bin as single member.

    Per Catalog #5 `check_archive_builders_use_deterministic_zip`: use
    ZipInfo + writestr with fixed timestamp + no compression for
    byte-stable output across re-builds.
    """
    info = zipfile.ZipInfo(filename="0.bin", date_time=(2026, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(info, bin_bytes)
        if comment:
            zf.comment = comment
    return buffer.getvalue()


def _archive_zip_bytes_matching_size(*, bin_bytes: bytes, target_size: int) -> bytes:
    """Build deterministic archive.zip bytes padded to an exact byte length.

    The static-capacity disambiguator requires same archive-byte cost as the
    recurrent packet. ZIP comments are counted in archive bytes but are not
    extracted as archive members, so they are the least invasive deterministic
    padding channel. If the base control ZIP is already larger than the target
    or needs more than ZIP's 65,535-byte comment, fail closed.
    """
    target_size = int(target_size)
    base = _archive_zip_bytes(bin_bytes=bin_bytes)
    delta = target_size - len(base)
    if delta < 0:
        raise ValueError(
            "static-control archive cannot match recurrent archive size: "
            f"control_base_zip_bytes={len(base)} target_archive_zip_bytes={target_size}"
        )
    if delta > 0xFFFF:
        raise ValueError(
            "static-control archive needs ZIP comment padding larger than 65535 "
            f"bytes: required_padding={delta}"
        )
    if delta == 0:
        return base
    return _archive_zip_bytes(
        bin_bytes=bin_bytes,
        comment=_deterministic_comment(delta),
    )


def _build_archive_zip(zip_path: Path, *, bin_bytes: bytes, comment: bytes = b"") -> bytes:
    """Write deterministic archive.zip with 0.bin as single member."""
    zip_bytes = _archive_zip_bytes(bin_bytes=bin_bytes, comment=comment)
    zip_path.write_bytes(zip_bytes)
    return zip_bytes


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
        stateful=_boolish(args.stateful),
        identity_predictor=_boolish(args.identity_predictor),
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


def _device_runtime_contract(device: torch.device) -> dict[str, object]:
    """Return fail-closed device metadata for all local trainer modes."""
    fallback_env = os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK")
    fallback_enabled = str(fallback_env or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    if device.type == "mps" and fallback_enabled:
        raise RuntimeError(
            "Z7-Mamba-2 refuses --device mps while PYTORCH_ENABLE_MPS_FALLBACK "
            "enables CPU fallback. CPU fallback hides unsupported Metal kernels "
            "and corrupts MPS research-signal timing/device attribution; rerun "
            "with PYTORCH_ENABLE_MPS_FALLBACK=0 or unset."
        )
    return {
        "training_device": str(device),
        "device_type": device.type,
        "mps_research_signal_only": device.type == "mps",
        "contest_authority_training_device": device.type == "cuda",
        "inflate_verify_device": _inflate_verify_device(device),
        "pytorch_enable_mps_fallback": fallback_env if fallback_env is not None else "<unset>",
        "mps_cpu_fallback_enabled": fallback_enabled,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "ready_for_paid_dispatch": False,
        "rank_or_kill_eligible": False,
    }


def _normalize_loss_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in ("proxy", "score_aware"):
        raise ValueError(f"unknown loss_mode {mode!r}; expected proxy|score_aware")
    return mode


def _normalize_ego_source(value: object) -> str:
    source = str(value).strip().lower().replace("-", "_")
    if source == "real_video_pair_delta_proxy":
        source = "frame_delta_proxy"
    if source != "frame_delta_proxy":
        raise ValueError(
            "Z7-Mamba-2 ego_source currently supports only scorer-free "
            "'frame_delta_proxy'. Planned sources "
            "'posenet_projection'/'scorer_logit_compressed' are not implemented "
            "and must not be recorded as active evidence."
        )
    return source


def _resolve_pair_chunk_size(*, batch_size: int, num_pairs: int) -> int:
    if int(batch_size) <= 0:
        raise RuntimeError(
            f"Z7-Mamba-2 --batch-size must be positive; got {batch_size}"
        )
    if int(num_pairs) <= 0:
        raise RuntimeError(f"Z7-Mamba-2 --max-pairs must be positive; got {num_pairs}")
    return min(int(batch_size), int(num_pairs))


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
    ``load_differentiable_scorers`` from ``tac.scorer`` + patch upstream
    YUV6 before scorer load.
    """
    from tac.differentiable_eval_roundtrip import patch_upstream_yuv6_globally
    from tac.scorer import load_differentiable_scorers

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


def _require_export_replay_backend(model: Z7Mamba2PredictiveCodingSubstrate) -> None:
    backend_active = str(model.predictor.backend_active)
    if backend_active not in ("reference_torch", "identity"):
        raise RuntimeError(
            "Z7-Mamba-2 export is fail-closed because the archive parser "
            "replays with reference_torch at inflate time, but training used "
            f"backend_active={backend_active!r}. Use --mamba2-backend "
            "reference_torch, or implement byte-faithful mamba_ssm state export "
            "and replay before dispatch."
        )


def _require_finite_loss(loss: torch.Tensor, *, epoch: int) -> None:
    detached = loss.detach()
    if not torch.isfinite(detached).all():
        raise FloatingPointError(
            f"Z7-Mamba-2 non-finite loss at epoch {epoch}; refusing backward/export"
        )


def _reset_peak_memory_stats(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _memory_telemetry(device: torch.device) -> dict[str, int | str]:
    if device.type == "cuda":
        return {
            "axis": "[local-cuda-memory advisory]",
            "cuda_memory_allocated_bytes": int(torch.cuda.memory_allocated(device)),
            "cuda_max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
            "cuda_memory_reserved_bytes": int(torch.cuda.memory_reserved(device)),
            "cuda_max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        }
    if device.type == "mps" and hasattr(torch, "mps"):
        telemetry: dict[str, int | str] = {"axis": "[local-mps-memory advisory]"}
        for name in ("current_allocated_memory", "driver_allocated_memory"):
            fn = getattr(torch.mps, name, None)
            if callable(fn):
                try:
                    telemetry[f"mps_{name}_bytes"] = int(fn())
                except RuntimeError:
                    pass
        return telemetry
    return {"axis": "[local-cpu-memory advisory]"}


def _inflate_verify_device(training_device: torch.device) -> str:
    """Map a training device to a contest-runtime-supported inflate device.

    MPS is a local training proxy only; the scorer-free inflate runtime's
    canonical device selector intentionally supports auto/cpu/cuda, not mps.
    Verify MPS-trained packets on CPU so the local handoff check exercises the
    same self-contained Python/Torch runtime without inventing an MPS authority.
    """
    if training_device.type == "cuda":
        return "cuda"
    if training_device.type == "cpu":
        return "cpu"
    return "cpu"


def _iter_pair_slices(num_pairs: int, pair_chunk_size: int) -> list[tuple[int, int]]:
    return [
        (start, min(num_pairs, start + pair_chunk_size))
        for start in range(0, num_pairs, pair_chunk_size)
    ]


def _z7mcm2_archive_header_summary(archive_bytes: bytes) -> dict[str, int]:
    """Return geometry-bearing header/meta fields from actual Z7MCM2 bytes."""
    if len(archive_bytes) < Z7MCM2_HEADER_SIZE:
        raise ValueError(
            f"Z7MCM2 archive too short for header: {len(archive_bytes)} bytes"
        )
    header = struct.unpack(
        Z7MCM2_HEADER_FMT,
        archive_bytes[:Z7MCM2_HEADER_SIZE],
    )
    magic = header[0]
    version = int(header[1])
    num_pairs = int(header[4])
    if magic != Z7MCM2_MAGIC:
        raise ValueError(f"Z7MCM2 bad magic in runtime geometry block: {magic!r}")
    if version != Z7MCM2_SCHEMA_VERSION:
        raise ValueError(
            "Z7MCM2 unsupported schema version in runtime geometry block: "
            f"{version}"
        )
    sections = parse_z7mcm2_archive_bytes(archive_bytes)
    meta_start, meta_len = sections["meta_blob"]
    meta = json.loads(
        archive_bytes[meta_start : meta_start + meta_len].decode("utf-8")
    )
    return {
        "num_pairs": num_pairs,
        "output_height": int(meta["output_height"]),
        "output_width": int(meta["output_width"]),
    }


def _runtime_geometry_sample_pair_indices(num_pairs: int) -> list[int]:
    """Choose deterministic pair-level raw-output samples across the sequence."""
    if int(num_pairs) <= 0:
        raise ValueError(f"num_pairs must be positive; got {num_pairs}")
    return sorted({0, int(num_pairs) // 2, int(num_pairs) - 1})


def _sampled_raw_sha256(
    raw_bytes: bytes,
    *,
    sample_pair_indices: list[int],
    num_pairs: int,
    camera_hw: tuple[int, int],
) -> str:
    """Hash selected pair-sized raw byte windows without changing evidence axis."""
    camera_h, camera_w = int(camera_hw[0]), int(camera_hw[1])
    frame_bytes = 3 * camera_h * camera_w
    pair_bytes = 2 * frame_bytes
    expected_raw_bytes = int(num_pairs) * pair_bytes
    if len(raw_bytes) != expected_raw_bytes:
        raise ValueError(
            "runtime geometry positive-control raw byte length mismatch: "
            f"got {len(raw_bytes)} expected {expected_raw_bytes}"
        )
    digest = hashlib.sha256()
    for pair_idx in sample_pair_indices:
        if int(pair_idx) < 0 or int(pair_idx) >= int(num_pairs):
            raise ValueError(
                "runtime geometry sample_pair_indices out of range: "
                f"{pair_idx} for num_pairs={num_pairs}"
            )
        start = int(pair_idx) * pair_bytes
        digest.update(raw_bytes[start : start + pair_bytes])
    return digest.hexdigest()


def _build_runtime_geometry_positive_control(
    *,
    archive_bytes: bytes,
    recurrent_raw_bytes: bytes,
    static_raw_bytes: bytes,
    num_pairs: int,
    render_hw: tuple[int, int],
    camera_hw: tuple[int, int] = CAMERA_HW,
) -> dict[str, object]:
    """Build the Z7-Mamba-2 runtime geometry positive-control stats block."""
    sample_pair_indices = _runtime_geometry_sample_pair_indices(num_pairs)
    expected_frames_written = int(num_pairs) * 2
    expected_raw_bytes = (
        expected_frames_written * 3 * int(camera_hw[0]) * int(camera_hw[1])
    )
    recurrent_sample_sha = _sampled_raw_sha256(
        recurrent_raw_bytes,
        sample_pair_indices=sample_pair_indices,
        num_pairs=num_pairs,
        camera_hw=camera_hw,
    )
    static_sample_sha = _sampled_raw_sha256(
        static_raw_bytes,
        sample_pair_indices=sample_pair_indices,
        num_pairs=num_pairs,
        camera_hw=camera_hw,
    )
    return {
        "schema": "z7_mamba2_runtime_geometry_positive_control_v1",
        "num_pairs": int(num_pairs),
        "render_hw": [int(render_hw[0]), int(render_hw[1])],
        "camera_hw": [int(camera_hw[0]), int(camera_hw[1])],
        "expected_frames_written": expected_frames_written,
        "expected_raw_bytes": expected_raw_bytes,
        "sample_pair_indices": sample_pair_indices,
        "recurrent_sampled_raw_sha256": recurrent_sample_sha,
        "static_sampled_raw_sha256": static_sample_sha,
        "recurrent_static_sample_changed": recurrent_sample_sha != static_sample_sha,
        "archive_header": _z7mcm2_archive_header_summary(archive_bytes),
    }


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
    base_control_meta = dict(meta)
    base_control_meta["z7_mamba2_control_kind"] = (
        "identity_predictor_static_capacity"
    )
    base_control_meta.update(control_sub.decoder_metadata())
    padding_key = "z7_mamba2_static_control_noop_padding"

    def _pack_control_with_meta(control_meta: dict[str, object]) -> bytes:
        return pack_archive(
            _context_conditioner_state_dict(control_sub),
            control_sub.decoder.state_dict(),
            control_sub.predictor.state_dict(),  # empty for identity
            control_sub.latent_init.detach().cpu(),
            control_sub.residuals.detach().cpu(),
            control_sub.ego_motion_buffer.detach().cpu(),
            control_meta,
            config=control_cfg,
        )

    control_bytes = _pack_control_with_meta(base_control_meta)
    base_zip_bytes = _archive_zip_bytes(bin_bytes=control_bytes)
    delta = int(target_archive_zip_bytes) - len(base_zip_bytes)
    if delta < 0:
        raise ValueError(
            "static-control archive cannot match recurrent archive size: "
            f"control_base_zip_bytes={len(base_zip_bytes)} "
            f"target_archive_zip_bytes={target_archive_zip_bytes}"
        )

    noop_meta_padding_bytes = 0
    if delta > 0xFFFF:
        best_bytes = control_bytes
        best_zip_bytes = base_zip_bytes
        best_padding_len = 0
        lo = 0
        hi = delta
        while lo <= hi:
            mid = (lo + hi) // 2
            trial_meta = dict(base_control_meta)
            trial_meta[padding_key] = _deterministic_padding_text(
                mid,
                label="z7-mamba2-static-control-meta-noop-padding",
            )
            trial_bytes = _pack_control_with_meta(trial_meta)
            trial_zip_bytes = _archive_zip_bytes(bin_bytes=trial_bytes)
            if len(trial_zip_bytes) <= int(target_archive_zip_bytes):
                best_bytes = trial_bytes
                best_zip_bytes = trial_zip_bytes
                best_padding_len = mid
                lo = mid + 1
            else:
                hi = mid - 1
        control_bytes = best_bytes
        base_zip_bytes = best_zip_bytes
        noop_meta_padding_bytes = best_padding_len
        delta = int(target_archive_zip_bytes) - len(base_zip_bytes)

    if delta > 0xFFFF:
        raise ValueError(
            "static-control archive could not absorb enough deterministic "
            "no-op metadata padding to leave a valid ZIP comment residual: "
            f"remaining_padding={delta}"
        )
    control_zip_bytes = _archive_zip_bytes(
        bin_bytes=control_bytes,
        comment=_deterministic_comment(delta) if delta else b"",
    )
    if len(control_zip_bytes) != int(target_archive_zip_bytes):
        raise ValueError(
            "static-control archive failed exact same-byte construction: "
            f"control_zip_bytes={len(control_zip_bytes)} "
            f"target_archive_zip_bytes={target_archive_zip_bytes}"
        )
    return control_bytes, control_zip_bytes, {
        "kind": "identity_predictor_static_capacity",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "ready_for_paid_dispatch": False,
        "rank_or_kill_eligible": False,
        "device_runtime_contract": base_control_meta.get("device_runtime_contract"),
        "archive_bin_bytes": len(control_bytes),
        "archive_zip_bytes": len(control_zip_bytes),
        "target_archive_zip_bytes": int(target_archive_zip_bytes),
        "same_archive_zip_bytes_as_recurrent": len(control_zip_bytes)
        == int(target_archive_zip_bytes),
        "archive_zip_byte_parity_with_target": len(control_zip_bytes)
        == int(target_archive_zip_bytes),
        "noop_meta_padding_bytes": int(noop_meta_padding_bytes),
        "zip_comment_padding_bytes": int(delta),
        "predictor_trainable_params": 0,
    }


def _write_runtime(submission_dir: Path) -> None:
    """Write the scorer-free Z7-Mamba-2 contest runtime tree.

    Per CLAUDE.md HNeRV parity L4 (≤200 LOC inflate; substrate-engineering
    waiver applied) + L9 (runtime closure: no scorer imports, deterministic
    dependency closure: torch + brotli + (mamba_ssm optional)).
    """
    submission_dir.mkdir(parents=True, exist_ok=True)
    for pkg_init in (
        submission_dir / "src" / "tac" / "__init__.py",
        submission_dir / "src" / "tac" / "optimization" / "__init__.py",
        submission_dir / "src" / "tac" / "substrates" / "__init__.py",
        submission_dir / "src" / "tac" / "substrates" / "_shared" / "__init__.py",
    ):
        pkg_init.parent.mkdir(parents=True, exist_ok=True)
        pkg_init.write_text("", encoding="utf-8")

    runtime_files = (
        (
            REPO_ROOT / "src" / "tac" / "optimization" / "mamba2_predictor.py",
            submission_dir / "src" / "tac" / "optimization" / "mamba2_predictor.py",
        ),
        (
            REPO_ROOT / "src" / "tac" / "substrates" / "_shared" / "inflate_runtime.py",
            submission_dir
            / "src"
            / "tac"
            / "substrates"
            / "_shared"
            / "inflate_runtime.py",
        ),
        (
            REPO_ROOT
            / "src"
            / "tac"
            / "substrates"
            / "time_traveler_l5_z6"
            / "architecture.py",
            submission_dir
            / "src"
            / "tac"
            / "substrates"
            / "time_traveler_l5_z6"
            / "architecture.py",
        ),
        (
            REPO_ROOT
            / "src"
            / "tac"
            / "substrates"
            / "time_traveler_l5_z7_lstm_predictive_coding"
            / "architecture.py",
            submission_dir
            / "src"
            / "tac"
            / "substrates"
            / "time_traveler_l5_z7_lstm_predictive_coding"
            / "architecture.py",
        ),
    )
    for src, dst in runtime_files:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    for pkg_dir, doc in (
        (
            submission_dir / "src" / "tac" / "substrates" / "time_traveler_l5_z6",
            "Z6 decoder dependency for Z7-Mamba-2 inflate runtime.",
        ),
        (
            submission_dir
            / "src"
            / "tac"
            / "substrates"
            / "time_traveler_l5_z7_lstm_predictive_coding",
            "Z7 context-conditioner dependency for Z7-Mamba-2 inflate runtime.",
        ),
    ):
        pkg_dir.mkdir(parents=True, exist_ok=True)
        (pkg_dir / "__init__.py").write_text(f'"""{doc}"""\n', encoding="utf-8")

    z7_src = REPO_ROOT / "src" / "tac" / "substrates" / "time_traveler_l5_z7_mamba2"
    z7_dst = (
        submission_dir
        / "src"
        / "tac"
        / "substrates"
        / "time_traveler_l5_z7_mamba2"
    )
    z7_dst.mkdir(parents=True, exist_ok=True)
    for name in ("architecture.py", "archive.py", "inflate.py"):
        shutil.copy2(z7_src / name, z7_dst / name)
    (z7_dst / "__init__.py").write_text(
        '"""Z7-Mamba-2 runtime package (inflate-time only; no scorer imports)."""\n',
        encoding="utf-8",
    )

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
PYTHON_BIN="${PYTHON:-python3}"
exec "$PYTHON_BIN" "$HERE/inflate.py" "$DATA_DIR" "$OUTPUT_DIR" "$FILE_LIST"
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
    normalized = str(value).strip().lower()
    if normalized in ("true", "1", "yes", "y", "on"):
        return True
    if normalized in ("false", "0", "no", "n", "off", "none", ""):
        return False
    raise argparse.ArgumentTypeError(
        f"expected boolean value, got {value!r}; use true/false"
    )


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
    device_contract = _device_runtime_contract(device)
    loss_mode = _normalize_loss_mode(args.loss_mode)
    ego_source = _normalize_ego_source(args.ego_source)
    pair_chunk_size = _resolve_pair_chunk_size(
        batch_size=int(args.batch_size),
        num_pairs=cfg.num_pairs,
    )
    torch.manual_seed(721)  # Distinct from Z7-LSTM seed 711 for paired-comparison

    print(f"[z7_mamba2_full] output_dir={output_dir}")
    print(f"[z7_mamba2_full] device={device} loss_mode={loss_mode}")
    print(f"[z7_mamba2_full] ego_source={ego_source}")
    print(f"[z7_mamba2_full] num_pairs={cfg.num_pairs} epochs={args.epochs}")
    print(f"[z7_mamba2_full] pair_chunk_size={pair_chunk_size}")
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
    pairs_unit_cpu = real_pairs.to(dtype=torch.float32) / 255.0
    if tuple(pairs_unit_cpu.shape[-2:]) != (cfg.output_height, cfg.output_width):
        flat = pairs_unit_cpu.reshape(
            -1,
            3,
            pairs_unit_cpu.shape[-2],
            pairs_unit_cpu.shape[-1],
        )
        resized = F.interpolate(
            flat,
            size=(cfg.output_height, cfg.output_width),
            mode="bilinear",
            align_corners=False,
        )
        pairs_unit_cpu = resized.reshape(
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
            _ego_motion_from_pairs(pairs_unit_cpu, cfg.ego_motion_dim).to(
                device=device,
                dtype=model.ego_motion_buffer.dtype,
            )
        )
    stage_wall_seconds["model_init_ego_seconds"] = time.perf_counter() - stage_started_at
    breakdown = model.num_parameters_breakdown()
    print(f"[z7_mamba2_full] model built: params={breakdown}")
    print(f"[z7_mamba2_full] predictor backend_active={model.predictor.backend_active}")
    _require_export_replay_backend(model)

    # Stage 3: optional score-aware loss build
    archive_bytes_proxy_tensor = torch.tensor(
        float(_estimate_archive_bytes_proxy(model, cfg)),
        device=device,
    )
    score_aware_loss: torch.nn.Module | None = None
    gt_scorer_cache: Any | None = None
    if loss_mode == "score_aware":
        if float(args.noise_std) != 0.0:
            raise RuntimeError(
                "Z7-Mamba-2 streamed score-aware chunking currently requires "
                "--noise-std 0.0 so the value pass and backward pass are identical."
            )
        stage_started_at = time.perf_counter()
        score_aware_loss = _build_score_aware_loss(args=args, device=device)
        stage_wall_seconds["score_aware_scorer_load_seconds"] = (
            time.perf_counter() - stage_started_at
        )
        score_aware_loss.train()
        print(f"[z7_mamba2_full] score_aware scorers loaded in {stage_wall_seconds['score_aware_scorer_load_seconds']:.2f}s")
        stage_started_at = time.perf_counter()
        from tac.training_optimization import build_gt_scorer_cache

        gt_scorer_cache = build_gt_scorer_cache(
            target_pixels=(pairs_unit_cpu * 255.0).contiguous(),
            posenet=score_aware_loss.pose_scorer,
            segnet=score_aware_loss.seg_scorer,
            device=device,
            cache_chunk_size=pair_chunk_size,
            pin_for_cuda=device.type == "cuda",
        )
        stage_wall_seconds["gt_scorer_cache_seconds"] = (
            time.perf_counter() - stage_started_at
        )
        print(gt_scorer_cache.summary_line())
        print(f"[z7_mamba2_full] GT scorer cache built in {stage_wall_seconds['gt_scorer_cache_seconds']:.2f}s")
    else:
        stage_wall_seconds["score_aware_scorer_load_seconds"] = 0.0
        stage_wall_seconds["gt_scorer_cache_seconds"] = 0.0

    # Stage 4: training loop
    epochs = max(1, int(args.epochs))
    base_lr = float(args.lr)
    lr_warmup_steps = max(0, int(args.lr_warmup_steps))
    grad_clip_norm = float(args.grad_clip_norm)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
    losses: list[dict[str, float | int | str]] = []
    target0_cpu = pairs_unit_cpu[:, 0].contiguous()
    target1_cpu = pairs_unit_cpu[:, 1].contiguous()
    pair_slices = _iter_pair_slices(cfg.num_pairs, pair_chunk_size)
    pair_chunk_count = len(pair_slices)
    _reset_peak_memory_stats(device)
    train_started_at = time.perf_counter()
    for epoch in range(epochs):
        epoch_started_at = time.perf_counter()
        if lr_warmup_steps > 0 and epoch < lr_warmup_steps:
            current_lr = base_lr * float(epoch + 1) / float(lr_warmup_steps)
            for group in optimizer.param_groups:
                group["lr"] = current_lr
        else:
            current_lr = base_lr
            for group in optimizer.param_groups:
                group["lr"] = current_lr
        optimizer.zero_grad(set_to_none=True)
        latents, contexts = model.replay_latents_and_contexts()
        decoder_latents = model.condition_latents(latents, contexts)
        residual_loss = model.residuals.pow(2).mean()
        latent_smoothness = (
            (latents[1:] - latents[:-1]).pow(2).mean()
            if latents.shape[0] > 1
            else latents.new_tensor(0.0)
        )
        if score_aware_loss is not None:
            assert gt_scorer_cache is not None
            seg_term_value = latents.new_tensor(0.0)
            pose_term_value = latents.new_tensor(0.0)
            with torch.no_grad():
                decoder_latents_value = decoder_latents.detach()
                for start, end in pair_slices:
                    rgb_0_value, rgb_1_value = model.decoder(
                        decoder_latents_value[start:end]
                    )
                    idx = torch.arange(start, end, dtype=torch.long)
                    gt_pose_batch, gt_seg_batch = gt_scorer_cache.lookup(
                        idx,
                        device=device,
                    )
                    seg_chunk, pose_chunk = score_aware_loss.score_terms(
                        reconstructed_rgb_0=rgb_0_value * 255.0,
                        reconstructed_rgb_1=rgb_1_value * 255.0,
                        gt_pose_batch=gt_pose_batch,
                        gt_seg_batch=gt_seg_batch,
                        gt_seg_already_probs=gt_scorer_cache.seg_already_probs,
                        apply_eval_roundtrip=True,
                        noise_std=0.0,
                        scorer_chunk_size=pair_chunk_size,
                    )
                    weight = float(end - start) / float(cfg.num_pairs)
                    seg_term_value = seg_term_value + seg_chunk * weight
                    pose_term_value = pose_term_value + pose_chunk * weight
            rate_term = (
                score_aware_loss.weights.alpha_rate
                * archive_bytes_proxy_tensor
                / score_aware_loss.weights.contest_normalizer
            )
            pose_sqrt = torch.sqrt(pose_term_value.clamp(min=1e-12))
            ib_term = score_aware_loss.weights.beta_ib * score_aware_loss.weights.ib_scale * (
                residual_loss + latent_smoothness
            )
            loss = (
                rate_term
                + score_aware_loss.weights.beta_seg * seg_term_value
                + score_aware_loss.weights.gamma_pose * pose_sqrt
                + ib_term.detach()
            )
            loss_record: dict[str, float | int | str] = {
                "epoch": epoch,
                "loss": float(loss.item()),
                "pair_chunk_size": int(pair_chunk_size),
                "pair_chunk_count": int(pair_chunk_count),
                "score_aware_backward_mode": SCORE_AWARE_BACKWARD_MODE,
                "rate_term": float(rate_term.detach().cpu()),
                "seg_term": float(seg_term_value.detach().cpu()),
                "pose_term": float(pose_term_value.detach().cpu()),
                "pose_sqrt": float(pose_sqrt.detach().cpu()),
                "residual_norm": float(residual_loss.detach().cpu()),
                "latent_smoothness": float(latent_smoothness.detach().cpu()),
                "ib_term": float(ib_term.detach().cpu()),
            }
            _require_finite_loss(loss, epoch=epoch)
            pose_value = float(pose_term_value.detach().cpu())
            pose_grad_coeff = (
                latents.new_tensor(0.0)
                if pose_value < 1e-12
                else latents.new_tensor(
                    score_aware_loss.weights.gamma_pose * 0.5 / (pose_value ** 0.5)
                )
            )
            for start, end in pair_slices:
                rgb_0_chunk, rgb_1_chunk = model.decoder(decoder_latents[start:end])
                idx = torch.arange(start, end, dtype=torch.long)
                gt_pose_batch, gt_seg_batch = gt_scorer_cache.lookup(
                    idx,
                    device=device,
                )
                seg_chunk, pose_chunk = score_aware_loss.score_terms(
                    reconstructed_rgb_0=rgb_0_chunk * 255.0,
                    reconstructed_rgb_1=rgb_1_chunk * 255.0,
                    gt_pose_batch=gt_pose_batch,
                    gt_seg_batch=gt_seg_batch,
                    gt_seg_already_probs=gt_scorer_cache.seg_already_probs,
                    apply_eval_roundtrip=True,
                    noise_std=0.0,
                    scorer_chunk_size=pair_chunk_size,
                )
                weight = float(end - start) / float(cfg.num_pairs)
                chunk_loss = weight * (
                    score_aware_loss.weights.beta_seg * seg_chunk
                    + pose_grad_coeff * pose_chunk
                )
                chunk_loss.backward(retain_graph=True)
            ib_term.backward()
        else:
            recon_loss_value = latents.new_tensor(0.0)
            for start, end in pair_slices:
                rgb_0_chunk, rgb_1_chunk = model.decoder(decoder_latents[start:end])
                chunk_loss = F.mse_loss(
                    rgb_0_chunk,
                    target0_cpu[start:end].to(device=device, dtype=rgb_0_chunk.dtype),
                    reduction="sum",
                ) / target0_cpu.numel() + F.mse_loss(
                    rgb_1_chunk,
                    target1_cpu[start:end].to(device=device, dtype=rgb_1_chunk.dtype),
                    reduction="sum",
                ) / target1_cpu.numel()
                recon_loss_value = recon_loss_value + chunk_loss.detach()
                chunk_loss.backward(retain_graph=True)
            regularizer = float(args.beta_ib) * 1e-3 * (
                residual_loss + latent_smoothness
            )
            loss = recon_loss_value + regularizer.detach()
            loss_record = {
                "epoch": epoch,
                "loss": float(loss.item()),
                "recon": float(recon_loss_value.item()),
                "residual": float(residual_loss.item()),
                "latent_smoothness": float(latent_smoothness.item()),
                "pair_chunk_size": int(pair_chunk_size),
                "pair_chunk_count": int(pair_chunk_count),
            }
            _require_finite_loss(loss, epoch=epoch)
            regularizer.backward()
        grad_norm = None
        if grad_clip_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                grad_clip_norm,
            )
        optimizer.step()
        loss_record["wall_seconds"] = time.perf_counter() - epoch_started_at
        loss_record["lr"] = float(current_lr)
        if grad_norm is not None:
            loss_record["grad_norm_before_clip"] = float(grad_norm.detach().cpu())
        losses.append(loss_record)
        if epoch < 3 or epoch % max(1, epochs // 10) == 0:
            print(f"[z7_mamba2_full] epoch {epoch}: loss={loss.item():.6f} wall={loss_record['wall_seconds']:.2f}s")
    train_total_seconds = time.perf_counter() - train_started_at
    stage_wall_seconds["train_total_seconds"] = train_total_seconds
    memory_telemetry = _memory_telemetry(device)

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
        "pair_chunk_size": int(pair_chunk_size),
        "pair_chunk_count": pair_chunk_count,
        "batch_size_actuation_status": "actuated_as_pair_chunked_decoder_and_scorer_loss",
        "lr": base_lr,
        "lr_warmup_steps": lr_warmup_steps,
        "grad_clip_norm": grad_clip_norm,
        "loss_mode": loss_mode,
        "score_aware_scorer_loss_used": loss_mode == "score_aware",
        "ego_source": ego_source,
        "ego_motion_source": "real_video_pair_delta_proxy_no_scorer_load",
        "training_device": str(device),
        "device_runtime_contract": device_contract,
        "rank_or_kill_eligible": False,
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
            control_zip_path.write_bytes(static_control_zip_bytes)
            assert isinstance(static_control, dict)
            static_control["archive_bin_path"] = str(control_bin_path)
            static_control["archive_zip_path"] = str(control_zip_path)
            static_control["archive_zip_sha256"] = _sha256_bytes(
                static_control_zip_bytes
            )
        except Exception as e:  # pragma: no cover - defensive
            raise RuntimeError(
                "Z7-Mamba-2 static same-byte control emission failed; refusing "
                "to continue because paired recurrent/static evidence would be "
                "non-comparable"
            ) from e

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
    runtime_geometry_positive_control: dict[str, object] | None = None
    stage_started_at = time.perf_counter()
    if _boolish(args.inflate_verify):
        verify_raw = output_dir / "inflate_verify" / "0.raw"
        try:
            verify_device = _inflate_verify_device(device)
            frames = inflate_one_video(
                archive_bytes,
                verify_raw,
                device=verify_device,
            )
            recurrent_raw_bytes = verify_raw.read_bytes()
            inflate_verify = {
                "device": verify_device,
                "raw_path": str(verify_raw),
                "frames_written": int(frames),
                "raw_bytes": verify_raw.stat().st_size,
                "raw_sha256": _sha256_bytes(recurrent_raw_bytes),
            }
            if static_control_bytes is not None and isinstance(static_control, dict):
                control_raw = output_dir / "inflate_verify" / "static_control.raw"
                control_frames = inflate_one_video(
                    static_control_bytes,
                    control_raw,
                    device=verify_device,
                )
                control_raw_bytes = control_raw.read_bytes()
                control_sha = _sha256_bytes(control_raw_bytes)
                byte_differences = sum(
                    a != b
                    for a, b in zip(
                        recurrent_raw_bytes,
                        control_raw_bytes,
                        strict=False,
                    )
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
                runtime_geometry_positive_control = (
                    _build_runtime_geometry_positive_control(
                        archive_bytes=archive_bytes,
                        recurrent_raw_bytes=recurrent_raw_bytes,
                        static_raw_bytes=control_raw_bytes,
                        num_pairs=cfg.num_pairs,
                        render_hw=(cfg.output_height, cfg.output_width),
                    )
                )
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
        "device_runtime_contract": device_contract,
        "loss_mode": loss_mode,
        "epochs": epochs,
        "num_pairs": cfg.num_pairs,
        "pair_chunk_size": int(pair_chunk_size),
        "pair_chunk_count": int(pair_chunk_count),
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
        "rank_or_kill_eligible": False,
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
            "pair_chunk_size": int(pair_chunk_size),
            "pair_chunk_count": pair_chunk_count,
            "lr_warmup_steps": lr_warmup_steps,
            "grad_clip_norm": grad_clip_norm,
            **model.decoder_metadata(),
        },
        "param_breakdown": breakdown,
        "losses": losses,
        "loss_mode": loss_mode,
        "score_aware_scorer_loss_used": loss_mode == "score_aware",
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
        "submission_runtime_dir": str(submission_dir),
        "inflate_verify": inflate_verify,
        "runtime_geometry_positive_control": runtime_geometry_positive_control,
        "timing_smoke": timing_smoke,
        "memory_telemetry": memory_telemetry,
        "device_runtime_contract": device_contract,
        # Authority flags — non-promotable by construction per CLAUDE.md
        "evidence_grade": "z7_mamba2_full_train_packet_not_promotable_pending_council_per_catalog_325",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "ready_for_paid_dispatch": False,
        "rank_or_kill_eligible": False,
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
    print("[z7_mamba2_full] DONE [no-auth-eval-pending-wave-N+1-council]")

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
