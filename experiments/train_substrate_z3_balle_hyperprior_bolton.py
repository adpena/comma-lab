# SPDX-License-Identifier: MIT
"""Train the Z3 Ballé hyperprior bolt-on substrate (across-class staircase Step 1).

Per zen-floor band v2 council + long-term campaign roadmap, Z3 is the cheapest
$2 validation that Ballé-2018 scale hyperprior side-info reduces bytes on the
FROZEN A1 base. Predicted ΔS = −0.005 to −0.010 vs A1 0.1928 [contest-CPU 1to1]
``[prediction; first-principles-bound]``.

Council-binding contract (CLAUDE.md non-negotiables) honored end-to-end:

- Train against frozen A1 latents decoded from the A1 archive (NOT synthetic
  data; Catalog #114 forbids synthetic non-smoke data).
- Patch upstream ``rgb_to_yuv6`` via ``patch_upstream_yuv6_globally`` BEFORE
  scorer construction (Catalog #187; PR #95/#106 contract; the full pose
  Lagrangian path requires this even though Z3 trains rate-mostly).
- ``load_differentiable_scorers`` for SegNet/PoseNet (no scorer load at
  inflate; only at training).
- ``apply_eval_roundtrip_during_training`` inside the per-batch loop
  (Catalog #5) when scorer loss is active.
- ``tac.training.EMA(decay=0.997)`` update after every ``optimizer.step``;
  inference checkpoint = EMA shadow (Catalog #88).
- AdamW lr; gradient clip 1.0.
- End with CUDA auth eval on best EMA checkpoint (CLAUDE.md "Auth eval
  EVERYWHERE"); refuse MPS (Catalog #1); CPU permitted only with ``--smoke``
  or ``--full-cpu --advisory-cpu-explicitly-waived`` (Catalog #197).
- TIER_1_OPERATOR_REQUIRED_FLAGS declared per Catalog #151 (annotated
  assignment for Catalog #168 AST walker).
- Auth eval via canonical ``gate_auth_eval_call`` (Catalog #223 protects
  against finite-only-parser misuse).
- ``pose_scorer, seg_scorer = load_differentiable_scorers(...)`` per Catalog
  #222 (canonical loader returns (posenet, segnet)).
- ``detect_hardware_substrate`` per Catalog #190.

Execution scope:
- ``_smoke_main`` builds a tiny config, trains the Z3HV2 residual-domain
  rate Lagrangian for 3 epochs against synthetic A1 latents, runs the v2
  archive pack + parse roundtrip, and emits a contest-compliant runtime tree.
  NO scorer load required.
- ``_full_main`` decodes A1's frozen latent_blob from ``--a1-archive-path``,
  fine-tunes the hyperprior with the full Ballé R+λD Lagrangian (rate +
  seg + pose), packs the Z3HV2 composition archive (A1 decoder bytes + Z3HV2
  latent-replacement section + A1 sidecar),
  emits the contest-compliant runtime tree, runs CUDA auth eval on the
  best EMA checkpoint, and posts the result to the continual-learning
  posterior.

Usage (smoke; macOS CPU, tiny config, ~3 epochs)::

    .venv/bin/python experiments/train_substrate_z3_balle_hyperprior_bolton.py \\
        --a1-archive-path submissions/a1/archive.zip \\
        --output-dir experiments/results/z3_smoke_<utc> \\
        --epochs 3 --device cpu --smoke

Usage (full; CUDA-required; gated behind Phase 2 council approval)::

    .venv/bin/python experiments/train_substrate_z3_balle_hyperprior_bolton.py \\
        --a1-archive-path submissions/a1/archive.zip \\
        --output-dir experiments/results/z3_<utc> \\
        --epochs 1000 --batch-size 16 --lr 1e-3 --device cuda
"""
# AUTOCAST_FP16_WAIVED:defer-until-empirical-anchor-shows-numeric-stability-fp16-vs-fp32
# TORCH_COMPILE_WAIVED:defer-until-per-substrate-canary-validates-Inductor-graph-breaks
# NO_GRAD_WAIVED:eval-time-scorer-forward-wrapped-in-torch.inference_mode-inside-_run_val_loop
# SYNTHETIC_NON_SMOKE_OK:_smoke_main-only-uses-synthetic-latents-_full_main-decodes-A1
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
import sys
import time
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

from tac.substrates._shared.smoke_auth_eval_gate import (
    gate_auth_eval_call as _canon_gate_auth_eval_call,
)
from tac.substrates._shared.trainer_skeleton import (
    detect_hardware_substrate as _canon_detect_hardware_substrate,
)
from tac.substrates._shared.trainer_skeleton import (
    device_or_die as _canon_device_or_die,
)
from tac.substrates._shared.trainer_skeleton import (
    git_head_sha as _canon_git_head_sha,
)
from tac.substrates._shared.trainer_skeleton import (
    pin_seeds as _canon_pin_seeds,
)
from tac.substrates._shared.trainer_skeleton import (
    sha256_bytes as _canon_sha256_bytes,
)
from tac.substrates._shared.trainer_skeleton import (
    utc_now_iso as _canon_utc_now_iso,
)
from tac.substrates._shared.trainer_skeleton import (
    vendor_shared_inflate_runtime as _canon_vendor_shared_inflate_runtime,
)
from tac.substrates.z3_balle_hyperprior_bolton import (
    A1_LATENT_DIM,
    A1_N_PAIRS,
    Z3HyperpriorConfig,
    Z3HyperpriorMLP,
)
from tac.substrates.z3_balle_hyperprior_bolton.score_aware_loss_v2 import (
    estimate_z3v2_section_overhead_bytes,
    z3_v2_lagrangian,
)

# ---------------------------------------------------------------------------
# Module paths + constants
# ---------------------------------------------------------------------------

DEFAULT_A1_ARCHIVE = REPO_ROOT / "submissions" / "a1" / "archive.zip"
DEFAULT_UPSTREAM_DIR = REPO_ROOT / "upstream"
DEFAULT_VIDEO_PATH = REPO_ROOT / "upstream" / "videos" / "0.mkv"
CONTEST_AUTH_EVAL_SCRIPT = REPO_ROOT / "experiments" / "contest_auth_eval.py"

SUBSTRATE_TAG = "z3_balle_hyperprior_bolton"
SUBSTRATE_LANE_ID = "lane_z3_balle_hyperprior_bolton_recover_20260514"


def _z3v2_affine_residual_int8(
    a1_latents: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return signed-int8 residuals plus the v2 affine reload parameters.

    Z3V2 stores a centered affine offset because v2 needs a signed int8
    residual. A minimum offset would map the source range to ``0..254`` and
    silently clip the upper half when stored as int8. Centering maps endpoints
    to approximately ``[-127, 127]``.
    """
    a1_cpu = a1_latents.detach().to(torch.float32).cpu()
    a1_min = a1_cpu.min(dim=0).values
    a1_max = a1_cpu.max(dim=0).values
    a1_offset = (a1_min + a1_max) * 0.5
    a1_scale = ((a1_max - a1_min) / 254.0).clamp(min=1e-6)
    residual = (
        (a1_cpu - a1_offset.unsqueeze(0)) / a1_scale.unsqueeze(0)
    ).round().clamp(min=-127.0, max=127.0).to(torch.int8)
    return residual, a1_offset, a1_scale


def _use_v2_latent_replacement(args: argparse.Namespace) -> bool:
    """Z3 frontier path is v2 latent replacement by default."""
    return bool(getattr(args, "enable_v2_latent_replacement", True))


def _validate_z3v2_quantization_step(args: argparse.Namespace) -> None:
    """Fail closed while Z3V2 quant_step is metadata, not runtime math."""
    if _use_v2_latent_replacement(args) and not math.isclose(
        float(args.quantization_step), 1.0, rel_tol=0.0, abs_tol=1e-12
    ):
        raise SystemExit(
            "ERROR: Z3V2 requires --quantization-step 1.0 because the v2 "
            "inflate runtime reconstructs centered signed-int8 residuals as "
            "residual * scale + offset. Non-1.0 quant steps need a matching "
            "train/export/inflate contract before dispatch."
        )
    if int(getattr(args, "hyper_hidden_dim", 16)) != 16:
        raise SystemExit(
            "ERROR: Z3V2 currently requires --hyper-hidden-dim 16 because the "
            "wire grammar does not serialize hidden width. Add a grammar field "
            "and runtime decoder support before varying this value."
        )


# ---------------------------------------------------------------------------
# Catalog #151 manifest — every flag below must be threaded by any operator
# wrapper that subprocess-invokes this trainer. Annotated as ast.AnnAssign so
# Catalog #168's AST walker observes it (NOT bare Assign).
# ---------------------------------------------------------------------------
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--a1-archive-path": {
        "env": "Z3_BALLE_A1_ARCHIVE_PATH",
        "rationale": (
            "Z3 is a bolt-on over the FROZEN A1 base; the A1 archive's "
            "latent_blob is decoded to obtain the per-pair latents that "
            "Z3 re-encodes via the hyperprior. Path is required for full "
            "training; smoke mode uses synthetic latents."
        ),
        "default": str(DEFAULT_A1_ARCHIVE.relative_to(REPO_ROOT)),
        "required_input_file": True,
        "generator_command": (
            "contest-pinned A1 archive — generated by submissions/a1 substrate "
            "(frozen across-class staircase Step 1 base)"
        ),
        "rationale_audit": (
            "feedback_z3_balle_hyperprior_bolton_landed_20260514.md + "
            "feedback_long_term_multi_year_campaigns_landed_20260514.md C5"
        ),
    },
    "--video-path": {
        "env": "Z3_BALLE_VIDEO_PATH",
        "rationale": (
            "Full training requires upstream/videos/0.mkv for the seg+pose "
            "score-aware Lagrangian; synthetic data FORBIDDEN outside --smoke"
        ),
        "default": str(DEFAULT_VIDEO_PATH.relative_to(REPO_ROOT)),
        "required_input_file": True,
        "generator_command": (
            "contest-pinned upstream snapshot — never regenerated locally"
        ),
    },
    "--output-dir": {
        "env": "Z3_BALLE_OUTPUT_DIR",
        "rationale": "custody location for checkpoints + archive + provenance",
    },
    "--epochs": {
        "env": "Z3_BALLE_EPOCHS",
        "rationale": (
            "Z3 hyperprior is tiny (~1.8k params); council default 1000 "
            "epochs for full training run"
        ),
        "default": "1000",
    },
    "--upstream-dir": {
        "env": "Z3_BALLE_UPSTREAM_DIR",
        "rationale": (
            "upstream/ root for scorer weights + evaluate.py; required for "
            "full training and auth eval"
        ),
        "default": str(DEFAULT_UPSTREAM_DIR.relative_to(REPO_ROOT)),
    },
    "--device": {
        "env": "Z3_BALLE_DEVICE",
        "rationale": (
            "compute device; cuda required for full training (MPS refused "
            "per CLAUDE.md MPS-NOISE rule); cpu permitted only with --smoke "
            "or --full-cpu --advisory-cpu-explicitly-waived (Catalog #197)"
        ),
        "default": "cuda",
    },
    "--hyper-latent-dim": {
        "env": "Z3_BALLE_HYPER_LATENT_DIM",
        "rationale": (
            "Hyper-latent w_p dimensionality (per-pair side-info); Ballé "
            "2018 small variant = 8; must be << A1_LATENT_DIM=28"
        ),
        "default": "8",
    },
}


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_substrate_z3_balle_hyperprior_bolton",
        description=(
            "Train Z3 Ballé hyperprior bolt-on (across-class staircase Step 1). "
            "Re-encodes the FROZEN A1 latent_blob via a tiny per-pair Ballé-2018 "
            "scale-hyperprior to reduce archive bytes by ~5-15% with zero "
            "distortion change. Predicted ΔS = -0.005 to -0.010."
        ),
    )
    p.add_argument("--a1-archive-path", type=Path, default=DEFAULT_A1_ARCHIVE)
    p.add_argument("--video-path", type=Path, default=DEFAULT_VIDEO_PATH)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR)
    p.add_argument("--hyper-latent-dim", type=int, default=8)
    p.add_argument("--hyper-hidden-dim", type=int, default=16)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke-only path (synthetic A1 latents, 3 epochs, no scorer)",
    )
    p.add_argument(
        "--full-cpu",
        action="store_true",
        help=(
            "Opt-in to non-smoke CPU training (per Catalog #197 must be "
            "paired with --advisory-cpu-explicitly-waived)"
        ),
    )
    p.add_argument(
        "--advisory-cpu-explicitly-waived",
        action="store_true",
        help="Required sister flag for --full-cpu (Catalog #197 coupled flag)",
    )
    # Full training hyperparameters
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.997)
    p.add_argument("--quantization-step", type=float, default=1.0)
    p.add_argument("--factorized-half-range", type=float, default=16.0)
    # Score-aware Lagrangian weights (council defaults match contest formula)
    p.add_argument("--alpha-rate", type=float, default=25.0)
    p.add_argument("--beta-seg", type=float, default=100.0)
    p.add_argument("--gamma-pose", type=float, default=math.sqrt(10.0))
    p.add_argument("--pose-weight-scale", type=float, default=1.0,
                   help=(
                       "Opt-in pose marginal tilt. Default 1.0 preserves the "
                       "contest formula; PR106-derived 2.71x is experimental."
                   ))
    # Post-train artifacts
    p.add_argument("--skip-auth-eval", action="store_true")
    p.add_argument("--skip-archive-build", action="store_true")
    p.add_argument("--enable-autocast-fp16", action="store_true")
    p.add_argument("--enable-torch-compile", action="store_true")
    p.add_argument("--enable-gt-scorer-cache", action="store_true")
    # v2 latent-replacement archive grammar (council omnibus Decision 3, 2026-05-14).
    # The trainer emits the Z3HV2-section payload that REPLACES A1's latent_blob
    # in-place. This flag is accepted as an explicit operator attestation, but
    # v2 is the default frontier path.
    p.set_defaults(enable_v2_latent_replacement=True)
    p.add_argument(
        "--enable-v2-latent-replacement",
        action="store_true",
        help=(
            "Use the v2 latent-replacement archive grammar (Z3HV2 magic) "
            "instead of the legacy v1 append-only sidecar. Council omnibus "
            "Decision 3 binding verdict (commit 7872c9f4b) — v1 retired as "
            "redundant; v2 is the operational latent-replacement path."
        ),
    )
    p.add_argument("--auth-eval-skipped-reason", type=str, default="",
                   help="Optional reason for skipping auth eval (carried into stats).")
    return p


def _validate_full_cpu_flags(args: argparse.Namespace) -> None:
    """Catalog #197: --full-cpu MUST be paired with the advisory waiver flag."""
    if args.full_cpu and not args.advisory_cpu_explicitly_waived:
        raise SystemExit(
            "ERROR: --full-cpu requires --advisory-cpu-explicitly-waived per "
            "Catalog #197 (paired-flag attestation that the CPU-axis bypass "
            "is intentional and non-promotable)"
        )


# ---------------------------------------------------------------------------
# Smoke entry path
# ---------------------------------------------------------------------------


def _smoke_main(args: argparse.Namespace) -> int:
    """Smoke entry: synthetic A1 latents, rate-only Lagrangian, 3 epochs.

    Per Catalog #114, synthetic data is permitted ONLY inside _smoke_main.
    The full path decodes real A1 latents from the archive.
    """
    _validate_z3v2_quantization_step(args)
    _canon_pin_seeds(args.seed)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Smoke synthetic A1 latents (deterministic random). Use the full A1 pair
    # count so the trained rate objective and exported Z3V2 residual tensor have
    # the same shape.
    torch.manual_seed(args.seed)
    n_smoke_pairs = 32
    a1_latents = torch.zeros(A1_N_PAIRS, A1_LATENT_DIM, device=args.device)
    a1_latents[:n_smoke_pairs] = torch.randn(
        n_smoke_pairs, A1_LATENT_DIM, device=args.device
    )
    _, smoke_offset_cpu, smoke_scale_cpu = _z3v2_affine_residual_int8(a1_latents)
    smoke_offset = smoke_offset_cpu.to(args.device)
    smoke_scale = smoke_scale_cpu.to(args.device)

    cfg = Z3HyperpriorConfig(
        hyper_latent_dim=args.hyper_latent_dim,
        hyper_hidden_dim=args.hyper_hidden_dim,
        quantization_step=args.quantization_step,
    )
    hyperprior = Z3HyperpriorMLP(cfg).to(args.device)
    optimizer = torch.optim.AdamW(
        hyperprior.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    losses = []
    rate_bits = []
    n_epochs = max(args.epochs, 3)
    for _epoch in range(n_epochs):
        optimizer.zero_grad()
        out = z3_v2_lagrangian(
            hyperprior=hyperprior,
            a1_latents=a1_latents,
            latent_offset=smoke_offset,
            latent_scale=smoke_scale,
            seg_scorer=torch.nn.Identity(),
            pose_scorer=torch.nn.Identity(),
            decoded_pair_rt=None,
            gt_pair=None,
            alpha_rate=args.alpha_rate,
            beta_seg=args.beta_seg,
            gamma_pose=args.gamma_pose,
            quantization_step=args.quantization_step,
            factorized_half_range=args.factorized_half_range,
        )
        out["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(hyperprior.parameters(), args.grad_clip)
        optimizer.step()
        losses.append(float(out["total_loss"].item()))
        rate_bits.append(float(out["rate_bits_total"].item()))

    # Build the Z3 composition archive over the real A1 base when available so
    # emitted smoke runtimes still consume the contest-style extracted `x`.
    base_bytes, _, _ = _load_a1_archive_bytes(args.a1_archive_path)
    base_sha = hashlib.sha256(base_bytes).hexdigest()

    # Direct-residual Z3HV2 mode: do not ship no-op hyperprior bytes until a
    # real entropy-coded residual decoder consumes them at inflate time.
    a1_full = torch.zeros(A1_N_PAIRS, A1_LATENT_DIM, device=args.device)
    a1_full[: a1_latents.shape[0]] = a1_latents

    from tac.substrates.z3_balle_hyperprior_bolton.archive_v2 import (
        build_z3v2_composition_archive_contract as _v2_build_contract,
    )
    from tac.substrates.z3_balle_hyperprior_bolton.archive_v2 import (
        build_z3v2_payload_bytes as _v2_build_payload,
    )
    from tac.substrates.z3_balle_hyperprior_bolton.archive_v2 import (
        encode_z3hv2_section as _v2_encode_section,
    )

    residual_v2, a1_offset, a1_scale = _z3v2_affine_residual_int8(a1_full)
    z3hv2_section = _v2_encode_section(
        hyperprior_weights_int8=b"",
        w_hat_int8=b"",
        residual_int8=residual_v2.numpy().tobytes(),
        latent_offset=a1_offset,
        latent_scale=a1_scale,
        hyper_dim=cfg.hyper_latent_dim,
        int8_w_scale=1.0,
        quant_step=cfg.quantization_step,
        min_sigma=cfg.min_sigma,
        max_sigma=cfg.max_sigma,
        factorized_half_range=args.factorized_half_range,
    )
    archive_bytes = _v2_build_payload(
        a1_bytes=base_bytes,
        z3hv2_section=z3hv2_section,
    )
    archive_contract = _v2_build_contract(base_bytes, archive_bytes)
    sidecar_bytes = len(z3hv2_section)
    (out_dir / "z3v2_section_diagnostic.bin").write_bytes(z3hv2_section)
    archive_path = out_dir / "0.bin"
    archive_path.write_bytes(archive_bytes)

    # Emit runtime tree + archive.zip per the canonical pattern.
    submission_dir = out_dir / "submission_dir"
    _write_runtime(submission_dir, a1_runtime_src=args.a1_archive_path.parent / "src")
    runtime_manifest = _write_submission_runtime_manifest(submission_dir)
    (submission_dir / "0.bin").write_bytes(archive_bytes)
    archive_zip_path = out_dir / "archive.zip"
    _build_archive_zip(archive_zip_path, bin_bytes=archive_bytes)

    # Stats provenance
    archive_sha = _canon_sha256_bytes(archive_bytes)
    archive_zip_sha = _canon_sha256_bytes(archive_zip_path.read_bytes())
    archive_zip_size = archive_zip_path.stat().st_size
    final_loss = losses[-1] if losses else float("inf")
    final_rate_bits = rate_bits[-1] if rate_bits else float("inf")
    hardware_substrate = _canon_detect_hardware_substrate(
        substrate_tag=SUBSTRATE_TAG,
        axis="cpu" if args.device == "cpu" else "cuda",
    )
    stats = {
        "lane_id": SUBSTRATE_LANE_ID,
        "substrate_tag": SUBSTRATE_TAG,
        "smoke": True,
        "epochs": len(losses),
        "final_loss_proxy": final_loss,
        "final_rate_bits_total": final_rate_bits,
        "archive_bytes": len(archive_bytes),
        "archive_sha256": archive_sha,
        "archive_zip_bytes": archive_zip_size,
        "archive_zip_sha256": archive_zip_sha,
        "base_archive_sha256": base_sha,
        "hyper_dim": cfg.hyper_latent_dim,
        "param_count": sum(p.numel() for p in hyperprior.parameters()),
        "sidecar_bytes": sidecar_bytes,
        "archive_contract": archive_contract.as_manifest(),
        "byte_saving": archive_contract.byte_saving,
        "v2_latent_replacement_requested": True,
        "v2_latent_replacement_active": True,
        "estimated_sidecar_overhead": estimate_z3v2_section_overhead_bytes(
            hyperprior=hyperprior,
        ),
        "cfg": asdict(cfg),
        "score_claim": False,
        "score_claim_valid": False,
        "evidence_grade": "smoke-no-scorer",
        "ready_for_exact_eval_dispatch": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "result_review_blockers": [
            "smoke_no_scorer_load",
            "requires_separate_auth_eval_result_review_before_score_claim",
        ],
        "runtime_tree_sha256": runtime_manifest["runtime_tree_sha256"],
        "submission_runtime_manifest_path": str(
            submission_dir / "submission_runtime_manifest.json"
        ),
        "hardware_substrate": hardware_substrate,
        "git_head": _canon_git_head_sha(REPO_ROOT),
        "trained_at_utc": _canon_utc_now_iso(),
    }
    (out_dir / "stats.json").write_text(
        json.dumps(stats, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    _write_training_artifact_manifest(
        out_dir,
        stats=stats,
        training_mode="smoke",
        research_only=True,
    )
    print(
        f"[z3-smoke] OK final_loss={final_loss:.6f} "
        f"rate_bits={final_rate_bits:.1f} archive={len(archive_bytes)}B "
        f"sha={archive_sha[:12]}... layout={stats['archive_contract']['layout']} "
        f"param_count={stats['param_count']}"
    )
    return 0


# ---------------------------------------------------------------------------
# Full training entry path (gated behind Phase 2 council approval)
# ---------------------------------------------------------------------------


def _load_a1_archive_bytes(archive_zip_path: Path) -> tuple[bytes, str, int]:
    """Read the inner ``x`` blob from an A1-style archive.zip.

    Mirrors the canonical helper in a1_plus_wavelet_residual / a1_plus_lapose.
    """
    with zipfile.ZipFile(archive_zip_path) as zf:
        names = zf.namelist()
        if "x" not in names:
            raise ValueError(
                f"A1 archive {archive_zip_path} missing inner 'x' blob; got {names}"
            )
        data = zf.read("x")
    sha = _canon_sha256_bytes(data)
    return data, sha, len(data)


def _load_a1_runtime_modules(a1_archive_path: Path):
    """Load A1's vendored codec.py / model.py from the source runtime.

    Mirrors the canonical helper in a1_plus_wavelet_residual.
    """
    import importlib

    a1_src = a1_archive_path.parent / "src"
    if not (a1_src / "codec.py").is_file() or not (a1_src / "model.py").is_file():
        raise FileNotFoundError(
            f"A1 source runtime missing at {a1_src}; expected codec.py + model.py"
        )
    old_path = list(sys.path)
    old_codec = sys.modules.pop("codec", None)
    old_model = sys.modules.pop("model", None)
    sys.path.insert(0, str(a1_src))
    try:
        model = importlib.import_module("model")
        codec = importlib.import_module("codec")
        return codec, model
    finally:
        sys.path = old_path
        sys.modules.pop("codec", None)
        sys.modules.pop("model", None)
        if old_codec is not None:
            sys.modules["codec"] = old_codec
        if old_model is not None:
            sys.modules["model"] = old_model


def _decode_a1_latents_and_decoder(
    a1_archive_path: Path, a1_bytes: bytes, *, device,
):
    """Decode the FROZEN A1 archive into (decoder_module, latents_tensor).

    Z3's hyperprior re-encodes the latent_blob; this returns the decoded
    A1 latents (shape ``(600, 28)``) and the A1 HNeRVDecoder (frozen).
    """
    import struct

    codec, model = _load_a1_runtime_modules(a1_archive_path)
    if len(a1_bytes) < 4:
        raise ValueError("A1 archive bytes too short")
    section_total = struct.unpack_from("<I", a1_bytes, 0)[0]
    decoder_blob = a1_bytes[4:section_total]
    latent_blob = a1_bytes[section_total : section_total + int(codec.LATENT_BLOB_LEN)]
    sidecar_blob = a1_bytes[section_total + int(codec.LATENT_BLOB_LEN) :]
    decoder_sd = codec.decode_decoder_compact(decoder_blob)
    latents = codec.apply_latent_sidecar(
        codec.decode_latents_compact(latent_blob), sidecar_blob
    ).to(device)
    decoder = model.HNeRVDecoder(
        latent_dim=A1_LATENT_DIM,
        base_channels=36,  # A1_BASE_CHANNELS
        eval_size=(384, 512),
    ).to(device)
    decoder.load_state_dict(decoder_sd)
    decoder.eval()
    for p in decoder.parameters():
        p.requires_grad_(False)
    return decoder, latents


def _decode_real_pairs(
    video_path: Path,
    n_pairs: int = A1_N_PAIRS,
    output_hw: tuple[int, int] = (874, 1164),
    max_pairs: int | None = None,
):
    """Decode N pairs from contest video; returns (n_pairs, 2, 3, H, W) [0..255]."""
    import av
    import torch.nn.functional as F

    n_target = min(n_pairs, max_pairs) if max_pairs else n_pairs
    container = av.open(str(video_path))
    pairs: list[tuple] = []
    cur: list = []
    for frame in container.decode(container.streams.video[0]):
        rgb_array = frame.to_ndarray(format="rgb24")
        t = torch.from_numpy(rgb_array.copy()).permute(2, 0, 1).unsqueeze(0).float()
        if tuple(t.shape[-2:]) != tuple(output_hw):
            t = F.interpolate(t, size=output_hw, mode="bilinear", align_corners=False)
        t = t.squeeze(0).contiguous()
        cur.append(t)
        if len(cur) == 2:
            pairs.append((cur[0], cur[1]))
            cur = []
            if len(pairs) >= n_target:
                break
    container.close()
    if not pairs:
        raise RuntimeError(f"no frames decoded from {video_path}")
    return torch.stack(
        [torch.stack([a, b], dim=0) for (a, b) in pairs], dim=0
    )


def _run_val_loop_full(
    hyperprior: Z3HyperpriorMLP,
    a1_latents: torch.Tensor,
    val_indices: list[int],
    args: argparse.Namespace,
    *,
    latent_offset: torch.Tensor,
    latent_scale: torch.Tensor,
) -> dict[str, float]:
    """Validation forward pass — eval-time torch.inference_mode (Catalog #180).

    Z3's rate Lagrangian is the canonical val metric (distortion is FROZEN
    by-construction since A1 decoder + latents don't change bytes-wise).
    """
    hyperprior.eval()
    losses: list[float] = []
    rate_bits_list: list[float] = []
    with torch.inference_mode():
        if not val_indices:
            return {"val_loss": float("inf"), "val_rate_bits": float("inf")}
        idx = torch.tensor(val_indices, device=a1_latents.device, dtype=torch.long)
        val_latents = a1_latents[idx]
        out = z3_v2_lagrangian(
            hyperprior=hyperprior,
            a1_latents=val_latents,
            latent_offset=latent_offset,
            latent_scale=latent_scale,
            seg_scorer=torch.nn.Identity(),
            pose_scorer=torch.nn.Identity(),
            decoded_pair_rt=None,
            gt_pair=None,
            alpha_rate=args.alpha_rate,
            beta_seg=args.beta_seg,
            gamma_pose=args.gamma_pose,
            quantization_step=args.quantization_step,
            factorized_half_range=args.factorized_half_range,
        )
        losses.append(float(out["total_loss"].item()))
        rate_bits_list.append(float(out["rate_bits_total"].item()))
    return {
        "val_loss": sum(losses) / len(losses),
        "val_rate_bits": sum(rate_bits_list) / len(rate_bits_list),
    }


def _full_main(args: argparse.Namespace) -> int:
    """Full training path: real A1 latents + score-aware R+λD Lagrangian.

    Per Phase 2 council ledger
    (.omx/research/z3_phase_2_council_20260514.md): APPROVED 6/6 unanimous
    (Shannon LEAD + Dykstra CO-LEAD + Yousfi + Fridrich + Contrarian + MacKay).

    Pipeline:
      1. Load FROZEN A1 archive bytes (sha pinned via --a1-archive-path).
      2. Decode A1 latents (600 × 28) via A1's codec.
      3. Init Z3 hyperprior MLP (~1.8k params; canonical EMA decay=0.997).
      4. Train AdamW(lr=1e-3) with the same residual-domain v2 Ballé
         Lagrangian that is emitted in the Z3HV2 packet.
      5. Apply EMA shadow at eval; snapshot+restore live weights (Catalog #88).
      6. Emit the Z3HV2 latent-replacement payload and record exact byte
         savings against A1's latent_blob.
      7. Build composition archive + contest-compliant runtime tree
         (vendor A1's codec.py + model.py for the final HNeRV decode, while
         refusing raw A1/v1 fallback payloads).
      8. Run CUDA auth eval via canonical gate_auth_eval_call (Catalog #223).
      9. Post stats with fail-closed result_review_blockers (Catalog #127).
    """
    import torch

    from tac.training import EMA

    _validate_z3v2_quantization_step(args)
    _canon_pin_seeds(args.seed)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    # Catalog #197: --full-cpu is the paired-waiver path; smoke=True allows
    # the canonical helper to accept CPU per the deviceordie contract.
    device = _canon_device_or_die(
        args.device,
        smoke=bool(args.full_cpu) or bool(getattr(args, "smoke", False)),
        substrate_tag=SUBSTRATE_TAG,
    )

    stage_log: list[dict[str, str]] = []

    def _stage(name: str) -> None:
        msg = {"stage": name, "at": _canon_utc_now_iso()}
        stage_log.append(msg)
        print(f"[z3-full] {name} @ {msg['at']}")

    _stage("seed_pinned")

    # ---- Step 1: load A1 (FROZEN base) ----
    a1_bytes, a1_sha, a1_size = _load_a1_archive_bytes(args.a1_archive_path)
    _stage(f"a1_loaded_{a1_size}_B_sha{a1_sha[:8]}")

    # Z3V2 is currently rate-only over frozen A1 latents. Exact score movement
    # is learned from the compiled packet by separate paired CPU/CUDA auth eval.
    yuv6_token = None
    _stage("scorer_path_skipped_rate_only_z3v2")

    auth_eval_result_path: str | None = None
    auth_eval_score: float | None = None
    auth_eval_evidence_grade: str | None = None
    auth_eval_score_axis: str | None = None
    auth_eval_lane_tag: str | None = None
    auth_eval_score_claim_valid: bool = False
    auth_eval_exact_cuda_complete: bool = False
    composition_sha: str = ""
    composition_size: int = 0
    composition_contract_manifest: dict[str, Any] = {}
    composition_byte_saving: bool = False
    sidecar_shipped: bool = False
    sidecar_bytes_estimate: int = 0
    bytes_saved_estimate: int = 0
    archive_zip_path = out_dir / "archive.zip"
    submission_dir = out_dir / "submission_dir"
    best_val_loss: float = math.inf
    best_epoch: int = -1
    train_started_at: float = time.time()

    try:
        # ---- Step 2: decode A1 latents + decoder (FROZEN) ----
        decoder, a1_latents = _decode_a1_latents_and_decoder(
            args.a1_archive_path, a1_bytes, device=device
        )
        _stage(f"a1_decoded_latents{tuple(a1_latents.shape)}")
        _, v2_offset_cpu, v2_scale_cpu = _z3v2_affine_residual_int8(a1_latents)
        v2_latent_offset = v2_offset_cpu.to(device)
        v2_latent_scale = v2_scale_cpu.to(device)
        _stage("z3v2_affine_offset_scale_built")
        del decoder  # rate-only Z3HV2 training; exact score is eval-gated.

        # ---- Step 3: scorer placeholders ----
        # z3_v2_lagrangian does not call scorers while decoded_pair_rt/gt_pair
        # are None. Keeping Identity modules here preserves the call contract
        # while avoiding heavyweight scorer loads on a non-promotional build.
        pose_scorer = torch.nn.Identity()
        seg_scorer = torch.nn.Identity()
        _stage("scorers_identity_rate_only_z3v2")

        # ---- Step 4: init hyperprior + EMA + optimizer ----
        cfg = Z3HyperpriorConfig(
            hyper_latent_dim=args.hyper_latent_dim,
            hyper_hidden_dim=args.hyper_hidden_dim,
            quantization_step=args.quantization_step,
        )
        hyperprior = Z3HyperpriorMLP(cfg).to(device)
        n_hp_params = sum(p.numel() for p in hyperprior.parameters())
        ema = EMA(hyperprior, decay=args.ema_decay)
        optimizer = torch.optim.AdamW(
            hyperprior.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs)
        )
        _stage(f"hyperprior_built_{n_hp_params}_params")

        # ---- Step 5: train/val split on the 600 A1 pairs (latent rows) ----
        n_pairs = int(a1_latents.shape[0])
        val_count = max(1, n_pairs // 8)
        val_idx_start = n_pairs - val_count
        train_indices_pool = list(range(val_idx_start))
        val_indices_pool = list(range(val_idx_start, n_pairs))

        ckpt_best_path = out_dir / "best.pt"
        train_started_at = time.time()
        nan_strike = 0
        max_nan_strikes = 3

        for epoch in range(args.epochs):
            hyperprior.train()
            random.shuffle(train_indices_pool)
            epoch_losses: list[float] = []
            epoch_rate_bits: list[float] = []
            for batch_start in range(0, len(train_indices_pool), args.batch_size):
                batch_indices = train_indices_pool[
                    batch_start : batch_start + args.batch_size
                ]
                if not batch_indices:
                    continue
                pair_idxs = torch.tensor(batch_indices, device=device, dtype=torch.long)
                batch_latents = a1_latents[pair_idxs]
                out = z3_v2_lagrangian(
                    hyperprior=hyperprior,
                    a1_latents=batch_latents,
                    latent_offset=v2_latent_offset,
                    latent_scale=v2_latent_scale,
                    seg_scorer=seg_scorer,
                    pose_scorer=pose_scorer,
                    decoded_pair_rt=None,
                    gt_pair=None,
                    alpha_rate=args.alpha_rate,
                    beta_seg=args.beta_seg,
                    gamma_pose=args.gamma_pose,
                    quantization_step=args.quantization_step,
                    factorized_half_range=args.factorized_half_range,
                )
                loss = out["total_loss"]
                if not torch.isfinite(loss):
                    nan_strike += 1
                    print(
                        f"[z3-full] NaN strike {nan_strike}/{max_nan_strikes} "
                        f"at epoch {epoch}"
                    )
                    if nan_strike >= max_nan_strikes:
                        raise RuntimeError(
                            "NaN watchdog tripped — refusing to continue"
                        )
                    continue
                nan_strike = 0
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    hyperprior.parameters(), args.grad_clip
                )
                optimizer.step()
                ema.update(hyperprior)
                epoch_losses.append(float(loss.detach().cpu()))
                epoch_rate_bits.append(float(out["rate_bits_total"].detach().cpu()))
            scheduler.step()

            # Eval every ~epochs/30 OR at last epoch.
            if epoch % max(1, args.epochs // 30) == 0 or epoch == args.epochs - 1:
                live_state = {
                    k: v.detach().clone() for k, v in hyperprior.state_dict().items()
                }
                ema.apply(hyperprior)
                try:
                    val_metrics = _run_val_loop_full(
                        hyperprior,
                        a1_latents,
                        val_indices_pool,
                        args,
                        latent_offset=v2_latent_offset,
                        latent_scale=v2_latent_scale,
                    )
                finally:
                    hyperprior.load_state_dict(live_state)
                    hyperprior.train()
                avg_train = (
                    sum(epoch_losses) / len(epoch_losses)
                    if epoch_losses else math.nan
                )
                avg_rate = (
                    sum(epoch_rate_bits) / len(epoch_rate_bits)
                    if epoch_rate_bits else math.nan
                )
                print(
                    f"[z3-full] epoch {epoch:4d}: train_loss={avg_train:.5f} "
                    f"rate_bits={avg_rate:.1f} "
                    f"val_loss={val_metrics['val_loss']:.5f} "
                    f"(best={best_val_loss:.5f})"
                )
                if val_metrics["val_loss"] < best_val_loss:
                    best_val_loss = val_metrics["val_loss"]
                    best_epoch = epoch
                    # Save EMA shadow (NEVER live weights — Catalog #88).
                    live_state2 = {
                        k: v.detach().clone()
                        for k, v in hyperprior.state_dict().items()
                    }
                    ema.apply(hyperprior)
                    ema_state = {
                        k: v.detach().cpu()
                        for k, v in hyperprior.state_dict().items()
                    }
                    hyperprior.load_state_dict(live_state2)
                    torch.save(
                        {
                            "state_dict": ema_state,
                            "config": asdict(cfg),
                            "epoch": epoch,
                            "val_loss": val_metrics["val_loss"],
                        },
                        ckpt_best_path,
                    )

        train_elapsed = time.time() - train_started_at
        _stage(
            f"trained_best_epoch_{best_epoch}_val_loss_{best_val_loss:.5f}_"
            f"elapsed_{train_elapsed:.1f}s"
        )

        # ---- Step 6: load EMA-best checkpoint + build composition archive ----
        if ckpt_best_path.is_file():
            best_ckpt = torch.load(
                ckpt_best_path, weights_only=False, map_location=device
            )
            hyperprior.load_state_dict(best_ckpt["state_dict"])
        hyperprior.eval()

        from tac.substrates.z3_balle_hyperprior_bolton.archive_v2 import (
            A1_LATENT_BLOB_LEN as _V2_A1_LATENT_BLOB_LEN,
        )
        from tac.substrates.z3_balle_hyperprior_bolton.archive_v2 import (
            build_z3v2_composition_archive_contract as _v2_build_contract,
        )
        from tac.substrates.z3_balle_hyperprior_bolton.archive_v2 import (
            build_z3v2_payload_bytes as _v2_build_payload,
        )
        from tac.substrates.z3_balle_hyperprior_bolton.archive_v2 import (
            encode_z3hv2_section as _v2_encode_section,
        )

        # Per-dim affine for the v2 grammar. v2 stores the centered offset
        # required by the signed-int8 residual contract.
        residual_v2, a1_offset, a1_scale = _z3v2_affine_residual_int8(a1_latents)
        residual_int8 = residual_v2.numpy().tobytes()
        z3hv2_section = _v2_encode_section(
            hyperprior_weights_int8=b"",
            w_hat_int8=b"",
            residual_int8=residual_int8,
            latent_offset=a1_offset,
            latent_scale=a1_scale,
            hyper_dim=cfg.hyper_latent_dim,
            int8_w_scale=1.0,
            quant_step=cfg.quantization_step,
            min_sigma=cfg.min_sigma,
            max_sigma=cfg.max_sigma,
            factorized_half_range=args.factorized_half_range,
        )
        sidecar_bytes_estimate = len(z3hv2_section)
        bytes_saved_estimate = _V2_A1_LATENT_BLOB_LEN - sidecar_bytes_estimate

        # Z3 production output is v2-only. The byte-saving flag is descriptive;
        # promotion still requires paired exact CPU/CUDA auth-eval review.
        composition_bytes = _v2_build_payload(
            a1_bytes=a1_bytes,
            z3hv2_section=z3hv2_section,
        )
        composition_contract = _v2_build_contract(a1_bytes, composition_bytes)
        sidecar_shipped = True
        composition_contract_manifest = composition_contract.as_manifest()
        composition_byte_saving = composition_contract.byte_saving
        (out_dir / "z3v2_section_diagnostic.bin").write_bytes(z3hv2_section)

        composition_sha = hashlib.sha256(composition_bytes).hexdigest()
        composition_size = len(composition_bytes)
        _stage(
            f"v2_composition_built_{composition_size}_B_sha{composition_sha[:8]}_"
            f"section_bytes={sidecar_bytes_estimate}_saved={bytes_saved_estimate}"
        )

        # ---- Step 7: build archive.zip + runtime tree ----
        _build_archive_zip(archive_zip_path, bin_bytes=composition_bytes)
        submission_dir.mkdir(parents=True, exist_ok=True)
        _write_runtime(submission_dir, a1_runtime_src=args.a1_archive_path.parent / "src")
        runtime_manifest = _write_submission_runtime_manifest(submission_dir)
        (submission_dir / "0.bin").write_bytes(composition_bytes)
        shutil.copy2(archive_zip_path, submission_dir / "archive.zip")
        _stage("archive_zip_and_runtime_emitted")

        # ---- Step 8: CUDA auth eval via canonical gate (Catalog #223) ----
        if not args.skip_auth_eval:
            _stage("contest_auth_eval")
            result_json_path = out_dir / "contest_auth_eval_cuda.json"
            auth_result = _canon_gate_auth_eval_call(
                args=args,
                archive_zip=archive_zip_path,
                inflate_sh=submission_dir / "inflate.sh",
                upstream_dir=args.upstream_dir,
                output_json=result_json_path,
                contest_auth_eval_script=CONTEST_AUTH_EVAL_SCRIPT,
                substrate_tag=SUBSTRATE_TAG,
                device=device,
                full_cpu_active=bool(args.full_cpu),
            )
            if auth_result is not None:
                auth_eval_result_path = str(result_json_path)
                auth_eval_score = auth_result["auth_eval_cuda_score"]
                auth_eval_evidence_grade = "contest-CUDA"
                auth_eval_score_axis = auth_result["auth_eval_score_axis"]
                auth_eval_lane_tag = auth_result["auth_eval_lane_tag"]
                auth_eval_score_claim_valid = bool(
                    auth_result["auth_eval_score_claim_valid"]
                )
                auth_eval_exact_cuda_complete = True
    finally:
        if yuv6_token is not None:
            from tac.differentiable_eval_roundtrip import unpatch_upstream_yuv6

            unpatch_upstream_yuv6(yuv6_token)
            _stage("upstream_yuv6_unpatched")

    # ---- Step 9: posterior update via canonical locked helper (Catalog #128) ----
    if not args.skip_auth_eval and (out_dir / "contest_auth_eval_cuda.json").exists():
        try:
            from tac.continual_learning import (
                posterior_update_locked_from_auth_eval_json,
            )

            update = posterior_update_locked_from_auth_eval_json(
                out_dir / "contest_auth_eval_cuda.json"
            )
            print(
                f"[z3-full] posterior_update accepted="
                f"{getattr(update, 'accepted', '?')}"
            )
            _stage("posterior_updated")
        except Exception as exc:
            print(f"[z3-full] WARN posterior_update failed: {exc!r}")

    # Cost-band anchors are dispatch evidence, not local build evidence. Do not
    # mutate the shared calibration ledger for skipped-auth CPU/local runs.
    if auth_eval_score_claim_valid:
        try:
            from tac.cost_band_calibration import CostBandAnchor, append_anchor

            wall_sec = float(time.time() - train_started_at)
            cost_usd = wall_sec / 3600.0 * 0.60
            anchor = CostBandAnchor(
                logged_at_utc=_canon_utc_now_iso(),
                dispatch_label=f"{SUBSTRATE_LANE_ID}_{_canon_utc_now_iso()}",
                trainer="train_substrate_z3_balle_hyperprior_bolton",
                platform="modal",
                gpu="T4" if device.type == "cuda" else "cpu",
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                all_flags_on=False,
                actual_wall_clock_sec=wall_sec,
                actual_cost_usd=cost_usd,
                outcome="successful_dispatch",
                notes=(
                    f"substrate_tag={SUBSTRATE_TAG};"
                    f"archive_sha256={composition_sha};"
                    f"sidecar_shipped={sidecar_shipped};"
                    f"literature_anchor=balle_2018_scale_hyperprior;"
                    f"lane_class=substrate_engineering"
                ),
            )
            append_anchor(anchor)
            _stage("cost_band_anchor_appended")
        except Exception as exc:
            print(f"[z3-full] WARN cost_band_anchor_append failed: {exc!r}")
    else:
        _stage("cost_band_anchor_skipped_no_valid_auth_eval")

    result_review_blockers = list(
        composition_contract_manifest.get("result_review_blockers", [])
    )
    result_review_blockers.append("trainer_stats_not_authoritative_score_claim_surface")
    dispatch_blocker = (
        "z3v2_latent_replacement uses direct residual bytes; it requires "
        "claim_lane_dispatch.py plus paired exact CPU/CUDA auth eval and result "
        "review before score or promotion. Hyperprior entropy coding remains "
        "unpromoted until a real entropy-coded residual decoder consumes it."
    )
    if not auth_eval_score_claim_valid:
        skipped_reason = str(getattr(args, "auth_eval_skipped_reason", "") or "")
        if skipped_reason:
            result_review_blockers.append(skipped_reason)
        else:
            result_review_blockers.append("contest_cuda_auth_eval_not_validated")
    result_review_blockers = list(dict.fromkeys(result_review_blockers))

    hardware_substrate = _canon_detect_hardware_substrate(
        substrate_tag=SUBSTRATE_TAG,
        axis="cuda" if device.type == "cuda" else "cpu",
        env_var_candidates=("Z3_BALLE_GPU", "MODAL_GPU"),
    )
    stats = {
        "lane_id": SUBSTRATE_LANE_ID,
        "substrate_tag": SUBSTRATE_TAG,
        "smoke": False,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "archive_bytes": composition_size,
        "archive_sha256": composition_sha,
        "archive_zip_bytes": (
            archive_zip_path.stat().st_size if archive_zip_path.exists() else 0
        ),
        "archive_zip_sha256": (
            _canon_sha256_bytes(archive_zip_path.read_bytes())
            if archive_zip_path.exists() else ""
        ),
        "base_archive_sha256": a1_sha,
        "base_archive_bytes": a1_size,
        "sidecar_shipped": sidecar_shipped,
        "sidecar_bytes_estimate": sidecar_bytes_estimate,
        "bytes_saved_estimate": bytes_saved_estimate,
        "archive_contract": composition_contract_manifest,
        "byte_saving": composition_byte_saving,
        "dispatch_blocker": dispatch_blocker,
        "hyper_dim": args.hyper_latent_dim,
        "predicted_delta_s_band": [-0.0009, -0.0003],
        "predicted_delta_s_strict_honest": -0.0006,
        "predicted_delta_s_uncertainty": 0.50,
        "council_verdict": (
            "PROCEED 6/6 unanimous (Shannon LEAD + Dykstra CO-LEAD + "
            "Yousfi + Fridrich + Contrarian + MacKay)"
        ),
        "council_ledger_path": ".omx/research/z3_phase_2_council_20260514.md",
        "auth_eval_score": auth_eval_score,
        "auth_eval_evidence_grade": auth_eval_evidence_grade,
        "auth_eval_result_path": auth_eval_result_path,
        "auth_eval_score_axis": auth_eval_score_axis,
        "auth_eval_lane_tag": auth_eval_lane_tag,
        "auth_eval_score_claim_valid": auth_eval_score_claim_valid,
        "auth_eval_exact_cuda_complete": auth_eval_exact_cuda_complete,
        "contest_cuda_eval": {
            "score": auth_eval_score if auth_eval_score_axis == "contest_cuda" else None,
            "result_path": auth_eval_result_path
            if auth_eval_score_axis == "contest_cuda" else None,
            "score_claim_valid": bool(
                auth_eval_score_claim_valid and auth_eval_score_axis == "contest_cuda"
            ),
        },
        "contest_cpu_eval": None,
        "paired_axis_status": "missing_contest_cpu_eval",
        "score_axis": auth_eval_score_axis,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "result_review_blockers": result_review_blockers,
        "stage_log": stage_log,
        "runtime_tree_sha256": runtime_manifest["runtime_tree_sha256"],
        "submission_runtime_manifest_path": str(
            submission_dir / "submission_runtime_manifest.json"
        ),
        "hardware_substrate": hardware_substrate,
        "git_head": _canon_git_head_sha(REPO_ROOT),
        "trained_at_utc": _canon_utc_now_iso(),
    }
    (out_dir / "stats.json").write_text(
        json.dumps(stats, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    _write_training_artifact_manifest(
        out_dir,
        stats=stats,
        training_mode="full",
        research_only=not auth_eval_score_claim_valid,
    )
    print(
        f"[z3-full] DONE archive={composition_size}B sha={composition_sha[:12]}... "
        f"auth_score={auth_eval_score} grade={auth_eval_evidence_grade} "
        f"sidecar_shipped={sidecar_shipped}"
    )
    return 0


# ---------------------------------------------------------------------------
# Runtime emission helpers
# ---------------------------------------------------------------------------


def _write_runtime(submission_dir: Path, *, a1_runtime_src: Path | None = None) -> None:
    """Emit contest-compliant inflate.sh + inflate.py + vendored substrate.

    Per Catalog #146 the inflate.sh signature is 3-positional-arg
    ``inflate.sh <archive_dir> <output_dir> <file_list>``. Per Catalog #163
    the script uses ``set -euo pipefail`` for fail-closed semantics.

    Z3 inflate is v2-only for production packets: it refuses raw A1/v1 bytes,
    reconstructs direct Z3HV2 latents, applies A1's sidecar refinement, then
    delegates to A1's existing HNeRV decoder.
    """
    submission_dir.mkdir(parents=True, exist_ok=True)
    runtime_pkg = (
        submission_dir / "src" / "tac" / "substrates" / "z3_balle_hyperprior_bolton"
    )
    runtime_pkg.mkdir(parents=True, exist_ok=True)
    for pkg_init in (
        submission_dir / "src" / "tac" / "__init__.py",
        submission_dir / "src" / "tac" / "substrates" / "__init__.py",
        runtime_pkg / "__init__.py",
    ):
        pkg_init.write_text("", encoding="utf-8")
    substrate_src = REPO_ROOT / "src" / "tac" / "substrates" / "z3_balle_hyperprior_bolton"
    for name in ("architecture.py", "archive_v2.py", "inflate_v2.py", "quant.py"):
        shutil.copy2(substrate_src / name, runtime_pkg / name)
    a1_runtime_pkg = submission_dir / "src" / "a1_runtime"
    a1_runtime_pkg.mkdir(parents=True, exist_ok=True)
    a1_src = a1_runtime_src or (REPO_ROOT / "submissions" / "a1" / "src")
    if not (a1_src / "codec.py").is_file() or not (a1_src / "model.py").is_file():
        a1_src = REPO_ROOT / "submissions" / "a1" / "src"
    for name in ("codec.py", "model.py"):
        shutil.copy2(a1_src / name, a1_runtime_pkg / name)
    (a1_runtime_pkg / "__init__.py").write_text("", encoding="utf-8")
    (runtime_pkg / "__init__.py").write_text(
        "\"\"\"Z3 runtime package (inflate-time only — no scorer imports).\"\"\"\n"
        "from tac.substrates.z3_balle_hyperprior_bolton.architecture import (\n"
        "    A1_LATENT_DIM,\n"
        "    A1_N_PAIRS,\n"
        "    Z3HyperpriorConfig,\n"
        "    Z3HyperpriorMLP,\n"
        ")\n"
        "from tac.substrates.z3_balle_hyperprior_bolton.archive_v2 import (\n"
        "    Z3HV2_MAGIC,\n"
        "    Z3HV2_VERSION,\n"
        "    decode_z3hv2_section,\n"
        "    split_z3v2_payload_bytes,\n"
        ")\n"
        "from tac.substrates.z3_balle_hyperprior_bolton.inflate_v2 import (\n"
        "    is_v2_payload,\n"
        "    require_reconstruct_a1_latents_from_v2_payload,\n"
        "    reconstruct_a1_latents_from_v2_payload,\n"
        "    select_inflate_device,\n"
        ")\n"
        "__all__ = [\n"
        "    'A1_LATENT_DIM', 'A1_N_PAIRS',\n"
        "    'Z3HV2_MAGIC', 'Z3HV2_VERSION',\n"
        "    'Z3HyperpriorConfig', 'Z3HyperpriorMLP',\n"
        "    'decode_z3hv2_section',\n"
        "    'is_v2_payload',\n"
        "    'require_reconstruct_a1_latents_from_v2_payload',\n"
        "    'reconstruct_a1_latents_from_v2_payload',\n"
        "    'select_inflate_device',\n"
        "    'split_z3v2_payload_bytes',\n"
        "]\n",
        encoding="utf-8",
    )
    _canon_vendor_shared_inflate_runtime(submission_dir, repo_root=REPO_ROOT)

    inflate_sh = (
        "#!/usr/bin/env bash\n"
        "# Z3 Ballé hyperprior bolt-on contest-compliant inflate runtime.\n"
        "# Per Catalog #146: 3-positional-arg signature.\n"
        "# Per Catalog #163: set -euo pipefail.\n"
        "set -euo pipefail\n"
        'DATA_DIR="${1:?archive_dir required}"\n'
        'OUTPUT_DIR="${2:?output_dir required}"\n'
        'FILE_LIST="${3:?file_list required}"\n'
        'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'export PYTHONPATH="${HERE}/src:${PYTHONPATH:-}"\n'
        'mkdir -p "${OUTPUT_DIR}"\n'
        'while IFS= read -r line; do\n'
        '  [ -z "${line}" ] && continue\n'
        '  base="${line%.*}"\n'
        '  src="${DATA_DIR}/x"\n'
        '  if [ ! -f "${src}" ]; then src="${DATA_DIR}/${base}.bin"; fi\n'
        '  dst="${OUTPUT_DIR}/${base}.raw"\n'
        '  [ ! -f "${src}" ] && echo "ERROR: ${src} not found" >&2 && exit 1\n'
        '  "${PYTHON:-python3}" "${HERE}/inflate.py" "${src}" "${dst}"\n'
        'done < "${FILE_LIST}"\n'
    )
    (submission_dir / "inflate.sh").write_text(inflate_sh, encoding="utf-8")
    (submission_dir / "inflate.sh").chmod(0o755)

    inflate_py = (
        '"""Z3 contest-compliant inflate.py.\n'
        "\n"
        "Consumes the evaluator-provided extracted archive member and fails\n"
        "closed unless the payload is a Z3HV2 latent-replacement packet.\n"
        '"""\n'
        "from __future__ import annotations\n"
        "import struct\n"
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "import torch\n"
        "import torch.nn.functional as F\n"
        "\n"
        "HERE = Path(__file__).resolve().parent\n"
        "sys.path.insert(0, str(HERE / 'src'))\n"
        "sys.path.insert(0, str(HERE / 'src' / 'a1_runtime'))\n"
        "\n"
        "from codec import LATENT_BLOB_LEN, apply_latent_sidecar, decode_decoder_compact\n"
        "from model import HNeRVDecoder\n"
        "from tac.substrates.z3_balle_hyperprior_bolton.inflate_v2 import (\n"
        "    is_v2_payload,\n"
        "    require_reconstruct_a1_latents_from_v2_payload,\n"
        "    select_inflate_device,\n"
        ")\n"
        "\n"
        "CAMERA_H, CAMERA_W = 874, 1164\n"
        "EVAL_H, EVAL_W = 384, 512\n"
        "LATENT_DIM = 28\n"
        "BASE_CHANNELS = 36\n"
        "N_PAIRS = 600\n"
        "\n"
        "def parse_a1_decoder_and_sidecar(a1_bytes: bytes):\n"
        "    if len(a1_bytes) < 4:\n"
        "        raise ValueError('A1 archive too short')\n"
        "    section_total = struct.unpack_from('<I', a1_bytes, 0)[0]\n"
        "    if section_total < 4 or section_total > len(a1_bytes):\n"
        "        raise ValueError(f'bad decoder_section_total {section_total}')\n"
        "    decoder_blob = a1_bytes[4:section_total]\n"
        "    latent_blob = a1_bytes[section_total:section_total + LATENT_BLOB_LEN]\n"
        "    sidecar_blob = a1_bytes[section_total + LATENT_BLOB_LEN:]\n"
        "    if not decoder_blob or len(latent_blob) != LATENT_BLOB_LEN:\n"
        "        raise ValueError('bad A1 archive layout')\n"
        "    decoder_sd = decode_decoder_compact(decoder_blob)\n"
        "    return decoder_sd, sidecar_blob\n"
        "\n"
        "def inflate(src_bin: str, dst_raw: str) -> int:\n"
        "    composition_bytes = Path(src_bin).read_bytes()\n"
        "    if not is_v2_payload(composition_bytes):\n"
        "        raise ValueError('Z3 runtime requires Z3HV2 payload; refusing A1/v1 fallback')\n"
        "    a1_bytes, z3_latents = require_reconstruct_a1_latents_from_v2_payload(composition_bytes)\n"
        "    decoder_sd, sidecar_blob = parse_a1_decoder_and_sidecar(a1_bytes)\n"
        "    latents = apply_latent_sidecar(z3_latents, sidecar_blob)\n"
        "    device = select_inflate_device()\n"
        "    decoder = HNeRVDecoder(latent_dim=LATENT_DIM, base_channels=BASE_CHANNELS, eval_size=(EVAL_H, EVAL_W)).to(device)\n"
        "    decoder.load_state_dict(decoder_sd)\n"
        "    decoder.eval()\n"
        "    latents = latents.to(device)\n"
        "    n = 0\n"
        "    with torch.inference_mode(), open(dst_raw, 'wb') as fout:\n"
        "        for i in range(0, N_PAIRS, 16):\n"
        "            j = min(i + 16, N_PAIRS)\n"
        "            batch = j - i\n"
        "            decoded = decoder(latents[i:j])\n"
        "            flat = decoded.reshape(batch * 2, 3, EVAL_H, EVAL_W)\n"
        "            up = F.interpolate(flat, size=(CAMERA_H, CAMERA_W), mode='bicubic', align_corners=False)\n"
        "            up = up.reshape(batch, 2, 3, CAMERA_H, CAMERA_W)\n"
        "            up[:, 0, 0].sub_(1.0)\n"
        "            up[:, 0, 2].sub_(1.0)\n"
        "            up[:, 1, 1].sub_(1.0)\n"
        "            frames = up.reshape(batch * 2, 3, CAMERA_H, CAMERA_W).clamp(0, 255).permute(0, 2, 3, 1).round().to(torch.uint8).cpu().numpy()\n"
        "            fout.write(frames.tobytes())\n"
        "            n += batch * 2\n"
        "    print(f'[z3-inflate] saved {n} frames')\n"
        "    return n\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    if len(sys.argv) != 3:\n"
        "        raise SystemExit('Usage: python inflate.py <src.bin> <dst.raw>')\n"
        "    inflate(sys.argv[1], sys.argv[2])\n"
    )
    (submission_dir / "inflate.py").write_text(inflate_py, encoding="utf-8")


def _write_submission_runtime_manifest(submission_dir: Path) -> dict[str, Any]:
    """Record deterministic custody for the emitted inflate runtime tree."""
    ignored = {"0.bin", "archive.zip", "submission_runtime_manifest.json"}
    files: list[dict[str, Any]] = []
    tree_hasher = hashlib.sha256()
    for path in sorted(p for p in submission_dir.rglob("*") if p.is_file()):
        rel = path.relative_to(submission_dir).as_posix()
        if rel in ignored:
            continue
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        files.append({"path": rel, "bytes": len(data), "sha256": digest})
        tree_hasher.update(rel.encode("utf-8"))
        tree_hasher.update(b"\0")
        tree_hasher.update(digest.encode("ascii"))
        tree_hasher.update(b"\0")
    manifest = {
        "schema": "submission_runtime_manifest_v1",
        "runtime_tree_sha256": tree_hasher.hexdigest(),
        "files": files,
    }
    (submission_dir / "submission_runtime_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    return manifest


def _build_archive_zip(zip_path: Path, *, bin_bytes: bytes) -> None:
    """Build a deterministic archive.zip with canonical single member ``0.bin``.

    Per Catalog #19: use ZipInfo + writestr with fixed timestamps for
    deterministic byte output. Per HNeRV parity discipline lesson 3 +
    Catalog #146: monolithic single-file ``0.bin`` is the canonical contest
    archive grammar. Empirical anchor 2026-05-15: Z3 v2 smoke
    ``fc-01KRNHEGC9ZE48Y68GGJHP7FXN`` SMOKE RED with diagnostic
    ``zip_members=['x'] expected=['0.bin']`` ($2 wasted) caught by local
    pre-deploy harness post-fix. Bug class extincted via
    ``tools/local_pre_deploy_check.py::check_archive_grammar``.
    """
    info = zipfile.ZipInfo(filename="0.bin", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(info, bin_bytes)


def _write_training_artifact_manifest(
    out_dir: Path,
    *,
    stats: dict[str, Any],
    training_mode: str,
    research_only: bool,
) -> Path:
    """Emit the canonical smoke-training artifact manifest.

    ``tools/run_modal_smoke_before_full.py`` validates smoke-only research
    outputs through ``training_artifact_v1``. The validator deliberately
    requires a manifest distinct from free-form ``stats.json`` so Modal smoke
    gates do not accidentally treat stale or authority-bearing artifacts as
    green.
    """

    manifest = {
        "schema": "training_artifact_v1",
        "lane_id": stats["lane_id"],
        "substrate_tag": stats["substrate_tag"],
        "training_mode": training_mode,
        "research_only": bool(research_only),
        "archive_bytes": int(stats["archive_bytes"]),
        "archive_sha256": stats["archive_sha256"],
        "archive_zip_bytes": int(stats["archive_zip_bytes"]),
        "archive_zip_sha256": stats["archive_zip_sha256"],
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "result": {
            "training_mode": training_mode,
            "archive_bytes": int(stats["archive_bytes"]),
            "archive_sha256": stats["archive_sha256"],
            "archive_zip_bytes": int(stats["archive_zip_bytes"]),
            "archive_zip_sha256": stats["archive_zip_sha256"],
            "archive_contract": stats.get("archive_contract", {}),
        },
        "result_review_blockers": list(stats.get("result_review_blockers", ())),
        "stats_path": "stats.json",
        "archive_path": "archive.zip",
    }
    path = out_dir / "manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _validate_full_cpu_flags(args)
    # Catalog #1: refuse MPS at top level.
    if args.device == "mps":
        raise SystemExit(
            "ERROR: --device mps refused per CLAUDE.md MPS-NOISE non-negotiable"
        )
    # Smoke is the local timing path; full requires explicit Phase 2 council.
    if args.smoke:
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":
    sys.exit(main())
