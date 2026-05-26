# SPDX-License-Identifier: MIT
"""Train the Z3-G1 scorer-class-conditional gating substrate.

Per Wunderkind G1 SUBSTITUTION-1:1 spec
(``feedback_wunderkind_visionary_scorer_as_cooperative_receiver_paradigm_shift_20260515.md``)
+ Phase B.1 evidence-review council binding verdict (operator approved 2026-05-15
$110-170 Tier 2 envelope, paired CPU+CUDA Modal T4 ~$5-10).

G1 REPLACES the Z3 v2 Ballé hyperprior MLP (`Z3HyperpriorMLP`,
~1.8k params, ~1KB int8 brotli'd) with a per-class sigma TABLE
(`Z3G1ScorerClassGatingHead`, 5 classes * 28 dims = 140 params, ~80B int8
brotli'd) + per-pair SegNet-class index (600 bytes int8 -> ~200B brotli).

Predicted ΔS = -0.005 to -0.015 vs A1 0.1928 [contest-CPU 1to1]
``[prediction; first-principles-bound]``. Cost ~$5-10 (paired CPU+CUDA
Modal T4 anchor).

Council-binding contract (CLAUDE.md non-negotiables) honored end-to-end:

- Train against frozen A1 latents decoded from the A1 archive (NOT synthetic
  data; Catalog #114 forbids synthetic non-smoke data).
- Patch upstream ``rgb_to_yuv6`` via ``patch_upstream_yuv6_globally`` BEFORE
  scorer construction (Catalog #187; PR #95/#106 contract). Required because
  G1 ACTUALLY runs SegNet on GT pairs at training time to compute per-pair
  class indices.
- ``pose_scorer, seg_scorer = load_differentiable_scorers(...)`` per Catalog
  #222 (canonical loader returns (posenet, segnet)).
- ``tac.training.EMA(decay=0.997)`` update after every ``optimizer.step``;
  inference checkpoint = EMA shadow (Catalog #88).
- ``tac.substrates._shared.smoke_auth_eval_gate.gate_auth_eval_call`` per
  Catalog #226 (refuses hand-rolled subprocess invocations).
- TIER_1_OPERATOR_REQUIRED_FLAGS declared per Catalog #151 as
  ``ast.AnnAssign`` (Catalog #168 AST walker observes both forms).
- ``detect_hardware_substrate`` per Catalog #190.

Usage (smoke; CPU; ~3 epochs; uses synthetic A1 latents + uniform-class indices)::

    .venv/bin/python experiments/train_substrate_z3_g1_scorer_softmax_hyperprior_gating.py \\
        --a1-archive-path submissions/a1/archive.zip \\
        --output-dir experiments/results/z3g1_smoke_<utc> \\
        --epochs 3 --device cpu --smoke

Usage (full; CUDA-required; gated behind operator approval per Phase B.1)::

    .venv/bin/python experiments/train_substrate_z3_g1_scorer_softmax_hyperprior_gating.py \\
        --a1-archive-path submissions/a1/archive.zip \\
        --output-dir experiments/results/z3g1_<utc> \\
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

# Import canonical helpers per Catalog #226 / #190 / #197 / #88 / #128
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

# G1 substrate primitives.
from tac.substrates.z3_g1_scorer_softmax_hyperprior_gating import (
    A1_LATENT_DIM,
    A1_N_PAIRS,
    G1_NUM_SCORER_CLASSES,
    Z3G1Config,
    Z3G1ScorerClassGatingHead,
    estimate_z3g1_section_overhead_bytes,
    g1_per_pair_dominant_class_from_segnet_argmax,
    z3_g1_lagrangian,
)

# Reuse Z3 v2 archive grammar (single source of truth) + helper routines
# from the Z3 v2 trainer for runtime emission, archive zip, A1 decoding.
# The Z3 v2 trainer file is loaded as a module via importlib to avoid
# requiring `experiments/` to be on sys.path.
import importlib.util as _ilu

_Z3_V2_TRAINER_PATH = REPO_ROOT / "experiments" / "train_substrate_z3_balle_hyperprior_bolton.py"
_z3v2_spec = _ilu.spec_from_file_location("_z3_v2_trainer", _Z3_V2_TRAINER_PATH)
_z3v2_module = _ilu.module_from_spec(_z3v2_spec)
_z3v2_spec.loader.exec_module(_z3v2_module)
_build_archive_zip = _z3v2_module._build_archive_zip
_decode_a1_latents_and_decoder = _z3v2_module._decode_a1_latents_and_decoder
_load_a1_archive_bytes = _z3v2_module._load_a1_archive_bytes
_write_runtime = _z3v2_module._write_runtime
_write_submission_runtime_manifest = _z3v2_module._write_submission_runtime_manifest
_write_training_artifact_manifest = _z3v2_module._write_training_artifact_manifest
_z3v2_affine_residual_int8 = _z3v2_module._z3v2_affine_residual_int8


# ---------------------------------------------------------------------------
# Module paths + constants
# ---------------------------------------------------------------------------

DEFAULT_A1_ARCHIVE = REPO_ROOT / "submissions" / "a1" / "archive.zip"
DEFAULT_UPSTREAM_DIR = REPO_ROOT / "upstream"
DEFAULT_VIDEO_PATH = REPO_ROOT / "upstream" / "videos" / "0.mkv"
CONTEST_AUTH_EVAL_SCRIPT = REPO_ROOT / "experiments" / "contest_auth_eval.py"

SUBSTRATE_TAG = "z3_g1_scorer_softmax_hyperprior_gating"
SUBSTRATE_LANE_ID = "lane_z3_g1_scorer_softmax_hyperprior_gating_20260515"


# ---------------------------------------------------------------------------
# Catalog #151 manifest — every flag below must be threaded by any operator
# wrapper that subprocess-invokes this trainer. Annotated as ast.AnnAssign so
# Catalog #168's AST walker observes it (NOT bare Assign).
# ---------------------------------------------------------------------------
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--a1-archive-path": {
        "env": "Z3_G1_A1_ARCHIVE_PATH",
        "rationale": (
            "Z3-G1 is a bolt-on over the FROZEN A1 base; the A1 archive's "
            "latent_blob is decoded to obtain the per-pair latents that "
            "Z3-G1 re-encodes via the scorer-class-conditional gating head. "
            "Path is required for full training; smoke mode uses synthetic "
            "latents."
        ),
        "default": str(DEFAULT_A1_ARCHIVE.relative_to(REPO_ROOT)),
        "required_input_file": True,
        "generator_command": (
            "contest-pinned A1 archive — generated by submissions/a1 substrate"
        ),
    },
    "--video-path": {
        "env": "Z3_G1_VIDEO_PATH",
        "rationale": (
            "Contest video is required to decode GT frames for SegNet "
            "class-index computation (compress-side scorer use is FREE "
            "per CLAUDE.md 'Strict scorer rule' rule #2)."
        ),
        "default": str(DEFAULT_VIDEO_PATH.relative_to(REPO_ROOT)),
        "required_input_file": True,
        "generator_command": (
            "contest video — provided by upstream/videos/0.mkv"
        ),
    },
    "--upstream-dir": {
        "env": "Z3_G1_UPSTREAM_DIR",
        "rationale": (
            "Upstream module + scorer weight directory required for "
            "load_differentiable_scorers (PoseNet/SegNet weight load)."
        ),
        "default": str(DEFAULT_UPSTREAM_DIR.relative_to(REPO_ROOT)),
        "required_input_file": False,
    },
    "--epochs": {
        "env": "Z3_G1_EPOCHS",
        "rationale": "Number of training epochs (smoke=100, full=1000).",
        "default": "1000",
        "required_input_file": False,
    },
    "--device": {
        "env": "Z3_G1_DEVICE",
        "rationale": (
            "Device name (cuda for full; cpu for smoke). MPS refused per "
            "Catalog #1."
        ),
        "default": "cuda",
        "required_input_file": False,
    },
}


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Train Z3-G1 scorer-class-conditional gating substrate "
            "(Wunderkind G1 substitution; replaces Ballé hyperprior MLP "
            "with per-class sigma table indexed by SegNet's per-pair "
            "dominant class)."
        )
    )
    p.add_argument(
        "--a1-archive-path",
        type=Path,
        default=DEFAULT_A1_ARCHIVE,
        help="Path to A1 archive.zip (frozen base).",
    )
    p.add_argument(
        "--video-path",
        type=Path,
        default=DEFAULT_VIDEO_PATH,
        help="Path to contest video (compress-side SegNet input).",
    )
    p.add_argument(
        "--upstream-dir",
        type=Path,
        default=DEFAULT_UPSTREAM_DIR,
        help="Upstream module + scorer weight directory.",
    )
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--full-cpu", action="store_true")
    p.add_argument("--advisory-cpu-explicitly-waived", action="store_true")
    p.add_argument("--lr", type=float, default=1e-3)
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
    # G1-specific.
    p.add_argument(
        "--num-scorer-classes",
        type=int,
        default=G1_NUM_SCORER_CLASSES,
        help="Number of SegNet output classes (5 per upstream/modules.py).",
    )
    p.add_argument(
        "--init-sigma",
        type=float,
        default=2.0,
        help="Initial sigma value for the per-class table.",
    )
    # Post-train artifacts
    p.add_argument("--skip-auth-eval", action="store_true")
    p.add_argument("--skip-archive-build", action="store_true")
    p.add_argument("--enable-autocast-fp16", action="store_true")
    p.add_argument("--enable-torch-compile", action="store_true")
    p.add_argument("--enable-gt-scorer-cache", action="store_true")
    # ------------------------------------------------------------------
    # Codex review bkrbqet3p 2026-05-15 (F1 + F2) self-protection flags.
    # Catalog #266 + #267. See `_validate_g1_research_only_flags`.
    # ------------------------------------------------------------------
    p.add_argument(
        "--allow-uniform-class-fallback",
        action="store_true",
        help=(
            "F2 (Catalog #267) opt-in: SegNet class derivation may "
            "silently fall back to deterministic uniform class assignment "
            "when scorer load/forward fails. Default False = FAIL-CLOSED "
            "for full runs (raise SystemExit). Required-paired with "
            "--research-only-direct-residual because uniform-class "
            "assignment yields a synthetic class prior that does NOT "
            "represent the substrate's intended scorer-class gating; the "
            "produced archive is not a contest-promotion candidate."
        ),
    )
    p.add_argument(
        "--research-only-direct-residual",
        action="store_true",
        help=(
            "F1 (Catalog #266) opt-in: explicitly acknowledge that the "
            "G1 trainer currently emits a DIRECT-RESIDUAL Z3HV2 packet "
            "(empty hyperprior_weights_int8 + w_hat_int8 slots) and the "
            "learned sigma table + class indices land in g1_diagnostic.pt "
            "OUTSIDE the archive, so any auth-eval measures the direct-"
            "residual control NOT the advertised G1 scorer-class gating. "
            "When set: stamps research_only=true, skips auth-eval, "
            "labels artifact as `direct_residual_g1_diagnostic_control`. "
            "When unset: full_main raises SystemExit before any GPU spend."
        ),
    )
    return p


def _validate_full_cpu_flags(args: argparse.Namespace) -> None:
    """Catalog #197: --full-cpu MUST be paired with the advisory waiver flag."""
    if args.full_cpu and not args.advisory_cpu_explicitly_waived:
        raise SystemExit(
            "ERROR: --full-cpu requires --advisory-cpu-explicitly-waived per "
            "Catalog #197 (paired-flag attestation that the CPU-axis bypass "
            "is intentional and non-promotable)"
        )


def _validate_g1_research_only_flags(args: argparse.Namespace) -> None:
    """Codex bkrbqet3p F1 (Catalog #266) + F2 (Catalog #267) gating.

    The G1 trainer currently produces a DIRECT-RESIDUAL Z3HV2 packet
    (empty hyperprior_weights_int8 + w_hat_int8 slots — see
    train_substrate_z3_g1_scorer_softmax_hyperprior_gating.py:866-916).
    The learned sigma table + class indices land in ``g1_diagnostic.pt``
    OUTSIDE the archive; the runtime never consumes them. Any auth-eval
    therefore measures a direct-residual Z3HV2 control, NOT the advertised
    G1 scorer-class gating. Per codex review recommendation #1, the
    trainer must EITHER (a) encode the sigma table + class indices into
    ``0.bin`` AND verify the inflate consumes them via byte-mutation smoke
    (Catalog #139), OR (b) force ``research_only=true``, skip auth-eval,
    and label the artifact as a direct-residual control. Until path (a)
    lands, the trainer requires ``--research-only-direct-residual`` for
    full_main to run.

    Sister F2 gate: ``--allow-uniform-class-fallback`` must be opt-in.
    Without it, the silent ``except Exception`` path at line 662 is
    fail-closed: any SegNet load / forward / video-decode failure raises
    instead of producing a synthetic uniform class prior. With it, the
    flag MUST be paired with ``--research-only-direct-residual`` because
    a uniform class prior cannot represent the substrate's intended
    scorer-class gating regardless of archive consumption.
    """
    if not args.research_only_direct_residual:
        raise SystemExit(
            "ERROR: G1 full_main requires --research-only-direct-residual "
            "per Codex review bkrbqet3p F1 / Catalog #266. The current G1 "
            "trainer emits a DIRECT-RESIDUAL Z3HV2 packet (empty "
            "hyperprior_weights_int8 + w_hat_int8 slots; sigma table + "
            "class indices in g1_diagnostic.pt OUTSIDE the archive). The "
            "runtime never consumes the G1 bytes, so any auth-eval would "
            "measure a direct-residual Z3HV2 control NOT the advertised "
            "G1 scorer-class gating. Either land an inflate-time consumer "
            "+ byte-mutation smoke (Catalog #139) before re-running, OR "
            "pass --research-only-direct-residual to acknowledge the "
            "trainer is producing a research-only direct-residual control."
        )
    if args.allow_uniform_class_fallback and not args.research_only_direct_residual:
        # Defensive: above branch already covers, but make the pairing
        # contract explicit for any future refactor that relaxes the
        # outer gate.
        raise SystemExit(
            "ERROR: --allow-uniform-class-fallback requires "
            "--research-only-direct-residual per Codex review bkrbqet3p "
            "F2 / Catalog #267 (paired-flag attestation that the uniform "
            "class prior is non-promotable and that the artifact is "
            "labeled research_only=true)."
        )


# ---------------------------------------------------------------------------
# G1-specific helpers
# ---------------------------------------------------------------------------


def _compute_per_pair_class_indices_from_video(
    video_path: Path,
    upstream_dir: Path,
    n_pairs: int,
    *,
    device: torch.device,
    num_classes: int = G1_NUM_SCORER_CLASSES,
) -> torch.Tensor:
    """Compute per-pair dominant SegNet class indices.

    For each pair p in [0, n_pairs), decode GT frame_0 from the contest
    video, run SegNet to obtain per-pixel argmax, and reduce to the
    dominant class. Returns a (n_pairs,) long tensor in
    [0, num_classes).

    Compress-side scorer use is FREE per CLAUDE.md 'Strict scorer rule'
    rule #2 (encoder may use the scorer; decoder may NOT load it).
    """
    import av
    import torch.nn.functional as F
    from tac.scorer import load_differentiable_scorers
    from tac.differentiable_eval_roundtrip import patch_upstream_yuv6_globally

    # Patch upstream YUV6 BEFORE loading scorers (Catalog #187).
    yuv6_token = patch_upstream_yuv6_globally(upstream_dir=upstream_dir)
    try:
        _, segnet = load_differentiable_scorers(upstream_dir, device=device)
        for p in segnet.parameters():
            p.requires_grad_(False)
        segnet.eval()

        container = av.open(str(video_path))
        gt_frames: list[torch.Tensor] = []
        # Decode 2 * n_pairs frames; we use frame_0 of each pair (even
        # indices) for SegNet class derivation.
        n_target_frames = 2 * n_pairs
        for idx, frame in enumerate(container.decode(container.streams.video[0])):
            if idx >= n_target_frames:
                break
            rgb = frame.to_ndarray(format="rgb24")
            # Convert to (3, H, W) float [0..255].
            t = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float()
            gt_frames.append(t)
        container.close()
        if len(gt_frames) < n_target_frames:
            raise RuntimeError(
                f"video_path={video_path} produced only {len(gt_frames)} "
                f"frames; expected {n_target_frames}"
            )

        # Take frame_0 of each pair (even index 0, 2, 4, ...) as the
        # SegNet input frame.
        frame_0_list = [gt_frames[2 * i] for i in range(n_pairs)]
        # SegNet preprocess_input: slice last frame from 5D (B,T,C,H,W)
        # OR accept 4D (B,C,H,W) per upstream/modules.py:
        #   x = x[:, -1, ...] if x.ndim == 5 else x
        #   x = F.interpolate(x, size=(384, 512), mode='bilinear')
        # Build (B, C, H, W) batch.
        with torch.inference_mode():
            class_indices_per_pair: list[int] = []
            batch_size = 8
            for batch_start in range(0, n_pairs, batch_size):
                batch_end = min(batch_start + batch_size, n_pairs)
                batch = torch.stack(
                    frame_0_list[batch_start:batch_end], dim=0
                ).to(device)
                # SegNet expects (B, C, H, W) per upstream contract.
                # `preprocess_input` will be called inside forward via
                # the canonical helper — but here we call segnet directly
                # because we need argmax, not the score formula. Use the
                # exposed preprocess pipeline manually.
                if batch.ndim == 4:
                    seg_input = F.interpolate(
                        batch, size=(384, 512), mode="bilinear",
                        align_corners=False,
                    )
                else:
                    raise ValueError(f"unexpected batch shape {batch.shape}")
                # SegNet forward returns (B, 5, H_out, W_out) logits.
                seg_logits = segnet(seg_input)
                # argmax along class dim.
                seg_argmax = seg_logits.argmax(dim=1)  # (B, H_out, W_out)
                classes = g1_per_pair_dominant_class_from_segnet_argmax(
                    seg_argmax, num_classes=num_classes
                )
                class_indices_per_pair.extend(classes.cpu().tolist())
        return torch.tensor(
            class_indices_per_pair, dtype=torch.long, device=device
        )
    finally:
        from tac.differentiable_eval_roundtrip import unpatch_upstream_yuv6
        unpatch_upstream_yuv6(yuv6_token)


def _smoke_main(args: argparse.Namespace) -> int:
    """Smoke entry: synthetic A1 latents + UNIFORM-class indices, 3 epochs.

    Per Catalog #114, synthetic data is permitted ONLY inside _smoke_main.
    Class indices are uniformly assigned per-pair (no SegNet load) so the
    smoke validates the gating head + AC coder + archive grammar without
    pulling in scorer weights.
    """
    _canon_pin_seeds(args.seed)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Smoke synthetic A1 latents (deterministic random); uniform class
    # indices across 5 classes.
    torch.manual_seed(args.seed)
    n_smoke_pairs = 32
    a1_latents = torch.zeros(A1_N_PAIRS, A1_LATENT_DIM, device=args.device)
    a1_latents[:n_smoke_pairs] = torch.randn(
        n_smoke_pairs, A1_LATENT_DIM, device=args.device
    )
    _, smoke_offset_cpu, smoke_scale_cpu = _z3v2_affine_residual_int8(a1_latents)
    smoke_offset = smoke_offset_cpu.to(args.device)
    smoke_scale = smoke_scale_cpu.to(args.device)
    # Synthetic class indices: cycle through 5 classes deterministically.
    class_indices = torch.arange(
        A1_N_PAIRS, device=args.device, dtype=torch.long
    ) % args.num_scorer_classes

    cfg = Z3G1Config(
        num_scorer_classes=args.num_scorer_classes,
        quantization_step=args.quantization_step,
        factorized_half_range=args.factorized_half_range,
        init_sigma=args.init_sigma,
    )
    gating_head = Z3G1ScorerClassGatingHead(cfg).to(args.device)
    optimizer = torch.optim.AdamW(
        gating_head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    losses: list[float] = []
    rate_bits: list[float] = []
    n_epochs = max(args.epochs, 3)
    for _epoch in range(n_epochs):
        optimizer.zero_grad()
        out = z3_g1_lagrangian(
            gating_head=gating_head,
            a1_latents=a1_latents,
            class_indices=class_indices,
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
        torch.nn.utils.clip_grad_norm_(gating_head.parameters(), args.grad_clip)
        optimizer.step()
        losses.append(float(out["total_loss"].item()))
        rate_bits.append(float(out["rate_bits_total"].item()))

    # Build the Z3 composition archive over real A1 base.
    base_bytes, _, _ = _load_a1_archive_bytes(args.a1_archive_path)
    base_sha = hashlib.sha256(base_bytes).hexdigest()

    # G1 reuses Z3 v2 archive grammar; ship direct-residual form (no
    # entropy-coded sigma table or class indices in the smoke packet).
    # The smoke is a TRAINING ARTIFACT, not a score claim.
    from tac.substrates.z3_balle_hyperprior_bolton.archive_v2 import (
        build_z3v2_composition_archive_contract as _v2_build_contract,
    )
    from tac.substrates.z3_balle_hyperprior_bolton.archive_v2 import (
        build_z3v2_payload_bytes as _v2_build_payload,
    )
    from tac.substrates.z3_balle_hyperprior_bolton.archive_v2 import (
        encode_z3hv2_section as _v2_encode_section,
    )

    a1_full = torch.zeros(A1_N_PAIRS, A1_LATENT_DIM, device=args.device)
    a1_full[: a1_latents.shape[0]] = a1_latents
    residual_v2, a1_offset, a1_scale = _z3v2_affine_residual_int8(a1_full)
    z3hv2_section = _v2_encode_section(
        hyperprior_weights_int8=b"",
        w_hat_int8=b"",
        residual_int8=residual_v2.numpy().tobytes(),
        latent_offset=a1_offset,
        latent_scale=a1_scale,
        hyper_dim=8,  # G1 reuses the v2 grammar's hyper_dim slot (placeholder)
        int8_w_scale=1.0,
        quant_step=cfg.quantization_step,
        min_sigma=cfg.min_sigma,
        max_sigma=cfg.max_sigma,
        factorized_half_range=cfg.factorized_half_range,
    )
    archive_bytes = _v2_build_payload(
        a1_bytes=base_bytes, z3hv2_section=z3hv2_section
    )
    archive_contract = _v2_build_contract(base_bytes, archive_bytes)
    sidecar_bytes = len(z3hv2_section)
    (out_dir / "z3g1_section_diagnostic.bin").write_bytes(z3hv2_section)
    archive_path = out_dir / "0.bin"
    archive_path.write_bytes(archive_bytes)

    # Emit runtime tree + archive.zip per the canonical pattern.
    submission_dir = out_dir / "submission_dir"
    _write_runtime(
        submission_dir, a1_runtime_src=args.a1_archive_path.parent / "src"
    )
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
        "num_scorer_classes": cfg.num_scorer_classes,
        "param_count": sum(p.numel() for p in gating_head.parameters()),
        "sidecar_bytes": sidecar_bytes,
        "archive_contract": archive_contract.as_manifest(),
        "byte_saving": archive_contract.byte_saving,
        "estimated_sidecar_overhead": estimate_z3g1_section_overhead_bytes(
            gating_head=gating_head,
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
            "smoke_uses_synthetic_a1_latents",
            "smoke_uses_uniform_class_indices_not_segnet_derived",
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
        f"[z3g1-smoke] OK final_loss={final_loss:.6f} "
        f"rate_bits={final_rate_bits:.1f} archive={len(archive_bytes)}B "
        f"sha={archive_sha[:12]}... layout={stats['archive_contract']['layout']} "
        f"param_count={stats['param_count']}"
    )
    return 0


def _full_main(args: argparse.Namespace) -> int:
    """Full training path: real A1 latents + SegNet-derived class indices.

    Pipeline:
      1. Load FROZEN A1 archive bytes (sha pinned via --a1-archive-path).
      2. Decode A1 latents (600 x 28) via A1's codec.
      3. Compute per-pair SegNet dominant class indices on GT frames
         (compress-side scorer use is FREE per CLAUDE.md rule #2).
      4. Init Z3-G1 gating head (140 params; canonical EMA decay=0.997).
      5. Train AdamW(lr=1e-3) with class-conditional Ballé R Lagrangian.
      6. Apply EMA shadow at eval; snapshot+restore live weights (Catalog #88).
      7. Emit Z3HV2 grammar (G1 variant in direct-residual mode for production
         safety) + archive composition.
      8. Run paired CUDA auth eval via canonical gate_auth_eval_call.
      9. Post stats with fail-closed result_review_blockers (Catalog #127).
    """
    from tac.training import EMA

    _canon_pin_seeds(args.seed)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _canon_device_or_die(
        args.device,
        smoke=bool(args.full_cpu) or bool(getattr(args, "smoke", False)),
        substrate_tag=SUBSTRATE_TAG,
    )

    stage_log: list[dict[str, str]] = []

    def _stage(name: str) -> None:
        msg = {"stage": name, "at": _canon_utc_now_iso()}
        stage_log.append(msg)
        print(f"[z3g1-full] {name} @ {msg['at']}")

    _stage("seed_pinned")

    # ---- Step 1: load A1 (FROZEN base) ----
    a1_bytes, a1_sha, a1_size = _load_a1_archive_bytes(args.a1_archive_path)
    _stage(f"a1_loaded_{a1_size}_B_sha{a1_sha[:8]}")

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

    # ---- Step 2: decode A1 latents ----
    decoder, a1_latents = _decode_a1_latents_and_decoder(
        args.a1_archive_path, a1_bytes, device=device
    )
    _stage(f"a1_decoded_latents{tuple(a1_latents.shape)}")
    _, v2_offset_cpu, v2_scale_cpu = _z3v2_affine_residual_int8(a1_latents)
    v2_latent_offset = v2_offset_cpu.to(device)
    v2_latent_scale = v2_scale_cpu.to(device)
    _stage("z3v2_affine_offset_scale_built")
    del decoder  # rate-only training; exact score is eval-gated.

    # ---- Step 3: compute per-pair SegNet class indices ----
    n_pairs = int(a1_latents.shape[0])
    _stage("computing_segnet_per_pair_class_indices")
    try:
        class_indices = _compute_per_pair_class_indices_from_video(
            args.video_path,
            args.upstream_dir,
            n_pairs=n_pairs,
            device=device,
            num_classes=args.num_scorer_classes,
        )
        _stage(
            f"class_indices_computed_n{class_indices.numel()}_"
            f"unique{int(class_indices.unique().numel())}"
        )
    except Exception as exc:
        # Codex bkrbqet3p F2 / Catalog #267 fail-closed gate. The previous
        # silent fallback to uniform class assignment converted scorer-load
        # / video-decode / dependency failures into a synthetic class prior
        # that flowed through archive build + auth-eval as if real. The new
        # contract: full runs MUST fail-closed unless the operator opts in
        # via --allow-uniform-class-fallback (which is itself paired with
        # --research-only-direct-residual per _validate_g1_research_only_flags).
        if not args.allow_uniform_class_fallback:
            raise SystemExit(
                "ERROR: SegNet per-pair class derivation failed: "
                f"{exc!r}. Per Codex review bkrbqet3p F2 / Catalog #267, "
                "the trainer is fail-closed. Either (a) fix the upstream "
                "scorer / video / device error, OR (b) re-run with "
                "--allow-uniform-class-fallback (only valid when paired "
                "with --research-only-direct-residual; produces a "
                "research-only synthetic-class-prior artifact tagged "
                "research_only=true and excluded from contest promotion)."
            ) from exc
        # Operator-opt-in path: emit the uniform fallback BUT loudly stamp
        # the artifact as research-only. The downstream archive build keeps
        # research_only=true (already set elsewhere) and skips auth-eval
        # via _g1_full_should_skip_auth_eval below.
        print(
            "[z3g1-full] OPERATOR-OPT-IN uniform class fallback "
            f"(--allow-uniform-class-fallback): SegNet class derivation "
            f"failed: {exc!r}. Class assignment is SYNTHETIC + non-promotable; "
            "research_only=true; auth-eval will be skipped per Catalog #267."
        )
        class_indices = (
            torch.arange(n_pairs, device=device, dtype=torch.long)
            % args.num_scorer_classes
        )
        # Mark on args so downstream archive + auth-eval gate can see it.
        args._g1_uniform_class_fallback_active = True  # type: ignore[attr-defined]
        _stage("class_indices_fallback_uniform_research_only")

    # Diagnostic: per-class histogram.
    histogram = torch.bincount(
        class_indices, minlength=args.num_scorer_classes
    )
    _stage(f"class_histogram={histogram.cpu().tolist()}")

    # ---- Step 4: init gating head + EMA + optimizer ----
    cfg = Z3G1Config(
        num_scorer_classes=args.num_scorer_classes,
        quantization_step=args.quantization_step,
        factorized_half_range=args.factorized_half_range,
        init_sigma=args.init_sigma,
    )
    gating_head = Z3G1ScorerClassGatingHead(cfg).to(device)
    n_g1_params = sum(p.numel() for p in gating_head.parameters())
    ema = EMA(gating_head, decay=args.ema_decay)
    optimizer = torch.optim.AdamW(
        gating_head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs)
    )
    _stage(f"gating_head_built_{n_g1_params}_params")

    # ---- Step 5: train/val split ----
    val_count = max(1, n_pairs // 8)
    val_idx_start = n_pairs - val_count
    train_indices_pool = list(range(val_idx_start))
    val_indices_pool = list(range(val_idx_start, n_pairs))

    ckpt_best_path = out_dir / "best.pt"
    train_started_at = time.time()
    nan_strike = 0
    max_nan_strikes = 3

    try:
        for epoch in range(args.epochs):
            gating_head.train()
            random.shuffle(train_indices_pool)
            epoch_losses: list[float] = []
            epoch_rate_bits: list[float] = []
            for batch_start in range(0, len(train_indices_pool), args.batch_size):
                batch_indices = train_indices_pool[
                    batch_start : batch_start + args.batch_size
                ]
                if not batch_indices:
                    continue
                pair_idxs = torch.tensor(
                    batch_indices, device=device, dtype=torch.long
                )
                batch_latents = a1_latents[pair_idxs]
                batch_classes = class_indices[pair_idxs]
                out = z3_g1_lagrangian(
                    gating_head=gating_head,
                    a1_latents=batch_latents,
                    class_indices=batch_classes,
                    latent_offset=v2_latent_offset,
                    latent_scale=v2_latent_scale,
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
                loss = out["total_loss"]
                if not torch.isfinite(loss):
                    nan_strike += 1
                    print(
                        f"[z3g1-full] NaN strike {nan_strike}/{max_nan_strikes} "
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
                    gating_head.parameters(), args.grad_clip
                )
                optimizer.step()
                ema.update(gating_head)
                epoch_losses.append(float(loss.detach().cpu()))
                epoch_rate_bits.append(
                    float(out["rate_bits_total"].detach().cpu())
                )
            scheduler.step()

            # Eval every ~epochs/30 OR at last epoch.
            if epoch % max(1, args.epochs // 30) == 0 or epoch == args.epochs - 1:
                live_state = {
                    k: v.detach().clone()
                    for k, v in gating_head.state_dict().items()
                }
                ema.apply(gating_head)
                try:
                    gating_head.eval()
                    with torch.inference_mode():
                        val_pair_idxs = torch.tensor(
                            val_indices_pool, device=device, dtype=torch.long
                        )
                        val_latents = a1_latents[val_pair_idxs]
                        val_classes = class_indices[val_pair_idxs]
                        val_out = z3_g1_lagrangian(
                            gating_head=gating_head,
                            a1_latents=val_latents,
                            class_indices=val_classes,
                            latent_offset=v2_latent_offset,
                            latent_scale=v2_latent_scale,
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
                    val_loss = float(val_out["total_loss"].item())
                finally:
                    gating_head.load_state_dict(live_state)
                    gating_head.train()
                avg_train = (
                    sum(epoch_losses) / len(epoch_losses)
                    if epoch_losses else math.nan
                )
                avg_rate = (
                    sum(epoch_rate_bits) / len(epoch_rate_bits)
                    if epoch_rate_bits else math.nan
                )
                print(
                    f"[z3g1-full] epoch {epoch:4d}: train_loss={avg_train:.5f} "
                    f"rate_bits={avg_rate:.1f} val_loss={val_loss:.5f} "
                    f"(best={best_val_loss:.5f})"
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    live_state2 = {
                        k: v.detach().clone()
                        for k, v in gating_head.state_dict().items()
                    }
                    ema.apply(gating_head)
                    ema_state = {
                        k: v.detach().cpu()
                        for k, v in gating_head.state_dict().items()
                    }
                    gating_head.load_state_dict(live_state2)
                    torch.save(
                        {
                            "state_dict": ema_state,
                            "config": asdict(cfg),
                            "epoch": epoch,
                            "val_loss": val_loss,
                        },
                        ckpt_best_path,
                    )

        train_elapsed = time.time() - train_started_at
        _stage(
            f"trained_best_epoch_{best_epoch}_val_loss_{best_val_loss:.5f}_"
            f"elapsed_{train_elapsed:.1f}s"
        )

        # ---- Step 6: load EMA-best checkpoint ----
        if ckpt_best_path.is_file():
            best_ckpt = torch.load(
                ckpt_best_path, weights_only=False, map_location=device
            )  # WEIGHTS_ONLY_FALSE_OK:trusted-local-checkpoint
            gating_head.load_state_dict(best_ckpt["state_dict"])
        gating_head.eval()

        # ---- Step 7: build composition archive ----
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

        residual_v2, a1_offset, a1_scale = _z3v2_affine_residual_int8(a1_latents)
        residual_int8 = residual_v2.numpy().tobytes()
        # Direct-residual mode for production safety: ship residual + affine
        # only (matches Z3 v2 production packet). The G1 sigma table + class
        # indices are TRAINING-TIME diagnostics until an entropy-coded
        # decoder consumes them at inflate. The byte savings predicted by
        # Wunderkind G1 require the AC-coded residual variant; this
        # production-safe form is the SMOKE-and-AUTH-EVAL surface.
        z3hv2_section = _v2_encode_section(
            hyperprior_weights_int8=b"",
            w_hat_int8=b"",
            residual_int8=residual_int8,
            latent_offset=a1_offset,
            latent_scale=a1_scale,
            hyper_dim=8,  # placeholder; G1 reuses v2 grammar
            int8_w_scale=1.0,
            quant_step=cfg.quantization_step,
            min_sigma=cfg.min_sigma,
            max_sigma=cfg.max_sigma,
            factorized_half_range=cfg.factorized_half_range,
        )
        sidecar_bytes_estimate = len(z3hv2_section)
        bytes_saved_estimate = _V2_A1_LATENT_BLOB_LEN - sidecar_bytes_estimate

        composition_bytes = _v2_build_payload(
            a1_bytes=a1_bytes, z3hv2_section=z3hv2_section
        )
        composition_contract = _v2_build_contract(a1_bytes, composition_bytes)
        sidecar_shipped = True
        composition_contract_manifest = composition_contract.as_manifest()
        composition_byte_saving = composition_contract.byte_saving
        (out_dir / "z3g1_section_diagnostic.bin").write_bytes(z3hv2_section)
        composition_sha = hashlib.sha256(composition_bytes).hexdigest()
        composition_size = len(composition_bytes)
        _stage(
            f"composition_built_{composition_size}_B_sha{composition_sha[:8]}_"
            f"section_bytes={sidecar_bytes_estimate}_saved={bytes_saved_estimate}"
        )

        # Save the learned G1 sigma table (diagnostic only, not in archive).
        sigma_int8, sigma_scale = gating_head.quantize_sigma_table_int8()
        torch.save(
            {
                "sigma_table_int8": sigma_int8.cpu(),
                "sigma_scale": sigma_scale,
                "class_indices": class_indices.cpu(),
                "class_histogram": histogram.cpu(),
                "config": asdict(cfg),
            },
            out_dir / "g1_diagnostic.pt",
        )

        # ---- Step 8: build archive.zip + runtime tree ----
        _build_archive_zip(archive_zip_path, bin_bytes=composition_bytes)
        submission_dir.mkdir(parents=True, exist_ok=True)
        _write_runtime(
            submission_dir, a1_runtime_src=args.a1_archive_path.parent / "src"
        )
        runtime_manifest = _write_submission_runtime_manifest(submission_dir)
        (submission_dir / "0.bin").write_bytes(composition_bytes)
        shutil.copy2(archive_zip_path, submission_dir / "archive.zip")
        _stage("archive_zip_and_runtime_emitted")

        # ---- Step 9: CUDA auth eval via canonical gate ----
        # Codex bkrbqet3p F1 (Catalog #266) + F2 (Catalog #267) auth-eval
        # gate. Skip when (a) operator passed --skip-auth-eval; (b) the
        # G1 archive is in direct-residual mode (always today — bytes are
        # not consumed by inflate); (c) the uniform-class fallback fired
        # (synthetic class prior cannot represent the substrate). The
        # direct-residual branch is hard-wired ON for the current
        # _v2_encode_section(hyperprior_weights_int8=b"", w_hat_int8=b"")
        # build above, so we ALWAYS skip auth-eval until a real consumer
        # lands. The skip stamps a research-only blocker into the
        # downstream review-blockers list.
        _g1_archive_consumes_hyperprior_bytes = False  # See line ~875 above
        _g1_uniform_fallback_active = bool(
            getattr(args, "_g1_uniform_class_fallback_active", False)
        )
        _g1_skip_auth_eval_reason: str | None = None
        if args.skip_auth_eval:
            _g1_skip_auth_eval_reason = "operator_set_--skip-auth-eval"
        elif not _g1_archive_consumes_hyperprior_bytes:
            _g1_skip_auth_eval_reason = (
                "catalog_266_g1_archive_direct_residual_no_consumer_for_"
                "hyperprior_or_class_bytes"
            )
        elif _g1_uniform_fallback_active:
            _g1_skip_auth_eval_reason = (
                "catalog_267_uniform_class_prior_synthetic_non_promotable"
            )
        if _g1_skip_auth_eval_reason is None:
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
        else:
            _stage(f"contest_auth_eval_SKIPPED_REASON_{_g1_skip_auth_eval_reason}")
            # Stamp the skip reason onto args so downstream blocker logic
            # surfaces it in result_review_blockers.
            existing = str(getattr(args, "auth_eval_skipped_reason", "") or "")
            sep = ";" if existing else ""
            args.auth_eval_skipped_reason = (  # type: ignore[attr-defined]
                f"{existing}{sep}{_g1_skip_auth_eval_reason}"
            )
    finally:
        pass

    # ---- Step 10: posterior update via canonical locked helper ----
    if not args.skip_auth_eval and (out_dir / "contest_auth_eval_cuda.json").exists():
        try:
            from tac.continual_learning import (
                posterior_update_locked_from_auth_eval_json,
            )
            update = posterior_update_locked_from_auth_eval_json(
                out_dir / "contest_auth_eval_cuda.json"
            )
            print(
                f"[z3g1-full] posterior_update accepted="
                f"{getattr(update, 'accepted', '?')}"
            )
            _stage("posterior_updated")
        except Exception as exc:
            print(f"[z3g1-full] WARN posterior_update failed: {exc!r}")

    if auth_eval_score_claim_valid:
        try:
            from tac.cost_band_calibration import CostBandAnchor, append_anchor
            wall_sec = float(time.time() - train_started_at)
            cost_usd = wall_sec / 3600.0 * 0.60
            anchor = CostBandAnchor(
                logged_at_utc=_canon_utc_now_iso(),
                dispatch_label=f"{SUBSTRATE_LANE_ID}_{_canon_utc_now_iso()}",
                trainer="train_substrate_z3_g1_scorer_softmax_hyperprior_gating",
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
                    f"literature_anchor=wunderkind_g1_scorer_class_gating;"
                    f"lane_class=substrate_engineering"
                ),
            )
            append_anchor(anchor)
            _stage("cost_band_anchor_appended")
        except Exception as exc:
            print(f"[z3g1-full] WARN cost_band_anchor_append failed: {exc!r}")
    else:
        _stage("cost_band_anchor_skipped_no_valid_auth_eval")

    result_review_blockers = list(
        composition_contract_manifest.get("result_review_blockers", [])
    )
    result_review_blockers.append(
        "trainer_stats_not_authoritative_score_claim_surface"
    )
    if not auth_eval_score_claim_valid:
        skipped_reason = str(getattr(args, "auth_eval_skipped_reason", "") or "")
        if skipped_reason:
            result_review_blockers.append(skipped_reason)
        else:
            result_review_blockers.append("contest_cuda_auth_eval_not_validated")
    result_review_blockers.append(
        "g1_production_packet_is_direct_residual_mode_per_z3_v2_grammar; "
        "g1_sigma_table_and_class_indices_remain_diagnostic_until_entropy_"
        "coded_decoder_lands"
    )
    result_review_blockers = list(dict.fromkeys(result_review_blockers))

    hardware_substrate = _canon_detect_hardware_substrate(
        substrate_tag=SUBSTRATE_TAG,
        axis="cuda" if device.type == "cuda" else "cpu",
        env_var_candidates=("Z3_G1_GPU", "MODAL_GPU"),
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
        "num_scorer_classes": cfg.num_scorer_classes,
        "predicted_delta_s_band": [-0.015, -0.005],
        "predicted_delta_s_strict_honest": -0.005,
        "predicted_delta_s_uncertainty": 0.50,
        "council_verdict": (
            "PROCEED Phase B.1 evidence-review council binding verdict"
            " 2026-05-15 (operator approved $110-170 Tier 2 envelope)"
        ),
        "literature_anchor": (
            "Wunderkind G1 SUBSTITUTION-1:1 spec; sister of Ballé 2018 "
            "scale-hyperprior with class-table replacing per-pair MLP"
        ),
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
                auth_eval_score_claim_valid
                and auth_eval_score_axis == "contest_cuda"
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
        f"[z3g1-full] DONE archive={composition_size}B sha={composition_sha[:12]}... "
        f"auth_score={auth_eval_score} grade={auth_eval_evidence_grade} "
        f"sidecar_shipped={sidecar_shipped}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _validate_full_cpu_flags(args)
    if args.device == "mps":
        raise SystemExit(
            "ERROR: --device mps refused per CLAUDE.md MPS-NOISE non-negotiable"
        )
    if args.smoke:
        return _smoke_main(args)
    # Codex bkrbqet3p F1 (Catalog #266) + F2 (Catalog #267) full-main
    # gating: refuse to spend GPU until the operator opts into the
    # research-only direct-residual contract.
    _validate_g1_research_only_flags(args)
    return _full_main(args)


if __name__ == "__main__":
    sys.exit(main())
