# SPDX-License-Identifier: MIT
"""Train the D4 Wyner-Ziv frame-0 substrate end-to-end (deep-math M7).

Per `.omx/research/deep_math_geometry_manifolds_synthesis_20260514.md` §6 D4
and §3.5 the strongest single-substrate bet for sub-0.188 gate clearance.
Predicted ΔS -0.025 to -0.045 vs PR101 0.193 baseline
``[mathematical-derivation; first-principles-bound]``.

Council-binding contract (CLAUDE.md non-negotiables) honored end-to-end:

- Train against ``upstream/videos/0.mkv`` decoded via pyav (synthetic data
  FORBIDDEN outside ``--smoke`` per Catalog #114).
- Patch upstream ``rgb_to_yuv6`` via ``patch_upstream_yuv6_globally`` BEFORE
  scorer construction (Catalog #187; PR #95/#106 contract).
- ``load_differentiable_scorers`` for SegNet/PoseNet (no scorer load at
  inflate; only at training).
- ``apply_eval_roundtrip_during_training`` inside the per-batch loop
  (Catalog #5).
- ``tac.training.EMA(decay=0.997)`` update after every ``optimizer.step``;
  inference checkpoint = EMA shadow (Catalog #88).
- Score-domain Lagrangian with Wyner-Ziv frame-0 residual term per
  HNeRV parity discipline lesson L6.
- AdamW lr cosine annealing; gradient clip 1.0; NaN watchdog (Council D).
- End with CUDA auth eval on best EMA checkpoint (CLAUDE.md "Auth eval
  EVERYWHERE"); refuse MPS (Catalog #1); CPU permitted only with ``--smoke``.
- Continual-learning posterior update via ``posterior_update_locked``
  (Catalog #128 atomic fcntl).
- Contest-compliant runtime emission with 3 positional args inflate.sh +
  ``set -euo pipefail`` per Catalog #146 + #163.
- TIER_1_OPERATOR_REQUIRED_FLAGS declared per Catalog #151 (annotated
  assignment for Catalog #168 AST walker).
- ``--full-cpu`` opt-in coupled with ``--advisory-cpu-explicitly-waived``
  per Catalog #197.
- Smoke=True entry path per the PR95++ Path B precedent.

V1 SCOPE (this landing):
- ``_smoke_main`` builds a tiny config, trains for ≤3 epochs on synthetic
  data, runs the archive pack + parse + inflate roundtrip, and emits a
  contest-compliant runtime tree. NO scorer load required.
- ``_full_main`` trains the D4 substrate against real ``upstream/videos/0.mkv``
  pairs decoded via pyav, runs the score-aware Lagrangian end-to-end with
  patched YUV6 + differentiable scorers + EMA(0.997), packs the WZF01
  archive against the smoke base provider (V1 self-contained custody), emits
  the contest-compliant runtime tree, runs CUDA auth eval on the best EMA
  checkpoint, and posts the result to the continual-learning posterior.
  UNLOCKED 2026-05-14 per operator approval ("approved, proceed with all")
  after the 5-round adversarial council unanimous SEAL documented in
  ``feedback_d4_wyner_ziv_frame_0_landed_20260514.md``.

Usage (smoke; macOS CPU, tiny config, ~3 epochs)::

    .venv/bin/python experiments/train_substrate_d4_wyner_ziv_frame_0.py \\
        --video-path upstream/videos/0.mkv \\
        --output-dir experiments/results/d4_smoke_<utc> \\
        --epochs 3 --device cpu --smoke

Usage (full; CUDA-required; PENDING Phase 2 council approval)::

    .venv/bin/python experiments/train_substrate_d4_wyner_ziv_frame_0.py \\
        --video-path upstream/videos/0.mkv \\
        --output-dir experiments/results/d4_<utc> \\
        --epochs 2000 --batch-size 1 --lr 5e-4 --device cuda
"""
# AUTOCAST_FP16_WAIVED:score-aware-scorer-path-pending-canonical-autocast-backport
# TORCH_COMPILE_WAIVED:defer-until-per-substrate-canary-validates-Inductor-graph-breaks
# TF32_WAIVED:opt-in-via-CLI-flag-to-keep-eval-roundtrip-numerics-deterministic
# NO_GRAD_WAIVED:eval-time-scorer-forward-wrapped-in-torch.inference_mode-inside-_run_val_loop
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import sys
import tempfile
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
    build_optimized_training_context as _build_optimized_training_context,
)
from tac.substrates._shared.trainer_skeleton import (
    decode_real_pairs as _canon_decode_real_pairs,
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
    require_contest_cuda_auth_eval_claim as _canon_require_contest_cuda_auth_eval_claim,
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
from tac.substrates.d4_wyner_ziv_frame_0 import (
    MotionModelMode,
    WynerZivFrame0Config,
    WynerZivFrame0Substrate,
    encode_residual_blob,
    pack_archive,
)

# ---------------------------------------------------------------------------
# Module paths + constants
# ---------------------------------------------------------------------------

DEFAULT_VIDEO_PATH = REPO_ROOT / "upstream" / "videos" / "0.mkv"
DEFAULT_UPSTREAM_DIR = REPO_ROOT / "upstream"
CONTEST_AUTH_EVAL_SCRIPT = REPO_ROOT / "experiments" / "contest_auth_eval.py"

EVAL_HW = (384, 512)
CAMERA_HW = (874, 1164)
N_PAIRS_FULL = 600
CONTEST_NORMALIZER = 37_545_489.0

SUBSTRATE_TAG = "d4_wyner_ziv_frame_0"
SUBSTRATE_LANE_ID = "lane_d4_wyner_ziv_frame_0_substrate_20260514"


def _path_is_under(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _resolve_auth_eval_json_paths(
    output_dir: Path,
    *,
    durable_root: Path | None = None,
) -> tuple[Path, Path]:
    """Return ``(gate_json, local_copy_json)`` for score-grade auth eval.

    Modal trainers run from a writable ``/tmp/pact`` copy. The canonical
    ``contest_auth_eval.py`` scorer refuses score-grade evidence paths under
    temp storage, so the gate writes to a non-temp path and then the trainer
    copies that JSON back into ``output_dir`` for artifact harvest.
    """

    local_copy_json = output_dir / "contest_auth_eval.json"
    temp_root = Path(tempfile.gettempdir())
    if not _path_is_under(local_copy_json, temp_root):
        return local_copy_json, local_copy_json
    root = durable_root
    if root is None:
        root = Path(
            os.environ.get(
                "D4_WYNER_ZIV_FRAME_0_AUTH_EVAL_ROOT",
                "/root/d4_wyner_ziv_frame_0_auth_eval",
            )
        )
    return root / output_dir.name / "contest_auth_eval.json", local_copy_json


# ---------------------------------------------------------------------------
# Catalog #151 manifest — every flag below must be threaded by any operator
# wrapper that subprocess-invokes this trainer. Annotated as ast.AnnAssign so
# Catalog #168's AST walker observes it (NOT bare Assign).
# ---------------------------------------------------------------------------
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--video-path": {
        "env": "D4_WYNER_ZIV_FRAME_0_VIDEO_PATH",
        "rationale": (
            "score-aware substrate MUST train against the contest video "
            "(upstream/videos/0.mkv); synthetic data is FORBIDDEN outside --smoke"
        ),
        "default": str(DEFAULT_VIDEO_PATH.relative_to(REPO_ROOT)),
        "required_input_file": True,
        "generator_command": (
            "contest-pinned upstream snapshot — never regenerated locally"
        ),
        "rationale_audit": (
            ".omx/research/deep_math_geometry_manifolds_synthesis_20260514.md §6 D4"
        ),
    },
    "--output-dir": {
        "env": "D4_WYNER_ZIV_FRAME_0_OUTPUT_DIR",
        "rationale": "custody location for checkpoints + archive + provenance",
    },
    "--epochs": {
        "env": "D4_WYNER_ZIV_FRAME_0_EPOCHS",
        "rationale": (
            "D4 substrate is small (motion params + per-pair residual); council "
            "default 2000 epochs for full training run"
        ),
        "default": "2000",
    },
    "--upstream-dir": {
        "env": "D4_WYNER_ZIV_FRAME_0_UPSTREAM_DIR",
        "rationale": (
            "upstream/ root for scorer weights + evaluate.py; required for "
            "full training and auth eval"
        ),
        "default": str(DEFAULT_UPSTREAM_DIR.relative_to(REPO_ROOT)),
    },
    "--device": {
        "env": "D4_WYNER_ZIV_FRAME_0_DEVICE",
        "rationale": (
            "compute device; cuda required for full training (MPS refused "
            "per CLAUDE.md MPS-NOISE rule); cpu permitted only with --smoke "
            "or --full-cpu --advisory-cpu-explicitly-waived"
        ),
        "default": "cuda",
    },
    "--motion-mode": {
        "env": "D4_WYNER_ZIV_FRAME_0_MOTION_MODE",
        "rationale": (
            "probe-disambiguator (Catalog #125 hook #6): pick se3_parametric "
            "OR optical_flow; ship both modes; council picks default per "
            "the probe verdict at first contest-CUDA anchor"
        ),
        "default": "se3_parametric",
    },
    "--residual-coarse-h": {
        "env": "D4_WYNER_ZIV_FRAME_0_RESIDUAL_COARSE_H",
        "rationale": (
            "Photometric residual coarse height (default 48 = 1/8 scorer "
            "height; balances reconstruction fidelity vs archive byte budget)"
        ),
        "default": "48",
    },
    "--residual-coarse-w": {
        "env": "D4_WYNER_ZIV_FRAME_0_RESIDUAL_COARSE_W",
        "rationale": (
            "Photometric residual coarse width (default 64 = 1/8 scorer width)"
        ),
        "default": "64",
    },
    "--base-archive-path": {
        "env": "D4_WYNER_ZIV_FRAME_0_BASE_ARCHIVE_PATH",
        "rationale": (
            "Path to the BASE substrate archive whose frame_1 reconstruction "
            "D4 composes with. Defaults to none (smoke uses a synthetic base "
            "for testing). Full training composes with A1/PR101/HDM8."
        ),
        "default": "",
    },
    "--max-pairs": {
        "env": "D4_WYNER_ZIV_FRAME_0_MAX_PAIRS",
        "rationale": (
            "Cap the n_pairs decoded from the contest video; default empty "
            "means use N_PAIRS_FULL=600. T4 (16GB) OOMs on 600-pair "
            "retain_graph=True full forward at 384x512; smoke-on-T4 sets "
            "150. A10G/A100 with 24-40GB can run the full 600."
        ),
        "default": "",
    },
    "--enable-gt-scorer-cache": {
        "env": "D4_WYNER_ZIV_FRAME_0_ENABLE_GT_SCORER_CACHE",
        "rationale": (
            "F3/GTScorerCache; caches frozen GT PoseNet+SegNet targets once "
            "and reuses indexed batches in the score-aware hot loop"
        ),
        "default": "true",
    },
}


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_substrate_d4_wyner_ziv_frame_0",
        description=(
            "Train D4 Wyner-Ziv frame-0 substrate (deep-math M7). "
            "Per-pair SE(3) motion (or optical flow) + photometric residual "
            "to derive frame_0 from a BASE substrate's frame_1."
        ),
    )
    p.add_argument("--video-path", type=Path, default=DEFAULT_VIDEO_PATH)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR
    )
    p.add_argument(
        "--motion-mode",
        type=str,
        default="se3_parametric",
        choices=["se3_parametric", "optical_flow"],
    )
    p.add_argument("--flow-grid-h", type=int, default=12)
    p.add_argument("--flow-grid-w", type=int, default=16)
    p.add_argument("--residual-coarse-h", type=int, default=48)
    p.add_argument("--residual-coarse-w", type=int, default=64)
    p.add_argument(
        "--base-archive-path",
        type=str,
        default="",
        help="Path to BASE substrate archive (required for full training)",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke-only path (tiny config, 3 epochs, synthetic base)",
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
    # Full training hyperparameters (UNLOCKED 2026-05-14 per operator approval)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.997,
                   help="EMA decay (CLAUDE.md non-negotiable default 0.997).")
    p.add_argument("--noise-std", type=float, default=0.5,
                   help="Per-pixel noise during training (proxy-auth gap closer).")
    p.add_argument("--val-every-epochs", type=int, default=20)
    p.add_argument("--val-pair-count", type=int, default=32)
    p.add_argument("--max-pairs", type=int, default=None,
                   help="Cap pair decode for fast smoke iteration.")
    # Score-aware Lagrangian weights (council defaults)
    p.add_argument("--alpha-rate", type=float, default=25.0)
    p.add_argument("--beta-seg", type=float, default=100.0)
    p.add_argument("--gamma-pose", type=float, default=math.sqrt(10.0))
    p.add_argument("--pose-weight-scale", type=float, default=1.0)
    p.add_argument("--lambda-residual", type=float, default=0.1)
    # Post-train artifacts
    p.add_argument("--skip-auth-eval", action="store_true")
    p.add_argument("--skip-archive-build", action="store_true")
    p.add_argument("--enable-autocast-fp16", action="store_true",
                   help="Catalog #172; deferred until canonical autocast wraps land.")
    p.add_argument(
        "--enable-gt-scorer-cache",
        dest="enable_gt_scorer_cache",
        action="store_true",
        default=True,
        help="Enable canonical GTScorerCache for frozen GT scorer targets.",
    )
    p.add_argument(
        "--disable-gt-scorer-cache",
        dest="enable_gt_scorer_cache",
        action="store_false",
        help="Disable GTScorerCache and recompute GT scorer targets per step.",
    )
    p.add_argument("--enable-tf32", action="store_true",
                   help="Catalog #178; deferred until paired CPU/CUDA anchor lands.")
    return p


def _validate_full_cpu_flags(args: argparse.Namespace) -> None:
    """Catalog #197: --full-cpu MUST be paired with the advisory waiver flag."""
    if args.full_cpu and not args.advisory_cpu_explicitly_waived:
        raise SystemExit(
            "ERROR: --full-cpu requires --advisory-cpu-explicitly-waived per "
            "Catalog #197 (paired-flag attestation that the CPU-axis bypass "
            "is intentional and non-promotable)"
        )


def _validate_auth_eval_pair_scope(args: argparse.Namespace) -> None:
    """Refuse exact auth eval for capped pair smokes.

    The contest evaluator expects one raw stream containing all 1200 frames
    (600 pairs). A capped ``--max-pairs`` smoke emits a shorter archive by
    design, so it is a timing/training artifact unless auth eval is skipped.
    """
    if args.skip_auth_eval:
        return
    if args.max_pairs is not None and args.max_pairs < N_PAIRS_FULL:
        raise SystemExit(
            "ERROR: --max-pairs below N_PAIRS_FULL=600 emits truncated raw "
            "outputs and cannot produce a contest auth-eval score. Use "
            "--skip-auth-eval for capped timing smokes, or omit --max-pairs "
            "for full 600-pair contest auth eval."
        )


def _is_pair_capped_smoke(args: argparse.Namespace) -> bool:
    """Return True when the run intentionally emits fewer than contest pairs."""
    return args.max_pairs is not None and args.max_pairs < N_PAIRS_FULL


# ---------------------------------------------------------------------------
# Smoke entry path
# ---------------------------------------------------------------------------

def _smoke_main(args: argparse.Namespace) -> int:
    """Smoke entry: tiny config, synthetic base, 3 epochs, no scorer load.

    Per the build prompt this is the v1 entry point that lands and is
    smoke-tested. The full CUDA training path is gated behind Phase 2
    council approval (raises ``NotImplementedError``) to prevent
    accidental $15 Modal dispatches before council green-up.
    """
    _canon_pin_seeds(args.seed)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Small smoke config: 4 pairs, 24x32 frames, 6x8 residual coarse, 3 epochs.
    motion_mode = (
        MotionModelMode.SE3_PARAMETRIC
        if args.motion_mode == "se3_parametric"
        else MotionModelMode.OPTICAL_FLOW
    )
    cfg = WynerZivFrame0Config(
        motion_mode=motion_mode,
        num_pairs=4,
        output_height=24,
        output_width=32,
        flow_grid_h=6,
        flow_grid_w=8,
        residual_coarse_h=6,
        residual_coarse_w=8,
    )
    substrate = WynerZivFrame0Substrate(cfg).to(args.device)

    # Smoke synthetic data: deterministic random pairs (this is the
    # `_smoke_main` allowed path per Catalog #114; non-smoke training MUST
    # use upstream/videos/0.mkv pyav decode).
    torch.manual_seed(args.seed)
    smoke_frame_1 = torch.rand(4, 3, 24, 32, device=args.device)
    smoke_frame_0_target = torch.rand(4, 3, 24, 32, device=args.device)

    # Minimal pixel-MSE training loop (smoke only — no scorer load).
    opt = torch.optim.AdamW(substrate.parameters(), lr=args.lr)
    losses = []
    for _epoch in range(max(args.epochs, 3)):
        opt.zero_grad()
        f0_pred, _ = substrate.reconstruct_pair(smoke_frame_1)
        loss = (f0_pred - smoke_frame_0_target).pow(2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(substrate.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.item()))

    # Build the archive (smoke base bytes; production runs use the real
    # base substrate's archive bytes).
    base_bytes = b"D4_SMOKE_BASE_v0"
    base_sha = hashlib.sha256(base_bytes).hexdigest()
    residual_blob = encode_residual_blob(
        substrate.residual_coarse.detach().cpu(),
        coarse_hw=(cfg.residual_coarse_h, cfg.residual_coarse_w),
    )
    motion_mode_int = 0 if motion_mode == MotionModelMode.SE3_PARAMETRIC else 1
    se3_flat = substrate.motion.se3_flat if motion_mode_int == 0 else None
    flow_uv = substrate.motion.flow_uv if motion_mode_int == 1 else None
    archive_bytes = pack_archive(
        motion_mode=motion_mode_int,
        se3_flat=se3_flat.detach().cpu() if se3_flat is not None else None,
        flow_uv=flow_uv.detach().cpu() if flow_uv is not None else None,
        residual_blob=residual_blob,
        meta={
            "base_substrate_id": "smoke_base_substrate_v0",
            "motion_mode_label": args.motion_mode,
            "smoke": True,
            "git_head": _canon_git_head_sha(REPO_ROOT),
            "trained_at_utc": _canon_utc_now_iso(),
        },
        base_substrate_archive_sha256_hex=base_sha,
        base_substrate_bytes=base_bytes,
        num_pairs=cfg.num_pairs,
        flow_grid_h=cfg.flow_grid_h if motion_mode_int == 1 else 0,
        flow_grid_w=cfg.flow_grid_w if motion_mode_int == 1 else 0,
        residual_coarse_h=cfg.residual_coarse_h,
        residual_coarse_w=cfg.residual_coarse_w,
    )
    archive_path = out_dir / "0.bin"
    archive_path.write_bytes(archive_bytes)

    # Emit runtime tree + archive.zip per the canonical pattern.
    submission_dir = out_dir / "submission_dir"
    _write_runtime(submission_dir)
    (submission_dir / "0.bin").write_bytes(archive_bytes)
    archive_zip_path = out_dir / "archive.zip"
    _build_archive_zip(archive_zip_path, bin_bytes=archive_bytes)

    # Stats provenance
    archive_sha = _canon_sha256_bytes(archive_bytes)
    archive_zip_sha = _canon_sha256_bytes(archive_zip_path.read_bytes())
    archive_zip_size = archive_zip_path.stat().st_size
    final_loss = losses[-1] if losses else float("inf")
    stats = {
        "lane_id": SUBSTRATE_LANE_ID,
        "substrate_tag": SUBSTRATE_TAG,
        "smoke": True,
        "epochs": len(losses),
        "final_loss_proxy": final_loss,
        "archive_bytes": len(archive_bytes),
        "archive_sha256": archive_sha,
        "archive_zip_bytes": archive_zip_size,
        "archive_zip_sha256": archive_zip_sha,
        "base_substrate_archive_sha256": base_sha,
        "motion_mode": args.motion_mode,
        "cfg": asdict(cfg),
        "score_claim_valid": False,
        "evidence_grade": "smoke-no-scorer",
        "ready_for_exact_eval_dispatch": False,
        "promotion_eligible": False,
        "council_phase_2_required_before_full_dispatch": False,
        "git_head": _canon_git_head_sha(REPO_ROOT),
        "trained_at_utc": _canon_utc_now_iso(),
    }
    (out_dir / "stats.json").write_text(
        json.dumps(stats, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    print(
        f"[d4-smoke] OK final_loss={final_loss:.6f} archive={len(archive_bytes)}B "
        f"sha={archive_sha[:12]}... motion={args.motion_mode}"
    )
    return 0


def _write_runtime(submission_dir: Path) -> None:
    """Emit contest-compliant inflate.sh + inflate.py + vendored substrate.

    Per Catalog #146 the inflate.sh signature is 3-positional-arg
    ``inflate.sh <archive_dir> <output_dir> <file_list>``. Per Catalog #163
    the script uses ``set -euo pipefail`` for fail-closed semantics. The
    vendored substrate package + shared inflate runtime live under
    ``submission_dir/src/`` per the canonical pattern.
    """
    submission_dir.mkdir(parents=True, exist_ok=True)
    runtime_pkg = submission_dir / "src" / "tac" / "substrates" / "d4_wyner_ziv_frame_0"
    runtime_pkg.mkdir(parents=True, exist_ok=True)
    for pkg_init in (
        submission_dir / "src" / "tac" / "__init__.py",
        submission_dir / "src" / "tac" / "substrates" / "__init__.py",
        runtime_pkg / "__init__.py",
    ):
        pkg_init.write_text("", encoding="utf-8")
    substrate_src = REPO_ROOT / "src" / "tac" / "substrates" / "d4_wyner_ziv_frame_0"
    for name in (
        "architecture.py",
        "archive.py",
        "frame0_synthesis.py",
        "inflate.py",
        "motion_model.py",
        "residual_codec.py",
    ):
        shutil.copy2(substrate_src / name, runtime_pkg / name)
    # Runtime __init__.py is MINIMAL — only the inflate-time modules. The full
    # package __init__.py eagerly imports score_aware_loss which would pull in
    # scorer code (forbidden at inflate time per CLAUDE.md "Strict scorer rule").
    (runtime_pkg / "__init__.py").write_text(
        "\"\"\"D4 runtime package (inflate-time only — no scorer imports).\"\"\"\n"
        "from tac.substrates.d4_wyner_ziv_frame_0.architecture import (\n"
        "    BASE_SHA_HEX_LEN,\n"
        "    EVAL_HW,\n"
        "    NUM_PAIRS,\n"
        "    MotionModelMode,\n"
        "    WynerZivFrame0Config,\n"
        "    WynerZivFrame0Substrate,\n"
        ")\n"
        "from tac.substrates.d4_wyner_ziv_frame_0.archive import (\n"
        "    WZF01_MAGIC,\n"
        "    WZF01_SCHEMA_VERSION,\n"
        "    WynerZivFrame0Archive,\n"
        "    pack_archive,\n"
        "    parse_archive,\n"
        ")\n"
        "from tac.substrates.d4_wyner_ziv_frame_0.frame0_synthesis import synthesize_frame_0\n"
        "from tac.substrates.d4_wyner_ziv_frame_0.motion_model import (\n"
        "    OpticalFlowField,\n"
        "    SE3MotionParams,\n"
        "    apply_optical_flow,\n"
        "    apply_se3_motion,\n"
        ")\n"
        "from tac.substrates.d4_wyner_ziv_frame_0.residual_codec import (\n"
        "    decode_residual_blob,\n"
        "    encode_residual_blob,\n"
        ")\n"
        "__all__ = [\n"
        "    'BASE_SHA_HEX_LEN', 'EVAL_HW', 'MotionModelMode', 'NUM_PAIRS',\n"
        "    'OpticalFlowField', 'SE3MotionParams', 'WZF01_MAGIC',\n"
        "    'WZF01_SCHEMA_VERSION', 'WynerZivFrame0Archive',\n"
        "    'WynerZivFrame0Config', 'WynerZivFrame0Substrate',\n"
        "    'apply_optical_flow', 'apply_se3_motion', 'decode_residual_blob',\n"
        "    'encode_residual_blob', 'pack_archive', 'parse_archive',\n"
        "    'synthesize_frame_0',\n"
        "]\n",
        encoding="utf-8",
    )
    _canon_vendor_shared_inflate_runtime(submission_dir, repo_root=REPO_ROOT)

    inflate_sh = (
        "#!/usr/bin/env bash\n"
        "# D4 Wyner-Ziv frame-0 contest-compliant inflate runtime.\n"
        "# Per Catalog #146: 3-positional-arg signature.\n"
        "# Per Catalog #163: set -euo pipefail.\n"
        "# Per CLAUDE.md \"Strict scorer rule\": no scorer at inflate time.\n"
        "set -euo pipefail\n"
        "HERE=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"\n"
        "DATA_DIR=\"$1\"\n"
        "OUTPUT_DIR=\"$2\"\n"
        "FILE_LIST=\"$3\"\n"
        "mkdir -p \"$OUTPUT_DIR\"\n"
        "exec \"${PYTHON:-python3}\" \"$HERE/inflate.py\" "
        "\"$DATA_DIR\" \"$OUTPUT_DIR\" \"$FILE_LIST\"\n"
    )
    (submission_dir / "inflate.sh").write_text(inflate_sh, encoding="utf-8")
    (submission_dir / "inflate.sh").chmod(0o755)

    inflate_py = (
        "#!/usr/bin/env python\n"
        '"""D4 Wyner-Ziv frame-0 contest-compliant inflate runtime.\n'
        "\n"
        "Delegates to the vendored substrate CLI, which decodes each WZF01\n"
        "archive into one contest .raw tensor stream per file_list entry.\n"
        "No scorer-network imports (strict-scorer-rule contract).\n"
        '"""\n'
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "HERE = Path(__file__).resolve().parent\n"
        "sys.path.insert(0, str(HERE / 'src'))\n"
        "from tac.substrates.d4_wyner_ziv_frame_0.inflate import main_cli\n"
        "\n"
        "def main() -> int:\n"
        "    return main_cli()\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    sys.exit(main())\n"
    )
    (submission_dir / "inflate.py").write_text(inflate_py, encoding="utf-8")


def _build_archive_zip(
    archive_zip_path: Path, *, bin_bytes: bytes
) -> None:
    """Deterministic archive.zip containing ONLY the data payload (0.bin)."""
    archive_zip_path.parent.mkdir(parents=True, exist_ok=True)
    fixed_ts = (2026, 1, 1, 0, 0, 0)
    with zipfile.ZipFile(archive_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zi = zipfile.ZipInfo("0.bin", date_time=fixed_ts)
        zi.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(zi, bin_bytes)


# ---------------------------------------------------------------------------
# Full entry path — UNLOCKED 2026-05-14 per operator approval
# ---------------------------------------------------------------------------
# The 5-round adversarial council (25/25 PROCEED unanimous) is documented in
# ``feedback_d4_wyner_ziv_frame_0_landed_20260514.md``. Operator approved
# Phase 2 dispatch 2026-05-14 with directive "approved, proceed with all".
# Council-binding contract (CLAUDE.md non-negotiables) honored end-to-end.

def _run_val_loop(
    substrate: WynerZivFrame0Substrate,
    loss_fn,
    gt_pair_tensor: torch.Tensor,
    val_pair_indices: list[int],
    archive_bytes_proxy: torch.Tensor,
    gt_cache,
    device,
) -> float:
    """Validation pass with EMA shadow + torch.inference_mode (Catalog #180).

    Per the D4 OOM fix (lane_d4_oom_fix_minibatch_reconstruct_20260514) the
    val loop reconstructs each val pair via the per-pair pair_indices path
    rather than once for all 600 pairs. torch.inference_mode keeps activation
    memory bounded; per-pair reconstruct keeps peak memory at O(1) pairs.
    """
    substrate.eval()
    losses: list[float] = []
    device = gt_pair_tensor.device
    with torch.inference_mode():
        # gt_pair_tensor shape: (N, 2, 3, H, W) in [0, 255]; convert to unit
        # range only the per-batch slice we need.
        for pair_idx in val_pair_indices:
            pair_idx = int(pair_idx)
            idx_tensor = torch.tensor([pair_idx], device=device, dtype=torch.long)
            f1_unit = gt_pair_tensor[idx_tensor, 1].contiguous() / 255.0
            f0_recon, f1_passthrough = substrate.reconstruct_pair(
                f1_unit, pair_indices=idx_tensor
            )
            rgb_0 = (f0_recon * 255.0).clamp(0.0, 255.0)
            rgb_1 = (f1_passthrough * 255.0).clamp(0.0, 255.0)
            gt_a = gt_pair_tensor[idx_tensor, 0]
            gt_b = gt_pair_tensor[idx_tensor, 1]
            gt_pose_batch = gt_seg_batch = None
            gt_seg_already_probs = None
            if gt_cache is not None:
                gt_pose_batch, gt_seg_batch = gt_cache.lookup(
                    idx_tensor, device=device
                )
                gt_seg_already_probs = gt_cache.seg_already_probs
            try:
                loss, _ = loss_fn(
                    reconstructed_rgb_0=rgb_0,
                    reconstructed_rgb_1=rgb_1,
                    gt_rgb_0=gt_a,
                    gt_rgb_1=gt_b,
                    archive_bytes_proxy=archive_bytes_proxy,
                    residual_coarse=substrate.residual_coarse,
                    apply_eval_roundtrip=True,
                    noise_std=0.0,
                    gt_pose_batch=gt_pose_batch,
                    gt_seg_batch=gt_seg_batch,
                    gt_seg_already_probs=gt_seg_already_probs,
                )
            except Exception as exc:
                print(f"[{SUBSTRATE_TAG}-val] WARN pair {pair_idx} val skipped: {exc!r}")
                continue
            if torch.isfinite(loss):
                losses.append(float(loss.detach().cpu()))
    return float(sum(losses) / len(losses)) if losses else math.inf


def _full_main(args: argparse.Namespace) -> int:
    """Full CUDA training path — score-aware Lagrangian end-to-end.

    UNLOCKED 2026-05-14 per operator approval. The 5-round adversarial
    council (25/25 PROCEED) is documented in
    ``feedback_d4_wyner_ziv_frame_0_landed_20260514.md``.
    """
    from tac.differentiable_eval_roundtrip import (
        patch_upstream_yuv6_globally,
        unpatch_upstream_yuv6,
    )
    from tac.scorer import load_differentiable_scorers
    from tac.substrates.d4_wyner_ziv_frame_0.score_aware_loss import (
        WynerZivFrame0LossWeights,
        WynerZivFrame0ScoreAwareLoss,
    )
    from tac.training import EMA

    _canon_pin_seeds(args.seed)
    device = _canon_device_or_die(args.device, smoke=False, substrate_tag=SUBSTRATE_TAG)

    if args.enable_tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage_log: list[dict[str, str]] = []

    def _stage(name: str) -> None:
        msg = {"stage": name, "at": _canon_utc_now_iso()}
        stage_log.append(msg)
        print(f"[{SUBSTRATE_TAG}-full] {name} @ {msg['at']}")

    _stage("seed_pinned")

    # 1. Patch upstream rgb_to_yuv6 BEFORE scorer construction (Catalog #187).
    yuv6_token = patch_upstream_yuv6_globally()
    _stage("upstream_yuv6_patched")

    auth_eval_gate_json_path, auth_eval_json_path = _resolve_auth_eval_json_paths(
        args.output_dir
    )
    archive_zip_path = args.output_dir / "archive.zip"
    archive_zip_sha = ""
    archive_zip_size = 0
    bin_sha = ""
    bin_size = 0
    n_params = 0
    best_val_lag = math.inf
    best_epoch = -1

    try:
        # 2. Load differentiable scorers (frozen).
        posenet, segnet = load_differentiable_scorers(args.upstream_dir, device=device)
        for p in list(posenet.parameters()) + list(segnet.parameters()):
            p.requires_grad_(False)
        posenet.eval()
        segnet.eval()
        _stage("scorers_loaded")

        # 3. Decode real target pairs.
        print(f"[{SUBSTRATE_TAG}-full] decoding pairs from {args.video_path}")
        gt_pair_tensor = _canon_decode_real_pairs(
            args.video_path,
            n_pairs=N_PAIRS_FULL,
            substrate_tag=SUBSTRATE_TAG,
            max_pairs=args.max_pairs,
            repo_root=REPO_ROOT,
        ).to(device)
        n_pairs = int(gt_pair_tensor.shape[0])
        _stage(f"pairs_decoded_{n_pairs}")
        # gt_pair_tensor: (n_pairs, 2, 3, 384, 512) in [0, 255].

        val_count = max(1, min(args.val_pair_count, max(1, n_pairs // 8)))
        val_idx_start = n_pairs - val_count
        train_indices_pool = list(range(val_idx_start))
        val_indices_pool = list(range(val_idx_start, n_pairs))

        # 4. Build substrate at full resolution + n_pairs.
        motion_mode = (
            MotionModelMode.SE3_PARAMETRIC
            if args.motion_mode == "se3_parametric"
            else MotionModelMode.OPTICAL_FLOW
        )
        cfg = WynerZivFrame0Config(
            motion_mode=motion_mode,
            num_pairs=n_pairs,
            output_height=EVAL_HW[0],
            output_width=EVAL_HW[1],
            flow_grid_h=args.flow_grid_h,
            flow_grid_w=args.flow_grid_w,
            residual_coarse_h=args.residual_coarse_h,
            residual_coarse_w=args.residual_coarse_w,
            residual_loss_weight=args.lambda_residual,
        )
        substrate = WynerZivFrame0Substrate(cfg).to(device)
        n_params = sum(p.numel() for p in substrate.parameters())
        print(f"[{SUBSTRATE_TAG}-full] substrate params: {n_params:,}")
        _stage(f"substrate_built_{n_params}_params")
        opt_ctx = _build_optimized_training_context(
            args,
            scorers=(posenet, segnet),
            gt_pairs=gt_pair_tensor,
            substrate_model=substrate,
            device=device,
        )
        substrate = opt_ctx.substrate_model
        gt_cache = opt_ctx.gt_cache
        if gt_cache is not None:
            print(gt_cache.summary_line())
            _stage("gt_scorer_cache_built")
        else:
            _stage("gt_scorer_cache_disabled")

        # 5. EMA shadow (Catalog #88).
        ema = EMA(substrate, decay=args.ema_decay)
        _stage(f"ema_wired_decay_{args.ema_decay}")

        # 6. Score-aware Lagrangian.
        weights = WynerZivFrame0LossWeights(
            alpha_rate=args.alpha_rate,
            beta_seg=args.beta_seg,
            gamma_pose=args.gamma_pose,
            pose_weight_scale=args.pose_weight_scale,
            lambda_residual=args.lambda_residual,
            contest_normalizer=CONTEST_NORMALIZER,
        )
        loss_fn = WynerZivFrame0ScoreAwareLoss(
            seg_scorer=segnet, pose_scorer=posenet, weights=weights
        )
        _stage("lagrangian_built")

        # 7. Optimizer (AdamW + cosine annealing).
        optimizer = torch.optim.AdamW(
            substrate.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs)
        )

        # 8. Train loop.
        train_started_at = time.time()
        ckpt_best_path = args.output_dir / "best.pt"
        nan_strike = 0
        max_nan_strikes = 3

        # Closed-form rate proxy: motion + residual + base bytes estimate.
        # SE3: ~5KB after brotli; optical_flow: ~30-50KB.
        # Residual: 50-150KB after brotli.
        # Base: 178KB (A1/PR101 placeholder for proxy).
        motion_bytes_proxy = (
            5_000 if motion_mode == MotionModelMode.SE3_PARAMETRIC else 40_000
        )
        residual_bytes_proxy = (
            cfg.residual_coarse_h * cfg.residual_coarse_w * 3 * n_pairs // 4
        )  # rough int8+brotli estimate
        base_bytes_proxy = 178_000
        total_proxy_bytes = motion_bytes_proxy + residual_bytes_proxy + base_bytes_proxy
        archive_bytes_proxy = torch.tensor(float(total_proxy_bytes), device=device)
        print(
            f"[{SUBSTRATE_TAG}-full] archive_bytes_proxy: motion={motion_bytes_proxy}B "
            f"residual={residual_bytes_proxy}B base={base_bytes_proxy}B "
            f"total={total_proxy_bytes}B"
        )

        # Pre-cache GT frame_1 batch tensor for forward pass.
        # Per the D4 OOM fix (lane_d4_oom_fix_minibatch_reconstruct_20260514)
        # we DO NOT call reconstruct_pair(gt_f1_all) once per epoch; that
        # full-600-pair forward (warp + residual upsample 48x64 -> 384x512)
        # needs ~13 GB activation memory and OOMs on T4 (14.56 GB capacity).
        # Instead, reconstruct the mini-batch inside the inner loop via the
        # new pair_indices kwarg so each backward releases its own graph.
        gt_f1_all = gt_pair_tensor[:, 1].contiguous() / 255.0  # unit-domain for reconstruct_pair

        for epoch in range(args.epochs):
            substrate.train()
            random.shuffle(train_indices_pool)
            epoch_losses: list[float] = []

            # Shuffle and batch through training indices.
            for batch_start in range(0, len(train_indices_pool), args.batch_size):
                batch_indices = train_indices_pool[batch_start : batch_start + args.batch_size]
                if not batch_indices:
                    continue
                batch_idx_tensor = torch.tensor(
                    batch_indices, device=device, dtype=torch.long
                )
                # Mini-batch reconstruct: index motion params + residual for
                # the selected pairs only; activation memory drops from
                # O(600) to O(batch_size).
                f1_batch = gt_f1_all.index_select(0, batch_idx_tensor)
                f0_recon, f1_unchanged = substrate.reconstruct_pair(
                    f1_batch, pair_indices=batch_idx_tensor
                )
                rgb_0 = (f0_recon * 255.0).clamp(0.0, 255.0)
                rgb_1 = (f1_unchanged * 255.0).clamp(0.0, 255.0)
                gt_0 = gt_pair_tensor[batch_idx_tensor, 0]
                gt_1 = gt_pair_tensor[batch_idx_tensor, 1]
                gt_pose_batch = gt_seg_batch = None
                gt_seg_already_probs = None
                if gt_cache is not None:
                    gt_pose_batch, gt_seg_batch = gt_cache.lookup(
                        batch_idx_tensor, device=device
                    )
                    gt_seg_already_probs = gt_cache.seg_already_probs

                loss, parts = loss_fn(
                    reconstructed_rgb_0=rgb_0,
                    reconstructed_rgb_1=rgb_1,
                    gt_rgb_0=gt_0,
                    gt_rgb_1=gt_1,
                    archive_bytes_proxy=archive_bytes_proxy,
                    residual_coarse=substrate.residual_coarse,
                    apply_eval_roundtrip=True,
                    noise_std=args.noise_std,
                    gt_pose_batch=gt_pose_batch,
                    gt_seg_batch=gt_seg_batch,
                    gt_seg_already_probs=gt_seg_already_probs,
                )

                if not torch.isfinite(loss):
                    nan_strike += 1
                    print(
                        f"[{SUBSTRATE_TAG}-full] NaN strike {nan_strike}/{max_nan_strikes}"
                    )
                    if nan_strike >= max_nan_strikes:
                        raise RuntimeError("NaN watchdog tripped")
                    continue
                nan_strike = 0

                optimizer.zero_grad()
                # retain_graph no longer needed: each batch builds its own
                # forward graph via mini-batched reconstruct_pair.
                loss.backward()
                torch.nn.utils.clip_grad_norm_(substrate.parameters(), args.grad_clip)
                optimizer.step()
                ema.update(substrate)
                epoch_losses.append(float(loss.detach().cpu()))

            scheduler.step()

            if epoch % max(1, args.val_every_epochs) == 0 or epoch == args.epochs - 1:
                # EMA snapshot+restore per CLAUDE.md "EMA non-negotiable".
                live_state = {
                    k: v.detach().clone() for k, v in substrate.state_dict().items()
                }
                ema.apply(substrate)
                ema_state_for_ckpt: dict[str, torch.Tensor] | None = None
                try:
                    val_lag = _run_val_loop(
                        substrate, loss_fn, gt_pair_tensor, val_indices_pool,
                        archive_bytes_proxy, gt_cache, device,
                    )
                    ema_state_for_ckpt = {
                        k: v.detach().cpu().clone()
                        for k, v in substrate.state_dict().items()
                    }
                finally:
                    substrate.load_state_dict(live_state)
                    substrate.train()

                avg_train = (
                    sum(epoch_losses) / len(epoch_losses)
                ) if epoch_losses else math.nan
                print(
                    f"[{SUBSTRATE_TAG}-full] epoch {epoch:4d}: train={avg_train:.5f} "
                    f"val={val_lag:.5f} (best={best_val_lag:.5f})"
                )
                if val_lag < best_val_lag and ema_state_for_ckpt is not None:
                    best_val_lag = val_lag
                    best_epoch = epoch
                    torch.save(
                        {
                            "state_dict": ema_state_for_ckpt,
                            "config": asdict(cfg),
                            "epoch": epoch,
                            "val_lag": val_lag,
                        },
                        ckpt_best_path,
                    )

        train_elapsed = time.time() - train_started_at
        _stage(
            f"trained_best_epoch_{best_epoch}_val_lag_{best_val_lag:.5f}_"
            f"elapsed_{train_elapsed:.1f}s"
        )

        # 9. Load best EMA checkpoint and build the archive.
        if not args.skip_archive_build:
            if ckpt_best_path.exists():
                best_ckpt = torch.load(
                    ckpt_best_path, weights_only=False, map_location=device
                )  # WEIGHTS_ONLY_FALSE_OK:trusted-local-checkpoint-from-this-process
                substrate.load_state_dict(best_ckpt["state_dict"])
            substrate.eval()

            # Pack archive: extract motion params + residual blob + smoke base bytes.
            motion_mode_int = 0 if motion_mode == MotionModelMode.SE3_PARAMETRIC else 1
            if motion_mode_int == 0:
                se3_flat = substrate.motion.se3_flat.detach().cpu()
                flow_uv = None
            else:
                se3_flat = None
                flow_uv = substrate.motion.flow_uv.detach().cpu()
            residual_blob = encode_residual_blob(
                substrate.residual_coarse.detach().cpu(),
                coarse_hw=(cfg.residual_coarse_h, cfg.residual_coarse_w),
            )
            # V1: smoke base provider for inflate-time custody (self-contained).
            # Phase 3 composes with A1/PR101/HDM8 via --base-archive-path.
            base_bytes = b"D4_FULL_BASE_v0_smoke_provider_self_contained"
            base_sha = hashlib.sha256(base_bytes).hexdigest()
            bin_bytes = pack_archive(
                motion_mode=motion_mode_int,
                se3_flat=se3_flat,
                flow_uv=flow_uv,
                residual_blob=residual_blob,
                meta={
                    "base_substrate_id": "smoke_base_substrate_v0",
                    "motion_mode_label": args.motion_mode,
                    "smoke": False,
                    "v1_self_contained_custody": True,
                    "best_val_lag": float(best_val_lag),
                    "best_epoch": int(best_epoch),
                    "git_head": _canon_git_head_sha(REPO_ROOT),
                    "trained_at_utc": _canon_utc_now_iso(),
                },
                base_substrate_archive_sha256_hex=base_sha,
                base_substrate_bytes=base_bytes,
                num_pairs=cfg.num_pairs,
                flow_grid_h=cfg.flow_grid_h if motion_mode_int == 1 else 0,
                flow_grid_w=cfg.flow_grid_w if motion_mode_int == 1 else 0,
                residual_coarse_h=cfg.residual_coarse_h,
                residual_coarse_w=cfg.residual_coarse_w,
            )
            bin_sha = _canon_sha256_bytes(bin_bytes)
            bin_size = len(bin_bytes)
            print(
                f"[{SUBSTRATE_TAG}-full] WZF01 archive: {bin_size} B "
                f"sha256={bin_sha[:16]}..."
            )
            _stage(f"archive_built_{bin_size}_B_sha{bin_sha[:8]}")

            # 10. Build runtime tree + archive.zip.
            submission_dir = args.output_dir / "submission_dir"
            _write_runtime(submission_dir)
            (submission_dir / "0.bin").write_bytes(bin_bytes)
            _build_archive_zip(archive_zip_path, bin_bytes=bin_bytes)
            archive_zip_sha = _canon_sha256_bytes(archive_zip_path.read_bytes())
            archive_zip_size = archive_zip_path.stat().st_size
            shutil.copy2(archive_zip_path, submission_dir / "archive.zip")
            _stage("archive_emitted")

            # 11. Auth eval ([contest-CUDA] inline) through the canonical gate.
            if not args.skip_auth_eval:
                auth_result = _canon_gate_auth_eval_call(
                    args=args,
                    archive_zip=archive_zip_path,
                    inflate_sh=submission_dir / "inflate.sh",
                    upstream_dir=args.upstream_dir,
                    output_json=auth_eval_gate_json_path,
                    contest_auth_eval_script=CONTEST_AUTH_EVAL_SCRIPT,
                    substrate_tag=SUBSTRATE_TAG,
                    device=device,
                )
                if auth_result is not None:
                    if auth_eval_gate_json_path != auth_eval_json_path:
                        auth_eval_json_path.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(auth_eval_gate_json_path, auth_eval_json_path)
                    _canon_require_contest_cuda_auth_eval_claim(
                        auth_eval_json_path,
                        archive_sha256=archive_zip_sha,
                        substrate_tag=SUBSTRATE_TAG,
                    )
                    _stage("auth_eval_cuda_done_valid_claim")
                else:
                    _stage("auth_eval_skipped_gate_refused")
    finally:
        unpatch_upstream_yuv6(yuv6_token)
        _stage("upstream_yuv6_unpatched")

    # 12. Posterior update (Catalog #128 atomic fcntl).
    if (not args.skip_auth_eval) and auth_eval_json_path.exists():
        try:
            from tac.continual_learning import (
                posterior_update_locked_from_auth_eval_json,
            )

            update = posterior_update_locked_from_auth_eval_json(auth_eval_json_path)
            print(
                f"[{SUBSTRATE_TAG}-full] posterior_update accepted="
                f"{getattr(update, 'accepted', '?')}"
            )
            _stage("posterior_updated")
        except Exception as exc:
            print(
                f"[{SUBSTRATE_TAG}-full] WARN posterior_update failed: {exc!r}"
            )

    # 13. Provenance.
    provenance = {
        "lane_id": SUBSTRATE_LANE_ID,
        "substrate_tag": SUBSTRATE_TAG,
        "started_at_utc": stage_log[0]["at"] if stage_log else _canon_utc_now_iso(),
        "completed_at_utc": _canon_utc_now_iso(),
        "git_head": _canon_git_head_sha(REPO_ROOT),
        "bin_sha256": bin_sha,
        "bin_bytes": bin_size,
        "archive_zip_sha256": archive_zip_sha,
        "archive_zip_bytes": archive_zip_size,
        "n_params": n_params,
        "best_val_lag": float(best_val_lag) if math.isfinite(best_val_lag) else None,
        "best_epoch": best_epoch,
        "epochs": args.epochs,
        "device": str(device),
        "trainer_proxy_axis": opt_ctx.eval_axis_label,
        "trainer_proxy_promotion_requirement": opt_ctx.promotion_requirement,
        "gt_scorer_cache_enabled": gt_cache is not None,
        "motion_mode": args.motion_mode,
        "council_phase_2_unanimous_seal": True,
        "council_review_anchor": (
            "feedback_d4_wyner_ziv_frame_0_landed_20260514.md"
        ),
        "design_memo": (
            ".omx/research/deep_math_geometry_manifolds_synthesis_20260514.md"
        ),
        "stage_log": stage_log,
        "auth_eval_gate_json_path": str(auth_eval_gate_json_path),
        "auth_eval_json_path": str(auth_eval_json_path),
        "hardware_substrate_cuda": _canon_detect_hardware_substrate(
            axis="cuda",
            substrate_tag=SUBSTRATE_TAG,
            env_var_candidates=("D4_WYNER_ZIV_FRAME_0_GPU", "MODAL_GPU"),
        ),
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )
    pair_capped_smoke = _is_pair_capped_smoke(args)
    manifest = {
        "schema": "d4_wzf0_training_artifact_manifest_v1",
        "lane_id": SUBSTRATE_LANE_ID,
        "substrate_tag": SUBSTRATE_TAG,
        "training_mode": "smoke" if pair_capped_smoke else "full",
        "research_only": pair_capped_smoke,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "archive_bytes": bin_size,
        "archive_sha256": bin_sha,
        "archive_zip_bytes": archive_zip_size,
        "archive_zip_sha256": archive_zip_sha,
        "max_pairs": args.max_pairs,
        "n_pairs_full_required_for_auth_eval": N_PAIRS_FULL,
        "auth_eval_skipped": bool(args.skip_auth_eval),
        "auth_eval_skipped_reason": (
            "pair_capped_smoke_emits_truncated_raw_stream"
            if pair_capped_smoke and args.skip_auth_eval
            else ""
        ),
        "result": {
            "training_mode": "smoke" if pair_capped_smoke else "full",
            "archive_bytes": bin_size,
            "archive_sha256": bin_sha,
            "archive_zip_bytes": archive_zip_size,
            "archive_zip_sha256": archive_zip_sha,
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"[{SUBSTRATE_TAG}-full] wrote {args.output_dir / 'provenance.json'}"
    )
    return 0


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _validate_full_cpu_flags(args)
    if args.smoke:
        return _smoke_main(args)
    _validate_auth_eval_pair_scope(args)
    return _full_main(args)


if __name__ == "__main__":
    sys.exit(main())
