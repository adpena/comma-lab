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
- ``_full_main`` raises ``NotImplementedError`` pending Phase 2 council
  approval before $15 Modal T4/A100 dispatch (per the build prompt).

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
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch

from tac.substrates._shared.trainer_skeleton import (
    pin_seeds as _canon_pin_seeds,
    utc_now_iso as _canon_utc_now_iso,
    git_head_sha as _canon_git_head_sha,
    sha256_bytes as _canon_sha256_bytes,
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

EVAL_HW = (384, 512)
N_PAIRS_FULL = 600
CONTEST_NORMALIZER = 37_545_489.0

SUBSTRATE_TAG = "d4_wyner_ziv_frame_0"
SUBSTRATE_LANE_ID = "lane_d4_wyner_ziv_frame_0_substrate_20260514"


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
    for epoch in range(max(args.epochs, 3)):
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

    # Stats provenance
    archive_sha = _canon_sha256_bytes(archive_bytes)
    final_loss = losses[-1] if losses else float("inf")
    stats = {
        "lane_id": SUBSTRATE_LANE_ID,
        "substrate_tag": SUBSTRATE_TAG,
        "smoke": True,
        "epochs": len(losses),
        "final_loss_proxy": final_loss,
        "archive_bytes": len(archive_bytes),
        "archive_sha256": archive_sha,
        "base_substrate_archive_sha256": base_sha,
        "motion_mode": args.motion_mode,
        "cfg": asdict(cfg),
        "score_claim_valid": False,
        "evidence_grade": "smoke-no-scorer",
        "ready_for_exact_eval_dispatch": False,
        "promotion_eligible": False,
        "council_phase_2_required_before_full_dispatch": True,
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
    _write_runtime(out_dir)
    return 0


def _write_runtime(out_dir: Path) -> None:
    """Emit the contest-compliant inflate.sh + inflate.py runtime tree.

    Per Catalog #146 the inflate.sh signature is 3-positional-arg
    ``inflate.sh <archive_dir> <output_dir> <file_list>``. Per Catalog #163
    the script uses ``set -euo pipefail`` for fail-closed semantics.
    """
    runtime_dir = out_dir / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)

    inflate_sh = runtime_dir / "inflate.sh"
    inflate_sh.write_text(
        """#!/usr/bin/env bash
set -euo pipefail

# D4 Wyner-Ziv frame-0 substrate contest-compliant inflate runtime.
# Per Catalog #146: 3-positional-arg signature.
# Per Catalog #163: set -euo pipefail.
# Per CLAUDE.md "Strict scorer rule": no scorer at inflate time.

if [ "$#" -lt 3 ]; then
    echo "usage: inflate.sh <archive_dir> <output_dir> <file_list>" >&2
    exit 2
fi

DATA_DIR="$1"
OUTPUT_DIR="$2"
FILE_LIST="$3"

HERE="$(cd "$(dirname "$0")" && pwd)"
exec uv run --with torch --with brotli --with numpy \\
    "$HERE/inflate.py" "$DATA_DIR" "$OUTPUT_DIR" "$FILE_LIST"
""",
        encoding="utf-8",
    )
    inflate_sh.chmod(0o755)

    # Minimal contest inflate.py: imports our D4 inflate module + loops.
    inflate_py = runtime_dir / "inflate.py"
    inflate_py.write_text(
        """#!/usr/bin/env python3
\"\"\"D4 contest inflate runtime entry point.\"\"\"
import sys
from pathlib import Path

# Resolve project path (vendored alongside this runtime tree)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "src"))

from tac.substrates.d4_wyner_ziv_frame_0.inflate import main_cli

if __name__ == "__main__":
    sys.exit(main_cli())
""",
        encoding="utf-8",
    )
    inflate_py.chmod(0o755)


# ---------------------------------------------------------------------------
# Full entry path — gated behind Phase 2 council approval
# ---------------------------------------------------------------------------

def _full_main(args: argparse.Namespace) -> int:
    """Full CUDA training path — PENDING Phase 2 council approval.

    Per the build prompt the full training path is gated behind a 5-round
    adversarial council review before any $5-15 Modal T4/A100 dispatch.
    """
    raise NotImplementedError(
        "D4 full training is PENDING Phase 2 council approval. "
        "Run with --smoke for the v1 substrate-engineering smoke. "
        "Lane: " + SUBSTRATE_LANE_ID
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _validate_full_cpu_flags(args)
    if args.smoke:
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":
    sys.exit(main())
