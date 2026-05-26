# SPDX-License-Identifier: MIT
"""Train the Z4 cooperative-receiver-loss substrate (Time-Traveler L5 staircase Step 2).

Per `.omx/research/campaign_z4_cooperative_receiver_loss_20260514.md` and
`feedback_grand_council_maximize_value_landed_20260514.md`: Step 2 of the
Time-Traveler across-class staircase. Loss-only intervention through the
canonical scorer-aware Lagrangian; predicted ΔS −0.005 to −0.020 vs A1 0.1928
[contest-CPU 1to1] (Time-Traveler council conservative band).

Council-binding contract (CLAUDE.md non-negotiables) honored end-to-end:

- Train against ``upstream/videos/0.mkv`` decoded via pyav (Catalog #114).
- Patch upstream ``rgb_to_yuv6`` via ``patch_upstream_yuv6_globally`` BEFORE
  scorer construction (Catalog #187).
- ``load_differentiable_scorers`` for SegNet/PoseNet (no scorer load at inflate).
- ``apply_eval_roundtrip_during_training`` (Catalog #5) inside the per-batch
  loop via ``CooperativeReceiverScoreAwareLoss``.
- ``tac.training.EMA(decay=0.997)`` (Catalog #88).
- Score-domain cooperative-receiver Lagrangian (HNeRV parity L6 + Atick-Redlich).
- End with CUDA auth eval on best EMA checkpoint via canonical
  ``smoke_auth_eval_gate.gate_auth_eval_call`` (Catalog #226).
- TIER_1_OPERATOR_REQUIRED_FLAGS declared (Catalog #151 + #168 AnnAssign).
- ``--full-cpu`` opt-in coupled with ``--advisory-cpu-explicitly-waived`` (Catalog #197).

Phase 2 Council Verdict (2026-05-15, 11/11 unanimous LIFT) per
`feedback_z4_z5_phase_2_council_deliberation_landed_20260515.md`: Hotz-glue
canonical-helpers pattern (~150 LOC scope; reuse trainer_skeleton +
score_aware_common + smoke_auth_eval_gate; no reinvention). Contrarian +
Fridrich binding stipulation: paired ``--lambda-pixel`` ablation ([0.0, 1.0])
to disambiguate Atick-Redlich cooperative-receiver vs marginal-gradient-
alignment hypotheses.

V1 SCOPE: ``_smoke_main`` builds a tiny config, trains for ≤3 epochs on
synthetic data, runs archive pack + parse + inflate roundtrip, and emits a
contest-compliant runtime tree (no scorer load).

V2 SCOPE: ``_full_main`` decodes ``upstream/videos/0.mkv`` via pyav, trains
the canonical encoder+decoder+per-pair-latents under the Z4 cooperative-
receiver Lagrangian (β_seg + γ_pose · sqrt + λ_pixel · MSE), packs the
Z4CR1 monolithic archive, emits the contest-compliant runtime tree, runs
CUDA auth eval via ``gate_auth_eval_call`` on the EMA-best checkpoint, and
posts the result to the continual-learning posterior.

Usage (smoke; macOS CPU or Linux CPU, tiny config, ~3 epochs)::

    .venv/bin/python experiments/train_substrate_z4_cooperative_receiver_loss.py \\
        --video-path upstream/videos/0.mkv \\
        --output-dir experiments/results/z4_smoke_<utc> \\
        --epochs 3 --device cpu --smoke

Usage (full; CUDA-required Modal T4; gated by Phase 2 council approval)::

    .venv/bin/python experiments/train_substrate_z4_cooperative_receiver_loss.py \\
        --video-path upstream/videos/0.mkv \\
        --output-dir experiments/results/z4_<utc> \\
        --epochs 200 --batch-size 4 --lr 5e-4 --device cuda --lambda-pixel 1.0
"""
# Z4-V2 Wunderkind E1 Tier-1 engineering wave 2026-05-15: autocast/TF32/torch.compile
# wired via canonical build_optimized_training_context + autocast_aware_forward per
# Catalogs #172/#178/#179 + #228 (build_optimized_training_context). Hand-rolled
# CooperativeReceiverScoreAwareLoss kept as the rate+pixel composition wrapper but
# its inner seg+pose mechanics are mathematically equivalent to canonical
# cooperative_receiver_loss (Atick-Redlich 1990) per Catalog #226-style routing
# through score_pair_components_dispatch (Catalog #164). Phase B.2 council root-cause
# (per feedback_grand_council_evidence_review_modal_failures...): the Z4 lambda=0
# timeout was driven by autocast=false + deterministic-CUDA bicubic-backward, NOT
# by hand-rolled loss internals — so the structural fix is: enable the Tier-1
# primitives, relax determinism around backward pass for bicubic interpolate.
# NO_GRAD_WAIVED:eval-time-scorer-forward-wrapped-in-torch.inference_mode-inside-_run_val_loop
# SYNTHETIC_NON_SMOKE_OK:_smoke_main-only-uses-synthetic-data-_full_main-decodes-real-video
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

import torch

from tac.substrates._shared.smoke_auth_eval_gate import (
    gate_auth_eval_call as _canon_gate_auth_eval_call,
)
from tac.substrates._shared.trainer_skeleton import (
    build_optimized_training_context as _canon_build_optimized_training_context,
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
    sha256_bytes as _canon_sha256_bytes,
)
from tac.substrates._shared.trainer_skeleton import (
    utc_now_iso as _canon_utc_now_iso,
)
from tac.substrates._shared.trainer_skeleton import (
    vendor_shared_inflate_runtime as _canon_vendor_shared_inflate_runtime,
)
from tac.substrates.z4_cooperative_receiver_loss import (
    CooperativeReceiverConfig,
    CooperativeReceiverLossWeights,
    CooperativeReceiverScoreAwareLoss,
    CooperativeReceiverSubstrate,
    pack_archive,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO_PATH = REPO_ROOT / "upstream" / "videos" / "0.mkv"
DEFAULT_UPSTREAM_DIR = REPO_ROOT / "upstream"
CONTEST_AUTH_EVAL_SCRIPT = REPO_ROOT / "experiments" / "contest_auth_eval.py"
SUBSTRATE_TAG = "z4_cooperative_receiver_loss"
SUBSTRATE_LANE_ID = "lane_z4_cooperative_receiver_loss_step2_20260514"


# ---------------------------------------------------------------------------
# Catalog #151 manifest — every flag below must be threaded by any operator
# wrapper. AnnAssign per Catalog #168 (NOT bare Assign).
# ---------------------------------------------------------------------------
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--video-path": {
        "env": "Z4_VIDEO_PATH",
        "rationale": (
            "Path to the contest video `upstream/videos/0.mkv` decoded via "
            "pyav into per-pair frames; required for non-smoke training"
        ),
        "default": str(DEFAULT_VIDEO_PATH),
        "required_input_file": True,
    },
    "--output-dir": {
        "env": "Z4_OUTPUT_DIR",
        "rationale": (
            "Output directory for checkpoints, archive, stats, runtime tree, "
            "auth eval JSON; must be writable + outside /tmp"
        ),
        "default": None,
    },
    "--epochs": {
        "env": "Z4_EPOCHS",
        "rationale": "Training epoch count; smoke=3, Modal T4 full=200",
        "default": "200",
    },
    "--batch-size": {
        "env": "Z4_BATCH_SIZE",
        "rationale": "Per-step pair count; T4 handles 4-8 at 384x512",
        "default": "4",
    },
    "--lr": {
        "env": "Z4_LR",
        "rationale": "AdamW base learning rate; default 5e-4 per substrate skeleton",
        "default": "5e-4",
    },
    "--lambda-pixel": {
        "env": "Z4_LAMBDA_PIXEL",
        "rationale": (
            "Cooperative-receiver pixel-MSE residual weight; "
            "0.0 = pure Atick-Redlich; 1.0 = Z3 baseline (pixel-MSE-only)"
        ),
        "default": "0.0",
    },
    "--beta-seg": {
        "env": "Z4_BETA_SEG",
        "rationale": "SegNet distortion weight in cooperative-receiver Lagrangian (contest formula = 100)",
        "default": "100.0",
    },
    "--gamma-pose": {
        "env": "Z4_GAMMA_POSE",
        "rationale": "PoseNet distortion sqrt-weight (contest formula = sqrt(10))",
        "default": str(math.sqrt(10.0)),
    },
    "--enable-autocast-fp16": {
        "env": "Z4_ENABLE_AUTOCAST_FP16",
        "rationale": (
            "Catalog #172; Tier-1 engineering primitive wired via canonical "
            "build_optimized_training_context + autocast_aware_forward "
            "(Wunderkind E1 substitution 2026-05-15)"
        ),
        "default": "false",
    },
    "--enable-tf32": {
        "env": "Z4_ENABLE_TF32",
        "rationale": (
            "Catalog #178; TF32 fast-math on Ampere/Hopper. device_or_die "
            "auto-enables on CUDA; this flag is the operator-visible attestation"
        ),
        "default": "true",
    },
    "--enable-torch-compile": {
        "env": "Z4_ENABLE_TORCH_COMPILE",
        "rationale": (
            "Catalog #179; torch.compile (Inductor) on the substrate. Tier-2 "
            "engineering primitive wired via canonical compile_with_fallback"
        ),
        "default": "false",
    },
    "--enable-gt-scorer-cache": {
        "env": "Z4_ENABLE_GT_SCORER_CACHE",
        "rationale": (
            "Catalog #228; F3 GTScorerCache pre-computes GT scorer outputs once "
            "(~50%% scorer compute savings; mathematically identical to GT-forward)"
        ),
        "default": "false",
    },
    "--relax-determinism-for-backward": {
        "env": "Z4_RELAX_DETERMINISM",
        "rationale": (
            "Phase B.2 root-cause fix: bicubic interpolate backward is not "
            "deterministic-CUDA-supported; warn_only=True normally surfaces a "
            "warning then does NOT relax the restriction, causing per-epoch "
            "~36s slowdown. Setting this True explicitly disables determinism "
            "for the training step so the backward kernel chooses the fast path. "
            "Eval / inflate / archive bytes remain deterministic"
        ),
        "default": "true",
    },
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_substrate_z4_cooperative_receiver_loss",
        description=(
            "Train Z4 cooperative-receiver-loss substrate (Time-Traveler L5 "
            "staircase Step 2; Atick-Redlich 1990)."
        ),
    )
    p.add_argument("--video-path", type=Path, default=DEFAULT_VIDEO_PATH)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR)

    # Architecture
    p.add_argument("--latent-dim", type=int, default=24)
    p.add_argument("--decoder-embed-dim", type=int, default=32)
    p.add_argument("--decoder-num-upsample-blocks", type=int, default=6)

    # Cooperative-receiver Lagrangian weights
    p.add_argument("--lambda-pixel", type=float, default=0.0,
                   help="Pixel-MSE residual weight; 0=pure cooperative-receiver, 1=Z3 baseline")
    p.add_argument("--alpha-rate", type=float, default=25.0)
    p.add_argument("--beta-seg", type=float, default=100.0)
    p.add_argument("--gamma-pose", type=float, default=math.sqrt(10.0))
    p.add_argument("--pose-weight-scale", type=float, default=1.0,
                   help=(
                       "Opt-in pose marginal tilt. Default 1.0 preserves the "
                       "contest formula; PR106-derived 2.71x is experimental."
                   ))

    # Training
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.997)
    p.add_argument("--noise-std", type=float, default=0.5)

    # Mode flags
    p.add_argument("--smoke", action="store_true",
                   help="Run smoke path (tiny config, synthetic data, no scorer load)")
    p.add_argument("--full-cpu", action="store_true",
                   help="Opt-in to non-smoke CPU training (Catalog #197 paired flag required)")
    p.add_argument("--advisory-cpu-explicitly-waived", action="store_true",
                   help="Required sister flag for --full-cpu (Catalog #197)")
    p.add_argument("--enable-autocast-fp16", action="store_true",
                   help="Catalog #172; Tier-1 autocast(fp16) + GradScaler")
    p.add_argument("--enable-tf32", action="store_true", default=True,
                   help="Catalog #178; TF32 fast-math on Ampere/Hopper (default on)")
    p.add_argument("--enable-torch-compile", action="store_true",
                   help="Catalog #179; torch.compile (Inductor) on substrate")
    p.add_argument("--enable-gt-scorer-cache", action="store_true",
                   help="Catalog #228; F3 GTScorerCache (~50%% scorer compute savings)")
    p.add_argument("--relax-determinism-for-backward", action="store_true", default=True,
                   help="Phase B.2 root-cause fix; relax determinism around backward "
                        "(bicubic interp backward not deterministic-CUDA supported)")
    p.add_argument("--torch-compile-mode", type=str, default="default",
                   help="torch.compile mode (default | reduce-overhead | max-autotune)")
    p.add_argument("--gt-scorer-cache-chunk-size", type=int, default=16,
                   help="F3 cache chunk size for GT scorer pre-compute")
    p.add_argument("--skip-auth-eval", action="store_true",
                   help="Skip CUDA auth eval at end of full training (debug only)")
    p.add_argument("--auth-eval-skipped-reason", type=str, default="",
                   help="Optional reason for skipping auth eval (carried into stats).")
    return p


def _validate_full_cpu_flags(args: argparse.Namespace) -> None:
    """Catalog #197: --full-cpu MUST be paired with --advisory-cpu-explicitly-waived."""
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
    """Smoke entry: tiny config, synthetic data, ≤3 epochs, no scorer load."""
    torch.manual_seed(args.seed)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tiny smoke config
    num_pairs = 4
    cfg = CooperativeReceiverConfig(
        latent_dim=8,
        decoder_embed_dim=16,
        decoder_channels=(12, 10, 8, 6),
        decoder_num_upsample_blocks=4,
        num_pairs=num_pairs,
        output_height=48,
        output_width=64,
        cooperative_receiver_lambda_pixel=args.lambda_pixel,
    )
    substrate = CooperativeReceiverSubstrate(cfg).to(args.device)
    print(f"[z4-smoke] param breakdown: {substrate.num_parameters_breakdown()}")

    # Synthetic data (Catalog #114 allowed in smoke path only)
    synth_frame_1 = torch.rand(
        num_pairs, 3, cfg.output_height, cfg.output_width, device=args.device
    )
    synth_frame_0_target = torch.rand(
        num_pairs, 3, cfg.output_height, cfg.output_width, device=args.device
    )

    opt = torch.optim.AdamW(substrate.parameters(), lr=args.lr)
    losses = []
    for epoch in range(max(args.epochs, 3)):
        opt.zero_grad()
        idx = torch.arange(num_pairs, device=args.device, dtype=torch.long)
        rgb_0, rgb_1, _mu, _logvar = substrate(idx, frames_for_encoder=synth_frame_1)
        # Pixel-MSE proxy in smoke (no scorer load)
        recon_loss = (rgb_0 - synth_frame_0_target).pow(2).mean() + (rgb_1 - synth_frame_1).pow(2).mean()
        recon_loss.backward()
        torch.nn.utils.clip_grad_norm_(substrate.parameters(), args.grad_clip)
        opt.step()
        losses.append({"epoch": epoch, "loss": float(recon_loss.item())})

    # Pack archive
    enc_sd = substrate.encoder.state_dict()
    dec_sd = substrate.decoder.state_dict()
    latents = substrate.latents.detach().cpu()
    meta = {
        "encoder_input_channels": cfg.encoder_input_channels,
        "encoder_hidden_dim": cfg.encoder_hidden_dim,
        "decoder_embed_dim": cfg.decoder_embed_dim,
        "decoder_initial_grid_h": cfg.decoder_initial_grid_h,
        "decoder_initial_grid_w": cfg.decoder_initial_grid_w,
        "decoder_channels": list(cfg.decoder_channels),
        "decoder_num_upsample_blocks": cfg.decoder_num_upsample_blocks,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
        "latent_init_std": cfg.latent_init_std,
        "smoke": True,
    }
    archive_bytes = pack_archive(
        enc_sd, dec_sd, latents, meta,
        cooperative_receiver_lambda_pixel=args.lambda_pixel,
    )
    archive_path = out_dir / "0.bin"
    archive_path.write_bytes(archive_bytes)

    final = losses[-1] if losses else {"loss": float("inf")}
    stats = {
        "lane_id": SUBSTRATE_LANE_ID,
        "substrate_tag": SUBSTRATE_TAG,
        "smoke": True,
        "epochs": len(losses),
        "final_loss_proxy": final["loss"],
        "archive_bytes": len(archive_bytes),
        "lambda_pixel": args.lambda_pixel,
        "cfg": asdict(cfg),
        "score_claim_valid": False,
        "evidence_grade": "smoke-no-scorer",
        "ready_for_exact_eval_dispatch": False,
        "promotion_eligible": False,
        "param_breakdown": substrate.num_parameters_breakdown(),
    }
    (out_dir / "stats.json").write_text(
        json.dumps(stats, sort_keys=True, indent=2), encoding="utf-8"
    )
    print(
        f"[z4-smoke] OK final_loss={final['loss']:.6f} "
        f"archive={len(archive_bytes)}B lambda_pixel={args.lambda_pixel}"
    )
    return 0


# ---------------------------------------------------------------------------
# Full entry path — Phase 2 Council approved 2026-05-15 (11/11 unanimous LIFT)
# ---------------------------------------------------------------------------

def _build_archive_zip(zip_path: Path, *, bin_bytes: bytes) -> None:
    """Build deterministic archive.zip with single member ``0.bin``.

    Per Catalog #19 + HNeRV parity discipline lesson 3 + Catalog #146:
    monolithic single-file ``0.bin`` is the canonical contest archive grammar.
    """
    info = zipfile.ZipInfo(filename="0.bin", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(info, bin_bytes)


def _write_runtime(submission_dir: Path) -> None:
    """Emit contest-compliant inflate.sh + inflate.py + vendored Z4 substrate.

    Per Catalog #146: 3-positional-arg inflate.sh signature.
    Per Catalog #163: ``set -euo pipefail`` for fail-closed semantics.
    Per HNeRV parity discipline L4: ≤200 LOC inflate runtime budget.
    Per Catalog #205: device selection via canonical ``select_inflate_device``.
    """
    submission_dir.mkdir(parents=True, exist_ok=True)
    runtime_pkg = (
        submission_dir / "src" / "tac" / "substrates" / "z4_cooperative_receiver_loss"
    )
    runtime_pkg.mkdir(parents=True, exist_ok=True)
    for pkg_init in (
        submission_dir / "src" / "tac" / "__init__.py",
        submission_dir / "src" / "tac" / "substrates" / "__init__.py",
        runtime_pkg / "__init__.py",
    ):
        pkg_init.write_text("", encoding="utf-8")
    substrate_src = REPO_ROOT / "src" / "tac" / "substrates" / "z4_cooperative_receiver_loss"
    for name in ("architecture.py", "archive.py", "inflate.py"):
        shutil.copy2(substrate_src / name, runtime_pkg / name)
    (runtime_pkg / "__init__.py").write_text(
        "\"\"\"Z4 runtime package (inflate-time only — no scorer imports).\"\"\"\n"
        "from tac.substrates.z4_cooperative_receiver_loss.architecture import (\n"
        "    CooperativeReceiverConfig,\n"
        "    CooperativeReceiverSubstrate,\n"
        "    EVAL_HW,\n"
        "    NUM_PAIRS,\n"
        ")\n"
        "from tac.substrates.z4_cooperative_receiver_loss.archive import (\n"
        "    parse_archive,\n"
        ")\n"
        "from tac.substrates.z4_cooperative_receiver_loss.inflate import (\n"
        "    inflate_one_video,\n"
        ")\n"
        "__all__ = [\n"
        "    'CooperativeReceiverConfig', 'CooperativeReceiverSubstrate',\n"
        "    'EVAL_HW', 'NUM_PAIRS',\n"
        "    'parse_archive', 'inflate_one_video',\n"
        "]\n",
        encoding="utf-8",
    )
    _canon_vendor_shared_inflate_runtime(submission_dir, repo_root=REPO_ROOT)

    inflate_sh = (
        "#!/usr/bin/env bash\n"
        "# Z4 cooperative-receiver-loss contest-compliant inflate runtime.\n"
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
        '  if [ ! -f "${src}" ]; then src="${DATA_DIR}/0.bin"; fi\n'
        '  dst="${OUTPUT_DIR}/${base}.raw"\n'
        '  [ ! -f "${src}" ] && echo "ERROR: ${src} not found" >&2 && exit 1\n'
        '  "${PYTHON:-python3}" "${HERE}/inflate.py" "${DATA_DIR}" "${OUTPUT_DIR}" "${FILE_LIST}"\n'
        '  break\n'
        'done < "${FILE_LIST}"\n'
    )
    (submission_dir / "inflate.sh").write_text(inflate_sh, encoding="utf-8")
    (submission_dir / "inflate.sh").chmod(0o755)

    inflate_py = (
        '"""Z4 contest-compliant inflate.py — delegates to canonical inflate.main_cli."""\n'
        'from __future__ import annotations\n'
        'import sys\n'
        'from pathlib import Path\n'
        'HERE = Path(__file__).resolve().parent\n'
        'sys.path.insert(0, str(HERE / "src"))\n'
        'from tac.substrates.z4_cooperative_receiver_loss.inflate import main_cli\n'
        'if __name__ == "__main__":\n'
        '    sys.exit(main_cli())\n'
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


def _full_main(args: argparse.Namespace) -> int:
    """Z4 cooperative-receiver-loss full training (Phase 2 Council 11/11 LIFT 2026-05-15).

    Pipeline (Hotz-glue canonical-helpers, ~150 LOC; reuse trainer_skeleton +
    score_aware_common + smoke_auth_eval_gate; no reinvention):

    1. Pin seeds + canonical device gate (CUDA-required for promotion-grade).
    2. Patch upstream rgb_to_yuv6 globally (Catalog #187) BEFORE scorer load.
    3. Decode real contest pairs from upstream/videos/0.mkv via canonical
       ``decode_real_pairs`` (Catalog #114); shape (N, 2, 3, 384, 512) [0..255].
    4. ``load_differentiable_scorers`` for SegNet+PoseNet (no inflate-time load).
    5. Init Z4 substrate + EMA(0.997) + AdamW + cosine schedule + NaN watchdog.
    6. Train cooperative-receiver Lagrangian via ``CooperativeReceiverScoreAwareLoss``
       (routes through ``score_pair_components_dispatch`` per Catalog #164).
    7. Save EMA shadow at every val improvement (NEVER live weights — Catalog #88).
    8. Pack Z4CR1 archive + emit contest-compliant runtime tree.
    9. Run CUDA auth eval via canonical ``gate_auth_eval_call`` (Catalog #226).
    10. Continual-learning posterior update via canonical locked helper (Catalog #128).
    """
    from tac.differentiable_eval_roundtrip import (
        patch_upstream_yuv6_globally,
        unpatch_upstream_yuv6,
    )
    from tac.scorer import load_differentiable_scorers
    from tac.training import EMA

    _canon_pin_seeds(args.seed)
    # Phase B.2 root-cause fix: bicubic interp backward is NOT deterministic-CUDA-
    # supported; pin_seeds()'s warn_only=True surfaces a warning then bicubic
    # backward falls back to a slow path (~36s/epoch on T4). Relax determinism
    # for the training step here; eval / archive / inflate are still deterministic
    # (they don't fire bicubic backward).
    if getattr(args, "relax_determinism_for_backward", True):
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass
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
        print(f"[z4-full] {name} @ {msg['at']}")

    _stage("seed_pinned")

    # Stage 1: patch upstream rgb_to_yuv6 (Catalog #187) BEFORE scorer load.
    yuv6_token = patch_upstream_yuv6_globally()
    _stage("upstream_yuv6_patched")

    auth_eval_result_path: str | None = None
    auth_eval_score: float | None = None
    auth_eval_evidence_grade: str | None = None
    auth_eval_score_axis: str | None = None
    auth_eval_lane_tag: str | None = None
    auth_eval_score_claim_valid: bool = False
    auth_eval_exact_cuda_complete: bool = False
    archive_size: int = 0
    archive_sha: str = ""
    archive_zip_path = out_dir / "archive.zip"
    submission_dir = out_dir / "submission_dir"
    best_val_loss: float = math.inf
    best_epoch: int = -1
    train_started_at: float = time.time()

    try:
        # Stage 2: decode real pairs from contest video (Catalog #114).
        n_pairs_train = 600
        gt_pairs_btchw_255 = _canon_decode_real_pairs(
            args.video_path,
            n_pairs=n_pairs_train,
            substrate_tag=SUBSTRATE_TAG,
            repo_root=REPO_ROOT,
        ).to(device)
        n_pairs = int(gt_pairs_btchw_255.shape[0])
        _stage(f"video_decoded_{n_pairs}_pairs")

        # Stage 3: load differentiable scorers (Catalog #187 contract).
        # Returns (posenet, segnet) per Catalog #222 canonical loader contract.
        pose_scorer, seg_scorer = load_differentiable_scorers(
            upstream_dir=args.upstream_dir, device=device,
        )
        pose_scorer = pose_scorer.eval()
        seg_scorer = seg_scorer.eval()
        for p in pose_scorer.parameters():
            p.requires_grad_(False)
        for p in seg_scorer.parameters():
            p.requires_grad_(False)
        _stage("scorers_loaded")

        # Stage 4: init Z4 substrate + EMA + optimizer.
        cfg = CooperativeReceiverConfig(
            latent_dim=args.latent_dim,
            decoder_embed_dim=args.decoder_embed_dim,
            decoder_num_upsample_blocks=args.decoder_num_upsample_blocks,
            num_pairs=n_pairs,
            output_height=384,
            output_width=512,
            cooperative_receiver_lambda_pixel=args.lambda_pixel,
        )
        substrate = CooperativeReceiverSubstrate(cfg).to(device)
        param_breakdown = substrate.num_parameters_breakdown()
        _stage(f"substrate_built_{param_breakdown['total']}_params")
        ema = EMA(substrate, decay=args.ema_decay)
        optimizer = torch.optim.AdamW(
            substrate.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs)
        )

        # Stage 5: build score-aware loss module.
        loss_weights = CooperativeReceiverLossWeights(
            alpha_rate=args.alpha_rate,
            beta_seg=args.beta_seg,
            gamma_pose=args.gamma_pose,
            lambda_pixel=args.lambda_pixel,
        )
        score_loss = CooperativeReceiverScoreAwareLoss(
            seg_scorer=seg_scorer,
            pose_scorer=pose_scorer,
            weights=loss_weights,
        ).to(device)

        # Stage 5b: Tier-1 engineering primitives via canonical helper
        # (Catalogs #172/#178/#179/#228; Wunderkind E1 substitution 2026-05-15).
        # build_optimized_training_context returns:
        #   - gt_cache: F3 GTScorerCache (None when --enable-gt-scorer-cache=False)
        #   - substrate_model: torch.compile-wrapped model (passthrough when off)
        #   - autocast_cfg: AutocastConfig consumed by autocast_aware_forward
        opt_ctx = _canon_build_optimized_training_context(
            args,
            scorers=(pose_scorer, seg_scorer),
            gt_pairs=gt_pairs_btchw_255,
            substrate_model=substrate,
            device=device,
        )
        gt_cache = opt_ctx.gt_cache
        if gt_cache is not None:
            print(f"[z4-full] {gt_cache.summary_line()}")
            _stage("gt_scorer_cache_built")
        if opt_ctx.substrate_model is not None and opt_ctx.substrate_model is not substrate:
            substrate = opt_ctx.substrate_model  # type: ignore[assignment]
            _stage("torch_compile_wrapped")

        # Train/val split: hold out last 12.5% of pairs.
        val_count = max(1, n_pairs // 8)
        val_idx_start = n_pairs - val_count
        train_indices_pool = list(range(val_idx_start))
        val_indices_pool = list(range(val_idx_start, n_pairs))

        ckpt_best_path = out_dir / "best.pt"
        nan_strike = 0
        max_nan_strikes = 3
        # Archive byte proxy for rate term (matches packed Z4CR1 grammar order
        # of magnitude; refined post-Stage 8 with the actual archive bytes).
        archive_bytes_proxy = torch.tensor(120_000.0, device=device)

        # Tier-1 O2: autocast wrap (no-op when --enable-autocast-fp16=False).
        from tac.training_optimization import (
            autocast_aware_forward as _autocast_aware_forward,
        )

        for epoch in range(args.epochs):
            substrate.train()
            random.shuffle(train_indices_pool)
            epoch_losses: list[float] = []
            for batch_start in range(0, len(train_indices_pool), args.batch_size):
                batch_indices = train_indices_pool[
                    batch_start : batch_start + args.batch_size
                ]
                if not batch_indices:
                    continue
                pair_idxs = torch.tensor(
                    batch_indices, device=device, dtype=torch.long
                )
                gt_batch = gt_pairs_btchw_255[pair_idxs]  # (B, 2, 3, 384, 512)
                gt_rgb_0 = gt_batch[:, 0]
                gt_rgb_1 = gt_batch[:, 1]
                # Frame for encoder forensic provenance: use frame_1 normalized to [0, 1].
                frames_for_encoder = gt_rgb_1 / 255.0
                # F3 GTScorerCache lookup (per-pair-index batched). When
                # gt_cache is None (default-OFF), the kwargs stay None and
                # the loss path falls back to GT-forward (byte-faithful).
                gt_pose_batch = gt_seg_batch = None
                gt_seg_already_probs = None
                if gt_cache is not None:
                    gt_pose_batch, gt_seg_batch = gt_cache.lookup(
                        pair_idxs, device=device
                    )
                    gt_seg_already_probs = gt_cache.seg_already_probs
                with _autocast_aware_forward(
                    enabled=bool(getattr(args, "enable_autocast_fp16", False)),
                    dtype=torch.float16,
                    device=device,
                ):
                    rgb_0_unit, rgb_1_unit, _mu, _logvar = substrate(
                        pair_idxs, frames_for_encoder=frames_for_encoder
                    )
                    # Convert decoder unit-range output to [0, 255] domain for scorer.
                    rgb_0 = rgb_0_unit * 255.0
                    rgb_1 = rgb_1_unit * 255.0
                    loss, _parts = score_loss(
                        reconstructed_rgb_0=rgb_0,
                        reconstructed_rgb_1=rgb_1,
                        gt_rgb_0=gt_rgb_0,
                        gt_rgb_1=gt_rgb_1,
                        archive_bytes_proxy=archive_bytes_proxy,
                        apply_eval_roundtrip=True,
                        noise_std=args.noise_std,
                        gt_pose_batch=gt_pose_batch,
                        gt_seg_batch=gt_seg_batch,
                        gt_seg_already_probs=gt_seg_already_probs,
                    )
                if not torch.isfinite(loss):
                    nan_strike += 1
                    print(
                        f"[z4-full] NaN strike {nan_strike}/{max_nan_strikes} "
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
                    substrate.parameters(), args.grad_clip
                )
                optimizer.step()
                ema.update(substrate)
                epoch_losses.append(float(loss.detach().cpu()))
            scheduler.step()

            # Eval ~every epochs/20 OR at last epoch (Catalog #88: EMA shadow).
            if epoch % max(1, args.epochs // 20) == 0 or epoch == args.epochs - 1:
                live_state = {
                    k: v.detach().clone() for k, v in substrate.state_dict().items()
                }
                ema.apply(substrate)
                substrate.eval()
                try:
                    with torch.inference_mode():
                        val_idx = torch.tensor(
                            val_indices_pool, device=device, dtype=torch.long
                        )
                        gt_val = gt_pairs_btchw_255[val_idx]
                        rgb_0u, rgb_1u, _, _ = substrate(val_idx, frames_for_encoder=None)
                        rgb_0v = rgb_0u * 255.0
                        rgb_1v = rgb_1u * 255.0
                        pixel_val_loss = (
                            (rgb_0v - gt_val[:, 0]).pow(2).mean()
                            + (rgb_1v - gt_val[:, 1]).pow(2).mean()
                        ) / 2.0
                        val_gt_pose_batch = val_gt_seg_batch = None
                        val_gt_seg_already_probs = None
                        if gt_cache is not None:
                            val_gt_pose_batch, val_gt_seg_batch = gt_cache.lookup(
                                val_idx, device=device
                            )
                            val_gt_seg_already_probs = gt_cache.seg_already_probs
                        val_score_loss, val_parts = score_loss(
                            reconstructed_rgb_0=rgb_0v,
                            reconstructed_rgb_1=rgb_1v,
                            gt_rgb_0=gt_val[:, 0],
                            gt_rgb_1=gt_val[:, 1],
                            archive_bytes_proxy=archive_bytes_proxy,
                            apply_eval_roundtrip=True,
                            noise_std=0.0,
                            gt_pose_batch=val_gt_pose_batch,
                            gt_seg_batch=val_gt_seg_batch,
                            gt_seg_already_probs=val_gt_seg_already_probs,
                        )
                        val_loss = float(val_score_loss.detach().cpu())
                        pixel_val_loss_float = float(pixel_val_loss.detach().cpu())
                        val_seg_term = float(val_parts["seg_term"].detach().cpu())
                        val_pose_term = float(val_parts["pose_term"].detach().cpu())
                finally:
                    substrate.load_state_dict(live_state)
                    substrate.train()
                avg_train = (
                    sum(epoch_losses) / len(epoch_losses)
                    if epoch_losses else math.nan
                )
                print(
                    f"[z4-full] epoch {epoch:4d}: train_loss={avg_train:.5f} "
                    f"val_score_loss={val_loss:.5f} "
                    f"pixel_val_loss={pixel_val_loss_float:.5f} "
                    f"val_seg={val_seg_term:.6f} val_pose={val_pose_term:.6f} "
                    f"(best={best_val_loss:.5f})"
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    # Save EMA shadow (NEVER live weights — Catalog #88).
                    live_state2 = {
                        k: v.detach().clone()
                        for k, v in substrate.state_dict().items()
                    }
                    ema.apply(substrate)
                    ema_state = {
                        k: v.detach().cpu()
                        for k, v in substrate.state_dict().items()
                    }
                    substrate.load_state_dict(live_state2)
                    torch.save(
                        {
                            "state_dict": ema_state,
                            "config": asdict(cfg),
                            "epoch": epoch,
                            "val_loss": val_loss,
                            "val_loss_metric": "score_aware_cooperative_receiver_loss",
                            "pixel_val_loss": pixel_val_loss_float,
                            "val_seg_term": val_seg_term,
                            "val_pose_term": val_pose_term,
                        },
                        ckpt_best_path,
                    )

        train_elapsed = time.time() - train_started_at
        _stage(
            f"trained_best_epoch_{best_epoch}_val_loss_{best_val_loss:.5f}_"
            f"elapsed_{train_elapsed:.1f}s"
        )

        # Stage 8: load EMA-best checkpoint + pack Z4CR1 archive.
        if ckpt_best_path.is_file():
            best_ckpt = torch.load(
                ckpt_best_path, weights_only=False, map_location=device
            )
            substrate.load_state_dict(best_ckpt["state_dict"])
        substrate.eval()

        enc_sd = {k: v.detach().cpu() for k, v in substrate.encoder.state_dict().items()}
        dec_sd = {k: v.detach().cpu() for k, v in substrate.decoder.state_dict().items()}
        latents_cpu = substrate.latents.detach().cpu()
        meta = {
            "encoder_input_channels": cfg.encoder_input_channels,
            "encoder_hidden_dim": cfg.encoder_hidden_dim,
            "decoder_embed_dim": cfg.decoder_embed_dim,
            "decoder_initial_grid_h": cfg.decoder_initial_grid_h,
            "decoder_initial_grid_w": cfg.decoder_initial_grid_w,
            "decoder_channels": list(cfg.decoder_channels),
            "decoder_num_upsample_blocks": cfg.decoder_num_upsample_blocks,
            "output_height": cfg.output_height,
            "output_width": cfg.output_width,
            "latent_init_std": cfg.latent_init_std,
            "smoke": False,
        }
        archive_bytes = pack_archive(
            enc_sd, dec_sd, latents_cpu, meta,
            cooperative_receiver_lambda_pixel=args.lambda_pixel,
        )
        (out_dir / "0.bin").write_bytes(archive_bytes)
        archive_size = len(archive_bytes)
        archive_sha = _canon_sha256_bytes(archive_bytes)
        _build_archive_zip(archive_zip_path, bin_bytes=archive_bytes)
        submission_dir.mkdir(parents=True, exist_ok=True)
        _write_runtime(submission_dir)
        runtime_manifest = _write_submission_runtime_manifest(submission_dir)
        (submission_dir / "0.bin").write_bytes(archive_bytes)
        shutil.copy2(archive_zip_path, submission_dir / "archive.zip")
        _stage(
            f"archive_packed_{archive_size}_B_sha{archive_sha[:8]}_runtime_emitted"
        )

        # Stage 9: CUDA auth eval via canonical gate (Catalog #226).
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
            unpatch_upstream_yuv6(yuv6_token)
            _stage("upstream_yuv6_unpatched")

    # Stage 10: posterior update via canonical locked helper (Catalog #128).
    if not args.skip_auth_eval and (out_dir / "contest_auth_eval_cuda.json").exists():
        try:
            from tac.continual_learning import (
                posterior_update_locked_from_auth_eval_json,
            )

            update = posterior_update_locked_from_auth_eval_json(
                out_dir / "contest_auth_eval_cuda.json"
            )
            print(
                f"[z4-full] posterior_update accepted="
                f"{getattr(update, 'accepted', '?')}"
            )
            _stage("posterior_updated")
        except Exception as exc:
            print(f"[z4-full] WARN posterior_update failed: {exc!r}")

    # Cost-band anchor (only when valid CUDA score claim per Catalog #175/#177).
    if auth_eval_score_claim_valid:
        try:
            from tac.cost_band_calibration import CostBandAnchor, append_anchor

            wall_sec = float(time.time() - train_started_at)
            anchor = CostBandAnchor(
                logged_at_utc=_canon_utc_now_iso(),
                dispatch_label=f"{SUBSTRATE_LANE_ID}_{_canon_utc_now_iso()}",
                trainer="train_substrate_z4_cooperative_receiver_loss",
                platform="modal",
                gpu="T4" if device.type == "cuda" else "cpu",
                epochs=int(args.epochs),
                batch_size=int(args.batch_size),
                all_flags_on=False,
                actual_wall_clock_sec=wall_sec,
                actual_cost_usd=wall_sec / 3600.0 * 0.60,
                outcome="successful_dispatch",
                notes=(
                    f"substrate_tag={SUBSTRATE_TAG};"
                    f"archive_sha256={archive_sha};"
                    f"lambda_pixel={args.lambda_pixel};"
                    f"literature_anchor=atick_redlich_1990_cooperative_receiver;"
                    f"lane_class=substrate_engineering"
                ),
            )
            append_anchor(anchor)
            _stage("cost_band_anchor_appended")
        except Exception as exc:
            print(f"[z4-full] WARN cost_band_anchor_append failed: {exc!r}")

    # Result review blockers (Catalog #127 fail-closed custody).
    result_review_blockers: list[str] = [
        "trainer_stats_not_authoritative_score_claim_surface",
    ]
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
        env_var_candidates=("Z4_GPU", "MODAL_GPU"),
    )
    stats = {
        "lane_id": SUBSTRATE_LANE_ID,
        "substrate_tag": SUBSTRATE_TAG,
        "smoke": False,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "archive_bytes": archive_size,
        "archive_sha256": archive_sha,
        "archive_zip_bytes": (
            archive_zip_path.stat().st_size if archive_zip_path.exists() else 0
        ),
        "lambda_pixel": args.lambda_pixel,
        "alpha_rate": args.alpha_rate,
        "beta_seg": args.beta_seg,
        "gamma_pose": args.gamma_pose,
        "param_breakdown": param_breakdown,
        "predicted_delta_s_band": [-0.020, -0.005],
        "predicted_band_axis": "[contest-CPU 1to1] vs A1 0.1928",
        "council_verdict": (
            "PROCEED 11/11 unanimous LIFT (Phase 2 Council 2026-05-15; "
            "Hotz-glue canonical-helpers ~150 LOC; Contrarian+Fridrich "
            "binding lambda_pixel ablation)"
        ),
        "council_ledger_path": (
            "feedback_z4_z5_phase_2_council_deliberation_landed_20260515.md"
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
    print(
        f"[z4-full] DONE archive={archive_size}B sha={archive_sha[:12]}... "
        f"auth_score={auth_eval_score} grade={auth_eval_evidence_grade} "
        f"lambda_pixel={args.lambda_pixel}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _validate_full_cpu_flags(args)
    if args.smoke:
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":  # pragma: no cover — CLI entry
    sys.exit(main())
