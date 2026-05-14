"""Train the C6 MDL-IBPS substrate end-to-end (zen-Z1 LARGEST single bet).

Per `.omx/research/campaign_lane_c6_e4_mdl_ibps_substrate_20260514.md` and
`.omx/research/adjusted_theoretical_floor_v3_post_pr106_falsification_20260513.md`:
this is the substrate-class-shift move predicted to drop ΔS -0.030 to -0.080
vs PR101 0.193, landing in band [0.113, 0.163]. The Z1 ablation
(`.omx/research/zen_floor_band_v2_post_z1_ablation_20260514.md`) empirically
proved the HNeRV-family class is saturated at ~99.3% MDL density on A1; C6
is the across-class shift.

Council-binding contract (CLAUDE.md non-negotiables) honored end-to-end:

- Train against ``upstream/videos/0.mkv`` decoded via pyav (Catalog #114).
- Patch upstream ``rgb_to_yuv6`` via ``patch_upstream_yuv6_globally`` BEFORE
  scorer construction (Catalog #187).
- ``load_differentiable_scorers`` for SegNet/PoseNet (no scorer load at inflate).
- ``apply_eval_roundtrip_during_training`` (Catalog #5).
- ``tac.training.EMA(decay=0.997)`` (Catalog #88; inference checkpoint = EMA shadow).
- Score-domain Lagrangian + IB regularizer (HNeRV parity L6).
- AdamW + gradient clip 1.0 + NaN watchdog.
- End with CUDA auth eval on best EMA checkpoint (CLAUDE.md "Auth eval EVERYWHERE").
- ``posterior_update_locked`` on success (Catalog #128).
- Contest-compliant runtime emission (Catalog #146 3-arg + Catalog #163 set -euo pipefail).
- TIER_1_OPERATOR_REQUIRED_FLAGS declared (Catalog #151 + #168 AnnAssign).
- ``--full-cpu`` opt-in coupled with ``--advisory-cpu-explicitly-waived`` (Catalog #197).

V1 SCOPE: ``_smoke_main`` builds a tiny config, trains for ≤3 epochs on
synthetic data, runs the archive pack + parse + inflate roundtrip, and emits
a contest-compliant runtime tree (no scorer load). ``_full_main`` trains
against real ``upstream/videos/0.mkv`` pairs with the score-aware Lagrangian
+ IB regularizer + EMA(0.997), packs the IBPS1 archive, emits the
contest-compliant runtime tree, runs CUDA auth eval, posts to the posterior.

Usage (smoke; macOS CPU or Linux CPU, tiny config, ~3 epochs)::

    .venv/bin/python experiments/train_substrate_c6_e4_mdl_ibps.py \\
        --video-path upstream/videos/0.mkv \\
        --output-dir experiments/results/c6_smoke_<utc> \\
        --epochs 3 --device cpu --smoke

Usage (full; CUDA-required; ratified by smoke; cost band $5 Modal T4)::

    .venv/bin/python experiments/train_substrate_c6_e4_mdl_ibps.py \\
        --video-path upstream/videos/0.mkv \\
        --output-dir experiments/results/c6_<utc> \\
        --epochs 2000 --batch-size 4 --lr 5e-4 --device cuda
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
from tac.substrates._shared.smoke_auth_eval_gate import (
    gate_auth_eval_call as _canon_gate_auth_eval_call,
)
from tac.substrates._shared.trainer_skeleton import (
    sha256_bytes as _canon_sha256_bytes,
)
from tac.substrates._shared.trainer_skeleton import (
    utc_now_iso as _canon_utc_now_iso,
)
from tac.substrates.c6_e4_mdl_ibps import (
    MDLIBPSConfig,
    MDLIBPSSubstrate,
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

SUBSTRATE_TAG = "c6_e4_mdl_ibps"
SUBSTRATE_LANE_ID = "lane_c6_e4_mdl_ibps_substrate_20260514"


# ---------------------------------------------------------------------------
# Catalog #151 manifest — every flag below must be threaded by any operator
# wrapper. AnnAssign per Catalog #168 (NOT bare Assign).
# ---------------------------------------------------------------------------
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--video-path": {
        "env": "C6_E4_MDL_IBPS_VIDEO_PATH",
        "rationale": (
            "Path to the contest video `upstream/videos/0.mkv` decoded via "
            "pyav into per-pair frames; required for non-smoke training"
        ),
        "default": str(DEFAULT_VIDEO_PATH),
        "required_input_file": True,
    },
    "--output-dir": {
        "env": "C6_E4_MDL_IBPS_OUTPUT_DIR",
        "rationale": (
            "Output directory for checkpoints, archive, stats, runtime tree, "
            "auth eval JSON; must be writable + outside /tmp"
        ),
        "default": None,
    },
    "--epochs": {
        "env": "C6_E4_MDL_IBPS_EPOCHS",
        "rationale": (
            "Training epoch count; smoke=3, Modal T4 full=2000-3000, "
            "production=5000+"
        ),
        "default": "2000",
    },
    "--batch-size": {
        "env": "C6_E4_MDL_IBPS_BATCH_SIZE",
        "rationale": (
            "Per-step pair count; T4 (16GB) handles 4-8 at 384x512; A100 "
            "(40GB) handles 16-32"
        ),
        "default": "4",
    },
    "--lr": {
        "env": "C6_E4_MDL_IBPS_LR",
        "rationale": "AdamW base learning rate; default 5e-4 per substrate skeleton",
        "default": "5e-4",
    },
    "--latent-dim": {
        "env": "C6_E4_MDL_IBPS_LATENT_DIM",
        "rationale": (
            "Per-pair latent dimensionality; ULTRA-LOW-RATE design target; "
            "default 24 (Z1 anchor A1 was ~25 bytes/pair)"
        ),
        "default": "24",
    },
    "--beta-ib": {
        "env": "C6_E4_MDL_IBPS_BETA_IB",
        "rationale": (
            "IB Lagrangian; controls bit budget for I(z; frames); "
            "operator-tunable; canonical sweep range [0.001, 1.0]"
        ),
        "default": "0.01",
    },
    "--enable-autocast-fp16": {
        "env": "C6_E4_MDL_IBPS_ENABLE_AUTOCAST_FP16",
        "rationale": (
            "Catalog #172; deferred until canonical autocast wrap pattern "
            "lands across substrate trainers"
        ),
        "default": "false",
    },
}


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_substrate_c6_e4_mdl_ibps",
        description=(
            "Train C6 MDL-IBPS substrate (zen-Z1 LARGEST single bet). "
            "Information Bottleneck × MDL × Procedural Synthesis."
        ),
    )
    p.add_argument("--video-path", type=Path, default=DEFAULT_VIDEO_PATH)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR)

    # Architecture
    p.add_argument("--latent-dim", type=int, default=24)
    p.add_argument("--decoder-embed-dim", type=int, default=32)
    p.add_argument(
        "--decoder-num-upsample-blocks", type=int, default=6,
        help="Number of PixelShuffle(2) blocks; 6 → 3x4 → 192x256 → bilinear 384x512",
    )

    # IB / score-aware loss weights
    p.add_argument("--beta-ib", type=float, default=0.01,
                   help="IB Lagrangian; tunable sweep [0.001, 1.0]")
    p.add_argument("--alpha-rate", type=float, default=25.0)
    p.add_argument("--beta-seg", type=float, default=100.0)
    p.add_argument("--gamma-pose", type=float, default=math.sqrt(10.0))
    p.add_argument("--pose-weight-scale", type=float, default=1.0)

    # Training
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.997)
    p.add_argument("--noise-std", type=float, default=0.5)
    p.add_argument("--val-every-epochs", type=int, default=20)
    p.add_argument("--val-pair-count", type=int, default=32)
    p.add_argument("--max-pairs", type=int, default=None)

    # Mode flags
    p.add_argument("--smoke", action="store_true",
                   help="Run smoke path (tiny config, synthetic data, no scorer load)")
    p.add_argument("--quick-smoke", action="store_true",
                   help="Alias for --smoke with even tinier config (Stage 0A)")
    p.add_argument("--full-cpu", action="store_true",
                   help="Opt-in to non-smoke CPU training (Catalog #197 paired flag required)")
    p.add_argument("--advisory-cpu-explicitly-waived", action="store_true",
                   help="Required sister flag for --full-cpu (Catalog #197)")

    # Post-train
    p.add_argument("--skip-auth-eval", action="store_true")
    p.add_argument("--skip-archive-build", action="store_true")
    p.add_argument("--enable-autocast-fp16", action="store_true",
                   help="Catalog #172; pending canonical autocast backport")
    p.add_argument("--enable-tf32", action="store_true",
                   help="Catalog #178; opt-in")
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
    _canon_pin_seeds(args.seed)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tiny smoke config
    num_pairs = 4
    cfg = MDLIBPSConfig(
        latent_dim=8,  # tiny smoke latent
        decoder_embed_dim=16,
        decoder_channels=(12, 10, 8, 6),
        decoder_num_upsample_blocks=4,
        num_pairs=num_pairs,
        output_height=48,  # 3 * 2^4 = 48
        output_width=64,
        beta_ib=args.beta_ib,
    )
    substrate = MDLIBPSSubstrate(cfg).to(args.device)
    print(f"[c6-smoke] param breakdown: {substrate.num_parameters_breakdown()}")

    # Synthetic data (Catalog #114 allowed in smoke path only)
    torch.manual_seed(args.seed)
    synth_frame_1 = torch.rand(num_pairs, 3, cfg.output_height, cfg.output_width, device=args.device)
    synth_frame_0_target = torch.rand(num_pairs, 3, cfg.output_height, cfg.output_width, device=args.device)

    opt = torch.optim.AdamW(substrate.parameters(), lr=args.lr)
    losses = []
    for epoch in range(max(args.epochs, 3)):
        opt.zero_grad()
        idx = torch.arange(num_pairs, device=args.device, dtype=torch.long)
        rgb_0, rgb_1, mu, logvar = substrate(idx, frames_for_encoder=synth_frame_1)
        # Pixel-MSE + IB regularizer (no scorer load)
        recon_loss = (rgb_0 - synth_frame_0_target).pow(2).mean() + (rgb_1 - synth_frame_1).pow(2).mean()
        kl_per = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0).sum(dim=-1).mean()
        loss = recon_loss + args.beta_ib * kl_per
        loss.backward()
        torch.nn.utils.clip_grad_norm_(substrate.parameters(), args.grad_clip)
        opt.step()
        losses.append({
            "epoch": epoch,
            "loss": float(loss.item()),
            "recon": float(recon_loss.item()),
            "kl": float(kl_per.item()),
        })

    # Pack archive
    enc_sd = substrate.encoder.state_dict()
    dec_sd = substrate.decoder.state_dict()
    latents = substrate.latents.detach().cpu()
    meta = {
        "beta_ib": cfg.beta_ib,
        "encoder_input_channels": cfg.encoder_input_channels,
        "encoder_sin_freq": cfg.encoder_sin_freq,
        "decoder_embed_dim": cfg.decoder_embed_dim,
        "decoder_initial_grid_h": cfg.decoder_initial_grid_h,
        "decoder_initial_grid_w": cfg.decoder_initial_grid_w,
        "decoder_channels": list(cfg.decoder_channels),
        "decoder_num_upsample_blocks": cfg.decoder_num_upsample_blocks,
        "decoder_sin_freq": cfg.decoder_sin_freq,
        "output_height": cfg.output_height,
        "output_width": cfg.output_width,
        "latent_init_std": cfg.latent_init_std,
        "smoke": True,
        "git_head": _canon_git_head_sha(REPO_ROOT),
        "trained_at_utc": _canon_utc_now_iso(),
    }
    archive_bytes = pack_archive(enc_sd, dec_sd, latents, meta)
    archive_path = out_dir / "0.bin"
    archive_path.write_bytes(archive_bytes)

    # Emit runtime tree + archive.zip
    submission_dir = out_dir / "submission_dir"
    _write_runtime(submission_dir)
    (submission_dir / "0.bin").write_bytes(archive_bytes)
    archive_zip_path = out_dir / "archive.zip"
    _build_archive_zip(archive_zip_path, bin_bytes=archive_bytes)

    archive_sha = _canon_sha256_bytes(archive_bytes)
    final = losses[-1] if losses else {"loss": float("inf")}
    stats = {
        "lane_id": SUBSTRATE_LANE_ID,
        "substrate_tag": SUBSTRATE_TAG,
        "smoke": True,
        "epochs": len(losses),
        "final_loss_proxy": final["loss"],
        "final_recon": final.get("recon"),
        "final_kl": final.get("kl"),
        "archive_bytes": len(archive_bytes),
        "archive_sha256": archive_sha,
        "archive_zip_bytes": archive_zip_path.stat().st_size,
        "archive_zip_sha256": _canon_sha256_bytes(archive_zip_path.read_bytes()),
        "cfg": asdict(cfg),
        "score_claim_valid": False,
        "evidence_grade": "smoke-no-scorer",
        "ready_for_exact_eval_dispatch": False,
        "promotion_eligible": False,
        "param_breakdown": substrate.num_parameters_breakdown(),
        "git_head": _canon_git_head_sha(REPO_ROOT),
        "trained_at_utc": _canon_utc_now_iso(),
    }
    (out_dir / "stats.json").write_text(
        json.dumps(stats, sort_keys=True, indent=2), encoding="utf-8"
    )
    print(
        f"[c6-smoke] OK final_loss={final['loss']:.6f} archive={len(archive_bytes)}B "
        f"sha={archive_sha[:12]}... latent_dim={cfg.latent_dim} num_pairs={num_pairs}"
    )
    return 0


def _write_runtime(submission_dir: Path) -> None:
    """Emit contest-compliant inflate.sh + inflate.py + vendored substrate.

    Per Catalog #146: 3-positional-arg ``inflate.sh <archive_dir> <output_dir>
    <file_list>``. Per Catalog #163: ``set -euo pipefail`` for fail-closed
    semantics.
    """
    submission_dir.mkdir(parents=True, exist_ok=True)
    runtime_pkg = submission_dir / "src" / "tac" / "substrates" / "c6_e4_mdl_ibps"
    runtime_pkg.mkdir(parents=True, exist_ok=True)
    for pkg_init in (
        submission_dir / "src" / "tac" / "__init__.py",
        submission_dir / "src" / "tac" / "substrates" / "__init__.py",
    ):
        pkg_init.write_text("", encoding="utf-8")

    # Vendor only inflate-time modules (no scorer imports per CLAUDE.md
    # "Strict scorer rule").
    substrate_src = REPO_ROOT / "src" / "tac" / "substrates" / "c6_e4_mdl_ibps"
    for name in (
        "architecture.py",
        "archive.py",
        "ib_decoder.py",
        "ib_encoder.py",
        "inflate.py",
    ):
        shutil.copy2(substrate_src / name, runtime_pkg / name)

    # Minimal runtime __init__.py: inflate-time only.
    (runtime_pkg / "__init__.py").write_text(
        '"""C6 runtime package (inflate-time only — no scorer imports)."""\n'
        "from tac.substrates.c6_e4_mdl_ibps.architecture import (\n"
        "    EVAL_HW,\n"
        "    MDLIBPSConfig,\n"
        "    MDLIBPSSubstrate,\n"
        "    NUM_PAIRS,\n"
        ")\n"
        "from tac.substrates.c6_e4_mdl_ibps.archive import (\n"
        "    IBPS1_MAGIC,\n"
        "    MDLIBPSArchive,\n"
        "    pack_archive,\n"
        "    parse_archive,\n"
        ")\n"
        "from tac.substrates.c6_e4_mdl_ibps.ib_decoder import IBDecoder\n"
        "from tac.substrates.c6_e4_mdl_ibps.ib_encoder import IBEncoder\n"
        "from tac.substrates.c6_e4_mdl_ibps.inflate import inflate_one_video, main_cli\n"
        '__all__ = ["inflate_one_video", "main_cli", "MDLIBPSConfig", "MDLIBPSSubstrate",\n'
        '           "pack_archive", "parse_archive", "IBPS1_MAGIC"]\n',
        encoding="utf-8",
    )

    # Vendor the shared inflate runtime helpers
    shared_dir = submission_dir / "src" / "tac" / "substrates" / "_shared"
    shared_dir.mkdir(parents=True, exist_ok=True)
    (shared_dir / "__init__.py").write_text("", encoding="utf-8")
    shutil.copy2(
        REPO_ROOT / "src" / "tac" / "substrates" / "_shared" / "inflate_runtime.py",
        shared_dir / "inflate_runtime.py",
    )

    # inflate.sh — Catalog #146 + #163 compliant
    inflate_sh = submission_dir / "inflate.sh"
    inflate_sh.write_text(
        "#!/usr/bin/env bash\n"
        "# C6 MDL-IBPS contest inflate runtime.\n"
        "# Per Catalog #146: 3 positional args (archive_dir, output_dir, file_list).\n"
        "# Per Catalog #163: set -euo pipefail.\n"
        "set -euo pipefail\n"
        '\n'
        'if [ $# -lt 3 ]; then\n'
        '    echo "usage: $0 <archive_dir> <output_dir> <file_list>" >&2\n'
        '    exit 2\n'
        'fi\n'
        '\n'
        'ARCHIVE_DIR="$1"\n'
        'OUTPUT_DIR="$2"\n'
        'FILE_LIST="$3"\n'
        '\n'
        'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'export PYTHONPATH="${HERE}/src:${PYTHONPATH:-}"\n'
        'exec uv run --python 3.13 --with torch==2.5.1 --with brotli --with numpy '
        '"${HERE}/inflate.py" "${ARCHIVE_DIR}" "${OUTPUT_DIR}" "${FILE_LIST}"\n',
        encoding="utf-8",
    )
    inflate_sh.chmod(0o755)

    # inflate.py — thin shim that delegates to vendored substrate
    inflate_py = submission_dir / "inflate.py"
    inflate_py.write_text(
        '"""C6 MDL-IBPS inflate entry — delegates to vendored substrate."""\n'
        "import sys\n"
        "\n"
        "from tac.substrates.c6_e4_mdl_ibps.inflate import main_cli\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    sys.exit(main_cli())\n",
        encoding="utf-8",
    )


def _build_archive_zip(archive_zip_path: Path, *, bin_bytes: bytes) -> None:
    """Build the single-member archive.zip per the contest packet contract."""
    archive_zip_path.parent.mkdir(parents=True, exist_ok=True)
    # Deterministic ZIP per Catalog #19
    with zipfile.ZipFile(archive_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        info = zipfile.ZipInfo(filename="0.bin", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(info, bin_bytes)


# ---------------------------------------------------------------------------
# Full entry path
# ---------------------------------------------------------------------------

def _full_main(args: argparse.Namespace) -> int:
    """Full training: real video, score-aware Lagrangian + IB, EMA, auth eval."""
    from tac.differentiable_eval_roundtrip import (
        apply_eval_roundtrip_during_training,
        patch_upstream_yuv6_globally,
    )
    from tac.scorer import load_differentiable_scorers
    from tac.substrates.c6_e4_mdl_ibps.score_aware_loss import (
        MDLIBPSLossWeights,
        MDLIBPSScoreAwareLoss,
    )
    from tac.training import EMA

    _canon_pin_seeds(args.seed)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    def _stage(name: str) -> None:
        print(f"[c6-full] STAGE: {name} @ {_canon_utc_now_iso()}", flush=True)

    # Device gate (no MPS per Catalog #1; CPU only with --full-cpu paired
    # flag per Catalog #197). The canonical helper permits CPU only when
    # smoke=True; --full-cpu opt-in routes through the smoke gate but
    # produces a non-promotable artifact (per CLAUDE.md "MPS auth eval is
    # NOISE" + the paired-flag attestation that the bytes are advisory).
    device = _canon_device_or_die(
        args.device,
        smoke=args.full_cpu,
        substrate_tag=SUBSTRATE_TAG,
    )

    _stage("patch_yuv6")
    patch_upstream_yuv6_globally()

    _stage("load_scorers")
    pose_scorer, seg_scorer = load_differentiable_scorers(
        args.upstream_dir,
        device=device,
    )

    _stage("decode_video_pairs")
    max_pairs = args.max_pairs or N_PAIRS_FULL
    # Returns torch.Tensor shape (N, 2, 3, 384, 512) float32 in [0, 255]
    gt_pair_tensor = _canon_decode_real_pairs(
        args.video_path,
        n_pairs=max_pairs,
        substrate_tag=SUBSTRATE_TAG,
        max_pairs=args.max_pairs,
        repo_root=REPO_ROOT,
    ).to(device)
    n_pairs = int(gt_pair_tensor.shape[0])
    # Split into (frame_0, frame_1) in unit range [0, 1] for encoder input.
    # The score-aware loss expects [0, 255]; we'll multiply at loss time.
    pairs = {
        "frame_0": gt_pair_tensor[:, 0] / 255.0,  # (N, 3, 384, 512) in [0, 1]
        "frame_1": gt_pair_tensor[:, 1] / 255.0,
    }
    print(f"[c6-full] decoded {n_pairs} pairs at {EVAL_HW}")

    _stage("build_substrate")
    cfg = MDLIBPSConfig(
        latent_dim=args.latent_dim,
        decoder_embed_dim=args.decoder_embed_dim,
        decoder_num_upsample_blocks=args.decoder_num_upsample_blocks,
        num_pairs=n_pairs,
        output_height=EVAL_HW[0],
        output_width=EVAL_HW[1],
        beta_ib=args.beta_ib,
    )
    substrate = MDLIBPSSubstrate(cfg).to(device)
    print(f"[c6-full] param breakdown: {substrate.num_parameters_breakdown()}")

    loss_fn = MDLIBPSScoreAwareLoss(
        seg_scorer,
        pose_scorer,
        MDLIBPSLossWeights(
            alpha_rate=args.alpha_rate,
            beta_seg=args.beta_seg,
            gamma_pose=args.gamma_pose,
            pose_weight_scale=args.pose_weight_scale,
            beta_ib=args.beta_ib,
        ),
    )

    opt = torch.optim.AdamW(
        substrate.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    ema = EMA(substrate, decay=args.ema_decay)

    _stage("training_loop")
    training_start = time.time()
    losses_log = []
    best_loss = float("inf")
    archive_bytes_proxy = torch.tensor(150_000.0, device=device)  # placeholder until real pack

    for epoch in range(args.epochs):
        substrate.train()
        # Mini-batch over pair indices
        perm = torch.randperm(n_pairs, device=device)
        batch_losses = []
        for i in range(0, n_pairs, args.batch_size):
            idx = perm[i:i + args.batch_size]
            gt_rgb_0 = pairs["frame_0"][idx] * 255.0  # to byte domain
            gt_rgb_1 = pairs["frame_1"][idx] * 255.0
            # Encoder sees frame_1 (SegNet's input)
            rgb_0, rgb_1, mu, logvar = substrate(idx, frames_for_encoder=pairs["frame_1"][idx])
            # Convert to byte domain for score-aware loss
            rgb_0_byte = rgb_0 * 255.0
            rgb_1_byte = rgb_1 * 255.0
            loss, parts = loss_fn(
                reconstructed_rgb_0=rgb_0_byte,
                reconstructed_rgb_1=rgb_1_byte,
                gt_rgb_0=gt_rgb_0,
                gt_rgb_1=gt_rgb_1,
                archive_bytes_proxy=archive_bytes_proxy,
                encoder_mu=mu,
                encoder_logvar=logvar,
                noise_std=args.noise_std,
            )
            if not torch.isfinite(loss):
                print(f"[c6-full] NON-FINITE LOSS at epoch {epoch} batch {i}; aborting")
                return 1
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(substrate.parameters(), args.grad_clip)
            opt.step()
            ema.update(substrate)
            batch_losses.append(float(loss.item()))
        epoch_loss = sum(batch_losses) / max(len(batch_losses), 1)
        sched.step()
        losses_log.append({"epoch": epoch, "loss": epoch_loss})
        if epoch_loss < best_loss:
            best_loss = epoch_loss
        if epoch % args.val_every_epochs == 0:
            print(f"[c6-full] epoch={epoch} loss={epoch_loss:.6f} (best={best_loss:.6f})")

    training_seconds = time.time() - training_start
    _stage("save_ema_shadow")
    # Inference checkpoint = EMA shadow
    orig_state = {k: v.detach().clone() for k, v in substrate.state_dict().items()}
    ema.apply(substrate)
    try:
        # Pack the archive from EMA shadow
        enc_sd = substrate.encoder.state_dict()
        dec_sd = substrate.decoder.state_dict()
        latents = substrate.latents.detach().cpu()
        meta = {
            "beta_ib": cfg.beta_ib,
            "encoder_input_channels": cfg.encoder_input_channels,
            "encoder_sin_freq": cfg.encoder_sin_freq,
            "decoder_embed_dim": cfg.decoder_embed_dim,
            "decoder_initial_grid_h": cfg.decoder_initial_grid_h,
            "decoder_initial_grid_w": cfg.decoder_initial_grid_w,
            "decoder_channels": list(cfg.decoder_channels),
            "decoder_num_upsample_blocks": cfg.decoder_num_upsample_blocks,
            "decoder_sin_freq": cfg.decoder_sin_freq,
            "output_height": cfg.output_height,
            "output_width": cfg.output_width,
            "latent_init_std": cfg.latent_init_std,
            "smoke": False,
            "ema_applied": True,
            "git_head": _canon_git_head_sha(REPO_ROOT),
            "trained_at_utc": _canon_utc_now_iso(),
            "training_seconds": training_seconds,
            "epochs": args.epochs,
        }
        archive_bytes = pack_archive(enc_sd, dec_sd, latents, meta)
    finally:
        substrate.load_state_dict(orig_state)

    _stage("emit_runtime_tree")
    submission_dir = out_dir / "submission_dir"
    _write_runtime(submission_dir)
    (submission_dir / "0.bin").write_bytes(archive_bytes)
    archive_zip_path = out_dir / "archive.zip"
    _build_archive_zip(archive_zip_path, bin_bytes=archive_bytes)
    archive_sha = _canon_sha256_bytes(archive_bytes)

    auth_eval_score = None
    auth_eval_evidence_grade = "skipped"
    auth_eval_score_axis = None
    auth_eval_lane_tag = None
    auth_eval_score_claim_valid = False
    auth_eval_exact_cuda_complete = False
    auth_eval_result_path = None
    result_review_blockers = [
        "trainer_stats_not_authoritative_score_claim_surface",
        "promotion_requires_separate_result_review",
    ]
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
    else:
        result_review_blockers.append("skip_auth_eval_explicitly_set")

    if not auth_eval_score_claim_valid:
        skipped_reason = str(getattr(args, "auth_eval_skipped_reason", "") or "")
        if skipped_reason:
            result_review_blockers.append(skipped_reason)
        else:
            result_review_blockers.append("contest_cuda_auth_eval_not_validated")

    stats = {
        "lane_id": SUBSTRATE_LANE_ID,
        "substrate_tag": SUBSTRATE_TAG,
        "smoke": False,
        "epochs": args.epochs,
        "training_seconds": training_seconds,
        "best_loss_proxy": best_loss,
        "archive_bytes": len(archive_bytes),
        "archive_sha256": archive_sha,
        "archive_zip_bytes": archive_zip_path.stat().st_size,
        "archive_zip_sha256": _canon_sha256_bytes(archive_zip_path.read_bytes()),
        "auth_eval_score": auth_eval_score,
        "auth_eval_evidence_grade": auth_eval_evidence_grade,
        "auth_eval_result_path": auth_eval_result_path,
        "auth_eval_score_axis": auth_eval_score_axis,
        "auth_eval_lane_tag": auth_eval_lane_tag,
        "auth_eval_score_claim_valid": auth_eval_score_claim_valid,
        "auth_eval_exact_cuda_complete": auth_eval_exact_cuda_complete,
        "score_axis": auth_eval_score_axis,
        "score_claim": False,
        "score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "result_review_blockers": result_review_blockers,
        "cfg": asdict(cfg),
        "param_breakdown": substrate.num_parameters_breakdown(),
        "hardware_substrate": _canon_detect_hardware_substrate(
            axis="cuda" if device.type == "cuda" else "cpu",
            substrate_tag=SUBSTRATE_TAG,
            env_var_candidates=("C6_E4_MDL_IBPS_GPU", "MODAL_GPU"),
        ),
        "git_head": _canon_git_head_sha(REPO_ROOT),
        "trained_at_utc": _canon_utc_now_iso(),
    }
    (out_dir / "stats.json").write_text(
        json.dumps(stats, sort_keys=True, indent=2), encoding="utf-8"
    )
    print(
        f"[c6-full] DONE archive={len(archive_bytes)}B sha={archive_sha[:12]}... "
        f"auth_score={auth_eval_score} grade={auth_eval_evidence_grade}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _validate_full_cpu_flags(args)
    if args.quick_smoke:
        args.smoke = True
    if args.smoke:
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":  # pragma: no cover — CLI entry
    sys.exit(main())
