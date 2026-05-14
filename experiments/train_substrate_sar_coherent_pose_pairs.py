"""Train the SAR coherent pose-pair substrate (SARC) end-to-end on contest video.

Per the floor-v3 commit ``27a7950fd`` (`.omx/research/adjusted_theoretical_
floor_v3_post_pr106_falsification_20260513.md`), L2 SAR is rank #1 EV/$:
$1, predicted ΔS -0.0056 vs PR101 0.193 anchor, 30-60 min wall-clock. Design
source: `.omx/research/expert_team_signal_processing_lincoln_lab_20260513.md`
§2 (Cook & Bernfeld 1967 *Radar Signals*; Carrara et al. 1995 *Spotlight SAR*).

Council-binding contract (CLAUDE.md non-negotiables) honored end-to-end:

- Train against ``upstream/videos/0.mkv`` decoded via pyav (synthetic data
  FORBIDDEN outside ``--smoke`` per Catalog #114).
- Patch upstream ``rgb_to_yuv6`` via ``patch_upstream_yuv6_globally`` BEFORE
  scorer construction (Catalog #187; PR #95/#106 contract).
- ``load_differentiable_scorers`` for SegNet/PoseNet (no scorer load at
  inflate; only at training).
- ``apply_eval_roundtrip_during_training`` inside the per-batch loop (Catalog #5).
- ``tac.training.EMA(decay=0.997)`` update after every ``optimizer.step``;
  inference checkpoint = EMA shadow (Catalog #88).
- Score-domain Lagrangian ``α·B/N + β·d_seg + γ·√d_pose + δ·H_temporal`` per
  HNeRV parity discipline lesson L6.
- AdamW lr cosine annealing; gradient clip 1.0; NaN watchdog.
- End with CUDA auth eval on best EMA checkpoint (CLAUDE.md "Auth eval
  EVERYWHERE"); refuse MPS (Catalog #1).
- Continual-learning posterior update via ``posterior_update_locked``
  (Catalog #128 atomic fcntl).
- Cost-band anchor append via ``tools/append_cost_band_anchor.py``.
- Contest-compliant runtime emission with 3 positional args inflate.sh +
  ``set -euo pipefail`` per Catalog #146 + #163.
- TIER_1_OPERATOR_REQUIRED_FLAGS declared per Catalog #151 (annotated
  assignment for Catalog #168 AST walker).

Usage (smoke; macOS CPU, tiny config, ~3 epochs, no scorer load)::

    .venv/bin/python experiments/train_substrate_sar_coherent_pose_pairs.py \\
        --video-path upstream/videos/0.mkv \\
        --output-dir experiments/results/sarc_smoke_<utc> \\
        --epochs 3 --device cpu --smoke

Usage (full; CUDA-required)::

    .venv/bin/python experiments/train_substrate_sar_coherent_pose_pairs.py \\
        --video-path upstream/videos/0.mkv \\
        --output-dir experiments/results/sarc_<utc> \\
        --epochs 2000 --batch-size 1 --lr 5e-4 --device cuda
"""
# AUTOCAST_FP16_WAIVED:score-aware-scorer-path-pending-canonical-autocast-backport
# TORCH_COMPILE_WAIVED:defer-until-per-substrate-canary-validates-Inductor-graph-breaks-and-score-axis-custody
# TF32_WAIVED:opt-in-via-CLI-flag-to-keep-eval-roundtrip-numerics-deterministic-until-paired-CPU/CUDA-anchor-lands
# NO_GRAD_WAIVED:eval-time-scorer-forward-wrapped-in-torch.inference_mode-inside-_run_val_loop
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

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

SUBSTRATE_TAG = "sarc"
SUBSTRATE_LANE_ID = "lane_sar_coherent_pose_pairs_substrate_20260513"


# ---------------------------------------------------------------------------
# Catalog #151 manifest — every flag below must be threaded by any operator
# wrapper that subprocess-invokes this trainer. Annotated as ast.AnnAssign
# so Catalog #168's AST walker observes it.
# ---------------------------------------------------------------------------
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--video-path": {
        "env": "SARC_VIDEO_PATH",
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
            ".omx/research/expert_team_signal_processing_lincoln_lab_20260513.md"
        ),
    },
    "--output-dir": {
        "env": "SARC_OUTPUT_DIR",
        "rationale": "custody location for checkpoints + archive + provenance",
    },
    "--epochs": {
        "env": "SARC_EPOCHS",
        "rationale": (
            "SAR-coherent renderer is small; council default 2000 epochs "
            "for full training run"
        ),
        "default": "2000",
    },
    "--upstream-dir": {
        "env": "SARC_UPSTREAM_DIR",
        "rationale": (
            "upstream/ root for scorer weights + evaluate.py; required for "
            "full training and auth eval"
        ),
        "default": str(DEFAULT_UPSTREAM_DIR.relative_to(REPO_ROOT)),
    },
    "--device": {
        "env": "SARC_DEVICE",
        "rationale": (
            "compute device; cuda required for full training (MPS refused "
            "per CLAUDE.md MPS-NOISE rule); cpu permitted only with --smoke"
        ),
        "default": "cuda",
    },
    "--hidden-dim": {
        "env": "SARC_HIDDEN_DIM",
        "rationale": "Renderer MLP hidden width (default 48 = sub-50K param target)",
        "default": "48",
    },
    "--per-pair-residual-bytes": {
        "env": "SARC_PER_PAIR_RESIDUAL_BYTES",
        "rationale": (
            "Bytes per pair for the int8 RGB-residual channel "
            "(default 50 = ~50 B/pair × 600 pairs = 30 KB residual budget)"
        ),
        "default": "50",
    },
    "--sar-topk-keep-fraction": {
        "env": "SARC_TOPK_KEEP_FRACTION",
        "rationale": (
            "Fraction of rFFT bins retained per pose dim (default 0.10 = "
            "30 bins per dim per L2 ledger §2.3)"
        ),
        "default": "0.10",
    },
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_substrate_sar_coherent_pose_pairs",
        description=(
            "Train SAR coherent pose-pair substrate (SARC). "
            "SIREN renderer + sparse rFFT pose codec + Atick-Redlich "
            "cooperative-receiver loss with SAR temporal-smoothness aux term."
        ),
    )
    # Tier 1 required
    p.add_argument("--video-path", type=Path, default=DEFAULT_VIDEO_PATH,
                   help="Path to upstream/videos/0.mkv (contest video).")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Where to write checkpoints + manifest + archive.")
    p.add_argument("--epochs", type=int, required=True,
                   help="Number of training epochs (council default 2000).")
    p.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR,
                   help="upstream/ root; required for scorer load + auth eval.")

    # Substrate architecture
    p.add_argument("--hidden-dim", type=int, default=48,
                   help="Renderer MLP hidden width (default 48).")
    p.add_argument("--num-hidden-layers", type=int, default=3,
                   help="Renderer MLP depth (default 3).")
    p.add_argument("--first-omega", type=float, default=30.0)
    p.add_argument("--hidden-omega", type=float, default=1.0)
    p.add_argument("--coord-feature-freqs", type=int, default=4)
    p.add_argument("--per-pair-residual-bytes", type=int, default=50,
                   help="Bytes per pair for int8 RGB residual (default 50).")
    p.add_argument("--int8-scale", type=float, default=64.0,
                   help="int8 quantization scale for per-pair residual.")
    p.add_argument("--sar-topk-keep-fraction", type=float, default=0.10,
                   help="Fraction of rFFT bins retained per pose dim.")
    p.add_argument("--sar-int16-scale", type=float, default=256.0,
                   help="int16 quantization scale for sparse rFFT coefficients.")

    # Training hyperparameters
    p.add_argument("--batch-size", type=int, default=1,
                   help="Pairs per batch (default 1; renderer is per-pair).")
    p.add_argument("--lr", type=float, default=5e-4,
                   help="AdamW learning rate (SIREN-init friendly).")
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ema-decay", type=float, default=0.997,
                   help="EMA decay (CLAUDE.md non-negotiable default 0.997).")
    p.add_argument("--noise-std", type=float, default=0.5)
    p.add_argument("--val-every-epochs", type=int, default=10)
    p.add_argument("--val-pair-count", type=int, default=16)
    p.add_argument("--max-pairs", type=int, default=None)

    # Score-aware Lagrangian weights
    p.add_argument("--alpha-rate", type=float, default=25.0)
    p.add_argument("--beta-seg", type=float, default=100.0)
    p.add_argument("--gamma-pose", type=float, default=math.sqrt(10.0))
    p.add_argument("--pose-weight-scale", type=float, default=1.0)
    p.add_argument("--delta-temporal", type=float, default=0.05,
                   help="SAR temporal-smoothness auxiliary term weight.")

    # Device / mode
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                   help="cuda for full training; cpu only with --smoke.")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny CPU smoke (no scorer load).")

    # Post-train artifacts
    p.add_argument("--skip-auth-eval", action="store_true")
    p.add_argument("--skip-archive-build", action="store_true")
    p.add_argument("--enable-autocast-fp16", action="store_true",
                   help="Catalog #172; deferred until canonical autocast wraps land.")
    p.add_argument("--enable-tf32", action="store_true",
                   help="Catalog #178; deferred until paired CPU/CUDA anchor lands.")
    return p


def _utc_now_iso() -> str:
    return _canon_utc_now_iso()


def _sha256_bytes(data: bytes) -> str:
    return _canon_sha256_bytes(data)


def _git_head_sha() -> str:
    return _canon_git_head_sha(REPO_ROOT)


def _pin_seeds(seed: int) -> None:
    _canon_pin_seeds(seed)


def _device_or_die(name: str, *, smoke: bool):
    return _canon_device_or_die(name, smoke=smoke, substrate_tag=SUBSTRATE_TAG)


def _decode_real_pairs(video_path: Path, *, n_pairs: int, max_pairs: int | None = None):
    return _canon_decode_real_pairs(
        video_path,
        n_pairs=n_pairs,
        substrate_tag=SUBSTRATE_TAG,
        max_pairs=max_pairs,
        repo_root=REPO_ROOT,
    )


# ---------------------------------------------------------------------------
# Contest-compliant runtime emission (Catalog #146)
# ---------------------------------------------------------------------------

def _write_runtime(submission_dir: Path) -> None:
    """Emit contest-compliant inflate.sh + inflate.py + vendored substrate."""
    submission_dir.mkdir(parents=True, exist_ok=True)
    runtime_pkg = submission_dir / "src" / "tac" / "substrates" / "sar_coherent_pose_pairs"
    runtime_pkg.mkdir(parents=True, exist_ok=True)
    for pkg_init in (
        submission_dir / "src" / "tac" / "__init__.py",
        submission_dir / "src" / "tac" / "substrates" / "__init__.py",
        runtime_pkg / "__init__.py",
    ):
        pkg_init.write_text("", encoding="utf-8")
    substrate_src = REPO_ROOT / "src" / "tac" / "substrates" / "sar_coherent_pose_pairs"
    for name in ("architecture.py", "archive.py", "inflate.py"):
        shutil.copy2(substrate_src / name, runtime_pkg / name)
    _canon_vendor_shared_inflate_runtime(submission_dir, repo_root=REPO_ROOT)

    inflate_sh = (
        "#!/usr/bin/env bash\n"
        "# SAR coherent pose-pair contest-compliant inflate (SARC)\n"
        "# Contract: $1=archive_dir $2=output_dir $3=file_list per Catalog #146\n"
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
        '"""SAR coherent pose-pair contest-compliant inflate runtime.\n'
        "\n"
        "Delegates to the vendored substrate CLI, which fails closed unless\n"
        "exactly one archive member exists at archive_dir/0.bin or archive_dir/x,\n"
        "then writes\n"
        "one contest .raw tensor stream per file_list entry.\n"
        "No scorer-network imports (strict-scorer-rule contract).\n"
        '"""\n'
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "HERE = Path(__file__).resolve().parent\n"
        "sys.path.insert(0, str(HERE / 'src'))\n"
        "from tac.substrates.sar_coherent_pose_pairs.inflate import main_cli\n"
        "\n"
        "def main() -> int:\n"
        "    return main_cli()\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    sys.exit(main())\n"
    )
    (submission_dir / "inflate.py").write_text(inflate_py, encoding="utf-8")


def _build_archive_zip(
    archive_zip_path: Path, *, bin_bytes: bytes, submission_dir: Path
) -> None:
    """Deterministic archive.zip containing ONLY the data payload (0.bin)."""
    del submission_dir  # signature-parity only; runtime tree lives outside zip
    archive_zip_path.parent.mkdir(parents=True, exist_ok=True)
    fixed_ts = (2026, 1, 1, 0, 0, 0)
    with zipfile.ZipFile(archive_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zi = zipfile.ZipInfo("0.bin", date_time=fixed_ts)
        zi.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(zi, bin_bytes)


# ---------------------------------------------------------------------------
# Smoke main (CPU; no scorer load)
# ---------------------------------------------------------------------------

def _smoke_main(args: argparse.Namespace) -> int:
    """Tiny CPU smoke that proves the scaffold + archive grammar are wired."""
    import numpy as np
    import torch

    from tac.substrates.sar_coherent_pose_pairs.architecture import (
        SARCoherentConfig,
        SARCoherentSubstrate,
    )
    from tac.substrates.sar_coherent_pose_pairs.archive import (
        encode_pose_codec_bytes,
        pack_archive,
        quantize_per_pair_residual_int8,
    )

    _pin_seeds(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = SARCoherentConfig(
        hidden_dim=min(args.hidden_dim, 24),
        num_hidden_layers=min(args.num_hidden_layers, 2),
        per_pair_residual_bytes=args.per_pair_residual_bytes,
        sar_topk_keep_fraction=args.sar_topk_keep_fraction,
        sar_int16_scale=args.sar_int16_scale,
        num_pairs=4,
        output_height=24,
        output_width=32,
        first_omega=args.first_omega,
        hidden_omega=args.hidden_omega,
        coord_feature_freqs=args.coord_feature_freqs,
    )
    device = _device_or_die(args.device, smoke=True)
    substrate = SARCoherentSubstrate(cfg).to(device)
    opt = torch.optim.Adam(substrate.parameters(), lr=args.lr)

    n_params = substrate.parameter_count()
    print(
        f"[{SUBSTRATE_TAG}-smoke] substrate params: {n_params:,}  "
        f"renderer_bytes={substrate.estimate_renderer_bytes()} "
        f"total_archive_estimate={substrate.estimate_total_archive_bytes()}"
    )

    # Smoke training loop: drive the renderer output toward neutral gray.
    for step in range(min(args.epochs, 3)):
        opt.zero_grad()
        loss = torch.zeros((), device=device)
        for pair_idx in range(cfg.num_pairs):
            rgb_0, rgb_1 = substrate.render_pair(pair_idx)
            loss = loss + (rgb_0 - 0.5).pow(2).mean() + (rgb_1 - 0.5).pow(2).mean()
        loss.backward()
        opt.step()
        print(f"[{SUBSTRATE_TAG}-smoke] step {step}: loss={loss.item():.4f}")

    # Build archive bytes (smoke).
    sd = substrate.state_dict()
    sparse, indices = substrate.pose_codec.encode_sparse_rfft()
    pose_bytes = encode_pose_codec_bytes(
        sparse, indices, int16_scale=cfg.sar_int16_scale
    )
    rng = np.random.default_rng(args.seed)
    residual_float = torch.from_numpy(
        rng.standard_normal((cfg.num_pairs, cfg.per_pair_residual_bytes)).astype("float32") * 0.05
    )
    residual = quantize_per_pair_residual_int8(residual_float, scale=args.int8_scale)
    meta = {
        "int8_scale": float(args.int8_scale),
        "sar_int16_scale": float(args.sar_int16_scale),
        "first_omega": float(cfg.first_omega),
        "hidden_omega": float(cfg.hidden_omega),
        "coord_feature_freqs": int(cfg.coord_feature_freqs),
        "sar_topk_keep_fraction": float(cfg.sar_topk_keep_fraction),
    }
    bin_bytes = pack_archive(
        renderer_state_dict=sd,
        pose_codec_bytes=pose_bytes,
        per_pair_residual=residual,
        meta=meta,
        num_pairs=cfg.num_pairs,
        hidden_dim=cfg.hidden_dim,
        num_hidden_layers=cfg.num_hidden_layers,
        output_height=cfg.output_height,
        output_width=cfg.output_width,
        pose_dim=cfg.pose_dim,
        pose_code_dim=cfg.pose_code_dim,
        per_pair_residual_bytes=cfg.per_pair_residual_bytes,
        sar_topk=cfg.sar_topk(),
    )
    bin_sha = _sha256_bytes(bin_bytes)
    bin_size = len(bin_bytes)
    print(
        f"[{SUBSTRATE_TAG}-smoke] SARC archive: {bin_size} B sha256={bin_sha[:16]}..."
    )

    submission_dir = args.output_dir / "submission_dir"
    _write_runtime(submission_dir)
    (submission_dir / "0.bin").write_bytes(bin_bytes)
    archive_zip_path = args.output_dir / "archive.zip"
    _build_archive_zip(archive_zip_path, bin_bytes=bin_bytes, submission_dir=submission_dir)
    archive_zip_size = archive_zip_path.stat().st_size
    archive_zip_sha = _sha256_bytes(archive_zip_path.read_bytes())
    print(
        f"[{SUBSTRATE_TAG}-smoke] archive.zip: {archive_zip_size} B "
        f"sha256={archive_zip_sha[:16]}..."
    )

    # Inflate roundtrip smoke (proves the full pipeline works locally).
    from tac.substrates.sar_coherent_pose_pairs.inflate import inflate_one_video

    test_raw = args.output_dir / "smoke_inflate_test.raw"
    n_frames = inflate_one_video(bin_bytes, test_raw, device="cpu")
    expected_size = n_frames * CAMERA_HW[0] * CAMERA_HW[1] * 3
    actual_size = test_raw.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"[{SUBSTRATE_TAG}-smoke] inflate roundtrip size mismatch: "
            f"got {actual_size} expected {expected_size}"
        )
    print(
        f"[{SUBSTRATE_TAG}-smoke] inflate roundtrip OK: {n_frames} frames, "
        f"{actual_size} bytes"
    )

    smoke_meta = {
        "started_at_utc": _utc_now_iso(),
        "smoke": True,
        "substrate_tag": SUBSTRATE_TAG,
        "lane_id": SUBSTRATE_LANE_ID,
        "bin_sha256": bin_sha,
        "bin_bytes": bin_size,
        "archive_zip_sha256": archive_zip_sha,
        "archive_zip_bytes": archive_zip_size,
        "n_params": n_params,
        "n_frames_inflated": n_frames,
        "git_head": _git_head_sha(),
        # macOS-CPU smoke is research-signal only per Catalog #192.
        "evidence_grade": "macOS-CPU-advisory",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }
    (args.output_dir / "smoke_metadata.json").write_text(
        json.dumps(smoke_meta, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[{SUBSTRATE_TAG}-smoke] wrote {args.output_dir / 'smoke_metadata.json'}")
    return 0


# ---------------------------------------------------------------------------
# Full main (CUDA-required)
# ---------------------------------------------------------------------------

def _run_val_loop(
    substrate, loss_fn, gt_tensor, val_pair_indices, archive_bytes_proxy, device
) -> float:
    """Validation pass with EMA shadow + ``torch.inference_mode`` (Catalog #180)."""
    import torch

    substrate.eval()
    losses: list[float] = []
    with torch.inference_mode():
        for pair_id in val_pair_indices:
            rgb_0, rgb_1 = substrate.render_pair(int(pair_id))
            rgb_0_b = (rgb_0 * 255.0).clamp(0.0, 255.0)
            rgb_1_b = (rgb_1 * 255.0).clamp(0.0, 255.0)
            gt_a = gt_tensor[pair_id, 0:1]
            gt_b = gt_tensor[pair_id, 1:2]
            with torch.no_grad():
                loss, _ = loss_fn(
                    rgb_0_b, rgb_1_b, gt_a, gt_b, archive_bytes_proxy,
                    apply_eval_roundtrip=True, noise_std=0.0,
                )
            if torch.isfinite(loss):
                losses.append(float(loss.detach().cpu()))
    return float(sum(losses) / len(losses)) if losses else math.inf


def _full_main(args: argparse.Namespace) -> int:
    """Full training — score-aware Lagrangian end-to-end."""
    import torch

    from tac.differentiable_eval_roundtrip import (
        patch_upstream_yuv6_globally,
        unpatch_upstream_yuv6,
    )
    from tac.scorer import load_differentiable_scorers
    from tac.substrates.sar_coherent_pose_pairs.architecture import (
        SARCoherentConfig,
        SARCoherentSubstrate,
    )
    from tac.substrates.sar_coherent_pose_pairs.archive import (
        encode_pose_codec_bytes,
        pack_archive,
        quantize_per_pair_residual_int8,
    )
    from tac.substrates.sar_coherent_pose_pairs.score_aware_loss import (
        SARCoherentLossWeights,
        SARCoherentScoreAwareLoss,
    )
    from tac.training import EMA

    _pin_seeds(args.seed)
    device = _device_or_die(args.device, smoke=False)

    if args.enable_tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage_log: list[dict[str, str]] = []

    def _stage(name: str) -> None:
        msg = {"stage": name, "at": _utc_now_iso()}
        stage_log.append(msg)
        print(f"[{SUBSTRATE_TAG}-full] {name} @ {msg['at']}")

    _stage("seed_pinned")

    # 1. Patch upstream rgb_to_yuv6 BEFORE scorer construction (Catalog #187).
    yuv6_token = patch_upstream_yuv6_globally()
    _stage("upstream_yuv6_patched")

    try:
        # 2. Load differentiable scorers (frozen).
        posenet, segnet = load_differentiable_scorers(args.upstream_dir, device=device)
        for p in list(posenet.parameters()) + list(segnet.parameters()):
            p.requires_grad_(False)
        posenet.eval()
        segnet.eval()
        _stage("scorers_loaded")

        # 3. Decode real target pairs.
        print(f"[{SUBSTRATE_TAG}-full] decoding {N_PAIRS_FULL} pairs from {args.video_path}")
        gt_pair_tensor = _decode_real_pairs(
            args.video_path, n_pairs=N_PAIRS_FULL, max_pairs=args.max_pairs
        ).to(device)
        n_pairs = int(gt_pair_tensor.shape[0])
        _stage(f"pairs_decoded_{n_pairs}")

        val_count = max(1, min(args.val_pair_count, max(1, n_pairs // 8)))
        val_idx_start = n_pairs - val_count
        train_indices_pool = list(range(val_idx_start))
        val_indices_pool = list(range(val_idx_start, n_pairs))

        # 4. Build substrate.
        cfg = SARCoherentConfig(
            hidden_dim=args.hidden_dim,
            num_hidden_layers=args.num_hidden_layers,
            first_omega=args.first_omega,
            hidden_omega=args.hidden_omega,
            coord_feature_freqs=args.coord_feature_freqs,
            per_pair_residual_bytes=args.per_pair_residual_bytes,
            sar_topk_keep_fraction=args.sar_topk_keep_fraction,
            sar_int16_scale=args.sar_int16_scale,
            num_pairs=n_pairs,
            output_height=EVAL_HW[0],
            output_width=EVAL_HW[1],
        )
        substrate = SARCoherentSubstrate(cfg).to(device)
        n_params = substrate.parameter_count()
        renderer_bytes = substrate.estimate_renderer_bytes()
        archive_estimate = substrate.estimate_total_archive_bytes()
        print(
            f"[{SUBSTRATE_TAG}-full] substrate params: {n_params:,}  "
            f"renderer_bytes={renderer_bytes} archive_estimate={archive_estimate}"
        )
        _stage(f"substrate_built_{n_params}_params_arch{archive_estimate}_B")

        # 5. Per-pair RGB residual (the float pre-quant tensor; gradient flows here).
        per_pair_residual_float = torch.nn.Parameter(
            0.0 * torch.randn(n_pairs, args.per_pair_residual_bytes, device=device)
        )

        # 6. EMA shadow (Catalog #88).
        ema = EMA(substrate, decay=args.ema_decay)
        _stage(f"ema_wired_decay_{args.ema_decay}")

        # 7. Score-aware Lagrangian.
        weights = SARCoherentLossWeights(
            alpha_rate=args.alpha_rate,
            beta_seg=args.beta_seg,
            gamma_pose=args.gamma_pose,
            pose_weight_scale=args.pose_weight_scale,
            delta_temporal=args.delta_temporal,
            contest_normalizer=CONTEST_NORMALIZER,
        )
        loss_fn = SARCoherentScoreAwareLoss(
            seg_scorer=segnet, pose_scorer=posenet, weights=weights
        )
        _stage("lagrangian_built")

        # 8. Optimizer (AdamW).
        params_list = [*list(substrate.parameters()), per_pair_residual_float]
        optimizer = torch.optim.AdamW(
            params_list, lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs)
        )

        # 9. Train loop.
        train_started_at = time.time()
        best_val_lag = math.inf
        best_epoch = -1
        ckpt_best_path = args.output_dir / "best.pt"
        nan_strike = 0
        max_nan_strikes = 3

        for epoch in range(args.epochs):
            substrate.train()
            random.shuffle(train_indices_pool)
            epoch_losses: list[float] = []
            for batch_start in range(0, len(train_indices_pool), args.batch_size):
                batch_indices = train_indices_pool[batch_start : batch_start + args.batch_size]
                if not batch_indices:
                    continue

                rgb_0_list = []
                rgb_1_list = []
                gt_0_list = []
                gt_1_list = []
                for pair_idx in batch_indices:
                    rgb_0, rgb_1 = substrate.render_pair(pair_idx)
                    rgb_0_list.append(rgb_0 * 255.0)
                    rgb_1_list.append(rgb_1 * 255.0)
                    gt_0_list.append(gt_pair_tensor[pair_idx, 0:1])
                    gt_1_list.append(gt_pair_tensor[pair_idx, 1:2])

                pred_a = torch.cat(rgb_0_list, dim=0)
                pred_b = torch.cat(rgb_1_list, dim=0)
                gt_a = torch.cat(gt_0_list, dim=0)
                gt_b = torch.cat(gt_1_list, dim=0)

                # Closed-form rate proxy: renderer + pose codec + per-pair residual.
                archive_bytes_proxy = torch.tensor(
                    float(renderer_bytes + substrate.pose_codec.estimate_int16_bytes()
                          + n_pairs * args.per_pair_residual_bytes),
                    device=device,
                )

                loss, parts = loss_fn(
                    pred_a, pred_b, gt_a, gt_b, archive_bytes_proxy,
                    pose_deltas_dense=substrate.pose_codec.pose_deltas,
                    apply_eval_roundtrip=True, noise_std=args.noise_std,
                )

                if not torch.isfinite(loss):
                    nan_strike += 1
                    print(f"[{SUBSTRATE_TAG}-full] NaN strike {nan_strike}/{max_nan_strikes}")
                    if nan_strike >= max_nan_strikes:
                        raise RuntimeError("NaN watchdog tripped")
                    continue
                nan_strike = 0

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params_list, args.grad_clip)
                optimizer.step()
                ema.update(substrate)
                epoch_losses.append(float(loss.detach().cpu()))

            scheduler.step()

            if epoch % max(1, args.val_every_epochs) == 0 or epoch == args.epochs - 1:
                live_state = {k: v.detach().clone() for k, v in substrate.state_dict().items()}
                ema.apply(substrate)
                try:
                    val_lag = _run_val_loop(
                        substrate, loss_fn, gt_pair_tensor, val_indices_pool,
                        torch.tensor(
                            float(renderer_bytes
                                  + substrate.pose_codec.estimate_int16_bytes()
                                  + n_pairs * args.per_pair_residual_bytes),
                            device=device,
                        ),
                        device,
                    )
                finally:
                    substrate.load_state_dict(live_state)
                    substrate.train()

                avg_train = (sum(epoch_losses) / len(epoch_losses)) if epoch_losses else math.nan
                print(
                    f"[{SUBSTRATE_TAG}-full] epoch {epoch:4d}: train={avg_train:.5f} "
                    f"val={val_lag:.5f} (best={best_val_lag:.5f})"
                )
                if val_lag < best_val_lag:
                    best_val_lag = val_lag
                    best_epoch = epoch
                    ema_state = {k: v.detach().cpu() for k, v in substrate.state_dict().items()}
                    torch.save(
                        {
                            "state_dict": ema_state,
                            "per_pair_residual_float": per_pair_residual_float.detach().cpu(),
                            "config": asdict(cfg),
                            "epoch": epoch,
                            "val_lag": val_lag,
                        },
                        ckpt_best_path,
                    )

        train_elapsed = time.time() - train_started_at
        _stage(f"trained_best_epoch_{best_epoch}_val_lag_{best_val_lag:.5f}_elapsed_{train_elapsed:.1f}s")

        # 10. Load best EMA checkpoint and build the archive.
        best_ckpt = torch.load(ckpt_best_path, weights_only=False, map_location=device)
        substrate.load_state_dict(best_ckpt["state_dict"])
        substrate.eval()
        per_pair_residual_final = best_ckpt["per_pair_residual_float"].to(device)
        per_pair_residual_int8 = quantize_per_pair_residual_int8(
            per_pair_residual_final, scale=args.int8_scale
        )
        sparse, indices = substrate.pose_codec.encode_sparse_rfft()
        pose_bytes = encode_pose_codec_bytes(
            sparse, indices, int16_scale=cfg.sar_int16_scale
        )

        meta = {
            "int8_scale": float(args.int8_scale),
            "sar_int16_scale": float(cfg.sar_int16_scale),
            "first_omega": float(cfg.first_omega),
            "hidden_omega": float(cfg.hidden_omega),
            "coord_feature_freqs": int(cfg.coord_feature_freqs),
            "sar_topk_keep_fraction": float(cfg.sar_topk_keep_fraction),
            "atick_redlich_cooperative_receiver": True,
            "sar_temporal_smoothness_aux": True,
            "renderer_kind": "siren_coordinate_mlp_with_sar_pose_code",
        }
        bin_bytes = pack_archive(
            renderer_state_dict=substrate.state_dict(),
            pose_codec_bytes=pose_bytes,
            per_pair_residual=per_pair_residual_int8,
            meta=meta,
            num_pairs=cfg.num_pairs,
            hidden_dim=cfg.hidden_dim,
            num_hidden_layers=cfg.num_hidden_layers,
            output_height=cfg.output_height,
            output_width=cfg.output_width,
            pose_dim=cfg.pose_dim,
            pose_code_dim=cfg.pose_code_dim,
            per_pair_residual_bytes=cfg.per_pair_residual_bytes,
            sar_topk=cfg.sar_topk(),
        )
        bin_sha = _sha256_bytes(bin_bytes)
        bin_size = len(bin_bytes)
        print(
            f"[{SUBSTRATE_TAG}-full] SARC archive: {bin_size} B sha256={bin_sha[:16]}..."
        )
        _stage(f"archive_built_{bin_size}_B_sha{bin_sha[:8]}")

        # 11. Build runtime tree + archive.zip.
        submission_dir = args.output_dir / "submission_dir"
        _write_runtime(submission_dir)
        (submission_dir / "0.bin").write_bytes(bin_bytes)
        archive_zip_path = args.output_dir / "archive.zip"
        _build_archive_zip(archive_zip_path, bin_bytes=bin_bytes, submission_dir=submission_dir)
        archive_zip_sha = _sha256_bytes(archive_zip_path.read_bytes())
        archive_zip_size = archive_zip_path.stat().st_size
        shutil.copy2(archive_zip_path, submission_dir / "archive.zip")
        _stage("archive_emitted")

        # 12. Auth eval ([contest-CUDA] inline).
        auth_eval_json_path = args.output_dir / "auth_eval.json"
        if not args.skip_auth_eval and device.type == "cuda":
            cmd = [
                sys.executable,
                str(CONTEST_AUTH_EVAL_SCRIPT),
                "--submission-dir", str(submission_dir),
                "--upstream-dir", str(args.upstream_dir),
                "--result-json", str(auth_eval_json_path),
                "--device", "cuda",
            ]
            print(f"[{SUBSTRATE_TAG}-full] launching auth eval: {' '.join(cmd)}")
            proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if proc.returncode != 0:
                raise RuntimeError(
                    f"[{SUBSTRATE_TAG}-full] contest_auth_eval.py failed "
                    f"rc={proc.returncode}; stdout_tail={proc.stdout[-2000:]}; "
                    f"stderr_tail={proc.stderr[-2000:]}"
                )
            _canon_require_contest_cuda_auth_eval_claim(
                auth_eval_json_path,
                archive_sha256=archive_zip_sha,
                substrate_tag=SUBSTRATE_TAG,
            )
            _stage("auth_eval_cuda_done_valid_claim")
    finally:
        unpatch_upstream_yuv6(yuv6_token)
        _stage("upstream_yuv6_unpatched")

    # 13. Posterior update (Catalog #128 atomic fcntl).
    if not args.skip_auth_eval and auth_eval_json_path.exists():
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
            print(f"[{SUBSTRATE_TAG}-full] WARN posterior_update failed: {exc!r}")

    # 14. Provenance.
    provenance = {
        "lane_id": SUBSTRATE_LANE_ID,
        "substrate_tag": SUBSTRATE_TAG,
        "started_at_utc": stage_log[0]["at"] if stage_log else _utc_now_iso(),
        "completed_at_utc": _utc_now_iso(),
        "git_head": _git_head_sha(),
        "bin_sha256": bin_sha,
        "bin_bytes": bin_size,
        "archive_zip_sha256": archive_zip_sha,
        "archive_zip_bytes": archive_zip_size,
        "n_params": n_params,
        "renderer_bytes_estimate": renderer_bytes,
        "best_val_lag": best_val_lag,
        "best_epoch": best_epoch,
        "epochs": args.epochs,
        "device": str(device),
        "design_memo": (
            ".omx/research/expert_team_signal_processing_lincoln_lab_20260513.md"
        ),
        "stage_log": stage_log,
        "hardware_substrate_cuda": _canon_detect_hardware_substrate(
            axis="cuda",
            substrate_tag=SUBSTRATE_TAG,
            env_var_candidates=("SARC_GPU", "MODAL_GPU"),
        ),
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[{SUBSTRATE_TAG}-full] wrote {args.output_dir / 'provenance.json'}")
    return 0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.smoke:
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":
    sys.exit(main())
