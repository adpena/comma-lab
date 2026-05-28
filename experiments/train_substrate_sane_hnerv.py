# SPDX-License-Identifier: MIT
"""Train the sane_hnerv substrate end-to-end on contest video.

Operator-callable training script per the Fields-medal grand council substrate
design wave (2026-05-12). OD-SUBSTRATE-4 wires ``_full_main``; OD-SUBSTRATE-1
(Vast.ai 4090 first-anchor) + OD-SUBSTRATE-2 (Lightning T4 sweep) dispatch
through this entry point.

Council-binding contract (CLAUDE.md non-negotiables) honored end-to-end:

* Train against ``upstream/videos/0.mkv`` via pyav (synthetic FORBIDDEN
  outside ``--smoke`` per Catalog #114).
* Patch ``rgb_to_yuv6`` via ``patch_upstream_yuv6_globally`` BEFORE scorer
  construction (PR #95/#106).
* ``load_differentiable_scorers`` for SegNet/PoseNet (no scorer load at
  inflate).
* ``apply_eval_roundtrip=True`` in the per-batch loop (Catalog #5).
* ``tac.training.EMA(decay=0.997)`` after every ``optimizer.step``; inference
  checkpoint = EMA shadow (CLAUDE.md "EMA — NON-NEGOTIABLE").
* Score-domain Lagrangian ``alpha*B/N + beta*d_seg + gamma*sqrt(d_pose)``
  (HNeRV parity L6).
* AdamW + cosine LR; gradient clip 1.0; NaN watchdog (Council D).
* End with CUDA auth eval per "Auth eval EVERYWHERE"; refuse MPS (Catalog
  #1); CPU only with ``--smoke``.
* Continual-learning posterior via ``posterior_update_locked`` (Catalog
  #128).
* Cost-band anchor via ``tools/append_cost_band_anchor.py``.
* Contest-compliant runtime emission (3-arg ``inflate.sh`` + ``inflate.py``
  with NO scorer imports) per Catalog #146.
* ``TIER_1_OPERATOR_REQUIRED_FLAGS`` per Catalog #151.

Usage (smoke; CPU, no scorer load)::

    .venv/bin/python experiments/train_substrate_sane_hnerv.py \\
        --video-path upstream/videos/0.mkv \\
        --output-dir experiments/results/sane_hnerv_smoke_<utc> \\
        --epochs 10 --device cpu --smoke

Usage (full; CUDA-required; threaded from operator wrapper)::

    .venv/bin/python experiments/train_substrate_sane_hnerv.py \\
        --video-path upstream/videos/0.mkv \\
        --output-dir experiments/results/sane_hnerv_<utc> \\
        --epochs 2000 --batch-size 32 --lr 5e-4 --grad-clip 1.0 --device cuda

Wave N+45 BIND step (2026-05-28): trainer compressed from 1302 -> ~700 LOC
for HNeRV parity L12 ``trainer LOC <= 1000``; ``lane_class=substrate_engineering``
promoted to top-level lane registry field for L7. PR-95-parity bound packet.
"""
# AUTOCAST_FP16_WAIVED:score-aware-scorer-path-pending-canonical-autocast-backport
# TORCH_COMPILE_WAIVED:defer-until-per-substrate-canary-validates-Inductor-graph-breaks-and-score-axis-custody
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tac.substrate_registry import SubstrateContract, register_substrate
from tac.substrates._shared.smoke_auth_eval_gate import (
    gate_auth_eval_call as _canon_gate_auth_eval_call,
)
from tac.substrates._shared.trainer_skeleton import (
    build_optimized_training_context as _canon_build_optimized_training_context,
    decode_real_pairs as _canonical_decode_real_pairs,
    detect_hardware_substrate as _canon_detect_hardware_substrate,
    device_or_die as _canonical_device_or_die,
    git_head_sha as _git_head_sha,
    pin_seeds as _pin_seeds,
    sha256_bytes as _sha256_bytes,
    torch_version_string as _torch_version_string,
    utc_now_iso as _utc_now_iso,
    vendor_shared_inflate_runtime as _canon_vendor_shared_inflate_runtime,
)
from tac.training_optimization import (
    autocast_aware_forward as _autocast_aware_forward,
    compile_with_fallback as _compile_with_fallback,
)

_SUBSTRATE_TAG = "sane_hnerv"

# ---------------------------------------------------------------------------
# Module paths + constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO_PATH = REPO_ROOT / "upstream" / "videos" / "0.mkv"
DEFAULT_UPSTREAM_DIR = REPO_ROOT / "upstream"
DEFAULT_VIDEO_NAMES_FILE = REPO_ROOT / "upstream" / "public_test_video_names.txt"
CONTEST_AUTH_EVAL_SCRIPT = REPO_ROOT / "experiments" / "contest_auth_eval.py"
CONTINUAL_LEARNING_AVAILABLE = True  # imported lazily inside _full_main
COST_BAND_TOOL = REPO_ROOT / "tools" / "append_cost_band_anchor.py"

EVAL_HW = (384, 512)
CAMERA_HW = (874, 1164)
N_PAIRS_FULL = 600  # 1200 frames / 2 = non-overlapping pairs
CONTEST_NORMALIZER = 37_545_489.0  # contest evaluate.py N constant


# ---------------------------------------------------------------------------
# Catalog #151 manifest (TIER_1 flags). See council R1-R7 + CLAUDE.md #151.
# Required keys per entry: ``env``, ``rationale``.
# Optional keys: ``default``, ``satisfied_by_profile``, ``requires``,
# ``rationale_audit``, ``required_input_file``, ``generator_command``.
# ---------------------------------------------------------------------------
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--video-path": {
        "env": "SANE_HNERV_VIDEO_PATH",
        "rationale": "score-aware substrate MUST train against contest video (upstream/videos/0.mkv); synthetic FORBIDDEN outside --smoke",
        "default": str(DEFAULT_VIDEO_PATH.relative_to(REPO_ROOT)),
        "satisfied_by_profile": (),
        "requires": (),
        "required_input_file": True,
        "generator_command": "contest-pinned upstream snapshot — never regenerated locally",
        "rationale_audit": ".omx/research/grand_council_fields_medal_substrate_design_20260512.md#13-lessons-L1",
    },
    "--output-dir": {
        "env": "SANE_HNERV_OUTPUT_DIR",
        "rationale": "custody location for checkpoints + archive + provenance",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--epochs": {
        "env": "SANE_HNERV_EPOCHS",
        "rationale": "substrate engineering pass; under-training silently regresses (council target: 2000)",
        "default": "2000",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--upstream-dir": {
        "env": "SANE_HNERV_UPSTREAM_DIR",
        "rationale": "upstream/ root for scorer weights + evaluate.py; required for full training + auth eval",
        "default": str(DEFAULT_UPSTREAM_DIR.relative_to(REPO_ROOT)),
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--device": {
        "env": "SANE_HNERV_DEVICE",
        "rationale": "compute device; cuda required for full (MPS refused per CLAUDE.md MPS-NOISE); cpu only with --smoke",
        "default": "cuda",
        "satisfied_by_profile": (),
        "requires": (),
    },
}


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_substrate_sane_hnerv",
        description="Train sane_hnerv substrate end-to-end (OD-SUBSTRATE-4 wired).",
    )
    # TIER_1 required
    p.add_argument("--video-path", type=Path, default=DEFAULT_VIDEO_PATH,
                   help="Path to upstream/videos/0.mkv (contest video; non-smoke required).")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Where to write checkpoints + manifest + archive.")
    p.add_argument("--epochs", type=int, required=True,
                   help="Number of training epochs (council default 2000 for full).")
    p.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR,
                   help="upstream/ root; required for scorer load + auth eval.")
    # Training hyperparameters
    p.add_argument("--batch-size", type=int, default=32, help="Pair indices per batch (council default 32).")
    p.add_argument("--lr", type=float, default=5e-4, help="AdamW learning rate.")
    p.add_argument("--weight-decay", type=float, default=1e-5, help="AdamW weight decay.")
    p.add_argument("--grad-clip", type=float, default=1.0, help="Gradient clip norm (Council D pattern).")
    p.add_argument("--seed", type=int, default=0, help="Deterministic seed for torch/numpy/random.")
    # Substrate architecture knobs
    p.add_argument("--latent-dim", type=int, default=28, help="Per-pair latent dimensionality (council default 28).")
    p.add_argument("--sin-frequency", type=float, default=30.0, help="SIREN sin activation frequency (NeRF default).")
    # Lagrangian weights (score-aware)
    p.add_argument("--alpha-rate", type=float, default=25.0, help="Rate-term coefficient (contest: 25.0).")
    p.add_argument("--beta-seg", type=float, default=100.0, help="SegNet distortion coefficient (contest: 100.0).")
    p.add_argument("--gamma-pose", type=float, default=math.sqrt(10.0),
                   help="PoseNet sqrt-term coefficient (contest: sqrt(10)).")
    p.add_argument("--pose-weight-scale", type=float, default=1.0,
                   help="Operating-point multiplier on top of sqrt(10) (default 1.0).")
    p.add_argument("--noise-std", type=float, default=0.5, help="STE noise std for eval-roundtrip (Hotz fix).")
    # EMA + scheduling
    p.add_argument("--ema-decay", type=float, default=0.997,
                   help="EMA decay (CLAUDE.md non-negotiable default 0.997 for weights).")
    p.add_argument("--val-every-epochs", type=int, default=10,
                   help="Run held-out proxy eval every N epochs.")
    p.add_argument("--val-pair-count", type=int, default=32, help="Pairs reserved for held-out proxy validation.")
    # Device / mode
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                   help="Compute device. 'cpu' permitted only with --smoke; mps rejected at parse time.")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny CPU smoke (no scorer load); never use this output for ranking.")
    p.add_argument("--max-pairs", type=int, default=None,
                   help="Cap on pairs decoded from video (debug only). Default = all 600 pairs.")
    # Post-train artifacts
    p.add_argument("--skip-auth-eval", action="store_true",
                   help="Skip final auth-eval subprocess (research dispatches only).")
    p.add_argument("--skip-archive-build", action="store_true",
                   help="Skip building archive.zip (trainer-only smoke).")
    # Tier-1 optimization CLI (TIER-1-OPT-BATCH 2026-05-14). Opt-in flags.
    p.add_argument("--enable-autocast-fp16", action="store_true", default=False,
                   help="FP16 autocast (1.5-2x on Ampere/Hopper). Catalog #172. CPU forbidden.")
    p.add_argument("--enable-torch-compile", action="store_true", default=False,
                   help="torch.compile/Inductor wrap (1.5-2x on A100+). Catalog #179. Falls back on error.")
    p.add_argument("--enable-gt-scorer-cache", action="store_true", default=False,
                   help="RESERVED: pre-compute GT scorer outputs once (~50%% savings); pending API extension.")
    return p


# ---------------------------------------------------------------------------
# Video decode + Lagrangian helpers
# ---------------------------------------------------------------------------

def _decode_real_pairs(video_path: Path, *, n_pairs: int, max_pairs: int | None = None):
    """Adapter: forward to canonical helper with our substrate tag."""
    return _canonical_decode_real_pairs(
        video_path, n_pairs=n_pairs, substrate_tag=_SUBSTRATE_TAG,
        max_pairs=max_pairs, repo_root=REPO_ROOT,
    )


def _archive_bytes_proxy_closed_form(model):
    """Closed-form upper-bound on archive bytes for the rate term.

    Per Ballé 2018 the rate term enters the Lagrangian via entropy of the
    quantized latents + encoded decoder weights. Proxy: fp16 weights (2B/elem) +
    int16 latents (2B/elem). Constant during training; gradient flows entirely
    through seg + pose terms. Phase 2 lane: replace with Ballé hyperprior.
    """
    import torch
    n_decoder = sum(p.numel() for n, p in model.named_parameters() if n != "latents")
    n_latent_elems = model.latents.numel()
    bytes_proxy = float(n_decoder * 2 + n_latent_elems * 2)
    device = next(model.parameters()).device
    return torch.tensor(bytes_proxy, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Contest-compliant runtime emission (Catalog #146 contract)
# ---------------------------------------------------------------------------

_INFLATE_SH_TEMPLATE = (
    "#!/usr/bin/env bash\n"
    "# sane_hnerv contest-compliant inflate (OD-SUBSTRATE-4 wired 2026-05-12)\n"
    "# Contract: $1=archive_dir $2=output_dir $3=file_list\n"
    "set -euo pipefail\n"
    'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
    'DATA_DIR="$1"\nOUTPUT_DIR="$2"\nFILE_LIST="$3"\n'
    'mkdir -p "$OUTPUT_DIR"\n'
    'exec "${PYTHON:-python3}" "$HERE/inflate.py" "$DATA_DIR" "$OUTPUT_DIR" "$FILE_LIST"\n'
)

_INFLATE_PY_TEMPLATE = (
    "#!/usr/bin/env python\n"
    '"""sane_hnerv contest-compliant inflate runtime.\n\n'
    "Reads archive_dir/0.bin via packaged substrate parser; writes one contest\n"
    ".raw stream per file_list entry. No scorer-network imports.\n"
    '"""\n'
    "import sys\nfrom pathlib import Path\n\n"
    "HERE = Path(__file__).resolve().parent\n"
    "sys.path.insert(0, str(HERE / 'src'))\n"
    "from tac.substrates.sane_hnerv.inflate import inflate_one_video, raw_output_path, select_inflate_device\n\n"
    "def main() -> int:\n"
    "    if len(sys.argv) != 4:\n"
    "        print('usage: inflate.py <archive_dir> <output_dir> <file_list>', file=sys.stderr)\n"
    "        return 2\n"
    "    archive_dir = Path(sys.argv[1])\n"
    "    output_dir = Path(sys.argv[2])\n"
    "    file_list_path = Path(sys.argv[3])\n"
    "    archive_bytes = (archive_dir / '0.bin').read_bytes()\n"
    "    device = select_inflate_device()\n"
    "    for line in file_list_path.read_text(encoding='utf-8').splitlines():\n"
    "        line = line.strip()\n"
    "        if not line:\n"
    "            continue\n"
    "        inflate_one_video(archive_bytes, raw_output_path(output_dir, line), device=device)\n"
    "    return 0\n\n"
    "if __name__ == '__main__':\n"
    "    sys.exit(main())\n"
)


def _write_runtime(submission_dir: Path) -> None:
    """Emit contest-compliant ``inflate.sh`` + ``inflate.py`` pair (Catalog #146).

    Vendors substrate package into ``src/tac/substrates/sane_hnerv/`` so the
    runtime is self-contained (PYTHONPATH self-containment per Catalog #295).
    """
    submission_dir.mkdir(parents=True, exist_ok=True)
    runtime_pkg = submission_dir / "src" / "tac" / "substrates" / "sane_hnerv"
    runtime_pkg.mkdir(parents=True, exist_ok=True)
    for pkg_init in (
        submission_dir / "src" / "tac" / "__init__.py",
        submission_dir / "src" / "tac" / "substrates" / "__init__.py",
        runtime_pkg / "__init__.py",
    ):
        pkg_init.write_text("", encoding="utf-8")
    substrate_src = REPO_ROOT / "src" / "tac" / "substrates" / "sane_hnerv"
    for name in ("architecture.py", "archive.py", "inflate.py"):
        shutil.copy2(substrate_src / name, runtime_pkg / name)
    _canon_vendor_shared_inflate_runtime(submission_dir, repo_root=REPO_ROOT)
    (submission_dir / "inflate.sh").write_text(_INFLATE_SH_TEMPLATE, encoding="utf-8")
    (submission_dir / "inflate.sh").chmod(0o755)
    (submission_dir / "inflate.py").write_text(_INFLATE_PY_TEMPLATE, encoding="utf-8")


def _build_archive_zip(archive_zip_path: Path, *, bin_bytes: bytes, submission_dir: Path) -> None:
    """Deterministic ``archive.zip`` of 0.bin + inflate.sh + inflate.py + runtime tree.

    Per Catalog #19 deterministic-zip: ZipInfo + writestr + fixed timestamp.
    """
    archive_zip_path.parent.mkdir(parents=True, exist_ok=True)
    fixed_ts = (2026, 1, 1, 0, 0, 0)
    with zipfile.ZipFile(archive_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zi = zipfile.ZipInfo("0.bin", date_time=fixed_ts)
        zi.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(zi, bin_bytes)
        for name in ("inflate.sh", "inflate.py"):
            src = submission_dir / name
            if not src.is_file():
                continue
            zi = zipfile.ZipInfo(name, date_time=fixed_ts)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(zi, src.read_bytes())
        runtime_root = submission_dir / "src"
        if runtime_root.is_dir():
            for src in sorted(runtime_root.rglob("*.py")):
                rel = src.relative_to(submission_dir).as_posix()
                zi = zipfile.ZipInfo(rel, date_time=fixed_ts)
                zi.compress_type = zipfile.ZIP_DEFLATED
                zf.writestr(zi, src.read_bytes())


# ---------------------------------------------------------------------------
# Device guard adapter
# ---------------------------------------------------------------------------

def _device_or_die(name: str, *, smoke: bool):
    return _canonical_device_or_die(name, smoke=smoke, substrate_tag=_SUBSTRATE_TAG)


# ---------------------------------------------------------------------------
# Smoke main (CPU; no scorer load)
# ---------------------------------------------------------------------------

def _smoke_main(args: argparse.Namespace) -> int:
    """Tiny CPU smoke that proves the scaffold is wired (no scorer load)."""
    import torch
    from tac.substrates.sane_hnerv.architecture import SaneHnervConfig, SaneHnervSubstrate

    _pin_seeds(args.seed)
    cfg = SaneHnervConfig(
        latent_dim=args.latent_dim, embed_dim=64,
        initial_grid_h=3, initial_grid_w=4,
        decoder_channels=(32, 16, 8), sin_frequency=args.sin_frequency,
        num_pairs=4, output_height=24, output_width=32, num_upsample_blocks=2,
    )
    device = _device_or_die(args.device, smoke=True)
    model = SaneHnervSubstrate(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[smoke] sane_hnerv params: {model.num_parameters():,}")
    for step in range(min(args.epochs, 3)):
        idx = torch.arange(cfg.num_pairs, device=device, dtype=torch.long)
        rgb_0, rgb_1 = model(idx)
        loss = rgb_0.abs().mean() + rgb_1.abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"[smoke] step {step}: loss={loss.item():.4f}")
    ckpt = {"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "config": asdict(cfg), "smoke": True}
    ckpt_path = args.output_dir / "smoke_checkpoint.pt"
    torch.save(ckpt, ckpt_path)
    print(f"[smoke] wrote {ckpt_path}")
    return 0


# ---------------------------------------------------------------------------
# Full main (CUDA-required; score-aware Lagrangian end-to-end)
# ---------------------------------------------------------------------------

def _full_main(args: argparse.Namespace) -> int:
    """Full training entry point — CUDA + score-aware scorers.

    OPERATOR-GATED. Wrapper (Vast.ai / Lightning / Modal) threads all TIER_1
    flags + runs auth-eval per "Auth eval EVERYWHERE" + "Submission auth
    eval — BOTH CPU AND CUDA". The `--device cuda` CLI literal is threaded
    by Catalog #226 helper invocation below.
    """
    import torch
    from tac.differentiable_eval_roundtrip import (
        patch_upstream_yuv6_globally, unpatch_upstream_yuv6,
    )
    from tac.scorer import load_differentiable_scorers
    from tac.substrates.sane_hnerv.architecture import SaneHnervConfig, SaneHnervSubstrate
    from tac.substrates.sane_hnerv.archive import pack_archive
    from tac.substrates.sane_hnerv.score_aware_loss import (
        SaneHnervScoreAwareLoss, ScoreAwareLossWeights,
    )
    from tac.training import EMA

    # 1. Pin seeds + select device
    _pin_seeds(args.seed)
    device = _device_or_die(args.device, smoke=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage_log: list[dict[str, Any]] = []

    def _stage(name: str) -> None:
        stage_log.append({"stage": name, "at": _utc_now_iso()})

    _stage("seed_pinned")

    # 2. Patch upstream rgb_to_yuv6 BEFORE scorer construction (PR #95/#106)
    yuv6_token = patch_upstream_yuv6_globally()
    _stage("upstream_yuv6_patched")

    try:
        # 3. Load differentiable scorers
        posenet, segnet = load_differentiable_scorers(args.upstream_dir, device=device)
        for p in list(posenet.parameters()) + list(segnet.parameters()):
            p.requires_grad_(False)
        posenet.eval()
        segnet.eval()
        _stage("scorers_loaded")

        # 4. Decode real frame pairs (NOT synthetic)
        print(f"[full] decoding pairs from {args.video_path} ...")
        pair_tensor = _decode_real_pairs(args.video_path, n_pairs=N_PAIRS_FULL, max_pairs=args.max_pairs)
        n_pairs = int(pair_tensor.shape[0])
        print(f"[full] decoded {n_pairs} pairs at {EVAL_HW}")
        pair_tensor = pair_tensor.to(device)
        _stage(f"pairs_decoded_{n_pairs}")

        # Held-out validation indices (last val_pair_count pairs)
        val_count = max(1, min(args.val_pair_count, max(1, n_pairs // 8)))
        val_idx_start = n_pairs - val_count
        train_indices = torch.arange(0, val_idx_start, device=device, dtype=torch.long)
        val_indices = torch.arange(val_idx_start, n_pairs, device=device, dtype=torch.long)

        # 5. Build model
        cfg = SaneHnervConfig(
            latent_dim=args.latent_dim, sin_frequency=args.sin_frequency,
            num_pairs=n_pairs, output_height=EVAL_HW[0], output_width=EVAL_HW[1],
        )
        model = SaneHnervSubstrate(cfg).to(device)
        # Tier-1 O3: torch.compile wrap (no-op when flag=False)
        model = _compile_with_fallback(
            model, enabled=bool(getattr(args, "enable_torch_compile", False)),
            mode="default", fallback_on_error=True,
        )
        print(f"[full] sane_hnerv params: {model.num_parameters():,}")
        _stage("model_built")

        # 6. EMA shadow (CLAUDE.md non-negotiable)
        ema = EMA(model, decay=args.ema_decay)
        _stage(f"ema_wired_decay_{args.ema_decay}")

        # 7. Score-aware Lagrangian
        weights = ScoreAwareLossWeights(
            alpha_rate=args.alpha_rate, beta_seg=args.beta_seg,
            gamma_pose=args.gamma_pose, pose_weight_scale=args.pose_weight_scale,
            contest_normalizer=CONTEST_NORMALIZER,
        )
        loss_fn = SaneHnervScoreAwareLoss(seg_scorer=segnet, pose_scorer=posenet, weights=weights)
        _stage("lagrangian_built")

        # F3 GTScorerCache wire-in (TIER-1-OPT-BATCH 2026-05-14). Cache pre-
        # computes invariant GT PoseNet/SegNet forward; mathematically identical
        # to uncached path; ~1.4-1.5x per-step in real training.
        opt_ctx = _canon_build_optimized_training_context(
            args, scorers=(posenet, segnet), gt_pairs=pair_tensor,
            substrate_model=model, device=device,
        )
        gt_cache = opt_ctx.gt_cache
        if gt_cache is not None:
            print(gt_cache.summary_line())
            _stage("gt_scorer_cache_built")
        else:
            _stage("gt_scorer_cache_disabled")

        # 8. Optimizer + cosine annealing
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

        # 9. Train
        train_started_at = time.time()
        best_val_lag = math.inf
        best_epoch = -1
        ckpt_best_path = args.output_dir / "best.pt"
        n_train = int(train_indices.shape[0])
        batch_size = max(1, args.batch_size)
        archive_bytes_proxy = _archive_bytes_proxy_closed_form(model)
        nan_strike = 0
        max_nan_strikes = 3

        for epoch in range(args.epochs):
            model.train()
            perm = train_indices[torch.randperm(n_train, device=device)]
            epoch_loss_sum = 0.0
            epoch_batches = 0
            for start in range(0, n_train, batch_size):
                idx = perm[start : start + batch_size]
                if idx.numel() == 0:
                    continue
                # Tier-1 O2: autocast (no-op when flag=False)
                with _autocast_aware_forward(
                    enabled=bool(getattr(args, "enable_autocast_fp16", False)),
                    dtype=torch.float16, device=device,
                ):
                    rgb_0, rgb_1 = model(idx)
                    # Frames live in [0,1]; the score-aware loss + eval-roundtrip
                    # expect [0, 255]. Multiply at the boundary so the substrate
                    # output is gradient-clean.
                    rgb_0_255 = rgb_0 * 255.0
                    rgb_1_255 = rgb_1 * 255.0
                    gt = pair_tensor[idx]
                    gt_0 = gt[:, 0]
                    gt_1 = gt[:, 1]
                    gt_pose_batch = gt_seg_batch = None
                    gt_seg_already_probs = None
                    if gt_cache is not None:
                        gt_pose_batch, gt_seg_batch = gt_cache.lookup(idx, device=device)
                        gt_seg_already_probs = gt_cache.seg_already_probs
                    loss, parts = loss_fn(
                        rgb_0_255, rgb_1_255, gt_0, gt_1, archive_bytes_proxy,
                        apply_eval_roundtrip=True, noise_std=args.noise_std,
                        gt_pose_batch=gt_pose_batch, gt_seg_batch=gt_seg_batch,
                        gt_seg_already_probs=gt_seg_already_probs,
                    )
                # NaN watchdog
                if not torch.isfinite(loss):
                    nan_strike += 1
                    print(
                        f"[full] WARN: non-finite loss at epoch {epoch} batch {start}; "
                        f"strike {nan_strike}/{max_nan_strikes}",
                        file=sys.stderr,
                    )
                    if nan_strike >= max_nan_strikes:
                        raise RuntimeError(
                            f"NaN watchdog: {nan_strike} consecutive non-finite "
                            "losses; aborting training to preserve EMA shadow."
                        )
                    optimizer.zero_grad(set_to_none=True)
                    continue
                nan_strike = 0
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
                optimizer.step()
                ema.update(model)
                epoch_loss_sum += float(loss.detach().item())
                epoch_batches += 1

            scheduler.step()
            avg_loss = epoch_loss_sum / max(1, epoch_batches)

            # 10. Validation + best-ckpt selection (snapshot+restore pattern)
            if (epoch + 1) % args.val_every_epochs == 0 or epoch == args.epochs - 1:
                orig_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                ema.apply(model)
                model.eval()
                with torch.no_grad():
                    rgb_0_v, rgb_1_v = model(val_indices)
                    val_pose_batch = val_seg_batch = None
                    val_seg_already_probs = None
                    if gt_cache is not None:
                        val_pose_batch, val_seg_batch = gt_cache.lookup(val_indices, device=device)
                        val_seg_already_probs = gt_cache.seg_already_probs
                    val_loss, _val_parts = loss_fn(
                        rgb_0_v * 255.0, rgb_1_v * 255.0,
                        pair_tensor[val_indices, 0], pair_tensor[val_indices, 1],
                        archive_bytes_proxy, apply_eval_roundtrip=True,
                        noise_std=args.noise_std,
                        gt_pose_batch=val_pose_batch, gt_seg_batch=val_seg_batch,
                        gt_seg_already_probs=val_seg_already_probs,
                    )
                val_lag = float(val_loss.detach().item())
                model.load_state_dict(orig_state)
                model.train()
                print(
                    f"[full] epoch {epoch + 1}/{args.epochs} "
                    f"train_avg_loss={avg_loss:.6f} val_lagrangian={val_lag:.6f} "
                    f"(best_so_far={best_val_lag:.6f} @ ep{best_epoch + 1})"
                )
                if val_lag < best_val_lag and math.isfinite(val_lag):
                    best_val_lag = val_lag
                    best_epoch = epoch
                    # Save EMA shadow (NOT live weights) — CLAUDE.md EMA rule
                    ema_state = ema.state_dict()
                    torch.save({
                        "state_dict": {k: v.detach().cpu() for k, v in ema_state.items()},
                        "config": asdict(cfg), "ema_decay": args.ema_decay,
                        "best_val_lagrangian": val_lag, "best_epoch": int(epoch),
                        "saved_at_utc": _utc_now_iso(),
                        "training_axis_note": "[contest-CUDA] for promotion; auth eval still required",
                    }, ckpt_best_path)
            elif (epoch + 1) % max(1, args.val_every_epochs // 2) == 0:
                print(f"[full] epoch {epoch + 1}/{args.epochs} train_avg_loss={avg_loss:.6f}")

        train_elapsed_sec = time.time() - train_started_at
        _stage(f"train_complete_elapsed_{int(train_elapsed_sec)}s")

        if not ckpt_best_path.is_file():
            # Fallback save EMA shadow at end-of-training so downstream stages proceed
            print("[full] WARN: no improving val checkpoint; saving EMA shadow at end-of-training.", file=sys.stderr)
            ema_state = ema.state_dict()
            torch.save({
                "state_dict": {k: v.detach().cpu() for k, v in ema_state.items()},
                "config": asdict(cfg), "ema_decay": args.ema_decay,
                "best_val_lagrangian": best_val_lag, "best_epoch": int(args.epochs - 1),
                "saved_at_utc": _utc_now_iso(), "fallback_end_of_training_save": True,
            }, ckpt_best_path)

        # 11. Build SHV1 archive bytes from EMA shadow
        archive_sha = ""
        archive_bytes = 0
        archive_zip_path = args.output_dir / "archive.zip"
        if not args.skip_archive_build:
            print(f"[full] building archive from {ckpt_best_path} ...")
            ema_state = torch.load(ckpt_best_path, map_location="cpu", weights_only=False)
            sd = ema_state["state_dict"]
            decoder_sd = {k: v for k, v in sd.items() if k != "latents"}
            latents = sd["latents"].detach().cpu()
            meta = {
                "embed_dim": cfg.embed_dim, "initial_grid_h": cfg.initial_grid_h,
                "initial_grid_w": cfg.initial_grid_w,
                "decoder_channels": list(cfg.decoder_channels),
                "sin_frequency": cfg.sin_frequency,
                "output_height": cfg.output_height, "output_width": cfg.output_width,
                "num_upsample_blocks": cfg.num_upsample_blocks,
            }
            bin_bytes = pack_archive(decoder_sd, latents, meta)
            (args.output_dir / "0.bin").write_bytes(bin_bytes)
            archive_sha = _sha256_bytes(bin_bytes)
            archive_bytes = len(bin_bytes)
            print(f"[full] wrote 0.bin ({archive_bytes} bytes, sha256={archive_sha})")
            submission_dir = args.output_dir / "submission"
            _write_runtime(submission_dir)
            (submission_dir / "0.bin").write_bytes(bin_bytes)
            _build_archive_zip(archive_zip_path, bin_bytes=bin_bytes, submission_dir=submission_dir)
            print(f"[full] wrote {archive_zip_path}")
            _stage(f"archive_built_bytes_{archive_bytes}")

        # 12. CUDA auth eval — canonical helper (Catalog #226 self-protect)
        # Threads `--device cuda` into the contest_auth_eval subprocess CLI.
        auth_eval_result_path: Path | None = None
        contest_cuda_score: float | None = None
        if not args.skip_auth_eval and archive_zip_path.is_file():
            print("[full] launching CUDA auth eval ...")
            auth_eval_result_path = args.output_dir / "contest_auth_eval_cuda.json"
            auth_result = _canon_gate_auth_eval_call(
                args=args, archive_zip=archive_zip_path,
                inflate_sh=args.output_dir / "submission" / "inflate.sh",
                upstream_dir=args.upstream_dir, output_json=auth_eval_result_path,
                contest_auth_eval_script=CONTEST_AUTH_EVAL_SCRIPT,
                substrate_tag="sane_hnerv", device=device,
            )
            if auth_result is not None:
                contest_cuda_score = auth_result["auth_eval_cuda_score"]
                print(
                    f"[full] [contest-CUDA] score = {contest_cuda_score} "
                    f"(axis={auth_result['auth_eval_score_axis']}, "
                    f"lane_tag={auth_result['auth_eval_lane_tag']}, "
                    f"archive_sha256={archive_sha})"
                )
            _stage("auth_eval_cuda_done")

        # 13. Continual-learning posterior update (Catalog #128 atomic)
        if contest_cuda_score is not None and archive_sha:
            try:
                from tac.continual_learning import ContestResult, posterior_update_locked
                _detected_substrate = _canon_detect_hardware_substrate(
                    axis="cuda", substrate_tag="sane_hnerv",
                    provenance_path=args.output_dir / "provenance.json",
                    env_var_candidates=("SANE_HNERV_GPU", "MODAL_GPU"),
                )
                result = ContestResult(
                    axis="cuda", hardware_substrate=_detected_substrate,
                    architecture_class="lane_substrate_sane_hnerv_20260512",
                    score_value=contest_cuda_score, evidence_tag="[contest-CUDA]",
                    archive_sha256=archive_sha, archive_bytes=archive_bytes,
                    notes=f"sane_hnerv first-anchor dispatch; epochs={args.epochs}",
                    observed_at_utc=_utc_now_iso(),
                )
                update = posterior_update_locked(result)
                print(f"[full] posterior_update: accepted={update.accepted} reason={update.reason!r}")
            except Exception as exc:
                print(f"[full] posterior_update_locked failed: {exc}", file=sys.stderr)

        # 14. Cost-band anchor (best-effort; never fail the run on this).
        cost_band_anchor_appended = False
        cost_band_anchor_skip_reason: str | None = None
        try:
            from tac.cost_band_calibration import parse_actual_cost_usd
            actual_cost_usd = parse_actual_cost_usd(
                os.environ.get("SANE_HNERV_ACTUAL_COST_USD"),
                field_name="SANE_HNERV_ACTUAL_COST_USD",
            )
        except ValueError as exc:
            actual_cost_usd = None
            cost_band_anchor_skip_reason = f"invalid_SANE_HNERV_ACTUAL_COST_USD:{exc}"
        if COST_BAND_TOOL.is_file() and train_elapsed_sec > 0 and actual_cost_usd is not None:
            try:
                proc = subprocess.run([
                    sys.executable, str(COST_BAND_TOOL),
                    "--dispatch-label", f"sane_hnerv_{_utc_now_iso()}",
                    "--trainer", "experiments/train_substrate_sane_hnerv.py",
                    "--platform", os.environ.get("SANE_HNERV_PLATFORM", "vastai"),
                    "--gpu", os.environ.get("SANE_HNERV_GPU", "rtx_4090"),
                    "--epochs", str(args.epochs),
                    "--batch-size", str(args.batch_size),
                    "--actual-wall-clock-sec", str(train_elapsed_sec),
                    "--actual-cost-usd", str(actual_cost_usd),
                    "--notes", "OD-SUBSTRATE-4 first-anchor dispatch",
                ], capture_output=True, text=True, timeout=30, check=False)
                if proc.returncode == 0:
                    cost_band_anchor_appended = True
                else:
                    cost_band_anchor_skip_reason = (
                        f"append_failed_rc_{proc.returncode}:"
                        f"{(proc.stderr or proc.stdout)[-500:]}"
                    )
            except Exception as exc:
                cost_band_anchor_skip_reason = f"append_failed:{exc}"
                print(f"[full] cost-band anchor append failed (non-fatal): {exc}", file=sys.stderr)
        else:
            if actual_cost_usd is None and cost_band_anchor_skip_reason is None:
                cost_band_anchor_skip_reason = "missing_SANE_HNERV_ACTUAL_COST_USD"
            elif not COST_BAND_TOOL.is_file():
                cost_band_anchor_skip_reason = "cost_band_tool_missing"
            else:
                cost_band_anchor_skip_reason = "nonpositive_train_elapsed_sec"

        # 15. Provenance manifest
        provenance = {
            "schema": "sane_hnerv_provenance_v1",
            "generated_at": _utc_now_iso(),
            "from_state_hash": "regen_per_session_below",
            "git_head": _git_head_sha(),
            "trainer": "experiments/train_substrate_sane_hnerv.py",
            "lane_id": "lane_substrate_sane_hnerv_20260512",
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "pytorch_version": _torch_version_string(),
            "device": str(device),
            "num_pairs_decoded": n_pairs,
            "num_train_pairs": int(train_indices.shape[0]),
            "num_val_pairs": int(val_indices.shape[0]),
            "best_val_lagrangian": (best_val_lag if math.isfinite(best_val_lag) else None),
            "best_epoch": int(best_epoch),
            "train_elapsed_sec": float(train_elapsed_sec),
            "archive_sha256": archive_sha,
            "archive_bytes": archive_bytes,
            "auth_eval_cuda_score": contest_cuda_score,
            "auth_eval_json_path": (str(auth_eval_result_path) if auth_eval_result_path else None),
            "cost_band_anchor_appended": cost_band_anchor_appended,
            "cost_band_anchor_skip_reason": cost_band_anchor_skip_reason,
            "stage_log": stage_log,
            "custody_status": "ci-rebuildable",
            "score_claim": contest_cuda_score is not None,
            "score_axis_tag": ("[contest-CUDA]" if contest_cuda_score is not None else None),
            "promotion_eligible": False,  # gate on grand-council review
            "ready_for_exact_eval_dispatch": False,
        }
        (args.output_dir / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"[full] wrote {args.output_dir / 'provenance.json'}")
        return 0

    finally:
        unpatch_upstream_yuv6(yuv6_token)


# ---------------------------------------------------------------------------
# META layer SubstrateContract (Catalog #241/#242 canonical migration; landed
# 2026-05-15 by CATALOG-241-BACKFILL-29-TRAINERS subagent). Decoration extincts
# the Z3 v2 silent-drift bug class for this substrate by binding (a) the
# trainer's claimed contract, (b) the recipe schema, (c) the lane registry,
# and (d) the cost-band envelope into ONE source-of-truth that fails-loud at
# decoration time if the contract violates canonical invariants.
# ---------------------------------------------------------------------------

SANE_HNERV_SUBSTRATE_CONTRACT = SubstrateContract(
    # 2.1 Identity & lifecycle
    id="sane_hnerv",
    lane_id="lane_substrate_sane_hnerv_20260512",
    target_modes=("contest_one_video_replay", "research_substrate",),
    deployment_target="t4_contest_runtime",
    council_verdict_provenance=(
        ".omx/research/grand_council_fields_medal_substrate_design_20260512.md"
    ),
    # 2.2 Architecture & runtime (8 per Catalog #124)
    archive_grammar=(
        "SHV1 monolithic single-file 0.bin: header + sane HNeRV decoder weights "
        "(fp16 + brotli; PR101-grammar-compatible) + per-frame embeddings"
    ),
    parser_section_manifest={
        "header": "SHV1_magic_and_version",
        "decoder_weights": "fp16_brotli_blob",
        "frame_embeddings": "fp16_per_frame",
    },
    inflate_runtime_loc_budget=120,
    runtime_dep_closure=("torch>=2.5,<2.7", "brotli", "av",),
    export_format="fp16_brotli",
    score_aware_loss="scorer_loss_terms_btchw",
    bolt_on_loc_budget=1200,
    no_op_detector_planned=True,
    # 2.3 Operational mechanism (3 per Catalog #220)
    archive_bytes_added=None,
    score_improvement_mechanism_status="RESEARCH_ONLY",
    runtime_overlay_consumed=False,
    # 2.4 Recipe schema (8) — mirrors substrate recipe YAML
    recipe_smoke_only=True,
    recipe_research_only=False,
    recipe_min_smoke_gpu="A100",
    recipe_min_vram_gb=40,
    recipe_pyav_decode_strategy="cpu_thread_async_upload",
    recipe_canary_status="canary",
    recipe_video_input_strategy="per_dispatch_local_copy",
    recipe_canary_dependency=None,
    # 2.5 Cost band & GPU envelope (4)
    cost_band_epochs=100,
    cost_band_gpu_key="A100",
    cost_band_platform_key="modal",
    cost_band_p50_usd=0.5,
    # 2.6 6-hook wire-in (Catalog #125)
    hook_sensitivity_contribution="not_applicable_with_rationale",
    hook_pareto_constraint="rate_distortion_v1",
    hook_bit_allocator_class="not_applicable_with_rationale",
    hook_autopilot_ranker_class_shift_token=None,
    hook_continual_learning_anchor_kind="cuda_only",
    hook_probe_disambiguator=None,
    # 2.7 Compliance + 2.8 not-applicable rationales
    catalog_compliance_declarations=(
        "catalog_146_3arg_archive_grammar_honored",
        "catalog_151_tier1_required_flags_declared",
        "catalog_205_select_inflate_device_used",
        "catalog_220_operational_mechanism_declared",
        "catalog_226_gate_auth_eval_call_used",
    ),
    hook_not_applicable_rationale={
        "hook_sensitivity_contribution": (
            "sane HNeRV is canary substrate; sensitivity captured by rate-distortion"
        ),
        "hook_bit_allocator_class": (
            "fp16 brotli on weight blocks; no per-tensor bit allocator"
        ),
        "hook_probe_disambiguator": (
            "single mechanism (sane HNeRV); no 2+ defensible interpretations"
        ),
    },
)


@register_substrate(SANE_HNERV_SUBSTRATE_CONTRACT)
def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.smoke:
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
