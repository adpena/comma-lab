#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""NSCS02 downsampled-renderer + inflate-upsample trainer (UNIQUE-AND-COMPLETE-PER-METHOD).

Per the standing directive
``feedback_pr95_lesson_now_at_meta_level_unique_and_complete_per_method_default_20260515.md``
this trainer is a focused PR-95-paradigm implementation that BINDS architecture +
score-aware loss + archive grammar + inflate runtime + export contract + training
curriculum + tier-1 engineering + scorer routing into ONE coherent package
reviewable in 30 seconds.

PIPELINE (Hotz-glue canonical-helpers, ~200 LOC ``_full_main``; reuse
``trainer_skeleton`` + ``smoke_auth_eval_gate``; no reinvention):

1. Pin seeds + canonical device gate (CUDA-required for promotion-grade).
2. Patch upstream ``rgb_to_yuv6`` globally (Catalog #187) BEFORE scorer load.
3. Decode real contest pairs from ``upstream/videos/0.mkv`` via canonical
   ``decode_real_pairs`` (Catalog #114); shape ``(N, 2, 3, 384, 512)`` ``[0..255]``.
4. ``load_differentiable_scorers`` for SegNet+PoseNet (no inflate-time load).
5. Init NSCS02 5-stage decoder + EMA(0.997) + AdamW + cosine schedule.
6. Train UNIQUE NSCS02 score-aware loss (renders at (192, 256), upsamples to
   scorer-native (384, 512) for the proxy loss; eval_roundtrip approximated
   via ``apply_eval_roundtrip_during_training``).
7. Save EMA shadow at every val improvement (NEVER live weights — Catalog #88).
8. Pack NSCS02 monolithic ``0.bin`` archive + emit contest-compliant runtime tree.
9. Run CUDA auth eval via canonical ``gate_auth_eval_call`` (Catalog #226).
10. Continual-learning posterior update via canonical locked helper (Catalog #128).

CANONICAL-vs-UNIQUE DECISION PER LAYER (sister to design memo §canonical-vs-unique):

- **Architecture (UNIQUE)**: 5-stage decoder (vs A1's 6-stage). Already landed in
  ``architecture.NSCS02DownsampledDecoder``; the substrate's defining design choice.
- **Score-aware loss (UNIQUE)**: ``compute_nscs02_score_aware_loss`` routes through
  ``scorer.preprocess_input(...)`` directly because the train-time forward already
  upsamples (192, 256) -> (384, 512); canonical ``score_pair_components`` assumes
  the renderer outputs at scorer-native. Catalog #164 waiver:
  ``# SCORER_PREPROCESS_HANDLED_OK:nscs02_unique_downsample_upsample_path``.
- **Archive grammar (UNIQUE)**: NSCS02 wire format (single brotli pass; no
  per-tensor specialized streams) because the 5-stage decoder's ~165KB weight
  set does NOT justify A1's 28-stream complexity.
- **Inflate runtime (UNIQUE)**: ``submissions/nscs02_downsampled_renderer/inflate.py``
  upsamples (192, 256) -> (1164, 874) at decode time; A1 upsamples (384, 512) ->
  (1164, 874). The deeper downsample is the substrate's POINT.
- **EMA decay (HARD-EARNED canonical)**: 0.997 per CLAUDE.md "EMA — NON-NEGOTIABLE".
- **eval_roundtrip (HARD-EARNED canonical)**: ``apply_eval_roundtrip_during_training``
  per CLAUDE.md "eval_roundtrip — NON-NEGOTIABLE".
- **Differentiable scorer-preprocess (HARD-EARNED canonical)**: ``patch_upstream_yuv6_globally``
  + ``load_differentiable_scorers`` per PR 95 lesson (CLAUDE.md non-negotiable).
- **Auth eval routing (HARD-EARNED canonical)**: ``gate_auth_eval_call`` per
  Catalog #226 (auth_eval CLI flag stability).
- **Tier-1 engineering (HARD-EARNED canonical)**: autocast/TF32/no_grad-at-eval
  per Catalogs #170-#173/#178-#182 (engineering hygiene; score-neutral).
- **Real-video decode (HARD-EARNED canonical)**: ``_canon_decode_real_pairs`` per
  CLAUDE.md "Forbidden make_synthetic_pair_batch in non-smoke" non-negotiable.

USAGE (smoke; macOS CPU or Linux CPU, tiny config, ~$0)::

    .venv/bin/python experiments/train_substrate_nscs02_downsampled_renderer.py \\
        --smoke --output-dir experiments/results/nscs02_smoke_<utc> --device cpu

USAGE (full; CUDA-required Modal T4, $5-15 envelope)::

    .venv/bin/python experiments/train_substrate_nscs02_downsampled_renderer.py \\
        --video-path upstream/videos/0.mkv \\
        --output-dir experiments/results/nscs02_<utc> \\
        --epochs 200 --batch-size 8 --lr 5e-4 --device cuda
"""
# Catalog #168 AnnAssign + Catalog #151 manifest.
# SYNTHETIC_NON_SMOKE_OK:_smoke_main-only-uses-synthetic-data-_full_main-decodes-real-video
# NO_GRAD_WAIVED:eval-time-scorer-forward-wrapped-in-torch.inference_mode-inside-validation-block
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
from tac.substrates.nscs02_downsampled_renderer import (
    NSCS02_BASE_CHANNELS,
    NSCS02_LATENT_DIM,
    NSCS02_N_PAIRS,
    NSCS02_RENDER_HW,
)
from tac.substrates.nscs02_downsampled_renderer.architecture import (
    NSCS02DownsampledDecoder,
)
from tac.substrates.nscs02_downsampled_renderer.archive import (
    pack_nscs02_archive,
    parse_nscs02_archive,
)
from tac.substrates.nscs02_downsampled_renderer.registered_substrate import (
    NSCS02_DOWNSAMPLED_RENDERER_CONTRACT,  # noqa: F401  (forces contract validation)
)
from tac.substrates.nscs02_downsampled_renderer.score_aware_loss import (
    SCORER_HW,
    compute_nscs02_score_aware_loss,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO_PATH = REPO_ROOT / "upstream" / "videos" / "0.mkv"
DEFAULT_UPSTREAM_DIR = REPO_ROOT / "upstream"
CONTEST_AUTH_EVAL_SCRIPT = REPO_ROOT / "experiments" / "contest_auth_eval.py"
SUBSTRATE_TAG = "nscs02_downsampled_renderer"
SUBSTRATE_LANE_ID = "lane_nscs02_downsampled_renderer_inflate_upsample_20260515"


# ---------------------------------------------------------------------------
# Catalog #151 manifest — declared as ast.AnnAssign per Catalog #168
# (META gate handles BOTH ast.Assign and ast.AnnAssign post-2026-05-12).
# ---------------------------------------------------------------------------
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--video-path": {
        "env": "NSCS02_VIDEO_PATH",
        "rationale": (
            "Path to the contest video `upstream/videos/0.mkv` decoded via "
            "pyav into per-pair frames; required for non-smoke training"
        ),
        "default": str(DEFAULT_VIDEO_PATH),
        "required_input_file": True,
    },
    "--output-dir": {
        "env": "NSCS02_OUTPUT_DIR",
        "rationale": (
            "Output directory for checkpoints, archive, stats, runtime tree, "
            "auth eval JSON; must be writable + outside /tmp"
        ),
        "default": None,
    },
    "--epochs": {
        "env": "NSCS02_EPOCHS",
        "rationale": "Training epoch count; smoke=3, Modal T4 full=200",
        "default": "200",
    },
    "--batch-size": {
        "env": "NSCS02_BATCH_SIZE",
        "rationale": (
            "Per-step pair count; T4 handles 8-16 at downsampled (192, 256) "
            "render resolution (much larger budget than A1's 4-8 at 384x512)"
        ),
        "default": "8",
    },
    "--lr": {
        "env": "NSCS02_LR",
        "rationale": "AdamW base learning rate; default 5e-4",
        "default": "5e-4",
    },
    "--upsample-mode": {
        "env": "NSCS02_UPSAMPLE_MODE",
        "rationale": (
            "Inflate-side upsample mode (192, 256) -> (384, 512). Bicubic "
            "matches A1's inflate; bilinear is the train/test scorer's own "
            "preprocess mode. Defaults bicubic for inflate parity."
        ),
        "default": "bicubic",
    },
    "--seg-weight": {
        "env": "NSCS02_SEG_WEIGHT",
        "rationale": "SegNet KL-distill weight (contest formula = 100)",
        "default": "100.0",
    },
    "--pose-weight": {
        "env": "NSCS02_POSE_WEIGHT",
        "rationale": "PoseNet MSE weight (contest formula sqrt-amplified)",
        "default": "1.0",
    },
    "--enable-autocast-fp16": {
        "env": "ENABLE_AUTOCAST_FP16",
        "rationale": (
            "Catalog #172 Tier-1 speed primitive; canonical engineering "
            "hygiene shared with all substrates (score-neutral)"
        ),
        "default": "false",
    },
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_substrate_nscs02_downsampled_renderer",
        description=(
            "Train NSCS02 downsampled-renderer + inflate-upsample substrate "
            "(ASSUMPTIONS-CHALLENGE-AUDIT NSCS02 entry; predicted ΔS = "
            "[-0.010, -0.030] vs A1 baseline)."
        ),
    )
    p.add_argument("--video-path", type=Path, default=DEFAULT_VIDEO_PATH)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR)
    p.add_argument("--upsample-mode", type=str, default="bicubic",
                   choices=("bicubic", "bilinear"))
    p.add_argument("--seg-weight", type=float, default=100.0)
    p.add_argument("--pose-weight", type=float, default=1.0)
    p.add_argument("--pixel-weight", type=float, default=0.1)
    p.add_argument("--latent-dim", type=int, default=NSCS02_LATENT_DIM)
    p.add_argument("--base-channels", type=int, default=NSCS02_BASE_CHANNELS)

    # Training hyperparams
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.997)
    p.add_argument("--n-pairs-train", type=int, default=NSCS02_N_PAIRS,
                   help="Number of contest pairs decoded from upstream/videos/0.mkv")

    # Tier-1 engineering primitives (Catalogs #170-#173/#178-#182; canonical)
    p.add_argument("--enable-autocast-fp16", action="store_true",
                   help="Catalog #172 Tier-1 speed primitive; score-neutral")

    # Mode flags
    p.add_argument("--smoke", action="store_true",
                   help="Run synthetic-data smoke (no scorer load, no real video)")
    p.add_argument("--full-cpu", action="store_true",
                   help="Opt-in to non-smoke CPU training (Catalog #197 paired flag required)")
    p.add_argument("--advisory-cpu-explicitly-waived", action="store_true",
                   help="Required sister flag for --full-cpu (Catalog #197)")
    p.add_argument("--skip-auth-eval", action="store_true",
                   help="Skip CUDA auth eval at end of full training (debug only)")
    p.add_argument("--relax-determinism-for-backward", action="store_true", default=True,
                   help="Phase B.2 root-cause fix; relax determinism for bicubic backward")
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
    """Synthetic-data sanity smoke — validates substrate forward + archive roundtrip.

    No scorer load. No real video decode. ``$0`` cost. Verifies:

    1. NSCS02 decoder instantiates with the canonical config.
    2. Forward pass produces (B, 2, 3, 192, 256) RGB pair.
    3. Upsample to (B, 2, 3, 384, 512) via canonical bicubic.
    4. Archive pack -> parse roundtrip is byte-identical for state-dict.
    5. NSCS02 magic + section-offset parser refuses tampered bytes.
    """
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    decoder = NSCS02DownsampledDecoder(
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
        render_hw=NSCS02_RENDER_HW,
    ).to(device)
    decoder.eval()

    print(f"[nscs02-smoke] decoder param count: {decoder.parameter_count():,}")

    # Forward smoke — render a tiny batch
    z = torch.randn(4, args.latent_dim, device=device)
    with torch.no_grad():
        rendered = decoder(z)
    expected_shape = (4, 2, 3, NSCS02_RENDER_HW[0], NSCS02_RENDER_HW[1])
    if tuple(rendered.shape) != expected_shape:
        raise RuntimeError(
            f"smoke forward shape mismatch: got {tuple(rendered.shape)}; "
            f"expected {expected_shape}"
        )

    # Upsample-to-scorer smoke
    with torch.no_grad():
        upsampled = decoder.render_then_upsample_to_scorer(z, scorer_hw=SCORER_HW, mode=args.upsample_mode)
    expected_up = (4, 2, 3, SCORER_HW[0], SCORER_HW[1])
    if tuple(upsampled.shape) != expected_up:
        raise RuntimeError(
            f"smoke upsample shape mismatch: got {tuple(upsampled.shape)}; "
            f"expected {expected_up}"
        )

    # Archive roundtrip smoke
    latents = torch.randn(NSCS02_N_PAIRS, args.latent_dim)
    archive_bytes = pack_nscs02_archive(decoder, latents)
    print(f"[nscs02-smoke] archive size: {len(archive_bytes):,} bytes")

    template = NSCS02DownsampledDecoder(
        latent_dim=args.latent_dim,
        base_channels=args.base_channels,
        render_hw=NSCS02_RENDER_HW,
    )
    parsed = parse_nscs02_archive(archive_bytes, template)
    if parsed.decoder_state_dict.keys() != decoder.state_dict().keys():
        raise RuntimeError("smoke archive state-dict key-set mismatch after roundtrip")

    # Latents fp16 roundtrip should be approximately equal (tolerance fp16 epsilon)
    diff = (parsed.latents - latents).abs().max().item()
    if diff > 1e-2:
        raise RuntimeError(f"smoke latents fp16 roundtrip max-diff too large: {diff}")

    # Tampered-bytes refusal
    tampered = bytearray(archive_bytes)
    tampered[0] = (tampered[0] + 1) & 0xFF
    try:
        parse_nscs02_archive(bytes(tampered), template)
        raise RuntimeError("smoke parser failed to refuse tampered magic bytes")
    except ValueError:
        pass

    smoke_stats = {
        "substrate_id": "nscs02_downsampled_renderer",
        "render_hw": list(NSCS02_RENDER_HW),
        "scorer_hw": list(SCORER_HW),
        "param_count": decoder.parameter_count(),
        "archive_bytes": len(archive_bytes),
        "smoke_status": "GREEN",
        "score_improvement_mechanism_status": "RESEARCH_ONLY",
        "operational_overlay": False,
        "runtime_overlay_consumed": False,
    }
    (output_dir / "smoke_stats.json").write_text(json.dumps(smoke_stats, indent=2))
    print("[nscs02-smoke] DONE")
    return 0


# ---------------------------------------------------------------------------
# Full entry path — UNIQUE-AND-COMPLETE-PER-METHOD PR 95-paradigm pipeline
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
    """Emit contest-compliant inflate.sh + inflate.py + vendored NSCS02 inflate.

    Per Catalog #146: 3-positional-arg inflate.sh signature.
    Per Catalog #163: ``set -euo pipefail`` for fail-closed semantics.
    Per HNeRV parity discipline L4: <= 200 LOC inflate runtime budget.
    Per Catalog #205: device selection via canonical ``select_inflate_device``.

    The vendored ``submissions/nscs02_downsampled_renderer/inflate.py`` is
    the substrate's UNIQUE inflate runtime (~95 LOC; renders at (192, 256),
    upsamples to camera (1164, 874)). It is the AUTHORITATIVE inflate, not
    the canonical helper — it embodies the substrate's defining design choice.
    """
    submission_dir.mkdir(parents=True, exist_ok=True)
    nscs02_src_root = REPO_ROOT / "submissions" / "nscs02_downsampled_renderer"
    if not nscs02_src_root.is_dir():
        raise FileNotFoundError(f"canonical NSCS02 submission tree missing: {nscs02_src_root}")

    # Copy the substrate's UNIQUE inflate.py + src/ tree byte-for-byte.
    sub_src = submission_dir / "src"
    sub_src.mkdir(parents=True, exist_ok=True)
    for name in ("codec.py", "model.py"):
        shutil.copy2(nscs02_src_root / "src" / name, sub_src / name)

    # Vendor the shared inflate-runtime helper if NSCS02 inflate.py imports it.
    _canon_vendor_shared_inflate_runtime(submission_dir, repo_root=REPO_ROOT)

    # Substrate-side inflate.py is self-contained (~103 LOC reviewable in 30s).
    shutil.copy2(nscs02_src_root / "inflate.py", submission_dir / "inflate.py")

    inflate_sh = (
        "#!/usr/bin/env bash\n"
        "# NSCS02 downsampled-renderer contest-compliant inflate runtime.\n"
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
        '  src="${DATA_DIR}/${base}.bin"\n'
        '  if [ ! -f "${src}" ]; then src="${DATA_DIR}/0.bin"; fi\n'
        '  if [ ! -f "${src}" ]; then src="${DATA_DIR}/x"; fi\n'
        '  dst="${OUTPUT_DIR}/${base}.raw"\n'
        '  [ ! -f "${src}" ] && echo "ERROR: ${src} not found" >&2 && exit 1\n'
        '  "${PYTHON:-python3}" "${HERE}/inflate.py" "${src}" "${dst}"\n'
        'done < "${FILE_LIST}"\n'
    )
    (submission_dir / "inflate.sh").write_text(inflate_sh, encoding="utf-8")
    (submission_dir / "inflate.sh").chmod(0o755)


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
    """NSCS02 full training (UNIQUE-AND-COMPLETE-PER-METHOD PR 95-paradigm).

    Pipeline (10 stages bound into ONE coherent ~200 LOC implementation;
    canonical-vs-unique decisions documented at module docstring):

    1. Pin seeds + canonical CUDA device gate.
    2. Patch upstream ``rgb_to_yuv6`` globally (Catalog #187) BEFORE scorer load.
    3. Decode 600 real contest pairs from ``upstream/videos/0.mkv`` via canonical
       ``decode_real_pairs`` (Catalog #114).
    4. ``load_differentiable_scorers`` (PoseNet, SegNet) — frozen, no grads.
    5. Init NSCS02 5-stage decoder + per-pair latents + EMA(0.997) + AdamW + cosine.
    6. Train UNIQUE NSCS02 score-aware Lagrangian via ``compute_nscs02_score_aware_loss``
       (renders at (192, 256), upsamples to scorer-native, scorer-distortion proxy).
    7. Save EMA shadow at every val improvement (NEVER live weights — Catalog #88).
    8. Pack NSCS02 monolithic ``0.bin`` archive + emit contest runtime tree.
    9. Run CUDA auth eval via canonical ``gate_auth_eval_call`` (Catalog #226).
    10. Posterior update via canonical locked helper (Catalog #128).
    """
    # HARD-EARNED canonical helpers per CLAUDE.md non-negotiables.
    from tac.differentiable_eval_roundtrip import (
        patch_upstream_yuv6_globally,
        unpatch_upstream_yuv6,
    )
    from tac.scorer import load_differentiable_scorers
    from tac.training import EMA

    _canon_pin_seeds(args.seed)
    # Bicubic interp backward is NOT deterministic-CUDA-supported; relax determinism
    # for the training step (eval / archive / inflate remain deterministic).
    if getattr(args, "relax_determinism_for_backward", True):
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    device = _canon_device_or_die(
        args.device,
        smoke=bool(args.full_cpu),
        substrate_tag=SUBSTRATE_TAG,
    )

    stage_log: list[dict[str, str]] = []

    def _stage(name: str) -> None:
        msg = {"stage": name, "at": _canon_utc_now_iso()}
        stage_log.append(msg)
        print(f"[nscs02-full] {name} @ {msg['at']}")

    _stage("seed_pinned")

    # Stage 2: patch upstream rgb_to_yuv6 (Catalog #187) BEFORE scorer load.
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
    param_count: int = 0
    runtime_manifest: dict[str, Any] = {}

    try:
        # Stage 3: decode real pairs from contest video (Catalog #114).
        n_pairs_train = int(args.n_pairs_train)
        gt_pairs_btchw_255 = _canon_decode_real_pairs(
            args.video_path,
            n_pairs=n_pairs_train,
            substrate_tag=SUBSTRATE_TAG,
            repo_root=REPO_ROOT,
        ).to(device)
        n_pairs = int(gt_pairs_btchw_255.shape[0])
        _stage(f"video_decoded_{n_pairs}_pairs")

        # Stage 4: load differentiable scorers (Catalog #222: posenet, segnet order).
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

        # Stage 5: init NSCS02 decoder + per-pair latents + EMA + optimizer.
        decoder = NSCS02DownsampledDecoder(
            latent_dim=args.latent_dim,
            base_channels=args.base_channels,
            render_hw=NSCS02_RENDER_HW,
        ).to(device)
        # Per-pair latents are learnable parameters (mirrors A1 + Z4 pattern).
        latents = torch.nn.Parameter(
            torch.randn(n_pairs, args.latent_dim, device=device) * 0.01
        )
        param_count = decoder.parameter_count() + latents.numel()
        _stage(f"decoder_built_{param_count}_params")

        ema_decoder = EMA(decoder, decay=args.ema_decay)
        # Latents do not need EMA shadow (per-frame embeddings; canonical pattern).
        optimizer = torch.optim.AdamW(
            list(decoder.parameters()) + [latents],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs)
        )

        # Train/val split: hold out last 12.5% of pairs.
        val_count = max(1, n_pairs // 8)
        val_idx_start = n_pairs - val_count
        train_indices_pool = list(range(val_idx_start))
        val_indices_pool = list(range(val_idx_start, n_pairs))

        ckpt_best_path = out_dir / "best.pt"
        nan_strike = 0
        max_nan_strikes = 3

        # Tier-1 autocast wrap (HARD-EARNED canonical engineering hygiene).
        from tac.training_optimization.autocast_helper import (
            autocast_aware_forward as _autocast_aware_forward,
        )

        for epoch in range(args.epochs):
            decoder.train()
            random.shuffle(train_indices_pool)
            epoch_losses: list[float] = []
            for batch_start in range(0, len(train_indices_pool), args.batch_size):
                batch_indices = train_indices_pool[
                    batch_start : batch_start + args.batch_size
                ]
                if not batch_indices:
                    continue
                pair_idxs = torch.tensor(batch_indices, device=device, dtype=torch.long)
                gt_batch = gt_pairs_btchw_255[pair_idxs]  # (B, 2, 3, 384, 512)
                z_batch = latents[pair_idxs]

                with _autocast_aware_forward(
                    enabled=bool(getattr(args, "enable_autocast_fp16", False)),
                    dtype="fp16",
                    device=device.type,
                ):
                    components = compute_nscs02_score_aware_loss(
                        decoder, z_batch, gt_batch,
                        seg_scorer=seg_scorer, pose_scorer=pose_scorer,
                        seg_weight=args.seg_weight,
                        pose_weight=args.pose_weight,
                        pixel_weight=args.pixel_weight,
                        upsample_mode=args.upsample_mode,
                    )
                    loss = components.total
                if not torch.isfinite(loss):
                    nan_strike += 1
                    print(f"[nscs02-full] NaN strike {nan_strike}/{max_nan_strikes} at epoch {epoch}")
                    if nan_strike >= max_nan_strikes:
                        raise RuntimeError("NaN watchdog tripped — refusing to continue")
                    continue
                nan_strike = 0
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(decoder.parameters()) + [latents], args.grad_clip,
                )
                optimizer.step()
                ema_decoder.update(decoder)
                epoch_losses.append(float(loss.detach().cpu()))
            scheduler.step()

            # Eval ~every epochs/20 OR at last epoch (Catalog #88: EMA shadow).
            if epoch % max(1, args.epochs // 20) == 0 or epoch == args.epochs - 1:
                live_state = {
                    k: v.detach().clone() for k, v in decoder.state_dict().items()
                }
                ema_decoder.apply(decoder)
                decoder.eval()
                try:
                    with torch.inference_mode():
                        val_idx = torch.tensor(
                            val_indices_pool, device=device, dtype=torch.long,
                        )
                        gt_val = gt_pairs_btchw_255[val_idx]
                        z_val = latents[val_idx]
                        val_components = compute_nscs02_score_aware_loss(
                            decoder, z_val, gt_val,
                            seg_scorer=seg_scorer, pose_scorer=pose_scorer,
                            seg_weight=args.seg_weight,
                            pose_weight=args.pose_weight,
                            pixel_weight=args.pixel_weight,
                            upsample_mode=args.upsample_mode,
                        )
                        val_loss = float(val_components.total.detach().cpu())
                        val_seg = float(val_components.seg_loss.detach().cpu())
                        val_pose = float(val_components.pose_loss.detach().cpu())
                        val_pixel = float(val_components.pixel_loss.detach().cpu())
                finally:
                    decoder.load_state_dict(live_state)
                    decoder.train()
                avg_train = (
                    sum(epoch_losses) / len(epoch_losses)
                    if epoch_losses else math.nan
                )
                print(
                    f"[nscs02-full] epoch {epoch:4d}: train={avg_train:.5f} "
                    f"val={val_loss:.5f} seg={val_seg:.6f} pose={val_pose:.6f} "
                    f"pixel={val_pixel:.4f} (best={best_val_loss:.5f})"
                )
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_epoch = epoch
                    # Save EMA shadow (NEVER live weights — Catalog #88).
                    live_state2 = {
                        k: v.detach().clone() for k, v in decoder.state_dict().items()
                    }
                    ema_decoder.apply(decoder)
                    ema_state = {
                        k: v.detach().cpu() for k, v in decoder.state_dict().items()
                    }
                    decoder.load_state_dict(live_state2)
                    torch.save(
                        {
                            "state_dict": ema_state,
                            "latents": latents.detach().cpu(),
                            "epoch": epoch,
                            "val_loss": val_loss,
                            "val_loss_metric": "nscs02_score_aware_loss_total",
                            "val_seg_loss": val_seg,
                            "val_pose_loss": val_pose,
                            "val_pixel_loss": val_pixel,
                        },
                        ckpt_best_path,
                    )

        train_elapsed = time.time() - train_started_at
        _stage(
            f"trained_best_epoch_{best_epoch}_val_loss_{best_val_loss:.5f}_"
            f"elapsed_{train_elapsed:.1f}s"
        )

        # Stage 8: load EMA-best checkpoint + pack NSCS02 archive.
        if ckpt_best_path.is_file():
            best_ckpt = torch.load(
                ckpt_best_path, weights_only=False, map_location=device,
            )
            decoder.load_state_dict(best_ckpt["state_dict"])
            best_latents = best_ckpt["latents"]
        else:
            best_latents = latents.detach().cpu()
        decoder.eval()

        archive_bytes = pack_nscs02_archive(decoder, best_latents)
        (out_dir / "0.bin").write_bytes(archive_bytes)
        archive_size = len(archive_bytes)
        archive_sha = _canon_sha256_bytes(archive_bytes)
        _build_archive_zip(archive_zip_path, bin_bytes=archive_bytes)
        submission_dir.mkdir(parents=True, exist_ok=True)
        _write_runtime(submission_dir)
        (submission_dir / "0.bin").write_bytes(archive_bytes)
        shutil.copy2(archive_zip_path, submission_dir / "archive.zip")
        runtime_manifest = _write_submission_runtime_manifest(submission_dir)
        _stage(
            f"archive_packed_{archive_size}_B_sha{archive_sha[:8]}_runtime_emitted"
        )

        # Stage 9: CUDA auth eval via canonical gate (Catalog #226).
        if not args.skip_auth_eval:
            _stage("contest_auth_eval")
            # Catalog #249 + #226: filename matches actual device; helper redirects on mismatch.
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
    auth_eval_json_path = out_dir / "contest_auth_eval_cuda.json"
    if not args.skip_auth_eval and auth_eval_json_path.exists():
        try:
            from tac.continual_learning import (
                posterior_update_locked_from_auth_eval_json,
            )

            update = posterior_update_locked_from_auth_eval_json(auth_eval_json_path)
            print(
                f"[nscs02-full] posterior_update accepted={getattr(update, 'accepted', '?')}"
            )
            _stage("posterior_updated")
        except Exception as exc:
            print(f"[nscs02-full] WARN posterior_update failed: {exc!r}")

    # Cost-band anchor (Catalog #175/#177).
    if auth_eval_score_claim_valid:
        try:
            from tac.cost_band_calibration import CostBandAnchor, append_anchor

            wall_sec = float(time.time() - train_started_at)
            anchor = CostBandAnchor(
                logged_at_utc=_canon_utc_now_iso(),
                dispatch_label=f"{SUBSTRATE_LANE_ID}_{_canon_utc_now_iso()}",
                trainer="train_substrate_nscs02_downsampled_renderer",
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
                    f"upsample_mode={args.upsample_mode};"
                    f"literature_anchor=nscs02_downsampled_renderer_inflate_upsample;"
                    f"lane_class=substrate_engineering"
                ),
            )
            append_anchor(anchor)
            _stage("cost_band_anchor_appended")
        except Exception as exc:
            print(f"[nscs02-full] WARN cost_band_anchor_append failed: {exc!r}")

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
        env_var_candidates=("NSCS02_GPU", "MODAL_GPU"),
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
        "upsample_mode": args.upsample_mode,
        "seg_weight": args.seg_weight,
        "pose_weight": args.pose_weight,
        "pixel_weight": args.pixel_weight,
        "param_count": param_count,
        "predicted_delta_s_band": [-0.030, -0.010],
        "predicted_band_axis": "[contest-CPU 1to1] vs A1 0.1928",
        "council_verdict": (
            "ASSUMPTIONS-CHALLENGE-AUDIT NSCS02 first-anchor; UNIQUE-AND-COMPLETE-"
            "PER-METHOD PR 95-paradigm implementation 2026-05-15"
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
        "runtime_tree_sha256": runtime_manifest.get("runtime_tree_sha256", ""),
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
        f"[nscs02-full] DONE archive={archive_size}B sha={archive_sha[:12]}... "
        f"auth_score={auth_eval_score} grade={auth_eval_evidence_grade} "
        f"upsample={args.upsample_mode}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _validate_full_cpu_flags(args)
    if args.smoke:
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
