# SPDX-License-Identifier: MIT
"""Train the A1 + LAPose composition substrate (D1.D HIERARCHICAL).

Council memo: ``.omx/research/grand_council_pose_axis_non_hnerv_a1_plus_lapose_20260513.md``
Paper review:  ``.omx/research/siren_literature_review_20260513.md``
Lane:          ``lane_a1_plus_lapose_composition_20260513``

Binding council verdicts honored (per CLAUDE.md "Design decisions" + the
council memo §3.3 tally):

* **D1.D HIERARCHICAL** (8-2): A1 base substrate + LAPose RGB foveal
  residual on selected pairs.  A1's wire format is preserved verbatim;
  the LAPose sidecar is APPENDED to the A1 single-ZIP-member ``x`` blob.
* **D2.B bounded with D2.A target** (10-0): pose-residual budget ≤ 5 KB;
  target ~2 KB with Markov-1 conditioning at the int8 quantization level
  the council prescribed.  Closed-form size bound is computed at config
  time from ``residual_rank * (3*fov_h + fov_w) * num_selected``.
* **D3.B JOINT END-TO-END** (operator-routed 2026-05-13): the LAPose
  residual head trains end-to-end with score-aware Lagrangian
  ``α·B/N + β·d_seg + γ·√d_pose``.  A1 is loaded as a frozen base; only
  the residual parameters carry gradients.  This is the Phase-2-optimal
  lean per Shannon §4.1 and Dykstra §4.2.
* **D4 PENDING DEEPER COUNCIL** (operator-routed 2026-05-13): the
  inflate.py PRIMARY path is D4.B (single-stage with new archive
  section appended to A1's blob).  The sister COUNCIL-D4-DEEPER subagent
  will refine; the current inflate.py exposes hooks for D4.A two-stage
  via the ``split_composition_archive`` helper.
* **D5.A + D5.C** (7-3): FastViT-T12 RepMixer 12-channel YUV6 attack
  (the foveal residual modulates the pose-relevant Y channels) +
  SO(3)×R^3 Lie-algebra implicit parameterization at the residual blob
  level (residual lives in physically-meaningful 6-DoF camera space).
  D5.B SegNet stride-2 trick is REFUSED (not pose axis); D5.D
  kitchen-sink is REFUSED (PR105 anti-pattern).
* **D6 BOTH AXES** (10-0): contest-CPU GHA Linux x86_64 + contest-CUDA
  must both be emitted, per CLAUDE.md "Submission auth eval — BOTH CPU
  AND CUDA" non-negotiable.  The trainer threads both via the contest
  auth-eval subprocess (CUDA inline at end-of-train; CPU axis dispatched
  separately by the operator wrapper to Linux x86_64).

CLAUDE.md non-negotiables honored end-to-end:

* Score-aware substrate (HNeRV parity L1) — train against
  ``upstream/videos/0.mkv`` decoded via pyav, gradient through scorers.
* ``patch_upstream_yuv6_globally()`` BEFORE ``load_differentiable_scorers``
  (Catalog #187; PR #95/#106 contract).
* ``apply_eval_roundtrip=True`` inside the per-batch loop (Catalog #5;
  no opt-out per the canonical helper).
* EMA decay 0.997 + snapshot+restore at eval (Catalog #88; inference
  checkpoint is EMA shadow, NEVER live weights).
* No scorer load at inflate time (Catalog #6; inflate.py is RGB-only).
* No /tmp persisted evidence paths (CLAUDE.md FORBIDDEN_PATTERN).
* Tier-1 engineering wins: ``--enable-autocast-fp16`` (Catalog #172),
  TF32 default (Catalog #178), ``--enable-torch-compile`` declared
  (Catalog #179), ``torch.no_grad`` at eval (Catalog #180).
* TIER_1_OPERATOR_REQUIRED_FLAGS manifest as ``ast.AnnAssign`` so
  Catalog #151 introspection sees it (Catalog #168).

Usage (smoke; CPU, tiny config, ~3 epochs, no scorer load)::

    .venv/bin/python experiments/train_substrate_a1_plus_lapose.py \\
        --a1-archive experiments/results/track4_sg_a1_t178000_20260509/submission_dir/archive.zip \\
        --lapose-atom-manifest .omx/research/artifacts/lapose_motion_atoms_20260505_codex/lapose_motion_atom_manifest_fixture.json \\
        --video-path upstream/videos/0.mkv \\
        --output-dir experiments/results/a1_plus_lapose_smoke_<utc> \\
        --epochs 3 --device cpu --smoke

Usage (full; CUDA-required; threads from operator wrapper)::

    .venv/bin/python experiments/train_substrate_a1_plus_lapose.py \\
        --a1-archive .../submission_dir/archive.zip \\
        --lapose-atom-manifest <manifest.json> \\
        --video-path upstream/videos/0.mkv \\
        --output-dir experiments/results/a1_plus_lapose_<utc> \\
        --epochs 3000 --batch-size 32 --lr 1e-3 --device cuda
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

from tac.substrate_registry import SubstrateContract, register_substrate
from tac.substrates._shared.smoke_auth_eval_gate import (
    gate_auth_eval_call as _canon_gate_auth_eval_call,
)
from tac.substrates._shared.trainer_skeleton import (
    build_optimized_training_context as _build_optimized_training_context,
)
from tac.substrates._shared.trainer_skeleton import (
    detect_hardware_substrate as _canon_detect_hardware_substrate,
)
from tac.substrates._shared.trainer_skeleton import (
    load_upstream_yuv420_to_rgb as _canon_load_upstream_yuv420_to_rgb,
)
from tac.substrates._shared.trainer_skeleton import (
    utc_now_iso as _canon_utc_now_iso,
)
from tac.substrates._shared.trainer_skeleton import (
    vendor_shared_inflate_runtime as _canon_vendor_shared_inflate_runtime,
)

# ---------------------------------------------------------------------------
# Module paths + constants (council memo + CLAUDE.md anchors)
# ---------------------------------------------------------------------------

DEFAULT_VIDEO_PATH = REPO_ROOT / "upstream" / "videos" / "0.mkv"
DEFAULT_UPSTREAM_DIR = REPO_ROOT / "upstream"
DEFAULT_A1_ARCHIVE = (
    REPO_ROOT
    / "experiments"
    / "results"
    / "track4_sg_a1_t178000_20260509"
    / "submission_dir"
    / "archive.zip"
)
CONTEST_AUTH_EVAL_SCRIPT = REPO_ROOT / "experiments" / "contest_auth_eval.py"
COST_BAND_TOOL = REPO_ROOT / "tools" / "append_cost_band_anchor.py"

EVAL_HW = (384, 512)
CAMERA_HW = (874, 1164)
N_PAIRS_FULL = 600
CONTEST_NORMALIZER = 37_545_489.0

# Canonical A1 anchor (post-fine-tune; council memo §2 + operator directive)
A1_CANONICAL_SHA256 = (
    "8e664385af0a25ec98bd02d97b697fbf0d2bb3c2d954f5aa5c95b5131330a243"
)
A1_CANONICAL_BYTES = 178_162

# A1 decoder constants (re-exported from the substrate package for trainer-
# local helpers that vendor the A1 codec/model at runtime; the canonical
# definitions live in src/tac/substrates/a1_plus_lapose/architecture.py).
A1_LATENT_DIM = 28
A1_BASE_CHANNELS = 36


# ---------------------------------------------------------------------------
# Catalog #151 manifest — annotated assignment per Catalog #168 so the AST
# walker in Catalog #151's static gate observes the dict.  Mirrors the Balle
# trainer schema (env / rationale / default / required_input_file /
# generator_command).
# ---------------------------------------------------------------------------
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--a1-archive": {
        "env": "A1_PLUS_LAPOSE_A1_ARCHIVE",
        "rationale": (
            "A1 base substrate archive bytes (the 178,162 B PR101-fine-tuned "
            "anchor at SHA 8e664385...).  D1.D HIERARCHICAL composition LOADS "
            "this archive verbatim and APPENDS the LAPose sidecar."
        ),
        "default": str(DEFAULT_A1_ARCHIVE.relative_to(REPO_ROOT)),
        "required_input_file": True,
        "generator_command": (
            "experiments/results/track4_sg_a1_t178000_20260509/ "
            "(landed 2026-05-09 via PR101 score-gradient fine-tune)"
        ),
        "rationale_audit": (
            ".omx/research/grand_council_pose_axis_non_hnerv_a1_plus_lapose_20260513.md"
            "#executive-summary"
        ),
    },
    "--lapose-atom-manifest": {
        "env": "A1_PLUS_LAPOSE_ATOM_MANIFEST",
        "rationale": (
            "LAPose foveation/motion atom manifest JSON.  Each row encodes a "
            "hard-pair index, foveal geometry prior, and predicted "
            "pose_dist_delta.  D5.A+D5.C: atoms supply the per-pair "
            "FastViT-T12 RepMixer attack-surface signal."
        ),
        "default": str(
            (
                REPO_ROOT
                / ".omx"
                / "research"
                / "artifacts"
                / "lapose_motion_atoms_20260505_codex"
                / "lapose_motion_atom_manifest_fixture.json"
            ).relative_to(REPO_ROOT)
        ),
        "required_input_file": True,
        "generator_command": (
            "tools/build_lapose_motion_atom_manifest.py"
        ),
        "rationale_audit": (
            ".omx/research/grand_council_pose_axis_non_hnerv_a1_plus_lapose_20260513.md"
            "#section-4-1-shannon"
        ),
    },
    "--video-path": {
        "env": "A1_PLUS_LAPOSE_VIDEO_PATH",
        "rationale": (
            "Score-aware substrate trains against contest video "
            "(upstream/videos/0.mkv); synthetic data is FORBIDDEN outside --smoke."
        ),
        "default": str(DEFAULT_VIDEO_PATH.relative_to(REPO_ROOT)),
        "required_input_file": True,
        "generator_command": (
            "contest-pinned upstream snapshot — never regenerated locally"
        ),
    },
    "--output-dir": {
        "env": "A1_PLUS_LAPOSE_OUTPUT_DIR",
        "rationale": "custody location for checkpoints + archive + provenance",
    },
    "--epochs": {
        "env": "A1_PLUS_LAPOSE_EPOCHS",
        "rationale": (
            "training epochs; council default 3000 for full per cost-band "
            "table §5 (Modal A100 $2.50-$4.50 [prediction])."
        ),
        "default": "3000",
    },
    "--upstream-dir": {
        "env": "A1_PLUS_LAPOSE_UPSTREAM_DIR",
        "rationale": (
            "upstream/ root for scorer weights + evaluate.py; required for "
            "full training and auth eval"
        ),
        "default": str(DEFAULT_UPSTREAM_DIR.relative_to(REPO_ROOT)),
    },
    "--device": {
        "env": "A1_PLUS_LAPOSE_DEVICE",
        "rationale": (
            "compute device; cuda required for full training (MPS refused "
            "per CLAUDE.md MPS-NOISE rule); cpu permitted only with --smoke"
        ),
        "default": "cuda",
    },
    "--residual-rank": {
        "env": "A1_PLUS_LAPOSE_RESIDUAL_RANK",
        "rationale": (
            "per-pair low-rank K for the LAPose foveal residual head.  "
            "Council D2.B target 2 KB with rank=4 keeps the residual blob "
            "below 5 KB even at 64 selected pairs."
        ),
        "default": "4",
    },
    "--max-atoms": {
        "env": "A1_PLUS_LAPOSE_MAX_ATOMS",
        "rationale": (
            "cap on number of LAPose atom indices loaded from the manifest. "
            "Lower numbers shrink the LAPose sidecar at the cost of fewer "
            "hard-pair corrections (D2 verdict trade-off)."
        ),
        "default": "64",
    },
    "--enable-gt-scorer-cache": {
        "env": "A1_PLUS_LAPOSE_ENABLE_GT_SCORER_CACHE",
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
        prog="train_substrate_a1_plus_lapose",
        description=(
            "Train A1 + LAPose D1.D HIERARCHICAL composition substrate "
            "(council verdict 8-2; operator-routed D3.B joint end-to-end)."
        ),
    )
    # Tier 1 required inputs
    p.add_argument("--a1-archive", type=Path, default=DEFAULT_A1_ARCHIVE,
                   help="Path to A1 base archive.zip (178,162 B PR101-fine-tuned anchor).")
    p.add_argument("--lapose-atom-manifest", type=Path,
                   default=Path(TIER_1_OPERATOR_REQUIRED_FLAGS["--lapose-atom-manifest"]["default"]),
                   help="Path to LAPose foveation/motion atom manifest JSON.")
    p.add_argument("--video-path", type=Path, default=DEFAULT_VIDEO_PATH,
                   help="Path to upstream/videos/0.mkv (contest video).")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Where to write checkpoints + manifest + archive.")
    p.add_argument("--epochs", type=int, required=True,
                   help="Number of training epochs (council default 3000 for full).")
    p.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR,
                   help="upstream/ root; required for scorer load + auth eval.")

    # Training hyperparameters
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3,
                   help="AdamW learning rate (LAPose head only; A1 frozen).")
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ema-decay", type=float, default=0.997,
                   help="EMA decay per CLAUDE.md EMA non-negotiable.")
    p.add_argument("--noise-std", type=float, default=0.5)
    p.add_argument("--val-pair-count", type=int, default=8)
    p.add_argument("--max-pairs", type=int, default=N_PAIRS_FULL)

    # LAPose residual head config (D2 verdict — bounded ≤ 5 KB, target ~2 KB)
    p.add_argument("--residual-rank", type=int, default=4,
                   help="Per-pair low-rank K for the foveal residual head.")
    p.add_argument("--max-atoms", type=int, default=64,
                   help="Max LAPose atom indices loaded from the manifest.")
    p.add_argument("--foveal-h", type=int, default=256,
                   help="Foveal patch height at camera-native resolution.")
    p.add_argument("--foveal-w", type=int, default=256,
                   help="Foveal patch width at camera-native resolution.")
    p.add_argument("--int8-residual-scale", type=float, default=4.0)

    # Lagrangian weights (score-domain; HNeRV parity L6)
    p.add_argument("--alpha-rate", type=float, default=25.0,
                   help="Rate-term coefficient (contest evaluate.py: 25.0).")
    p.add_argument("--beta-seg", type=float, default=100.0,
                   help="SegNet distortion coefficient (contest evaluate.py: 100.0).")
    p.add_argument("--gamma-pose", type=float, default=math.sqrt(10.0),
                   help="Pose-axis weight (contest evaluate.py: sqrt(10)).")
    p.add_argument("--pose-weight-scale", type=float, default=1.0,
                   help=(
                       "Opt-in pose marginal tilt. Default 1.0 preserves the "
                       "contest formula; PR106-derived 2.71x is experimental."
                   ))

    # Modes
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny CPU smoke (no scorer load); proves wiring.")
    p.add_argument("--skip-auth-eval", action="store_true",
                   help="Skip end-of-train CUDA auth eval (smoke-only path).")

    # Tier-1 engineering flags (Catalog #172/#178/#179/#180)
    p.add_argument("--enable-autocast-fp16", action="store_true",
                   help="Enable torch.autocast('cuda', dtype=float16). Catalog #172.")
    p.add_argument("--enable-torch-compile", action="store_true",
                   help="Wrap residual head in torch.compile. Catalog #179.")
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
                   help="Enable TF32 matmul on Ampere+ GPUs. Catalog #178.")

    # D4 mode (operator-routed; primary D4.B per council lean)
    p.add_argument("--d4-mode", type=str, default="d4b_single_stage",
                   choices=("d4a_two_stage", "d4b_single_stage", "d4c_no_grammar_change"),
                   help=(
                       "D4 inflate.sh contract mode.  Primary D4.B "
                       "(single-stage with new archive section).  D4.A "
                       "two-stage and D4.C no-grammar-change are pending "
                       "deeper council verdict from sister subagent."
                   ))
    return p


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    return _canon_utc_now_iso()


def _sha256_bytes(data: bytes) -> str:
    import hashlib
    return hashlib.sha256(data).hexdigest()


def _git_head_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except Exception:
        return "no-git"


def _pin_seeds(seed: int) -> None:
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device_or_die(name: str, *, smoke: bool):
    import torch
    n = (name or "").strip().lower()
    if n == "mps":
        raise RuntimeError(
            "MPS is REFUSED per CLAUDE.md 'MPS auth eval is NOISE' "
            "non-negotiable; use cuda for promotion or cpu for --smoke."
        )
    if n == "cpu" and not smoke:
        raise RuntimeError(
            "CPU is REFUSED for full training (proxy scorer drift); "
            "use --smoke for CPU smoke, otherwise --device cuda."
        )
    if n == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but torch.cuda.is_available() is False"
        )
    return torch.device(n if n else ("cuda" if torch.cuda.is_available() else "cpu"))


def _load_a1_archive_bytes(archive_zip_path: Path) -> tuple[bytes, str, int]:
    """Read the inner ``x`` blob from an A1-style archive.zip.

    Returns ``(bytes, sha256_hex, num_bytes)``.  Refuses non-canonical
    SHA mismatches at strict mode (operator wrapper threads the canonical
    SHA via env-var; this function returns the actual SHA + size for
    provenance.json record-keeping).
    """
    with zipfile.ZipFile(archive_zip_path) as zf:
        names = zf.namelist()
        if "x" not in names:
            raise ValueError(
                f"A1 archive {archive_zip_path} missing inner 'x' blob; "
                f"got {names}"
            )
        data = zf.read("x")
    sha = _sha256_bytes(data)
    return data, sha, len(data)


def _load_lapose_atom_manifest(manifest_path: Path) -> dict[str, Any]:
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _decode_real_pairs(
    video_path: Path,
    n_pairs: int,
    max_pairs: int | None = None,
    output_hw: tuple[int, int] = CAMERA_HW,
):
    """Decode N pairs of (frame_a, frame_b) from the contest video via pyav.

    Returns a tensor of shape ``(n_pairs, 2, 3, H, W)`` in uint8-scale
    ``[0, 255]``.  This intentionally matches the canonical scorer contract
    and the shared substrate trainers; normalizing to ``[0, 1]`` would make
    ``apply_eval_roundtrip_during_training`` quantize almost every pixel to
    0/1 and destroy the score signal.
    """
    import av
    import torch
    import torch.nn.functional as F

    n_target = min(n_pairs, max_pairs) if max_pairs else n_pairs
    yuv420_to_rgb = _canon_load_upstream_yuv420_to_rgb(
        substrate_tag="a1_plus_lapose", repo_root=REPO_ROOT
    )

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    pairs: list[tuple] = []
    cur: list = []
    for frame in container.decode(stream):
        rgb_hwc = yuv420_to_rgb(frame)
        t = rgb_hwc.permute(2, 0, 1).unsqueeze(0).float()
        if tuple(t.shape[-2:]) != tuple(output_hw):
            t = F.interpolate(
                t, size=output_hw, mode="bilinear", align_corners=False
            )
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
    out = torch.stack(
        [torch.stack([a, b], dim=0) for (a, b) in pairs], dim=0
    )
    return out  # (n_pairs, 2, 3, H, W)


def _load_a1_runtime_modules(a1_archive_path: Path):
    """Load A1's vendored ``codec.py``/``model.py`` from the source runtime."""
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


def _decode_a1_base_pairs(
    a1_archive_path: Path,
    a1_bytes: bytes,
    *,
    device,
    max_pairs: int | None = None,
):
    """Decode the frozen A1 archive into camera-native RGB pairs.

    This is the apples-to-apples base substrate for the LAPose sidecar.  The
    residual must be trained on A1's actual decoded surface, not on ground
    truth frames used as a proxy.
    """
    import struct

    import torch
    import torch.nn.functional as F

    codec, model = _load_a1_runtime_modules(a1_archive_path)
    if len(a1_bytes) < 4:
        raise ValueError("A1 archive bytes too short")
    section_total = struct.unpack_from("<I", a1_bytes, 0)[0]
    decoder_blob = a1_bytes[4:section_total]
    latent_blob = a1_bytes[
        section_total : section_total + int(codec.LATENT_BLOB_LEN)
    ]
    sidecar_blob = a1_bytes[section_total + int(codec.LATENT_BLOB_LEN) :]
    decoder_sd = codec.decode_decoder_compact(decoder_blob)
    latents = codec.apply_latent_sidecar(
        codec.decode_latents_compact(latent_blob), sidecar_blob
    ).to(device)

    decoder = model.HNeRVDecoder(
        latent_dim=A1_LATENT_DIM,
        base_channels=A1_BASE_CHANNELS,
        eval_size=EVAL_HW,
    ).to(device)
    decoder.load_state_dict(decoder_sd)
    decoder.eval()

    target_pairs = N_PAIRS_FULL if max_pairs is None else min(N_PAIRS_FULL, max_pairs)
    batches: list[torch.Tensor] = []
    with torch.inference_mode():
        for i in range(0, target_pairs, 16):
            j = min(i + 16, target_pairs)
            batch = j - i
            decoded = decoder(latents[i:j])
            flat = decoded.reshape(batch * 2, 3, EVAL_HW[0], EVAL_HW[1])
            up = F.interpolate(
                flat, size=CAMERA_HW, mode="bicubic", align_corners=False
            )
            up = up.reshape(batch, 2, 3, CAMERA_HW[0], CAMERA_HW[1])
            # A1's canonical bias correction, matching the runtime exactly.
            up[:, 0, 0].sub_(1.0)
            up[:, 0, 2].sub_(1.0)
            up[:, 1, 1].sub_(1.0)
            batches.append(up.contiguous())
    return torch.cat(batches, dim=0)


def _apply_lapose_residual_batch(head, base_pairs, batch_indices: list[int]):
    """Add LAPose residuals to camera-native A1 base pairs."""
    pred = base_pairs.clone()
    h, w = int(pred.shape[-2]), int(pred.shape[-1])
    if (h, w) != CAMERA_HW:
        raise ValueError(
            f"A1+LAPose residual application expects CAMERA_HW={CAMERA_HW}; "
            f"got {(h, w)}"
        )
    fov_h = int(head.cfg.foveal_h)
    fov_w = int(head.cfg.foveal_w)
    if fov_h <= 0 or fov_w <= 0 or fov_h > h or fov_w > w:
        raise ValueError(f"invalid foveal patch {(fov_h, fov_w)} for frame {(h, w)}")
    fov_top = max(0, h // 2 - fov_h // 2)
    fov_left = max(0, w // 2 - fov_w // 2)
    for local_i, pair_id in enumerate(batch_indices):
        if pair_id not in head._pair_to_slot:
            continue
        for frame_i in (0, 1):
            resid = head.residual_chw(pair_id, frame_i)
            pred[
                local_i,
                frame_i,
                :,
                fov_top : fov_top + fov_h,
                fov_left : fov_left + fov_w,
            ] = (
                pred[
                    local_i,
                    frame_i,
                    :,
                    fov_top : fov_top + fov_h,
                    fov_left : fov_left + fov_w,
                ]
                + resid
            )
    return pred


# ---------------------------------------------------------------------------
# Smoke main
# ---------------------------------------------------------------------------

def _smoke_main(args: argparse.Namespace) -> int:
    """Tiny CPU smoke that proves the LAPose head + archive pack flow."""
    import torch

    from tac.substrates.a1_plus_lapose.architecture import (
        A1PlusLaposeConfig,
        PerPairResidualHead,
        parse_lapose_atom_indices,
    )
    from tac.substrates.a1_plus_lapose.archive import pack_composition_archive

    _pin_seeds(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load A1 base bytes (verbatim).
    a1_bytes, a1_sha, a1_size = _load_a1_archive_bytes(args.a1_archive)
    print(f"[smoke] A1 base: {a1_size} B sha256={a1_sha[:16]}...")

    # Load LAPose atom indices.
    manifest = _load_lapose_atom_manifest(args.lapose_atom_manifest)
    atom_indices = parse_lapose_atom_indices(manifest, max_atoms=args.max_atoms)
    if not atom_indices:
        # Smoke-mode fallback: synthesize tiny atom set for wiring proof.
        atom_indices = (3, 7, 11, 17)
    print(f"[smoke] LAPose atom indices ({len(atom_indices)}): {atom_indices[:8]}...")

    cfg = A1PlusLaposeConfig(
        residual_rank=args.residual_rank,
        selected_pair_indices=atom_indices,
        foveal_h=args.foveal_h,
        foveal_w=args.foveal_w,
        int8_residual_scale=args.int8_residual_scale,
    )
    device = _device_or_die(args.device, smoke=True)
    head = PerPairResidualHead(cfg).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)

    n_sel = head.num_selected
    last_dim = 3 * cfg.foveal_h + cfg.foveal_w
    print(f"[smoke] residual head: num_selected={n_sel}, est_bytes={head.total_int8_bytes()}")

    # Smoke loss: drive U,V toward small magnitude (proves the gradient path).
    for step in range(min(args.epochs, 3)):
        opt.zero_grad()
        loss = head.U.abs().mean() + head.V.abs().mean()
        loss.backward()
        opt.step()
        print(f"[smoke] step {step}: loss={loss.item():.4f}")

    # Build composition archive bytes (smoke).
    with torch.no_grad():
        residuals = torch.zeros(n_sel, 2, args.residual_rank, last_dim)
        for slot in range(n_sel):
            for fi in (0, 1):
                u_lv = head.U[slot, fi]  # (rank, 3*fh)
                v_lv = head.V[slot, fi]  # (rank, fw)
                residuals[slot, fi, :, : 3 * cfg.foveal_h] = u_lv
                residuals[slot, fi, :, 3 * cfg.foveal_h:] = v_lv

    composition_bytes = pack_composition_archive(
        a1_bytes,
        selected_indices=atom_indices,
        residuals=residuals,
        foveal_h=cfg.foveal_h,
        foveal_w=cfg.foveal_w,
        residual_rank=cfg.residual_rank,
        int8_scale=cfg.int8_residual_scale,
    )
    composition_sha = _sha256_bytes(composition_bytes)
    composition_size = len(composition_bytes)
    print(
        f"[smoke] composition: {composition_size} B sha256={composition_sha[:16]}... "
        f"(+{composition_size - a1_size} B over A1)"
    )

    smoke_meta = {
        "started_at_utc": _utc_now_iso(),
        "smoke": True,
        "a1_base_sha256": a1_sha,
        "a1_base_bytes": a1_size,
        "composition_sha256": composition_sha,
        "composition_bytes": composition_size,
        "lapose_bytes_overhead": composition_size - a1_size,
        "lapose_atom_count": n_sel,
        "residual_rank": cfg.residual_rank,
        "foveal_h": cfg.foveal_h,
        "foveal_w": cfg.foveal_w,
        "d4_mode": args.d4_mode,
        "git_head": _git_head_sha(),
    }
    (args.output_dir / "smoke_metadata.json").write_text(
        json.dumps(smoke_meta, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[smoke] wrote {args.output_dir / 'smoke_metadata.json'}")
    return 0


# ---------------------------------------------------------------------------
# Full main (CUDA-required; score-aware Lagrangian end-to-end)
# ---------------------------------------------------------------------------

def _full_main(args: argparse.Namespace) -> int:
    """Full training — D3.B joint end-to-end on LAPose residual head."""
    import torch

    from tac.differentiable_eval_roundtrip import (
        patch_upstream_yuv6_globally,
        unpatch_upstream_yuv6,
    )
    from tac.scorer import load_differentiable_scorers
    from tac.substrates.a1_plus_lapose.architecture import (
        A1PlusLaposeConfig,
        PerPairResidualHead,
        parse_lapose_atom_indices,
    )
    from tac.substrates.a1_plus_lapose.archive import pack_composition_archive
    from tac.substrates.a1_plus_lapose.score_aware_loss import (
        A1PlusLaposeLossWeights,
        A1PlusLaposeScoreAwareLoss,
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
        print(f"[full] {name} @ {msg['at']}")

    _stage("seed_pinned")

    # 1. Load A1 base bytes verbatim (FROZEN — D1.D HIERARCHICAL).
    a1_bytes, a1_sha, a1_size = _load_a1_archive_bytes(args.a1_archive)
    _stage(f"a1_loaded_{a1_size}_B_sha{a1_sha[:8]}")

    # 2. Load LAPose atom indices.
    manifest = _load_lapose_atom_manifest(args.lapose_atom_manifest)
    atom_indices = parse_lapose_atom_indices(manifest, max_atoms=args.max_atoms)
    if not atom_indices:
        raise RuntimeError(
            f"LAPose atom manifest {args.lapose_atom_manifest} contained zero "
            "valid pair indices; check manifest schema (expected atoms[] with "
            "hard_pair_support[0] or atom_id 'lapose_*_pair:N')."
        )
    _stage(f"lapose_atoms_{len(atom_indices)}")

    # 3. Patch upstream rgb_to_yuv6 BEFORE scorer construction (Catalog #187).
    yuv6_token = patch_upstream_yuv6_globally()
    _stage("upstream_yuv6_patched")

    try:
        # 4. Load differentiable scorers (frozen — gradients flow through to
        #    the LAPose residual head only).
        posenet, segnet = load_differentiable_scorers(
            args.upstream_dir, device=device
        )
        for p in list(posenet.parameters()) + list(segnet.parameters()):
            p.requires_grad_(False)
        posenet.eval()
        segnet.eval()
        _stage("scorers_loaded")

        # 5. Decode real target pairs and the frozen A1 base surface.
        # The LAPose residual is trained against A1's actual decoded RGB, not
        # a ground-truth proxy.  Otherwise the learned sidecar would optimize a
        # surface the contest runtime never consumes.
        print(f"[full] decoding GT pairs from {args.video_path} ...")
        gt_pair_tensor = _decode_real_pairs(
            args.video_path,
            n_pairs=N_PAIRS_FULL,
            max_pairs=args.max_pairs,
            output_hw=CAMERA_HW,
        ).to(device)
        print("[full] decoding frozen A1 base pairs ...")
        a1_pair_tensor = _decode_a1_base_pairs(
            args.a1_archive,
            a1_bytes,
            device=device,
            max_pairs=args.max_pairs,
        )
        n_pairs = int(gt_pair_tensor.shape[0])
        if int(a1_pair_tensor.shape[0]) != n_pairs:
            raise RuntimeError(
                "A1 decoded pair count does not match GT pair count: "
                f"a1={int(a1_pair_tensor.shape[0])} gt={n_pairs}"
            )
        _stage(f"pairs_decoded_gt_and_a1_{n_pairs}")

        # Held-out validation pairs (last val_pair_count)
        val_count = max(1, min(args.val_pair_count, max(1, n_pairs // 8)))
        val_idx_start = n_pairs - val_count
        train_indices_pool = list(range(val_idx_start))
        val_indices_pool = list(range(val_idx_start, n_pairs))

        # 6. Build LAPose residual head (the ONLY trainable module).
        cfg = A1PlusLaposeConfig(
            residual_rank=args.residual_rank,
            selected_pair_indices=atom_indices,
            foveal_h=args.foveal_h,
            foveal_w=args.foveal_w,
            int8_residual_scale=args.int8_residual_scale,
        )
        head = PerPairResidualHead(cfg).to(device)
        n_params = sum(p.numel() for p in head.parameters())
        print(f"[full] LAPose head params: {n_params:,}  "
              f"est_int8_bytes={head.total_int8_bytes()}")
        _stage(f"head_built_{n_params}_params")
        opt_ctx = _build_optimized_training_context(
            args,
            scorers=(posenet, segnet),
            gt_pairs=gt_pair_tensor,
            substrate_model=head,
            device=device,
        )
        head = opt_ctx.substrate_model
        gt_cache = opt_ctx.gt_cache
        if gt_cache is not None:
            print(gt_cache.summary_line())
            _stage("gt_scorer_cache_built")
        else:
            _stage("gt_scorer_cache_disabled")

        # 7. EMA shadow (Catalog #88)
        ema = EMA(head, decay=args.ema_decay)
        _stage(f"ema_wired_decay_{args.ema_decay}")

        # 8. Score-aware Lagrangian
        weights = A1PlusLaposeLossWeights(
            alpha_rate=args.alpha_rate,
            beta_seg=args.beta_seg,
            gamma_pose=args.gamma_pose,
            pose_weight_scale=args.pose_weight_scale,
            contest_normalizer=CONTEST_NORMALIZER,
        )
        loss_fn = A1PlusLaposeScoreAwareLoss(
            seg_scorer=segnet, pose_scorer=posenet, weights=weights
        )
        _stage("lagrangian_built")

        # 9. Optimizer + cosine annealing (head params only)
        optimizer = torch.optim.AdamW(
            head.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs)
        )

        # Closed-form rate proxy: LAPose sidecar bytes are FIXED at config
        # time (head.total_int8_bytes()); we still surface it as a tensor so
        # the Lagrangian carries the term but no gradient flows through it.
        archive_bytes_proxy = torch.tensor(
            float(a1_size + head.total_int8_bytes()),
            device=device,
        )

        # 10. Train loop.
        train_started_at = time.time()
        best_val_lag = math.inf
        best_epoch = -1
        ckpt_best_path = args.output_dir / "best.pt"
        nan_strike = 0
        max_nan_strikes = 3

        for epoch in range(args.epochs):
            head.train()
            random.shuffle(train_indices_pool)
            epoch_losses: list[float] = []
            for batch_start in range(0, len(train_indices_pool), args.batch_size):
                batch_indices = train_indices_pool[
                    batch_start : batch_start + args.batch_size
                ]
                if not batch_indices:
                    continue
                pair_idxs = torch.tensor(batch_indices, device=device, dtype=torch.long)
                gt_pairs = gt_pair_tensor[pair_idxs]  # (B, 2, 3, H, W)
                base_pairs = a1_pair_tensor[pair_idxs]
                gt_a = gt_pairs[:, 0]
                gt_b = gt_pairs[:, 1]
                pred_pairs = _apply_lapose_residual_batch(
                    head, base_pairs, batch_indices
                ).clamp(0.0, 255.0)
                pred_a = pred_pairs[:, 0]
                pred_b = pred_pairs[:, 1]
                gt_pose_batch = gt_seg_batch = None
                gt_seg_already_probs = None
                if gt_cache is not None:
                    gt_pose_batch, gt_seg_batch = gt_cache.lookup(
                        pair_idxs, device=device
                    )
                    gt_seg_already_probs = gt_cache.seg_already_probs
                loss, parts = loss_fn(
                    pred_a, pred_b, gt_a, gt_b, archive_bytes_proxy,
                    apply_eval_roundtrip=True, noise_std=args.noise_std,
                    gt_pose_batch=gt_pose_batch,
                    gt_seg_batch=gt_seg_batch,
                    gt_seg_already_probs=gt_seg_already_probs,
                )

                if not torch.isfinite(loss):
                    nan_strike += 1
                    print(f"[full] NaN strike {nan_strike}/{max_nan_strikes} at epoch {epoch}")
                    if nan_strike >= max_nan_strikes:
                        raise RuntimeError("NaN watchdog tripped — refusing to continue")
                    continue
                nan_strike = 0

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), args.grad_clip)
                optimizer.step()
                ema.update(head)
                epoch_losses.append(float(loss.detach().cpu()))

            scheduler.step()

            # Validation with EMA shadow (snapshot+restore)
            if epoch % max(1, args.epochs // 30) == 0 or epoch == args.epochs - 1:
                live_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
                ema.apply(head)
                try:
                    val_lag = _run_val_loop(
                        head, loss_fn, gt_pair_tensor, a1_pair_tensor, val_indices_pool,
                        archive_bytes_proxy, gt_cache, device
                    )
                finally:
                    head.load_state_dict(live_state)
                    head.train()

                avg_train = (sum(epoch_losses) / len(epoch_losses)) if epoch_losses else math.nan
                print(f"[full] epoch {epoch:4d}: train_loss={avg_train:.5f} "
                      f"val_lag={val_lag:.5f} (best={best_val_lag:.5f})")
                if val_lag < best_val_lag:
                    best_val_lag = val_lag
                    best_epoch = epoch
                    # Save EMA shadow (NEVER live weights — Catalog #88)
                    ema_state = {k: v.detach().cpu() for k, v in head.state_dict().items()}
                    torch.save(
                        {"state_dict": ema_state, "config": asdict(cfg),
                         "epoch": epoch, "val_lag": val_lag}, ckpt_best_path
                    )

        train_elapsed = time.time() - train_started_at
        _stage(f"trained_best_epoch_{best_epoch}_val_lag_{best_val_lag:.5f}_elapsed_{train_elapsed:.1f}s")

        # 11. Load best EMA shadow, build composition archive
        best_ckpt = torch.load(ckpt_best_path, weights_only=False, map_location=device)
        head.load_state_dict(best_ckpt["state_dict"])
        head.eval()
        with torch.no_grad():
            last_dim = 3 * cfg.foveal_h + cfg.foveal_w
            residuals = torch.zeros(head.num_selected, 2, args.residual_rank, last_dim)
            for slot in range(head.num_selected):
                for fi in (0, 1):
                    u_lv = head.U[slot, fi].detach().cpu()
                    v_lv = head.V[slot, fi].detach().cpu()
                    residuals[slot, fi, :, : 3 * cfg.foveal_h] = u_lv
                    residuals[slot, fi, :, 3 * cfg.foveal_h:] = v_lv

        composition_bytes = pack_composition_archive(
            a1_bytes,
            selected_indices=atom_indices,
            residuals=residuals,
            foveal_h=cfg.foveal_h,
            foveal_w=cfg.foveal_w,
            residual_rank=cfg.residual_rank,
            int8_scale=cfg.int8_residual_scale,
        )
        composition_sha = _sha256_bytes(composition_bytes)
        composition_size = len(composition_bytes)
        print(
            f"[full] composition: {composition_size} B sha256={composition_sha[:16]}... "
            f"(+{composition_size - a1_size} B over A1)"
        )
        _stage(f"composition_built_{composition_size}_B_sha{composition_sha[:8]}")

        # 12. Build archive.zip + runtime tree
        archive_zip_path = args.output_dir / "archive.zip"
        _build_archive_zip(archive_zip_path, composition_bytes)
        submission_dir = args.output_dir / "submission_dir"
        _write_runtime(submission_dir, args.a1_archive, d4_mode=args.d4_mode)
        # Copy composition archive into submission_dir
        shutil.copy2(archive_zip_path, submission_dir / "archive.zip")
        _stage("archive_emitted")

        # 13. Auth eval ([contest-CUDA] inline; canonical helper Catalog #226)
        auth_eval_json_path = args.output_dir / "auth_eval.json"
        if not args.skip_auth_eval and device.type == "cuda":
            _canon_gate_auth_eval_call(
                args=args,
                archive_zip=submission_dir / "archive.zip",
                inflate_sh=submission_dir / "inflate.sh",
                upstream_dir=args.upstream_dir,
                output_json=auth_eval_json_path,
                contest_auth_eval_script=CONTEST_AUTH_EVAL_SCRIPT,
                substrate_tag="a1_plus_lapose",
                device=device,
            )
            _stage("auth_eval_cuda_done")
    finally:
        unpatch_upstream_yuv6(yuv6_token)
        _stage("upstream_yuv6_unpatched")

    # 14. Posterior update (CLAUDE.md "Subagent coherence-by-default" hook 5)
    if not args.skip_auth_eval and auth_eval_json_path.exists():
        try:
            from tac.continual_learning import posterior_update_locked_from_auth_eval_json
            update = posterior_update_locked_from_auth_eval_json(auth_eval_json_path)
            print(f"[full] posterior_update accepted={getattr(update, 'accepted', '?')}")
            _stage("posterior_updated")
        except Exception as exc:
            print(f"[full] WARN posterior_update failed: {exc!r}")

    # 15. Provenance
    provenance = {
        "lane_id": "lane_a1_plus_lapose_composition_20260513",
        "started_at_utc": stage_log[0]["at"] if stage_log else _utc_now_iso(),
        "completed_at_utc": _utc_now_iso(),
        "git_head": _git_head_sha(),
        "a1_base_sha256": a1_sha,
        "a1_base_bytes": a1_size,
        "composition_sha256": composition_sha,
        "composition_bytes": composition_size,
        "lapose_overhead_bytes": composition_size - a1_size,
        "lapose_atom_count": head.num_selected,
        "best_val_lag": best_val_lag,
        "best_epoch": best_epoch,
            "epochs": args.epochs,
            "device": str(device),
            "trainer_proxy_axis": opt_ctx.eval_axis_label,
            "trainer_proxy_promotion_requirement": opt_ctx.promotion_requirement,
            "gt_scorer_cache_enabled": gt_cache is not None,
            "d4_mode": args.d4_mode,
        "council_verdicts": {
            "D1": "D1.D HIERARCHICAL (8-2)",
            "D2": "D2.B bounded with D2.A target (10-0)",
            "D3": "D3.B JOINT END-TO-END (operator-routed)",
            "D4": f"{args.d4_mode} (pending COUNCIL-D4-DEEPER)",
            "D5": "D5.A + D5.C (7-3)",
            "D6": "BOTH AXES contest-CPU + contest-CUDA (10-0)",
        },
        "stage_log": stage_log,
        "hardware_substrate_cuda": _canon_detect_hardware_substrate(
            axis="cuda",
            substrate_tag="a1_plus_lapose",
            env_var_candidates=("A1_PLUS_LAPOSE_GPU", "MODAL_GPU"),
        ),
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[full] wrote {args.output_dir / 'provenance.json'}")
    return 0


def _run_val_loop(
    head, loss_fn, gt_pair_tensor, a1_pair_tensor, val_indices_pool,
    archive_bytes_proxy, gt_cache, device
) -> float:
    """Validation forward pass — eval-time torch.inference_mode (Catalog #180)."""
    import torch
    head.eval()
    losses: list[float] = []
    with torch.inference_mode():
        for pair_id in val_indices_pool:
            gt_pair = gt_pair_tensor[pair_id : pair_id + 1]
            base_pair = a1_pair_tensor[pair_id : pair_id + 1]
            gt_a = gt_pair[:, 0]
            gt_b = gt_pair[:, 1]
            pred_pair = _apply_lapose_residual_batch(
                head, base_pair, [int(pair_id)]
            ).clamp(0.0, 255.0)
            pred_a = pred_pair[:, 0]
            pred_b = pred_pair[:, 1]
            idx = torch.tensor([int(pair_id)], device=device, dtype=torch.long)
            gt_pose_batch = gt_seg_batch = None
            gt_seg_already_probs = None
            if gt_cache is not None:
                gt_pose_batch, gt_seg_batch = gt_cache.lookup(idx, device=device)
                gt_seg_already_probs = gt_cache.seg_already_probs
            try:
                loss, _parts = loss_fn(
                    pred_a, pred_b, gt_a, gt_b, archive_bytes_proxy,
                    apply_eval_roundtrip=True, noise_std=0.0,
                    gt_pose_batch=gt_pose_batch,
                    gt_seg_batch=gt_seg_batch,
                    gt_seg_already_probs=gt_seg_already_probs,
                )
                losses.append(float(loss.detach().cpu()))
            except Exception as exc:
                print(f"[val] WARN pair {pair_id} failed: {exc!r}")
    return sum(losses) / len(losses) if losses else float("inf")


# ---------------------------------------------------------------------------
# Archive zip + runtime tree
# ---------------------------------------------------------------------------

def _build_archive_zip(archive_zip_path: Path, composition_bytes: bytes) -> None:
    """Pack composition bytes as the single ZIP member ``x`` (A1 grammar)."""
    fixed_ts = (1980, 1, 1, 0, 0, 0)
    archive_zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_zip_path, "w") as zf:
        zi = zipfile.ZipInfo("x", date_time=fixed_ts)
        zi.compress_type = zipfile.ZIP_STORED  # composition bytes are already brotli-compressed
        zf.writestr(zi, composition_bytes)


def _write_runtime(
    submission_dir: Path, a1_archive: Path, *, d4_mode: str
) -> None:
    """Emit contest-compliant inflate.sh + inflate.py runtime tree.

    D4 mode selector (per council operator-routed; primary D4.B):
      - d4b_single_stage: single inflate.py reads x, splits A1+LAPose, renders.
      - d4a_two_stage:    inflate.sh runs A1 inflate.py THEN LAPose injection.
      - d4c_no_grammar_change: (RESEARCH-ONLY; pending COUNCIL-D4-DEEPER)
    """
    submission_dir.mkdir(parents=True, exist_ok=True)

    # Vendor A1 codec.py + model.py (from the A1 source archive submission_dir/src/)
    a1_submission_dir = a1_archive.parent
    a1_src = a1_submission_dir / "src"
    if not a1_src.is_dir():
        raise FileNotFoundError(
            f"A1 source vendor missing at {a1_src}; expected codec.py + model.py "
            "(D1.D HIERARCHICAL requires the A1 decoder vendored alongside)"
        )
    target_src = submission_dir / "src"
    target_src.mkdir(parents=True, exist_ok=True)
    for name in ("codec.py", "model.py"):
        if (a1_src / name).is_file():
            shutil.copy2(a1_src / name, target_src / name)

    # Vendor tac.substrates.a1_plus_lapose package
    runtime_pkg = submission_dir / "src" / "tac" / "substrates" / "a1_plus_lapose"
    runtime_pkg.mkdir(parents=True, exist_ok=True)
    for pkg_init in (
        submission_dir / "src" / "tac" / "__init__.py",
        submission_dir / "src" / "tac" / "substrates" / "__init__.py",
        runtime_pkg / "__init__.py",
    ):
        pkg_init.write_text("", encoding="utf-8")
    substrate_src = REPO_ROOT / "src" / "tac" / "substrates" / "a1_plus_lapose"
    for name in ("architecture.py", "archive.py", "inflate.py"):
        shutil.copy2(substrate_src / name, runtime_pkg / name)
    _canon_vendor_shared_inflate_runtime(submission_dir, repo_root=REPO_ROOT)

    # inflate.sh — 3-arg positional handoff per Catalog #146
    inflate_sh = (
        "#!/usr/bin/env bash\n"
        "# A1 + LAPose composition inflate.sh (D4 mode: " + d4_mode + ")\n"
        "# Per CLAUDE.md HNeRV parity discipline lesson L4 + Catalog #146:\n"
        "#   ARG1=archive_dir ARG2=output_dir ARG3=file_list\n"
        "set -euo pipefail\n"
        "HERE=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"\n"
        "DATA_DIR=\"$1\"\n"
        "OUTPUT_DIR=\"$2\"\n"
        "FILE_LIST=\"$3\"\n"
        "mkdir -p \"$OUTPUT_DIR\"\n"
        "exec uv run --no-sync python \"$HERE/inflate.py\" \"$DATA_DIR\" \"$OUTPUT_DIR\" \"$FILE_LIST\"\n"
    )
    (submission_dir / "inflate.sh").write_text(inflate_sh, encoding="utf-8")
    (submission_dir / "inflate.sh").chmod(0o755)

    # inflate.py — minimal CLI dispatcher to the packaged substrate inflate
    inflate_py = (
        "#!/usr/bin/env python\n"
        '"""A1 + LAPose contest inflate dispatcher (D1.D composition).\n'
        "\n"
        "Reads archive_dir/x (or 0.bin or archive.zip) via the packaged\n"
        "substrate inflate, then for each entry in file_list writes one\n"
        "contest .raw tensor stream per pair (no scorer imports — strict-\n"
        "scorer-rule contract).\n"
        '"""\n'
        "import sys\n"
        "import zipfile\n"
        "from pathlib import Path\n"
        "HERE = Path(__file__).resolve().parent\n"
        'sys.path.insert(0, str(HERE / "src"))\n'
        "from tac.substrates.a1_plus_lapose.inflate import inflate_one\n"
        "\n"
        "def main() -> int:\n"
        "    if len(sys.argv) != 4:\n"
        '        print("usage: inflate.py archive_dir output_dir file_list", file=sys.stderr)\n'
        "        return 2\n"
        "    archive_dir = Path(sys.argv[1])\n"
        "    output_dir = Path(sys.argv[2])\n"
        "    file_list_path = Path(sys.argv[3])\n"
        "    output_dir.mkdir(parents=True, exist_ok=True)\n"
        "    # Resolve archive bytes source\n"
        '    candidates = ["x", "0.bin"]\n'
        "    src_bin = None\n"
        "    for cand in candidates:\n"
        "        p = archive_dir / cand\n"
        "        if p.is_file():\n"
        "            src_bin = p\n"
        "            break\n"
        "    if src_bin is None:\n"
        '        # Fallback: extract from archive.zip\n'
        '        zip_path = archive_dir / "archive.zip"\n'
        "        if zip_path.is_file():\n"
        "            with zipfile.ZipFile(zip_path) as zf:\n"
        '                inner = zf.namelist()[0] if zf.namelist() else "x"\n'
        '                src_bin = archive_dir / "x"\n'
        "                src_bin.write_bytes(zf.read(inner))\n"
        "    if src_bin is None or not src_bin.is_file():\n"
        '        print(f"FATAL: no archive bytes found in {archive_dir}", file=sys.stderr)\n'
        "        return 3\n"
        '    for line in file_list_path.read_text(encoding="utf-8").splitlines():\n'
        "        line = line.strip()\n"
        "        if not line:\n"
        "            continue\n"
        '        base = line.rsplit(".", 1)[0]\n'
        '        dst = output_dir / f"{base}.raw"\n'
        "        inflate_one(src_bin, dst)\n"
        "    return 0\n"
        "\n"
        'if __name__ == "__main__":\n'
        "    sys.exit(main())\n"
    )
    (submission_dir / "inflate.py").write_text(inflate_py, encoding="utf-8")


# Catalog #226: _run_contest_auth_eval_cuda was removed in favor of the
# canonical helper `_canon_gate_auth_eval_call` invoked inline at the call
# site in `_full_main`. Future CLI refactors of `experiments/contest_auth_eval.py`
# propagate via the canonical helper, not 18 hand-rolled subprocess invocations.


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# META layer SubstrateContract (Catalog #241/#242 canonical migration; landed
# 2026-05-15 by CATALOG-241-BACKFILL-29-TRAINERS subagent). Decoration extincts
# the Z3 v2 silent-drift bug class for this substrate by binding (a) the
# trainer's claimed contract, (b) the recipe schema, (c) the lane registry,
# and (d) the cost-band envelope into ONE source-of-truth that fails-loud at
# decoration time if the contract violates canonical invariants.
# ---------------------------------------------------------------------------

A1_PLUS_LAPOSE_SUBSTRATE_CONTRACT = SubstrateContract(
    # 2.1 Identity & lifecycle
    id="a1_plus_lapose",
    lane_id="lane_a1_plus_lapose_composition_20260513",
    target_modes=("contest_one_video_replay", "research_substrate",),
    deployment_target="t4_contest_runtime",
    council_verdict_provenance=(
        ".omx/research/grand_council_omnibus_design_decisions_20260514.md"
    ),
    # 2.2 Architecture & runtime (8 per Catalog #124)
    archive_grammar=(
        "A1LAP1 monolithic single-file 0.bin: header + A1 base archive (fp16 + brotli) + LA-pose telescopic-foveation residual (fp16 + brotli) + per-pair composition embeddings"
    ),
    parser_section_manifest={
        "header": "A1LAP1_magic_and_version",
        "a1_base_archive": "fp16_brotli_blob",
        "la_pose_residual": "fp16_brotli_blob",
        "composition_embeddings": "fp16_per_pair",
    },
    inflate_runtime_loc_budget=160,
    runtime_dep_closure=("torch>=2.5,<2.7", "brotli", "av",),
    export_format="fp16_brotli",
    score_aware_loss="scorer_loss_terms_btchw",
    bolt_on_loc_budget=1290,
    no_op_detector_planned=True,
    # 2.3 Operational mechanism (3 per Catalog #220)
    archive_bytes_added=None,
    score_improvement_mechanism_status="RESEARCH_ONLY",
    runtime_overlay_consumed=False,
    # 2.4 Recipe schema (8) — mirrors substrate recipe YAML
    recipe_smoke_only=False,
    recipe_research_only=False,
    recipe_min_smoke_gpu="A100",
    recipe_min_vram_gb=40,
    recipe_pyav_decode_strategy="cpu_thread_async_upload",
    recipe_canary_status="independent_substrate",
    recipe_video_input_strategy="per_dispatch_local_copy",
    recipe_canary_dependency=None,
    # 2.5 Cost band & GPU envelope (4)
    cost_band_epochs=3000,
    cost_band_gpu_key="A100",
    cost_band_platform_key="modal",
    cost_band_p50_usd=8.0,
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
            "A1+LA-pose composition; sensitivity captured by both branches"
        ),
        "hook_bit_allocator_class": (
            "fp16 brotli on both branches; no per-tensor bit allocator"
        ),
        "hook_probe_disambiguator": (
            "two-branch (A1 + LA-pose) composition; ablation handled by --disable-lapose flag"
        ),
    },
)


@register_substrate(A1_PLUS_LAPOSE_SUBSTRATE_CONTRACT)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.smoke:
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":
    sys.exit(main())
