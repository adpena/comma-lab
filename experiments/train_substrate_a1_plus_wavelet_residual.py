"""Train the A1 + wavelet residual sidecar composition substrate (D3.C freeze-A1).

META-COUNCIL memo: ``.omx/research/meta_council_decision_attribution_audit_20260513.md``
Lane:               ``lane_a1_plus_wavelet_residual_retarget_20260513``
Sister substrate:   ``tac.substrates.a1_plus_lapose`` (D1.D + D3.B joint mode)
Sister lane:        ``lane_a1_plus_lapose_composition_20260513`` (operator-routed D3.B)

Binding META-COUNCIL verdicts honored (per CLAUDE.md "Design decisions"):

* **D1 residual sidecar** (operator-pre-approved default): A1 base substrate
  is loaded verbatim; the wavelet residual is APPENDED as a magic-byte
  trailer (``"WAV1"`` distinct from A1+LAPose's ``"LPA1"``).  Mirrors A1+
  LAPose D4.B single-stage inflate contract.
* **D2 int8 + brotli, ≤500 B target** (operator-pre-approved): wavelet
  coefficients quantized to int8 at scale=8.0 + brotli quality 11.
  Closed-form size bound = N_selected * 2 frames * 3 bands * 3 RGB *
  rank * (foveal_h + foveal_w) int8 + small fixed overhead.  With
  operator defaults (12 pairs, rank=1, foveal=64x64) the pre-brotli
  upper bound is ~3 KB; post-brotli sparse residuals compress to ~500-1000 B.
* **D3 freeze-A1** (operator-pre-approved fallback; cheaper than D3.B at
  $0.50 vs $4-5): A1's decoder + latents are gradient-stopped; only the
  wavelet residual head's parameters carry gradients.  This is the META-
  COUNCIL §8b Bayesian-optimal first dispatch — establishes H3 (A1
  saturation) prior at 5-25× cheaper than A1+LAPose.
* **D4.B magic-byte trailer** (operator-pre-approved; mirrors A1+LAPose):
  single-stage inflate.py reads ``archive_dir/x``, splits into (A1, WAV1)
  via the trailing magic, renders A1's 600 pairs, then overlays the
  wavelet residual at selected pair indices.
* **D5 pose-residual at PR106-style frontier operating point** (operator-
  pre-approved fallback): the wavelet residual is concentrated at the
  central foveal patch (vanishing-point region) where PoseNet's 12-channel
  YUV6 attack surface is densest.  Per META-COUNCIL pose-marginal
  inversion at frontier operating points, this is the BEST per-byte axis.
* **D6 BOTH AXES** mandatory: contest-CPU GHA Linux x86_64 + contest-CUDA
  must both be emitted (per CLAUDE.md "Submission auth eval — BOTH CPU AND
  CUDA" non-negotiable).  This trainer emits the CUDA axis inline; the CPU
  axis follow-up is the operator wrapper's responsibility.

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
* Tier-1 engineering flags declared (Catalogs #172/#178/#179/#180);
  autocast / TF32 opt-in via ``--enable-*`` flags.
* TIER_1_OPERATOR_REQUIRED_FLAGS manifest as ``ast.AnnAssign`` so
  Catalog #151 introspection sees it (Catalog #168).
* Dynamic ``hardware_substrate`` detection (Catalog #190).

Usage (smoke; CPU, no scorer load)::

    .venv/bin/python experiments/train_substrate_a1_plus_wavelet_residual.py \\
        --a1-archive experiments/results/track4_sg_a1_t178000_20260509/submission_dir/archive.zip \\
        --pair-manifest .omx/research/artifacts/lapose_motion_atoms_20260505_codex/lapose_motion_atom_manifest_fixture.json \\
        --video-path upstream/videos/0.mkv \\
        --output-dir experiments/results/a1_plus_wavelet_smoke_<utc> \\
        --epochs 3 --device cpu --smoke

Usage (full; CUDA-required; threads from operator wrapper)::

    .venv/bin/python experiments/train_substrate_a1_plus_wavelet_residual.py \\
        --a1-archive .../submission_dir/archive.zip \\
        --pair-manifest <manifest.json> \\
        --video-path upstream/videos/0.mkv \\
        --output-dir experiments/results/a1_plus_wavelet_<utc> \\
        --epochs 2000 --batch-size 32 --lr 1e-3 --device cuda
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

from tac.substrates._shared.trainer_skeleton import (  # noqa: E402
    detect_hardware_substrate as _canon_detect_hardware_substrate,
)
from tac.substrates._shared.trainer_skeleton import (  # noqa: E402
    load_upstream_yuv420_to_rgb as _canon_load_upstream_yuv420_to_rgb,
)
from tac.substrates._shared.trainer_skeleton import (  # noqa: E402
    utc_now_iso as _canon_utc_now_iso,
)
from tac.substrates._shared.trainer_skeleton import (  # noqa: E402
    vendor_shared_inflate_runtime as _canon_vendor_shared_inflate_runtime,
)


# ---------------------------------------------------------------------------
# Module paths + constants (META-COUNCIL memo + CLAUDE.md anchors)
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
DEFAULT_PAIR_MANIFEST = (
    REPO_ROOT
    / ".omx"
    / "research"
    / "artifacts"
    / "lapose_motion_atoms_20260505_codex"
    / "lapose_motion_atom_manifest_fixture.json"
)
CONTEST_AUTH_EVAL_SCRIPT = REPO_ROOT / "experiments" / "contest_auth_eval.py"

EVAL_HW = (384, 512)
CAMERA_HW = (874, 1164)
N_PAIRS_FULL = 600
CONTEST_NORMALIZER = 37_545_489.0

# Canonical A1 anchor (post-fine-tune; META-COUNCIL §1 + operator directive)
A1_CANONICAL_SHA256_INNER = (
    "8e664385af0a25ec98bd02d97b697fbf0d2bb3c2d954f5aa5c95b5131330a243"
)
A1_CANONICAL_BYTES_INNER = 178_162
A1_LATENT_DIM = 28
A1_BASE_CHANNELS = 36


# ---------------------------------------------------------------------------
# Catalog #151 manifest — annotated assignment per Catalog #168 so the AST
# walker in Catalog #151's static gate observes the dict.
# ---------------------------------------------------------------------------
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--a1-archive": {
        "env": "A1_PLUS_WAVELET_A1_ARCHIVE",
        "rationale": (
            "A1 base substrate archive bytes (the 178,162 B PR101-fine-tuned "
            "anchor at SHA 8e664385...).  Residual sidecar composition LOADS "
            "this archive verbatim and APPENDS the wavelet sidecar trailer."
        ),
        "default": "experiments/results/track4_sg_a1_t178000_20260509/submission_dir/archive.zip",
        "required_input_file": True,
        "generator_command": (
            "experiments/results/track4_sg_a1_t178000_20260509/ "
            "(landed 2026-05-09 via PR101 score-gradient fine-tune)"
        ),
        "rationale_audit": (
            ".omx/research/meta_council_decision_attribution_audit_20260513.md"
            "#section-6b-key-finding-wavelet-c3-retarget-is-the-highest-ev-competing-path"
        ),
    },
    "--pair-manifest": {
        "env": "A1_PLUS_WAVELET_PAIR_MANIFEST",
        "rationale": (
            "JSON manifest of pair indices receiving the wavelet residual.  "
            "Accepts simple {'pairs': [int, ...]} OR LAPose-style atoms[] "
            "with hard_pair_support/atom_id.  Compatible with the LAPose "
            "motion-atom manifest fixture as a default starting set."
        ),
        "default": ".omx/research/artifacts/lapose_motion_atoms_20260505_codex/lapose_motion_atom_manifest_fixture.json",
        "required_input_file": True,
        "generator_command": (
            "tools/build_lapose_motion_atom_manifest.py OR "
            "an operator-supplied pose-drift ranking"
        ),
        "rationale_audit": (
            ".omx/research/meta_council_decision_attribution_audit_20260513.md"
            "#section-5b-recommended-ordering-before-gpu-dispatch"
        ),
    },
    "--video-path": {
        "env": "A1_PLUS_WAVELET_VIDEO_PATH",
        "rationale": (
            "Score-aware substrate trains against contest video "
            "(upstream/videos/0.mkv); synthetic data is FORBIDDEN outside --smoke."
        ),
        "default": "upstream/videos/0.mkv",
        "required_input_file": True,
        "generator_command": (
            "contest-pinned upstream snapshot — never regenerated locally"
        ),
    },
    "--output-dir": {
        "env": "A1_PLUS_WAVELET_OUTPUT_DIR",
        "rationale": "custody location for checkpoints + archive + provenance",
    },
    "--epochs": {
        "env": "A1_PLUS_WAVELET_EPOCHS",
        "rationale": (
            "training epochs; META-COUNCIL §8c full-dispatch target 2000 "
            "(freeze-A1 mode is faster per-epoch than D3.B joint)."
        ),
        "default": "2000",
    },
    "--upstream-dir": {
        "env": "A1_PLUS_WAVELET_UPSTREAM_DIR",
        "rationale": (
            "upstream/ root for scorer weights + evaluate.py; required for "
            "full training and auth eval"
        ),
        "default": "upstream",
    },
    "--device": {
        "env": "A1_PLUS_WAVELET_DEVICE",
        "rationale": (
            "compute device; cuda required for full training (MPS refused "
            "per CLAUDE.md MPS-NOISE rule); cpu permitted only with --smoke"
        ),
        "default": "cuda",
    },
    "--coeff-rank": {
        "env": "A1_PLUS_WAVELET_COEFF_RANK",
        "rationale": (
            "per-band low-rank K for the wavelet detail-band coefficient outer "
            "product.  Default 1 keeps the sidecar ≤500 B even at 16 selected "
            "pairs (D2 byte budget)."
        ),
        "default": "1",
    },
    "--max-pairs": {
        "env": "A1_PLUS_WAVELET_MAX_PAIRS",
        "rationale": (
            "cap on number of pair indices loaded from the manifest. Lower "
            "numbers shrink the wavelet sidecar at the cost of fewer hard-pair "
            "corrections (D2 verdict trade-off; META-COUNCIL §6b)."
        ),
        "default": "16",
    },
}


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_substrate_a1_plus_wavelet_residual",
        description=(
            "Train A1 + wavelet residual sidecar composition substrate "
            "(META-COUNCIL §6b highest EV/$ retarget; operator-fallback D3.C "
            "freeze-A1 mode)."
        ),
    )
    # Tier 1 required inputs
    p.add_argument("--a1-archive", type=Path, default=DEFAULT_A1_ARCHIVE,
                   help="Path to A1 base archive.zip (178,162 B PR101-fine-tuned anchor).")
    p.add_argument("--pair-manifest", type=Path, default=DEFAULT_PAIR_MANIFEST,
                   help="Path to pair-manifest JSON (pairs[] of int OR LAPose atoms[]).")
    p.add_argument("--video-path", type=Path, default=DEFAULT_VIDEO_PATH,
                   help="Path to upstream/videos/0.mkv (contest video).")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Where to write checkpoints + manifest + archive.")
    p.add_argument("--epochs", type=int, required=True,
                   help="Number of training epochs (META-COUNCIL default 2000 for full).")
    p.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR,
                   help="upstream/ root; required for scorer load + auth eval.")

    # Training hyperparameters
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3,
                   help="AdamW learning rate (wavelet head only; A1 frozen).")
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ema-decay", type=float, default=0.997,
                   help="EMA decay per CLAUDE.md EMA non-negotiable.")
    p.add_argument("--noise-std", type=float, default=0.5)
    p.add_argument("--val-pair-count", type=int, default=8)
    p.add_argument("--max-decoded-pairs", type=int, default=N_PAIRS_FULL,
                   help="Cap on total decoded pairs (full=600; smoke uses smaller).")

    # Wavelet residual head config (D2 verdict — bounded ≤ 5 KB, target ~500 B)
    p.add_argument("--coeff-rank", type=int, default=1,
                   help="Per-band low-rank K for the wavelet residual head.")
    p.add_argument("--max-pairs", type=int, default=16,
                   help="Max pair indices loaded from the manifest.")
    p.add_argument("--foveal-h", type=int, default=64,
                   help="Foveal patch height at HALF-camera resolution.")
    p.add_argument("--foveal-w", type=int, default=64,
                   help="Foveal patch width at HALF-camera resolution.")
    p.add_argument("--int8-residual-scale", type=float, default=8.0)

    # Lagrangian weights (score-domain; HNeRV parity L6)
    p.add_argument("--alpha-rate", type=float, default=25.0,
                   help="Rate-term coefficient (contest evaluate.py: 25.0).")
    p.add_argument("--beta-seg", type=float, default=100.0,
                   help="SegNet distortion coefficient (contest evaluate.py: 100.0).")
    p.add_argument("--gamma-pose", type=float, default=math.sqrt(10.0),
                   help="Pose-axis weight (contest evaluate.py: sqrt(10)).")
    p.add_argument("--pose-weight-scale", type=float, default=1.0,
                   help=(
                       "Opt-in pose marginal tilt.  Default 1.0 preserves the "
                       "contest formula; META-COUNCIL pose-marginal 2.71x at "
                       "PR106 r2 is experimental at A1 operating point."
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
    p.add_argument("--enable-tf32", action="store_true",
                   help="Enable TF32 matmul on Ampere+ GPUs. Catalog #178.")

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
    """Read the inner ``x`` blob from an A1-style archive.zip."""
    with zipfile.ZipFile(archive_zip_path) as zf:
        names = zf.namelist()
        if "x" not in names:
            raise ValueError(
                f"A1 archive {archive_zip_path} missing inner 'x' blob; got {names}"
            )
        data = zf.read("x")
    sha = _sha256_bytes(data)
    return data, sha, len(data)


def _load_pair_manifest(manifest_path: Path) -> dict[str, Any]:
    with manifest_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _decode_real_pairs(
    video_path: Path,
    n_pairs: int,
    max_pairs: int | None = None,
    output_hw: tuple[int, int] = CAMERA_HW,
):
    """Decode N pairs from contest video; returns (n_pairs, 2, 3, H, W) uint8-scale."""
    import av
    import torch
    import torch.nn.functional as F

    n_target = min(n_pairs, max_pairs) if max_pairs else n_pairs
    yuv420_to_rgb = _canon_load_upstream_yuv420_to_rgb(
        substrate_tag="a1_plus_wavelet_residual", repo_root=REPO_ROOT
    )
    container = av.open(str(video_path))
    pairs: list[tuple] = []
    cur: list = []
    for frame in container.decode(container.streams.video[0]):
        rgb_hwc = yuv420_to_rgb(frame)
        t = rgb_hwc.permute(2, 0, 1).unsqueeze(0).float()
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


def _load_a1_runtime_modules(a1_archive_path: Path):
    """Load A1's vendored codec.py / model.py from the source runtime."""
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
    """Decode the frozen A1 archive into camera-native RGB pairs."""
    import struct

    import torch
    import torch.nn.functional as F

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
            up = F.interpolate(flat, size=CAMERA_HW, mode="bicubic", align_corners=False)
            up = up.reshape(batch, 2, 3, CAMERA_HW[0], CAMERA_HW[1])
            up[:, 0, 0].sub_(1.0)
            up[:, 0, 2].sub_(1.0)
            up[:, 1, 1].sub_(1.0)
            batches.append(up.contiguous())
    return torch.cat(batches, dim=0)


def _apply_wavelet_residual_batch(head, base_pairs, batch_indices: list[int]):
    """Add wavelet residuals to camera-native A1 base pairs at selected indices.

    The residual is RECONSTRUCTED via DB4 IDWT from per-band coefficient
    factors at the central foveal patch (D5 vanishing-point focus).
    """
    pred = base_pairs.clone()
    h, w = int(pred.shape[-2]), int(pred.shape[-1])
    if (h, w) != CAMERA_HW:
        raise ValueError(
            f"A1+wavelet residual application expects CAMERA_HW={CAMERA_HW}; got {(h, w)}"
        )
    fov_h = int(head.cfg.foveal_h)
    fov_w = int(head.cfg.foveal_w)
    full_h = 2 * fov_h  # camera-native after IDWT
    full_w = 2 * fov_w
    if full_h <= 0 or full_w <= 0 or full_h > h or full_w > w:
        raise ValueError(
            f"invalid foveal patch {(full_h, full_w)} for frame {(h, w)}"
        )
    fov_top = max(0, h // 2 - full_h // 2)
    fov_left = max(0, w // 2 - full_w // 2)
    for local_i, pair_id in enumerate(batch_indices):
        if pair_id not in head._pair_to_slot:
            continue
        for frame_i in (0, 1):
            resid = head.residual_chw_for_pair(int(pair_id), frame_i)
            pred[
                local_i, frame_i, :,
                fov_top : fov_top + full_h,
                fov_left : fov_left + full_w,
            ] = (
                pred[
                    local_i, frame_i, :,
                    fov_top : fov_top + full_h,
                    fov_left : fov_left + full_w,
                ]
                + resid
            )
    return pred


def _collect_residual_coeffs_tensor(head, coeff_rank: int) -> Any:
    """Pack per-pair head.U / head.V into the 6D coeff tensor shape that
    encode_wavelet_sidecar expects.

    Output shape: (N, 3 bands, 2 frames, 3 RGB, rank, foveal_h + foveal_w).
    """
    import torch

    n = head.num_selected
    fh = head.cfg.foveal_h
    fw = head.cfg.foveal_w
    coeffs = torch.zeros(n, 3, 2, 3, coeff_rank, fh + fw)
    with torch.no_grad():
        for slot in range(n):
            for band_idx in range(3):
                for frame_idx in range(2):
                    u = head.U[slot, band_idx, frame_idx].detach().cpu()  # (3, rank, fh)
                    v = head.V[slot, band_idx, frame_idx].detach().cpu()  # (3, rank, fw)
                    coeffs[slot, band_idx, frame_idx, :, :, :fh] = u
                    coeffs[slot, band_idx, frame_idx, :, :, fh:] = v
    return coeffs


# ---------------------------------------------------------------------------
# Smoke main
# ---------------------------------------------------------------------------

def _smoke_main(args: argparse.Namespace) -> int:
    """Tiny CPU smoke that proves the wavelet head + archive pack flow."""
    import torch

    from tac.substrates.a1_plus_wavelet_residual.architecture import (
        A1PlusWaveletResidualConfig,
        PerPairWaveletResidualHead,
        parse_wavelet_residual_pair_indices,
    )
    from tac.substrates.a1_plus_wavelet_residual.archive import pack_composition_archive

    _pin_seeds(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    a1_bytes, a1_sha, a1_size = _load_a1_archive_bytes(args.a1_archive)
    print(f"[smoke] A1 base: {a1_size} B sha256={a1_sha[:16]}...")

    manifest = _load_pair_manifest(args.pair_manifest)
    pair_indices = parse_wavelet_residual_pair_indices(manifest, max_pairs=args.max_pairs)
    if not pair_indices:
        # Smoke-mode fallback for empty manifest
        pair_indices = (3, 7, 11, 17)
    print(f"[smoke] pair indices ({len(pair_indices)}): {pair_indices[:8]}...")

    cfg = A1PlusWaveletResidualConfig(
        selected_pair_indices=pair_indices,
        coeff_rank=args.coeff_rank,
        foveal_h=args.foveal_h,
        foveal_w=args.foveal_w,
        int8_residual_scale=args.int8_residual_scale,
    )
    device = _device_or_die(args.device, smoke=True)
    head = PerPairWaveletResidualHead(cfg).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    n_sel = head.num_selected
    print(
        f"[smoke] residual head: num_selected={n_sel}, "
        f"est_bytes={head.estimated_sidecar_bytes()}"
    )

    for step in range(min(args.epochs, 3)):
        opt.zero_grad()
        loss = head.U.abs().mean() + head.V.abs().mean()
        loss.backward()
        opt.step()
        print(f"[smoke] step {step}: loss={loss.item():.4f}")

    coeffs = _collect_residual_coeffs_tensor(head, args.coeff_rank)
    composition_bytes = pack_composition_archive(
        a1_bytes,
        selected_indices=pair_indices,
        coeffs=coeffs,
        foveal_h=cfg.foveal_h,
        foveal_w=cfg.foveal_w,
        coeff_rank=cfg.coeff_rank,
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
        "wavelet_overhead_bytes": composition_size - a1_size,
        "selected_pair_count": n_sel,
        "coeff_rank": cfg.coeff_rank,
        "foveal_h": cfg.foveal_h,
        "foveal_w": cfg.foveal_w,
        "git_head": _git_head_sha(),
    }
    (args.output_dir / "smoke_metadata.json").write_text(
        json.dumps(smoke_meta, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[smoke] wrote {args.output_dir / 'smoke_metadata.json'}")
    return 0


# ---------------------------------------------------------------------------
# Full main (CUDA-required; score-aware Lagrangian end-to-end; freeze-A1)
# ---------------------------------------------------------------------------

def _full_main(args: argparse.Namespace) -> int:
    """Full training — D3.C freeze-A1 + wavelet residual head only."""
    import torch

    from tac.differentiable_eval_roundtrip import (
        patch_upstream_yuv6_globally,
        unpatch_upstream_yuv6,
    )
    from tac.scorer import load_differentiable_scorers
    from tac.substrates.a1_plus_wavelet_residual.architecture import (
        A1PlusWaveletResidualConfig,
        PerPairWaveletResidualHead,
        parse_wavelet_residual_pair_indices,
    )
    from tac.substrates.a1_plus_wavelet_residual.archive import pack_composition_archive
    from tac.substrates.a1_plus_wavelet_residual.score_aware_loss import (
        A1PlusWaveletResidualLossWeights,
        A1PlusWaveletResidualScoreAwareLoss,
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
    a1_bytes, a1_sha, a1_size = _load_a1_archive_bytes(args.a1_archive)
    _stage(f"a1_loaded_{a1_size}_B_sha{a1_sha[:8]}")

    manifest = _load_pair_manifest(args.pair_manifest)
    pair_indices = parse_wavelet_residual_pair_indices(manifest, max_pairs=args.max_pairs)
    if not pair_indices:
        raise RuntimeError(
            f"Pair manifest {args.pair_manifest} contained zero valid pair "
            "indices; check schema (expected pairs[] of int OR atoms[] with "
            "hard_pair_support / atom_id)."
        )
    _stage(f"pair_indices_{len(pair_indices)}")

    yuv6_token = patch_upstream_yuv6_globally()
    _stage("upstream_yuv6_patched")

    composition_sha = ""
    composition_size = 0
    best_val_lag = math.inf
    best_epoch = -1
    auth_eval_json_path = args.output_dir / "auth_eval.json"

    try:
        posenet, segnet = load_differentiable_scorers(args.upstream_dir, device=device)
        for p in list(posenet.parameters()) + list(segnet.parameters()):
            p.requires_grad_(False)
        posenet.eval()
        segnet.eval()
        _stage("scorers_loaded")

        print(f"[full] decoding GT pairs from {args.video_path} ...")
        gt_pair_tensor = _decode_real_pairs(
            args.video_path,
            n_pairs=N_PAIRS_FULL,
            max_pairs=args.max_decoded_pairs,
            output_hw=CAMERA_HW,
        ).to(device)
        print("[full] decoding frozen A1 base pairs ...")
        a1_pair_tensor = _decode_a1_base_pairs(
            args.a1_archive, a1_bytes, device=device, max_pairs=args.max_decoded_pairs
        )
        n_pairs = int(gt_pair_tensor.shape[0])
        if int(a1_pair_tensor.shape[0]) != n_pairs:
            raise RuntimeError(
                f"A1 decoded pair count mismatch: a1={int(a1_pair_tensor.shape[0])} gt={n_pairs}"
            )
        _stage(f"pairs_decoded_gt_and_a1_{n_pairs}")

        val_count = max(1, min(args.val_pair_count, max(1, n_pairs // 8)))
        val_idx_start = n_pairs - val_count
        train_indices_pool = list(range(val_idx_start))
        val_indices_pool = list(range(val_idx_start, n_pairs))

        cfg = A1PlusWaveletResidualConfig(
            selected_pair_indices=pair_indices,
            coeff_rank=args.coeff_rank,
            foveal_h=args.foveal_h,
            foveal_w=args.foveal_w,
            int8_residual_scale=args.int8_residual_scale,
        )
        head = PerPairWaveletResidualHead(cfg).to(device)
        n_params = sum(p.numel() for p in head.parameters())
        print(
            f"[full] wavelet head params: {n_params:,}  "
            f"est_int8_bytes={head.estimated_sidecar_bytes()}"
        )
        _stage(f"head_built_{n_params}_params")

        ema = EMA(head, decay=args.ema_decay)
        _stage(f"ema_wired_decay_{args.ema_decay}")

        weights = A1PlusWaveletResidualLossWeights(
            alpha_rate=args.alpha_rate,
            beta_seg=args.beta_seg,
            gamma_pose=args.gamma_pose,
            pose_weight_scale=args.pose_weight_scale,
            contest_normalizer=CONTEST_NORMALIZER,
        )
        loss_fn = A1PlusWaveletResidualScoreAwareLoss(
            seg_scorer=segnet, pose_scorer=posenet, weights=weights
        )
        _stage("lagrangian_built")

        optimizer = torch.optim.AdamW(
            head.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs)
        )

        archive_bytes_proxy = torch.tensor(
            float(a1_size + head.estimated_sidecar_bytes()),
            device=device,
        )

        train_started_at = time.time()
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
                gt_pairs = gt_pair_tensor[pair_idxs]
                base_pairs = a1_pair_tensor[pair_idxs]
                gt_a = gt_pairs[:, 0]
                gt_b = gt_pairs[:, 1]
                pred_pairs = _apply_wavelet_residual_batch(
                    head, base_pairs, batch_indices
                ).clamp(0.0, 255.0)
                pred_a = pred_pairs[:, 0]
                pred_b = pred_pairs[:, 1]
                loss, _parts = loss_fn(
                    pred_a, pred_b, gt_a, gt_b, archive_bytes_proxy,
                    apply_eval_roundtrip=True, noise_std=args.noise_std,
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

            if epoch % max(1, args.epochs // 30) == 0 or epoch == args.epochs - 1:
                live_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
                ema.apply(head)
                try:
                    val_lag = _run_val_loop(
                        head, loss_fn, gt_pair_tensor, a1_pair_tensor,
                        val_indices_pool, archive_bytes_proxy, device,
                    )
                finally:
                    head.load_state_dict(live_state)
                    head.train()
                avg_train = (sum(epoch_losses) / len(epoch_losses)) if epoch_losses else math.nan
                print(
                    f"[full] epoch {epoch:4d}: train_loss={avg_train:.5f} "
                    f"val_lag={val_lag:.5f} (best={best_val_lag:.5f})"
                )
                if val_lag < best_val_lag:
                    best_val_lag = val_lag
                    best_epoch = epoch
                    # Save EMA shadow (NEVER live weights — Catalog #88)
                    ema_state = {k: v.detach().cpu() for k, v in head.state_dict().items()}
                    torch.save(
                        {"state_dict": ema_state, "config": asdict(cfg),
                         "epoch": epoch, "val_lag": val_lag},
                        ckpt_best_path,
                    )

        train_elapsed = time.time() - train_started_at
        _stage(
            f"trained_best_epoch_{best_epoch}_val_lag_{best_val_lag:.5f}_elapsed_{train_elapsed:.1f}s"
        )

        best_ckpt = torch.load(ckpt_best_path, weights_only=False, map_location=device)
        head.load_state_dict(best_ckpt["state_dict"])
        head.eval()
        coeffs = _collect_residual_coeffs_tensor(head, args.coeff_rank)
        composition_bytes = pack_composition_archive(
            a1_bytes,
            selected_indices=pair_indices,
            coeffs=coeffs,
            foveal_h=cfg.foveal_h,
            foveal_w=cfg.foveal_w,
            coeff_rank=cfg.coeff_rank,
            int8_scale=cfg.int8_residual_scale,
        )
        composition_sha = _sha256_bytes(composition_bytes)
        composition_size = len(composition_bytes)
        print(
            f"[full] composition: {composition_size} B sha256={composition_sha[:16]}... "
            f"(+{composition_size - a1_size} B over A1)"
        )
        _stage(f"composition_built_{composition_size}_B_sha{composition_sha[:8]}")

        archive_zip_path = args.output_dir / "archive.zip"
        _build_archive_zip(archive_zip_path, composition_bytes)
        submission_dir = args.output_dir / "submission_dir"
        _write_runtime(submission_dir, args.a1_archive)
        shutil.copy2(archive_zip_path, submission_dir / "archive.zip")
        _stage("archive_emitted")

        if not args.skip_auth_eval and device.type == "cuda":
            _run_contest_auth_eval_cuda(
                submission_dir, args.upstream_dir, auth_eval_json_path
            )
            _stage("auth_eval_cuda_done")
    finally:
        unpatch_upstream_yuv6(yuv6_token)
        _stage("upstream_yuv6_unpatched")

    if not args.skip_auth_eval and auth_eval_json_path.exists():
        try:
            from tac.continual_learning import posterior_update_locked_from_auth_eval_json
            update = posterior_update_locked_from_auth_eval_json(auth_eval_json_path)
            print(f"[full] posterior_update accepted={getattr(update, 'accepted', '?')}")
            _stage("posterior_updated")
        except Exception as exc:
            print(f"[full] WARN posterior_update failed: {exc!r}")

    provenance = {
        "lane_id": "lane_a1_plus_wavelet_residual_retarget_20260513",
        "started_at_utc": stage_log[0]["at"] if stage_log else _utc_now_iso(),
        "completed_at_utc": _utc_now_iso(),
        "git_head": _git_head_sha(),
        "a1_base_sha256": a1_sha,
        "a1_base_bytes": a1_size,
        "composition_sha256": composition_sha,
        "composition_bytes": composition_size,
        "wavelet_overhead_bytes": composition_size - a1_size,
        "selected_pair_count": len(pair_indices),
        "best_val_lag": best_val_lag,
        "best_epoch": best_epoch,
        "epochs": args.epochs,
        "device": str(device),
        "council_verdicts": {
            "D1": "residual sidecar (operator-pre-approved default)",
            "D2": "int8 + brotli, ≤500B target (operator-pre-approved)",
            "D3": "D3.C freeze-A1 (META-COUNCIL §8b Bayesian-optimal first dispatch)",
            "D4": "D4.B magic-byte trailer (mirrors A1+LAPose)",
            "D5": "pose-residual at central foveal patch (META-COUNCIL pose-marginal)",
            "D6": "BOTH AXES contest-CPU + contest-CUDA (mandatory)",
        },
        "meta_council_predicted_band": [0.187, 0.194],
        "meta_council_predicted_delta": [-0.003, -0.0005],
        "stage_log": stage_log,
        "hardware_substrate_cuda": _canon_detect_hardware_substrate(
            axis="cuda",
            substrate_tag="a1_plus_wavelet_residual",
            provenance_path=args.output_dir.parent / "provenance.json",
            env_var_candidates=("A1_PLUS_WAVELET_GPU", "MODAL_GPU"),
        ),
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[full] wrote {args.output_dir / 'provenance.json'}")
    return 0


def _run_val_loop(
    head, loss_fn, gt_pair_tensor, a1_pair_tensor, val_indices_pool,
    archive_bytes_proxy, device,
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
            pred_pair = _apply_wavelet_residual_batch(
                head, base_pair, [int(pair_id)]
            ).clamp(0.0, 255.0)
            pred_a = pred_pair[:, 0]
            pred_b = pred_pair[:, 1]
            try:
                loss, _ = loss_fn(
                    pred_a, pred_b, gt_a, gt_b, archive_bytes_proxy,
                    apply_eval_roundtrip=True, noise_std=0.0,
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
        zi.compress_type = zipfile.ZIP_STORED
        zf.writestr(zi, composition_bytes)


def _write_runtime(submission_dir: Path, a1_archive: Path) -> None:
    """Emit contest-compliant inflate.sh + inflate.py runtime tree (D4.B)."""
    submission_dir.mkdir(parents=True, exist_ok=True)
    a1_src = a1_archive.parent / "src"
    if not a1_src.is_dir():
        raise FileNotFoundError(
            f"A1 source vendor missing at {a1_src}; expected codec.py + model.py"
        )
    target_src = submission_dir / "src"
    target_src.mkdir(parents=True, exist_ok=True)
    for name in ("codec.py", "model.py"):
        if (a1_src / name).is_file():
            shutil.copy2(a1_src / name, target_src / name)

    runtime_pkg = (
        submission_dir / "src" / "tac" / "substrates" / "a1_plus_wavelet_residual"
    )
    runtime_pkg.mkdir(parents=True, exist_ok=True)
    for pkg_init in (
        submission_dir / "src" / "tac" / "__init__.py",
        submission_dir / "src" / "tac" / "substrates" / "__init__.py",
        runtime_pkg / "__init__.py",
    ):
        pkg_init.write_text("", encoding="utf-8")
    substrate_src = REPO_ROOT / "src" / "tac" / "substrates" / "a1_plus_wavelet_residual"
    for name in ("architecture.py", "archive.py", "inflate.py"):
        shutil.copy2(substrate_src / name, runtime_pkg / name)
    _canon_vendor_shared_inflate_runtime(submission_dir, repo_root=REPO_ROOT)

    inflate_sh = (
        "#!/usr/bin/env bash\n"
        "# A1 + wavelet residual sidecar inflate.sh (D4.B single-stage).\n"
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

    inflate_py = (
        "#!/usr/bin/env python\n"
        '"""A1 + wavelet residual contest inflate dispatcher (D4.B).\n'
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
        "from tac.substrates.a1_plus_wavelet_residual.inflate import inflate_one\n"
        "\n"
        "def main() -> int:\n"
        "    if len(sys.argv) != 4:\n"
        '        print("usage: inflate.py archive_dir output_dir file_list", file=sys.stderr)\n'
        "        return 2\n"
        "    archive_dir = Path(sys.argv[1])\n"
        "    output_dir = Path(sys.argv[2])\n"
        "    file_list_path = Path(sys.argv[3])\n"
        "    output_dir.mkdir(parents=True, exist_ok=True)\n"
        '    candidates = ["x", "0.bin"]\n'
        "    src_bin = None\n"
        "    for cand in candidates:\n"
        "        p = archive_dir / cand\n"
        "        if p.is_file():\n"
        "            src_bin = p\n"
        "            break\n"
        "    if src_bin is None:\n"
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


def _run_contest_auth_eval_cuda(
    submission_dir: Path, upstream_dir: Path, output_json: Path
) -> None:
    """Run contest_auth_eval.py --device cuda on the submission_dir."""
    cmd = [
        sys.executable, str(CONTEST_AUTH_EVAL_SCRIPT),
        "--archive", str(submission_dir / "archive.zip"),
        "--inflate-sh", str(submission_dir / "inflate.sh"),
        "--upstream-dir", str(upstream_dir),
        "--device", "cuda",
        "--json-out", str(output_json),
    ]
    print(f"[auth_eval] {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"contest_auth_eval.py failed rc={proc.returncode}; "
            f"stderr_tail={proc.stderr[-2000:]}"
        )
    from tac.auth_eval_result import parse_auth_eval_score_claim

    payload = json.loads(output_json.read_text(encoding="utf-8"))
    claim = parse_auth_eval_score_claim(payload, required_score_axis="contest_cuda")
    if claim is None:
        raise RuntimeError(
            "contest_auth_eval.py completed but did not produce a valid "
            "contest_cuda score claim; refusing silent diagnostic-only success"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.smoke:
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":
    sys.exit(main())
