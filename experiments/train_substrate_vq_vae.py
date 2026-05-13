"""Train the vq_vae substrate end-to-end on contest video.

Operator-callable training script per the WAVE-1-A operator-orchestrated
parallel deployment (2026-05-12). Mirrors the canonical sane_hnerv + siren
trainer scaffold and binds the 13 HNeRV parity-discipline lessons to the
VQ-VAE persistent-codebook substrate (van den Oord, Vinyals, Kavukcuoglu -
"Neural Discrete Representation Learning", NeurIPS 2017).

Council-binding contract (CLAUDE.md non-negotiables) honored end-to-end:

- Train against ``upstream/videos/0.mkv`` decoded via pyav (NOT synthetic data;
  synthetic batches are FORBIDDEN outside ``--smoke`` per Catalog #114).
- Patch upstream ``rgb_to_yuv6`` via ``patch_upstream_yuv6_globally`` BEFORE
  scorer construction (PR #95/#106 contract - see CLAUDE.md "eval_roundtrip -
  NON-NEGOTIABLE" section).
- ``load_differentiable_scorers`` for SegNet/PoseNet (no scorer load at
  inflate; only at training).
- ``apply_eval_roundtrip_during_training`` inside the per-batch loop
  (eval_roundtrip=True default; never False per Catalog #5).
- ``tac.training.EMA(decay=0.997)`` weight EMA update after every
  ``optimizer.step``; inference checkpoint = EMA shadow, NEVER live weights
  (CLAUDE.md "EMA - NON-NEGOTIABLE").
- Codebook EMA decay 0.99 (van den Oord persistent N_c/m_c form per CLAUDE.md
  "EMA - non-negotiable" codebook exception); applied via the substrate's
  ``_quantize`` STE path - codebook entries adapt faster than weights by
  design.
- Score-domain Lagrangian
  ``alpha*B(theta)/N + beta*d_seg + gamma*sqrt(d_pose) + lambda*commitment``
  per HNeRV parity lesson L6 + van den Oord 2017 commitment-loss term.
- AdamW lr cosine annealing; gradient clip 1.0; NaN watchdog per Council D.
- End with CUDA auth eval on best EMA checkpoint per CLAUDE.md "Auth eval
  EVERYWHERE"; refuse MPS (Catalog #1); CPU permitted only with ``--smoke``.
- Continual-learning posterior update via ``posterior_update_locked``
  (Catalog #128 atomic fcntl).
- Cost-band anchor append via ``tools/append_cost_band_anchor.py``.
- Contest-compliant runtime emission (inflate.sh / inflate.py with 3
  positional args + ``set -euo pipefail`` + NO scorer imports) per
  Catalog #146 semantics.
- TIER_1_OPERATOR_REQUIRED_FLAGS declared per Catalog #151 for wire-up.

Discrete-rate-axis distinction vs sane_hnerv / cool_chic / siren:
  VQ-VAE pays log2(K) bits per spatial cell rather than 16 bits per
  float-latent. With K=512, D=8, grid 48x64 (downsample=8), per-pair index
  bytes = 48*64*2 frames * 9 bits / 8 = 6,912 B; codebook = 512 * 8 * 2 B =
  8,192 B; total << HNeRV's continuous-latent budget. The hypothesis
  (Council Phase 5): the discrete-rate-axis architectural prior matches
  PoseNet / SegNet's quantized-output sensitivity better than continuous
  latents at the same param budget.

The full pipeline runs in this order:

    parse-args -> seed-pin -> patch-yuv6 -> load-scorers -> decode-video
        -> build-model -> EMA -> Lagrangian -> AdamW + cosine
        -> per-epoch train + best-ckpt by val Lagrangian
        -> save EMA shadow -> build archive.zip
        -> emit contest-compliant runtime
        -> [optional] CUDA auth eval -> tag [contest-CUDA]
        -> append cost-band anchor
        -> append continual-learning posterior anchor
        -> provenance.json

Usage (smoke; CPU, tiny config, ~3 epochs, no scorer load)::

    .venv/bin/python experiments/train_substrate_vq_vae.py \\
        --video-path upstream/videos/0.mkv \\
        --output-dir experiments/results/vq_vae_smoke_<utc> \\
        --epochs 3 \\
        --device cpu --smoke

Usage (full; CUDA-required; threads from operator wrapper)::

    .venv/bin/python experiments/train_substrate_vq_vae.py \\
        --video-path upstream/videos/0.mkv \\
        --output-dir experiments/results/vq_vae_<utc> \\
        --epochs 2000 --batch-size 16 --lr 5e-4 --grad-clip 1.0 \\
        --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
import zipfile
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Module paths + constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
"""Resolves to the pact repo root. Used for canonical defaults."""

DEFAULT_VIDEO_PATH = REPO_ROOT / "upstream" / "videos" / "0.mkv"
DEFAULT_UPSTREAM_DIR = REPO_ROOT / "upstream"
DEFAULT_VIDEO_NAMES_FILE = REPO_ROOT / "upstream" / "public_test_video_names.txt"
CONTEST_AUTH_EVAL_SCRIPT = REPO_ROOT / "experiments" / "contest_auth_eval.py"
COST_BAND_TOOL = REPO_ROOT / "tools" / "append_cost_band_anchor.py"

# Eval-roundtrip target resolution (per upstream evaluate.py):
EVAL_HW = (384, 512)
CAMERA_HW = (874, 1164)
N_PAIRS_FULL = 600  # 1200 frames / 2 = non-overlapping pairs
CONTEST_NORMALIZER = 37_545_489.0  # contest evaluate.py N constant


# ---------------------------------------------------------------------------
# Catalog #151 manifest - every flag below must be threaded by any operator
# wrapper that subprocess-invokes this trainer. Schema mirrors the canonical
# sane_hnerv / siren manifests per council R1-R7 (see CLAUDE.md catalog #151).
# ---------------------------------------------------------------------------
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--video-path": {
        "env": "VQ_VAE_VIDEO_PATH",
        "rationale": (
            "score-aware substrate MUST train against the contest video "
            "(upstream/videos/0.mkv); synthetic data is FORBIDDEN outside --smoke"
        ),
        "default": str(DEFAULT_VIDEO_PATH.relative_to(REPO_ROOT)),
        "satisfied_by_profile": (),
        "requires": (),
        "required_input_file": True,
        "generator_command": (
            "contest-pinned upstream snapshot - never regenerated locally"
        ),
        "rationale_audit": (
            ".omx/research/grand_council_fields_medal_substrate_design_20260512.md"
            "#13-lessons-L1"
        ),
    },
    "--output-dir": {
        "env": "VQ_VAE_OUTPUT_DIR",
        "rationale": "custody location for checkpoints + archive + provenance",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--epochs": {
        "env": "VQ_VAE_EPOCHS",
        "rationale": (
            "VQ-VAE substrate engineering pass; under-training silently "
            "regresses (council target: 2000)"
        ),
        "default": "2000",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--batch-size": {
        "env": "VQ_VAE_BATCH_SIZE",
        "rationale": (
            "Per-pair feature grid + STE quantization is memory-moderate; "
            "batch=16 is the dispatch-safe default and must be explicit "
            "in operator recipes"
        ),
        "default": "16",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--upstream-dir": {
        "env": "VQ_VAE_UPSTREAM_DIR",
        "rationale": (
            "upstream/ root for scorer weights + evaluate.py; required for "
            "full training (non-smoke) and auth eval"
        ),
        "default": str(DEFAULT_UPSTREAM_DIR.relative_to(REPO_ROOT)),
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--device": {
        "env": "VQ_VAE_DEVICE",
        "rationale": (
            "compute device; cuda required for full training (MPS refused "
            "per CLAUDE.md MPS-NOISE rule); cpu permitted only with --smoke"
        ),
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
        prog="train_substrate_vq_vae",
        description=(
            "Train vq_vae persistent-codebook substrate end-to-end "
            "(WAVE-1-A wired)."
        ),
    )

    # ---- TIER_1 required ----
    p.add_argument(
        "--video-path",
        type=Path,
        default=DEFAULT_VIDEO_PATH,
        help="Path to upstream/videos/0.mkv (contest video; non-smoke required).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Where to write checkpoints + manifest + archive.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="Number of training epochs (council default 2000 for full).",
    )
    p.add_argument(
        "--upstream-dir",
        type=Path,
        default=DEFAULT_UPSTREAM_DIR,
        help="upstream/ root; required for scorer load + auth eval.",
    )

    # ---- Training hyperparameters ----
    p.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of pair indices per batch (council default 16 for VQ-VAE).",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="AdamW learning rate.",
    )
    p.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="AdamW weight decay.",
    )
    p.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Gradient clip norm (Council D pattern).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Manual seed for torch / numpy / random (deterministic).",
    )

    # ---- Substrate architecture knobs ----
    p.add_argument(
        "--codebook-size",
        type=int,
        default=512,
        help=(
            "K: number of codebook entries. Rate = log2(K) bits per cell. "
            "van den Oord 2017 default 512."
        ),
    )
    p.add_argument(
        "--embedding-dim",
        type=int,
        default=8,
        help="D: per-entry embedding dim (van den Oord 2017 default 8).",
    )
    p.add_argument(
        "--encoder-hidden",
        type=int,
        default=24,
        help="Tiny encoder hidden channels.",
    )
    p.add_argument(
        "--decoder-hidden",
        type=int,
        default=24,
        help="Tiny decoder hidden channels.",
    )
    p.add_argument(
        "--grid-downsample",
        type=int,
        default=8,
        help=(
            "Spatial-grid downsample factor (must be power of two). H/8 x W/8 "
            "default with EVAL_HW=(384, 512) yields 48x64 index grids."
        ),
    )

    # ---- Lagrangian weights (score-aware) ----
    p.add_argument(
        "--alpha-rate",
        type=float,
        default=25.0,
        help="Rate-term coefficient (contest evaluate.py: 25.0).",
    )
    p.add_argument(
        "--beta-seg",
        type=float,
        default=100.0,
        help="SegNet distortion coefficient (contest evaluate.py: 100.0).",
    )
    p.add_argument(
        "--gamma-pose",
        type=float,
        default=math.sqrt(10.0),
        help="PoseNet sqrt-term coefficient (contest evaluate.py: sqrt(10)).",
    )
    p.add_argument(
        "--pose-weight-scale",
        type=float,
        default=1.0,
        help=(
            "Optional operating-point multiplier layered on top of the contest sqrt(10) "
            "pose coefficient; default 1.0 keeps trainer losses apples-to-apples."
        ),
    )
    p.add_argument(
        "--commitment-cost",
        type=float,
        default=0.25,
        help=(
            "VQ-VAE commitment loss weight (van den Oord 2017 default 0.25). "
            "Pulls encoder output toward the chosen codebook entry."
        ),
    )
    p.add_argument(
        "--noise-std",
        type=float,
        default=0.5,
        help="STE noise std for eval-roundtrip simulation (Hotz fix).",
    )

    # ---- EMA + scheduling ----
    p.add_argument(
        "--ema-decay",
        type=float,
        default=0.997,
        help=(
            "Weight EMA decay (CLAUDE.md non-negotiable default 0.997). "
            "Codebook EMA stays at 0.99 per van den Oord persistent buffer."
        ),
    )
    p.add_argument(
        "--val-every-epochs",
        type=int,
        default=10,
        help="Run held-out proxy eval every N epochs for best-ckpt selection.",
    )
    p.add_argument(
        "--val-pair-count",
        type=int,
        default=32,
        help="Number of pairs reserved for held-out proxy validation.",
    )

    # ---- Device / mode ----
    p.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cuda",
        help=(
            "Compute device. 'cpu' permitted only with --smoke (CLAUDE.md "
            "'MPS auth eval is NOISE'; mps is rejected at parse time)."
        ),
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Tiny CPU smoke (no scorer load, tiny config; never use this "
            "output for ranking)."
        ),
    )
    p.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help=(
            "Cap on number of pairs decoded from the video (debug only). "
            "Default decodes all 600 pairs."
        ),
    )

    # ---- Post-train artifacts ----
    p.add_argument(
        "--skip-auth-eval",
        action="store_true",
        help=(
            "Skip the final auth-eval subprocess (only useful for in-flight "
            "research dispatches that prefer harvest-and-rescore later)."
        ),
    )
    p.add_argument(
        "--skip-archive-build",
        action="store_true",
        help="Skip building the archive.zip (e.g. for trainer-only smoke).",
    )

    return p


# ---------------------------------------------------------------------------
# Video decode (real frame pairs from upstream/videos/0.mkv)
# ---------------------------------------------------------------------------

def _load_upstream_yuv420_to_rgb():
    """Load upstream's PyAV YUV420->RGB helper without patching upstream."""
    import importlib.util

    frame_utils_path = REPO_ROOT / "upstream" / "frame_utils.py"
    if not frame_utils_path.is_file():
        raise FileNotFoundError(
            f"upstream/frame_utils.py not found at {frame_utils_path}; "
            "verify --upstream-dir is correct."
        )
    spec = importlib.util.spec_from_file_location(
        "pact_vq_vae_upstream_frame_utils", frame_utils_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"unable to load upstream frame_utils.py from {frame_utils_path}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.yuv420_to_rgb


def _decode_real_pairs(
    video_path: Path,
    *,
    n_pairs: int,
    max_pairs: int | None = None,
):
    """Decode real contest pairs (0,1), (2,3), ... at EVAL_HW (384, 512).

    Returns:
        torch.Tensor shape ``(N, 2, 3, 384, 512)`` float32 in ``[0, 255]``.
    """
    import torch
    import torch.nn.functional as F

    if not video_path.is_file():
        raise FileNotFoundError(
            f"real target video not found: {video_path}. Non-smoke training "
            "requires upstream/videos/0.mkv."
        )
    try:
        import av  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError(
            "pyav (`av`) is required for non-smoke vq_vae training; "
            "run `uv pip install av`"
        ) from exc

    yuv420_to_rgb = _load_upstream_yuv420_to_rgb()
    target_pairs = n_pairs if max_pairs is None else min(n_pairs, max_pairs)
    frames_needed = target_pairs * 2
    frames_chw: list[torch.Tensor] = []
    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            rgb_hwc = yuv420_to_rgb(frame)
            rgb_chw = rgb_hwc.permute(2, 0, 1).unsqueeze(0).float()
            resized = F.interpolate(
                rgb_chw, size=EVAL_HW, mode="bilinear", align_corners=False
            )
            frames_chw.append(resized.squeeze(0).contiguous())
            if len(frames_chw) >= frames_needed:
                break
    finally:
        container.close()
    if len(frames_chw) < frames_needed:
        raise RuntimeError(
            f"{video_path} yielded {len(frames_chw)} frame(s), "
            f"need {frames_needed}"
        )
    stacked = torch.stack(frames_chw[:frames_needed])  # (frames, 3, H, W)
    # (N, 2, 3, H, W) per upstream AVVideoDataset pair order
    return torch.stack([stacked[0::2], stacked[1::2]], dim=1)


# ---------------------------------------------------------------------------
# Lagrangian helpers
# ---------------------------------------------------------------------------

def _archive_bytes_proxy_closed_form(model, cfg):
    # type: (...) -> 'torch.Tensor'  # forward-ref; torch is imported lazily
    """Closed-form upper-bound on archive bytes for the rate term.

    For VQ-VAE the archive carries:
      - codebook bytes:   K * D * 2 (fp16)
      - decoder bytes:    sum(decoder_param.numel()) * 2 (fp16)
      - indices bytes:    num_pairs * 2 * h_grid * w_grid * 2 (int16 storage)

    This proxy is a constant during training (no parameter dependence beyond
    static counts), so the rate term is a constant offset; gradient flows
    through the seg + pose + commitment terms. The Phase 2 follow-up wires
    a differentiable rate proxy (Ballé hyperprior style) over the codebook
    usage histogram.
    """
    import torch

    # Codebook + decoder ONLY enter the archive (training-only encoder /
    # per-pair features are excluded by runtime_state_dict_for_archive).
    n_runtime_params = 0
    for name, p in model.named_parameters():
        if name == "codebook" or name.startswith("decoder."):
            n_runtime_params += p.numel()
    h_grid = cfg.output_height // cfg.grid_downsample
    w_grid = cfg.output_width // cfg.grid_downsample
    n_index_elems = cfg.num_pairs * 2 * h_grid * w_grid
    bytes_proxy = float(n_runtime_params * 2 + n_index_elems * 2)
    device = next(model.parameters()).device
    return torch.tensor(bytes_proxy, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Contest-compliant runtime emission (Catalog #146 contract)
# ---------------------------------------------------------------------------

def _write_runtime(submission_dir: Path) -> None:
    """Emit the contest-compliant ``inflate.sh`` + ``inflate.py`` pair.

    The substrate's monolithic ``0.bin`` is the archive grammar; the runtime
    is a thin reader that calls ``parse_archive`` and renders frames.

    Per Catalog #146 semantics:

    * 3-positional-arg ``inflate.sh`` ($1=archive_dir $2=output_dir $3=file_list)
    * ``set -euo pipefail``
    * No runtime network/dep fetches
    * No scorer code imports in ``inflate.py``
    * Per-video loop in ``inflate.py``
    * ``inflate.py`` <= 200 LOC (substrate-engineering waiver per HNeRV
      parity lesson L4); the substrate's own inflate runtime is ~94 LOC.
    """
    submission_dir.mkdir(parents=True, exist_ok=True)

    inflate_sh = (
        "#!/usr/bin/env bash\n"
        "# vq_vae contest-compliant inflate (WAVE-1-A wired 2026-05-12)\n"
        "# Contract: $1=archive_dir $2=output_dir $3=file_list\n"
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

    # inflate.py reads the substrate's 0.bin via tac.substrates.vq_vae.
    # NO scorer code is imported. Per-video loop over file_list.
    inflate_py = (
        "#!/usr/bin/env python\n"
        "\"\"\"vq_vae contest-compliant inflate runtime.\n"
        "\n"
        "Reads archive_dir/0.bin via the packaged substrate parser, then for\n"
        "each base in file_list writes per-frame .png under output_dir/<base>/.\n"
        "No scorer-network imports (strict-scorer-rule contract).\n"
        "\"\"\"\n"
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "HERE = Path(__file__).resolve().parent\n"
        "sys.path.insert(0, str(HERE / 'src'))\n"
        "from tac.substrates.vq_vae.inflate import inflate_one_video\n"
        "\n"
        "def main() -> int:\n"
        "    if len(sys.argv) != 4:\n"
        "        print('usage: inflate.py <archive_dir> <output_dir> <file_list>',\n"
        "              file=sys.stderr)\n"
        "        return 2\n"
        "    archive_dir = Path(sys.argv[1])\n"
        "    output_dir = Path(sys.argv[2])\n"
        "    file_list_path = Path(sys.argv[3])\n"
        "    archive_bytes = (archive_dir / '0.bin').read_bytes()\n"
        "    for line in file_list_path.read_text(encoding='utf-8').splitlines():\n"
        "        line = line.strip()\n"
        "        if not line:\n"
        "            continue\n"
        "        base = line.rsplit('.', 1)[0]\n"
        "        inflate_one_video(archive_bytes, output_dir / base, device='cpu')\n"
        "    return 0\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    sys.exit(main())\n"
    )
    (submission_dir / "inflate.py").write_text(inflate_py, encoding="utf-8")


def _build_archive_zip(
    archive_zip_path: Path,
    *,
    bin_bytes: bytes,
    submission_dir: Path,
) -> None:
    """Deterministic archive.zip containing 0.bin + inflate.sh + inflate.py.

    Per Catalog #19 ``check_archive_builders_use_deterministic_zip``: use
    ZipInfo + writestr with fixed timestamp + DEFLATE.
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


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_head_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False, timeout=5,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return "<unknown>"


def _pin_seeds(seed: int) -> None:
    import torch

    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def _device_or_die(name: str, *, smoke: bool):
    import torch

    if name == "cpu":
        if not smoke:
            raise SystemExit(
                "[vq_vae] --device cpu is permitted only with --smoke per "
                "CLAUDE.md 'MPS auth eval is NOISE' + 'EMA - non-negotiable' "
                "+ full-training-needs-CUDA convention. Use --device cuda for "
                "promotion-grade training."
            )
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                "[vq_vae] --device cuda requested but cuda not available"
            )
        return torch.device("cuda")
    raise SystemExit(f"[vq_vae] unknown --device {name!r}")


# ---------------------------------------------------------------------------
# Smoke main (CPU; no scorer load)
# ---------------------------------------------------------------------------

def _smoke_main(args: argparse.Namespace) -> int:
    """Tiny CPU smoke that proves the scaffold is wired (no scorer load)."""
    import torch

    from tac.substrates.vq_vae.architecture import VqVaeConfig, VqVaeSubstrate

    _pin_seeds(args.seed)

    cfg = VqVaeConfig(
        codebook_size=16,
        embedding_dim=4,
        encoder_hidden=8,
        decoder_hidden=8,
        grid_downsample=8,
        num_pairs=4,
        output_height=24,
        output_width=32,
    )
    device = _device_or_die(args.device, smoke=True)
    model = VqVaeSubstrate(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[smoke] vq_vae params: {model.num_parameters():,}")
    for step in range(min(args.epochs, 3)):
        idx = torch.arange(cfg.num_pairs, device=device, dtype=torch.long)
        rgb_0, rgb_1 = model(idx)
        commitment = model.compute_commitment_loss(idx)
        loss = rgb_0.abs().mean() + rgb_1.abs().mean() + 0.25 * commitment
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(
            f"[smoke] step {step}: loss={loss.item():.4f} "
            f"commitment={commitment.item():.6f}"
        )

    ckpt = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": asdict(cfg),
        "smoke": True,
    }
    ckpt_path = args.output_dir / "smoke_checkpoint.pt"
    torch.save(ckpt, ckpt_path)
    print(f"[smoke] wrote {ckpt_path}")
    return 0


# ---------------------------------------------------------------------------
# Full main (CUDA-required; score-aware Lagrangian end-to-end)
# ---------------------------------------------------------------------------

def _full_main(args: argparse.Namespace) -> int:
    """Full training entry point - requires CUDA + score-aware scorers."""
    import torch

    from tac.differentiable_eval_roundtrip import (
        patch_upstream_yuv6_globally,
        unpatch_upstream_yuv6,
    )
    from tac.scorer import load_differentiable_scorers
    from tac.substrates.vq_vae.architecture import VqVaeConfig, VqVaeSubstrate
    from tac.substrates.vq_vae.archive import pack_archive
    from tac.substrates.vq_vae.score_aware_loss import (
        ScoreAwareLossWeights,
        VqVaeScoreAwareLoss,
    )
    from tac.training import EMA

    # 1. Pin seeds
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
        posenet, segnet = load_differentiable_scorers(
            args.upstream_dir, device=device
        )
        for p in list(posenet.parameters()) + list(segnet.parameters()):
            p.requires_grad_(False)
        posenet.eval()
        segnet.eval()
        _stage("scorers_loaded")

        # 4. Decode real frame pairs (NOT synthetic)
        print(f"[full] decoding pairs from {args.video_path} ...")
        pair_tensor = _decode_real_pairs(
            args.video_path,
            n_pairs=N_PAIRS_FULL,
            max_pairs=args.max_pairs,
        )
        n_pairs = int(pair_tensor.shape[0])
        print(f"[full] decoded {n_pairs} pairs at {EVAL_HW}")
        pair_tensor = pair_tensor.to(device)
        _stage(f"pairs_decoded_{n_pairs}")

        val_count = max(1, min(args.val_pair_count, max(1, n_pairs // 8)))
        val_idx_start = n_pairs - val_count
        train_indices = torch.arange(0, val_idx_start, device=device, dtype=torch.long)
        val_indices = torch.arange(val_idx_start, n_pairs, device=device, dtype=torch.long)

        # 5. Build model
        cfg = VqVaeConfig(
            codebook_size=args.codebook_size,
            embedding_dim=args.embedding_dim,
            encoder_hidden=args.encoder_hidden,
            decoder_hidden=args.decoder_hidden,
            grid_downsample=args.grid_downsample,
            num_pairs=n_pairs,
            output_height=EVAL_HW[0],
            output_width=EVAL_HW[1],
            commitment_cost=args.commitment_cost,
        )
        model = VqVaeSubstrate(cfg).to(device)
        print(f"[full] vq_vae params: {model.num_parameters():,}")
        _stage("model_built")

        # 6. EMA shadow (CLAUDE.md non-negotiable; weight EMA decay 0.997).
        # Codebook EMA decay 0.99 lives in the substrate's _quantize STE
        # path; the weight EMA here is the standard tac.training.EMA.
        ema = EMA(model, decay=args.ema_decay)
        _stage(f"ema_wired_decay_{args.ema_decay}")

        # 7. Score-aware Lagrangian
        weights = ScoreAwareLossWeights(
            alpha_rate=args.alpha_rate,
            beta_seg=args.beta_seg,
            gamma_pose=args.gamma_pose,
            pose_weight_scale=args.pose_weight_scale,
            contest_normalizer=CONTEST_NORMALIZER,
            commitment_weight=args.commitment_cost,
        )
        loss_fn = VqVaeScoreAwareLoss(
            seg_scorer=segnet,
            pose_scorer=posenet,
            weights=weights,
        )
        _stage("lagrangian_built")

        # 8. Optimizer + cosine annealing
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, args.epochs)
        )

        # 9. Train
        train_started_at = time.time()
        best_val_lag = math.inf
        best_epoch = -1
        ckpt_best_path = args.output_dir / "best.pt"

        n_train = int(train_indices.shape[0])
        batch_size = max(1, args.batch_size)
        archive_bytes_proxy = _archive_bytes_proxy_closed_form(model, cfg)

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
                rgb_0, rgb_1 = model(idx)
                commitment = model.compute_commitment_loss(idx)
                # Frames in [0,1]; score-aware loss + eval-roundtrip expect [0, 255]
                rgb_0_255 = rgb_0 * 255.0
                rgb_1_255 = rgb_1 * 255.0
                gt = pair_tensor[idx]  # (B, 2, 3, H, W) in [0, 255]
                gt_0 = gt[:, 0]
                gt_1 = gt[:, 1]
                loss, parts = loss_fn(
                    rgb_0_255,
                    rgb_1_255,
                    gt_0,
                    gt_1,
                    archive_bytes_proxy,
                    commitment,
                    apply_eval_roundtrip=True,
                    noise_std=args.noise_std,
                )
                if not torch.isfinite(loss):
                    nan_strike += 1
                    print(
                        f"[full] WARN: non-finite loss at epoch {epoch} batch "
                        f"{start}; strike {nan_strike}/{max_nan_strikes}",
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
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=args.grad_clip
                    )
                optimizer.step()
                ema.update(model)
                epoch_loss_sum += float(loss.detach().item())
                epoch_batches += 1

            scheduler.step()
            avg_loss = epoch_loss_sum / max(1, epoch_batches)

            # 10. Validation + best-ckpt selection (snapshot+restore pattern)
            if (epoch + 1) % args.val_every_epochs == 0 or epoch == args.epochs - 1:
                orig_state = {
                    k: v.detach().clone() for k, v in model.state_dict().items()
                }
                ema.apply(model)
                model.eval()
                with torch.no_grad():
                    rgb_0_v, rgb_1_v = model(val_indices)
                    commitment_v = model.compute_commitment_loss(val_indices)
                    val_loss, _val_parts = loss_fn(
                        rgb_0_v * 255.0,
                        rgb_1_v * 255.0,
                        pair_tensor[val_indices, 0],
                        pair_tensor[val_indices, 1],
                        archive_bytes_proxy,
                        commitment_v,
                        apply_eval_roundtrip=True,
                        noise_std=args.noise_std,
                    )
                val_lag = float(val_loss.detach().item())
                # Restore live weights
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
                    # Save EMA shadow (NOT live weights) - CLAUDE.md EMA rule
                    ema_state = ema.state_dict()
                    torch.save(
                        {
                            "state_dict": {
                                k: v.detach().cpu() for k, v in ema_state.items()
                            },
                            "config": asdict(cfg),
                            "ema_decay": args.ema_decay,
                            "best_val_lagrangian": val_lag,
                            "best_epoch": int(epoch),
                            "saved_at_utc": _utc_now_iso(),
                            "training_axis_note": (
                                "[contest-CUDA] for promotion; auth eval still required"
                            ),
                        },
                        ckpt_best_path,
                    )
            else:
                if (epoch + 1) % max(1, args.val_every_epochs // 2) == 0:
                    print(
                        f"[full] epoch {epoch + 1}/{args.epochs} "
                        f"train_avg_loss={avg_loss:.6f}"
                    )

        train_elapsed_sec = time.time() - train_started_at
        _stage(f"train_complete_elapsed_{int(train_elapsed_sec)}s")

        if not ckpt_best_path.is_file():
            print(
                "[full] WARN: no improving val checkpoint observed; "
                "saving EMA shadow at end-of-training.",
                file=sys.stderr,
            )
            ema_state = ema.state_dict()
            torch.save(
                {
                    "state_dict": {
                        k: v.detach().cpu() for k, v in ema_state.items()
                    },
                    "config": asdict(cfg),
                    "ema_decay": args.ema_decay,
                    "best_val_lagrangian": best_val_lag,
                    "best_epoch": int(args.epochs - 1),
                    "saved_at_utc": _utc_now_iso(),
                    "fallback_end_of_training_save": True,
                },
                ckpt_best_path,
            )

        # 11. Build the VQV1 archive bytes from the EMA shadow
        archive_sha = ""
        archive_bytes = 0
        archive_zip_path = args.output_dir / "archive.zip"
        if not args.skip_archive_build:
            print(f"[full] building archive from {ckpt_best_path} ...")
            ema_state = torch.load(
                ckpt_best_path, map_location="cpu", weights_only=False
            )  # WEIGHTS_ONLY_FALSE_OK:trusted-ema-shadow-from-this-trainer
            sd = ema_state["state_dict"]
            # Reconstruct a CPU model from the EMA shadow so we can compute
            # codebook indices via the substrate's encode_indices_for_archive().
            cpu_model = VqVaeSubstrate(cfg).to("cpu")
            cpu_model.load_state_dict(sd, strict=False)
            cpu_model.eval()
            indices = cpu_model.encode_indices_for_archive()
            runtime_sd = cpu_model.runtime_state_dict_for_archive()
            meta = {
                "encoder_hidden": cfg.encoder_hidden,
                "decoder_hidden": cfg.decoder_hidden,
                "grid_downsample": cfg.grid_downsample,
                "output_height": cfg.output_height,
                "output_width": cfg.output_width,
                "commitment_cost": cfg.commitment_cost,
                "codebook_ema_decay": cfg.codebook_ema_decay,
            }
            bin_bytes = pack_archive(
                runtime_sd,
                indices,
                meta,
                codebook_size=cfg.codebook_size,
                embedding_dim=cfg.embedding_dim,
            )
            (args.output_dir / "0.bin").write_bytes(bin_bytes)
            archive_sha = _sha256_bytes(bin_bytes)
            archive_bytes = len(bin_bytes)
            print(f"[full] wrote 0.bin ({archive_bytes} bytes, sha256={archive_sha})")

            submission_dir = args.output_dir / "submission"
            _write_runtime(submission_dir)
            (submission_dir / "0.bin").write_bytes(bin_bytes)
            _build_archive_zip(
                archive_zip_path,
                bin_bytes=bin_bytes,
                submission_dir=submission_dir,
            )
            print(f"[full] wrote {archive_zip_path}")
            _stage(f"archive_built_bytes_{archive_bytes}")

        # 12. CUDA auth eval (CLAUDE.md "Auth eval EVERYWHERE")
        auth_eval_result_path: Path | None = None
        contest_cuda_score: float | None = None
        if not args.skip_auth_eval and archive_zip_path.is_file():
            print("[full] launching CUDA auth eval ...")
            auth_eval_result_path = args.output_dir / "contest_auth_eval_cuda.json"
            cmd = [
                sys.executable,
                str(CONTEST_AUTH_EVAL_SCRIPT),
                "--archive", str(archive_zip_path),
                "--inflate-sh", str(args.output_dir / "submission" / "inflate.sh"),
                "--upstream-dir", str(args.upstream_dir),
                "--device", "cuda",
                "--json-out", str(auth_eval_result_path),
            ]
            try:
                proc = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=3600
                )
                if proc.returncode != 0:
                    print(
                        f"[full] auth eval rc={proc.returncode}; "
                        f"stderr=\n{proc.stderr[-2000:]}",
                        file=sys.stderr,
                    )
                else:
                    if auth_eval_result_path.is_file():
                        try:
                            from tac.auth_eval_result import (
                                parse_finite_auth_eval_score,
                            )

                            ae = json.loads(auth_eval_result_path.read_text())
                            parsed_score = parse_finite_auth_eval_score(
                                ae,
                                require_component_recompute=True,
                            )
                            if parsed_score is None:
                                print(
                                    "[full] auth eval JSON did not contain a "
                                    "finite, component-coherent score; no "
                                    "[contest-CUDA] score claim will be recorded",
                                    file=sys.stderr,
                                )
                            else:
                                contest_cuda_score = parsed_score.score
                                print(
                                    f"[full] [contest-CUDA] score = "
                                    f"{contest_cuda_score} "
                                    f"(source={parsed_score.source_key}, "
                                    f"archive_sha256={archive_sha})"
                                )
                        except Exception as exc:
                            print(
                                f"[full] could not parse auth eval JSON: {exc}",
                                file=sys.stderr,
                            )
            except subprocess.TimeoutExpired:
                print("[full] auth eval TIMEOUT (>3600s)", file=sys.stderr)
            _stage("auth_eval_cuda_done")

        # 13. Continual-learning posterior update (Catalog #128 atomic)
        if contest_cuda_score is not None and archive_sha:
            try:
                from tac.continual_learning import (
                    ContestResult,
                    posterior_update_locked,
                )

                result = ContestResult(
                    axis="cuda",
                    hardware_substrate="linux_x86_64_t4",
                    architecture_class="lane_substrate_vq_vae_20260512",
                    score_value=contest_cuda_score,
                    evidence_tag="[contest-CUDA]",
                    archive_sha256=archive_sha,
                    archive_bytes=archive_bytes,
                    notes=f"vq_vae first-anchor dispatch; epochs={args.epochs}",
                    observed_at_utc=_utc_now_iso(),
                )
                update = posterior_update_locked(result)
                print(
                    f"[full] posterior_update: accepted={update.accepted} "
                    f"reason={update.reason!r}"
                )
            except Exception as exc:
                print(
                    f"[full] posterior_update_locked failed: {exc}",
                    file=sys.stderr,
                )

        # 14. Cost-band anchor (best-effort; never fail the run on this).
        cost_band_anchor_appended = False
        cost_band_anchor_skip_reason: str | None = None
        try:
            from tac.cost_band_calibration import parse_actual_cost_usd

            actual_cost_usd = parse_actual_cost_usd(
                os.environ.get("VQ_VAE_ACTUAL_COST_USD"),
                field_name="VQ_VAE_ACTUAL_COST_USD",
            )
        except ValueError as exc:
            actual_cost_usd = None
            cost_band_anchor_skip_reason = (
                f"invalid_VQ_VAE_ACTUAL_COST_USD:{exc}"
            )
        if (
            COST_BAND_TOOL.is_file()
            and train_elapsed_sec > 0
            and actual_cost_usd is not None
        ):
            try:
                proc = subprocess.run(
                    [
                        sys.executable, str(COST_BAND_TOOL),
                        "--dispatch-label", f"vq_vae_{_utc_now_iso()}",
                        "--trainer", "experiments/train_substrate_vq_vae.py",
                        "--platform", os.environ.get("VQ_VAE_PLATFORM", "modal"),
                        "--gpu", os.environ.get("VQ_VAE_GPU", "A100"),
                        "--epochs", str(args.epochs),
                        "--batch-size", str(args.batch_size),
                        "--actual-wall-clock-sec", str(train_elapsed_sec),
                        "--actual-cost-usd", str(actual_cost_usd),
                        "--notes", "WAVE-1-A first-anchor dispatch",
                    ],
                    capture_output=True, text=True, timeout=30, check=False,
                )
                if proc.returncode == 0:
                    cost_band_anchor_appended = True
                else:
                    cost_band_anchor_skip_reason = (
                        f"append_failed_rc_{proc.returncode}:"
                        f"{(proc.stderr or proc.stdout)[-500:]}"
                    )
            except Exception as exc:
                cost_band_anchor_skip_reason = f"append_failed:{exc}"
                print(
                    f"[full] cost-band anchor append failed (non-fatal): {exc}",
                    file=sys.stderr,
                )
        else:
            if actual_cost_usd is None and cost_band_anchor_skip_reason is None:
                cost_band_anchor_skip_reason = "missing_VQ_VAE_ACTUAL_COST_USD"
            elif not COST_BAND_TOOL.is_file():
                cost_band_anchor_skip_reason = "cost_band_tool_missing"
            else:
                cost_band_anchor_skip_reason = "nonpositive_train_elapsed_sec"

        # 15. Provenance manifest
        provenance = {
            "schema": "vq_vae_provenance_v1",
            "generated_at": _utc_now_iso(),
            "from_state_hash": "regen_per_session_below",
            "git_head": _git_head_sha(),
            "trainer": "experiments/train_substrate_vq_vae.py",
            "lane_id": "lane_substrate_vq_vae_20260512",
            "args": {
                k: (str(v) if isinstance(v, Path) else v)
                for k, v in vars(args).items()
            },
            "pytorch_version": _torch_version_string(),
            "device": str(device),
            "num_pairs_decoded": n_pairs,
            "num_train_pairs": int(train_indices.shape[0]),
            "num_val_pairs": int(val_indices.shape[0]),
            "best_val_lagrangian": (
                best_val_lag if math.isfinite(best_val_lag) else None
            ),
            "best_epoch": int(best_epoch),
            "train_elapsed_sec": float(train_elapsed_sec),
            "archive_sha256": archive_sha,
            "archive_bytes": archive_bytes,
            "auth_eval_cuda_score": contest_cuda_score,
            "auth_eval_json_path": (
                str(auth_eval_result_path) if auth_eval_result_path else None
            ),
            "cost_band_anchor_appended": cost_band_anchor_appended,
            "cost_band_anchor_skip_reason": cost_band_anchor_skip_reason,
            "stage_log": stage_log,
            "custody_status": "ci-rebuildable",
            "score_claim": contest_cuda_score is not None,
            "score_axis_tag": (
                "[contest-CUDA]" if contest_cuda_score is not None else None
            ),
            # WAVE-1-A is a TRAINER BUILD landing, not a DISPATCH landing.
            # Both flags stay False per CLAUDE.md "score_claim must remain
            # False until council review" + "ready_for_exact_eval_dispatch
            # gates on byte-closure proof".
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
        (args.output_dir / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"[full] wrote {args.output_dir / 'provenance.json'}")
        return 0

    finally:
        unpatch_upstream_yuv6(yuv6_token)


def _torch_version_string() -> str:
    try:
        import torch

        return f"{torch.__version__}"
    except Exception:
        return "<unknown>"


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.smoke:
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
