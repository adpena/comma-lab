"""Train the ff_nerv substrate end-to-end on contest video.

Operator-callable training script per the Fields-medal grand council substrate
design wave (2026-05-12). WAVE-3-C wires ``_full_main`` so the trainer is
dispatch-ready as a MEDIUM-target HNeRV-family architectural-twist attack;
council Phase 5 prediction ~0.18-0.20 [contest-CUDA].

Architectural twist vs ``sane_hnerv``: the renderer predicts **band-limited
2D DCT coefficient grids** per frame-pair; the inflate-time path reconstructs
RGB frames via inverse 2D DCT (IDCT2). The decoder learns COEFFICIENTS, not
BASIS — the IDCT2 basis is deterministic. Low-frequency coefficients
dominate perceptual content, so the rate term is naturally banded; the L0
SKETCH retains a 64x64 frequency grid (vs the 384x512 RGB target).

The hypothesis: at the PR106 r2 operating point (pose_avg ~ 3.4e-5), the
contest PoseNet's gradient-favored frequency content is concentrated in the
low-frequency band. A renderer that explicitly only generates the low-freq
DCT coefficients should beat the un-constrained HNeRV variant because the
high-freq waste is structurally eliminated.

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
- ``tac.training.EMA(decay=0.997)`` update after every ``optimizer.step``;
  inference checkpoint = EMA shadow, NEVER live weights (CLAUDE.md "EMA -
  NON-NEGOTIABLE").
- Score-domain Lagrangian ``alpha*B(theta)/N + beta*d_seg + gamma*sqrt(d_pose)``
  per HNeRV parity lesson L6. The frequency-domain architecture provides the
  inductive bias; the loss shape matches sane_hnerv.
- AdamW lr cosine annealing; gradient clip 1.0; NaN watchdog per Council D.
- End with CUDA auth eval on best EMA checkpoint per CLAUDE.md "Auth eval
  EVERYWHERE"; refuse MPS (Catalog #1); CPU permitted only with ``--smoke``.
- Continual-learning posterior update via ``posterior_update_locked``
  (Catalog #128 atomic fcntl).
- Cost-band anchor append via ``tools/append_cost_band_anchor.py``.
- Contest-compliant runtime emission (inflate.sh / inflate.py with 3
  positional args + ``set -euo pipefail`` + <= 100 LOC inflate.py + NO scorer
  imports) per Catalog #146 semantics.
- TIER_1_OPERATOR_REQUIRED_FLAGS declared per Catalog #151 for wire-up.

Architectural risk (council Round 3 - NVIDIA-grade):
- HIGHEST RISK of the WAVE-3 trio. The IDCT2 basis is a deterministic
  einsum (``hi,bcij,wj->bchw``); gradient flow through it is well-defined
  but expensive (B * 3 * 64 * 64 -> B * 3 * 384 * 512 einsum per pair).
  At batch_size=32 this is 32 * 3 * 64 * 64 = 393K coefficients per pair,
  einsum'd against (384, 64) and (512, 64) basis matrices; the intermediate
  tensor (B, 3, 384, 64) is 24M floats = 96MB. A100 (40GB) handles this.
  T4 (16GB) would need batch_size cut to 8-16.
- The DCT basis is computed once at __init__ and registered as a non-
  persistent buffer (NOT in state_dict). At inflate time the basis is
  rebuilt from the config (deterministic), so the archive doesn't pay for
  it. This is essential: the basis is ~24K floats; storing it would add
  ~50KB to the rate term for zero score benefit (the inflate path can
  trivially rebuild it).
- The 64x64 freq-band cutoff is a hard prior. If the contest video has
  significant high-frequency content (sharp edges, fine textures) that
  PoseNet rewards, this prior is wrong and ff_nerv will saturate. The
  council Phase 5 prediction acknowledges this; L1 promotion may sweep
  freq_grid_h/w over {32, 48, 64, 96, 128}.

Usage (smoke; CPU, tiny config, ~10 epochs, no scorer load)::

    .venv/bin/python experiments/train_substrate_ff_nerv.py \\
        --video-path upstream/videos/0.mkv \\
        --output-dir experiments/results/ff_nerv_smoke_<utc> \\
        --epochs 10 \\
        --device cpu --smoke

Usage (full; CUDA-required; threads from operator wrapper)::

    .venv/bin/python experiments/train_substrate_ff_nerv.py \\
        --video-path upstream/videos/0.mkv \\
        --output-dir experiments/results/ff_nerv_<utc> \\
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
import shutil
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
# sane_hnerv manifest per council R1-R7 (see CLAUDE.md catalog #151).
# ---------------------------------------------------------------------------
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--video-path": {
        "env": "FF_NERV_VIDEO_PATH",
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
        "env": "FF_NERV_OUTPUT_DIR",
        "rationale": "custody location for checkpoints + archive + provenance",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--epochs": {
        "env": "FF_NERV_EPOCHS",
        "rationale": (
            "ff_nerv substrate engineering pass; under-training silently regresses "
            "(council target: 2000)"
        ),
        "default": "2000",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--batch-size": {
        "env": "FF_NERV_BATCH_SIZE",
        "rationale": (
            "IDCT2 einsum memory footprint is high at full resolution; A100 fits "
            "batch=32, T4 fits batch=8-16. Dispatch wrapper must thread explicitly."
        ),
        "default": "16",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--upstream-dir": {
        "env": "FF_NERV_UPSTREAM_DIR",
        "rationale": (
            "upstream/ root for scorer weights + evaluate.py; required for full "
            "training (non-smoke) and auth eval"
        ),
        "default": str(DEFAULT_UPSTREAM_DIR.relative_to(REPO_ROOT)),
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--device": {
        "env": "FF_NERV_DEVICE",
        "rationale": (
            "compute device; cuda required for full training (MPS refused per "
            "CLAUDE.md MPS-NOISE rule); cpu permitted only with --smoke"
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
        prog="train_substrate_ff_nerv",
        description="Train ff_nerv frequency-domain substrate end-to-end (WAVE-3-C wired).",
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
        help=(
            "Number of pair indices per batch. ff_nerv IDCT2 einsum memory "
            "footprint is high; A100 fits 32, T4 fits 8-16. Default 16 is "
            "the dispatch-safe T4/A100 common."
        ),
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
        "--latent-dim",
        type=int,
        default=16,
        help="Per-pair latent dimensionality (ff_nerv default 16).",
    )
    p.add_argument(
        "--freq-grid-h",
        type=int,
        default=64,
        help="Vertical DCT basis count retained (low-freq band height).",
    )
    p.add_argument(
        "--freq-grid-w",
        type=int,
        default=64,
        help="Horizontal DCT basis count retained.",
    )
    p.add_argument(
        "--sin-frequency",
        type=float,
        default=30.0,
        help="SIREN sin activation frequency (NeRF default).",
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
        default=1.0,
        help="PoseNet distortion coefficient (contest evaluate.py: 1.0).",
    )
    p.add_argument(
        "--pose-weight-scale",
        type=float,
        default=2.71,
        help=(
            "Operating-point-aware pose-marginal multiplier. At PR106-r2 "
            "(pose_avg ~ 3.4e-5) the pose marginal is 2.71x SegNet's (CLAUDE.md "
            "'SegNet vs PoseNet - operating-point dependent')."
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
            "EMA decay (CLAUDE.md non-negotiable default 0.997 for weights)."
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
        help="Compute device. 'cpu' permitted only with --smoke (CLAUDE.md "
             "'MPS auth eval is NOISE'; mps is rejected at parse time).",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Tiny CPU smoke (no scorer load, tiny config, synthetic targets "
            "OK because --smoke; never use this output for ranking)."
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
        "pact_ff_nerv_upstream_frame_utils", frame_utils_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load upstream frame_utils.py from {frame_utils_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.yuv420_to_rgb


def _decode_real_pairs(
    video_path: Path,
    *,
    n_pairs: int,
    max_pairs: int | None = None,
):
    """Decode real contest pairs (0,1), (2,3), ... at EVAL_HW (384, 512)."""
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
            "pyav (`av`) is required for non-smoke ff_nerv training; "
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
            f"{video_path} yielded {len(frames_chw)} frame(s), need {frames_needed}"
        )
    stacked = torch.stack(frames_chw[:frames_needed])
    return torch.stack([stacked[0::2], stacked[1::2]], dim=1)


# ---------------------------------------------------------------------------
# Lagrangian helpers
# ---------------------------------------------------------------------------

def _archive_bytes_proxy_closed_form(model, *, num_pairs: int, latent_dim: int):
    # type: (...) -> 'torch.Tensor'
    """Closed-form upper-bound on archive bytes for the rate term.

    ff_nerv archive grammar FFV1: fp16 decoder weights + int16 latents +
    utf8-json meta. The IDCT2 basis is rebuilt at inflate-time from the
    config (deterministic), so the basis is NOT counted in the archive
    bytes.

    Proxy: bytes ~= (num_decoder_params * 2) + (num_pairs * latent_dim * 2).
    Constant during training (no parameter dependence); gradient flows
    through the seg + pose terms.
    """
    import torch

    excluded = {"latents", "idct_basis_h", "idct_basis_w"}
    n_decoder = sum(
        p.numel() for n, p in model.named_parameters() if n not in excluded
    )
    bytes_proxy = float(n_decoder * 2 + num_pairs * latent_dim * 2)
    device = next(model.parameters()).device
    return torch.tensor(bytes_proxy, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Contest-compliant runtime emission (Catalog #146 contract)
# ---------------------------------------------------------------------------

def _write_runtime(submission_dir: Path) -> None:
    """Emit the contest-compliant ``inflate.sh`` + ``inflate.py`` pair."""
    submission_dir.mkdir(parents=True, exist_ok=True)
    runtime_pkg = submission_dir / "src" / "tac" / "substrates" / "ff_nerv"
    runtime_pkg.mkdir(parents=True, exist_ok=True)
    # Vendor only the inflate-time ff_nerv package surface. No scorer imports.
    for pkg_init in (
        submission_dir / "src" / "tac" / "__init__.py",
        submission_dir / "src" / "tac" / "substrates" / "__init__.py",
        runtime_pkg / "__init__.py",
    ):
        pkg_init.write_text("", encoding="utf-8")
    substrate_src = REPO_ROOT / "src" / "tac" / "substrates" / "ff_nerv"
    for name in ("architecture.py", "archive.py", "inflate.py"):
        shutil.copy2(substrate_src / name, runtime_pkg / name)

    inflate_sh = (
        "#!/usr/bin/env bash\n"
        "# ff_nerv contest-compliant inflate (WAVE-3-C wired 2026-05-12)\n"
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

    inflate_py = (
        "#!/usr/bin/env python\n"
        "\"\"\"ff_nerv contest-compliant inflate runtime.\n"
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
        "from tac.substrates.ff_nerv.inflate import inflate_one_video\n"
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
    """Deterministic archive.zip containing 0.bin + inflate.sh + inflate.py."""
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
                "[ff_nerv] --device cpu is permitted only with --smoke per "
                "CLAUDE.md 'MPS auth eval is NOISE' + 'EMA - non-negotiable' "
                "+ full-training-needs-CUDA convention. Use --device cuda for "
                "promotion-grade training."
            )
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                "[ff_nerv] --device cuda requested but cuda not available"
            )
        return torch.device("cuda")
    raise SystemExit(f"[ff_nerv] unknown --device {name!r}")


# ---------------------------------------------------------------------------
# Smoke main (CPU; no scorer load)
# ---------------------------------------------------------------------------

def _smoke_main(args: argparse.Namespace) -> int:
    """Tiny CPU smoke that proves the scaffold is wired (no scorer load)."""
    import torch

    from tac.substrates.ff_nerv.architecture import (
        FfnervConfig,
        FfnervSubstrate,
    )

    _pin_seeds(args.seed)

    cfg = FfnervConfig(
        latent_dim=args.latent_dim,
        embed_dim=32,
        initial_grid_h=4,
        initial_grid_w=4,
        decoder_channels=(24, 16, 12, 8),
        sin_frequency=args.sin_frequency,
        num_upsample_blocks=4,
        num_pairs=4,
        output_height=32,
        output_width=64,
        freq_grid_h=16,
        freq_grid_w=16,
    )
    device = _device_or_die(args.device, smoke=True)
    model = FfnervSubstrate(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[smoke] ff_nerv params: {model.num_parameters():,}")
    for step in range(min(args.epochs, 3)):
        idx = torch.arange(cfg.num_pairs, device=device, dtype=torch.long)
        rgb_0, rgb_1 = model(idx)
        loss = rgb_0.abs().mean() + rgb_1.abs().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"[smoke] step {step}: loss={loss.item():.4f}")

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
    from tac.substrates.ff_nerv.architecture import (
        FfnervConfig,
        FfnervSubstrate,
    )
    from tac.substrates.ff_nerv.archive import pack_archive
    from tac.substrates.ff_nerv.score_aware_loss import (
        FfnervScoreAwareLoss,
        ScoreAwareLossWeights,
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

        # Held-out validation indices (last val_pair_count pairs)
        val_count = max(1, min(args.val_pair_count, max(1, n_pairs // 8)))
        val_idx_start = n_pairs - val_count
        train_indices = torch.arange(0, val_idx_start, device=device, dtype=torch.long)
        val_indices = torch.arange(val_idx_start, n_pairs, device=device, dtype=torch.long)

        # 5. Build model
        cfg = FfnervConfig(
            latent_dim=args.latent_dim,
            sin_frequency=args.sin_frequency,
            num_pairs=n_pairs,
            output_height=EVAL_HW[0],
            output_width=EVAL_HW[1],
            freq_grid_h=args.freq_grid_h,
            freq_grid_w=args.freq_grid_w,
        )
        model = FfnervSubstrate(cfg).to(device)
        print(f"[full] ff_nerv params: {model.num_parameters():,}")
        _stage("model_built")

        # 6. EMA shadow (CLAUDE.md non-negotiable)
        ema = EMA(model, decay=args.ema_decay)
        _stage(f"ema_wired_decay_{args.ema_decay}")

        # 7. Score-aware Lagrangian (same shape as sane_hnerv; frequency-domain
        # inductive bias is architectural, not loss-driven)
        weights = ScoreAwareLossWeights(
            alpha_rate=args.alpha_rate,
            beta_seg=args.beta_seg,
            gamma_pose=args.gamma_pose,
            pose_weight_scale=args.pose_weight_scale,
            contest_normalizer=CONTEST_NORMALIZER,
        )
        loss_fn = FfnervScoreAwareLoss(
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
        archive_bytes_proxy = _archive_bytes_proxy_closed_form(
            model, num_pairs=n_pairs, latent_dim=cfg.latent_dim,
        )

        # NaN watchdog (Council D pattern)
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
                # Frames in [0,1]; score-aware loss + eval-roundtrip expect [0, 255]
                rgb_0_255 = rgb_0 * 255.0
                rgb_1_255 = rgb_1 * 255.0
                gt = pair_tensor[idx]
                gt_0 = gt[:, 0]
                gt_1 = gt[:, 1]
                loss, parts = loss_fn(
                    rgb_0_255, rgb_1_255, gt_0, gt_1, archive_bytes_proxy,
                    apply_eval_roundtrip=True,
                    noise_std=args.noise_std,
                )
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
                    val_loss, _val_parts = loss_fn(
                        rgb_0_v * 255.0,
                        rgb_1_v * 255.0,
                        pair_tensor[val_indices, 0],
                        pair_tensor[val_indices, 1],
                        archive_bytes_proxy,
                        apply_eval_roundtrip=True,
                        noise_std=args.noise_std,
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
                    # Save EMA shadow (NOT live weights) - CLAUDE.md EMA rule
                    ema_state = ema.state_dict()
                    torch.save(
                        {
                            "state_dict": {k: v.detach().cpu() for k, v in ema_state.items()},
                            "config": asdict(cfg),
                            "ema_decay": args.ema_decay,
                            "best_val_lagrangian": val_lag,
                            "best_epoch": int(epoch),
                            "saved_at_utc": _utc_now_iso(),
                            "training_axis_note": "[contest-CUDA] for promotion; auth eval still required",
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
                    "state_dict": {k: v.detach().cpu() for k, v in ema_state.items()},
                    "config": asdict(cfg),
                    "ema_decay": args.ema_decay,
                    "best_val_lagrangian": best_val_lag,
                    "best_epoch": int(args.epochs - 1),
                    "saved_at_utc": _utc_now_iso(),
                    "fallback_end_of_training_save": True,
                },
                ckpt_best_path,
            )

        # 11. Build the FFV1 archive bytes from the EMA shadow
        archive_sha = ""
        archive_bytes = 0
        archive_zip_path = args.output_dir / "archive.zip"
        if not args.skip_archive_build:
            print(f"[full] building archive from {ckpt_best_path} ...")
            ema_state = torch.load(ckpt_best_path, map_location="cpu", weights_only=False)
            sd = ema_state["state_dict"]
            # Decoder state_dict excludes per-pair latents AND the non-persistent
            # idct basis buffers (which are rebuilt deterministically at inflate).
            excluded = {"latents", "idct_basis_h", "idct_basis_w"}
            decoder_sd = {k: v for k, v in sd.items() if k not in excluded}
            latents = sd["latents"].detach().cpu()
            meta = {
                "embed_dim": cfg.embed_dim,
                "initial_grid_h": cfg.initial_grid_h,
                "initial_grid_w": cfg.initial_grid_w,
                "decoder_channels": list(cfg.decoder_channels),
                "sin_frequency": cfg.sin_frequency,
                "output_height": cfg.output_height,
                "output_width": cfg.output_width,
                "num_upsample_blocks": cfg.num_upsample_blocks,
            }
            bin_bytes = pack_archive(
                decoder_sd,
                latents,
                meta,
                freq_grid_h=cfg.freq_grid_h,
                freq_grid_w=cfg.freq_grid_w,
            )
            (args.output_dir / "0.bin").write_bytes(bin_bytes)
            archive_sha = _sha256_bytes(bin_bytes)
            archive_bytes = len(bin_bytes)
            print(f"[full] wrote 0.bin ({archive_bytes} bytes, sha256={archive_sha})")

            # Emit contest-compliant runtime alongside the bin
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
                        f"[full] auth eval rc={proc.returncode}; stderr=\n{proc.stderr[-2000:]}",
                        file=sys.stderr,
                    )
                else:
                    if auth_eval_result_path.is_file():
                        try:
                            from tac.auth_eval_result import parse_finite_auth_eval_score

                            ae = json.loads(auth_eval_result_path.read_text())
                            parsed_score = parse_finite_auth_eval_score(
                                ae,
                                require_component_recompute=True,
                            )
                            if parsed_score is None:
                                print(
                                    "[full] auth eval JSON did not contain a finite, "
                                    "component-coherent score; no [contest-CUDA] "
                                    "score claim will be recorded",
                                    file=sys.stderr,
                                )
                            else:
                                contest_cuda_score = parsed_score.score
                                print(
                                    f"[full] [contest-CUDA] score = {contest_cuda_score} "
                                    f"(source={parsed_score.source_key}, "
                                    f"archive_sha256={archive_sha})"
                                )
                        except Exception as exc:
                            print(f"[full] could not parse auth eval JSON: {exc}", file=sys.stderr)
            except subprocess.TimeoutExpired:
                print("[full] auth eval TIMEOUT (>3600s)", file=sys.stderr)
            _stage("auth_eval_cuda_done")

        # 13. Continual-learning posterior update (Catalog #128 atomic)
        if contest_cuda_score is not None and archive_sha:
            try:
                from tac.continual_learning import ContestResult, posterior_update_locked

                result = ContestResult(
                    axis="cuda",
                    hardware_substrate="linux_x86_64_t4",
                    architecture_class="lane_substrate_ff_nerv_20260512",
                    score_value=contest_cuda_score,
                    evidence_tag="[contest-CUDA]",
                    archive_sha256=archive_sha,
                    archive_bytes=archive_bytes,
                    notes=f"ff_nerv first-anchor dispatch; epochs={args.epochs}; freq_grid={cfg.freq_grid_h}x{cfg.freq_grid_w}",
                    observed_at_utc=_utc_now_iso(),
                )
                update = posterior_update_locked(result)
                print(
                    f"[full] posterior_update: accepted={update.accepted} "
                    f"reason={update.reason!r}"
                )
            except Exception as exc:
                print(f"[full] posterior_update_locked failed: {exc}", file=sys.stderr)

        # 14. Cost-band anchor (best-effort; never fail the run on this).
        cost_band_anchor_appended = False
        cost_band_anchor_skip_reason: str | None = None
        try:
            from tac.cost_band_calibration import parse_actual_cost_usd

            actual_cost_usd = parse_actual_cost_usd(
                os.environ.get("FF_NERV_ACTUAL_COST_USD"),
                field_name="FF_NERV_ACTUAL_COST_USD",
            )
        except ValueError as exc:
            actual_cost_usd = None
            cost_band_anchor_skip_reason = f"invalid_FF_NERV_ACTUAL_COST_USD:{exc}"
        if COST_BAND_TOOL.is_file() and train_elapsed_sec > 0 and actual_cost_usd is not None:
            try:
                proc = subprocess.run(
                    [
                        sys.executable, str(COST_BAND_TOOL),
                        "--dispatch-label", f"ff_nerv_{_utc_now_iso()}",
                        "--trainer", "experiments/train_substrate_ff_nerv.py",
                        "--platform", os.environ.get("FF_NERV_PLATFORM", "modal"),
                        "--gpu", os.environ.get("FF_NERV_GPU", "A100"),
                        "--epochs", str(args.epochs),
                        "--batch-size", str(args.batch_size),
                        "--actual-wall-clock-sec", str(train_elapsed_sec),
                        "--actual-cost-usd", str(actual_cost_usd),
                        "--notes", "WAVE-3-C first-anchor dispatch",
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
                print(f"[full] cost-band anchor append failed (non-fatal): {exc}", file=sys.stderr)
        else:
            if actual_cost_usd is None and cost_band_anchor_skip_reason is None:
                cost_band_anchor_skip_reason = "missing_FF_NERV_ACTUAL_COST_USD"
            elif not COST_BAND_TOOL.is_file():
                cost_band_anchor_skip_reason = "cost_band_tool_missing"
            else:
                cost_band_anchor_skip_reason = "nonpositive_train_elapsed_sec"

        # 15. Provenance manifest
        provenance = {
            "schema": "ff_nerv_provenance_v1",
            "generated_at": _utc_now_iso(),
            "from_state_hash": "regen_per_session_below",
            "git_head": _git_head_sha(),
            "trainer": "experiments/train_substrate_ff_nerv.py",
            "lane_id": "lane_substrate_ff_nerv_20260512",
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
