# SPDX-License-Identifier: MIT
"""Train the grayscale_lut substrate end-to-end on contest video (WAVE-4).

Operator-callable training script per the Fields-medal grand council substrate
design wave (2026-05-12). Mirrors ``experiments/train_substrate_sane_hnerv.py``
end-to-end with substrate-specific differences for the Selfcomp/szabolcs-cs
PR #56 grayscale-LUT analog mask paradigm:

* Per-pair grayscale stream stored as fp32 at training time, quantized to uint8
  + brotli at archive export (the dominant rate term).
* FiLM-conditioned RGB decoder (~94K params per Selfcomp's anchor) maps
  ``(grayscale + per-pair embedding) -> RGB``.
* Score-domain Lagrangian ``alpha*B/N + beta*d_seg + gamma*sqrt(d_pose) + tv``
  per HNeRV parity lesson L6, with an additional total-variation regularizer
  on the analog grayscale (smoother grayscale -> smaller brotli output).

Council-binding contract (CLAUDE.md non-negotiables) honored end-to-end:

- Train against ``upstream/videos/0.mkv`` decoded via pyav (NOT synthetic data;
  synthetic batches are FORBIDDEN outside ``--smoke`` per Catalog #114).
- Patch upstream ``rgb_to_yuv6`` via ``patch_upstream_yuv6_globally`` BEFORE
  scorer construction (PR #95/#106 contract).
- ``load_differentiable_scorers`` for SegNet/PoseNet (no scorer load at
  inflate; only at training).
- ``apply_eval_roundtrip_during_training`` inside the per-batch loop
  (eval_roundtrip=True default; never False per Catalog #5).
- ``tac.training.EMA(decay=0.997)`` update after every ``optimizer.step``;
  inference checkpoint = EMA shadow, NEVER live weights (CLAUDE.md "EMA -
  NON-NEGOTIABLE").
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

Usage (smoke; CPU, tiny config, ~10 epochs, no scorer load)::

    .venv/bin/python experiments/train_substrate_grayscale_lut.py \\
        --video-path upstream/videos/0.mkv \\
        --output-dir experiments/results/grayscale_lut_smoke_<utc> \\
        --epochs 10 \\
        --device cpu --smoke

Usage (full; CUDA-required; threads from operator wrapper)::

    .venv/bin/python experiments/train_substrate_grayscale_lut.py \\
        --video-path upstream/videos/0.mkv \\
        --output-dir experiments/results/grayscale_lut_<utc> \\
        --epochs 2000 --batch-size 16 --lr 5e-4 --grad-clip 1.0 \\
        --device cuda
"""
# AUTOCAST_FP16_WAIVED:score-aware-scorer-path-pending-canonical-autocast-backport


# TORCH_COMPILE_WAIVED:defer-until-per-substrate-canary-validates-Inductor-graph-breaks-and-score-axis-custody
from __future__ import annotations

import argparse
import json
import math
import os
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
    decode_real_pairs as _decode_real_pairs_canonical,
)
from tac.substrates._shared.trainer_skeleton import (
    detect_hardware_substrate as _canon_detect_hardware_substrate,
)
from tac.substrates._shared.trainer_skeleton import (  # TF32_WAIVED: canonical helper trainer_skeleton.device_or_die imported as _device_or_die_canonical wires TF32 per Catalog #178; substring scan misses aliased import per Catalog #270 protocol
    device_or_die as _device_or_die_canonical,
)
from tac.substrates._shared.trainer_skeleton import (
    git_head_sha as _git_head_sha,
)
from tac.substrates._shared.trainer_skeleton import (
    load_upstream_yuv420_to_rgb as _load_upstream_yuv420_to_rgb_canonical,
)
from tac.substrates._shared.trainer_skeleton import (
    pin_seeds as _pin_seeds,
)
from tac.substrates._shared.trainer_skeleton import (
    sha256_bytes as _sha256_bytes,
)
from tac.substrates._shared.trainer_skeleton import (
    torch_version_string as _torch_version_string,
)
from tac.substrates._shared.trainer_skeleton import (
    utc_now_iso as _utc_now_iso,
)

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
# Catalog #151 manifest — every flag below must be threaded by any operator
# wrapper that subprocess-invokes this trainer.
# ---------------------------------------------------------------------------
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--video-path": {
        "env": "GRAYSCALE_LUT_VIDEO_PATH",
        "rationale": (
            "score-aware substrate MUST train against the contest video "
            "(upstream/videos/0.mkv); synthetic data is FORBIDDEN outside --smoke"
        ),
        "default": str(DEFAULT_VIDEO_PATH.relative_to(REPO_ROOT)),
        "satisfied_by_profile": (),
        "requires": (),
        "required_input_file": True,
        "generator_command": (
            "contest-pinned upstream snapshot — never regenerated locally"
        ),
        "rationale_audit": (
            ".omx/research/grand_council_fields_medal_substrate_design_20260512.md"
            "#13-lessons-L1"
        ),
    },
    "--output-dir": {
        "env": "GRAYSCALE_LUT_OUTPUT_DIR",
        "rationale": "custody location for checkpoints + archive + provenance",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--epochs": {
        "env": "GRAYSCALE_LUT_EPOCHS",
        "rationale": (
            "substrate engineering pass; under-training silently regresses "
            "(council target: 2000)"
        ),
        "default": "2000",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--upstream-dir": {
        "env": "GRAYSCALE_LUT_UPSTREAM_DIR",
        "rationale": (
            "upstream/ root for scorer weights + evaluate.py; required for full "
            "training (non-smoke) and auth eval"
        ),
        "default": str(DEFAULT_UPSTREAM_DIR.relative_to(REPO_ROOT)),
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--device": {
        "env": "GRAYSCALE_LUT_DEVICE",
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
        prog="train_substrate_grayscale_lut",
        description=(
            "Train grayscale_lut substrate end-to-end (WAVE-4-GRAYSCALE-LUT; "
            "Selfcomp PR #56 paradigm)."
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
        help="Number of pair indices per batch (council default 16 for grayscale_lut).",
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
        "--grayscale-downsample",
        type=int,
        default=4,
        help=(
            "Spatial downsample factor for the analog grayscale stream "
            "(council default 4; H/4 x W/4 = 96 x 128 grid)."
        ),
    )
    p.add_argument(
        "--decoder-hidden",
        type=int,
        default=48,
        help="Hidden channels of the colorization decoder (council default 48).",
    )
    p.add_argument(
        "--decoder-blocks",
        type=int,
        default=4,
        help="Number of FiLM-conditioned decoder blocks (council default 4).",
    )
    p.add_argument(
        "--embedding-dim",
        type=int,
        default=16,
        help="Per-pair embedding dimensionality for FiLM conditioning.",
    )
    p.add_argument(
        "--lut-bits",
        type=int,
        default=8,
        help=(
            "Bit-depth of the grayscale tone-map LUT (1-8; default 8 = "
            "byte-stable uint8 backward-compat). Per AA HIGH verdict "
            "2026-05-21 + OVERNIGHT-EE-RESUME §13: lut_bits=5 (32-level) "
            "matches STC residual sidecar cover-signal granularity; lut_bits=4 "
            "was PR #56 cargo-cult. Lower lut_bits = smaller brotli output "
            "via entropy reduction (NO archive schema bump; GLV1 preserved)."
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
        help="PoseNet distortion coefficient multiplier (sqrt(10) baked into loss).",
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
        "--grayscale-tv-weight",
        type=float,
        default=0.01,
        help=(
            "Total-variation regularizer weight on the analog grayscale field. "
            "Smoother grayscale -> smaller brotli output (Selfcomp PR #56 anchor)."
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
            "EMA decay (CLAUDE.md non-negotiable default 0.997 for weights). "
            "Codebook EMAs keep their own 0.99."
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
    # Tier-1 optimization CLI surface (TIER-1-OPT-BATCH 2026-05-14).
    p.add_argument(
        "--enable-autocast-fp16",
        action="store_true",
        default=False,
        help="Wrap forward in torch.autocast(fp16) (Catalog #172; 1.5-2x speedup).",
    )
    p.add_argument(
        "--enable-torch-compile",
        action="store_true",
        default=False,
        help="Wrap substrate with torch.compile / Inductor (Catalog #179).",
    )
    p.add_argument(
        "--enable-gt-scorer-cache",
        action="store_true",
        default=False,
        help=(
            "RESERVED (O1): GT-scorer-output cache; wire-in pending per-substrate "
            "score_aware_loss API extension."
        ),
    )


    return p


# ---------------------------------------------------------------------------
# Video decode (real frame pairs from upstream/videos/0.mkv)
# ---------------------------------------------------------------------------

def _load_upstream_yuv420_to_rgb():
    """Thin substrate-tag-bound wrapper over the canonical helper (CANON-DEDUP-1)."""
    return _load_upstream_yuv420_to_rgb_canonical(
        substrate_tag="grayscale_lut", repo_root=REPO_ROOT
    )


def _decode_real_pairs(
    video_path: Path,
    *,
    n_pairs: int,
    max_pairs: int | None = None,
):
    """Thin substrate-tag-bound wrapper over the canonical helper (CANON-DEDUP-1)."""
    return _decode_real_pairs_canonical(
        video_path,
        n_pairs=n_pairs,
        substrate_tag="grayscale_lut",
        max_pairs=max_pairs,
        repo_root=REPO_ROOT,
    )


def _initialize_grayscale_from_pairs(model, pair_tensor) -> None:
    """Initialize the substrate's grayscale parameter from the GT video.

    The grayscale-LUT paradigm exploits the GT luminance channel as the
    analog stream. Initializing from BT.601 luminance of frame 0 in each
    pair gives the decoder a meaningful starting point and avoids the
    cold-start where the grayscale stream is mid-gray everywhere.
    """
    import torch
    import torch.nn.functional as F

    cfg = model.cfg
    h_g = cfg.output_height // cfg.grayscale_downsample
    w_g = cfg.output_width // cfg.grayscale_downsample

    n_pairs = int(pair_tensor.shape[0])
    if n_pairs != cfg.num_pairs:
        return  # silently skip; trainer will use mid-gray init

    # BT.601 luminance from frame 0 of each pair (matches yuv6 contract)
    rgb_0 = pair_tensor[:, 0]  # (N, 3, H, W) in [0, 255]
    luma = (
        0.299 * rgb_0[:, 0]
        + 0.587 * rgb_0[:, 1]
        + 0.114 * rgb_0[:, 2]
    ).unsqueeze(1) / 255.0  # (N, 1, H, W) in [0, 1]
    luma_ds = F.interpolate(luma, size=(h_g, w_g), mode="bilinear", align_corners=False)
    with torch.no_grad():
        model.grayscale.copy_(luma_ds.to(model.grayscale.device))


# ---------------------------------------------------------------------------
# Lagrangian helpers
# ---------------------------------------------------------------------------

def _archive_bytes_proxy_closed_form(model):
    # type: (...) -> 'torch.Tensor'  # forward-ref; torch is imported lazily
    """Closed-form upper-bound on archive bytes for the rate term.

    The grayscale_lut substrate has TWO rate components:
      (a) decoder state_dict (stem + FiLM blocks + heads + pair_embedding)
          stored as fp16 + brotli (~50% brotli savings on weights).
      (b) per-pair grayscale stream stored as uint8 + brotli (~30-40% brotli
          savings on natural-video luminance).

    We use a non-tight but monotone proxy:
      decoder_bytes ~= num_decoder_params * 2  (fp16, no brotli savings counted)
      grayscale_bytes ~= num_pairs * 1 * H/D * W/D  (uint8, no brotli savings)

    The proxy is constant during training (no parameter dependence), so the
    rate term is a constant offset; gradient flows entirely through the
    seg + pose + tv terms. The TV regularizer is what shapes the grayscale
    field toward smoother (= more compressible) configurations.
    """
    import torch

    cfg = model.cfg
    n_decoder = sum(
        p.numel()
        for n, p in model.named_parameters()
        if n != "grayscale"
    )
    h_g = cfg.output_height // cfg.grayscale_downsample
    w_g = cfg.output_width // cfg.grayscale_downsample
    grayscale_bytes = cfg.num_pairs * 1 * h_g * w_g  # uint8 = 1 byte/elem
    bytes_proxy = float(n_decoder * 2 + grayscale_bytes)
    device = next(model.parameters()).device
    return torch.tensor(bytes_proxy, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Contest-compliant runtime emission (Catalog #146 contract)
# ---------------------------------------------------------------------------

def _write_runtime(submission_dir: Path) -> None:
    """Emit the contest-compliant ``inflate.sh`` + ``inflate.py`` pair.

    Per Catalog #146 semantics:

    * 3-positional-arg ``inflate.sh`` ($1=archive_dir $2=output_dir $3=file_list)
    * ``set -euo pipefail``
    * No runtime network/dep fetches
    * No scorer code imports in ``inflate.py``
    * Per-video loop in ``inflate.py``
    * ``inflate.py`` ≤ 100 LOC (substrate inflate is ~90 LOC; CLI glue keeps
      the wrapper under the HNeRV parity lesson L4 budget).
    """
    submission_dir.mkdir(parents=True, exist_ok=True)

    inflate_sh = (
        "#!/usr/bin/env bash\n"
        "# grayscale_lut contest-compliant inflate (WAVE-4-GRAYSCALE-LUT 2026-05-12)\n"
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
        "\"\"\"grayscale_lut contest-compliant inflate runtime.\n"
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
        "from tac.substrates.grayscale_lut.inflate import inflate_one_video\n"
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

def _device_or_die(name: str, *, smoke: bool):
    """Thin substrate-tag-bound wrapper over the canonical helper (CANON-DEDUP-1)."""
    return _device_or_die_canonical(name, smoke=smoke, substrate_tag="grayscale_lut")


# ---------------------------------------------------------------------------
# Smoke main (CPU; no scorer load)
# ---------------------------------------------------------------------------

def _smoke_main(args: argparse.Namespace) -> int:
    """Tiny CPU smoke that proves the scaffold is wired (no scorer load)."""
    import torch

    from tac.substrates.grayscale_lut.architecture import (
        GrayscaleLutConfig,
        GrayscaleLutSubstrate,
    )

    _pin_seeds(args.seed)

    # Tiny config that fits in CPU RAM (24 x 32 output, 6 x 8 grayscale)
    cfg = GrayscaleLutConfig(
        grayscale_downsample=args.grayscale_downsample,
        decoder_hidden=16,
        decoder_blocks=2,
        embedding_dim=args.embedding_dim,
        num_pairs=4,
        output_height=24,
        output_width=32,
        lut_bits=args.lut_bits,
    )
    device = _device_or_die(args.device, smoke=True)
    model = GrayscaleLutSubstrate(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[smoke] grayscale_lut params: {model.num_parameters():,}")
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
    """Full training entry point — requires CUDA + score-aware scorers."""
    import torch

    from tac.differentiable_eval_roundtrip import (
        patch_upstream_yuv6_globally,
        unpatch_upstream_yuv6,
    )
    from tac.scorer import load_differentiable_scorers
    from tac.substrates.grayscale_lut.architecture import (
        GrayscaleLutConfig,
        GrayscaleLutSubstrate,
    )
    from tac.substrates.grayscale_lut.archive import pack_archive
    from tac.substrates.grayscale_lut.score_aware_loss import (
        GrayscaleLutScoreAwareLoss,
        ScoreAwareLossWeights,
    )
    from tac.training import EMA

    # 1. Pin seeds (deterministic CUDA where possible)
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
        cfg = GrayscaleLutConfig(
            grayscale_downsample=args.grayscale_downsample,
            decoder_hidden=args.decoder_hidden,
            decoder_blocks=args.decoder_blocks,
            embedding_dim=args.embedding_dim,
            num_pairs=n_pairs,
            output_height=EVAL_HW[0],
            output_width=EVAL_HW[1],
            lut_bits=args.lut_bits,
        )
        model = GrayscaleLutSubstrate(cfg).to(device)
        print(f"[full] grayscale_lut params: {model.num_parameters():,}")
        # Initialize grayscale field from GT luminance for warm start
        _initialize_grayscale_from_pairs(model, pair_tensor)
        _stage("model_built_with_gt_luminance_init")

        # 6. EMA shadow (CLAUDE.md non-negotiable)
        ema = EMA(model, decay=args.ema_decay)
        _stage(f"ema_wired_decay_{args.ema_decay}")

        # 7. Score-aware Lagrangian
        weights = ScoreAwareLossWeights(
            alpha_rate=args.alpha_rate,
            beta_seg=args.beta_seg,
            gamma_pose=args.gamma_pose,
            pose_weight_scale=args.pose_weight_scale,
            contest_normalizer=CONTEST_NORMALIZER,
            grayscale_tv_weight=args.grayscale_tv_weight,
        )
        loss_fn = GrayscaleLutScoreAwareLoss(
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
        archive_bytes_proxy = _archive_bytes_proxy_closed_form(model)

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
                # Substrate outputs in [0, 1]; loss expects [0, 255]
                rgb_0_255 = rgb_0 * 255.0
                rgb_1_255 = rgb_1 * 255.0
                gt = pair_tensor[idx]
                gt_0 = gt[:, 0]
                gt_1 = gt[:, 1]
                loss, parts = loss_fn(
                    rgb_0_255, rgb_1_255, gt_0, gt_1, archive_bytes_proxy,
                    grayscale_param=model.grayscale,
                    apply_eval_roundtrip=True,
                    noise_std=args.noise_std,
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
                        grayscale_param=model.grayscale,
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
                    # Save EMA shadow (NOT live weights) — CLAUDE.md EMA rule
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

        # 11. Build the GLV1 archive bytes from the EMA shadow
        archive_sha = ""
        archive_bytes = 0
        archive_zip_path = args.output_dir / "archive.zip"
        if not args.skip_archive_build:
            print(f"[full] building archive from {ckpt_best_path} ...")
            ema_ckpt = torch.load(ckpt_best_path, map_location="cpu", weights_only=False)
            sd = ema_ckpt["state_dict"]

            # Re-instantiate the model on CPU with the EMA weights to use the
            # quantize_grayscale_for_archive + runtime_state_dict_for_archive
            # helpers (avoids replicating the byte-packing logic here).
            cpu_model = GrayscaleLutSubstrate(cfg).to("cpu")
            cpu_model.load_state_dict(sd, strict=False)
            grayscale_uint8 = cpu_model.quantize_grayscale_for_archive()
            decoder_sd = cpu_model.runtime_state_dict_for_archive()

            meta = {
                "decoder_hidden": cfg.decoder_hidden,
                "decoder_blocks": cfg.decoder_blocks,
                # OVERNIGHT-TT Phase 2 BUILD 2026-05-21 + AA HIGH verdict:
                # lut_bits surfaced in meta for observability + downstream STC
                # sidecar consumers; archive bytes preserve uint8 schema per
                # Catalog #110/#113 (GLV1 unchanged), so meta key is observability-only.
                "lut_bits": cfg.lut_bits,
            }
            bin_bytes = pack_archive(
                decoder_sd,
                grayscale_uint8,
                meta,
                num_pairs=cfg.num_pairs,
                grayscale_downsample=cfg.grayscale_downsample,
                embedding_dim=cfg.embedding_dim,
                output_height=cfg.output_height,
                output_width=cfg.output_width,
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

        # 12. CUDA auth eval — canonical helper (Catalog #226 self-protect)
        auth_eval_result_path: Path | None = None
        contest_cuda_score: float | None = None
        if not args.skip_auth_eval and archive_zip_path.is_file():
            print("[full] launching CUDA auth eval ...")
            auth_eval_result_path = args.output_dir / "contest_auth_eval_cuda.json"
            auth_result = _canon_gate_auth_eval_call(
                args=args,
                archive_zip=archive_zip_path,
                inflate_sh=args.output_dir / "submission" / "inflate.sh",
                upstream_dir=args.upstream_dir,
                output_json=auth_eval_result_path,
                contest_auth_eval_script=CONTEST_AUTH_EVAL_SCRIPT,
                substrate_tag="grayscale_lut",
                device=device,
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

                # Per CLAUDE.md SIREN audit 2026-05-13 CRITICAL #1 + Catalog
                # #190: detect substrate dynamically from remote driver
                # provenance.json, then env vars, then nvidia-smi.
                _detected_substrate = _canon_detect_hardware_substrate(
                    axis="cuda",
                    substrate_tag="grayscale_lut",
                    provenance_path=args.output_dir / "provenance.json",
                    env_var_candidates=("GRAYSCALE_LUT_GPU", "MODAL_GPU"),
                )
                result = ContestResult(
                    axis="cuda",
                    hardware_substrate=_detected_substrate,
                    architecture_class="lane_substrate_grayscale_lut_20260512",
                    score_value=contest_cuda_score,
                    evidence_tag="[contest-CUDA]",
                    archive_sha256=archive_sha,
                    archive_bytes=archive_bytes,
                    notes=f"grayscale_lut first-anchor dispatch; epochs={args.epochs}",
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
                os.environ.get("GRAYSCALE_LUT_ACTUAL_COST_USD"),
                field_name="GRAYSCALE_LUT_ACTUAL_COST_USD",
            )
        except ValueError as exc:
            actual_cost_usd = None
            cost_band_anchor_skip_reason = (
                f"invalid_GRAYSCALE_LUT_ACTUAL_COST_USD:{exc}"
            )
        if COST_BAND_TOOL.is_file() and train_elapsed_sec > 0 and actual_cost_usd is not None:
            try:
                proc = subprocess.run(
                    [
                        sys.executable, str(COST_BAND_TOOL),
                        "--dispatch-label", f"grayscale_lut_{_utc_now_iso()}",
                        "--trainer", "experiments/train_substrate_grayscale_lut.py",
                        "--platform", os.environ.get("GRAYSCALE_LUT_PLATFORM", "modal"),
                        "--gpu", os.environ.get("GRAYSCALE_LUT_GPU", "A100"),
                        "--epochs", str(args.epochs),
                        "--batch-size", str(args.batch_size),
                        "--actual-wall-clock-sec", str(train_elapsed_sec),
                        "--actual-cost-usd",
                        str(actual_cost_usd),
                        "--notes", "WAVE-4-GRAYSCALE-LUT first-anchor dispatch",
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
                cost_band_anchor_skip_reason = "missing_GRAYSCALE_LUT_ACTUAL_COST_USD"
            elif not COST_BAND_TOOL.is_file():
                cost_band_anchor_skip_reason = "cost_band_tool_missing"
            else:
                cost_band_anchor_skip_reason = "nonpositive_train_elapsed_sec"

        # 15. Provenance manifest
        provenance = {
            "schema": "grayscale_lut_provenance_v1",
            "generated_at": _utc_now_iso(),
            "from_state_hash": "regen_per_session_below",
            "git_head": _git_head_sha(),
            "trainer": "experiments/train_substrate_grayscale_lut.py",
            "lane_id": "lane_substrate_grayscale_lut_20260512",
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


# ---------------------------------------------------------------------------
# META layer SubstrateContract (Catalog #241/#242 canonical migration; first
# POC migration landed by META-CONSOLIDATION-CRITICAL-PLUS-POC subagent
# 2026-05-15). The decoration extincts the Z3 v2 silent-drift bug class for
# this substrate by binding (a) the trainer's claimed contract, (b) the
# recipe schema, (c) the lane registry, and (d) the cost-band envelope into
# ONE source-of-truth that fails-loud at decoration time if the contract
# violates canonical invariants.
#
# Field values are extracted from:
#   - .omx/operator_authorize_recipes/substrate_grayscale_lut_modal_a100_dispatch.yaml
#   - .omx/state/lane_registry.json (lane_substrate_grayscale_lut_20260512)
#   - this trainer's TIER_1_OPERATOR_REQUIRED_FLAGS + AUTOCAST_FP16_WAIVED +
#     TORCH_COMPILE_WAIVED markers + canonical hook usage (gate_auth_eval_call,
#     patch_upstream_yuv6_globally, load_differentiable_scorers, EMA,
#     pack_archive, inflate.sh emission)
# ---------------------------------------------------------------------------

GRAYSCALE_LUT_SUBSTRATE_CONTRACT = SubstrateContract(
    # 2.1 Identity & lifecycle
    id="grayscale_lut",
    lane_id="lane_substrate_grayscale_lut_20260512",
    target_modes=("contest_one_video_replay", "research_substrate"),
    deployment_target="t4_contest_runtime",
    council_verdict_provenance=(
        ".omx/research/grand_council_fields_medal_substrate_design_20260512.md"
    ),
    # 2.2 Architecture & runtime (8 per Catalog #124)
    archive_grammar=(
        "GLV1 monolithic single-file 0.bin: header (magic=GLV1, version=1) + "
        "decoder weights (fp16 + brotli) + per-pair grayscale stream "
        "(uint8 H/4 x W/4 + brotli; dominant rate term) + per-pair FiLM "
        "embedding (fp16, 16-dim per pair)"
    ),
    parser_section_manifest={
        "header": "GLV1_magic_and_version",
        "decoder_weights": "fp16_brotli_blob",
        "grayscale_stream": "uint8_brotli_per_pair",
        "film_embeddings": "fp16_per_pair",
    },
    inflate_runtime_loc_budget=120,
    runtime_dep_closure=("torch>=2.5,<2.7", "brotli", "av"),
    export_format="fp16_brotli",
    score_aware_loss="scorer_loss_terms_btchw",
    bolt_on_loc_budget=1200,  # trainer is ~1144 LOC at landing
    no_op_detector_planned=True,
    # 2.3 Operational mechanism (3 per Catalog #220)
    archive_bytes_added=None,  # full substrate (not a sidecar)
    score_improvement_mechanism_status="RESEARCH_ONLY",
    runtime_overlay_consumed=False,
    # 2.4 Recipe schema (8) — mirrors substrate_grayscale_lut_modal_a100_dispatch.yaml
    recipe_smoke_only=False,
    recipe_research_only=False,  # recipe declares contest-eligible (with research_substrate target also)
    recipe_min_smoke_gpu="A100",  # Catalog #215: A100 full → A100 smoke parity
    recipe_min_vram_gb=40,
    recipe_pyav_decode_strategy="cpu_thread_async_upload",
    recipe_canary_status="post_canary_dependent",
    recipe_video_input_strategy="per_dispatch_local_copy",
    recipe_canary_dependency="sane_hnerv",
    # 2.5 Cost band & GPU envelope (4)
    cost_band_epochs=2000,
    cost_band_gpu_key="A100",
    cost_band_platform_key="modal",
    cost_band_p50_usd=5.50,
    # 2.6 6-hook wire-in (Catalog #125)
    hook_sensitivity_contribution="not_applicable_with_rationale",
    hook_pareto_constraint="rate_distortion_v1",
    hook_bit_allocator_class="not_applicable_with_rationale",
    hook_autopilot_ranker_class_shift_token=None,  # within-class baseline (Selfcomp PR#56 paradigm)
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
            "grayscale-LUT is a within-class substrate; sensitivity is the "
            "rate-distortion frontier already captured by hook_pareto_constraint"
        ),
        "hook_bit_allocator_class": (
            "grayscale stream is uint8 quantized at archive time, FiLM is fp16; "
            "no per-tensor bit allocator (single-precision-per-substream)"
        ),
        "hook_probe_disambiguator": (
            "single mechanism (analog grayscale + FiLM-conditioned colorization "
            "decoder); no 2+ defensible interpretations"
        ),
    },
)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


@register_substrate(GRAYSCALE_LUT_SUBSTRATE_CONTRACT)
def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.smoke:
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
