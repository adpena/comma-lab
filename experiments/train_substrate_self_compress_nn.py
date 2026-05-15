# SPDX-License-Identifier: MIT
"""Train the self_compress_nn (delta) substrate end-to-end on contest video.

WAVE-1-B operator-orchestrated parallel deployment build (2026-05-12). The delta
candidate from the Fields-medal grand council 2026-05-12 substrate-design
wave: MDL-driven weight clustering during training (van den Oord persistent
codebook EMA pattern), Selfcomp / Quantizr-faithful block-FP self-compression
(Quantizr 0.33 anchor 2026-04-21 + Selfcomp PR #56 1.017-bpw block-FP weight
self-compression). Council Phase 5 prediction: ~0.17 HIGH-target.

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
  NON-NEGOTIABLE"). Codebook EMA buffers (van den Oord pattern) keep their
  own 0.99 decay tracked persistently inside ``_VQCodebook``.
- Score-domain Lagrangian
  ``alpha*B(theta)/N + beta*d_seg + gamma*sqrt(d_pose) + lambda_mdl*commit``
  per HNeRV parity lesson L6 with the delta MDL/codebook commit term added.
- AdamW lr cosine annealing; gradient clip 1.0; NaN watchdog per Council D.
- Codebook EMA step after every ``optimizer.step()`` per van den Oord 2017.
- End with CUDA auth eval on best EMA checkpoint per CLAUDE.md "Auth eval
  EVERYWHERE"; refuse MPS (Catalog #1); CPU permitted only with ``--smoke``.
- Continual-learning posterior update via ``posterior_update_locked``
  (Catalog #128 atomic fcntl).
- Cost-band anchor append via ``tools/append_cost_band_anchor.py``.
- Contest-compliant runtime emission (inflate.sh / inflate.py with 3
  positional args + ``set -euo pipefail`` + <= 200 LOC inflate.py per L4
  waiver + NO scorer imports) per Catalog #146 semantics.
- TIER_1_OPERATOR_REQUIRED_FLAGS declared per Catalog #151 for wire-up.

The full pipeline runs in this order:

    parse-args -> seed-pin -> patch-yuv6 -> load-scorers -> decode-video
        -> build-model (with shared VQ codebook) -> EMA -> Lagrangian
        -> AdamW + cosine -> per-epoch train + per-batch codebook EMA step
        + best-ckpt by val Lagrangian -> save EMA shadow
        -> build SCV1 archive (codebook + cluster_indices + latents + meta)
        -> emit contest-compliant runtime
        -> [optional] CUDA auth eval -> tag [contest-CUDA]
        -> append cost-band anchor
        -> append continual-learning posterior anchor
        -> provenance.json

Usage (smoke; CPU, tiny config, no scorer load)::

    .venv/bin/python experiments/train_substrate_self_compress_nn.py \\
        --video-path upstream/videos/0.mkv \\
        --output-dir experiments/results/self_compress_nn_smoke_<utc> \\
        --epochs 10 \\
        --device cpu --smoke

Usage (full; CUDA-required; threads from operator wrapper)::

    .venv/bin/python experiments/train_substrate_self_compress_nn.py \\
        --video-path upstream/videos/0.mkv \\
        --output-dir experiments/results/self_compress_nn_<utc> \\
        --epochs 2000 --batch-size 32 --lr 5e-4 --grad-clip 1.0 \\
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

# Tier-1 optimization helpers (TIER-1-OPT-BATCH 2026-05-14; CLAUDE.md
# Catalog #172/#179). The O1 GT-scorer cache flag is declared but reserved
# pending per-substrate score_aware_loss API extension.
from tac.substrates._shared.trainer_skeleton import (
    build_optimized_training_context as _canon_build_optimized_training_context,
)

# Canonical substrate-trainer helpers (CANON-DEDUP-1 commit ac1cfc41).
# Replaces ~70 LOC of inlined helpers with a single import per the
# 2026-05-13 substrate-trainer dedup migration wave.
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
    torch_version_string as _canon_torch_version_string,
)
from tac.substrates._shared.trainer_skeleton import (
    utc_now_iso as _canon_utc_now_iso,
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
# Catalog #151 manifest - every flag below must be threaded by any operator
# wrapper that subprocess-invokes this trainer. Schema mirrors the canonical
# Track 1 Ballé manifest per council R1-R7 (see CLAUDE.md catalog #151).
#
# Required keys per entry: ``env``, ``rationale``.
# Optional keys: ``default``, ``satisfied_by_profile``, ``requires``,
# ``rationale_audit``, ``required_input_file``, ``generator_command``.
# ---------------------------------------------------------------------------
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--video-path": {
        "env": "SELF_COMPRESS_NN_VIDEO_PATH",
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
        "env": "SELF_COMPRESS_NN_OUTPUT_DIR",
        "rationale": "custody location for checkpoints + archive + provenance",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--epochs": {
        "env": "SELF_COMPRESS_NN_EPOCHS",
        "rationale": (
            "substrate engineering pass; under-training silently regresses "
            "(council target: 2000)"
        ),
        "default": "2000",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--upstream-dir": {
        "env": "SELF_COMPRESS_NN_UPSTREAM_DIR",
        "rationale": (
            "upstream/ root for scorer weights + evaluate.py; required for full "
            "training (non-smoke) and auth eval"
        ),
        "default": str(DEFAULT_UPSTREAM_DIR.relative_to(REPO_ROOT)),
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--device": {
        "env": "SELF_COMPRESS_NN_DEVICE",
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
        prog="train_substrate_self_compress_nn",
        description=(
            "Train delta self_compress_nn substrate end-to-end "
            "(WAVE-1-B operator-orchestrated dispatch)."
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
        default=32,
        help="Number of pair indices per batch (council default 32).",
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

    # ---- Substrate architecture knobs (council delta defaults) ----
    p.add_argument(
        "--latent-dim",
        type=int,
        default=28,
        help="Per-pair latent dimensionality (council default 28).",
    )
    p.add_argument(
        "--sin-frequency",
        type=float,
        default=30.0,
        help="SIREN sin activation frequency (NeRF default).",
    )
    p.add_argument(
        "--codebook-k",
        type=int,
        default=256,
        help=(
            "Cluster count K. log2(K)=bits/group for cluster-index storage. "
            "Council delta SKETCH default: 256 (8 bits/group)."
        ),
    )
    p.add_argument(
        "--codebook-dv",
        type=int,
        default=8,
        help=(
            "Per-cluster vector dim D_v. Each weight tensor reshaped to "
            "(-1, D_v) groups before clustering. Council delta default: 8 "
            "(amortizes codebook cost over 8 weights)."
        ),
    )
    p.add_argument(
        "--codebook-ema-decay",
        type=float,
        default=0.99,
        help=(
            "Van den Oord codebook EMA decay (per CLAUDE.md EMA "
            "codebook-specific clause: 0.99 - codebook adapts faster than "
            "weights)."
        ),
    )

    # ---- Lagrangian weights (score-aware + delta MDL) ----
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
        "--lambda-mdl",
        type=float,
        default=0.25,
        help=(
            "Weight on the codebook MDL/commit term. Mirrors VQ-VAE 2017's beta. "
            "Sweep range [0.05, 1.0] per council delta design."
        ),
    )
    p.add_argument(
        "--noise-std",
        type=float,
        default=0.5,
        help="STE noise std for eval-roundtrip simulation (Hotz fix; reserved).",
    )

    # ---- EMA + scheduling ----
    p.add_argument(
        "--ema-decay",
        type=float,
        default=0.997,
        help=(
            "EMA decay (CLAUDE.md non-negotiable default 0.997 for weights). "
            "Codebook EMA buffers keep their own 0.99 internally."
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

def _decode_real_pairs(
    video_path: Path,
    *,
    n_pairs: int,
    max_pairs: int | None = None,
):
    """Decode real contest pairs (0,1), (2,3), ... at EVAL_HW (384, 512).

    Thin wrapper around ``tac.substrates._shared.trainer_skeleton``'s
    canonical ``decode_real_pairs`` (CANON-DEDUP-1) with ``substrate_tag``
    curried for this trainer.
    """
    return _canon_decode_real_pairs(
        video_path,
        n_pairs=n_pairs,
        substrate_tag="self_compress_nn",
        max_pairs=max_pairs,
        repo_root=REPO_ROOT,
    )


# ---------------------------------------------------------------------------
# delta archive-bytes proxy (codebook + indices, NOT full weights)
# ---------------------------------------------------------------------------

def _archive_bytes_proxy_closed_form(model):
    # type: (...) -> 'torch.Tensor'  # forward-ref; torch is imported lazily
    """Closed-form upper-bound on delta archive bytes for the rate term.

    Per the council delta archive grammar (SCV1, see ``archive.py``), the on-disk
    size is dominated by:

        codebook        ~  K * D_v * 2 bytes       (fp16; brotli compresses)
        cluster_indices ~  num_quantized_groups * 2 bytes (int16)
        latents         ~  num_pairs * latent_dim * 2 bytes (int16)
        layer_meta      ~  small JSON, negligible
        SCV1 header     =  35 bytes (fixed)

    For the council SKETCH default (K=256, D_v=8, ~600K weights / 8 ~= 75K
    groups, 600 pairs * 28 dims), this works out to:

        2*256*8 + 2*75000 + 2*600*28 + 35 ~= 4096 + 150000 + 33600 + 35
                                          ~= 187,731 bytes

    vs alpha (~360KB) - the ~2x rate saving is the delta score-axis attack vector.

    The proxy is a non-tight closed-form - it does NOT model brotli compression
    on the codebook (which typically cuts ~30-40%). For the OD-SUBSTRATE-WAVE-1
    first-anchor dispatch this constant offset is acceptable because the
    rate-term gradient flows through seg/pose; the absolute proxy is only used
    to size the rate-term scalar.

    Returns:
        Scalar tensor (on the same device as ``model``) - the proxy bytes.
    """
    import torch

    # Quantized-weight groups: every _QuantizedConv2d / _QuantizedLinear
    # contributes (numel / D_v) groups -> int16 indices.
    n_quantized_groups = 0
    for m in model.modules():
        # Lazy-import the marker types to avoid circular-import pain
        from tac.substrates.self_compress_nn.architecture import (
            _QuantizedConv2d,
            _QuantizedLinear,
        )

        if isinstance(m, (_QuantizedConv2d, _QuantizedLinear)):
            n_quantized_groups += int(m.weight.numel() // model.codebook.dv)
    codebook_bytes = int(model.codebook.k * model.codebook.dv * 2)
    indices_bytes = int(n_quantized_groups * 2)
    latent_bytes = int(model.latents.numel() * 2)
    fixed_header = 35  # SCV1_HEADER_SIZE
    bytes_proxy = float(
        codebook_bytes + indices_bytes + latent_bytes + fixed_header
    )
    device = next(model.parameters()).device
    return torch.tensor(bytes_proxy, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Codebook EMA step helper (van den Oord 2017 persistent buffers)
# ---------------------------------------------------------------------------

def _codebook_ema_step(model) -> None:
    """Run one van den Oord codebook EMA update on every quantized layer.

    Called AFTER ``optimizer.step()`` per CLAUDE.md "EMA - non-negotiable"
    codebook clause. The persistent ``ema_N`` / ``ema_sum`` buffers track
    cluster occupancy + sum so the codebook moves in lock-step with the
    surrounding weights.
    """
    import torch

    from tac.substrates.self_compress_nn.architecture import (
        _QuantizedConv2d,
        _QuantizedLinear,
    )

    with torch.no_grad():
        for m in model.modules():
            if not isinstance(m, (_QuantizedConv2d, _QuantizedLinear)):
                continue
            # Re-flatten the post-step weights into D_v groups.
            groups = m.weight.reshape(-1, model.codebook.dv)
            # Recompute indices (cheap; same logic as forward)
            dist = (
                groups.pow(2).sum(dim=1, keepdim=True)
                + model.codebook.codebook.pow(2).sum(dim=1).unsqueeze(0)
                - 2.0 * groups @ model.codebook.codebook.t()
            )
            indices = dist.argmin(dim=1)
            model.codebook.ema_step(groups, indices)


# ---------------------------------------------------------------------------
# Contest-compliant runtime emission (Catalog #146 contract)
# ---------------------------------------------------------------------------

def _write_runtime(submission_dir: Path) -> None:
    """Emit the contest-compliant ``inflate.sh`` + ``inflate.py`` pair.

    The substrate's monolithic ``0.bin`` is the SCV1 archive grammar; the
    runtime is a thin reader that calls ``parse_archive`` and renders frames.

    Per Catalog #146 semantics (extended to self_compress_nn):

    * 3-positional-arg ``inflate.sh`` ($1=archive_dir $2=output_dir $3=file_list)
    * ``set -euo pipefail``
    * No runtime network/dep fetches
    * No scorer code imports in ``inflate.py``
    * Per-video loop in ``inflate.py``
    * ``inflate.py`` <= 200 LOC (substrate inflate is ~155 LOC; we add ~30 LOC
      of CLI glue, total ~185 LOC - under HNeRV parity lesson L4 with the
      documented <= 200 LOC waiver for substrate engineering / codebook decode).
    """
    submission_dir.mkdir(parents=True, exist_ok=True)

    inflate_sh = (
        "#!/usr/bin/env bash\n"
        "# self_compress_nn contest-compliant inflate (WAVE-1-B 2026-05-12)\n"
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

    # inflate.py reads the SCV1 substrate archive via
    # tac.substrates.self_compress_nn. NO scorer code is imported. Per-video
    # loop over file_list. Per Catalog #146 'check_inflate_sh_handles_br_centrally'
    # has nothing to do here (no .br files) - runtime deps are torch + brotli +
    # numpy (substrate parser).
    inflate_py = (
        "#!/usr/bin/env python\n"
        "\"\"\"self_compress_nn contest-compliant inflate runtime (delta).\n"
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
        "from tac.substrates.self_compress_nn.inflate import inflate_one_video\n"
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

# Substrate-agnostic helpers delegate to the canonical
# ``tac.substrates._shared.trainer_skeleton`` module (CANON-DEDUP-1
# commit ac1cfc41). Thin wrappers preserve the original module-local
# names so existing call sites stay byte-faithful.

def _utc_now_iso() -> str:
    return _canon_utc_now_iso()


def _sha256_bytes(data: bytes) -> str:
    return _canon_sha256_bytes(data)


def _git_head_sha() -> str:
    return _canon_git_head_sha(REPO_ROOT)


def _pin_seeds(seed: int) -> None:
    _canon_pin_seeds(seed)


def _device_or_die(name: str, *, smoke: bool):
    return _canon_device_or_die(name, smoke=smoke, substrate_tag="self_compress_nn")


def _torch_version_string() -> str:
    return _canon_torch_version_string()


# ---------------------------------------------------------------------------
# Smoke main (CPU; no scorer load)
# ---------------------------------------------------------------------------

def _smoke_main(args: argparse.Namespace) -> int:
    """Tiny CPU smoke that proves the scaffold is wired (no scorer load)."""
    import torch

    from tac.substrates.self_compress_nn.architecture import (
        SelfCompressNnConfig,
        SelfCompressNnSubstrate,
    )

    _pin_seeds(args.seed)

    # 2-block tiny config that fits in CPU RAM. Note: codebook D_v MUST divide
    # every quantized weight tensor's numel - we pick D_v=4 here so the small
    # smoke channels (tiny grids, few channels) all pass divisibility.
    cfg = SelfCompressNnConfig(
        latent_dim=args.latent_dim,
        embed_dim=64,
        initial_grid_h=3,
        initial_grid_w=4,
        decoder_channels=(32, 16, 8),
        sin_frequency=args.sin_frequency,
        num_pairs=4,
        output_height=24,
        output_width=32,
        num_upsample_blocks=2,
        codebook_k=64,
        codebook_dv=4,
        codebook_ema_decay=args.codebook_ema_decay,
    )
    device = _device_or_die(args.device, smoke=True)
    model = SelfCompressNnSubstrate(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[smoke] self_compress_nn full params: {model.num_parameters():,}")
    proxy_bytes = _archive_bytes_proxy_closed_form(model)
    print(f"[smoke] delta archive proxy bytes: {int(proxy_bytes.item()):,}")
    for step in range(min(args.epochs, 3)):
        idx = torch.arange(cfg.num_pairs, device=device, dtype=torch.long)
        rgb_0, rgb_1, commit = model(idx)
        loss = rgb_0.abs().mean() + rgb_1.abs().mean() + 0.25 * commit
        opt.zero_grad()
        loss.backward()
        opt.step()
        # Codebook EMA step (van den Oord)
        _codebook_ema_step(model)
        print(
            f"[smoke] step {step}: loss={loss.item():.4f} "
            f"commit={float(commit.detach().item()):.4f}"
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
    """Full training entry point - requires CUDA + score-aware scorers.

    This path is OPERATOR-GATED. The wrapper (Vast.ai / Lightning / Modal)
    threads all TIER_1 flags + runs the auth-eval afterward per CLAUDE.md
    "Auth eval EVERYWHERE" + "Submission auth eval - BOTH CPU AND CUDA".
    """
    import torch

    from tac.differentiable_eval_roundtrip import (
        patch_upstream_yuv6_globally,
        unpatch_upstream_yuv6,
    )
    from tac.scorer import load_differentiable_scorers
    from tac.substrates.self_compress_nn.architecture import (
        SelfCompressNnConfig,
        SelfCompressNnSubstrate,
    )
    from tac.substrates.self_compress_nn.archive import pack_archive
    from tac.substrates.self_compress_nn.score_aware_loss import (
        SelfCompressNnScoreAwareLoss,
        SelfCompressNnScoreAwareLossWeights,
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

        # 5. Build model - delta council SKETCH defaults are tuned so every
        # quantized weight tensor's numel divides codebook_dv.
        cfg = SelfCompressNnConfig(
            latent_dim=args.latent_dim,
            sin_frequency=args.sin_frequency,
            num_pairs=n_pairs,
            output_height=EVAL_HW[0],
            output_width=EVAL_HW[1],
            codebook_k=args.codebook_k,
            codebook_dv=args.codebook_dv,
            codebook_ema_decay=args.codebook_ema_decay,
        )
        model = SelfCompressNnSubstrate(cfg).to(device)
        print(
            f"[full] self_compress_nn full params: {model.num_parameters():,} "
            f"(K={cfg.codebook_k}, D_v={cfg.codebook_dv})"
        )
        proxy_bytes = _archive_bytes_proxy_closed_form(model)
        print(
            f"[full] delta archive proxy bytes: {int(proxy_bytes.item()):,} "
            "(codebook + indices + latents + header)"
        )
        _stage("model_built")

        # 6. EMA shadow (CLAUDE.md non-negotiable). Codebook EMA buffers are
        # tracked separately inside _VQCodebook (van den Oord pattern, decay
        # 0.99); this EMA covers the surrounding pre-clustering weights at
        # 0.997 per CLAUDE.md weights default.
        ema = EMA(model, decay=args.ema_decay)
        _stage(f"ema_wired_decay_{args.ema_decay}")

        # 7. Score-aware Lagrangian (with delta MDL/codebook commit term)
        weights = SelfCompressNnScoreAwareLossWeights(
            alpha_rate=args.alpha_rate,
            beta_seg=args.beta_seg,
            gamma_pose=args.gamma_pose,
            pose_weight_scale=args.pose_weight_scale,
            lambda_mdl=args.lambda_mdl,
            contest_normalizer=CONTEST_NORMALIZER,
        )
        loss_fn = SelfCompressNnScoreAwareLoss(
            seg_scorer=segnet,
            pose_scorer=posenet,
            weights=weights,
        )
        _stage("lagrangian_built")

        # F3 GTScorerCache wire-in (F3-BACKPORT-WAVE-V2 2026-05-14).
        opt_ctx = _canon_build_optimized_training_context(
            args,
            scorers=(posenet, segnet),
            gt_pairs=pair_tensor,
            substrate_model=model,
            device=device,
        )
        gt_cache = opt_ctx.gt_cache
        if gt_cache is not None:
            print(gt_cache.summary_line())
            _stage("gt_scorer_cache_built")
        else:
            _stage("gt_scorer_cache_disabled")

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
                rgb_0, rgb_1, commit_loss = model(idx)
                # Frames live in [0,1]; the score-aware loss + eval-roundtrip
                # expect [0, 255]. Multiply at the boundary so the substrate
                # output is gradient-clean.
                rgb_0_255 = rgb_0 * 255.0
                rgb_1_255 = rgb_1 * 255.0
                gt = pair_tensor[idx]  # (B, 2, 3, H, W) already in [0, 255]
                gt_0 = gt[:, 0]
                gt_1 = gt[:, 1]
                # F3 GTScorerCache lookup (per-pair-index batched).
                gt_pose_batch = gt_seg_batch = None
                gt_seg_already_probs = None
                if gt_cache is not None:
                    gt_pose_batch, gt_seg_batch = gt_cache.lookup(
                        idx, device=device
                    )
                    gt_seg_already_probs = gt_cache.seg_already_probs
                loss, parts = loss_fn(
                    rgb_0_255, rgb_1_255, gt_0, gt_1,
                    archive_bytes_proxy, commit_loss,
                    apply_eval_roundtrip=True,
                    noise_std=args.noise_std,
                    gt_pose_batch=gt_pose_batch,
                    gt_seg_batch=gt_seg_batch,
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
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=args.grad_clip
                    )
                optimizer.step()
                # Update EMA shadow on full weights AFTER optimizer.step()
                ema.update(model)
                # Update codebook centroids via van den Oord persistent EMA
                # AFTER optimizer.step() per CLAUDE.md "EMA - non-negotiable"
                # codebook clause.
                _codebook_ema_step(model)
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
                    rgb_0_v, rgb_1_v, commit_v = model(val_indices)
                    # F3 cache lookup for val pairs.
                    val_pose_batch = val_seg_batch = None
                    val_seg_already_probs = None
                    if gt_cache is not None:
                        val_pose_batch, val_seg_batch = gt_cache.lookup(
                            val_indices, device=device
                        )
                        val_seg_already_probs = gt_cache.seg_already_probs
                    val_loss, _val_parts = loss_fn(
                        rgb_0_v * 255.0,
                        rgb_1_v * 255.0,
                        pair_tensor[val_indices, 0],
                        pair_tensor[val_indices, 1],
                        archive_bytes_proxy,
                        commit_v,
                        apply_eval_roundtrip=True,
                        noise_std=args.noise_std,
                        gt_pose_batch=val_pose_batch,
                        gt_seg_batch=val_seg_batch,
                        gt_seg_already_probs=val_seg_already_probs,
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
                                "[contest-CUDA] for promotion; "
                                "auth eval still required"
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

        # 11. Build the SCV1 archive bytes from the EMA shadow.
        #
        # The delta archive contains (codebook, cluster_indices, layer_meta,
        # latents, meta) - NOT full weight tensors. We reconstruct cluster
        # indices by walking the EMA-loaded model and calling
        # ``export_layer_meta_and_indices()``.
        archive_sha = ""
        archive_bytes = 0
        archive_zip_path = args.output_dir / "archive.zip"
        if not args.skip_archive_build:
            print(f"[full] building delta SCV1 archive from {ckpt_best_path} ...")
            ema_state = torch.load(
                ckpt_best_path, map_location="cpu", weights_only=False
            )
            sd = ema_state["state_dict"]

            # Reconstruct an in-memory model instance, load EMA weights, then
            # snapshot codebook + cluster_indices + latents.
            recon = SelfCompressNnSubstrate(cfg).to("cpu")
            # The EMA shadow includes the persistent codebook EMA buffers; we
            # must allow load_state_dict to skip any keys the EMA shadow may
            # have dropped (codebook EMA buffers are persistent_buffer; should
            # be present, but be defensive).
            recon.load_state_dict(sd, strict=False)
            layer_meta, cluster_indices = recon.export_layer_meta_and_indices()
            latents = recon.latents.detach().cpu()
            codebook = recon.codebook.codebook.detach().cpu()

            # Sidecar meta for inflate-time substrate reconstruction.
            # The delta inflate.py reads non-quantized weights (latent_embed,
            # biases) from this `extra_state` dict so the codebook+indices
            # only have to encode the quantized weight tensors.
            extra_state: dict[str, list[float]] = {}
            quantized_names = {entry["name"] for entry in layer_meta}
            for k, v in sd.items():
                if k in ("latents",):
                    continue
                if k in quantized_names:
                    continue
                if "ema_N" in k or "ema_sum" in k:
                    # Persistent VQ EMA buffers - not needed at inflate time
                    # (codebook is restored directly).
                    continue
                if k.endswith(".codebook"):
                    # codebook is stored separately
                    continue
                extra_state[k] = v.detach().cpu().to(torch.float32).flatten().tolist()

            meta = {
                "embed_dim": cfg.embed_dim,
                "initial_grid_h": cfg.initial_grid_h,
                "initial_grid_w": cfg.initial_grid_w,
                "decoder_channels": list(cfg.decoder_channels),
                "sin_frequency": cfg.sin_frequency,
                "output_height": cfg.output_height,
                "output_width": cfg.output_width,
                "num_upsample_blocks": cfg.num_upsample_blocks,
                "codebook_ema_decay": cfg.codebook_ema_decay,
                "commit_loss_weight": cfg.commit_loss_weight,
                "extra_state": extra_state,
                # extra_state shapes for inflate-time reshape
                "extra_state_shapes": {
                    k: list(sd[k].shape) for k in extra_state
                },
            }
            bin_bytes = pack_archive(
                codebook=codebook,
                layer_cluster_indices=cluster_indices,
                layer_meta=layer_meta,
                latents=latents,
                meta=meta,
            )
            (args.output_dir / "0.bin").write_bytes(bin_bytes)
            archive_sha = _sha256_bytes(bin_bytes)
            archive_bytes = len(bin_bytes)
            print(
                f"[full] wrote 0.bin ({archive_bytes} bytes, sha256={archive_sha})"
            )

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
                substrate_tag="self_compress_nn",
                device=device,
            )
            if auth_result is not None:
                contest_cuda_score = auth_result["auth_eval_cuda_score"]
                print(
                    f"[full] [contest-CUDA] score = "
                    f"{contest_cuda_score} "
                    f"(axis={auth_result['auth_eval_score_axis']}, "
                    f"lane_tag={auth_result['auth_eval_lane_tag']}, "
                    f"archive_sha256={archive_sha})"
                )
            _stage("auth_eval_cuda_done")

        # 13. Continual-learning posterior update (Catalog #128 atomic)
        if contest_cuda_score is not None and archive_sha:
            try:
                from tac.continual_learning import (
                    ContestResult,
                    posterior_update_locked,
                )

                # Per CLAUDE.md SIREN audit 2026-05-13 CRITICAL #1 + Catalog
                # #190: detect substrate dynamically from remote driver
                # provenance.json, then env vars, then nvidia-smi.
                _detected_substrate = _canon_detect_hardware_substrate(
                    axis="cuda",
                    substrate_tag="self_compress_nn",
                    provenance_path=args.output_dir / "provenance.json",
                    env_var_candidates=("SELF_COMPRESS_NN_GPU", "MODAL_GPU"),
                )
                result = ContestResult(
                    axis="cuda",
                    hardware_substrate=_detected_substrate,
                    architecture_class="lane_substrate_self_compress_nn_20260512",
                    score_value=contest_cuda_score,
                    evidence_tag="[contest-CUDA]",
                    archive_sha256=archive_sha,
                    archive_bytes=archive_bytes,
                    notes=(
                        "self_compress_nn (delta) WAVE-1-B first-anchor dispatch; "
                        f"epochs={args.epochs} K={args.codebook_k} "
                        f"D_v={args.codebook_dv}"
                    ),
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
                os.environ.get("SELF_COMPRESS_NN_ACTUAL_COST_USD"),
                field_name="SELF_COMPRESS_NN_ACTUAL_COST_USD",
            )
        except ValueError as exc:
            actual_cost_usd = None
            cost_band_anchor_skip_reason = (
                f"invalid_SELF_COMPRESS_NN_ACTUAL_COST_USD:{exc}"
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
                        "--dispatch-label",
                        f"self_compress_nn_{_utc_now_iso()}",
                        "--trainer",
                        "experiments/train_substrate_self_compress_nn.py",
                        "--platform",
                        os.environ.get("SELF_COMPRESS_NN_PLATFORM", "modal"),
                        "--gpu",
                        os.environ.get("SELF_COMPRESS_NN_GPU", "A100"),
                        "--epochs", str(args.epochs),
                        "--batch-size", str(args.batch_size),
                        "--actual-wall-clock-sec", str(train_elapsed_sec),
                        "--actual-cost-usd",
                        str(actual_cost_usd),
                        "--notes",
                        "WAVE-1-B self_compress_nn first-anchor dispatch",
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
                cost_band_anchor_skip_reason = (
                    "missing_SELF_COMPRESS_NN_ACTUAL_COST_USD"
                )
            elif not COST_BAND_TOOL.is_file():
                cost_band_anchor_skip_reason = "cost_band_tool_missing"
            else:
                cost_band_anchor_skip_reason = "nonpositive_train_elapsed_sec"

        # 15. Provenance manifest
        provenance = {
            "schema": "self_compress_nn_provenance_v1",
            "generated_at": _utc_now_iso(),
            "from_state_hash": "regen_per_session_below",
            "git_head": _git_head_sha(),
            "trainer": "experiments/train_substrate_self_compress_nn.py",
            "lane_id": "lane_substrate_self_compress_nn_20260512",
            "wave_dispatch_lane_id": (
                "lane_wave1_self_compress_nn_trainer_build_20260512"
            ),
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
            "promotion_eligible": False,  # gate on grand-council review of result
            "ready_for_exact_eval_dispatch": False,
            "literature_anchor": (
                "Selfcomp PR #56 1.017-bpw block-FP weight self-compression "
                "+ Quantizr 0.33 anchor 2026-04-21 + van den Oord VQ-VAE 2017 "
                "persistent codebook EMA"
            ),
            "council_phase_5_prediction": 0.17,
        }
        (args.output_dir / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"[full] wrote {args.output_dir / 'provenance.json'}")
        return 0

    finally:
        unpatch_upstream_yuv6(yuv6_token)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# META layer SubstrateContract (Catalog #241/#242 canonical migration; landed
# 2026-05-15 by CATALOG-241-BACKFILL-29-TRAINERS subagent). Decoration extincts
# the Z3 v2 silent-drift bug class for this substrate by binding (a) the
# trainer's claimed contract, (b) the recipe schema, (c) the lane registry,
# and (d) the cost-band envelope into ONE source-of-truth that fails-loud at
# decoration time if the contract violates canonical invariants.
# ---------------------------------------------------------------------------

SELF_COMPRESS_NN_SUBSTRATE_CONTRACT = SubstrateContract(
    # 2.1 Identity & lifecycle
    id="self_compress_nn",
    lane_id="lane_substrate_self_compress_nn_20260512",
    target_modes=("contest_one_video_replay", "research_substrate",),
    deployment_target="t4_contest_runtime",
    council_verdict_provenance=(
        ".omx/research/grand_council_fields_medal_substrate_design_20260512.md"
    ),
    # 2.2 Architecture & runtime (8 per Catalog #124)
    archive_grammar=(
        "SCNN1 monolithic single-file 0.bin: header + self-compressing NN decoder weights (block-fp + brotli per Selfcomp PR#56 paradigm) + scale tables (fp16) + per-pair embeddings"
    ),
    parser_section_manifest={
        "header": "SCNN1_magic_and_version",
        "decoder_weights": "block_fp_brotli_blob",
        "scale_tables": "fp16_brotli_blob",
        "pair_embeddings": "fp16_per_pair",
    },
    inflate_runtime_loc_budget=130,
    runtime_dep_closure=("torch>=2.5,<2.7", "brotli", "av",),
    export_format="custom",
    score_aware_loss="scorer_loss_terms_btchw",
    bolt_on_loc_budget=1400,
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
    recipe_canary_status="post_canary_dependent",
    recipe_video_input_strategy="per_dispatch_local_copy",
    recipe_canary_dependency="sane_hnerv",
    # 2.5 Cost band & GPU envelope (4)
    cost_band_epochs=2000,
    cost_band_gpu_key="A100",
    cost_band_platform_key="modal",
    cost_band_p50_usd=5.5,
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
            "Selfcomp block-fp self-compression; sensitivity captured by block scale tables + Hessian quant per CLAUDE.md grand council"
        ),
        "hook_bit_allocator_class": (
            "block-fp per-block bit allocation; per-substream not per-tensor"
        ),
        "hook_probe_disambiguator": (
            "single mechanism (block-fp self-compression); no 2+ defensible interpretations"
        ),
    },
)


@register_substrate(SELF_COMPRESS_NN_SUBSTRATE_CONTRACT)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.smoke:
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
