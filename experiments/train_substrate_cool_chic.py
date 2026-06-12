# SPDX-License-Identifier: MIT
"""Train the cool_chic substrate end-to-end on contest video.

Operator-callable training script per the Fields-medal grand council substrate
design wave (2026-05-12). PHASE-B2-BUILD wires ``_full_main`` so the trainer
is dispatch-ready as a HIGH-target attack on beating PR101's 0.193 [contest-CUDA].

Council prediction (Ladune et al., ICCV 2023 + grand council Phase 5):
target ~0.165 [contest-CUDA]. Cool-Chic flips the NeRV recipe: most parameters
live in **per-frame latents**, the renderer is a TINY shared synthesis MLP, and
rate is paid via an autoregressive (AR) density estimate over the latents.

Council-binding contract (CLAUDE.md non-negotiables) honored end-to-end:

- Train against ``upstream/videos/0.mkv`` decoded via pyav (NOT synthetic data;
  synthetic batches are FORBIDDEN outside ``--smoke`` per Catalog #114).
- Patch upstream ``rgb_to_yuv6`` via ``patch_upstream_yuv6_globally`` BEFORE
  scorer construction (PR #95/#106 contract — see CLAUDE.md "eval_roundtrip —
  NON-NEGOTIABLE" section).
- ``load_differentiable_scorers`` for SegNet/PoseNet (no scorer load at
  inflate; only at training).
- ``apply_eval_roundtrip_during_training`` inside the per-batch loop
  (eval_roundtrip=True default; never False per Catalog #5).
- ``tac.training.EMA(decay=0.997)`` update after every ``optimizer.step``;
  inference checkpoint = EMA shadow, NEVER live weights (CLAUDE.md "EMA —
  NON-NEGOTIABLE").
- Score-domain Lagrangian ``alpha*B(theta)/N + beta*d_seg + gamma*sqrt(d_pose)``
  per HNeRV parity lesson L6 + AR-prior NLL rate term (Cool-Chic distinguishing
  primitive).
- AdamW lr cosine annealing; gradient clip 1.0; NaN watchdog per Council D.
- End with CUDA auth eval on best EMA checkpoint per CLAUDE.md "Auth eval
  EVERYWHERE"; refuse MPS (Catalog #1); CPU permitted only with ``--smoke``.
- Continual-learning posterior update via ``posterior_update_locked``
  (Catalog #128 atomic fcntl).
- Cost-band anchor append via ``tools/append_cost_band_anchor.py``.
- Contest-compliant runtime emission per Catalog #146 semantics.
- TIER_1_OPERATOR_REQUIRED_FLAGS declared per Catalog #151 for wire-up.

Architectural risk (council Round 3 — NVIDIA-grade):
- AR prior NLL term is computed across ALL pairs each step (not per-batch);
  this is the substrate's distinguishing rate primitive but adds non-trivial
  compute per epoch. Acceptable on A100; T4 may need pair-batched AR.
- Per-pair latent grids dominate parameter budget (~180K of ~200K params);
  EMA shadow tracks these and can grow archive bytes if quantization scale
  drifts. The score-aware loss should pull latents toward Mallat-style
  sparsity automatically via the AR rate term.
- Score-aware gradient flow REQUIRES patched yuv6 + load_differentiable_scorers;
  PoseNet's @torch.no_grad() helper otherwise severs the chain.

Usage (smoke; CPU, tiny config, ~10 epochs, no scorer load)::

    .venv/bin/python experiments/train_substrate_cool_chic.py \\
        --video-path upstream/videos/0.mkv \\
        --output-dir experiments/results/cool_chic_smoke_<utc> \\
        --epochs 10 \\
        --device cpu --smoke

Usage (full; CUDA-required; threads from operator wrapper)::

    .venv/bin/python experiments/train_substrate_cool_chic.py \\
        --video-path upstream/videos/0.mkv \\
        --output-dir experiments/results/cool_chic_<utc> \\
        --epochs 2000 --batch-size 8 --lr 5e-4 --grad-clip 1.0 \\
        --device cuda
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

# Tier-1 optimization helpers (TIER-1-OPT-BATCH 2026-05-14; CLAUDE.md
# Catalog #172/#179). The O1 GT-scorer cache flag is declared but reserved
# pending per-substrate score_aware_loss API extension.
from tac.substrates._shared.trainer_skeleton import (
    build_optimized_training_context as _canon_build_optimized_training_context,
)

# Canonical substrate-trainer helpers (CANON-DEDUP-1 landing, commit ac1cfc41).
# Per CLAUDE.md "Beauty, simplicity, and developer experience": dedup the
# trainer-skeleton boilerplate by importing the shared, byte-faithful helpers.
from tac.substrates._shared.trainer_skeleton import (
    decode_real_pairs as _canonical_decode_real_pairs,
)
from tac.substrates._shared.trainer_skeleton import (
    detect_hardware_substrate as _canon_detect_hardware_substrate,
)
from tac.substrates._shared.trainer_skeleton import (
    device_or_die as _canonical_device_or_die,
)
from tac.substrates._shared.trainer_skeleton import (
    git_head_sha as _git_head_sha,
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
from tac.substrates._shared.trainer_skeleton import (
    vendor_shared_inflate_runtime as _canon_vendor_shared_inflate_runtime,
)

_SUBSTRATE_TAG = "cool_chic"

# ---------------------------------------------------------------------------
# Module paths + constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_VIDEO_PATH = REPO_ROOT / "upstream" / "videos" / "0.mkv"
DEFAULT_UPSTREAM_DIR = REPO_ROOT / "upstream"
DEFAULT_VIDEO_NAMES_FILE = REPO_ROOT / "upstream" / "public_test_video_names.txt"
CONTEST_AUTH_EVAL_SCRIPT = REPO_ROOT / "experiments" / "contest_auth_eval.py"
COST_BAND_TOOL = REPO_ROOT / "tools" / "append_cost_band_anchor.py"

EVAL_HW = (384, 512)
CAMERA_HW = (874, 1164)
N_PAIRS_FULL = 600
CONTEST_NORMALIZER = 37_545_489.0


# ---------------------------------------------------------------------------
# Catalog #151 manifest
# ---------------------------------------------------------------------------
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--video-path": {
        "env": "COOL_CHIC_VIDEO_PATH",
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
        "env": "COOL_CHIC_OUTPUT_DIR",
        "rationale": "custody location for checkpoints + archive + provenance",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--epochs": {
        "env": "COOL_CHIC_EPOCHS",
        "rationale": (
            "Cool-Chic substrate engineering pass; under-training silently "
            "regresses (council target: 2000)"
        ),
        "default": "2000",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--batch-size": {
        "env": "COOL_CHIC_BATCH_SIZE",
        "rationale": (
            "AR-prior NLL is computed across ALL pairs each step (substrate "
            "distinguishing primitive); per-batch latent decode + AR forward "
            "memory budget tuned for A100 24GB at batch=8 default"
        ),
        "default": "8",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--upstream-dir": {
        "env": "COOL_CHIC_UPSTREAM_DIR",
        "rationale": (
            "upstream/ root for scorer weights + evaluate.py; required for full "
            "training (non-smoke) and auth eval"
        ),
        "default": str(DEFAULT_UPSTREAM_DIR.relative_to(REPO_ROOT)),
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--device": {
        "env": "COOL_CHIC_DEVICE",
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
        prog="train_substrate_cool_chic",
        description="Train cool_chic per-frame AR latent substrate end-to-end (PHASE-B2-BUILD wired).",
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
        "--batch-size", type=int, default=8,
        help=(
            "Number of pair indices per batch. AR-prior NLL is computed across "
            "ALL pairs per step (substrate distinguishing primitive); per-batch "
            "latent decode + synthesis fits batch=8 in A100 24GB. Reduce to 4 "
            "if VRAM-bound (T4)."
        ),
    )
    p.add_argument("--lr", type=float, default=5e-4, help="AdamW learning rate.")
    p.add_argument("--weight-decay", type=float, default=1e-5, help="AdamW weight decay.")
    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="Gradient clip norm (Council D pattern).")
    p.add_argument("--seed", type=int, default=0, help="Manual seed (deterministic).")

    # ---- Substrate architecture knobs ----
    p.add_argument("--latent-channels-coarse", type=int, default=4,
                   help="Coarse-scale latent channels (council default 4).")
    p.add_argument("--latent-channels-fine", type=int, default=4,
                   help="Fine-scale latent channels (council default 4).")
    p.add_argument("--coarse-scale-factor", type=int, default=16,
                   help="Coarse-scale spatial downsample (H/16, W/16).")
    p.add_argument("--fine-scale-factor", type=int, default=8,
                   help="Fine-scale spatial downsample (H/8, W/8).")
    p.add_argument("--synthesis-hidden", type=int, default=32,
                   help="Hidden size of shared synthesis MLP.")
    p.add_argument("--synthesis-layers", type=int, default=3,
                   help="Layers of synthesis MLP (incl. output).")
    p.add_argument("--ar-prior-hidden", type=int, default=24,
                   help="Hidden size of AR prior conditional density network.")

    # ---- Lagrangian weights (score-aware) ----
    p.add_argument("--alpha-rate", type=float, default=25.0,
                   help="Rate-term coefficient (contest evaluate.py: 25.0).")
    p.add_argument("--beta-seg", type=float, default=100.0,
                   help="SegNet distortion coefficient (contest evaluate.py: 100.0).")
    p.add_argument("--gamma-pose", type=float, default=math.sqrt(10.0),
                   help="PoseNet sqrt-term coefficient (contest evaluate.py: sqrt(10)).")
    p.add_argument(
        "--pose-weight-scale", type=float, default=1.0,
        help=(
            "Optional operating-point multiplier layered on top of the contest "
            "sqrt(10) pose coefficient; default 1.0."
        ),
    )
    p.add_argument("--ar-rate-weight", type=float, default=1.0,
                   help="Weight on AR-prior NLL term (bits proxy).")
    p.add_argument("--noise-std", type=float, default=0.5,
                   help="STE noise std for eval-roundtrip simulation (Hotz fix).")

    # ---- EMA + scheduling ----
    p.add_argument("--ema-decay", type=float, default=0.997,
                   help="EMA decay (CLAUDE.md non-negotiable default 0.997 for weights).")
    p.add_argument("--val-every-epochs", type=int, default=10,
                   help="Run held-out proxy eval every N epochs.")
    p.add_argument("--val-pair-count", type=int, default=32,
                   help="Number of pairs reserved for held-out proxy validation.")

    # ---- Device / mode ----
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                   help="AUTHORITY/eval device. 'cpu' permitted only with --smoke. "
                        "MPS is FORBIDDEN here (CLAUDE.md 'MPS auth eval is NOISE').")
    p.add_argument(
        "--train-device",
        choices=["cuda", "cpu", "mps"],
        default=None,
        help=(
            "GRADIENT/training backend ONLY (may be 'mps' = Apple GPU). When set, "
            "the per-step decoder+scorer forward/backward runs here while the EXACT "
            "d_seg/d_pose authority eval that picks BEST stays on --device (CPU/CUDA). "
            "Used by experiments/launch_cool_chic_mps_smoke.py; MPS is NEVER score "
            "authority. Defaults to --device (single-device legacy)."
        ),
    )
    p.add_argument("--smoke", action="store_true",
                   help="Tiny CPU smoke (no scorer load, tiny config).")
    p.add_argument("--max-pairs", type=int, default=None,
                   help="Cap on number of pairs decoded (debug only).")

    # ---- Post-train artifacts ----
    p.add_argument("--skip-auth-eval", action="store_true",
                   help="Skip the final auth-eval subprocess.")
    p.add_argument("--skip-archive-build", action="store_true",
                   help="Skip building the archive.zip.")
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

# The canonical decoder lives in tac.substrates._shared.trainer_skeleton;
# this thin adapter pins the substrate tag for importlib-spec collision-safety
# (per the helper's substrate_tag-keyed module name) and keeps the existing
# call-site signature stable.

def _decode_real_pairs(video_path: Path, *, n_pairs: int, max_pairs: int | None = None):
    """Adapter: forward to the canonical helper with our substrate tag."""
    return _canonical_decode_real_pairs(
        video_path,
        n_pairs=n_pairs,
        substrate_tag=_SUBSTRATE_TAG,
        max_pairs=max_pairs,
        repo_root=REPO_ROOT,
    )


# ---------------------------------------------------------------------------
# Lagrangian helpers
# ---------------------------------------------------------------------------

def _archive_bytes_proxy_closed_form(model):
    """Closed-form upper-bound on archive bytes for the rate term.

    Cool-Chic pays bytes via:
    1. Synthesis MLP weights (~10K params * 2 fp16 = 20K bytes)
    2. AR prior weights (~10K params * 2 fp16 = 20K bytes)
    3. Per-pair latents (int16): num_pairs * (C_c * h_c * w_c + C_f * h_f * w_f) * 2

    The proxy is constant during training (no parameter dependence) so this
    term is a constant offset; the AR-prior NLL term (passed separately) is
    the differentiable rate primitive.
    """
    import torch

    n_decoder = sum(
        p.numel() for n, p in model.named_parameters()
        if not n.startswith("latents_")
    )
    n_latent_elems = sum(
        p.numel() for n, p in model.named_parameters()
        if n.startswith("latents_")
    )
    bytes_proxy = float(n_decoder * 2 + n_latent_elems * 2)
    device = next(model.parameters()).device
    return torch.tensor(bytes_proxy, dtype=torch.float32, device=device)


# ---------------------------------------------------------------------------
# Contest-compliant runtime emission (Catalog #146 contract)
# ---------------------------------------------------------------------------

def _write_runtime(submission_dir: Path) -> None:
    """Emit the contest-compliant ``inflate.sh`` + ``inflate.py`` pair."""
    submission_dir.mkdir(parents=True, exist_ok=True)
    runtime_pkg = submission_dir / "src" / "tac" / "substrates" / "cool_chic"
    runtime_pkg.mkdir(parents=True, exist_ok=True)
    # Vendor only inflate-time modules. Do not ship score-aware training code
    # or scorer imports in the runtime tree.
    for pkg_init in (
        submission_dir / "src" / "tac" / "__init__.py",
        submission_dir / "src" / "tac" / "substrates" / "__init__.py",
        runtime_pkg / "__init__.py",
    ):
        pkg_init.write_text("", encoding="utf-8")
    substrate_src = REPO_ROOT / "src" / "tac" / "substrates" / "cool_chic"
    for name in ("architecture.py", "archive.py", "inflate.py"):
        shutil.copy2(substrate_src / name, runtime_pkg / name)
    _canon_vendor_shared_inflate_runtime(submission_dir, repo_root=REPO_ROOT)

    inflate_sh = (
        "#!/usr/bin/env bash\n"
        "# cool_chic contest-compliant inflate (PHASE-B2-BUILD wired 2026-05-12)\n"
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
        "\"\"\"cool_chic contest-compliant inflate runtime.\n"
        "\n"
        "Reads archive_dir/0.bin via the packaged substrate parser, then writes\n"
        "one contest .raw tensor stream per file_list entry.\n"
        "No scorer-network imports (strict-scorer-rule contract).\n"
        "\"\"\"\n"
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "HERE = Path(__file__).resolve().parent\n"
        "sys.path.insert(0, str(HERE / 'src'))\n"
        "from tac.substrates.cool_chic.inflate import inflate_one_video, raw_output_path, select_inflate_device\n"
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
        "    device = select_inflate_device()\n"
        "    for line in file_list_path.read_text(encoding='utf-8').splitlines():\n"
        "        line = line.strip()\n"
        "        if not line:\n"
        "            continue\n"
        "        inflate_one_video(archive_bytes, raw_output_path(output_dir, line), device=device)\n"
        "    return 0\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    sys.exit(main())\n"
    )
    (submission_dir / "inflate.py").write_text(inflate_py, encoding="utf-8")


def _build_archive_zip(archive_zip_path: Path, *, bin_bytes: bytes, submission_dir: Path) -> None:
    """Deterministic charged archive.zip containing only the data packet."""
    archive_zip_path.parent.mkdir(parents=True, exist_ok=True)
    fixed_ts = (2026, 1, 1, 0, 0, 0)
    with zipfile.ZipFile(archive_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zi = zipfile.ZipInfo("0.bin", date_time=fixed_ts)
        zi.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(zi, bin_bytes)
        # archive.zip is the charged data packet. Runtime files live beside it
        # in submission_dir and are evaluated through --inflate-sh custody.


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _device_or_die(name: str, *, smoke: bool):
    """Adapter: forward to the canonical helper with our substrate tag."""
    return _canonical_device_or_die(name, smoke=smoke, substrate_tag=_SUBSTRATE_TAG)


def _device_or_die_cpu_advisory_for_mps_split():
    """Resolve the CPU AUTHORITY device for an MPS-trained advisory split run.

    The MPS-split path (``--device cpu --train-device mps``) trains the gradient
    on the Apple GPU and uses the CPU as the (NON-PROMOTABLE, advisory) authority
    for BEST-selection + archive auth-eval — exactly the validated basin design in
    ``experiments/launch_split_by_head_basin.py``. The canonical ``device_or_die``
    refuses bare ``cpu`` outside ``--smoke``; here we route through its
    ``allow_full_cpu`` advisory exception (the run is advisory by construction —
    the archive must still go through ``upstream/evaluate.py`` for any score).
    """
    return _canonical_device_or_die(
        "cpu", smoke=False, substrate_tag=_SUBSTRATE_TAG, allow_full_cpu=True
    )


def _resolve_train_device(name: str | None, authority_device):
    """Resolve the TRAIN/GRADIENT device (may be ``mps``) given the AUTHORITY device.

    The split mirrors ``src/tac/torch_vehicle/driver.py``: the gradient graph may
    run on the Apple GPU (``mps``) for the ~100x fwd+bwd speedup, while the EXACT
    d_seg/d_pose that pick BEST + the archive auth-eval ALWAYS run on the CPU/CUDA
    authority. MPS is NEVER score authority (CLAUDE.md "MPS auth eval is NOISE").

    When ``name`` is ``mps`` we apply ``patch_scorer_for_mps()`` (the upstream
    BatchNorm-backward stride fix) BEFORE the scorers are loaded onto MPS and
    return its report so the trainer can surface which device-compat patches fired.

    Args:
        name: the requested train device, one of {None, "cpu", "cuda", "mps"}.
            None means "single-device legacy" — train on the authority device.
        authority_device: the resolved ``torch.device`` from ``_device_or_die``.

    Returns:
        ``(train_device, mps_patch_report)`` where ``train_device`` is a
        ``torch.device`` and ``mps_patch_report`` is a dict (empty unless mps).
    """
    import torch

    if name is None:
        return authority_device, {}
    if name == "mps":
        if not (
            getattr(torch.backends, "mps", None) is not None
            and torch.backends.mps.is_available()
        ):
            raise SystemExit(
                f"[{_SUBSTRATE_TAG}] --train-device mps requested but torch MPS is "
                "not available on this host"
            )
        from tac.torch_mps_compat import patch_scorer_for_mps

        report = patch_scorer_for_mps()  # BatchNorm-backward MPS stride fix
        return torch.device("mps"), report
    if name == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit(
                f"[{_SUBSTRATE_TAG}] --train-device cuda requested but cuda not available"
            )
        return torch.device("cuda"), {}
    if name == "cpu":
        return torch.device("cpu"), {}
    raise SystemExit(f"[{_SUBSTRATE_TAG}] unknown --train-device {name!r}")


# ---------------------------------------------------------------------------
# Smoke main (CPU; no scorer load)
# ---------------------------------------------------------------------------

def _smoke_main(args: argparse.Namespace) -> int:
    """Tiny CPU smoke that proves the scaffold is wired (no scorer load)."""
    import torch

    from tac.substrates.cool_chic.architecture import CoolChicConfig, CoolChicSubstrate

    _pin_seeds(args.seed)

    cfg = CoolChicConfig(
        latent_channels_coarse=args.latent_channels_coarse,
        latent_channels_fine=args.latent_channels_fine,
        coarse_scale_factor=args.coarse_scale_factor,
        fine_scale_factor=args.fine_scale_factor,
        synthesis_hidden=16,
        synthesis_layers=2,
        ar_prior_hidden=12,
        num_pairs=4,
        output_height=64,
        output_width=96,
    )
    device = _device_or_die(args.device, smoke=True)
    model = CoolChicSubstrate(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[smoke] cool_chic params: {model.num_parameters():,}")
    for step in range(min(args.epochs, 3)):
        idx = torch.arange(cfg.num_pairs, device=device, dtype=torch.long)
        rgb_0, rgb_1 = model(idx)
        ar_log_p = model.compute_ar_log_prob()
        loss = rgb_0.abs().mean() + rgb_1.abs().mean() + (-ar_log_p) * 1e-8
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"[smoke] step {step}: loss={loss.item():.4f} ar_log_p={ar_log_p.item():.2f}")

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
    from tac.substrates.cool_chic.architecture import (
        CoolChicConfig,
        CoolChicSubstrate,
    )
    from tac.substrates.cool_chic.archive import pack_archive
    from tac.substrates.cool_chic.score_aware_loss import (
        CoolChicScoreAwareLoss,
        ScoreAwareLossWeights,
    )
    from tac.training import EMA

    # 1. Pin seeds
    _pin_seeds(args.seed)
    # AUTHORITY device (CPU-TRUSTED or CUDA) — the exact d_seg/d_pose that pick
    # the BEST EMA checkpoint AND the archive build + auth-eval run HERE. MPS is
    # FORBIDDEN as the authority (CLAUDE.md "MPS auth eval is NOISE": 23x pose
    # drift corrupts argmax). _device_or_die refuses mps + refuses bare cpu
    # outside --smoke (we keep the canonical production gate intact) — EXCEPT the
    # MPS-split advisory path: when --train-device mps is requested, the Apple GPU
    # carries the gradient and the CPU authority is the (NON-PROMOTABLE, advisory)
    # BEST-selection axis, exactly the validated launch_split_by_head_basin.py
    # design. That advisory CPU authority is admitted via allow_full_cpu.
    _mps_split_requested = args.train_device == "mps"
    if args.device == "cpu" and _mps_split_requested:
        # MPS-trained / CPU-authority advisory split (basin pattern). CPU is the
        # advisory authority, not a promotion claim.
        authority_device = _device_or_die_cpu_advisory_for_mps_split()
    else:
        authority_device = _device_or_die(args.device, smoke=False)
    # TRAIN/GRADIENT device — the per-step decoder+scorer forward/backward MAY run
    # on mps (the Apple GPU; the whole point of the local-MPS path). It is NEVER
    # score authority. Defaults to the authority device (single-device legacy).
    train_device, mps_patch_report = _resolve_train_device(
        args.train_device, authority_device
    )
    split_device = train_device != authority_device
    # ``device`` keeps the existing name used by the body for the AUTHORITY device
    # (archive build / auth-eval / continual-learning), so the downstream code is
    # unchanged; the new ``train_device`` only governs the gradient graph.
    device = authority_device

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage_log: list[dict[str, Any]] = []

    def _stage(name: str) -> None:
        stage_log.append({"stage": name, "at": _utc_now_iso()})

    _stage("seed_pinned")
    if split_device:
        print(
            f"[full] SPLIT-DEVICE: train/gradient={train_device} (research-signal) "
            f"| authority/eval={authority_device} (CPU-TRUSTED). MPS is NEVER score "
            f"authority (CLAUDE.md 'MPS auth eval is NOISE'). mps_patches="
            f"{mps_patch_report}",
            flush=True,
        )
        _stage(f"split_device_train_{train_device}_authority_{authority_device}")

    # 2. Patch upstream rgb_to_yuv6 BEFORE scorer construction
    yuv6_token = patch_upstream_yuv6_globally()
    _stage("upstream_yuv6_patched")

    try:
        # 3. Load differentiable scorers.
        # TRAINING scorers live on ``train_device`` (the gradient backend, may be
        # MPS). When split, a SECOND authority scorer set lives on the CPU/CUDA
        # authority — the BEST-selection val eval routes through THAT set so the
        # score that picks the inference checkpoint is NEVER an MPS forward pass
        # (CLAUDE.md "MPS auth eval is NOISE"). When not split, the two handles are
        # the same object (single-device legacy, byte-identical to the prior path).
        posenet, segnet = load_differentiable_scorers(
            args.upstream_dir, device=train_device
        )
        for p in list(posenet.parameters()) + list(segnet.parameters()):
            p.requires_grad_(False)
        posenet.eval()
        segnet.eval()
        if split_device:
            posenet_auth, segnet_auth = load_differentiable_scorers(
                args.upstream_dir, device=authority_device
            )
            for p in list(posenet_auth.parameters()) + list(segnet_auth.parameters()):
                p.requires_grad_(False)
            posenet_auth.eval()
            segnet_auth.eval()
            _stage("scorers_loaded_split_train_and_authority")
        else:
            posenet_auth, segnet_auth = posenet, segnet
            _stage("scorers_loaded")

        # 4. Decode real frame pairs
        print(f"[full] decoding pairs from {args.video_path} ...")
        pair_tensor = _decode_real_pairs(
            args.video_path, n_pairs=N_PAIRS_FULL, max_pairs=args.max_pairs,
        )
        n_pairs = int(pair_tensor.shape[0])
        print(f"[full] decoded {n_pairs} pairs at {EVAL_HW}")
        # The GT pairs live on the TRAIN device (the per-step loss consumes them);
        # the authority val eval moves its slice to the authority device explicitly.
        pair_tensor = pair_tensor.to(train_device)
        _stage(f"pairs_decoded_{n_pairs}")

        # Held-out validation indices (on the train device; the authority val eval
        # moves its slice to the authority device).
        val_count = max(1, min(args.val_pair_count, max(1, n_pairs // 8)))
        val_idx_start = n_pairs - val_count
        train_indices = torch.arange(0, val_idx_start, device=train_device, dtype=torch.long)
        val_indices = torch.arange(val_idx_start, n_pairs, device=train_device, dtype=torch.long)

        # 5. Build model
        cfg = CoolChicConfig(
            latent_channels_coarse=args.latent_channels_coarse,
            latent_channels_fine=args.latent_channels_fine,
            coarse_scale_factor=args.coarse_scale_factor,
            fine_scale_factor=args.fine_scale_factor,
            synthesis_hidden=args.synthesis_hidden,
            synthesis_layers=args.synthesis_layers,
            ar_prior_hidden=args.ar_prior_hidden,
            num_pairs=n_pairs,
            output_height=EVAL_HW[0],
            output_width=EVAL_HW[1],
        )
        # The model trains on the gradient backend (train_device, may be MPS).
        model = CoolChicSubstrate(cfg).to(train_device)
        print(f"[full] cool_chic params: {model.num_parameters():,}")
        _stage("model_built")

        # 6. EMA shadow (CLAUDE.md non-negotiable). The shadow tracks the live
        # (train_device) weights; the authority val eval moves a CPU copy of the
        # shadow onto the authority device before scoring.
        ema = EMA(model, decay=args.ema_decay)
        _stage(f"ema_wired_decay_{args.ema_decay}")

        # 7. Score-aware Lagrangian.
        weights = ScoreAwareLossWeights(
            alpha_rate=args.alpha_rate,
            beta_seg=args.beta_seg,
            gamma_pose=args.gamma_pose,
            pose_weight_scale=args.pose_weight_scale,
            contest_normalizer=CONTEST_NORMALIZER,
            ar_rate_weight=args.ar_rate_weight,
        )
        # Training loss runs on the gradient backend (train_device scorers).
        loss_fn = CoolChicScoreAwareLoss(
            seg_scorer=segnet, pose_scorer=posenet, weights=weights,
        )
        # AUTHORITY loss for BEST-selection val eval runs on the CPU/CUDA authority
        # scorers (NEVER MPS). When not split, this is the same loss object.
        if split_device:
            loss_fn_auth = CoolChicScoreAwareLoss(
                seg_scorer=segnet_auth, pose_scorer=posenet_auth, weights=weights,
            )
        else:
            loss_fn_auth = loss_fn
        _stage("lagrangian_built")

        # F3 GTScorerCache wire-in (F3-BACKPORT-WAVE-V2 2026-05-14). The canonical
        # context builds an AutocastConfig that REFUSES device_type='mps' (CLAUDE.md
        # "MPS auth eval is NOISE"), so on the MPS-split path we pass the AUTHORITY
        # device here (autocast is CPU/CUDA-only). The GT-scorer cache is reserved
        # (--enable-gt-scorer-cache defaults False); when enabled it caches GT-scorer
        # outputs and would need a train_device-aware extension — gated for now.
        opt_ctx_device = authority_device if split_device else train_device
        if split_device and getattr(args, "enable_gt_scorer_cache", False):
            raise SystemExit(
                f"[{_SUBSTRATE_TAG}] --enable-gt-scorer-cache is not yet supported on "
                "the MPS-split path (the GT-scorer cache must be built on the train "
                "device; deferred). Run without it for the MPS smoke."
            )
        opt_ctx = _canon_build_optimized_training_context(
            args,
            scorers=(posenet, segnet),
            gt_pairs=pair_tensor,
            substrate_model=model,
            device=opt_ctx_device,
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

        nan_strike = 0
        max_nan_strikes = 3

        for epoch in range(args.epochs):
            model.train()
            perm = train_indices[torch.randperm(n_train, device=train_device)]
            epoch_loss_sum = 0.0
            epoch_batches = 0
            for start in range(0, n_train, batch_size):
                idx = perm[start : start + batch_size]
                if idx.numel() == 0:
                    continue
                rgb_0, rgb_1 = model(idx)
                rgb_0_255 = rgb_0 * 255.0
                rgb_1_255 = rgb_1 * 255.0
                gt = pair_tensor[idx]
                gt_0 = gt[:, 0]
                gt_1 = gt[:, 1]
                # AR-prior NLL is the substrate's distinguishing rate primitive.
                # Computed across ALL pairs (not just batch) — it's the rate
                # cost of the FULL latent ensemble, not the batch's slice.
                ar_log_prob = model.compute_ar_log_prob()
                # F3 GTScorerCache lookup (per-pair-index batched).
                gt_pose_batch = gt_seg_batch = None
                gt_seg_already_probs = None
                if gt_cache is not None:
                    gt_pose_batch, gt_seg_batch = gt_cache.lookup(
                        idx, device=train_device
                    )
                    gt_seg_already_probs = gt_cache.seg_already_probs
                loss, parts = loss_fn(
                    rgb_0_255, rgb_1_255, gt_0, gt_1,
                    archive_bytes_proxy, ar_log_prob,
                    apply_eval_roundtrip=True,
                    noise_std=args.noise_std,
                    gt_pose_batch=gt_pose_batch,
                    gt_seg_batch=gt_seg_batch,
                    gt_seg_already_probs=gt_seg_already_probs,
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

            # 10. Validation + best-ckpt selection. The BEST-selection score is the
            # AUTHORITY score (CPU/CUDA scorers, NEVER MPS). On the SPLIT path we
            # snapshot the EMA shadow to CPU, load it into a fresh AUTHORITY model
            # on the authority device, and score with the authority scorers there —
            # so the number that picks the inference checkpoint carries ZERO MPS
            # drift (CLAUDE.md "MPS auth eval is NOISE"). On the single-device path
            # the original in-place apply-shadow path is preserved byte-for-byte.
            if (epoch + 1) % args.val_every_epochs == 0 or epoch == args.epochs - 1:
                if split_device:
                    ema_snapshot_cpu = {
                        k: v.detach().cpu().clone() for k, v in ema.state_dict().items()
                    }
                    auth_model = CoolChicSubstrate(cfg).to(authority_device)
                    auth_model.load_state_dict(
                        {k: v.to(authority_device) for k, v in ema_snapshot_cpu.items()}
                    )
                    auth_model.eval()
                    auth_bytes_proxy = _archive_bytes_proxy_closed_form(auth_model)
                    val_idx_auth = val_indices.to(authority_device)
                    gt_val_0 = pair_tensor[val_indices, 0].to(authority_device)
                    gt_val_1 = pair_tensor[val_indices, 1].to(authority_device)
                    with torch.no_grad():
                        rgb_0_v, rgb_1_v = auth_model(val_idx_auth)
                        ar_log_p_v = auth_model.compute_ar_log_prob()
                        val_loss, _val_parts = loss_fn_auth(
                            rgb_0_v * 255.0, rgb_1_v * 255.0,
                            gt_val_0, gt_val_1,
                            auth_bytes_proxy, ar_log_p_v,
                            apply_eval_roundtrip=True,
                            noise_std=args.noise_std,
                        )
                    val_lag = float(val_loss.detach().item())
                    del auth_model
                else:
                    orig_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    ema.apply(model)
                    model.eval()
                    with torch.no_grad():
                        rgb_0_v, rgb_1_v = model(val_indices)
                        ar_log_p_v = model.compute_ar_log_prob()
                        # F3 cache lookup for val pairs.
                        val_pose_batch = val_seg_batch = None
                        val_seg_already_probs = None
                        if gt_cache is not None:
                            val_pose_batch, val_seg_batch = gt_cache.lookup(
                                val_indices, device=device
                            )
                            val_seg_already_probs = gt_cache.seg_already_probs
                        val_loss, _val_parts = loss_fn(
                            rgb_0_v * 255.0, rgb_1_v * 255.0,
                            pair_tensor[val_indices, 0],
                            pair_tensor[val_indices, 1],
                            archive_bytes_proxy, ar_log_p_v,
                            apply_eval_roundtrip=True,
                            noise_std=args.noise_std,
                            gt_pose_batch=val_pose_batch,
                            gt_seg_batch=val_seg_batch,
                            gt_seg_already_probs=val_seg_already_probs,
                        )
                    val_lag = float(val_loss.detach().item())
                    model.load_state_dict(orig_state)
                    model.train()
                axis_tag = (
                    "[contest-CPU advisory]" if split_device else "[authority-device]"
                )
                print(
                    f"[full] epoch {epoch + 1}/{args.epochs} "
                    f"train_avg_loss={avg_loss:.6f} val_lagrangian={val_lag:.6f} "
                    f"{axis_tag} (best_so_far={best_val_lag:.6f} @ ep{best_epoch + 1})"
                )
                if val_lag < best_val_lag and math.isfinite(val_lag):
                    best_val_lag = val_lag
                    best_epoch = epoch
                    ema_state = ema.state_dict()
                    torch.save(
                        {
                            "state_dict": {k: v.detach().cpu() for k, v in ema_state.items()},
                            "config": asdict(cfg),
                            "ema_decay": args.ema_decay,
                            "best_val_lagrangian": val_lag,
                            "best_epoch": int(epoch),
                            "saved_at_utc": _utc_now_iso(),
                            "train_device": str(train_device),
                            "authority_device": str(authority_device),
                            "best_selection_axis": (
                                "[contest-CPU advisory] (MPS-trained, CPU-authority "
                                "BEST-selected); NON-PROMOTABLE until upstream/evaluate.py"
                                if split_device
                                else "[contest-CUDA] for promotion; auth eval still required"
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

        # 11. Build the CCV1 archive bytes from the EMA shadow
        archive_sha = ""
        archive_bytes = 0
        archive_zip_path = args.output_dir / "archive.zip"
        if not args.skip_archive_build:
            print(f"[full] building archive from {ckpt_best_path} ...")
            ema_state = torch.load(ckpt_best_path, map_location="cpu", weights_only=False)
            sd = ema_state["state_dict"]

            # Split state_dict into synthesis / ar_prior / latents per CCV1 grammar
            synthesis_sd: dict[str, torch.Tensor] = {}
            ar_prior_sd: dict[str, torch.Tensor] = {}
            latents_coarse: torch.Tensor | None = None
            latents_fine: torch.Tensor | None = None
            for k, v in sd.items():
                if k == "latents_coarse":
                    latents_coarse = v.detach().cpu()
                elif k == "latents_fine":
                    latents_fine = v.detach().cpu()
                elif k.startswith("synthesis."):
                    synthesis_sd[k[len("synthesis."):]] = v
                elif k.startswith("ar_prior_coarse."):
                    ar_prior_sd[f"coarse.{k[len('ar_prior_coarse.'):]}"] = v
                elif k.startswith("ar_prior_fine."):
                    ar_prior_sd[f"fine.{k[len('ar_prior_fine.'):]}"] = v
                elif k == "frame_offset":
                    # Carry frame_offset in synthesis state (the synthesis is
                    # the consumer; inflate.py loads synthesis state into the
                    # full model so this is the closest match)
                    synthesis_sd[k] = v
                else:
                    # Catch any unexpected key by routing to synthesis_sd
                    print(f"[full] WARN: unexpected state_dict key {k}; routing to synthesis_sd")
                    synthesis_sd[k] = v

            if latents_coarse is None or latents_fine is None:
                raise RuntimeError("EMA shadow missing latents_coarse / latents_fine")

            meta = {
                "coarse_scale_factor": cfg.coarse_scale_factor,
                "fine_scale_factor": cfg.fine_scale_factor,
                "synthesis_hidden": cfg.synthesis_hidden,
                "synthesis_layers": cfg.synthesis_layers,
                "ar_prior_hidden": cfg.ar_prior_hidden,
                "output_height": cfg.output_height,
                "output_width": cfg.output_width,
            }
            bin_bytes = pack_archive(
                synthesis_sd, ar_prior_sd, latents_coarse, latents_fine, meta,
            )
            (args.output_dir / "0.bin").write_bytes(bin_bytes)
            archive_sha = _sha256_bytes(bin_bytes)
            archive_bytes = len(bin_bytes)
            print(f"[full] wrote 0.bin ({archive_bytes} bytes, sha256={archive_sha})")

            submission_dir = args.output_dir / "submission"
            _write_runtime(submission_dir)
            (submission_dir / "0.bin").write_bytes(bin_bytes)
            _build_archive_zip(
                archive_zip_path, bin_bytes=bin_bytes, submission_dir=submission_dir,
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
                substrate_tag="cool_chic",
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
                    substrate_tag="cool_chic",
                    provenance_path=args.output_dir / "provenance.json",
                    env_var_candidates=("COOL_CHIC_GPU", "MODAL_GPU"),
                )
                result = ContestResult(
                    axis="cuda",
                    hardware_substrate=_detected_substrate,
                    architecture_class="lane_substrate_cool_chic_20260512",
                    score_value=contest_cuda_score,
                    evidence_tag="[contest-CUDA]",
                    archive_sha256=archive_sha,
                    archive_bytes=archive_bytes,
                    notes=f"cool_chic first-anchor dispatch; epochs={args.epochs}",
                    observed_at_utc=_utc_now_iso(),
                )
                update = posterior_update_locked(result)
                print(
                    f"[full] posterior_update: accepted={update.accepted} "
                    f"reason={update.reason!r}"
                )
            except Exception as exc:
                print(f"[full] posterior_update_locked failed: {exc}", file=sys.stderr)

        # 14. Cost-band anchor
        cost_band_anchor_appended = False
        cost_band_anchor_skip_reason: str | None = None
        try:
            from tac.cost_band_calibration import parse_actual_cost_usd

            actual_cost_usd = parse_actual_cost_usd(
                os.environ.get("COOL_CHIC_ACTUAL_COST_USD"),
                field_name="COOL_CHIC_ACTUAL_COST_USD",
            )
        except ValueError as exc:
            actual_cost_usd = None
            cost_band_anchor_skip_reason = f"invalid_COOL_CHIC_ACTUAL_COST_USD:{exc}"
        if COST_BAND_TOOL.is_file() and train_elapsed_sec > 0 and actual_cost_usd is not None:
            try:
                proc = subprocess.run(
                    [
                        sys.executable, str(COST_BAND_TOOL),
                        "--dispatch-label", f"cool_chic_{_utc_now_iso()}",
                        "--trainer", "experiments/train_substrate_cool_chic.py",
                        "--platform", os.environ.get("COOL_CHIC_PLATFORM", "modal"),
                        "--gpu", os.environ.get("COOL_CHIC_GPU", "A100"),
                        "--epochs", str(args.epochs),
                        "--batch-size", str(args.batch_size),
                        "--actual-wall-clock-sec", str(train_elapsed_sec),
                        "--actual-cost-usd", str(actual_cost_usd),
                        "--notes", "PHASE-B2-BUILD first-anchor dispatch",
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
                cost_band_anchor_skip_reason = "missing_COOL_CHIC_ACTUAL_COST_USD"
            elif not COST_BAND_TOOL.is_file():
                cost_band_anchor_skip_reason = "cost_band_tool_missing"
            else:
                cost_band_anchor_skip_reason = "nonpositive_train_elapsed_sec"

        # 15. Provenance manifest
        provenance = {
            "schema": "cool_chic_provenance_v1",
            "generated_at": _utc_now_iso(),
            "from_state_hash": "regen_per_session_below",
            "git_head": _git_head_sha(),
            "trainer": "experiments/train_substrate_cool_chic.py",
            "lane_id": "lane_substrate_cool_chic_20260512",
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
# META layer SubstrateContract (Catalog #241/#242 canonical migration; landed
# 2026-05-15 by CATALOG-241-BACKFILL-29-TRAINERS subagent). Decoration extincts
# the Z3 v2 silent-drift bug class for this substrate by binding (a) the
# trainer's claimed contract, (b) the recipe schema, (c) the lane registry,
# and (d) the cost-band envelope into ONE source-of-truth that fails-loud at
# decoration time if the contract violates canonical invariants.
# ---------------------------------------------------------------------------

COOL_CHIC_SUBSTRATE_CONTRACT = SubstrateContract(
    # 2.1 Identity & lifecycle
    id="cool_chic",
    lane_id="lane_substrate_cool_chic_20260512",
    target_modes=("contest_one_video_replay", "research_substrate",),
    deployment_target="t4_contest_runtime",
    council_verdict_provenance=(
        ".omx/research/grand_council_fields_medal_substrate_design_20260512.md"
    ),
    # 2.2 Architecture & runtime (8 per Catalog #124)
    archive_grammar=(
        "CCV1 monolithic single-file 0.bin: header (magic=CCV1) + tiny synthesis MLP weights (fp16+brotli) + per-frame latents (fp4 LSQ + AR-prior NLL) + AR-prior network weights (fp16+brotli)"
    ),
    parser_section_manifest={
        "header": "CCV1_magic_and_version",
        "synthesis_mlp_weights": "fp16_brotli_blob",
        "per_frame_latents": "fp4_lsq_arith_coded",
        "ar_prior_weights": "fp16_brotli_blob",
    },
    inflate_runtime_loc_budget=120,
    runtime_dep_closure=("torch>=2.5,<2.7", "brotli", "av", "constriction",),
    export_format="fp16_brotli",
    score_aware_loss="scorer_loss_terms_btchw",
    bolt_on_loc_budget=1100,
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
            "Cool-Chic substrate; sensitivity captured by AR-prior NLL rate term"
        ),
        "hook_bit_allocator_class": (
            "fp4 LSQ on latents + fp16 brotli on tiny MLP; no per-tensor bit allocator"
        ),
        "hook_probe_disambiguator": (
            "single mechanism (per-frame latent + tiny synthesis MLP + AR prior); no 2+ defensible interpretations"
        ),
    },
)


@register_substrate(COOL_CHIC_SUBSTRATE_CONTRACT)



def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.smoke:
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
