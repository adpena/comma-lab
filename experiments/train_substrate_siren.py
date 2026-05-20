# SPDX-License-Identifier: MIT
"""Train the siren substrate end-to-end on contest video.

Operator-callable training script per the Fields-medal grand council substrate
design wave (2026-05-12). PHASE-B2-BUILD wires ``_full_main`` so the trainer
can produce a first-anchor packet after operator authorization; local training
losses remain proxy signals until archive auth eval lands.

Dispatch contract default: ``naked_siren_replacement`` with
``--activation-family siren``. In this mode SIREN is the
purely-coordinate-based counterpart to NeRV/HNeRV/A1 and REPLACES that
substrate: ZERO bytes go to latents, and all variation across frames is encoded
in the MLP weights themselves via the selected byte-closed INR activation
family. Residual-on-HNeRV/A1 and hybrid-domain-prior contracts are named in
``tac.substrates.siren.dispatch_contract`` but intentionally fail closed here
until they have their own byte-closed builders.

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
  per HNeRV parity lesson L6.
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

Architectural risk (council Round 3 — NVIDIA-grade):
- SIREN's first-layer omega=30 init is high-frequency by design; gradient
  magnitudes can explode if the LR is set too aggressively. Council
  default lr=5e-4 with cosine annealing is conservative.
- Pure-coordinate substrates lack the per-frame inductive bias of HNeRV's
  per-frame latent; the network must memorize all 600 pair signatures via
  the t-axis encoding. This is the experiment.
- Score-aware gradient flow REQUIRES patched yuv6 + load_differentiable_scorers;
  PoseNet's @torch.no_grad() helper otherwise severs the chain.

Usage (smoke; CPU, tiny config, ~10 epochs, no scorer load)::

    .venv/bin/python experiments/train_substrate_siren.py \\
        --video-path upstream/videos/0.mkv \\
        --output-dir experiments/results/siren_smoke_<utc> \\
        --epochs 10 \\
        --device cpu --smoke

Usage (full; CUDA-required; threads from operator wrapper)::

    .venv/bin/python experiments/train_substrate_siren.py \\
        --video-path upstream/videos/0.mkv \\
        --output-dir experiments/results/siren_<utc> \\
        --epochs 2000 --batch-size 1 --lr 5e-4 --grad-clip 1.0 \\
        --device cuda
"""
# AUTOCAST_FP16_WAIVED:score-aware-scorer-path-pending-canonical-autocast-backport


# TORCH_COMPILE_WAIVED:defer-until-per-substrate-canary-validates-Inductor-graph-breaks-and-score-axis-custody
from __future__ import annotations

import argparse
import hashlib
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

# Canonical substrate-trainer helpers (CANON-DEDUP-1 commit ac1cfc41).
# Replaces ~70 LOC of inlined helpers with a single import per the
# 2026-05-13 substrate-trainer dedup migration wave.
from tac.substrates._shared.trainer_skeleton import (
    build_optimized_training_context as _build_optimized_training_context,
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
    torch_version_string as _canon_torch_version_string,
)
from tac.substrates._shared.trainer_skeleton import (
    utc_now_iso as _canon_utc_now_iso,
)
from tac.substrates._shared.trainer_skeleton import (
    vendor_shared_inflate_runtime as _canon_vendor_shared_inflate_runtime,
)
from tac.substrates.siren.activation_family import (
    ACTIVATION_FAMILY_IDS,
    DEFAULT_ACTIVATION_FAMILY,
    activation_family_manifest,
)
from tac.substrates.siren.dispatch_contract import (
    NAKED_SIREN_REPLACEMENT,
    require_train_substrate_siren_contract,
    siren_dispatch_contract_manifest,
)

# Tier-1 optimization helpers (TIER-1-OPT-BATCH 2026-05-14; CLAUDE.md
# Catalog #172/#179). The O1 GT-scorer cache flag is declared but reserved
# pending per-substrate score_aware_loss API extension.

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
# wrapper that subprocess-invokes this trainer. Schema mirrors the canonical
# sane_hnerv manifest per council R1-R7 (see CLAUDE.md catalog #151).
# ---------------------------------------------------------------------------
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--video-path": {
        "env": "SIREN_VIDEO_PATH",
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
        "env": "SIREN_OUTPUT_DIR",
        "rationale": "custody location for checkpoints + archive + provenance",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--epochs": {
        "env": "SIREN_EPOCHS",
        "rationale": (
            "SIREN substrate engineering pass; under-training silently regresses "
            "(council target: 2000)"
        ),
        "default": "2000",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--batch-size": {
        "env": "SIREN_BATCH_SIZE",
        "rationale": (
            "full-resolution coordinate-MLP renders are memory-heavy; batch=1 is "
            "the dispatch-safe default and must be explicit in operator recipes"
        ),
        "default": "1",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--upstream-dir": {
        "env": "SIREN_UPSTREAM_DIR",
        "rationale": (
            "upstream/ root for scorer weights + evaluate.py; required for full "
            "training (non-smoke) and auth eval"
        ),
        "default": str(DEFAULT_UPSTREAM_DIR.relative_to(REPO_ROOT)),
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--device": {
        "env": "SIREN_DEVICE",
        "rationale": (
            "compute device; cuda required for full training (MPS refused per "
            "CLAUDE.md MPS-NOISE rule); cpu permitted only with --smoke"
        ),
        "default": "cuda",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--dispatch-contract": {
        "env": "SIREN_DISPATCH_CONTRACT",
        "rationale": (
            "distinguishes naked SIREN replacement from residual-on-HNeRV/A1 "
            "and hybrid-domain-prior contracts; wrong contract means wrong archive"
        ),
        "default": NAKED_SIREN_REPLACEMENT,
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--activation-family": {
        "env": "SIREN_ACTIVATION_FAMILY",
        "rationale": (
            "byte-closed INR activation family under SRV1 metadata; default "
            "siren preserves naked_siren_replacement behavior"
        ),
        "default": DEFAULT_ACTIVATION_FAMILY,
        "satisfied_by_profile": (),
        "requires": ("--dispatch-contract",),
    },
}


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_substrate_siren",
        description="Train siren coordinate-MLP substrate end-to-end (PHASE-B2-BUILD wired).",
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
        default=1,
        help=(
            "Number of pair indices per batch. Full-resolution coordinate-MLP "
            "renders are memory-heavy, so the dispatch default is 1."
        ),
    )
    p.add_argument(
        "--lr",
        type=float,
        default=5e-4,
        help="AdamW learning rate (SIREN high-freq init: keep conservative).",
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
        help="Gradient clip norm (Council D pattern; SIREN first-layer omega=30 amplifies).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Manual seed for torch / numpy / random (deterministic).",
    )

    # ---- Substrate architecture knobs ----
    p.add_argument(
        "--hidden-dim",
        type=int,
        default=128,
        help="Hidden width of SIREN MLP layers (council default 128).",
    )
    p.add_argument(
        "--num-hidden-layers",
        type=int,
        default=6,
        help="Number of SIREN hidden layers (council default 6).",
    )
    p.add_argument(
        "--first-omega",
        type=float,
        default=30.0,
        help="SIREN first-layer omega (Sitzmann signature; NeRF default 30).",
    )
    p.add_argument(
        "--hidden-omega",
        type=float,
        default=1.0,
        help="SIREN downstream omega (Sitzmann standard 1.0).",
    )
    p.add_argument(
        "--activation-family",
        choices=list(ACTIVATION_FAMILY_IDS),
        default=DEFAULT_ACTIVATION_FAMILY,
        help=(
            "INR activation family serialized in SRV1 metadata. Default "
            "'siren' preserves the naked_siren_replacement contract; other "
            "modes are local activation-family probes, not separate dispatch contracts."
        ),
    )
    p.add_argument(
        "--wire-scale",
        type=float,
        default=1.0,
        help="Positive Gabor/window scale for --activation-family wire.",
    )
    p.add_argument(
        "--bacon-bandwidth-scale",
        type=float,
        default=1.0,
        help=(
            "Positive bandwidth multiplier for the BACON-style per-layer "
            "omega schedule."
        ),
    )
    p.add_argument(
        "--dispatch-contract",
        choices=[
            "naked_siren_replacement",
            "siren_residual_on_hnerv_a1",
            "hybrid_siren_domain_prior",
        ],
        default=NAKED_SIREN_REPLACEMENT,
        help=(
            "SIREN/INR packet contract. This trainer supports only "
            "naked_siren_replacement and fails closed for residual/hybrid "
            "contracts to avoid wrong-archive dispatch."
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
        default=True,
        help=(
            "Build the canonical GT-scorer-output cache once and reuse it "
            "inside the score-aware training loop (F3/O1; enabled by default)."
        ),
    )
    p.add_argument(
        "--disable-gt-scorer-cache",
        dest="enable_gt_scorer_cache",
        action="store_false",
        help="Disable the F3/O1 GTScorerCache path for memory/debug runs.",
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
        substrate_tag="siren",
        max_pairs=max_pairs,
        repo_root=REPO_ROOT,
    )


# ---------------------------------------------------------------------------
# Lagrangian helpers
# ---------------------------------------------------------------------------

def _archive_bytes_proxy_closed_form(model):
    # type: (...) -> 'torch.Tensor'
    """Closed-form upper-bound on archive bytes for the rate term.

    SIREN has NO latents — ALL parameters live in the MLP. The proxy is
    ``num_params * 2`` (fp16 weights). This is a constant during training,
    so the rate term is a constant offset; gradient flows entirely through
    seg + pose terms. A future Phase 2 lane will replace this with a
    differentiable rate proxy (post-quantization bit accounting).
    """
    import torch

    n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    bytes_proxy = float(n_total * 2)  # fp16
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
    """
    submission_dir.mkdir(parents=True, exist_ok=True)
    runtime_pkg = submission_dir / "src" / "tac" / "substrates" / "siren"
    runtime_pkg.mkdir(parents=True, exist_ok=True)
    # Vendor only the inflate-time SIREN package surface. Do not copy
    # score-aware training modules or scorer imports into the runtime tree.
    for pkg_init in (
        submission_dir / "src" / "tac" / "__init__.py",
        submission_dir / "src" / "tac" / "substrates" / "__init__.py",
        runtime_pkg / "__init__.py",
    ):
        pkg_init.write_text("", encoding="utf-8")
    substrate_src = REPO_ROOT / "src" / "tac" / "substrates" / "siren"
    for name in ("activation_family.py", "architecture.py", "archive.py", "inflate.py"):
        shutil.copy2(substrate_src / name, runtime_pkg / name)
    _canon_vendor_shared_inflate_runtime(submission_dir, repo_root=REPO_ROOT)

    inflate_sh = (
        "#!/usr/bin/env bash\n"
        "# siren contest-compliant inflate (PHASE-B2-BUILD wired 2026-05-12)\n"
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
        "\"\"\"siren contest-compliant inflate runtime.\n"
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
        "from tac.substrates.siren.inflate import inflate_one_video, raw_output_path, select_inflate_device\n"
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


def _build_archive_zip(
    archive_zip_path: Path,
    *,
    bin_bytes: bytes,
    submission_dir: Path,
) -> None:
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
    return _canon_device_or_die(name, smoke=smoke, substrate_tag="siren")


# ---------------------------------------------------------------------------
# Smoke main (CPU; no scorer load)
# ---------------------------------------------------------------------------

def _smoke_main(args: argparse.Namespace) -> int:
    """Tiny CPU smoke that proves the scaffold is wired (no scorer load)."""
    import torch

    from tac.substrates.siren.architecture import SirenConfig, SirenSubstrate

    _pin_seeds(args.seed)

    cfg = SirenConfig(
        hidden_dim=32,
        num_hidden_layers=3,
        first_omega=args.first_omega,
        hidden_omega=args.hidden_omega,
        num_pairs=4,
        output_height=24,
        output_width=32,
        activation_family=args.activation_family,
        wire_scale=args.wire_scale,
        bacon_bandwidth_scale=args.bacon_bandwidth_scale,
    )
    device = _device_or_die(args.device, smoke=True)
    model = SirenSubstrate(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[smoke] siren params: {model.num_parameters():,}")
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
    """Full training entry point — requires CUDA + score-aware scorers.

    This path is OPERATOR-GATED. The wrapper (Vast.ai / Lightning / Modal)
    threads all TIER_1 flags + runs the auth-eval afterward per CLAUDE.md
    "Auth eval EVERYWHERE" + "Submission auth eval — BOTH CPU AND CUDA".
    """
    import torch

    from tac.differentiable_eval_roundtrip import (
        patch_upstream_yuv6_globally,
        unpatch_upstream_yuv6,
    )
    from tac.scorer import load_differentiable_scorers
    from tac.substrates.siren.architecture import SirenConfig, SirenSubstrate
    from tac.substrates.siren.archive import pack_archive
    from tac.substrates.siren.score_aware_loss import (
        ScoreAwareLossWeights,
        SirenScoreAwareLoss,
    )
    from tac.training import EMA

    dispatch_contract = require_train_substrate_siren_contract(args.dispatch_contract)
    dispatch_contracts = siren_dispatch_contract_manifest()

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
        cfg = SirenConfig(
            hidden_dim=args.hidden_dim,
            num_hidden_layers=args.num_hidden_layers,
            first_omega=args.first_omega,
            hidden_omega=args.hidden_omega,
            num_pairs=n_pairs,
            output_height=EVAL_HW[0],
            output_width=EVAL_HW[1],
            activation_family=args.activation_family,
            wire_scale=args.wire_scale,
            bacon_bandwidth_scale=args.bacon_bandwidth_scale,
        )
        model = SirenSubstrate(cfg).to(device)
        print(f"[full] siren params: {model.num_parameters():,}")
        print(f"[full] activation_family: {cfg.activation_family}")
        _stage("model_built")
        _stage(f"activation_family_{cfg.activation_family}")

        opt_ctx = _build_optimized_training_context(
            args,
            scorers=(posenet, segnet),
            gt_pairs=pair_tensor,
            substrate_model=model,
            device=device,
        )
        model = opt_ctx.substrate_model
        gt_cache = opt_ctx.gt_cache
        if gt_cache is not None:
            print(gt_cache.summary_line())
            _stage("gt_scorer_cache_built")
        else:
            _stage("gt_scorer_cache_disabled")

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
        )
        loss_fn = SirenScoreAwareLoss(
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
                # Frames in [0,1]; score-aware loss + eval-roundtrip expect [0, 255]
                rgb_0_255 = rgb_0 * 255.0
                rgb_1_255 = rgb_1 * 255.0
                gt = pair_tensor[idx]  # (B, 2, 3, H, W) in [0, 255]
                gt_0 = gt[:, 0]
                gt_1 = gt[:, 1]
                gt_pose_batch = gt_seg_batch = None
                gt_seg_already_probs = None
                if gt_cache is not None:
                    gt_pose_batch, gt_seg_batch = gt_cache.lookup(idx, device=device)
                    gt_seg_already_probs = gt_cache.seg_already_probs
                loss, parts = loss_fn(
                    rgb_0_255, rgb_1_255, gt_0, gt_1, archive_bytes_proxy,
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

            # 10. Validation + best-ckpt selection (snapshot+restore pattern)
            if (epoch + 1) % args.val_every_epochs == 0 or epoch == args.epochs - 1:
                orig_state = {
                    k: v.detach().clone() for k, v in model.state_dict().items()
                }
                ema.apply(model)
                model.eval()
                with torch.no_grad():
                    rgb_0_v, rgb_1_v = model(val_indices)
                    val_gt_pose_batch = val_gt_seg_batch = None
                    val_gt_seg_already_probs = None
                    if gt_cache is not None:
                        val_gt_pose_batch, val_gt_seg_batch = gt_cache.lookup(
                            val_indices, device=device
                        )
                        val_gt_seg_already_probs = gt_cache.seg_already_probs
                    val_loss, _val_parts = loss_fn(
                        rgb_0_v * 255.0,
                        rgb_1_v * 255.0,
                        pair_tensor[val_indices, 0],
                        pair_tensor[val_indices, 1],
                        archive_bytes_proxy,
                        apply_eval_roundtrip=True,
                        noise_std=args.noise_std,
                        gt_pose_batch=val_gt_pose_batch,
                        gt_seg_batch=val_gt_seg_batch,
                        gt_seg_already_probs=val_gt_seg_already_probs,
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
                    # Save EMA shadow (NOT live weights) — CLAUDE.md EMA rule
                    ema_state = ema.state_dict()
                    torch.save(
                        {
                            "state_dict": {k: v.detach().cpu() for k, v in ema_state.items()},
                            "config": asdict(cfg),
                            "ema_decay": args.ema_decay,
                            "best_val_lagrangian": val_lag,
                            "best_val_lagrangian_evidence_grade": "training_proxy_non_authoritative",
                            "best_val_lagrangian_score_claim": False,
                            "best_val_lagrangian_promotion_eligible": False,
                            "best_epoch": int(epoch),
                            "saved_at_utc": _utc_now_iso(),
                            "training_axis_note": (
                                "[training-proxy] score-aware Lagrangian only; "
                                "not [contest-CUDA]; archive auth eval still required"
                            ),
                            "score_claim": False,
                            "promotion_eligible": False,
                            "ready_for_exact_eval_dispatch": False,
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
            # Edge case: no val pass found a finite improvement; save the
            # EMA shadow at end-of-training so downstream stages can proceed.
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
                    "best_val_lagrangian_evidence_grade": "training_proxy_non_authoritative",
                    "best_val_lagrangian_score_claim": False,
                    "best_val_lagrangian_promotion_eligible": False,
                    "best_epoch": int(args.epochs - 1),
                    "saved_at_utc": _utc_now_iso(),
                    "fallback_end_of_training_save": True,
                    "score_claim": False,
                    "promotion_eligible": False,
                    "ready_for_exact_eval_dispatch": False,
                },
                ckpt_best_path,
            )

        # 11. Build the SRV1 archive bytes from the EMA shadow
        archive_sha = ""
        archive_bytes = 0
        payload_bin_sha = ""
        payload_bin_bytes = 0
        archive_zip_path = args.output_dir / "archive.zip"
        if not args.skip_archive_build:
            print(f"[full] building archive from {ckpt_best_path} ...")
            ema_state = torch.load(ckpt_best_path, map_location="cpu", weights_only=False)
            sd = ema_state["state_dict"]
            # Drop deterministic buffer per architecture.runtime_state_dict_for_archive
            decoder_sd = {k: v for k, v in sd.items() if k != "_spatial_coords"}
            meta = {
                "first_omega": cfg.first_omega,
                "hidden_omega": cfg.hidden_omega,
                "coord_dim": cfg.coord_dim,
                "output_dim": cfg.output_dim,
                "activation_family": cfg.activation_family,
                "wire_scale": cfg.wire_scale,
                "bacon_bandwidth_scale": cfg.bacon_bandwidth_scale,
                "dispatch_contract": dispatch_contract.contract_id,
                "archive_role": dispatch_contract.archive_role,
                "hnerv_a1_relationship": dispatch_contract.hnerv_a1_relationship,
            }
            bin_bytes = pack_archive(
                decoder_sd,
                meta,
                num_pairs=cfg.num_pairs,
                hidden_dim=cfg.hidden_dim,
                num_hidden_layers=cfg.num_hidden_layers,
                output_height=cfg.output_height,
                output_width=cfg.output_width,
            )
            (args.output_dir / "0.bin").write_bytes(bin_bytes)
            payload_bin_sha = _sha256_bytes(bin_bytes)
            payload_bin_bytes = len(bin_bytes)
            print(
                f"[full] wrote 0.bin "
                f"({payload_bin_bytes} bytes, sha256={payload_bin_sha})"
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
            archive_bytes = archive_zip_path.stat().st_size
            archive_sha = hashlib.sha256(archive_zip_path.read_bytes()).hexdigest()
            print(
                f"[full] wrote {archive_zip_path} "
                f"({archive_bytes} bytes, sha256={archive_sha})"
            )
            _stage(f"archive_built_bytes_{archive_bytes}")

        # 12. CUDA auth eval — canonical helper (Catalog #226 self-protect)
        auth_eval_result_path: Path | None = None
        auth_eval_score: float | None = None
        auth_eval_score_axis: str | None = None
        auth_eval_lane_tag: str | None = None
        auth_eval_score_claim_valid = False
        auth_eval_exact_cuda_complete = False
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
                substrate_tag="siren",
                device=device,
            )
            if auth_result is not None:
                contest_cuda_score = auth_result["auth_eval_cuda_score"]
                auth_eval_score = contest_cuda_score
                auth_eval_score_axis = auth_result["auth_eval_score_axis"]
                auth_eval_lane_tag = auth_result["auth_eval_lane_tag"]
                auth_eval_score_claim_valid = bool(
                    auth_result["auth_eval_score_claim_valid"]
                )
                auth_eval_exact_cuda_complete = bool(
                    auth_result["auth_eval_exact_cuda_complete"]
                )
                print(
                    f"[full] {auth_eval_lane_tag or '[contest-CUDA]'} score = "
                    f"{contest_cuda_score} (axis={auth_eval_score_axis}, "
                    f"archive_sha256={archive_sha})"
                )
            _stage("auth_eval_cuda_done")

        # 13. Continual-learning posterior update (Catalog #128 atomic)
        if contest_cuda_score is not None and archive_sha:
            try:
                from tac.continual_learning import ContestResult, posterior_update_locked

                # Per CLAUDE.md SIREN audit 2026-05-13 CRITICAL #1 + Catalog
                # #190: detect substrate from remote driver provenance.json,
                # then SIREN_GPU / MODAL_GPU env vars, then nvidia-smi. Never
                # silently hardcode T4 when the recipe targets A100.
                _detected_substrate = _canon_detect_hardware_substrate(
                    axis="cuda",
                    substrate_tag="siren",
                    provenance_path=args.output_dir / "provenance.json",
                    env_var_candidates=("SIREN_GPU", "MODAL_GPU"),
                )
                result = ContestResult(
                    axis="cuda",
                    hardware_substrate=_detected_substrate,
                    architecture_class="lane_substrate_siren_20260512",
                    score_value=contest_cuda_score,
                    evidence_tag="[contest-CUDA]",
                    archive_sha256=archive_sha,
                    archive_bytes=archive_bytes,
                    notes=(
                        f"siren first-anchor dispatch; epochs={args.epochs}; "
                        f"hardware_substrate_detected={_detected_substrate}"
                    ),
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
                os.environ.get("SIREN_ACTUAL_COST_USD"),
                field_name="SIREN_ACTUAL_COST_USD",
            )
        except ValueError as exc:
            actual_cost_usd = None
            cost_band_anchor_skip_reason = f"invalid_SIREN_ACTUAL_COST_USD:{exc}"
        if COST_BAND_TOOL.is_file() and train_elapsed_sec > 0 and actual_cost_usd is not None:
            try:
                proc = subprocess.run(
                    [
                        sys.executable, str(COST_BAND_TOOL),
                        "--dispatch-label", f"siren_{_utc_now_iso()}",
                        "--trainer", "experiments/train_substrate_siren.py",
                        "--platform", os.environ.get("SIREN_PLATFORM", "modal"),
                        "--gpu", os.environ.get("SIREN_GPU", "A100"),
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
                cost_band_anchor_skip_reason = "missing_SIREN_ACTUAL_COST_USD"
            elif not COST_BAND_TOOL.is_file():
                cost_band_anchor_skip_reason = "cost_band_tool_missing"
            else:
                cost_band_anchor_skip_reason = "nonpositive_train_elapsed_sec"

        # 15. Provenance manifest
        provenance = {
            "schema": "siren_provenance_v1",
            "generated_at": _utc_now_iso(),
            "from_state_hash": "regen_per_session_below",
            "git_head": _git_head_sha(),
            "trainer": "experiments/train_substrate_siren.py",
            "lane_id": "lane_substrate_siren_20260512",
            "dispatch_contract": dispatch_contract.contract_id,
            "dispatch_contract_summary": dispatch_contract.summary,
            "archive_role": dispatch_contract.archive_role,
            "hnerv_a1_relationship": dispatch_contract.hnerv_a1_relationship,
            "dispatch_contracts_distinguished": dispatch_contracts,
            "activation_family": cfg.activation_family,
            "activation_family_manifest": activation_family_manifest(),
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
            "best_val_lagrangian_evidence_grade": "training_proxy_non_authoritative",
            "best_val_lagrangian_score_claim": False,
            "best_val_lagrangian_promotion_eligible": False,
            "proxy_score_authority": False,
            "proxy_authority_blockers": [
                "training_lagrangian_is_not_archive_auth_eval",
                "archive_bytes_proxy_is_not_measured_archive_size",
                "exact_cuda_auth_eval_required_before_score_claim",
            ],
            "best_epoch": int(best_epoch),
            "train_elapsed_sec": float(train_elapsed_sec),
            "archive_sha256": archive_sha,
            "archive_bytes": archive_bytes,
            "payload_bin_sha256": payload_bin_sha,
            "payload_bin_bytes": payload_bin_bytes,
            "auth_eval_score": auth_eval_score,
            "auth_eval_score_axis": auth_eval_score_axis,
            "auth_eval_lane_tag": auth_eval_lane_tag,
            "auth_eval_score_claim_valid": auth_eval_score_claim_valid,
            "auth_eval_exact_cuda_complete": auth_eval_exact_cuda_complete,
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
    return _canon_torch_version_string()


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

SIREN_SUBSTRATE_CONTRACT = SubstrateContract(
    # 2.1 Identity & lifecycle
    id="siren",
    lane_id="lane_substrate_siren_20260512",
    target_modes=("contest_one_video_replay", "research_substrate",),
    deployment_target="t4_contest_runtime",
    council_verdict_provenance=(
        ".omx/research/grand_council_fields_medal_substrate_design_20260512.md"
    ),
    # 2.2 Architecture & runtime (8 per Catalog #124)
    archive_grammar=(
        "SIRV1 monolithic single-file 0.bin: header + SIREN sinusoidal-MLP weights (fp16 + brotli; per-coordinate-MLP) + omega_0 frequency parameter"
    ),
    parser_section_manifest={
        "header": "SIRV1_magic_and_version",
        "siren_mlp_weights": "fp16_brotli_blob",
        "omega_0": "fp32_scalar",
    },
    inflate_runtime_loc_budget=100,
    runtime_dep_closure=("torch>=2.5,<2.7", "brotli", "av",),
    export_format="fp16_brotli",
    score_aware_loss="scorer_loss_terms_btchw",
    bolt_on_loc_budget=1330,
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
            "SIREN coordinate-MLP; rate-distortion captures sensitivity"
        ),
        "hook_bit_allocator_class": (
            "fp16 brotli on MLP weights; no per-tensor bit allocator"
        ),
        "hook_probe_disambiguator": (
            "single mechanism (sinusoidal-activation coordinate MLP); no 2+ defensible interpretations"
        ),
    },
)


@register_substrate(SIREN_SUBSTRATE_CONTRACT)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if not args.smoke:
        require_train_substrate_siren_contract(args.dispatch_contract)
    if args.smoke:
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
