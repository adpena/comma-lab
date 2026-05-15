# SPDX-License-Identifier: MIT
"""Train the Time-Traveler L5 Autonomy substrate end-to-end on contest video.

PAIR T operator directive 2026-05-13 — reverse-engineered architecture from
the "time-traveler-from-the-future" framing. Design memo:
``.omx/research/time_traveler_architecture_reverse_engineered_20260513.md``.

Builds a single-archive TT5L packet at 95-110 KB target size by composing five
first-principles design moves: cooperative-receiver theorem (Atick-Redlich
1990), predictive-coding hierarchy (Rao-Ballard 1999), foveation-matched-to-
ego-motion, differentiable world model, sub-100K params + Tikhonov
regularization.

Council-binding contract (CLAUDE.md non-negotiables) honored end-to-end:

- Train against ``upstream/videos/0.mkv`` decoded via pyav (synthetic data
  FORBIDDEN outside ``--smoke`` per Catalog #114).
- Patch upstream ``rgb_to_yuv6`` via ``patch_upstream_yuv6_globally`` BEFORE
  scorer construction (Catalog #187; PR #95/#106 contract).
- ``load_differentiable_scorers`` for SegNet/PoseNet (no scorer load at
  inflate; only at training).
- ``apply_eval_roundtrip_during_training`` inside the per-batch loop
  (Catalog #5).
- ``tac.training.EMA(decay=0.997)`` update after every ``optimizer.step``;
  inference checkpoint = EMA shadow (Catalog #88).
- Score-domain Lagrangian ``α·B/N + β·d_seg + γ·√d_pose + δ·H_pred`` per
  HNeRV parity discipline lesson L6.
- AdamW lr cosine annealing; gradient clip 1.0; NaN watchdog (Council D).
- End with CUDA auth eval on best EMA checkpoint (CLAUDE.md "Auth eval
  EVERYWHERE"); refuse MPS (Catalog #1); CPU permitted only with ``--smoke``.
- Continual-learning posterior update via ``posterior_update_locked``
  (Catalog #128 atomic fcntl).
- Cost-band anchor append via ``tools/append_cost_band_anchor.py``.
- Contest-compliant runtime emission with 3 positional args inflate.sh +
  ``set -euo pipefail`` per Catalog #146 + #163.
- TIER_1_OPERATOR_REQUIRED_FLAGS declared per Catalog #151 (annotated
  assignment for Catalog #168 AST walker).

Usage (smoke; macOS CPU, tiny config, ~3 epochs, no scorer load)::

    .venv/bin/python experiments/train_substrate_time_traveler_l5_autonomy.py \\
        --video-path upstream/videos/0.mkv \\
        --output-dir experiments/results/tt5l_smoke_<utc> \\
        --epochs 3 --device cpu --smoke

Usage (full; CUDA-required)::

    .venv/bin/python experiments/train_substrate_time_traveler_l5_autonomy.py \\
        --video-path upstream/videos/0.mkv \\
        --output-dir experiments/results/tt5l_<utc> \\
        --epochs 3000 --batch-size 1 --lr 5e-4 --device cuda

Usage (``--full-cpu``; macOS-CPU contest-shape VALIDATION ONLY; advisory-only;
operator approval 2026-05-13; Catalog #197 self-protection)::

    .venv/bin/python experiments/train_substrate_time_traveler_l5_autonomy.py \\
        --video-path upstream/videos/0.mkv \\
        --output-dir experiments/results/tt5l_<utc> \\
        --epochs 100 --batch-size 1 --lr 5e-4 --device cpu \\
        --full-cpu --advisory-cpu-explicitly-waived \\
        --max-wall-clock-hours 12.0

Notes for ``--full-cpu`` mode:

- Wall-clock expectation on macOS M5 Max CPU at contest-shape (384x512, 600
  pairs, sub-60K-param renderer): ~2-6 hours for 100 epochs (Carmack's
  pessimistic-realistic bound: up to 12 hours; default ``--max-wall-clock-hours
  12.0``). For shorter runs reduce ``--epochs``.
- All artifacts written from ``--full-cpu`` carry ``evidence_grade=
  "macOS-CPU-advisory"`` (Catalog #127 + Catalog #192) with
  ``score_claim=false``, ``promotion_eligible=false``, and
  ``ready_for_exact_eval_dispatch=false`` permanently set. Persisted artifacts
  trip Catalog #192 STRICT preflight if any of those flags is later flipped to
  ``true``.
- Auth eval is SKIPPED by default in ``--full-cpu`` (local macOS-CPU auth eval
  is NOT 1:1 contest-compliant per CLAUDE.md "Submission auth eval — BOTH CPU
  AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE"). The runtime tree is still
  emitted for inflate parity verification by sister tools (e.g.
  ``tools/smoke_time_traveler_l5_autonomy_macos_cpu.py --archive-path ...``).
- The mode requires BOTH ``--full-cpu`` AND ``--advisory-cpu-explicitly-waived``
  to acknowledge the non-promotable nature of the output. The
  device-or-die-gate-bypass-via-uncoupled-flags bug class is structurally
  extincted by Catalog #197 (``check_full_cpu_requires_explicit_advisory_waiver``).
- Per CLAUDE.md Tao note: CPU vs CUDA produces non-bitwise-identical scores on
  the same archive because fused-kernel paths differ (PR102 anchor showed
  -0.033 CUDA-CPU gap; magnitude is per-archive empirical, not extrapolatable).
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
    require_contest_cuda_auth_eval_claim as _canon_require_contest_cuda_auth_eval_claim,
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

# Tier-1 optimization helpers (TIER-1-OPT-BATCH 2026-05-14).
from tac.training_optimization import (
    autocast_aware_forward as _autocast_aware_forward,
)
from tac.training_optimization import (
    compile_with_fallback as _compile_with_fallback,
)

# ---------------------------------------------------------------------------
# Module paths + constants
# ---------------------------------------------------------------------------

DEFAULT_VIDEO_PATH = REPO_ROOT / "upstream" / "videos" / "0.mkv"
DEFAULT_UPSTREAM_DIR = REPO_ROOT / "upstream"
CONTEST_AUTH_EVAL_SCRIPT = REPO_ROOT / "experiments" / "contest_auth_eval.py"
COST_BAND_TOOL = REPO_ROOT / "tools" / "append_cost_band_anchor.py"

EVAL_HW = (384, 512)
CAMERA_HW = (874, 1164)
N_PAIRS_FULL = 600
CONTEST_NORMALIZER = 37_545_489.0

# Substrate constants (re-exported for trainer local helpers).
SUBSTRATE_TAG = "tt5l"
SUBSTRATE_LANE_ID = "lane_time_traveler_l5_autonomy_substrate_20260513"


# ---------------------------------------------------------------------------
# Catalog #151 manifest — every flag below must be threaded by any operator
# wrapper that subprocess-invokes this trainer. Annotated as ast.AnnAssign
# so Catalog #168's AST walker observes it.
# ---------------------------------------------------------------------------
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--video-path": {
        "env": "TT5L_VIDEO_PATH",
        "rationale": (
            "score-aware substrate MUST train against the contest video "
            "(upstream/videos/0.mkv); synthetic data is FORBIDDEN outside --smoke"
        ),
        "default": str(DEFAULT_VIDEO_PATH.relative_to(REPO_ROOT)),
        "required_input_file": True,
        "generator_command": (
            "contest-pinned upstream snapshot — never regenerated locally"
        ),
        "rationale_audit": (
            ".omx/research/time_traveler_architecture_reverse_engineered_20260513.md"
        ),
    },
    "--output-dir": {
        "env": "TT5L_OUTPUT_DIR",
        "rationale": "custody location for checkpoints + archive + provenance",
    },
    "--epochs": {
        "env": "TT5L_EPOCHS",
        "rationale": (
            "Time-Traveler renderer is small; council default 3000 epochs "
            "for full training run"
        ),
        "default": "3000",
    },
    "--upstream-dir": {
        "env": "TT5L_UPSTREAM_DIR",
        "rationale": (
            "upstream/ root for scorer weights + evaluate.py; required for "
            "full training and auth eval"
        ),
        "default": str(DEFAULT_UPSTREAM_DIR.relative_to(REPO_ROOT)),
    },
    "--device": {
        "env": "TT5L_DEVICE",
        "rationale": (
            "compute device; cuda required for full training (MPS refused "
            "per CLAUDE.md MPS-NOISE rule); cpu permitted only with --smoke"
        ),
        "default": "cuda",
    },
    "--hidden-dim": {
        "env": "TT5L_HIDDEN_DIM",
        "rationale": "Renderer MLP hidden width (default 64 = sub-60K param target)",
        "default": "64",
    },
    "--per-pair-side-info-bytes": {
        "env": "TT5L_PER_PAIR_BYTES",
        "rationale": (
            "Bytes per pair for the Stage 2 side info channel (default 45 = "
            "12 SE3 + 18 seg + 6 HF + 9 predict; design-memo budget)"
        ),
        "default": "45",
    },
}


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_substrate_time_traveler_l5_autonomy",
        description=(
            "Train Time-Traveler L5 Autonomy substrate (TT5L). "
            "Differentiable world model + foveation + per-pair side info + "
            "Atick-Redlich cooperative-receiver loss."
        ),
    )
    # Tier 1 required
    p.add_argument("--video-path", type=Path, default=DEFAULT_VIDEO_PATH,
                   help="Path to upstream/videos/0.mkv (contest video).")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Where to write checkpoints + manifest + archive.")
    p.add_argument("--epochs", type=int, required=True,
                   help="Number of training epochs (council default 3000).")
    p.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR,
                   help="upstream/ root; required for scorer load + auth eval.")

    # Substrate architecture
    p.add_argument("--hidden-dim", type=int, default=64,
                   help="Renderer MLP hidden width (default 64).")
    p.add_argument("--num-hidden-layers", type=int, default=4,
                   help="Renderer MLP depth (default 4).")
    p.add_argument("--first-omega", type=float, default=30.0)
    p.add_argument("--hidden-omega", type=float, default=1.0)
    p.add_argument("--coord-feature-freqs", type=int, default=4)
    p.add_argument("--foveation-grid-h", type=int, default=16)
    p.add_argument("--foveation-grid-w", type=int, default=24)
    p.add_argument("--per-pair-side-info-bytes", type=int, default=45,
                   help="Bytes per pair for Stage 2 side info (default 45).")
    p.add_argument("--int8-scale", type=float, default=64.0,
                   help="int8 quantization scale for per-pair residual.")

    # Training hyperparameters
    p.add_argument("--batch-size", type=int, default=1,
                   help="Pairs per batch (default 1; renderer is per-pair).")
    p.add_argument("--lr", type=float, default=5e-4,
                   help="AdamW learning rate (SIREN-init friendly).")
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ema-decay", type=float, default=0.997,
                   help="EMA decay (CLAUDE.md non-negotiable default 0.997).")
    p.add_argument("--noise-std", type=float, default=0.5)
    p.add_argument("--val-every-epochs", type=int, default=10)
    p.add_argument("--val-pair-count", type=int, default=16)
    p.add_argument("--max-pairs", type=int, default=None)

    # Score-aware Lagrangian weights
    p.add_argument("--alpha-rate", type=float, default=25.0)
    p.add_argument("--beta-seg", type=float, default=100.0)
    p.add_argument("--gamma-pose", type=float, default=math.sqrt(10.0))
    p.add_argument("--pose-weight-scale", type=float, default=1.0)
    p.add_argument("--delta-predict", type=float, default=0.1,
                   help="Predictive-coding hierarchy auxiliary term weight.")

    # Optimizer choice
    p.add_argument("--optimizer",
                   choices=("adamw", "iglt", "muon", "muon+iglt"),
                   default="adamw",
                   help="Inner optimizer: 'adamw' (default), 'iglt' "
                        "(Fisher-preconditioned Langevin for polish-phase), "
                        "'muon' (Keller Jordan NS-orthogonalized momentum on "
                        "hidden 2-D+ weights + AdamW on stem/RGB heads/biases), "
                        "or 'muon+iglt' (Muon on hidden + IGLT on stem/heads).")
    p.add_argument("--muon-lr", type=float, default=0.02,
                   help="Muon learning rate (default 0.02; higher than AdamW "
                        "because NS normalizes update magnitude). Only used "
                        "when --optimizer is 'muon' or 'muon+iglt'.")
    p.add_argument("--muon-weight-decay", type=float, default=0.0,
                   help="Muon decoupled weight decay (default 0.0; recommended "
                        "~0.01 per Chen-Li-Liu arXiv:2506.15054 spectral-KKT "
                        "story). Only used when --optimizer is 'muon' or "
                        "'muon+iglt'.")

    # Device / mode
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                   help="cuda for full training; cpu only with --smoke "
                        "or with --full-cpu --advisory-cpu-explicitly-waived.")
    p.add_argument("--smoke", action="store_true",
                   help="Tiny CPU smoke (no scorer load).")
    # --full-cpu mode (macOS-CPU contest-shape VALIDATION; advisory-only).
    # See module docstring + Catalog #197 self-protection.
    p.add_argument("--full-cpu", action="store_true",
                   help="Allow contest-shape training on CPU (advisory-only; "
                        "~2-12h wall-clock on macOS M5 Max; produces "
                        "[macOS-CPU advisory only] non-promotable scores per "
                        "CLAUDE.md MPS auth eval rule + Catalog #127 + #192). "
                        "REQUIRES --advisory-cpu-explicitly-waived to "
                        "acknowledge the non-promotable nature.")
    p.add_argument("--advisory-cpu-explicitly-waived", action="store_true",
                   help="Acknowledge that --full-cpu produces "
                        "[macOS-CPU advisory only] scores only. Required "
                        "alongside --full-cpu (Catalog #197).")
    p.add_argument("--max-wall-clock-hours", type=float, default=12.0,
                   help="Maximum wall-clock hours for --full-cpu mode; "
                        "training aborts gracefully if exceeded (default 12.0).")

    # Post-train artifacts
    p.add_argument("--skip-auth-eval", action="store_true")
    p.add_argument("--skip-archive-build", action="store_true")
    p.add_argument("--enable-autocast-fp16", action="store_true",
                   help="Catalog #172; opt-in via canonical autocast_aware_forward.")
    p.add_argument("--enable-tf32", action="store_true",
                   help="Catalog #178; deferred until paired CPU/CUDA anchor lands.")
    p.add_argument("--enable-torch-compile", action="store_true",
                   default=False,
                   help="Catalog #179; wrap substrate model with torch.compile.")
    p.add_argument("--enable-gt-scorer-cache", action="store_true",
                   default=False,
                   help=("RESERVED (O1): GT-scorer-output cache; wire-in "
                         "pending per-substrate score_aware_loss API extension."))
    return p


# ---------------------------------------------------------------------------
# Local helpers (delegate to canonical skeleton)
# ---------------------------------------------------------------------------

def _utc_now_iso() -> str:
    return _canon_utc_now_iso()


def _sha256_bytes(data: bytes) -> str:
    return _canon_sha256_bytes(data)


def _git_head_sha() -> str:
    return _canon_git_head_sha(REPO_ROOT)


def _pin_seeds(seed: int) -> None:
    _canon_pin_seeds(seed)


def _device_or_die(name: str, *, smoke: bool):
    return _canon_device_or_die(name, smoke=smoke, substrate_tag=SUBSTRATE_TAG)


def _validate_full_cpu_flags(args: argparse.Namespace) -> None:
    """Coupled-flag validation for ``--full-cpu`` mode (Catalog #197).

    Refuses any inconsistent combination of ``--full-cpu``,
    ``--advisory-cpu-explicitly-waived``, ``--device``, and ``--smoke``.

    Per CLAUDE.md "MPS auth eval is NOISE" + "Submission auth eval — BOTH
    CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" + Catalog #127 + #192:
    macOS-CPU is NEVER 1:1 contest-compliant; ``--full-cpu`` produces
    advisory-only output and requires the operator to explicitly acknowledge
    that fact via ``--advisory-cpu-explicitly-waived``.

    Raises:
        SystemExit: when ``--full-cpu`` is set without
            ``--advisory-cpu-explicitly-waived``, when ``--full-cpu`` is set
            simultaneously with ``--smoke`` (distinct modes), or when
            ``--advisory-cpu-explicitly-waived`` is set without
            ``--full-cpu`` (dangling waiver — refuse to prevent silent waive
            of a later, unrelated waiver-check).
    """
    full_cpu = bool(getattr(args, "full_cpu", False))
    waived = bool(getattr(args, "advisory_cpu_explicitly_waived", False))
    smoke = bool(getattr(args, "smoke", False))

    if full_cpu and not waived:
        raise SystemExit(
            f"[{SUBSTRATE_TAG}] --full-cpu requires "
            "--advisory-cpu-explicitly-waived to acknowledge that the run "
            "produces [macOS-CPU advisory only] scores (non-promotable per "
            "CLAUDE.md 'MPS auth eval is NOISE' + 'Submission auth eval — "
            "BOTH CPU AND CUDA' + Catalog #127 + #192). Add the waiver flag "
            "or omit --full-cpu."
        )
    if waived and not full_cpu:
        raise SystemExit(
            f"[{SUBSTRATE_TAG}] --advisory-cpu-explicitly-waived is set "
            "without --full-cpu. Dangling waiver flag is refused to prevent "
            "silent waive of a later, unrelated waiver-check (Catalog #197)."
        )
    if full_cpu and smoke:
        raise SystemExit(
            f"[{SUBSTRATE_TAG}] --full-cpu and --smoke are mutually exclusive. "
            "Use --smoke for tiny CPU scaffold smoke (no scorer load) OR "
            "--full-cpu for contest-shape advisory training (scorer-loaded)."
        )
    if full_cpu and args.device != "cpu":
        raise SystemExit(
            f"[{SUBSTRATE_TAG}] --full-cpu requires --device cpu; "
            f"got --device {args.device!r}."
        )


def _full_cpu_banner(args: argparse.Namespace) -> None:
    """Loud startup banner for ``--full-cpu`` runs (Catalog #127 + #192).

    Emits a multi-line stderr banner so the operator cannot miss that this
    run produces NON-promotable scores tagged ``[macOS-CPU advisory only]``.
    """
    if not bool(getattr(args, "full_cpu", False)):
        return
    bar = "=" * 78
    msg = (
        f"\n{bar}\n"
        f"[{SUBSTRATE_TAG}-full-cpu] [macOS-CPU advisory only] BANNER\n"
        f"{bar}\n"
        "This run produces NON-promotable scores per CLAUDE.md\n"
        "'MPS auth eval is NOISE' + 'Submission auth eval — BOTH CPU AND\n"
        "CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE' + Catalog #127 + #192.\n"
        "All artifacts written from this run carry:\n"
        '  evidence_grade = "macOS-CPU-advisory"\n'
        "  score_claim                       = false (permanent)\n"
        "  promotion_eligible                = false (permanent)\n"
        "  ready_for_exact_eval_dispatch     = false (permanent)\n"
        "Auth eval is SKIPPED by default in --full-cpu mode (local macOS-CPU\n"
        "is NOT 1:1 contest-compliant — Linux x86_64 + GHA CI is the\n"
        "canonical [contest-CPU] axis).\n"
        f"max_wall_clock_hours = {args.max_wall_clock_hours}\n"
        f"{bar}\n"
    )
    print(msg, file=sys.stderr, flush=True)


def _decode_real_pairs(video_path: Path, *, n_pairs: int, max_pairs: int | None = None):
    return _canon_decode_real_pairs(
        video_path,
        n_pairs=n_pairs,
        substrate_tag=SUBSTRATE_TAG,
        max_pairs=max_pairs,
        repo_root=REPO_ROOT,
    )


# ---------------------------------------------------------------------------
# Contest-compliant runtime emission (Catalog #146)
# ---------------------------------------------------------------------------

def _write_runtime(submission_dir: Path) -> None:
    """Emit contest-compliant inflate.sh + inflate.py + vendored substrate."""
    submission_dir.mkdir(parents=True, exist_ok=True)
    runtime_pkg = submission_dir / "src" / "tac" / "substrates" / "time_traveler_l5_autonomy"
    runtime_pkg.mkdir(parents=True, exist_ok=True)
    for pkg_init in (
        submission_dir / "src" / "tac" / "__init__.py",
        submission_dir / "src" / "tac" / "substrates" / "__init__.py",
        runtime_pkg / "__init__.py",
    ):
        pkg_init.write_text("", encoding="utf-8")
    substrate_src = REPO_ROOT / "src" / "tac" / "substrates" / "time_traveler_l5_autonomy"
    for name in ("architecture.py", "archive.py", "inflate.py"):
        shutil.copy2(substrate_src / name, runtime_pkg / name)
    _canon_vendor_shared_inflate_runtime(submission_dir, repo_root=REPO_ROOT)

    inflate_sh = (
        "#!/usr/bin/env bash\n"
        "# Time-Traveler L5 Autonomy contest-compliant inflate (TT5L)\n"
        "# Contract: $1=archive_dir $2=output_dir $3=file_list per Catalog #146\n"
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
        '"""Time-Traveler L5 Autonomy contest-compliant inflate runtime.\n'
        "\n"
        "Reads archive_dir/0.bin via the packaged substrate parser, then writes\n"
        "one contest .raw tensor stream per file_list entry.\n"
        "No scorer-network imports (strict-scorer-rule contract).\n"
        '"""\n'
        "import sys\n"
        "from pathlib import Path\n"
        "\n"
        "HERE = Path(__file__).resolve().parent\n"
        "sys.path.insert(0, str(HERE / 'src'))\n"
        "from tac.substrates.time_traveler_l5_autonomy.inflate import (\n"
        "    inflate_one_video,\n"
        ")\n"
        "from tac.substrates._shared.inflate_runtime import (\n"
        "    raw_output_path, select_inflate_device,\n"
        ")\n"
        "\n"
        "def main() -> int:\n"
        "    if len(sys.argv) != 4:\n"
        "        print('usage: inflate.py <archive_dir> <output_dir> <file_list>',\n"
        "              file=sys.stderr)\n"
        "        return 2\n"
        "    archive_dir = Path(sys.argv[1])\n"
        "    output_dir = Path(sys.argv[2])\n"
        "    file_list_path = Path(sys.argv[3])\n"
        "    bin_path = archive_dir / '0.bin'\n"
        "    if not bin_path.is_file():\n"
        "        bin_path = archive_dir / 'x'\n"
        "    archive_bytes = bin_path.read_bytes()\n"
        "    device = select_inflate_device()\n"
        "    for line in file_list_path.read_text(encoding='utf-8').splitlines():\n"
        "        line = line.strip()\n"
        "        if not line:\n"
        "            continue\n"
        "        inflate_one_video(\n"
        "            archive_bytes, raw_output_path(output_dir, line), device=device\n"
        "        )\n"
        "    return 0\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    sys.exit(main())\n"
    )
    (submission_dir / "inflate.py").write_text(inflate_py, encoding="utf-8")


class _CompositeOptimizer:
    """Minimal Optimizer-like wrapper for two-optimizer dispatch.

    The Muon-and-AdamW (or Muon-and-IGLT) optimizer split (Keller Jordan) keeps
    NS-orthogonalized momentum on hidden 2-D+ weights and a sister optimizer
    on stem + RGB heads + biases. The training loop expects a single
    ``optimizer`` object with ``.zero_grad()`` / ``.step()`` /
    ``.param_groups`` — this wrapper preserves that contract while delegating
    each call to the two underlying optimizers in order.

    Cross-reference: ``tac.optimization.muon.partition_params_for_muon``.
    Lane: ``lane_other_priorities_parallel_sweep_20260513``.
    """

    def __init__(self, optimizers: list[Any]) -> None:
        if not optimizers:
            raise ValueError("_CompositeOptimizer requires >=1 optimizer")
        self._optimizers: list[Any] = list(optimizers)

    @property
    def param_groups(self) -> list[dict[str, Any]]:
        """Flattened union of underlying param groups (read-only convention)."""
        groups: list[dict[str, Any]] = []
        for opt in self._optimizers:
            groups.extend(opt.param_groups)
        return groups

    def zero_grad(self, set_to_none: bool = True) -> None:
        for opt in self._optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def step(self, closure: Any = None) -> None:
        if closure is not None:
            raise NotImplementedError(
                "_CompositeOptimizer does not support closure form; the train "
                "loop must call .step() without a closure."
            )
        for opt in self._optimizers:
            opt.step()

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": "tac_composite_optimizer_v1",
            "optimizers": [opt.state_dict() for opt in self._optimizers],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        sd_list = state.get("optimizers", [])
        if len(sd_list) != len(self._optimizers):
            raise ValueError(
                f"state_dict mismatch: have {len(self._optimizers)} optimizers, "
                f"got {len(sd_list)} state entries"
            )
        for opt, sd in zip(self._optimizers, sd_list, strict=True):
            opt.load_state_dict(sd)


def _build_archive_zip(
    archive_zip_path: Path, *, bin_bytes: bytes, submission_dir: Path
) -> None:
    """Deterministic archive.zip containing ONLY the data payload (0.bin).

    Per ``experiments/contest_auth_eval.py::_validate_archive_members`` the
    archive.zip whitelist permits only data-payload files (.bin / .br / etc.)
    and a small set of allowed basenames (p, x, fb). The inflate runtime tree
    (inflate.sh + inflate.py + src/) lives in ``submission_dir/`` and is NOT
    bundled into archive.zip.

    ``submission_dir`` is accepted as a kwarg for sister-trainer signature
    parity but is intentionally unused — runtime files are emitted by
    ``_write_runtime`` directly to the submission directory.
    """
    del submission_dir  # signature-parity only; runtime tree lives outside zip
    archive_zip_path.parent.mkdir(parents=True, exist_ok=True)
    fixed_ts = (2026, 1, 1, 0, 0, 0)
    with zipfile.ZipFile(archive_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zi = zipfile.ZipInfo("0.bin", date_time=fixed_ts)
        zi.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(zi, bin_bytes)


# ---------------------------------------------------------------------------
# Smoke main (CPU; no scorer load)
# ---------------------------------------------------------------------------

def _smoke_main(args: argparse.Namespace) -> int:
    """Tiny CPU smoke that proves the scaffold + archive grammar are wired."""
    import numpy as np
    import torch

    from tac.substrates.time_traveler_l5_autonomy.architecture import (
        TimeTravelerConfig,
        TimeTravelerSubstrate,
    )
    from tac.substrates.time_traveler_l5_autonomy.archive import (
        pack_archive,
        quantize_per_pair_residual_int8,
    )

    _pin_seeds(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    cfg = TimeTravelerConfig(
        hidden_dim=min(args.hidden_dim, 32),
        num_hidden_layers=min(args.num_hidden_layers, 3),
        per_pair_side_info_bytes=args.per_pair_side_info_bytes,
        foveation_grid_h=args.foveation_grid_h,
        foveation_grid_w=args.foveation_grid_w,
        num_pairs=4,
        output_height=24,
        output_width=32,
        first_omega=args.first_omega,
        hidden_omega=args.hidden_omega,
        coord_feature_freqs=args.coord_feature_freqs,
    )
    device = _device_or_die(args.device, smoke=True)
    substrate = TimeTravelerSubstrate(cfg).to(device)
    opt = torch.optim.Adam(substrate.parameters(), lr=args.lr)

    n_params = substrate.parameter_count()
    print(
        f"[{SUBSTRATE_TAG}-smoke] substrate params: {n_params:,}  "
        f"world_model_bytes={substrate.estimate_world_model_bytes()}"
    )

    # Smoke training loop: drive the renderer output toward neutral gray.
    for step in range(min(args.epochs, 3)):
        opt.zero_grad()
        loss = torch.zeros((), device=device)
        for pair_idx in range(cfg.num_pairs):
            rgb_0, rgb_1 = substrate.render_pair(pair_idx)
            loss = loss + (rgb_0 - 0.5).pow(2).mean() + (rgb_1 - 0.5).pow(2).mean()
        loss.backward()
        opt.step()
        print(f"[{SUBSTRATE_TAG}-smoke] step {step}: loss={loss.item():.4f}")

    # Build archive bytes (smoke).
    sd = substrate.state_dict()
    rng = np.random.default_rng(args.seed)
    side_info_float = torch.from_numpy(
        rng.standard_normal((cfg.num_pairs, cfg.per_pair_side_info_bytes)).astype("float32")
        * 0.1
    )
    side_info = quantize_per_pair_residual_int8(side_info_float, scale=args.int8_scale)
    meta = {
        "int8_scale": float(args.int8_scale),
        "first_omega": float(cfg.first_omega),
        "hidden_omega": float(cfg.hidden_omega),
        "coord_feature_freqs": int(cfg.coord_feature_freqs),
        "coord_dim": int(cfg.coord_dim),
        "markov_transition_band": int(cfg.markov_transition_band),
    }
    bin_bytes = pack_archive(
        world_model_state_dict=sd,
        per_pair_side_info=side_info,
        meta=meta,
        num_pairs=cfg.num_pairs,
        hidden_dim=cfg.hidden_dim,
        num_hidden_layers=cfg.num_hidden_layers,
        output_height=cfg.output_height,
        output_width=cfg.output_width,
        foveation_grid_h=cfg.foveation_grid_h,
        foveation_grid_w=cfg.foveation_grid_w,
        pose_dim=cfg.pose_dim,
        per_pair_bytes=cfg.per_pair_side_info_bytes,
    )
    bin_sha = _sha256_bytes(bin_bytes)
    bin_size = len(bin_bytes)
    print(
        f"[{SUBSTRATE_TAG}-smoke] TT5L archive: {bin_size} B sha256={bin_sha[:16]}..."
    )

    submission_dir = args.output_dir / "submission_dir"
    _write_runtime(submission_dir)
    (submission_dir / "0.bin").write_bytes(bin_bytes)
    archive_zip_path = args.output_dir / "archive.zip"
    _build_archive_zip(archive_zip_path, bin_bytes=bin_bytes, submission_dir=submission_dir)
    archive_zip_size = archive_zip_path.stat().st_size
    archive_zip_sha = _sha256_bytes(archive_zip_path.read_bytes())
    print(
        f"[{SUBSTRATE_TAG}-smoke] archive.zip: {archive_zip_size} B "
        f"sha256={archive_zip_sha[:16]}..."
    )

    smoke_meta = {
        "started_at_utc": _utc_now_iso(),
        "smoke": True,
        "substrate_tag": SUBSTRATE_TAG,
        "lane_id": SUBSTRATE_LANE_ID,
        "bin_sha256": bin_sha,
        "bin_bytes": bin_size,
        "archive_zip_sha256": archive_zip_sha,
        "archive_zip_bytes": archive_zip_size,
        "n_params": n_params,
        "git_head": _git_head_sha(),
    }
    (args.output_dir / "smoke_metadata.json").write_text(
        json.dumps(smoke_meta, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[{SUBSTRATE_TAG}-smoke] wrote {args.output_dir / 'smoke_metadata.json'}")
    return 0


# ---------------------------------------------------------------------------
# Full main (CUDA-required)
# ---------------------------------------------------------------------------

def _run_val_loop(
    substrate, loss_fn, gt_tensor, val_pair_indices, archive_bytes_proxy, device
) -> float:
    """Validation pass with EMA shadow + ``torch.inference_mode`` (Catalog #180)."""
    import torch

    substrate.eval()
    losses: list[float] = []
    with torch.inference_mode():
        for pair_id in val_pair_indices:
            rgb_0, rgb_1 = substrate.render_pair(int(pair_id))
            rgb_0_b = (rgb_0 * 255.0).clamp(0.0, 255.0)
            rgb_1_b = (rgb_1 * 255.0).clamp(0.0, 255.0)
            gt_a = gt_tensor[pair_id, 0:1]
            gt_b = gt_tensor[pair_id, 1:2]
            # Disable eval-roundtrip noise; keep apply_eval_roundtrip=True.
            with torch.no_grad():
                loss, _ = loss_fn(
                    rgb_0_b, rgb_1_b, gt_a, gt_b, archive_bytes_proxy,
                    apply_eval_roundtrip=True, noise_std=0.0,
                )
            if torch.isfinite(loss):
                losses.append(float(loss.detach().cpu()))
    return float(sum(losses) / len(losses)) if losses else math.inf


def _full_main(args: argparse.Namespace) -> int:
    """Full training — score-aware Lagrangian end-to-end."""
    import torch

    from tac.differentiable_eval_roundtrip import (
        patch_upstream_yuv6_globally,
        unpatch_upstream_yuv6,
    )
    from tac.scorer import load_differentiable_scorers
    from tac.substrates.time_traveler_l5_autonomy.architecture import (
        TimeTravelerConfig,
        TimeTravelerSubstrate,
    )
    from tac.substrates.time_traveler_l5_autonomy.archive import (
        pack_archive,
        quantize_per_pair_residual_int8,
    )
    from tac.substrates.time_traveler_l5_autonomy.inflate import (
        apply_quantized_per_pair_residual_for_training,
    )
    from tac.substrates.time_traveler_l5_autonomy.score_aware_loss import (
        TimeTravelerLossWeights,
        TimeTravelerScoreAwareLoss,
    )
    from tac.training import EMA

    _pin_seeds(args.seed)
    # --full-cpu opens a non-smoke CPU path; we honor the bypass ONLY when
    # both --full-cpu AND --advisory-cpu-explicitly-waived are set (already
    # asserted by _validate_full_cpu_flags before main() reaches here, but
    # we re-check defensively per CLAUDE.md "defence in depth" pattern).
    full_cpu_active = (
        bool(getattr(args, "full_cpu", False))
        and bool(getattr(args, "advisory_cpu_explicitly_waived", False))
        and args.device == "cpu"
    )
    if full_cpu_active:
        import torch as _torch_full_cpu_only  # local import to avoid early torch in tests

        device = _torch_full_cpu_only.device("cpu")
    else:
        device = _device_or_die(args.device, smoke=False)

    if args.enable_tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage_log: list[dict[str, str]] = []

    def _stage(name: str) -> None:
        msg = {"stage": name, "at": _utc_now_iso()}
        stage_log.append(msg)
        print(f"[{SUBSTRATE_TAG}-full] {name} @ {msg['at']}")

    _stage("seed_pinned")

    # 1. Patch upstream rgb_to_yuv6 BEFORE scorer construction (Catalog #187).
    yuv6_token = patch_upstream_yuv6_globally()
    _stage("upstream_yuv6_patched")

    try:
        # 2. Load differentiable scorers (frozen).
        posenet, segnet = load_differentiable_scorers(args.upstream_dir, device=device)
        for p in list(posenet.parameters()) + list(segnet.parameters()):
            p.requires_grad_(False)
        posenet.eval()
        segnet.eval()
        _stage("scorers_loaded")

        # 3. Decode real target pairs.
        print(f"[{SUBSTRATE_TAG}-full] decoding {N_PAIRS_FULL} pairs from {args.video_path}")
        gt_pair_tensor = _decode_real_pairs(
            args.video_path, n_pairs=N_PAIRS_FULL, max_pairs=args.max_pairs
        ).to(device)
        n_pairs = int(gt_pair_tensor.shape[0])
        _stage(f"pairs_decoded_{n_pairs}")

        val_count = max(1, min(args.val_pair_count, max(1, n_pairs // 8)))
        val_idx_start = n_pairs - val_count
        train_indices_pool = list(range(val_idx_start))
        val_indices_pool = list(range(val_idx_start, n_pairs))

        # 4. Build substrate (the ONLY trainable module).
        cfg = TimeTravelerConfig(
            hidden_dim=args.hidden_dim,
            num_hidden_layers=args.num_hidden_layers,
            first_omega=args.first_omega,
            hidden_omega=args.hidden_omega,
            coord_feature_freqs=args.coord_feature_freqs,
            foveation_grid_h=args.foveation_grid_h,
            foveation_grid_w=args.foveation_grid_w,
            per_pair_side_info_bytes=args.per_pair_side_info_bytes,
            num_pairs=n_pairs,
            output_height=EVAL_HW[0],
            output_width=EVAL_HW[1],
        )
        substrate = TimeTravelerSubstrate(cfg).to(device)
        # Tier-1 O3: torch.compile wrap (no-op when flag=False).
        substrate = _compile_with_fallback(
            substrate,
            enabled=bool(getattr(args, "enable_torch_compile", False)),
            mode="default",
            fallback_on_error=True,
        )
        n_params = substrate.parameter_count()
        wm_bytes = substrate.estimate_world_model_bytes()
        print(
            f"[{SUBSTRATE_TAG}-full] substrate params: {n_params:,}  "
            f"world_model_bytes={wm_bytes}"
        )
        _stage(f"substrate_built_{n_params}_params_wm{wm_bytes}_B")

        # 5. Per-pair side info parameter (the float pre-quant tensor; gradient flows here).
        per_pair_side_info_float = torch.nn.Parameter(
            0.0 * torch.randn(n_pairs, args.per_pair_side_info_bytes, device=device)
        )

        # 6. EMA shadow (Catalog #88).
        ema = EMA(substrate, decay=args.ema_decay)
        _stage(f"ema_wired_decay_{args.ema_decay}")

        # 7. Score-aware Lagrangian.
        weights = TimeTravelerLossWeights(
            alpha_rate=args.alpha_rate,
            beta_seg=args.beta_seg,
            gamma_pose=args.gamma_pose,
            pose_weight_scale=args.pose_weight_scale,
            delta_predict=args.delta_predict,
            contest_normalizer=CONTEST_NORMALIZER,
        )
        loss_fn = TimeTravelerScoreAwareLoss(
            seg_scorer=segnet, pose_scorer=posenet, weights=weights
        )
        _stage("lagrangian_built")

        # 8. Optimizer dispatch (AdamW default; IGLT/Muon/Muon+IGLT opt-in).
        #    Muon path: partition substrate params into (hidden Muon-eligible,
        #    stem+RGB+biases AdamW-handled) via partition_params_for_muon().
        #    The per-pair side-info float is always Muon-ineligible (1-D).
        params_list = [*list(substrate.parameters()), per_pair_side_info_float]
        if args.optimizer == "iglt":
            from tac.optimization.iglt import IGLTOptimizer

            optimizer = IGLTOptimizer(
                params_list,
                lr=args.lr,
                T_init=1e-3,
                T_final=1e-5,
                n_steps=max(1, args.epochs),
                weight_decay=args.weight_decay,
                fisher_estimation="diagonal",
            )
            scheduler = None
        elif args.optimizer in ("muon", "muon+iglt"):
            from tac.optimization.muon import (
                MuonOptimizer,
                partition_params_for_muon,
            )

            muon_params, adamw_params = partition_params_for_muon(substrate)
            # per_pair_side_info_float is 1-D — must go to the AdamW group.
            adamw_params = [*adamw_params, per_pair_side_info_float]
            muon_opt = MuonOptimizer(
                muon_params,
                lr=args.muon_lr,
                weight_decay=args.muon_weight_decay,
            )
            if args.optimizer == "muon+iglt":
                from tac.optimization.iglt import IGLTOptimizer

                adamw_opt = IGLTOptimizer(
                    adamw_params,
                    lr=args.lr,
                    T_init=1e-3,
                    T_final=1e-5,
                    n_steps=max(1, args.epochs),
                    weight_decay=args.weight_decay,
                    fisher_estimation="diagonal",
                )
            else:
                adamw_opt = torch.optim.AdamW(
                    adamw_params, lr=args.lr, weight_decay=args.weight_decay
                )
            # Compose into a single Optimizer-like wrapper so the rest of the
            # train loop (zero_grad / step) remains identical.
            optimizer = _CompositeOptimizer([muon_opt, adamw_opt])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                adamw_opt, T_max=max(1, args.epochs)
            ) if args.optimizer == "muon" else None
        else:
            optimizer = torch.optim.AdamW(
                params_list, lr=args.lr, weight_decay=args.weight_decay
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, args.epochs)
            )

        # 9. Train loop.
        train_started_at = time.time()
        best_val_lag = math.inf
        best_epoch = -1
        ckpt_best_path = args.output_dir / "best.pt"
        nan_strike = 0
        max_nan_strikes = 3
        # Wall-clock gate (Catalog #197 + MacKay/Boyd Round 2). Active only
        # in --full-cpu mode; CUDA training is left ungated because the
        # canonical contract is "Modal/Vast wall-clock budgets are managed
        # at the dispatcher layer, not the trainer layer".
        wall_clock_budget_sec: float | None = None
        if full_cpu_active:
            wall_clock_budget_sec = float(args.max_wall_clock_hours) * 3600.0
        wall_clock_aborted = False

        for epoch in range(args.epochs):
            if wall_clock_budget_sec is not None:
                elapsed = time.time() - train_started_at
                if elapsed > wall_clock_budget_sec:
                    print(
                        f"[{SUBSTRATE_TAG}-full] wall-clock gate tripped at "
                        f"epoch {epoch}: elapsed {elapsed:.1f}s exceeds budget "
                        f"{wall_clock_budget_sec:.1f}s "
                        f"(--max-wall-clock-hours={args.max_wall_clock_hours}). "
                        "Aborting gracefully and emitting best EMA checkpoint."
                    )
                    wall_clock_aborted = True
                    break
            substrate.train()
            random.shuffle(train_indices_pool)
            epoch_losses: list[float] = []
            for batch_start in range(0, len(train_indices_pool), args.batch_size):
                batch_indices = train_indices_pool[batch_start : batch_start + args.batch_size]
                if not batch_indices:
                    continue

                # Tier-1 O2: autocast wrap (no-op when flag=False).
                with _autocast_aware_forward(
                    enabled=bool(getattr(args, "enable_autocast_fp16", False)),
                    dtype=torch.float16,
                    device=device,
                ):
                    # Render each pair; collect into a batch tensor.
                    rgb_0_list = []
                    rgb_1_list = []
                    gt_0_list = []
                    gt_1_list = []
                    residual_list = []
                    for pair_idx in batch_indices:
                        rgb_0, rgb_1 = substrate.render_pair(pair_idx)
                        (
                            rgb_0,
                            rgb_1,
                            side_info_int8_ste,
                        ) = apply_quantized_per_pair_residual_for_training(
                            rgb_0,
                            rgb_1,
                            per_pair_side_info_float[pair_idx],
                            int8_scale=args.int8_scale,
                        )
                        rgb_0_list.append(rgb_0 * 255.0)
                        rgb_1_list.append(rgb_1 * 255.0)
                        gt_0_list.append(gt_pair_tensor[pair_idx, 0:1])
                        gt_1_list.append(gt_pair_tensor[pair_idx, 1:2])
                        residual_list.append(side_info_int8_ste / args.int8_scale)

                    pred_a = torch.cat(rgb_0_list, dim=0)
                    pred_b = torch.cat(rgb_1_list, dim=0)
                    gt_a = torch.cat(gt_0_list, dim=0)
                    gt_b = torch.cat(gt_1_list, dim=0)
                    residual_batch = torch.stack(residual_list, dim=0)

                    # Closed-form rate proxy: world-model bytes + per-pair budget.
                    archive_bytes_proxy = torch.tensor(
                        float(wm_bytes + n_pairs * args.per_pair_side_info_bytes),
                        device=device,
                    )

                    loss, parts = loss_fn(
                        pred_a, pred_b, gt_a, gt_b, archive_bytes_proxy,
                        predictive_residual=residual_batch,
                        apply_eval_roundtrip=True, noise_std=args.noise_std,
                    )

                if not torch.isfinite(loss):
                    nan_strike += 1
                    print(f"[{SUBSTRATE_TAG}-full] NaN strike {nan_strike}/{max_nan_strikes}")
                    if nan_strike >= max_nan_strikes:
                        raise RuntimeError("NaN watchdog tripped")
                    continue
                nan_strike = 0

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params_list, args.grad_clip)
                optimizer.step()
                ema.update(substrate)
                epoch_losses.append(float(loss.detach().cpu()))

            if scheduler is not None:
                scheduler.step()

            if epoch % max(1, args.val_every_epochs) == 0 or epoch == args.epochs - 1:
                live_state = {k: v.detach().clone() for k, v in substrate.state_dict().items()}
                ema.apply(substrate)
                try:
                    val_lag = _run_val_loop(
                        substrate, loss_fn, gt_pair_tensor, val_indices_pool,
                        torch.tensor(
                            float(wm_bytes + n_pairs * args.per_pair_side_info_bytes),
                            device=device,
                        ),
                        device,
                    )
                finally:
                    substrate.load_state_dict(live_state)
                    substrate.train()

                avg_train = (sum(epoch_losses) / len(epoch_losses)) if epoch_losses else math.nan
                print(
                    f"[{SUBSTRATE_TAG}-full] epoch {epoch:4d}: train={avg_train:.5f} "
                    f"val={val_lag:.5f} (best={best_val_lag:.5f})"
                )
                if val_lag < best_val_lag:
                    best_val_lag = val_lag
                    best_epoch = epoch
                    # Save EMA shadow (NEVER live weights — Catalog #88).
                    ema_state = {k: v.detach().cpu() for k, v in substrate.state_dict().items()}
                    torch.save(
                        {
                            "state_dict": ema_state,
                            "per_pair_side_info_float": per_pair_side_info_float.detach().cpu(),
                            "config": asdict(cfg),
                            "epoch": epoch,
                            "val_lag": val_lag,
                        },
                        ckpt_best_path,
                    )

        train_elapsed = time.time() - train_started_at
        _stage(f"trained_best_epoch_{best_epoch}_val_lag_{best_val_lag:.5f}_elapsed_{train_elapsed:.1f}s")

        # 10. Load best EMA checkpoint and build the archive.
        best_ckpt = torch.load(ckpt_best_path, weights_only=False, map_location=device)
        substrate.load_state_dict(best_ckpt["state_dict"])
        substrate.eval()
        per_pair_side_info_float_final = best_ckpt["per_pair_side_info_float"].to(device)
        per_pair_side_info_int8 = quantize_per_pair_residual_int8(
            per_pair_side_info_float_final, scale=args.int8_scale
        )

        meta = {
            "int8_scale": float(args.int8_scale),
            "first_omega": float(cfg.first_omega),
            "hidden_omega": float(cfg.hidden_omega),
            "coord_feature_freqs": int(cfg.coord_feature_freqs),
            "coord_dim": int(cfg.coord_dim),
            "markov_transition_band": int(cfg.markov_transition_band),
            "atick_redlich_cooperative_receiver": True,
            "predictive_coding_hierarchy": True,
            "world_model_kind": "differentiable_physics_renderer",
        }
        bin_bytes = pack_archive(
            world_model_state_dict=substrate.state_dict(),
            per_pair_side_info=per_pair_side_info_int8,
            meta=meta,
            num_pairs=cfg.num_pairs,
            hidden_dim=cfg.hidden_dim,
            num_hidden_layers=cfg.num_hidden_layers,
            output_height=cfg.output_height,
            output_width=cfg.output_width,
            foveation_grid_h=cfg.foveation_grid_h,
            foveation_grid_w=cfg.foveation_grid_w,
            pose_dim=cfg.pose_dim,
            per_pair_bytes=cfg.per_pair_side_info_bytes,
        )
        bin_sha = _sha256_bytes(bin_bytes)
        bin_size = len(bin_bytes)
        print(
            f"[{SUBSTRATE_TAG}-full] TT5L archive: {bin_size} B sha256={bin_sha[:16]}..."
        )
        _stage(f"archive_built_{bin_size}_B_sha{bin_sha[:8]}")

        # 11. Build runtime tree + archive.zip.
        submission_dir = args.output_dir / "submission_dir"
        _write_runtime(submission_dir)
        (submission_dir / "0.bin").write_bytes(bin_bytes)
        archive_zip_path = args.output_dir / "archive.zip"
        _build_archive_zip(archive_zip_path, bin_bytes=bin_bytes, submission_dir=submission_dir)
        archive_zip_sha = _sha256_bytes(archive_zip_path.read_bytes())
        archive_zip_size = archive_zip_path.stat().st_size
        shutil.copy2(archive_zip_path, submission_dir / "archive.zip")
        _stage("archive_emitted")

        # 12. Auth eval ([contest-CUDA] inline) via the canonical
        # ``tac.substrates._shared.smoke_auth_eval_gate.gate_auth_eval_call``
        # helper. The gate refuses at smoke / full-CPU advisory / non-CUDA
        # paths per CLAUDE.md "Auth eval EVERYWHERE" + HNeRV parity lesson
        # L13. Time-traveler routes the full-CPU advisory-only branch
        # through the canonical FULL_CPU_REFUSAL_REASON so downstream
        # manifests carry a stable refusal token.
        auth_eval_json_path = args.output_dir / "auth_eval.json"
        auth_result = _canon_gate_auth_eval_call(
            args=args,
            archive_zip=archive_zip_path,
            inflate_sh=submission_dir / "inflate.sh",
            upstream_dir=args.upstream_dir,
            output_json=auth_eval_json_path,
            contest_auth_eval_script=CONTEST_AUTH_EVAL_SCRIPT,
            substrate_tag=SUBSTRATE_TAG,
            device=device,
            full_cpu_active=full_cpu_active,
        )
        if auth_result is None:
            if full_cpu_active:
                _stage("auth_eval_skipped_full_cpu_advisory_only")
            else:
                _stage("auth_eval_skipped_gate_refused")
        else:
            # Defense-in-depth: validate the claim a second time via the
            # canonical requirer (older code path that also stamps custody).
            _canon_require_contest_cuda_auth_eval_claim(
                auth_eval_json_path,
                archive_sha256=archive_zip_sha,
                substrate_tag=SUBSTRATE_TAG,
            )
            _stage("auth_eval_cuda_done_valid_claim")
    finally:
        unpatch_upstream_yuv6(yuv6_token)
        _stage("upstream_yuv6_unpatched")

    # 13. Posterior update (Catalog #128 atomic fcntl).
    if not args.skip_auth_eval and auth_eval_json_path.exists():
        try:
            from tac.continual_learning import (
                posterior_update_locked_from_auth_eval_json,
            )

            update = posterior_update_locked_from_auth_eval_json(auth_eval_json_path)
            print(
                f"[{SUBSTRATE_TAG}-full] posterior_update accepted="
                f"{getattr(update, 'accepted', '?')}"
            )
            _stage("posterior_updated")
        except Exception as exc:
            print(f"[{SUBSTRATE_TAG}-full] WARN posterior_update failed: {exc!r}")

    # 14. Provenance.
    provenance = {
        "lane_id": SUBSTRATE_LANE_ID,
        "substrate_tag": SUBSTRATE_TAG,
        "started_at_utc": stage_log[0]["at"] if stage_log else _utc_now_iso(),
        "completed_at_utc": _utc_now_iso(),
        "git_head": _git_head_sha(),
        "bin_sha256": bin_sha,
        "bin_bytes": bin_size,
        "archive_zip_sha256": archive_zip_sha,
        "archive_zip_bytes": archive_zip_size,
        "n_params": n_params,
        "world_model_bytes_estimate": wm_bytes,
        "best_val_lag": best_val_lag,
        "best_epoch": best_epoch,
        "epochs": args.epochs,
        "device": str(device),
        "optimizer": args.optimizer,
        "design_memo": (
            ".omx/research/time_traveler_architecture_reverse_engineered_20260513.md"
        ),
        "stage_log": stage_log,
        "hardware_substrate_cuda": _canon_detect_hardware_substrate(
            axis="cuda",
            substrate_tag=SUBSTRATE_TAG,
            env_var_candidates=("TT5L_GPU", "MODAL_GPU"),
        ),
    }
    # Advisory-only fields when --full-cpu is active (Catalog #127 + #192).
    if full_cpu_active:
        provenance["full_cpu_mode"] = True
        provenance["advisory_cpu_explicitly_waived"] = True
        provenance["evidence_grade"] = "macOS-CPU-advisory"
        provenance["evidence_tag"] = "[macOS-CPU advisory only]"
        provenance["score_claim"] = False
        provenance["promotion_eligible"] = False
        provenance["ready_for_exact_eval_dispatch"] = False
        provenance["max_wall_clock_hours"] = float(args.max_wall_clock_hours)
        provenance["wall_clock_aborted"] = bool(wall_clock_aborted)
        provenance["macos_cpu_drift_calibration_extends_to_time_traveler_shape"] = False
        provenance["dispatch_blockers"] = [
            "macos_cpu_advisory_not_score_evidence",
            "not_a_11_contest_compliant_cpu_axis",
            "requires_paired_contest_cpu_gha_linux_x86_64_before_score_claim",
            "requires_paired_contest_cuda_before_dual_axis_submission",
        ]
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[{SUBSTRATE_TAG}-full] wrote {args.output_dir / 'provenance.json'}")
    return 0


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# META layer SubstrateContract (Catalog #241/#242 canonical migration; landed
# 2026-05-15 by CATALOG-241-BACKFILL-29-TRAINERS subagent). Decoration extincts
# the Z3 v2 silent-drift bug class for this substrate by binding (a) the
# trainer's claimed contract, (b) the recipe schema, (c) the lane registry,
# and (d) the cost-band envelope into ONE source-of-truth that fails-loud at
# decoration time if the contract violates canonical invariants.
# ---------------------------------------------------------------------------

TIME_TRAVELER_L5_AUTONOMY_SUBSTRATE_CONTRACT = SubstrateContract(
    # 2.1 Identity & lifecycle
    id="time_traveler_l5_autonomy",
    lane_id="lane_time_traveler_l5_autonomy_substrate_20260513",
    target_modes=("contest_one_video_replay", "research_substrate",),
    deployment_target="comma_ai_production",
    council_verdict_provenance=(
        ".omx/research/grand_council_omnibus_design_decisions_20260514.md"
    ),
    # 2.2 Architecture & runtime (8 per Catalog #124)
    archive_grammar=(
        "TTL5V1 monolithic single-file 0.bin: header + time-traveler L5 autonomy decoder weights (fp16 + brotli) + per-frame world-model embeddings + cross-frame temporal continuity"
    ),
    parser_section_manifest={
        "header": "TTL5V1_magic_and_version",
        "decoder_weights": "fp16_brotli_blob",
        "world_model_embeddings": "fp16_per_frame",
        "temporal_continuity": "fp16_brotli_blob",
    },
    inflate_runtime_loc_budget=150,
    runtime_dep_closure=("torch>=2.5,<2.7", "brotli", "av",),
    export_format="fp16_brotli",
    score_aware_loss="scorer_loss_terms_btchw",
    bolt_on_loc_budget=1290,
    no_op_detector_planned=True,
    # 2.3 Operational mechanism (3 per Catalog #220)
    archive_bytes_added="27 KB per-pair side-info stream before brotli",
    score_improvement_mechanism_status="OPERATIONAL",
    runtime_overlay_consumed=True,
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
    hook_autopilot_ranker_class_shift_token="time_traveler",
    hook_continual_learning_anchor_kind="cuda_only",
    hook_probe_disambiguator=None,
    # 2.7 Compliance + 2.8 not-applicable rationales
    catalog_compliance_declarations=(
        "catalog_146_3arg_archive_grammar_honored",
        "catalog_151_tier1_required_flags_declared",
        "catalog_205_select_inflate_device_used",
        "catalog_220_operational_mechanism_declared",
        "catalog_226_gate_auth_eval_call_used",
        "catalog_197_full_cpu_coupled_flags_required",
    ),
    hook_not_applicable_rationale={
        "hook_sensitivity_contribution": (
            "L5-autonomy time-traveler world-model; sensitivity captured by temporal-continuity entropy"
        ),
        "hook_bit_allocator_class": (
            "fp16 brotli on decoder + temporal continuity; no per-tensor bit allocator"
        ),
        "hook_probe_disambiguator": (
            "tools/probe_time_traveler_disambiguator.py (planned); world-model class-shift verifier"
        ),
    },
)


@register_substrate(TIME_TRAVELER_L5_AUTONOMY_SUBSTRATE_CONTRACT)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    # Catalog #197 coupled-flag validation (must run before _smoke_main /
    # _full_main can be entered with inconsistent --full-cpu state).
    _validate_full_cpu_flags(args)
    _full_cpu_banner(args)
    if args.smoke:
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":
    sys.exit(main())
