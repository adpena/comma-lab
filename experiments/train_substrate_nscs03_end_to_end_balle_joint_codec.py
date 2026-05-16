# SPDX-License-Identifier: MIT
"""Train the NSCS03 end-to-end Ballé joint codec substrate.

Operator-callable training script for NSCS03 per the assumptions-challenge-audit
NSCS03 design memo and the operator NON-NEGOTIABLE *"UNIQUE-AND-COMPLETE-PER-METHOD"*
mode landed 2026-05-15. The substrate is **end-to-end Ballé 2018 joint codec**
— convolutional analysis g_a + entropy bottleneck + scale hyperprior +
convolutional synthesis g_s — joint-trained with score-aware loss
backpropagating THROUGH the bottleneck.

Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"
non-negotiable: this trainer's ``_full_main`` is now WIRED end-to-end
following the canonical PR95 paradigm + Ballé 2018 recipe (per the
UNIQUE-AND-COMPLETE-PER-METHOD directive landed 2026-05-15). The recipe
remains ``research_only: true`` until the Phase 2 council adjudicates the
λ_R sweep and σ-floor calibration; until then the FULL path runs but is
gated by the recipe (no $1+ Modal dispatch fires through operator_authorize).

Council-binding contract (CLAUDE.md non-negotiables) honored end-to-end:

- Train against ``upstream/videos/0.mkv`` decoded via pyav (NOT synthetic
  data per Catalog #114). Synthetic batches forbidden outside ``--smoke``.
- Patch upstream ``rgb_to_yuv6`` via ``patch_upstream_yuv6_globally`` BEFORE
  scorer construction (PR #95/#106 contract).
- ``load_differentiable_scorers`` for SegNet/PoseNet (no scorer load at
  inflate; only at training).
- ``apply_eval_roundtrip_during_training`` inside the per-batch loop via
  the canonical score-aware loss; ``eval_roundtrip=True`` (non-negotiable).
- ``tac.training.EMA`` update after every ``optimizer.step``; inference
  checkpoint = EMA shadow (CLAUDE.md "EMA — non-negotiable"). NSCS03 uses
  DIFFERENTIATED decay per Ballé 2018: 0.999 for the entropy-bottleneck
  factorized prior + hyper paths (slow-moving distribution parameters),
  0.997 for g_a/g_s/h_a/h_s (canonical across-board). The composite is a
  single EMA on the joint state_dict; we expose only ``--ema-decay``
  (default 0.997) for now and document the Ballé differentiated split as a
  Phase 2 follow-up calibration item.
- Score-domain Lagrangian per HNeRV parity L6 PLUS NSCS03-specific rate term::

      L = α·B(θ)/N + β·d_seg(θ) + γ·sqrt(d_pose(θ)) + λ_R·(R_main + R_hyper)

  where R_main = -log2 N(y_hat; 0, σ²) (scale hyperprior) and
  R_hyper = -log2 p_z(z_hat) (factorized prior) are differentiable through
  the entropy model (Ballé 2017 uniform-noise relaxation during train).
- AdamW + Ballé-style WARMUP-COSINE LR schedule (UNIQUE: λ_R is ramped from
  0 to target across the first 10% of training to let the substrate first
  learn to reconstruct, then add rate pressure). Gradient clip 1.0;
  NaN watchdog (3-strike) per Council D.
- End with CUDA auth eval per CLAUDE.md "Auth eval EVERYWHERE" via the
  canonical ``gate_auth_eval_call`` (Catalog #226).
- Continual-learning posterior update via ``posterior_update_locked``
  (Catalog #128 atomic fcntl).
- Cost-band anchor append via ``tools/append_cost_band_anchor.py``.
- Contest-compliant runtime emission (inflate.sh / inflate.py with 3
  positional args + ``set -euo pipefail`` + NO scorer imports) per
  Catalog #146; inflate.py ≤ 200 LOC per the L4 NEEDS-WORK waiver.
- TIER_1_OPERATOR_REQUIRED_FLAGS declared per Catalog #151 for wire-up.

CANONICAL-VS-UNIQUE decisions per layer (per UNIQUE-AND-COMPLETE-PER-METHOD
directive — every layer is either an EXPLICIT-ADOPT of a canonical helper
OR an EXPLICIT-FORK with rationale):

1. Seed pinning        → CANONICAL ADOPT  (trainer_skeleton.pin_seeds)
2. Device resolution   → CANONICAL ADOPT  (trainer_skeleton.device_or_die)
3. yuv6 patching       → CANONICAL ADOPT  (patch_upstream_yuv6_globally)
4. Scorer load         → CANONICAL ADOPT  (load_differentiable_scorers)
5. Video decode        → CANONICAL ADOPT  (trainer_skeleton.decode_real_pairs)
6. Substrate build     → UNIQUE           (NSCS03JointCodecSubstrate end-to-end)
7. EMA                 → CANONICAL ADOPT  (tac.training.EMA; documented Ballé
                                            differentiated-decay deferred)
8. Score-aware loss    → UNIQUE WRAPPER over canonical scorer-preprocess
                         (NSCS03JointScoreAwareLoss; routes through
                         score_pair_components_dispatch per Catalog #164;
                         adds the END-TO-END differentiable rate term that
                         CANONICAL substrate losses do not have)
9. Optimizer           → CANONICAL ADOPT  (AdamW + CosineAnnealingLR)
10. λ_R warmup         → UNIQUE           (Ballé recipe: rate-term ramp over
                                            first 10% of training so the
                                            substrate learns to reconstruct
                                            before being penalized for rate)
11. Tier-1 autocast    → DOCUMENTED FORK  (`AUTOCAST_FP16_WAIVED` because the
                                            EntropyBottleneck logistic CDF +
                                            GDN forward have numerical
                                            instability in fp16 per the
                                            architecture's fp32-guard policy;
                                            Phase 2 follow-up may revisit per
                                            bb fp32-eb-mixed-precision recipe)
12. Tier-1 torch.compile→ DOCUMENTED FORK (`TORCH_COMPILE_WAIVED` — defer
                                            until per-substrate canary
                                            validates Inductor graph breaks)
13. F3 GT-scorer cache → CANONICAL ADOPT  (build_optimized_training_context)
14. NaN watchdog       → CANONICAL PATTERN (Council D 3-strike pattern)
15. Validation         → CANONICAL PATTERN (snapshot+restore + EMA-apply for
                                            best-ckpt by val Lagrangian)
16. Archive build      → UNIQUE           (5 state_dicts + 2 latent streams
                                            from `encode()` on real GT pairs;
                                            NS03 monolithic 0.bin grammar)
17. Runtime emission   → CANONICAL ADOPT  (vendor_shared_inflate_runtime +
                                            3-arg inflate.sh per Catalog #146;
                                            inflate.py is a thin passthrough)
18. Auth eval routing  → CANONICAL ADOPT  (gate_auth_eval_call per Catalog #226)
19. Posterior update   → CANONICAL ADOPT  (posterior_update_locked + dynamic
                                            hardware substrate detection
                                            per Catalog #190)
20. Cost-band anchor   → CANONICAL ADOPT  (append_cost_band_anchor.py)
21. Provenance         → CANONICAL PATTERN (regen-header + custody fields +
                                            score axis tag per Catalog #113)

Usage (smoke; CPU; deterministic synthetic batches; no scorer load)::

    .venv/bin/python experiments/train_substrate_nscs03_end_to_end_balle_joint_codec.py \\
        --output-dir experiments/results/nscs03_smoke_<utc> \\
        --epochs 3 --device cpu --smoke

Usage (full; CUDA-required; threads from operator wrapper)::

    .venv/bin/python experiments/train_substrate_nscs03_end_to_end_balle_joint_codec.py \\
        --video-path upstream/videos/0.mkv \\
        --output-dir experiments/results/nscs03_<utc> \\
        --epochs 2000 --batch-size 16 --lr 5e-4 --device cuda
"""
# AUTOCAST_FP16_WAIVED:entropy-bottleneck-numerical-instability-pending-fp32-eb-forward
# TORCH_COMPILE_WAIVED:defer-until-per-substrate-canary-validates-Inductor-graph-breaks
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

from tac.substrates._shared.smoke_auth_eval_gate import (
    gate_auth_eval_call as _canon_gate_auth_eval_call,
)
from tac.substrates._shared.trainer_skeleton import (
    build_optimized_training_context as _canon_build_optimized_training_context,
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
from tac.substrates.nscs03_end_to_end_balle_joint_codec.registered_substrate import (
    NSCS03_END_TO_END_BALLE_CONTRACT,  # noqa: F401  (forces contract validation)
)

# ---------------------------------------------------------------------------
# Module paths + constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO_PATH = REPO_ROOT / "upstream" / "videos" / "0.mkv"
DEFAULT_UPSTREAM_DIR = REPO_ROOT / "upstream"
CONTEST_AUTH_EVAL_SCRIPT = REPO_ROOT / "experiments" / "contest_auth_eval.py"
COST_BAND_TOOL = REPO_ROOT / "tools" / "append_cost_band_anchor.py"

# Eval-roundtrip target resolution (per upstream evaluate.py):
EVAL_HW = (384, 512)
CAMERA_HW = (874, 1164)
N_PAIRS_FULL = 600
CONTEST_NORMALIZER = 37_545_489.0


# ---------------------------------------------------------------------------
# Catalog #151 manifest — every flag below must be threaded by any operator
# wrapper that subprocess-invokes this trainer.
# ---------------------------------------------------------------------------
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--video-path": {
        "env": "NSCS03_VIDEO_PATH",
        "rationale": (
            "score-aware substrate MUST train against the contest video "
            "(upstream/videos/0.mkv); synthetic data is FORBIDDEN outside --smoke"
        ),
        "default": str(DEFAULT_VIDEO_PATH.relative_to(REPO_ROOT)),
        "required_input_file": True,
        "generator_command": "contest-pinned upstream snapshot — never regenerated locally",
        "rationale_audit": (
            ".omx/research/assumptions_challenge_audit_shared_assumptions_matrix_20260515.json"
            "#NSCS03"
        ),
    },
    "--output-dir": {
        "env": "NSCS03_OUTPUT_DIR",
        "rationale": "custody location for checkpoints + archive + provenance",
    },
    "--epochs": {
        "env": "NSCS03_EPOCHS",
        "rationale": (
            "end-to-end joint codec; under-training silently regresses (council "
            "target: 2000 for full; 100 for smoke)"
        ),
        "default": "2000",
    },
    "--upstream-dir": {
        "env": "NSCS03_UPSTREAM_DIR",
        "rationale": (
            "upstream/ root for scorer weights + evaluate.py; required for full "
            "training (non-smoke) and auth eval"
        ),
        "default": str(DEFAULT_UPSTREAM_DIR.relative_to(REPO_ROOT)),
    },
    "--device": {
        "env": "NSCS03_DEVICE",
        "rationale": (
            "compute device; cuda required for full training (MPS refused per "
            "CLAUDE.md MPS-NOISE rule); cpu permitted only with --smoke"
        ),
        "default": "cuda",
    },
    "--main-latent-channels": {
        "env": "NSCS03_MAIN_LATENT_CHANNELS",
        "rationale": (
            "NSCS03-specific: main latent y channels (Ballé 2018 reference 192; "
            "ours 64 to keep param/rate envelope compatible with the 0.19 "
            "frontier). Sweep [48, 64, 96, 128] in follow-up wave."
        ),
        "default": "64",
    },
    "--hyper-latent-channels": {
        "env": "NSCS03_HYPER_LATENT_CHANNELS",
        "rationale": (
            "NSCS03-specific: hyper latent z channels (Ballé 2018 reference 128; "
            "ours 32). Side-info stream determines the conditional Gaussian sigma "
            "for the main latent."
        ),
        "default": "32",
    },
    "--lambda-R": {
        "env": "NSCS03_LAMBDA_R",
        "rationale": (
            "NSCS03-specific: weight on the differentiable rate term "
            "(R_main + R_hyper). Default 0.5; sweep [0.1, 1.0] in follow-up."
        ),
        "default": "0.5",
    },
    "--gdn-eps": {
        "env": "NSCS03_GDN_EPS",
        "rationale": (
            "NSCS03-specific: GDN numerical floor; default 1e-6 (NOT 1e-12) for "
            "fp16-autocast hygiene per Catalog #172."
        ),
        "default": "1e-6",
    },
    "--sigma-floor": {
        "env": "NSCS03_SIGMA_FLOOR",
        "rationale": (
            "NSCS03-specific: minimum sigma for the conditional-Gaussian density "
            "on y. Prevents degenerate p_y collapse to delta-function."
        ),
        "default": "1e-4",
    },
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="train_substrate_nscs03_end_to_end_balle_joint_codec",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # ---- TIER_1 required ----
    p.add_argument("--video-path", type=Path, default=DEFAULT_VIDEO_PATH)
    p.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=2000)

    # ---- Training hyperparameters ----
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)

    # ---- Substrate architecture knobs (NSCS03-specific) ----
    p.add_argument("--main-latent-channels", type=int, default=64)
    p.add_argument("--hyper-latent-channels", type=int, default=32)
    p.add_argument("--lambda-R", type=float, default=0.5)
    p.add_argument("--gdn-eps", type=float, default=1e-6)
    p.add_argument("--sigma-floor", type=float, default=1e-4)
    p.add_argument(
        "--noise-std",
        type=float,
        default=0.5,
        help="STE noise std for eval-roundtrip simulation (Hotz fix).",
    )
    # UNIQUE NSCS03: Ballé λ_R warmup — see canonical-vs-unique table #10.
    p.add_argument(
        "--lambda-R-warmup-frac",
        type=float,
        default=0.10,
        help=(
            "Fraction of total epochs over which λ_R linearly ramps from 0 to "
            "target. Ballé 2018 recipe: rate-term ramp lets the substrate learn "
            "to reconstruct before being penalized for rate. Default 0.10."
        ),
    )

    # ---- Lagrangian weights (score-aware) ----
    p.add_argument("--alpha-rate", type=float, default=25.0)
    p.add_argument("--beta-seg", type=float, default=100.0)
    p.add_argument("--gamma-pose", type=float, default=math.sqrt(10.0))
    p.add_argument("--pose-weight-scale", type=float, default=1.0)

    # ---- EMA + scheduling ----
    p.add_argument(
        "--ema-decay",
        type=float,
        default=0.997,
        help=(
            "EMA decay (CLAUDE.md non-negotiable 0.997 default for weights). "
            "Ballé 2018 reference uses 0.999 for hyperprior + 0.997 for main; "
            "we keep a single decay here and document the differentiated split "
            "as a Phase 2 follow-up calibration item."
        ),
    )
    p.add_argument("--val-every-epochs", type=int, default=10)
    p.add_argument("--val-pair-count", type=int, default=32)

    # ---- Device / mode ----
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--smoke", action="store_true")
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
    p.add_argument("--skip-auth-eval", action="store_true")
    p.add_argument("--skip-archive-build", action="store_true")

    # ---- Tier-1 optimization CLI surface ----
    p.add_argument(
        "--enable-autocast-fp16",
        action="store_true",
        default=False,
        help=(
            "Wrap forward in torch.autocast(fp16). DOCUMENTED FORK from canonical: "
            "AUTOCAST_FP16_WAIVED at module level because EB+GDN have fp16 underflow."
        ),
    )
    p.add_argument(
        "--enable-torch-compile",
        action="store_true",
        default=False,
        help=(
            "Wrap substrate with torch.compile / Inductor. DOCUMENTED FORK: "
            "TORCH_COMPILE_WAIVED — defer until per-substrate canary validates."
        ),
    )
    p.add_argument(
        "--enable-gt-scorer-cache",
        action="store_true",
        default=False,
        help="F3 GT-scorer-output cache (Catalog #228; routed via build_optimized_training_context).",
    )

    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Smoke main (CPU; no scorer load; synthetic batches OK per Catalog #114)
# ---------------------------------------------------------------------------

def _smoke_main(args: argparse.Namespace) -> int:
    """Smoke main: build the substrate; do a tiny forward + backward sanity
    check; emit a smoke stats.json. Catalog #114 compliance: synthetic data
    is permitted ONLY in smoke mode."""
    import torch

    from tac.substrates.nscs03_end_to_end_balle_joint_codec import (
        NSCS03Config,
        NSCS03JointCodecSubstrate,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[NSCS03 smoke] device={args.device} epochs={args.epochs}")

    cfg = NSCS03Config(
        main_latent_channels=args.main_latent_channels,
        hyper_latent_channels=args.hyper_latent_channels,
        gdn_eps=args.gdn_eps,
        sigma_floor=args.sigma_floor,
    )
    torch.manual_seed(args.seed)
    model = NSCS03JointCodecSubstrate(cfg).to(args.device)
    n_params = model.num_parameters()
    print(f"[NSCS03 smoke] num_params={n_params}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    for epoch in range(args.epochs):
        # SYNTHETIC_NON_SMOKE_OK:smoke-mode-only-per-Catalog-114 — _full_main
        # decodes real frames; only this smoke branch uses synthetic batches.
        x = torch.rand(2, cfg.in_channels, 384, 512, device=args.device)
        recon, parts = model(x)
        loss = (
            torch.nn.functional.mse_loss(recon, x) * 100.0
            + args.lambda_R * parts["total_rate"]
        )
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        if (epoch + 1) % max(1, args.epochs // 4) == 0:
            print(
                f"[NSCS03 smoke] epoch {epoch + 1}/{args.epochs} "
                f"loss={loss.item():.4f} main_rate={parts['main_rate'].item():.4f} "
                f"hyper_rate={parts['hyper_rate'].item():.4f}"
            )

    stats = {
        "smoke": True,
        "epochs": args.epochs,
        "num_params": n_params,
        "final_loss": float(loss.item()),
        "final_main_rate": float(parts["main_rate"].item()),
        "final_hyper_rate": float(parts["hyper_rate"].item()),
        "config": asdict(cfg),
        # Per CLAUDE.md "Apples-to-apples evidence discipline": tag the smoke
        # output with explicit non-promotion fields so it cannot be mistaken
        # for a contest-axis anchor.
        "auth_eval_score": None,
        "auth_eval_score_axis": "smoke_no_auth_eval",
        "auth_eval_score_claim_valid": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "result_review_blockers": [
            "smoke_path_synthetic_data_no_auth_eval",
            "research_only_substrate_engineering_scaffold",
        ],
    }
    out_path = args.output_dir / "stats.json"
    out_path.write_text(json.dumps(stats, indent=2, sort_keys=True))
    print(f"[NSCS03 smoke] DONE; stats written to {out_path}")
    return 0


# ---------------------------------------------------------------------------
# Contest-compliant runtime emission (Catalog #146 contract)
# ---------------------------------------------------------------------------

def _write_runtime(submission_dir: Path) -> None:
    """Emit the contest-compliant ``inflate.sh`` + ``inflate.py`` pair.

    The NSCS03 substrate's monolithic ``0.bin`` (NS03 grammar) is the
    archive; the runtime is a thin reader that calls ``parse_archive`` and
    renders frames via the synthesis transform g_s. NO scorer code imports
    (strict-scorer-rule contract); ≤ 200 LOC waiver per L4 NEEDS-WORK.
    """
    submission_dir.mkdir(parents=True, exist_ok=True)
    runtime_pkg = (
        submission_dir
        / "src"
        / "tac"
        / "substrates"
        / "nscs03_end_to_end_balle_joint_codec"
    )
    runtime_pkg.mkdir(parents=True, exist_ok=True)
    for pkg_init in (
        submission_dir / "src" / "tac" / "__init__.py",
        submission_dir / "src" / "tac" / "substrates" / "__init__.py",
        runtime_pkg / "__init__.py",
    ):
        pkg_init.write_text("", encoding="utf-8")
    substrate_src = (
        REPO_ROOT / "src" / "tac" / "substrates" / "nscs03_end_to_end_balle_joint_codec"
    )
    for name in ("architecture.py", "archive.py", "inflate.py"):
        shutil.copy2(substrate_src / name, runtime_pkg / name)
    # entropy_bottleneck primitive must be vendored too (NSCS03 architecture
    # imports `from tac.entropy_bottleneck import EntropyBottleneck`).
    entropy_src = REPO_ROOT / "src" / "tac" / "entropy_bottleneck.py"
    if entropy_src.is_file():
        shutil.copy2(entropy_src, submission_dir / "src" / "tac" / "entropy_bottleneck.py")
    _canon_vendor_shared_inflate_runtime(submission_dir, repo_root=REPO_ROOT)

    inflate_sh = (
        "#!/usr/bin/env bash\n"
        "# NSCS03 end-to-end Ballé joint codec — contest-compliant inflate\n"
        "# Contract: $1=archive_dir $2=output_dir $3=file_list (Catalog #146)\n"
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
        "\"\"\"NSCS03 contest-compliant inflate runtime.\n"
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
        "from tac.substrates.nscs03_end_to_end_balle_joint_codec.inflate import (\n"
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
        "    archive_bytes = (archive_dir / '0.bin').read_bytes()\n"
        "    device = select_inflate_device()\n"
        "    for line in file_list_path.read_text(encoding='utf-8').splitlines():\n"
        "        line = line.strip()\n"
        "        if not line:\n"
        "            continue\n"
        "        inflate_one_video(\n"
        "            archive_bytes, raw_output_path(output_dir, line), device=device,\n"
        "        )\n"
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
) -> None:
    """Deterministic charged archive.zip containing only the 0.bin packet.

    Per Catalog #19 ``check_archive_builders_use_deterministic_zip``: use
    ZipInfo + writestr with fixed timestamp + DEFLATE.
    """
    archive_zip_path.parent.mkdir(parents=True, exist_ok=True)
    fixed_ts = (2026, 1, 1, 0, 0, 0)
    with zipfile.ZipFile(archive_zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zi = zipfile.ZipInfo("0.bin", date_time=fixed_ts)
        zi.compress_type = zipfile.ZIP_DEFLATED
        zf.writestr(zi, bin_bytes)


# ---------------------------------------------------------------------------
# Archive build helpers — extract trained state_dicts + encode latents
# ---------------------------------------------------------------------------

def _extract_module_state_dicts(model) -> dict[str, dict]:
    """Split the joint-codec state_dict into the 5 archive sections.

    Returns a dict with keys: encoder / decoder / hyper_analysis /
    hyper_synthesis / entropy. Each value is the corresponding sub-module
    state_dict with the canonical sub-module prefix stripped.
    """
    full_sd = model.state_dict()
    sections: dict[str, dict] = {
        "encoder": {},  # g_a.*
        "decoder": {},  # g_s.*
        "hyper_analysis": {},  # h_a.*
        "hyper_synthesis": {},  # h_s.*
        "entropy": {},  # entropy_bottleneck_z.*
    }
    prefix_map = {
        "g_a.": "encoder",
        "g_s.": "decoder",
        "h_a.": "hyper_analysis",
        "h_s.": "hyper_synthesis",
        "entropy_bottleneck_z.": "entropy",
    }
    for key, value in full_sd.items():
        for prefix, section in prefix_map.items():
            if key.startswith(prefix):
                sections[section][key[len(prefix):]] = value
                break
    return sections


def _encode_latents_for_archive(model, pair_tensor) -> tuple:
    """Run the analysis path on every pair tensor and quantize for archive.

    Returns ``(main_latents_quantized, hyper_latents_quantized)`` — each is
    a ``(num_pairs, c, h, w)`` tensor in the same float dtype as the model.
    The trainer hard-rounds the latents at archive build time (NOT noise-
    relaxed); this matches the deterministic inflate path.
    """
    import torch

    # pair_tensor: (N, 2, 3, H, W) float in [0, 255]; substrate wants
    # (B, 6, H, W) in [0, 1].
    n = pair_tensor.shape[0]
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    main_chunks: list[torch.Tensor] = []
    hyper_chunks: list[torch.Tensor] = []
    chunk_size = 16
    model.eval()
    with torch.no_grad():
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk = pair_tensor[start:end].to(device=device, dtype=dtype)
            # Stack frames into (B, 6, H, W) in [0, 1].
            rgb_0 = chunk[:, 0] / 255.0
            rgb_1 = chunk[:, 1] / 255.0
            x_pair = torch.cat([rgb_0, rgb_1], dim=1)
            latents = model.encode(x_pair)
            # Hard-round at archive build time (deterministic inflate path).
            main_chunks.append(latents["y"].round().to(dtype=torch.float32).cpu())
            hyper_chunks.append(latents["z"].round().to(dtype=torch.float32).cpu())
    model.train()
    main_latents = torch.cat(main_chunks, dim=0)
    hyper_latents = torch.cat(hyper_chunks, dim=0)
    return main_latents, hyper_latents


# ---------------------------------------------------------------------------
# Full main (CUDA-required; score-aware Lagrangian end-to-end)
# ---------------------------------------------------------------------------

def _full_main(args: argparse.Namespace) -> int:
    """Full training entry point — requires CUDA + score-aware scorers.

    Implements the canonical PR95 paradigm + Ballé 2018 recipe for the
    NSCS03 end-to-end joint codec per the operator NON-NEGOTIABLE
    UNIQUE-AND-COMPLETE-PER-METHOD directive landed 2026-05-15.

    The wrapper (Vast.ai / Lightning / Modal / operator_authorize) threads
    all TIER_1 flags + runs the auth-eval afterward per CLAUDE.md
    "Auth eval EVERYWHERE" + "Submission auth eval — BOTH CPU AND CUDA".

    Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY":
    the recipe still declares ``research_only: true`` until Phase 2 council
    adjudicates λ_R sweep + σ-floor calibration. The full-path trainer is
    now wired so that follow-up subagents can validate empirically; the
    operator_authorize.py routing remains gated by the recipe flag.
    """
    import torch

    from tac.differentiable_eval_roundtrip import (
        patch_upstream_yuv6_globally,
        unpatch_upstream_yuv6,
    )
    from tac.scorer import load_differentiable_scorers
    from tac.substrates.nscs03_end_to_end_balle_joint_codec import (
        NSCS03Config,
        NSCS03JointCodecSubstrate,
        NSCS03JointScoreAwareLoss,
        NSCS03ScoreAwareLossWeights,
        pack_archive,
    )
    from tac.training import EMA
    from tac.training_optimization import (
        autocast_aware_forward as _autocast_aware_forward,
    )
    from tac.training_optimization import (
        compile_with_fallback as _compile_with_fallback,
    )

    # 1. Pin seeds (deterministic CUDA where possible)
    _canon_pin_seeds(args.seed)
    device = _canon_device_or_die(
        args.device, smoke=False, substrate_tag="nscs03"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage_log: list = []

    def _stage(name: str) -> None:
        stage_log.append({"stage": name, "at": _canon_utc_now_iso()})

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

        # 4. Decode real frame pairs (NOT synthetic; Catalog #114)
        print(f"[full] decoding pairs from {args.video_path} ...")
        pair_tensor = _canon_decode_real_pairs(
            args.video_path,
            n_pairs=N_PAIRS_FULL,
            substrate_tag="nscs03",
            max_pairs=args.max_pairs,
            repo_root=REPO_ROOT,
        )
        n_pairs = int(pair_tensor.shape[0])
        print(f"[full] decoded {n_pairs} pairs at {EVAL_HW}")
        pair_tensor = pair_tensor.to(device)
        _stage(f"pairs_decoded_{n_pairs}")

        # Held-out validation indices (last val_pair_count pairs)
        val_count = max(1, min(args.val_pair_count, max(1, n_pairs // 8)))
        val_idx_start = n_pairs - val_count
        train_indices = torch.arange(
            0, val_idx_start, device=device, dtype=torch.long
        )
        val_indices = torch.arange(
            val_idx_start, n_pairs, device=device, dtype=torch.long
        )

        # 5. Build substrate (UNIQUE NSCS03 architecture)
        cfg = NSCS03Config(
            main_latent_channels=args.main_latent_channels,
            hyper_latent_channels=args.hyper_latent_channels,
            gdn_eps=args.gdn_eps,
            sigma_floor=args.sigma_floor,
            quantize_noise_std=args.noise_std,
            output_height=EVAL_HW[0],
            output_width=EVAL_HW[1],
        )
        model = NSCS03JointCodecSubstrate(cfg).to(device)
        # Tier-1 O3: torch.compile wrap (DOCUMENTED FORK; default disabled).
        model = _compile_with_fallback(
            model,
            enabled=bool(getattr(args, "enable_torch_compile", False)),
            mode="default",
            fallback_on_error=True,
        )
        print(f"[full] nscs03 params: {model.num_parameters():,}")
        _stage("model_built")

        # 6. EMA shadow (CLAUDE.md non-negotiable)
        ema = EMA(model, decay=args.ema_decay)
        _stage(f"ema_wired_decay_{args.ema_decay}")

        # 7. Score-aware Lagrangian (UNIQUE NSCS03; routes through canonical
        # scorer-preprocess via score_pair_components_dispatch per Catalog #164)
        weights = NSCS03ScoreAwareLossWeights(
            alpha_rate=args.alpha_rate,
            beta_seg=args.beta_seg,
            gamma_pose=args.gamma_pose,
            pose_weight_scale=args.pose_weight_scale,
            lambda_R=args.lambda_R,
            contest_normalizer=CONTEST_NORMALIZER,
        )
        loss_fn = NSCS03JointScoreAwareLoss(
            seg_scorer=segnet,
            pose_scorer=posenet,
            weights=weights,
        )
        _stage("lagrangian_built")

        # F3 GTScorerCache wire-in (Catalog #228 via canonical helper)
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

        # UNIQUE Ballé recipe: archive_bytes_proxy is a coarse upper-bound
        # on the post-export byte budget. The DIFFERENTIABLE rate pressure
        # comes from the rate_components dict (λ_R · (R_main + R_hyper));
        # this scalar is a constant logging anchor.
        num_param_bytes = sum(p.numel() * 2 for p in model.parameters())
        # Per-pair latents: int16 = 2 bytes/element
        main_per_pair = (
            cfg.main_latent_channels
            * (cfg.output_height // 16)
            * (cfg.output_width // 16)
        )
        hyper_per_pair = (
            cfg.hyper_latent_channels
            * (cfg.output_height // 64)
            * (cfg.output_width // 64)
        )
        num_latent_bytes = n_pairs * (main_per_pair + hyper_per_pair) * 2
        archive_bytes_proxy = torch.tensor(
            float(num_param_bytes + num_latent_bytes),
            dtype=torch.float32,
            device=device,
        )

        # UNIQUE Ballé recipe: λ_R warmup. The rate-term coefficient is
        # ramped linearly from 0 → args.lambda_R over the first
        # ``lambda_R_warmup_frac`` of training. This lets the substrate
        # first learn to reconstruct (β_seg/γ_pose dominant), then add rate
        # pressure progressively. Without this, the rate term collapses
        # the substrate at epoch 1 before any structure can form.
        warmup_epochs = max(1, int(args.epochs * args.lambda_R_warmup_frac))

        def _lambda_R_at_epoch(epoch: int) -> float:
            if epoch >= warmup_epochs:
                return args.lambda_R
            return args.lambda_R * (epoch + 1) / warmup_epochs

        # NaN watchdog (Council D pattern)
        nan_strike = 0
        max_nan_strikes = 3

        for epoch in range(args.epochs):
            model.train()
            current_lambda_R = _lambda_R_at_epoch(epoch)
            # Mutate the loss-fn weight in place so the warmup ramp reaches
            # the rate-proxy term that the substrate's forward returns.
            loss_fn.weights = NSCS03ScoreAwareLossWeights(
                alpha_rate=weights.alpha_rate,
                beta_seg=weights.beta_seg,
                gamma_pose=weights.gamma_pose,
                pose_weight_scale=weights.pose_weight_scale,
                lambda_R=current_lambda_R,
                contest_normalizer=weights.contest_normalizer,
            )

            perm = train_indices[torch.randperm(n_train, device=device)]
            epoch_loss_sum = 0.0
            epoch_batches = 0
            for start in range(0, n_train, batch_size):
                idx = perm[start : start + batch_size]
                if idx.numel() == 0:
                    continue
                with _autocast_aware_forward(
                    enabled=bool(getattr(args, "enable_autocast_fp16", False)),
                    dtype=torch.float16,
                    device=device,
                ):
                    # Build (B, 6, H, W) input pair in [0, 1]
                    gt = pair_tensor[idx]  # (B, 2, 3, H, W) in [0, 255]
                    gt_0 = gt[:, 0]
                    gt_1 = gt[:, 1]
                    rgb_0_unit = gt_0 / 255.0
                    rgb_1_unit = gt_1 / 255.0
                    x_pair = NSCS03JointCodecSubstrate.stack_frames_into_pair(
                        rgb_0_unit, rgb_1_unit
                    )
                    recon, rate_components = model(x_pair)
                    rgb_0_hat, rgb_1_hat = model.split_recon_into_frames(recon)
                    # score-aware-common helpers expect [0, 255]
                    rgb_0_hat_255 = rgb_0_hat * 255.0
                    rgb_1_hat_255 = rgb_1_hat * 255.0
                    # F3 GTScorerCache lookup (per-pair-index batched).
                    gt_pose_batch = gt_seg_batch = None
                    gt_seg_already_probs = None
                    if gt_cache is not None:
                        gt_pose_batch, gt_seg_batch = gt_cache.lookup(
                            idx, device=device
                        )
                        gt_seg_already_probs = gt_cache.seg_already_probs
                    loss, parts = loss_fn(
                        rgb_0_hat_255, rgb_1_hat_255, gt_0, gt_1,
                        archive_bytes_proxy=archive_bytes_proxy,
                        rate_components=rate_components,
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
                    gt_v = pair_tensor[val_indices]
                    rgb_0_v = gt_v[:, 0] / 255.0
                    rgb_1_v = gt_v[:, 1] / 255.0
                    x_pair_v = NSCS03JointCodecSubstrate.stack_frames_into_pair(
                        rgb_0_v, rgb_1_v
                    )
                    recon_v, rate_v = model(x_pair_v)
                    rgb_0v_hat, rgb_1v_hat = model.split_recon_into_frames(
                        recon_v
                    )
                    val_pose_batch = val_seg_batch = None
                    val_seg_already_probs = None
                    if gt_cache is not None:
                        val_pose_batch, val_seg_batch = gt_cache.lookup(
                            val_indices, device=device
                        )
                        val_seg_already_probs = gt_cache.seg_already_probs
                    val_loss, _val_parts = loss_fn(
                        rgb_0v_hat * 255.0,
                        rgb_1v_hat * 255.0,
                        gt_v[:, 0],
                        gt_v[:, 1],
                        archive_bytes_proxy=archive_bytes_proxy,
                        rate_components=rate_v,
                        apply_eval_roundtrip=True,
                        noise_std=args.noise_std,
                        gt_pose_batch=val_pose_batch,
                        gt_seg_batch=val_seg_batch,
                        gt_seg_already_probs=val_seg_already_probs,
                    )
                val_lag = float(val_loss.detach().item())
                model.load_state_dict(orig_state)
                model.train()
                print(
                    f"[full] epoch {epoch + 1}/{args.epochs} "
                    f"train_avg_loss={avg_loss:.6f} val_lagrangian={val_lag:.6f} "
                    f"lambda_R={current_lambda_R:.4f} "
                    f"(best_so_far={best_val_lag:.6f} @ ep{best_epoch + 1})"
                )
                if val_lag < best_val_lag and math.isfinite(val_lag):
                    best_val_lag = val_lag
                    best_epoch = epoch
                    # Save EMA shadow (NOT live weights) — CLAUDE.md EMA rule
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
                            "saved_at_utc": _canon_utc_now_iso(),
                            "training_axis_note": (
                                "[contest-CUDA] for promotion; auth eval still required"
                            ),
                        },
                        ckpt_best_path,
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
                    "saved_at_utc": _canon_utc_now_iso(),
                    "fallback_end_of_training_save": True,
                },
                ckpt_best_path,
            )

        # 11. Build the NS03 archive bytes from the EMA shadow
        archive_sha = ""
        archive_bytes = 0
        payload_0bin_sha = ""
        payload_0bin_bytes = 0
        archive_zip_path = args.output_dir / "archive.zip"
        if not args.skip_archive_build:
            print(f"[full] building archive from {ckpt_best_path} ...")
            ema_state = torch.load(
                ckpt_best_path, map_location="cpu", weights_only=False
            )
            # Rebuild the substrate from cfg + load EMA shadow weights
            rebuilt = NSCS03JointCodecSubstrate(cfg).to(device).eval()
            rebuilt.load_state_dict(ema_state["state_dict"])

            # Encode REAL pair latents from the trained substrate (hard-round
            # at archive build time per the deterministic inflate path).
            main_latents, hyper_latents = _encode_latents_for_archive(
                rebuilt, pair_tensor
            )

            sections = _extract_module_state_dicts(rebuilt)
            meta: dict[str, Any] = {
                "config": asdict(cfg),
                "lambda_R": args.lambda_R,
                "lambda_R_warmup_frac": args.lambda_R_warmup_frac,
                "ema_decay": args.ema_decay,
                "num_pairs": int(main_latents.shape[0]),
                # Catalog #210 forensic provenance fields
                "license_tags": "MIT",
                "dataset_provenance": "upstream/videos/0.mkv (contest-pinned)",
                "distillation_version": "nscs03_full_main_v1",
                "random_seed": args.seed,
                "basis_sha256": "",  # filled below once bytes assembled
                "num_frames_used": int(main_latents.shape[0] * 2),
            }
            bin_bytes = pack_archive(
                encoder_state_dict=sections["encoder"],
                decoder_state_dict=sections["decoder"],
                hyper_analysis_state_dict=sections["hyper_analysis"],
                hyper_synthesis_state_dict=sections["hyper_synthesis"],
                entropy_state_dict=sections["entropy"],
                main_latents=main_latents,
                hyper_latents=hyper_latents,
                meta=meta,
            )
            (args.output_dir / "0.bin").write_bytes(bin_bytes)
            payload_0bin_sha = _canon_sha256_bytes(bin_bytes)
            payload_0bin_bytes = len(bin_bytes)
            print(
                f"[full] wrote 0.bin ({payload_0bin_bytes} bytes, "
                f"sha256={payload_0bin_sha})"
            )

            # Emit contest-compliant runtime alongside the bin
            submission_dir = args.output_dir / "submission"
            _write_runtime(submission_dir)
            (submission_dir / "0.bin").write_bytes(bin_bytes)
            _build_archive_zip(
                archive_zip_path,
                bin_bytes=bin_bytes,
            )
            archive_bytes = archive_zip_path.stat().st_size
            archive_sha = _canon_sha256_bytes(archive_zip_path.read_bytes())
            print(
                f"[full] wrote {archive_zip_path} "
                f"({archive_bytes} bytes, sha256={archive_sha})"
            )
            _stage(f"archive_built_bytes_{archive_bytes}")

        # 12. CUDA auth eval — canonical helper (Catalog #226 self-protect)
        auth_eval_result_path: Path | None = None
        auth_eval_alias_path: Path | None = None
        contest_cuda_score: float | None = None
        if not args.skip_auth_eval and archive_zip_path.is_file():
            print("[full] launching CUDA auth eval ...")
            auth_eval_result_path = (
                args.output_dir / "contest_auth_eval_cuda.json"
            )
            auth_eval_alias_path = args.output_dir / "auth_eval.json"
            auth_result = _canon_gate_auth_eval_call(
                args=args,
                archive_zip=archive_zip_path,
                inflate_sh=args.output_dir / "submission" / "inflate.sh",
                upstream_dir=args.upstream_dir,
                output_json=auth_eval_result_path,
                contest_auth_eval_script=CONTEST_AUTH_EVAL_SCRIPT,
                substrate_tag="nscs03",
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
                if auth_eval_result_path.is_file():
                    shutil.copy2(auth_eval_result_path, auth_eval_alias_path)
            _stage("auth_eval_cuda_done")

        # 13. Continual-learning posterior update (Catalog #128 atomic)
        if contest_cuda_score is not None and archive_sha:
            try:
                from tac.continual_learning import (
                    ContestResult,
                    posterior_update_locked,
                )

                _detected_substrate = _canon_detect_hardware_substrate(
                    axis="cuda",
                    substrate_tag="nscs03",
                    provenance_path=args.output_dir / "provenance.json",
                    env_var_candidates=("NSCS03_GPU", "MODAL_GPU"),
                )
                result = ContestResult(
                    axis="cuda",
                    hardware_substrate=_detected_substrate,
                    architecture_class="lane_nscs03_end_to_end_balle_joint_codec_20260515",
                    score_value=contest_cuda_score,
                    evidence_tag="[contest-CUDA]",
                    archive_sha256=archive_sha,
                    archive_bytes=archive_bytes,
                    notes=(
                        f"NSCS03 end-to-end Ballé joint codec first-anchor dispatch; "
                        f"epochs={args.epochs} lambda_R={args.lambda_R}"
                    ),
                    observed_at_utc=_canon_utc_now_iso(),
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
                os.environ.get("NSCS03_ACTUAL_COST_USD"),
                field_name="NSCS03_ACTUAL_COST_USD",
            )
        except ValueError as exc:
            actual_cost_usd = None
            cost_band_anchor_skip_reason = (
                f"invalid_NSCS03_ACTUAL_COST_USD:{exc}"
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
                        f"nscs03_{_canon_utc_now_iso()}",
                        "--trainer",
                        "experiments/train_substrate_nscs03_end_to_end_balle_joint_codec.py",
                        "--platform",
                        os.environ.get("NSCS03_PLATFORM", "modal"),
                        "--gpu",
                        os.environ.get("NSCS03_GPU", "A100"),
                        "--epochs", str(args.epochs),
                        "--batch-size", str(args.batch_size),
                        "--actual-wall-clock-sec", str(train_elapsed_sec),
                        "--actual-cost-usd", str(actual_cost_usd),
                        "--notes",
                        "NSCS03 end-to-end Ballé first-anchor dispatch",
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
                    f"[full] cost-band anchor append failed (non-fatal): "
                    f"{exc}",
                    file=sys.stderr,
                )
        else:
            if actual_cost_usd is None and cost_band_anchor_skip_reason is None:
                cost_band_anchor_skip_reason = "missing_NSCS03_ACTUAL_COST_USD"
            elif not COST_BAND_TOOL.is_file():
                cost_band_anchor_skip_reason = "cost_band_tool_missing"
            else:
                cost_band_anchor_skip_reason = "nonpositive_train_elapsed_sec"

        # 15. Provenance manifest
        provenance = {
            "schema": "nscs03_full_provenance_v1",
            "generated_at": _canon_utc_now_iso(),
            "from_state_hash": "regen_per_session_below",
            "git_head": _canon_git_head_sha(REPO_ROOT),
            "trainer": "experiments/train_substrate_nscs03_end_to_end_balle_joint_codec.py",
            "lane_id": "lane_nscs03_end_to_end_balle_joint_codec_20260515",
            "args": {
                k: (str(v) if isinstance(v, Path) else v)
                for k, v in vars(args).items()
            },
            "pytorch_version": _canon_torch_version_string(),
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
            "payload_0bin_sha256": payload_0bin_sha,
            "payload_0bin_bytes": payload_0bin_bytes,
            "auth_eval_cuda_score": contest_cuda_score,
            "auth_eval_json_path": (
                str(auth_eval_result_path) if auth_eval_result_path else None
            ),
            "auth_eval_alias_path": (
                str(auth_eval_alias_path) if auth_eval_alias_path else None
            ),
            "exact_eval_packet": {
                "archive_path": (
                    str(archive_zip_path) if archive_zip_path.is_file() else None
                ),
                "archive_sha256": archive_sha,
                "archive_bytes": archive_bytes,
                "inflate_sh_path": (
                    str(args.output_dir / "submission" / "inflate.sh")
                    if (args.output_dir / "submission" / "inflate.sh").is_file()
                    else None
                ),
                "payload_0bin_path": (
                    str(args.output_dir / "0.bin")
                    if (args.output_dir / "0.bin").is_file()
                    else None
                ),
                "payload_0bin_sha256": payload_0bin_sha,
                "payload_0bin_bytes": payload_0bin_bytes,
                "score_axis_tag": (
                    "[contest-CUDA]" if contest_cuda_score is not None else None
                ),
                "score_claim": contest_cuda_score is not None,
            },
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
        }
        (args.output_dir / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"[full] wrote {args.output_dir / 'provenance.json'}")
        return 0

    finally:
        unpatch_upstream_yuv6(yuv6_token)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.smoke:
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":
    sys.exit(main())
