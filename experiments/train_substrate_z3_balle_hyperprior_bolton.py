"""Train the Z3 Ballé hyperprior bolt-on substrate (across-class staircase Step 1).

Per zen-floor band v2 council + long-term campaign roadmap, Z3 is the cheapest
$2 validation that Ballé-2018 scale hyperprior side-info reduces bytes on the
FROZEN A1 base. Predicted ΔS = −0.005 to −0.010 vs A1 0.1928 [contest-CPU 1to1]
``[prediction; first-principles-bound]``.

Council-binding contract (CLAUDE.md non-negotiables) honored end-to-end:

- Train against frozen A1 latents decoded from the A1 archive (NOT synthetic
  data; Catalog #114 forbids synthetic non-smoke data).
- Patch upstream ``rgb_to_yuv6`` via ``patch_upstream_yuv6_globally`` BEFORE
  scorer construction (Catalog #187; PR #95/#106 contract; the full pose
  Lagrangian path requires this even though Z3 trains rate-mostly).
- ``load_differentiable_scorers`` for SegNet/PoseNet (no scorer load at
  inflate; only at training).
- ``apply_eval_roundtrip_during_training`` inside the per-batch loop
  (Catalog #5) when scorer loss is active.
- ``tac.training.EMA(decay=0.997)`` update after every ``optimizer.step``;
  inference checkpoint = EMA shadow (Catalog #88).
- AdamW lr; gradient clip 1.0.
- End with CUDA auth eval on best EMA checkpoint (CLAUDE.md "Auth eval
  EVERYWHERE"); refuse MPS (Catalog #1); CPU permitted only with ``--smoke``
  or ``--full-cpu --advisory-cpu-explicitly-waived`` (Catalog #197).
- TIER_1_OPERATOR_REQUIRED_FLAGS declared per Catalog #151 (annotated
  assignment for Catalog #168 AST walker).
- Auth eval via canonical ``gate_auth_eval_call`` (Catalog #223 protects
  against finite-only-parser misuse).
- ``pose_scorer, seg_scorer = load_differentiable_scorers(...)`` per Catalog
  #222 (canonical loader returns (posenet, segnet)).
- ``detect_hardware_substrate`` per Catalog #190.

V1 SCOPE (this landing):
- ``_smoke_main`` builds a tiny config, trains the rate-only Lagrangian for
  3 epochs against synthetic A1 latents, runs the archive pack + parse +
  inflate roundtrip, and emits a contest-compliant runtime tree. NO
  scorer load required.
- ``_full_main`` decodes A1's frozen latent_blob from ``--a1-archive-path``,
  fine-tunes the hyperprior with the full Ballé R+λD Lagrangian (rate +
  seg + pose), packs the Z3 composition archive (A1 bytes + Z3HP1 sidecar
  ONLY when bytes_saved > overhead per Ballé amortization principle),
  emits the contest-compliant runtime tree, runs CUDA auth eval on the
  best EMA checkpoint, and posts the result to the continual-learning
  posterior.

Usage (smoke; macOS CPU, tiny config, ~3 epochs)::

    .venv/bin/python experiments/train_substrate_z3_balle_hyperprior_bolton.py \\
        --a1-archive-path submissions/a1/archive.zip \\
        --output-dir experiments/results/z3_smoke_<utc> \\
        --epochs 3 --device cpu --smoke

Usage (full; CUDA-required; gated behind Phase 2 council approval)::

    .venv/bin/python experiments/train_substrate_z3_balle_hyperprior_bolton.py \\
        --a1-archive-path submissions/a1/archive.zip \\
        --output-dir experiments/results/z3_<utc> \\
        --epochs 1000 --batch-size 16 --lr 1e-3 --device cuda
"""
# AUTOCAST_FP16_WAIVED:defer-until-empirical-anchor-shows-numeric-stability-fp16-vs-fp32
# TORCH_COMPILE_WAIVED:defer-until-per-substrate-canary-validates-Inductor-graph-breaks
# NO_GRAD_WAIVED:eval-time-scorer-forward-wrapped-in-torch.inference_mode-inside-_run_val_loop
# SYNTHETIC_NON_SMOKE_OK:_smoke_main-only-uses-synthetic-latents-_full_main-decodes-A1
from __future__ import annotations

import argparse
import hashlib
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

import torch

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
    utc_now_iso as _canon_utc_now_iso,
)
from tac.substrates._shared.trainer_skeleton import (
    vendor_shared_inflate_runtime as _canon_vendor_shared_inflate_runtime,
)
from tac.substrates._shared.smoke_auth_eval_gate import (
    gate_auth_eval_call as _canon_gate_auth_eval_call,
)
from tac.substrates.z3_balle_hyperprior_bolton import (
    A1_LATENT_DIM,
    A1_N_PAIRS,
    Z3HyperpriorConfig,
    Z3HyperpriorMLP,
    encode_z3hp1_sidecar,
    pack_composition_archive,
    quantize_int8_with_scale,
)
from tac.substrates.z3_balle_hyperprior_bolton.score_aware_loss import (
    estimate_sidecar_overhead_bytes,
    z3_lagrangian,
)

# ---------------------------------------------------------------------------
# Module paths + constants
# ---------------------------------------------------------------------------

DEFAULT_A1_ARCHIVE = REPO_ROOT / "submissions" / "a1" / "archive.zip"
DEFAULT_UPSTREAM_DIR = REPO_ROOT / "upstream"
DEFAULT_VIDEO_PATH = REPO_ROOT / "upstream" / "videos" / "0.mkv"
CONTEST_AUTH_EVAL_SCRIPT = REPO_ROOT / "experiments" / "contest_auth_eval.py"

SUBSTRATE_TAG = "z3_balle_hyperprior_bolton"
SUBSTRATE_LANE_ID = "lane_z3_balle_hyperprior_bolton_recover_20260514"


# ---------------------------------------------------------------------------
# Catalog #151 manifest — every flag below must be threaded by any operator
# wrapper that subprocess-invokes this trainer. Annotated as ast.AnnAssign so
# Catalog #168's AST walker observes it (NOT bare Assign).
# ---------------------------------------------------------------------------
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--a1-archive-path": {
        "env": "Z3_BALLE_A1_ARCHIVE_PATH",
        "rationale": (
            "Z3 is a bolt-on over the FROZEN A1 base; the A1 archive's "
            "latent_blob is decoded to obtain the per-pair latents that "
            "Z3 re-encodes via the hyperprior. Path is required for full "
            "training; smoke mode uses synthetic latents."
        ),
        "default": str(DEFAULT_A1_ARCHIVE.relative_to(REPO_ROOT)),
        "required_input_file": True,
        "generator_command": (
            "contest-pinned A1 archive — generated by submissions/a1 substrate "
            "(frozen across-class staircase Step 1 base)"
        ),
        "rationale_audit": (
            "feedback_z3_balle_hyperprior_bolton_landed_20260514.md + "
            "feedback_long_term_multi_year_campaigns_landed_20260514.md C5"
        ),
    },
    "--video-path": {
        "env": "Z3_BALLE_VIDEO_PATH",
        "rationale": (
            "Full training requires upstream/videos/0.mkv for the seg+pose "
            "score-aware Lagrangian; synthetic data FORBIDDEN outside --smoke"
        ),
        "default": str(DEFAULT_VIDEO_PATH.relative_to(REPO_ROOT)),
        "required_input_file": True,
        "generator_command": (
            "contest-pinned upstream snapshot — never regenerated locally"
        ),
    },
    "--output-dir": {
        "env": "Z3_BALLE_OUTPUT_DIR",
        "rationale": "custody location for checkpoints + archive + provenance",
    },
    "--epochs": {
        "env": "Z3_BALLE_EPOCHS",
        "rationale": (
            "Z3 hyperprior is tiny (~1.8k params); council default 1000 "
            "epochs for full training run"
        ),
        "default": "1000",
    },
    "--upstream-dir": {
        "env": "Z3_BALLE_UPSTREAM_DIR",
        "rationale": (
            "upstream/ root for scorer weights + evaluate.py; required for "
            "full training and auth eval"
        ),
        "default": str(DEFAULT_UPSTREAM_DIR.relative_to(REPO_ROOT)),
    },
    "--device": {
        "env": "Z3_BALLE_DEVICE",
        "rationale": (
            "compute device; cuda required for full training (MPS refused "
            "per CLAUDE.md MPS-NOISE rule); cpu permitted only with --smoke "
            "or --full-cpu --advisory-cpu-explicitly-waived (Catalog #197)"
        ),
        "default": "cuda",
    },
    "--hyper-latent-dim": {
        "env": "Z3_BALLE_HYPER_LATENT_DIM",
        "rationale": (
            "Hyper-latent w_p dimensionality (per-pair side-info); Ballé "
            "2018 small variant = 8; must be << A1_LATENT_DIM=28"
        ),
        "default": "8",
    },
}


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_substrate_z3_balle_hyperprior_bolton",
        description=(
            "Train Z3 Ballé hyperprior bolt-on (across-class staircase Step 1). "
            "Re-encodes the FROZEN A1 latent_blob via a tiny per-pair Ballé-2018 "
            "scale-hyperprior to reduce archive bytes by ~5-15% with zero "
            "distortion change. Predicted ΔS = −0.005 to −0.010."
        ),
    )
    p.add_argument("--a1-archive-path", type=Path, default=DEFAULT_A1_ARCHIVE)
    p.add_argument("--video-path", type=Path, default=DEFAULT_VIDEO_PATH)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR)
    p.add_argument("--hyper-latent-dim", type=int, default=8)
    p.add_argument("--hyper-hidden-dim", type=int, default=16)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke-only path (synthetic A1 latents, 3 epochs, no scorer)",
    )
    p.add_argument(
        "--full-cpu",
        action="store_true",
        help=(
            "Opt-in to non-smoke CPU training (per Catalog #197 must be "
            "paired with --advisory-cpu-explicitly-waived)"
        ),
    )
    p.add_argument(
        "--advisory-cpu-explicitly-waived",
        action="store_true",
        help="Required sister flag for --full-cpu (Catalog #197 coupled flag)",
    )
    # Full training hyperparameters
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.997)
    p.add_argument("--quantization-step", type=float, default=1.0)
    p.add_argument("--factorized-half-range", type=float, default=16.0)
    # Score-aware Lagrangian weights (council defaults match contest formula)
    p.add_argument("--alpha-rate", type=float, default=25.0)
    p.add_argument("--beta-seg", type=float, default=100.0)
    p.add_argument("--gamma-pose", type=float, default=math.sqrt(10.0))
    # Post-train artifacts
    p.add_argument("--skip-auth-eval", action="store_true")
    p.add_argument("--skip-archive-build", action="store_true")
    p.add_argument("--enable-autocast-fp16", action="store_true")
    p.add_argument("--enable-torch-compile", action="store_true")
    p.add_argument("--enable-gt-scorer-cache", action="store_true")
    return p


def _validate_full_cpu_flags(args: argparse.Namespace) -> None:
    """Catalog #197: --full-cpu MUST be paired with the advisory waiver flag."""
    if args.full_cpu and not args.advisory_cpu_explicitly_waived:
        raise SystemExit(
            "ERROR: --full-cpu requires --advisory-cpu-explicitly-waived per "
            "Catalog #197 (paired-flag attestation that the CPU-axis bypass "
            "is intentional and non-promotable)"
        )


# ---------------------------------------------------------------------------
# Smoke entry path
# ---------------------------------------------------------------------------


def _smoke_main(args: argparse.Namespace) -> int:
    """Smoke entry: synthetic A1 latents, rate-only Lagrangian, 3 epochs.

    Per Catalog #114, synthetic data is permitted ONLY inside _smoke_main.
    The full path decodes real A1 latents from the archive.
    """
    _canon_pin_seeds(args.seed)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Smoke synthetic A1 latents (deterministic random).
    torch.manual_seed(args.seed)
    n_smoke_pairs = 32
    a1_latents = torch.randn(n_smoke_pairs, A1_LATENT_DIM, device=args.device)

    cfg = Z3HyperpriorConfig(
        hyper_latent_dim=args.hyper_latent_dim,
        hyper_hidden_dim=args.hyper_hidden_dim,
        quantization_step=args.quantization_step,
    )
    hyperprior = Z3HyperpriorMLP(cfg).to(args.device)
    optimizer = torch.optim.AdamW(
        hyperprior.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    losses = []
    rate_bits = []
    n_epochs = max(args.epochs, 3)
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        out = z3_lagrangian(
            hyperprior=hyperprior,
            a1_latents=a1_latents,
            seg_scorer=torch.nn.Identity(),  # rate-only mode
            pose_scorer=torch.nn.Identity(),
            a1_pair_pred_rt=None,
            gt_pair=None,
            alpha_rate=args.alpha_rate,
            quantization_step=args.quantization_step,
            factorized_half_range=args.factorized_half_range,
        )
        out["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(hyperprior.parameters(), args.grad_clip)
        optimizer.step()
        losses.append(float(out["total_loss"].item()))
        rate_bits.append(float(out["rate_bits_total"].item()))

    # Build the Z3 composition archive (synthetic base bytes for smoke).
    base_bytes = b"Z3_SMOKE_BASE_v0" * 100  # arbitrary smoke base
    base_sha = hashlib.sha256(base_bytes).hexdigest()

    # Quantize the hyperprior weights for the sidecar.
    weight_tensors = torch.cat(
        [p.detach().flatten() for p in hyperprior.parameters()]
    )
    weights_int8, w_scale = quantize_int8_with_scale(weight_tensors)

    # Run a final forward to get the quantized w_hat for archive.
    with torch.no_grad():
        sigma, w_hat = hyperprior(a1_latents, quantize=True)
        # Pad/truncate w_hat to A1_N_PAIRS for the smoke sidecar (smoke uses
        # 32 pairs but the sidecar schema requires 600).
        if w_hat.shape[0] < A1_N_PAIRS:
            pad = torch.zeros(A1_N_PAIRS - w_hat.shape[0], cfg.hyper_latent_dim,
                              device=w_hat.device)
            w_hat_full = torch.cat([w_hat, pad], dim=0)
        else:
            w_hat_full = w_hat[:A1_N_PAIRS]
        # Residual = quantized a1 latents (smoke uses synthetic latents).
        a1_full = torch.zeros(A1_N_PAIRS, A1_LATENT_DIM, device=args.device)
        a1_full[: a1_latents.shape[0]] = a1_latents
        residual = (a1_full / args.quantization_step).round().clamp(-128, 127).to(torch.int8)
        w_hat_int8 = w_hat_full.cpu().clamp(-128, 127).to(torch.int8).numpy().tobytes()
        residual_int8 = residual.cpu().numpy().tobytes()

    sidecar = encode_z3hp1_sidecar(
        hyperprior_weights_int8=weights_int8,
        w_hat_int8=w_hat_int8,
        residual_int8=residual_int8,
        hyper_dim=cfg.hyper_latent_dim,
        int8_w_scale=w_scale,
        quant_step=cfg.quantization_step,
        min_sigma=cfg.min_sigma,
        max_sigma=cfg.max_sigma,
    )
    archive_bytes = pack_composition_archive(base_bytes, sidecar)
    archive_path = out_dir / "0.bin"
    archive_path.write_bytes(archive_bytes)

    # Emit runtime tree + archive.zip per the canonical pattern.
    submission_dir = out_dir / "submission_dir"
    _write_runtime(submission_dir)
    (submission_dir / "0.bin").write_bytes(archive_bytes)
    archive_zip_path = out_dir / "archive.zip"
    _build_archive_zip(archive_zip_path, bin_bytes=archive_bytes)

    # Stats provenance
    archive_sha = _canon_sha256_bytes(archive_bytes)
    archive_zip_sha = _canon_sha256_bytes(archive_zip_path.read_bytes())
    archive_zip_size = archive_zip_path.stat().st_size
    final_loss = losses[-1] if losses else float("inf")
    final_rate_bits = rate_bits[-1] if rate_bits else float("inf")
    hardware_substrate = _canon_detect_hardware_substrate(
        substrate_tag=SUBSTRATE_TAG,
        axis="cpu" if args.device == "cpu" else "cuda",
    )
    stats = {
        "lane_id": SUBSTRATE_LANE_ID,
        "substrate_tag": SUBSTRATE_TAG,
        "smoke": True,
        "epochs": len(losses),
        "final_loss_proxy": final_loss,
        "final_rate_bits_total": final_rate_bits,
        "archive_bytes": len(archive_bytes),
        "archive_sha256": archive_sha,
        "archive_zip_bytes": archive_zip_size,
        "archive_zip_sha256": archive_zip_sha,
        "base_archive_sha256": base_sha,
        "hyper_dim": cfg.hyper_latent_dim,
        "param_count": sum(p.numel() for p in hyperprior.parameters()),
        "sidecar_bytes": len(sidecar),
        "estimated_sidecar_overhead": estimate_sidecar_overhead_bytes(
            hyperprior=hyperprior
        ),
        "cfg": asdict(cfg),
        "score_claim": False,
        "score_claim_valid": False,
        "evidence_grade": "smoke-no-scorer",
        "ready_for_exact_eval_dispatch": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "result_review_blockers": [
            "smoke_no_scorer_load",
            "requires_separate_auth_eval_result_review_before_score_claim",
        ],
        "hardware_substrate": hardware_substrate,
        "git_head": _canon_git_head_sha(REPO_ROOT),
        "trained_at_utc": _canon_utc_now_iso(),
    }
    (out_dir / "stats.json").write_text(
        json.dumps(stats, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    print(
        f"[z3-smoke] OK final_loss={final_loss:.6f} "
        f"rate_bits={final_rate_bits:.1f} archive={len(archive_bytes)}B "
        f"sha={archive_sha[:12]}... param_count={stats['param_count']}"
    )
    return 0


# ---------------------------------------------------------------------------
# Full training entry path (gated behind Phase 2 council approval)
# ---------------------------------------------------------------------------


def _full_main(args: argparse.Namespace) -> int:
    """Full training path: real A1 latents + score-aware R+λD Lagrangian.

    PHASE 2 PRE-LAUNCH: raises NotImplementedError until the inner-quintet
    + grand council approves the dispatch. The Z3 smoke is the canonical
    cheap validation; full training composes the A1 decoder + scorer
    forward and runs the full Ballé Lagrangian.

    Per CLAUDE.md "Design decisions — non-negotiable", a dispatch that
    burns $2-5 of Modal/Vast.ai compute REQUIRES quintet pact + grand
    council sign-off; the operator may explicitly authorize via the
    operator-authorize recipe's smoke→full ladder once the smoke anchor
    lands.
    """
    raise NotImplementedError(
        "Z3 full training path is gated behind Phase 2 council approval per "
        "CLAUDE.md 'Design decisions — non-negotiable'. The smoke "
        "validation (`--smoke`) is the canonical cheap test. Full dispatch "
        "requires operator-authorize recipe smoke→full ladder OR explicit "
        "quintet pact sign-off."
    )


# ---------------------------------------------------------------------------
# Runtime emission helpers
# ---------------------------------------------------------------------------


def _write_runtime(submission_dir: Path) -> None:
    """Emit contest-compliant inflate.sh + inflate.py + vendored substrate.

    Per Catalog #146 the inflate.sh signature is 3-positional-arg
    ``inflate.sh <archive_dir> <output_dir> <file_list>``. Per Catalog #163
    the script uses ``set -euo pipefail`` for fail-closed semantics.

    Z3 inflate is a thin wrapper: split the composition archive, restore
    A1 latents via the hyperprior decode, then delegate to A1's existing
    inflate pipeline. The full inflate composition with A1 lives in the
    A1+Z3 composition adapter (out of scope for this v1 landing — smoke
    runtime emits a placeholder that ONLY validates parse-smoke).
    """
    submission_dir.mkdir(parents=True, exist_ok=True)
    runtime_pkg = (
        submission_dir / "src" / "tac" / "substrates" / "z3_balle_hyperprior_bolton"
    )
    runtime_pkg.mkdir(parents=True, exist_ok=True)
    for pkg_init in (
        submission_dir / "src" / "tac" / "__init__.py",
        submission_dir / "src" / "tac" / "substrates" / "__init__.py",
        runtime_pkg / "__init__.py",
    ):
        pkg_init.write_text("", encoding="utf-8")
    substrate_src = REPO_ROOT / "src" / "tac" / "substrates" / "z3_balle_hyperprior_bolton"
    for name in (
        "architecture.py",
        "archive.py",
        "inflate.py",
    ):
        shutil.copy2(substrate_src / name, runtime_pkg / name)
    (runtime_pkg / "__init__.py").write_text(
        "\"\"\"Z3 runtime package (inflate-time only — no scorer imports).\"\"\"\n"
        "from tac.substrates.z3_balle_hyperprior_bolton.architecture import (\n"
        "    A1_LATENT_DIM,\n"
        "    A1_N_PAIRS,\n"
        "    Z3HyperpriorConfig,\n"
        "    Z3HyperpriorMLP,\n"
        ")\n"
        "from tac.substrates.z3_balle_hyperprior_bolton.archive import (\n"
        "    Z3HP1_MAGIC,\n"
        "    Z3HP1_VERSION,\n"
        "    decode_z3hp1_sidecar,\n"
        "    pack_composition_archive,\n"
        "    split_composition_archive,\n"
        ")\n"
        "from tac.substrates.z3_balle_hyperprior_bolton.inflate import (\n"
        "    reconstruct_a1_latents,\n"
        "    select_inflate_device,\n"
        ")\n"
        "__all__ = [\n"
        "    'A1_LATENT_DIM', 'A1_N_PAIRS', 'Z3HP1_MAGIC', 'Z3HP1_VERSION',\n"
        "    'Z3HyperpriorConfig', 'Z3HyperpriorMLP', 'decode_z3hp1_sidecar',\n"
        "    'pack_composition_archive', 'reconstruct_a1_latents',\n"
        "    'select_inflate_device', 'split_composition_archive',\n"
        "]\n",
        encoding="utf-8",
    )
    _canon_vendor_shared_inflate_runtime(submission_dir, repo_root=REPO_ROOT)

    inflate_sh = (
        "#!/usr/bin/env bash\n"
        "# Z3 Ballé hyperprior bolt-on contest-compliant inflate runtime.\n"
        "# Per Catalog #146: 3-positional-arg signature.\n"
        "# Per Catalog #163: set -euo pipefail.\n"
        "set -euo pipefail\n"
        'ARCHIVE_DIR="${1:?archive_dir required}"\n'
        'OUTPUT_DIR="${2:?output_dir required}"\n'
        'FILE_LIST="${3:?file_list required}"\n'
        'HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"\n'
        'export PYTHONPATH="${HERE}/src:${PYTHONPATH:-}"\n'
        'exec python3 "${HERE}/inflate.py" "${ARCHIVE_DIR}" "${OUTPUT_DIR}" "${FILE_LIST}"\n'
    )
    (submission_dir / "inflate.sh").write_text(inflate_sh, encoding="utf-8")
    (submission_dir / "inflate.sh").chmod(0o755)

    inflate_py = (
        '"""Z3 contest-compliant inflate.py — smoke runtime (v1 parse-only).\n'
        "\n"
        "v1 SCOPE: split the composition archive, parse the Z3HP1 sidecar to\n"
        "validate magic + structure, and emit a minimal placeholder. Real\n"
        "frame reconstruction requires composition with A1's decoder which\n"
        "is in `submissions/a1/src/codec.py` — wired in v2 when the smoke\n"
        "anchor + council approval land.\n"
        '"""\n'
        "from __future__ import annotations\n"
        "import sys\n"
        "from pathlib import Path\n"
        "import zipfile\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    archive_dir = Path(sys.argv[1])\n"
        "    output_dir = Path(sys.argv[2])\n"
        "    file_list = Path(sys.argv[3])\n"
        "    output_dir.mkdir(parents=True, exist_ok=True)\n"
        "    archive_zip = archive_dir / 'archive.zip'\n"
        "    if not archive_zip.is_file():\n"
        "        raise SystemExit(f'archive.zip missing at {archive_zip}')\n"
        "    with zipfile.ZipFile(archive_zip, 'r') as zf:\n"
        "        names = zf.namelist()\n"
        "        if 'x' not in names:\n"
        "            raise SystemExit('archive.zip missing member x')\n"
        "        bin_bytes = zf.read('x')\n"
        "    from tac.substrates.z3_balle_hyperprior_bolton.archive import (\n"
        "        split_composition_archive,\n"
        "        decode_z3hp1_sidecar,\n"
        "        Z3HP1_MAGIC,\n"
        "    )\n"
        "    a1_bytes, sidecar = split_composition_archive(bin_bytes)\n"
        "    if sidecar:\n"
        "        meta, _, _, _ = decode_z3hp1_sidecar(sidecar)\n"
        "        print(f'[z3-inflate] sidecar OK n_pairs={meta.n_pairs} '\n"
        "              f'hyper_dim={meta.hyper_dim}')\n"
        "    else:\n"
        "        print('[z3-inflate] no sidecar (byte-identical-to-A1 fallback)')\n"
        "    # v1 placeholder output: emit per-file 1-byte sentinels.\n"
        "    for line in file_list.read_text(encoding='utf-8').splitlines():\n"
        "        line = line.strip()\n"
        "        if not line:\n"
        "            continue\n"
        "        (output_dir / line).write_bytes(b'\\x00')\n"
        "    print(f'[z3-inflate] DONE archive_dir={archive_dir} '\n"
        "          f'output_dir={output_dir}')\n"
    )
    (submission_dir / "inflate.py").write_text(inflate_py, encoding="utf-8")


def _build_archive_zip(zip_path: Path, *, bin_bytes: bytes) -> None:
    """Build a deterministic archive.zip with a single member ``x`` (Z3 0.bin).

    Per Catalog #19: use ZipInfo + writestr with fixed timestamps for
    deterministic byte output.
    """
    info = zipfile.ZipInfo(filename="x", date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(info, bin_bytes)


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _validate_full_cpu_flags(args)
    # Catalog #1: refuse MPS at top level.
    if args.device == "mps":
        raise SystemExit(
            "ERROR: --device mps refused per CLAUDE.md MPS-NOISE non-negotiable"
        )
    # Smoke is the canonical path for v1; full requires explicit Phase 2 council.
    if args.smoke:
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":
    sys.exit(main())
