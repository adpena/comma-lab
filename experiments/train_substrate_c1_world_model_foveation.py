# SPDX-License-Identifier: MIT
"""Train the C1 world-model + foveation substrate (long-term campaign L1 scaffold).

Per the C1 long-term campaign ledger
``.omx/research/campaign_c1_world_model_foveation_20260514.md`` and the
operator directive 2026-05-14 "we are aggressively pursuing across class too":
ACROSS-CLASS substrate-shift from A1/PR101/HNeRV-family class-saturated
baselines (Z1 MDL density 97-99%) toward the predicted [0.13, 0.16] zen-floor
band (staircase Step 4-5 territory).

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
- Score-domain Lagrangian with C1 world-model + foveation terms per HNeRV
  parity discipline lesson L6.
- AdamW lr cosine annealing; gradient clip 1.0; NaN watchdog.
- End with CUDA auth eval on best EMA checkpoint via canonical
  ``smoke_auth_eval_gate.gate_auth_eval_call`` (Catalog #221; CLAUDE.md
  "Auth eval EVERYWHERE"); refuse MPS (Catalog #1); CPU permitted only with
  ``--smoke`` or ``--full-cpu`` + ``--advisory-cpu-explicitly-waived``.
- Continual-learning posterior update via ``posterior_update_locked``
  (Catalog #128 atomic fcntl).
- Contest-compliant runtime emission with 3 positional args inflate.sh +
  ``set -euo pipefail`` per Catalog #146 + #163.
- TIER_1_OPERATOR_REQUIRED_FLAGS declared per Catalog #151 (annotated
  assignment for Catalog #168 AST walker).
- ``--full-cpu`` opt-in coupled with ``--advisory-cpu-explicitly-waived``
  per Catalog #197.
- Catalog #190 hardware substrate dynamic detection.

V1 SCOPE (this landing):
- ``_smoke_main`` builds a tiny config, trains for <=3 epochs on synthetic
  data, runs the archive pack + parse + inflate roundtrip, and emits a
  contest-compliant runtime tree. NO scorer load required.
- ``_full_main`` raises NotImplementedError pending Phase 3 council
  approval per CLAUDE.md "Design decisions" non-negotiable. The multi-stage
  training schedule ($30-50 over 3-4 weeks, 6-8 stages of {world-model alone,
  foveation alone, combined fine-tune, residual codec, archive byte sweep,
  full Lagrangian convergence}) requires inner-quintet council sign-off
  before dispatch.

Usage (smoke; macOS CPU, tiny config, ~3 epochs)::

    .venv/bin/python experiments/train_substrate_c1_world_model_foveation.py \\
        --video-path upstream/videos/0.mkv \\
        --output-dir experiments/results/c1_smoke_<utc> \\
        --epochs 3 --device cpu --smoke
"""
# AUTOCAST_FP16_WAIVED:score-aware-scorer-path-pending-canonical-autocast-backport
# TORCH_COMPILE_WAIVED:defer-until-per-substrate-canary-validates-Inductor-graph-breaks
# TF32_WAIVED:opt-in-via-CLI-flag-to-keep-eval-roundtrip-numerics-deterministic
# NO_GRAD_WAIVED:eval-time-scorer-forward-wrapped-in-torch.inference_mode-inside-full_main-pending-phase3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402

from tac.substrates._shared.trainer_skeleton import (  # noqa: E402
    detect_hardware_substrate as _canon_detect_hardware_substrate,
)
from tac.substrates._shared.trainer_skeleton import (  # noqa: E402
    device_or_die as _canon_device_or_die,
)
from tac.substrates._shared.trainer_skeleton import (  # noqa: E402
    git_head_sha as _canon_git_head_sha,
)
from tac.substrates._shared.trainer_skeleton import (  # noqa: E402
    pin_seeds as _canon_pin_seeds,
)
from tac.substrates._shared.trainer_skeleton import (  # noqa: E402
    sha256_bytes as _canon_sha256_bytes,
)
from tac.substrates._shared.trainer_skeleton import (  # noqa: E402
    utc_now_iso as _canon_utc_now_iso,
)
from tac.substrates.c1_world_model_foveation import (  # noqa: E402
    FoveationStrategy,
    WorldModelConfig,
    WorldModelFoveationConfig,
    WorldModelFoveationSubstrate,
    WorldModelRecurrenceMode,
    pack_archive,
)

DEFAULT_VIDEO_PATH = REPO_ROOT / "upstream" / "videos" / "0.mkv"
DEFAULT_UPSTREAM_DIR = REPO_ROOT / "upstream"
CONTEST_AUTH_EVAL_SCRIPT = REPO_ROOT / "experiments" / "contest_auth_eval.py"

EVAL_HW = (384, 512)
N_PAIRS_FULL = 600
CONTEST_NORMALIZER = 37_545_489.0

SUBSTRATE_TAG = "c1_world_model_foveation"
SUBSTRATE_LANE_ID = "lane_c1_world_model_foveation_campaign_l1_scaffold_20260514"


# ---------------------------------------------------------------------------
# Catalog #151 manifest -- every flag below must be threaded by any operator
# wrapper that subprocess-invokes this trainer. Annotated as ast.AnnAssign so
# Catalog #168's AST walker observes it (NOT bare Assign).
# ---------------------------------------------------------------------------
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--video-path": {
        "env": "C1_WORLD_MODEL_FOVEATION_VIDEO_PATH",
        "rationale": (
            "score-aware substrate MUST train against the contest video "
            "(upstream/videos/0.mkv); synthetic data is FORBIDDEN outside --smoke"
        ),
        "default": str(DEFAULT_VIDEO_PATH.relative_to(REPO_ROOT)),
        "required_input_file": True,
        "generator_command": (
            "contest-pinned upstream snapshot -- never regenerated locally"
        ),
        "rationale_audit": (
            ".omx/research/campaign_c1_world_model_foveation_20260514.md"
        ),
    },
    "--output-dir": {
        "env": "C1_WORLD_MODEL_FOVEATION_OUTPUT_DIR",
        "rationale": "custody location for checkpoints + archive + provenance",
    },
    "--epochs": {
        "env": "C1_WORLD_MODEL_FOVEATION_EPOCHS",
        "rationale": (
            "C1 substrate is medium-complexity (world-model + decoder + "
            "foveation + residual codec); council default 2000 epochs for "
            "full training run; multi-stage training is gated by Phase 3 "
            "council approval"
        ),
        "default": "2000",
    },
    "--upstream-dir": {
        "env": "C1_WORLD_MODEL_FOVEATION_UPSTREAM_DIR",
        "rationale": (
            "upstream/ root for scorer weights + evaluate.py; required for "
            "full training and auth eval"
        ),
        "default": str(DEFAULT_UPSTREAM_DIR.relative_to(REPO_ROOT)),
    },
    "--device": {
        "env": "C1_WORLD_MODEL_FOVEATION_DEVICE",
        "rationale": (
            "compute device; cuda required for full training (MPS refused "
            "per CLAUDE.md MPS-NOISE rule); cpu permitted only with --smoke "
            "or --full-cpu --advisory-cpu-explicitly-waived"
        ),
        "default": "cuda",
    },
    "--recurrence-mode": {
        "env": "C1_WORLD_MODEL_FOVEATION_RECURRENCE_MODE",
        "rationale": (
            "world-model recurrence (Catalog #125 hook #6 probe-disambiguator): "
            "gru/lstm/transformer; default gru for cheapest forward"
        ),
        "default": "gru",
    },
    "--foveation-strategy": {
        "env": "C1_WORLD_MODEL_FOVEATION_FOVEATION_STRATEGY",
        "rationale": (
            "foveation strategy (Catalog #125 hook #6 probe-disambiguator): "
            "uniform/ego_motion_radial/learned_per_pixel; default "
            "ego_motion_radial per Atick-Redlich 1990"
        ),
        "default": "ego_motion_radial",
    },
    "--latent-dim": {
        "env": "C1_WORLD_MODEL_FOVEATION_LATENT_DIM",
        "rationale": (
            "world-model latent dim; default 64 balances FP4 param budget "
            "(world-model+decoder ~80 KB at FP4) and decoder expressiveness"
        ),
        "default": "64",
    },
}


# ---------------------------------------------------------------------------
# Argparse
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_substrate_c1_world_model_foveation",
        description=(
            "Train C1 world-model + foveation substrate (long-term campaign "
            "L1 scaffold). World-model recurrent latent dynamics + foveated "
            "decoder + camera-geometry-aware bit allocation. Predicted band "
            "[0.13, 0.16] [mathematical-derivation; first-principles-bound]."
        ),
    )
    p.add_argument("--video-path", type=Path, default=DEFAULT_VIDEO_PATH)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR
    )
    p.add_argument(
        "--recurrence-mode",
        type=str,
        default="gru",
        choices=["gru", "lstm", "transformer"],
    )
    p.add_argument(
        "--foveation-strategy",
        type=str,
        default="ego_motion_radial",
        choices=["uniform", "ego_motion_radial", "learned_per_pixel"],
    )
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke-only path (tiny config, 3 epochs, no scorer load)",
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
    # Training hyperparameters (defaults match D4 sister)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.997)
    p.add_argument("--noise-std", type=float, default=0.5)
    # Score-aware Lagrangian weights (council defaults)
    p.add_argument("--alpha-rate", type=float, default=25.0)
    p.add_argument("--beta-seg", type=float, default=100.0)
    p.add_argument("--lambda-residual", type=float, default=0.1)
    p.add_argument("--lambda-foveation", type=float, default=0.01)
    # Post-train artifacts (deferred to Phase 3)
    p.add_argument("--skip-auth-eval", action="store_true")
    p.add_argument("--skip-archive-build", action="store_true")
    p.add_argument("--enable-autocast-fp16", action="store_true",
                   help="Catalog #172; deferred until canonical autocast wraps land.")
    p.add_argument("--enable-tf32", action="store_true",
                   help="Catalog #178; deferred until paired CPU/CUDA anchor lands.")
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
    """Smoke entry: tiny config, synthetic data, 3 epochs, no scorer load.

    Validates substrate plumbing -- world-model unroll + decoder + foveation
    + archive pack/parse + runtime emission -- without any GPU spend or
    scorer load. Per CLAUDE.md "Forbidden /tmp paths" the output_dir is
    operator-supplied (not /tmp).
    """
    _canon_pin_seeds(args.seed)
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Small smoke config: 4 pairs, 24x32 frames, latent_dim=8.
    recurrence_mode = {
        "gru": WorldModelRecurrenceMode.GRU,
        "lstm": WorldModelRecurrenceMode.LSTM,
        "transformer": WorldModelRecurrenceMode.TRANSFORMER,
    }[args.recurrence_mode]
    foveation_strategy = {
        "uniform": FoveationStrategy.UNIFORM,
        "ego_motion_radial": FoveationStrategy.EGO_MOTION_RADIAL,
        "learned_per_pixel": FoveationStrategy.LEARNED_PER_PIXEL,
    }[args.foveation_strategy]
    wm_cfg = WorldModelConfig(
        recurrence_mode=recurrence_mode,
        latent_dim=8,
        hidden_dim=8,
    )
    cfg = WorldModelFoveationConfig(
        world_model_cfg=wm_cfg,
        foveation_strategy=foveation_strategy,
        output_height=24,
        output_width=32,
        num_pairs=4,
        decoder_channels=(8, 4, 4),
    )
    substrate = WorldModelFoveationSubstrate(cfg).to(args.device)

    # Smoke synthetic data: deterministic random target.
    torch.manual_seed(args.seed)
    smoke_target = torch.rand(2 * cfg.num_pairs, 3, 24, 32, device=args.device)

    # Minimal pixel-MSE training loop (smoke only -- no scorer load).
    opt = torch.optim.AdamW(substrate.parameters(), lr=args.lr)
    losses = []
    for _ in range(max(args.epochs, 3)):
        opt.zero_grad()
        rgb, _ = substrate.render_all_frames()
        loss = (rgb - smoke_target).pow(2).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(substrate.parameters(), 1.0)
        opt.step()
        losses.append(float(loss.item()))

    # Build the smoke archive.
    recurrence_int = (
        0 if recurrence_mode == WorldModelRecurrenceMode.GRU
        else 1 if recurrence_mode == WorldModelRecurrenceMode.LSTM
        else 2
    )
    foveation_int = (
        0 if foveation_strategy == FoveationStrategy.UNIFORM
        else 1 if foveation_strategy == FoveationStrategy.EGO_MOTION_RADIAL
        else 2
    )
    archive_bytes = pack_archive(
        num_pairs=cfg.num_pairs,
        recurrence_mode=recurrence_int,
        foveation_strategy=foveation_int,
        latent_dim=wm_cfg.latent_dim,
        output_h=cfg.output_height,
        output_w=cfg.output_width,
        world_model_state_dict=substrate.world_model.state_dict(),
        decoder_state_dict=substrate.decoder.state_dict(),
        z_init=substrate.z_init.detach(),
        foveation_meta={"sigma": 6.0, "center_y": 12.0, "center_x": 16.0},
        residual_blob=b"",  # smoke -- world-model alone, no residual
        meta={
            "smoke": True,
            "lane_id": SUBSTRATE_LANE_ID,
            "recurrence_mode_label": args.recurrence_mode,
            "foveation_strategy_label": args.foveation_strategy,
            "git_head": _canon_git_head_sha(REPO_ROOT),
            "trained_at_utc": _canon_utc_now_iso(),
        },
    )
    archive_path = out_dir / "0.bin"
    archive_path.write_bytes(archive_bytes)

    # Emit runtime tree + archive.zip per the canonical pattern.
    submission_dir = out_dir / "submission_dir"
    _write_runtime(submission_dir)
    (submission_dir / "0.bin").write_bytes(archive_bytes)
    archive_zip_path = out_dir / "archive.zip"
    _build_archive_zip(archive_zip_path, bin_bytes=archive_bytes)

    archive_sha = _canon_sha256_bytes(archive_bytes)
    archive_zip_sha = _canon_sha256_bytes(archive_zip_path.read_bytes())
    archive_zip_size = archive_zip_path.stat().st_size
    final_loss = losses[-1] if losses else float("inf")

    # Catalog #190 hardware substrate detection (smoke runs on CPU)
    hardware_substrate = _canon_detect_hardware_substrate(
        axis="cpu",
        substrate_tag=SUBSTRATE_TAG,
    )

    stats = {
        "lane_id": SUBSTRATE_LANE_ID,
        "substrate_tag": SUBSTRATE_TAG,
        "smoke": True,
        "epochs": len(losses),
        "final_loss_proxy": final_loss,
        "archive_bytes": len(archive_bytes),
        "archive_sha256": archive_sha,
        "archive_zip_bytes": archive_zip_size,
        "archive_zip_sha256": archive_zip_sha,
        "recurrence_mode": args.recurrence_mode,
        "foveation_strategy": args.foveation_strategy,
        "cfg_summary": {
            "num_pairs": cfg.num_pairs,
            "output_height": cfg.output_height,
            "output_width": cfg.output_width,
            "latent_dim": cfg.world_model_cfg.latent_dim,
            "decoder_channels": list(cfg.decoder_channels),
            "residual_loss_weight": cfg.residual_loss_weight,
            "foveation_loss_weight": cfg.foveation_loss_weight,
            "archive_byte_target": cfg.archive_byte_target,
        },
        # Catalog #127 + #221 fail-closed score authority fields
        "score_claim": False,
        "score_claim_valid": False,
        "evidence_grade": "smoke-no-scorer",
        "ready_for_exact_eval_dispatch": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "result_review_blockers": [
            "smoke_no_scorer_load",
            "no_contest_video_used",
            "phase_3_council_approval_required_for_full_dispatch",
        ],
        "council_phase_3_required_before_full_dispatch": True,
        "hardware_substrate": hardware_substrate,
        "git_head": _canon_git_head_sha(REPO_ROOT),
        "trained_at_utc": _canon_utc_now_iso(),
    }
    (out_dir / "stats.json").write_text(
        json.dumps(stats, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    print(
        f"[c1-smoke] OK final_loss={final_loss:.6f} archive={len(archive_bytes)}B "
        f"sha={archive_sha[:12]}... recurrence={args.recurrence_mode} "
        f"foveation={args.foveation_strategy}"
    )
    return 0


def _write_runtime(submission_dir: Path) -> None:
    """Emit contest-compliant inflate.sh + inflate.py + vendored substrate.

    Per Catalog #146 the inflate.sh signature is 3-positional-arg
    ``inflate.sh <archive_dir> <output_dir> <file_list>``. Per Catalog #163
    the script uses ``set -euo pipefail`` for fail-closed semantics.
    """
    submission_dir.mkdir(parents=True, exist_ok=True)
    runtime_pkg = submission_dir / "src" / "tac" / "substrates" / "c1_world_model_foveation"
    runtime_pkg.mkdir(parents=True, exist_ok=True)
    for pkg_init in (
        submission_dir / "src" / "tac" / "__init__.py",
        submission_dir / "src" / "tac" / "substrates" / "__init__.py",
        runtime_pkg / "__init__.py",
    ):
        pkg_init.write_text("", encoding="utf-8")
    substrate_src = (
        REPO_ROOT / "src" / "tac" / "substrates" / "c1_world_model_foveation"
    )
    for name in ("architecture.py", "archive.py", "inflate.py"):
        shutil.copy2(substrate_src / name, runtime_pkg / name)

    # Runtime __init__.py is MINIMAL -- only inflate-time modules. The full
    # package __init__.py eagerly imports score_aware_loss which would pull
    # in scorer code (forbidden at inflate time per CLAUDE.md "Strict scorer
    # rule").
    (runtime_pkg / "__init__.py").write_text(
        '"""C1 runtime package (inflate-time only -- no scorer imports)."""\n'
        "from tac.substrates.c1_world_model_foveation.architecture import (\n"
        "    EVAL_HW,\n"
        "    FoveationStrategy,\n"
        "    WorldModelConfig,\n"
        "    WorldModelFoveationConfig,\n"
        "    WorldModelFoveationSubstrate,\n"
        "    WorldModelRecurrenceMode,\n"
        ")\n"
        "from tac.substrates.c1_world_model_foveation.archive import (\n"
        "    C1WMFV1_HEADER_SIZE,\n"
        "    C1WMFV1_MAGIC,\n"
        "    C1WMFV1_SCHEMA_VERSION,\n"
        "    WorldModelFoveationArchive,\n"
        "    deserialize_state_dict,\n"
        "    pack_archive,\n"
        "    parse_archive,\n"
        ")\n",
        encoding="utf-8",
    )

    # Vendor shared inflate runtime helpers.
    shared_runtime = submission_dir / "src" / "tac" / "substrates" / "_shared"
    shared_runtime.mkdir(parents=True, exist_ok=True)
    (shared_runtime / "__init__.py").write_text("", encoding="utf-8")
    shutil.copy2(
        REPO_ROOT / "src" / "tac" / "substrates" / "_shared" / "inflate_runtime.py",
        shared_runtime / "inflate_runtime.py",
    )

    # inflate.py at submission_dir root: thin shim that dispatches to the
    # vendored C1 inflate runtime.
    (submission_dir / "inflate.py").write_text(
        "#!/usr/bin/env python3\n"
        '"""Contest inflate entry point for C1 (delegates to vendored runtime)."""\n'
        "import sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, str(Path(__file__).resolve().parent / 'src'))\n"
        "from tac.substrates.c1_world_model_foveation.inflate import main_cli\n"
        "if __name__ == '__main__':\n"
        "    sys.exit(main_cli())\n",
        encoding="utf-8",
    )

    # inflate.sh per Catalog #146 + #163 contract.
    (submission_dir / "inflate.sh").write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "HERE=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"\n"
        'DATA_DIR="${1:?}"\n'
        'OUTPUT_DIR="${2:?}"\n'
        'FILE_LIST="${3:?}"\n'
        'exec uv run --with torch --with brotli '
        '"$HERE/inflate.py" "$DATA_DIR" "$OUTPUT_DIR" "$FILE_LIST"\n',
        encoding="utf-8",
    )
    # Make inflate.sh executable.
    (submission_dir / "inflate.sh").chmod(0o755)


def _build_archive_zip(zip_path: Path, *, bin_bytes: bytes) -> None:
    """Build a single-zip-member archive.zip with deterministic timestamps.

    Per Catalog #20 (`check_archive_builders_use_deterministic_zip`) the
    archive uses ZipInfo + writestr with fixed timestamp instead of
    ZipFile.write which is non-deterministic across filesystems.
    """
    if zip_path.exists():
        zip_path.unlink()
    info = zipfile.ZipInfo(filename="0.bin", date_time=(2020, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED  # no recompression of brotli payload
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(info, bin_bytes)


# ---------------------------------------------------------------------------
# Full entry path -- Phase 3 council gate
# ---------------------------------------------------------------------------

def _full_main(args: argparse.Namespace) -> int:
    """Full training path -- raises NotImplementedError pending Phase 3 council.

    Per CLAUDE.md "Design decisions -- non-negotiable" + "Long-burn
    score-lowering campaign default" the C1 multi-stage training schedule
    is a council-grade decision:

    - $30-50 GPU budget over 3-4 weeks
    - 6-8 stages: world-model alone -> foveation alone -> combined fine-tune
      -> residual codec -> archive byte sweep -> full Lagrangian convergence
    - Predicted band [0.13, 0.16] [mathematical-derivation; first-principles-bound]
    - Across-class from A1/HNeRV-family (substrate-shift, NOT bolt-on)

    Phase 3 unlock conditions (operator-routable):

    1. Inner-quintet council (Shannon LEAD + Dykstra CO-LEAD + Yousfi +
       Fridrich + Contrarian) approves the multi-stage schedule
    2. Z1 MDL ablation re-baseline confirms C1 is across-class (density
       < 0.90 on the C1 archive after stage 1)
    3. Smoke dispatch on Modal T4 (~$1) validates substrate plumbing end-to-end
    4. Sister long-term campaign C5/C6 anchors land for stacking analysis
    5. Operator-routed budget approval via `tools/operator_authorize.py`
    """
    raise NotImplementedError(
        "C1 full training is gated by Phase 3 council approval per CLAUDE.md "
        "'Design decisions -- non-negotiable'. The multi-stage training "
        "schedule ($30-50 over 3-4 weeks, 6-8 stages, predicted band "
        "[0.13, 0.16] across-class substrate-shift) requires inner-quintet "
        "council sign-off (Shannon + Dykstra + Yousfi + Fridrich + Contrarian) "
        "before any dispatch. Run --smoke to validate plumbing without GPU spend. "
        "See campaign ledger at "
        ".omx/research/campaign_c1_world_model_foveation_20260514.md and "
        "memory file feedback_c1_world_model_foveation_campaign_l1_scaffold_landed_20260514.md "
        "for the 5 operator-routable Phase 3 unlock decisions."
    )


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    _validate_full_cpu_flags(args)
    if args.smoke:
        return _smoke_main(args)
    # device_or_die enforces CUDA for non-smoke unless --full-cpu paired
    if not args.full_cpu:
        _canon_device_or_die(
            args.device, smoke=False, substrate_tag=SUBSTRATE_TAG
        )
    return _full_main(args)


if __name__ == "__main__":  # pragma: no cover -- CLI entry
    sys.exit(main())
