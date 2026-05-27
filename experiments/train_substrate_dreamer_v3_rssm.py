# SPDX-License-Identifier: MIT
"""Train DreamerV3 RSSM categorical posterior substrate — L0 MLX-LOCAL SCAFFOLD.

Per the 2026-05-19 T3 grand council per-substrate symposium
(``.omx/research/council_t3_dreamerv3_rssm_paradigm_bridge_per_substrate_symposium_20260519.md``)
verdict PROCEED_WITH_REVISIONS + 6 binding op-routables; this trainer is the
canonical landing of op-routable #2 (Path B2 design memo + scaffold).

L0 SCAFFOLD SCOPE (research_only per CLAUDE.md "Substrate scaffolds MUST be
COMPLETE or RESEARCH-ONLY"):

- MLX-local-only at L0 per CLAUDE.md "MLX portable-local-substrate authority"
  + corrected #1258 empirical anchor 2026-05-26 (|S_MLX-S_PT|=0.000011 = 72×
  smaller than PR110 frontier delta 0.000789 — MLX is contest-grade at all
  score-granularities for Path 3 substrate iteration).
- Synthetic frame targets at L0 (smoke); per CLAUDE.md "Forbidden
  make_synthetic_pair_batch in non-smoke" the _full_main path raises
  NotImplementedError until L1+ wires:
  (a) PyTorch port via canonical Path 3 export bridge (sister #1251 + #1257);
  (b) score-aware loss via tac.substrates._shared.score_aware_common per Catalog #164;
  (c) real frame loader from upstream/videos/0.mkv (NOT synthetic);
  (d) Modal smoke per symposium tier_1 ($0.30 25ep) before any $5-15 dispatch.
- NO paid CUDA at L0 (no Modal/Vast.ai/Lightning).
- NO contest score claim at L0 (axis_tag=[macOS-MLX research-signal]).
- NO archive byte mutation gate at L0 (planned per Catalog #272
  distinguishing-feature: per-pair category index mutation).

Usage (MLX-local smoke; ~2 epochs; synthetic frame targets — MSE proxy)::

    .venv/bin/python experiments/train_substrate_dreamer_v3_rssm.py \\
        --output-dir experiments/results/dreamer_v3_rssm_l0_smoke_<utc> \\
        --epochs 2 --num-pairs 16 --smoke

Usage (full; PENDING L1+ PyTorch port — currently raises NotImplementedError)::

    .venv/bin/python experiments/train_substrate_dreamer_v3_rssm.py \\
        --output-dir experiments/results/dreamer_v3_rssm_l1_<utc> \\
        --epochs 100  # FAILS: _full_main raises NotImplementedError

Discipline:
- CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" ✓
- CLAUDE.md "MLX portable-local-substrate authority" ✓ (non-promotable markers)
- CLAUDE.md "Forbidden make_synthetic_pair_batch in non-smoke" ✓ (smoke-only)
- Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS declared as ast.AnnAssign ✓
- Catalog #168 AST walker observes both Assign + AnnAssign ✓
- Catalog #240 recipe-vs-trainer-state consistency: research_only=true ✓
"""
# AUTOCAST_FP16_WAIVED:L0-mlx-scaffold-no-pytorch-training-path-yet-defer-until-l1-port
# TORCH_COMPILE_WAIVED:L0-mlx-scaffold-no-pytorch-training-path-yet-defer-until-l1-port
# NO_GRAD_WAIVED:L0-mlx-scaffold-no-pytorch-eval-path-yet-defer-until-l1-port-with-canonical-helper
# SYNTHETIC_NON_SMOKE_OK:L0-mlx-scaffold-synthetic-frames-only-in-smoke-_full_main-raises-NotImplementedError-per-catalog-240
# DISPATCH_OPTIMIZATION_PROTOCOL_OK:l0-mlx-scaffold-no-paid-dispatch-research_only-true-per-claude-md-substrate-scaffolds-must-be-complete-or-research-only

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np

# Canonical helpers per Catalog #190 / #197
from tac.substrates._shared.trainer_skeleton import (
    detect_hardware_substrate as _canon_detect_hardware_substrate,
)
from tac.substrates._shared.trainer_skeleton import (
    pin_seeds as _canon_pin_seeds,
)
from tac.substrates._shared.trainer_skeleton import (
    utc_now_iso as _canon_utc_now_iso,
)

# Substrate primitives
from tac.substrates.dreamer_v3_rssm import (
    CANONICAL_EQUATION_IDS,
    DreamerV3RSSMConfig,
    DreamerV3RSSMSubstrateMLX,
    pack_archive,
)

try:  # pragma: no cover - exercised in environments with MLX installed
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten
except Exception as exc:  # pragma: no cover
    mx = None  # type: ignore[assignment]
    optim = None  # type: ignore[assignment]
    tree_flatten = None  # type: ignore[assignment]
    _MLX_IMPORT_ERROR: Exception | None = exc
else:
    _MLX_IMPORT_ERROR = None


# ---------------------------------------------------------------------------
# Catalog #151 manifest — EVERY flag below must be threaded by any operator
# wrapper that subprocess-invokes this trainer. Annotated as ast.AnnAssign
# so Catalog #168's AST walker observes it. The L0 scaffold has NO required
# input files (MLX-local synthetic; no archive prerequisite, no video, no
# upstream scorer load) — per Catalog #152 required_input_file=False for all.
# ---------------------------------------------------------------------------
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--output-dir": {
        "env": "DREAMER_V3_RSSM_OUTPUT_DIR",
        "rationale": (
            "L0 scaffold output directory for the substrate's MLX-local "
            "training artifacts: stats JSON + archive.zip (smoke-only) + "
            "decoder param manifest + observability surface."
        ),
        "default": "",
        "required_input_file": False,
    },
    "--epochs": {
        "env": "DREAMER_V3_RSSM_EPOCHS",
        "rationale": (
            "Number of MLX-local training epochs. L0 smoke default 2; "
            "full training pending L1+ PyTorch port + Modal $0.30 25ep "
            "smoke per symposium tier_1 ladder."
        ),
        "default": "2",
        "required_input_file": False,
    },
    "--num-pairs": {
        "env": "DREAMER_V3_RSSM_NUM_PAIRS",
        "rationale": (
            "Per-pair latent count. L0 smoke default 16 (small for fast "
            "MLX-local iteration); full = 600 per contest pair count."
        ),
        "default": "16",
        "required_input_file": False,
    },
    "--smoke": {
        "env": "DREAMER_V3_RSSM_SMOKE",
        "rationale": (
            "Synthetic-frame MLX-local smoke flag. Required at L0 per "
            "CLAUDE.md 'Forbidden make_synthetic_pair_batch in non-smoke'."
        ),
        "default": "1",
        "required_input_file": False,
    },
}


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Train DreamerV3 RSSM categorical posterior substrate "
            "(Hafner 2024 paradigm-bridge candidate; L0 MLX-local "
            "scaffold per per-substrate symposium op-routable #2)."
        )
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for stats, archive, manifest.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Training epochs (L0 smoke default 2).",
    )
    p.add_argument(
        "--num-pairs",
        type=int,
        default=16,
        help="Per-pair latent count (L0 smoke default 16; full=600).",
    )
    p.add_argument(
        "--num-groups",
        type=int,
        default=24,
        help="Categorical groups G (default 24 per symposium C6 adaptation).",
    )
    p.add_argument(
        "--num-categories",
        type=int,
        default=256,
        help="Categories per group K (default 256 per symposium 8-bit packing).",
    )
    p.add_argument(
        "--base-channels",
        type=int,
        default=24,
        help="Decoder base channels (default 24).",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate (default 1e-3).",
    )
    p.add_argument(
        "--gumbel-temperature",
        type=float,
        default=1.0,
        help="Gumbel-Softmax temperature τ at training start.",
    )
    p.add_argument(
        "--gumbel-temperature-final",
        type=float,
        default=0.1,
        help="Gumbel-Softmax temperature τ at training end (annealed linearly).",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Synthetic-frame MLX-local smoke (required at L0 per Catalog #114).",
    )
    p.add_argument(
        "--write-archive",
        action="store_true",
        help="Write RSSMC1 archive after smoke (NOT a contest score claim).",
    )
    p.add_argument(
        "--video-path",
        type=Path,
        default=Path("upstream/videos/0.mkv"),
        help="Real contest video for --full MLX-first score-aware training "
        "(Catalog #114; real video, never synthetic in non-smoke).",
    )
    p.add_argument(
        "--full-lr",
        type=float,
        default=1e-3,
        help="Learning rate for --full MLX score-aware training.",
    )
    p.add_argument(
        "--distillation-weight",
        type=float,
        default=0.5,
        help="Weight on the gradient-reachable Hinton-KL T=2.0 scorer "
        "surrogate term in the MLX --full score-aware loss (0.0 disables).",
    )
    return p


# ---------------------------------------------------------------------------
# L0 SCAFFOLD smoke main (MLX-local; synthetic frame targets)
# ---------------------------------------------------------------------------


def _smoke_main(args: argparse.Namespace) -> int:
    """L0 MLX-local synthetic-frame smoke (NOT contest score claim).

    Generates synthetic random (B, 2, 3, 384, 512) target frames, runs N epochs
    of Gumbel-Softmax sampling + MSE loss against the synthetic targets, and
    (optionally) emits a byte-deterministic RSSMC1 archive. Demonstrates the
    full L0 contract: per-pair logits + Gumbel-Softmax + decoder forward +
    archive pack + observability surface.

    Per CLAUDE.md "Forbidden empirical-claim-without-evidence-tag" + Catalog
    #287: every output carries [macOS-MLX research-signal] axis tag + score_claim=false.
    """
    if mx is None or optim is None:
        raise RuntimeError(
            f"MLX not available: {_MLX_IMPORT_ERROR!r}. "
            "L0 scaffold trainer requires MLX (macOS Apple Silicon)."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_at_utc = _canon_utc_now_iso()
    started_at_perf = time.perf_counter()

    _canon_pin_seeds(int(args.seed))
    mx.random.seed(int(args.seed))

    cfg = DreamerV3RSSMConfig(
        num_groups=int(args.num_groups),
        num_categories=int(args.num_categories),
        base_channels=int(args.base_channels),
        num_pairs=int(args.num_pairs),
        gumbel_temperature=float(args.gumbel_temperature),
        use_straight_through=True,
    )
    model = DreamerV3RSSMSubstrateMLX(cfg)
    manifest = model.architecture_manifest()
    print(f"[L0-SCAFFOLD-MLX-SMOKE] config: {manifest}")

    # Synthetic targets (B, 2, 3, 384, 512) in [0, 255]
    target_key = mx.random.key(int(args.seed) + 1)
    targets = mx.random.uniform(
        low=0.0,
        high=255.0,
        shape=(cfg.num_pairs, 2, 3, *cfg.eval_size),
        key=target_key,
    )

    # Optimizer: AdamW for the substrate params (decoder + cat_proj + per-pair logits)
    optimizer = optim.AdamW(learning_rate=float(args.lr))

    def loss_fn(model_inner: DreamerV3RSSMSubstrateMLX, indices: Any) -> Any:
        rgb_pair, _cat_indices, _soft = model_inner.forward_training(indices)
        # MSE proxy in [0, 255]^2 space against synthetic target frames
        target_batch = mx.take(targets, indices, axis=0)
        diff = rgb_pair - target_batch
        return mx.mean(diff * diff)

    loss_and_grad = nn.value_and_grad(model, loss_fn)

    epoch_losses: list[float] = []
    for epoch in range(int(args.epochs)):
        # Anneal Gumbel temperature linearly start → final
        frac = (epoch + 1) / max(int(args.epochs), 1)
        cur_tau = (
            float(args.gumbel_temperature) * (1.0 - frac)
            + float(args.gumbel_temperature_final) * frac
        )
        object.__setattr__(model.cfg, "_runtime_tau", cur_tau)  # diagnostic only
        # Full-batch (small num_pairs at L0)
        indices = mx.arange(cfg.num_pairs)
        loss, grads = loss_and_grad(model, indices)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        epoch_loss = float(loss)
        epoch_losses.append(epoch_loss)
        print(
            f"[L0-SCAFFOLD-MLX-SMOKE] epoch {epoch + 1}/{int(args.epochs)} "
            f"loss={epoch_loss:.4f} tau={cur_tau:.3f}"
        )

    elapsed_seconds = time.perf_counter() - started_at_perf
    completed_at_utc = _canon_utc_now_iso()
    hw_substrate = _canon_detect_hardware_substrate(
        axis="cpu",
        substrate_tag="dreamer_v3_rssm_l0_smoke",
    )

    # Catalog #324 + sister phantom-random-init discipline: this is L0 smoke;
    # no Tier-C density measurement claim; no predicted_band promotion.
    stats: dict[str, Any] = {
        "schema_version": "dreamer_v3_rssm_l0_smoke_stats_v1",
        "substrate_id": "dreamer_v3_rssm_l0_mlx_scaffold",
        "substrate_tag": "dreamer_v3_rssm_l0_mlx_scaffold_20260526",
        "lane_id": "lane_dreamer_v3_rssm_mlx_scaffold_20260526",
        "started_at_utc": started_at_utc,
        "completed_at_utc": completed_at_utc,
        "elapsed_seconds": elapsed_seconds,
        "hardware_substrate": hw_substrate,
        "architecture_manifest": manifest,
        "config": {
            "num_groups": cfg.num_groups,
            "num_categories": cfg.num_categories,
            "base_channels": cfg.base_channels,
            "num_pairs": cfg.num_pairs,
            "gumbel_temperature_start": float(args.gumbel_temperature),
            "gumbel_temperature_final": float(args.gumbel_temperature_final),
            "lr": float(args.lr),
            "epochs": int(args.epochs),
            "seed": int(args.seed),
        },
        "epoch_losses": epoch_losses,
        "final_loss_mse_synthetic": float(epoch_losses[-1]) if epoch_losses else None,
        "canonical_equation_refs": list(CANONICAL_EQUATION_IDS),
        # Non-promotable markers per CLAUDE.md "MLX portable-local-substrate authority"
        # + Catalog #127/#192/#317/#341
        "axis_tag": "[macOS-MLX research-signal]",
        "evidence_grade": "macOS-MLX-research-signal",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "score_claim_valid": False,
        "result_review_blockers": [
            "l0_mlx_scaffold_synthetic_frame_targets_not_contest_video",
            "l0_no_score_aware_loss_no_segnet_no_posenet_feedback",
            "l0_no_paired_contest_cpu_plus_cuda_anchor",
            "l1_pytorch_port_pending_per_path_3_cascade_sister_1251_1257",
        ],
    }
    stats_path = args.output_dir / "stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")
    print(f"[L0-SCAFFOLD-MLX-SMOKE] wrote stats to {stats_path}")

    if bool(args.write_archive):
        # Build deterministic RSSMC1 archive from MLX state_dict + last-epoch
        # argmax indices (per-pair category indices stored as int8 in archive).
        flat = dict(tree_flatten(model.parameters()))
        sd_numpy = {
            k: np.array(v).astype(np.float32)
            for k, v in flat.items()
            if not k.startswith("logits")
        }
        # Get argmax indices via final eval forward
        final_indices = mx.argmax(model.logits, axis=-1)  # type: ignore[union-attr]
        final_indices_np = np.array(final_indices).astype(np.int32)

        archive_bytes = pack_archive(
            sd_numpy,
            final_indices_np,
            meta={
                "schema_version": "dreamer_v3_rssm_rssmc1_v1",
                "gumbel_temperature_final": float(args.gumbel_temperature_final),
                "use_straight_through": True,
                "lane_id": "lane_dreamer_v3_rssm_mlx_scaffold_20260526",
                "axis_tag": "[macOS-MLX research-signal]",
                "score_claim": False,
                "canonical_equation_refs": list(CANONICAL_EQUATION_IDS),
            },
            num_groups=cfg.num_groups,
            num_categories=cfg.num_categories,
            num_pairs=cfg.num_pairs,
            decoder_latent_dim=cfg.decoder_latent_dim,
            base_channels=cfg.base_channels,
        )
        archive_path = args.output_dir / "archive.zip"
        # Write as single-member ZIP for consistency with PR95 contest packet pattern
        import zipfile
        info = zipfile.ZipInfo("0.bin", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.external_attr = 0o100644 << 16
        with zipfile.ZipFile(archive_path, "w") as zf:
            zf.comment = b""
            zf.writestr(info, archive_bytes)
        archive_sha = hashlib.sha256(archive_path.read_bytes()).hexdigest()
        archive_size = archive_path.stat().st_size
        print(
            f"[L0-SCAFFOLD-MLX-SMOKE] wrote archive to {archive_path} "
            f"(member_bytes={len(archive_bytes):,}, zip_bytes={archive_size:,}, "
            f"sha256={archive_sha[:16]}…)"
        )
        # Update stats with archive provenance
        stats["archive_member_bytes"] = len(archive_bytes)
        stats["archive_zip_bytes"] = archive_size
        stats["archive_sha256"] = archive_sha
        try:
            stats["archive_path"] = str(archive_path.resolve().relative_to(REPO_ROOT))
        except ValueError:
            # Archive written outside repo root (rare in normal use); keep absolute
            stats["archive_path"] = str(archive_path.resolve())
        stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")

    print("[L0-SCAFFOLD-MLX-SMOKE] DONE")
    return 0


# ---------------------------------------------------------------------------
# Full main (PENDING L1+ PyTorch port; raises NotImplementedError per Catalog #240)
# ---------------------------------------------------------------------------


def _full_main(args: argparse.Namespace) -> int:
    """MLX-first score-aware full training via the canonical MLX harness.

    MLX-SCORE-AWARE-HARNESS-WAVE 2026-05-27: this ``_full_main`` now routes
    through the canonical substrate-AGNOSTIC harness
    ``tac.substrates._shared.mlx_score_aware_full_main.run_mlx_score_aware_full_main``
    (sister of ``pact_nerv_full_main.py`` for MLX-first substrates). The harness
    extinguishes the prior ``NotImplementedError`` by binding:

    1. Real contest-video targets via ``decode_mlx_targets`` (Catalog #114; the
       prior "real frame loader" deferral is now satisfied — NO synthetic in
       non-smoke per CLAUDE.md "Forbidden make_synthetic_pair_batch").
    2. Gradient-reachable score-aware loss (reconstruction MSE + Hinton-KL
       T=2.0 scorer surrogate) — the prior "score-aware loss" deferral is now
       satisfied (Catalog #164 sister discipline; MLX-native gradient path).
    3. Canonical EMA shadow / OOM-safe step / telemetry / Provenance /
       posterior anchor via ``run_long_training``.

    ## Canonical-vs-unique decision per layer (Catalog #290)

    - ADOPT_CANONICAL: training loop / EMA / score-aware-loss / Provenance /
      posterior anchor (the harness + ``run_long_training``).
    - FORK (this substrate's UNIQUE primitive): the DreamerV3 RSSM categorical
      posterior + Gumbel-Softmax STE + HNeRV decoder (``module.py`` —
      ``DreamerV3RSSMSubstrateMLX``).

    ## Dispatch gating (Catalog #325)

    MLX-LOCAL ONLY ($0 M5 Max); the harness fails closed on a non-MLX host
    (NO CPU/CUDA paid-dispatch leak per Catalog #1 + #317). The recipe stays
    ``dispatch_enabled: false`` + ``research_only: true``; output is
    non-promotable ``[macOS-MLX research-signal]`` per Catalog #192/#341.

    Still DEFERRED to the PyTorch sister L2 / paid-dispatch path (Catalog #325
    + #315 symposium revisions): per-axis SegNet/PoseNet decomposition,
    MLX->PyTorch export bridge (#1251), Catalog #319 deliverability_proof,
    Catalog #270 Tier 1/2/3 dispatch-protocol declarations, paired
    [contest-CUDA] + [contest-CPU] anchor.
    """
    from tac.substrates._shared.mlx_score_aware_full_main import (
        RendererBundle,
        decode_mlx_targets,
        run_mlx_score_aware_full_main,
    )

    if mx is None:
        raise RuntimeError(
            f"MLX not available: {_MLX_IMPORT_ERROR!r}. dreamer_v3_rssm "
            "_full_main is MLX-local (Apple Silicon); there is NO CPU/CUDA "
            "fallback per Catalog #1 + #317."
        )

    cfg = DreamerV3RSSMConfig(
        num_groups=int(args.num_groups),
        num_categories=int(args.num_categories),
        base_channels=int(args.base_channels),
        num_pairs=int(args.num_pairs),
        gumbel_temperature=float(args.gumbel_temperature),
        use_straight_through=True,
    )
    model = DreamerV3RSSMSubstrateMLX(cfg)
    out_h, out_w = cfg.eval_size  # HNeRV decoder hardcodes 384x512 output.
    target_rgb_0, target_rgb_1 = decode_mlx_targets(
        args.video_path,
        num_pairs=int(args.num_pairs),
        output_height=out_h,
        output_width=out_w,
    )
    bundle = RendererBundle(
        model=model,
        target_rgb_0=target_rgb_0,
        target_rgb_1=target_rgb_1,
        num_pairs=int(args.num_pairs),
        forward_convention="call_b2chw_255",
        distillation_weight=float(args.distillation_weight),
    )
    artifact = run_mlx_score_aware_full_main(
        bundle=bundle,
        substrate_id="dreamer_v3_rssm",
        lane_id="lane_dreamer_v3_rssm_mlx_scaffold_20260526",
        output_dir=args.output_dir,
        epochs=int(args.epochs),
        batch_pair_indices_per_step=min(int(args.num_pairs), 8),
        learning_rate=float(args.full_lr),
        seed=int(args.seed),
        notes=(
            "DreamerV3 RSSM MLX-first score-aware full training via canonical "
            "mlx_score_aware_full_main harness; real contest video + "
            "reconstruction + Hinton-KL T=2.0 scorer surrogate; non-promotable "
            "[macOS-MLX research-signal] per Catalog #192/#317/#341; per-axis + "
            "MLX->PyTorch bridge + paired CUDA/CPU anchor DEFERRED to sister L2."
        ),
    )
    print(
        f"[dreamer_v3_rssm:_full_main] DONE epochs={artifact.total_epochs_completed} "
        f"promotable={artifact.promotable} wall={artifact.total_wall_clock_seconds:.1f}s "
        f"artifact={args.output_dir / 'training_artifact.json'}"
    )
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if bool(args.smoke):
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":  # pragma: no cover — CLI smoke
    sys.exit(main())
