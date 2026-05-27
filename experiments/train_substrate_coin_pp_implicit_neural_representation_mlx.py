# SPDX-License-Identifier: MIT
"""COIN++ implicit neural representation MLX-first score-aware trainer.
# NO_GRAD_WAIVED:MLX_substrate_trainer_uses_mlx_value_and_grad_lazy_eval_no_pytorch_autograd_per_mlx_first_canonical_doctrine_8th_standing_directive
# AUTOCAST_FP16_WAIVED:MLX_substrate_trainer_does_not_use_PyTorch_CUDA_autocast_fp16_primitive_per_mlx_first_canonical_doctrine_8th_standing_directive
# TORCH_COMPILE_WAIVED:MLX_substrate_trainer_has_no_pytorch_training_path_per_mlx_first_canonical_doctrine_8th_standing_directive
# SYNTHETIC_NON_SMOKE_OK:synthetic_targets_only_in_smoke_full_path_decodes_real_contest_video_via_decode_mlx_targets_catalog_114
# DISPATCH_OPTIMIZATION_PROTOCOL_OK:mlx_local_no_paid_dispatch_research_only_true_per_claude_md_substrate_scaffolds_must_be_complete_or_research_only

Path 3 candidate #K per operator directive 2026-05-26.

MLX-SCORE-AWARE-HARNESS-WAVE 2026-05-27: ``_full_main`` is UNBLOCKED. The prior
``NotImplementedError`` (Catalog #240(c)) is replaced by a route through the
canonical substrate-AGNOSTIC harness
``tac.substrates._shared.mlx_score_aware.run_mlx_score_aware_full_main``. The
unblock required landing the COIN++ base SIREN coord-MLP + per-pair modulation
network as a single trainable ``mlx.nn.Module`` (``CoinPPRendererMLX`` in
``mlx_renderer.py``) — the prior blocker was that the substrate shipped only
config + cost estimators with NO renderer forward implemented.

## Canonical-vs-unique decision per layer (Catalog #290)

- ADOPT_CANONICAL_BECAUSE_SERVES: training loop / EMA / score-aware loss /
  Provenance / posterior anchor (the harness + ``run_long_training``).
- FORK_BECAUSE_PRINCIPLED_MISMATCH (this substrate's UNIQUE primitive): the
  COIN++ meta-modulated SIREN coord-MLP (``mlx_renderer.CoinPPRendererMLX``;
  Dupont et al. 2022 — per-pair modulation codes + shared FiLM-conditioned
  base network).

## Dispatch gating (Catalog #325)

MLX-LOCAL ONLY ($0 M5 Max); the harness fails closed on a non-MLX host (NO
CPU/CUDA paid-dispatch leak per Catalog #1 + #317). Any recipe stays
``dispatch_enabled: false`` + ``research_only: true``; output is non-promotable
``[macOS-MLX research-signal]`` per Catalog #192/#341. Per-substrate symposium
per Catalog #325 + MLX->PyTorch bridge + paired CUDA/CPU anchor remain DEFERRED
to the PyTorch sister L2 / paid-dispatch path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


# ---------------------------------------------------------------------------
# Catalog #151 manifest (ast.AnnAssign per Catalog #168). --full requires the
# real contest video at --video-path (required_input_file=True).
# ---------------------------------------------------------------------------
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--output-dir": {
        "env": "COIN_PP_OUTPUT_DIR",
        "rationale": (
            "Output dir for MLX-local training artifacts: training_artifact "
            "JSON + EMA checkpoint + observability surface (NOT /tmp per "
            "Catalog #208)."
        ),
        "default": "",
        "required_input_file": False,
    },
    "--epochs": {
        "env": "COIN_PP_EPOCHS",
        "rationale": (
            "Number of MLX-local training epochs. Full training pending "
            "per-substrate symposium per Catalog #325 before any paid dispatch."
        ),
        "default": "2",
        "required_input_file": False,
    },
    "--video-path": {
        "env": "COIN_PP_VIDEO_PATH",
        "rationale": (
            "Real contest video for --full score-aware training (Catalog "
            "#114; real video, never synthetic in non-smoke)."
        ),
        "default": "upstream/videos/0.mkv",
        "required_input_file": True,
    },
}


def _full_main(args: argparse.Namespace) -> int:
    """MLX-first score-aware full training via the canonical MLX harness.

    Routes the trainable :class:`CoinPPRendererMLX` through the
    substrate-AGNOSTIC harness binding real contest-video targets (Catalog
    #114) + gradient-reachable score-aware loss (reconstruction MSE + Hinton-KL
    T=2.0 scorer surrogate; Catalog #164) + canonical EMA / OOM-safe /
    telemetry / Provenance / posterior anchor via ``run_long_training``.
    """
    from tac.substrates._shared.mlx_score_aware import (
        RendererBundle,
        decode_mlx_targets,
        run_mlx_score_aware_full_main,
    )
    from tac.substrates.coin_pp_implicit_neural_representation.mlx_renderer import (
        CoinPPImplicitNeuralRepresentationConfig,
        CoinPPRendererMLX,
    )

    if args.output_dir is None:
        raise SystemExit(
            "--output-dir is required for --full training "
            "(Catalog #151 TIER_1_OPERATOR_REQUIRED_FLAGS)."
        )

    cfg = CoinPPImplicitNeuralRepresentationConfig(
        mod_dim=int(args.mod_dim),
        pos_dim=int(args.pos_dim),
        hidden_dim=int(args.hidden_dim),
        num_hidden_layers=int(args.num_hidden_layers),
        num_pairs=int(args.num_pairs),
    )
    model = CoinPPRendererMLX(cfg)
    target_rgb_0, target_rgb_1 = decode_mlx_targets(
        args.video_path,
        num_pairs=int(args.num_pairs),
        output_height=cfg.eval_h,
        output_width=cfg.eval_w,
    )
    bundle = RendererBundle(
        model=model,
        target_rgb_0=target_rgb_0,
        target_rgb_1=target_rgb_1,
        num_pairs=int(args.num_pairs),
        forward_convention="reconstruct_pair_nchw01",
        distillation_weight=float(args.distillation_weight),
    )
    artifact = run_mlx_score_aware_full_main(
        bundle=bundle,
        substrate_id="coin_pp_implicit_neural_representation",
        lane_id="lane_path_3_k_coin_pp_implicit_neural_representation_20260526",
        output_dir=args.output_dir,
        epochs=int(args.epochs),
        batch_pair_indices_per_step=min(int(args.num_pairs), 4),
        learning_rate=float(args.full_lr),
        seed=int(args.seed),
        notes=(
            "COIN++ implicit neural representation MLX-first score-aware full "
            "training via canonical mlx_score_aware harness; real contest video "
            "+ reconstruction + Hinton-KL T=2.0 scorer surrogate + per-pair "
            "modulation codes (Dupont 2022); non-promotable [macOS-MLX "
            "research-signal] per Catalog #192/#317/#341; per-axis + MLX->PyTorch "
            "bridge + paired CUDA/CPU anchor DEFERRED to sister L2."
        ),
    )
    print(
        f"[coin_pp:_full_main] DONE epochs={artifact.total_epochs_completed} "
        f"promotable={artifact.promotable} "
        f"wall={artifact.total_wall_clock_seconds:.1f}s "
        f"artifact={args.output_dir / 'training_artifact.json'}"
    )
    return 0


def _smoke_main(args: argparse.Namespace) -> int:
    """MLX-local smoke manifest (config + estimators; no training).

    Emits non-promotable research-signal markers per Catalog #341 +
    Catalog #317. Output destination is operator-configurable.
    """
    from tac.substrates.coin_pp_implicit_neural_representation.mlx_renderer import (
        CoinPPImplicitNeuralRepresentationConfig,
        estimate_archive_bytes,
    )

    cfg = CoinPPImplicitNeuralRepresentationConfig(
        mod_dim=int(args.mod_dim),
        pos_dim=int(args.pos_dim),
        hidden_dim=int(args.hidden_dim),
        num_hidden_layers=int(args.num_hidden_layers),
        num_pairs=min(int(args.num_pairs), 8),
    )
    estimated_bytes = estimate_archive_bytes(cfg)

    output_dir = Path(args.output_dir or ".omx/research/path_3_k_coin_pp_smoke")
    output_dir_str = str(output_dir.resolve())
    if output_dir_str.startswith(("/tmp/", "/private/tmp/")):
        print(
            f"ERROR: output-dir {output_dir} under /tmp per CLAUDE.md "
            "FORBIDDEN_PATTERN 'Forbidden /tmp paths in any persisted artifact'",
            file=sys.stderr,
        )
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)

    smoke_manifest = {
        "schema_version": "coin_pp_smoke_manifest_v1_20260526",
        "substrate_id": "coin_pp_implicit_neural_representation",
        "lane_id": "lane_path_3_k_coin_pp_implicit_neural_representation_20260526",
        "config": {
            "mod_dim": cfg.mod_dim,
            "pos_dim": cfg.pos_dim,
            "hidden_dim": cfg.hidden_dim,
            "num_hidden_layers": cfg.num_hidden_layers,
            "num_pairs": cfg.num_pairs,
            "eval_h": cfg.eval_h,
            "eval_w": cfg.eval_w,
            "modulation_quant_bits": cfg.modulation_quant_bits,
        },
        "estimated_archive_bytes": int(estimated_bytes),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "promotable": False,
        "evidence_grade": "macOS-MLX-research-signal",
        "axis_tag": "[macOS-MLX research-signal]",
        "predicted_delta_adjustment": 0.0,
    }
    manifest_path = output_dir / "smoke_manifest.json"
    manifest_path.write_text(
        json.dumps(smoke_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"[coin_pp smoke] manifest written to: {manifest_path} "
        f"(estimated_archive_bytes={estimated_bytes}) "
        "[macOS-MLX research-signal] non-promotable per Catalog #341",
        file=sys.stderr,
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="COIN++ MLX-first score-aware trainer (Path 3 candidate K)."
    )
    p.add_argument("--smoke", action="store_true", help="Emit smoke manifest only.")
    p.add_argument(
        "--full",
        action="store_true",
        help="Run full MLX score-aware training via the canonical harness.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num-pairs", type=int, default=4)
    p.add_argument("--mod-dim", type=int, default=64)
    p.add_argument("--pos-dim", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=64)
    p.add_argument("--num-hidden-layers", type=int, default=3)
    p.add_argument(
        "--epochs", type=int, default=2, help="MLX score-aware epochs (--full)."
    )
    p.add_argument(
        "--output-dir", type=Path, default=None, help="Output dir (NOT /tmp)."
    )
    p.add_argument(
        "--video-path",
        type=Path,
        default=Path("upstream/videos/0.mkv"),
        help="Real contest video for --full score-aware training (Catalog #114).",
    )
    p.add_argument("--full-lr", type=float, default=1e-3)
    p.add_argument(
        "--distillation-weight",
        type=float,
        default=0.5,
        help="Weight on the gradient-reachable Hinton-KL T=2.0 scorer surrogate "
        "term in the --full score-aware loss (0.0 disables).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.full:
        return _full_main(args)
    if args.smoke:
        return _smoke_main(args)
    _build_parser().print_help()
    return 1


__all__ = ["_full_main", "_smoke_main", "main"]


if __name__ == "__main__":  # pragma: no cover — CLI entry
    sys.exit(main())
