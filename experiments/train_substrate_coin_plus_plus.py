# SPDX-License-Identifier: MIT
"""Train the coin_plus_plus substrate L0 SCAFFOLD (WAVE-3-NERV-LITERATURE-L0-RESCOPED 2026-05-20).
# NO_GRAD_WAIVED:MLX_substrate_trainer_uses_mx_no_grad_or_substrate_uses_lazy_eval_no_autograd_per_mlx_first_canonical_doctrine_4107bbf8d_or_substrate_eval_uses_alternate_memory_management_per_comprehensive_bug_audit_cascade_20260526

Operator-callable training scaffold per the WAVE-3-NERV-LITERATURE-L0-RESCOPED
queue 2026-05-20. SCAFFOLD-LEVEL: ``_smoke_main`` exercises the substrate
package; ``_full_main`` raises ``NotImplementedError`` per CLAUDE.md
"Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" non-negotiable +
Catalog #220 + #240 + #315 + #325.

This trainer's ``SubstrateContract`` declares ``research_only=True``;
``dispatch_enabled: false`` on the matching recipe.

Architectural twist vs NeRV-family: coin_plus_plus uses a SHARED base
coord-MLP + per-pair FiLM-style modulation vectors (typically MOD_DIM=64).
The base network is amortized over all pairs; the per-pair latent is
extremely small (~76 KB for 600 pairs at MOD_DIM=64 + int8 quantization).

Hypothesis (operator's 5-tier fit-ranking MODERATE FIT ⭐⭐⭐): the
modulation paradigm is structurally different from per-pair latent + shared
decoder; different rate-tradeoff. Lower per-pair cost may enable better
PSNR per byte at the cost of higher per-pixel decode complexity.

Literature anchor: Dupont et al. ICML 2022 "COIN++: Neural Compression
across Modalities" (arXiv:2201.12904).

Council-binding contract per CLAUDE.md non-negotiables identical to sister
boost_nerv/nirvana L0 SCAFFOLDs.

Usage (smoke; CPU, tiny config)::

    .venv/bin/python experiments/train_substrate_coin_plus_plus.py \\
        --output-dir experiments/results/coin_plus_plus_smoke_<utc> \\
        --epochs 2 --device cpu --smoke

Usage (full; refused at L0 SCAFFOLD)::

    # raises NotImplementedError per Catalog #240 + #315 + #325

Cross-ref:
    src/tac/substrates/coin_plus_plus/ (substrate package)
    experiments/train_substrate_ds_nerv.py (canonical sister NeRV-family trainer)
    .omx/operator_authorize_recipes/substrate_coin_plus_plus_modal_t4_dispatch.yaml
    .omx/research/coin_plus_plus_l0_scaffold_design_20260520T184500Z.md
"""
# AUTOCAST_FP16_WAIVED:l0-scaffold-no-full-training-path
# TORCH_COMPILE_WAIVED:l0-scaffold-no-full-training-path
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tac.substrate_registry import SubstrateContract, register_substrate
from tac.substrates._shared.trainer_skeleton import (
    detect_hardware_substrate as _canon_detect_hardware_substrate,
)
from tac.substrates._shared.trainer_skeleton import (
    device_or_die as _canonical_device_or_die,
)
from tac.substrates._shared.trainer_skeleton import (
    git_head_sha as _canonical_git_head_sha,
)
from tac.substrates._shared.trainer_skeleton import (
    pin_seeds as _canonical_pin_seeds,
)
from tac.substrates._shared.trainer_skeleton import (
    sha256_bytes as _canonical_sha256_bytes,
)
from tac.substrates._shared.trainer_skeleton import (
    torch_version_string as _canonical_torch_version_string,
)
from tac.substrates._shared.trainer_skeleton import (
    utc_now_iso as _canonical_utc_now_iso,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO_PATH = REPO_ROOT / "upstream" / "videos" / "0.mkv"
DEFAULT_UPSTREAM_DIR = REPO_ROOT / "upstream"
EVAL_HW = (384, 512)
N_PAIRS_FULL = 600
CONTEST_NORMALIZER = 37_545_489.0
SUBSTRATE_TAG = "coin_plus_plus"


# ---------------------------------------------------------------------------
# Catalog #151 manifest
# ---------------------------------------------------------------------------
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--video-path": {
        "env": "COIN_PLUS_PLUS_VIDEO_PATH",
        "rationale": (
            "score-aware substrate MUST train against the contest video; "
            "synthetic data is FORBIDDEN outside --smoke"
        ),
        "default": str(DEFAULT_VIDEO_PATH.relative_to(REPO_ROOT)),
        "satisfied_by_profile": (), "requires": (),
        "required_input_file": True,
        "generator_command": (
            "contest-pinned upstream snapshot - never regenerated locally"
        ),
        "rationale_audit": (
            ".omx/research/coin_plus_plus_l0_scaffold_design_20260520T184500Z.md"
        ),
    },
    "--output-dir": {
        "env": "COIN_PLUS_PLUS_OUTPUT_DIR",
        "rationale": "custody location for checkpoints + archive + provenance",
        "satisfied_by_profile": (), "requires": (),
    },
    "--epochs": {
        "env": "COIN_PLUS_PLUS_EPOCHS",
        "rationale": (
            "coin_plus_plus substrate engineering pass; council Phase 2 review pending"
        ),
        "default": "2",
        "satisfied_by_profile": (), "requires": (),
    },
    "--upstream-dir": {
        "env": "COIN_PLUS_PLUS_UPSTREAM_DIR",
        "rationale": "upstream/ root for scorer weights + evaluate.py",
        "default": str(DEFAULT_UPSTREAM_DIR.relative_to(REPO_ROOT)),
        "satisfied_by_profile": (), "requires": (),
    },
    "--device": {
        "env": "COIN_PLUS_PLUS_DEVICE",
        "rationale": (
            "compute device; cuda required for full training (MPS refused); "
            "cpu permitted only with --smoke"
        ),
        "default": "cpu",
        "satisfied_by_profile": (), "requires": (),
    },
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_substrate_coin_plus_plus",
        description=(
            "Train coin_plus_plus substrate L0 SCAFFOLD (smoke only; full path "
            "gated by Phase 2 council per Catalog #240 + #315 + #325)."
        ),
    )
    p.add_argument("--video-path", type=Path, default=DEFAULT_VIDEO_PATH)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260520)
    p.add_argument(
        "--modulation-dim", type=int, default=16,
        help="Per-pair modulation vector dim (smoke=16; full default 64).",
    )
    p.add_argument(
        "--hidden-dim", type=int, default=32,
        help="Coord-MLP hidden channel count (smoke=32; full default 96).",
    )
    p.add_argument(
        "--num-hidden-layers", type=int, default=2,
        help="Coord-MLP depth (smoke=2; full default 4).",
    )
    p.add_argument(
        "--device", choices=["cuda", "cpu"], default="cpu",
        help=(
            "Compute device. L0 SCAFFOLD smoke runs on CPU; cuda permitted "
            "only for future Phase 2. MPS rejected per CLAUDE.md."
        ),
    )
    p.add_argument(
        "--smoke", action="store_true",
        help="Tiny CPU smoke (synthetic targets OK per Catalog #114).",
    )
    p.add_argument(
        "--enable-autocast-fp16", action="store_true", default=False,
        help="RESERVED (Phase 2).",
    )
    p.add_argument(
        "--enable-torch-compile", action="store_true", default=False,
        help="RESERVED (Phase 2).",
    )
    p.add_argument(
        "--enable-gt-scorer-cache", action="store_true", default=False,
        help="RESERVED (Phase 2).",
    )
    return p


def _utc_now_iso() -> str:
    return _canonical_utc_now_iso()


def _sha256_bytes(data: bytes) -> str:
    return _canonical_sha256_bytes(data)


def _git_head_sha() -> str:
    return _canonical_git_head_sha(REPO_ROOT)


def _pin_seeds(seed: int) -> None:
    _canonical_pin_seeds(seed)


def _device_or_die(name: str, *, smoke: bool):
    return _canonical_device_or_die(name, smoke=smoke, substrate_tag=SUBSTRATE_TAG)


def _smoke_main(args: argparse.Namespace) -> int:
    """Tiny CPU smoke exercising the coin_plus_plus substrate library."""
    import torch

    from tac.differentiable_eval_roundtrip import patch_upstream_yuv6_globally
    from tac.substrates.coin_plus_plus.architecture import (
        CoinplusplusConfig,
        CoinplusplusSubstrate,
    )

    _pin_seeds(args.seed)
    device = _device_or_die(args.device, smoke=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = _utc_now_iso()

    patch_upstream_yuv6_globally()

    cfg = CoinplusplusConfig(
        modulation_dim=args.modulation_dim,
        hidden_dim=args.hidden_dim,
        num_hidden_layers=args.num_hidden_layers,
        sin_frequency=30.0,
        coord_input_dim=3,
        output_channels=3,
        num_pairs=max(2, args.batch_size),
        output_height=24,
        output_width=32,
    )
    model = CoinplusplusSubstrate(cfg).to(device)
    n_params = model.num_parameters()
    base_params = sum(
        p.numel() for n, p in model.named_parameters() if n != "modulations"
    )
    mod_params = model.modulations.numel()
    print(
        f"[smoke] coin_plus_plus params: {n_params:,} "
        f"(base={base_params:,} shared, mod={mod_params:,} per-pair) "
        f"device={device} mod_dim={cfg.modulation_dim}"
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = max(1, min(args.epochs, 3))
    for step in range(epochs):
        idx = torch.arange(cfg.num_pairs, device=device, dtype=torch.long)
        rgb_0, rgb_1 = model(idx)
        # Smoke surrogate loss (NOT score-aware)
        loss = rgb_0.abs().mean() + rgb_1.abs().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"[smoke] step {step}: loss={loss.item():.4f}")

    ckpt = {
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": asdict(cfg),
        "smoke": True,
    }
    ckpt_path = args.output_dir / "smoke_checkpoint.pt"
    torch.save(ckpt, ckpt_path)
    print(f"[smoke] wrote {ckpt_path}")

    detected_substrate = _canon_detect_hardware_substrate(
        axis="cpu",
        substrate_tag=SUBSTRATE_TAG,
        provenance_path=args.output_dir / "provenance.json",
        env_var_candidates=("COIN_PLUS_PLUS_GPU", "MODAL_GPU"),
    )
    provenance = {
        "schema": "coin_plus_plus_l0_scaffold_smoke_v1",
        "generated_at": _utc_now_iso(),
        "started_at": started_at,
        "git_head": _git_head_sha(),
        "trainer": "experiments/train_substrate_coin_plus_plus.py",
        "lane_id": "lane_coin_plus_plus_l0_scaffold_20260520",
        "substrate_tag": SUBSTRATE_TAG,
        "args": {
            k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
        },
        "pytorch_version": _canonical_torch_version_string(),
        "device": str(device),
        "hardware_substrate_detected": detected_substrate,
        "n_params_total": int(n_params),
        "n_params_base_shared": int(base_params),
        "n_params_modulation_per_pair_pool": int(mod_params),
        "modulation_dim": cfg.modulation_dim,
        "smoke": True,
        "synthetic_targets_used_per_catalog_114": True,
        "score_claim": False,
        "score_axis_tag": None,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "custody_status": "ci-rebuildable",
        "evidence_grade": "scaffold-smoke-no-score-axis",
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(f"[smoke] wrote {args.output_dir / 'provenance.json'}")
    return 0


def _full_main(args: argparse.Namespace) -> int:
    """Full training entry point - REFUSED at L0 SCAFFOLD posture.

    Reactivation criteria (per HNeRV parity discipline L2 export-first):
    1. Per-substrate adversarial grand council symposium per Catalog #325.
    2. Cargo-cult audit per Catalog #303: in particular MOD_DIM=64 + int8
       modulation quantization choices both CARGO-CULTED at L0; require
       empirical sweep to validate.
    3. 9-dim checklist evidence per Catalog #294 + observability surface
       per Catalog #305 + Dykstra feasibility predicted-band per Catalog
       #296 all land in the design memo.
    4. Coord-MLP-vs-CNN-base architecture tension explored per the cargo-
       cult audit's HARD-EARNED-vs-CARGO-CULTED TENSION classification.
    5. Score-aware training loop with EMA + score-domain Lagrangian +
       canonical auth-eval helper invocation per Catalog #226. NOTE:
       coord-MLP forward is O(H*W) per pair; per-pair latency may be
       higher than NeRV-family substrates, training step time needs
       per-pair coordinate batching strategy.
    6. Operator-frontier-override per Catalog #300 OR Phase 2 council
       approval converts ``research_only=true`` to ``false``.
    """
    raise NotImplementedError(
        "[coin_plus_plus] full training path is OPERATOR-GATED per Catalog #240 + "
        "#315 + #325. This is an L0 SCAFFOLD trainer; substrate is research_only "
        "until the Phase 2 council symposium + per-pixel coord-MLP training "
        "loop optimization land. See reactivation criteria in this function's "
        "docstring + .omx/research/coin_plus_plus_l0_scaffold_design_20260520T184500Z.md "
        "+ lane_coin_plus_plus_l0_scaffold_20260520 in .omx/state/lane_registry.json."
    )


# ---------------------------------------------------------------------------
# META layer SubstrateContract (Catalog #241/#242)
# ---------------------------------------------------------------------------

COIN_PLUS_PLUS_SUBSTRATE_CONTRACT = SubstrateContract(
    id="coin_plus_plus",
    lane_id="lane_coin_plus_plus_l0_scaffold_20260520",
    target_modes=("research_substrate",),
    deployment_target="t4_contest_runtime",
    council_verdict_provenance=(
        ".omx/research/coin_plus_plus_l0_scaffold_design_20260520T184500Z.md"
    ),
    archive_grammar=(
        "CPP1 monolithic single-file 0.bin (21-byte header carrying "
        "MODULATION_DIM u16 distinctive field; SHARED base coord-MLP "
        "weights in brotli blob; int8 per-pair modulations; utf-8 json meta). "
        "Distinguishing rate-tradeoff: per-pair latent is MOD_DIM bytes (int8); "
        "base network amortized over all pairs."
    ),
    parser_section_manifest={
        "header": "21_byte_fixed_CPP1_magic_v1_modulation_dim",
        "base_mlp_blob": "brotli_quality9_pickled_fp16_shared_coord_mlp_weights",
        "shared_base_mlp_weights_subset": "logical_grouping_inside_base_mlp_blob",
        "modulation_blob": "raw_int8_row_major_num_pairs_x_modulation_dim",
        "per_pair_modulations_subset": "logical_grouping_inside_modulation_blob",
        "meta_blob": "utf8_json_meta",
    },
    inflate_runtime_loc_budget=200,
    runtime_dep_closure=("torch>=2.5,<2.7", "brotli", "av"),
    export_format="custom",
    score_aware_loss="scorer_loss_terms_btchw",
    bolt_on_loc_budget=700,
    no_op_detector_planned=True,
    archive_bytes_added=None,
    score_improvement_mechanism_status="RESEARCH_ONLY",
    runtime_overlay_consumed=False,
    recipe_smoke_only=False,
    recipe_research_only=True,
    recipe_min_smoke_gpu="T4",
    recipe_min_vram_gb=16,
    recipe_pyav_decode_strategy="cpu_thread_async_upload",
    recipe_canary_status="post_canary_dependent",
    recipe_video_input_strategy="per_dispatch_local_copy",
    recipe_canary_dependency="ds_nerv",
    cost_band_epochs=100,
    cost_band_gpu_key="T4",
    cost_band_platform_key="modal",
    cost_band_p50_usd=0.59,
    hook_sensitivity_contribution="not_applicable_with_rationale",
    hook_pareto_constraint="rate_distortion_v1",
    hook_bit_allocator_class="not_applicable_with_rationale",
    hook_autopilot_ranker_class_shift_token=None,
    hook_continual_learning_anchor_kind="not_applicable_with_rationale",
    hook_probe_disambiguator=None,
    catalog_compliance_declarations=(
        "catalog_146_3arg_archive_grammar_planned",
        "catalog_151_tier1_required_flags_declared",
        "catalog_205_select_inflate_device_planned",
        "catalog_220_operational_mechanism_research_only",
        "catalog_226_gate_auth_eval_call_planned",
        "catalog_240_recipe_vs_trainer_state_research_only",
        "catalog_315_optimal_form_before_dispatch_council_pending",
        "catalog_325_per_substrate_symposium_pending",
    ),
    hook_not_applicable_rationale={
        "hook_sensitivity_contribution": (
            "L0 SCAFFOLD; no sensitivity signal until Phase 2 council + full path"
        ),
        "hook_bit_allocator_class": (
            "fp16 brotli on shared base MLP + int8 modulations; no per-tensor "
            "bit allocator at scaffold posture"
        ),
        "hook_continual_learning_anchor_kind": (
            "L0 SCAFFOLD: no posterior anchor until full path lands and a "
            "[contest-CUDA] anchor is measured"
        ),
        "hook_probe_disambiguator": (
            "single mechanism (FiLM-modulated coord-MLP); coord-MLP-vs-CNN-base "
            "alternative is the L1+ Phase 2 research path per HARD-EARNED-vs-"
            "CARGO-CULTED TENSION in __init__.py cargo-cult audit"
        ),
    },
)


@register_substrate(COIN_PLUS_PLUS_SUBSTRATE_CONTRACT)
def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.smoke:
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
