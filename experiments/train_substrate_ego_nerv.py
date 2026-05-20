# SPDX-License-Identifier: MIT
"""Train the ego_nerv substrate L0 SCAFFOLD (operator-routed BUILD-1 2026-05-20).

Operator-callable training scaffold per the BUILD-1 NeRV-trio queue fill
2026-05-20. SCAFFOLD-LEVEL: ``_smoke_main`` exercises the existing
``tac.ego_nerv_as_renderer`` substrate library (Phase-A egocentric pose
conditioning); ``_full_main`` raises ``NotImplementedError`` per CLAUDE.md
"Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" non-negotiable +
Catalog #220 (substrate L1+ scaffold operational mechanism declaration) +
Catalog #240 (recipe-vs-trainer-state consistency).

This trainer's ``SubstrateContract`` declares ``research_only=True``;
``dispatch_enabled: false`` on the matching recipe ensures no paid GPU dispatch
may fire from this scaffold per CLAUDE.md "Substrate MUST be at OPTIMAL FORM
before paid empirical dispatch" non-negotiable (Catalog #315) +
"PER-SUBSTRATE OPTIMAL FORM via adversarial grand council symposium"
non-negotiable (Catalog #325).

Architectural twist vs ``tc_nerv`` (canonical reference):
``ego_nerv`` substitutes a FiLM-modulated egocentric pose table for the
SIREN+PixelShuffle temporal-consistency regularizer. Per-pair pose
(yaw/pitch/roll/x/y/z velocity) is shipped in the archive as a tiny
``pose_table`` (~7 KB for 600 pairs); the decoder consumes it via FiLM
modulation on the latent. Hypothesis: driving video is egomotion-dominated
so pose-conditioning enables ~1.25x param efficiency at fixed bytes (Wang
2024 Ego-NeRV / Park 2023 driving-NeRF).

The substrate library at ``src/tac/ego_nerv_as_renderer.py`` already
implements the encoder (compress-time only), renderer (FiLM + PixelShuffle),
latent table, pose table, ``train_step_ego_nerv``, ``export_ego_nerv_to_archive``
(archive format ID 0x68), and ``_make_synthetic_pair_batch_for_smoke``.
This scaffold trainer wires those primitives through the canonical
substrate-trainer skeleton + CLAUDE.md non-negotiable disciplines.

Council-binding contract (CLAUDE.md non-negotiables) honored end-to-end:

- ``patch_upstream_yuv6_globally()`` BEFORE scorer construction (PR #95/#106
  contract; CLAUDE.md "eval_roundtrip - NON-NEGOTIABLE").
- ``tac.training.EMA(decay=0.997)`` per CLAUDE.md "EMA - NON-NEGOTIABLE"
  (smoke path skips full EMA bookkeeping to stay CPU-light; full path
  reactivates per the reactivation criteria below).
- Smoke uses ``_make_synthetic_pair_batch_for_smoke`` (synthetic targets
  OK because ``--smoke``; never use this output for ranking per Catalog
  #114).
- Non-smoke path raises ``NotImplementedError`` (L0 SCAFFOLD posture);
  reactivation criteria pinned in lane registry notes per HNeRV parity
  discipline L2 (export-first design) + Catalog #220.
- TIER_1_OPERATOR_REQUIRED_FLAGS declared per Catalog #151 for wire-up.
- ``@register_substrate(SubstrateContract(...))`` per Catalog #241 META layer.
- Canonical scorer/device helpers per Catalog #164 / #205 / #226.

Usage (smoke; CPU, tiny config, ~2 steps, synthetic batches)::

    .venv/bin/python experiments/train_substrate_ego_nerv.py \\
        --output-dir experiments/results/ego_nerv_smoke_<utc> \\
        --epochs 2 --device cpu --smoke

Usage (full; refused at L0 SCAFFOLD until Phase 2 council convocation)::

    # raises NotImplementedError per Catalog #240 + #315 + #325

Cross-ref:
    src/tac/ego_nerv_as_renderer.py (substrate library)
    experiments/train_substrate_tc_nerv.py (canonical sister trainer)
    experiments/train_ego_nerv_as_renderer.py (pre-canonical scaffold)
    .omx/operator_authorize_recipes/substrate_ego_nerv_modal_a10g_diagnostic_dispatch.yaml
    .omx/research/ego_nerv_l0_scaffold_design_20260520T<utc>.md
"""
# AUTOCAST_FP16_WAIVED:l0-scaffold-no-full-training-path
# TORCH_COMPILE_WAIVED:l0-scaffold-no-full-training-path
from __future__ import annotations

import argparse
import json
import math
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
SUBSTRATE_TAG = "ego_nerv"


# ---------------------------------------------------------------------------
# Catalog #151 manifest - every flag below must be threaded by any operator
# wrapper that subprocess-invokes this trainer. Schema mirrors the canonical
# tc_nerv manifest per Catalog #151 + #168 AnnAssign-aware extractor.
# ---------------------------------------------------------------------------
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--video-path": {
        "env": "EGO_NERV_VIDEO_PATH",
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
            ".omx/research/ego_nerv_l0_scaffold_design_20260520T140000Z.md"
        ),
    },
    "--output-dir": {
        "env": "EGO_NERV_OUTPUT_DIR",
        "rationale": "custody location for checkpoints + archive + provenance",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--epochs": {
        "env": "EGO_NERV_EPOCHS",
        "rationale": (
            "ego_nerv substrate engineering pass; council Phase 2 review pending "
            "before non-smoke training is unlocked"
        ),
        "default": "2",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--upstream-dir": {
        "env": "EGO_NERV_UPSTREAM_DIR",
        "rationale": (
            "upstream/ root for scorer weights + evaluate.py; required for full "
            "training (non-smoke) and auth eval"
        ),
        "default": str(DEFAULT_UPSTREAM_DIR.relative_to(REPO_ROOT)),
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--device": {
        "env": "EGO_NERV_DEVICE",
        "rationale": (
            "compute device; cuda required for full training (MPS refused per "
            "CLAUDE.md MPS-NOISE rule); cpu permitted only with --smoke"
        ),
        "default": "cpu",
        "satisfied_by_profile": (),
        "requires": (),
    },
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_substrate_ego_nerv",
        description=(
            "Train ego_nerv substrate L0 SCAFFOLD (smoke only; full path "
            "gated by Phase 2 council per Catalog #240 + #315 + #325)."
        ),
    )
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
        default=2,
        help="Number of training epochs (smoke default 2; non-smoke gated).",
    )
    p.add_argument(
        "--upstream-dir",
        type=Path,
        default=DEFAULT_UPSTREAM_DIR,
        help="upstream/ root; required for scorer load + auth eval (non-smoke).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Number of pair indices per batch (smoke default 2).",
    )
    p.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="AdamW learning rate (smoke only).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=20260520,
        help="Manual seed for torch / numpy / random (deterministic).",
    )
    p.add_argument(
        "--latent-dim",
        type=int,
        default=16,
        help="Per-pair latent dimensionality (ego_nerv default 16).",
    )
    p.add_argument(
        "--pose-dim",
        type=int,
        default=6,
        help="Egocentric pose dimensionality (yaw/pitch/roll/x/y/z).",
    )
    p.add_argument(
        "--film-hidden-dim",
        type=int,
        default=64,
        help="FiLM modulator hidden dim.",
    )
    p.add_argument(
        "--base-channels",
        type=int,
        default=36,
        help="Decoder base channels (smoke uses 8).",
    )
    p.add_argument(
        "--n-pairs",
        type=int,
        default=4,
        help="Number of per-pair latents (smoke default 4; full = 600).",
    )
    p.add_argument(
        "--ema-decay",
        type=float,
        default=0.997,
        help="EMA decay (CLAUDE.md non-negotiable default 0.997 for weights).",
    )
    p.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        default="cpu",
        help=(
            "Compute device. L0 SCAFFOLD smoke runs on CPU; cuda permitted "
            "only for future Phase 2 full-mode dispatch. MPS rejected per "
            "CLAUDE.md 'MPS auth eval is NOISE'."
        ),
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Tiny CPU smoke (synthetic targets OK because --smoke; never use "
            "this output for ranking per Catalog #114)."
        ),
    )
    p.add_argument(
        "--enable-autocast-fp16",
        action="store_true",
        default=False,
        help="RESERVED (Phase 2): Wrap forward in torch.autocast(fp16).",
    )
    p.add_argument(
        "--enable-torch-compile",
        action="store_true",
        default=False,
        help="RESERVED (Phase 2): Wrap substrate with torch.compile / Inductor.",
    )
    p.add_argument(
        "--enable-gt-scorer-cache",
        action="store_true",
        default=False,
        help="RESERVED (Phase 2): GT-scorer-output cache (Catalog #228).",
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
    """Tiny CPU smoke exercising the ego_nerv substrate library.

    Loads ``EgoNeRVRenderer`` / ``EgoNeRVLatentTable`` / ``EgoNeRVPoseTable``
    from ``tac.ego_nerv_as_renderer``, runs a forward-pass shape check via
    ``_make_synthetic_pair_batch_for_smoke`` (synthetic targets OK because
    ``--smoke`` per Catalog #114), and writes a smoke checkpoint + provenance.
    No scorer load; no archive build; no auth-eval (canonical Catalog #226
    helper wires those for the future Phase 2 full path).
    """
    import torch

    from tac.ego_nerv_as_renderer import (
        EgoNeRVConfig,
        EgoNeRVLatentTable,
        EgoNeRVPoseTable,
        EgoNeRVRenderer,
        _make_synthetic_pair_batch_for_smoke,
    )
    from tac.differentiable_eval_roundtrip import patch_upstream_yuv6_globally

    _pin_seeds(args.seed)
    device = _device_or_die(args.device, smoke=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = _utc_now_iso()

    # CLAUDE.md "eval_roundtrip - NON-NEGOTIABLE": patch upstream rgb_to_yuv6
    # BEFORE any scorer construction. Smoke does not load scorers, but the
    # patch is idempotent and surfaces any installation issue early.
    patch_upstream_yuv6_globally()

    cfg = EgoNeRVConfig(
        latent_dim=args.latent_dim,
        pose_dim=args.pose_dim,
        film_hidden_dim=args.film_hidden_dim,
        base_channels=8,  # smoke-tiny per substrate library convention
        n_pairs=max(2, args.n_pairs),
        cuda_required=False,
    )
    renderer = EgoNeRVRenderer(cfg).to(device)
    latent_table = EgoNeRVLatentTable(cfg.n_pairs, cfg.latent_dim).to(device)
    pose_table = EgoNeRVPoseTable(cfg.n_pairs, cfg.pose_dim).to(device)

    n_params = (
        sum(p.numel() for p in renderer.parameters())
        + sum(p.numel() for p in latent_table.parameters())
        + sum(p.numel() for p in pose_table.parameters())
    )
    print(f"[smoke] ego_nerv total params: {n_params:,} device={device}")

    epochs = max(1, min(args.epochs, 3))
    for step in range(epochs):
        pair_indices, _gt_pairs = _make_synthetic_pair_batch_for_smoke(
            batch_size=max(1, args.batch_size),
            latent_dim=cfg.latent_dim,
            eval_size=cfg.eval_size,
            n_pairs=cfg.n_pairs,
            seed=args.seed + step,
        )
        pair_indices = pair_indices.to(device)
        z = latent_table(pair_indices)
        pose = pose_table(pair_indices)
        decoded = renderer(z, pose)
        if decoded.shape != (max(1, args.batch_size), 2, 3, *cfg.eval_size):
            raise RuntimeError(
                f"[smoke] decoded shape mismatch: got {tuple(decoded.shape)}; "
                f"expected ({args.batch_size}, 2, 3, {cfg.eval_size[0]}, "
                f"{cfg.eval_size[1]})"
            )
        # Smoke surrogate loss (NOT score-aware): mean magnitude.
        loss = decoded.abs().mean()
        print(f"[smoke] step {step}: loss={loss.item():.4f}")

    ckpt = {
        "renderer": {k: v.detach().cpu() for k, v in renderer.state_dict().items()},
        "latent_table": {
            k: v.detach().cpu() for k, v in latent_table.state_dict().items()
        },
        "pose_table": {
            k: v.detach().cpu() for k, v in pose_table.state_dict().items()
        },
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
        env_var_candidates=("EGO_NERV_GPU", "MODAL_GPU"),
    )
    provenance = {
        "schema": "ego_nerv_l0_scaffold_smoke_v1",
        "generated_at": _utc_now_iso(),
        "started_at": started_at,
        "git_head": _git_head_sha(),
        "trainer": "experiments/train_substrate_ego_nerv.py",
        "lane_id": "lane_ego_nerv_l0_scaffold_20260520",
        "substrate_tag": SUBSTRATE_TAG,
        "args": {
            k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
        },
        "pytorch_version": _canonical_torch_version_string(),
        "device": str(device),
        "hardware_substrate_detected": detected_substrate,
        "n_params": int(n_params),
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

    Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"
    non-negotiable + Catalog #220 (substrate L1+ scaffold operational
    mechanism declaration) + Catalog #240 (recipe-vs-trainer-state
    consistency) + Catalog #315 (OPTIMAL FORM before paid dispatch) +
    Catalog #325 (per-substrate symposium before paid dispatch): this
    scaffold trainer's full path is council-gated. The matching recipe
    declares ``research_only: true`` + ``dispatch_enabled: false`` so
    no paid GPU dispatch can fire until reactivation criteria below land.

    Reactivation criteria (per HNeRV parity discipline L2 export-first):
    1. Per-substrate adversarial grand council symposium per CLAUDE.md
       "PER-SUBSTRATE OPTIMAL FORM via adversarial grand council
       symposium" non-negotiable (Catalog #325) returns PROCEED or
       PROCEED_WITH_REVISIONS verdict.
    2. ``src/tac/substrates/ego_nerv/`` canonical substrate package
       lands with ``architecture.py`` + ``archive.py`` + ``inflate.py``
       + ``score_aware_loss.py`` matching tc_nerv structure (currently
       the substrate library lives at ``src/tac/ego_nerv_as_renderer.py``
       in legacy monolithic format; canonical package migration is the
       blocker for the full ``_full_main`` path).
    3. ``submissions/ego_nerv_substrate/inflate.{py,sh}`` smoke proves
       a contest-compliant 3-positional-arg runtime per Catalog #146.
    4. Cargo-cult audit per Catalog #303 + 9-dim checklist per Catalog
       #294 + observability surface per Catalog #305 + Dykstra
       feasibility predicted-band per Catalog #296 all land in the
       design memo.
    5. Operator-frontier-override per Catalog #300 Mission alignment
       Consequence 1 OR Phase 2 council approval converts
       ``research_only=true`` to ``false`` in the recipe + this
       trainer's ``SubstrateContract``.
    """
    raise NotImplementedError(
        "[ego_nerv] full training path is OPERATOR-GATED per Catalog #240 + "
        "#315 + #325. This is an L0 SCAFFOLD trainer; substrate is research_only "
        "until the Phase 2 council symposium + canonical package migration land. "
        "See reactivation criteria in this function's docstring + "
        ".omx/research/ego_nerv_l0_scaffold_design_20260520T140000Z.md "
        "+ lane_ego_nerv_l0_scaffold_20260520 in .omx/state/lane_registry.json."
    )


# ---------------------------------------------------------------------------
# META layer SubstrateContract (Catalog #241/#242)
# ---------------------------------------------------------------------------

EGO_NERV_SUBSTRATE_CONTRACT = SubstrateContract(
    id="ego_nerv",
    lane_id="lane_ego_nerv_l0_scaffold_20260520",
    target_modes=("research_substrate",),
    deployment_target="t4_contest_runtime",
    council_verdict_provenance=(
        ".omx/research/ego_nerv_l0_scaffold_design_20260520T140000Z.md"
    ),
    archive_grammar=(
        "ego_nerv_phase_a_monolithic_singlefile_0bin (format ID 0x68): "
        "16-byte header + 5 length-prefixed sections (decoder_blob brotli "
        "int8 + scale_table fp16 + latent_blob brotli uint8 asym-delta + "
        "pose_table fp16 + sidecar empty Phase A); schema "
        "ARCHIVE_GRAMMAR_EGO_NERV in src/tac/ego_nerv_as_renderer.py"
    ),
    parser_section_manifest={
        "header": "16_byte_fixed_eNRV_magic_v1",
        "decoder_blob": "brotli_int8_codes_schema_driven",
        "scale_table": "fp16_raw_one_per_schema_entry",
        "latent_blob": "brotli_uint8_asym_delta_split",
        "pose_table": "fp16_raw_n_pairs_x_pose_dim",
        "sidecar_blob": "brotli_optional_phase_b_empty_phase_a",
    },
    inflate_runtime_loc_budget=200,
    runtime_dep_closure=("torch>=2.5,<2.7", "brotli", "av"),
    export_format="custom",
    score_aware_loss="scorer_loss_terms_btchw",
    bolt_on_loc_budget=1200,
    no_op_detector_planned=True,
    archive_bytes_added=None,
    score_improvement_mechanism_status="RESEARCH_ONLY",
    runtime_overlay_consumed=False,
    recipe_smoke_only=False,
    recipe_research_only=True,
    recipe_min_smoke_gpu="A10G",
    recipe_min_vram_gb=22,
    recipe_pyav_decode_strategy="cpu_thread_async_upload",
    recipe_canary_status="post_canary_dependent",
    recipe_video_input_strategy="per_dispatch_local_copy",
    recipe_canary_dependency="sane_hnerv",
    cost_band_epochs=100,
    cost_band_gpu_key="A10G",
    cost_band_platform_key="modal",
    cost_band_p50_usd=1.10,
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
            "fp16 brotli on weight blocks via ego_nerv_as_renderer library; "
            "no per-tensor bit allocator at scaffold posture"
        ),
        "hook_continual_learning_anchor_kind": (
            "L0 SCAFFOLD: no posterior anchor until full path lands and a "
            "[contest-CUDA] anchor is measured"
        ),
        "hook_probe_disambiguator": (
            "single mechanism (egocentric pose FiLM conditioning); no 2+ "
            "defensible interpretations at scaffold posture"
        ),
    },
)


@register_substrate(EGO_NERV_SUBSTRATE_CONTRACT)
def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.smoke:
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
