# SPDX-License-Identifier: MIT
"""Train the boost_nerv substrate L0 SCAFFOLD (WAVE-3-NERV-LITERATURE-L0-RESCOPED 2026-05-20).

Operator-callable training scaffold per the WAVE-3-NERV-LITERATURE-L0-RESCOPED
queue 2026-05-20. SCAFFOLD-LEVEL: ``_smoke_main`` exercises the substrate
package; ``_full_main`` raises ``NotImplementedError`` per CLAUDE.md
"Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" non-negotiable +
Catalog #220 (substrate L1+ scaffold operational mechanism declaration) +
Catalog #240 (recipe-vs-trainer-state consistency) + Catalog #315 (OPTIMAL
FORM before paid dispatch) + Catalog #325 (per-substrate symposium).

This trainer's ``SubstrateContract`` declares ``research_only=True``;
``dispatch_enabled: false`` on the matching recipe ensures no paid GPU
dispatch may fire from this scaffold.

Architectural twist vs ``ds_nerv`` (canonical sister): boost_nerv adds an
iterative residual-refinement chain (``NUM_BOOSTING_ROUNDS=2`` by default)
on top of the DepthSep base decoder. Each boosting round consumes (rgb_in,
z) and predicts a residual clamped to [-0.1, 0.1] before addition.

Hypothesis (operator's 5-tier fit-ranking HIGH FIT ⭐⭐⭐⭐⭐): driving video
has heavy-tailed reconstruction error; a boosting chain progressively
reduces worst-case error without inflating the base substrate's parameter
budget. The boosting paradigm composes WITH any base substrate.

Literature anchor: Liu et al. ECCV 2024 "BoostNeRV: Iterative Refinement
for Implicit Neural Video Representations".

Council-binding contract (CLAUDE.md non-negotiables) honored end-to-end:

- TIER_1_OPERATOR_REQUIRED_FLAGS declared per Catalog #151 for wire-up.
- ``@register_substrate(SubstrateContract(...))`` per Catalog #241 META layer.
- Canonical scorer/device helpers per Catalog #164 / #205 / #226.
- Smoke uses synthetic targets per Catalog #114 (``--smoke`` opt-in only).
- Non-smoke path raises ``NotImplementedError`` (L0 SCAFFOLD posture);
  reactivation criteria pinned in lane registry notes per HNeRV parity L2.

Usage (smoke; CPU, tiny config, ~2 steps, synthetic batches)::

    .venv/bin/python experiments/train_substrate_boost_nerv.py \\
        --output-dir experiments/results/boost_nerv_smoke_<utc> \\
        --epochs 2 --device cpu --smoke

Usage (full; refused at L0 SCAFFOLD until Phase 2 council convocation)::

    # raises NotImplementedError per Catalog #240 + #315 + #325

Cross-ref:
    src/tac/substrates/boost_nerv/ (substrate package)
    experiments/train_substrate_ds_nerv.py (canonical sister trainer)
    experiments/train_substrate_ego_nerv.py (sister L0 SCAFFOLD pattern)
    .omx/operator_authorize_recipes/substrate_boost_nerv_modal_t4_dispatch.yaml
    .omx/research/boost_nerv_l0_scaffold_design_20260520T184500Z.md
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
SUBSTRATE_TAG = "boost_nerv"


# ---------------------------------------------------------------------------
# Catalog #151 manifest - every flag below must be threaded by any operator
# wrapper that subprocess-invokes this trainer. Schema mirrors the canonical
# ego_nerv L0 SCAFFOLD manifest per Catalog #151 + #168 AnnAssign-aware extractor.
# ---------------------------------------------------------------------------
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--video-path": {
        "env": "BOOST_NERV_VIDEO_PATH",
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
            ".omx/research/boost_nerv_l0_scaffold_design_20260520T184500Z.md"
        ),
    },
    "--output-dir": {
        "env": "BOOST_NERV_OUTPUT_DIR",
        "rationale": "custody location for checkpoints + archive + provenance",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--epochs": {
        "env": "BOOST_NERV_EPOCHS",
        "rationale": (
            "boost_nerv substrate engineering pass; council Phase 2 review pending "
            "before non-smoke training is unlocked"
        ),
        "default": "2",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--upstream-dir": {
        "env": "BOOST_NERV_UPSTREAM_DIR",
        "rationale": (
            "upstream/ root for scorer weights + evaluate.py; required for full "
            "training (non-smoke) and auth eval"
        ),
        "default": str(DEFAULT_UPSTREAM_DIR.relative_to(REPO_ROOT)),
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--device": {
        "env": "BOOST_NERV_DEVICE",
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
        prog="train_substrate_boost_nerv",
        description=(
            "Train boost_nerv substrate L0 SCAFFOLD (smoke only; full path "
            "gated by Phase 2 council per Catalog #240 + #315 + #325)."
        ),
    )
    p.add_argument(
        "--video-path", type=Path, default=DEFAULT_VIDEO_PATH,
        help="Path to upstream/videos/0.mkv (contest video; non-smoke required).",
    )
    p.add_argument(
        "--output-dir", type=Path, required=True,
        help="Where to write checkpoints + manifest + archive.",
    )
    p.add_argument(
        "--epochs", type=int, default=2,
        help="Number of training epochs (smoke default 2; non-smoke gated).",
    )
    p.add_argument(
        "--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR,
        help="upstream/ root; required for scorer load + auth eval (non-smoke).",
    )
    p.add_argument(
        "--batch-size", type=int, default=2,
        help="Number of pair indices per batch (smoke default 2).",
    )
    p.add_argument(
        "--seed", type=int, default=20260520,
        help="Manual seed for torch / numpy / random (deterministic).",
    )
    p.add_argument(
        "--latent-dim", type=int, default=8,
        help="Per-pair latent dimensionality (smoke uses tiny default).",
    )
    p.add_argument(
        "--num-boosting-rounds", type=int, default=2,
        help="Iterative residual rounds (BoostNeRV distinctive parameter; smoke=2).",
    )
    p.add_argument(
        "--boosting-gain-clamp", type=float, default=0.1,
        help="Per-round residual gain clamp magnitude.",
    )
    p.add_argument(
        "--device", choices=["cuda", "cpu"], default="cpu",
        help=(
            "Compute device. L0 SCAFFOLD smoke runs on CPU; cuda permitted "
            "only for future Phase 2 full-mode dispatch. MPS rejected per "
            "CLAUDE.md 'MPS auth eval is NOISE'."
        ),
    )
    p.add_argument(
        "--smoke", action="store_true",
        help=(
            "Tiny CPU smoke (synthetic targets OK because --smoke; never use "
            "this output for ranking per Catalog #114)."
        ),
    )
    p.add_argument(
        "--enable-autocast-fp16", action="store_true", default=False,
        help="RESERVED (Phase 2): Wrap forward in torch.autocast(fp16).",
    )
    p.add_argument(
        "--enable-torch-compile", action="store_true", default=False,
        help="RESERVED (Phase 2): Wrap substrate with torch.compile / Inductor.",
    )
    p.add_argument(
        "--enable-gt-scorer-cache", action="store_true", default=False,
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
    """Tiny CPU smoke exercising the boost_nerv substrate library.

    Builds a tiny ``BoostnervSubstrate``, runs a forward-pass shape check
    on synthetic batches (synthetic targets OK because ``--smoke`` per
    Catalog #114), and writes a smoke checkpoint + provenance. No scorer
    load; no archive build; no auth-eval.
    """
    import torch

    from tac.differentiable_eval_roundtrip import patch_upstream_yuv6_globally
    from tac.substrates.boost_nerv.architecture import (
        BoostnervConfig,
        BoostnervSubstrate,
    )

    _pin_seeds(args.seed)
    device = _device_or_die(args.device, smoke=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = _utc_now_iso()

    # CLAUDE.md "eval_roundtrip - NON-NEGOTIABLE": patch upstream rgb_to_yuv6
    # BEFORE any scorer construction. Smoke does not load scorers, but the
    # patch is idempotent and surfaces any installation issue early.
    patch_upstream_yuv6_globally()

    cfg = BoostnervConfig(
        latent_dim=args.latent_dim,
        embed_dim=16,
        initial_grid_h=3,
        initial_grid_w=4,
        decoder_channels=(16, 12, 8),
        sin_frequency=30.0,
        num_upsample_blocks=3,
        num_boosting_rounds=args.num_boosting_rounds,
        boosting_gain_clamp=args.boosting_gain_clamp,
        boosting_hidden_dim=8,
        num_pairs=max(2, args.batch_size),
        output_height=24,
        output_width=32,
    )
    model = BoostnervSubstrate(cfg).to(device)
    n_params = model.num_parameters()
    print(f"[smoke] boost_nerv params: {n_params:,} device={device} "
          f"num_boosting_rounds={cfg.num_boosting_rounds}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = max(1, min(args.epochs, 3))
    for step in range(epochs):
        idx = torch.arange(cfg.num_pairs, device=device, dtype=torch.long)
        rgb_0, rgb_1 = model(idx)
        # Smoke surrogate loss (NOT score-aware): mean magnitude.
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
        env_var_candidates=("BOOST_NERV_GPU", "MODAL_GPU"),
    )
    provenance = {
        "schema": "boost_nerv_l0_scaffold_smoke_v1",
        "generated_at": _utc_now_iso(),
        "started_at": started_at,
        "git_head": _git_head_sha(),
        "trainer": "experiments/train_substrate_boost_nerv.py",
        "lane_id": "lane_boost_nerv_l0_scaffold_20260520",
        "substrate_tag": SUBSTRATE_TAG,
        "args": {
            k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
        },
        "pytorch_version": _canonical_torch_version_string(),
        "device": str(device),
        "hardware_substrate_detected": detected_substrate,
        "n_params": int(n_params),
        "num_boosting_rounds": cfg.num_boosting_rounds,
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
    2. Cargo-cult audit per Catalog #303: in particular the ``CARGO-CULTED``
       choices documented in ``src/tac/substrates/boost_nerv/__init__.py``
       (num_boosting_rounds=2; gain clamp [-0.1, 0.1]; shared latent
       across rounds) MUST be challenged with empirical sweeps OR
       reclassified as HARD-EARNED with citation.
    3. 9-dim checklist evidence per Catalog #294 + observability surface
       per Catalog #305 + Dykstra feasibility predicted-band per Catalog
       #296 all land in the design memo.
    4. Score-aware training loop with EMA + score-domain Lagrangian +
       canonical auth-eval helper invocation per Catalog #226.
    5. Operator-frontier-override per Catalog #300 Mission alignment
       Consequence 1 OR Phase 2 council approval converts
       ``research_only=true`` to ``false`` in the recipe + this
       trainer's ``SubstrateContract``.
    """
    raise NotImplementedError(
        "[boost_nerv] full training path is OPERATOR-GATED per Catalog #240 + "
        "#315 + #325. This is an L0 SCAFFOLD trainer; substrate is research_only "
        "until the Phase 2 council symposium + score-aware Lagrangian wire-in land. "
        "See reactivation criteria in this function's docstring + "
        ".omx/research/boost_nerv_l0_scaffold_design_20260520T184500Z.md "
        "+ lane_boost_nerv_l0_scaffold_20260520 in .omx/state/lane_registry.json."
    )


# ---------------------------------------------------------------------------
# META layer SubstrateContract (Catalog #241/#242)
# ---------------------------------------------------------------------------

BOOST_NERV_SUBSTRATE_CONTRACT = SubstrateContract(
    id="boost_nerv",
    lane_id="lane_boost_nerv_l0_scaffold_20260520",
    target_modes=("research_substrate",),
    deployment_target="t4_contest_runtime",
    council_verdict_provenance=(
        ".omx/research/boost_nerv_l0_scaffold_design_20260520T184500Z.md"
    ),
    archive_grammar=(
        "BSV1 monolithic single-file 0.bin (22-byte header carrying "
        "NUM_BOOSTING_ROUNDS u8 distinctive field; base + boosting "
        "decoder weights in single brotli blob; int16 latents; utf-8 json meta)"
    ),
    parser_section_manifest={
        "header": "22_byte_fixed_BSV1_magic_v1_num_boosting_rounds",
        "decoder_blob": "brotli_quality9_pickled_fp16_base_plus_boosting_heads",
        "boosting_residual_heads_subset": "logical_grouping_inside_decoder_blob",
        "base_decoder_weights_subset": "logical_grouping_inside_decoder_blob",
        "latent_blob": "raw_int16_row_major",
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
            "fp16 brotli on combined base+boosting weight blob; no per-tensor "
            "bit allocator at scaffold posture (boosting heads quantization "
            "is the L1+ research path per cargo-cult audit)"
        ),
        "hook_continual_learning_anchor_kind": (
            "L0 SCAFFOLD: no posterior anchor until full path lands and a "
            "[contest-CUDA] anchor is measured"
        ),
        "hook_probe_disambiguator": (
            "single mechanism (iterative residual chain); boosting paradigm "
            "alternatives (per-round latents vs shared latent) deferred to L1+ "
            "sweep design"
        ),
    },
)


@register_substrate(BOOST_NERV_SUBSTRATE_CONTRACT)
def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.smoke:
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
