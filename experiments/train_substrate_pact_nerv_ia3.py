# SPDX-License-Identifier: MIT
"""Train the pact_nerv_ia3 substrate L0 SCAFFOLD (WAVE-3-PACT-NERV-IA3-L0-BUILD-STAGE-1 2026-05-20).

Operator-callable training scaffold per the PACT-NERV-DESIGN-SYMPOSIUM
HYBRID Stage 1 verdict 2026-05-20 (commit `5371d4dd4`). SCAFFOLD-LEVEL:
``_smoke_main`` exercises the substrate package; ``_full_main`` raises
``NotImplementedError`` per CLAUDE.md "Substrate scaffolds MUST be COMPLETE
or RESEARCH-ONLY" non-negotiable + Catalog #220 (substrate L1+ scaffold
operational mechanism declaration) + Catalog #240 (recipe-vs-trainer-state
consistency) + Catalog #315 (OPTIMAL FORM before paid dispatch) + Catalog
#325 (per-substrate symposium).

This trainer's ``SubstrateContract`` declares ``research_only=True``;
``dispatch_enabled: false`` on the matching recipe ensures no paid GPU
dispatch may fire from this scaffold until Stage 1 dispatch operator-gated.

Architectural twist vs ``boost_nerv`` (canonical sister): pact_nerv_ia3
replaces boost_nerv's iterative-residual chain with IA3 γ-only ego-pose-
conditioned per-block modulation (Liu et al. 2022 arXiv:2205.05638). The
distinguishing primitive: element-wise learnable γ rescaling (~6x more
parameter-efficient than full FiLM γ+β) with NO β bias projection.

Hypothesis (per PACT-NERV symposium Stage 1 verdict): for the contest rate
term, γ-only halves conditioning bytes vs γ+β. The empirical question
Stage 1 tests: does the β term carry significant per-frame signal on our
specific driving video?

Literature anchor: Liu et al. 2022 "IA3: Infused Adapter by Inhibiting
and Amplifying Inner Activations", arXiv:2205.05638. FILM-FAMILY-RESEARCH
Section 10 Recommendation #5 cites this as the HARD-EARNED-LITERATURE
rate-extremal variant.

Council-binding contract (CLAUDE.md non-negotiables) honored end-to-end:

- TIER_1_OPERATOR_REQUIRED_FLAGS declared per Catalog #151 for wire-up.
- ``@register_substrate(SubstrateContract(...))`` per Catalog #241 META layer.
- Canonical scorer/device helpers per Catalog #164 / #205 / #226.
- Smoke uses synthetic targets per Catalog #114 (``--smoke`` opt-in only).
- Non-smoke path raises ``NotImplementedError`` (L0 SCAFFOLD posture);
  reactivation criteria pinned per HNeRV parity L2.
- eval_roundtrip MANDATORY DEFAULT per Catalog #6: smoke patches
  differentiable yuv6 BEFORE any future scorer construction.

Usage (smoke; CPU, tiny config, ~2 steps, synthetic batches)::

    .venv/bin/python experiments/train_substrate_pact_nerv_ia3.py \\
        --output-dir experiments/results/pact_nerv_ia3_smoke_<utc> \\
        --epochs 2 --device cpu --smoke

Usage (full; refused at L0 SCAFFOLD until Stage 1 operator dispatch lands)::

    # raises NotImplementedError per Catalog #240 + #315 + #325

Cross-ref:
    src/tac/substrates/pact_nerv_ia3/ (substrate package)
    experiments/train_substrate_boost_nerv.py (canonical sister L0 SCAFFOLD pattern)
    experiments/train_substrate_ds_nerv.py (canonical base decoder sister)
    .omx/operator_authorize_recipes/substrate_pact_nerv_ia3_modal_t4_dispatch.yaml
    .omx/research/pact_nerv_ia3_l0_scaffold_design_20260520T<UTC>.md
    .omx/research/council_per_substrate_symposium_pact_nerv_*_20260520T185500Z.md
    .omx/research/film_family_alternatives_bleeding_edge_research_20260520T184150Z.md
"""
# AUTOCAST_FP16_WAIVED:l0-scaffold-no-full-training-path-stage-1-operator-gated
# TORCH_COMPILE_WAIVED:l0-scaffold-no-full-training-path-stage-1-operator-gated
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
SUBSTRATE_TAG = "pact_nerv_ia3"


# ---------------------------------------------------------------------------
# Catalog #151 manifest - every flag below must be threaded by any operator
# wrapper that subprocess-invokes this trainer. Schema mirrors the canonical
# boost_nerv L0 SCAFFOLD manifest per Catalog #151 + #168 AnnAssign-aware extractor.
# ---------------------------------------------------------------------------
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--video-path": {
        "env": "PACT_NERV_IA3_VIDEO_PATH",
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
            ".omx/research/pact_nerv_ia3_l0_scaffold_design_20260520T<UTC>.md"
        ),
    },
    "--output-dir": {
        "env": "PACT_NERV_IA3_OUTPUT_DIR",
        "rationale": "custody location for checkpoints + archive + provenance",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--epochs": {
        "env": "PACT_NERV_IA3_EPOCHS",
        "rationale": (
            "pact_nerv_ia3 substrate engineering pass; council Stage 1 dispatch "
            "operator-gated before non-smoke training is unlocked"
        ),
        "default": "2",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--upstream-dir": {
        "env": "PACT_NERV_IA3_UPSTREAM_DIR",
        "rationale": (
            "upstream/ root for scorer weights + evaluate.py; required for full "
            "training (non-smoke) and auth eval"
        ),
        "default": str(DEFAULT_UPSTREAM_DIR.relative_to(REPO_ROOT)),
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--device": {
        "env": "PACT_NERV_IA3_DEVICE",
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
        prog="train_substrate_pact_nerv_ia3",
        description=(
            "Train pact_nerv_ia3 substrate L0 SCAFFOLD (smoke only; full path "
            "gated by PACT-NERV symposium Stage 1 dispatch per Catalog "
            "#240 + #315 + #325)."
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
        "--pose-dim", type=int, default=6,
        help=(
            "Ego-pose conditioning dimensionality (contest canonical: 6 per "
            "upstream/modules.py PoseNet first 6 dims)."
        ),
    )
    p.add_argument(
        "--ia3-init-delta-std", type=float, default=0.01,
        help=(
            "Initialization stddev for IA3 γ_proj weights (γ_init=1.0 + Δ "
            "residual form per IA3 paper §3.2)."
        ),
    )
    p.add_argument(
        "--device", choices=["cuda", "cpu"], default="cpu",
        help=(
            "Compute device. L0 SCAFFOLD smoke runs on CPU; cuda permitted "
            "only for future Stage 1 dispatch. MPS rejected per CLAUDE.md "
            "'MPS auth eval is NOISE'."
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
        help="RESERVED (Stage 1): Wrap forward in torch.autocast(fp16).",
    )
    p.add_argument(
        "--enable-torch-compile", action="store_true", default=False,
        help="RESERVED (Stage 1): Wrap substrate with torch.compile / Inductor.",
    )
    p.add_argument(
        "--enable-gt-scorer-cache", action="store_true", default=False,
        help="RESERVED (Stage 1): GT-scorer-output cache (Catalog #228).",
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
    """Tiny CPU smoke exercising the pact_nerv_ia3 substrate library.

    Builds a tiny ``PactNervIa3Substrate``, runs a forward-pass shape check
    on synthetic batches (synthetic targets OK because ``--smoke`` per
    Catalog #114), and writes a smoke checkpoint + provenance. No scorer
    load; no archive build; no auth-eval.

    Per CLAUDE.md "eval_roundtrip - NON-NEGOTIABLE, HIGHEST EMPHASIS" +
    Catalog #6 MANDATORY DEFAULT: calls patch_upstream_yuv6_globally()
    BEFORE any scorer construction so the future Stage 1 PoseNet gradient
    path remains differentiable.
    """
    import torch

    from tac.differentiable_eval_roundtrip import patch_upstream_yuv6_globally
    from tac.substrates.pact_nerv_ia3.architecture import (
        PactNervIa3Config,
        PactNervIa3Substrate,
    )

    _pin_seeds(args.seed)
    device = _device_or_die(args.device, smoke=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = _utc_now_iso()

    # CLAUDE.md "eval_roundtrip - NON-NEGOTIABLE": patch upstream rgb_to_yuv6
    # BEFORE any scorer construction. Smoke does not load scorers, but the
    # patch is idempotent and surfaces any installation issue early. This
    # ensures Stage 1 dispatches inherit the patched path automatically.
    patch_upstream_yuv6_globally()

    cfg = PactNervIa3Config(
        latent_dim=args.latent_dim,
        embed_dim=16,
        initial_grid_h=3,
        initial_grid_w=4,
        decoder_channels=(16, 12, 8),
        sin_frequency=30.0,
        num_upsample_blocks=3,
        pose_dim=args.pose_dim,
        ia3_init_delta_std=args.ia3_init_delta_std,
        num_pairs=max(2, args.batch_size),
        output_height=24,
        output_width=32,
    )
    model = PactNervIa3Substrate(cfg).to(device)
    n_params = model.num_parameters()
    n_ia3_params = model.num_ia3_modulation_parameters()
    print(
        f"[smoke] pact_nerv_ia3 params: {n_params:,} (IA3 modulation: "
        f"{n_ia3_params:,}; ~{100*n_ia3_params/max(n_params, 1):.1f}% of total) "
        f"device={device} pose_dim={cfg.pose_dim}"
    )

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
        env_var_candidates=("PACT_NERV_IA3_GPU", "MODAL_GPU"),
    )
    provenance = {
        "schema": "pact_nerv_ia3_l0_scaffold_smoke_v1",
        "generated_at": _utc_now_iso(),
        "started_at": started_at,
        "git_head": _git_head_sha(),
        "trainer": "experiments/train_substrate_pact_nerv_ia3.py",
        "lane_id": "lane_pact_nerv_ia3_l0_scaffold_20260520",
        "substrate_tag": SUBSTRATE_TAG,
        "args": {
            k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
        },
        "pytorch_version": _canonical_torch_version_string(),
        "device": str(device),
        "hardware_substrate_detected": detected_substrate,
        "n_params": int(n_params),
        "n_ia3_modulation_params": int(n_ia3_params),
        "pose_dim": cfg.pose_dim,
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
    scaffold trainer's full path is council-gated.

    Per the PACT-NERV-DESIGN-SYMPOSIUM HYBRID Stage 1 verdict (commit
    `5371d4dd4`): Stage 1 = Pact-NeRV-IA3 ~$0.30 Modal T4 single-primitive
    γ-only rate-extremal smoke (the cheapest empirical experiment). This
    L0 SCAFFOLD lands the substrate package + trainer + recipe + driver
    + tests; the actual $0.30 Stage 1 dispatch is operator-gated.

    Reactivation criteria (per HNeRV parity discipline L2 export-first):
    1. PACT-NERV symposium Stage 1 dispatch operator-gated approval per
       CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM via adversarial grand
       council symposium" (Catalog #325).
    2. Cargo-cult audit per Catalog #303: the ``CARGO-CULTED`` choices
       documented in ``src/tac/substrates/pact_nerv_ia3/__init__.py``
       (per-block modulation; shared γ_proj across frame_0/frame_1) MUST
       be challenged with empirical sweeps OR reclassified as HARD-EARNED.
    3. 9-dim checklist evidence per Catalog #294 + observability surface
       per Catalog #305 + Dykstra feasibility predicted-band per Catalog
       #296 all land in the design memo.
    4. Score-aware training loop with EMA + score-domain Lagrangian +
       canonical auth-eval helper invocation per Catalog #226. The
       PactNervIa3ScoreAwareLoss helper at
       ``src/tac/substrates/pact_nerv_ia3/score_aware_loss.py`` provides
       the loss surface; the trainer's full path must wire it in.
    5. Operator-frontier-override per Catalog #300 Mission alignment
       Consequence 1 OR Stage 1 approval converts ``research_only=true``
       to ``false`` in the recipe + this trainer's ``SubstrateContract``.
    """
    raise NotImplementedError(
        "[pact_nerv_ia3] full training path is OPERATOR-GATED per Catalog #240 + "
        "#315 + #325. This is an L0 SCAFFOLD trainer; substrate is research_only "
        "until PACT-NERV symposium Stage 1 dispatch lands. "
        "See reactivation criteria in this function's docstring + "
        ".omx/research/pact_nerv_ia3_l0_scaffold_design_20260520T*.md "
        "+ lane_pact_nerv_ia3_l0_scaffold_20260520 in .omx/state/lane_registry.json."
    )


# ---------------------------------------------------------------------------
# META layer SubstrateContract (Catalog #241/#242)
# ---------------------------------------------------------------------------

PACT_NERV_IA3_SUBSTRATE_CONTRACT = SubstrateContract(
    id="pact_nerv_ia3",
    lane_id="lane_pact_nerv_ia3_l0_scaffold_20260520",
    target_modes=("research_substrate",),
    deployment_target="t4_contest_runtime",
    council_verdict_provenance=(
        ".omx/research/council_per_substrate_symposium_pact_nerv_"
        "score_axis_aware_foveated_ego_motion_full_stack_synergy_"
        "eval_roundtrip_20260520T185500Z.md"
    ),
    archive_grammar=(
        "PIA3 monolithic single-file 0.bin (26-byte header carrying "
        "POSE_DIM u8 distinctive field + EGO_POSE_BLOB_LEN u32; base "
        "+ IA3 γ_proj head weights in single brotli blob; int16 "
        "latents + int16 ego_poses; utf-8 json meta)"
    ),
    parser_section_manifest={
        "header": "26_byte_fixed_PIA3_magic_v1_pose_dim_ego_pose_blob_len",
        "decoder_blob": "brotli_quality9_pickled_fp16_base_plus_ia3_gamma_proj_heads",
        "ia3_gamma_proj_heads_subset": "logical_grouping_inside_decoder_blob",
        "base_decoder_weights_subset": "logical_grouping_inside_decoder_blob",
        "latent_blob": "raw_int16_row_major",
        "ego_pose_blob": "raw_int16_row_major",
        "meta_blob": "utf8_json_meta",
    },
    inflate_runtime_loc_budget=150,
    runtime_dep_closure=("torch>=2.5,<2.7", "brotli", "av"),
    export_format="custom",
    score_aware_loss="scorer_loss_terms_btchw",
    bolt_on_loc_budget=600,
    no_op_detector_planned=True,
    archive_bytes_added=None,
    score_improvement_mechanism_status="RESEARCH_ONLY",
    runtime_overlay_consumed=False,
    recipe_smoke_only=False,
    recipe_research_only=True,
    recipe_min_smoke_gpu="T4",
    recipe_min_vram_gb=12,
    recipe_pyav_decode_strategy="cpu_thread_async_upload",
    recipe_canary_status="independent_substrate",
    recipe_video_input_strategy="per_dispatch_local_copy",
    recipe_canary_dependency=None,
    cost_band_epochs=100,
    cost_band_gpu_key="T4",
    cost_band_platform_key="modal",
    cost_band_p50_usd=0.30,
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
        "catalog_325_per_substrate_symposium_stage_1_dispatch_operator_gated",
        "catalog_6_eval_roundtrip_mandatory_default_patched_in_smoke",
    ),
    hook_not_applicable_rationale={
        "hook_sensitivity_contribution": (
            "L0 SCAFFOLD; no sensitivity signal until Stage 1 dispatch + full path"
        ),
        "hook_bit_allocator_class": (
            "fp16 brotli on combined base+IA3-γ_proj weight blob; no per-tensor "
            "bit allocator at scaffold posture (IA3 γ_proj per-head quantization "
            "is the L1+ research path per cargo-cult audit shared-γ_proj alternative)"
        ),
        "hook_continual_learning_anchor_kind": (
            "L0 SCAFFOLD: no posterior anchor until full path lands and a "
            "[contest-CUDA] anchor is measured at Stage 1 dispatch"
        ),
        "hook_probe_disambiguator": (
            "single mechanism (IA3 γ-only modulation); FiLM vs IA3 disambiguation "
            "IS the Stage 1 dispatch's empirical purpose per PACT-NERV symposium "
            "Section 13"
        ),
    },
)


@register_substrate(PACT_NERV_IA3_SUBSTRATE_CONTRACT)
def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.smoke:
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
