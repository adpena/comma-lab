# SPDX-License-Identifier: MIT
"""Train the pact_nerv_distilled_scorer substrate L0 SCAFFOLD.

WAVE-3-PACT-NERV-G2-MID-LOC-L0-BUILD 2026-05-20. Variant #6 of PACT-NERV-ULTIMATE
(commit `e3ad4243a`) Group 2 mid-LOC apparatus-aligned variants.

Distinguishing primitive: Hinton-distilled internal scorer surrogate
(arXiv:1503.02531; KL-T=2.0). At Stage 1 dispatch the surrogate is co-trained
to mimic frozen SegNet + PoseNet logits; its globally-pooled feature vector
then conditions the HNeRV decoder via per-block channel-bias projection.

This trainer's `SubstrateContract` declares `research_only=True`;
`dispatch_enabled: false` on the matching recipe per CLAUDE.md "Substrate
scaffolds MUST be COMPLETE or RESEARCH-ONLY" + Catalog #240 + #315 + #325.

Usage (smoke; CPU)::

    .venv/bin/python experiments/train_substrate_pact_nerv_distilled_scorer.py \\
        --output-dir experiments/results/pact_nerv_distilled_scorer_smoke_<utc> \\
        --epochs 2 --device cpu --smoke
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
SUBSTRATE_TAG = "pact_nerv_distilled_scorer"


TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--video-path": {
        "env": "PACT_NERV_DISTILLED_SCORER_VIDEO_PATH",
        "rationale": (
            "score-aware substrate MUST train against the contest video; "
            "synthetic data is FORBIDDEN outside --smoke"
        ),
        "default": str(DEFAULT_VIDEO_PATH.relative_to(REPO_ROOT)),
        "satisfied_by_profile": (),
        "requires": (),
        "required_input_file": True,
        "generator_command": "contest-pinned upstream snapshot",
    },
    "--output-dir": {
        "env": "PACT_NERV_DISTILLED_SCORER_OUTPUT_DIR",
        "rationale": "custody location for checkpoints + archive + provenance",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--epochs": {
        "env": "PACT_NERV_DISTILLED_SCORER_EPOCHS",
        "rationale": "Stage 1 dispatch operator-gated before non-smoke",
        "default": "2",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--upstream-dir": {
        "env": "PACT_NERV_DISTILLED_SCORER_UPSTREAM_DIR",
        "rationale": "upstream/ root for scorer weights + auth eval",
        "default": str(DEFAULT_UPSTREAM_DIR.relative_to(REPO_ROOT)),
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--device": {
        "env": "PACT_NERV_DISTILLED_SCORER_DEVICE",
        "rationale": (
            "compute device; cuda required for full training; cpu permitted "
            "only with --smoke; MPS refused per CLAUDE.md MPS-NOISE rule"
        ),
        "default": "cpu",
        "satisfied_by_profile": (),
        "requires": (),
    },
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_substrate_pact_nerv_distilled_scorer",
        description=(
            "Train pact_nerv_distilled_scorer L0 SCAFFOLD (smoke only; full path "
            "gated by Catalog #240 + #315 + #325)."
        ),
    )
    p.add_argument("--video-path", type=Path, default=DEFAULT_VIDEO_PATH)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260520)
    p.add_argument("--latent-dim", type=int, default=8)
    p.add_argument("--surrogate-hidden", type=int, default=8)
    p.add_argument("--surrogate-feature-dim", type=int, default=4)
    p.add_argument("--distill-temperature", type=float, default=2.0)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--enable-autocast-fp16", action="store_true", default=False)
    p.add_argument("--enable-torch-compile", action="store_true", default=False)
    p.add_argument("--enable-gt-scorer-cache", action="store_true", default=False)
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
    """Tiny CPU smoke exercising the substrate library."""
    import torch

    from tac.differentiable_eval_roundtrip import patch_upstream_yuv6_globally
    from tac.substrates.pact_nerv_distilled_scorer.architecture import (
        PactNervDistilledScorerConfig,
        PactNervDistilledScorerSubstrate,
    )

    _pin_seeds(args.seed)
    device = _device_or_die(args.device, smoke=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = _utc_now_iso()

    # Per CLAUDE.md eval_roundtrip non-negotiable: patch BEFORE any scorer.
    patch_upstream_yuv6_globally()

    cfg = PactNervDistilledScorerConfig(
        latent_dim=args.latent_dim,
        embed_dim=16,
        initial_grid_h=3,
        initial_grid_w=4,
        decoder_channels=(16, 12, 8),
        sin_frequency=30.0,
        num_upsample_blocks=3,
        surrogate_hidden=args.surrogate_hidden,
        surrogate_feature_dim=args.surrogate_feature_dim,
        distill_temperature=args.distill_temperature,
        num_pairs=max(2, args.batch_size),
        output_height=24,
        output_width=32,
    )
    model = PactNervDistilledScorerSubstrate(cfg).to(device)
    n_params = model.num_parameters()
    n_surr_params = model.num_surrogate_parameters()
    print(
        f"[smoke] pact_nerv_distilled_scorer params: {n_params:,} (surrogate: "
        f"{n_surr_params:,}; ~{100*n_surr_params/max(n_params, 1):.1f}% of total) "
        f"device={device} T={cfg.distill_temperature}"
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = max(1, min(args.epochs, 3))
    for step in range(epochs):
        idx = torch.arange(cfg.num_pairs, device=device, dtype=torch.long)
        rgb_0, rgb_1 = model(idx)
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
        env_var_candidates=(
            "PACT_NERV_DISTILLED_SCORER_GPU",
            "MODAL_GPU",
        ),
    )
    provenance = {
        "schema": "pact_nerv_distilled_scorer_l0_scaffold_smoke_v1",
        "generated_at": _utc_now_iso(),
        "started_at": started_at,
        "git_head": _git_head_sha(),
        "trainer": "experiments/train_substrate_pact_nerv_distilled_scorer.py",
        "lane_id": "lane_pact_nerv_distilled_scorer_l0_scaffold_20260520",
        "substrate_tag": SUBSTRATE_TAG,
        "args": {
            k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
        },
        "pytorch_version": _canonical_torch_version_string(),
        "device": str(device),
        "hardware_substrate_detected": detected_substrate,
        "n_params": int(n_params),
        "n_surrogate_params": int(n_surr_params),
        "distill_temperature": cfg.distill_temperature,
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
    """Full training entry point — REFUSED at L0 SCAFFOLD posture.

    Reactivation criteria (per HNeRV parity discipline L2 export-first):
    1. PACT-NERV symposium Stage 1 dispatch operator-gated approval per
       CLAUDE.md "PER-SUBSTRATE OPTIMAL FORM via adversarial grand
       council symposium" (Catalog #325).
    2. Cargo-cult audit per Catalog #303 surfaces the CARGO-CULTED choices
       documented in `src/tac/substrates/pact_nerv_distilled_scorer/__init__.py`.
    3. 9-dim checklist evidence per Catalog #294 + observability surface
       per Catalog #305 + Dykstra feasibility predicted-band per Catalog #296.
    4. Score-aware training loop with EMA + KL-T=2.0 distillation + canonical
       auth-eval helper invocation per Catalog #226.
    5. Operator-frontier-override per Catalog #300 Mission alignment
       Consequence 1 OR Stage 1 approval converts `research_only=true` to
       `false` in the recipe + this trainer's `SubstrateContract`.
    """
    raise NotImplementedError(
        "[pact_nerv_distilled_scorer] full training path is OPERATOR-GATED per "
        "Catalog #240 + #315 + #325. This is an L0 SCAFFOLD trainer; substrate "
        "is research_only until PACT-NERV symposium Stage 1 dispatch lands."
    )


PACT_NERV_DISTILLED_SCORER_SUBSTRATE_CONTRACT = SubstrateContract(
    id="pact_nerv_distilled_scorer",
    lane_id="lane_pact_nerv_distilled_scorer_l0_scaffold_20260520",
    target_modes=("research_substrate",),
    deployment_target="t4_contest_runtime",
    council_verdict_provenance=(
        ".omx/research/pact_nerv_ultimate_research_and_design_"
        "20260520T193443Z.md"
    ),
    archive_grammar=(
        "PDS monolithic single-file 0.bin (21-byte header; base + surrogate "
        "decoder state_dict in single brotli blob; int16 latents; utf-8 json meta)"
    ),
    parser_section_manifest={
        "header": "21_byte_fixed_PDS_magic_v1",
        "decoder_blob": "brotli_quality9_pickled_fp16_base_plus_distilled_surrogate",
        "surrogate_weights_subset": "logical_grouping_inside_decoder_blob_surrogate_prefix",
        "latent_blob": "raw_int16_row_major",
        "meta_blob": "utf8_json_meta",
    },
    inflate_runtime_loc_budget=150,
    runtime_dep_closure=("torch>=2.5,<2.7", "brotli", "av"),
    export_format="custom",
    score_aware_loss="scorer_loss_terms_btchw",
    bolt_on_loc_budget=350,
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
    cost_band_p50_usd=0.40,
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
        "consumes_scorer_response_dataset_nonpromotional",
    ),
    hook_not_applicable_rationale={
        "hook_sensitivity_contribution": (
            "L0 SCAFFOLD; no sensitivity signal until Stage 1 + full path"
        ),
        "hook_bit_allocator_class": (
            "fp16 brotli on base+surrogate weight blob; no per-tensor bit "
            "allocator at scaffold posture"
        ),
        "hook_continual_learning_anchor_kind": (
            "L0 SCAFFOLD: no posterior anchor until [contest-CUDA] at Stage 1"
        ),
        "hook_probe_disambiguator": (
            "single mechanism (Hinton KL-T=2.0 distillation); distill-vs-direct "
            "is the Stage 1 dispatch's empirical purpose"
        ),
    },
)


@register_substrate(PACT_NERV_DISTILLED_SCORER_SUBSTRATE_CONTRACT)
def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.smoke:
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
