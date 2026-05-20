# SPDX-License-Identifier: MIT
"""Train pact_nerv_neural_codec_e2e_cross L0 SCAFFOLD (WAVE-3-PACT-NERV-G4-CROSS-CODEC 2026-05-20).

Group 4 CROSS-CODEC composition variant per PACT-NERV-ULTIMATE variant #18
(end-to-end neural codec compositing: two HNeRV branches + Ballé-style
hyperprior gate routes per-pair bits between them).

Literature: Ballé et al. 2018 "Variational image compression with a scale
hyperprior" (arXiv:1802.01436) + Atick-Redlich 1990 cooperative-receiver
framing + CROSS-CANDIDATE finding #3 empirical SUPER_ADDITIVE signature.
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
    torch_version_string as _canonical_torch_version_string,
)
from tac.substrates._shared.trainer_skeleton import (
    utc_now_iso as _canonical_utc_now_iso,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIDEO_PATH = REPO_ROOT / "upstream" / "videos" / "0.mkv"
DEFAULT_UPSTREAM_DIR = REPO_ROOT / "upstream"
SUBSTRATE_TAG = "pact_nerv_neural_codec_e2e_cross"


TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--video-path": {
        "env": "PACT_NERV_NCEC_VIDEO_PATH",
        "rationale": "score-aware substrate trains against contest video",
        "default": str(DEFAULT_VIDEO_PATH.relative_to(REPO_ROOT)),
        "satisfied_by_profile": (),
        "requires": (),
        "required_input_file": True,
        "generator_command": "contest-pinned upstream snapshot",
        "rationale_audit": (
            ".omx/research/pact_nerv_neural_codec_e2e_cross_l0_scaffold_design_20260520T<UTC>.md"
        ),
    },
    "--output-dir": {
        "env": "PACT_NERV_NCEC_OUTPUT_DIR",
        "rationale": "checkpoints + archive + provenance",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--epochs": {
        "env": "PACT_NERV_NCEC_EPOCHS",
        "rationale": "substrate engineering pass; Stage 1 operator-gated",
        "default": "2",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--upstream-dir": {
        "env": "PACT_NERV_NCEC_UPSTREAM_DIR",
        "rationale": "upstream/ root for scorer weights",
        "default": str(DEFAULT_UPSTREAM_DIR.relative_to(REPO_ROOT)),
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--device": {
        "env": "PACT_NERV_NCEC_DEVICE",
        "rationale": "cuda required for full; cpu only with --smoke",
        "default": "cpu",
        "satisfied_by_profile": (),
        "requires": (),
    },
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_substrate_pact_nerv_neural_codec_e2e_cross",
        description="Train pact_nerv_neural_codec_e2e_cross L0 SCAFFOLD (smoke only).",
    )
    p.add_argument("--video-path", type=Path, default=DEFAULT_VIDEO_PATH)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260520)
    p.add_argument("--latent-dim-a", type=int, default=8)
    p.add_argument("--latent-dim-b", type=int, default=8)
    p.add_argument("--hyperprior-hidden", type=int, default=16)
    p.add_argument("--gate-init-bias", type=float, default=0.0)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--enable-autocast-fp16", action="store_true", default=False)
    p.add_argument("--enable-torch-compile", action="store_true", default=False)
    p.add_argument("--enable-gt-scorer-cache", action="store_true", default=False)
    return p


def _smoke_main(args: argparse.Namespace) -> int:
    import torch
    from tac.differentiable_eval_roundtrip import patch_upstream_yuv6_globally
    from tac.substrates.pact_nerv_neural_codec_e2e_cross.architecture import (
        PactNervNeuralCodecE2ECrossConfig,
        PactNervNeuralCodecE2ECrossSubstrate,
    )

    _canonical_pin_seeds(args.seed)
    device = _canonical_device_or_die(args.device, smoke=True, substrate_tag=SUBSTRATE_TAG)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = _canonical_utc_now_iso()
    patch_upstream_yuv6_globally()

    cfg = PactNervNeuralCodecE2ECrossConfig(
        latent_dim_a=args.latent_dim_a, latent_dim_b=args.latent_dim_b,
        embed_dim=16, initial_grid_h=3, initial_grid_w=4,
        decoder_channels=(16, 12, 8), sin_frequency=30.0,
        num_upsample_blocks=3, num_pairs=max(2, args.batch_size),
        output_height=24, output_width=32,
        hyperprior_hidden=args.hyperprior_hidden,
        gate_init_bias=args.gate_init_bias,
    )
    model = PactNervNeuralCodecE2ECrossSubstrate(cfg).to(device)
    n_params = model.num_parameters()
    print(
        f"[smoke] pact_nerv_neural_codec_e2e_cross params: {n_params:,} device={device} "
        f"hyperprior_hidden={cfg.hyperprior_hidden} gate_init_bias={cfg.gate_init_bias}"
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
        # Observability: per-pair gate values
        with torch.no_grad():
            gates = model.gate_values(idx)
        print(
            f"[smoke] step {step}: loss={loss.item():.4f} "
            f"gate_mean={gates.mean().item():.3f} gate_min={gates.min().item():.3f} "
            f"gate_max={gates.max().item():.3f}"
        )

    torch.save(
        {
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "config": asdict(cfg), "smoke": True,
        },
        args.output_dir / "smoke_checkpoint.pt",
    )

    detected = _canon_detect_hardware_substrate(
        axis="cpu", substrate_tag=SUBSTRATE_TAG,
        provenance_path=args.output_dir / "provenance.json",
        env_var_candidates=("PACT_NERV_NCEC_GPU", "MODAL_GPU"),
    )
    provenance = {
        "schema": "pact_nerv_neural_codec_e2e_cross_l0_scaffold_smoke_v1",
        "generated_at": _canonical_utc_now_iso(), "started_at": started_at,
        "git_head": _canonical_git_head_sha(REPO_ROOT),
        "trainer": "experiments/train_substrate_pact_nerv_neural_codec_e2e_cross.py",
        "lane_id": "lane_pact_nerv_neural_codec_e2e_cross_l0_scaffold_20260520",
        "substrate_tag": SUBSTRATE_TAG,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "pytorch_version": _canonical_torch_version_string(), "device": str(device),
        "hardware_substrate_detected": detected, "n_params": int(n_params),
        "hyperprior_hidden": cfg.hyperprior_hidden,
        "gate_init_bias": cfg.gate_init_bias,
        "smoke": True,
        "synthetic_targets_used_per_catalog_114": True,
        "score_claim": False, "score_axis_tag": None,
        "promotion_eligible": False, "ready_for_exact_eval_dispatch": False,
        "custody_status": "ci-rebuildable",
        "evidence_grade": "scaffold-smoke-no-score-axis",
    }
    (args.output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )
    return 0


def _full_main(args: argparse.Namespace) -> int:
    """Full path - REFUSED at L0 SCAFFOLD per Catalog #240/#315/#325.

    Reactivation criteria:
    1. PACT-NERV-ULTIMATE STAIRCASE Step 18 dispatch operator-gated per Catalog #325.
    2. Cargo-cult audit per Catalog #303 (hyperprior gate via sigmoid + symmetric
       branches CARGO-CULTED; L1 needs per-region learned gate per Ballé 2018 §3.3
       autoregressive hyperprior + heterogeneous branches per CROSS-CANDIDATE
       finding #3 SUPER_ADDITIVE signature).
    3. 9-dim checklist + observability + Dykstra per Catalog #294/#305/#296.
    4. End-to-end neural-codec composition contract: both branches jointly trained
       + hyperprior gate trained + Catalog #322 SUPER_ADDITIVE empirical proof
       at the gate-bytes byte-mutation surface (Catalog #139).
    5. Score-aware Lagrangian + canonical auth-eval helper per Catalog #226.
    6. Operator-frontier-override per Catalog #300 OR Stage 1 approval.
    """
    raise NotImplementedError(
        "[pact_nerv_neural_codec_e2e_cross] full training path is OPERATOR-GATED "
        "per Catalog #240 + #315 + #325. L0 SCAFFOLD; substrate is research_only "
        "until PACT-NERV-ULTIMATE STAIRCASE Step 18 dispatch operator-gated. "
        "End-to-end neural-codec composition (two HNeRV branches + Ballé "
        "hyperprior gate) needs L1 per-region autoregressive hyperprior per "
        "Ballé 2018 §3.3 + heterogeneous branches per CROSS-CANDIDATE finding #3. "
        "See lane_pact_nerv_neural_codec_e2e_cross_l0_scaffold_20260520 in "
        "lane_registry."
    )


PACT_NERV_NCEC_SUBSTRATE_CONTRACT = SubstrateContract(
    id="pact_nerv_neural_codec_e2e_cross",
    lane_id="lane_pact_nerv_neural_codec_e2e_cross_l0_scaffold_20260520",
    target_modes=("research_substrate",),
    deployment_target="t4_contest_runtime",
    council_verdict_provenance=(
        ".omx/research/pact_nerv_ultimate_research_and_design_20260520T193443Z.md"
    ),
    archive_grammar=(
        "NCEC monolithic single-file 0.bin (35-byte header; "
        "decoder_a + decoder_b + hyperprior gate + per-pair latents A,B; "
        "end-to-end neural-codec composition per PACT-NERV-ULTIMATE Variant #18 + "
        "Ballé 2018 hyperprior + Atick-Redlich 1990 cooperative-receiver)"
    ),
    parser_section_manifest={
        "header": "35_byte_fixed_NCEC_magic_v1_latent_dim_a_b_num_pairs",
        "decoder_a_blob": "brotli_quality9_pickled_fp16_hnerv_branch_a_weights",
        "decoder_b_blob": "brotli_quality9_pickled_fp16_hnerv_branch_b_weights",
        "hyperprior_blob": "brotli_quality9_pickled_fp16_hyperprior_gate_weights",
        "latents_a_blob": "raw_int16_row_major_per_pair_latents_a",
        "latents_b_blob": "raw_int16_row_major_per_pair_latents_b",
        "meta_blob": "utf8_json_meta_includes_hyperprior_hidden_gate_init_bias",
    },
    inflate_runtime_loc_budget=200,
    runtime_dep_closure=("torch>=2.5,<2.7", "brotli", "av"),
    export_format="custom",
    score_aware_loss="scorer_loss_terms_btchw",
    bolt_on_loc_budget=400,
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
    cost_band_p50_usd=0.50,
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
        "catalog_322_hyperprior_gate_bytes_super_additive_proof_pending",
        "catalog_139_no_op_detector_planned_for_all_three_blobs",
    ),
    hook_not_applicable_rationale={
        "hook_sensitivity_contribution": "L0 SCAFFOLD; no sensitivity signal",
        "hook_bit_allocator_class": (
            "End-to-end neural-codec at per-pair granularity (gate routes "
            "between two branches); no per-tensor bit allocator at scaffold"
        ),
        "hook_continual_learning_anchor_kind": (
            "L0 SCAFFOLD: no posterior anchor until [contest-CUDA] anchor lands"
        ),
        "hook_probe_disambiguator": (
            "Hyperprior gate distribution + per-branch latent distribution "
            "sweep IS the Stage 1 dispatch's empirical purpose per "
            "PACT-NERV-ULTIMATE Variant #18; Ballé 2018 hyperprior provides "
            "the predicted SUPER_ADDITIVE classification mechanism"
        ),
    },
)


@register_substrate(PACT_NERV_NCEC_SUBSTRATE_CONTRACT)
def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.smoke:
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
