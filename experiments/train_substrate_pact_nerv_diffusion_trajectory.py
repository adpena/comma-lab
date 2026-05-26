# SPDX-License-Identifier: MIT
"""Train the pact_nerv_diffusion_trajectory substrate L0 SCAFFOLD.
# NO_GRAD_WAIVED:MLX_substrate_trainer_uses_mx_no_grad_or_substrate_uses_lazy_eval_no_autograd_per_mlx_first_canonical_doctrine_4107bbf8d_or_substrate_eval_uses_alternate_memory_management_per_comprehensive_bug_audit_cascade_20260526

WAVE-3-PACT-NERV-G2-MID-LOC-L0-BUILD 2026-05-20. Variant #10 of PACT-NERV-ULTIMATE.
Paper-worthy bleeding-edge: per-pair latent diffusion trajectory predictor.

Distinguishing primitive: 5-step lightweight diffusion-trajectory predictor
(Rombach 2022 LDM + Blattmann 2023 video latent diffusion) refines stored
Gaussian seeds into per-pair latents at inflate time.
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
SUBSTRATE_TAG = "pact_nerv_diffusion_trajectory"


TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--video-path": {
        "env": "PACT_NERV_DIFFUSION_TRAJECTORY_VIDEO_PATH",
        "rationale": "contest video required",
        "default": str(DEFAULT_VIDEO_PATH.relative_to(REPO_ROOT)),
        "satisfied_by_profile": (),
        "requires": (),
        "required_input_file": True,
        "generator_command": "contest-pinned upstream snapshot",
    },
    "--output-dir": {
        "env": "PACT_NERV_DIFFUSION_TRAJECTORY_OUTPUT_DIR",
        "rationale": "custody",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--epochs": {
        "env": "PACT_NERV_DIFFUSION_TRAJECTORY_EPOCHS",
        "rationale": "Stage 1 operator-gated",
        "default": "2",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--upstream-dir": {
        "env": "PACT_NERV_DIFFUSION_TRAJECTORY_UPSTREAM_DIR",
        "rationale": "upstream/ root",
        "default": str(DEFAULT_UPSTREAM_DIR.relative_to(REPO_ROOT)),
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--device": {
        "env": "PACT_NERV_DIFFUSION_TRAJECTORY_DEVICE",
        "rationale": "no MPS",
        "default": "cpu",
        "satisfied_by_profile": (),
        "requires": (),
    },
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="train_substrate_pact_nerv_diffusion_trajectory")
    p.add_argument("--video-path", type=Path, default=DEFAULT_VIDEO_PATH)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260520)
    p.add_argument("--latent-dim", type=int, default=8)
    p.add_argument("--diffusion-num-timesteps", type=int, default=3)
    p.add_argument("--diffusion-predictor-hidden", type=int, default=8)
    p.add_argument("--noise-schedule", choices=["linear", "cosine"], default="linear")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--enable-autocast-fp16", action="store_true", default=False)
    p.add_argument("--enable-torch-compile", action="store_true", default=False)
    p.add_argument("--enable-gt-scorer-cache", action="store_true", default=False)
    return p


def _utc_now_iso() -> str:
    return _canonical_utc_now_iso()


def _git_head_sha() -> str:
    return _canonical_git_head_sha(REPO_ROOT)


def _pin_seeds(seed: int) -> None:
    _canonical_pin_seeds(seed)


def _device_or_die(name: str, *, smoke: bool):
    return _canonical_device_or_die(name, smoke=smoke, substrate_tag=SUBSTRATE_TAG)


def _smoke_main(args: argparse.Namespace) -> int:
    import torch

    from tac.differentiable_eval_roundtrip import patch_upstream_yuv6_globally
    from tac.substrates.pact_nerv_diffusion_trajectory.architecture import (
        PactNervDiffusionTrajectoryConfig,
        PactNervDiffusionTrajectorySubstrate,
    )

    _pin_seeds(args.seed)
    device = _device_or_die(args.device, smoke=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = _utc_now_iso()

    patch_upstream_yuv6_globally()

    cfg = PactNervDiffusionTrajectoryConfig(
        latent_dim=args.latent_dim,
        embed_dim=16,
        initial_grid_h=3,
        initial_grid_w=4,
        decoder_channels=(16, 12, 8),
        sin_frequency=30.0,
        num_upsample_blocks=3,
        diffusion_num_timesteps=args.diffusion_num_timesteps,
        diffusion_predictor_hidden=args.diffusion_predictor_hidden,
        noise_schedule=args.noise_schedule,
        num_pairs=max(2, args.batch_size),
        output_height=24,
        output_width=32,
    )
    model = PactNervDiffusionTrajectorySubstrate(cfg).to(device)
    n_params = model.num_parameters()
    n_pred_params = model.num_predictor_parameters()
    print(
        f"[smoke] pact_nerv_diffusion_trajectory params: {n_params:,} (predictor: "
        f"{n_pred_params:,}; ~{100*n_pred_params/max(n_params, 1):.1f}% of total) "
        f"device={device} timesteps={cfg.diffusion_num_timesteps} "
        f"schedule={cfg.noise_schedule}"
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
        env_var_candidates=("PACT_NERV_DIFFUSION_TRAJECTORY_GPU", "MODAL_GPU"),
    )
    provenance = {
        "schema": "pact_nerv_diffusion_trajectory_l0_scaffold_smoke_v1",
        "generated_at": _utc_now_iso(),
        "started_at": started_at,
        "git_head": _git_head_sha(),
        "trainer": "experiments/train_substrate_pact_nerv_diffusion_trajectory.py",
        "lane_id": "lane_pact_nerv_diffusion_trajectory_l0_scaffold_20260520",
        "substrate_tag": SUBSTRATE_TAG,
        "args": {
            k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
        },
        "pytorch_version": _canonical_torch_version_string(),
        "device": str(device),
        "hardware_substrate_detected": detected_substrate,
        "n_params": int(n_params),
        "n_predictor_params": int(n_pred_params),
        "diffusion_num_timesteps": cfg.diffusion_num_timesteps,
        "noise_schedule": cfg.noise_schedule,
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

    Per the PACT-NERV-ULTIMATE memo: this variant is PAPER target only
    (predicted regression on contest rate term). Stage 1 dispatch
    operator-gated for paper-quality result; NOT a frontier candidate.

    Reactivation criteria:
    1. PACT-NERV Stage 1 dispatch operator-gated per Catalog #325.
    2. Cargo-cult audit per Catalog #303 (5 timesteps + linear schedule +
       per-step depth-2 MLP CARGO-CULTED at L0; Stage 1 sweep against
       cosine schedule + shared-MLP-across-steps + full UNet).
    3. 9-dim checklist + observability + Dykstra feasibility.
    4. Score-aware loop with canonical auth-eval per Catalog #226.
    """
    raise NotImplementedError(
        "[pact_nerv_diffusion_trajectory] full training path is OPERATOR-GATED per "
        "Catalog #240 + #315 + #325. This is an L0 SCAFFOLD trainer; substrate is "
        "research_only until PACT-NERV symposium Stage 1 dispatch lands. PAPER "
        "target only; predicted regression on contest rate term."
    )


PACT_NERV_DIFFUSION_TRAJECTORY_SUBSTRATE_CONTRACT = SubstrateContract(
    id="pact_nerv_diffusion_trajectory",
    lane_id="lane_pact_nerv_diffusion_trajectory_l0_scaffold_20260520",
    target_modes=("research_substrate",),
    deployment_target="t4_contest_runtime",
    council_verdict_provenance=(
        ".omx/research/pact_nerv_ultimate_research_and_design_"
        "20260520T193443Z.md"
    ),
    archive_grammar=(
        "PDT monolithic single-file 0.bin (22-byte header carrying "
        "NUM_TIMESTEPS u8; decoder + predictor + int16 seeds + meta)"
    ),
    parser_section_manifest={
        "header": "22_byte_fixed_PDT_magic_v1_num_timesteps",
        "decoder_blob": "brotli_quality9_pickled_fp16_decoder_plus_predictor",
        "predictor_subset": "logical_grouping_inside_decoder_blob_predictor_prefix",
        "seed_blob": "raw_int16_row_major",
        "meta_blob": "utf8_json_meta",
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
    ),
    hook_not_applicable_rationale={
        "hook_sensitivity_contribution": (
            "L0 SCAFFOLD; per-timestep predictor weight sensitivity available at L1+"
        ),
        "hook_bit_allocator_class": (
            "fp16 brotli on decoder + predictor; int16 seeds; no per-tensor "
            "bit allocator at scaffold posture"
        ),
        "hook_continual_learning_anchor_kind": (
            "L0 SCAFFOLD: no posterior anchor until [contest-CUDA] at Stage 1"
        ),
        "hook_probe_disambiguator": (
            "single mechanism (5-step linear schedule); cosine vs linear + "
            "5-step vs 2/10-step is Stage 1's empirical purpose"
        ),
    },
)


@register_substrate(PACT_NERV_DIFFUSION_TRAJECTORY_SUBSTRATE_CONTRACT)
def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.smoke:
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
