# SPDX-License-Identifier: MIT
"""Train pact_nerv_selector_v2 L0 SCAFFOLD (WAVE-3-PACT-NERV-G3-SELECTOR-EXTENSIONS 2026-05-20).
# NO_GRAD_WAIVED:MLX_substrate_trainer_uses_mx_no_grad_or_substrate_uses_lazy_eval_no_autograd_per_mlx_first_canonical_doctrine_4107bbf8d_or_substrate_eval_uses_alternate_memory_management_per_comprehensive_bug_audit_cascade_20260526

Group 3 SELECTOR-PARADIGM-EXTENSIONS variant per PACT-NERV-ULTIMATE (commit
``e3ad4243a``) variant #11. Direct empirical extension of CROSS-CANDIDATE
finding #1 (+259 bytes / +0.00333 ratio).

L0 SCAFFOLD posture: ``_smoke_main`` exercises substrate package + arithmetic
coder primitive; ``_full_main`` raises ``NotImplementedError`` per CLAUDE.md
"Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" + Catalog #240/#315/#325.

Literature anchor: Witten-Neal-Cleary 1987 + Said 2004 (arithmetic coding;
fractional-bit precision vs Huffman integer-bit code-lengths).

Cross-ref:
    src/tac/substrates/pact_nerv_selector_v2/ (substrate package)
    .omx/operator_authorize_recipes/substrate_pact_nerv_selector_v2_modal_t4_dispatch.yaml
    .omx/research/pact_nerv_selector_v2_l0_scaffold_design_20260520T<UTC>.md
    .omx/research/pact_nerv_ultimate_research_and_design_20260520T193443Z.md (canonical)
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
CONTEST_NORMALIZER = 37_545_489.0
SUBSTRATE_TAG = "pact_nerv_selector_v2"


TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--video-path": {
        "env": "PACT_NERV_SELECTOR_V2_VIDEO_PATH",
        "rationale": (
            "score-aware substrate MUST train against contest video "
            "(upstream/videos/0.mkv); synthetic data FORBIDDEN outside --smoke"
        ),
        "default": str(DEFAULT_VIDEO_PATH.relative_to(REPO_ROOT)),
        "satisfied_by_profile": (),
        "requires": (),
        "required_input_file": True,
        "generator_command": "contest-pinned upstream snapshot",
        "rationale_audit": (
            ".omx/research/pact_nerv_selector_v2_l0_scaffold_design_20260520T<UTC>.md"
        ),
    },
    "--output-dir": {
        "env": "PACT_NERV_SELECTOR_V2_OUTPUT_DIR",
        "rationale": "custody location for checkpoints + archive + provenance",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--epochs": {
        "env": "PACT_NERV_SELECTOR_V2_EPOCHS",
        "rationale": (
            "pact_nerv_selector_v2 substrate engineering pass; Stage 1 "
            "dispatch operator-gated before non-smoke training"
        ),
        "default": "2",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--upstream-dir": {
        "env": "PACT_NERV_SELECTOR_V2_UPSTREAM_DIR",
        "rationale": "upstream/ root for scorer weights + evaluate.py",
        "default": str(DEFAULT_UPSTREAM_DIR.relative_to(REPO_ROOT)),
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--device": {
        "env": "PACT_NERV_SELECTOR_V2_DEVICE",
        "rationale": (
            "compute device; cuda required for full training (MPS refused "
            "per CLAUDE.md MPS-NOISE rule); cpu permitted only with --smoke"
        ),
        "default": "cpu",
        "satisfied_by_profile": (),
        "requires": (),
    },
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_substrate_pact_nerv_selector_v2",
        description=(
            "Train pact_nerv_selector_v2 L0 SCAFFOLD (smoke only; full path "
            "gated by Stage 1 dispatch per Catalog #240 + #315 + #325)."
        ),
    )
    p.add_argument("--video-path", type=Path, default=DEFAULT_VIDEO_PATH)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260520)
    p.add_argument("--latent-dim", type=int, default=8)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cpu")
    p.add_argument("--smoke", action="store_true")
    p.add_argument(
        "--enable-autocast-fp16", action="store_true", default=False,
        help="RESERVED (Stage 1): autocast(fp16) wrapper.",
    )
    p.add_argument(
        "--enable-torch-compile", action="store_true", default=False,
        help="RESERVED (Stage 1): torch.compile wrapper.",
    )
    p.add_argument(
        "--enable-gt-scorer-cache", action="store_true", default=False,
        help="RESERVED (Stage 1): GT-scorer-output cache.",
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
    """Tiny CPU smoke exercising pact_nerv_selector_v2 substrate + arithmetic coder.

    Per CLAUDE.md "eval_roundtrip - NON-NEGOTIABLE" + Catalog #6 MANDATORY
    DEFAULT: calls patch_upstream_yuv6_globally() BEFORE any scorer
    construction so the future Stage 1 PoseNet gradient path remains
    differentiable.
    """
    import torch

    from tac.differentiable_eval_roundtrip import patch_upstream_yuv6_globally
    from tac.substrates.pact_nerv_selector_v2.architecture import (
        ArithmeticSelectorCoder,
        PactNervSelectorV2Config,
        PactNervSelectorV2Substrate,
    )

    _pin_seeds(args.seed)
    device = _device_or_die(args.device, smoke=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = _utc_now_iso()

    # CLAUDE.md non-negotiable: patch yuv6 BEFORE any scorer load.
    patch_upstream_yuv6_globally()

    cfg = PactNervSelectorV2Config(
        latent_dim=args.latent_dim,
        embed_dim=16,
        initial_grid_h=3,
        initial_grid_w=4,
        decoder_channels=(16, 12, 8),
        sin_frequency=30.0,
        num_upsample_blocks=3,
        num_pairs=max(2, args.batch_size),
        output_height=24,
        output_width=32,
        selector_palette_size=16,
    )
    model = PactNervSelectorV2Substrate(cfg).to(device)
    n_params = model.num_parameters()
    print(f"[smoke] pact_nerv_selector_v2 params: {n_params:,} device={device}")

    # Smoke-test the arithmetic coder primitive
    coder = ArithmeticSelectorCoder(palette_size=cfg.selector_palette_size)
    syms = [i % cfg.selector_palette_size for i in range(cfg.num_pairs)]
    sel_bytes = coder.encode(syms)
    print(
        f"[smoke] arithmetic coder: {len(syms)} syms -> {len(sel_bytes)} bytes "
        f"(ideal {coder.encoded_bit_length(syms)} bits)"
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
    torch.save(ckpt, args.output_dir / "smoke_checkpoint.pt")

    detected = _canon_detect_hardware_substrate(
        axis="cpu",
        substrate_tag=SUBSTRATE_TAG,
        provenance_path=args.output_dir / "provenance.json",
        env_var_candidates=("PACT_NERV_SELECTOR_V2_GPU", "MODAL_GPU"),
    )
    provenance = {
        "schema": "pact_nerv_selector_v2_l0_scaffold_smoke_v1",
        "generated_at": _utc_now_iso(),
        "started_at": started_at,
        "git_head": _git_head_sha(),
        "trainer": "experiments/train_substrate_pact_nerv_selector_v2.py",
        "lane_id": "lane_pact_nerv_selector_v2_l0_scaffold_20260520",
        "substrate_tag": SUBSTRATE_TAG,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "pytorch_version": _canonical_torch_version_string(),
        "device": str(device),
        "hardware_substrate_detected": detected,
        "n_params": int(n_params),
        "selector_palette_size": cfg.selector_palette_size,
        "selector_bytes_encoded": len(sel_bytes),
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
    return 0


def _full_main(args: argparse.Namespace) -> int:
    """Full path - REFUSED at L0 SCAFFOLD posture.

    Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY" +
    Catalog #220 + #240 + #315 + #325. PACT-NERV-ULTIMATE Variant #11
    Stage 1 dispatch operator-gated.

    Reactivation criteria:
    1. PACT-NERV-ULTIMATE STAIRCASE Step 11 dispatch operator-gated per
       Catalog #325.
    2. Cargo-cult audit per Catalog #303 (static-cum-freq + per-symbol
       encoding CARGO-CULTED at L0; sweep at L1).
    3. 9-dim checklist + observability surface + Dykstra feasibility per
       Catalog #294 / #305 / #296 in design memo.
    4. Score-aware Lagrangian wire-in + canonical auth-eval helper per
       Catalog #226.
    5. Operator-frontier-override per Catalog #300 OR Stage 1 approval.
    """
    raise NotImplementedError(
        "[pact_nerv_selector_v2] full training path is OPERATOR-GATED per "
        "Catalog #240 + #315 + #325. L0 SCAFFOLD; substrate is research_only "
        "until PACT-NERV-ULTIMATE STAIRCASE Step 11 dispatch operator-gated. "
        "See reactivation criteria in this function's docstring + "
        ".omx/research/pact_nerv_selector_v2_l0_scaffold_design_20260520T*.md "
        "+ lane_pact_nerv_selector_v2_l0_scaffold_20260520 in lane_registry."
    )


PACT_NERV_SELECTOR_V2_SUBSTRATE_CONTRACT = SubstrateContract(
    id="pact_nerv_selector_v2",
    lane_id="lane_pact_nerv_selector_v2_l0_scaffold_20260520",
    target_modes=("research_substrate",),
    deployment_target="t4_contest_runtime",
    council_verdict_provenance=(
        ".omx/research/pact_nerv_ultimate_research_and_design_20260520T193443Z.md"
    ),
    archive_grammar=(
        "PSV2 monolithic single-file 0.bin (26-byte header carrying "
        "PALETTE_SIZE u8 + SELECTOR_BLOB_LEN u32 distinctive fields; "
        "base decoder weights in brotli blob; int16 latents; arithmetic-"
        "coded u8 selectors over FEC6 k=16 palette per Witten 1987)"
    ),
    parser_section_manifest={
        "header": "26_byte_fixed_PSV2_magic_v1_palette_size",
        "decoder_blob": "brotli_quality9_pickled_fp16_base_decoder",
        "latent_blob": "raw_int16_row_major",
        "selector_blob": "arithmetic_coded_u8_selectors_witten_1987_static_fec6_k16",
        "meta_blob": "utf8_json_meta_includes_cum_freq_table",
    },
    inflate_runtime_loc_budget=200,
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
            "L0 SCAFFOLD; no sensitivity signal until Stage 1 dispatch"
        ),
        "hook_bit_allocator_class": (
            "arithmetic coder operates at selector-symbol granularity; no "
            "per-tensor bit allocator at scaffold posture"
        ),
        "hook_continual_learning_anchor_kind": (
            "L0 SCAFFOLD: no posterior anchor until [contest-CUDA] anchor lands"
        ),
        "hook_probe_disambiguator": (
            "arithmetic vs Huffman fractional-bit precision IS the Stage 1 "
            "dispatch's empirical purpose per PACT-NERV-ULTIMATE Variant #11"
        ),
    },
)


@register_substrate(PACT_NERV_SELECTOR_V2_SUBSTRATE_CONTRACT)
def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.smoke:
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
