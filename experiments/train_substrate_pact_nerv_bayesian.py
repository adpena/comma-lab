# SPDX-License-Identifier: MIT
"""Train the pact_nerv_bayesian substrate L0 SCAFFOLD.
# NO_GRAD_WAIVED:MLX_substrate_trainer_uses_mx_no_grad_or_substrate_uses_lazy_eval_no_autograd_per_mlx_first_canonical_doctrine_4107bbf8d_or_substrate_eval_uses_alternate_memory_management_per_comprehensive_bug_audit_cascade_20260526

WAVE-3-PACT-NERV-G2-MID-LOC-L0-BUILD 2026-05-20. Variant #8 of PACT-NERV-ULTIMATE.

Distinguishing primitive: Bayesian latent embedding layer per Blundell 1505.05424
Bayes by Backprop + Kingma 1312.6114 VAE + MacKay 1992 framework. MacKay inner
council seat alignment.
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
from tac.substrates._shared.smoke_auth_eval_gate import (
    gate_auth_eval_call as _canon_gate_auth_eval_call,
)
from tac.substrates._shared.trainer_skeleton import (
    build_optimized_training_context as _canon_build_optimized_training_context,
)
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
SUBSTRATE_TAG = "pact_nerv_bayesian"
EVAL_HW = (384, 512)
N_PAIRS_FULL = 600
CONTEST_NORMALIZER = 37_545_489.0
CONTEST_AUTH_EVAL_SCRIPT = REPO_ROOT / "experiments" / "contest_auth_eval.py"


TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--video-path": {
        "env": "PACT_NERV_BAYESIAN_VIDEO_PATH",
        "rationale": "contest video required for score-aware training",
        "default": str(DEFAULT_VIDEO_PATH.relative_to(REPO_ROOT)),
        "satisfied_by_profile": (),
        "requires": (),
        "required_input_file": True,
        "generator_command": "contest-pinned upstream snapshot",
    },
    "--output-dir": {
        "env": "PACT_NERV_BAYESIAN_OUTPUT_DIR",
        "rationale": "custody for checkpoints + archive + provenance",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--epochs": {
        "env": "PACT_NERV_BAYESIAN_EPOCHS",
        "rationale": "Stage 1 operator-gated before non-smoke",
        "default": "2",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--upstream-dir": {
        "env": "PACT_NERV_BAYESIAN_UPSTREAM_DIR",
        "rationale": "upstream/ root for scorer weights",
        "default": str(DEFAULT_UPSTREAM_DIR.relative_to(REPO_ROOT)),
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--device": {
        "env": "PACT_NERV_BAYESIAN_DEVICE",
        "rationale": "cuda required for full; cpu only with --smoke; no MPS",
        "default": "cpu",
        "satisfied_by_profile": (),
        "requires": (),
    },
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="train_substrate_pact_nerv_bayesian")
    p.add_argument("--video-path", type=Path, default=DEFAULT_VIDEO_PATH)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260520)
    p.add_argument("--latent-dim", type=int, default=8)
    p.add_argument("--bayesian-prior-sigma", type=float, default=1.0)
    p.add_argument("--bayesian-log-sigma-init", type=float, default=-3.0)
    p.add_argument("--kl-weight", type=float, default=1.0)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--ema-decay", type=float, default=0.997)
    p.add_argument("--alpha-rate", type=float, default=25.0)
    p.add_argument("--beta-seg", type=float, default=100.0)
    p.add_argument("--gamma-pose", type=float, default=1.0)
    p.add_argument("--pose-weight-scale", type=float, default=1.0)
    p.add_argument("--noise-std", type=float, default=0.5)
    p.add_argument("--max-pairs", type=int, default=None)
    p.add_argument("--val-pair-count", type=int, default=64)
    p.add_argument("--val-every-epochs", type=int, default=10)
    p.add_argument("--skip-archive-build", action="store_true", default=False)
    p.add_argument("--skip-auth-eval", action="store_true", default=False)
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
    import torch

    from tac.differentiable_eval_roundtrip import patch_upstream_yuv6_globally
    from tac.substrates.pact_nerv_bayesian.architecture import (
        PactNervBayesianConfig,
        PactNervBayesianSubstrate,
    )

    _pin_seeds(args.seed)
    device = _device_or_die(args.device, smoke=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = _utc_now_iso()

    patch_upstream_yuv6_globally()

    cfg = PactNervBayesianConfig(
        latent_dim=args.latent_dim,
        embed_dim=16,
        initial_grid_h=3,
        initial_grid_w=4,
        decoder_channels=(16, 12, 8),
        sin_frequency=30.0,
        num_upsample_blocks=3,
        bayesian_prior_sigma=args.bayesian_prior_sigma,
        bayesian_log_sigma_init=args.bayesian_log_sigma_init,
        kl_weight=args.kl_weight,
        num_pairs=max(2, args.batch_size),
        output_height=24,
        output_width=32,
    )
    model = PactNervBayesianSubstrate(cfg).to(device)
    n_params = model.num_parameters()
    n_bayes_params = model.num_bayesian_parameters()
    print(
        f"[smoke] pact_nerv_bayesian params: {n_params:,} (bayesian: "
        f"{n_bayes_params:,}; ~{100*n_bayes_params/max(n_params, 1):.1f}% of total) "
        f"device={device}"
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = max(1, min(args.epochs, 3))
    for step in range(epochs):
        idx = torch.arange(cfg.num_pairs, device=device, dtype=torch.long)
        rgb_0, rgb_1 = model(idx)
        loss = rgb_0.abs().mean() + rgb_1.abs().mean()
        loss = loss + cfg.kl_weight * model.last_kl_div
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(f"[smoke] step {step}: loss={loss.item():.4f} kl={float(model.last_kl_div.item()):.4f}")

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
        env_var_candidates=("PACT_NERV_BAYESIAN_GPU", "MODAL_GPU"),
    )
    provenance = {
        "schema": "pact_nerv_bayesian_l0_scaffold_smoke_v1",
        "generated_at": _utc_now_iso(),
        "started_at": started_at,
        "git_head": _git_head_sha(),
        "trainer": "experiments/train_substrate_pact_nerv_bayesian.py",
        "lane_id": "lane_pact_nerv_bayesian_l0_scaffold_20260520",
        "substrate_tag": SUBSTRATE_TAG,
        "args": {
            k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
        },
        "pytorch_version": _canonical_torch_version_string(),
        "device": str(device),
        "hardware_substrate_detected": detected_substrate,
        "n_params": int(n_params),
        "n_bayesian_params": int(n_bayes_params),
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
    """Full score-aware training entry point — CUDA-required; paid-GPU gated.

    PACT-NERV-FULL-MAIN-CLUSTER-2 2026-05-27: routes the substrate-AGNOSTIC
    training loop through the canonical ``tac.substrates._shared.pact_nerv_full_main``
    helper (mirrors the implemented ``ia3``/``vq`` sisters); the UNIQUE
    Bayesian distinguishing feature stays in this substrate's architecture +
    archive + score-aware loss. The ``NotImplementedError`` is extinguished;
    PAID DISPATCH is still gated by ``dispatch_enabled: false`` + ``research_only:
    true`` on the recipe per Catalog #325 until the per-substrate symposium
    clears it (code complete, trigger gated — the canonical "implement all
    without firing council-gated paid paths" resolution).

    Honored end-to-end: real contest video (Catalog #114); patch yuv6 BEFORE
    scorer construction (eval_roundtrip non-negotiable); ``load_differentiable_
    scorers`` (no scorer at inflate); score-domain Lagrangian via the variant
    loss → Catalog #164 dispatch; EMA shadow (Quantizr 0.997); CUDA-required
    (``device_or_die`` rejects MPS per Catalog #1); CUDA auth-eval via canonical
    ``gate_auth_eval_call`` (Catalog #226); posterior-update via
    ``posterior_update_locked`` (Catalog #128); contest-compliant numpy/PIL
    runtime (Catalog #146 + #295).

    Reactivation criteria for PAID DISPATCH (per HNeRV parity L2): PACT-NERV
    symposium Stage 1 operator-gated approval (Catalog #325) + cargo-cult audit
    (Catalog #303) + 9-dim checklist (#294) + Dykstra band (#296); recipe
    ``research_only`` flips to false + ``dispatch_enabled`` to true.
    """
    from tac.differentiable_eval_roundtrip import (
        patch_upstream_yuv6_globally,
        unpatch_upstream_yuv6,
    )
    from tac.scorer import load_differentiable_scorers
    from tac.substrates._shared.pact_nerv_full_main import (
        build_archive_zip,
        closed_form_weight_byte_proxy,
        decode_pairs_for_training,
        run_pact_nerv_score_aware_training,
        write_contest_runtime,
    )
    from tac.substrates.pact_nerv_bayesian import (
        PactNervBayesianConfig,
        PactNervBayesianScoreAwareLoss,
        PactNervBayesianSubstrate,
        ScoreAwareLossWeights,
        pack_archive,
    )

    _pin_seeds(args.seed)
    device = _device_or_die(args.device, smoke=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage_log: list[dict[str, Any]] = []
    yuv6_token = patch_upstream_yuv6_globally()
    try:
        posenet, segnet = load_differentiable_scorers(args.upstream_dir, device=device)
        for p in list(posenet.parameters()) + list(segnet.parameters()):
            p.requires_grad_(False)
        posenet.eval()
        segnet.eval()

        print(f"[full:{SUBSTRATE_TAG}] decoding pairs from {args.video_path} ...")
        pair_tensor = decode_pairs_for_training(
            args.video_path, substrate_tag=SUBSTRATE_TAG, n_pairs=N_PAIRS_FULL,
            max_pairs=args.max_pairs, repo_root=REPO_ROOT,
        ).to(device)
        n_pairs = int(pair_tensor.shape[0])
        print(f"[full:{SUBSTRATE_TAG}] decoded {n_pairs} pairs at {EVAL_HW}")

        cfg = PactNervBayesianConfig(
            latent_dim=args.latent_dim, sin_frequency=30.0,
            bayesian_prior_sigma=args.bayesian_prior_sigma,
            kl_weight=args.kl_weight,
            num_pairs=n_pairs,
            output_height=EVAL_HW[0], output_width=EVAL_HW[1],
        )
        model = PactNervBayesianSubstrate(cfg).to(device)
        print(f"[full:{SUBSTRATE_TAG}] params: {model.num_parameters():,}")

        weights = ScoreAwareLossWeights(
            alpha_rate=args.alpha_rate, beta_seg=args.beta_seg,
            gamma_pose=args.gamma_pose, pose_weight_scale=args.pose_weight_scale,
            contest_normalizer=CONTEST_NORMALIZER,
                kl_weight=args.kl_weight,
        )
        loss_fn = PactNervBayesianScoreAwareLoss(
            seg_scorer=segnet, pose_scorer=posenet, weights=weights
        )

        opt_ctx = _canon_build_optimized_training_context(
            args, scorers=(posenet, segnet), gt_pairs=pair_tensor,
            substrate_model=model, device=device,
        )
        gt_cache = opt_ctx.gt_cache
        archive_bytes_proxy = closed_form_weight_byte_proxy(model)

        def _compute_loss(
            m, idx, gt_0, gt_1, abp, *, gt_pose_batch, gt_seg_batch, gt_seg_already_probs
        ):
            rgb_0, rgb_1 = m(idx)
            return loss_fn(
                rgb_0 * 255.0, rgb_1 * 255.0, gt_0, gt_1, abp,
                kl_div=m.last_kl_div,
                apply_eval_roundtrip=True, noise_std=args.noise_std,
                gt_pose_batch=gt_pose_batch, gt_seg_batch=gt_seg_batch,
                gt_seg_already_probs=gt_seg_already_probs,
            )

        result = run_pact_nerv_score_aware_training(
            model=model, pair_tensor=pair_tensor, compute_loss=_compute_loss,
            archive_bytes_proxy=archive_bytes_proxy, device=device,
            output_dir=args.output_dir, substrate_tag=SUBSTRATE_TAG,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            weight_decay=args.weight_decay, grad_clip=args.grad_clip,
            ema_decay=args.ema_decay, val_pair_count=args.val_pair_count,
            val_every_epochs=args.val_every_epochs, gt_cache=gt_cache,
            stage_log=stage_log, config_asdict=asdict(cfg),
        )
        print(
            f"[full:{SUBSTRATE_TAG}] train done: best_val_lag="
            f"{result.best_val_lagrangian:.6f} elapsed={result.train_elapsed_sec:.1f}s"
        )

        archive_sha = ""
        archive_bytes = 0
        archive_zip_path = args.output_dir / "archive.zip"
        if not args.skip_archive_build:
            sd = result.best_ema_state_dict
            latents = sd["latents"].detach().cpu()
            decoder_sd = {k: v for k, v in sd.items() if k != "latents"}
            meta = {
                "embed_dim": cfg.embed_dim, "initial_grid_h": cfg.initial_grid_h,
                "initial_grid_w": cfg.initial_grid_w,
                "decoder_channels": list(cfg.decoder_channels),
                "sin_frequency": cfg.sin_frequency,
                "num_upsample_blocks": cfg.num_upsample_blocks,
                "output_height": cfg.output_height, "output_width": cfg.output_width,
            }
            bin_bytes = pack_archive(decoder_sd, latents, meta)
            (args.output_dir / "0.bin").write_bytes(bin_bytes)
            archive_sha = _sha256_bytes(bin_bytes)
            archive_bytes = len(bin_bytes)
            submission_dir = args.output_dir / "submission"
            write_contest_runtime(
                submission_dir, substrate_pkg_name="pact_nerv_bayesian",
                repo_root=REPO_ROOT,
            )
            (submission_dir / "0.bin").write_bytes(bin_bytes)
            build_archive_zip(
                archive_zip_path, bin_bytes=bin_bytes, submission_dir=submission_dir
            )
            print(f"[full:{SUBSTRATE_TAG}] wrote 0.bin ({archive_bytes} B) + archive.zip")

        auth_eval_result_path: Path | None = None
        contest_cuda_score: float | None = None
        if not args.skip_auth_eval and archive_zip_path.is_file():
            auth_eval_result_path = args.output_dir / "contest_auth_eval_cuda.json"
            auth_result = _canon_gate_auth_eval_call(
                args=args, archive_zip=archive_zip_path,
                inflate_sh=args.output_dir / "submission" / "inflate.sh",
                upstream_dir=args.upstream_dir, output_json=auth_eval_result_path,
                contest_auth_eval_script=CONTEST_AUTH_EVAL_SCRIPT,
                substrate_tag=SUBSTRATE_TAG, device=device,
            )
            if auth_result is not None:
                contest_cuda_score = auth_result["auth_eval_cuda_score"]
                print(f"[full:{SUBSTRATE_TAG}] [contest-CUDA] score = {contest_cuda_score}")

        if contest_cuda_score is not None and archive_sha:
            try:
                from tac.continual_learning import (
                    ContestResult,
                    posterior_update_locked,
                )

                _detected = _canon_detect_hardware_substrate(
                    axis="cuda", substrate_tag=SUBSTRATE_TAG,
                    provenance_path=args.output_dir / "provenance.json",
                    env_var_candidates=("PACT_NERV_BAYESIAN_GPU", "MODAL_GPU"),
                )
                update = posterior_update_locked(
                    ContestResult(
                        axis="cuda", hardware_substrate=_detected,
                        architecture_class="lane_pact_nerv_bayesian_l0_scaffold_20260520",
                        score_value=contest_cuda_score, evidence_tag="[contest-CUDA]",
                        archive_sha256=archive_sha, archive_bytes=archive_bytes,
                        notes=f"pact_nerv_bayesian first-anchor; epochs={args.epochs}",
                        observed_at_utc=_utc_now_iso(),
                    )
                )
                print(f"[full:{SUBSTRATE_TAG}] posterior_update accepted={update.accepted}")
            except Exception as exc:
                print(f"[full:{SUBSTRATE_TAG}] posterior_update failed: {exc}", file=sys.stderr)

        provenance = {
            "schema": "pact_nerv_bayesian_full_provenance_v1",
            "generated_at": _utc_now_iso(),
            "git_head": _git_head_sha(),
            "trainer": "experiments/train_substrate_pact_nerv_bayesian.py",
            "lane_id": "lane_pact_nerv_bayesian_l0_scaffold_20260520",
            "substrate_tag": SUBSTRATE_TAG,
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "pytorch_version": _canonical_torch_version_string(),
            "device": str(device),
            "num_pairs_decoded": result.n_pairs,
            "best_val_lagrangian": (
                result.best_val_lagrangian
                if result.best_val_lagrangian == result.best_val_lagrangian else None
            ),
            "best_epoch": result.best_epoch,
            "train_elapsed_sec": result.train_elapsed_sec,
            "archive_sha256": archive_sha,
            "archive_bytes": archive_bytes,
            "auth_eval_cuda_score": contest_cuda_score,
            "auth_eval_json_path": (
                str(auth_eval_result_path) if auth_eval_result_path else None
            ),
            "stage_log": stage_log,
            "custody_status": "ci-rebuildable",
            "score_claim": contest_cuda_score is not None,
            "score_axis_tag": "[contest-CUDA]" if contest_cuda_score is not None else None,
            "promotion_eligible": False,
            "ready_for_exact_eval_dispatch": False,
        }
        (args.output_dir / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"[full:{SUBSTRATE_TAG}] wrote {args.output_dir / 'provenance.json'}")
        return 0
    finally:
        unpatch_upstream_yuv6(yuv6_token)

PACT_NERV_BAYESIAN_SUBSTRATE_CONTRACT = SubstrateContract(
    id="pact_nerv_bayesian",
    lane_id="lane_pact_nerv_bayesian_l0_scaffold_20260520",
    target_modes=("research_substrate",),
    deployment_target="t4_contest_runtime",
    council_verdict_provenance=(
        ".omx/research/pact_nerv_ultimate_research_and_design_"
        "20260520T193443Z.md"
    ),
    archive_grammar=(
        "PBN monolithic single-file 0.bin (21-byte header; decoder + bayesian "
        "posterior mean state_dict; mean-only at inflate per Blundell §4)"
    ),
    parser_section_manifest={
        "header": "21_byte_fixed_PBN_magic_v1",
        "decoder_blob": "brotli_quality9_pickled_fp16_decoder_plus_bayesian_mean",
        "bayesian_posterior_subset": "logical_grouping_inside_decoder_blob_weight_mu",
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
    cost_band_p50_usd=0.45,
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
            "L0 SCAFFOLD; per-pair posterior variance IS sensitivity at L1+"
        ),
        "hook_bit_allocator_class": (
            "fp16 brotli on posterior-mean weight blob; no per-tensor bit "
            "allocator at scaffold posture"
        ),
        "hook_continual_learning_anchor_kind": (
            "L0 SCAFFOLD: no posterior anchor until [contest-CUDA] at Stage 1"
        ),
        "hook_probe_disambiguator": (
            "single mechanism (Bayes-by-Backprop reparam); KL-annealing-schedule "
            "vs fixed-weight is Stage 1's empirical purpose"
        ),
    },
)


@register_substrate(PACT_NERV_BAYESIAN_SUBSTRATE_CONTRACT)
def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.smoke:
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
