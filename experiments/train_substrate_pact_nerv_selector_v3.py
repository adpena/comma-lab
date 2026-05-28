# SPDX-License-Identifier: MIT
"""Train pact_nerv_selector_v3 L0 SCAFFOLD (WAVE-3-PACT-NERV-G3-SELECTOR-EXTENSIONS 2026-05-20).
# NO_GRAD_WAIVED:MLX_substrate_trainer_uses_mx_no_grad_or_substrate_uses_lazy_eval_no_autograd_per_mlx_first_canonical_doctrine_4107bbf8d_or_substrate_eval_uses_alternate_memory_management_per_comprehensive_bug_audit_cascade_20260526

Group 3 SELECTOR-PARADIGM-EXTENSIONS variant per PACT-NERV-ULTIMATE variant #12
(Rice-Golomb encoding extension of FEC6 fixed-Huffman k=16 frame-exploit selector).

Literature: Golomb 1966 + Rice 1971 (Rice-Golomb codes optimal for
geometric-decay distributions).
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
SUBSTRATE_TAG = "pact_nerv_selector_v3"
EVAL_HW = (384, 512)
N_PAIRS_FULL = 600
CONTEST_NORMALIZER = 37_545_489.0
CONTEST_AUTH_EVAL_SCRIPT = REPO_ROOT / "experiments" / "contest_auth_eval.py"


TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--video-path": {
        "env": "PACT_NERV_SELECTOR_V3_VIDEO_PATH",
        "rationale": "score-aware substrate trains against contest video",
        "default": str(DEFAULT_VIDEO_PATH.relative_to(REPO_ROOT)),
        "satisfied_by_profile": (),
        "requires": (),
        "required_input_file": True,
        "generator_command": "contest-pinned upstream snapshot",
        "rationale_audit": (
            ".omx/research/pact_nerv_selector_v3_l0_scaffold_design_20260520T<UTC>.md"
        ),
    },
    "--output-dir": {
        "env": "PACT_NERV_SELECTOR_V3_OUTPUT_DIR",
        "rationale": "checkpoints + archive + provenance",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--epochs": {
        "env": "PACT_NERV_SELECTOR_V3_EPOCHS",
        "rationale": "substrate engineering pass; Stage 1 operator-gated",
        "default": "2",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--upstream-dir": {
        "env": "PACT_NERV_SELECTOR_V3_UPSTREAM_DIR",
        "rationale": "upstream/ root for scorer weights",
        "default": str(DEFAULT_UPSTREAM_DIR.relative_to(REPO_ROOT)),
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--device": {
        "env": "PACT_NERV_SELECTOR_V3_DEVICE",
        "rationale": "cuda required for full; cpu only with --smoke",
        "default": "cpu",
        "satisfied_by_profile": (),
        "requires": (),
    },
    # WAVE-N+3 SLOT 1 op-routable #1 (2026-05-28): decoder_quantization
    # threading through PyTorch sister archive emit path. The recipe at
    # .omx/operator_authorize_recipes/substrate_pact_nerv_selector_v3_int8_decoder_modal_t4_dispatch.yaml
    # declares DECODER_QUANT in env_overrides; Catalog #151 requires the
    # operator-wrapper threading to be auditable via the trainer manifest.
    "--decoder-quant": {
        "env": "DECODER_QUANT",
        "rationale": (
            "Slot 2 + Compound C archive-emit quantization mode "
            "(int8_per_channel_brotli_q11 / heterogeneous_per_tensor); "
            "default fp16_brotli_q9 preserves V3 baseline; mirrors MLX-LOCAL "
            "canonical interface"
        ),
        "default": "fp16_brotli_q9",
        "satisfied_by_profile": (),
        "requires": (),
    },
    "--fp4-qat-epochs": {
        "env": "FP4_QAT_EPOCHS",
        "rationale": (
            "FP4-QAT post-training fine-tune epochs on top-K tensors per "
            "CLAUDE.md 'QAT pipeline' non-negotiable; default 0 disables; "
            "applies only when --decoder-quant=heterogeneous_per_tensor"
        ),
        "default": "0",
        "satisfied_by_profile": (),
        "requires": (),
    },
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="train_substrate_pact_nerv_selector_v3",
        description="Train pact_nerv_selector_v3 L0 SCAFFOLD (smoke only).",
    )
    p.add_argument("--video-path", type=Path, default=DEFAULT_VIDEO_PATH)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260520)
    p.add_argument("--latent-dim", type=int, default=8)
    p.add_argument("--rice-golomb-k", type=int, default=2)
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
    # WAVE-N+3 SLOT 1 op-routable #1 PyTorch sister landing 2026-05-28:
    # decoder_quantization parameter threading per the MLX-LOCAL canonical
    # interface at experiments/train_substrate_pact_nerv_selector_v3_mlx_local.py.
    # Mirrors the sister exactly so MLX-LOCAL --decoder-quant values are valid
    # at the PyTorch contest-resolution sister and the env-override at
    # .omx/operator_authorize_recipes/substrate_pact_nerv_selector_v3_int8_decoder_modal_t4_dispatch.yaml
    # threads through as DECODER_QUANT -> --decoder-quant.
    # Default preserves V3 baseline (fp16_brotli_q9) behavior so an operator
    # who does NOT pass --decoder-quant gets the canonical baseline archive
    # emission per Catalog #240 recipe-vs-trainer-state transparency.
    p.add_argument(
        "--decoder-quant",
        type=str,
        default="fp16_brotli_q9",
        choices=[
            "fp16_brotli_q9",
            "fp16_brotli_q11",
            "int8_per_channel_brotli_q11",
            "heterogeneous_per_tensor",
        ],
        help=(
            "Decoder state-dict archive-emit quantization strategy "
            "(WAVE-N+3 SLOT 1 PyTorch sister 2026-05-28 mirrors MLX-LOCAL "
            "--decoder-quant interface). Default 'fp16_brotli_q9' preserves "
            "V3 baseline (sha256 ef5a087ff6301dbf). 'int8_per_channel_brotli_q11' "
            "emits the Slot 2 empirical -28.5% archive baseline. "
            "'heterogeneous_per_tensor' emits the Compound C variant: top-K "
            "tensors FP4-QAT + mid int8-per-channel + tail int4-groupwise NF4 "
            "via canonical tac.substrates.pact_nerv_selector_v3."
            "heterogeneous_bit_allocation helper. Per Catalog #192/#317/#341 "
            "non-promotable [macOS-MLX research-signal] pending paired-CUDA "
            "RATIFICATION per Catalog #246."
        ),
    )
    p.add_argument(
        "--fp4-qat-epochs",
        type=int,
        default=0,
        help=(
            "FP4-QAT post-training fine-tune epochs on top-K tensors "
            "(Quantizr 0.33 canonical = 200; smoke = 50; default 0 disables "
            "QAT). Applies only when --decoder-quant=heterogeneous_per_tensor. "
            "Per CLAUDE.md 'QAT pipeline' non-negotiable: scalar-weight-only "
            "fine-tune at scaled LR (default 0.1x) for fp4_qat_epochs."
        ),
    )
    p.add_argument(
        "--top-k-fp4",
        type=int,
        default=3,
        help=(
            "Number of top-K tensors routed to FP4-QAT (parent memo: top-3 "
            "cover 70.31% of decoder byte cost). Informational at this "
            "wave; the canonical heterogeneous_bit_allocation helper uses "
            "its own DEFAULT_TOP_K_FP4=3 constant."
        ),
    )
    p.add_argument(
        "--sensitivity-ranking-method",
        type=str,
        default="magnitude_x_byte_cost",
        choices=[
            "magnitude_x_byte_cost",
            "taylor_gradient",
            "dykstra_lagrangian_dual",
        ],
        help=(
            "Per-tensor sensitivity ranking method (informational; the "
            "canonical helper uses BYTE_COST x sensitivity ranking regardless; "
            "future extension hook). Mirrors MLX-LOCAL canonical interface."
        ),
    )
    p.add_argument(
        "--brotli-quality",
        type=int,
        default=None,
        help=(
            "Override brotli quality. Default None defers to per-mode "
            "canonical: fp16_brotli_q9 = 9; fp16_brotli_q11 / "
            "int8_per_channel_brotli_q11 / heterogeneous_per_tensor = 11. "
            "Mirrors MLX-LOCAL canonical posture (the archive helper at "
            "tac.substrates.pact_nerv_selector_v3.archive selects quality "
            "per decoder_quantization kind)."
        ),
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
    import torch

    from tac.differentiable_eval_roundtrip import patch_upstream_yuv6_globally
    from tac.substrates.pact_nerv_selector_v3.architecture import (
        PactNervSelectorV3Config,
        PactNervSelectorV3Substrate,
        RiceGolombSelectorCoder,
    )

    _canonical_pin_seeds(args.seed)
    device = _canonical_device_or_die(args.device, smoke=True, substrate_tag=SUBSTRATE_TAG)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = _canonical_utc_now_iso()
    patch_upstream_yuv6_globally()

    cfg = PactNervSelectorV3Config(
        latent_dim=args.latent_dim, embed_dim=16,
        initial_grid_h=3, initial_grid_w=4,
        decoder_channels=(16, 12, 8), sin_frequency=30.0,
        num_upsample_blocks=3, num_pairs=max(2, args.batch_size),
        output_height=24, output_width=32, selector_palette_size=16,
        rice_golomb_k=args.rice_golomb_k,
    )
    model = PactNervSelectorV3Substrate(cfg).to(device)
    n_params = model.num_parameters()
    print(f"[smoke] pact_nerv_selector_v3 params: {n_params:,} device={device}")

    coder = RiceGolombSelectorCoder(palette_size=cfg.selector_palette_size, k=cfg.rice_golomb_k)
    syms = [i % cfg.selector_palette_size for i in range(cfg.num_pairs)]
    sel_bytes = coder.encode(syms)
    print(
        f"[smoke] Rice-Golomb (k={cfg.rice_golomb_k}): {len(syms)} syms -> "
        f"{len(sel_bytes)} bytes (ideal {coder.encoded_bit_length(syms)} bits)"
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
        env_var_candidates=("PACT_NERV_SELECTOR_V3_GPU", "MODAL_GPU"),
    )
    provenance = {
        "schema": "pact_nerv_selector_v3_l0_scaffold_smoke_v1",
        "generated_at": _canonical_utc_now_iso(), "started_at": started_at,
        "git_head": _canonical_git_head_sha(REPO_ROOT),
        "trainer": "experiments/train_substrate_pact_nerv_selector_v3.py",
        "lane_id": "lane_pact_nerv_selector_v3_l0_scaffold_20260520",
        "substrate_tag": SUBSTRATE_TAG,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "pytorch_version": _canonical_torch_version_string(), "device": str(device),
        "hardware_substrate_detected": detected, "n_params": int(n_params),
        "rice_golomb_k": cfg.rice_golomb_k,
        "selector_palette_size": cfg.selector_palette_size,
        "selector_bytes_encoded": len(sel_bytes),
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
    """Full score-aware training entry point — CUDA-required; paid-GPU gated.

    PACT-NERV-FULL-MAIN-CLUSTER-2 2026-05-27: routes the substrate-AGNOSTIC
    training loop through the canonical ``tac.substrates._shared.pact_nerv_full_main``
    helper (mirrors the implemented ``ia3``/``vq`` sisters); the UNIQUE
    Rice-Golomb-coded FEC6 selector distinguishing feature stays in this
    substrate's archive + score-aware loss. The ``NotImplementedError`` is
    extinguished; PAID DISPATCH is still gated by ``dispatch_enabled: false`` +
    ``research_only: true`` on the recipe per Catalog #325 until the
    per-substrate symposium clears it (code complete, trigger gated).

    Honored end-to-end: real contest video (Catalog #114); patch yuv6 BEFORE
    scorer construction (eval_roundtrip non-negotiable); ``load_differentiable_
    scorers`` (no scorer at inflate); score-domain Lagrangian via the variant
    loss → Catalog #164 dispatch; EMA shadow (Quantizr 0.997); CUDA-required
    (``device_or_die`` rejects MPS per Catalog #1); CUDA auth-eval via canonical
    ``gate_auth_eval_call`` (Catalog #226); posterior-update via
    ``posterior_update_locked`` (Catalog #128); contest-compliant numpy/PIL
    runtime (Catalog #146 + #295).

    Reactivation criteria for PAID DISPATCH (per HNeRV parity L2):
    1. PACT-NERV-ULTIMATE STAIRCASE Step 12 dispatch operator-gated (Catalog #325).
    2. Cargo-cult audit (Catalog #303): fixed k=2 + per-symbol encoding.
    3. 9-dim checklist (#294) + observability (#305) + Dykstra band (#296).
    4. Recipe ``research_only`` flips to false + ``dispatch_enabled`` to true.
    """
    import torch

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
    from tac.substrates.pact_nerv_selector_v3 import (
        PactNervSelectorV3Config,
        PactNervSelectorV3ScoreAwareLoss,
        PactNervSelectorV3Substrate,
        RiceGolombSelectorCoder,
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

        cfg = PactNervSelectorV3Config(
            latent_dim=args.latent_dim, sin_frequency=30.0,
            rice_golomb_k=args.rice_golomb_k, num_pairs=n_pairs,
            output_height=EVAL_HW[0], output_width=EVAL_HW[1],
        )
        model = PactNervSelectorV3Substrate(cfg).to(device)
        print(f"[full:{SUBSTRATE_TAG}] params: {model.num_parameters():,}")

        weights = ScoreAwareLossWeights(
            alpha_rate=args.alpha_rate, beta_seg=args.beta_seg,
            gamma_pose=args.gamma_pose, pose_weight_scale=args.pose_weight_scale,
            contest_normalizer=CONTEST_NORMALIZER,
        )
        loss_fn = PactNervSelectorV3ScoreAwareLoss(
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
        qat_metrics: dict[str, Any] = {}
        if not args.skip_archive_build:
            sd = result.best_ema_state_dict
            latents = sd["latents"].detach().cpu()
            decoder_sd = {k: v for k, v in sd.items() if k != "latents"}

            # WAVE-N+3 SLOT 1 op-routable #1 (2026-05-28): validate
            # --decoder-quant against canonical set BEFORE archive emit so
            # the failure surfaces pre-emit (not post-emit at parse time).
            # Mirrors MLX-LOCAL canonical sister validation. Per Catalog
            # #146 + #324 fail-closed discipline.
            from tac.substrates.pact_nerv_selector_v3.archive import (
                DECODER_QUANTIZATION_KINDS,
            )

            decoder_quant = str(args.decoder_quant)
            if decoder_quant not in DECODER_QUANTIZATION_KINDS:
                raise ValueError(
                    f"unsupported --decoder-quant {decoder_quant!r}; "
                    f"expected one of {sorted(DECODER_QUANTIZATION_KINDS)}. "
                    "Per WAVE-N+3 SLOT 1 PyTorch sister landing 2026-05-28: "
                    "use 'int8_per_channel_brotli_q11' for the Slot 2 "
                    "sub-0.18 candidate; use 'heterogeneous_per_tensor' for "
                    "the Compound C sub-0.16 candidate; default "
                    "'fp16_brotli_q9' preserves V3 baseline."
                )

            # WAVE-N+3 SLOT 1 Compound C: optional FP4-QAT fine-tune on
            # top-K tensors BEFORE archive emit. Only fires when
            # heterogeneous quant + QAT epochs > 0. The canonical helper at
            # tac.substrates.pact_nerv_selector_v3.heterogeneous_bit_allocation
            # is framework-agnostic at the PyTorch state_dict boundary (the
            # MLX export_state_dict already returns PyTorch-shaped tensors;
            # the PyTorch trainer's decoder_sd is native PyTorch). Per
            # CLAUDE.md "QAT pipeline" non-negotiable + Quantizr 0.33
            # canonical pattern: scalar-weight-only fine-tune at scaled LR
            # (default 0.1x) for fp4_qat_epochs.
            if (
                decoder_quant == "heterogeneous_per_tensor"
                and int(args.fp4_qat_epochs) > 0
            ):
                from tac.substrates.pact_nerv_selector_v3.heterogeneous_bit_allocation import (
                    apply_fp4_qat_finetune_on_top_k_tensors,
                    compute_per_tensor_sensitivity_via_taylor_expansion,
                    derive_heterogeneous_bit_allocation,
                )

                sd_torch_cpu = {
                    k: v.detach().to("cpu", dtype=torch.float32)
                    for k, v in decoder_sd.items()
                }
                sensitivity = compute_per_tensor_sensitivity_via_taylor_expansion(
                    sd_torch_cpu
                )
                allocation = derive_heterogeneous_bit_allocation(
                    sd_torch_cpu, sensitivity
                )
                qat_result = apply_fp4_qat_finetune_on_top_k_tensors(
                    sd_torch_cpu,
                    allocation,
                    qat_epochs=int(args.fp4_qat_epochs),
                    qat_learning_rate_scale=0.1,
                    base_learning_rate=float(args.lr),
                    seed=int(args.seed),
                )
                # Substitute QAT-fine-tuned tensors back so the archive
                # emit's heterogeneous_per_tensor quantization runs over
                # grid-snapped floats (near-zero quantization error per
                # Quantizr).
                for name in qat_result.fp4_tensors_finetuned:
                    decoder_sd[name] = (
                        qat_result.fine_tuned_state_dict[name].detach()
                    )
                qat_metrics = {
                    "fp4_tensors_finetuned": list(qat_result.fp4_tensors_finetuned),
                    "qat_epochs": qat_result.qat_epochs,
                    "qat_learning_rate": qat_result.qat_learning_rate,
                    "final_qat_loss": qat_result.final_qat_loss,
                    "per_tensor_cos_pre_qat": dict(qat_result.per_tensor_cos_pre_qat),
                    "per_tensor_cos_post_qat": dict(qat_result.per_tensor_cos_post_qat),
                    "allocation_rationale": allocation.rationale,
                    "rationale": qat_result.rationale,
                }
                (args.output_dir / "qat_metrics.json").write_text(
                    json.dumps(qat_metrics, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                print(
                    f"[full:{SUBSTRATE_TAG}] FP4-QAT finished: "
                    f"{len(qat_result.fp4_tensors_finetuned)} tensors, "
                    f"{qat_result.qat_epochs} epochs, "
                    f"final_loss={qat_result.final_qat_loss:.6f}"
                )

            # L0 selector posture: per-pair palette mode 0 (the inflate omits
            # selector application at scaffold; Stage 1 wires deterministic FEC6).
            coder = RiceGolombSelectorCoder(
                palette_size=cfg.selector_palette_size, k=cfg.rice_golomb_k
            )
            selector_bytes = coder.encode([0] * result.n_pairs)
            meta = {
                "embed_dim": cfg.embed_dim, "initial_grid_h": cfg.initial_grid_h,
                "initial_grid_w": cfg.initial_grid_w,
                "decoder_channels": list(cfg.decoder_channels),
                "sin_frequency": cfg.sin_frequency,
                "num_upsample_blocks": cfg.num_upsample_blocks,
                "output_height": cfg.output_height, "output_width": cfg.output_width,
                "rice_golomb_k": cfg.rice_golomb_k,
            }
            bin_bytes = pack_archive(
                decoder_sd, latents, selector_bytes, meta,
                palette_size=cfg.selector_palette_size,
                decoder_quantization=decoder_quant,
            )
            (args.output_dir / "0.bin").write_bytes(bin_bytes)
            archive_sha = _sha256_bytes(bin_bytes)
            archive_bytes = len(bin_bytes)
            submission_dir = args.output_dir / "submission"
            write_contest_runtime(
                submission_dir, substrate_pkg_name="pact_nerv_selector_v3",
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
                    env_var_candidates=("PACT_NERV_SELECTOR_V3_GPU", "MODAL_GPU"),
                )
                update = posterior_update_locked(
                    ContestResult(
                        axis="cuda", hardware_substrate=_detected,
                        architecture_class="lane_pact_nerv_selector_v3_l0_scaffold_20260520",
                        score_value=contest_cuda_score, evidence_tag="[contest-CUDA]",
                        archive_sha256=archive_sha, archive_bytes=archive_bytes,
                        notes=f"pact_nerv_selector_v3 first-anchor; epochs={args.epochs}",
                        observed_at_utc=_utc_now_iso(),
                    )
                )
                print(f"[full:{SUBSTRATE_TAG}] posterior_update accepted={update.accepted}")
            except Exception as exc:
                print(f"[full:{SUBSTRATE_TAG}] posterior_update failed: {exc}", file=sys.stderr)

        provenance = {
            "schema": "pact_nerv_selector_v3_full_provenance_v2_decoder_quant_extended",
            "generated_at": _utc_now_iso(),
            "git_head": _git_head_sha(),
            "trainer": "experiments/train_substrate_pact_nerv_selector_v3.py",
            "lane_id": "lane_pact_nerv_selector_v3_l0_scaffold_20260520",
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
            # WAVE-N+3 SLOT 1 op-routable #1 (2026-05-28): record the
            # decoder_quantization kind + QAT metrics so paired-CUDA
            # RATIFICATION cycle has the full provenance for the apples-to-
            # apples comparison (per CLAUDE.md "Apples-to-apples evidence
            # discipline" non-negotiable).
            "decoder_quantization": str(args.decoder_quant),
            "fp4_qat_epochs": int(args.fp4_qat_epochs),
            "top_k_fp4": int(args.top_k_fp4),
            "sensitivity_ranking_method": str(args.sensitivity_ranking_method),
            "brotli_quality_override": (
                int(args.brotli_quality) if args.brotli_quality is not None else None
            ),
            "qat_metrics": qat_metrics if qat_metrics else None,
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


PACT_NERV_SELECTOR_V3_SUBSTRATE_CONTRACT = SubstrateContract(
    id="pact_nerv_selector_v3",
    lane_id="lane_pact_nerv_selector_v3_l0_scaffold_20260520",
    target_modes=("research_substrate",),
    deployment_target="t4_contest_runtime",
    council_verdict_provenance=(
        ".omx/research/pact_nerv_ultimate_research_and_design_20260520T193443Z.md"
    ),
    archive_grammar=(
        "PSV3 monolithic single-file 0.bin (26-byte header; Rice-Golomb-coded "
        "u8 selectors over FEC6 k=16 palette per Golomb 1966 + Rice 1971)"
    ),
    parser_section_manifest={
        "header": "26_byte_fixed_PSV3_magic_v1_palette_size",
        "decoder_blob": "brotli_quality9_pickled_fp16_base_decoder",
        "latent_blob": "raw_int16_row_major",
        "selector_blob": "rice_golomb_coded_u8_selectors_golomb_1966_rice_1971_fec6_k16",
        "meta_blob": "utf8_json_meta_includes_rice_golomb_k",
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
        "hook_sensitivity_contribution": "L0 SCAFFOLD; no sensitivity signal",
        "hook_bit_allocator_class": (
            "Rice-Golomb at selector-symbol granularity; no per-tensor "
            "bit allocator at scaffold posture"
        ),
        "hook_continual_learning_anchor_kind": (
            "L0 SCAFFOLD: no posterior anchor until [contest-CUDA] anchor lands"
        ),
        "hook_probe_disambiguator": (
            "Rice-Golomb k sweep IS the Stage 1 dispatch's empirical purpose "
            "per PACT-NERV-ULTIMATE Variant #12"
        ),
    },
)


@register_substrate(PACT_NERV_SELECTOR_V3_SUBSTRATE_CONTRACT)
def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.smoke:
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
