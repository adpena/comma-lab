# SPDX-License-Identifier: MIT
"""Train the nirvana substrate L0 SCAFFOLD (WAVE-3-NERV-LITERATURE-L0-RESCOPED 2026-05-20).
# NO_GRAD_WAIVED:MLX_substrate_trainer_uses_mx_no_grad_or_substrate_uses_lazy_eval_no_autograd_per_mlx_first_canonical_doctrine_4107bbf8d_or_substrate_eval_uses_alternate_memory_management_per_comprehensive_bug_audit_cascade_20260526

Operator-callable training scaffold per the WAVE-3-NERV-LITERATURE-L0-RESCOPED
queue 2026-05-20. SCAFFOLD-LEVEL: ``_smoke_main`` exercises the substrate
package; ``_full_main`` now routes through the canonical score-aware training
loop and remains CUDA/recipe gated for paid dispatch per Catalog #325.

This trainer's ``SubstrateContract`` declares ``research_only=True``;
``dispatch_enabled: false`` on the matching recipe.

Architectural twist vs ``ds_nerv``: nirvana decodes the frame as a grid of
PATCH_GRID_H x PATCH_GRID_W patches (default 4x4 = 16 patches). Each
patch is decoded by a SHARED per-patch decoder + a learned per-patch
embedding that distinguishes the slot.

Hypothesis (operator's 5-tier fit-ranking MODERATE-HIGH FIT ⭐⭐⭐⭐):
patch-wise specialization stacks orthogonally with global NeRV substrates.
The adaptive per-patch scheduler (training-time only; not in L0 scaffold)
allocates more compute to high-error patches.

Literature anchor: Maiya et al. CVPR 2024 "NIRVANA: Neural Implicit
Representations of Videos with Adaptive Networks and Autoregressive
patch-wise Modeling".

Council-binding contract per CLAUDE.md non-negotiables identical to sister
boost_nerv/ego_nerv L0 SCAFFOLDs.

Usage (smoke; CPU, tiny config)::

    .venv/bin/python experiments/train_substrate_nirvana.py \\
        --output-dir experiments/results/nirvana_smoke_<utc> \\
        --epochs 2 --device cpu --smoke

Usage (full; CUDA-gated and recipe-gated)::

    .venv/bin/python experiments/train_substrate_nirvana.py \\
        --output-dir experiments/results/nirvana_full_<utc> \\
        --device cuda --epochs 1

Cross-ref:
    src/tac/substrates/nirvana/ (substrate package)
    experiments/train_substrate_ds_nerv.py (canonical sister trainer)
    .omx/operator_authorize_recipes/substrate_nirvana_modal_t4_dispatch.yaml
    .omx/research/nirvana_l0_scaffold_design_20260520T184500Z.md
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
CONTEST_AUTH_EVAL_SCRIPT = REPO_ROOT / "experiments" / "contest_auth_eval.py"
EVAL_HW = (384, 512)
N_PAIRS_FULL = 600
CONTEST_NORMALIZER = 37_545_489.0
SUBSTRATE_TAG = "nirvana"


# ---------------------------------------------------------------------------
# Catalog #151 manifest
# ---------------------------------------------------------------------------
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--video-path": {
        "env": "NIRVANA_VIDEO_PATH",
        "rationale": (
            "score-aware substrate MUST train against the contest video; "
            "synthetic data is FORBIDDEN outside --smoke"
        ),
        "default": str(DEFAULT_VIDEO_PATH.relative_to(REPO_ROOT)),
        "satisfied_by_profile": (),
        "requires": (),
        "required_input_file": True,
        "generator_command": (
            "contest-pinned upstream snapshot - never regenerated locally"
        ),
        "rationale_audit": (
            ".omx/research/nirvana_l0_scaffold_design_20260520T184500Z.md"
        ),
    },
    "--output-dir": {
        "env": "NIRVANA_OUTPUT_DIR",
        "rationale": "custody location for checkpoints + archive + provenance",
        "satisfied_by_profile": (), "requires": (),
    },
    "--epochs": {
        "env": "NIRVANA_EPOCHS",
        "rationale": (
            "nirvana substrate engineering pass; council Phase 2 review pending"
        ),
        "default": "2",
        "satisfied_by_profile": (), "requires": (),
    },
    "--upstream-dir": {
        "env": "NIRVANA_UPSTREAM_DIR",
        "rationale": "upstream/ root for scorer weights + evaluate.py",
        "default": str(DEFAULT_UPSTREAM_DIR.relative_to(REPO_ROOT)),
        "satisfied_by_profile": (), "requires": (),
    },
    "--device": {
        "env": "NIRVANA_DEVICE",
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
        prog="train_substrate_nirvana",
        description=(
            "Train nirvana substrate L0 SCAFFOLD (smoke only; full path "
            "gated by Phase 2 council per Catalog #240 + #315 + #325)."
        ),
    )
    p.add_argument("--video-path", type=Path, default=DEFAULT_VIDEO_PATH)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--upstream-dir", type=Path, default=DEFAULT_UPSTREAM_DIR)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seed", type=int, default=20260520)
    p.add_argument("--latent-dim", type=int, default=8)
    p.add_argument(
        "--patch-grid-h", type=int, default=2,
        help="Number of patches vertically (smoke=2; full default 4).",
    )
    p.add_argument(
        "--patch-grid-w", type=int, default=2,
        help="Number of patches horizontally (smoke=2; full default 4).",
    )
    p.add_argument(
        "--patch-embed-dim", type=int, default=4,
        help="Per-patch learned embedding dim (smoke=4; full default 8).",
    )
    # Full-mode score-aware training flags (CLASS-SHIFT-FULL-MAIN-CLUSTER
    # pattern; run_pact_nerv_score_aware_training consumes these).
    p.add_argument("--lr", type=float, default=5e-4, help="AdamW learning rate.")
    p.add_argument(
        "--weight-decay", type=float, default=0.0, help="AdamW weight decay."
    )
    p.add_argument(
        "--grad-clip", type=float, default=1.0,
        help="Gradient L2-norm clip (Council D NaN-watchdog companion).",
    )
    p.add_argument(
        "--ema-decay", type=float, default=0.997,
        help="EMA shadow decay (Quantizr 0.997 default per CLAUDE.md EMA rule).",
    )
    p.add_argument(
        "--alpha-rate", type=float, default=25.0,
        help="Score-domain Lagrangian rate weight (contest canonical 25.0).",
    )
    p.add_argument(
        "--beta-seg", type=float, default=100.0,
        help="Score-domain Lagrangian SegNet weight (contest canonical 100.0).",
    )
    p.add_argument(
        "--gamma-pose", type=float, default=1.0,
        help="Score-domain Lagrangian PoseNet sqrt weight scale.",
    )
    p.add_argument(
        "--pose-weight-scale", type=float, default=1.0,
        help="Pose marginal scaling (2.71 at PR106-r2 operating point).",
    )
    p.add_argument(
        "--noise-std", type=float, default=0.5,
        help="eval_roundtrip noise std (threaded; never disables roundtrip).",
    )
    p.add_argument(
        "--max-pairs", type=int, default=None,
        help="Cap decoded pairs (full default = 600 contest pairs).",
    )
    p.add_argument(
        "--val-pair-count", type=int, default=64,
        help="Held-out validation pair count.",
    )
    p.add_argument(
        "--val-every-epochs", type=int, default=10,
        help="Run validation + best-checkpoint every N epochs.",
    )
    p.add_argument(
        "--skip-archive-build", action="store_true", default=False,
        help="Skip 0.bin + archive.zip emission (training-only diagnostic).",
    )
    p.add_argument(
        "--skip-auth-eval", action="store_true", default=False,
        help="Skip CUDA auth eval (training-only diagnostic).",
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
    """Tiny CPU smoke exercising the nirvana substrate library."""
    import torch

    from tac.differentiable_eval_roundtrip import patch_upstream_yuv6_globally
    from tac.substrates.nirvana.architecture import (
        NirvanaConfig,
        NirvanaSubstrate,
    )

    _pin_seeds(args.seed)
    device = _device_or_die(args.device, smoke=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_at = _utc_now_iso()

    patch_upstream_yuv6_globally()

    cfg = NirvanaConfig(
        latent_dim=args.latent_dim,
        patch_embed_dim=args.patch_embed_dim,
        patch_grid_h=args.patch_grid_h,
        patch_grid_w=args.patch_grid_w,
        embed_dim=16,
        initial_patch_grid_h=3,
        initial_patch_grid_w=4,
        decoder_channels=(16, 12, 8),
        sin_frequency=30.0,
        num_upsample_blocks=3,
        num_pairs=max(2, args.batch_size),
        output_height=24,
        output_width=32,
    )
    model = NirvanaSubstrate(cfg).to(device)
    n_params = model.num_parameters()
    print(f"[smoke] nirvana params: {n_params:,} device={device} "
          f"patch_grid={cfg.patch_grid_h}x{cfg.patch_grid_w} "
          f"num_patches={cfg.patch_grid_h * cfg.patch_grid_w}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    epochs = max(1, min(args.epochs, 3))
    for step in range(epochs):
        idx = torch.arange(cfg.num_pairs, device=device, dtype=torch.long)
        rgb_0, rgb_1 = model(idx)
        # Smoke surrogate loss (NOT score-aware; NOT adaptive per-patch)
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
        env_var_candidates=("NIRVANA_GPU", "MODAL_GPU"),
    )
    provenance = {
        "schema": "nirvana_l0_scaffold_smoke_v1",
        "generated_at": _utc_now_iso(),
        "started_at": started_at,
        "git_head": _git_head_sha(),
        "trainer": "experiments/train_substrate_nirvana.py",
        "lane_id": "lane_nirvana_l0_scaffold_20260520",
        "substrate_tag": SUBSTRATE_TAG,
        "args": {
            k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
        },
        "pytorch_version": _canonical_torch_version_string(),
        "device": str(device),
        "hardware_substrate_detected": detected_substrate,
        "n_params": int(n_params),
        "patch_grid_h": cfg.patch_grid_h,
        "patch_grid_w": cfg.patch_grid_w,
        "num_patches": cfg.patch_grid_h * cfg.patch_grid_w,
        "smoke": True,
        "synthetic_targets_used_per_catalog_114": True,
        "adaptive_scheduler_active": False,
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

    CLASS-SHIFT-FULL-MAIN-CLUSTER 2026-05-27: routes the substrate-AGNOSTIC
    training loop through the canonical
    ``tac.substrates._shared.pact_nerv_full_main`` helper (mirrors the
    implemented ``ds_nerv``/``pact_nerv_ia3`` sisters); the UNIQUE NIRVANA
    distinguishing feature (patch-wise NeRV decode + stitch;
    ``patch_grid_h``/``patch_grid_w``/``patch_embed_dim``) stays in this
    substrate's architecture + archive + score-aware loss. The
    ``NotImplementedError`` is extinguished; PAID DISPATCH is still gated by
    ``dispatch_enabled: false`` + ``research_only: true`` on the recipe per
    Catalog #325 (code complete, trigger gated).

    Honored end-to-end: real contest video (Catalog #114); patch yuv6 BEFORE
    scorer construction (eval_roundtrip non-negotiable); ``load_differentiable_
    scorers`` (no scorer at inflate); score-domain Lagrangian via the variant
    loss → Catalog #164 dispatch; EMA shadow (Quantizr 0.997); CUDA-required
    (``device_or_die`` rejects MPS per Catalog #1); CUDA auth-eval via canonical
    ``gate_auth_eval_call`` (Catalog #226); posterior-update via
    ``posterior_update_locked`` (Catalog #128); contest-compliant runtime
    (Catalog #146 + #295).

    Reactivation criteria for PAID DISPATCH (per HNeRV parity L2): per-substrate
    symposium operator-gated approval (Catalog #325) + cargo-cult audit (Catalog
    #303 — patch-grid + shared-patch-decoder choices flagged CARGO-CULTED) +
    9-dim checklist (#294) + Dykstra band (#296); recipe ``research_only`` flips
    to false + ``dispatch_enabled`` to true. NOTE: the adaptive per-patch
    scheduler (NIRVANA distinctive contribution) is a future L1+ enhancement
    layered on top of this canonical loop.
    """
    from tac.differentiable_eval_roundtrip import (
        patch_upstream_yuv6_globally,
        unpatch_upstream_yuv6,
    )
    from tac.scorer import load_differentiable_scorers
    from tac.substrates._shared.numpy_portable_inflate import (
        write_numpy_portable_contest_runtime,
    )
    from tac.substrates._shared.pact_nerv_full_main import (
        build_archive_zip,
        closed_form_weight_byte_proxy,
        decode_pairs_for_training,
        run_pact_nerv_score_aware_training,
    )
    from tac.substrates.nirvana import (
        NirvanaConfig,
        NirvanaScoreAwareLoss,
        NirvanaSubstrate,
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

        cfg = NirvanaConfig(
            latent_dim=args.latent_dim,
            patch_embed_dim=args.patch_embed_dim,
            patch_grid_h=args.patch_grid_h,
            patch_grid_w=args.patch_grid_w,
            num_pairs=n_pairs,
            output_height=EVAL_HW[0], output_width=EVAL_HW[1],
        )
        model = NirvanaSubstrate(cfg).to(device)
        print(f"[full:{SUBSTRATE_TAG}] params: {model.num_parameters():,} "
              f"patch_grid={cfg.patch_grid_h}x{cfg.patch_grid_w}")

        weights = ScoreAwareLossWeights(
            alpha_rate=args.alpha_rate, beta_seg=args.beta_seg,
            gamma_pose=args.gamma_pose, pose_weight_scale=args.pose_weight_scale,
            contest_normalizer=CONTEST_NORMALIZER,
        )
        loss_fn = NirvanaScoreAwareLoss(
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
        if not args.skip_archive_build:
            sd = result.best_ema_state_dict
            latents = sd["latents"].detach().cpu()
            decoder_sd = {k: v for k, v in sd.items() if k != "latents"}
            meta = {
                "embed_dim": cfg.embed_dim,
                "initial_patch_grid_h": cfg.initial_patch_grid_h,
                "initial_patch_grid_w": cfg.initial_patch_grid_w,
                "decoder_channels": list(cfg.decoder_channels),
                "sin_frequency": cfg.sin_frequency,
                "num_upsample_blocks": cfg.num_upsample_blocks,
                "output_height": cfg.output_height,
                "output_width": cfg.output_width,
            }
            bin_bytes = pack_archive(
                decoder_sd, latents, meta,
                patch_grid_h=cfg.patch_grid_h,
                patch_grid_w=cfg.patch_grid_w,
                patch_embed_dim=cfg.patch_embed_dim,
            )
            (args.output_dir / "0.bin").write_bytes(bin_bytes)
            archive_sha = _sha256_bytes(bin_bytes)
            archive_bytes = len(bin_bytes)
            submission_dir = args.output_dir / "submission"
            write_numpy_portable_contest_runtime(
                submission_dir,
                substrate_pkg_name="nirvana",
                repo_root=REPO_ROOT,
                runtime_module_files=("archive_numpy.py", "inflate.py"),
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
                    env_var_candidates=("NIRVANA_GPU", "MODAL_GPU"),
                )
                update = posterior_update_locked(
                    ContestResult(
                        axis="cuda", hardware_substrate=_detected,
                        architecture_class="lane_nirvana_l0_scaffold_20260520",
                        score_value=contest_cuda_score, evidence_tag="[contest-CUDA]",
                        archive_sha256=archive_sha, archive_bytes=archive_bytes,
                        notes=f"nirvana first-anchor; epochs={args.epochs}",
                        observed_at_utc=_utc_now_iso(),
                    )
                )
                print(f"[full:{SUBSTRATE_TAG}] posterior_update accepted={update.accepted}")
            except Exception as exc:
                print(f"[full:{SUBSTRATE_TAG}] posterior_update failed: {exc}", file=sys.stderr)

        provenance = {
            "schema": "nirvana_full_provenance_v1",
            "generated_at": _utc_now_iso(),
            "git_head": _git_head_sha(),
            "trainer": "experiments/train_substrate_nirvana.py",
            "lane_id": "lane_nirvana_l0_scaffold_20260520",
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


# ---------------------------------------------------------------------------
# META layer SubstrateContract (Catalog #241/#242)
# ---------------------------------------------------------------------------

NIRVANA_SUBSTRATE_CONTRACT = SubstrateContract(
    id="nirvana",
    lane_id="lane_nirvana_l0_scaffold_20260520",
    target_modes=("research_substrate",),
    deployment_target="t4_contest_runtime",
    council_verdict_provenance=(
        ".omx/research/nirvana_l0_scaffold_design_20260520T184500Z.md"
    ),
    archive_grammar=(
        "NRV1 monolithic single-file 0.bin (25-byte header carrying "
        "PATCH_GRID_H/W u8 + PATCH_EMBED_DIM u16 distinctive fields; "
        "shared per-patch decoder weights + patch embeddings in single "
        "brotli blob; int16 latents; utf-8 json meta)"
    ),
    parser_section_manifest={
        "header": "25_byte_fixed_NRV1_magic_v1_patch_grid_embed_dim",
        "decoder_blob": "brotli_quality9_pickled_fp16_shared_decoder_plus_patch_embeddings",
        "per_patch_decoder_weights_subset": "logical_grouping_inside_decoder_blob",
        "patch_embeddings_subset": "logical_grouping_inside_decoder_blob",
        "latent_blob": "raw_int16_row_major",
        "meta_blob": "utf8_json_meta",
    },
    inflate_runtime_loc_budget=200,
    runtime_dep_closure=("torch>=2.5,<2.7", "brotli", "av"),
    export_format="custom",
    score_aware_loss="scorer_loss_terms_btchw",
    bolt_on_loc_budget=750,
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
            "fp16 brotli on shared per-patch decoder + patch embeddings blob; "
            "no per-tensor bit allocator at scaffold posture (per-patch "
            "decoder weight independence is the L1+ research path per "
            "cargo-cult audit)"
        ),
        "hook_continual_learning_anchor_kind": (
            "L0 SCAFFOLD: no posterior anchor until full path lands and a "
            "[contest-CUDA] anchor is measured"
        ),
        "hook_probe_disambiguator": (
            "single mechanism (patch-wise + shared decoder); the adaptive "
            "scheduler alternative (training-time only) is the L1+ Phase 2 "
            "research path"
        ),
    },
)


@register_substrate(NIRVANA_SUBSTRATE_CONTRACT)
def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.smoke:
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
