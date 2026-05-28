# SPDX-License-Identifier: MIT
# TF32_WAIVED:canonical_trainer_skeleton.device_or_die_enables_tf32_per_helper_trainer_skeleton_py_lines_714_715_via_torch_backends_cuda_matmul_allow_tf32_true_and_torch_backends_cudnn_allow_tf32_true_per_sister_a1_plus_lapose_d1_d4_pact_nerv_ia3_paired_cuda_dispatch_pattern_landed_20260528
"""Train the pact_nerv_ia3 substrate L0 SCAFFOLD (WAVE-3-PACT-NERV-IA3-L0-BUILD-STAGE-1 2026-05-20).
# NO_GRAD_WAIVED:MLX_substrate_trainer_uses_mx_no_grad_or_substrate_uses_lazy_eval_no_autograd_per_mlx_first_canonical_doctrine_4107bbf8d_or_substrate_eval_uses_alternate_memory_management_per_comprehensive_bug_audit_cascade_20260526

Operator-callable training scaffold per the PACT-NERV-DESIGN-SYMPOSIUM
HYBRID Stage 1 verdict 2026-05-20 (commit `5371d4dd4`). SCAFFOLD-LEVEL:
``_smoke_main`` exercises the substrate package; ``_full_main`` raises
``NotImplementedError`` per CLAUDE.md "Substrate scaffolds MUST be COMPLETE
or RESEARCH-ONLY" non-negotiable + Catalog #220 (substrate L1+ scaffold
operational mechanism declaration) + Catalog #240 (recipe-vs-trainer-state
consistency) + Catalog #315 (OPTIMAL FORM before paid dispatch) + Catalog
#325 (per-substrate symposium).

SUPERSEDED 2026-05-28 per PACT-NERV-FULL-MAIN-IMPLEMENTATION-WAVE commit
259292757 + CATALOG-240-LANE-SCRIPT-DRIFT-FIX task #1437: ``_full_main`` is
now IMPLEMENTED (canonical ``tac.substrates._shared.pact_nerv_full_main``
helper + score-aware loss + ``gate_auth_eval_call`` all wired). The
``NotImplementedError`` is extinguished at the function-body level. PAID
DISPATCH is STILL gated by ``dispatch_enabled: false`` + ``research_only:
true`` on the L0 SCAFFOLD recipe per Catalog #325 until the per-substrate
symposium clears it; the PAIRED-DISPATCH recipe at
``.omx/operator_authorize_recipes/substrate_pact_nerv_ia3_modal_t4_paired_dispatch.yaml``
carries the canonical operator-frontier-override per Catalog #300
Consequence 1 for the L0->L1 promotion path. Original SCAFFOLD claim
preserved above per Catalog #110/#113 APPEND-ONLY HISTORICAL_PROVENANCE.

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
- Non-smoke path IS IMPLEMENTED post-2026-05-28 commit 259292757 (was
  ``NotImplementedError`` at L0 SCAFFOLD landing 2026-05-20); paid dispatch
  remains gated by recipe-level ``dispatch_enabled:false`` per Catalog #325.
- eval_roundtrip MANDATORY DEFAULT per Catalog #6: smoke patches
  differentiable yuv6 BEFORE any future scorer construction.

Usage (smoke; CPU, tiny config, ~2 steps, synthetic batches)::

    .venv/bin/python experiments/train_substrate_pact_nerv_ia3.py \\
        --output-dir experiments/results/pact_nerv_ia3_smoke_<utc> \\
        --epochs 2 --device cpu --smoke

Usage (full; IMPLEMENTED post-2026-05-28 commit 259292757; gated by recipe
``dispatch_enabled:false`` per Catalog #325 until per-substrate symposium
clears or the canonical operator-frontier-override per Catalog #300 is
invoked on the paired-dispatch recipe)::

    .venv/bin/python experiments/train_substrate_pact_nerv_ia3.py \\
        --video-path upstream/videos/0.mkv \\
        --output-dir experiments/results/pact_nerv_ia3_full_<utc> \\
        --epochs 100 --device cuda \\
        --upstream-dir upstream

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
from tac.substrates._shared.smoke_auth_eval_gate import (
    gate_auth_eval_call as _canon_gate_auth_eval_call,
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
EVAL_HW = (384, 512)
N_PAIRS_FULL = 600
CONTEST_NORMALIZER = 37_545_489.0
SUBSTRATE_TAG = "pact_nerv_ia3"
CONTEST_AUTH_EVAL_SCRIPT = REPO_ROOT / "experiments" / "contest_auth_eval.py"


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
    # Full-path (non-smoke) score-aware training flags. Reachable only when
    # the recipe flips research_only:false + dispatch_enabled:true per
    # Catalog #325; the body trains a real score-aware Lagrangian.
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
    """Full score-aware training entry point — CUDA-required; paid-GPU gated.

    PACT-NERV-FULL-MAIN-IMPLEMENTATION-WAVE 2026-05-27: the substrate-AGNOSTIC
    training loop is the canonical ``tac.substrates._shared.pact_nerv_full_main``
    helper (mirrors the implemented ``ds_nerv`` sister); the UNIQUE IA3 γ-only
    distinguishing feature stays in this substrate's architecture + archive +
    score-aware loss. The ``NotImplementedError`` is extinguished; the PAID
    DISPATCH is still gated by ``dispatch_enabled: false`` + ``research_only:
    true`` on the recipe per Catalog #325 until the per-substrate symposium
    clears it (code complete, trigger gated — the canonical "implement all
    without firing council-gated paid paths" resolution).

    Per CLAUDE.md non-negotiables honored end-to-end: real contest video
    (Catalog #114); ``patch_upstream_yuv6_globally`` BEFORE scorer construction
    (eval_roundtrip non-negotiable); ``load_differentiable_scorers`` (no scorer
    at inflate); score-domain Lagrangian via ``PactNervIa3ScoreAwareLoss`` →
    Catalog #164 ``score_pair_components_dispatch``; EMA shadow (Quantizr
    0.997); CUDA-required (``device_or_die`` rejects MPS per Catalog #1); CUDA
    auth-eval via canonical ``gate_auth_eval_call`` (Catalog #226);
    posterior-update via ``posterior_update_locked`` (Catalog #128);
    contest-compliant numpy/PIL-portable runtime emission (Catalog #146 + #295).

    Reactivation criteria for PAID DISPATCH (per HNeRV parity L2):
    1. PACT-NERV symposium Stage 1 operator-gated approval (Catalog #325).
    2. Cargo-cult audit (Catalog #303): per-block modulation; shared γ_proj.
    3. 9-dim checklist (#294) + observability (#305) + Dykstra band (#296).
    4. Recipe ``research_only`` flips to false + ``dispatch_enabled`` to true.
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
    from tac.substrates._shared.trainer_skeleton import (
        build_optimized_training_context as _canon_build_optimized_training_context,
    )
    from tac.substrates.pact_nerv_ia3 import (
        PactNervIa3Config,
        PactNervIa3ScoreAwareLoss,
        PactNervIa3Substrate,
        ScoreAwareLossWeights,
        pack_archive,
    )

    _pin_seeds(args.seed)
    device = _device_or_die(args.device, smoke=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage_log: list[dict[str, Any]] = []

    def _stage(name: str) -> None:
        stage_log.append({"stage": name, "at": _utc_now_iso()})

    _stage("seed_pinned")
    yuv6_token = patch_upstream_yuv6_globally()
    _stage("upstream_yuv6_patched")

    try:
        posenet, segnet = load_differentiable_scorers(args.upstream_dir, device=device)
        for p in list(posenet.parameters()) + list(segnet.parameters()):
            p.requires_grad_(False)
        posenet.eval()
        segnet.eval()
        _stage("scorers_loaded")

        print(f"[full:{SUBSTRATE_TAG}] decoding pairs from {args.video_path} ...")
        pair_tensor = decode_pairs_for_training(
            args.video_path,
            substrate_tag=SUBSTRATE_TAG,
            n_pairs=N_PAIRS_FULL,
            max_pairs=args.max_pairs,
            repo_root=REPO_ROOT,
        ).to(device)
        n_pairs = int(pair_tensor.shape[0])
        print(f"[full:{SUBSTRATE_TAG}] decoded {n_pairs} pairs at {EVAL_HW}")
        _stage(f"pairs_decoded_{n_pairs}")

        cfg = PactNervIa3Config(
            latent_dim=args.latent_dim,
            sin_frequency=30.0,
            pose_dim=args.pose_dim,
            ia3_init_delta_std=args.ia3_init_delta_std,
            num_pairs=n_pairs,
            output_height=EVAL_HW[0],
            output_width=EVAL_HW[1],
        )
        model = PactNervIa3Substrate(cfg).to(device)
        print(
            f"[full:{SUBSTRATE_TAG}] params: {model.num_parameters():,} "
            f"(IA3 γ modulation: {model.num_ia3_modulation_parameters():,})"
        )
        _stage("model_built")

        weights = ScoreAwareLossWeights(
            alpha_rate=args.alpha_rate,
            beta_seg=args.beta_seg,
            gamma_pose=args.gamma_pose,
            pose_weight_scale=args.pose_weight_scale,
            contest_normalizer=CONTEST_NORMALIZER,
        )
        loss_fn = PactNervIa3ScoreAwareLoss(
            seg_scorer=segnet, pose_scorer=posenet, weights=weights
        )
        _stage("lagrangian_built")

        opt_ctx = _canon_build_optimized_training_context(
            args,
            scorers=(posenet, segnet),
            gt_pairs=pair_tensor,
            substrate_model=model,
            device=device,
        )
        gt_cache = opt_ctx.gt_cache
        _stage("gt_scorer_cache_built" if gt_cache is not None else "gt_scorer_cache_disabled")

        archive_bytes_proxy = closed_form_weight_byte_proxy(model)

        def _compute_loss(
            m, idx, gt_0, gt_1, abp, *, gt_pose_batch, gt_seg_batch, gt_seg_already_probs
        ):
            rgb_0, rgb_1 = m(idx)
            return loss_fn(
                rgb_0 * 255.0, rgb_1 * 255.0, gt_0, gt_1, abp,
                apply_eval_roundtrip=True,
                noise_std=args.noise_std,
                gt_pose_batch=gt_pose_batch,
                gt_seg_batch=gt_seg_batch,
                gt_seg_already_probs=gt_seg_already_probs,
            )

        result = run_pact_nerv_score_aware_training(
            model=model,
            pair_tensor=pair_tensor,
            compute_loss=_compute_loss,
            archive_bytes_proxy=archive_bytes_proxy,
            device=device,
            output_dir=args.output_dir,
            substrate_tag=SUBSTRATE_TAG,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            ema_decay=args.ema_decay,
            val_pair_count=args.val_pair_count,
            val_every_epochs=args.val_every_epochs,
            gt_cache=gt_cache,
            stage_log=stage_log,
            config_asdict=asdict(cfg),
        )
        print(
            f"[full:{SUBSTRATE_TAG}] train done: best_val_lag="
            f"{result.best_val_lagrangian:.6f} @ ep{result.best_epoch + 1} "
            f"elapsed={result.train_elapsed_sec:.1f}s"
        )

        archive_sha = ""
        archive_bytes = 0
        archive_zip_path = args.output_dir / "archive.zip"
        if not args.skip_archive_build:
            sd = result.best_ema_state_dict
            latents = sd["latents"].detach().cpu()
            ego_poses = sd["ego_poses"].detach().cpu()
            decoder_sd = {
                k: v for k, v in sd.items() if k not in ("latents", "ego_poses")
            }
            meta = {
                "embed_dim": cfg.embed_dim,
                "initial_grid_h": cfg.initial_grid_h,
                "initial_grid_w": cfg.initial_grid_w,
                "decoder_channels": list(cfg.decoder_channels),
                "sin_frequency": cfg.sin_frequency,
                "output_height": cfg.output_height,
                "output_width": cfg.output_width,
                "num_upsample_blocks": cfg.num_upsample_blocks,
                "ia3_init_delta_std": cfg.ia3_init_delta_std,
            }
            bin_bytes = pack_archive(
                decoder_sd, latents, ego_poses, meta, pose_dim=cfg.pose_dim
            )
            (args.output_dir / "0.bin").write_bytes(bin_bytes)
            archive_sha = _sha256_bytes(bin_bytes)
            archive_bytes = len(bin_bytes)
            print(f"[full:{SUBSTRATE_TAG}] wrote 0.bin ({archive_bytes} B sha={archive_sha[:16]})")

            submission_dir = args.output_dir / "submission"
            write_contest_runtime(
                submission_dir,
                substrate_pkg_name="pact_nerv_ia3",
                repo_root=REPO_ROOT,
            )
            (submission_dir / "0.bin").write_bytes(bin_bytes)
            build_archive_zip(
                archive_zip_path, bin_bytes=bin_bytes, submission_dir=submission_dir
            )
            print(f"[full:{SUBSTRATE_TAG}] wrote {archive_zip_path}")
            _stage(f"archive_built_bytes_{archive_bytes}")

        auth_eval_result_path: Path | None = None
        contest_cuda_score: float | None = None
        if not args.skip_auth_eval and archive_zip_path.is_file():
            print(f"[full:{SUBSTRATE_TAG}] launching CUDA auth eval ...")
            auth_eval_result_path = args.output_dir / "contest_auth_eval_cuda.json"
            auth_result = _canon_gate_auth_eval_call(
                args=args,
                archive_zip=archive_zip_path,
                inflate_sh=args.output_dir / "submission" / "inflate.sh",
                upstream_dir=args.upstream_dir,
                output_json=auth_eval_result_path,
                contest_auth_eval_script=CONTEST_AUTH_EVAL_SCRIPT,
                substrate_tag=SUBSTRATE_TAG,
                device=device,
            )
            if auth_result is not None:
                contest_cuda_score = auth_result["auth_eval_cuda_score"]
                print(
                    f"[full:{SUBSTRATE_TAG}] [contest-CUDA] score = "
                    f"{contest_cuda_score} (archive_sha256={archive_sha})"
                )
            _stage("auth_eval_cuda_done")

        if contest_cuda_score is not None and archive_sha:
            try:
                from tac.continual_learning import (
                    ContestResult,
                    posterior_update_locked,
                )

                _detected = _canon_detect_hardware_substrate(
                    axis="cuda",
                    substrate_tag=SUBSTRATE_TAG,
                    provenance_path=args.output_dir / "provenance.json",
                    env_var_candidates=("PACT_NERV_IA3_GPU", "MODAL_GPU"),
                )
                update = posterior_update_locked(
                    ContestResult(
                        axis="cuda",
                        hardware_substrate=_detected,
                        architecture_class="lane_pact_nerv_ia3_l0_scaffold_20260520",
                        score_value=contest_cuda_score,
                        evidence_tag="[contest-CUDA]",
                        archive_sha256=archive_sha,
                        archive_bytes=archive_bytes,
                        notes=f"pact_nerv_ia3 first-anchor; epochs={args.epochs}",
                        observed_at_utc=_utc_now_iso(),
                    )
                )
                print(
                    f"[full:{SUBSTRATE_TAG}] posterior_update: "
                    f"accepted={update.accepted} reason={update.reason!r}"
                )
            except Exception as exc:
                print(f"[full:{SUBSTRATE_TAG}] posterior_update failed: {exc}", file=sys.stderr)

        provenance = {
            "schema": "pact_nerv_ia3_full_provenance_v1",
            "generated_at": _utc_now_iso(),
            "git_head": _git_head_sha(),
            "trainer": "experiments/train_substrate_pact_nerv_ia3.py",
            "lane_id": "lane_pact_nerv_ia3_l0_scaffold_20260520",
            "substrate_tag": SUBSTRATE_TAG,
            "args": {
                k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
            },
            "pytorch_version": _canonical_torch_version_string(),
            "device": str(device),
            "num_pairs_decoded": result.n_pairs,
            "num_train_pairs": result.n_train_pairs,
            "num_val_pairs": result.n_val_pairs,
            "best_val_lagrangian": (
                result.best_val_lagrangian
                if result.best_val_lagrangian == result.best_val_lagrangian
                else None
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
            "score_axis_tag": (
                "[contest-CUDA]" if contest_cuda_score is not None else None
            ),
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
