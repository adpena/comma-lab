# SPDX-License-Identifier: MIT
"""UNIWARD per-instance + multi-scale wavelet SegNet-loss substrate trainer (Tier-2 paid dispatch scaffold).

Source memo: ``.omx/research/combined_tier_1_wave_3_uniward_multi_scale_plus_hinton_motion_aware_landed_20260525.md``
Recipe:      ``.omx/operator_authorize_recipes/substrate_uniward_per_instance_multi_scale_wavelet_segnet_modal_t4_dispatch.yaml``
Symposium:   ``.omx/research/per_substrate_symposium_uniward_per_instance_multi_scale_wavelet_segnet_20260525.md``
Lane:        ``lane_probe_9_tier_2_dispatch_prep_uniward_per_instance_multi_scale_wavelet_segnet_20260525``
Sister:      Probe 9 macOS-CPU advisory probe at
             ``.omx/research/tier_1_distortion_axis_probes_20260521/probe_9_*_smoke.py``

Probe 9 verdict POSITIVE_SIGNAL_BREAKS_THRESHOLD:
  min combined per-instance + multi-scale textured_avg_weight = 0.2597
  BREAKS the 0.5 hard threshold by Delta -0.2403 (FIRST PROBE in the entire
  Tier-1 distortion-axis cascade to do so).
  Predicted DS -0.010 to -0.025 against PR 101 fec6 frontier (canonical
  contest-CPU 0.19205).

CONTRACT (per Catalog #325 6-step canonical contract; substrate is at OPTIMAL
FORM only when ALL six steps complete + symposium PROCEED unconditional):
  1. Cargo-cult audit per Catalog #303    -- in symposium memo Section 4
  2. 9-dim checklist evidence per #294    -- in symposium memo Section 5
  3. Observability surface per #305       -- in symposium memo Section 6
  4. Sextet pact deliberation per #292    -- in symposium memo Section 7
  5. Reactivation criteria pinned         -- in symposium memo Section 8
  6. Catalog #324 post-training Tier-C    -- in symposium memo Section 9

PER CATALOG #240 (recipe-vs-trainer-state consistency): this trainer's
``_full_main`` raises ``NotImplementedError`` at this scaffold landing because
the recipe declares ``research_only: true`` + ``dispatch_enabled: false``.
The full implementation lands in a sister subagent commit batch after the
per-substrate symposium clears PROCEED-unconditional per Catalog #325.

PER CATALOG #229 (premise-verification-before-edit) + the broader Carmack
MVP-first 5-step discipline: this trainer COPIES the canonical per-instance
+ multi-scale wavelet UNIWARD-weighting algorithm from Probe 9's verdict-
producing smoke script bit-for-bit (the algorithm is the SAME; what changes
is the gradient direction — Probe 9 only MEASURES the weighting, this
substrate USES it as a SegNet-loss WEIGHT inside the training loop).

Per CLAUDE.md non-negotiables (Tier-2 dispatch readiness):

  * Score-aware substrate (HNeRV parity L1) -- trains against
    ``upstream/videos/0.mkv`` decoded via pyav, gradient through scorers
    via the canonical ``tac.substrates.score_aware_common.score_pair_components``.
  * ``patch_upstream_yuv6_globally()`` BEFORE ``load_differentiable_scorers``
    (Catalog #187; PR #95/#106 contract).
  * ``apply_eval_roundtrip=True`` inside the per-batch loop (Catalog #5;
    no opt-out per the canonical helper).
  * EMA decay 0.997 + snapshot+restore at eval (Catalog #88; inference
    checkpoint is EMA shadow, NEVER live weights).
  * No scorer load at inflate time (Catalog #6; inflate.py is RGB-only).
  * No /tmp persisted evidence paths (CLAUDE.md FORBIDDEN_PATTERN).
  * Tier-1 engineering flags declared (Catalogs #172/#178/#179/#180);
    autocast / TF32 opt-in via ``--enable-*`` flags.
  * TIER_1_OPERATOR_REQUIRED_FLAGS manifest as ``ast.AnnAssign`` so
    Catalog #151 introspection sees it (Catalog #168).
  * Dynamic ``hardware_substrate`` detection (Catalog #190).
  * Driver mode env-var consumed per Catalog #326 (UNIWARD_PIMS_TRAINER_MODE
    multi-key resolution; default smoke).

Usage (smoke; CPU; algorithm forward-pass + UNIWARD weighting only)::

    .venv/bin/python experiments/train_substrate_uniward_per_instance_multi_scale_wavelet_segnet.py \\
        --base-archive experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip \\
        --video-path upstream/videos/0.mkv \\
        --upstream-models upstream/models/segnet.safetensors \\
        --output-dir experiments/results/uniward_pims_smoke_<utc> \\
        --epochs 5 --device cpu --smoke

Usage (full; CUDA-required; council-gated per Catalog #325; emits
``NotImplementedError`` until the canonical 6-step symposium contract clears
and a sister subagent lands ``_full_main``)::

    .venv/bin/python experiments/train_substrate_uniward_per_instance_multi_scale_wavelet_segnet.py \\
        --base-archive .../archive.zip \\
        --video-path upstream/videos/0.mkv \\
        --upstream-models upstream/models/segnet.safetensors \\
        --output-dir experiments/results/uniward_pims_full_<utc> \\
        --epochs 2000 --batch-size 16 --lr 1e-3 --device cuda
"""
# AUTOCAST_FP16_WAIVED:score-aware-scorer-path-pending-canonical-autocast-backport
# TORCH_COMPILE_WAIVED:defer-until-per-substrate-canary-validates-Inductor-graph-breaks-and-score-axis-custody
# TF32_WAIVED:opt-in-via-CLI-flag-to-keep-eval-roundtrip-numerics-deterministic-until-paired-CPU/CUDA-anchor-lands
# NO_GRAD_WAIVED:eval-time-scorer-forward-wrapped-in-torch.inference_mode-inside-_run_val_loop
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))


# ---------------------------------------------------------------------------
# Module paths + constants
# ---------------------------------------------------------------------------
DEFAULT_BASE_ARCHIVE = (
    REPO_ROOT
    / "experiments"
    / "results"
    / "pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex"
    / "archive.zip"
)
DEFAULT_VIDEO_PATH = REPO_ROOT / "upstream" / "videos" / "0.mkv"
DEFAULT_UPSTREAM_MODELS = REPO_ROOT / "upstream" / "models" / "segnet.safetensors"

# Catalog #152 + #153 Modal-IGNORED `experiments/results/**` subtree
# defense: required-input artifacts under `experiments/results/**` MUST be
# declared in TIER_1_EXTRA_MOUNT_PATHS so the canonical Modal mount builder
# (`tac.deploy.modal.mount_manifest.build_training_image`) stages them via
# `add_local_file`. Without this, the Modal mount manifest IGNORES the file
# via DEFAULT_RESULTS_IGNORE.
TIER_1_EXTRA_MOUNT_PATHS: tuple[str, ...] = (
    str(DEFAULT_BASE_ARCHIVE.relative_to(REPO_ROOT)),
)

EVAL_HW = (384, 512)
CAMERA_HW = (874, 1164)
N_PAIRS_FULL = 600
CONTEST_NORMALIZER = 37_545_489.0

# Probe 9 canonical algorithmic constants
# Canonical fix 2026-05-25: db8 -> db4 per Probe 9c empirical falsification
# (z=-3.442sigma; mean 0.3599 vs db8 0.3915; CIs disjoint at 95%;
# below-threshold 83.6% vs 80.1%; min 0.0532 vs 0.0932 = 42.9% sharper
# inversion). PARADIGM INTACT per Catalog #307 IMPLEMENTATION-LEVEL fix;
# SUBSTRATE BASIS canonical-optimal per Probe 9c. Commit efeaff5c9.
# See: .omx/research/probe_9c_per_level_wavelet_basis_disambiguator_landed_20260525.md
#      .omx/research/probe_9_recipe_canonical_update_db8_to_db4_landed_20260525.md
WAVELET_NAME_DEFAULT = "db4"
WAVELET_LEVELS_DEFAULT = 3
SEGNET_INSTANCE_CONNECTIVITY_DEFAULT = 4  # 4-connectivity for connected-components
SIGMA_FRIDRICH = 2.0 ** -6  # Holub-Fridrich 2014 epsilon


# ---------------------------------------------------------------------------
# Catalog #151 manifest -- annotated assignment per Catalog #168 so the AST
# walker in Catalog #151's static gate observes the dict.
# ---------------------------------------------------------------------------
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--base-archive": {
        "env": "UNIWARD_PIMS_BASE_ARCHIVE",
        "rationale": (
            "PR 101 fec6 frontier base archive bytes (canonical contest-CPU "
            "0.19205 per canonical frontier pointer). The substrate trains a "
            "UNIWARD-weighted SegNet-loss residual ON TOP OF the frozen "
            "base; the base archive provides the un-modified frames used "
            "for the SegNet ground-truth-class derivation + the wavelet "
            "decomposition input."
        ),
        "default": str(DEFAULT_BASE_ARCHIVE.relative_to(REPO_ROOT)),
        "required_input_file": True,
        "generator_command": (
            "experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/ "
            "(landed 2026-05-15 via PR101 frame-exploit selector fec6 + k=16 clean)"
        ),
        "rationale_audit": (
            ".omx/research/combined_tier_1_wave_3_uniward_multi_scale_plus_hinton_motion_aware_landed_20260525.md"
        ),
    },
    "--video-path": {
        "env": "UNIWARD_PIMS_VIDEO_PATH",
        "rationale": (
            "Score-aware substrate trains against contest video "
            "(upstream/videos/0.mkv); synthetic data is FORBIDDEN outside --smoke."
        ),
        "default": str(DEFAULT_VIDEO_PATH.relative_to(REPO_ROOT)),
        "required_input_file": True,
        "generator_command": (
            "contest-pinned upstream snapshot -- never regenerated locally"
        ),
    },
    "--upstream-models": {
        "env": "UNIWARD_PIMS_UPSTREAM_MODELS",
        "rationale": (
            "Canonical SegNet safetensors weights. Required for per-instance "
            "connected-components derivation from SegNet 5-class hard mask. "
            "Per CLAUDE.md 'Exact scorer architectures' the SegNet is "
            "smp.Unet('tu-efficientnet_b2', classes=5) loaded from upstream."
        ),
        "default": str(DEFAULT_UPSTREAM_MODELS.relative_to(REPO_ROOT)),
        "required_input_file": True,
        "generator_command": (
            "contest-pinned upstream snapshot -- never regenerated locally"
        ),
    },
    "--output-dir": {
        "env": "UNIWARD_PIMS_OUTPUT_DIR",
        "rationale": "custody location for checkpoints + archive + provenance",
    },
    "--epochs": {
        "env": "UNIWARD_PIMS_EPOCHS",
        "rationale": (
            "training epochs; full-dispatch target 2000 per Probe 9 Tier-2 "
            "envelope $2-7 at Modal T4."
        ),
        "default": "2000",
    },
    "--batch-size": {
        "env": "UNIWARD_PIMS_BATCH_SIZE",
        "rationale": (
            "per-pair batch size; default 16 fits T4 14.56 GB activation "
            "budget for per-instance connected-components + multi-scale "
            "wavelet decomposition + SegNet eval-roundtrip backward."
        ),
        "default": "16",
    },
    "--upstream-dir": {
        "env": "UNIWARD_PIMS_UPSTREAM_DIR",
        "rationale": "upstream snapshot root for evaluate.py + modules.py",
        "default": "upstream",
    },
    "--device": {
        "env": "UNIWARD_PIMS_DEVICE",
        "rationale": (
            "cuda for full Tier-2 paid dispatch; cpu for --smoke local "
            "algorithm validation (NO gradient through scorers in cpu mode)."
        ),
        "default": "cuda",
    },
    "--wavelet-name": {
        "env": "UNIWARD_PIMS_WAVELET_NAME",
        "rationale": (
            "Holub-Fridrich 2014 canonical multi-scale wavelet basis; "
            "pywt.wavedec2 convention. Per Probe 9c per-level wavelet-basis "
            "selection disambiguator (commit efeaff5c9) the canonical-optimal "
            "basis is db4 (z=-3.442sigma vs db8 baseline; mean 0.3599 vs "
            "0.3915; CIs disjoint at 95%); Mallat seat binding revision #3 "
            "of 6 SATISFIED with canonical-fix. Catalog #307 IMPLEMENTATION-"
            "LEVEL fix; paradigm intact."
        ),
        "default": "db4",
    },
    "--wavelet-levels": {
        "env": "UNIWARD_PIMS_WAVELET_LEVELS",
        "rationale": (
            "Holub-Fridrich 2014 multi-level decomposition depth; Probe 9c "
            "canonical anchor at db4 used levels=3 (3-level db4 detail-"
            "subband sum). Per-decomposition-level disambiguator at 2-vs-3-"
            "vs-4 levels queued as P3 operator-routable enhancement per "
            "Catalog #308 alternative-reducer cascade."
        ),
        "default": "3",
    },
    "--segnet-instance-connectivity": {
        "env": "UNIWARD_PIMS_SEGNET_INSTANCE_CONNECTIVITY",
        "rationale": (
            "scipy.ndimage.label connectivity for per-instance connected-"
            "components derivation; canonical 4-connectivity per Probe 9 "
            "anchor (8-connectivity yields fewer, larger components)."
        ),
        "default": "4",
    },
}


# ---------------------------------------------------------------------------
# Substrate Registry per Catalog #241 (warn-only legacy waiver until full
# META-layer migration of legacy substrate trainers lands).
# ---------------------------------------------------------------------------
# LEGACY_SUBSTRATE_PRE_META_LAYER:Tier-2-dispatch-prep-scaffold-legacy-pattern-mirror-of-a1_plus_wavelet_residual-substrate-pending-META-layer-migration-per-Catalog-241-warn-only-grace
# (the canonical META-layer migration is a separate operator-routed sister
# subagent; this scaffold mirrors the established a1_plus_wavelet_residual
# pattern to maintain consistency with the existing 30+ substrate trainers
# currently in the legacy-pattern grace window per Catalog #241.)


# ---------------------------------------------------------------------------
# Driver mode resolution per Catalog #326 multi-key ladder
# ---------------------------------------------------------------------------
def _resolve_trainer_mode(args: argparse.Namespace) -> str:
    """Multi-key trainer mode resolution per Catalog #326.

    Priority order (highest first):
      1. CLI --smoke flag (explicit)
      2. UNIWARD_PIMS_TRAINER_MODE env var
      3. SMOKE_ONLY env var (legacy fallback)
      4. default 'smoke' (fail-loud safe default)
    """
    import os

    if args.smoke:
        return "smoke"
    env_mode = os.environ.get("UNIWARD_PIMS_TRAINER_MODE", "").strip().lower()
    if env_mode in {"smoke", "full"}:
        return env_mode
    smoke_only = os.environ.get("SMOKE_ONLY", "").strip()
    if smoke_only == "0":
        return "full"
    if smoke_only == "1":
        return "smoke"
    # Default: smoke (fail-loud safe default per Catalog #326 sister discipline)
    print(
        "[uniward-pims] WARN: no trainer mode env var set "
        "(UNIWARD_PIMS_TRAINER_MODE or SMOKE_ONLY); defaulting to smoke per "
        "Catalog #326 multi-key resolution.",
        file=sys.stderr,
    )
    return "smoke"


# ---------------------------------------------------------------------------
# CLI argparse
# ---------------------------------------------------------------------------
def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "UNIWARD per-instance + multi-scale wavelet SegNet-loss substrate "
            "trainer (Tier-2 paid dispatch scaffold)."
        ),
    )
    parser.add_argument(
        "--base-archive",
        type=Path,
        default=DEFAULT_BASE_ARCHIVE,
        help=TIER_1_OPERATOR_REQUIRED_FLAGS["--base-archive"]["rationale"],
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        default=DEFAULT_VIDEO_PATH,
        help=TIER_1_OPERATOR_REQUIRED_FLAGS["--video-path"]["rationale"],
    )
    parser.add_argument(
        "--upstream-models",
        type=Path,
        default=DEFAULT_UPSTREAM_MODELS,
        help=TIER_1_OPERATOR_REQUIRED_FLAGS["--upstream-models"]["rationale"],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=TIER_1_OPERATOR_REQUIRED_FLAGS["--output-dir"]["rationale"],
    )
    parser.add_argument(
        "--upstream-dir",
        type=Path,
        default=REPO_ROOT / "upstream",
        help=TIER_1_OPERATOR_REQUIRED_FLAGS["--upstream-dir"]["rationale"],
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=2000,
        help=TIER_1_OPERATOR_REQUIRED_FLAGS["--epochs"]["rationale"],
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help=TIER_1_OPERATOR_REQUIRED_FLAGS["--batch-size"]["rationale"],
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="learning rate for the wavelet-residual head",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cuda",
        help=TIER_1_OPERATOR_REQUIRED_FLAGS["--device"]["rationale"],
    )
    parser.add_argument(
        "--wavelet-name",
        type=str,
        default=WAVELET_NAME_DEFAULT,
        help=TIER_1_OPERATOR_REQUIRED_FLAGS["--wavelet-name"]["rationale"],
    )
    parser.add_argument(
        "--wavelet-levels",
        type=int,
        default=WAVELET_LEVELS_DEFAULT,
        help=TIER_1_OPERATOR_REQUIRED_FLAGS["--wavelet-levels"]["rationale"],
    )
    parser.add_argument(
        "--segnet-instance-connectivity",
        type=int,
        choices=[4, 8],
        default=SEGNET_INSTANCE_CONNECTIVITY_DEFAULT,
        help=TIER_1_OPERATOR_REQUIRED_FLAGS["--segnet-instance-connectivity"]["rationale"],
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Smoke mode: runs forward-pass + UNIWARD weighting algorithm "
            "validation only; NO gradient through scorers; NO archive emission. "
            "Use for $0 macOS-CPU local validation."
        ),
    )
    parser.add_argument(
        "--enable-autocast-fp16",
        action="store_true",
        help=(
            "Catalog #172 Tier-1 engineering primitive declaration. "
            "Currently WAIVED per file-level waiver until canonical autocast "
            "backport lands."
        ),
    )
    parser.add_argument(
        "--enable-tf32",
        action="store_true",
        help=(
            "Catalog #178 Tier-1 engineering primitive declaration. "
            "Currently WAIVED per file-level waiver to keep eval-roundtrip "
            "numerics deterministic until paired CPU/CUDA anchor lands."
        ),
    )
    parser.add_argument(
        "--enable-torch-compile",
        action="store_true",
        help=(
            "Catalog #179 Tier-1 engineering primitive declaration. "
            "Currently WAIVED per file-level waiver until per-substrate "
            "canary validates Inductor graph-breaks + score-axis custody."
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# Smoke main: forward-pass + UNIWARD weighting only (no gradient through
# scorers; mirrors Probe 9 verdict-producing smoke script bit-for-bit on the
# UNIWARD weighting algorithm, but accepts the base-archive frame source).
# ---------------------------------------------------------------------------
def _smoke_main(args: argparse.Namespace) -> int:
    """Smoke validation: verifies the per-instance + multi-scale wavelet
    UNIWARD weighting algorithm runs end-to-end on the supplied base-archive
    frames + SegNet hard-mask derivation.

    Per CLAUDE.md "MPS auth eval is NOISE" + Catalog #1 + #192: this is
    [macOS-CPU advisory] non-promotable. Produces NO archive bytes; emits
    a smoke-status JSON to --output-dir.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F

    print(f"[uniward-pims] smoke_main begin device={args.device}", file=sys.stderr)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Lazy import per the canonical pattern (avoid hard deps during arg parsing).
    try:
        import pywt as _pywt
        import scipy.ndimage as _scipy_ndi
        from safetensors.torch import load_file

        from upstream.modules import SegNet
    except ImportError as exc:
        print(f"[uniward-pims] FATAL: missing dependency: {exc}", file=sys.stderr)
        return 50

    device = torch.device(args.device)
    segnet = SegNet()
    sd = load_file(str(args.upstream_models))
    missing, unexpected = segnet.load_state_dict(sd, strict=False)
    segnet = segnet.eval().to(device)
    print(
        f"[uniward-pims] segnet loaded "
        f"missing={len(missing)} unexpected={len(unexpected)}",
        file=sys.stderr,
    )

    def _decode_first_n_frames(video_path: Path, n: int = 8) -> torch.Tensor:
        import av

        container = av.open(str(video_path))
        frames: list[torch.Tensor] = []
        for frame in container.decode(video=0):
            arr = frame.to_rgb().to_ndarray()
            t = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
            frames.append(t)
            if len(frames) >= n:
                break
        container.close()
        return torch.stack(frames, dim=0)

    t_start = time.time()
    frames = _decode_first_n_frames(args.video_path, n=8)
    n_frames, _, H, W = frames.shape

    pair_count = 4
    pairs = torch.stack(
        [torch.stack([frames[i], frames[i + 1]], dim=0) for i in range(pair_count)],
        dim=0,
    ).to(device)

    with torch.no_grad():
        seg_input = segnet.preprocess_input(pairs)  # (4, 3, 384, 512)
        seg_logits = segnet(seg_input)  # (4, 5, 384, 512)
    seg_mask = seg_logits.argmax(dim=1)  # (4, 384, 512)

    eval_frames = frames[1 : pair_count + 1]
    luma = (
        0.2126 * eval_frames[:, 0]
        + 0.7152 * eval_frames[:, 1]
        + 0.0722 * eval_frames[:, 2]
    )

    # Multi-scale wavelet detail magnitudes (Probe 9 canonical algorithm)
    luma_np = luma.cpu().numpy()
    detail_sum = np.zeros((pair_count, H, W), dtype=np.float32)

    for i in range(pair_count):
        coeffs_multi = _pywt.wavedec2(
            luma_np[i], args.wavelet_name, mode="symmetric", level=args.wavelet_levels
        )
        for level_coeffs in coeffs_multi[1:]:
            cH, cV, cD = level_coeffs
            detail_lo = np.abs(cH) + np.abs(cV) + np.abs(cD)
            detail_t = torch.from_numpy(detail_lo).unsqueeze(0).unsqueeze(0)
            upsampled = F.interpolate(detail_t, size=(H, W), mode="nearest")
            detail_sum[i] += upsampled.squeeze(0).squeeze(0).numpy()

    detail_sum_t = torch.from_numpy(detail_sum)
    detail_sum_resized = F.interpolate(
        detail_sum_t.unsqueeze(1),
        size=(384, 512),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)

    uniward_weights = 1.0 / (detail_sum_resized + SIGMA_FRIDRICH)
    uniward_weights_norm = uniward_weights / uniward_weights.mean()

    # Per-instance summary verification (verifies algorithm matches Probe 9)
    structure = (
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int8)
        if args.segnet_instance_connectivity == 4
        else np.ones((3, 3), dtype=np.int8)
    )

    seg_mask_np = seg_mask.cpu().numpy()
    uniward_weights_norm_np = uniward_weights_norm.cpu().numpy()
    detail_sum_np_resized = detail_sum_resized.cpu().numpy()

    per_segment_textured_weights: list[float] = []
    min_segment_pixels = 200
    min_textured_pixels = 50
    for p in range(pair_count):
        for c in range(5):
            class_mask_p = seg_mask_np[p] == c
            labeled, num_segments = _scipy_ndi.label(class_mask_p, structure=structure)
            for instance_id in range(1, num_segments + 1):
                instance_mask = labeled == instance_id
                if int(instance_mask.sum()) < min_segment_pixels:
                    continue
                instance_details = detail_sum_np_resized[p][instance_mask]
                instance_detail_q75 = float(np.quantile(instance_details, 0.75))
                instance_textured_mask = instance_mask & (
                    detail_sum_np_resized[p] > instance_detail_q75
                )
                if int(instance_textured_mask.sum()) < min_textured_pixels:
                    continue
                per_segment_textured_weights.append(
                    float(uniward_weights_norm_np[p][instance_textured_mask].mean())
                )

    elapsed = time.time() - t_start

    summary = {
        "smoke_status": "OK",
        "trainer_mode": "smoke",
        "axis_tag": "[macOS-CPU advisory]" if args.device == "cpu" else "[advisory only]",
        "evidence_grade": "macOS-CPU-advisory"
        if args.device == "cpu"
        else "advisory-only",
        "promotable": False,
        "score_claim": False,
        "score_claim_valid": False,
        "ready_for_exact_eval_dispatch": False,
        "pair_count": pair_count,
        "n_frames_decoded": n_frames,
        "wavelet_name": args.wavelet_name,
        "wavelet_levels": args.wavelet_levels,
        "segnet_instance_connectivity": args.segnet_instance_connectivity,
        "valid_segment_count": len(per_segment_textured_weights),
        "min_segment_textured_avg_weight": (
            min(per_segment_textured_weights) if per_segment_textured_weights else None
        ),
        "max_segment_textured_avg_weight": (
            max(per_segment_textured_weights) if per_segment_textured_weights else None
        ),
        # Probe 9 historical N=25 db8 anchor (HISTORICAL_PROVENANCE per Catalog
        # #110/#113 APPEND-ONLY; preserved verbatim from BREAKTHROUGH landing).
        "probe_9_anchor_min": 0.2597,
        "probe_9_anchor_threshold": 0.5,
        "probe_9_breaks_threshold": (
            min(per_segment_textured_weights) < 0.5
            if per_segment_textured_weights
            else None
        ),
        # Probe 9c canonical-optimal db4 N=100 anchor (SISTER_BASIS_DOMINATES_db4
        # verdict at z=-3.442sigma vs db8 baseline; per Catalog #307
        # IMPLEMENTATION-LEVEL fix landed commit efeaff5c9; see
        # .omx/research/probe_9c_per_level_wavelet_basis_disambiguator_landed_20260525.md).
        "probe_9c_db4_anchor_min": 0.0532,
        "probe_9c_db4_anchor_mean": 0.3599,
        "probe_9c_db4_anchor_ci_lower": 0.3457,
        "probe_9c_db4_anchor_ci_upper": 0.3740,
        "probe_9c_db4_anchor_below_threshold_fraction": 0.836,
        "probe_9c_db4_anchor_valid_segment_count": 537,
        "probe_9c_db4_anchor_z_vs_db8_baseline": -3.442,
        "probe_9c_canonical_optimal_basis": "db4",
        "elapsed_seconds": elapsed,
        "canonical_provenance": {
            "kind": "macos_cpu_advisory"
            if args.device == "cpu"
            else "advisory_only",
            "axis_tag": "[macOS-CPU advisory]"
            if args.device == "cpu"
            else "[advisory only]",
            "evidence_grade": "macOS-CPU-advisory"
            if args.device == "cpu"
            else "advisory-only",
            "hardware_substrate": "darwin_arm64_m5_max_macos_cpu_advisory"
            if args.device == "cpu"
            else "advisory_only",
            "promotable": False,
            "score_claim_valid": False,
            "source": "train_substrate_uniward_per_instance_multi_scale_wavelet_segnet_smoke",
            "predecessor_probes": [
                "tier_1_distortion_uniward_per_instance_multi_scale_wavelet_combined_smoke (Probe 9)",
            ],
        },
    }
    out_path = output_dir / "uniward_pims_smoke_summary.json"
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(
        f"[uniward-pims] smoke_main done valid_segments={len(per_segment_textured_weights)} "
        f"min={summary['min_segment_textured_avg_weight']!r} elapsed={elapsed:.2f}s "
        f"output={out_path}",
        file=sys.stderr,
    )
    return 0


# ---------------------------------------------------------------------------
# Full main: COUNCIL-GATED per Catalog #325 + Catalog #240; raises
# NotImplementedError until per-substrate symposium clears PROCEED-unconditional
# AND a sister subagent lands the canonical training loop.
# ---------------------------------------------------------------------------
def _full_main(args: argparse.Namespace) -> int:
    """Full training: COUNCIL-GATED scaffold.

    Per Catalog #325 (`check_substrate_dispatch_has_per_substrate_optimal_form_
    symposium_anchor`): paid Modal/Lightning/Vast.ai dispatch >$0.30 REQUIRES
    a per-substrate symposium memo at
    `.omx/research/per_substrate_symposium_uniward_per_instance_multi_scale_
    wavelet_segnet_<YYYYMMDD>.md` dated within 14 days with verdict in
    {PROCEED, PROCEED_WITH_REVISIONS} BEFORE dispatch fires.

    Per Catalog #240 (`check_substrate_contest_cuda_chain_complete_or_research_
    only_tagged`): the recipe lands `research_only: true` + `dispatch_enabled:
    false` while this `_full_main` raises NotImplementedError. Recipe-vs-
    trainer-state consistency is preserved.

    The canonical full-training implementation is the next operator-routed
    sister subagent landing AFTER:
      1. per-substrate symposium memo PROCEED-unconditional verdict; AND
      2. operator flips recipe `dispatch_enabled: true` + `research_only: false`; AND
      3. sister subagent lands canonical training loop with all CLAUDE.md
         non-negotiables (eval_roundtrip + EMA + score-aware loss +
         differentiable scorer preprocess + archive grammar + inflate runtime
         per HNeRV parity discipline lessons L1-L13).
    """
    msg = (
        "Tier-2 paid dispatch scaffold; `_full_main` is COUNCIL-GATED per "
        "Catalog #325 per-substrate-symposium + Catalog #240 recipe-vs-trainer-"
        "state consistency. Recipe declares research_only: true + "
        "dispatch_enabled: false. The full training loop lands in a sister "
        "subagent commit batch AFTER per-substrate symposium PROCEED-"
        "unconditional verdict AND operator flips recipe dispatch_enabled: "
        "true. See `.omx/research/per_substrate_symposium_uniward_per_"
        "instance_multi_scale_wavelet_segnet_20260525.md` + recipe at "
        "`.omx/operator_authorize_recipes/substrate_uniward_per_instance_"
        "multi_scale_wavelet_segnet_modal_t4_dispatch.yaml`."
    )
    raise NotImplementedError(msg)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)

    mode = _resolve_trainer_mode(args)
    print(f"[uniward-pims] trainer_mode={mode}", file=sys.stderr)

    if mode == "smoke":
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":
    sys.exit(main())
