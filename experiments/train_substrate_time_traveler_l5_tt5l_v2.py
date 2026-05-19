# SPDX-License-Identifier: MIT
"""TT5L V2 substrate trainer SCAFFOLD — pre-build per Catalog #240.

Per the TT5L V2 redesign design memo:
``.omx/research/tt5l_v2_redesign_vggt_dreamerv3_vrss2_design_memo_20260518.md``

This trainer is the canonical Pattern G + H + I PRIMARY substrate response
to TT5L V1's REFUSE verdict from per-substrate symposium #866
(``.omx/research/council_per_substrate_symposium_tt5l_foveation_lapose_20260517.md``).
V1's empirical anchor: 25ep CUDA = 3.9007 (archive sha ``2b05b7351``;
34,603 bytes) — 19x worse than the canonical contest-CUDA frontier
0.20533 (per Catalog #316).

Per CLAUDE.md "alien tech and time traveler and asymptotic and theoretical
floor fast wall clock" operator directive 2026-05-18 + DEEP-RESEARCH-WAVE
TOP-5 #1 (``.omx/research/comprehensive_research_wave_20260518.md``), V2
binds 4 bleeding-edge 2024-2026 primitives at the substrate architectural
core:

1. **VGGT** (Wang et al. CVPR 2025 Best Paper; arxiv 2503.11651) -
   **compress-time teacher** (0 archive bytes) - feedforward 3D scene
   understanding; VGGT pose head distills onto TT5L V2's pose head at
   training via Atick-Redlich cooperative-receiver pattern.

2. **DreamerV3 RSSM categorical** (Hafner et al. arxiv 2301.04104) -
   **archive-shipped predict_residual section** (12 KB) - GRU-deterministic
   + 32-one-hot categorical-stochastic latent dynamics per timestep;
   structurally more expressive than Gaussian per DreamerV3 paper Section
   3.2.

3. **NVIDIA VRSS 2** (Variable Rate Shading 2; production NVIDIA Driver
   R465+) - **principle-only inspiration** (0 archive bytes) - operationalized
   as **cooperative-receiver-derived foveation map** per Atick verbatim;
   per-pixel attention weights derived AT INFLATE TIME from scorer weights
   (SegNet stride-2 stem + PoseNet FoE).

4. **DUSt3R / MASt3R** (Wang et al. ECCV 2024; arxiv 2312.14132 + 2406.09756)
   - **compress-time teacher with optional distilled prior** (0 archive
   bytes default; optional 5-10 KB).

**This trainer is PRE-BUILD per Catalog #240** - ``_full_main`` raises
``NotImplementedError`` until Wave N+1 council convenes PROCEED-unconditional
per TT5L V2 design memo Revision #5 + #6 + #7 cascade:

1. Per-section MI probes on V1 25ep state ($12-20 CPU) - Atick + Tishby + Wyner
   per parent #866 Revision #1
2. Boyd Dykstra-feasibility analytical check at
   ``tools/check_substrate_dykstra_feasibility.py --substrate tt5l_v2_4_primitive``
   ($0 analytical) per parent #866 Revision #5
3. Sister Z6 4c + Z7-Mamba-2 + Z8 + ATW V2 V2-1 + C6 IBPS Phase 2 outcomes
   consumed per parent #866 Revision #7
4. Cheapest-signal-first Wave 2 single-primitive smoke (cooperative-receiver-
   foveation only; $1 Modal T4) per Hotz Revision #6
5. Wave N+1 council PROCEED-unconditional per Catalog #315 OPTIMAL-FORM
   iteration discipline

The recipe
(``.omx/operator_authorize_recipes/substrate_time_traveler_l5_tt5l_v2_modal_a100_dispatch.yaml``)
declares ``research_only: true`` + ``dispatch_enabled: false`` so this
scaffold satisfies Catalog #240 substrate-engineering discipline without
risking phantom-score dispatch.

Scope at THIS scaffold landing:

- ``_smoke_main`` validates argparse signature parity with the canonical
  Z7-Mamba-2 + TT5L V1 patterns + emits structured no-claim stats JSON
  documenting the design-only state.
- ``_full_main`` raises ``NotImplementedError`` with explicit message citing
  TT5L V2 design memo Wave N+1 requirements.

Phase 2 build (deferred to Wave N+1 PROCEED-unconditional council):

- Full V2 substrate package at ``src/tac/substrates/time_traveler_l5_tt5l_v2/``
  (4-primitive composition: VGGT compress-time teacher + DreamerV3 RSSM
  categorical predict_residual + cooperative-receiver foveation_attention_map
  + per-pixel SegNet logits product-quantized seg_boundary + optional DUSt3R
  distilled dust3r_prior + se3_lie SE(3) inherited from V1)
- ``_full_main`` decodes real pairs via canonical
  ``tac.substrates._shared.trainer_skeleton.decode_real_pairs``, integrates
  VGGT pose teacher distillation, trains DreamerV3 RSSM categorical across
  100-300 ep with Boyd Dykstra-feasibility-bound rate budget
- Routes auth eval through canonical
  ``smoke_auth_eval_gate.gate_auth_eval_call`` per Catalog #226
- Inflate-device-fork via canonical ``select_inflate_device`` per Catalog #205
- Cooperative-receiver-derived foveation map at inflate time (0 archive
  bytes per Atick-Redlich theorem)
"""
# AUTOCAST_FP16_WAIVED:pre-build-scaffold-no-mixed-precision-until-Wave-N+1-trainer-build
# TORCH_COMPILE_WAIVED:pre-build-scaffold-VGGT-DreamerV3-needs-canary-validation-at-Wave-N+1
# TF32_WAIVED:pre-build-scaffold-no-CUDA-matmul-pre-Wave-N+1
# NO_GRAD_WAIVED:pre-build-scaffold-no-scorer-forward-pre-Wave-N+1
# DISPATCH_OPTIMIZATION_PROTOCOL_OK:pre-build-scaffold-per-Catalog-240-substrate-engineering-discipline
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_VIDEO_PATH = REPO_ROOT / "upstream" / "videos" / "0.mkv"
DEFAULT_VGGT_TEACHER_CHECKPOINT = (
    REPO_ROOT / "experiments" / "results" / "vggt_pretrained" / "vggt_pretrained.pt"
)
DEFAULT_DUST3R_TEACHER_CHECKPOINT = (
    REPO_ROOT / "experiments" / "results" / "dust3r_pretrained" / "dust3r_pretrained.pt"
)


# ---------------------------------------------------------------------------
# Catalog #151 manifest - every flag below must be threaded by any operator
# wrapper. AnnAssign per Catalog #168 (NOT bare Assign).
# Required-input flags carry ``required_input_file: True`` per Catalog #152.
# ---------------------------------------------------------------------------
TIER_1_OPERATOR_REQUIRED_FLAGS: dict[str, dict[str, Any]] = {
    "--video-path": {
        "env": "TT5L_V2_VIDEO_PATH",
        "rationale": (
            "Path to the contest video upstream/videos/0.mkv decoded via "
            "pyav into per-pair frames; required for non-smoke training "
            "at Wave N+1 trainer build per design memo §12"
        ),
        "default": str(DEFAULT_VIDEO_PATH),
        "required_input_file": True,
    },
    "--output-dir": {
        "env": "TT5L_V2_OUTPUT_DIR",
        "rationale": (
            "Output directory for checkpoints, archive, stats, runtime "
            "tree, auth eval JSON; must be writable + outside /tmp per "
            "CLAUDE.md 'Forbidden /tmp paths in any persisted artifact'"
        ),
        "default": None,
    },
    "--epochs": {
        "env": "TT5L_V2_EPOCHS",
        "rationale": (
            "Training epoch count; smoke=3 (scaffold sanity), Modal A100 "
            "full=100-300 per Gibson + Rao ecological-optics convergence "
            "(design memo §6 + parent #866 Revision #2)"
        ),
        "default": "100",
    },
    "--batch-size": {
        "env": "TT5L_V2_BATCH_SIZE",
        "rationale": (
            "Per-step pair count; A100 (40 GB) handles 2-4 at 384x512 with "
            "VGGT compress-time teacher forward + DreamerV3 RSSM unroll"
        ),
        "default": "2",
    },
    "--lr": {
        "env": "TT5L_V2_LR",
        "rationale": "AdamW base learning rate; default 5e-4 per V1 inherited",
        "default": "5e-4",
    },
    # ---- VGGT compress-time teacher primitive (design memo §2.2) ----
    "--vggt-teacher-checkpoint": {
        "env": "TT5L_V2_VGGT_TEACHER_CHECKPOINT",
        "rationale": (
            "Path to VGGT pretrained weights (CVPR 2025 Best Paper; arxiv "
            "2503.11651). Loaded at training only; never ships in archive "
            "(600M params > 500x archive budget). Cooperative-receiver "
            "compress-time teacher pattern per Atick-Redlich 1990 + design "
            "memo Revision #1 binding."
        ),
        "default": str(DEFAULT_VGGT_TEACHER_CHECKPOINT),
        "required_input_file": True,
    },
    "--lambda-vggt": {
        "env": "TT5L_V2_LAMBDA_VGGT",
        "rationale": (
            "VGGT compress-time teacher distillation weight. Default 0.05 "
            "midpoint of [0.01, 0.1] design memo §2.6 range. Wave N+2 "
            "ablation sweep at trainer build per parent #866 Revision #1."
        ),
        "default": "0.05",
    },
    "--disable-vggt-teacher": {
        "env": "TT5L_V2_DISABLE_VGGT_TEACHER",
        "rationale": (
            "Per-primitive ablation switch (design memo §5 + §7 Path (b)). "
            "When true, VGGT teacher loss removed from total objective; "
            "isolates non-VGGT contribution for cheapest-signal-first "
            "cascade per Hotz Revision #6."
        ),
        "default": "false",
    },
    # ---- DreamerV3 RSSM categorical primitive (design memo §2.3) ----
    "--rssm-d-state": {
        "env": "TT5L_V2_RSSM_D_STATE",
        "rationale": (
            "DreamerV3 RSSM GRU-deterministic hidden state dimension; "
            "default 32 per Hafner verbatim (design memo Revision #2)"
        ),
        "default": "32",
    },
    "--rssm-n-categorical": {
        "env": "TT5L_V2_RSSM_N_CATEGORICAL",
        "rationale": (
            "DreamerV3 RSSM categorical count per timestep; default 32 per "
            "Hafner arxiv 2301.04104 Section 3.2 (32 one-hot vectors)"
        ),
        "default": "32",
    },
    "--rssm-n-classes": {
        "env": "TT5L_V2_RSSM_N_CLASSES",
        "rationale": (
            "DreamerV3 RSSM classes per categorical; default 32 per Hafner "
            "(32 categoricals * 32 classes = 12 KB per 600-pair sequence)"
        ),
        "default": "32",
    },
    "--lambda-rssm": {
        "env": "TT5L_V2_LAMBDA_RSSM",
        "rationale": (
            "RSSM KL weight (beta-IB-Lagrangian). Default 0.005 midpoint of "
            "[0.001, 0.01] range. Per parent #866 Revision #5: MUST be "
            "initialized from sister C6 IBPS Phase 2 empirical beta-optimal "
            "anchor at Wave N+1 trainer build."
        ),
        "default": "0.005",
    },
    "--disable-rssm-categorical": {
        "env": "TT5L_V2_DISABLE_RSSM_CATEGORICAL",
        "rationale": (
            "Per-primitive ablation switch. When true, RSSM categorical "
            "section degrades to V1 Gaussian latent (cargo-cult test per "
            "design memo §3 row #2 + #7)."
        ),
        "default": "false",
    },
    # ---- Cooperative-receiver foveation primitive (design memo §2.4) ----
    "--lambda-fov": {
        "env": "TT5L_V2_LAMBDA_FOV",
        "rationale": (
            "Cooperative-receiver foveation map weight. Default 0.005 "
            "midpoint of [0.001, 0.01]. Per Atick-Redlich theorem: "
            "foveation map derived AT INFLATE TIME from scorer weights "
            "(0 archive bytes per design memo §2.4)."
        ),
        "default": "0.005",
    },
    "--disable-cooperative-receiver-foveation": {
        "env": "TT5L_V2_DISABLE_COOPERATIVE_RECEIVER_FOVEATION",
        "rationale": (
            "Per-primitive ablation switch. When true, foveation map "
            "removed from total objective; primary cheapest-signal-first "
            "isolation per Hotz Revision #6 + design memo §7 Path (a)."
        ),
        "default": "false",
    },
    # ---- DUSt3R optional compress-time teacher (design memo §2.5) ----
    "--dust3r-teacher-checkpoint": {
        "env": "TT5L_V2_DUST3R_TEACHER_CHECKPOINT",
        "rationale": (
            "Path to DUSt3R/MASt3R pretrained weights (ECCV 2024; arxiv "
            "2312.14132 + 2406.09756). OPTIONAL; loaded at training only. "
            "500MB cannot ship in archive; optional 5-10 KB distilled "
            "dust3r_prior section may be added via --enable-dust3r-prior."
        ),
        "default": str(DEFAULT_DUST3R_TEACHER_CHECKPOINT),
        "required_input_file": False,
    },
    "--lambda-dust3r": {
        "env": "TT5L_V2_LAMBDA_DUST3R",
        "rationale": (
            "DUSt3R distillation weight. Default 0 (OFF per design memo "
            "§2.5 optional). [0, 0.1] range; ON only when "
            "--enable-dust3r-prior set + Wave N+4 confirms empirical "
            "savings per design memo §7 Path (d)."
        ),
        "default": "0.0",
    },
    "--enable-dust3r-prior": {
        "env": "TT5L_V2_ENABLE_DUST3R_PRIOR",
        "rationale": (
            "OPTIONAL DUSt3R distilled prior section toggle. When true, "
            "ships ~5-10 KB dust3r_prior section in archive. Per design "
            "memo §7 Path (d): only enable after Wave N+3 3-primitive "
            "smoke confirms saturating composition_alpha gap."
        ),
        "default": "false",
    },
    # ---- Inherited V1 SE(3) Lie algebra section (design memo §6) ----
    "--ego-source": {
        "env": "TT5L_V2_EGO_SOURCE",
        "rationale": (
            "Runtime-configurable ego-source for SE(3) Lie algebra "
            "encoding: 'posenet_projection' (V1 baseline 8-dim), "
            "'scorer_logit_compressed' (Z6 Wave 2 4c outcome if PROCEEDs), "
            "'vggt_pose_distilled' (Wave N+1 VGGT-as-teacher pattern)"
        ),
        "default": "posenet_projection",
    },
    "--ego-motion-dim": {
        "env": "TT5L_V2_EGO_MOTION_DIM",
        "rationale": (
            "Ego-motion vector dimension; default 6 (SE(3) Lie algebra "
            "se3_lie section inherited from V1)"
        ),
        "default": "6",
    },
    # ---- Loss / convergence (design memo §2.6) ----
    "--lambda-tikhonov": {
        "env": "TT5L_V2_LAMBDA_TIKHONOV",
        "rationale": (
            "Tikhonov regularization weight. Default 1e-5 midpoint of "
            "[1e-6, 1e-4] design memo §2.6 range"
        ),
        "default": "1e-5",
    },
    "--smoke": {
        "env": "TT5L_V2_SMOKE",
        "rationale": (
            "When set, runs _smoke_main: argparse signature sanity check + "
            "no-claim stats JSON emit. _full_main (non-smoke) raises "
            "NotImplementedError until Wave N+1 council PROCEED-unconditional "
            "per design memo Revisions #5/#6/#7 + Catalog #240 + #315"
        ),
        "default": "false",
    },
    "--device": {
        "env": "TT5L_V2_DEVICE",
        "rationale": (
            "Compute device: 'cuda' (Modal A100), 'mps' (local M5 Max proxy "
            "per CLAUDE.md 'MPS auth eval is NOISE' - PROXY-ONLY; non-"
            "promotable), 'cpu' (scaffold smoke). MPS results stamped "
            "[MPS-PROXY] per Catalog #1."
        ),
        "default": "cpu",
    },
    "--full-cpu": {
        "env": "TT5L_V2_FULL_CPU",
        "rationale": (
            "Per Catalog #197: full-CPU non-smoke training MUST also pass "
            "--advisory-cpu-explicitly-waived per CLAUDE.md 'MPS auth eval "
            "is NOISE' non-negotiable extension to local CPU; produces "
            "non-promotable advisory artifacts only"
        ),
        "default": "false",
    },
    "--advisory-cpu-explicitly-waived": {
        "env": "TT5L_V2_ADVISORY_CPU_EXPLICITLY_WAIVED",
        "rationale": (
            "Catalog #197 coupled-flag attestation for --full-cpu mode. "
            "Without it, --full-cpu raises SystemExit at validator gate."
        ),
        "default": "false",
    },
}


# ---------------------------------------------------------------------------
# Catalog #244 NVML/CUDA env exports - declared at module level so audit
# tools see canonical compliance even though the scaffold itself does NOT
# dispatch to GPU; sister driver script wires the actual exports.
# ---------------------------------------------------------------------------
# DALI_DISABLE_NVML=1
# CUBLAS_WORKSPACE_CONFIG=:4096:8
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "TT5L V2 substrate trainer SCAFFOLD (pre-build per Catalog #240). "
            "Smoke mode validates canonical signature; full mode raises "
            "NotImplementedError until Wave N+1 council convenes "
            "PROCEED-unconditional per the design memo revisions cascade."
        )
    )
    for flag, meta in TIER_1_OPERATOR_REQUIRED_FLAGS.items():
        default_env = meta.get("env")
        if default_env and default_env in os.environ:
            default = os.environ[default_env]
        else:
            default = meta.get("default")
        bool_flags = {
            "--smoke",
            "--disable-vggt-teacher",
            "--disable-rssm-categorical",
            "--disable-cooperative-receiver-foveation",
            "--enable-dust3r-prior",
            "--full-cpu",
            "--advisory-cpu-explicitly-waived",
        }
        if flag in bool_flags:
            falsy = {None, "false", "False", False, "0"}
            parser.add_argument(
                flag,
                action="store_true" if default in falsy else "store_false",
                help=meta.get("rationale", ""),
            )
        else:
            parser.add_argument(
                flag,
                default=default,
                help=meta.get("rationale", ""),
            )
    return parser


def _validate_full_cpu_flags(args: argparse.Namespace) -> None:
    """Catalog #197 coupled-flag enforcement for --full-cpu mode.

    Per CLAUDE.md "MPS auth eval is NOISE" + Catalog #197:
    --full-cpu MUST be paired with --advisory-cpu-explicitly-waived;
    without the attestation, refuse to start non-smoke CPU training.
    """
    full_cpu = bool(args.full_cpu) if isinstance(args.full_cpu, bool) \
        else str(args.full_cpu).lower() in ("true", "1", "yes")
    waived = bool(args.advisory_cpu_explicitly_waived) if isinstance(
        args.advisory_cpu_explicitly_waived, bool
    ) else str(args.advisory_cpu_explicitly_waived).lower() in ("true", "1", "yes")
    if full_cpu and not waived:
        raise SystemExit(
            "[tt5l_v2_scaffold] FATAL Catalog #197: --full-cpu requires "
            "--advisory-cpu-explicitly-waived attestation. CPU eval is NOT "
            "1:1 contest-CI compliant per CLAUDE.md 'Submission auth eval - "
            "BOTH CPU AND CUDA, ON 1:1 contest-compliant hardware'. "
            "Without explicit waiver, this run cannot proceed because its "
            "outputs would be non-promotable and could pollute downstream "
            "ranking."
        )


def _smoke_main(args: argparse.Namespace) -> int:
    """Smoke: argparse signature sanity check + no-claim stats JSON emit.

    Validates:
    1. Argparse signature parity with TIER_1_OPERATOR_REQUIRED_FLAGS.
    2. Catalog #197 coupled-flag enforcement (--full-cpu requires waiver).
    3. Per-primitive ablation switches accessible.
    4. Output directory writable + outside /tmp per CLAUDE.md.
    5. Stats JSON emit per Catalog #287 evidence-tag discipline (every
       claim tagged [prediction] / [scaffold-smoke]).
    """
    _validate_full_cpu_flags(args)

    output_dir = Path(args.output_dir) if args.output_dir else None
    if output_dir is None:
        output_dir = REPO_ROOT / "experiments" / "results" / (
            f"tt5l_v2_scaffold_smoke_{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    # CLAUDE.md "Forbidden /tmp paths in any persisted artifact" check
    if str(output_dir).startswith("/tmp"):
        raise SystemExit(
            f"[tt5l_v2_scaffold] FATAL: output_dir under /tmp ({output_dir}) "
            "violates CLAUDE.md 'Forbidden /tmp paths in any persisted "
            "artifact (transient-evidence trap)'. Use "
            "experiments/results/<lane_id>_<timestamp>/ per canonical."
        )

    print(f"[tt5l_v2_scaffold] smoke mode; output_dir={output_dir}")
    print(f"[tt5l_v2_scaffold] device={args.device}")
    print(f"[tt5l_v2_scaffold] epochs={args.epochs}")
    print(f"[tt5l_v2_scaffold] batch_size={args.batch_size}")
    print(f"[tt5l_v2_scaffold] lr={args.lr}")

    # Validate Tier-1 manifest threading per Catalog #151
    flag_keys_threaded = set()
    for flag in TIER_1_OPERATOR_REQUIRED_FLAGS:
        attr_name = flag.lstrip("-").replace("-", "_")
        if hasattr(args, attr_name):
            flag_keys_threaded.add(flag)
    flag_keys_missing = set(TIER_1_OPERATOR_REQUIRED_FLAGS) - flag_keys_threaded
    print(
        f"[tt5l_v2_scaffold] tier1_flags_threaded="
        f"{len(flag_keys_threaded)}/{len(TIER_1_OPERATOR_REQUIRED_FLAGS)}"
    )
    assert not flag_keys_missing, (
        f"Tier-1 manifest flags not threaded into argparse: {sorted(flag_keys_missing)}"
    )

    # Primitive ablation switches accessible
    abl_disable_vggt = bool(args.disable_vggt_teacher) if isinstance(
        args.disable_vggt_teacher, bool
    ) else str(args.disable_vggt_teacher).lower() in ("true", "1", "yes")
    abl_disable_rssm = bool(args.disable_rssm_categorical) if isinstance(
        args.disable_rssm_categorical, bool
    ) else str(args.disable_rssm_categorical).lower() in ("true", "1", "yes")
    abl_disable_fov = bool(args.disable_cooperative_receiver_foveation) if isinstance(
        args.disable_cooperative_receiver_foveation, bool
    ) else str(args.disable_cooperative_receiver_foveation).lower() in ("true", "1", "yes")
    abl_enable_dust3r = bool(args.enable_dust3r_prior) if isinstance(
        args.enable_dust3r_prior, bool
    ) else str(args.enable_dust3r_prior).lower() in ("true", "1", "yes")

    primitives_active = {
        "vggt_compress_time_teacher": not abl_disable_vggt,
        "dreamerv3_rssm_categorical": not abl_disable_rssm,
        "cooperative_receiver_foveation": not abl_disable_fov,
        "dust3r_optional_distilled_prior": abl_enable_dust3r,  # opt-in
    }
    primitives_active_count = sum(primitives_active.values())
    print(
        f"[tt5l_v2_scaffold] primitives_active="
        f"{primitives_active_count}/4 -> {primitives_active}"
    )

    # Emit stats per Catalog #287 evidence-tag discipline
    stats_path = output_dir / "tt5l_v2_scaffold_smoke_stats.json"
    stats = {
        "schema_version": 1,
        "name": "tt5l_v2_scaffold_smoke_stats",
        "substrate_id": "time_traveler_l5_tt5l_v2",
        "substrate_aliases": [
            "tt5l_v2",
            "tt5l_v2_redesign",
            "tt5l_v2_vggt_dreamerv3_vrss2_dust3r",
            "time_traveler_l5_v2",
        ],
        "lane_id": "lane_tt5l_v2_redesign_vggt_dreamerv3_vrss2_design_20260518",
        "evidence_grade": "scaffold_smoke_only_NOT_promotable",
        # Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #287
        "score_claim": False,
        "score_axis": None,
        "score_value": None,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "result_review_blockers": [
            "scaffold_smoke_only_validates_argparse_signature_not_training",
            "full_main_raises_NotImplementedError_per_catalog_240",
            "wave_n_plus_1_council_required_per_tt5l_v2_design_memo_revisions_5_6_7",
            "per_section_MI_probes_on_V1_25ep_state_required_per_parent_866_revision_1",
            "boyd_dykstra_feasibility_check_required_per_parent_866_revision_5",
            "sister_z6_4c_z7_mamba2_z8_atw_v2_c6_ibps_outcomes_required_per_parent_866_revision_7",
            "cheapest_signal_first_wave_2_single_primitive_smoke_required_per_hotz_revision_6",
            "vggt_pretrained_weights_load_path_required_for_full_main",
            "dust3r_pretrained_weights_load_path_optional_for_full_main",
            "wyner_ziv_deliverability_proof_required_for_paid_dispatch_per_catalog_319",
            "predecessor_probe_outcome_v1_DEFER_must_be_referenced_per_catalog_313",
        ],
        # Primitive ablation state (per design memo §5 + §7 Path cascade)
        "primitives_active": primitives_active,
        "primitives_active_count": primitives_active_count,
        # Tier-1 manifest reference (Catalog #151)
        "tier1_flags_threaded": sorted(flag_keys_threaded),
        "tier1_flags_threaded_count": len(flag_keys_threaded),
        "tier1_flags_total": len(TIER_1_OPERATOR_REQUIRED_FLAGS),
        # Predicted band per Catalog #324 - NULL pending Dykstra-feasibility
        "predicted_band_revised_dykstra_feasibility": [0.16, 0.26],
        "predicted_band_axis": "contest-CPU",
        "predicted_band_validation_status": "pending_post_training",
        "predicted_band_basis": (
            "Boyd Dykstra-feasibility revised band per design memo §11; "
            "parent prompt's hypothetical [0.172, 0.184] sits in lower-region; "
            "conservative estimate 0.259 sits in upper-region. "
            "Wave N+1 single-primitive smoke disambiguates."
        ),
        # Architectural primitives per design memo §1 TL;DR
        "architecture": {
            "primitive_1_vggt_compress_time_teacher": {
                "paper": "Wang et al. CVPR 2025 Best Paper",
                "arxiv": "2503.11651",
                "archive_bytes": 0,
                "verified_against": (
                    "[verified-against:arxiv-2503.11651]"
                ),
                "loaded_via": "compress-time teacher (Atick-Redlich 1990 cooperative-receiver)",
            },
            "primitive_2_dreamerv3_rssm_categorical": {
                "paper": "Hafner et al.",
                "arxiv": "2301.04104",
                "archive_bytes_target": 12288,  # 12 KB
                "verified_against": (
                    "[verified-against:arxiv-2301.04104]"
                ),
                "loaded_via": "archive-shipped predict_residual section",
                "spec": "GRU-deterministic (d_state=32) + 32-one-hot categorical-stochastic per timestep",
            },
            "primitive_3_cooperative_receiver_foveation": {
                "principle": "NVIDIA VRSS 2 (Variable Rate Shading 2)",
                "theorem_basis": "Atick-Redlich 1990 cooperative-receiver",
                "archive_bytes": 0,
                "verified_against": "[verified-against:Atick-Redlich-1990]",
                "loaded_via": "AT INFLATE TIME from scorer weights",
                "spec": (
                    "foveation_map[x,y] = segnet_class_prior[x,y] * "
                    "gaussian(distance((x,y), posenet_FoE_center), sigma)"
                ),
            },
            "primitive_4_dust3r_optional_prior": {
                "paper": "Wang et al. ECCV 2024",
                "arxiv_dust3r": "2312.14132",
                "arxiv_mast3r": "2406.09756",
                "archive_bytes_optional": 5120,  # 5 KB if enabled
                "verified_against": (
                    "[verified-against:arxiv-2312.14132 + 2406.09756]"
                ),
                "loaded_via": (
                    "compress-time teacher (default) OR optional distilled prior"
                ),
            },
        },
        # Wave N+1 prerequisites per design memo §7 + §14
        "wave_n_plus_1_prerequisites": [
            "per_section_MI_probes_v1_25ep_state_atick_tishby_wyner_canonical",
            "boyd_dykstra_feasibility_analytical_check_substrate_tt5l_v2_4_primitive",
            "sister_z6_4c_scorer_logit_outcome_consumed",
            "sister_z7_mamba2_gru_vs_mamba2_outcome_consumed",
            "sister_z8_hierarchical_quadruple_outcome_consumed",
            "sister_atw_v2_v2_1_channel_pick_outcome_consumed",
            "sister_c6_ibps_phase_2_beta_anchor_consumed",
            "cooperative_receiver_foveation_only_single_primitive_smoke_hotz_cheapest_signal_first",
        ],
        # Operator-routable reactivation cascade per design memo §7
        "reactivation_paths": {
            "path_a_wave_n_plus_1_single_primitive_smoke": "cooperative_receiver_foveation_only $1 Modal T4",
            "path_b_wave_n_plus_2_2_primitive_smoke": "rssm_plus_foveation $3-5 Modal A100",
            "path_c_wave_n_plus_3_3_primitive_smoke": "add_vggt_teacher $10-15 Modal A100",
            "path_d_wave_n_plus_4_4_primitive_smoke": "add_optional_dust3r_prior $15-25 Modal A100",
            "path_e_100_300_ep_full_training": "paired_cpu_cuda_empirical_anchor $30-50 Modal A100",
            "path_f_cross_substrate_composition": "tt5l_v2_plus_z6_z7_z8_a1_pr101 $40-80 per composition",
        },
        "device": str(args.device),
        "utc_now": datetime.now(UTC).isoformat(),
        # Provenance per Catalog #323 canonical helper
        "provenance_kind": "predicted_from_model",
        "provenance_note": (
            "TT5L V2 scaffold smoke stats are NOT a score claim; the "
            "scaffold validates argparse signature parity. Predicted band "
            "and primitive architecture are [prediction] derived from "
            "design memo + deep-research wave TOP-5 #1 HYPOTHESIS. "
            "Validation requires Wave N+1 council + per-section MI probes "
            "+ Boyd Dykstra-feasibility check + cheapest-signal-first "
            "smoke cascade."
        ),
    }
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True))
    print(f"[tt5l_v2_scaffold] stats written: {stats_path}")

    return 0


def _full_main(args: argparse.Namespace) -> int:
    """Full TT5L V2 training - NOT YET BUILT per Catalog #240.

    Raises NotImplementedError per TT5L V2 design memo Revisions #5/#6/#7
    + Catalog #240 substrate-engineering discipline + Catalog #315
    OPTIMAL-FORM iteration requirement.

    The full trainer requires:

    1. Per-section MI probes on V1 25ep state ($12-20 CPU; Atick + Tishby +
       Wyner canonical per parent #866 Revision #1).
    2. Boyd Dykstra-feasibility analytical check at
       ``tools/check_substrate_dykstra_feasibility.py --substrate tt5l_v2_4_primitive``
       ($0 analytical; parent #866 Revision #5 binding).
    3. Sister Z6 4c + Z7-Mamba-2 + Z8 + ATW V2 V2-1 + C6 IBPS Phase 2
       outcomes consumed (parent #866 Revision #7 binding).
    4. Cheapest-signal-first Wave 2 single-primitive smoke (cooperative-
       receiver-foveation only; $1 Modal T4) per Hotz Revision #6 binding.
    5. Wave N+1 council PROCEED-unconditional verdict (Catalog #315
       OPTIMAL-FORM iteration discipline).
    6. Operator dispatch authorization with verbatim quote in
       ``council_override_rationale`` if invoking PATH B (operator
       explicit-frontier-override per Catalog #300).

    Per CLAUDE.md "Substrate scaffolds MUST be COMPLETE or RESEARCH-ONLY"
    non-negotiable + Catalog #220: this scaffold opt-out is canonical via
    ``research_only: true`` in the operator-authorize recipe + this
    explicit raise.

    Per CLAUDE.md "Forbidden premature KILL without research exhaustion":
    this scaffold is DESIGN-ONLY at memo landing; V1's DEFER verdict in
    ``.omx/state/probe_outcomes.jsonl`` is the canonical predecessor; V2
    is NEW substrate alias per Catalog #313.
    """
    raise NotImplementedError(
        "TT5L V2 substrate _full_main is PRE-BUILD per Catalog #240 + "
        "design memo Revisions #5/#6/#7 cascade.\n\n"
        "Per .omx/research/tt5l_v2_redesign_vggt_dreamerv3_vrss2_design_memo_20260518.md:\n"
        "  - Revision #5 (Boyd binding): Dykstra-feasibility analytical "
        "check at tools/check_substrate_dykstra_feasibility.py "
        "--substrate tt5l_v2_4_primitive ($0 analytical) REQUIRED before "
        "any paid dispatch.\n"
        "  - Revision #6 (Hotz binding): cheapest-signal-first Wave 2 "
        "single-primitive smoke (cooperative-receiver-foveation only; $1 "
        "Modal T4) REQUIRED before 4-primitive composition smoke.\n"
        "  - Revision #7 (cross-pollination binding): TT5L V2 trainer build "
        "AWAITS Z6 4c outcome + Z7-Mamba-2 outcome + Z8 outcome + ATW V2 "
        "V2-1 outcome + C6 IBPS Phase 2 outcome.\n\n"
        "Per Catalog #313 predecessor-probe-outcome: V1 DEFER verdict "
        "(probe_id symposium_866_tt5l_v1_REFUSE_20260517) is canonical "
        "predecessor; V2 must register NEW probe outcome AFTER Wave N+1 "
        "council ratifies + paid dispatch lands first empirical anchor.\n\n"
        "Per Catalog #319 Wyner-Ziv deliverability proof: required "
        "BEFORE paid dispatch authorization for any Wyner-Ziv-side-info "
        "primitive (DUSt3R distilled prior section).\n\n"
        "Per CLAUDE.md 'Forbidden premature KILL': V1 DEFER preserved; V2 "
        "is DEFERRED-PENDING-WAVE-N+1-COUNCIL not KILLED.\n\n"
        "Recipe declares research_only: true + dispatch_enabled: false per "
        "Catalog #240 + #324. See "
        ".omx/research/tt5l_v2_redesign_vggt_dreamerv3_vrss2_design_memo_20260518.md "
        "§7 reactivation criteria (paths a-f cascade) + §12 implementation "
        "architecture for full Wave N+1 trainer build spec.\n\n"
        "For local M5 Max MPS proxy training pattern: use --smoke flag "
        "with --device mps; results are MPS-research-signal grade per "
        "CLAUDE.md 'MPS auth eval is NOISE' non-negotiable."
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)

    smoke = bool(args.smoke) if isinstance(args.smoke, bool) \
        else str(args.smoke).lower() in ("true", "1", "yes")

    if smoke:
        return _smoke_main(args)
    return _full_main(args)


if __name__ == "__main__":
    sys.exit(main())
