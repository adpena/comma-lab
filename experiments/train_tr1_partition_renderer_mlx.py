#!/usr/bin/env python
"""ddm_tb1 — SPEC_tr1 trained partition→pixel renderer (token grid + conv renderer).

THE renderer build (fork-adjudicated GO, fd2 verdict SEG_REALIZATION_GAP_AT_UINT8_DOMINANT):
a per-frame TOKEN GRID latent field + a small trained partition→pixel conv renderer,
trained SCORER-IN-LOOP through the full contest-exact R operator (bicubic↑384→874 →
uint8-STE → bilinear↓→512×384) against the frozen SegNet — descent THROUGH the
quantization (the fd2 lesson), never propose-then-quantize.

TWO RACED VARIANTS under matched counted-byte accounting (charter A2 / eu1 R1):
  * ``plain``  — ordinary trained conv weights (counted, int-quantized at export).
  * ``lotto``  — G1-LOTTO supermask: fixed-seed PRNG-generated conv bank (FREE generic
    expansion per rule-118) + learned binary supermask + per-channel modulations
    (COUNTED), plus a COUNTED selector/config ledger for every video-selected choice
    (seed, grid geometry, width, density — the eu1 rule-118 selector-accounting flag,
    adjudicated in the tb1 design memo).

A1 (fd2 BINDING TRANSFER LESSON) is wired from day one: the training loop validates
REALIZED argmax flips (render → fp32 → bicubic↑camera → uint8 → frozen CPU-torch
SegNet argmax) on a pre-registered gate set as periodic in-training telemetry AND a
stage-exit gate. Smooth-loss descent without realized-flip improvement = the inherited
gap → typed ``A1_REALIZATION_GAP_ALARM`` (never silent) + stage-exit REFUSE.

Pose: TERMINAL (#383) — default ``pose_objective_weight=0`` on the seg trunk.
JD1 adds an args-only, default-off joint pose-finish gate that enters only after
the seg/constrain boundary and then uses the same scorer-native PoseNet path as
``make_loss_fn``. frame_1-only seg rendering remains; frame_0 is structurally
seg-free and is rendered only when the JD1 pose gate is armed.

Evidence axis: ``[macOS-CPU/MLX advisory]`` — score_claim=False, promotion_eligible=
False. Realized d_seg rows here are ADVISORY (frozen CPU-torch scorer on macOS); the
pointer 0.1910828242 [contest-CPU] moves ONLY through a byte-closed evaluate.py row.

COMPOSED (not duplicated) from the canonical witness substrate:
  * ``make_loss_fn`` (+ its measured seg-loss forms ce/tau_softplus/l7/unify_tau,
    margin weighting) via its ``render_fn`` hook with ``compute_pose=False``;
  * ``_apply_R`` (contest-exact MLX R, uint8-STE @ camera);
  * ``_torch_R_to_camera_uint8`` + ``cpu_verdict_d_seg_argmax_batch`` (the CPU-torch
    realized authority path, bit-exact batched);
  * ``open_stored_npy_memmap`` GT cache access (lstars/margins; no camera frames).

Resumability P0: crash-resumable (``--resume-from``), per-stage EMA-shadow
checkpoints under distinct stage-encoded filenames, atomic tmp+rename, periodic
intra-stage saves, ``--max-wall-minutes`` bounded windows that checkpoint on exit.

THREE MEASURED DESIGN FORCES (operator recall directive 2026-07-28, wired at T0):
  * NONLINEAR — the argmax target is a STAIRCASE (fd2: faithful-flip window empty)
    => ``margin_hinge`` joins the raced seg-form set (step-native surrogate lineage);
    A1's realized-flip gate is the nonlinearity's ground truth over any smooth
    surrogate. Pose stays TERMINAL (the sqrt(10*d_pose) marginal-flip treatment).
  * ASYMMETRY — sn1 MEASURED sided SegNet decision distance
    (codex_findings_ddm_sn1_segnet_telemetry_asymmetry_20260723_codex.md + SSD
    ddm_sn1_segnet_telemetry_asymmetry_n600_20260723/): per-class structure is
    extreme (Lane 36% of partition cost at 0.59% of pixels) => ``--class-weight-lane``
    per-GT-class asymmetric loss weight (seg_pixel_w hook); the FULL sided 5x5
    directional-confusion loss is a registered DUTY_TO_MEASURE lever (below), not
    half-wired. uint8 rounding is directionally asymmetric through R => the token
    STE is RACED (``--token-ste round|dither``), never assumed.
  * SYMMETRY — Einstein decomposition d_seg = d_cov + d_gauge: pair-dependence must
    factor through (xi, R) or rate is WASTED => tokens default to SHARED-BASE +
    per-frame DELTA (identity-xi zeroth-order advection; full tac.lie xi-advection
    is a registered DUTY_TO_MEASURE lever), and ALL token fields are ZERO-INIT so
    counted token capacity is loss-driven only (gradients vanish along ker(A) =>
    no counted bytes spent on the gauge orbit by construction).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import sys
import time
import zipfile
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve()
WORKTREE = _HERE.parents[1]
# Shared-venv hijack guard (MEMORY ☠️🐍): the editable install resolves ``tac`` to the
# MAIN checkout. Insert THIS tree first so tb1 modules (src/tac/witness_dsl/spec_tr1_*)
# and experiments.* resolve to the worktree that carries them.
for _p in (str(WORKTREE), str(WORKTREE / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tac.optimization.ddm_gd1_gate_estimator import GateDesign, horvitz_thompson_mean
from tac.witness_dsl.scope_laws import resolve_scope_law, scope_law_geometry_hash

SEG_H, SEG_W = 384, 512
DEFAULT_GT_CACHE = "/Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
POINTER_LINE = "0.1910828242 [contest-CPU] UNMOVED"
PG1_SEG_GRAD_Q3_MODES = ("off", "on")

# ddm_pg1: canonical frame_1 yuv6 pose-null projector. This is the same 6x12
# block constraint matrix as experiments/ddm_sq1_pose_null_constrained_paint.py.
_YUV6_LUMA_WEIGHTS = (0.299, 0.587, 0.114)

# Pre-registered A1 gate geometry (fd2 instrument geometry: block 447-450 + 32 rng(0)
# off-block samples). At --num-pairs below 600 the gate set is ALL training pairs.
GATE_BLOCK_PAIRS = (447, 448, 449, 450)
GATE_OFFBLOCK_SAMPLE = 32

# BI1 (#924) birth seed/amplify path.  These are intentionally argparse-only
# runtime knobs so the default-OFF arm preserves TR1Config, config_hash,
# telemetry schema, and checkpoint bytes.  ON-only telemetry carries provenance.
TR1_BIRTH_SEED_CLASS_IDS: dict[str, int] = {"lane": 1, "movable": 3}
TR1_BIRTH_SEED_DEFAULT_CLASSES = "lane"

# TK1 (2026-08-05): PE3 and cheapdct4 consumers are args-only runtime knobs.
# OFF must preserve TR1Config/config_hash/checkpoint bytes.  ON-only telemetry
# records the active cache custody and score_claim=False.  The PE3 SHA is the
# receiver-closed LC1/PK1 section this arm is contracted to consume.
PE3_CONDITIONING_EXPECTED_SECTION_SHA256 = (
    "5cc024ad32df7fedb18afb75dbed6be9c1af948dac826a1736cb1084949855c2"
)
PE3_CONDITIONING_MODE_ORDER = ("generator_pair_bisector", "depth_conditioned_curve")
CHEAPDCT4_STAGE2_SECTION_NAMES = (
    "od8_stage2_cheapdct4_qcoeffs",
    "od8_stage2_cheapdct4_synthetic",
)
CONTEST_DENOMINATOR_BYTES = 37_545_489
JD1_POSE_FINISH_SCHEMA = "ddm_jd1_tr1_joint_pose_finish_runtime.v1"
JD1_POSE_FINISH_MODES = ("off", "joint_loss")
JD1_POSE_FINISH_ENGAGE_ON = ("post_knee", "start_epoch")
JD1_SEG_HOLD_FLOOR_SOURCES = (
    "off",
    "last_pre_pose_epoch_loss",
    "checkpoint_tail_ep_loss",
    "explicit",
)
JD1_SEG_HOLD_SPACES = ("loss", "realized")
JD1_EMA_STAGE_SCOPES = ("off", "window")
JD1_EMA_MODES = ("geometric", "plateau_tail_average")
JD1_EMA_TAIL_ANCHOR_OFF = -1
JD1_EMA_TAIL_STATE_KEYS = (
    "ema_mode",
    "ema_tail_anchor_epoch",
    "ema_tail_configured_anchor_epoch",
    "ema_tail_average_active",
    "ema_tail_update_count",
    "ema_tail_anchor_global_step",
    "ema_tail_anchor_reason",
    "ema_tail_last_live_weight",
)
JD1_LIVE_GATE_TELEMETRY = ("off", "on")
JD1_LR_ANNEAL_SCHEMA = "ddm_la1_jd1_lr_anneal.v1"
JD1_LR_ANNEAL_MODES = ("off", "derived_tail")
JD1_LR_FINAL_FRAC_DERIVED = 0.0
JD1_LR_DERIVATION_TIME_CONSTANTS = 2.0
JD1_FINISHER_SCHEMA = "ddm_wp1_jd1_muon_finisher.v1"
JD1_FINISHER_MODES = ("off", "muon")
JD1_MUON_FINISHER_NS_STEPS = 5
TR1_MUON_RENDERER_WEIGHT_PREFIXES = (
    "w_conv",
    "w_up",
    "w_head",
    "s_conv",
    "s_up",
    "s_head",
)

# Pre-registered A1 alarm thresholds (tb1 charter T1: "never scale a loop whose
# realized-flip telemetry is flat"). Smooth descended but realized did not:
A1_SMOOTH_DROP_REL = 0.02      # smooth loss fell >= 2% since previous gate ...
A1_REALIZED_DROP_REL = 0.005   # ... while realized gate d_seg fell < 0.5%  -> ALARM
A1_CONSECUTIVE_REFUSE = 2      # this many consecutive alarms => stage-exit REFUSE

# ---- ddm_bp1 (#824) boundary reset race ------------------------------------------------
# MLX ``optim.Adam`` betas default (VERIFIED from the installed signature, not recalled:
# ``Adam(learning_rate, betas=[0.9, 0.999], eps=1e-08, bias_correction=False)``). This
# trainer never overrides betas, so beta2 is always 0.999 — which is exactly the value at
# which ``_adam_bias_correction_for`` passes ``reference_semantics`` through verbatim.
RESET_ADAM_BETAS: tuple[float, float] = (0.9, 0.999)
# ``sum_t (eta(t) - 1)`` converges with time constant 1/(1-beta2) = 1000 steps; 20k steps is
# 20 time constants => the reported impulse is the converged value, not a window artifact.
BOUNDARY_IMPULSE_CONVERGENCE_STEPS = 20_000

# "Off is a tracked queue, never a forgotten default": named levers DESIGNED here but
# NOT half-wired — each carries its receipt and its activation state (never-fired).
DUTY_TO_MEASURE: tuple[dict[str, str], ...] = (
    {"lever": "sided_confusion_matrix_loss", "state": "never-fired",
     "receipt": ".omx/research/codex_findings_ddm_sn1_segnet_telemetry_asymmetry_20260723_codex.md",
     "note": "directional (Road->Lane != Lane->Road) loss weights per the sn1 strict "
             "sided-tolerance contract; needs runner-up-class plumbing in the loss form"},
    {"lever": "xi_advected_token_sharing", "state": "never-fired",
     "receipt": "MEMORY einstein d_seg=d_cov+d_gauge cluster; engine tac.lie",
     "note": "SE(3)/Chasles advection of the shared token base per pair; shared_base "
             "is the identity-xi zeroth-order approximation shipped at T0"},
    {"lever": "renderer_bit_depth_race_int4_int5", "state": "never-fired",
     "receipt": "SPEC_tr1 G3 (bit-depth-DOF law; int8 export is the T0 estimate)",
     "note": "QAT int4/int5 export race; T0 ledger prices int8 only"},
    {"lever": "trunk_forces_360_set", "state": "never-fired",
     "receipt": "#360 (temporal screw-consistency / MarginBandSatisficing #459 / "
                "tie-locus displacement / R-phase alignment)",
     "note": "witness-vehicle-derived in-trunk forces; EXCLUDED from the T1 base-loop "
             "race to avoid confounding the A2 arm comparison; MarginBandSatisficing is "
             "the first T2+ candidate (min-S-over-solution-SET law: stop over-deepening "
             "margins past the flip boundary)"},
    {"lever": "perclass_pair_surface_tension_sigma_ccprime", "state": "never-fired",
     "receipt": "#382 Gamma-limit per-class-pair sigma law",
     "note": "NO scalar length/MCF term exists in the tr1 loss => the Lane-erasure "
             "mechanism is absent BY CONSTRUCTION; sigma_cc' becomes binding only if a "
             "curvature/length regularizer is added"},
    {"lever": "update_rms_matched_optimizer_race", "state": "logged-not-enforced",
     "receipt": "#685 px1 (fair optimizer A/Bs need update-RMS matching)",
     "note": "Adam per-param normalization approximately equalizes update RMS across "
             "arms; the per-gate param_delta_rms telemetry MEASURES it instead of "
             "assuming it — enforcement lever queued if arms diverge >2x"},
    {"lever": "lane_pool_topology_loss_race", "state": "never-fired",
     "receipt": "steer #3: clDice held by curriculum_dsl/typed_config/gauge (+#260 "
                "Metal kernel, ANCESTOR-vehicle +26% per L18 — RE-RACE never adopt); "
                "persistence/Betti in curriculum_dsl + spec_c2; sn1 sided weights; "
                "#382 sigma_cc'; op1 P3 row 4 OASIS per-pixel class balancing "
                "(class_weight_lane IS this lever family's simplest member)",
     "note": "POOLS LAW: these ALL draw the SAME Lane-error pool — COMPETE never "
             "sum; race per-lever at own-optimum then KKT-waterfill winners; blind "
             "stacking is the named non-additivity bug"},
    {"lever": "renderer_conditioning_clade_geo_icpe_vs_spade_mini", "state": "never-fired",
     "receipt": "op1 P3 row 1 (CLADE TPAMI 2021 arXiv 2012.04644: class-adaptive "
                "modulation at ~39% less param overhead than a SPADE subnet; ICPE "
                "positional maps computed at decode = rule-118 FREE — the openpilot "
                "features v-174, d(v)=488.3/(v-192), dist-to-boundary; geometry "
                "custody op1 P2: fx=400.27/fy=399.82/c=(256,192), two-horizon-roles "
                "174 lane-IPM / 192 pose-geometric)",
     "note": "matched A/B at (D=16,c=4) EQUAL counted bytes: CLADE+geo-ICPE vs "
             "mini-SPADE; adopt lower native d_seg arm; transfer falsified if "
             "mini-SPADE wins by >10%; DSL lever name renderer_conditioning"},
    {"lever": "row_anisotropic_D_foveation", "state": "GATE-PASSED-QUEUED",
     "receipt": "op1 P3 row 2 + tb1 MEASURED $0 gate "
                "(/Volumes/VertigoDataTier/pact/ddm_tb1_20260728/"
                "op1_row_foveation_gate.json): 72.1% of flip-prone mass (GT margin<thr,"
                " stable across thr 0.05/0.1/0.25, n600 margins memmap) lies in rows "
                "160-240 (21% of rows) >= the pre-registered 50% adoption criterion; "
                "best 81-row band 166-246 @ 72.7%",
     "note": "ENTERS the S1.2 grid race as ONE variant lane (D=8 rows 160-240, D=16 "
             "elsewhere) — raced, never unconditionally adopted; IMAGE-PLANE ONLY "
             "(NO BEV lane: #609-v2 exact-chart KILL, Road 39.02/Lane 47.12 px p50 "
             "ruling residual; re-entry only via the memo's F1^F2 falsifier)"},
    {"lever": "boundary_gated_token_code_width", "state": "never-fired",
     "receipt": "op1 P3 row 3 (PointRend logic at the coder; boundary 2,436 px/frame "
                "= 1.2% of pixels; partition piecewise-constant)",
     "note": "$0 gate owed: H(cell|neighbors) interior vs boundary cells on GT tokens "
             "at (D,c); adopt iff >=15% token-stream saving vs uniform c; feeds G4"},
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TR1Config:
    variant: str  # "plain" | "lotto"
    num_pairs: int
    grid_downsample: int          # D in {8, 16, 32}: 384/D x 512/D token lattice
    code_width: int               # c in {2, 4, 6}
    renderer_width: int           # w
    token_quant_levels: int       # description-level lattice (eval_roundtrip at tokens)
    seed: int
    lotto_seed: int
    lotto_mask_density_init: float
    seg_form_start: str           # "ce" (event-switches to tau_softplus at the knee)
    w_seg: float
    lr: float
    batch_pairs: int
    epochs: int
    gate_every: int
    ema_decay: float              # resolved (LawRef-derived or explicit)
    ema_decay_provenance: str
    token_temporal_mode: str      # "shared_base" (identity-xi) | "independent" (A/B arm)
    token_ste: str                # "round" | "dither" (RACED — uint8 rounding is asymmetric)
    class_weight_lane: float      # 1.0 = off; sn1 asymmetry lever (per-GT-class weight)
    margin_target: float          # margin_hinge form target (raced lever)
    token_init_mode: str = "zero"  # "zero" (tb1 gauge-hygiene control) | "solve_project"
    # solve_project = v3 ANALYTIC chart projection (area-mean GT downsample -> lattice;
    # base = temporal mean, delta = residual). No pretrain knob: v1 joint-L2 and
    # v2 tokens-only-gradient formulations are MEASURED inadmissible (see main()).
    basin_handoff: str = "off"  # "on" = operator 2026-07-28 train-ONLY-to-condition rule:
    # on basin-entry (TerminalSolve §16.1 validity: quadratic crawl + topology stable +
    # no transitions remaining) STOP training permanently and hand off to the SOLVE
    # executors (#423 GN/CG + eg1 QDBS rail + #383 terminal pose; v19 realized acceptance).
    # ---- QA24 5-piece composed re-burn (sg1 §3; each with a pre-registered falsifier) ----
    token_cell_mask: str | None = None    # §3.1 coarse-from-birth: path to (grid_h,grid_w)
    # bool npy (True = KEEP). Inactive cells are MULTIPLICATIVELY zeroed in the token field
    # (gradients vanish there => never learned => excluded from the coded token stream). None
    # = uniform dense grid (tb1 control). Derived: gr1 cell_drop50 keep-384 (99.61% flip mass).
    margin_weighted_loss: str = "off"     # §3.2 boundary-annulus form fix: "on" builds the
    # canonical make_loss_fn with margin_weighted=True (100% of flips are in the bottom GT-margin
    # decile — sg1 §1.3; the uniform loss spends its budget on the never-flipping deep interior).
    margin_weight_temp: float = 1.0       # inverse-margin reweight temperature (make_loss_fn).
    # ---- ddm_tp2 row 3: #274 spike/coherent per-pixel reweight (PORTED producer) ----
    # Default OFF and both scalars 1.0 => no map is built => seg_pixel_w unchanged =>
    # BYTE-IDENTICAL. See build_spike_coherent_codes + the ti1 admission evidence above.
    seg_spike_reweight: bool = False
    seg_spike_downweight: float = 1.0     # weight on SPIKE px (conceded; race start 0.25)
    seg_coherent_upweight: float = 1.0    # weight on COHERENT px (race start = MEASURED lift)
    w_rate: float = 0.0                   # §3.4 rate-in-loss (stl1 row-8 LAW): weight on the
    # differentiable token-entropy surrogate added to the seg loss (0.0 = distortion-only = tb1
    # control). The explicit form of the §3.3(b) redistribution co-benefit.
    rate_model: str = "entropy"           # "entropy" = marginal soft-histogram of the quantized
    # token lattice; "smevr_surrogate" = temporal-DELTA soft-histogram (closer to the actual
    # zlib-on-delta coder token_stream_bytes runs). Both real + differentiable.
    token_quant_anneal: str = "off"       # §3.3(a) lattice annealing / staged quantizer: "off"
    # = STE engaged from birth (tb1 control); "at_knee" = float tokens (no STE) until the CE->tau
    # knee EVENT, then engage the STE (find the basin in float, refine on the shipped lattice).
    composed_s_gate_subset: int = 0       # §3.5 QA77-lite: >0 = at stage exits run the bounded
    # terminal pose+photometric solve on this many pairs and record COMPOSED S (100*d_seg +
    # sqrt(10*d_pose) + rate) so stage/endpoint decisions see the sky/hood-freeze pose cost
    # (co9 Knee-A externality); 0 = off. VERDICT-level only (never differentiated through).
    composed_s_subset_ids: str | None = None  # §3.5 subset SELECTION (MAIN QA66 signal): path
    # to an .npy of pair indices to use as the composed-S subset. Pose is content/tail-limited
    # (QA66: top-17 pairs = 74.3% of pose mass) => run the bounded solve on the POSE-MASS TAIL
    # for max signal/sec. None => the first composed_s_gate_subset pairs (head, uniform).
    token_rowband_spec: str | None = None  # QA84 (census §4.2) VARIABLE-CELL GRAMMAR: path to a
    # RowBandGrammar spec .json (or inline json) => the token grid is D8 (fine) with BULK rows
    # TIED in 2x2 blocks (D16-effective) and the op1 flip-band rows FREE at D8 (foveation). The
    # tie is a differentiable gather; byte-close prices the tied field via SMEVR (bulk blocks
    # code ~free) + the band spec. Requires --grid-downsample 8. None => uniform grid (control).
    renderer_head_mode: str = "rgb"       # QA83 (census §4.1) OUTPUT-SPACE FACTORIZATION: the
    # renderer head output space. "rgb" = 3-channel RGB via sigmoid*255 (control = current burn);
    # "class_field" = k=1 class-field scalar c(x) -> a FIXED monotone gray lift L:R->R^3 (the
    # 1-luma-channel ur-instance; comma10k class_values live on one luma axis); "class_field_photo"
    # = k=2 (class c + margin-slack-confined luma photometric channel p). The lift is generic
    # decoder code (rule-118 FREE); only the k-channel token field is counted. Reduces the head
    # conv output DOF (renderer_bytes) and, at matched TOTAL bytes vs "rgb", tests the ~2x
    # effective-capacity-per-byte claim. Default "rgb" => exact current behavior (resume-safe).
    head_photo_slack_gain: float = 0.05   # QA83/QA80 margin-slack budget for the class_field_photo
    # luma photometric channel: a CONSERVATIVE fixed gain (~13/255 luma) so the pose-legible
    # perturbation stays inside the boundary-annulus flip-distance budget (band lemma => ~zero seg
    # flips). The EXACT per-pixel budget d=|m|/||dw|| (QA80 flip-distance field) is the named
    # compress-time scorer refinement; this fixed gain is the scorer-free build default.
    byte_ledger_coder: str = "smevr"      # QA86(b) / census T5: coder used to PRICE the token
    # stream for stage/telemetry decisions. "smevr" = the SHIPPED r7 coder (decisions match the
    # archive); "zlib" = the legacy temporal-delta surrogate (decision-noise vs shipped bytes,
    # kept for a byte-continuous live-burn resume). NEVER changes shipped/trained bytes.
    composed_s_delta_ref: str | None = None   # §3.5 ADOPTED form (MAIN Option A 2026-07-30):
    # path to the GT-ideal delta reference .npz (ddm_bc1_delta_baseline.py: baseline_dpose +
    # knee_sensitivity + tail_ids). When set, the composed-S runs the DEGRADED DIRECTIONAL-DELTA
    # instrument (d_pose(GT_f0, burn_f1) - baseline, DIRECTIONAL ONLY — sign+trend of the co9
    # Knee-A pose-recoverability externality; NEVER an absolute pose_contrib/endpoint S) on the
    # ref table's knee-A tail. The absolute bounded-solve pose_contrib is INSTANCE-DEAD on this
    # vehicle pre-joint-re-solve (measured, 4 solvers). None => the absolute composed_s (kept as
    # reference-only; not for endpoint acceptance).
    distill_field_cache: str | None = None    # QA75 (ddm_dw1): path to the concatenated b2b
    # teacher distill-logit cache (P,5,384,512) f16 (tools/ddm_dw1_build_distill_field_cache.py).
    # None => distill OFF => BYTE-IDENTICAL to the plain continuation control (window B).
    distill_weight: float = 0.0               # w_distill (ADDITIVE to the seg loss). 0.0 => OFF.
    # DERIVED rung = w_seg (100.0, the S-exact d_seg weight): the distill term is a d_seg surrogate
    # on the FEASIBLE teacher, so it shares the seg weight (own-optimum: raced, not asserted).
    distill_temp: float = 2.0                 # KD softmax temperature T (kd_logits form). Rung:
    # project-canonical KD temperature (Quantizr/PR95 kl_on_logits T=2.0; Hinton 2015 lineage).
    distill_form: str = "kd_logits"           # {kd_logits | margin_field | argmax_ce}; the winner
    # of the ddm_dw1 loss-form mini-race (own-optimum law; never a borrowed default).
    distill_attack_temp: float = 0.0          # >0 => emphasise the low-GT-margin boundary annulus
    # (QA74 attack set; exp(-GT_margin/temp) normalised); 0 = uniform. RACED dimension (not optional).
    head_range_relax: str = "off"             # {off | linear} (ddm_dw1 Window C, MAIN charter):
    # "linear" adds a trainable per-channel output residual gain (init 0 => warm-start-EQUIVALENT to
    # rgb) to sigmoid*255, de-saturating the head so gradients reach out-of-chart (dark) pixels — the
    # off-RGB output-chart probe (pj1 range-wall 67.95). ADVISORY-NON-DEPLOYABLE (a head change breaks
    # the E1 receiver arch tr1_lotto_combined_ema_v1); its slope is the decision signal for a receiver
    # rev, NOT a deployable row. "off" => no new param => resume/checkpoint byte-compatible.
    # ---- ax1 Pool-A token-byte levers (ddm_pa1b #793; DEFAULT-OFF => byte-identical control) ----
    token_quant_margin_coupling: str = "off"  # ax1 §2a {off|on}: per-cell EFFECTIVE quant levels
    # allocated by the MEASURED QA80 flip-distance field (allocation LAW = rank transform of the
    # field's own flip-mass; no bare constant). "on" builds a FIXED (non-trainable) per-cell level
    # map => not checkpointed, not in EMA => byte-identical resume. "off" => scalar-L control.
    token_quant_coupling_field: str | None = None  # QA80 field custody dir (ddm_zb1_qa80_field);
    # required when margin-coupling "on" (fail-closed if the SSD tier is not mounted).
    token_quant_coupling_min_levels: int = 0  # coarse-floor endpoint of the allocation ladder;
    # 0 => derive (base_levels // 4, a lattice-friendly coarse floor). base endpoint = quant_levels.
    token_delta_group_sparsity: str = "off"   # ax1 §4a {off|on}: group-L2 shrinkage on per-pair
    # token deltas (98.8% image-stationarity has NO train-side force). Loss term only => no param.
    delta_sparsity_weight: float = 0.0        # w_delta_sparsity (ADDITIVE); 0.0 => OFF. The
    # TRAIN-side twin of the export-side ν null-snap (gc10 F2): both drive stationary deltas to 0.
    delta_sparsity_engage: str = "after_base_stability"  # {after_base_stability|from_step_0}:
    # §7 ordering derives "engage AFTER base stability" (shrinking deltas against a moving base is
    # noise); the base-stability EVENT = the CE->tau knee (loss plateau). "from_step_0" = the gc10
    # F2 ν-snap warm-start holder (a burn warm-started from the snapped export keeps bytes low).
    delta_sparsity_weight_field: str = "uniform"  # {uniform|xi_informed}: xi_informed RELAXES the
    # shrinkage where the ego-motion prior says deltas legitimately move (lane/movable, DERIVED
    # from the QA80 winner-class field), TIGHTENS on the static mass (ax1 §5).
    # ---- ddm_lg1 (#808) CONSTRAIN-AND-PROTECT layer (default-OFF => byte-identical) ----
    lane_guard: bool = False              # master switch; False => tac.optimization.lane_guard
    # is NEVER invoked (no state, no RNG, seg_pixel_w path unchanged => bit-identical to a
    # pre-lg1 run). ON: hold realized Lane error <= the ep641 endpoint budget via a
    # primal-dual multiplier + protect born-lane support + emphasise low-margin Lane pixels,
    # all as ADDITIVE per-pixel weights in the EXISTING seg_pixel_w hook (b4s consumes via the
    # engagement spec). Budget = xp1 base_lane_S_units 0.12589; eta/step/floor DERIVED.
    lane_guard_budget_s: float = 0.0      # 0.0 => LANE_BUDGET_S_UNITS (0.12589, xp1 ep641)
    lane_guard_eta: float = 0.0           # 0.0 => derive_eta_lambda() (~66.2)
    lane_guard_lambda_step_cap: float = 0.0   # 0.0 => derive_lambda_step_cap() (0.1)
    lane_guard_lambda_max: float = 5.0    # bounded safety ceiling (5x the natural weight unit)
    lane_guard_born_weight: float = 0.0   # 0.0 => born-lane protection OFF
    lane_guard_margin_floor_weight: float = 0.0   # 0.0 => margin-floor emphasis OFF
    lane_guard_lambda_init: float = 0.0   # warm-start dual (b4s rollback+raise-lambda path:
    # state resets at relaunch; the supervisor re-fires with the last lambda + one step)
    # ---- ddm_bs2 (#871) BUDGET SCHEDULE: the constant budget can never bind -------------
    # MEASURED on the burn-4 primary telemetry (64 lane_guard gate rows, windows 01-03):
    # lambda_lane == 0.0 on 64/64 gates and g < 0 on 64/64, because budget_s was pinned at
    # the ep641 STARTING level while realized Lane descended 0.122438 -> 0.072225.  The
    # constant licensed the primal to give back ALL 0.050213 S-units of won Lane before the
    # guard could respond.  ON: budget becomes a monotone non-increasing RATCHET that locks
    # in won Lane, with a deadband whose sigma is measured online and whose k is calibrated
    # against the null.  DEFAULT-OFF so sealed tickets recompile bit-identical; when off the
    # guard now SELF-REPORTS its inertness (inertness_alarm in the gate telemetry).
    lane_guard_ratchet: bool = False
    lane_guard_ratchet_horizon: int = 0   # 0 => DERIVE the run's planned gate total
                                          # (ddm_lp1 #934; was "gates seen", which
                                          # under-priced the deadband early: 3/64 false
                                          # positives MEASURED on the burn-4 series)
    # ---- ddm_p4x (#920) LANE EXISTENCE PRIMITIVE + per-class BIRTH MATRIX ---------------
    # A SEPARATE loss TERM at COMPONENT granularity -- NOT another seg_pixel_w addend.
    # Every lane_guard mechanism above is a per-pixel weight, which is why the cg1r ledger
    # records protection=ABSENT for the ANNIHILATE verb specifically: a rim-peel guard
    # up-weights currently-WON support and does nothing for a whole component being lost.
    # cg1r MEASURED per-flip depth as direction-SYMMETRIC (1.074x Road<->Lane) while the
    # count asymmetry runs to 15.88x -- the discount is VOLUMETRIC, so protection must be
    # existence/component-level. s(c) = logsumexp_beta(live margin over c) -> the word's
    # WITNESS pixel; loss = mean_c w_c * relu(target - s(c)). O(#components) ~ 27.6/frame.
    # DEFAULT-OFF (weight 0.0) => term never built => BYTE-IDENTICAL.
    existence_hinge_weight: float = 0.0        # 0.0 => OFF
    existence_hinge_classes: str = "lane,movable"  # measured: the only two with real annihilation
    existence_hinge_beta: float = 0.0          # 0.0 => per-class DERIVED log(mean_area)/tolerance
    existence_hinge_target: float = 0.0        # 0.0 => bare existence (the decision boundary)
    existence_hinge_weight_policy: str = ""    # "" => per-class policy from the BIRTH_MATRIX
    existence_hinge_connectivity: int = 8      # 8 = Rosenfeld/receiver; 4 = gt2's own grammar
    # ---- ddm_bp1 (#824) BOUNDARY RESET RACE — arm selector (default = arm B incumbent) ----
    adam_bias_correction: bool = False    # gc15 §7 / tac.optimization.reset_operator: False =
    # ARM_B_ZERO_RESET (MLX's own Adam default => arm A/control trains BIT-IDENTICALLY to every
    # pre-#824 run); True = ARM_BPRIME_BIAS_CORRECTED. MEASURED (this module's test, real MLX):
    # optim.Adam(lr) and optim.Adam(lr, bias_correction=False) produce IDENTICAL updates, and
    # the corrected/uncorrected step ratio is exactly 1/eta(t), eta(t)=(1-b1^t)/sqrt(1-b2^t) —
    # eta(1)=3.1623, max eta(12)=6.5685, and 1212.57 excess sign-steps = 16.168 epochs at 75
    # steps/epoch (MEASURED here at n=20k = 20 time constants, i.e. the CONVERGED sum; the
    # reset_operator docstring's ~1203/~16.0 is the same quantity over a shorter window)
    # of free displacement per moment reset. This field IS in the config identity (it changes
    # training), unlike the read-only telemetry/probe flags which are args-only.

    @property
    def grid_h(self) -> int:
        return SEG_H // self.grid_downsample

    @property
    def grid_w(self) -> int:
        return SEG_W // self.grid_downsample

    @property
    def n_upsample(self) -> int:
        n = round(np.log2(self.grid_downsample))
        if 2 ** n != self.grid_downsample:
            raise ValueError(
                f"grid_downsample must be a power of 2 dividing 384 and 512 (got "
                f"{self.grid_downsample}); D=12 is EXCLUDED (512/12 non-integer lattice "
                "— tb1 design-memo deviation from SPEC S1.2's {8,12,16} race set)")
        return n

    def canonical_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))

    def config_hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()


def derive_ema_decay(total_updates: int) -> tuple[float, str]:
    """EMA decay from RUN GEOMETRY (registered LawRef ema_decay_run_geometry_v1).

    Pin warmup_fraction phi=0.5 (the two-time-constant warmup completes halfway
    through the bounded window) => d = 1 - 2/(phi*U). Clamped to [0.9, 0.9995]
    for very short smoke windows. NEVER the flat borrowed 0.997.
    """
    try:
        from tac.canonical_equations.evaluators import eval_ema_decay_run_geometry

        d = float(eval_ema_decay_run_geometry({
            "mode": "decay_from_warmup_fraction",
            "warmup_fraction": 0.5,
            "updates_per_run": max(int(total_updates), 8),
        }))
        prov = (f"DERIVED ema_decay_run_geometry_v1 decay_from_warmup_fraction "
                f"phi=0.5 U={total_updates} -> {d:.6f}")
    except Exception as exc:  # LawRef unavailable: closed form of the SAME law
        d = 1.0 - 2.0 / (0.5 * max(int(total_updates), 8))
        prov = (f"DERIVED closed-form d=1-2/(phi*U) phi=0.5 U={total_updates} -> "
                f"{d:.6f} (LawRef evaluator import failed: {exc})")
    # QA86(c) / census T6 FIX: the guard window is DERIVED from run geometry, NEVER a
    # constant. The old [0.9, 0.9995] clamp was a tiny-smoke guard whose UPPER cap
    # BOUND OVER the derived long-run value: at U=30,000 (400 ep x 75 batches) the
    # phi=0.5 law gives 0.999867, but 0.9995 collapsed the two-time-constant warmup
    # window from 2/(1-d)=15,038 -> 4,000 steps, violating the phi=0.5 design that the
    # whole LawRef exists to honor (constants-are-poison instance). The law with
    # phi=0.5 already guarantees 0<d<1 (max(U,8) => phi*U>=4>2 => d>0; finite U => d<1);
    # the ONLY defensible bound is that warmup must COMPLETE within the run
    # (phi<=1 => d <= 1 - 2/U). That ceiling is strictly <1 (no frozen shadow) AND is
    # always >= the phi=0.5 derived value (1-4/U <= 1-2/U), so it NEVER binds the
    # design — it only catches a degenerate explicit/short-window request. No floor:
    # the phi=0.5 law is self-consistent at every scale (a smoke's warmup completes at
    # its own halfway point); a constant floor would distort short-run shadows.
    u_eff = max(int(total_updates), 8)
    d_ceiling = 1.0 - 2.0 / u_eff  # phi=1 warmup-fills-run bound (strictly < 1)
    if d > d_ceiling:
        d = d_ceiling
        prov += f"; run-geometry ceiling d<=1-2/U={d_ceiling:.6f} bound (no constant clamp)"
    else:
        prov += f"; within run-geometry window (ceiling 1-2/U={d_ceiling:.6f}, no constant clamp)"
    return d, prov


# ---------------------------------------------------------------------------
# Model (MLX). Trainable trees only; the LOTTO fixed bank is hidden from MLX's
# parameter traversal inside a plain object (regenerable from counted seed).
# ---------------------------------------------------------------------------
class _FixedBank:
    """Opaque (non-Module) holder so MLX does not treat fixed weights as trainable."""

    def __init__(self, tensors: dict[str, Any]):
        self.tensors = tensors


#: QA83 comma10k class luma anchors (the ur-instance: one luma axis separates the 5
#: classes; class_values 41/76/90/124/161). Documented reference for the lift's
#: initialization span; the seg gradient through R places c(x), and the margin-optimal
#: RGB refinement (v14 / rank-4 head) is a named compress-time scorer step (a tiny counted
#: lift table if adopted). The default lift is a full [0,255] gray ramp that spans them.
_COMMA10K_LUMA_ANCHORS: tuple[float, ...] = (41.0, 76.0, 90.0, 124.0, 161.0)


def _head_out_ch(cfg: "TR1Config") -> int:
    """QA83 factorized-head output channel count: rgb=3, class_field=1, class_field_photo=2."""
    return {"rgb": 3, "class_field": 1, "class_field_photo": 2}[
        getattr(cfg, "renderer_head_mode", "rgb")]


def _apply_head(mx, x, cfg: "TR1Config"):
    """QA83 (census §4.1) apply the factorized OUTPUT head. ``x`` is the raw head-conv
    output (…, out_ch); returns (…, 3) RGB in [0, 255] (pre-R). The lift is FIXED generic
    decoder code (rule-118 free); only the k-channel token field is counted."""
    mode = getattr(cfg, "renderer_head_mode", "rgb")
    if mode == "rgb":
        return mx.sigmoid(x) * 255.0                      # exact current behavior (control)
    c = mx.sigmoid(x[..., 0:1])                           # class field in [0,1] (…,1)
    luma = c * 255.0                                      # monotone gray-ramp lift (…,1)
    rgb = mx.concatenate([luma, luma, luma], axis=-1)     # 1-luma-channel ur-instance -> (…,3)
    if mode == "class_field_photo":
        # margin-slack-confined luma photometric channel (QA80 band lemma): a small fixed-gain
        # luma modulation carries pose-legible structure at ~zero seg cost. tanh(p) in [-1,1].
        p = mx.tanh(x[..., 1:2]) * (float(cfg.head_photo_slack_gain) * 255.0)
        rgb = mx.clip(rgb + p, 0.0, 255.0)
    return rgb


def _conv_shapes(cfg: TR1Config) -> list[tuple[str, tuple[int, int, int, int]]]:
    """Conv weight shapes (MLX layout: (C_out, kh, kw, C_in)). RF ~= 3 conv layers
    per SPEC S1.3 (conv0 + one conv per x2 upsample + head). QA83: the head output
    channel count is the factorized-head DOF (rgb=3, class_field=1, class_field_photo=2)."""
    w = cfg.renderer_width
    shapes: list[tuple[str, tuple[int, int, int, int]]] = [("conv0", (w, 3, 3, cfg.code_width))]
    for k in range(cfg.n_upsample):
        shapes.append((f"up{k}", (w, 3, 3, w)))
    shapes.append(("head", (_head_out_ch(cfg), 3, 3, w)))
    return shapes


def build_module(cfg: TR1Config):
    import mlx.core as mx
    import mlx.nn as nn

    shapes = _conv_shapes(cfg)

    class TR1Module(nn.Module):
        def __init__(self):
            super().__init__()
            rng = np.random.default_rng(cfg.seed)
            # Token fields (SYMMETRY force): ZERO-INIT so counted token capacity is
            # loss-driven only — gradients vanish along ker(A), so no counted bytes
            # land on the gauge orbit by construction. shared_base = identity-xi
            # zeroth-order advection (base coded ONCE + per-frame deltas).
            tok_shape = (cfg.num_pairs, cfg.grid_h, cfg.grid_w, cfg.code_width)
            if cfg.token_temporal_mode == "shared_base":
                self.tokens_base = mx.zeros((cfg.grid_h, cfg.grid_w, cfg.code_width))
                self.tokens_delta = mx.zeros(tok_shape)
            elif cfg.token_temporal_mode == "independent":
                self.tokens = mx.zeros(tok_shape)
            else:
                raise ValueError(f"unknown token_temporal_mode {cfg.token_temporal_mode!r}")
            # Deterministic subtractive-dither field (RACED STE; seeded => decoder-
            # regenerable from the counted selector ledger; FREE generic expansion).
            self._dither = _FixedBank({"u": mx.array(
                (np.random.default_rng(cfg.seed + 7).random(
                    (cfg.grid_h, cfg.grid_w, cfg.code_width)) - 0.5).astype(np.float32))})
            # §3.1 COARSE-FROM-BIRTH cell mask (sg1 §2): a (gh,gw,1) fixed {0,1} field.
            # Multiplied into the token field in raw_tokens => inactive cells are exactly 0
            # (renderer sees 0 there) AND their gradient vanishes (multiply-by-0) => they never
            # leave the zero lattice point => no counted bytes on them (byte-close excludes them).
            # None => all-ones (uniform dense grid, the tb1 control).
            if cfg.token_cell_mask is not None:
                m = np.load(cfg.token_cell_mask)
                if m.shape != (cfg.grid_h, cfg.grid_w):
                    raise ValueError(
                        f"token_cell_mask shape {m.shape} != grid ({cfg.grid_h},{cfg.grid_w}) "
                        f"at D={cfg.grid_downsample}; fail-closed (never-invent geometry)")
                cell_keep = m.astype(np.float32)[..., None]  # (gh,gw,1)
            else:
                cell_keep = np.ones((cfg.grid_h, cfg.grid_w, 1), dtype=np.float32)
            self._cell_mask = _FixedBank({"keep": mx.array(cell_keep)})
            # QA84 (census §4.2) row-band variable-cell tiling: build the grammar and validate
            # its fine dims match the (D8) grid; the tie is applied in raw_tokens (differentiable
            # gather) so render + byte-close see the SAME tied field.
            self._rowband = _build_rowband_grammar(cfg)
            # ax1 Pool-A (ddm_pa1b #793): FIXED per-cell effective-level map (margin-coupled quant)
            # + xi-informed delta-sparsity weight field, from the MEASURED QA80 field. Both are
            # _FixedBank (non-trainable => not checkpointed, not in EMA => byte-identical resume).
            # None => no buffer => the forward + byte-close are bit-identical to the control.
            _lvl_np, _dw_np = _build_pool_a_banks(cfg)
            if _lvl_np is not None:
                lm = _lvl_np.astype(np.float32)[..., None]                   # (gh,gw,1) level COUNT
                if self._rowband is not None:  # tie the level map so tied VALUE cells share a level
                    lm = np.asarray(self._rowband.apply_tie_np(lm[None])[0], dtype=np.float32)
                self._level_map = _FixedBank({"L": mx.array(lm)})
            else:
                self._level_map = None
            self._delta_sparsity_weight_field = (
                _FixedBank({"w": mx.array(_dw_np.astype(np.float32))}) if _dw_np is not None
                else None)
            # engagement is event-driven (§7 base-stability = the CE->tau knee); "from_step_0" is
            # the gc10 F2 nu-snap warm-start holder. Plain bool (never in MLX trainable traversal).
            self._delta_sparsity_engaged = (
                cfg.token_delta_group_sparsity == "on"
                and cfg.delta_sparsity_engage == "from_step_0")
            # §3.3(a) lattice-anneal / staged quantizer: when "at_knee" the STE is DISENGAGED
            # (float tokens) until the CE->tau knee EVENT flips this to True (see main()); "off"
            # engages the STE from birth (tb1 control). Plain bool attribute (NOT an mx.array =>
            # never enters MLX's trainable traversal).
            self._quant_engaged = (cfg.token_quant_anneal != "at_knee")
            if cfg.variant == "plain":
                for name, shp in shapes:
                    fan_in = shp[1] * shp[2] * shp[3]
                    std = float(np.sqrt(2.0 / fan_in))
                    setattr(self, f"w_{name}", mx.array(
                        (rng.standard_normal(shp) * std).astype(np.float32)))
                    setattr(self, f"b_{name}", mx.zeros((shp[0],)))
                self._bank = _FixedBank({})
            elif cfg.variant == "lotto":
                # Fixed-seed generated bank (FREE generic PRNG expansion; the SEED is
                # COUNTED in the selector ledger). Scores + per-out-channel modulations
                # + biases are the COUNTED learned payload.
                lrng = np.random.default_rng(cfg.lotto_seed)
                bank: dict[str, Any] = {}
                init_score = float(np.log(cfg.lotto_mask_density_init /
                                          max(1e-6, 1.0 - cfg.lotto_mask_density_init)))
                for name, shp in shapes:
                    fan_in = shp[1] * shp[2] * shp[3]
                    # Signed-constant bank (edge-popup style magnitude, He-scaled).
                    mag = float(np.sqrt(2.0 / fan_in))
                    bank[name] = mx.array(
                        (lrng.choice([-1.0, 1.0], size=shp) * mag).astype(np.float32))
                    setattr(self, f"s_{name}", mx.array(
                        (rng.standard_normal(shp) * 0.05 + init_score).astype(np.float32)))
                    setattr(self, f"g_{name}", mx.ones((shp[0],)))
                    setattr(self, f"b_{name}", mx.zeros((shp[0],)))
                self._bank = _FixedBank(bank)
            else:
                raise ValueError(f"unknown variant {cfg.variant!r}")

            # Window C (ddm_dw1, MAIN charter): warm-start-equivalent output-chart relax.
            # A trainable per-channel gain on a LINEAR residual of the head pre-activation,
            # INIT 0 => at the resumed weights the head output == sigmoid(x)*255 EXACTLY
            # (warm-start equivalence; verified by first-epoch loss == Window A). As the gain
            # trains it de-saturates the sigmoid so gradients reach out-of-chart (dark) pixels
            # — the direct test of whether the rgb output chart binds. Only built when
            # requested (default "off" => no new param => resume/checkpoint byte-compatible).
            if getattr(cfg, "head_range_relax", "off") == "linear":
                if getattr(cfg, "renderer_head_mode", "rgb") != "rgb":
                    raise ValueError(
                        "head_range_relax='linear' requires renderer_head_mode='rgb' "
                        f"(got {cfg.renderer_head_mode!r}); the residual is added to the "
                        "3-channel head pre-activation — fail closed")
                self.head_relax_gain = mx.zeros((_head_out_ch(cfg),))

        # -- description-level eval_roundtrip: token lattice with STE ------------
        def raw_tokens(self, idx: int):
            if cfg.token_temporal_mode == "shared_base":
                t = self.tokens_base + self.tokens_delta[idx]
            else:
                t = self.tokens[idx]
            pe3 = getattr(self, "_pe3_conditioning", None)
            if pe3 is not None:
                feats = pe3.tensors["features"][:, int(idx)]  # (mode, gh, gw, c)
                gates = self.pe3_conditioning_gate.reshape((-1, 1, 1, 1))
                t = t + mx.sum(gates * mx.stop_gradient(feats), axis=0)
            # §3.1 coarse-from-birth: zero the inactive cells (fixed {0,1} mask, stop-grad).
            t = t * mx.stop_gradient(self._cell_mask.tensors["keep"])
            if self._rowband is not None:  # QA84 §4.2: tie bulk 2x2 blocks (D16-effective)
                t = self._rowband.apply_tie_mx(mx, t)
            return t

        def quantized_tokens(self, idx: int):
            t = mx.clip(self.raw_tokens(idx), -1.0, 1.0)  # (gh, gw, c)
            # §3.3(a) lattice anneal: before the knee (at_knee mode) the STE is disengaged =>
            # float tokens (find the basin in float, refine on the shipped lattice after).
            if not self._quant_engaged:
                return t
            # ax1 §2a margin-coupled quant: per-cell effective L from the QA80 field when the
            # level map is present; else the scalar-L control (byte-identical). L broadcasts
            # (gh,gw,1) over the (gh,gw,c) token field.
            if self._level_map is not None:
                L = self._level_map.tensors["L"] - 1.0                     # (gh,gw,1) per-cell
            else:
                L = float(cfg.token_quant_levels - 1)
            x01 = (t + 1.0) * 0.5
            if cfg.token_ste == "dither":
                u = mx.stop_gradient(self._dither.tensors["u"])
                q = (mx.round(x01 * L + u) - u) / L * 2.0 - 1.0
            else:  # "round"
                q = mx.round(x01 * L) / L * 2.0 - 1.0
            return t + mx.stop_gradient(q - t)

        def _weight(self, name: str):
            if cfg.variant == "plain":
                return getattr(self, f"w_{name}")
            s = getattr(self, f"s_{name}")
            soft = mx.sigmoid(s)
            hard = (s > 0.0).astype(soft.dtype)
            mask = soft + mx.stop_gradient(hard - soft)  # STE binary supermask
            g = getattr(self, f"g_{name}")               # per-out-channel modulation
            wfix = mx.stop_gradient(self._bank.tensors[name])
            return wfix * mask * g.reshape((-1, 1, 1, 1))

        def render_frame(self, idx: int):
            """(1, SEG_H, SEG_W, 3) float RGB in [0, 255] (pre-R)."""
            x = self.quantized_tokens(idx)[None]  # (1, gh, gw, c)
            x = mx.conv2d(x, self._weight("conv0"), padding=1) + self.b_conv0
            x = nn.gelu(x)
            for k in range(cfg.n_upsample):
                x = mx.repeat(mx.repeat(x, 2, axis=1), 2, axis=2)  # nearest x2
                x = mx.conv2d(x, self._weight(f"up{k}"), padding=1) + getattr(self, f"b_up{k}")
                x = nn.gelu(x)
            x = mx.conv2d(x, self._weight("head"), padding=1) + self.b_head
            out = _apply_head(mx, x, cfg)  # QA83 factorized head (rgb control = sigmoid*255)
            if getattr(cfg, "head_range_relax", "off") == "linear":
                # de-saturating linear residual, gain INIT 0 => identity at warm start.
                out = out + self.head_relax_gain.reshape((1, 1, 1, -1)) * x
            return out

    return TR1Module()


def parse_tr1_birth_seed_classes(raw: str) -> tuple[str, ...]:
    """Parse BI1 class names in a stable order; fail closed on unknown classes."""
    names = tuple(dict.fromkeys(s.strip().lower() for s in str(raw).split(",") if s.strip()))
    if not names:
        raise ValueError("--tr1-birth-seed-classes must name at least one class")
    unknown = [n for n in names if n not in TR1_BIRTH_SEED_CLASS_IDS]
    if unknown:
        raise ValueError(
            f"--tr1-birth-seed-classes unknown {unknown}; choose from "
            f"{sorted(TR1_BIRTH_SEED_CLASS_IDS)}"
        )
    return names


def _tr1_birth_token_pattern(class_id: int, code_width: int, weight: float) -> np.ndarray:
    """Class-distinct deterministic token direction; generic code, no learned table."""
    idx = np.arange(int(code_width), dtype=np.int32)
    signs = np.where(((idx + int(class_id)) % 2) == 0, 1.0, -1.0).astype(np.float32)
    return (float(weight) * signs).astype(np.float32)


def _downsample_birth_weight(weight: np.ndarray, cfg: TR1Config) -> np.ndarray:
    """Max-pool a scorer-plane birth support to the TR1 token lattice."""
    w = np.asarray(weight, dtype=np.float32)
    if w.shape != (SEG_H, SEG_W):
        raise ValueError(f"birth weight shape {w.shape} != {(SEG_H, SEG_W)}")
    d = int(cfg.grid_downsample)
    return w.reshape(cfg.grid_h, d, cfg.grid_w, d).max(axis=(1, 3)).astype(np.float32)


def build_tr1_birth_seed_bank(
    cfg: TR1Config,
    lstars: Any,
    *,
    weight: float,
    classes: str = TR1_BIRTH_SEED_DEFAULT_CLASSES,
    dilate_px: int = 1,
    persist: str = "inverse_thickness",
) -> tuple[dict[str, np.ndarray] | None, dict[str, Any]]:
    """Build the BI1 scorer-free token birth bank from GT argmax islands.

    The bank is a train-time initialization/anchor target on the token field.  It
    does not call SegNet/PoseNet and is never an inflate-time payload.
    """
    w0 = float(weight)
    names = parse_tr1_birth_seed_classes(classes)
    if w0 <= 0.0:
        return None, {
            "active": False,
            "classes": list(names),
            "seed_weight": w0,
            "reason": "weight<=0",
        }
    if int(dilate_px) < 0:
        raise ValueError("--tr1-birth-seed-dilate-px must be >= 0")
    if persist not in ("uniform", "inverse_thickness"):
        raise ValueError("--tr1-birth-amplify-persist must be uniform|inverse_thickness")
    if getattr(cfg, "token_rowband_spec", None) is not None:
        raise ValueError("BI1 birth seed currently requires untied token cells; "
                         "--token-rowband-spec needs an explicit tied-target projection")

    from tac.boundary_math.island_protection import eased_island_masks, island_persistence_weight

    target = np.zeros((cfg.num_pairs, cfg.grid_h, cfg.grid_w, cfg.code_width), dtype=np.float32)
    mask = np.zeros((cfg.num_pairs, cfg.grid_h, cfg.grid_w, 1), dtype=np.float32)
    seeded_by_class = dict.fromkeys(names, 0)
    seeded_pairs: set[int] = set()
    class_ids = {n: TR1_BIRTH_SEED_CLASS_IDS[n] for n in names}
    lane_cls = class_ids.get("lane")
    movable_cls = class_ids.get("movable")

    for pair_idx in range(int(cfg.num_pairs)):
        im = eased_island_masks(
            np.asarray(lstars[pair_idx], dtype=np.int64),
            lane_cls,
            movable_cls,
            dilate_px=int(dilate_px),
        )
        masks = {"lane": im.lane_mask, "movable": im.movable_mask}
        for name in names:
            cmask = masks[name]
            if cmask is None or not np.asarray(cmask, bool).any():
                continue
            px_weight = island_persistence_weight(np.asarray(cmask, bool), kind=persist)
            cell_w = _downsample_birth_weight(px_weight, cfg)
            active = cell_w > 0.0
            n_cells = int(active.sum())
            if n_cells == 0:
                continue
            seeded_by_class[name] += n_cells
            seeded_pairs.add(pair_idx)
            pattern = _tr1_birth_token_pattern(class_ids[name], cfg.code_width, w0)
            target[pair_idx] += cell_w[..., None] * pattern[None, None, :]
            mask[pair_idx, ..., 0] = np.maximum(mask[pair_idx, ..., 0], cell_w)

    np.clip(target, -1.0, 1.0, out=target)
    # Masked-lattice coverage check (replaces the former categorical refusal): a
    # keep-masked cell is multiplied to exactly 0 with vanishing gradient AND is
    # excluded from byte-close, so a seed outside the keep mask would be a
    # gradient-dead target the byte ledger never prices. Permit a masked lattice
    # ONLY when every seeded cell is kept — the union-mask workflow (keep ∪ seeds,
    # materialized as a mask FILE so trainer/byte-close/receiver see one object,
    # and the birthed cells' rate is counted) is how an ON arm satisfies this.
    uncovered_cells = 0
    if getattr(cfg, "token_cell_mask", None) is not None:
        keep = np.load(cfg.token_cell_mask)
        if keep.shape != (cfg.grid_h, cfg.grid_w):
            raise ValueError(
                f"token_cell_mask shape {keep.shape} != grid ({cfg.grid_h},{cfg.grid_w})")
        seeded_any = mask[..., 0].max(axis=0) > 0.0
        uncovered = seeded_any & (keep.astype(bool) == False)  # noqa: E712
        uncovered_cells = int(uncovered.sum())
        if uncovered_cells > 0:
            raise ValueError(
                f"BI1 birth seed: {uncovered_cells} seeded cell(s) fall OUTSIDE the "
                f"token keep mask ({cfg.token_cell_mask}) and would be gradient-dead + "
                f"unpriced. Materialize the union mask (keep | seeded cells) as a new "
                f".npy and pass it as --token-cell-mask so byte-close counts the "
                f"birthed cells.")
    summary = {
        "active": True,
        "mechanism": "BI1 token-lattice birth seed plus scorer-free amplify anchor",
        "classes": list(names),
        "seed_weight": w0,
        "dilate_px": int(dilate_px),
        "persist": persist,
        "pairs_with_seed": len(seeded_pairs),
        "seeded_cells_total": int(sum(seeded_by_class.values())),
        "seeded_cells_by_class": seeded_by_class,
        "target_abs_sum": float(np.abs(target).sum()),
        "mask_sum": float(mask.sum()),
        "masked_lattice": getattr(cfg, "token_cell_mask", None) is not None,
        "uncovered_seed_cells": uncovered_cells,
        "score_claim": False,
    }
    if summary["target_abs_sum"] <= 0.0:
        raise ValueError("BI1 birth seed produced zero token target; refusing an inert ON arm")
    return {"target": target, "mask": mask}, summary


def attach_tr1_birth_seed_bank(
    model: Any,
    cfg: TR1Config,
    lstars: Any,
    *,
    weight: float,
    classes: str = TR1_BIRTH_SEED_DEFAULT_CLASSES,
    dilate_px: int = 1,
    persist: str = "inverse_thickness",
    apply_live_seed: bool = True,
) -> dict[str, Any]:
    """Attach BI1 target/mask banks and optionally seed the live token field."""
    bank_np, summary = build_tr1_birth_seed_bank(
        cfg, lstars, weight=weight, classes=classes, dilate_px=dilate_px, persist=persist
    )
    if bank_np is None:
        return summary
    import mlx.core as mx

    bank = _FixedBank({k: mx.array(v) for k, v in bank_np.items()})
    model._tr1_birth_seed = bank
    if apply_live_seed:
        target = bank.tensors["target"]
        if cfg.token_temporal_mode == "shared_base":
            model.tokens_delta = model.tokens_delta + target
        else:
            model.tokens = model.tokens + target
        mx.eval(model.parameters())
    summary["applied_to_live_tokens"] = bool(apply_live_seed)
    return summary


def tr1_birth_amplify_term(mdl: Any, ids: Sequence[int]):
    """Scorer-free BI1 anchor: keep seeded token support from washing out."""
    import mlx.core as mx

    bank = getattr(mdl, "_tr1_birth_seed", None)
    if bank is None or not ids:
        return mx.array(0.0)
    target = bank.tensors["target"]
    mask = bank.tensors["mask"]
    acc = None
    for idx in ids:
        i = int(idx)
        diff = (mdl.raw_tokens(i) - target[i]) * mask[i]
        denom = mx.sum(mask[i]) * diff.shape[-1] + 1e-8
        term = mx.sum(diff * diff) / denom
        acc = term if acc is None else acc + term
    return acc / len(ids)


def _sha256_bytes(blob: bytes) -> str:
    return hashlib.sha256(blob).hexdigest()


def _sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_single_member_zip_or_raw(path: Path) -> bytes:
    if not path.is_file():
        raise ValueError(f"cache path does not exist: {path}")
    if path.suffix == ".zip":
        with zipfile.ZipFile(path, "r") as zf:
            names = [name for name in zf.namelist() if not name.endswith("/")]
            for name in ("archive/0.bin", "0.bin"):
                if name in names:
                    return zf.read(name)
            if len(names) == 1:
                return zf.read(names[0])
            raise ValueError(f"cannot identify single payload member in {path}: {names}")
    return path.read_bytes()


def _extract_pe3_conditioning_section(cache_path: Path) -> tuple[bytes, dict[str, Any]]:
    """Reuse the PE3 receiver parser and fail closed on the LC1/PK1 section SHA."""
    blob = _load_single_member_zip_or_raw(cache_path)
    import inflate_runner_v4d as receiver

    if blob.startswith(receiver.PE3_EDGE_MAGIC):
        section = blob
    else:
        from tac.optimization.ddm_ix2_archive_container import parse_payload

        _bulk, sections = parse_payload(blob)
        matches = [
            bytes(section)
            for section in sections
            if bytes(section).startswith(receiver.PE3_EDGE_MAGIC)
        ]
        if len(matches) != 1:
            raise ValueError(
                f"expected exactly one PE3EDGE1 section in {cache_path}, found {len(matches)}"
            )
        section = matches[0]
    meta = dict(receiver._pe3_parse_edge_field(section))
    got_sha = str(meta.get("section_sha256") or _sha256_bytes(section))
    if got_sha != PE3_CONDITIONING_EXPECTED_SECTION_SHA256:
        raise ValueError(
            "PE3 conditioning cache SHA mismatch: "
            f"{got_sha} != {PE3_CONDITIONING_EXPECTED_SECTION_SHA256}"
        )
    return section, meta


def _parse_pe3_conditioning_components(section: bytes) -> tuple[list[list[dict[str, Any]]], dict[str, Any]]:
    """Parse PE3 components via the receiver's record decoders, preserving mode identity."""
    import inflate_runner_v4d as receiver

    if len(section) < receiver.PE3_EDGE_HEADER.size:
        raise ValueError("PE3 section header truncated")
    (
        magic,
        version,
        seg_h,
        seg_w,
        n_pairs,
        kind,
        codec,
        raw_len,
        frame_record_count,
        raw_sha,
    ) = receiver.PE3_EDGE_HEADER.unpack_from(section, 0)
    if magic != receiver.PE3_EDGE_MAGIC or version != receiver.PE3_EDGE_VERSION:
        raise ValueError("PE3 section magic/version differs")
    if int(kind) != receiver._PE3_HYBRID:
        raise ValueError("PE3 section kind differs")
    if int(frame_record_count) != int(n_pairs):
        raise ValueError("PE3 frame record count differs")
    raw = receiver._pe1_decode_body(int(codec), section[receiver.PE3_EDGE_HEADER.size:])
    if len(raw) != int(raw_len):
        raise ValueError("PE3 raw body length differs")
    if hashlib.sha256(raw).digest() != raw_sha:
        raise ValueError("PE3 raw body SHA differs")

    rows: list[list[dict[str, Any]]] = []
    mode_counts = dict.fromkeys(PE3_CONDITIONING_MODE_ORDER, 0)
    component_count = 0
    offset = 0
    for _pair in range(int(n_pairs)):
        count, offset = receiver._pe1_read_varint(raw, offset)
        pair_rows: list[dict[str, Any]] = []
        for _ in range(int(count)):
            length, offset = receiver._pe1_read_varint(raw, offset)
            record = raw[offset:offset + int(length)]
            if len(record) != int(length):
                raise ValueError("PE3 component record truncated")
            offset += int(length)
            if not record:
                raise ValueError("PE3 empty component record")
            mode = int(record[0])
            payload = record[1:]
            if mode == receiver._PE3_MODE_CURVE:
                indices, classes = receiver._pe1_curve_indices(payload, int(seg_h), int(seg_w))
            elif mode == receiver._PE3_MODE_GENERATOR:
                indices, classes = receiver._pe1_generator_indices(payload, int(seg_h), int(seg_w))
            else:
                raise ValueError(f"unknown PE3 component mode {mode}")
            mode_name = receiver._PE3_MODE_NAMES[mode]
            if mode_name not in mode_counts:
                raise ValueError(f"unexpected PE3 mode name {mode_name!r}")
            if int(indices.size) != int(classes.size):
                raise ValueError("PE3 component index/class length differs")
            mode_counts[mode_name] += 1
            component_count += 1
            pair_rows.append({
                "mode_name": mode_name,
                "indices": np.asarray(indices, dtype=np.int32),
                "classes": np.asarray(classes, dtype=np.uint8),
                "record_bytes": int(length),
            })
        rows.append(pair_rows)
    if offset != len(raw):
        raise ValueError("PE3 raw body has trailing bytes")
    meta = {
        "seg_h": int(seg_h),
        "seg_w": int(seg_w),
        "n_pairs": int(n_pairs),
        "raw_bytes": int(raw_len),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "section_bytes": len(section),
        "section_sha256": _sha256_bytes(section),
        "component_records": int(component_count),
        "mode_counts": mode_counts,
    }
    return rows, meta


def _pe3_class_pattern(classes: np.ndarray, code_width: int) -> np.ndarray:
    cls = np.asarray(classes, dtype=np.int32).reshape(-1)
    channels = np.arange(int(code_width), dtype=np.int32).reshape(1, -1)
    signs = np.where(((cls.reshape(-1, 1) + channels) % 2) == 0, 1.0, -1.0)
    scale = (cls.reshape(-1, 1).astype(np.float32) + 1.0) / 5.0
    return (signs.astype(np.float32) * scale).astype(np.float32)


def _token_grid_proximity(active: np.ndarray) -> np.ndarray:
    active = np.asarray(active, dtype=bool)
    out = np.zeros(active.shape, dtype=np.float32)
    pts = np.argwhere(active)
    if pts.size == 0:
        return out
    yy, xx = np.indices(active.shape, dtype=np.float32)
    best = np.full(active.shape, np.inf, dtype=np.float32)
    for r, c in pts:
        d2 = (yy - float(r)) ** 2 + (xx - float(c)) ** 2
        best = np.minimum(best, d2)
    out = 1.0 / (1.0 + np.sqrt(best, dtype=np.float32))
    out[active] = 1.0
    return out.astype(np.float32)


def pe3_conditioning_features_from_components(
    cfg: TR1Config,
    components_by_pair: Sequence[Sequence[dict[str, Any]]],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build per-mode PE3 prior channels on the TR1 token lattice.

    These channels are INPUT conditioning only.  They never replace labels or
    targets; learned mode gates decide whether each grammar mode is trusted.
    """
    if len(components_by_pair) < int(cfg.num_pairs):
        raise ValueError(
            f"PE3 cache has {len(components_by_pair)} pairs < --num-pairs {cfg.num_pairs}"
        )
    mode_index = {name: i for i, name in enumerate(PE3_CONDITIONING_MODE_ORDER)}
    features = np.zeros(
        (len(PE3_CONDITIONING_MODE_ORDER), cfg.num_pairs, cfg.grid_h, cfg.grid_w, cfg.code_width),
        dtype=np.float32,
    )
    counts = np.zeros((*features.shape[:-1], 1), dtype=np.float32)
    described_pixels = dict.fromkeys(PE3_CONDITIONING_MODE_ORDER, 0)
    pairs_by_mode = {name: set() for name in PE3_CONDITIONING_MODE_ORDER}
    d = int(cfg.grid_downsample)
    slots = SEG_H * SEG_W
    for pair_idx in range(int(cfg.num_pairs)):
        for comp in components_by_pair[pair_idx]:
            mode_name = str(comp["mode_name"])
            if mode_name not in mode_index:
                raise ValueError(f"unknown PE3 conditioning mode {mode_name!r}")
            indices = np.asarray(comp["indices"], dtype=np.int64).reshape(-1)
            classes = np.asarray(comp["classes"], dtype=np.uint8).reshape(-1)
            if indices.size != classes.size:
                raise ValueError("PE3 conditioning component index/class length differs")
            if indices.size == 0:
                continue
            if int(indices.min()) < 0 or int(indices.max()) >= slots:
                raise ValueError("PE3 conditioning index outside scorer grid")
            mi = mode_index[mode_name]
            rr = (indices // SEG_W) // d
            cc = (indices % SEG_W) // d
            rr = np.clip(rr, 0, cfg.grid_h - 1).astype(np.int64)
            cc = np.clip(cc, 0, cfg.grid_w - 1).astype(np.int64)
            vals = _pe3_class_pattern(classes, cfg.code_width)
            for ch in range(int(cfg.code_width)):
                np.add.at(features[mi, pair_idx, :, :, ch], (rr, cc), vals[:, ch])
            np.add.at(counts[mi, pair_idx, :, :, 0], (rr, cc), 1.0)
            described_pixels[mode_name] += int(indices.size)
            pairs_by_mode[mode_name].add(pair_idx)
    active = counts[..., 0] > 0.0
    features = np.divide(features, np.maximum(counts, 1.0), out=features, where=counts > 0.0)
    for mi in range(len(PE3_CONDITIONING_MODE_ORDER)):
        for pair_idx in range(int(cfg.num_pairs)):
            prox = _token_grid_proximity(active[mi, pair_idx])
            features[mi, pair_idx, :, :, 0] += 0.25 * prox
    np.clip(features, -1.0, 1.0, out=features)
    summary = {
        "active": True,
        "mechanism": "TK1 PE3 conditioning-only prior channels with learned per-mode trust gates",
        "mode_order": list(PE3_CONDITIONING_MODE_ORDER),
        "gate_init": "zeros",
        "pairs": int(cfg.num_pairs),
        "grid": [int(cfg.grid_h), int(cfg.grid_w)],
        "channels_per_mode": int(cfg.code_width),
        "described_pixels_by_mode": described_pixels,
        "pairs_with_signal_by_mode": {
            name: len(pairs_by_mode[name]) for name in PE3_CONDITIONING_MODE_ORDER
        },
        "feature_abs_sum_by_mode": {
            name: float(np.abs(features[i]).sum())
            for i, name in enumerate(PE3_CONDITIONING_MODE_ORDER)
        },
        "score_claim": False,
        "label_replacement": False,
    }
    return features, summary


def build_pe3_conditioning_bank(cfg: TR1Config, cache_path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    section, parseback_meta = _extract_pe3_conditioning_section(cache_path)
    rows, component_meta = _parse_pe3_conditioning_components(section)
    features, summary = pe3_conditioning_features_from_components(cfg, rows)
    summary.update({
        "cache_path": str(cache_path),
        "section_bytes": int(parseback_meta["section_bytes"]),
        "section_sha256": str(parseback_meta["section_sha256"]),
        "raw_sha256": str(parseback_meta["raw_sha256"]),
        "component_records": int(component_meta["component_records"]),
        "mode_counts": dict(component_meta["mode_counts"]),
        "raster_sha256": str(parseback_meta.get("raster_sha256", "")),
    })
    return {"features": features}, summary


def attach_pe3_conditioning_bank(model: Any, cfg: TR1Config, cache_path: Path) -> dict[str, Any]:
    bank_np, summary = build_pe3_conditioning_bank(cfg, cache_path)
    import mlx.core as mx

    model._pe3_conditioning = _FixedBank({k: mx.array(v) for k, v in bank_np.items()})
    model.pe3_conditioning_gate = mx.zeros((len(PE3_CONDITIONING_MODE_ORDER),), dtype=mx.float32)
    mx.eval(model.parameters())
    return summary


def _read_varint(payload: bytes, offset: int) -> tuple[int, int]:
    value = 0
    shift = 0
    while True:
        if offset >= len(payload):
            raise ValueError("varint is truncated")
        byte = int(payload[offset])
        offset += 1
        value |= (byte & 0x7F) << shift
        if (byte & 0x80) == 0:
            return value, offset
        shift += 7
        if shift > 63:
            raise ValueError("varint is too long")


def decode_cheapdct4_stage2_payload(payload: bytes) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    magic = b"OD8S2C1\0"
    if not payload.startswith(magic):
        raise ValueError("cheapdct4 stage2 payload magic mismatch")
    offset = len(magic)
    n_records, offset = _read_varint(payload, offset)
    k, offset = _read_varint(payload, offset)
    source_len, offset = _read_varint(payload, offset)
    source_b = payload[offset:offset + source_len]
    if len(source_b) != source_len:
        raise ValueError("cheapdct4 stage2 source is truncated")
    offset += source_len
    records: list[dict[str, Any]] = []
    for _ in range(int(n_records)):
        pair, offset = _read_varint(payload, offset)
        count, offset = _read_varint(payload, offset)
        if count != 3 * int(k) * int(k):
            raise ValueError(f"cheapdct4 qcoeff count {count} != {3 * int(k) * int(k)}")
        qbytes = payload[offset:offset + count * 2]
        if len(qbytes) != count * 2:
            raise ValueError("cheapdct4 qcoeff payload truncated")
        offset += count * 2
        q = np.frombuffer(qbytes, dtype="<i2").reshape(3, int(k) * int(k)).astype(np.int16, copy=True)
        records.append({
            "pair": int(pair),
            "k": int(k),
            "qcoeffs": q,
            "coeff_sha256": _sha256_bytes(qbytes),
        })
    if offset != len(payload):
        raise ValueError("cheapdct4 stage2 payload has trailing bytes")
    return records, {
        "record_count": int(n_records),
        "k": int(k),
        "source": source_b.decode("ascii", errors="replace"),
        "raw_int16_coeff_bytes": int(sum(r["qcoeffs"].size * 2 for r in records)),
        "payload_sha256": _sha256_bytes(payload),
    }


def _extract_cheapdct4_stage2_from_od5_packet(packet_path: Path) -> tuple[bytes, dict[str, Any]]:
    if not packet_path.is_file():
        raise ValueError(f"cheapdct4 packet path does not exist: {packet_path}")
    from tac.optimization import ddm_od4_weak_stage1_packet as od4

    packet = packet_path.read_bytes()
    parsed = od4.parse_od5_packet(packet)
    matches = [section for section in parsed.sections if section.name in CHEAPDCT4_STAGE2_SECTION_NAMES]
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one cheapdct4 stage2 section in {packet_path}, found {len(matches)}"
        )
    section = matches[0]
    return section.payload, {
        "packet_path": str(packet_path),
        "packet_bytes": len(packet),
        "packet_sha256": _sha256_bytes(packet),
        "packet_sections": [s.name for s in parsed.sections],
        "stage2_section_name": section.name,
    }


def load_cheapdct4_pose_accounting_cache(cache_path: Path) -> dict[str, Any]:
    """Decode OD9 stage2 qcoeffs and bind them to OD9's measured n32 pose term."""
    if not cache_path.is_file():
        raise ValueError(f"cheapdct4 pose cache does not exist: {cache_path}")
    if cache_path.suffix != ".json":
        raise ValueError("cheapdct4 pose accounting requires the OD9 receipt JSON cache")
    receipt = json.loads(cache_path.read_text())
    artifacts = receipt.get("artifacts") or {}
    packet_info = artifacts.get("best_combined_packet") or artifacts.get("absolute_persisted_packet")
    if not isinstance(packet_info, dict) or not packet_info.get("path") or not packet_info.get("sha256"):
        raise ValueError("OD9 receipt lacks a packet path+sha for cheapdct4 stage2 consumption")
    packet_path = Path(str(packet_info["path"]))
    expected_packet_sha = str(packet_info["sha256"])
    got_packet_sha = _sha256_path(packet_path)
    if got_packet_sha != expected_packet_sha:
        raise ValueError(
            f"cheapdct4 packet SHA mismatch: {got_packet_sha} != {expected_packet_sha}"
        )
    payload, packet_meta = _extract_cheapdct4_stage2_from_od5_packet(packet_path)
    records, stage2_meta = decode_cheapdct4_stage2_payload(payload)
    pose_scope = receipt.get("pose_subset_scope") or {}
    d_pose = pose_scope.get("d_pose_after_stage2_cheapdct_mean_n32")
    if d_pose is None:
        raise ValueError("OD9 receipt lacks d_pose_after_stage2_cheapdct_mean_n32")
    stage2_row = None
    for row in receipt.get("combined_table") or []:
        if row and row[0] == "stage2_only_cheapdct4_qcoeffs":
            stage2_row = row
            break
    if stage2_row is None:
        raise ValueError("OD9 receipt lacks stage2_only_cheapdct4_qcoeffs byte row")
    d_pose_f = float(d_pose)
    projected_bytes = int(stage2_row[2])
    return {
        "schema": "tk1_cheapdct4_pose_accounting.v1",
        "mode": "accounting",
        "cache_path": str(cache_path),
        "cache_sha256": _sha256_path(cache_path),
        **packet_meta,
        **stage2_meta,
        "record_pairs": [int(r["pair"]) for r in records],
        "coeff_sha256_head": [str(r["coeff_sha256"]) for r in records[:4]],
        "n32_coded_bytes": int(stage2_row[1]),
        "projected_n600_bytes": projected_bytes,
        "d_pose_after_stage2_cheapdct_mean_n32": d_pose_f,
        "pose_contribution_n32": math.sqrt(10.0 * d_pose_f),
        "rate_contribution_projected_n600": 25.0 * projected_bytes / CONTEST_DENOMINATOR_BYTES,
        "not_projected_to_n600": bool(pose_scope.get("not_projected_to_n600", True)),
        "axis": receipt.get("axis", {}),
        "selection": receipt.get("selection"),
        "score_claim": False,
        "full_in_loop_consumption": False,
        "accounting_scope": "decoded OD9 cheapdct4 qcoeff carriage plus OD9 measured n32 pose term",
    }


def attach_cheapdct4_accounting_to_receipt(receipt: dict[str, Any],
                                           accounting: dict[str, Any] | None) -> None:
    if accounting is None:
        return
    receipt["cheapdct4_pose_accounting"] = accounting
    composed = receipt.get("composed_s_verdict")
    if isinstance(composed, dict):
        composed["cheapdct4_pose_accounting"] = accounting


def pose_null_projector_np() -> np.ndarray:
    """Return sq1's (12,12) projector onto the frame_1 yuv6-null subspace."""
    a = np.zeros((6, 12), dtype=np.float64)
    for p in range(4):
        a[p, 3 * p: 3 * p + 3] = _YUV6_LUMA_WEIGHTS
        a[4, 3 * p + 0] = 0.25
        a[5, 3 * p + 2] = 0.25
    p = np.eye(12, dtype=np.float64) - np.linalg.pinv(a) @ a
    if not (np.allclose(p @ p, p) and np.abs(a @ p).max() < 1e-10):
        raise RuntimeError("pose-null projector construction failed its algebraic checks")
    if np.linalg.matrix_rank(p) != 6:
        raise RuntimeError("pose-null projector rank drifted from 6")
    return p


# PROJECT_PARITY_WAIVED: training frame-DELTA constraint applied before storage; receiver decodes post-projection bytes (parity by construction)
def project_frame1_pose_null_nhwc_np(delta: np.ndarray,
                                     projector: np.ndarray | None = None) -> np.ndarray:
    """Project an NHWC scorer-lattice frame delta blockwise through sq1's P."""
    x = np.asarray(delta)
    if x.ndim != 4 or x.shape[-1] != 3:
        raise ValueError(f"expected NHWC RGB delta, got shape {x.shape}")
    b, h, w, c = x.shape
    if h % 2 or w % 2:
        raise ValueError(f"height/width must be even for yuv6 2x2 blocks, got {(h, w)}")
    p = pose_null_projector_np() if projector is None else np.asarray(projector, dtype=np.float64)
    y = x.reshape(b, h // 2, 2, w // 2, 2, c)
    y = y.transpose(0, 1, 3, 2, 4, 5).reshape(b, h // 2, w // 2, 12)
    y = y @ p.T
    y = y.reshape(b, h // 2, w // 2, 2, 2, c).transpose(0, 1, 3, 2, 4, 5)
    return y.reshape(x.shape).astype(x.dtype, copy=False)


# PROJECT_PARITY_WAIVED: MLX cotangent (gradient-path) twin of the np projector; gradients are never stored
def project_frame1_pose_null_nhwc_mlx(delta, projector=None):
    """MLX twin of project_frame1_pose_null_nhwc_np for cotangent projection."""
    import mlx.core as mx

    p = mx.array(pose_null_projector_np().astype(np.float32)) if projector is None else projector
    b, h, w, c = delta.shape
    if c != 3 or h % 2 or w % 2:
        raise ValueError(f"expected even NHWC RGB delta, got shape {delta.shape}")
    y = mx.reshape(delta, (b, h // 2, 2, w // 2, 2, c))
    y = mx.transpose(y, (0, 1, 3, 2, 4, 5))
    y = mx.reshape(y, (b, h // 2, w // 2, 12))
    y = y @ mx.transpose(p)
    y = mx.reshape(y, (b, h // 2, w // 2, 2, 2, c))
    y = mx.transpose(y, (0, 1, 3, 2, 4, 5))
    return mx.reshape(y, (b, h, w, c))


def q3_project_seg_gradient_identity(frame):
    """Forward identity whose VJP projects the frame cotangent through Q3.

    This is the pg1 training-time constraint: rendered frame_1 bytes are unchanged,
    but the SEG loss gradient entering the renderer is restricted to sq1's exact
    float yuv6-null subspace. Pose loss calls must use the unwrapped render path.
    """
    import mlx.core as mx

    @mx.custom_function
    def _identity(x):
        return x

    @_identity.vjp
    def _identity_vjp(primals, cotangent, output):
        return (project_frame1_pose_null_nhwc_mlx(cotangent),)

    return _identity(frame)


def apply_seg_grad_q3_project(frame, mode: str):
    """Apply the default-off pg1 seg-gradient projector to a rendered frame."""
    if mode == "off":
        return frame
    if mode == "on":
        return q3_project_seg_gradient_identity(frame)
    raise ValueError(f"seg_grad_q3_project must be off|on, got {mode!r}")


def make_render_fn(seg_grad_q3_project: str = "off"):
    """render_fn for the canonical ``make_loss_fn`` hook:
    (model, coord_feats, code_idx, render_h, render_w) -> R(render) (1,384,512,3)."""
    from experiments.train_witness_realized_through_R_mlx import _apply_R

    if seg_grad_q3_project not in PG1_SEG_GRAD_Q3_MODES:
        raise ValueError(f"seg_grad_q3_project must be off|on, got {seg_grad_q3_project!r}")

    def render_fn(model, coord_feats, code_idx, render_h, render_w):
        frame = _apply_R(model.render_frame(int(code_idx)))
        return apply_seg_grad_q3_project(frame, seg_grad_q3_project)

    return render_fn


def _soft_hist_entropy_bits(vals, levels: int, temp: float = 0.15):
    """§3.4 differentiable marginal soft-histogram entropy (bits/token) of token VALUES.

    ``vals`` (any mx shape, expected in [-1, 1]) are soft-assigned to the ``levels`` lattice
    bins (temperature ``temp`` softmax over squared distance to bin centers) and averaged into
    a probability vector ``p``; the return is Shannon entropy ``-sum p log2 p``. Minimizing it
    CLUMPS the learned token distribution at fewer lattice levels => lower token entropy =>
    fewer coded bytes (the stl1 row-8 rate-in-loss LAW; the explicit form of the §3.3(b)
    redistribution co-benefit). Fully differentiable in ``vals`` (STE-free soft assignment)."""
    import mlx.core as mx

    L = float(levels - 1)
    x01 = mx.clip((vals + 1.0) * 0.5, 0.0, 1.0) * L                 # -> [0, L]
    centers = mx.arange(levels).astype(mx.float32)                  # (levels,)
    d2 = (x01.reshape((-1, 1)) - centers.reshape((1, -1))) ** 2     # (N, levels)
    soft = mx.softmax(-d2 / max(float(temp), 1e-6), axis=-1)        # (N, levels)
    p = mx.mean(soft, axis=0) + 1e-12                               # (levels,)
    return -mx.sum(p * mx.log(p)) / float(np.log(2.0))             # bits/token


# ---------------------------------------------------------------------------
# COUNTED-byte ledger (measured with a real compressor on the real quantized
# payloads; labeled COUNTED-ESTIMATE until the E4/WS1 exporter grammar wires in).
# ---------------------------------------------------------------------------
def quantize_tokens_np(tokens: np.ndarray, levels: int) -> np.ndarray:
    t = np.clip(tokens, -1.0, 1.0)
    return np.round((t + 1.0) * 0.5 * (levels - 1)).astype(np.uint8)


def token_stream_bytes(tokens_np: np.ndarray, levels: int,
                       keep_mask: np.ndarray | None = None) -> int:
    """Temporal-delta (mod 256) + zlib-9 on the quantized token lattice (P,gh,gw,c).

    §3.1: when ``keep_mask`` (gh,gw bool) is given, ONLY the kept cells are coded (the
    coarse-from-birth grid excludes the inactive cells from the token stream — they are
    zero by construction, but the byte-close must not pay their compressed residue)."""
    if keep_mask is not None:
        tokens_np = tokens_np[:, keep_mask, :]  # (P, n_kept, c)
    q = quantize_tokens_np(tokens_np, levels)
    delta = q.copy()
    delta[1:] = (q[1:].astype(np.int16) - q[:-1].astype(np.int16)) % 256
    return len(zlib.compress(delta.astype(np.uint8).tobytes(), 9))


def token_stream_bytes_smevr(full_codes_u8: np.ndarray, levels: int) -> int:
    """QA86(b) / census T5 FIX: price the token field with the SHIPPED coder (SMEVR,
    the landed r7 token coder ``experiments/ddm_r7_token_coder.py``) instead of the
    zlib-temporal-delta surrogate ``token_stream_bytes`` used for stage/telemetry
    decisions. The archive ships SMEVR (r7 race receipt: SMEVR decisively wins;
    ddm_gd1 Hilbert race: the SMEVR-2D control reproduces the stored member bytes
    EXACTLY), so decisions that price in zlib are decision-noise vs shipped bytes.

    ``full_codes_u8`` is the FULL per-frame quantized token field (P,gh,gw,c) uint8 in
    [0,levels): SMEVR does its OWN temporal mode-base + modulo-delta factorization
    internally, and inactive (dropped) cells are 0 by construction so their delta
    stream codes ~free — matching the shipped keep-mask savings without a separate
    kept-cell restriction. Deterministic + lossless (the r7 encode is exact)."""
    from experiments.ddm_r7_token_coder import encode_token_codes

    return len(encode_token_codes(
        np.ascontiguousarray(full_codes_u8), levels=int(levels), codec="smevr"))


def _build_rowband_grammar(cfg: "TR1Config"):
    """QA84 §4.2: build the RowBandGrammar from ``cfg.token_rowband_spec`` (a spec .json path
    OR inline json), validating its fine dims match the (D8) grid (fail-closed never-invent
    geometry). Returns None when unset (uniform grid = control)."""
    spec = getattr(cfg, "token_rowband_spec", None)
    if not spec:
        return None
    from pathlib import Path

    from tac.witness_dsl.qa84_rowband_grammar_20260731 import RowBandGrammar

    text = Path(spec).read_text() if Path(spec).exists() else str(spec)
    g = RowBandGrammar.from_spec_json(text)
    if (g.fine_gh, g.fine_gw) != (cfg.grid_h, cfg.grid_w):
        raise ValueError(
            f"rowband grammar fine dims ({g.fine_gh},{g.fine_gw}) != grid "
            f"({cfg.grid_h},{cfg.grid_w}) at D={cfg.grid_downsample}; row-band needs the FINE "
            f"(D8) base — pass --grid-downsample 8 with a matching spec (fail-closed)")
    return g


def _build_pool_a_banks(cfg: "TR1Config") -> tuple[np.ndarray | None, np.ndarray | None]:
    """ax1 Pool-A (ddm_pa1b #793): build the FIXED (non-trainable) per-cell level map
    (margin-coupled quant) + the xi-informed delta-sparsity weight field from the MEASURED QA80
    flip-distance field custody. At most ONE field load (both levers share it). Returns
    (level_map (gh,gw) int | None, delta_weight (gh,gw) float | None). None,None when both OFF
    => build_module adds no buffers => byte-identical control. Fail-closed if a field-consuming
    lever is on without the custody dir (never-invent)."""
    need_level = cfg.token_quant_margin_coupling == "on"
    need_xi = (cfg.token_delta_group_sparsity == "on"
               and cfg.delta_sparsity_weight_field == "xi_informed")
    if not (need_level or need_xi):
        return None, None
    if cfg.token_delta_group_sparsity == "on" and cfg.token_temporal_mode != "shared_base":
        raise ValueError("delta group-sparsity requires --token-temporal-mode shared_base "
                         "(the per-pair deltas it shrinks only exist in shared_base) — fail-closed")
    if not cfg.token_quant_coupling_field:
        raise ValueError("ax1 Pool-A field levers require --token-quant-coupling-field "
                         "(the QA80 flip-distance custody dir) — fail-closed (never-invent)")
    from tac.witness_dsl.ax1_pool_a_levers_20260730 import (
        load_qa80_cell_field,
        margin_coupled_level_map,
        xi_informed_delta_weight,
    )

    agg = load_qa80_cell_field(cfg.grid_h, cfg.grid_w, downsample=cfg.grid_downsample,
                               field_custody=cfg.token_quant_coupling_field)
    level_map = None
    if need_level:
        base = int(cfg.token_quant_levels)
        min_lv = cfg.token_quant_coupling_min_levels or max(2, base // 4)  # lattice-friendly floor
        level_map = margin_coupled_level_map(agg.flip_mass, base_levels=base, min_levels=min_lv)
    delta_weight = xi_informed_delta_weight(agg.dynamic_frac) if need_xi else None
    return level_map, delta_weight


def _full_token_field_np(model, cfg: "TR1Config") -> np.ndarray:
    """Reconstruct the FULL per-frame token field (P,gh,gw,c) EXACTLY as it renders/ships
    (``raw_tokens`` = tie(cell_mask*(base+delta)) — cell_mask then QA84 row-band tie)."""
    keep3 = np.asarray(model._cell_mask.tensors["keep"], dtype=np.float32)  # (gh,gw,1)
    if cfg.token_temporal_mode == "shared_base":
        base_np = np.asarray(model.tokens_base, dtype=np.float32)[None]      # (1,gh,gw,c)
        delta_np = np.asarray(model.tokens_delta, dtype=np.float32)          # (P,gh,gw,c)
        field = (base_np + delta_np) * keep3
    else:
        field = np.asarray(model.tokens, dtype=np.float32) * keep3
    grammar = getattr(model, "_rowband", None)
    if grammar is not None:  # QA84: byte-close must see the SAME tied field the renderer ships
        field = grammar.apply_tie_np(field)
    return field


def _int8_tensor_bytes(w: np.ndarray) -> bytes:
    scale = float(np.max(np.abs(w))) / 127.0 if np.max(np.abs(w)) > 0 else 1.0
    q = np.clip(np.round(w / scale), -127, 127).astype(np.int8)
    return q.tobytes() + np.float16(scale).tobytes()


def selector_ledger_blob(cfg: TR1Config) -> bytes:
    """rule-118 selector accounting (eu1 flag, adjudicated in the tb1 memo): every
    decoder-visible VIDEO-SELECTED choice is COUNTED for BOTH variants — arch topology
    id, grid geometry, code width, renderer width, quant levels, STE mode + dither
    seed, temporal mode, and (lotto) PRNG seed + mask density. The PRNG/dither
    EXPANSION is FREE generic code; the SELECTION is what the archive pays for."""
    sel: dict[str, Any] = {
        "arch": f"tr1_{cfg.variant}_v1",
        "grid_downsample": cfg.grid_downsample, "code_width": cfg.code_width,
        "renderer_width": cfg.renderer_width,
        "token_quant_levels": cfg.token_quant_levels,
        "token_ste": cfg.token_ste, "dither_seed": cfg.seed + 7,
        "token_temporal_mode": cfg.token_temporal_mode,
    }
    if cfg.variant == "lotto":
        sel["lotto_seed"] = cfg.lotto_seed
        sel["mask_density_init"] = cfg.lotto_mask_density_init
    return json.dumps(sel, sort_keys=True, separators=(",", ":")).encode()


def _token_bytes_zlib(model, cfg: "TR1Config") -> int:
    """Legacy zlib temporal-delta token price (census T5 pre-fix control / fast path)."""
    keep = np.asarray(model._cell_mask.tensors["keep"], dtype=np.float32)[..., 0] > 0.5
    keep_arg = keep if not keep.all() else None  # None => uniform grid (no selection needed)
    if cfg.token_temporal_mode == "shared_base":
        base_np = np.asarray(model.tokens_base, dtype=np.float32)[None]
        delta_np = np.asarray(model.tokens_delta, dtype=np.float32)
        base_code = (base_np[:, keep, :] if keep_arg is not None else base_np)
        # base coded once; the per-frame delta stream rides the same lattice.
        return (len(zlib.compress(quantize_tokens_np(base_code, cfg.token_quant_levels).tobytes(), 9))
                + token_stream_bytes(delta_np, cfg.token_quant_levels, keep_mask=keep_arg))
    return token_stream_bytes(np.asarray(model.tokens, dtype=np.float32),
                              cfg.token_quant_levels, keep_mask=keep_arg)


def counted_bytes_ledger(model, cfg: TR1Config) -> dict[str, Any]:
    """Per-stream COUNTED bytes for the current EMA/live params (rule-118 boundary):
    tokens + (plain: int8 weights | lotto: mask+modulations+biases) + selector.

    QA86(b) / census T5: the token stream is priced with ``cfg.byte_ledger_coder``
    ("smevr" default = the SHIPPED r7 coder; "zlib" = the legacy temporal-delta
    surrogate). The zlib price is ALSO recorded (cheap) for decomposable observability
    (max-observability non-negotiable); ``total_counted_bytes`` sums ONLY the FOUR real
    streams — tokens + renderer + selector + rowband spec (QA84 §4.2) — never the
    observability keys. The counted-vs-observability split is pinned by
    ``src/tac/tests/test_ddm_tb1_tr1_renderer.py`` (key classification; an unclassified
    new key fails) and ``test_ddm_b2b_burn2_composition.py`` (nonzero rowband term)."""
    coder = getattr(cfg, "byte_ledger_coder", "smevr")
    tok_b_zlib = _token_bytes_zlib(model, cfg)
    if coder == "smevr":
        full = _full_token_field_np(model, cfg)
        q = quantize_tokens_np(full, cfg.token_quant_levels)
        try:
            tok_b = token_stream_bytes_smevr(q, cfg.token_quant_levels)
        except Exception:  # r7 coder unavailable => fall back to zlib (never crash a gate)
            tok_b, coder = tok_b_zlib, "zlib_fallback"
    else:
        tok_b = tok_b_zlib
    ledger: dict[str, Any] = {"tokens_bytes": int(tok_b)}
    obs: dict[str, int | str] = {"tokens_bytes_zlib": int(tok_b_zlib),
                                 "token_ledger_coder": coder}
    if coder == "smevr":  # SMEVR succeeded (fallback relabels to zlib_fallback => omit)
        obs["tokens_bytes_smevr"] = int(tok_b)
    shapes = _conv_shapes(cfg)
    if cfg.variant == "plain":
        blob = b"".join(
            _int8_tensor_bytes(np.asarray(getattr(model, f"w_{n}"), dtype=np.float32))
            for n, _ in shapes)
        blob += b"".join(
            _int8_tensor_bytes(np.asarray(getattr(model, f"b_{n}"), dtype=np.float32))
            for n, _ in shapes)
        ledger["renderer_bytes"] = len(zlib.compress(blob, 9))
    else:
        mask_bits = np.concatenate([
            (np.asarray(getattr(model, f"s_{n}"), dtype=np.float32) > 0).astype(np.uint8).ravel()
            for n, _ in shapes])
        mask_blob = zlib.compress(np.packbits(mask_bits).tobytes(), 9)
        mods = b"".join(
            np.asarray(getattr(model, f"g_{n}"), dtype=np.float16).tobytes() +
            np.asarray(getattr(model, f"b_{n}"), dtype=np.float16).tobytes()
            for n, _ in shapes)
        ledger["renderer_bytes"] = len(mask_blob) + len(mods)
    ledger["selector_ledger_bytes"] = len(selector_ledger_blob(cfg))
    # QA84 §4.2: the row-band grammar spec is COUNTED decoder side-info (few bytes).
    grammar = getattr(model, "_rowband", None)
    ledger["rowband_spec_bytes"] = int(grammar.band_spec_bytes()) if grammar is not None else 0
    # total sums ONLY the four real streams (obs keys are merged AFTER, never summed).
    ledger["total_counted_bytes"] = int(
        ledger["tokens_bytes"] + ledger["renderer_bytes"]
        + ledger["selector_ledger_bytes"] + ledger["rowband_spec_bytes"])
    ledger.update(obs)  # decomposable observability (excluded from the total by construction)
    return ledger


# ---------------------------------------------------------------------------
# A1 realized gate (fd2 lesson): render fp32 on the MLX CPU stream (the MLX-GPU
# forward is reduced-precision — the witness PORT-FIDELITY lesson), lift to
# camera uint8 with the TORCH-authority R, frozen CPU SegNet argmax.
# ---------------------------------------------------------------------------
def topology_per_class(realized: np.ndarray, gts: list[np.ndarray]) -> dict[str, list[int]]:
    """A1 anti-aliasing telemetry (steer #3): SegNet sees REGIONS and the measured
    error mode is ERASURE of low-persistence components (lane dashes; error ~
    1/persistence) — equal flip counts can hide wrong component structure. Per class:
    Betti-0 (connected components) realized vs GT, GT components ERASED (zero realized
    overlap), and the smallest SURVIVING GT component (px) — erasure/birth failures
    become visible the moment they happen. Score-neutral, default-on."""
    from scipy import ndimage

    b0_r = [0] * 5
    b0_g = [0] * 5
    erased = [0] * 5
    min_surv = [0] * 5  # 0 = none survived / class absent
    for i in range(realized.shape[0]):
        for c in range(5):
            _, nr = ndimage.label(realized[i] == c)
            lg, ng = ndimage.label(gts[i] == c)
            b0_r[c] += int(nr)
            b0_g[c] += int(ng)
            for comp in range(1, ng + 1):
                m = lg == comp
                if not np.any(realized[i][m] == c):
                    erased[c] += 1
                else:
                    sz = int(np.count_nonzero(m))
                    if min_surv[c] == 0 or sz < min_surv[c]:
                        min_surv[c] = sz
    return {"betti0_realized": b0_r, "betti0_gt": b0_g,
            "gt_components_erased": erased, "smallest_surviving_gt_component_px": min_surv}


# ---------------------------------------------------------------------------
# ddm_bs3 (#909) — the FULL-SCOPE / WRONG-PROJECTION cures for the two gate
# scalars.  Both statistics below already ran over the FULL gate set; each was
# structurally unable to register the defect it is trusted for, because it is a
# contraction whose KERNEL is occupied on real data:
#
#   ``realized_gate_dseg_mean``      = mean over pairs of a per-pixel error RATE.
#       Kernel: every redistribution of the SAME total error mass across CLASSES.
#       The campaign's binding structure is per-class (lane erasure, the Undriv
#       watch, the per-class floors), so that kernel is exactly where the live
#       defects sit.  ``topology_per_class`` is per-class TOPOLOGY (component
#       counts / erasures), never per-class error MASS -- it does not close this.
#   ``realized_flips_vs_prev_gate``  = count of pixels with ``realized != prev``.
#       Kernel: the SIGN of every flip.  wrong->right and right->wrong both add 1.
#       MEASURED on the burn-4 gate series (61 consecutive gate pairs, real
#       n600-cache GT, 36-pair gate, 7,077,888 px compared per gate): the NET
#       error-pixel movement is a MEDIAN 5.4% of the counted flips -- i.e. ~94.6%
#       of what this counter reports cancels, and the counter cannot say so.
#
# Both cures are EXACT PARTITIONS of the blind scalar (not new proxies), pure
# numpy, score-neutral, default-on read-only telemetry (the "observability that
# cannot change the bytes defaults ON" rule).  Cost is negligible against the
# SegNet forward the gate already paid for.
# ---------------------------------------------------------------------------
# Every score-neutral detailed gate telemetry key. These are telemetry.jsonl-ONLY:
# they are stripped before the row enters ``telemetry_tail`` (baked into the
# checkpoint meta), so these observability landings leave checkpoint bytes
# untouched. Pinned against the producing functions by
# ``test_ddm_bs3_gate_projection_kernel`` so a future field cannot silently leak
# into the checkpoint.
BS3_TELEMETRY_ONLY_KEYS = frozenset({
    "realized_gate_pair_ids",
    "realized_gate_dseg_per_pair",
    "realized_gate_dseg_mean_ht",
    "realized_gate_dseg_mean_ht_design",
    "realized_gate_dseg_per_pair_sd",
    "realized_gate_dseg_per_pair_q50",
    "realized_gate_dseg_per_pair_q90",
    "realized_gate_dseg_per_pair_q95",
    "realized_gate_dseg_per_pair_gt_2x_mean_n",
    "realized_gate_dseg_by_gt_class",
    "realized_flips_toward_gt",
    "realized_flips_away_from_gt",
    "realized_flips_lateral",
    "realized_flips_net_error_px",
    "realized_class_l1_rel_since_prev_gate",
    "realized_mean_hid_class_motion",
    "realized_gate_dpose_per_pair",
    "realized_gate_dpose_mean",
    "realized_gate_dpose_per_pair_max",
    "realized_gate_dpose_per_pair_sd",
    "realized_gate_dpose_per_pair_q50",
    "realized_gate_dpose_per_pair_q90",
    "realized_gate_dpose_per_pair_q95",
    "realized_gate_dpose_wall_seconds",
    "realized_gate_dpose_axis",
    "realized_gate_dpose_label",
    "realized_gate_dpose_semantics",
    "realized_gate_dpose_gate36_n600_calibration",
})


def checkpoint_safe_telemetry_row(row: dict[str, Any]) -> dict[str, Any]:
    """Any telemetry row MINUS the ddm_bs3 telemetry-only fields.

    ``telemetry_tail`` is baked into the checkpoint meta, so anything appended to
    it changes checkpoint bytes. The bs3 decomposition fields are pure read-only
    observability and go to ``telemetry.jsonl`` via ``tlog`` only -- exactly the
    rule the v9 telemetry port already follows. Kept as a named function (rather
    than an inline comprehension) so a test can pin BOTH that the strip list is
    complete AND that EVERY call site actually applies it.  Applied to the epoch
    row too (where it is a no-op) so the invariant is TOTAL -- 'nothing entering
    telemetry_tail carries a bs3 key' -- rather than true of one call site and
    unchecked at the other."""
    return {k: v for k, v in row.items() if k not in BS3_TELEMETRY_ONLY_KEYS}


def dseg_by_gt_class(realized: np.ndarray, gts: list[np.ndarray],
                     n_classes: int = 5) -> list[float]:
    """EXACT partition of the gate's mean d_seg by GROUND-TRUTH class.

    Every error pixel has exactly one GT class, so
    ``sum(dseg_by_gt_class(...)) == mean(d_seg)`` identically (to fp rounding).
    Element ``c`` is the share of the gate's realized d_seg owed to pixels whose
    GT is class ``c`` -- the per-class error MASS the mean contracts away.

    VACUITY: refuses an empty gate set rather than returning zeros, which would
    be indistinguishable from a perfect gate."""
    if realized.shape[0] == 0 or not gts:
        raise ValueError("dseg_by_gt_class: empty gate set is VACUOUS, never 0.0")
    if realized.shape[0] != len(gts):
        raise ValueError(
            f"dseg_by_gt_class: {realized.shape[0]} realized maps vs {len(gts)} GT maps")
    n = len(gts)
    out = [0.0] * n_classes
    for i, g in enumerate(gts):
        # (round-1 self-review) A GT label >= n_classes would be silently dropped
        # from every bucket, so the partition would under-sum the very scalar it
        # claims to decompose -- a wrong-projection defect introduced BY the cure.
        # Refuse instead.
        if g.size and int(np.max(g)) >= n_classes:
            raise ValueError(
                f"dseg_by_gt_class: GT label {int(np.max(g))} >= n_classes={n_classes}; "
                "the partition would silently under-sum the mean it decomposes")
        wrong = realized[i] != g
        for c in range(n_classes):
            out[c] += float(np.count_nonzero(wrong & (g == c))) / g.size
    return [v / n for v in out]


def gd1_realized_gate_dseg_fields(
    gate_ids: Sequence[int],
    dsegs: Sequence[float],
    n_population: int,
) -> dict[str, Any]:
    """GD1 A1 repair fields: old mean plus per-pair and HT telemetry.

    The legacy ``realized_gate_dseg_mean`` remains the exact unweighted mean so
    current consumers and historical series stay comparable.  The repaired
    statistic is emitted under a new key, ``realized_gate_dseg_mean_ht``.
    """
    ids = tuple(int(i) for i in gate_ids)
    vals = np.asarray(dsegs, dtype=np.float64)
    if vals.ndim != 1:
        raise ValueError(f"gd1_realized_gate_dseg_fields: expected 1D dsegs, got {vals.shape}")
    if len(ids) != vals.shape[0]:
        raise ValueError(
            f"gd1_realized_gate_dseg_fields: {len(ids)} gate ids vs {vals.shape[0]} dsegs")
    if vals.shape[0] == 0:
        raise ValueError("gd1_realized_gate_dseg_fields: empty gate set is VACUOUS")
    mean = float(np.mean(vals))
    ht_design = "legacy_mean_noncanonical"
    ht_mean = mean
    if len(ids) == int(n_population):
        ht_design = "full_population_exact_mean"
    elif int(n_population) >= (max(GATE_BLOCK_PAIRS) + 1):
        block_set = set(GATE_BLOCK_PAIRS)
        id_set = set(ids)
        srs_ids = tuple(i for i in ids if i not in block_set)
        if (
            block_set.issubset(id_set)
            and len(srs_ids) == GATE_OFFBLOCK_SAMPLE
            and len(ids) == len(GATE_BLOCK_PAIRS) + GATE_OFFBLOCK_SAMPLE
        ):
            design = GateDesign(
                n_population=int(n_population),
                block_ids=GATE_BLOCK_PAIRS,
                srs_ids=srs_ids,
            )
            gate_values = {i: float(v) for i, v in zip(ids, vals, strict=True)}
            ht_mean = horvitz_thompson_mean(design, gate_values)
            ht_design = "gd1_block_plus_srs_horvitz_thompson"
    return {
        "realized_gate_pair_ids": list(ids),
        "realized_gate_dseg_per_pair": [float(v) for v in vals],
        "realized_gate_dseg_mean": mean,
        "realized_gate_dseg_mean_ht": float(ht_mean),
        "realized_gate_dseg_mean_ht_design": ht_design,
        "realized_gate_dseg_per_pair_max": float(np.max(vals)),
        "realized_gate_dseg_per_pair_sd": (
            float(np.std(vals, ddof=1)) if vals.shape[0] > 1 else None),
        "realized_gate_dseg_per_pair_q50": float(np.quantile(vals, 0.50)),
        "realized_gate_dseg_per_pair_q90": float(np.quantile(vals, 0.90)),
        "realized_gate_dseg_per_pair_q95": float(np.quantile(vals, 0.95)),
        "realized_gate_dseg_per_pair_gt_2x_mean_n": int(
            np.count_nonzero(vals > (2.0 * mean)) if mean > 0.0
            else np.count_nonzero(vals > 0.0)),
    }


def realized_gate_dpose_fields(
    gate_ids: Sequence[int],
    dposes: Sequence[float],
    *,
    wall_seconds: float,
) -> dict[str, Any]:
    """POSE-denominated A1 gate telemetry over the same pair set as d_seg.

    This is an advisory trend channel only: the n600 endpoint probe remains the
    boundary authority. The values are per-pair first-6 PoseNet MSEs after the
    same MLX ``_apply_R`` + yuv12 path used by
    ``experiments/ddm_jd4_endpoint_n600_both_bases.py``.
    """
    ids = tuple(int(i) for i in gate_ids)
    vals = np.asarray(dposes, dtype=np.float64)
    if vals.ndim != 1:
        raise ValueError(f"realized_gate_dpose_fields: expected 1D dposes, got {vals.shape}")
    if len(ids) != vals.shape[0]:
        raise ValueError(
            f"realized_gate_dpose_fields: {len(ids)} gate ids vs {vals.shape[0]} dposes")
    if vals.shape[0] == 0:
        raise ValueError("realized_gate_dpose_fields: empty gate set is VACUOUS")
    return {
        "realized_gate_dpose_per_pair": [float(v) for v in vals],
        "realized_gate_dpose_mean": float(np.mean(vals)),
        "realized_gate_dpose_per_pair_max": float(np.max(vals)),
        "realized_gate_dpose_per_pair_sd": (
            float(np.std(vals, ddof=1)) if vals.shape[0] > 1 else None),
        "realized_gate_dpose_per_pair_q50": float(np.quantile(vals, 0.50)),
        "realized_gate_dpose_per_pair_q90": float(np.quantile(vals, 0.90)),
        "realized_gate_dpose_per_pair_q95": float(np.quantile(vals, 0.95)),
        "realized_gate_dpose_wall_seconds": float(wall_seconds),
        "realized_gate_dpose_axis": "[macOS-CPU frozen-scorer advisory]",
        "realized_gate_dpose_label": "advisory_trend_channel_n600_probe_authority",
        "realized_gate_dpose_semantics": (
            "PoseNet first-6 MSE vs gt_poses[idx][:6] on "
            "(_apply_R(render(max(idx-1,0))), _apply_R(render(idx))) yuv12"),
        "realized_gate_dpose_gate36_n600_calibration": "banked jd4/jd5/jd6 ratio 1.002-1.146",
    }


def realized_gate_pose_yuv12(f0: Any, f1: Any) -> Any:
    """Endpoint-probe yuv12 packing for a PoseNet pair."""
    import mlx.core as mx

    from tac.local_acceleration.pr95_hnerv_mlx_training import rgb_to_yuv6_mlx

    pair = mx.stack([f0[0], f1[0]], axis=0)[None]
    yuv = rgb_to_yuv6_mlx(pair)
    b, t, h2, w2, c6 = yuv.shape
    return mx.reshape(mx.transpose(yuv, (0, 2, 3, 1, 4)), (b, h2, w2, t * c6))


def realized_gate_dposes(model: Any, gate_ids: Sequence[int], gt_poses: Any,
                         pose_adapter: Any) -> list[float]:
    """Read-only gate d_pose pass, outside gradient and training state updates."""
    import mlx.core as mx

    from experiments.train_witness_realized_through_R_mlx import _apply_R

    out: list[float] = []
    with mx.stream(mx.cpu):
        for idx in gate_ids:
            i = int(idx)
            f1 = _apply_R(model.render_frame(i))
            f0 = _apply_R(model.render_frame(max(i - 1, 0)))
            pose_out = pose_adapter.posenet(realized_gate_pose_yuv12(f0, f1))
            pose = pose_out["pose"] if isinstance(pose_out, dict) else pose_out
            mx.eval(pose)
            p6 = np.asarray(pose, dtype=np.float64).ravel()[:6]
            tgt = np.asarray(gt_poses[i], dtype=np.float64).ravel()[:6]
            out.append(float(np.mean((p6 - tgt) ** 2)))
    return out


def flip_direction_counts(realized: np.ndarray, prev_realized: np.ndarray,
                          gts: list[np.ndarray]) -> dict[str, int]:
    """SIGN-RESOLVE ``realized_flips_vs_prev_gate`` into an exact 3-way partition.

    A pixel that changed between gates is exactly one of:
      ``toward_gt``  was wrong, now right      (improvement)
      ``away_from_gt`` was right, now wrong    (regression)
      ``lateral``    wrong before and after, but a DIFFERENT wrong class
    (the fourth case -- right before and after -- forces ``prev == realized`` and
    so is not a flip at all).  Therefore
      ``toward + away + lateral == count(realized != prev)`` and
      ``away - toward == `` the exact change in the gate's error-pixel count,
    which is the pixel-unit d_seg movement.  The bare count is the sum of three
    terms that routinely cancel; this reports them separately."""
    if realized.shape[0] == 0 or not gts:
        raise ValueError("flip_direction_counts: empty gate set is VACUOUS, never 0")
    if realized.shape != prev_realized.shape:
        raise ValueError(
            f"flip_direction_counts: shape {realized.shape} vs prev {prev_realized.shape}")
    # (round-1 self-review) The loop is over ``gts``; a short ``gts`` would silently
    # count fewer pairs than the incumbent ``count(realized != prev)`` covers, so the
    # 3-way partition would no longer sum to it -- and nothing would say so.
    if len(gts) != realized.shape[0]:
        raise ValueError(
            f"flip_direction_counts: {len(gts)} GT maps vs {realized.shape[0]} realized "
            "-- the partition would not cover the incumbent count's denominator")
    toward = away = lateral = 0
    for i, g in enumerate(gts):
        err_prev = prev_realized[i] != g
        err_cur = realized[i] != g
        changed = realized[i] != prev_realized[i]
        toward += int(np.count_nonzero(changed & err_prev & ~err_cur))
        away += int(np.count_nonzero(changed & ~err_prev & err_cur))
        lateral += int(np.count_nonzero(changed & err_prev & err_cur))
    return {"realized_flips_toward_gt": toward,
            "realized_flips_away_from_gt": away,
            "realized_flips_lateral": lateral,
            "realized_flips_net_error_px": away - toward}


def realized_gate(model, gate_ids: tuple[int, ...], lstars, seg_cpu,
                  prev_realized: np.ndarray | None, *,
                  pose_adapter: Any | None = None,
                  gt_poses: Any | None = None) -> dict[str, Any]:
    import mlx.core as mx

    from experiments.train_witness_realized_through_R_mlx import (
        _torch_R_to_camera_uint8,
        cpu_verdict_d_seg_argmax_batch,
    )

    t0 = time.monotonic()
    frames: list[np.ndarray] = []
    with mx.stream(mx.cpu):
        for i in gate_ids:
            rgb = model.render_frame(int(i))
            mx.eval(rgb)
            frames.append(np.asarray(rgb, dtype=np.float32)[0])
    cams = [_torch_R_to_camera_uint8(f) for f in frames]
    gts = [np.asarray(lstars[i], dtype=np.int64) for i in gate_ids]
    dsegs, realized = cpu_verdict_d_seg_argmax_batch(seg_cpu, cams, gts)
    realized = np.asarray(realized)
    row: dict[str, Any] = {
        "gate_ids_n": len(gate_ids),
        **gd1_realized_gate_dseg_fields(gate_ids, dsegs, len(lstars)),
        # ddm_bs3 (#909): EXACT per-GT-class partition of the mean -- the kernel
        # the scalar contracts away, and the axis the campaign actually watches.
        "realized_gate_dseg_by_gt_class": dseg_by_gt_class(realized, gts),
        "gate_render_stream": "mlx_cpu_fp32",
        "gate_wall_seconds": time.monotonic() - t0,
    }
    if (pose_adapter is None) != (gt_poses is None):
        raise ValueError("realized_gate d_pose channel requires both pose_adapter and gt_poses")
    if pose_adapter is not None and gt_poses is not None:
        pose_t0 = time.monotonic()
        dposes = realized_gate_dposes(model, gate_ids, gt_poses, pose_adapter)
        row.update(realized_gate_dpose_fields(
            gate_ids, dposes, wall_seconds=time.monotonic() - pose_t0))
        row["gate_wall_seconds"] = time.monotonic() - t0
    if prev_realized is not None and prev_realized.shape == realized.shape:
        row["realized_flips_vs_prev_gate"] = int(np.count_nonzero(realized != prev_realized))
        # ddm_bs3 (#909): sign-resolve that count. MEASURED on burn-4, ~94.6% of
        # the flips it reports cancel; the bare magnitude cannot distinguish a
        # gate that improved from one that regressed by the same amount.
        row.update(flip_direction_counts(realized, prev_realized, gts))
    row["topology_per_class"] = topology_per_class(realized, gts)
    row["_realized_argmax"] = realized
    return row


BASIN_THRESHOLDS = {
    "smooth_rel_per_window": 0.01, "dseg_rel_per_window": 0.02,
    "lane_b0_delta_max": 2, "lane_erased_delta_max": 1,
}

# ddm_tp2 (2026-08-02) — the basin predicate keys on the LOSS FORM, never on the DISPLAY NAME.
# MEASURED BUG (fixed here): the predicate compared ``row["stage"] == "seg_trunk_tau"`` by exact
# string, but ``stage`` is a display label whose text VARIES WITH A LAUNCH FLAG:
#   ``--seg-form-start ce``          -> "seg_trunk_ce", then the knee event sets "seg_trunk_tau"  [fires]
#   ``--seg-form-start tau_softplus``-> "seg_trunk_tau_softplus" FOREVER                          [NEVER fires]
# and in the second case ``knee_switched = stage != "seg_trunk_ce"`` is True from epoch 0, so BOTH
# form-switch blocks are guarded off and the label can never be rewritten. Such a run trains the
# IDENTICAL tau_softplus loss and is structurally incapable of entering the basin. The semantic the
# predicate actually wants is the FORM the loss is running (``state_form["form"]``, the state machine
# the switch events mutate) -- that value is "tau_softplus" in BOTH launch paths.
BASIN_TERMINAL_SEG_FORM = "tau_softplus"
SEG_TRUNK_CE_STAGE = "seg_trunk_ce"


def initial_stage_label(seg_form_start: str) -> str:
    """The DISPLAY label for the opening stage. Sole owner of the label convention.

    NOTE (ddm_tp2): this is a LABEL, not a semantic. It is NOT stable under
    ``--seg-form-start``: "ce" -> "seg_trunk_ce" (rewritten to "seg_trunk_tau" by the knee
    event) but "tau_softplus" -> "seg_trunk_tau_softplus", which no event ever rewrites.
    Never key a predicate on it; key on ``state_form["form"]`` (see BASIN_TERMINAL_SEG_FORM).
    """
    return (SEG_TRUNK_CE_STAGE if seg_form_start == "ce"
            else f"seg_trunk_{seg_form_start}")


# ---- ddm_en1 (2026-08-05) row 1: --margin-weighted-loss now reaches tau_softplus ----------
# ddm_tp2 MEASURED the bug: ``--margin-weighted-loss on`` was threaded into make_loss_fn for
# EVERY form (it sets ``apply_mw``), but only three of the five branches ever READ it. In
# ``experiments/train_witness_realized_through_R_mlx.py::make_loss_fn.loss_fn`` the historical
# ``margin_hinge`` / ``unify_tau`` / ``ce`` branches guarded on ``if apply_mw:``; EN1 adds the
# same real consumer to ``tau_softplus``. ``l7_softplus`` remains deliberately outside this
# lever because that stage carries its OWN hard-pixel weight.
#
# MEASURED blast radius: all four TR1 windows (ddm_b4s_20260731/window_01..03,
# ddm_r1c_20260731/window_01) launched margin_weighted_loss='on' and ran the tau_softplus
# form for 100% of their trained epochs -> the flag had ZERO effect on every epoch of the
# burn lineage. window_03 alone: 199/199 gate rows at form tau_softplus, ep807-945.
#
# POST-FIX CONTRACT: the existing DSL lever ``tr1_seg_margin_weight`` is now a real tau/ce
# training-force candidate, default OFF and RACED, never retroactively asserted for old windows.
# A run that starts at ce remains allowed because the knee transition ce -> tau_softplus keeps
# honoring the same lever. The guard still refuses any reachable form that does not consume it.
MARGIN_WEIGHTED_HONORING_SEG_FORMS = frozenset(
    {"ce", "tau_softplus", "unify_tau", "margin_hinge"})

# ---- ddm_tp2 (2026-08-02) row 3: #274 spike/coherent PRODUCER, ported onto the live vehicle ----
# This is a PORT, not new machinery. The lever (--seg-spike-reweight / --seg-spike-downweight /
# --seg-coherent-upweight) is BUILT at
# ``experiments/train_levelset_witness_realized_through_R_mlx.py:9467-9494`` with a DSL Lever
# (witness_dsl/curriculum_dsl.py:4477) and a gauge default (witness_dsl/gauge.py:1043), and its
# activation-ledger row reads "ever_fired": false. Its PRODUCER had no counterpart on TR1; the
# CONSUMER already does (``seg_pixel_w``, built in pair_loss from class_weight_lane + lane_guard
# and multiplied into the per-pixel seg map before the mean by every seg form). So only the
# producer moves.
#
# THE FIELD: a pixel is a SPIKE at pair pi when its GT argmax differs from BOTH temporal
# neighbours (single-frame flicker a per-frame witness structurally cannot fit); COHERENT when it
# is temporally unstable but still matches >= 1 neighbour (the winnable boundary residual).
# Endpoints (pi in {0, P-1}) have one neighbour and are left neutral -- hence 598 interior pairs
# at n600, matching the ddm_ti1 measurement scope exactly.
#
# WHY IT IS ADMISSIBLE (ddm_ti1, MEASURED n600 / 598 interior pairs / 117,571,584 px / ZERO scorer
# forwards; receipt .omx/research/ddm_ti1_nonredundancy_probe_receipt_bins40_20260802.json):
# TR1's per-pixel seg loss is PER-PAIR SEPARABLE -- for pair t it reads only lstars[t], margins[t]
# and the student's own logits -- so every weight it can express is measurable w.r.t.
# sigma(class, GT margin). A CROSS-PAIR field built from lstars[t +/- 1] is outside that
# sigma-algebra BY CONSTRUCTION. That is the non-redundancy frontier, and it is why a per-pair
# teacher (lr1) was structurally doomed. MEASURED: 79% (spike) / 77% (coherent) of the field's
# variance is unexplained by sigma(class, GT margin), carrying stratified Mantel-Haenszel
# flip-risk lift 1.757 / 1.299 against a hash-noise null of 1.02 and a pair-shuffled control of
# 1.02 / 1.00.
SEG_SPIKE_MH_LIFT_N600 = 1.7573955039587674       # MEASURED (ti1 bins40 receipt, 40 margin bins)
SEG_COHERENT_MH_LIFT_N600 = 1.2988561739557636    # MEASURED (same receipt)

# ASYMMETRIC PRICING (ddm_ti1 §3a). Nothing in the existing #274 record priced these apart: the
# gauge carries a single SEG_SPIKE_DOWNWEIGHT_DEFAULT and both trainer flags default to a
# symmetric inert 1.0. They are NOT symmetric, and they are priced by DIFFERENT rules:
#   * COHERENT is RISK-PROPORTIONAL. The lever's job is to align per-pixel loss weight with
#     conditional flip risk, so the winnable set takes its own MEASURED stratified lift directly
#     as the up-weight. The number IS the measurement; no arithmetic is invented on top of it.
#   * SPIKE is CONCESSION-PRICED, deliberately NOT risk-proportional. Its lift is HIGHER (1.757),
#     but ~88.6% of that set is irreducible single-frame appearance change (#274's own measured
#     note), where smooth is optimal -- so it is conceded at the inherited 0.25 rather than
#     up-weighted to 1.757. Chasing it would spend capacity on error the witness cannot remove.
# These are RACE STARTING VALUES, not defaults: both flags default to 1.0 (inert) so the port is
# byte-identical, and ti1 §7 requires the two scalars be raced SEPARATELY, never moved together.
SEG_SPIKE_DOWNWEIGHT_RACE_START = 0.25                        # conceded (inherited gauge value)
SEG_COHERENT_UPWEIGHT_RACE_START = SEG_COHERENT_MH_LIFT_N600  # risk-proportional (DERIVED)

# uint8 code lattice for the per-pair field. Stored as codes (P,H,W)=118 MB at n600 rather than
# float maps (472 MB): the field has exactly three values, the scalars are applied through a
# 3-entry LUT at use time, and the memory ceiling is shared with the gate/GT caches.
SPIKE_CODE_STABLE, SPIKE_CODE_COHERENT, SPIKE_CODE_SPIKE = 0, 1, 2


def build_spike_coherent_codes(lstars, num_pairs: int) -> np.ndarray:
    """#274 producer: (P,H,W) uint8 temporal-instability codes from the GT argmax.

    Scorer-free and theta-INDEPENDENT (it reads only frozen GT), so it is computed once
    before training and never again. Endpoint pairs stay STABLE (one neighbour only).
    """
    prev = np.asarray(lstars[0], dtype=np.int64)
    cur = np.asarray(lstars[1], dtype=np.int64) if num_pairs > 1 else None
    codes = np.zeros((num_pairs, *prev.shape), dtype=np.uint8)
    for pi in range(1, num_pairs - 1):
        nxt = np.asarray(lstars[pi + 1], dtype=np.int64)
        dp, dn = (cur != prev), (cur != nxt)
        spike = dp & dn                 # differs from BOTH neighbours = unfittable flicker
        coherent = (dp | dn) & (~spike)  # unstable but matches >=1 = winnable boundary
        codes[pi][coherent] = SPIKE_CODE_COHERENT
        codes[pi][spike] = SPIKE_CODE_SPIKE
        prev, cur = cur, nxt
    return codes


def spike_weight_lut(downweight: float, upweight: float) -> np.ndarray:
    """3-entry LUT indexed by the codes above. 1.0/1.0 => all-ones => byte-identical."""
    return np.array([1.0, float(upweight), float(downweight)], dtype=np.float32)


def assert_spike_scalars_have_their_gate(seg_spike_reweight: bool, downweight: float,
                                         upweight: float) -> None:
    """A value flag with no gate is a SILENT no-op -- the same declared-but-inert genus as row 2
    (and the exact trap gauge.py documents for --seg-spike-downweight). Fail closed."""
    if seg_spike_reweight:
        return
    set_without_gate = [n for n, v in (("--seg-spike-downweight", downweight),
                                       ("--seg-coherent-upweight", upweight)) if v != 1.0]
    if set_without_gate:
        raise SystemExit(
            f"REFUSED: {set_without_gate} set without --seg-spike-reweight, which gates the "
            f"producer -- the value would be silently ignored (no weight map is ever built).\n"
            f"  Pass --seg-spike-reweight to arm the lever, or drop the value flag(s).\n"
            f"  Race starting values (ddm_ti1 §3a, priced ASYMMETRICALLY and raced SEPARATELY): "
            f"--seg-spike-downweight {SEG_SPIKE_DOWNWEIGHT_RACE_START} (concession-priced) | "
            f"--seg-coherent-upweight {SEG_COHERENT_UPWEIGHT_RACE_START:.6f} "
            f"(risk-proportional = its MEASURED lift).")


def reachable_seg_forms(seg_form_start: str) -> frozenset[str]:
    """Every loss form a run launched at ``seg_form_start`` can occupy.

    Only ``ce`` has an outgoing transition: the knee event (and its F2 midpoint FALLBACK,
    which makes the switch unconditional) moves ce -> tau_softplus. Every other start is
    terminal, because ``knee_switched`` is True from epoch 0 for a non-ce label. This is why
    the check below is on REACHABLE forms and not merely on the START form: a run launched at
    ``ce`` with the flag on is honored at first and then goes silently inert at the knee --
    which is exactly what happened to the whole b4s burn lineage.
    """
    return frozenset({seg_form_start} | ({"tau_softplus"} if seg_form_start == "ce" else set()))


def assert_margin_weighted_loss_is_honored(seg_form_start: str, margin_weighted_loss: str) -> None:
    """Fail closed when ``--margin-weighted-loss on`` would be inert for any reachable form."""
    if margin_weighted_loss != "on":
        return
    inert = sorted(reachable_seg_forms(seg_form_start) - MARGIN_WEIGHTED_HONORING_SEG_FORMS)
    if not inert:
        return
    raise SystemExit(
        f"REFUSED: --margin-weighted-loss on is INERT for seg form(s) {inert} that a run "
        f"launched at --seg-form-start {seg_form_start!r} will occupy.\n"
        f"  Forms that honor it: {sorted(MARGIN_WEIGHTED_HONORING_SEG_FORMS)} (they guard on "
        f"`if apply_mw:`).\n"
        f"  Forms that ignore it: ['l7_softplus'] (deliberately carries its own hard-pixel "
        f"weight).\n"
        f"  FIX: drop --margin-weighted-loss for that form, or launch a form that honors it. "
        f"This remains a score-affecting training-force lever, so race it before any claim.")


def basin_entry_fires(w: list[dict]) -> bool:
    """TerminalSolve §16.1 validity predicate over the last-3-gate window (pure logic;
    unit-tested; consumed by main()'s basin-handoff block). Conditions: (a) quadratic
    crawl in BOTH smooth and realized channels; (b) lane topology stable; (c) shadow
    basis + terminal seg FORM throughout (no transitions remaining); zero A1 alarms
    in-window (linearization fidelity)."""
    t = BASIN_THRESHOLDS
    return (len(w) == 3
            and all(x["basis"] == "ema_shadow" for x in w)
            and all(x.get("form") == BASIN_TERMINAL_SEG_FORM for x in w)
            and not any(x["alarm"] for x in w)
            and w[0]["smooth"] > 0 and w[0]["dseg"] > 0
            and (w[0]["smooth"] - w[-1]["smooth"]) / abs(w[0]["smooth"])
                < t["smooth_rel_per_window"]
            and (w[0]["dseg"] - w[-1]["dseg"]) / w[0]["dseg"] < t["dseg_rel_per_window"]
            and abs(w[-1]["lane_b0"] - w[0]["lane_b0"]) <= t["lane_b0_delta_max"]
            and abs(w[-1]["lane_er"] - w[0]["lane_er"]) <= t["lane_erased_delta_max"])


def resolve_gate_ids(num_pairs: int) -> tuple[int, ...]:
    """Pre-registered A1 gate set: all pairs below n600; else fd2 instrument geometry
    (block 447-450 + 32 rng(0)-sampled off-block pairs)."""
    if num_pairs < 600:
        return tuple(range(num_pairs))
    rng = np.random.default_rng(0)
    off = [p for p in range(600) if p not in set(GATE_BLOCK_PAIRS)]
    return GATE_BLOCK_PAIRS + tuple(
        int(x) for x in rng.choice(off, size=GATE_OFFBLOCK_SAMPLE, replace=False))


def a1_smooth_excluding_delta_penalty(ep_loss: float, engaged: bool, weight: float,
                                      penalty: float) -> float:
    """The A1 gate's smooth-loss INPUT with the delta-sparsity penalty term EXCLUDED (ddm_pa1r
    confound fix, 2026-07-31).  MEASURED incident (delta_sparsity_tail, w=0.03 from_step_0 on the
    B resume): the engage transient relaxes the w·penalty term fast (ep444→449 ep_loss −11.3%,
    mostly penalty relaxation) while realized d_seg is flat → ``a1_adjudicate`` misreads the drop
    as a REALIZATION GAP → 2 consecutive alarms → ``a1_realization_gap_refuse`` stop at ep454.
    The instrument must read the SAME quantity in a lever arm as in the control (the seg+rate
    smooth), so the gate input is ``ep_loss − weight·penalty`` when the force is engaged.
    Byte-identical when the lever is off (engaged False or weight 0 ⇒ returns ep_loss exactly).
    Sister of the confound self-protection rule (a new loss term with its own relaxation dynamics
    entered the A1 instrument's input un-corrected — the L1 'instrument reads the wrong quantity'
    class)."""
    if engaged and weight > 0.0:
        return float(ep_loss) - float(weight) * float(penalty)
    return float(ep_loss)


def reset_arm_for(cfg: TR1Config):
    """The pre-registered ``tac.optimization.reset_operator`` arm this config selects (#824).

    Closes the ``TR1ResetOperatorWiring`` charter's BUILT-ELSEWHERE-UNWIRED-HERE grade: the
    operator module was built + tested with zero trainer importers.  This trainer zeroes both
    Adam moments at every boundary (fresh ``optim.Adam``, all six ``save_checkpoint`` sites pass
    ``opt_state_flat={}``), which is exactly ``what='both', to='zero', structure='uniform'`` —
    so the ONLY free knob here is ``bias_correction``, and the two reachable arms are B and B'.
    Arms A and C need ``requires_persistence`` (opt-state save/load) and are OUT OF SCOPE for
    #824 by MEASURED verdict: ``opt_flat`` has one repo-wide hit (the ``load_checkpoint``
    return) and nothing reads it, and nothing writes it — C is a BUILD, not a port.
    """
    from tac.optimization.reset_operator import (
        ARM_B_ZERO_RESET,
        ARM_BPRIME_BIAS_CORRECTED,
    )

    return ARM_BPRIME_BIAS_CORRECTED if cfg.adam_bias_correction else ARM_B_ZERO_RESET


def a1_alarm_summary(gate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """First-class summary of ``A1_REALIZATION_GAP_ALARM`` firings (#824 round-2, MAIN).

    The alarm's semantics — smooth loss fell >=2% while realized d_seg fell <0.5% — are the
    restart decomposition's claim arriving through a COMPLETELY DIFFERENT channel, which is what
    makes it strong: it was built for another job.  It fired **6 times** in the burn (ep649/659,
    ep674/714, ep814/899 — exactly 2 per window, every window) and the b4s supervisor never
    greps it (``grep -c a1_alarm`` = 0); the decision JSONs propagate only ``final_gate_a1``, and
    none of the six was at a final gate, so **all six were invisible to every decision record**.
    The in-trainer guard (``A1_CONSECUTIVE_REFUSE``) is live and correctly did not fire — no two
    were consecutive — which is exactly why silence there was mistaken for absence.

    Pure over the gate rows the caller already holds; emitted per window and on the boundary row.
    Pre-registered read: if B' collapses the boundary jump, this count should FALL too.
    """
    fired = [r for r in gate_rows if r.get("a1_alarm")]
    return {
        "a1_alarm_count": len(fired),
        "a1_alarm_epochs": [int(r["epoch"]) for r in fired if r.get("epoch") is not None],
        "a1_classifications": sorted({str(r.get("a1_classification")) for r in gate_rows}),
        "gates_seen": len(gate_rows),
        "note": "corroborating channel for #824: smooth fell >=2% while realized fell <0.5%. "
                "6 firings in the burn were invisible to every decision record (only "
                "final_gate_a1 was propagated and none was at a final gate)",
    }


def tail_slope_adjudication(gate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Amendment-3 boundary self-adjudication (#874/#935 censored-cap genus; operator
    2026-08-05 "fix censorship sources everywhere"): the window-boundary verdict
    authority is the MEASURED tail-slope fit over the gate history — never the
    per-gate 5-epoch interval label (a1 FLAT censored a real 6.2-sigma descent twice
    on 2026-08-05). ADDITIVE ONLY: never changes a1_alarm / a1_classification; pure
    over the rows the caller already holds.
    """
    pts = sorted(
        (float(r["epoch"]), float(r["realized_gate_dseg_mean"]))
        for r in gate_rows
        if r.get("epoch") is not None and r.get("realized_gate_dseg_mean") is not None
    )
    if len(pts) < 3:
        return {"verdict": "insufficient_gate_rows", "n_gate_rows": len(pts)}
    try:
        from tac.optimization.trajectory_stopping import adjudicate_tail_slope
        payload = adjudicate_tail_slope([p[0] for p in pts], [p[1] for p in pts]).to_payload()
    except Exception as exc:  # fail-open telemetry: a broken fit must not kill the receipt
        return {"verdict": "adjudication_error", "error": f"{type(exc).__name__}: {exc}",
                "n_gate_rows": len(pts)}
    payload["n_gate_rows"] = len(pts)
    return payload


def parent_boundary_ema_decay_fields(meta: Mapping[str, Any]) -> dict[str, Any]:
    """Return the parent decay basis used by cross-boundary telemetry."""
    cfg_meta = meta.get("cfg") if isinstance(meta, Mapping) else None
    jd1_meta = meta.get("jd1_pose_finish") if isinstance(meta, Mapping) else None
    parent_cfg_ema_decay = (
        float(cfg_meta["ema_decay"])
        if isinstance(cfg_meta, Mapping) and cfg_meta.get("ema_decay") is not None
        else None
    )
    parent_active_ema_decay = (
        float(jd1_meta["active_ema_decay"])
        if isinstance(jd1_meta, Mapping) and jd1_meta.get("active_ema_decay") is not None
        else None
    )
    if parent_active_ema_decay is not None:
        parent_boundary_ema_decay = parent_active_ema_decay
        source = "checkpoint jd1_pose_finish.active_ema_decay"
    else:
        parent_boundary_ema_decay = parent_cfg_ema_decay
        source = "checkpoint cfg.ema_decay fallback"
    return {
        "parent_ema_decay": parent_boundary_ema_decay,
        "parent_boundary_ema_decay": parent_boundary_ema_decay,
        "parent_boundary_ema_decay_source": source,
        "parent_cfg_ema_decay": parent_cfg_ema_decay,
        "parent_active_ema_decay": parent_active_ema_decay,
        "parent_active_ema_decay_provenance": (
            str(jd1_meta.get("active_ema_decay_provenance", ""))
            if isinstance(jd1_meta, Mapping) else None
        ),
    }


def resume_ema_decay_fields(parent_fields: Mapping[str, Any], *,
                            child_cfg_ema_decay: float,
                            active_ema_decay: float,
                            active_ema_decay_provenance: str) -> dict[str, Any]:
    """Return resume-event fields after JD1 checkpoint state has been restored."""
    parent_decay = parent_fields.get("parent_boundary_ema_decay")
    held = (
        parent_decay is not None
        and abs(float(parent_decay) - float(active_ema_decay)) <= 1e-12
    )
    return {
        **dict(parent_fields),
        "child_ema_decay": float(active_ema_decay),
        "child_cfg_ema_decay": float(child_cfg_ema_decay),
        "post_restore_active_ema_decay": float(active_ema_decay),
        "post_restore_active_ema_decay_provenance": str(active_ema_decay_provenance),
        "ema_basis_held": bool(held),
        "ema_decay_held": bool(held),
    }


def boundary_jump_row(parent_tail: list[dict[str, Any]], parent_ema_decay: float | None,
                      child_ema_decay: float, resume_epoch: int,
                      gate_row: dict[str, Any], arm: str, *,
                      parent_cfg_ema_decay: float | None = None) -> dict[str, Any] | None:
    """The #824 BOUNDARY-JUMP typed row: the resume interval, isolated (pure; $0 unit-testable).

    WHY this and not an end-state read-out (MEASURED, R1-C, 64 gate readings → 63 intervals):
    the two window-restart intervals sum −1.85083e-4 = **168.6%** of the ep644→945 net while the
    61 TRAINING intervals sum **+7.53e-5 = −68.6%** — training net-REGRESSED and the restarts
    paid for the descent.  The restarts rank 2nd and 6th most-negative of 63; on the RAW
    telescoping basis that matches the 168.6% effect size, exact enumeration of all C(63,2)=1953
    pairs gives p=22/1953=0.0113, Bonferroni ×2 ⇒ **p≈0.0225** (the per-epoch normalization gives
    0.0056/0.011; the claim survives both ways, but 0.0225 is the figure that pairs with the
    quoted effect size — cite THAT).  Re-based on the corrected seg-only −1.96949e-4 the restart
    is **+34.6%** of the descent.  Note the split gc15's mechanism actually PREDICTED —
    boundary+17 epochs, the impulse window — **FAILED** (d_seg ROSE there): the significant window
    is NARROWER than the theory, which is itself a constraint on any mechanism claim.  An
    end-state readout averages that one short interval into ~140 epochs and dilutes it away.

    ``parent_tail`` is the parent checkpoint's ``meta['telemetry_tail']`` (its last ≤4 gate rows).
    Returns None when the parent carries no usable gate anchor (fail-open on ABSENCE of history,
    never on a DRIFT — drift is the ``ema_basis_held`` flag below, which the caller fails closed on).

    ``ema_basis_held`` is the load-bearing caveat: the gate reads the EMA shadow, so if the parent
    and child resolved DIFFERENT ``ema_decay`` the shadow's own averaging length moved underneath
    the measurement and the two readings are not commensurable.  ``derive_ema_decay`` consumes
    ``epochs*(num_pairs//batch_pairs)``, so an ``--epochs`` change alone moves it (the burn ran
    U=49,950/60,450/70,950 ⇒ a different decay at EVERY boundary).  Held ⇔ both arms pin the same
    explicit ``--ema-decay`` (or run identical geometry).
    """
    anchors = [r for r in (parent_tail or [])
               if isinstance(r, dict) and r.get("realized_gate_dseg_mean") is not None
               and r.get("epoch") is not None]
    if not anchors or gate_row.get("realized_gate_dseg_mean") is None:
        return None
    parent = max(anchors, key=lambda r: int(r["epoch"]))
    p_epoch, p_dseg = int(parent["epoch"]), float(parent["realized_gate_dseg_mean"])
    c_epoch, c_dseg = int(gate_row["epoch"]), float(gate_row["realized_gate_dseg_mean"])
    span = c_epoch - p_epoch
    # LEG 1 — the DECAY leg (the original condition): same averaging LENGTH.
    decay_held = (parent_ema_decay is not None
                  and abs(float(parent_ema_decay) - float(child_ema_decay)) <= 1e-12)
    # LEG 2 — the BASIS leg (ddm_op2, 2026-08-03, MEASURED defect OP2-2): same OBJECT.
    # The row already carried both basis fields and did not use them. `ema_basis_held` was
    # computed from the decay ALONE, so it stamped `true` on window_02's boundary — where
    # parent_gate_basis='live_ema_warmup' (window_01 gates on LIVE weights, global_step below
    # the EMA warmup) and first_gate_basis='ema_shadow' (every RESUMED window sets
    # global_step = ema_warmup_updates ⇒ "resume ⇒ warm shadow"). MEASURED: parent ep44 read
    # 0.0157596 (live) against child ep49's 0.5118894 (shadow), a 32x apparent collapse
    # certified as "commensurable" beside a boundary_dseg_delta of +0.496. The magnitude is
    # fully explained by the sealed decay: 1/(1-0.9999199) = 12,487 updates = 166.5 epochs, so
    # at ep46 the shadow still carries exp(-3450/12487) = 76% weight on INITIALIZATION.
    # Matching decays are necessary and NOT sufficient — two shadows of equal length read off
    # different weight sets are still two different quantities (the L1 "instrument reads the
    # wrong quantity" class that `a1_smooth_excluding_delta_penalty` documents, one level up).
    # FAIL-CLOSED on an unverifiable basis: an absent basis field cannot certify commensurability.
    p_basis, c_basis = parent.get("gate_params"), gate_row.get("gate_params")
    basis_held = (p_basis is not None and c_basis is not None and p_basis == c_basis)
    held = bool(decay_held and basis_held)
    return {
        "event": "boundary_jump", "arm": arm,
        "parent_gate_epoch": p_epoch, "parent_gate_dseg": p_dseg,
        "parent_gate_basis": p_basis,
        "first_gate_epoch": c_epoch, "first_gate_dseg": c_dseg,
        "first_gate_basis": c_basis,
        "resume_epoch": int(resume_epoch),
        "boundary_span_epochs": span,
        "boundary_dseg_delta": c_dseg - p_dseg,
        "boundary_dseg_per_epoch": (c_dseg - p_dseg) / span if span > 0 else None,
        "ema_basis_held": held,
        # Which LEG failed, so a reader never has to re-derive it from the two basis strings.
        "ema_decay_held": bool(decay_held),
        "gate_basis_held": bool(basis_held),
        "parent_ema_decay": (None if parent_ema_decay is None else float(parent_ema_decay)),
        "parent_cfg_ema_decay": (
            None if parent_cfg_ema_decay is None else float(parent_cfg_ema_decay)
        ),
        "child_ema_decay": float(child_ema_decay),
        "score_claim": False, "evidence_axis": "[macOS-CPU/MLX advisory]",
        "caveat": "ADVISORY realized-argmax gate on the fd2 gate subset, NOT an exact-eval row; "
                  "commensurable with the parent reading ONLY when ema_basis_held is true "
                  "(BOTH ema_decay_held — same averaging length — AND gate_basis_held — same "
                  "object: live weights vs EMA shadow are not one series)",
    }


def gate_interval_fields(prev: dict[str, Any] | None,
                         cur: dict[str, Any]) -> dict[str, Any]:
    """Per-gate-INTERVAL decomposition fields (#824; pure).  Emitting the interval at write time
    is what makes the 63-interval analysis reproducible from telemetry alone instead of by
    post-hoc pairing — the reconstruction step that hid the restart effect for a whole burn."""
    if prev is None:
        return {"interval_epochs": None, "interval_dseg_delta": None,
                "interval_dseg_per_epoch": None}
    try:
        span = int(cur["epoch"]) - int(prev["epoch"])
        d = float(cur["realized_gate_dseg_mean"]) - float(prev["realized_gate_dseg_mean"])
    except (KeyError, TypeError, ValueError):
        return {"interval_epochs": None, "interval_dseg_delta": None,
                "interval_dseg_per_epoch": None}
    return {"interval_epochs": span, "interval_dseg_delta": d,
            "interval_dseg_per_epoch": (d / span if span > 0 else None)}


def a1_class_motion_fields(prev: dict[str, Any], cur: dict[str, Any],
                           rz_drop: float) -> dict[str, Any]:
    """ddm_bs3 (#909) GUARD: refuse a FLAT realized MEAN to stand as "nothing moved".

    A1's whole decision is a relative drop of ONE scalar (the gate d_seg mean).
    By the triangle inequality the per-class L1 motion
    ``sum_c |cur_c - prev_c| / rz_prev`` is ALWAYS >= ``|rz_drop|``; the gap
    between them is exactly the motion the mean cancels. When the mean reads
    flat (below ``A1_REALIZED_DROP_REL``) while the class composition moved at
    or above that same threshold, the scalar's kernel is demonstrably occupied
    for THIS gate and the flat reading is not evidence of a flat state.

    ADDITIVE ONLY: this never changes ``a1_alarm`` or ``a1_classification`` --
    it records that the sole-evidence scalar was insufficient here. Returns an
    empty dict when either row predates the per-class vector (old telemetry),
    so the qualification is absent rather than silently False."""
    pv = prev.get("realized_gate_dseg_by_gt_class")
    cv = cur.get("realized_gate_dseg_by_gt_class")
    if not isinstance(pv, (list, tuple)) or not isinstance(cv, (list, tuple)) or len(pv) != len(cv):
        return {}
    denom = max(abs(float(sum(pv))), 1e-12)
    l1_rel = sum(abs(float(b) - float(a)) for a, b in zip(pv, cv, strict=True)) / denom
    flat_mean = abs(rz_drop) < A1_REALIZED_DROP_REL
    return {
        "realized_class_l1_rel_since_prev_gate": float(l1_rel),
        "realized_mean_hid_class_motion": bool(flat_mean and l1_rel >= A1_REALIZED_DROP_REL),
    }


def a1_adjudicate(prev: dict[str, Any] | None, cur: dict[str, Any],
                  smooth_prev: float | None, smooth_cur: float) -> dict[str, Any]:
    """Typed A1 verdict per gate: coupled descent vs realization gap (never silent)."""
    out = {"a1_alarm": False, "a1_classification": "FIRST_GATE"}
    if prev is None or smooth_prev is None:
        return out
    sm_drop = (smooth_prev - smooth_cur) / max(abs(smooth_prev), 1e-12)
    rz_prev = prev["realized_gate_dseg_mean"]
    rz_cur = cur["realized_gate_dseg_mean"]
    rz_drop = (rz_prev - rz_cur) / max(abs(rz_prev), 1e-12)
    out["smooth_rel_drop_since_prev_gate"] = float(sm_drop)
    out["realized_rel_drop_since_prev_gate"] = float(rz_drop)
    out.update(a1_class_motion_fields(prev, cur, rz_drop))
    if sm_drop >= A1_SMOOTH_DROP_REL and rz_drop < A1_REALIZED_DROP_REL:
        out["a1_alarm"] = True
        out["a1_classification"] = "A1_REALIZATION_GAP_ALARM"
    elif rz_drop >= A1_REALIZED_DROP_REL:
        out["a1_classification"] = "COUPLED_DESCENT"
    else:
        out["a1_classification"] = "FLAT"
    return out


# ---------------------------------------------------------------------------
# ddm_tp1 (#804) — v9-line confound-cure TELEMETRY PORT (vh1 row 7; burn-4 §3.1).
# Pure / MLX-free / unit-tested row + alarm builders. Emitted by main() ONLY when
# ``--telemetry-v9-port on`` (default on => score-neutral observability is not hidden;
# the flag is threaded via ``args`` ONLY, never TR1Config, and new rows go to the
# telemetry.jsonl via ``tlog`` ONLY — never ``telemetry_tail`` (which is baked into
# the checkpoint meta), so checkpoints are FLAG-INVARIANT).  The reusable Q1-Q7
# producers (``term_inert_rows``, ``lever_engage_row``, ``deterministic_strata``,
# ``ProducerResumeState``) live in ``tac.witness_control.telemetry_producers`` and the
# #404 positive-control canary in ``tac.witness_control.verdict_trend_alarm`` — this
# is a PORT (reuse the v9 producers), not a reimplementation.  READ-ONLY / score-neutral.
# ---------------------------------------------------------------------------
# The exact top-level addends of ``batch_loss``.  "seg" is the distortion term
# (KD distill, when active, folds into it — it is added inside ``pair_loss``).
TR1_BASE_LOSS_TERM_KEYS: tuple[str, ...] = ("seg", "rate", "delta_sparsity")
TR1_LOSS_TERM_KEYS: tuple[str, ...] = (
    *TR1_BASE_LOSS_TERM_KEYS,
    "pose",
    "birth_amplify",
)
# (#321) term_domination — INTENT-RESTORED predicate (b4s first-fire calibration
# 2026-07-31, MAIN adjudication of the burn4 window_01 FALSE POSITIVE): the v9 alarm
# exists to catch a NON-scored term crowding out the SCORED objective ("the scored seg
# signal may be a passenger").  The originally-ported any-term>ceiling predicate fired
# on seg=0.6783 in a seg-only burn — the scored objective dominating BY DESIGN (it
# would fire on every telemetry-on TR1 run).  Alarm predicates are per-vehicle
# CALIBRATION OBJECTS (constants-are-poison applied to predicates).  Corrected:
#   (a) any NON-scored term share > TR1_TERMDOM_FRAC  (v9 caps-law single-term ceiling,
#       MEASURED-ANCHOR provenance: the v9 caps law "each new force <=15%, sum <=40%,
#       any single term >40% => term_domination")
#   (b) the SCORED term share < TR1_SCORED_FLOOR = 1 - TR1_TERMDOM_FRAC  (DERIVED as
#       the complement of the caps-law non-scored AGGREGATE cap: the three post-weight
#       shares sum to ~1 — the loss_terms ``sum_minus_total`` self-check — so
#       scored < 0.60  <=>  non-scored aggregate > 0.40 = seg-as-passenger, the
#       original v9 meaning; no bare constant).
TR1_TERMDOM_FRAC = 0.40
TR1_TERMDOM_MIN_ROWS = 3
TR1_SCORED_TERM = "seg"  # the scored objective on this seg-only vehicle (pose #383 terminal)
TR1_POSE_TERM = "pose"
TR1_SCORED_FLOOR = 1.0 - TR1_TERMDOM_FRAC


def tr1_active_loss_term_keys(*, jd1_pose_finish_active: bool = False,
                              birth_amplify_active: bool = False) -> tuple[str, ...]:
    optional = {
        "pose": bool(jd1_pose_finish_active),
        "birth_amplify": bool(birth_amplify_active),
    }
    return tuple(
        key for key in TR1_LOSS_TERM_KEYS
        if key in TR1_BASE_LOSS_TERM_KEYS or optional.get(key, False)
    )


def tr1_active_scored_terms(*, jd1_pose_finish_active: bool = False) -> tuple[str, ...]:
    return (
        (TR1_SCORED_TERM, TR1_POSE_TERM)
        if jd1_pose_finish_active else (TR1_SCORED_TERM,)
    )


def tr1_loss_terms_row(terms: dict[str, float], total: float, *, ep: int,
                       accum_batch: int, accepted_frac: float, weights_stepped: bool,
                       stage: str, seg_form: str,
                       loss_term_keys: Sequence[str] | None = None) -> dict[str, Any]:
    """(#304) Canonical per-term ``loss_terms`` row for the TR1 top-level loss
    decomposition.  Stable complete key set (missing terms -> 0.0 so the schema is
    config-stable); ``sum_terms`` + ``sum_minus_total`` make the breakdown
    self-checking; ``accepted_frac`` + ``weights_stepped`` are the C6 LIVENESS stamps
    (#402 — a reader can tell a frozen epoch from a converging one).  Pure / MLX-free /
    unit-tested; score-neutral."""
    keys = tuple(loss_term_keys or TR1_BASE_LOSS_TERM_KEYS)
    t = {k: float(terms.get(k, 0.0)) for k in keys}
    ssum = float(sum(t.values()))
    return {
        "stage": "loss_terms", "ep": int(ep), "accum_batch": int(accum_batch),
        "terms": {k: round(v, 6) for k, v in t.items()},
        "total": round(float(total), 6), "sum_terms": round(ssum, 6),
        "sum_minus_total": round(ssum - float(total), 8),
        "accepted_frac": round(float(accepted_frac), 4),
        "weights_stepped": bool(weights_stepped),
        "seg_form": str(seg_form), "tr1_stage": str(stage),
        "score_neutral": True,
    }


def tr1_term_domination_alarms(terms: dict[str, float], total: float,
                               streaks: dict[str, int], *,
                               frac: float = TR1_TERMDOM_FRAC,
                               min_rows: int = TR1_TERMDOM_MIN_ROWS,
                               scored_term: str = TR1_SCORED_TERM,
                               scored_terms: Sequence[str] | None = None,
                               loss_term_keys: Sequence[str] | None = None,
                               scored_floor: float = TR1_SCORED_FLOOR) -> list[dict[str, Any]]:
    """(#321) term_domination, INTENT-RESTORED (see the constants block above): the
    SCORED term is EXEMPT from the ceiling (it dominating a seg-only burn is the
    design); alarm when (a) a NON-scored post-weight addend > ``frac`` of the loss, OR
    (b) the SCORED term's share < ``scored_floor`` (seg-as-passenger, the original v9
    meaning).  Both clauses ``min_rows``-sustained; mutates ``streaks`` (per-term run
    length) IN PLACE and returns the alarm rows that just CROSSED the sustained
    threshold (edge-triggered: a persistent violation emits once per crossing, not
    every row).  Pure / unit-tested."""
    tot_abs = abs(float(total)) + 1e-12
    rows: list[dict[str, Any]] = []
    keys = tuple(loss_term_keys or TR1_BASE_LOSS_TERM_KEYS)
    scored_set = set(scored_terms or (scored_term,))
    scored_label = "+".join(k for k in keys if k in scored_set) or str(scored_term)
    scored_share = (
        sum(abs(float(terms.get(k, 0.0))) for k in keys if k in scored_set) / tot_abs
    )
    scored_streak_key = f"{scored_label}::__scored_floor"
    scored_violated = scored_share < float(scored_floor)
    scored_streak = int(streaks.get(scored_streak_key, 0))
    scored_streak = scored_streak + 1 if scored_violated else 0
    streaks[scored_streak_key] = scored_streak
    if scored_streak == int(min_rows):
        rows.append({
            "event": "confound_alarm", "kind": "term_domination",
            "term": scored_label,
            "predicate": "scored_below_floor",
            "frac_of_loss": round(float(scored_share), 4),
            "sustained_rows": int(scored_streak),
            "note": "the SCORED term share fell below the caps-law floor — non-scored "
                    "terms crowd the scored objective (seg-as-passenger; v9 #321 "
                    "intent, b4s 2026-07-31 first-fire calibration)",
            "score_neutral": True,
        })
    for name in keys:
        if name in scored_set:
            continue
        share = abs(float(terms.get(name, 0.0))) / tot_abs
        violated = share > float(frac)
        predicate = "nonscored_above_ceiling"
        note = ("a NON-scored post-weight term exceeds the v9 caps-law single-term "
                "ceiling; the scored seg signal may be a passenger (v9 #321 port)")
        streak = int(streaks.get(name, 0))
        streak = streak + 1 if violated else 0
        streaks[name] = streak
        if streak == int(min_rows):
            rows.append({
                "event": "confound_alarm", "kind": "term_domination", "term": str(name),
                "predicate": predicate,
                "frac_of_loss": round(float(share), 4), "sustained_rows": int(streak),
                "note": note,
                "score_neutral": True,
            })
    return rows


# ---------------------------------------------------------------------------
# Checkpointing (P0): atomic npz; EMA shadow saved; distinct stage-encoded names.
# ---------------------------------------------------------------------------
def _tree_to_flat(params: dict[str, Any]) -> dict[str, np.ndarray]:
    from mlx.utils import tree_flatten

    return {k: np.asarray(v) for k, v in tree_flatten(params)}


def save_checkpoint(path: Path, *, model, ema: dict[str, Any], opt_state_flat: dict[str, np.ndarray],
                    epoch: int, stage: str, cfg: TR1Config, telemetry_tail: list[dict],
                    extra_meta: dict[str, Any] | None = None,
                    extra_npz_arrays: dict[str, np.ndarray] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    for k, v in _tree_to_flat(model.trainable_parameters()).items():
        payload[f"param::{k}"] = v
    for k, v in ema.items():
        payload[f"ema::{k}"] = np.asarray(v)
    for k, v in opt_state_flat.items():
        payload[f"opt::{k}"] = v
    payload["meta::epoch"] = np.array([epoch], dtype=np.int64)
    meta_payload = {"stage": stage, "cfg": asdict(cfg), "config_hash": cfg.config_hash(),
                    "telemetry_tail": telemetry_tail[-4:]}
    if extra_meta:
        meta_payload.update(extra_meta)
    meta = json.dumps(meta_payload).encode()
    payload["meta::json"] = np.frombuffer(meta, dtype=np.uint8)
    for k, v in (extra_npz_arrays or {}).items():
        if k.startswith(("param::", "ema::", "opt::", "meta::")):
            raise ValueError(f"extra checkpoint payload key {k!r} uses a reserved prefix")
        payload[k] = np.asarray(v)
    tmp = path.parent / (path.name + ".tmp.npz")  # endswith .npz => savez keeps the name
    np.savez(tmp, **payload)
    os.replace(str(tmp), str(path))  # atomic tmp+rename (P0 resumability)


class ResumeGeometryMismatch(RuntimeError):
    """A checkpoint's parameter geometry does not match the live model (fail-closed)."""


#: MLX ``Optimizer.state`` entries that are NOT per-parameter (scalars), so they carry no model
#: geometry and are exempt from the shape check.  VERIFIED against the installed mlx 0.31.2
#: (``tree_flatten(optim.Adam(...).state)`` after one update ⇒ ``step`` (uint64), ``learning_rate``
#: (float32), then ``<param path>.m`` / ``<param path>.v`` at the parameter's own shape).
OPT_STATE_SCALAR_KEYS = ("step", "learning_rate")
#: Per-parameter Adam moment suffixes.  ``<param path>.m`` must have ``<param path>``'s shape.
OPT_STATE_MOMENT_SUFFIXES = (".m", ".v")


def opt_state_param_path(key: str) -> str | None:
    """The model-parameter path a flattened optimizer-state key belongs to, or None for a scalar.

    ``'tokens_base.m' -> 'tokens_base'``; ``'step' -> None``.  Pure; the single place that knows
    the MLX state-key convention, so the guard and the restore path cannot disagree about it.
    """
    if key in OPT_STATE_SCALAR_KEYS:
        return None
    for prefix in ("states.0.", "states.1."):
        if key.startswith(prefix):
            inner = key[len(prefix):]
            if inner in OPT_STATE_SCALAR_KEYS:
                return None
            for suffix in OPT_STATE_MOMENT_SUFFIXES:
                if inner.endswith(suffix):
                    return inner[: -len(suffix)]
            return None
    for suffix in OPT_STATE_MOMENT_SUFFIXES:
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return None


def assert_resume_geometry_compatible(ckpt_param_shapes: dict[str, tuple[int, ...]],
                                      model_param_shapes: dict[str, tuple[int, ...]],
                                      ckpt_opt_shapes: dict[str, tuple[int, ...]] | None = None,
                                      ) -> list[str]:
    """REFUSE a resume whose checkpoint params do not fit the live model.

    ddm_gd4 G1 (MEASURED defect, 2026-08-02): ``mlx.nn.Module.update`` assigns a
    wrong-SHAPED array with no complaint, so a ds=16 checkpoint loaded into a ds=32
    model silently produced ``tokens_delta`` (P,24,32,4) inside a renderer that
    expects (P,12,16,4) — while ``up4`` stayed at init and was absorbed by the
    resume block's ``backfilled`` "new param since the checkpoint" path, which
    logs a truthful-looking line. A multi-day run whose first crash-recovery step
    is ``--resume-from`` cannot carry that: the wrong stage checkpoint out of the
    burn directory is one tab-completion away.

    STRUCTURAL by construction (compares the checkpoint's ``param::`` tree against
    the live ``model.trainable_parameters()``), so it needs no config argument and
    protects EVERY caller, not just the trainer's resume block. Returns the list of
    params present in the model but absent from the checkpoint (the legitimate
    newly-introduced-lever case the caller backfills); raises on any shape conflict
    or on a checkpoint param the model does not have.

    ``ckpt_opt_shapes`` (ddm_op2, OP2-1) extends the SAME guard over the persisted
    optimizer-moment tree rather than letting that payload travel around it. Every
    ``<param>.m`` / ``<param>.v`` carries the parameter's own shape, so a ds=16
    moment tree is exactly as inadmissible in a ds=32 model as the params are —
    and ``optimizer.state`` assignment is the same silent-reshape surface
    ``Module.update`` is. Scalars (``step`` / ``learning_rate``) are geometry-free
    and exempt. ``None`` (the default) ⇒ byte-identical behaviour to the pre-OP2-1
    guard, so no existing caller changes.
    """
    shape_conflicts = sorted(
        f"{k}: ckpt{tuple(ckpt_param_shapes[k])} != model{tuple(model_param_shapes[k])}"
        for k in (ckpt_param_shapes.keys() & model_param_shapes.keys())
        if tuple(ckpt_param_shapes[k]) != tuple(model_param_shapes[k]))
    orphaned = sorted(ckpt_param_shapes.keys() - model_param_shapes.keys())
    for _k, _shape in sorted((ckpt_opt_shapes or {}).items()):
        _path = opt_state_param_path(_k)
        if _path is None:
            continue                      # step / learning_rate: geometry-free
        if _path not in model_param_shapes:
            orphaned.append(f"opt::{_k} (no model param {_path!r})")
        elif tuple(_shape) != tuple(model_param_shapes[_path]):
            shape_conflicts.append(
                f"opt::{_k}: ckpt{tuple(_shape)} != model{tuple(model_param_shapes[_path])}")
    shape_conflicts, orphaned = sorted(shape_conflicts), sorted(orphaned)
    if shape_conflicts or orphaned:
        raise ResumeGeometryMismatch(
            "resume REFUSED — checkpoint geometry does not match the live model "
            "(never silently reshape a warm start). "
            f"shape conflicts: {shape_conflicts or 'none'}; "
            f"checkpoint params absent from the model: {orphaned or 'none'}. "
            "This is the geometry-bearing config set (grid_downsample / code_width / "
            "renderer_width / num_pairs / variant / token_temporal_mode / "
            "renderer_head_mode): a checkpoint may only resume into a model built "
            "with the SAME values. Start a fresh run instead of resuming across it.")
    return sorted(model_param_shapes.keys() - ckpt_param_shapes.keys())


def no_opt_state(reason: str) -> dict[str, np.ndarray]:
    """An EXPLICIT "this checkpoint deliberately carries no optimizer state", with the reason.

    ddm_op2 (OP2-1): a bare ``opt_state_flat={}`` literal cannot be told apart from a site that
    silently forgot to thread the state — which is exactly how all six callsites came to drop it
    (``#824`` recorded the result as reset-arm B, and ``ddm_gd5`` §3.6 measured its price at
    16.167 epochs per boundary). Every callsite now passes either the run's resolver or this,
    so "none" is a stated decision carrying its rationale, and the sister gate
    ``check_tr1_save_checkpoint_threads_optimizer_state`` can refuse the bare literal.
    """
    if not isinstance(reason, str) or len(reason.strip()) < 8:
        raise ValueError("no_opt_state(reason=...) needs a substantive rationale "
                         "(placeholder/empty rejected)")
    return {}


def optimizer_state_to_flat(optimizer) -> dict[str, np.ndarray]:
    """Flatten ``optimizer.state`` to the ``opt::``-prefixable numpy payload (ddm_op2, OP2-1).

    Returns ``{}`` before the optimizer has any per-parameter state (a freshly constructed
    optimizer that has neither ``.init()``-ed nor stepped carries only ``step`` /
    ``learning_rate``); persisting a moment-free state would be a truthful-looking checkpoint
    that restores nothing, which is the silent-drop class this whole fix exists to close.
    """
    from mlx.utils import tree_flatten

    flat = {k: np.asarray(v) for k, v in tree_flatten(optimizer.state)}
    if not any(opt_state_param_path(k) is not None for k in flat):
        return {}
    return flat


class OptimizerStateRestoreError(RuntimeError):
    """The persisted optimizer state cannot be restored onto the live optimizer (fail-closed)."""


def restore_optimizer_state(optimizer, model, opt_flat: dict[str, np.ndarray]) -> dict[str, Any]:
    """Restore persisted Adam moments onto ``optimizer``; returns a typed telemetry row.

    THE DEFECT THIS CLOSES (ddm_op2 OP2-1, MEASURED by ``ddm_gd5`` §3.6 and corroborated here).
    All six ``save_checkpoint`` callsites passed ``opt_state_flat={}``, so **no checkpoint on
    disk carried optimizer state** and every resume constructed a fresh ``optim.Adam`` with both
    moments zeroed — the pre-registered ``#824`` reset-operator **arm B**
    (``what='both', to='zero', structure='uniform'``). The trainer's own ``optimizer_arm`` row
    ships the price: ``boundary_impulse_epochs_per_reset = 16.167``. Executed in 30-minute
    windows at the MEASURED ~46 epochs/window, a 666-epoch run pays that at ~13.5 boundaries
    ⇒ **~218 of 666 epochs (33%) spent re-converging a deliberately reset Adam**, leaving
    ~450 effective epochs against an incumbent lineage that reached ep945. ``#824`` scoped
    arms A/C out as "a BUILD, not a port" because ``opt_flat`` had one repo-wide hit that
    nothing read; the measurement made the BUILD load-bearing.

    POSITIVE CONTROL (MEASURED, ddm_op2, mlx 0.31.2, in-tree as
    ``test_ddm_op2_optimizer_state_persistence.py``): 3 steps → snapshot → rebuild → restore →
    3 steps reproduces an uninterrupted 6-step run to **max abs param diff 0.0**, while the
    reset path diverges by 5.5e-2. A boundary with persistence is a no-op on trained bytes;
    that is the whole claim, and it is measured rather than argued.

    ``learning_rate`` is deliberately NOT restored: the live ``cfg.lr`` must win, or a window
    that changed ``--lr`` would silently keep the parent's (a silent-wrong of exactly the kind
    the resume-geometry guard exists to refuse). A difference is reported on the row, LOUD.
    """
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten

    moments = {k: v for k, v in opt_flat.items() if opt_state_param_path(k) is not None}
    if not moments:
        raise OptimizerStateRestoreError(
            "optimizer-state restore REFUSED — the checkpoint carries no per-parameter moments "
            f"(keys: {sorted(opt_flat) or 'none'}). Resuming from it would silently be an "
            "arm-B zero reset while claiming persistence.")
    # Materialize the per-parameter state so the moments have somewhere to land: MLX creates
    # it lazily on the first `update`, so a freshly constructed optimizer has only the scalars.
    optimizer.init(model.trainable_parameters())
    live = dict(tree_flatten(optimizer.state))
    missing = sorted(k for k in live if opt_state_param_path(k) is not None and k not in opt_flat)
    extra = sorted(k for k in moments if k not in live)
    if extra:
        raise OptimizerStateRestoreError(
            f"optimizer-state restore REFUSED — checkpoint moments absent from the live "
            f"optimizer: {extra}. Never assign optimizer state the live tree does not have.")
    parent_lr = (float(opt_flat["learning_rate"]) if "learning_rate" in opt_flat else None)
    live_lr = float(live["learning_rate"]) if "learning_rate" in live else None
    for k, v in moments.items():
        live[k] = mx.array(v)
    if "step" in opt_flat:
        live["step"] = mx.array(opt_flat["step"])
    optimizer.state = tree_unflatten(list(live.items()))
    mx.eval(optimizer.state)
    return {
        "event": "optimizer_state_restored", "arm": "ddm_op2",
        "moments_restored": len(moments),
        "step_restored": (int(np.asarray(opt_flat["step"]).reshape(-1)[0])
                          if "step" in opt_flat else None),
        # A param the checkpoint has no moment for (a lever introduced since) starts at zero
        # moment BY DESIGN — that is the honest state, and it is NAMED rather than silent.
        "moments_missing_start_at_zero": missing,
        "parent_learning_rate": parent_lr, "live_learning_rate": live_lr,
        "learning_rate_restored": False,
        "learning_rate_differs": (parent_lr is not None and live_lr is not None
                                  and abs(parent_lr - live_lr) > 1e-12),
        "score_claim": False, "evidence_axis": "[macOS-CPU/MLX advisory]",
        "note": "#824 arm C (persisted (m,v)): the boundary reset impulse "
                "(16.167 epochs/reset) is not paid. learning_rate is NOT restored — the live "
                "cfg.lr wins by design.",
    }


def load_checkpoint(path: Path, model) -> dict[str, Any]:
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten

    z = np.load(path, allow_pickle=False)
    params = [(k[len("param::"):], mx.array(z[k])) for k in z.files if k.startswith("param::")]
    opt = {k[len("opt::"):]: z[k] for k in z.files if k.startswith("opt::")}
    # P0 fail-closed geometry guard BEFORE any assignment (see the docstring above).
    # ddm_op2 (OP2-1): the optimizer-moment tree goes THROUGH this guard, never around it.
    new_params = assert_resume_geometry_compatible(
        {k: tuple(v.shape) for k, v in params},
        {k: tuple(np.asarray(v).shape)
         for k, v in tree_flatten(model.trainable_parameters())},
        {k: tuple(np.asarray(v).shape) for k, v in opt.items()})
    model.update(tree_unflatten(params))
    ema = {k[len("ema::"):]: mx.array(z[k]) for k in z.files if k.startswith("ema::")}
    meta = json.loads(bytes(z["meta::json"]).decode())
    return {"epoch": int(z["meta::epoch"][0]), "ema": ema, "opt_flat": opt, "meta": meta,
            "params_new_since_checkpoint": new_params}


def ema_snapshot_swap(model, ema: dict[str, Any]):
    """Return the live flat params and swap in the EMA shadow (caller restores)."""
    import mlx.core as mx
    from mlx.utils import tree_flatten, tree_unflatten

    live = dict(tree_flatten(model.trainable_parameters()))
    model.update(tree_unflatten([(k, mx.array(ema[k])) for k in ema]))
    return live


def ema_restore(model, live: dict[str, Any]):
    from mlx.utils import tree_unflatten

    model.update(tree_unflatten(list(live.items())))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variant", choices=("plain", "lotto"), required=True)
    ap.add_argument("--num-pairs", type=int, default=600)
    # ddm_gd4 (mt1 §5 #1 / gd3 §6.1): 32 ADMITTED. The trainer was already ds=32-ready —
    # grid_h/grid_w derive from SEG_H//D, SEG_W//D; n_upsample from log2(D); _conv_shapes,
    # the param-registration loop and render_frame all iterate range(n_upsample) — so this
    # argparse tuple was the entire blocker. MEASURED ds=32-ready (gd4 G1 probe): 7 convs,
    # grid 12x16, render (1,384,512,3), up4 registered trainable with NONZERO gradient and a
    # 79.33-absmax causal effect on the render (vs ds=16 up3's 76.08) — NOT an inert layer.
    ap.add_argument("--grid-downsample", type=int, default=16, choices=(8, 16, 32))
    ap.add_argument("--code-width", type=int, default=4, choices=(2, 4, 6))
    ap.add_argument("--renderer-width", type=int, default=24)
    ap.add_argument("--token-quant-levels", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lotto-seed", type=int, default=118)
    ap.add_argument("--lotto-mask-density-init", type=float, default=0.5)
    ap.add_argument("--seg-form-start", default="ce",
                    choices=("ce", "tau_softplus", "unify_tau", "margin_hinge"))
    ap.add_argument("--margin-target", type=float, default=1.0,
                    help="margin_hinge target (RACED lever; step-native lineage)")
    ap.add_argument("--class-weight-lane", type=float, default=1.0,
                    help="sn1 asymmetry lever: loss weight on GT Lane pixels (1.0 = off)")
    # ---- ddm_lg1 (#808) CONSTRAIN-AND-PROTECT layer (default-OFF => byte-identical) ----
    ap.add_argument("--lane-guard", action="store_true",
                    help="lg1: primal-dual Lane constraint + born-lane protection + margin-floor "
                         "(DEFAULT-OFF; byte-identical when absent). Budget = xp1 ep641 Lane S.")
    ap.add_argument("--lane-guard-budget-s", type=float, default=0.0,
                    help="lg1 constraint budget in S-units (0.0 => LANE_BUDGET_S_UNITS 0.12589, xp1)")
    ap.add_argument("--lane-guard-eta", type=float, default=0.0,
                    help="lg1 dual step size (0.0 => DERIVED derive_eta_lambda ~66.2)")
    ap.add_argument("--lane-guard-lambda-step-cap", type=float, default=0.0,
                    help="lg1 per-gate dual-step ceiling (0.0 => DERIVED 0.1; caps-law)")
    ap.add_argument("--lane-guard-lambda-max", type=float, default=5.0,
                    help="lg1 bounded lambda ceiling (5x the natural per-Lane-pixel weight unit)")
    ap.add_argument("--lane-guard-born-weight", type=float, default=0.0,
                    help="lg1 born-lane protection weight (0.0 => OFF; scaled by Lane head sensitivity)")
    ap.add_argument("--lane-guard-margin-floor-weight", type=float, default=0.0,
                    help="lg1 low-margin Lane emphasis weight (0.0 => OFF; floor DERIVED from QA80 p10)")
    ap.add_argument("--lane-guard-lambda-init", type=float, default=0.0,
                    help="lg1 warm-start dual multiplier (supervisor rollback+raise-lambda relaunch)")
    ap.add_argument("--lane-guard-ratchet", action="store_true",
                    help="bs2 #871: make the Lane budget a monotone RATCHET that locks in won "
                         "Lane (deadband sigma MEASURED online, k CALIBRATED vs the null). "
                         "DEFAULT-OFF: the constant budget was MEASURED inert on burn-4 "
                         "(lambda==0 on 64/64 gates), so off is the known-inert arm.")
    ap.add_argument("--lane-guard-ratchet-horizon", type=int, default=0,
                    help="bs2 #871 deadband horizon in gates. 0 => DERIVED from this run's "
                         "epochs/gate_every (the planned gate TOTAL), which is the correct "
                         "multiple-comparisons burden and gives a stationary deadband. "
                         "ddm_lp1 #934 replaced the old 0 => 'gates seen' default, MEASURED "
                         "to fire 3/64 false positives on the burn-4 series.")
    # ---- ddm_p4x (#920) LANE EXISTENCE PRIMITIVE + per-class BIRTH MATRIX ---------------
    ap.add_argument("--existence-hinge-weight", type=float, default=0.0,
                    help="p4x #920: weight of the COMPONENT-level existence hinge "
                         "s(c)=logsumexp_beta(live margin over GT component c); "
                         "loss = mean_c w_c*relu(target - s(c)). 0.0 => OFF (byte-identical). "
                         "This is a SEPARATE TERM, not a seg_pixel_w addend: cg1r MEASURED "
                         "per-flip depth as direction-symmetric (1.074x), so the lane-erasure "
                         "discount is VOLUMETRIC and per-pixel reweights are expected null on "
                         "the ANNIHILATE verb.")
    ap.add_argument("--existence-hinge-classes", default="lane,movable",
                    help="p4x comma list of protected classes. MEASURED default: the only two "
                         "with a materially non-zero word-annihilation rate (Lane 54.38%%, "
                         "Movable 16.20%% at 8-conn); MyCar annihilates 0 words in 600 frames.")
    ap.add_argument("--existence-hinge-beta", type=float, default=0.0,
                    help="p4x softmax sharpness (0.0 => per-class DERIVED as "
                         "log(mean_component_area)/tolerance; Lane 7.4587, Movable 12.9896). "
                         "beta->inf recovers the exact witness pixel.")
    ap.add_argument("--existence-hinge-target", type=float, default=0.0,
                    help="p4x hinge target in margin units (0.0 = bare existence, i.e. the "
                         "decision boundary itself; >0 adds a survival cushion)")
    ap.add_argument("--existence-hinge-weight-policy", default="",
                    choices=("", "uniform", "sqrt_area", "area"),
                    help="p4x per-component weight w_c ('' => per-class BIRTH_MATRIX policy: "
                         "Lane uniform (no interior), Movable sqrt_area (has interior)). RACED.")
    ap.add_argument("--existence-hinge-connectivity", type=int, default=8, choices=(4, 8),
                    help="p4x component grammar. 8 = Rosenfeld/receiver-consolidation (default, "
                         "physical); 4 = gt2's own published grammar (MEASURED by p4x: gt2 is "
                         "4-connected). Per-WORD rates are NOT comparable across grammars.")
    # ---- ddm_bi1 (#924) TR1 BIRTH seed/amplify path -------------------------------
    # Args-only by design: OFF must preserve TR1Config/config_hash/checkpoint bytes. ON emits
    # its own telemetry row and is a duty-to-measure lever, not a score claim.
    ap.add_argument("--tr1-birth-seed-weight", type=float, default=0.0,
                    help="BI1: initialize TR1 token cells on GT Lane/Movable birth supports. "
                         "0.0 => OFF and byte-identical. Lane is the default first rung.")
    ap.add_argument("--tr1-birth-seed-classes", default=TR1_BIRTH_SEED_DEFAULT_CLASSES,
                    help="BI1 comma list from {lane,movable}; default lane super-nuclei first.")
    ap.add_argument("--tr1-birth-seed-dilate-px", type=int, default=1,
                    help="BI1 scorer-plane support growth before token-grid max-pool.")
    ap.add_argument("--tr1-birth-amplify-weight", type=float, default=0.0,
                    help="BI1 scorer-free token-anchor weight that keeps seeded supports from "
                         "being immediately erased. Requires --tr1-birth-seed-weight > 0.")
    ap.add_argument("--tr1-birth-amplify-persist", default="inverse_thickness",
                    choices=("uniform", "inverse_thickness"),
                    help="BI1 seed/amplify support weighting; inverse_thickness emphasizes "
                         "lowest-persistence island pixels.")
    # ---- ddm_tk1 (2026-08-05) PE3 conditioning-only slot ----------------------------
    # Args-only by contract: OFF preserves TR1Config/config_hash/checkpoint bytes. ON parses
    # the receiver-closed PE3EDGE1 section and attaches fixed prior channels plus learnable
    # per-mode trust gates. It is NEVER a label-replacement target.
    ap.add_argument("--pe3-conditioning-cache", type=Path, default=None,
                    help="TK1 PE3 conditioning cache: raw PE3EDGE1 section, IX2 payload, or "
                         "single-member archive carrying the LC1/PK1 PE3 section. Required when "
                         "--pe3-conditioning-mode conditioning_only; refused on SHA mismatch.")
    ap.add_argument("--pe3-conditioning-mode", default="off",
                    choices=("off", "conditioning_only"),
                    help="TK1 PE3 consumer mode. off = byte-identical control; conditioning_only "
                         "= feed PE3 boundary/mode/transition prior channels into the TR1 trunk "
                         "through learned trust gates. No target replacement loss exists.")
    # ---- ddm_tk1 (2026-08-05) cheapdct4 pose-carriage accounting ---------------------
    # Accounting, not a full in-loop renderer consumption: it decodes OD9's stage2 qcoeffs and
    # reports OD9's measured subset pose term in composed-S receipts.
    ap.add_argument("--cheapdct4-pose-cache", type=Path, default=None,
                    help="TK1 cheapdct4 pose-accounting cache: OD9_RECEIPT.json. The receipt's "
                         "packet SHA is rechecked, then the stage2_qcoeffs section is decoded.")
    ap.add_argument("--cheapdct4-pose-mode", default="off",
                    choices=("off", "accounting"),
                    help="TK1 cheapdct4 consumer mode. off = byte-identical control; accounting "
                         "= decode OD9 stage2_qcoeffs and attach the measured n32 pose term to "
                         "trainer/composed-S receipts. Does not claim full joint descent.")
    # ---- ddm_jd1 (2026-08-05) TR1 joint pose-finish + stage-2 constrain ----------------
    # Args-only: OFF must preserve TR1Config/config_hash/checkpoint bytes. ON consumes the
    # GT cache's real gt_poses through make_loss_fn's existing PoseNet path, after the seg
    # stage has reached the constrain boundary. No scorer result is claimed by this build.
    ap.add_argument("--jd1-pose-finish-mode", default="off",
                    choices=JD1_POSE_FINISH_MODES,
                    help="JD1 pose gate. off = existing TR1 seg-only path; joint_loss = after "
                         "the configured engage condition, thread gt_poses into make_loss_fn "
                         "with compute_pose=True and --jd1-w-pose.")
    ap.add_argument("--jd1-pose-finish-engage-on", default="post_knee",
                    choices=JD1_POSE_FINISH_ENGAGE_ON,
                    help="JD1 engagement predicate. post_knee waits until the CE->tau/base-"
                         "stable boundary; start_epoch uses only --jd1-pose-finish-start-epoch.")
    ap.add_argument("--jd1-pose-finish-start-epoch", type=int, default=0,
                    help="JD1 minimum epoch before pose-finish may engage. 0 is allowed only "
                         "with post_knee, which still waits for the seg/constrain boundary.")
    ap.add_argument("--jd1-w-pose", type=float, default=0.0,
                    help="JD1 score-domain pose weight passed to make_loss_fn after engagement. "
                         "0.0 means the pose path is off and must be paired with mode=off.")
    ap.add_argument("--jd1-pose-eps", type=float, default=1e-8,
                    help="JD1 pose sqrt epsilon passed to make_loss_fn; default mirrors the "
                         "shared loss function.")
    ap.add_argument("--jd1-seg-hold-weight", type=float, default=0.0,
                    help="JD1 stage-2 constrain hinge. When pose-finish is active, add "
                         "weight*relu(seg_proxy_batch - (floor + margin)); 0.0 = off.")
    ap.add_argument("--jd1-seg-hold-floor-source", default="off",
                    choices=JD1_SEG_HOLD_FLOOR_SOURCES,
                    help="JD1 seg-hold floor source. last_pre_pose_epoch_loss latches the final "
                         "seg-only epoch loss at engagement; checkpoint_tail_ep_loss reads the "
                         "parent checkpoint telemetry tail on resume; explicit reads "
                         "--jd1-seg-hold-floor.")
    ap.add_argument("--jd1-seg-hold-floor", type=float, default=0.0,
                    help="JD1 explicit seg-hold floor, required when floor-source=explicit.")
    ap.add_argument("--jd1-seg-hold-margin", type=float, default=0.0,
                    help="JD1 non-negative slack added to the seg-hold floor.")
    ap.add_argument("--jd1-seg-hold-space", default="loss",
                    choices=JD1_SEG_HOLD_SPACES,
                    help="JD1 seg guard surface. loss = legacy differentiable hinge on "
                         "seg_proxy_batch. realized = jd3 gate-space controller: latch "
                         "realized d_seg at the first post-engagement gate, then rollback "
                         "to the previous gate checkpoint and retreat pose pressure when "
                         "later realized gates exceed floor+margin.")
    ap.add_argument("--jd1-realized-hold-margin", type=float, default=0.0,
                    help="JD3 realized-space hold slack. 0.0 derives from the first "
                         "post-engagement gate as sd(per-pair d_seg)/sqrt(n_gate); "
                         ">0 is an explicit d_seg-rate margin.")
    ap.add_argument("--jd1-realized-hold-pose-retreat", type=float, default=0.0,
                    help="JD3 pose-pressure retreat factor after a realized-hold breach. "
                         "0.0 derives the bisection factor 0.5; explicit values must be "
                         "in (0,1).")
    ap.add_argument("--jd1-realized-hold-max-retreats", type=int, default=0,
                    help="JD3 maximum rollback+retreat events. 0 derives from the A1 "
                         "consecutive-refuse count.")
    ap.add_argument("--jd1-ema-stage-scope", default="off",
                    choices=JD1_EMA_STAGE_SCOPES,
                    help="JD3 EMA scope. off = legacy run/parent-chain EMA. window = "
                         "at joint-pose engagement or engaged resume, preserve the parent "
                         "shadow in the entry checkpoint and re-anchor EMA to live weights "
                         "with decay derived from the remaining stage window.")
    ap.add_argument("--jd1-ema-mode", default="geometric",
                    choices=JD1_EMA_MODES,
                    help="DY2 EMA update mode for JD1 windows. geometric = existing "
                         "decay update. plateau_tail_average = run geometric EMA until "
                         "--jd1-ema-tail-anchor-epoch, then reset shadow to live weights "
                         "and update it as a growing-horizon Polyak tail average.")
    ap.add_argument("--jd1-ema-tail-anchor-epoch", type=int,
                    default=JD1_EMA_TAIL_ANCHOR_OFF,
                    help="DY2 explicit absolute epoch that anchors plateau_tail_average. "
                         "-1 = off. Required when --jd1-ema-mode plateau_tail_average.")
    ap.add_argument("--jd1-live-gate-telemetry", default="off",
                    choices=JD1_LIVE_GATE_TELEMETRY,
                    help="JD3 observability. on additionally logs a live-weight realized "
                         "d_seg row at each gate while leaving the EMA gate row/checkpoint "
                         "tail unchanged.")
    ap.add_argument("--jd1-force-ema-reanchor-on-resume", action="store_true",
                    help="JD4 continuation guard. Default off preserves legacy resume bytes; "
                         "when set on a JD1 window resume, ignore a carried "
                         "stage_ema_reanchored latch and re-derive the stage EMA decay from "
                         "the new window geometry.")
    ap.add_argument("--token-temporal-mode", default="shared_base",
                    choices=("shared_base", "independent"),
                    help="shared_base = identity-xi advection (Einstein d_cov/d_gauge force)")
    ap.add_argument("--token-ste", default="round", choices=("round", "dither"),
                    help="RACED: uint8 rounding is directionally asymmetric through R")
    ap.add_argument("--w-seg", type=float, default=100.0)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--jd1-lr-anneal", default="off", choices=JD1_LR_ANNEAL_MODES,
                    help="ddm_la1 terminal JD1 LR anneal. off = legacy flat LR, with no "
                         "telemetry/checkpoint/config changes. derived_tail = derive a "
                         "tail-only cosine damping schedule from the parent JD1 telemetry "
                         "window plus beta2/EMA memory constants.")
    ap.add_argument("--jd1-lr-final-frac", type=float,
                    default=JD1_LR_FINAL_FRAC_DERIVED,
                    help="ddm_la1 terminal LR fraction. 0.0 = derive from the parent "
                         "tail oscillation sd/(sd+half_range); explicit values must be "
                         "in (0,1) and require --jd1-lr-anneal derived_tail.")
    ap.add_argument("--jd1-finisher", default="off", choices=JD1_FINISHER_MODES,
                    help="ddm_wp1 terminal JD1 finisher. off = existing Adam path; muon = "
                         "Case-B-only terminal optimizer switch: MLX Muon on renderer matrix "
                         "tensors (MLX flattens conv filters) plus Adam fallback for tokens, "
                         "biases, gains, and gates. Args-only; default off preserves "
                         "TR1Config/config_hash/checkpoint bytes.")
    ap.add_argument("--batch-pairs", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--gate-every", type=int, default=5)
    ap.add_argument("--ema-decay", type=float, default=None,
                    help="explicit override; default = DERIVED from run geometry (LawRef). "
                         "PIN THIS across A/B arms: derive_ema_decay consumes total_updates = "
                         "epochs*(num_pairs//batch_pairs), so an --epochs change silently moves "
                         "the EMA shadow length the realized gate READS (#824 R1-C confound)")
    # ---- ddm_bp1 (#824) boundary reset race: arm selector + boundary instrument ----
    ap.add_argument("--adam-bias-correction", default="off", choices=("off", "on"),
                    help="#824 reset-race ARM SELECTOR (tac.optimization.reset_operator): 'off' "
                         "= ARM_B_ZERO_RESET (MLX Adam's own default => bit-identical to every "
                         "pre-#824 run); 'on' = ARM_BPRIME_BIAS_CORRECTED (bias_correction=True "
                         "=> the post-reset step is lr*sign(g), removing the eta(t) impulse "
                         "worth 1212.57 excess sign-steps = 16.168 epochs per boundary at 75 "
                         "steps/epoch, 81.7%% of it inside the first 13 epochs). "
                         "on|off is used (not store_true) so the DSL compiles a VALUED flag")
    ap.add_argument("--boundary-probe", default="off", choices=("off", "on"),
                    help="#824 boundary instrument (READ-ONLY; args-only, never TR1Config => "
                         "trained bytes flag-invariant). 'on' adds (a) a POSITIVE-CONTROL re-gate "
                         "at the resume epoch BEFORE any training (must reproduce the parent "
                         "checkpoint's last gate; if it does not, the instrument is untrusted and "
                         "no verdict is admissible) and (b) a FAIL-CLOSED refusal when the parent "
                         "and child EMA decay differ (the measuring instrument's own averaging "
                         "length must not drift under the measurement). Costs ONE extra gate")
    ap.add_argument("--gt-cache", type=Path, default=Path(DEFAULT_GT_CACHE))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--resume-from", type=Path, default=None)
    ap.add_argument("--max-wall-minutes", type=float, default=90.0)
    ap.add_argument("--full-confirm", action="store_true",
                    help="run the full num-pairs realized confirm at the final stage exit")
    ap.add_argument("--verdict-chunk", type=int, default=32,
                    help="pairs per CPU SegNet verdict chunk (<=120 per the charter)")
    ap.add_argument("--mlx-device", default="gpu", choices=("gpu", "cpu"))
    ap.add_argument("--deterministic-r", action="store_true",
                    help="ddm_dt1 (#903): route the R operator through the atomics-free fused "
                         "Metal kernel so the R VJP is run-to-run BIT-IDENTICAL. DEFAULT-OFF "
                         "(absent => the historical mx.vjp scatter backward, byte-identical to "
                         "every prior run). MEASURED 2026-08-03: the default backward is "
                         "NON-deterministic even WITHIN one process (the upsample VJP is a "
                         "scatter; the downsample VJP is clean), which makes 40/41 checkpoint "
                         "arrays differ between two runs of the same seed+config+inputs. With "
                         "this flag: 41/41 arrays and 134/134 telemetry fields bit-identical "
                         "across 4 runs. The fused FORWARD is bit-identical to the reference "
                         "(max|delta|=0 at the real 384x512->874x1164->384x512 geometry) and the "
                         "grad differs from the reference by ~1 ULP -- i.e. it picks one FIXED "
                         "member of the noise cloud the reference was already sampling from. "
                         "Also ~4.5x faster on the R grad. Requires a Metal GPU: REFUSES rather "
                         "than silently falling back.")
    ap.add_argument("--token-init-mode", default="zero", choices=("zero", "solve_project"),
                    help="lv1 B solve-init: v3 ANALYTIC projection of the materializable "
                         "solution-set member (GT frame_1 at the render plane, area-mean "
                         "downsampled) into token space as base+delta BEFORE the scorer "
                         "loop (eu1 teacher-as-init-oracle mechanism); zero = tb1 control")
    ap.add_argument("--basin-handoff", default="off", choices=("off", "on"),
                    help="operator 2026-07-28 basin rule: train ONLY to condition; on "
                         "basin-entry detection (quadratic crawl + topology stable + no "
                         "transitions remaining, TerminalSolve validity conditions) STOP "
                         "and hand off to the solve executors (handoff receipt written; "
                         "full-confirm at handoff = the v19 realized acceptance baseline)")
    # ---- QA24 5-piece composed re-burn (sg1 §3) ----
    ap.add_argument("--token-cell-mask", type=str, default=None,
                    help="§3.1 coarse-from-birth: path to a (grid_h,grid_w) bool .npy (True="
                         "KEEP). Inactive cells are multiplicatively zeroed in the token field "
                         "(no gradient, no coded bytes). None = uniform dense grid")
    ap.add_argument("--margin-weighted-loss", default="off", choices=("off", "on"),
                    help="§3.2 boundary-annulus form fix: 'on' builds make_loss_fn with "
                         "margin_weighted=True (100%% of flips at small GT-margin; sg1 §1.3)")
    # ---- ddm_tp2 row 3: #274 spike/coherent reweight, ported from the levelset trainer ----
    ap.add_argument("--seg-spike-reweight", action="store_true",
                    help="ddm_tp2/#274 PORT: build the per-pair temporal-instability weight map "
                         "(SPIKE = GT argmax differs from BOTH neighbours = unfittable flicker; "
                         "COHERENT = unstable but matches >=1 = winnable boundary) and fold it "
                         "MULTIPLICATIVELY into seg_pixel_w. Gate only; the scalars carry the "
                         "magnitude. DEFAULT-OFF => byte-identical.")
    ap.add_argument("--seg-spike-downweight", type=float, default=1.0,
                    help=f"weight on SPIKE pixels (1.0 = inert). CONCESSION-priced, deliberately "
                         f"NOT risk-proportional (~88.6%% irreducible; smooth is optimal there). "
                         f"Race start {SEG_SPIKE_DOWNWEIGHT_RACE_START}. Requires "
                         f"--seg-spike-reweight.")
    ap.add_argument("--seg-coherent-upweight", type=float, default=1.0,
                    help=f"weight on COHERENT pixels (1.0 = inert). RISK-PROPORTIONAL: race start "
                         f"{SEG_COHERENT_UPWEIGHT_RACE_START:.6f} == its MEASURED stratified "
                         f"flip-risk lift (ddm_ti1 n600). Raced SEPARATELY from the spike scalar "
                         f"-- the two are asymmetric. Requires --seg-spike-reweight.")
    ap.add_argument("--seg-grad-q3-project", default="off", choices=PG1_SEG_GRAD_Q3_MODES,
                    help="PG1/#889: when on, the SEG loss gradient entering rendered frame_1 is "
                         "projected blockwise through sq1's exact float yuv6 pose-null projector "
                         "P (Q3). Forward pixels are unchanged. The JD1 pose path uses the "
                         "unwrapped render loss, so pose gradients are untouched. Args-only; "
                         "default off is byte-identical and carries no resumable state.")
    ap.add_argument("--margin-weight-temp", type=float, default=1.0,
                    help="inverse-margin reweight temperature (make_loss_fn margin_weight_temp)")
    ap.add_argument("--w-rate", type=float, default=0.0,
                    help="§3.4 rate-in-loss weight (stl1 row-8 LAW): differentiable token-entropy "
                         "surrogate added to the seg loss (0.0 = distortion-only control)")
    ap.add_argument("--rate-model", default="entropy", choices=("entropy", "smevr_surrogate"),
                    help="§3.4 'entropy' = marginal soft-histogram of the quantized token lattice; "
                         "'smevr_surrogate' = temporal-delta soft-histogram (the zlib-on-delta "
                         "coder surrogate)")
    ap.add_argument("--token-quant-anneal", default="off", choices=("off", "at_knee"),
                    help="§3.3(a) lattice annealing: 'off' = STE from birth; 'at_knee' = float "
                         "tokens until the CE->tau knee EVENT, then engage the STE (basin in "
                         "float, refine on the shipped lattice)")
    ap.add_argument("--composed-s-gate-subset", type=int, default=0,
                    help="§3.5 QA77-lite: >0 = at stage exits run the bounded pose+photometric "
                         "solve on this many pairs and record COMPOSED S (prices the co9 "
                         "sky/hood-freeze pose cost); 0 = off. VERDICT-level only")
    ap.add_argument("--composed-s-subset-ids", type=str, default=None,
                    help="§3.5 subset SELECTION (MAIN QA66): path to an .npy of pair indices "
                         "(the pose-mass TAIL, top-17 = 74.3%%) to use as the composed-S "
                         "subset; None = the first --composed-s-gate-subset pairs")
    ap.add_argument("--composed-s-delta-ref", type=str, default=None,
                    help="§3.5 ADOPTED (MAIN Option A): path to the GT-ideal delta reference "
                         ".npz (ddm_bc1_delta_baseline.py). When set, the composed-S runs the "
                         "DEGRADED DIRECTIONAL-DELTA instrument (Knee-A pose externality sign+"
                         "trend, NEVER an absolute S) on the ref table's knee-A tail")
    ap.add_argument("--byte-ledger-coder", default="smevr", choices=("smevr", "zlib"),
                    help="QA86(b) / census T5: coder used to PRICE the token stream for stage/"
                         "telemetry decisions. 'smevr' (default) = the SHIPPED r7 coder (decisions "
                         "match the archive); 'zlib' = the legacy temporal-delta surrogate (kept "
                         "for a byte-continuous live-burn resume). NEVER changes trained/shipped bytes")
    ap.add_argument("--renderer-head-mode", default="rgb",
                    choices=("rgb", "class_field", "class_field_photo"),
                    help="QA83 (census §4.1) OUTPUT-SPACE FACTORIZATION: 'rgb' = 3-ch control; "
                         "'class_field' = k=1 class scalar + fixed gray lift (1-luma-channel "
                         "ur-instance); 'class_field_photo' = k=2 (class + margin-slack luma "
                         "photometric). Lift = rule-118-free code; only k-ch tokens counted")
    ap.add_argument("--head-photo-slack-gain", type=float, default=0.05,
                    help="QA83/QA80: conservative fixed luma-slack gain for class_field_photo "
                         "(~13/255); the exact per-pixel band-lemma budget is the QA80 scorer step")
    ap.add_argument("--token-rowband-spec", type=str, default=None,
                    help="QA84 (census §4.2): RowBandGrammar spec .json path (or inline json) => "
                         "D8 base with bulk 2x2 tie (D16-effective) + op1 flip-band free at D8. "
                         "Requires --grid-downsample 8. None = uniform grid (control)")
    # ---- QA75 solve-frame distillation (ddm_dw1) ----
    ap.add_argument("--distill-field-cache", type=str, default=None,
                    help="QA75: path to the concatenated b2b teacher distill-logit cache "
                         "(P,5,384,512) f16. None = distill OFF (byte-identical control)")
    ap.add_argument("--distill-weight", type=float, default=0.0,
                    help="w_distill added to the seg loss; 0.0 = OFF. DERIVED rung = w_seg=100")
    ap.add_argument("--distill-temp", type=float, default=2.0,
                    help="KD temperature T (kd_logits form); Quantizr/PR95 T=2.0 provenance rung")
    ap.add_argument("--distill-form", default="kd_logits",
                    choices=("kd_logits", "margin_field", "argmax_ce"),
                    help="QA75 loss form; the ddm_dw1 mini-race winner (own-optimum law)")
    ap.add_argument("--distill-attack-temp", type=float, default=0.0,
                    help="emphasise low-GT-margin boundary annulus (attack set) via "
                         "exp(-GT_margin/temp); 0 = uniform. RACED dimension")
    ap.add_argument("--head-range-relax", default="off", choices=("off", "linear"),
                    help="Window C (MAIN charter): 'linear' adds a warm-start-equivalent trainable "
                         "output residual (de-saturate the sigmoid head); ADVISORY-NON-DEPLOYABLE "
                         "(breaks the E1 receiver). 'off' = rgb control (deployable, resume-safe)")
    # ---- ax1 Pool-A token-byte levers (ddm_pa1b #793; default-off => byte-identical control) ----
    ap.add_argument("--token-quant-margin-coupling", default="off", choices=("off", "on"),
                    help="ax1 §2a: 'on' allocates per-cell EFFECTIVE quant levels by the MEASURED "
                         "QA80 flip-distance field (rank-transform allocation LAW, no bare const); "
                         "fixed non-trainable level map => byte-identical resume. 'off' = scalar-L")
    ap.add_argument("--token-quant-coupling-field", type=str, default=None,
                    help="QA80 field custody dir (ddm_zb1_qa80_field_20260730); required when "
                         "--token-quant-margin-coupling on (fail-closed if the SSD tier is absent)")
    ap.add_argument("--token-quant-coupling-min-levels", type=int, default=0,
                    help="coarse-floor endpoint of the per-cell level ladder; 0 => derive "
                         "(quant_levels//4). base endpoint = --token-quant-levels")
    ap.add_argument("--token-delta-group-sparsity", default="off", choices=("off", "on"),
                    help="ax1 §4a: 'on' adds a group-L2 shrinkage on per-pair token deltas "
                         "(98.806%% image-stationarity has NO train-side force). Loss term => no param")
    ap.add_argument("--delta-sparsity-weight", type=float, default=0.0,
                    help="w_delta_sparsity (additive to the seg loss); 0.0 => OFF. The TRAIN-side "
                         "twin of the export-side ν null-snap (gc10 F2)")
    ap.add_argument("--delta-sparsity-engage", default="after_base_stability",
                    choices=("after_base_stability", "from_step_0"),
                    help="§7: 'after_base_stability' engages at the CE->tau knee EVENT (never "
                         "epoch-hardcoded); 'from_step_0' = the gc10 F2 ν-snap warm-start holder")
    ap.add_argument("--delta-sparsity-weight-field", default="uniform",
                    choices=("uniform", "xi_informed"),
                    help="ax1 §5: 'xi_informed' relaxes shrinkage on dynamic (lane/movable) cells, "
                         "tightens on the static mass (DERIVED from the QA80 winner-class field)")
    # ---- ddm_tp1 (#804) v9-line confound-cure telemetry PORT (vh1 row 7; burn-4 §3.1) ----
    ap.add_argument("--telemetry-v9-port", default="on", choices=("off", "on"),
                    help="v9 telemetry port: 'on' emits ADDITIVE read-only rows to "
                         "telemetry.jsonl — per-term loss_terms (#304), term_domination + "
                         "term_inert alarms (#321), a #404 positive-control sentinel, and "
                         "canonical lever_engage companions (Q7). DEFAULT 'on' because this is "
                         "score-neutral observability: the flag is threaded via args ONLY (never "
                         "TR1Config => config_hash + every checkpoint stay flag-invariant) and "
                         "new rows go to tlog/JSONL ONLY (never telemetry_tail, the checkpoint-"
                         "baked tail). READ-ONLY: no grad, no RNG advance (fixed dither bank + "
                         "isolated order_rng), no model/opt-state mutation. Use 'off' only for "
                         "explicit sealed-lineage log equivalence checks.")
    # ---- ddm_op2 (OP2-1) optimizer-state persistence — #824 arm C, the BUILD ----
    ap.add_argument("--persist-optimizer-state", default="off", choices=("off", "on"),
                    help="'on' saves Adam moments into every training checkpoint (opt:: keys) "
                         "and restores them on --resume-from, so a window boundary is a NO-OP on "
                         "trained bytes (MEASURED: restore reproduces an uninterrupted run to "
                         "max-abs-diff 0.0; reset diverges 5.5e-2). Closes #824 arm C. DEFAULT "
                         "'off' => BYTE-IDENTICAL trained/checkpoint bytes to every pre-OP2-1 "
                         "run: the flag is threaded via args ONLY (never TR1Config => "
                         "config_hash, ema_decay and the sealed ticket are all flag-invariant, "
                         "the telemetry_v9_port precedent), and with it off save_checkpoint "
                         "writes no opt:: keys at all. It is OFF rather than ON because a LIVE "
                         "sealed chain is mid-flight and switching the reset arm underneath it "
                         "would make its own windows incommensurable -- turn it ON for the next "
                         "FROM-SCRATCH run, where it is worth ~218 of 666 epochs.")
    # ---- ddm_pt2 (2026-08-03) THE PORT: make_loss_fn forces already in TR1's call path ----
    # MEASURED: this trainer imports the SAME ``make_loss_fn`` the retired levelset trainer uses
    # (line ~33 / the call below) but passes 5 of its 18 parameters. Four fully-implemented seg
    # forces therefore sit INSIDE TR1's own loss graph, unreachable for one reason only: this
    # argparse never declared their flags. Porting them is threading, not new machinery -- the
    # ddm_tp2 row-3 precedent ("only the producer moves"), inverted: here only the PARAMETER moves.
    #
    # WHY THESE FOUR AND NOT ``margin_weight_fn``: focal / fisher-density / natural-grad fold into
    # ``seg_pixel_w`` (or transform ``seg_logits``), the surface EVERY seg form honors before the
    # mean -- including ``tau_softplus``, which the live burn lineage occupies ~100% of its epochs.
    # ``margin_weight_fn`` is owned by the already-registered DSL lever ``tr1_seg_margin_weight``;
    # EN1 wires its tau consumer and the guard above keeps any unhonored future form fail-closed.
    #
    # ARGS-ONLY, never TR1Config (the --persist-optimizer-state / --telemetry-v9-port precedent):
    # ``canonical_json`` is ``asdict(self)``, so a new config field would move ``config_hash`` for
    # EVERY run including the off case and break the sealed lineage's flag-invariance. Cost stated
    # rather than hidden: two runs that differ only in these flags share a config_hash, so the
    # ``ported_loss_forces`` telemetry row below (score-neutral => ALWAYS emitted) is what carries
    # their state into the run record.
    ap.add_argument("--seg-focal-gamma", type=float, default=0.0,
                    help="ddm_pt2 PORT of the retired-trainer SegFocalGamma lever: focal per-pixel "
                         "reweight (1-p_y)^gamma from the REALIZED softmax, stop-grad + mean-1 "
                         "renormalized, folded MULTIPLICATIVELY into seg_pixel_w (so it applies to "
                         "every seg form, tau_softplus included). 0.0 = OFF = byte-identical.")
    ap.add_argument("--fisher-density-weight", type=float, default=0.0,
                    help="ddm_pt2 PORT of FisherDensityWeight: blend lambda in [0,1] on the exact "
                         "registered law tr g = (1/2)sech^2(m/2) as a per-pixel seg weight "
                         "(fisher_curvature_equals_categorical_fisher_trace_caustic_v1, rho=0.978). "
                         "Mean-1 renormalized + stop-grad => reallocates the gradient BUDGET, does "
                         "not change the loss form. 0.0 = OFF = byte-identical.")
    ap.add_argument("--fisher-density-source", default="model", choices=("model", "gt"),
                    help="'model' = live signed top1-top2 margin of the realized logits (the "
                         "Fisher-NATURAL choice: the metric at the current point of the flow); "
                         "'gt' = the cached GT margin field (a stationary importance PRIOR). "
                         "Requires --fisher-density-weight > 0.")
    ap.add_argument("--head-natural-grad", default="off", choices=("off", "on"),
                    help="ddm_pt2 PORT of HeadNaturalGradient: forward-IDENTITY / backward-g+ "
                         "transform on seg_logits, so the seg descent direction becomes the Fisher "
                         "natural gradient of the categorical head (exact rank-4 closed form, O(K) "
                         "per pixel). Loss VALUE and every activation are unchanged; only the "
                         "backward pass is preconditioned. 'off' = byte-identical.")
    ap.add_argument("--head-natural-grad-eps", type=float, default=1e-3,
                    help="damping added to p before the 1/p division (near-degenerate simplex "
                         "corners). Requires --head-natural-grad on.")
    ap.add_argument("--tau-softplus-tau", type=float, default=0.3,
                    help="ddm_pt2 PORT: tau of the tau_softplus seg form, mean(tau*softplus(-m/tau)). "
                         "This trainer previously took make_loss_fn's default with no way to set it, "
                         "while the live burn lineage runs tau_softplus for ~100%% of its epochs -- "
                         "i.e. the ONE scalar shaping the live loss was unreachable. Default 0.3 == "
                         "that same default => byte-identical.")
    return ap


def assert_ported_force_scalars_have_their_gate(
    fisher_density_weight: float, fisher_density_source: str,
    head_natural_grad: str, head_natural_grad_eps: float) -> None:
    """ddm_pt2, same genus as ``assert_spike_scalars_have_their_gate``: a value flag whose gate is
    off is a SILENT no-op, and declared-on-but-inert is this file's own documented dominant defect.

    Fail closed BEFORE training rather than let a run believe a force it declared was active.
    """
    inert: list[str] = []
    if fisher_density_weight <= 0.0 and fisher_density_source != "model":
        inert.append(f"--fisher-density-source {fisher_density_source}")
    if head_natural_grad != "on" and float(head_natural_grad_eps) != 1e-3:
        inert.append(f"--head-natural-grad-eps {head_natural_grad_eps}")
    if inert:
        raise SystemExit(
            f"REFUSED: {inert} set without the gate that makes it read.\n"
            f"  --fisher-density-source is read ONLY when --fisher-density-weight > 0.\n"
            f"  --head-natural-grad-eps is read ONLY when --head-natural-grad on.\n"
            f"  Arm the gate, or drop the value flag -- a declared-but-ignored force is the "
            f"exact silent-artifact class this trainer already refuses for --margin-weighted-loss "
            f"and --seg-spike-downweight.")


def validate_tk1_consumer_args(args: Any) -> None:
    """Fail-closed guards for TK1 args-only consumers."""
    if args.pe3_conditioning_mode == "conditioning_only" and args.pe3_conditioning_cache is None:
        raise SystemExit("--pe3-conditioning-mode conditioning_only requires --pe3-conditioning-cache")
    if args.pe3_conditioning_mode == "off" and args.pe3_conditioning_cache is not None:
        raise SystemExit("--pe3-conditioning-cache was provided while --pe3-conditioning-mode off")
    if args.pe3_conditioning_cache is not None and not args.pe3_conditioning_cache.is_file():
        raise SystemExit(f"--pe3-conditioning-cache missing: {args.pe3_conditioning_cache}")
    if args.cheapdct4_pose_mode == "accounting" and args.cheapdct4_pose_cache is None:
        raise SystemExit("--cheapdct4-pose-mode accounting requires --cheapdct4-pose-cache")
    if args.cheapdct4_pose_mode == "off" and args.cheapdct4_pose_cache is not None:
        raise SystemExit("--cheapdct4-pose-cache was provided while --cheapdct4-pose-mode off")
    if args.cheapdct4_pose_cache is not None and not args.cheapdct4_pose_cache.is_file():
        raise SystemExit(f"--cheapdct4-pose-cache missing: {args.cheapdct4_pose_cache}")


def jd1_pose_finish_armed(args: Any) -> bool:
    """Whether JD1 may ever build a PoseNet graph for this run."""
    return str(args.jd1_pose_finish_mode) != "off"


def derive_jd1_stage_ema_decay(
    remaining_epochs: int,
    steps_per_epoch: int,
    *,
    horizon_epochs: int | None = None,
) -> tuple[float, str]:
    """JD3 stage-window EMA law: same LawRef, but U is the joint-pose window."""
    window_epochs = max(1, int(remaining_epochs))
    steps = max(1, int(steps_per_epoch))
    horizon = max(1, int(horizon_epochs if horizon_epochs is not None else window_epochs))
    row = resolve_scope_law("jd3_stage_ema_decay", {
        "remaining_epochs": window_epochs,
        "steps_per_epoch": steps,
        "run_geometry_hash": scope_law_geometry_hash(
            steps_per_epoch=steps,
            horizon_epochs=horizon,
            window_epochs=window_epochs,
        ),
    })
    return float(row["resolved_value"]), str(row["provenance"])


def refuse_declared_vs_resolved_jd1_ema_decay(
    declared_decay: float | None,
    resolved_decay: float,
    *,
    resolution_hash: str,
) -> None:
    """Fail closed when a literal EMA flag disagrees with the live scope law."""
    if declared_decay is None:
        return
    if abs(float(declared_decay) - float(resolved_decay)) <= 1e-12:
        return
    raise SystemExit(
        "REFUSED: --ema-decay declares a literal EMA decay that conflicts with "
        "the JD1 stage scope-law resolution "
        f"({float(declared_decay)} != {float(resolved_decay)}; "
        f"resolution_hash={resolution_hash}). Drop the literal "
        "or make it match the resolved-at-consumption value."
    )


def jd1_ema_tail_anchor_epoch(value: int) -> int | None:
    """Normalize the explicit tail-average anchor epoch, or None for default-off."""
    v = int(value)
    if v < JD1_EMA_TAIL_ANCHOR_OFF:
        raise ValueError("--jd1-ema-tail-anchor-epoch must be -1/off or >= 0")
    return None if v == JD1_EMA_TAIL_ANCHOR_OFF else v


def jd1_ema_initial_state(args: Any) -> dict[str, Any]:
    """Return JD1 EMA state keys only when the new mode needs checkpoint custody.

    The default geometric path returns an empty dict so old JD1 checkpoints remain
    byte-identical when the new flags are absent.
    """
    anchor = jd1_ema_tail_anchor_epoch(int(args.jd1_ema_tail_anchor_epoch))
    if str(args.jd1_ema_mode) == "geometric" and anchor is None:
        return {}
    return {
        "ema_mode": str(args.jd1_ema_mode),
        "ema_tail_anchor_epoch": anchor,
        "ema_tail_configured_anchor_epoch": anchor,
        "ema_tail_average_active": False,
        "ema_tail_update_count": 0,
        "ema_tail_anchor_global_step": None,
        "ema_tail_anchor_reason": None,
        "ema_tail_last_live_weight": None,
    }


def jd1_ema_checkpoint_payload(args: Any, state: Mapping[str, Any]) -> dict[str, Any]:
    """Checkpoint-resumable JD1 EMA mode payload.

    Geometric default stays absent unless a resumed/new checkpoint already carries
    one of these keys. Tail-average mode persists mode, anchor, and update count.
    """
    carries_tail_state = any(k in state for k in JD1_EMA_TAIL_STATE_KEYS)
    if str(args.jd1_ema_mode) == "geometric" and not carries_tail_state:
        return {}
    anchor = jd1_ema_tail_anchor_epoch(int(args.jd1_ema_tail_anchor_epoch))
    return {
        "ema_mode": str(state.get("ema_mode", args.jd1_ema_mode)),
        "ema_tail_anchor_epoch": state.get("ema_tail_anchor_epoch", anchor),
        "ema_tail_configured_anchor_epoch": state.get(
            "ema_tail_configured_anchor_epoch", anchor),
        "ema_tail_average_active": bool(state.get("ema_tail_average_active", False)),
        "ema_tail_update_count": int(state.get("ema_tail_update_count", 0)),
        "ema_tail_anchor_global_step": state.get("ema_tail_anchor_global_step"),
        "ema_tail_anchor_reason": state.get("ema_tail_anchor_reason"),
        "ema_tail_last_live_weight": state.get("ema_tail_last_live_weight"),
    }


def jd1_ema_tail_average_active(state: Mapping[str, Any]) -> bool:
    return (
        str(state.get("ema_mode", "geometric")) == "plateau_tail_average"
        and bool(state.get("ema_tail_average_active", False))
    )


def jd1_ema_tail_average_live_weight(updates_since_anchor: int) -> float:
    """Polyak live-sample weight after a reset-to-live anchor.

    ``updates_since_anchor`` is the count already folded into the mean after the
    anchor sample. The next settled live iterate gets weight 1/(k+2), so the
    anchor live weights remain sample 0 of the growing horizon.
    """
    k = int(updates_since_anchor)
    if k < 0:
        raise ValueError("updates_since_anchor must be >= 0")
    row = resolve_scope_law("jd1_plateau_tail_average_ema", {"updates_since_anchor": k})
    return float(row["resolved_value"])


def jd1_ema_gate_basis_label(
    *, global_step: int, ema_warmup_updates: int, state: Mapping[str, Any]
) -> str:
    if jd1_ema_tail_average_active(state):
        return "ema_tail_average"
    return "ema_shadow" if int(global_step) >= int(ema_warmup_updates) else "live_ema_warmup"


def jd1_forced_resume_start_epoch(
    *,
    saved_epoch: int,
    checkpoint_tail: Sequence[dict[str, Any]],
    force_reanchor_on_resume: bool,
) -> tuple[int, dict[str, Any] | None]:
    """Return the resume epoch, preserving legacy saved_epoch+1 unless forced.

    Final checkpoints are written with an exclusive epoch, while intra-stage checkpoints
    use the measured epoch.  JD4 force mode is a continuation-only repair: it may use the
    parent telemetry tail to recover the next training epoch for the new window geometry.
    """
    legacy_start = int(saved_epoch) + 1
    if not force_reanchor_on_resume:
        return legacy_start, None
    tail_epochs = [
        int(row["epoch"])
        for row in checkpoint_tail
        if isinstance(row, dict) and row.get("epoch") is not None
    ]
    if not tail_epochs:
        return legacy_start, None
    tail_next = max(tail_epochs) + 1
    if tail_next >= legacy_start:
        return legacy_start, None
    return tail_next, {
        "event": "jd1_force_resume_epoch_reanchor",
        "saved_epoch": int(saved_epoch),
        "legacy_start_epoch": int(legacy_start),
        "tail_last_epoch": int(max(tail_epochs)),
        "forced_start_epoch": int(tail_next),
        "score_claim": False,
    }


def jd1_lr_parent_telemetry_path(resume_from: Path | str | None) -> Path | None:
    if resume_from is None:
        return None
    p = Path(resume_from)
    if p.parent.name == "checkpoints":
        return p.parent.parent / "telemetry.jsonl"
    return p.parent / "telemetry.jsonl"


def jd1_lr_epoch_signal_from_telemetry(telemetry_path: Path | str) -> tuple[list[float], str]:
    """Extract the terminal JD1 loss signal used for LR-tail damping.

    Pose-itemized loss rows win when present.  The jd5 telemetry did not itemize a pose
    loss term, so the fallback is the active joint-pose-finish epoch loss over the same
    window, explicitly labelled in the returned source.
    """
    path = Path(telemetry_path)
    pose_values: list[float] = []
    epoch_values: list[float] = []
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("event") == "loss_terms" and row.get("tr1_stage") == "joint_pose_finish":
                terms = row.get("terms")
                if isinstance(terms, Mapping):
                    for key, value in terms.items():
                        if "pose" in str(key) and isinstance(value, (int, float)):
                            pose_values.append(float(value))
            if row.get("event") == "epoch" and row.get("jd1_pose_finish_active"):
                value = row.get("ep_loss")
                if isinstance(value, (int, float)):
                    epoch_values.append(float(value))
    if pose_values:
        return pose_values, "loss_terms.terms.*pose*"
    return epoch_values, "epoch.ep_loss[jd1_pose_finish_active]"


def jd1_lr_tail_oscillation_stats(values: Sequence[float], tail_epochs: int) -> dict[str, Any]:
    vals = [float(v) for v in values if np.isfinite(float(v))]
    if not vals:
        raise ValueError("JD1 LR anneal cannot derive from an empty telemetry signal")
    tail = vals[-max(1, int(tail_epochs)):]
    arr = np.asarray(tail, dtype=np.float64)
    mean = float(np.mean(arr))
    sd = float(np.std(arr))
    half_range = float((np.max(arr) - np.min(arr)) / 2.0)
    diffs = np.diff(arr)
    nonzero = [float(d) for d in diffs if float(d) != 0.0]
    sign_changes = sum(1 for a, b in zip(nonzero, nonzero[1:]) if (a < 0) != (b < 0))
    denom = abs(mean) if mean != 0.0 else 1.0
    return {
        "n": int(arr.size),
        "mean": mean,
        "sd": sd,
        "half_range": half_range,
        "rel_half_range": float(half_range / denom),
        "sign_changes": int(sign_changes),
        "sign_change_denominator": max(0, len(nonzero) - 1),
    }


def derive_jd1_lr_tail_schedule(
    *,
    base_lr: float,
    start_epoch: int,
    end_epoch: int,
    steps_per_epoch: int,
    beta2: float,
    active_ema_decay: float,
    resume_from: Path | str | None,
    explicit_final_frac: float = JD1_LR_FINAL_FRAC_DERIVED,
) -> dict[str, Any]:
    """Derive the ddm_la1 terminal LR anneal from measured parent-window telemetry.

    The horizon is the larger of the optimizer beta2 variance-memory window and the
    active EMA memory window, both converted into epochs at the live batch geometry.
    The derived final fraction is ``sd / (sd + half_range)`` over that tail window:
    a flat tail leaves the LR unchanged, while a large alternating half-range relative
    to its settled dispersion damps the terminal live iterate.
    """
    if float(base_lr) <= 0.0:
        raise ValueError("base_lr must be > 0")
    if int(end_epoch) <= int(start_epoch):
        raise ValueError("end_epoch must be greater than start_epoch")
    if int(steps_per_epoch) <= 0:
        raise ValueError("steps_per_epoch must be > 0")
    if not (0.0 <= float(beta2) < 1.0):
        raise ValueError("beta2 must be in [0,1)")
    if not (0.0 <= float(active_ema_decay) < 1.0):
        raise ValueError("active_ema_decay must be in [0,1)")
    if float(explicit_final_frac) < 0.0 or float(explicit_final_frac) >= 1.0:
        raise ValueError("explicit_final_frac must be 0.0 (derive) or in (0,1)")

    beta2_epochs = int(math.ceil(
        JD1_LR_DERIVATION_TIME_CONSTANTS / max(1.0 - float(beta2), 1e-12)
        / int(steps_per_epoch)
    ))
    ema_epochs = int(math.ceil(
        JD1_LR_DERIVATION_TIME_CONSTANTS / max(1.0 - float(active_ema_decay), 1e-12)
        / int(steps_per_epoch)
    ))
    window_epochs = int(end_epoch) - int(start_epoch)
    tail_epochs = max(1, min(window_epochs, max(beta2_epochs, ema_epochs)))
    onset_epoch = int(end_epoch) - tail_epochs

    telemetry_path = jd1_lr_parent_telemetry_path(resume_from)
    if telemetry_path is None or not telemetry_path.exists():
        raise ValueError(
            "JD1 LR derived_tail requires --resume-from whose parent run has telemetry.jsonl"
        )
    signal, signal_source = jd1_lr_epoch_signal_from_telemetry(telemetry_path)
    stats = jd1_lr_tail_oscillation_stats(signal, tail_epochs)
    if float(explicit_final_frac) > 0.0:
        final_frac = float(explicit_final_frac)
        final_frac_source = "explicit"
    elif stats["sd"] + stats["half_range"] > 0.0:
        final_frac = float(stats["sd"] / (stats["sd"] + stats["half_range"]))
        final_frac_source = "derived_sd_over_sd_plus_half_range"
    else:
        final_frac = 1.0
        final_frac_source = "derived_flat_tail_no_damping"
    return {
        "schema": JD1_LR_ANNEAL_SCHEMA,
        "mode": "derived_tail",
        "base_lr": float(base_lr),
        "final_lr": float(base_lr) * float(final_frac),
        "final_frac": float(final_frac),
        "final_frac_source": final_frac_source,
        "start_epoch": int(start_epoch),
        "end_epoch": int(end_epoch),
        "onset_epoch": int(onset_epoch),
        "tail_epochs": int(tail_epochs),
        "steps_per_epoch": int(steps_per_epoch),
        "beta2": float(beta2),
        "beta2_memory_epochs_c2": int(beta2_epochs),
        "active_ema_decay": float(active_ema_decay),
        "active_ema_memory_epochs_c2": int(ema_epochs),
        "telemetry_path": str(telemetry_path),
        "signal_source": signal_source,
        "oscillation": stats,
        "score_claim": False,
    }


def jd1_lr_at_epoch(epoch: int, schedule: Mapping[str, Any]) -> float:
    base_lr = float(schedule["base_lr"])
    onset = int(schedule["onset_epoch"])
    end = int(schedule["end_epoch"])
    final_frac = float(schedule["final_frac"])
    if int(epoch) < onset:
        return base_lr
    if end <= onset:
        return base_lr * final_frac
    progress = min(1.0, max(0.0, (int(epoch) - onset + 1) / float(end - onset)))
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    frac = final_frac + (1.0 - final_frac) * cosine
    return float(base_lr * frac)


def tr1_muon_finisher_param_filter(name: str, weight: Any) -> bool:
    """Route TR1 renderer weight tensors to Muon; tokens/vectors/biases stay Adam.

    TR1's renderer trainables are convolution tensors (``w_*`` for plain, ``s_*``
    supermask logits for lotto). MLX Muon's own implementation flattens 4-D
    convolution filters into 2-D matrices before Newton-Schulz, so these are the
    renderer matrix group. Token grids, biases, per-channel gains, PE3 gates, and
    any 0-D/1-D leaves stay on the Adam fallback.
    """
    if int(getattr(weight, "ndim", 0)) < 2:
        return False
    low = str(name).lower()
    return any(low.startswith(prefix) for prefix in TR1_MUON_RENDERER_WEIGHT_PREFIXES)


def _tree_items_plain(tree: Any, prefix: str = "") -> list[tuple[str, Any]]:
    """Small tree flattener for pure tests and split telemetry without importing MLX utils."""
    if isinstance(tree, Mapping):
        out: list[tuple[str, Any]] = []
        for key, value in tree.items():
            child = str(key) if not prefix else f"{prefix}.{key}"
            out.extend(_tree_items_plain(value, child))
        return out
    if isinstance(tree, (list, tuple)):
        out = []
        for idx, value in enumerate(tree):
            child = str(idx) if not prefix else f"{prefix}.{idx}"
            out.extend(_tree_items_plain(value, child))
        return out
    return [(prefix, tree)]


def tr1_muon_adam_split_counts(params: Any) -> tuple[int, int]:
    """Count ``(Muon leaves, Adam leaves)`` under the TR1 finisher partition."""
    n_muon = n_adam = 0
    for name, leaf in _tree_items_plain(params):
        if tr1_muon_finisher_param_filter(name, leaf):
            n_muon += 1
        else:
            n_adam += 1
    return n_muon, n_adam


def derive_jd1_muon_momentum(beta1: float) -> tuple[float, str]:
    """Muon momentum derived from the outgoing TR1 Adam first-moment time constant."""
    if not 0.0 <= float(beta1) < 1.0:
        raise ValueError("beta1 must be in [0,1)")
    return (
        float(beta1),
        "DERIVED from TR1 optimizer beta1: warm-started Muon v consumes Adam m, "
        "so preserving the first-moment decay keeps the boundary time constant "
        "instead of importing the witness 0.95 default",
    )


def build_tr1_jd1_muon_finisher_optimizer(
    *,
    muon_lr: float,
    adam_lr: float,
    muon_momentum: float,
    muon_ns_steps: int = JD1_MUON_FINISHER_NS_STEPS,
    muon_lr_final_frac: float = 1.0,
    muon_anneal_steps: int = 0,
    adam_bias_correction: bool = False,
) -> Any:
    """Build the TR1 JD1 Muon finisher optimizer.

    Reuses MLX's real ``optim.Muon`` Newton-Schulz implementation and
    ``optim.MultiOptimizer``. The filter is TR1-specific because this vehicle's
    renderer leaves are conv tensors/supermask logits, not ``*.weight`` module
    leaves from the older witness trainer.
    """
    if float(muon_lr) <= 0.0:
        raise ValueError("muon_lr must be > 0")
    if float(adam_lr) <= 0.0:
        raise ValueError("adam_lr must be > 0")
    if not 0.0 <= float(muon_momentum) < 1.0:
        raise ValueError("muon_momentum must be in [0,1)")
    if int(muon_ns_steps) < 1:
        raise ValueError("muon_ns_steps must be >= 1")
    if not 0.0 < float(muon_lr_final_frac) <= 1.0:
        raise ValueError("muon_lr_final_frac must be in (0,1]")
    if int(muon_anneal_steps) < 0:
        raise ValueError("muon_anneal_steps must be >= 0")

    import mlx.optimizers as optim

    if float(muon_lr_final_frac) < 1.0 and int(muon_anneal_steps) > 0:
        muon_learning_rate: Any = optim.cosine_decay(
            float(muon_lr),
            int(muon_anneal_steps),
            end=float(muon_lr) * float(muon_lr_final_frac),
        )
    else:
        muon_learning_rate = float(muon_lr)
    muon = optim.Muon(
        learning_rate=muon_learning_rate,
        momentum=float(muon_momentum),
        weight_decay=0.0,
        nesterov=True,
        ns_steps=int(muon_ns_steps),
    )
    adam = optim.Adam(learning_rate=float(adam_lr), bias_correction=bool(adam_bias_correction))
    return optim.MultiOptimizer([muon, adam], [tr1_muon_finisher_param_filter])


def seed_tr1_muon_momentum_from_adam(muon_opt: Any, old_adam_state: Any) -> int:
    """Seed initialized Muon ``v`` leaves from outgoing Adam ``m`` leaves."""
    from mlx.utils import tree_flatten, tree_unflatten

    m_map = {k[:-2]: v for (k, v) in tree_flatten(old_adam_state) if k.endswith(".m")}
    new_flat: list[tuple[str, Any]] = []
    n_seed = 0
    for key, value in tree_flatten(muon_opt.state):
        if key.endswith(".v"):
            src = m_map.get(key[:-2])
            if src is not None and tuple(src.shape) == tuple(value.shape):
                value = src.astype(value.dtype)
                n_seed += 1
        new_flat.append((key, value))
    if n_seed:
        muon_opt.state = tree_unflatten(new_flat)
        muon_opt._initialized = True
    return n_seed


def jd1_should_reanchor_stage_ema(args: Any, state: Mapping[str, Any], *, reason: str) -> bool:
    """Pure JD1 stage-EMA reanchor predicate for ticket and resume tests."""
    if args.jd1_ema_stage_scope != "window":
        return False
    if not (jd1_pose_finish_armed(args) and state.get("engaged")):
        return False
    if not bool(state.get("stage_ema_reanchored")):
        return True
    return bool(
        args.jd1_force_ema_reanchor_on_resume
        and str(reason) == "resume_inside_joint_pose_finish"
    )


def derive_jd1_realized_hold_max_retreats(value: int) -> tuple[int, str]:
    row = resolve_scope_law("jd3_max_retreats_a1_policy", {
        "explicit_max_retreats": int(value),
        "a1_consecutive_refuse": A1_CONSECUTIVE_REFUSE,
    })
    return int(row["resolved_value"]), str(row["provenance"])


def derive_jd1_realized_hold_pose_retreat(value: float) -> tuple[float, str]:
    row = resolve_scope_law("jd3_pose_retreat_bisection", {
        "explicit_pose_retreat": float(value),
    })
    return float(row["resolved_value"]), str(row["provenance"])


def derive_jd1_realized_hold_margin(args: Any, gate_row: dict[str, Any]) -> tuple[float, str]:
    row = resolve_scope_law("jd3_realized_hold_margin", {
        "explicit_margin": float(args.jd1_realized_hold_margin),
        "realized_gate_dseg_per_pair_sd": gate_row.get("realized_gate_dseg_per_pair_sd"),
        "realized_gate_pair_ids": gate_row.get("realized_gate_pair_ids"),
    })
    return float(row["resolved_value"]), str(row["provenance"])


def validate_jd1_pose_finish_args(args: Any) -> None:
    """Fail closed on JD1 value flags that would otherwise be declared-but-unread."""
    try:
        _tail_anchor = jd1_ema_tail_anchor_epoch(int(args.jd1_ema_tail_anchor_epoch))
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if int(args.jd1_pose_finish_start_epoch) < 0:
        raise SystemExit("--jd1-pose-finish-start-epoch must be >= 0")
    if float(args.jd1_w_pose) < 0.0:
        raise SystemExit("--jd1-w-pose must be >= 0")
    if float(args.jd1_pose_eps) <= 0.0:
        raise SystemExit("--jd1-pose-eps must be > 0")
    if float(args.jd1_seg_hold_weight) < 0.0:
        raise SystemExit("--jd1-seg-hold-weight must be >= 0")
    if float(args.jd1_seg_hold_floor) < 0.0:
        raise SystemExit("--jd1-seg-hold-floor must be >= 0")
    if float(args.jd1_seg_hold_margin) < 0.0:
        raise SystemExit("--jd1-seg-hold-margin must be >= 0")
    if float(args.jd1_realized_hold_margin) < 0.0:
        raise SystemExit("--jd1-realized-hold-margin must be >= 0")
    if int(args.jd1_realized_hold_max_retreats) < 0:
        raise SystemExit("--jd1-realized-hold-max-retreats must be >= 0")
    if (float(args.jd1_realized_hold_pose_retreat) < 0.0
            or float(args.jd1_realized_hold_pose_retreat) >= 1.0):
        raise SystemExit("--jd1-realized-hold-pose-retreat must be 0.0 or in (0,1)")
    if float(args.jd1_lr_final_frac) < 0.0 or float(args.jd1_lr_final_frac) >= 1.0:
        raise SystemExit("--jd1-lr-final-frac must be 0.0 (derive) or in (0,1)")

    inert: list[str] = []
    if not jd1_pose_finish_armed(args):
        if int(args.jd1_pose_finish_start_epoch) != 0:
            inert.append("--jd1-pose-finish-start-epoch")
        if float(args.jd1_w_pose) != 0.0:
            inert.append("--jd1-w-pose")
        if float(args.jd1_pose_eps) != 1e-8:
            inert.append("--jd1-pose-eps")
        if float(args.jd1_seg_hold_weight) != 0.0:
            inert.append("--jd1-seg-hold-weight")
        if args.jd1_seg_hold_floor_source != "off":
            inert.append("--jd1-seg-hold-floor-source")
        if float(args.jd1_seg_hold_floor) != 0.0:
            inert.append("--jd1-seg-hold-floor")
        if float(args.jd1_seg_hold_margin) != 0.0:
            inert.append("--jd1-seg-hold-margin")
        if args.jd1_seg_hold_space != "loss":
            inert.append("--jd1-seg-hold-space")
        if float(args.jd1_realized_hold_margin) != 0.0:
            inert.append("--jd1-realized-hold-margin")
        if float(args.jd1_realized_hold_pose_retreat) != 0.0:
            inert.append("--jd1-realized-hold-pose-retreat")
        if int(args.jd1_realized_hold_max_retreats) != 0:
            inert.append("--jd1-realized-hold-max-retreats")
        if args.jd1_ema_stage_scope != "off":
            inert.append("--jd1-ema-stage-scope")
        if args.jd1_ema_mode != "geometric":
            inert.append("--jd1-ema-mode")
        if _tail_anchor is not None:
            inert.append("--jd1-ema-tail-anchor-epoch")
        if args.jd1_live_gate_telemetry != "off":
            inert.append("--jd1-live-gate-telemetry")
        if args.jd1_force_ema_reanchor_on_resume:
            inert.append("--jd1-force-ema-reanchor-on-resume")
        if args.jd1_lr_anneal != "off":
            inert.append("--jd1-lr-anneal")
        if float(args.jd1_lr_final_frac) != JD1_LR_FINAL_FRAC_DERIVED:
            inert.append("--jd1-lr-final-frac")
        if args.jd1_finisher != "off":
            inert.append("--jd1-finisher")
        if inert:
            raise SystemExit(
                "REFUSED: JD1 value flags set while --jd1-pose-finish-mode off: "
                f"{sorted(inert)}. Arm --jd1-pose-finish-mode joint_loss or drop the values.")
        return

    if float(args.jd1_w_pose) <= 0.0:
        raise SystemExit("--jd1-pose-finish-mode joint_loss requires --jd1-w-pose > 0")
    if args.jd1_ema_mode == "geometric":
        if _tail_anchor is not None:
            raise SystemExit(
                "--jd1-ema-tail-anchor-epoch requires --jd1-ema-mode plateau_tail_average")
    elif args.jd1_ema_mode == "plateau_tail_average":
        if args.jd1_ema_stage_scope != "window":
            raise SystemExit(
                "--jd1-ema-mode plateau_tail_average requires --jd1-ema-stage-scope window")
        if _tail_anchor is None:
            raise SystemExit(
                "--jd1-ema-mode plateau_tail_average requires "
                "--jd1-ema-tail-anchor-epoch >= 0")
    if (args.jd1_pose_finish_engage_on == "start_epoch"
            and int(args.jd1_pose_finish_start_epoch) <= 0):
        raise SystemExit("--jd1-pose-finish-engage-on start_epoch requires a positive start epoch")
    if args.jd1_seg_hold_space == "realized":
        if float(args.jd1_seg_hold_weight) <= 0.0:
            raise SystemExit("--jd1-seg-hold-space realized requires --jd1-seg-hold-weight > 0")
        if args.jd1_live_gate_telemetry != "on":
            raise SystemExit("--jd1-seg-hold-space realized requires --jd1-live-gate-telemetry on")
    if args.jd1_force_ema_reanchor_on_resume:
        if args.resume_from is None:
            raise SystemExit("--jd1-force-ema-reanchor-on-resume requires --resume-from")
        if args.jd1_ema_stage_scope != "window":
            raise SystemExit(
                "--jd1-force-ema-reanchor-on-resume requires --jd1-ema-stage-scope window")
    if float(args.jd1_seg_hold_weight) > 0.0:
        if args.jd1_seg_hold_floor_source == "off":
            raise SystemExit("--jd1-seg-hold-weight requires --jd1-seg-hold-floor-source != off")
        if (args.jd1_seg_hold_floor_source == "explicit"
                and float(args.jd1_seg_hold_floor) <= 0.0):
            raise SystemExit("--jd1-seg-hold-floor-source explicit requires --jd1-seg-hold-floor > 0")
        if args.jd1_seg_hold_floor_source == "checkpoint_tail_ep_loss" and args.resume_from is None:
            raise SystemExit("--jd1-seg-hold-floor-source checkpoint_tail_ep_loss requires --resume-from")
    elif args.jd1_seg_hold_floor_source != "off" or float(args.jd1_seg_hold_floor) != 0.0:
        raise SystemExit("JD1 seg-hold floor flags require --jd1-seg-hold-weight > 0")
    if args.jd1_lr_anneal == "derived_tail":
        if args.resume_from is None:
            raise SystemExit("--jd1-lr-anneal derived_tail requires --resume-from")
    elif float(args.jd1_lr_final_frac) != JD1_LR_FINAL_FRAC_DERIVED:
        raise SystemExit("--jd1-lr-final-frac requires --jd1-lr-anneal derived_tail")
    if args.jd1_finisher == "muon":
        if args.resume_from is None:
            raise SystemExit("--jd1-finisher muon is Case-B boundary-only and requires --resume-from")
        if args.jd1_pose_finish_engage_on != "start_epoch":
            raise SystemExit(
                "--jd1-finisher muon requires --jd1-pose-finish-engage-on start_epoch so "
                "the optimizer switch is a terminal boundary action, not a mid-window default")


def jd1_pose_finish_should_engage(args: Any, *, epoch: int, stage: str) -> bool:
    """The JD1 stage predicate, kept pure for tests and ticket validation."""
    if not jd1_pose_finish_armed(args):
        return False
    if int(epoch) < int(args.jd1_pose_finish_start_epoch):
        return False
    if args.jd1_pose_finish_engage_on == "post_knee":
        return str(stage) != SEG_TRUNK_CE_STAGE
    return True


def jd1_resolve_seg_hold_floor(
    args: Any,
    *,
    ep_losses: Sequence[float],
    checkpoint_tail: Sequence[dict[str, Any]],
) -> float | None:
    """Resolve the JD1 seg-hold floor at the exact pose-engagement boundary."""
    if float(args.jd1_seg_hold_weight) <= 0.0:
        return None
    source = str(args.jd1_seg_hold_floor_source)
    if source == "explicit":
        floor = float(args.jd1_seg_hold_floor)
    elif source == "last_pre_pose_epoch_loss":
        if not ep_losses:
            raise RuntimeError("JD1 seg-hold floor source last_pre_pose_epoch_loss has no local history")
        floor = float(ep_losses[-1])
    elif source == "checkpoint_tail_ep_loss":
        floor = math.nan
        for row in reversed(tuple(checkpoint_tail)):
            if isinstance(row, dict) and row.get("ep_loss") is not None:
                floor = float(row["ep_loss"])
                break
        if not math.isfinite(floor):
            raise RuntimeError("JD1 seg-hold floor source checkpoint_tail_ep_loss found no ep_loss")
    else:
        raise RuntimeError(f"JD1 seg-hold floor source {source!r} is not armed")
    if not math.isfinite(floor) or floor < 0.0:
        raise RuntimeError(f"JD1 seg-hold floor resolved invalid value {floor!r}")
    return floor


def main() -> int:
    args = build_argparser().parse_args()
    if args.verdict_chunk > 120:
        raise SystemExit("--verdict-chunk must be <= 120 (charter n600 chunk rule)")
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    deadline = started + float(args.max_wall_minutes) * 60.0

    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten

    from experiments.train_witness_realized_through_R_mlx import make_loss_fn
    from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap
    from tac.boundary_math.seg_core import load_real_segnet
    from tac.local_acceleration.mlx_scorer_adapters import (
        load_mlx_distortion_scorer_adapter_from_upstream,
    )

    steps_per_epoch = max(1, args.num_pairs // max(1, args.batch_pairs))
    total_updates = args.epochs * steps_per_epoch
    if args.ema_decay is not None:
        ema_decay, ema_prov = float(args.ema_decay), f"EXPLICIT --ema-decay {args.ema_decay}"
    else:
        ema_decay, ema_prov = derive_ema_decay(total_updates)

    cfg = TR1Config(
        variant=args.variant, num_pairs=args.num_pairs, grid_downsample=args.grid_downsample,
        code_width=args.code_width, renderer_width=args.renderer_width,
        token_quant_levels=args.token_quant_levels, seed=args.seed, lotto_seed=args.lotto_seed,
        lotto_mask_density_init=args.lotto_mask_density_init, seg_form_start=args.seg_form_start,
        w_seg=args.w_seg, lr=args.lr, batch_pairs=args.batch_pairs, epochs=args.epochs,
        gate_every=args.gate_every, ema_decay=ema_decay, ema_decay_provenance=ema_prov,
        token_temporal_mode=args.token_temporal_mode, token_ste=args.token_ste,
        class_weight_lane=args.class_weight_lane, margin_target=args.margin_target,
        token_init_mode=args.token_init_mode, basin_handoff=args.basin_handoff,
        token_cell_mask=args.token_cell_mask,
        margin_weighted_loss=args.margin_weighted_loss,
        seg_spike_reweight=args.seg_spike_reweight,
        seg_spike_downweight=args.seg_spike_downweight,
        seg_coherent_upweight=args.seg_coherent_upweight,
        margin_weight_temp=args.margin_weight_temp,
        w_rate=args.w_rate, rate_model=args.rate_model,
        token_quant_anneal=args.token_quant_anneal,
        composed_s_gate_subset=args.composed_s_gate_subset,
        composed_s_subset_ids=args.composed_s_subset_ids,
        composed_s_delta_ref=args.composed_s_delta_ref,
        byte_ledger_coder=args.byte_ledger_coder,
        renderer_head_mode=args.renderer_head_mode,
        head_photo_slack_gain=args.head_photo_slack_gain,
        token_rowband_spec=args.token_rowband_spec,
        distill_field_cache=args.distill_field_cache,
        distill_weight=args.distill_weight,
        distill_temp=args.distill_temp,
        distill_form=args.distill_form,
        distill_attack_temp=args.distill_attack_temp,
        head_range_relax=args.head_range_relax,
        token_quant_margin_coupling=args.token_quant_margin_coupling,
        token_quant_coupling_field=args.token_quant_coupling_field,
        token_quant_coupling_min_levels=args.token_quant_coupling_min_levels,
        token_delta_group_sparsity=args.token_delta_group_sparsity,
        delta_sparsity_weight=args.delta_sparsity_weight,
        delta_sparsity_engage=args.delta_sparsity_engage,
        delta_sparsity_weight_field=args.delta_sparsity_weight_field,
        lane_guard=args.lane_guard,
        lane_guard_budget_s=args.lane_guard_budget_s,
        lane_guard_eta=args.lane_guard_eta,
        lane_guard_lambda_step_cap=args.lane_guard_lambda_step_cap,
        lane_guard_lambda_max=args.lane_guard_lambda_max,
        lane_guard_born_weight=args.lane_guard_born_weight,
        lane_guard_margin_floor_weight=args.lane_guard_margin_floor_weight,
        existence_hinge_weight=args.existence_hinge_weight,
        existence_hinge_classes=args.existence_hinge_classes,
        existence_hinge_beta=args.existence_hinge_beta,
        existence_hinge_target=args.existence_hinge_target,
        existence_hinge_weight_policy=args.existence_hinge_weight_policy,
        existence_hinge_connectivity=args.existence_hinge_connectivity,
        lane_guard_lambda_init=args.lane_guard_lambda_init,
        lane_guard_ratchet=args.lane_guard_ratchet,
        lane_guard_ratchet_horizon=args.lane_guard_ratchet_horizon,
        adam_bias_correction=(args.adam_bias_correction == "on"),
    )
    (out_dir / "tr1_config.json").write_text(cfg.canonical_json() + "\n")
    telemetry_path = out_dir / "telemetry.jsonl"

    def tlog(row: dict[str, Any]) -> None:
        row = dict(row)
        row.setdefault("t_wall", time.monotonic() - started)
        with telemetry_path.open("a") as fh:
            fh.write(json.dumps(row, sort_keys=True) + "\n")

    tlog({"event": "start", "pointer": POINTER_LINE, "score_claim": False,
          "evidence_axis": "[macOS-CPU/MLX advisory]", "config_hash": cfg.config_hash(),
          "cfg": asdict(cfg), "pid": os.getpid()})

    try:
        parse_tr1_birth_seed_classes(args.tr1_birth_seed_classes)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if float(args.tr1_birth_seed_weight) < 0.0:
        raise SystemExit("--tr1-birth-seed-weight must be >= 0")
    if float(args.tr1_birth_amplify_weight) < 0.0:
        raise SystemExit("--tr1-birth-amplify-weight must be >= 0")
    if float(args.tr1_birth_amplify_weight) > 0.0 and float(args.tr1_birth_seed_weight) <= 0.0:
        raise SystemExit("--tr1-birth-amplify-weight requires --tr1-birth-seed-weight > 0")
    validate_tk1_consumer_args(args)
    validate_jd1_pose_finish_args(args)
    _jd1_pose_finish_enabled = jd1_pose_finish_armed(args)
    resolved_scope_laws: list[dict[str, Any]] = []

    def _resolve_scope_law(name: str, inputs: dict[str, Any]) -> dict[str, Any]:
        row = resolve_scope_law(name, inputs)
        resolved_scope_laws.append(dict(row))
        tlog({"event": "scope_law_resolution", **row})
        return row

    # GT: memmapped lstars/margins from the shared frozen-authority cache; frozen CPU SegNet.
    lstars = open_stored_npy_memmap(args.gt_cache, "lstars")
    margins = open_stored_npy_memmap(args.gt_cache, "margins")
    if lstars.shape[0] < cfg.num_pairs:
        raise SystemExit(f"gt cache has {lstars.shape[0]} pairs < --num-pairs {cfg.num_pairs}")
    gt_poses = open_stored_npy_memmap(args.gt_cache, "gt_poses")
    if gt_poses.shape[0] < cfg.num_pairs or gt_poses.shape[1] < 6:
        raise SystemExit(
            f"gt cache gt_poses shape {gt_poses.shape} incompatible with --num-pairs "
            f"{cfg.num_pairs} and PoseNet-6 targets")
    seg_cpu = load_real_segnet("cpu")

    # QA75 solve-frame distillation teacher (ddm_dw1): the concatenated b2b SegNet FIELD,
    # memmapped (P,5,384,512) f16. None => distill OFF => byte-identical to the control.
    distill_mm = None
    if cfg.distill_field_cache is not None and cfg.distill_weight != 0.0:
        distill_mm = np.load(cfg.distill_field_cache, mmap_mode="r")
        if distill_mm.shape[0] < cfg.num_pairs:
            raise SystemExit(
                f"distill field cache has {distill_mm.shape[0]} pairs < --num-pairs "
                f"{cfg.num_pairs}")
        if distill_mm.shape[1:] != (5, SEG_H, SEG_W):
            raise SystemExit(f"distill field cache shape {distill_mm.shape} != (P,5,384,512)")
        tlog({"event": "distill_field_ready", "cache": str(cfg.distill_field_cache),
              "shape": list(distill_mm.shape), "form": cfg.distill_form,
              "weight": cfg.distill_weight, "temp": cfg.distill_temp,
              "attack_temp": cfg.distill_attack_temp})

    mx.set_default_device(mx.gpu if args.mlx_device == "gpu" else mx.cpu)
    mx.random.seed(cfg.seed)

    # ddm_dt1 (#903) DETERMINISM: the default MLX-GPU R backward is an atomics/scatter
    # upsample VJP whose accumulation order varies run-to-run AND within one process
    # (MEASURED .omx/research/ddm_dt1_determinism_floor_20260803.md). ~1 ULP in the
    # gradient, but Adam's first step is essentially sign(g), so a 1-ULP gradient flip
    # becomes a full lr-sized parameter step and then amplifies chaotically. Gated =>
    # absent, this block is skipped and the run is byte-identical to every prior run.
    # NOT silent either way: the chosen mode is always logged (no-silent-guard rule).
    if args.deterministic_r:
        from tac.local_acceleration.metal_fused_r_operator import metal_fused_r_available

        from experiments.train_witness_realized_through_R_mlx import set_fused_r_kernel
        if not metal_fused_r_available():
            already_det = args.mlx_device == "cpu"
            raise SystemExit(
                "--deterministic-r REFUSED: the fused Metal R kernel needs a Metal GPU "
                f"default device (current: {mx.default_device()}). Refusing rather than "
                "silently running under a flag that promises determinism. "
                + ("--mlx-device cpu is ALREADY run-to-run bit-identical (MEASURED "
                   "2026-08-03, 3/3 windows, 41/41 arrays), so just DROP --deterministic-r."
                   if already_det else
                   "Either run on a Metal GPU, or use --mlx-device cpu (MEASURED "
                   "deterministic, slower), or drop the flag and accept the MEASURED "
                   "run-to-run floor."))
        set_fused_r_kernel(True)
    tlog({"event": "r_operator_mode", "deterministic_r": bool(args.deterministic_r),
          "mlx_device": args.mlx_device,
          "note": ("fused atomics-free Metal R VJP (run-to-run bit-identical)"
                   if args.deterministic_r else
                   "default mx.vjp scatter backward: MEASURED non-deterministic on GPU "
                   "(deterministic on --mlx-device cpu); lever A/Bs that RETRAIN are inside "
                   "the noise floor unless the delta clears it"),
          "receipt": ".omx/research/ddm_dt1_determinism_floor_20260803.md",
          "score_claim": False})

    # MLX scorer adapter (training-gradient device; NEVER a score) + canonical loss.
    upstream_root = str(Path(sys.modules["tac"].__file__).resolve().parents[2] / "upstream")
    adapter = load_mlx_distortion_scorer_adapter_from_upstream(upstream_root, device="cpu")
    tlog({"event": "realized_gate_dpose_channel_config",
          "port": "ddm_bd1_970",
          "status": "on",
          "default_on": True,
          "gate_cadence": "a1_gate",
          "basis": "normal a1_gate basis (EMA shadow after warmup; live only during warmup)",
          "live_basis_gate_pass_added": False,
          "gt_poses_loaded": True,
          "pose_path": "_apply_R(render(max(idx-1,0))) + _apply_R(render(idx)) + yuv12 "
                       "+ frozen MLX PoseNet first-6 MSE vs gt_poses[idx][:6]",
          "label": "advisory trend channel; n600 endpoint probe remains boundary authority",
          "score_claim": False})
    # §3.2 boundary-annulus form fix: 100% of realized flips sit in the bottom GT-margin decile
    # (sg1 §1.3) => reweight the per-pixel seg loss toward the small-margin boundary annulus.
    # ddm_tp2 row 2: fail closed BEFORE any training if the flag would be inert for a form
    # this run will occupy (declared-on + silently-ignored is the day's dominant genus).
    assert_margin_weighted_loss_is_honored(cfg.seg_form_start, cfg.margin_weighted_loss)
    # ddm_tp2 row 3: same genus -- a magnitude with no gate would be silently ignored.
    assert_spike_scalars_have_their_gate(cfg.seg_spike_reweight, cfg.seg_spike_downweight,
                                         cfg.seg_coherent_upweight)
    # ddm_pt2 row 4: same genus again -- a value with no gate would be silently ignored.
    assert_ported_force_scalars_have_their_gate(
        args.fisher_density_weight, args.fisher_density_source,
        args.head_natural_grad, args.head_natural_grad_eps)
    def _build_loss_fn_for_render(render_fn):
        return make_loss_fn(adapter, SEG_H, SEG_W, score_domain=True,
                            pose_eps=float(args.jd1_pose_eps),
                            seg_loss=cfg.seg_form_start,
                            margin_weighted=(cfg.margin_weighted_loss == "on"),
                            margin_weight_temp=cfg.margin_weight_temp,
                            render_fn=render_fn,
                            # ---- ddm_pt2 THE PORT (args-only; every default reproduces the
                            # pre-pt2 call exactly, so an unflagged run is byte-identical) ----
                            tau_softplus_tau=args.tau_softplus_tau,
                            focal_gamma=args.seg_focal_gamma,
                            fisher_density_weight=args.fisher_density_weight,
                            fisher_density_source=args.fisher_density_source,
                            head_natural_grad=(args.head_natural_grad == "on"),
                            head_natural_grad_eps=args.head_natural_grad_eps)

    loss_fn = _build_loss_fn_for_render(make_render_fn(args.seg_grad_q3_project))
    # PG1: if JD1 pose finish is active, pose loss must see the original frame_1 cotangent.
    # Split seg and pose through two canonical make_loss_fn instances instead of duplicating
    # scorer logic. When PG1 is off, aliasing preserves the legacy single-call path.
    pose_loss_fn = (
        loss_fn if args.seg_grad_q3_project == "off"
        else _build_loss_fn_for_render(make_render_fn("off"))
    )
    _pg1_projector = pose_null_projector_np()
    tlog({"event": "seg_grad_q3_project_config",
          "port": "ddm_pg1",
          "mode": args.seg_grad_q3_project,
          "active": bool(args.seg_grad_q3_project == "on"),
          "projector_sha256": hashlib.sha256(
              np.ascontiguousarray(_pg1_projector).tobytes()).hexdigest(),
          "projector_rank": int(np.linalg.matrix_rank(_pg1_projector)),
          "projector_idempotence_max_abs": float(np.max(np.abs(_pg1_projector @ _pg1_projector
                                                               - _pg1_projector))),
          "forward_identity": True,
          "pose_grad_path": ("legacy_single_loss_fn"
                             if args.seg_grad_q3_project == "off"
                             else "unwrapped_pose_loss_fn_when_jd1_pose_active"),
          "resumable_state": "none_args_only",
          "scope_law_status": "FORMALIZATION_PENDING_NOT_APPLICABLE_BINARY_FLAG",
          "scope_law_fire_order": ("register a T3_LIVE_ADAPTED law only if a future dynamic "
                                   "q3_first schedule or live scalar value is introduced"),
          "canonical_equation": "pose_null_subspace_is_ac_only_v1",
          "score_claim": False})
    # Score-neutral observability => ALWAYS emitted ("off is a tracked queue" / the read-only
    # telemetry rule). These forces are args-only, so config_hash cannot distinguish an armed run
    # from a control; this row is what carries their state into the run record. It also states
    # which forces are ACTIVE, so a reader never has to infer engagement from a flag's presence.
    tlog({"event": "ported_loss_forces", "port": "ddm_pt2",
          "tau_softplus_tau": float(args.tau_softplus_tau),
          "seg_focal_gamma": float(args.seg_focal_gamma),
          "fisher_density_weight": float(args.fisher_density_weight),
          "fisher_density_source": args.fisher_density_source,
          "head_natural_grad": args.head_natural_grad,
          "head_natural_grad_eps": float(args.head_natural_grad_eps),
          "active": sorted(n for n, on in (
              ("seg_focal_gamma", args.seg_focal_gamma > 0.0),
              ("fisher_density_weight", args.fisher_density_weight > 0.0),
              ("head_natural_grad", args.head_natural_grad == "on"),
              ("tau_softplus_tau_nondefault", float(args.tau_softplus_tau) != 0.3)) if on),
          "seg_form_start": cfg.seg_form_start,
          "reachable_seg_forms": sorted(reachable_seg_forms(cfg.seg_form_start)),
          "note": "PORT of make_loss_fn parameters this trainer already imports but never passed "
                  "(retired-trainer levers SegFocalGamma / FisherDensityWeight / "
                  "HeadNaturalGradient + the tau_softplus scalar). focal + fisher fold into "
                  "seg_pixel_w and natural-grad transforms seg_logits, so ALL of them are honored "
                  "by every seg form including tau_softplus. The separate tr1_seg_margin_weight "
                  "lever now also has a tau_softplus consumer (EN1). "
                  "Args-only => config_hash flag-invariant. score_claim=False",
          "score_claim": False})
    tlog({"event": "jd1_joint_pose_finish_config",
          "schema": JD1_POSE_FINISH_SCHEMA,
          "mode": args.jd1_pose_finish_mode,
          "engage_on": args.jd1_pose_finish_engage_on,
          "start_epoch": int(args.jd1_pose_finish_start_epoch),
          "w_pose": float(args.jd1_w_pose),
          "pose_eps": float(args.jd1_pose_eps),
          "seg_hold_weight": float(args.jd1_seg_hold_weight),
          "seg_hold_floor_source": args.jd1_seg_hold_floor_source,
          "seg_hold_floor": float(args.jd1_seg_hold_floor),
          "seg_hold_margin": float(args.jd1_seg_hold_margin),
          "seg_hold_space": args.jd1_seg_hold_space,
          "realized_hold_margin": float(args.jd1_realized_hold_margin),
          "realized_hold_pose_retreat": float(args.jd1_realized_hold_pose_retreat),
          "realized_hold_max_retreats": int(args.jd1_realized_hold_max_retreats),
          "ema_stage_scope": args.jd1_ema_stage_scope,
          "ema_mode": args.jd1_ema_mode,
          "ema_tail_anchor_epoch": int(args.jd1_ema_tail_anchor_epoch),
          "force_ema_reanchor_on_resume": bool(args.jd1_force_ema_reanchor_on_resume),
          "live_gate_telemetry": args.jd1_live_gate_telemetry,
          "gt_poses_loaded": bool(gt_poses is not None),
          "active": False,
          "note": "JD1 is args-only: off preserves TR1Config/config_hash/checkpoint bytes; "
                  "joint_loss consumes the gt_poses memmap through make_loss_fn with "
                  "compute_pose=True only after the engagement predicate fires.",
          "score_claim": False})
    tlog({"event": "jd1_muon_finisher_config",
          "schema": JD1_FINISHER_SCHEMA,
          "mode": args.jd1_finisher,
          "active": False,
          "case_b_only": True,
          "requires_resume": bool(args.jd1_finisher == "muon"),
          "optimizer_split": "Muon(renderer conv/supermask tensors flattened by MLX) + Adam(rest)",
          "momentum_source": "TR1 Adam beta1" if args.jd1_finisher == "muon" else None,
          "lr_source": "ddm_la1 parent-tail final-frac law" if args.jd1_finisher == "muon" else None,
          "note": "Args-only default-off finisher. Muon is legal only as a resumed JD1 "
                  "terminal boundary action; the runtime refuses a launch where it would "
                  "first engage mid-window.",
          "score_claim": False})

    model = build_module(cfg)
    mx.eval(model.parameters())

    # lv1 Phase B — SOLVE-INIT (eu1 "teacher-to-packet": the solved object is an
    # INITIALIZATION ORACLE, never a shippable payload). Custody verdict (lv1 memo):
    # the q1-lineage C1/BOX solved frames exist only as 277.7M archive-form candidates
    # (ms2r_r3 stage_checkpoints/04_candidate); the canonical MATERIALIZABLE
    # solution-set member in this trainer's own data path is the GT frame itself
    # (d_seg == 0 through R by construction of the GT labels). Projection = bounded
    # L2 fit of (tokens + renderer) to bilinear-resized GT frame_1 at the render
    # plane — NO scorer in the pretrain loop (train-least: don't learn what is
    # solved). Skipped on resume (state comes from the checkpoint; P0 resumability).
    if cfg.token_init_mode == "solve_project" and args.resume_from is None:
        import torch
        import torch.nn.functional as F
        gt_f1 = open_stored_npy_memmap(args.gt_cache, "gt_f1")  # (P,874,1164,3) u8 camera
        tgt = np.empty((cfg.num_pairs, SEG_H, SEG_W, 3), dtype=np.uint8)
        for i in range(cfg.num_pairs):
            fr = torch.from_numpy(
                np.asarray(gt_f1[i], dtype=np.float32)).permute(2, 0, 1)[None]
            dn = F.interpolate(fr, size=(SEG_H, SEG_W), mode="bilinear",
                               align_corners=False)
            tgt[i] = dn[0].permute(1, 2, 0).clamp(0, 255).round().to(torch.uint8).numpy()
        tlog({"event": "solve_init_targets_ready", "pairs": int(cfg.num_pairs),
              "note": "GT frame_1 bilinear->render plane = the materializable "
                      "solution-set member (q1-lineage C1/BOX frames exist only as "
                      "277.7M archive-form candidates; custody in the lv1 memo)"})

        # v3 ANALYTIC projection (both gradient formulations MEASURED inadmissible:
        # v1 JOINT tokens+renderer L2 fit -> GELU-dead mean-image basin (custody
        # b_solveinit_v1_aborted_joint_pretrain_gelu_dead_basin: l2 frozen to 9 digits
        # from pretrain_epoch 2; downstream scorer gnorm ~1e-9, ep_loss flat);
        # v2 TOKENS-ONLY gradient fit through the frozen random decoder -> glacial
        # (measured dl2 ~2e-5/update at n4; injects ~nothing). verdict_scope:
        # FORMULATION for both; paradigm intact. v3 = zeroth-order chart projection:
        # tokens := area-mean downsample of the solution-set member into the lattice,
        # split as base = temporal mean (the 98.806% image-stationary static scene,
        # op1) + per-frame delta residual — the deterministic, dead-basin-impossible
        # projection. The renderer starts at its gelu-ALIVE zero-init either way; the
        # A/B measures GT-structured vs zero tokens as the starting description.
        gh, gw, D = cfg.grid_h, cfg.grid_w, cfg.grid_downsample
        ds = tgt.reshape(cfg.num_pairs, gh, D, gw, D, 3).astype(np.float32)
        ds = ds.mean(axis=(2, 4)) / 255.0 * 2.0 - 1.0  # (P, gh, gw, 3) in [-1, 1]
        tok = np.zeros((cfg.num_pairs, gh, gw, cfg.code_width), dtype=np.float32)
        nch = min(3, cfg.code_width)
        tok[..., :nch] = ds[..., :nch]
        if cfg.token_temporal_mode == "shared_base":
            base = tok.mean(axis=0)
            delta = np.clip(tok - base[None], -1.0, 1.0)
            model.tokens_base = mx.array(base)
            model.tokens_delta = mx.array(delta)
            tlog({"event": "solve_init_projected", "mode": "shared_base",
                  "base_absmax": float(np.abs(base).max()),
                  "delta_absmax": float(np.abs(delta).max()),
                  "delta_rms": float(np.sqrt(np.mean(delta ** 2)))})
        else:
            model.tokens = mx.array(np.clip(tok, -1.0, 1.0))
            tlog({"event": "solve_init_projected", "mode": "independent",
                  "tok_absmax": float(np.abs(tok).max())})
        mx.eval(model.parameters())
        save_checkpoint(out_dir / "checkpoints" / "stage_solve_init_pretrain.npz",
                        model=model,
                        ema={k: mx.array(v)
                             for k, v in tree_flatten(model.trainable_parameters())},
                        opt_state_flat=no_opt_state(
                            "solve-init pretrain uses its OWN local projection optimizer "
                            "and the block is `args.resume_from is None`-gated, so no "
                            "resume ever reads moments from it (ddm_op2 OP2-1)"),
                        epoch=-1, stage="solve_init_pretrain",
                        cfg=cfg, telemetry_tail=[])
        del tgt, ds, tok
        # Scorer-loop Adam moments are created FRESH below (warm-start re-anchor
        # law #517/#518); the EMA shadow initializes from the post-projection params
        # (fresh warmup window => live-basis gates until W, same as the control).

    birth_seed_amplify_weight = 0.0
    if float(args.tr1_birth_seed_weight) > 0.0:
        _birth_seed_summary = attach_tr1_birth_seed_bank(
            model,
            cfg,
            lstars,
            weight=float(args.tr1_birth_seed_weight),
            classes=args.tr1_birth_seed_classes,
            dilate_px=int(args.tr1_birth_seed_dilate_px),
            persist=args.tr1_birth_amplify_persist,
            apply_live_seed=(args.resume_from is None),
        )
        birth_seed_amplify_weight = float(args.tr1_birth_amplify_weight)
        tlog({"event": "tr1_birth_seed_init",
              **_birth_seed_summary,
              "amplify_weight": birth_seed_amplify_weight,
              "resume_from_checkpoint": bool(args.resume_from is not None),
              "note": "BI1 #924 builds p4x's previously-unbuilt TR1 seed/amplify BIRTH "
                      "path. Runtime flags are args-only so OFF preserves config_hash and "
                      "checkpoint bytes; ON initializes GT birth support in the token lattice "
                      "and adds a scorer-free anchor term. No SegNet/PoseNet forward is used "
                      "by this mechanism; score_claim=False."})

    pe3_conditioning_summary = None
    if args.pe3_conditioning_mode == "conditioning_only":
        try:
            pe3_conditioning_summary = attach_pe3_conditioning_bank(
                model, cfg, args.pe3_conditioning_cache
            )
        except Exception as exc:
            raise SystemExit(f"PE3 conditioning cache refused: {exc}") from exc
        tlog({"event": "pe3_conditioning_init",
              **pe3_conditioning_summary,
              "mode": args.pe3_conditioning_mode,
              "note": "TK1 consumes PE3 as conditioning-only prior channels. The grammar is "
                      "not a target replacement; learned per-mode gates can down-weight "
                      "generator_pair_bisector independently of depth_conditioned_curve. "
                      "score_claim=False."})

    cheapdct4_pose_accounting = None
    if args.cheapdct4_pose_mode == "accounting":
        try:
            cheapdct4_pose_accounting = load_cheapdct4_pose_accounting_cache(
                args.cheapdct4_pose_cache
            )
        except Exception as exc:
            raise SystemExit(f"cheapdct4 pose accounting cache refused: {exc}") from exc
        tlog({"event": "cheapdct4_pose_accounting_init",
              **cheapdct4_pose_accounting,
              "note": "TK1 decodes OD9 stage2_qcoeffs and reports OD9's measured n32 "
                      "pose term for composed-S accounting. This is not full in-loop "
                      "joint-descent consumption; score_claim=False."})

    # ddm_bp1 (#824) reset-race arm selector. REUSES the levelset trainer's already-unit-tested
    # gate (never reimplemented): _adam_bias_correction_for(beta2, reference_semantics=) returns
    # `reference_semantics` verbatim at MLX's default beta2=0.999, which is the only beta2 this
    # trainer uses (optim.Adam betas default [0.9, 0.999]). arm B => False == MLX's own default
    # => optimizer construction and every trained byte are IDENTICAL to a pre-#824 run (MEASURED
    # against the real optimizer in test_ddm_bp1_boundary_reset_race.py, not asserted).
    from experiments.train_levelset_witness_realized_through_R_mlx import (
        _adam_bias_correction_for,
    )
    from tac.optimization.reset_operator import boundary_impulse_epochs, resolve_arm_name

    _reset_arm = reset_arm_for(cfg)
    _bias_correction = _adam_bias_correction_for(
        RESET_ADAM_BETAS[1], reference_semantics=cfg.adam_bias_correction)
    if bool(_bias_correction) != bool(cfg.adam_bias_correction):  # defensive: gate must be exact
        raise SystemExit(
            f"#824 arm-selector contract broken: _adam_bias_correction_for returned "
            f"{_bias_correction} for adam_bias_correction={cfg.adam_bias_correction}")
    optimizer = optim.Adam(learning_rate=cfg.lr, bias_correction=_bias_correction)
    tlog({"event": "optimizer_arm",
          "arm": resolve_arm_name(_reset_arm), "reset_operator": _reset_arm.describe(),
          "bias_correction": bool(_bias_correction), "betas": list(RESET_ADAM_BETAS),
          "lr": cfg.lr, "steps_per_epoch": steps_per_epoch,
          "boundary_impulse_epochs_per_reset": boundary_impulse_epochs(
              BOUNDARY_IMPULSE_CONVERGENCE_STEPS, steps_per_epoch, RESET_ADAM_BETAS),
          "note": "arm B (bias_correction False) IS MLX's Adam default => trained bytes "
                  "identical to every pre-#824 run; arm B' removes the eta(t) reset impulse",
          "persist_optimizer_state": args.persist_optimizer_state,
          "score_claim": False})

    # ddm_op2 (OP2-1): the ONE place a training checkpoint's optimizer payload is resolved.
    # OFF (default) => `{}` => zero `opt::` keys => checkpoint bytes identical to every
    # pre-OP2-1 run, so the LIVE sealed chain is untouched. ON => #824 arm C.
    _persist_opt = (args.persist_optimizer_state == "on")

    def _opt_state() -> dict[str, np.ndarray]:
        if not _persist_opt:
            return no_opt_state("--persist-optimizer-state off (default): trained/checkpoint "
                                "bytes stay identical to every pre-OP2-1 run (#824 arm B)")
        return optimizer_state_to_flat(optimizer)

    # ddm_bp1 (#824) boundary instrument state (args-only, never TR1Config => trained bytes are
    # flag-invariant; the telemetry_v9_port precedent). Interval decomposition + the boundary_jump
    # row are FREE (derived from values already computed) and therefore default ON per the
    # "score-neutral observability is not gate-able" rule; only the positive-control RE-GATE costs
    # compute, so only THAT is behind --boundary-probe, and both arms set it identically.
    boundary_probe = (args.boundary_probe == "on")
    boundary_parent_tail: list[dict[str, Any]] = []
    boundary_parent_ema_decay: float | None = None
    boundary_ema_held = True          # a fresh (non-resume) run has no basis to drift from
    boundary_jump_emitted = False
    # ddm_op2 (OP2-2): the STRICT decay+basis verdict, known only once the first post-resume gate
    # has run (the basis is a property of the READING, not of the resume). None until then, so the
    # receipt can never present the decay-only leg as if it were the strict flag.
    boundary_strict_basis_held: bool | None = None
    boundary_gate_basis_held: bool | None = None
    boundary_parent_cfg_ema_decay: float | None = None

    active_ema_decay = float(cfg.ema_decay)
    active_ema_decay_provenance = str(cfg.ema_decay_provenance)
    jd1_effective_w_pose = float(args.jd1_w_pose)
    _jd1_realized_retreat, _jd1_realized_retreat_prov = (
        derive_jd1_realized_hold_pose_retreat(float(args.jd1_realized_hold_pose_retreat)))
    _jd1_realized_max_retreats, _jd1_realized_max_retreats_prov = (
        derive_jd1_realized_hold_max_retreats(int(args.jd1_realized_hold_max_retreats)))
    jd1_realized_hold_state: dict[str, Any] = {
        "active": bool(args.jd1_seg_hold_space == "realized"),
        "floor": None,
        "floor_epoch": None,
        "floor_gate_basis": None,
        "margin": None,
        "margin_provenance": None,
        "pose_retreat_factor": float(_jd1_realized_retreat),
        "pose_retreat_provenance": _jd1_realized_retreat_prov,
        "max_retreats": int(_jd1_realized_max_retreats),
        "max_retreats_provenance": _jd1_realized_max_retreats_prov,
        "retreats": 0,
        "history": [],
        "scope_law_resolution_hashes": [],
    }
    ema: dict[str, Any] = {k: mx.array(v) for k, v in tree_flatten(model.trainable_parameters())}
    start_epoch = 0
    stage = initial_stage_label(cfg.seg_form_start)
    jd1_pose_finish_state: dict[str, Any] = {
        "schema": JD1_POSE_FINISH_SCHEMA,
        "enabled": bool(_jd1_pose_finish_enabled),
        "engaged": False,
        "engaged_epoch": None,
        "engaged_stage": None,
        "engaged_global_step": None,
        "seg_hold_floor": None,
        "seg_hold_floor_source": args.jd1_seg_hold_floor_source,
        "seg_hold_margin": float(args.jd1_seg_hold_margin),
        "w_pose": float(args.jd1_w_pose),
        "effective_w_pose": float(jd1_effective_w_pose),
        "seg_hold_space": args.jd1_seg_hold_space,
        "realized_hold": dict(jd1_realized_hold_state),
        "ema_stage_scope": args.jd1_ema_stage_scope,
        "active_ema_decay": float(active_ema_decay),
        "active_ema_decay_provenance": active_ema_decay_provenance,
        "stage_ema_reanchored": False,
        "force_ema_reanchor_on_resume": bool(args.jd1_force_ema_reanchor_on_resume),
        "live_gate_telemetry": args.jd1_live_gate_telemetry,
        "resolved_scope_laws": list(resolved_scope_laws),
    }
    jd1_pose_finish_state.update(jd1_ema_initial_state(args))
    jd1_muon_schedule: dict[str, Any] | None = None
    jd1_finisher_active = False

    def _jd1_checkpoint_extra_meta() -> dict[str, Any] | None:
        if not (_jd1_pose_finish_enabled and jd1_pose_finish_state.get("engaged")):
            return None
        jd1_pose_finish_state.update({
            "effective_w_pose": float(jd1_effective_w_pose),
            "realized_hold": dict(jd1_realized_hold_state),
            "active_ema_decay": float(active_ema_decay),
            "active_ema_decay_provenance": active_ema_decay_provenance,
            "ema_stage_scope": args.jd1_ema_stage_scope,
            "force_ema_reanchor_on_resume": bool(args.jd1_force_ema_reanchor_on_resume),
            "live_gate_telemetry": args.jd1_live_gate_telemetry,
            "seg_hold_space": args.jd1_seg_hold_space,
            "resolved_scope_laws": list(resolved_scope_laws),
            "finisher": (dict(jd1_muon_schedule)
                         if jd1_muon_schedule is not None else {
                             "schema": JD1_FINISHER_SCHEMA,
                             "mode": args.jd1_finisher,
                             "active": False,
                         }),
            **jd1_ema_checkpoint_payload(args, jd1_pose_finish_state),
        })
        return {"jd1_pose_finish": dict(jd1_pose_finish_state)}

    if args.resume_from is not None:
        st = load_checkpoint(args.resume_from, model)
        ema = st["ema"]
        start_epoch = st["epoch"] + 1
        stage = st["meta"].get("stage", stage)
        _saved_jd1 = st["meta"].get("jd1_pose_finish")
        if _saved_jd1 is not None:
            if not isinstance(_saved_jd1, dict) or _saved_jd1.get("schema") != JD1_POSE_FINISH_SCHEMA:
                raise SystemExit("resume REFUSED: checkpoint JD1 pose-finish metadata schema differs")
            if bool(_saved_jd1.get("engaged")) and not _jd1_pose_finish_enabled:
                raise SystemExit(
                    "resume REFUSED: checkpoint is inside JD1 joint_pose_finish but this launch "
                    "sets --jd1-pose-finish-mode off")
            _saved_ema_mode = _saved_jd1.get("ema_mode")
            if (_saved_ema_mode is not None
                    and str(_saved_ema_mode) != str(args.jd1_ema_mode)):
                raise SystemExit(
                    "resume REFUSED: checkpoint JD1 EMA mode "
                    f"{_saved_ema_mode!r} != launch --jd1-ema-mode {args.jd1_ema_mode!r}")
            jd1_pose_finish_state.update(_saved_jd1)
            if _saved_jd1.get("effective_w_pose") is not None:
                jd1_effective_w_pose = float(_saved_jd1["effective_w_pose"])
            if isinstance(_saved_jd1.get("realized_hold"), dict):
                jd1_realized_hold_state.update(_saved_jd1["realized_hold"])
            if isinstance(_saved_jd1.get("resolved_scope_laws"), list):
                resolved_scope_laws[:] = [dict(row) for row in _saved_jd1["resolved_scope_laws"]]
            if _saved_jd1.get("active_ema_decay") is not None:
                active_ema_decay = float(_saved_jd1["active_ema_decay"])
                active_ema_decay_provenance = str(
                    _saved_jd1.get("active_ema_decay_provenance",
                                   active_ema_decay_provenance))
            jd1_pose_finish_state["force_ema_reanchor_on_resume"] = bool(
                args.jd1_force_ema_reanchor_on_resume)
            _saved_finisher = _saved_jd1.get("finisher")
            if isinstance(_saved_finisher, dict) and _saved_finisher.get("active"):
                if args.jd1_finisher != "muon":
                    raise SystemExit(
                        "resume REFUSED: checkpoint is inside the JD1 Muon finisher but "
                        "this launch sets --jd1-finisher off")
                jd1_muon_schedule = dict(_saved_finisher)
        if (stage == "joint_pose_finish" and _jd1_pose_finish_enabled
                and not jd1_pose_finish_state.get("engaged")):
            jd1_pose_finish_state.update({
                "engaged": True,
                "engaged_epoch": start_epoch,
                "engaged_stage": stage,
                "engaged_global_step": None,
                "seg_hold_floor": (float(args.jd1_seg_hold_floor)
                                   if args.jd1_seg_hold_floor_source == "explicit" else None),
                "effective_w_pose": float(jd1_effective_w_pose),
                "seg_hold_space": args.jd1_seg_hold_space,
                "realized_hold": dict(jd1_realized_hold_state),
                "ema_stage_scope": args.jd1_ema_stage_scope,
                "active_ema_decay": float(active_ema_decay),
                "active_ema_decay_provenance": active_ema_decay_provenance,
                "force_ema_reanchor_on_resume": bool(args.jd1_force_ema_reanchor_on_resume),
                "resolved_scope_laws": list(resolved_scope_laws),
            })
        if (jd1_pose_finish_state.get("engaged")
                and float(args.jd1_seg_hold_weight) > 0.0
                and jd1_pose_finish_state.get("seg_hold_floor") is None):
            raise SystemExit(
                "resume REFUSED: JD1 seg-hold is armed but the checkpoint lacks a latched "
                "seg_hold_floor")
        # Resume-registry hygiene (ddm_dw1 Window C, guard 7): a trainable param INTRODUCED
        # since the checkpoint (e.g. head_relax_gain) is absent from the loaded shadow. Backfill
        # BOTH the live param (load_checkpoint's model.update already leaves it at its init) and
        # the EMA shadow from the model's INIT value so live and shadow start warm-start-equivalent
        # (head_relax_gain init 0 => head == sigmoid(x)*255 at ep0). No-op when no new params
        # (Windows A/B) => byte-identical resume.
        model_init = dict(tree_flatten(model.trainable_parameters()))
        backfilled = [k for k in model_init if k not in ema]
        for k in backfilled:
            ema[k] = mx.array(model_init[k])
        # ddm_op2 (OP2-1): restore the Adam moments, so the boundary is a no-op on trained bytes
        # instead of a 16.167-epoch re-convergence impulse (#824 arm C; ddm_gd5 §3.6). The
        # payload already passed the resume-geometry guard inside load_checkpoint.
        _resume_opt_flat = st.get("opt_flat") or {}
        if (_persist_opt and _resume_opt_flat
                and any(str(k).startswith("states.") for k in _resume_opt_flat)):
            if jd1_muon_schedule is None:
                raise OptimizerStateRestoreError(
                    "optimizer-state restore REFUSED — checkpoint carries MultiOptimizer "
                    "state but no JD1 Muon finisher metadata. Refusing to guess the split.")
            optimizer = build_tr1_jd1_muon_finisher_optimizer(
                muon_lr=float(jd1_muon_schedule["muon_lr"]),
                adam_lr=float(jd1_muon_schedule["adam_lr"]),
                muon_momentum=float(jd1_muon_schedule["muon_momentum"]),
                muon_ns_steps=int(jd1_muon_schedule["muon_ns_steps"]),
                muon_lr_final_frac=float(jd1_muon_schedule.get("muon_lr_final_frac", 1.0)),
                muon_anneal_steps=int(jd1_muon_schedule.get("muon_anneal_steps", 0)),
                adam_bias_correction=bool(_bias_correction),
            )
            jd1_finisher_active = True
            tlog({"event": "jd1_muon_finisher_resume_optimizer",
                  "epoch": int(start_epoch),
                  "resume_from": str(args.resume_from),
                  "schema": JD1_FINISHER_SCHEMA,
                  "muon_lr": float(jd1_muon_schedule["muon_lr"]),
                  "adam_lr": float(jd1_muon_schedule["adam_lr"]),
                  "muon_momentum": float(jd1_muon_schedule["muon_momentum"]),
                  "muon_ns_steps": int(jd1_muon_schedule["muon_ns_steps"]),
                  "muon_lr_final_frac": float(
                      jd1_muon_schedule.get("muon_lr_final_frac", 1.0)),
                  "muon_anneal_steps": int(jd1_muon_schedule.get("muon_anneal_steps", 0)),
                  "score_claim": False})
        if _persist_opt:
            if _resume_opt_flat:
                tlog({**restore_optimizer_state(optimizer, model, _resume_opt_flat),
                      "epoch": start_epoch, "resume_from": str(args.resume_from)})
            else:
                # LOUD, never silent: persistence was REQUESTED but the parent has no moments,
                # so this boundary IS an arm-B zero reset and must not be reported as arm C.
                tlog({"event": "confound_alarm", "kind": "optimizer_state_absent_on_resume",
                      "epoch": start_epoch, "resume_from": str(args.resume_from),
                      "note": "--persist-optimizer-state on, but the parent checkpoint carries "
                              "no opt:: keys (written before OP2-1, or by a run with the flag "
                              "off). This boundary pays the full arm-B reset impulse "
                              "(16.167 epochs); every LATER boundary in this chain will not."})
        elif _resume_opt_flat:
            # The mirror case: the parent persisted moments and this run is discarding them.
            tlog({"event": "confound_alarm", "kind": "optimizer_state_discarded_on_resume",
                  "epoch": start_epoch, "resume_from": str(args.resume_from),
                  "moments_available": len(_resume_opt_flat),
                  "note": "the parent checkpoint CARRIES optimizer moments and this run is "
                          "resetting them (--persist-optimizer-state off) — a deliberate arm-B "
                          "boundary, named here so it is never mistaken for arm C."})
        # §3.3(a) resume-past-knee: if the STE anneal already engaged (we resume in a post-CE
        # stage), re-engage it so the token forward matches the checkpointed lattice state.
        if cfg.token_quant_anneal == "at_knee" and stage != "seg_trunk_ce":
            model._quant_engaged = True
        # ax1 §4a: if we resume PAST the base-stability knee, re-engage delta-sparsity so the
        # shrinkage force matches the checkpointed schedule state (resume-registry hygiene).
        if (cfg.token_delta_group_sparsity == "on"
                and cfg.delta_sparsity_engage == "after_base_stability"
                and stage != "seg_trunk_ce"):
            model._delta_sparsity_engaged = True
        # ddm_bp1 (#824): the BOUNDARY ANCHOR — the parent's last realized-gate reading and the
        # ema_decay it was read under. The gate reads the EMA SHADOW, so a parent/child decay
        # mismatch moves the instrument's own averaging length underneath the measurement
        # (derive_ema_decay consumes epochs*(num_pairs//batch_pairs) ⇒ an --epochs change alone
        # moves it: the burn ran U=49,950/60,450/70,950, a different decay at EVERY boundary).
        boundary_parent_tail = list(st["meta"].get("telemetry_tail") or [])
        _forced_start_epoch, _forced_start_row = jd1_forced_resume_start_epoch(
            saved_epoch=int(st["epoch"]),
            checkpoint_tail=boundary_parent_tail,
            force_reanchor_on_resume=bool(args.jd1_force_ema_reanchor_on_resume),
        )
        if _forced_start_row is not None:
            start_epoch = int(_forced_start_epoch)
            tlog({**_forced_start_row, "resume_from": str(args.resume_from)})
        _pcfg = st["meta"].get("cfg") or {}
        # ddm_gd4 G1 sister guard: the LOTTO bank is generated from lotto_seed and is
        # NOT a trainable param, so a seed change survives the structural geometry check
        # with identical shapes while making every trained supermask meaningless (the
        # masks index a different random bank). Fail closed on the one silent-wrong the
        # shape check cannot see.
        if cfg.variant == "lotto" and isinstance(_pcfg, dict) and "lotto_seed" in _pcfg:
            if int(_pcfg["lotto_seed"]) != int(cfg.lotto_seed):
                raise ResumeGeometryMismatch(
                    f"resume REFUSED — lotto_seed {int(_pcfg['lotto_seed'])} (checkpoint) != "
                    f"{int(cfg.lotto_seed)} (this run). The fixed bank is regenerated from the "
                    "seed and is not checkpointed, so the trained supermasks would index a "
                    "DIFFERENT random bank at identical shapes (silent-wrong).")
        _parent_decay_fields = parent_boundary_ema_decay_fields(st["meta"])
        boundary_parent_ema_decay = _parent_decay_fields["parent_boundary_ema_decay"]
        boundary_parent_cfg_ema_decay = _parent_decay_fields["parent_cfg_ema_decay"]
        _resume_decay_fields = resume_ema_decay_fields(
            _parent_decay_fields,
            child_cfg_ema_decay=cfg.ema_decay,
            active_ema_decay=active_ema_decay,
            active_ema_decay_provenance=active_ema_decay_provenance,
        )
        boundary_ema_held = bool(_resume_decay_fields["ema_decay_held"])
        tlog({"event": "resume", "resume_from": str(args.resume_from), "epoch": start_epoch,
              "stage": stage, "quant_engaged": bool(model._quant_engaged),
              "ema_backfilled_new_params": backfilled,
              "jd1_pose_finish": dict(jd1_pose_finish_state),
              **_resume_decay_fields,
              # ddm_op2 (OP2-2): this row runs BEFORE any gate, so it can only know the DECAY
              # leg. Named explicitly so no reader mistakes it for the strict decay+basis flag
              # (that one is on the `boundary_jump` row, at the first post-resume gate).
              "gate_basis_held": None,
              "held_scope": "decay_only_gate_basis_not_yet_observable",
              "parent_gate_anchors": len(boundary_parent_tail)})
        if not boundary_ema_held:
            # L1 runtime alarm — LOUD, never silent (confound self-protection).
            tlog({"event": "confound_alarm", "kind": "ema_basis_drift",
                  "epoch": start_epoch, **_resume_decay_fields,
                  "note": "the realized gate reads the EMA shadow; parent and child resolved "
                          "DIFFERENT decays ⇒ the shadow's averaging length drifted under the "
                          "measurement and cross-boundary gate readings are NOT commensurable. "
                          "Pin --ema-decay (bypasses derive_ema_decay) or hold --epochs fixed."})
            if boundary_probe:
                # FAIL-CLOSED, scoped to the boundary experiment: an A/B whose two arms differ in
                # the measurement basis as well as the optimizer cannot be read at all.
                raise SystemExit(
                    "#824 --boundary-probe on REFUSES: EMA basis drift across the resume "
                    f"(parent {boundary_parent_ema_decay} != child {active_ema_decay}). Pin an "
                    "explicit --ema-decay equal to the parent's, or hold --epochs identical.")
        # NOTE: Adam moments are re-anchored fresh (warm-start re-anchor law #517/#518):
        # a bounded-window resume restarts moment estimation at the resume geometry. WHICH reset
        # operator that is, is now the DSL-selected #824 arm (see `optimizer_arm` above).

    if _jd1_pose_finish_enabled and args.jd1_seg_hold_space == "realized":
        _resolved_names = {str(row.get("name")) for row in resolved_scope_laws}
        if "jd3_pose_retreat_bisection" not in _resolved_names:
            _row = _resolve_scope_law("jd3_pose_retreat_bisection", {
                "explicit_pose_retreat": float(args.jd1_realized_hold_pose_retreat),
            })
            jd1_realized_hold_state["pose_retreat_factor"] = float(_row["resolved_value"])
            jd1_realized_hold_state["pose_retreat_provenance"] = str(_row["provenance"])
            jd1_realized_hold_state.setdefault("scope_law_resolution_hashes", []).append(
                _row["resolution_hash"]
            )
        if "jd3_max_retreats_a1_policy" not in _resolved_names:
            _row = _resolve_scope_law("jd3_max_retreats_a1_policy", {
                "explicit_max_retreats": int(args.jd1_realized_hold_max_retreats),
                "a1_consecutive_refuse": A1_CONSECUTIVE_REFUSE,
            })
            jd1_realized_hold_state["max_retreats"] = int(_row["resolved_value"])
            jd1_realized_hold_state["max_retreats_provenance"] = str(_row["provenance"])
            jd1_realized_hold_state.setdefault("scope_law_resolution_hashes", []).append(
                _row["resolution_hash"]
            )
        jd1_pose_finish_state["realized_hold"] = dict(jd1_realized_hold_state)
        jd1_pose_finish_state["resolved_scope_laws"] = list(resolved_scope_laws)

    # Gate set (pre-registered): all pairs when num_pairs < 600, else fd2 geometry.
    gate_ids = resolve_gate_ids(cfg.num_pairs)

    # ddm_tp2 row 3 (#274 PORT): build the theta-INDEPENDENT temporal-instability field ONCE.
    # Gated => off leaves spike_codes None => pair_loss never touches it => byte-identical.
    spike_codes = spike_lut = None
    if cfg.seg_spike_reweight:
        spike_codes = build_spike_coherent_codes(lstars, cfg.num_pairs)
        spike_lut = spike_weight_lut(cfg.seg_spike_downweight, cfg.seg_coherent_upweight)
        # Score-neutral observability defaults ON (it only READS): these counts are the port's
        # own cross-check against ddm_ti1 / ddm_fl1, which measured 625,297 spike px at n600.
        tlog({"event": "seg_spike_reweight_ready", "score_neutral": True,
              "pairs": int(cfg.num_pairs),
              "interior_pairs": int(max(cfg.num_pairs - 2, 0)),
              "spike_px_total": int((spike_codes == SPIKE_CODE_SPIKE).sum()),
              "coherent_px_total": int((spike_codes == SPIKE_CODE_COHERENT).sum()),
              "downweight": float(cfg.seg_spike_downweight),
              "upweight": float(cfg.seg_coherent_upweight),
              "byte_identical_scalars": bool(cfg.seg_spike_downweight == 1.0
                                             and cfg.seg_coherent_upweight == 1.0),
              "measured_lift_spike": SEG_SPIKE_MH_LIFT_N600,
              "measured_lift_coherent": SEG_COHERENT_MH_LIFT_N600,
              "note": "#274 producer ported from the levelset trainer; cross-pair field, "
                      "outside sigma(class, GT margin) BY CONSTRUCTION (ddm_ti1). A/B owed "
                      "(needs GO); canonical effective-frontier pointer UNMOVED"})

    # ---- ddm_p4x (#920) EXISTENCE primitive setup --------------------------------------
    # GT is STATIC per pair index, so the component index is built ONCE per pair and cached
    # (MEASURED 5.3 ms/frame, 3.2 s for the whole n600 corpus -- never on the hot path).
    # The compact index is ~9 KB/pair => ~5.5 MB for n600, so no storage-tier decision.
    # OFF => this whole block is skipped, nothing is imported, no state exists.
    _exist_cfg = None
    _exist_cache: dict[int, Any] = {}
    if cfg.existence_hinge_weight > 0.0:
        from tac.optimization import existence_hinge as _eh
        _CLASS_IDS = {"road": _eh.ROAD, "lane": _eh.LANE, "undrivable": _eh.UNDRIVABLE,
                      "movable": _eh.MOVABLE, "mycar": _eh.MYCAR}
        _names = [s.strip().lower() for s in cfg.existence_hinge_classes.split(",") if s.strip()]
        _unknown = [n for n in _names if n not in _CLASS_IDS]
        if _unknown:
            raise SystemExit(f"--existence-hinge-classes: unknown {_unknown}; "
                             f"choose from {sorted(_CLASS_IDS)}")
        _exist_cfg = _eh.ExistenceHingeConfig(
            weight=float(cfg.existence_hinge_weight),
            protected_classes=tuple(_CLASS_IDS[n] for n in _names),
            beta_override=(cfg.existence_hinge_beta or None),
            target_override=(cfg.existence_hinge_target or None),
            weight_policy_override=(cfg.existence_hinge_weight_policy or None),
        )
        _exist_cfg.validate()
        tlog({"event": "existence_hinge_init",
              "weight": float(cfg.existence_hinge_weight),
              "classes": _names,
              "connectivity": int(cfg.existence_hinge_connectivity),
              "betas": {_eh.BIRTH_MATRIX[c].class_name: _exist_cfg.policy_for(c).beta
                        for c in _exist_cfg.protected_classes},
              "weight_policies": {_eh.BIRTH_MATRIX[c].class_name:
                                  _exist_cfg.policy_for(c).weight_policy
                                  for c in _exist_cfg.protected_classes},
              "annihilate_ceiling_s": _eh.protected_ceiling_s(
                  _exist_cfg.protected_classes, int(cfg.existence_hinge_connectivity)),
              "grammar_note": "gt2's published per-word rates are 4-CONNECTED (p4x MEASURED); "
                              "per-word capture fractions are NOT comparable across grammars"})

    def _existence_pack(idx: int):
        """Per-pair (pixel_flat, mask, betas, targets, weights) as mx arrays; cached."""
        pack = _exist_cache.get(idx)
        if pack is None:
            ci = _eh.build_component_index(
                np.asarray(lstars[idx], dtype=np.int64),
                _exist_cfg.protected_classes,
                connectivity=int(cfg.existence_hinge_connectivity))
            if ci.n_comp == 0:
                pack = False  # sentinel: this pair has no protected words
            else:
                b, t = _eh.component_betas_targets(ci, _exist_cfg)
                pack = (mx.array(ci.pixel_flat.astype(np.int32)),
                        mx.array(_eh.membership_mask_np(ci)),
                        mx.array(b), mx.array(t),
                        mx.array(_eh.component_weights(ci, _exist_cfg)))
            _exist_cache[idx] = pack
        return None if pack is False else pack

    def pair_loss(mdl, idx: int, form: str, *, pose_active: bool | None = None,
                  terms_out: dict[str, Any] | None = None):
        if pose_active is None:
            pose_active = bool(jd1_pose_finish_state.get("engaged"))
        lstar = np.asarray(lstars[idx], dtype=np.int64)
        lstar_oh = mx.array((lstar[..., None] == np.arange(5)).astype(np.float32))[None]
        margin = mx.array(np.asarray(margins[idx], dtype=np.float32))
        if pose_active:
            if gt_poses is None:
                raise RuntimeError("JD1 pose finish active without gt_poses memmap")
            pose_tgt = mx.array(np.asarray(gt_poses[idx], dtype=np.float32)[:6])
            w_pose = float(jd1_effective_w_pose)
            pose_code0 = max(int(idx) - 1, 0)
        else:
            pose_tgt = mx.zeros((6,))
            w_pose = 0.0
            pose_code0 = int(idx)
        # sn1 ASYMMETRY lever: per-GT-class weight on Lane pixels (class index 1 —
        # canonical comma10k order, MEASURED; NEVER luma-sort re-derived).
        seg_pixel_w = None
        w_np = None
        if cfg.class_weight_lane != 1.0:
            w_np = 1.0 + (cfg.class_weight_lane - 1.0) * (lstar == 1).astype(np.float32)
        # ddm_lg1 (#808): fold the CONSTRAIN-AND-PROTECT addend (lambda_lane + born-lane
        # protection + margin-floor emphasis) into the SAME per-pixel weight. Gated =>
        # byte-identical when off (block skipped; w_np stays None).
        if cfg.lane_guard:
            _lg_margin = (np.asarray(margins[idx], dtype=np.float32)
                          if (lane_guard_cfg.margin_floor_weight > 0.0
                              and lane_guard_state.margin_floor is not None) else None)
            _lg_add = _lane_guard.pixel_weight_addend(
                lstar, _lg_margin, lane_guard_state, lane_guard_cfg, idx)
            if _lg_add is not None:
                w_np = (_lg_add + 1.0) if w_np is None else (w_np + _lg_add)
        # ddm_tp2 row 3 (#274): fold the temporal-instability weight in MULTIPLICATIVELY (the
        # same composition rule the levelset trainer uses for focal/Fisher), AFTER the additive
        # class/lane-guard accumulation, so it SCALES the composed weight rather than replacing it.
        if spike_codes is not None:
            _sp_w = spike_lut[spike_codes[idx]]          # (H,W) float32 via 3-entry LUT
            w_np = _sp_w if w_np is None else (w_np * _sp_w)
        if w_np is not None:
            seg_pixel_w = mx.array(w_np)[None]
        # QA75 teacher logits for THIS pair (precomputed b2b scorer response); None => OFF.
        distill_logits = None
        if distill_mm is not None:
            dl = np.asarray(distill_mm[idx], dtype=np.float32)         # (5,H,W)
            distill_logits = mx.array(np.transpose(dl, (1, 2, 0)))[None]  # (1,H,W,5)
        # ddm_p4x (#920): COMPONENT-level existence term. None => never built => byte-identical.
        existence_pack = _existence_pack(idx) if _exist_cfg is not None else None
        if args.seg_grad_q3_project == "on" and pose_active:
            seg_terms: dict[str, Any] | None = {} if terms_out is not None else None
            pose_terms: dict[str, Any] | None = {} if terms_out is not None else None
            seg_loss = loss_fn(
                mdl, None, idx, idx, lstar_oh, margin, pose_tgt,
                cfg.w_seg, 0.0, 0.0, cfg.margin_target, seg_form=form,
                seg_pixel_w=seg_pixel_w, compute_pose=False,
                terms_out=seg_terms,
                distill_logits=distill_logits, distill_weight=cfg.distill_weight,
                distill_temp=cfg.distill_temp, distill_form=cfg.distill_form,
                distill_attack_temp=cfg.distill_attack_temp,
                existence_pack=existence_pack,
                existence_weight=cfg.existence_hinge_weight)
            pose_loss = pose_loss_fn(
                mdl, None, pose_code0, idx, lstar_oh, margin, pose_tgt,
                0.0, w_pose, 0.0, cfg.margin_target, seg_form=form,
                seg_pixel_w=None, compute_pose=True,
                terms_out=pose_terms,
                distill_logits=None, distill_weight=0.0,
                distill_temp=cfg.distill_temp, distill_form=cfg.distill_form,
                distill_attack_temp=cfg.distill_attack_temp,
                existence_pack=None, existence_weight=0.0)
            if terms_out is not None:
                terms_out.update(seg_terms or {})
                if pose_terms is not None and "pose" in pose_terms:
                    terms_out["pose"] = pose_terms["pose"]
            return seg_loss + pose_loss
        return loss_fn(mdl, None, pose_code0, idx, lstar_oh, margin, pose_tgt,
                       cfg.w_seg, w_pose, 0.0, cfg.margin_target, seg_form=form,
                       seg_pixel_w=seg_pixel_w, compute_pose=bool(pose_active),
                       terms_out=terms_out,
                       distill_logits=distill_logits, distill_weight=cfg.distill_weight,
                       distill_temp=cfg.distill_temp, distill_form=cfg.distill_form,
                       distill_attack_temp=cfg.distill_attack_temp,
                       existence_pack=existence_pack,
                       existence_weight=cfg.existence_hinge_weight)

    state_form = {"form": cfg.seg_form_start}

    # ddm_lg1 (#808) CONSTRAIN-AND-PROTECT state. Built ONLY when the lever is on;
    # off => these are never referenced (pair_loss + the gate block are cfg.lane_guard-gated)
    # => byte-identical. Budget = xp1 ep641 Lane S; eta/step DERIVED (constants-are-poison).
    lane_guard_cfg = lane_guard_state = None
    if cfg.lane_guard:
        from tac.optimization import lane_guard as _lane_guard
        # ddm_lp1 #934: the deadband horizon is a MULTIPLE-COMPARISONS burden and must be
        # the run's TOTAL gate count, not the elapsed one.  Leaving --lane-guard-ratchet-
        # horizon at 0 now derives that total from this run's own gate cadence; the old
        # 0 => "max(mean_gates, gates_seen)" fallback under-priced the burden early and
        # was MEASURED to produce 3/64 false-positive engagements on the burn-4 series.
        _rh, _rh_prov = _lane_guard.derive_planned_gate_horizon(cfg.epochs, cfg.gate_every)
        _ratchet_horizon = int(cfg.lane_guard_ratchet_horizon or _rh)
        lane_guard_cfg = _lane_guard.LaneGuardConfig(
            enabled=True,
            budget_s=(cfg.lane_guard_budget_s or _lane_guard.LANE_BUDGET_S_UNITS),
            eta_lambda=cfg.lane_guard_eta,
            lambda_step_cap=cfg.lane_guard_lambda_step_cap,
            lambda_max=cfg.lane_guard_lambda_max,
            born_protect_weight=cfg.lane_guard_born_weight,
            margin_floor_weight=cfg.lane_guard_margin_floor_weight,
            budget_ratchet=cfg.lane_guard_ratchet,
            ratchet_horizon_gates=_ratchet_horizon,
        ).resolved()
        lane_guard_state = _lane_guard.LaneGuardState(
            lambda_lane=max(0.0, min(cfg.lane_guard_lambda_init,
                                     lane_guard_cfg.lambda_max)))
        tlog({"event": "lane_guard_init", "budget_s": lane_guard_cfg.budget_s,
              "lambda_init": lane_guard_state.lambda_lane,
              "eta_lambda": lane_guard_cfg.eta_lambda,
              "lambda_step_cap": lane_guard_cfg.lambda_step_cap,
              "lambda_max": lane_guard_cfg.lambda_max,
              "born_protect_weight": lane_guard_cfg.born_protect_weight,
              "margin_floor_weight": lane_guard_cfg.margin_floor_weight,
              "lane_sensitivity_ratio": lane_guard_cfg.lane_sensitivity_ratio,
              "budget_ratchet": lane_guard_cfg.budget_ratchet,
              "ratchet_mean_gates": lane_guard_cfg.ratchet_mean_gates,
              "ratchet_horizon_gates": lane_guard_cfg.ratchet_horizon_gates,
              "ratchet_horizon_provenance": _rh_prov,
              "ratchet_horizon_source": ("operator --lane-guard-ratchet-horizon"
                                         if cfg.lane_guard_ratchet_horizon
                                         else "derived from epochs/gate_every (ddm_lp1 #934)"),
              "eta_provenance": _lane_guard.derive_eta_lambda()[1],
              "note": "lg1 CONSTRAIN-AND-PROTECT; realized-g read from a1 gate (zero new "
                      "scorer passes); score_neutral verdict authority unchanged"})

    # §3.4 rate-in-loss keep-cell indices (kept cells only enter the token-entropy surrogate,
    # so the zeroed inactive cells do not bias the histogram with a constant spike at bin-0).
    keep_bool_np = np.asarray(model._cell_mask.tensors["keep"], dtype=np.float32)[..., 0] > 0.5
    keep_flat_np = np.flatnonzero(keep_bool_np.ravel()).astype(np.int64)

    def token_rate_term(mdl, ids: list[int]):
        """Differentiable per-token entropy surrogate over the batch's KEPT cells.
        'entropy' = marginal token histogram; 'smevr_surrogate' = consecutive-frame temporal
        delta histogram (the zlib-on-delta coder surrogate token_stream_bytes runs)."""
        keep_idx = mx.array(keep_flat_np)
        c = cfg.code_width
        if cfg.rate_model == "smevr_surrogate":
            sids = sorted(int(i) for i in ids)
            if len(sids) < 2:
                return mx.array(0.0)
            deltas = []
            prev = mx.take(mx.reshape(mdl.raw_tokens(sids[0]), (-1, c)), keep_idx, axis=0)
            for i in sids[1:]:
                cur = mx.take(mx.reshape(mdl.raw_tokens(i), (-1, c)), keep_idx, axis=0)
                deltas.append(0.5 * (cur - prev))  # scale delta [-2,2] -> [-1,1]
                prev = cur
            vals = mx.concatenate(deltas, axis=0)
        else:  # "entropy": marginal histogram of the kept-cell token values
            vals = mx.concatenate(
                [mx.take(mx.reshape(mdl.raw_tokens(int(i)), (-1, c)), keep_idx, axis=0)
                 for i in ids], axis=0)
        return _soft_hist_entropy_bits(vals, cfg.token_quant_levels)

    def delta_sparsity_term(mdl, ids: list[int]):
        """ax1 §4a group-L2 (group-lasso) shrinkage on the per-pair token deltas of the batch's
        KEPT cells (the 98.806% image-stationary mass has no train-side force). The group is a
        (pair,cell) delta over its ``c`` channels; whole-cell deltas → 0 → SMEVR zero-delta runs.
        xi-informed weight (§5) relaxes the shrinkage on dynamic (lane/movable) cells. Mirrors
        ``delta_group_sparsity_penalty`` (numpy authority) exactly."""
        keep_idx = mx.array(keep_flat_np)
        c = cfg.code_width
        ds = mx.stack(
            [mx.take(mx.reshape(mdl.tokens_delta[int(i)], (-1, c)), keep_idx, axis=0) for i in ids],
            axis=0)                                            # (B, n_kept, c)
        g = mx.sqrt(mx.sum(ds * ds, axis=-1) + 1e-8)          # (B, n_kept) per-group L2
        if mdl._delta_sparsity_weight_field is not None:
            wf = mx.take(mx.reshape(mdl._delta_sparsity_weight_field.tensors["w"], (-1,)),
                         keep_idx, axis=0)                     # (n_kept,)
            g = g * wf[None]
        return mx.mean(g)

    def batch_loss(mdl, ids: list[int]):
        acc = None
        for i in ids:
            li = pair_loss(mdl, int(i), state_form["form"])
            acc = li if acc is None else acc + li
        acc = acc / len(ids)
        if (jd1_pose_finish_state.get("engaged")
                and float(args.jd1_seg_hold_weight) > 0.0
                and args.jd1_seg_hold_space == "loss"):
            floor = jd1_pose_finish_state.get("seg_hold_floor")
            if floor is None:
                raise RuntimeError("JD1 seg-hold active without a latched floor")
            seg_acc = None
            for i in ids:
                sli = pair_loss(mdl, int(i), state_form["form"], pose_active=False)
                seg_acc = sli if seg_acc is None else seg_acc + sli
            seg_acc = seg_acc / len(ids)
            floor_with_margin = float(floor) + float(args.jd1_seg_hold_margin)
            acc = acc + float(args.jd1_seg_hold_weight) * mx.maximum(
                seg_acc - floor_with_margin, 0.0)
        if cfg.w_rate > 0.0:  # §3.4 (0.0 => byte-identical to the distortion-only control)
            acc = acc + cfg.w_rate * token_rate_term(mdl, ids)
        # ax1 §4a delta group-sparsity: only once ENGAGED (base-stability event / from_step_0) and
        # weight>0 => byte-identical to the control until engagement (gc10 F2 twin of the ν snap).
        if mdl._delta_sparsity_engaged and cfg.delta_sparsity_weight > 0.0:
            acc = acc + cfg.delta_sparsity_weight * delta_sparsity_term(mdl, ids)
        if birth_seed_amplify_weight > 0.0:
            acc = acc + birth_seed_amplify_weight * tr1_birth_amplify_term(mdl, ids)
        return acc

    vg = nn.value_and_grad(model, batch_loss)

    prev_gate_row: dict[str, Any] | None = None
    prev_gate_smooth: float | None = None
    prev_realized: np.ndarray | None = None
    prev_gate_basis: str | None = None
    a1_consecutive = 0
    # #85 EMA shadow-lag instrument guard (measured live 2026-07-28: at decay 0.99867 a
    # ~675-step shadow is ~41% zero-init seed and the half-warmed mixed render scored
    # 0.842 — WORSE than gray init — firing a FALSE A1 alarm chain). The law's own
    # warmup boundary W = 2/(1-d) decides the gate basis: LIVE params before W,
    # EMA shadow after; the basis change REBASES the A1 comparison (one gate).
    ema_warmup_updates = int(np.ceil(2.0 / max(1.0 - active_ema_decay, 1e-9)))
    global_step = 0 if args.resume_from is None else ema_warmup_updates  # resume => warm shadow
    ep_losses: list[float] = []
    telemetry_tail: list[dict] = []
    gnorm_hist: list[float] = []
    basin_window: list[dict] = []  # basin-entry detector state (basin_handoff == "on")
    gate_param_snapshot: dict[str, np.ndarray] | None = None
    prev_gate_snapshot: dict[str, Any] | None = None
    order_rng = np.random.default_rng(cfg.seed + 1)
    knee_switched = stage != SEG_TRUNK_CE_STAGE
    jd1_lr_schedule: dict[str, Any] | None = None
    jd1_lr_current = float(cfg.lr)

    def _copy_mx_tree(tree: dict[str, Any]) -> dict[str, Any]:
        return {k: mx.array(np.asarray(v).copy()) for k, v in tree.items()}

    def _jd1_parent_shadow_payload(parent_ema: dict[str, Any]) -> dict[str, np.ndarray]:
        return {f"jd1_parent_ema::{k}": np.asarray(v).copy() for k, v in parent_ema.items()}

    def _apply_jd1_stage_ema_reanchor(*, epoch: int, reason: str) -> None:
        nonlocal ema, active_ema_decay, active_ema_decay_provenance
        nonlocal ema_warmup_updates, global_step
        if not jd1_should_reanchor_stage_ema(args, jd1_pose_finish_state, reason=reason):
            return
        old_carried_decay = jd1_pose_finish_state.get("active_ema_decay")
        old_carried_prov = jd1_pose_finish_state.get("active_ema_decay_provenance")
        old_reanchor_epoch = jd1_pose_finish_state.get("stage_ema_reanchored_epoch")
        forced = bool(jd1_pose_finish_state.get("stage_ema_reanchored"))
        parent_ema = _copy_mx_tree(ema)
        remaining_epochs = max(1, int(cfg.epochs) - int(epoch))
        ema_resolution = _resolve_scope_law("jd3_stage_ema_decay", {
            "remaining_epochs": remaining_epochs,
            "steps_per_epoch": steps_per_epoch,
            "run_geometry_hash": scope_law_geometry_hash(
                steps_per_epoch=steps_per_epoch,
                horizon_epochs=int(cfg.epochs),
                window_epochs=remaining_epochs,
            ),
        })
        new_decay = float(ema_resolution["resolved_value"])
        new_prov = str(ema_resolution["provenance"])
        refuse_declared_vs_resolved_jd1_ema_decay(
            args.ema_decay,
            new_decay,
            resolution_hash=str(ema_resolution["resolution_hash"]),
        )
        ema = _copy_mx_tree(dict(tree_flatten(model.trainable_parameters())))
        active_ema_decay = float(new_decay)
        active_ema_decay_provenance = str(new_prov)
        ema_warmup_updates = int(np.ceil(2.0 / max(1.0 - active_ema_decay, 1e-9)))
        global_step = max(int(global_step), int(ema_warmup_updates))
        jd1_pose_finish_state.update({
            "stage_ema_reanchored": True,
            "stage_ema_reanchored_epoch": int(epoch),
            "stage_ema_reanchor_reason": reason,
            "stage_ema_reanchor_forced_from_resume": bool(forced),
            "stage_ema_reanchor_previous_epoch": old_reanchor_epoch,
            "stage_ema_reanchor_previous_decay": (None if old_carried_decay is None
                                                  else float(old_carried_decay)),
            "stage_ema_reanchor_previous_provenance": old_carried_prov,
            "parent_ema_preserved_key_prefix": "jd1_parent_ema::",
            "active_ema_decay": float(active_ema_decay),
            "active_ema_decay_provenance": active_ema_decay_provenance,
            "ema_warmup_updates": int(ema_warmup_updates),
            "stage_ema_scope_law_resolution_hash": ema_resolution["resolution_hash"],
            "resolved_scope_laws": list(resolved_scope_laws),
            **jd1_ema_initial_state(args),
        })
        save_checkpoint(out_dir / "checkpoints" / "stage_joint_pose_finish_entry.npz",
                        model=model, ema=ema, opt_state_flat=_opt_state(), epoch=epoch,
                        stage=stage, cfg=cfg, telemetry_tail=telemetry_tail,
                        extra_meta=_jd1_checkpoint_extra_meta(),
                        extra_npz_arrays=_jd1_parent_shadow_payload(parent_ema))
        tlog({"event": "jd1_stage_ema_reanchor",
              "epoch": int(epoch), "stage": stage, "reason": reason,
              "forced_from_resume": bool(forced),
              "old_carried_decay": (None if old_carried_decay is None
                                    else float(old_carried_decay)),
              "old_carried_decay_provenance": old_carried_prov,
              "new_derived_decay": float(active_ema_decay),
              "new_derived_decay_provenance": active_ema_decay_provenance,
              "parent_shadow_preserved_prefix": "jd1_parent_ema::",
              "active_ema_decay": float(active_ema_decay),
              "active_ema_decay_provenance": active_ema_decay_provenance,
              "scope_law_resolution_hash": ema_resolution["resolution_hash"],
              "ema_mode": args.jd1_ema_mode,
              "ema_tail_anchor_epoch": int(args.jd1_ema_tail_anchor_epoch),
              "ema_warmup_updates": int(ema_warmup_updates),
              "checkpoint": "checkpoints/stage_joint_pose_finish_entry.npz",
              "score_claim": False})

    def _log_jd1_ema_mode(*, epoch: int, reason: str) -> None:
        payload = jd1_ema_checkpoint_payload(args, jd1_pose_finish_state)
        tlog({
            "event": "jd1_ema_mode",
            "epoch": int(epoch),
            "stage": stage,
            "reason": reason,
            "mode": args.jd1_ema_mode,
            "ema_stage_scope": args.jd1_ema_stage_scope,
            "active_ema_decay": float(active_ema_decay),
            "ema_tail_anchor_epoch": int(args.jd1_ema_tail_anchor_epoch),
            "ema_tail_average_active": bool(payload.get("ema_tail_average_active", False)),
            "ema_tail_update_count": int(payload.get("ema_tail_update_count", 0)),
            "gate_basis": jd1_ema_gate_basis_label(
                global_step=global_step,
                ema_warmup_updates=ema_warmup_updates,
                state=jd1_pose_finish_state,
            ),
            "score_claim": False,
        })

    def _resolve_jd1_lr_anneal_schedule(*, epoch: int, reason: str) -> None:
        nonlocal jd1_lr_schedule
        if args.jd1_lr_anneal != "derived_tail":
            return
        if jd1_lr_schedule is not None:
            return
        try:
            jd1_lr_schedule = derive_jd1_lr_tail_schedule(
                base_lr=float(cfg.lr),
                start_epoch=int(epoch),
                end_epoch=int(cfg.epochs),
                steps_per_epoch=int(steps_per_epoch),
                beta2=float(RESET_ADAM_BETAS[1]),
                active_ema_decay=float(active_ema_decay),
                resume_from=args.resume_from,
                explicit_final_frac=float(args.jd1_lr_final_frac),
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        jd1_pose_finish_state["lr_anneal"] = dict(jd1_lr_schedule)
        tlog({"event": "jd1_lr_anneal_config",
              "epoch": int(epoch),
              "stage": stage,
              "reason": reason,
              **jd1_lr_schedule})

    def _apply_jd1_lr_anneal_for_epoch(*, epoch: int) -> float:
        nonlocal jd1_lr_current
        if jd1_finisher_active:
            return jd1_lr_current
        if jd1_lr_schedule is None or not jd1_pose_finish_state.get("engaged"):
            jd1_lr_current = float(cfg.lr)
            return jd1_lr_current
        jd1_lr_current = float(jd1_lr_at_epoch(epoch, jd1_lr_schedule))
        optimizer.learning_rate = jd1_lr_current
        jd1_pose_finish_state["lr_anneal_last_epoch"] = int(epoch)
        jd1_pose_finish_state["lr_anneal_last_lr"] = float(jd1_lr_current)
        return jd1_lr_current

    def _derive_jd1_muon_finisher_schedule(*, epoch: int, reason: str) -> dict[str, Any]:
        nonlocal jd1_muon_schedule
        if jd1_muon_schedule is not None:
            return dict(jd1_muon_schedule)
        try:
            lr_tail = derive_jd1_lr_tail_schedule(
                base_lr=float(cfg.lr),
                start_epoch=int(epoch),
                end_epoch=int(cfg.epochs),
                steps_per_epoch=int(steps_per_epoch),
                beta2=float(RESET_ADAM_BETAS[1]),
                active_ema_decay=float(active_ema_decay),
                resume_from=args.resume_from,
                explicit_final_frac=float(args.jd1_lr_final_frac),
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        muon_momentum, momentum_source = derive_jd1_muon_momentum(RESET_ADAM_BETAS[0])
        final_frac = float(lr_tail["final_frac"])
        anneal_steps = (
            max(1, (int(cfg.epochs) - int(epoch)) * int(steps_per_epoch))
            if final_frac < 1.0 else 0
        )
        jd1_muon_schedule = {
            "schema": JD1_FINISHER_SCHEMA,
            "mode": "muon",
            "active": False,
            "reason": reason,
            "switch_epoch": int(epoch),
            "muon_lr": float(lr_tail["base_lr"]),
            "adam_lr": float(lr_tail["base_lr"]),
            "muon_lr_final_frac": final_frac,
            "muon_final_lr": float(lr_tail["base_lr"]) * final_frac,
            "muon_anneal_steps": int(anneal_steps),
            "muon_momentum": float(muon_momentum),
            "muon_momentum_source": momentum_source,
            "muon_ns_steps": int(JD1_MUON_FINISHER_NS_STEPS),
            "lr_tail_derivation": dict(lr_tail),
            "score_claim": False,
        }
        return dict(jd1_muon_schedule)

    def _activate_jd1_muon_finisher(*, epoch: int, reason: str) -> None:
        nonlocal optimizer, jd1_finisher_active, jd1_muon_schedule, jd1_lr_current
        if args.jd1_finisher != "muon":
            return
        if jd1_finisher_active:
            return
        if not jd1_pose_finish_state.get("engaged"):
            raise RuntimeError("JD1 Muon finisher activation reached before JD1 engagement")
        schedule = _derive_jd1_muon_finisher_schedule(epoch=epoch, reason=reason)
        n_muon, n_adam = tr1_muon_adam_split_counts(model.trainable_parameters())
        if n_muon <= 0:
            raise SystemExit(
                "--jd1-finisher muon REFUSED: TR1 Muon renderer-weight split routed zero "
                "leaves. This would be an inert finisher.")
        old_adam_state = (
            optimizer.state if type(optimizer).__name__ in ("Adam", "AdamW") else None
        )
        new_optimizer = build_tr1_jd1_muon_finisher_optimizer(
            muon_lr=float(schedule["muon_lr"]),
            adam_lr=float(schedule["adam_lr"]),
            muon_momentum=float(schedule["muon_momentum"]),
            muon_ns_steps=int(schedule["muon_ns_steps"]),
            muon_lr_final_frac=float(schedule["muon_lr_final_frac"]),
            muon_anneal_steps=int(schedule["muon_anneal_steps"]),
            adam_bias_correction=bool(_bias_correction),
        )
        new_optimizer.init(model.trainable_parameters())
        mx.eval(new_optimizer.state)
        warm_seeded = 0
        if old_adam_state is not None:
            try:
                warm_seeded = seed_tr1_muon_momentum_from_adam(
                    new_optimizer.optimizers[0],
                    old_adam_state,
                )
            except Exception as exc:
                warm_seeded = -1
                tlog({"event": "jd1_muon_warm_start_failed_cold_fallback",
                      "epoch": int(epoch),
                      "error": f"{type(exc).__name__}: {exc}",
                      "score_claim": False})
            mx.eval(new_optimizer.state)
        optimizer = new_optimizer
        jd1_finisher_active = True
        jd1_lr_current = float(schedule["muon_lr"])
        schedule["active"] = True
        schedule["n_muon_leaves"] = int(n_muon)
        schedule["n_adam_leaves"] = int(n_adam)
        schedule["warm_start_from_adam_m"] = bool(old_adam_state is not None)
        schedule["muon_warm_seeded_leaves"] = int(warm_seeded)
        jd1_muon_schedule = dict(schedule)
        jd1_pose_finish_state["finisher"] = dict(schedule)
        tlog({"event": "jd1_muon_finisher_switch",
              "epoch": int(epoch),
              "stage": stage,
              "reason": reason,
              **schedule,
              "optimizer_split": "Muon(TR1 renderer tensors) + Adam(tokens/biases/gains/gates)",
              "note": "Case-B terminal optimizer switch. MLX Muon performs real "
                      "Newton-Schulz orthogonalized momentum and flattens conv filters "
                      "to 2-D matrices internally; tokens and non-matrix leaves stay Adam.",
              "score_claim": False})

    def _maybe_apply_jd1_ema_tail_anchor(*, epoch: int, reason: str) -> None:
        nonlocal ema
        if args.jd1_ema_mode != "plateau_tail_average":
            return
        if not jd1_pose_finish_state.get("engaged"):
            return
        if jd1_ema_tail_average_active(jd1_pose_finish_state):
            return
        anchor_epoch = jd1_ema_tail_anchor_epoch(int(args.jd1_ema_tail_anchor_epoch))
        if anchor_epoch is None or int(epoch) < int(anchor_epoch):
            return
        ema = _copy_mx_tree(dict(tree_flatten(model.trainable_parameters())))
        jd1_pose_finish_state.update({
            "ema_mode": "plateau_tail_average",
            "ema_tail_anchor_epoch": int(epoch),
            "ema_tail_configured_anchor_epoch": int(anchor_epoch),
            "ema_tail_average_active": True,
            "ema_tail_update_count": 0,
            "ema_tail_anchor_global_step": int(global_step),
            "ema_tail_anchor_reason": reason,
            "ema_tail_last_live_weight": None,
        })
        tlog({
            "event": "jd1_ema_tail_average_anchor",
            "epoch": int(epoch),
            "stage": stage,
            "reason": reason,
            "configured_anchor_epoch": int(anchor_epoch),
            "global_step": int(global_step),
            "gate_basis": "ema_tail_average",
            "score_claim": False,
            "note": "Reset EMA shadow to live weights and switch future updates to a "
                    "growing-horizon Polyak tail average over the anchor plus settled "
                    "post-anchor live iterates.",
        })

    def _capture_jd1_gate_snapshot(epoch: int) -> dict[str, Any]:
        return {
            "epoch": int(epoch),
            "model": _copy_mx_tree(dict(tree_flatten(model.trainable_parameters()))),
            "ema": _copy_mx_tree(ema),
            "opt_flat": optimizer_state_to_flat(optimizer),
            "global_step": int(global_step),
            "ep_losses": list(ep_losses),
            "telemetry_tail": copy.deepcopy(telemetry_tail),
            "gnorm_hist": list(gnorm_hist),
            "basin_window": copy.deepcopy(basin_window),
            "gate_param_snapshot": (None if gate_param_snapshot is None
                                    else {k: v.copy() for k, v in gate_param_snapshot.items()}),
            "prev_gate_row": copy.deepcopy(prev_gate_row),
            "prev_gate_smooth": prev_gate_smooth,
            "prev_realized": (None if prev_realized is None else prev_realized.copy()),
            "prev_gate_basis": prev_gate_basis,
            "a1_consecutive": int(a1_consecutive),
            "order_rng_state": copy.deepcopy(order_rng.bit_generator.state),
            "knee_switched": bool(knee_switched),
            "stage": stage,
            "state_form": dict(state_form),
            "boundary_jump_emitted": bool(boundary_jump_emitted),
            "boundary_strict_basis_held": boundary_strict_basis_held,
            "boundary_gate_basis_held": boundary_gate_basis_held,
            "jd1_pose_finish_state": copy.deepcopy(jd1_pose_finish_state),
            "jd1_realized_hold_state": copy.deepcopy(jd1_realized_hold_state),
            "jd1_effective_w_pose": float(jd1_effective_w_pose),
            "active_ema_decay": float(active_ema_decay),
            "active_ema_decay_provenance": active_ema_decay_provenance,
            "ema_warmup_updates": int(ema_warmup_updates),
            "model_quant_engaged": bool(model._quant_engaged),
            "model_delta_sparsity_engaged": bool(model._delta_sparsity_engaged),
            "lane_guard_state": copy.deepcopy(lane_guard_state),
        }

    def _restore_jd1_gate_snapshot(snapshot: dict[str, Any]) -> None:
        nonlocal ema, global_step, ep_losses, telemetry_tail, gnorm_hist, basin_window
        nonlocal gate_param_snapshot, prev_gate_row, prev_gate_smooth, prev_realized
        nonlocal prev_gate_basis, a1_consecutive, knee_switched, stage
        nonlocal boundary_jump_emitted, boundary_strict_basis_held, boundary_gate_basis_held
        nonlocal jd1_pose_finish_state, jd1_realized_hold_state, jd1_effective_w_pose
        nonlocal active_ema_decay, active_ema_decay_provenance, ema_warmup_updates
        nonlocal lane_guard_state
        from mlx.utils import tree_unflatten

        model.update(tree_unflatten(list(snapshot["model"].items())))
        ema = _copy_mx_tree(snapshot["ema"])
        opt_flat = snapshot.get("opt_flat") or {}
        if opt_flat:
            restore_optimizer_state(optimizer, model, opt_flat)
        elif args.jd1_seg_hold_space == "realized":
            raise RuntimeError("JD3 rollback requires optimizer state in the gate snapshot")
        global_step = int(snapshot["global_step"])
        ep_losses = list(snapshot["ep_losses"])
        telemetry_tail = copy.deepcopy(snapshot["telemetry_tail"])
        gnorm_hist = list(snapshot["gnorm_hist"])
        basin_window = copy.deepcopy(snapshot["basin_window"])
        gate_param_snapshot = (None if snapshot["gate_param_snapshot"] is None
                               else {k: v.copy()
                                     for k, v in snapshot["gate_param_snapshot"].items()})
        prev_gate_row = copy.deepcopy(snapshot["prev_gate_row"])
        prev_gate_smooth = snapshot["prev_gate_smooth"]
        prev_realized = (None if snapshot["prev_realized"] is None
                         else snapshot["prev_realized"].copy())
        prev_gate_basis = snapshot["prev_gate_basis"]
        a1_consecutive = int(snapshot["a1_consecutive"])
        order_rng.bit_generator.state = copy.deepcopy(snapshot["order_rng_state"])
        knee_switched = bool(snapshot["knee_switched"])
        stage = str(snapshot["stage"])
        state_form.clear()
        state_form.update(snapshot["state_form"])
        boundary_jump_emitted = bool(snapshot["boundary_jump_emitted"])
        boundary_strict_basis_held = snapshot["boundary_strict_basis_held"]
        boundary_gate_basis_held = snapshot["boundary_gate_basis_held"]
        jd1_pose_finish_state = copy.deepcopy(snapshot["jd1_pose_finish_state"])
        jd1_realized_hold_state = copy.deepcopy(snapshot["jd1_realized_hold_state"])
        jd1_effective_w_pose = float(snapshot["jd1_effective_w_pose"])
        active_ema_decay = float(snapshot["active_ema_decay"])
        active_ema_decay_provenance = str(snapshot["active_ema_decay_provenance"])
        ema_warmup_updates = int(snapshot["ema_warmup_updates"])
        model._quant_engaged = bool(snapshot["model_quant_engaged"])
        model._delta_sparsity_engaged = bool(snapshot["model_delta_sparsity_engaged"])
        lane_guard_state = copy.deepcopy(snapshot["lane_guard_state"])
        mx.eval(model.parameters(), optimizer.state, *ema.values())

    if (args.jd1_finisher == "muon"
            and not jd1_pose_finish_state.get("engaged")
            and not jd1_pose_finish_should_engage(args, epoch=start_epoch, stage=stage)):
        raise SystemExit(
            "--jd1-finisher muon REFUSED: JD1 would not be active at the resumed start "
            f"epoch {start_epoch}. Case-B is a terminal boundary action; choose a "
            "checkpoint/start_epoch where JD1 is already active at launch.")

    if jd1_pose_finish_state.get("engaged"):
        _apply_jd1_stage_ema_reanchor(epoch=start_epoch,
                                      reason="resume_inside_joint_pose_finish")
        _log_jd1_ema_mode(epoch=start_epoch, reason="resume_inside_joint_pose_finish")
        _resolve_jd1_lr_anneal_schedule(epoch=start_epoch,
                                        reason="resume_inside_joint_pose_finish")
        _activate_jd1_muon_finisher(epoch=start_epoch,
                                    reason="resume_inside_joint_pose_finish")
    if knee_switched and state_form["form"] == "ce":
        # #517-twin re-anchor (2026-07-30 qa86 EMA-resume incident): the form state
        # machine must position to the RESTORED stage, not --seg-form-start. Without
        # this, a mid-stage resume trains the opening CE form against a tau-stage
        # checkpoint FOREVER (both switch events are knee_switched-guarded; measured
        # ep_loss 0.673 -> 9.611 at resumed ep202 before MAIN killed it).
        state_form["form"] = "tau_softplus"
        tlog({"event": "resume_form_reanchor", "stage": stage,
              "form": state_form["form"], "seg_form_start": cfg.seg_form_start})
    stop_reason = "epochs_complete"

    # ddm_bp1 (#824) POSITIVE-CONTROL re-gate (CLAUDE.md L3 verdict-clearance: a known-effect
    # canary the apparatus must register, else the instrument is untrusted and NO verdict is
    # admissible). Runs the realized gate on the RESTORED state BEFORE any training: it must
    # reproduce the parent checkpoint's last gate reading from the same bytes on the same basis.
    # READ-ONLY (tlog only, never telemetry_tail => the next checkpoint is byte-identical to an
    # off run) and identical in BOTH arms, so it cannot confound the A/B. Costs one gate.
    if boundary_probe and args.resume_from is not None:
        # Tolerance DERIVED, not a bare constant: d_seg is the mean per-pixel argmax disagreement
        # over the gate set, so its smallest possible non-zero change is ONE pixel flip =
        # 1/(n_gate*H*W). Half that quantum accepts only a bit-exact reproduction while staying
        # robust to float summation order across processes.
        boundary_pc_tol = 0.5 / float(len(gate_ids) * SEG_H * SEG_W)
        _pc_basis = jd1_ema_gate_basis_label(
            global_step=global_step,
            ema_warmup_updates=ema_warmup_updates,
            state=jd1_pose_finish_state,
        )
        _pc_live = ema_snapshot_swap(model, ema) if _pc_basis != "live_ema_warmup" else None
        try:
            _pc_row = realized_gate(model, gate_ids, lstars, seg_cpu, None)
        finally:
            if _pc_live is not None:
                ema_restore(model, _pc_live)
        _pc_row.pop("_realized_argmax", None)
        _pc_anchors = [r for r in boundary_parent_tail
                       if isinstance(r, dict) and r.get("realized_gate_dseg_mean") is not None]
        _pc_parent = (max(_pc_anchors, key=lambda r: int(r.get("epoch", -1)))
                      if _pc_anchors else None)
        _pc_ref = (float(_pc_parent["realized_gate_dseg_mean"]) if _pc_parent else None)
        _pc_now = float(_pc_row["realized_gate_dseg_mean"])
        _pc_delta = (None if _pc_ref is None else _pc_now - _pc_ref)
        _pc_ok = (_pc_ref is not None and _pc_parent.get("gate_params") == _pc_basis
                  and abs(_pc_delta) <= boundary_pc_tol)
        tlog({"event": "boundary_positive_control", "epoch": start_epoch,
              "basis": _pc_basis, "parent_basis": (_pc_parent or {}).get("gate_params"),
              "parent_gate_epoch": (None if not _pc_parent else int(_pc_parent.get("epoch", -1))),
              "parent_gate_dseg": _pc_ref, "reproduced_dseg": _pc_now,
              "abs_delta": (None if _pc_delta is None else abs(_pc_delta)),
              "tolerance": boundary_pc_tol, "reproduced": bool(_pc_ok),
              "score_claim": False, "evidence_axis": "[macOS-CPU/MLX advisory]",
              "note": "the restored state must reproduce the parent's last gate BEFORE training; "
                      "reproduced=false ⇒ the instrument is untrusted, no #824 verdict admissible"})
        if not _pc_ok:
            tlog({"event": "confound_alarm", "kind": "boundary_positive_control_failed",
                  "epoch": start_epoch, "parent_gate_dseg": _pc_ref,
                  "reproduced_dseg": _pc_now,
                  "note": "canary invisible/inconsistent — see CLAUDE.md L3 verdict-clearance"})

    # ddm_tp1 (#804) v9 telemetry PORT setup (READ-ONLY; gated => byte-identical when off).
    # All state + the reusable-producer imports live behind the flag so an OFF run has ZERO
    # new import side effects and ZERO new state. Trained/checkpoint bytes are flag-invariant
    # (flag not in cfg; new rows via tlog only, never telemetry_tail).
    _tel_v9 = (args.telemetry_v9_port == "on")
    _tel_termdom_streaks: dict[str, int] = {}
    _tel_inert_state = None
    _tel_lever_engage = None
    _tel_term_inert_rows = None
    _tel_strata: tuple[int, ...] = ()
    _tel_pos_ctrl_emitted = False
    _tel_loss_term_keys = tr1_active_loss_term_keys(
        jd1_pose_finish_active=_jd1_pose_finish_enabled,
        birth_amplify_active=(birth_seed_amplify_weight > 0.0),
    )
    _tel_scored_terms = tr1_active_scored_terms(
        jd1_pose_finish_active=_jd1_pose_finish_enabled,
    )
    if _tel_v9:
        from tac.witness_control.telemetry_producers import (
            ProducerResumeState as _TelResumeState,
            deterministic_strata as _tel_det_strata,
            lever_engage_row as _tel_lever_engage,
            term_inert_rows as _tel_term_inert_rows,
        )
        _tel_inert_state = _TelResumeState()
        _tel_strata = _tel_det_strata(cfg.num_pairs, min(8, cfg.num_pairs))
        tlog({"event": "telemetry_v9_port", "status": "on",
              "strata_ids": list(_tel_strata), "loss_term_keys": list(_tel_loss_term_keys),
              "termdom_frac": TR1_TERMDOM_FRAC, "termdom_min_rows": TR1_TERMDOM_MIN_ROWS,
              "termdom_scored_term": TR1_SCORED_TERM,
              "termdom_scored_terms": list(_tel_scored_terms),
              "termdom_scored_floor": TR1_SCORED_FLOOR,
              "note": "additive read-only rows; trained/checkpoint bytes flag-invariant "
                      "(flag via args not cfg; new rows via tlog not telemetry_tail)",
              "score_neutral": True})

    epoch = start_epoch
    while epoch < cfg.epochs:
        if time.monotonic() > deadline:
            stop_reason = "max_wall_minutes"
            tlog({"event": "wall_clock_stop", "epoch": epoch})
            break
        if (not jd1_pose_finish_state.get("engaged")
                and jd1_pose_finish_should_engage(args, epoch=epoch, stage=stage)):
            _prev_stage = stage
            _seg_hold_floor = jd1_resolve_seg_hold_floor(
                args, ep_losses=ep_losses, checkpoint_tail=boundary_parent_tail)
            jd1_pose_finish_state.update({
                "enabled": True,
                "engaged": True,
                "engaged_epoch": int(epoch),
                "engaged_stage": "joint_pose_finish",
                "engaged_global_step": int(global_step),
                "previous_stage": _prev_stage,
                "seg_hold_floor": _seg_hold_floor,
                "seg_hold_floor_source": args.jd1_seg_hold_floor_source,
                "seg_hold_margin": float(args.jd1_seg_hold_margin),
                "w_pose": float(args.jd1_w_pose),
                "effective_w_pose": float(jd1_effective_w_pose),
                "seg_hold_space": args.jd1_seg_hold_space,
                "realized_hold": dict(jd1_realized_hold_state),
                "ema_stage_scope": args.jd1_ema_stage_scope,
                "active_ema_decay": float(active_ema_decay),
                "active_ema_decay_provenance": active_ema_decay_provenance,
                "force_ema_reanchor_on_resume": bool(args.jd1_force_ema_reanchor_on_resume),
                "live_gate_telemetry": args.jd1_live_gate_telemetry,
            })
            stage = "joint_pose_finish"
            prev_gate_smooth = None
            if args.jd1_ema_stage_scope == "window":
                _apply_jd1_stage_ema_reanchor(epoch=epoch, reason="fresh_joint_pose_engagement")
            else:
                save_checkpoint(out_dir / "checkpoints" / "stage_joint_pose_finish_entry.npz",
                                model=model, ema=ema, opt_state_flat=_opt_state(), epoch=epoch,
                                stage=stage, cfg=cfg, telemetry_tail=telemetry_tail,
                                extra_meta=_jd1_checkpoint_extra_meta())
            tlog({"event": "jd1_pose_finish_engage",
                  "schema": JD1_POSE_FINISH_SCHEMA,
                  "epoch": epoch,
                  "global_step": int(global_step),
                  "previous_stage": _prev_stage,
                  "stage": stage,
                  "engage_on": args.jd1_pose_finish_engage_on,
                  "start_epoch": int(args.jd1_pose_finish_start_epoch),
                  "w_pose": float(args.jd1_w_pose),
                  "effective_w_pose": float(jd1_effective_w_pose),
                  "seg_hold_weight": float(args.jd1_seg_hold_weight),
                  "seg_hold_floor": _seg_hold_floor,
                  "seg_hold_floor_source": args.jd1_seg_hold_floor_source,
                  "seg_hold_margin": float(args.jd1_seg_hold_margin),
                  "seg_hold_space": args.jd1_seg_hold_space,
                  "ema_stage_scope": args.jd1_ema_stage_scope,
                  "force_ema_reanchor_on_resume": bool(args.jd1_force_ema_reanchor_on_resume),
                  "active_ema_decay": float(active_ema_decay),
                  "checkpoint": "checkpoints/stage_joint_pose_finish_entry.npz",
                  "note": "JD1 joint pose-finish is now active: pair_loss consumes gt_poses "
                          "and builds make_loss_fn's PoseNet path; A1 smooth baseline rebased.",
                  "score_claim": False})
            _log_jd1_ema_mode(epoch=epoch, reason="fresh_joint_pose_engagement")
            _resolve_jd1_lr_anneal_schedule(epoch=epoch,
                                            reason="fresh_joint_pose_engagement")
            _activate_jd1_muon_finisher(epoch=epoch,
                                        reason="fresh_joint_pose_engagement")
        _maybe_apply_jd1_ema_tail_anchor(epoch=epoch, reason="explicit_epoch")
        _apply_jd1_lr_anneal_for_epoch(epoch=epoch)
        perm = order_rng.permutation(cfg.num_pairs)
        ep_loss, steps = 0.0, 0
        last_gnorm = None
        for b0 in range(0, cfg.num_pairs, cfg.batch_pairs):
            ids = [int(i) for i in perm[b0:b0 + cfg.batch_pairs]]
            loss, grads = vg(model, ids)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state, loss)
            lv = float(loss)
            if not np.isfinite(lv):
                tlog({"event": "confound_alarm", "kind": "nonfinite_loss", "epoch": epoch})
                stop_reason = "nonfinite_loss"
                break
            ep_loss += lv
            steps += 1
            global_step += 1
            flat = tree_flatten(model.trainable_parameters())
            if jd1_ema_tail_average_active(jd1_pose_finish_state):
                _tail_k = int(jd1_pose_finish_state.get("ema_tail_update_count", 0))
                _tail_row = _resolve_scope_law(
                    "jd1_plateau_tail_average_ema",
                    {"updates_since_anchor": _tail_k},
                )
                _live_w = float(_tail_row["resolved_value"])
                for k, v in flat:
                    ema[k] = ema[k] + _live_w * (v - ema[k])
                jd1_pose_finish_state["ema_tail_update_count"] = _tail_k + 1
                jd1_pose_finish_state["ema_tail_last_live_weight"] = float(_live_w)
                jd1_pose_finish_state["ema_tail_last_scope_law_resolution_hash"] = (
                    _tail_row["resolution_hash"]
                )
                jd1_pose_finish_state["resolved_scope_laws"] = list(resolved_scope_laws)
            else:
                for k, v in flat:
                    ema[k] = active_ema_decay * ema[k] + (1.0 - active_ema_decay) * v
            if b0 + cfg.batch_pairs >= cfg.num_pairs:  # last batch: gnorm telemetry
                from mlx.utils import tree_flatten as _tf

                sq = 0.0
                for _k, g in _tf(grads):
                    sq += float(mx.sum(mx.square(g)))
                last_gnorm = float(np.sqrt(sq))
        if stop_reason == "nonfinite_loss":
            break
        ep_loss /= max(steps, 1)
        ep_losses.append(ep_loss)
        row = {"event": "epoch", "epoch": epoch, "stage": stage, "seg_form": state_form["form"],
               "ep_loss": ep_loss, "weights_stepped": steps > 0, "steps": steps,
               "gnorm_last_batch": last_gnorm,
               "jd1_pose_finish_active": bool(jd1_pose_finish_state.get("engaged")),
               "jd1_seg_hold_floor": jd1_pose_finish_state.get("seg_hold_floor"),
               "jd1_effective_w_pose": (float(jd1_effective_w_pose)
                                        if jd1_pose_finish_state.get("engaged") else 0.0),
               "jd1_finisher": args.jd1_finisher,
               "jd1_finisher_active": bool(jd1_finisher_active),
               "active_ema_decay": float(active_ema_decay),
               "jd1_ema_mode": args.jd1_ema_mode,
               "jd1_ema_tail_average_active": bool(
                   jd1_pose_finish_state.get("ema_tail_average_active", False)),
               "jd1_ema_tail_update_count": int(
                   jd1_pose_finish_state.get("ema_tail_update_count", 0))}
        if jd1_lr_schedule is not None:
            row.update({
                "jd1_lr_anneal": args.jd1_lr_anneal,
                "jd1_lr": float(jd1_lr_current),
                "jd1_lr_anneal_onset_epoch": int(jd1_lr_schedule["onset_epoch"]),
                "jd1_lr_final_frac": float(jd1_lr_schedule["final_frac"]),
                "jd1_lr_tail_epochs": int(jd1_lr_schedule["tail_epochs"]),
                "jd1_lr_signal_source": jd1_lr_schedule["signal_source"],
            })
        if jd1_finisher_active and jd1_muon_schedule is not None:
            row.update({
                "jd1_muon_lr": float(jd1_muon_schedule["muon_lr"]),
                "jd1_muon_final_lr": float(jd1_muon_schedule["muon_final_lr"]),
                "jd1_muon_lr_final_frac": float(jd1_muon_schedule["muon_lr_final_frac"]),
                "jd1_muon_momentum": float(jd1_muon_schedule["muon_momentum"]),
            })
        # Confound immune system (day-one, L1 runtime alarms — LOUD, never silent):
        if ep_loss == 0.0:
            tlog({"event": "confound_alarm", "kind": "frozen_epoch", "epoch": epoch,
                  "note": "ep_loss==0.0 liveness ALERT (#304 median-freeze class)"})
        if last_gnorm is not None:
            gnorm_hist.append(last_gnorm)
            if len(gnorm_hist) >= 4:
                med = float(np.median(gnorm_hist[:-1][-8:]))
                if med > 0 and last_gnorm > 100.0 * med:
                    tlog({"event": "confound_alarm", "kind": "gnorm_hijack",
                          "epoch": epoch, "gnorm": last_gnorm, "trailing_median": med})
        # Event-driven form switch (never a PR95 stage skeleton): CE knee -> tau_softplus.
        if (not knee_switched and state_form["form"] == "ce" and len(ep_losses) >= 4):
            w = ep_losses[-4:]
            rel = (w[0] - w[-1]) / max(abs(w[0]), 1e-12) / 3.0
            if rel < 0.01:
                save_checkpoint(out_dir / "checkpoints" / "stage_seg_trunk_ce_exit.npz",
                                model=model, ema=ema, opt_state_flat=_opt_state(), epoch=epoch,
                                stage=stage, cfg=cfg, telemetry_tail=telemetry_tail,
                                extra_meta=_jd1_checkpoint_extra_meta())
                state_form["form"] = "tau_softplus"
                stage = "seg_trunk_tau"
                knee_switched = True
                row["event_knee_switch"] = {"epoch": epoch, "rel_per_epoch": rel,
                                            "new_form": "tau_softplus"}
                if cfg.token_quant_anneal == "at_knee" and not model._quant_engaged:
                    # §3.3(a) lattice anneal: basin found in float -> engage the STE, refine
                    # on the shipped lattice. The token forward snaps to the quantized lattice;
                    # the A1 basis rebase below (prev_gate_smooth=None) absorbs the scale jump.
                    model._quant_engaged = True
                    row["event_quant_engage"] = {"epoch": epoch, "trigger": "ce_tau_knee"}
                # A1 basis REBASE: the smooth-loss SCALE changes with the form —
                # comparing tau_softplus loss against a CE baseline would fire a
                # FALSE realization-gap alarm at the next gate. One-gate rebase.
                prev_gate_smooth = None
                # ax1 §4a delta-sparsity ENGAGE (base-stability event = the CE->tau knee): shrink
                # the per-pair deltas now that the shared base has found its basin (§7: shrinking
                # deltas against a moving base is noise). Event-driven, never epoch-hardcoded.
                if (cfg.token_delta_group_sparsity == "on"
                        and cfg.delta_sparsity_engage == "after_base_stability"
                        and not model._delta_sparsity_engaged):
                    model._delta_sparsity_engaged = True
                    row["event_delta_sparsity_engage"] = {
                        "epoch": epoch, "trigger": "ce_tau_knee_base_stable"}
        # F2 EVENT-FALLBACK (triggers-forces P0): if the CE knee never fires, the
        # form switch still fires at the window midpoint — an event with a fallback,
        # never a stranded stage (recorded as fallback, distinct from the knee).
        if not knee_switched and state_form["form"] == "ce" and epoch >= cfg.epochs // 2:
            save_checkpoint(out_dir / "checkpoints" / "stage_seg_trunk_ce_exit.npz",
                            model=model, ema=ema, opt_state_flat=_opt_state(), epoch=epoch,
                            stage=stage, cfg=cfg, telemetry_tail=telemetry_tail,
                            extra_meta=_jd1_checkpoint_extra_meta())
            state_form["form"] = "tau_softplus"
            stage = "seg_trunk_tau"
            knee_switched = True
            prev_gate_smooth = None
            row["event_knee_fallback"] = {"epoch": epoch, "kind": "F2_midpoint_fallback"}
            if cfg.token_quant_anneal == "at_knee" and not model._quant_engaged:
                model._quant_engaged = True  # §3.3(a) engage STE at the F2 fallback knee too
                row["event_quant_engage"] = {"epoch": epoch, "trigger": "F2_midpoint_fallback"}
            if (cfg.token_delta_group_sparsity == "on"
                    and cfg.delta_sparsity_engage == "after_base_stability"
                    and not model._delta_sparsity_engaged):
                model._delta_sparsity_engaged = True  # ax1 §4a engage at the F2 fallback knee too
                row["event_delta_sparsity_engage"] = {
                    "epoch": epoch, "trigger": "F2_midpoint_fallback"}
        tlog(row)
        telemetry_tail.append(checkpoint_safe_telemetry_row(row))

        # ddm_tp1 (#804) Q7 lever_engage COMPANIONS: for every event fired into the epoch
        # row this epoch, ALSO emit the canonical uniform {stage:lever_engage,...} row (v9
        # schema) so a reader sees engages in ONE vocabulary. Reads the already-built `row`
        # (no new event logic); tlog-only + gated => byte-identical when off.
        if _tel_v9:
            for _evk, _lever, _via in (
                ("event_knee_switch", "seg_form_ce_to_tau", "ce_tau_knee"),
                ("event_knee_fallback", "seg_form_ce_to_tau", "F2_midpoint_fallback"),
                ("event_quant_engage", "token_quant_ste", "quant_anneal"),
                ("event_delta_sparsity_engage", "token_delta_group_sparsity",
                 "delta_sparsity_engage"),
            ):
                if _evk in row:
                    _ev = row[_evk]
                    _extra = {"source_event": _evk}
                    if isinstance(_ev, dict) and "trigger" in _ev:
                        _extra["trigger"] = _ev["trigger"]
                    tlog(_tel_lever_engage(_lever, status="fired", epoch=epoch,
                                           via=_via, extra=_extra))

        # A1 realized gate. Basis = EMA shadow once warm (W = 2/(1-d) updates), LIVE
        # params before that (the #85 shadow-lag guard above; basis recorded, LOUD).
        if (epoch + 1) % cfg.gate_every == 0 or epoch == cfg.epochs - 1:
            gate_basis = jd1_ema_gate_basis_label(
                global_step=global_step,
                ema_warmup_updates=ema_warmup_updates,
                state=jd1_pose_finish_state,
            )
            live_np = {k: np.asarray(v) for k, v in tree_flatten(model.trainable_parameters())}
            if args.jd1_live_gate_telemetry == "on":
                live_gate_row = realized_gate(model, gate_ids, lstars, seg_cpu, None)
                live_gate_row.pop("_realized_argmax", None)
                tlog({
                    "event": "jd1_live_basis_gate",
                    "epoch": epoch,
                    "stage": stage,
                    "gate_params": "live_weights",
                    "active_ema_decay": float(active_ema_decay),
                    "jd1_ema_mode": args.jd1_ema_mode,
                    "jd1_ema_tail_update_count": int(
                        jd1_pose_finish_state.get("ema_tail_update_count", 0)),
                    "jd1_ema_tail_average_active": bool(
                        jd1_pose_finish_state.get("ema_tail_average_active", False)),
                    "effective_w_pose": (float(jd1_effective_w_pose)
                                         if jd1_pose_finish_state.get("engaged") else 0.0),
                    "score_claim": False,
                    **live_gate_row,
                })
            live = ema_snapshot_swap(model, ema) if gate_basis != "live_ema_warmup" else None
            try:
                gate_row = realized_gate(
                    model, gate_ids, lstars, seg_cpu, prev_realized,
                    pose_adapter=adapter, gt_poses=gt_poses)
                ledger = counted_bytes_ledger(model, cfg)
            finally:
                if live is not None:
                    ema_restore(model, live)
            realized_argmax = gate_row.pop("_realized_argmax")
            gate_row["gate_params"] = gate_basis
            gate_row["ema_warmup_updates"] = ema_warmup_updates
            gate_row["global_step"] = global_step
            gate_row["jd1_ema_mode"] = args.jd1_ema_mode
            gate_row["jd1_ema_tail_update_count"] = int(
                jd1_pose_finish_state.get("ema_tail_update_count", 0))
            gate_row["jd1_ema_tail_average_active"] = bool(
                jd1_pose_finish_state.get("ema_tail_average_active", False))
            gate_row["jd1_ema_tail_anchor_epoch"] = jd1_pose_finish_state.get(
                "ema_tail_anchor_epoch")
            # Basis change (live->shadow) REBASES the A1 comparison + flip baseline.
            if prev_gate_basis is not None and gate_basis != prev_gate_basis:
                prev_gate_smooth = None
                gate_row.pop("realized_flips_vs_prev_gate", None)
            prev_gate_basis = gate_basis
            # #685 px1 race-fairness telemetry: MEASURED update magnitude per arm —
            # RMS of the live-param delta accumulated since the previous gate.
            if gate_param_snapshot is not None:
                num = sum(float(np.sum((live_np[k] - gate_param_snapshot[k]) ** 2))
                          for k in live_np)
                den = sum(v.size for v in live_np.values())
                gate_row["param_delta_rms_since_prev_gate"] = float(np.sqrt(num / max(den, 1)))
            gate_param_snapshot = live_np
            # ddm_pa1r confound fix: the A1 smooth input EXCLUDES the delta-sparsity penalty
            # (the engage transient's penalty relaxation is not seg progress; feeding raw
            # ep_loss fired a false a1_realization_gap_refuse at ep454 on the w=0.03 arm).
            # Penalty computed full-P on the LIVE params (ep_loss's own basis); byte-identical
            # when the lever is off (a1_smooth == ep_loss exactly; no penalty forward run).
            a1_smooth = ep_loss
            if model._delta_sparsity_engaged and cfg.delta_sparsity_weight > 0.0:
                pen_now = float(delta_sparsity_term(model, list(range(cfg.num_pairs))))
                a1_smooth = a1_smooth_excluding_delta_penalty(
                    ep_loss, True, cfg.delta_sparsity_weight, pen_now)
                gate_row["delta_penalty_now"] = pen_now
                gate_row["a1_smooth_input"] = a1_smooth
            a1 = a1_adjudicate(prev_gate_row, gate_row, prev_gate_smooth, a1_smooth)
            gate_row.update(a1)
            gate_row.update(ledger)
            gate_row.update({"event": "a1_gate", "epoch": epoch, "ep_loss": ep_loss,
                             "weights_stepped": True, "stage": stage,
                             "seg_form": state_form["form"]})
            # ddm_bp1 (#824) per-INTERVAL decomposition, emitted at write time so the 63-interval
            # analysis is reproducible from telemetry alone (post-hoc pairing is what hid the
            # restart effect for a whole burn). Free: derived from values already in hand.
            gate_row.update(gate_interval_fields(prev_gate_row, gate_row))
            gate_row["epochs_since_resume"] = epoch - start_epoch
            gate_row["reset_arm"] = resolve_arm_name(_reset_arm)
            tlog(gate_row)
            # The BOUNDARY JUMP itself: the FIRST post-resume interval, isolated. ~35% of the
            # corrected seg descent lived in this one short interval last burn; an end-state
            # readout averages it into ~140 training epochs and dilutes it below resolution.
            if not boundary_jump_emitted and args.resume_from is not None:
                _bj = boundary_jump_row(
                    boundary_parent_tail, boundary_parent_ema_decay, active_ema_decay,
                    start_epoch, gate_row, resolve_arm_name(_reset_arm) or "unknown",
                    parent_cfg_ema_decay=boundary_parent_cfg_ema_decay)
                boundary_jump_emitted = True
                if _bj is not None:
                    boundary_strict_basis_held = bool(_bj["ema_basis_held"])
                    boundary_gate_basis_held = bool(_bj["gate_basis_held"])
                    if not _bj["gate_basis_held"]:
                        # L1 runtime alarm — LOUD, never silent. The parent and child gate
                        # readings came off DIFFERENT weight sets (live vs EMA shadow), so the
                        # boundary_dseg_delta on this row is a BASIS SWITCH, not a training
                        # change, and no cross-boundary d_seg comparison may use it.
                        tlog({"event": "confound_alarm", "kind": "gate_basis_switch",
                              "epoch": int(gate_row.get("epoch", start_epoch)),
                              "parent_gate_basis": _bj["parent_gate_basis"],
                              "first_gate_basis": _bj["first_gate_basis"],
                              "boundary_dseg_delta": _bj["boundary_dseg_delta"],
                              "note": "parent and child gate readings are taken off DIFFERENT "
                                      "objects (live weights vs EMA shadow) — the apparent jump "
                                      "is a readout basis switch. Compare SAME-basis windows "
                                      "only; ema_basis_held is False on this boundary."})
                    # MEASUREMENT-BASIS invariant (round-2): a RESUMED run reports the ema_shadow
                    # basis from its FIRST gate while a FRESH run reads live_ema_warmup for U/2
                    # updates — so a fresh arm and a resumed arm are read on DIFFERENT instruments
                    # and their comparison is void. Recorded here so a comparer can check it; the
                    # ticket builder enforces sameness (only it can see both arms).
                    _bj["gate_basis_mode"] = "resumed_warm_shadow"
                    # INCLUDE the current gate: telemetry_tail is appended AFTER this block, and
                    # at the FIRST post-resume gate the tail is otherwise empty — the summary
                    # would report 0 alarms even when THIS gate alarmed. (Caught in round-2
                    # self-review of my own instrumentation.)
                    _bj["a1_alarms"] = a1_alarm_summary([*telemetry_tail, gate_row])
                    tlog(_bj)
            # ddm_bs3 (#909): the new decomposition fields go to telemetry.jsonl (tlog,
            # above) ONLY -- never into telemetry_tail, which is BAKED INTO THE CHECKPOINT
            # meta (:1602). Keeping them out preserves the trainer's standing
            # checkpoint-byte-invariance law (the same reason the v9 telemetry port is
            # tlog-only). Nothing is lost: prev_gate_row is the in-process local (:2726),
            # so the guard still sees the previous gate's per-class vector.
            telemetry_tail.append(checkpoint_safe_telemetry_row(gate_row))
            print(json.dumps({k: gate_row[k] for k in
                              ("epoch", "realized_gate_dseg_mean", "a1_classification",
                               "total_counted_bytes")}), flush=True)
            if (jd1_realized_hold_state.get("active")
                    and jd1_pose_finish_state.get("engaged")):
                if jd1_realized_hold_state.get("floor") is None:
                    _floor_row = _resolve_scope_law("jd3_realized_hold_floor_latch", {
                        "realized_gate_dseg_mean": gate_row["realized_gate_dseg_mean"],
                    })
                    _margin_row = _resolve_scope_law("jd3_realized_hold_margin", {
                        "explicit_margin": float(args.jd1_realized_hold_margin),
                        "realized_gate_dseg_per_pair_sd": gate_row.get(
                            "realized_gate_dseg_per_pair_sd"),
                        "realized_gate_pair_ids": gate_row.get("realized_gate_pair_ids"),
                    })
                    jd1_realized_hold_state.update({
                        "floor": float(_floor_row["resolved_value"]),
                        "floor_epoch": int(epoch),
                        "floor_gate_basis": gate_basis,
                        "margin": float(_margin_row["resolved_value"]),
                        "margin_provenance": str(_margin_row["provenance"]),
                        "floor_provenance": str(_floor_row["provenance"]),
                        "scope_law_resolution_hashes": [
                            *list(jd1_realized_hold_state.get(
                                "scope_law_resolution_hashes") or []),
                            _floor_row["resolution_hash"],
                            _margin_row["resolution_hash"],
                        ],
                    })
                    jd1_pose_finish_state["realized_hold"] = dict(jd1_realized_hold_state)
                    jd1_pose_finish_state["resolved_scope_laws"] = list(resolved_scope_laws)
                    tlog({"event": "jd1_realized_hold_latch",
                          "epoch": epoch,
                          "floor": float(jd1_realized_hold_state["floor"]),
                          "margin": float(jd1_realized_hold_state["margin"]),
                          "margin_provenance": jd1_realized_hold_state["margin_provenance"],
                          "floor_provenance": jd1_realized_hold_state["floor_provenance"],
                          "scope_law_resolution_hashes": [
                              _floor_row["resolution_hash"],
                              _margin_row["resolution_hash"],
                          ],
                          "gate_basis": gate_basis,
                          "score_claim": False})
                else:
                    _floor = float(jd1_realized_hold_state["floor"])
                    _margin = float(jd1_realized_hold_state["margin"] or 0.0)
                    _limit = _floor + _margin
                    _current = float(gate_row["realized_gate_dseg_mean"])
                    if _current > _limit:
                        _retreats = int(jd1_realized_hold_state.get("retreats", 0))
                        _max_retreats = int(jd1_realized_hold_state.get("max_retreats", 0))
                        if prev_gate_snapshot is None:
                            tlog({"event": "jd1_realized_hold_refuse",
                                  "epoch": epoch,
                                  "reason": "breach_without_previous_gate_snapshot",
                                  "realized_dseg": _current,
                                  "limit": _limit,
                                  "score_claim": False})
                            stop_reason = "jd1_realized_hold_no_previous_snapshot"
                            break
                        if _retreats >= _max_retreats:
                            tlog({"event": "jd1_realized_hold_refuse",
                                  "epoch": epoch,
                                  "reason": "max_retreats_exhausted",
                                  "retreats": _retreats,
                                  "max_retreats": _max_retreats,
                                  "realized_dseg": _current,
                                  "limit": _limit,
                                  "score_claim": False})
                            stop_reason = "jd1_realized_hold_exhausted"
                            break
                        _rollback_epoch = int(prev_gate_snapshot["epoch"])
                        _old_w_pose = float(jd1_effective_w_pose)
                        _new_w_pose = _old_w_pose * float(
                            jd1_realized_hold_state["pose_retreat_factor"])
                        _restore_jd1_gate_snapshot(prev_gate_snapshot)
                        jd1_effective_w_pose = float(_new_w_pose)
                        jd1_realized_hold_state["retreats"] = (
                            int(jd1_realized_hold_state.get("retreats", 0)) + 1)
                        jd1_realized_hold_state.setdefault("history", []).append({
                            "breach_epoch": int(epoch),
                            "rollback_to_epoch": _rollback_epoch,
                            "breach_dseg": _current,
                            "limit": _limit,
                            "old_w_pose": _old_w_pose,
                            "new_w_pose": float(jd1_effective_w_pose),
                        })
                        jd1_pose_finish_state.update({
                            "effective_w_pose": float(jd1_effective_w_pose),
                            "realized_hold": dict(jd1_realized_hold_state),
                        })
                        tlog({"event": "jd1_realized_hold_rollback",
                              "breach_epoch": int(epoch),
                              "rollback_to_epoch": _rollback_epoch,
                              "next_epoch": _rollback_epoch + 1,
                              "breach_dseg": _current,
                              "limit": _limit,
                              "old_w_pose": _old_w_pose,
                              "new_w_pose": float(jd1_effective_w_pose),
                              "retreat_factor": float(
                                  jd1_realized_hold_state["pose_retreat_factor"]),
                              "retreats": int(jd1_realized_hold_state["retreats"]),
                              "max_retreats": int(jd1_realized_hold_state["max_retreats"]),
                              "score_claim": False})
                        epoch = _rollback_epoch + 1
                        continue
            if a1["a1_alarm"]:
                a1_consecutive += 1
                if a1_consecutive >= A1_CONSECUTIVE_REFUSE:
                    tlog({"event": "a1_stage_exit_refuse", "epoch": epoch,
                          "consecutive_alarms": a1_consecutive,
                          "note": "fd2 inherited gap signature — REROUTE, never scale"})
                    stop_reason = "a1_realization_gap_refuse"
                    break
            else:
                a1_consecutive = 0
            # ddm_lg1 (#808): dual ascent + protection refresh at GATE cadence, reading the
            # gate's EXISTING realized argmax (zero new scorer passes). lambda is CONSTANT
            # between gates (caps-law reconciliation: dual variables update at constraint-
            # evaluation cadence with a bounded step, never per-step). Gated => byte-identical
            # when off (block skipped; telemetry untouched).
            if cfg.lane_guard:
                _lg_gts = np.stack([np.asarray(lstars[i], dtype=np.int64) for i in gate_ids])
                _lg_margins = None
                if (lane_guard_cfg.margin_floor_weight > 0.0
                        and lane_guard_state.margin_floor is None):
                    _lg_margins = {int(i): np.asarray(margins[i], dtype=np.float32)
                                   for i in gate_ids}
                _lg_row = _lane_guard.gate_update(
                    lane_guard_state, lane_guard_cfg, realized_argmax, _lg_gts,
                    gate_ids, lane_margins_by_id=_lg_margins)
                _lg_row.update({"epoch": epoch, "stage": stage, "gate_basis": gate_basis})
                tlog(_lg_row)
            prev_gate_row, prev_gate_smooth, prev_realized = gate_row, a1_smooth, realized_argmax
            save_checkpoint(out_dir / "checkpoints" / f"intra_{stage}_ep{epoch:05d}.npz",
                            model=model, ema=ema, opt_state_flat=_opt_state(), epoch=epoch,
                            stage=stage, cfg=cfg, telemetry_tail=telemetry_tail,
                            extra_meta=_jd1_checkpoint_extra_meta())
            prev_gate_snapshot = _capture_jd1_gate_snapshot(epoch)

            # ddm_tp1 (#804) v9 telemetry PORT emissions (READ-ONLY; gated => byte-identical
            # when off). Params are LIVE here (the EMA-shadow gate swap was restored in the
            # gate's finally). The per-term recompute runs the SAME deterministic forwards the
            # loss uses on a small fixed strata subset (no order_rng, no mx.random, no
            # model/opt mutation) and NEVER touches telemetry_tail => the checkpoint just
            # written is byte-identical to an off run.
            if _tel_v9:
                _tp_ids = [int(i) for i in _tel_strata]
                _tp_pair_terms: dict[str, Any] = {}
                _tp_pose_active = bool(jd1_pose_finish_state.get("engaged"))
                for i in _tp_ids:
                    _one_terms: dict[str, Any] = {}
                    pair_loss(
                        model, i, state_form["form"],
                        pose_active=_tp_pose_active,
                        terms_out=_one_terms,
                    )
                    for _name, _value in _one_terms.items():
                        _tp_pair_terms[_name] = (
                            _value if _name not in _tp_pair_terms
                            else _tp_pair_terms[_name] + _value
                        )
                _tp_terms = {
                    _name: float(_value / max(len(_tp_ids), 1))
                    for _name, _value in _tp_pair_terms.items()
                }
                _tp_rate = (float(cfg.w_rate * token_rate_term(model, _tp_ids))
                            if cfg.w_rate > 0.0 else 0.0)
                _tp_ds = (float(cfg.delta_sparsity_weight
                                * delta_sparsity_term(model, _tp_ids))
                          if (model._delta_sparsity_engaged
                              and cfg.delta_sparsity_weight > 0.0) else 0.0)
                _tp_birth = (
                    float(birth_seed_amplify_weight * tr1_birth_amplify_term(model, _tp_ids))
                    if birth_seed_amplify_weight > 0.0 else 0.0
                )
                _tp_terms["rate"] = _tp_rate
                _tp_terms["delta_sparsity"] = _tp_ds
                if birth_seed_amplify_weight > 0.0:
                    _tp_terms["birth_amplify"] = _tp_birth
                _tp_total = sum(float(_tp_terms.get(k, 0.0)) for k in _tel_loss_term_keys)
                tlog(tr1_loss_terms_row(
                    _tp_terms, _tp_total, ep=epoch, accum_batch=steps,
                    accepted_frac=(1.0 if steps > 0 else 0.0),
                    weights_stepped=(steps > 0), stage=stage,
                    seg_form=state_form["form"],
                    loss_term_keys=_tel_loss_term_keys))
                for _dom in tr1_term_domination_alarms(_tp_terms, _tp_total,
                                                       _tel_termdom_streaks,
                                                       loss_term_keys=_tel_loss_term_keys,
                                                       scored_terms=_tel_scored_terms):
                    tlog(_dom)
                _tp_engaged = {
                    "seg": True,
                    "rate": bool(cfg.w_rate > 0.0),
                    "delta_sparsity": bool(model._delta_sparsity_engaged
                                           and cfg.delta_sparsity_weight > 0.0),
                    "pose": bool(_jd1_pose_finish_enabled
                                 and jd1_pose_finish_state.get("engaged")),
                    "birth_amplify": bool(birth_seed_amplify_weight > 0.0),
                }
                for _inert in _tel_term_inert_rows(
                        _tp_terms, engaged=_tp_engaged, epoch=epoch,
                        state=_tel_inert_state):
                    tlog(_inert)
                if not _tel_pos_ctrl_emitted:
                    # (#404) $0 synthetic known-effect sentinel: the verdict-trend instrument
                    # MUST register a known d_seg descent AND stay quiet on a co-descending
                    # run. If clearance is False the trend meter is UNTRUSTED for this run.
                    from tac.witness_control.verdict_trend_alarm import (
                        canary_suite as _tel_canary,
                    )
                    _pc = _tel_canary()
                    tlog({"stage": "positive_control", "epoch": epoch,
                          "canary_passed": bool(_pc.passed),
                          "verdict_clearance": bool(_pc.verdict_clearance()),
                          "descent_positive_registered": bool(
                              _pc.descent_positive_registered),
                          "negative_fired": bool(_pc.negative_fired),
                          "reason": _pc.reason,
                          "note": "#404 known-effect verdict-trend sentinel; clearance False "
                                  "=> the trend instrument is UNTRUSTED for this run",
                          "score_neutral": True})
                    _tel_pos_ctrl_emitted = True

            # ---- BASIN-ENTRY HANDOFF (operator ×2 2026-07-28: "train only to condition;
            # if basin hit, solve only, preferable always"). Detection = the TerminalSolve
            # §16.1 validity conditions, honestly disambiguated: (a) quadratic crawl in
            # BOTH smooth and realized channels; (b) partition topology STABLE (lane =
            # the live nucleation channel); (c) form switch done + EMA shadow warm (no
            # transitions remaining); plus zero A1 alarms in-window (COUPLED_DESCENT =
            # the smooth-realized linearization fidelity that fd2 measured MISSING on
            # the unconditioned pixel-lattice lift — conditioning creates solve-validity).
            # Saddle/grokking (#216/#475) disambiguation is POST-solve by contract (see
            # the handoff receipt): a stalled solve + still-descending training = saddle
            # => resume training, re-arm doubled window. ----
            if cfg.basin_handoff == "on":
                topo = gate_row.get("topology_per_class", {})
                basin_window.append({
                    # ddm_tp2: ``form`` is the PREDICATE key (the loss form the state machine is
                    # actually running, launch-flag-invariant); ``stage`` is retained for the
                    # receipt/telemetry as a human label ONLY -- never compared.
                    "epoch": epoch, "basis": gate_basis, "stage": stage,
                    "form": state_form["form"],
                    "dseg": float(gate_row["realized_gate_dseg_mean"]),
                    "smooth": float(ep_loss), "alarm": bool(a1["a1_alarm"]),
                    "lane_b0": int(topo.get("betti0_realized", [0] * 5)[1]),
                    "lane_er": int(topo.get("gt_components_erased", [0] * 5)[1])})
                basin_window = basin_window[-3:]
                w = basin_window
                if basin_entry_fires(w):
                    save_checkpoint(out_dir / "checkpoints" / "stage_basin_entry.npz",
                                    model=model, ema=ema, opt_state_flat=_opt_state(), epoch=epoch,
                                    stage="basin_entry", cfg=cfg,
                                    telemetry_tail=telemetry_tail,
                                    extra_meta=_jd1_checkpoint_extra_meta())
                    handoff = {
                        "schema": "ddm_lv1_basin_handoff_receipt.v1",
                        "fired_epoch": epoch, "window": w,
                        "rule": ("operator 2026-07-28: TRAIN ONLY TO CONDITION; on "
                                 "basin-entry switch to SOLVE-ONLY permanently"),
                        "validity": ("TerminalSolve §16.1 (a)+(b)+(c) + zero-alarm window "
                                     "(linearization fidelity; fd2's empty faithful-flip "
                                     "window was the UNCONDITIONED pixel-lattice lift)"),
                        "executors": {
                            "quadratic_solve": ("tools/quadratic_basin_finisher_probe.py "
                                                "(#423 damped Newton-CG, head+full stages; "
                                                "per-pair token blocks separable given the "
                                                "frozen renderer)"),
                            "discrete_rail": ("eg1 E3 crash-safe QDBS terminal rail "
                                              "(cf7172e747; all-49 closure 218ed874c7)"),
                            "pose": "#383 terminal 6-eq GN on frozen composed frames"},
                        "acceptance": ("v19 REALIZED: solve steps accepted ONLY on realized "
                                       "joint dS<0 through the flip gate, vs the handoff "
                                       "full-confirm baseline — never smooth say-so"),
                        "saddle_disambiguation": ("#216/#475: solve STALLS while training "
                                                  "resumed from stage_basin_entry.npz still "
                                                  "descends => saddle/grokking plateau => "
                                                  "RESUME training, re-arm doubled window"),
                        "thresholds": {
                            **BASIN_THRESHOLDS,
                            "provenance": ("PROVISIONAL-derived: separates the §16.1 "
                                           "measured witness quadratic crawl (~0.2%/25ep) "
                                           "from tb1 T2 active descent (~5-8%/gate) by "
                                           "~10x on each side; rederivation trigger = the "
                                           "burn's own gate-delta distribution")},
                        "checkpoint": "checkpoints/stage_basin_entry.npz",
                    }
                    (out_dir / "basin_handoff_receipt.json").write_text(
                        json.dumps(handoff, indent=2, sort_keys=True) + "\n")
                    tlog({"event": "basin_entry_handoff", "epoch": epoch,
                          "window": w, "checkpoint": handoff["checkpoint"]})
                    stop_reason = "basin_entry_handoff"
                    break

        epoch += 1

    # Terminal stage checkpoint (distinct stage-encoded name; EMA shadow inside).
    save_checkpoint(out_dir / "checkpoints" / f"stage_{stage}_final.npz",
                    model=model, ema=ema, opt_state_flat=_opt_state(), epoch=len(ep_losses) + start_epoch,
                    stage=stage, cfg=cfg, telemetry_tail=telemetry_tail,
                    extra_meta=_jd1_checkpoint_extra_meta())

    receipt: dict[str, Any] = {
        "schema": "ddm_tb1_tr1_window_receipt.v1",
        "pointer": POINTER_LINE, "score_claim": False, "promotion_eligible": False,
        "evidence_axis": "[macOS-CPU/MLX advisory]",
        "variant": cfg.variant, "config_hash": cfg.config_hash(), "cfg": asdict(cfg),
        "stop_reason": stop_reason, "epochs_ran": len(ep_losses),
        "final_ep_loss": ep_losses[-1] if ep_losses else None,
        "final_gate": {k: v for k, v in (prev_gate_row or {}).items() if not k.startswith("_")},
        "elapsed_seconds": time.monotonic() - started,
        # ddm_bp1 (#824): the reset arm + the A1 alarm channel as FIRST-CLASS receipt fields.
        # Every A1 alarm of the burn was invisible to every decision record because only
        # `final_gate_a1` was propagated and none of the six firings was at a final gate.
        "reset_arm": resolve_arm_name(_reset_arm),
        "reset_operator": _reset_arm.describe(),
        "a1_alarms": a1_alarm_summary(telemetry_tail),
        "gate_basis_mode": (
            "plateau_tail_average" if jd1_ema_tail_average_active(jd1_pose_finish_state)
            else ("resumed_warm_shadow" if args.resume_from is not None
                  else "fresh_live_then_shadow")
        ),
        # ddm_op2 (OP2-2): the receipt reports the STRICT decay+basis verdict once the first
        # post-resume gate has observed the basis; before that it can only report the decay leg,
        # and it says which one it is rather than presenting the weaker leg under the strict name.
        "ema_basis_held": (bool(boundary_strict_basis_held)
                           if boundary_strict_basis_held is not None
                           else bool(boundary_ema_held)),
        "ema_decay_held": bool(boundary_ema_held),
        "gate_basis_held": boundary_gate_basis_held,
        "held_scope": ("decay_and_gate_basis" if boundary_strict_basis_held is not None
                       else "decay_only_no_post_resume_gate_observed"),
        # amendment-3 (#874/#935): the receipt SELF-ADJUDICATES its boundary — measured
        # tail-slope verdict, so no reader mistakes a capped endpoint for convergence.
        "tail_slope_adjudication": tail_slope_adjudication(telemetry_tail),
        "boundary_probe": args.boundary_probe,
        "pe3_conditioning": (
            pe3_conditioning_summary
            if pe3_conditioning_summary is not None
            else {"active": False, "mode": "off", "score_claim": False}
        ),
        "cheapdct4_pose_accounting": (
            cheapdct4_pose_accounting
            if cheapdct4_pose_accounting is not None
            else {"active": False, "mode": "off", "score_claim": False}
        ),
        "jd1_pose_finish": {
            **dict(jd1_pose_finish_state),
            "mode": args.jd1_pose_finish_mode,
            "engage_on": args.jd1_pose_finish_engage_on,
            "seg_hold_weight": float(args.jd1_seg_hold_weight),
            "seg_hold_space": args.jd1_seg_hold_space,
            "effective_w_pose": float(jd1_effective_w_pose),
            "realized_hold": dict(jd1_realized_hold_state),
            "ema_stage_scope": args.jd1_ema_stage_scope,
            "ema_mode": args.jd1_ema_mode,
            "active_ema_decay": float(active_ema_decay),
            "active_ema_decay_provenance": active_ema_decay_provenance,
            "finisher": (
                dict(jd1_muon_schedule)
                if jd1_muon_schedule is not None
                else {"schema": JD1_FINISHER_SCHEMA, "mode": args.jd1_finisher,
                      "active": False, "score_claim": False}
            ),
            "force_ema_reanchor_on_resume": bool(args.jd1_force_ema_reanchor_on_resume),
            "live_gate_telemetry": args.jd1_live_gate_telemetry,
            "score_claim": False,
        },
    }

    # Optional full realized confirm (chunked <=120; EMA shadow).
    if args.full_confirm and stop_reason in ("epochs_complete", "max_wall_minutes",
                                             "basin_entry_handoff"):
        from experiments.train_witness_realized_through_R_mlx import (
            _torch_R_to_camera_uint8,
            cpu_verdict_d_seg_batch,
        )

        confirm_basis = jd1_ema_gate_basis_label(
            global_step=global_step,
            ema_warmup_updates=ema_warmup_updates,
            state=jd1_pose_finish_state,
        )
        live = ema_snapshot_swap(model, ema) if confirm_basis != "live_ema_warmup" else None
        try:
            t0 = time.monotonic()
            all_dsegs: list[float] = []
            for c0 in range(0, cfg.num_pairs, args.verdict_chunk):
                idxs = list(range(c0, min(c0 + args.verdict_chunk, cfg.num_pairs)))
                frames = []
                with mx.stream(mx.cpu):
                    for i in idxs:
                        rgb = model.render_frame(i)
                        mx.eval(rgb)
                        frames.append(np.asarray(rgb, dtype=np.float32)[0])
                cams = [_torch_R_to_camera_uint8(f) for f in frames]
                gts = [np.asarray(lstars[i], dtype=np.int64) for i in idxs]
                all_dsegs.extend(cpu_verdict_d_seg_batch(seg_cpu, cams, gts))
            receipt["full_confirm"] = {
                "n_pairs": cfg.num_pairs,
                "realized_dseg_mean": float(np.mean(all_dsegs)),
                "realized_dseg_max": float(np.max(all_dsegs)),
                "wall_seconds": time.monotonic() - t0,
                "verdict_chunk": args.verdict_chunk,
                "confirm_params": confirm_basis,
            }
        finally:
            if live is not None:
                ema_restore(model, live)

    # §3.5 QA77-LITE composed-S ENDPOINT verdict (co9 Knee-A pricing). VERDICT-level: does NOT
    # change any trained token/weight/byte (the burn is seg-only). Fires only when the operator
    # enables it (composed_s_gate_subset>0). Fails GRACEFULLY (advisory; NEVER crashes the burn).
    # ADOPTED form (MAIN Option A 2026-07-30) = the DEGRADED DIRECTIONAL-DELTA when a delta-ref
    # is set: d_pose(GT_f0, burn_f1) - baseline, DIRECTIONAL ONLY (sign+trend of the Knee-A
    # pose-recoverability externality; NEVER an absolute pose_contrib/endpoint S — the absolute
    # bounded solve is INSTANCE-DEAD on this vehicle, measured across 4 solvers). Without a
    # delta-ref it falls back to the absolute composed_s (reference-only, not endpoint-acceptance).
    if cfg.composed_s_gate_subset > 0:
        try:
            from experiments.ddm_composed_s_verdict import ComposedSVerdict
            from experiments.train_witness_realized_through_R_mlx import (
                _torch_R_to_camera_uint8,
                cpu_verdict_d_seg_batch,
            )

            n_sub = min(int(cfg.composed_s_gate_subset), cfg.num_pairs)
            delta_ref = (np.load(cfg.composed_s_delta_ref)
                         if cfg.composed_s_delta_ref is not None else None)
            if delta_ref is not None:  # ADOPTED: knee-A tail from the reference table
                sub_ids = [int(i) for i in delta_ref["tail_ids"].ravel()[:n_sub]
                           if 0 <= int(i) < cfg.num_pairs]
            elif cfg.composed_s_subset_ids is not None:  # MAIN QA66 pose-mass tail
                sub_ids = [int(i) for i in np.load(cfg.composed_s_subset_ids).ravel()[:n_sub]
                           if 0 <= int(i) < cfg.num_pairs]
            else:
                sub_ids = list(range(n_sub))
            cbasis = jd1_ema_gate_basis_label(
                global_step=global_step,
                ema_warmup_updates=ema_warmup_updates,
                state=jd1_pose_finish_state,
            )
            live = ema_snapshot_swap(model, ema) if cbasis != "live_ema_warmup" else None
            try:
                frames = []
                with mx.stream(mx.cpu):
                    for i in sub_ids:
                        rgb = model.render_frame(i)
                        mx.eval(rgb)
                        frames.append(np.asarray(rgb, dtype=np.float32)[0])
                cams = [_torch_R_to_camera_uint8(f) for f in frames]
                gts = [np.asarray(lstars[i], dtype=np.int64) for i in sub_ids]
                dseg_sub = float(np.mean(cpu_verdict_d_seg_batch(seg_cpu, cams, gts)))
                total_bytes = counted_bytes_ledger(model, cfg)["total_counted_bytes"]
            finally:
                if live is not None:
                    ema_restore(model, live)
            verdict = ComposedSVerdict(cfg.num_pairs)
            if not verdict.available:
                receipt["composed_s_verdict"] = {"skipped": verdict.reason,
                                                 "score_claim": False}
            elif delta_ref is not None:  # ADOPTED directional-delta
                gt_f0 = open_stored_npy_memmap(args.gt_cache, "gt_f0")
                gt_f0_sub = [np.asarray(gt_f0[i], dtype=np.uint8) for i in sub_ids]
                receipt["composed_s_verdict"] = verdict.delta_verdict(
                    sub_ids, cams, gt_f0_sub, delta_ref["baseline_dpose"],
                    dseg_sub, total_bytes)
            else:  # reference-only absolute solve (NOT endpoint acceptance)
                receipt["composed_s_verdict"] = verdict.composed_s(
                    sub_ids, cams, dseg_sub, total_bytes)
        except Exception as exc:  # advisory instrument NEVER crashes the burn
            receipt["composed_s_verdict"] = {"error": repr(exc), "score_claim": False}

    attach_cheapdct4_accounting_to_receipt(receipt, cheapdct4_pose_accounting)

    rp = out_dir / "tr1_window_receipt.json"
    tmp = rp.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(str(tmp), str(rp))
    print(json.dumps({"receipt": str(rp), "stop_reason": stop_reason,
                      "score_claim": False}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
