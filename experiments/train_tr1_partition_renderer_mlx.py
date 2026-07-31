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

Pose: TERMINAL (#383) — ``pose_objective_weight=0`` on the seg trunk; NO PoseNet in
this trainer. frame_1-only rendering (SegNet reads the last frame; frame_0 is
structurally seg-free).

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
import hashlib
import json
import os
import sys
import time
import zlib
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

SEG_H, SEG_W = 384, 512
DEFAULT_GT_CACHE = "/Users/adpena/Projects/pact/experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
POINTER_LINE = "0.1910828242 [contest-CPU] UNMOVED"

# Pre-registered A1 gate geometry (fd2 instrument geometry: block 447-450 + 32 rng(0)
# off-block samples). At --num-pairs below 600 the gate set is ALL training pairs.
GATE_BLOCK_PAIRS = (447, 448, 449, 450)
GATE_OFFBLOCK_SAMPLE = 32

# Pre-registered A1 alarm thresholds (tb1 charter T1: "never scale a loop whose
# realized-flip telemetry is flat"). Smooth descended but realized did not:
A1_SMOOTH_DROP_REL = 0.02      # smooth loss fell >= 2% since previous gate ...
A1_REALIZED_DROP_REL = 0.005   # ... while realized gate d_seg fell < 0.5%  -> ALARM
A1_CONSECUTIVE_REFUSE = 2      # this many consecutive alarms => stage-exit REFUSE

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
    grid_downsample: int          # D in {8, 16}: 384/D x 512/D token lattice
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


def make_render_fn():
    """render_fn for the canonical ``make_loss_fn`` hook:
    (model, coord_feats, code_idx, render_h, render_w) -> R(render) (1,384,512,3)."""
    from experiments.train_witness_realized_through_R_mlx import _apply_R

    def render_fn(model, coord_feats, code_idx, render_h, render_w):
        return _apply_R(model.render_frame(int(code_idx)))

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
    (max-observability non-negotiable); ``total_counted_bytes`` sums ONLY the three real
    streams (never the observability keys)."""
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


def realized_gate(model, gate_ids: tuple[int, ...], lstars, seg_cpu,
                  prev_realized: np.ndarray | None) -> dict[str, Any]:
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
        "realized_gate_dseg_mean": float(np.mean(dsegs)),
        "realized_gate_dseg_per_pair_max": float(np.max(dsegs)),
        "gate_render_stream": "mlx_cpu_fp32",
        "gate_wall_seconds": time.monotonic() - t0,
    }
    if prev_realized is not None and prev_realized.shape == realized.shape:
        row["realized_flips_vs_prev_gate"] = int(np.count_nonzero(realized != prev_realized))
    row["topology_per_class"] = topology_per_class(realized, gts)
    row["_realized_argmax"] = realized
    return row


BASIN_THRESHOLDS = {
    "smooth_rel_per_window": 0.01, "dseg_rel_per_window": 0.02,
    "lane_b0_delta_max": 2, "lane_erased_delta_max": 1,
}


def basin_entry_fires(w: list[dict]) -> bool:
    """TerminalSolve §16.1 validity predicate over the last-3-gate window (pure logic;
    unit-tested; consumed by main()'s basin-handoff block). Conditions: (a) quadratic
    crawl in BOTH smooth and realized channels; (b) lane topology stable; (c) shadow
    basis + tau stage throughout (no transitions remaining); zero A1 alarms in-window
    (linearization fidelity)."""
    t = BASIN_THRESHOLDS
    return (len(w) == 3
            and all(x["basis"] == "ema_shadow" for x in w)
            and all(x["stage"] == "seg_trunk_tau" for x in w)
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
# ``--telemetry-v9-port on`` (default off => byte-identical trained/checkpoint bytes;
# the flag is threaded via ``args`` ONLY, never TR1Config, and new rows go to the
# telemetry.jsonl via ``tlog`` ONLY — never ``telemetry_tail`` (which is baked into
# the checkpoint meta), so checkpoints are FLAG-INVARIANT).  The reusable Q1-Q7
# producers (``term_inert_rows``, ``lever_engage_row``, ``deterministic_strata``,
# ``ProducerResumeState``) live in ``tac.witness_control.telemetry_producers`` and the
# #404 positive-control canary in ``tac.witness_control.verdict_trend_alarm`` — this
# is a PORT (reuse the v9 producers), not a reimplementation.  READ-ONLY / score-neutral.
# ---------------------------------------------------------------------------
# The exact top-level addends of ``batch_loss`` (mean per-pair seg distortion +
# w_rate*rate surrogate + delta_sparsity_weight*group-L2).  "seg" is the distortion
# term (KD distill, when active, folds into it — it is added inside ``pair_loss``).
TR1_LOSS_TERM_KEYS: tuple[str, ...] = ("seg", "rate", "delta_sparsity")
# (#321) one post-weight addend dominating > FRAC of the loss for >= MIN_ROWS sustained
# rows => a "scored seg signal may be a passenger" alarm (mirrors v9 ``_TERMDOM_FRAC``).
TR1_TERMDOM_FRAC = 0.40
TR1_TERMDOM_MIN_ROWS = 3


def tr1_loss_terms_row(terms: dict[str, float], total: float, *, ep: int,
                       accum_batch: int, accepted_frac: float, weights_stepped: bool,
                       stage: str, seg_form: str) -> dict[str, Any]:
    """(#304) Canonical per-term ``loss_terms`` row for the TR1 top-level loss
    decomposition.  Stable complete key set (missing terms -> 0.0 so the schema is
    config-stable); ``sum_terms`` + ``sum_minus_total`` make the breakdown
    self-checking; ``accepted_frac`` + ``weights_stepped`` are the C6 LIVENESS stamps
    (#402 — a reader can tell a frozen epoch from a converging one).  Pure / MLX-free /
    unit-tested; score-neutral."""
    t = {k: float(terms.get(k, 0.0)) for k in TR1_LOSS_TERM_KEYS}
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
                               min_rows: int = TR1_TERMDOM_MIN_ROWS) -> list[dict[str, Any]]:
    """(#321) term_domination: a single post-weight addend > ``frac`` of the loss for
    ``min_rows`` sustained rows.  Mutates ``streaks`` (per-term run length) IN PLACE and
    returns the alarm rows that just CROSSED the sustained threshold (edge-triggered, so
    a persistent domination emits once per crossing, not every row).  Pure / unit-tested;
    mirrors the v9 trainer's inline term_domination streak logic (post-weight addends)."""
    tot_abs = abs(float(total)) + 1e-12
    rows: list[dict[str, Any]] = []
    for name in TR1_LOSS_TERM_KEYS:
        share = abs(float(terms.get(name, 0.0))) / tot_abs
        streak = int(streaks.get(name, 0))
        streak = streak + 1 if share > float(frac) else 0
        streaks[name] = streak
        if streak == int(min_rows):
            rows.append({
                "event": "confound_alarm", "kind": "term_domination", "term": str(name),
                "frac_of_loss": round(float(share), 4), "sustained_rows": int(streak),
                "note": "one post-weight loss term dominates >40% of the loss; the scored "
                        "seg signal may be a passenger (v9 #321 port)",
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
                    epoch: int, stage: str, cfg: TR1Config, telemetry_tail: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, np.ndarray] = {}
    for k, v in _tree_to_flat(model.trainable_parameters()).items():
        payload[f"param::{k}"] = v
    for k, v in ema.items():
        payload[f"ema::{k}"] = np.asarray(v)
    for k, v in opt_state_flat.items():
        payload[f"opt::{k}"] = v
    payload["meta::epoch"] = np.array([epoch], dtype=np.int64)
    meta = json.dumps({"stage": stage, "cfg": asdict(cfg), "config_hash": cfg.config_hash(),
                       "telemetry_tail": telemetry_tail[-4:]}).encode()
    payload["meta::json"] = np.frombuffer(meta, dtype=np.uint8)
    tmp = path.parent / (path.name + ".tmp.npz")  # endswith .npz => savez keeps the name
    np.savez(tmp, **payload)
    os.replace(str(tmp), str(path))  # atomic tmp+rename (P0 resumability)


def load_checkpoint(path: Path, model) -> dict[str, Any]:
    import mlx.core as mx
    from mlx.utils import tree_unflatten

    z = np.load(path, allow_pickle=False)
    params = [(k[len("param::"):], mx.array(z[k])) for k in z.files if k.startswith("param::")]
    model.update(tree_unflatten(params))
    ema = {k[len("ema::"):]: mx.array(z[k]) for k in z.files if k.startswith("ema::")}
    opt = {k[len("opt::"):]: z[k] for k in z.files if k.startswith("opt::")}
    meta = json.loads(bytes(z["meta::json"]).decode())
    return {"epoch": int(z["meta::epoch"][0]), "ema": ema, "opt_flat": opt, "meta": meta}


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
    ap.add_argument("--grid-downsample", type=int, default=16, choices=(8, 16))
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
    ap.add_argument("--token-temporal-mode", default="shared_base",
                    choices=("shared_base", "independent"),
                    help="shared_base = identity-xi advection (Einstein d_cov/d_gauge force)")
    ap.add_argument("--token-ste", default="round", choices=("round", "dither"),
                    help="RACED: uint8 rounding is directionally asymmetric through R")
    ap.add_argument("--w-seg", type=float, default=100.0)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--batch-pairs", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--gate-every", type=int, default=5)
    ap.add_argument("--ema-decay", type=float, default=None,
                    help="explicit override; default = DERIVED from run geometry (LawRef)")
    ap.add_argument("--gt-cache", type=Path, default=Path(DEFAULT_GT_CACHE))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--resume-from", type=Path, default=None)
    ap.add_argument("--max-wall-minutes", type=float, default=90.0)
    ap.add_argument("--full-confirm", action="store_true",
                    help="run the full num-pairs realized confirm at the final stage exit")
    ap.add_argument("--verdict-chunk", type=int, default=32,
                    help="pairs per CPU SegNet verdict chunk (<=120 per the charter)")
    ap.add_argument("--mlx-device", default="gpu", choices=("gpu", "cpu"))
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
                         "(98.806% image-stationarity has NO train-side force). Loss term => no param")
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
    ap.add_argument("--telemetry-v9-port", default="off", choices=("off", "on"),
                    help="v9 telemetry port: 'on' emits ADDITIVE read-only rows to "
                         "telemetry.jsonl — per-term loss_terms (#304), term_domination + "
                         "term_inert alarms (#321), a #404 positive-control sentinel, and "
                         "canonical lever_engage companions (Q7). DEFAULT 'off' => BYTE-IDENTICAL "
                         "trained/checkpoint bytes: the flag is threaded via args ONLY (never "
                         "TR1Config => config_hash + every checkpoint stay flag-invariant) and "
                         "new rows go to tlog/JSONL ONLY (never telemetry_tail, the checkpoint-"
                         "baked tail). READ-ONLY: no grad, no RNG advance (fixed dither bank + "
                         "isolated order_rng), no model/opt-state mutation. The flag exists ONLY "
                         "for the sealed r1c-lineage byte-identity guarantee (default-off-is-"
                         "orphan reconciliation: score-neutral telemetry is off here NOT by "
                         "orphaning but because a sealed live run demands trained-byte invariance)")
    return ap


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

    total_updates = args.epochs * max(1, args.num_pairs // max(1, args.batch_pairs))
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

    # GT: memmapped lstars/margins from the shared frozen-authority cache; frozen CPU SegNet.
    lstars = open_stored_npy_memmap(args.gt_cache, "lstars")
    margins = open_stored_npy_memmap(args.gt_cache, "margins")
    if lstars.shape[0] < cfg.num_pairs:
        raise SystemExit(f"gt cache has {lstars.shape[0]} pairs < --num-pairs {cfg.num_pairs}")
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

    # MLX scorer adapter (training-gradient device; NEVER a score) + canonical loss.
    upstream_root = str(Path(sys.modules["tac"].__file__).resolve().parents[2] / "upstream")
    adapter = load_mlx_distortion_scorer_adapter_from_upstream(upstream_root, device="cpu")
    # §3.2 boundary-annulus form fix: 100% of realized flips sit in the bottom GT-margin decile
    # (sg1 §1.3) => reweight the per-pixel seg loss toward the small-margin boundary annulus.
    loss_fn = make_loss_fn(adapter, SEG_H, SEG_W, score_domain=True,
                           seg_loss=cfg.seg_form_start,
                           margin_weighted=(cfg.margin_weighted_loss == "on"),
                           margin_weight_temp=cfg.margin_weight_temp,
                           render_fn=make_render_fn())

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
                        opt_state_flat={}, epoch=-1, stage="solve_init_pretrain",
                        cfg=cfg, telemetry_tail=[])
        del tgt, ds, tok
        # Scorer-loop Adam moments are created FRESH below (warm-start re-anchor
        # law #517/#518); the EMA shadow initializes from the post-projection params
        # (fresh warmup window => live-basis gates until W, same as the control).

    optimizer = optim.Adam(learning_rate=cfg.lr)

    ema: dict[str, Any] = {k: mx.array(v) for k, v in tree_flatten(model.trainable_parameters())}
    start_epoch = 0
    stage = "seg_trunk_ce" if cfg.seg_form_start == "ce" else f"seg_trunk_{cfg.seg_form_start}"
    if args.resume_from is not None:
        st = load_checkpoint(args.resume_from, model)
        ema = st["ema"]
        start_epoch = st["epoch"] + 1
        stage = st["meta"].get("stage", stage)
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
        tlog({"event": "resume", "resume_from": str(args.resume_from), "epoch": start_epoch,
              "stage": stage, "quant_engaged": bool(model._quant_engaged),
              "ema_backfilled_new_params": backfilled})
        # NOTE: Adam moments are re-anchored fresh (warm-start re-anchor law #517/#518):
        # a bounded-window resume restarts moment estimation at the resume geometry.

    # Gate set (pre-registered): all pairs when num_pairs < 600, else fd2 geometry.
    gate_ids = resolve_gate_ids(cfg.num_pairs)

    def pair_loss(mdl, idx: int, form: str):
        lstar = np.asarray(lstars[idx], dtype=np.int64)
        lstar_oh = mx.array((lstar[..., None] == np.arange(5)).astype(np.float32))[None]
        margin = mx.array(np.asarray(margins[idx], dtype=np.float32))
        pose_tgt = mx.zeros((6,))
        # sn1 ASYMMETRY lever: per-GT-class weight on Lane pixels (class index 1 —
        # canonical comma10k order, MEASURED; NEVER luma-sort re-derived).
        seg_pixel_w = None
        if cfg.class_weight_lane != 1.0:
            w_np = 1.0 + (cfg.class_weight_lane - 1.0) * (lstar == 1).astype(np.float32)
            seg_pixel_w = mx.array(w_np)[None]
        # QA75 teacher logits for THIS pair (precomputed b2b scorer response); None => OFF.
        distill_logits = None
        if distill_mm is not None:
            dl = np.asarray(distill_mm[idx], dtype=np.float32)         # (5,H,W)
            distill_logits = mx.array(np.transpose(dl, (1, 2, 0)))[None]  # (1,H,W,5)
        return loss_fn(mdl, None, idx, idx, lstar_oh, margin, pose_tgt,
                       cfg.w_seg, 0.0, 0.0, cfg.margin_target, seg_form=form,
                       seg_pixel_w=seg_pixel_w, compute_pose=False,
                       distill_logits=distill_logits, distill_weight=cfg.distill_weight,
                       distill_temp=cfg.distill_temp, distill_form=cfg.distill_form,
                       distill_attack_temp=cfg.distill_attack_temp)

    state_form = {"form": cfg.seg_form_start}

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
        if cfg.w_rate > 0.0:  # §3.4 (0.0 => byte-identical to the distortion-only control)
            acc = acc + cfg.w_rate * token_rate_term(mdl, ids)
        # ax1 §4a delta group-sparsity: only once ENGAGED (base-stability event / from_step_0) and
        # weight>0 => byte-identical to the control until engagement (gc10 F2 twin of the ν snap).
        if mdl._delta_sparsity_engaged and cfg.delta_sparsity_weight > 0.0:
            acc = acc + cfg.delta_sparsity_weight * delta_sparsity_term(mdl, ids)
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
    ema_warmup_updates = int(np.ceil(2.0 / max(1.0 - cfg.ema_decay, 1e-9)))
    global_step = 0 if args.resume_from is None else ema_warmup_updates  # resume => warm shadow
    ep_losses: list[float] = []
    telemetry_tail: list[dict] = []
    gnorm_hist: list[float] = []
    basin_window: list[dict] = []  # basin-entry detector state (basin_handoff == "on")
    gate_param_snapshot: dict[str, np.ndarray] | None = None
    order_rng = np.random.default_rng(cfg.seed + 1)
    knee_switched = stage != "seg_trunk_ce"
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
              "strata_ids": list(_tel_strata), "loss_term_keys": list(TR1_LOSS_TERM_KEYS),
              "termdom_frac": TR1_TERMDOM_FRAC, "termdom_min_rows": TR1_TERMDOM_MIN_ROWS,
              "note": "additive read-only rows; trained/checkpoint bytes flag-invariant "
                      "(flag via args not cfg; new rows via tlog not telemetry_tail)",
              "score_neutral": True})

    for epoch in range(start_epoch, cfg.epochs):
        if time.monotonic() > deadline:
            stop_reason = "max_wall_minutes"
            tlog({"event": "wall_clock_stop", "epoch": epoch})
            break
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
            for k, v in flat:
                ema[k] = cfg.ema_decay * ema[k] + (1.0 - cfg.ema_decay) * v
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
               "gnorm_last_batch": last_gnorm}
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
                                model=model, ema=ema, opt_state_flat={}, epoch=epoch,
                                stage=stage, cfg=cfg, telemetry_tail=telemetry_tail)
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
                            model=model, ema=ema, opt_state_flat={}, epoch=epoch,
                            stage=stage, cfg=cfg, telemetry_tail=telemetry_tail)
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
        telemetry_tail.append(row)

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
            gate_basis = "ema_shadow" if global_step >= ema_warmup_updates else "live_ema_warmup"
            live_np = {k: np.asarray(v) for k, v in tree_flatten(model.trainable_parameters())}
            live = ema_snapshot_swap(model, ema) if gate_basis == "ema_shadow" else None
            try:
                gate_row = realized_gate(model, gate_ids, lstars, seg_cpu, prev_realized)
                ledger = counted_bytes_ledger(model, cfg)
            finally:
                if live is not None:
                    ema_restore(model, live)
            realized_argmax = gate_row.pop("_realized_argmax")
            gate_row["gate_params"] = gate_basis
            gate_row["ema_warmup_updates"] = ema_warmup_updates
            gate_row["global_step"] = global_step
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
            tlog(gate_row)
            telemetry_tail.append(dict(gate_row.items()))
            print(json.dumps({k: gate_row[k] for k in
                              ("epoch", "realized_gate_dseg_mean", "a1_classification",
                               "total_counted_bytes")}), flush=True)
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
            prev_gate_row, prev_gate_smooth, prev_realized = gate_row, a1_smooth, realized_argmax
            save_checkpoint(out_dir / "checkpoints" / f"intra_{stage}_ep{epoch:05d}.npz",
                            model=model, ema=ema, opt_state_flat={}, epoch=epoch,
                            stage=stage, cfg=cfg, telemetry_tail=telemetry_tail)

            # ddm_tp1 (#804) v9 telemetry PORT emissions (READ-ONLY; gated => byte-identical
            # when off). Params are LIVE here (the EMA-shadow gate swap was restored in the
            # gate's finally). The per-term recompute runs the SAME deterministic forwards the
            # loss uses on a small fixed strata subset (no order_rng, no mx.random, no
            # model/opt mutation) and NEVER touches telemetry_tail => the checkpoint just
            # written is byte-identical to an off run.
            if _tel_v9:
                _tp_ids = [int(i) for i in _tel_strata]
                _tp_seg = float(mx.mean(mx.stack(
                    [pair_loss(model, i, state_form["form"]) for i in _tp_ids])))
                _tp_rate = (float(cfg.w_rate * token_rate_term(model, _tp_ids))
                            if cfg.w_rate > 0.0 else 0.0)
                _tp_ds = (float(cfg.delta_sparsity_weight
                                * delta_sparsity_term(model, _tp_ids))
                          if (model._delta_sparsity_engaged
                              and cfg.delta_sparsity_weight > 0.0) else 0.0)
                _tp_terms = {"seg": _tp_seg, "rate": _tp_rate, "delta_sparsity": _tp_ds}
                _tp_total = _tp_seg + _tp_rate + _tp_ds
                tlog(tr1_loss_terms_row(
                    _tp_terms, _tp_total, ep=epoch, accum_batch=steps,
                    accepted_frac=(1.0 if steps > 0 else 0.0),
                    weights_stepped=(steps > 0), stage=stage,
                    seg_form=state_form["form"]))
                for _dom in tr1_term_domination_alarms(_tp_terms, _tp_total,
                                                       _tel_termdom_streaks):
                    tlog(_dom)
                _tp_engaged = {
                    "seg": True,
                    "rate": bool(cfg.w_rate > 0.0),
                    "delta_sparsity": bool(model._delta_sparsity_engaged
                                           and cfg.delta_sparsity_weight > 0.0),
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
                    "epoch": epoch, "basis": gate_basis, "stage": stage,
                    "dseg": float(gate_row["realized_gate_dseg_mean"]),
                    "smooth": float(ep_loss), "alarm": bool(a1["a1_alarm"]),
                    "lane_b0": int(topo.get("betti0_realized", [0] * 5)[1]),
                    "lane_er": int(topo.get("gt_components_erased", [0] * 5)[1])})
                basin_window = basin_window[-3:]
                w = basin_window
                if basin_entry_fires(w):
                    save_checkpoint(out_dir / "checkpoints" / "stage_basin_entry.npz",
                                    model=model, ema=ema, opt_state_flat={}, epoch=epoch,
                                    stage="basin_entry", cfg=cfg,
                                    telemetry_tail=telemetry_tail)
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

    # Terminal stage checkpoint (distinct stage-encoded name; EMA shadow inside).
    save_checkpoint(out_dir / "checkpoints" / f"stage_{stage}_final.npz",
                    model=model, ema=ema, opt_state_flat={}, epoch=len(ep_losses) + start_epoch,
                    stage=stage, cfg=cfg, telemetry_tail=telemetry_tail)

    receipt: dict[str, Any] = {
        "schema": "ddm_tb1_tr1_window_receipt.v1",
        "pointer": POINTER_LINE, "score_claim": False, "promotion_eligible": False,
        "evidence_axis": "[macOS-CPU/MLX advisory]",
        "variant": cfg.variant, "config_hash": cfg.config_hash(), "cfg": asdict(cfg),
        "stop_reason": stop_reason, "epochs_ran": len(ep_losses),
        "final_ep_loss": ep_losses[-1] if ep_losses else None,
        "final_gate": {k: v for k, v in (prev_gate_row or {}).items() if not k.startswith("_")},
        "elapsed_seconds": time.monotonic() - started,
    }

    # Optional full realized confirm (chunked <=120; EMA shadow).
    if args.full_confirm and stop_reason in ("epochs_complete", "max_wall_minutes",
                                             "basin_entry_handoff"):
        from experiments.train_witness_realized_through_R_mlx import (
            _torch_R_to_camera_uint8,
            cpu_verdict_d_seg_batch,
        )

        confirm_basis = "ema_shadow" if global_step >= ema_warmup_updates else "live_ema_warmup"
        live = ema_snapshot_swap(model, ema) if confirm_basis == "ema_shadow" else None
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
            cbasis = "ema_shadow" if global_step >= ema_warmup_updates else "live_ema_warmup"
            live = ema_snapshot_swap(model, ema) if cbasis == "ema_shadow" else None
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

    rp = out_dir / "tr1_window_receipt.json"
    tmp = rp.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(str(tmp), str(rp))
    print(json.dumps({"receipt": str(rp), "stop_reason": stop_reason,
                      "score_claim": False}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
