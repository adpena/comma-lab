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
    d = min(max(d, 0.9), 0.9995)
    return d, prov


# ---------------------------------------------------------------------------
# Model (MLX). Trainable trees only; the LOTTO fixed bank is hidden from MLX's
# parameter traversal inside a plain object (regenerable from counted seed).
# ---------------------------------------------------------------------------
class _FixedBank:
    """Opaque (non-Module) holder so MLX does not treat fixed weights as trainable."""

    def __init__(self, tensors: dict[str, Any]):
        self.tensors = tensors


def _conv_shapes(cfg: TR1Config) -> list[tuple[str, tuple[int, int, int, int]]]:
    """Conv weight shapes (MLX layout: (C_out, kh, kw, C_in)). RF ~= 3 conv layers
    per SPEC S1.3 (conv0 + one conv per x2 upsample + head)."""
    w = cfg.renderer_width
    shapes: list[tuple[str, tuple[int, int, int, int]]] = [("conv0", (w, 3, 3, cfg.code_width))]
    for k in range(cfg.n_upsample):
        shapes.append((f"up{k}", (w, 3, 3, w)))
    shapes.append(("head", (3, 3, 3, w)))
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

        # -- description-level eval_roundtrip: token lattice with STE ------------
        def raw_tokens(self, idx: int):
            if cfg.token_temporal_mode == "shared_base":
                t = self.tokens_base + self.tokens_delta[idx]
            else:
                t = self.tokens[idx]
            # §3.1 coarse-from-birth: zero the inactive cells (fixed {0,1} mask, stop-grad).
            return t * mx.stop_gradient(self._cell_mask.tensors["keep"])

        def quantized_tokens(self, idx: int):
            t = mx.clip(self.raw_tokens(idx), -1.0, 1.0)  # (gh, gw, c)
            # §3.3(a) lattice anneal: before the knee (at_knee mode) the STE is disengaged =>
            # float tokens (find the basin in float, refine on the shipped lattice after).
            if not self._quant_engaged:
                return t
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
            return mx.sigmoid(x) * 255.0

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


def counted_bytes_ledger(model, cfg: TR1Config) -> dict[str, int]:
    """Per-stream COUNTED bytes for the current EMA/live params (rule-118 boundary):
    tokens + (plain: int8 weights | lotto: mask+modulations+biases) + selector."""
    # §3.1: the coarse-from-birth keep-mask (gh,gw bool) excludes inactive cells from coding.
    keep = np.asarray(model._cell_mask.tensors["keep"], dtype=np.float32)[..., 0] > 0.5
    keep_arg = keep if not keep.all() else None  # None => uniform grid (no selection needed)
    if cfg.token_temporal_mode == "shared_base":
        base_np = np.asarray(model.tokens_base, dtype=np.float32)[None]
        delta_np = np.asarray(model.tokens_delta, dtype=np.float32)
        base_code = (base_np[:, keep, :] if keep_arg is not None else base_np)
        # base coded once; the per-frame delta stream rides the same lattice.
        tok_b = (len(zlib.compress(quantize_tokens_np(base_code, cfg.token_quant_levels).tobytes(), 9))
                 + token_stream_bytes(delta_np, cfg.token_quant_levels, keep_mask=keep_arg))
    else:
        tok_b = token_stream_bytes(np.asarray(model.tokens, dtype=np.float32),
                                   cfg.token_quant_levels, keep_mask=keep_arg)
    ledger: dict[str, int] = {"tokens_bytes": int(tok_b)}
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
    ledger["total_counted_bytes"] = int(sum(ledger.values()))
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
        # §3.3(a) resume-past-knee: if the STE anneal already engaged (we resume in a post-CE
        # stage), re-engage it so the token forward matches the checkpointed lattice state.
        if cfg.token_quant_anneal == "at_knee" and stage != "seg_trunk_ce":
            model._quant_engaged = True
        tlog({"event": "resume", "resume_from": str(args.resume_from), "epoch": start_epoch,
              "stage": stage, "quant_engaged": bool(model._quant_engaged)})
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
        return loss_fn(mdl, None, idx, idx, lstar_oh, margin, pose_tgt,
                       cfg.w_seg, 0.0, 0.0, cfg.margin_target, seg_form=form,
                       seg_pixel_w=seg_pixel_w, compute_pose=False)

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

    def batch_loss(mdl, ids: list[int]):
        acc = None
        for i in ids:
            li = pair_loss(mdl, int(i), state_form["form"])
            acc = li if acc is None else acc + li
        acc = acc / len(ids)
        if cfg.w_rate > 0.0:  # §3.4 (0.0 => byte-identical to the distortion-only control)
            acc = acc + cfg.w_rate * token_rate_term(mdl, ids)
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
    stop_reason = "epochs_complete"

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
        tlog(row)
        telemetry_tail.append(row)

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
            a1 = a1_adjudicate(prev_gate_row, gate_row, prev_gate_smooth, ep_loss)
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
            prev_gate_row, prev_gate_smooth, prev_realized = gate_row, ep_loss, realized_argmax
            save_checkpoint(out_dir / "checkpoints" / f"intra_{stage}_ep{epoch:05d}.npz",
                            model=model, ema=ema, opt_state_flat={}, epoch=epoch,
                            stage=stage, cfg=cfg, telemetry_tail=telemetry_tail)

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
    # enables the bounded pose solve (composed_s_gate_subset>0). Fails GRACEFULLY (advisory).
    # The solver is the PROVEN warp-constrained FD-Jacobian LM-GN (eg1/tt1 approach; MAIN recall
    # steer 2026-07-30) — the razor-sharp realized pose landscape needs damped GN, not Adam. The
    # per-pair residual_mse_traj + relins_run in the verdict SELF-DOCUMENT convergence; firing is
    # gated on a convergence smoke (bc1), so no untrustworthy verdict reaches an endpoint decision.
    if cfg.composed_s_gate_subset > 0:
        try:
            from experiments.ddm_composed_s_verdict import ComposedSVerdict
            from experiments.train_witness_realized_through_R_mlx import (
                _torch_R_to_camera_uint8,
                cpu_verdict_d_seg_batch,
            )

            n_sub = min(int(cfg.composed_s_gate_subset), cfg.num_pairs)
            if cfg.composed_s_subset_ids is not None:  # MAIN QA66 pose-mass tail
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
            if verdict.available:
                receipt["composed_s_verdict"] = verdict.composed_s(
                    sub_ids, cams, dseg_sub, total_bytes)
            else:
                receipt["composed_s_verdict"] = {"skipped": verdict.reason,
                                                 "score_claim": False}
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
