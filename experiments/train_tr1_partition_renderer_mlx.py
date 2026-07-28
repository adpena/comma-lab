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
                return self.tokens_base + self.tokens_delta[idx]
            return self.tokens[idx]

        def quantized_tokens(self, idx: int):
            t = mx.clip(self.raw_tokens(idx), -1.0, 1.0)  # (gh, gw, c)
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


# ---------------------------------------------------------------------------
# COUNTED-byte ledger (measured with a real compressor on the real quantized
# payloads; labeled COUNTED-ESTIMATE until the E4/WS1 exporter grammar wires in).
# ---------------------------------------------------------------------------
def quantize_tokens_np(tokens: np.ndarray, levels: int) -> np.ndarray:
    t = np.clip(tokens, -1.0, 1.0)
    return np.round((t + 1.0) * 0.5 * (levels - 1)).astype(np.uint8)


def token_stream_bytes(tokens_np: np.ndarray, levels: int) -> int:
    """Temporal-delta (mod 256) + zlib-9 on the quantized token lattice (P,gh,gw,c)."""
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
    if cfg.token_temporal_mode == "shared_base":
        base_np = np.asarray(model.tokens_base, dtype=np.float32)[None]
        delta_np = np.asarray(model.tokens_delta, dtype=np.float32)
        # base coded once; the per-frame delta stream rides the same lattice.
        tok_b = (len(zlib.compress(quantize_tokens_np(base_np, cfg.token_quant_levels).tobytes(), 9))
                 + token_stream_bytes(delta_np, cfg.token_quant_levels))
    else:
        tok_b = token_stream_bytes(np.asarray(model.tokens, dtype=np.float32),
                                   cfg.token_quant_levels)
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
    row["_realized_argmax"] = realized
    return row


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
    loss_fn = make_loss_fn(adapter, SEG_H, SEG_W, score_domain=True,
                           seg_loss=cfg.seg_form_start, margin_weighted=False,
                           render_fn=make_render_fn())

    model = build_module(cfg)
    mx.eval(model.parameters())
    optimizer = optim.Adam(learning_rate=cfg.lr)

    ema: dict[str, Any] = {k: mx.array(v) for k, v in tree_flatten(model.trainable_parameters())}
    start_epoch = 0
    stage = "seg_trunk_ce" if cfg.seg_form_start == "ce" else f"seg_trunk_{cfg.seg_form_start}"
    if args.resume_from is not None:
        st = load_checkpoint(args.resume_from, model)
        ema = st["ema"]
        start_epoch = st["epoch"] + 1
        stage = st["meta"].get("stage", stage)
        tlog({"event": "resume", "resume_from": str(args.resume_from), "epoch": start_epoch,
              "stage": stage})
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

    def batch_loss(mdl, ids: list[int]):
        acc = None
        for i in ids:
            li = pair_loss(mdl, int(i), state_form["form"])
            acc = li if acc is None else acc + li
        return acc / len(ids)

    vg = nn.value_and_grad(model, batch_loss)

    prev_gate_row: dict[str, Any] | None = None
    prev_gate_smooth: float | None = None
    prev_realized: np.ndarray | None = None
    a1_consecutive = 0
    ep_losses: list[float] = []
    telemetry_tail: list[dict] = []
    gnorm_hist: list[float] = []
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
        tlog(row)
        telemetry_tail.append(row)

        # A1 realized gate (EMA shadow — inference artifacts come from the shadow).
        if (epoch + 1) % cfg.gate_every == 0 or epoch == cfg.epochs - 1:
            live = ema_snapshot_swap(model, ema)
            try:
                gate_row = realized_gate(model, gate_ids, lstars, seg_cpu, prev_realized)
                ledger = counted_bytes_ledger(model, cfg)
            finally:
                ema_restore(model, live)
            realized_argmax = gate_row.pop("_realized_argmax")
            # #685 px1 race-fairness telemetry: MEASURED update magnitude per arm —
            # RMS of the live-param delta accumulated since the previous gate.
            live_np = {k: np.asarray(v) for k, v in live.items()}
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
    if args.full_confirm and stop_reason in ("epochs_complete", "max_wall_minutes"):
        from experiments.train_witness_realized_through_R_mlx import (
            _torch_R_to_camera_uint8,
            cpu_verdict_d_seg_batch,
        )

        live = ema_snapshot_swap(model, ema)
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
            }
        finally:
            ema_restore(model, live)

    rp = out_dir / "tr1_window_receipt.json"
    tmp = rp.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    os.replace(str(tmp), str(rp))
    print(json.dumps({"receipt": str(rp), "stop_reason": stop_reason,
                      "score_claim": False}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
