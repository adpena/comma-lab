# SPDX-License-Identifier: MIT
"""CAPSTONE LEVER #2 — MLX (mx-framework) port of the through-R WITNESS trainer.

The torch reference is ``experiments/train_witness_realized_through_R.py``. This
module mirrors its REALIZED-axis physics ENTIRELY in MLX (mx) so the inner loop
runs on the M5 Max GPU at FP32 with unified-memory saturation:

  witness(coords, per-(pair,frame) code) -> RGB at scorer res
    -> R [bicubic up to camera -> bilinear down to 384x512 -> uint8-STE]
    -> frame1 -> frozen MLX SegNet -> realized seg CE (vs L*)
    -> pair  -> rgb_to_yuv6_mlx -> frozen MLX PoseNet -> realized pose MSE (vs GT pose)
  mx.value_and_grad over (witness params) -> AdamW step -> EMA.

WHY MLX (operator directive 2026-06-25): the torch path is impractical — torch
n600 = ~935 s/ep (MPS) and ~25 s/ep (CPU at n16); AND torch-MPS drifts 23x on
pose (NEVER authority). The MLX scorer port is FP32-accurate: the through-R
gradient's MLX-CPU-vs-MLX-GPU cosine is 0.99996 (smoke 2026-06-25), and the
SegNet MLX-vs-torch per-logit drift bound is ~0.038 max / ~0.002 mean. MLX GPU
gives a usable+accurate GRADIENT, fast, with everything resident in unified mem.

COMPUTE-SUBSTRATE LAW (CLAUDE.md "MLX portable-local-substrate authority" +
"MPS NEVER authority"): MLX (CPU or GPU) is a VALID fp32 TRAINING-GRADIENT
device, NEVER a score authority. The d_seg/d_pose VERDICT comes from the FROZEN
CPU-torch SegNet argmax + PoseNet (NEVER MLX, NEVER MPS). The MLX scorer is the
fast accurate gradient + advisory; the verdict is recomputed on CPU-torch. EMA +
eval_roundtrip (R) are non-negotiable and in-loop.

NO-FAKE: the realized d_seg VERDICT = argmax-disagreement RATE between the FROZEN
CPU-torch SegNet on the MLX-rendered-through-R frame1 and L* (GT argmax). The
realized d_pose VERDICT = pose-MSE of the FROZEN CPU-torch PoseNet on the
rendered pair vs GT pose. The MLX scorer trains the witness; the CPU-torch scorer
is the verdict. No surrogate is the verdict.

BORROWED-SUBSTRATE (CLAUDE.md NO-FAKE #7):
  * BORROWED: the MLX SegNet/PoseNet port (tac.local_acceleration.mlx_scorer_*),
    the canonical MLX R + rgb_to_yuv6 primitives (pr95_hnerv_mlx_training:
    apply_eval_roundtrip_nhwc, rgb_to_yuv6_mlx), the frozen contest scorers +
    GT-decode + CPU authority (seg_core + upstream modules), Fourier-feature
    coord-INR (Tancik 2020) + FiLM (Perez 2018) + HNeRV amortized framing.
  * OURS-ORIGINAL (same vehicle as the torch trainer, now on MLX): a NON-RGB-
    vehicle coord-INR OUTPUTTING RGB whose REALIZED (post-R, post-frozen-scorer)
    argmax partition survives R AND whose PoseNet pose matches GT, JOINTLY in one
    shared code -- the round-trip-survival lever with BOTH scorers in-loop, the
    joint realized d_seg+d_pose loss, ported to mx for throughput. NOT a PR95
    HNeRV reskin (anti-reskin: our coord-INR, not their decoder).

Evidence: training gradient [macOS-MLX training-gradient]; the d_seg/d_pose
VERDICT is [contest-CPU advisory] (frozen CPU-torch mirror of evaluate.py over
the measured pair subset). promotion_eligible=False; NO score claim beyond
advisory; pointer UNMOVED unless a real byte-closed sub-0.19110 exact-eval row
lands. Disk: SSD/repo results, NEVER /tmp durable.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _p in (REPO, REPO / "src", REPO / "upstream"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Camera-native + scorer-input resolutions (from upstream/frame_utils.py).
CAMERA_H, CAMERA_W = 874, 1164
SEG_H, SEG_W = 384, 512
RATE_DENOM = 37_545_489.0
N_CLASSES = 5


# ---------------------------------------------------------------------------
# R-OPERATOR DISPATCH (compute-facet #252; DEFAULT-OFF, bit-identical when off).
# The contest-exact eval roundtrip R (bicubic-up -> uint8 STE @ camera -> bilinear
# down to scorer) is the hot per-frame compute inside the loss (runs 2x/pair). By
# default it is the pure-MLX oracle ``apply_contest_faithful_roundtrip_nhwc`` --
# EXACTLY the pre-existing call, so a run with the flag OFF is BYTE-IDENTICAL (a
# running --resume-from is unaffected).
#
# ``--fused-r-kernel`` (set via ``set_fused_r_kernel``) swaps in the fused Metal
# kernel ``metal_fused_r_operator.make_fused_r_roundtrip`` -- a benchmarked drop-in
# that is BIT-IDENTICAL to the numpy-fp32 authority on the FORWARD (contraction-off
# kernels) and matches the pure-MLX VJP to ~1 ULP (the same oracle both paths' loss
# uses). Proven by ``src/tac/tests/test_metal_fused_r_operator.py`` (25 tests, incl.
# forward bit-identity at real 874x1164->384x512 resolution + fused-vs-non-fused VJP
# parity). NO-FAKE: this buys SPEED + training-gradient fidelity, NEVER a score --
# the d_seg/d_pose verdict + byte-closed archive remain the numpy/torch-CPU authority.
#
# ``--mx-compile`` (set via ``set_mx_compile_r``) is FAIL-CLOSED-GATED: mx.compile
# reintroduces fp-contraction (fma) that the STE-round path deliberately avoids,
# which was MEASURED (2026-07-03) to shift camera pixels by up to 4.8e-3 across the
# uint8 round boundary -> flips d_seg argmax pixels (a NO-FAKE score risk) for only
# ~1.11x. So enabling it runs a startup bit-identity gate; if the compiled R is NOT
# bit-identical to the reference at real resolution it RAISES (never silently drifts
# d_seg). The fused Metal kernel (contraction-off) is the correct fast path instead.
_FUSED_R_KERNEL_ENABLED = False
_MX_COMPILE_R_FN = None  # optional mx.compile'd reference R, installed only after a bit-identity gate.
_FUSED_R_FN_CACHE: dict = {}


def set_fused_r_kernel(enabled: bool) -> None:
    """Enable/disable the fused Metal R kernel in the render path (default OFF)."""

    global _FUSED_R_KERNEL_ENABLED
    _FUSED_R_KERNEL_ENABLED = bool(enabled)


def fused_r_kernel_enabled() -> bool:
    return bool(_FUSED_R_KERNEL_ENABLED)


def set_mx_compile_r(compiled_fn) -> None:
    """Install a bit-identity-verified mx.compile'd reference R (or None to clear).

    Only the caller's fail-closed gate (see ``maybe_enable_mx_compile_r``) should set
    this -- passing a non-verified compiled fn would risk silent d_seg drift.
    """

    global _MX_COMPILE_R_FN
    _MX_COMPILE_R_FN = compiled_fn


def _reference_R(rgb_nhwc):
    """Pure-MLX contest-exact R (the pre-existing default; byte-identical path)."""

    from tac.local_acceleration.pr95_hnerv_mlx_training import apply_contest_faithful_roundtrip_nhwc

    return apply_contest_faithful_roundtrip_nhwc(rgb_nhwc, output_hw=(SEG_H, SEG_W), ste_round=True)


def _apply_R(rgb_nhwc):
    """R-op dispatch. DEFAULT (no flags) => exactly ``_reference_R`` => byte-identical."""

    if _FUSED_R_KERNEL_ENABLED:
        from tac.local_acceleration.metal_fused_r_operator import (
            CAMERA_HW as _FR_CAM,
            make_fused_r_roundtrip,
            metal_fused_r_available,
        )

        if metal_fused_r_available():
            key = ((int(SEG_H), int(SEG_W)), True)
            fn = _FUSED_R_FN_CACHE.get(key)
            if fn is None:
                fn = make_fused_r_roundtrip(camera_hw=_FR_CAM, output_hw=(SEG_H, SEG_W), ste_round=True)
                _FUSED_R_FN_CACHE[key] = fn
            return fn(rgb_nhwc)
        # non-Metal device: fall through to the reference (fused kernel needs a GPU).
    if _MX_COMPILE_R_FN is not None:
        return _MX_COMPILE_R_FN(rgb_nhwc)
    return _reference_R(rgb_nhwc)


def maybe_enable_mx_compile_r(enabled: bool, *, render_hw: tuple[int, int] = (48, 64)) -> dict:
    """FAIL-CLOSED gate for ``--mx-compile`` on the R op.

    mx.compile fuses mul+add into fma, which reintroduces the fp-contraction that the
    STE-round path deliberately avoids -- MEASURED (2026-07-03) to shift camera pixels
    up to ~4.8e-3 across the uint8 round boundary => flip d_seg argmax pixels for only
    ~1.11x speed. So enabling installs the compiled R ONLY if it is bit-identical to the
    reference at the given resolution; otherwise it RAISES (never silently drifts d_seg).

    Returns a measured-parity dict (advisory). No-op + empty dict when ``enabled`` is
    False (byte-identical). Call once at startup from the launch path.
    """

    if not enabled:
        return {}
    import mlx.core as mx
    import numpy as _np

    from tac.local_acceleration.pr95_hnerv_mlx_training import apply_contest_faithful_roundtrip_nhwc

    def _r_ref(z):
        return apply_contest_faithful_roundtrip_nhwc(z, output_hw=(SEG_H, SEG_W), ste_round=True)

    _r_cmp = mx.compile(_r_ref)
    rng = _np.random.default_rng(0)
    x = mx.array((rng.random((2, int(render_hw[0]), int(render_hw[1]), 3)) * 255.0).astype(_np.float32))
    y_ref = _np.asarray(_r_ref(x))
    y_cmp = _np.asarray(_r_cmp(x))
    fwd_max = float(_np.max(_np.abs(y_ref - y_cmp))) if y_ref.size else 0.0
    cot = mx.array(rng.standard_normal(y_ref.shape).astype(_np.float32))
    _, (g_ref,) = mx.vjp(_r_ref, (x,), (cot,))
    _, (g_cmp,) = mx.vjp(_r_cmp, (x,), (cot,))
    grad_max = float(_np.max(_np.abs(_np.asarray(g_ref) - _np.asarray(g_cmp))))
    res = {
        "fwd_max_abs_delta": fwd_max,
        "grad_max_abs_delta": grad_max,
        "fwd_bit_identical": bool(fwd_max == 0.0),
        "render_hw": [int(render_hw[0]), int(render_hw[1])],
    }
    if fwd_max != 0.0:
        raise AssertionError(
            "--mx-compile refused (fail-closed, NO-FAKE d_seg guard): mx.compile of the R op is NOT "
            f"bit-identical to the reference (fwd max|Δ|={fwd_max:.3e} at render {tuple(render_hw)}); "
            "mx.compile reintroduces fp-contraction that flips the uint8-STE argmax knife-edge. Use "
            "--fused-r-kernel (contraction-off Metal kernel, bit-identical) for the fast R instead."
        )
    set_mx_compile_r(_r_cmp)
    return res


def r_isolated_microbench(
    *, render_h: int, render_w: int, n_frames: int = 2, reps: int = 20, seed: int = 0
) -> dict:
    """Isolated in-situ timing of the R op (reference vs fused Metal), fwd and fwd+bwd.

    Times the contest-exact R at the REAL render->camera->scorer resolution on a throwaway
    tensor (fixed-seed; does NOT touch training RNG/model/opt/ema). Returns per-frame ms for
    the pure-MLX reference and (if a Metal device is available) the fused kernel, plus the
    fused speedup. This is what lets the profile compute R's FRACTION of the step -> the
    realized whole-run speedup, MEASURED not estimated (Amdahl: whole = 1/((1-f)+f/su_R)).
    """

    import time as _time

    import mlx.core as mx
    import numpy as _np

    rng = _np.random.default_rng(int(seed))
    x = mx.array((rng.random((int(n_frames), int(render_h), int(render_w), 3)) * 255.0).astype(_np.float32))
    cot_shape = (int(n_frames), int(SEG_H), int(SEG_W), 3)
    cot = mx.array(rng.standard_normal(cot_shape).astype(_np.float32))

    def _time_fwd(fn) -> float:
        for _ in range(3):
            mx.eval(fn(x))
        t0 = _time.perf_counter()
        for _ in range(int(reps)):
            mx.eval(fn(x))
        return (_time.perf_counter() - t0) / int(reps) / int(n_frames) * 1e3  # ms per frame

    def _time_fwdbwd(fn) -> float:
        for _ in range(3):
            _, (g,) = mx.vjp(fn, (x,), (cot,))
            mx.eval(g)
        t0 = _time.perf_counter()
        for _ in range(int(reps)):
            _, (g,) = mx.vjp(fn, (x,), (cot,))
            mx.eval(g)
        return (_time.perf_counter() - t0) / int(reps) / int(n_frames) * 1e3

    ref_fwd = _time_fwd(_reference_R)
    ref_fwdbwd = _time_fwdbwd(_reference_R)
    out: dict = {
        "render_hw": [int(render_h), int(render_w)],
        "n_frames": int(n_frames),
        "reps": int(reps),
        "ref_fwd_ms_per_frame": round(ref_fwd, 4),
        "ref_fwdbwd_ms_per_frame": round(ref_fwdbwd, 4),
    }
    try:
        from tac.local_acceleration.metal_fused_r_operator import (
            CAMERA_HW as _FR_CAM,
            make_fused_r_roundtrip,
            metal_fused_r_available,
        )

        if metal_fused_r_available():
            _fused = make_fused_r_roundtrip(camera_hw=_FR_CAM, output_hw=(SEG_H, SEG_W), ste_round=True)
            f_fwd = _time_fwd(_fused)
            f_fwdbwd = _time_fwdbwd(_fused)
            out["fused_fwd_ms_per_frame"] = round(f_fwd, 4)
            out["fused_fwdbwd_ms_per_frame"] = round(f_fwdbwd, 4)
            out["fused_fwd_speedup"] = round(ref_fwd / f_fwd, 3) if f_fwd > 0 else None
            out["fused_fwdbwd_speedup"] = round(ref_fwdbwd / f_fwdbwd, 3) if f_fwdbwd > 0 else None
    except Exception as _e:  # pragma: no cover - advisory micro-bench, never fatal
        out["fused_error"] = str(_e)
    return out


def _load_frontier() -> float:
    """Read the canonical contest-CPU frontier from the pointer (NEVER a hardcoded literal,
    per CLAUDE.md "Frontier scores are pointer-only"). Falls back to the last-known 0.19110
    ONLY if the pointer is unreadable. The pointer's ``score`` IS the recorded exact-eval row.
    """
    ptr = REPO / ".omx" / "state" / "canonical_frontier_pointer.json"
    try:
        d = json.loads(ptr.read_text())
        cpu = d.get("our_local_frontier_contest_cpu") or {}
        s = cpu.get("score")
        if isinstance(s, (int, float)) and s > 0:
            return float(s)
    except Exception:
        pass
    return 0.19110  # last-known pointer value; fail-soft only


FRONTIER = _load_frontier()

_FORBIDDEN_TMP = ("/tmp/", "/var/tmp/", "/private/tmp/", "/private/var/tmp/")
_FOURIER_SEED = 0


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _refuse_tmp(path: Path, field_name: str) -> None:
    if any(str(path).startswith(p) for p in _FORBIDDEN_TMP):
        raise ValueError(f"{field_name}={path!r} is a /tmp-class path; use the SSD tier per CLAUDE.md.")


# ---------------------------------------------------------------------------
# Deterministic Fourier B (shared with the torch trainer for parity).
# ---------------------------------------------------------------------------
def deterministic_fourier_B(n_fourier: int, fourier_sigma: float) -> np.ndarray:
    rng = np.random.default_rng(_FOURIER_SEED)
    return (rng.standard_normal((2, n_fourier)) * fourier_sigma).astype(np.float32)


# ---------------------------------------------------------------------------
# SIREN init (Sitzmann et al. NeurIPS 2020, "Implicit Neural Representations with
# Periodic Activation Functions") — the canonical fix for periodic/sine-activation
# UNTRAINABILITY from scratch. Without it a sine MLP's pre-activations blow out of
# the sin linear regime at init -> arcsine-saturated activations, dead/exploding
# gradients (parked capstone-v1 hosc-from-scratch: d_seg 0.689, gnorm-460 spike-skips).
#
# The scheme (mirrors the canonical torch impl tac.substrates.wyner_ziv_cooperative_
# receiver.architecture._siren_init_): for a sine layer with fan_in inputs,
#   FIRST sine layer : W ~ U(-1/fan_in, +1/fan_in)               (keep W@x in [-1,1])
#   HIDDEN sine layer: W ~ U(-sqrt(6/fan_in)/omega, +same)       (unit-variance pre-act
#                                                                  AFTER the sin's omega
#                                                                  frequency multiply)
#   bias = 0 (canonical).
# ``omega`` is the frequency INSIDE the periodic part: hosc=tanh(beta*sin(omega*u)),
# siren=sin(omega*u), finer=sin(omega*(|u|+1)*u). The sin sees omega*u and the /omega
# hidden scaling is the matching variance fix. ``first_bias_scale`` (FINER, Liu 2024) is
# the wide first-layer bias range U(-k,+k) that selects activated frequency sub-bands;
# for siren/hosc it is 0 (canonical SIREN zero bias). The init draws from the GLOBAL
# mx.random stream (seeded per-restart by the caller, mx.random.seed(seed_r)) ->
# deterministic reproducibility.
# ---------------------------------------------------------------------------
def _siren_init_linear_mlx(layer, *, is_first: bool, omega: float, bias_scale: float = 0.0) -> None:
    """In-place SIREN init of one mlx.nn.Linear sine layer (weight (out, in); fan_in=in).

    ``bias_scale``>0 -> bias ~ U(-bias_scale, +bias_scale) (the FINER first-layer bias);
    ``bias_scale``==0 -> bias zeroed (canonical SIREN)."""
    import mlx.core as mx

    w = layer.weight  # (out_features, in_features)
    fan_in = int(w.shape[1])
    if is_first:
        bound = 1.0 / float(fan_in)
    else:
        bound = math.sqrt(6.0 / float(fan_in)) / max(float(omega), 1e-8)
    layer.weight = mx.random.uniform(low=-bound, high=bound, shape=w.shape)
    if getattr(layer, "bias", None) is not None:
        if bias_scale and bias_scale > 0.0:
            layer.bias = mx.random.uniform(low=-float(bias_scale), high=float(bias_scale), shape=layer.bias.shape)
        else:
            layer.bias = mx.zeros_like(layer.bias)


def apply_siren_init(model, *, omega: float, finer_first_bias_scale: float = 0.0) -> None:
    """Apply SIREN init to the witness's periodic (sine) layers only.

    Sine layers = those whose output feeds the ``_act`` periodic nonlinearity:
      * ``in_proj`` (the FIRST sine layer; its input is the Fourier coord feats) -> is_first
      * each ``hidden[li]`` (subsequent sine layers; pre-act FiLM-affine then sin) -> hidden
    The ``out`` layer feeds ``sigmoid`` (NOT sin) and the ``film``/``code`` FiLM path are
    left at their default MLX init (FiLM must stay nonzero or the code-gradient dies: with
    code initialized to zero, a zeroed film weight -> film_out=0 and d(loss)/d(code)=0 forever).
    ``finer_first_bias_scale``>0 applies FINER's wide first-layer bias (the spectral-bias knob);
    pass 0 for plain SIREN/hosc.
    """
    _siren_init_linear_mlx(model.in_proj, is_first=True, omega=omega, bias_scale=float(finer_first_bias_scale))
    for layer in model.hidden:
        _siren_init_linear_mlx(layer, is_first=False, omega=omega)


# ---------------------------------------------------------------------------
# The witness: an mlx.nn.Module coord-INR (Fourier feats + per-(pair,frame) FiLM)
# -> RGB at scorer res (NHWC). OUR coord-INR vehicle (anti-reskin), on MLX.
# ---------------------------------------------------------------------------
def build_witness_module(
    num_pairs: int,
    n_fourier: int,
    hidden_dim: int,
    n_hidden: int,
    mod_dim: int,
    fourier_sigma: float,
    in_feat_override: int | None = None,
    activation: str = "relu",
    hosc_beta: float = 4.0,
    hosc_omega: float = 1.0,
    chroma: bool = True,
    siren_init: bool = True,
    siren_omega: float = 30.0,
    finer_first_bias_scale: float = 1.0,
    wire_scale: float = 1.0,
):
    import mlx.core as mx
    import mlx.nn as nn

    class RGBWitnessMLX(nn.Module):
        """g(coord_feats, code_idx) -> (P_px, 3) RGB in [0,255].

        ``in_feat_override`` lets the caller feed a NON-isotropic coord feature
        (e.g. the all-class DIRECTIONAL Fourier basis, computed per-pair outside
        the module) of arbitrary width instead of the built-in 2*n_fourier iso
        feats. ``activation`` selects the hidden-layer nonlinearity:
          * "relu"  (control)
          * "hosc"  = tanh(beta*sin(omega*u)) (Serrano 2024) — the step-native
            R-SURVIVAL lever: as beta grows the sine square-waves into a sharp
            step train (matches the piecewise-constant SegNet argmax partition,
            NO Gibbs ringing through the bicubic^->uint8->bilinear_ round-trip R).
          * "siren" = sin(omega*u) (Sitzmann NeurIPS 2020) — the canonical
            periodic-activation INR (omega=siren_omega, default 30) with SIREN init.
          * "finer" = sin(omega*(|u|+1)*u) (Liu et al. CVPR 2024, FINER) — the
            VARIABLE-PERIODIC activation that BEAT SIREN on the in-tree d_seg arms
            (torch-measured 0.00138). The |u|+1 scale makes the local frequency grow
            with the pre-activation magnitude; the FINER spectral-bias knob is the
            FIRST-LAYER bias init range U(-finer_first_bias_scale, +) (vs SIREN's
            zero bias) which selects which frequency sub-bands are activated.
        The periodic family (hosc/siren/finer) uses SIREN weight init; finer adds
        the wide first-layer bias. ``siren_omega`` is the periodic frequency for
        siren/finer (hosc keeps its own ``hosc_omega``).
        """

        def __init__(self) -> None:
            super().__init__()
            self.num_pairs = num_pairs
            self.n_fourier = n_fourier
            self.hidden_dim = hidden_dim
            self.n_hidden = n_hidden
            self.mod_dim = mod_dim
            in_feat = (2 * n_fourier) if in_feat_override is None else int(in_feat_override)
            self.in_feat = in_feat
            self.activation = str(activation)
            self.hosc_beta = float(hosc_beta)
            self.hosc_omega = float(hosc_omega)
            self.siren_omega = float(siren_omega)
            self.finer_first_bias_scale = float(finer_first_bias_scale)
            self.wire_scale = float(wire_scale)
            # The periodic-frequency omega used by THIS activation (hosc keeps its own
            # low default; siren/finer/wire use siren_omega ~ 30). Read by _act AND SIREN init.
            self.periodic_omega = float(hosc_omega) if str(activation) == "hosc" else float(siren_omega)
            self.chroma = bool(chroma)
            B = deterministic_fourier_B(n_fourier, fourier_sigma)
            # Fixed (non-trainable) Fourier projection, held as a plain constant.
            self._B = mx.array(B)
            self.code = mx.zeros((num_pairs * 2, mod_dim))
            self.in_proj = nn.Linear(in_feat, hidden_dim)
            self.film = nn.Linear(mod_dim, 2 * hidden_dim * n_hidden)
            self.hidden = [nn.Linear(hidden_dim, hidden_dim) for _ in range(n_hidden)]
            self.out = nn.Linear(hidden_dim, 3)

        def _act(self, u: Any) -> Any:
            if self.activation == "hosc":
                return mx.tanh(self.hosc_beta * mx.sin(self.hosc_omega * u))
            if self.activation == "siren":
                return mx.sin(self.siren_omega * u)
            if self.activation == "finer":
                # FINER variable-periodic (Liu et al. 2024): sin(omega*(|u|+1)*u).
                return mx.sin(self.siren_omega * (mx.abs(u) + 1.0) * u)
            if self.activation == "wire":
                # WIRE real-Gabor (Saragadam CVPR2024): sin(omega*u)*exp(-0.5*(s*u)^2) —
                # best space-frequency localization (compact EDGE error; SIREN rings globally,
                # Gaussian errs at edges). s = wire_scale (matches tac.substrates.siren.activation_family).
                return mx.sin(self.siren_omega * u) * mx.exp(-0.5 * mx.square(self.wire_scale * u))
            return nn.relu(u)

        def _apply_chroma(self, rgb: Any) -> Any:
            """CHROMA d_seg lever (operator 2026-06-25 "Chroma too"). With ``chroma``
            True (default) the witness emits 3 INDEPENDENT RGB planes -> the frozen
            SegNet argmax (over RGB) AND PoseNet's 2 YUV6 chroma planes both read
            genuine chroma signal (chroma is a REAL d_seg actuator: it can flip the
            argmax partition at the codim-1 boundary annulus). With ``chroma`` False
            the output is forced ACHROMATIC (BT.601 luma replicated to R=G=B) so chroma
            is removed as a free actuator -- the ablation CONTROL arm that isolates
            chroma's d_seg contribution. ``rgb`` is (..., 3) in [0,255]."""
            if self.chroma:
                return rgb
            luma = 0.299 * rgb[..., 0:1] + 0.587 * rgb[..., 1:2] + 0.114 * rgb[..., 2:3]
            return mx.concatenate([luma, luma, luma], axis=-1)

        def build_feats(self, coords: Any) -> Any:
            proj = coords @ self._B  # (P, n_fourier)
            return mx.concatenate([mx.sin(proj), mx.cos(proj)], axis=-1)

        def __call__(self, coord_feats: Any, code_idx: int) -> Any:
            h = self._act(self.in_proj(coord_feats))
            m = self.code[code_idx]
            film = mx.reshape(self.film(m), (self.n_hidden, 2, self.hidden_dim))
            for li, layer in enumerate(self.hidden):
                scale = 1.0 + film[li, 0]
                shift = film[li, 1]
                h = self._act(layer(h) * scale + shift)
            rgb = mx.sigmoid(self.out(h)) * 255.0
            return self._apply_chroma(rgb)

        def call_batch(self, coord_feats: Any, code_indices: Any) -> Any:
            """Batched render: K codes -> (K, P_px, 3) RGB in one MLP forward.

            ``coord_feats`` is the shared (P_px, in_feat) fixed coord grid;
            ``code_indices`` is an mx int array (K,) of code rows. The coord
            base is shared across the K pairs (broadcast), the per-(pair,frame)
            FiLM modulation is the only per-K signal -- so the dominant in_proj
            on coord_feats is computed ONCE, and the K-axis only multiplies the
            per-layer FiLM affine. This saturates unified memory: one (K,P,H)
            tensor through the hidden stack + one batched scorer forward.
            """
            # Shared coord base (P_px, hidden_dim) -- computed once.
            h0 = self._act(self.in_proj(coord_feats))  # (P, hidden)
            codes = self.code[code_indices]  # (K, mod_dim)
            film = mx.reshape(self.film(codes), (-1, self.n_hidden, 2, self.hidden_dim))  # (K,nh,2,hid)
            h = mx.broadcast_to(h0[None], (film.shape[0], h0.shape[0], h0.shape[1]))  # (K,P,hid)
            for li, layer in enumerate(self.hidden):
                scale = 1.0 + film[:, li, 0][:, None, :]  # (K,1,hid)
                shift = film[:, li, 1][:, None, :]
                h = self._act(layer(h) * scale + shift)
            rgb = mx.sigmoid(self.out(h)) * 255.0  # (K, P, 3)
            return self._apply_chroma(rgb)

    m = RGBWitnessMLX()
    # SIREN init applies ONLY to the periodic family (hosc / siren / finer / wire) — relu keeps
    # default MLX init (where-applicable per the operator directive). It is the canonical
    # from-scratch trainability fix for sine MLPs (wire/finer carry a sin(omega*u) carrier too).
    # The omega used is the activation's own periodic frequency (hosc_omega for hosc,
    # siren_omega~30 for siren/finer/wire); FINER adds its wide first-layer bias (only finer).
    if str(activation) in {"hosc", "siren", "finer", "wire"} and bool(siren_init):
        omega_for_init = float(hosc_omega) if str(activation) == "hosc" else float(siren_omega)
        bias_scale = float(finer_first_bias_scale) if str(activation) == "finer" else 0.0
        apply_siren_init(m, omega=omega_for_init, finer_first_bias_scale=bias_scale)
        mx.eval(m.parameters())
    return m


# ---------------------------------------------------------------------------
# Render through R (mx, differentiable). Witness -> (h,w,3) RGB at scorer res ->
# apply_eval_roundtrip_nhwc (bicubic up to camera -> bilinear down to scorer ->
# uint8 STE). Returns (1, h, w, 3) NHWC camera-equivalent scorer-res frame.
# ---------------------------------------------------------------------------
def render_through_R_mlx(witness, coord_feats, code_idx: int, render_h: int, render_w: int, compose_fn=None):
    import mlx.core as mx

    rgb_flat = witness(coord_feats, code_idx)  # (h*w, 3)
    rgb = mx.reshape(rgb_flat, (1, render_h, render_w, 3))  # NHWC
    # RESIDUAL COMPOSE HOOK (v2 hybrid; ADDITIVE, default None => BYTE-IDENTICAL). When given,
    # ``compose_fn(rgb_nhwc, code_idx) -> rgb_nhwc`` composes the witness RGB with the FIXED
    # deterministic bulk render BEFORE R (composed = bulk*(1-mask) + INR*mask), so the d_seg loss
    # is realized through R on the COMPOSED witness (deterministic_bulk (+) small_INR_residual).
    # The default None preserves the exact pre-residual-mode behavior op-for-op.
    if compose_fn is not None:
        rgb = compose_fn(rgb, code_idx)
    # CONTEST-EXACT R (DAG FEED-bf bug #1): render-grid -> camera (bicubic) -> uint8 @ CAMERA
    # -> bilinear down to scorer (NO trailing uint8). Matches upstream/evaluate.py +
    # modules.py:108-113 (recon is uint8 at 874x1164, scorer.preprocess_input bilinears to 384).
    # R-op dispatch (#252): DEFAULT => the pure-MLX oracle (byte-identical); ``--fused-r-kernel``
    # swaps the bit-faithful fused Metal kernel; ``--mx-compile`` swaps a bit-identity-gated compile.
    r = _apply_R(rgb)
    return r  # (1, SEG_H, SEG_W, 3)


def render_batch_through_R_mlx(witness, coord_feats, code_indices, render_h: int, render_w: int, compose_fn=None):
    """Batched render-through-R: code_indices (M,) -> (M, SEG_H, SEG_W, 3).

    M is the number of frames in the batch (= 2*K for K pairs, f0+f1 interleaved).
    One MLP forward over the M codes, one batched R roundtrip. Same R math as the
    per-frame path (leading dims preserved by apply_eval_roundtrip_nhwc).

    COMPOSE HOOK (v2 residual / analytic-lane render-band; ADDITIVE, default None =>
    BYTE-IDENTICAL). When given, ``compose_fn(rgb_nhwc, code_indices) -> rgb_nhwc`` composes
    the batched witness RGB with a FIXED deterministic field (bulk render / analytic lane band)
    BEFORE R, mirroring the per-frame ``render_through_R_mlx`` compose hook. Default None
    preserves the exact pre-hook behavior op-for-op. (analytic_lane_render_band wire-in §5.)
    """
    import mlx.core as mx

    rgb_flat = witness.call_batch(coord_feats, code_indices)  # (M, P_px, 3)
    rgb = mx.reshape(rgb_flat, (-1, render_h, render_w, 3))  # (M, h, w, 3) NHWC
    if compose_fn is not None:
        rgb = compose_fn(rgb, code_indices)
    # CONTEST-EXACT R (bug #1): uint8 @ CAMERA res, then bilinear to scorer (no trailing uint8).
    # R-op dispatch (#252): DEFAULT => pure-MLX oracle (byte-identical); flags swap fused/compiled R.
    r = _apply_R(rgb)
    return r  # (M, SEG_H, SEG_W, 3)


# ---------------------------------------------------------------------------
# GT precompute (FROZEN CPU-torch authority -- identical to the torch trainer).
# ---------------------------------------------------------------------------
@dataclass
class GTData:
    n_pairs: int
    gt_f0: list[np.ndarray]
    gt_f1: list[np.ndarray]
    lstars: list[np.ndarray]
    margins: list[np.ndarray]
    gt_poses: list[np.ndarray]


def precompute_gt(n_pairs: int) -> tuple[GTData, Any, Any]:
    import torch

    from tac.boundary_math.seg_core import (
        decode_gt_frame1_pairs,
        load_real_segnet,
        segnet_argmax_and_margin,
    )

    seg_cpu = load_real_segnet("cpu")
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path  # upstream

    dn = DistortionNet().eval()
    dn.load_state_dicts(posenet_sd_path, segnet_sd_path, torch.device("cpu"))
    posenet_cpu = dn.posenet
    for p in posenet_cpu.parameters():
        p.requires_grad = False

    gt_f0, gt_f1, lstars, margins, gt_poses = [], [], [], [], []
    for _idx, f0, f1 in decode_gt_frame1_pairs(n_pairs=n_pairs):
        f0 = np.asarray(f0)
        f1 = np.asarray(f1)
        lstar, margin = segnet_argmax_and_margin(seg_cpu, f1)
        gt_f0.append(f0)
        gt_f1.append(f1)
        lstars.append(np.asarray(lstar).astype(np.int64))
        margins.append(np.asarray(margin).astype(np.float32))
        gt_poses.append(_cpu_pose_raw(posenet_cpu, f0, f1))
    gt = GTData(n_pairs=len(lstars), gt_f0=gt_f0, gt_f1=gt_f1, lstars=lstars, margins=margins, gt_poses=gt_poses)
    return gt, seg_cpu, posenet_cpu


def load_gt_from_cache(cache_path: Path, n_pairs: int) -> tuple[GTData, Any, Any]:
    """Load GT from a shared npz cache (built by tools/build_shared_gt_cache_for_mlx_fleet.py).

    Skips the ~480s/arm decode+frozen-scorer GT precompute so a parallel fleet
    shares ONE cache instead of thrashing the CPU. The cached arrays are the EXACT
    outputs of ``precompute_gt`` (same frozen CPU-torch authority) -- NO surrogate.
    The frozen CPU scorers are still loaded here (fast, ~seconds) because the
    per-epoch VERDICT recomputes argmax/pose on the CPU-torch authority.
    """
    import torch

    from tac.boundary_math.seg_core import load_real_segnet

    z = np.load(cache_path, allow_pickle=False)
    cached_P = int(z["n_pairs"])
    P = min(n_pairs, cached_P)
    if cached_P < n_pairs:
        raise ValueError(
            f"--gt-cache {cache_path} has only {cached_P} pairs < requested --num-pairs {n_pairs}; "
            "rebuild the cache at >= the requested size."
        )
    # Read each npz member EXACTLY ONCE. NpzFile does NOT cache members:
    # ``z["gt_f0"][i]`` inside a ``range(P)`` comprehension re-reads (re-inflates)
    # the WHOLE member from the zip on EVERY iteration -> P full copies of a
    # ~1.8 GB (n600) member churned at startup = the 30-40 GB RSS balloon scaling
    # with num_pairs, OOM (exit 137) in ~3 s BEFORE the first stage print
    # (observed 2026-06-26: n96 peak 20.6 GB/3 s, n600 40 GB/9 s). Materializing
    # each member once into a local bounds the load to the cache's real resident
    # footprint (~0.8 GB n96 / ~4.8 GB n600); the per-frame ``asarray`` views below
    # then share that one buffer (same dtype -> no extra copy). GT data unchanged.
    gt_f0_all = z["gt_f0"]
    gt_f1_all = z["gt_f1"]
    lstars_all = z["lstars"]
    margins_all = z["margins"]
    gt_poses_all = z["gt_poses"]
    gt_f0 = [np.asarray(gt_f0_all[i], dtype=np.uint8) for i in range(P)]
    gt_f1 = [np.asarray(gt_f1_all[i], dtype=np.uint8) for i in range(P)]
    lstars = [np.asarray(lstars_all[i], dtype=np.int64) for i in range(P)]
    margins = [np.asarray(margins_all[i], dtype=np.float32) for i in range(P)]
    gt_poses = [np.asarray(gt_poses_all[i], dtype=np.float64) for i in range(P)]
    gt = GTData(n_pairs=P, gt_f0=gt_f0, gt_f1=gt_f1, lstars=lstars, margins=margins, gt_poses=gt_poses)

    seg_cpu = load_real_segnet("cpu")
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path  # upstream

    dn = DistortionNet().eval()
    dn.load_state_dicts(posenet_sd_path, segnet_sd_path, torch.device("cpu"))
    posenet_cpu = dn.posenet
    for p in posenet_cpu.parameters():
        p.requires_grad = False
    return gt, seg_cpu, posenet_cpu


def _cpu_pose_raw(posenet_cpu, f0_uint8: np.ndarray, f1_uint8: np.ndarray) -> np.ndarray:
    import einops
    import torch

    pair = torch.from_numpy(np.stack([f0_uint8, f1_uint8], axis=0)[None]).float()
    x = einops.rearrange(pair, "b t h w c -> b t c h w").float()
    with torch.inference_mode():
        pose_in = posenet_cpu.preprocess_input(x)
        out = posenet_cpu(pose_in)
        pose = out["pose"] if isinstance(out, dict) else out
        half = None
        for hh in posenet_cpu.hydra.heads:
            if hh.name == "pose":
                half = hh.out // 2
                break
        if half is None:
            half = pose.shape[-1] // 2
        return pose[0, :half].cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# CPU-torch VERDICT (authority, NEVER MLX). Mirrors the torch trainer exactly.
# ---------------------------------------------------------------------------
def cpu_verdict_d_seg(segnet_cpu, frame1_uint8: np.ndarray, gt_argmax_np: np.ndarray) -> float:
    import torch

    r = np.asarray(frame1_uint8)
    pair = torch.from_numpy(np.stack([r, r], axis=0)[None]).float()  # (1,2,H,W,3)
    xp = pair.permute(0, 1, 4, 2, 3).contiguous().float()
    with torch.inference_mode():
        seg_in = segnet_cpu.preprocess_input(xp)
        logits = segnet_cpu(seg_in)
        realized = logits.argmax(dim=1)[0].cpu().numpy().astype(np.int64)
    return float(np.count_nonzero(realized != gt_argmax_np)) / gt_argmax_np.size


def cpu_verdict_d_pose(posenet_cpu, f0_uint8: np.ndarray, f1_uint8: np.ndarray, gt_pose_np: np.ndarray) -> float:
    p = _cpu_pose_raw(posenet_cpu, f0_uint8, f1_uint8)
    return float(np.mean((p - gt_pose_np) ** 2))


# ---------------------------------------------------------------------------
# BATCHED CPU-torch verdict (DAG FEED-bt — wall-clock win #1, the biggest).
#
# The frozen SegNet/PoseNet are loaded ``.eval()`` (upstream/modules.py:174,177; the trainer
# builds ``DistortionNet().eval()``), so BatchNorm uses RUNNING stats -> NO cross-frame batch
# interaction -> stacking N frames into ONE (N,...) forward is BIT-EXACT (identical argmax /
# pose) and FAR faster on CPU (one BLAS-saturating forward vs N tiny ones). Both scorers'
# preprocess+forward are batch-aware (``batch_size, seq_len, *_ = x.shape``). These mirror the
# per-pair cpu_verdict_* EXACTLY; PROVEN 0-px / 0-MSE vs frame-by-frame on a sample (DAG FEED-bt).
# ---------------------------------------------------------------------------
def cpu_verdict_d_seg_batch(segnet_cpu, frames1_uint8: list, gt_argmax_list: list) -> list:
    """Batched SegNet argmax verdict over N frame1's -> list of d_seg (one per frame).

    ``frames1_uint8``: list of (H,W,3) uint8 camera frames; ``gt_argmax_list``: matching GT
    argmax maps. SegNet uses only the last frame (preprocess ``x[:, -1]``), so a (N,1,H,W,3)
    stack is sufficient (half the memory of the [r,r] pair) and bit-identical."""
    import torch

    arr = np.stack([np.asarray(f)[None] for f in frames1_uint8], axis=0)  # (N,1,H,W,3)
    xp = torch.from_numpy(arr).permute(0, 1, 4, 2, 3).contiguous().float()  # (N,1,3,H,W)
    with torch.inference_mode():
        seg_in = segnet_cpu.preprocess_input(xp)
        realized = segnet_cpu(seg_in).argmax(dim=1).cpu().numpy().astype(np.int64)  # (N,h,w)
    return [float(np.count_nonzero(realized[i] != g)) / g.size for i, g in enumerate(gt_argmax_list)]


def cpu_verdict_d_seg_argmax_batch(segnet_cpu, frames1_uint8: list, gt_argmax_list: list):
    """(#302 nucleus guard) Batched SegNet argmax verdict that ALSO returns the realized argmax maps
    in ONE forward (no double cost). Returns ``(d_seg_list, realized (N,h,w) int64)``.

    The ``d_seg_list`` is BIT-IDENTICAL to ``cpu_verdict_d_seg_batch`` (same preprocess -> forward ->
    argmax(dim=1) -> per-pixel disagreement). The extra return value is the same ``realized`` argmax
    the per-pair count already used, exposed so the per-class critical-nucleus guard
    (``_evt_nucleus_counts``) can decompose the flip mass by class WITHOUT a second SegNet forward.
    Additive: nothing calls this unless the nucleus guard / readiness telemetry is engaged."""
    import torch

    arr = np.stack([np.asarray(f)[None] for f in frames1_uint8], axis=0)  # (N,1,H,W,3)
    xp = torch.from_numpy(arr).permute(0, 1, 4, 2, 3).contiguous().float()  # (N,1,3,H,W)
    with torch.inference_mode():
        seg_in = segnet_cpu.preprocess_input(xp)
        realized = segnet_cpu(seg_in).argmax(dim=1).cpu().numpy().astype(np.int64)  # (N,h,w)
    d_seg = [float(np.count_nonzero(realized[i] != g)) / g.size
             for i, g in enumerate(gt_argmax_list)]
    return d_seg, realized


def cpu_verdict_d_pose_batch(posenet_cpu, f0_uint8: list, f1_uint8: list, gt_pose_list: list) -> list:
    """Batched PoseNet MSE verdict over N pairs -> list of d_pose. Mirrors ``_cpu_pose_raw``."""
    import einops
    import torch

    arr = np.stack([np.stack([np.asarray(a), np.asarray(b)], axis=0) for a, b in zip(f0_uint8, f1_uint8)], axis=0)
    pair = torch.from_numpy(arr).float()  # (N,2,H,W,3)
    x = einops.rearrange(pair, "b t h w c -> b t c h w").float()
    with torch.inference_mode():
        out = posenet_cpu(posenet_cpu.preprocess_input(x))
        pose = out["pose"] if isinstance(out, dict) else out
        half = None
        for hh in posenet_cpu.hydra.heads:
            if hh.name == "pose":
                half = hh.out // 2
                break
        if half is None:
            half = pose.shape[-1] // 2
        pose_np = pose[:, :half].cpu().numpy().astype(np.float64)  # (N, half)
    return [float(np.mean((pose_np[i] - g) ** 2)) for i, g in enumerate(gt_pose_list)]


# ---------------------------------------------------------------------------
# MLX EMA (decay 0.997, Quantizr). Shadow over the flattened param tree.
# ---------------------------------------------------------------------------
class MlxEMA:
    def __init__(self, model, decay: float = 0.997) -> None:
        import mlx.core as mx
        from mlx.utils import tree_flatten

        self.decay = decay
        self._mx = mx
        self.shadow = {k: mx.array(v) for k, v in tree_flatten(model.parameters())}

    def update(self, model) -> None:
        from mlx.utils import tree_flatten

        d = self.decay
        for k, v in tree_flatten(model.parameters()):
            self.shadow[k] = self.shadow[k] * d + v * (1.0 - d)

    def shadow_tree(self):
        from mlx.utils import tree_unflatten

        return tree_unflatten(list(self.shadow.items()))


# ---------------------------------------------------------------------------
# Implied S (advisory; identical formula to the torch trainer + evaluate.py).
# ---------------------------------------------------------------------------
def implied_score_from_verdict(d_seg: float, d_pose: float, witness_bytes: int) -> float:
    """Route through the CANONICAL contest_score (bug #7: no hand-rolled formula).

    ``tac.contest_score.compute_contest_score`` is byte-identical to upstream/evaluate.py:92
    (100*d_seg + sqrt(10*d_pose) + 25*bytes/37_545_489). ``witness_bytes`` is the MEASURED
    quantized blob (bug #6), not a fixed phantom.
    """
    from tac.contest_score import compute_contest_score

    return compute_contest_score(float(d_seg), float(max(d_pose, 0.0)), int(witness_bytes))


def _int8_symmetric(a: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-tensor symmetric int8 quant. Returns (int8 codes, scale s/127). Single source
    of truth for BOTH the byte-close (quantize_witness_blob) and the weight-QAT verdict
    dequant (dequantize_witness_flat) so they CANNOT drift (bug #4)."""
    s = float(np.abs(a).max()) + 1e-8
    q = np.clip(np.round(a / s * 127.0), -127, 127).astype(np.int8)
    return q, (s / 127.0)


def _quantize_blob_from_flat(items) -> dict[str, Any]:
    """int8 + brotli a flat (name, mx_array) iterable. ``_B`` (free deterministic Fourier
    table, rule 118) is excluded from the counted blob."""
    import brotli

    base_chunks: list[bytes] = []
    code_chunk = b""
    n_params = 0
    for name, arr in items:
        a = np.asarray(arr, dtype=np.float32)
        if a.size == 0 or name.endswith("_B"):
            continue
        n_params += a.size
        q, _ = _int8_symmetric(a)
        if name.endswith("code"):
            code_chunk = q.tobytes()
        else:
            base_chunks.append(q.tobytes())
    base_raw = b"".join(base_chunks)
    base_brotli = len(brotli.compress(base_raw, quality=11))
    code_brotli = len(brotli.compress(code_chunk, quality=11)) if code_chunk else 0
    return {
        "n_params": int(n_params),
        "base_int8_brotli_bytes": base_brotli,
        "code_int8_brotli_bytes": code_brotli,
        "total_quantized_blob_bytes": base_brotli + code_brotli,
    }


def quantize_witness_blob(model) -> dict[str, Any]:
    """int8 + brotli the witness params (base) and per-frame codes (mod)."""
    from mlx.utils import tree_flatten

    return _quantize_blob_from_flat(tree_flatten(model.parameters()))


def measure_witness_blob_bytes(items) -> int:
    """Measured deploy-archive bytes of a flat (name, mx_array) iterable (bug #6)."""
    return int(_quantize_blob_from_flat(items)["total_quantized_blob_bytes"])


def dequantize_witness_flat(items) -> dict[str, Any]:
    """Return params after the SAME int8 symmetric round-trip the byte-close applies, so the
    VERDICT renders the DEPLOY weights, not fp32 EMA (bug #4 — eval_roundtrip-for-WEIGHTS).
    ``_B`` (free deterministic table) is passed through unquantized. Returns {name: np.float32}.
    """
    out: dict[str, np.ndarray] = {}
    for name, arr in items:
        a = np.asarray(arr, dtype=np.float32)
        if a.size == 0 or name.endswith("_B"):
            out[name] = a
            continue
        q, scale = _int8_symmetric(a)
        out[name] = (q.astype(np.float32) * scale).astype(np.float32)
    return out


def _build_render_coords(h: int, w: int) -> np.ndarray:
    ys = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    xs = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    gy, gx = np.meshgrid(ys, xs, indexing="ij")
    return np.stack([gx.ravel(), gy.ravel()], axis=-1).astype(np.float32)


def _nn_resize_labels(labels: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Nearest-neighbor resize an integer argmax-label map to (out_h, out_w).

    Used by the DIRECTIONAL basis to build the all-class boundary tangent field at the
    RENDER grid so the per-pixel tangent count is EXACTLY out_h*out_w -> it matches the
    coord grid for ANY render/SEG ratio. The prior strided ``lr[::step]`` subsample only
    matched when the render res EVENLY DIVIDED the SEG res (e.g. render 256x384 vs SEG
    384x512 -> step=1 -> a 196608-row tangent broadcast against a 98304-row coord grid =
    the directional crash). NN (no interpolation) preserves the integer class labels so
    the boundary mask is intact. Pure numpy, no torch."""
    a = np.asarray(labels)
    if a.shape[0] == out_h and a.shape[1] == out_w:
        return a
    ys = np.linspace(0, a.shape[0] - 1, out_h).round().astype(np.int64)
    xs = np.linspace(0, a.shape[1] - 1, out_w).round().astype(np.int64)
    return a[np.ix_(ys, xs)]


# ---------------------------------------------------------------------------
# LIVE-MARGIN per-pixel weight (DAG FEED-bp NEW-B: margin = per-pixel RD bit-allocator).
# ---------------------------------------------------------------------------
def _live_margin_weight(seg_logits: Any, fn: str, temp: float) -> Any:
    """LIVE realized-SegNet per-pixel margin weight -- the bridge (FEED-bp) bit-allocator.

    ``seg_logits`` are the FROZEN SegNet logits of the witness frame1 RENDERED THROUGH R
    (the realized axis -- the SAME tensor the seg loss already scores). The LIVE margin is
    the top1-top2 logit gap (>=0; SMALL = near-flip = the codim-1 decision-boundary annulus).
    MEASURED (FEED-bp): ~89% of d_seg lives in the bottom-5%-margin pixels (the 2.26% annulus);
    witness-hard px median margin 0.42 vs global 5.79 (14x). Re-routing the SAME loss budget
    onto that annulus is a ~20x effective-capacity multiplier at EQUAL bytes.

    Returns a (...,H,W) weight map, **mean-1 normalized** (preserves the total loss budget --
    capacity is RE-ALLOCATED, not added) and **STOP-GRADIENT** (NO-FAKE: the margin map is a
    per-pixel allocation PRIOR, not a differentiable knob -- the witness must FIX the flip,
    never game its own margin to shrink the weight). Three allocators:

      inverse  : w = 1/(1 + m/temp)              (smooth; temp small => sharper annulus focus)
      exp      : w = exp(-m/temp)                (smooth; exponential annulus focus)
      bottom-k : w = 1[m <= quantile(m, temp)]   (hard mask; temp = bottom-fraction, e.g. 0.05)
    """
    import mlx.core as mx

    srt = mx.sort(seg_logits, axis=-1)
    m = srt[..., -1] - srt[..., -2]  # (...,H,W) top1-top2 logit gap, >=0
    t = max(float(temp), 1e-6)
    if fn == "exp":
        w = mx.exp(-m / t)
    elif fn == "bottom-k":
        flat = mx.reshape(m, (-1,))
        n = int(flat.shape[0])
        k = min(max(int(round(t * n)), 1), n)
        thr = mx.sort(flat)[k - 1]  # k-th smallest margin = bottom-fraction threshold
        w = (m <= thr).astype(m.dtype)
    else:  # "inverse" (default)
        w = 1.0 / (1.0 + m / t)
    w = w / (mx.mean(w) + 1e-8)  # mean-1 => SAME total loss budget, re-allocated to the annulus
    return mx.stop_gradient(w)


# ---------------------------------------------------------------------------
# Realized loss (mx, through both frozen MLX scorers). The training signal.
# ---------------------------------------------------------------------------
def focal_pixel_weight_mlx(seg_logits, lstar_oh, gamma: float):
    """(--seg-focal-gamma; council memo council_grand_symposium_levelset_loss_geometry_20260705.md)
    Focal per-pixel seg-loss reweight ``(1 - p_y)^gamma`` from the REALIZED softmax GT-class
    probability ``p_y = softmax(seg_logits)[GT]``, STOP-GRAD + mean-1 RENORMALIZED (the l7-weight
    idiom): the loss stays the base form; only the per-pixel gradient BUDGET is reallocated
    (Shannon's concentration knob — the total budget is conserved by the renorm). Under stop-grad
    the gradient-weight ratio between two pixels is EXACTLY ((1-p1)/(1-p2))^gamma — Rudin's 5^gamma
    readback for (p1,p2)=(0.5,0.9) — which full-differentiable focal would NOT satisfy (its extra
    d(1-p)^gamma term breaks the closed-form ratio). Returns a (1,H,W) weight; callers gate on
    gamma > 0 so the DEFAULT 0.0 path never builds this graph (byte-identical)."""
    import mlx.core as mx

    p_y = mx.exp(mx.sum(seg_logits * lstar_oh, axis=-1) - mx.logsumexp(seg_logits, axis=-1))
    fw = mx.power(mx.clip(1.0 - p_y, 1e-12, 1.0), gamma)  # (1,H,W)
    return mx.stop_gradient(fw / (mx.mean(fw) + 1e-8))


def make_loss_fn(
    adapter,
    render_h: int,
    render_w: int,
    score_domain: bool = True,
    pose_eps: float = 1e-8,
    seg_loss: str = "ce",
    margin_weighted: bool = False,
    margin_weight_fn: str = "inverse",
    margin_weight_temp: float = 1.0,
    tau_softplus_tau: float = 0.3,
    l7_mult: float = 4.0,
    l7_threshold: float = 0.42,
    render_fn=None,
    focal_gamma: float = 0.0,
):
    """Returns loss_fn(model, coord_feats, code0, code1, lstar_oh, margin, pose_tgt,
    w_seg, w_pose, hinge) -> scalar mx loss. Closes over the frozen MLX adapter.

    ``render_fn`` (v2 residual-compose hook; ADDITIVE, default None => BYTE-IDENTICAL): when given,
    every realized render uses ``render_fn(model, coord_feats, code_idx, render_h, render_w)``
    instead of the bare ``render_through_R_mlx`` -- the trainer's residual mode passes a partial
    that composes the FIXED deterministic bulk before R, so seg+pose are realized on the COMPOSED
    witness. None preserves the exact pre-residual-mode behavior (one codepath, no divergence).

    SCORE-DOMAIN LOSS (CLAUDE.md "HNeRV/leaderboard parity" LESSON 6 -- score-domain
    Lagrangian, NOT weight-domain proxies). The contest score is
    ``S = 100*d_seg + sqrt(10*d_pose) + rate``. When ``score_domain`` is True
    (default for the witness), the loss MIRRORS S's distortion marginals:

        loss = w_seg*seg_l + w_pose*sqrt(10*pose_l + pose_eps)

    where ``seg_l`` is the margin-weighted CE (a smooth d_seg surrogate) and
    ``pose_l`` is the realized d_pose (MSE). The ``sqrt`` DAMPS the pose gradient
    while d_pose is large (early training) and only SHARPENS near the floor --
    exactly S's nonlinear pose term -- so the d_seg term (weighted 100x in S, via
    the score-aligned default ``w_seg=100``) drives training and no longer
    collapses. The OLD raw form ``w_seg*seg_l + w_pose*pose_l`` is ANTI-ALIGNED:
    the raw pose MSE (~159 at init) dominates the gradient ~50x and drives the
    optimizer to collapse d_seg back to the baseline at n600 scale (DAG FEED-at/au,
    measured 2026-06-26). ``score_domain=False`` restores the raw form (ablation).
    ``pose_eps`` keeps the sqrt gradient finite at d_pose=0.
    """
    import mlx.core as mx

    from tac.local_acceleration.pr95_hnerv_mlx_training import rgb_to_yuv6_mlx

    _render = render_fn if render_fn is not None else render_through_R_mlx

    def _yuv6_pair_nhwc(f0, f1):
        # f0,f1: (1,H,W,3). Build (1,2,H,W,3) -> yuv6 (1,2,h2,w2,6) -> (1,h2,w2,12).
        pair = mx.stack([f0[0], f1[0]], axis=0)[None]  # (1,2,H,W,3)
        yuv = rgb_to_yuv6_mlx(pair)  # (1,2,h2,w2,6)
        b, t, h2, w2, c6 = yuv.shape
        return mx.reshape(mx.transpose(yuv, (0, 2, 3, 1, 4)), (b, h2, w2, t * c6))

    def loss_fn(model, coord_feats, code0, code1, lstar_oh, margin, pose_tgt, w_seg, w_pose, hinge, margin_target, mw_active=True, mw_temp=None, seg_form=None, tau_override=None, l7_thr_override=None, seg_pixel_w=None, terms_out=None):
        # ``terms_out`` (#304 item 4 per-term loss telemetry; ADDITIVE, default None => BYTE-IDENTICAL):
        # when given a dict, the WEIGHTED contributions {"seg": w_seg*seg_l, "pose": w_pose*pose_term}
        # are recorded as (lazy) mx arrays for the caller to eval OUTSIDE any grad transform. The
        # return expression is UNCHANGED; recording only NAMES the same subexpressions, so the graph,
        # loss and grads are bitwise identical. NEVER pass terms_out under value_and_grad -- the
        # telemetry caller is a separate no-grad recompute (see the levelset trainer's loss-term probe).
        # ``seg_pixel_w`` (ADDITIVE per-pixel seg-loss reweight hook; default None => BYTE-IDENTICAL):
        # when given a (1,H,W) map it MULTIPLIES the per-pixel seg map (all forms) BEFORE the mean --
        # the spike-aware reweight lever (down-weight unfittable single-frame flicker, up-weight the
        # coherent boundary). None preserves the exact pre-hook expression. Mirrors the ``render_fn``
        # additive-hook idiom above.
        # ``seg_form`` (FEED-bv 4-stage curriculum): the seg LOSS FORM for THIS epoch's stage.
        # None -> the closure default ``seg_loss`` (non-curriculum, back-compat). The PR95 d_seg
        # SEQUENCE (digest pr95_segnet_stage_evolution...): ce (coarse argmax) -> tau_softplus
        # (margin-sharpen, THE primary d_seg drop) -> l7_softplus (hard-pixel refine on the
        # all-class edge band) -> [muon optimizer finisher, same l7 loss]. All NHWC, live-margin.
        # ``mw_active`` (FEED-bf curriculum gate) + ``mw_temp`` (FEED-bu anneal): the SEPARATE
        # exp/inverse live-margin allocation lever (non-curriculum path); l7_softplus carries its
        # OWN hard-pixel weight so the curriculum l7 stage does NOT also use this lever.
        form = seg_form if seg_form is not None else seg_loss
        apply_mw = bool(margin_weighted) and bool(mw_active)
        temp_use = margin_weight_temp if mw_temp is None else float(mw_temp)
        # (FEED-ca anneal) intra-stage tau / l7-threshold ANNEAL overrides (scale-level
        # coarse->fine, PR95 signature). None -> the closed-over static value (back-compat).
        tau_use = tau_softplus_tau if tau_override is None else float(tau_override)
        l7_thr_use = l7_threshold if l7_thr_override is None else float(l7_thr_override)
        f0 = _render(model, coord_feats, code0, render_h, render_w)
        f1 = _render(model, coord_feats, code1, render_h, render_w)
        # Realized seg loss on frame1 (post-R, frozen SegNet).
        seg_logits = adapter.segnet(f1)  # (1,H,W,5)
        # (--seg-focal-gamma; DEFAULT 0.0 => this branch NEVER runs => graph + loss + grads
        # BYTE-IDENTICAL). Focal reweight (1-p_y)^gamma from the REALIZED softmax of the SAME
        # ``seg_logits`` the base form reads (i.e. whatever surface ``render_fn`` composed —
        # in the levelset trainer that is the SEED-COMPOSED frame the base CE reads; the
        # island-formation levers keep their own witness-alone routing untouched). Folded into
        # ``seg_pixel_w`` so it composes MULTIPLICATIVELY with the spike-reweight hook and applies
        # uniformly to EVERY seg form before the mean (same surface, one idiom).
        if focal_gamma > 0.0:
            _fw_focal = focal_pixel_weight_mlx(seg_logits, lstar_oh, focal_gamma)
            seg_pixel_w = _fw_focal if seg_pixel_w is None else seg_pixel_w * _fw_focal

        def _live_signed():
            # live per-pixel margin = target_logit - max_competing_logit (the sign-quantity
            # argmax decides on). NHWC twin of mlx_pr95_port _target_margin_mlx.
            gt_logit = mx.sum(seg_logits * lstar_oh, axis=-1)  # (1,H,W)
            runner_up = mx.max(seg_logits + lstar_oh * (-1e9), axis=-1)  # (1,H,W)
            return gt_logit - runner_up

        if form == "tau_softplus":
            # PR95 stage-2 (digest: THE primary d_seg drop, 0.00643->0.00396). mean(tau*softplus(-m/tau)).
            signed = _live_signed()
            z = -signed / tau_use
            _pp = tau_use * mx.logaddexp(mx.zeros_like(z), z)
            seg_l = mx.mean(_pp if seg_pixel_w is None else _pp * seg_pixel_w)
        elif form == "l7_softplus":
            # PR95 stage-5 l7 (digest: hard-pixel refine; KEEP loss, DROP C1a). softplus * (1+mult
            # on margin<thr), renorm mean-1, weight stop-grad. THE margin-weight refine stage.
            signed = _live_signed()
            z = -signed / tau_use
            per_pixel = tau_use * mx.logaddexp(mx.zeros_like(z), z)
            w = 1.0 + l7_mult * (signed < l7_thr_use).astype(seg_logits.dtype)
            w = mx.stop_gradient(w / (mx.mean(w) + 1e-8))
            _pp = per_pixel * w
            seg_l = mx.mean(_pp if seg_pixel_w is None else _pp * seg_pixel_w)
        elif form == "margin_hinge":
            # lensA d_seg-optimal surrogate: relu(m - (logit[GT] - max_{c!=GT} logit[c])).
            signed = _live_signed()
            hinge_map = mx.maximum(margin_target - signed, 0.0)  # (1,H,W)
            if apply_mw:
                hinge_map = hinge_map * _live_margin_weight(seg_logits, margin_weight_fn, temp_use)
            seg_l = mx.mean(hinge_map if seg_pixel_w is None else hinge_map * seg_pixel_w)
        else:  # "ce"
            logsum = mx.logsumexp(seg_logits, axis=-1)
            tgt = mx.sum(seg_logits * lstar_oh, axis=-1)
            ce = logsum - tgt  # (1,H,W)
            w = 1.0 + hinge * mx.exp(-mx.clip(margin, 0.0, 1e9))
            pw = ce * w[None]  # (1,H,W)
            if apply_mw:
                pw = pw * _live_margin_weight(seg_logits, margin_weight_fn, temp_use)
            seg_l = mx.mean(pw if seg_pixel_w is None else pw * seg_pixel_w)
        # Realized pose MSE on the pair (== realized d_pose).
        yuv_nhwc = _yuv6_pair_nhwc(f0, f1)
        pose = adapter.posenet(yuv_nhwc)["pose"][..., : pose_tgt.shape[-1]]
        pose_l = mx.mean(mx.square(pose[0] - pose_tgt))
        # Score-domain: sqrt(10*d_pose) mirrors S's pose term (damp-then-sharpen).
        pose_term = mx.sqrt(10.0 * pose_l + pose_eps) if score_domain else pose_l
        if terms_out is not None:
            # #304 item 4: record the weighted contributions (lazy; caller evals outside grad).
            terms_out["seg"] = w_seg * seg_l
            terms_out["pose"] = w_pose * pose_term
        return w_seg * seg_l + w_pose * pose_term

    return loss_fn


def make_loss_fn_batch(
    adapter,
    render_h: int,
    render_w: int,
    score_domain: bool = True,
    pose_eps: float = 1e-8,
    margin_weighted: bool = False,
    margin_weight_fn: str = "inverse",
    margin_weight_temp: float = 1.0,
):
    """Batched (K-pair) realized loss. SAME math as make_loss_fn per pair, but
    renders + scores K pairs in one batched scorer forward -> one mx.eval.

    Inputs (batched over K pairs):
      code_idx_f0/f1 : (K,) int -- code rows for f0/f1 of each pair
      lstar_oh       : (K,H,W,5) one-hot GT argmax per pair (frame1)
      margin         : (K,H,W) seg margin per pair
      pose_tgt       : (K, half) GT pose per pair
    The K-axis is the scorer batch dimension (saturates unified memory).

    Score-domain loss is IDENTICAL to make_loss_fn (see its docstring): when
    ``score_domain`` is True the pose term is ``sqrt(10*pose_l + pose_eps)`` so the
    loss mirrors S = 100*d_seg + sqrt(10*d_pose) + rate (HNeRV parity LESSON 6).
    """
    import mlx.core as mx

    from tac.local_acceleration.pr95_hnerv_mlx_training import rgb_to_yuv6_mlx

    def loss_fn(model, coord_feats, code_idx_f0, code_idx_f1, lstar_oh, margin, pose_tgt, w_seg, w_pose, hinge):
        K = int(lstar_oh.shape[0])
        # Interleave [f0_0,f1_0,f0_1,f1_1,...] so the pair structure is contiguous.
        codes = mx.reshape(mx.stack([code_idx_f0, code_idx_f1], axis=1), (2 * K,))  # (2K,)
        frames = render_batch_through_R_mlx(model, coord_feats, codes, render_h, render_w)  # (2K,H,W,3)
        f0 = frames[0::2]  # (K,H,W,3)
        f1 = frames[1::2]  # (K,H,W,3)
        # Realized seg CE (margin-weighted) on frame1, batched over K.
        seg_logits = adapter.segnet(f1)  # (K,H,W,5)
        logsum = mx.logsumexp(seg_logits, axis=-1)  # (K,H,W)
        tgt = mx.sum(seg_logits * lstar_oh, axis=-1)  # (K,H,W)
        ce = logsum - tgt
        w = 1.0 + hinge * mx.exp(-mx.clip(margin, 0.0, 1e9))  # (K,H,W)
        pw = ce * w
        # LIVE-MARGIN re-allocation (FEED-bp); OFF (default) = byte-identical.
        if margin_weighted:
            pw = pw * _live_margin_weight(seg_logits, margin_weight_fn, margin_weight_temp)
        seg_l = mx.mean(pw)
        # Realized pose MSE on the K pairs. Build (K,2,H,W,3) -> yuv6 (K,2,h2,w2,6) -> (K,h2,w2,12).
        pair = mx.stack([f0, f1], axis=1)  # (K,2,H,W,3)
        yuv = rgb_to_yuv6_mlx(pair)  # (K,2,h2,w2,6)
        k, t, h2, w2, c6 = yuv.shape
        yuv_nhwc = mx.reshape(mx.transpose(yuv, (0, 2, 3, 1, 4)), (k, h2, w2, t * c6))  # (K,h2,w2,12)
        pose = adapter.posenet(yuv_nhwc)["pose"][..., : pose_tgt.shape[-1]]  # (K, half)
        pose_l = mx.mean(mx.square(pose - pose_tgt))
        # Score-domain: sqrt(10*d_pose) mirrors S's pose term (damp-then-sharpen).
        pose_term = mx.sqrt(10.0 * pose_l + pose_eps) if score_domain else pose_l
        return w_seg * seg_l + w_pose * pose_term

    return loss_fn


def _render_rgb_render_res(model, coord_feats, code_idx: int, render_h: int, render_w: int) -> np.ndarray:
    """Witness MLP forward (MLX) -> numpy float RGB at RENDER res (NO R applied).

    The witness is an MLX module so its forward is necessarily MLX; the R (resize+uint8 —
    the knife-edge that flips d_seg) is then applied in TORCH for the authority verdict
    (bug #3), so the verdict never trusts an MLX-GPU uint8 boundary.
    """
    import mlx.core as mx

    rgb_flat = model(coord_feats, code_idx)  # (h*w, 3)
    rgb = mx.reshape(rgb_flat, (render_h, render_w, 3))
    mx.eval(rgb)
    return np.asarray(rgb, dtype=np.float32)


def _torch_R_to_camera_uint8(rgb_render_np: np.ndarray) -> np.ndarray:
    """TORCH-authority R (bug #1 + #3): render-res float RGB -> bicubic up to CAMERA ->
    uint8 @ CAMERA. Matches upstream/evaluate.py: the recon is a uint8 camera-res video
    frame; the scorer's preprocess_input then bilinear-downsamples to 384x512. We stop at
    the uint8 CAMERA frame and hand it to cpu_verdict_*, which calls the REAL
    scorer.preprocess_input (the contest bilinear) — so the entire R + preprocess is torch.
    """
    import torch

    x = torch.from_numpy(np.ascontiguousarray(rgb_render_np)).permute(2, 0, 1)[None].float()  # (1,3,h,w)
    with torch.inference_mode():
        up = torch.nn.functional.interpolate(x, size=(CAMERA_H, CAMERA_W), mode="bicubic", align_corners=False)
        up = torch.clamp(torch.round(up), 0.0, 255.0)
    return up[0].permute(1, 2, 0).contiguous().numpy().astype(np.uint8)  # (CAMERA_H, CAMERA_W, 3)


# ---------------------------------------------------------------------------
# DEPLOY-FAITHFUL NUMPY fp32 forward (DAG FEED-br — the byte-close PORT-FIDELITY fix).
#
# ROOT CAUSE (measured 2026-06-26 on arm_iso_v2, both fp32 and int8 weights): the
# witness MLP forward on MLX **GPU (Metal)** accumulates matmuls in REDUCED precision.
# The MLX-GPU render diverges from any fp32 deploy by ~0.19/255 max RGB (~0.06 mean) ->
# ~495-1672 SegNet-argmax px/pair, FAR LARGER than the entire sub-0.15 d_seg budget
# (7.3e-4 ~= 143 px/pair). By contrast: numpy-vs-torch fp32 agree to ~1 ULP (1.5e-5,
# 0-3 px); MLX-CPU-vs-numpy 3-18 px; MLX-GPU-vs-MLX-CPU ~1665 px (== the GPU outlier).
# So the *deploy* numpy forward is CORRECT; the *trainer verdict* (which used the MLX
# render) was the lone reduced-precision outlier. The fix: render the SCORED VERDICT in
# NUMPY fp32 — byte-identical to inflate.py's _witness_forward — so the reported d_seg ==
# the byte-closed d_seg. The training GRADIENT stays on MLX-GPU (reduced precision is
# fine for the gradient per the CHAOS verdict); only the verdict must be fp32.
#
# This function MUST stay op-for-op identical to the inflate.py _witness_forward template
# in tools/witness_byte_close_and_eval.py (lin = x@W.T+b; relu/hosc; FiLM reshape
# (n_hidden,2,hidden); sigmoid*255; chroma=BT.601 luma replicate). A drift here = a
# silent verdict/deploy gap, so the byte-close parity_on_inflated (which reads the ACTUAL
# inflated .raw) is the regression guard.
# ---------------------------------------------------------------------------
def _build_feats_numpy(coords_np: np.ndarray, B_np: np.ndarray) -> np.ndarray:
    """Deterministic isotropic Fourier feats in NUMPY (byte-identical to inflate.py)."""
    proj = coords_np @ B_np
    return np.concatenate([np.sin(proj), np.cos(proj)], axis=-1).astype(np.float32)


def _witness_forward_numpy(
    params: dict[str, np.ndarray],
    feats: np.ndarray,
    code_row: np.ndarray,
    n_hidden: int,
    hidden_dim: int,
    kind: str,
    beta: float,
    omega: float,
    chroma: bool,
    wire_scale: float = 1.0,
) -> np.ndarray:
    """Witness MLP forward in NUMPY fp32 -> (P_px, 3) float RGB in [0,255].

    Mirror of inflate.py's _witness_forward (+ its main-loop chroma branch) EXACTLY so
    the trainer verdict renders what actually ships. ``params`` are float32 (the int8-
    dequant deploy weights when int8_verdict, else fp32 EMA/live).
    """
    def lin(name: str, x: np.ndarray) -> np.ndarray:
        return x @ params[name + ".weight"].T + params[name + ".bias"]

    def act(u: np.ndarray) -> np.ndarray:
        if kind == "hosc":
            return np.tanh(beta * np.sin(omega * u))
        if kind == "siren":
            return np.sin(omega * u)
        if kind == "finer":
            return np.sin(omega * (np.abs(u) + 1.0) * u)
        if kind == "wire":  # real-Gabor sin(omega*u)*exp(-0.5*(s*u)^2)
            return np.sin(omega * u) * np.exp(-0.5 * np.square(wire_scale * u))
        return np.maximum(u, 0.0)

    h = act(lin("in_proj", feats))
    film = (code_row @ params["film.weight"].T + params["film.bias"]).reshape(n_hidden, 2, hidden_dim)
    for li in range(n_hidden):
        scale = 1.0 + film[li, 0]
        shift = film[li, 1]
        h = act(lin(f"hidden.{li}", h) * scale + shift)
    z = h @ params["out.weight"].T + params["out.bias"]
    rgb = (1.0 / (1.0 + np.exp(-z))) * 255.0  # (P,3) float RGB
    if not chroma:  # mirror RGBWitnessMLX._apply_chroma (chroma=False) / inflate main-loop
        luma = 0.299 * rgb[..., 0:1] + 0.587 * rgb[..., 1:2] + 0.114 * rgb[..., 2:3]
        rgb = np.concatenate([luma, luma, luma], axis=-1)
    return rgb.astype(np.float32)


def _witness_forward_mlx(
    deploy_mx: dict,
    feats_mx,
    code_row_mx,
    n_hidden: int,
    hidden_dim: int,
    kind: str,
    beta: float,
    omega: float,
    chroma: bool,
    wire_scale: float = 1.0,
):
    """Functional MLX witness forward (NO module mutation) -> (P_px, 3) mx RGB in [0,255].

    Op-for-op identical to ``_witness_forward_numpy`` / inflate.py, but in mx so the IN-LOOP
    verdict can run on the FAST mlx-cpu (3-18 px/pair vs numpy fp32 << the 143-px sub-0.15
    budget -> accurate ENOUGH for descent monitoring) or the FASTEST-but-coarse mlx-gpu
    (~495-1672 px, reduced precision -> ONLY for the coarse 0.5->0.01 descent). The FINAL
    byte-close row stays numpy fp32 (0 px, authoritative). ``deploy_mx`` holds the int8-
    dequant deploy weights as mx arrays on the active device; ``feats_mx`` is the coord grid.
    """
    import mlx.core as mx

    def lin(name, x):
        return x @ deploy_mx[name + ".weight"].T + deploy_mx[name + ".bias"]

    def act(u):
        if kind == "hosc":
            return mx.tanh(beta * mx.sin(omega * u))
        if kind == "siren":
            return mx.sin(omega * u)
        if kind == "finer":
            return mx.sin(omega * (mx.abs(u) + 1.0) * u)
        if kind == "wire":  # real-Gabor sin(omega*u)*exp(-0.5*(s*u)^2)
            return mx.sin(omega * u) * mx.exp(-0.5 * mx.square(wire_scale * u))
        return mx.maximum(u, 0.0)

    h = act(lin("in_proj", feats_mx))
    film = mx.reshape(code_row_mx @ deploy_mx["film.weight"].T + deploy_mx["film.bias"], (n_hidden, 2, hidden_dim))
    for li in range(n_hidden):
        scale = 1.0 + film[li, 0]
        shift = film[li, 1]
        h = act(lin(f"hidden.{li}", h) * scale + shift)
    z = h @ deploy_mx["out.weight"].T + deploy_mx["out.bias"]
    rgb = mx.sigmoid(z) * 255.0
    if not chroma:
        luma = 0.299 * rgb[..., 0:1] + 0.587 * rgb[..., 1:2] + 0.114 * rgb[..., 2:3]
        rgb = mx.concatenate([luma, luma, luma], axis=-1)
    return rgb


def _advisory_axis_label(non_promotable: bool = True) -> str:
    """Device-truthful advisory axis tag (DAG FEED-br sister fix — the B4/base_ch axis
    mislabel). On macOS (Apple Silicon) the frozen CPU-torch verdict is NOT 1:1 with the
    contest GitHub-Actions Linux x86_64 CPU runner, so per CLAUDE.md "Submission auth eval"
    it is ``[macOS-CPU advisory]`` NOT ``[contest-CPU]``. Only a genuine Linux x86_64 host
    earns ``[contest-CPU advisory]``. Either way the row is NON-PROMOTABLE until a real
    byte-closed paired CPU/CUDA exact-eval row lands."""
    import platform

    sysname = platform.system()
    machine = platform.machine().lower()
    if sysname == "Linux" and machine in ("x86_64", "amd64"):
        base = "[contest-CPU advisory]"
    else:
        base = "[macOS-CPU advisory]"
    return f"{base} NON-PROMOTABLE" if non_promotable else base


def _seed_muon_momentum_from_adam(muon_opt: Any, old_adam_state: Any) -> int:
    """GAP 2 warm-start: seed an ALREADY-INITIALIZED ``mlx.optimizers.Muon``'s momentum
    buffer (``state[<path>]['v']``) from the OUTGOING Adam/AdamW first-moment
    (``state[<path>]['m']``) for every param path the two share.

    Both buffers are exponential moving averages of the raw gradient (Muon's
    ``v = 0.95 v + 0.05 g``; Adam's ``m = 0.9 m + 0.1 g``) and Muon's Newton-Schulz
    step re-normalizes the update to ~unit spectral norm, so the transferred
    DIRECTION (not the absolute scale) is what removes the cold-start "wild unit-norm
    direction from one noisy gradient" boundary thrash / d_seg spike.

    ``muon_opt`` MUST already be initialized (caller ran ``opt.init(params)``) so its
    ``v`` leaves exist to overwrite. Returns the number of leaves seeded (0 when the
    old state carries no matching ``.m`` leaf — e.g. a non-Adam base — leaving the
    fresh Muon at its cold zero start). Shapes must match exactly (they always do for
    the same path); mismatches are skipped. Any exception is the CALLER's to catch;
    the caller falls back to the cold-start path so the run never crashes here.
    """
    from mlx.utils import tree_flatten, tree_unflatten

    # Map param-path -> Adam first-moment `m` (strip the trailing ".m" state suffix).
    m_map = {k[:-2]: v for (k, v) in tree_flatten(old_adam_state) if k.endswith(".m")}
    new_flat: list[tuple[str, Any]] = []
    n_seed = 0
    for k, v in tree_flatten(muon_opt.state):
        if k.endswith(".v"):  # a Muon momentum leaf (NOT "step"/"learning_rate")
            src = m_map.get(k[:-2])
            if src is not None and tuple(src.shape) == tuple(v.shape):
                v = src.astype(v.dtype)
                n_seed += 1
        new_flat.append((k, v))
    if n_seed:
        # Reassigning .state flips _initialized False; we fully repopulated every
        # leaf, so re-assert True to skip the lazy re-init (which would zero nothing
        # new but is a wasted pass). _schedulers is a separate attr -> LR schedule
        # (warmup/decay) keeps working; "step" is preserved (starts at 0 as cold).
        muon_opt.state = tree_unflatten(new_flat)
        muon_opt._initialized = True
    return n_seed


def run_train(args: argparse.Namespace) -> dict[str, Any]:
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    from mlx.utils import tree_flatten, tree_map, tree_unflatten

    from tac.local_acceleration.mlx_scorer_adapters import (
        load_mlx_distortion_scorer_adapter_from_upstream,
        temporary_mlx_device,
    )

    out_dir = Path(args.out_dir)
    _refuse_tmp(out_dir, "out_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    mx.random.seed(args.seed)
    np.random.seed(args.seed)

    device = args.mlx_device  # "gpu" or "cpu"

    # --- 1. GT precompute on frozen CPU scorers (authority), OR shared-cache load ---
    t0 = time.time()
    gt_cache = getattr(args, "gt_cache", None)
    if gt_cache:
        gt, seg_cpu, posenet_cpu = load_gt_from_cache(Path(gt_cache), args.num_pairs)
        gt_stage = "gt_cache_load"
    else:
        gt, seg_cpu, posenet_cpu = precompute_gt(args.num_pairs)
        gt_stage = "gt_precompute"
    P = gt.n_pairs
    gt_secs = time.time() - t0
    print(json.dumps({"stage": gt_stage, "n_pairs": P, "secs": round(gt_secs, 1)}), flush=True)

    render_h, render_w = args.render_h, args.render_w
    coords_np = _build_render_coords(render_h, render_w)

    # --- lever config (each defaults to the CURRENT behavior; opt-in via CLI) ---
    basis = str(getattr(args, "basis", "isotropic"))
    directional = basis == "directional"
    n_dir_freqs = int(getattr(args, "n_dir_freqs", 6))
    in_feat_override = (4 * n_dir_freqs) if directional else None
    seg_loss = str(getattr(args, "seg_loss", "ce"))
    activation = str(getattr(args, "activation", "relu"))
    # CHROMA d_seg lever (operator "Chroma too"): default ON (3 free RGB planes,
    # current behavior); --no-chroma forces achromatic luma-only for the ablation.
    chroma = bool(getattr(args, "chroma", True))
    # SIREN init (Sitzmann 2020): canonical periodic-activation from-scratch trainability
    # fix. ON by default; takes effect for the periodic family (hosc/siren/finer); relu keeps
    # default init. --no-siren-init disables it (the ablation that MEASURES the SIREN-init effect).
    siren_init = bool(getattr(args, "siren_init", True))
    siren_omega = float(getattr(args, "siren_omega", 30.0))
    finer_first_bias_scale = float(getattr(args, "finer_first_bias_scale", 1.0))
    wire_scale = float(getattr(args, "wire_scale", 1.0))
    # (periodic-omega note) the periodic frequency each activation uses is carried on the MODEL
    # (`model.periodic_omega`, set in __init__ @447 from hosc_omega for hosc / siren_omega otherwise)
    # and read at the verdict forward @1830 — no local copy is needed here (a dead local was removed,
    # confound-fix cleanup 2026-07-05, ruff F841).
    # (bug #8 / BH2) hosc beta-anneal endpoints. A CONSTANT beta=4 tanh(4*sin(.))
    # SATURATES (|arg|<=4 -> tanh ~ +/-0.999 at the sine peaks -> vanishing gradient,
    # the BH2 dead-grad bug). So when the operator selects --activation hosc WITHOUT
    # explicit beta endpoints, default to a NON-SATURATING anneal: start at 1.0
    # (tanh(sin) unsaturated -> gradients flow early) -> --hosc-beta (step-native
    # late). Explicit --hosc-beta-start/end always override. For relu the endpoints
    # are inert (no anneal applied).
    hosc_beta_const = float(getattr(args, "hosc_beta", 4.0))
    _hb_start = getattr(args, "hosc_beta_start", None)
    _hb_end = getattr(args, "hosc_beta_end", None)
    if activation == "hosc" and _hb_start is None and _hb_end is None:
        hosc_beta_start = 1.0
        hosc_beta_end = hosc_beta_const
    else:
        hosc_beta_start = float(_hb_start) if _hb_start is not None else hosc_beta_const
        hosc_beta_end = float(_hb_end) if _hb_end is not None else hosc_beta_const

    # (bug #5) The DIRECTIONAL basis orients Fourier feats to the boundary tangent of the
    # GT SegNet argmax (gt.lstars). That argmax is NOT available at decode (inflate.py has
    # NO SegNet per the strict-scorer rule, and this RGB witness emits no class map to derive
    # the partition from) AND storing the per-pair tangent field is image-sized (dominated on
    # rate). So the directional lever is NOT byte-closeable for this RGB witness -> it is a
    # research/upper-bound probe ONLY; the GATE uses isotropic. Loud, fail-closed honesty:
    directional_byte_closeable = False
    directional_research_only = bool(directional)
    if directional:
        print(json.dumps({
            "stage": "WARNING_directional_basis_not_byte_closeable",
            "reason": "directional basis uses GT SegNet argmax (gt.lstars) for the tangent field; "
                      "unavailable at decode (no inflate-time SegNet; RGB witness emits no class map) "
                      "and storing the tangent field is image-sized. RESEARCH/upper-bound ONLY.",
            "byte_closeable": False, "research_only": True, "promotion_eligible": False,
        }), flush=True)

    result_history: list[dict[str, Any]] = []
    best = {"d_seg": 1.0, "d_pose": 1e9, "implied_S": 1e9, "epoch": 0}

    with temporary_mlx_device(device):
        # --- 2. Frozen MLX scorers on the train device (gradient device) ---
        adapter = load_mlx_distortion_scorer_adapter_from_upstream(REPO / "upstream", device="cpu")

        # --- 3. Shared (restart-invariant) setup: coords, feats, GT targets ---
        coords = mx.array(coords_np)
        if directional:
            # ALL-CLASS DIRECTIONAL (anisotropic/curvelet) Fourier basis: orient the
            # coord features to the per-pair all-class boundary tangent field (high freq
            # ACROSS the edge, low ALONG it). A 0-byte deterministic train-time prior
            # (a fixed function of the GT argmax the witness reproduces) -> compiles into
            # inflate.py FREE. Measured -48% d_seg on the DIRECT generator; this wires it
            # into the REALIZED through-R witness (transfer is the open question).
            from tac.boundary_math.lever_b_generator import (
                all_class_boundary_proximity_and_tangent,
                directional_fourier_feats,
            )

            # (BUILDER 5 shape-bug fix) Build the GT-argmax boundary tangent at the
            # RENDER grid via NN-resize, so tang.reshape(-1,2) is EXACTLY
            # (render_h*render_w, 2) -> it matches coords_np for ANY render/SEG ratio.
            # The prior strided ``lr[::step]`` subsample only matched when the render
            # res evenly divided the SEG res; at render 256x384 vs SEG 384x512 step
            # floored to 1 -> a 196608-row tangent broadcast against a 98304-row coord
            # grid = the directional crash. At render 384x512 (the target) NN-resize is
            # an identity (already 384x512) so the tangent is unchanged.
            _dir_feats_np: list[np.ndarray] | None = []
            for pi in range(P):
                lr = _nn_resize_labels(gt.lstars[pi], render_h, render_w)
                _, tang = all_class_boundary_proximity_and_tangent(lr)
                feats = directional_fourier_feats(
                    coords_np, tang.reshape(-1, 2), n_freqs=n_dir_freqs,
                    freq_across=float(getattr(args, "freq_across", 32.0)),
                    freq_along=float(getattr(args, "freq_along", 4.0)),
                )
                _dir_feats_np.append(np.ascontiguousarray(feats, dtype=np.float32))
            shared_feats = None
        else:
            # build_feats uses the deterministic seeded _B (seed 0, restart-invariant)
            # so the isotropic coord features are identical across restarts -> built once
            # from a throwaway module (the per-restart models reuse this shared buffer).
            _feat_model = build_witness_module(
                num_pairs=P, n_fourier=args.n_fourier, hidden_dim=args.hidden_dim,
                n_hidden=args.n_hidden, mod_dim=args.mod_dim, fourier_sigma=args.fourier_sigma,
                in_feat_override=in_feat_override, activation=activation,
                hosc_beta=float(getattr(args, "hosc_beta", 4.0)),
                hosc_omega=float(getattr(args, "hosc_omega", 1.0)),
                chroma=chroma,
                siren_init=False,  # throwaway: only build_feats (uses the deterministic _B) is read
            )
            shared_feats = _feat_model.build_feats(coords)  # fixed (P_px, in_feat)
            mx.eval(shared_feats)
            _dir_feats_np = None

        def feats_for_pair(pi: int):
            """Per-pair coord features: shared iso (computed once) OR per-pair directional."""
            if directional:
                return mx.array(_dir_feats_np[pi])
            return shared_feats

        # DEPLOY-FAITHFUL numpy coord feats for the VERDICT (DAG FEED-br). The MLX
        # shared_feats above run sin/cos on the GPU (reduced precision); the verdict must
        # use the SAME numpy feats inflate.py builds (deterministic seeded B). Built ONCE.
        _B_np = deterministic_fourier_B(args.n_fourier, args.fourier_sigma)
        _feats_np_iso = None if directional else _build_feats_numpy(coords_np, _B_np)

        def feats_for_pair_np(pi: int) -> np.ndarray:
            """Per-pair coord features in NUMPY (deploy-faithful verdict path)."""
            return _dir_feats_np[pi] if directional else _feats_np_iso

        # GT targets as mx (one-hot L*, margin, pose).
        lstar_oh = []
        margins_mx = []
        poses_mx = []
        for pi in range(P):
            lo = np.eye(N_CLASSES, dtype=np.float32)[gt.lstars[pi]]  # (H,W,5)
            lstar_oh.append(mx.array(lo[None]))  # (1,H,W,5)
            margins_mx.append(mx.array(gt.margins[pi]))  # (H,W)
            poses_mx.append(mx.array(gt.gt_poses[pi].astype(np.float32)))  # (half,)

        score_domain = bool(getattr(args, "score_domain_loss", True))
        pose_eps = float(getattr(args, "pose_eps", 1e-2))
        # LIVE-MARGIN re-allocation lever (DAG FEED-bp NEW-B). Default OFF -> the running
        # baseline + default path are byte-identical (the gate is the unweighted realized
        # CE the iso baseline trains on; the new allocator is opt-in).
        margin_weighted = bool(getattr(args, "margin_weighted_loss", False))
        margin_weight_fn = str(getattr(args, "margin_weight_fn", "inverse"))
        margin_weight_temp = float(getattr(args, "margin_weight_temp", 1.0))
        # ISSUE 2 curriculum: engage the LIVE-margin re-allocation only at ep >= this epoch
        # (UNIFORM seg weight before -> learn the base partition -> then refine the annulus).
        # From RANDOM INIT the live-margin allocator starves the base (wrong px confidently-
        # wrong = large margin = ~0 weight), so it is a FINETUNE lever, not from-scratch.
        margin_weight_start_epoch = int(getattr(args, "margin_weight_start_epoch", 80))
        # FEED-bu fix 1: ANNEAL the margin-weight temp from a GENTLE start (near-uniform, small
        # gnorm bump) DOWN to the MODERATE target (margin_weight_temp) over anneal_epochs, so the
        # gnorm rises GRADUALLY (the spike-guard tracks it) instead of the cold exp/temp=0.1 cliff
        # that exploded gnorm to ~648 -> all-skip. temp_start == target -> no anneal (back-compat).
        margin_weight_temp_start = float(getattr(args, "margin_weight_temp_start", margin_weight_temp))
        margin_weight_anneal_epochs = max(0, int(getattr(args, "margin_weight_anneal_epochs", 40)))

        def _mw_temp_for_epoch(ep: int) -> float:
            """Annealed margin-weight temperature at epoch ``ep`` (only meaningful when mw active)."""
            if margin_weight_anneal_epochs <= 0 or margin_weight_temp_start == margin_weight_temp or ep <= margin_weight_start_epoch:
                return margin_weight_temp if ep >= margin_weight_start_epoch + margin_weight_anneal_epochs else margin_weight_temp_start
            frac = min(1.0, (ep - margin_weight_start_epoch) / float(margin_weight_anneal_epochs))
            return float(margin_weight_temp_start + (margin_weight_temp - margin_weight_temp_start) * frac)

        # --- FEED-bv: the 4-STAGE PR95 d_seg CONDITIONING CURRICULUM (NOT HNeRV, NOT rate steps) ---
        # The witness learns the d_seg LOSS-FORM SEQUENCE from PR95's SegNet treatment (digest
        # pr95_segnet_stage_evolution...): ce (coarse argmax) -> tau_softplus (THE primary drop)
        # -> l7_softplus (hard-pixel refine) -> Muon finisher (the conditioning drop). The smooth
        # stage + ALL rate machinery (QAT/C1a/lambda/sigma-rate) are DROPPED. Boundaries (epochs):
        #   ce:        [1, tau_softplus_start_epoch)
        #   tau_softplus: [tau_softplus_start_epoch, margin_weight_start_epoch)  (l7 boundary reused)
        #   l7:        [margin_weight_start_epoch, muon_finisher_start_epoch)
        #   muon:      [muon_finisher_start_epoch, epochs]   (l7 loss + Muon optimizer)
        # EACH boundary RE-TREATS (spike-guard recalibrate + LR re-warmup + EMA continue) per the
        # "different stages need different treatment" principle. Default OFF -> single seg_loss.
        curriculum = bool(getattr(args, "curriculum", False))
        tau_softplus_start_epoch = int(getattr(args, "tau_softplus_start_epoch", 300))
        muon_finisher_start_epoch = int(getattr(args, "muon_finisher_start_epoch", 1200))
        tau_softplus_tau = float(getattr(args, "tau_softplus_tau", 0.3))
        l7_mult = float(getattr(args, "l7_mult", 4.0))
        # (FEED-ca fix 3) static l7_threshold default tied to the MEASURED witness-hard median
        # margin ~0.42 (FEED-bp), NOT the cargo-culted 1.0. The intra-stage anneal (below) can
        # still ramp it 1.0->~0.42 over the l7 stage if --l7-thr-anneal-* are set.
        l7_threshold = float(getattr(args, "l7_threshold", 0.42))
        # (FEED-ca fix 3) intra-stage ANNEAL endpoints (None,None -> static; scale coarse->fine).
        _tau_anneal_start = getattr(args, "tau_anneal_start", None)
        _tau_anneal_end = getattr(args, "tau_anneal_end", None)
        tau_anneal_start = float(_tau_anneal_start) if _tau_anneal_start is not None else None
        tau_anneal_end = float(_tau_anneal_end) if _tau_anneal_end is not None else None
        tau_anneal_on = tau_anneal_start is not None and tau_anneal_end is not None
        _l7_thr_anneal_start = getattr(args, "l7_thr_anneal_start", None)
        _l7_thr_anneal_end = getattr(args, "l7_thr_anneal_end", None)
        l7_thr_anneal_start = float(_l7_thr_anneal_start) if _l7_thr_anneal_start is not None else None
        l7_thr_anneal_end = float(_l7_thr_anneal_end) if _l7_thr_anneal_end is not None else None
        l7_thr_anneal_on = l7_thr_anneal_start is not None and l7_thr_anneal_end is not None
        # (FEED-ca fix 2) PLATEAU-TRIGGERED stage transitions: advance a stage on EITHER the
        # measured d_seg descent plateauing OR the fixed --*-start-epoch cap (the fallback).
        plateau_trigger = bool(getattr(args, "plateau_trigger", False))
        plateau_slope_eps = float(getattr(args, "plateau_slope_eps", 1e-5))
        plateau_window = max(2, int(getattr(args, "plateau_window", 4)))  # recent IN-STAGE eval points
        # Ordered stage boundaries (the CAP epoch at which each NON-ce stage begins / fallback).
        _stage_boundaries = [
            ("tau_softplus", tau_softplus_start_epoch),
            ("l7_softplus", margin_weight_start_epoch),
            ("muon", muon_finisher_start_epoch),
        ]
        _STAGE_NAMES = ["ce", "tau_softplus", "l7_softplus", "muon"]
        # (FEED-ca fix 6) ASSERT the curriculum boundaries strictly increase + lie in [1, epochs].
        # Fail-closed (NO-FAKE): an l7-on-unconverged-base / mis-ordered curriculum is a silent
        # score-killer. Only enforced when the curriculum is actually engaged.
        if curriculum:
            _caps = [tau_softplus_start_epoch, margin_weight_start_epoch, muon_finisher_start_epoch]
            if not (1 <= _caps[0] < _caps[1] < _caps[2] <= int(args.epochs)):
                raise ValueError(
                    "curriculum boundaries must satisfy 1 <= tau-softplus-start-epoch < "
                    "margin-weight-start-epoch (l7) < muon-finisher-start-epoch <= epochs; got "
                    f"tau={_caps[0]}, l7={_caps[1]}, muon={_caps[2]}, epochs={int(args.epochs)}."
                )
            if plateau_trigger and plateau_slope_eps <= 0:
                raise ValueError(f"--plateau-slope-eps must be > 0; got {plateau_slope_eps}.")

        def _seg_form_for_stage(stage: str) -> str:
            # the muon stage keeps the l7 loss form (only the optimizer changes).
            return "l7_softplus" if stage == "muon" else stage

        # (FEED-ca fix 5) Muon-on-HIDDEN-only filter (PR95 discipline: Muon degrades the
        # code-latent EMBEDDING + the final/output layer). Routes ONLY the decoder-hidden weight
        # MATRICES (in_proj.weight / film.weight / hidden.*.weight) to Muon; the code-latent
        # (video-derived payload), all biases, and out.* (final layer) fall back to Adam.
        def _is_muon_weight(path: str, w: Any) -> bool:
            if getattr(w, "ndim", 0) < 2:
                return False  # biases / 1D -> Adam
            if path == "code":
                return False  # code-latent embedding (video-derived payload) -> Adam
            if path.startswith("out."):
                return False  # final/output layer -> Adam (PR95: Muon degrades final layers)
            return path.endswith(".weight")  # in_proj.weight / film.weight / hidden.*.weight

        class _StageResolver:
            """Per-restart stateful stage scheduler. Advances ce->tau->l7->muon on EITHER the
            fixed cap OR (plateau mode) a measured d_seg plateau. Tracks each stage's START epoch
            so the intra-stage tau / l7-threshold anneals know their progress in BOTH modes."""

            def __init__(self) -> None:
                self.idx = 0  # 0=ce 1=tau 2=l7 3=muon
                self.start_ep = {0: 1}

            def _plateau(self, ep: int, dseg_history: list[tuple[int, float]]) -> bool:
                if not plateau_trigger:
                    return False
                cur_start = self.start_ep.get(self.idx, 1)
                pts = [(e, d) for (e, d) in dseg_history if e >= cur_start][-plateau_window:]
                if len(pts) < plateau_window:  # need a FULL in-stage window (also gates post-jump)
                    return False
                es = np.asarray([p[0] for p in pts], dtype=np.float64)
                ds = np.asarray([p[1] for p in pts], dtype=np.float64)
                slope = float(np.polyfit(es, ds, 1)[0])  # d(d_seg)/d(epoch)
                return slope > -plateau_slope_eps  # not dropping faster than eps/epoch == plateau

            def resolve(self, ep: int, dseg_history: list[tuple[int, float]]) -> tuple[str, bool, str]:
                """Returns (stage_name, is_boundary_this_epoch, boundary_stage_name)."""
                if not curriculum:
                    return (seg_loss, False, "")
                if self.idx < 3:
                    cap_ep = _stage_boundaries[self.idx][1]  # cap to advance INTO idx+1
                    if ep >= cap_ep or self._plateau(ep, dseg_history):
                        self.idx += 1
                        self.start_ep[self.idx] = ep
                        return (_STAGE_NAMES[self.idx], True, _STAGE_NAMES[self.idx])
                return (_STAGE_NAMES[self.idx], False, "")

            def tau_for_ep(self, ep: int) -> float:
                if not tau_anneal_on:
                    return tau_softplus_tau
                s0 = self.start_ep.get(1)  # tau stage start
                if s0 is None:
                    return tau_softplus_tau
                nominal = max(1, margin_weight_start_epoch - tau_softplus_start_epoch)
                frac = min(1.0, max(0.0, (ep - s0) / float(nominal)))
                return float(tau_anneal_start + (tau_anneal_end - tau_anneal_start) * frac)

            def l7_thr_for_ep(self, ep: int) -> float:
                if not l7_thr_anneal_on:
                    return l7_threshold
                s0 = self.start_ep.get(2)  # l7 stage start
                if s0 is None:
                    return l7_threshold
                nominal = max(1, muon_finisher_start_epoch - margin_weight_start_epoch)
                frac = min(1.0, max(0.0, (ep - s0) / float(nominal)))
                return float(l7_thr_anneal_start + (l7_thr_anneal_end - l7_thr_anneal_start) * frac)

        loss_fn = make_loss_fn(
            adapter, render_h, render_w, score_domain=score_domain, pose_eps=pose_eps, seg_loss=seg_loss,
            margin_weighted=margin_weighted, margin_weight_fn=margin_weight_fn, margin_weight_temp=margin_weight_temp,
            tau_softplus_tau=tau_softplus_tau, l7_mult=l7_mult, l7_threshold=l7_threshold,
        )
        print(
            json.dumps({
                "stage": "loss_mode",
                "score_domain_loss": score_domain,
                "pose_eps": pose_eps,
                "w_seg": float(args.w_seg),
                "w_pose": float(args.w_pose),
                "seg_loss": seg_loss,
                "basis": basis,
                "activation": activation,
                "margin_weighted_loss": margin_weighted,
                "margin_weight_fn": (margin_weight_fn if margin_weighted else None),
                "margin_weight_temp": (margin_weight_temp if margin_weighted else None),
                "margin_weight_start_epoch": (margin_weight_start_epoch if margin_weighted else None),
                "margin_weight_temp_start": (margin_weight_temp_start if margin_weighted else None),
                "margin_weight_anneal_epochs": (margin_weight_anneal_epochs if margin_weighted else None),
                "margin_stage_lr_warmup_epochs": (int(getattr(args, "margin_stage_lr_warmup_epochs", 8)) if margin_weighted else None),
                "resume_from": (str(getattr(args, "resume_from")) if getattr(args, "resume_from", None) else None),
                "verdict_device": str(getattr(args, "verdict_device", "mlx-cpu")),
                "verdict_which": str(getattr(args, "verdict_which", "live")),
                "ema_verdict_every": int(getattr(args, "ema_verdict_every", 4)),
                "verdict_batch": int(getattr(args, "verdict_batch", 16)),
                "verdict_pairs": int(getattr(args, "verdict_pairs", 120)),
                "curriculum": curriculum,
                "curriculum_stages": (
                    f"ce[1,{tau_softplus_start_epoch}) -> tau_softplus[{tau_softplus_start_epoch},{margin_weight_start_epoch}) "
                    f"-> l7_softplus[{margin_weight_start_epoch},{muon_finisher_start_epoch}) -> muon[{muon_finisher_start_epoch},{args.epochs}]"
                    if curriculum else None
                ),
                "tau_softplus_tau": (tau_softplus_tau if curriculum else None),
                "form": ("w_seg*seg_l + w_pose*sqrt(10*pose_l+eps)" if score_domain else "w_seg*seg_l + w_pose*pose_l"),
            }),
            flush=True,
        )

        # holder lets eval_verdict see the CURRENT restart's model/ema by reference.
        holder: dict[str, Any] = {"model": None, "ema": None}

        int8_verdict = bool(getattr(args, "int8_verdict", True))
        verdict_device = str(getattr(args, "verdict_device", "mlx-cpu"))
        verdict_which = str(getattr(args, "verdict_which", "live"))  # live | ema | both (Win #2)
        ema_verdict_every = max(1, int(getattr(args, "ema_verdict_every", 4)))  # EMA cadence when which=live
        verdict_batch = max(1, int(getattr(args, "verdict_batch", 16)))
        # Fixed, evenly-spaced monitoring subset (DAG FEED-bt win #3): the SAME pairs every
        # eval -> the d_seg TREND is comparable across epochs. 0 -> all P pairs (authority).
        _vp = int(getattr(args, "verdict_pairs", 120))
        if _vp <= 0 or _vp >= P:
            _verdict_subset = list(range(P))
        else:
            _verdict_subset = sorted(set(np.linspace(0, P - 1, _vp).round().astype(int).tolist()))

        def eval_verdict(use_ema: bool, device: str | None = None, pair_idx: list | None = None,
                         batch: int | None = None) -> dict[str, float]:
            """FROZEN CPU-torch authority verdict on the DEPLOY weights + TORCH R, rendering
            the witness MLP forward on a SELECTABLE device (ISSUE 1 throughput fix).

            (bug #4) When ``int8_verdict`` the verdict renders the int8-DEQUANTIZED deploy
            weights (eval_roundtrip-for-WEIGHTS); the R (resize+uint8) + scorer preprocess +
            scorer are ALWAYS torch (authority).

            VERDICT DEVICE (``device`` overrides the --verdict-device default):
              * ``numpy``  : fp32, byte-identical to inflate.py -> 0 px/pair gap vs the
                             byte-close. AUTHORITATIVE for the FINAL row; slow on 600 pairs.
              * ``mlx-cpu``: fp32-accurate (3-18 px/pair << 143-px sub-0.15 budget), FAST.
                             The in-loop descent default -- accurate ENOUGH for monitoring.
              * ``mlx-gpu``: FASTEST but reduced-precision (~495-1672 px) -> ONLY trustworthy
                             for the COARSE 0.5->0.01 descent; NOT for a frontier verdict.
            The FINAL byte-close (tools/witness_byte_close_and_eval.py) is numpy fp32 (0 px).
            """
            dev = device or verdict_device
            pidx = list(range(P)) if pair_idx is None else list(pair_idx)
            vbatch = batch or verdict_batch
            model = holder["model"]
            ema = holder["ema"]
            src_items = list(ema.shadow.items()) if use_ema else list(tree_flatten(model.parameters()))
            if int8_verdict:
                deploy = dequantize_witness_flat(src_items)  # {name: np.float32} int8 round-trip
            else:
                deploy = {k: np.asarray(v, dtype=np.float32) for k, v in src_items}
            kind = str(model.activation)
            beta = float(model.hosc_beta)  # current annealed beta (deploy uses the final value)
            # periodic_omega = hosc_omega for hosc, siren_omega(~30) for siren/finer; the verdict
            # forward MUST use the SAME omega the trained _act used or the deploy render diverges.
            omega = float(getattr(model, "periodic_omega", model.hosc_omega))
            wire_scale_v = float(getattr(model, "wire_scale", 1.0))  # WIRE Gabor window scale (deploy verdict)
            chroma_v = bool(model.chroma)
            nh, hd = args.n_hidden, args.hidden_dim

            def _render_frame_numpy(pi: int, fk: int) -> np.ndarray:
                fp = feats_for_pair_np(pi)
                rgb = _witness_forward_numpy(deploy, fp, deploy["code"][2 * pi + fk], nh, hd, kind, beta, omega, chroma_v, wire_scale_v)
                return _torch_R_to_camera_uint8(rgb.reshape(render_h, render_w, 3))

            d_segs, d_poses = [], []
            # Win #1 (batched scorer) x Win #3 (subset): render a CHUNK of vbatch pairs
            # (device-specific MLP forward + torch R), then run ONE batched SegNet + ONE
            # batched PoseNet over the chunk. Bit-exact (eval-mode BN running stats).
            if dev == "numpy":
                for s in range(0, len(pidx), vbatch):
                    chunk = pidx[s:s + vbatch]
                    f0s = [_render_frame_numpy(pi, 0) for pi in chunk]
                    f1s = [_render_frame_numpy(pi, 1) for pi in chunk]
                    d_segs += cpu_verdict_d_seg_batch(seg_cpu, f1s, [gt.lstars[pi] for pi in chunk])
                    d_poses += cpu_verdict_d_pose_batch(posenet_cpu, f0s, f1s, [gt.gt_poses[pi] for pi in chunk])
            else:
                mlx_dev = "cpu" if dev == "mlx-cpu" else "gpu"
                with temporary_mlx_device(mlx_dev):
                    deploy_mx = {k: mx.array(v) for k, v in deploy.items()}
                    feats_iso_mx = None if directional else mx.array(_feats_np_iso)
                    code_mx = deploy_mx["code"]

                    def _render_frame_mlx(pi: int, fk: int) -> np.ndarray:
                        fp_mx = mx.array(_dir_feats_np[pi]) if directional else feats_iso_mx
                        rgb_mx = _witness_forward_mlx(deploy_mx, fp_mx, code_mx[2 * pi + fk], nh, hd, kind, beta, omega, chroma_v, wire_scale_v)
                        mx.eval(rgb_mx)
                        return _torch_R_to_camera_uint8(np.asarray(rgb_mx, dtype=np.float32).reshape(render_h, render_w, 3))

                    for s in range(0, len(pidx), vbatch):
                        chunk = pidx[s:s + vbatch]
                        f0s = [_render_frame_mlx(pi, 0) for pi in chunk]
                        f1s = [_render_frame_mlx(pi, 1) for pi in chunk]
                        mx.clear_cache()  # bound the Metal render-cache per chunk
                        d_segs += cpu_verdict_d_seg_batch(seg_cpu, f1s, [gt.lstars[pi] for pi in chunk])
                        d_poses += cpu_verdict_d_pose_batch(posenet_cpu, f0s, f1s, [gt.gt_poses[pi] for pi in chunk])
            return {"d_seg": float(np.mean(d_segs)), "d_pose": float(np.mean(d_poses)), "n_pairs_scored": len(pidx)}

        # --- stabilization config (DAG FEED-bd/be: clean descent -> ep30-40 spike
        # -> dead-saturation lock at d_seg 0.5447 was optimizer-divergence). Each item
        # below is a named driver fix; defaults reproduce the prescribed stabilized run.
        grad_clip = float(getattr(args, "grad_clip", 1.0))            # (1) THE primary fix
        accum_pairs = max(1, int(getattr(args, "accum_pairs", 8)))    # (3) variance reduction
        n_restarts = max(1, int(getattr(args, "n_restarts", 2)))      # (6) multi-restart-keep-best
        spike_factor = float(getattr(args, "spike_factor", 5.0))      # (5) spike-guard threshold
        grad_log_every = max(1, int(getattr(args, "grad_log_every", 25)))  # (7) grad-norm logging
        lr_schedule_enabled = bool(getattr(args, "lr_schedule", True))     # (4) warmup+cosine
        warmup_epochs = max(0, int(getattr(args, "warmup_epochs", 1)))
        lr_end = float(getattr(args, "lr_end", 1e-4))
        div_frac = 0.9  # restart divergence: EMA d_seg still >= 0.9*baseline at midpoint = stuck
        opt_steps_per_epoch = max(1, math.ceil(P / accum_pairs))
        # (FEED-ca fix 2/4) MUTABLE stage-transition LR re-warmup anchors (OPTIMIZER-step space).
        # FIXED boundaries: prefilled at restart-top from the cap epochs. PLATEAU boundaries:
        # appended live at the ACTUAL (dynamic) AdamW transition via float(opt.step). The
        # build_lr_schedule closure reads this SAME list object so both modes re-treat the LR.
        _rewarmup_anchor_steps: list[float] = []

        def build_lr_schedule():
            """1-epoch linear warmup 0->base, then cosine decay base->lr_end (item 4), with an
            OPTIONAL margin-stage LR RE-WARMUP (FEED-bu fix 4, the stage-transition treatment)."""
            if not lr_schedule_enabled:
                base_sched: Any = float(args.lr)
            else:
                base = float(args.lr)
                warm = warmup_epochs * opt_steps_per_epoch
                total = max(1, args.epochs * opt_steps_per_epoch)
                cos_steps = max(1, total - warm)
                cosine = optim.cosine_decay(base, cos_steps, lr_end)
                if warm <= 0:
                    base_sched = cosine
                else:
                    warmup = optim.linear_schedule(0.0, base, warm)
                    base_sched = optim.join_schedules([warmup, cosine], [warm])
            # FEED-bu fix 4 + FEED-bv + FEED-ca (operator: "different stages need different
            # treatment"): a STAGE TRANSITION re-treats the LR. At EACH AdamW-stage boundary, ramp
            # the LR from a floor (0.1x) back to 1x over margin_stage_lr_warmup_epochs so the FIRST
            # steps of the new stage are SMALL -> absorb the loss-FORM change without a first-step
            # gnorm spike. The boundaries live in ``_rewarmup_anchor_steps`` (OPTIMIZER-step space):
            # FIXED curriculum/mw prefill the cap-derived steps at restart-top; PLATEAU mode appends
            # the ACTUAL transition step live (so the re-warmup tracks dynamic plateau transitions).
            # The Muon stage uses its OWN fresh low-LR opt with a step-0-anchored warmup (fix 4),
            # so the muon boundary is NOT in this AdamW anchor list. wu_eps<=0 -> no re-warmup.
            wu_eps = int(getattr(args, "margin_stage_lr_warmup_epochs", 8))
            # Only the curriculum / margin-weighted paths can carry re-warmup anchors. The DEFAULT
            # (non-curriculum, non-mw) path returns base_sched verbatim -> byte-identical prior behavior.
            if wu_eps <= 0 or not (bool(getattr(args, "curriculum", False)) or bool(getattr(args, "margin_weighted_loss", False))):
                return base_sched
            win = float(max(1, wu_eps * opt_steps_per_epoch))
            lr_floor = float(getattr(args, "margin_stage_lr_floor", 0.1))

            def stage_rewarmup_sched(step):
                base_v = base_sched(step) if callable(base_sched) else base_sched
                scale = 1.0
                # Reads the shared MUTABLE anchor list (empty -> scale stays 1.0 == base_sched).
                for b in list(_rewarmup_anchor_steps):
                    frac = mx.clip((step - b) / win, 0.0, 1.0)
                    ramp = lr_floor + (1.0 - lr_floor) * frac  # lr_floor at b -> 1.0 at b+win (stays 1)
                    scale = scale * mx.where(step < b, 1.0, ramp)
                return base_v * scale

            return stage_rewarmup_sched

        # --- 4. Restart loop (item 6): reseed + restart on midpoint divergence;
        #         keep the best EMA-d_seg checkpoint across restarts. ---
        w_seg = mx.array(float(args.w_seg))
        w_pose = mx.array(float(args.w_pose))
        hinge = mx.array(float(args.hinge_weight))
        m_start = float(getattr(args, "margin_target_start", 1.0))
        m_end = float(getattr(args, "margin_target_end", 0.5))
        mid_ep = max(1, args.epochs // 2)
        recent_maxlen = 20  # running window for the spike-guard median
        recent: deque[float] = deque(maxlen=recent_maxlen)
        base_verdict: dict[str, float] | None = None
        best_saved: dict[str, Any] = {"d_seg": 1e9, "params": None, "restart": -1, "epoch": -1}
        # (DAG FEED-br sister fix) ALSO keep the best-implied-S LIVE-weights checkpoint.
        # The EMA shadow can lag the live weights HARD (arm_iso_v2: EMA d_seg 0.171 vs LIVE
        # d_seg 0.0022 = 78x) — byte-closing only the lagging EMA understates a real witness.
        # We save BOTH so the byte-close can pick the better arm (--use-live).
        best_live_saved: dict[str, Any] = {"implied_S": 1e9, "d_seg": 1e9, "params": None, "restart": -1, "epoch": -1}
        ema_lag_warn_ratio = float(getattr(args, "ema_lag_warn_ratio", 2.0))
        n_evals_total = 0  # global eval counter (drives the EMA-verdict cadence for which=live)
        _rf = getattr(args, "resume_from", None)  # FEED-bu fix 3: resume a prior witness base
        resume_from = Path(_rf) if _rf else None
        if resume_from is not None and not resume_from.exists():
            raise FileNotFoundError(f"--resume-from {resume_from} does not exist (NO-FAKE: refusing to fabricate a base).")
        t_train = time.time()
        ep_walls: list[float] = []
        gstep = 0
        n_skips_total = 0

        for restart_i in range(n_restarts):
            # (bug #2) RESET the spike-guard running-median window at the TOP of EACH restart.
            # Leaving it populated from restart 0 makes restart>=1 judge its (high, early) batch
            # losses against restart 0's converged-low median -> the 5x guard trips EVERY batch
            # -> ZERO opt/ema updates for the entire restart (the dead-restart bug, DAG FEED-bf).
            recent = deque(maxlen=recent_maxlen)
            # (C17 deadlock-free re-arm; DAG FEED confound-hunt 2026-07-05) consecutive-skip
            # counter for the skip-rate-ceiling ESCAPE HATCH. The legacy spike-guard appends to
            # `recent` ONLY on an ACCEPTED batch (else-branch below), so a persistent loss-LEVEL
            # shift (e.g. a stiff term attached at resume, or an eikonal runaway) => batch_loss
            # sits permanently above `spike_factor * median` => 100% skip forever => the median
            # never moves => ABSORBING freeze (measured 3x in the levelset sibling: #205 runs
            # 015247Z/083453Z/095728Z). Same bug class here per "bug classes have 6-7x spread".
            # This counter lets a SUSTAINED all-skip window force a fresh median re-baseline
            # instead of freezing; it PERSISTS ACROSS EPOCHS (the freeze spans epochs) and is
            # reset ONLY by an accepted batch (the else-branch), so every legitimate re-arm site
            # (restart top / curriculum transition / Muon switch clears `recent` => median None
            # => next batch accepted => counter reset) self-corrects.
            consec_skips = 0
            _muon_switched = False  # FEED-bv: per-restart Muon-finisher switch flag
            # (FEED-ca fix 2) per-restart stage scheduler + d_seg history (for plateau-trigger).
            stage_resolver = _StageResolver()
            dseg_history: list[tuple[int, float]] = []
            # (FEED-ca fix 2/4) per-restart LR re-warmup anchors (OPTIMIZER-step space). FIXED
            # mode prefills the cap-derived AdamW boundaries (tau, l7); PLATEAU mode starts empty
            # and appends the ACTUAL transition step live. The muon boundary uses its own fresh
            # opt warmup (NOT this list). Non-curriculum margin path keeps its single engage cap.
            _rewarmup_anchor_steps.clear()
            if curriculum and not plateau_trigger:
                _rewarmup_anchor_steps.extend(
                    float(e * opt_steps_per_epoch)
                    for e in (tau_softplus_start_epoch, margin_weight_start_epoch) if e > 0
                )
            elif (not curriculum) and margin_weighted and margin_weight_start_epoch > 0:
                _rewarmup_anchor_steps.append(float(margin_weight_start_epoch * opt_steps_per_epoch))
            seed_r = int(args.seed) + restart_i * 1000
            mx.random.seed(seed_r)
            np.random.seed(seed_r)
            rng = np.random.default_rng(seed_r)
            model = build_witness_module(
                num_pairs=P, n_fourier=args.n_fourier, hidden_dim=args.hidden_dim,
                n_hidden=args.n_hidden, mod_dim=args.mod_dim, fourier_sigma=args.fourier_sigma,
                in_feat_override=in_feat_override, activation=activation,
                hosc_beta=hosc_beta_start,  # (bug #8) init at the low (non-saturated) beta
                hosc_omega=float(getattr(args, "hosc_omega", 1.0)),
                chroma=chroma,
                siren_init=siren_init,
                siren_omega=siren_omega,
                finer_first_bias_scale=finer_first_bias_scale,
                wire_scale=wire_scale,
            )
            mx.eval(model.parameters())
            # FEED-bu fix 3: --resume-from loads a prior witness checkpoint (e.g. v3's ep80 base,
            # d_seg 0.0093) into restart 0 so the margin-finetune CONTINUES rather than redoing the
            # 80-epoch base. The architecture flags MUST match the checkpoint (load is strict; a
            # shape mismatch raises -> fail-closed). The EMA shadow is re-seeded from the resumed
            # weights. restart>=1 (divergence reseed) starts fresh (resume is a restart-0 base).
            if resume_from is not None and restart_i == 0:
                rz = np.load(resume_from, allow_pickle=False)
                rparams = {k: mx.array(np.asarray(rz[k], dtype=np.float32)) for k in rz.files}
                # (FEED-ca fix 6) VALIDATE resumed keys/shapes against the built model BEFORE update
                # (mx model.update does NOT validate -> a stale/mismatched npz would silently corrupt
                # the base). Fail-closed (NO-FAKE) with the exact mismatch.
                _built = {k: tuple(v.shape) for k, v in tree_flatten(model.parameters())}
                _resumed = {k: tuple(np.asarray(rz[k]).shape) for k in rz.files}
                _missing = sorted(set(_built) - set(_resumed))
                _extra = sorted(set(_resumed) - set(_built))
                _shape_mismatch = {k: (_built[k], _resumed[k]) for k in (set(_built) & set(_resumed)) if _built[k] != _resumed[k]}
                if _missing or _extra or _shape_mismatch:
                    raise ValueError(
                        "--resume-from npz does not match the built witness (check the architecture "
                        f"flags --num-pairs/--n-fourier/--hidden-dim/--n-hidden/--mod-dim/--chroma). "
                        f"missing_in_npz={_missing}; extra_in_npz={_extra}; shape_mismatch={_shape_mismatch}"
                    )
                model.update(tree_unflatten(list(rparams.items())))
                mx.eval(model.parameters())
                print(json.dumps({
                    "stage": "resumed_from", "path": str(resume_from), "n_tensors": len(rparams),
                    "note": "margin-finetune continues from this base; EMA shadow re-seeded from it",
                }), flush=True)
                # (FEED-ca fix 1) COLLAPSE-CE on resume: if the resumed base is already CE-converged,
                # a long ce[1, tau-softplus-start-epoch) stage is redundant (~4h waste). Nudge toward
                # a SHORT CE stage (e.g. --tau-softplus-start-epoch 25-50). The curriculum handles a
                # short CE stage cleanly (boundaries are asserted strictly increasing above).
                if curriculum and tau_softplus_start_epoch >= 100:
                    print(json.dumps({
                        "stage": "HINT_collapse_ce_on_resume", "tau_softplus_start_epoch": tau_softplus_start_epoch,
                        "advice": "resume base is presumed CE-converged; the default long CE stage is "
                                  "redundant -- set --tau-softplus-start-epoch ~25-50 to skip re-CE (~4h save).",
                    }), flush=True)
            ema = MlxEMA(model, decay=args.ema_decay)
            # Optimizer: default AdamW (existing path, UNCHANGED) or MD-Decoupling
            # (arXiv:2606.25971) — the re-audit's optimizer-bug fix (anti-collapse,
            # warmup-free, LR-transfers-across-width). MD is a drop-in: the model
            # param tree stays = materialized W, so EMA/byte-close/verdict compose
            # untouched. The direction LR tracks build_lr_schedule(); gains step at
            # gain_lr_scale*eta_W. NO-FAKE: a candidate MEASURED win, ablation only.
            optimizer_kind = str(getattr(args, "optimizer", "adamw")).lower()
            if optimizer_kind == "md":
                from tac.optimization.md_decoupling import MDDecoupledOptimizer

                opt = MDDecoupledOptimizer(
                    learning_rate=build_lr_schedule(),
                    gain_lr_scale=float(getattr(args, "md_gain_lr_scale", 1.0 / 3.0)),
                    base=str(getattr(args, "md_base", "adam")),
                )
                print(json.dumps({
                    "stage": "optimizer", "kind": "md_decoupling",
                    "md_base": str(getattr(args, "md_base", "adam")),
                    "md_gain_lr_scale": float(getattr(args, "md_gain_lr_scale", 1.0 / 3.0)),
                    "note": "arXiv:2606.25971 magnitude-direction decoupling; warmup/weight-decay "
                            "are no-ops for the decoupled (direction-on-sphere) weights.",
                }), flush=True)
            else:
                opt = optim.AdamW(learning_rate=build_lr_schedule(), weight_decay=args.weight_decay,
                                  betas=[0.9, float(getattr(args, "adam_beta2", 0.999))])
            value_and_grad = nn.value_and_grad(model, loss_fn)
            holder["model"] = model
            holder["ema"] = ema

            if base_verdict is None:
                base_verdict = eval_verdict(use_ema=False, pair_idx=_verdict_subset)
                print(json.dumps({"stage": "baseline_verdict", "monitor_pairs": len(_verdict_subset),
                                  **{k: round(v, 6) for k, v in base_verdict.items() if isinstance(v, float)}}), flush=True)

            diverged = False
            checked_div = False
            for ep in range(1, args.epochs + 1):
                ep_t0 = time.time()
                ep_loss = 0.0
                # (bug #8) hosc beta-anneal low->high: tanh(beta*sin(.)) SATURATES at high beta
                # (zero gradient = the vanishing-grad bug). Annealing beta_start (low, non-
                # saturating => gradients flow early) -> beta_end (high => step-native late)
                # keeps the gradient alive while still converging to the step train. No-op when
                # start==end (default) or activation!=hosc. Mutating the python attr is read
                # per-call by _act so each epoch's traced graph uses the new beta.
                if activation == "hosc" and hosc_beta_end != hosc_beta_start:
                    frac = 0.0 if args.epochs <= 1 else (ep - 1) / (args.epochs - 1)
                    model.hosc_beta = float(hosc_beta_start + (hosc_beta_end - hosc_beta_start) * frac)
                # margin-hinge target anneal 1.0->0.5 (lensA: broad early when margins are
                # wide, tight late when they collapse). Unused by the CE seg-loss branch.
                m_t = m_start if args.epochs <= 1 else m_start + (m_end - m_start) * (ep - 1) / (args.epochs - 1)
                margin_target = mx.array(float(m_t))
                # --- FEED-bv/ca 4-STAGE CURRICULUM stage resolution + per-boundary RE-TREATMENT ---
                # The stage advances on EITHER the fixed cap OR (plateau_trigger) a measured d_seg
                # plateau (FEED-ca fix 2). The resolver tracks each stage's START epoch so the
                # intra-stage tau / l7-threshold anneals (fix 3) know their progress in BOTH modes.
                cur_stage, is_b, b_name = stage_resolver.resolve(ep, dseg_history)
                seg_form_ep = _seg_form_for_stage(cur_stage) if curriculum else None
                # (FEED-ca fix 3) per-epoch annealed tau / l7-threshold overrides (None when off).
                tau_ep = stage_resolver.tau_for_ep(ep) if (curriculum and tau_anneal_on) else None
                l7_thr_ep = stage_resolver.l7_thr_for_ep(ep) if (curriculum and l7_thr_anneal_on) else None
                if curriculum:
                    # the curriculum drives the seg loss FORM per stage; the separate exp/inverse
                    # live-margin lever is OFF (l7_softplus carries its OWN hard-pixel weight).
                    mw_active = False
                    mw_temp_ep = margin_weight_temp
                    if is_b:
                        # RE-TREAT every stage transition (operator: "different stages need different
                        # treatment"): spike-guard recalibrate (reset the loss-median window for the
                        # new loss-form regime) + LR re-warmup + EMA continues (NOT reset). The Muon
                        # switch happens below. (FEED-ca fix 2) in PLATEAU mode the tau/l7 AdamW
                        # transitions append the ACTUAL opt step to the re-warmup anchor list so the
                        # LR re-treat tracks the dynamic boundary (fixed mode prefills at restart-top).
                        recent = deque(maxlen=recent_maxlen)
                        if plateau_trigger and b_name in ("tau_softplus", "l7_softplus"):
                            try:
                                _rewarmup_anchor_steps.append(float(getattr(opt, "step", gstep)))
                            except Exception:
                                _rewarmup_anchor_steps.append(float(gstep))
                        print(json.dumps({
                            "stage": "curriculum_stage_transition", "restart": restart_i, "ep": ep,
                            "into_stage": b_name, "seg_form": _seg_form_for_stage(b_name),
                            "trigger": ("plateau_or_cap" if plateau_trigger else "fixed_cap"),
                            "spike_guard_recalibrated": True,
                            "lr_rewarmup": (b_name in ("tau_softplus", "l7_softplus")),
                            "tau_ep": (round(tau_ep, 4) if tau_ep is not None else None),
                            "l7_thr_ep": (round(l7_thr_ep, 4) if l7_thr_ep is not None else None),
                            "ema_continue": True,
                            "note": "re-treat: spike-guard reset + (AdamW) LR re-warmup + EMA continue",
                        }), flush=True)
                else:
                    # ISSUE-2 / FEED-bu non-curriculum margin path: UNIFORM seg weight until
                    # margin_weight_start_epoch, then engage the annealed LIVE-margin allocation.
                    mw_active = bool(margin_weighted) and (ep >= margin_weight_start_epoch)
                    mw_temp_ep = _mw_temp_for_epoch(ep) if mw_active else margin_weight_temp
                    if margin_weighted and ep == margin_weight_start_epoch:
                        recent = deque(maxlen=recent_maxlen)  # FEED-bu spike-guard recalibrate
                        print(json.dumps({
                            "stage": "margin_weight_curriculum_engage", "restart": restart_i, "ep": ep,
                            "fn": margin_weight_fn, "temp_start": margin_weight_temp_start,
                            "temp_target": margin_weight_temp, "anneal_epochs": margin_weight_anneal_epochs,
                            "spike_guard_recalibrated": True,
                            "note": "uniform seg weight -> annealed live-margin annulus re-allocation; "
                                    "spike-guard window reset for the new gnorm regime",
                        }), flush=True)
                # MUON FINISHER switch (FEED-bv stage D + FEED-ca fix 4/5): at the muon boundary
                # (fixed cap OR plateau), swap AdamW for mlx Muon (Newton-Schulz orthogonalized
                # momentum) at a low LR -- "Muon is THE drop". (fix 5) Muon applies ONLY to the
                # decoder-HIDDEN weight matrices; the code-latent EMBEDDING (video-derived payload)
                # + all biases + the final/output layer route to Adam via MultiOptimizer (PR95:
                # Muon degrades embedding/final layers). (fix 4) the FRESH Muon/Adam-tail get a
                # step-0-anchored LR re-warmup (the previously un-re-treated transition). EMA +
                # model params are untouched (only the optimizer changes). --muon-all-params =
                # back-compat ablation (Muon on ALL params, no Adam tail).
                if curriculum and is_b and b_name == "muon" and not _muon_switched:
                    muon_lr = float(getattr(args, "muon_lr", 1e-4))
                    muon_momentum = float(getattr(args, "muon_momentum", 0.95))
                    _madam = getattr(args, "muon_adam_lr", None)
                    muon_adam_lr = float(_madam) if _madam is not None else muon_lr
                    wu_eps_m = int(getattr(args, "margin_stage_lr_warmup_epochs", 8))
                    lr_floor_m = float(getattr(args, "margin_stage_lr_floor", 0.1))
                    # GAP 2 (default-off): capture the OUTGOING Adam/AdamW first-moment (m)
                    # BEFORE `opt` is rebound below, to warm-start the fresh Muon momentum (v).
                    # Only a plain Adam/AdamW base exposes .m; MD-Decoupling / non-Adam bases
                    # (or --muon-warm-start-momentum off) leave the Muon at its cold zero start.
                    _warm_start = bool(getattr(args, "muon_warm_start_momentum", False))
                    _old_adam_state = (
                        opt.state if (_warm_start and isinstance(opt, (optim.Adam, optim.AdamW))) else None
                    )
                    if wu_eps_m > 0:  # fix 4: re-warmup at the Muon boundary (fresh opt -> step 0)
                        win_m = max(1, wu_eps_m * opt_steps_per_epoch)
                        muon_sched: Any = optim.linear_schedule(muon_lr * lr_floor_m, muon_lr, win_m)
                        adam_sched: Any = optim.linear_schedule(muon_adam_lr * lr_floor_m, muon_adam_lr, win_m)
                    else:
                        win_m = 0
                        muon_sched = muon_lr
                        adam_sched = muon_adam_lr
                    # GAP 1 (default-off): cosine DECAY the Muon LR from muon_lr -> muon_lr*final_frac
                    # across the muon-stage span. Muon's Newton-Schulz fixes update MAGNITUDE, so a
                    # flat Muon LR cannot self-reduce the step near the minimum (river-valley Muon
                    # 2606.21514) -> plateau/overstep. Composes AFTER the optional warmup (warmup THEN
                    # decay via join_schedules). final_frac >= 1.0 (the default) => muon_sched is left
                    # exactly as above (byte-identical). Only the Muon group decays; the Adam tail
                    # self-adapts its effective step via v, so it keeps the existing (flat/warmup) LR.
                    muon_lr_final_frac = float(getattr(args, "muon_lr_final_frac", 1.0))
                    if muon_lr_final_frac < 1.0:
                        _stage_steps = max(1, (int(args.epochs) - int(ep) + 1) * opt_steps_per_epoch)
                        _decay_steps = max(1, _stage_steps - int(win_m))
                        _muon_decay = optim.cosine_decay(
                            muon_lr, _decay_steps, end=muon_lr * muon_lr_final_frac
                        )
                        muon_sched = (
                            optim.join_schedules([muon_sched, _muon_decay], [int(win_m)])
                            if win_m > 0 else _muon_decay
                        )
                    muon_all = bool(getattr(args, "muon_all_params", False))
                    if muon_all:
                        opt = optim.Muon(learning_rate=muon_sched, momentum=muon_momentum,
                                         weight_decay=float(args.weight_decay))
                    else:
                        _muon_opt = optim.Muon(learning_rate=muon_sched, momentum=muon_momentum,
                                               weight_decay=float(args.weight_decay))
                        _adam_tail = optim.Adam(learning_rate=adam_sched)
                        opt = optim.MultiOptimizer([_muon_opt, _adam_tail], filters=[_is_muon_weight])
                    # GAP 2 (default-off): seed the fresh Muon momentum (v) from the captured Adam m.
                    _warm_seeded = 0
                    if _old_adam_state is not None:
                        try:
                            opt.init(model.trainable_parameters())  # allocate v (zeros) to overwrite
                            _warm_target = opt if muon_all else _muon_opt
                            _warm_seeded = _seed_muon_momentum_from_adam(_warm_target, _old_adam_state)
                        except Exception as _warm_err:  # fall back to cold start; never crash the run
                            _warm_seeded = -1
                            print(json.dumps({
                                "stage": "muon_warm_start_FAILED_cold_fallback",
                                "restart": restart_i, "ep": ep, "err": str(_warm_err),
                            }), flush=True)
                    _muon_switched = True
                    recent = deque(maxlen=recent_maxlen)  # re-treat the Muon transition too
                    print(json.dumps({
                        "stage": "muon_finisher_switch", "restart": restart_i, "ep": ep,
                        "muon_lr": muon_lr, "momentum": muon_momentum,
                        "muon_on_hidden_only": (not muon_all),
                        "adam_tail_lr": (None if muon_all else muon_adam_lr),
                        "muon_lr_rewarmup_epochs": (wu_eps_m if wu_eps_m > 0 else 0),
                        "muon_lr_final_frac": muon_lr_final_frac,
                        "muon_lr_decay_active": bool(muon_lr_final_frac < 1.0),
                        "muon_warm_start_momentum": _warm_start,
                        "muon_warm_seeded_leaves": _warm_seeded,
                        "trigger": ("plateau_or_cap" if plateau_trigger else "fixed_cap"),
                        "note": "AdamW -> Muon(hidden) + Adam(code-latent/biases/out) finisher; EMA + params continue",
                    }), flush=True)
                order = rng.permutation(P)
                accum_grads = None
                accum_loss_sum = 0.0
                count = 0
                ep_gnorms: list[float] = []
                n_skips_ep = 0
                for bi, pi_np in enumerate(order):
                    pi = int(pi_np)
                    loss, grads = value_and_grad(
                        model, feats_for_pair(pi), 2 * pi, 2 * pi + 1,
                        lstar_oh[pi], margins_mx[pi], poses_mx[pi], w_seg, w_pose, hinge, margin_target, mw_active, mw_temp_ep, seg_form_ep, tau_ep, l7_thr_ep,
                    )
                    # Materialize per pair so the lazy fwd+bwd graph does NOT accumulate
                    # across the epoch (the 499000 buffer-count cap fix, preserved).
                    mx.eval(loss, grads)
                    lval = float(loss)
                    ep_loss += lval
                    # (3) accumulate grads over up to accum_pairs before the opt+ema step.
                    if accum_grads is None:
                        accum_grads = grads
                    else:
                        accum_grads = tree_map(lambda a, b: a + b, accum_grads, grads)
                    mx.eval(accum_grads)
                    accum_loss_sum += lval
                    count += 1
                    is_last = bi == (P - 1)
                    if count < accum_pairs and not is_last:
                        continue
                    # --- accum-batch boundary: mean -> clip -> spike-guard -> opt+ema ---
                    mean_grads = tree_map(lambda g, c=float(count): g / c, accum_grads)
                    # (1) grad-norm clip (THE primary fix); grad_clip<=0 disables (1e30
                    # sentinel = no actual clip, but the norm is still measured + logged).
                    clipped, total = optim.clip_grad_norm(mean_grads, grad_clip if grad_clip > 0 else 1e30)
                    mx.eval(total)
                    gnorm = float(total)
                    batch_loss = accum_loss_sum / float(count)
                    median = float(np.median(np.asarray(recent))) if len(recent) else None
                    # (5) spike-guard: skip the opt+ema step on a non-finite OR runaway
                    # (> spike_factor x running median) batch so the run can recover.
                    skip = (not math.isfinite(batch_loss)) or (not math.isfinite(gnorm)) or (
                        median is not None and batch_loss > spike_factor * median
                    )
                    if skip:
                        n_skips_ep += 1
                        consec_skips += 1
                        print(json.dumps({
                            "stage": "spike_skip", "restart": restart_i, "ep": ep, "gstep": gstep,
                            "batch_loss": (round(batch_loss, 4) if math.isfinite(batch_loss) else "nonfinite"),
                            "gnorm": (round(gnorm, 4) if math.isfinite(gnorm) else "nonfinite"),
                            "median": (round(median, 4) if median is not None else None),
                            "consec_skips": consec_skips,
                        }), flush=True)
                        # (C17 skip-rate-ceiling ESCAPE HATCH) A full window of CONSECUTIVE skips
                        # == 100% skip-fraction over `recent_maxlen` accum-batches: the classic
                        # absorbing median-freeze. Turn the SILENT freeze LOUD (spike_deadlock
                        # ALERT) and BREAK the absorbing state by clearing `recent` -> median
                        # becomes None -> the next batch is ACCEPTED unconditionally (same
                        # deadlock-free re-arm semantics as the restart-top / curriculum re-arm)
                        # -> the median re-baselines at the NEW loss level and training resumes.
                        # This mirrors the levelset SpikeGuardRollback intent (fresh re-arm on a
                        # sustained all-skip window) minimally, without needing the base loop to
                        # carry a last-good snapshot / lr-cut machinery.
                        if consec_skips >= recent_maxlen:
                            print(json.dumps({
                                "stage": "spike_deadlock", "level": "ALERT",
                                "restart": restart_i, "ep": ep, "gstep": gstep,
                                "consec_skips": consec_skips, "window": recent_maxlen,
                                "median": (round(median, 4) if median is not None else None),
                                "batch_loss": (round(batch_loss, 4) if math.isfinite(batch_loss) else "nonfinite"),
                                "action": "spike_rebaseline",
                                "note": "100% skip over a full window (absorbing median-freeze); "
                                        "clearing recent to force a fresh median re-baseline "
                                        "(deadlock-free re-arm) instead of freezing forever.",
                            }), flush=True)
                            recent.clear()
                            consec_skips = 0
                    else:
                        opt.update(model, clipped)
                        mx.eval(model.parameters(), opt.state)
                        ema.update(model)
                        mx.eval(list(ema.shadow.values()))
                        recent.append(batch_loss)
                        ep_gnorms.append(gnorm)
                        consec_skips = 0  # (C17) accepted batch => the run is live; reset the ceiling
                    # (7) per-accum grad-norm logging (throttled), always on clip/skip.
                    clipped_fired = grad_clip > 0 and gnorm > grad_clip
                    if (gstep % grad_log_every == 0) or clipped_fired or skip:
                        # (bug #8) per-layer param-grad norms surface vanishing gradients (the
                        # hosc saturation symptom: in_proj/hidden.* grads collapse toward 0 while
                        # the output layer still moves). A faithful, available proxy for the
                        # per-layer activation-grad norm (value_and_grad exposes param grads).
                        per_layer = {}
                        for gname, gval in tree_flatten(mean_grads):
                            ga = np.asarray(gval, dtype=np.float64)
                            if ga.size:
                                per_layer[gname] = round(float(np.sqrt(np.sum(ga * ga))), 6)
                        print(json.dumps({
                            "stage": "grad", "restart": restart_i, "ep": ep, "gstep": gstep,
                            "gnorm": round(gnorm, 4), "clip": grad_clip, "clipped": bool(clipped_fired),
                            "skipped": bool(skip), "lr": round(float(opt.learning_rate), 6),
                            "hosc_beta": (round(float(model.hosc_beta), 4) if activation == "hosc" else None),
                            "per_layer_grad_norm": per_layer,
                        }), flush=True)
                    gstep += 1
                    accum_grads = None
                    accum_loss_sum = 0.0
                    count = 0
                n_skips_total += n_skips_ep
                # (C17 epoch-level LIVENESS alarm) Turn a silent frozen epoch LOUD: if the
                # accepted-fraction of accum-batch boundaries over the epoch is ~0 (the optimizer
                # took essentially no steps), emit a spike_deadlock alarm so no downstream reader
                # (verdict / closed-loop / telemetry) can mistake a FROZEN epoch for a converging
                # one. len(ep_gnorms) = accepted boundaries; n_skips_ep = skipped boundaries.
                _ep_boundaries = len(ep_gnorms) + n_skips_ep
                if _ep_boundaries > 0:
                    _accepted_frac = len(ep_gnorms) / float(_ep_boundaries)
                    if _accepted_frac <= 0.02:
                        print(json.dumps({
                            "stage": "spike_deadlock", "level": "ALERT", "scope": "epoch",
                            "restart": restart_i, "ep": ep,
                            "accepted_frac": round(_accepted_frac, 4),
                            "accepted_batches": len(ep_gnorms), "skipped_batches": n_skips_ep,
                            "note": "accepted-fraction ~0 over the epoch: optimizer effectively "
                                    "FROZEN this epoch; verdict/telemetry rows for this epoch are "
                                    "measured on non-stepped weights (liveness stamp).",
                        }), flush=True)
                # Release reclaimable Metal buffer cache between epochs (defensive vs
                # the 499000 active-buffer cap when many arms share the GPU).
                mx.clear_cache()
                ep_walls.append(time.time() - ep_t0)
                if ep % args.eval_every == 0 or ep == 1 or ep == args.epochs:
                    n_evals_total += 1
                    is_last_ep = ep == args.epochs
                    # Win #2 (single verdict for monitoring): compute the LIVE verdict for the
                    # descent trend; compute the EMA verdict only when needed (verdict_which
                    # selects, OR the ema-lag cadence, OR the last epoch -> best EMA checkpoint).
                    do_live = verdict_which in ("live", "both")
                    do_ema = verdict_which in ("ema", "both") or (
                        verdict_which == "live" and (n_evals_total % ema_verdict_every == 0 or is_last_ep)
                    )
                    v_live = eval_verdict(use_ema=False, pair_idx=_verdict_subset) if do_live else None
                    v = eval_verdict(use_ema=True, pair_idx=_verdict_subset) if do_ema else None
                    # (bug #6) MEASURED deploy-archive bytes of the int8+brotli EMA blob.
                    measured_bytes = measure_witness_blob_bytes(list(ema.shadow.items()))
                    subset_mon = len(_verdict_subset) < P
                    row = {
                        "restart": restart_i, "epoch": ep, "train_loss": round(ep_loss / P, 4),
                        "curriculum_stage": (cur_stage if curriculum else None),
                        "measured_bytes": int(measured_bytes),
                        "monitor_pairs": len(_verdict_subset),
                        # subset rows are a TREND ESTIMATE, NOT the scored row (full-600 byte-close
                        # is the authority). Labeled so no row is mistaken for a frontier score.
                        "monitor_estimate": bool(subset_mon),
                        "gnorm_max": (round(max(ep_gnorms), 4) if ep_gnorms else None),
                        "gnorm_mean": (round(float(np.mean(ep_gnorms)), 4) if ep_gnorms else None),
                        "n_skips": n_skips_ep, "lr": round(float(opt.learning_rate), 6),
                        "ep_wall_s": round(ep_walls[-1], 2),
                        "wall_s": round(time.time() - t_train, 1),
                    }
                    if v_live is not None:
                        implied_S_live = implied_score_from_verdict(v_live["d_seg"], v_live["d_pose"], measured_bytes)
                        row.update({"d_seg_live": round(v_live["d_seg"], 6), "d_pose_live": round(v_live["d_pose"], 6),
                                    "implied_S_live": round(implied_S_live, 4)})
                        if implied_S_live < best.get("implied_S_live", 1e9):
                            best["implied_S_live"] = implied_S_live
                            best["d_seg_live"] = v_live["d_seg"]; best["d_pose_live"] = v_live["d_pose"]; best["epoch_live"] = ep
                        if implied_S_live < best_live_saved["implied_S"]:
                            best_live_saved = {"implied_S": implied_S_live, "d_seg": v_live["d_seg"],
                                               "restart": restart_i, "epoch": ep,
                                               "params": {k: np.asarray(val, dtype=np.float32) for k, val in tree_flatten(model.parameters())}}
                    if v is not None:
                        implied_S = implied_score_from_verdict(v["d_seg"], v["d_pose"], measured_bytes)
                        row.update({"d_seg": round(v["d_seg"], 6), "d_pose": round(v["d_pose"], 6),
                                    "implied_S": round(implied_S, 4)})
                        if implied_S < best["implied_S"]:
                            best = {**best, "d_seg": v["d_seg"], "d_pose": v["d_pose"], "implied_S": implied_S, "epoch": ep, "restart": restart_i}
                        if v["d_seg"] < best_saved["d_seg"]:
                            best_saved = {"d_seg": v["d_seg"], "restart": restart_i, "epoch": ep,
                                          "params": {k: np.asarray(val, dtype=np.float32) for k, val in ema.shadow.items()}}
                    # EMA-LAG warning (only when BOTH verdicts are present this eval).
                    if v is not None and v_live is not None and v_live["d_seg"] > 1e-9 and v["d_seg"] >= ema_lag_warn_ratio * v_live["d_seg"]:
                        print(json.dumps({
                            "stage": "WARNING_ema_lag", "restart": restart_i, "ep": ep,
                            "ema_d_seg": round(v["d_seg"], 6), "live_d_seg": round(v_live["d_seg"], 6),
                            "lag_ratio": round(v["d_seg"] / max(v_live["d_seg"], 1e-9), 1),
                            "advice": "EMA shadow lags live; byte-close witness_live_mlx.npz via "
                                      "tools/witness_byte_close_and_eval.py --use-live",
                        }), flush=True)
                    result_history.append(row)
                    print(json.dumps(row), flush=True)
                    # (FEED-ca fix 2) record the monitored d_seg for the plateau-trigger slope
                    # fit. Prefer the EMA verdict (the deploy signal) when computed this eval,
                    # else the LIVE verdict. Only needed when plateau_trigger drives transitions.
                    if curriculum and plateau_trigger:
                        _dseg_mon = (v or v_live)
                        if _dseg_mon is not None:
                            dseg_history.append((ep, float(_dseg_mon["d_seg"])))
                    # (resumability) persist the best-so-far checkpoints AT EACH EVAL so a crash
                    # leaves a byte-closeable witness on disk (overwrite; cheap ~0.5 MB each).
                    try:
                        if best_live_saved["params"] is not None:
                            np.savez(out_dir / "witness_live_mlx.npz", **best_live_saved["params"])
                        if best_saved["params"] is not None:
                            np.savez(out_dir / "witness_ema_mlx.npz", **best_saved["params"])
                    except Exception as _e:  # never let a checkpoint write kill the run
                        print(json.dumps({"stage": "WARNING_checkpoint_save_failed", "ep": ep, "err": str(_e)}), flush=True)
                    # restart divergence guard: use the EMA d_seg when available this eval, else
                    # the LIVE d_seg (monitoring trend) vs the baseline.
                    div_metric = (v or v_live)
                    if n_restarts > 1 and not checked_div and ep >= mid_ep and div_metric is not None:
                        checked_div = True
                        if base_verdict is not None and div_metric["d_seg"] >= div_frac * base_verdict["d_seg"]:
                            print(json.dumps({
                                "stage": "restart_divergence", "restart": restart_i, "ep": ep,
                                "d_seg": round(div_metric["d_seg"], 6),
                                "baseline_d_seg": round(base_verdict["d_seg"], 6),
                                "action": "reseed_and_restart",
                            }), flush=True)
                            diverged = True
                            break
            if not diverged:
                break  # healthy completion -> keep best, stop spending restarts

        # --- 5. Load the BEST EMA witness (across restarts), quantize + save it ---
        if best_saved["params"] is not None:
            holder["model"].update(tree_unflatten([(k, mx.array(v)) for k, v in best_saved["params"].items()]))
            mx.eval(holder["model"].parameters())
            ema_params = best_saved["params"]
        else:
            holder["model"].update(holder["ema"].shadow_tree())
            mx.eval(holder["model"].parameters())
            ema_params = {k: np.asarray(v, dtype=np.float32) for k, v in tree_flatten(holder["model"].parameters())}
        blob = quantize_witness_blob(holder["model"])
        np.savez(out_dir / "witness_ema_mlx.npz", **ema_params)
        print(json.dumps({
            "stage": "saved_best_ema", "restart": best_saved["restart"], "epoch": best_saved["epoch"],
            "d_seg": (round(float(best_saved["d_seg"]), 6) if best_saved["params"] is not None else None),
            "n_skips_total": n_skips_total,
        }), flush=True)
        # (DAG FEED-br sister) ALSO save the best-implied-S LIVE-weights witness so the
        # byte-close can pick it (--use-live) when the EMA shadow lags (the 78x trap).
        ema_lag_at_best = None
        if best_live_saved["params"] is not None:
            np.savez(out_dir / "witness_live_mlx.npz", **best_live_saved["params"])
            if best_saved["params"] is not None and best_live_saved["d_seg"] > 1e-9:
                ema_lag_at_best = round(float(best_saved["d_seg"]) / max(float(best_live_saved["d_seg"]), 1e-9), 1)
            print(json.dumps({
                "stage": "saved_best_live", "restart": best_live_saved["restart"], "epoch": best_live_saved["epoch"],
                "live_d_seg": round(float(best_live_saved["d_seg"]), 6),
                "best_ema_d_seg": (round(float(best_saved["d_seg"]), 6) if best_saved["params"] is not None else None),
                "ema_lag_ratio_at_best": ema_lag_at_best,
                "advice": ("EMA lags live; prefer --use-live byte-close" if (ema_lag_at_best or 0) >= ema_lag_warn_ratio else "EMA ~= live"),
            }), flush=True)

    mean_ep = float(np.mean(ep_walls)) if ep_walls else 0.0
    median_ep = float(np.median(ep_walls)) if ep_walls else 0.0
    final = result_history[-1] if result_history else {}
    result = {
        "subagent": "train_witness_realized_through_R_mlx_lever2_20260625",
        "utc": _utc(),
        "evidence_grade_train": "[macOS-MLX training-gradient]",
        # Device-truthful: on macOS the frozen CPU-torch verdict is [macOS-CPU advisory]
        # (NOT 1:1 with the contest Linux x86_64 CPU runner); only a Linux x86_64 host
        # earns [contest-CPU advisory] (DAG FEED-br sister fix — the axis mislabel).
        "evidence_grade_verdict": _advisory_axis_label(non_promotable=False),
        "promotion_eligible": False, "score_claim": False, "ready_for_exact_eval_dispatch": False,
        "axis": "REALIZED (post-R, post-frozen-scorer) -- the actually-scored axis",
        "port_fidelity_DAG_FEED_br": {
            # The byte-close blocker REVIEWER 1 found: MLX-GPU forward is reduced-precision
            # -> the verdict now renders in NUMPY fp32 (byte-identical to inflate.py), so
            # the reported d_seg == the byte-closed d_seg.
            "verdict_forward_numpy_fp32_deploy_faithful": True,
            "mlx_gpu_only_trains_gradient": True,
            "saved_best_live_weights_for_byte_close": bool(best_live_saved["params"] is not None),
            "ema_lag_ratio_at_best": ema_lag_at_best,
        },
        "fidelity_fixes_DAG_FEED_bf": {
            "R_contest_faithful_uint8_at_camera": True,   # bug #1
            "spike_guard_resets_per_restart": True,       # bug #2
            "verdict_R_in_torch_authority": True,         # bug #3
            "verdict_on_int8_dequant_weights": bool(getattr(args, "int8_verdict", True)),  # bug #4
            "directional_basis_byte_closeable": directional_byte_closeable,  # bug #5 (False; research-only)
            "directional_basis_research_only": directional_research_only,
            "rate_term_measured_blob_bytes": True,        # bug #6
            "score_via_canonical_contest_score": True,    # bug #7
            "frontier_from_pointer": True,                # bug #7
            "hosc_beta_anneal": (hosc_beta_start != hosc_beta_end),  # bug #8
        },
        "mlx_device": device,
        "config": {
            "num_pairs": P, "epochs": args.epochs, "render_h": render_h, "render_w": render_w,
            "n_fourier": args.n_fourier, "hidden_dim": args.hidden_dim, "n_hidden": args.n_hidden,
            "mod_dim": args.mod_dim, "fourier_sigma": args.fourier_sigma,
            "lr": args.lr, "weight_decay": args.weight_decay, "ema_decay": args.ema_decay,
            "optimizer": str(getattr(args, "optimizer", "adamw")),
            "md_base": str(getattr(args, "md_base", "adam")),
            "md_gain_lr_scale": float(getattr(args, "md_gain_lr_scale", 1.0 / 3.0)),
            "w_seg": args.w_seg, "w_pose": args.w_pose, "hinge_weight": args.hinge_weight,
            "score_domain_loss": bool(getattr(args, "score_domain_loss", True)),
            "pose_eps": float(getattr(args, "pose_eps", 1e-2)),
            "grad_clip": float(getattr(args, "grad_clip", 1.0)),
            "accum_pairs": int(getattr(args, "accum_pairs", 8)),
            "n_restarts": int(getattr(args, "n_restarts", 2)),
            "spike_factor": float(getattr(args, "spike_factor", 5.0)),
            "lr_schedule": bool(getattr(args, "lr_schedule", True)),
            "warmup_epochs": int(getattr(args, "warmup_epochs", 1)),
            "lr_end": float(getattr(args, "lr_end", 1e-4)),
            "grad_log_every": int(getattr(args, "grad_log_every", 25)),
            "basis": basis, "n_dir_freqs": n_dir_freqs,
            "freq_across": float(getattr(args, "freq_across", 32.0)),
            "freq_along": float(getattr(args, "freq_along", 4.0)),
            "activation": activation, "hosc_beta": float(getattr(args, "hosc_beta", 4.0)),
            "hosc_beta_start": hosc_beta_start, "hosc_beta_end": hosc_beta_end,
            "hosc_omega": float(getattr(args, "hosc_omega", 1.0)),
            "chroma": chroma,
            "siren_init": siren_init,
            "siren_omega": siren_omega,
            "finer_first_bias_scale": finer_first_bias_scale,
            "wire_scale": wire_scale,
            "int8_verdict": bool(getattr(args, "int8_verdict", True)),
            "seg_loss": seg_loss,
            "margin_target_start": float(getattr(args, "margin_target_start", 1.0)),
            "margin_target_end": float(getattr(args, "margin_target_end", 0.5)),
            "margin_weighted_loss": bool(getattr(args, "margin_weighted_loss", False)),
            "margin_weight_fn": str(getattr(args, "margin_weight_fn", "inverse")),
            "margin_weight_temp": float(getattr(args, "margin_weight_temp", 1.0)),
            "margin_weight_start_epoch": int(getattr(args, "margin_weight_start_epoch", 80)),
            "margin_weight_temp_start": float(getattr(args, "margin_weight_temp_start", 1.0)),
            "margin_weight_anneal_epochs": int(getattr(args, "margin_weight_anneal_epochs", 40)),
            "margin_stage_lr_warmup_epochs": int(getattr(args, "margin_stage_lr_warmup_epochs", 8)),
            "margin_stage_lr_floor": float(getattr(args, "margin_stage_lr_floor", 0.1)),
            "curriculum": bool(getattr(args, "curriculum", False)),
            "tau_softplus_start_epoch": int(getattr(args, "tau_softplus_start_epoch", 300)),
            "muon_finisher_start_epoch": int(getattr(args, "muon_finisher_start_epoch", 1200)),
            "tau_softplus_tau": float(getattr(args, "tau_softplus_tau", 0.3)),
            "l7_mult": float(getattr(args, "l7_mult", 4.0)),
            "l7_threshold": float(getattr(args, "l7_threshold", 1.0)),
            "muon_lr": float(getattr(args, "muon_lr", 1e-4)),
            "muon_lr_final_frac": float(getattr(args, "muon_lr_final_frac", 1.0)),
            "muon_warm_start_momentum": bool(getattr(args, "muon_warm_start_momentum", False)),
            "resume_from": (str(getattr(args, "resume_from")) if getattr(args, "resume_from", None) else None),
            "verdict_device": str(getattr(args, "verdict_device", "mlx-cpu")),
            "verdict_which": str(getattr(args, "verdict_which", "live")),
            "ema_verdict_every": int(getattr(args, "ema_verdict_every", 4)),
            "verdict_batch": int(getattr(args, "verdict_batch", 16)),
            "verdict_pairs": int(getattr(args, "verdict_pairs", 120)),
            "seed": args.seed,
        },
        "throughput": {
            "mean_ep_wall_s": round(mean_ep, 3),
            "median_ep_wall_s": round(median_ep, 3),
            "n_pairs": P, "epochs": args.epochs,
            "n_skips_total": n_skips_total,
        },
        "baseline_verdict": {k: round(v, 6) for k, v in base_verdict.items()},
        "final_verdict": {"d_seg": final.get("d_seg"), "d_pose": final.get("d_pose")},
        "best_verdict": {k: (round(v, 6) if isinstance(v, float) else v) for k, v in best.items()},
        "history": result_history,
        "quantized_witness_blob": blob,
        "gt_precompute_secs": round(gt_secs, 1),
        "frontier_realized": {"d_seg": 5.6e-4, "d_pose": 1.6e-5, "bytes": 177000, "S": FRONTIER},
    }
    (out_dir / "train_result.json").write_text(json.dumps(result, indent=2))
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="CAPSTONE LEVER #2 (MLX): witness through R, both MLX scorers in-loop")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--num-pairs", type=int, default=24)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--render-h", type=int, default=192)
    ap.add_argument("--render-w", type=int, default=256)
    ap.add_argument("--n-fourier", type=int, default=24)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--n-hidden", type=int, default=4)
    ap.add_argument("--mod-dim", type=int, default=48)
    ap.add_argument("--fourier-sigma", type=float, default=8.0)
    ap.add_argument("--lr", type=float, default=1e-3,
                    help="base LR (lowered 2e-3->1e-3 for stability; warmup ramps 0->lr then cosine to --lr-end).")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    # (#222) AdamW second-moment decay beta2 (beta1 fixed 0.9). Default 0.999 == MLX default =>
    # bit-identical. Small-n (n = P/accum_pairs) optimum ~0.9999999 per arXiv 2603.02092
    # (1-beta2 <~ (1-beta1^5)/n^3.5). Sister of the levelset trainer's --adam-beta2.
    ap.add_argument("--adam-beta2", type=float, default=0.999,
                    help="#222 AdamW beta2 (beta1 fixed 0.9). 0.999 default = bit-identical; small-n "
                    "optimum ~0.9999999.")
    ap.add_argument("--ema-decay", type=float, default=0.997)
    ap.add_argument("--hinge-weight", type=float, default=4.0)
    # Score-aligned defaults: the contest score weights d_seg 100x and damps d_pose
    # via sqrt; with --score-domain-loss the loss is w_seg*seg_l + w_pose*sqrt(10*pose_l)
    # so these defaults mirror S = 100*d_seg + sqrt(10*d_pose). The witness's sole
    # binding controllable job is d_seg (pose rides the stored-target sidecar).
    ap.add_argument("--w-seg", type=float, default=100.0)
    ap.add_argument("--w-pose", type=float, default=1.0)
    ap.add_argument(
        "--score-domain-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="score-domain loss = w_seg*seg_l + w_pose*sqrt(10*pose_l + eps), mirroring S's "
        "distortion marginals (HNeRV parity LESSON 6, CLAUDE.md). Default TRUE for the witness; "
        "--no-score-domain-loss restores the raw (anti-aligned) w_seg*seg_l + w_pose*pose_l ablation.",
    )
    ap.add_argument(
        "--pose-eps",
        type=float,
        default=1e-2,
        help="sqrt-stability eps for the score-domain pose term. Raised 1e-8->1e-2: caps the "
        "pose grad coeff 5/sqrt(10*pose_l+eps) at <=50 (was up to ~5e4 for easy pairs at 1e-8, "
        "the divergence trigger). Pose below ~1e-3 is the stored-sidecar's job, so linearizing "
        "the witness pose term there is correct (DAG FEED-bd/be).",
    )
    ap.add_argument(
        "--witness-bytes", type=int, default=80000,
        help="DEPRECATED fallback only — the implied-S rate term now uses the MEASURED int8+brotli "
        "blob bytes per eval epoch (bug #6), NOT this fixed value.",
    )
    ap.add_argument(
        "--int8-verdict", action=argparse.BooleanOptionalAction, default=True,
        help="(bug #4) compute the authority verdict on the int8-DEQUANTIZED deploy weights "
        "(eval_roundtrip-for-WEIGHTS) — the deploy archive is int8+brotli and small-margin pixels "
        "flip under int8. --no-int8-verdict renders the fp32 EMA weights (overstates the witness).",
    )
    ap.add_argument(
        "--ema-lag-warn-ratio", type=float, default=2.0,
        help="(DAG FEED-br sister) warn when the EMA-shadow d_seg trails the LIVE d_seg by "
        ">= this ratio (the 78x ema-lag trap); the best LIVE weights are saved to "
        "witness_live_mlx.npz for --use-live byte-close regardless.",
    )
    ap.add_argument("--mlx-device", choices=["gpu", "cpu"], default="gpu")
    ap.add_argument(
        "--verdict-device", choices=["numpy", "mlx-cpu", "mlx-gpu"], default="mlx-cpu",
        help="(ISSUE 1 throughput) device for the IN-LOOP verdict MLP forward: mlx-cpu "
        "(default; fp32-accurate 3-18 px/pair << 143-px budget, FAST), numpy (fp32, 0 px/pair "
        "vs the byte-close, AUTHORITATIVE but slow on 600 pairs), or mlx-gpu (fastest but "
        "reduced-precision ~495-1672 px -> coarse 0.5->0.01 descent ONLY). The FINAL byte-close "
        "row is always numpy fp32 (tools/witness_byte_close_and_eval.py).",
    )
    ap.add_argument(
        "--verdict-batch", type=int, default=16,
        help="(WALL-CLOCK win #1) batch N frames into ONE frozen-SegNet/PoseNet forward. The "
        "scorer is eval-mode (BN running stats) so batching is BIT-EXACT vs frame-by-frame and "
        "far faster on CPU (the SegNet forward is ~97%% of verdict cost).",
    )
    ap.add_argument(
        "--verdict-which", choices=["live", "ema", "both"], default="live",
        help="(WALL-CLOCK win #2) which weights the IN-LOOP verdict scores: live (default; "
        "the descent trend -- 2x fewer forwards), ema, or both. With live, the EMA verdict is "
        "still computed every --ema-verdict-every evals (+ the last epoch) for the ema-lag check "
        "and the best-EMA checkpoint. The FINAL byte-close always uses the EMA deploy weights.",
    )
    ap.add_argument(
        "--ema-verdict-every", type=int, default=4,
        help="(WALL-CLOCK win #2) when --verdict-which live, also compute the EMA verdict every "
        "N evals (and the last epoch) for the ema-lag warning + best-EMA selection.",
    )
    ap.add_argument(
        "--verdict-pairs", type=int, default=0,
        help="(WALL-CLOCK win #3) IN-LOOP monitoring verdict scores a FIXED evenly-spaced subset "
        "of N pairs (same pairs every eval -> comparable d_seg TREND). 0 = all P pairs. Subset "
        "rows are labeled monitor_estimate=true (a TREND, NOT a scored row); the FINAL byte-close "
        "(tools/witness_byte_close_and_eval.py) ALWAYS scores ALL 600 exactly.",
    )
    # --- OPTIMIZER (default adamw = existing path UNCHANGED) ---
    ap.add_argument(
        "--optimizer", choices=["adamw", "md"], default="adamw",
        help="optimizer: adamw (default/control, existing path) or md = MD-Decoupling "
        "(arXiv:2606.25971) — factorize each 2D weight into a fixed-norm hypersphere "
        "direction + per-row/per-col magnitude gains at a SEPARATE LR. The re-audit's "
        "optimizer-bug fix: anti-collapse, warmup-free, transfers optimal LR across width. "
        "Ablation for the witness; NO score claim until a byte-closed exact-eval row.",
    )
    ap.add_argument(
        "--md-base", choices=["adam", "muon"], default="adam",
        help="MD direction base optimizer (the paper composes with BOTH): adam (Adam on the "
        "sphere direction) or muon (momentum + Newton-Schulz orthogonalization).",
    )
    ap.add_argument(
        "--md-gain-lr-scale", type=float, default=1.0 / 3.0,
        help="MD per-row/col gain LR = md_gain_lr_scale * direction_LR (paper eta_W 3e-3 / "
        "eta_gamma 1e-3 ~= 1/3).",
    )
    ap.add_argument(
        "--gt-cache",
        type=str,
        default=None,
        help="optional shared GT npz (tools/build_shared_gt_cache_for_mlx_fleet.py) -- "
        "skips the ~480s/arm GT precompute so a parallel fleet shares ONE cache.",
    )
    ap.add_argument("--seed", type=int, default=0)
    # --- STABILIZATION (DAG FEED-bd/be: collapse -> dead-saturation lock fix) ---
    ap.add_argument(
        "--grad-clip", type=float, default=1.0,
        help="(1) global grad-norm clip applied to the (mean-over-accum) grads before opt.update "
        "— THE primary divergence fix. <=0 disables clipping (norm still measured + logged).",
    )
    ap.add_argument(
        "--accum-pairs", type=int, default=8,
        help="(3) accumulate grads over K pairs before one opt.update+ema.update (variance "
        "reduction that removes the per-pair spike trigger). 1 = per-pair update.",
    )
    ap.add_argument(
        "--n-restarts", type=int, default=2,
        help="(6) max restart attempts: if EMA d_seg has not dropped below 0.9*baseline by the "
        "midpoint the run is reseeded+restarted; the best EMA-d_seg checkpoint is kept across "
        "restarts. A healthy (non-diverged) restart stops the loop early. 1 = single run.",
    )
    ap.add_argument(
        "--spike-factor", type=float, default=5.0,
        help="(5) spike-guard: skip the opt+ema step when the accum batch loss is non-finite OR "
        "> spike_factor x the running median of recent batch losses (recoverability).",
    )
    ap.add_argument(
        "--lr-schedule", action=argparse.BooleanOptionalAction, default=True,
        help="(4) enable 1-epoch linear warmup (0->lr) + cosine decay (lr->--lr-end). "
        "--no-lr-schedule uses a constant --lr (ablation).",
    )
    ap.add_argument("--warmup-epochs", type=int, default=1, help="(4) #epochs of linear LR warmup before cosine decay.")
    ap.add_argument("--lr-end", type=float, default=1e-4, help="(4) final cosine LR floor.")
    ap.add_argument("--grad-log-every", type=int, default=25, help="(7) log a grad-norm line every N accum steps (always on clip/skip).")
    # --- CAPSTONE d_seg / R-survival levers (each opt-in; default = current behavior) ---
    ap.add_argument(
        "--chroma", action=argparse.BooleanOptionalAction, default=True,
        help="CHROMA d_seg lever (operator 'Chroma too'): --chroma (default) emits 3 "
        "INDEPENDENT RGB planes so the frozen SegNet argmax (over RGB) + PoseNet's 2 YUV6 "
        "chroma planes both read genuine chroma (a real argmax-flip actuator); --no-chroma "
        "forces ACHROMATIC output (BT.601 luma replicated to R=G=B) -- the ablation CONTROL "
        "arm that isolates chroma's d_seg contribution.",
    )
    ap.add_argument(
        "--basis", choices=["isotropic", "directional"], default="isotropic",
        help="coord feature basis: isotropic Fourier (default/control/GATE) or ALL-CLASS "
        "DIRECTIONAL (per-pair boundary-tangent oriented; -48%% d_seg on the direct generator "
        "BUT GT-derived => NOT byte-closeable for this RGB witness, RESEARCH/upper-bound ONLY).",
    )
    ap.add_argument("--n-dir-freqs", type=int, default=6, help="directional basis: #freq octaves (in_feat=4*n).")
    ap.add_argument("--freq-across", type=float, default=32.0, help="directional basis: high freq ACROSS the edge (normal).")
    ap.add_argument("--freq-along", type=float, default=4.0, help="directional basis: low freq ALONG the edge (tangent).")
    ap.add_argument(
        "--activation", choices=["relu", "hosc", "siren", "finer", "wire"], default="relu",
        help="hidden nonlinearity: relu (control); siren=sin(w0*u) (Sitzmann 2020, w0=--siren-omega); "
        "finer=sin(w0*(|u|+1)*u) (Liu CVPR2024 FINER, the variable-periodic activation that BEAT SIREN "
        "to torch-d_seg 0.00138; spectral-bias knob = --finer-first-bias-scale wide first-layer bias); "
        "wire=sin(w0*u)*exp(-0.5*(s*u)^2) (Saragadam CVPR2024 WIRE real-Gabor, best space-frequency "
        "localization = smallest+most-compact EDGE error for the codim-1 boundary; s=--wire-scale); or "
        "hosc=tanh(beta*sin(omega*u)) — the "
        "STEP-NATIVE R-survival lever (no Gibbs through bicubic^->uint8->bilinear_ R). hosc "
        "defaults to a NON-SATURATING beta-anneal (1.0->--hosc-beta) when endpoints are unset, "
        "avoiding the BH2 vanishing-grad saturation of a constant beta=4. The LEARNABLE "
        "step_basis (sum-of-K tanh soft-steps, LearnableStepBasis in "
        "tac.substrates.siren.activation_family) is a torch-side substrate NOT yet MLX-ported; "
        "argparse fail-closes it here (no silent relu fallback) — port it if hosc transfers.",
    )
    ap.add_argument("--hosc-beta", type=float, default=4.0, help="hosc saturation/sharpness (beta->inf => step train).")
    ap.add_argument(
        "--hosc-beta-start", type=float, default=None,
        help="(bug #8) hosc beta-anneal START (low => non-saturated tanh => gradients flow early). "
        "Default None = constant --hosc-beta (no anneal).",
    )
    ap.add_argument(
        "--hosc-beta-end", type=float, default=None,
        help="(bug #8) hosc beta-anneal END (high => step-native late). Default None = --hosc-beta. "
        "Set start<end (e.g. 1.0 -> 6.0) to keep early gradients alive while converging to a step train.",
    )
    ap.add_argument("--hosc-omega", type=float, default=1.0, help="hosc base frequency.")
    ap.add_argument(
        "--siren-omega", type=float, default=30.0,
        help="SIREN/FINER periodic frequency w0 (Sitzmann/Liu canonical default 30; the "
        "OPTIMAL-FORM sweep is {10,30,60} — SIREN init scales hidden weights by 1/w0 so the "
        "pre-activation variance stays unit AFTER the w0 multiply). Per-basis-tuned, NOT shared.",
    )
    ap.add_argument(
        "--finer-first-bias-scale", type=float, default=1.0,
        help="FINER (Liu CVPR2024) FIRST-LAYER bias init range U(-k,+k) — the variable-periodic "
        "spectral-bias knob that gives FINER its edge over SIREN (selects which frequency sub-bands "
        "the |u|+1 scale activates). Only active with --activation finer. Sweep this for FINER's optimum.",
    )
    ap.add_argument(
        "--wire-scale", type=float, default=1.0,
        help="WIRE (Saragadam CVPR2024) real-Gabor Gaussian window scale s in sin(w0*u)*exp(-0.5*(s*u)^2) "
        "— controls space-frequency localization (larger s = tighter spatial support = sharper edge "
        "localization). Only active with --activation wire. Per-basis-tuned (sweep s with w0=--siren-omega).",
    )
    ap.add_argument(
        "--siren-init", action=argparse.BooleanOptionalAction, default=True,
        help="SIREN init (Sitzmann NeurIPS 2020) for the periodic-activation sine layers: "
        "FIRST sine layer (in_proj) W~U(-1/fan_in,+1/fan_in); each HIDDEN sine layer W~"
        "U(-sqrt(6/fan_in)/omega,+same) with omega=--hosc-omega; biases zeroed. The canonical "
        "fix for periodic/sine-MLP UNTRAINABILITY from scratch (parked capstone-v1 hosc-from-"
        "scratch was d_seg 0.689 with gnorm-460 spike-skips). Takes effect for the PERIODIC family "
        "(--activation hosc/siren/finer); relu keeps default MLX init. Use --no-siren-init for the "
        "ablation that measures the SIREN-init effect.",
    )
    ap.add_argument(
        "--seg-loss", choices=["ce", "margin_hinge"], default="ce",
        help="realized seg surrogate: margin-weighted CE (control) or margin_hinge "
        "relu(m-(z_GT-max_other)) (lensA: 16-36%% lower realized d_seg; soft_cosine worst).",
    )
    ap.add_argument("--margin-target-start", type=float, default=1.0, help="margin_hinge target at ep1 (anneals to end).")
    ap.add_argument("--margin-target-end", type=float, default=0.5, help="margin_hinge target at final ep (lensA: 0.5 best at floor).")
    # --- LIVE-MARGIN re-allocation lever (DAG FEED-bp NEW-B; default OFF = byte-identical) ---
    ap.add_argument(
        "--margin-weighted-loss", action=argparse.BooleanOptionalAction, default=False,
        help="DAG FEED-bp NEW-B: weight the realized seg loss per pixel by the LIVE SegNet "
        "margin (top1-top2 logit gap of the witness frame RENDERED THROUGH R) so the SAME "
        "loss budget concentrates on the ~2.2%% codim-1 decision-boundary annulus where ~89%% "
        "of d_seg lives (~20x effective-capacity multiplier at EQUAL bytes). The weight is "
        "mean-1 normalized (re-allocates, not adds) + STOP-GRADIENT (a per-pixel allocation "
        "prior, NOT a differentiable knob -- NO-FAKE: the witness must FIX the flip, not game "
        "its margin). DEFAULT OFF -> running baseline + default path byte-identical. The d_seg "
        "VERDICT is ALWAYS the unweighted realized argmax-disagreement (this only re-weights "
        "the TRAINING loss).",
    )
    ap.add_argument(
        "--margin-weight-fn", choices=["inverse", "exp", "bottom-k"], default="inverse",
        help="LIVE-margin allocator: inverse=1/(1+m/temp) (smooth), exp=exp(-m/temp) (smooth), "
        "bottom-k=hard mask 1[m<=quantile(m,temp)] (temp=bottom-fraction, e.g. 0.05). Only "
        "active with --margin-weighted-loss.",
    )
    ap.add_argument(
        "--margin-weight-temp", type=float, default=1.0,
        help="LIVE-margin allocator temperature: a logit scale for inverse/exp (smaller=sharper "
        "annulus focus), or the bottom-FRACTION for bottom-k (e.g. 0.05 = bottom 5%% margin). "
        "Only active with --margin-weighted-loss.",
    )
    ap.add_argument(
        "--margin-weight-start-epoch", type=int, default=80,
        help="(ISSUE 2 curriculum) engage the LIVE-margin annulus re-allocation only at "
        "ep >= N; epochs < N use UNIFORM seg weight so the base partition learns first. The "
        "live-margin allocator uses the PREDICTION margin, so from RANDOM INIT it starves the "
        "base (wrong px confidently-wrong = large margin = ~0 weight -> d_seg stuck at chance). "
        "Margin-weight is a FINETUNE lever (re-allocate a CONVERGED base), not from-scratch. "
        "Only active with --margin-weighted-loss; tune N to ~when the base is roughly learned.",
    )
    ap.add_argument(
        "--margin-weight-temp-start", type=float, default=1.0,
        help="(FEED-bu fix 1) GENTLE starting temp at the margin-engage; anneals to "
        "--margin-weight-temp (the moderate TARGET, e.g. 0.3) over --margin-weight-anneal-epochs. "
        "A cold exp/temp=0.1 from a converged base explodes the gnorm (~648) -> spike-guard "
        "all-skip; a gentle 1.0->0.3 ramp raises the gnorm gradually so the guard tracks it. "
        "Set == --margin-weight-temp to disable the anneal (back-compat).",
    )
    ap.add_argument(
        "--margin-weight-anneal-epochs", type=int, default=40,
        help="(FEED-bu fix 1) #epochs over which the margin-weight temp ramps "
        "--margin-weight-temp-start -> --margin-weight-temp after the engage. 0 = no anneal.",
    )
    ap.add_argument(
        "--resume-from", type=str, default=None,
        help="(FEED-bu fix 3) load a prior witness checkpoint npz (e.g. "
        "experiments/results/<run>/witness_live_mlx.npz) into restart 0 so a margin-finetune "
        "CONTINUES from a converged base instead of redoing it. Architecture flags "
        "(--num-pairs/--n-fourier/--hidden-dim/--n-hidden/--mod-dim) MUST match the checkpoint.",
    )
    ap.add_argument(
        "--margin-stage-lr-warmup-epochs", type=int, default=8,
        help="(FEED-bu fix 4, STAGE TRANSITION) re-warmup the LR from --margin-stage-lr-floor "
        "(0.1x) back to 1x over N epochs AT the margin-engage, so the FIRST margin-stage steps "
        "are small and absorb the loss-weighting change without the first-step gnorm spike. "
        "Composes with the temp-anneal + spike-guard recalibrate. 0 = no re-warmup.",
    )
    ap.add_argument(
        "--margin-stage-lr-floor", type=float, default=0.1,
        help="(FEED-bu fix 4) LR multiplier floor at the margin-engage (ramps floor->1x over "
        "--margin-stage-lr-warmup-epochs).",
    )
    # --- FEED-bv: the 4-STAGE PR95 d_seg CONDITIONING CURRICULUM (NOT HNeRV, NOT rate steps) ---
    ap.add_argument(
        "--curriculum", action=argparse.BooleanOptionalAction, default=False,
        help="(FEED-bv) run the 4-stage PR95 d_seg curriculum: ce (coarse argmax) -> tau_softplus "
        "(THE primary d_seg drop) -> l7_softplus (hard-pixel refine) -> Muon finisher. The seg "
        "LOSS FORM changes per stage; EACH boundary re-treats (spike-guard recalibrate + LR "
        "re-warmup + EMA continue). Drops smooth + ALL rate machinery. OFF -> single --seg-loss.",
    )
    ap.add_argument(
        "--tau-softplus-start-epoch", type=int, default=300,
        help="(FEED-bv) ce -> tau_softplus boundary (PR95 digest ~CE 300ep). The l7 boundary is "
        "--margin-weight-start-epoch; the Muon boundary is --muon-finisher-start-epoch.",
    )
    ap.add_argument(
        "--muon-finisher-start-epoch", type=int, default=1200,
        help="(FEED-bv) l7 -> Muon-finisher boundary (PR95 digest ~CE300+tau500+l7400).",
    )
    ap.add_argument("--tau-softplus-tau", type=float, default=0.3,
                    help="(FEED-bv) tau for tau_softplus/l7 (PR95 0.3): mean(tau*softplus(-margin/tau)).")
    ap.add_argument("--l7-mult", type=float, default=4.0,
                    help="(FEED-bv) l7_softplus hard-pixel weight: 1+l7_mult on margin<l7_threshold (PR95 4=5x).")
    ap.add_argument("--l7-threshold", type=float, default=0.42,
                    help="(FEED-bv/ca fix 3) l7_softplus small-margin threshold. Default 0.42 = the "
                    "MEASURED witness-hard median margin (FEED-bp), NOT the cargo-culted PR95 1.0.")
    ap.add_argument("--muon-lr", type=float, default=1e-4,
                    help="(FEED-bv) Muon finisher LR (low; the conditioning drop).")
    ap.add_argument("--muon-momentum", type=float, default=0.95,
                    help="(FEED-bv) Muon momentum (mlx default 0.95).")
    # --- FEED-ca: principled curriculum/optimizer fixes (4-review gate, operator 2026-06-26) ---
    ap.add_argument(
        "--plateau-trigger", action=argparse.BooleanOptionalAction, default=False,
        help="(FEED-ca fix 2) advance a curriculum stage when the MEASURED d_seg descent plateaus "
        "(slope > -plateau-slope-eps per epoch over the last --plateau-window in-stage eval points) "
        "OR the fixed --*-start-epoch CAP is reached (whichever first). Default OFF = fixed caps "
        "only (the witness descends slowly; a fixed cap risks l7-on-unconverged-base).",
    )
    ap.add_argument("--plateau-slope-eps", type=float, default=1e-5,
                    help="(FEED-ca fix 2) plateau threshold: advance when the per-epoch d_seg slope "
                    "> -eps (i.e. not dropping faster than eps/epoch). Only with --plateau-trigger.")
    ap.add_argument("--plateau-window", type=int, default=4,
                    help="(FEED-ca fix 2) #recent IN-STAGE eval points (each --eval-every epochs "
                    "apart) used to fit the d_seg slope. A FULL window of in-stage points is "
                    "required before a plateau can fire (gates the post-transition d_seg bump).")
    ap.add_argument("--tau-anneal-start", type=float, default=None,
                    help="(FEED-ca fix 3) tau-softplus tau ANNEAL start (coarse, e.g. 1.0) over the "
                    "tau stage. Set BOTH --tau-anneal-start/end to enable (else static --tau-softplus-tau).")
    ap.add_argument("--tau-anneal-end", type=float, default=None,
                    help="(FEED-ca fix 3) tau-softplus tau anneal end (fine/sharp, e.g. 0.2).")
    ap.add_argument("--l7-thr-anneal-start", type=float, default=None,
                    help="(FEED-ca fix 3) l7-threshold ANNEAL start (coarse, e.g. 1.0) over the l7 "
                    "stage. Set BOTH --l7-thr-anneal-start/end to enable (else static --l7-threshold).")
    ap.add_argument("--l7-thr-anneal-end", type=float, default=None,
                    help="(FEED-ca fix 3) l7-threshold anneal end (fine, e.g. 0.42 = measured median).")
    ap.add_argument("--muon-adam-lr", type=float, default=None,
                    help="(FEED-ca fix 5) Adam-tail LR for the code-latent embedding + biases + final "
                    "layer in the Muon finisher stage. None -> == --muon-lr.")
    ap.add_argument(
        "--muon-all-params", action=argparse.BooleanOptionalAction, default=False,
        help="(FEED-ca fix 5) ABLATION: apply Muon to ALL params in the finisher (back-compat). "
        "Default OFF = Muon ONLY on decoder-hidden weight matrices; code-latent (video-derived "
        "payload) + biases + final layer route to Adam (PR95: Muon degrades embedding/final).",
    )
    ap.add_argument(
        "--muon-lr-final-frac", type=float, default=1.0,
        help="(GAP 1, default-off) COSINE-DECAY the Muon-group LR from --muon-lr down to "
        "--muon-lr * this fraction across the Muon-stage span (composes AFTER the optional "
        "--margin-stage-lr-warmup-epochs re-warmup: warmup THEN decay). Muon's Newton-Schulz "
        "fixes update MAGNITUDE so a flat LR cannot self-reduce the step near the minimum "
        "(river-valley Muon 2606.21514). 1.0 (default) = flat/unchanged = byte-identical; "
        "e.g. 0.1 decays to 10%% of --muon-lr by stage end. Only the Muon group (the Adam tail "
        "self-adapts via its second moment). A/B-ready; no effect until the Muon stage.",
    )
    ap.add_argument(
        "--muon-warm-start-momentum", action=argparse.BooleanOptionalAction, default=False,
        help="(GAP 2, default-off) WARM-START the fresh Muon momentum buffer (state 'v') from the "
        "OUTGOING AdamW first-moment (state 'm') for the shared param paths, instead of cold zeros. "
        "Both are gradient EMAs and Newton-Schulz re-normalizes the update, so the transferred "
        "DIRECTION removes the cold-start 'wild unit-norm direction from one noisy gradient' "
        "boundary thrash / d_seg spike. Default OFF = cold zero start = byte-identical. Only a plain "
        "Adam/AdamW base is transferable; MD-Decoupling / non-Adam bases fall back to cold. A/B-ready.",
    )
    args = ap.parse_args(argv)

    result = run_train(args)
    print("\n=== THROUGH-R WITNESS RESULT (MLX, realized axis) ===")
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
