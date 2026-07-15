# SPDX-License-Identifier: MIT
"""texture_trunk — a BAND-DESIGNED per-class stationary texture trunk (T of W=(G, ξ, T)).

WHY (operator P0 #395, 2026-07-10 "pursue that texture trunk as a p0"):
The live level-set witness composes RGB as ``sigmoid(softmax(phi/temp)@palette + out_tex(h))*255``
where ``out_tex = Linear(hidden -> 3)`` is a LINEAR readout of the PARTITION trunk's hidden features
(``experiments/train_levelset_witness_realized_through_R_mlx.py::_compose_rgb``). Its expressible
textures are bounded by the partition trunk's frequency content — and the partition objective PULLS
those features smooth OFF the boundary while the texture objective WANTS in-region oscillation (the
P12 antagonism). The mod32cap witness (d_seg 0.0048) already beats the flat-paint floor (0.0416) 8.7×
via a MODEST ``out_tex`` (‖W‖~0.06) — evidence texture is worth a lot AND that the linear head is
capacity-starved for it.

WHAT (the decomposition-by-addition): keep the partition trunk G untouched; ADD a TINY SEPARATE
texture trunk T whose BASIS is texture-native — a fixed bank of per-pixel stationary oscillatory
(Gabor/grating) features, per-class fitted coefficients, PLACED through the existing partition
softmax masks and (optionally) annulus-attenuated near the decision boundary. T does NOT read the
partition trunk's hidden state (that is the whole point: decouple the two objectives); it reads only
the softmax MASKS for placement. So the two objectives live on disjoint parameter sets.

THE BAND IS DICTATED BY MEASUREMENT, NOT A GUESS (clause-B, minimal-dim = the geometry's bound):
:mod:`tac.through_r.stem_perception` read the FROZEN SegNet's EfficientNet-B2 ``conv_stem`` verbatim
(``.omx/research/segnet_texture_perception_20260710.md``). The stride-2 stem has a hard alias wall:
the finest texture PERIOD that survives to the first MBConv is ``2·stride = 4`` seg-input px
(``stem_nyquist()['finest_period_seg_input_px'] = 4.0``). The per-class through-R price list found the
ONLY texture that flips Road/Lane out of the flat-paint Undrivable basin is a **period-4** (= the
Nyquist) high-contrast luminance grating; **period-2 aliases away, period-8/16 read as flat**. Road
and Lane have a NEGATIVE flat floor — no constant colour classifies them. So the texture trunk's
feature bank is pinned to the surviving band ``[period-4 .. period-8]`` render-px (the render grid IS
384×512 = the seg-input grid, so render-px ≡ seg-input px for the surviving band; ``witness_autoconfig``
``render_h=384, render_w=512``), NOTHING at period < 4 (aliased) or period > band-hi (that is the
palette base's DC job). ``Law: segnet_stem_nyquist_alias_wall_v1`` bounds the bank support; the trunk
band = the stem transfer function's pass-band.

#300 GUARD / THE INVARIANT (witness-OWNED gradient-through, NOT a compose-addend of stored bytes):
the per-class coefficients ``W_tex`` are TRAINABLE MLX parameters of the witness that receive gradient
from the SEG loss every step — the trunk trains JOINTLY through the frozen SegNet, so the texture it
paints CANNOT flip argmax (the loss punishes any texture that does; Unit A measured that a region-wide
texture INJECTED without gradient flips argmax +0.228 — this trunk is the opposite of injection). The
deterministic Gabor BANK is a fixed non-parameter buffer (rule-118 FREE at decode, regenerable from
(H, W, band)); only the fitted ``W_tex`` coefficients are COUNTED. Byte-close = quantize ``W_tex``
(counted) + regenerate the bank (free). This is a normal witness submodule (byte-accounted like
``out_tex``/``palette``), default-OFF so ``model.parameters()`` / EMA / checkpoints / byte-close are
BYTE-IDENTICAL to the pre-trunk witness when the flag is not set (the ``film_per_layer`` idiom).

AUTHORITY: this module SYNTHESIZES texture + reports its DESIGN properties (band-limits, param count,
counted bytes). It makes NO score claim. A d_seg VERDICT is n600-only through
:func:`tac.through_r.measure_through_r` / byte-closed exact eval. The pointer (contest-CPU 0.19110) is
UNMOVED — everything here is MEANS. Numbers are ``[macOS advisory · research-signal · NON-PROMOTABLE]``.

COMPUTE-SUBSTRATE LAW (CLAUDE.md): the numpy bank + numpy reference forward are the PORTABLE
deterministic authority (no MLX needed); the MLX module mirrors them bit-for-bit by CONSTRUCTION
(``mx.array`` of the same numpy bank + the same blend algebra). NO MPS as a score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from tac.witness_control.perclass_verdict import N_CLASSES

__all__ = [
    "STEM_NYQUIST_PERIOD_PX",
    "TextureBandSpec",
    "TextureTrunkError",
    "band_limit_report",
    "build_gabor_bank_numpy",
    "counted_bytes",
    "default_band_spec",
    "make_texture_trunk_mlx",
    "texture_trunk_numpy_forward",
]

# The finest texture PERIOD (render/seg-input px) that survives the frozen SegNet stride-2 stem =
# 2·stride = 4 (MEASURED, tac.through_r.stem_perception.stem_nyquist). The band floor is pinned here;
# below it, texture aliases away BEFORE the first MBConv and carries zero class-discriminative signal.
STEM_NYQUIST_PERIOD_PX: float = 4.0


class TextureTrunkError(ValueError):
    """Raised on a mis-shaped / out-of-band / non-authority texture-trunk input."""


# --------------------------------------------------------------------------- #
# The band spec — clause-B: the bank support is the stem transfer pass-band.   #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TextureBandSpec:
    """The band-designed feature-bank spec. ``F`` = ``len(periods) * len(orientations) * n_phase``.

    * ``periods`` — texture PERIODS in render-px (≡ seg-input px, render grid is 384×512). Every entry
      MUST lie in ``[period_lo, period_hi]`` = the stem pass-band ``[STEM_NYQUIST_PERIOD_PX, band_hi]``.
      A period below ``STEM_NYQUIST_PERIOD_PX`` aliases away (dead); a period above ``band_hi`` reads as
      flat (the palette base's DC job) — both are REFUSED at construction (clause-B minimal-dim: the
      geometry's bound, not a guess).
    * ``orientations_deg`` — grating normals (deg). Orientation is a downstream (MBConv) property, so a
      small isotropic set (default 4) spans it at minimal cost.
    * ``n_phase`` — 1 (cosine only) or 2 (cosine + sine quadrature; captures arbitrary phase per
      period/orientation as a linear combination, so the fitted head is phase-free).
    * ``band_hi`` — the pass-band ceiling (default 8.0 render-px ≈ 2× the Nyquist; the price-list
      breadth the operator named "~4-8"). Kept explicit so the clause-B justification is queryable.
    """

    periods: tuple[float, ...] = (4.0, 6.0, 8.0)
    orientations_deg: tuple[float, ...] = (0.0, 45.0, 90.0, 135.0)
    n_phase: int = 2
    band_hi: float = 8.0

    def __post_init__(self) -> None:
        if not self.periods:
            raise TextureTrunkError("band spec needs >=1 period")
        if self.n_phase not in (1, 2):
            raise TextureTrunkError(f"n_phase must be 1 or 2, got {self.n_phase}")
        if self.band_hi < STEM_NYQUIST_PERIOD_PX:
            raise TextureTrunkError(
                f"band_hi {self.band_hi} < stem Nyquist {STEM_NYQUIST_PERIOD_PX} (empty pass-band)")
        for p in self.periods:
            if not (STEM_NYQUIST_PERIOD_PX <= float(p) <= self.band_hi + 1e-9):
                raise TextureTrunkError(
                    f"period {p} render-px is OUT OF the stem pass-band "
                    f"[{STEM_NYQUIST_PERIOD_PX}, {self.band_hi}] — periods below the Nyquist alias "
                    "away, periods above read as flat (palette base's DC job). Clause-B: the bank "
                    "support is the measured stem transfer pass-band, never a guess.")

    @property
    def n_features(self) -> int:
        """``F`` — the bank feature count = periods × orientations × phases."""
        return len(self.periods) * len(self.orientations_deg) * int(self.n_phase)


def default_band_spec() -> TextureBandSpec:
    """The canonical band spec: periods {4,6,8} × 4 orientations × cos/sin quadrature (F=24).

    Pinned to the MEASURED stem pass-band (``segnet_texture_perception_20260710.md``): floor at the
    period-4 Nyquist (the only period that flips Road/Lane through R), ceiling at period-8 (the
    operator's "~4-8"; period-16 read as flat in the price list)."""
    return TextureBandSpec()


# --------------------------------------------------------------------------- #
# The deterministic Gabor bank (rule-118 FREE — regenerable from (H, W, spec)).#
# --------------------------------------------------------------------------- #
def build_gabor_bank_numpy(h: int, w: int, spec: TextureBandSpec | None = None) -> np.ndarray:
    """The stationary oscillatory feature bank ``(H*W, F)`` float32 — deterministic, no RNG.

    Feature ``(period p, orientation θ, phase φ∈{0, π/2})`` at pixel ``(y, x)`` =
    ``cos(2π (x cosθ + y sinθ)/p − φ)`` over the render grid (row-major, matching
    ``_build_render_coords``: x fastest). Pure sinusoids ⇒ each feature is EXACTLY band-limited to its
    period and is ZERO-MEAN over the grid (texture is a zero-mean perturbation ON TOP of the palette
    DC). No randomness ⇒ bit-identical every host (rule-118 free table). Pixel coords are RENDER-px
    (0..W-1, 0..H-1) ≡ seg-input px (render grid = 384×512), so the periods are in seg-px directly."""
    if h < 2 or w < 2:
        raise TextureTrunkError(f"render grid must be >= 2x2, got ({h},{w})")
    spec = spec if spec is not None else default_band_spec()
    yy, xx = np.mgrid[0:h, 0:w]
    xf = xx.astype(np.float64).ravel()  # (P,) — x fastest (row-major), matches render coords
    yf = yy.astype(np.float64).ravel()
    phases = (0.0,) if spec.n_phase == 1 else (0.0, 0.5 * np.pi)
    cols: list[np.ndarray] = []
    for p in spec.periods:
        for o in spec.orientations_deg:
            th = np.radians(float(o))
            proj = xf * np.cos(th) + yf * np.sin(th)  # (P,) grating-normal coordinate
            base = 2.0 * np.pi * proj / float(p)
            for ph in phases:
                cols.append(np.cos(base - ph))
    bank = np.stack(cols, axis=-1).astype(np.float32)  # (P, F)
    assert bank.shape == (h * w, spec.n_features), (bank.shape, (h * w, spec.n_features))
    return bank


# --------------------------------------------------------------------------- #
# Band-limit + byte-accounting reports (design authority; NON-PROMOTABLE).     #
# --------------------------------------------------------------------------- #
def band_limit_report(h: int, w: int, spec: TextureBandSpec | None = None) -> dict[str, Any]:
    """Per-feature spectral peak period (render-px) via 2-D FFT — verifies EVERY feature is in-band.

    Each pure-sinusoid feature's 2-D FFT concentrates at a single |frequency|; the peak's period =
    ``1/|f|`` in px. Returns the min/max peak period across the bank + an ``all_in_band`` flag. This is
    the spectral proof that the bank carries NOTHING outside the measured stem pass-band."""
    spec = spec if spec is not None else default_band_spec()
    bank = build_gabor_bank_numpy(h, w, spec)  # (P, F)
    peaks: list[float] = []
    for j in range(bank.shape[1]):
        img = bank[:, j].reshape(h, w).astype(np.float64)
        img = img - img.mean()
        F = np.fft.fftshift(  # FFT_TOOL_USE_OK:measures texture spectrum only
            np.fft.fft2(img)  # FFT_TOOL_USE_OK:measures texture spectrum only
        )
        mag = np.abs(F)
        cy, cx = h // 2, w // 2
        mag[cy, cx] = 0.0
        py, px = np.unravel_index(int(np.argmax(mag)), mag.shape)
        fy = (py - cy) / float(h)
        fx = (px - cx) / float(w)
        fmag = float(np.hypot(fx, fy))
        peaks.append(1.0 / fmag if fmag > 0 else float("inf"))
    peaks_arr = np.asarray(peaks, dtype=np.float64)
    lo, hi = STEM_NYQUIST_PERIOD_PX, spec.band_hi
    # spectral bin resolution allows a small tolerance around the exact period
    tol = 1.0
    in_band = bool(np.all((peaks_arr >= lo - tol) & (peaks_arr <= hi + tol)))
    return {
        "n_features": int(spec.n_features),
        "peak_period_min_px": float(peaks_arr.min()),
        "peak_period_max_px": float(peaks_arr.max()),
        "band_lo_px": float(lo),
        "band_hi_px": float(hi),
        "all_in_band": in_band,
        "per_feature_peak_period_px": [float(x) for x in peaks_arr],
    }


def counted_bytes(spec: TextureBandSpec | None = None, *, n_classes: int = N_CLASSES,
                  quant_bits: int = 8) -> dict[str, Any]:
    """The COUNTED rate of the texture trunk = quantized per-class coefficients (bank is FREE).

    ``W_tex`` is ``(F, n_classes, 3)`` (+ ``(n_classes, 3)`` bias). Only these fitted coefficients are
    video-derived ⇒ counted; the Gabor bank regenerates from (H, W, spec) ⇒ rule-118 free. Reports raw
    bytes at ``quant_bits`` and the archive-rate-term contribution ``25·bytes/37_545_489`` so the
    matched-bytes A/B is exact. (Real byte-close would entropy-code these; this is the UNCODED upper
    bound.)"""
    spec = spec if spec is not None else default_band_spec()
    n_coeff = spec.n_features * int(n_classes) * 3 + int(n_classes) * 3  # weights + bias
    raw_bytes = n_coeff * float(quant_bits) / 8.0
    return {
        "n_features": int(spec.n_features),
        "n_classes": int(n_classes),
        "n_coeff": int(n_coeff),
        "quant_bits": int(quant_bits),
        "raw_bytes_uncoded": float(raw_bytes),
        "rate_term_uncoded_S": float(25.0 * raw_bytes / 37_545_489.0),
        "note": "counted = W_tex + bias only; the Gabor bank is rule-118 free (regenerable).",
    }


# --------------------------------------------------------------------------- #
# The numpy reference forward (portable authority; the MLX module mirrors it). #
# --------------------------------------------------------------------------- #
def _annulus_gate(soft: np.ndarray, n_classes: int, power: float) -> np.ndarray | None:
    """Margin-aware attenuation ``g`` in [0,1] from the softmax peak (compose-time boundary proxy).

    Near a decision boundary the softmax spreads across classes ⇒ ``peak = max_k soft_k`` → ~1/K;
    in a cell interior ``peak`` → 1. ``g = clip((peak-1/K)/(1-1/K), 0, 1)**power`` ⇒ attenuate texture
    in the ~4.7%-area boundary annulus (where injecting texture flips argmax — Unit A) and let it bloom
    in interiors. ``power == 0`` ⇒ no attenuation (returns None; the caller skips the multiply)."""
    if power <= 0.0:
        return None
    peak = soft.max(axis=-1)  # (...,)
    frac = 1.0 / float(n_classes)
    g = np.clip((peak - frac) / (1.0 - frac), 0.0, 1.0) ** float(power)
    return g[..., None]  # (..., 1) to broadcast over RGB


def texture_trunk_numpy_forward(
    bank: np.ndarray,
    w_tex: np.ndarray,
    soft: np.ndarray,
    *,
    bias: np.ndarray | None = None,
    n_classes: int = N_CLASSES,
    annulus_power: float = 0.0,
) -> np.ndarray:
    """The stationary texture-trunk forward: per-class band texture PLACED by the softmax masks.

    ``bank`` ``(P, F)`` fixed features; ``w_tex`` ``(F, K, 3)`` fitted per-class coefficients; ``soft``
    ``(..., P, K)`` the partition softmax masks (the SAME masks ``_compose_rgb`` computes). Per class
    ``k`` the band texture is ``bank @ w_tex[:,k,:]`` ``(P,3)``; the output blends them by the placement
    masks: ``texture[...,p,:] = Σ_k soft[...,p,k] · (bank@w_tex[:,k,:])[p]`` — placement-aware BY
    CONSTRUCTION. Optional ``annulus_power`` attenuates near the boundary. Returns ``(..., P, 3)`` to be
    ADDED into the pre-sigmoid ``base + out_tex`` term. Pure numpy (portable authority)."""
    bank = np.asarray(bank, dtype=np.float64)
    w_tex = np.asarray(w_tex, dtype=np.float64)
    soft = np.asarray(soft, dtype=np.float64)
    P, Fdim = bank.shape
    if w_tex.shape != (Fdim, n_classes, 3):
        raise TextureTrunkError(f"w_tex must be ({Fdim},{n_classes},3), got {w_tex.shape}")
    if soft.shape[-1] != n_classes or soft.shape[-2] != P:
        raise TextureTrunkError(f"soft must be (...,{P},{n_classes}), got {soft.shape}")
    class_tex = np.einsum("pf,fkc->pkc", bank, w_tex)  # (P, K, 3)
    if bias is not None:
        class_tex = class_tex + np.asarray(bias, dtype=np.float64)[None]  # (P,K,3) + (1,K,3)
    tex = np.einsum("...pk,pkc->...pc", soft, class_tex)  # (..., P, 3)
    g = _annulus_gate(soft, n_classes, annulus_power)
    if g is not None:
        tex = tex * g
    return tex.astype(np.float32)


# --------------------------------------------------------------------------- #
# The MLX witness submodule (byte-accounted like out_tex/palette).            #
# --------------------------------------------------------------------------- #
def make_texture_trunk_mlx(
    render_h: int,
    render_w: int,
    spec: TextureBandSpec | None = None,
    *,
    n_classes: int = N_CLASSES,
    annulus_power: float = 0.0,
    coeff_scale: float = 0.02,
    seed: int = 0,
):
    """Build the MLX ``TextureTrunk`` submodule. Returns an ``nn.Module`` with:

    * a FIXED Gabor bank buffer ``bank_B`` ``(P, F)`` — the ``_B`` suffix is the trainer's canonical
      rule-118 marker: ``measure_witness_blob_bytes`` / ``_quantize_blob_from_flat`` /
      ``_load_decoder_params`` ALL exclude ``*_B`` keys from the COUNTED byte-close blob (regenerated
      free at decode from (H,W,spec)), and ``freeze()`` keeps it out of the optimizer/gradient. Only
      ``w_tex``/``bias`` are counted;
    * trainable ``w_tex`` ``(F, K, 3)`` (small deterministic init, scale ``coeff_scale``) + ``bias``
      ``(K, 3)`` init 0 — the ONLY counted params;
    * ``__call__(soft) -> (..., P, 3)`` the additive pre-sigmoid texture term, mirroring
      :func:`texture_trunk_numpy_forward` in EXACT algebra (same bank, same blend; fp32 parity —
      the numpy reference runs fp64 then casts, so agreement is float32-level, not bit-exact).

    The init is small (``coeff_scale`` ~ ‖out_tex‖) and DETERMINISTIC (seeded numpy) so a run is
    reproducible. NO score claim; the trunk trains jointly through the seg loss (the #300 invariant)."""
    import mlx.core as mx
    import mlx.nn as nn

    spec = spec if spec is not None else default_band_spec()
    bank_np = build_gabor_bank_numpy(render_h, render_w, spec)  # (P, F)
    F = int(spec.n_features)
    rng = np.random.default_rng(int(seed))
    # small deterministic init in [-coeff_scale, coeff_scale] — texture starts as a faint perturbation
    w_init = (rng.standard_normal((F, int(n_classes), 3)).astype(np.float32) * float(coeff_scale))

    class TextureTrunk(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.n_classes = int(n_classes)
            self.annulus_power = float(annulus_power)
            self.n_features = F
            self.render_h = int(render_h)
            self.render_w = int(render_w)
            # FROZEN bank: a fixed rule-118 buffer. The ``_B`` suffix is the trainer's canonical
            # free-table marker (measure_witness_blob_bytes / _quantize_blob_from_flat /
            # _load_decoder_params all skip ``*_B`` => NOT counted, regenerated free at decode).
            # freeze() additionally keeps it out of the optimizer/gradient. (EMA still tracks it like
            # the base witness's ``_B`` Fourier table — EMA of a constant stays the constant exactly.)
            self.bank_B = mx.array(bank_np)
            self.w_tex = mx.array(w_init)          # (F, K, 3) COUNTED
            self.bias = mx.zeros((int(n_classes), 3))  # (K, 3) COUNTED, init 0
            self.freeze(keys=["bank_B"])

        def __call__(self, soft):
            # soft: (..., P, K). class_tex: (P, K, 3) = bank @ w_tex + bias.
            class_tex = mx.einsum("pf,fkc->pkc", self.bank_B, self.w_tex) + self.bias[None]
            tex = mx.einsum("...pk,pkc->...pc", soft, class_tex)  # (..., P, 3)
            if self.annulus_power > 0.0:
                peak = mx.max(soft, axis=-1)  # (..., P)
                frac = 1.0 / float(self.n_classes)
                g = mx.clip((peak - frac) / (1.0 - frac), 0.0, 1.0) ** self.annulus_power
                tex = tex * g[..., None]
            return tex

    return TextureTrunk()
