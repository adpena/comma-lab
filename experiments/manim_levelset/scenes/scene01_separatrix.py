"""Scene 1 — THE SEPARATRIX.

A 3Blue1Brown-style animation of the single most central object in our level-set
witness: the SegNet argmax partition and the codim-1 boundary (the "separatrix")
that literally IS the d_seg metric.

The math shown is FAITHFUL, not decorative:

  - 5 class LOGIT fields z_k(x) are quadratics (the log of Gaussian class models).
  - softmax_τ(z) = softmax(z / τ) is the smooth (temperature-τ) class posterior.
  - As τ → 0 the smooth blend crystallizes into HARD argmax cells. Because the
    logits are quadratic, that τ→0 partition is exactly a POWER / LAGUERRE diagram
    — which is our measured `deepmath #284` result: the τ→0 witness is tropical
    (max-plus), and its cells are Laguerre cells. Nothing here is faked.
  - The cell boundaries form the SEPARATRIX: a codim-1 set (a CURVE in 2D, a
    SURFACE in the real 512×384 image). d_seg = the pixels where our witness's
    argmax disagrees with the scorer's — i.e. it lives ON this curve, not in the
    2D volume. A curve is cheap to describe: that is the whole task-space thesis.
  - τ = ε = ħ: the softmax temperature is simultaneously the viscosity ε of the
    level-set flow and the semiclassical scale ħ (deepmath #284).

Render (isolated venv, no LaTeX needed — Pango/Unicode typography):

    cd experiments/manim_levelset
    .venv/bin/manim -qm scenes/scene01_separatrix.py Separatrix
"""
from __future__ import annotations

import numpy as np
from manim import (
    BLACK, WHITE, GREY_B, YELLOW, config,
    Scene, ImageMobject, Text, VGroup, ValueTracker, always_redraw,
    FadeIn, FadeOut, Write, UP, DOWN, LEFT, RIGHT, ORIGIN, rate_functions,
)

config.background_color = BLACK

# ── palette — a calm, 3B1B-adjacent 5-class scheme (RGB 0-255) ───────────────
CLASS_RGB = np.array(
    [
        [ 45,  92, 168],   # 0 road      — deep blue
        [232, 197,  71],   # 1 lane      — warm gold (the fragile rare class)
        [ 66, 132, 121],   # 2 undrivable— teal
        [196,  84,  73],   # 3 movable   — clay red
        [120,  96, 162],   # 4 my-car    — muted violet
    ],
    dtype=np.float64,
)

# ── the 5 quadratic logit fields on a fixed grid (computed ONCE) ─────────────
_N = 360                                  # field resolution
_XS = np.linspace(-3.2, 3.2, _N)
_YS = np.linspace(-2.0, 2.0, _N)
_GX, _GY = np.meshgrid(_XS, _YS)          # (N, N)

# class centers + precisions + biases → z_k = -0.5*a_k*||x-c_k||^2 + b_k.
# (asymmetric a_k, b_k make the τ→0 tessellation a genuine LAGUERRE / power
#  diagram with curved, non-perpendicular-bisector edges — not a plain Voronoi.)
_CENTERS = np.array([[-1.7, -0.7], [0.15, 0.55], [1.85, 0.35], [0.9, -1.05], [-0.6, 1.0]])
_PREC = np.array([1.00, 1.65, 0.85, 1.25, 1.10])      # a_k
_BIAS = np.array([0.35, -0.05, 0.25, 0.10, -0.15])    # b_k

_Z = np.stack(
    [
        -0.5 * _PREC[k] * ((_GX - _CENTERS[k, 0]) ** 2 + (_GY - _CENTERS[k, 1]) ** 2) + _BIAS[k]
        for k in range(5)
    ],
    axis=-1,
)                                          # (N, N, 5)

_ARGMAX = _Z.argmax(axis=-1)               # τ→0 partition (constant in τ)
# separatrix mask: a pixel whose argmax differs from a 4-neighbour → boundary
_EDGE = np.zeros((_N, _N), dtype=bool)
_EDGE[:-1, :] |= _ARGMAX[:-1, :] != _ARGMAX[1:, :]
_EDGE[1:, :] |= _ARGMAX[:-1, :] != _ARGMAX[1:, :]
_EDGE[:, :-1] |= _ARGMAX[:, :-1] != _ARGMAX[:, 1:]
_EDGE[:, 1:] |= _ARGMAX[:, :-1] != _ARGMAX[:, 1:]
_EDGE_RGB = np.array([90, 240, 255], dtype=np.float64)   # bright cyan separatrix


def _softmax_rgb(tau: float, glow: float) -> np.ndarray:
    """Blend the 5 class colours by the temperature-τ posterior, then paint the
    separatrix on top with intensity `glow`. Returns a (N, N, 3) uint8 image."""
    z = _Z / max(tau, 1e-4)
    z -= z.max(axis=-1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=-1, keepdims=True)                    # (N, N, 5) posterior
    rgb = np.tensordot(p, CLASS_RGB, axes=([2], [0]))     # (N, N, 3) blended
    if glow > 0:
        g = glow * _EDGE[..., None]
        rgb = (1.0 - g) * rgb + g * _EDGE_RGB
    return np.clip(rgb, 0, 255).astype(np.uint8)


class Separatrix(Scene):
    def construct(self) -> None:
        tau = ValueTracker(1.0)
        glow = ValueTracker(0.0)

        # the live field image, redrawn every frame from the current (τ, glow)
        def _field() -> ImageMobject:
            img = ImageMobject(_softmax_rgb(tau.get_value(), glow.get_value()))
            img.height = 6.0
            img.move_to(0.35 * DOWN)
            return img

        field = always_redraw(_field)

        # ── title ────────────────────────────────────────────────────────────
        title = Text("The Separatrix", font="Helvetica Neue", weight="BOLD").scale(0.95)
        subtitle = Text(
            "the boundary that IS the score",
            font="Helvetica Neue", color=GREY_B,
        ).scale(0.42).next_to(title, DOWN, buff=0.18)
        titlecard = VGroup(title, subtitle).move_to(ORIGIN)
        self.play(Write(title), run_time=1.1)
        self.play(FadeIn(subtitle, shift=0.2 * UP), run_time=0.7)
        self.wait(0.5)
        self.play(FadeOut(titlecard), run_time=0.6)

        # ── the smooth field appears (τ = 1) ─────────────────────────────────
        self.add(field)
        self.play(FadeIn(field), run_time=0.8)

        cap = Text(
            "softmaxτ  of 5 class logits  over ℝ²",
            font="Helvetica Neue", color=WHITE,
        ).scale(0.44).to_edge(UP, buff=0.45)
        self.play(FadeIn(cap, shift=0.2 * DOWN), run_time=0.7)

        # live τ readout (top-right)
        tau_label = always_redraw(
            lambda: Text(
                f"τ = {tau.get_value():.2f}",
                font="Menlo", color=YELLOW,
            ).scale(0.5).to_corner(UP + RIGHT, buff=0.5)
        )
        self.play(FadeIn(tau_label), run_time=0.5)
        self.wait(0.6)

        # ── anneal τ → 0 : the blend crystallizes into hard cells ─────────────
        anneal = Text(
            "τ → 0", font="Menlo", color=YELLOW,
        ).scale(0.55).next_to(cap, DOWN, buff=0.25)
        self.play(FadeIn(anneal), run_time=0.4)
        self.play(
            tau.animate.set_value(0.03),
            run_time=4.5, rate_func=rate_functions.ease_in_out_sine,
        )
        self.wait(0.4)

        argmax_cap = Text(
            "the argmax partition   ·   softmax(quadratics) → Laguerre / power diagram",
            font="Helvetica Neue", color=GREY_B,
        ).scale(0.36).to_edge(DOWN, buff=0.35)
        self.play(FadeOut(anneal), FadeIn(argmax_cap, shift=0.15 * UP), run_time=0.8)
        self.wait(0.6)

        # ── reveal the separatrix (glow up the codim-1 boundary) ─────────────
        self.play(glow.animate.set_value(1.0), run_time=1.6,
                  rate_func=rate_functions.ease_out_cubic)
        sep_cap = Text(
            "the separatrix  ·  codim-1  ·  this curve IS d_seg",
            font="Helvetica Neue", color=WHITE,
        ).scale(0.42).to_edge(DOWN, buff=0.35)
        self.play(FadeOut(argmax_cap), FadeIn(sep_cap, shift=0.15 * UP), run_time=0.8)
        # pulse the glow to draw the eye to the boundary
        self.play(glow.animate.set_value(0.55), run_time=0.8, rate_func=rate_functions.there_and_back)
        self.play(glow.animate.set_value(1.0), run_time=0.6)
        self.wait(1.0)

        # ── the thesis card ──────────────────────────────────────────────────
        self.play(FadeOut(cap), FadeOut(tau_label), FadeOut(sep_cap), run_time=0.7)
        self.play(field.animate.set_opacity(0.28), run_time=0.8)

        eq = Text("τ  =  ε  =  ℏ", font="Menlo", weight="BOLD").scale(1.15)
        eq_sub = Text(
            "softmax temperature  =  level-set viscosity  =  semiclassical scale",
            font="Helvetica Neue", color=GREY_B,
        ).scale(0.38).next_to(eq, DOWN, buff=0.22)
        thesis = Text(
            "a boundary is a curve, not a volume  →  cheap to describe",
            font="Helvetica Neue", color=WHITE,
        ).scale(0.44).next_to(eq_sub, DOWN, buff=0.5)
        thesis2 = Text(
            "…this is the task-space witness.",
            font="Helvetica Neue", color=YELLOW,
        ).scale(0.40).next_to(thesis, DOWN, buff=0.22)

        self.play(Write(eq), run_time=1.2)
        self.play(FadeIn(eq_sub, shift=0.15 * UP), run_time=0.7)
        self.wait(0.3)
        self.play(FadeIn(thesis, shift=0.15 * UP), run_time=0.8)
        self.play(FadeIn(thesis2, shift=0.15 * UP), run_time=0.7)
        self.wait(2.0)
        self.play(
            FadeOut(VGroup(eq, eq_sub, thesis, thesis2)), FadeOut(field), run_time=1.0
        )
        self.wait(0.4)
