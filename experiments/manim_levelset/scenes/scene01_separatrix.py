"""Scene 1 — THE SEPARATRIX  (the abstract INTRO, resolving into openpilot).

3Blue1Brown-style intro to the central object of our level-set witness: the
SegNet argmax partition and the codim-1 separatrix that IS d_seg — abstract
softmax-τ math that then RESOLVES into the real openpilot / comma10k
segmentation of the hardest contest frame (bridging straight into scene 2).

FAITHFUL math (NO-FAKE):
  - z_k(x) quadratic logits → softmax_τ → anneal τ→0 → the argmax partition is a
    LAGUERRE / power diagram (our deepmath #284 result).
  - the cell boundaries = the SEPARATRIX (codim-1) = d_seg.
  - the abstract cells then dissolve into the REAL comma10k segmentation of
    frame 196 (its actual colors + labels).
  - τ = ε = ℏ (softmax temperature = level-set viscosity = semiclassical scale).

Render:  ./render.sh -qh scenes/scene01_separatrix.py Separatrix
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from manim import (
    Scene, ImageMobject, Text, VGroup, Square, ValueTracker, always_redraw,
    FadeIn, FadeOut, Write, GrowFromCenter, UP, DOWN, LEFT, RIGHT, ORIGIN,
    rate_functions,
)

import _style as st

_ASSETS = Path(__file__).resolve().parent.parent / "assets"

# ── the 5 quadratic logit fields on a fixed grid (computed ONCE) ─────────────
_N = 360
_XS = np.linspace(-3.2, 3.2, _N)
_YS = np.linspace(-2.0, 2.0, _N)
_GX, _GY = np.meshgrid(_XS, _YS)
_CENTERS = np.array([[-1.7, -0.7], [0.15, 0.55], [1.85, 0.35], [0.9, -1.05], [-0.6, 1.0]])
_PREC = np.array([1.00, 1.65, 0.85, 1.25, 1.10])
_BIAS = np.array([0.35, -0.05, 0.25, 0.10, -0.15])
_Z = np.stack(
    [-0.5 * _PREC[k] * ((_GX - _CENTERS[k, 0]) ** 2 + (_GY - _CENTERS[k, 1]) ** 2) + _BIAS[k]
     for k in range(5)], axis=-1,
)
_ARGMAX = _Z.argmax(axis=-1)
_EDGE = np.zeros((_N, _N), dtype=bool)
_EDGE[:-1, :] |= _ARGMAX[:-1, :] != _ARGMAX[1:, :]
_EDGE[1:, :] |= _ARGMAX[:-1, :] != _ARGMAX[1:, :]
_EDGE[:, :-1] |= _ARGMAX[:, :-1] != _ARGMAX[:, 1:]
_EDGE[:, 1:] |= _ARGMAX[:, :-1] != _ARGMAX[:, 1:]
_EDGE_RGB = np.array([79, 214, 224], dtype=np.float64)   # st.CYAN separatrix


def _softmax_rgb(tau: float, glow: float) -> np.ndarray:
    z = _Z / max(tau, 1e-4)
    z -= z.max(axis=-1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=-1, keepdims=True)
    rgb = np.tensordot(p, st.COMMA10K_RGB, axes=([2], [0]))   # comma10k palette
    if glow > 0:
        g = glow * _EDGE[..., None]
        rgb = (1.0 - g) * rgb + g * _EDGE_RGB
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _legend():
    """openpilot / comma10k class swatches + labels."""
    rows = VGroup()
    for hexc, name in zip(st.COMMA10K_HEX, st.COMMA10K_LABEL):
        sw = Square(side_length=0.24, fill_color=hexc, fill_opacity=1.0,
                    stroke_width=0)
        lbl = Text(name, font=st.FONT, color=st.MUTED).scale(0.30).next_to(sw, RIGHT, buff=0.12)
        rows.add(VGroup(sw, lbl))
    return rows.arrange(RIGHT, buff=0.5)


class Separatrix(Scene):
    def construct(self) -> None:
        tau = ValueTracker(1.0)
        glow = ValueTracker(0.0)

        def _field() -> ImageMobject:
            img = ImageMobject(_softmax_rgb(tau.get_value(), glow.get_value()))
            img.height = 5.9
            img.move_to(0.3 * DOWN)
            return img

        field = always_redraw(_field)

        # ── title ────────────────────────────────────────────────────────────
        card = st.titlecard("01 · the separatrix", "The Separatrix",
                            "the boundary that is the score").move_to(ORIGIN)
        self.play(Write(card[1]), run_time=1.0)
        self.play(FadeIn(card[0], shift=0.15 * DOWN), GrowFromCenter(card[2]), run_time=0.6)
        self.play(FadeIn(card[3], shift=0.1 * UP), run_time=0.5)
        self.wait(0.6)
        self.play(FadeOut(card), run_time=0.6)

        # ── smooth field (τ=1) + softmax equation ────────────────────────────
        self.add(field)
        self.play(FadeIn(field), run_time=0.8)
        softmax_eq = st.eq(
            r"p_k(\mathbf{x};\tau)=\frac{e^{\,z_k(\mathbf{x})/\tau}}"
            r"{\sum_j e^{\,z_j(\mathbf{x})/\tau}}",
        ).to_edge(UP, buff=0.42)
        self.play(Write(softmax_eq), run_time=1.3)
        tau_label = always_redraw(
            lambda: st.mono(f"τ = {tau.get_value():.2f}").to_corner(UP + RIGHT, buff=0.5)
        )
        self.play(FadeIn(tau_label), run_time=0.5)
        self.wait(0.5)

        # ── anneal τ → 0 ─────────────────────────────────────────────────────
        anneal = st.eq(r"\tau \to 0", color=st.GOLD).scale(1.05).next_to(softmax_eq, DOWN, buff=0.3)
        self.play(Write(anneal), run_time=0.5)
        self.play(tau.animate.set_value(0.03), run_time=4.3,
                  rate_func=rate_functions.ease_in_out_sine)
        self.wait(0.3)
        argmax_eq = st.eq(
            r"\lim_{\tau\to 0}\ \mathrm{softmax}_\tau(z)=\mathbb{1}\!\left[\arg\max_k z_k\right]",
            scale=0.62).to_edge(DOWN, buff=0.9)
        laguerre = st.caption("softmax of quadratics  →  Laguerre / power diagram").to_edge(DOWN, buff=0.42)
        self.play(FadeOut(anneal), Write(argmax_eq), run_time=1.0)
        self.play(FadeIn(laguerre, shift=0.1 * UP), run_time=0.5)
        self.wait(0.5)

        # ── reveal the separatrix ────────────────────────────────────────────
        self.play(glow.animate.set_value(1.0), run_time=1.5, rate_func=rate_functions.ease_out_cubic)
        sep_eq = st.eq(r"\partial\Omega\ \Longleftrightarrow\ d_{\mathrm{seg}}",
                       scale=0.66).to_edge(DOWN, buff=0.85)
        sep_cap = st.caption("codim-1 · this curve is d_seg", color=st.INK).to_edge(DOWN, buff=0.42)
        self.play(FadeOut(argmax_eq), FadeOut(laguerre), Write(sep_eq),
                  FadeIn(sep_cap, shift=0.1 * UP), run_time=1.0)
        self.play(glow.animate.set_value(0.55), run_time=0.7, rate_func=rate_functions.there_and_back)
        self.play(glow.animate.set_value(1.0), run_time=0.5)
        self.wait(0.7)

        # ── RESOLVE into the real openpilot / comma10k segmentation ──────────
        seg = ImageMobject(str(_ASSETS / "hardest_separatrix.png"))
        seg.height = 5.9
        seg.move_to(0.3 * DOWN)
        seg.set_opacity(0.0)
        self.add(seg)
        resolve_kick = st.kicker("the same partition — on a real road").to_edge(UP, buff=0.45)
        self.play(FadeOut(softmax_eq), FadeOut(sep_eq), FadeOut(tau_label),
                  FadeOut(sep_cap), run_time=0.5)
        self.play(field.animate.set_opacity(0.0), seg.animate.set_opacity(1.0),
                  FadeIn(resolve_kick, shift=0.1 * DOWN), run_time=1.6,
                  rate_func=rate_functions.ease_in_out_sine)
        op_cap = st.caption("openpilot segmentation · comma10k", color=st.INK).to_edge(DOWN, buff=0.42)
        legend = _legend().next_to(op_cap, UP, buff=0.28)
        self.play(FadeIn(op_cap, shift=0.1 * UP), run_time=0.5)
        self.play(FadeIn(legend, lag_ratio=0.12), run_time=1.0)
        self.wait(1.4)

        # ── thesis card ──────────────────────────────────────────────────────
        self.play(FadeOut(resolve_kick), FadeOut(op_cap), FadeOut(legend),
                  seg.animate.set_opacity(0.22), run_time=0.8)
        eq = st.eq(r"\tau \;=\; \varepsilon \;=\; \hbar", scale=1.7)
        eq_sub = st.caption("softmax temperature  =  level-set viscosity  =  semiclassical scale").next_to(eq, DOWN, buff=0.28)
        thesis = st.body("a boundary is a curve, not a volume  →  cheap to describe").next_to(eq_sub, DOWN, buff=0.5)
        hero = st.hero("the task-space witness", scale=0.6).next_to(thesis, DOWN, buff=0.28)
        self.play(Write(eq), run_time=1.3)
        self.play(FadeIn(eq_sub, shift=0.12 * UP), run_time=0.6)
        self.wait(0.2)
        self.play(FadeIn(thesis, shift=0.12 * UP), run_time=0.7)
        self.play(FadeIn(hero, shift=0.12 * UP), run_time=0.7)
        self.wait(2.0)
        self.play(FadeOut(VGroup(eq, eq_sub, thesis, hero)), FadeOut(seg), run_time=1.0)
        self.wait(0.3)
