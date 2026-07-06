"""Scene 1 — THE SEPARATRIX  (abstract intro, resolving into openpilot).

3Blue1Brown-style intro to the central object of our level-set witness. Clean
layout grammar (one top equation, one bottom caption — never stacked) and a
deliberate one-idea-per-beat pace.

FAITHFUL math (NO-FAKE): quadratic logits → softmax_τ → τ→0 gives a LAGUERRE /
power diagram (deepmath #284); its boundaries are the codim-1 separatrix = d_seg;
the abstract cells then RESOLVE into the real comma10k / openpilot segmentation
of frame 196; τ = ε = ℏ.

Render:  ./render.sh -qh scenes/scene01_separatrix.py Separatrix
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from manim import (
    Scene, ImageMobject, Square, VGroup, ValueTracker, always_redraw,
    FadeIn, FadeOut, Write, GrowFromCenter, UP, DOWN, RIGHT, ORIGIN, rate_functions,
)

import _style as st

_ASSETS = Path(__file__).resolve().parent.parent / "assets"

# per-frame separatrix stack — the abstract partition RESOLVES into the real one,
# then plays LIVE on the moving footage (same stack scenes 2/3 use)
_EGO = np.load(_ASSETS / "ego_clip.npy")
_SEP_STACK = np.load(_ASSETS / "ego_sep_stack.npy")
_KS = len(_EGO)
_HS = _KS // 2

# ── 5 quadratic logit fields on a fixed grid (computed ONCE) ─────────────────
_N = 360
_XS, _YS = np.linspace(-3.2, 3.2, _N), np.linspace(-2.0, 2.0, _N)
_GX, _GY = np.meshgrid(_XS, _YS)
_CENTERS = np.array([[-1.7, -0.7], [0.15, 0.55], [1.85, 0.35], [0.9, -1.05], [-0.6, 1.0]])
_PREC = np.array([1.00, 1.65, 0.85, 1.25, 1.10])
_BIAS = np.array([0.35, -0.05, 0.25, 0.10, -0.15])
_Z = np.stack([-0.5 * _PREC[k] * ((_GX - _CENTERS[k, 0]) ** 2 + (_GY - _CENTERS[k, 1]) ** 2) + _BIAS[k]
               for k in range(5)], axis=-1)
_ARGMAX = _Z.argmax(axis=-1)
_EDGE = np.zeros((_N, _N), dtype=bool)
_EDGE[:-1, :] |= _ARGMAX[:-1, :] != _ARGMAX[1:, :]
_EDGE[1:, :] |= _ARGMAX[:-1, :] != _ARGMAX[1:, :]
_EDGE[:, :-1] |= _ARGMAX[:, :-1] != _ARGMAX[:, 1:]
_EDGE[:, 1:] |= _ARGMAX[:, :-1] != _ARGMAX[:, 1:]
_EDGE_RGB = np.array([79, 214, 224], dtype=np.float64)

_FIELD_H = 4.7
_FIELD_POS = 0.15 * DOWN     # sits low so the top equation + bottom caption bands are clear


def _softmax_rgb(tau: float, glow: float) -> np.ndarray:
    z = _Z / max(tau, 1e-4)
    z -= z.max(axis=-1, keepdims=True)
    p = np.exp(z); p /= p.sum(axis=-1, keepdims=True)
    rgb = np.tensordot(p, st.COMMA10K_RGB, axes=([2], [0]))
    if glow > 0:
        g = glow * _EDGE[..., None]
        rgb = (1.0 - g) * rgb + g * _EDGE_RGB
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _legend():
    rows = VGroup()
    for hexc, name in zip(st.COMMA10K_HEX, st.COMMA10K_LABEL):
        sw = Square(side_length=0.22, fill_color=hexc, fill_opacity=1.0, stroke_width=0)
        lbl = st.caption(name, color=st.MUTED, scale=0.30).next_to(sw, RIGHT, buff=0.12)
        rows.add(VGroup(sw, lbl))
    return rows.arrange(RIGHT, buff=0.46)


class Separatrix(Scene):
    def construct(self) -> None:
        tau, glow = ValueTracker(1.0), ValueTracker(0.0)
        field = always_redraw(
            lambda: ImageMobject(_softmax_rgb(tau.get_value(), glow.get_value()))
            .set(height=_FIELD_H).move_to(_FIELD_POS)
        )

        # ── beat 1 · title ───────────────────────────────────────────────────
        card = st.titlecard("01 · the separatrix", "The Separatrix",
                            "the boundary that is the score").move_to(ORIGIN)
        self.play(Write(card[1]), run_time=st.T_WRITE)
        self.play(FadeIn(card[0], shift=0.15 * DOWN), GrowFromCenter(card[2]), run_time=st.T_FADE)
        self.play(FadeIn(card[3], shift=0.1 * UP), run_time=st.T_FADE)
        self.wait(st.HOLD)
        self.play(FadeOut(card), run_time=st.T_FADE)

        # ── beat 2 · the smooth field + the softmax law ─────────────────────
        self.add(field)
        self.play(FadeIn(field), run_time=st.T_FADE)
        eq = st.top(st.eq(
            r"p_k(\mathbf{x};\tau)=\frac{e^{\,z_k(\mathbf{x})/\tau}}{\sum_j e^{\,z_j(\mathbf{x})/\tau}}"
        ))
        self.play(Write(eq), run_time=st.T_WRITE)
        tau_ro = always_redraw(lambda: st.corner_tr(st.mono(f"τ = {tau.get_value():.2f}")))
        self.play(FadeIn(tau_ro), run_time=st.T_FADE)
        self.wait(st.HOLD)

        # ── beat 3 · anneal τ → 0  (the hero move) ──────────────────────────
        anneal = st.eq(r"\tau \to 0", color=st.GOLD).scale(1.05).next_to(eq, DOWN, buff=0.28)
        self.play(Write(anneal), run_time=st.T_FADE)
        self.play(tau.animate.set_value(0.03), run_time=st.T_HERO,
                  rate_func=rate_functions.ease_in_out_sine)
        self.play(FadeOut(anneal), run_time=st.T_FADE)
        cap = st.bottom(st.caption("the argmax partition — a Laguerre / power diagram"))
        self.play(FadeIn(cap, shift=0.12 * UP), run_time=st.T_FADE)
        self.wait(st.HOLD)

        # ── beat 4 · the separatrix ─────────────────────────────────────────
        self.play(glow.animate.set_value(1.0), run_time=st.T_MORPH, rate_func=rate_functions.ease_out_cubic)
        eq2 = st.top(st.eq(r"\partial\Omega \ \Longleftrightarrow\ d_{\mathrm{seg}}"))
        cap2 = st.bottom(st.caption("its boundaries are the separatrix — where d_seg lives", color=st.INK))
        self.play(FadeOut(eq), FadeOut(cap), Write(eq2), FadeIn(cap2, shift=0.12 * UP), run_time=st.T_MORPH)
        self.play(glow.animate.set_value(0.5), run_time=st.T_FADE, rate_func=rate_functions.there_and_back)
        self.play(glow.animate.set_value(1.0), run_time=st.T_FADE)
        self.wait(st.HOLD)

        # ── beat 5 · resolve into openpilot — then WATCH IT MOVE ────────────
        ci_t = ValueTracker(float(_HS))

        def _sidx() -> int:
            return int(np.clip(ci_t.get_value(), 0, _KS - 1))

        live_seg = always_redraw(
            lambda: ImageMobject(_SEP_STACK[_sidx()]).set(height=_FIELD_H).move_to(_FIELD_POS))
        kick = st.top(st.kicker("the same partition — on a real road"))
        self.play(FadeOut(eq2), FadeOut(cap2), FadeOut(tau_ro), run_time=st.T_FADE)
        self.add(live_seg)                                 # opaque — covers the abstract field
        self.play(FadeIn(live_seg), FadeIn(kick, shift=0.1 * DOWN),
                  run_time=st.T_MORPH, rate_func=rate_functions.ease_in_out_sine)
        self.remove(field)
        legend = st.bottom(_legend())
        self.play(FadeIn(legend, lag_ratio=0.12), run_time=st.T_WRITE)
        self.wait(st.HOLD)
        # the abstract idea IS the real partition — and now it MOVES with the scene
        movecap = st.bottom(st.caption(
            "the same partition, live — it tracks the road as the scene flows", color=st.INK))
        self.play(FadeOut(legend), FadeIn(movecap, shift=0.1 * UP), run_time=st.T_FADE)
        self.play(ci_t.animate.set_value(0.0), run_time=st.T_FADE, rate_func=rate_functions.ease_in_out_sine)
        self.play(ci_t.animate.set_value(_KS - 1), run_time=8.0, rate_func=rate_functions.linear)
        self.play(ci_t.animate.set_value(float(_HS)), run_time=st.T_MORPH, rate_func=rate_functions.ease_in_out_sine)
        self.wait(st.BEAT)
        # freeze to a static snapshot for the thesis
        seg = ImageMobject(_SEP_STACK[_HS]).set(height=_FIELD_H).move_to(_FIELD_POS)
        self.add(seg); self.bring_to_back(seg); self.remove(live_seg)

        # ── beat 6 · thesis ─────────────────────────────────────────────────
        self.play(FadeOut(kick), FadeOut(movecap), seg.animate.set_opacity(0.20), run_time=st.T_FADE)
        e = st.eq(r"\tau \;=\; \varepsilon \;=\; \hbar", scale=1.7)
        e_sub = st.caption("softmax temperature = level-set viscosity = semiclassical scale").next_to(e, DOWN, buff=0.3)
        thesis = st.body("a boundary is a curve, not a volume  →  cheap to describe").next_to(e_sub, DOWN, buff=0.55)
        hero = st.hero("the task-space witness", scale=0.6).next_to(thesis, DOWN, buff=0.3)
        self.play(Write(e), run_time=st.T_WRITE)
        self.play(FadeIn(e_sub, shift=0.1 * UP), run_time=st.T_FADE)
        self.wait(st.BEAT)
        self.play(FadeIn(thesis, shift=0.1 * UP), run_time=st.T_FADE)
        self.play(FadeIn(hero, shift=0.1 * UP), run_time=st.T_FADE)
        self.wait(st.HOLD_LONG)
        self.play(FadeOut(VGroup(e, e_sub, thesis, hero)), FadeOut(seg), run_time=st.T_MORPH)
        self.wait(st.BEAT)
