"""Scene 3 — ONE TWIST WARPS THE WORLD  (the se(3) screw).

The dramatic 3D beat. FAITHFUL math (our measured findings, deepmath / pose
dual-use):

  - Chasles' theorem: every rigid-body motion is a SCREW — a rotation about an
    axis plus a translation along it. The camera's ego-motion between the two
    frames of a scored pair is one such screw.
  - It is a single se(3) TWIST  ξ = (v, ω) ∈ ℝ⁶,  with  exp(ξ̂) ∈ SE(3).
  - d_pose = MSE on the 6 PoseNet scalars — which IS this twist ξ (identifiable
    up to affine). So encoding ξ solves the pose axis.
  - The SAME ξ generates the optical-flow warp that carries the segmentation
    frame₀ → frame₁ — the temporal-consistency lever for d_seg.
  - So: encode ξ ONCE (≈1–2 KB, FiLM-conditioned), it serves BOTH scored axes.
    seg ⊥ pose → free to combine.

Render:  ./render.sh -qh scenes/scene03_screw.py Screw
"""
from __future__ import annotations

import numpy as np
from manim import (
    ThreeDScene, ParametricFunction, Line, Dot3D, VGroup, ValueTracker,
    always_redraw, FadeIn, FadeOut, Write, Create, Arrow, UP, DOWN, LEFT, RIGHT,
    ORIGIN, rate_functions,
)

import _style as st

_R = 1.15          # helix radius
_PITCH = 0.42      # translation per radian along the axis
_TURNS = 3.1       # number of turns
_T0, _T1 = -_TURNS * np.pi, _TURNS * np.pi


def _helix(t: float):
    return np.array([_R * np.cos(t), _R * np.sin(t), _PITCH * t])


class Screw(ThreeDScene):
    def construct(self) -> None:
        self.set_camera_orientation(phi=68 * np.pi / 180, theta=-50 * np.pi / 180, zoom=0.95)

        # ── fixed 2D title (HUD) ─────────────────────────────────────────────
        card = st.titlecard("03 · the screw", "One Twist Warps the World",
                            "every rigid motion is a screw").move_to(ORIGIN)
        self.add_fixed_in_frame_mobjects(card)
        card.set_opacity(0)
        self.play(card.animate.set_opacity(1), run_time=st.T_WRITE)
        self.wait(st.HOLD)
        self.play(FadeOut(card), run_time=st.T_FADE)

        # ── the screw: axis + helix ─────────────────────────────────────────
        axis = Line(_helix(_T0) * [0, 0, 1], _helix(_T1) * [0, 0, 1],
                    color=st.GOLD, stroke_width=3)
        helix = ParametricFunction(_helix, t_range=[_T0, _T1, 0.05],
                                   color=st.CYAN, stroke_width=4)
        self.play(Create(axis), run_time=st.T_FADE)
        self.begin_ambient_camera_rotation(rate=0.14)
        self.play(Create(helix), run_time=st.T_HERO * 0.8, rate_func=rate_functions.ease_in_out_sine)

        # a frame (the camera/car) screwing along the helix
        t = ValueTracker(_T0)
        bead = always_redraw(lambda: Dot3D(_helix(t.get_value()), radius=0.13, color=st.CORAL))
        self.add(bead)
        self.play(t.animate.set_value(_T1), run_time=st.T_HERO, rate_func=rate_functions.ease_in_out_sine)
        self.wait(st.BEAT)

        # ── the twist ξ (fixed HUD equation) ────────────────────────────────
        xi = st.eq(r"\xi=(\mathbf{v},\boldsymbol{\omega})\in\mathfrak{se}(3),"
                   r"\quad e^{\hat\xi}\in SE(3)", scale=0.62)
        st.top(xi)
        self.add_fixed_in_frame_mobjects(xi); xi.set_opacity(0)
        chasles = st.bottom(st.caption("Chasles — rotation about an axis + translation along it"))
        self.add_fixed_in_frame_mobjects(chasles); chasles.set_opacity(0)
        self.play(xi.animate.set_opacity(1), chasles.animate.set_opacity(1), run_time=st.T_WRITE)
        self.wait(st.HOLD_LONG)

        # ── the dual-use payoff ─────────────────────────────────────────────
        self.stop_ambient_camera_rotation()
        self.play(FadeOut(chasles), helix.animate.set_stroke(opacity=0.25),
                  axis.animate.set_stroke(opacity=0.25), FadeOut(bead), run_time=st.T_FADE)

        left = st.body("d_pose", color=st.INK, scale=0.6)
        left_sub = st.caption("= the 6 PoseNet scalars", scale=0.34).next_to(left, DOWN, buff=0.14)
        L = VGroup(left, left_sub).to_edge(LEFT, buff=1.1)
        right = st.body("d_seg", color=st.INK, scale=0.6)
        right_sub = st.caption("= ξ warps the partition\nframe₀ → frame₁", scale=0.34).next_to(right, DOWN, buff=0.14)
        R = VGroup(right, right_sub).to_edge(RIGHT, buff=1.1)
        hub = st.mono("ξ", color=st.GOLD, scale=0.9).move_to(0.6 * UP)
        aL = Arrow(hub.get_left(), L.get_top() + 0.2 * DOWN, color=st.FAINT, stroke_width=3, buff=0.25)
        aR = Arrow(hub.get_right(), R.get_top() + 0.2 * DOWN, color=st.FAINT, stroke_width=3, buff=0.25)
        grp = VGroup(hub, aL, aR, L, R)
        self.add_fixed_in_frame_mobjects(grp); grp.set_opacity(0)
        self.play(FadeOut(xi), run_time=st.T_FADE)
        self.play(FadeIn(hub), run_time=st.T_FADE)
        self.play(Create(aL), Create(aR), FadeIn(L, shift=0.2 * RIGHT), FadeIn(R, shift=0.2 * LEFT),
                  run_time=st.T_MORPH)
        self.wait(st.HOLD)

        hero = st.hero("encode ξ once — it serves both scores", scale=0.56).to_edge(DOWN, buff=0.6)
        self.add_fixed_in_frame_mobjects(hero); hero.set_opacity(0)
        self.play(hero.animate.set_opacity(1), run_time=st.T_FADE)
        self.wait(st.HOLD_LONG)
        self.play(FadeOut(VGroup(grp, hero)), FadeOut(helix), FadeOut(axis), run_time=st.T_MORPH)
        self.wait(st.BEAT)
