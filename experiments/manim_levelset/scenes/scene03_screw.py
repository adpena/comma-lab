"""Scene 3 — ONE TWIST WARPS THE WORLD  (the se(3) screw).

The dramatic beat, in two movements:
  (A) the ABSTRACT screw — a 3D helix + axis, a frame screwing along it, the
      twist ξ, Chasles' theorem;
  (B) the PAYOFF — that screw-derived ego-motion RENDERED ON THE REAL CONTEST
      VIDEO: a contiguous run of real frames plays (the motion the twists ξ
      produce), then the dual-use — the SAME ξ is d_pose AND warps the
      partition for d_seg.

FAITHFUL / NO-FAKE: the cached gt_poses ARE the twists ξ (the 6 PoseNet scalars
per pair); the played frames are the real gt_f1 run those twists produced. No
approximation, no synthetic warp — the actual footage the actual motion made.

Render:  ./render.sh -qh scenes/scene03_screw.py Screw
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from manim import (
    ThreeDScene, ParametricFunction, Line, Dot3D, ImageMobject, VGroup, Arrow,
    ValueTracker, always_redraw, FadeIn, FadeOut, Write, Create, DEGREES,
    UP, DOWN, LEFT, RIGHT, ORIGIN, rate_functions,
)

import _style as st

_ASSETS = Path(__file__).resolve().parent.parent / "assets"
_EGO = np.load(_ASSETS / "ego_clip.npy")            # (K, H, W, 3) real frames
_EGO_POSES = np.load(_ASSETS / "ego_poses.npy")     # (K, 6) the twists ξ
_K = len(_EGO)

_R, _PITCH, _TURNS = 1.15, 0.42, 3.1
_T0, _T1 = -_TURNS * np.pi, _TURNS * np.pi


def _helix(t: float):
    return np.array([_R * np.cos(t), _R * np.sin(t), _PITCH * t])


class Screw(ThreeDScene):
    def construct(self) -> None:
        self.set_camera_orientation(phi=68 * DEGREES, theta=-50 * DEGREES, zoom=0.95)

        # ── beat 1 · title (fixed HUD) ──────────────────────────────────────
        card = st.titlecard("03 · the screw", "One Twist Warps the World",
                            "every rigid motion is a screw").move_to(ORIGIN)
        self.add_fixed_in_frame_mobjects(card); card.set_opacity(0)
        self.play(card.animate.set_opacity(1), run_time=st.T_WRITE)
        self.wait(st.HOLD)
        self.play(FadeOut(card), run_time=st.T_FADE)

        # ── beat 2 · the abstract screw ─────────────────────────────────────
        axis = Line(_helix(_T0) * [0, 0, 1], _helix(_T1) * [0, 0, 1], color=st.GOLD, stroke_width=3)
        helix = ParametricFunction(_helix, t_range=[_T0, _T1, 0.05], color=st.CYAN, stroke_width=4)
        self.play(Create(axis), run_time=st.T_FADE)
        self.begin_ambient_camera_rotation(rate=0.13)
        self.play(Create(helix), run_time=st.T_HERO * 0.75, rate_func=rate_functions.ease_in_out_sine)
        t = ValueTracker(_T0)
        bead = always_redraw(lambda: Dot3D(_helix(t.get_value()), radius=0.13, color=st.CORAL))
        self.add(bead)
        self.play(t.animate.set_value(_T1), run_time=st.T_HERO, rate_func=rate_functions.ease_in_out_sine)

        xi = st.eq(r"\xi=(\mathbf{v},\boldsymbol{\omega})\in\mathfrak{se}(3),\quad e^{\hat\xi}\in SE(3)", scale=0.6)
        st.top(xi)
        chasles = st.bottom(st.caption("Chasles — a rotation about an axis + a translation along it"))
        self.add_fixed_in_frame_mobjects(xi, chasles); xi.set_opacity(0); chasles.set_opacity(0)
        self.play(xi.animate.set_opacity(1), chasles.animate.set_opacity(1), run_time=st.T_WRITE)
        self.wait(st.HOLD)

        # ── transition · face front, retire the abstract screw ──────────────
        self.stop_ambient_camera_rotation()
        self.play(FadeOut(chasles), FadeOut(xi), FadeOut(bead),
                  helix.animate.set_stroke(opacity=0.0), axis.animate.set_stroke(opacity=0.0),
                  run_time=st.T_FADE)
        self.move_camera(phi=0, theta=-90 * DEGREES, zoom=1.0, run_time=st.T_MORPH)

        # ── beat 3 · the screw-derived motion on the real contest video ─────
        # (camera is face-on now → normal 2D mobjects render flat)
        ci = ValueTracker(0.0)
        clip = always_redraw(
            lambda: ImageMobject(_EGO[int(np.clip(ci.get_value(), 0, _K - 1))])
            .set(height=4.9).move_to(0.2 * UP)
        )
        self.add(clip)
        kick = st.top(st.kicker("the screw-derived ego-motion — on the contest video"))
        ro = always_redraw(lambda: st.corner_tr(
            st.mono(f"ξ · v={_EGO_POSES[int(np.clip(ci.get_value(),0,_K-1)),0]:.1f}", scale=0.4)))
        cap = st.bottom(st.caption("each pair's motion is one twist ξ — here they play out"))
        self.play(FadeIn(clip), FadeIn(kick, shift=0.1 * DOWN), FadeIn(ro),
                  FadeIn(cap, shift=0.1 * UP), run_time=st.T_FADE)
        self.play(ci.animate.set_value(_K - 1), run_time=st.T_HERO * 1.3, rate_func=rate_functions.linear)
        self.wait(st.BEAT)

        # ── beat 4 · the dual-use ───────────────────────────────────────────
        self.play(FadeOut(cap), FadeOut(kick), FadeOut(ro), clip.animate.set_opacity(0.18),
                  run_time=st.T_FADE)
        L = VGroup(st.body("d_pose", color=st.INK, scale=0.6),
                   st.caption("= the 6 PoseNet scalars", scale=0.32)).arrange(DOWN, buff=0.14).to_edge(LEFT, buff=1.2)
        Rr = VGroup(st.body("d_seg", color=st.INK, scale=0.6),
                    st.caption("= ξ warps the partition\nframe₀ → frame₁", scale=0.32)).arrange(DOWN, buff=0.14).to_edge(RIGHT, buff=1.2)
        hub = st.mono("ξ", color=st.GOLD, scale=0.95).move_to(0.7 * UP)
        aL = Arrow(hub.get_left(), L.get_top() + 0.15 * DOWN, color=st.FAINT, stroke_width=3, buff=0.25)
        aR = Arrow(hub.get_right(), Rr.get_top() + 0.15 * DOWN, color=st.FAINT, stroke_width=3, buff=0.25)
        self.play(FadeIn(hub), run_time=st.T_FADE)
        self.play(Create(aL), Create(aR), FadeIn(L, shift=0.2 * RIGHT), FadeIn(Rr, shift=0.2 * LEFT),
                  run_time=st.T_MORPH)
        self.wait(st.HOLD)

        hero = st.hero("encode ξ once — it serves both scores", scale=0.56).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(hero, shift=0.1 * UP), run_time=st.T_FADE)
        self.wait(st.HOLD_LONG)
        self.play(FadeOut(VGroup(hub, aL, aR, L, Rr, hero)), FadeOut(clip), run_time=st.T_MORPH)
        self.wait(st.BEAT)
