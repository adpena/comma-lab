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

import json
from pathlib import Path

import numpy as np
from manim import (
    ThreeDScene, ParametricFunction, Line, Dot, Dot3D, ImageMobject, VGroup, Arrow,
    Circle, ValueTracker, always_redraw, FadeIn, FadeOut, Write, Create, GrowFromCenter,
    DEGREES, UP, DOWN, LEFT, RIGHT, ORIGIN, rate_functions,
)

import _style as st

_ASSETS = Path(__file__).resolve().parent.parent / "assets"
_EGO = np.load(_ASSETS / "ego_clip.npy")            # (K, H, W, 3) real frames
_EGO_POSES = np.load(_ASSETS / "ego_poses.npy")     # (K, 6) the twists ξ
_FLOW = np.load(_ASSETS / "ego_flow_uv.npy")        # (K, gy, gx, 2) measured flow
_GRID = np.load(_ASSETS / "ego_flow_grid.npy")      # (gy, gx, 2) arrow anchors (x,y px)
_FOE = np.load(_ASSETS / "ego_foe.npy")             # (K, 2) focus of expansion (px)
_CW, _CH = (int(x) for x in np.load(_ASSETS / "ego_clip_wh.npy"))
_HOOD_OVERLAY = str(_ASSETS / "ego_hood_overlay.png")   # my_car null (RGBA)
_HOOD_REFLECT = np.load(_ASSETS / "ego_hood_reflect.npy")  # (R,2) px — specular flow spots
_META = json.loads((_ASSETS / "meta.json").read_text())
_FBC = _META.get("flow_by_class", {"hood_mycar_px": 0.18, "road_px": 2.48, "ratio_road_over_hood": 13.6})
_K = len(_EGO)
_GY, _GX = _FLOW.shape[1], _FLOW.shape[2]

# clip placement on stage → pixel↔scene mapping for the flow overlay
_IMGH = 4.9
_IMGW = _IMGH * _CW / _CH
_CX, _CY = 0.0, 0.2
_FLOW_SCALE = 16.0     # amplify the small (≈1 px) inter-frame flow for visibility


def _px2scene(px: float, py: float) -> np.ndarray:
    return np.array([(px / _CW - 0.5) * _IMGW + _CX,
                     (0.5 - py / _CH) * _IMGH + _CY, 0.0])


def _flow_arrows(i: int, scale: float = _FLOW_SCALE, cap_px: float = 26.0,
                 thr_px: float = 0.35) -> VGroup:
    """The MEASURED optical-flow field at clip index i, drawn in scene coords.

    Faithful: arrows are the real Farneback flow the ego-motion produces (not ξ
    itself — ξ ∈ se(3) is what INDUCES this projected field). Small flow is
    dropped (sky / near-FoE) and long arrows are length-capped for legibility;
    directions are never altered.
    """
    grp = VGroup()
    fuv = _FLOW[i]
    for r in range(_GY):
        for c in range(_GX):
            u, v = float(fuv[r, c, 0]), float(fuv[r, c, 1])
            if np.hypot(u, v) < thr_px:
                continue
            du, dv = u * scale, v * scale
            dmag = np.hypot(du, dv)
            if dmag > cap_px:
                du *= cap_px / dmag
                dv *= cap_px / dmag
            x, y = float(_GRID[r, c, 0]), float(_GRID[r, c, 1])
            grp.add(Arrow(_px2scene(x, y), _px2scene(x + du, y + dv), buff=0.0,
                          stroke_width=2.4, color=st.CYAN, tip_length=0.11,
                          max_tip_length_to_length_ratio=0.42))
    return grp


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

        # ── beat 3 · the screw-derived motion, drawn LIVE on the video ──────
        # (camera is face-on now → normal 2D mobjects render flat)
        _HARD = _K // 2                                    # the hardest frame (Scene 2's)
        ci = ValueTracker(0.0)

        def _idx() -> int:
            return int(np.clip(ci.get_value(), 0, _K - 1))

        # the clip, the flow field, and the FoE ALL update per displayed frame:
        # you WATCH the measured motion field stream as the car actually drives.
        clip = always_redraw(
            lambda: ImageMobject(_EGO[_idx()]).set(height=_IMGH).move_to(_CX * RIGHT + _CY * UP))
        flow_live = always_redraw(lambda: _flow_arrows(_idx()))
        foe_live = always_redraw(
            lambda: Dot(_px2scene(float(_FOE[_idx(), 0]), float(_FOE[_idx(), 1])),
                        radius=0.075, color=st.CORAL))
        self.add(clip)
        kick = st.top(st.kicker("the ego-motion, made visible — drawn live on the footage"))
        xi_ro = st.corner_tr(st.mono(r"ξ=(v,ω)∈se(3)", scale=0.4))
        cap = st.bottom(st.caption(
            "the optical flow the ego-screw ξ induces — watch it stream from the focus of expansion",
            color=st.INK))
        self.play(FadeIn(clip), FadeIn(kick, shift=0.1 * DOWN), FadeIn(xi_ro),
                  FadeIn(cap, shift=0.1 * UP), run_time=st.T_FADE)
        self.add(flow_live, foe_live)
        self.play(FadeIn(flow_live), FadeIn(foe_live), run_time=st.T_FADE)
        # THE DYNAMIC PAYOFF: play the run slowly and watch the field move WITH it
        self.play(ci.animate.set_value(_K - 1), run_time=9.0, rate_func=rate_functions.linear)
        self.play(ci.animate.set_value(_HARD), run_time=st.T_MORPH, rate_func=rate_functions.ease_in_out_sine)

        # freeze on the hardest frame: swap the live overlays for static snapshots
        # so the FoE emphasis + hood beats annotate a stable image (smooth fades)
        frozen = ImageMobject(_EGO[_HARD]).set(height=_IMGH).move_to(_CX * RIGHT + _CY * UP)
        flow = _flow_arrows(_HARD)
        foe_pt = _px2scene(float(_FOE[_HARD, 0]), float(_FOE[_HARD, 1]))
        foe_dot = Dot(foe_pt, radius=0.075, color=st.CORAL)
        self.add(frozen); self.bring_to_back(frozen)
        self.add(flow, foe_dot); self.remove(clip, flow_live, foe_live)
        foe_ring = Circle(radius=0.30, color=st.CORAL, stroke_width=2.5).move_to(foe_pt).set_opacity(0.7)
        foe_lbl = st.caption("focus of expansion", color=st.CORAL, scale=0.30).next_to(foe_pt, UP, buff=0.28)
        self.play(FadeIn(foe_lbl, shift=0.08 * DOWN), Create(foe_ring), run_time=st.T_FADE)
        self.play(foe_ring.animate.scale(1.35).set_opacity(0.0), rate_func=rate_functions.ease_out_sine,
                  run_time=st.T_MORPH)
        self.wait(st.BEAT)

        # ── beat 3c · the ego-hood NULL — the FIXED POINT of the SAME ξ ──────
        # my_car is rigidly bolted to the camera → zero relative flow. It falls
        # out of the same ξ: the world (road) is transported past a still body.
        hood_ov = ImageMobject(_HOOD_OVERLAY).set(height=_IMGH).move_to(_CX * RIGHT + _CY * UP)
        kick2 = st.top(st.kicker("my_car — the fixed frame of ξ", color=st.INK))
        hcap = st.bottom(st.caption(
            f"rigidly bolted to the camera — the FIXED POINT of ξ: "
            f"{_FBC['hood_mycar_px']:.2f}px vs {_FBC['road_px']:.2f}px on the road "
            f"({_FBC['ratio_road_over_hood']:.0f}× stiller)", color=st.INK))
        self.play(FadeIn(hood_ov), FadeOut(kick), FadeIn(kick2, shift=0.1 * DOWN),
                  FadeOut(cap), FadeIn(hcap, shift=0.1 * UP), run_time=st.T_MORPH)
        self.wait(st.HOLD)

        # the honest nuance: the wet hood is a curved MIRROR → its specular
        # reflections are virtual images of the moving world, so THEY flow.
        refl = VGroup(*[
            Circle(radius=0.05, color=st.CYAN, stroke_width=2.5).move_to(_px2scene(float(p[0]), float(p[1])))
            for p in _HOOD_REFLECT
        ])
        rcap = st.bottom(st.caption(
            "even here the wet hood mirrors the moving world — its reflections flow", color=st.CYAN))
        if len(refl):
            self.play(*[GrowFromCenter(d) for d in refl], FadeOut(hcap),
                      FadeIn(rcap, shift=0.1 * UP), run_time=st.T_WRITE)
        else:
            self.play(FadeOut(hcap), FadeIn(rcap, shift=0.1 * UP), run_time=st.T_WRITE)
        self.wait(st.HOLD)

        # ── beat 4 · the dual-use (frozen backdrop already in place) ────────
        self.play(FadeOut(rcap), FadeOut(kick2), FadeOut(xi_ro), FadeOut(flow),
                  FadeOut(foe_dot), FadeOut(foe_lbl), FadeOut(hood_ov), FadeOut(refl),
                  frozen.animate.set_opacity(0.18), run_time=st.T_FADE)
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
        self.play(FadeOut(VGroup(hub, aL, aR, L, Rr, hero)), FadeOut(frozen), run_time=st.T_MORPH)
        self.wait(st.BEAT)
