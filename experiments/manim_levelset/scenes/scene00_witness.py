"""Scene 0 — THE TASK-SPACE WITNESS  (the cold open / the idea).

Frames the whole series BEFORE the separatrix (01) and the screw (03): the
contest doesn't score how the video LOOKS — it scores what two frozen networks
SEE. So you don't reconstruct the pixels; you reproduce the scorer's output — a
tiny "witness" that lands in the same evaluator cells. And it all falls out of
one object: a level-set flow.

FAITHFUL (NO-FAKE): the score is exactly
    S = 100·d_seg + √(10·d_pose) + 25·bytes/N,      N = 37,545,489
(upstream/evaluate.py). d_seg = SegNet argmax disagreement, d_pose = MSE on the
6 PoseNet scalars, bytes = |archive.zip|. The witness stores the task-sufficient
statistic (the segmentation boundary + the pose twist), not RGB.

Render:  ./render.sh -qh scenes/scene00_witness.py Witness
"""
from __future__ import annotations

from pathlib import Path

from manim import (
    Scene, ImageMobject, MathTex, VGroup, FadeIn, FadeOut, Write, GrowFromCenter,
    UP, DOWN, ORIGIN, rate_functions,
)

import _style as st

_ASSETS = Path(__file__).resolve().parent.parent / "assets"
_STAGE_H = 4.7
_STAGE_POS = 0.2 * DOWN


class Witness(Scene):
    def _img(self, name: str):
        return ImageMobject(str(_ASSETS / name)).set(height=_STAGE_H).move_to(_STAGE_POS)

    def construct(self) -> None:
        # ── beat 1 · title ───────────────────────────────────────────────────
        card = st.titlecard("00 · the idea", "The Task-Space Witness",
                            "code what the scorer sees — not the pixels").move_to(ORIGIN)
        self.play(Write(card[1]), run_time=st.T_WRITE)
        self.play(FadeIn(card[0], shift=0.15 * DOWN), GrowFromCenter(card[2]), run_time=st.T_FADE)
        self.play(FadeIn(card[3], shift=0.1 * UP), run_time=st.T_FADE)
        self.wait(st.HOLD)
        self.play(FadeOut(card), run_time=st.T_FADE)

        # ── beat 2 · the contest ────────────────────────────────────────────
        frame = self._img("hardest_frame.png")
        self.play(FadeIn(frame), run_time=st.T_FADE)
        cap = st.bottom(st.caption("comma.ai — compress a minute of driving"))
        self.play(FadeIn(cap, shift=0.12 * UP), run_time=st.T_FADE)
        self.wait(st.HOLD)

        # ── beat 3 · but you're scored on what two nets SEE ─────────────────
        seg = self._img("hardest_argmax.png"); seg.set_opacity(0.0)
        self.add(seg)
        cap2 = st.bottom(st.caption("but you're scored on what two frozen networks SEE", color=st.INK))
        self.play(frame.animate.set_opacity(0.0), seg.animate.set_opacity(1.0),
                  FadeOut(cap), FadeIn(cap2, shift=0.12 * UP), run_time=st.T_MORPH)
        self.wait(st.HOLD)

        # ── beat 4 · the score equation (the hero) ──────────────────────────
        self.play(seg.animate.set_opacity(0.16), FadeOut(cap2), run_time=st.T_FADE)
        S = MathTex(
            r"S", r"=", r"100\,d_{\mathrm{seg}}", r"+", r"\sqrt{10\,d_{\mathrm{pose}}}",
            r"+", r"25\,\tfrac{\text{bytes}}{N}",
        ).scale(1.05).move_to(0.4 * UP)
        S[2].set_color(st.BLUE)     # d_seg
        S[4].set_color(st.GOLD)     # d_pose
        S[6].set_color(st.CORAL)    # bytes
        self.play(Write(S[0]), Write(S[1]), run_time=st.T_FADE)
        self.play(Write(S[2]), run_time=st.T_FADE)
        c = st.bottom(st.caption("d_seg — does SegNet see the same segmentation?", color=st.BLUE))
        self.play(FadeIn(c, shift=0.1 * UP), run_time=st.T_FADE)
        self.wait(st.HOLD)
        self.play(Write(S[3]), Write(S[4]), run_time=st.T_FADE)
        c2 = st.bottom(st.caption("d_pose — does PoseNet read the same motion?", color=st.GOLD))
        self.play(FadeOut(c), FadeIn(c2, shift=0.1 * UP), run_time=st.T_FADE)
        self.wait(st.HOLD)
        self.play(Write(S[5]), Write(S[6]), run_time=st.T_FADE)
        c3 = st.bottom(st.caption("bytes — how small is the archive?", color=st.CORAL))
        self.play(FadeOut(c2), FadeIn(c3, shift=0.1 * UP), run_time=st.T_FADE)
        self.wait(st.HOLD_LONG)

        # ── beat 5 · the reframe ────────────────────────────────────────────
        self.play(FadeOut(c3), S.animate.scale(0.62).to_edge(UP, buff=0.5), run_time=st.T_MORPH)
        r1 = st.body("so reproduce the scorer's output — not the pixels.").move_to(0.4 * UP)
        r2 = st.caption("coding for machines: code the task, not the picture").next_to(r1, DOWN, buff=0.3)
        self.play(FadeIn(r1, shift=0.1 * UP), run_time=st.T_FADE)
        self.play(FadeIn(r2, shift=0.1 * UP), run_time=st.T_FADE)
        self.wait(st.HOLD_LONG)

        # ── beat 6 · the witness ────────────────────────────────────────────
        self.play(FadeOut(VGroup(r1, r2)), run_time=st.T_FADE)
        w1 = st.body("the witness: the segmentation boundary + the pose twist").move_to(0.4 * UP)
        w2 = st.caption("a few kilobytes that land in the same evaluator cells").next_to(w1, DOWN, buff=0.3)
        self.play(FadeIn(w1, shift=0.1 * UP), run_time=st.T_FADE)
        self.play(FadeIn(w2, shift=0.1 * UP), run_time=st.T_FADE)
        self.wait(st.HOLD)

        # ── beat 7 · one object ─────────────────────────────────────────────
        self.play(FadeOut(VGroup(w1, w2)), FadeOut(seg), FadeOut(S), run_time=st.T_FADE)
        one = st.heading("it all falls out of one object", scale=0.72)
        flow = st.hero("a level-set flow", scale=0.62).next_to(one, DOWN, buff=0.3)
        facets = st.mono("distortion · representation · curriculum · pose · rate",
                         color=st.MUTED, scale=0.36).next_to(flow, DOWN, buff=0.5)
        self.play(Write(one), run_time=st.T_WRITE)
        self.play(FadeIn(flow, shift=0.1 * UP), run_time=st.T_FADE)
        self.play(FadeIn(facets), run_time=st.T_FADE)
        self.wait(st.HOLD)

        # ── beat 8 · handoff to the series ──────────────────────────────────
        idx = st.caption("01  the separatrix       02  the hardest frame       03  the screw",
                         color=st.MUTED).next_to(facets, DOWN, buff=0.6)
        self.play(FadeIn(idx), run_time=st.T_FADE)
        self.wait(st.HOLD_LONG)
        self.play(FadeOut(VGroup(one, flow, facets, idx)), run_time=st.T_MORPH)
        self.wait(st.BEAT)
