"""Scene 2 — THE HARDEST FRAME.

Fast-forward the ACTUAL contest video (600 scored pairs), slam to the single
hardest frame for d_seg, then watch the math work on it: the real openpilot /
comma10k SegNet argmax partition, the separatrix that IS d_seg, and the margin
field (= the Fisher metric). Every asset is genuine scorer output (NO-FAKE),
exported by scenes/_prep_hardest_frame.py from the gt_n600.npz cache.

Render:
    cd experiments/manim_levelset
    ../../.venv/bin/python scenes/_prep_hardest_frame.py   # once
    ./render.sh -qh scenes/scene02_hardest_frame.py HardestFrame
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from manim import (
    Scene, ImageMobject, Text, VGroup, Square, ValueTracker, always_redraw,
    FadeIn, FadeOut, Write, GrowFromCenter, UP, DOWN, LEFT, RIGHT, ORIGIN,
    rate_functions,
)

import _style as st

_ASSETS = Path(__file__).resolve().parent.parent / "assets"
_META = json.loads((_ASSETS / "meta.json").read_text())
_MONT = np.load(_ASSETS / "montage.npy")
_MIDX = np.load(_ASSETS / "montage_idx.npy")
_HARD = _META["hardest_frame"]
_LANE_PX = _META["lane_px"]
_BND_PX = _META["boundary_px"]


def _legend():
    rows = VGroup()
    for hexc, name in zip(st.COMMA10K_HEX, st.COMMA10K_LABEL):
        sw = Square(side_length=0.24, fill_color=hexc, fill_opacity=1.0, stroke_width=0)
        lbl = Text(name, font=st.FONT, color=st.MUTED).scale(0.30).next_to(sw, RIGHT, buff=0.12)
        rows.add(VGroup(sw, lbl))
    return rows.arrange(RIGHT, buff=0.5)


class HardestFrame(Scene):
    def _img(self, name: str, height: float = 6.1):
        m = ImageMobject(str(_ASSETS / name))
        m.height = height
        m.move_to(0.3 * DOWN)
        return m

    def construct(self) -> None:
        # ── title ────────────────────────────────────────────────────────────
        card = st.titlecard("02 · the hardest frame", "The Hardest Frame",
                            "600 scored pairs — one is the worst").move_to(ORIGIN)
        self.play(Write(card[1]), run_time=1.0)
        self.play(FadeIn(card[0], shift=0.15 * DOWN), GrowFromCenter(card[2]), run_time=0.6)
        self.play(FadeIn(card[3], shift=0.1 * UP), run_time=0.5)
        self.wait(0.5)
        self.play(FadeOut(card), run_time=0.5)

        # ── fast-forward scrub through the real video ────────────────────────
        idx = ValueTracker(0.0)
        scrub = always_redraw(
            lambda: ImageMobject(_MONT[int(np.clip(idx.get_value(), 0, len(_MONT) - 1))])
            .set(height=5.6).move_to(0.3 * DOWN)
        )
        counter = always_redraw(
            lambda: st.mono(f"frame {int(_MIDX[int(np.clip(idx.get_value(),0,len(_MONT)-1))]):3d} / 599")
            .to_corner(UP + RIGHT, buff=0.5)
        )
        ff = st.kicker("fast-forward").to_corner(UP + LEFT, buff=0.5)
        self.add(scrub, counter)
        self.play(FadeIn(scrub), FadeIn(counter), FadeIn(ff), run_time=0.4)
        self.play(idx.animate.set_value(len(_MONT) - 1), run_time=3.2,
                  rate_func=rate_functions.ease_in_out_sine)
        self.wait(0.2)
        self.play(FadeOut(scrub), FadeOut(counter), FadeOut(ff), run_time=0.4)

        # ── land on the hardest frame (full res) ─────────────────────────────
        frame = self._img("hardest_frame.png")
        self.play(FadeIn(frame), run_time=0.7)
        stamp = st.mono(f"frame {_HARD}  ·  hardest for d_seg").to_edge(UP, buff=0.45)
        self.play(FadeIn(stamp, shift=0.15 * DOWN), run_time=0.6)
        self.wait(0.8)

        # ── the openpilot / comma10k argmax partition ────────────────────────
        cap = st.body("what the scorer sees:  the openpilot segmentation").to_edge(DOWN, buff=0.42)
        argmax = self._img("hardest_argmax.png")
        self.play(FadeIn(cap, shift=0.12 * UP), run_time=0.6)
        self.add(argmax); argmax.set_opacity(0.0)
        self.play(frame.animate.set_opacity(0.0), argmax.animate.set_opacity(1.0), run_time=1.3)
        legend = _legend().next_to(cap, UP, buff=0.26)
        self.play(FadeIn(legend, lag_ratio=0.12), run_time=0.8)
        self.wait(0.9)

        # ── the separatrix ───────────────────────────────────────────────────
        sep = self._img("hardest_separatrix.png")
        cap2 = st.body(f"d_seg lives on the boundary  ·  {_BND_PX:,} px  ·  a curve, not a volume").to_edge(DOWN, buff=0.42)
        self.add(sep); sep.set_opacity(0.0)
        self.play(argmax.animate.set_opacity(0.0), sep.animate.set_opacity(1.0),
                  FadeOut(cap), FadeIn(cap2, shift=0.12 * UP), run_time=1.1)
        lane_note = st.caption(
            f"the fragile lane class — {_LANE_PX:,} px — is the most flip-prone (the erasure long-tail)",
            color=st.CORAL).next_to(cap2, UP, buff=0.24)
        self.play(FadeOut(legend), FadeIn(lane_note, shift=0.1 * UP), run_time=0.7)
        self.wait(1.1)

        # ── the margin field = the Fisher metric ─────────────────────────────
        margin = self._img("hardest_margin.png")
        cap3 = st.body("the margin field  =  the Fisher metric").to_edge(DOWN, buff=0.42)
        self.add(margin); margin.set_opacity(0.0)
        self.play(sep.animate.set_opacity(0.0), margin.animate.set_opacity(1.0),
                  FadeOut(cap2), FadeOut(lane_note), FadeIn(cap3, shift=0.12 * UP), run_time=1.2)
        fisher = st.mono("Fisher curvature ↔ −margin   ·   Pearson 0.978  (measured)",
                         color=st.MUTED).next_to(cap3, UP, buff=0.24)
        self.play(FadeIn(fisher), run_time=0.6)
        self.wait(1.5)

        # ── closing thesis ───────────────────────────────────────────────────
        self.play(FadeOut(stamp), FadeOut(cap3), FadeOut(fisher),
                  margin.animate.set_opacity(0.2), run_time=0.8)
        c1 = st.heading("amortize this partition at low bytes", scale=0.66)
        c2 = st.caption("spend capacity on the boundary, not the volume").next_to(c1, DOWN, buff=0.24)
        c3 = st.hero("the level-set task-space witness", scale=0.58).next_to(c2, DOWN, buff=0.4)
        self.play(Write(c1), run_time=1.1)
        self.play(FadeIn(c2, shift=0.12 * UP), run_time=0.6)
        self.play(FadeIn(c3, shift=0.12 * UP), run_time=0.7)
        self.wait(2.0)
        self.play(FadeOut(VGroup(c1, c2, c3)), FadeOut(margin), run_time=1.0)
        self.wait(0.3)
