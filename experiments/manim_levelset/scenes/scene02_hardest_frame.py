"""Scene 2 — THE HARDEST FRAME.

Fast-forward through the ACTUAL contest video (upstream/videos/0.mkv, all 600
scored pairs), slam to a stop on the single hardest frame for d_seg, then watch
the math work on it — the real SegNet argmax partition, the separatrix that IS
d_seg, and the margin field (= the Fisher metric).

All assets are REAL contest data, exported by `scenes/_prep_hardest_frame.py`
from the cached frozen-SegNet outputs (gt_n600.npz). NO-FAKE: nothing here is a
toy — the frame, the argmax, and the margin are the genuine scorer's outputs.

Prep once, then render:
    cd experiments/manim_levelset
    ../../.venv/bin/python scenes/_prep_hardest_frame.py
    .venv/bin/manim -qm scenes/scene02_hardest_frame.py HardestFrame
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from manim import (
    BLACK, WHITE, GREY_B, YELLOW, config,
    Scene, ImageMobject, Text, VGroup, Dot, ValueTracker, always_redraw,
    FadeIn, FadeOut, Write, UP, DOWN, LEFT, RIGHT, ORIGIN, rate_functions,
)

config.background_color = BLACK

_ASSETS = Path(__file__).resolve().parent.parent / "assets"
_META = json.loads((_ASSETS / "meta.json").read_text())
_MONT = np.load(_ASSETS / "montage.npy")           # (72,168,224,3) uint8
_MIDX = np.load(_ASSETS / "montage_idx.npy")       # real frame numbers
_HARD = _META["hardest_frame"]
_LANE_PX = _META["lane_px"]
_BND_PX = _META["boundary_px"]

_LEGEND = [
    ("road", [45, 92, 168]),
    ("lane", [232, 197, 71]),
    ("undrivable", [66, 132, 121]),
    ("movable", [196, 84, 73]),
    ("my-car", [120, 96, 162]),
]


class HardestFrame(Scene):
    def _img(self, name: str, height: float = 6.2):
        m = ImageMobject(str(_ASSETS / name))
        m.height = height
        m.move_to(0.35 * DOWN)
        return m

    def construct(self) -> None:
        # ── title ────────────────────────────────────────────────────────────
        t1 = Text("The Hardest Frame", font="Helvetica Neue", weight="BOLD").scale(0.95)
        t2 = Text("600 scored pairs — one is the worst",
                  font="Helvetica Neue", color=GREY_B).scale(0.42).next_to(t1, DOWN, buff=0.18)
        title = VGroup(t1, t2).move_to(ORIGIN)
        self.play(Write(t1), run_time=1.0)
        self.play(FadeIn(t2, shift=0.2 * UP), run_time=0.6)
        self.wait(0.4)
        self.play(FadeOut(title), run_time=0.5)

        # ── fast-forward scrub through the real video ────────────────────────
        idx = ValueTracker(0.0)
        scrub = always_redraw(
            lambda: ImageMobject(_MONT[int(np.clip(idx.get_value(), 0, len(_MONT) - 1))])
            .set(height=5.6).move_to(0.35 * DOWN)
        )
        counter = always_redraw(
            lambda: Text(
                f"frame {int(_MIDX[int(np.clip(idx.get_value(),0,len(_MONT)-1))]):3d} / 599",
                font="Menlo", color=YELLOW,
            ).scale(0.5).to_corner(UP + RIGHT, buff=0.5)
        )
        ff = Text("⏩  fast-forward", font="Helvetica Neue", color=GREY_B).scale(0.4).to_corner(UP + LEFT, buff=0.5)
        self.add(scrub, counter)
        self.play(FadeIn(scrub), FadeIn(counter), FadeIn(ff), run_time=0.4)
        self.play(idx.animate.set_value(len(_MONT) - 1), run_time=3.2,
                  rate_func=rate_functions.ease_in_out_sine)
        self.wait(0.2)
        self.play(FadeOut(scrub), FadeOut(counter), FadeOut(ff), run_time=0.4)

        # ── land on the hardest frame (full res) ─────────────────────────────
        frame = self._img("hardest_frame.png")
        self.play(FadeIn(frame), run_time=0.7)
        stamp = Text(f"frame {_HARD}  ·  the hardest for d_seg",
                     font="Menlo", color=YELLOW).scale(0.5).to_edge(UP, buff=0.45)
        self.play(FadeIn(stamp, shift=0.2 * DOWN), run_time=0.6)
        self.wait(0.9)

        # ── what the scorer sees: the SegNet argmax partition ────────────────
        cap = Text("what the scorer sees:  the SegNet argmax partition",
                   font="Helvetica Neue", color=WHITE).scale(0.42).to_edge(DOWN, buff=0.4)
        argmax = self._img("hardest_argmax.png")
        self.play(FadeIn(cap, shift=0.15 * UP), run_time=0.6)
        self.add(argmax)
        argmax.set_opacity(0.0)
        self.play(frame.animate.set_opacity(0.0), argmax.animate.set_opacity(1.0),
                  run_time=1.3)

        # class legend (bottom strip of colored dots)
        chips = VGroup()
        for name, rgb in _LEGEND:
            d = Dot(radius=0.09, color="#%02x%02x%02x" % tuple(rgb))
            lbl = Text(name, font="Helvetica Neue", color=GREY_B).scale(0.30).next_to(d, RIGHT, buff=0.08)
            chips.add(VGroup(d, lbl))
        chips.arrange(RIGHT, buff=0.45).next_to(cap, UP, buff=0.25)
        self.play(FadeIn(chips), run_time=0.6)
        self.wait(1.0)

        # ── the separatrix: d_seg lives on the boundary ──────────────────────
        sep = self._img("hardest_separatrix.png")
        cap2 = Text(
            f"d_seg lives on the boundary  ·  {_BND_PX:,} px  ·  a curve, not a volume",
            font="Helvetica Neue", color=WHITE).scale(0.40).to_edge(DOWN, buff=0.4)
        self.add(sep); sep.set_opacity(0.0)
        self.play(argmax.animate.set_opacity(0.0), sep.animate.set_opacity(1.0),
                  FadeOut(cap), FadeIn(cap2, shift=0.15 * UP), run_time=1.1)
        self.wait(0.4)
        lane_note = Text(
            f"the fragile lane class — {_LANE_PX:,} px — is the most flip-prone (the erasure long-tail)",
            font="Helvetica Neue", color="#e8c547").scale(0.34).next_to(cap2, UP, buff=0.22)
        self.play(FadeOut(chips), FadeIn(lane_note, shift=0.1 * UP), run_time=0.7)
        self.wait(1.2)

        # ── the margin field = the Fisher metric ─────────────────────────────
        margin = self._img("hardest_margin.png")
        cap3 = Text(
            "the margin field  =  the Fisher metric   (flat interior · bright boundary annulus)",
            font="Helvetica Neue", color=WHITE).scale(0.38).to_edge(DOWN, buff=0.4)
        self.add(margin); margin.set_opacity(0.0)
        self.play(sep.animate.set_opacity(0.0), margin.animate.set_opacity(1.0),
                  FadeOut(cap2), FadeOut(lane_note), FadeIn(cap3, shift=0.15 * UP), run_time=1.2)
        fisher = Text("Fisher curvature ↔ −margin   ·   Pearson 0.978  (measured)",
                      font="Menlo", color=GREY_B).scale(0.34).next_to(cap3, UP, buff=0.22)
        self.play(FadeIn(fisher), run_time=0.6)
        self.wait(1.6)

        # ── closing thesis ───────────────────────────────────────────────────
        self.play(FadeOut(stamp), FadeOut(cap3), FadeOut(fisher),
                  margin.animate.set_opacity(0.22), run_time=0.9)
        c1 = Text("amortize THIS partition at low bytes", font="Helvetica Neue", weight="BOLD").scale(0.7)
        c2 = Text("spend capacity on the boundary, not the volume",
                  font="Helvetica Neue", color=GREY_B).scale(0.42).next_to(c1, DOWN, buff=0.25)
        c3 = Text("…the level-set task-space witness.",
                  font="Helvetica Neue", color=YELLOW).scale(0.44).next_to(c2, DOWN, buff=0.4)
        self.play(Write(c1), run_time=1.1)
        self.play(FadeIn(c2, shift=0.15 * UP), run_time=0.6)
        self.play(FadeIn(c3, shift=0.15 * UP), run_time=0.7)
        self.wait(2.0)
        self.play(FadeOut(VGroup(c1, c2, c3)), FadeOut(margin), run_time=1.0)
        self.wait(0.3)
