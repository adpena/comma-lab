"""Scene 4 — TWO PARADIGMS  (PR95/HNeRV full-RGB reconstruction  vs  the task-space witness).

The conceptual heart of the capstone, told as a fair fight under ONE ruler — on
the REAL contest frame and the REAL SegNet partition.

The PR95/HNeRV family (the medal-class leaderboard winners) and our task-space
level-set witness are scored by the SAME contest score. They make OPPOSITE bets
on where the bytes go:

  PR95/HNeRV  →  reconstructs full RGB with a UNIFORM coordinate decoder
                 (PixelShuffle + sin); bytes spread evenly over EVERY pixel;
                 spectral bias → Gibbs ringing exactly at the argmax edge.
  witness     →  reproduces the SCORER (amortizes the SegNet argmax, no RGB);
                 the flat interiors are FREE, so it pours the SAME bytes onto the
                 measured 0.72% boundary band — ~139× the byte-density where
                 d_seg actually lives — in a curvelet/step-native basis that is
                 the provably-optimal sparse chart for a curved codim-1 edge
                 (Candès–Donoho): it minimizes distortion AND rate at once.

NO-FAKE (the load-bearing honesty): PR95 is the MEASURED frontier (S = 0.19110,
contest-CPU, a real medal-class result). The witness's advantage is a MEASURED-
grounded PREDICTION (interiors free #139; 0.72% binding residual; bc20 rate
0.059 vs 0.118; curvelet optimality) — but it is NOT yet byte-closed below PR95.
The exact eval is the only judge, and it has not ruled yet. This scene makes the
case that the reallocation dominates; it does not claim a win the pointer has
not recorded.

Measured numbers (from our own record; CLAUDE.md §WITNESS CAPSTONE + the DAG):
  S = 100·d_seg + √(10·d_pose) + 25·bytes/N,  N = 37,545,489  (upstream/evaluate.py)
  frontier 0.19110 ≈ rate 0.118 + 100·d_seg(≈5.6e-4)=0.056 + √(10·d_pose)=0.018
  trilemma: bc20 (shrunk)  rate 0.059 · d_seg under-capacity → S ≈ 0.31
            bc36 (PR95-size) d_seg ≈ 6e-4 · rate 0.118      → S ≈ 0.19  (= just PR95)
            witness            adequate d_seg AT low rate    → the sub-0.15 arm
  reallocation: d_seg lives on 0.72% of pixels → ~139× byte-density vs uniform.

Render:  ./render.sh -qh scenes/scene04_paradigms.py Paradigms
"""
from __future__ import annotations

from pathlib import Path

import _style as st
from manim import (
    DOWN,
    LEFT,
    ORIGIN,
    UP,
    Create,
    Dot,
    FadeIn,
    FadeOut,
    GrowFromCenter,
    ImageMobject,
    Scene,
    VGroup,
    Write,
)

_ASSETS = Path(__file__).resolve().parent.parent / "assets"
_STAGE_H = 4.4


class Paradigms(Scene):
    # ── helpers ──────────────────────────────────────────────────────────────
    def _img(self, name: str):
        return ImageMobject(str(_ASSETS / name)).set(height=_STAGE_H).move_to(0.35 * UP)

    def _col_lines(self, lines, accent):
        items = [st.mono(lines[0], color=accent, scale=0.40)]
        items += [st.body(s, color=st.INK, scale=0.40) for s in lines[1:]]
        return VGroup(*items).arrange(DOWN, aligned_edge=LEFT, buff=0.28)

    def _grid_over(self, img, accent, nx=8, ny=6):
        """A uniform dot grid covering the image extent — capacity spread evenly,
        the boundary treated no differently from the flat interior."""
        cx, cy, _ = img.get_center()
        w, h = img.width, img.height
        dots = VGroup()
        for i in range(nx):
            for j in range(ny):
                x = cx - w / 2 + w * (i + 0.5) / nx
                y = cy - h / 2 + h * (j + 0.5) / ny
                dots.add(Dot([x, y, 0], radius=0.03, color=accent).set_opacity(0.85))
        return dots

    def construct(self) -> None:
        # ── beat 1 · title ───────────────────────────────────────────────────
        card = st.titlecard("04 · two paradigms",
                            "Reconstruct  vs  the witness",
                            "same ruler — opposite bets on where the bytes go").move_to(ORIGIN)
        self.play(Write(card[1]), run_time=st.T_WRITE)
        self.play(FadeIn(card[0], shift=0.15 * DOWN), GrowFromCenter(card[2]), run_time=st.T_FADE)
        self.play(FadeIn(card[3], shift=0.1 * UP), run_time=st.T_FADE)
        self.wait(st.HOLD)
        self.play(FadeOut(card), run_time=st.T_FADE)

        # ── beat 2 · the shared ruler + the MEASURED frontier ───────────────
        ruler = st.eq(r"S \;=\; 100\,d_{\mathrm{seg}} \;+\; \sqrt{10\,d_{\mathrm{pose}}}"
                      r"\;+\; 25\,\tfrac{\mathrm{bytes}}{N}", scale=0.8).move_to(0.7 * UP)
        cap = st.bottom(st.caption("both paradigms are scored by the same three terms — "
                                   "neither can hide"))
        self.play(Write(ruler), run_time=st.T_WRITE)
        self.play(FadeIn(cap, shift=0.12 * UP), run_time=st.T_FADE)
        self.wait(st.HOLD)
        decomp = st.eq(r"0.19110 \;\approx\; \underbrace{0.118}_{\text{rate}}"
                       r"\;+\; \underbrace{0.056}_{100\,d_{\mathrm{seg}}}"
                       r"\;+\; \underbrace{0.018}_{\sqrt{10\,d_{\mathrm{pose}}}}",
                       scale=0.62).next_to(ruler, DOWN, buff=0.7)
        cap2 = st.bottom(st.caption("the measured frontier — a real medal-class result  "
                                    "[contest-CPU]", color=st.INK))
        self.play(FadeIn(decomp, shift=0.1 * UP), FadeOut(cap), FadeIn(cap2, shift=0.12 * UP),
                  run_time=st.T_MORPH)
        self.wait(st.HOLD_LONG)
        self.play(FadeOut(ruler), FadeOut(decomp), FadeOut(cap2), run_time=st.T_FADE)

        # ── beat 3 · PR95/HNeRV — reconstruct EVERY pixel, uniformly ────────
        frame = self._img("hardest_frame.png")
        head_l = st.kicker("PR95 / HNeRV — reconstruct the pixels", color=st.CORAL).to_edge(UP, buff=0.6)
        self.play(FadeIn(frame), FadeIn(head_l, shift=0.1 * DOWN), run_time=st.T_FADE)
        grid = self._grid_over(frame, st.CORAL)
        capL = st.bottom(st.caption("uniform coordinate decoder — the same bytes on every pixel, "
                                    "boundary and flat road alike"))
        self.play(Create(grid), FadeIn(capL, shift=0.12 * UP), run_time=st.T_WRITE)
        self.wait(st.HOLD)
        capL2 = st.bottom(st.caption("spectral bias → Gibbs ringing exactly at the argmax edge",
                                     color=st.CORAL))
        self.play(FadeOut(capL), FadeIn(capL2, shift=0.12 * UP), run_time=st.T_FADE)
        self.wait(st.HOLD_LONG)
        self.play(FadeOut(frame), FadeOut(grid), FadeOut(head_l), FadeOut(capL2), run_time=st.T_FADE)

        # ── beat 4 · the trilemma — a full-RGB decoder can't have both ──────
        tri_head = st.heading("a full-RGB decoder can't have both", scale=0.6).to_edge(UP, buff=0.9)
        rows = VGroup(
            st.mono("bc20  shrink it    rate 0.059  ·  d_seg under-capacity   →  S ≈ 0.31",
                    color=st.MUTED, scale=0.42),
            st.mono("bc36  PR95-size    d_seg ≈ 6e-4  ·  rate 0.118           →  S ≈ 0.19",
                    color=st.MUTED, scale=0.42),
            st.mono("witness            adequate d_seg  AT  low rate          →  sub-0.15 arm",
                    color=st.CYAN, scale=0.42),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.42).move_to(0.1 * DOWN)
        cap3 = st.bottom(st.caption("shrink → loses d_seg;  keep size → pays the rate.  "
                                    "both horns dominated"))
        self.play(Write(tri_head), run_time=st.T_WRITE)
        self.play(FadeIn(rows[0], shift=0.1 * UP), run_time=st.T_FADE)
        self.play(FadeIn(rows[1], shift=0.1 * UP), run_time=st.T_FADE)
        self.play(FadeIn(cap3, shift=0.12 * UP), run_time=st.T_FADE)
        self.wait(st.HOLD)
        self.play(FadeIn(rows[2], shift=0.1 * UP), run_time=st.T_MORPH)   # the escape
        self.wait(st.HOLD_LONG)
        self.play(FadeOut(tri_head), FadeOut(rows), FadeOut(cap3), run_time=st.T_FADE)

        # ── beat 5 · the witness — reproduce the SCORER, then the separatrix ─
        seg = self._img("hardest_argmax.png")
        head_r = st.kicker("the witness — reproduce what the scorer SEES", color=st.CYAN).to_edge(UP, buff=0.6)
        capR = st.bottom(st.caption("not the pixels — the SegNet argmax partition.  the flat "
                                    "interiors are FREE (locally constant)"))
        self.play(FadeIn(seg), FadeIn(head_r, shift=0.1 * DOWN), run_time=st.T_FADE)
        self.play(FadeIn(capR, shift=0.12 * UP), run_time=st.T_FADE)
        self.wait(st.HOLD)
        # crossfade the partition into the REAL measured separatrix (the codim-1 boundary)
        sep = self._img("hardest_separatrix.png")
        capR2 = st.bottom(st.caption("d_seg lives on the 0.72% separatrix — the codim-1 boundary.  "
                                     "pour the SAME bytes THERE", color=st.CYAN))
        self.play(FadeOut(seg), FadeIn(sep), FadeOut(capR), FadeIn(capR2, shift=0.12 * UP),
                  run_time=st.T_MORPH)
        self.wait(st.HOLD_LONG)
        self.play(FadeOut(sep), FadeOut(head_r), FadeOut(capR2), run_time=st.T_FADE)

        # ── beat 6 · why the reallocation should dominate (predicted, grounded) ─
        why_head = st.heading("same bytes — 139× the density where it counts", scale=0.58).to_edge(UP, buff=0.9)
        why = VGroup(
            st.mono("uniform    100% of pixels    →   1× byte-density on the edge",
                    color=st.MUTED, scale=0.42),
            st.mono("witness    0.72% of pixels   →  ~139× byte-density on the edge",
                    color=st.CYAN, scale=0.42),
            st.body("+ curvelet / step-native is the optimal sparse chart for a curved",
                    color=st.INK, scale=0.40),
            st.body("  codim-1 edge (Candès–Donoho): it cuts distortion AND rate at once.",
                    color=st.INK, scale=0.40),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.34).move_to(0.0 * DOWN)
        self.play(Write(why_head), run_time=st.T_WRITE)
        self.play(FadeIn(why[0], shift=0.1 * UP), run_time=st.T_FADE)
        self.play(FadeIn(why[1], shift=0.1 * UP), run_time=st.T_MORPH)
        self.play(FadeIn(why[2:], shift=0.1 * UP), run_time=st.T_FADE)
        self.wait(st.HOLD_LONG)
        self.play(FadeOut(why_head), FadeOut(why), run_time=st.T_FADE)

        # ── beat 7 · the honest close (NO-FAKE) ─────────────────────────────
        l1 = st.body("the arithmetic says the reallocation dominates.", scale=0.5)
        l2 = st.body("PR95 is the MEASURED frontier — 0.19110.", scale=0.5)
        l3 = st.body("the witness is that prediction — target sub-0.15,", color=st.MUTED, scale=0.46)
        l4 = st.body("not yet byte-closed below it.", color=st.MUTED, scale=0.46)
        l5 = st.caption("the exact eval is the only judge — and it hasn't ruled yet", color=st.MUTED)
        stack = VGroup(l1, l2, l3, l4, l5).arrange(DOWN, buff=0.30).move_to(0.5 * UP)
        self.play(FadeIn(l1, shift=0.1 * UP), run_time=st.T_FADE)
        self.play(FadeIn(l2, shift=0.1 * UP), run_time=st.T_FADE)
        self.play(FadeIn(VGroup(l3, l4, l5), shift=0.1 * UP), run_time=st.T_FADE)
        self.wait(st.HOLD)
        punch = st.hero("same ruler.  the bet is where the bytes go.").next_to(stack, DOWN, buff=0.6)
        self.play(Write(punch), run_time=st.T_WRITE)
        self.wait(st.HOLD_LONG)
        self.play(FadeOut(stack), FadeOut(punch), run_time=st.T_FADE)
