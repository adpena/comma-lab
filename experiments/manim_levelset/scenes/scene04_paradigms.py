"""Scene 4 — TWO PARADIGMS  (PR95/HNeRV full-RGB reconstruction  vs  the task-space witness).

The conceptual heart of the capstone, told as a fair fight under ONE ruler.

The PR95/HNeRV family (the medal-class leaderboard winners) and our task-space
level-set witness are scored by the SAME contest score. They make OPPOSITE bets
on where the bytes go:

  PR95/HNeRV  →  optimizes full-RGB reconstruction with a UNIFORM coordinate
                 decoder (PixelShuffle + sin); bytes spread evenly across the
                 frame; spectral bias → Gibbs ringing exactly at the argmax edge.
  witness     →  optimizes the SCORER directly (amortizes the SegNet argmax, no
                 RGB); the flat interiors are FREE, so it reallocates the same
                 bytes onto the ~0.72% boundary band; a curvelet / step-native
                 basis matched to the piecewise-constant target (no Gibbs).

NO-FAKE (the load-bearing honesty of this scene): PR95 is the MEASURED frontier
(S = 0.19110, contest-CPU, a real medal-class result). The witness is the DESIGN
/ hypothesis — target sub-0.15 — and is NOT yet byte-closed below PR95. The
pointer moves only through one byte-closed upstream/evaluate.py exact row. This
scene shows the two paradigms + WHY we believe the reallocation dominates; it
does NOT claim a win the exact eval has not yet delivered.

Measured numbers (from our own record; see CLAUDE.md §WITNESS CAPSTONE + the DAG):
  S = 100·d_seg + √(10·d_pose) + 25·bytes/N,  N = 37,545,489  (upstream/evaluate.py)
  frontier 0.19110 ≈ rate 0.118 + 100·d_seg(≈5.6e-4)=0.056 + √(10·d_pose)=0.018
  trilemma: bc20 (shrunk)  rate 0.059 · d_seg under-capacity → S ≈ 0.31
            bc36 (PR95-size) d_seg ≈ 6e-4 · rate 0.118      → S ≈ 0.19  (= just PR95)
            witness            adequate d_seg AT low rate    → the sub-0.15 arm

Render:  ./render.sh -qh scenes/scene04_paradigms.py Paradigms
"""
from __future__ import annotations

from manim import (
    Scene, VGroup, Rectangle, Line, Dot, MathTex, FadeIn, FadeOut, Write, Create,
    GrowFromCenter, UP, DOWN, LEFT, RIGHT, ORIGIN,
)

import _style as st


class Paradigms(Scene):
    # ── small helpers (kept local; the film-wide grammar lives in _style) ────
    def _col_lines(self, lines, accent):
        """A column of short body lines; first line is a mono header in `accent`."""
        items = [st.mono(lines[0], color=accent, scale=0.40)]
        items += [st.body(s, color=st.INK, scale=0.40) for s in lines[1:]]
        g = VGroup(*items).arrange(DOWN, aligned_edge=LEFT, buff=0.30)
        return g

    def _uniform_grid(self, accent):
        """A frame whose CAPACITY is spread evenly — a dot in every cell, the
        boundary treated no differently from the flat interior."""
        box = Rectangle(width=3.4, height=2.3, color=st.FAINT, stroke_width=2.0)
        dots = VGroup()
        nx, ny = 7, 5
        for i in range(nx):
            for j in range(ny):
                x = -1.7 + 3.4 * (i + 0.5) / nx
                y = -1.15 + 2.3 * (j + 0.5) / ny
                dots.add(Dot([x, y, 0], radius=0.035, color=accent).set_opacity(0.75))
        return VGroup(box, dots)

    def _boundary_band(self, accent):
        """The same frame, capacity REALLOCATED — a curved boundary band lit, the
        flat interiors dark (free). This is where d_seg lives."""
        box = Rectangle(width=3.4, height=2.3, color=st.FAINT, stroke_width=2.0)
        # a gently curved separatrix across the box (the codim-1 boundary)
        pts = []
        import numpy as np
        xs = np.linspace(-1.6, 1.6, 40)
        ys = 0.55 * np.sin(0.9 * xs + 0.4) - 0.1
        band = VGroup()
        for x, y in zip(xs, ys):
            band.add(Dot([x, y, 0], radius=0.045, color=accent))
        curve = VGroup(*band)
        return VGroup(box, curve)

    def construct(self) -> None:
        # ── beat 1 · title ───────────────────────────────────────────────────
        card = st.titlecard("04 · two paradigms",
                            "Reconstruction  vs  the witness",
                            "same ruler — opposite bets on where the bytes go").move_to(ORIGIN)
        self.play(Write(card[1]), run_time=st.T_WRITE)
        self.play(FadeIn(card[0], shift=0.15 * DOWN), GrowFromCenter(card[2]), run_time=st.T_FADE)
        self.play(FadeIn(card[3], shift=0.1 * UP), run_time=st.T_FADE)
        self.wait(st.HOLD)
        self.play(FadeOut(card), run_time=st.T_FADE)

        # ── beat 2 · the shared ruler (neither can hide) ────────────────────
        ruler = st.eq(r"S \;=\; 100\,d_{\mathrm{seg}} \;+\; \sqrt{10\,d_{\mathrm{pose}}}"
                      r"\;+\; 25\,\tfrac{\mathrm{bytes}}{N}", scale=0.8).move_to(0.7 * UP)
        cap = st.bottom(st.caption("both paradigms are scored by the same three terms — "
                                   "neither can hide"))
        self.play(Write(ruler), run_time=st.T_WRITE)
        self.play(FadeIn(cap, shift=0.12 * UP), run_time=st.T_FADE)
        self.wait(st.HOLD)
        # decompose the MEASURED frontier under the ruler
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

        # ── beat 3 · PR95/HNeRV — reconstruct the pixels, uniformly ─────────
        head_l = st.kicker("PR95 / HNeRV — reconstruct", color=st.CORAL).to_edge(UP, buff=0.7)
        grid = self._uniform_grid(st.CORAL).move_to(0.3 * UP)
        txt_l = self._col_lines([
            "optimizes  ∑ |x − x̂|²   (full RGB)",
            "uniform coordinate decoder",
            "PixelShuffle + bilinear-skip + sin",
            "bytes spread EVENLY over the frame",
            "spectral bias → Gibbs at the edge",
        ], st.CORAL)
        txt_l.scale(0.9).to_edge(DOWN, buff=0.7)
        self.play(FadeIn(head_l, shift=0.1 * DOWN), Create(grid), run_time=st.T_WRITE)
        self.play(FadeIn(txt_l, shift=0.1 * UP), run_time=st.T_FADE)
        self.wait(st.HOLD_LONG)
        self.play(FadeOut(head_l), FadeOut(grid), FadeOut(txt_l), run_time=st.T_FADE)

        # ── beat 4 · the trilemma — both horns of a full-RGB decoder ────────
        tri_head = st.heading("a full-RGB decoder can't have both", scale=0.62).to_edge(UP, buff=0.9)
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

        # ── beat 5 · the witness — reallocate onto the boundary ─────────────
        head_r = st.kicker("the witness — reproduce the scorer", color=st.CYAN).to_edge(UP, buff=0.7)
        band = self._boundary_band(st.CYAN).move_to(0.3 * UP)
        txt_r = self._col_lines([
            "optimizes  d_seg  directly  (no RGB)",
            "flat interiors are FREE (constant)",
            "the 0.72% boundary band is ALL scored",
            "reallocate the SAME bytes onto it",
            "curvelet / step-native — no Gibbs",
        ], st.CYAN)
        txt_r.scale(0.9).to_edge(DOWN, buff=0.7)
        self.play(FadeIn(head_r, shift=0.1 * DOWN), Create(band), run_time=st.T_WRITE)
        self.play(FadeIn(txt_r, shift=0.1 * UP), run_time=st.T_FADE)
        self.wait(st.HOLD_LONG)
        self.play(FadeOut(head_r), FadeOut(band), FadeOut(txt_r), run_time=st.T_FADE)

        # ── beat 6 · the honest state (NO-FAKE payoff) ──────────────────────
        l1 = st.body("PR95 is the MEASURED frontier —  S = 0.19110.", scale=0.5)
        l2 = st.body("the witness is the DESIGN.  target sub-0.15.", scale=0.5)
        l3 = st.body("not yet byte-closed below PR95.", color=st.MUTED, scale=0.46)
        l4 = st.caption("the pointer moves only through one byte-closed exact eval", color=st.MUTED)
        stack = VGroup(l1, l2, l3, l4).arrange(DOWN, buff=0.34).move_to(0.6 * UP)
        self.play(FadeIn(l1, shift=0.1 * UP), run_time=st.T_FADE)
        self.play(FadeIn(l2, shift=0.1 * UP), run_time=st.T_FADE)
        self.play(FadeIn(l3, shift=0.1 * UP), FadeIn(l4, shift=0.1 * UP), run_time=st.T_FADE)
        self.wait(st.HOLD)
        punch = st.hero("same ruler.  the bet is where the bytes go.").next_to(stack, DOWN, buff=0.7)
        self.play(Write(punch), run_time=st.T_WRITE)
        self.wait(st.HOLD_LONG)
        self.play(FadeOut(stack), FadeOut(punch), run_time=st.T_FADE)
