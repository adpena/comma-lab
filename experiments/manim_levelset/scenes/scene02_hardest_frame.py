"""Scene 2 — THE HARDEST FRAME.

Fast-forward the ACTUAL contest video (600 scored pairs), slam to the single
hardest frame for d_seg, then watch the math work on it: the real comma10k /
openpilot argmax partition, the separatrix that IS d_seg, and the margin field
(= the Fisher metric). Every asset is genuine scorer output (NO-FAKE), exported
by scenes/_prep_hardest_frame.py.

Clean layout grammar (one top line + one bottom caption/legend, never stacked)
and deliberate one-idea-per-beat pacing, shared via _style with scene 1.

Render:
    ../../.venv/bin/python scenes/_prep_hardest_frame.py   # once
    ./render.sh -qh scenes/scene02_hardest_frame.py HardestFrame
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from manim import (
    Scene, ImageMobject, Square, VGroup, ValueTracker, always_redraw,
    FadeIn, FadeOut, Write, GrowFromCenter, UP, DOWN, RIGHT, ORIGIN, rate_functions,
)

import _style as st

_ASSETS = Path(__file__).resolve().parent.parent / "assets"
_META = json.loads((_ASSETS / "meta.json").read_text())
_MONT = np.load(_ASSETS / "montage.npy")
_MIDX = np.load(_ASSETS / "montage_idx.npy")
_HARD, _LANE_PX, _BND_PX = _META["hardest_frame"], _META["lane_px"], _META["boundary_px"]

# per-frame dynamic stacks — the math drawn LIVE on the moving footage
_EGO = np.load(_ASSETS / "ego_clip.npy")            # (K,H,W,3) real frames
_SEP_STACK = np.load(_ASSETS / "ego_sep_stack.npy")   # argmax + separatrix per frame
_MARGIN_STACK = np.load(_ASSETS / "ego_margin_stack.npy")  # margin field per frame
_KS = len(_EGO)
_HS = _KS // 2                                       # the hardest frame's clip index

_STAGE_H = 4.9
_STAGE_POS = 0.15 * UP


def _legend():
    rows = VGroup()
    for hexc, name in zip(st.COMMA10K_HEX, st.COMMA10K_LABEL):
        sw = Square(side_length=0.22, fill_color=hexc, fill_opacity=1.0, stroke_width=0)
        lbl = st.caption(name, color=st.MUTED, scale=0.30).next_to(sw, RIGHT, buff=0.12)
        rows.add(VGroup(sw, lbl))
    return rows.arrange(RIGHT, buff=0.46)


class HardestFrame(Scene):
    def _img(self, name: str):
        return ImageMobject(str(_ASSETS / name)).set(height=_STAGE_H).move_to(_STAGE_POS)

    def construct(self) -> None:
        # ── beat 1 · title ───────────────────────────────────────────────────
        card = st.titlecard("02 · the hardest frame", "The Hardest Frame",
                            "600 scored pairs — one is the worst").move_to(ORIGIN)
        self.play(Write(card[1]), run_time=st.T_WRITE)
        self.play(FadeIn(card[0], shift=0.15 * DOWN), GrowFromCenter(card[2]), run_time=st.T_FADE)
        self.play(FadeIn(card[3], shift=0.1 * UP), run_time=st.T_FADE)
        self.wait(st.HOLD)
        self.play(FadeOut(card), run_time=st.T_FADE)

        # ── beat 2 · fast-forward the real video ────────────────────────────
        idx = ValueTracker(0.0)
        scrub = always_redraw(
            lambda: ImageMobject(_MONT[int(np.clip(idx.get_value(), 0, len(_MONT) - 1))])
            .set(height=_STAGE_H).move_to(_STAGE_POS)
        )
        counter = always_redraw(
            lambda: st.corner_tr(st.mono(f"frame {int(_MIDX[int(np.clip(idx.get_value(),0,len(_MONT)-1))]):3d} / 599"))
        )
        ff = st.corner_tl(st.kicker("fast-forward"))
        self.add(scrub, counter)
        self.play(FadeIn(scrub), FadeIn(counter), FadeIn(ff), run_time=st.T_FADE)
        self.play(idx.animate.set_value(len(_MONT) - 1), run_time=3.4, rate_func=rate_functions.ease_in_out_sine)
        self.wait(st.BEAT)
        self.play(FadeOut(scrub), FadeOut(counter), FadeOut(ff), run_time=st.T_FADE)

        # ── beat 3 · land on the hardest frame ──────────────────────────────
        frame = self._img("hardest_frame.png")
        self.play(FadeIn(frame), run_time=st.T_FADE)
        stamp = st.top(st.mono(f"frame {_HARD}  ·  hardest for d_seg"))
        self.play(FadeIn(stamp, shift=0.12 * DOWN), run_time=st.T_FADE)
        self.wait(st.HOLD)

        # ── beat 4 · the openpilot segmentation ─────────────────────────────
        argmax = self._img("hardest_argmax.png"); argmax.set_opacity(0.0)
        self.add(argmax)
        kick = st.corner_tl(st.kicker("openpilot · comma10k"))
        self.play(frame.animate.set_opacity(0.0), argmax.animate.set_opacity(1.0),
                  FadeIn(kick), run_time=st.T_MORPH)
        legend = st.bottom(_legend())
        self.play(FadeIn(legend, lag_ratio=0.12), run_time=st.T_WRITE)
        self.wait(st.HOLD)

        # ── beat 5 · the separatrix ─────────────────────────────────────────
        sep = self._img("hardest_separatrix.png"); sep.set_opacity(0.0)
        self.add(sep)
        cap = st.bottom(st.caption(f"d_seg lives on the boundary — {_BND_PX:,} px — a curve, not a volume", color=st.INK))
        self.play(argmax.animate.set_opacity(0.0), sep.animate.set_opacity(1.0),
                  FadeOut(legend), FadeIn(cap, shift=0.12 * UP), run_time=st.T_MORPH)
        self.wait(st.HOLD)
        note = st.bottom(st.caption(f"the fragile lane class — {_LANE_PX:,} px — is the most flip-prone", color=st.CORAL))
        self.play(FadeOut(cap), FadeIn(note, shift=0.12 * UP), run_time=st.T_FADE)
        self.wait(st.HOLD)

        # ── beat 6 · the margin field = the Fisher metric ───────────────────
        margin = self._img("hardest_margin.png"); margin.set_opacity(0.0)
        self.add(margin)
        fisher = st.top(st.mono("Fisher curvature  ↔  −margin      Pearson 0.978  (measured)", color=st.MUTED))
        cap3 = st.bottom(st.caption("the margin field  =  the Fisher metric", color=st.INK))
        self.play(sep.animate.set_opacity(0.0), margin.animate.set_opacity(1.0),
                  FadeOut(note), FadeOut(stamp), FadeIn(cap3, shift=0.12 * UP),
                  FadeIn(fisher, shift=0.12 * DOWN), run_time=st.T_MORPH)
        self.wait(st.HOLD_LONG)

        # ── beat 6b · NOW WATCH IT MOVE — the separatrix live on the video ───
        # the detailed static breakdown becomes DYNAMIC: the codim-1 boundary
        # that IS d_seg tracks the moving scene, redrawn per displayed frame.
        ci_t = ValueTracker(float(_HS))

        def _sidx() -> int:
            return int(np.clip(ci_t.get_value(), 0, _KS - 1))

        live = always_redraw(
            lambda: ImageMobject(_SEP_STACK[_sidx()]).set(height=_STAGE_H).move_to(_STAGE_POS))
        movekick = st.corner_tl(st.kicker("live · the boundary that IS d_seg"))
        movecap = st.bottom(st.caption(
            "now watch it move — the separatrix tracks the partition as the scene flows", color=st.INK))
        self.add(live)
        self.play(FadeOut(cap3), FadeOut(fisher), FadeOut(kick), margin.animate.set_opacity(0.0),
                  FadeIn(live), FadeIn(movekick), FadeIn(movecap, shift=0.12 * UP), run_time=st.T_MORPH)
        self.play(ci_t.animate.set_value(0.0), run_time=st.T_FADE, rate_func=rate_functions.ease_in_out_sine)
        self.play(ci_t.animate.set_value(_KS - 1), run_time=8.0, rate_func=rate_functions.linear)
        self.play(ci_t.animate.set_value(float(_HS)), run_time=st.T_MORPH, rate_func=rate_functions.ease_in_out_sine)
        self.wait(st.BEAT)
        # freeze: swap the live stack for a static separatrix snapshot for the thesis
        sep_frozen = ImageMobject(_SEP_STACK[_HS]).set(height=_STAGE_H).move_to(_STAGE_POS)
        self.add(sep_frozen); self.bring_to_back(sep_frozen); self.remove(live)

        # ── beat 7 · thesis ─────────────────────────────────────────────────
        self.play(FadeOut(movecap), FadeOut(movekick),
                  sep_frozen.animate.set_opacity(0.18), run_time=st.T_FADE)
        c1 = st.heading("amortize this partition at low bytes", scale=0.66)
        c2 = st.caption("spend capacity on the boundary, not the volume").next_to(c1, DOWN, buff=0.28)
        c3 = st.hero("the level-set task-space witness", scale=0.58).next_to(c2, DOWN, buff=0.44)
        self.play(Write(c1), run_time=st.T_WRITE)
        self.play(FadeIn(c2, shift=0.1 * UP), run_time=st.T_FADE)
        self.play(FadeIn(c3, shift=0.1 * UP), run_time=st.T_FADE)
        self.wait(st.HOLD_LONG)
        self.play(FadeOut(VGroup(c1, c2, c3)), FadeOut(sep_frozen), run_time=st.T_MORPH)
        self.wait(st.BEAT)
