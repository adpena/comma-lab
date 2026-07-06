"""Shared design system for the level-set animations.

Typography + palette informed by 3Blue1Brown (Computer-Modern math, deep-dark
canvas, restraint), comma.ai / openpilot (mono, the signature coral, minimal),
and OpenAI (Inter, generous whitespace, sentence case — never all-caps
shouting). ONE coherent system so every scene reads as one film.

  fonts   Inter (prose)  ·  SF Mono (technical readouts)  ·  Computer Modern (math)
  palette deep near-black canvas · soft near-white ink · one coral hero accent ·
          gold for temperature · a whisper of blue for structure
"""
from __future__ import annotations

from manim import Text, MathTex, Line, VGroup, config, LEFT, RIGHT

# ── palette ──────────────────────────────────────────────────────────────────
BG    = "#0B0B0D"   # canvas — deep, slightly cool near-black
INK   = "#F2F2F4"   # primary text — soft near-white (premium, not pure white)
MUTED = "#8A8A93"   # secondary text
FAINT = "#3C3C44"   # rules / tertiary
CORAL = "#FF5C39"   # THE hero accent (comma) — used sparingly, for emphasis only
GOLD  = "#E9C46A"   # temperature / τ
BLUE  = "#6FB1EA"   # structure / math accent (a whisper)
CYAN  = "#4FD6E0"   # separatrix data glow (matches the field)

# ── fonts (installed OFL — verified resolvable) ──────────────────────────────
DISPLAY = "Space Grotesk"   # headings / hero — distinctive, techy (comma energy)
FONT    = "Inter"           # body / captions — the modern OpenAI/tech sans
MONO    = "JetBrains Mono"  # technical readouts — clean, premium mono

config.background_color = BG

# ── openpilot / comma10k segmentation (THE real class palette + labels) ──────
# canonical comma10k mask colors, contest zero-based order (NON-NEGOTIABLE),
# from src/tac/categorical_candidate_runtime_skeleton.py — the actual scorer's
# classes, so the animation's segmentation IS openpilot's, not a lookalike.
import numpy as _np

COMMA10K_HEX = ["#402020", "#ff0000", "#808060", "#00ff66", "#cc00ff"]
COMMA10K_LABEL = ["road", "lane markings", "undrivable", "movable", "my car"]
COMMA10K_RGB = _np.array(
    [[64, 32, 32], [255, 0, 0], [128, 128, 96], [0, 255, 102], [204, 0, 255]],
    dtype=_np.float64,
)


# ── type scale (all in Manim units) ──────────────────────────────────────────
_H1, _H2, _BODY, _CAP, _KICK, _MONO = 0.95, 0.62, 0.46, 0.38, 0.30, 0.44


def kicker(s: str, color: str = MUTED):
    """Small mono editorial label, e.g. '01 · the separatrix'."""
    return Text(s.upper(), font=MONO, color=color).scale(_KICK)


def heading(s: str, color: str = INK, weight: str = "SEMIBOLD", scale: float = _H1):
    return Text(s, font=FONT, color=color, weight=weight).scale(scale)


def body(s: str, color: str = INK, weight: str = "NORMAL", scale: float = _BODY):
    return Text(s, font=FONT, color=color, weight=weight).scale(scale)


def caption(s: str, color: str = MUTED, scale: float = _CAP):
    return Text(s, font=FONT, color=color, weight="NORMAL").scale(scale)


def hero(s: str, scale: float = 0.7):
    """The one coral, bold emphasis line per scene."""
    return Text(s, font=FONT, color=CORAL, weight="BOLD").scale(scale)


def mono(s: str, color: str = GOLD, scale: float = _MONO):
    return Text(s, font=MONO, color=color).scale(scale)


def eq(tex: str, color: str = INK, scale: float = 0.72):
    return MathTex(tex, color=color).scale(scale)


def rule(width: float = 1.6, color: str = FAINT, stroke: float = 1.5):
    return Line(LEFT * width / 2, RIGHT * width / 2, color=color, stroke_width=stroke)


def titlecard(kick: str, title: str, sub: str):
    """A composed, spaced title block: kicker · heading · thin rule · subtitle."""
    k = kicker(kick)
    h = heading(title)
    r = rule(2.2)
    s = caption(sub)
    g = VGroup(k, h, r, s).arrange_submobjects(
        direction=__import__("manim").DOWN, buff=0.28
    )
    # tighten kicker→heading, loosen rule spacing for an editorial feel
    h.next_to(k, __import__("manim").DOWN, buff=0.22)
    r.next_to(h, __import__("manim").DOWN, buff=0.30)
    s.next_to(r, __import__("manim").DOWN, buff=0.26)
    return VGroup(k, h, r, s)
