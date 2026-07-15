# SPDX-License-Identifier: MIT
"""DSL-leg tests for the #B-4 AdaptiveGradClip + #509 LaneBandStaticCache levers.

The levers must be DSL-held (never hand flags), LawRef-custodied where scientific,
and composable by bare name (zero-required-arg single-Lever factories) so the
launcher's ``--dsl-lever`` path can arm them. $0 pure-python."""
from __future__ import annotations

from tac.witness_dsl.curriculum_dsl import AdaptiveGradClip, LaneBandStaticCache, Lever


def test_adaptive_grad_clip_overrides_arm_autoclip_mode():
    lever = AdaptiveGradClip()  # default False = launcher-composable (round-trip bug, memo §2c)
    assert isinstance(lever, Lever)
    assert lever.name == "adaptive_grad_clip_autoclip"
    assert lever.overrides["--grad-clip-mode"] == "autoclip"
    assert lever.overrides["--grad-clip-percentile"] == 10.0
    assert lever.overrides["--grad-clip-window"] == 1000
    assert lever.overrides["--grad-clip-warmup-steps"] == 10


def test_adaptive_grad_clip_scientific_declaration_carries_lawref_custody():
    lever = AdaptiveGradClip(scientific_declaration=True)  # explicit: spec-authored path
    # the three NUMERIC constants carry LawRef custody; the string mode flag cannot
    # (LawRef inputs are numeric) and stays a plain override.
    for flag in ("--grad-clip-percentile", "--grad-clip-window", "--grad-clip-warmup-steps"):
        assert flag in lever.lawrefs, flag
        assert lever.lawrefs[flag].equation_id == "autoclip_percentile_threshold_v1"
        assert flag in lever.constant_manifest
        assert lever.runtime_receipt_schemas[flag] == "v9_config_compile.v1"
    assert "--grad-clip-mode" not in lever.lawrefs
    assert lever.constant_refs == lever.lawrefs  # provenance-gate alias


def test_adaptive_grad_clip_custom_constants_flow_through():
    lever = AdaptiveGradClip(percentile=25.0, window=500, warmup_steps=3,
                             scientific_declaration=False)
    assert lever.overrides["--grad-clip-percentile"] == 25.0
    assert lever.overrides["--grad-clip-window"] == 500
    assert lever.overrides["--grad-clip-warmup-steps"] == 3


def test_lane_band_static_cache_lever_both_arms():
    on = LaneBandStaticCache()
    off = LaneBandStaticCache(enabled=False)
    assert on.overrides == {"--lane-band-cache-static": True}
    assert off.overrides == {"--lane-band-cache-static": False}
    assert on.name == "lane_band_static_cache"


def test_both_levers_are_composable_by_bare_name():
    from tac.witness_dsl.lever_registry import name_composable_levers

    names = name_composable_levers()
    assert "AdaptiveGradClip" in names
    assert "LaneBandStaticCache" in names


def test_trainer_argparse_accepts_the_lever_flags():
    """never-invent-flags: every emitted flag must exist on the real trainer parser."""
    from tac.witness_dsl.lever_registry import completeness

    comp = completeness()
    for flag in ("--grad-clip-mode", "--grad-clip-percentile", "--grad-clip-window",
                 "--grad-clip-warmup-steps", "--lane-band-cache-static"):
        assert flag not in comp.unmapped, f"{flag} should be DSL-mapped"
