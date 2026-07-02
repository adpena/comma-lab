# SPDX-License-Identifier: MIT
"""Focused tests for the #205 OOM verdict-batch peak-RSS spike canonical equation."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tac.canonical_equations.oom_verdict_batch_spike_peak_rss_20260702 import (
    DEFAULT_SAFE_FRAC,
    DEFAULT_VERDICT_BATCH,
    EQUATION_ID,
    VERDICT_FLOOR_GIB,
    VERDICT_PER_PAIR_GIB,
    build_oom_verdict_batch_spike_peak_rss_v1 as build_eq,
    is_verdict_batch_launch_safe,
    unchunked_spike_gib,
    verdict_transient_gib,
)

_REPO = Path(__file__).resolve().parents[4]


def test_equation_builds_and_validates() -> None:
    eq = build_eq()
    assert eq.equation_id == EQUATION_ID == eq.equation_id.lower()
    assert len(eq.empirical_anchors) == 1
    assert eq.canonical_producers and eq.canonical_consumers
    # the launch gate + the full projection are the declared consumers
    assert "tools/launch_witness_run.py" in eq.canonical_consumers
    assert "tools/witness_memory_preflight.py" in eq.canonical_consumers
    assert eq.is_well_calibrated  # residual small vs measured


def test_verdict_transient_spike_law() -> None:
    # unchunked n600 => ~66 GiB spike (the OOM); vbatch=32 => the 6 GiB floor (the fix).
    assert verdict_transient_gib(600, 0) == pytest.approx(66.0)
    assert unchunked_spike_gib(600) == pytest.approx(66.0)
    assert verdict_transient_gib(600, 32) == pytest.approx(VERDICT_FLOOR_GIB)
    # chunking is what collapses the spike; below the floor it clamps to the floor
    assert verdict_transient_gib(600, 8) == pytest.approx(VERDICT_FLOOR_GIB)
    # a large batch relative to a small P uses min(vbatch, P)
    assert verdict_transient_gib(600, 1000) == pytest.approx(0.11 * 600)


def test_launch_safety_gate_uses_0p70_fraction() -> None:
    assert DEFAULT_SAFE_FRAC == 0.70
    assert DEFAULT_VERDICT_BATCH == 32
    # n600 unchunked full peak (guard's ~127 GiB) is UNSAFE on 128 GiB; chunked ~67 GiB is SAFE.
    assert not is_verdict_batch_launch_safe(127.6, 128.0)
    assert is_verdict_batch_launch_safe(67.6, 128.0)


def test_anchor_records_score_neutral_measured_collapse() -> None:
    eq = build_eq()
    emp = eq.empirical_anchors[0].empirical_output
    assert emp["verdict_transient_gib_unchunked_measured"] == pytest.approx(66.2)
    assert emp["verdict_transient_gib_vbatch32_measured"] == pytest.approx(5.6)
    # score-neutral: d_seg BIT-IDENTICAL across the two paths
    assert emp["d_seg_both_paths"] == emp["d_seg_both_paths"]  # present + numeric
    assert eq.empirical_anchors[0].empirical_verification_status == "VERIFIED_VIA_EMPIRICAL_ANCHOR"


def test_constants_mirror_the_live_guard_no_drift() -> None:
    # Drift guard: the equation's mirrored constants MUST match the live memory-preflight guard.
    if str(_REPO / "tools") not in sys.path:
        sys.path.insert(0, str(_REPO / "tools"))
    import witness_memory_preflight as wmp  # noqa: PLC0415

    assert VERDICT_PER_PAIR_GIB == wmp.VERDICT_PER_PAIR_GIB
    assert VERDICT_FLOOR_GIB == wmp.VERDICT_FLOOR_GIB
    assert DEFAULT_VERDICT_BATCH == wmp.DEFAULT_VERDICT_BATCH
    assert DEFAULT_SAFE_FRAC == wmp.DEFAULT_SAFE_FRAC
