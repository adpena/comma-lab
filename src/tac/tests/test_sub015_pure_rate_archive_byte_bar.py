# SPDX-License-Identifier: MIT
"""Tests for the sub-0.15 pure-rate archive byte bar.

The law exists to retire COPIED byte counts, so the tests that matter are the ones
that prove (a) the bar is the same off every base that shares the frontier's frozen
distortion, (b) it is DIFFERENT off a base whose distortion differs -- which is what
makes the pure-rate caveat load-bearing rather than decorative -- and (c) the
pointer-reading path fails closed instead of falling back to a default.
"""
from __future__ import annotations

import json
from decimal import Decimal, getcontext

import pytest

from tac.canonical_equations import sub015_pure_rate_archive_byte_bar_20260816 as law

# Complete-S rows, [contest-CUDA T4 n600], from the primary verdict memos.
CP135 = (0.16195513827824176, 186_252)
MC36 = (0.1619344578804448, 186_269)
E480B_V2 = (0.1600920261571558, 183_502)
HV1 = (0.15959729295498598, 182_759)


def test_bar_is_identical_across_the_frozen_distortion_lineage() -> None:
    """MC36 -> e480b v2 -> hv1 are pure-rate steps, so they share one bar."""
    bars = [law.pure_rate_byte_bar(*base) for base in (MC36, E480B_V2, HV1)]
    assert max(bars) - min(bars) < 1e-6, "frozen distortion must give one shared bar"
    assert round(bars[-1], 4) == law.BAR_BYTES_FROZEN_DISTORTION_LINEAGE


def test_cp135_gives_a_different_bar_because_mc36_changed_distortion() -> None:
    """The counter-example that scopes the identity. ddm_fb1 claimed all FOUR bases
    agree; cp135 does not, because cp135 -> MC36 moved seg by -37 flips."""
    live = law.pure_rate_byte_bar(*HV1)
    cp135 = law.pure_rate_byte_bar(*CP135)
    assert round(cp135, 4) == law.BAR_BYTES_CP135_SUPERSEDED_DISTORTION
    assert cp135 != pytest.approx(live, abs=1.0)
    assert cp135 - live == pytest.approx(law.CP135_BAR_OFFSET_BYTES, abs=1e-3)
    # the offset is the distortion move, not a rate artefact
    assert law.distortion_leg(*CP135) - law.distortion_leg(*HV1) == pytest.approx(3.2e-5, rel=1e-6)


def test_required_cut_off_the_live_frontier() -> None:
    assert law.required_cut_bytes(*HV1) == pytest.approx(14413.4023, abs=1e-3)
    assert law.required_cut_bytes(*E480B_V2) == pytest.approx(15156.4023, abs=1e-3)
    # the rung moves by exactly the pointer's byte step, which is why a DELTA goes
    # stale on every rate move and this derivation does not
    assert law.required_cut_bytes(*E480B_V2) - law.required_cut_bytes(*HV1) == pytest.approx(
        E480B_V2[1] - HV1[1], abs=1e-6
    )


def test_matches_exact_decimal_arithmetic() -> None:
    """Float path must agree with a 44-digit reference, not merely look plausible."""
    getcontext().prec = 44
    n = Decimal(law.RATE_DENOMINATOR_BYTES)
    s, b = Decimal(str(HV1[0])), Decimal(HV1[1])
    exact = (Decimal("0.15") - (s - Decimal(25) * b / n)) * n / Decimal(25)
    assert law.pure_rate_byte_bar(*HV1) == pytest.approx(float(exact), abs=1e-6)


def test_negative_bar_when_distortion_alone_exceeds_the_target() -> None:
    """Honest arithmetic, not an error: no rate work reaches a target the distortion
    legs already blow through."""
    assert law.pure_rate_byte_bar(0.40, 100_000) < 0


def test_denominator_is_an_input_not_a_constant_of_nature() -> None:
    assert law.pure_rate_byte_bar(*HV1, denominator=40_000_000) != pytest.approx(
        law.pure_rate_byte_bar(*HV1), abs=1.0
    )
    with pytest.raises(ValueError):
        law.pure_rate_byte_bar(*HV1, denominator=0)


def test_pointer_path_agrees_with_the_direct_derivation_and_names_its_base() -> None:
    got = law.pure_rate_byte_bar_from_pointer()
    assert got["bar_bytes"] == pytest.approx(law.pure_rate_byte_bar(got["base_score"], got["base_archive_bytes"]))
    assert got["base_archive_sha256"]
    assert got["score_claim"] is False
    assert "PURE-RATE" in got["valid_only_for"]


def test_pointer_path_fails_closed_on_a_broken_pointer(tmp_path) -> None:
    """A bar that silently falls back to a hardcoded default is the defect wearing the
    cure's clothes."""
    (tmp_path / ".omx" / "state").mkdir(parents=True)
    (tmp_path / ".omx" / "state" / "canonical_frontier_pointer.json").write_text(
        json.dumps({"effective_frontier": {}}), encoding="utf-8"
    )
    with pytest.raises(KeyError):
        law.pure_rate_byte_bar_from_pointer(repo_root=tmp_path)
    with pytest.raises(FileNotFoundError):
        law.pure_rate_byte_bar_from_pointer(repo_root=tmp_path / "absent")


def test_equation_registers_with_the_caveat_encoded_not_merely_mentioned() -> None:
    eq = law.build_sub015_pure_rate_archive_byte_bar_v1()
    assert eq.equation_id == "sub015_pure_rate_archive_byte_bar_v1"
    dov = eq.domain_of_validity
    assert "does_not_apply_to" in dov
    assert "d_seg" in dov["does_not_apply_to"] and "d_pose" in dov["does_not_apply_to"]
    assert "re-measure" in dov["does_not_apply_to"]
    anchor = eq.empirical_anchors[0]
    assert anchor.empirical_output["cp135_counter_example"]["verdict"].startswith("REFUTED")
    assert anchor.provenance.score_claim_valid is False


def test_pointer_path_refuses_an_upstream_only_frontier(tmp_path) -> None:
    """effective_frontier can select the upstream leaderboard, whose archive we do not
    hold. Deriving a byte bar off it would be false authority."""
    (tmp_path / ".omx" / "state").mkdir(parents=True)
    (tmp_path / ".omx" / "state" / "canonical_frontier_pointer.json").write_text(
        json.dumps(
            {
                "effective_frontier": {
                    "source": "upstream_leaderboard_snapshot",
                    "score": 0.162,
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(KeyError, match="no local archive record"):
        law.pure_rate_byte_bar_from_pointer(repo_root=tmp_path)
