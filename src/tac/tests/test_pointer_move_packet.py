# SPDX-License-Identifier: MIT
"""The pointer-move packet, pinned against the two rows it was replayed on.

These fixtures are the SCALAR fields of the two harvested contest-CUDA receipts of
2026-09-04 (fs1 `fc-01M1PM1KR3CQN5E5BC62WE4AD7`, fs2 `fc-01M1Q6W3R8WWDQPRFYSF7SWTKP`),
copied verbatim. Pinning them here is what makes the replay reproducible without the
SSD custody paths being mounted: the whole claim of this apparatus is that the memo
numbers are a FUNCTION of these fields, so the function is asserted, not described.
"""

from __future__ import annotations

import math

import pytest

from tac.pointer_move import (
    HarvestRefusal,
    PacketPlan,
    PriorAnchor,
    cross_check_against_report,
    ordinal_word,
    parse_evaluator_report,
    pointer_move_event,
    render_memo,
    render_pointer_line,
    score_row_from_harvest,
    target_arithmetic,
)

FS1 = {
    "score_recomputed_from_components": 0.14786319521362173,
    "final_score": 0.15,
    "avg_segnet_dist": 0.00020139,
    "avg_posenet_dist": 6.17e-06,
    "archive_size_bytes": 180022,
    "expected_archive_sha256": "50fcaf1ac3c8504abdf3e0daff7c5bce32104f19d8de4a7ba207816f32e708cf",
    "expected_runtime_tree_sha256": "fbf4aaf436aa02814d0558bfbc2bf4307502bdac49a7616b66bcfa31b44ca43c",
    "n_samples": 600,
    "score_axis": "contest_cuda",
    "evidence_grade": "contest-CUDA",
    "passed": True,
    "validation_errors": [],
    "modal_elapsed_seconds": 545.465250242,
    "gpu_model": "Tesla T4",
}

FS2 = {
    "score_recomputed_from_components": 0.14784474152757654,
    "final_score": 0.15,
    "avg_segnet_dist": 0.00020139,
    "avg_posenet_dist": 6.14e-06,
    "archive_size_bytes": 180023,
    "expected_archive_sha256": "a8f3a3791499b2b62ee4d16bc67f15f819f454dc9b88e3cce04fe50a30427bb6",
    "expected_runtime_tree_sha256": "915d25f93ad69a07443caaba3c57d3484afc10620dd8dd98b089556c06d71d34",
    "n_samples": 600,
    "score_axis": "contest_cuda",
    "evidence_grade": "contest-CUDA",
    "passed": True,
    "validation_errors": [],
    "modal_elapsed_seconds": 656.929998781,
    "gpu_model": "Tesla T4",
}

FS1_REPORT = (
    "=== Evaluation results over 600 samples ===\n"
    "  Average PoseNet Distortion: 0.00000617\n"
    "  Average SegNet Distortion: 0.00020139\n"
    "  Submission file size: 180,022 bytes\n"
    "  Original uncompressed size: 37,545,489 bytes\n"
    "  Compression Rate: 0.00479477\n"
    "  Final score: 100*segnet_dist + √(10*posenet_dist) + 25*rate = 0.15\n"
)


def test_score_is_recomputed_not_read_from_final_score() -> None:
    row = score_row_from_harvest(FS1)
    assert row.score == 0.14786319521362173
    assert row.score != FS1["final_score"]


def test_both_rows_reproduce_their_receipts_exactly() -> None:
    for payload in (FS1, FS2):
        row = score_row_from_harvest(payload)
        assert row.score == payload["score_recomputed_from_components"]


def test_component_terms_match_the_committed_memos() -> None:
    fs1 = score_row_from_harvest(FS1)
    assert fs1.rate_term == 0.11986926045895953
    assert fs1.seg_term == 0.020139
    assert fs1.pose_term == math.sqrt(10 * 6.17e-06)
    fs2 = score_row_from_harvest(FS2)
    assert fs2.rate_term == 0.11986992631791266
    assert fs2.pose_term == math.sqrt(10 * 6.14e-06)


def test_target_arithmetic_reproduces_the_fs1_memo_corners() -> None:
    arith = target_arithmetic(score_row_from_harvest(FS1))
    assert arith.gap == 0.02786319521362174
    assert round(arith.rate_corner_max_bytes, 1) == 138176.5
    assert round(arith.rate_corner_demand_bytes, 1) == 41845.5
    assert f"{arith.distortion_corner_max:.5g}" == "0.00013074"
    assert round(arith.distortion_corner_reduction_x, 1) == 214.1
    assert round(arith.zero_distortion_max_bytes, 3) == 180218.347
    assert round(arith.zero_distortion_margin_bytes, 3) == 196.347


def test_target_arithmetic_reproduces_the_fs2_memo_corners() -> None:
    arith = target_arithmetic(score_row_from_harvest(FS2))
    assert round(arith.rate_corner_max_bytes, 1) == 138205.2
    assert round(arith.rate_corner_demand_bytes, 1) == 41817.8
    assert round(arith.distortion_corner_reduction_x, 1) == 215.1
    assert round(arith.zero_distortion_margin_bytes, 3) == 195.347


def test_delta_against_the_prior_pointer_matches_the_memo() -> None:
    fs1 = score_row_from_harvest(FS1)
    afr1 = PriorAnchor(label="afr1", score=0.14797617125559104, archive_bytes=180002,
                       d_seg=0.00020139, d_pose=6.37e-06)
    plan = PacketPlan(row=fs1, prior=afr1, arithmetic=target_arithmetic(fs1),
                      move_number=24, beats_prior=True)
    assert plan.delta_score == -0.00011297604196930378
    fs2 = score_row_from_harvest(FS2)
    prior_fs1 = PriorAnchor(label="fs1", score=fs1.score, archive_bytes=180022,
                            d_seg=0.00020139, d_pose=6.17e-06)
    plan2 = PacketPlan(row=fs2, prior=prior_fs1, arithmetic=target_arithmetic(fs2),
                       move_number=25, beats_prior=True)
    assert plan2.delta_score == -1.8453686045194484e-05


def test_report_parse_handles_thousands_separators() -> None:
    parsed = parse_evaluator_report(FS1_REPORT)
    assert parsed["archive_bytes"] == 180022.0
    assert parsed["uncompressed_bytes"] == 37545489.0
    assert parsed["n_samples"] == 600.0


def test_report_cross_check_is_clean_on_the_real_row() -> None:
    assert cross_check_against_report(score_row_from_harvest(FS1), FS1_REPORT) == []


def test_report_cross_check_catches_a_byte_disagreement() -> None:
    tampered = FS1_REPORT.replace("180,022 bytes", "180,023 bytes")
    problems = cross_check_against_report(score_row_from_harvest(FS1), tampered)
    assert problems and "report bytes 180023" in problems[0]


def test_refuses_when_the_receipt_disagrees_with_the_recompute() -> None:
    bad = dict(FS1, score_recomputed_from_components=0.1478)
    with pytest.raises(HarvestRefusal, match="disagrees"):
        score_row_from_harvest(bad)


def test_refuses_a_truncated_archive_sha() -> None:
    bad = dict(FS1, expected_archive_sha256="50fcaf1a", archive_sha256=None)
    with pytest.raises(HarvestRefusal, match="64-hex"):
        score_row_from_harvest(bad)


def test_refuses_without_a_byte_count() -> None:
    bad = {k: v for k, v in FS1.items() if k != "archive_size_bytes"}
    with pytest.raises(HarvestRefusal):
        score_row_from_harvest(bad)


def test_ordinal_words_cover_the_live_range() -> None:
    assert ordinal_word(24) == "TWENTY-FOURTH"
    assert ordinal_word(25) == "TWENTY-FIFTH"
    assert ordinal_word(20) == "TWENTIETH"
    assert ordinal_word(30) == "THIRTIETH"
    assert ordinal_word(3) == "THIRD"
    with pytest.raises(ValueError):
        ordinal_word(0)


def _plan(payload: dict, prior: PriorAnchor, n: int) -> PacketPlan:
    row = score_row_from_harvest(payload, lane_id="ddm_fs1", call_id="fc-TEST")
    return PacketPlan(row=row, prior=prior, arithmetic=target_arithmetic(row),
                      move_number=n, beats_prior=True)


def test_memo_carries_the_computed_numbers_and_the_full_shas() -> None:
    prior = PriorAnchor(label="afr1", score=0.14797617125559104, archive_bytes=180002,
                        d_seg=0.00020139, d_pose=6.37e-06)
    memo = render_memo(
        _plan(FS1, prior, 24),
        date_utc="2026-09-04",
        axis_label="contest-CUDA T4 n600",
        headline="a test row",
        mechanism="mechanism prose",
        custody_lines=["harvest somewhere"],
        not_claimed="claims nothing",
        equations_leg="none",
    )
    assert "TWENTY-FOURTH POINTER MOVE" in memo
    assert "0.14786319521362173" in memo
    assert FS1["expected_archive_sha256"] in memo
    assert "0.02786319521362174" in memo
    assert "-41,845.5 B" in memo
    assert "214.1× reduction" in memo
    assert "196.347 B under" in memo
    assert "mechanism prose" in memo


def test_pointer_line_and_event_carry_the_full_identity() -> None:
    prior = PriorAnchor(label="afr1", score=0.14797617125559104, archive_bytes=180002,
                        d_seg=0.00020139, d_pose=6.37e-06)
    plan = _plan(FS1, prior, 24)
    line = render_pointer_line(plan, axis_label="contest-CUDA T4 n600")
    assert FS1["expected_archive_sha256"] in line
    assert FS1["expected_runtime_tree_sha256"] in line
    event = pointer_move_event(plan, axis_label="contest-CUDA T4 n600",
                               memo_path="/x.md", at_utc="2026-09-04T00:00:00Z")
    assert event["archive_sha256"] == FS1["expected_archive_sha256"]
    assert event["delta_score"] == -0.00011297604196930378


def _poller_module():
    """Load tools/modal_harvest_poller.py by path.

    ``tools/tests`` is not in ``testpaths``, so a test that lives there never runs in a
    default ``pytest`` invocation. The canonical terminal claim row is the single most
    error-prone artifact of a pointer move, so its test belongs where it is collected.
    """

    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[3] / "tools" / "modal_harvest_poller.py"
    spec = importlib.util.spec_from_file_location("_hv1_poller", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_terminal_claim_note_binds_both_full_shas() -> None:
    """The compliance checker demands 64 hex; a truncated sha still reds.

    MEASURED 2026-09-04: MAIN's hand-written fs2 terminal row carried
    ``runtime_tree_sha256=915d25f93ad6`` — 12 hex — and
    ``dispatch_claim_terminal_runtime_tree_sha_bound`` stayed RED.
    """

    import re

    binding = re.compile(
        r"\b(?:archive_sha256|runtime_tree_sha256)\b\s*[:=]\s*[0-9a-fA-F]{64}\b"
    )
    for payload in (FS1, FS2):
        note = _poller_module().canonical_terminal_claim_notes(payload, "fc-TEST")
        bound = {m.group(0).split("=")[0] for m in binding.finditer(note)}
        assert bound == {"archive_sha256", "runtime_tree_sha256"}, note
        assert payload["expected_archive_sha256"] in note
        assert payload["expected_runtime_tree_sha256"] in note
        assert f"score={payload['score_recomputed_from_components']}" in note
        assert "final_score" not in note


def test_terminal_claim_note_omits_a_sha_the_receipt_lacks() -> None:
    """An absent sha stays absent: a plausible-looking one would forge the binding."""

    payload = {k: v for k, v in FS1.items() if k != "expected_runtime_tree_sha256"}
    note = _poller_module().canonical_terminal_claim_notes(payload, "fc-TEST")
    assert "runtime_tree_sha256=" not in note
    assert FS1["expected_archive_sha256"] in note
