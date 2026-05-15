# SPDX-License-Identifier: MIT

from __future__ import annotations

from tac.packet_compiler.pr101_fec7_selector import (
    build_fec7_candidates,
    decode_fec7_selector_payload,
    empirical_entropy_floor_bytes,
    pairmod_entropy_floor_bytes,
    profile_selector_encodings,
)


def _fec6_like_codes() -> list[int]:
    counts = {
        0: 134,
        1: 35,
        2: 129,
        3: 9,
        4: 25,
        5: 13,
        6: 11,
        7: 71,
        8: 10,
        9: 24,
        10: 7,
        11: 16,
        12: 6,
        13: 92,
        14: 17,
        15: 1,
    }
    codes: list[int] = []
    # Interleave the histogram so pairmod contexts see non-trivial ordering.
    while len(codes) < 600:
        for code in sorted(counts):
            if counts[code] > 0:
                codes.append(code)
                counts[code] -= 1
    return codes


def test_fec7_candidates_are_byte_closed_roundtrips() -> None:
    codes = _fec6_like_codes()
    candidates = build_fec7_candidates(codes, pairmod_contexts=(2, 8, 25))

    assert {candidate.name for candidate in candidates} == {
        "fec7_global_pr103_range_u8_hist",
        "fec7_split_none_pr103_range",
        "fec7_pairmod2_pr84_context_range",
        "fec7_pairmod8_pr84_context_range",
        "fec7_pairmod25_pr84_context_range",
    }
    for candidate in candidates:
        assert decode_fec7_selector_payload(candidate.payload) == codes
        assert candidate.payload_bytes == len(candidate.payload)
        assert candidate.charged_model_bytes >= 0
        assert candidate.metadata_bytes > 0


def test_entropy_profile_preserves_no_score_and_blocks_79_byte_target() -> None:
    codes = _fec6_like_codes()
    profile = profile_selector_encodings(
        codes,
        fec6_selector_payload_bytes=249,
        target_saving_bytes=79,
        pairmod_contexts=(2, 8, 25),
    )

    assert profile["score_claim"] is False
    assert profile["dispatch_attempted"] is False
    assert profile["ready_for_exact_eval_dispatch"] is False
    assert profile["global_entropy_floor_bytes"] == empirical_entropy_floor_bytes(codes)
    assert profile["global_entropy_floor_saving_vs_fec6_bytes"] < 79
    assert profile["can_meet_target_with_charged_fec7_prototype"] is False
    assert profile["explicit_blocker"]["blocked"] is True
    assert profile["best_charged_candidate"]["saving_vs_fec6_selector_bytes"] < 79


def test_pairmod_context_floor_is_reported_separately_from_charged_bytes() -> None:
    codes = _fec6_like_codes()
    floor = pairmod_entropy_floor_bytes(codes, context_mod=25)
    profile = profile_selector_encodings(
        codes,
        fec6_selector_payload_bytes=249,
        target_saving_bytes=79,
        pairmod_contexts=(25,),
    )

    assert profile["theoretical_context_lower_bounds"][0]["context_mod"] == 25
    assert profile["theoretical_context_lower_bounds"][0]["zero_model_entropy_bytes"] == floor
    assert profile["theoretical_context_lower_bounds"][0]["byte_closed"] is False
    pairmod_candidate = next(
        row
        for row in profile["charged_candidates"]
        if row["name"] == "fec7_pairmod25_pr84_context_range"
    )
    assert pairmod_candidate["charged_model_bytes"] == 25 * 16
    assert pairmod_candidate["payload_bytes"] > floor
