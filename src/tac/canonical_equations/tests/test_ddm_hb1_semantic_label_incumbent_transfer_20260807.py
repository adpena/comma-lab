# SPDX-License-Identifier: MIT
from __future__ import annotations

import pytest

from tac.canonical_equations.ddm_hb1_semantic_label_incumbent_transfer_20260807 import (
    EQUATION_ID,
    build_ddm_hb1_semantic_label_incumbent_transfer_v1,
    semantic_label_s_rate,
    semantic_label_transfer_status,
)


def test_external_pr130_hpac_is_not_admissible_without_target_payload_proof() -> None:
    status = semantic_label_transfer_status(
        incumbent_bytes=142_001,
        external_stream_bytes=116_980,
        external_model_bytes=15_164,
        trained_on_target_payload=False,
        decode_equality_on_target_payload=False,
    )
    assert status["external_total_bytes"] == 132_144
    assert status["delta_bytes_if_transfer_were_valid"] == -9_857
    assert status["transfer_admissible"] is False
    assert status["incumbent_stands"] is True


def test_target_payload_proof_allows_real_byte_comparison() -> None:
    status = semantic_label_transfer_status(
        incumbent_bytes=142_001,
        external_stream_bytes=116_980,
        external_model_bytes=15_164,
        trained_on_target_payload=True,
        decode_equality_on_target_payload=True,
    )
    assert status["transfer_admissible"] is True
    assert status["incumbent_stands"] is False


def test_rate_terms_match_hb1_receipt() -> None:
    assert semantic_label_s_rate(142_001) == pytest.approx(0.094552637)
    assert semantic_label_s_rate(173_617) == pytest.approx(0.115604434)


def test_equation_builds_byte_only_non_promoting_anchor() -> None:
    eq = build_ddm_hb1_semantic_label_incumbent_transfer_v1()
    assert eq.equation_id == EQUATION_ID
    assert eq.empirical_anchors[0].empirical_output["hpac_on_our_payload_measured"] is False
    assert eq.domain_of_validity["score_claim"] is False
