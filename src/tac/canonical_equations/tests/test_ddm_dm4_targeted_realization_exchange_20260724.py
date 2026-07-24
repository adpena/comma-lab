# SPDX-License-Identifier: MIT
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tac.canonical_equations.ddm_dm2_semantic_realization_exchange import (
    build_ddm_dm2_semantic_to_realized_rgb_exchange_v1,
)
from tac.canonical_equations.ddm_dm4_targeted_realization_exchange_20260724 import (
    ANCHOR_ID,
    append_dm4_exchange_anchor,
    build_dm4_exchange_anchor,
)
from tac.canonical_equations.registry import register_canonical_equation


def test_dm4_anchor_is_exact_nonpromotable_improvement() -> None:
    anchor = build_dm4_exchange_anchor()
    assert anchor.anchor_id == ANCHOR_ID
    assert anchor.residual == pytest.approx(0.0)
    assert anchor.empirical_output["score_claim"] is False
    assert anchor.empirical_output["effective_bytes_per_semantic_byte"] == pytest.approx(2065.85149776928)
    assert anchor.empirical_output["ratio_fraction_of_dm2"] == pytest.approx(0.8184019437709267)
    assert anchor.empirical_output["fallback_pair_ids"] == [55]


def test_dm4_append_preserves_dm2_anchor(tmp_path: Path) -> None:
    registry = tmp_path / "registry.jsonl"
    lock = tmp_path / "registry.jsonl.lock"
    base = build_ddm_dm2_semantic_to_realized_rgb_exchange_v1()
    register_canonical_equation(base, path=registry, lock_path=lock)
    updated = append_dm4_exchange_anchor(path=registry, lock_path=lock)
    anchor_ids = [anchor.anchor_id for anchor in updated.empirical_anchors]
    assert anchor_ids == [
        "ddm_dm2_25_exact_semantic_rows_l3_realization_20260724",
        ANCHOR_ID,
    ]
    events = [json.loads(line) for line in registry.read_text().splitlines()]
    assert [event["event_type"] for event in events] == [
        "registered",
        "anchor_appended",
    ]
