# SPDX-License-Identifier: MIT
"""Tests for the bit_allocator_per_pair_consumer cathedral package."""
from __future__ import annotations

from typing import Any

import numpy as np

from tac.cathedral.consumer_contract import HookNumber, validate_consumer_module
from tac.cathedral_consumers import bit_allocator_per_pair_consumer as consumer


def test_contract_constants() -> None:
    assert consumer.CONSUMER_NAME == "bit_allocator_per_pair_consumer"
    assert consumer.CONSUMER_VERSION == "1.0"
    assert consumer.CONSUMER_HOOK_NUMBERS == (HookNumber.BIT_ALLOCATOR,)
    assert consumer.CONSUMER_HOOK_NUMBERS == (3,)


def test_discoverability_compatible_callable_surface() -> None:
    reg = validate_consumer_module(
        consumer,
        module_path="tac.cathedral_consumers.bit_allocator_per_pair_consumer",
    )
    assert reg.contract_compliant is True, reg.validation_errors
    assert callable(consumer.update_from_anchor)
    assert callable(consumer.consume_candidate)


def test_missing_anchor_fail_closed_no_score_claim(monkeypatch) -> None:
    from tac import master_gradient_consumers

    def missing_anchor(**_: Any) -> tuple[np.ndarray, dict[str, Any]]:
        raise FileNotFoundError("no synthetic anchor")

    monkeypatch.setattr(
        master_gradient_consumers,
        "load_per_pair_gradient_from_anchor",
        missing_anchor,
    )

    verdict = consumer.consume_candidate({"archive_sha256": "a" * 64})

    assert verdict["predicted_delta_adjustment"] == 0.0
    assert verdict["axis_tag"] == "[predicted]"
    assert verdict["promotable"] is False
    assert verdict["score_claim"] is False
    assert verdict["promotion_eligible"] is False
    assert "no per-pair master-gradient anchor" in verdict["rationale"]


def test_valid_anchor_emits_hook_3_observability_only_verdict(monkeypatch) -> None:
    from tac import master_gradient_consumers

    gradient = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 2.0, 0.0], [0.0, 0.0, 0.0]],
            [[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        ],
        dtype=np.float64,
    )
    anchor = {
        "archive_sha256": "b" * 64,
        "measurement_axis": "[contest-CUDA]",
        "measurement_hardware": "nvidia_t4",
        "measurement_method": "per_pair_master_gradient_test",
        "measurement_utc": "2026-05-19T07:20:00Z",
    }

    monkeypatch.setattr(
        master_gradient_consumers,
        "load_per_pair_gradient_from_anchor",
        lambda **_: (gradient, anchor),
    )

    verdict = consumer.consume_candidate({"archive_sha256": "b" * 64})

    assert HookNumber.BIT_ALLOCATOR in consumer.CONSUMER_HOOK_NUMBERS
    assert verdict["predicted_delta_adjustment"] == 0.0
    assert verdict["axis_tag"] == "[predicted]"
    assert verdict["promotable"] is False
    assert verdict["score_claim"] is False
    assert verdict["authority"] == "predicted_observability_only"
    notes = verdict["notes"]["bit_allocator_per_pair_consumer"]
    assert notes["n_bytes"] == 4
    assert notes["n_pairs"] == 2
    assert notes["measurement_axis"] == "[contest-CUDA]"
    assert notes["measurement_hardware"] == "nvidia_t4"
    assert notes["requested_helper_missing"] is True
    assert notes["source_helper"] == "load_per_pair_gradient_from_anchor"


def test_top_k_notes_are_deterministic(monkeypatch) -> None:
    from tac import master_gradient_consumers

    # Overall byte sensitivities: byte2=4, byte1=2, byte3=2, byte0=1.
    # The byte1/byte3 tie must resolve by lower byte index.
    gradient = np.array(
        [
            [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            [[0.0, 2.0, 0.0], [0.0, 0.0, 0.0]],
            [[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        ],
        dtype=np.float64,
    )
    anchor = {
        "archive_sha256": "c" * 64,
        "measurement_axis": "[contest-CPU]",
        "measurement_hardware": "linux_x86_64",
        "measurement_method": "per_pair_master_gradient_test",
        "measurement_utc": "2026-05-19T07:20:00Z",
    }

    monkeypatch.setattr(
        master_gradient_consumers,
        "load_per_pair_gradient_from_anchor",
        lambda **_: (gradient, anchor),
    )

    verdict1 = consumer.consume_candidate({"archive_sha256": "c" * 64})
    verdict2 = consumer.consume_candidate({"archive_sha256": "c" * 64})
    notes1 = verdict1["notes"]["bit_allocator_per_pair_consumer"]
    notes2 = verdict2["notes"]["bit_allocator_per_pair_consumer"]

    assert notes1 == notes2
    top_bytes = notes1["top_k_byte_indices_by_abs_sensitivity"]
    assert [row["byte_index"] for row in top_bytes] == [2, 1, 3, 0]
    assert [row["absolute_sensitivity"] for row in top_bytes] == [4.0, 2.0, 2.0, 1.0]

    per_pair = notes1["per_pair_top_k_byte_indices"]
    assert per_pair[0]["pair_index"] == 0
    assert [row["byte_index"] for row in per_pair[0]["top_byte_indices"]] == [
        2,
        1,
        0,
        3,
    ]
    assert per_pair[1]["pair_index"] == 1
    assert [row["byte_index"] for row in per_pair[1]["top_byte_indices"]] == [
        2,
        3,
        0,
        1,
    ]


def test_invalid_gradient_shape_fails_closed_no_score_claim(monkeypatch) -> None:
    from tac import master_gradient_consumers

    monkeypatch.setattr(
        master_gradient_consumers,
        "load_per_pair_gradient_from_anchor",
        lambda **_: (np.zeros((4, 2), dtype=np.float64), {"archive_sha256": "d" * 64}),
    )

    verdict = consumer.consume_candidate({"archive_sha256": "d" * 64})

    assert verdict["predicted_delta_adjustment"] == 0.0
    assert verdict["axis_tag"] == "[predicted]"
    assert verdict["promotable"] is False
    assert verdict["score_claim"] is False
    assert "invalid per-pair master-gradient payload" in verdict["rationale"]
