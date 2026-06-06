# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np

from tac.analysis.source_forward_bit_flip_falsification import (
    BIT_FLIP_FALSIFICATION_SCHEMA,
    build_array_bit_flip_falsification,
    build_named_arrays_bit_flip_falsification,
)


def test_array_bit_flip_falsification_passes_for_bit_identical_outputs() -> None:
    official = np.array([[0.25, 0.5], [0.75, 1.0]], dtype=np.float64)
    proof = build_array_bit_flip_falsification(
        component_id="mfu",
        official_output=official,
        portable_output=official.copy(),
        tolerance=0.0,
        false_authority={"score_claim": False},
    )

    assert proof["schema"] == BIT_FLIP_FALSIFICATION_SCHEMA
    assert proof["component_id"] == "mfu"
    assert proof["passed"] is True
    assert proof["falsifies_when_perturbed"] is True
    assert proof["baseline_official_output_sha256"] == proof[
        "baseline_portable_output_sha256"
    ]
    assert proof["perturbed_portable_output_sha256"] != proof[
        "baseline_official_output_sha256"
    ]
    assert proof["negative_control_max_abs_error"] > proof["tolerance"]
    assert proof["negative_control_output_hashes_bit_identical"] is False
    assert proof["score_claim"] is False


def test_named_bit_flip_falsification_fails_when_baseline_outputs_differ() -> None:
    official = {"a": np.array([1.0, 2.0], dtype=np.float64)}
    portable = {"a": np.array([1.0, 2.125], dtype=np.float64)}

    proof = build_named_arrays_bit_flip_falsification(
        component_id="tub",
        official_outputs=official,
        portable_outputs=portable,
        tolerance=0.0,
    )

    assert proof["passed"] is False
    assert proof["falsifies_when_perturbed"] is False
    assert proof["baseline_official_output_sha256"] != proof[
        "baseline_portable_output_sha256"
    ]


def test_bit_flip_falsification_fails_closed_on_invalid_tolerance() -> None:
    official = np.array([1.0, 2.0], dtype=np.float64)
    proof = build_array_bit_flip_falsification(
        component_id="hfr",
        official_output=official,
        portable_output=official.copy(),
        tolerance=float("nan"),
    )

    assert proof["passed"] is False
    assert proof["falsifies_when_perturbed"] is False
    assert proof["tolerance"] is None
    assert proof["bit_flip_byte_offset"] is None
