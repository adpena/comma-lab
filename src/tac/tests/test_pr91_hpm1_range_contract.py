# SPDX-License-Identifier: MIT
from __future__ import annotations

import numpy as np
import pytest

from tac import pr86_hpac_codec as hpac
from tac.pr86_hpac_codec import (
    _categorical_from_probs,
    resolve_hpac_probability_variant,
)
from tac.pr91_hpm1_range_contract import (
    audit_pr91_hpm1_range_contract_from_rows,
)


def _encode_symbols(
    rows: list[np.ndarray],
    symbols: list[int],
    *,
    probability_variant: str = "source_float64_perfect_false",
    prob_eps: float = 1e-7,
) -> np.ndarray:
    if hpac.constriction is None:
        pytest.skip("constriction range coder not available")
    resolved = resolve_hpac_probability_variant(probability_variant)
    encoder = hpac.constriction.stream.queue.RangeEncoder()
    for row, symbol in zip(rows, symbols, strict=True):
        encoder.encode(
            int(symbol),
            _categorical_from_probs(row, prob_eps=prob_eps, variant=resolved),
        )
    return np.ascontiguousarray(
        np.asarray(encoder.get_compressed(), dtype=np.uint32),
        dtype=np.uint32,
    )


def test_range_contract_manifest_refuses_dispatch_and_records_first_mismatch() -> None:
    rows = [
        np.asarray(row, dtype=np.float32)
        for row in (
            [0.45, 0.20, 0.15, 0.12, 0.08],
            [0.10, 0.52, 0.16, 0.12, 0.10],
            [0.08, 0.12, 0.55, 0.15, 0.10],
            [0.10, 0.10, 0.14, 0.56, 0.10],
            [0.12, 0.08, 0.10, 0.18, 0.52],
            [0.50, 0.16, 0.12, 0.12, 0.10],
            [0.10, 0.50, 0.18, 0.12, 0.10],
            [0.42, 0.14, 0.12, 0.20, 0.12],
        )
    ]
    reference_symbols = [0, 1, 2, 3, 4, 0, 1, 0]
    submitted_symbols = [0, 1, 2, 3, 4, 0, 1, 2]
    submitted_words = _encode_symbols(rows, submitted_symbols)

    report = audit_pr91_hpm1_range_contract_from_rows(
        rows,
        reference_symbols,
        {"source_little_uint32": submitted_words},
        probability_variants=("source_float64_perfect_false",),
        prob_eps=1e-7,
    )

    assert report["score_claim"] is False
    assert report["dispatch_allowed"] is False
    assert report["dispatch_attempted"] is False
    assert report["dispatch_performed"] is False
    assert report["ready_for_exact_eval_dispatch"] is False
    assert report["classification"]["likely_blocker_class"] == (
        "context_row_probability_or_reference_symbol_grammar"
    )
    mismatch = report["first_mismatch_evidence"][
        "source_little_submitted_reference_decode_mismatch"
    ]
    assert mismatch == {
        "word_order_candidate": "source_little_uint32",
        "probability_variant": "source_float64_perfect_false",
        "symbol_index": 7,
        "decoded_symbol": 2,
        "reference_symbol": 0,
        "matched_prefix_symbol_count": 7,
    }
    local_mismatch = report["first_mismatch_evidence"][
        "local_reemit_word_mismatch"
    ]
    assert local_mismatch["word_index"] == 0
    assert local_mismatch["byte_offset"] == 0
