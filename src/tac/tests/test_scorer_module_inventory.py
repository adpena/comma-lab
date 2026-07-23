"""Unit tests for scorer-module inventory custody."""

from __future__ import annotations

import json

import pytest

from tac.optimization.scorer_module_inventory import (
    SCHEMA,
    ScorerInventoryError,
    canonical_json_bytes,
    validate_receipt,
    wrap_receipt,
    write_receipt_once,
)


def _body() -> dict:
    return {
        "schema": SCHEMA,
        "source_strata": {
            "A_evaluator_composition": {
                "semantics": {"pairing": {"pair_count_required_by_this_atlas": 600}}
            },
            "C_loaded_artifacts": {
                "posenet": {
                    "module_state_match": {"status": "EXACT_NAMES_AND_SHAPES"}
                },
                "segnet": {
                    "module_state_match": {"status": "EXACT_NAMES_AND_SHAPES"}
                },
            },
        },
    }


def test_wrapped_receipt_hash_is_canonical_and_valid() -> None:
    first = wrap_receipt(_body())
    second = wrap_receipt(json.loads(canonical_json_bytes(_body())))
    assert first == second
    validate_receipt(first)


def test_receipt_refuses_hash_mismatch_and_non_n600() -> None:
    receipt = wrap_receipt(_body())
    receipt["body"]["schema"] = "changed"
    with pytest.raises(ScorerInventoryError, match="schema"):
        validate_receipt(receipt)
    body = _body()
    body["source_strata"]["A_evaluator_composition"]["semantics"]["pairing"][
        "pair_count_required_by_this_atlas"
    ] = 599
    with pytest.raises(ScorerInventoryError, match="n600"):
        validate_receipt(wrap_receipt(body))


def test_write_once_refuses_different_receipt(tmp_path) -> None:
    path = tmp_path / "receipt.json"
    receipt = wrap_receipt(_body())
    write_receipt_once(path, receipt)
    write_receipt_once(path, receipt)
    changed = _body()
    changed["extra"] = True
    with pytest.raises(FileExistsError, match="overwrite"):
        write_receipt_once(path, wrap_receipt(changed))
