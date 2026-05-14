# SPDX-License-Identifier: MIT
"""Tests for canonical adapter composition primitives."""

from __future__ import annotations

import pytest
import torch

from tac.composition.adapters import (
    AdapterCompositionError,
    AdapterRecord,
    DeterministicHypernetworkComposer,
    FrozenLinearLayer,
    adapter_delta_matrix,
    adapter_record_from_pr95,
    compose_tropical_adapter_delta,
    fold_adapter_chain,
    fold_adapter_record_into_weight,
    fold_tropical_adapter_metadata,
    product_of_experts_gating,
    stable_softmax,
    tropical_adapter_weights,
)


def _lora_record() -> AdapterRecord:
    return AdapterRecord(
        name="blocks.0",
        kind="lora",
        rank=1,
        alpha=2.0,
        A=torch.tensor([[1.0, 2.0, 3.0]]),
        B=torch.tensor([[4.0], [5.0]]),
    )


def test_adapter_record_from_pr95_validates_and_preserves_metadata() -> None:
    rec = adapter_record_from_pr95(
        {
            "name": "blocks.0",
            "kind": "lora",
            "rank": 1,
            "alpha": 1.0,
            "A": torch.ones(1, 3),
            "B": torch.ones(2, 1),
            "metadata": {"source": "unit"},
        }
    )
    assert rec.target_weight_name == "blocks.0.weight"
    assert rec.to_metadata()["score_claim"] is False
    assert dict(rec.metadata) == {"source": "unit"}
    with pytest.raises(AdapterCompositionError):
        adapter_record_from_pr95({"name": "x", "kind": "lora"})


def test_adapter_record_shape_validation_fail_closed() -> None:
    with pytest.raises(AdapterCompositionError):
        AdapterRecord(
            name="bad",
            kind="lora",
            rank=2,
            alpha=1.0,
            A=torch.ones(1, 3),
            B=torch.ones(2, 2),
        )
    with pytest.raises(AdapterCompositionError):
        AdapterRecord(
            name="bad",
            kind="dora",
            rank=1,
            alpha=1.0,
            A=torch.ones(1, 3),
            B=torch.ones(2, 1),
        )


def test_adapter_delta_matrix_and_lora_fold_match_pr95_contract() -> None:
    rec = _lora_record()
    base = torch.zeros(2, 3)
    delta = adapter_delta_matrix(rec)
    expected = torch.tensor([[8.0, 16.0, 24.0], [10.0, 20.0, 30.0]])
    assert torch.allclose(delta, expected)
    folded = fold_adapter_record_into_weight(base, rec)
    assert torch.allclose(folded, expected)


def test_dora_fold_uses_magnitude_and_row_norm() -> None:
    rec = AdapterRecord(
        name="blocks.0",
        kind="dora",
        rank=1,
        alpha=1.0,
        A=torch.zeros(1, 2),
        B=torch.zeros(2, 1),
        magnitude=torch.tensor([5.0, 10.0]),
    )
    base = torch.tensor([[3.0, 4.0], [6.0, 8.0]])
    folded = fold_adapter_record_into_weight(base, rec)
    assert torch.allclose(folded, base)


def test_fold_adapter_chain_copies_state_and_fails_on_missing_target() -> None:
    rec = _lora_record()
    state = {"blocks.0.weight": torch.zeros(2, 3)}
    out = fold_adapter_chain(state, (rec,))
    assert "blocks.0.weight" in out
    assert torch.allclose(state["blocks.0.weight"], torch.zeros(2, 3))
    assert not torch.allclose(out["blocks.0.weight"], state["blocks.0.weight"])
    with pytest.raises(AdapterCompositionError):
        fold_adapter_chain({}, (rec,))
    assert fold_adapter_chain({}, (rec,), on_missing="ignore") == {}


def test_tropical_hard_max_is_deterministic_on_ties() -> None:
    logits = torch.tensor([1.0, 2.0, 2.0])
    result = tropical_adapter_weights(logits, ("a", "b", "c"), mode="hard_max")
    assert result.selected_branch_id == "b"
    assert torch.allclose(result.weights, torch.tensor([0.0, 1.0, 0.0]))


def test_softmax_and_tropical_delta_composition_are_stable() -> None:
    logits = torch.tensor([1000.0, 1001.0])
    weights = stable_softmax(logits)
    assert torch.isfinite(weights).all()
    assert torch.allclose(weights.sum(), torch.tensor(1.0))
    delta, selection = compose_tropical_adapter_delta(
        (torch.zeros(2, 2), torch.ones(2, 2)),
        logits,
        ("zero", "one"),
        mode="softmax",
    )
    assert selection.selected_branch_id == "one"
    assert torch.all(delta > 0.5)
    meta = fold_tropical_adapter_metadata(
        target_name="blocks.0.weight",
        selection=selection,
        output_delta=delta,
    )
    assert meta["score_claim"] is False
    with pytest.raises(AdapterCompositionError):
        compose_tropical_adapter_delta(
            (torch.zeros(2, 2), torch.ones(3, 2)),
            logits,
            ("a", "b"),
        )


def test_hypernetwork_composer_is_deterministic_and_normalized() -> None:
    composer = DeterministicHypernetworkComposer(
        adapter_ids=("a", "b"),
        layers=(
            FrozenLinearLayer(
                weight=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                bias=torch.zeros(2),
            ),
            FrozenLinearLayer(
                weight=torch.tensor([[1.0, -1.0], [-1.0, 1.0]]),
                bias=torch.zeros(2),
            ),
        ),
        activation="identity",
    )
    out_a = composer(torch.tensor([2.0, 1.0]))
    out_b = composer(torch.tensor([2.0, 1.0]))
    assert out_a.as_mapping()["a"] > out_a.as_mapping()["b"]
    assert torch.allclose(out_a.weights.sum(), torch.tensor(1.0))
    assert torch.allclose(out_a.weights, out_b.weights)
    assert composer.to_json() == composer.to_json()
    assert composer.to_metadata()["ready_for_exact_eval_dispatch"] is False


def test_hypernetwork_rejects_hidden_shape_state() -> None:
    with pytest.raises(AdapterCompositionError):
        DeterministicHypernetworkComposer(
            adapter_ids=("a",),
            layers=(
                FrozenLinearLayer(weight=torch.ones(2, 2), bias=torch.zeros(2)),
            ),
        )


def test_product_of_experts_gating_normalizes_and_handles_offsets() -> None:
    experts = torch.tensor([[1000.0, 1001.0], [-100.0, -99.0]])
    result = product_of_experts_gating(experts)
    assert torch.isfinite(result.log_weights).all()
    assert torch.allclose(result.weights.sum(), torch.tensor(1.0))
    assert result.weights[1] > result.weights[0]
    with pytest.raises(AdapterCompositionError):
        product_of_experts_gating(
            torch.tensor([[0.0, float("-inf")], [float("-inf"), 0.0]])
        )
