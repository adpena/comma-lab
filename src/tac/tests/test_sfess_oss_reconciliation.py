# SPDX-License-Identifier: MIT
"""Canaries for the SFESS paper-reconciliation learned-logit arm."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import tac.sfess_oss_reconciliation as reconciliation
from tac.sfess_cached_replay import (
    CountedCachedOracle,
    SFESSGradientSample,
    load_cached_objective_jsonl,
)
from tac.sfess_oss_reconciliation import run_learned_logit_sfess


def _table(path: Path, values: list[float], n_bits: int) -> object:
    rows = []
    for index, value in enumerate(values):
        rows.append({
            "candidate_mask": [(index >> bit) & 1 for bit in range(n_bits)],
            "candidate_value": value,
            "estimator": "exact_enumeration",
            "function_evals_after": index + 1,
            "proposal_index": index,
        })
    encoded = ("\n".join(json.dumps(row) for row in rows) + "\n").encode()
    path.write_bytes(encoded)
    return load_cached_objective_jsonl(
        path,
        expected_sha256=hashlib.sha256(encoded).hexdigest(),
        n_bits=n_bits,
    )


def test_learned_logits_are_deterministic_and_budget_exact(tmp_path: Path) -> None:
    table = _table(tmp_path / "objective.jsonl", [8.0, 3.0, 1.0, 7.0], 2)

    def run_once() -> object:
        oracle = CountedCachedOracle(table, budget=10, authorize_lookup=lambda _mask: True)
        return run_learned_logit_sfess(
            oracle,
            k=1,
            samples_per_gradient=2,
            seed=9,
            learning_rate=1.0e-4,
            comparison_noise_floor_s=0.0,
        )

    first = run_once()
    second = run_once()
    assert first == second
    assert first.calls == 10
    assert first.best_value == 1.0
    assert first.best_mask == (0, 1)
    assert first.final_logits[1] > first.final_logits[0]
    assert first.gradient_steps == first.strict_gate_calls
    assert first.accepted_optimizer_updates + first.rejected_optimizer_updates == first.gradient_steps


def test_rejected_exact_gate_retains_logits_and_adam_state(tmp_path: Path, monkeypatch) -> None:
    table = _table(
        tmp_path / "rejection.jsonl",
        [9.0, 5.0, 7.0, 8.0, 1.0, 6.0, 4.0, 3.0],
        3,
    )
    calls = 0

    def rejected_then_accepted_gradient(evaluate, *_args, **_kwargs) -> SFESSGradientSample:
        nonlocal calls
        masks = ((0, 1, 0), (0, 0, 1))
        values = tuple(evaluate(np.asarray(mask, dtype=np.uint8)) for mask in masks)
        gradient = (
            np.asarray([1.0, -1.0, 0.0])
            if calls == 0
            else np.asarray([1.0, 0.0, -1.0])
        )
        calls += 1
        return SFESSGradientSample(
            gradient=gradient,
            masks=masks,
            values=values,
            scores=(np.zeros(3), np.zeros(3)),
        )

    monkeypatch.setattr(
        reconciliation,
        "sfess_leave_one_out_gradient",
        rejected_then_accepted_gradient,
    )
    oracle = CountedCachedOracle(table, budget=7, authorize_lookup=lambda _mask: True)
    result = run_learned_logit_sfess(
        oracle,
        k=1,
        samples_per_gradient=2,
        seed=3,
        learning_rate=1.0,
        comparison_noise_floor_s=0.0,
    )
    initial_logit = np.log(0.5)
    assert result.best_mask == (0, 0, 1)
    assert result.best_value == 1.0
    assert result.final_logits == pytest.approx(
        (initial_logit - 1.0, initial_logit, initial_logit + 1.0)
    )
    assert result.gradient_steps == 2
    assert result.accepted_optimizer_updates == 1
    assert result.rejected_optimizer_updates == 1
    assert result.strict_gate_calls == 2


def test_zero_variance_group_skips_adam_and_strict_gate(tmp_path: Path) -> None:
    table = _table(tmp_path / "flat.jsonl", [2.0, 1.0, 1.0, 2.0], 2)
    oracle = CountedCachedOracle(table, budget=8, authorize_lookup=lambda _mask: True)
    result = run_learned_logit_sfess(
        oracle,
        k=1,
        samples_per_gradient=2,
        seed=4,
        learning_rate=1.0e-4,
        comparison_noise_floor_s=1.0e-12,
    )
    assert result.calls == 8
    assert result.zero_variance_skips >= 1
    assert result.gradient_steps == 0
    assert result.accepted_optimizer_updates == 0
    assert result.rejected_optimizer_updates == 0
    assert result.strict_gate_calls == 0
    assert np.allclose(result.final_logits, 0.0)
