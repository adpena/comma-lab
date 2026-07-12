# SPDX-License-Identifier: MIT
"""Canaries and exact finite-support checks for cached SFESS replay."""
from __future__ import annotations

import hashlib
import json
from itertools import product
from pathlib import Path

import numpy as np
import pytest

import tac.sfess_cached_replay as sfess_core
from tac.sfess_cached_replay import (
    PINNED_UGC64_SHA256,
    CountedCachedOracle,
    SFESSError,
    SFESSFixedKSearch,
    cached_state_sha256,
    exact_k_subset_logit_score,
    load_cached_objective_jsonl,
    poisson_binomial_pmf_dft,
    sample_conditional_bernoulli_k_subset,
    sfess_leave_one_out_gradient,
)

REPO = Path(__file__).resolve().parents[3]
CACHED_64 = (
    REPO
    / "experiments/results/ugc_terminal_polish_ab_20260712/"
    "search_exact_enumeration_accepted_proposals.jsonl"
)


def _write_rows(path: Path, rows: list[dict[str, object]]) -> str:
    data = ("\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n").encode()
    path.write_bytes(data)
    return hashlib.sha256(data).hexdigest()


def _rows(n_bits: int, values: list[float] | None = None) -> list[dict[str, object]]:
    if values is None:
        values = [float(index) for index in range(1 << n_bits)]
    return [
        {
            "accepted": False,
            "candidate_mask": [(index >> bit) & 1 for bit in range(n_bits)],
            "candidate_value": values[index],
            "delta_s": 0.0,
            "estimator": "exact_enumeration",
            "function_evals_after": index + 1,
            "proposal_index": index,
            "reason": "test",
            "seed": 1,
        }
        for index in range(1 << n_bits)
    ]


def test_positive_canary_loads_sha_pinned_real_64_state_objective() -> None:
    table = load_cached_objective_jsonl(CACHED_64)
    assert table.source_sha256 == PINNED_UGC64_SHA256
    assert table.n_bits == 6
    assert table.state_count == 64
    assert table.value([0, 0, 0, 0, 0, 0]) == pytest.approx(0.19081182131424618)
    assert table.value([1, 1, 1, 1, 1, 1]) == pytest.approx(0.19080359202934188)
    assert table.order_sha256 != table.objective_sha256
    assert table.state_sha256([0] * 6) == cached_state_sha256(
        [0] * 6, 0.19081182131424618
    )


def test_negative_canary_rejects_source_hash_drift(tmp_path: Path) -> None:
    path = tmp_path / "states.jsonl"
    path.write_bytes(CACHED_64.read_bytes() + b"\n")
    with pytest.raises(SFESSError, match="SHA mismatch"):
        load_cached_objective_jsonl(path)


@pytest.mark.parametrize("fault", ["missing", "duplicate", "nonfinite", "missing_key"])
def test_loader_rejects_missing_duplicate_nonfinite_and_missing_fields(
    tmp_path: Path, fault: str
) -> None:
    rows = _rows(3)
    if fault == "missing":
        rows.pop()
    elif fault == "duplicate":
        rows[3]["candidate_mask"] = rows[2]["candidate_mask"]
    elif fault == "nonfinite":
        rows[4]["candidate_value"] = float("nan")
    else:
        del rows[4]["candidate_value"]
    path = tmp_path / f"{fault}.jsonl"
    expected = _write_rows(path, rows)
    with pytest.raises(SFESSError):
        load_cached_objective_jsonl(path, expected_sha256=expected, n_bits=3)


def test_loader_enforces_little_endian_bit_order_asymmetry(tmp_path: Path) -> None:
    rows = _rows(3)
    assert rows[1]["candidate_mask"] == [1, 0, 0]
    assert rows[2]["candidate_mask"] == [0, 1, 0]
    rows[1]["candidate_mask"], rows[2]["candidate_mask"] = (
        rows[2]["candidate_mask"],
        rows[1]["candidate_mask"],
    )
    path = tmp_path / "wrong_order.jsonl"
    expected = _write_rows(path, rows)
    with pytest.raises(SFESSError, match="little-endian"):
        load_cached_objective_jsonl(path, expected_sha256=expected, n_bits=3)


def test_dft_poisson_binomial_matches_polynomial_reference() -> None:
    probabilities = np.array([0.03, 0.2, 0.51, 0.77, 0.93])
    polynomial = np.array([1.0])
    for probability in probabilities:
        polynomial = np.convolve(polynomial, [1.0 - probability, probability])
    assert np.allclose(poisson_binomial_pmf_dft(probabilities), polynomial, atol=2e-15)
    with pytest.raises(SFESSError, match=r"\[0, 1\]"):
        poisson_binomial_pmf_dft([0.2, 1.1])


def test_conditional_sampler_is_exact_k_and_matches_small_support_distribution() -> None:
    logits = np.log(np.array([0.2, 0.4, 0.7]) / np.array([0.8, 0.6, 0.3]))
    rng = np.random.default_rng(1234)
    draws = [sample_conditional_bernoulli_k_subset(logits, 1, rng) for _ in range(20_000)]
    assert all(int(mask.sum()) == 1 for mask in draws)
    empirical = np.mean(np.stack(draws), axis=0)
    odds = np.exp(logits)
    expected = odds / odds.sum()
    assert np.allclose(empirical, expected, atol=0.012, rtol=0.0)


def test_exact_logit_score_has_zero_mean_under_fixed_k_distribution() -> None:
    logits = np.array([-0.7, 0.2, 1.1])
    support = [np.eye(3, dtype=np.uint8)[index] for index in range(3)]
    odds = np.exp(logits)
    mass = odds / odds.sum()
    scores = [exact_k_subset_logit_score(mask, logits, 1) for mask in support]
    for mask, score in zip(support, scores, strict=True):
        assert np.allclose(score, mask.astype(np.float64) - mass, atol=2e-15)
    mean_score = sum(
        probability * score for probability, score in zip(mass, scores, strict=True)
    )
    assert np.allclose(mean_score, 0.0, atol=2e-15)


def test_leave_one_out_control_variate_is_exactly_unbiased_on_small_support() -> None:
    logits = np.array([-0.6, 0.1, 0.8])
    support = [np.eye(3, dtype=np.uint8)[index] for index in range(3)]
    values = np.array([1.3, -0.4, 2.1])
    odds = np.exp(logits)
    mass = odds / odds.sum()
    scores = [exact_k_subset_logit_score(mask, logits, 1) for mask in support]
    exact = sum(probability * value * score for probability, value, score in zip(mass, values, scores, strict=True))

    # M=2 exhaustive expectation of the leave-one-out estimator.  This is a
    # deterministic stronger canary than a Monte Carlo confidence interval.
    expected_estimator = np.zeros(3)
    for first, second in product(range(3), repeat=2):
        probability = mass[first] * mass[second]
        estimator = 0.5 * (
            (values[first] - values[second]) * scores[first]
            + (values[second] - values[first]) * scores[second]
        )
        expected_estimator += probability * estimator
    assert np.allclose(expected_estimator, exact, atol=2e-15)

    sample = sfess_leave_one_out_gradient(
        lambda mask: float(values[int(np.argmax(mask))]),
        logits,
        1,
        2,
        np.random.default_rng(7),
    )
    assert sample.gradient.shape == (3,)
    assert len(sample.values) == len(sample.masks) == 2


def test_production_leave_one_out_uses_exact_one_over_m_coefficient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    masks = (
        np.array([1, 0, 0], dtype=np.uint8),
        np.array([0, 1, 0], dtype=np.uint8),
        np.array([0, 0, 1], dtype=np.uint8),
        np.array([1, 0, 0], dtype=np.uint8),
        np.array([0, 1, 0], dtype=np.uint8),
    )
    cursor = 0

    def fixed_sampler(*_args: object, **_kwargs: object) -> np.ndarray:
        nonlocal cursor
        mask = masks[cursor]
        cursor += 1
        return mask.copy()

    monkeypatch.setattr(sfess_core, "sample_conditional_bernoulli_k_subset", fixed_sampler)
    values_by_index = np.array([1.25, -0.5, 2.75])
    sample = sfess_core.sfess_leave_one_out_gradient(
        lambda mask: float(values_by_index[int(np.argmax(mask))]),
        np.array([-0.4, 0.2, 0.9]),
        1,
        5,
        np.random.default_rng(11),
    )
    values = np.asarray(sample.values)
    scores = np.stack(sample.scores)
    baselines = (values.sum() - values) / 4.0
    unnormalized = np.sum((values - baselines)[:, None] * scores, axis=0)
    expected = unnormalized / 5.0
    assert np.allclose(sample.gradient, expected, atol=1e-15, rtol=0.0)
    assert not np.allclose(sample.gradient, unnormalized, atol=1e-15, rtol=0.0)


def test_counted_oracle_enforces_authorization_budget_and_only_records_queries() -> None:
    table = load_cached_objective_jsonl(CACHED_64)
    oracle = CountedCachedOracle(
        table, budget=2, authorize_lookup=lambda mask: int(mask.sum()) <= 1
    )
    zero = np.zeros(6, dtype=np.uint8)
    one = zero.copy()
    one[0] = 1
    assert oracle(zero, purpose="positive") == table.value(zero)
    assert oracle(one, purpose="positive") == table.value(one)
    assert oracle.calls == 2
    assert len(oracle.records) == 2
    assert not hasattr(oracle, "values")
    with pytest.raises(SFESSError, match="budget exhausted"):
        oracle(zero, purpose="negative")

    refused = CountedCachedOracle(table, budget=1, authorize_lookup=lambda mask: False)
    with pytest.raises(SFESSError, match="not authorized"):
        refused(zero, purpose="negative")
    assert refused.calls == 0


def test_fixed_k_search_retains_incumbent_on_non_improvement_and_counts_padding(
    tmp_path: Path,
) -> None:
    # Among k=1 states, [0,1] is strictly better than the deterministic initial
    # [1,0].  The second attempted swap goes back uphill and must be rejected.
    path = tmp_path / "two_bit.jsonl"
    expected_sha = _write_rows(path, _rows(2, [9.0, 3.0, 1.0, 8.0]))
    table = load_cached_objective_jsonl(path, expected_sha256=expected_sha, n_bits=2)
    oracle = CountedCachedOracle(table, budget=8, authorize_lookup=lambda mask: True)
    result = SFESSFixedKSearch(
        oracle, n_bits=2, k=1, samples_per_gradient=2, seed=9, comparison_noise_floor_s=0.0
    ).run(8, tmp_path / "snapshot.json")
    assert result.current_mask == result.best_mask == (0, 1)
    assert result.current_value == result.best_value == 1.0
    assert result.accepted == 1
    assert result.calls == 8
    assert result.padding == 1
    assert result.complete
    assert all(sum(record.mask) == 1 for record in result.query_records)


def test_registered_noise_floor_controls_acceptance_and_resume_replay(tmp_path: Path) -> None:
    initial = 1.0
    subfloor = initial - 5.0e-13
    path = tmp_path / "subfloor.jsonl"
    expected_sha = _write_rows(path, _rows(2, [3.0, initial, subfloor, 4.0]))
    table = load_cached_objective_jsonl(path, expected_sha256=expected_sha, n_bits=2)
    snapshot = tmp_path / "subfloor_snapshot.json"
    oracle = CountedCachedOracle(table, budget=4, authorize_lookup=lambda mask: True)
    result = SFESSFixedKSearch(
        oracle,
        n_bits=2,
        k=1,
        samples_per_gradient=2,
        seed=9,
        comparison_noise_floor_s=1.0e-12,
    ).run(4, snapshot)
    assert result.query_records[-1].purpose == "strict_exact_gate"
    assert result.query_records[-1].value == subfloor
    assert result.current_value == initial
    assert result.accepted == 0

    resumed_oracle = CountedCachedOracle(table, budget=4, authorize_lookup=lambda mask: True)
    resumed = SFESSFixedKSearch.resume_from(
        snapshot,
        resumed_oracle,
        expected_k=1,
        expected_samples_per_gradient=2,
        expected_seed=9,
        expected_comparison_noise_floor_s=1.0e-12,
    )
    assert resumed.current_value == initial
    assert resumed.accepted == 0

    payload = json.loads(snapshot.read_text())
    payload["comparison_noise_floor_s"] = 0.0
    snapshot.write_text(json.dumps(payload), encoding="utf-8")
    forged_oracle = CountedCachedOracle(table, budget=4, authorize_lookup=lambda mask: True)
    with pytest.raises(SFESSError, match="comparison noise floor mismatch"):
        SFESSFixedKSearch.resume_from(
            snapshot,
            forged_oracle,
            expected_k=1,
            expected_samples_per_gradient=2,
            expected_seed=9,
            expected_comparison_noise_floor_s=1.0e-12,
        )


def test_deterministic_resume_matches_uninterrupted_query_sequence(tmp_path: Path) -> None:
    table = load_cached_objective_jsonl(CACHED_64)
    full_oracle = CountedCachedOracle(table, budget=20, authorize_lookup=lambda mask: True)
    full = SFESSFixedKSearch(
        full_oracle,
        n_bits=6,
        k=3,
        samples_per_gradient=3,
        seed=2407158,
        comparison_noise_floor_s=1.0e-12,
    ).run(20, tmp_path / "full.json")

    partial_oracle = CountedCachedOracle(table, budget=20, authorize_lookup=lambda mask: True)
    partial = SFESSFixedKSearch(
        partial_oracle,
        n_bits=6,
        k=3,
        samples_per_gradient=3,
        seed=2407158,
        comparison_noise_floor_s=1.0e-12,
    ).run(20, tmp_path / "resume.json", stop_after_calls=9)
    assert partial.calls == 9
    assert not partial.complete

    # resume_from restores and validates the snapshot's counted records into a
    # fresh oracle; callers do not need to duplicate that parsing themselves.
    authorized_query_indices: list[int] = []
    resumed_oracle: CountedCachedOracle

    def authorize_resumed(mask: np.ndarray) -> bool:
        authorized_query_indices.append(resumed_oracle.calls + 1)
        return True

    resumed_oracle = CountedCachedOracle(table, budget=20, authorize_lookup=authorize_resumed)
    resumed_search = SFESSFixedKSearch.resume_from(
        tmp_path / "resume.json",
        resumed_oracle,
        expected_k=3,
        expected_samples_per_gradient=3,
        expected_seed=2407158,
        expected_comparison_noise_floor_s=1.0e-12,
    )
    resumed = resumed_search.run(20, tmp_path / "resume.json")
    assert resumed == full
    assert authorized_query_indices[0] == partial.calls + 1
    assert authorized_query_indices == list(range(partial.calls + 1, 21))


def test_resume_rejects_snapshot_objective_fingerprint_drift(tmp_path: Path) -> None:
    table = load_cached_objective_jsonl(CACHED_64)
    oracle = CountedCachedOracle(table, budget=8, authorize_lookup=lambda mask: True)
    partial = SFESSFixedKSearch(
        oracle,
        n_bits=6,
        k=3,
        samples_per_gradient=2,
        seed=3,
        comparison_noise_floor_s=1.0e-12,
    ).run(8, tmp_path / "resume.json", stop_after_calls=4)
    payload = json.loads((tmp_path / "resume.json").read_text())
    payload["objective_sha256"] = "0" * 64
    (tmp_path / "resume.json").write_text(json.dumps(payload))
    resumed_oracle = CountedCachedOracle(
        table, budget=8, authorize_lookup=lambda mask: True, prior_records=partial.query_records
    )
    with pytest.raises(SFESSError, match="objective_sha256 mismatch"):
        SFESSFixedKSearch.resume_from(
            tmp_path / "resume.json",
            resumed_oracle,
            expected_k=3,
            expected_samples_per_gradient=2,
            expected_seed=3,
            expected_comparison_noise_floor_s=1.0e-12,
        )


@pytest.mark.parametrize(
    ("fault", "reason"),
    [
        ("purpose", "purpose.*not registered"),
        ("accepted", "counters disagree"),
        ("padding", "counters disagree"),
        ("rng", "RNG state disagrees"),
        ("sample_mask", "sample mask/RNG trace mismatch"),
    ],
)
def test_resume_rederives_semantic_trace_instead_of_trusting_snapshot_fields(
    tmp_path: Path, fault: str, reason: str
) -> None:
    table = load_cached_objective_jsonl(CACHED_64)
    snapshot = tmp_path / f"resume_{fault}.json"
    oracle = CountedCachedOracle(table, budget=20, authorize_lookup=lambda mask: True)
    SFESSFixedKSearch(
        oracle,
        n_bits=6,
        k=3,
        samples_per_gradient=3,
        seed=77,
        comparison_noise_floor_s=1.0e-12,
    ).run(
        20, snapshot, stop_after_calls=9
    )
    payload = json.loads(snapshot.read_text())
    if fault == "purpose":
        payload["query_records"][1]["purpose"] = "forged_not_sfess_sample"
    elif fault == "accepted":
        payload["accepted"] += 1
    elif fault == "padding":
        payload["padding"] += 1
    elif fault == "rng":
        payload["rng_state"]["state"]["state"] += 1
    else:
        record = payload["query_records"][1]
        replacement = np.roll(np.asarray(record["mask"], dtype=np.uint8), 1)
        assert int(replacement.sum()) == 3 and replacement.tolist() != record["mask"]
        record["mask"] = replacement.tolist()
        record["value"] = table.value(replacement)
        record["state_sha256"] = table.state_sha256(replacement)
    snapshot.write_text(json.dumps(payload), encoding="utf-8")

    resumed_oracle = CountedCachedOracle(table, budget=20, authorize_lookup=lambda mask: True)
    with pytest.raises(SFESSError, match=reason):
        SFESSFixedKSearch.resume_from(
            snapshot,
            resumed_oracle,
            expected_k=3,
            expected_samples_per_gradient=3,
            expected_seed=77,
            expected_comparison_noise_floor_s=1.0e-12,
        )
    assert resumed_oracle.calls == 0
