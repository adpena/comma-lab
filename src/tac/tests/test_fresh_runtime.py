"""Regression tests for bounded FreSh initialization orchestration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import tac.witness_init.fresh_runtime as fresh_runtime
from tac.witness_init.fresh_frequency_shift import wasserstein1_cdf_l1
from tac.witness_init.fresh_runtime import (
    FreShCandidate,
    deterministic_evenly_spaced_indices,
    fresh_init_scorer_accounting,
    ordered_fresh_candidates,
    run_fresh_initialization_sweep,
    score_fresh_committed_state,
    write_fresh_committed_state_receipt,
    write_fresh_receipt,
)


def _vertical_partition(*, height: int = 8, width: int = 10, split: int = 5) -> np.ndarray:
    labels = np.zeros((height, width), dtype=np.int32)
    labels[:, split:] = 1
    return labels


def _island_partition(*, height: int = 8, width: int = 10) -> np.ndarray:
    labels = np.zeros((height, width), dtype=np.int64)
    labels[2:6, 3:7] = 2
    return labels


def _dash_comb_partition(*, fixed_edge: int | None, dash_shift: int = 0) -> np.ndarray:
    labels = np.ones((32, 48), dtype=np.int64)
    if fixed_edge is not None:
        labels[:, :fixed_edge] = 0
    for row in (5 + dash_shift, 13 + dash_shift, 21 + dash_shift):
        labels[row : row + 2, 35:42] = 2
    return labels


def _run_small_sweep(targets: list[np.ndarray] | None = None):
    target_maps = targets or [_vertical_partition() for _ in range(5)]

    def render(candidate: FreShCandidate) -> np.ndarray:
        if candidate.freq_along == 8.0:
            return np.zeros((8, 10), dtype=np.int64)
        return _vertical_partition()

    return run_fresh_initialization_sweep(
        target_label_maps=target_maps,
        requested_sample_count=3,
        spectrum_size=6,
        frequency_candidates=(8.0, 16.0),
        bias_candidates=(0.0,),
        baseline_frequency=8.0,
        baseline_bias=0.0,
        render_candidate=render,
    )


def test_evenly_spaced_sampling_is_integer_deterministic_and_bounded() -> None:
    assert deterministic_evenly_spaced_indices(10, 4) == (0, 3, 6, 9)
    assert deterministic_evenly_spaced_indices(8, 10) == tuple(range(8))
    assert deterministic_evenly_spaced_indices(10, 1) == (4,)
    assert deterministic_evenly_spaced_indices(10, 4) == deterministic_evenly_spaced_indices(10, 4)
    with pytest.raises(ValueError, match="positive integer"):
        deterministic_evenly_spaced_indices(0, 4)
    with pytest.raises(ValueError, match="positive integer"):
        deterministic_evenly_spaced_indices(4, True)


def test_cartesian_sweep_puts_exact_baseline_first_and_stably_deduplicates() -> None:
    candidates = ordered_fresh_candidates(
        (8.0, 14.0, 8.0, 25.6),
        (0.0, 1.0, 0.0),
        baseline_frequency=14.0,
        baseline_bias=1.0,
    )
    assert candidates == (
        FreShCandidate(14.0, 1.0),
        FreShCandidate(8.0, 0.0),
        FreShCandidate(8.0, 1.0),
        FreShCandidate(14.0, 0.0),
        FreShCandidate(25.6, 0.0),
        FreShCandidate(25.6, 1.0),
    )
    with pytest.raises(ValueError, match="exact baseline"):
        ordered_fresh_candidates(
            (8.0,),
            (0.0,),
            baseline_frequency=16.0,
            baseline_bias=0.0,
        )
    with pytest.raises(ValueError, match="finite"):
        ordered_fresh_candidates(
            (np.nan,),
            (0.0,),
            baseline_frequency=8.0,
            baseline_bias=0.0,
        )
    with pytest.raises(ValueError, match="finite"):
        ordered_fresh_candidates(
            (True,),
            (0.0,),
            baseline_frequency=1.0,
            baseline_bias=0.0,
        )


def test_sweep_reuses_one_cold_output_and_selects_exact_boundary_spectrum() -> None:
    calls: list[FreShCandidate] = []
    target_maps = [_vertical_partition() for _ in range(7)]

    def render(candidate: FreShCandidate) -> np.ndarray:
        calls.append(candidate)
        if candidate == FreShCandidate(8.0, 0.0):
            return _island_partition()
        return _vertical_partition()

    result = run_fresh_initialization_sweep(
        target_label_maps=target_maps,
        requested_sample_count=4,
        spectrum_size=6,
        frequency_candidates=(8.0, 16.0),
        bias_candidates=(0.0,),
        baseline_frequency=8.0,
        baseline_bias=0.0,
        render_candidate=render,
    )

    assert result.sampled_pair_indices == (0, 2, 4, 6)
    assert result.initialization_draws == 1
    assert result.init_scorer_forward_calls == len(result.ordered_candidates)
    assert result.init_scorer_pair_equivalents == len(result.ordered_candidates)
    assert result.distance_averaging_axis == "target_pairs"
    assert calls == list(result.ordered_candidates)
    assert len(calls) == 2  # Once per candidate, never once per target.
    assert result.selection.candidate == FreShCandidate(16.0, 0.0)
    assert result.selection.candidate_index == 1
    assert result.selection.mean_distance == pytest.approx(0.0, abs=1e-15)
    assert tuple(distance.pair_index for distance in result.selection.target_distances) == (0, 2, 4, 6)
    assert all(distance.wasserstein1 == pytest.approx(0.0) for distance in result.selection.target_distances)
    assert all(len(target.spectrum) == 6 for target in result.targets)
    assert all(candidate.spectrum is None or len(candidate.spectrum) == 6 for candidate in result.candidates)


def test_residual_weighted_selection_ignores_dominant_non_residual_edge() -> None:
    residual_weight = np.zeros((32, 48), dtype=np.float64)
    residual_weight[:, 31:] = 1.0

    def sweep(fixed_edge: int | None):
        target = _dash_comb_partition(fixed_edge=fixed_edge)

        def render(candidate: FreShCandidate) -> np.ndarray:
            if candidate.freq_along == 8.0:
                # Residual-exact candidate deliberately misses the dominant
                # global edge, so only the residual surface should select it.
                return _dash_comb_partition(fixed_edge=None)
            return _dash_comb_partition(fixed_edge=fixed_edge, dash_shift=3)

        return run_fresh_initialization_sweep(
            target_label_maps=[target],
            spectral_weight_maps=[residual_weight],
            requested_sample_count=1,
            spectrum_size=16,
            frequency_candidates=(8.0, 16.0),
            bias_candidates=(0.0,),
            baseline_frequency=8.0,
            baseline_bias=0.0,
            render_candidate=render,
        )

    without_outside_mass = sweep(None)
    with_dominant_outside_mass = sweep(12)
    assert without_outside_mass.selection.candidate == FreShCandidate(8.0, 0.0)
    assert with_dominant_outside_mass.selection.candidate == FreShCandidate(8.0, 0.0)
    selected, globally_better = with_dominant_outside_mass.candidates
    assert selected.target_distances[0].wasserstein1 == pytest.approx(0.0, abs=1e-15)
    assert selected.target_distances[0].global_boundary_wasserstein1 > (
        globally_better.target_distances[0].global_boundary_wasserstein1
    )


def test_committed_state_requires_identical_weight_maps_and_reuses_both_distances() -> None:
    weights = np.zeros((8, 10), dtype=np.float64)
    weights[:, 3:7] = 1.0
    result = run_fresh_initialization_sweep(
        target_label_maps=[_island_partition()],
        spectral_weight_maps=[weights],
        requested_sample_count=1,
        spectrum_size=6,
        frequency_candidates=(8.0,),
        bias_candidates=(0.0,),
        baseline_frequency=8.0,
        baseline_bias=0.0,
        render_candidate=lambda _candidate: _island_partition(),
    )
    assert result.spectral_weight_map_sha256s == (result.targets[0].spectral_weight_sha256,)
    telemetry = score_fresh_committed_state(_island_partition(), result, spectral_weight_maps=[weights])
    assert telemetry.target_distances[0].wasserstein1 == pytest.approx(0.0)
    assert telemetry.target_distances[0].global_boundary_wasserstein1 == pytest.approx(0.0)
    with pytest.raises(ValueError, match="must match the weighted/unweighted"):
        score_fresh_committed_state(_island_partition(), result)
    changed = weights.copy()
    changed[0, 0] = 1.0
    with pytest.raises(ValueError, match="hash does not match selection"):
        score_fresh_committed_state(_island_partition(), result, spectral_weight_maps=[changed])


def test_weight_maps_fail_closed_on_negative_or_wrong_count() -> None:
    kwargs = {
        "target_label_maps": [_vertical_partition()],
        "requested_sample_count": 1,
        "spectrum_size": 6,
        "frequency_candidates": (8.0,),
        "bias_candidates": (0.0,),
        "baseline_frequency": 8.0,
        "baseline_bias": 0.0,
        "render_candidate": lambda _candidate: _vertical_partition(),
    }
    with pytest.raises(ValueError, match="non-negative"):
        run_fresh_initialization_sweep(**kwargs, spectral_weight_maps=[-np.ones((8, 10), dtype=np.float64)])
    with pytest.raises(ValueError, match="supplies 2 maps for 1 targets"):
        run_fresh_initialization_sweep(
            **kwargs,
            spectral_weight_maps=[np.ones((8, 10)), np.ones((8, 10))],
        )


def test_per_target_wasserstein_telemetry_is_exact_and_ties_are_stable() -> None:
    targets = [_vertical_partition(), _island_partition()]

    def render(_candidate: FreShCandidate) -> np.ndarray:
        return _vertical_partition()

    result = run_fresh_initialization_sweep(
        target_label_maps=targets,
        requested_sample_count=2,
        spectrum_size=6,
        frequency_candidates=(8.0, 16.0),
        bias_candidates=(0.0,),
        baseline_frequency=8.0,
        baseline_bias=0.0,
        render_candidate=render,
    )
    baseline = result.candidates[0]
    assert baseline.spectrum is not None
    expected = tuple(wasserstein1_cdf_l1(baseline.spectrum, target.spectrum) for target in result.targets)
    assert tuple(distance.wasserstein1 for distance in baseline.target_distances) == pytest.approx(expected)
    assert tuple(distance.residual_wasserstein1 for distance in baseline.target_distances) == pytest.approx(expected)
    assert tuple(distance.global_boundary_wasserstein1 for distance in baseline.target_distances) == pytest.approx(
        expected
    )
    assert baseline.mean_distance == pytest.approx(sum(expected) / len(expected))
    assert result.selection.candidate == result.baseline_candidate
    assert result.selection.candidate_index == 0


def test_zero_boundary_candidate_is_rejected_without_aborting_valid_siblings() -> None:
    result = _run_small_sweep()
    rejected, eligible = result.candidates
    assert rejected.status == "rejected_degenerate"
    assert rejected.rejection_reason == "zero_boundary"
    assert rejected.boundary_pixels == 0
    assert rejected.spectrum is None
    assert rejected.target_distances == ()
    assert eligible.status == "eligible"
    assert result.selection.candidate == eligible.candidate


def test_all_degenerate_candidates_fail_closed() -> None:
    with pytest.raises(ValueError, match="all FreSh candidates"):
        run_fresh_initialization_sweep(
            target_label_maps=[_vertical_partition()],
            requested_sample_count=1,
            spectrum_size=6,
            frequency_candidates=(8.0, 16.0),
            bias_candidates=(0.0,),
            baseline_frequency=8.0,
            baseline_bias=0.0,
            render_candidate=lambda _candidate: np.zeros((8, 10), dtype=np.int64),
        )


def test_target_shape_mismatch_fails_before_any_candidate_render() -> None:
    calls = 0

    def render(_candidate: FreShCandidate) -> np.ndarray:
        nonlocal calls
        calls += 1
        return _vertical_partition()

    with pytest.raises(ValueError, match="target shapes differ"):
        run_fresh_initialization_sweep(
            target_label_maps=[_vertical_partition(), np.zeros((9, 10), dtype=np.int64)],
            requested_sample_count=1,
            spectrum_size=6,
            frequency_candidates=(8.0,),
            bias_candidates=(0.0,),
            baseline_frequency=8.0,
            baseline_bias=0.0,
            render_candidate=render,
        )
    assert calls == 0


def test_target_and_boundary_hashes_are_deterministic_and_content_bound() -> None:
    first = _run_small_sweep([_vertical_partition()])
    second = _run_small_sweep([_vertical_partition()])
    changed = _run_small_sweep([_vertical_partition(split=3)])

    assert first.targets[0].label_sha256 == second.targets[0].label_sha256
    assert first.targets[0].boundary_sha256 == second.targets[0].boundary_sha256
    assert len(first.targets[0].label_sha256) == 64
    assert len(first.targets[0].boundary_sha256) == 64
    assert first.targets[0].label_sha256 != changed.targets[0].label_sha256
    assert first.targets[0].boundary_sha256 != changed.targets[0].boundary_sha256


def test_post_structured_state_is_measured_against_frozen_selection_targets() -> None:
    result = _run_small_sweep([_vertical_partition(), _island_partition()])
    telemetry = score_fresh_committed_state(_vertical_partition(), result)
    expected = tuple(wasserstein1_cdf_l1(telemetry.spectrum, target.spectrum) for target in result.targets)
    assert tuple(distance.wasserstein1 for distance in telemetry.target_distances) == pytest.approx(expected)
    assert telemetry.mean_distance == pytest.approx(sum(expected) / len(expected))
    assert telemetry.boundary_pixels > 0
    with pytest.raises(ValueError, match="zero boundary"):
        score_fresh_committed_state(np.zeros((8, 10), dtype=np.int64), result)


def test_receipt_is_canonical_atomic_and_sha_bound(tmp_path: Path) -> None:
    result = _run_small_sweep()
    destination = tmp_path / "durable" / "fresh_receipt.json"
    digest = write_fresh_receipt(
        destination,
        result,
        provenance={"seed": 7, "axis": "macOS-MLX advisory"},
    )
    encoded = destination.read_bytes()
    payload = json.loads(encoded)
    assert digest == hashlib.sha256(encoded).hexdigest()
    assert payload["schema"] == "tac.witness_init.fresh_runtime.v1"
    assert payload["result"]["selection"]["candidate"] == {
        "bias_k": 0.0,
        "freq_along": 16.0,
    }
    assert payload["provenance"]["seed"] == 7
    assert not list(destination.parent.glob(f".{destination.name}.*.tmp"))
    # Canonical JSON bytes and hash repeat for identical evidence.
    assert write_fresh_receipt(destination, result, provenance={"seed": 7, "axis": "macOS-MLX advisory"}) == digest
    assert destination.read_bytes() == encoded


def test_post_structured_receipt_links_exact_selection_bytes(tmp_path: Path) -> None:
    result = _run_small_sweep()
    selection_path = tmp_path / "selection.json"
    selection_sha = write_fresh_receipt(selection_path, result)
    telemetry = score_fresh_committed_state(_vertical_partition(), result)
    accounting = fresh_init_scorer_accounting(result)
    destination = tmp_path / "post_structured.json"
    digest = write_fresh_committed_state_receipt(
        destination,
        telemetry,
        selection_receipt_sha256=selection_sha,
        scorer_accounting=accounting,
        provenance={"stage": "after_structured_init"},
    )
    encoded = destination.read_bytes()
    payload = json.loads(encoded)
    assert digest == hashlib.sha256(encoded).hexdigest()
    assert payload["schema"] == "tac.witness_init.fresh_committed_state.v1"
    assert payload["selection_receipt_sha256"] == selection_sha
    assert payload["result"]["mean_distance"] == pytest.approx(telemetry.mean_distance)
    assert payload["result"]["total_init_scorer_forward_calls"] == (
        result.init_scorer_forward_calls + 1
    )
    assert payload["result"]["total_init_scorer_pair_equivalents"] == (
        result.init_scorer_pair_equivalents + 1
    )
    with pytest.raises(ValueError, match="64 hexadecimal"):
        write_fresh_committed_state_receipt(
            destination,
            telemetry,
            selection_receipt_sha256="not-a-sha",
            scorer_accounting=accounting,
        )


def test_committed_scorer_accounting_includes_mandatory_post_selection_forward() -> None:
    result = _run_small_sweep()
    accounting = fresh_init_scorer_accounting(result)
    assert accounting.selection_scorer_forward_calls == len(result.ordered_candidates)
    assert accounting.selection_scorer_pair_equivalents == len(result.ordered_candidates)
    assert accounting.committed_state_scorer_forward_calls == 1
    assert accounting.committed_state_scorer_pair_equivalents == 1
    assert accounting.total_init_scorer_forward_calls == len(result.ordered_candidates) + 1
    assert accounting.total_init_scorer_pair_equivalents == len(result.ordered_candidates) + 1


def test_receipt_replace_failure_preserves_previous_bytes_and_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _run_small_sweep()
    destination = tmp_path / "fresh_receipt.json"
    destination.write_bytes(b"prior durable evidence\n")

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(fresh_runtime.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected"):
        write_fresh_receipt(destination, result)
    assert destination.read_bytes() == b"prior durable evidence\n"
    assert not list(destination.parent.glob(f".{destination.name}.*.tmp"))


def test_receipt_refuses_tmp_even_before_writing() -> None:
    result = _run_small_sweep()
    with pytest.raises(ValueError, match="cannot be under /tmp"):
        write_fresh_receipt("/tmp/fresh_runtime_forbidden/receipt.json", result)
