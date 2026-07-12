"""Bounded-memory runtime orchestration for FreSh witness initialization.

The trainer-facing callback renders exactly one cold, realized-through-R
SegNet argmax label map for each ``(freq_along, bias_k)`` candidate.  Because
the cold per-pair codes are all zero, that one initialized output is shared
across every sampled target pair.  This module reduces both targets and each
candidate to FreSh boundary spectra immediately; image-sized candidate outputs
are never retained across sweep iterations.

This is an initialization selector, not a score authority.  A contest score
still requires the untouched upstream evaluator on exact archive bytes.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from tac.witness_init.fresh_frequency_shift import (
    fresh_spectrum,
    label_boundary_target_map,
    wasserstein1_cdf_l1,
)


@dataclass(frozen=True, order=True)
class FreShCandidate:
    """One directional-frequency and first-layer-bias initialization."""

    freq_along: float
    bias_k: float


@dataclass(frozen=True)
class FreShTargetSpectrum:
    """Durable spectral identity for one sampled target pair."""

    pair_index: int
    shape: tuple[int, int]
    label_sha256: str
    boundary_sha256: str
    boundary_pixels: int
    spectrum: tuple[float, ...]
    spectral_weight_sha256: str | None
    residual_spectrum: tuple[float, ...]


@dataclass(frozen=True)
class FreShTargetDistance:
    """One exact CDF-L1 Wasserstein distance to a sampled target."""

    pair_index: int
    residual_wasserstein1: float
    wasserstein1: float
    global_boundary_wasserstein1: float


@dataclass(frozen=True)
class FreShCandidateTelemetry:
    """Per-candidate spectrum, hashes, and per-target distance telemetry."""

    candidate: FreShCandidate
    candidate_index: int
    status: Literal["eligible", "rejected_degenerate"]
    label_sha256: str
    boundary_sha256: str
    boundary_pixels: int
    spectrum: tuple[float, ...] | None
    target_distances: tuple[FreShTargetDistance, ...]
    mean_distance: float | None
    rejection_reason: Literal["zero_boundary", "zero_non_dc_spectrum"] | None


@dataclass(frozen=True)
class FreShRuntimeSelection:
    """The stable minimum-distance eligible candidate."""

    candidate: FreShCandidate
    candidate_index: int
    mean_distance: float
    target_distances: tuple[FreShTargetDistance, ...]
    label_sha256: str
    boundary_sha256: str


@dataclass(frozen=True)
class FreShRuntimeResult:
    """Complete bounded FreSh sweep result suitable for a durable receipt."""

    requested_sample_count: int
    total_target_pairs: int
    sampled_pair_indices: tuple[int, ...]
    initialization_draws: int
    init_scorer_forward_calls: int
    init_scorer_pair_equivalents: int
    distance_averaging_axis: Literal["target_pairs"]
    spectrum_size: int
    baseline_candidate: FreShCandidate
    ordered_candidates: tuple[FreShCandidate, ...]
    targets: tuple[FreShTargetSpectrum, ...]
    uses_spectral_weight_maps: bool
    spectral_weight_map_sha256s: tuple[str, ...] | None
    candidates: tuple[FreShCandidateTelemetry, ...]
    selection: FreShRuntimeSelection


@dataclass(frozen=True)
class FreShCommittedStateTelemetry:
    """Spectrum of the final initialization state handed to epoch zero."""

    label_sha256: str
    boundary_sha256: str
    boundary_pixels: int
    spectrum: tuple[float, ...]
    target_distances: tuple[FreShTargetDistance, ...]
    mean_distance: float


@dataclass(frozen=True)
class FreShInitScorerAccounting:
    """Exact scorer calls from sweep start through the committed epoch-0 state."""

    selection_scorer_forward_calls: int
    selection_scorer_pair_equivalents: int
    committed_state_scorer_forward_calls: int
    committed_state_scorer_pair_equivalents: int
    total_init_scorer_forward_calls: int
    total_init_scorer_pair_equivalents: int


CandidateRenderer = Callable[[FreShCandidate], ArrayLike]


def deterministic_evenly_spaced_indices(
    total_pairs: int,
    requested_count: int,
) -> tuple[int, ...]:
    """Return deterministic, endpoint-covering, nearly even pair indices.

    If fewer pairs exist than requested, every pair is used.  For one sample,
    the lower midpoint is used rather than privileging the first pair.  For two
    or more samples, integer arithmetic includes both endpoints and avoids
    floating-point or rounding-mode drift.
    """

    total = _positive_int(total_pairs, name="total_pairs")
    requested = _positive_int(requested_count, name="requested_count")
    count = min(total, requested)
    if count == 1:
        return ((total - 1) // 2,)
    return tuple(index * (total - 1) // (count - 1) for index in range(count))


def ordered_fresh_candidates(
    frequency_candidates: Sequence[float],
    bias_candidates: Sequence[float],
    *,
    baseline_frequency: float,
    baseline_bias: float,
) -> tuple[FreShCandidate, ...]:
    """Return the ordered Cartesian sweep with the exact baseline first.

    Candidate axes are stable-de-duplicated without tolerance.  The baseline
    must occur in the supplied Cartesian product; silently inventing an
    unswept control would make the comparison non-reproducible.
    """

    frequencies = _validated_axis(
        frequency_candidates,
        name="frequency_candidates",
        strictly_positive=True,
    )
    biases = _validated_axis(
        bias_candidates,
        name="bias_candidates",
        strictly_positive=False,
    )
    baseline = FreShCandidate(
        freq_along=_finite_float(
            baseline_frequency,
            name="baseline_frequency",
            strictly_positive=True,
        ),
        bias_k=_finite_float(
            baseline_bias,
            name="baseline_bias",
            strictly_positive=False,
        ),
    )
    cartesian = tuple(FreShCandidate(freq_along=frequency, bias_k=bias) for frequency in frequencies for bias in biases)
    if baseline not in cartesian:
        raise ValueError(
            "the exact baseline (baseline_frequency, baseline_bias) must occur in the supplied Cartesian candidate grid"
        )
    return (baseline, *(candidate for candidate in cartesian if candidate != baseline))


def run_fresh_initialization_sweep(
    *,
    target_label_maps: Sequence[ArrayLike] | NDArray[np.integer[Any]],
    requested_sample_count: int,
    spectrum_size: int,
    frequency_candidates: Sequence[float],
    bias_candidates: Sequence[float],
    baseline_frequency: float,
    baseline_bias: float,
    render_candidate: CandidateRenderer,
    spectral_weight_maps: Sequence[ArrayLike] | NDArray[np.floating[Any]] | None = None,
) -> FreShRuntimeResult:
    """Run a deterministic FreSh sweep without retaining candidate images.

    ``render_candidate`` is called once, in ``ordered_fresh_candidates`` order,
    for each candidate.  Its return value must be a two-dimensional integer
    SegNet argmax label map produced by the cold witness through the real R
    surface.  One map is compared to every sampled target spectrum because the
    cold witness has zero per-pair codes.
    """

    target_maps = _as_target_map_sequence(target_label_maps)
    weight_maps = _as_optional_weight_map_sequence(spectral_weight_maps, len(target_maps))
    requested = _positive_int(requested_sample_count, name="requested_sample_count")
    spectrum_n = _positive_int(spectrum_size, name="spectrum_size")
    if not callable(render_candidate):
        raise ValueError("render_candidate must be callable")

    ordered_candidates = ordered_fresh_candidates(
        frequency_candidates,
        bias_candidates,
        baseline_frequency=baseline_frequency,
        baseline_bias=baseline_bias,
    )
    target_shape = _validate_all_target_shapes(target_maps)
    if weight_maps is not None:
        weight_maps = tuple(
            _canonical_weight_map(
                weight,
                expected_shape=target_shape,
                name=f"spectral_weight_maps[{index}]",
            )
            for index, weight in enumerate(weight_maps)
        )
    sample_indices = deterministic_evenly_spaced_indices(len(target_maps), requested)
    targets = tuple(
        _target_spectrum_record(
            pair_index=pair_index,
            labels=target_maps[pair_index],
            spectral_weight=None if weight_maps is None else weight_maps[pair_index],
            expected_shape=target_shape,
            spectrum_size=spectrum_n,
        )
        for pair_index in sample_indices
    )
    telemetry: list[FreShCandidateTelemetry] = []
    best: FreShCandidateTelemetry | None = None
    for candidate_index, candidate in enumerate(ordered_candidates):
        # The image-sized value exists for this iteration only.  Everything
        # retained below is a hash, scalar, or ``spectrum_n``-length tuple.
        labels = _canonical_label_map(
            render_candidate(candidate),
            name=f"candidate {candidate!r} output",
        )
        if labels.shape != target_shape:
            raise ValueError(
                f"candidate {candidate!r} output shape {labels.shape} does not match target shape {target_shape}"
            )
        boundary = label_boundary_target_map(labels)
        label_sha256 = _canonical_array_sha256(labels)
        boundary_sha256 = _canonical_array_sha256(boundary)
        boundary_pixels = int(np.count_nonzero(boundary))

        if boundary_pixels == 0:
            record = FreShCandidateTelemetry(
                candidate=candidate,
                candidate_index=candidate_index,
                status="rejected_degenerate",
                label_sha256=label_sha256,
                boundary_sha256=boundary_sha256,
                boundary_pixels=0,
                spectrum=None,
                target_distances=(),
                mean_distance=None,
                rejection_reason="zero_boundary",
            )
        else:
            try:
                candidate_spectrum = fresh_spectrum(boundary, spectrum_n)
            except ValueError as exc:
                # A real boundary can still carry no mass in the retained
                # anti-diagonals.  That candidate is spectrally degenerate,
                # not a reason to discard valid siblings in the same sweep.
                if not str(exc).startswith("FreSh non-DC spectrum has zero"):
                    raise
                record = FreShCandidateTelemetry(
                    candidate=candidate,
                    candidate_index=candidate_index,
                    status="rejected_degenerate",
                    label_sha256=label_sha256,
                    boundary_sha256=boundary_sha256,
                    boundary_pixels=boundary_pixels,
                    spectrum=None,
                    target_distances=(),
                    mean_distance=None,
                    rejection_reason="zero_non_dc_spectrum",
                )
            else:
                try:
                    target_distances = tuple(
                        _candidate_target_distance(
                            boundary=boundary,
                            global_spectrum=candidate_spectrum,
                            target=target,
                            spectral_weight=(None if weight_maps is None else weight_maps[target.pair_index]),
                            spectrum_size=spectrum_n,
                        )
                        for target in targets
                    )
                except ValueError as exc:
                    if not str(exc).startswith("FreSh non-DC spectrum has zero"):
                        raise
                    record = FreShCandidateTelemetry(
                        candidate=candidate,
                        candidate_index=candidate_index,
                        status="rejected_degenerate",
                        label_sha256=label_sha256,
                        boundary_sha256=boundary_sha256,
                        boundary_pixels=boundary_pixels,
                        spectrum=None,
                        target_distances=(),
                        mean_distance=None,
                        rejection_reason="zero_non_dc_spectrum",
                    )
                    telemetry.append(record)
                    del labels, boundary
                    continue
                mean_distance = math.fsum(distance.residual_wasserstein1 for distance in target_distances) / len(
                    target_distances
                )
                record = FreShCandidateTelemetry(
                    candidate=candidate,
                    candidate_index=candidate_index,
                    status="eligible",
                    label_sha256=label_sha256,
                    boundary_sha256=boundary_sha256,
                    boundary_pixels=boundary_pixels,
                    spectrum=tuple(float(value) for value in candidate_spectrum),
                    target_distances=target_distances,
                    mean_distance=float(mean_distance),
                    rejection_reason=None,
                )
                # Strict improvement preserves candidate ordering on ties.
                if best is None or mean_distance < _eligible_mean(best):
                    best = record

        telemetry.append(record)
        del labels, boundary

    if best is None:
        raise ValueError("all FreSh candidates are spectrally degenerate")

    selection = FreShRuntimeSelection(
        candidate=best.candidate,
        candidate_index=best.candidate_index,
        mean_distance=_eligible_mean(best),
        target_distances=best.target_distances,
        label_sha256=best.label_sha256,
        boundary_sha256=best.boundary_sha256,
    )
    return FreShRuntimeResult(
        requested_sample_count=requested,
        total_target_pairs=len(target_maps),
        sampled_pair_indices=sample_indices,
        # Witness-specific deterministic adaptation of FreSh Algorithm 1: we
        # score the exact seeded initialization that will train, rather than
        # averaging stochastic initialization draws that will not be used.
        initialization_draws=1,
        init_scorer_forward_calls=len(ordered_candidates),
        init_scorer_pair_equivalents=len(ordered_candidates),
        distance_averaging_axis="target_pairs",
        spectrum_size=spectrum_n,
        baseline_candidate=ordered_candidates[0],
        ordered_candidates=ordered_candidates,
        targets=targets,
        uses_spectral_weight_maps=weight_maps is not None,
        spectral_weight_map_sha256s=(
            None if weight_maps is None else tuple(_canonical_array_sha256(weight) for weight in weight_maps)
        ),
        candidates=tuple(telemetry),
        selection=selection,
    )


def score_fresh_committed_state(
    labels: ArrayLike,
    selection_result: FreShRuntimeResult,
    *,
    spectral_weight_maps: Sequence[ArrayLike] | NDArray[np.floating[Any]] | None = None,
) -> FreShCommittedStateTelemetry:
    """Measure the final post-prefit label map against the frozen FreSh targets.

    Candidate selection happens before optional structured prefit.  This second
    measurement binds the actual state that epoch zero receives and exposes any
    spectral drift instead of assuming the selected candidate survived.
    """

    if not isinstance(selection_result, FreShRuntimeResult):
        raise ValueError("selection_result must be a FreShRuntimeResult")
    weight_maps = _as_optional_weight_map_sequence(
        spectral_weight_maps,
        selection_result.total_target_pairs,
    )
    if selection_result.uses_spectral_weight_maps != (weight_maps is not None):
        raise ValueError("spectral_weight_maps must match the weighted/unweighted FreSh selection")
    if weight_maps is not None:
        expected_hashes = selection_result.spectral_weight_map_sha256s
        if expected_hashes is None:
            raise ValueError("weighted FreSh selection is missing spectral weight hashes")
        expected_shape = selection_result.targets[0].shape
        weight_maps = tuple(
            _canonical_weight_map(
                weight,
                expected_shape=expected_shape,
                name=f"spectral_weight_maps[{index}]",
            )
            for index, weight in enumerate(weight_maps)
        )
        for index, (weight, expected_hash) in enumerate(zip(weight_maps, expected_hashes, strict=True)):
            if _canonical_array_sha256(weight) != expected_hash:
                raise ValueError(f"spectral_weight_maps[{index}] hash does not match selection")
    canonical_labels = _canonical_label_map(labels, name="committed FreSh state")
    expected_shape = selection_result.targets[0].shape
    if canonical_labels.shape != expected_shape:
        raise ValueError(
            f"committed FreSh state shape {canonical_labels.shape} does not match target shape {expected_shape}"
        )
    boundary = label_boundary_target_map(canonical_labels)
    boundary_pixels = int(np.count_nonzero(boundary))
    if boundary_pixels == 0:
        raise ValueError("committed FreSh state has zero boundary pixels")
    spectrum = fresh_spectrum(boundary, selection_result.spectrum_size)
    distances = tuple(
        _candidate_target_distance(
            boundary=boundary,
            global_spectrum=spectrum,
            target=target,
            spectral_weight=(None if weight_maps is None else weight_maps[target.pair_index]),
            spectrum_size=selection_result.spectrum_size,
        )
        for target in selection_result.targets
    )
    mean_distance = math.fsum(distance.residual_wasserstein1 for distance in distances) / len(distances)
    return FreShCommittedStateTelemetry(
        label_sha256=_canonical_array_sha256(canonical_labels),
        boundary_sha256=_canonical_array_sha256(boundary),
        boundary_pixels=boundary_pixels,
        spectrum=tuple(float(value) for value in spectrum),
        target_distances=distances,
        mean_distance=float(mean_distance),
    )


def fresh_init_scorer_accounting(
    selection_result: FreShRuntimeResult,
) -> FreShInitScorerAccounting:
    """Add the mandatory committed-state scorer forward to sweep accounting."""

    if not isinstance(selection_result, FreShRuntimeResult):
        raise ValueError("selection_result must be a FreShRuntimeResult")
    selection_forwards = _positive_int(
        selection_result.init_scorer_forward_calls,
        name="selection init_scorer_forward_calls",
    )
    selection_pairs = _positive_int(
        selection_result.init_scorer_pair_equivalents,
        name="selection init_scorer_pair_equivalents",
    )
    return FreShInitScorerAccounting(
        selection_scorer_forward_calls=selection_forwards,
        selection_scorer_pair_equivalents=selection_pairs,
        committed_state_scorer_forward_calls=1,
        committed_state_scorer_pair_equivalents=1,
        total_init_scorer_forward_calls=selection_forwards + 1,
        total_init_scorer_pair_equivalents=selection_pairs + 1,
    )


def write_fresh_receipt(
    path: str | Path,
    result: FreShRuntimeResult,
    *,
    provenance: Mapping[str, Any] | None = None,
) -> str:
    """Atomically write a durable canonical JSON receipt and return its SHA256.

    Receipt paths resolving below ``/tmp`` or ``/private/tmp`` are refused:
    an operator-facing evidence receipt must survive process cleanup.  The file
    and containing directory are fsynced around the same-directory replace.
    """

    if not isinstance(result, FreShRuntimeResult):
        raise ValueError("result must be a FreShRuntimeResult")
    payload = {
        "schema": "tac.witness_init.fresh_runtime.v1",
        "claim_scope": "init_time_spectral_selection_not_contest_score",
        "provenance": {} if provenance is None else dict(provenance),
        "result": asdict(result),
    }
    return _write_canonical_receipt(path, payload)


def write_fresh_committed_state_receipt(
    path: str | Path,
    telemetry: FreShCommittedStateTelemetry,
    *,
    selection_receipt_sha256: str,
    scorer_accounting: FreShInitScorerAccounting,
    provenance: Mapping[str, Any] | None = None,
) -> str:
    """Write the post-structured state measurement linked to selection bytes."""

    if not isinstance(telemetry, FreShCommittedStateTelemetry):
        raise ValueError("telemetry must be FreShCommittedStateTelemetry")
    if not isinstance(scorer_accounting, FreShInitScorerAccounting):
        raise ValueError("scorer_accounting must be FreShInitScorerAccounting")
    if (
        not isinstance(selection_receipt_sha256, str)
        or len(selection_receipt_sha256) != 64
        or any(character not in "0123456789abcdef" for character in selection_receipt_sha256.lower())
    ):
        raise ValueError("selection_receipt_sha256 must be 64 hexadecimal characters")
    payload = {
        "schema": "tac.witness_init.fresh_committed_state.v1",
        "claim_scope": "post_init_spectral_telemetry_not_contest_score",
        "selection_receipt_sha256": selection_receipt_sha256,
        "provenance": {} if provenance is None else dict(provenance),
        "result": {**asdict(telemetry), **asdict(scorer_accounting)},
    }
    return _write_canonical_receipt(path, payload)


def _write_canonical_receipt(path: str | Path, payload: Mapping[str, Any]) -> str:
    destination = Path(path).expanduser()
    _refuse_temporary_receipt_path(destination)
    if destination.exists() and destination.is_dir():
        raise ValueError(f"receipt path is a directory: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        encoded = (
            json.dumps(
                payload,
                allow_nan=False,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("receipt payload must contain only finite JSON values") from exc

    file_descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        directory_descriptor = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except BaseException:
        try:
            os.close(file_descriptor)
        except OSError:
            pass
        temporary.unlink(missing_ok=True)
        raise

    return hashlib.sha256(encoded).hexdigest()


def _as_target_map_sequence(
    target_label_maps: Sequence[ArrayLike] | NDArray[np.integer[Any]],
) -> tuple[ArrayLike, ...]:
    if isinstance(target_label_maps, np.ndarray):
        if target_label_maps.ndim == 2:
            return (target_label_maps,)
        if target_label_maps.ndim != 3:
            raise ValueError("target_label_maps ndarray must have shape (H, W) or (P, H, W)")
        if target_label_maps.shape[0] == 0:
            raise ValueError("target_label_maps must not be empty")
        return tuple(target_label_maps[index] for index in range(target_label_maps.shape[0]))
    try:
        maps = tuple(target_label_maps)
    except TypeError as exc:
        raise ValueError("target_label_maps must be a non-empty sequence") from exc
    if not maps:
        raise ValueError("target_label_maps must not be empty")
    return maps


def _validate_all_target_shapes(target_maps: Sequence[ArrayLike]) -> tuple[int, int]:
    expected_shape: tuple[int, int] | None = None
    for pair_index, raw_labels in enumerate(target_maps):
        labels = np.asarray(raw_labels)
        if labels.ndim != 2 or min(labels.shape) <= 0:
            raise ValueError(f"target pair {pair_index} must have positive shape (H, W), got {labels.shape}")
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError(f"target pair {pair_index} labels must have an integer dtype")
        shape = (int(labels.shape[0]), int(labels.shape[1]))
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise ValueError(f"target shapes differ: pair 0 has {expected_shape}, pair {pair_index} has {shape}")
    assert expected_shape is not None
    return expected_shape


def _target_spectrum_record(
    *,
    pair_index: int,
    labels: ArrayLike,
    spectral_weight: ArrayLike | None,
    expected_shape: tuple[int, int],
    spectrum_size: int,
) -> FreShTargetSpectrum:
    canonical_labels = _canonical_label_map(labels, name=f"target pair {pair_index}")
    if canonical_labels.shape != expected_shape:
        raise ValueError(
            f"target pair {pair_index} shape {canonical_labels.shape} does not match target shape {expected_shape}"
        )
    boundary = label_boundary_target_map(canonical_labels)
    boundary_pixels = int(np.count_nonzero(boundary))
    if boundary_pixels == 0:
        raise ValueError(f"target pair {pair_index} has zero boundary pixels")
    try:
        spectrum = fresh_spectrum(boundary, spectrum_size)
    except ValueError as exc:
        if str(exc).startswith("FreSh non-DC spectrum has zero"):
            raise ValueError(f"target pair {pair_index} has zero non-DC boundary spectrum") from exc
        raise
    weight = (
        None
        if spectral_weight is None
        else _canonical_weight_map(
            spectral_weight,
            expected_shape=expected_shape,
            name=f"spectral_weight_maps[{pair_index}]",
        )
    )
    residual_spectrum = spectrum if weight is None else fresh_spectrum(boundary * weight, spectrum_size)
    return FreShTargetSpectrum(
        pair_index=pair_index,
        shape=expected_shape,
        label_sha256=_canonical_array_sha256(canonical_labels),
        boundary_sha256=_canonical_array_sha256(boundary),
        boundary_pixels=boundary_pixels,
        spectrum=tuple(float(value) for value in spectrum),
        spectral_weight_sha256=None if weight is None else _canonical_array_sha256(weight),
        residual_spectrum=tuple(float(value) for value in residual_spectrum),
    )


def _candidate_target_distance(
    *,
    boundary: NDArray[np.bool_],
    global_spectrum: ArrayLike,
    target: FreShTargetSpectrum,
    spectral_weight: ArrayLike | None,
    spectrum_size: int,
) -> FreShTargetDistance:
    residual_spectrum = global_spectrum
    if spectral_weight is not None:
        weight = _canonical_weight_map(
            spectral_weight,
            expected_shape=target.shape,
            name=f"spectral_weight_maps[{target.pair_index}]",
        )
        residual_spectrum = fresh_spectrum(boundary * weight, spectrum_size)
    residual_wasserstein1 = wasserstein1_cdf_l1(residual_spectrum, target.residual_spectrum)
    return FreShTargetDistance(
        pair_index=target.pair_index,
        residual_wasserstein1=residual_wasserstein1,
        # Preserve the original field as the selection distance.
        wasserstein1=residual_wasserstein1,
        global_boundary_wasserstein1=wasserstein1_cdf_l1(global_spectrum, target.spectrum),
    )


def _as_optional_weight_map_sequence(
    spectral_weight_maps: Sequence[ArrayLike] | NDArray[np.floating[Any]] | None,
    target_count: int,
) -> tuple[ArrayLike, ...] | None:
    if spectral_weight_maps is None:
        return None
    maps = _as_target_map_sequence(spectral_weight_maps)
    if len(maps) != target_count:
        raise ValueError(f"spectral_weight_maps supplies {len(maps)} maps for {target_count} targets")
    return maps


def _canonical_weight_map(
    weights: ArrayLike,
    *,
    expected_shape: tuple[int, int],
    name: str,
) -> NDArray[np.float64]:
    array = np.asarray(weights, dtype=np.float64)
    if array.shape != expected_shape:
        raise ValueError(f"{name} shape {array.shape} does not match target shape {expected_shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    if np.any(array < 0.0):
        raise ValueError(f"{name} must be non-negative")
    if not np.any(array > 0.0):
        raise ValueError(f"{name} must contain positive mass")
    return np.ascontiguousarray(array)


def _canonical_label_map(labels: ArrayLike, *, name: str) -> NDArray[np.int64]:
    array = np.asarray(labels)
    if array.ndim != 2 or min(array.shape) <= 0:
        raise ValueError(f"{name} must have positive shape (H, W), got {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name} must have an integer dtype")
    return np.ascontiguousarray(array, dtype=np.int64)


def _canonical_array_sha256(array: NDArray[Any]) -> str:
    contiguous = np.ascontiguousarray(array)
    dtype = contiguous.dtype
    if dtype.itemsize > 1:
        little_endian_dtype = dtype.newbyteorder("<")
        contiguous = contiguous.astype(little_endian_dtype, copy=False)
        dtype = little_endian_dtype
    header = json.dumps(
        {"dtype": dtype.str, "shape": list(contiguous.shape)},
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\x00")
    digest.update(contiguous.tobytes(order="C"))
    return digest.hexdigest()


def _eligible_mean(record: FreShCandidateTelemetry) -> float:
    if record.status != "eligible" or record.mean_distance is None:
        raise AssertionError("internal error: selected FreSh record is not eligible")
    return record.mean_distance


def _positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _finite_float(value: float, *, name: str, strictly_positive: bool) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not np.isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    if strictly_positive and number <= 0.0:
        raise ValueError(f"{name} must be strictly positive")
    if not strictly_positive and number < 0.0:
        raise ValueError(f"{name} must be non-negative")
    return number


def _validated_axis(
    values: Sequence[float],
    *,
    name: str,
    strictly_positive: bool,
) -> tuple[float, ...]:
    try:
        raw_values = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be a non-empty sequence") from exc
    if not raw_values:
        raise ValueError(f"{name} must not be empty")
    unique: list[float] = []
    for index, value in enumerate(raw_values):
        normalized = _finite_float(
            value,
            name=f"{name}[{index}]",
            strictly_positive=strictly_positive,
        )
        if normalized not in unique:
            unique.append(normalized)
    return tuple(unique)


def _refuse_temporary_receipt_path(path: Path) -> None:
    if not path.name:
        raise ValueError("receipt path must name a file")
    resolved = path.resolve(strict=False)
    temporary_roots = {
        Path("/tmp").resolve(strict=False),
        Path("/private/tmp").resolve(strict=False),
    }
    if any(resolved == root or resolved.is_relative_to(root) for root in temporary_roots):
        raise ValueError("receipt path must be durable and cannot be under /tmp")
