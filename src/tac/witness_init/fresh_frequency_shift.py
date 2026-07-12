"""Pure-NumPy FreSh spectrum selection for witness-INR initialization.

FreSh (Kania et al., arXiv:2410.05050) selects the candidate whose untrained
output spectrum is nearest to the target spectrum under 1-D Wasserstein-1.
This module deliberately contains no framework or trainer dependencies so the
same selection is reproducible for NumPy, MLX, and Torch initializers.

The repository's witness use omits the DC term: it is representable by an
output bias and contains no directional boundary detail.  The remaining bins
are the anti-diagonals ``i + j = d`` of the unshifted 2-D DFT magnitude, with
``d = 1, ..., spectrum_size``.  Channel magnitudes are accumulated before the
single L1 normalization required by FreSh.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]
Float32Array = NDArray[np.float32]

# Match the already-shipped FINER first-layer stream.  FreSh may select k=0,
# which FINER itself rejects, so this module owns the all-zero row explicitly.
FINER_BIAS_RNG_SALT = 20260707


@dataclass(frozen=True)
class FreShSelection:
    """Stable result of a FreSh candidate sweep.

    ``candidate`` is the mapping key supplied by the caller.  Ties retain the
    first mapping entry, matching the paper's strict-improvement pseudocode.
    """

    candidate: Hashable
    index: int
    mean_distance: float
    distances: tuple[float, ...]


def _as_finite_signal(signal: ArrayLike, *, name: str) -> FloatArray:
    """Return a finite ``(channels, height, width)`` signal or fail closed."""

    array = np.asarray(signal, dtype=np.float64)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim != 3:
        raise ValueError(f"{name} must have shape (H, W) or (C, H, W), got {array.shape}")
    if array.shape[0] <= 0 or array.shape[1] <= 0 or array.shape[2] <= 0:
        raise ValueError(f"{name} must have positive channel and spatial dimensions, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def _as_nonempty_samples(
    signals: ArrayLike | Sequence[ArrayLike], *, name: str
) -> tuple[ArrayLike, ...]:
    """Treat one ndarray as one sample, and reject an empty sample collection."""

    if isinstance(signals, np.ndarray):
        return (signals,)
    try:
        samples = tuple(signals)
    except TypeError as exc:
        raise ValueError(f"{name} must be an array or a non-empty sequence of arrays") from exc
    if not samples:
        raise ValueError(f"{name} must not be empty")
    return samples


def _as_probability_spectrum(spectrum: ArrayLike, *, name: str) -> FloatArray:
    """Return a finite, non-negative, normalized one-dimensional spectrum."""

    array = np.asarray(spectrum, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    if np.any(array < 0.0):
        raise ValueError(f"{name} must be non-negative")
    mass = float(array.sum(dtype=np.float64))
    if mass <= 0.0:
        raise ValueError(f"{name} must have strictly positive mass")
    return array / mass


def _as_nonempty_spectra(
    spectra: ArrayLike | Sequence[ArrayLike], *, name: str
) -> tuple[ArrayLike, ...]:
    """Normalize ``(n,)`` or ``(samples,n)`` input to a spectrum tuple."""

    try:
        array = np.asarray(spectra, dtype=np.float64)
    except (TypeError, ValueError):
        array = np.asarray((), dtype=np.float64)
    if array.ndim == 1 and array.size > 0:
        return (array,)
    if array.ndim == 2 and array.shape[0] > 0 and array.shape[1] > 0:
        return tuple(array[index] for index in range(array.shape[0]))
    raise ValueError(f"{name} must have shape (n,) or (samples, n) with positive dimensions")


def fresh_spectrum(signal: ArrayLike, spectrum_size: int) -> FloatArray:
    """Return FreSh's DC-omitted, L1-normalized anti-diagonal spectrum.

    Each channel is transformed independently with ``fft2``; magnitudes, not
    complex coefficients or powers, are accumulated.  The first returned bin
    corresponds to degree one, so an output of length ``n`` contains degrees
    one through ``n``.  A zero-mass spectrum is invalid because it cannot be a
    probability distribution for the Wasserstein comparison.
    """

    if isinstance(spectrum_size, bool) or not isinstance(spectrum_size, (int, np.integer)):
        raise ValueError("spectrum_size must be a positive integer")
    if spectrum_size <= 0:
        raise ValueError("spectrum_size must be a positive integer")

    array = _as_finite_signal(signal, name="signal")
    _, height, width = array.shape
    max_non_dc_degree = height + width - 2
    if spectrum_size > max_non_dc_degree:
        raise ValueError(
            "spectrum_size exceeds available non-DC anti-diagonals: "
            f"requested {spectrum_size}, maximum {max_non_dc_degree}"
        )

    magnitude = np.abs(np.fft.fft2(array, axes=(-2, -1)))
    degrees = np.add.outer(np.arange(height), np.arange(width))
    spectrum = np.empty(spectrum_size, dtype=np.float64)
    for offset, degree in enumerate(range(1, spectrum_size + 1)):
        spectrum[offset] = float(magnitude[:, degrees == degree].sum(dtype=np.float64))

    mass = float(spectrum.sum(dtype=np.float64))
    if not np.isfinite(mass) or mass <= 0.0:
        raise ValueError("FreSh non-DC spectrum has zero or non-finite mass")
    return spectrum / mass


def wasserstein1_cdf_l1(left: ArrayLike, right: ArrayLike) -> float:
    """Compute exact discrete 1-D Wasserstein-1 as CDF-L1.

    Inputs may be unnormalized non-negative spectra; both are normalized here
    so callers cannot accidentally compare scale rather than frequency shape.
    """

    lhs = _as_probability_spectrum(left, name="left spectrum")
    rhs = _as_probability_spectrum(right, name="right spectrum")
    if lhs.shape != rhs.shape:
        raise ValueError("spectra must be non-empty one-dimensional arrays of equal shape")
    return float(np.abs(np.cumsum(lhs) - np.cumsum(rhs)).sum(dtype=np.float64))


def select_fresh_spectra(
    target_spectra: ArrayLike | Sequence[ArrayLike],
    candidate_spectra: Mapping[Hashable, ArrayLike | Sequence[ArrayLike]],
) -> FreShSelection:
    """Select a candidate from already-materialized spectra.

    This is the bounded-memory trainer API: initialized renders can be reduced
    immediately to their usual 64 bins instead of retaining image-sized arrays
    for the whole frequency/bias sweep.  One candidate spectrum may be reused
    against every target sample, which is exact when all per-pair codes are zero
    and the cold witness therefore has one shared initialized output.
    """

    targets = _as_nonempty_spectra(target_spectra, name="target_spectra")
    normalized_targets = tuple(
        _as_probability_spectrum(sample, name=f"target spectrum {index}")
        for index, sample in enumerate(targets)
    )
    if not candidate_spectra:
        raise ValueError("candidate_spectra must not be empty")

    best: FreShSelection | None = None
    for index, (candidate, raw_spectra) in enumerate(candidate_spectra.items()):
        spectra = _as_nonempty_spectra(raw_spectra, name=f"candidate {candidate!r} spectra")
        if len(spectra) == 1 and len(normalized_targets) > 1:
            spectra = spectra * len(normalized_targets)
        if len(spectra) != len(normalized_targets):
            raise ValueError(
                f"candidate {candidate!r} supplies {len(spectra)} spectra for "
                f"{len(normalized_targets)} target spectra"
            )
        normalized_candidates = tuple(
            _as_probability_spectrum(sample, name=f"candidate {candidate!r} spectrum {sample_index}")
            for sample_index, sample in enumerate(spectra)
        )
        distances = tuple(
            wasserstein1_cdf_l1(model, target)
            for model, target in zip(normalized_candidates, normalized_targets, strict=True)
        )
        mean_distance = float(np.mean(distances, dtype=np.float64))
        proposed = FreShSelection(candidate, index, mean_distance, distances)
        if best is None or proposed.mean_distance < best.mean_distance:
            best = proposed

    assert best is not None  # candidate_spectra was checked above.
    return best


def select_fresh_configuration(
    target_samples: ArrayLike | Sequence[ArrayLike],
    candidate_outputs: Mapping[Hashable, ArrayLike | Sequence[ArrayLike]],
    spectrum_size: int,
) -> FreShSelection:
    """Select the initial-output candidate minimizing mean FreSh distance.

    A single candidate output is reused for every target sample.  Otherwise a
    candidate must supply exactly one initialized output per target sample.  A
    candidate mapping is intentionally ordered: on an exact tie, its earliest
    entry wins deterministically, as in Algorithm 1 of the FreSh paper.
    """

    targets = _as_nonempty_samples(target_samples, name="target_samples")
    if not candidate_outputs:
        raise ValueError("candidate_outputs must not be empty")
    target_arrays = tuple(_as_finite_signal(sample, name=f"target sample {index}") for index, sample in enumerate(targets))
    target_spectra = tuple(fresh_spectrum(sample, spectrum_size) for sample in target_arrays)
    candidate_spectra: dict[Hashable, tuple[FloatArray, ...]] = {}
    for candidate, raw_outputs in candidate_outputs.items():
        outputs = _as_nonempty_samples(raw_outputs, name=f"candidate {candidate!r} outputs")
        if len(outputs) == 1 and len(targets) > 1:
            outputs = outputs * len(targets)
        if len(outputs) != len(targets):
            raise ValueError(
                f"candidate {candidate!r} supplies {len(outputs)} outputs for {len(targets)} target samples"
            )
        output_arrays = tuple(
            _as_finite_signal(output, name=f"candidate {candidate!r} output {sample_index}")
            for sample_index, output in enumerate(outputs)
        )
        for sample_index, (output, target) in enumerate(zip(output_arrays, target_arrays, strict=True)):
            if output.shape != target.shape:
                raise ValueError(
                    f"candidate {candidate!r} output {sample_index} shape {output.shape} "
                    f"does not match target shape {target.shape}"
                )
        candidate_spectra[candidate] = tuple(
            fresh_spectrum(output, spectrum_size) for output in output_arrays
        )
    return select_fresh_spectra(target_spectra, candidate_spectra)


def label_boundary_target_map(labels: ArrayLike) -> NDArray[np.bool_]:
    """Return the 4-connected class-transition map for ``(..., H, W)`` labels."""

    array = np.asarray(labels)
    if array.ndim < 2 or array.shape[-2] <= 0 or array.shape[-1] <= 0:
        raise ValueError(f"labels must have shape (..., H, W) with positive spatial dimensions, got {array.shape}")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError("labels must have an integer dtype")

    boundary = np.zeros(array.shape, dtype=bool)
    horizontal = array[..., :, 1:] != array[..., :, :-1]
    vertical = array[..., 1:, :] != array[..., :-1, :]
    boundary[..., :, 1:] |= horizontal
    boundary[..., :, :-1] |= horizontal
    boundary[..., 1:, :] |= vertical
    boundary[..., :-1, :] |= vertical
    return boundary


def derived_tangent_frequency_multipliers(tangent_deficit: float = 3.2) -> tuple[float, float, float]:
    """Return ``(1, sqrt(deficit), deficit)`` for a measured tangent deficit."""

    deficit = float(tangent_deficit)
    if not np.isfinite(deficit) or deficit <= 0.0:
        raise ValueError("tangent_deficit must be finite and strictly positive")
    return (1.0, float(np.sqrt(deficit)), deficit)


def tangent_frequency_candidates(
    current_frequency: float,
    *,
    reference_frequency: float = 8.0,
    tangent_deficit: float = 3.2,
) -> tuple[float, ...]:
    """Return the stable FreSh bank sweep for the measured tangent deficit.

    The current setting is deliberately first, so an exact spectral tie keeps
    the baseline.  The remaining candidates are the reference frequency times
    ``(1, sqrt(deficit), deficit)``.  Stable de-duplication prevents a repeated
    baseline from changing tie behavior.
    """

    current = float(current_frequency)
    reference = float(reference_frequency)
    if not np.isfinite(current) or current <= 0.0:
        raise ValueError("current_frequency must be finite and strictly positive")
    if not np.isfinite(reference) or reference <= 0.0:
        raise ValueError("reference_frequency must be finite and strictly positive")
    raw = (
        current,
        *(reference * multiplier for multiplier in derived_tangent_frequency_multipliers(tangent_deficit)),
    )
    unique: list[float] = []
    for candidate in raw:
        if not any(np.isclose(candidate, prior, rtol=0.0, atol=1e-12) for prior in unique):
            unique.append(float(candidate))
    return tuple(unique)


def inclusive_bias_width_grid(
    minimum: float = 0.0,
    maximum: float = 3.0,
    step: float = 0.1,
) -> tuple[float, ...]:
    """Return a decimal-stable inclusive FreSh/FINER bias-width sweep."""

    lo, hi, delta = float(minimum), float(maximum), float(step)
    if not all(np.isfinite(value) for value in (lo, hi, delta)):
        raise ValueError("bias grid bounds and step must be finite")
    if lo < 0.0 or hi < lo or delta <= 0.0:
        raise ValueError("bias grid requires 0 <= minimum <= maximum and step > 0")
    count_float = (hi - lo) / delta
    count = round(count_float)
    if not np.isclose(count_float, count, rtol=0.0, atol=1e-9):
        raise ValueError("bias grid step must divide maximum - minimum exactly")
    return tuple(float(lo + index * delta) for index in range(count + 1))


def deterministic_first_layer_bias_candidates(
    bias_widths: Sequence[float],
    width: int,
    *,
    seed: int,
    salt: int = FINER_BIAS_RNG_SALT,
) -> Float32Array:
    """Generate aligned ``U(-k,k)`` first-layer bias candidates.

    Every row scales the *same* standardized random vector, so ``k`` is the
    only changing factor in the sweep.  For positive ``k`` this is exactly the
    existing trainer's ``_finer_bias_init_values(seed, k, width)`` stream;
    ``k=0`` is the SIREN zero-bias baseline.  No process-global RNG is touched.
    """

    if isinstance(width, bool) or not isinstance(width, (int, np.integer)) or width <= 0:
        raise ValueError("width must be a positive integer")
    widths = np.asarray(tuple(bias_widths), dtype=np.float64)
    if widths.ndim != 1 or widths.size == 0:
        raise ValueError("bias_widths must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(widths)) or np.any(widths < 0.0):
        raise ValueError("bias_widths must be finite and non-negative")
    rng = np.random.default_rng(int(seed) + int(salt))
    standardized = rng.uniform(-1.0, 1.0, size=int(width))
    return (widths[:, None] * standardized[None, :]).astype(np.float32, copy=False)
