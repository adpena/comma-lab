# SPDX-License-Identifier: MIT
"""Canonical equations for FreSh frequency-shift witness initialization.

FreSh (Kania et al., arXiv:2410.05050) compares the spectrum of a target
signal with spectra rendered by *untrained* INR configurations.  It omits the
DC coefficient, sums 2-D DFT magnitudes on anti-diagonals ``i + j = d``, and
selects the configuration with minimum discrete one-dimensional
Wasserstein-1 distance.  This module makes those equations executable without
MLX, Torch, trainer, or filesystem state.

The witness adaptation is deliberately narrow.  The measured 3.2x
along-tangent deficit defines a geometric candidate ladder; FreSh still
selects among the candidates.  It does not assert that the ladder endpoint is
optimal.  In particular, the already-refuted bounded-warm-start along=26
formulation and the still-open from-scratch initialization formulation have
separate machine-readable verdict scopes below.

This is a non-promotable initialization law.  Only ``upstream/evaluate.py``
on exact archive bytes can support a score claim, and the fixed-quality
epochs comparison remains an empirical obligation.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    VERIFIED_VIA_SOURCE_INSPECTION,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import (
    build_provenance_for_predicted,
    build_provenance_for_research_sidecar,
)

EQUATION_ID = "fresh_frequency_shift_init_v1"
FRESH_PAPER_ARXIV = "2410.05050"
DEFAULT_SPECTRUM_SIZE = 64

# Value-provenance ladder: these are measured inputs, not guessed defaults.
MEASURED_REFERENCE_TANGENT_FREQUENCY = 8.0
MEASURED_ALONG_TANGENT_DEFICIT = 3.2
MEASURED_SCORER_EPOCH_FRACTION = 0.95
MEASURED_GROUPED_BACKWARD_SPEEDUP = 16.9

# FreSh/FINER candidate sweep used by the initialization lever.  A trainer
# draws one standardized bias vector and scales that same vector by k, so k
# is the only bias candidate confound.
FRESH_BIAS_K_MIN = 0.0
FRESH_BIAS_K_MAX = 3.0
FRESH_BIAS_K_STEP = 0.1

_UTC = "2026-07-12T00:00:00Z"
_SURVEY = ".omx/research/fast_witness_training_oss_survey_20260712.md"
_DAG = ".omx/research/sub015_DAG_topaiml_reopen_and_pursuit_plan_20260611.md"
_WARM_START_VERDICT = ".omx/research/owed16v2_verdict_20260710.json"


@dataclass(frozen=True)
class FormulationVerdictScope:
    """Machine-readable verdict scope; one failed formulation is not a family verdict."""

    level: str
    formulation: str
    status: str
    authority: str
    evidence: str

    def to_dict(self) -> dict[str, str]:
        """Return a JSON-safe representation for canonical-equation metadata."""

        return asdict(self)


SETTLED_WARM_START_ALONG26_SCOPE = FormulationVerdictScope(
    level="formulation",
    formulation=(
        "bounded_warm_start_fine_tune_ep650_to_700_from_mod32cap_ep650_best; "
        "self_orient_ON_freq_along_26_vs_OFF; seed_0"
    ),
    status=(
        "REFUTED: along-heavy allocation did not beat self-orient OFF; do not reopen this "
        "bounded-warm-start formulation"
    ),
    authority="n600 through-R frozen CPU-torch SegNet; [macOS-CPU advisory], non-promotable",
    evidence=_WARM_START_VERDICT,
)

OPEN_FRESH_FROM_SCRATCH_SCOPE = FormulationVerdictScope(
    level="formulation",
    formulation=(
        "from_scratch_cold_init_FreSh_spectral_selection_before_partition_formation; "
        "matched baseline-vs-FreSh fixed-d_seg comparison"
    ),
    status="OPEN/OWED: initialization-time spectral selection has not been adjudicated by the warm-start arm",
    authority="faithful-slice measurement is advisory; governed n600 exact validation remains owed",
    evidence=_SURVEY,
)


@dataclass(frozen=True)
class FreshSpectrumSelection:
    """Deterministic result of a FreSh spectrum-candidate comparison."""

    index: int
    mean_wasserstein1: float
    per_target_wasserstein1: tuple[float, ...]


@dataclass(frozen=True)
class FixedQualityReduction:
    """Exact fixed-quality epoch/scorer-call/wall-clock accounting identity."""

    baseline_epochs: int
    initialized_epochs: int
    pairs_per_epoch: int
    baseline_scorer_calls: int
    initialized_scorer_calls: int
    baseline_init_scorer_calls: int
    initialized_init_scorer_calls: int
    baseline_total_scorer_calls: int
    initialized_total_scorer_calls: int
    epoch_reduction_fraction: float
    scorer_call_reduction_fraction: float
    total_scorer_call_reduction_fraction: float
    baseline_wall_seconds: float
    initialized_wall_seconds: float
    wall_clock_reduction_fraction: float
    scorer_seconds_saved: float


def _positive_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _finite_float(value: float, *, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def fresh_antidiagonal_spectrum(
    signal: ArrayLike,
    spectrum_size: int = DEFAULT_SPECTRUM_SIZE,
    *,
    normalize: bool = True,
) -> NDArray[np.float64]:
    r"""Compute FreSh's exact DC-omitted anti-diagonal spectrum.

    For a signal ``A`` with channel-first shape ``(C, H, W)`` (or a single
    ``(H, W)`` channel), the raw retained bins are

    ``S(A,d) = sum_c sum_{i+j=d} |FFT2(A_c)[i,j]|``, for ``d=1,...,n``.

    The unshifted 2-D DFT is intentional: bin zero is the omitted DC term.
    When ``normalize=True``, the result is divided by its retained L1 mass so
    it is a probability mass function for Wasserstein-1.  A constant signal
    therefore has a valid raw all-zero spectrum but no normalized spectrum.
    """

    n = _positive_integer(spectrum_size, name="spectrum_size")
    array = np.asarray(signal, dtype=np.float64)
    if array.ndim == 2:
        array = array[np.newaxis, ...]
    if array.ndim != 3:
        raise ValueError(f"signal must have shape (H, W) or channel-first (C, H, W), got {array.shape}")
    if any(dimension <= 0 for dimension in array.shape):
        raise ValueError(f"signal dimensions must all be positive, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("signal contains NaN or Inf")

    _, height, width = array.shape
    max_degree = height + width - 2
    if n > max_degree:
        raise ValueError(
            "spectrum_size exceeds available non-DC anti-diagonals: "
            f"requested {n}, maximum {max_degree}"
        )

    magnitude = np.abs(np.fft.fft2(array, axes=(-2, -1)))
    degrees = np.add.outer(np.arange(height), np.arange(width))
    spectrum = np.empty(n, dtype=np.float64)
    for output_index, degree in enumerate(range(1, n + 1)):
        spectrum[output_index] = float(
            magnitude[:, degrees == degree].sum(dtype=np.float64)
        )

    if not normalize:
        return spectrum
    mass = float(spectrum.sum(dtype=np.float64))
    if not math.isfinite(mass) or mass <= 0.0:
        raise ValueError("normalized FreSh spectrum requires strictly positive non-DC mass")
    return spectrum / mass


def fresh_wasserstein1_cdf_l1(left: ArrayLike, right: ArrayLike) -> float:
    r"""Return exact unit-bin discrete W1 as L1 distance between CDFs.

    Inputs may be unnormalized non-negative spectra.  Each is normalized to a
    probability mass function before computing
    ``sum_d |sum_{k<=d} (p_k - q_k)|``.
    """

    lhs = np.asarray(left, dtype=np.float64)
    rhs = np.asarray(right, dtype=np.float64)
    if lhs.ndim != 1 or rhs.ndim != 1 or lhs.size == 0 or lhs.shape != rhs.shape:
        raise ValueError("spectra must be non-empty one-dimensional arrays of equal shape")
    if not np.all(np.isfinite(lhs)) or not np.all(np.isfinite(rhs)):
        raise ValueError("spectra contain NaN or Inf")
    if np.any(lhs < 0.0) or np.any(rhs < 0.0):
        raise ValueError("spectra must be non-negative")
    lhs_mass = float(lhs.sum(dtype=np.float64))
    rhs_mass = float(rhs.sum(dtype=np.float64))
    if lhs_mass <= 0.0 or rhs_mass <= 0.0:
        raise ValueError("spectra must each have strictly positive mass")
    cdf_delta = np.cumsum(lhs / lhs_mass) - np.cumsum(rhs / rhs_mass)
    return float(np.abs(cdf_delta).sum(dtype=np.float64))


def select_fresh_spectrum_candidate(
    target_spectra: ArrayLike,
    candidate_spectra: ArrayLike,
) -> FreshSpectrumSelection:
    r"""Select ``argmin_j mean_t W1(target_t, candidate_j)`` deterministically.

    ``target_spectra`` is ``(T, n)`` or ``(n,)`` and ``candidate_spectra`` is
    ``(J, n)`` or ``(n,)``.  Exact ties retain the lowest candidate index,
    matching strict-improvement pseudocode and avoiding platform-dependent
    tie resolution.
    """

    targets = np.asarray(target_spectra, dtype=np.float64)
    candidates = np.asarray(candidate_spectra, dtype=np.float64)
    if targets.ndim == 1:
        targets = targets[np.newaxis, ...]
    if candidates.ndim == 1:
        candidates = candidates[np.newaxis, ...]
    if targets.ndim != 2 or candidates.ndim != 2:
        raise ValueError("target_spectra and candidate_spectra must be one- or two-dimensional")
    if targets.shape[0] == 0 or candidates.shape[0] == 0 or targets.shape[1] == 0:
        raise ValueError("target and candidate spectrum collections must be non-empty")
    if targets.shape[1] != candidates.shape[1]:
        raise ValueError("target and candidate spectra must use the same number of bins")

    best_index = -1
    best_mean = math.inf
    best_distances: tuple[float, ...] = ()
    for candidate_index, candidate in enumerate(candidates):
        distances = tuple(
            fresh_wasserstein1_cdf_l1(target, candidate) for target in targets
        )
        mean_distance = float(np.mean(distances, dtype=np.float64))
        if mean_distance < best_mean:
            best_index = candidate_index
            best_mean = mean_distance
            best_distances = distances

    return FreshSpectrumSelection(
        index=best_index,
        mean_wasserstein1=best_mean,
        per_target_wasserstein1=best_distances,
    )


def tangent_frequency_candidates(
    reference_frequency: float = MEASURED_REFERENCE_TANGENT_FREQUENCY,
    deficit_factor: float = MEASURED_ALONG_TANGENT_DEFICIT,
) -> tuple[float, float, float]:
    r"""Return the geometric baseline/midpoint/deficit-closing candidate law.

    ``f_j = f_ref * delta**(j/2)``, ``j in {0,1,2}``.  With measured inputs
    this is ``(8, 8*sqrt(3.2), 8*3.2)``.  The endpoint is a candidate, not an
    optimum claim; FreSh's W1 selector may retain the baseline or midpoint.
    """

    reference = _finite_float(reference_frequency, name="reference_frequency")
    deficit = _finite_float(deficit_factor, name="deficit_factor")
    if reference <= 0.0:
        raise ValueError("reference_frequency must be strictly positive")
    if deficit < 1.0:
        raise ValueError("deficit_factor must be at least one")
    return (reference, reference * math.sqrt(deficit), reference * deficit)


def fresh_bias_scale_candidates() -> tuple[float, ...]:
    """Return the exact inclusive ``k=0.0,...,3.0`` grid in 0.1 steps."""

    steps = round((FRESH_BIAS_K_MAX - FRESH_BIAS_K_MIN) / FRESH_BIAS_K_STEP)
    # Integer division avoids representation drift such as ``0.1 * 3``
    # serializing as ``0.30000000000000004`` in provenance manifests.
    return tuple(index / 10.0 for index in range(steps + 1))


def fixed_quality_reduction_identity(
    baseline_epochs: int,
    initialized_epochs: int,
    *,
    pairs_per_epoch: int = 600,
    epoch_seconds: float = 1.0,
    baseline_init_seconds: float = 0.0,
    initialized_init_seconds: float = 0.0,
    baseline_init_scorer_calls: int = 0,
    initialized_init_scorer_calls: int = 0,
    scorer_epoch_fraction: float = MEASURED_SCORER_EPOCH_FRACTION,
) -> FixedQualityReduction:
    r"""Evaluate the fixed-quality epochs/scorer-call/wall-clock identity.

    For ``P`` pairs per epoch, ``N_train(E)=P*E`` and
    ``N_total=N_init+N_train``.  With constant per-epoch
    cost ``C``, ``T(E)=T_init+E*C``.  Thus, when one-time init overheads are
    equal, the epoch reduction, scorer-call reduction, and wall-clock
    reduction are identical.  Unequal init overhead is retained explicitly
    rather than silently treated as zero.  ``scorer_epoch_fraction`` only
    decomposes the saved seconds; it does not change the total-time identity.
    """

    baseline = _positive_integer(baseline_epochs, name="baseline_epochs")
    initialized = _nonnegative_integer(initialized_epochs, name="initialized_epochs")
    pairs = _positive_integer(pairs_per_epoch, name="pairs_per_epoch")
    seconds = _finite_float(epoch_seconds, name="epoch_seconds")
    baseline_init = _finite_float(baseline_init_seconds, name="baseline_init_seconds")
    initialized_init = _finite_float(
        initialized_init_seconds,
        name="initialized_init_seconds",
    )
    baseline_init_calls = _nonnegative_integer(
        baseline_init_scorer_calls,
        name="baseline_init_scorer_calls",
    )
    initialized_init_calls = _nonnegative_integer(
        initialized_init_scorer_calls,
        name="initialized_init_scorer_calls",
    )
    scorer_fraction = _finite_float(scorer_epoch_fraction, name="scorer_epoch_fraction")
    if seconds <= 0.0:
        raise ValueError("epoch_seconds must be strictly positive")
    if baseline_init < 0.0 or initialized_init < 0.0:
        raise ValueError("initialization times must be non-negative")
    if not 0.0 <= scorer_fraction <= 1.0:
        raise ValueError("scorer_epoch_fraction must be in [0, 1]")

    baseline_calls = baseline * pairs
    initialized_calls = initialized * pairs
    baseline_total_calls = baseline_init_calls + baseline_calls
    initialized_total_calls = initialized_init_calls + initialized_calls
    epoch_reduction = 1.0 - initialized / baseline
    baseline_wall = baseline_init + baseline * seconds
    initialized_wall = initialized_init + initialized * seconds
    return FixedQualityReduction(
        baseline_epochs=baseline,
        initialized_epochs=initialized,
        pairs_per_epoch=pairs,
        baseline_scorer_calls=baseline_calls,
        initialized_scorer_calls=initialized_calls,
        baseline_init_scorer_calls=baseline_init_calls,
        initialized_init_scorer_calls=initialized_init_calls,
        baseline_total_scorer_calls=baseline_total_calls,
        initialized_total_scorer_calls=initialized_total_calls,
        epoch_reduction_fraction=epoch_reduction,
        scorer_call_reduction_fraction=1.0 - initialized_calls / baseline_calls,
        total_scorer_call_reduction_fraction=(
            1.0 - initialized_total_calls / baseline_total_calls
        ),
        baseline_wall_seconds=baseline_wall,
        initialized_wall_seconds=initialized_wall,
        wall_clock_reduction_fraction=1.0 - initialized_wall / baseline_wall,
        scorer_seconds_saved=(baseline - initialized) * seconds * scorer_fraction,
    )


_LAW = (
    "S(A,d)=sum_c sum_{i+j=d}|FFT2(A_c)[i,j]|, d=1..n (DC omitted); "
    "W1(p,q)=sum_d|CDF_p(d)-CDF_q(d)|; "
    "j*=argmin_j mean_t W1(S(omega_lane_t*boundary(target_t)),"
    "S(omega_lane_t*boundary(init_j))); global boundary W1 is diagnostic; "
    "f_parallel_j=f_ref*delta^(j/2), j=0,1,2; "
    "N_total=N_init+P*E; T=T_init+E*C_epoch"
)
_LAW_SHA256 = hashlib.sha256(_LAW.encode("utf-8")).hexdigest()


def build_fresh_frequency_shift_init_v1() -> CanonicalEquation:
    """Build the source-grounded, non-promotable FreSh initialization law."""

    anchor_formula = EmpiricalAnchor(
        anchor_id="fresh_source_formula_arxiv2410_05050_20260712",
        measurement_utc=_UTC,
        inputs={
            "paper": f"FreSh arXiv:{FRESH_PAPER_ARXIV}",
            "spectrum_size_default": DEFAULT_SPECTRUM_SIZE,
        },
        predicted_output={
            "spectrum": "DC-omitted anti-diagonal DFT-magnitude bins",
            "distance": "unit-bin 1-D Wasserstein-1 by CDF-L1",
        },
        empirical_output={
            "source_inspection": "equations and strict-minimum selection reproduced in executable form",
            "score_claim": "none",
        },
        residual=0.0,
        source_artifact=_SURVEY,
        measurement_method="primary-paper equation and released-algorithm source inspection",
        empirical_verification_status=VERIFIED_VIA_SOURCE_INSPECTION,
        provenance=build_provenance_for_research_sidecar(
            _SURVEY,
            reactivation_criteria=(
                "measure epochs-to-fixed-d_seg on a matched faithful slice, then validate n600 "
                "through the governed launch path"
            ),
            measurement_axis="[external paper claim]",
            hardware_substrate="not_applicable_source_inspection",
            captured_at_utc=_UTC,
        ),
    )
    anchor_deficit = EmpiricalAnchor(
        anchor_id="along_tangent_deficit_3p2_measured_input_20260703",
        measurement_utc="2026-07-03T00:00:00Z",
        inputs={
            "reference_freq_along": MEASURED_REFERENCE_TANGENT_FREQUENCY,
            "target": "lane-dash along-tangent residual",
        },
        predicted_output={"candidate_endpoint": 25.6},
        empirical_output={
            "deficit_factor": MEASURED_ALONG_TANGENT_DEFICIT,
            "scope": "measured input for candidate generation; endpoint is not an optimum claim",
        },
        residual=0.0,
        source_artifact=_DAG,
        measurement_method="4-lens residual spectral decomposition on the witness",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            _DAG,
            reactivation_criteria=(
                "recompute the residual spectrum if the clip, target surface, or directional basis changes"
            ),
            measurement_axis="[macOS-MLX research-signal]",
            hardware_substrate="apple_m5_max_mlx",
            captured_at_utc="2026-07-03T00:00:00Z",
        ),
    )
    anchor_warm_start = EmpiricalAnchor(
        anchor_id="warm_start_along26_formulation_refuted_20260710",
        measurement_utc="2026-07-10T15:31:20Z",
        inputs=SETTLED_WARM_START_ALONG26_SCOPE.to_dict(),
        predicted_output={
            "hypothesis": "fixed along-heavy allocation closes the 3.2x deficit and beats OFF"
        },
        empirical_output={
            "verdict": SETTLED_WARM_START_ALONG26_SCOPE.status,
            "delta_rebalanced_minus_off_ep700": 3.2e-05,
            "delta_units": "d_seg_argmax_disagreement_rate",
            "noise_floor": "UNMEASURED (single seed; instance-level delta)",
            "open_scope": OPEN_FRESH_FROM_SCRATCH_SCOPE.formulation,
        },
        residual=3.2e-05,
        source_artifact=_WARM_START_VERDICT,
        measurement_method="matched n600 through-R warm-start along26-vs-OFF formulation A/B",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        noise_floor=None,
        noise_floor_provenance=None,
        provenance=build_provenance_for_research_sidecar(
            _WARM_START_VERDICT,
            reactivation_criteria=(
                "do not rerun bounded warm-start along26; the distinct open arm is from-scratch "
                "init-time FreSh selection"
            ),
            measurement_axis="[macOS-CPU advisory]",
            hardware_substrate="apple_m5_max_cpu_torch",
            captured_at_utc="2026-07-10T15:31:20Z",
        ),
    )
    anchor_cost = EmpiricalAnchor(
        anchor_id="witness_epoch_scorer_fraction_and_grouped_backward_20260712",
        measurement_utc=_UTC,
        inputs={"vehicle": "live MLX level-set witness trainer"},
        predicted_output={"reason_to_optimize": "fewer epochs reduce scorer-gradient calls linearly"},
        empirical_output={
            "scorer_fraction_of_epoch": MEASURED_SCORER_EPOCH_FRACTION,
            "grouped_backward_speedup": MEASURED_GROUPED_BACKWARD_SPEEDUP,
            "scope": "internal wall-clock profile; not a score claim",
        },
        residual=0.0,
        source_artifact=_SURVEY,
        measurement_method="existing MLX/Metal compute-facet profile summarized by the OSS survey",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            _SURVEY,
            reactivation_criteria="re-profile after scorer, batching, device, or kernel-stack changes",
            measurement_axis="[macOS-MLX research-signal]",
            hardware_substrate="apple_m5_max_mlx",
            captured_at_utc=_UTC,
        ),
    )

    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name="FreSh DC-omitted spectral initialization and fixed-quality epoch reduction",
        one_line_summary=(
            "Choose the from-scratch frequency/bias initialization whose thin-lane residual "
            "through-R spectrum minimizes CDF-L1 W1; account for init and training scorer calls."
        ),
        latex_form=(
            r"S_d(A)=\sum_c\sum_{i+j=d}|\mathcal{F}(A_c)_{ij}|,\ d=1{:}n;\quad "
            r"W_1(p,q)=\sum_d|\sum_{k\le d}(p_k-q_k)|;\quad "
            r"j^*=\arg\min_j\frac1T\sum_tW_1(S(\omega_t\odot B(A_t)),"
            r"S(\omega_t\odot B(\hat A_j)));\quad "
            r"f_{\parallel,j}=f_0\delta^{j/2};\quad N_{seg}=N_{init}+PE,\ "
            r"T=T_{init}+EC"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.fresh_frequency_shift_init_20260712:"
            "select_fresh_spectrum_candidate"
        ),
        domain_of_validity={
            "vehicle": "from-scratch v7.5/v9 level-set witness initialization",
            "target": "SegNet-argmax boundary/partition target through the actual R surface",
            "selection_surface": (
                "thin/dashed class-1 boundary residual weight map; all-class boundary W1 retained "
                "only as a diagnostic"
            ),
            "spectrum_shape": "(C,H,W), unshifted FFT2, anti-diagonals d=1..n, DC omitted",
            "frequency_candidates": tangent_frequency_candidates(),
            "frequency_candidate_semantics": (
                "baseline/geometric-midpoint/3.2x endpoint; the W1 selection may retain baseline"
            ),
            "bias_scale_candidates": fresh_bias_scale_candidates(),
            "bias_candidate_semantics": "one shared standardized bias vector scaled by k; k only confound",
            "settled_excluded_formulation": SETTLED_WARM_START_ALONG26_SCOPE.to_dict(),
            "open_formulation": OPEN_FRESH_FROM_SCRATCH_SCOPE.to_dict(),
            "score_authority": "none; upstream/evaluate.py on exact archive bytes remains owed",
            "measurement_obligation": (
                "matched epochs-to-fixed-d_seg on faithful slice; governed n600 validation named as owed"
            ),
        },
        units_in={
            "signal": "channel-first scalar field",
            "frequency": "cycles_per_unit",
            "epochs": "integer passes over pair set",
            "epoch_seconds": "seconds_per_epoch",
        },
        units_out={
            "spectrum": "unitless probability mass over anti-diagonal index",
            "wasserstein1": "frequency-bin distance",
            "epoch_reduction_fraction": "unitless signed fraction",
            "wall_clock_reduction_fraction": "unitless signed fraction",
        },
        empirical_anchors=(anchor_formula, anchor_deficit, anchor_warm_start, anchor_cost),
        predicted_vs_empirical_residual={
            "fresh_source_formula": 0.0,
            "along_tangent_deficit_measured_input": 0.0,
            "warm_start_fixed_along26_hypothesis": 3.2e-05,
            "epoch_cost_profile_measured_input": 0.0,
        },
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.witness_init.fresh_frequency_shift",
            "tac.witness_dsl.curriculum_dsl.FreshFrequencyShift",
            "experiments.train_levelset_witness_realized_through_R_mlx",
        ),
        canonical_producers=(_SURVEY, _DAG, _WARM_START_VERDICT),
        provenance=build_provenance_for_predicted(
            model_id=EQUATION_ID,
            inputs_sha256=_LAW_SHA256,
            measurement_axis="[predicted build law]",
            hardware_substrate="numpy-portable",
            captured_at_utc=_UTC,
        ),
    )


def populate_fresh_frequency_shift_init_equation(
    *,
    path=None,
    lock_path=None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Append the equation through the locked registry helper when explicitly invoked."""

    from tac.canonical_equations.registry import register_canonical_equation

    equation = build_fresh_frequency_shift_init_v1()
    register_canonical_equation(
        equation,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes=(
            "fresh_frequency_shift_init_20260712; source law + measured 3.2x input; "
            "from-scratch fixed-d_seg result owed"
        ),
    )
    return equation


__all__ = [
    "DEFAULT_SPECTRUM_SIZE",
    "EQUATION_ID",
    "FRESH_BIAS_K_MAX",
    "FRESH_BIAS_K_MIN",
    "FRESH_BIAS_K_STEP",
    "FRESH_PAPER_ARXIV",
    "MEASURED_ALONG_TANGENT_DEFICIT",
    "MEASURED_GROUPED_BACKWARD_SPEEDUP",
    "MEASURED_REFERENCE_TANGENT_FREQUENCY",
    "MEASURED_SCORER_EPOCH_FRACTION",
    "OPEN_FRESH_FROM_SCRATCH_SCOPE",
    "SETTLED_WARM_START_ALONG26_SCOPE",
    "FixedQualityReduction",
    "FormulationVerdictScope",
    "FreshSpectrumSelection",
    "build_fresh_frequency_shift_init_v1",
    "fixed_quality_reduction_identity",
    "fresh_antidiagonal_spectrum",
    "fresh_bias_scale_candidates",
    "fresh_wasserstein1_cdf_l1",
    "populate_fresh_frequency_shift_init_equation",
    "select_fresh_spectrum_candidate",
    "tangent_frequency_candidates",
]
