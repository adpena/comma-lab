# SPDX-License-Identifier: MIT
"""Description-length search inside an exact uint8 resize fibre.

The scorer-facing plane is held fixed.  Candidate camera frames are generated
only by integer combinations of the complete #580 resize-kernel basis and are
therefore certified with exact resize numerators before coder admission.

Search order is deliberately level aware: global chart moves, then coherent
class/stratum object moves, then tile-local residual moves.  A lower level is
admitted only if the same deterministic coder is smaller.  The implementation
is a bounded heuristic, not a global MDL optimum and not a score authority.
"""

from __future__ import annotations

import hashlib
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

from tac.contest_score import RATE_WEIGHT, UNCOMPRESSED_SIZE_BYTES
from tac.optimization.resize_full_kernel import FullResizeKernel

MDL_MEMBER_SCHEMA = "mdl_polytope_member.v1"
MDL_ORIGIN_SOLVE_SCHEMA = "mdl_polytope_member_origin_solve.v1"
ZLIB_LEVEL = 9
DEFAULT_TILE_SCORER_HW = (16, 16)
DEFAULT_CALIBRATION_ROWS = 192
DEFAULT_RIDGE = 1.0e-6

CandidateName = Literal[
    "canonical",
    "chart_zero",
    "chart_horizontal",
    "chart_vertical",
    "chart_neighbor_mean",
    "chart_temporal_xi",
]

CANDIDATE_ORDER: tuple[CandidateName, ...] = (
    "canonical",
    "chart_zero",
    "chart_horizontal",
    "chart_vertical",
    "chart_neighbor_mean",
    "chart_temporal_xi",
)

FEATURE_NAMES = (
    "intercept",
    "piecewise_breaks",
    "gradient_l1",
    "kernel_move_l1",
    "temporal_xi_l1",
    "nonzero_values",
    "unique_values",
)


class MdlMemberError(ValueError):
    """Fail-closed invalid geometry, calibration, or exactness claim."""


def _sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def zlib9_bytes(value: np.ndarray) -> int:
    """The diagnostic coder used consistently for calibration and admission."""

    raw = np.ascontiguousarray(value)
    if raw.dtype != np.uint8:
        raise MdlMemberError("zlib9 coder accepts uint8 arrays only")
    return len(zlib.compress(raw.tobytes(), level=ZLIB_LEVEL))


def modular_uint8_residual(member: np.ndarray, origin: np.ndarray) -> np.ndarray:
    """Return the exact one-byte group residual ``member - origin (mod 256)``.

    The mapping is bijective for uint8 arrays, unlike a clipped signed
    difference.  It is therefore a real residual-code surface: the decoder
    reconstructs with uint8 modular addition and no side information.
    """

    x = np.asarray(member)
    base = np.asarray(origin)
    if x.dtype != np.uint8 or base.dtype != np.uint8 or x.shape != base.shape or x.size == 0:
        raise MdlMemberError("member and origin must be same-shape nonempty uint8 arrays")
    return np.subtract(x, base, dtype=np.uint8)


def reconstruct_modular_uint8(origin: np.ndarray, residual: np.ndarray) -> np.ndarray:
    """Invert :func:`modular_uint8_residual` exactly."""

    base = np.asarray(origin)
    code = np.asarray(residual)
    if base.dtype != np.uint8 or code.dtype != np.uint8 or base.shape != code.shape or base.size == 0:
        raise MdlMemberError("origin and residual must be same-shape nonempty uint8 arrays")
    return np.add(base, code, dtype=np.uint8)


def residual_zlib9_bytes(member: np.ndarray, origin: np.ndarray) -> int:
    """Measure the real zlib-9 context code on an exact modular residual."""

    residual = modular_uint8_residual(member, origin)
    if not np.array_equal(reconstruct_modular_uint8(origin, residual), member):
        raise MdlMemberError("modular residual failed exact parse-back")
    return len(zlib.compress(residual.tobytes(), level=ZLIB_LEVEL))


def _extended_gcd(first: int, second: int) -> tuple[int, int, int]:
    old_r, r = abs(int(first)), abs(int(second))
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_s, s = s, old_s - quotient * s
        old_t, t = t, old_t - quotient * t
    if first < 0:
        old_s = -old_s
    if second < 0:
        old_t = -old_t
    return old_r, old_s, old_t


def _nearest_integer_ratio(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise MdlMemberError("integer reduction denominator must be positive")
    magnitude = (2 * abs(numerator) + denominator) // (2 * denominator)
    return -magnitude if numerator < 0 else magnitude


def _integer_dot(first: Sequence[int], second: Sequence[int]) -> int:
    return sum(
        int(left) * int(right)
        for left, right in zip(first, second, strict=True)
    )


def _rank_three_integer_rows(rows: np.ndarray) -> bool:
    value = np.asarray(rows, dtype=object)
    if value.shape != (3, 4):
        return False
    for omitted in range(4):
        columns = [index for index in range(4) if index != omitted]
        a, b, c = (
            [int(value[row, column]) for column in columns]
            for row in range(3)
        )
        determinant = (
            a[0] * (b[1] * c[2] - b[2] * c[1])
            - a[1] * (b[0] * c[2] - b[2] * c[0])
            + a[2] * (b[0] * c[1] - b[1] * c[0])
        )
        if determinant:
            return True
    return False


def reduce_integer_kernel_basis(basis: np.ndarray) -> np.ndarray:
    """Deterministically size-reduce a rank-three integer basis.

    This is the exact-integer fallback required when fplll is unavailable.
    Every update is unimodular row subtraction or a row permutation, so the
    represented lattice is unchanged.  It is a reduction/preconditioner, not
    an LLL optimality claim and not the coder objective.
    """

    raw = np.asarray(basis)
    if raw.shape != (3, 4) or raw.dtype.kind not in "iu":
        raise MdlMemberError("kernel basis must be a 3x4 integer matrix")
    rows = raw.astype(object, copy=True)
    if not _rank_three_integer_rows(rows):
        raise MdlMemberError("kernel basis must have rank three")
    for _pass in range(64):
        prior = rows.copy()
        order = sorted(
            range(3),
            key=lambda index: (
                _integer_dot(rows[index], rows[index]),
                tuple(int(value) for value in rows[index]),
            ),
        )
        rows = rows[order]
        for index in range(1, 3):
            for earlier in range(index):
                denominator = _integer_dot(rows[earlier], rows[earlier])
                numerator = _integer_dot(rows[index], rows[earlier])
                quotient = _nearest_integer_ratio(numerator, denominator)
                if quotient:
                    candidate = rows[index] - quotient * rows[earlier]
                    candidate_norm = _integer_dot(candidate, candidate)
                    current_norm = _integer_dot(rows[index], rows[index])
                    if candidate_norm < current_norm or (
                        candidate_norm == current_norm
                        and tuple(int(value) for value in candidate)
                        < tuple(int(value) for value in rows[index])
                    ):
                        rows[index] = candidate
        if np.array_equal(rows, prior):
            break
    else:
        raise MdlMemberError("integer kernel size reduction did not converge")
    try:
        reduced = np.asarray(rows, dtype=np.int64)
    except OverflowError as exc:
        raise MdlMemberError("reduced integer kernel basis exceeds int64") from exc
    if not _rank_three_integer_rows(reduced):
        raise MdlMemberError("integer kernel reduction lost rank")
    return reduced


def saturated_integer_kernel_basis(coefficients: Sequence[int]) -> np.ndarray:
    """Return a saturated, size-reduced Z-basis of ``ker(coefficients)``.

    Successive exact unimodular column operations transform ``c`` to
    ``(gcd(c),0,0,0)``.  The remaining columns are therefore the full integer
    kernel, not the high-index atom sublattice used by the original #602
    projector.  This is the narrow #586 mechanism adoption.
    """

    try:
        row = [int(value) for value in coefficients]
    except (TypeError, ValueError) as exc:
        raise MdlMemberError("coefficients must be four positive integers") from exc
    if len(row) != 4 or any(value <= 0 for value in row):
        raise MdlMemberError("coefficients must be four positive integers")
    transform = np.eye(4, dtype=object)
    work = list(row)
    for column in range(1, 4):
        first, second = work[0], work[column]
        divisor, bezout_first, bezout_second = _extended_gcd(first, second)
        two_by_two = np.asarray(
            [
                [bezout_first, -(second // divisor)],
                [bezout_second, first // divisor],
            ],
            dtype=object,
        )
        transform[:, [0, column]] = transform[:, [0, column]] @ two_by_two
        work[0], work[column] = divisor, 0
    basis_object = transform[:, 1:].T
    try:
        basis = np.asarray(basis_object, dtype=np.int64)
    except OverflowError as exc:
        raise MdlMemberError("saturated kernel basis exceeds int64") from exc
    coefficient_row = np.asarray(row, dtype=np.int64)
    if not np.array_equal(basis @ coefficient_row, np.zeros(3, dtype=np.int64)):
        raise MdlMemberError("saturated kernel construction failed exact nullity")
    reduced = reduce_integer_kernel_basis(basis)
    if not np.array_equal(reduced @ coefficient_row, np.zeros(3, dtype=np.int64)):
        raise MdlMemberError("integer reduction escaped the exact kernel")
    return reduced


def lawref_manifest() -> list[dict[str, Any]]:
    """Return the value-provenance ladder for every solver constant.

    The constants are either exact outputs of registered structural laws,
    measured coder anchors, or task-scoped search-grid values.  This receipt
    surface prevents the numeric defaults from silently acquiring authority.
    """

    return [
        {
            "name": "resize_geometry_and_integer_kernel",
            "value": "runtime FullResizeKernel geometry plus saturated exact-integer local basis",
            "law_id": "separable_resize_full_kernel_direct_sum_v1",
            "ladder_class": "derived_live",
            "crosswalk": "#586 exact saturation plus integer size reduction; no sieve analogy",
        },
        {
            "name": "coder_level",
            "value": ZLIB_LEVEL,
            "law_id": "partition_temporal_transport_amortization_jitter_bound_v1",
            "ladder_class": "measured_anchor",
            "scope": "zlib-9 diagnostic proxy only",
        },
        {
            "name": "tile_scorer_hw",
            "value": list(DEFAULT_TILE_SCORER_HW),
            "law_id": "realization_necessity_preimage_per_stratum_v1",
            "ladder_class": "hardcoded_waiver",
            "waiver": (
                "task-scoped dyadic chart search granularity; recalibrate when scorer geometry, "
                "coder, or stratum statistics change"
            ),
        },
        {
            "name": "calibration_rows",
            "value": DEFAULT_CALIBRATION_ROWS,
            "law_id": "witness_measured_reverse_waterfill_v1",
            "ladder_class": "hardcoded_waiver",
            "waiver": (
                "bounded n16 calibration budget exceeding the delegated >=20-tile floor; "
                "recalibrate on coder or feature change"
            ),
        },
        {
            "name": "ridge",
            "value": DEFAULT_RIDGE,
            "law_id": "witness_measured_reverse_waterfill_v1",
            "ladder_class": "hardcoded_waiver",
            "waiver": "numerical stabilization only; refit on feature scaling change",
        },
        {
            "name": "admission_threshold_delta_s_per_byte",
            "value": RATE_WEIGHT / UNCOMPRESSED_SIZE_BYTES,
            "law_id": "witness_measured_reverse_waterfill_v1",
            "ladder_class": "derived_at_config",
        },
    ]


@dataclass(frozen=True)
class ProxyCalibration:
    feature_names: tuple[str, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    row_count: int
    pearson_r: float
    spearman_r: float
    mean_absolute_error_bytes: float

    def predict(self, features: np.ndarray) -> np.ndarray:
        raw = np.asarray(features, dtype=np.float64)
        if raw.ndim != 2 or raw.shape[1] != len(self.feature_names):
            raise MdlMemberError("proxy feature matrix shape mismatch")
        means = np.asarray(self.means)
        scales = np.asarray(self.scales)
        coeff = np.asarray(self.coefficients)
        return ((raw - means) / scales) @ coeff

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "means": list(self.means),
            "scales": list(self.scales),
            "coefficients": list(self.coefficients),
            "row_count": self.row_count,
            "pearson_r": self.pearson_r,
            "spearman_r": self.spearman_r,
            "mean_absolute_error_bytes": self.mean_absolute_error_bytes,
        }

    @classmethod
    def from_dict(cls, row: Mapping[str, Any]) -> ProxyCalibration:
        return cls(
            tuple(str(v) for v in row["feature_names"]),
            tuple(float(v) for v in row["means"]),
            tuple(float(v) for v in row["scales"]),
            tuple(float(v) for v in row["coefficients"]),
            int(row["row_count"]),
            float(row["pearson_r"]),
            float(row["spearman_r"]),
            float(row["mean_absolute_error_bytes"]),
        )


@dataclass(frozen=True)
class LevelResult:
    name: str
    frame: np.ndarray
    coder_bytes: int
    delta_bytes_vs_previous: int
    selected_groups: int
    exact_numerators_equal: bool


@dataclass(frozen=True)
class MemberSolveResult:
    canonical: np.ndarray
    selected: np.ndarray
    levels: tuple[LevelResult, ...]
    candidate_names: tuple[str, ...]
    exact_numerators_equal: bool
    selected_sha256: str
    canonical_sha256: str

    @property
    def canonical_bytes(self) -> int:
        return self.levels[0].coder_bytes

    @property
    def selected_bytes(self) -> int:
        return self.levels[-1].coder_bytes


@dataclass(frozen=True)
class OriginSolveResult:
    """Exact-fibre member selected by a real residual coder."""

    canonical: np.ndarray
    selected: np.ndarray
    origin: np.ndarray
    canonical_member_bytes: int
    canonical_residual_bytes: int
    proposed_residual_bytes: int
    selected_residual_bytes: int
    selected_candidate: str
    changed_values: int
    exact_numerators_equal: bool
    canonical_sha256: str
    selected_sha256: str
    origin_sha256: str

    @property
    def bytes_b_over_a(self) -> float:
        return self.selected_residual_bytes / self.canonical_member_bytes

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": MDL_ORIGIN_SOLVE_SCHEMA,
            "coder": "zlib-9 over exact modular uint8 residual",
            "canonical_member_bytes": self.canonical_member_bytes,
            "canonical_residual_bytes": self.canonical_residual_bytes,
            "proposed_residual_bytes": self.proposed_residual_bytes,
            "selected_residual_bytes": self.selected_residual_bytes,
            "bytes_b_over_a": self.bytes_b_over_a,
            "selected_candidate": self.selected_candidate,
            "changed_values": self.changed_values,
            "exact_numerators_equal": self.exact_numerators_equal,
            "canonical_sha256": self.canonical_sha256,
            "selected_sha256": self.selected_sha256,
            "origin_sha256": self.origin_sha256,
            "byte_partition": {
                "FREE": "generic modular add and saturated-basis construction code",
                "NULL": "ker(A) coordinates pinned to origin; no payload credit",
                "COUNTED": "zlib-9 residual bytes",
            },
            "pool_id": "solver_member_selection",
            "score_claim": False,
        }


def _rankdata(value: np.ndarray) -> np.ndarray:
    order = np.argsort(value, kind="mergesort")
    ranks = np.empty(len(value), dtype=np.float64)
    start = 0
    while start < len(value):
        end = start + 1
        while end < len(value) and value[order[end]] == value[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def fit_proxy(features: np.ndarray, actual_bytes: np.ndarray) -> ProxyCalibration:
    """Fit the computable D1 byte proxy and report calibration-sample correlations."""

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(actual_bytes, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES) or len(x) != len(y):
        raise MdlMemberError("invalid proxy calibration arrays")
    if len(x) < 20 or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise MdlMemberError("proxy calibration needs at least 20 finite rows")
    means = x.mean(axis=0)
    scales = x.std(axis=0)
    scales[scales == 0.0] = 1.0
    # The explicit intercept column must remain one after normalization.
    means[0] = 0.0
    scales[0] = 1.0
    z = (x - means) / scales
    augmented_x = np.concatenate(
        (z, np.sqrt(DEFAULT_RIDGE) * np.eye(z.shape[1])), axis=0
    )
    augmented_y = np.concatenate((y, np.zeros(z.shape[1], dtype=np.float64)))
    coeff = np.linalg.lstsq(augmented_x, augmented_y, rcond=None)[0]
    pred = z @ coeff
    if not np.all(np.isfinite(coeff)) or not np.all(np.isfinite(pred)):
        raise MdlMemberError("proxy calibration produced non-finite values")
    pearson = float(np.corrcoef(pred, y)[0, 1]) if np.std(pred) and np.std(y) else 0.0
    rp, ry = _rankdata(pred), _rankdata(y)
    spearman = float(np.corrcoef(rp, ry)[0, 1]) if np.std(rp) and np.std(ry) else 0.0
    return ProxyCalibration(
        FEATURE_NAMES,
        tuple(float(v) for v in means),
        tuple(float(v) for v in scales),
        tuple(float(v) for v in coeff),
        len(x),
        pearson,
        spearman,
        float(np.mean(np.abs(pred - y))),
    )


class MdlPolytopeMemberSolver:
    """Chart/object-first bounded MDL search inside one exact resize fibre."""

    def __init__(
        self,
        kernel: FullResizeKernel | None = None,
        *,
        tile_scorer_hw: tuple[int, int] = DEFAULT_TILE_SCORER_HW,
    ) -> None:
        self.kernel = kernel or FullResizeKernel.build()
        th, tw = (int(tile_scorer_hw[0]), int(tile_scorer_hw[1]))
        if th < 1 or tw < 1:
            raise MdlMemberError("tile dimensions must be positive")
        self.tile_scorer_hw = (th, tw)
        self._row_indices = np.asarray(
            [support.indices for support in self.kernel.operator.row_supports], dtype=np.intp
        )
        self._col_indices = np.asarray(
            [support.indices for support in self.kernel.operator.col_supports], dtype=np.intp
        )
        self._basis, self._pinv = self._build_local_basis()
        self._basis_norm_summary_cache: dict[str, float | int] | None = None
        self._basis_type_cache: tuple[np.ndarray, np.ndarray] | None = None
        self._facet_dimension_lookup_cache: np.ndarray | None = None

    def _build_local_basis(self) -> tuple[np.ndarray, np.ndarray]:
        h, w = self.kernel.scorer_h, self.kernel.scorer_w
        basis = np.zeros((h, w, 4, 3), dtype=np.int64)
        cache: dict[tuple[int, ...], np.ndarray] = {}
        row_supports = self.kernel.operator.row_supports
        col_supports = self.kernel.operator.col_supports
        for row_index, row_support in enumerate(row_supports):
            for col_index, col_support in enumerate(col_supports):
                coefficients = tuple(
                    int(value)
                    for value in np.outer(
                        row_support.numerators, col_support.numerators
                    ).reshape(-1)
                )
                reduced = cache.get(coefficients)
                if reduced is None:
                    reduced = saturated_integer_kernel_basis(coefficients)
                    cache[coefficients] = reduced
                basis[row_index, col_index] = reduced.T
        return basis, np.linalg.pinv(basis.astype(np.float64))

    def canonicalize(self, exact_member: np.ndarray) -> np.ndarray:
        raw = np.asarray(exact_member)
        if raw.dtype != np.uint8 or raw.shape != (
            self.kernel.camera_h,
            self.kernel.camera_w,
            3,
        ):
            raise MdlMemberError("member must be camera-shaped uint8 RGB")
        numerators, denominator = self.kernel.operator.apply_numerators(raw)
        if np.any(numerators % denominator):
            raise MdlMemberError("source member does not map to an integer scorer plane")
        target = (numerators // denominator).astype(np.uint8)
        canonical = self.kernel.operator.realize_factor2_uint8(target)
        check, check_denominator = self.kernel.operator.apply_numerators(canonical)
        if denominator != check_denominator or not np.array_equal(numerators, check):
            raise MdlMemberError("canonical support fill failed exact numerator equality")
        return canonical

    @staticmethod
    def _preference(
        canonical: np.ndarray,
        name: CandidateName,
        temporal: np.ndarray | None,
    ) -> np.ndarray:
        x = canonical.astype(np.float64)
        if name == "canonical":
            return x
        if name == "chart_zero":
            return np.zeros_like(x)
        if name == "chart_horizontal":
            return np.concatenate((x[:, :1], x[:, :-1]), axis=1)
        if name == "chart_vertical":
            return np.concatenate((x[:1], x[:-1]), axis=0)
        if name == "chart_neighbor_mean":
            return (
                np.roll(x, 1, axis=0)
                + np.roll(x, -1, axis=0)
                + np.roll(x, 1, axis=1)
                + np.roll(x, -1, axis=1)
            ) / 4.0
        if name == "chart_temporal_xi":
            return x if temporal is None else temporal.astype(np.float64)
        raise MdlMemberError(f"unknown candidate {name!r}")

    def project_preference(
        self,
        canonical: np.ndarray,
        name: CandidateName,
        *,
        temporal: np.ndarray | None = None,
    ) -> np.ndarray:
        """Coordinate-descent projection onto the bounded integer kernel lattice."""

        source = np.asarray(canonical)
        preferred = self._preference(source, name, temporal)
        h, w = self.kernel.scorer_h, self.kernel.scorer_w
        blocks = source[
            self._row_indices[:, None, :, None],
            self._col_indices[None, :, None, :],
            :,
        ].reshape(h, w, 4, 3).transpose(0, 1, 3, 2).astype(np.int64)
        pref_blocks = preferred[
            self._row_indices[:, None, :, None],
            self._col_indices[None, :, None, :],
            :,
        ].reshape(h, w, 4, 3).transpose(0, 1, 3, 2)
        delta = pref_blocks - blocks
        coefficients = np.rint(
            np.einsum("hwkf,hwcf->hwck", self._pinv, delta)
        ).astype(np.int64)
        selected = np.zeros(blocks.shape[:3], dtype=np.bool_)
        out_blocks = blocks.copy()
        # Dyadic backtracking is bounded and deterministic.  If its ten-rung
        # budget does not reach a feasible point, fail closed; do not silently
        # substitute a move that was not measured by this candidate search.
        for shift in range(10):
            trial_coefficients = np.rint(coefficients / (2**shift)).astype(np.int64)
            trial = blocks + np.einsum(
                "hwfk,hwck->hwcf", self._basis, trial_coefficients
            )
            fits = np.all((trial >= 0) & (trial <= 255), axis=-1) & ~selected
            out_blocks[fits] = trial[fits]
            selected |= fits
        if not np.all(selected):
            raise MdlMemberError("bounded lattice backtracking failed to reach zero move")
        out = source.copy()
        packed = out_blocks.transpose(0, 1, 3, 2).reshape(h, w, 2, 2, 3)
        out[
            self._row_indices[:, None, :, None],
            self._col_indices[None, :, None, :],
            :,
        ] = packed.astype(np.uint8)
        self._require_exact(source, out)
        return out

    def generate_candidates(
        self, canonical: np.ndarray, *, temporal: np.ndarray | None = None
    ) -> dict[str, np.ndarray]:
        return {
            name: (
                np.asarray(canonical).copy()
                if name == "canonical"
                else self.project_preference(canonical, name, temporal=temporal)
            )
            for name in CANDIDATE_ORDER
        }

    def solve_against_origin(
        self,
        canonical: np.ndarray,
        *,
        origin: np.ndarray,
    ) -> OriginSolveResult:
        """Select the shortest exact-fibre residual against a free prediction.

        The saturated local basis proposes a closest-vector member.  The
        proposal is admitted only when the real zlib-9 residual is strictly
        shorter; ties deterministically retain the canonical member.  Frozen
        scorer admission remains a caller responsibility on the realized
        uint8 output.
        """

        source = np.asarray(canonical)
        base = np.asarray(origin)
        expected_shape = (
            self.kernel.camera_h,
            self.kernel.camera_w,
            3,
        )
        if (
            source.dtype != np.uint8
            or base.dtype != np.uint8
            or source.shape != expected_shape
            or base.shape != expected_shape
        ):
            raise MdlMemberError("canonical and origin must be camera-shaped uint8 RGB")
        proposed = self.project_preference(
            source,
            "chart_temporal_xi",
            temporal=base,
        )
        canonical_member_bytes = zlib9_bytes(source)
        canonical_residual_bytes = residual_zlib9_bytes(source, base)
        proposed_residual_bytes = residual_zlib9_bytes(proposed, base)
        if proposed_residual_bytes < canonical_residual_bytes:
            selected = proposed
            selected_bytes = proposed_residual_bytes
            selected_candidate = "saturated_local_cvp"
        else:
            selected = source.copy()
            selected_bytes = canonical_residual_bytes
            selected_candidate = "canonical_tie_break"
        self._require_exact(source, selected)
        return OriginSolveResult(
            source.copy(),
            selected,
            base.copy(),
            canonical_member_bytes,
            canonical_residual_bytes,
            proposed_residual_bytes,
            selected_bytes,
            selected_candidate,
            int(np.count_nonzero(selected != source)),
            True,
            _sha256_array(source),
            _sha256_array(selected),
            _sha256_array(base),
        )

    def _require_exact(self, reference: np.ndarray, candidate: np.ndarray) -> None:
        a, da = self.kernel.operator.apply_numerators(reference)
        b, db = self.kernel.operator.apply_numerators(candidate)
        if da != db or not np.array_equal(a, b):
            raise MdlMemberError("candidate escaped the exact integer resize fibre")

    def basis_norm_summary(self) -> dict[str, float | int]:
        """Return the exact invariant distribution summary of local basis norms."""

        if self._basis_norm_summary_cache is None:
            norms = np.linalg.norm(self._basis.astype(np.float64), axis=2).reshape(-1)
            self._basis_norm_summary_cache = {
                "count": int(norms.size),
                "norm_min": float(norms.min()),
                "norm_p50": float(np.quantile(norms, 0.5)),
                "norm_p95": float(np.quantile(norms, 0.95)),
                "norm_max": float(norms.max()),
            }
        return dict(self._basis_norm_summary_cache)

    def local_facet_dimensions(self, member: np.ndarray) -> np.ndarray:
        """Return exact continuous face dimensions for every local RGB block.

        Each resized output coordinate has a rank-three affine fibre. Camera
        coordinates at 0 or 255 activate box facets. The returned dimension is
        ``3 - rank(B_active)`` for the saturated local basis ``B``. It measures
        local polytope degeneracy; it does not claim integer-neighbour reach or
        expose nonexistent optimizer duals.
        """

        value = np.asarray(member)
        expected_shape = (self.kernel.camera_h, self.kernel.camera_w, 3)
        if value.dtype != np.uint8 or value.shape != expected_shape:
            raise MdlMemberError("facet member must be camera-shaped uint8 RGB")
        h, w = self.kernel.scorer_h, self.kernel.scorer_w
        blocks = value[
            self._row_indices[:, None, :, None],
            self._col_indices[None, :, None, :],
            :,
        ].reshape(h, w, 4, 3).transpose(0, 1, 3, 2)
        active_masks = np.zeros(blocks.shape[:3], dtype=np.uint8)
        active_bounds = (blocks == 0) | (blocks == 255)
        for coordinate in range(4):
            active_masks |= active_bounds[..., coordinate].astype(np.uint8) << coordinate

        if self._basis_type_cache is None:
            unique, inverse = np.unique(
                self._basis.reshape(h * w, 12), axis=0, return_inverse=True
            )
            self._basis_type_cache = (
                unique.reshape(-1, 4, 3),
                inverse.reshape(h, w),
            )
        unique_basis, basis_types = self._basis_type_cache
        if self._facet_dimension_lookup_cache is None:
            lookup = np.empty((len(unique_basis), 16), dtype=np.uint8)
            for basis_type, basis in enumerate(unique_basis):
                for mask in range(16):
                    active_rows = [
                        index for index in range(4) if mask & (1 << index)
                    ]
                    rank = (
                        0
                        if not active_rows
                        else int(
                            np.linalg.matrix_rank(
                                basis[active_rows].astype(np.float64)
                            )
                        )
                    )
                    lookup[basis_type, mask] = 3 - rank
            self._facet_dimension_lookup_cache = lookup
        return self._facet_dimension_lookup_cache[
            basis_types[..., None], active_masks
        ]

    def tile_slices(self) -> list[tuple[slice, slice, slice, slice]]:
        th, tw = self.tile_scorer_hw
        rows: list[tuple[slice, slice, slice, slice]] = []
        for y0 in range(0, self.kernel.scorer_h, th):
            y1 = min(y0 + th, self.kernel.scorer_h)
            camera_y = self._row_indices[y0:y1]
            for x0 in range(0, self.kernel.scorer_w, tw):
                x1 = min(x0 + tw, self.kernel.scorer_w)
                camera_x = self._col_indices[x0:x1]
                rows.append(
                    (
                        slice(y0, y1),
                        slice(x0, x1),
                        slice(int(camera_y.min()), int(camera_y.max()) + 1),
                        slice(int(camera_x.min()), int(camera_x.max()) + 1),
                    )
                )
        return rows

    @staticmethod
    def tile_stratum(labels: np.ndarray) -> str:
        x = np.asarray(labels)
        edge = bool(np.any(x[1:] != x[:-1]) or np.any(x[:, 1:] != x[:, :-1]))
        if not edge:
            return "cell"
        saddle = False
        if x.shape[0] > 1 and x.shape[1] > 1:
            # A vectorized exact 3-class test would allocate a large one-hot;
            # bounded tile loops keep this receipt path simple and deterministic.
            for y in range(x.shape[0] - 1):
                for z in range(x.shape[1] - 1):
                    if len({int(v) for v in x[y : y + 2, z : z + 2].flat}) >= 3:
                        saddle = True
                        break
                if saddle:
                    break
        return "saddle" if saddle else "edge"

    @staticmethod
    def tile_class(labels: np.ndarray) -> int:
        values, counts = np.unique(np.asarray(labels), return_counts=True)
        return int(values[np.argmax(counts)])

    @staticmethod
    def features(
        patch: np.ndarray,
        canonical_patch: np.ndarray,
        temporal_patch: np.ndarray,
    ) -> np.ndarray:
        x = np.asarray(patch, dtype=np.int16)
        c = np.asarray(canonical_patch, dtype=np.int16)
        t = np.asarray(temporal_patch, dtype=np.int16)
        breaks = np.count_nonzero(x[1:] != x[:-1]) + np.count_nonzero(x[:, 1:] != x[:, :-1])
        gradient = np.abs(x[1:] - x[:-1]).sum() + np.abs(x[:, 1:] - x[:, :-1]).sum()
        return np.asarray(
            [
                1.0,
                float(breaks),
                float(gradient),
                float(np.abs(x - c).sum()),
                float(np.abs(x - t).sum()),
                float(np.count_nonzero(x)),
                float(len(np.unique(x))),
            ],
            dtype=np.float64,
        )

    def calibration_rows(
        self,
        candidates: Mapping[str, np.ndarray],
        *,
        temporal: np.ndarray,
        max_rows: int = DEFAULT_CALIBRATION_ROWS,
    ) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
        if max_rows < 20:
            raise MdlMemberError("calibration row cap must be at least 20")
        tiles = self.tile_slices()
        names = tuple(name for name in CANDIDATE_ORDER if name in candidates)
        rows: list[np.ndarray] = []
        actual: list[int] = []
        metadata: list[dict[str, Any]] = []
        canonical = candidates["canonical"]
        # Coprime deterministic stride covers the full frame without RNG state.
        tile_index = 0
        while len(rows) < max_rows:
            tile = tiles[(tile_index * 97) % len(tiles)]
            name = names[tile_index % len(names)]
            _sy, _sx, cy, cx = tile
            patch = candidates[name][cy, cx]
            rows.append(self.features(patch, canonical[cy, cx], temporal[cy, cx]))
            actual.append(zlib9_bytes(patch))
            metadata.append({"tile_index": (tile_index * 97) % len(tiles), "candidate": name})
            tile_index += 1
        return np.stack(rows), np.asarray(actual, dtype=np.float64), metadata

    def _compose_groups(
        self,
        base: np.ndarray,
        candidates: Mapping[str, np.ndarray],
        group_choices: Mapping[Any, str],
        group_keys: Sequence[Any],
    ) -> np.ndarray:
        out = np.asarray(base).copy()
        for tile, key in zip(self.tile_slices(), group_keys, strict=True):
            name = group_choices.get(key, "canonical")
            if name == "canonical":
                continue
            sy, sx, _cy, _cx = tile
            row_ids = self._row_indices[sy]
            col_ids = self._col_indices[sx]
            out[
                row_ids[:, None, :, None],
                col_ids[None, :, None, :],
                :,
            ] = candidates[name][
                row_ids[:, None, :, None],
                col_ids[None, :, None, :],
                :,
            ]
        self._require_exact(base, out)
        return out

    def solve(
        self,
        canonical: np.ndarray,
        *,
        temporal: np.ndarray,
        labels: np.ndarray,
        calibration: ProxyCalibration,
    ) -> MemberSolveResult:
        """Run chart, object, then tile-residual coder-admitted search."""

        candidates = self.generate_candidates(canonical, temporal=temporal)
        canonical_bytes = zlib9_bytes(canonical)
        levels: list[LevelResult] = [
            LevelResult("canonical", canonical.copy(), canonical_bytes, 0, 0, True)
        ]

        # Level 1: one coherent chart move over the full frame.
        chart_sizes = {name: zlib9_bytes(frame) for name, frame in candidates.items()}
        chart_name = min(chart_sizes, key=lambda name: (chart_sizes[name], name != "canonical", name))
        chart = candidates[chart_name]
        chart_bytes = chart_sizes[chart_name]
        levels.append(
            LevelResult(
                "chart",
                chart.copy(),
                chart_bytes,
                chart_bytes - canonical_bytes,
                int(chart_name != "canonical"),
                True,
            )
        )

        tiles = self.tile_slices()
        keys: list[tuple[int, str]] = []
        tile_predictions: list[dict[str, float]] = []
        for sy, sx, cy, cx in tiles:
            tile_labels = labels[sy, sx]
            keys.append((self.tile_class(tile_labels), self.tile_stratum(tile_labels)))
            prediction: dict[str, float] = {}
            for name, frame in candidates.items():
                row = self.features(frame[cy, cx], canonical[cy, cx], temporal[cy, cx])[None]
                prediction[name] = float(calibration.predict(row)[0])
            tile_predictions.append(prediction)

        # Level 2: each class/stratum object chooses one coherent chart move.
        group_choices: dict[tuple[int, str], str] = {}
        for key in sorted(set(keys)):
            indices = [i for i, value in enumerate(keys) if value == key]
            totals = {
                name: sum(tile_predictions[i][name] for i in indices)
                for name in candidates
            }
            group_choices[key] = min(totals, key=lambda name: (totals[name], name != "canonical", name))
        object_frame = self._compose_groups(chart, candidates, group_choices, keys)
        object_bytes = zlib9_bytes(object_frame)
        if object_bytes >= chart_bytes:
            object_frame, object_bytes, admitted_groups = chart.copy(), chart_bytes, 0
        else:
            admitted_groups = sum(name != "canonical" for name in group_choices.values())
        levels.append(
            LevelResult(
                "object_class_stratum",
                object_frame.copy(),
                object_bytes,
                object_bytes - chart_bytes,
                admitted_groups,
                True,
            )
        )

        # Level 3: last-resort tile/cell integer moves after coherent levels.
        tile_choices = {
            index: min(
                prediction,
                key=lambda name: (prediction[name], name != "canonical", name),
            )
            for index, prediction in enumerate(tile_predictions)
        }
        tile_frame = self._compose_groups(
            object_frame,
            candidates,
            tile_choices,
            list(range(len(tiles))),
        )
        tile_bytes = zlib9_bytes(tile_frame)
        if tile_bytes >= object_bytes:
            tile_frame, tile_bytes, admitted_tiles = object_frame.copy(), object_bytes, 0
        else:
            admitted_tiles = sum(name != "canonical" for name in tile_choices.values())
        levels.append(
            LevelResult(
                "pixel_tile_residual",
                tile_frame,
                tile_bytes,
                tile_bytes - object_bytes,
                admitted_tiles,
                True,
            )
        )
        self._require_exact(canonical, tile_frame)
        return MemberSolveResult(
            canonical.copy(),
            tile_frame,
            tuple(levels),
            tuple(candidates),
            True,
            _sha256_array(tile_frame),
            _sha256_array(canonical),
        )


__all__ = [
    "CANDIDATE_ORDER",
    "DEFAULT_CALIBRATION_ROWS",
    "DEFAULT_TILE_SCORER_HW",
    "FEATURE_NAMES",
    "MDL_MEMBER_SCHEMA",
    "MDL_ORIGIN_SOLVE_SCHEMA",
    "MdlMemberError",
    "MdlPolytopeMemberSolver",
    "MemberSolveResult",
    "OriginSolveResult",
    "ProxyCalibration",
    "fit_proxy",
    "lawref_manifest",
    "modular_uint8_residual",
    "reconstruct_modular_uint8",
    "reduce_integer_kernel_basis",
    "residual_zlib9_bytes",
    "saturated_integer_kernel_basis",
    "zlib9_bytes",
]
