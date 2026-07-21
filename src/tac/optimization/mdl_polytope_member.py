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


def lawref_manifest() -> list[dict[str, Any]]:
    """Return the value-provenance ladder for every solver constant.

    The constants are either exact outputs of registered structural laws,
    measured coder anchors, or task-scoped search-grid values.  This receipt
    surface prevents the numeric defaults from silently acquiring authority.
    """

    return [
        {
            "name": "resize_geometry_and_integer_kernel",
            "value": "runtime FullResizeKernel geometry",
            "law_id": "separable_resize_full_kernel_direct_sum_v1",
            "ladder_class": "derived_live",
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

    def _build_local_basis(self) -> tuple[np.ndarray, np.ndarray]:
        def primitive(first: int, second: int) -> tuple[int, int]:
            divisor = int(np.gcd(abs(first), abs(second)))
            return first // divisor, second // divisor

        row_null = np.asarray(self.kernel.height.primitive_null_atoms(), dtype=np.int64)
        row_space = np.asarray(
            [primitive(*support.numerators) for support in self.kernel.operator.row_supports],
            dtype=np.int64,
        )
        col_null = np.asarray(self.kernel.width.primitive_null_atoms(), dtype=np.int64)
        h, w = self.kernel.scorer_h, self.kernel.scorer_w
        basis = np.zeros((h, w, 4, 3), dtype=np.int64)
        basis[:, :, 0, 0] = row_null[:, 0, None]
        basis[:, :, 2, 0] = row_null[:, 1, None]
        basis[:, :, 1, 1] = row_null[:, 0, None]
        basis[:, :, 3, 1] = row_null[:, 1, None]
        basis[:, :, :, 2] = (
            row_space[:, None, :, None] * col_null[None, :, None, :]
        ).reshape(h, w, 4)
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

    def _require_exact(self, reference: np.ndarray, candidate: np.ndarray) -> None:
        a, da = self.kernel.operator.apply_numerators(reference)
        b, db = self.kernel.operator.apply_numerators(candidate)
        if da != db or not np.array_equal(a, b):
            raise MdlMemberError("candidate escaped the exact integer resize fibre")

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
    "MdlMemberError",
    "MdlPolytopeMemberSolver",
    "MemberSolveResult",
    "ProxyCalibration",
    "fit_proxy",
    "lawref_manifest",
    "zlib9_bytes",
]
