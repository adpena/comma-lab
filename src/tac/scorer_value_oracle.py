# SPDX-License-Identifier: MIT
"""Typed, read-only facade over landed scorer-value producers.

The oracle does not measure, estimate, or rebuild any scorer value.  It binds
the rows of the DDM-366 dimension-completeness contract to existing producer
artifacts, rehashes the selected artifact at consumption time, and returns the
producer payload together with its lineage.

Large NPZ values are intentionally lazy.  Reading the descriptor is cheap and
SHA-checks the descriptor receipt.  :meth:`ScorerValueOracle.open_npz_member`
then SHA-checks the external NPZ bytes before returning the existing read-only
``ZIP_STORED`` memmap.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

if TYPE_CHECKING:
    import numpy as np

from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap
from tac.optimization.ddm_metric_custody_bundle import (
    ComponentId,
    MetricCustodyError,
    load_metric_custody_bundle,
)
from tac.repo_io import sha256_file

SCHEMA: Final = "ddm_scorer_value_oracle.v1"
COVERAGE_SCHEMA: Final = "ddm_scorer_value_oracle_coverage.v1"
PRODUCER_VALUE_SCHEMA: Final = "ddm_scorer_value_oracle_producer_value.v1"
CONTRACT_PATH: Final = ".omx/research/ddm_366_dimension_completeness_contract_20260724.md"


class OracleError(ValueError):
    """Base class for typed scorer-value oracle failures."""


class StaleProducerError(OracleError):
    """A producer artifact no longer matches its sealed lineage."""


class TypedGapError(OracleError):
    """A DDM-366 row has no machine-readable producer binding."""


class DimensionRow(StrEnum):
    """Exact ``Dim`` keys from the DDM-366 completeness table."""

    SUB_PIXEL_PLACEMENT = "sub-pixel placement (874-res, pre-R)"
    STEM_LATTICE_PHASE = "stem-lattice phase (stride-2)"
    RESIZE_KERNEL_SUPPORT_NULLITY = "resize-kernel support / nullity"
    ERF_NEIGHBORHOOD = "ERF neighborhood"
    CELL_LAGUERRE_ARGMAX_POLYTOPE = "cell (Laguerre/argmax polytope)"
    STRATUM_CLASS_HYPERPLANES = "stratum / class hyperplanes"
    FRAME_ROLES = "frame roles"
    PAIR_POSE_SCREW = "pair (pose screw, 6-of-12)"
    TEMPORAL_FLICKER = "temporal / flicker"
    CLIP_N600_STATIONARITY = "clip (n600 stationarity)"
    AMPLITUDE_UINT8_DEADZONE = "amplitude / uint8 deadzone"
    GAIN_SE_STATE_DEPENDENCE = "gain (SE state-dependence)"
    GAIN_NORMALIZATION_AFFINE = "gain (normalization affine)"
    FREQUENCY_R_PASSBAND = "frequency / R passband"
    YUV6_LUMA_PHASES = "YUV6 luma phases"
    CHROMA_POSE_NULL = "chroma pose-null"
    MARGIN_FISHER_SURROGATE = "margin (Fisher surrogate)"
    NULL_GAUGE_ENERGY = "null/gauge energy"
    POSE_DIMS_7_12 = "pose dims 7–12"
    RATE_ARCHIVE_BYTES_ONLY = "rate (archive bytes only)"
    SCORE_AXES_WEIGHTS = "score axes + weights"

    @classmethod
    def coerce(cls, value: DimensionRow | str) -> DimensionRow:
        """Accept an exact contract key or a stable enum member name."""

        if isinstance(value, cls):
            return value
        try:
            return cls(value)
        except ValueError:
            try:
                return cls[value]
            except KeyError as exc:
                raise OracleError(f"unknown DDM-366 dimension row: {value!r}") from exc


class CoverageStatus(StrEnum):
    WRAPPED = "WRAPPED"
    TYPED_GAP = "TYPED-GAP"


class FreshnessMode(StrEnum):
    FAIL_CLOSED = "FAIL_CLOSED"
    STALE_ADVISORY = "STALE_ADVISORY"


class FreshnessStatus(StrEnum):
    FRESH = "FRESH"
    STALE_ADVISORY = "STALE_ADVISORY"
    TYPED_GAP = "TYPED_GAP"
    NOT_CHECKED = "NOT_CHECKED"


class PayloadKind(StrEnum):
    JSON = "JSON"
    MS4D_BUNDLE = "MS4D_BUNDLE"


@dataclass(frozen=True, slots=True)
class ProducerLineage:
    """Expected and observed identity of one consumed producer artifact."""

    path: str
    role: str
    expected_sha256: str
    expected_bytes: int
    observed_sha256: str | None
    observed_bytes: int | None

    @property
    def fresh(self) -> bool:
        return (
            self.observed_sha256 == self.expected_sha256
            and self.observed_bytes == self.expected_bytes
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "role": self.role,
            "expected_sha256": self.expected_sha256,
            "expected_bytes": self.expected_bytes,
            "observed_sha256": self.observed_sha256,
            "observed_bytes": self.observed_bytes,
            "fresh": self.fresh,
        }


@dataclass(frozen=True, slots=True)
class RowBinding:
    """One DDM-366 row bound to an already-landed producer artifact."""

    row: DimensionRow
    producer: str
    path: str
    sha256: str
    bytes: int
    schema: str
    validity_horizon: str
    value_kind: str
    authority_scope: str
    selector: tuple[str | int, ...] = ()
    payload_kind: PayloadKind = PayloadKind.JSON
    ms4d_component: ComponentId | None = None

    def __post_init__(self) -> None:
        if len(self.sha256) != 64:
            raise ValueError(f"{self.row.value}: producer SHA-256 must have 64 hex characters")
        try:
            bytes.fromhex(self.sha256)
        except ValueError as exc:
            raise ValueError(f"{self.row.value}: producer SHA-256 must be hexadecimal") from exc
        if self.bytes < 1:
            raise ValueError(f"{self.row.value}: producer byte count must be positive")
        if (
            not self.producer
            or not self.path
            or not self.schema
            or not self.validity_horizon
            or not self.authority_scope
        ):
            raise ValueError(f"{self.row.value}: producer metadata must be nonempty")
        if (self.payload_kind is PayloadKind.MS4D_BUNDLE) != (
            self.ms4d_component is not None
        ):
            raise ValueError(
                f"{self.row.value}: MS4D payloads require exactly one component selector"
            )


@dataclass(frozen=True, slots=True)
class ProducerBinding:
    """One non-DDM-366 JSON producer admitted through the same freshness gate."""

    producer_id: str
    producer: str
    path: str
    sha256: str
    bytes: int
    schema: str
    validity_horizon: str
    authority_scope: str
    selector: tuple[str | int, ...] = ()

    def __post_init__(self) -> None:
        if (
            not self.producer_id
            or not self.producer
            or not self.path
            or not self.schema
            or not self.validity_horizon
            or not self.authority_scope
        ):
            raise ValueError("external producer binding metadata must be nonempty")
        if not isinstance(self.sha256, str) or len(self.sha256) != 64:
            raise ValueError(f"{self.producer_id}: producer SHA-256 must have 64 hex characters")
        try:
            bytes.fromhex(self.sha256)
        except ValueError as exc:
            raise ValueError(f"{self.producer_id}: producer SHA-256 must be hexadecimal") from exc
        if isinstance(self.bytes, bool) or not isinstance(self.bytes, int) or self.bytes < 1:
            raise ValueError(f"{self.producer_id}: producer byte count must be positive")


@dataclass(frozen=True, slots=True)
class ProducerValue:
    """Freshness-tagged value from a sealed producer outside the row contract."""

    producer_id: str
    producer: str
    value: Any
    lineage: tuple[ProducerLineage, ...]
    freshness: FreshnessStatus
    freshness_tag: str
    validity_horizon: str
    authority_scope: str

    def require_value(self) -> Any:
        if self.freshness is not FreshnessStatus.FRESH:
            raise StaleProducerError(
                f"{self.producer_id}: value is not fresh ({self.freshness.value})"
            )
        return self.value

    def to_dict(self, *, include_value: bool = True) -> dict[str, Any]:
        out = {
            "schema": PRODUCER_VALUE_SCHEMA,
            "producer_id": self.producer_id,
            "producer": self.producer,
            "lineage": [item.to_dict() for item in self.lineage],
            "freshness": self.freshness.value,
            "freshness_tag": self.freshness_tag,
            "validity_horizon": self.validity_horizon,
            "authority_scope": self.authority_scope,
        }
        if include_value:
            out["value"] = self.value
        return out


@dataclass(frozen=True, slots=True)
class TypedGap:
    row: DimensionRow
    producer_expected: str
    reason: str
    next_action: str

    def __post_init__(self) -> None:
        if not self.producer_expected or not self.reason or not self.next_action:
            raise ValueError(f"{self.row.value}: typed gap fields must be nonempty")


@dataclass(frozen=True, slots=True)
class OracleRow:
    """One typed value or typed gap returned by the facade."""

    row: DimensionRow
    coverage: CoverageStatus
    producer: str
    value_kind: str
    value: Any
    lineage: tuple[ProducerLineage, ...]
    freshness: FreshnessStatus
    freshness_tag: str
    validity_horizon: str
    authority_scope: str
    gap_reason: str | None = None
    next_action: str | None = None

    def require_value(self) -> Any:
        if self.coverage is CoverageStatus.TYPED_GAP:
            raise TypedGapError(f"{self.row.value}: {self.gap_reason}")
        if self.freshness is not FreshnessStatus.FRESH:
            raise StaleProducerError(
                f"{self.row.value}: value is not fresh ({self.freshness.value})"
            )
        return self.value

    def to_dict(self, *, include_value: bool = True) -> dict[str, Any]:
        out = {
            "schema": SCHEMA,
            "contract_path": CONTRACT_PATH,
            "row": self.row.value,
            "coverage": self.coverage.value,
            "producer": self.producer,
            "value_kind": self.value_kind,
            "lineage": [item.to_dict() for item in self.lineage],
            "freshness": self.freshness.value,
            "freshness_tag": self.freshness_tag,
            "validity_horizon": self.validity_horizon,
            "authority_scope": self.authority_scope,
            "gap_reason": self.gap_reason,
            "next_action": self.next_action,
        }
        if include_value:
            out["value"] = self.value
        return out


def _select(value: Any, selector: Sequence[str | int]) -> Any:
    current = value
    for part in selector:
        try:
            current = current[part]
        except (KeyError, IndexError, TypeError) as exc:
            raise OracleError(f"producer selector failed at {part!r}") from exc
    return current


def _is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


_TARGET_RECEIPT = (
    ".omx/research/ddm_v12_obligation_n600_20260722T161517Z/"
    "ddm_v12_obligation_search_n600_receipt.json"
)
_MS4D = (
    ".omx/research/ddm_ms4d_direct_metric_completion_20260724T155932Z/"
    "BUNDLE-COMPLETE.json"
)
_SOLVED_PLANES = ".omx/research/ddm_full_precision_target_planes_603_20260722T010130Z.json"
_PROTOTYPES = (
    ".omx/research/prereq_surfaces_flush_20260720/"
    "surface_2_rank4_prototype_bank.json"
)
_DM1 = (
    ".omx/research/ddm_dm1_25_row_solved_value_pricing_20260724T123443Z/"
    "ddm_dm1_25_row_solved_value_pricing_receipt.json"
)
_DM4_CONFIG = ".omx/research/configs/ddm_dm4_targeted_realization_cures_20260724.json"
_RESIZE = ".omx/research/null_compiler_full_kernel_20260720T163500Z.json"
_G4 = (
    ".omx/research/ddm_g4_spatial_stationarity_n600_20260722T212138Z/"
    "ddm_g4_spatial_stationarity_receipt.json"
)
_PF2 = (
    ".omx/research/ddm_ms5_pf2_bucket_assignment_20260724T044736Z/"
    "pf2_bucket_assignment_table.json"
)

_TARGET_ID = (
    "eab2ef2478fb07f6a3242781887442c3fc49e9c34e10bd73a93f25d9a0262f0a",
    1_157_303,
    "direct_description_v12_obligation_drain_receipt.v1",
)
_MS4D_ID = (
    "d670eff3dd01d61a24bdebedf045fa8cde2528953660dc6d1e64ba9c2fa94e25",
    2_799,
    "ddm_metric_custody_bundle.v1",
)
_SOLVED_PLANES_ID = (
    "a8d94f0f8338036fb3224a92078eff1f1fb5fd2eb598ed994a1f965b6561efb2",
    39_399,
    "direct_description_full_precision_target_planes.v1",
)
_PROTOTYPES_ID = (
    "bb1f79839c2e2ab8c41856ca0545bd4147d63deccbe0b1200e2ba65e5fa3501f",
    1_530,
    "rank4_valid_cell_prototypes_v1",
)
_DM1_ID = (
    "4c2fe77927e300e341d5ce9ce00ae8a37c58dbebbde8e5860fe514958990de28",
    77_687,
    "ddm_dm1_solved_value_pricing.v1",
)
_DM4_CONFIG_ID = (
    "60b681b3a016c3b8c6b94c159fa5437b5e1982ff2a9eb6421140679c5fed5633",
    5_272,
    "ddm_dm4_targeted_realization_cures_config.v1",
)
_RESIZE_ID = (
    "76af5e7f8d155363a6668b4ee7bca576ea448472ea0cd1e7f938577bb4adfd74",
    7_444,
    "resize_null_preimage_full_kernel_measurement.v1",
)
_G4_ID = (
    "bea555b95aeaa11f4209df5333010c41c5495dd789def2a4f7a2a91973f3408c",
    16_547,
    "ddm_g4_spatial_stationarity_receipt.v1",
)
_PF2_ID = (
    "20fa2b2ce2bd96b91c64d4e1342109dd7dab399d4769cd372dbf67fbcdf97d8d",
    1_426_698,
    "ddm_ms5_pf2_bucket_assignment_table.v1",
)


def _binding(
    row: DimensionRow,
    producer: str,
    path: str,
    identity: tuple[str, int, str],
    horizon: str,
    value_kind: str,
    authority_scope: str,
    selector: tuple[str | int, ...] = (),
    payload_kind: PayloadKind = PayloadKind.JSON,
    ms4d_component: ComponentId | None = None,
) -> RowBinding:
    return RowBinding(
        row=row,
        producer=producer,
        path=path,
        sha256=identity[0],
        bytes=identity[1],
        schema=identity[2],
        validity_horizon=horizon,
        value_kind=value_kind,
        authority_scope=authority_scope,
        selector=selector,
        payload_kind=payload_kind,
        ms4d_component=ms4d_component,
    )


_HASH_UNTIL_REPLACED = "content-hash valid until a newer sealed producer artifact lands"
_HASH_UNTIL_SCORER = "content-hash valid until frozen scorer or target-cache custody changes"
_HASH_UNTIL_RECEIVER = "content-hash valid until receiver, R geometry, or scorer custody changes"

DEFAULT_BINDINGS: Final[tuple[RowBinding, ...]] = (
    _binding(
        DimensionRow.SUB_PIXEL_PLACEMENT,
        "M2/#577 identified-optimal unrounded scorer planes",
        _SOLVED_PLANES,
        _SOLVED_PLANES_ID,
        _HASH_UNTIL_RECEIVER,
        "unrounded_scorer_reference_plane_manifest",
        "wraps the M2/#577 unrounded scorer-plane manifest; it does not claim a "
        "machine-readable #149 camera-grid placement law",
    ),
    _binding(
        DimensionRow.STEM_LATTICE_PHASE,
        "DM4 scorer-recursive support configuration",
        _DM4_CONFIG,
        _DM4_CONFIG_ID,
        _HASH_UNTIL_SCORER,
        "scorer_recursive_stem_support",
        "provides the sealed stride-2 stem lattice and scorer-recursive support rule",
        ("scorer_recursive_support",),
    ),
    _binding(
        DimensionRow.RESIZE_KERNEL_SUPPORT_NULLITY,
        "#580 exact separable resize kernel and projector",
        _RESIZE,
        _RESIZE_ID,
        _HASH_UNTIL_RECEIVER,
        "resize_kernel_support_nullity_receipt",
        "provides exact resize support, nullity, projector, and uint8 reachability custody",
    ),
    _binding(
        DimensionRow.ERF_NEIGHBORHOOD,
        "DM4 scorer-recursive support configuration",
        _DM4_CONFIG,
        _DM4_CONFIG_ID,
        _HASH_UNTIL_SCORER,
        "measured_erf_r50_support_config",
        "provides measured registered r50=85 px and its provenance; r90 remains outside "
        "this producer's authority",
        ("scorer_recursive_support",),
    ),
    _binding(
        DimensionRow.CELL_LAGUERRE_ARGMAX_POLYTOPE,
        "gt_n600 frozen SegNet target cache",
        _TARGET_RECEIPT,
        _TARGET_ID,
        _HASH_UNTIL_SCORER,
        "npz_member_descriptor:lstars",
        "provides SHA-bound access to the frozen n600 lstars partition; cache bytes are "
        "rehashed only when a member is opened",
        ("target_custody",),
    ),
    _binding(
        DimensionRow.STRATUM_CLASS_HYPERPLANES,
        "#583 frozen rank-4 prototype bank",
        _PROTOTYPES,
        _PROTOTYPES_ID,
        _HASH_UNTIL_SCORER,
        "rank4_head_hyperplanes_and_prototypes",
        "provides the frozen rank-4 valid-cell head prototypes and their source custody",
    ),
    _binding(
        DimensionRow.FRAME_ROLES,
        "M2/#577 identified-optimal unrounded scorer planes",
        _SOLVED_PLANES,
        _SOLVED_PLANES_ID,
        _HASH_UNTIL_SCORER,
        "paired_unrounded_reference_plane_manifest",
        "provides distinct y0/y1 unrounded scorer planes and paired pose6 source custody",
    ),
    _binding(
        DimensionRow.PAIR_POSE_SCREW,
        "MS4D pose quadratic and active tube",
        _MS4D,
        _MS4D_ID,
        _HASH_UNTIL_RECEIVER,
        "validated_ms4d_pose_metric_data",
        "provides the n600 six-scalar centers, low-rank pose quadratics, and active tubes",
        payload_kind=PayloadKind.MS4D_BUNDLE,
        ms4d_component=ComponentId.POSE_METRIC,
    ),
    _binding(
        DimensionRow.TEMPORAL_FLICKER,
        "G4 stationarity and flip-frequency maps",
        _G4,
        _G4_ID,
        _HASH_UNTIL_RECEIVER,
        "g4_stationarity_decomposition",
        "provides measured G4 temporal-class and flip-mass decomposition; realized "
        "receiver acceptance remains outside this artifact",
        ("summary", "stationarity_decomposition"),
    ),
    _binding(
        DimensionRow.CLIP_N600_STATIONARITY,
        "G4 stationarity and k-times amortization",
        _G4,
        _G4_ID,
        _HASH_UNTIL_RECEIVER,
        "g4_n600_summary",
        "provides n600 stationarity, concentration, and k-times amortization opportunity data",
        ("summary",),
    ),
    _binding(
        DimensionRow.AMPLITUDE_UINT8_DEADZONE,
        "MS4D realized second-order custody",
        _MS4D,
        _MS4D_ID,
        _HASH_UNTIL_RECEIVER,
        "validated_ms4d_composite_r_second_order_data",
        "provides direct scorer-intrinsic model Hessians and adjoint readbacks; its "
        "secant_status explicitly remains NOT_APPLICABLE without an actuator input",
        payload_kind=PayloadKind.MS4D_BUNDLE,
        ms4d_component=ComponentId.COMPOSITE_R_SECOND_ORDER,
    ),
    _binding(
        DimensionRow.GAIN_SE_STATE_DEPENDENCE,
        "DM1 L4 solved-value records",
        _DM1,
        _DM1_ID,
        _HASH_UNTIL_SCORER,
        "l4_solved_value_rows",
        "provides operating-point L4 solved-value demands and margins; it does not "
        "promote those records to a complete SE gain-state response curve",
        ("rows",),
    ),
    _binding(
        DimensionRow.MARGIN_FISHER_SURROGATE,
        "MS4D margin-Fisher custody",
        _MS4D,
        _MS4D_ID,
        _HASH_UNTIL_SCORER,
        "validated_ms4d_seg_metric_data",
        "provides all 1,200 PF2 rank-4 margin-Fisher Grams, spectra, normals, and "
        "lambda ranges",
        payload_kind=PayloadKind.MS4D_BUNDLE,
        ms4d_component=ComponentId.SEG_METRIC,
    ),
    _binding(
        DimensionRow.RATE_ARCHIVE_BYTES_ONLY,
        "MS5/MS6/RG3 PF2 assignment table",
        _PF2,
        _PF2_ID,
        _HASH_UNTIL_REPLACED,
        "pf2_1200_bucket_assignment_table",
        "provides the 1,200-way admission-state assignment table; exact archive-byte "
        "pricing remains the caller's realized-rate responsibility",
    ),
)

DEFAULT_GAPS: Final[tuple[TypedGap, ...]] = (
    TypedGap(
        DimensionRow.FREQUENCY_R_PASSBAND,
        "machine-readable measured R passband and along-tangent deficit artifact",
        "#580 seals spatial support/nullity but does not contain the frequency response "
        "or 3.2x along-tangent deficit",
        "seal the already-measured passband curve as a source-SHA-bound artifact",
    ),
    TypedGap(
        DimensionRow.YUV6_LUMA_PHASES,
        "machine-readable four-luma-phase scorer-target artifact",
        "the pose metric contains six-output targets and tubes, not the four named YUV6 "
        "luma phase channels",
        "seal the existing differentiable YUV6 phase law and source SHA as a producer artifact",
    ),
    TypedGap(
        DimensionRow.CHROMA_POSE_NULL,
        "machine-readable joint Seg/Pose chroma-null projector artifact",
        "#580 proves resize nullity only and cannot stand in for the Pose-null intersection",
        "land the existing joint-null projector receipt with Seg/Pose lineage",
    ),
    TypedGap(
        DimensionRow.NULL_GAUGE_ENERGY,
        "machine-readable joint null/gauge energy artifact",
        "#580 supplies ker(A), but no single sealed artifact carries both ker(A) and the "
        "52 percent head-gauge energy",
        "land a composite custody receipt over the existing null projector and gauge measurement",
    ),
    TypedGap(
        DimensionRow.POSE_DIMS_7_12,
        "machine-readable frozen scorer objective-dimension receipt",
        "MS4D intentionally contains the scored first six PoseNet outputs and does not "
        "encode the structural exclusion of outputs 7-12",
        "land an upstream-source-SHA-bound PoseNet objective-dimension receipt",
    ),
    TypedGap(
        DimensionRow.GAIN_NORMALIZATION_AFFINE,
        "machine-readable frozen normalization-affine receipt",
        "the 127.5/63.75 affine is source-inspected but has no dedicated sealed value artifact",
        "land a source-SHA-bound normalization-affine receipt; do not copy constants into this facade",
    ),
    TypedGap(
        DimensionRow.SCORE_AXES_WEIGHTS,
        "machine-readable upstream evaluate.py score-functional receipt",
        "the score formula is documented but the isolated worktree has no sealed evaluate.py value artifact",
        "land an upstream-evaluate-SHA-bound score-functional receipt; do not duplicate the formula here",
    ),
)


class ScorerValueOracle:
    """Read-only DDM-366 facade with freshness checked at every value read."""

    def __init__(
        self,
        repository_root: str | Path,
        *,
        bindings: Sequence[RowBinding] = DEFAULT_BINDINGS,
        gaps: Sequence[TypedGap] = DEFAULT_GAPS,
    ) -> None:
        root = Path(repository_root).expanduser().resolve(strict=True)
        if not root.is_dir():
            raise OracleError(f"repository root is not a directory: {root}")
        binding_map = {binding.row: binding for binding in bindings}
        gap_map = {gap.row: gap for gap in gaps}
        duplicates = set(binding_map).intersection(gap_map)
        if duplicates:
            raise OracleError(
                "rows cannot be both wrapped and typed gaps: "
                + ",".join(sorted(row.value for row in duplicates))
            )
        if len(binding_map) != len(bindings) or len(gap_map) != len(gaps):
            raise OracleError("duplicate row in scorer-value oracle registry")
        unknown = set(DimensionRow).difference(binding_map).difference(gap_map)
        if unknown:
            raise OracleError(
                "oracle registry omits DDM-366 rows: "
                + ",".join(sorted(row.value for row in unknown))
            )
        self.repository_root = root
        self._bindings = binding_map
        self._gaps = gap_map

    def _resolve(self, path: str) -> Path:
        candidate = Path(path).expanduser()
        return candidate if candidate.is_absolute() else self.repository_root / candidate

    def _lineage(self, binding: RowBinding) -> tuple[ProducerLineage, Path | None]:
        path = self._resolve(binding.path)
        observed_bytes: int | None = None
        observed_sha: str | None = None
        if path.is_file():
            observed_bytes = path.stat().st_size
            observed_sha = sha256_file(path)
        return (
            ProducerLineage(
                path=binding.path,
                role=binding.producer,
                expected_sha256=binding.sha256,
                expected_bytes=binding.bytes,
                observed_sha256=observed_sha,
                observed_bytes=observed_bytes,
            ),
        ), path if path.is_file() else None

    def _producer_lineage(
        self,
        binding: ProducerBinding,
    ) -> tuple[ProducerLineage, Path | None]:
        path = self._resolve(binding.path)
        observed_bytes: int | None = None
        observed_sha: str | None = None
        if path.is_file():
            observed_bytes = path.stat().st_size
            observed_sha = sha256_file(path)
        return (
            ProducerLineage(
                path=binding.path,
                role=binding.producer,
                expected_sha256=binding.sha256,
                expected_bytes=binding.bytes,
                observed_sha256=observed_sha,
                observed_bytes=observed_bytes,
            ),
        ), path if path.is_file() else None

    def _gap_row(self, row: DimensionRow) -> OracleRow:
        gap = self._gaps[row]
        return OracleRow(
            row=row,
            coverage=CoverageStatus.TYPED_GAP,
            producer=gap.producer_expected,
            value_kind="typed_gap",
            value=None,
            lineage=(),
            freshness=FreshnessStatus.TYPED_GAP,
            freshness_tag="[typed-gap]",
            validity_horizon="no validity until a sealed producer lands",
            authority_scope="no producer authority",
            gap_reason=gap.reason,
            next_action=gap.next_action,
        )

    @staticmethod
    def _stale_row(
        binding: RowBinding,
        lineage: tuple[ProducerLineage, ...],
    ) -> OracleRow:
        return OracleRow(
            row=binding.row,
            coverage=CoverageStatus.WRAPPED,
            producer=binding.producer,
            value_kind=binding.value_kind,
            value=None,
            lineage=lineage,
            freshness=FreshnessStatus.STALE_ADVISORY,
            freshness_tag="[stale-advisory]",
            validity_horizon=binding.validity_horizon,
            authority_scope=binding.authority_scope,
        )

    def read(
        self,
        row: DimensionRow | str,
        *,
        freshness_mode: FreshnessMode | str = FreshnessMode.FAIL_CLOSED,
    ) -> OracleRow:
        """Read one producer row, checking SHA, bytes, schema, and selector."""

        key = DimensionRow.coerce(row)
        try:
            mode = FreshnessMode(freshness_mode)
        except (TypeError, ValueError) as exc:
            raise OracleError(f"unknown freshness mode: {freshness_mode!r}") from exc
        if key in self._gaps:
            return self._gap_row(key)
        binding = self._bindings[key]
        lineage, path = self._lineage(binding)
        fresh = path is not None and all(item.fresh for item in lineage)
        if not fresh:
            detail = ",".join(
                f"{item.path}:expected={item.expected_sha256}/{item.expected_bytes}"
                f":observed={item.observed_sha256}/{item.observed_bytes}"
                for item in lineage
            )
            if mode is FreshnessMode.FAIL_CLOSED:
                raise StaleProducerError(f"{key.value}: stale producer artifact: {detail}")
            return self._stale_row(binding, lineage)
        assert path is not None
        try:
            if binding.payload_kind is PayloadKind.MS4D_BUNDLE:
                bundle = load_metric_custody_bundle(
                    path,
                    repository_root=self.repository_root,
                    require_complete=True,
                )
                assert binding.ms4d_component is not None
                component = bundle.components[binding.ms4d_component]
                if component.data_artifact is None:
                    raise StaleProducerError(
                        f"{key.value}: COMPLETE MS4D component omitted its data artifact"
                    )
                data_path = component.data_artifact.revalidate(
                    repository_root=self.repository_root
                )
                data_payload = json.loads(data_path.read_text(encoding="utf-8"))
                lineage = (
                    *lineage,
                    ProducerLineage(
                        path=component.data_artifact.path,
                        role=component.data_artifact.role,
                        expected_sha256=component.data_artifact.sha256,
                        expected_bytes=component.data_artifact.bytes,
                        observed_sha256=sha256_file(data_path),
                        observed_bytes=data_path.stat().st_size,
                    ),
                )
                value: Any = {
                    "bundle_id": bundle.bundle_id,
                    "status": bundle.status.value,
                    "component_id": binding.ms4d_component.value,
                    "component_status": component.status.value,
                    "data_artifact": component.data_artifact.to_dict(),
                    "data": data_payload,
                    "headline_flags": bundle.headline_flags(),
                }
            else:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if not _is_mapping(payload) or payload.get("schema") != binding.schema:
                    raise StaleProducerError(
                        f"{key.value}: producer schema drift "
                        f"{payload.get('schema') if _is_mapping(payload) else type(payload).__name__}"
                    )
                value = _select(payload, binding.selector)
        except (
            MetricCustodyError,
            OracleError,
            OSError,
            UnicodeError,
            json.JSONDecodeError,
        ) as exc:
            if mode is FreshnessMode.STALE_ADVISORY:
                return self._stale_row(binding, lineage)
            if isinstance(exc, StaleProducerError):
                raise
            raise StaleProducerError(
                f"{key.value}: producer validation failed: {exc}"
            ) from exc
        return OracleRow(
            row=key,
            coverage=CoverageStatus.WRAPPED,
            producer=binding.producer,
            value_kind=binding.value_kind,
            value=value,
            lineage=lineage,
            freshness=FreshnessStatus.FRESH,
            freshness_tag="[fresh]",
            validity_horizon=binding.validity_horizon,
            authority_scope=binding.authority_scope,
        )

    def require(self, row: DimensionRow | str) -> Any:
        """Return a fresh value or raise for both staleness and typed gaps."""

        return self.read(row).require_value()

    def read_json_producer(
        self,
        binding: ProducerBinding,
        *,
        freshness_mode: FreshnessMode | str = FreshnessMode.FAIL_CLOSED,
    ) -> ProducerValue:
        """Read a sealed JSON producer not represented by a DDM-366 row.

        Callers must supply the expected content identity.  This keeps campaign
        receipts and nested typed configs on the same fail-closed path as the
        row registry without pretending they are additional dimension rows.
        """

        try:
            mode = FreshnessMode(freshness_mode)
        except (TypeError, ValueError) as exc:
            raise OracleError(f"unknown freshness mode: {freshness_mode!r}") from exc
        lineage, path = self._producer_lineage(binding)
        if path is None or not all(item.fresh for item in lineage):
            detail = ",".join(
                f"{item.path}:expected={item.expected_sha256}/{item.expected_bytes}"
                f":observed={item.observed_sha256}/{item.observed_bytes}"
                for item in lineage
            )
            if mode is FreshnessMode.FAIL_CLOSED:
                raise StaleProducerError(
                    f"{binding.producer_id}: stale producer artifact: {detail}"
                )
            return ProducerValue(
                producer_id=binding.producer_id,
                producer=binding.producer,
                value=None,
                lineage=lineage,
                freshness=FreshnessStatus.STALE_ADVISORY,
                freshness_tag="[stale-advisory]",
                validity_horizon=binding.validity_horizon,
                authority_scope=binding.authority_scope,
            )
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not _is_mapping(payload) or payload.get("schema") != binding.schema:
                raise StaleProducerError(
                    f"{binding.producer_id}: producer schema drift "
                    f"{payload.get('schema') if _is_mapping(payload) else type(payload).__name__}"
                )
            value = _select(payload, binding.selector)
        except (
            OracleError,
            OSError,
            UnicodeError,
            json.JSONDecodeError,
        ) as exc:
            if mode is FreshnessMode.STALE_ADVISORY:
                return ProducerValue(
                    producer_id=binding.producer_id,
                    producer=binding.producer,
                    value=None,
                    lineage=lineage,
                    freshness=FreshnessStatus.STALE_ADVISORY,
                    freshness_tag="[stale-advisory]",
                    validity_horizon=binding.validity_horizon,
                    authority_scope=binding.authority_scope,
                )
            if isinstance(exc, StaleProducerError):
                raise
            raise StaleProducerError(
                f"{binding.producer_id}: producer validation failed: {exc}"
            ) from exc
        return ProducerValue(
            producer_id=binding.producer_id,
            producer=binding.producer,
            value=value,
            lineage=lineage,
            freshness=FreshnessStatus.FRESH,
            freshness_tag="[fresh]",
            validity_horizon=binding.validity_horizon,
            authority_scope=binding.authority_scope,
        )

    def require_json_producer(self, binding: ProducerBinding) -> Any:
        """Return one fresh external producer value or fail closed."""

        return self.read_json_producer(binding).require_value()

    def require_artifact(
        self,
        reference: Mapping[str, Any],
        *,
        role: str,
    ) -> ProducerLineage:
        """Rehash one nested artifact reference before a caller trusts it."""

        raw_path = reference.get("path")
        expected_sha = reference.get("sha256")
        expected_bytes = reference.get("bytes")
        if (
            not isinstance(raw_path, str)
            or not isinstance(expected_sha, str)
            or isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or expected_bytes < 1
            or not role
        ):
            raise OracleError("artifact reference requires path, SHA-256, bytes, and role")
        binding = ProducerBinding(
            producer_id=f"artifact:{role}",
            producer=role,
            path=raw_path,
            sha256=expected_sha,
            bytes=expected_bytes,
            schema="opaque_artifact",
            validity_horizon="content-hash valid until a replacement artifact lands",
            authority_scope="exact artifact bytes only",
        )
        lineage, path = self._producer_lineage(binding)
        if path is None or not all(item.fresh for item in lineage):
            item = lineage[0]
            raise StaleProducerError(
                f"{binding.producer_id}: stale artifact: expected="
                f"{item.expected_sha256}/{item.expected_bytes}:observed="
                f"{item.observed_sha256}/{item.observed_bytes}"
            )
        return lineage[0]

    def coverage_report(self, *, verify: bool = True) -> dict[str, Any]:
        """Report all contract rows as ``WRAPPED`` or ``TYPED-GAP``."""

        rows: list[dict[str, Any]] = []
        for row in DimensionRow:
            if row in self._gaps:
                value = self._gap_row(row)
            elif verify:
                value = self.read(row, freshness_mode=FreshnessMode.STALE_ADVISORY)
            else:
                binding = self._bindings[row]
                value = OracleRow(
                    row=row,
                    coverage=CoverageStatus.WRAPPED,
                    producer=binding.producer,
                    value_kind=binding.value_kind,
                    value=None,
                    lineage=(),
                    freshness=FreshnessStatus.NOT_CHECKED,
                    freshness_tag="[not-checked]",
                    validity_horizon=binding.validity_horizon,
                    authority_scope=binding.authority_scope,
                )
            rows.append(value.to_dict(include_value=False))
        counts = {
            status.value: sum(item["coverage"] == status.value for item in rows)
            for status in CoverageStatus
        }
        stale = sum(item["freshness"] == FreshnessStatus.STALE_ADVISORY.value for item in rows)
        return {
            "schema": COVERAGE_SCHEMA,
            "contract_path": CONTRACT_PATH,
            "row_count": len(rows),
            "counts": counts,
            "stale_advisory_count": stale,
            "verified_at_consumption": verify,
            "rows": rows,
        }

    def open_npz_member(
        self,
        row: DimensionRow | str,
        member: str,
    ) -> np.memmap:
        """Open a lazy target-cache member after rehashing the external NPZ."""

        result = self.read(row)
        value = result.require_value()
        if not isinstance(value, Mapping):
            raise OracleError(f"{result.row.value}: value is not an NPZ descriptor")
        path_value = value.get("cache_path") or value.get("path")
        expected_sha = value.get("cache_sha256") or value.get("sha256")
        expected_bytes = value.get("cache_bytes") or value.get("bytes")
        if (
            not isinstance(path_value, str)
            or not isinstance(expected_sha, str)
            or isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
        ):
            raise OracleError(f"{result.row.value}: malformed NPZ descriptor")
        path = self._resolve(path_value)
        if not path.is_file():
            raise StaleProducerError(f"{result.row.value}: external NPZ is missing: {path_value}")
        if path.stat().st_size != expected_bytes or sha256_file(path) != expected_sha:
            raise StaleProducerError(f"{result.row.value}: external NPZ lineage drift: {path_value}")
        return open_stored_npy_memmap(path, member)

    # Admission-state accessors.  Each remains a thin alias over ``read``.
    def argmax_partition(self) -> OracleRow:
        return self.read(DimensionRow.CELL_LAGUERRE_ARGMAX_POLYTOPE)

    def margin_fisher(self) -> OracleRow:
        return self.read(DimensionRow.MARGIN_FISHER_SURROGATE)

    def reference_planes(self) -> OracleRow:
        return self.read(DimensionRow.SUB_PIXEL_PLACEMENT)

    def pose_reference_and_tube(self) -> OracleRow:
        return self.read(DimensionRow.PAIR_POSE_SCREW)

    def head_hyperplanes(self) -> OracleRow:
        return self.read(DimensionRow.STRATUM_CLASS_HYPERPLANES)

    def l4_records(self) -> OracleRow:
        return self.read(DimensionRow.GAIN_SE_STATE_DEPENDENCE)

    def resize_support_nullity(self) -> OracleRow:
        return self.read(DimensionRow.RESIZE_KERNEL_SUPPORT_NULLITY)

    def realized_second_order(self) -> OracleRow:
        return self.read(DimensionRow.AMPLITUDE_UINT8_DEADZONE)

    def stationarity_maps(self) -> OracleRow:
        return self.read(DimensionRow.CLIP_N600_STATIONARITY)

    def bucket_assignments(self) -> OracleRow:
        return self.read(DimensionRow.RATE_ARCHIVE_BYTES_ONLY)


__all__ = [
    "CONTRACT_PATH",
    "COVERAGE_SCHEMA",
    "DEFAULT_BINDINGS",
    "DEFAULT_GAPS",
    "PRODUCER_VALUE_SCHEMA",
    "SCHEMA",
    "CoverageStatus",
    "DimensionRow",
    "FreshnessMode",
    "FreshnessStatus",
    "OracleError",
    "OracleRow",
    "PayloadKind",
    "ProducerBinding",
    "ProducerLineage",
    "ProducerValue",
    "RowBinding",
    "ScorerValueOracle",
    "StaleProducerError",
    "TypedGap",
    "TypedGapError",
]
