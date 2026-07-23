# SPDX-License-Identifier: MIT
"""Scorer-Derived Worldsheet Language v1 (SDWL1).

SDWL1 describes a deliberately bounded fact inventory derived from frozen
evaluator geometry.  It does not describe source pixels and is not a witness
or contest archive format.  Every numeric clause is encoded by the repository
left/up arithmetic coder; JSON is restricted to canonical lexicon and schema
sections.
"""

from __future__ import annotations

import hashlib
import json
import struct
import zlib
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any, Final

import numpy as np
from scipy import ndimage

from tac.optimization.arith_selfcomp_rate_coders import (
    RateCoderError,
    decode_spatial_context_arithmetic,
    encode_spatial_context_arithmetic,
)


class SDWL1Error(ValueError):
    """Raised when an SDWL1 inventory or wire object is invalid."""


class SubjectKind(StrEnum):
    """The complete SDWL1 subject type vocabulary."""

    PARTITION_CELL = "partition_cell"
    SEPARATRIX = "separatrix"
    LANE_CHART = "lane_chart"
    PAIR_SCREW = "pair_screw"
    RESIZE_RANGE_ATOM = "resize_range_atom"


class PredicateKind(StrEnum):
    """The complete SDWL1 predicate type vocabulary."""

    DECLARE = "declare"
    HOLD = "hold"
    DEFORM = "deform"
    TOPOLOGY_DELTA = "topology_delta"
    TRANSPORT = "transport"
    PROJECT_RANGE = "project_range"
    OMIT_KERNEL = "omit_kernel"


class ModifierKind(StrEnum):
    """The complete SDWL1 modifier type vocabulary."""

    STRATUM = "stratum"
    FRAME_ROLE = "frame_role"
    MARGIN_BAND = "margin_band"
    ERF_BAND = "erf_band"
    SCALE_BAND = "scale_band"
    CHROMA_PHASE = "chroma_phase"
    HEAD_NORMAL = "head_normal"
    ROAD_FRAME = "road_frame"


class SentenceLayout(StrEnum):
    """Numeric section layouts measured by the SDWL1 harness."""

    MONOLITHIC = "monolithic"
    TYPED_SECTION = "typed_section"
    STRATUM_SECTION = "stratum_section"


class TemporalMode(StrEnum):
    """Temporal treatment of the complete per-pair fact tensor."""

    ABSOLUTE = "absolute"
    CAUSAL_DELTA = "causal_delta"


@dataclass(frozen=True)
class DerivationEntry:
    """One named, repository-local derivation authority."""

    derivation_id: str
    statement: str
    sources: tuple[str, ...]
    evidence: str


@dataclass(frozen=True)
class SubjectSpec:
    """Immutable subject-language record."""

    kind: SubjectKind
    derivation_refs: tuple[str, ...]


@dataclass(frozen=True)
class PredicateSpec:
    """Immutable predicate-language record."""

    kind: PredicateKind
    derivation_refs: tuple[str, ...]


@dataclass(frozen=True)
class ModifierSpec:
    """Immutable modifier-language record."""

    kind: ModifierKind
    derivation_refs: tuple[str, ...]


@dataclass(frozen=True)
class PartitionCellFact:
    """One exact declared partition-cell record."""

    stratum: int
    area: int
    coordinate_sum_y: int
    coordinate_sum_x: int
    bbox_min_y: int
    bbox_min_x: int
    bbox_max_y_exclusive: int
    bbox_max_x_exclusive: int
    connected_component_count_4: int


@dataclass(frozen=True)
class SeparatrixFact:
    """One exact declared class-separatrix and margin-stratum record."""

    stratum: int
    horizontal_neighbor_cuts: int
    vertical_neighbor_cuts: int
    margin_band_counts: tuple[int, int, int, int]


@dataclass(frozen=True)
class PairScrewFact:
    """The six source float64 bit patterns for one pair."""

    float64_bit_patterns: tuple[int, int, int, int, int, int]


@dataclass(frozen=True)
class SentenceOptions:
    """Wire-layout and same-semantics MDL counterfactual switches."""

    layout: SentenceLayout
    temporal_mode: TemporalMode
    explicit_frame_indices: bool = False
    repeated_provenance: bool = False
    redundant_event_masks: bool = False
    split_topology_vocabulary: bool = False


CLASS_NAMES: Final = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
MARGIN_BANDS: Final = (
    (0.0, 0.1, "[0,0.1)"),
    (0.1, 0.5, "[0.1,0.5)"),
    (0.5, 1.0, "[0.5,1)"),
    (1.0, float("inf"), "[1,inf)"),
)
PAIR_RECORD_COUNT: Final = 11
PAIR_SCALAR_FACT_COUNT: Final = 76
SEMANTIC_ROWS: Final = 11
SEMANTIC_WIDTH: Final = 8
SEMANTIC_DTYPE: Final = np.dtype("<i8")
_SEMANTIC_ROW_WIDTHS: Final = (8, 8, 8, 8, 8, 6, 6, 6, 6, 6, 6)

DERIVATION_REGISTRY: Final = (
    DerivationEntry(
        "frozen_scorer_geometry",
        "SegNet consumes only the last RGB frame; PoseNet consumes the official two-frame YUV6 input.",
        ("upstream/modules.py", "upstream/frame_utils.py"),
        "DERIVED_FROM_FROZEN_SOURCE",
    ),
    DerivationEntry(
        "shared_bilinear_evaluator_map",
        "Both scorer paths share the bilinear evaluator map to 512x384.",
        ("upstream/modules.py", ".omx/research/ADVISORY_evaluator_video_geometry_20260710.md"),
        "DERIVED_FROM_FROZEN_SOURCE",
    ),
    DerivationEntry(
        "pose_chroma_2x2_box",
        "PoseNet chroma is a 2x2 box average, so sub-two-pixel chroma structure is invisible.",
        (
            "upstream/frame_utils.py",
            "src/tac/canonical_equations/posenet_luma_chroma_asymmetry_20260710.py",
        ),
        "DERIVED_AND_MEASURED",
    ),
    DerivationEntry(
        "rank4_laguerre_flip_distance_559",
        "The frozen SegNet head is rank four and its pairwise flip distance is margin divided by head-normal norm.",
        (
            "src/tac/canonical_equations/segnet_head_rank4_flipdist_20260715.py",
            ".omx/research/lane_channel_deep_refactorization_20260716.md",
        ),
        "DERIVED_AND_MEASURED",
    ),
    DerivationEntry(
        "resize_range_kernel_split_580",
        "The scorer-visible resize component is range(A); the measured kernel component is omitted.",
        (
            "src/tac/boundary_math/range_a_projection.py",
            ".omx/research/rep_mine_solved_binary_20260721T045500Z.md",
        ),
        "DERIVED_AND_MEASURED",
    ),
    DerivationEntry(
        "exact_resize_adjoint_391",
        "The exact bilinear resize adjoint maps scorer-plane obligations back to camera coordinates.",
        (
            "src/tac/optimization/solve_diff_operator_mining.py",
            ".omx/research/costate_organ_v2_exact_anchor_DAG_FEED_20260721T015900Z.md",
        ),
        "DERIVED_AND_TESTED",
    ),
    DerivationEntry(
        "scorer_erf_radius_bands",
        "Measured scorer effective-receptive-field radii define local and global influence bands.",
        (
            ".omx/research/segnet_recursive_fractal_factorization_20260715.md",
            "src/tac/canonical_equations/segnet_head_rank4_flipdist_20260715.py",
        ),
        "MEASURED",
    ),
    DerivationEntry(
        "fisher_margin_colocation",
        "Frozen-head categorical Fisher concentration is co-located with the winner-rival margin field.",
        (
            "src/tac/canonical_equations/deepmath_amortizing_argmax_laws_20260704.py",
            ".omx/research/deepmath_lens_infogeo_naturalgrad_20260704.md",
        ),
        "DERIVED_AND_MEASURED",
    ),
    DerivationEntry(
        "morse_smale_cells_separatrices",
        "Argmax strata form level-set/Morse-Smale cells with codimension-one separatrices.",
        (
            "src/tac/canonical_equations/witness_measured_findings_20260701.py",
            ".omx/research/projection_unification_and_eight_lenses_20260715.md",
        ),
        "DERIVED_AND_MEASURED",
    ),
    DerivationEntry(
        "se3_chasles_pair_screw",
        "One two-frame pair motion is one se(3)/Chasles screw object with six frozen outputs.",
        (
            ".omx/research/deepmath_lens_microlocal_se3_code_20260704.md",
            "src/tac/canonical_equations/partition_temporal_transport_amortization_20260715.py",
        ),
        "DERIVED",
    ),
    DerivationEntry(
        "power_diagram_cells_284_539",
        "The max-of-affine head induces weighted Laguerre/power-diagram cells.",
        (
            "src/tac/boundary_math/power_diagram_witness.py",
            "src/tac/canonical_equations/witness_measured_findings_20260701.py",
        ),
        "DERIVED_AND_MEASURED",
    ),
    DerivationEntry(
        "road_frame_geometry_145_325_326_327",
        "Road-frame homography, camera intrinsics/extrinsics, Lane polynomial basis, ego kinematics, horizon, and camera height define the road chart.",
        (
            "src/tac/boundary_math/lane_ground_factorization.py",
            "src/tac/optimization/predict_project_schema.py",
            ".omx/research/comma_openpilot_domain_tricks_20260619T035417Z.md",
        ),
        "DERIVED_AND_MEASURED",
    ),
    DerivationEntry(
        "polytope_kkt_tropical_whitney",
        "Margin polytopes, waterfill/KKT duality, tropical piecewise-linear structure, and Whitney bounds constrain admission dimension.",
        (
            "src/tac/optimization/direct_description_polytope_membership.py",
            "src/tac/canonical_equations/cgauge_parametrization_optima_20260711.py",
            "src/tac/canonical_equations/seg_rate_breakeven_and_head_gauge_laws_20260719.py",
        ),
        "DERIVED_AND_MEASURED",
    ),
)
_DERIVATIONS: Final = {entry.derivation_id: entry for entry in DERIVATION_REGISTRY}

SUBJECT_SPECS: Final = (
    SubjectSpec(
        SubjectKind.PARTITION_CELL,
        ("morse_smale_cells_separatrices", "power_diagram_cells_284_539"),
    ),
    SubjectSpec(
        SubjectKind.SEPARATRIX,
        ("morse_smale_cells_separatrices", "fisher_margin_colocation"),
    ),
    SubjectSpec(
        SubjectKind.LANE_CHART,
        ("road_frame_geometry_145_325_326_327", "rank4_laguerre_flip_distance_559"),
    ),
    SubjectSpec(SubjectKind.PAIR_SCREW, ("frozen_scorer_geometry", "se3_chasles_pair_screw")),
    SubjectSpec(
        SubjectKind.RESIZE_RANGE_ATOM,
        ("resize_range_kernel_split_580", "exact_resize_adjoint_391"),
    ),
)
PREDICATE_SPECS: Final = (
    PredicateSpec(PredicateKind.DECLARE, ("frozen_scorer_geometry",)),
    PredicateSpec(PredicateKind.HOLD, ("polytope_kkt_tropical_whitney",)),
    PredicateSpec(PredicateKind.DEFORM, ("morse_smale_cells_separatrices",)),
    PredicateSpec(PredicateKind.TOPOLOGY_DELTA, ("morse_smale_cells_separatrices",)),
    PredicateSpec(PredicateKind.TRANSPORT, ("se3_chasles_pair_screw",)),
    PredicateSpec(PredicateKind.PROJECT_RANGE, ("resize_range_kernel_split_580", "exact_resize_adjoint_391")),
    PredicateSpec(PredicateKind.OMIT_KERNEL, ("resize_range_kernel_split_580",)),
)
MODIFIER_SPECS: Final = (
    ModifierSpec(ModifierKind.STRATUM, ("morse_smale_cells_separatrices",)),
    ModifierSpec(ModifierKind.FRAME_ROLE, ("frozen_scorer_geometry",)),
    ModifierSpec(ModifierKind.MARGIN_BAND, ("fisher_margin_colocation",)),
    ModifierSpec(ModifierKind.ERF_BAND, ("scorer_erf_radius_bands",)),
    ModifierSpec(ModifierKind.SCALE_BAND, ("shared_bilinear_evaluator_map",)),
    ModifierSpec(ModifierKind.CHROMA_PHASE, ("pose_chroma_2x2_box",)),
    ModifierSpec(ModifierKind.HEAD_NORMAL, ("rank4_laguerre_flip_distance_559",)),
    ModifierSpec(ModifierKind.ROAD_FRAME, ("road_frame_geometry_145_325_326_327",)),
)


def validate_derivation_coverage() -> None:
    """Fail if any grammar element lacks a known named derivation."""

    if len(_DERIVATIONS) != len(DERIVATION_REGISTRY):
        raise SDWL1Error("derivation registry IDs must be unique")
    for element in (*SUBJECT_SPECS, *PREDICATE_SPECS, *MODIFIER_SPECS):
        if not element.derivation_refs:
            raise SDWL1Error(f"{element.kind.value} has no named derivation")
        unknown = sorted(set(element.derivation_refs) - set(_DERIVATIONS))
        if unknown:
            raise SDWL1Error(f"{element.kind.value} references unknown derivations: {unknown}")


validate_derivation_coverage()


def _semantic_sha256(tensor: np.ndarray) -> str:
    canonical = np.ascontiguousarray(tensor, dtype=SEMANTIC_DTYPE)
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class FactInventory:
    """The complete exact declared fact tensor."""

    tensor: np.ndarray
    source_height: int
    source_width: int
    semantic_sha256: str

    def __post_init__(self) -> None:
        tensor = np.ascontiguousarray(self.tensor, dtype=SEMANTIC_DTYPE)
        if tensor.ndim != 3 or tensor.shape[1:] != (SEMANTIC_ROWS, SEMANTIC_WIDTH):
            raise SDWL1Error(f"semantic tensor must be (pairs,{SEMANTIC_ROWS},{SEMANTIC_WIDTH}); got {tensor.shape}")
        if tensor.shape[0] <= 0:
            raise SDWL1Error("semantic tensor must describe at least one pair")
        if np.any(tensor[:, 5:10, 6:]) or np.any(tensor[:, 10, 6:]):
            raise SDWL1Error("noncanonical nonzero values in semantic padding columns")
        if int(self.source_height) <= 0 or int(self.source_width) <= 0:
            raise SDWL1Error("source geometry must be positive")
        digest = _semantic_sha256(tensor)
        if digest != self.semantic_sha256:
            raise SDWL1Error(f"semantic tensor hash {digest} != declared {self.semantic_sha256}")
        tensor.setflags(write=False)
        object.__setattr__(self, "tensor", tensor)

    @property
    def pair_count(self) -> int:
        return int(self.tensor.shape[0])

    @property
    def described_fact_count(self) -> int:
        """Return the number of described, non-padding scalar facts."""

        return self.described_scalar_fact_count

    @property
    def described_record_count(self) -> int:
        """Return the number of typed subject records."""

        return self.pair_count * PAIR_RECORD_COUNT

    @property
    def described_scalar_fact_count(self) -> int:
        """Return the number of non-padding scalar facts."""

        return self.pair_count * PAIR_SCALAR_FACT_COUNT


@dataclass(frozen=True)
class ProductionCounts:
    """Measured deterministic grammar-production counts."""

    subjects: tuple[tuple[str, int], ...]
    predicates: tuple[tuple[str, int], ...]
    modifiers: tuple[tuple[str, int], ...]
    topology_births: int
    topology_deaths: int

    def as_dict(self) -> dict[str, Any]:
        """Return a canonical-JSON-ready representation."""

        return {
            "modifiers": dict(self.modifiers),
            "predicates": dict(self.predicates),
            "subjects": dict(self.subjects),
            "topology_births": self.topology_births,
            "topology_deaths": self.topology_deaths,
        }


def measure_production_counts(tensor: np.ndarray) -> ProductionCounts:
    """Infer uncharged grammar productions from exact temporal record values."""

    values = np.ascontiguousarray(tensor, dtype=SEMANTIC_DTYPE)
    if values.ndim != 3 or values.shape[1:] != (SEMANTIC_ROWS, SEMANTIC_WIDTH):
        raise SDWL1Error("production counts require the canonical semantic tensor shape")
    pair_count = int(values.shape[0])
    predicate_counts = {kind.value: 0 for kind in PredicateKind}
    topology_births = 0
    topology_deaths = 0
    if pair_count:
        predicate_counts[PredicateKind.DECLARE.value] = PAIR_RECORD_COUNT
    for pair_index in range(1, pair_count):
        previous = values[pair_index - 1]
        current = values[pair_index]
        for row, width in enumerate(_SEMANTIC_ROW_WIDTHS):
            if np.array_equal(previous[row, :width], current[row, :width]):
                predicate_counts[PredicateKind.HOLD.value] += 1
            elif row < 5 and previous[row, 7] != current[row, 7]:
                predicate_counts[PredicateKind.TOPOLOGY_DELTA.value] += 1
                if current[row, 7] > previous[row, 7]:
                    topology_births += 1
                else:
                    topology_deaths += 1
            elif row == 10:
                predicate_counts[PredicateKind.TRANSPORT.value] += 1
            else:
                predicate_counts[PredicateKind.DEFORM.value] += 1
    if sum(predicate_counts.values()) != pair_count * PAIR_RECORD_COUNT:
        raise SDWL1Error("predicate production accounting does not cover every typed record")
    subject_counts = {kind.value: 0 for kind in SubjectKind}
    subject_counts[SubjectKind.PARTITION_CELL.value] = pair_count * 5
    subject_counts[SubjectKind.SEPARATRIX.value] = pair_count * 5
    subject_counts[SubjectKind.PAIR_SCREW.value] = pair_count
    modifier_counts = {kind.value: 0 for kind in ModifierKind}
    modifier_counts[ModifierKind.STRATUM.value] = pair_count * 10
    modifier_counts[ModifierKind.FRAME_ROLE.value] = pair_count * PAIR_RECORD_COUNT
    modifier_counts[ModifierKind.MARGIN_BAND.value] = pair_count * 5 * len(MARGIN_BANDS)
    return ProductionCounts(
        subjects=tuple((kind.value, subject_counts[kind.value]) for kind in SubjectKind),
        predicates=tuple((kind.value, predicate_counts[kind.value]) for kind in PredicateKind),
        modifiers=tuple((kind.value, modifier_counts[kind.value]) for kind in ModifierKind),
        topology_births=topology_births,
        topology_deaths=topology_deaths,
    )


_CONNECTIVITY_4: Final = np.array(
    [[False, True, False], [True, True, True], [False, True, False]],
    dtype=bool,
)


def _cell_fact(mask: np.ndarray, stratum: int) -> PartitionCellFact:
    ys, xs = np.nonzero(mask)
    area = int(ys.size)
    if area:
        min_y = int(ys.min())
        min_x = int(xs.min())
        max_y = int(ys.max()) + 1
        max_x = int(xs.max()) + 1
        component_count = int(ndimage.label(mask, structure=_CONNECTIVITY_4)[1])
    else:
        min_y = min_x = max_y = max_x = -1
        component_count = 0
    return PartitionCellFact(
        stratum=stratum,
        area=area,
        coordinate_sum_y=int(ys.sum(dtype=np.int64)),
        coordinate_sum_x=int(xs.sum(dtype=np.int64)),
        bbox_min_y=min_y,
        bbox_min_x=min_x,
        bbox_max_y_exclusive=max_y,
        bbox_max_x_exclusive=max_x,
        connected_component_count_4=component_count,
    )


def _separatrix_fact(mask: np.ndarray, margin: np.ndarray, stratum: int) -> SeparatrixFact:
    horizontal = int(np.count_nonzero(mask[:, 1:] != mask[:, :-1]))
    vertical = int(np.count_nonzero(mask[1:, :] != mask[:-1, :]))
    counts = tuple(int(np.count_nonzero(mask & (margin >= low) & (margin < high))) for low, high, _name in MARGIN_BANDS)
    return SeparatrixFact(
        stratum=stratum,
        horizontal_neighbor_cuts=horizontal,
        vertical_neighbor_cuts=vertical,
        margin_band_counts=counts,  # type: ignore[arg-type]
    )


def _float64_words_as_int64(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values)
    if source.dtype.kind != "f" or source.dtype.itemsize != 8:
        raise SDWL1Error(f"pair screw source must be float64; got {source.dtype}")
    little = np.ascontiguousarray(source, dtype="<f8")
    return little.view("<u8").view("<i8")


def extract_fact_inventory(
    lstars: np.ndarray,
    margins: np.ndarray,
    gt_poses: np.ndarray,
) -> FactInventory:
    """Extract all and only the declared SDWL1 facts from frozen arrays."""

    labels = np.asarray(lstars)
    margin_values = np.asarray(margins)
    poses = np.asarray(gt_poses)
    if labels.ndim != 3:
        raise SDWL1Error(f"lstars must be (pairs,height,width); got {labels.shape}")
    if margin_values.shape != labels.shape:
        raise SDWL1Error(f"margins shape {margin_values.shape} != lstars shape {labels.shape}")
    if poses.shape != (labels.shape[0], 6):
        raise SDWL1Error(f"gt_poses shape {poses.shape} != {(labels.shape[0], 6)}")
    if not np.issubdtype(labels.dtype, np.integer):
        raise SDWL1Error(f"lstars must be integer; got {labels.dtype}")
    if not np.issubdtype(margin_values.dtype, np.floating):
        raise SDWL1Error(f"margins must be floating; got {margin_values.dtype}")
    if poses.dtype.kind != "f" or poses.dtype.itemsize != 8:
        raise SDWL1Error(f"gt_poses must preserve source float64 values; got {poses.dtype}")
    if labels.size and (int(labels.min()) < 0 or int(labels.max()) >= len(CLASS_NAMES)):
        raise SDWL1Error("lstars values must be canonical class IDs 0..4")
    if not np.isfinite(margin_values).all() or np.any(margin_values < 0):
        raise SDWL1Error("margins must be finite and nonnegative")
    if not np.isfinite(poses).all():
        raise SDWL1Error("gt_poses must be finite")

    pair_count, height, width = labels.shape
    tensor = np.zeros((pair_count, SEMANTIC_ROWS, SEMANTIC_WIDTH), dtype=SEMANTIC_DTYPE)
    screw_words = _float64_words_as_int64(poses)
    for pair_index in range(pair_count):
        frame = labels[pair_index]
        pair_margin = margin_values[pair_index]
        for stratum in range(len(CLASS_NAMES)):
            mask = frame == stratum
            cell = _cell_fact(mask, stratum)
            tensor[pair_index, stratum] = (
                cell.area,
                cell.coordinate_sum_y,
                cell.coordinate_sum_x,
                cell.bbox_min_y,
                cell.bbox_min_x,
                cell.bbox_max_y_exclusive,
                cell.bbox_max_x_exclusive,
                cell.connected_component_count_4,
            )
            separatrix = _separatrix_fact(mask, pair_margin, stratum)
            tensor[pair_index, 5 + stratum, :6] = (
                separatrix.horizontal_neighbor_cuts,
                separatrix.vertical_neighbor_cuts,
                *separatrix.margin_band_counts,
            )
        tensor[pair_index, 10, :6] = screw_words[pair_index]
    return FactInventory(
        tensor=tensor,
        source_height=int(height),
        source_width=int(width),
        semantic_sha256=_semantic_sha256(tensor),
    )


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize finite JSON in the one admitted canonical form."""

    try:
        return json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SDWL1Error("value is not finite canonical JSON") from exc


def _parse_canonical_json(payload: bytes, name: str) -> dict[str, Any]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SDWL1Error(f"{name} is malformed canonical JSON") from exc
    if not isinstance(value, dict):
        raise SDWL1Error(f"{name} must be a canonical JSON object")
    if canonical_json_bytes(value) != payload:
        raise SDWL1Error(f"{name} is not canonical JSON")
    return value


def _spec_row(
    spec: SubjectSpec | PredicateSpec | ModifierSpec,
    use_count: int,
) -> dict[str, Any]:
    return {
        "name": spec.kind.value,
        "provenance": list(spec.derivation_refs),
        "use_count": use_count,
    }


def _canonical_provenance_manifest() -> dict[str, Any]:
    def provenance_row(
        spec: SubjectSpec | PredicateSpec | ModifierSpec,
    ) -> dict[str, Any]:
        return {
            "name": spec.kind.value,
            "provenance": list(spec.derivation_refs),
        }

    return {
        "derivation_registry": [asdict(entry) for entry in DERIVATION_REGISTRY],
        "modifiers": [provenance_row(spec) for spec in MODIFIER_SPECS],
        "predicates": [provenance_row(spec) for spec in PREDICATE_SPECS],
        "schema": "sdwl1.provenance.v1",
        "subjects": [provenance_row(spec) for spec in SUBJECT_SPECS],
    }


def canonical_provenance_digest() -> bytes:
    """Return the fixed digest of the complete named derivation surface."""

    return hashlib.sha256(canonical_json_bytes(_canonical_provenance_manifest())).digest()


def build_lexicon(inventory: FactInventory, options: SentenceOptions) -> dict[str, Any]:
    """Build the pruned lexicon for the admitted fact inventory."""

    measured = measure_production_counts(inventory.tensor)
    subject_counts = dict(measured.subjects)
    predicate_counts = dict(measured.predicates)
    modifier_counts = dict(measured.modifiers)
    subjects = [
        _spec_row(spec, subject_counts.get(spec.kind.value, 0))
        for spec in SUBJECT_SPECS
        if subject_counts.get(spec.kind.value, 0)
    ]
    predicates = [
        _spec_row(spec, predicate_counts.get(spec.kind.value, 0))
        for spec in PREDICATE_SPECS
        if predicate_counts.get(spec.kind.value, 0)
    ]
    modifiers = [
        _spec_row(spec, modifier_counts.get(spec.kind.value, 0))
        for spec in MODIFIER_SPECS
        if modifier_counts.get(spec.kind.value, 0)
    ]
    if options.split_topology_vocabulary:
        predicates = [row for row in predicates if row["name"] != PredicateKind.TOPOLOGY_DELTA.value]
        predicates.extend(
            [
                {
                    "name": "topology_birth",
                    "provenance": ["morse_smale_cells_separatrices"],
                    "use_count": measured.topology_births,
                },
                {
                    "name": "topology_death",
                    "provenance": ["morse_smale_cells_separatrices"],
                    "use_count": measured.topology_deaths,
                },
            ]
        )
    result: dict[str, Any] = {
        "derivation_registry": [asdict(entry) for entry in DERIVATION_REGISTRY],
        "modifiers": modifiers,
        "predicates": predicates,
        "schema": "sdwl1.lexicon.v1",
        "subjects": subjects,
        "zero_use_vocabulary_pruned": not options.split_topology_vocabulary,
    }
    return result


_PACKET_MAGIC: Final = b"SDWL1PK\x00"
_PACKET_VERSION: Final = 1
_PACKET_HEADER: Final = struct.Struct("<8sHHQ32s")
_SECTION_HEADER: Final = struct.Struct("<4sQ32s")
_COLLECTION_MAGIC: Final = b"SDWL1IC\x00"
_COLLECTION_VERSION: Final = 1
_COLLECTION_HEADER: Final = struct.Struct("<8sHQQ32s32s")
_DESCRIPTION_HEADER: Final = struct.Struct("<Q32s")
_KNOWN_TAGS: Final = frozenset(
    {
        b"LEXJ",
        b"SCHJ",
        b"MONO",
        b"CELL",
        b"SEPR",
        b"SCRW",
        b"S000",
        b"S001",
        b"S002",
        b"S003",
        b"S004",
        b"FIDX",
        b"EVNT",
        b"PROV",
    }
)


def _frame_sections(sections: list[tuple[bytes, bytes]]) -> bytes:
    seen: set[bytes] = set()
    framed: list[bytes] = []
    for tag, payload in sections:
        if tag not in _KNOWN_TAGS:
            raise SDWL1Error(f"unknown section tag {tag!r}")
        if tag in seen:
            raise SDWL1Error(f"duplicate section tag {tag!r}")
        seen.add(tag)
        framed.append(_SECTION_HEADER.pack(tag, len(payload), hashlib.sha256(payload).digest()) + payload)
    body = b"".join(framed)
    return (
        _PACKET_HEADER.pack(
            _PACKET_MAGIC,
            _PACKET_VERSION,
            len(sections),
            len(body),
            hashlib.sha256(body).digest(),
        )
        + body
    )


def _parse_sections(packet: bytes) -> list[tuple[bytes, bytes]]:
    if not isinstance(packet, bytes):
        raise SDWL1Error("SDWL1 packet must be immutable bytes")
    if len(packet) < _PACKET_HEADER.size:
        raise SDWL1Error("truncated SDWL1 packet header")
    magic, version, section_count, body_size, body_sha = _PACKET_HEADER.unpack_from(packet)
    if magic != _PACKET_MAGIC or version != _PACKET_VERSION:
        raise SDWL1Error("unknown SDWL1 packet magic or version")
    expected_size = _PACKET_HEADER.size + int(body_size)
    if len(packet) != expected_size:
        relation = "truncated" if len(packet) < expected_size else "trailing"
        raise SDWL1Error(f"{relation} bytes outside SDWL1 packet")
    body = packet[_PACKET_HEADER.size :]
    if hashlib.sha256(body).digest() != body_sha:
        raise SDWL1Error("SDWL1 packet body hash drift")
    offset = 0
    sections: list[tuple[bytes, bytes]] = []
    seen: set[bytes] = set()
    for _index in range(int(section_count)):
        if len(body) - offset < _SECTION_HEADER.size:
            raise SDWL1Error("truncated SDWL1 section header")
        tag, payload_size, payload_sha = _SECTION_HEADER.unpack_from(body, offset)
        offset += _SECTION_HEADER.size
        if tag not in _KNOWN_TAGS:
            raise SDWL1Error(f"unknown SDWL1 section tag {tag!r}")
        if tag in seen:
            raise SDWL1Error(f"duplicate SDWL1 section tag {tag!r}")
        seen.add(tag)
        stop = offset + int(payload_size)
        if stop > len(body):
            raise SDWL1Error(f"truncated SDWL1 section {tag!r}")
        payload = body[offset:stop]
        offset = stop
        if hashlib.sha256(payload).digest() != payload_sha:
            raise SDWL1Error(f"SDWL1 section {tag!r} hash drift")
        sections.append((tag, payload))
    if offset != len(body):
        raise SDWL1Error("trailing bytes after framed SDWL1 sections")
    return sections


def _temporal_encode(tensor: np.ndarray, mode: TemporalMode) -> np.ndarray:
    source = np.ascontiguousarray(tensor, dtype=SEMANTIC_DTYPE)
    if mode is TemporalMode.ABSOLUTE or source.shape[0] <= 1:
        return source.copy()
    encoded = source.copy()
    np.subtract(source[1:, :10], source[:-1, :10], out=encoded[1:, :10], dtype=np.int64)
    source_pose_u64 = source[:, 10:11, :6].view("<u8")
    encoded_pose_u64 = encoded[:, 10:11, :6].view("<u8")
    np.subtract(
        source_pose_u64[1:],
        source_pose_u64[:-1],
        out=encoded_pose_u64[1:],
        dtype=np.uint64,
    )
    return encoded


def _temporal_decode(tensor: np.ndarray, mode: TemporalMode) -> np.ndarray:
    encoded = np.ascontiguousarray(tensor, dtype=SEMANTIC_DTYPE)
    if mode is TemporalMode.ABSOLUTE or encoded.shape[0] <= 1:
        return encoded.copy()
    decoded = encoded.copy()
    decoded[:, :10] = np.add.accumulate(encoded[:, :10], axis=0, dtype=np.int64)
    encoded_pose_u64 = encoded[:, 10:11, :6].view("<u8")
    decoded_pose_u64 = decoded[:, 10:11, :6].view("<u8")
    decoded_pose_u64[0] = encoded_pose_u64[0]
    for pair_index in range(1, encoded.shape[0]):
        np.add(
            decoded_pose_u64[pair_index - 1],
            encoded_pose_u64[pair_index],
            out=decoded_pose_u64[pair_index],
            dtype=np.uint64,
        )
    return decoded


def _causal_event_mask(tensor: np.ndarray) -> np.ndarray:
    """Return the redundant record-event mask derived from causal deltas."""

    causal = _temporal_encode(tensor, TemporalMode.CAUSAL_DELTA)
    return np.any(causal != 0, axis=2, keepdims=True).astype(np.int8)


def _numeric_arrays(encoded: np.ndarray, layout: SentenceLayout) -> list[tuple[bytes, np.ndarray]]:
    if layout is SentenceLayout.MONOLITHIC:
        return [(b"MONO", encoded)]
    if layout is SentenceLayout.TYPED_SECTION:
        return [
            (b"CELL", encoded[:, :5, :]),
            (b"SEPR", encoded[:, 5:10, :]),
            (b"SCRW", encoded[:, 10:11, :]),
        ]
    result = [
        (f"S{stratum:03d}".encode("ascii"), encoded[:, (stratum, 5 + stratum), :])
        for stratum in range(len(CLASS_NAMES))
    ]
    result.append((b"SCRW", encoded[:, 10:11, :]))
    return result


def _numeric_tags(layout: SentenceLayout) -> list[bytes]:
    if layout is SentenceLayout.MONOLITHIC:
        tags = [b"MONO"]
    elif layout is SentenceLayout.TYPED_SECTION:
        tags = [b"CELL", b"SEPR", b"SCRW"]
    else:
        tags = [b"S000", b"S001", b"S002", b"S003", b"S004", b"SCRW"]
    return tags


def _expected_section_tags(options: SentenceOptions) -> list[bytes]:
    tags = [b"LEXJ", b"SCHJ"]
    if options.repeated_provenance:
        tags.append(b"PROV")
    tags.extend(_numeric_tags(options.layout))
    if options.explicit_frame_indices:
        tags.append(b"FIDX")
    if options.redundant_event_masks:
        tags.append(b"EVNT")
    return tags


def _schema(inventory: FactInventory, options: SentenceOptions, lexicon_sha256: str) -> dict[str, Any]:
    production_counts = measure_production_counts(inventory.tensor)
    return {
        "bbox_convention": "half_open; empty=(-1,-1,-1,-1)",
        "cell_columns": [
            "area",
            "coordinate_sum_y",
            "coordinate_sum_x",
            "bbox_min_y",
            "bbox_min_x",
            "bbox_max_y_exclusive",
            "bbox_max_x_exclusive",
            "connected_component_count_4",
        ],
        "class_names": list(CLASS_NAMES),
        "counterfactuals": {
            "explicit_frame_indices": options.explicit_frame_indices,
            "redundant_event_masks": options.redundant_event_masks,
            "repeated_provenance": options.repeated_provenance,
            "split_topology_vocabulary": options.split_topology_vocabulary,
        },
        "described_fact_count": inventory.described_scalar_fact_count,
        "described_fraction": 1,
        "described_record_count": inventory.described_record_count,
        "described_scalar_fact_count": inventory.described_scalar_fact_count,
        "layout": options.layout.value,
        "lexicon_sha256": lexicon_sha256,
        "margin_bands": [name for _low, _high, name in MARGIN_BANDS],
        "numeric_section_tags": [tag.decode("ascii") for tag in _numeric_tags(options.layout)],
        "pair_count": inventory.pair_count,
        "pair_record_count": PAIR_RECORD_COUNT,
        "pair_scalar_fact_count": PAIR_SCALAR_FACT_COUNT,
        "production_counts": production_counts.as_dict(),
        "provenance_digest_sha256": canonical_provenance_digest().hex(),
        "schema": "sdwl1.subject_schema.v1",
        "semantic_dtype": "<i8",
        "semantic_sha256": inventory.semantic_sha256,
        "semantic_shape": list(inventory.tensor.shape),
        "separatrix_columns": [
            "horizontal_neighbor_cuts",
            "vertical_neighbor_cuts",
            "margin_[0,0.1)",
            "margin_[0.1,0.5)",
            "margin_[0.5,1)",
            "margin_[1,inf)",
            "padding_zero_0",
            "padding_zero_1",
        ],
        "source_geometry": [inventory.source_height, inventory.source_width],
        "screw_columns": [
            "pose0_f64_bits_as_i64",
            "pose1_f64_bits_as_i64",
            "pose2_f64_bits_as_i64",
            "pose3_f64_bits_as_i64",
            "pose4_f64_bits_as_i64",
            "pose5_f64_bits_as_i64",
            "padding_zero_0",
            "padding_zero_1",
        ],
        "temporal_mode": options.temporal_mode.value,
    }


def serialize_sentence(inventory: FactInventory, options: SentenceOptions) -> bytes:
    """Serialize one complete whole-clip SDWL1 sentence."""

    lexicon_payload = canonical_json_bytes(build_lexicon(inventory, options))
    schema_payload = canonical_json_bytes(_schema(inventory, options, hashlib.sha256(lexicon_payload).hexdigest()))
    encoded = _temporal_encode(inventory.tensor, options.temporal_mode)
    sections: list[tuple[bytes, bytes]] = [(b"LEXJ", lexicon_payload), (b"SCHJ", schema_payload)]
    if options.repeated_provenance:
        sections.append((b"PROV", canonical_provenance_digest() * inventory.pair_count))
    for tag, array in _numeric_arrays(encoded, options.layout):
        sections.append((tag, encode_spatial_context_arithmetic(array)))
    if options.explicit_frame_indices:
        indices = np.arange(inventory.pair_count, dtype=np.int64).reshape(-1, 1, 1)
        sections.append((b"FIDX", encode_spatial_context_arithmetic(indices)))
    if options.redundant_event_masks:
        event_mask = _causal_event_mask(inventory.tensor)
        sections.append((b"EVNT", encode_spatial_context_arithmetic(event_mask)))
    return _frame_sections(sections)


def _options_from_schema(schema: dict[str, Any]) -> SentenceOptions:
    counterfactuals = schema.get("counterfactuals")
    if not isinstance(counterfactuals, dict) or set(counterfactuals) != {
        "explicit_frame_indices",
        "redundant_event_masks",
        "repeated_provenance",
        "split_topology_vocabulary",
    }:
        raise SDWL1Error("invalid SDWL1 counterfactual schema")
    if any(type(value) is not bool for value in counterfactuals.values()):
        raise SDWL1Error("SDWL1 counterfactual switches must be booleans")
    try:
        return SentenceOptions(
            layout=SentenceLayout(schema["layout"]),
            temporal_mode=TemporalMode(schema["temporal_mode"]),
            explicit_frame_indices=counterfactuals["explicit_frame_indices"],
            repeated_provenance=counterfactuals["repeated_provenance"],
            redundant_event_masks=counterfactuals["redundant_event_masks"],
            split_topology_vocabulary=counterfactuals["split_topology_vocabulary"],
        )
    except (KeyError, ValueError) as exc:
        raise SDWL1Error("unknown SDWL1 layout or temporal mode") from exc


def _decode_numeric(payload: bytes, tag: bytes) -> np.ndarray:
    try:
        return decode_spatial_context_arithmetic(payload)
    except RateCoderError as exc:
        raise SDWL1Error(f"invalid or noncanonical arithmetic stream in {tag!r}") from exc


def _reassemble_numeric(
    payloads: dict[bytes, bytes],
    options: SentenceOptions,
    pair_count: int,
) -> np.ndarray:
    expected = (pair_count, SEMANTIC_ROWS, SEMANTIC_WIDTH)
    if options.layout is SentenceLayout.MONOLITHIC:
        tensor = _decode_numeric(payloads[b"MONO"], b"MONO")
        if tensor.shape != expected:
            raise SDWL1Error(f"MONO shape {tensor.shape} != {expected}")
        return tensor
    tensor = np.zeros(expected, dtype=SEMANTIC_DTYPE)
    if options.layout is SentenceLayout.TYPED_SECTION:
        arrays = {
            b"CELL": ((pair_count, 5, SEMANTIC_WIDTH), slice(0, 5)),
            b"SEPR": ((pair_count, 5, SEMANTIC_WIDTH), slice(5, 10)),
            b"SCRW": ((pair_count, 1, SEMANTIC_WIDTH), slice(10, 11)),
        }
        for tag, (shape, target) in arrays.items():
            value = _decode_numeric(payloads[tag], tag)
            if value.shape != shape:
                raise SDWL1Error(f"{tag!r} shape {value.shape} != {shape}")
            tensor[:, target, :] = value
        return tensor
    for stratum in range(len(CLASS_NAMES)):
        tag = f"S{stratum:03d}".encode("ascii")
        value = _decode_numeric(payloads[tag], tag)
        shape = (pair_count, 2, SEMANTIC_WIDTH)
        if value.shape != shape:
            raise SDWL1Error(f"{tag!r} shape {value.shape} != {shape}")
        tensor[:, stratum, :] = value[:, 0, :]
        tensor[:, 5 + stratum, :] = value[:, 1, :]
    screw = _decode_numeric(payloads[b"SCRW"], b"SCRW")
    if screw.shape != (pair_count, 1, SEMANTIC_WIDTH):
        raise SDWL1Error(f"SCRW shape {screw.shape} is invalid")
    tensor[:, 10:11, :] = screw
    return tensor


def decode_sentence(packet: bytes) -> FactInventory:
    """Strictly decode and validate one complete whole-clip sentence."""

    sections = _parse_sections(packet)
    if len(sections) < 3 or [tag for tag, _payload in sections[:2]] != [b"LEXJ", b"SCHJ"]:
        raise SDWL1Error("SDWL1 packet must begin with LEXJ then SCHJ")
    lexicon = _parse_canonical_json(sections[0][1], "SDWL1 lexicon")
    schema = _parse_canonical_json(sections[1][1], "SDWL1 subject schema")
    if schema.get("schema") != "sdwl1.subject_schema.v1":
        raise SDWL1Error("unknown SDWL1 subject schema")
    options = _options_from_schema(schema)
    if hashlib.sha256(sections[0][1]).hexdigest() != schema.get("lexicon_sha256"):
        raise SDWL1Error("SDWL1 lexicon hash drift")
    try:
        pair_count = int(schema["pair_count"])
        source_height, source_width = schema["source_geometry"]
    except (KeyError, TypeError, ValueError) as exc:
        raise SDWL1Error("invalid SDWL1 geometry or pair count") from exc
    if pair_count <= 0 or type(source_height) is not int or type(source_width) is not int:
        raise SDWL1Error("SDWL1 geometry and pair count must be positive integers")
    expected_tags = _expected_section_tags(options)
    actual_tags = [tag for tag, _payload in sections]
    if actual_tags != expected_tags:
        raise SDWL1Error(f"SDWL1 section order/tags {actual_tags} != {expected_tags}")
    payloads = dict(sections[2:])
    if schema.get("provenance_digest_sha256") != canonical_provenance_digest().hex():
        raise SDWL1Error("SDWL1 canonical provenance digest drift")
    if options.repeated_provenance:
        expected_provenance = canonical_provenance_digest() * pair_count
        if payloads[b"PROV"] != expected_provenance:
            raise SDWL1Error("noncanonical repeated per-pair provenance")
    encoded = _reassemble_numeric(payloads, options, pair_count)
    if options.explicit_frame_indices:
        indices = _decode_numeric(payloads[b"FIDX"], b"FIDX")
        expected_indices = np.arange(pair_count, dtype=np.int64).reshape(-1, 1, 1)
        if not np.array_equal(indices, expected_indices):
            raise SDWL1Error("noncanonical explicit frame indices")
    if options.redundant_event_masks:
        event_mask = _decode_numeric(payloads[b"EVNT"], b"EVNT")
        if event_mask.shape != (pair_count, PAIR_RECORD_COUNT, 1):
            raise SDWL1Error("noncanonical redundant event-mask shape")
    tensor = _temporal_decode(encoded, options.temporal_mode)
    if options.redundant_event_masks and not np.array_equal(event_mask, _causal_event_mask(tensor)):
        raise SDWL1Error("redundant event mask does not match the derived causal stream")
    expected_lexicon = _parse_canonical_json(
        canonical_json_bytes(
            build_lexicon(
                FactInventory(
                    tensor=tensor,
                    source_height=source_height,
                    source_width=source_width,
                    semantic_sha256=_semantic_sha256(tensor),
                ),
                options,
            )
        ),
        "canonical SDWL1 lexicon",
    )
    if lexicon != expected_lexicon:
        raise SDWL1Error("SDWL1 lexicon drift or unmeasured production counts")
    semantic_shape = schema.get("semantic_shape")
    if semantic_shape != [pair_count, SEMANTIC_ROWS, SEMANTIC_WIDTH]:
        raise SDWL1Error("SDWL1 semantic shape drift")
    digest = _semantic_sha256(tensor)
    if digest != schema.get("semantic_sha256"):
        raise SDWL1Error("SDWL1 semantic tensor hash drift")
    if schema.get("described_record_count") != pair_count * PAIR_RECORD_COUNT:
        raise SDWL1Error("SDWL1 described record count drift")
    if schema.get("described_scalar_fact_count") != pair_count * PAIR_SCALAR_FACT_COUNT:
        raise SDWL1Error("SDWL1 described scalar-fact count drift")
    if schema.get("described_fact_count") != pair_count * PAIR_SCALAR_FACT_COUNT:
        raise SDWL1Error("SDWL1 described fact count compatibility field drift")
    if schema.get("pair_record_count") != PAIR_RECORD_COUNT:
        raise SDWL1Error("SDWL1 per-pair record count drift")
    if schema.get("pair_scalar_fact_count") != PAIR_SCALAR_FACT_COUNT:
        raise SDWL1Error("SDWL1 per-pair scalar-fact count drift")
    if schema.get("production_counts") != measure_production_counts(tensor).as_dict():
        raise SDWL1Error("SDWL1 production count drift")
    if schema.get("described_fraction") != 1:
        raise SDWL1Error("SDWL1 described fraction must be exactly one")
    inventory = FactInventory(
        tensor=tensor,
        source_height=source_height,
        source_width=source_width,
        semantic_sha256=digest,
    )
    expected_schema = _schema(
        inventory,
        options,
        hashlib.sha256(sections[0][1]).hexdigest(),
    )
    if schema != expected_schema:
        raise SDWL1Error("SDWL1 subject schema is not the canonical schema for the decoded facts")
    return inventory


def serialize_independent_descriptions(
    inventory: FactInventory,
    layout: SentenceLayout,
) -> bytes:
    """Serialize independent absolute descriptions, resetting at every pair."""

    descriptions: list[bytes] = []
    for pair_index in range(inventory.pair_count):
        tensor = inventory.tensor[pair_index : pair_index + 1]
        pair_inventory = FactInventory(
            tensor=tensor,
            source_height=inventory.source_height,
            source_width=inventory.source_width,
            semantic_sha256=_semantic_sha256(tensor),
        )
        descriptions.append(
            serialize_sentence(
                pair_inventory,
                SentenceOptions(layout=layout, temporal_mode=TemporalMode.ABSOLUTE),
            )
        )
    body = b"".join(
        _DESCRIPTION_HEADER.pack(len(description), hashlib.sha256(description).digest()) + description
        for description in descriptions
    )
    return (
        _COLLECTION_HEADER.pack(
            _COLLECTION_MAGIC,
            _COLLECTION_VERSION,
            len(descriptions),
            len(body),
            hashlib.sha256(body).digest(),
            bytes.fromhex(inventory.semantic_sha256),
        )
        + body
    )


def decode_independent_descriptions(collection: bytes) -> FactInventory:
    """Strictly decode all independently reset descriptions."""

    if not isinstance(collection, bytes) or len(collection) < _COLLECTION_HEADER.size:
        raise SDWL1Error("truncated SDWL1 independent collection")
    magic, version, count, body_size, body_sha, semantic_sha = _COLLECTION_HEADER.unpack_from(collection)
    if magic != _COLLECTION_MAGIC or version != _COLLECTION_VERSION:
        raise SDWL1Error("unknown SDWL1 independent collection magic or version")
    expected_size = _COLLECTION_HEADER.size + int(body_size)
    if len(collection) != expected_size:
        relation = "truncated" if len(collection) < expected_size else "trailing"
        raise SDWL1Error(f"{relation} SDWL1 independent collection bytes")
    body = collection[_COLLECTION_HEADER.size :]
    if hashlib.sha256(body).digest() != body_sha:
        raise SDWL1Error("SDWL1 independent collection hash drift")
    offset = 0
    inventories: list[FactInventory] = []
    for _index in range(int(count)):
        if len(body) - offset < _DESCRIPTION_HEADER.size:
            raise SDWL1Error("truncated independent description header")
        size, digest = _DESCRIPTION_HEADER.unpack_from(body, offset)
        offset += _DESCRIPTION_HEADER.size
        stop = offset + int(size)
        if stop > len(body):
            raise SDWL1Error("truncated independent description")
        description = body[offset:stop]
        offset = stop
        if hashlib.sha256(description).digest() != digest:
            raise SDWL1Error("independent description hash drift")
        decoded = decode_sentence(description)
        if decoded.pair_count != 1:
            raise SDWL1Error("independent description must contain exactly one pair")
        inventories.append(decoded)
    if offset != len(body):
        raise SDWL1Error("trailing bytes after independent descriptions")
    if not inventories:
        raise SDWL1Error("independent collection must contain at least one description")
    geometry = {(item.source_height, item.source_width) for item in inventories}
    if len(geometry) != 1:
        raise SDWL1Error("independent descriptions disagree on source geometry")
    tensor = np.concatenate([item.tensor for item in inventories], axis=0)
    digest = _semantic_sha256(tensor)
    if bytes.fromhex(digest) != semantic_sha:
        raise SDWL1Error("independent collection semantic tensor hash drift")
    source_height, source_width = geometry.pop()
    return FactInventory(
        tensor=tensor,
        source_height=source_height,
        source_width=source_width,
        semantic_sha256=digest,
    )


@dataclass(frozen=True)
class SerializationMeasurement:
    """Complete inner-frame and outer-deflate measurement for one row."""

    inner_framed_bytes: int
    inner_sha256: str
    outer_deflate_bytes: int
    outer_deflate_sha256: str
    exact_parseback: bool
    described_record_count: int
    described_scalar_fact_count: int
    described_fact_count: int
    described_fraction: int
    bytes_per_described_fact: float
    production_counts: dict[str, Any]
    outer_payload: bytes


def decompress_outer_payload(payload: bytes) -> bytes:
    """Strictly decode one complete zlib payload, rejecting truncation/trailers."""

    decoder = zlib.decompressobj()
    try:
        inner = decoder.decompress(payload) + decoder.flush()
    except zlib.error as exc:
        raise SDWL1Error("invalid outer-zlib payload") from exc
    if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise SDWL1Error("truncated or trailing outer-zlib payload")
    return inner


def measure_serialization(
    inventory: FactInventory,
    *,
    options: SentenceOptions | None = None,
    independent_layout: SentenceLayout | None = None,
) -> SerializationMeasurement:
    """Measure one complete serialization, including complete zlib level-9 bytes."""

    if (options is None) == (independent_layout is None):
        raise SDWL1Error("select exactly one whole-sentence or independent serialization")
    if independent_layout is not None:
        inner = serialize_independent_descriptions(inventory, independent_layout)
        decoded = decode_independent_descriptions(inner)
    else:
        assert options is not None
        inner = serialize_sentence(inventory, options)
        decoded = decode_sentence(inner)
    exact = decoded.semantic_sha256 == inventory.semantic_sha256 and np.array_equal(decoded.tensor, inventory.tensor)
    outer = zlib.compress(inner, level=9)
    if decompress_outer_payload(outer) != inner:
        raise SDWL1Error("complete outer-zlib payload failed strict parse-back")
    scalar_facts = inventory.described_scalar_fact_count
    return SerializationMeasurement(
        inner_framed_bytes=len(inner),
        inner_sha256=hashlib.sha256(inner).hexdigest(),
        outer_deflate_bytes=len(outer),
        outer_deflate_sha256=hashlib.sha256(outer).hexdigest(),
        exact_parseback=exact,
        described_record_count=inventory.described_record_count,
        described_scalar_fact_count=scalar_facts,
        described_fact_count=scalar_facts,
        described_fraction=1,
        bytes_per_described_fact=len(outer) / scalar_facts,
        production_counts=measure_production_counts(inventory.tensor).as_dict(),
        outer_payload=outer,
    )


__all__ = [
    "CLASS_NAMES",
    "DERIVATION_REGISTRY",
    "MARGIN_BANDS",
    "PAIR_RECORD_COUNT",
    "PAIR_SCALAR_FACT_COUNT",
    "SEMANTIC_ROWS",
    "SEMANTIC_WIDTH",
    "FactInventory",
    "ModifierKind",
    "ModifierSpec",
    "PairScrewFact",
    "PartitionCellFact",
    "PredicateKind",
    "PredicateSpec",
    "ProductionCounts",
    "SDWL1Error",
    "SentenceLayout",
    "SentenceOptions",
    "SeparatrixFact",
    "SerializationMeasurement",
    "SubjectKind",
    "SubjectSpec",
    "TemporalMode",
    "build_lexicon",
    "canonical_json_bytes",
    "canonical_provenance_digest",
    "decode_independent_descriptions",
    "decode_sentence",
    "decompress_outer_payload",
    "extract_fact_inventory",
    "measure_production_counts",
    "measure_serialization",
    "serialize_independent_descriptions",
    "serialize_sentence",
    "validate_derivation_coverage",
]
