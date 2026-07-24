# SPDX-License-Identifier: MIT
"""Five-type derivation tags for the SHA-frozen DDM SN1 evidence.

SN1 predates the binding five-type amendment.  This module does not rewrite
its historical measurements.  It validates a content-addressed compatibility
addendum over those measurements, using the already-landed PF2 representation
type vocabulary as the sole type source.

The layer homes are the recursive evaluator stack:

* L1_PROGRAM: counted description and grammar;
* L2_RECEIVER_R: deterministic expansion, uint8, and resize ``R``;
* L3_SCORER_FEATURE: frozen scorer activations and dynamics;
* L4_SCORER_DECISION: logits, argmax cells, and Pose outputs; and
* L5_VERDICT: evaluator terms and authority disposition.
"""

from __future__ import annotations

import ast
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

ADDENDUM_SCHEMA: Final = "ddm_sn1_five_type_derivation_addendum.v1"
TAG_SCHEMA: Final = "ddm_sn1_five_type_stream_tag.v1"
EVIDENCE_AXIS: Final = "[macOS-CPU frozen-SegNet+PoseNet advisory]"
_REPRESENTATION_TYPE_SOURCE: Final = (
    Path(__file__).resolve().parents[1] / "optimization" / "ddm_dimension_conditioned_two_type.py"
)
LAYER_HOMES: Final = (
    "L1_PROGRAM",
    "L2_RECEIVER_R",
    "L3_SCORER_FEATURE",
    "L4_SCORER_DECISION",
    "L5_VERDICT",
)
EVALUATE_RECURSION_LEVELS: Final = (
    "L0_SCORE_SIGNATURE",
    "L1_TERM_NATIVE_GEOMETRY",
    "L2_TEMPORAL_COMPOSITION",
)
METRIC_GEOMETRIES: Final = (
    "SEG_MARGIN_FISHER_RANK4",
    "POSE_EXACT_OUTPUT_QUADRATIC_LE6",
    "RATE_EXACT_BYTES",
    "IDENTITY_NOOP_GAUGE",
)


class DDMSN1FiveTypeError(ValueError):
    """A malformed or stale SN1 five-type declaration."""


def _read_canonical_representation_types() -> tuple[str, ...]:
    """Read the canonical tuple without importing its optional codec stack.

    The owner module imports Brotli and NumPy because it also implements
    codecs.  A typing-only receipt must remain buildable in the stdlib-only
    control environment, so this reads that module's literal assignment while
    retaining the source file as the single source of truth.
    """

    tree = ast.parse(_REPRESENTATION_TYPE_SOURCE.read_text(encoding="utf-8"))
    for node in tree.body:
        if (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "REPRESENTATION_TYPES"
            and node.value is not None
        ):
            value = ast.literal_eval(node.value)
            if isinstance(value, tuple) and len(value) == 5 and all(isinstance(item, str) and item for item in value):
                return value
    raise DDMSN1FiveTypeError("canonical REPRESENTATION_TYPES literal is absent or malformed")


CANONICAL_REPRESENTATION_TYPES: Final = _read_canonical_representation_types()


def sha256_file(path: Path) -> str:
    """Hash a file in bounded chunks."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _lower_sha256(value: str, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise DDMSN1FiveTypeError(f"{field} must be a lowercase SHA-256")
    return value


def _nonempty(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DDMSN1FiveTypeError(f"{field} must be a nonempty string")
    return value


@dataclass(frozen=True, slots=True)
class SN1FiveTypeStreamTag:
    """One type/home/recursion declaration bound to exact historical bytes.

    This is intentionally an SN1 compatibility row, not a second global enum.
    ``representation_type`` is validated against the canonical
    ``REPRESENTATION_TYPES`` tuple.  MAIN may mechanically adapt these rows to
    ``TypedStreamTag`` if the independently owned TS1 schema lands first.
    """

    stream_id: str
    artifact_path: str
    artifact_sha256: str
    artifact_selector: str
    representation_type: str
    layer_home: str
    evaluate_recursion_level: str
    derivation: str
    metric_geometry: str
    first_rung: str
    verdict_scope: str
    score_claim: bool = False

    def __post_init__(self) -> None:
        for field in (
            "stream_id",
            "artifact_path",
            "artifact_selector",
            "derivation",
            "first_rung",
            "verdict_scope",
        ):
            _nonempty(getattr(self, field), field)
        _lower_sha256(self.artifact_sha256, "artifact_sha256")
        if self.representation_type not in CANONICAL_REPRESENTATION_TYPES:
            raise DDMSN1FiveTypeError("representation_type must reuse canonical REPRESENTATION_TYPES")
        if self.layer_home not in LAYER_HOMES:
            raise DDMSN1FiveTypeError(f"unknown layer home: {self.layer_home!r}")
        if self.evaluate_recursion_level not in EVALUATE_RECURSION_LEVELS:
            raise DDMSN1FiveTypeError("evaluate_recursion_level must cite L0, L1, or L2")
        if self.metric_geometry not in METRIC_GEOMETRIES:
            raise DDMSN1FiveTypeError(f"unknown metric geometry: {self.metric_geometry!r}")
        if self.score_claim is not False:
            raise DDMSN1FiveTypeError("SN1 addendum rows must set score_claim=false")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": TAG_SCHEMA,
            "stream_id": self.stream_id,
            "artifact": {
                "path": self.artifact_path,
                "sha256": self.artifact_sha256,
                "selector": self.artifact_selector,
            },
            "representation_type": self.representation_type,
            "layer_home": self.layer_home,
            "evaluate_recursion_level": self.evaluate_recursion_level,
            "derivation": self.derivation,
            "metric_geometry": self.metric_geometry,
            "identity_euclidean_control": False,
            "first_rung": self.first_rung,
            "verdict_scope": self.verdict_scope,
            "evidence_axis": EVIDENCE_AXIS,
            "research_only": True,
            "score_claim": False,
            "promotion_eligible": False,
        }


def validate_artifact_identity(
    *,
    repo_root: Path,
    tag: SN1FiveTypeStreamTag,
) -> None:
    """Fail closed when a consumed historical artifact is stale."""

    artifact = Path(tag.artifact_path)
    path = artifact if artifact.is_absolute() else repo_root / artifact
    if not path.is_file():
        raise DDMSN1FiveTypeError(f"typed artifact is missing: {tag.artifact_path}")
    observed = sha256_file(path)
    if observed != tag.artifact_sha256:
        raise DDMSN1FiveTypeError(
            f"typed artifact SHA drift for {tag.stream_id}: expected {tag.artifact_sha256}, observed {observed}"
        )


def build_five_type_addendum(
    *,
    repo_root: Path,
    source_receipts: list[dict[str, str]],
    tags: list[SN1FiveTypeStreamTag],
    typed_stream_tag_status: str,
) -> dict[str, Any]:
    """Validate and serialize a complete, metric-first SN1 addendum."""

    if not tags:
        raise DDMSN1FiveTypeError("at least one typed stream is required")
    if len({tag.stream_id for tag in tags}) != len(tags):
        raise DDMSN1FiveTypeError("stream_id values must be unique")
    _nonempty(typed_stream_tag_status, "typed_stream_tag_status")

    receipt_rows: list[dict[str, str]] = []
    for receipt in source_receipts:
        path_value = _nonempty(receipt.get("path", ""), "source_receipt.path")
        expected = _lower_sha256(receipt.get("sha256", ""), "source_receipt.sha256")
        path = Path(path_value)
        resolved = path if path.is_absolute() else repo_root / path
        if not resolved.is_file():
            raise DDMSN1FiveTypeError(f"source receipt is missing: {path_value}")
        observed = sha256_file(resolved)
        if observed != expected:
            raise DDMSN1FiveTypeError(f"source receipt SHA drift: expected {expected}, observed {observed}")
        receipt_rows.append({"path": path_value, "sha256": expected})

    for tag in tags:
        validate_artifact_identity(repo_root=repo_root, tag=tag)

    present_types = {tag.representation_type for tag in tags}
    missing_types = sorted(set(CANONICAL_REPRESENTATION_TYPES) - present_types)
    if missing_types:
        raise DDMSN1FiveTypeError(f"addendum must close all five representation types: {missing_types}")
    present_homes = {tag.layer_home for tag in tags}
    missing_homes = sorted(set(LAYER_HOMES) - present_homes)
    if missing_homes:
        raise DDMSN1FiveTypeError(f"addendum must cover the full L1-L5 stack: {missing_homes}")

    return {
        "schema": ADDENDUM_SCHEMA,
        "lane_id": "ddm_sn1_segnet_telemetry_asymmetry",
        "authority": {
            "typed_stream_tag_status": typed_stream_tag_status,
            "representation_type_source": ("tac.optimization.ddm_dimension_conditioned_two_type.REPRESENTATION_TYPES"),
            "representation_type_source_sha256": hashlib.sha256(_REPRESENTATION_TYPE_SOURCE.read_bytes()).hexdigest(),
            "parallel_enum_created": False,
        },
        "evaluate_py_derivation": {
            "L0_SCORE_SIGNATURE": (
                "evaluate.py composes discrete Seg disagreement, the exact "
                "Pose output quadratic, and exact archive rate"
            ),
            "L1_TERM_NATIVE_GEOMETRY": (
                "Seg argmax cells generate SKELETON; rank-4 margin-Fisher "
                "distances generate FIBER; R/scorer-null directions generate "
                "GAUGE; target mismatches generate RESIDUAL"
            ),
            "L2_TEMPORAL_COMPOSITION": (
                "the ordered pair/clip trajectory generates CONNECTION and event/innovation typing"
            ),
        },
        "layer_home_legend": {
            "L1_PROGRAM": "counted description and grammar",
            "L2_RECEIVER_R": "deterministic expansion, uint8, and resize R",
            "L3_SCORER_FEATURE": "frozen scorer activations and dynamics",
            "L4_SCORER_DECISION": "logits, argmax cells, and Pose outputs",
            "L5_VERDICT": "evaluator terms and authority disposition",
        },
        "metric_first": {
            "seg": "margin-Fisher and rank-4 winner/rival head geometry",
            "pose": "exact <=6-dimensional Pose output quadratic",
            "euclidean": "labeled control only; no emitted row uses it",
        },
        "source_receipts": receipt_rows,
        "typed_stream_count": len(tags),
        "representation_types_covered": sorted(present_types),
        "layer_homes_covered": list(LAYER_HOMES),
        "rows": [tag.to_dict() for tag in tags],
        "evidence_axis": EVIDENCE_AXIS,
        "pointer_moved": False,
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "main_landing_review_required": True,
    }


__all__ = [
    "ADDENDUM_SCHEMA",
    "CANONICAL_REPRESENTATION_TYPES",
    "EVALUATE_RECURSION_LEVELS",
    "EVIDENCE_AXIS",
    "LAYER_HOMES",
    "METRIC_GEOMETRIES",
    "TAG_SCHEMA",
    "DDMSN1FiveTypeError",
    "SN1FiveTypeStreamTag",
    "build_five_type_addendum",
    "sha256_file",
    "validate_artifact_identity",
]
