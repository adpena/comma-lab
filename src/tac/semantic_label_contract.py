"""Semantic-label contracts for comma/openpilot-derived mask lanes.

The contest SegNet path uses five class channels. comma10k documents those
labels as one-based IDs; scorer/runtime tensors use the same order zero-based:

    0 road, 1 lane_markings, 2 undrivable, 3 movable, 4 my_car

The public Selfcomp submission adds a separate grayscale wire codebook over
those class IDs. The codebook is not a semantic relabeling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class SemanticClass:
    """A single semantic class in the comma10k/contest SegNet contract."""

    class_id: int
    comma10k_id: int
    name: str
    comma10k_color: str
    description: str


CONTEST_SEGNET_CLASSES: Final[tuple[SemanticClass, ...]] = (
    SemanticClass(0, 1, "road", "#402020", "drivable road surface"),
    SemanticClass(1, 2, "lane_markings", "#ff0000", "lane markings"),
    SemanticClass(2, 3, "undrivable", "#808060", "undrivable scene regions, including sky"),
    SemanticClass(3, 4, "movable", "#00ff66", "vehicles, people, and animals"),
    SemanticClass(4, 5, "my_car", "#cc00ff", "ego car interior, mounts, and wires"),
)

COMMA10K_INTERIOR_ONLY_CLASSES: Final[tuple[SemanticClass, ...]] = (
    SemanticClass(
        5,
        6,
        "movable_in_my_car",
        "#00ccff",
        "people inside the car; comma10k imgsd-only, not a contest SegNet channel",
    ),
)

NUM_CONTEST_SEGNET_CLASSES: Final[int] = len(CONTEST_SEGNET_CLASSES)

CONTEST_SEGNET_CLASS_NAMES: Final[dict[int, str]] = {
    item.class_id: item.name for item in CONTEST_SEGNET_CLASSES
}

CONTEST_SEGNET_CLASS_NAME_TUPLE: Final[tuple[str, ...]] = tuple(
    item.name for item in CONTEST_SEGNET_CLASSES
)

CONTEST_SEGNET_COMMA10K_COLORS: Final[dict[int, str]] = {
    item.class_id: item.comma10k_color for item in CONTEST_SEGNET_CLASSES
}

# Public PR56/Selfcomp grayscale targets. Keys are contest class IDs, not new
# semantic labels. The source code exposes only this index-ordered list:
# CLASS_TARGETS = [0, 255, 64, 192, 128].
SELFCOMP_CLASS_TO_GRAY: Final[dict[int, int]] = {
    0: 0,
    1: 255,
    2: 64,
    3: 192,
    4: 128,
}

# Default for CLADE/SPADE per-class parameter quantization. Road and lane
# channels carry high PoseNet/SegNet sensitivity; movable is still important;
# undrivable/ego-car tolerate stronger compression unless exact evidence says
# otherwise.
SEMANTIC_QUANTIZATION_DEFAULT_BITS: Final[dict[int, int]] = {
    0: 8,
    1: 8,
    2: 4,
    3: 6,
    4: 4,
}


def validate_contest_class_table() -> None:
    """Fail closed if class IDs, names, or Selfcomp targets drift."""

    expected_ids = list(range(NUM_CONTEST_SEGNET_CLASSES))
    class_ids = [item.class_id for item in CONTEST_SEGNET_CLASSES]
    if class_ids != expected_ids:
        raise ValueError(f"contest class IDs must be dense {expected_ids}, got {class_ids}")
    if sorted(SELFCOMP_CLASS_TO_GRAY) != expected_ids:
        raise ValueError(
            "Selfcomp grayscale keys must match contest class IDs: "
            f"{sorted(SELFCOMP_CLASS_TO_GRAY)} vs {expected_ids}"
        )
    gray_values = list(SELFCOMP_CLASS_TO_GRAY.values())
    if len(set(gray_values)) != len(gray_values):
        raise ValueError(f"Selfcomp gray targets must be unique, got {gray_values}")


validate_contest_class_table()


__all__ = [
    "COMMA10K_INTERIOR_ONLY_CLASSES",
    "CONTEST_SEGNET_CLASSES",
    "CONTEST_SEGNET_CLASS_NAMES",
    "CONTEST_SEGNET_CLASS_NAME_TUPLE",
    "CONTEST_SEGNET_COMMA10K_COLORS",
    "NUM_CONTEST_SEGNET_CLASSES",
    "SELFCOMP_CLASS_TO_GRAY",
    "SEMANTIC_QUANTIZATION_DEFAULT_BITS",
    "SemanticClass",
    "validate_contest_class_table",
]
