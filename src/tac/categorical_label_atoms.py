# SPDX-License-Identifier: MIT
"""Deterministic typed label atoms for categorical mask candidates."""

from __future__ import annotations

from typing import Any, Final

from tac.categorical_openpilot_mask_prior_contract import (
    COMPRESSION_TIME_ONLY_USAGE,
    RUNTIME_LABEL_CONTRACT,
)
from tac.semantic_label_contract import (
    CONTEST_SEGNET_CLASSES,
    SELFCOMP_CLASS_TO_GRAY,
    SEMANTIC_QUANTIZATION_DEFAULT_BITS,
)

SCHEMA_VERSION: Final[int] = 1
CATEGORICAL_TYPED_LABEL_ATOMS_KIND: Final[str] = "categorical_typed_label_atoms"
CATEGORICAL_TYPED_LABEL_ATOMS_CONTRACT: Final[str] = "categorical_typed_label_atoms_v1"

OPENPILOT_PRIOR_HINTS: Final[dict[str, str]] = {
    "road": "road_geometry_track_prior",
    "lane_markings": "lane_marking_track_prior",
    "undrivable": "scene_layout_prior",
    "movable": "dynamic_object_filter_prior",
    "my_car": "ego_car_filter_prior",
}


def semantic_priority_weight_ppm(class_id: int) -> int:
    """Return deterministic per-class priority weight in parts per million."""

    total_bits = sum(SEMANTIC_QUANTIZATION_DEFAULT_BITS.values())
    return round(SEMANTIC_QUANTIZATION_DEFAULT_BITS[class_id] * 1_000_000 / total_bits)


def canonical_categorical_label_atom_rows() -> list[dict[str, Any]]:
    """Return class-typed atom rows in contest zero-based comma10k order."""

    rows: list[dict[str, Any]] = []
    for item in CONTEST_SEGNET_CLASSES:
        rows.append(
            {
                "atom_id": f"contest_class_{item.class_id}_{item.name}",
                "atom_type": "semantic_class_label_prior",
                "stack_stage": "representation",
                "label_contract": RUNTIME_LABEL_CONTRACT,
                "usage": COMPRESSION_TIME_ONLY_USAGE,
                "runtime_consumed": False,
                "class_id": item.class_id,
                "comma10k_id": item.comma10k_id,
                "name": item.name,
                "comma10k_color": item.comma10k_color,
                "selfcomp_gray": SELFCOMP_CLASS_TO_GRAY[item.class_id],
                "default_quant_bits": SEMANTIC_QUANTIZATION_DEFAULT_BITS[item.class_id],
                "semantic_priority_weight_ppm": semantic_priority_weight_ppm(item.class_id),
                "openpilot_prior_hint": OPENPILOT_PRIOR_HINTS[item.name],
                "runtime_policy": "runtime use requires charged archive member provenance",
            }
        )
    return rows


def build_categorical_typed_label_atoms() -> dict[str, Any]:
    """Build the deterministic atom manifest embedded in categorical surfaces."""

    atoms = canonical_categorical_label_atom_rows()
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": CATEGORICAL_TYPED_LABEL_ATOMS_KIND,
        "typed_label_atoms_contract": CATEGORICAL_TYPED_LABEL_ATOMS_CONTRACT,
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "label_contract": RUNTIME_LABEL_CONTRACT,
        "atom_count": len(atoms),
        "atoms": atoms,
    }


__all__ = [
    "CATEGORICAL_TYPED_LABEL_ATOMS_CONTRACT",
    "CATEGORICAL_TYPED_LABEL_ATOMS_KIND",
    "OPENPILOT_PRIOR_HINTS",
    "SCHEMA_VERSION",
    "build_categorical_typed_label_atoms",
    "canonical_categorical_label_atom_rows",
    "semantic_priority_weight_ppm",
]
