#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Derive the exact 36-row RG3 residual-family vocabulary from RG2 residue."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Final

import numpy as np

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO / "src", REPO):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.optimization import direct_description_coupled_margin as coupled  # noqa: E402
from tac.optimization import direct_description_preuint8_channel as preuint8  # noqa: E402
from tac.optimization.ddm_min_description_contract import (  # noqa: E402
    LayerHome,
    StreamType,
    TypedStreamTag,
)
from tac.optimization.ddm_pf2_bucket_assignment import (  # noqa: E402
    canonical_bytes,
    canonical_sha256,
)
from tac.optimization.ddm_rg1_receiver_grammar import (  # noqa: E402
    RG3ResidualCoordinateV1,
    derive_rg3_class_birth_address,
    derive_rg3_finer_event_local_band,
    derive_rg3_fisher_margin_band,
    receive_rg1_receiver_grammar,
)

SCHEMA: Final = "ddm_rg3_residual_family_assignment.v1"
EXPECTED_SUMMARY_FILE_SHA256: Final = "15b12224e3abb0d93f4fb9693402794d27969783b1d796114f0208277fe5a9ed"
EXPECTED_SUMMARY_CONTENT_SHA256: Final = "9fe683f58a60b88cf23b201e7d7ee2756f6165c57eae5e63b12b840dcc32ac46"
EXPECTED_OUTER_BASE_SHA256: Final = "dc767b59c9e8671b6870e0f9f17a24cfe900dd0f2ae2a251825e41566b52e4c9"
EXPECTED_NESTED_CARRIER_SHA256: Final = "7990fce786aac1f24bcb977882348867ca2d9cbc4d95d0337dd1167e593f46c6"
EXPECTED_MARGIN_SHA256: Final = "177d22f0ef16e31f9de0229606f72e69d22dd550b7ff55342f82d01ebe6f228d"
EXPECTED_MARGIN_BYTES: Final = 600 * 384 * 512 * 2
EXPECTED_FAMILY_COUNTS: Final = {
    "EVENT_LOCAL_SKELETON_CLASS_BIRTH_PRODUCTION": 10,
    "FINER_EVENT_LOCAL_SKELETON_AMPLITUDE_CODEBOOK": 9,
    "FISHER_MARGIN_PER_STRATUM_SKELETON_AMPLITUDE_CODEBOOK": 17,
}
DEFAULT_SUMMARY = REPO / (
    ".omx/research/ddm_rg2_skeleton_amplitude_productions_20260724T094305Z/"
    "ddm_rg2_receiver_support_summary.json"
)
DEFAULT_BASE = REPO / (
    ".omx/research/ddm_v19c_correction_saturation_20260723T063500Z/"
    "ddm_v19c_final_n600.zip.receipt-bytes"
)
DEFAULT_MARGIN = Path(
    "/Volumes/VertigoDataTier/pact/lever_b_score_native_argmax_smoke_20260610/"
    "targets_n600/gt_segnet_margin.f16"
)
DEFAULT_OUTPUT = REPO / (
    ".omx/research/ddm_rg3_residual_family_productions_20260724T110418Z/"
    "ddm_rg3_residual_family_assignment.json"
)


class RG3AssignmentError(RuntimeError):
    """Input custody or one of the 36 typed derivations differs."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_bound(path: Path, expected: str, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RG3AssignmentError(f"{label} is absent or not a regular file: {path}")
    observed = _sha256(path)
    if observed != expected:
        raise RG3AssignmentError(
            f"{label} SHA-256 differs: expected {expected}, observed {observed}"
        )
    try:
        value = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RG3AssignmentError(f"{label} is not canonical JSON") from exc
    if not isinstance(value, dict):
        raise RG3AssignmentError(f"{label} root must be an object")
    return value


def _nested_carrier(path: Path) -> bytes:
    if _sha256(path) != EXPECTED_OUTER_BASE_SHA256:
        raise RG3AssignmentError("V19C outer base SHA-256 differs")
    outer = path.read_bytes()
    pre_members, _ = preuint8.parse_preuint8_q8_archive(outer)
    coupled_members, _ = coupled.parse_coupled_margin_archive(
        pre_members[preuint8.BASE_MEMBER]
    )
    carrier = coupled_members[coupled.BASE_MEMBER]
    if hashlib.sha256(carrier).hexdigest() != EXPECTED_NESTED_CARRIER_SHA256:
        raise RG3AssignmentError("nested carrier SHA-256 differs")
    return carrier


def _margin_maps(path: Path) -> np.memmap:
    if (
        not path.is_file()
        or path.is_symlink()
        or path.stat().st_size != EXPECTED_MARGIN_BYTES
        or _sha256(path) != EXPECTED_MARGIN_SHA256
    ):
        raise RG3AssignmentError("n600 Fisher-margin source custody differs")
    return np.memmap(
        path,
        mode="r",
        dtype="<f2",
        shape=(600, 384, 512),
    )


def build_assignment(
    *,
    summary_path: Path,
    base_path: Path,
    margin_path: Path,
) -> dict[str, Any]:
    """Build one surgical RG3 address and its admissible magnitudes per residue."""

    summary = _load_bound(
        summary_path,
        EXPECTED_SUMMARY_FILE_SHA256,
        "RG2 support summary",
    )
    if summary.get("summary_content_sha256") != EXPECTED_SUMMARY_CONTENT_SHA256:
        raise RG3AssignmentError("RG2 summary content SHA-256 differs")
    derivation = summary.get("receiver_coordinate_derivation")
    residue = derivation.get("residual") if isinstance(derivation, dict) else None
    family_counts = (
        derivation.get("next_coordinate_family_counts")
        if isinstance(derivation, dict)
        else None
    )
    if (
        not isinstance(residue, list)
        or len(residue) != 36
        or family_counts != EXPECTED_FAMILY_COUNTS
        or summary.get("g3_top24_coverage", {}).get("missing_block_count") != 36
    ):
        raise RG3AssignmentError("RG2 summary does not carry the exact 36-row residue")

    receiver = receive_rg1_receiver_grammar(
        _nested_carrier(base_path),
        verify_member_effects=False,
    )
    margins = _margin_maps(margin_path)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    all_actuator_ids: list[str] = []
    for source in residue:
        if not isinstance(source, dict) or not isinstance(source.get("typed_key"), dict):
            raise RG3AssignmentError("RG2 residual row is malformed")
        pair_id = source.get("pair_id")
        bucket_id = source.get("bucket_id")
        typed = source["typed_key"]
        class_ids = typed.get("class_ids")
        families = source.get("candidate_coordinate_families")
        if (
            isinstance(pair_id, bool)
            or not isinstance(pair_id, int)
            or not 0 <= pair_id < 600
            or not isinstance(bucket_id, str)
            or not isinstance(class_ids, list)
            or len(class_ids) != 2
            or class_ids != sorted(set(class_ids))
            or not isinstance(families, list)
            or len(families) != 1
            or families[0] not in EXPECTED_FAMILY_COUNTS
        ):
            raise RG3AssignmentError("RG2 residual typed address differs")
        identity = (pair_id, bucket_id)
        if identity in seen:
            raise RG3AssignmentError("RG2 residual pair/bucket address is duplicated")
        seen.add(identity)
        family = str(families[0])
        prior_band = source.get("rg2_receiver_derived_row_band")
        if family == "EVENT_LOCAL_SKELETON_CLASS_BIRTH_PRODUCTION":
            if source.get("rg2_receiver_actuator_id") is not None or prior_band is not None:
                raise RG3AssignmentError("class-birth row unexpectedly has an RG2 address")
            row_band, fine_band = derive_rg3_class_birth_address(
                receiver,
                pair_index=pair_id,
            )
            magnitudes = (1,)
            selection_metric = "receiver_class_agnostic_boundary_mass"
        elif family == "FINER_EVENT_LOCAL_SKELETON_AMPLITUDE_CODEBOOK":
            if isinstance(prior_band, bool) or not isinstance(prior_band, int):
                raise RG3AssignmentError("finer-event row lacks its RG2 band")
            row_band = prior_band
            fine_band = derive_rg3_finer_event_local_band(
                receiver,
                pair_index=pair_id,
                class_a=class_ids[0],
                class_b=class_ids[1],
                row_band=row_band,
            )
            magnitudes = (1, 2)
            selection_metric = "typed_receiver_interface_mass"
        else:
            if isinstance(prior_band, bool) or not isinstance(prior_band, int):
                raise RG3AssignmentError("Fisher-stratum row lacks its RG2 band")
            row_band = prior_band
            fine_band = derive_rg3_fisher_margin_band(
                receiver,
                pair_index=pair_id,
                class_a=class_ids[0],
                class_b=class_ids[1],
                row_band=row_band,
                margin_map=margins[pair_id],
            )
            magnitudes = (1, 2)
            selection_metric = "categorical_fisher_trace_half_sech2_top1_top2_margin"

        actuator_ids = []
        for magnitude in magnitudes:
            coordinate = RG3ResidualCoordinateV1(
                pair_index=pair_id,
                class_a=class_ids[0],
                class_b=class_ids[1],
                family=family,
                temporal_class=typed["g4_temporal_class"],
                row_band=row_band,
                fine_band=fine_band,
                signed_quanta=magnitude,
            )
            actuator_ids.append(coordinate.actuator_id)
        all_actuator_ids.extend(actuator_ids)
        rows.append(
            {
                "schema": "ddm_rg3_residual_family_assignment_row.v1",
                "pair_id": pair_id,
                "bucket_id": bucket_id,
                "typed_key": dict(typed),
                "selected_coordinate_family": family,
                "receiver_derived_row_band": row_band,
                "receiver_derived_fine_band": fine_band,
                "selection_metric": selection_metric,
                "receiver_actuator_ids": actuator_ids,
                "signed_directions": [
                    "NEGATIVE_ONE_QUANTUM",
                    "POSITIVE_ONE_QUANTUM",
                ],
                "admissible_magnitudes": list(magnitudes),
                "source_rg2_receiver_actuator_id": source.get(
                    "rg2_receiver_actuator_id"
                ),
                "source_rg2_reason": source.get("reason"),
                "causal_join_status": "UNMEASURED_PENDING_RG3_SIGNED_PROBE",
                "score_units_per_byte_status": "OWED_NOT_ADMITTED",
            }
        )
    rows.sort(key=lambda value: (value["pair_id"], value["bucket_id"]))
    if len(set(all_actuator_ids)) != len(all_actuator_ids):
        raise RG3AssignmentError("RG3 admissible actuator IDs are not unique")
    if len(all_actuator_ids) != 62:
        raise RG3AssignmentError(
            f"RG3 actuator count differs: expected 62, observed {len(all_actuator_ids)}"
        )

    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "lane_id": "lane_ddm_rg3_residual_family_productions_20260724",
        "input_custody": {
            "rg2_summary_path": str(summary_path.resolve()),
            "rg2_summary_file_sha256": EXPECTED_SUMMARY_FILE_SHA256,
            "rg2_summary_content_sha256": EXPECTED_SUMMARY_CONTENT_SHA256,
            "outer_v19c_base_sha256": EXPECTED_OUTER_BASE_SHA256,
            "nested_carrier_sha256": EXPECTED_NESTED_CARRIER_SHA256,
            "fisher_margin_path": str(margin_path.resolve()),
            "fisher_margin_sha256": EXPECTED_MARGIN_SHA256,
            "fisher_margin_dtype": "float16",
            "fisher_margin_shape": [600, 384, 512],
        },
        "derivation": {
            "class_birth": (
                "one two-cell typed interface seed at the maximum-mass "
                "class-agnostic receiver-boundary subband"
            ),
            "finer_event_local": (
                "refine only the assigned RG2 64-row interface band into one "
                "maximum-interface-mass 16-row subband"
            ),
            "fisher_margin_per_stratum": (
                "refine only the assigned RG2 cell band by maximum summed "
                "categorical Fisher trace 0.5*sech^2(m/2) over typed receiver support"
            ),
            "no_label_or_scorer_payload": True,
        },
        "row_count": len(rows),
        "actuator_count": len(all_actuator_ids),
        "new_signed_probe_count": 2 * len(all_actuator_ids),
        "family_counts": {
            family: sum(
                row["selected_coordinate_family"] == family for row in rows
            )
            for family in sorted(EXPECTED_FAMILY_COUNTS)
        },
        "rows": rows,
        "typed_stream_tag": TypedStreamTag(
            type=StreamType.SKELETON,
            layer_home=LayerHome.L3_RASTER,
            evaluate_py_recursion_level_cited=(
                "L3_raster counted RG3 class-birth/refined codebook symbol "
                "-> evaluator-owned R -> L4_scorer_feature"
            ),
            counted_bytes=0,
            free_receiver_code=True,
        ).to_dict(),
        "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
        "score_claim": False,
        "pointer": "0.1910828242 [contest-CPU]",
        "pointer_moved": False,
        "research_only": True,
        "verdict_scope": "INSTANCE_EXTENDED_GRAMMAR_RG3_ASSIGNMENT_ONLY",
        "main_landing_review_required": True,
    }
    payload["assignment_content_sha256"] = canonical_sha256(payload)
    return payload


def _publish(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(canonical_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--base", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--margin", type=Path, default=DEFAULT_MARGIN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    assignment = build_assignment(
        summary_path=args.summary,
        base_path=args.base,
        margin_path=args.margin,
    )
    _publish(args.output, assignment)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
