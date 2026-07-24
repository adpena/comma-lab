#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Derive the exact 64-row RG2 SKELETON amplitude vocabulary from RG1 residue."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Final

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
    SkeletonAmplitudeCoordinateV1,
    derive_skeleton_amplitude_row_band,
    receive_rg1_receiver_grammar,
)
from tac.optimization.direct_description_minimizer import (  # noqa: E402
    DirectDescriptionError,
)

SCHEMA: Final = "ddm_rg2_skeleton_amplitude_assignment.v1"
EXPECTED_SUMMARY_FILE_SHA256: Final = "1960687be407b89eb3ccf97f5a5abad1ef62eb939600d164285c94059fc893da"
EXPECTED_SUMMARY_CONTENT_SHA256: Final = "0293c976bf6209bcb314d641a19ac38062b72cca2a4be87f99d2d8aaf337505c"
EXPECTED_OUTER_BASE_SHA256: Final = "dc767b59c9e8671b6870e0f9f17a24cfe900dd0f2ae2a251825e41566b52e4c9"
EXPECTED_NESTED_CARRIER_SHA256: Final = "7990fce786aac1f24bcb977882348867ca2d9cbc4d95d0337dd1167e593f46c6"
DEFAULT_SUMMARY = REPO / (
    ".omx/research/ddm_rg1_receiver_grammar_extension_20260724T080402Z/ddm_rg1_receiver_support_summary.json"
)
DEFAULT_BASE = REPO / (
    ".omx/research/ddm_v19c_correction_saturation_20260723T063500Z/ddm_v19c_final_n600.zip.receipt-bytes"
)
DEFAULT_OUTPUT = REPO / (
    ".omx/research/ddm_rg2_skeleton_amplitude_productions_20260724T094305Z/ddm_rg2_skeleton_amplitude_assignment.json"
)


class RG2AssignmentError(RuntimeError):
    """Input custody or typed-residue derivation differs."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_bound(path: Path, expected: str, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise RG2AssignmentError(f"{label} is absent or not a regular file: {path}")
    observed = _sha256(path)
    if observed != expected:
        raise RG2AssignmentError(f"{label} SHA-256 differs: expected {expected}, observed {observed}")
    try:
        value = json.loads(path.read_bytes())
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RG2AssignmentError(f"{label} is not canonical JSON") from exc
    if not isinstance(value, dict):
        raise RG2AssignmentError(f"{label} root must be an object")
    return value


def _nested_carrier(path: Path) -> bytes:
    if _sha256(path) != EXPECTED_OUTER_BASE_SHA256:
        raise RG2AssignmentError("V19C outer base SHA-256 differs")
    outer = path.read_bytes()
    pre_members, _ = preuint8.parse_preuint8_q8_archive(outer)
    coupled_members, _ = coupled.parse_coupled_margin_archive(pre_members[preuint8.BASE_MEMBER])
    carrier = coupled_members[coupled.BASE_MEMBER]
    observed = hashlib.sha256(carrier).hexdigest()
    if observed != EXPECTED_NESTED_CARRIER_SHA256:
        raise RG2AssignmentError("nested carrier SHA-256 differs")
    return carrier


def _family(stratum: object) -> str:
    if stratum == "boundary":
        return "EVENT_LOCAL_BOUNDARY"
    if stratum == "cell":
        return "PER_STRATUM_ROW_BAND"
    raise RG2AssignmentError(f"residual class stratum is unknown: {stratum!r}")


def build_assignment(*, summary_path: Path, base_path: Path) -> dict[str, Any]:
    """Build one canonical coordinate address for every exact RG1 missing block."""

    summary = _load_bound(
        summary_path,
        EXPECTED_SUMMARY_FILE_SHA256,
        "RG1 support summary",
    )
    if summary.get("summary_content_sha256") != EXPECTED_SUMMARY_CONTENT_SHA256:
        raise RG2AssignmentError("RG1 summary content SHA-256 differs")
    derivation = summary.get("receiver_coordinate_derivation")
    residue = derivation.get("residual") if isinstance(derivation, dict) else None
    if (
        not isinstance(residue, list)
        or len(residue) != 64
        or summary.get("g3_top24_coverage", {}).get("missing_block_count") != 64
    ):
        raise RG2AssignmentError("RG1 summary does not carry the exact 64-row residue")

    carrier = _nested_carrier(base_path)
    receiver = receive_rg1_receiver_grammar(carrier, verify_member_effects=False)
    rows = []
    seen: set[tuple[int, str]] = set()
    for source in residue:
        if not isinstance(source, dict) or not isinstance(source.get("typed_key"), dict):
            raise RG2AssignmentError("RG1 residual row is malformed")
        pair_id = source.get("pair_id")
        bucket_id = source.get("bucket_id")
        typed = source["typed_key"]
        class_ids = typed.get("class_ids")
        if (
            isinstance(pair_id, bool)
            or not isinstance(pair_id, int)
            or not 0 <= pair_id < 600
            or not isinstance(bucket_id, str)
            or not isinstance(class_ids, list)
            or len(class_ids) != 2
            or class_ids != sorted(set(class_ids))
        ):
            raise RG2AssignmentError("RG1 residual typed address differs")
        identity = (pair_id, bucket_id)
        if identity in seen:
            raise RG2AssignmentError("RG1 residual pair/bucket address is duplicated")
        seen.add(identity)
        family = _family(typed.get("class_stratum"))
        try:
            row_band = derive_skeleton_amplitude_row_band(
                receiver,
                pair_index=pair_id,
                class_a=class_ids[0],
                class_b=class_ids[1],
                family=family,
            )
        except DirectDescriptionError as exc:
            if "has no receiver support" not in str(exc):
                raise
            row_band = None
        coordinate = (
            SkeletonAmplitudeCoordinateV1(
                pair_index=pair_id,
                class_a=class_ids[0],
                class_b=class_ids[1],
                family=family,
                temporal_class=typed["g4_temporal_class"],
                row_band=row_band,
                signed_quanta=1,
            )
            if row_band is not None
            else None
        )
        prior_families = list(source.get("candidate_coordinate_families", ()))
        selected_family = (
            "EVENT_LOCAL_SKELETON_BOUNDARY_PRODUCTION"
            if family == "EVENT_LOCAL_BOUNDARY"
            else "PER_STRATUM_SKELETON_AMPLITUDE_FIELD"
        )
        rows.append(
            {
                "schema": "ddm_rg2_skeleton_amplitude_assignment_row.v1",
                "pair_id": pair_id,
                "bucket_id": bucket_id,
                "typed_key": dict(typed),
                "selected_coordinate_family": selected_family,
                "receiver_family": family,
                "receiver_derived_row_band": row_band,
                "receiver_actuator_id": (
                    coordinate.actuator_id if coordinate is not None else None
                ),
                "signed_directions": [
                    "NEGATIVE_ONE_QUANTUM",
                    "POSITIVE_ONE_QUANTUM",
                ],
                "existing_rg1_candidate_families": prior_families[:-1],
                "existing_rg1_disposition": ("PRESERVED_MEASURED_NO_JOIN_AT_THIS_EXACT_PAIR_BUCKET"),
                "selection_reason": (
                    "boundary typed key requires an event-local production"
                    if family == "EVENT_LOCAL_BOUNDARY"
                    else "cell typed key requires a per-stratum row-band amplitude field"
                ),
                "causal_join_status": (
                    "UNMEASURED_PENDING_RG2_SIGNED_PROBE"
                    if coordinate is not None
                    else "UNREACHABLE_NO_SHA_BOUND_RECEIVER_CLASS_PAIR_SUPPORT"
                ),
                "unreachable_reason": (
                    None
                    if coordinate is not None
                    else (
                        "Both typed class roles have zero SHA-bound receiver support "
                        "at this exact pair; RG2 cannot derive a legal event/band address."
                    )
                ),
                "score_units_per_byte_status": "OWED_NOT_ADMITTED",
            }
        )
    rows.sort(key=lambda value: (value["pair_id"], value["bucket_id"]))
    actuator_ids = [
        str(row["receiver_actuator_id"])
        for row in rows
        if row["receiver_actuator_id"] is not None
    ]
    if len(set(actuator_ids)) != len(actuator_ids):
        raise RG2AssignmentError("RG2 admissible coordinates are not unique")
    payload = {
        "schema": SCHEMA,
        "lane_id": "lane_ddm_rg2_skeleton_amplitude_productions_20260724",
        "input_custody": {
            "rg1_summary_path": str(summary_path.resolve()),
            "rg1_summary_file_sha256": EXPECTED_SUMMARY_FILE_SHA256,
            "rg1_summary_content_sha256": EXPECTED_SUMMARY_CONTENT_SHA256,
            "outer_v19c_base_sha256": EXPECTED_OUTER_BASE_SHA256,
            "nested_carrier_sha256": EXPECTED_NESTED_CARRIER_SHA256,
        },
        "derivation": (
            "boundary -> event-local SKELETON production; cell -> per-stratum "
            "SKELETON row-band field; band is argmax of SHA-bound receiver support "
            "mass and consumes no scorer labels or PF2 spatial coordinates"
        ),
        "row_count": len(rows),
        "admissible_coordinate_count": len(actuator_ids),
        "unreachable_row_count": len(rows) - len(actuator_ids),
        "family_counts": {
            family: sum(row["receiver_family"] == family for row in rows) for family in _family_names(rows)
        },
        "rows": rows,
        "typed_stream_tag": TypedStreamTag(
            type=StreamType.SKELETON,
            layer_home=LayerHome.L3_RASTER,
            evaluate_py_recursion_level_cited=(
                "L3_raster counted receiver-derived RG2 SKELETON amplitude "
                "mask -> evaluator-owned R -> L4_scorer_feature"
            ),
            counted_bytes=0,
            free_receiver_code=True,
        ).to_dict(),
        "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
        "score_claim": False,
        "pointer": "0.1910828242 [contest-CPU]",
        "pointer_moved": False,
        "research_only": True,
        "verdict_scope": "INSTANCE_EXTENDED_GRAMMAR_RG2_ASSIGNMENT_ONLY",
        "main_landing_review_required": True,
    }
    payload["assignment_content_sha256"] = canonical_sha256(payload)
    return payload


def _family_names(rows: list[dict[str, Any]]) -> tuple[str, ...]:
    return tuple(sorted({str(row["receiver_family"]) for row in rows}))


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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    assignment = build_assignment(summary_path=args.summary, base_path=args.base)
    _publish(args.output, assignment)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
