#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Summarize the SHA-bound DDM MS6 receiver-support sweep.

This is a read-only consumer of one or more landed v2 checkpoint roots and the
MS5 assignment-table schema.  When an additive grammar supersedes selected
probes, the assignment table's exact checkpoint SHA chooses the authoritative
revision without deleting the prior infeasibility evidence.  It does not infer
joins: G3 coverage is proven only when every PF2 bucket containing a
preregistered hard pair has a measured assignment whose exact joined pair IDs
contain that pair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO / "src", REPO):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.optimization.ddm_pf2_bucket_assignment import (  # noqa: E402
    canonical_bytes,
    canonical_sha256,
    validate_assignment_table,
)

SCHEMA: Final = "ddm_ms6_receiver_support_resume_summary.v1"
CHECKPOINT_SCHEMA: Final = "ddm_ms6_receiver_support_probe_checkpoint.v2"
TABLE_SCHEMA: Final = "ddm_ms5_pf2_bucket_assignment_table.v1"
G3_SCHEMA: Final = "ddm_g3_hard_pair_registry.v1"
RG2_ASSIGNMENT_SCHEMA: Final = "ddm_rg2_skeleton_amplitude_assignment.v1"
RG3_ASSIGNMENT_SCHEMA: Final = "ddm_rg3_residual_family_assignment.v1"
EXPECTED_BASE_SHA256: Final = "dc767b59c9e8671b6870e0f9f17a24cfe900dd0f2ae2a251825e41566b52e4c9"
EXPECTED_PF2_SHA256: Final = "85084f7bd3a03dbd1b9f04fe6a9b84df4948a6caf64620beef42da8924345f73"
EXPECTED_G3_SHA256: Final = "0c9ce6d0ce2b2c0830400f096438355242527d40f682fc1b201f67d8d951a4e4"
DIRECTIONS: Final = ("NEGATIVE_ONE_QUANTUM", "POSITIVE_ONE_QUANTUM")
DEFAULT_CHECKPOINTS = (
    Path("/Volumes/VertigoDataTier/pact")
    / "ddm_ms6_receiver_support_measurement_20260724T052034Z"
    / "probe_checkpoints_v2"
)
DEFAULT_TABLE = REPO / (
    ".omx/research/ddm_ms6_receiver_support_measurement_20260724T052034Z/"
    "pf2_bucket_assignment_table.json"
)
DEFAULT_G3 = REPO / (
    ".omx/research/ddm_g3_score_atlas_n600_20260722T204000Z/"
    "hard_pair_registry.json"
)
DEFAULT_OUTPUT = REPO / (
    ".omx/research/ddm_ms6_receiver_support_measurement_20260724T052034Z/"
    "ddm_ms6_receiver_support_resume_summary.json"
)


class SummaryError(RuntimeError):
    """The summary inputs do not satisfy the landed custody contract."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise SummaryError(f"expected JSON object: {path}")
    return value


def _publish(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    payload = canonical_bytes(value)
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _probe_metrics(row: Mapping[str, Any]) -> dict[str, int]:
    raster = row.get("raster_support")
    scorer = row.get("scorer")
    hits = row.get("bucket_hits")
    if not isinstance(raster, Mapping) or not isinstance(scorer, Mapping) or not isinstance(hits, list):
        raise SummaryError("checkpoint support/scorer/bucket fields are malformed")
    return {
        "raster_pair_count": int(raster.get("pair_count", 0)),
        "camera_value_count": int(raster.get("camera_value_count", 0)),
        "composite_r_cell_count": int(raster.get("composite_r_cell_count", 0)),
        "scorer_pair_count": int(scorer.get("forward_pair_count", 0)),
        "bucket_hit_count": len(hits),
        "perturbed_event_count": sum(int(hit["event_count"]) for hit in hits),
    }


def _signed_asymmetry(negative: int, positive: int) -> float:
    denominator = negative + positive
    return 0.0 if denominator == 0 else (positive - negative) / denominator


def _distribution(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0}
    ordered = sorted(float(value) for value in values)

    def quantile(fraction: float) -> float:
        position = fraction * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        "count": len(ordered),
        "minimum": ordered[0],
        "p25": quantile(0.25),
        "median": statistics.median(ordered),
        "p75": quantile(0.75),
        "maximum": ordered[-1],
        "mean": statistics.fmean(ordered),
        "negative_dominant_count": sum(value < 0.0 for value in ordered),
        "exact_tie_count": sum(value == 0.0 for value in ordered),
        "positive_dominant_count": sum(value > 0.0 for value in ordered),
    }


def _load_checkpoints(
    roots: Sequence[Path],
    *,
    expected_checkpoint_sha256: Mapping[tuple[str, str], str],
) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
    candidates: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    status_counts: Counter[str] = Counter()
    for root in roots:
        if not root.is_dir():
            raise SummaryError(f"checkpoint root is absent: {root}")
        for path in sorted(root.glob("*.json")):
            row = _read_object(path)
            if (
                row.get("schema") != CHECKPOINT_SCHEMA
                or row.get("base_archive_sha256") != EXPECTED_BASE_SHA256
                or row.get("threads") != 4
                or row.get("seed") != 1234
                or row.get("deterministic_algorithms") is not True
                or row.get("score_claim") is not False
            ):
                raise SummaryError(f"checkpoint custody differs: {path}")
            key = (str(row["receiver_actuator_id"]), str(row["direction_id"]))
            if key[1] not in DIRECTIONS:
                raise SummaryError(f"malformed checkpoint key: {key}")
            artifact = row.get("event_artifact")
            if artifact is not None:
                artifact_path = Path(str(artifact["path"]))
                if _sha256(artifact_path) != artifact["sha256"]:
                    raise SummaryError(f"event artifact SHA differs: {artifact_path}")
            checkpoint_sha = _sha256(path)
            by_sha = candidates.setdefault(key, {})
            if checkpoint_sha in by_sha:
                raise SummaryError(f"duplicate checkpoint bytes for key: {key}")
            by_sha[checkpoint_sha] = {
                **row,
                "_checkpoint_path": str(path),
                "_checkpoint_sha256": checkpoint_sha,
            }

    result: dict[tuple[str, str], dict[str, Any]] = {}
    for key, expected_sha in expected_checkpoint_sha256.items():
        row = candidates.get(key, {}).get(expected_sha)
        if row is None:
            raise SummaryError(f"assignment-bound checkpoint is absent: {key} {expected_sha}")
        result[key] = row
        status_counts[str(row["status"])] += 1
    unexpected = set(candidates) - set(expected_checkpoint_sha256)
    if unexpected:
        raise SummaryError(f"checkpoint roots contain unbound keys: {sorted(unexpected)}")
    digest_rows = [
        str(result[key]["_checkpoint_sha256"])
        for key in sorted(result)
    ]
    superseded_count = sum(len(rows) - 1 for rows in candidates.values())
    return result, {
        "completed_probe_count": len(result),
        "required_probe_count": len(expected_checkpoint_sha256),
        "superseded_checkpoint_count": superseded_count,
        "status_counts": dict(sorted(status_counts.items())),
        "checkpoint_digest_chain_sha256": hashlib.sha256(
            "".join(digest_rows).encode("ascii")
        ).hexdigest(),
    }


def _g3_coverage(
    table_rows: Sequence[Mapping[str, Any]],
    top24: Sequence[int],
) -> dict[str, Any]:
    pair_rows = []
    missing_blocks = []
    for pair_id in top24:
        required = sorted(
            str(row["bucket_id"])
            for row in table_rows
            if pair_id in row["pf2_membership_pair_ids"]
        )
        joined = sorted(
            str(row["bucket_id"])
            for row in table_rows
            if pair_id in row["pf2_membership_pair_ids"] and pair_id in row["pair_ids"]
        )
        missing = sorted(set(required) - set(joined))
        pair_rows.append(
            {
                "pair_id": pair_id,
                "required_bucket_count": len(required),
                "joined_bucket_count": len(joined),
                "missing_bucket_ids": missing,
            }
        )
        missing_blocks.extend(
            {"pair_id": pair_id, "bucket_id": bucket_id} for bucket_id in missing
        )
    return {
        "definition": (
            "A G3 hard block is joined only when the exact hard pair belongs to "
            "the PF2 bucket and also appears in that bucket row's measured joined pair_ids."
        ),
        "top24_pair_count": len(top24),
        "fully_joined_pair_count": sum(not row["missing_bucket_ids"] for row in pair_rows),
        "missing_block_count": len(missing_blocks),
        "coverage_proven": not missing_blocks,
        "pairs": pair_rows,
        "missing_blocks": missing_blocks,
    }


def _candidate_coordinate_families(atlas_key: Mapping[str, Any]) -> list[str]:
    class_pair = str(atlas_key["class_pair"])
    temporal = str(atlas_key["g4_temporal_class"])
    stratum = str(atlas_key["class_stratum"])
    families = []
    if "Lane" in class_pair:
        families.append("LANE_PROGRAM_BAND_COORDINATE")
    if "Movable" in class_pair:
        families.append("BOUNDED_G1_POLYGON_COORDINATE")
    if temporal == "TRANSIENT":
        families.append("PAIR_LOCAL_POST_SOLVE_CORRECTION")
    if stratum == "boundary":
        families.append("EVENT_LOCAL_SKELETON_BOUNDARY_PRODUCTION")
    else:
        families.append("PER_STRATUM_SKELETON_AMPLITUDE_FIELD")
    return families


def _coordinate_derivation(
    table_rows: Sequence[Mapping[str, Any]],
    coverage: Mapping[str, Any],
    *,
    rg2_assignment: Mapping[str, Any] | None,
    rg3_assignment: Mapping[str, Any] | None = None,
    checkpoints: Mapping[tuple[str, str], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    by_bucket = {str(row["bucket_id"]): row for row in table_rows}
    rg2_by_key = (
        {
            (int(row["pair_id"]), str(row["bucket_id"])): row
            for row in rg2_assignment["rows"]
        }
        if rg2_assignment is not None
        else {}
    )
    rg3_by_key = (
        {
            (int(row["pair_id"]), str(row["bucket_id"])): row
            for row in rg3_assignment["rows"]
        }
        if rg3_assignment is not None
        else {}
    )
    residue = []
    for missing in coverage["missing_blocks"]:
        bucket_id = str(missing["bucket_id"])
        pair_id = int(missing["pair_id"])
        row = by_bucket[bucket_id]
        atlas_key = row["atlas_key"]
        rg2_row = rg2_by_key.get((pair_id, bucket_id))
        rg3_row = rg3_by_key.get((pair_id, bucket_id))
        unreachable = (
            rg2_row is not None
            and rg2_row["causal_join_status"]
            == "UNREACHABLE_NO_SHA_BOUND_RECEIVER_CLASS_PAIR_SUPPORT"
        )
        if rg3_assignment is not None:
            next_families = []
            reason = "RG3_MEASURED_RESIDUAL_DID_NOT_CAUSALLY_JOIN_EXACT_PAIR_BUCKET"
        elif unreachable:
            next_families = ["EVENT_LOCAL_SKELETON_CLASS_BIRTH_PRODUCTION"]
            reason = "RG2_ADDRESS_UNREACHABLE_ZERO_SHA_BOUND_CLASS_PAIR_SUPPORT"
        elif rg2_assignment is not None:
            next_families = (
                ["FINER_EVENT_LOCAL_SKELETON_AMPLITUDE_CODEBOOK"]
                if str(atlas_key["class_stratum"]) == "boundary"
                else ["FISHER_MARGIN_PER_STRATUM_SKELETON_AMPLITUDE_CODEBOOK"]
            )
            reason = "NO_MEASURED_RG2_AMPLITUDE_JOIN_AT_EXACT_PAIR_BUCKET"
        else:
            next_families = _candidate_coordinate_families(atlas_key)
            reason = "NO_MEASURED_RG1_PROBE_JOIN_AT_EXACT_PAIR_BUCKET"
        residual_row = {
                "pair_id": pair_id,
                "bucket_id": bucket_id,
                "typed_key": dict(atlas_key),
                "candidate_coordinate_families": next_families,
                "reason": reason,
                **(
                    {
                        "rg2_receiver_actuator_id": rg2_row["receiver_actuator_id"],
                        "rg2_receiver_derived_row_band": rg2_row["receiver_derived_row_band"],
                    }
                    if rg2_row is not None
                    else {}
                ),
                **(
                    {
                        "rg3_family": rg3_row["selected_coordinate_family"],
                        "rg3_receiver_actuator_ids": rg3_row[
                            "receiver_actuator_ids"
                        ],
                    }
                    if rg3_row is not None
                    else {}
                ),
            }
        if rg3_row is not None and checkpoints is not None:
            probe_evidence = []
            exact_join_seen = False
            target_bucket_seen = False
            for actuator_id in rg3_row["receiver_actuator_ids"]:
                for direction_id in DIRECTIONS:
                    probe = checkpoints.get((str(actuator_id), direction_id))
                    if probe is None:
                        raise SummaryError(
                            f"RG3 assignment-bound checkpoint is absent: "
                            f"{actuator_id} {direction_id}"
                        )
                    target_hits = [
                        hit
                        for hit in probe["bucket_hits"]
                        if str(hit["bucket_id"]) == bucket_id
                    ]
                    target_bucket_seen = target_bucket_seen or bool(target_hits)
                    joined = any(pair_id in hit["pair_ids"] for hit in target_hits)
                    exact_join_seen = exact_join_seen or joined
                    probe_evidence.append(
                        {
                            "receiver_actuator_id": actuator_id,
                            "direction_id": direction_id,
                            "status": probe["status"],
                            "target_bucket_hit": bool(target_hits),
                            "target_pair_joined": joined,
                            "target_bucket_event_count": sum(
                                int(hit["event_count"]) for hit in target_hits
                            ),
                            "checkpoint_sha256": probe["_checkpoint_sha256"],
                        }
                    )
            if exact_join_seen:
                raise SummaryError(
                    f"RG3 residual unexpectedly contains an exact joined row: "
                    f"{pair_id} {bucket_id}"
                )
            statuses = Counter(str(row["status"]) for row in probe_evidence)
            if set(statuses) == {"MEASURED_EMPTY_RASTER_SUPPORT"}:
                blocker = "NO_RECEIVER_RASTER_SUPPORT_IN_ANY_COUNTED_RG3_MAGNITUDE_OR_SIGN"
            elif not target_bucket_seen:
                blocker = "NO_TARGET_BUCKET_EVENT_CHANGED_BY_ANY_COUNTED_RG3_MAGNITUDE_OR_SIGN"
            else:
                blocker = "TARGET_BUCKET_CHANGED_BUT_REQUIRED_PAIR_NEVER_JOINED"
            derived_next_family = {
                "EVENT_LOCAL_SKELETON_CLASS_BIRTH_PRODUCTION": (
                    "WORLDSHEET_EVENT_INDEXED_TYPED_INTERFACE_ARC"
                ),
                "FINER_EVENT_LOCAL_SKELETON_AMPLITUDE_CODEBOOK": (
                    "CURVELET_OR_SHEARLET_BOUNDARY_ARC_CODEBOOK"
                ),
                "FISHER_MARGIN_PER_STRATUM_SKELETON_AMPLITUDE_CODEBOOK": (
                    "FISHER_MARGIN_SITE_LOCAL_PER_STRATUM_CODEBOOK"
                ),
            }[rg3_row["selected_coordinate_family"]]
            residual_row["rg3_probe_blocker"] = {
                "classification": blocker,
                "status_counts": dict(sorted(statuses.items())),
                "probe_count": len(probe_evidence),
                "probes": probe_evidence,
                "derived_next_coordinate_family": derived_next_family,
                "next_coordinate_family_status": (
                    "ADVISORY_DERIVATION_ONLY; NO_RG4_AUTHORIZED"
                ),
            }
        residue.append(residual_row)
    if rg3_assignment is not None:
        derivation_rule = (
            "RG3 is the authorized terminal residual-family pass. Any exact "
            "pair-bucket block still missing after both signs of every counted "
            "RG3 magnitude is reported as a scoped blocker; no RG4 family is inferred."
        )
        coordinate_counts = {}
        verdict_scope = "INSTANCE_EXTENDED_GRAMMAR_RG3"
    elif rg2_assignment is not None:
        derivation_rule = (
            "RG2 amplitude rows that remain unjoined after both signed one-quantum "
            "measurements require a finer event-local or Fisher-margin per-stratum "
            "amplitude codebook. Rows with zero SHA-bound support require a distinct "
            "class-birth production; RG2 does not invent an address for them."
        )
        coordinate_counts = {
            family: sum(
                family in row["candidate_coordinate_families"] for row in residue
            )
            for family in sorted(
                {
                    family
                    for row in residue
                    for family in row["candidate_coordinate_families"]
                }
            )
        }
        verdict_scope = "INSTANCE_EXTENDED_GRAMMAR_RG2"
    else:
        derivation_rule = (
            "Lane class keys require counted Lane-band productions; Movable keys "
            "require bounded G1 polygon coordinates; transient keys require pair-local "
            "post-solve corrections; remaining boundary/cell keys require event-local "
            "or per-stratum SKELETON production coordinates."
        )
        coordinate_counts = {
            "lane_program": 24,
            "pair_local_post_solve_correction": 6,
            "bounded_geometry_alternative": 10,
        }
        verdict_scope = "INSTANCE_EXTENDED_GRAMMAR_RG1"
    return {
        "derivation_rule": derivation_rule,
        "next_coordinate_family_counts": coordinate_counts,
        "residual_missing_block_count": len(residue),
        "residual": residue,
        "next_authorized_family_status": (
            "NO_RG4_AUTHORIZED" if rg3_assignment is not None and residue else "NOT_APPLICABLE"
        ),
        "verdict_scope": verdict_scope,
    }


def _assigned_bucket_summary(row: Mapping[str, Any]) -> dict[str, Any]:
    assignments = row.get("measured_probe_assignments")
    if not isinstance(assignments, list) or not assignments:
        raise SummaryError("assigned bucket summary requires measured probe assignments")
    return {
        "bucket_id": row["bucket_id"],
        "assignment_status": row["assignment_status"],
        "assignment_row_count": len(assignments),
        "unique_actuator_count": len(row["receiver_actuator_ids"]),
        "direction_count": len(row["direction_ids"]),
        "joined_pair_count": len(row["pair_ids"]),
        "probe_event_incidence_count": sum(
            int(assignment["perturbed_event_count"]) for assignment in assignments
        ),
        "probe_event_incidence_semantics": (
            "Sum over actuator-direction assignment rows; the same raw PF2 event "
            "may occur in multiple signed probes, so this is not unique-event cardinality."
        ),
    }


def build_summary(
    *,
    checkpoint_roots: Sequence[Path],
    assignment_table_path: Path,
    g3_path: Path,
    rg2_assignment_path: Path | None = None,
    rg3_assignment_path: Path | None = None,
) -> dict[str, Any]:
    table = _read_object(assignment_table_path)
    if table.get("schema") != TABLE_SCHEMA:
        raise SummaryError("assignment-table schema differs")
    validate_assignment_table(table, expected_pf2_sha256=EXPECTED_PF2_SHA256)
    g3_sha = _sha256(g3_path)
    g3 = _read_object(g3_path)
    if g3_sha != EXPECTED_G3_SHA256 or g3.get("schema") != G3_SCHEMA:
        raise SummaryError("G3 registry custody differs")
    top24 = g3.get("top24")
    if not isinstance(top24, list) or len(top24) != 24:
        raise SummaryError("G3 top24 registry is malformed")
    rg2_assignment = (
        _read_object(rg2_assignment_path) if rg2_assignment_path is not None else None
    )
    if (
        rg2_assignment is not None
        and rg2_assignment.get("schema") != RG2_ASSIGNMENT_SCHEMA
    ):
        raise SummaryError("RG2 assignment schema differs")
    rg3_assignment = (
        _read_object(rg3_assignment_path) if rg3_assignment_path is not None else None
    )
    if (
        rg3_assignment is not None
        and rg3_assignment.get("schema") != RG3_ASSIGNMENT_SCHEMA
    ):
        raise SummaryError("RG3 assignment schema differs")

    table_probe_sha = {
        (str(row["receiver_actuator_id"]), str(row["direction_id"])): str(row["checkpoint_sha256"])
        for row in table["probe_results"]
    }
    checkpoints, sweep = _load_checkpoints(
        checkpoint_roots,
        expected_checkpoint_sha256=table_probe_sha,
    )
    checkpoint_sha = {key: str(row["_checkpoint_sha256"]) for key, row in checkpoints.items()}
    if table_probe_sha != checkpoint_sha:
        raise SummaryError("assignment table does not bind the complete current checkpoint set")

    actuator_ids = list(table["foreign_key_vocabulary"]["receiver_actuator_stable_ids"])
    actuator_rows = []
    asymmetry_values: dict[str, list[float]] = {}
    status_pair_counts: Counter[str] = Counter()
    for actuator_id in actuator_ids:
        directions = {
            direction: checkpoints.get((actuator_id, direction))
            for direction in DIRECTIONS
        }
        metrics = {
            direction: (_probe_metrics(row) if row is not None else None)
            for direction, row in directions.items()
        }
        statuses = {
            direction: (str(row["status"]) if row is not None else "NOT_MEASURED")
            for direction, row in directions.items()
        }
        status_pair_counts[f"{statuses[DIRECTIONS[0]]}|{statuses[DIRECTIONS[1]]}"] += 1
        paired_measured = all(status.startswith("MEASURED_") for status in statuses.values())
        signed = {}
        if paired_measured:
            negative = metrics[DIRECTIONS[0]]
            positive = metrics[DIRECTIONS[1]]
            assert negative is not None and positive is not None
            for metric in negative:
                value = _signed_asymmetry(negative[metric], positive[metric])
                signed[metric] = value
                asymmetry_values.setdefault(metric, []).append(value)
        actuator_rows.append(
            {
                "receiver_actuator_id": actuator_id,
                "statuses": statuses,
                "support_by_direction": metrics,
                "signed_asymmetry_positive_minus_negative_over_sum": signed,
            }
        )

    table_rows = table["rows"]
    assigned_bucket_rows = []
    for row in table_rows:
        assignments = row["measured_probe_assignments"]
        if assignments:
            assigned_bucket_rows.append(_assigned_bucket_summary(row))

    g3_coverage = _g3_coverage(table_rows, top24)
    summary: dict[str, Any] = {
        "schema": SCHEMA,
        "input_custody": {
            "checkpoint_roots": [str(root.resolve()) for root in checkpoint_roots],
            "checkpoint_digest_chain_sha256": sweep["checkpoint_digest_chain_sha256"],
            "assignment_table": {
                "path": str(assignment_table_path.resolve()),
                "sha256": _sha256(assignment_table_path),
                "content_sha256": table["table_content_sha256"],
            },
            "g3_registry": {
                "path": str(g3_path.resolve()),
                "sha256": g3_sha,
            },
            **(
                {
                    "rg2_assignment": {
                        "path": str(rg2_assignment_path.resolve()),
                        "sha256": _sha256(rg2_assignment_path),
                        "content_sha256": rg2_assignment["assignment_content_sha256"],
                    }
                }
                if rg2_assignment_path is not None and rg2_assignment is not None
                else {}
            ),
            **(
                {
                    "rg3_assignment": {
                        "path": str(rg3_assignment_path.resolve()),
                        "sha256": _sha256(rg3_assignment_path),
                        "content_sha256": rg3_assignment["assignment_content_sha256"],
                    }
                }
                if rg3_assignment_path is not None and rg3_assignment is not None
                else {}
            ),
            "base_archive_sha256": EXPECTED_BASE_SHA256,
            "pf2_receipt_sha256": EXPECTED_PF2_SHA256,
        },
        "probe_sweep": sweep,
        "assignment_coverage": table["coverage"],
        "per_actuator_support": actuator_rows,
        "per_bucket_join_counts": {
            "assigned_bucket_count": len(assigned_bucket_rows),
            "unassigned_bucket_count": len(table_rows) - len(assigned_bucket_rows),
            "assigned_buckets": assigned_bucket_rows,
        },
        "sign_asymmetry": {
            "definition": "(positive_support - negative_support) / (positive_support + negative_support)",
            "paired_status_counts": dict(sorted(status_pair_counts.items())),
            "metric_distributions": {
                metric: _distribution(values)
                for metric, values in sorted(asymmetry_values.items())
            },
        },
        "g3_top24_coverage": g3_coverage,
        "receiver_coordinate_derivation": _coordinate_derivation(
            table_rows,
            g3_coverage,
            rg2_assignment=rg2_assignment,
            rg3_assignment=rg3_assignment,
            checkpoints=checkpoints,
        ),
        "producer_rerun_eligible": False,
        "producer_rerun_reason": "set after the G3 coverage proof below",
        "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
        "score_claim": False,
        "pointer": "0.1910828242 [contest-CPU]",
        "pointer_moved": False,
        "research_only": True,
        "main_landing_review_required": True,
        "verdict_scope": (
            "INSTANCE_V19C_RG3_RESIDUAL_FAMILY_SIGNED_MAGNITUDE_SWEEP"
            if rg3_assignment is not None
            else "INSTANCE_V19C_RG2_SKELETON_AMPLITUDE_ONE_QUANTUM_SWEEP"
            if rg2_assignment is not None
            else "INSTANCE_V19C_RG1_ENDPOINT_ONE_QUANTUM_SWEEP"
            if len(checkpoints) > 748
            else "INSTANCE_V19C_ENDPOINT_ONE_QUANTUM_SWEEP"
        ),
    }
    eligible = bool(summary["g3_top24_coverage"]["coverage_proven"])
    summary["producer_rerun_eligible"] = eligible
    summary["producer_rerun_reason"] = (
        "G3 top24 exact pair-by-bucket assignment coverage proven."
        if eligible
        else "MS4 held: exact G3 top24 pair-by-bucket assignment blocks remain missing."
    )
    summary["summary_content_sha256"] = canonical_sha256(summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        action="append",
        dest="checkpoint_roots",
        help="Repeat for split checkpoint custody; assignment-table SHA bindings select rows.",
    )
    parser.add_argument("--assignment-table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--g3", type=Path, default=DEFAULT_G3)
    parser.add_argument("--rg2-assignment", type=Path)
    parser.add_argument("--rg3-assignment", type=Path)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = build_summary(
        checkpoint_roots=args.checkpoint_roots or [DEFAULT_CHECKPOINTS],
        assignment_table_path=args.assignment_table,
        g3_path=args.g3,
        rg2_assignment_path=args.rg2_assignment,
        rg3_assignment_path=args.rg3_assignment,
    )
    _publish(args.output, summary)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
