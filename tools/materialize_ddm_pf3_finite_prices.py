#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Materialize existing RG3 coordinates into PF3 five-edge measurements.

This is a local-CPU, research-only replay.  It rebuilds only SHA-bound RG3
coordinates that already produced real PF2 event hits, measures their exact
V19C same-object scorer deltas, and races exact parse-back coders.  Candidate
archives and immutable per-coordinate checkpoints live on the primary SSD.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, StrictInt, StrictStr

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO / "src", REPO):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap  # noqa: E402
from tac.optimization.ddm_ms7_receiver_edges import (  # noqa: E402
    _race_row,
    decode_coded_receiver_object,
    encode_brotli_q11,
    race_same_receiver_object,
)
from tac.optimization.ddm_pf3_finite_price_materialization import (  # noqa: E402
    POSE_COORDINATES,
    SCORED_PIXELS,
    build_rd1_fail_closed_backfill,
    conditional_coordinate_price,
    joint_distortion_delta,
    materialized_bucket_report,
)
from tac.optimization.direct_description_entropy_priced_member import (  # noqa: E402
    CLASS_NAMES,
    rfc8785_canonicalize,
)
from tools.measure_ddm_ms6_receiver_support import (  # noqa: E402
    _compile_probe,
    _fast_receiver,
    _probe_context,
)
from tools.measure_ddm_v14_realization_fidelity import (  # noqa: E402
    DDMV14RealizationFidelityConfigV1,
    _forward,
    _load_models,
)

RUN_ID: Final = "ddm_pf3_finite_price_materialization_20260725T193409Z"
LANE_ID: Final = "ddm_pf3_finite_price_materialization"
DELEGATION_KEY: Final = (
    "codex_delegate:ddm_pf3_finite_price_materialization:20260725T193409Z"
)
CONFIG = REPO / ".omx/research/configs/ddm_pf3_finite_price_materialization_20260725.json"
OUTPUT = REPO / ".omx/research" / RUN_ID
HEIGHT: Final = 384
WIDTH: Final = 512
PAIR_COUNT: Final = 600
CHECKPOINT_SCHEMA: Final = "ddm_pf3_coordinate_measurement_checkpoint.v1"
RECEIPT_SCHEMA: Final = "ddm_pf3_finite_price_materialization_receipt.v1"
BASE_CODER_FRAME_MANIFEST_SCHEMA: Final = "ddm_pf3_base_coder_frame_manifest.v1"


class PF3MeasurementError(ValueError):
    """Typed PF3 source, replay, or custody failure."""


class DDMF3FinitePriceConfigV1(BaseModel):
    """Strict local-only PF3 materialization contract."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema: Literal["DDMF3FinitePriceConfigV1"]
    run_id: Literal[RUN_ID]
    lane_id: Literal[LANE_ID]
    delegation_checkpoint_key: Literal[DELEGATION_KEY]
    rg3_receipt_path: StrictStr
    rg3_receipt_sha256: StrictStr
    rg3_assignment_path: StrictStr
    rg3_assignment_sha256: StrictStr
    direct_metric_path: StrictStr
    direct_metric_sha256: StrictStr
    rd1_duals_path: StrictStr
    rd1_duals_sha256: StrictStr
    base_archive_path: StrictStr
    base_archive_sha256: StrictStr
    base_receipt_path: StrictStr
    base_receipt_sha256: StrictStr
    scorer_config_path: StrictStr
    scorer_config_sha256: StrictStr
    bulk_root: StrictStr
    minimum_free_bytes: StrictInt
    seed: Literal[1234]
    scorer_threads: Literal[4]
    scorer_batch_size: Literal[16]
    expected_rg3_probe_count: Literal[124]
    expected_event_hit_probe_count: Literal[68]
    expected_occupied_bucket_count: Literal[37]
    expected_rd1_cell_count: Literal[162]
    top_mass_first: Literal[True]
    research_only: Literal[True]
    execution_allowed: Literal[False]
    score_claim: Literal[False]
    main_review_required: Literal[True]
    local_cost_usd: Literal[0]
    no_rg5: Literal[True]
    stretch_allowed_only_after_finite_rd1_price: Literal[True]


def _resolve(raw: str) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else REPO / path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_bound_json(path: Path, expected_sha256: str, role: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink() or _sha256_file(path) != expected_sha256:
        raise PF3MeasurementError(f"{role} SHA/path custody differs")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise PF3MeasurementError(f"{role} must be a JSON object")
    return value


def _read_config(path: Path) -> tuple[DDMF3FinitePriceConfigV1, str]:
    payload = path.read_bytes()
    config = DDMF3FinitePriceConfigV1.model_validate_json(payload)
    if not Path(config.bulk_root).is_absolute() or not config.bulk_root.startswith(
        "/Volumes/VertigoDataTier/pact/"
    ):
        raise PF3MeasurementError("PF3 bulk root must use the primary SSD tier")
    return config, hashlib.sha256(payload).hexdigest()


def _publish(path: Path, value: Mapping[str, Any] | bytes) -> None:
    payload = bytes(value) if isinstance(value, bytes) else rfc8785_canonicalize(dict(value))
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise PF3MeasurementError(f"immutable output differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _artifact(path: Path, role: str) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    try:
        display = str(resolved.relative_to(REPO))
    except ValueError:
        display = str(resolved)
    return {
        "path": display,
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
        "role": role,
    }


def _storage_preflight(config: DDMF3FinitePriceConfigV1) -> dict[str, Any]:
    bulk = Path(config.bulk_root)
    bulk.mkdir(parents=True, exist_ok=True)
    checkpoint = bulk / "stage_checkpoints/00_storage_preflight.json"
    usage = shutil.disk_usage(bulk)
    if usage.free < config.minimum_free_bytes:
        raise PF3MeasurementError("PF3 SSD preflight has insufficient free bytes")
    if checkpoint.exists():
        stored = json.loads(checkpoint.read_bytes())
        if (
            stored.get("selected_tier") != str(bulk)
            or stored.get("minimum_free_bytes") != config.minimum_free_bytes
            or stored.get("passed") is not True
        ):
            raise PF3MeasurementError("PF3 storage-preflight resume differs")
        return stored
    receipt_path = OUTPUT / "receipt.json"
    if receipt_path.exists():
        prior = json.loads(receipt_path.read_bytes()).get("storage_preflight")
        if (
            not isinstance(prior, dict)
            or prior.get("selected_tier") != str(bulk)
            or prior.get("minimum_free_bytes") != config.minimum_free_bytes
            or prior.get("passed") is not True
        ):
            raise PF3MeasurementError("PF3 prior receipt storage preflight differs")
        _publish(checkpoint, prior)
        return prior
    value = {
        "selected_tier": str(bulk),
        "free_bytes": usage.free,
        "minimum_free_bytes": config.minimum_free_bytes,
        "passed": True,
        "cleanup_policy": (
            "candidate archives and exact coder winners are durable measurement "
            "artifacts; per-process temporary files are atomically replaced"
        ),
    }
    _publish(checkpoint, value)
    return value


def _endpoint(receipt: Mapping[str, Any]) -> tuple[int, float]:
    curve = receipt.get("curve")
    endpoint = curve.get("n600_endpoint") if isinstance(curve, Mapping) else None
    c1 = receipt.get("c1_bucket_attribution")
    final = c1.get("measured_v19c_final") if isinstance(c1, Mapping) else None
    if (
        not isinstance(endpoint, Mapping)
        or not isinstance(final, Mapping)
        or not isinstance(final.get("residual_errors"), int)
        or not isinstance(final.get("role_errors"), int)
        or not isinstance(endpoint.get("d_pose"), str)
        or not isinstance(endpoint.get("d_seg"), str)
    ):
        raise PF3MeasurementError("V19C endpoint custody is malformed")
    errors = int(final["residual_errors"]) + int(final["role_errors"])
    d_pose = float(endpoint["d_pose"])
    if not math.isclose(
        errors / SCORED_PIXELS,
        float(endpoint["d_seg"]),
        rel_tol=0.0,
        abs_tol=5e-13,
    ):
        raise PF3MeasurementError("V19C endpoint Seg count and d_seg differ")
    return errors, d_pose


def _load_inventory(
    config: DDMF3FinitePriceConfigV1,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    rg3_receipt_path = _resolve(config.rg3_receipt_path)
    receipt = _read_bound_json(
        rg3_receipt_path,
        config.rg3_receipt_sha256,
        "RG3 measurement receipt",
    )
    assignment_path = _resolve(config.rg3_assignment_path)
    assignment = _read_bound_json(
        assignment_path,
        config.rg3_assignment_sha256,
        "RG3 coordinate assignment",
    )
    direct = _read_bound_json(
        _resolve(config.direct_metric_path),
        config.direct_metric_sha256,
        "MS4D direct metric",
    )
    rd1 = _read_bound_json(
        _resolve(config.rd1_duals_path),
        config.rd1_duals_sha256,
        "RD1 typed dual source",
    )
    table_ref = receipt.get("assignment_table")
    sweep = receipt.get("probe_sweep")
    if (
        receipt.get("schema") != "ddm_ms5_pf2_bucket_assignment_receipt.v1"
        or receipt.get("measurement_schema")
        != "ddm_ms6_receiver_support_measurement_receipt.v1"
        or not isinstance(table_ref, Mapping)
        or not isinstance(sweep, Mapping)
        or assignment.get("schema") != "ddm_rg3_residual_family_assignment.v1"
        or assignment.get("new_signed_probe_count") != config.expected_rg3_probe_count
    ):
        raise PF3MeasurementError("RG3 receipt/assignment schema differs")
    table_path = _resolve(str(table_ref["path"]))
    table = _read_bound_json(table_path, str(table_ref["file_sha256"]), "PF2 assignment table")
    actuator_owner = {
        actuator_id: str(row["bucket_id"])
        for row in assignment.get("rows", ())
        for actuator_id in row.get("receiver_actuator_ids", ())
    }
    if len(actuator_owner) != 62:
        raise PF3MeasurementError("RG3 actuator vocabulary is not the sealed 62 coordinates")
    results = [
        dict(row)
        for row in table.get("probe_results", ())
        if row.get("receiver_actuator_id") in actuator_owner
    ]
    if len(results) != config.expected_rg3_probe_count:
        raise PF3MeasurementError("PF3 did not recover the sealed 124 signed RG3 probes")
    checkpoint_root = Path(str(sweep["checkpoint_root"]))
    inventory = []
    for result in sorted(
        results,
        key=lambda row: (str(row["receiver_actuator_id"]), str(row["direction_id"])),
    ):
        safe = str(result["receiver_actuator_id"]).replace(".", "_")
        checkpoint_path = checkpoint_root / f"{safe}__{result['direction_id']}.json"
        checkpoint = _read_bound_json(
            checkpoint_path,
            str(result["checkpoint_sha256"]),
            "RG3 per-probe checkpoint",
        )
        if (
            checkpoint.get("receiver_actuator_id") != result["receiver_actuator_id"]
            or checkpoint.get("direction_id") != result["direction_id"]
            or checkpoint.get("status") != result["status"]
        ):
            raise PF3MeasurementError("RG3 probe-result/checkpoint identity differs")
        event = checkpoint.get("event_artifact")
        if isinstance(event, Mapping):
            event_path = Path(str(event["path"]))
            if (
                not event_path.is_file()
                or event_path.is_symlink()
                or _sha256_file(event_path) != event["sha256"]
            ):
                raise PF3MeasurementError("RG3 event artifact custody differs")
        inventory.append(
            {
                **result,
                "owner_bucket_id": actuator_owner[str(result["receiver_actuator_id"])],
                "source_checkpoint": checkpoint,
                "source_checkpoint_path": str(checkpoint_path),
            }
        )
    hit = [
        row
        for row in inventory
        if row["status"] == "MEASURED_ARGMAX_PERTURBATION"
    ]
    if len(hit) != config.expected_event_hit_probe_count:
        raise PF3MeasurementError("PF3 event-hit inventory is not the sealed 68 probes")
    occupied = [
        dict(row)
        for row in direct.get("rows", ())
        if isinstance(row.get("event_count"), int) and row["event_count"] > 0
    ]
    rd1_rows = rd1.get("dimension_duals", {}).get("bucket_rows")
    if (
        len(occupied) != config.expected_occupied_bucket_count
        or not isinstance(rd1_rows, list)
        or len(rd1_rows) != config.expected_rd1_cell_count
    ):
        raise PF3MeasurementError("MS4D/RD1 source cardinality differs")
    event_mass = {str(row["bucket_id"]): int(row["event_count"]) for row in occupied}
    hit.sort(
        key=lambda row: (
            -max(
                event_mass.get(str(bucket["bucket_id"]), 0)
                for bucket in row["source_checkpoint"]["bucket_hits"]
            ),
            str(row["receiver_actuator_id"]),
            str(row["direction_id"]),
        )
    )
    universe = {
        str(bucket["bucket_id"])
        for row in hit
        for bucket in row["source_checkpoint"]["bucket_hits"]
    }
    uncovered = set(universe)
    greedy_cover: list[dict[str, Any]] = []
    remaining = list(hit)
    while uncovered:
        selected = min(
            remaining,
            key=lambda row: (
                -sum(
                    event_mass.get(str(bucket["bucket_id"]), 0)
                    for bucket in row["source_checkpoint"]["bucket_hits"]
                    if bucket["bucket_id"] in uncovered
                ),
                -sum(
                    bucket["bucket_id"] in uncovered
                    for bucket in row["source_checkpoint"]["bucket_hits"]
                ),
                str(row["receiver_actuator_id"]),
                str(row["direction_id"]),
            ),
        )
        covered = {
            str(bucket["bucket_id"])
            for bucket in selected["source_checkpoint"]["bucket_hits"]
        }
        if not covered & uncovered:
            raise PF3MeasurementError("existing RG3 set cover stopped before coverage")
        greedy_cover.append(selected)
        uncovered -= covered
        remaining.remove(selected)
    owner_exact = [
        row
        for row in hit
        if row["owner_bucket_id"]
        in {
            str(bucket["bucket_id"])
            for bucket in row["source_checkpoint"]["bucket_hits"]
        }
    ]
    # The first five top-mass rows are deterministic scorer/coder controls.
    # This fixed inclusion also makes interrupted replays byte-stable.
    selected_identities = {
        (row["receiver_actuator_id"], row["direction_id"])
        for row in [*hit[:5], *greedy_cover, *owner_exact]
    }
    selected_inventory = [
        row
        for row in hit
        if (row["receiver_actuator_id"], row["direction_id"])
        in selected_identities
    ]
    selected_coverage = {
        str(bucket["bucket_id"])
        for row in selected_inventory
        for bucket in row["source_checkpoint"]["bucket_hits"]
    }
    if selected_coverage != universe:
        raise PF3MeasurementError("selected PF3 coordinates do not preserve RG3 bucket reach")
    # The SHA-bound `table` is returned so callers reuse the verified bytes
    # instead of re-reading the path unbound; unverified bytes must never reach
    # a published custody record.
    return selected_inventory, receipt, assignment, direct, rd1, table


def _verify_scorer_config(
    config: DDMF3FinitePriceConfigV1,
) -> tuple[DDMV14RealizationFidelityConfigV1, dict[str, Any]]:
    path = _resolve(config.scorer_config_path)
    if _sha256_file(path) != config.scorer_config_sha256:
        raise PF3MeasurementError("frozen scorer config SHA differs")
    scorer_config = DDMV14RealizationFidelityConfigV1.model_validate_json(path.read_bytes())
    if (
        scorer_config.seed != config.seed
        or scorer_config.scorer_threads != config.scorer_threads
        or scorer_config.scorer_batch_size != config.scorer_batch_size
        or scorer_config.pair_count != PAIR_COUNT
    ):
        raise PF3MeasurementError("frozen scorer geometry differs from PF3")
    target = Path(scorer_config.target_cache_path)
    if (
        not target.is_file()
        or target.is_symlink()
        or target.stat().st_size != scorer_config.target_cache_bytes
        or _sha256_file(target) != scorer_config.target_cache_sha256
    ):
        raise PF3MeasurementError("n600 target-cache custody differs")
    return scorer_config, _artifact(target, "n600 GT Seg labels and Pose6 targets")


def _event_measurements(
    *,
    source_checkpoint: Mapping[str, Any],
    pair_id: int,
    base_cells: np.ndarray,
    candidate_cells: np.ndarray,
    labels: np.ndarray,
    selected_local: int,
) -> list[dict[str, Any]]:
    event_ref = source_checkpoint.get("event_artifact")
    if not isinstance(event_ref, Mapping):
        raise PF3MeasurementError("event-hitting RG3 checkpoint lacks event artifact")
    event_path = Path(str(event_ref["path"]))
    output = []
    with np.load(event_path, allow_pickle=False) as archive:
        expected = {str(row["bucket_id"]) for row in source_checkpoint["bucket_hits"]}
        if set(archive.files) != expected:
            raise PF3MeasurementError("RG3 event artifact bucket vocabulary differs")
        for hit in source_checkpoint["bucket_hits"]:
            bucket_id = str(hit["bucket_id"])
            event_ids = np.asarray(archive[bucket_id], dtype=np.uint32)
            if (
                event_ids.ndim != 1
                or event_ids.size != int(hit["event_count"])
                or hashlib.sha256(event_ids.astype("<u4", copy=False).tobytes()).hexdigest()
                != hit["event_ids_sha256"]
            ):
                raise PF3MeasurementError("RG3 event IDs differ from checkpoint custody")
            pairs = event_ids.astype(np.uint64) // (HEIGHT * WIDTH)
            if not np.all(pairs == pair_id):
                raise PF3MeasurementError("PF3 coordinate event IDs escaped its exact pair")
            local_flat = (event_ids.astype(np.uint64) % (HEIGHT * WIDTH)).astype(np.intp)
            base = base_cells[selected_local].reshape(-1)[local_flat]
            candidate = candidate_cells[selected_local].reshape(-1)[local_flat]
            target = labels[selected_local].reshape(-1)[local_flat]
            base_error = base != target
            candidate_error = candidate != target
            per_target = {}
            for class_id, class_name in enumerate(CLASS_NAMES):
                mask = target == class_id
                if np.any(mask):
                    per_target[class_name] = {
                        "sites": int(np.count_nonzero(mask)),
                        "base_errors": int(np.count_nonzero(base_error & mask)),
                        "candidate_errors": int(np.count_nonzero(candidate_error & mask)),
                        "delta_errors": int(
                            np.count_nonzero(candidate_error & mask)
                            - np.count_nonzero(base_error & mask)
                        ),
                    }
            output.append(
                {
                    "bucket_id": bucket_id,
                    "event_count": int(event_ids.size),
                    "event_ids_sha256": hit["event_ids_sha256"],
                    "changed_event_cells_verified": int(np.count_nonzero(base != candidate)),
                    "base_event_errors": int(np.count_nonzero(base_error)),
                    "candidate_event_errors": int(np.count_nonzero(candidate_error)),
                    "event_delta_errors": int(
                        np.count_nonzero(candidate_error) - np.count_nonzero(base_error)
                    ),
                    "per_target_stratum": per_target,
                }
            )
    return output


def _pricing_assignment(
    *,
    bucket_measurements: list[dict[str, Any]],
    per_class_delta: Mapping[str, int],
    g4_by_bucket: Mapping[str, str],
    marginal_bytes: int,
    joint_delta: float,
) -> dict[str, Any] | None:
    proposals = []
    for bucket in bucket_measurements:
        for class_name, event_row in bucket["per_target_stratum"].items():
            dimension_delta = 100.0 * int(per_class_delta[class_name]) / SCORED_PIXELS
            price = conditional_coordinate_price(
                delta_counted_bytes=marginal_bytes,
                delta_D_dimension=dimension_delta,
                delta_D_joint=joint_delta,
            )
            proposals.append(
                {
                    "bucket_id": bucket["bucket_id"],
                    "target_stratum": class_name,
                    "g4_temporal_class": g4_by_bucket[bucket["bucket_id"]],
                    "event_delta_errors": event_row["delta_errors"],
                    "dimension_delta_errors": int(per_class_delta[class_name]),
                    **price,
                }
            )
    eligible = [
        row
        for row in proposals
        if row["event_delta_errors"] < 0
        and row["lambda_bytes_per_D_dimension"] is not None
    ]
    if not eligible:
        return {
            "status": "NO_FINITE_CONDITIONAL_COORDINATE_SECANT",
            "proposal_status_counts": dict(Counter(row["status"] for row in proposals)),
        }
    selected = min(
        eligible,
        key=lambda row: (
            float(row["delta_D_joint_guard"]),
            float(row["delta_D_dimension"]),
            int(row["event_delta_errors"]),
            str(row["bucket_id"]),
            str(row["target_stratum"]),
        ),
    )
    return {
        **selected,
        "selection_policy": (
            "event-correcting target stratum; joint-improving exact same-object edge; "
            "largest joint then dimension reduction with deterministic identity tie-break"
        ),
        "actionable_for_rd1": False,
        "rd1_blocker": (
            "PF3_COORDINATE_TO_RD1_EXISTING_DUAL_EDGE_APPLICATION_OPERATOR_ABSENT"
        ),
    }


def _encode_settled_candidate_coder(raw: bytes) -> tuple[dict[str, Any], bytes]:
    """Encode one candidate with the real MS7 race winner.

    MS7 established E4/Brotli-Q11 as the same-object owner and the PF3 base
    race rechecks the complete menu. Re-running slow losing arithmetic coders
    for every coordinate is not needed to measure the E4-owned marginal home;
    no per-candidate winner claim is made.
    """

    row, frame = _race_row(
        codec="E4_BROTLI_Q11",
        raw=raw,
        encoded=encode_brotli_q11(raw),
        implementation="optional:brotli quality=11",
    )
    return (
        {
            "schema": "ddm_pf3_settled_candidate_coder.v1",
            "coder_payload_owner": "DDM_MS7_SAME_OBJECT_RECEIVER_ARCHIVE_PAYLOAD",
            "selected": row,
            "policy": (
                "E4_BROTLI_Q11 settled by MS7 and rechecked by this run's complete "
                "V19C base race; candidate row is a fixed-codec price, not a race"
            ),
        },
        frame,
    )


def _base_coder_frame_manifest(
    *,
    raw: bytes,
    codecs: Sequence[str],
    frames: Mapping[str, bytes],
    race_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Record exactly which base coders produced a frame on this host.

    The optional coders (brotli / constriction / zstandard) legitimately produce
    no frame when their dependency is absent, so "all seven files present" is the
    wrong resume invariant: it turned a missing optional dependency into a
    permanently poisoned bulk directory.  The manifest is the typed truth about
    which codecs were raced, so a later process can tell a by-design partial set
    apart from a truncated one.
    """

    reasons = {
        str(row["codec"]): row.get("unavailable_reason")
        for row in race_rows
        if not row.get("available")
    }
    rows = []
    for codec in codecs:
        frame = frames.get(codec)
        if frame is None:
            rows.append(
                {
                    "codec": codec,
                    "available": False,
                    "framed_bytes": None,
                    "frame_sha256": None,
                    "unavailable_reason": (
                        str(reasons[codec])
                        if reasons.get(codec) is not None
                        else "OPTIONAL_CODER_PRODUCED_NO_FRAME_ON_THIS_HOST"
                    ),
                }
            )
            continue
        rows.append(
            {
                "codec": codec,
                "available": True,
                "framed_bytes": len(frame),
                "frame_sha256": hashlib.sha256(frame).hexdigest(),
                "unavailable_reason": None,
            }
        )
    return {
        "schema": BASE_CODER_FRAME_MANIFEST_SCHEMA,
        "same_object_raw_bytes": len(raw),
        "same_object_raw_sha256": hashlib.sha256(raw).hexdigest(),
        "rows": rows,
    }


def _resume_base_coder_race(
    *,
    raw: bytes,
    manifest: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Rebuild the race from a manifest, verifying every resumed frame."""

    if (
        manifest.get("schema") != BASE_CODER_FRAME_MANIFEST_SCHEMA
        or manifest.get("same_object_raw_sha256") != hashlib.sha256(raw).hexdigest()
    ):
        raise PF3MeasurementError("base coder-frame manifest is not bound to this raw object")
    manifest_rows = manifest.get("rows")
    if not isinstance(manifest_rows, list) or {
        str(row["codec"]) for row in manifest_rows
    } != set(paths):
        raise PF3MeasurementError("base coder-frame manifest codec set differs")
    frames: dict[str, bytes] = {}
    rows: list[dict[str, Any]] = []
    for entry in manifest_rows:
        codec = str(entry["codec"])
        path = paths[codec]
        if not entry.get("available"):
            if path.exists():
                raise PF3MeasurementError(
                    f"base coder-frame manifest marks {codec} unavailable but a frame exists"
                )
            rows.append(
                {
                    "codec": codec,
                    "available": False,
                    "framed_bytes": None,
                    "frame_sha256": None,
                    "parseback_exact": False,
                    "implementation": None,
                    "unavailable_reason": entry.get("unavailable_reason"),
                }
            )
            continue
        if not path.is_file() or path.is_symlink():
            raise PF3MeasurementError(f"resumed {codec} base frame is missing from the bulk set")
        frame = path.read_bytes()
        if (
            hashlib.sha256(frame).hexdigest() != entry.get("frame_sha256")
            or len(frame) != entry.get("framed_bytes")
        ):
            raise PF3MeasurementError(f"resumed {codec} base frame custody differs")
        parsed = frame if codec == "RAW_COMPACT" else decode_coded_receiver_object(frame)
        if parsed != raw:
            raise PF3MeasurementError(f"resumed {codec} base frame parse-back differs")
        frames[codec] = frame
        rows.append(
            {
                "codec": codec,
                "available": True,
                "framed_bytes": len(frame),
                "frame_sha256": hashlib.sha256(frame).hexdigest(),
                "parseback_exact": True,
                "implementation": "resumed exact frame from the recorded PF3 base race",
            }
        )
    eligible = [row for row in rows if row["available"]]
    if not eligible:
        raise PF3MeasurementError("resumed base coder-frame set has no available coder")
    winner = min(eligible, key=lambda row: (int(row["framed_bytes"]), str(row["codec"])))
    race = {
        "schema": "ddm_ms7_same_object_coder_race.v1",
        "same_object_raw_bytes": len(raw),
        "same_object_raw_sha256": hashlib.sha256(raw).hexdigest(),
        "coder_payload_owner": "DDM_MS7_SAME_OBJECT_RECEIVER_ARCHIVE_PAYLOAD",
        "rows": rows,
        "winner": {
            "codec": winner["codec"],
            "framed_bytes": winner["framed_bytes"],
            "frame_sha256": winner["frame_sha256"],
            "parseback_exact": True,
        },
        "resume_source": "PRESERVED_FIRST_PROCESS_EXACT_FRAMES",
        "cross_process_custody_note": (
            "a later full-race attempt produced different ZSTD trained-dictionary "
            "bytes and was refused; preserved first-process frame remains exact"
        ),
    }
    return race, frames


def _load_or_race_base_coders(
    raw: bytes,
    bulk: Path,
) -> tuple[dict[str, Any], dict[str, bytes], dict[str, dict[str, Any]]]:
    """Resume exact base frames instead of retraining ZSTD on every process."""

    codecs = (
        "RAW_COMPACT",
        "ZLIB9",
        "RAW_LZMA1",
        "ORDER1_CONTEXT_ARITHMETIC",
        "E4_BROTLI_Q11",
        "CONSTRICTION_ORDER1_CONTEXT_ANS",
        "ZSTD19_TRAINED_DICTIONARY",
    )
    frame_dir = bulk / "coder_base_frames"
    paths = {codec: frame_dir / f"{codec}.bin" for codec in codecs}
    manifest_path = frame_dir / "_manifest.json"
    existing = {codec: path.exists() for codec, path in paths.items()}
    if manifest_path.is_file():
        race, frames = _resume_base_coder_race(
            raw=raw,
            manifest=json.loads(manifest_path.read_bytes()),
            paths=paths,
        )
    elif all(existing.values()):
        # Legacy bulk directories predate the manifest.  A complete seven-frame
        # set is unambiguous, so accept it and write the manifest so the next
        # process resumes through the typed path.
        race, frames = _resume_base_coder_race(
            raw=raw,
            manifest=_base_coder_frame_manifest(
                raw=raw,
                codecs=codecs,
                frames={codec: paths[codec].read_bytes() for codec in codecs},
                race_rows=(),
            ),
            paths=paths,
        )
        _publish(
            manifest_path,
            _base_coder_frame_manifest(
                raw=raw,
                codecs=codecs,
                frames=frames,
                race_rows=race["rows"],
            ),
        )
    elif any(existing.values()):
        present = sorted(codec for codec, ok in existing.items() if ok)
        raise PF3MeasurementError(
            "partial base coder-frame resume set is unsafe: no manifest at "
            f"{manifest_path} and only {present} present; delete {frame_dir} to re-race"
        )
    else:
        race, frames = race_same_receiver_object(raw)
        frames = {codec: frames[codec] for codec in codecs if codec in frames}
        for codec, frame in sorted(frames.items()):
            _publish(paths[codec], frame)
        _publish(
            manifest_path,
            _base_coder_frame_manifest(
                raw=raw,
                codecs=codecs,
                frames=frames,
                race_rows=race["rows"],
            ),
        )
        race["resume_source"] = "FRESH_COMPLETE_RACE"
        race["cross_process_custody_note"] = None
    artifacts = {
        codec: _artifact(paths[codec], f"V19C base exact parse-back {codec} frame")
        for codec in codecs
        if codec in frames
    }
    return race, frames, artifacts


def _measure_candidate(
    *,
    index: int,
    inventory_row: Mapping[str, Any],
    config: DDMF3FinitePriceConfigV1,
    config_sha256: str,
    context: Any,
    base_receiver: Any,
    endpoint_errors: int,
    endpoint_d_pose: float,
    labels_all: np.ndarray,
    poses_all: np.ndarray,
    segnet: Any,
    posenet: Any,
    scorer_custody: Mapping[str, Any],
    base_batch_cache: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    replayed_batch_starts: set[int],
    base_coder_frames: Mapping[str, bytes],
    g4_by_bucket: Mapping[str, str],
) -> dict[str, Any]:
    bulk = Path(config.bulk_root)
    checkpoint = bulk / "stage_checkpoints/02_candidates" / f"{index:03d}.json"
    if checkpoint.exists():
        value = json.loads(checkpoint.read_bytes())
        if (
            value.get("schema") != CHECKPOINT_SCHEMA
            or value.get("typed_config_sha256") != config_sha256
            or value.get("source_checkpoint_sha256") != inventory_row["checkpoint_sha256"]
        ):
            raise PF3MeasurementError("PF3 candidate resume checkpoint differs")
        batch_ids = value["five_pf3_edges"]["candidate_delta"]["batch_geometry"][
            "pair_ids"
        ]
        replayed_batch_starts.add(int(batch_ids[0]))
        return value
    source = inventory_row["source_checkpoint"]
    archive, pair_ids, infeasible = _compile_probe(
        context,
        actuator_id=str(inventory_row["receiver_actuator_id"]),
        direction_id=str(inventory_row["direction_id"]),
    )
    if archive is None or infeasible is not None or len(pair_ids) != 1:
        raise PF3MeasurementError("sealed event-hitting RG3 probe no longer rebuilds one pair")
    pair_id = int(pair_ids[0])
    archive_sha256 = hashlib.sha256(archive).hexdigest()
    expected_candidate = source.get("candidate_archive")
    if (
        not isinstance(expected_candidate, Mapping)
        or archive_sha256 != expected_candidate.get("sha256")
        or len(archive) != expected_candidate.get("bytes")
    ):
        raise PF3MeasurementError("rebuilt RG3 candidate archive differs from source checkpoint")
    archive_path = bulk / "candidate_archives" / f"{index:03d}_{archive_sha256[:16]}.zip"
    _publish(archive_path, archive)
    batch_start = pair_id // config.scorer_batch_size * config.scorer_batch_size
    batch_stop = min(batch_start + config.scorer_batch_size, PAIR_COUNT)
    batch_ids = tuple(range(batch_start, batch_stop))
    selected_local = batch_ids.index(pair_id)
    cached = base_batch_cache.get(batch_start)
    if cached is None:
        base_camera = base_receiver.render_camera_pairs(batch_ids)
        base_cells, base_pose = _forward(segnet, posenet, base_camera)
        labels = np.asarray(labels_all[np.asarray(batch_ids, dtype=np.intp)], dtype=np.uint8)
        poses = np.asarray(poses_all[np.asarray(batch_ids, dtype=np.intp)], dtype=np.float64)
        cached = (base_cells, base_pose, labels, poses)
        base_batch_cache[batch_start] = cached
    base_cells, base_pose, labels, poses = cached
    candidate_receiver = _fast_receiver(archive)
    base_selected_camera = base_receiver.render_camera_pairs((pair_id,))
    candidate_selected_camera = candidate_receiver.render_camera_pairs((pair_id,))
    uint8_delta = candidate_selected_camera.astype(np.int16) - base_selected_camera.astype(
        np.int16
    )
    nonzero = np.abs(uint8_delta[uint8_delta != 0])
    if nonzero.size == 0:
        raise PF3MeasurementError("RG3 coordinate died before real uint8")
    candidate_camera = candidate_receiver.render_camera_pairs(batch_ids)
    candidate_cells, candidate_pose = _forward(segnet, posenet, candidate_camera)
    exact_replay_performed = batch_start not in replayed_batch_starts
    if exact_replay_performed:
        replay_cells, replay_pose = _forward(segnet, posenet, candidate_camera)
        if not np.array_equal(candidate_cells, replay_cells) or not np.array_equal(
            candidate_pose, replay_pose
        ):
            raise PF3MeasurementError("PF3 scorer replay is nondeterministic")
        replayed_batch_starts.add(batch_start)
    other = np.ones(len(batch_ids), dtype=np.bool_)
    other[selected_local] = False
    if (
        not np.array_equal(candidate_cells[other], base_cells[other])
        or not np.array_equal(candidate_pose[other], base_pose[other])
    ):
        raise PF3MeasurementError("pair-local PF3 coordinate changed another batch member")
    base_errors_batch = int(np.count_nonzero(base_cells != labels))
    candidate_errors_batch = int(np.count_nonzero(candidate_cells != labels))
    delta_errors = candidate_errors_batch - base_errors_batch
    base_pose_sse = float(np.square(base_pose - poses).sum(dtype=np.float64))
    candidate_pose_sse = float(np.square(candidate_pose - poses).sum(dtype=np.float64))
    delta_pose_sse = candidate_pose_sse - base_pose_sse
    candidate_errors = endpoint_errors + delta_errors
    candidate_d_pose = (
        endpoint_d_pose * POSE_COORDINATES + delta_pose_sse
    ) / POSE_COORDINATES
    distortion = joint_distortion_delta(
        base_errors=endpoint_errors,
        candidate_errors=candidate_errors,
        base_d_pose=endpoint_d_pose,
        candidate_d_pose=candidate_d_pose,
    )
    per_class_delta = {}
    for class_id, class_name in enumerate(CLASS_NAMES):
        mask = labels == class_id
        per_class_delta[class_name] = int(
            np.count_nonzero((candidate_cells != labels) & mask)
            - np.count_nonzero((base_cells != labels) & mask)
        )
    bucket_measurements = _event_measurements(
        source_checkpoint=source,
        pair_id=pair_id,
        base_cells=base_cells,
        candidate_cells=candidate_cells,
        labels=labels,
        selected_local=selected_local,
    )
    coder_race, winning_frame = _encode_settled_candidate_coder(archive)
    selected_coder = coder_race["selected"]
    winning_codec = str(selected_coder["codec"])
    if winning_codec not in base_coder_frames:
        raise PF3MeasurementError("candidate winner codec is absent from base same-object race")
    frame_path = bulk / "coder_winners" / f"{index:03d}_{winning_codec}.bin"
    _publish(frame_path, winning_frame)
    marginal_bytes = len(winning_frame) - len(base_coder_frames[winning_codec])
    pricing = _pricing_assignment(
        bucket_measurements=bucket_measurements,
        per_class_delta=per_class_delta,
        g4_by_bucket=g4_by_bucket,
        marginal_bytes=marginal_bytes,
        joint_delta=distortion["delta_D_joint"],
    )
    candidate_id = (
        f"pf3_{str(inventory_row['receiver_actuator_id']).replace('.', '_')}"
        f"__{str(inventory_row['direction_id']).lower()}"
    )
    value = {
        "schema": CHECKPOINT_SCHEMA,
        "run_id": RUN_ID,
        "lane_id": LANE_ID,
        "candidate_id": candidate_id,
        "coordinate_id": inventory_row["receiver_actuator_id"],
        "coordinate_owner_bucket_id": inventory_row["owner_bucket_id"],
        "direction_id": inventory_row["direction_id"],
        "source_checkpoint_path": inventory_row["source_checkpoint_path"],
        "source_checkpoint_sha256": inventory_row["checkpoint_sha256"],
        "typed_config_sha256": config_sha256,
        "pair_id": pair_id,
        "five_edges_complete": True,
        "five_pf3_edges": {
            "receiver_object_builder": {
                "pipeline": [
                    "tools.measure_ddm_ms6_receiver_support:_compile_probe",
                    "tools.measure_ddm_ms6_receiver_support:_probe_context",
                ],
                "candidate_archive": _artifact(
                    archive_path,
                    "deterministically rebuilt one-coordinate V19C receiver object",
                ),
                "source_archive_identity_reproduced": True,
            },
            "realized_uint8_quantum": {
                "changed_channel_values": int(nonzero.size),
                "minimum_nonzero_absolute_step": int(nonzero.min()),
                "maximum_nonzero_absolute_step": int(nonzero.max()),
                "changed_pair_ids": [pair_id],
                "deadzone_crossed": int(nonzero.min()) >= 1,
            },
            "candidate_delta": {
                "base_global_errors": endpoint_errors,
                "candidate_global_errors": candidate_errors,
                "delta_global_errors": delta_errors,
                "base_global_d_pose": endpoint_d_pose,
                "candidate_global_d_pose": candidate_d_pose,
                "delta_global_pose_sse_6d": delta_pose_sse,
                **distortion,
                "per_target_stratum_delta_errors": per_class_delta,
                "changed_argmax_cells": int(np.count_nonzero(candidate_cells != base_cells)),
                "candidate_exact_replay_performed": exact_replay_performed,
                "scorer_batch_geometry_replay_gate": (
                    "ONE_EXACT_REPLAY_PER_DISTINCT_N600_BATCH16_GEOMETRY"
                ),
                "batch_geometry": {
                    "pair_ids": list(batch_ids),
                    "batch_size": len(batch_ids),
                    "selected_pair_local_index": selected_local,
                    "authority": "MATCHES_V19C_N600_BATCH16_ENDPOINT_GEOMETRY",
                },
            },
            "dimension_rate_home": {
                "stream_type": "SKELETON",
                "layer_home": "L3_RASTER",
                "member": "production/residual_family_coordinates.rg3rf",
                "physical_edge": "V19C_BASE_TO_ONE_EXACT_RG3_COORDINATE",
                "same_codec_base_bytes": len(base_coder_frames[winning_codec]),
                "same_codec_candidate_bytes": len(winning_frame),
                "delta_counted_bytes_dimension": marginal_bytes,
                "shared_cost_nonadditive": True,
            },
            "coder_payload_owner": {
                "owner": coder_race["coder_payload_owner"],
                "codec": winning_codec,
                "candidate_frame": _artifact(
                    frame_path,
                    "candidate fixed-owner exact parse-back coder frame",
                ),
                "candidate_codec_race_performed": False,
                "candidate_codec_selection_authority": (
                    "MS7 settled E4 owner plus fresh complete base same-object race"
                ),
                "candidate_parseback_exact": True,
                "base_same_codec_parseback_exact": True,
            },
        },
        "bucket_measurements": bucket_measurements,
        "conditional_coordinate_price": pricing,
        "scorer_custody": dict(scorer_custody),
        "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
        "score_claim": False,
        "research_only": True,
        "verdict_scope": "INSTANCE one existing RG3 coordinate x V19C n600 endpoint",
    }
    _publish(checkpoint, value)
    return value


def materialize(config_path: Path = CONFIG) -> dict[str, Any]:
    config, config_sha256 = _read_config(config_path)
    storage = _storage_preflight(config)
    inventory, rg3_receipt, assignment, direct, rd1, assignment_table = _load_inventory(config)
    bulk = Path(config.bulk_root)
    inventory_checkpoint = {
        "schema": "ddm_pf3_inventory_checkpoint.v1",
        "typed_config_sha256": config_sha256,
        "sealed_rg3_probe_count": config.expected_rg3_probe_count,
        "sealed_event_hit_probe_count": config.expected_event_hit_probe_count,
        "selected_coordinate_count": len(inventory),
        "selection_policy": (
            "top-five controls union weighted top-mass greedy set cover union "
            "coordinate-owner exact bucket hits"
        ),
        "ordered_top_mass_first": True,
        "ordered_source_checkpoint_sha256": [
            row["checkpoint_sha256"] for row in inventory
        ],
        "status": "COMPLETE",
    }
    _publish(
        bulk / "stage_checkpoints/01_selected_inventory_complete.json",
        inventory_checkpoint,
    )
    base_archive_path = _resolve(config.base_archive_path)
    if (
        not base_archive_path.is_file()
        or base_archive_path.is_symlink()
        or _sha256_file(base_archive_path) != config.base_archive_sha256
    ):
        raise PF3MeasurementError("V19C base archive custody differs")
    base_archive = base_archive_path.read_bytes()
    base_receipt = _read_bound_json(
        _resolve(config.base_receipt_path),
        config.base_receipt_sha256,
        "V19C base receipt",
    )
    endpoint_errors, endpoint_d_pose = _endpoint(base_receipt)
    scorer_config, target_custody = _verify_scorer_config(config)
    labels_all = open_stored_npy_memmap(Path(scorer_config.target_cache_path), "lstars")
    poses_all = open_stored_npy_memmap(Path(scorer_config.target_cache_path), "gt_poses")
    segnet, posenet, scorer_custody = _load_models(scorer_config)
    context = _probe_context(base_archive)
    base_receiver = _fast_receiver(base_archive)
    base_race, base_frames, base_frame_artifacts = _load_or_race_base_coders(
        base_archive,
        bulk,
    )
    occupied = [
        dict(row)
        for row in direct["rows"]
        if isinstance(row.get("event_count"), int) and row["event_count"] > 0
    ]
    g4_by_bucket = {
        str(row["bucket_id"]): str(row["g4_temporal_class"]) for row in occupied
    }
    candidate_rows = []
    base_batch_cache: dict[
        int,
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ] = {}
    replayed_batch_starts: set[int] = set()
    for index, inventory_row in enumerate(inventory, start=1):
        candidate_rows.append(
            _measure_candidate(
                index=index,
                inventory_row=inventory_row,
                config=config,
                config_sha256=config_sha256,
                context=context,
                base_receiver=base_receiver,
                endpoint_errors=endpoint_errors,
                endpoint_d_pose=endpoint_d_pose,
                labels_all=labels_all,
                poses_all=poses_all,
                segnet=segnet,
                posenet=posenet,
                scorer_custody=scorer_custody,
                base_batch_cache=base_batch_cache,
                replayed_batch_starts=replayed_batch_starts,
                base_coder_frames=base_frames,
                g4_by_bucket=g4_by_bucket,
            )
        )
    bucket_report = materialized_bucket_report(occupied, candidate_rows)
    # Reuse the SHA-bound table `_load_inventory` already verified.  A bare
    # re-read here dropped both the SHA check and the symlink refusal, and its
    # rows land in `_publish`ed immutable receipt fields.
    table = assignment_table
    probe_rows = [
        row
        for row in table["probe_results"]
        if row["receiver_actuator_id"]
        in {
            actuator
            for assignment_row in assignment["rows"]
            for actuator in assignment_row["receiver_actuator_ids"]
        }
    ]
    for bucket in bucket_report["rows"]:
        if bucket["fully_materialized"]:
            continue
        assignment_rows = [
            row
            for row in assignment["rows"]
            if row["bucket_id"] == bucket["bucket_id"]
        ]
        coordinate_ids = {
            actuator
            for row in assignment_rows
            for actuator in row["receiver_actuator_ids"]
        }
        status_counts = Counter(
            row["status"]
            for row in probe_rows
            if row["receiver_actuator_id"] in coordinate_ids
        )
        bucket["existing_rg3_assignment_row_count"] = len(assignment_rows)
        bucket["existing_rg3_coordinate_count"] = len(coordinate_ids)
        bucket["assigned_coordinate_probe_status_counts"] = dict(status_counts)
        if coordinate_ids:
            bucket["blocker"] = (
                "ASSIGNED_RG3_COORDINATES_BUILD_AND_REALIZE_BUT_ZERO_"
                "TARGET_BUCKET_EVENT_HIT"
            )
            bucket["failed_materialization_leg"] = "MEASURE_TARGET_BUCKET_CAUSAL_JOIN"
        else:
            bucket["blocker"] = "NO_EXISTING_RG1_RG3_COORDINATE_ASSIGNED_TO_BUCKET"
            bucket["failed_materialization_leg"] = "BIND_EXISTING_COORDINATE"
    conditional_rows = [
        {
            "candidate_id": row["candidate_id"],
            "coordinate_id": row["coordinate_id"],
            "direction_id": row["direction_id"],
            **row["conditional_coordinate_price"],
        }
        for row in candidate_rows
        if row.get("conditional_coordinate_price") is not None
    ]
    candidate_checkpoint_artifacts = [
        _artifact(
            bulk / "stage_checkpoints/02_candidates" / f"{index:03d}.json",
            "immutable PF3 same-object coordinate measurement",
        )
        for index in range(1, len(candidate_rows) + 1)
    ]
    candidate_delta_errors = [
        int(row["five_pf3_edges"]["candidate_delta"]["delta_global_errors"])
        for row in candidate_rows
    ]
    candidate_joint_deltas = [
        float(row["five_pf3_edges"]["candidate_delta"]["delta_D_joint"])
        for row in candidate_rows
    ]
    candidate_rate_deltas = [
        int(
            row["five_pf3_edges"]["dimension_rate_home"][
                "delta_counted_bytes_dimension"
            ]
        )
        for row in candidate_rows
    ]
    rd1_rows = rd1["dimension_duals"]["bucket_rows"]
    rd1_source_path = _resolve(config.rd1_duals_path)
    rd1_backfill = build_rd1_fail_closed_backfill(
        rd1_rows,
        conditional_coordinate_rows=conditional_rows,
        source=_artifact(rd1_source_path, "sealed 162-cell RD1 typed dual source"),
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    bucket_path = OUTPUT / "materialized_occupied_buckets.json"
    prices_path = OUTPUT / "conditional_coordinate_prices.json"
    rd1_path = OUTPUT / "rd1_dual_backfill_pf3.json"
    _publish(bucket_path, bucket_report)
    _publish(
        prices_path,
        {
            "schema": "ddm_pf3_conditional_coordinate_prices.v1",
            "physical_edge": "V19C_BASE_TO_ONE_EXACT_RG3_COORDINATE",
            "measured_candidate_count": len(candidate_rows),
            "finite_conditional_coordinate_price_count": sum(
                row.get("lambda_bytes_per_D_dimension") is not None
                for row in conditional_rows
            ),
            "rows": conditional_rows,
            "actionable_rd1_price_count": 0,
            "blocker": rd1_backfill["blocker"],
            "score_claim": False,
            "verdict_scope": "INSTANCE/PRECONDITION",
        },
    )
    _publish(rd1_path, rd1_backfill)
    completed_checkpoint = {
        "schema": "ddm_pf3_complete_checkpoint.v1",
        "typed_config_sha256": config_sha256,
        "measured_candidate_count": len(candidate_rows),
        "bucket_report": _artifact(bucket_path, "PF3 37-bucket five-edge report"),
        "conditional_prices": _artifact(
            prices_path,
            "physical-coordinate conditional secants",
        ),
        "rd1_backfill": _artifact(
            rd1_path,
            "fail-closed 162-cell RD1 backfill",
        ),
        "status": "COMPLETE",
    }
    _publish(
        bulk / "stage_checkpoints/03_materialization_complete_v3.json",
        completed_checkpoint,
    )
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "run_id": RUN_ID,
        "lane_id": LANE_ID,
        "delegation_checkpoint_key": config.delegation_checkpoint_key,
        "typed_config": config.model_dump(mode="json"),
        "typed_config_sha256": config_sha256,
        "authority": {
            "research_only": True,
            "execution_allowed": False,
            "score_claim": False,
            "promotion_eligible": False,
            "main_landing_review_required": True,
            "local_cpu_baseline": "0.1910828242 [contest-CPU]",
            "competitive_frontier": (
                "official leaderboard displayed 0.172; canonical effective-frontier "
                "repair is owned by main and is not duplicated here"
            ),
            "competitive_frontier_moved_by_this_arm": False,
            "local_cost_usd": 0,
            "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
        },
        "storage_preflight": storage,
        "source_custody": {
            "rg3_receipt": _artifact(
                _resolve(config.rg3_receipt_path),
                "sealed RG1-RG3 receiver-support measurement",
            ),
            "rg3_assignment": _artifact(
                _resolve(config.rg3_assignment_path),
                "existing scorer-recursive coordinate vocabulary",
            ),
            "direct_metric": _artifact(
                _resolve(config.direct_metric_path),
                "MS4D 37 occupied buckets plus 25 direct blocks",
            ),
            "rd1_duals": _artifact(
                rd1_source_path,
                "sealed 162-cell typed dual cube",
            ),
            "base_archive": _artifact(
                base_archive_path,
                "V19C exact same-object base receiver",
            ),
            "base_receipt": _artifact(
                _resolve(config.base_receipt_path),
                "V19C n600 endpoint custody",
            ),
            "scorer_config": _artifact(
                _resolve(config.scorer_config_path),
                "frozen local CPU scorer geometry",
            ),
            "target_cache": target_custody,
            "rg4_custody_note": (
                "RG4 commit cb34bbb0f1 remains SHA-pinned through the external ms2rp "
                "receipt; PF3 neither modifies nor republishes RG4 evidence"
            ),
        },
        "inventory": {
            "sealed_signed_probe_count": config.expected_rg3_probe_count,
            "sealed_event_hit_probe_count": config.expected_event_hit_probe_count,
            "materialized_coordinate_count": len(candidate_rows),
            "coordinate_selection_policy": (
                "top-five controls union weighted top-mass greedy set cover union "
                "coordinate-owner exact bucket hits"
            ),
            "source_status_counts": dict(
                Counter(row["status"] for row in probe_rows)
            ),
            "top_mass_first": True,
            "candidate_checkpoint_custody": {
                "count": len(candidate_checkpoint_artifacts),
                "artifacts": candidate_checkpoint_artifacts,
                "digest_chain_sha256": hashlib.sha256(
                    "".join(
                        row["sha256"] for row in candidate_checkpoint_artifacts
                    ).encode()
                ).hexdigest(),
            },
            "measurement_extrema": {
                "delta_global_errors_min": min(candidate_delta_errors),
                "delta_global_errors_max": max(candidate_delta_errors),
                "delta_D_joint_min": min(candidate_joint_deltas),
                "delta_D_joint_max": max(candidate_joint_deltas),
                "delta_counted_bytes_min": min(candidate_rate_deltas),
                "delta_counted_bytes_max": max(candidate_rate_deltas),
                "all_measured_coordinate_edges_worsen_joint_D": all(
                    value > 0.0 for value in candidate_joint_deltas
                ),
            },
        },
        "base_coder_race": {
            **base_race,
            "frame_artifacts": base_frame_artifacts,
            "cross_process_zstd_finding": (
                "MEASURED: a fresh-process trained-dictionary frame differed from "
                "the preserved first-process exact frame; runner now resumes and "
                "parse-back verifies preserved frames instead of retraining"
            ),
        },
        "outputs": {
            "materialized_buckets": _artifact(
                bucket_path,
                "37 occupied buckets with exact five-edge coverage/blockers",
            ),
            "conditional_coordinate_prices": _artifact(
                prices_path,
                "finite physical coordinate secants, non-RD1 until edge identity",
            ),
            "rd1_backfill": _artifact(
                rd1_path,
                "162-cell null-preserving RD1 backfill",
            ),
                "ssd_complete_checkpoint": _artifact(
                bulk / "stage_checkpoints/03_materialization_complete_v3.json",
                "resumable PF3 completion checkpoint",
            ),
        },
        "counts": {
            "occupied_bucket_count": bucket_report["occupied_bucket_count"],
            "fully_materialized_occupied_bucket_count": bucket_report[
                "fully_materialized_occupied_bucket_count"
            ],
            "still_unmaterialized_occupied_bucket_count": bucket_report[
                "still_unmaterialized_occupied_bucket_count"
            ],
            "conditional_coordinate_price_count": rd1_backfill[
                "conditional_coordinate_price_count"
            ],
            "finite_actionable_rd1_price_count": rd1_backfill[
                "lambda_measured_cell_count"
            ],
            "still_null_rd1_price_count": rd1_backfill[
                "still_null_lambda_cell_count"
            ],
            "direct_rg4_exclusion_count_retained": len(direct["direct_blocks"]),
        },
        "stretch": {
            "ran": False,
            "reason": (
                "0 actionable RD1 prices; the 25 RG4 exclusions cannot be allocated "
                "to zero and no box member can be priced honestly"
            ),
        },
        "triality": {
            "dsl": str(CONFIG.relative_to(REPO)),
            "dag": f".omx/research/{RUN_ID}/DAG_FEED.json",
            "equations": [
                "cgauge_master_action_v1",
                "ddm_restricted_realized_lambda_continuation_v1",
            ],
            "equation_registration": (
                "NO_NEW_PRICE_LAW; PF3 reuses the registered nonlinear joint action "
                "and RD1 secant definition. The missing object is a typed counted "
                "application operator, not an unregistered equation."
            ),
        },
        "directives_consumed": [
            {
                "utc": "2026-07-19T19:42:07Z",
                "application": "top-mass-first ordering and exact rate break-even test",
            },
            {
                "utc": "2026-07-19T19:48:01Z",
                "application": "joint scorer action guards each conditional dimension gain",
            },
            {
                "utc": "2026-07-24T14:45:16Z",
                "application": "only existing scorer-recursive RG3 coordinates rebuilt",
            },
            {
                "utc": "2026-07-24T20:01:23Z",
                "application": (
                    "generic receiver/coder machinery is free; rate delta prices only "
                    "the irreducible RG3 coordinate packet"
                ),
            },
            {
                "utc": "2026-07-24T23:09:25Z",
                "application": (
                    "typed counted application operator treated as the exact remaining "
                    "RD1 edge-identity blocker"
                ),
            },
            {
                "utc": "2026-07-25T19:52:29Z",
                "application": (
                    "0.1910828242 retained only as the local contest-CPU baseline; "
                    "official displayed 0.172 is the competitive frontier and main owns "
                    "the canonical pointer repair"
                ),
            },
            {
                "utc": "2026-07-25T20:01:10Z",
                "application": (
                    "preserved all completed scorer checkpoints; spatial arithmetic "
                    "runtime was not repeated per candidate; exact fixed-owner E4 "
                    "parse-back retained"
                ),
            },
            {
                "utc": "2026-07-25T20:08:44Z",
                "application": (
                    "bounded deterministic 16-coordinate selection completed and "
                    "receipt sealed without expanding into receiver rendering"
                ),
            },
        ],
        "verdict": (
            f"{bucket_report['fully_materialized_occupied_bucket_count']}_OF_37_"
            "OCCUPIED_BUCKETS_FIVE_EDGE_MATERIALIZED; "
            f"{rd1_backfill['conditional_coordinate_price_count']}_FINITE_"
            "CONDITIONAL_COORDINATE_SECANTS; 0_OF_162_ACTIONABLE_RD1_PRICES; "
            "PRECONDITION_EDGE_APPLICATION_IDENTITY_BLOCKED"
        ),
        "verdict_scope": (
            "PRECONDITION/INSTANCE: existing RG3 coordinates at the V19C endpoint; "
            "not a family or paradigm negative"
        ),
        "main_landing_review_required": True,
    }
    receipt_path = OUTPUT / "receipt.json"
    _publish(receipt_path, receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG)
    args = parser.parse_args()
    receipt = materialize(args.config)
    print(json.dumps({"verdict": receipt["verdict"], "counts": receipt["counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
