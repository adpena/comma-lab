#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Measure the missing PF2 actuator/direction foreign key through the receiver.

This research-only sweep is deliberately metric-first and fail-closed:

* every probe is one named signed quantum around one SHA-bound V19C endpoint;
* camera support is realized before any scorer forward;
* SegNet runs only for pairs whose exact composite-R support intersects an
  occupied PF2 raw-event coordinate;
* exact changed PF2 event IDs are retained in SSD checkpoints;
* receiver-inexpressible Lane/G2CS1 probes are explicit infeasible rows, never
  silently omitted or converted into spatial-prior joins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import resource
import shutil
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO / "src", REPO):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap  # noqa: E402
from tac.optimization import direct_description_coupled_margin as coupled  # noqa: E402
from tac.optimization import direct_description_preuint8_channel as preuint8  # noqa: E402
from tac.optimization.ddm_g3_score_atlas import reconstruct_v12_state  # noqa: E402
from tac.optimization.ddm_pf2_bucket_assignment import (  # noqa: E402
    ASSIGNMENT_RECEIPT_SCHEMA,
    PROBE_RESULT_SCHEMA,
    build_measured_assignment_table,
    canonical_bytes,
    canonical_sha256,
    intersect_argmax_delta_with_pf2_events,
    reconstruct_bucket_event_ids,
    validate_assignment_table,
)
from tac.optimization.ddm_rg1_receiver_grammar import (  # noqa: E402
    LaneProgramCoordinateV1,
    SkeletonAmplitudeCoordinateV1,
    compile_rg1_receiver_grammar,
    compile_rg2_receiver_grammar,
    project_polygon_center,
    receive_rg1_receiver_grammar,
)
from tac.optimization.ddm_runtime_sensitivity import (  # noqa: E402
    composite_r_support_mask,
    forward_seg_argmax,
)
from tac.optimization.direct_description_carrier_compose import (  # noqa: E402
    REALIZATION_STATIC_RULE_MEMBER,
    WORLDSHEET_G1_MEMBER,
    compile_carrier_compose_archive,
    parse_carrier_compose_archive,
    receive_carrier_compose_archive,
)
from tac.optimization.direct_description_g1_worldsheet import (  # noqa: E402
    encode_lifted_g1_movable_worldsheet,
    lift_g1_movable_worldsheet,
)
from tac.optimization.direct_description_minimizer import DirectDescriptionError  # noqa: E402
from tac.optimization.predictor_upgrade_xi_chart import LaneCoefficientDelta  # noqa: E402
from tac.scorer import load_default_scorers  # noqa: E402
from tools.assign_ddm_ms5_pf2_buckets import (  # noqa: E402
    _xi_event_ids,
    read_bound,
    resolve_repo_path,
    verify_large_bound,
)
from tools.measure_ddm_v19c_correction_saturation import (  # noqa: E402
    _compile_coupled_margin_fast,
    _compile_preuint8_fast,
)

RUN_ID: Final = "ddm_ms6_receiver_support_measurement_20260724T052034Z"
RG1_RUN_ID: Final = "ddm_rg1_receiver_grammar_extension_20260724T080402Z"
RG2_RUN_ID: Final = "ddm_rg2_skeleton_amplitude_productions_20260724T094305Z"
LANE_ID: Final = "lane_ddm_ms6_receiver_support_measurement_20260724"
RECEIPT_SCHEMA: Final = "ddm_ms6_receiver_support_measurement_receipt.v1"
CHECKPOINT_SCHEMA: Final = "ddm_ms6_receiver_support_probe_checkpoint.v2"
EVENT_INDEX_SCHEMA: Final = "ddm_ms6_pf2_event_index.v1"
BASELINE_CELLS_SCHEMA: Final = "ddm_ms6_baseline_seg_cells.v1"
EXPECTED_TABLE_SHA256: Final = "20fa2b2ce2bd96b91c64d4e1342109dd7dab399d4769cd372dbf67fbcdf97d8d"
EXPECTED_MS5_RECEIPT_SHA256: Final = "3d0b9fcc738a1092bad495b0dbce2b022451e1442814a7cc274da41e43d455d6"
EXPECTED_PF2_SHA256: Final = "85084f7bd3a03dbd1b9f04fe6a9b84df4948a6caf64620beef42da8924345f73"
EXPECTED_BASE_SHA256: Final = "dc767b59c9e8671b6870e0f9f17a24cfe900dd0f2ae2a251825e41566b52e4c9"
EXPECTED_SEGNET_SHA256: Final = "68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6"
EXPECTED_MODULES_SHA256: Final = "065961ba97023e393e27818760b0dc8efaa8dd53c5d4cc70a2db8ee1b3cf49aa"
MIN_FREE_BYTES: Final = 2 * 1024**3
HEIGHT: Final = 384
WIDTH: Final = 512
PAIR_COUNT: Final = 600

DEFAULT_TABLE = REPO / (".omx/research/ddm_ms5_pf2_bucket_assignment_20260724T044736Z/pf2_bucket_assignment_table.json")
DEFAULT_MS5_RECEIPT = REPO / (
    ".omx/research/ddm_ms5_pf2_bucket_assignment_20260724T044736Z/ddm_ms5_pf2_bucket_assignment_receipt.json"
)
DEFAULT_PF2 = REPO / (
    ".omx/research/ddm_pf2_dimension_conditioned_two_type_20260724T020205Z/"
    "ddm_pf2_dimension_conditioned_two_type_receipt.json"
)
DEFAULT_BASE = REPO / (
    ".omx/research/ddm_v19c_correction_saturation_20260723T063500Z/ddm_v19c_final_n600.zip.receipt-bytes"
)
DEFAULT_UPSTREAM = Path("/Users/adpena/Projects/pact/upstream")
DEFAULT_BULK = Path("/Volumes/VertigoDataTier/pact") / RUN_ID
DEFAULT_RECEIPTS = REPO / ".omx/research" / RUN_ID
DEFAULT_PRIOR_CHECKPOINTS = DEFAULT_BULK / "probe_checkpoints_v2"
DEFAULT_RG2_ASSIGNMENT = REPO / (
    ".omx/research/ddm_rg2_skeleton_amplitude_productions_20260724T094305Z/ddm_rg2_skeleton_amplitude_assignment.json"
)
DEFAULT_RG2_BULK = Path("/Volumes/VertigoDataTier/pact") / RG2_RUN_ID
DEFAULT_RG2_RECEIPTS = REPO / ".omx/research" / RG2_RUN_ID
DEFAULT_RG2_CACHE = Path("/Volumes/VertigoDataTier/pact") / "ddm_rg1_receiver_grammar_extension_20260724T080402Z"
DEFAULT_RG2_PRIOR_ROOTS = (
    Path("/Volumes/VertigoDataTier/pact") / RUN_ID / "probe_checkpoints_v2",
    DEFAULT_RG2_CACHE / "probe_checkpoints_rg1_v2",
)
EXPECTED_RG1_TABLE_SHA256: Final = "2274c8e654262b90ef35a604280c6c8a4e07a7403480b568d1c3ac8ea8141170"
EXPECTED_RG1_RECEIPT_SHA256: Final = "26e77a9cacda11d65fd2cff48f272f1dcbf9bdb79a760a97bc5da6b950101d2f"
EXPECTED_RG2_ASSIGNMENT_SCHEMA: Final = "ddm_rg2_skeleton_amplitude_assignment.v1"

G2G_RECEIPT_SHA256: Final = "fa49a2ca71cb2960b1e497d425f05c4a496cc7634c45b2e193e3977dfa0667da"
G2CS1_QUANTA: Final = {
    "g2g.g2cs1.pair000.line04.coefficient03": 0.008202752098441124,
    "g2g.g2cs1.pair022.line04.coefficient03": 0.008202752098441124,
    "g2g.g2cs1.pair030.line04.coefficient03": 0.03801910579204559,
    "g2g.g2cs1.pair034.line04.coefficient03": 0.009356264024972916,
    "g2g.g2cs1.pair037.line04.coefficient03": 0.009072740562260151,
    "g2g.g2cs1.pair046.line03.coefficient03": 0.007532086689025164,
}


class MS6MeasurementError(RuntimeError):
    """A source, receiver probe, or checkpoint failed strict custody."""


@dataclass(frozen=True, slots=True)
class ProbeContext:
    base_archive: bytes
    base_receiver: Any
    base_carrier: bytes
    carrier_members: Mapping[str, bytes]
    carrier_receiver: Any
    g1: Any
    coupled_program: Any
    preuint8_program: Any


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(path: Path, expected: str, label: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise MS6MeasurementError(f"{label} is absent or not a regular file: {path}")
    observed = _sha256_file(path)
    if observed != expected:
        raise MS6MeasurementError(f"{label} SHA-256 differs: expected {expected}, observed {observed}")
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": observed}


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise MS6MeasurementError(f"expected JSON object: {path}")
    return value


def _portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO.resolve()))
    except ValueError:
        return str(path.resolve())


def _storage_preflight(path: Path) -> dict[str, Any]:
    parent = path.expanduser().resolve().parent
    if not str(parent).startswith(("/Volumes/VertigoDataTier/pact", "/Volumes/APDataStore/pact")):
        raise MS6MeasurementError("bulk output must use the governed SSD waterfall")
    free = shutil.disk_usage(parent).free
    if free < MIN_FREE_BYTES:
        raise MS6MeasurementError(f"storage preflight refused: free={free}, required={MIN_FREE_BYTES}")
    return {
        "status": "PASS",
        "tier": str(parent),
        "observed_free_bytes": free,
        "required_free_bytes": MIN_FREE_BYTES,
        "auto_cleanup": (
            "candidate archives are deterministic rebuildable scratch and are "
            "not persisted; event/checkpoint evidence is preserved"
        ),
    }


def _torch_setup() -> None:
    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.manual_seed(1234)
    np.random.seed(1234)
    torch.use_deterministic_algorithms(True)


def _load_pf2_event_index(
    *,
    pf2_path: Path,
    bulk: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build or reopen the exact occupied PF2 raw-event index."""

    index_path = bulk / "pf2_occupied_event_index.npz"
    receipt_path = bulk / "pf2_occupied_event_index_receipt.json"
    pf2_payload = read_bound(pf2_path, EXPECTED_PF2_SHA256, "PF2 receipt")
    if index_path.exists() and receipt_path.exists():
        receipt = _read_json(receipt_path)
        if (
            receipt.get("schema") != EVENT_INDEX_SCHEMA
            or receipt.get("pf2_receipt_sha256") != EXPECTED_PF2_SHA256
            or receipt.get("index_sha256") != _sha256_file(index_path)
        ):
            raise MS6MeasurementError("PF2 event-index resume custody differs")
        with np.load(index_path, allow_pickle=False) as stored:
            result = {
                str(bucket_id): np.asarray(stored[str(array_name)], dtype=np.uint32)
                for bucket_id, array_name in receipt["bucket_arrays"].items()
            }
        return result, receipt

    pf2 = json.loads(pf2_payload)
    custody = pf2.get("input_custody")
    if not isinstance(custody, Mapping):
        raise MS6MeasurementError("PF2 receipt lacks input custody")
    g4_binding = custody["g4_receipt"]
    g4_path = resolve_repo_path(str(g4_binding["path"]))
    g4_payload = read_bound(g4_path, str(g4_binding["sha256"]), "G4 receipt")
    g4 = json.loads(g4_payload)
    recurrence = next(row for row in g4["outputs"] if row["path"].endswith("01_recurrence_arrays.npz"))
    tracks = next(row for row in g4["outputs"] if row["path"].endswith("xi_proxy_tracks.jsonl"))
    recurrence_path = resolve_repo_path(str(recurrence["path"]))
    tracks_path = resolve_repo_path(str(tracks["path"]))
    recurrence_payload = read_bound(recurrence_path, str(recurrence["sha256"]), "G4 recurrence arrays")
    tracks_payload = read_bound(tracks_path, str(tracks["sha256"]), "G4 xi tracks")
    with np.load(recurrence_path, allow_pickle=False) as stored:
        transition_counts = np.asarray(stored["transition_counts"], dtype=np.uint16)

    v12_binding = custody["v12_receipt"]
    v12_path = resolve_repo_path(str(v12_binding["path"]))
    v12_payload = read_bound(v12_path, str(v12_binding["sha256"]), "V12 receipt")
    v12 = json.loads(v12_payload)
    state = reconstruct_v12_state(REPO, v12, n_pairs=PAIR_COUNT)
    predicted = np.asarray(state.final_cells, dtype=np.uint8)
    target_binding = v12["target_custody"]
    target_path = Path(str(target_binding["cache_path"]))
    target_custody = verify_large_bound(
        target_path,
        str(target_binding["cache_sha256"]),
        int(target_binding["cache_bytes"]),
        "V12 target cache",
    )
    target = np.asarray(open_stored_npy_memmap(target_path, "lstars"), dtype=np.uint8)
    event_index = reconstruct_bucket_event_ids(
        pf2_receipt=pf2,
        predicted=predicted,
        target=target,
        transition_counts=transition_counts,
        xi_event_ids=_xi_event_ids(tracks_payload),
    )
    arrays = {
        f"bucket_{index:04d}": values for index, (_bucket_id, values) in enumerate(event_index.items()) if values.size
    }
    bucket_arrays = {
        bucket_id: f"bucket_{index:04d}" for index, (bucket_id, values) in enumerate(event_index.items()) if values.size
    }
    index_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = index_path.with_name(f".{index_path.name}.tmp.{os.getpid()}")
    with temporary.open("xb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, index_path)
    receipt = {
        "schema": EVENT_INDEX_SCHEMA,
        "pf2_receipt_sha256": EXPECTED_PF2_SHA256,
        "index_path": str(index_path),
        "index_bytes": index_path.stat().st_size,
        "index_sha256": _sha256_file(index_path),
        "bucket_arrays": bucket_arrays,
        "occupied_bucket_count": len(bucket_arrays),
        "raw_event_count": sum(int(value.size) for value in event_index.values()),
        "input_hash_lineage": {
            "pf2": {"path": str(pf2_path), "bytes": len(pf2_payload), "sha256": EXPECTED_PF2_SHA256},
            "g4": {"path": str(g4_path), "bytes": len(g4_payload), "sha256": g4_binding["sha256"]},
            "g4_recurrence": {
                "path": str(recurrence_path),
                "bytes": len(recurrence_payload),
                "sha256": recurrence["sha256"],
            },
            "g4_xi_tracks": {
                "path": str(tracks_path),
                "bytes": len(tracks_payload),
                "sha256": tracks["sha256"],
            },
            "v12": {"path": str(v12_path), "bytes": len(v12_payload), "sha256": v12_binding["sha256"]},
            "target_cache": target_custody,
        },
        "score_claim": False,
        "research_only": True,
    }
    _publish(receipt_path, canonical_bytes(receipt))
    return event_index, receipt


def _probe_context(base_archive: bytes) -> ProbeContext:
    pre_members, _ = preuint8.parse_preuint8_q8_archive(base_archive)
    coupled_archive = pre_members[preuint8.BASE_MEMBER]
    coupled_members, _ = coupled.parse_coupled_margin_archive(coupled_archive)
    carrier = coupled_members[coupled.BASE_MEMBER]
    carrier_members, _ = parse_carrier_compose_archive(carrier)
    carrier_receiver = receive_carrier_compose_archive(
        carrier,
        verify_member_effects=False,
    )
    if WORLDSHEET_G1_MEMBER not in carrier_members or carrier_receiver.scorer_solved_templates is None:
        raise MS6MeasurementError("V19C endpoint lacks its J2 worldsheet/template receiver state")
    return ProbeContext(
        base_archive=base_archive,
        base_receiver=_fast_receiver(base_archive),
        base_carrier=carrier,
        carrier_members=carrier_members,
        carrier_receiver=carrier_receiver,
        g1=lift_g1_movable_worldsheet(carrier_members[WORLDSHEET_G1_MEMBER]),
        coupled_program=coupled.decode_coupled_margin_program(coupled_members[coupled.PROGRAM_MEMBER]),
        preuint8_program=preuint8.decode_preuint8_q8_program(pre_members[preuint8.PROGRAM_MEMBER]),
    )


def _fast_receiver(archive: bytes) -> preuint8.PreUint8Q8ReceiverV1:
    """Reopen the exact outer grammars while skipping repeated V15 no-op proofs.

    V19C already published strict member-effect custody for the SHA-bound base.
    Every probe still parses and re-encodes all three ZIP grammars and executes
    the same receiver methods; only the quadratic repetition of isolated
    member-effect proofs is omitted from the inner sweep.
    """

    pre_members, _ = preuint8.parse_preuint8_q8_archive(archive)
    coupled_archive = pre_members[preuint8.BASE_MEMBER]
    coupled_members, _ = coupled.parse_coupled_margin_archive(coupled_archive)
    carrier = coupled_members[coupled.BASE_MEMBER]
    carrier_receiver = receive_rg1_receiver_grammar(
        carrier,
        verify_member_effects=False,
    )
    coupled_program = coupled.decode_coupled_margin_program(coupled_members[coupled.PROGRAM_MEMBER])
    coupled_receiver = coupled.CoupledMarginReceiverV1(
        coupled_archive,
        carrier_receiver,
        coupled_program,
        {},
    )
    pre_program = preuint8.decode_preuint8_q8_program(pre_members[preuint8.PROGRAM_MEMBER])
    return preuint8.PreUint8Q8ReceiverV1(
        archive,
        coupled_receiver,
        pre_program,
        {},
    )


_ISLAND = re.compile(r"^j2\.island\.track(\d+)\.center_([xy])$")
_TEMPLATE = re.compile(r"^j2\.template\.row(\d+)\.rgb_([rgb])$")
_LANE = re.compile(
    r"^j2\.lane\.line(\d+)\."
    r"(dash_phase_origin_q8|dash_phase_xi_gain_q8|width_bias_q8|width_slope_q12)$"
)
_G2CS1 = re.compile(r"^g2g\.g2cs1\.pair(\d+)\.line(\d+)\.coefficient(\d+)$")
_RG2_SKELETON = re.compile(
    r"^rg2\.skeleton\.pair(\d{3})\.class([0-4])_([0-4])\."
    r"(boundary|cell)\.(static_in_image|transient)\.band(\d{2})$"
)
_BOUNDED_SUFFIX: Final = ".bounded_clamp"


def _compile_probe(
    context: ProbeContext,
    *,
    actuator_id: str,
    direction_id: str,
) -> tuple[bytes | None, tuple[int, ...], str | None]:
    sign = -1 if direction_id == "NEGATIVE_ONE_QUANTUM" else 1
    g1 = context.g1
    templates = context.carrier_receiver.scorer_solved_templates
    if templates is None:
        raise MS6MeasurementError("V19C template bank disappeared")
    pair_ids: tuple[int, ...]
    bounded = actuator_id.endswith(_BOUNDED_SUFFIX)
    base_actuator_id = actuator_id.removesuffix(_BOUNDED_SUFFIX)
    island = _ISLAND.fullmatch(base_actuator_id)
    template = _TEMPLATE.fullmatch(actuator_id)
    lane = _LANE.fullmatch(actuator_id)
    g2cs1 = _G2CS1.fullmatch(actuator_id)
    rg2_skeleton = _RG2_SKELETON.fullmatch(actuator_id)
    if island:
        object_id = int(island.group(1))
        axis = island.group(2)
        track = next((value for value in g1.tracks if value.object_id == object_id), None)
        if track is None:
            return None, (), "J2_STABLE_ISLAND_ID_ABSENT_FROM_V19C_BASE"
        selected = set(track.knot_indices)
        knots = tuple(
            replace(
                knot,
                center_x=knot.center_x + (sign if axis == "x" else 0),
                center_y=knot.center_y + (sign if axis == "y" else 0),
            )
            if index in selected
            else knot
            for index, knot in enumerate(g1.knots)
        )
        if bounded:
            templates_by_ref = {value.template_ref: value for value in g1.templates}
            knots = tuple(
                replace(
                    knot,
                    center_x=project_polygon_center(
                        knot.center_x,
                        tuple(point[0] for point in templates_by_ref[knot.template_ref].relative_vertices_xy),
                        WIDTH,
                    ),
                    center_y=project_polygon_center(
                        knot.center_y,
                        tuple(point[1] for point in templates_by_ref[knot.template_ref].relative_vertices_xy),
                        HEIGHT,
                    ),
                )
                for knot in knots
            )
        elif any(not 0 <= knot.center_x < WIDTH or not 0 <= knot.center_y < HEIGHT for knot in knots):
            return None, tuple(range(track.birth_pair, track.death_pair_exclusive)), "ONE_QUANTUM_ESCAPES_SCORER_GRID"
        g1 = replace(g1, knots=knots)
        pair_ids = tuple(range(track.birth_pair, track.death_pair_exclusive))
    elif template:
        row_index = int(template.group(1))
        channel = {"r": 0, "g": 1, "b": 2}[template.group(2)]
        if not 0 <= row_index < len(templates.templates):
            return None, (), "J2_STABLE_TEMPLATE_ID_ABSENT_FROM_V19C_BASE"
        rows = list(templates.templates)
        source = rows[row_index]
        rgb = np.frombuffer(source.rgb_u8, dtype=np.uint8).reshape(-1, 3).astype(np.int16)
        rgb[:, channel] = np.clip(rgb[:, channel] + sign, 0, 255)
        rows[row_index] = replace(source, rgb_u8=np.asarray(rgb, dtype=np.uint8).tobytes())
        templates = replace(templates, templates=tuple(rows))
        pair_ids = tuple(range(PAIR_COUNT))
    elif lane:
        coordinate = LaneProgramCoordinateV1(
            line_index=int(lane.group(1)),
            field=lane.group(2),
            signed_quanta=sign,
        )
        carrier = compile_rg1_receiver_grammar(
            context.base_carrier,
            lane_coordinates=(coordinate,),
        )
        coupled_archive = _compile_coupled_margin_fast(carrier, context.coupled_program)
        return _compile_preuint8_fast(coupled_archive, context.preuint8_program), tuple(range(PAIR_COUNT)), None
    elif g2cs1:
        magnitude = G2CS1_QUANTA.get(actuator_id)
        if magnitude is None:
            return None, (), "G2CS1_ACTUATOR_LACKS_SHA_BOUND_QUANTUM"
        correction = LaneCoefficientDelta(
            pair_index=int(g2cs1.group(1)),
            line_index=int(g2cs1.group(2)),
            coefficient_index=int(g2cs1.group(3)),
            coefficient_delta=sign * magnitude,
        )
        carrier = compile_rg1_receiver_grammar(
            context.base_carrier,
            corrections=(correction,),
        )
        coupled_archive = _compile_coupled_margin_fast(carrier, context.coupled_program)
        return _compile_preuint8_fast(coupled_archive, context.preuint8_program), (correction.pair_index,), None
    elif rg2_skeleton:
        pair_index = int(rg2_skeleton.group(1))
        stratum = rg2_skeleton.group(4)
        coordinate = SkeletonAmplitudeCoordinateV1(
            pair_index=pair_index,
            class_a=int(rg2_skeleton.group(2)),
            class_b=int(rg2_skeleton.group(3)),
            family=("EVENT_LOCAL_BOUNDARY" if stratum == "boundary" else "PER_STRATUM_ROW_BAND"),
            temporal_class=rg2_skeleton.group(5).upper(),
            row_band=int(rg2_skeleton.group(6)),
            signed_quanta=sign,
        )
        carrier = compile_rg2_receiver_grammar(
            context.base_carrier,
            skeleton_amplitudes=(coordinate,),
        )
        coupled_archive = _compile_coupled_margin_fast(
            carrier,
            context.coupled_program,
        )
        return (
            _compile_preuint8_fast(coupled_archive, context.preuint8_program),
            (pair_index,),
            None,
        )
    else:
        return None, (), "ACTUATOR_ID_HAS_NO_V19C_RECEIVER_COMPILER"

    carrier, _ = compile_carrier_compose_archive(
        context.carrier_members["predictor.zip"],
        worldsheet_g1_payload=encode_lifted_g1_movable_worldsheet(g1),
        realization_profile=context.carrier_receiver.realization_profile,
        realization_static_rule_payload=context.carrier_members.get(REALIZATION_STATIC_RULE_MEMBER, b""),
        realization_static_rule_id=context.carrier_receiver.realization_static_rule_id,
        scorer_solved_templates=templates,
    )
    coupled_archive = _compile_coupled_margin_fast(carrier, context.coupled_program)
    archive = _compile_preuint8_fast(coupled_archive, context.preuint8_program)
    return archive, pair_ids, None


def _occupied_mask(event_index: Mapping[str, np.ndarray]) -> np.ndarray:
    mask = np.zeros((PAIR_COUNT, HEIGHT, WIDTH), dtype=np.bool_)
    flat = mask.reshape(-1)
    for ids in event_index.values():
        flat[np.asarray(ids, dtype=np.intp)] = True
    return mask


def _baseline_cells(
    *,
    context: ProbeContext,
    segnet: Any,
    bulk: Path,
    scorer_custody: Mapping[str, Any],
) -> np.memmap:
    path = bulk / "baseline_v19c_seg_cells_u8.npy"
    receipt_path = bulk / "baseline_v19c_seg_cells_receipt.json"
    if path.exists() and receipt_path.exists():
        receipt = _read_json(receipt_path)
        if (
            receipt.get("schema") != BASELINE_CELLS_SCHEMA
            or receipt.get("base_archive_sha256") != EXPECTED_BASE_SHA256
            or receipt.get("scorer_custody") != scorer_custody
            or receipt.get("cells_sha256") != _sha256_file(path)
        ):
            raise MS6MeasurementError("baseline Seg cells resume custody differs")
        return np.load(path, mmap_mode="r", allow_pickle=False)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    cells = np.lib.format.open_memmap(
        temporary,
        mode="w+",
        dtype=np.uint8,
        shape=(PAIR_COUNT, HEIGHT, WIDTH),
    )
    for start in range(0, PAIR_COUNT, 32):
        pair_ids = tuple(range(start, min(start + 32, PAIR_COUNT)))
        camera = context.base_receiver.render_camera_pairs(pair_ids)
        cells[start : start + len(pair_ids)] = forward_seg_argmax(segnet=segnet, camera=camera)
        cells.flush()
    del cells
    os.replace(temporary, path)
    receipt = {
        "schema": BASELINE_CELLS_SCHEMA,
        "base_archive_sha256": EXPECTED_BASE_SHA256,
        "scorer_custody": dict(scorer_custody),
        "cells_path": str(path),
        "cells_bytes": path.stat().st_size,
        "cells_sha256": _sha256_file(path),
        "scorer_batch_size": 32,
        "threads": 4,
        "seed": 1234,
        "deterministic_algorithms": True,
        "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
        "score_claim": False,
    }
    _publish(receipt_path, canonical_bytes(receipt))
    return np.load(path, mmap_mode="r", allow_pickle=False)


def _merge_hits(
    destination: dict[str, list[np.ndarray]],
    hits: Mapping[str, Mapping[str, Any]],
) -> None:
    for bucket_id, row in hits.items():
        destination.setdefault(bucket_id, []).append(np.asarray(row["event_ids"], dtype=np.uint32))


def _measure_probe(
    *,
    context: ProbeContext,
    actuator_id: str,
    direction_id: str,
    event_index: Mapping[str, np.ndarray],
    occupied: np.ndarray,
    baseline_cells: np.ndarray,
    segnet: Any,
    scorer_custody: Mapping[str, Any],
    checkpoint_root: Path,
    checkpoint_run_id: str = RUN_ID,
) -> Path:
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    safe_id = actuator_id.replace(".", "_")
    checkpoint = checkpoint_root / f"{safe_id}__{direction_id}.json"
    event_path = checkpoint_root / f"{safe_id}__{direction_id}.events.npz"
    if checkpoint.exists():
        value = _read_json(checkpoint)
        if (
            value.get("schema") != CHECKPOINT_SCHEMA
            or value.get("run_id") != checkpoint_run_id
            or value.get("receiver_actuator_id") != actuator_id
            or value.get("direction_id") != direction_id
            or value.get("base_archive_sha256") != EXPECTED_BASE_SHA256
            or value.get("scorer_custody") != scorer_custody
            or value.get("threads") != 4
            or value.get("seed") != 1234
            or value.get("deterministic_algorithms") is not True
            or (
                value.get("event_artifact") is not None
                and _sha256_file(Path(value["event_artifact"]["path"])) != value["event_artifact"]["sha256"]
            )
        ):
            raise MS6MeasurementError(f"probe resume checkpoint differs: {checkpoint}")
        return checkpoint

    started = time.monotonic()
    try:
        archive, pair_ids, infeasible = _compile_probe(
            context,
            actuator_id=actuator_id,
            direction_id=direction_id,
        )
    except (DirectDescriptionError, ValueError, OverflowError) as exc:
        archive, pair_ids, infeasible = None, (), f"{type(exc).__name__}: {exc}"
    if archive is None:
        payload = {
            "schema": CHECKPOINT_SCHEMA,
            "run_id": checkpoint_run_id,
            "receiver_actuator_id": actuator_id,
            "direction_id": direction_id,
            "status": "INFEASIBLE_RECEIVER_QUANTUM",
            "infeasible_reason": infeasible,
            "base_archive_sha256": EXPECTED_BASE_SHA256,
            "scorer_custody": dict(scorer_custody),
            "candidate_archive": None,
            "raster_support": {"pair_count": 0, "camera_value_count": 0, "composite_r_cell_count": 0},
            "scorer": {"forward_pair_count": 0, "batch_size": 32},
            "bucket_hits": [],
            "event_artifact": None,
            "elapsed_seconds": time.monotonic() - started,
            "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "threads": 4,
            "seed": 1234,
            "deterministic_algorithms": True,
            "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
            "score_claim": False,
            "research_only": True,
            "verdict_scope": "INSTANCE_V19C_ENDPOINT_ONE_SIGNED_QUANTUM",
        }
        _publish(checkpoint, canonical_bytes(payload))
        return checkpoint

    candidate = _fast_receiver(archive)
    camera_values = 0
    support_cells = 0
    support_pairs: set[int] = set()
    scorer_pairs: set[int] = set()
    merged: dict[str, list[np.ndarray]] = {}
    for start in range(0, len(pair_ids), 32):
        chunk = pair_ids[start : start + 32]
        base_camera = context.base_receiver.render_camera_pairs(chunk)
        candidate_camera = candidate.render_camera_pairs(chunk)
        camera_values += int(np.count_nonzero(base_camera != candidate_camera))
        support = composite_r_support_mask(
            segnet=segnet,
            baseline_camera=base_camera,
            perturbed_camera=candidate_camera,
        )
        support_cells += int(np.count_nonzero(support))
        for local, pair_id in enumerate(chunk):
            if np.any(support[local]):
                support_pairs.add(pair_id)
        shortlist = [local for local, pair_id in enumerate(chunk) if np.any(support[local] & occupied[pair_id])]
        if not shortlist:
            continue
        selected_pairs = [int(chunk[local]) for local in shortlist]
        scorer_pairs.update(selected_pairs)
        candidate_argmax = forward_seg_argmax(
            segnet=segnet,
            camera=candidate_camera[np.asarray(shortlist, dtype=np.intp)],
        )
        baseline_argmax = np.asarray(
            baseline_cells[np.asarray(selected_pairs, dtype=np.intp)],
            dtype=np.uint8,
        )
        hits = intersect_argmax_delta_with_pf2_events(
            pair_ids=selected_pairs,
            baseline_cells=baseline_argmax,
            perturbed_cells=candidate_argmax,
            bucket_event_ids=event_index,
        )
        _merge_hits(merged, hits)

    event_arrays = {
        bucket_id: np.unique(np.concatenate(values)).astype("<u4", copy=False) for bucket_id, values in merged.items()
    }
    event_artifact = None
    if event_arrays:
        temporary = event_path.with_name(f".{event_path.name}.tmp.{os.getpid()}")
        with temporary.open("xb") as handle:
            np.savez_compressed(handle, **event_arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, event_path)
        event_artifact = {
            "path": str(event_path),
            "bytes": event_path.stat().st_size,
            "sha256": _sha256_file(event_path),
        }
    bucket_hits = []
    for bucket_id, event_ids in sorted(event_arrays.items()):
        pairs = sorted({int(value) for value in event_ids.astype(np.uint64) // (HEIGHT * WIDTH)})
        bucket_hits.append(
            {
                "bucket_id": bucket_id,
                "pair_ids": pairs,
                "event_count": int(event_ids.size),
                "event_ids_sha256": hashlib.sha256(event_ids.tobytes(order="C")).hexdigest(),
            }
        )
    status = (
        "MEASURED_ARGMAX_PERTURBATION"
        if bucket_hits
        else "MEASURED_EMPTY_RASTER_SUPPORT"
        if camera_values == 0 or support_cells == 0
        else "MEASURED_EMPTY_NO_OCCUPIED_BUCKET_OVERLAP"
        if not scorer_pairs
        else "MEASURED_EMPTY_ARGMAX_INVARIANT"
    )
    payload = {
        "schema": CHECKPOINT_SCHEMA,
        "run_id": checkpoint_run_id,
        "receiver_actuator_id": actuator_id,
        "direction_id": direction_id,
        "status": status,
        "infeasible_reason": None,
        "base_archive_sha256": EXPECTED_BASE_SHA256,
        "scorer_custody": dict(scorer_custody),
        "candidate_archive": {
            "bytes": len(archive),
            "sha256": hashlib.sha256(archive).hexdigest(),
            "preservation": "REBUILDABLE_FROM_SHA_BOUND_BASE_PLUS_TYPED_ONE_QUANTUM_PROBE",
        },
        "raster_support": {
            "pair_count": len(support_pairs),
            "pair_ids": sorted(support_pairs),
            "camera_value_count": camera_values,
            "composite_r_cell_count": support_cells,
        },
        "scorer": {"forward_pair_count": len(scorer_pairs), "pair_ids": sorted(scorer_pairs), "batch_size": 32},
        "bucket_hits": bucket_hits,
        "event_artifact": event_artifact,
        "elapsed_seconds": time.monotonic() - started,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "threads": 4,
        "seed": 1234,
        "deterministic_algorithms": True,
        "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
        "score_claim": False,
        "research_only": True,
        "verdict_scope": "INSTANCE_V19C_ENDPOINT_ONE_SIGNED_QUANTUM",
    }
    _publish(checkpoint, canonical_bytes(payload))
    return checkpoint


def _probe_result(path: Path) -> dict[str, Any]:
    checkpoint = _read_json(path)
    return {
        "schema": PROBE_RESULT_SCHEMA,
        "receiver_actuator_id": checkpoint["receiver_actuator_id"],
        "direction_id": checkpoint["direction_id"],
        "status": checkpoint["status"],
        "bucket_hits": checkpoint["bucket_hits"],
        "checkpoint_sha256": _sha256_file(path),
    }


def _load_rg2_assignment(path: Path) -> tuple[dict[str, Any], list[str]]:
    value = _read_json(path)
    payload = dict(value)
    claimed = payload.pop("assignment_content_sha256", None)
    rows = value.get("rows")
    if (
        value.get("schema") != EXPECTED_RG2_ASSIGNMENT_SCHEMA
        or claimed != canonical_sha256(payload)
        or not isinstance(rows, list)
        or len(rows) != 64
        or value.get("row_count") != 64
        or not isinstance(value.get("admissible_coordinate_count"), int)
        or value.get("score_claim") is not False
    ):
        raise MS6MeasurementError("RG2 assignment custody differs")
    if any(not isinstance(row, Mapping) for row in rows):
        raise MS6MeasurementError("RG2 assignment rows are malformed")
    actuator_ids = [
        row.get("receiver_actuator_id")
        for row in rows
        if row.get("receiver_actuator_id") is not None
    ]
    if (
        any(
            not isinstance(value, str) or _RG2_SKELETON.fullmatch(value) is None
            for value in actuator_ids
        )
        or len(set(actuator_ids)) != len(actuator_ids)
        or len(actuator_ids) != value["admissible_coordinate_count"]
        or not actuator_ids
    ):
        raise MS6MeasurementError("RG2 assignment actuator vocabulary differs")
    return value, sorted(actuator_ids)


def run(args: argparse.Namespace) -> Path:
    storage = _storage_preflight(args.bulk_output)
    args.bulk_output.mkdir(parents=True, exist_ok=True)
    args.receipt_output.mkdir(parents=True, exist_ok=True)
    _torch_setup()
    rg2_mode = args.rg2_assignment is not None
    table_custody = _verify(
        args.assignment_table,
        EXPECTED_RG1_TABLE_SHA256 if rg2_mode else EXPECTED_TABLE_SHA256,
        "RG1 merged assignment table" if rg2_mode else "MS5 assignment table",
    )
    ms5_custody = _verify(
        args.ms5_receipt,
        EXPECTED_RG1_RECEIPT_SHA256 if rg2_mode else EXPECTED_MS5_RECEIPT_SHA256,
        "RG1 measurement receipt" if rg2_mode else "MS5 assignment receipt",
    )
    base_custody = _verify(args.base_archive, EXPECTED_BASE_SHA256, "V19C base archive")
    segnet_custody = _verify(
        args.upstream / "models/segnet.safetensors",
        EXPECTED_SEGNET_SHA256,
        "SegNet weights",
    )
    modules_custody = _verify(args.upstream / "modules.py", EXPECTED_MODULES_SHA256, "upstream modules")
    base_table = _read_json(args.assignment_table)
    validate_assignment_table(base_table, expected_pf2_sha256=EXPECTED_PF2_SHA256)
    rg2_assignment = None
    rg2_ids: list[str] = []
    if rg2_mode:
        rg2_assignment, rg2_ids = _load_rg2_assignment(args.rg2_assignment)
        base_table = json.loads(json.dumps(base_table))
        vocabulary = base_table["foreign_key_vocabulary"]["receiver_actuator_stable_ids"]
        vocabulary.extend(rg2_ids)
        vocabulary[:] = sorted(set(vocabulary))
        payload = dict(base_table)
        payload.pop("table_content_sha256", None)
        base_table["table_content_sha256"] = canonical_sha256(payload)
        validate_assignment_table(base_table, expected_pf2_sha256=EXPECTED_PF2_SHA256)
    elif args.rg1_retry_from is not None:
        base_table = json.loads(json.dumps(base_table))
        vocabulary = base_table["foreign_key_vocabulary"]["receiver_actuator_stable_ids"]
        prior_infeasible = []
        for path in sorted(args.rg1_retry_from.glob("*.json")):
            prior = _read_json(path)
            if prior.get("status") == "INFEASIBLE_RECEIVER_QUANTUM":
                prior_infeasible.append(prior)
        geometry_ids = sorted(
            {
                str(row["receiver_actuator_id"])
                for row in prior_infeasible
                if str(row.get("infeasible_reason", "")).startswith("DirectDescriptionError: G1 Movable polygon")
            }
        )
        vocabulary.extend(f"{value}{_BOUNDED_SUFFIX}" for value in geometry_ids)
        vocabulary[:] = sorted(set(vocabulary))
        payload = dict(base_table)
        payload.pop("table_content_sha256", None)
        base_table["table_content_sha256"] = canonical_sha256(payload)
        validate_assignment_table(base_table, expected_pf2_sha256=EXPECTED_PF2_SHA256)
    event_index, event_receipt = _load_pf2_event_index(
        pf2_path=args.pf2,
        bulk=args.cache_root or args.bulk_output,
    )
    if len([value for value in event_index.values() if value.size]) != 37:
        raise MS6MeasurementError("PF2 occupied-bucket count differs from MS5 custody")
    if sum(int(value.size) for value in event_index.values()) != 4_011_236:
        raise MS6MeasurementError("PF2 raw-event mass differs from MS5 custody")
    occupied = _occupied_mask(event_index)
    base_archive = args.base_archive.read_bytes()
    context = _probe_context(base_archive)
    _posenet, segnet = load_default_scorers(args.upstream, device="cpu")
    scorer_custody = {"segnet": segnet_custody, "modules": modules_custody}
    baseline_cells = _baseline_cells(
        context=context,
        segnet=segnet,
        bulk=args.cache_root or args.bulk_output,
        scorer_custody=scorer_custody,
    )

    vocabulary = list(base_table["foreign_key_vocabulary"]["receiver_actuator_stable_ids"])
    directions = list(base_table["foreign_key_vocabulary"]["direction_ids"])
    if rg2_mode:
        probes = [(actuator, direction) for actuator in rg2_ids for direction in directions]
        if len(probes) != 2 * len(rg2_ids):
            raise MS6MeasurementError(f"RG2 probe vocabulary differs: {len(probes)}")
    elif args.rg1_retry_from is None:
        probes = [(actuator, direction) for actuator in vocabulary for direction in directions]
        if len(probes) != 748:
            raise MS6MeasurementError(f"stable probe vocabulary differs: {len(probes)}")
    else:
        retry_ids = {
            str(row["receiver_actuator_id"])
            for row in prior_infeasible
            if str(row["receiver_actuator_id"]).startswith(("j2.lane.", "g2g.g2cs1."))
        }
        retry_ids.update(f"{value}{_BOUNDED_SUFFIX}" for value in geometry_ids)
        probes = [(actuator, direction) for actuator in sorted(retry_ids) for direction in directions]
        if len(probes) != 80:
            raise MS6MeasurementError(f"RG1 retry vocabulary differs: {len(probes)}")
    selected = probes[args.probe_start :]
    if args.probe_limit is not None:
        selected = selected[: args.probe_limit]
    checkpoint_root = args.bulk_output / (
        "probe_checkpoints_rg2_l3_v2"
        if rg2_mode
        else "probe_checkpoints_rg1_v2"
        if args.rg1_retry_from is not None
        else "probe_checkpoints_v2"
    )
    checkpoints = []
    for actuator_id, direction_id in selected:
        checkpoints.append(
            _measure_probe(
                context=context,
                actuator_id=actuator_id,
                direction_id=direction_id,
                event_index=event_index,
                occupied=occupied,
                baseline_cells=baseline_cells,
                segnet=segnet,
                scorer_custody=scorer_custody,
                checkpoint_root=checkpoint_root,
                checkpoint_run_id=RG2_RUN_ID if rg2_mode else RUN_ID,
            )
        )
    all_checkpoints = sorted(checkpoint_root.glob("*.json"))
    if rg2_mode:
        prior_by_identity: dict[tuple[str, str], Path] = {}
        for root in args.prior_checkpoint_roots:
            for path in sorted(root.glob("*.json")):
                value = _read_json(path)
                identity = (
                    str(value["receiver_actuator_id"]),
                    str(value["direction_id"]),
                )
                # Roots are ordered oldest to newest. RG1 deliberately remeasured
                # 30 old actuator identities, so the later root is authoritative.
                prior_by_identity[identity] = path
        prior_paths = [prior_by_identity[key] for key in sorted(prior_by_identity)]
        if len(prior_paths) != 768:
            raise MS6MeasurementError(
                "RG2 requires the exact 768-row RG1 merged producer table after "
                f"newer checkpoints supersede repeated identities; observed {len(prior_paths)}"
            )
        all_checkpoints = prior_paths + all_checkpoints
    elif args.rg1_retry_from is not None:
        replaced = {(actuator, direction) for actuator, direction in probes if not actuator.endswith(_BOUNDED_SUFFIX)}
        prior_paths = []
        for path in sorted(args.rg1_retry_from.glob("*.json")):
            value = _read_json(path)
            identity = (str(value["receiver_actuator_id"]), str(value["direction_id"]))
            if identity not in replaced:
                prior_paths.append(path)
        all_checkpoints = prior_paths + all_checkpoints
    probe_results = [_probe_result(path) for path in all_checkpoints]
    measured_table = build_measured_assignment_table(
        base_table=base_table,
        expected_pf2_sha256=EXPECTED_PF2_SHA256,
        probe_results=probe_results,
    )
    table_path = args.receipt_output / "pf2_bucket_assignment_table.json"
    table_payload = canonical_bytes(measured_table)
    _publish(table_path, table_payload)
    statuses: dict[str, int] = {}
    for row in probe_results:
        statuses[row["status"]] = statuses.get(row["status"], 0) + 1
    checkpoint_seconds = [float(_read_json(path)["elapsed_seconds"]) for path in all_checkpoints]
    receipt: dict[str, Any] = {
        "schema": ASSIGNMENT_RECEIPT_SCHEMA,
        "measurement_schema": RECEIPT_SCHEMA,
        "run_id": RG2_RUN_ID if rg2_mode else RG1_RUN_ID if args.rg1_retry_from is not None else RUN_ID,
        "lane_id": (
            "lane_ddm_rg2_skeleton_amplitude_productions_20260724"
            if rg2_mode
            else "lane_ddm_rg1_receiver_grammar_extension_20260724"
            if args.rg1_retry_from is not None
            else LANE_ID
        ),
        "input_hash_lineage": {
            "ms5_assignment_table": table_custody,
            "ms5_assignment_receipt": ms5_custody,
            "pf2_event_index_receipt": event_receipt,
            "v19c_base_archive": base_custody,
            "scorer": scorer_custody,
            **(
                {
                    "rg2_assignment": {
                        "path": str(args.rg2_assignment.resolve()),
                        "sha256": _sha256_file(args.rg2_assignment),
                        "content_sha256": rg2_assignment["assignment_content_sha256"],
                    }
                }
                if rg2_mode and rg2_assignment is not None
                else {}
            ),
        },
        "assignment_table": {
            "path": _portable(table_path),
            "bytes": len(table_payload),
            "file_sha256": hashlib.sha256(table_payload).hexdigest(),
            "content_sha256": measured_table["table_content_sha256"],
            "schema": measured_table["schema"],
        },
        "probe_sweep": {
            "required_probe_count": len(vocabulary) * len(directions),
            "completed_probe_count": len(probe_results),
            "status_counts": statuses,
            "checkpoint_schema": CHECKPOINT_SCHEMA,
            "checkpoint_root": str(checkpoint_root),
            "superseded_checkpoint_root": str(args.bulk_output / "probe_checkpoints"),
            "superseded_checkpoint_disposition": ("PRESERVED_NOT_CONSUMED; v1 lacked per-checkpoint scorer custody"),
            "checkpoint_digest_chain_sha256": hashlib.sha256(
                "".join(_sha256_file(path) for path in all_checkpoints).encode("ascii")
            ).hexdigest(),
            "wallclock_seconds_sum": sum(checkpoint_seconds),
            "wallclock_seconds_max": max(checkpoint_seconds, default=0.0),
            "resumable_per_actuator": True,
            "all_checkpoints_preserved": True,
            "rg1_retry_only": args.rg1_retry_from is not None and not rg2_mode,
            "rg2_new_coordinates_only": rg2_mode,
            "prior_infeasible_evidence_preserved": args.rg1_retry_from is not None or rg2_mode,
        },
        "coverage": measured_table["coverage"],
        "producer_rerun": {
            "eligible": False,
            "ms4_harness_invoked": False,
            "reason": (
                "MS4 remains fail-closed until the measured table covers the "
                "required G3 hard-block buckets with receiver-effective probes."
            ),
        },
        "storage_preflight": storage,
        "verdict": measured_table["verdict"],
        "verdict_scope": (
            "INSTANCE_EXTENDED_GRAMMAR_RG2"
            if rg2_mode
            else "INSTANCE_EXTENDED_GRAMMAR_RG1"
            if args.rg1_retry_from is not None
            else "INSTANCE_V19C_ENDPOINT_ONE_QUANTUM_SWEEP"
        ),
        "evidence_axis": "[macOS-CPU frozen-scorer advisory]",
        "score_claim": False,
        "pointer": "0.1910828242 [contest-CPU]",
        "pointer_moved": False,
        "research_only": True,
        "main_landing_review_required": True,
    }
    if rg2_mode:
        rg2_rows = [_read_json(path) for path in sorted(checkpoint_root.glob("*.json"))]
        candidate_shas = {
            str(row["candidate_archive"]["sha256"]) for row in rg2_rows if row.get("candidate_archive") is not None
        }
        receipt["rg2_extension"] = {
            "schema": "ddm_rg2_skeleton_amplitude_measurement.v1",
            "residual_row_count": 64,
            "new_coordinate_count": len(rg2_ids),
            "unreachable_assignment_row_count": 64 - len(rg2_ids),
            "new_signed_probe_count": len(probes),
            "all_compiled_as_single_coordinate_packets": len(rg2_rows)
            == 2 * len(rg2_ids),
            "unique_candidate_archive_sha256_count": len(candidate_shas),
            "inactive_extension_identity": {
                "proven": compile_rg2_receiver_grammar(context.base_carrier) == context.base_carrier,
                "nested_carrier_sha256": hashlib.sha256(context.base_carrier).hexdigest(),
                "outer_v19c_base_sha256": EXPECTED_BASE_SHA256,
            },
            "grammar_verdict_scope": "INSTANCE_EXTENDED_GRAMMAR_RG2",
            "score_units_per_byte_status": "OWED_NOT_ADMITTED",
            "typed_stream_layer": "SKELETON/L3_raster",
            "superseded_l1_labeled_checkpoint_root": str(
                args.bulk_output / "probe_checkpoints_rg2_v2"
            ),
            "superseded_l1_labeled_checkpoint_disposition": (
                "PRESERVED_NOT_CONSUMED; candidate manifest overstated L1 ownership"
            ),
        }
    elif args.rg1_retry_from is not None:
        rg1_rows = [_read_json(path) for path in sorted(checkpoint_root.glob("*.json"))]
        lane_rows = [row for row in rg1_rows if str(row["receiver_actuator_id"]).startswith("j2.lane.")]
        lane_candidate_shas = {
            str(row["candidate_archive"]["sha256"]) for row in lane_rows if row.get("candidate_archive") is not None
        }
        original_infeasible_counts = {
            "lane": sum(str(row["receiver_actuator_id"]).startswith("j2.lane.") for row in prior_infeasible),
            "g2cs1": sum(str(row["receiver_actuator_id"]).startswith("g2g.g2cs1.") for row in prior_infeasible),
            "geometry_escape": sum(
                not str(row["receiver_actuator_id"]).startswith(("j2.lane.", "g2g.g2cs1.")) for row in prior_infeasible
            ),
        }
        receipt["rg1_extension"] = {
            "schema": "ddm_rg1_receiver_grammar_extension_measurement.v1",
            "g2g_quantum_receipt_sha256": G2G_RECEIPT_SHA256,
            "new_probe_count": len(probes),
            "bounded_geometry_probe_ids": [f"{value}{_BOUNDED_SUFFIX}" for value in geometry_ids],
            "original_infeasible_rows_preserved": len(prior_infeasible),
            "original_infeasible_counts": original_infeasible_counts,
            "grammar_verdict_scope": "INSTANCE_EXTENDED_GRAMMAR_RG1",
            "checkpoint_run_id_policy": (
                f"{RUN_ID} retained inside v2 probe rows for exact backward-compatible "
                "resume; this receipt owns the RG1 run identity"
            ),
            "inactive_extension_identity": {
                "proven": compile_rg1_receiver_grammar(context.base_carrier) == context.base_carrier,
                "nested_carrier_sha256": hashlib.sha256(context.base_carrier).hexdigest(),
                "outer_v19c_base_sha256": EXPECTED_BASE_SHA256,
            },
            "lane_coordinate_isolation": {
                "coordinate_count": len({str(row["receiver_actuator_id"]) for row in lane_rows}),
                "signed_probe_count": len(lane_rows),
                "unique_candidate_archive_sha256_count": len(lane_candidate_shas),
                "all_compiled_as_single_coordinate_packets": len(lane_rows) == 48,
                "all_receiver_feasible": all(row["status"] != "INFEASIBLE_RECEIVER_QUANTUM" for row in lane_rows),
            },
        }
    receipt["receipt_content_sha256"] = canonical_sha256(receipt)
    receipt_path = args.receipt_output / "ddm_ms6_receiver_support_measurement_receipt.json"
    _publish(receipt_path, canonical_bytes(receipt))
    return receipt_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignment-table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--ms5-receipt", type=Path, default=DEFAULT_MS5_RECEIPT)
    parser.add_argument("--pf2", type=Path, default=DEFAULT_PF2)
    parser.add_argument("--base-archive", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--upstream", type=Path, default=DEFAULT_UPSTREAM)
    parser.add_argument("--bulk-output", type=Path, default=DEFAULT_BULK)
    parser.add_argument("--receipt-output", type=Path, default=DEFAULT_RECEIPTS)
    parser.add_argument("--probe-start", type=int, default=0)
    parser.add_argument("--probe-limit", type=int)
    parser.add_argument(
        "--rg1-retry-from",
        type=Path,
        help="Preserved v2 checkpoint root; rerun only formerly infeasible RG1 coordinates.",
    )
    parser.add_argument(
        "--rg2-assignment",
        type=Path,
        help="Run only the 64 assignment-bound RG2 coordinates in both signs.",
    )
    parser.add_argument(
        "--prior-checkpoint-root",
        type=Path,
        action="append",
        dest="prior_checkpoint_roots",
        help="Repeat for exact prior v2 roots; RG2 requires the 748-row and 80-row roots.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        help="Reuse SHA-validated PF2 event-index and baseline-cell caches.",
    )
    args = parser.parse_args()
    if args.rg1_retry_from is not None and args.rg2_assignment is not None:
        parser.error("--rg1-retry-from and --rg2-assignment are mutually exclusive")
    if args.rg2_assignment is not None:
        args.prior_checkpoint_roots = args.prior_checkpoint_roots or list(DEFAULT_RG2_PRIOR_ROOTS)
        if args.bulk_output == DEFAULT_BULK:
            args.bulk_output = DEFAULT_RG2_BULK
        if args.receipt_output == DEFAULT_RECEIPTS:
            args.receipt_output = DEFAULT_RG2_RECEIPTS
        if args.assignment_table == DEFAULT_TABLE:
            args.assignment_table = REPO / (
                ".omx/research/ddm_rg1_receiver_grammar_extension_20260724T080402Z/pf2_bucket_assignment_table.json"
            )
        if args.ms5_receipt == DEFAULT_MS5_RECEIPT:
            args.ms5_receipt = REPO / (
                ".omx/research/ddm_rg1_receiver_grammar_extension_20260724T080402Z/"
                "ddm_ms6_receiver_support_measurement_receipt.json"
            )
        if args.cache_root is None:
            args.cache_root = DEFAULT_RG2_CACHE
    else:
        args.prior_checkpoint_roots = args.prior_checkpoint_roots or []
    if args.probe_start < 0 or (args.probe_limit is not None and args.probe_limit <= 0):
        parser.error("--probe-start must be nonnegative and --probe-limit positive")
    return args


def main() -> int:
    receipt = run(parse_args())
    print(receipt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
