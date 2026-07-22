#!/usr/bin/env python3
"""Measure the three landed G4 static-cell fields through the V14 RGB receiver."""

from __future__ import annotations

import argparse
import hashlib
import json
import lzma
import math
import os
import struct
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap  # noqa: E402
from tac.optimization.direct_description_carrier_compose import (  # noqa: E402
    WORLDSHEET_G1_MEMBER,
    _decode_lane_knots,
    _decode_lane_programs,
    compile_carrier_compose_archive,
    parse_carrier_compose_archive,
    prove_carrier_archive_fail_closed,
    receive_carrier_compose_archive,
)
from tac.optimization.direct_description_entropy_priced_member import (  # noqa: E402
    CLASS_NAMES,
    _storage_preflight,
)
from tac.optimization.direct_description_minimizer import (  # noqa: E402
    POINTER_SCORE_TEXT,
    DirectDescriptionError,
    _read_regular_file_once,
    _sha256,
    rfc8785_canonicalize,
)
from tools.measure_ddm_v14_realization_fidelity import (  # noqa: E402
    DDMV14RealizationFidelityConfigV1,
    _forward,
    _load_models,
)

RESULT_SCHEMA = "ddm_v14_g4_receiver_projection_receipt.v1"
AXIS = "[macOS-CPU frozen-scorer advisory]"
CANDIDATES = (
    "movable_midband_parametric",
    "horizon_row_parametric",
    "static_image_sparse_all",
)
LZMA_FILTERS = [{"id": lzma.FILTER_LZMA1, "dict_size": 1 << 20, "lc": 3, "lp": 0, "pb": 2}]
CONTEST_DENOMINATOR_BYTES = 37_545_489


class DDMV14G4ProjectionConfigV1(BaseModel):
    """SHA-bound local-only companion config with one candidate stage per invocation."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, populate_by_name=True)

    schema_: Literal["DDMV14G4ProjectionConfigV1"] = Field(
        default="DDMV14G4ProjectionConfigV1", alias="schema", serialization_alias="schema"
    )
    run_id: StrictStr
    seed: Literal[1234] = 1234
    base_config_path: StrictStr
    base_config_sha256: StrictStr
    base_archive_path: StrictStr
    base_archive_sha256: StrictStr
    base_measurement_path: StrictStr
    base_measurement_sha256: StrictStr
    base_receipt_path: StrictStr
    base_receipt_sha256: StrictStr
    g4_receipt_path: StrictStr
    g4_receipt_sha256: StrictStr
    g4_summary_path: StrictStr
    g4_summary_sha256: StrictStr
    g4_recurrence_path: StrictStr
    g4_recurrence_bytes: StrictInt
    g4_recurrence_sha256: StrictStr
    candidate_ladder: tuple[
        Literal[
            "movable_midband_parametric",
            "horizon_row_parametric",
            "static_image_sparse_all",
        ],
        ...,
    ] = CANDIDATES
    max_candidate_stages_per_invocation: Literal[1] = 1
    dseg_gate: Literal[0.00116] = 0.00116
    archive_box_bytes: Literal[200000] = 200000
    research_only: Literal[True] = True
    execution_allowed: Literal[False] = False
    score_claim: Literal[False] = False

    @model_validator(mode="after")
    def _valid(self) -> DDMV14G4ProjectionConfigV1:
        for name in (
            "base_config_sha256",
            "base_archive_sha256",
            "base_measurement_sha256",
            "base_receipt_sha256",
            "g4_receipt_sha256",
            "g4_summary_sha256",
            "g4_recurrence_sha256",
        ):
            value = getattr(self, name)
            if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
                raise ValueError(f"{name} must be lowercase SHA-256")
        if tuple(self.candidate_ladder) != CANDIDATES:
            raise ValueError("G4 receiver projection requires the preregistered three-candidate ladder")
        return self

    def typed_hash(self) -> str:
        return _sha256(rfc8785_canonicalize(self.model_dump(mode="json", by_alias=True)))


def _bound_bytes(path: Path, expected: str, name: str) -> bytes:
    raw = _read_regular_file_once(path)
    if _sha256(raw) != expected:
        raise DirectDescriptionError(f"{name} SHA-256 mismatch")
    return raw


def _bound_json(path: Path, expected: str, name: str) -> dict[str, Any]:
    raw = _bound_bytes(path, expected, name)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DirectDescriptionError(f"{name} is malformed JSON") from exc
    if not isinstance(value, dict):
        raise DirectDescriptionError(f"{name} must be one JSON object")
    return value


def _publish(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if _read_regular_file_once(path) != payload:
            raise DirectDescriptionError(f"immutable output differs: {path}")
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _uleb128(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        out.append(byte | (0x80 if value else 0))
        if not value:
            return bytes(out)


def _sparse_all_payload(transition_counts: np.ndarray) -> tuple[bytes, dict[str, int]]:
    gains = np.zeros((25, 384, 512), dtype=np.int16)
    for source in range(5):
        collateral = transition_counts[source * 5 + source].astype(np.int32)
        for target in range(5):
            code = source * 5 + target
            if source != target:
                gains[code] = np.clip(
                    transition_counts[code].astype(np.int32) - collateral,
                    np.iinfo(np.int16).min,
                    np.iinfo(np.int16).max,
                )
    best_code = gains.argmax(axis=0).astype(np.uint8)
    best_gain = np.take_along_axis(gains, best_code[None], axis=0)[0]
    recurrent = np.take_along_axis(transition_counts, best_code[None], axis=0)[0] >= 2
    selected = recurrent & (best_gain > 0)
    indices = np.flatnonzero(selected)
    codes = best_code.reshape(-1)[indices]
    body = bytearray(struct.pack(">4sBHHI", b"G4SR", 1, 384, 512, len(indices)))
    previous = -1
    for index, code in zip(indices, codes, strict=True):
        body.extend(_uleb128(int(index) - previous - 1))
        body.append(int(code))
        previous = int(index)
    coded = lzma.compress(bytes(body), format=lzma.FORMAT_RAW, filters=LZMA_FILTERS)
    payload = bytes([1]) + coded
    return payload, {
        "rule_count": len(indices),
        "net_cell_flips_fixed": int(best_gain[selected].sum(dtype=np.int64)),
        "raw_bytes": len(body),
        "selected_bytes": len(payload),
        "raw_sha256": hashlib.sha256(body).hexdigest(),
    }


def _payloads(transition_counts: np.ndarray) -> tuple[dict[str, bytes], dict[str, dict[str, int]]]:
    sparse, sparse_meta = _sparse_all_payload(transition_counts)
    payloads = {
        "movable_midband_parametric": bytes([0])
        + struct.pack(">4sBHHBB", b"G4MB", 1, 174, 215, 1, 0),
        "horizon_row_parametric": bytes([0])
        + struct.pack(">4sBHBBBB", b"G4HR", 1, 212, 4, 2, 0, 0),
        "static_image_sparse_all": sparse,
    }
    metadata = {
        "movable_midband_parametric": {"selected_bytes": len(payloads["movable_midband_parametric"])},
        "horizon_row_parametric": {"selected_bytes": len(payloads["horizon_row_parametric"])},
        "static_image_sparse_all": sparse_meta,
    }
    return payloads, metadata


def _compile_variants(
    config: DDMV14G4ProjectionConfigV1,
    root: Path,
    payloads: dict[str, bytes],
) -> dict[str, tuple[bytes, Any]]:
    base = _bound_bytes(Path(config.base_archive_path), config.base_archive_sha256, "V14 base archive")
    members, _ = parse_carrier_compose_archive(base)
    base_receiver = receive_carrier_compose_archive(base)
    if base_receiver.realization_profile is None or base_receiver.realization_static_rule_codes is not None:
        raise DirectDescriptionError("G4 projection base must be the unruled V14 realization archive")
    variants: dict[str, tuple[bytes, Any]] = {}
    for name in config.candidate_ladder:
        archive, _ = compile_carrier_compose_archive(
            members["predictor.zip"],
            worldsheet_g1_payload=members[WORLDSHEET_G1_MEMBER],
            lane_programs=_decode_lane_programs(members.get("predict/lane_periodic_programs.ddlp", b"")),
            lane_knots=_decode_lane_knots(members.get("predict/lane_drift_knots.ddlk", b"")),
            realization_profile=base_receiver.realization_profile,
            realization_static_rule_payload=payloads[name],
            realization_static_rule_id=name,
        )
        path = root / f"ddm_v14_g4_{name}_n600.not_a_candidate.zip.receipt-bytes"
        _publish(path, archive)
        variants[name] = (archive, receive_carrier_compose_archive(archive))
    _publish(
        root / "stage_checkpoints/00_receiver_closed_archives.json",
        rfc8785_canonicalize(
            {
                "schema": "ddm_v14_g4_receiver_closed_archives.v1",
                "typed_config_sha256": config.typed_hash(),
                "archives": {
                    name: {
                        "bytes": len(archive),
                        "sha256": _sha256(archive),
                        "rule_payload_bytes": len(payloads[name]),
                        "rule_payload_sha256": _sha256(payloads[name]),
                        "receiver_custody": dict(receiver.custody),
                    }
                    for name, (archive, receiver) in variants.items()
                },
            }
        ),
    )
    return variants


def _measure(
    *,
    name: str,
    archive: bytes,
    receiver: Any,
    base_config: DDMV14RealizationFidelityConfigV1,
    config: DDMV14G4ProjectionConfigV1,
    root: Path,
    labels: np.ndarray,
    poses: np.ndarray,
    segnet: Any,
    posenet: Any,
    probe_loci: tuple[tuple[int, int], ...],
) -> dict[str, Any]:
    stage = root / "stage_checkpoints" / name
    archive_sha = _sha256(archive)
    for start in range(0, 600, 16):
        stop = min(start + 16, 600)
        checkpoint = stage / f"batch_{start:04d}_{stop:04d}.json"
        if checkpoint.exists():
            row = json.loads(_read_regular_file_once(checkpoint))
            if row.get("archive_sha256") != archive_sha or row.get("typed_config_sha256") != config.typed_hash():
                raise DirectDescriptionError("G4 projection batch checkpoint identity differs")
            continue
        camera = receiver.render_camera_pairs(tuple(range(start, stop)))
        cells, pose6 = _forward(segnet, posenet, camera)
        if start == 0:
            replay_cells, replay_pose6 = _forward(segnet, posenet, camera)
            if not np.array_equal(cells, replay_cells) or not np.array_equal(pose6, replay_pose6):
                raise DirectDescriptionError("G4 projection deterministic first-batch replay failed")
        target = np.ascontiguousarray(labels[start:stop])
        target_pose = np.ascontiguousarray(poses[start:stop])
        errors = cells != target
        strata = {}
        for class_id, class_name in enumerate(CLASS_NAMES):
            mask = target == class_id
            strata[class_name] = {
                "errors": int(np.count_nonzero(errors & mask)),
                "sites": int(np.count_nonzero(mask)),
            }
        row = {
            "schema": "ddm_v14_g4_receiver_projection_batch.v1",
            "candidate": name,
            "start": start,
            "stop": stop,
            "archive_sha256": archive_sha,
            "typed_config_sha256": config.typed_hash(),
            "errors": int(np.count_nonzero(errors)),
            "sites": int(errors.size),
            "pose_squared_error_sum": float(np.square(pose6 - target_pose).sum(dtype=np.float64)),
            "pose_values": int(pose6.size),
            "per_stratum": strata,
            "static_loci": [
                {
                    "row": y,
                    "col": x,
                    "errors": int(np.count_nonzero(errors[:, y, x])),
                    "predicted_class_counts": np.bincount(cells[:, y, x], minlength=5).astype(int).tolist(),
                }
                for y, x in probe_loci
            ],
        }
        _publish(checkpoint, rfc8785_canonicalize(row))
    checkpoints = sorted(stage.glob("batch_*.json"))
    if len(checkpoints) != 38:
        raise DirectDescriptionError("G4 projection candidate lacks all batch checkpoints")
    rows = [json.loads(_read_regular_file_once(path)) for path in checkpoints]
    errors = sum(int(row["errors"]) for row in rows)
    sites = sum(int(row["sites"]) for row in rows)
    pose_sum = sum(float(row["pose_squared_error_sum"]) for row in rows)
    pose_values = sum(int(row["pose_values"]) for row in rows)
    per_stratum = {}
    for class_name in CLASS_NAMES:
        class_errors = sum(int(row["per_stratum"][class_name]["errors"]) for row in rows)
        class_sites = sum(int(row["per_stratum"][class_name]["sites"]) for row in rows)
        per_stratum[class_name] = {
            "errors": class_errors,
            "sites": class_sites,
            "d_seg": f"{class_errors / class_sites:.12f}",
        }
    loci = []
    for index, (y, x) in enumerate(probe_loci):
        locus_errors = sum(int(row["static_loci"][index]["errors"]) for row in rows)
        counts = np.sum(
            np.asarray([row["static_loci"][index]["predicted_class_counts"] for row in rows]), axis=0
        ).astype(int)
        loci.append({"row": y, "col": x, "errors": locus_errors, "predicted_class_counts": counts.tolist()})
    return {
        "candidate": name,
        "archive_bytes": len(archive),
        "archive_sha256": archive_sha,
        "d_seg": f"{errors / sites:.12f}",
        "d_pose": f"{pose_sum / pose_values:.12f}",
        "errors": errors,
        "sites": sites,
        "per_stratum": per_stratum,
        "static_loci": loci,
        "batch_count": len(rows),
        "batch_digest_chain_sha256": _sha256(b"".join(_read_regular_file_once(path) for path in checkpoints)),
        "receiver_custody": dict(receiver.custody),
        "score_claim": False,
    }


def run(config: DDMV14G4ProjectionConfigV1, root: Path, semantic_argv: list[str]) -> Path:
    storage = _storage_preflight(root.resolve())
    root.mkdir(parents=True, exist_ok=True)
    receipt_path = root / "ddm_v14_g4_receiver_projection_receipt.json"
    if receipt_path.exists():
        receipt = json.loads(_read_regular_file_once(receipt_path))
        if receipt.get("typed_config_sha256") != config.typed_hash():
            raise DirectDescriptionError("completed G4 projection receipt typed config differs")
        print(json.dumps({"resumed": True, "complete": True, "receipt": str(receipt_path)}))
        return receipt_path
    base_config_raw = _bound_bytes(Path(config.base_config_path), config.base_config_sha256, "V14 base config")
    base_config = DDMV14RealizationFidelityConfigV1.model_validate_json(base_config_raw)
    if (base_config.pair_start, base_config.pair_count) != (0, 600):
        raise DirectDescriptionError("G4 receiver projection requires the full n600 base")
    base_measurement = _bound_json(
        Path(config.base_measurement_path), config.base_measurement_sha256, "V14 islands measurement"
    )
    base_receipt = _bound_json(Path(config.base_receipt_path), config.base_receipt_sha256, "V14 n600 receipt")
    if base_receipt.get("selected_candidate") != "islands" or base_receipt.get("score_claim") is not False:
        raise DirectDescriptionError("V14 n600 base receipt authority differs")
    g4_receipt = _bound_json(Path(config.g4_receipt_path), config.g4_receipt_sha256, "G4 receipt")
    g4_summary = _bound_json(Path(config.g4_summary_path), config.g4_summary_sha256, "G4 summary")
    recurrence_path = Path(config.g4_recurrence_path)
    if recurrence_path.stat().st_size != config.g4_recurrence_bytes:
        raise DirectDescriptionError("G4 recurrence array byte count differs")
    _bound_bytes(recurrence_path, config.g4_recurrence_sha256, "G4 recurrence arrays")
    with np.load(recurrence_path, allow_pickle=False) as arrays:
        transition_counts = np.asarray(arrays["transition_counts"])
        flip_frequency = np.asarray(arrays["flip_frequency"])
    if transition_counts.shape != (25, 384, 512) or flip_frequency.shape != (384, 512):
        raise DirectDescriptionError("G4 recurrence array geometry differs")
    payloads, payload_metadata = _payloads(transition_counts)
    source_rows = {row["opportunity_id"]: row for row in g4_summary["top5_amortization_opportunities"]}
    for name in config.candidate_ladder:
        expected = int(source_rows[name]["byte_measurement"]["selected_bytes"])
        if len(payloads[name]) != expected:
            raise DirectDescriptionError(f"G4 {name} selected payload bytes differ")
    if payload_metadata["static_image_sparse_all"] != {
        "rule_count": 19_661,
        "net_cell_flips_fixed": 920_921,
        "raw_bytes": 39_438,
        "selected_bytes": 4_107,
        "raw_sha256": "033b75ee6b902a328efc34b48fc142a98c11962a41a1b4fd5c32104be730d24a",
    }:
        raise DirectDescriptionError("G4 sparse-all reconstruction differs from the landed row")
    variants = _compile_variants(config, root, payloads)
    flat_probe = np.argsort(-flip_frequency.reshape(-1).astype(np.int64), kind="stable")[:8]
    probe_loci = tuple((int(index // 512), int(index % 512)) for index in flat_probe)
    stage_paths = {
        name: root / "stage_checkpoints" / f"01_{name}_measurement.json" for name in config.candidate_ladder
    }
    missing = [name for name, path in stage_paths.items() if not path.exists()]
    if missing:
        labels = open_stored_npy_memmap(Path(base_config.target_cache_path), "lstars")
        poses = open_stored_npy_memmap(Path(base_config.target_cache_path), "gt_poses")
        segnet, posenet, scorer_custody = _load_models(base_config)
        name = missing[0]
        result = _measure(
            name=name,
            archive=variants[name][0],
            receiver=variants[name][1],
            base_config=base_config,
            config=config,
            root=root,
            labels=labels,
            poses=poses,
            segnet=segnet,
            posenet=posenet,
            probe_loci=probe_loci,
        )
        result["scorer_custody"] = scorer_custody
        _publish(stage_paths[name], rfc8785_canonicalize(result))
        print(json.dumps({"resumed": False, "complete": False, "measured_stage": name}))
        return stage_paths[name]
    measured = {name: json.loads(_read_regular_file_once(path)) for name, path in stage_paths.items()}
    base_dseg = float(base_measurement["d_seg"])
    base_bytes = int(base_measurement["archive_bytes"])
    realized_rows = []
    for name in config.candidate_ladder:
        row = measured[name]
        source = source_rows[name]
        delta_bytes = int(row["archive_bytes"]) - base_bytes
        delta_dseg = float(row["d_seg"]) - base_dseg
        delta_dpose = float(row["d_pose"]) - float(base_measurement["d_pose"])
        delta_seg_score = 100.0 * delta_dseg
        delta_pose_score = math.sqrt(10.0 * float(row["d_pose"])) - math.sqrt(
            10.0 * float(base_measurement["d_pose"])
        )
        delta_rate_score = 25.0 * delta_bytes / CONTEST_DENOMINATOR_BYTES
        delta_joint_score = delta_seg_score + delta_pose_score + delta_rate_score
        realized_rows.append(
            {
                "candidate": name,
                "source_cell_space_delta_d_seg": source["cell_space_delta_d_seg"],
                "receiver_realization_fraction_of_cell_forecast": f"{-delta_dseg / float(source['cell_space_delta_d_seg']):.12f}",
                "source_selected_payload_bytes": source["byte_measurement"]["selected_bytes"],
                "target_derived_aggregate_rule": True,
                "per_frame_ground_truth_argmax_table_present": False,
                "contest_compliance_status": "RESEARCH_ONLY_NOT_A_CANDIDATE",
                "exact_archive_delta_bytes": delta_bytes,
                "receiver_delta_d_seg": f"{delta_dseg:.12f}",
                "receiver_delta_d_pose": f"{delta_dpose:.12f}",
                "advisory_joint_score_delta": {
                    "seg_term": f"{delta_seg_score:.12f}",
                    "pose_sqrt_term": f"{delta_pose_score:.12f}",
                    "rate_term": f"{delta_rate_score:.12f}",
                    "total": f"{delta_joint_score:.12f}",
                    "improves_joint_objective": delta_joint_score < 0.0,
                    "score_gain_per_added_byte": f"{-delta_joint_score / delta_bytes:.12f}",
                    "authority_surface": AXIS,
                    "score_claim": False,
                },
                "receiver_result": row,
            }
        )
    selected = min([base_measurement, *measured.values()], key=lambda row: float(row["d_seg"]))
    gate_pass = float(selected["d_seg"]) <= config.dseg_gate and int(selected["archive_bytes"]) <= config.archive_box_bytes
    free_context = g4_summary["free_context"]["real_coder_measurement"]
    context_free = int(free_context["context_free_raster"]["selected_bytes"])
    contextual = int(free_context["aggregate_pixel_time_order"]["selected_bytes"])
    producer_paths = (
        REPO_ROOT / "tools/measure_ddm_v14_g4_receiver_projection.py",
        REPO_ROOT / "src/tac/optimization/direct_description_carrier_compose.py",
    )
    receipt = {
        "schema": RESULT_SCHEMA,
        "lane_id": "ddm_v14_realization_fidelity",
        "tasks": [603, 613, 578],
        "run_id": config.run_id,
        "typed_config": config.model_dump(mode="json", by_alias=True),
        "typed_config_sha256": config.typed_hash(),
        "semantic_argv": semantic_argv,
        "base": {
            "archive": {"path": config.base_archive_path, "sha256": config.base_archive_sha256},
            "measurement": base_measurement,
            "receipt": {"path": config.base_receipt_path, "sha256": config.base_receipt_sha256},
        },
        "g4_custody": {
            "receipt": {"path": config.g4_receipt_path, "sha256": config.g4_receipt_sha256},
            "summary": {"path": config.g4_summary_path, "sha256": config.g4_summary_sha256},
            "recurrence_arrays": {
                "path": config.g4_recurrence_path,
                "bytes": config.g4_recurrence_bytes,
                "sha256": config.g4_recurrence_sha256,
            },
            "receipt_pointer": g4_receipt["pointer"],
        },
        "payload_reconstruction": payload_metadata,
        "realized_ladder": realized_rows,
        "selected_candidate": selected["candidate"],
        "static_locus_diagnostics": [
            {
                "row": y,
                "col": x,
                "g4_v12_source_flip_count": int(flip_frequency[y, x]),
                "candidate_rows": [
                    {"candidate": name, **measured[name]["static_loci"][index]}
                    for name in config.candidate_ladder
                ],
            }
            for index, (y, x) in enumerate(probe_loci)
        ],
        "free_context_byte_close": {
            "context_free_innovation_bytes": context_free,
            "aggregate_pixel_time_order_bytes": contextual,
            "measured_savings_bytes": context_free - contextual,
            "measured_savings_fraction": (context_free - contextual) / context_free,
            "derived_base_plus_contextual_stream_bytes_before_container_overhead": base_bytes + contextual,
            "applied_to_candidate_archive": False,
            "reason": (
                "the G4 saving prices a future innovation stream absent from these static-field archives; "
                "subtracting it from exact archive bytes would fake byte closure"
            ),
        },
        "fail_closed_mutation_proofs": {
            name: prove_carrier_archive_fail_closed(archive) for name, (archive, _receiver) in variants.items()
        },
        "fork": {
            "condition": "selected receiver-closed n600 d_seg <=0.00116 at <=200000 exact bytes",
            "passed": gate_pass,
            "disposition": (
                "FLAG_MAIN_FOR_R6_EXACT_EVAL_NO_MODAL_DISPATCH"
                if gate_pass
                else "STATIC_CELL_FORECAST_DOES_NOT_TRANSFER_TO_RGB_RECEIVER_DIRECT_SCORER_SOLVE_OPEN"
            ),
        },
        "producer_custody": [
            {
                "path": str(path.relative_to(REPO_ROOT)),
                "bytes": path.stat().st_size,
                "sha256": _sha256(_read_regular_file_once(path)),
            }
            for path in producer_paths
        ],
        "storage_preflight": storage,
        "resume": {"batch_size": 16, "per_batch_checkpoints": True, "all_preserved": True},
        "stores_consulted": [
            "CLAUDE.md",
            "AGENTS.md",
            "PROGRAM.md",
            "docs/operating_manual_craft_handoff.md",
            config.base_receipt_path,
            config.g4_receipt_path,
            config.g4_summary_path,
            config.g4_recurrence_path,
            ".omx/state/lane_registry.json",
            ".omx/state/subagent_progress.jsonl",
        ],
        "pointer": f"{POINTER_SCORE_TEXT} [contest-CPU]",
        "pointer_moved": False,
        "evidence_axis": AXIS,
        "research_only": True,
        "execution_allowed": False,
        "score_claim": False,
        "d_seg_claim": False,
        "d_pose_claim": False,
        "promotion_eligible": False,
        "main_landing_review_required": True,
    }
    _publish(receipt_path, rfc8785_canonicalize(receipt))
    print(json.dumps({"resumed": False, "complete": True, "receipt": str(receipt_path)}))
    return receipt_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()
    config = DDMV14G4ProjectionConfigV1.model_validate_json(_read_regular_file_once(args.config))
    run(
        config,
        args.output_directory,
        [
            "tools/measure_ddm_v14_g4_receiver_projection.py",
            "--config",
            str(args.config),
            "--output-directory",
            str(args.output_directory),
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
