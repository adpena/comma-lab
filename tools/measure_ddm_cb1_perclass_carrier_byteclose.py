#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Byte-close RG4 per-class carriers and measure their exact n600 receiver rows."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import sys
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, model_validator

REPO_ROOT = Path(__file__).resolve().parents[1]
for _path in (REPO_ROOT / "src", REPO_ROOT):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from tac.boundary_math.power_diagram_witness import open_stored_npy_memmap  # noqa: E402
from tac.optimization.ddm_hood_static_reassert import (  # noqa: E402
    class_transition_rows,
    decode_stored_support,
)
from tac.optimization.ddm_realized_flip_menu import advisory_objective  # noqa: E402
from tac.optimization.ddm_rg4_g3_blocks_and_active_tube import (  # noqa: E402
    build_source_local_composition_archive,
    parse_source_local_composition_archive,
    receive_source_local_pc1_camera_pairs,
)
from tac.optimization.ddm_runtime_exporter import (  # noqa: E402
    compile_cb1_rg4_runtime_packet,
)
from tac.optimization.ddm_ws1_warm_start import (  # noqa: E402
    parse_ws1_warm_start_archive,
    receive_ws1_warm_start_archive,
)
from tac.optimization.direct_description_carrier_compose import (  # noqa: E402
    CLASS_ORDER,
    LANE_KNOT_MEMBER,
    LANE_PROGRAM_MEMBER,
    REALIZATION_PROFILE_MEMBER,
    SCORER_SOLVED_TEMPLATE_MEMBER,
    WORLDSHEET_G1_MEMBER,
    _decode_lane_knots,
    _decode_lane_programs,
    compile_carrier_compose_archive,
    encode_static_class_mask_rule,
    parse_carrier_compose_archive,
    receive_carrier_compose_archive,
)
from tac.optimization.direct_description_measurement_ladder import (  # noqa: E402
    rfc8785_canonicalize,
)
from tac.process_group_kill import run_in_process_group  # noqa: E402

SCHEMA = "DDMCB1PerClassCarrierBytecloseConfigV1"
RECEIPT_SCHEMA = "ddm_cb1_perclass_carrier_byteclose_measurement.v1"
RUN_ID = "ddm_cb1_perclass_carrier_byteclose_20260725T203310Z"
EVIDENCE_AXIS = "[macOS-CPU frozen-scorer advisory]"
DELEGATION_KEY = "codex_delegate:ddm_cb1_perclass_carrier_byteclose:20260725T203310Z"
CLASS_NAMES = dict(enumerate(CLASS_ORDER))
CAMERA_SHAPE = (600, 2, 874, 1164, 3)
OUTPUT_BYTES = int(np.prod(CAMERA_SHAPE))
MANIFEST_MEMBER = "manifest.json"
PREDICTOR_MEMBER = "predictor.zip"
EXPECTED_CURRENT_CARRIER_MEMBERS = {
    MANIFEST_MEMBER,
    PREDICTOR_MEMBER,
    WORLDSHEET_G1_MEMBER,
    REALIZATION_PROFILE_MEMBER,
    SCORER_SOLVED_TEMPLATE_MEMBER,
}
EXPECTED_LANE_DONOR_MEMBERS = {
    MANIFEST_MEMBER,
    PREDICTOR_MEMBER,
    LANE_PROGRAM_MEMBER,
    LANE_KNOT_MEMBER,
}


class CB1Error(RuntimeError):
    """Raised when CB1 custody, receiver consumption, or measurement differs."""


class CB1Config(BaseModel):
    """Typed, local-only and resumable CB1 measurement contract."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_: Literal["DDMCB1PerClassCarrierBytecloseConfigV1"] = Field(
        default=SCHEMA,
        alias="schema",
        serialization_alias="schema",
    )
    run_id: Literal["ddm_cb1_perclass_carrier_byteclose_20260725T203310Z"] = RUN_ID
    base_source_archive_path: StrictStr
    base_source_archive_bytes: StrictInt = Field(gt=0)
    base_source_archive_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    mc1_receipt_path: StrictStr
    mc1_receipt_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    mc1_static_support_path: StrictStr
    mc1_static_support_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    lane_donor_archive_path: StrictStr
    lane_donor_archive_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    menu1_config_path: StrictStr
    menu1_config_sha256: StrictStr = Field(pattern=r"^[0-9a-f]{64}$")
    output_directory: StrictStr
    checkpoint_root: StrictStr
    minimum_free_bytes: StrictInt = Field(ge=40 * 1024**3)
    pair_count: Literal[600] = 600
    source_batch_pairs: Literal[32] = 32
    scorer_batch_pairs: Literal[16] = 16
    scorer_threads: Literal[4] = 4
    seed: Literal[210] = 210
    candidate_order: tuple[
        Literal["control", "mycar_static_mask", "lane_band_v13_polished"],
        Literal["control", "mycar_static_mask", "lane_band_v13_polished"],
        Literal["control", "mycar_static_mask", "lane_band_v13_polished"],
    ] = ("control", "mycar_static_mask", "lane_band_v13_polished")
    execution_allowed: Literal[True] = True
    exact_eval_allowed: Literal[False] = False
    paid_dispatch_allowed: Literal[False] = False
    frontier_mutation_allowed: Literal[False] = False
    research_only: Literal[True] = True
    score_claim: Literal[False] = False

    @model_validator(mode="after")
    def _valid_paths(self) -> CB1Config:
        if tuple(self.candidate_order) != (
            "control",
            "mycar_static_mask",
            "lane_band_v13_polished",
        ):
            raise ValueError("candidate_order must preserve cheapest-first CB1 policy")
        if Path(self.output_directory).is_absolute():
            raise ValueError("output_directory must be repository-relative")
        if not Path(self.checkpoint_root).is_absolute() or not (
            self.checkpoint_root.startswith("/Volumes/VertigoDataTier/pact/")
            or self.checkpoint_root.startswith("/Volumes/APDataStore/pact/")
        ):
            raise ValueError("checkpoint_root must use governed SSD custody")
        return self

    def stable_hash(self) -> str:
        return _sha256(
            rfc8785_canonicalize(self.model_dump(mode="json", by_alias=True))
        )


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    total = 0
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
            total += len(chunk)
    return total, digest.hexdigest()


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _read_bound(
    value: str,
    expected_sha256: str,
    *,
    label: str,
    expected_bytes: int | None = None,
) -> bytes:
    path = _resolve(value)
    if path.is_symlink() or not path.is_file():
        raise CB1Error(f"{label} must be one regular file: {path}")
    payload = path.read_bytes()
    if _sha256(payload) != expected_sha256 or (
        expected_bytes is not None and len(payload) != expected_bytes
    ):
        raise CB1Error(f"{label} custody differs")
    return payload


def _publish(path: Path, payload: bytes, *, executable: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise CB1Error(f"immutable artifact differs: {path}")
    else:
        temporary = path.with_name(path.name + f".partial.{os.getpid()}")
        with temporary.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    if executable:
        path.chmod(0o755)
    return path


def _publish_json(path: Path, value: Any) -> Path:
    return _publish(path, rfc8785_canonicalize(value) + b"\n")


def _write_npz(path: Path, *, cells: np.ndarray, pose6: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        with np.load(path, allow_pickle=False) as stored:
            if not np.array_equal(stored["cells"], cells) or not np.array_equal(
                stored["pose6"], pose6
            ):
                raise CB1Error(f"preserved scorer arrays differ: {path}")
        return
    temporary = path.with_name(path.name + f".partial.{os.getpid()}")
    with temporary.open("xb") as handle:
        np.savez_compressed(handle, cells=cells, pose6=pose6)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise CB1Error(f"JSON receipt is absent: {path}")
    value = json.loads(path.read_bytes())
    if not isinstance(value, dict):
        raise CB1Error(f"JSON receipt is not an object: {path}")
    return value


def _self_detected_hood_class(mc1_receipt: Mapping[str, Any]) -> int:
    derivation = mc1_receipt.get("support_derivation")
    if not isinstance(derivation, Mapping):
        raise CB1Error("MC1 support derivation is absent")
    rows = derivation.get("class_evidence")
    if (
        not isinstance(rows, list)
        or len(rows) != len(CLASS_ORDER)
        or any(not isinstance(row, Mapping) for row in rows)
    ):
        raise CB1Error("MC1 class evidence differs")
    if sorted(int(row["class_id"]) for row in rows) != list(range(len(CLASS_ORDER))):
        raise CB1Error("MC1 class evidence does not cover canonical classes exactly once")
    scored = [
        (float(row["bottom_share"]) * float(row["static_iou"]), row)
        for row in rows
    ]
    best_score = max(score for score, _row in scored)
    winners = [row for score, row in scored if score == best_score]
    if len(winners) != 1:
        raise CB1Error("MC1 spatial/static class evidence does not have a unique winner")
    detected = winners[0]
    detected_class = int(detected["class_id"])
    if detected_class != int(derivation.get("detected_class_id", -1)):
        raise CB1Error("MC1 self-detected class does not reproduce from spatial evidence")
    return detected_class


def _candidate_archives(
    config: CB1Config,
) -> tuple[dict[str, bytes], dict[str, dict[str, Any]]]:
    source = _read_bound(
        config.base_source_archive_path,
        config.base_source_archive_sha256,
        label="RG4 source-local base",
        expected_bytes=config.base_source_archive_bytes,
    )
    mc1_payload = _read_bound(
        config.mc1_receipt_path,
        config.mc1_receipt_sha256,
        label="MC1 receipt",
    )
    mc1_receipt = json.loads(mc1_payload)
    if (
        mc1_receipt.get("schema") != "ddm_mc1_hood_static_reassert_measurement.v1"
        or mc1_receipt.get("score_claim") is not False
    ):
        raise CB1Error("MC1 authority receipt differs")
    support_payload = _read_bound(
        config.mc1_static_support_path,
        config.mc1_static_support_sha256,
        label="MC1 static support",
    )
    support = decode_stored_support(support_payload)
    if support.shape != (384, 512) or support.dtype != np.bool_:
        raise CB1Error("MC1 static support geometry differs")
    detected_class = _self_detected_hood_class(mc1_receipt)

    parent_archive, packet, source_manifest = parse_source_local_composition_archive(
        source
    )
    parsed_parent = parse_ws1_warm_start_archive(parent_archive)
    current_members, _homes = parse_carrier_compose_archive(
        parsed_parent.carrier_archive
    )
    if set(current_members) != EXPECTED_CURRENT_CARRIER_MEMBERS:
        raise CB1Error(
            "current carrier has unhandled members; refusing signal-losing rewrap"
        )
    current_receiver = receive_carrier_compose_archive(
        parsed_parent.carrier_archive,
        verify_member_effects=False,
    )
    if (
        current_receiver.realization_profile is None
        or current_receiver.scorer_solved_templates is None
        or current_receiver.lane_programs
        or current_receiver.lane_knots
        or current_receiver.realization_static_rule_id is not None
    ):
        raise CB1Error("current carrier topology differs from the typed CB1 base")

    lane_donor = _read_bound(
        config.lane_donor_archive_path,
        config.lane_donor_archive_sha256,
        label="polished V13 Lane carrier donor",
    )
    lane_members, _lane_homes = parse_carrier_compose_archive(lane_donor)
    if set(lane_members) != EXPECTED_LANE_DONOR_MEMBERS:
        raise CB1Error("Lane donor contains an unexpected carrier member")
    lane_programs = _decode_lane_programs(lane_members[LANE_PROGRAM_MEMBER])
    lane_knots = _decode_lane_knots(lane_members[LANE_KNOT_MEMBER])
    if not lane_programs or not lane_knots:
        raise CB1Error("Lane donor lacks its counted periodic program or drift knots")

    common = {
        "worldsheet_g1_payload": current_members[WORLDSHEET_G1_MEMBER],
        "realization_profile": current_receiver.realization_profile,
        "scorer_solved_templates": current_receiver.scorer_solved_templates,
    }
    mycar_rule = encode_static_class_mask_rule(
        support,
        target_class=detected_class,
    )
    mycar_carrier, _ = compile_carrier_compose_archive(
        current_members[PREDICTOR_MEMBER],
        **common,
        realization_static_rule_payload=mycar_rule,
        realization_static_rule_id="mycar_static_mask",
    )
    lane_carrier, _ = compile_carrier_compose_archive(
        current_members[PREDICTOR_MEMBER],
        **common,
        lane_programs=lane_programs,
        lane_knots=lane_knots,
    )
    carrier_candidates = {
        "mycar_static_mask": mycar_carrier,
        "lane_band_v13_polished": lane_carrier,
    }
    archives = {"control": source}
    metadata: dict[str, dict[str, Any]] = {
        "control": {
            "carrier_archive_bytes": len(parsed_parent.carrier_archive),
            "carrier_archive_sha256": _sha256(parsed_parent.carrier_archive),
            "mechanism": "merged_RG4_source_local_control",
        }
    }
    for candidate_id, carrier in carrier_candidates.items():
        candidate_parent = parsed_parent.rewrap_carrier(carrier)
        candidate = build_source_local_composition_archive(
            parent_archive=candidate_parent,
            parent_sha256=_sha256(candidate_parent),
            packet=packet,
        )
        reparsed_parent, reparsed_packet, _ = (
            parse_source_local_composition_archive(candidate)
        )
        if (
            reparsed_parent != candidate_parent
            or reparsed_packet.q_xi.tobytes() != packet.q_xi.tobytes()
            or reparsed_packet.q_luma_phase.tobytes()
            != packet.q_luma_phase.tobytes()
        ):
            raise CB1Error(f"{candidate_id} source-local parse-back differs")
        archives[candidate_id] = candidate
        receiver = receive_carrier_compose_archive(
            carrier,
            verify_member_effects=False,
        )
        metadata[candidate_id] = {
            "carrier_archive_bytes": len(carrier),
            "carrier_archive_sha256": _sha256(carrier),
            "source_archive_bytes": len(candidate),
            "source_archive_sha256": _sha256(candidate),
            "mechanism": (
                "self_detected_static_support_receiver_paint"
                if candidate_id == "mycar_static_mask"
                else "v13_polished_lane_periodic_program_plus_drift_knots"
            ),
            "mycar": (
                {
                    "detected_class_id": detected_class,
                    "static_rule_bytes": len(mycar_rule),
                    "static_rule_sha256": _sha256(mycar_rule),
                    "support_active_sites": int(np.count_nonzero(support)),
                    "support_source_bytes": len(support_payload),
                    "support_source_sha256": _sha256(support_payload),
                }
                if candidate_id == "mycar_static_mask"
                else None
            ),
            "lane": (
                {
                    "program_count": len(lane_programs),
                    "program_payload_bytes": len(
                        lane_members[LANE_PROGRAM_MEMBER]
                    ),
                    "program_payload_sha256": _sha256(
                        lane_members[LANE_PROGRAM_MEMBER]
                    ),
                    "knot_count": len(lane_knots),
                    "knot_payload_bytes": len(lane_members[LANE_KNOT_MEMBER]),
                    "knot_payload_sha256": _sha256(
                        lane_members[LANE_KNOT_MEMBER]
                    ),
                }
                if candidate_id == "lane_band_v13_polished"
                else None
            ),
            "receiver_custody": dict(receiver.custody),
        }
    metadata["control"]["source_archive_bytes"] = len(source)
    metadata["control"]["source_archive_sha256"] = _sha256(source)
    metadata["control"]["source_schema"] = source_manifest["schema"]
    return archives, metadata


def _prepare_rg4_renderer(source_archive: bytes) -> tuple[Any, Any, Any]:
    parent_archive, packet, _manifest = parse_source_local_composition_archive(
        source_archive
    )
    receiver = receive_ws1_warm_start_archive(parent_archive)
    try:
        movable_layer = next(
            layer for layer in receiver.layers if layer.role == "Movable"
        )
    except StopIteration as exc:
        raise CB1Error("RG4 parent lacks a Movable layer") from exc
    return receiver, packet, movable_layer


def _render_prepared_rg4_batch(
    receiver: Any,
    packet: Any,
    movable_layer: Any,
    pair_ids: Sequence[int],
) -> np.ndarray:
    indexes = tuple(int(value) for value in pair_ids)
    parent = receiver.render_camera_pairs(indexes)
    masks = np.stack(
        [
            receiver._mask_for_layer(
                movable_layer,
                pair_id,
                replace_g1_movable=True,
            )
            for pair_id in indexes
        ],
        axis=0,
    ).astype(np.bool_)
    camera = receive_source_local_pc1_camera_pairs(
        parent_camera=parent,
        packet=packet,
        pair_ids=indexes,
        movable_masks=masks,
    )
    if (
        camera.dtype != np.uint8
        or camera.shape != (len(indexes), *CAMERA_SHAPE[1:])
        or not camera.flags.c_contiguous
    ):
        raise CB1Error("RG4 source render geometry differs")
    return camera


def _render_rg4_batch(source_archive: bytes, pair_ids: Sequence[int]) -> np.ndarray:
    receiver, packet, movable_layer = _prepare_rg4_renderer(source_archive)
    return _render_prepared_rg4_batch(
        receiver,
        packet,
        movable_layer,
        pair_ids,
    )


def _source_output_identity(
    *,
    candidate_id: str,
    source_archive: bytes,
    config: CB1Config,
    root: Path,
) -> dict[str, Any]:
    checkpoint = root / "source_identity" / candidate_id
    checkpoint.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    total = 0
    rows: list[dict[str, Any]] = []
    source_sha256 = _sha256(source_archive)
    receiver, packet, movable_layer = _prepare_rg4_renderer(source_archive)
    for start in range(0, 600, config.source_batch_pairs):
        stop = min(start + config.source_batch_pairs, 600)
        raw_path = checkpoint / f"pairs_{start:04d}_{stop:04d}.raw"
        row_path = checkpoint / f"pairs_{start:04d}_{stop:04d}.json"
        expected_bytes = (stop - start) * int(np.prod(CAMERA_SHAPE[1:]))
        if raw_path.is_file() and row_path.is_file():
            row = _load_json(row_path)
            if (
                row
                != {
                    "bytes": row.get("bytes"),
                    "candidate_id": candidate_id,
                    "pair_start": start,
                    "pair_stop": stop,
                    "sha256": row.get("sha256"),
                    "source_archive_sha256": source_sha256,
                }
                or row["bytes"] != expected_bytes
                or _sha256_file(raw_path) != (row["bytes"], row["sha256"])
            ):
                raise CB1Error("source output-identity checkpoint differs")
        elif raw_path.exists() or row_path.exists():
            raise CB1Error("source output-identity checkpoint is incomplete")
        else:
            camera = _render_prepared_rg4_batch(
                receiver,
                packet,
                movable_layer,
                range(start, stop),
            )
            payload = camera.tobytes(order="C")
            _publish(raw_path, payload)
            row = {
                "bytes": len(payload),
                "candidate_id": candidate_id,
                "pair_start": start,
                "pair_stop": stop,
                "sha256": _sha256(payload),
                "source_archive_sha256": source_sha256,
            }
            _publish_json(row_path, row)
        with raw_path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                digest.update(chunk)
                total += len(chunk)
        rows.append(row)
        print(
            json.dumps(
                {
                    "stage": "source_identity",
                    "candidate_id": candidate_id,
                    "pair_range": [start, stop],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if total != OUTPUT_BYTES:
        raise CB1Error("source output-identity byte count differs")
    return {
        "bytes": total,
        "candidate_id": candidate_id,
        "sha256": digest.hexdigest(),
        "source_archive_sha256": source_sha256,
        "all_stage_checkpoints_preserved": True,
        "stage_count": len(rows),
    }


def _extract_packet(archive: bytes, output: Path) -> None:
    expected: dict[str, bytes] = {}
    with zipfile.ZipFile(io.BytesIO(archive), "r") as handle:
        for info in handle.infolist():
            if (
                info.is_dir()
                or info.filename.startswith("/")
                or ".." in Path(info.filename).parts
                or "\\" in info.filename
            ):
                raise CB1Error("packet ZIP member is unsafe")
            expected[info.filename] = handle.read(info.filename)
    for name, payload in expected.items():
        _publish(output / name, payload)
    observed = sorted(
        path.relative_to(output).as_posix()
        for path in output.rglob("*")
        if path.is_file()
    )
    if observed != sorted(expected):
        raise CB1Error("extracted packet tree contains extra members")


def _byteclose_candidate(
    *,
    candidate_id: str,
    source_archive: bytes,
    identity: Mapping[str, Any],
    config: CB1Config,
    root: Path,
    output_directory: Path,
) -> dict[str, Any]:
    archive, packet = compile_cb1_rg4_runtime_packet(
        source_archive,
        state_name=f"{candidate_id}:rg4_source_local",
        output_bytes=int(identity["bytes"]),
        output_sha256=str(identity["sha256"]),
    )
    packet_dir = output_directory / "packets" / candidate_id
    archive_path = _publish(packet_dir / "archive.zip", archive)
    runtime_dir = output_directory / "runtime"
    runtime_path = _publish(
        runtime_dir / "inflate.py",
        bytes(packet["runtime_payload"]),
        executable=True,
    )
    script_path = _publish(
        runtime_dir / "inflate.sh",
        bytes(packet["inflate_sh"]),
        executable=True,
    )
    runtime_root = root / "runtime" / candidate_id
    extracted = runtime_root / "archive"
    _extract_packet(archive, extracted)
    video_names = _publish(runtime_root / "video_names.txt", b"0.mkv\n")
    output_root = runtime_root / "output"
    final_path = output_root / "0.raw"
    environment = dict(os.environ)
    environment["PYTHON"] = sys.executable
    completed = run_in_process_group(
        [
            str(script_path),
            str(extracted),
            str(output_root),
            str(video_names),
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=30 * 60,
    )
    stdout_lines = [row for row in completed.stdout.splitlines() if row.strip()]
    if len(stdout_lines) != 1:
        raise CB1Error("inflate runtime did not emit one receipt")
    inflate_receipt = json.loads(stdout_lines[0])
    if (
        inflate_receipt.get("schema")
        != "ddm_cb1_rg4_perclass_runtime_inflate_receipt.v1"
        or inflate_receipt.get("score_claim") is not False
        or _sha256_file(final_path)
        != (int(identity["bytes"]), str(identity["sha256"]))
    ):
        raise CB1Error("byte-closed runtime output differs from source identity")
    _publish_json(runtime_root / "inflate_stdout_receipt.json", inflate_receipt)
    return {
        "archive_bytes": len(archive),
        "archive_path": str(archive_path.relative_to(REPO_ROOT)),
        "archive_sha256": _sha256(archive),
        "coder": packet["coder"],
        "compiler_determinism_x2": packet["compiler_determinism_x2"],
        "inflate_receipt": inflate_receipt,
        "inflate_sh_sha256": _sha256(bytes(packet["inflate_sh"])),
        "member_homes": packet["member_homes"],
        "packet_parseback_source_byte_identical": packet[
            "packet_parseback_source_byte_identical"
        ],
        "raw_path": str(final_path),
        "raw_sha256": str(identity["sha256"]),
        "runtime_source_bundle_embedded": True,
        "runtime_sha256": _sha256(bytes(packet["runtime_payload"])),
        "runtime_path": str(runtime_path.relative_to(REPO_ROOT)),
    }


def _score_candidate(
    *,
    candidate_id: str,
    byteclose: Mapping[str, Any],
    config: CB1Config,
    root: Path,
    labels: np.ndarray,
    poses: np.ndarray,
    segnet: Any,
    posenet: Any,
) -> dict[str, Any]:
    from tools import measure_ddm_menu1_realized_flip_menu as menu1

    raw_path = Path(str(byteclose["raw_path"]))
    if _sha256_file(raw_path) != (OUTPUT_BYTES, str(byteclose["raw_sha256"])):
        raise CB1Error("scorer input raw identity differs")
    camera = np.memmap(
        raw_path,
        mode="r",
        dtype=np.uint8,
        shape=CAMERA_SHAPE,
    )
    score_root = root / "scores" / candidate_id
    rows: list[dict[str, Any]] = []
    first_replayed = False
    for start in range(0, 600, config.scorer_batch_pairs):
        stop = min(start + config.scorer_batch_pairs, 600)
        row_path = score_root / f"batch_{start:04d}_{stop:04d}.json"
        array_path = score_root / f"batch_{start:04d}_{stop:04d}.npz"
        if row_path.is_file() and array_path.is_file():
            row = _load_json(row_path)
            if (
                row.get("typed_config_sha256") != config.stable_hash()
                or row.get("candidate_id") != candidate_id
                or row.get("raw_sha256") != byteclose["raw_sha256"]
            ):
                raise CB1Error("preserved scorer checkpoint differs")
            rows.append(row)
            continue
        if row_path.exists() or array_path.exists():
            raise CB1Error("scorer checkpoint is incomplete")
        batch = np.array(camera[start:stop], dtype=np.uint8, copy=True, order="C")
        cells, pose6 = menu1._forward(segnet, posenet, batch)
        if not first_replayed:
            replay_cells, replay_pose6 = menu1._forward(segnet, posenet, batch)
            if not np.array_equal(cells, replay_cells) or not np.array_equal(
                pose6, replay_pose6
            ):
                raise CB1Error("first scorer batch deterministic replay differs")
            first_replayed = True
        target = np.asarray(labels[start:stop], dtype=np.uint8)
        target_pose = np.asarray(poses[start:stop], dtype=np.float64)
        if candidate_id == "control":
            control_cells = cells
        else:
            control_path = (
                root
                / "scores"
                / "control"
                / f"batch_{start:04d}_{stop:04d}.npz"
            )
            with np.load(control_path, allow_pickle=False) as stored:
                control_cells = np.asarray(stored["cells"], dtype=np.uint8)
        per_class = class_transition_rows(
            before=control_cells,
            after=cells,
            target=target,
            class_names=CLASS_NAMES,
        )
        row = {
            "schema": "ddm_cb1_perclass_carrier_n600_batch.v1",
            "typed_config_sha256": config.stable_hash(),
            "candidate_id": candidate_id,
            "pair_range": [start, stop],
            "raw_sha256": byteclose["raw_sha256"],
            "camera_sha256": _sha256(batch.tobytes()),
            "cells_sha256": _sha256(cells.tobytes()),
            "pose6_sha256": _sha256(pose6.tobytes()),
            "errors": int(np.count_nonzero(cells != target)),
            "sites": int(cells.size),
            "pose_squared_error_sum": float(
                np.square(pose6 - target_pose).sum(dtype=np.float64)
            ),
            "pose_coordinates": int(pose6.size),
            "per_class": per_class,
            "evidence_axis": EVIDENCE_AXIS,
            "research_only": True,
            "score_claim": False,
        }
        _write_npz(array_path, cells=cells, pose6=pose6)
        _publish_json(row_path, row)
        rows.append(row)
        print(
            json.dumps(
                {
                    "stage": "score",
                    "candidate_id": candidate_id,
                    "pair_range": [start, stop],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    errors = sum(int(row["errors"]) for row in rows)
    sites = sum(int(row["sites"]) for row in rows)
    pose_sse = sum(float(row["pose_squared_error_sum"]) for row in rows)
    pose_coordinates = sum(int(row["pose_coordinates"]) for row in rows)
    per_class: dict[str, dict[str, Any]] = {}
    for name in CLASS_ORDER:
        class_sites = sum(int(row["per_class"][name]["sites"]) for row in rows)
        per_class[name] = {
            key: sum(int(row["per_class"][name][key]) for row in rows)
            for key in (
                "sites",
                "errors_before",
                "errors_after",
                "errors_corrected",
                "errors_introduced",
                "delta_errors_realized",
            )
        }
        per_class[name]["d_seg_after"] = (
            per_class[name]["errors_after"] / class_sites
        )
    d_pose = pose_sse / pose_coordinates
    return {
        "candidate_id": candidate_id,
        "archive_bytes": int(byteclose["archive_bytes"]),
        "archive_sha256": byteclose["archive_sha256"],
        "errors": errors,
        "sites": sites,
        "d_seg": errors / sites,
        "d_pose": d_pose,
        "advisory_objective": advisory_objective(
            errors=errors,
            sites=sites,
            d_pose=d_pose,
            bytes_=int(byteclose["archive_bytes"]),
        ),
        "per_class": per_class,
        "batch_count": len(rows),
        "all_batches_checkpointed_and_preserved": True,
        "evidence_axis": EVIDENCE_AXIS,
        "research_only": True,
        "score_claim": False,
    }


def _delta_row(
    *,
    candidate: Mapping[str, Any],
    control: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    delta_s = float(candidate["advisory_objective"]) - float(
        control["advisory_objective"]
    )
    delta_bytes = int(candidate["archive_bytes"]) - int(control["archive_bytes"])
    per_class = {
        name: {
            "sites": int(candidate["per_class"][name]["sites"]),
            "control_errors": int(control["per_class"][name]["errors_after"]),
            "candidate_errors": int(candidate["per_class"][name]["errors_after"]),
            "delta_errors_realized": (
                int(control["per_class"][name]["errors_after"])
                - int(candidate["per_class"][name]["errors_after"])
            ),
            "control_d_seg": float(control["per_class"][name]["d_seg_after"]),
            "candidate_d_seg": float(candidate["per_class"][name]["d_seg_after"]),
            "delta_d_seg": (
                float(candidate["per_class"][name]["d_seg_after"])
                - float(control["per_class"][name]["d_seg_after"])
            ),
        }
        for name in CLASS_ORDER
    }
    if delta_s < 0.0:
        disposition = "ADMIT_STRICT_NEGATIVE_JOINT_DELTA_S_TO_C1_WATERFILL"
        uphill_reason = None
    else:
        disposition = "REJECT_UPHILL_FROM_C1_WATERFILL"
        pose_delta = float(candidate["d_pose"]) - float(control["d_pose"])
        corrected = sum(
            max(0, int(row["delta_errors_realized"])) for row in per_class.values()
        )
        collateral = sum(
            max(0, -int(row["delta_errors_realized"])) for row in per_class.values()
        )
        uphill_reason = {
            "primary_leg": (
                "pose_survival"
                if pose_delta > 0.0
                else "paint_collateral"
                if collateral > corrected
                else "quantized_receiver_realization"
            ),
            "pose_delta": pose_delta,
            "seg_errors_corrected_net_positive_buckets": corrected,
            "seg_errors_added_net_negative_buckets": collateral,
        }
    return {
        "schema": "ddm_c1_bucket_attribution_row.v1",
        "candidate_id": candidate["candidate_id"],
        "application_stage": "RG4 source-local PC1 after per-class carrier rewrap",
        "composition_pool_id": "cb1_perclass_carrier",
        "parent_candidate_id": "control",
        "incremental_archive_bytes": delta_bytes,
        "control_archive_bytes": int(control["archive_bytes"]),
        "candidate_archive_bytes": int(candidate["archive_bytes"]),
        "control_d_seg": float(control["d_seg"]),
        "candidate_d_seg": float(candidate["d_seg"]),
        "delta_d_seg": float(candidate["d_seg"]) - float(control["d_seg"]),
        "control_d_pose": float(control["d_pose"]),
        "candidate_d_pose": float(candidate["d_pose"]),
        "delta_d_pose": float(candidate["d_pose"]) - float(control["d_pose"]),
        "delta_joint_s": delta_s,
        "joint_score_units_improved_per_counted_byte": (
            None if delta_bytes <= 0 else -delta_s / delta_bytes
        ),
        "per_class": per_class,
        "carrier_metadata": metadata,
        "waterfill_eligible": delta_s < 0.0,
        "disposition": disposition,
        "uphill_reason": uphill_reason,
        "verdict_scope": (
            "INSTANCE: exact carrier payload on merged RG4 source-local base; "
            "negative does not close the broader per-class carrier family"
        ),
        "evidence_axis": EVIDENCE_AXIS,
        "research_only": True,
        "score_claim": False,
    }


def run(config_path: Path) -> Path:
    config_payload = config_path.read_bytes()
    config = CB1Config.model_validate_json(config_payload, strict=True)
    if config.stable_hash() != _sha256(
        rfc8785_canonicalize(
            config.model_dump(mode="json", by_alias=True)
        )
    ):
        raise CB1Error("typed config hash is unstable")
    output_directory = (REPO_ROOT / config.output_directory).resolve()
    try:
        output_directory.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise CB1Error("output_directory escaped repository") from exc
    root = Path(config.checkpoint_root)
    root.mkdir(parents=True, exist_ok=True)
    observed_free = shutil.disk_usage(root).free
    if observed_free < config.minimum_free_bytes:
        raise CB1Error(
            f"storage preflight failed: {observed_free} < {config.minimum_free_bytes}"
        )
    archives, metadata = _candidate_archives(config)
    sources_root = root / "sources"
    for candidate_id, archive in archives.items():
        _publish(
            sources_root / f"{candidate_id}.zip.receipt-bytes",
            archive,
        )

    identities: dict[str, dict[str, Any]] = {}
    byteclose: dict[str, dict[str, Any]] = {}
    for candidate_id in config.candidate_order:
        identity = _source_output_identity(
            candidate_id=candidate_id,
            source_archive=archives[candidate_id],
            config=config,
            root=root,
        )
        identities[candidate_id] = identity
        byteclose[candidate_id] = _byteclose_candidate(
            candidate_id=candidate_id,
            source_archive=archives[candidate_id],
            identity=identity,
            config=config,
            root=root,
            output_directory=output_directory,
        )

    from tools import measure_ddm_menu1_realized_flip_menu as menu1

    menu_payload = _read_bound(
        config.menu1_config_path,
        config.menu1_config_sha256,
        label="MENU1 scorer config",
    )
    menu_config = menu1.Menu1Config.model_validate_json(menu_payload)
    labels = open_stored_npy_memmap(Path(menu_config.target_cache_path), "lstars")
    poses = open_stored_npy_memmap(Path(menu_config.target_cache_path), "gt_poses")
    segnet, posenet, scorer_custody = menu1._load_models(menu_config)
    scores: dict[str, dict[str, Any]] = {}
    for candidate_id in config.candidate_order:
        scores[candidate_id] = _score_candidate(
            candidate_id=candidate_id,
            byteclose=byteclose[candidate_id],
            config=config,
            root=root,
            labels=labels,
            poses=poses,
            segnet=segnet,
            posenet=posenet,
        )
    control = scores["control"]
    rows = [
        _delta_row(
            candidate=scores[candidate_id],
            control=control,
            metadata=metadata[candidate_id],
        )
        for candidate_id in config.candidate_order
        if candidate_id != "control"
    ]
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "run_id": RUN_ID,
        "delegation_checkpoint_key": DELEGATION_KEY,
        "typed_config": config.model_dump(mode="json", by_alias=True),
        "typed_config_sha256": config.stable_hash(),
        "storage_preflight": {
            "checkpoint_root": str(root),
            "minimum_free_bytes": config.minimum_free_bytes,
            "observed_free_bytes": observed_free,
            "status": "PASS",
            "cleanup_policy": (
                "preserve source-identity, runtime-stage, final-raw, and scorer "
                "checkpoints; no uncertified deletion"
            ),
        },
        "source_archives": metadata,
        "source_output_identities": identities,
        "byteclosed_packets": byteclose,
        "measurements": scores,
        "c1_bucket_attribution_rows": rows,
        "scorer_custody": scorer_custody,
        "competitive_target": {
            "official_leaderboard_best_displayed": 0.172,
            "local_0_1910828242_role": "custody_baseline_only",
        },
        "exact_eval": False,
        "pointer_moved": False,
        "paid_dispatch": False,
        "training": False,
        "research_only": True,
        "score_claim": False,
        "evidence_axis": EVIDENCE_AXIS,
        "main_landing_review_required": True,
        "verdict": (
            "CB1_HAS_STRICT_NEGATIVE_JOINT_ROW"
            if any(row["waterfill_eligible"] for row in rows)
            else "CB1_MEASURED_ROWS_NOT_JOINT_POSITIVE"
        ),
    }
    receipt_path = output_directory / "ddm_cb1_perclass_carrier_byteclose_receipt.json"
    _publish_json(receipt_path, receipt)
    print(
        json.dumps(
            {
                "receipt": str(receipt_path),
                "receipt_sha256": _sha256(receipt_path.read_bytes()),
                "verdict": receipt["verdict"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return receipt_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    config_path = (REPO_ROOT / args.config).resolve()
    try:
        config_path.relative_to(REPO_ROOT)
    except ValueError as exc:
        raise CB1Error("config path must stay inside the repository") from exc
    run(config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
