#!/usr/bin/env python
# SPDX-License-Identifier: MIT
"""pe4 PE3 runtime consumption and conditional-transport byte pricing.

This arm is scorer-free.  It proves the PE3EDGE1 optional section is consumed by
the v4d receiver, reproves absent-section identity on qo1 with the extended
receiver, stages the PE3 75KB hybrid as the fourth scorer candidate, and prices
the PE3 transport subset through the same Brotli/LZMA1/SMEVR coder race used by
PE1/PE3.  It does not run SegNet, PoseNet, or upstream/evaluate.py.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import brotli
import numpy as np

REPO: Final = Path(__file__).resolve().parents[1]
for _path in (REPO / "src", REPO / "experiments"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import ddm_bd1_class_field_receiver as bd1
import ddm_pe1_per_edge_partition_race as pe1

BASE_SUB: Final = Path("/Volumes/VertigoDataTier/pact/ddm_qo1_20260804/sub_auto_pairbit")
PE3_SOURCE_SUB: Final = Path(
    "/Volumes/VertigoDataTier/pact/ddm_pe3_20260805/pe3_20260805T000000Z/"
    "sub_auto_pairbit_pe3_hybrid_75kb"
)
DEFAULT_RESEARCH_DIR: Final = REPO / ".omx/research/ddm_pe4_20260805"
DEFAULT_SSD_DIR: Final = Path("/Volumes/VertigoDataTier/pact/ddm_pe4_20260805")
DEFAULT_GT_CACHE: Final = pe1.DEFAULT_GT_CACHE
DEFAULT_CURRENT_ARGMAX: Final = pe1.DEFAULT_CURRENT_ARGMAX
DEFAULT_PE3_RECEIPT: Final = REPO / ".omx/research/ddm_pe3_20260805/ddm_pe3_hybrid_receipt.json"
BASE_RAW_BYTES: Final = 3_662_409_600
BASE_RAW_SHA256: Final = "3ce7d269a7080a4024a576694cd0ddc697099c64cd02fdd2bb879339e4b03f31"
BASELINE_S: Final = 0.7539807296911207
BASELINE_BYTES: Final = 357_836
BASELINE_AXIS: Final = "[macOS-CPU advisory]"
PE3_ARCHIVE_BYTES: Final = 432_428
PE3_ARCHIVE_SHA256: Final = "3f08c7fdd1c2746fa456ef8b6d8005e850d1a3acac5665a5d08b2ef17585b5e0"
PE3_SECTION_SHA256: Final = "5cc024ad32df7fedb18afb75dbed6be9c1af948dac826a1736cb1084949855c2"
PE3_SECTION_BYTES: Final = 74_408
TRANSPORT_MAX_DISTANCE_PX: Final = 24.0


class PE4Error(ValueError):
    """PE4 failed a receiver, staging, or pricing invariant."""


@dataclass(frozen=True, slots=True)
class CoderRow:
    codec: str
    bytes: int
    sha256: str
    artifact_path: str


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, CoderRow):
        return {
            "codec": value.codec,
            "bytes": value.bytes,
            "sha256": value.sha256,
            "artifact_path": value.artifact_path,
        }
    if isinstance(value, pe1.CoderResult):
        return {
            "codec": value.codec,
            "bytes": value.bytes,
            "sha256": value.sha256,
            "artifact_path": value.artifact_path,
        }
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [jsonable(item) for item in value]
    return value


def storage_snapshot(path: Path, required_free_bytes: int) -> dict[str, Any]:
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    ok = int(usage.free) >= int(required_free_bytes)
    return {
        "path": str(path),
        "required_free_bytes": int(required_free_bytes),
        "total_bytes": int(usage.total),
        "used_bytes": int(usage.used),
        "free_bytes": int(usage.free),
        "ok": bool(ok),
    }


def materialize_pe3_receiver_copy(*, ssd_dir: Path, reuse_existing: bool) -> Path:
    out_dir = ssd_dir / "sub_auto_pairbit_pe4_pe3_hybrid_75kb_receiver"
    if out_dir.exists():
        if not reuse_existing:
            raise PE4Error(f"receiver-copy dir already exists: {out_dir}")
        return out_dir
    bd1.copy_runtime_tree(PE3_SOURCE_SUB, out_dir)
    shutil.copy2(PE3_SOURCE_SUB / "archive.zip", out_dir / "archive.zip")
    payload = bd1.read_archive_payload(out_dir / "archive.zip")
    (out_dir / "archive" / "0.bin").write_bytes(payload)
    return out_dir


def import_generated_runner(out_dir: Path) -> Any:
    sys.path.insert(0, str(out_dir))
    try:
        spec = importlib.util.spec_from_file_location(
            "pe4_generated_inflate_runner",
            out_dir / "inflate_runner.py",
        )
        if spec is None or spec.loader is None:
            raise PE4Error(f"could not load generated receiver from {out_dir}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        try:
            sys.path.remove(str(out_dir))
        except ValueError:
            pass


def receiver_copy_ledger(out_dir: Path) -> dict[str, Any]:
    ledger = bd1.build_local_ledger(
        out_dir / "archive.zip",
        joint_names=(
            "config",
            "renderer",
            "selector",
            "pose_warp",
            "frame0_pose_repair",
            "pe3_hybrid_75kb",
        ),
    )
    if ledger["archive_bytes"] != PE3_ARCHIVE_BYTES:
        raise PE4Error("PE3 receiver-copy archive byte count differs")
    if ledger["archive_sha256"] != PE3_ARCHIVE_SHA256:
        raise PE4Error("PE3 receiver-copy archive SHA differs")
    section = ledger["joint_sections"][-1]
    if section["sha256"] != PE3_SECTION_SHA256 or section["raw_bytes"] != PE3_SECTION_BYTES:
        raise PE4Error("PE3 optional section identity differs")
    return ledger


def pe3_runtime_smoke(out_dir: Path) -> dict[str, Any]:
    module = import_generated_runner(out_dir)
    decoder = module.Decoder(out_dir / "archive")
    field = decoder._pe3_edge_field
    if field is None:
        raise PE4Error("PE3 field did not parse into the receiver")
    if decoder._pe1_edge_field is not None:
        raise PE4Error("PE3 candidate unexpectedly also parsed a PE1 field")
    pair_index = next((i for i, count in enumerate(field["pair_counts"]) if int(count) > 0), None)
    if pair_index is None:
        raise PE4Error("PE3 field has no painted pairs")
    with_field = decoder.f1(pair_index)
    decoder._pe3_edge_field = None
    without_field = decoder.f1(pair_index)
    changed = np.any(with_field != without_field, axis=2)
    decoder2 = module.Decoder(out_dir / "archive")
    raster_hash_1 = field["raster_sha256"]
    raster_hash_2 = decoder2._pe3_edge_field["raster_sha256"]
    return {
        "kind": field["kind_name"],
        "mode_counts": field["mode_counts"],
        "section_bytes": field["section_bytes"],
        "section_sha256": field["section_sha256"],
        "raw_bytes": field["raw_bytes"],
        "raw_sha256": field["raw_sha256"],
        "component_records": field["component_records"],
        "painted_pairs": int(sum(1 for count in field["pair_counts"] if int(count) > 0)),
        "painted_pixels_total": int(sum(field["pair_counts"])),
        "smoke_pair": int(pair_index),
        "smoke_pair_painted_pixels": int(field["pair_counts"][pair_index]),
        "camera_pixels_changed": int(changed.sum()),
        "frame1_without_field_sha256": bd1.sha256_bytes(without_field.tobytes()),
        "frame1_with_field_sha256": bd1.sha256_bytes(with_field.tobytes()),
        "mutated": bool(np.any(changed)),
        "deterministic_raster_hash_first": raster_hash_1,
        "deterministic_raster_hash_second": raster_hash_2,
        "deterministic_raster_match": raster_hash_1 == raster_hash_2,
    }


def run_identity_decode(*, identity_dir: Path, reuse_existing: bool) -> dict[str, Any]:
    raw_path = identity_dir / "inflated" / "0.raw"
    if identity_dir.exists():
        if not reuse_existing:
            raise PE4Error(f"identity proof dir already exists: {identity_dir}")
        if not raw_path.exists():
            raise PE4Error(f"identity proof dir exists without raw output: {identity_dir}")
        raw_bytes = raw_path.stat().st_size
        raw_sha = sha256_file(raw_path)
        return {
            "reused_existing": True,
            "command": None,
            "output_raw": str(raw_path),
            "raw_bytes": raw_bytes,
            "raw_sha256": raw_sha,
            "expected_raw_bytes": BASE_RAW_BYTES,
            "expected_raw_sha256": BASE_RAW_SHA256,
            "byte_identical_to_qo1_shipped_decode": raw_bytes == BASE_RAW_BYTES
            and raw_sha == BASE_RAW_SHA256,
            "wall_seconds": None,
        }
    return bd1.run_identity_decode(
        base_sub=BASE_SUB,
        identity_dir=identity_dir,
        expected_raw_sha256=BASE_RAW_SHA256,
        expected_raw_bytes=BASE_RAW_BYTES,
    )


def race_coders(
    *,
    label: str,
    raw: bytes,
    records: tuple[bytes, ...],
    artifact_dir: Path,
) -> tuple[CoderRow, ...]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    encoded = {
        "brotli-q11": bytes(brotli.compress(raw, quality=11)),
        "lzma1-raw": pe1.lzma1_raw(raw),
        "smevr-r7-nibble": bd1.smevr_records(list(records)),
    }
    if brotli.decompress(encoded["brotli-q11"]) != raw:
        raise PE4Error(f"{label}: Brotli roundtrip failed")
    if pe1.unlzma1_raw(encoded["lzma1-raw"], len(raw)) != raw:
        raise PE4Error(f"{label}: LZMA1 roundtrip failed")
    if tuple(bd1.unsmevr_records(encoded["smevr-r7-nibble"])) != records:
        raise PE4Error(f"{label}: SMEVR record roundtrip failed")
    rows: list[CoderRow] = []
    for codec, payload in sorted(encoded.items(), key=lambda item: len(item[1])):
        path = artifact_dir / f"{label}.{codec}.bin"
        path.write_bytes(payload)
        rows.append(
            CoderRow(
                codec=codec,
                bytes=len(payload),
                sha256=sha256_bytes(payload),
                artifact_path=str(path),
            )
        )
    return tuple(rows)


def generator_fields(params: pe1.GeneratorParams) -> tuple[int, ...]:
    y0, x0, y1, x1 = params.bbox
    return (
        int(y0),
        int(x0),
        int(y1 - y0),
        int(x1 - x0),
        int(params.gen_a_q4[0]),
        int(params.gen_a_q4[1]),
        int(params.gen_b_q4[0]),
        int(params.gen_b_q4[1]),
    )


def transport_record(
    *,
    params: pe1.GeneratorParams,
    track_id: int,
    previous: tuple[int, ...] | None,
) -> bytes:
    fields = generator_fields(params)
    record = bytearray([params.edge[0], params.edge[1]])
    record += pe1.varint(track_id)
    if previous is None:
        record.append(0)
        for value in fields:
            record += pe1.write_zigzag(value)
    else:
        record.append(1)
        for value, prev in zip(fields, previous, strict=True):
            record += pe1.write_zigzag(value - prev)
    return bytes(record)


def load_component_state(
    *,
    gt_cache: Path,
    current_argmax: Path,
) -> tuple[list[pe1.Component], dict[int, pe1.GeneratorParams]]:
    lstars = pe1.open_stored_npy_memmap(gt_cache, "lstars")
    current = pe1.load_current_argmax(current_argmax)
    components, _extraction = pe1.extract_components(lstars, current)
    all_ids = frozenset(comp.uid for comp in components)
    _rep, params_by_uid = pe1.build_generator_representation(
        components=components,
        lstars=lstars,
        selected_ids=all_ids,
    )
    return components, params_by_uid


def build_conditional_frame_records(
    *,
    components: list[pe1.Component],
    params_by_uid: dict[int, pe1.GeneratorParams],
    tracks: dict[int, int],
    winning_tracks: set[int],
) -> tuple[bytes, ...]:
    by_pair: list[list[pe1.Component]] = [[] for _ in range(pe1.N_PAIRS)]
    for comp in components:
        by_pair[comp.pair].append(comp)
    previous_by_track: dict[int, tuple[int, ...]] = {}
    frame_records: list[bytes] = []
    for comps in by_pair:
        out = bytearray(pe1.varint(len(comps)))
        for comp in comps:
            params = params_by_uid[comp.uid]
            track_id = tracks[comp.uid]
            if track_id in winning_tracks:
                previous = previous_by_track.get(track_id)
                record = transport_record(params=params, track_id=track_id, previous=previous)
                previous_by_track[track_id] = generator_fields(params)
            else:
                record = pe1.encode_generator_params(params)
            out += pe1.varint(len(record)) + record
        frame_records.append(bytes(out))
    return tuple(frame_records)


def conditional_transport_pricing(
    *,
    gt_cache: Path,
    current_argmax: Path,
    artifact_dir: Path,
) -> dict[str, Any]:
    t0 = time.time()
    components, params_by_uid = load_component_state(
        gt_cache=gt_cache,
        current_argmax=current_argmax,
    )
    all_ids = frozenset(comp.uid for comp in components)
    tracks = pe1.track_generator_components(
        components,
        params_by_uid,
        max_distance_px=TRANSPORT_MAX_DISTANCE_PX,
    )
    independent_rep = pe1.build_generator_representation_from_params(
        components=components,
        params_by_uid=params_by_uid,
        selected_ids=all_ids,
        surface_id="pe4_independent_generator_all_tracks",
    )
    all_transport_rep = pe1.build_generator_transport_representation(
        components=components,
        params_by_uid=params_by_uid,
        selected_ids=all_ids,
        max_distance_px=TRANSPORT_MAX_DISTANCE_PX,
    )

    by_pair: list[list[pe1.Component]] = [[] for _ in range(pe1.N_PAIRS)]
    for comp in components:
        by_pair[comp.pair].append(comp)
    per_track: dict[int, dict[str, Any]] = {}
    previous_by_track: dict[int, tuple[int, ...]] = {}
    for comps in by_pair:
        for comp in comps:
            params = params_by_uid[comp.uid]
            fields = generator_fields(params)
            track_id = tracks[comp.uid]
            gen_record = pe1.encode_generator_params(params)
            independent_len = len(pe1.varint(len(gen_record))) + len(gen_record)
            tr_record = transport_record(
                params=params,
                track_id=track_id,
                previous=previous_by_track.get(track_id),
            )
            previous_by_track[track_id] = fields
            transport_len = len(pe1.varint(len(tr_record))) + len(tr_record)
            row = per_track.setdefault(
                track_id,
                {
                    "track_id": int(track_id),
                    "components": 0,
                    "independent_record_bytes": 0,
                    "transport_record_bytes": 0,
                    "flip_mass": 0,
                },
            )
            row["components"] += 1
            row["independent_record_bytes"] += independent_len
            row["transport_record_bytes"] += transport_len
            row["flip_mass"] += int(comp.flip_mass)
    for row in per_track.values():
        row["transport_won"] = row["transport_record_bytes"] < row["independent_record_bytes"]

    sorted_track_ids = sorted(per_track)
    winning_tracks = {tid for tid, row in per_track.items() if row["transport_won"]}
    selector = np.asarray([tid in winning_tracks for tid in sorted_track_ids], dtype=np.uint8)
    selector_payload = np.packbits(selector, bitorder="big").tobytes()
    conditional_frames = build_conditional_frame_records(
        components=components,
        params_by_uid=params_by_uid,
        tracks=tracks,
        winning_tracks=winning_tracks,
    )
    conditional_raw = selector_payload + b"".join(conditional_frames)
    conditional_records = (selector_payload, *conditional_frames)

    independent_coders = race_coders(
        label="pe4_transport_independent_generator",
        raw=independent_rep.raw,
        records=independent_rep.frame_records,
        artifact_dir=artifact_dir,
    )
    all_transport_coders = race_coders(
        label="pe4_transport_all_tracks_transport",
        raw=all_transport_rep.raw,
        records=all_transport_rep.frame_records,
        artifact_dir=artifact_dir,
    )
    conditional_coders = race_coders(
        label="pe4_transport_subset_conditional",
        raw=conditional_raw,
        records=conditional_records,
        artifact_dir=artifact_dir,
    )
    best_independent = min(independent_coders, key=lambda row: row.bytes)
    best_conditional = min(conditional_coders, key=lambda row: row.bytes)
    best_all_transport = min(all_transport_coders, key=lambda row: row.bytes)
    transport_winners = [row for row in per_track.values() if row["transport_won"]]
    static_receipt_projection = {
        "tracks": len(per_track),
        "transport_won_tracks": len(transport_winners),
        "aggregate_independent_record_bytes": sum(
            row["independent_record_bytes"] for row in per_track.values()
        ),
        "aggregate_transport_record_bytes": sum(
            row["transport_record_bytes"] for row in per_track.values()
        ),
        "selector_bytes": len(selector_payload),
        "conditional_record_bytes_with_1bit_selector": len(selector_payload)
        + sum(
            row["transport_record_bytes"] if row["transport_won"] else row["independent_record_bytes"]
            for row in per_track.values()
        ),
    }
    return {
        "schema": "ddm_pe4_conditional_transport_pricing.v1",
        "axis": "[macOS-CPU advisory / scorer-free real-coder byte pricing]",
        "selection_mode": "n600 all PE1 generator tracks; no prefix; scorer-free",
        "track_count": len(per_track),
        "component_count": len(components),
        "winning_tracks": len(transport_winners),
        "winning_track_fraction": len(transport_winners) / len(per_track) if per_track else 0.0,
        "raw_record_model": static_receipt_projection,
        "selector": {
            "coding": "one packed bit per sorted track id; prepended as a standalone coder record",
            "bytes": len(selector_payload),
            "sha256": sha256_bytes(selector_payload),
        },
        "raw_streams": {
            "independent_bytes": len(independent_rep.raw),
            "all_transport_bytes": len(all_transport_rep.raw),
            "conditional_bytes_including_selector": len(conditional_raw),
            "conditional_delta_vs_independent_raw": len(conditional_raw) - len(independent_rep.raw),
            "conditional_sha256": sha256_bytes(conditional_raw),
        },
        "coder_race": {
            "independent": list(independent_coders),
            "all_transport": list(all_transport_coders),
            "subset_conditional": list(conditional_coders),
        },
        "best": {
            "independent": best_independent,
            "all_transport": best_all_transport,
            "subset_conditional": best_conditional,
            "conditional_net_delta_bytes_vs_independent": best_conditional.bytes
            - best_independent.bytes,
            "conditional_wins": best_conditional.bytes < best_independent.bytes,
        },
        "sample_winning_tracks": sorted(
            transport_winners,
            key=lambda row: (-row["flip_mass"], row["track_id"]),
        )[:10],
        "verdict_scope": (
            "FORMULATION-scoped to PE1 local generator-parameter tracks with a packed "
            "per-track selector and the real Brotli/LZMA1/SMEVR coder race; no scorer "
            "claim and no PE3-r2 archive promotion unless the encoded subset wins."
        ),
        "wall_seconds": round(time.time() - t0, 1),
    }


def write_stage_script(*, path: Path, scorer_out_dir: Path, receiver_copy: Path) -> None:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        f"cd {REPO}",
        'DEVICE="${1:-cpu}"',
        'BATCH_SIZE="${BATCH_SIZE:-16}"',
        'NUM_THREADS="${NUM_THREADS:-4}"',
        f'OUT_ROOT="{scorer_out_dir}"',
        'mkdir -p "$OUT_ROOT"',
        'echo "[pe4] fourth scorer candidate; run only after MAIN harvests the active PE2 batch" >&2',
        'echo "[pe4] scoring pe3_hybrid_75kb_receiver on $DEVICE" >&2',
        ".venv/bin/python experiments/ddm_fz2_byteclose_and_eval.py "
        f"--sub-dir {receiver_copy} "
        '--out "$OUT_ROOT/pe3_hybrid_75kb_n600_${DEVICE}.json" '
        '--inflate-out "$OUT_ROOT/pe3_hybrid_75kb_inflate_${DEVICE}" '
        '--device "$DEVICE" '
        '--batch-size "$BATCH_SIZE" '
        '--num-threads "$NUM_THREADS"',
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    path.chmod(0o755)


def write_queue_note(path: Path, receipt: dict[str, Any]) -> None:
    script = receipt["staged_scorer_job"]["script"]
    candidate = receipt["pe3_candidate"]
    lines = [
        "# PE4 Fourth-Candidate Scorer Queue Note",
        "",
        "Status: **QUEUED-WITH-FIRE-ORDER / NOT RUN BY PE4**.",
        "",
        "Fire condition: MAIN harvests the active PE2 three-candidate batch, confirms the single scorer slot is free, then claims one follow-on scorer job for the PE3 75KB hybrid.",
        "",
        f"Exact fire command from repo root: `bash {script} cpu`",
        "",
        "Axis warning: running the command on this Mac is `[macOS-CPU advisory]`; contest authority still requires the contest-CPU or contest-CUDA host.",
        "",
        "| candidate | receiver-copy archive bytes | receiver-copy archive sha256 | submission dir |",
        "|---|---:|---|---|",
        (
            f"| PE3 hybrid 75KB | `{candidate['archive_bytes']}` | "
            f"`{candidate['archive_sha256']}` | `{candidate['receiver_copy']}` |"
        ),
        "",
        "Batch contract:",
        "",
        "- This is the fourth candidate after PE2's PE1 full, PE1 surgical, and BF1 batch.",
        "- The archive bytes are the PE3 75KB hybrid archive; the receiver copy uses the PE4 PE3EDGE1-consuming runtime.",
        "- The staged script uses the canonical byteclose/evaluate wrapper per exact archive.",
        "- No scorer was run while PE4 generated this note.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_candidate_spec_if_win(path: Path, pricing: dict[str, Any]) -> str:
    if not pricing["best"]["conditional_wins"]:
        return "FOLDED"
    best = pricing["best"]["subset_conditional"]
    lines = [
        "# PE3-r2 conditional-transport candidate spec",
        "",
        "Status: **BYTE-SPEC-STAGED / SCORER-FREE / RECEIVER-UNIMPLEMENTED**.",
        "",
        "PE4 measured the subset-conditional transport stream as a real-coder byte win over independent generator coding. This is a PE3-r2 spec only; it is not a scored candidate until a receiver section is implemented and byte-closed.",
        "",
        f"- Best conditional codec: `{best['codec']}`.",
        f"- Best conditional bytes: `{best['bytes']}`.",
        f"- Net bytes vs independent: `{pricing['best']['conditional_net_delta_bytes_vs_independent']}`.",
        f"- Selector bytes: `{pricing['selector']['bytes']}`.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return "QUEUED-WITH-FIRE-ORDER"


def write_markdown_receipt(path: Path, receipt: dict[str, Any]) -> None:
    identity = receipt["old_archive_identity_proof"]
    runtime = receipt["pe3_candidate"]["runtime_consumption"]
    pricing = receipt["conditional_transport_pricing"]
    lines = [
        "# PE4 PE3 runtime consumption + conditional transport pricing - 2026-08-05",
        "",
        "Status: **RECEIVER-CLOSED / ABSENT-IDENTITY-PROVED / FOURTH-CANDIDATE-STAGED / CONDITIONAL-TRANSPORT-PRICED / SCORER-FREE**.",
        "",
        "Axis: `[macOS-CPU advisory / scorer-free receiver-byte custody]`.",
        "`score_claim=false`, `promotion_eligible=false`, `n600_scorer_job=false`.",
        "",
        "## Runtime Consumption",
        "",
        f"- PE3 archive bytes: `{receipt['pe3_candidate']['archive_bytes']}`; sha256 `{receipt['pe3_candidate']['archive_sha256']}`.",
        f"- PE3 section bytes: `{runtime['section_bytes']}`; raw bytes `{runtime['raw_bytes']}`; modes `{runtime['mode_counts']}`.",
        f"- Smoke pair `{runtime['smoke_pair']}` changed `{runtime['camera_pixels_changed']}` camera pixels from PE3 paint.",
        f"- Deterministic raster hash matched on reparse: `{runtime['deterministic_raster_match']}`.",
        "",
        "## Absent-Section Identity",
        "",
        f"- qo1 raw bytes: `{identity['raw_bytes']}`; expected `{identity['expected_raw_bytes']}`.",
        f"- qo1 raw sha256: `{identity['raw_sha256']}`.",
        f"- Byte-identical to the shipped qo1 decode: `{identity['byte_identical_to_qo1_shipped_decode']}`.",
        "",
        "## Conditional Transport Pricing",
        "",
        "| stream | best codec | best bytes | raw bytes |",
        "|---|---|---:|---:|",
        (
            f"| independent generator | `{pricing['best']['independent']['codec']}` | "
            f"`{pricing['best']['independent']['bytes']}` | `{pricing['raw_streams']['independent_bytes']}` |"
        ),
        (
            f"| all transport | `{pricing['best']['all_transport']['codec']}` | "
            f"`{pricing['best']['all_transport']['bytes']}` | `{pricing['raw_streams']['all_transport_bytes']}` |"
        ),
        (
            f"| subset conditional | `{pricing['best']['subset_conditional']['codec']}` | "
            f"`{pricing['best']['subset_conditional']['bytes']}` | "
            f"`{pricing['raw_streams']['conditional_bytes_including_selector']}` |"
        ),
        "",
        f"- Winning tracks: `{pricing['winning_tracks']}/{pricing['track_count']}`.",
        f"- Packed selector bytes: `{pricing['selector']['bytes']}`.",
        f"- Net encoded bytes vs independent: `{pricing['best']['conditional_net_delta_bytes_vs_independent']}`.",
        f"- Conditional transport verdict: `{'WIN' if pricing['best']['conditional_wins'] else 'NEGATIVE'}`.",
        f"- verdict_scope: {pricing['verdict_scope']}",
        "",
        "## Staged Scorer Job",
        "",
        f"- Script: `{receipt['staged_scorer_job']['script']}`.",
        f"- Queue note: `{receipt['staged_scorer_job']['queue_note']}`.",
        f"- Manifest: `{receipt['staged_scorer_job']['manifest']}`.",
        "- PE4 did not run SegNet, PoseNet, or `upstream/evaluate.py`.",
        "",
        "## RECALL EVIDENCE",
        "",
    ]
    for item in receipt["recall_evidence"]:
        lines.append(f"- `{item['source']}`: {item['finding']} Plan impact: {item['plan_impact']}")
    lines.extend(["", "## Follow-On Disposition", ""])
    for item in receipt["follow_on_disposition"]:
        lines.append(f"- {item['status']}: {item['action']}")
    lines.extend(
        [
            "",
            "## NEXT-IF-RESUMED",
            "",
            receipt["next_if_resumed"],
            "",
            f"Own-vehicle frontier line: `S = {BASELINE_S} @ {BASELINE_BYTES:,} B {BASELINE_AXIS}`; PE4 did not run a scorer and did not move the contest pointer.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-dir", type=Path, default=DEFAULT_RESEARCH_DIR)
    parser.add_argument("--ssd-dir", type=Path, default=DEFAULT_SSD_DIR)
    parser.add_argument("--gt-cache", type=Path, default=DEFAULT_GT_CACHE)
    parser.add_argument("--current-argmax", type=Path, default=DEFAULT_CURRENT_ARGMAX)
    parser.add_argument("--pe3-receipt", type=Path, default=DEFAULT_PE3_RECEIPT)
    parser.add_argument("--reuse-existing", action="store_true")
    parser.add_argument("--skip-identity-decode", action="store_true")
    args = parser.parse_args(argv)

    args.research_dir.mkdir(parents=True, exist_ok=True)
    args.ssd_dir.mkdir(parents=True, exist_ok=True)
    storage = storage_snapshot(args.ssd_dir, required_free_bytes=BASE_RAW_BYTES + 1_000_000_000)
    if not storage["ok"]:
        raise PE4Error("SSD tier lacks enough free space for the absent-identity raw proof")

    receiver_copy = materialize_pe3_receiver_copy(ssd_dir=args.ssd_dir, reuse_existing=args.reuse_existing)
    ledger = receiver_copy_ledger(receiver_copy)
    runtime = pe3_runtime_smoke(receiver_copy)
    if not runtime["mutated"] or not runtime["deterministic_raster_match"]:
        raise PE4Error("PE3 runtime consumption proof failed")

    if args.skip_identity_decode:
        identity = {
            "skipped": True,
            "byte_identical_to_qo1_shipped_decode": False,
            "raw_bytes": None,
            "raw_sha256": None,
            "expected_raw_bytes": BASE_RAW_BYTES,
            "expected_raw_sha256": BASE_RAW_SHA256,
        }
    else:
        identity = run_identity_decode(
            identity_dir=args.ssd_dir / "qo1_identity_pe4_extended_receiver",
            reuse_existing=args.reuse_existing,
        )
        if not identity["byte_identical_to_qo1_shipped_decode"]:
            raise PE4Error("qo1 absent-section identity proof failed")

    pricing = jsonable(
        conditional_transport_pricing(
            gt_cache=args.gt_cache,
            current_argmax=args.current_argmax,
            artifact_dir=args.ssd_dir / "conditional_transport_payloads",
        )
    )

    scorer_out_dir = args.ssd_dir / "scorer_batch"
    stage_script = args.research_dir / "stage_pe4_fourth_candidate_scorer_batch.sh"
    queue_note = args.research_dir / "PE4_QUEUE_NOTE.md"
    manifest_path = args.research_dir / "pe4_fourth_candidate_scorer_manifest.json"
    write_stage_script(path=stage_script, scorer_out_dir=scorer_out_dir, receiver_copy=receiver_copy)

    pe3r2_spec = args.research_dir / "PE3_R2_CANDIDATE_SPEC_QUEUED.md"
    pe3r2_status = write_candidate_spec_if_win(pe3r2_spec, pricing)
    if pe3r2_status == "FOLDED" and pe3r2_spec.exists():
        pe3r2_spec.unlink()

    manifest = {
        "schema": "ddm_pe4_fourth_candidate_scorer_manifest.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "score_claim": False,
        "n600_scorer_job": "staged_not_run",
        "single_scorer_slot_owner_to_claim": "MAIN after PE2 batch done-receipt",
        "candidate": {
            "label": "PE3 hybrid 75KB",
            "receiver_copy": str(receiver_copy),
            "archive_bytes": ledger["archive_bytes"],
            "archive_sha256": ledger["archive_sha256"],
            "source_sub": str(PE3_SOURCE_SUB),
        },
        "stage_script": str(stage_script),
    }
    manifest_path.write_text(json.dumps(jsonable(manifest), indent=1, sort_keys=True), encoding="utf-8")

    receipt = {
        "schema": "ddm_pe4_runtime_transport_receipt.v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "axis": "[macOS-CPU advisory / scorer-free receiver-byte custody]",
        "score_claim": False,
        "promotion_eligible": False,
        "n600_scorer_job": False,
        "storage_preflight": storage,
        "old_archive_identity_proof": identity,
        "pe3_candidate": {
            "label": "PE3 hybrid 75KB",
            "source_sub": str(PE3_SOURCE_SUB),
            "receiver_copy": str(receiver_copy),
            "archive_bytes": ledger["archive_bytes"],
            "archive_sha256": ledger["archive_sha256"],
            "ledger": ledger,
            "runtime_consumption": runtime,
        },
        "conditional_transport_pricing": pricing,
        "staged_scorer_job": {
            "status": "QUEUED-WITH-FIRE-ORDER",
            "script": str(stage_script),
            "queue_note": str(queue_note),
            "manifest": str(manifest_path),
            "scorer_out_dir": str(scorer_out_dir),
            "run_by_pe4": False,
        },
        "recall_evidence": [
            {
                "source": ".omx/tmp/codex_runs/_common_contract.md",
                "finding": "PE4 is scorer-free because PE2 owns the active scorer slot; follow-ons must be queued with a fire order.",
                "plan_impact": "the fourth candidate is staged only, and no scorer process is launched.",
            },
            {
                "source": ".omx/research/operator_directive_per_edge_optimality_criteria_20260805.md",
                "finding": "addendum 3 requires hybrid/per-level and conditional-per-axis treatment, while addendum 4 keeps seg-first ordering.",
                "plan_impact": "PE4 consumes the PE3 hybrid section and prices transport as a conditional subset instead of promoting aggregate transport.",
            },
            {
                "source": ".omx/research/ddm_pe2_20260805/PE2_RECEIPT_20260805.md",
                "finding": "PE2's receiver proof pattern is materialize receiver copy, prove absent identity, smoke optional section mutation, stage scorer script.",
                "plan_impact": "PE4 mirrors that receiver-byte proof for PE3EDGE1 and stages a separate fourth-candidate script.",
            },
            {
                "source": ".omx/research/ddm_pe3_20260805/PE3_RECEIPT_20260805.md and ddm_pe3_hybrid_receipt.json",
                "finding": "PE3 75KB archive is byte-closed and parse-back-only; conditional transport headline was record-byte pre-entropy, not archive-byte.",
                "plan_impact": "PE4 verifies runtime RGB consumption and re-prices transport through real coders before any PE3-r2 spec.",
            },
            {
                "source": "content searches: PE3EDGE1, conditional_transport, selector_bytes, SMEVR, #940 over .omx/research + experiments/src/tools",
                "finding": "no prior current PE3EDGE1 runtime consumer or encoded subset-conditional transport receipt was found in the searched scope.",
                "plan_impact": "PE4 implements the missing receiver consumer and records a scoped negative/win from fresh coder outputs.",
            },
        ],
        "follow_on_disposition": [
            {
                "status": "FIRED",
                "action": "PE3EDGE1 runtime consumption is implemented in the v4d receiver and proved on the PE3 75KB candidate copy.",
            },
            {
                "status": "FIRED",
                "action": "qo1 absent-section identity is re-proved under the PE3-capable receiver.",
            },
            {
                "status": "QUEUED-WITH-FIRE-ORDER",
                "action": "MAIN runs the PE4 fourth-candidate scorer script only after the active PE2 batch is harvested and the scorer slot is free.",
            },
            {
                "status": pe3r2_status,
                "action": (
                    "PE3-r2 conditional-transport spec is staged only if the encoded subset conditional stream beats independent coding; otherwise the scoped negative is the terminal disposition."
                ),
            },
        ],
        "next_if_resumed": (
            "Start from PE4_RECEIPT_20260805.md and ddm_pe4_runtime_transport_receipt.json. "
            "Do not run a scorer until MAIN harvests the PE2 batch and claims the slot. "
            f"If the slot is free, run `bash {stage_script} cpu` for a local advisory row or the same script on authority hardware. "
            "If conditional transport won, implement the queued PE3-r2 receiver section before staging any scorer row."
        ),
        "own_vehicle_frontier": {
            "S": BASELINE_S,
            "archive_bytes": BASELINE_BYTES,
            "axis": BASELINE_AXIS,
            "pointer_moved": False,
        },
    }

    receipt_json = args.research_dir / "ddm_pe4_runtime_transport_receipt.json"
    receipt_md = args.research_dir / "PE4_RECEIPT_20260805.md"
    receipt_json.write_text(json.dumps(jsonable(receipt), indent=1, sort_keys=True), encoding="utf-8")
    write_queue_note(queue_note, receipt)
    write_markdown_receipt(receipt_md, receipt)
    print(
        json.dumps(
            {
                "receipt": str(receipt_json),
                "markdown": str(receipt_md),
                "receiver_copy": str(receiver_copy),
                "conditional_net_delta_bytes_vs_independent": pricing["best"][
                    "conditional_net_delta_bytes_vs_independent"
                ],
                "conditional_wins": pricing["best"]["conditional_wins"],
                "stage_script": str(stage_script),
            },
            indent=1,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
