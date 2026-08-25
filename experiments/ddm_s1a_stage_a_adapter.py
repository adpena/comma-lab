#!/usr/bin/env python3
"""Build and seal the S1 Stage-A adapters without launching training or scorers.

The module binds WD3's trained Film-W96 renderer to the exact GB1 container.  It
also re-proves RJ2's compensation and production carrier encoder on that body,
materializes two seed-specific scorer-free births from one verified initializer,
and emits the typed MAIN-owned order.  The only mutable archive section in a
Stage-A training candidate is the renderer/semantic section.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import struct
import subprocess
import sys
import zipfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch

REPO: Final = Path(__file__).resolve().parents[1]
SRC: Final = REPO / "src"
for _root in (REPO, SRC):
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from experiments import ddm_po1_t4_error_feedback_pose_compensation as po1
from experiments import ddm_rj1_renderer_joint_move as rj1
from experiments import ddm_rj2_joint_renderer_object_change as rj2
from experiments import ddm_wd3_scorer_aware_width_distillation as wd3
from experiments import ddm_wd3_student_receiver as receiver

OUTPUT_ROOT: Final = Path("/Volumes/APDataStore/pact/ddm_s1a_stage_a_adapter")
GB1_ARCHIVE: Final = Path(
    "/Volumes/APDataStore/pact/ddm_gb1_groupbin8_conditioning/retained/candidate_gb1_groupbin8_surprise.zip"
)
GB1_RUNTIME: Final = Path("/Volumes/APDataStore/pact/ddm_gb1_groupbin8_conditioning/runtime_fire_v1")
GB1_ARCHIVE_BYTES: Final = 180_215
GB1_ARCHIVE_SHA256: Final = "ba1f3830cd51b820d7f9b834a1dcc12e8776a0260f9da57a4e8e0944b988e3a4"
RJ1_ROOT: Final = Path("/Volumes/VertigoDataTier/pact/ddm_rj1_renderer_joint_move/precompile_r1")
RJ1_INVENTORY: Final = RJ1_ROOT / "CUSTODY_INVENTORY.json"
RJ1_ERRATA: Final = RJ1_ROOT / "CUSTODY_INVENTORY_ERRATA.json"
RJ1_INITIALIZER: Final = RJ1_ROOT / "rungs/film_amortized_flat_w96/renderer_initialization.pt"
RJ1_INITIALIZER_BYTES: Final = 253_955
RJ1_INITIALIZER_SHA256: Final = "e74ba046af251808ef105cf0a2295f6133efa194360148f3110762765b9db434"
RJ2_REPLAY: Final = Path("/Volumes/APDataStore/pact/ddm_rj2_joint_renderer_object_change/reviewed_replay_r1")
WD3_CACHE_RESULT: Final = Path(
    "/Volumes/APDataStore/pact/ddm_wd3_scorer_aware_width_distillation/"
    "teacher_scorer_cache/TEACHER_SCORER_CACHE_RESULT.json"
)
GT_CACHE: Final = REPO / "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
SEEDS: Final = (20260815, 20260816)
EPOCHS: Final = 65
MEASURED_SECONDS_PER_EPOCH: Final = 93.23
FIXED_RESIDUAL_BYTES: Final = 96
RX1_HEADER: Final = struct.Struct("<4sBBBBHHH")
AXIS: Final = "[macOS-CPU scorer-free exact byte/container apparatus]"


class S1AError(RuntimeError):
    """A Stage-A custody, object, receiver, or launch-order invariant failed."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise S1AError(f"required file is absent: {path}")
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def atomic_bytes(path: Path, value: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not path.is_file() or path.read_bytes() != value:
            raise S1AError(f"refusing to overwrite differing retained payload: {path}")
        return file_record(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return file_record(path)


def atomic_json(path: Path, value: object) -> dict[str, Any]:
    return atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())


def atomic_npy(path: Path, value: np.ndarray) -> dict[str, Any]:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(value), allow_pickle=False)
    return atomic_bytes(path, buffer.getvalue())


def canonical_sha256(value: object) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def storage_preflight(output: Path, minimum_free_bytes: int = 8 * 1024**3) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(output)
    if usage.free < minimum_free_bytes:
        raise S1AError(f"APDataStore storage preflight failed: {usage.free} < {minimum_free_bytes}")
    return {
        "schema": "ddm_s1a_storage_preflight.v1",
        "path": str(output.resolve()),
        "free_bytes": usage.free,
        "minimum_free_bytes": minimum_free_bytes,
        "pass": True,
    }


def verify_rj1_custody() -> dict[str, Any]:
    if sha256_file(RJ1_INVENTORY) != "dd3b89b7f9d68f11f3d828457316b748b796aa55fea36304d566dd5cd2f8467c":
        raise S1AError("RJ1 custody inventory SHA differs")
    inventory = json.loads(RJ1_INVENTORY.read_text(encoding="utf-8"))
    errata = json.loads(RJ1_ERRATA.read_text(encoding="utf-8"))
    if inventory.get("schema") != "ddm_rj1_retained_tree.v1" or inventory.get("file_count") != 192:
        raise S1AError("RJ1 inventory schema/denominator differs")
    if errata.get("original_inventory_sha256") != sha256_file(RJ1_INVENTORY):
        raise S1AError("RJ1 errata does not bind the consumed inventory")
    voided = {str(row["path"]) for row in errata.get("voided_records", [])}
    if len(voided) != 3 or any(".___pycache__" not in value for value in voided):
        raise S1AError("RJ1 errata void set differs")
    verified = 0
    payload_bytes = 0
    for row in inventory["files"]:
        relative = str(row["relative_path"])
        if relative in voided:
            continue
        path = Path(row["path"])
        if file_record(path) != {key: row[key] for key in ("path", "bytes", "sha256")}:
            raise S1AError(f"RJ1 inventory payload drifted: {relative}")
        verified += 1
        payload_bytes += int(row["bytes"])
    if verified != 189:
        raise S1AError(f"RJ1 current-payload denominator differs: {verified}/189")
    initializer = file_record(RJ1_INITIALIZER)
    if initializer["bytes"] != RJ1_INITIALIZER_BYTES or initializer["sha256"] != RJ1_INITIALIZER_SHA256:
        raise S1AError("RJ1 Film-W96 initializer differs")
    return {
        "schema": "ddm_s1a_rj1_custody_reproof.v1",
        "status": "PASS",
        "inventory": file_record(RJ1_INVENTORY),
        "errata": file_record(RJ1_ERRATA),
        "verified_current_records_numerator": verified,
        "current_payload_records_denominator": 189,
        "voided_metadata_records": sorted(voided),
        "verified_payload_bytes": payload_bytes,
        "initializer": initializer,
        "source_tree_read_only": True,
    }


def read_archive(path: Path) -> dict[str, Any]:
    archive = path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        if bundle.namelist() != ["p"]:
            raise S1AError("GB1 archive must contain exactly member p")
        info = bundle.getinfo("p")
        member = bundle.read("p")
    if len(member) < RX1_HEADER.size:
        raise S1AError("GB1 member is shorter than its RX1 header")
    magic, version, codec, table_mode, reserved, hpac_bytes, semantic_bytes, carrier_bytes = RX1_HEADER.unpack_from(
        member
    )
    if magic != b"RX1M":
        raise S1AError("GB1 RX1 magic differs")
    offset = RX1_HEADER.size
    sections: dict[str, bytes] = {"header": member[:offset]}
    for name, length in (("hpac", hpac_bytes), ("semantic", semantic_bytes), ("carrier", carrier_bytes)):
        sections[name] = member[offset : offset + length]
        offset += length
    sections["tail"] = member[offset:]
    if len(sections["tail"]) <= FIXED_RESIDUAL_BYTES:
        raise S1AError("GB1 tail lacks the fixed residual plus token stream")
    sections["fixed_residual"] = sections["tail"][:FIXED_RESIDUAL_BYTES]
    sections["token_stream"] = sections["tail"][FIXED_RESIDUAL_BYTES:]
    framing = {
        "magic_hex": magic.hex(),
        "version": version,
        "codec": codec,
        "table_mode": table_mode,
        "reserved": reserved,
        "hpac_bytes": hpac_bytes,
        "carrier_bytes": carrier_bytes,
        "tail_bytes": len(sections["tail"]),
        "zip": {
            "date_time": list(info.date_time),
            "create_system": info.create_system,
            "external_attr": info.external_attr,
            "compress_type": info.compress_type,
            "flag_bits": info.flag_bits,
            "extra_hex": info.extra.hex(),
            "comment_hex": info.comment.hex(),
        },
    }
    if rj1.deterministic_zip(member) != archive:
        raise S1AError("GB1 identity archive rebuild differs")
    return {
        "archive": archive,
        "member": member,
        "magic": magic,
        "version": version,
        "codec": codec,
        "table_mode": table_mode,
        "reserved": reserved,
        "framing": framing,
        **sections,
    }


def section_record(value: bytes) -> dict[str, Any]:
    return {"bytes": len(value), "sha256": sha256_bytes(value)}


def stage_a_binding(custody_receipt: Path) -> dict[str, Any]:
    return {
        "schema": "ddm_s1a_stage_a_binding.v1",
        "enabled": True,
        "base_archive": str(GB1_ARCHIVE.resolve()),
        "base_archive_sha256": GB1_ARCHIVE_SHA256,
        "base_runtime": str(GB1_RUNTIME.resolve()),
        "initializer": str(RJ1_INITIALIZER.resolve()),
        "initializer_sha256": RJ1_INITIALIZER_SHA256,
        "custody_receipt": str(custody_receipt.resolve()),
        "adapter_module": str(Path(__file__).resolve()),
        "adapter_sha256": sha256_file(Path(__file__).resolve()),
        "registered_seeds": list(SEEDS),
        "renderer_only_mutable": True,
        "untouched_sections": ["hpac", "carrier", "fixed_residual", "token_stream", "framing"],
    }


def _patch_stage_a_runtime(runtime: Path, archive: Path) -> dict[str, Any]:
    copy_receipt = rj2.prepare_runtime_copy(GB1_RUNTIME, runtime)
    if not copy_receipt["resumed"] and copy_receipt["archive_present_before_patch"]:
        raise S1AError("candidate runtime copy already carried an archive before binding")
    residual_path = runtime / "runtime/residual_archive.py"
    residual = residual_path.read_text(encoding="utf-8")
    old_residual = 'tagged_semantic = semantic_body.startswith((b"SD1M", b"SM3R", b"WD2S"))'
    new_residual = 'tagged_semantic = semantic_body.startswith((b"SD1M", b"SM3R", b"WD2S", b"WD3Q"))'
    if new_residual in residual and old_residual not in residual:
        runtime_archive = file_record(runtime / "archive.zip")
        candidate_archive = file_record(archive)
        if any(runtime_archive[key] != candidate_archive[key] for key in ("bytes", "sha256")):
            raise S1AError("resumed Stage-A runtime carries a different candidate archive")
        wd2_patch = {"schema": "ddm_rj1_runtime_patch.v1", "resumed_stage_a_patch": True}
    else:
        wd2_patch = rj1.patch_runtime(GB1_RUNTIME, runtime, archive)
        residual = residual_path.read_text(encoding="utf-8")
    if old_residual in residual and new_residual not in residual:
        residual = residual.replace(old_residual, new_residual, 1)
        residual_path.write_text(residual, encoding="utf-8")
    elif residual.count(new_residual) != 1 or old_residual in residual:
        raise S1AError("Stage-A residual WD3 dispatch patch point differs")
    f26_path = runtime / "runtime/f26_inflate.py"
    f26 = f26_path.read_text(encoding="utf-8")
    replacements = (
        (
            '(WANS1_MAGIC, b"SD1M", b"SM3R", b"WD2S")',
            '(WANS1_MAGIC, b"SD1M", b"SM3R", b"WD2S", b"WD3Q")',
        ),
        (
            'if parts.semantic_blob.startswith(b"WD2S"):',
            'if parts.semantic_blob.startswith((b"WD2S", b"WD3Q")):',
        ),
        (
            'receiver_path = renderer_dir / "wd2_receiver.py"',
            'receiver_path = renderer_dir / ("wd3_receiver.py" if parts.semantic_blob.startswith(b"WD3Q") else "wd2_receiver.py")',
        ),
    )
    changed = False
    for old, new in replacements:
        if old in f26 and new not in f26:
            f26 = f26.replace(old, new, 1)
            changed = True
        elif f26.count(new) != 1 or old in f26:
            raise S1AError(f"Stage-A F26 patch point differs: {old[:40]}")
    if changed:
        f26_path.write_text(f26, encoding="utf-8")
    shutil.copy2(Path(receiver.__file__).resolve(), runtime / "cpr1/wd3_receiver.py")
    runtime_archive = file_record(runtime / "archive.zip")
    candidate_archive = file_record(archive)
    if any(runtime_archive[key] != candidate_archive[key] for key in ("bytes", "sha256")):
        raise S1AError("candidate runtime archive binding differs")
    if runtime_archive["sha256"] == GB1_ARCHIVE_SHA256:
        raise S1AError("candidate runtime still carries the source GB1 archive")
    return {
        "copy": copy_receipt,
        "wd2_patch": wd2_patch,
        "wd3_dispatch": True,
        "source_archive_excluded_on_atomic_copy": copy_receipt["source_archive_excluded"],
        "source_archive_absent_after_candidate_bind": True,
        "runtime_archive_equals_candidate": True,
        "candidate_archive": runtime_archive,
    }


_PARSEBACK_PROGRAM: Final = r"""
import hashlib, importlib.util, json, sys
from pathlib import Path
runtime, archive, expected = map(Path, sys.argv[1:])
sys.path.insert(0, str(runtime))
sys.path.insert(0, str(runtime / 'cpr1'))
from runtime.residual_archive import read_residual_archive
parts = read_residual_archive(archive)
receiver_path = runtime / 'cpr1/wd3_receiver.py'
spec = importlib.util.spec_from_file_location('_s1a_wd3_receiver', receiver_path)
module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
packet = expected.read_bytes()
model = module.unpack_student(parts.semantic_blob)
allocation = module.packet_allocation(packet)
repacked = module.pack_student(model, allocation)
report = {
  'semantic_sha256': hashlib.sha256(parts.semantic_blob).hexdigest(),
  'expected_sha256': hashlib.sha256(packet).hexdigest(),
  'packet_exact': parts.semantic_blob == packet,
  'repack_exact': repacked == packet,
  'hpac_sha256': hashlib.sha256(parts.hpac_blob).hexdigest(),
  'carrier_sha256': hashlib.sha256(parts.carrier_blob).hexdigest(),
}
if not report['packet_exact'] or not report['repack_exact']:
    raise SystemExit(json.dumps(report, sort_keys=True))
print(json.dumps(report, sort_keys=True))
"""


def _fresh_parseback(runtime: Path, archive: Path, packet: Path, transcript: Path) -> dict[str, Any]:
    completed = subprocess.run(
        [str(REPO / ".venv/bin/python"), "-c", _PARSEBACK_PROGRAM, str(runtime), str(archive), str(packet)],
        capture_output=True,
        text=True,
        check=False,
        env={"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0"},
    )
    transcript_record = atomic_bytes(transcript, (completed.stdout + "\n--- STDERR ---\n" + completed.stderr).encode())
    if completed.returncode:
        raise S1AError("fresh-process Stage-A receiver parse-back failed")
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    return {"status": "PASS", "report": report, "transcript": transcript_record}


def retain_renderer_candidate(
    *,
    root: Path,
    packet: bytes,
    allocation: Mapping[str, Any],
    binding: Mapping[str, Any],
) -> dict[str, Any]:
    """Replace only GB1's renderer block and prove all untouched bytes."""

    if binding.get("base_archive_sha256") != GB1_ARCHIVE_SHA256:
        raise S1AError("Stage-A packet binding is not the exact GB1 body")
    if sha256_file(Path(binding["base_archive"])) != GB1_ARCHIVE_SHA256:
        raise S1AError("exact GB1 body drifted before candidate construction")
    source = read_archive(Path(binding["base_archive"]))
    semantic_stream, member, archive = rj1.build_archive(source, packet)
    candidate = read_archive_bytes(archive)
    immutable = ("hpac", "carrier", "fixed_residual", "token_stream")
    assertions = {name: source[name] == candidate[name] for name in immutable}
    assertions["framing"] = source["framing"] == candidate["framing"]
    assertions["renderer_is_only_mutable_section"] = source["semantic"] != candidate["semantic"]
    if not all(assertions.values()):
        raise S1AError(f"GB1 untouched-section preservation failed: {assertions}")
    packet_record = atomic_bytes(root / "semantic.wd3q", packet)
    payloads = {
        "student_packet": packet_record,
        "semantic_ck2_brotli_q11": atomic_bytes(root / "semantic.ck2.br", semantic_stream),
        "member": atomic_bytes(root / "p", member),
        "archive": atomic_bytes(root / "archive.zip", archive),
        "archive_repeat": atomic_bytes(root / "archive.repeat.zip", rj1.deterministic_zip(member)),
    }
    runtime = root / "submission"
    runtime_patch = _patch_stage_a_runtime(runtime, Path(payloads["archive"]["path"]))
    parseback = _fresh_parseback(
        runtime,
        Path(payloads["archive"]["path"]),
        Path(packet_record["path"]),
        root / "PARSEBACK_TRANSCRIPT.txt",
    )
    preservation = {
        "schema": "ddm_s1a_gb1_section_preservation.v1",
        "base_archive": file_record(Path(binding["base_archive"])),
        "candidate_archive": payloads["archive"],
        "assertions": assertions,
        "source_sections": {name: section_record(source[name]) for name in (*immutable, "semantic")},
        "candidate_sections": {name: section_record(candidate[name]) for name in (*immutable, "semantic")},
        "framing": source["framing"],
        "semantic_length_field_is_the_only_expected_header_byte_change": True,
    }
    preservation_record = atomic_json(root / "SECTION_PRESERVATION.json", preservation)
    return {
        "schema": "ddm_wd3_retained_packet_archive.v1",
        "body": "gb1_groupbin8_surprise_exact",
        "payloads": payloads,
        "runtime_patch": runtime_patch,
        "archive_binding": runtime_patch["candidate_archive"],
        "allocation": dict(allocation),
        "receiver_parse_back_exact": parseback["report"]["packet_exact"] and parseback["report"]["repack_exact"],
        "archive_repeat_byte_identical": payloads["archive"]["sha256"] == payloads["archive_repeat"]["sha256"],
        "untouched_sections_byte_identical": all(assertions.values()),
        "section_preservation_receipt": preservation_record,
        "parseback": parseback,
        "archive_bytes": len(archive),
        "rate_contribution": 25.0 * len(archive) / wd3.RATE_DENOMINATOR,
    }


def read_archive_bytes(archive: bytes) -> dict[str, Any]:
    with zipfile.ZipFile(io.BytesIO(archive)) as bundle:
        member = bundle.read("p")
        info = bundle.getinfo("p")
    temporary = {
        "archive": archive,
        "member": member,
    }
    magic, version, codec, table_mode, reserved, hpac_bytes, semantic_bytes, carrier_bytes = RX1_HEADER.unpack_from(
        member
    )
    offset = RX1_HEADER.size
    temporary["header"] = member[:offset]
    for name, length in (("hpac", hpac_bytes), ("semantic", semantic_bytes), ("carrier", carrier_bytes)):
        temporary[name] = member[offset : offset + length]
        offset += length
    temporary["tail"] = member[offset:]
    temporary["fixed_residual"] = temporary["tail"][:FIXED_RESIDUAL_BYTES]
    temporary["token_stream"] = temporary["tail"][FIXED_RESIDUAL_BYTES:]
    temporary["framing"] = {
        "magic_hex": magic.hex(),
        "version": version,
        "codec": codec,
        "table_mode": table_mode,
        "reserved": reserved,
        "hpac_bytes": hpac_bytes,
        "carrier_bytes": carrier_bytes,
        "tail_bytes": len(temporary["tail"]),
        "zip": {
            "date_time": list(info.date_time),
            "create_system": info.create_system,
            "external_attr": info.external_attr,
            "compress_type": info.compress_type,
            "flag_bits": info.flag_bits,
            "extra_hex": info.extra.hex(),
            "comment_hex": info.comment.hex(),
        },
    }
    return temporary


def reprove_rj2_adapters(output: Path) -> dict[str, Any]:
    """Execute RJ2's real solver algebra and production coder on the GB1 body."""

    source = read_archive(GB1_ARCHIVE)
    parts, state = po1.load_carrier(GB1_ARCHIVE, GB1_RUNTIME)
    dx2_source = read_archive(rj2.DX2_ARCHIVE)
    if source["carrier"] != dx2_source["carrier"]:
        raise S1AError("GB1 carrier bytes differ from the RJ2 reviewed object")
    modules = rj2._runtime_riders(GB1_RUNTIME)
    cap1, selector = modules.carrier_repack.split_frame0_selector_carrier(parts.carrier_blob)
    if selector != state.selector or len(cap1) < 50:
        raise S1AError("GB1 predictor metadata surface differs")
    encoded = rj2.encode_carrier_stream(
        state,
        state.codes.astype(np.int32),
        runtime=GB1_RUNTIME,
        retention_root=output / "retained/carrier_identity",
        source_predictor_metadata=cap1[14:50],
    )
    if encoded["stream"] != source["carrier"]:
        raise S1AError("RJ2 production carrier encoder is not byte-identical on GB1")

    smoke = RJ2_REPLAY / "retained/smoke_pair_0000"
    source_pair = RJ2_REPLAY / "retained/source_pair_0000"
    base_codes = np.load(smoke / "carrier_codes_base.int16.npy", allow_pickle=False)
    jacobian = np.load(smoke / "carrier_jacobian.float64.npy", allow_pickle=False)
    expected_update = np.load(smoke / "carrier_update.float64.npy", allow_pickle=False)
    proposed = np.load(smoke / "carrier_codes_proposed_pair.int16.npy", allow_pickle=False)
    original_pose = np.load(source_pair / "original_pose6.npy", allow_pickle=False)
    trained_pose = np.load(smoke / "trained_pose.float32.npy", allow_pickle=False).reshape(-1)[:6]
    if not np.array_equal(base_codes, state.codes):
        raise S1AError("RJ2 compensation base codes differ from GB1")
    update, diagnostics = po1.solve_damped_least_squares(
        jacobian,
        np.asarray(original_pose, dtype=np.float64).reshape(-1)[:6] - np.asarray(trained_pose, dtype=np.float64),
        damping=0.05,
        max_code_step=4.0,
    )
    resolved_codes = po1.quantize_int12_update(base_codes[0], update).astype(np.int16, copy=False)
    replay_max_abs = float(np.max(np.abs(update - expected_update)))
    if replay_max_abs > 1e-5 or not np.array_equal(resolved_codes, proposed):
        raise S1AError("RJ2 compensation solve replay differs")
    retained = {
        "update": atomic_npy(output / "retained/compensation_reproof/carrier_update.float64.npy", update),
        "proposed_codes": atomic_npy(
            output / "retained/compensation_reproof/carrier_codes_proposed_pair.int16.npy", resolved_codes
        ),
    }
    receipt = {
        "schema": "ddm_s1a_rj2_adapter_reproof.v1",
        "axis": AXIS,
        "score_claim": False,
        "gb1_carrier_equals_rj2_dx2_carrier": True,
        "carrier_state_codes_exact": True,
        "compensation_binding_pass": True,
        "compensation_float_replay_max_abs": replay_max_abs,
        "compensation_float_replay_atol": 1e-5,
        "compensation_float_replay_note": "RJ2 retained trained_pose as float32 while its local solve consumed the live pose tensor; the code-lattice verdict is exact",
        "compensation_int12_codes_exact": True,
        "carrier_production_chain": "CAP1 then DX2 then RR5 then Brotli q9/lgwin16",
        "carrier_stream_byte_identical": True,
        "gb1_carrier": section_record(source["carrier"]),
        "source_predictor_metadata": section_record(cap1[14:50]),
        "encoder_retained": encoded["retained"],
        "compensation_inputs": {
            "jacobian": file_record(smoke / "carrier_jacobian.float64.npy"),
            "original_pose6": file_record(source_pair / "original_pose6.npy"),
            "trained_pose6": file_record(smoke / "trained_pose.float32.npy"),
            "base_codes": file_record(smoke / "carrier_codes_base.int16.npy"),
        },
        "solve_diagnostics": diagnostics,
        "retained": retained,
    }
    atomic_json(output / "RJ2_ADAPTER_REPROOF.json", receipt)
    return receipt


def stage_b_contract(output: Path) -> dict[str, Any]:
    with zipfile.ZipFile(GT_CACHE) as bundle:
        pose_payload = bundle.read("gt_poses.npy")
    pose6 = np.load(io.BytesIO(pose_payload), allow_pickle=False)
    if pose6.shape != (600, 6) or pose6.dtype != np.dtype("<f8"):
        raise S1AError(f"Pose6 target member geometry differs: shape={pose6.shape}, dtype={pose6.dtype}")
    contract = {
        "schema": "ddm_s1a_stage_b_fingerprint_contract.v1",
        "status": "AWAITING_STAGE_A_TRAINING_OUTPUT",
        "producer": "stage_a_selected_seed_window",
        "consumer": "experiments/ddm_jg2_tail_reencode.py",
        "required_moved_object": {
            "runtime_tree": {"path": "absolute_path", "tree_sha256": "sha256", "source_archive_excluded": True},
            "archive": {"path": "absolute_path", "bytes": "int", "sha256": "sha256"},
            "receiver_frame1_field": {
                "path": "absolute_path",
                "bytes": 600 * 3 * receiver.CAMERA_H * receiver.CAMERA_W,
                "sha256": "sha256",
                "dtype": "uint8",
                "shape": [600, 3, receiver.CAMERA_H, receiver.CAMERA_W],
            },
            "receiver_consumed_token_field": {
                "path": "absolute_path",
                "bytes": 600 * receiver.EVAL_H * receiver.EVAL_W,
                "sha256": "sha256",
                "dtype": "uint8",
                "shape": [600, receiver.EVAL_H, receiver.EVAL_W],
                "must_be_decoded_by": "the same moved runtime/archive fingerprint above",
            },
        },
        "jg2_mirror_consistency": {
            "runtime_root_must_equal_moved_runtime": True,
            "pointer_archive_member_must_equal_runtime_archive_member": True,
            "expect_pointer_sha256_must_equal_moved_archive_sha256": True,
            "control_600_must_reencode_receiver_consumed_field_byte_identically": True,
            "edited_encode_delta_trustworthy_only_after_control": True,
        },
        "pose6_target_custody": {
            "container": file_record(GT_CACHE),
            "member": "gt_poses.npy",
            "member_bytes": len(pose_payload),
            "member_sha256": sha256_bytes(pose_payload),
            "dtype": pose6.dtype.str,
            "shape": list(pose6.shape),
        },
        "disposition": "QUEUED-WITH-A-FIRE-ORDER",
        "owner": "MAIN-designated Stage-B moved-field producer",
        "consumer_store": "/Volumes/APDataStore/pact/ddm_s1_trained_renderer_diagonal/stage_b/",
        "fire_trigger": "one Stage-A seed/window is selected and its moved runtime, archive, realized frame-1 field, and receiver-decoded token field all carry matching fingerprints",
    }
    atomic_json(output / "STAGE_B_FINGERPRINT_CONTRACT.json", contract)
    return contract


def _cheap_config(mode: str, seed: int) -> dict[str, Any]:
    if mode == "off":
        return {
            "mode": "off",
            "allocation_family": "uniform_bits",
            "uniform_bits": [],
            "rung_weights": [],
            "base_weight": 1.0,
            "sampler_seed": seed,
        }
    return {
        "mode": "sampled",
        "allocation_family": "uniform_bits",
        "uniform_bits": [3, 2],
        "rung_weights": [1.0, 1.0],
        "base_weight": 1.0,
        "sampler_seed": seed,
    }


def config_for(
    *,
    seed: int,
    mode: str,
    action: str,
    output: Path,
    resume_root: Path,
    resume_from: Path | None,
    custody_receipt: Path,
) -> dict[str, Any]:
    cache = json.loads(WD3_CACHE_RESULT.read_text(encoding="utf-8"))
    config = wd3.blocked_config_template()
    config.update(
        {
            "action": action,
            "output": str(output.resolve()),
            "seed": seed,
            "device": "cpu" if action == "prepare_arm_birth" else "mps",
            "chunk_pairs": 60,
            "checkpoint_every_epochs": 5,
            "minimum_free_bytes": 8 * 1024**3,
            "teacher_cache_result": str(WD3_CACHE_RESULT.resolve()),
            "resume_from": None if resume_from is None else str(resume_from.resolve()),
            "resume_root": str(resume_root.resolve()),
            "arm": "W96_flattened",
            "completed_arms": list(wd3.ARM_ORDER),
            "negative_confirmed_arms": ["W0_warm", "W0_reset", "D56", "F64"],
            "capacity_pressure_confirmed": True,
            "subsets": {
                "controller_n60": wd3.evenly_strided_indices().tolist(),
                "negative_n120": list(map(int, cache["negative_n120"])),
                "controller_kind": "evenly_strided",
                "negative_kind": "seeded_stratified_random",
                "prefix": False,
            },
            "epochs": EPOCHS,
            "batch_pairs": 1,
            "stage_a": stage_a_binding(custody_receipt),
            "cheap_to_shrink": _cheap_config(mode, seed),
            "expected_builder_sha256": sha256_file(Path(wd3.__file__).resolve()),
            "expected_receiver_sha256": sha256_file(Path(receiver.__file__).resolve()),
        }
    )
    return config


def _write_configs_and_births(output: Path, custody_receipt: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    birth_rows = []
    births: dict[int, Path] = {}
    for seed in SEEDS:
        root = output / f"births/seed_{seed}"
        config = config_for(
            seed=seed,
            mode="off",
            action="prepare_arm_birth",
            output=root,
            resume_root=root / "checkpoints",
            resume_from=None,
            custody_receipt=custody_receipt,
        )
        config_path = root / "BIRTH_CONFIG.json"
        atomic_json(config_path, config)
        receipt = wd3.prepare_arm_birth(config)
        birth = Path(receipt["checkpoint"]["path"])
        births[seed] = birth
        birth_rows.append(
            {
                "seed": seed,
                "config": file_record(config_path),
                "config_sha256": wd3.canonical_sha256(config),
                "birth": file_record(birth),
                "initializer_loaded_strict": True,
                "training_launched": False,
            }
        )
    left = torch.load(births[SEEDS[0]], map_location="cpu", weights_only=False)
    right = torch.load(births[SEEDS[1]], map_location="cpu", weights_only=False)
    if not all(
        torch.equal(left["live_state_dict"][name], right["live_state_dict"][name]) for name in left["live_state_dict"]
    ):
        raise S1AError("seed-specific births do not share the exact RJ1 initializer")
    if torch.equal(left["rng"]["generator"], right["rng"]["generator"]):
        raise S1AError("seed-specific birth RNG states did not vary")
    variation = {
        "schema": "ddm_s1a_seed_variation_reproof.v1",
        "weights_identical_numerator": len(left["live_state_dict"]),
        "weights_identical_denominator": len(left["live_state_dict"]),
        "optimizer_state_identical": left["optimizer_state_dict"] == right["optimizer_state_dict"],
        "rng_generator_differs": True,
        "path": "seed -> seed_everything -> seed-specific birth RNG -> restored training generator -> permutation -> trajectory",
        "overwrite_after_knob": False,
        "births": birth_rows,
    }
    atomic_json(output / "SEED_VARIATION_REPROOF.json", variation)

    train_rows = []
    for mode, seed in (("off", SEEDS[0]), ("off", SEEDS[1]), ("on", SEEDS[0])):
        tag = f"{mode}_seed_{seed}"
        root = output / f"training/{tag}"
        config = config_for(
            seed=seed,
            mode=mode,
            action="train",
            output=root,
            resume_root=root / "resume",
            resume_from=births[seed],
            custody_receipt=custody_receipt,
        )
        path = output / f"launch_requests/{tag}.json"
        atomic_json(path, config)
        fire_order = wd3.compile_fire_order(config)
        structural = [
            value
            for value in fire_order["blockers"]
            if not any(
                token in value
                for token in (
                    "lane is unclaimed",
                    "launch authorization remains false",
                    "r5 PID 63183 exit is not verified",
                )
            )
        ]
        if structural:
            raise S1AError(f"Stage-A request has non-MAIN structural blockers: {structural}")
        train_rows.append(
            {
                "arm": tag,
                "seed": seed,
                "treatment": mode,
                "request": file_record(path),
                "request_config_sha256": wd3.canonical_sha256(config),
                "resume_from": file_record(births[seed]),
                "main_owned_gate_fields": ["scorer_lane", "metal_lane", "launch_authorized", "r5_exit_verified"],
                "exact_command_argv_after_main_compiles": [
                    ".venv/bin/python",
                    "experiments/ddm_wd3_scorer_aware_width_distillation.py",
                    "train",
                    "--compiled-config",
                    str((root / "COMPILED_CONFIG.json").resolve()),
                ],
                "apparatus_blockers": [],
                "main_gate_blockers": fire_order["blockers"],
            }
        )
    return train_rows, variation


def _memory_and_wall_preflight(output: Path) -> dict[str, Any]:
    vm = subprocess.run(["vm_stat"], capture_output=True, text=True, check=True).stdout
    page_size = 16_384
    free_pages = 0
    for line in vm.splitlines():
        if line.startswith(("Pages free:", "Pages inactive:", "Pages speculative:")):
            free_pages += int(line.split(":", 1)[1].strip().rstrip("."))
    available_bytes = free_pages * page_size
    f64_receipt = Path(
        "/Volumes/APDataStore/pact/ddm_wd3_scorer_aware_width_distillation/fire_rung5_f64/launcher/safe_run_status.json"
    )
    measured = json.loads(f64_receipt.read_text(encoding="utf-8"))
    limit_mib = 65_536
    if available_bytes < limit_mib * 1024**2:
        raise S1AError("current unified-memory availability is below the sealed launcher limit")
    off_seconds = 2 * EPOCHS * MEASURED_SECONDS_PER_EPOCH
    on_seconds = EPOCHS * MEASURED_SECONDS_PER_EPOCH * 2
    receipt = {
        "schema": "ddm_s1a_memory_wall_preflight.v1",
        "axis": AXIS,
        "available_memory_bytes_numerator": available_bytes,
        "planned_launcher_limit_bytes_denominator": limit_mib * 1024**2,
        "available_over_limit_ratio": available_bytes / (limit_mib * 1024**2),
        "measured_f64_peak_rss_mib_numerator": measured["peak_rss_mib"],
        "planned_launcher_limit_mib_denominator": limit_mib,
        "measured_peak_over_limit_ratio": measured["peak_rss_mib"] / limit_mib,
        "memory_receipt": file_record(f64_receipt),
        "wall_projection": {
            "seconds_per_epoch_measured": MEASURED_SECONDS_PER_EPOCH,
            "seconds_per_epoch_source_receipt": str(
                (REPO / ".omx/research/ddm_s1_trained_renderer_diagonal_20260825.md").resolve()
            ),
            "epochs_per_run_denominator": EPOCHS,
            "off_runs_numerator": 2,
            "on_runs_numerator": 1,
            "on_scorer_pass_factor_projection": 2,
            "off_projected_seconds": off_seconds,
            "on_projected_seconds": on_seconds,
            "total_projected_hours": (off_seconds + on_seconds) / 3600.0,
            "label": "PROJECTED_FROM_MEASURED_F64_SECONDS_PER_EPOCH; ON factor is operation-count projection",
        },
        "sequential_only": True,
        "pass": True,
    }
    atomic_json(output / "MEMORY_WALL_PREFLIGHT.json", receipt)
    return receipt


def seal(output: Path) -> dict[str, Any]:
    if output.resolve() != OUTPUT_ROOT.resolve():
        raise S1AError(f"seal output must be the arm root {OUTPUT_ROOT}")
    storage = storage_preflight(output)
    custody = verify_rj1_custody()
    custody_record = atomic_json(output / "RJ1_CUSTODY_REPROOF.json", custody)
    source = read_archive(GB1_ARCHIVE)
    if len(source["archive"]) != GB1_ARCHIVE_BYTES or sha256_bytes(source["archive"]) != GB1_ARCHIVE_SHA256:
        raise S1AError("exact GB1 archive pin differs")
    identity = {
        "schema": "ddm_s1a_gb1_identity_reproof.v1",
        "archive": file_record(GB1_ARCHIVE),
        "runtime_archive": file_record(GB1_RUNTIME / "archive.zip"),
        "archive_equals_runtime_archive": GB1_ARCHIVE.read_bytes() == (GB1_RUNTIME / "archive.zip").read_bytes(),
        "identity_rebuild_byte_identical": rj1.deterministic_zip(source["member"]) == source["archive"],
        "sections": {
            name: section_record(source[name])
            for name in ("hpac", "semantic", "carrier", "fixed_residual", "token_stream")
        },
        "framing": source["framing"],
    }
    if not identity["archive_equals_runtime_archive"]:
        raise S1AError("GB1 runtime does not carry the exact frontier archive")
    atomic_json(output / "GB1_IDENTITY_REPROOF.json", identity)

    initializer = torch.load(RJ1_INITIALIZER, map_location="cpu", weights_only=False)
    model = receiver.StudentSemanticRenderer(wd3.ARM_SPECS["W96_flattened"])
    model.load_state_dict(initializer, strict=True)
    allocation = receiver.uniform_allocation(model, 4)
    candidate = retain_renderer_candidate(
        root=output / "retained/initializer_candidate",
        packet=receiver.pack_student(model, allocation),
        allocation=receiver.allocation_telemetry(model, allocation),
        binding=stage_a_binding(Path(custody_record["path"])),
    )
    adapter_reproof = reprove_rj2_adapters(output)
    stage_b = stage_b_contract(output)
    train_rows, variation = _write_configs_and_births(output, Path(custody_record["path"]))
    _memory_and_wall_preflight(output)
    launch = {
        "schema": "ddm_s1a_main_owned_launch_order.v1",
        "disposition": "QUEUED-WITH-A-FIRE-ORDER",
        "owner": "MAIN",
        "consumer_store": str((OUTPUT_ROOT / "training").resolve()) + "/",
        "fire_trigger": "MAIN supplies fresh distinct scorer/Metal lane claims, launch_authorized=true, and r5_exit_verified=true; compile each request without source drift; run OFF seeds sequentially before deciding whether to fire ON",
        "order": ["off_seed_20260815", "off_seed_20260816", "on_seed_20260815"],
        "runs": train_rows,
        "off_first_gate": "measure the two-seed OFF floor and its unresolvable delta-S in gap units before ON; underpowered floor removes KILL but not the WIN branch",
        "resumable_from_disk": True,
        "checkpoint_every_epochs": 5,
        "stage_end_checkpoint_preserved": True,
        "all_payloads_retained": True,
        "memory_preflight": file_record(output / "MEMORY_WALL_PREFLIGHT.json"),
        "training_launched": False,
        "scorer_invocations": 0,
        "metal_invocations": 0,
    }
    atomic_json(output / "MAIN_LAUNCH_ORDER.json", launch)
    result = {
        "schema": "ddm_s1a_stage_a_seal.v1",
        "status": "APPARATUS_COMPLETE_MAIN_GATES_PENDING",
        "axis": AXIS,
        "score_claim": False,
        "frontier_moved": False,
        "storage": storage,
        "custody": custody_record,
        "gb1_identity": file_record(output / "GB1_IDENTITY_REPROOF.json"),
        "initializer_candidate": candidate,
        "rj2_adapter_reproof": file_record(output / "RJ2_ADAPTER_REPROOF.json"),
        "stage_b_contract": file_record(output / "STAGE_B_FINGERPRINT_CONTRACT.json"),
        "seed_variation": file_record(output / "SEED_VARIATION_REPROOF.json"),
        "memory_wall_preflight": file_record(output / "MEMORY_WALL_PREFLIGHT.json"),
        "launch_order": file_record(output / "MAIN_LAUNCH_ORDER.json"),
        "per_seed_readiness": variation["births"],
        "compensation_binding_pass": adapter_reproof["compensation_binding_pass"],
        "carrier_binding_pass": adapter_reproof["carrier_stream_byte_identical"],
        "stage_b_disposition": stage_b["disposition"],
        "training_launched": False,
        "scorer_invocations": 0,
        "metal_invocations": 0,
    }
    atomic_json(output / "S1A_CHAIN_SEAL.json", result)
    return result


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    seal_parser = sub.add_parser("seal")
    seal_parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    return root


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    result = seal(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
