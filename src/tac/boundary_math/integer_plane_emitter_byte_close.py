# SPDX-License-Identifier: MIT
"""Counted, strict byte-close adapter for the C2 integer-plane emitter.

The archive owns every video-derived byte: the counted base generator packet
and a quantized EMA/live quotient residual.  Decoder and exact factor-2 lattice
code are generic rule-118 program state.  Capped decoding is local evidence
only; the full archive byte count remains the rate measurement.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal

import numpy as np

from tac.boundary_math.integer_plane_banded_trainer import (
    LOGICAL_PAIR_COUNT,
    canonical_json,
    sha256_file,
)
from tac.boundary_math.integer_plane_emitter import (
    CAMERA_HEIGHT,
    CAMERA_WIDTH,
    PLANE_COUNT,
    RGB_CHANNELS,
    SCORER_HEIGHT,
    SCORER_WIDTH,
    QuotientResidualState,
    StructuredEmitterState,
    factor2_operator,
    numpy_uint8,
    realize_all_factor2,
)
from tac.boundary_math.pdw2_spatial_receiver import (
    PDW2_COEFFICIENT_ONLY_SPATIAL_NONIDENTIFIABILITY,
)
from tac.boundary_math.power_diagram_witness import (
    TARGET_COMPARISON_VERDICT,
    decode_pdw2,
    encode_pdw2,
)
from tac.boundary_math.shared_receiver_admission import BLOCKER_ID as SHARED_RECEIVER_BLOCKER_ID
from tac.witness_dsl.integer_plane_emitter_policy import (
    IntegerPlaneEmitterStageCheckpoint,
    PolicyMode,
)
from tac.witness_dsl.v10_production_receiver import (
    ARITHMETIC_ID as V10_ARITHMETIC_ID,
)
from tac.witness_dsl.v10_production_receiver import (
    RECEIVER_CONTRACT_ID as V10_RECEIVER_CONTRACT_ID,
)

ARCHIVE_SCHEMA: Final = "c2_integer_plane_counted_archive.v1"
BYTE_CLOSE_SCHEMA: Final = "c2_integer_plane_byte_close_receipt.v1"
MANIFEST_NAME: Final = "ipe_manifest.json"
BASE_PACKET_NAME: Final = "0.bin"
CODES_NAME: Final = "ipe_codes.f16"
HEAD_NAME: Final = "ipe_quotient_residual_head.f16"
PDW2_NAME: Final = "seg_head_target.pdw2"
REPAIR_NAME: Final = "ipe_repair.i8"
_BASE_NAMES: Final = (BASE_PACKET_NAME, MANIFEST_NAME, PDW2_NAME, CODES_NAME, HEAD_NAME)
PDW2_ROLE: Final = "training_only_target_certificate"


class C2ByteCloseError(ValueError):
    """Fail-closed counted archive or decoder custody violation."""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _zip_eof_offset(path: Path) -> int:
    size = path.stat().st_size
    tail_size = min(size, 65_557)
    with path.open("rb") as handle:
        handle.seek(size - tail_size)
        tail = handle.read()
    marker = tail.rfind(b"PK\x05\x06")
    if marker < 0 or marker + 22 > len(tail):
        raise C2ByteCloseError("archive has no terminal ZIP EOCD")
    comment_length = int.from_bytes(tail[marker + 20 : marker + 22], "little")
    return size - tail_size + marker + 22 + comment_length


def _strict_single_base_packet(path: Path) -> tuple[bytes, str]:
    resolved = path.expanduser().resolve(strict=True)
    if _zip_eof_offset(resolved) != resolved.stat().st_size:
        raise C2ByteCloseError("base archive has trailing bytes")
    with zipfile.ZipFile(resolved, "r") as archive:
        infos = archive.infolist()
        if len(infos) != 1 or infos[0].filename != BASE_PACKET_NAME or infos[0].is_dir() or infos[0].flag_bits & 1:
            raise C2ByteCloseError("base archive must contain exactly unencrypted 0.bin")
        packet = archive.read(infos[0])
    if not packet:
        raise C2ByteCloseError("base generator packet is empty")
    return packet, sha256_file(resolved)


def _tensor(payload: Any, name: str) -> np.ndarray:
    if not isinstance(payload, dict) or set(payload) != {"dtype", "shape", "data"}:
        raise C2ByteCloseError(f"{name} tensor payload fields mismatch")
    if payload["dtype"] != "float32":
        raise C2ByteCloseError(f"{name} must be float32")
    try:
        result = np.asarray(payload["data"], dtype=np.float32).reshape(payload["shape"])
    except (TypeError, ValueError) as exc:
        raise C2ByteCloseError(f"{name} tensor payload shape mismatch") from exc
    if not np.isfinite(result).all():
        raise C2ByteCloseError(f"{name} contains nonfinite values")
    return result


def _checkpoint_residual(
    checkpoint: IntegerPlaneEmitterStageCheckpoint,
    authority: Literal["ema", "live"],
) -> tuple[np.ndarray, np.ndarray]:
    if checkpoint.policy_contract.get("mode") != PolicyMode.BANDED_TRAINING.value:
        raise C2ByteCloseError("byte-close requires an active band-training checkpoint")
    state = checkpoint.ema_shadow if authority == "ema" else checkpoint.live_residual_parameters
    codes = _tensor(state["pair_plane_codes"], f"{authority}.pair_plane_codes")
    head = _tensor(state["shared_rgb_head"], f"{authority}.shared_rgb_head")
    if codes.shape[0] != LOGICAL_PAIR_COUNT or codes.shape[1] != PLANE_COUNT:
        raise C2ByteCloseError("checkpoint residual must cover exactly 600 two-plane pairs")
    if head.shape != (codes.shape[2], RGB_CHANNELS):
        raise C2ByteCloseError("checkpoint shared head geometry mismatch")
    return codes, head


def _zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    info.compress_type = zipfile.ZIP_DEFLATED
    return info


def _write_deterministic_zip(path: Path, members: list[tuple[str, bytes]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise C2ByteCloseError(f"archive overwrite refused: {path}")
    partial = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    if partial.exists():
        raise C2ByteCloseError(f"stale archive temporary requires review: {partial}")
    try:
        with zipfile.ZipFile(partial, "x", allowZip64=False) as archive:
            for name, payload in members:
                archive.writestr(_zip_info(name), payload, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        with partial.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(partial, path)
    finally:
        if partial.exists():
            partial.unlink()


@dataclass(frozen=True, slots=True)
class ParsedCountedArchive:
    path: Path
    manifest: dict[str, Any]
    base_packet: bytes
    pdw2_packet: bytes
    codes: np.ndarray
    head: np.ndarray
    repair: np.ndarray | None
    archive_bytes: int
    archive_sha256: str
    sections: tuple[dict[str, Any], ...]
    zip_overhead_bytes: int

    @property
    def residual(self) -> QuotientResidualState:
        return QuotientResidualState(self.codes, self.head, int(self.manifest["seed"]))


def build_counted_archive(
    *,
    base_archive: str | Path,
    checkpoint_path: str | Path,
    output: str | Path,
    pdw2_packet: bytes,
    pdw2_role: str = PDW2_ROLE,
    authority: Literal["ema", "live"] = "ema",
    repair: np.ndarray | None = None,
) -> dict[str, Any]:
    """Build a deterministic counted archive from a hash-bound stage state."""

    if pdw2_role != PDW2_ROLE:
        raise C2ByteCloseError(
            "PDW2 receiver-consumed authority refused: "
            f"{SHARED_RECEIVER_BLOCKER_ID}; #553 is target-only and the scorer-free "
            "spatial/RGB pullback plus n600 hard-oracle admission are not implemented "
            "by the #543 factor-2 receiver"
        )
    try:
        pdw2 = bytes(pdw2_packet)
        parsed_pdw2 = decode_pdw2(pdw2)
    except (TypeError, ValueError) as exc:
        raise C2ByteCloseError("PDW2 packet is not strict canonical #553 bytes") from exc
    if encode_pdw2(parsed_pdw2) != pdw2:
        raise C2ByteCloseError("PDW2 packet failed canonical parse/re-encode")

    base_path = Path(base_archive)
    checkpoint_path = Path(checkpoint_path).expanduser().resolve(strict=True)
    packet, base_archive_sha = _strict_single_base_packet(base_path)
    checkpoint_bytes = checkpoint_path.read_bytes()
    checkpoint = IntegerPlaneEmitterStageCheckpoint.from_bytes(checkpoint_bytes)
    codes, head = _checkpoint_residual(checkpoint, authority)
    # The archive authority is the explicitly quantized fp16 state.  Decode and
    # canonical NumPy comparison both use these parse-backed values.
    codes_bytes = np.ascontiguousarray(codes, dtype="<f2").tobytes(order="C")
    head_bytes = np.ascontiguousarray(head, dtype="<f2").tobytes(order="C")
    repair_bytes: bytes | None = None
    if repair is not None:
        repair_array = np.asarray(repair)
        if repair_array.dtype != np.int8 or repair_array.shape != (
            LOGICAL_PAIR_COUNT,
            PLANE_COUNT,
            SCORER_HEIGHT,
            SCORER_WIDTH,
            RGB_CHANNELS,
        ):
            raise C2ByteCloseError("repair must be int8 [600,2,384,512,3]")
        repair_bytes = np.ascontiguousarray(repair_array).tobytes(order="C")
    custody = checkpoint.rng_state.get("run_custody")
    if not isinstance(custody, dict):
        raise C2ByteCloseError("checkpoint is missing active run custody")
    manifest = {
        "schema": ARCHIVE_SCHEMA,
        "pair_count": LOGICAL_PAIR_COUNT,
        "geometry": [PLANE_COUNT, SCORER_HEIGHT, SCORER_WIDTH, RGB_CHANNELS],
        "residual_width": int(codes.shape[2]),
        "seed": int(checkpoint.rng_state["seed"]),
        "authority": authority,
        "quantization": "little_endian_ieee754_binary16_v1",
        "residual_head_semantics": "c2_rgb_residual_factor_not_segmentation_head_coefficients",
        "pdw2_role": pdw2_role,
        "pdw2_verdict": TARGET_COMPARISON_VERDICT,
        "pdw2_spatial_receiver_consumed": False,
        "receiver_contract_id": V10_RECEIVER_CONTRACT_ID,
        "receiver_arithmetic_id": V10_ARITHMETIC_ID,
        "receiver_composition_status": "factor2_lattice_reused_pdw2_spatial_pullback_absent",
        "policy_sha256": checkpoint.policy_contract["policy_sha256"],
        "trainer_config_sha256": checkpoint.config_sha256,
        "checkpoint_sha256": _sha256(checkpoint_bytes),
        "base_archive_sha256": base_archive_sha,
        "base_packet_sha256": _sha256(packet),
        "base_decoder_sha256": custody["base_decoder_sha256"],
        "band_sha256": custody["band_sha256"],
        "band_mode": custody["band_mode"],
        "sections": {
            BASE_PACKET_NAME: {"bytes": len(packet), "sha256": _sha256(packet)},
            PDW2_NAME: {"bytes": len(pdw2), "sha256": _sha256(pdw2)},
            CODES_NAME: {"bytes": len(codes_bytes), "sha256": _sha256(codes_bytes)},
            HEAD_NAME: {"bytes": len(head_bytes), "sha256": _sha256(head_bytes)},
            REPAIR_NAME: None
            if repair_bytes is None
            else {"bytes": len(repair_bytes), "sha256": _sha256(repair_bytes)},
        },
        "repair_semantics": None
        if repair_bytes is None
        else "dense_saturated_int8_delta_after_emitter_before_factor2_v1",
        "generic_decoder_counted_bytes": 0,
        "score_claim": False,
    }
    manifest_bytes = canonical_json(manifest)
    members = [
        (BASE_PACKET_NAME, packet),
        (MANIFEST_NAME, manifest_bytes),
        (PDW2_NAME, pdw2),
        (CODES_NAME, codes_bytes),
        (HEAD_NAME, head_bytes),
    ]
    if repair_bytes is not None:
        members.append((REPAIR_NAME, repair_bytes))
    output_path = Path(output).expanduser().resolve()
    _write_deterministic_zip(output_path, members)
    parsed = parse_counted_archive(output_path)
    return archive_receipt(parsed)


def parse_counted_archive(path: str | Path) -> ParsedCountedArchive:
    resolved = Path(path).expanduser().resolve(strict=True)
    if _zip_eof_offset(resolved) != resolved.stat().st_size:
        raise C2ByteCloseError("counted archive carries trailing bytes")
    with zipfile.ZipFile(resolved, "r") as archive:
        infos = archive.infolist()
        names = [info.filename for info in infos]
        if len(names) != len(set(names)) or any(info.is_dir() or info.flag_bits & 1 for info in infos):
            raise C2ByteCloseError("duplicate, directory, or encrypted archive member")
        if MANIFEST_NAME not in names:
            raise C2ByteCloseError("counted archive manifest is missing")
        manifest_raw = archive.read(MANIFEST_NAME)
        try:
            manifest = json.loads(manifest_raw.decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise C2ByteCloseError("counted manifest is not ASCII JSON") from exc
        if canonical_json(manifest) != manifest_raw:
            raise C2ByteCloseError("counted manifest is not canonical JSON")
        expected_manifest = {
            "schema",
            "pair_count",
            "geometry",
            "residual_width",
            "seed",
            "authority",
            "quantization",
            "residual_head_semantics",
            "pdw2_role",
            "pdw2_verdict",
            "pdw2_spatial_receiver_consumed",
            "receiver_contract_id",
            "receiver_arithmetic_id",
            "receiver_composition_status",
            "policy_sha256",
            "trainer_config_sha256",
            "checkpoint_sha256",
            "base_archive_sha256",
            "base_packet_sha256",
            "base_decoder_sha256",
            "band_sha256",
            "band_mode",
            "sections",
            "repair_semantics",
            "generic_decoder_counted_bytes",
            "score_claim",
        }
        if not isinstance(manifest, dict) or set(manifest) != expected_manifest:
            raise C2ByteCloseError("counted manifest fields mismatch")
        if (
            manifest["schema"] != ARCHIVE_SCHEMA
            or manifest["pair_count"] != LOGICAL_PAIR_COUNT
            or manifest["geometry"] != [PLANE_COUNT, SCORER_HEIGHT, SCORER_WIDTH, RGB_CHANNELS]
            or manifest["quantization"] != "little_endian_ieee754_binary16_v1"
            or manifest["authority"] not in {"ema", "live"}
            or manifest["score_claim"] is not False
            or manifest["residual_head_semantics"] != "c2_rgb_residual_factor_not_segmentation_head_coefficients"
            or manifest["pdw2_role"] != PDW2_ROLE
            or manifest["pdw2_verdict"] != TARGET_COMPARISON_VERDICT
            or manifest["pdw2_spatial_receiver_consumed"] is not False
            or manifest["receiver_contract_id"] != V10_RECEIVER_CONTRACT_ID
            or manifest["receiver_arithmetic_id"] != V10_ARITHMETIC_ID
            or manifest["receiver_composition_status"] != "factor2_lattice_reused_pdw2_spatial_pullback_absent"
        ):
            raise C2ByteCloseError("counted manifest sealed values mismatch")
        repair_record = manifest["sections"].get(REPAIR_NAME)
        expected_names = list(_BASE_NAMES) + ([] if repair_record is None else [REPAIR_NAME])
        if names != expected_names:
            raise C2ByteCloseError("archive member order/names mismatch or unknown section")
        payloads = {name: archive.read(name) for name in names if name != MANIFEST_NAME}
        sections: list[dict[str, Any]] = []
        compressed_sum = 0
        for info in infos:
            compressed_sum += info.compress_size
            data = manifest_raw if info.filename == MANIFEST_NAME else payloads[info.filename]
            record = manifest["sections"].get(info.filename)
            if info.filename == MANIFEST_NAME:
                declared = {"bytes": len(data), "sha256": _sha256(data)}
            else:
                declared = record
                if not isinstance(declared, dict) or set(declared) != {"bytes", "sha256"}:
                    raise C2ByteCloseError(f"section record mismatch: {info.filename}")
                if declared != {"bytes": len(data), "sha256": _sha256(data)}:
                    raise C2ByteCloseError(f"section custody mismatch: {info.filename}")
            sections.append(
                {
                    "name": info.filename,
                    "uncompressed_bytes": len(data),
                    "compressed_bytes": info.compress_size,
                    "sha256": _sha256(data),
                }
            )
    width = int(manifest["residual_width"])
    if not 1 <= width <= 64:
        raise C2ByteCloseError("residual width is invalid")
    codes_raw, head_raw = payloads[CODES_NAME], payloads[HEAD_NAME]
    try:
        parsed_pdw2 = decode_pdw2(payloads[PDW2_NAME])
    except ValueError as exc:
        raise C2ByteCloseError("counted PDW2 section is invalid") from exc
    if encode_pdw2(parsed_pdw2) != payloads[PDW2_NAME]:
        raise C2ByteCloseError("counted PDW2 section is noncanonical")
    expected_codes = LOGICAL_PAIR_COUNT * PLANE_COUNT * width
    if len(codes_raw) != expected_codes * 2 or len(head_raw) != width * RGB_CHANNELS * 2:
        raise C2ByteCloseError("quantized residual byte lengths mismatch")
    codes = np.frombuffer(codes_raw, dtype="<f2").astype(np.float32).reshape(LOGICAL_PAIR_COUNT, PLANE_COUNT, width)
    head = np.frombuffer(head_raw, dtype="<f2").astype(np.float32).reshape(width, RGB_CHANNELS)
    repair = None
    if repair_record is not None:
        expected_repair = LOGICAL_PAIR_COUNT * PLANE_COUNT * SCORER_HEIGHT * SCORER_WIDTH * RGB_CHANNELS
        if len(payloads[REPAIR_NAME]) != expected_repair:
            raise C2ByteCloseError("repair byte length mismatch")
        repair = np.frombuffer(payloads[REPAIR_NAME], dtype=np.int8).reshape(
            LOGICAL_PAIR_COUNT, PLANE_COUNT, SCORER_HEIGHT, SCORER_WIDTH, RGB_CHANNELS
        )
    archive_bytes = resolved.stat().st_size
    overhead = archive_bytes - compressed_sum
    if overhead < 0 or compressed_sum + overhead != archive_bytes:
        raise C2ByteCloseError("archive section byte accounting failed")
    return ParsedCountedArchive(
        resolved,
        manifest,
        payloads[BASE_PACKET_NAME],
        payloads[PDW2_NAME],
        codes,
        head,
        repair,
        archive_bytes,
        sha256_file(resolved),
        tuple(sections),
        overhead,
    )


def archive_receipt(parsed: ParsedCountedArchive) -> dict[str, Any]:
    compressed = sum(row["compressed_bytes"] for row in parsed.sections)
    return {
        "schema": ARCHIVE_SCHEMA,
        "archive": str(parsed.path),
        "archive_bytes": parsed.archive_bytes,
        "archive_sha256": parsed.archive_sha256,
        "sections": list(parsed.sections),
        "compressed_section_bytes": compressed,
        "zip_overhead_bytes": parsed.zip_overhead_bytes,
        "accounted_archive_bytes": compressed + parsed.zip_overhead_bytes,
        "base_video_payload_counted": True,
        "residual_video_payload_counted": True,
        "repair_video_payload_counted": parsed.repair is not None,
        "pdw2_video_target_payload_counted": True,
        "pdw2_spatial_receiver_consumed": False,
        "pdw2_promotion_blocker": PDW2_COEFFICIENT_ONLY_SPATIAL_NONIDENTIFIABILITY,
        "ema_authority": parsed.manifest["authority"] == "ema",
        "score_claim": False,
    }


def _project_camera_to_scorer(camera: np.ndarray) -> np.ndarray:
    operator = factor2_operator()
    out = np.empty(
        (camera.shape[0], PLANE_COUNT, SCORER_HEIGHT, SCORER_WIDTH, RGB_CHANNELS),
        dtype=np.uint8,
    )
    for pair_index in range(camera.shape[0]):
        for plane_index in range(PLANE_COUNT):
            numerator, denominator = operator.apply_numerators(camera[pair_index, plane_index])
            out[pair_index, plane_index] = np.clip(
                np.rint(numerator.astype(np.float64) / denominator), 0.0, 255.0
            ).astype(np.uint8)
    return out


def decode_counted_archive(
    *,
    archive: str | Path,
    base_decoder: str | Path,
    scratch_root: str | Path,
    pair_cap: int,
    output_raw: str | Path,
    workers: int = 1,
) -> dict[str, Any]:
    """Parse back and decode a bounded prefix through the exact C2 receiver."""

    parsed = parse_counted_archive(archive)
    if not 2 <= pair_cap <= LOGICAL_PAIR_COUNT:
        raise C2ByteCloseError("pair_cap must be in [2,600]")
    decoder = Path(base_decoder).expanduser().resolve(strict=True)
    if sha256_file(decoder) != parsed.manifest["base_decoder_sha256"]:
        raise C2ByteCloseError("base decoder SHA custody mismatch")
    scratch = Path(scratch_root).expanduser().resolve()
    scratch.mkdir(parents=True, exist_ok=True)
    output = Path(output_raw).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise C2ByteCloseError(f"decoded raw overwrite refused: {output}")
    with tempfile.TemporaryDirectory(prefix="c2_byte_close_", dir=scratch) as temp_name:
        temp = Path(temp_name)
        packet = temp / BASE_PACKET_NAME
        base_raw = temp / "base.raw"
        packet.write_bytes(parsed.base_packet)
        env = os.environ.copy()
        env.update({"INFLATE_MAX_PAIRS": str(pair_cap), "INFLATE_WORKERS": str(workers)})
        proc = subprocess.run(
            [sys.executable, str(decoder), str(packet), str(base_raw)],
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode:
            raise C2ByteCloseError(f"base decoder failed rc={proc.returncode}: {proc.stderr[-1000:]}")
        expected_base = pair_cap * PLANE_COUNT * CAMERA_HEIGHT * CAMERA_WIDTH * RGB_CHANNELS
        if not base_raw.is_file() or base_raw.stat().st_size != expected_base:
            raise C2ByteCloseError("capped base raw size mismatch")
        camera = np.memmap(
            base_raw,
            mode="r",
            dtype=np.uint8,
            shape=(pair_cap, PLANE_COUNT, CAMERA_HEIGHT, CAMERA_WIDTH, RGB_CHANNELS),
        )
        base_planes = _project_camera_to_scorer(camera)
        structured = StructuredEmitterState.from_base(
            base_planes.astype(np.float32), residual_width=parsed.codes.shape[2]
        )
        residual = QuotientResidualState(parsed.codes[:pair_cap], parsed.head, int(parsed.manifest["seed"]))
        canonical = numpy_uint8(structured, residual, require_distinct_planes=True)
        if parsed.repair is not None:
            canonical = np.clip(canonical.astype(np.int16) + parsed.repair[:pair_cap].astype(np.int16), 0, 255).astype(
                np.uint8
            )
        lattice = realize_all_factor2(canonical)
        if not all(row.certified_exact and row.numerator_exact for row in lattice.rows):
            raise C2ByteCloseError("factor-2 receiver proof failed")
        decoded_scorer = _project_camera_to_scorer(lattice.camera_planes)
        if not np.array_equal(decoded_scorer, canonical):
            raise C2ByteCloseError("decoded scorer bytes differ from canonical NumPy emitter")
        partial = output.with_name(f".{output.name}.tmp.{os.getpid()}")
        partial.write_bytes(np.ascontiguousarray(lattice.camera_planes).tobytes(order="C"))
        with partial.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(partial, output)
    return {
        "schema": BYTE_CLOSE_SCHEMA,
        "scope": "capped_non_n600_non_score" if pair_cap < LOGICAL_PAIR_COUNT else "full_n600_decode_non_score",
        "pair_cap": pair_cap,
        "logical_pair_count": LOGICAL_PAIR_COUNT,
        "archive_bytes_full": parsed.archive_bytes,
        "archive_sha256": parsed.archive_sha256,
        "decoded_raw": str(output),
        "decoded_raw_bytes_capped": output.stat().st_size,
        "decoded_raw_sha256": sha256_file(output),
        "canonical_numpy_scorer_sha256": _sha256(np.ascontiguousarray(canonical).tobytes()),
        "parse_back_scorer_sha256": _sha256(np.ascontiguousarray(decoded_scorer).tobytes()),
        "numpy_decode_equal": True,
        "factor2_exact": True,
        "section_accounting": archive_receipt(parsed),
        "base_decoder_stdout_tail": proc.stdout[-500:],
        "score_claim": False,
    }


def compare_capped_archives(
    *,
    pre_archive: str | Path,
    post_archive: str | Path,
    base_decoder: str | Path,
    cache: str | Path,
    upstream: str | Path,
    scratch_root: str | Path,
    output_root: str | Path,
    pair_cap: int,
    cpu_threads: int = 1,
) -> dict[str, Any]:
    """Measure pre/post hard-oracle rows through the real capped receiver.

    This deliberately imports the already hash-pinned C2 scorer/cache custody
    helper instead of creating a second unpinned scorer import path.
    """

    from tools.measure_c2_integer_plane_emitter import (
        _distortion,
        _distortion_outputs,
        _load_distortion_net,
        _load_real_cache,
    )

    if not 2 <= pair_cap <= 24:
        raise C2ByteCloseError("hard-oracle capped comparison requires 2..24 pairs")
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    pre_raw = root / "pre.raw"
    post_raw = root / "post.raw"
    pre_decode = decode_counted_archive(
        archive=pre_archive,
        base_decoder=base_decoder,
        scratch_root=scratch_root,
        pair_cap=pair_cap,
        output_raw=pre_raw,
    )
    post_decode = decode_counted_archive(
        archive=post_archive,
        base_decoder=base_decoder,
        scratch_root=scratch_root,
        pair_cap=pair_cap,
        output_raw=post_raw,
    )
    cache_fields, cache_sha = _load_real_cache(Path(cache).expanduser().resolve())
    model, torch, scorer_hashes = _load_distortion_net(Path(upstream).expanduser().resolve(), cpu_threads)
    shape = (pair_cap, PLANE_COUNT, CAMERA_HEIGHT, CAMERA_WIDTH, RGB_CHANNELS)
    pre_frames = np.memmap(pre_raw, mode="r", dtype=np.uint8, shape=shape)
    post_frames = np.memmap(post_raw, mode="r", dtype=np.uint8, shape=shape)
    rows: dict[str, list[dict[str, float]]] = {"pre": [], "post": []}
    for pair_index in range(pair_cap):
        source_outputs = _distortion_outputs(
            model,
            torch,
            np.asarray(cache_fields["gt_f0"][pair_index]),
            np.asarray(cache_fields["gt_f1"][pair_index]),
        )
        for label, frames in (("pre", pre_frames), ("post", post_frames)):
            candidate = _distortion_outputs(
                model,
                torch,
                np.asarray(frames[pair_index, 0]),
                np.asarray(frames[pair_index, 1]),
            )
            rows[label].append(_distortion(model, candidate, source_outputs))
    parsed_pre = parse_counted_archive(pre_archive)
    parsed_post = parse_counted_archive(post_archive)

    def summary(label: str, parsed: ParsedCountedArchive) -> dict[str, Any]:
        return {
            "archive_bytes": parsed.archive_bytes,
            "archive_sha256": parsed.archive_sha256,
            "d_seg": float(np.mean([row["d_seg"] for row in rows[label]])),
            "d_pose": float(np.mean([row["d_pose"] for row in rows[label]])),
            "pairs": pair_cap,
        }

    pre = summary("pre", parsed_pre)
    post = summary("post", parsed_post)
    del pre_frames, post_frames
    cleanup_rows = []
    for decoded in (pre_decode, post_decode):
        raw_path = Path(decoded["decoded_raw"])
        cleanup_rows.append(
            {
                "original_path": str(raw_path),
                "bytes": decoded["decoded_raw_bytes_capped"],
                "sha256": decoded["decoded_raw_sha256"],
                "reason": "success-only scorer input deterministically rebuildable from hash-bound archive/decoder",
                "cold_store_destination": None,
                "false_authority_flags": {
                    "score_claim": False,
                    "promotion_eligible": False,
                    "scope": f"prefix_n{pair_cap}_non_n600_non_promotion",
                },
            }
        )
    cleanup_authorization = {
        "schema": "c2_integer_plane_decoded_raw_cleanup.v1",
        "certify_or_block": True,
        "deletion_action": "authorized_success_only_after_this_manifest_fsync",
        "rows": cleanup_rows,
    }
    cleanup_path = root / "decoded_raw_cleanup.json"
    if cleanup_path.exists():
        raise C2ByteCloseError(f"cleanup manifest overwrite refused: {cleanup_path}")
    cleanup_path.write_bytes(canonical_json(cleanup_authorization))
    with cleanup_path.open("rb") as handle:
        os.fsync(handle.fileno())
    for row in cleanup_rows:
        Path(row["original_path"]).unlink()
    return {
        "schema": "c2_integer_plane_pre_post_hard_oracle.v1",
        "axis": "[macOS-CPU capped hard-oracle, non-score]",
        "scope": f"prefix_n{pair_cap}_non_n600_non_promotion",
        "cache_sha256": cache_sha,
        "scorer_hashes": scorer_hashes,
        "pre": pre,
        "post": post,
        "delta": {
            "d_seg": post["d_seg"] - pre["d_seg"],
            "d_pose": post["d_pose"] - pre["d_pose"],
            "archive_bytes": post["archive_bytes"] - pre["archive_bytes"],
        },
        "training_moved_d_seg": post["d_seg"] != pre["d_seg"],
        "rate_counted": True,
        "pre_decode": pre_decode,
        "post_decode": post_decode,
        "cleanup": {
            **cleanup_authorization,
            "manifest": str(cleanup_path),
            "manifest_sha256": sha256_file(cleanup_path),
            "deleted_after_certification": True,
            "certify_or_block": True,
        },
        "score_claim": False,
        "pointer_mutation": False,
    }


__all__ = [
    "ARCHIVE_SCHEMA",
    "BYTE_CLOSE_SCHEMA",
    "PDW2_ROLE",
    "C2ByteCloseError",
    "ParsedCountedArchive",
    "archive_receipt",
    "build_counted_archive",
    "compare_capped_archives",
    "decode_counted_archive",
    "parse_counted_archive",
]
