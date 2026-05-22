# SPDX-License-Identifier: MIT
"""Materialize byte-closed decoder-q selective runtime candidates.

This module consumes a DQS1 packet plan, or a bridge plan plus the FEC6 parent
archive, and emits a submission directory whose only charged selective bytes
are appended to archive.zip member ``x``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from tac.optimization.decoder_q_selective_runtime_packet import (
    BRIDGE_SCHEMA,
    FALSE_AUTHORITY,
    affected_frames_for_pairs,
    build_decoder_q_selective_runtime_packet_plan,
    pack_dqs1_payload,
    unpack_dqs1_payload,
)
from tac.optimization.decoder_q_selective_runtime_packet import (
    SCHEMA as PACKET_SCHEMA,
)
from tac.optimization.fec6_decoder_mutations import extract_fec6_decoder_blob

MATERIALIZER_SCHEMA = "decoder_q_selective_runtime_materialization.v1"
TOOL = "tac.optimization.decoder_q_selective_runtime_materializer"


class DecoderQSelectiveRuntimeMaterializerError(ValueError):
    """Raised when selective runtime materialization would lose custody."""


@dataclass(frozen=True)
class StoredMember:
    """Single ZIP_STORED member plus custody metadata."""

    name: str
    data: bytes
    zip_bytes: int
    zip_sha256: str
    crc32_hex: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "member_name": self.name,
            "member_bytes": len(self.data),
            "member_sha256": sha256_bytes(self.data),
            "zip_bytes": self.zip_bytes,
            "zip_sha256": self.zip_sha256,
            "compress_type": "ZIP_STORED",
            "crc32_hex": self.crc32_hex,
        }


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DecoderQSelectiveRuntimeMaterializerError(f"{path}: expected JSON object")
    return payload


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _require_false_authority(payload: dict[str, Any], *, label: str) -> None:
    for key in FALSE_AUTHORITY:
        if payload.get(key) is not False:
            raise DecoderQSelectiveRuntimeMaterializerError(
                f"{label} must keep {key}=false"
            )


def _require_equal(*, label: str, expected: Any, actual: Any) -> None:
    if expected != actual:
        raise DecoderQSelectiveRuntimeMaterializerError(
            f"{label} mismatch: expected {expected!r}, got {actual!r}"
        )


def _as_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise DecoderQSelectiveRuntimeMaterializerError(f"{label} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise DecoderQSelectiveRuntimeMaterializerError(
            f"{label} must be an integer"
        ) from exc
    if isinstance(value, str):
        if str(result) != value:
            raise DecoderQSelectiveRuntimeMaterializerError(f"{label} must be integral")
    elif result != value:
        raise DecoderQSelectiveRuntimeMaterializerError(f"{label} must be integral")
    return result


def _resolve_path(path: Path | str, *, repo_root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else repo_root / candidate


def _replace_once(text: str, old: str, new: str, *, label: str) -> str:
    if old not in text:
        raise DecoderQSelectiveRuntimeMaterializerError(
            f"runtime patch point missing: {label}"
        )
    return text.replace(old, new, 1)


def _replace_one_of(
    text: str,
    replacements: list[tuple[str, str]],
    *,
    label: str,
) -> str:
    for old, new in replacements:
        if old in text:
            return text.replace(old, new, 1)
    raise DecoderQSelectiveRuntimeMaterializerError(
        f"runtime patch point missing: {label}"
    )


def read_single_stored_member(archive_zip: Path) -> StoredMember:
    if not archive_zip.is_file():
        raise DecoderQSelectiveRuntimeMaterializerError(f"archive missing: {archive_zip}")
    with zipfile.ZipFile(archive_zip) as zf:
        infos = zf.infolist()
        if len(infos) != 1:
            raise DecoderQSelectiveRuntimeMaterializerError(
                f"{archive_zip}: expected exactly one ZIP member, found {len(infos)}"
            )
        info = infos[0]
        if info.compress_type != zipfile.ZIP_STORED:
            raise DecoderQSelectiveRuntimeMaterializerError(
                f"{archive_zip}: member {info.filename!r} must be ZIP_STORED"
            )
        data = zf.read(info.filename)
    return StoredMember(
        name=info.filename,
        data=data,
        zip_bytes=archive_zip.stat().st_size,
        zip_sha256=file_sha256(archive_zip),
        crc32_hex=f"{info.CRC:08x}",
    )


def write_single_stored_member(archive_zip: Path, *, member_name: str, data: bytes) -> None:
    archive_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_zip, "w", compression=zipfile.ZIP_STORED) as zf:
        info = zipfile.ZipInfo(member_name)
        info.compress_type = zipfile.ZIP_STORED
        info.date_time = (1980, 1, 1, 0, 0, 0)
        zf.writestr(info, data)


def parse_dqs1_payload(payload: bytes) -> dict[str, Any]:
    """Parse the compact DQS1 wire payload used inside archive member ``x``."""

    parsed = unpack_dqs1_payload(payload)
    pairs = [int(pair) for pair in parsed["pair_indices"]]
    for pair in pairs:
        if pair >= 600:
            raise DecoderQSelectiveRuntimeMaterializerError(
                f"DQS1 selected pair outside FEC6 range: {pair}"
            )
    return {
        "magic": "DQS1",
        "frame_policy": parsed["frame_policy"],
        "frame_policy_code": parsed["frame_policy_code"],
        "mode_byte": parsed["mode_byte"],
        "pair_encoding": parsed["pair_encoding"],
        "pair_encoding_code": parsed["pair_encoding_code"],
        "storage_index": parsed["storage_index"],
        "q_offset": parsed["q_offset"],
        "delta": parsed["delta"],
        "pair_indices": pairs,
        "pair_count": len(pairs),
        "affected_frame_indices": affected_frames_for_pairs(
            pairs,
            frame_policy=str(parsed["frame_policy"]),
        ),
        "payload_bytes": len(payload),
        "payload_sha256": sha256_bytes(payload),
    }


def _storage_index_from_packet_plan(plan: dict[str, Any]) -> int:
    mutation = plan.get("mutation")
    if not isinstance(mutation, dict):
        raise DecoderQSelectiveRuntimeMaterializerError("packet plan mutation missing")
    tensor = mutation.get("tensor")
    if isinstance(tensor, dict) and "storage_index" in tensor:
        return _as_int(tensor["storage_index"], label="mutation tensor.storage_index")
    if "storage_index" in mutation:
        return _as_int(mutation["storage_index"], label="mutation storage_index")
    raise DecoderQSelectiveRuntimeMaterializerError(
        "packet plan mutation tensor.storage_index missing"
    )


def dqs1_payload_from_packet_plan(plan: dict[str, Any]) -> bytes:
    if plan.get("schema") != PACKET_SCHEMA:
        raise DecoderQSelectiveRuntimeMaterializerError("packet plan schema mismatch")
    _require_false_authority(plan, label="packet plan")
    packet = plan.get("selective_packet")
    mutation = plan.get("mutation")
    if not isinstance(packet, dict) or not isinstance(mutation, dict):
        raise DecoderQSelectiveRuntimeMaterializerError(
            "packet plan must contain selective_packet and mutation objects"
        )
    pairs = packet.get("selected_pair_indices")
    if not isinstance(pairs, list):
        raise DecoderQSelectiveRuntimeMaterializerError(
            "packet plan selected_pair_indices missing"
        )
    payload = pack_dqs1_payload(
        pair_indices=[_as_int(pair, label="selected pair") for pair in pairs],
        frame_policy=str(packet.get("frame_policy")),
        storage_index=_storage_index_from_packet_plan(plan),
        q_offset=_as_int(mutation.get("q_offset"), label="mutation q_offset"),
        delta=_as_int(mutation.get("delta"), label="mutation delta"),
        pair_encoding=str(packet.get("pair_encoding", "raw_u16")),
    )
    expected_sha = packet.get("payload_sha256")
    if expected_sha is not None and str(expected_sha) != sha256_bytes(payload):
        raise DecoderQSelectiveRuntimeMaterializerError(
            "repacked DQS1 payload does not match packet plan payload_sha256"
        )
    expected_bytes = packet.get("payload_bytes")
    if expected_bytes is not None and _as_int(expected_bytes, label="payload_bytes") != len(payload):
        raise DecoderQSelectiveRuntimeMaterializerError(
            "repacked DQS1 payload length does not match packet plan"
        )
    return payload


def verify_packet_plan_base_archive(
    packet_plan: dict[str, Any],
    *,
    base_archive: Path,
    repo_root: Path,
) -> None:
    """Fail closed if a loaded packet plan does not describe ``base_archive``."""

    archive_meta = packet_plan.get("base_archive")
    if not isinstance(archive_meta, dict):
        raise DecoderQSelectiveRuntimeMaterializerError(
            "packet plan base_archive metadata missing"
        )
    expected_path = archive_meta.get("path")
    if isinstance(expected_path, str) and expected_path:
        _require_equal(
            label="packet plan base_archive.path",
            expected=_resolve_path(expected_path, repo_root=repo_root).resolve(),
            actual=base_archive.resolve(),
        )

    base_member = read_single_stored_member(base_archive)
    checks = {
        "zip_bytes": base_archive.stat().st_size,
        "zip_sha256": file_sha256(base_archive),
        "member_name": base_member.name,
        "member_bytes": len(base_member.data),
        "member_sha256": sha256_bytes(base_member.data),
        "compress_type": "ZIP_STORED",
        "crc32_hex": base_member.crc32_hex,
        "decoder_sha256": sha256_bytes(extract_fec6_decoder_blob(base_member.data)),
    }
    for key, actual in checks.items():
        if key not in archive_meta:
            raise DecoderQSelectiveRuntimeMaterializerError(
                f"packet plan base_archive.{key} missing"
            )
        _require_equal(
            label=f"packet plan base_archive.{key}",
            expected=archive_meta[key],
            actual=actual,
        )


def load_or_build_packet_plan(
    plan_path: Path,
    *,
    base_archive: Path,
    repo_root: Path,
    frame_policy: str = "pair_all_frames",
    max_units: int | None = None,
) -> dict[str, Any]:
    """Load a packet plan, or derive one from a bridge plan."""

    payload = load_json_object(plan_path)
    schema = payload.get("schema")
    if schema == PACKET_SCHEMA:
        _require_false_authority(payload, label="packet plan")
        verify_packet_plan_base_archive(
            payload,
            base_archive=base_archive,
            repo_root=repo_root,
        )
        return payload
    if schema == BRIDGE_SCHEMA:
        return build_decoder_q_selective_runtime_packet_plan(
            payload,
            base_archive=base_archive,
            repo_root=repo_root,
            frame_policy=frame_policy,
            max_units=max_units,
        )
    raise DecoderQSelectiveRuntimeMaterializerError(
        f"unsupported plan schema {schema!r}; expected {PACKET_SCHEMA} or {BRIDGE_SCHEMA}"
    )


def build_selective_inflate_py(base_inflate_text: str) -> str:
    """Patch the FEC6 inflate runtime with archive-local DQS1 selective decode."""

    out = _replace_once(
        base_inflate_text,
        '"""Inflate PR101 HNeRV with an archive-charged FES1 frame selector."""',
        '"""Inflate FEC6 with an archive-charged DQS1 decoder-q selector."""',
        label="runtime docstring",
    )
    out = _replace_once(
        out,
        "from pathlib import Path\n\nimport torch",
        "from pathlib import Path\n\nimport numpy as np\nimport torch",
        label="numpy import",
    )
    out = _replace_once(
        out,
        "from codec import parse_archive  # type: ignore[import-not-found]",
        """from codec import (  # type: ignore[import-not-found]
    CONV4_STORAGE_PERMS,
    DECODER_BLOB_LEN,
    DECODER_BYTE_MAPS,
    DECODER_STORAGE_ORDER,
    DECODER_STREAM_ENDS,
    decode_mapped_u8,
    decompress_brotli_streams,
    parse_archive,
)""",
        label="codec imports",
    )
    out = _replace_once(
        out,
        "OUTER_MAGIC = b\"FP11\"\n",
        """OUTER_MAGIC = b"FP11"
DQS1_MAGIC = b"DQS1"
DQS1_FRAME_POLICY_BY_CODE = {1: "pair_all_frames", 2: "segnet_last_frame_only"}
DQS1_PAIR_ENCODING_BY_CODE = {0: "raw_u16", 1: "sorted_gap_uleb"}

""",
        label="DQS1 constants",
    )

    start = out.index("def parse_pr101_frame_selector_archive(")
    end = out.index("\ndef apply_dynamic_mode(", start)
    replacement = r'''def parse_dqs1_payload(payload: bytes) -> dict[str, object]:
    if len(payload) < 11:
        raise ValueError("DQS1 payload truncated")
    if payload[:4] != DQS1_MAGIC:
        raise ValueError(f"DQS1 magic mismatch: {payload[:4]!r}")
    mode_byte = int(payload[4])
    frame_policy_code = mode_byte & 0x0F
    pair_encoding_code = mode_byte >> 4
    frame_policy = DQS1_FRAME_POLICY_BY_CODE.get(frame_policy_code)
    if frame_policy is None:
        raise ValueError(f"unsupported DQS1 frame policy code: {frame_policy_code}")
    pair_encoding = DQS1_PAIR_ENCODING_BY_CODE.get(pair_encoding_code)
    if pair_encoding is None:
        raise ValueError(f"unsupported DQS1 pair encoding code: {pair_encoding_code}")
    storage_index = int(payload[5])
    q_offset = struct.unpack_from("<H", payload, 6)[0]
    delta = struct.unpack_from("<b", payload, 8)[0]
    count = struct.unpack_from("<H", payload, 9)[0]

    def pack_uleb128(value: int) -> bytes:
        if value < 0:
            raise ValueError("DQS1 ULEB value must be non-negative")
        out = bytearray()
        remaining = int(value)
        while True:
            byte = remaining & 0x7F
            remaining >>= 7
            if remaining:
                out.append(byte | 0x80)
            else:
                out.append(byte)
                return bytes(out)

    def unpack_uleb128(offset: int) -> tuple[int, int]:
        value = 0
        shift = 0
        start = offset
        while offset < len(payload):
            byte = payload[offset]
            value |= (byte & 0x7F) << shift
            offset += 1
            if byte & 0x80 == 0:
                if payload[start:offset] != pack_uleb128(value):
                    raise ValueError("noncanonical DQS1 ULEB gap")
                return value, offset
            shift += 7
            if shift > 28:
                raise ValueError("DQS1 ULEB gap is too large")
        raise ValueError("truncated DQS1 ULEB gap")

    if pair_encoding == "raw_u16":
        expected_len = 11 + count * 2
        if len(payload) != expected_len:
            raise ValueError(f"DQS1 payload length mismatch: expected {expected_len}, got {len(payload)}")
        pair_indices = [
            int(struct.unpack_from("<H", payload, 11 + offset * 2)[0])
            for offset in range(count)
        ]
    elif pair_encoding == "sorted_gap_uleb":
        pair_indices = []
        previous = None
        offset = 11
        for _ in range(count):
            value, offset = unpack_uleb128(offset)
            if previous is not None and value <= 0:
                raise ValueError("DQS1 sorted_gap_uleb gap must be positive")
            pair_index = value if previous is None else previous + value
            if pair_index > 65535:
                raise ValueError("DQS1 selected pair outside u16 range")
            pair_indices.append(int(pair_index))
            previous = int(pair_index)
        if offset != len(payload):
            raise ValueError("DQS1 payload has trailing pair-index bytes")
    else:
        raise ValueError(f"unsupported DQS1 pair encoding {pair_encoding!r}")
    if pair_indices != sorted(pair_indices):
        raise ValueError("DQS1 selected pair indices must be sorted")
    if len(set(pair_indices)) != len(pair_indices):
        raise ValueError("DQS1 selected pair indices contain duplicates")
    for pair_index in pair_indices:
        if pair_index >= 600:
            raise ValueError(f"DQS1 selected pair outside FEC6 range: {pair_index}")
    return {
        "frame_policy": frame_policy,
        "frame_policy_code": frame_policy_code,
        "mode_byte": mode_byte,
        "pair_encoding": pair_encoding,
        "pair_encoding_code": pair_encoding_code,
        "storage_index": storage_index,
        "q_offset": q_offset,
        "delta": delta,
        "pair_indices": tuple(pair_indices),
    }


def parse_pr101_frame_selector_archive(
    bin_bytes: bytes,
) -> tuple[bytes, str, list[int], tuple[tuple[str, tuple[int, ...], int], ...], dict[str, object] | None]:
    if len(bin_bytes) < 10:
        raise ValueError("PR101 frame-selector wrapper truncated before header")
    magic = bin_bytes[:4]
    if magic != OUTER_MAGIC:
        raise ValueError(f"PR101 frame-selector magic mismatch: {magic!r}")
    pos = 4
    (source_len,) = struct.unpack_from("<I", bin_bytes, pos)
    pos += 4
    source_payload = bin_bytes[pos : pos + source_len]
    pos += source_len
    if len(source_payload) != source_len:
        raise ValueError("PR101 source payload truncated")
    if pos + 2 > len(bin_bytes):
        raise ValueError("PR101 frame-selector wrapper truncated before selector length")
    (selector_len,) = struct.unpack_from("<H", bin_bytes, pos)
    pos += 2
    selector_payload = bin_bytes[pos : pos + selector_len]
    pos += selector_len
    if len(selector_payload) != selector_len:
        raise ValueError("FES1 selector payload truncated")
    dqs1_packet = None
    if pos != len(bin_bytes):
        dqs1_packet = parse_dqs1_payload(bin_bytes[pos:])
        pos = len(bin_bytes)
    selector_kind, selector_codes, selector_specs = unpack_pr101_selector(selector_payload)
    return source_payload, selector_kind, selector_codes, selector_specs, dqs1_packet


def _tensor_index_for_stored_q_offset(q_offset: int, shape: tuple[int, ...], storage_index: int) -> tuple[int, ...]:
    if len(shape) == 4:
        storage_perm = tuple(int(axis) for axis in CONV4_STORAGE_PERMS[int(storage_index)])
        stored_shape = tuple(shape[axis] for axis in storage_perm)
        stored_index = np.unravel_index(int(q_offset), stored_shape)
        tensor_index = [0] * len(shape)
        for storage_axis, tensor_axis in enumerate(storage_perm):
            tensor_index[int(tensor_axis)] = int(stored_index[storage_axis])
        return tuple(tensor_index)
    return tuple(int(value) for value in np.unravel_index(int(q_offset), shape))


def apply_dqs1_patch_to_decoder_state(
    decoder_sd: dict[str, torch.Tensor],
    source_payload: bytes,
    dqs1_packet: dict[str, object],
    meta: dict[str, object],
) -> dict[str, torch.Tensor]:
    decoder_blob = source_payload[:DECODER_BLOB_LEN]
    raw = decompress_brotli_streams(decoder_blob, len(DECODER_STREAM_ENDS))
    probe = HNeRVDecoder(
        latent_dim=int(meta["latent_dim"]),
        base_channels=int(meta["base_channels"]),
        eval_size=tuple(meta["eval_size"]),
    )
    items = list(probe.state_dict().items())
    target_storage_index = int(dqs1_packet["storage_index"])
    q_offset = int(dqs1_packet["q_offset"])
    delta = int(dqs1_packet["delta"])
    pos = 0
    for storage_index in DECODER_STORAGE_ORDER:
        name, tensor = items[int(storage_index)]
        shape = tuple(int(value) for value in tuple(tensor.shape))
        numel = int(tensor.numel())
        q_start = pos
        q_end = q_start + numel
        scale_start = q_end
        scale_end = scale_start + 2
        if scale_end > len(raw):
            raise ValueError("DQS1 decoder raw stream truncated")
        if int(storage_index) != target_storage_index:
            pos = scale_end
            continue
        if q_offset < 0 or q_offset >= numel:
            raise ValueError(f"DQS1 q_offset {q_offset} outside tensor {name} numel={numel}")
        q_bytes = np.frombuffer(raw[q_start:q_end], dtype=np.uint8).copy()
        q_values = decode_mapped_u8(q_bytes, DECODER_BYTE_MAPS.get(int(storage_index), "zig"))
        q_before = int(q_values[q_offset])
        q_after = int(np.clip(q_before + delta, -127, 127))
        if q_after == q_before:
            raise ValueError("DQS1 q-domain mutation is clipped/no-op")
        scale = float(np.frombuffer(raw, dtype=np.float16, count=1, offset=scale_start)[0])
        tensor_index = _tensor_index_for_stored_q_offset(q_offset, shape, int(storage_index))
        out = {key: value.clone() for key, value in decoder_sd.items()}
        patched = out[name].clone()
        patched[tensor_index] = patched[tensor_index] + float(q_after - q_before) * scale
        out[name] = patched
        return out
    raise ValueError(f"DQS1 storage_index {target_storage_index} not found in decoder schema")

'''.rstrip()
    out = out[:start] + replacement + out[end:]

    out = _replace_one_of(
        out,
        [
            (
                "source_payload, selector_kind, selector_codes, selector_specs = parse_pr101_frame_selector_archive(Path(src_bin).read_bytes())\n"
                "    decoder_sd, latents, meta = parse_archive(source_payload)\n\n"
                "    device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")",
                "source_payload, selector_kind, selector_codes, selector_specs, dqs1_packet = parse_pr101_frame_selector_archive(Path(src_bin).read_bytes())\n"
                "    decoder_sd, latents, meta = parse_archive(source_payload)\n\n"
                "    device = torch.device(\"cuda\" if torch.cuda.is_available() else \"cpu\")",
            ),
            (
                "source_payload, selector_kind, selector_codes, selector_specs = parse_pr101_frame_selector_archive(src_path.read_bytes())\n"
                "    decoder_sd, latents, meta = parse_archive(source_payload)\n",
                "source_payload, selector_kind, selector_codes, selector_specs, dqs1_packet = parse_pr101_frame_selector_archive(src_path.read_bytes())\n"
                "    decoder_sd, latents, meta = parse_archive(source_payload)\n",
            ),
        ],
        label="inflate archive parser assignment",
    )
    out = _replace_once(
        out,
        "    decoder.load_state_dict(decoder_sd)\n"
        "    decoder.eval()\n\n"
        "    latents = latents.to(device)",
        "    decoder.load_state_dict(decoder_sd)\n"
        "    decoder.eval()\n\n"
        "    mutated_decoder = None\n"
        "    selected_pairs: set[int] = set()\n"
        "    frame_policy = \"pair_all_frames\"\n"
        "    if dqs1_packet is not None:\n"
        "        mutated_sd = apply_dqs1_patch_to_decoder_state(decoder_sd, source_payload, dqs1_packet, meta)\n"
        "        mutated_decoder = HNeRVDecoder(\n"
        "            latent_dim=meta[\"latent_dim\"],\n"
        "            base_channels=meta[\"base_channels\"],\n"
        "            eval_size=tuple(meta[\"eval_size\"]),\n"
        "        ).to(device)\n"
        "        mutated_decoder.load_state_dict(mutated_sd)\n"
        "        mutated_decoder.eval()\n"
        "        selected_pairs = {int(value) for value in dqs1_packet[\"pair_indices\"]}\n"
        "        frame_policy = str(dqs1_packet[\"frame_policy\"])\n\n"
        "    latents = latents.to(device)",
        label="mutated decoder construction",
    )
    out = _replace_once(
        out,
        "            decoded = decoder(latents[i:j])\n"
        "            flat = decoded.reshape(batch * 2, 3, eval_h, eval_w)",
        "            decoded = decoder(latents[i:j])\n"
        "            if mutated_decoder is not None:\n"
        "                selected_local = [pair - i for pair in range(i, j) if pair in selected_pairs]\n"
        "                if selected_local:\n"
        "                    selected_local_tensor = torch.tensor(\n"
        "                        selected_local,\n"
        "                        dtype=torch.long,\n"
        "                        device=decoded.device,\n"
        "                    )\n"
        "                    decoded = decoded.clone()\n"
        "                    mutated_decoded = mutated_decoder(latents[i:j])\n"
        "                    if frame_policy == \"pair_all_frames\":\n"
        "                        decoded[selected_local_tensor] = mutated_decoded[selected_local_tensor]\n"
        "                    elif frame_policy == \"segnet_last_frame_only\":\n"
        "                        decoded[selected_local_tensor, 1] = mutated_decoded[selected_local_tensor, 1]\n"
        "                    else:\n"
        "                        raise ValueError(f\"unsupported DQS1 frame policy {frame_policy!r}\")\n"
        "            flat = decoded.reshape(batch * 2, 3, eval_h, eval_w)",
        label="selective batch splice",
    )
    return out


def copy_submission_runtime(
    *,
    source_dir: Path,
    output_dir: Path,
    force: bool = False,
) -> None:
    if not source_dir.is_dir():
        raise DecoderQSelectiveRuntimeMaterializerError(
            f"base submission dir missing: {source_dir}"
        )
    if output_dir.exists():
        if not force:
            raise DecoderQSelectiveRuntimeMaterializerError(
                f"output dir already exists: {output_dir}"
            )
        shutil.rmtree(output_dir)
    ignore = shutil.ignore_patterns("__pycache__", "*.pyc")
    shutil.copytree(source_dir, output_dir, ignore=ignore)


def materialize_selective_runtime_candidate(
    *,
    plan_path: Path,
    base_submission_dir: Path,
    base_archive: Path,
    output_dir: Path,
    repo_root: Path,
    frame_policy: str = "pair_all_frames",
    max_units: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Create a byte-closed selective runtime submission directory."""

    base_archive = _resolve_path(base_archive, repo_root=repo_root)
    base_submission_dir = _resolve_path(base_submission_dir, repo_root=repo_root)
    output_dir = _resolve_path(output_dir, repo_root=repo_root)
    packet_plan = load_or_build_packet_plan(
        _resolve_path(plan_path, repo_root=repo_root),
        base_archive=base_archive,
        repo_root=repo_root,
        frame_policy=frame_policy,
        max_units=max_units,
    )
    dqs1_payload = dqs1_payload_from_packet_plan(packet_plan)
    parsed_dqs1 = parse_dqs1_payload(dqs1_payload)
    base_member = read_single_stored_member(base_archive)
    candidate_member = base_member.data + dqs1_payload

    copy_submission_runtime(
        source_dir=base_submission_dir,
        output_dir=output_dir,
        force=force,
    )
    write_single_stored_member(
        output_dir / "archive.zip",
        member_name=base_member.name,
        data=candidate_member,
    )
    base_inflate = (base_submission_dir / "inflate.py").read_text(encoding="utf-8")
    (output_dir / "inflate.py").write_text(
        build_selective_inflate_py(base_inflate),
        encoding="utf-8",
    )
    write_json(output_dir / "decoder_q_selective_runtime_packet_plan.json", packet_plan)

    materialized_member = read_single_stored_member(output_dir / "archive.zip")
    manifest = {
        "schema": MATERIALIZER_SCHEMA,
        "producer": TOOL,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        **FALSE_AUTHORITY,
        "candidate_generation_only": True,
        "requires_exact_auth_eval_before_score_claim": True,
        "evidence_grade": "macOS-MLX-research-signal",
        "allowed_use": "non_authoritative_selective_runtime_probe",
        "plan_path": str(_resolve_path(plan_path, repo_root=repo_root)),
        "base_submission_dir": str(base_submission_dir),
        "output_submission_dir": str(output_dir),
        "base_archive": {
            "path": str(base_archive),
            **base_member.as_dict(),
        },
        "materialized_archive": {
            "path": str(output_dir / "archive.zip"),
            **materialized_member.as_dict(),
        },
        "dqs1_payload": parsed_dqs1,
        "archive_member_delta": {
            "base_member_bytes": len(base_member.data),
            "dqs1_payload_bytes": len(dqs1_payload),
            "materialized_member_bytes": len(candidate_member),
            "all_selective_bytes_inside_member_x": True,
            "external_sidecars_required_at_inflate_time": False,
        },
        "runtime_contract": {
            "copies_fec6_runtime": True,
            "modified_inflate_py": True,
            "derives_mutated_decoder_from_base_decoder_and_dqs1_patch": True,
            "stores_second_full_decoder": False,
            "selector_applied_after_selective_frame_stitching": True,
        },
        "dispatch_blockers": [
            "official inflate.sh raw-output locality control not run",
            "local advisory scorer not run on selective packet",
            "exact contest auth eval not run",
        ],
    }
    write_json(output_dir / "selective_runtime_manifest.json", manifest)
    return manifest


__all__ = [
    "MATERIALIZER_SCHEMA",
    "TOOL",
    "DecoderQSelectiveRuntimeMaterializerError",
    "affected_frames_for_pairs",
    "build_selective_inflate_py",
    "dqs1_payload_from_packet_plan",
    "load_or_build_packet_plan",
    "materialize_selective_runtime_candidate",
    "parse_dqs1_payload",
    "read_single_stored_member",
    "write_single_stored_member",
]
