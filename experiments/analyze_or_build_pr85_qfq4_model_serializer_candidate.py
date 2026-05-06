#!/usr/bin/env python3
"""Fail-closed PR85 probe for a PR90/QFQ4-style model serializer transfer.

This tool is intentionally local-only. It may write a candidate archive only
when a PR85 model segment lowered into the QFQ4-style byte order has exact
decoded tensor parity and the existing PR85 no-edit runtime can load that
format. Otherwise it records the precise blockers and leaves exact eval locked.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.pr85_bundle import (  # noqa: E402
    pack_pr85_bundle,
    parse_pr85_bundle,
    validate_pr85_member_name,
)
from tac.qh0_record_serializer import (  # noqa: E402
    QH0Record,
    parse_qh0_record_set,
    record_set_summary,
    sha256_bytes,
)
from tac.qh0_renderer_codec import FP4_POS_LEVELS, decode_qh0_state_dict  # noqa: E402
from tac.quantizr_faithful_renderer import build_quantizr_faithful_renderer  # noqa: E402


TOOL = "experiments/analyze_or_build_pr85_qfq4_model_serializer_candidate.py"
SCHEMA = "pr85_qfq4_model_serializer_probe_v1"
MANIFEST_SCHEMA = "pr85_qfq4_model_serializer_candidate_v1"
BLOCKER_SCHEMA = "pr85_qfq4_model_serializer_dispatch_blocker_v1"
DEFAULT_PR85_DIR = REPO_ROOT / "experiments/results/public_pr85_intake_20260503_codex"
DEFAULT_PR90_DIR = REPO_ROOT / "experiments/results/public_pr90_intake_20260504_worker"
DEFAULT_ARCHIVE = DEFAULT_PR85_DIR / "archive.zip"
DEFAULT_REPLAY_INFLATE = DEFAULT_PR85_DIR / "replay_submission/inflate.py"
DEFAULT_PR90_PAYLOAD_PROBE = DEFAULT_PR90_DIR / "payload_probe.json"
DEFAULT_PR90_FLAT_FP4_CODEC = (
    DEFAULT_PR90_DIR / "pr90_src/submissions/qrepro/flat_fp4_codec.py"
)
DEFAULT_PR90_INFLATE = DEFAULT_PR90_DIR / "pr90_src/submissions/qrepro/inflate.py"
DEFAULT_OUT_DIR = (
    REPO_ROOT / "experiments/results/pr85_qfq4_model_serializer_probe_20260504_worker"
)
DEFAULT_ROBUST_CURRENT = REPO_ROOT / "submissions/robust_current"
ORIGINAL_VIDEO_BYTES = 37_545_489
QFQ4_MAGIC = b"QFQ4\0"
FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
KNOWN_PUBLIC_PR85 = {
    "archive_bytes": 236_328,
    "archive_sha256": "eb18df2f1b364e513f36933116ec0c1011c6d3bbf8022e16495e0508c6258f5e",
}
WORKER_G_PLANNING_ONLY = {
    "candidate_id": "pr90_qfq4_style_pr85_model_serializer_probe",
    "estimated_delta_bytes": -689,
    "estimated_rate_score_delta": -0.000458776819,
    "required_before_dispatch": ["decoded_model_tensor_parity", "pr85_runtime_output_parity"],
}


class QFQ4ProbeError(RuntimeError):
    """Raised when the transfer probe cannot produce a valid local artifact."""


@dataclass(frozen=True)
class QFQ4ProbeVariant:
    """One QFQ4-style lowering attempt for the PR85 model records."""

    variant_id: str
    raw_payload: bytes
    inner_brotli_payload: bytes
    outer_model_segment: bytes
    qrow_policy: str
    qrow_rows_quantized: int
    qrow_rows_fp16: int
    qrow_contract_note: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _repo_rel(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _read_json(path: Path, *, required: bool = True) -> dict[str, Any]:
    if not path.is_file():
        if required:
            raise QFQ4ProbeError(f"required JSON artifact not found: {_repo_rel(path)}")
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.is_file() else ""


def _brotli_compress(data: bytes, *, quality: int = 11, lgwin: int = 24) -> bytes:
    try:
        import brotli
    except ImportError as exc:  # pragma: no cover - environment guard
        raise QFQ4ProbeError("brotli is required for PR85 QFQ4 probe") from exc
    return brotli.compress(data, quality=int(quality), lgwin=int(lgwin))


def _brotli_decompress(data: bytes, *, label: str) -> bytes:
    try:
        import brotli
    except ImportError as exc:  # pragma: no cover - environment guard
        raise QFQ4ProbeError("brotli is required for PR85 QFQ4 probe") from exc
    try:
        return brotli.decompress(data)
    except brotli.error as exc:
        raise QFQ4ProbeError(f"{label} is not Brotli-decodable") from exc


def _read_single_member_archive(path: Path, *, expected_name: str) -> tuple[dict[str, Any], bytes]:
    if not path.is_file():
        raise QFQ4ProbeError(f"archive not found: {_repo_rel(path)}")
    with zipfile.ZipFile(path, "r") as zf:
        infos = [info for info in zf.infolist() if not info.is_dir()]
        names = [info.filename for info in infos]
        if names != [expected_name]:
            raise QFQ4ProbeError(
                f"archive must contain exactly [{expected_name!r}]; got {names!r}"
            )
        info = infos[0]
        if expected_name == "x":
            validate_pr85_member_name(info.filename)
        raw = zf.read(info)
    archive_sha = _sha256_file(path)
    return (
        {
            "path": _repo_rel(path),
            "archive_bytes": int(path.stat().st_size),
            "archive_sha256": archive_sha,
            "member_name": info.filename,
            "member_file_size": int(info.file_size),
            "member_compress_size": int(info.compress_size),
            "member_crc32_hex": f"{info.CRC:08x}",
            "member_sha256": sha256_bytes(raw),
            "zip_compress_type": int(info.compress_type),
        },
        raw,
    )


def _source_header_mode(bundle_format: str) -> str:
    return "explicit_30" if bundle_format == "pr85_explicit_30byte_lengths" else "v5"


def _zip_info_x() -> zipfile.ZipInfo:
    info = zipfile.ZipInfo("x", FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_STORED
    info.external_attr = 0o644 << 16
    info.create_system = 3
    return info


def _write_single_x_archive(path: Path, x_payload: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(_zip_info_x(), x_payload, compress_type=zipfile.ZIP_STORED)
    return _read_single_member_archive(path, expected_name="x")[0]


def _module_weight_order(model: torch.nn.Module) -> list[tuple[str, torch.nn.Module]]:
    ordered: list[tuple[str, torch.nn.Module]] = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Embedding)):
            ordered.append((name, module))
    return ordered


def _record_bytes(record: QH0Record, *, skip_kind: bool = True) -> bytes:
    return record.direct_record[1:] if skip_kind else record.direct_record


def _planed_fp16_bytes(raw: bytes) -> bytes:
    if len(raw) % 2:
        raise QFQ4ProbeError(f"QFQ4 fp16 byte plane must have even length, got {len(raw)}")
    return raw[1::2] + raw[0::2]


def _unplane_fp16_bytes(planed: bytes) -> bytes:
    if len(planed) % 2:
        raise QFQ4ProbeError(f"QFQ4 planed fp16 byte stream has odd length {len(planed)}")
    half = len(planed) // 2
    out = bytearray(len(planed))
    out[1::2] = planed[:half]
    out[0::2] = planed[half:]
    return bytes(out)


def _pack_little_bit_mask(bits: np.ndarray) -> bytes:
    return np.packbits(bits.astype(np.uint8), bitorder="little").tobytes()


def _unpack_little_bit_mask(raw: bytes, rows: int) -> np.ndarray:
    return np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little")[:rows].astype(bool)


def _unpack_nibbles_np(packed: np.ndarray, count: int) -> np.ndarray:
    hi = (packed >> 4) & 0x0F
    lo = packed & 0x0F
    out = np.empty(packed.size * 2, dtype=np.uint8)
    out[0::2] = hi
    out[1::2] = lo
    return out[:count]


def _dequant_fp4_np(
    nibbles: np.ndarray,
    scales: np.ndarray,
    shape: Sequence[int],
    codebook: np.ndarray,
) -> torch.Tensor:
    flat_n = int(np.prod(shape))
    if scales.size == 0:
        raise QFQ4ProbeError("cannot dequantize FP4 tensor with zero scales")
    block_size = nibbles.size // scales.size
    nib = nibbles.reshape(-1, block_size).astype(np.int64)
    signs = (nib >> 3).astype(np.int64)
    mag = (nib & 0x7).astype(np.int64)
    levels = codebook.astype(np.float32)
    q = levels[mag]
    q = np.where(signs.astype(bool), -q, q)
    dq = q * scales.astype(np.float32)[:, None]
    return torch.from_numpy(dq.reshape(-1)[:flat_n].reshape(tuple(shape)).copy()).float()


def _records_by_name(records: Sequence[QH0Record]) -> dict[str, QH0Record]:
    out: dict[str, QH0Record] = {}
    for record in records:
        if record.name in out:
            raise QFQ4ProbeError(f"duplicate QH0 record name: {record.name}")
        out[record.name] = record
    return out


def _collect_pr85_qfq4_layout(record_map: Mapping[str, QH0Record]) -> dict[str, Any]:
    model = build_quantizr_faithful_renderer()
    packed_modules: list[dict[str, Any]] = []
    fp16_modules: list[dict[str, Any]] = []
    bias_modules: list[dict[str, Any]] = []
    covered: set[str] = set()

    for name, module in _module_weight_order(model):
        key = f"{name}.weight"
        record = record_map.get(key)
        if record is None:
            raise QFQ4ProbeError(f"missing QH0 module weight record: {key}")
        shape = tuple(int(x) for x in module.weight.shape)
        numel = int(module.weight.numel())
        if tuple(record.tensor_shape) != shape:
            raise QFQ4ProbeError(
                f"QH0 record shape mismatch for {key}: {record.tensor_shape} != {shape}"
            )
        if record.record_kind == "fp4":
            blocks = (numel + 31) // 32
            packed_len = blocks * 32 // 2
            scales_len = blocks * 2
            body = _record_bytes(record)
            if len(body) != packed_len + scales_len:
                raise QFQ4ProbeError(
                    f"bad FP4 record length for {key}: got={len(body)} "
                    f"expected={packed_len + scales_len}"
                )
            packed_modules.append(
                {
                    "name": name,
                    "shape": shape,
                    "numel": numel,
                    "packed_len": packed_len,
                    "scales_len": scales_len,
                    "packed": body[:packed_len],
                    "scales": body[packed_len:],
                }
            )
        elif record.record_kind == "fp16":
            body = _record_bytes(record)
            if len(body) != numel * 2:
                raise QFQ4ProbeError(f"bad FP16 record length for {key}")
            fp16_modules.append({"name": name, "shape": shape, "numel": numel, "data": body})
        else:
            raise QFQ4ProbeError(
                f"QFQ4 transfer cannot encode module weight {key} kind {record.record_kind}"
            )
        covered.add(key)

        bias = getattr(module, "bias", None)
        if bias is not None:
            bias_key = f"{name}.bias"
            bias_record = record_map.get(bias_key)
            if bias_record is None:
                raise QFQ4ProbeError(f"missing QH0 module bias record: {bias_key}")
            if bias_record.record_kind != "fp16_bias":
                raise QFQ4ProbeError(
                    f"QFQ4 transfer cannot encode module bias {bias_key} kind "
                    f"{bias_record.record_kind}"
                )
            body = _record_bytes(bias_record, skip_kind=False)
            shape = tuple(int(x) for x in bias.shape)
            if len(body) != int(bias.numel()) * 2:
                raise QFQ4ProbeError(f"bad FP16 bias record length for {bias_key}")
            bias_modules.append({"name": name, "shape": shape, "numel": int(bias.numel()), "data": body})
            covered.add(bias_key)

    special = "frame1_head.block1.film_proj.weight"
    omitted_zero_init = {
        "frame2_head.block1.film_proj.weight",
        "frame2_head.block1.film_proj.bias",
    }
    dense_float: list[dict[str, Any]] = []
    dense_int: list[dict[str, Any]] = []
    special_record: QH0Record | None = None
    for name, tensor in model.state_dict().items():
        if name in covered or name in omitted_zero_init:
            continue
        record = record_map.get(name)
        if record is None:
            raise QFQ4ProbeError(f"missing QH0 dense record: {name}")
        if name == special:
            special_record = record
            continue
        if record.record_kind == "fp16_dense":
            dense_float.append(
                {
                    "name": name,
                    "shape": tuple(int(x) for x in tensor.shape),
                    "numel": int(tensor.numel()),
                    "data": _record_bytes(record),
                }
            )
        else:
            dense_int.append(
                {
                    "name": name,
                    "shape": tuple(int(x) for x in tensor.shape),
                    "numel": int(tensor.numel()),
                    "record_kind": record.record_kind,
                    "data": _record_bytes(record),
                }
            )
    if special_record is None:
        raise QFQ4ProbeError("PR85 model is missing frame1 FiLM special tensor")
    return {
        "model": model,
        "packed_modules": packed_modules,
        "fp16_modules": fp16_modules,
        "bias_modules": bias_modules,
        "dense_float": dense_float,
        "dense_int": dense_int,
        "special_record": special_record,
        "omitted_zero_init": sorted(omitted_zero_init),
    }


def build_pr85_qfq4_probe_variant(
    record_map: Mapping[str, QH0Record],
    *,
    qrow_policy: str,
) -> QFQ4ProbeVariant:
    """Lower PR85 QH0 records into a PR90/QFQ4-style grouped byte payload."""

    layout = _collect_pr85_qfq4_layout(record_map)
    packed_modules = layout["packed_modules"]
    packed_order = sorted(packed_modules, key=lambda item: (".pw" in item["name"], item["name"]))
    special: QH0Record = layout["special_record"]
    if special.record_kind != "int8_row_scale":
        raise QFQ4ProbeError(
            f"QFQ4 probe only supports PR85 special int8 row-scale tensor; got {special.record_kind}"
        )
    rows, cols = (int(special.tensor_shape[0]), int(special.tensor_shape[1]))
    special_body = _record_bytes(special)
    q_i8_len = rows * cols
    q_i8 = np.frombuffer(special_body[:q_i8_len], dtype=np.int8).copy().reshape(rows, cols)
    special_scales = np.frombuffer(special_body[q_i8_len:], dtype="<f2").copy()
    if special_scales.shape[0] != rows:
        raise QFQ4ProbeError("PR85 special int8 row-scale record has wrong scale count")

    if qrow_policy == "shifted_int8_rows":
        q_bits = np.ones(rows, dtype=bool)
        qrow_q = (q_i8.astype(np.int16) + 128).astype(np.uint8).tobytes()
        qrow_min = (-128.0 * special_scales.astype(np.float32)).astype("<f2").tobytes()
        qrow_scale = special_scales.astype("<f2").tobytes()
        qrow_fp = b""
        note = (
            "Stores every PR85 int8 row-scale row as QFQ4 uint8 rows with "
            "min=-128*scale; PR90 QFQ4 then casts reconstructed rows to fp16."
        )
    elif qrow_policy == "all_fp16_rows":
        source_state, _ = decode_qh0_state_dict(
            b"QH0" + b"".join(record.qh0_record for record in record_map.values()),
            device="cpu",
        )
        # The source_state path above depends on insertion order, so use the
        # reviewed decoder on the record-set source in the caller when possible.
        special_tensor = source_state.get("frame1_head.block1.film_proj.weight")
        if special_tensor is None:
            raise QFQ4ProbeError("decoded source state is missing special FiLM tensor")
        q_bits = np.zeros(rows, dtype=bool)
        qrow_q = b""
        qrow_min = b""
        qrow_scale = b""
        qrow_fp = special_tensor.detach().cpu().numpy().astype("<f2").tobytes()
        note = "Stores every PR85 special row through QFQ4's fp16 row fallback."
    else:
        raise QFQ4ProbeError(f"unsupported qrow policy: {qrow_policy!r}")

    mask_bytes = _pack_little_bit_mask(q_bits)
    fp16_bytes = b"".join(item["scales"] for item in packed_order)
    fp16_bytes += b"".join(item["data"] for item in layout["dense_float"])
    fp16_bytes += b"".join(item["data"] for item in layout["bias_modules"])
    fp16_bytes += b"".join(item["data"] for item in layout["fp16_modules"])
    fp16_bytes += qrow_min + qrow_scale + qrow_fp
    raw_payload = b"".join(item["packed"] for item in packed_order)
    raw_payload += _planed_fp16_bytes(fp16_bytes)
    raw_payload += qrow_q + mask_bytes
    raw_payload += b"".join(item["data"] for item in reversed(layout["dense_int"]))
    raw_payload += FP4_POS_LEVELS.detach().cpu().numpy().astype("<f4").tobytes()
    inner = _brotli_compress(raw_payload, quality=11, lgwin=24)
    outer = _brotli_compress(QFQ4_MAGIC + inner, quality=11, lgwin=24)
    return QFQ4ProbeVariant(
        variant_id=f"qfq4_pr85_{qrow_policy}",
        raw_payload=raw_payload,
        inner_brotli_payload=inner,
        outer_model_segment=outer,
        qrow_policy=qrow_policy,
        qrow_rows_quantized=int(q_bits.sum()),
        qrow_rows_fp16=int(rows - q_bits.sum()),
        qrow_contract_note=note,
    )


def decode_pr85_qfq4_probe_payload(blob: bytes, *, device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    """Decode the local PR85 QFQ4-style payload with PR90's grouped contract."""

    raw = _brotli_decompress(blob, label="QFQ4 inner model body")
    pay = memoryview(raw)
    offset = 0
    model = build_quantizr_faithful_renderer()
    packed_modules: list[tuple[str, tuple[int, ...], int, int]] = []
    fp16_modules: list[tuple[str, tuple[int, ...], int]] = []
    bias_modules: list[tuple[str, int]] = []
    covered: set[str] = set()
    for name, module in _module_weight_order(model):
        shape = tuple(int(x) for x in module.weight.shape)
        weight_numel = int(module.weight.numel())
        fp16_weight = isinstance(module, torch.nn.Embedding) or name.endswith(".head")
        if fp16_weight:
            fp16_modules.append((name, shape, weight_numel))
        else:
            nblocks = (weight_numel + 31) // 32
            packed_modules.append((name, shape, nblocks * 32 // 2, nblocks * 2))
        covered.add(f"{name}.weight")
        if isinstance(module, torch.nn.Conv2d) and module.bias is not None:
            bias_modules.append((name, int(module.bias.numel())))
            covered.add(f"{name}.bias")

    packed_order = sorted(packed_modules, key=lambda item: (".pw" in item[0], item[0]))
    packed_bytes: dict[str, np.ndarray] = {}
    for name, _shape, packed_len, _scales_len in packed_order:
        packed_bytes[name] = np.frombuffer(pay[offset : offset + packed_len], dtype="<u1").copy()
        offset += packed_len

    special = "frame1_head.block1.film_proj.weight"
    special_shape: tuple[int, int] | None = None
    dense_float: list[tuple[str, tuple[int, ...], int]] = []
    dense_int: list[tuple[str, tuple[int, ...], int]] = []
    omitted_zero_init = {
        "frame2_head.block1.film_proj.weight",
        "frame2_head.block1.film_proj.bias",
    }
    for name, tensor in model.state_dict().items():
        if name in covered or name in omitted_zero_init:
            continue
        if name == special:
            special_shape = tuple(int(x) for x in tensor.shape)  # type: ignore[assignment]
        elif torch.is_floating_point(tensor):
            dense_float.append((name, tuple(int(x) for x in tensor.shape), int(tensor.numel())))
        else:
            dense_int.append((name, tuple(int(x) for x in tensor.shape), int(tensor.numel())))
    if special_shape is None:
        raise QFQ4ProbeError("QFQ4 requires frame1 FiLM qrow tensor")

    rows, cols = special_shape
    mask_nbytes = (rows + 7) // 8
    known_fp16 = (
        sum(scales_len for _name, _shape, _packed_len, scales_len in packed_modules)
        + sum(n_bias * 2 for _name, n_bias in bias_modules)
        + sum(weight_numel * 2 for _name, _shape, weight_numel in fp16_modules)
        + sum(numel * 2 for _name, _shape, numel in dense_float)
    )
    known_raw = 32 + sum(numel * 8 for _name, _shape, numel in dense_int) + mask_nbytes
    remaining = len(pay) - offset
    const_total = known_fp16 + known_raw + rows * cols * 2
    denom = cols - 4
    if denom <= 0 or (const_total - remaining) % denom:
        raise QFQ4ProbeError("invalid QFQ4 grouped lengths")
    n_q = (const_total - remaining) // denom
    if n_q < 0 or n_q > rows:
        raise QFQ4ProbeError("invalid QFQ4 qrow count")
    n_fp = rows - n_q
    fp16_len = known_fp16 + n_q * 4 + n_fp * cols * 2
    raw_len = known_raw + n_q * cols
    if fp16_len + raw_len != remaining:
        raise QFQ4ProbeError("invalid QFQ4 length split")

    fp16_bytes = _unplane_fp16_bytes(bytes(pay[offset : offset + fp16_len]))
    offset += fp16_len
    fp16_pay = memoryview(fp16_bytes)
    fp16_offset = 0
    raw_offset = offset

    qrow_q = np.frombuffer(pay[raw_offset : raw_offset + n_q * cols], dtype=np.uint8).astype(
        np.float32
    ).reshape(n_q, cols)
    raw_offset += n_q * cols
    mask_bytes = bytes(pay[raw_offset : raw_offset + mask_nbytes])
    raw_offset += mask_nbytes
    dense_int_bytes: dict[str, np.ndarray] = {}
    for name, shape, numel in reversed(dense_int):
        nbytes = numel * 8
        dense_int_bytes[name] = np.frombuffer(pay[raw_offset : raw_offset + nbytes], dtype="<i8").copy().reshape(shape)
        raw_offset += nbytes
    codebook = np.frombuffer(pay[raw_offset : raw_offset + 32], dtype="<f4").copy()
    raw_offset += 32
    if raw_offset != len(pay):
        raise QFQ4ProbeError("QFQ4 raw group length mismatch")

    bits = _unpack_little_bit_mask(mask_bytes, rows)
    if int(bits.sum()) != n_q:
        raise QFQ4ProbeError("QFQ4 qrow mask count mismatch")

    scale_bytes: dict[str, np.ndarray] = {}
    for name, _shape, _packed_len, scales_len in packed_order:
        scale_bytes[name] = np.frombuffer(
            fp16_pay[fp16_offset : fp16_offset + scales_len], dtype="<f2"
        ).copy()
        fp16_offset += scales_len

    dense_float_bytes: dict[str, np.ndarray] = {}
    for name, shape, numel in dense_float:
        nbytes = numel * 2
        dense_float_bytes[name] = np.frombuffer(
            fp16_pay[fp16_offset : fp16_offset + nbytes], dtype="<f2"
        ).copy().reshape(shape)
        fp16_offset += nbytes

    bias_bytes: dict[str, np.ndarray] = {}
    for name, n_bias in bias_modules:
        nbytes = n_bias * 2
        bias_bytes[name] = np.frombuffer(
            fp16_pay[fp16_offset : fp16_offset + nbytes], dtype="<f2"
        ).copy()
        fp16_offset += nbytes

    fp16_weight_bytes: dict[str, np.ndarray] = {}
    for name, shape, weight_numel in fp16_modules:
        nbytes = weight_numel * 2
        fp16_weight_bytes[name] = np.frombuffer(
            fp16_pay[fp16_offset : fp16_offset + nbytes], dtype="<f2"
        ).copy().reshape(shape)
        fp16_offset += nbytes

    qrow_min = np.frombuffer(fp16_pay[fp16_offset : fp16_offset + n_q * 2], dtype="<f2").astype(
        np.float32
    )
    fp16_offset += n_q * 2
    qrow_scale = np.frombuffer(
        fp16_pay[fp16_offset : fp16_offset + n_q * 2], dtype="<f2"
    ).astype(np.float32)
    fp16_offset += n_q * 2
    qrow_fp = np.frombuffer(
        fp16_pay[fp16_offset : fp16_offset + n_fp * cols * 2], dtype="<f2"
    ).copy().reshape(n_fp, cols)
    fp16_offset += n_fp * cols * 2
    if fp16_offset != len(fp16_pay):
        raise QFQ4ProbeError("QFQ4 fp16 group length mismatch")

    state_dict: dict[str, torch.Tensor] = {}
    for name, shape, _packed_len, _scales_len in packed_modules:
        packed = packed_bytes[name]
        scales = scale_bytes[name]
        nibbles = _unpack_nibbles_np(packed, packed.shape[0] * 2)
        state_dict[f"{name}.weight"] = _dequant_fp4_np(nibbles, scales, shape, codebook).to(device)
    for name, arr in fp16_weight_bytes.items():
        state_dict[f"{name}.weight"] = torch.from_numpy(arr).float().to(device)
    for name, arr in bias_bytes.items():
        state_dict[f"{name}.bias"] = torch.from_numpy(arr).float().to(device)
    for name, arr in dense_float_bytes.items():
        state_dict[name] = torch.from_numpy(arr).float().to(device)
    for name, arr in dense_int_bytes.items():
        state_dict[name] = torch.from_numpy(arr).to(device)

    qrow = np.empty((rows, cols), dtype=np.float16)
    qrow[bits] = (qrow_q * qrow_scale[:, None] + qrow_min[:, None]).astype(np.float16)
    qrow[~bits] = qrow_fp
    state_dict[special] = torch.from_numpy(qrow).float().to(device)
    return state_dict


def compare_state_dicts(
    source_state: Mapping[str, torch.Tensor],
    candidate_state: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    source_keys = set(source_state)
    candidate_keys = set(candidate_state)
    if source_keys != candidate_keys:
        mismatches.append(
            {
                "kind": "key_set",
                "missing": sorted(source_keys - candidate_keys),
                "extra": sorted(candidate_keys - source_keys),
            }
        )
    for key in sorted(source_keys & candidate_keys):
        left = source_state[key].detach().cpu()
        right = candidate_state[key].detach().cpu()
        if tuple(left.shape) != tuple(right.shape):
            mismatches.append(
                {
                    "kind": "shape",
                    "name": key,
                    "source_shape": list(left.shape),
                    "candidate_shape": list(right.shape),
                }
            )
            continue
        if left.dtype != right.dtype:
            mismatches.append(
                {
                    "kind": "dtype",
                    "name": key,
                    "source_dtype": str(left.dtype),
                    "candidate_dtype": str(right.dtype),
                }
            )
            continue
        if not torch.equal(left, right):
            diff = (left.float() - right.float()).abs()
            mismatches.append(
                {
                    "kind": "value",
                    "name": key,
                    "max_abs_diff": float(diff.max().item()) if diff.numel() else 0.0,
                    "changed_elements": int((diff != 0).sum().item()),
                    "numel": int(diff.numel()),
                }
            )
            if len(mismatches) >= 16:
                break
    return {
        "decoded_tensor_parity": not mismatches,
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "source_tensor_count": len(source_state),
        "candidate_tensor_count": len(candidate_state),
    }


def qfq4_runtime_compatibility(
    *,
    replay_inflate_py: Path = DEFAULT_REPLAY_INFLATE,
    robust_current_dir: Path = DEFAULT_ROBUST_CURRENT,
) -> dict[str, Any]:
    replay_text = _read_text(replay_inflate_py)
    robust_renderer_text = _read_text(robust_current_dir / "inflate_renderer.py")
    robust_unpacker_text = _read_text(robust_current_dir / "unpack_renderer_payload.py")
    replay_has_qfq4_loader = (
        "QFQ4" in replay_text
        or "QROW_GROUPED4_MODEL_BR_MAGIC" in replay_text
        or "decode_qrow_grouped4_payload_for_model" in replay_text
    )
    replay_single_x = "load_compact_archive_bundle" in replay_text and 'data_dir / "x"' in replay_text
    robust_has_qfq4_loader = (
        "QFQ4" in robust_renderer_text
        or "QROW_GROUPED4_MODEL_BR_MAGIC" in robust_renderer_text
        or "decode_qrow_grouped4_payload_for_model" in robust_renderer_text
    )
    robust_single_x = (
        'path = data_dir / "x"' in robust_unpacker_text
        or 'ARCHIVE_DIR/x' in robust_unpacker_text
        or '"/x"' in robust_unpacker_text
    )
    blockers: list[str] = []
    if not replay_has_qfq4_loader:
        blockers.append("public_pr85_replay_missing_QFQ4_model_loader")
    if not replay_single_x:
        blockers.append("public_pr85_replay_missing_single_x_loader")
    if not robust_has_qfq4_loader:
        blockers.append("robust_current_missing_QFQ4_renderer_loader")
    if not robust_single_x:
        blockers.append("robust_current_missing_pr85_single_x_unpacker")
    runtime_can_decode_without_edits = replay_has_qfq4_loader and replay_single_x
    return {
        "qfq4_magic": QFQ4_MAGIC.decode("ascii", errors="replace"),
        "runtime_can_decode_without_edits": runtime_can_decode_without_edits,
        "dispatch_unlocked": runtime_can_decode_without_edits,
        "public_pr85_replay_inflate_py": _repo_rel(replay_inflate_py),
        "public_pr85_replay_qfq4_model_loader": replay_has_qfq4_loader,
        "public_pr85_replay_single_x_loader": replay_single_x,
        "robust_current_dir": _repo_rel(robust_current_dir),
        "robust_current_qfq4_renderer_loader": robust_has_qfq4_loader,
        "robust_current_single_x_unpacker": robust_single_x,
        "blockers": blockers,
        "blocker_class": None if runtime_can_decode_without_edits else "runtime_incompatibility",
        "minimal_runtime_implementation_needed": (
            None
            if runtime_can_decode_without_edits
            else "Add a no-sidecar PR85 single-x QFQ4 model loader and prove output parity."
        ),
    }


def pr90_source_evidence(
    *,
    payload_probe_json: Path = DEFAULT_PR90_PAYLOAD_PROBE,
    pr90_inflate_py: Path = DEFAULT_PR90_INFLATE,
    pr90_flat_fp4_codec: Path = DEFAULT_PR90_FLAT_FP4_CODEC,
) -> dict[str, Any]:
    payload_probe = _read_json(payload_probe_json, required=False)
    inflate_text = _read_text(pr90_inflate_py)
    codec_text = _read_text(pr90_flat_fp4_codec)
    model_slice = {}
    slices = payload_probe.get("slices")
    if isinstance(slices, dict) and isinstance(slices.get("model_body"), dict):
        model_slice = dict(slices["model_body"])
    return {
        "payload_probe_json": _repo_rel(payload_probe_json),
        "pr90_inflate_py": _repo_rel(pr90_inflate_py),
        "pr90_flat_fp4_codec": _repo_rel(pr90_flat_fp4_codec),
        "payload_probe_present": bool(payload_probe),
        "payload_sha256": payload_probe.get("payload_sha256"),
        "payload_len": payload_probe.get("payload_len"),
        "split_mode": payload_probe.get("split_mode"),
        "model_body_slice": model_slice,
        "model_decode": payload_probe.get("model_decode", {}),
        "pr90_runtime_has_qfq4_branch": (
            "QROW_GROUPED4_MODEL_BR_MAGIC" in inflate_text
            and "decode_qrow_grouped4_payload_for_model" in inflate_text
        ),
        "pr90_codec_has_qfq4_decoder": "def decode_qrow_grouped4_payload_for_model" in codec_text,
    }


def _candidate_row(
    *,
    variant: QFQ4ProbeVariant,
    source_model_segment_bytes: int,
    parity: Mapping[str, Any],
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    inner_delta = len(variant.inner_brotli_payload) - source_model_segment_bytes
    outer_delta = len(variant.outer_model_segment) - source_model_segment_bytes
    parity_passed = bool(parity.get("decoded_tensor_parity"))
    runtime_passed = bool(runtime.get("runtime_can_decode_without_edits"))
    return {
        "candidate_id": variant.variant_id,
        "qrow_policy": variant.qrow_policy,
        "qrow_contract_note": variant.qrow_contract_note,
        "qrow_rows_quantized": variant.qrow_rows_quantized,
        "qrow_rows_fp16": variant.qrow_rows_fp16,
        "qfq4_raw_payload_bytes": len(variant.raw_payload),
        "qfq4_raw_payload_sha256": sha256_bytes(variant.raw_payload),
        "qfq4_inner_brotli_bytes": len(variant.inner_brotli_payload),
        "qfq4_inner_brotli_sha256": sha256_bytes(variant.inner_brotli_payload),
        "candidate_outer_pr85_model_segment_bytes": len(variant.outer_model_segment),
        "candidate_outer_pr85_model_segment_sha256": sha256_bytes(variant.outer_model_segment),
        "inner_body_delta_bytes_vs_source_model_segment_advisory": inner_delta,
        "outer_pr85_model_delta_bytes_vs_source": outer_delta,
        "outer_pr85_archive_delta_bytes_vs_source_formula": outer_delta,
        "rate_score_delta_if_components_identical_formula_only": (
            outer_delta * 25.0 / ORIGINAL_VIDEO_BYTES
        ),
        "decoded_tensor_parity": parity,
        "runtime_compatibility": runtime,
        "buildable": parity_passed and runtime_passed and outer_delta < 0,
        "build_blockers": [
            blocker
            for blocker, failed in (
                ("decoded_model_tensor_parity_failed", not parity_passed),
                ("pr85_runtime_missing_qfq4_loader", not runtime_passed),
                ("no_pr85_outer_model_byte_win", outer_delta >= 0),
            )
            if failed
        ],
    }


def _build_dispatch_blocker(
    *,
    rows: Sequence[Mapping[str, Any]],
    best_screened: Mapping[str, Any],
    blocker_class: str | None,
    blocker: str | None,
    runtime: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the standalone fail-closed dispatch contract for QFQ4."""

    best_parity = best_screened.get("decoded_tensor_parity", {})
    return {
        "schema": BLOCKER_SCHEMA,
        "tool": TOOL,
        "dispatch": False,
        "score_claim": False,
        "remote_gpu_dispatch_performed": False,
        "blocker_class": blocker_class,
        "blocker": blocker,
        "blockers": list(best_screened.get("build_blockers") or []),
        "screened_candidate_count": len(rows),
        "candidate_archive_emitted": False,
        "best_formula_only_candidate": {
            "candidate_id": best_screened.get("candidate_id"),
            "outer_pr85_model_delta_bytes_vs_source": best_screened.get(
                "outer_pr85_model_delta_bytes_vs_source"
            ),
            "candidate_outer_pr85_model_segment_bytes": best_screened.get(
                "candidate_outer_pr85_model_segment_bytes"
            ),
            "candidate_outer_pr85_model_segment_sha256": best_screened.get(
                "candidate_outer_pr85_model_segment_sha256"
            ),
            "build_blockers": best_screened.get("build_blockers"),
        },
        "decoded_tensor_parity_gate": best_parity,
        "runtime_compatibility_gate": runtime,
        "runtime_output_parity_gate": {
            "passed": False,
            "status": "not_run_fail_closed_before_runtime_loader",
            "required": True,
            "requirement": (
                "QFQ4 may only become dispatchable after the scored PR85/STBM runtime "
                "can decode the model and a local renderer/output parity gate passes."
            ),
        },
        "required_before_dispatch": [
            "decoded_model_tensor_parity_or_reviewed_runtime_output_parity",
            "robust_current_pr85_single_x_QFQ4_model_loader",
            "local_renderer_output_parity_on_source_vs_candidate",
            "model_only_archive_audit_preserves_mask_pose_post_shift_frac_bias_region_randmulti",
            "fresh_lane_claim_before_any_remote_exact_eval",
        ],
        "adversarial_decision": (
            "The current QFQ4 byte win is formula-only. It is fail-closed because "
            "decoded tensor parity fails and the scored runtime lacks QFQ4 support."
        ),
    }


def build_or_block_probe(
    *,
    archive: Path = DEFAULT_ARCHIVE,
    out_dir: Path = DEFAULT_OUT_DIR,
    replay_inflate_py: Path = DEFAULT_REPLAY_INFLATE,
    robust_current_dir: Path = DEFAULT_ROBUST_CURRENT,
    pr90_payload_probe_json: Path = DEFAULT_PR90_PAYLOAD_PROBE,
    pr90_inflate_py: Path = DEFAULT_PR90_INFLATE,
    pr90_flat_fp4_codec: Path = DEFAULT_PR90_FLAT_FP4_CODEC,
    qrow_policies: Sequence[str] = ("shifted_int8_rows", "all_fp16_rows"),
) -> dict[str, Any]:
    source_archive, x_raw = _read_single_member_archive(archive, expected_name="x")
    archive_sha = source_archive["archive_sha256"]
    source_archive["known_public_pr85_v5_match"] = {
        "matches": (
            source_archive["archive_bytes"] == KNOWN_PUBLIC_PR85["archive_bytes"]
            and archive_sha == KNOWN_PUBLIC_PR85["archive_sha256"]
        ),
        "expected_archive_bytes": KNOWN_PUBLIC_PR85["archive_bytes"],
        "expected_archive_sha256": KNOWN_PUBLIC_PR85["archive_sha256"],
    }
    bundle = parse_pr85_bundle(x_raw)
    source_model_segment = bytes(bundle.segments["model"])
    source_qh0 = _brotli_decompress(source_model_segment, label="PR85 source model segment")
    source_state, source_decode_report = decode_qh0_state_dict(source_qh0, device="cpu")
    record_set = parse_qh0_record_set(source_qh0)
    record_map = _records_by_name(record_set.records)
    runtime = qfq4_runtime_compatibility(
        replay_inflate_py=replay_inflate_py,
        robust_current_dir=robust_current_dir,
    )

    rows: list[dict[str, Any]] = []
    variants: list[QFQ4ProbeVariant] = []
    for policy in qrow_policies:
        variant = build_pr85_qfq4_probe_variant(record_map, qrow_policy=policy)
        variants.append(variant)
        candidate_state = decode_pr85_qfq4_probe_payload(variant.inner_brotli_payload, device="cpu")
        parity = compare_state_dicts(source_state, candidate_state)
        rows.append(
            _candidate_row(
                variant=variant,
                source_model_segment_bytes=len(source_model_segment),
                parity=parity,
                runtime=runtime,
            )
        )

    built_candidates: list[dict[str, Any]] = []
    header_mode = _source_header_mode(bundle.format)
    buildable = [row for row in rows if row["buildable"]]
    by_variant = {variant.variant_id: variant for variant in variants}
    for row in sorted(
        buildable,
        key=lambda item: (
            int(item["outer_pr85_model_delta_bytes_vs_source"]),
            str(item["candidate_id"]),
        ),
    ):
        variant = by_variant[str(row["candidate_id"])]
        segments = dict(bundle.segments)
        segments["model"] = variant.outer_model_segment
        candidate_x = pack_pr85_bundle(segments, header_mode=header_mode)
        candidate_dir = out_dir / str(row["candidate_id"])
        candidate_archive = _write_single_x_archive(candidate_dir / "archive.zip", candidate_x)
        archive_delta = int(candidate_archive["archive_bytes"]) - int(source_archive["archive_bytes"])
        if archive_delta >= 0:
            continue
        manifest = {
            "schema": MANIFEST_SCHEMA,
            "tool": TOOL,
            "score_claim": False,
            "dispatch_performed": False,
            "remote_gpu_dispatch_performed": False,
            "candidate_id": row["candidate_id"],
            "source_archive": source_archive,
            "source_model_segment": {
                "bytes": len(source_model_segment),
                "sha256": sha256_bytes(source_model_segment),
                "decoded_qh0_bytes": len(source_qh0),
                "decoded_qh0_sha256": sha256_bytes(source_qh0),
            },
            "qfq4_probe": row,
            "candidate_archive": candidate_archive,
            "candidate_archive_delta_bytes_vs_source": archive_delta,
            "exact_eval_eligibility": {
                "eligible": True,
                "dispatch_unlocked": True,
                "requires_lane_claim_before_remote_eval": True,
                "score_claim_from_this_artifact": False,
            },
        }
        (candidate_dir / "manifest.json").write_bytes(_json_bytes(manifest))
        built_candidates.append(manifest)

    best_screened = min(
        rows,
        key=lambda item: (
            int(item["outer_pr85_model_delta_bytes_vs_source"]),
            str(item["candidate_id"]),
        ),
    )
    if built_candidates:
        blocker_class = None
        blocker = None
        dispatch_unlocked = True
    else:
        blockers: list[str] = []
        if not any(row["decoded_tensor_parity"]["decoded_tensor_parity"] for row in rows):
            blockers.append("decoded_model_tensor_parity_failed")
        if not runtime["runtime_can_decode_without_edits"]:
            blockers.append("pr85_runtime_missing_qfq4_loader")
        if not any(int(row["outer_pr85_model_delta_bytes_vs_source"]) < 0 for row in rows):
            blockers.append("no_pr85_outer_model_byte_win")
        if "decoded_model_tensor_parity_failed" in blockers and "pr85_runtime_missing_qfq4_loader" in blockers:
            blocker_class = "tensor_parity_failed_and_runtime_incompatible"
        elif "decoded_model_tensor_parity_failed" in blockers:
            blocker_class = "tensor_parity_failed"
        elif "pr85_runtime_missing_qfq4_loader" in blockers:
            blocker_class = "runtime_incompatibility"
        else:
            blocker_class = "no_real_byte_win"
        blocker = "; ".join(blockers)
        dispatch_unlocked = False

    dispatch_blocker = (
        None
        if built_candidates
        else _build_dispatch_blocker(
            rows=rows,
            best_screened=best_screened,
            blocker_class=blocker_class,
            blocker=blocker,
            runtime=runtime,
        )
    )

    summary = {
        "schema": SCHEMA,
        "tool": TOOL,
        "dispatch": False,
        "planning_only": not bool(built_candidates),
        "score_claim": False,
        "dispatch_performed": False,
        "remote_gpu_dispatch_performed": False,
        "source_archive": source_archive,
        "source_bundle": {
            "format": bundle.format,
            "header_bytes": bundle.header_bytes,
            "segment_lengths": bundle.segment_lengths,
        },
        "source_model_segment": {
            "bytes": len(source_model_segment),
            "sha256": sha256_bytes(source_model_segment),
            "decoded_qh0_bytes": len(source_qh0),
            "decoded_qh0_sha256": sha256_bytes(source_qh0),
            "decode_report": source_decode_report.__dict__,
        },
        "record_set": record_set_summary(record_set),
        "worker_g_planning_only_context": WORKER_G_PLANNING_ONLY,
        "pr90_source_evidence": pr90_source_evidence(
            payload_probe_json=pr90_payload_probe_json,
            pr90_inflate_py=pr90_inflate_py,
            pr90_flat_fp4_codec=pr90_flat_fp4_codec,
        ),
        "runtime_compatibility": runtime,
        "screened_candidate_count": len(rows),
        "built_candidate_count": len(built_candidates),
        "dispatch_unlocked": dispatch_unlocked,
        "blocker_class": blocker_class,
        "blocker": blocker,
        "structured_blocker_json": _repo_rel(out_dir / "dispatch_blocker.json")
        if dispatch_blocker is not None
        else None,
        "structured_blocker": dispatch_blocker,
        "best_screened_candidate": best_screened,
        "best_built_candidate": _candidate_summary(built_candidates[0]) if built_candidates else None,
        "candidate_manifests": [
            {
                "candidate_id": candidate["candidate_id"],
                "manifest_path": _repo_rel(out_dir / candidate["candidate_id"] / "manifest.json"),
                "archive_path": candidate["candidate_archive"]["path"],
                "archive_bytes": candidate["candidate_archive"]["archive_bytes"],
                "archive_sha256": candidate["candidate_archive"]["archive_sha256"],
                "archive_delta_bytes_vs_source": candidate["candidate_archive_delta_bytes_vs_source"],
            }
            for candidate in built_candidates
        ],
        "screened_candidates": rows,
        "runtime_output_parity_gate": {
            "status": "not_run_runtime_loader_missing_qfq4"
            if not runtime["runtime_can_decode_without_edits"]
            else "not_run_decoded_tensor_parity_failed"
            if not any(row["decoded_tensor_parity"]["decoded_tensor_parity"] for row in rows)
            else "required_before_exact_eval",
            "passed": False,
            "requirement": "PR85 no-edit runtime output must match the source archive before GPU dispatch.",
        },
        "exact_eval_readiness": {
            "ready": bool(built_candidates),
            "dispatch_unlocked": dispatch_unlocked,
            "gpu_dispatch_performed": False,
            "requires_lane_claim_before_remote_eval": True,
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "candidate_summary.json").write_bytes(_json_bytes(summary))
    if dispatch_blocker is not None:
        (out_dir / "dispatch_blocker.json").write_bytes(_json_bytes(dispatch_blocker))
    return summary


def _candidate_summary(candidate: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if candidate is None:
        return None
    return {
        "candidate_id": candidate["candidate_id"],
        "archive_path": candidate["candidate_archive"]["path"],
        "archive_bytes": candidate["candidate_archive"]["archive_bytes"],
        "archive_sha256": candidate["candidate_archive"]["archive_sha256"],
        "archive_delta_bytes_vs_source": candidate["candidate_archive_delta_bytes_vs_source"],
    }


def _parse_csv(text: str) -> tuple[str, ...]:
    values = tuple(part.strip() for part in text.split(",") if part.strip())
    if not values:
        raise argparse.ArgumentTypeError("must contain at least one value")
    return values


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--replay-inflate-py", type=Path, default=DEFAULT_REPLAY_INFLATE)
    parser.add_argument("--robust-current-dir", type=Path, default=DEFAULT_ROBUST_CURRENT)
    parser.add_argument("--pr90-payload-probe-json", type=Path, default=DEFAULT_PR90_PAYLOAD_PROBE)
    parser.add_argument("--pr90-inflate-py", type=Path, default=DEFAULT_PR90_INFLATE)
    parser.add_argument("--pr90-flat-fp4-codec", type=Path, default=DEFAULT_PR90_FLAT_FP4_CODEC)
    parser.add_argument("--qrow-policies", type=_parse_csv, default=("shifted_int8_rows", "all_fp16_rows"))
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    summary = build_or_block_probe(
        archive=args.archive,
        out_dir=args.out_dir,
        replay_inflate_py=args.replay_inflate_py,
        robust_current_dir=args.robust_current_dir,
        pr90_payload_probe_json=args.pr90_payload_probe_json,
        pr90_inflate_py=args.pr90_inflate_py,
        pr90_flat_fp4_codec=args.pr90_flat_fp4_codec,
        qrow_policies=args.qrow_policies,
    )
    print(
        json.dumps(
            {
                "summary_path": _repo_rel(args.out_dir / "candidate_summary.json"),
                "built_candidate_count": summary["built_candidate_count"],
                "dispatch_unlocked": summary["dispatch_unlocked"],
                "blocker_class": summary["blocker_class"],
                "blocker": summary["blocker"],
                "best_screened_candidate": {
                    key: summary["best_screened_candidate"][key]
                    for key in (
                        "candidate_id",
                        "outer_pr85_model_delta_bytes_vs_source",
                        "qfq4_inner_brotli_bytes",
                        "candidate_outer_pr85_model_segment_bytes",
                        "build_blockers",
                    )
                },
                "best_built_candidate": summary["best_built_candidate"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
