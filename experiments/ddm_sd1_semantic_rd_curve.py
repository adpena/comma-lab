#!/usr/bin/env python3
"""Measure a decoded PR130 semantic quantization rate-distortion curve.

This experiment deliberately does not modify the public PR130 receiver.  It
defines a counted, checkpoint-template-bound research format for non-int4
semantic payloads, rebuilds the complete CPR1 archive around each payload,
parses the archive back, and evaluates only the decoded tensors.  The legacy
int4 cell is required to reproduce the frozen semantic blob and archive
byte-for-byte.

The measured objective is the semantic-leg change

    100 * delta_d_seg + 25 * delta_archive_bytes / 37_545_489

It is not a full candidate score: the public receiver is int4-only and a
changed semantic frame can also change PoseNet output.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import io
import json
import lzma
import os
import platform
import re
import shutil
import struct
import sys
import time
import zipfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from safetensors.torch import load_file



# fp16's smallest positive (subnormal) value, 2**-24 = 5.960464e-08.  A positive
# floor applied in fp32 BELOW this rounds to EXACTLY 0.0 when narrowed to fp16,
# silently re-opening the divide-by-zero / zero-scale the floor exists to close.
# The floor must therefore be re-applied AFTER the cast.
_FP16_MIN_POSITIVE = 5.960464477539063e-08

ORIGINAL_BYTES = 37_545_489
EXPECTED_BASE_ARCHIVE_BYTES = 191_052
EXPECTED_BASE_ARCHIVE_SHA256 = (
    "0491d5df84fc70b62b3f7ccf8894f5e1b81c616de46a052e4423fc1e18fdc7cd"
)
EXPECTED_BASE_SEMANTIC_BYTES = 40_252
EXPECTED_BASE_SEMANTIC_SHA256 = (
    "9b98360bd56918b5a414ace375c29790b7fe9f7f55cf423c0564ef4e62a39b99"
)
EXPECTED_CHECKPOINT_SHA256 = (
    "3948ccfcd44778dc42affee18a10c3f3baa434d1a2eb2345a013146c1dbfb647"
)
EXPECTED_OFFICIAL_ADA_CACHE_SHA256 = (
    "382d7dfe38b37c0cc5017e5645032faa045af6924db66e0b67549cc96c840195"
)
EXPECTED_RENDERER_ORACLE_SHA256 = (
    "ffdf098801863ff8bffe8bd818ce101928dd75b4937cbbffb2e225bddbc12f4b"
)
EXPECTED_UPSTREAM_MODULES_SHA256 = (
    "065961ba97023e393e27818760b0dc8efaa8dd53c5d4cc70a2db8ee1b3cf49aa"
)
EXPECTED_SEGNET_SHA256 = (
    "68956e328d4c5d875389a1a444870e6bac1c052c9986123827af95c07c6991b6"
)
EXPECTED_MPS_BASE_MISMATCHES = 33_703
EXPECTED_DENOMINATOR = 600 * 384 * 512
MIXED_MAGIC = b"SD1M"
MIXED_VERSION = 1
LZMA_FILTERS = [{
    "id": lzma.FILTER_LZMA2,
    "dict_size": 1 << 16,
    "lc": 0,
    "lp": 1,
    "pb": 0,
    "mode": lzma.MODE_NORMAL,
    "nice_len": 273,
    "mf": lzma.MF_BT4,
    "depth": 0,
}]


@dataclasses.dataclass(frozen=True)
class BaseArchive:
    archive_bytes: bytes
    payload: bytes
    model_compressed: bytes
    model_raw: bytes
    semantic_blob: bytes
    carrier_bytes: int
    model_suffix: bytes
    tokens: bytes


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    atomic_write_bytes(path, payload)


def pack_signed_bits(codes: torch.Tensor, bits: int) -> bytes:
    if not 2 <= bits <= 8:
        raise ValueError("signed bit packer supports 2 through 8 bits")
    values = codes.detach().cpu().numpy().astype(np.int16, copy=False).reshape(-1)
    unsigned = (values & ((1 << bits) - 1)).astype(np.uint16, copy=False)
    shifts = np.arange(bits, dtype=np.uint16)
    bitstream = ((unsigned[:, None] >> shifts) & 1).astype(np.uint8).reshape(-1)
    return np.packbits(bitstream, bitorder="little").tobytes()


def unpack_signed_bits(
    blob: memoryview, count: int, bits: int
) -> tuple[torch.Tensor, memoryview]:
    if not 2 <= bits <= 8:
        raise ValueError("signed bit unpacker supports 2 through 8 bits")
    byte_count = (count * bits + 7) // 8
    if len(blob) < byte_count:
        raise ValueError("truncated signed code stream")
    packed = np.frombuffer(blob[:byte_count], dtype=np.uint8)
    bitstream = np.unpackbits(packed, bitorder="little")[:count * bits]
    bitstream = bitstream.reshape(count, bits).astype(np.int16, copy=False)
    shifts = (1 << np.arange(bits, dtype=np.int16))[None]
    unsigned = (bitstream * shifts).sum(axis=1, dtype=np.int16)
    sign = 1 << (bits - 1)
    values = np.where(unsigned >= sign, unsigned - (1 << bits), unsigned)
    return (
        torch.from_numpy(values.astype(np.int8, copy=False)),
        blob[byte_count:],
    )


def quantized_tensor(
    name: str, value: torch.Tensor, bits: int
) -> tuple[torch.Tensor, bytes]:
    source = value.detach().cpu().float()
    if source.ndim < 2:
        stored = source.to(torch.float16)
        array = stored.numpy().astype("<f2", copy=False)
        return stored.float(), array.tobytes()
    limit = (1 << (bits - 1)) - 1
    embedding = name.endswith("embed.weight")
    reduce_dims = (
        tuple(range(source.ndim - 1))
        if embedding
        else tuple(range(1, source.ndim))
    )
    scale = source.abs().amax(dim=reduce_dims, keepdim=True).clamp_min(1e-8)
    scale = (scale / limit).to(torch.float16).clamp(min=_FP16_MIN_POSITIVE)
    codes = (source / scale.float()).round().clamp(-limit, limit).to(torch.int8)
    restored = codes.float() * scale.float()
    scale_bytes = scale.reshape(-1).numpy().astype("<f2", copy=False).tobytes()
    return restored, scale_bytes + pack_signed_bits(codes, bits)


def quantized_names(state: Mapping[str, torch.Tensor]) -> list[str]:
    return [name for name, value in state.items() if value.ndim >= 2]


def _pack_depth_nibbles(bits: list[int]) -> bytes:
    values = np.asarray(bits, dtype=np.uint8)
    if np.any((values < 2) | (values > 8)):
        raise ValueError("mixed semantic bit depths must be in [2, 8]")
    if values.size % 2:
        values = np.pad(values, (0, 1))
    return (values[0::2] | (values[1::2] << 4)).tobytes()


def _unpack_depth_nibbles(blob: bytes, count: int) -> list[int]:
    packed_count = (count + 1) // 2
    if len(blob) < packed_count:
        raise ValueError("truncated mixed semantic allocation")
    packed = np.frombuffer(blob[:packed_count], dtype=np.uint8)
    values = np.empty(packed_count * 2, dtype=np.uint8)
    values[0::2] = packed & 0xF
    values[1::2] = packed >> 4
    result = values[:count].astype(int).tolist()
    if any(bit < 2 or bit > 8 for bit in result):
        raise ValueError("invalid bit depth in mixed semantic allocation")
    return result


def pack_semantic_state(
    state: Mapping[str, torch.Tensor],
    allocation: Mapping[str, int],
    *,
    legacy_int4: bool,
) -> tuple[bytes, OrderedDict[str, torch.Tensor]]:
    names = quantized_names(state)
    if set(allocation) != set(names):
        raise ValueError("allocation keys do not match quantized state tensors")
    if legacy_int4 and any(int(allocation[name]) != 4 for name in names):
        raise ValueError("legacy semantic payload is int4-only")
    packed = bytearray()
    if not legacy_int4:
        packed.extend(MIXED_MAGIC)
        packed.extend(bytes([MIXED_VERSION, len(names)]))
        packed.extend(_pack_depth_nibbles([int(allocation[name]) for name in names]))
    expected: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, value in state.items():
        bits = int(allocation[name]) if value.ndim >= 2 else 4
        restored, payload = quantized_tensor(name, value, bits)
        expected[name] = restored
        packed.extend(payload)
    return bytes(packed), expected


def unpack_semantic_state(
    blob: bytes,
    template: Mapping[str, torch.Tensor],
) -> tuple[OrderedDict[str, torch.Tensor], OrderedDict[str, int], str]:
    names = quantized_names(template)
    remaining = memoryview(blob)
    if blob.startswith(MIXED_MAGIC):
        if len(remaining) < 6:
            raise ValueError("truncated mixed semantic header")
        version = int(remaining[4])
        count = int(remaining[5])
        if version != MIXED_VERSION or count != len(names):
            raise ValueError("unsupported mixed semantic header")
        depth_bytes = (count + 1) // 2
        depths = _unpack_depth_nibbles(bytes(remaining[6:6 + depth_bytes]), count)
        remaining = remaining[6 + depth_bytes:]
        format_name = "sd1_mixed_v1"
    else:
        depths = [4] * len(names)
        format_name = "legacy_int4"
    allocation = OrderedDict(zip(names, depths, strict=True))
    restored: OrderedDict[str, torch.Tensor] = OrderedDict()
    for name, template_value in template.items():
        shape = tuple(template_value.shape)
        count = template_value.numel()
        if template_value.ndim < 2:
            byte_count = count * 2
            if len(remaining) < byte_count:
                raise ValueError(f"truncated fp16 tensor {name}")
            value = np.frombuffer(remaining[:byte_count], dtype="<f2").copy()
            restored[name] = torch.from_numpy(value).reshape(shape).float()
            remaining = remaining[byte_count:]
            continue
        embedding = name.endswith("embed.weight")
        scale_count = shape[-1] if embedding else shape[0]
        scale_bytes = scale_count * 2
        if len(remaining) < scale_bytes:
            raise ValueError(f"truncated scale tensor {name}")
        scales = torch.from_numpy(
            np.frombuffer(remaining[:scale_bytes], dtype="<f2").copy()
        ).float()
        remaining = remaining[scale_bytes:]
        codes, remaining = unpack_signed_bits(remaining, count, allocation[name])
        scale_shape = [1] * len(shape)
        scale_shape[-1 if embedding else 0] = scale_count
        restored[name] = (
            codes.reshape(shape).float() * scales.reshape(scale_shape)
        )
    if remaining:
        raise ValueError(f"semantic payload has {len(remaining)} trailing bytes")
    return restored, allocation, format_name


def assert_states_equal(
    expected: Mapping[str, torch.Tensor],
    actual: Mapping[str, torch.Tensor],
) -> None:
    if list(expected) != list(actual):
        raise ValueError("decoded state keys or ordering differ")
    for name in expected:
        if not torch.equal(expected[name], actual[name]):
            delta = (expected[name] - actual[name]).abs().max().item()
            raise ValueError(f"decoded tensor mismatch for {name}: max_abs={delta}")


def read_base_archive(path: Path) -> BaseArchive:
    archive_bytes = path.read_bytes()
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        if archive.namelist() != ["p"]:
            raise ValueError("base archive must contain exactly member p")
        payload = archive.read("p")
    if len(payload) < 4:
        raise ValueError("base payload is truncated")
    model_size = struct.unpack_from("<I", payload)[0]
    model_compressed = payload[4:4 + model_size]
    if len(model_compressed) != model_size:
        raise ValueError("base model stream is truncated")
    model_raw = lzma.decompress(model_compressed)
    if len(model_raw) < 8:
        raise ValueError("base raw model bundle is truncated")
    semantic_bytes, carrier_bytes = struct.unpack_from("<II", model_raw)
    semantic_end = 8 + semantic_bytes
    carrier_end = semantic_end + carrier_bytes
    if carrier_end >= len(model_raw):
        raise ValueError("base model bundle lacks carrier or HPAC data")
    return BaseArchive(
        archive_bytes=archive_bytes,
        payload=payload,
        model_compressed=model_compressed,
        model_raw=model_raw,
        semantic_blob=model_raw[8:semantic_end],
        carrier_bytes=carrier_bytes,
        model_suffix=model_raw[semantic_end:],
        tokens=payload[4 + model_size:],
    )


def deterministic_zip(payload: bytes) -> bytes:
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", allowZip64=False) as archive:
        info = zipfile.ZipInfo("p", date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.create_system = 3
        info.external_attr = 0o100644 << 16
        archive.writestr(info, payload)
    return output.getvalue()


def rebuild_archive(base: BaseArchive, semantic_blob: bytes) -> bytes:
    model_raw = (
        struct.pack("<II", len(semantic_blob), base.carrier_bytes)
        + semantic_blob
        + base.model_suffix
    )
    model_compressed = lzma.compress(
        model_raw, format=lzma.FORMAT_XZ, filters=LZMA_FILTERS
    )
    payload = struct.pack("<I", len(model_compressed)) + model_compressed + base.tokens
    return deterministic_zip(payload)


def semantic_blob_from_archive(
    archive_bytes: bytes, base: BaseArchive
) -> bytes:
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        if archive.namelist() != ["p"]:
            raise ValueError("candidate archive member set differs")
        payload = archive.read("p")
    model_size = struct.unpack_from("<I", payload)[0]
    model_raw = lzma.decompress(payload[4:4 + model_size])
    semantic_size, carrier_size = struct.unpack_from("<II", model_raw)
    semantic_end = 8 + semantic_size
    if carrier_size != base.carrier_bytes:
        raise ValueError("candidate changed carrier length")
    if model_raw[semantic_end:] != base.model_suffix:
        raise ValueError("candidate changed carrier or HPAC bytes")
    if payload[4 + model_size:] != base.tokens:
        raise ValueError("candidate changed token bytes")
    return model_raw[8:semantic_end]


def sanitize(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def semantic_delta_s(
    mismatches: int,
    archive_bytes: int,
    baseline_mismatches: int,
    baseline_archive_bytes: int,
    denominator: int,
) -> tuple[float, float, int]:
    delta_dseg = (mismatches - baseline_mismatches) / denominator
    delta_archive = archive_bytes - baseline_archive_bytes
    value = 100.0 * delta_dseg + 25.0 * delta_archive / ORIGINAL_BYTES
    return value, delta_dseg, delta_archive


def stratified_random_pair_ids(
    *, seed: int, strata: int = 10, pairs_per_stratum: int = 12
) -> list[int]:
    if 600 % strata:
        raise ValueError("600 pairs must divide evenly into screening strata")
    stratum_size = 600 // strata
    if not 1 <= pairs_per_stratum <= stratum_size:
        raise ValueError("invalid pairs-per-stratum count")
    generator = np.random.default_rng(seed)
    selected: list[int] = []
    for stratum in range(strata):
        start = stratum * stratum_size
        choices = generator.choice(
            np.arange(start, start + stratum_size),
            size=pairs_per_stratum,
            replace=False,
        )
        selected.extend(int(value) for value in choices)
    return sorted(selected)


def load_or_initialize_progress(
    path: Path,
    fingerprints: Mapping[str, Any],
    argv: list[str],
    axis_label: str,
) -> dict[str, Any]:
    if path.exists():
        progress = json.loads(path.read_text())
        if progress.get("fingerprints") != fingerprints:
            raise ValueError("resume receipt fingerprints do not match this run")
        return progress
    progress = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "updated_at_utc": utc_now(),
        "argv": argv,
        "fingerprints": dict(fingerprints),
        "axis": axis_label,
        "score_claim": False,
        "promotion_eligible": False,
        "objective_scope": "semantic_leg_delta_s_only; pose_not_measured",
        "results": [],
    }
    atomic_write_json(path, progress)
    return progress


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.inference_mode()
def evaluate_state(
    model: torch.nn.Module,
    segnet: torch.nn.Module,
    render_for_seg: Any,
    decoded_state: Mapping[str, torch.Tensor],
    conditioning: torch.Tensor,
    target: torch.Tensor,
    pair_ids: list[int],
    *,
    batch_size: int,
    device: torch.device,
) -> tuple[int, int, float]:
    model.load_state_dict(decoded_state, strict=True)
    model.eval()
    mismatches = 0
    denominator = 0
    synchronize(device)
    started = time.monotonic()
    for start in range(0, len(pair_ids), batch_size):
        selected = pair_ids[start:start + batch_size]
        selected_cpu = torch.tensor(selected, dtype=torch.long)
        pair_idx = selected_cpu.to(device)
        input_tokens = conditioning[selected_cpu].to(device)
        target_tokens = target[selected_cpu].to(device)
        frame = render_for_seg(model, input_tokens, pair_idx, exact_path=True)
        prediction = segnet(frame).argmax(dim=1)
        mismatches += int((prediction != target_tokens).sum().item())
        denominator += target_tokens.numel()
    synchronize(device)
    return mismatches, denominator, time.monotonic() - started


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge-root", type=Path, required=True)
    parser.add_argument("--conditioning-cache", type=Path, required=True)
    parser.add_argument("--target-cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-archive", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--resume-from", type=Path, required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--cpu-threads", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument("--minimum-free-bytes", type=int, default=100_000_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.batch_size <= 120:
        raise ValueError("batch size must be in [1, 120]")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    free_bytes = shutil.disk_usage(args.out_dir).free
    if free_bytes < args.minimum_free_bytes:
        raise RuntimeError(
            f"storage preflight refused: free={free_bytes} required={args.minimum_free_bytes}"
        )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if device.type == "cpu":
        if not 1 <= args.cpu_threads <= (os.cpu_count() or args.cpu_threads):
            raise ValueError("cpu thread count exceeds the available logical CPUs")
        torch.set_num_threads(args.cpu_threads)
        torch.set_num_interop_threads(1)
    source_path = Path(__file__).resolve()
    repo_root = source_path.parents[1]
    renderer_oracle_path = (
        repo_root / "src" / "tac" / "pr130_lift" / "lifted"
        / "semantic_renderer_oracle.py"
    )
    upstream_modules_path = args.challenge_root.resolve() / "modules.py"
    segnet_path = args.challenge_root.resolve() / "models" / "segnet.safetensors"
    source_sha256 = sha256_file(source_path)
    checkpoint_sha256 = sha256_file(args.checkpoint)
    conditioning_sha256 = sha256_file(args.conditioning_cache)
    target_sha256 = sha256_file(args.target_cache)
    renderer_oracle_sha256 = sha256_file(renderer_oracle_path)
    upstream_modules_sha256 = sha256_file(upstream_modules_path)
    segnet_sha256 = sha256_file(segnet_path)
    pinned_hashes = {
        "checkpoint": (checkpoint_sha256, EXPECTED_CHECKPOINT_SHA256),
        "conditioning cache": (
            conditioning_sha256,
            EXPECTED_OFFICIAL_ADA_CACHE_SHA256,
        ),
        "target cache": (target_sha256, EXPECTED_OFFICIAL_ADA_CACHE_SHA256),
        "semantic renderer oracle": (
            renderer_oracle_sha256,
            EXPECTED_RENDERER_ORACLE_SHA256,
        ),
        "upstream modules": (
            upstream_modules_sha256,
            EXPECTED_UPSTREAM_MODULES_SHA256,
        ),
        "SegNet weights": (segnet_sha256, EXPECTED_SEGNET_SHA256),
    }
    for label, (actual, expected) in pinned_hashes.items():
        if actual != expected:
            raise ValueError(
                f"{label} SHA-256 differs from the pinned SD1 object: "
                f"got {actual}, expected {expected}"
            )
    platform_system = platform.system()
    platform_label = "macOS" if platform_system == "Darwin" else platform_system
    axis_label = (
        f"[{platform_label}-{device.type.upper()} advisory; "
        "retained official-Ada conditioning and target]"
    )
    screen_pair_ids = stratified_random_pair_ids(seed=args.seed)
    screen_pair_sha256 = sha256_bytes(
        np.asarray(screen_pair_ids, dtype="<i2").tobytes()
    )

    fingerprints = {
        "measurement_source": {
            "path": str(source_path),
            "sha256": source_sha256,
        },
        "semantic_renderer_oracle": {
            "path": str(renderer_oracle_path),
            "sha256": renderer_oracle_sha256,
        },
        "upstream_modules": {
            "path": str(upstream_modules_path),
            "sha256": upstream_modules_sha256,
        },
        "segnet_weights": {
            "path": str(segnet_path),
            "bytes": segnet_path.stat().st_size,
            "sha256": segnet_sha256,
        },
        "base_archive": {
            "path": str(args.base_archive.resolve()),
            "bytes": args.base_archive.stat().st_size,
            "sha256": sha256_file(args.base_archive),
        },
        "checkpoint": {
            "path": str(args.checkpoint.resolve()),
            "bytes": args.checkpoint.stat().st_size,
            "sha256": checkpoint_sha256,
        },
        "conditioning_cache": {
            "path": str(args.conditioning_cache.resolve()),
            "bytes": args.conditioning_cache.stat().st_size,
            "sha256": conditioning_sha256,
        },
        "target_cache": {
            "path": str(args.target_cache.resolve()),
            "bytes": args.target_cache.stat().st_size,
            "sha256": target_sha256,
        },
        "challenge_root": str(args.challenge_root.resolve()),
        "out_dir": str(args.out_dir.resolve()),
        "resume_from": str(args.resume_from.resolve()),
        "seed": args.seed,
        "device": str(device),
        "batch_size": args.batch_size,
        "cpu_threads": args.cpu_threads if device.type == "cpu" else None,
        "minimum_free_bytes": args.minimum_free_bytes,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
        "screen": {
            "method": "seeded_random_without_replacement_within_10_contiguous_60-pair_strata",
            "pairs_per_stratum": 12,
            "pair_ids_sha256": screen_pair_sha256,
            "n": len(screen_pair_ids),
        },
    }
    progress = load_or_initialize_progress(
        args.resume_from, fingerprints, sys.argv, axis_label
    )
    by_id = {record["candidate_id"]: record for record in progress["results"]}

    base = read_base_archive(args.base_archive)
    if len(base.archive_bytes) != EXPECTED_BASE_ARCHIVE_BYTES:
        raise ValueError("base archive byte count differs from CPR1")
    if sha256_bytes(base.archive_bytes) != EXPECTED_BASE_ARCHIVE_SHA256:
        raise ValueError("base archive SHA-256 differs from CPR1")
    if len(base.semantic_blob) != EXPECTED_BASE_SEMANTIC_BYTES:
        raise ValueError("base semantic byte count differs from CPR1")
    if sha256_bytes(base.semantic_blob) != EXPECTED_BASE_SEMANTIC_SHA256:
        raise ValueError("base semantic SHA-256 differs from CPR1")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state: OrderedDict[str, torch.Tensor] = OrderedDict(checkpoint["state_dict"])
    config = checkpoint["config"]
    qnames = quantized_names(state)
    if len(qnames) != 16 or sum(state[name].numel() for name in qnames) != 63_936:
        raise ValueError("semantic checkpoint tensor census differs from pinned stage 08")

    sys.path.insert(0, str(args.challenge_root.resolve()))
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "tac" / "pr130_lift" / "lifted"))
    import modules  # pylint: disable=import-error,import-outside-toplevel
    from semantic_renderer_oracle import (  # pylint: disable=import-error,import-outside-toplevel
        SemanticTokenRenderer,
        render_for_seg,
    )

    conditioning_cache = torch.load(
        args.conditioning_cache, map_location="cpu", weights_only=False
    )
    target_cache = torch.load(args.target_cache, map_location="cpu", weights_only=False)
    conditioning = conditioning_cache["seg"].long()
    target = target_cache["seg"].long()
    if tuple(conditioning.shape) != (600, 384, 512):
        raise ValueError(f"conditioning shape differs: {tuple(conditioning.shape)}")
    if tuple(target.shape) != tuple(conditioning.shape):
        raise ValueError("target shape differs from conditioning")
    full_pair_ids = list(range(600))
    progress["screen"] = {
        "method": "seeded_random_without_replacement_within_10_contiguous_60-pair_strata",
        "seed": args.seed,
        "pairs_per_stratum": 12,
        "pair_ids": screen_pair_ids,
        "pair_ids_sha256": screen_pair_sha256,
        "n": len(screen_pair_ids),
    }
    progress["updated_at_utc"] = utc_now()
    atomic_write_json(args.resume_from, progress)

    segnet = modules.SegNet().eval().to(device)
    segnet.load_state_dict(load_file(modules.segnet_sd_path, device=str(device)))
    model = SemanticTokenRenderer(
        width=int(config["width"]),
        blocks=int(config["blocks"]),
        frame_dim=int(config["frame_dim"]),
        num_pairs=600,
    ).to(device)

    archives_dir = args.out_dir / "archives"
    allocations_dir = args.out_dir / "allocations"
    archives_dir.mkdir(parents=True, exist_ok=True)
    allocations_dir.mkdir(parents=True, exist_ok=True)

    base_allocation = OrderedDict((name, 4) for name in qnames)

    def run_candidate(
        candidate_id: str,
        allocation: Mapping[str, int],
        *,
        legacy_int4: bool = False,
        family: str,
        pair_ids: list[int],
        population_kind: str,
        baseline_candidate_id: str,
    ) -> dict[str, Any]:
        normalized = OrderedDict((name, int(allocation[name])) for name in qnames)
        semantic_blob, expected_state = pack_semantic_state(
            state, normalized, legacy_int4=legacy_int4
        )
        if legacy_int4 and semantic_blob != base.semantic_blob:
            raise ValueError("legacy q4 semantic pack is not byte-identical to CPR1")
        archive_bytes = rebuild_archive(base, semantic_blob)
        parsed_blob = semantic_blob_from_archive(archive_bytes, base)
        if parsed_blob != semantic_blob:
            raise ValueError(f"archive semantic parse-back differs for {candidate_id}")
        decoded_state, decoded_allocation, format_name = unpack_semantic_state(
            parsed_blob, state
        )
        assert_states_equal(expected_state, decoded_state)
        if decoded_allocation != normalized:
            raise ValueError(f"allocation parse-back differs for {candidate_id}")
        if legacy_int4 and archive_bytes != base.archive_bytes:
            raise ValueError("legacy q4 full archive is not byte-identical to CPR1")

        archive_path = archives_dir / f"{sanitize(candidate_id)}.zip"
        allocation_path = allocations_dir / f"{sanitize(candidate_id)}.json"
        allocation_receipt = {
            "schema_version": 1,
            "candidate_id": candidate_id,
            "format": format_name,
            "allocation": dict(normalized),
            "allocation_is_counted_payload": not legacy_int4,
            "semantic_blob_bytes": len(semantic_blob),
            "semantic_blob_sha256": sha256_bytes(semantic_blob),
            "archive_bytes": len(archive_bytes),
            "archive_sha256": sha256_bytes(archive_bytes),
        }
        if candidate_id in by_id:
            prior = by_id[candidate_id]
            expected_resume_fields = {
                "allocation": dict(normalized),
                "population_kind": population_kind,
                "population_n": len(pair_ids),
                "pair_ids": pair_ids,
                "pair_ids_sha256": sha256_bytes(
                    np.asarray(pair_ids, dtype="<i2").tobytes()
                ),
                "baseline_candidate_id": baseline_candidate_id,
                "format": format_name,
                "semantic_blob_bytes": len(semantic_blob),
                "semantic_blob_sha256": sha256_bytes(semantic_blob),
                "archive_bytes": len(archive_bytes),
                "archive_sha256": sha256_bytes(archive_bytes),
                "axis": axis_label,
            }
            for key, expected in expected_resume_fields.items():
                if prior.get(key) != expected:
                    raise ValueError(
                        f"resume field {key} differs for {candidate_id}"
                    )
            if Path(prior["archive_path"]).read_bytes() != archive_bytes:
                raise ValueError(
                    f"resume archive artifact differs for {candidate_id}"
                )
            prior_allocation_receipt = json.loads(
                Path(prior["allocation_path"]).read_text()
            )
            if prior_allocation_receipt != allocation_receipt:
                raise ValueError(
                    f"resume allocation receipt differs for {candidate_id}"
                )
            print(json.dumps({"resume_skip": candidate_id}), flush=True)
            return prior

        atomic_write_bytes(archive_path, archive_bytes)
        atomic_write_json(allocation_path, allocation_receipt)
        mismatches, denominator, elapsed = evaluate_state(
            model,
            segnet,
            render_for_seg,
            decoded_state,
            conditioning,
            target,
            pair_ids,
            batch_size=args.batch_size,
            device=device,
        )
        expected_denominator = len(pair_ids) * 384 * 512
        if denominator != expected_denominator:
            raise ValueError("scorer denominator differs from requested population")
        baseline = by_id.get(baseline_candidate_id)
        if candidate_id == baseline_candidate_id:
            if device.type == "mps" and mismatches != EXPECTED_MPS_BASE_MISMATCHES:
                if len(pair_ids) == 600:
                    raise ValueError(
                        "q4 MPS scorer positive control differs: "
                        f"got {mismatches}, expected {EXPECTED_MPS_BASE_MISMATCHES}"
                    )
            baseline_mismatches = mismatches
            baseline_bytes = len(archive_bytes)
        elif baseline is None:
            raise ValueError("q4 baseline must run before other candidates")
        else:
            baseline_mismatches = int(baseline["mismatches"])
            baseline_bytes = int(baseline["archive_bytes"])
        delta_s, delta_dseg, delta_archive = semantic_delta_s(
            mismatches,
            len(archive_bytes),
            baseline_mismatches,
            baseline_bytes,
            denominator,
        )
        record = {
            "candidate_id": candidate_id,
            "family": family,
            "population_kind": population_kind,
            "population_n": len(pair_ids),
            "pair_ids": pair_ids,
            "pair_ids_sha256": sha256_bytes(
                np.asarray(pair_ids, dtype="<i2").tobytes()
            ),
            "baseline_candidate_id": baseline_candidate_id,
            "format": format_name,
            "allocation": dict(normalized),
            "allocation_is_counted_payload": not legacy_int4,
            "semantic_blob_bytes": len(semantic_blob),
            "semantic_blob_sha256": sha256_bytes(semantic_blob),
            "archive_bytes": len(archive_bytes),
            "archive_sha256": sha256_bytes(archive_bytes),
            "archive_path": str(archive_path.resolve()),
            "allocation_path": str(allocation_path.resolve()),
            "mismatches": mismatches,
            "denominator": denominator,
            "d_seg": mismatches / denominator,
            "delta_d_seg": delta_dseg,
            "delta_archive_bytes": delta_archive,
            "semantic_leg_delta_s": delta_s,
            "elapsed_seconds": elapsed,
            "axis": axis_label,
            "score_claim": False,
            "promotion_eligible": False,
            "receiver_status": (
                "PUBLIC_RECEIVER_CLOSED" if legacy_int4
                else "RESEARCH_PARSEBACK_ONLY_PUBLIC_RECEIVER_BLOCKED"
            ),
            "pose_status": "NOT_MEASURED; full score unavailable",
            "written_at_utc": utc_now(),
        }
        progress["results"].append(record)
        progress["updated_at_utc"] = utc_now()
        atomic_write_json(args.resume_from, progress)
        by_id[candidate_id] = record
        print(json.dumps(record, sort_keys=True), flush=True)
        return record

    baseline = run_candidate(
        "uniform_q4_legacy_n600",
        base_allocation,
        legacy_int4=True,
        family="uniform",
        pair_ids=full_pair_ids,
        population_kind="verdict_n600",
        baseline_candidate_id="uniform_q4_legacy_n600",
    )
    screen_baseline = run_candidate(
        "uniform_q4_legacy_screen_n120",
        base_allocation,
        legacy_int4=True,
        family="uniform_screen_control",
        pair_ids=screen_pair_ids,
        population_kind="screen_stratified_random_n120",
        baseline_candidate_id="uniform_q4_legacy_screen_n120",
    )
    uniform_full_records = [baseline]
    for bits in (3, 5):
        allocation = OrderedDict((name, bits) for name in qnames)
        uniform_full_records.append(
            run_candidate(
                f"uniform_q{bits}_n600",
                allocation,
                family="uniform",
                pair_ids=full_pair_ids,
                population_kind="verdict_n600",
                baseline_candidate_id="uniform_q4_legacy_n600",
            )
        )

    single_records: list[dict[str, Any]] = []
    for name in qnames:
        for bits in (3, 5):
            allocation = OrderedDict(base_allocation)
            allocation[name] = bits
            candidate_id = f"single_{sanitize(name)}_q{bits}"
            single_records.append(
                run_candidate(
                    candidate_id,
                    allocation,
                    family="single_tensor",
                    pair_ids=screen_pair_ids,
                    population_kind="screen_stratified_random_n120",
                    baseline_candidate_id="uniform_q4_legacy_screen_n120",
                )
            )

    improving_moves = sorted(
        (
            record for record in single_records
            if float(record["semantic_leg_delta_s"]) < 0.0
        ),
        key=lambda record: float(record["semantic_leg_delta_s"]),
    )
    selected_tensors: set[str] = set()
    cumulative = OrderedDict(base_allocation)
    prefix_records: list[dict[str, Any]] = []
    for move in improving_moves:
        changed = [
            name for name in qnames
            if int(move["allocation"][name]) != int(base_allocation[name])
        ]
        if len(changed) != 1:
            raise ValueError("single-tensor result changed multiple tensors")
        name = changed[0]
        if name in selected_tensors:
            continue
        selected_tensors.add(name)
        cumulative[name] = int(move["allocation"][name])
        prefix_records.append(
            run_candidate(
                f"greedy_prefix_{len(prefix_records) + 1:02d}_{sanitize(name)}_q{cumulative[name]}",
                cumulative,
                family="joint_greedy_prefix",
                pair_ids=screen_pair_ids,
                population_kind="screen_stratified_random_n120",
                baseline_candidate_id="uniform_q4_legacy_screen_n120",
            )
        )

    screen_candidates = [screen_baseline, *single_records, *prefix_records]
    best_screen = min(
        screen_candidates,
        key=lambda record: float(record["semantic_leg_delta_s"]),
    )
    best_screen_allocation = OrderedDict(
        (name, int(best_screen["allocation"][name])) for name in qnames
    )
    matching_uniform = next(
        (
            record for record in uniform_full_records
            if record["allocation"] == dict(best_screen_allocation)
        ),
        None,
    )
    if matching_uniform is not None:
        selected_full = matching_uniform
    else:
        selected_full = run_candidate(
            "selected_mixed_n600",
            best_screen_allocation,
            family="selected_mixed_joint_replay",
            pair_ids=full_pair_ids,
            population_kind="verdict_n600",
            baseline_candidate_id="uniform_q4_legacy_n600",
        )
    full_verdict_records = [*uniform_full_records]
    if selected_full["candidate_id"] not in {
        record["candidate_id"] for record in full_verdict_records
    }:
        full_verdict_records.append(selected_full)
    best = min(
        full_verdict_records,
        key=lambda record: float(record["semantic_leg_delta_s"]),
    )
    interaction_canary: dict[str, Any] | None = None
    if len(prefix_records) >= 2:
        first = prefix_records[0]
        second = prefix_records[1]
        first_change = next(
            name for name in qnames
            if first["allocation"][name] != base_allocation[name]
        )
        second_changes = [
            name for name in qnames
            if second["allocation"][name] != base_allocation[name]
        ]
        second_change = next(name for name in second_changes if name != first_change)
        single_first = next(
            record for record in single_records
            if record["allocation"][first_change] == first["allocation"][first_change]
            and sum(
                record["allocation"][name] != base_allocation[name]
                for name in qnames
            ) == 1
        )
        single_second = next(
            record for record in single_records
            if record["allocation"][second_change] == second["allocation"][second_change]
            and sum(
                record["allocation"][name] != base_allocation[name]
                for name in qnames
            ) == 1
        )
        interaction_canary = {
            "first_tensor": first_change,
            "second_tensor": second_change,
            "joint_candidate_id": second["candidate_id"],
            "cross_term_semantic_leg_delta_s": (
                float(second["semantic_leg_delta_s"])
                - float(single_first["semantic_leg_delta_s"])
                - float(single_second["semantic_leg_delta_s"])
            ),
            "definition": "deltaS(A_union_B)-deltaS(A)-deltaS(B)",
        }

    final = {
        **progress,
        "completed_at_utc": utc_now(),
        "status": "complete",
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "cpu_threads": (
                torch.get_num_threads() if device.type == "cpu" else None
            ),
        },
        "positive_controls": {
            "q4_archive_byte_identical": (
                baseline["archive_sha256"] == EXPECTED_BASE_ARCHIVE_SHA256
            ),
            "q4_semantic_blob_byte_identical": (
                baseline["semantic_blob_sha256"] == EXPECTED_BASE_SEMANTIC_SHA256
            ),
            "q4_mismatches": baseline["mismatches"],
            "q4_denominator": baseline["denominator"],
            "retained_mps_q4_mismatches": EXPECTED_MPS_BASE_MISMATCHES,
            "q4_mismatch_delta_vs_retained_mps": (
                int(baseline["mismatches"]) - EXPECTED_MPS_BASE_MISMATCHES
            ),
        },
        "best_screen": best_screen,
        "selected_full_replay": selected_full,
        "full_verdict_candidate_ids": [
            record["candidate_id"] for record in full_verdict_records
        ],
        "best_measured_semantic_leg": best,
        "interaction_canary": interaction_canary,
        "boundaries": [
            "Non-int4 archives use an isolated counted research format; the public receiver does not parse them.",
            "Pose was not measured; semantic_leg_delta_s is not a full archive score.",
            "All non-int4 cells are post-hoc re-quantizations of the shipped q4-QAT master, not bit-matched QAT optima.",
            "No width or capacity checkpoint compatible with the shipped state was available.",
        ],
    }
    atomic_write_json(args.out_dir / "results.json", final)
    atomic_write_json(args.resume_from, final)
    print(
        json.dumps(
            {
                "status": "complete",
                "results": len(final["results"]),
                "best_candidate": best["candidate_id"],
                "best_semantic_leg_delta_s": best["semantic_leg_delta_s"],
                "results_path": str((args.out_dir / "results.json").resolve()),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
