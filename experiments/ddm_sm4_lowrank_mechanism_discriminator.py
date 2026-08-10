#!/usr/bin/env python3
"""Run the retained DDM-SM4 rank x precision x centering discriminator.

The grid uses the real PR130 checkpoint and the real temporal-reversion token
driver.  Every semantic field, compressed model section, outer packet, ZIP,
repeat build, and decoded state is retained on the SSD before the next cell.
Weight error selects exactly one matched-byte survivor for a real ``inflate.sh``
parity screen; weight error is never presented as a scorer result.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import shutil
import struct
import sys
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import ddm_cp2_composition_receiver_and_harness as cp2
from experiments import ddm_sd1_semantic_rd_curve as sd1
from experiments import ddm_sm3_semantic_representation as sm3
from experiments.ddm_sm4_runtime import sm4r_receiver

REPO = Path(__file__).resolve().parents[1]
SSD_ROOT = Path("/Volumes/VertigoDataTier/pact")
DEFAULT_OUTPUT = SSD_ROOT / "ddm_sm4_20260810"
DEFAULT_RESUME = DEFAULT_OUTPUT / "resume.json"
DEFAULT_CHECKPOINT = sm3.DEFAULT_CHECKPOINT
DEFAULT_TOKEN_DRIVER = cp2.AI1_TEMPORAL_ROOT / "retained/temporal_reversion/archive.zip"
DEFAULT_BASE_RAW = (
    SSD_ROOT
    / "ddm_ai1_20260809/temporal_v3/submission_temporal_reversion/inflated/0.raw"
)
DEFAULT_RECEIVER_PYTHON = cp2.DEFAULT_RECEIVER_PYTHON

CHECKPOINT_SHA256 = "3948ccfcd44778dc42affee18a10c3f3baa434d1a2eb2345a013146c1dbfb647"
TOKEN_DRIVER_SHA256 = "0f5a797fda844ee63f6057fdb7203f6578b135b4e12deafa98d6ddc3260a5c84"
TOKEN_DRIVER_BYTES = 188_636
BASE_RAW_SHA256 = "a18eb42ca6f14e1a96a0a09f11136ae491b560a66c342be32348fb5db85fa03b"
BASE_RAW_BYTES = 3_662_409_600
BASE_ARCHIVE_BYTES = 191_052
ORIGINAL_BYTES = 37_545_489
SEED = 20_260_810
GRID_RANKS = tuple(range(8, 49))
GRID_BITS = (4, 5, 6, 7, 8)
GRID_CENTERED = (False, True)
TARGET_NAMES = tuple(sorted(sm3.LOWRANK_NAMES))
RUNTIME_MODULE_NAMES = (*cp2.RUNTIME_IMPORT_NAMES, "sm4r_receiver")


@dataclass(frozen=True)
class LowrankBasis:
    row_mean: torch.Tensor | None
    u: torch.Tensor
    singular: torch.Tensor
    vh: torch.Tensor


@dataclass(frozen=True)
class DriverContext:
    receiver: ModuleType
    token_codec: str
    model_codec: str
    tokens: bytes
    model_suffix: bytes
    carrier_bytes: int
    original_semantic: bytes
    original_models_raw: bytes


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_bytes(path: Path, payload: bytes, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    if executable:
        path.chmod(0o755)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode())


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def require_record(path: Path, *, size: int, digest: str, label: str) -> None:
    if not path.is_file() or path.stat().st_size != size or sha256_file(path) != digest:
        raise RuntimeError(f"{label} differs from byte-and-SHA pin: {path}")


def require_ssd(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(SSD_ROOT.resolve())
    except ValueError as error:
        raise RuntimeError(f"DDM-SM4 bulk evidence must stay below {SSD_ROOT}: {resolved}") from error
    return resolved


def storage_preflight(path: Path, minimum_free_bytes: int) -> dict[str, Any]:
    root = require_ssd(path)
    root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(root).free
    if free < minimum_free_bytes:
        raise RuntimeError(f"storage preflight refused DDM-SM4: free={free}, required={minimum_free_bytes}")
    return {"path": str(root), "free_bytes": free, "minimum_free_bytes": minimum_free_bytes}


def load_state(checkpoint_path: Path) -> OrderedDict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = OrderedDict(checkpoint["state_dict"])
    missing = set(TARGET_NAMES) - set(state)
    if missing:
        raise RuntimeError(f"PR130 checkpoint lacks DDM-SM4 tensors: {sorted(missing)}")
    return state


def canonical_basis(value: torch.Tensor, centered: bool) -> LowrankBasis:
    rows = int(value.shape[0])
    matrix = value.detach().cpu().double().reshape(rows, -1)
    stored_mean = None
    if centered:
        # Center around the value the receiver can actually reconstruct.
        stored_mean = matrix.mean(dim=1).to(torch.float16).float()
        matrix = matrix - stored_mean.double()[:, None]
    u, singular, vh = torch.linalg.svd(matrix, full_matrices=False)
    for index in range(len(singular)):
        pivot = int(torch.argmax(u[:, index].abs()).item())
        if float(u[pivot, index]) < 0:
            u[:, index] = -u[:, index]
            vh[index] = -vh[index]
    return LowrankBasis(stored_mean, u, singular, vh)


def factors(basis: LowrankBasis, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    singular = basis.singular[:rank]
    root = singular.sqrt()
    left = (basis.u[:, :rank] * root[None]).float()
    right = (root[:, None] * basis.vh[:rank]).float()
    return left, right


def quantized_payload(
    name: str,
    value: torch.Tensor,
    bits: int,
) -> tuple[bytes, torch.Tensor]:
    restored, scales, codes = sm3.quantized_components(name, value, bits)
    encoded = scales.tobytes() + sd1.pack_signed_bits(torch.from_numpy(codes), bits)
    return encoded, restored


def pack_candidate(
    state: Mapping[str, torch.Tensor],
    bases: Mapping[tuple[str, bool], LowrankBasis],
    *,
    rank: int,
    bits: int,
    centered: bool,
) -> tuple[bytes, OrderedDict[str, torch.Tensor], dict[str, Any]]:
    qnames = sd1.quantized_names(state)
    mask = sm3.mask_for_names(qnames, sm3.LOWRANK_NAMES)
    flags = sm4r_receiver.FLAG_CENTERED if centered else 0
    payload = bytearray(sm4r_receiver.MAGIC + bytes([sm4r_receiver.VERSION, rank, bits, flags]))
    payload.extend(struct.pack("<H", mask))
    expected: OrderedDict[str, torch.Tensor] = OrderedDict()
    factor_wire_bytes = 0
    row_mean_bytes = 0
    per_tensor_wire: dict[str, int] = {}
    for name, value in state.items():
        before = len(payload)
        if value.ndim < 2:
            stored = value.detach().cpu().to(torch.float16)
            payload.extend(stored.numpy().astype("<f2", copy=False).tobytes())
            expected[name] = stored.float()
        elif name not in sm3.LOWRANK_NAMES:
            encoded, restored = sm3.standard_q4_payload(name, value)
            payload.extend(encoded)
            expected[name] = restored
        else:
            basis = bases[(name, centered)]
            if basis.row_mean is not None:
                mean_payload = basis.row_mean.numpy().astype("<f2", copy=False).tobytes()
                payload.extend(mean_payload)
                row_mean_bytes += basis.row_mean.numel() * np.dtype("<f2").itemsize
            left, right = factors(basis, rank)
            left_payload, left_restored = quantized_payload("factor.left", left, bits)
            right_payload, right_restored = quantized_payload("factor.right", right, bits)
            payload.extend(left_payload)
            payload.extend(right_payload)
            factor_wire_bytes += len(left_payload) + len(right_payload)
            matrix = left_restored @ right_restored
            if basis.row_mean is not None:
                matrix += basis.row_mean[:, None]
            expected[name] = matrix.reshape(value.shape)
        per_tensor_wire[name] = len(payload) - before
    details = {
        "rank": rank,
        "factor_bits": bits,
        "centered": centered,
        "selection_mask": mask,
        "selected_tensor_names": list(TARGET_NAMES),
        "factor_wire_bytes": factor_wire_bytes,
        "row_mean_bytes": row_mean_bytes,
        "per_tensor_wire_bytes": per_tensor_wire,
    }
    return bytes(payload), expected, details


def assert_state_equal(
    expected: Mapping[str, torch.Tensor],
    actual: Mapping[str, torch.Tensor],
) -> None:
    if list(expected) != list(actual):
        raise RuntimeError("SM4 decoded state order differs")
    for name in expected:
        if not torch.equal(expected[name], actual[name]):
            delta = float((expected[name] - actual[name]).abs().max())
            raise RuntimeError(f"SM4 decoded state differs for {name}: max_abs={delta}")


def weight_metrics(
    source: Mapping[str, torch.Tensor],
    decoded: Mapping[str, torch.Tensor],
) -> dict[str, Any]:
    rows: dict[str, Any] = {}
    squared_error = 0.0
    squared_source = 0.0
    count = 0
    for name in TARGET_NAMES:
        original = source[name].detach().cpu().double().reshape(-1)
        restored = decoded[name].detach().cpu().double().reshape(-1)
        error = restored - original
        error_ss = float(error.square().sum())
        source_ss = float(original.square().sum())
        squared_error += error_ss
        squared_source += source_ss
        count += original.numel()
        rows[name] = {
            "shape": list(source[name].shape),
            "parameters": original.numel(),
            "relative_l2": math.sqrt(error_ss / source_ss),
            "rmse": math.sqrt(error_ss / original.numel()),
            "mean_error": float(error.mean()),
            "max_abs_error": float(error.abs().max()),
        }
    return {
        "denominator_tensors": len(TARGET_NAMES),
        "denominator_parameters": count,
        "aggregate_relative_l2": math.sqrt(squared_error / squared_source),
        "aggregate_rmse": math.sqrt(squared_error / count),
        "per_tensor": rows,
    }


def load_driver(output: Path, driver_path: Path) -> DriverContext:
    receiver = cp2.load_reference_receiver()
    member = cp2.read_stored_member(driver_path)
    parts = receiver.split_payload(member)
    decoded = receiver.decode_models(parts.models, model_codec=parts.model_codec)
    semantic, semantic_end, carrier_bytes = cp2.semantic_field(decoded.raw)
    source_root = output / "retained/sources"
    atomic_bytes(source_root / "driver.archive.zip", driver_path.read_bytes())
    atomic_bytes(source_root / "driver.payload.p", member)
    atomic_bytes(source_root / "driver.models.raw", decoded.raw)
    atomic_bytes(source_root / "driver.models.xz", parts.models)
    atomic_bytes(source_root / "driver.tokens.bin", parts.tokens)
    atomic_bytes(source_root / "driver.semantic.bin", semantic)
    atomic_bytes(source_root / "driver.model_suffix.bin", decoded.raw[semantic_end:])
    rebuilt = cp2.encode_archive(
        receiver,
        decoded.raw,
        parts.tokens,
        token_codec=parts.token_codec,
        model_codec=parts.model_codec,
    )
    cp2.retain_encoding(source_root / "driver_rebuild", rebuilt)
    if rebuilt[2] != driver_path.read_bytes():
        raise RuntimeError("temporal token driver rebuild is not byte-identical")
    return DriverContext(
        receiver=receiver,
        token_codec=parts.token_codec,
        model_codec=parts.model_codec,
        tokens=parts.tokens,
        model_suffix=decoded.raw[semantic_end:],
        carrier_bytes=carrier_bytes,
        original_semantic=semantic,
        original_models_raw=decoded.raw,
    )


def candidate_id(rank: int, bits: int, centered: bool) -> str:
    return f"r{rank:02d}_b{bits}_{'centered' if centered else 'uncentered'}"


def verify_retained(receipt: Mapping[str, Any]) -> None:
    for record in receipt["retained"].values():
        require_record(
            Path(record["path"]),
            size=int(record["bytes"]),
            digest=str(record["sha256"]),
            label="retained SM4 candidate artifact",
        )


def retain_grid_cell(
    output: Path,
    context: DriverContext,
    state: Mapping[str, torch.Tensor],
    bases: Mapping[tuple[str, bool], LowrankBasis],
    *,
    rank: int,
    bits: int,
    centered: bool,
) -> dict[str, Any]:
    cell_id = candidate_id(rank, bits, centered)
    root = output / "retained/grid" / cell_id
    receipt_path = root / "receipt.json"
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        verify_retained(receipt)
        return receipt

    semantic, expected, details = pack_candidate(
        state,
        bases,
        rank=rank,
        bits=bits,
        centered=centered,
    )
    # P0: retain the candidate field before it is consumed by any compressor.
    atomic_bytes(root / "semantic.bin", semantic)
    decoded = sm4r_receiver.unpack_sm4r_or_none(semantic, state)
    if decoded is None:
        raise RuntimeError("SM4 receiver refused an SM4 field")
    assert_state_equal(expected, decoded)
    state_blob = sm3.state_wire(decoded)
    atomic_bytes(root / "decoded_state.sm3state", state_blob)

    models_raw = (
        struct.pack("<II", len(semantic), context.carrier_bytes)
        + semantic
        + context.model_suffix
    )
    atomic_bytes(root / "models.raw", models_raw)
    first = cp2.encode_archive(
        context.receiver,
        models_raw,
        context.tokens,
        token_codec=context.token_codec,
        model_codec=context.model_codec,
    )
    first_records = cp2.retain_encoding(root / "build_a", first)
    second = cp2.encode_archive(
        context.receiver,
        models_raw,
        context.tokens,
        token_codec=context.token_codec,
        model_codec=context.model_codec,
    )
    second_records = cp2.retain_encoding(root / "build_b", second)
    if first != second:
        raise RuntimeError(f"SM4 repeated build differs for {cell_id}")

    parsed_member = cp2.read_stored_member(root / "build_a/archive.zip")
    parsed_parts = context.receiver.split_payload(parsed_member)
    parsed_models = context.receiver.decode_models(
        parsed_parts.models,
        model_codec=parsed_parts.model_codec,
    )
    if parsed_models.raw != models_raw or parsed_parts.tokens != context.tokens:
        raise RuntimeError(f"outer receiver parse-back differs for {cell_id}")
    core_models, _ = context.receiver.split_optional_temporal_reversion(parsed_models.raw)
    parsed_semantic, _, _ = cp2.semantic_field(core_models)
    if parsed_semantic != semantic:
        raise RuntimeError(f"semantic parse-back differs for {cell_id}")

    metrics = weight_metrics(state, decoded)
    retained = {
        "semantic": file_record(root / "semantic.bin"),
        "models_raw": file_record(root / "models.raw"),
        "decoded_state": file_record(root / "decoded_state.sm3state"),
        "models_a": first_records["models"],
        "member_a": first_records["member"],
        "archive_a": first_records["archive"],
        "models_b": second_records["models"],
        "member_b": second_records["member"],
        "archive_b": second_records["archive"],
    }
    receipt = {
        "schema": "ddm_sm4_grid_cell.v1",
        "candidate_id": cell_id,
        "complete": True,
        "written_at_utc": utc_now(),
        "axis": "[macOS-CPU advisory; scorer-free weight error and exact archive bytes]",
        "score_claim": False,
        "details": details,
        "semantic_bytes": len(semantic),
        "archive_bytes": len(first[2]),
        "archive_delta_vs_pr130_bytes": len(first[2]) - BASE_ARCHIVE_BYTES,
        "rate_only_delta_s_vs_pr130": 25 * (len(first[2]) - BASE_ARCHIVE_BYTES) / ORIGINAL_BYTES,
        "weight_error": metrics,
        "checks": {
            "independent_sm4_receiver_matches_packer_state": True,
            "archive_repeat_byte_equal": True,
            "outer_receiver_models_raw_byte_equal": True,
            "outer_receiver_tokens_byte_equal": True,
            "semantic_parseback_byte_equal": True,
        },
        "retained": retained,
    }
    atomic_json(receipt_path, receipt)
    return receipt


def mechanism_summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    indexed = {cell["candidate_id"]: cell for cell in cells}
    budget_cell = indexed[candidate_id(32, 4, False)]
    budget = int(budget_cell["archive_bytes"])
    eligible = [cell for cell in cells if int(cell["archive_bytes"]) <= budget]
    best = min(
        eligible,
        key=lambda cell: (
            float(cell["weight_error"]["aggregate_relative_l2"]),
            int(cell["archive_bytes"]),
            str(cell["candidate_id"]),
        ),
    )
    equal_precision = indexed[candidate_id(16, 8, False)]
    r32_error = float(budget_cell["weight_error"]["aggregate_relative_l2"])
    r16_error = float(equal_precision["weight_error"]["aggregate_relative_l2"])
    per_tensor_wins = {
        name: (
            float(equal_precision["weight_error"]["per_tensor"][name]["relative_l2"])
            < float(budget_cell["weight_error"]["per_tensor"][name]["relative_l2"])
        )
        for name in TARGET_NAMES
    }
    if r16_error < r32_error and all(per_tensor_wins.values()):
        verdict = "FACTOR_PRECISION_COMPOUNDING_SUPPORTED_AT_EQUAL_BYTE_LAW"
    elif r16_error >= r32_error and not any(per_tensor_wins.values()):
        verdict = "RANK_INSUFFICIENCY_SUPPORTED_AT_EQUAL_BYTE_LAW"
    else:
        verdict = "MIXED_MECHANISM_ACROSS_TENSORS"
    uncentered_eligible = [cell for cell in eligible if not cell["details"]["centered"]]
    centered_eligible = [cell for cell in eligible if cell["details"]["centered"]]
    best_uncentered = min(uncentered_eligible, key=lambda cell: cell["weight_error"]["aggregate_relative_l2"])
    best_centered = min(centered_eligible, key=lambda cell: cell["weight_error"]["aggregate_relative_l2"])
    return {
        "matched_archive_budget_bytes": budget,
        "matched_budget_reference": budget_cell["candidate_id"],
        "eligible_cells": len(eligible),
        "grid_cells": len(cells),
        "factor_precision_discriminator": {
            "r32_int4": {
                "candidate_id": budget_cell["candidate_id"],
                "archive_bytes": budget_cell["archive_bytes"],
                "semantic_bytes": budget_cell["semantic_bytes"],
                "aggregate_relative_l2": r32_error,
            },
            "r16_int8": {
                "candidate_id": equal_precision["candidate_id"],
                "archive_bytes": equal_precision["archive_bytes"],
                "semantic_bytes": equal_precision["semantic_bytes"],
                "aggregate_relative_l2": r16_error,
            },
            "r16_int8_error_over_r32_int4_error": r16_error / r32_error,
            "per_tensor_r16_int8_wins": per_tensor_wins,
            "verdict": verdict,
            "falsifier": (
                "The double-int4 compounding hypothesis is falsified if matched-law "
                "r16-int8 fails to lower aggregate error and the same five per-tensor errors."
            ),
        },
        "centering_discriminator": {
            "best_uncentered": best_uncentered["candidate_id"],
            "best_uncentered_error": best_uncentered["weight_error"]["aggregate_relative_l2"],
            "best_centered": best_centered["candidate_id"],
            "best_centered_error": best_centered["weight_error"]["aggregate_relative_l2"],
            "centered_error_over_uncentered_error": (
                float(best_centered["weight_error"]["aggregate_relative_l2"])
                / float(best_uncentered["weight_error"]["aggregate_relative_l2"])
            ),
        },
        "selected_weight_screen_survivor": best["candidate_id"],
        "selection_rule": (
            "minimum aggregate relative L2 across the exhaustive declared grid at or below "
            "the real temporal-composed r32-int4 archive byte budget; weight error is a "
            "mechanism screen, not a score claim"
        ),
    }


def run_grid(args: argparse.Namespace) -> dict[str, Any]:
    output = require_ssd(args.output)
    preflight = storage_preflight(output, args.minimum_free_bytes)
    require_record(args.checkpoint, size=args.checkpoint.stat().st_size, digest=CHECKPOINT_SHA256, label="checkpoint")
    require_record(
        args.token_driver,
        size=TOKEN_DRIVER_BYTES,
        digest=TOKEN_DRIVER_SHA256,
        label="temporal token driver",
    )
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.use_deterministic_algorithms(True)
    state = load_state(args.checkpoint)
    context = load_driver(output, args.token_driver)
    if context.original_semantic != sd1.read_base_archive(sm3.DEFAULT_BASE_ARCHIVE).semantic_blob:
        raise RuntimeError("temporal driver changed the PR130 semantic field")
    bases = {
        (name, centered): canonical_basis(state[name], centered)
        for name in TARGET_NAMES
        for centered in GRID_CENTERED
    }
    source_sha = sha256_file(Path(__file__).resolve())
    runtime_sha = sha256_file(Path(sm4r_receiver.__file__).resolve())
    progress = {
        "schema": "ddm_sm4_resume.v1",
        "complete": False,
        "stage": "grid",
        "seed": SEED,
        "source_sha256": source_sha,
        "runtime_source_sha256": runtime_sha,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "token_driver_sha256": TOKEN_DRIVER_SHA256,
        "completed_candidates": [],
        "updated_at_utc": utc_now(),
    }
    if args.resume_from.is_file():
        prior = json.loads(args.resume_from.read_text())
        for key in ("schema", "seed", "source_sha256", "runtime_source_sha256", "checkpoint_sha256", "token_driver_sha256"):
            if prior.get(key) != progress[key]:
                raise RuntimeError(f"SM4 resume binding differs: {key}")
        progress = prior
        progress["stage"] = "grid"
        progress["complete"] = False
        progress["updated_at_utc"] = utc_now()
    atomic_json(args.resume_from, progress)

    cells: list[dict[str, Any]] = []
    for centered in GRID_CENTERED:
        for bits in GRID_BITS:
            for rank in GRID_RANKS:
                receipt = retain_grid_cell(
                    output,
                    context,
                    state,
                    bases,
                    rank=rank,
                    bits=bits,
                    centered=centered,
                )
                cells.append(receipt)
                cell_id = receipt["candidate_id"]
                if cell_id not in progress["completed_candidates"]:
                    progress["completed_candidates"].append(cell_id)
                    progress["updated_at_utc"] = utc_now()
                    atomic_json(args.resume_from, progress)

    shipped_q4, shipped_state = sd1.pack_semantic_state(
        state,
        OrderedDict((name, 4) for name in sd1.quantized_names(state)),
        legacy_int4=True,
    )
    if shipped_q4 != context.original_semantic:
        raise RuntimeError("shipped q4 positive control differs")
    result = {
        "schema": "ddm_sm4_grid_result.v1",
        "complete": True,
        "written_at_utc": utc_now(),
        "axis": "[macOS-CPU advisory; scorer-free real weights and real archive bytes]",
        "score_claim": False,
        "seed": SEED,
        "grid": {
            "ranks": list(GRID_RANKS),
            "factor_bits": list(GRID_BITS),
            "centered": list(GRID_CENTERED),
            "cells": len(cells),
        },
        "positive_control_shipped_q4_weight_error": weight_metrics(state, shipped_state),
        "mechanism": mechanism_summary(cells),
        "cells": cells,
        "storage_preflight": preflight,
        "sources": {
            "measurement_source": file_record(Path(__file__).resolve()),
            "runtime_source": file_record(Path(sm4r_receiver.__file__).resolve()),
            "checkpoint": file_record(args.checkpoint),
            "token_driver": file_record(args.token_driver),
            "sm3_reuse_commit": "d3650d6c68764385cad2d32faa394af7c87360c6",
            "cp2_reuse_commit": "58d270898002cde052b4ad34506b14984db06d49",
        },
    }
    atomic_json(output / "GRID_RESULT.json", result)
    progress["stage"] = "grid_complete"
    progress["complete"] = True
    progress["grid_result"] = file_record(output / "GRID_RESULT.json")
    progress["updated_at_utc"] = utc_now()
    atomic_json(args.resume_from, progress)
    return result


def transformed_inflate_source() -> bytes:
    source = (cp2.CP2_RUNTIME / "inflate.py").read_text()
    replacements = (
        (
            "from sm3r_receiver import unpack_sm3r_or_none\n",
            "from sm3r_receiver import unpack_sm3r_or_none\n"
            "from sm4r_receiver import MAGIC as SM4R_MAGIC\n"
            "from sm4r_receiver import unpack_sm4r_or_none\n",
        ),
        (
            "semantic_blob.startswith((SEMANTIC_MIXED_MAGIC, SM3R_MAGIC))",
            "semantic_blob.startswith((SEMANTIC_MIXED_MAGIC, SM3R_MAGIC, SM4R_MAGIC))",
        ),
        (
            "semantic_state = unpack_sm3r_or_none(semantic_blob, semantic.state_dict())\n"
            "    if semantic_state is None:\n"
            "        semantic_state = unpack_semantic(semantic_blob, semantic.state_dict())",
            "semantic_state = unpack_sm4r_or_none(semantic_blob, semantic.state_dict())\n"
            "    if semantic_state is None:\n"
            "        semantic_state = unpack_sm3r_or_none(semantic_blob, semantic.state_dict())\n"
            "    if semantic_state is None:\n"
            "        semantic_state = unpack_semantic(semantic_blob, semantic.state_dict())",
        ),
    )
    for before, after in replacements:
        if source.count(before) != 1:
            raise RuntimeError(f"CP2 inflate transformation anchor count differs: {before[:60]!r}")
        source = source.replace(before, after)
    return source.encode()


def stage_runtime(submission: Path) -> dict[str, Any]:
    records: dict[str, Any] = {}
    for name, digest in cp2.RUNTIME_REFERENCE_HASHES.items():
        source = cp2.REFERENCE_RUNTIME / name
        require_record(source, size=source.stat().st_size, digest=digest, label=f"runtime {name}")
        destination = submission / name
        atomic_bytes(destination, source.read_bytes(), executable=name == "inflate.sh")
        records[name] = {**file_record(destination), "source": str(source), "source_sha256": digest}
    source_pairs = {
        "sm3r_receiver.py": cp2.CP2_RUNTIME / "sm3r_receiver.py",
        "sm4r_receiver.py": Path(sm4r_receiver.__file__).resolve(),
    }
    for name, source in source_pairs.items():
        destination = submission / name
        atomic_bytes(destination, source.read_bytes())
        records[name] = {**file_record(destination), "source": str(source), "source_sha256": sha256_file(source)}
    atomic_bytes(submission / "inflate.py", transformed_inflate_source(), executable=True)
    records["inflate.py"] = {
        **file_record(submission / "inflate.py"),
        "source": str(cp2.CP2_RUNTIME / "inflate.py"),
        "source_sha256": sha256_file(cp2.CP2_RUNTIME / "inflate.py"),
        "transformation_source": str(Path(__file__).resolve()),
    }
    manifest = {
        "schema": "ddm_sm4_runtime_dependencies.v1",
        "score_claim": False,
        "borrowed_runtime": str(cp2.REFERENCE_RUNTIME),
        "third_party_dependencies_unchanged": ["constriction==0.5.0", "numpy", "torch"],
        "files": records,
    }
    atomic_json(submission / "runtime-dependencies.json", manifest)
    records["runtime-dependencies.json"] = file_record(submission / "runtime-dependencies.json")
    return records


def load_staged_inflate(submission: Path) -> ModuleType:
    prior_path = list(sys.path)
    prior_modules = {name: sys.modules.get(name) for name in RUNTIME_MODULE_NAMES}
    for name in RUNTIME_MODULE_NAMES:
        sys.modules.pop(name, None)
    sys.path.insert(0, str(submission))
    spec = importlib.util.spec_from_file_location("ddm_sm4_staged_inflate", submission / "inflate.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load staged DDM-SM4 inflate runtime")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = prior_path
        for name in RUNTIME_MODULE_NAMES:
            sys.modules.pop(name, None)
            if prior_modules[name] is not None:
                sys.modules[name] = prior_modules[name]
    return module


def stage_survivor(args: argparse.Namespace) -> dict[str, Any]:
    output = require_ssd(args.output)
    grid = json.loads((output / "GRID_RESULT.json").read_text())
    selected = grid["mechanism"]["selected_weight_screen_survivor"]
    cell_root = output / "retained/grid" / selected
    cell = json.loads((cell_root / "receipt.json").read_text())
    verify_retained(cell)
    selected_root = output / "selected" / selected
    submission = selected_root / "submission"
    archive_source = Path(cell["retained"]["archive_a"]["path"])
    member_source = Path(cell["retained"]["member_a"]["path"])
    state_source = Path(cell["retained"]["decoded_state"]["path"])
    atomic_bytes(submission / "archive.zip", archive_source.read_bytes())
    atomic_bytes(submission / "archive/p", member_source.read_bytes())
    runtime_files = stage_runtime(submission)

    receiver = cp2.load_reference_receiver()
    member = cp2.read_stored_member(submission / "archive.zip")
    parts = receiver.split_payload(member)
    decoded_models = receiver.decode_models(parts.models, model_codec=parts.model_codec)
    core_models, temporal = receiver.split_optional_temporal_reversion(decoded_models.raw)
    semantic, semantic_end, carrier_bytes = cp2.semantic_field(core_models)
    runtime = load_staged_inflate(submission)
    semantic_model, basis, coeff = runtime.unpack_semantic_pose(core_models[: semantic_end + carrier_bytes])
    parsed_state = sm3.state_wire(semantic_model.state_dict())
    atomic_bytes(selected_root / "retained/parseback.semantic_state.sm3state", parsed_state)
    atomic_bytes(
        selected_root / "retained/parseback.basis.f32le",
        basis.detach().cpu().numpy().astype("<f4", copy=False).tobytes(order="C"),
    )
    atomic_bytes(
        selected_root / "retained/parseback.coeff.f32le",
        coeff.detach().cpu().numpy().astype("<f4", copy=False).tobytes(order="C"),
    )
    if parsed_state != state_source.read_bytes():
        raise RuntimeError("staged SM4 receiver state differs from grid packer state")
    temporal_path = selected_root / "retained/temporal_reversion.bin"
    if temporal is not None:
        atomic_bytes(temporal_path, temporal.packed)

    build_receipt = {
        "schema": "ddm_sm4_selected_archive.v1",
        "complete": True,
        "written_at_utc": utc_now(),
        "axis": "[macOS-CPU advisory; scorer-free shipped receiver parse-back]",
        "score_claim": False,
        "selected_candidate": selected,
        "selection_rule": grid["mechanism"]["selection_rule"],
        "actual_archive_bytes": cell["archive_bytes"],
        "actual_archive_delta_bytes": cell["archive_delta_vs_pr130_bytes"],
        "rate_only_delta_s": cell["rate_only_delta_s_vs_pr130"],
        "submission": str(submission.resolve()),
        "checks": {
            "archive_repeat_byte_equal": cell["checks"]["archive_repeat_byte_equal"],
            "outer_receiver_models_raw_byte_identical": True,
            "outer_receiver_token_payload_byte_identical": True,
            "semantic_parseback_byte_identical": True,
            "shipped_sm4r_state_equals_packer_state": True,
            "real_inflate_sh_staged": True,
            "real_inflate_sh_executed": False,
        },
        "retained": {
            "archive": file_record(submission / "archive.zip"),
            "submission_member": file_record(submission / "archive/p"),
            "semantic": cell["retained"]["semantic"],
            "semantic_state": file_record(selected_root / "retained/parseback.semantic_state.sm3state"),
            "basis": file_record(selected_root / "retained/parseback.basis.f32le"),
            "coeff": file_record(selected_root / "retained/parseback.coeff.f32le"),
        },
        "runtime_files": runtime_files,
    }
    if temporal is not None:
        build_receipt["retained"]["temporal_reversion"] = file_record(temporal_path)
    atomic_json(selected_root / "build_receipt.json", build_receipt)
    atomic_json(output / "SELECTED.json", build_receipt)
    return build_receipt


def inflate_survivor(args: argparse.Namespace) -> dict[str, Any]:
    selected = json.loads((args.output / "SELECTED.json").read_text())
    selected_root = Path(selected["submission"]).parent
    return cp2.inflate_candidate(
        selected_root,
        python=args.python,
        timeout_seconds=args.timeout_seconds,
        minimum_free_bytes=args.inflate_minimum_free_bytes,
    )


def parity_accumulator() -> dict[str, Any]:
    return {
        "count": 0,
        "changed": 0,
        "abs_sum": 0.0,
        "max_abs": 0,
        "base_sum": 0.0,
        "candidate_sum": 0.0,
        "base_square_sum": 0.0,
        "candidate_square_sum": 0.0,
        "cross_sum": 0.0,
    }


def update_parity(acc: dict[str, Any], base: np.ndarray, candidate: np.ndarray) -> None:
    left = base.astype(np.float64)
    right = candidate.astype(np.float64)
    delta = np.abs(left - right)
    acc["count"] += left.size
    acc["changed"] += int(np.count_nonzero(delta))
    acc["abs_sum"] += float(delta.sum())
    acc["max_abs"] = max(acc["max_abs"], int(delta.max(initial=0)))
    acc["base_sum"] += float(left.sum())
    acc["candidate_sum"] += float(right.sum())
    acc["base_square_sum"] += float(np.square(left).sum())
    acc["candidate_square_sum"] += float(np.square(right).sum())
    acc["cross_sum"] += float((left * right).sum())


def finish_parity(acc: Mapping[str, Any]) -> dict[str, Any]:
    count = int(acc["count"])
    base_mean = float(acc["base_sum"]) / count
    candidate_mean = float(acc["candidate_sum"]) / count
    base_var = float(acc["base_square_sum"]) / count - base_mean**2
    candidate_var = float(acc["candidate_square_sum"]) / count - candidate_mean**2
    covariance = float(acc["cross_sum"]) / count - base_mean * candidate_mean
    correlation = covariance / math.sqrt(base_var * candidate_var) if base_var > 0 and candidate_var > 0 else None
    return {
        "denominator_channel_values": count,
        "changed_channel_values": int(acc["changed"]),
        "changed_fraction": int(acc["changed"]) / count,
        "mean_abs_delta": float(acc["abs_sum"]) / count,
        "max_abs_delta": int(acc["max_abs"]),
        "base_mean": base_mean,
        "candidate_mean": candidate_mean,
        "mean_shift": candidate_mean - base_mean,
        "base_sd": math.sqrt(max(0.0, base_var)),
        "candidate_sd": math.sqrt(max(0.0, candidate_var)),
        "candidate_sd_over_base_sd": math.sqrt(max(0.0, candidate_var) / base_var),
        "pearson_correlation": correlation,
    }


def parity_screen(args: argparse.Namespace) -> dict[str, Any]:
    selected = json.loads((args.output / "SELECTED.json").read_text())
    selected_root = Path(selected["submission"]).parent
    inflate = json.loads((selected_root / "receiver_parseback/inflate_receipt.json").read_text())
    candidate_raw = Path(inflate["raw"]["path"])
    require_record(args.base_raw, size=BASE_RAW_BYTES, digest=BASE_RAW_SHA256, label="PR130 base raw")
    require_record(
        candidate_raw,
        size=int(inflate["raw"]["bytes"]),
        digest=str(inflate["raw"]["sha256"]),
        label="SM4 candidate raw",
    )
    frame_values = 874 * 1_164 * 3
    base = np.memmap(args.base_raw, dtype=np.uint8, mode="r", shape=(1_200, frame_values))
    candidate = np.memmap(candidate_raw, dtype=np.uint8, mode="r", shape=(1_200, frame_values))
    accumulators = {"even_pose_carrier": parity_accumulator(), "odd_semantic": parity_accumulator()}
    for first in range(0, 1_200, 8):
        last = min(1_200, first + 8)
        for parity, key in ((0, "even_pose_carrier"), (1, "odd_semantic")):
            indices = np.arange(first + ((parity - first) % 2), last, 2)
            if indices.size:
                update_parity(accumulators[key], base[indices], candidate[indices])
    metrics = {key: finish_parity(value) for key, value in accumulators.items()}
    result = {
        "schema": "ddm_sm4_frame_parity_screen.v1",
        "complete": True,
        "written_at_utc": utc_now(),
        "axis": "[macOS-CPU advisory; scorer-free exact retained RAW parity]",
        "score_claim": False,
        "candidate": selected["selected_candidate"],
        "frame_denominator": 1_200,
        "pair_denominator": 600,
        "parity_frame_denominator_each": 600,
        "metrics": metrics,
        "r32_int4_reference": {
            "scope": "full exact retained raw parity from DDM MAIN paired-eval receipt",
            "even_mean_abs_delta": 0.0,
            "even_changed_fraction": 0.0,
            "odd_mean_abs_delta": 67.71,
            "odd_changed_fraction": 0.9926,
            "sampled_odd_correlation_range": [0.30, 0.32],
        },
        "retained": {
            "base_raw": file_record(args.base_raw),
            "candidate_raw": file_record(candidate_raw),
            "candidate_archive": selected["retained"]["archive"],
        },
    }
    atomic_json(args.output / "PARITY_SCREEN.json", result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("grid", "stage", "inflate", "parity", "all"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume-from", type=Path, default=DEFAULT_RESUME)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--token-driver", type=Path, default=DEFAULT_TOKEN_DRIVER)
    parser.add_argument("--base-raw", type=Path, default=DEFAULT_BASE_RAW)
    parser.add_argument("--python", type=Path, default=DEFAULT_RECEIVER_PYTHON)
    parser.add_argument("--minimum-free-bytes", type=int, default=2 << 30)
    parser.add_argument("--inflate-minimum-free-bytes", type=int, default=12 << 30)
    parser.add_argument("--timeout-seconds", type=int, default=1_800)
    args = parser.parse_args()
    args.output = require_ssd(args.output)
    args.resume_from = require_ssd(args.resume_from)
    if args.minimum_free_bytes <= 0 or args.inflate_minimum_free_bytes <= 0 or args.timeout_seconds <= 0:
        parser.error("storage floors and timeout must be positive")
    return args


def main() -> None:
    args = parse_args()
    commands = ("grid", "stage", "inflate", "parity") if args.command == "all" else (args.command,)
    result: dict[str, Any] = {}
    for command in commands:
        if command == "grid":
            result = run_grid(args)
        elif command == "stage":
            result = stage_survivor(args)
        elif command == "inflate":
            result = inflate_survivor(args)
        else:
            result = parity_screen(args)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
