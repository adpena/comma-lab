#!/usr/bin/env python3
"""RX2 Stage-0 telemetry and exact current-MC36 training-cache materializer.

The entropy rungs in this file are deliberately weak count models.  They are
NON-KILL-AUTHORITY for the IHS1 family: a projected win accelerates training,
but no projected loss can stop the charter-mandated real neural-prior fit.

Large, rebuildable artifacts are written to APDataStore because the charter's
VertigoDataTier work root has insufficient free space.  Small receipts remain
at the mandated root and bind every spill artifact by path, bytes, and SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any

import brotli
import numpy as np
import torch

WORK_ROOT = Path("/Volumes/VertigoDataTier/pact/ddm_rx2_current_mc36_label_hpac")
BULK_ROOT = Path("/Volumes/APDataStore/pact/ddm_rx2_current_mc36_label_hpac")
SPATIAL_TOKENS = Path(
    "/Volumes/VertigoDataTier/pact/ddm_rx1_rate_attack_20260814/retained/coders/"
    "tq1c_table_on/decoded_spatial_tokens.rc64.bin"
)
MC36_ARCHIVE = Path("/Volumes/VertigoDataTier/pact/ddm_mc35_successor_drop532_pair105/micro35_candidate/archive.zip")
EXPECTED_SPATIAL_SHA256 = "9ba2e52b3096585895970066b389bf1261ebc203d5b828cdea056c13858aea52"
EXPECTED_ARCHIVE_SHA256 = "f0ba4bb41d55fff85542f2a17dfe682508aa4f9ab50ef51cda573d79f0c4b1de"
TOKEN_SHAPE = (600, 384, 512)
TOKEN_COUNT = math.prod(TOKEN_SHAPE)
MC36_MODEL_BYTES = 70_835
MC36_TOKEN_BYTES = 115_238
MC36_MODEL_TOKEN_BYTES = MC36_MODEL_BYTES + MC36_TOKEN_BYTES
MC36_ARCHIVE_BYTES = 186_269
MIN_VERTIGO_FREE_BYTES = 256 << 20
MIN_BULK_FREE_BYTES = 10 << 30


class RX2Stage0Error(RuntimeError):
    """Fail-closed error for an RX2 Stage-0 custody or telemetry defect."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with tmp.open("wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _atomic_json(path: Path, payload: Any) -> None:
    _atomic_bytes(path, (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode())


def _atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(payload, tmp)
        with tmp.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def _artifact(path: Path, *, role: str) -> dict[str, Any]:
    return {
        "path": str(path),
        "role": role,
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def preflight() -> dict[str, Any]:
    pins = (
        (SPATIAL_TOKENS, TOKEN_COUNT, EXPECTED_SPATIAL_SHA256, "MC36 spatial token field"),
        (MC36_ARCHIVE, MC36_ARCHIVE_BYTES, EXPECTED_ARCHIVE_SHA256, "MC36 archive"),
    )
    rows = []
    for path, expected_bytes, expected_sha256, role in pins:
        if not path.is_file():
            raise RX2Stage0Error(f"missing pinned {role}: {path}")
        observed_bytes = path.stat().st_size
        observed_sha256 = _sha256_file(path)
        if observed_bytes != expected_bytes or observed_sha256 != expected_sha256:
            raise RX2Stage0Error(
                f"pinned {role} changed: expected {expected_bytes}/{expected_sha256}, "
                f"observed {observed_bytes}/{observed_sha256}"
            )
        rows.append(_artifact(path, role=role))
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    BULK_ROOT.mkdir(parents=True, exist_ok=True)
    vertigo_free = shutil.disk_usage(WORK_ROOT).free
    bulk_free = shutil.disk_usage(BULK_ROOT).free
    if vertigo_free < MIN_VERTIGO_FREE_BYTES:
        raise RX2Stage0Error(f"Vertigo metadata tier has {vertigo_free} free bytes; need {MIN_VERTIGO_FREE_BYTES}")
    if bulk_free < MIN_BULK_FREE_BYTES:
        raise RX2Stage0Error(f"APDataStore bulk tier has {bulk_free} free bytes; need {MIN_BULK_FREE_BYTES}")
    return {
        "schema": "ddm_rx2_stage0_preflight.v1",
        "status": "PASS",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "authority": "[byte-only scorer-free]",
        "score_claim": False,
        "work_root": str(WORK_ROOT),
        "bulk_root": str(BULK_ROOT),
        "storage_routing_reason": (
            "Vertigo had insufficient capacity for multi-gigabyte RX2 outputs; "
            "small receipts stay under the charter root and bulk spills to APDataStore"
        ),
        "vertigo_free_bytes": vertigo_free,
        "bulk_free_bytes": bulk_free,
        "inputs": rows,
    }


def _conditional_entropy_bits(counts: np.ndarray) -> float:
    counts64 = counts.astype(np.float64, copy=False)
    totals = counts64.sum(axis=1, keepdims=True)
    probabilities = np.divide(counts64, totals, out=np.zeros_like(counts64), where=totals > 0)
    terms = np.zeros_like(probabilities)
    positive = probabilities > 0
    terms[positive] = -counts64[positive] * np.log2(probabilities[positive])
    return float(terms.sum())


def _serialize_count_table(name: str, counts: np.ndarray) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    table_dir = WORK_ROOT / "stage0" / "count_tables"
    raw_path = table_dir / f"{name}.u64le"
    compressed_path = table_dir / f"{name}.u64le.br11"
    shape_path = table_dir / f"{name}.shape.json"
    raw = np.asarray(counts, dtype="<u8", order="C").tobytes(order="C")
    compressed = brotli.compress(raw, mode=brotli.MODE_GENERIC, quality=11)
    _atomic_bytes(raw_path, raw)
    _atomic_bytes(compressed_path, compressed)
    _atomic_json(
        shape_path,
        {
            "schema": "ddm_rx2_count_table_shape.v1",
            "name": name,
            "dtype": "<u8",
            "shape": list(counts.shape),
            "sum": int(counts.sum()),
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
        },
    )
    artifacts = [
        _artifact(raw_path, role=f"{name}_count_table_raw"),
        _artifact(compressed_path, role=f"{name}_count_table_brotli_q11"),
        _artifact(shape_path, role=f"{name}_count_table_shape"),
    ]
    return (
        {
            "table_representation": "Brotli-q11-compressed little-endian uint64 counts",
            "table_bytes": compressed_path.stat().st_size,
            "table_path": str(compressed_path),
            "table_sha256": _sha256_file(compressed_path),
        },
        artifacts,
    )


def telemetry() -> dict[str, Any]:
    before = preflight()
    tokens = np.memmap(SPATIAL_TOKENS, mode="r", dtype=np.uint8, shape=TOKEN_SHAPE)
    order0 = np.zeros((1, 5), dtype=np.uint64)
    spatial = np.zeros((36, 5), dtype=np.uint64)
    spatial_previous = np.zeros((216, 5), dtype=np.uint64)
    started = time.time()
    for frame_index in range(TOKEN_SHAPE[0]):
        frame = tokens[frame_index]
        order0[0] += np.bincount(frame.reshape(-1), minlength=5).astype(np.uint64)
        previous = None if frame_index == 0 else tokens[frame_index - 1]
        for row_index in range(TOKEN_SHAPE[1]):
            target = frame[row_index]
            left = np.empty(TOKEN_SHAPE[2], dtype=np.uint8)
            left[0] = 5
            left[1:] = target[:-1]
            up = np.full(TOKEN_SHAPE[2], 5, dtype=np.uint8) if row_index == 0 else frame[row_index - 1]
            spatial_context = left.astype(np.int16) + 6 * up.astype(np.int16)
            spatial += (
                np.bincount(
                    spatial_context * 5 + target,
                    minlength=36 * 5,
                )
                .reshape(36, 5)
                .astype(np.uint64)
            )
            previous_token = np.full(TOKEN_SHAPE[2], 5, dtype=np.uint8) if previous is None else previous[row_index]
            combined_context = spatial_context * 6 + previous_token.astype(np.int16)
            spatial_previous += (
                np.bincount(
                    combined_context * 5 + target,
                    minlength=216 * 5,
                )
                .reshape(216, 5)
                .astype(np.uint64)
            )
    if not (int(order0.sum()) == int(spatial.sum()) == int(spatial_previous.sum()) == TOKEN_COUNT):
        raise RX2Stage0Error("count-model denominators do not equal the exact token population")
    rung_specs = (
        ("order0", order0, "no context"),
        ("causal_spatial", spatial, "left token plus upper token; boundary sentinel=5"),
        (
            "causal_spatial_previous_frame",
            spatial_previous,
            "left plus upper plus same-position previous-frame token; boundary sentinel=5",
        ),
    )
    artifacts: list[dict[str, Any]] = []
    rungs = []
    for name, counts, context in rung_specs:
        entropy_bits = _conditional_entropy_bits(counts)
        table_row, table_artifacts = _serialize_count_table(name, counts)
        artifacts.extend(table_artifacts)
        projected_token_bytes = math.ceil(entropy_bits / 8)
        projected_total = MC36_MODEL_BYTES + projected_token_bytes + table_row["table_bytes"]
        rungs.append(
            {
                "rung": name,
                "context": context,
                "population_tokens": TOKEN_COUNT,
                "context_states": int(counts.shape[0]),
                "empirical_conditional_entropy_bits_per_token": entropy_bits / TOKEN_COUNT,
                "ideal_empirical_token_bytes": projected_token_bytes,
                "model_bytes_held_at_mc36": MC36_MODEL_BYTES,
                **table_row,
                "projected_model_tokens_table_bytes": projected_total,
                "delta_vs_mc36_model_tokens_bytes": projected_total - MC36_MODEL_TOKEN_BYTES,
                "accelerated_go": projected_total < MC36_MODEL_TOKEN_BYTES,
                "authority": "NON-KILL-AUTHORITY",
            }
        )
    result_path = WORK_ROOT / "stage0" / "stage0_entropy_telemetry.json"
    result = {
        "schema": "ddm_rx2_stage0_entropy_telemetry.v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "axis": "[byte-only scorer-free; empirical entropy projection]",
        "score_claim": False,
        "authority": "NON-KILL-AUTHORITY",
        "verdict_boundary": (
            "These exact full-field count-model entropies bound only weak context classes. "
            "They may accelerate GO and can never justify NO-GO for trained IHS1."
        ),
        "elapsed_seconds": time.time() - started,
        "input": before["inputs"][0],
        "incumbent": {
            "model_bytes": MC36_MODEL_BYTES,
            "token_bytes": MC36_TOKEN_BYTES,
            "model_plus_token_bytes": MC36_MODEL_TOKEN_BYTES,
            "archive_bytes": MC36_ARCHIVE_BYTES,
        },
        "rungs": rungs,
        "accelerated_go": any(row["accelerated_go"] for row in rungs),
        "artifacts": artifacts,
    }
    _atomic_json(result_path, result)
    result["result_artifact"] = _artifact(result_path, role="stage0_telemetry")
    _atomic_json(result_path, result)
    return result


def materialize_cache() -> dict[str, Any]:
    before = preflight()
    cache_path = BULK_ROOT / "inputs" / "mc36_spatial_tokens_uint8.pt"
    source = np.memmap(SPATIAL_TOKENS, mode="r", dtype=np.uint8, shape=TOKEN_SHAPE)
    seg = torch.from_numpy(np.array(source, copy=True, order="C"))
    payload = {
        "schema": "ddm_rx2_mc36_training_cache.v1",
        "seg": seg,
        "spatial_token_sha256": EXPECTED_SPATIAL_SHA256,
        "source_path": str(SPATIAL_TOKENS),
        "source_bytes": TOKEN_COUNT,
        "shape": list(TOKEN_SHAPE),
        "dtype": "uint8",
    }
    if cache_path.exists():
        existing = torch.load(cache_path, map_location="cpu", weights_only=False)
        existing_seg = existing.get("seg") if isinstance(existing, dict) else None
        if not isinstance(existing_seg, torch.Tensor) or not torch.equal(existing_seg, seg):
            raise RX2Stage0Error(f"existing cache differs; refusing overwrite: {cache_path}")
    else:
        _atomic_torch_save(cache_path, payload)
    loaded = torch.load(cache_path, map_location="cpu", weights_only=False)
    raw_sha256 = hashlib.sha256(loaded["seg"].contiguous().numpy().tobytes(order="C")).hexdigest()
    if raw_sha256 != EXPECTED_SPATIAL_SHA256:
        raise RX2Stage0Error("materialized training cache changed the exact token bytes")
    result = {
        "schema": "ddm_rx2_mc36_training_cache_receipt.v1",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "axis": "[byte-only scorer-free]",
        "score_claim": False,
        "source": before["inputs"][0],
        "cache": _artifact(cache_path, role="exact_mc36_uint8_training_cache"),
        "cache_tensor_raw_sha256": raw_sha256,
        "exact_token_identity": True,
    }
    receipt_path = WORK_ROOT / "inputs" / "mc36_training_cache_receipt.json"
    _atomic_json(receipt_path, result)
    result["receipt"] = _artifact(receipt_path, role="training_cache_receipt")
    _atomic_json(receipt_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "telemetry", "cache", "stage0"))
    args = parser.parse_args()
    if args.stage == "preflight":
        result = preflight()
    elif args.stage == "telemetry":
        result = telemetry()
    elif args.stage == "cache":
        result = materialize_cache()
    else:
        result = {"telemetry": telemetry(), "cache": materialize_cache()}
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
