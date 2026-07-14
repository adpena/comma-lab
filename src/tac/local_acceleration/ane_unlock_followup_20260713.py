# SPDX-License-Identifier: MIT
"""Receipt and stored-NPZ utilities for the local ANE follow-up lane."""

from __future__ import annotations

import hashlib
import json
import os
import struct
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

LANE_ID = "lane_ane_unlock_followup_20260713"
CHECKPOINT_ID = "ane_unlock_followup"
LOCAL_AXIS = "[macOS ANE/CoreML/MLX local advisory] NON-PROMOTABLE"
N600 = 600


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=target.parent, prefix=f".{target.name}.", delete=False
    ) as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, target)


def base_receipt(**extra: Any) -> dict[str, Any]:
    return {
        "schema": "ane_unlock_followup_receipt.v1",
        "lane_id": LANE_ID,
        "checkpoint_id": CHECKPOINT_ID,
        "axis": LOCAL_AXIS,
        "research_only": True,
        "score_claim": False,
        "promotion_eligible": False,
        "pointer_moved": False,
        **extra,
    }


def require_real_n600(count: int) -> None:
    if isinstance(count, bool) or int(count) != N600:
        raise ValueError(f"full fidelity authority in this lane requires exactly n600, got {count!r}")


def stored_npy_memmap(npz_path: str | Path, member: str) -> np.memmap:
    """Memory-map an uncompressed ``.npy`` member without duplicating its bytes.

    The canonical n600 cache stores members with ZIP_STORED.  Compressed members
    fail closed because mapping compressed bytes would silently corrupt inputs.
    """

    archive_path = Path(npz_path)
    with zipfile.ZipFile(archive_path) as archive:
        info = archive.getinfo(member)
        if info.compress_type != zipfile.ZIP_STORED:
            raise ValueError(f"{member} is compressed; a zero-copy memmap is not possible")
        header_offset = int(info.header_offset)
    with archive_path.open("rb") as handle:
        handle.seek(header_offset)
        raw = handle.read(30)
        if len(raw) != 30:
            raise ValueError("truncated ZIP local header")
        signature, _, _, _, _, _, _, _, _, name_len, extra_len = struct.unpack(
            "<IHHHHHIIIHH", raw
        )
        if signature != 0x04034B50:
            raise ValueError("invalid ZIP local header signature")
        npy_start = header_offset + 30 + name_len + extra_len
        handle.seek(npy_start)
        version = np.lib.format.read_magic(handle)
        if version == (1, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(handle)
        elif version == (2, 0):
            shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(handle)
        else:
            raise ValueError(f"unsupported npy version {version}")
        data_offset = handle.tell()
    order = "F" if fortran_order else "C"
    return np.memmap(archive_path, mode="r", dtype=dtype, offset=data_offset, shape=shape, order=order)


def flip_summary(reference_logits: np.ndarray, candidate_logits: np.ndarray) -> dict[str, Any]:
    if reference_logits.shape != candidate_logits.shape:
        raise ValueError(f"shape mismatch {reference_logits.shape} != {candidate_logits.shape}")
    if reference_logits.ndim != 4:
        raise ValueError("expected NCHW logits")
    flips = reference_logits.argmax(1) != candidate_logits.argmax(1)
    per_pair = flips.reshape(flips.shape[0], -1).mean(1)
    return {
        "n_real_states": int(flips.shape[0]),
        "argmax_flips": int(flips.sum()),
        "total_pixels": int(flips.size),
        "aggregate_flip_fraction": float(flips.mean()),
        "worst_pair_index": int(np.argmax(per_pair)),
        "worst_pair_flip_fraction": float(np.max(per_pair)),
        "per_pair_flip_fraction": [float(value) for value in per_pair],
    }


__all__ = [
    "CHECKPOINT_ID",
    "LANE_ID",
    "LOCAL_AXIS",
    "N600",
    "atomic_json",
    "base_receipt",
    "flip_summary",
    "require_real_n600",
    "sha256_file",
    "stored_npy_memmap",
]
