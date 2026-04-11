from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .arithmetic import flatten_tokens_for_gpt_arithmetic
from .data import TokenRecord
from .frequency_coder import decode_uint16_prev_symbol_stream, encode_uint16_prev_symbol_stream


def _chunk_slices(count: int, chunk_count: int) -> list[slice]:
    if chunk_count <= 0:
        raise ValueError("chunk_count must be positive")
    actual = min(chunk_count, count)
    base, remainder = divmod(count, actual)
    offsets = []
    start = 0
    for index in range(actual):
        size = base + (1 if index < remainder else 0)
        offsets.append(slice(start, start + size))
        start += size
    return offsets


def _record_payload(record: TokenRecord) -> np.ndarray:
    return flatten_tokens_for_gpt_arithmetic(record.tokens, layout="position_major").astype(np.uint16)


def encode_corpus_global_prev_symbol_position_major(
    *,
    records: list[TokenRecord],
    output_dir: str | Path,
    chunk_count: int = 1,
) -> dict[str, object]:
    if chunk_count <= 0:
        raise ValueError("chunk_count must be positive")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    ordered = sorted(records, key=lambda item: item.file_name)
    payloads = [_record_payload(record) for record in ordered]
    slices = _chunk_slices(len(payloads), chunk_count)
    manifest_records: list[dict[str, object]] = []

    for chunk_index, chunk_slice in enumerate(slices):
        chunk_payloads = payloads[chunk_slice]
        concatenated = np.concatenate(chunk_payloads) if chunk_payloads else np.array([], dtype=np.uint16)
        encoded = encode_uint16_prev_symbol_stream(concatenated)
        (root / f"chunk_{chunk_index:03d}.tpc").write_bytes(encoded.encoded_bytes)
        for record, payload in zip(ordered[chunk_slice], chunk_payloads):
            manifest_records.append(
                {
                    "file_name": record.file_name,
                    "token_count": int(payload.size),
                    "chunk_index": chunk_index,
                }
            )

    manifest = {
        "chunk_count": len(slices),
        "records": manifest_records,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "output_dir": str(root),
        "record_count": len(ordered),
        "chunk_count": len(slices),
    }


def decode_corpus_global_prev_symbol_position_major(*, encoded_dir: str | Path, output_dir: str | Path) -> dict[str, object]:
    root = Path(encoded_dir)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "manifest.json").read_text())
    decoded_chunks: dict[int, np.ndarray] = {}
    chunk_offsets: dict[int, int] = {}

    for record in manifest["records"]:
        chunk_index = int(record["chunk_index"])
        if chunk_index not in decoded_chunks:
            decoded_chunks[chunk_index] = decode_uint16_prev_symbol_stream((root / f"chunk_{chunk_index:03d}.tpc").read_bytes())
            chunk_offsets[chunk_index] = 0
        decoded = decoded_chunks[chunk_index]
        offset = chunk_offsets[chunk_index]
        token_count = int(record["token_count"])
        payload = decoded[offset : offset + token_count]
        chunk_offsets[chunk_index] = offset + token_count
        body = payload[:-1]
        streams = body.reshape(128, -1)
        frames = streams[:, 1:].T.reshape(-1, 8, 16).astype(np.int16)
        with (target / str(record["file_name"])).open("wb") as handle:
            np.save(handle, frames)

    return {
        "output_dir": str(target),
        "record_count": len(manifest["records"]),
    }
