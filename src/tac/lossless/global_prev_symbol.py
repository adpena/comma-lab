from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from .arithmetic import flatten_tokens_for_gpt_arithmetic
from .data import TokenRecord, build_token_records, commavq_original_bytes, load_commavq_dataset
from .frequency_coder import decode_uint16_prev_symbol_stream, encode_uint16_prev_symbol_stream
from .transforms import apply_frame_order, invert_frame_order, recursive_bisect_frame_order


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


def _resolve_frame_order(frame_order: str, frame_count: int) -> np.ndarray | None:
    if frame_order == "canonical":
        return None
    if frame_order == "recursive_bisect":
        return recursive_bisect_frame_order(frame_count)
    raise ValueError(f"unsupported frame_order: {frame_order}")


def _clip_signature(record: TokenRecord) -> np.ndarray:
    arr = np.asarray(record.tokens, dtype=np.float32)
    flat = arr.reshape(arr.shape[0], -1)
    first = flat[0]
    middle = flat[flat.shape[0] // 2]
    last = flat[-1]
    mean = flat.mean(axis=0)
    return np.concatenate([first, middle, last, mean], axis=0)


def _transition_signature(record: TokenRecord, *, vocab_size: int = 64) -> np.ndarray:
    arr = np.asarray(record.tokens, dtype=np.int32).reshape(-1)
    clipped = np.clip(arr, 0, vocab_size - 1)
    hist = np.bincount(clipped, minlength=vocab_size).astype(np.float32)
    deltas = np.abs(np.diff(arr, prepend=arr[:1]))
    delta_bins = np.minimum(deltas // 16, vocab_size - 1)
    delta_hist = np.bincount(delta_bins, minlength=vocab_size).astype(np.float32)
    head = arr[:256].astype(np.float32)
    tail = arr[-256:].astype(np.float32)
    return np.concatenate([hist, delta_hist, head, tail], axis=0)


def _greedy_nn_order(features: np.ndarray, records: list[TokenRecord]) -> list[int]:
    distances = np.abs(features[:, None, :] - features[None, :, :]).sum(axis=2)
    current = int(np.argmin(distances.sum(axis=1)))
    remaining = set(range(len(records)))
    order: list[int] = []
    while remaining:
        order.append(current)
        remaining.remove(current)
        if not remaining:
            break
        current = min(remaining, key=lambda idx: (float(distances[current, idx]), records[idx].file_name))
    return order


def _recursive_pca_order(features: np.ndarray, records: list[TokenRecord], indices: list[int]) -> list[int]:
    if len(indices) <= 2:
        return list(indices)
    sub = features[indices]
    centered = sub - sub.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    direction = vt[0]
    projections = centered @ direction
    ranked = sorted(zip(indices, projections.tolist()), key=lambda item: (item[1], records[item[0]].file_name))
    midpoint = len(ranked) // 2
    left = [idx for idx, _value in ranked[:midpoint]]
    right = [idx for idx, _value in ranked[midpoint:]]
    return _recursive_pca_order(features, records, left) + _recursive_pca_order(features, records, right)


def _normalize_label(value: object) -> tuple[object, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(value)
    return (value,)


def _exact_clip_rank_key(record: TokenRecord) -> tuple[int, ...]:
    arr = np.asarray(record.tokens, dtype=np.int64)
    flat = arr.reshape(arr.shape[0], -1)
    summary = np.concatenate(
        [
            flat[0],
            flat[flat.shape[0] // 2],
            flat[-1],
            flat.sum(axis=0, dtype=np.int64),
        ],
        axis=0,
    )
    return tuple(int(value) for value in summary.tolist())


def _label_slot_key(label: tuple[object, ...], *, slots: tuple[int, ...]) -> tuple[object, ...]:
    return tuple(label[index] if index < len(label) else None for index in slots)


def _greedy_bucket_order(groups: dict[tuple[object, ...], list[TokenRecord]]) -> list[tuple[object, ...]]:
    bucket_keys = list(groups)
    if len(bucket_keys) <= 1:
        return bucket_keys
    features = np.stack(
        [
            np.stack([_clip_signature(record) for record in groups[bucket_key]], axis=0).mean(axis=0)
            for bucket_key in bucket_keys
        ],
        axis=0,
    )
    distances = np.abs(features[:, None, :] - features[None, :, :]).sum(axis=2)
    current = int(np.argmin(distances.sum(axis=1)))
    remaining = set(range(len(bucket_keys)))
    order: list[int] = []
    while remaining:
        order.append(current)
        remaining.remove(current)
        if not remaining:
            break
        current = min(remaining, key=lambda idx: (float(distances[current, idx]), repr(bucket_keys[idx])))
    return [bucket_keys[idx] for idx in order]


def order_token_records(
    records: list[TokenRecord],
    *,
    strategy: str,
    label_map: dict[str, object] | None = None,
    explicit_order: list[str] | None = None,
) -> list[TokenRecord]:
    items = list(records)
    if strategy == "canonical":
        return sorted(items, key=lambda item: item.file_name)
    if strategy == "explicit":
        if explicit_order is None:
            raise ValueError("explicit_order is required for explicit strategy")
        by_name = {record.file_name: record for record in items}
        ordered = []
        for file_name in explicit_order:
            if file_name not in by_name:
                raise ValueError(f"explicit_order references unknown file_name: {file_name}")
            ordered.append(by_name[file_name])
        if len(ordered) != len(items):
            raise ValueError("explicit_order must contain every file exactly once")
        return ordered
    if strategy == "clip_greedy_nn":
        features = np.stack([_clip_signature(record) for record in items], axis=0)
        order = _greedy_nn_order(features, items)
        return [items[idx] for idx in order]
    if strategy == "clip_recursive_pca":
        features = np.stack([_clip_signature(record) for record in items], axis=0)
        order = _recursive_pca_order(features, items, list(range(len(items))))
        return [items[idx] for idx in order]
    if strategy == "transition_recursive_pca":
        features = np.stack([_transition_signature(record) for record in items], axis=0)
        order = _recursive_pca_order(features, items, list(range(len(items))))
        return [items[idx] for idx in order]
    if strategy == "label_grouped_clip_greedy_nn":
        if label_map is None:
            raise ValueError("label_map is required for label_grouped_clip_greedy_nn")
        groups: dict[tuple[object, ...], list[TokenRecord]] = {}
        for record in items:
            label = _normalize_label(label_map.get(record.file_name, ("unlabeled",)))
            groups.setdefault(label, []).append(record)
        ordered: list[TokenRecord] = []
        for label in sorted(groups):
            group = groups[label]
            features = np.stack([_clip_signature(record) for record in group], axis=0)
            order = _greedy_nn_order(features, group)
            ordered.extend(group[idx] for idx in order)
        return ordered
    if strategy == "hybrid_thresh8_parent046_label_greedy":
        if label_map is None:
            raise ValueError("label_map is required for hybrid_thresh8_parent046_label_greedy")
        labels_by_name = {
            record.file_name: _normalize_label(label_map.get(record.file_name, ("unlabeled",)))
            for record in items
        }
        full_label_counts: dict[tuple[object, ...], int] = {}
        for label in labels_by_name.values():
            full_label_counts[label] = full_label_counts.get(label, 0) + 1
        groups: dict[tuple[object, ...], list[TokenRecord]] = {}
        for record in items:
            label = labels_by_name[record.file_name]
            if full_label_counts[label] > 8:
                bucket_key = ("full",) + label
            else:
                bucket_key = ("parent",) + _label_slot_key(label, slots=(0, 4, 6))
            groups.setdefault(bucket_key, []).append(record)
        ordered: list[TokenRecord] = []
        for bucket_key in _greedy_bucket_order(groups):
            ordered.extend(
                sorted(
                    groups[bucket_key],
                    key=lambda record: (_exact_clip_rank_key(record), record.file_name),
                )
            )
        return ordered
    if strategy == "label_lexicographic_clip_rank":
        if label_map is None:
            raise ValueError("label_map is required for label_lexicographic_clip_rank")
        return sorted(
            items,
            key=lambda record: (
                _normalize_label(label_map.get(record.file_name, ("unlabeled",))),
                _exact_clip_rank_key(record),
                record.file_name,
            ),
        )
    raise ValueError(f"unsupported strategy: {strategy}")


def benchmark_global_prev_symbol_record_order_sample(
    *,
    output_path: str | Path,
    split=None,
    max_records: int = 64,
    strategy: str = "canonical",
    frame_order: str = "canonical",
    labels_path: str | Path | None = None,
    order_path: str | Path | None = None,
) -> dict[str, object]:
    if max_records <= 0:
        raise ValueError("max_records must be positive")

    dataset = load_commavq_dataset(split=split, num_proc=1)
    train_split = dataset["train"]
    if hasattr(train_split, "__getitem__"):
        examples = [train_split[index] for index in range(max_records)]
    else:
        examples = []
        for example in train_split:
            examples.append(example)
            if len(examples) >= max_records:
                break
    records = build_token_records(examples)
    label_map = None
    explicit_order = None
    if labels_path is not None:
        payload = json.loads(Path(labels_path).read_text())
        if not isinstance(payload, dict):
            raise ValueError("labels_path must point to a JSON object mapping file_name to labels")
        label_map = payload
    if order_path is not None:
        payload = json.loads(Path(order_path).read_text())
        if isinstance(payload, dict) and "records" in payload:
            explicit_order = [str(item["file_name"]) for item in payload["records"]]
        elif isinstance(payload, list):
            explicit_order = [str(item) for item in payload]
        else:
            raise ValueError("order_path must contain a JSON list of file names or an object with a records array")
    ordered_records = order_token_records(
        records,
        strategy=strategy,
        label_map=label_map,
        explicit_order=explicit_order,
    )

    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded_dir = target.with_suffix("")
    if encoded_dir.exists():
        for path in sorted(encoded_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
    encode_corpus_global_prev_symbol_position_major(
        records=ordered_records,
        output_dir=encoded_dir,
        chunk_count=1,
        frame_order=frame_order,
        preserve_input_order=(strategy != "canonical"),
    )
    archive_bytes = sum(path.stat().st_size for path in encoded_dir.rglob("*") if path.is_file())
    with tempfile.TemporaryDirectory() as tmpdir:
        decoded_dir = Path(tmpdir) / "decoded"
        decode_corpus_global_prev_symbol_position_major(encoded_dir=encoded_dir, output_dir=decoded_dir)
        exact_match = all(
            np.array_equal(np.load(decoded_dir / record.file_name, allow_pickle=False), np.asarray(record.tokens))
            for record in records
        )
    result = {
        "command": "lossless_global_prev_symbol_order_sample",
        "output_path": str(target),
        "encoded_dir": str(encoded_dir),
        "strategy": strategy,
        "frame_order": frame_order,
        "labels_path": str(labels_path) if labels_path is not None else None,
        "order_path": str(order_path) if order_path is not None else None,
        "max_records": len(records),
        "archive_bytes": archive_bytes,
        "original_bytes": commavq_original_bytes(len(records)),
        "compression_ratio": commavq_original_bytes(len(records)) / archive_bytes,
        "exact_match": exact_match,
    }
    target.write_text(json.dumps(result, indent=2) + "\n")
    return result


def _record_payload(record: TokenRecord, *, frame_order: str = "canonical") -> np.ndarray:
    tokens = np.asarray(record.tokens)
    permutation = _resolve_frame_order(frame_order, int(tokens.shape[0]))
    if permutation is not None:
        tokens = apply_frame_order(tokens, permutation)
    return flatten_tokens_for_gpt_arithmetic(tokens, layout="position_major").astype(np.uint16)


def encode_corpus_global_prev_symbol_position_major(
    *,
    records: list[TokenRecord],
    output_dir: str | Path,
    chunk_count: int = 1,
    frame_order: str = "canonical",
    preserve_input_order: bool = False,
) -> dict[str, object]:
    if chunk_count <= 0:
        raise ValueError("chunk_count must be positive")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    ordered = list(records) if preserve_input_order else sorted(records, key=lambda item: item.file_name)
    payloads = [_record_payload(record, frame_order=frame_order) for record in ordered]
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
        "frame_order": frame_order,
        "preserve_input_order": preserve_input_order,
        "records": manifest_records,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return {
        "output_dir": str(root),
        "record_count": len(ordered),
        "chunk_count": len(slices),
        "frame_order": frame_order,
        "record_order": "input" if preserve_input_order else "canonical",
    }


def decode_corpus_global_prev_symbol_position_major(*, encoded_dir: str | Path, output_dir: str | Path) -> dict[str, object]:
    root = Path(encoded_dir)
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((root / "manifest.json").read_text())
    frame_order = str(manifest.get("frame_order", "canonical"))
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
        permutation = _resolve_frame_order(frame_order, int(frames.shape[0]))
        if permutation is not None:
            frames = invert_frame_order(frames, permutation).astype(np.int16, copy=False)
        with (target / str(record["file_name"])).open("wb") as handle:
            np.save(handle, frames)

    return {
        "output_dir": str(target),
        "record_count": len(manifest["records"]),
        "frame_order": frame_order,
    }
