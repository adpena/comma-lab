from __future__ import annotations

from dataclasses import dataclass
import json
from functools import partial
from pathlib import Path
from collections.abc import Mapping, Sequence
from typing import Optional

from .data import COMMAVQ_DATASET_NAME, load_commavq_dataset
from .profiles import PROFILES

FRAME_BOS_TOKEN = 1024
SEGMENT_EOT_TOKEN = 1025
LAYOUTS = {"frame_major", "position_major"}


@dataclass(frozen=True)
class GPTArithmeticProfileConfig:
    profile: str
    method: str
    model: str
    context_tokens: int
    dataset_name: str = COMMAVQ_DATASET_NAME

    def __post_init__(self) -> None:
        if self.method != "gpt_arithmetic":
            raise ValueError(f"unsupported arithmetic method: {self.method}")
        if self.model not in {"small", "large"}:
            raise ValueError(f"unsupported arithmetic model: {self.model}")
        if self.context_tokens <= 0:
            raise ValueError("context_tokens must be positive")


@dataclass(frozen=True)
class GPTArithmeticPlan:
    profile: str
    method: str
    model: str
    context_tokens: int
    dataset_name: str
    layout: str
    split: tuple[str, ...]
    work_dir: str | None
    status: str = "planned"
    measured: bool = False

    def __post_init__(self) -> None:
        if self.method != "gpt_arithmetic":
            raise ValueError(f"unsupported arithmetic method: {self.method}")
        if not self.profile.strip():
            raise ValueError("profile must be a non-empty string")
        if self.layout not in LAYOUTS:
            raise ValueError(f"unsupported arithmetic layout: {self.layout}")
        if not self.split:
            raise ValueError("split must contain at least one entry")
        if self.status != "planned":
            raise ValueError("gpt arithmetic scaffold only supports planned status")
        if self.measured:
            raise ValueError("gpt arithmetic scaffold does not claim measured results")


@dataclass(frozen=True)
class GPTArithmeticEstimate:
    profile: str
    method: str
    model: str
    context_tokens: int
    dataset_name: str
    layout: str
    split: tuple[str, ...]
    work_dir: str | None
    example_count: int
    frames_per_example: int
    tokens_per_frame: int
    flat_tokens_per_example: int
    total_flat_tokens: int
    status: str = "estimated"
    measured: bool = False

    def __post_init__(self) -> None:
        if self.method != "gpt_arithmetic":
            raise ValueError(f"unsupported arithmetic method: {self.method}")
        if self.example_count <= 0:
            raise ValueError("example_count must be positive")
        if self.layout not in LAYOUTS:
            raise ValueError(f"unsupported arithmetic layout: {self.layout}")
        if self.frames_per_example <= 0:
            raise ValueError("frames_per_example must be positive")
        if self.tokens_per_frame <= 0:
            raise ValueError("tokens_per_frame must be positive")
        if self.flat_tokens_per_example <= 0:
            raise ValueError("flat_tokens_per_example must be positive")
        if self.total_flat_tokens <= 0:
            raise ValueError("total_flat_tokens must be positive")
        if self.status != "estimated":
            raise ValueError("gpt arithmetic estimate only supports estimated status")
        if self.measured:
            raise ValueError("gpt arithmetic estimate does not claim measured results")


def _normalize_split(split: str | Sequence[str] | None) -> tuple[str, ...]:
    if split is None:
        return ("challenge",)
    if isinstance(split, str):
        value = split.strip()
        if not value:
            raise ValueError("split must be non-empty")
        return (value,)
    normalized = tuple(str(item).strip() for item in split if str(item).strip())
    if not normalized:
        raise ValueError("split must contain at least one non-empty entry")
    return normalized


def load_gpt_arithmetic_profile(profile: str) -> GPTArithmeticProfileConfig:
    try:
        config = PROFILES[profile]
    except KeyError as exc:
        raise ValueError(f"unknown gpt arithmetic profile: {profile}") from exc

    method = str(config.get("method", ""))
    model = str(config.get("model", ""))
    context_tokens = int(config.get("context_tokens", 0))
    return GPTArithmeticProfileConfig(
        profile=profile,
        method=method,
        model=model,
        context_tokens=context_tokens,
        dataset_name=COMMAVQ_DATASET_NAME,
    )


def flatten_tokens_for_gpt_arithmetic(
    tokens,
    *,
    layout: str = "frame_major",
    bos_token: int = FRAME_BOS_TOKEN,
    eot_token: int = SEGMENT_EOT_TOKEN,
):
    import numpy as np

    array = np.asarray(tokens).astype(np.int16)
    if array.ndim < 2:
        raise ValueError("tokens must have at least frame and feature dimensions")
    if layout not in LAYOUTS:
        raise ValueError(f"unsupported arithmetic layout: {layout}")
    frames = array.shape[0]
    flat_per_frame = array.reshape(frames, -1).astype(np.int16)
    if layout == "frame_major":
        bos = np.full((frames, 1), bos_token, dtype=np.int16)
        with_bos = np.concatenate([bos, flat_per_frame], axis=1)
        flattened = with_bos.reshape(-1)
    else:
        positions = flat_per_frame.shape[1]
        streams = []
        for position in range(positions):
            streams.append(np.array([bos_token], dtype=np.int16))
            streams.append(flat_per_frame[:, position])
        flattened = np.concatenate(streams, axis=0) if streams else np.array([], dtype=np.int16)
    return np.concatenate([flattened, np.array([eot_token], dtype=np.int16)], axis=0)


def _train_split(dataset):
    if not isinstance(dataset, Mapping):
        raise ValueError("dataset_loader must return a mapping of splits to datasets")
    if "train" not in dataset:
        raise ValueError("dataset_loader must provide a 'train' split")
    return dataset["train"]


def _example_count(train_split) -> int:
    count = getattr(train_split, "num_rows", None)
    if count is not None:
        return int(count)
    try:
        return len(train_split)
    except TypeError as exc:
        raise ValueError("train split must provide num_rows or len() for arithmetic estimation") from exc


def _first_example(train_split):
    if hasattr(train_split, "__getitem__"):
        return train_split[0]
    iterator = iter(train_split)
    try:
        return next(iterator)
    except StopIteration as exc:
        raise ValueError("train split is empty") from exc


def _encode_example_to_ids(example: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(example, Mapping) or "token.npy" not in example:
        raise ValueError("dataset example must provide token.npy")
    raise RuntimeError("_encode_example_to_ids requires bound layout; use a partial wrapper")


def _encode_example_to_ids_for_layout(example: Mapping[str, object], *, layout: str) -> dict[str, object]:
    if not isinstance(example, Mapping) or "token.npy" not in example:
        raise ValueError("dataset example must provide token.npy")
    ids = flatten_tokens_for_gpt_arithmetic(example["token.npy"], layout=layout).astype("uint16")
    return {
        "ids": ids,
        "len": int(ids.size),
    }


def _mapped_column(rows: object, key: str):
    if isinstance(rows, Mapping):
        return rows[key]
    if hasattr(rows, "__getitem__"):
        return rows[key]
    raise ValueError(f"mapped dataset output is missing column {key!r}")


def estimate_gpt_arithmetic_workload(
    profile: str,
    *,
    split: str | Sequence[str] | None = "challenge",
    work_dir: str | Path | None = None,
    dataset_loader=None,
    num_proc: Optional[int] = None,
    layout: str = "frame_major",
) -> GPTArithmeticEstimate:
    import numpy as np

    config = load_gpt_arithmetic_profile(profile)
    dataset = load_commavq_dataset(
        split=split,
        dataset_loader=dataset_loader,
        dataset_name=config.dataset_name,
        num_proc=num_proc,
    )
    train_split = _train_split(dataset)
    example_count = _example_count(train_split)
    example = _first_example(train_split)
    if not isinstance(example, Mapping) or "token.npy" not in example:
        raise ValueError("dataset example must provide token.npy")
    token_array = np.asarray(example["token.npy"])
    if token_array.ndim < 2:
        raise ValueError("token.npy must have at least frame and feature dimensions")
    frames_per_example = int(token_array.shape[0])
    raw_tokens_per_frame = int(np.prod(token_array.shape[1:]))
    if layout == "frame_major":
        tokens_per_frame = raw_tokens_per_frame + 1
        flat_tokens_per_example = frames_per_example * tokens_per_frame + 1
    elif layout == "position_major":
        tokens_per_frame = raw_tokens_per_frame + 1
        flat_tokens_per_example = raw_tokens_per_frame * (frames_per_example + 1) + 1
    else:
        raise ValueError(f"unsupported arithmetic layout: {layout}")
    total_flat_tokens = example_count * flat_tokens_per_example
    return GPTArithmeticEstimate(
        profile=config.profile,
        method=config.method,
        model=config.model,
        context_tokens=config.context_tokens,
        dataset_name=config.dataset_name,
        layout=layout,
        split=_normalize_split(split),
        work_dir=str(work_dir) if work_dir is not None else None,
        example_count=example_count,
        frames_per_example=frames_per_example,
        tokens_per_frame=tokens_per_frame,
        flat_tokens_per_example=flat_tokens_per_example,
        total_flat_tokens=total_flat_tokens,
    )


def materialize_gpt_arithmetic_stream(
    profile: str,
    *,
    split: str | Sequence[str] | None = "challenge",
    output_path: str | Path,
    dataset_loader=None,
    num_proc: Optional[int] = None,
    layout: str = "frame_major",
) -> dict[str, object]:
    import numpy as np

    estimate = estimate_gpt_arithmetic_workload(
        profile,
        split=split,
        work_dir=None,
        dataset_loader=dataset_loader,
        num_proc=num_proc,
        layout=layout,
    )
    config = load_gpt_arithmetic_profile(profile)
    dataset = load_commavq_dataset(
        split=split,
        dataset_loader=dataset_loader,
        dataset_name=config.dataset_name,
        num_proc=num_proc,
    )
    train_split = _train_split(dataset)
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(train_split, "map"):
        encoder = partial(_encode_example_to_ids_for_layout, layout=layout)
        mapped = train_split.map(
            encoder,
            desc="prepare_gpt_arithmetic",
            num_proc=num_proc,
            load_from_cache_file=False,
        )
        lengths = _mapped_column(mapped, "len")
        total_len = int(sum(int(item) for item in lengths))
        if hasattr(mapped, "shard"):
            arr = np.memmap(target, dtype=np.uint16, mode="w+", shape=(total_len,))
            total_batches = min(max(len(lengths), 1), 1024)
            idx = 0
            for batch_idx in range(total_batches):
                batch = mapped.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format("numpy")
                ids = batch["ids"]
                if len(ids) == 0:
                    continue
                arr_batch = np.concatenate([np.asarray(item, dtype=np.uint16) for item in ids], axis=0)
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()
            token_count = total_len
            return {
                "command": "lossless_prepare",
                "profile": estimate.profile,
                "method": estimate.method,
                "layout": estimate.layout,
                "output_path": str(target),
                "example_count": estimate.example_count,
                "token_count": token_count,
                "dtype": "uint16",
                "split": list(estimate.split),
                "measured": False,
            }
        ids_column = _mapped_column(mapped, "ids")
        pieces = [np.asarray(item, dtype=np.uint16) for item in ids_column]
    else:
        pieces = []
        for example in train_split:
            encoded = _encode_example_to_ids_for_layout(example, layout=layout)
            pieces.append(np.asarray(encoded["ids"], dtype=np.uint16))

    if pieces:
        stream = np.concatenate(pieces, axis=0)
    else:
        stream = np.array([], dtype=np.uint16)
    stream.tofile(target)
    return {
        "command": "lossless_prepare",
        "profile": estimate.profile,
        "method": estimate.method,
        "layout": estimate.layout,
        "output_path": str(target),
        "example_count": estimate.example_count,
        "token_count": int(stream.size),
        "dtype": "uint16",
        "split": list(estimate.split),
        "measured": False,
    }


def estimate_empirical_entropy_bits(tokens) -> float:
    import numpy as np

    arr = np.asarray(tokens)
    if arr.size == 0:
        raise ValueError("tokens must be non-empty")
    _, counts = np.unique(arr, return_counts=True)
    probs = counts / counts.sum()
    return float(-(probs * np.log2(probs)).sum())


def write_symbol_frequency_report(*, token_path: str | Path, output_path: str | Path) -> dict[str, object]:
    import numpy as np

    source = Path(token_path)
    target = Path(output_path)
    if source.stat().st_size % 2 != 0:
        raise ValueError(f"token stream must contain an even number of bytes: {source}")
    tokens = np.fromfile(source, dtype=np.uint16)
    if tokens.size == 0:
        raise ValueError(f"token stream is empty: {source}")
    unique, counts = np.unique(tokens, return_counts=True)
    payload = {
        "command": "lossless_frequency_report",
        "token_path": str(source),
        "token_count": int(tokens.size),
        "unique_symbols": int(unique.size),
        "empirical_bits_per_token": estimate_empirical_entropy_bits(tokens),
        "top_symbols": [
            {"symbol": int(symbol), "count": int(count)}
            for symbol, count in sorted(zip(unique.tolist(), counts.tolist()), key=lambda item: item[1], reverse=True)[:16]
        ],
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def build_gpt_arithmetic_plan(
    profile: str,
    *,
    split: str | Sequence[str] | None = "challenge",
    work_dir: str | Path | None = None,
    layout: str = "frame_major",
) -> GPTArithmeticPlan:
    config = load_gpt_arithmetic_profile(profile)
    return GPTArithmeticPlan(
        profile=config.profile,
        method=config.method,
        model=config.model,
        context_tokens=config.context_tokens,
        dataset_name=config.dataset_name,
        layout=layout,
        split=_normalize_split(split),
        work_dir=str(work_dir) if work_dir is not None else None,
    )
